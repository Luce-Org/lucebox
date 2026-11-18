// Smoke test: load the DeepSeek-V4-Flash DSpark drafter GGUF and check its bindings.
//
//   test_ds4_dspark_load <path-to-drafter.gguf>
//
// Verifies the "deepseek4-dflash-draft" GGUF <-> DSparkDrafter loader contract:
// all block tensors + DSpark heads bind, dspark_enabled, capture ids present.

#include "deepseek4/deepseek4_dspark.h"

#include "ggml.h"
#include "ggml-backend.h"
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#define DS4_HAVE_GPU 1
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

static uint64_t fnv1a64(const std::vector<float> & values) {
    uint64_t hash = 1469598103934665603ull;
    const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    }
    return hash;
}

int main(int argc, char ** argv) {
    using namespace luce::common;
    const char * path = argc > 1 ? argv[1]
        : "/home/lucebox2/models/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4.gguf";

    ggml_backend_t backend = nullptr;
#ifdef DS4_HAVE_GPU
    backend = ggml_backend_cuda_init(0);
#endif
    if (!backend) { std::fprintf(stderr, "FAIL: no GPU backend\n"); return 2; }

    DSparkDrafter d;
    if (!load_deepseek4_dspark_drafter(path, backend, d)) {
        std::fprintf(stderr, "FAIL: load: %s\n", deepseek4_dspark_last_error());
        return 1;
    }

    // Contract checks.
    int rc = 0;
    auto need = [&](bool cond, const char * what) {
        if (!cond) { std::fprintf(stderr, "FAIL: missing %s\n", what); rc = 1; }
    };
    need(d.core.n_layer == 3, "n_layer==3");
    need(d.main_proj && d.main_norm, "main_proj/main_norm (dflash.fc/hidden_norm)");
    need(d.markov_w1 && d.markov_w2, "markov heads");
    need(d.dspark_enabled, "dspark_enabled");
    need((int)d.capture_layer_ids.size() == d.n_target_layers, "capture_layer_ids count");
    need(d.core.output == nullptr && d.core.tok_embd == nullptr, "tied embed/lm_head (no output tensors)");
    for (int il = 0; il < d.core.n_layer; il++) {
        const auto & L = d.core.layers[il];
        need(L.attn_q_a && L.attn_q_b && L.attn_kv && L.attn_output_a && L.attn_output_b,
             "MLA weights");
        need(L.ffn_gate_exps && L.ffn_up_exps && L.ffn_down_exps, "routed experts");
        need(L.ffn_gate_shexp && L.ffn_up_shexp && L.ffn_down_shexp, "shared expert");
        need(L.hc_attn_fn && L.hc_ffn_fn, "HC weights");
        need(d.core.compress_ratios[il] == 0, "compress_ratio==0");
        need(L.attn_compressor_kv == nullptr, "no compressor tensors");
    }
    // ── Exercise the drafter forward with dummy inputs ──────────────
    // Validates the whole graph runs on the GPU (HC ops, MoE, rocmfp
    // matmuls, bidirectional attention, tail) and returns finite output.
    if (rc == 0) {
        const int n_embd = d.core.n_embd;
        const int block  = d.block_size;
        const int fc_in  = d.n_target_layers * n_embd;
        const char * cle = std::getenv("DS4_CTX_LEN");
        const int ctx_len = cle ? atoi(cle) : 8;
        const char * variant_env = std::getenv("DS4_INPUT_VARIANT");
        const int input_variant = variant_env ? atoi(variant_env) : 0;
        const float phase = 0.73f * (float) input_variant;
        std::vector<float> noise((size_t) n_embd * block);
        std::vector<float> feats((size_t) fc_in * ctx_len);
        for (size_t i = 0; i < noise.size(); i++) noise[i] = 0.06f * std::sin(0.31f * (float) i + 1.3f + phase);
        for (size_t i = 0; i < feats.size(); i++) feats[i] = 0.05f * std::cos(0.17f * (float) i + 0.4f - phase);
        std::vector<float> hidden, confidence_hidden;
        const int committed = 64;  // arbitrary >= ctx_len
        std::fprintf(stderr, "\n── drafter forward (ctx_len=%d block=%d) ──\n", ctx_len, block);
        if (!deepseek4_dspark_draft_forward(backend, d, noise.data(), feats.data(),
                                            ctx_len, committed, hidden,
                                            &confidence_hidden)) {
            std::fprintf(stderr, "FAIL: drafter forward returned false\n");
            rc = 1;
        } else {
            bool finite = true; double sumsq = 0.0;
            for (float v : hidden) { if (!std::isfinite(v)) finite = false; sumsq += (double) v * v; }
            std::fprintf(stderr, "forward out: %zu floats, finite=%d, rms=%.4f, hash=%016llx, first=[%.4f %.4f %.4f]\n",
                         hidden.size(), (int) finite,
                         std::sqrt(sumsq / (double) hidden.size()),
                         (unsigned long long) fnv1a64(hidden),
                         hidden.size() > 0 ? hidden[0] : 0.0f,
                         hidden.size() > 1 ? hidden[1] : 0.0f,
                         hidden.size() > 2 ? hidden[2] : 0.0f);
            need(finite, "finite forward output");
            need(hidden.size() == (size_t) n_embd * block, "forward output size");
            bool confidence_finite = true;
            for (float v : confidence_hidden) {
                if (!std::isfinite(v)) confidence_finite = false;
            }
            need(confidence_finite, "finite pre-norm confidence hidden");
            need(confidence_hidden.size() == (size_t) n_embd * block,
                 "pre-norm confidence hidden size");

            // The speculative draft-ahead path reads both graph outputs after
            // an asynchronous launch. Keep its hidden/confidence pairing
            // covered by the same end-to-end model smoke as the sync path.
            std::vector<float> async_hidden;
            std::vector<float> async_confidence_hidden;
            const bool async_ok = deepseek4_dspark_draft_forward_async(
                backend, d, noise.data(), feats.data(), ctx_len, committed);
            if (async_ok) {
                deepseek4_dspark_draft_wait(backend);
            }
            const bool async_read_ok = async_ok &&
                deepseek4_dspark_draft_read_async_output(
                    backend, async_hidden, &async_confidence_hidden);
            need(async_read_ok, "async draft output read");
            if (async_read_ok) {
                need(async_hidden == hidden, "async normalized hidden identity");
                need(async_confidence_hidden == confidence_hidden,
                     "async confidence hidden identity");
            }

            const char * bench_env = std::getenv("DS4_BENCH_ITERS");
            const int bench_iters = bench_env ? std::max(0, std::atoi(bench_env)) : 0;
            if (bench_iters > 0) {
                constexpr int warmups = 5;
                std::vector<double> timings;
                timings.reserve((size_t) bench_iters);
                for (int i = 0; i < warmups + bench_iters; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    const bool run_ok = deepseek4_dspark_draft_forward(
                        backend, d, noise.data(), feats.data(), ctx_len,
                        committed, hidden, &confidence_hidden);
                    const auto end = std::chrono::steady_clock::now();
                    need(run_ok, "successful benchmark draft forward");
                    if (!run_ok) break;
                    if (i >= warmups) {
                        timings.push_back(
                            std::chrono::duration<double, std::milli>(
                                end - start).count());
                    }
                }
                if (!timings.empty()) {
                    std::sort(timings.begin(), timings.end());
                    const double median_ms = timings[timings.size() / 2];
                    std::fprintf(stderr,
                                 "[bench] ctx_len=%d block=%d median=%.3f ms "
                                 "(%zu samples)\n",
                                 ctx_len, block, median_ms, timings.size());
                }
            }
        }
    }

    std::fprintf(stderr, rc == 0 ? "\nPASS: DSpark drafter load + forward OK\n" : "\nFAIL\n");

    free_deepseek4_dspark_drafter(d);
    ggml_backend_free(backend);
    return rc;
}
