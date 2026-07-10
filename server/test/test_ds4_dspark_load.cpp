// Smoke test: load the DeepSeek-V4-Flash DSpark drafter GGUF and dump bindings.
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
#include <string>
#include <vector>

namespace dflash::common { void deepseek4_dspark_dump(const DSparkDrafter &); }

int main(int argc, char ** argv) {
    using namespace dflash::common;
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
    deepseek4_dspark_dump(d);

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
    // ── Weight sanity: dequantize main_proj directly (no matmul) ────
    if (rc == 0 && d.main_proj) {
        ggml_init_params ip{}; ip.mem_size = 64u*1024*1024; ip.no_alloc = true;
        ggml_context * c = ggml_init(ip);
        ggml_cgraph * g = ggml_new_graph(c);
        ggml_tensor * f32 = ggml_cast(c, d.main_proj, GGML_TYPE_F32);
        ggml_set_output(f32); ggml_build_forward_expand(g, f32);
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (al && ggml_gallocr_alloc_graph(al, g) && ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS) {
            size_t ne = ggml_nelements(f32);
            std::vector<float> buf(ne);
            ggml_backend_tensor_get(f32, buf.data(), 0, sizeof(float)*ne);
            size_t nnan=0; double ss=0; float mn=1e30f, mx=-1e30f;
            for (float v : buf) { if(!std::isfinite(v)) nnan++; else { ss+=(double)v*v; if(v<mn)mn=v; if(v>mx)mx=v; } }
            std::fprintf(stderr, "[weight-check] main_proj dequant: ne=%zu nnan=%zu rms=%.5f min=%.4f max=%.4f\n",
                         ne, nnan, std::sqrt(ss/(double)ne), mn, mx);
        } else { std::fprintf(stderr, "[weight-check] main_proj cast failed (type %s not castable?)\n", ggml_type_name(d.main_proj->type)); }
        if (al) ggml_gallocr_free(al); ggml_free(c);
    }

    // ── Isolate: mul_mat(weight, ones) for main_proj and attn_q_a ───
    if (rc == 0) {
        auto mm_check = [&](const char * nm, ggml_tensor * W) {
            if (!W) return;
            ggml_init_params ip{}; ip.mem_size = 64u*1024*1024; ip.no_alloc = true;
            ggml_context * c = ggml_init(ip);
            ggml_cgraph * g = ggml_new_graph(c);
            ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, W->ne[0], 1);
            ggml_set_input(x);
            ggml_tensor * y = ggml_mul_mat(c, W, x);
            ggml_set_output(y); ggml_build_forward_expand(g, y);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (al && ggml_gallocr_alloc_graph(al, g)) {
                std::vector<float> ones((size_t)W->ne[0], 1.0f);
                ggml_backend_tensor_set(x, ones.data(), 0, sizeof(float)*ones.size());
                if (ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS) {
                    size_t ne = ggml_nelements(y); std::vector<float> buf(ne);
                    ggml_backend_tensor_get(y, buf.data(), 0, sizeof(float)*ne);
                    size_t nn=0; for(float v:buf) if(!std::isfinite(v)) nn++;
                    std::fprintf(stderr, "[mm-check] %-12s mul_mat(W[%lld,%lld], ones): nnan=%zu/%zu first=%.3f\n",
                                 nm,(long long)W->ne[0],(long long)W->ne[1],nn,ne,buf.empty()?0:buf[0]);
                }
            }
            if (al) ggml_gallocr_free(al); ggml_free(c);
        };
        mm_check("main_proj", d.main_proj);
        mm_check("blk0.attn_q_a", d.core.layers[0].attn_q_a);
        mm_check("blk0.ffn_gate_shexp", d.core.layers[0].ffn_gate_shexp);
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
        std::vector<float> noise((size_t) n_embd * block);
        std::vector<float> feats((size_t) fc_in * ctx_len);
        for (size_t i = 0; i < noise.size(); i++) noise[i] = 0.06f * std::sin(0.31f * (float) i + 1.3f);
        for (size_t i = 0; i < feats.size(); i++) feats[i] = 0.05f * std::cos(0.17f * (float) i + 0.4f);
        std::vector<float> hidden;
        const int committed = 64;  // arbitrary >= ctx_len
        std::fprintf(stderr, "\n── drafter forward (ctx_len=%d block=%d) ──\n", ctx_len, block);
        if (!deepseek4_dspark_draft_forward(backend, d, noise.data(), feats.data(),
                                            ctx_len, committed, hidden)) {
            std::fprintf(stderr, "FAIL: drafter forward returned false\n");
            rc = 1;
        } else {
            bool finite = true; double sumsq = 0.0;
            for (float v : hidden) { if (!std::isfinite(v)) finite = false; sumsq += (double) v * v; }
            std::fprintf(stderr, "forward out: %zu floats, finite=%d, rms=%.4f, first=[%.4f %.4f %.4f]\n",
                         hidden.size(), (int) finite,
                         std::sqrt(sumsq / (double) hidden.size()),
                         hidden.size() > 0 ? hidden[0] : 0.0f,
                         hidden.size() > 1 ? hidden[1] : 0.0f,
                         hidden.size() > 2 ? hidden[2] : 0.0f);
            need(finite, "finite forward output");
            need(hidden.size() == (size_t) n_embd * block, "forward output size");
        }
    }

    std::fprintf(stderr, rc == 0 ? "\nPASS: DSpark drafter load + forward OK\n" : "\nFAIL\n");

    free_deepseek4_dspark_drafter(d);
    ggml_backend_free(backend);
    return rc;
}
