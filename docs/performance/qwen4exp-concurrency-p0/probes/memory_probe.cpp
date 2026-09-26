// Temporary P0 allocation probe. This is evidence tooling, not a server target.
#include "qwen4exp_cache.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_internal.h"
#include "ggml-cuda.h"

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace luce::common;

static size_t bytes(const std::vector<ggml_tensor *> & v) {
    size_t n = 0;
    for (auto * t : v) if (t) n += ggml_nbytes(t);
    return n;
}

static void report_rss(const char * phase) {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmHWM:", 0) == 0) {
            std::printf("[mem] %s %s\n", phase, line.c_str());
        }
    }
    struct rusage u{};
    getrusage(RUSAGE_SELF, &u);
    std::printf("[mem] %s ru_maxrss_kib=%ld\n", phase, u.ru_maxrss);
    for (const char * path : {"/sys/class/drm/card2/device/mem_info_gtt_used",
                              "/proc/meminfo"}) {
        std::ifstream stat(path);
        if (std::string(path) == "/sys/class/drm/card2/device/mem_info_gtt_used") {
            std::string value;
            if (stat >> value) std::printf("[mem] %s gtt_used_bytes=%s\n", phase, value.c_str());
        } else {
            std::string line;
            while (std::getline(stat, line)) {
                if (line.rfind("MemAvailable:", 0) == 0) {
                    std::printf("[mem] %s %s\n", phase, line.c_str());
                    break;
                }
            }
        }
    }
    std::fflush(stdout);
}

static uint64_t cache_edge_hash(ggml_backend_t backend, const Qwen4ExpCache & c) {
    uint64_t h = 1469598103934665603ull;
    auto add = [&](ggml_tensor * t) {
        if (!t) return;
        const size_t n = ggml_nbytes(t), k = std::min<size_t>(n, 64);
        uint8_t b[64];
        ggml_backend_tensor_get(t, b, 0, k);
        for (size_t i=0;i<k;++i) h=(h^b[i])*1099511628211ull;
        if (n > k) {
            ggml_backend_tensor_get(t, b, n-k, k);
            for (size_t i=0;i<k;++i) h=(h^b[i])*1099511628211ull;
        }
    };
    for (auto * t:c.attn_k) add(t);
    for (auto * t:c.attn_v) add(t);
    for (auto * t:c.indexer_k) add(t);
    for (auto * t:c.ssm_state) add(t);
    for (auto * t:c.conv_state) add(t);
    for (auto * t:c.ple_conv_state) add(t);
    (void)backend;
    return h;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    constexpr int ctx = 32768;
    constexpr int slots = 4;
    constexpr int prefill_tokens = 16366;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 3;
    Qwen4ExpWeights w;
    auto t0 = std::chrono::steady_clock::now();
    if (!load_qwen4exp_gguf(argv[1], backend, w)) return 4;
    std::printf("[mem] weights_loaded seconds=%.3f layers=%d full=%d linear=%d vocab=%d\n",
        std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(),
        w.n_layer, w.n_layer / w.full_attention_interval, w.n_layer - w.n_layer / w.full_attention_interval, w.n_vocab);
    std::printf("[shape] n_embd=%d kvheads=%d head_k=%d head_v=%d ssm_state=%d value_heads=%d inner=%d conv=%d n_group=%d indexer_dim=%d ple_layers=%zu ple_ngram=%d ple_kernel=%d hc=%d\n",
        w.n_embd, w.n_head_kv, w.n_embd_head_k, w.n_embd_head_v, w.ssm_d_state,
        w.linear_value_heads, w.ssm_d_inner, w.ssm_d_conv, w.ssm_n_group,
        w.indexer_head_size, w.ple_layer_ids.size(), w.ple_ngram_size, w.ple_conv_kernel, w.n_hc);
    if (!w.layers.empty()) {
        auto type = [](const char * n, ggml_tensor * t) {
            if (t) std::printf("[qtype] %s %s [%lld,%lld,%lld]\n", n, ggml_type_name(t->type),
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
        };
        type("dense.hc_attn_down", w.layers[0].hc_attn_down);
        type("dense.hc_ffn_down", w.layers[0].hc_ffn_down);
        type("dense.attn_q", w.layers[3].attn_qkv);
        type("moe.gate_exps", w.layers[0].ffn_gate_exps);
        type("hc.gamma", w.layers[0].hc_attn_norm);
    }
    report_rss("weights");

    std::vector<Qwen4ExpCache> caches(slots);
    for (int i = 0; i < slots; ++i) {
        auto t = std::chrono::steady_clock::now();
        if (!create_qwen4exp_cache(backend, w, ctx, GGML_TYPE_F16, caches[i])) {
            std::fprintf(stderr, "[mem] cache_create_failed slot=%d\n", i);
            return 5;
        }
        const auto & c = caches[i];
        const size_t kv = bytes(c.attn_k) + bytes(c.attn_v);
        const size_t ix = bytes(c.indexer_k);
        const size_t ssm = bytes(c.ssm_state);
        const size_t conv = bytes(c.conv_state);
        const size_t ple = bytes(c.ple_conv_state);
        std::printf("[cache] slot=%d ctx=%d seconds=%.3f kv=%zu indexer=%zu ssm=%zu conv=%zu ple=%zu total=%zu\n",
            i, c.max_ctx, std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count(),
            kv, ix, ssm, conv, ple, kv+ix+ssm+conv+ple);
        report_rss("cache");
    }

    const uint64_t state_before = cache_edge_hash(backend, caches[1]);
    const auto * workspace_before = caches[1].decode_workspace.ctx;
    const int old_pos = caches[1].cur_pos, old_indexer = caches[1].indexer_blocks;
    const auto old_ple = caches[1].ple_prev;
    const int32_t bad_token = 7;
    std::vector<float> overflow_logits;
    const auto overflow = qwen4exp_forward(backend,w,caches[1],&bad_token,1,ctx,overflow_logits);
    const uint64_t state_after = cache_edge_hash(backend, caches[1]);
    const bool overflow_isolated = !overflow.ok && state_before==state_after &&
        caches[1].cur_pos==old_pos && caches[1].indexer_blocks==old_indexer &&
        caches[1].ple_prev==old_ple && caches[1].decode_workspace.ctx==workspace_before;
    std::printf("[overflow] request_ok=%d pos0=%d n=1 max_ctx=%d state_hash_before=%016llx state_hash_after=%016llx isolated=%d\n",
        overflow.ok,ctx,ctx,(unsigned long long)state_before,(unsigned long long)state_after,overflow_isolated);
    std::vector<float> after_overflow_logits;
    const auto valid_after_overflow = qwen4exp_forward(backend,w,caches[1],&bad_token,1,0,after_overflow_logits);
    std::printf("[overflow] subsequent_valid_ok=%d logits=%zu\n",valid_after_overflow.ok,after_overflow_logits.size());
    if (!overflow_isolated || !valid_after_overflow.ok) return 7;
    report_rss("decode_workspace_ready");

    std::vector<int32_t> tokens(prefill_tokens);
    for (int i = 0; i < prefill_tokens; ++i) tokens[i] = (i * 7919 + 13) % w.n_vocab;
    std::vector<float> logits;
    auto p0 = std::chrono::steady_clock::now();
    auto result = qwen4exp_forward(backend, w, caches[0], tokens.data(), prefill_tokens, 0, logits);
    auto p1 = std::chrono::steady_clock::now();
    std::printf("[prefill] ok=%d tokens=%d seconds=%.3f logits=%zu cache_pos=%d\n", result.ok,
        prefill_tokens, std::chrono::duration<double>(p1-p0).count(), logits.size(), caches[0].cur_pos);
    report_rss("prefill_done");
    std::this_thread::sleep_for(std::chrono::seconds(8));

    for (auto & c : caches) free_qwen4exp_cache(c);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);
    return result.ok ? 0 : 6;
}
