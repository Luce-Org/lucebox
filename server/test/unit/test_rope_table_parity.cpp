// Parity test for the per-forward M-RoPE cos/sin table (rope.cu rope_multi_table, default on the HIP per-token
// path; LUCE_ATTN_PREFILL_LEGACY=1 restores the inline FP64 angle code). The table must be bit-identical to the inline
// code and must not go stale: two graph computes reuse the same positions tensor with different contents, and
// one graph holds two rope calls on two positions tensors (the Q/K/indexer table and the pooled-positions table).
// The env switch is read once per process, so the legacy arm runs in a child process (this binary with
// --hash under LUCE_ATTN_PREFILL_LEGACY=1) and the two arms compare a hash of every output. SKIP (77) without a GPU.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// Qwen4exp prefill shapes: Q [256,24,T] + pooled [128,1,T/4] in one graph, two computes with different positions.
uint64_t run(ggml_backend_t gpu, int T) {
    const int n_dims = 64, Tp = T / 4;
    int sections[4] = { 11, 11, 10, 0 };
    ggml_init_params ip{ 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 24, T);
    ggml_tensor * k   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, T);
    ggml_tensor * pl  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 1, Tp);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
    ggml_tensor * ppos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * Tp);
    for (ggml_tensor * t : { q, k, pl, pos, ppos }) ggml_set_input(t);
    ggml_tensor * outs[3] = {
        ggml_rope_multi(ctx, q, pos, nullptr, n_dims, sections, GGML_ROPE_TYPE_MROPE, 0, 1e7f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
        ggml_rope_multi(ctx, k, pos, nullptr, n_dims, sections, GGML_ROPE_TYPE_MROPE, 0, 1e7f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
        ggml_rope_multi(ctx, pl, ppos, nullptr, n_dims, sections, GGML_ROPE_TYPE_MROPE, 0, 1e7f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
    };
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (ggml_tensor * o : outs) { ggml_set_output(o); ggml_build_forward_expand(graph, o); }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buf) { ggml_free(ctx); return 0; }
    uint32_t r = 7u;
    for (ggml_tensor * t : { q, k, pl }) {
        std::vector<float> v(ggml_nelements(t));
        for (float & x : v) { r = r * 1664525u + 1013904223u; x = ((r >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
    uint64_t h = 1469598103934665603ull;
    for (int pass = 0; pass < 2; ++pass) {   // pass 1: same positions tensor, shifted contents (cache must refresh)
        std::vector<int32_t> p(4 * T), pp(4 * Tp);
        for (int s = 0; s < 4; ++s) {
            for (int i = 0; i < T; ++i) p[s * T + i] = s < 3 ? i + 1000 * pass : 0;
            for (int i = 0; i < Tp; ++i) pp[s * Tp + i] = s < 3 ? 4 * i + 3 + 1000 * pass : 0;
        }
        ggml_backend_tensor_set(pos, p.data(), 0, ggml_nbytes(pos));
        ggml_backend_tensor_set(ppos, pp.data(), 0, ggml_nbytes(ppos));
        if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(buf); ggml_free(ctx); return 0; }
        for (ggml_tensor * o : outs) {
            std::vector<float> v(ggml_nelements(o));
            ggml_backend_tensor_get(o, v.data(), 0, ggml_nbytes(o));
            h = fnv(v.data(), v.size() * sizeof(float), h);
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return h;
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::printf("[rope-table-parity] SKIP: no GPU\n"); return 77; }
    uint64_t h = 1469598103934665603ull;
    for (int T : { 4136, 1028, 68 }) {
        const uint64_t ht = run(gpu, T);
        if (!ht) { std::printf("[rope-table-parity] FAIL: compute T=%d\n", T); return 1; }
        h = fnv(&ht, sizeof ht, h);
    }
    ggml_backend_free(gpu);
    if (argc > 1 && !std::strcmp(argv[1], "--hash")) { std::printf("%llx\n", (unsigned long long) h); return 0; }
    if (getenv("LUCE_ATTN_PREFILL_LEGACY") && atoi(getenv("LUCE_ATTN_PREFILL_LEGACY"))) { std::printf("[rope-table-parity] FAIL: run without LUCE_ATTN_PREFILL_LEGACY=1\n"); return 1; }
    const std::string cmd = std::string("LUCE_ATTN_PREFILL_LEGACY=1 '") + argv[0] + "' --hash";
    FILE * f = popen(cmd.c_str(), "r");
    char line[64] = {};
    const bool got = f && std::fgets(line, sizeof line, f);
    if (f) pclose(f);
    const uint64_t hl = got ? std::strtoull(line, nullptr, 16) : 0;
    const bool ok = got && hl == h;
    std::printf("[rope-table-parity] table %llx legacy %llx %s\n", (unsigned long long) h, (unsigned long long) hl, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
