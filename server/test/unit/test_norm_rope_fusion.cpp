// Parity + match test for the fused RMS_NORM -> MUL -> ROPE [-> PERMUTE -> CONT] kernel (rope.cu
// rope_multi_heads_norm, default on the HIP per-token path; LUCE_ATTN_FUSE_LEGACY=1 keeps the separate kernels).
// Rebuilds the qwen4exp attention chains: Q from the strided [q | gate] projection view [256,24,T] -> norm -> gamma
// -> M-RoPE (n_dims 64) -> permute -> cont (the 5-op chain), K [256,2,T] and the indexer q [128,4,T] without the
// permute (the 3-op chain). The fused arm must match on every chain (LUCE_NORM_ROPE_DEBUG=1 "[norm-rope] fused"
// lines, 3 per compute) and hash-equal the legacy arm (child process). Without the qwen4exp profile (a Qwen3.5-style
// q/k-norm M-RoPE graph) nothing may fuse: the switch must change neither the output nor the kernels. SKIP (77)
// without a GPU.
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

uint64_t run(ggml_backend_t gpu, int T) {
    const int D = 256, Hq = 24, Hk = 2, Di = 128, Hi = 4, n_dims = 64;
    int sections[4] = { 11, 11, 10, 0 };
    ggml_init_params ip{ 64 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * qfull = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * D * Hq, T);
    ggml_tensor * kraw  = ggml_new_tensor_2d(c, GGML_TYPE_F32, D * Hk, T);
    ggml_tensor * iraw  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Di * Hi, T);
    ggml_tensor * gq    = ggml_new_tensor_1d(c, GGML_TYPE_F32, D);
    ggml_tensor * gk    = ggml_new_tensor_1d(c, GGML_TYPE_F32, D);
    ggml_tensor * gi    = ggml_new_tensor_2d(c, GGML_TYPE_F32, Di, Hi);   // per-head gamma
    ggml_tensor * pos   = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * T);
    for (ggml_tensor * t : { qfull, kraw, iraw, gq, gk, gi, pos }) ggml_set_input(t);
    const float eps = 1e-6f;
    const size_t qe = sizeof(float);
    ggml_tensor * Q = ggml_rms_norm(c, ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, 0), eps);
    Q = ggml_mul(c, Q, gq);
    Q = ggml_rope_multi(c, Q, pos, nullptr, n_dims, sections, GGML_ROPE_TYPE_MROPE, 0, 1e7f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Q = ggml_cont(c, ggml_permute(c, Q, 0, 2, 1, 3));   // [D, T, Hq]
    ggml_tensor * K = ggml_rms_norm(c, ggml_reshape_3d(c, kraw, D, Hk, T), eps);
    K = ggml_mul(c, K, gk);
    K = ggml_rope_multi(c, K, pos, nullptr, n_dims, sections, GGML_ROPE_TYPE_MROPE, 0, 1e7f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * I = ggml_rms_norm(c, ggml_reshape_3d(c, iraw, Di, Hi, T), eps);
    I = ggml_mul(c, I, gi);
    I = ggml_rope_multi(c, I, pos, nullptr, n_dims, sections, GGML_ROPE_TYPE_MROPE, 0, 1e7f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_cgraph * gf = ggml_new_graph(c);
    for (ggml_tensor * o : { Q, K, I }) { ggml_set_output(o); ggml_build_forward_expand(gf, o); }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    if (!buf) { ggml_free(c); return 0; }
    uint32_t r = 7u;
    for (ggml_tensor * t : { qfull, kraw, iraw, gq, gk, gi }) {
        std::vector<float> v(ggml_nelements(t));
        for (float & x : v) { r = r * 1664525u + 1013904223u; x = 2.0f * (((r >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
    std::vector<int32_t> p(4 * T);
    for (int s = 0; s < 4; ++s) for (int i = 0; i < T; ++i) p[s * T + i] = s < 3 ? i + 4096 : 0;
    ggml_backend_tensor_set(pos, p.data(), 0, ggml_nbytes(pos));
    uint64_t h = 1469598103934665603ull;
    if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(buf); ggml_free(c); return 0; }
    for (ggml_tensor * t : { Q, K, I }) {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        h = fnv(v.data(), v.size() * sizeof(float), h);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return h;
}

uint64_t run_all(ggml_backend_t gpu, bool & ok) {
    uint64_t h = 1469598103934665603ull;
    for (int T : { 1028, 300, 64 }) {
        const uint64_t ht = run(gpu, T);
        if (!ht) { std::printf("[norm-rope-fusion] FAIL: compute T=%d\n", T); ok = false; }
        h = fnv(&ht, sizeof ht, h);
    }
    return h;
}

// Child: this binary with `mode` (--hash, or --hash-off without the qwen4exp profile), stdout+stderr merged.
// Returns the hash line; counts "[norm-rope] fused" lines.
bool child(const char * self, const char * env, const char * mode, uint64_t & hash, int & fused) {
    const std::string cmd = std::string(env) + " '" + self + "' " + mode + " 2>&1";
    FILE * f = popen(cmd.c_str(), "r");
    if (!f) return false;
    char line[256]; bool got = false; fused = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strstr(line, "[norm-rope] fused")) ++fused;
        else if (!std::strncmp(line, "hash ", 5)) { hash = std::strtoull(line + 5, nullptr, 16); got = true; }
    }
    pclose(f);
    return got;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && (!std::strcmp(argv[1], "--hash") || !std::strcmp(argv[1], "--hash-off"))) {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (!gpu) return 77;
        if (!std::strcmp(argv[1], "--hash")) {
            ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);   // the fusions are gated on the qwen4exp profile
        }
        bool ok = true;
        const uint64_t h = run_all(gpu, ok);
        ggml_backend_free(gpu);
        std::printf("hash %llx\n", (unsigned long long) h);
        return ok ? 0 : 1;
    }
    {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (!gpu) { std::printf("[norm-rope-fusion] SKIP: no GPU\n"); return 77; }
        ggml_backend_free(gpu);
    }
    uint64_t hf = 0, hl = 0; int nf = 0, nl = 0;
    uint64_t of = 0, ol = 0; int nof = 0, nol = 0;
    const bool gf = child(argv[0], "LUCE_NORM_ROPE_DEBUG=1 LUCE_ATTN_FUSE_LEGACY=0", "--hash", hf, nf);
    const bool gl = child(argv[0], "LUCE_ATTN_FUSE_LEGACY=1", "--hash", hl, nl);
    const bool go = child(argv[0], "LUCE_NORM_ROPE_DEBUG=1 LUCE_ATTN_FUSE_LEGACY=0", "--hash-off", of, nof);
    const bool gol = child(argv[0], "LUCE_NORM_ROPE_DEBUG=1 LUCE_ATTN_FUSE_LEGACY=1", "--hash-off", ol, nol);
    // qwen4exp: 3 chains x 3 sizes fused, none on the legacy arm. Profile off: none fused, same output either way.
    const bool ok = gf && gl && hf == hl && nf == 9 && nl == 0 && go && gol && of == ol && nof == 0 && nol == 0;
    std::printf("[norm-rope-fusion] fused %llx (fused lines %d, want 9) legacy %llx (fused lines %d); "
                "profile off %s %llx (fused lines %d, want 0) legacy %llx %s\n",
                (unsigned long long) hf, nf, (unsigned long long) hl, nl, go ? "ran" : "CRASHED",
                (unsigned long long) of, nof, (unsigned long long) ol, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
