// Parity + match test for the fused PLE conv (ple-conv.cu: concat tail + ple_conv_kernel, default on HIP;
// LUCE_PLE_FUSE_LEGACY=1 keeps the unfused concat / 4 transposed CONT taps / mul / add / silu nodes).
// Rebuilds the exact qwen4exp prefill subgraph (qwen4exp_graph.cpp build_ple): the transposed conv input is a
// reshape_2d([hc*n_embd, T]) of the 3-D grouped-norm output, which the matcher once rejected (ne[2] != 1 on the
// 3-D root), silently leaving ~85 ms/request of unfused copies at p4096. The fused arm must (a) match on every
// compute (no "[ple-conv] no match" under LUCE_PLE_DEBUG=1) and (b) hash-equal the legacy arm (child process)
// on the silu output and the updated conv state. SKIP (77) without a GPU.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cmath>
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

// n_embd 2560, hc 4, kernel 4, dilation 3 (hist 9), conv weight F16 [4, hc*n_embd]: the Qwen3.8-Flash-Next PLE layer.
uint64_t run(ggml_backend_t gpu, int T) {
    const int64_t n_embd = 2560, hc = 4, C = n_embd * hc, kern = 4, dil = 3, hist = (kern - 1) * dil;
    ggml_init_params ip{ 256 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * gated = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, T);
    ggml_tensor * nw    = ggml_new_tensor_1d(c, GGML_TYPE_F32, C);
    ggml_tensor * state = ggml_new_tensor_1d(c, GGML_TYPE_F32, hist * C);
    // The conv weight must be a weight (its views sit inside the fused subgraph with an external view_src, which
    // ggml_can_fuse_subgraph_ext only allows for constant tensors): own context, buffer usage WEIGHTS.
    ggml_context * cw = ggml_init(ip);
    ggml_tensor * W     = ggml_new_tensor_2d(cw, GGML_TYPE_F16, kern, C);
    ggml_backend_buffer_t bufw = ggml_backend_alloc_ctx_tensors(cw, gpu);
    if (!bufw) { ggml_free(cw); ggml_free(c); return 0; }
    ggml_backend_buffer_set_usage(bufw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (ggml_tensor * t : { gated, nw, state }) ggml_set_input(t);
    ggml_cgraph * gf = ggml_new_graph(c);

    ggml_tensor * g3 = ggml_mul(c, ggml_rms_norm(c, ggml_reshape_3d(c, gated, n_embd, hc, T), 1e-6f), ggml_reshape_2d(c, nw, n_embd, hc));
    ggml_tensor * normalized = ggml_reshape_2d(c, g3, C, T);
    ggml_tensor * norm_t = ggml_transpose(c, ggml_reshape_2d(c, normalized, C, T));
    ggml_tensor * ple_state = ggml_cont(c, ggml_reshape_3d(c, state, hist, C, 1));
    ggml_tensor * padded = ggml_concat(c, ple_state, norm_t, 0);
    ggml_build_forward_expand(gf, ggml_cpy(c,
        ggml_cont(c, ggml_view_3d(c, padded, hist, C, 1, padded->nb[1], padded->nb[2], (size_t) T * padded->nb[0])),
        ggml_reshape_3d(c, state, hist, C, 1)));
    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        const int64_t start = hist - (kern - 1 - k) * dil;
        ggml_tensor * shifted = ggml_cont(c, ggml_transpose(c,
            ggml_view_3d(c, padded, T, C, 1, padded->nb[1], padded->nb[2], ggml_row_size(padded->type, start))));
        if (k == 0) shifted->src[1] = norm_t;
        ggml_tensor * wk = ggml_cont(c, ggml_view_2d(c, W, 1, C, W->nb[1], k * W->nb[0]));
        wk = ggml_cast(c, ggml_reshape_1d(c, wk, C), GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(c, shifted, wk);
        conv_out = conv_out ? ggml_add(c, conv_out, term) : term;
    }
    ggml_tensor * out = ggml_silu(c, conv_out);   // consumed through a RESHAPE in the graph: no CONT after the silu
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    if (!buf) { ggml_backend_buffer_free(bufw); ggml_free(cw); ggml_free(c); return 0; }
    uint32_t r = 11u;
    auto fill = [&](ggml_tensor * t, float scale) {
        std::vector<float> v(ggml_nelements(t));
        for (float & x : v) { r = r * 1664525u + 1013904223u; x = scale * (((r >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(v.size());
            for (size_t i = 0; i < v.size(); ++i) h[i] = ggml_fp32_to_fp16(v[i]);
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
    };
    fill(gated, 2.0f); fill(nw, 1.0f); fill(state, 1.0f); fill(W, 0.5f);
    uint64_t h = 1469598103934665603ull;
    for (int pass = 0; pass < 2; ++pass) {   // pass 1 reads the state pass 0 wrote back
        if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(buf); ggml_backend_buffer_free(bufw); ggml_free(cw); ggml_free(c); return 0; }
        for (ggml_tensor * t : { out, state }) {
            std::vector<float> v(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
            h = fnv(v.data(), v.size() * sizeof(float), h);
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(bufw);
    ggml_free(cw);
    ggml_free(c);
    return h;
}

uint64_t run_all(ggml_backend_t gpu, bool & ok) {
    uint64_t h = 1469598103934665603ull;
    for (int T : { 1028, 300 }) {
        const uint64_t ht = run(gpu, T);
        if (!ht) { std::printf("[ple-conv-fusion] FAIL: compute T=%d\n", T); ok = false; }
        h = fnv(&ht, sizeof ht, h);
    }
    return h;
}

// Child: this binary with --hash, stdout+stderr merged. Returns the hash line; counts "[ple-conv] no match" lines.
bool child(const char * self, const char * env, uint64_t & hash, int & nomatch) {
    const std::string cmd = std::string(env) + " '" + self + "' --hash 2>&1";
    FILE * f = popen(cmd.c_str(), "r");
    if (!f) return false;
    char line[256]; bool got = false; nomatch = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strstr(line, "[ple-conv] no match")) ++nomatch;
        else if (!std::strncmp(line, "hash ", 5)) { hash = std::strtoull(line + 5, nullptr, 16); got = true; }
    }
    pclose(f);
    return got;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "--hash")) {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (!gpu) return 77;
        ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);   // the fusions are gated on the qwen4exp profile
        bool ok = true;
        const uint64_t h = run_all(gpu, ok);
        ggml_backend_free(gpu);
        std::printf("hash %llx\n", (unsigned long long) h);
        return ok ? 0 : 1;
    }
    {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (!gpu) { std::printf("[ple-conv-fusion] SKIP: no GPU\n"); return 77; }
        ggml_backend_free(gpu);
    }
    uint64_t hf = 0, hl = 0; int nf = 0, nl = 0;
    const bool gf = child(argv[0], "LUCE_PLE_DEBUG=1 LUCE_PLE_FUSE_LEGACY=0", hf, nf);
    const bool gl = child(argv[0], "LUCE_PLE_FUSE_LEGACY=1", hl, nl);
    const bool ok = gf && gl && hf == hl && nf == 0;
    std::printf("[ple-conv-fusion] fused %llx (no-match lines %d) legacy %llx %s\n",
                (unsigned long long) hf, nf, (unsigned long long) hl, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
