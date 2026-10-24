// Node-order + parity test for the PLE grouped norm (qwen4exp_graph.cpp ple_grouped_norm): RMS_NORM [n_embd, hc, T]
// -> MUL by the gamma weight viewed [n_embd, hc]. The backend fuses RMS_NORM + MUL only when the two nodes are
// adjacent, and a gamma RESHAPE visited as the MUL's second operand lands between them; expanding the view first (the
// default, LUCE_PLE_FUSE_LEGACY=1 keeps the old order) makes them adjacent. Checks the adjacency on the expanded graph
// and that all three forms hash-equal (fused kernel forms (scale * x) * gamma like the separate ops). SKIP (77)
// without a GPU.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// Graph forms: EXPANDED = gamma view expanded first (the PLE grouped norms, default); VIEW_LEGACY = the view visited
// as the MUL operand (LUCE_PLE_FUSE_LEGACY=1); ROW_LEGACY = the old decode PLE form, MUL on a 2-D RESHAPE of the norm
// output by the flat gamma (the RESHAPE sits between the two).
enum Form { EXPANDED, VIEW_LEGACY, ROW_LEGACY };

// Returns the output hash; `adjacent` reports whether the MUL node directly follows the RMS_NORM node.
uint64_t run(ggml_backend_t gpu, int T, Form form, bool & adjacent) {
    const int n_embd = 2560, hc = 4;
    ggml_init_params ip{ 16 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_embd, hc, T);
    ggml_tensor * w = ggml_new_tensor_1d(c, GGML_TYPE_F32, n_embd * hc);
    ggml_set_input(x); ggml_set_input(w);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_tensor * gamma = ggml_reshape_2d(c, w, n_embd, hc);
    if (form == EXPANDED) ggml_build_forward_expand(gf, gamma);
    ggml_tensor * out = form == ROW_LEGACY
        ? ggml_reshape_3d(c, ggml_mul(c, ggml_reshape_2d(c, ggml_rms_norm(c, x, 1e-6f), n_embd * hc, T), w), n_embd, hc, T)
        : ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), gamma);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    adjacent = false;
    for (int i = 0; i + 1 < ggml_graph_n_nodes(gf); ++i) {
        if (ggml_graph_node(gf, i)->op == GGML_OP_RMS_NORM) adjacent = ggml_graph_node(gf, i + 1)->op == GGML_OP_MUL;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    if (!buf) { ggml_free(c); return 0; }
    uint32_t r = 7u;
    for (ggml_tensor * t : { x, w }) {
        std::vector<float> v(ggml_nelements(t));
        for (float & f : v) { r = r * 1664525u + 1013904223u; f = 2.0f * (((r >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
    uint64_t h = 0;
    if (ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS) {
        std::vector<float> v(ggml_nelements(out));
        ggml_backend_tensor_get(out, v.data(), 0, ggml_nbytes(out));
        h = fnv(v.data(), v.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return h;
}

}  // namespace

int main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::printf("[ple-norm-mul] SKIP: no GPU\n"); return 77; }
    ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);   // the fusions are gated on the qwen4exp profile
    bool ok = true;
    for (int T : { 1028, 300, 1 }) {
        bool adj_new = false, adj_old = false, adj_row = false;
        const uint64_t hn = run(gpu, T, EXPANDED, adj_new), ho = run(gpu, T, VIEW_LEGACY, adj_old), hr = run(gpu, T, ROW_LEGACY, adj_row);
        const bool pass = hn && hn == ho && hn == hr && adj_new && !adj_old && !adj_row;
        std::printf("[ple-norm-mul] T=%d expanded-first %llx (adjacent %d) legacy %llx (adjacent %d) row-legacy %llx (adjacent %d) %s\n", T,
                    (unsigned long long) hn, adj_new, (unsigned long long) ho, adj_old, (unsigned long long) hr, adj_row, pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
