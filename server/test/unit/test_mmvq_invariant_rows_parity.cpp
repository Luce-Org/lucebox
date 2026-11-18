// Parity test for the weight rows per wave on the batch-invariant Q8_0 multi-column MMVQ launch (mmvq.cu
// c_invariant_rpb, ggml_backend_cuda_set_mmvq_invariant_rows_override; LUCE_MMVQ_INVARIANT_ROWS=1 keeps one row per
// wave). The batch-invariant contract is that every column of a T-column product is byte-identical to the one-column
// product of that column (the MTP verify rows must reproduce T=1 decode); handing a wave 2 or 4 rows must not change
// that. Every qwen4exp dense Q8_0 shape (K x M) is run at T = 2, 3, 4, 5, 8 and, under
// ggml_backend_cuda_set_mmvq_invariant_max_ncols(16), T = 9, 13, 16 (one Q8_0 launch on the generic kernel, eight
// columns per launch on the RDNA4 matvec), with 1 / 2 / 4 rows per wave, on every GPU, and each
// column is memcmp-compared with its own single-column MUL_MAT (the ordinary T=1 kernel, batch-invariance off).
// Row counts that are not multiples of 4 (or 2) must fall back to one row per wave; 40960 rows take the non-persistent
// grid of the LDS path (one block per 16 row groups, as the 248320-row lm head does). The LDS activation staging
// (c_invariant_lds_waves, ggml_backend_cuda_set_mmvq_invariant_lds_override; LUCE_MMVQ_INVARIANT_LDS=0 turns it
// off) is run the same way at 0 / 4 / 8 / 16 / 32 waves per block for every rows setting. The model's dense F32 and
// BF16 products (MMVF: alpha, beta and the loader's stacked alpha|beta view, the router, the shared expert's gate, the
// HC inject, the QSA indexer projections) are held to the same contract at the same widths, and the stacked alpha|beta
// product's rows must equal the separate alpha and beta products' (a batched verify group of any width takes the
// stacked one). SKIP (77) without a GPU.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

namespace {

struct Shape { int64_t K, M; const char * name; ggml_type type = GGML_TYPE_Q8_0; };

const Shape k_shapes[] = {
    { 2560, 10240, "attn_qkv" },  { 2560, 6144, "attn_gate" }, { 6144, 2560, "ssm_out" },
    { 2560, 12288, "attn_q" },    { 2560, 512, "attn_k" },     { 640, 2560, "shexp_down" },
    { 2560, 640, "shexp_up" },    { 320, 10240, "hc_up" },     { 10240, 320, "hc_down" },
    { 2560, 1002, "rows%4!=0" },  { 2560, 1001, "rows odd" },   { 2560, 40960, "rows>32K" },
    { 10240, 4, "hc_inject" },
    { 2560, 48, "ssm_alpha", GGML_TYPE_F32 },   { 2560, 48, "ssm_beta", GGML_TYPE_F32 },
    { 2560, 96, "alpha|beta", GGML_TYPE_F32 },  { 2560, 512, "router", GGML_TYPE_F32 },
    { 2560, 1, "shexp_gate", GGML_TYPE_F32 },   { 10240, 4, "hc_inject", GGML_TYPE_F32 },
    { 2560, 512, "idx_q", GGML_TYPE_BF16 },     { 2560, 128, "idx_k", GGML_TYPE_BF16 },
};
const int k_widths[] = { 2, 3, 4, 5, 8, 9, 13, 16 };
const int k_rows[] = { 1, 2, 4 };
const int k_lds[] = { 0, 4, 8, 16, 32 };

// The weights' bytes in `type` from F32 values.
std::vector<uint8_t> encode(ggml_type type, const std::vector<float> & wf, int64_t K, int64_t M) {
    std::vector<uint8_t> out(ggml_row_size(type, K) * (size_t) M);
    if (type == GGML_TYPE_F32) std::memcpy(out.data(), wf.data(), out.size());
    else if (type == GGML_TYPE_BF16) ggml_fp32_to_bf16_row(wf.data(), (ggml_bf16_t *) out.data(), K * M);
    else ggml_quantize_chunk(type, wf.data(), out.data(), 0, M, K, nullptr);
    return out;
}

// One-column products (T=1 kernel) of every column of x [K, T], as T x M floats.
std::vector<float> reference(ggml_backend_t gpu, ggml_type type, const std::vector<uint8_t> & wq,
                             const std::vector<float> & x, int64_t K, int64_t M, int T) {
    ggml_init_params ip{ 64 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(c, type, K, M);
    ggml_tensor * src1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, T);
    ggml_set_input(w); ggml_set_input(src1);
    ggml_cgraph * gf = ggml_new_graph(c);
    std::vector<ggml_tensor *> outs;
    for (int j = 0; j < T; ++j) {
        ggml_tensor * col = ggml_view_2d(c, src1, K, 1, src1->nb[1], (size_t) j * src1->nb[1]);
        ggml_tensor * o = ggml_mul_mat(c, w, col);
        ggml_set_output(o); ggml_build_forward_expand(gf, o); outs.push_back(o);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(src1, x.data(), 0, x.size() * sizeof(float));
    const bool prev = ggml_backend_cuda_set_mmvq_batch_invariant(false);
    const bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_cuda_set_mmvq_batch_invariant(prev);
    std::vector<float> out((size_t) T * M);
    if (ok) for (int j = 0; j < T; ++j) ggml_backend_tensor_get(outs[(size_t) j], out.data() + (size_t) j * M, 0, M * sizeof(float));
    else out.clear();
    ggml_backend_buffer_free(buf); ggml_free(c);
    return out;
}

// The T-column product on the batch-invariant path with `rows` weight rows per wave and `lds` waves per block
// staging the activation in LDS (0 = off).
std::vector<float> batched(ggml_backend_t gpu, ggml_type type, const std::vector<uint8_t> & wq,
                           const std::vector<float> & x, int64_t K, int64_t M, int T, int rows, int lds) {
    ggml_init_params ip{ 64 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(c, type, K, M);
    ggml_tensor * src1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, T);
    ggml_set_input(w); ggml_set_input(src1);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_tensor * o = ggml_mul_mat(c, w, src1);
    ggml_set_output(o); ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(src1, x.data(), 0, x.size() * sizeof(float));
    const bool prev = ggml_backend_cuda_set_mmvq_batch_invariant(true);
    const int prev_width = ggml_backend_cuda_set_mmvq_invariant_max_ncols(T > 8 ? 16 : 0);
    const int prev_rows = ggml_backend_cuda_set_mmvq_invariant_rows_override(rows);
    const int prev_lds = ggml_backend_cuda_set_mmvq_invariant_lds_override(lds);
    const bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_cuda_set_mmvq_invariant_lds_override(prev_lds);
    ggml_backend_cuda_set_mmvq_invariant_rows_override(prev_rows);
    ggml_backend_cuda_set_mmvq_invariant_max_ncols(prev_width);
    ggml_backend_cuda_set_mmvq_batch_invariant(prev);
    std::vector<float> out((size_t) T * M);
    if (ok) ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
    else out.clear();
    ggml_backend_buffer_free(buf); ggml_free(c);
    return out;
}

}  // namespace

int main() {
    const int n_dev = ggml_backend_cuda_get_device_count();
    if (n_dev < 1) { std::printf("[mmvq-invariant-rows] SKIP: no GPU\n"); return 77; }
    bool ok = true;
    int cases = 0;
    for (int dev = 0; dev < n_dev; ++dev) {
    ggml_backend_t gpu = ggml_backend_cuda_init(dev);
    if (!gpu) { std::printf("[mmvq-invariant-rows] device %d: no backend\n", dev); ok = false; continue; }
    std::printf("[mmvq-invariant-rows] device %d %s\n", dev, ggml_backend_name(gpu));
    uint32_t r = 7u;
    auto rnd = [&r]() { r = r * 1664525u + 1013904223u; return (float) (r >> 8) / (float) (1u << 24) * 2.0f - 1.0f; };
    for (const Shape & s : k_shapes) {
        std::vector<float> wf((size_t) s.K * s.M);
        for (float & v : wf) v = rnd();
        const std::vector<uint8_t> wq = encode(s.type, wf, s.K, s.M);
        // Rows per wave and LDS staging are Q8_0 launch settings; the float products run once per width.
        const bool q8 = s.type == GGML_TYPE_Q8_0;
        for (int T : k_widths) {
            std::vector<float> x((size_t) s.K * T);
            for (float & v : x) v = rnd();
            const std::vector<float> want = reference(gpu, s.type, wq, x, s.K, s.M, T);
            for (int rows : k_rows) for (int lds : k_lds) {
                if (!q8 && (rows != 1 || lds != 0)) continue;
                const std::vector<float> got = batched(gpu, s.type, wq, x, s.K, s.M, T, rows, lds);
                const bool same = !want.empty() && got.size() == want.size() &&
                    std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0;
                size_t first_bad = 0;
                if (!same && got.size() == want.size()) {
                    while (first_bad < want.size() && std::memcmp(&got[first_bad], &want[first_bad], sizeof(float)) == 0) ++first_bad;
                }
                std::printf("[mmvq-invariant-rows] %-10s %-4s K=%lld M=%lld T=%d rows=%d lds=%d %s%s\n", s.name,
                            ggml_type_name(s.type), (long long) s.K, (long long) s.M, T, rows, lds, same ? "PASS" : "FAIL",
                            same ? "" : (std::string(" first_diff col=") + std::to_string(first_bad / (size_t) s.M) +
                                         " row=" + std::to_string(first_bad % (size_t) s.M)).c_str());
                ok = ok && same;
                ++cases;
            }
        }
    }
    // The stacked alpha|beta view (the loader's ssm_alpha_beta: alpha's rows, then beta's) against the two products,
    // at one column and every batched width.
    {
        const int64_t K = 2560, H = 48;
        std::vector<float> wf((size_t) K * 2 * H);
        for (float & v : wf) v = rnd();
        const std::vector<float> af(wf.begin(), wf.begin() + K * H), bf(wf.begin() + K * H, wf.end());
        const std::vector<uint8_t> ab = encode(GGML_TYPE_F32, wf, K, 2 * H), a = encode(GGML_TYPE_F32, af, K, H),
                                   b = encode(GGML_TYPE_F32, bf, K, H);
        std::vector<int> widths = { 1 };
        widths.insert(widths.end(), std::begin(k_widths), std::end(k_widths));
        for (int T : widths) {
            std::vector<float> x((size_t) K * T);
            for (float & v : x) v = rnd();
            const std::vector<float> got = batched(gpu, GGML_TYPE_F32, ab, x, K, 2 * H, T, 1, 0);
            const std::vector<float> ga = batched(gpu, GGML_TYPE_F32, a, x, K, H, T, 1, 0);
            const std::vector<float> gb = batched(gpu, GGML_TYPE_F32, b, x, K, H, T, 1, 0);
            bool same = got.size() == (size_t) (2 * H * T) && ga.size() == (size_t) (H * T) && gb.size() == ga.size();
            for (int j = 0; same && j < T; ++j) {
                same = std::memcmp(&got[(size_t) j * 2 * H], &ga[(size_t) j * H], H * sizeof(float)) == 0 &&
                       std::memcmp(&got[(size_t) (j * 2 + 1) * H], &gb[(size_t) j * H], H * sizeof(float)) == 0;
            }
            std::printf("[mmvq-invariant-rows] alpha|beta = alpha, beta T=%d %s\n", T, same ? "PASS" : "FAIL");
            ok = ok && same;
            ++cases;
        }
    }
    ggml_backend_free(gpu);
    }
    std::printf("[mmvq-invariant-rows] cases=%d %s\n", cases, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
