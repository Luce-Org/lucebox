// Parity test for the qsa3 prefill attention store with the sigmoid output gate folded in (qsa.cu
// qsa3_attn_kernel<true>, used by the FLASH_ATTN_EXT -> GATED_RMS_NORM_F16 fusion; LUCE_ATTN_FUSE_LEGACY=1 keeps
// the two kernels). The gated launch's F16 [d, head, query] output must be byte-identical to the plain launch's
// F32 output passed through the gate kernel (ggml_cuda_test_gated_rms_norm_f16 without gamma) on the same q,
// packed K/V, union and the strided [q | gate] projection view. gfx1151 only; SKIP (77) elsewhere.

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" int  ggml_cuda_qsa3_union_cap(int ns);
extern "C" void ggml_cuda_qsa3_union_launch(const int * ids, size_t i1, int n_q, int ns, int nk, int * srow, int * sflag,
        uint16_t * ublk, uint16_t * umask, int * ucount, int phases, int rows_variant, hipStream_t stream);
extern "C" void ggml_cuda_qsa3_attn_launch(
        const float * q, size_t q1, size_t q2, const uint16_t * pk, const uint16_t * pv, const uint16_t * mask, size_t m1,
        const uint16_t * ublk, const uint16_t * umask, const int * ucount, int n_q, int ns, int nk, int n_kv_head,
        float scale, float * out, size_t o1, size_t o2, hipStream_t stream);
extern "C" void ggml_cuda_qsa3_attn_gated_launch(
        const float * q, size_t q1, size_t q2, const uint16_t * pk, const uint16_t * pv, const uint16_t * mask, size_t m1,
        const uint16_t * ublk, const uint16_t * umask, const int * ucount, int n_q, int ns, int nk, int n_kv_head,
        float scale, const float * gate, size_t g1, size_t g2, uint16_t * out, size_t o1, size_t o2, hipStream_t stream);
extern "C" int ggml_cuda_test_gated_rms_norm_f16(int wave, const float * x, const float * gamma, const float * z, void * dst,
        int ncols, int nh, int T, int64_t x_s1, int64_t x_s2, int64_t z_sh, int64_t z_st, int gamma_rows, float eps,
        void * stream);

namespace {

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); return false; } } while (0)

// One id row for query position `pos`, the way qsa_decode_ids writes it: `budget` sorted visible blocks x 4 cells
// (-1 for cells past pos), then the ratio-1 tail cells.
std::vector<int32_t> make_row(std::mt19937 & rng, int pos, int ns, int nk) {
    const int ratio = 4, budget = (ns - (ratio - 1)) / ratio, nb = nk / ratio, nvis = (pos + 1) / ratio;
    std::vector<int> blocks;
    if (nvis <= budget) {
        for (int b = 0; b < nvis; ++b) blocks.push_back(b);
        for (int b = nvis; (int) blocks.size() < budget; ++b) blocks.push_back(b < nb ? b : nb + b);
    } else {
        std::vector<int> all(nvis);
        for (int b = 0; b < nvis; ++b) all[b] = b;
        std::shuffle(all.begin(), all.end(), rng);
        blocks.assign(all.begin(), all.begin() + budget);
        std::sort(blocks.begin(), blocks.end());
    }
    std::vector<int32_t> row(ns, -1);
    for (int j = 0; j < budget * ratio; ++j) {
        const int first = ratio * blocks[j / ratio];
        row[j] = (first + ratio - 1 <= pos) ? first + j % ratio : -1;
    }
    const int br = ((pos + 1) / ratio) * ratio;
    for (int j = 0; j < ratio - 1; ++j) row[budget * ratio + j] = (br + j <= pos) ? br + j : -1;
    return row;
}

template <typename T> struct dev {
    T * p = nullptr;
    ~dev() { if (p) (void) hipFree(p); }
    bool alloc(size_t n) { return hipMalloc(&p, n * sizeof(T)) == hipSuccess; }
};

bool run(int T, int nk, bool & ok) {
    const int D = 256, Hq = 24, Hkv = 2, ns = 4 * 64 + 3, cap = ggml_cuda_qsa3_union_cap(ns);
    const int ngroups = (T + 3) / 4, nblk = nk / 4;
    std::mt19937 rng(1234u + T);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> q((size_t) D * T * Hq), qfull((size_t) 2 * D * Hq * T);
    for (float & x : q) x = u(rng);
    for (float & x : qfull) x = 4.0f * u(rng);
    std::vector<uint16_t> pk((size_t) nblk * Hkv * 1024), pv((size_t) nblk * Hkv * 1024);
    for (uint16_t & x : pk) x = ggml_fp32_to_fp16(u(rng));
    for (uint16_t & x : pv) x = ggml_fp32_to_fp16(u(rng));
    std::vector<int32_t> ids((size_t) T * ns);
    for (int t = 0; t < T; ++t) { const auto row = make_row(rng, t, ns, nk); std::copy(row.begin(), row.end(), ids.begin() + (size_t) t * ns); }

    dev<float> dq, dqfull, dout; dev<uint16_t> dpk, dpv, dublk, dumask, dgate_legacy, dgate_fused; dev<int> dids, dsrow, dsflag, ducount;
    if (!dq.alloc(q.size()) || !dqfull.alloc(qfull.size()) || !dout.alloc((size_t) D * Hq * T) || !dpk.alloc(pk.size()) ||
        !dpv.alloc(pv.size()) || !dublk.alloc((size_t) ngroups * cap) || !dumask.alloc((size_t) ngroups * cap) ||
        !dgate_legacy.alloc((size_t) D * Hq * T) || !dgate_fused.alloc((size_t) D * Hq * T) || !dids.alloc(ids.size()) ||
        !dsrow.alloc((size_t) T * ns) || !dsflag.alloc(T) || !ducount.alloc(ngroups)) { std::fprintf(stderr, "alloc failed\n"); return false; }
    HIP_OK(hipMemcpy(dq.p, q.data(), q.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dqfull.p, qfull.data(), qfull.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dpk.p, pk.data(), pk.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dpv.p, pv.data(), pv.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dids.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
    hipStream_t st = nullptr;
    ggml_cuda_qsa3_union_launch(dids.p, (size_t) ns * 4, T, ns, nk, dsrow.p, dsflag.p, dublk.p, dumask.p, ducount.p, 3, 2, st);
    const float scale = 1.0f / 16.0f;
    // q [D, T, Hq]: query stride D*4, head stride D*T*4. Legacy out [D, Hq, T] F32; the gate view sits at float offset
    // D in the [2*D*Hq, T] projection (head stride 2*D, query stride 2*D*Hq).
    const float * gate = dqfull.p + D;
    ggml_cuda_qsa3_attn_launch(dq.p, (size_t) D * 4, (size_t) D * T * 4, dpk.p, dpv.p, nullptr, 0, dublk.p, dumask.p, ducount.p,
        T, ns, nk, Hkv, scale, dout.p, (size_t) D * 4, (size_t) D * Hq * 4, st);
    if (!ggml_cuda_test_gated_rms_norm_f16(0, dout.p, nullptr, gate, dgate_legacy.p, D, Hq, T, D, (int64_t) D * Hq,
            2 * D, (int64_t) 2 * D * Hq, 1, 0.0f, st)) { std::fprintf(stderr, "gate kernel failed\n"); return false; }
    ggml_cuda_qsa3_attn_gated_launch(dq.p, (size_t) D * 4, (size_t) D * T * 4, dpk.p, dpv.p, nullptr, 0, dublk.p, dumask.p, ducount.p,
        T, ns, nk, Hkv, scale, gate, (size_t) 2 * D * 4, (size_t) 2 * D * Hq * 4, dgate_fused.p, (size_t) D * 2, (size_t) D * Hq * 2, st);
    HIP_OK(hipDeviceSynchronize());
    std::vector<uint16_t> a((size_t) D * Hq * T), b((size_t) D * Hq * T);
    HIP_OK(hipMemcpy(a.data(), dgate_legacy.p, a.size() * 2, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(b.data(), dgate_fused.p, b.size() * 2, hipMemcpyDeviceToHost));
    size_t bad = 0, nonzero = 0;
    for (size_t i = 0; i < a.size(); ++i) { bad += a[i] != b[i]; nonzero += a[i] != 0; }
    std::printf("[qsa3-gate] T=%d nk=%d: %zu / %zu F16 mismatches (%zu nonzero) %s\n", T, nk, bad, a.size(), nonzero, bad ? "FAIL" : "PASS");
    if (bad || nonzero < a.size() / 2) ok = false;
    return true;
}

}  // namespace

int main() {
    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) { std::printf("[qsa3-gate] SKIP: no GPU\n"); return 77; }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess || std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0) {
        std::printf("[qsa3-gate] SKIP: %s is not gfx1151\n", prop.gcnArchName);
        return 77;
    }
    bool ok = true;
    for (int T : { 1028, 300, 129 }) {
        if (!run(T, 2048, ok)) return 1;
    }
    return ok ? 0 : 1;
}
