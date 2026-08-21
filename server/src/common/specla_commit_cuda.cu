// Fused SpecLA DeltaConstruct commit — see specla_commit_cuda.h.

#include "specla_commit_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace dflash::common {

namespace {

constexpr int kBlock = 256;
constexpr int kMaxAccept = 64;  // >= every verify window / ddtree budget in use

// Grid: x over the S_k*S_v state elements of one (head, layer) plane,
//       y over head-layer planes (hl = h + H*l).
// Factor element (·, h, l, t) lives at block index fo = h + H*(l + L*t) —
// identical for fk (stride S_k), fv (stride S_v), and fg (stride 1).
__global__ void specla_commit_kernel(float * const * ssm_ptrs,
                                     const float * __restrict__ fk,
                                     const float * __restrict__ fv,
                                     const float * __restrict__ fg,
                                     const int32_t * __restrict__ idx,
                                     int A, int S_k, int S_v, int H, int L) {
    const int hl = blockIdx.y;
    const int l  = hl / H;
    const int h  = hl % H;

    // Per-plane scalars shared by every thread: the accepted-end gate and the
    // per-token decay weights along the accepted path.
    __shared__ float s_w[kMaxAccept];
    __shared__ float s_gA_exp;
    __shared__ int   s_tok[kMaxAccept];
    if (threadIdx.x < A) {
        s_tok[threadIdx.x] = idx[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int tA = s_tok[A - 1];
        const float gA = fg[(size_t)h + (size_t)H * (l + (size_t)L * tA)];
        s_gA_exp = expf(gA);
        for (int t = 0; t < A; t++) {
            const size_t fo = (size_t)h + (size_t)H * (l + (size_t)L * s_tok[t]);
            s_w[t] = expf(gA - fg[fo]);
        }
    }
    __syncthreads();

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= S_k * S_v) return;
    const int i = e % S_k;   // state row (k-dim, ne0)
    const int c = e / S_k;   // state column

    float * s_plane = ssm_ptrs[l] + (size_t)h * S_k * S_v;
    float acc = s_gA_exp * s_plane[(size_t)c * S_k + i];
    for (int t = 0; t < A; t++) {
        const size_t fo = (size_t)h + (size_t)H * (l + (size_t)L * s_tok[t]);
        acc += s_w[t] * fk[(size_t)fo * S_k + i] * fv[(size_t)fo * S_v + c];
    }
    s_plane[(size_t)c * S_k + i] = acc;
}

__global__ void specla_compact_kernel(
        const float * __restrict__ src_k,
        const float * __restrict__ src_v,
        const float * __restrict__ src_g,
        const float * __restrict__ src_conv,
        float * __restrict__ dst_k,
        float * __restrict__ dst_v,
        float * __restrict__ dst_g,
        float * __restrict__ dst_conv,
        const int32_t * __restrict__ idx,
        int A, int S_k, int S_v, int H, int L, int C) {
    const size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    const size_t k_n = (size_t)A*L*H*S_k;
    const size_t v_n = (size_t)A*L*H*S_v;
    const size_t g_n = (size_t)A*L*H;
    const size_t c_n = (size_t)A*L*C;
    if (e < k_n) {
        const int t = e/(L*H*S_k);
        const size_t inner = e%(L*H*S_k);
        dst_k[e] = src_k[(size_t)idx[t]*L*H*S_k + inner];
    }
    if (e < v_n) {
        const int t = e/(L*H*S_v);
        const size_t inner = e%(L*H*S_v);
        dst_v[e] = src_v[(size_t)idx[t]*L*H*S_v + inner];
    }
    if (e < g_n) {
        const int t = e/(L*H);
        const size_t inner = e%(L*H);
        dst_g[e] = src_g[(size_t)idx[t]*L*H + inner];
    }
    if (e < c_n) {
        const int t = e/(L*C);
        const size_t inner = e%(L*C);
        dst_conv[e] = src_conv[(size_t)idx[t]*L*C + inner];
    }
}

__global__ void specla_flush_state_raw_kernel(
        float * const * ssm_ptrs,
        const float * __restrict__ fk,
        const float * __restrict__ fv,
        const float * __restrict__ fg,
        int A, int S_k, int S_v, int H, int L) {
    const int hl = blockIdx.y;
    const int l = hl/H;
    const int h = hl%H;
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= S_k*S_v) return;
    const int row = e%S_k;
    const int col = e/S_k;
    float * plane = ssm_ptrs[l] + (size_t)h*S_k*S_v;
    float acc = plane[(size_t)col*S_k + row];
    for (int t = 0; t < A; ++t) {
        const size_t hl_t = (size_t)h + (size_t)H*(l + (size_t)L*t);
        acc = fmaf(fk[hl_t*S_k + row], fv[hl_t*S_v + col],
                   expf(fg[hl_t])*acc);
    }
    plane[(size_t)col*S_k + row] = acc;
}

__global__ void specla_flush_conv_raw_kernel(
        float * const * conv_ptrs,
        const float * __restrict__ conv,
        int A, int L, int C, int d_conv) {
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= L*C) return;
    const int l = e/C;
    const int channel = e%C;
    float * state = conv_ptrs[l] + (size_t)channel*(d_conv - 1);
    for (int t = 0; t < A; ++t) {
        for (int j = 0; j < d_conv - 2; ++j) state[j] = state[j + 1];
        state[d_conv - 2] =
            conv[(size_t)channel + (size_t)C*(l + (size_t)L*t)];
    }
}

}  // namespace

bool specla_commit_fused(float * const * ssm_ptrs_dev,
                         const float * fk,
                         const float * fv,
                         const float * fg,
                         const int32_t * idx_dev,
                         int A, int S_k, int S_v, int H, int n_delta,
                         void * stream,
                         bool * launched) {
    if (launched) *launched = false;
    if (!ssm_ptrs_dev || !fk || !fv || !fg || !idx_dev) return false;
    if (A <= 0 || A > kMaxAccept || S_k <= 0 || S_v <= 0 || H <= 0 || n_delta <= 0) {
        return false;
    }
    const int planes = H * n_delta;
    if (planes > 65535) return false;  // grid.y limit

    dim3 block(kBlock);
    dim3 grid(((unsigned)(S_k * S_v) + kBlock - 1) / kBlock, (unsigned)planes);
    (void)cudaGetLastError();  // discard any unrelated prior launch status
    specla_commit_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        ssm_ptrs_dev, fk, fv, fg, idx_dev, A, S_k, S_v, H, n_delta);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (launched) *launched = true;
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess;
}

bool specla_compact_fused(
        const float * src_k, const float * src_v, const float * src_g,
        const float * src_conv, float * dst_k, float * dst_v, float * dst_g,
        float * dst_conv, const int32_t * idx_dev, int A, int S_k, int S_v,
        int H, int n_delta, int conv_channels, void * stream) {
    if (!src_k || !src_v || !src_g || !src_conv || !dst_k || !dst_v ||
        !dst_g || !dst_conv || !idx_dev || A <= 0 || A > kMaxAccept) {
        return false;
    }
    const size_t n = std::max(
        std::max((size_t)A*n_delta*H*S_k, (size_t)A*n_delta*H*S_v),
        std::max((size_t)A*n_delta*H, (size_t)A*n_delta*conv_channels));
    const dim3 block(kBlock);
    const dim3 grid((unsigned)((n + kBlock - 1)/kBlock));
    (void)cudaGetLastError();
    specla_compact_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        src_k, src_v, src_g, src_conv, dst_k, dst_v, dst_g, dst_conv,
        idx_dev, A, S_k, S_v, H, n_delta, conv_channels);
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess;
}

bool specla_flush_raw_fused(
        float * const * ssm_ptrs_dev, float * const * conv_ptrs_dev,
        const float * fk, const float * fv, const float * fg,
        const float * conv, int A, int S_k, int S_v, int H, int n_delta,
        int conv_channels, int d_conv, void * stream) {
    if (!ssm_ptrs_dev || !conv_ptrs_dev || !fk || !fv || !fg || !conv ||
        A <= 0 || A > kMaxAccept || d_conv < 2) return false;
    const dim3 block(kBlock);
    const dim3 state_grid(
        ((unsigned)(S_k*S_v) + kBlock - 1)/kBlock,
        (unsigned)(H*n_delta));
    (void)cudaGetLastError();
    specla_flush_state_raw_kernel<<<state_grid, block, 0, (cudaStream_t)stream>>>(
        ssm_ptrs_dev, fk, fv, fg, A, S_k, S_v, H, n_delta);
    const dim3 conv_grid(
        ((unsigned)(n_delta*conv_channels) + kBlock - 1)/kBlock);
    specla_flush_conv_raw_kernel<<<conv_grid, block, 0, (cudaStream_t)stream>>>(
        conv_ptrs_dev, conv, A, n_delta, conv_channels, d_conv);
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess;
}

bool specla_commit_conv_raw_fused(float * const * conv_ptrs_dev,
                                  const float * conv,
                                  int A, int n_tokens,
                                  int n_delta, int conv_channels,
                                  int d_conv, void * stream) {
    if (!conv_ptrs_dev || !conv || A <= 0 || A > n_tokens ||
        n_tokens <= 0 ||
        n_delta <= 0 || conv_channels <= 0 || d_conv < 2) {
        return false;
    }
    const dim3 block(kBlock);
    const dim3 grid(
        ((unsigned)(n_delta*conv_channels) + kBlock - 1)/kBlock);
    (void)cudaGetLastError();
    specla_flush_conv_raw_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        conv_ptrs_dev, conv, A, n_delta, conv_channels, d_conv);
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess;
}

bool specla_rotate_pending_factors(const SpeclaFactorBanks & banks,
                                   const int32_t * idx_dev,
                                   int pending_bank,
                                   bool walked_sibling,
                                   int commit_n,
                                   int S_k, int S_v, int H,
                                   int n_delta, int conv_channels,
                                   void * stream,
                                   int * out_pending_bank) {
    if (!out_pending_bank || pending_bank < 0 || pending_bank > 1 ||
        commit_n <= 0) {
        return false;
    }
    for (int b = 0; b < 2; ++b) {
        if (!banks.k[b] || !banks.v[b] || !banks.g[b] || !banks.conv[b]) {
            return false;
        }
    }

    const int current_bank = 1 - pending_bank;
    if (!walked_sibling) {
        // A chain acceptance is already in path order: only the host-side
        // bank role changes.
        *out_pending_bank = current_bank;
        return true;
    }

    // A sibling walk scatters the accepted path across the current bank.
    // Compact it into the old pending bank, which then becomes pending.
    if (!idx_dev) return false;
    const bool ok = specla_compact_fused(
        banks.k[current_bank], banks.v[current_bank],
        banks.g[current_bank], banks.conv[current_bank],
        banks.k[pending_bank], banks.v[pending_bank],
        banks.g[pending_bank], banks.conv[pending_bank],
        idx_dev, commit_n, S_k, S_v, H, n_delta, conv_channels, stream);
    if (ok) *out_pending_bank = pending_bank;
    return ok;
}

bool specla_flush_pending_factors(const SpeclaFactorBanks & banks,
                                  float * const * ssm_ptrs_dev,
                                  float * const * conv_ptrs_dev,
                                  int pending_bank,
                                  int pending_count,
                                  int S_k, int S_v, int H,
                                  int n_delta, int conv_channels, int d_conv,
                                  void * stream) {
    if (pending_count <= 0) return true;
    if (pending_bank < 0 || pending_bank > 1 || !ssm_ptrs_dev ||
        !conv_ptrs_dev || !banks.k[pending_bank] || !banks.v[pending_bank] ||
        !banks.g[pending_bank] || !banks.conv[pending_bank]) {
        return false;
    }
    return specla_flush_raw_fused(
        ssm_ptrs_dev, conv_ptrs_dev,
        banks.k[pending_bank], banks.v[pending_bank],
        banks.g[pending_bank], banks.conv[pending_bank],
        pending_count, S_k, S_v, H, n_delta, conv_channels, d_conv, stream);
}

}  // namespace dflash::common
