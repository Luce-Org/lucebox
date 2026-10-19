#include "gated-norm.cuh"
#include "qwen4exp-common.cuh"
#include <cstdlib>
#include <cstring>

// One block per (head, token) row, the thread mapping and reduction of rms_norm_f32<256, true>.
// NORM = false: sigmoid(z) * x only (the attention output gate), same product as unary_mul(sigmoid).
template <int block_size, bool NORM>
static __global__ void gated_rms_norm_f16_kernel(const float * x, const float * gamma, const float * z, half * dst,
        const int ncols, const int64_t x_s1, const int64_t x_s2, const int64_t z_sh, const int64_t z_st, const int gamma_rows,
        const float eps) {
    const int h = blockIdx.x, t = blockIdx.y, nh = gridDim.x, tid = threadIdx.x;
    const float * xr = x + t * x_s2 + h * x_s1;
    const float * zr = z + t * z_st + (int64_t) h * z_sh;
    half * d = dst + ((int64_t) t * nh + h) * ncols;
    if constexpr (!NORM) {
        for (int col = tid; col < ncols; col += block_size) d[col] = gn_f2h_sat(q4x_sigmoid(zr[col]) * xr[col]);
        return;
    }
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = xr[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    const float * g  = gamma + (int64_t) (h % gamma_rows) * ncols;
    for (int col = tid; col < ncols; col += block_size) {
        const float v = scale * xr[col] * g[col];
        d[col] = gn_f2h_sat(q4x_sigmoid(zr[col]) * v);
    }
}

// Rows of <= 128 (GDN heads): two rows per 256-thread block, one per 128-thread half, so no thread idles. Each half
// reproduces block_reduce<SUM, 256> over [its 4 warp partials, 4 zero partials] exactly (same warp_reduce_sum on the
// same lane values), so the result is bit-identical to gated_rms_norm_f16_kernel<256, true>.
static __global__ void gated_rms_norm2_f16_kernel(const float * x, const float * gamma, const float * z, half * dst,
        const int ncols, const int nh, const int64_t x_s1, const int64_t x_s2, const int64_t z_sh, const int64_t z_st,
        const int gamma_rows, const float eps) {
    __shared__ float s_sum[8];
    const int tid = threadIdx.x, grp = tid >> 7, gt = tid & 127, lane = tid & 31, warp = tid >> 5;
    const int h = blockIdx.x * 2 + grp, t = blockIdx.y;
    const bool live = h < nh;
    const float * xr = x + t * x_s2 + (int64_t) (live ? h : 0) * x_s1;
    float tmp = 0.0f;
    if (live && gt < ncols) { const float xi = xr[gt]; tmp += xi * xi; }
    tmp = warp_reduce_sum(tmp);
    if (lane == 0) s_sum[warp] = tmp;
    __syncthreads();
    tmp = lane < 4 ? s_sum[grp * 4 + lane] : 0.0f;
    tmp = warp_reduce_sum(tmp);
    if (!live || gt >= ncols) return;
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    const float * g  = gamma + (int64_t) (h % gamma_rows) * ncols;
    const float * zr = z + t * z_st + (int64_t) h * z_sh;
    half * d = dst + ((int64_t) t * nh + h) * ncols;
    const float v = scale * xr[gt] * g[gt];
    d[gt] = gn_f2h_sat(q4x_sigmoid(zr[gt]) * v);
}

// Wave-per-row ncols == 128 norm (LUCE_GDN_NORM_LEGACY=1 keeps gated_rms_norm2_f16_kernel). One 32-lane wave owns
// ROWS (head, token) rows in turn; lane l holds columns l, 32+l, 64+l, 96+l -- exactly the values that warp w of the
// 128-thread mapping held in lane l -- so the four per-warp xor-tree partials p_w and their (p0+p2)+(p1+p3)
// combination (what block_reduce's second stage computes over [p0..p3, 0, ...]) reproduce gated_rms_norm2_f16_kernel
// bit for bit, with no LDS and no block barrier. 8 waves per block. (A wave-per-row version of the gamma == NULL
// 256-wide gate was measured slower than the block kernel on gfx1151, 1.90 vs 1.54 ms at T=4131, so it was dropped.)
template <int ROWS>
static __global__ void __launch_bounds__(256) gated_rms_norm_wave_kernel(const float * __restrict__ x,
        const float * __restrict__ gamma, const float * __restrict__ z, half * __restrict__ dst, const int ncols,
        const int nh, const int64_t n_rows, const int64_t x_s1, const int64_t x_s2, const int64_t z_sh,
        const int64_t z_st, const int gamma_rows, const float eps) {
    const int lane = threadIdx.x & 31;
    const int64_t row0 = ((int64_t) blockIdx.x * 8 + (threadIdx.x >> 5)) * ROWS;
    for (int r = 0; r < ROWS; ++r) {
        const int64_t row = row0 + r;
        if (row >= n_rows) return;
        const int t = (int) (row / nh), h = (int) (row - (int64_t) t * nh);
        const float * xr = x + t * x_s2 + (int64_t) h * x_s1;
        const float * zr = z + t * z_st + (int64_t) h * z_sh;
        half * d = dst + row * ncols;
        // All loads of a row are issued before its first store so the memory latency overlaps.
        float xv[4], p[4], zv[4], gv[4];
        const float * g = gamma + (int64_t) (h % gamma_rows) * ncols;
#pragma unroll
        for (int j = 0; j < 4; ++j) { zv[j] = zr[lane + 32 * j]; gv[j] = g[lane + 32 * j]; }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // The HIP device build runs -ffp-contract=fast-honor-pragmas: without this pragma the square is
            // contracted into warp_reduce_sum's first add (fma(x, x, shfl)), which the block kernels never do.
#pragma clang fp contract(off)
            xv[j] = xr[lane + 32 * j];
            const float sq = xv[j] * xv[j];
            p[j] = warp_reduce_sum(sq);
        }
        const float tmp   = (p[0] + p[2]) + (p[1] + p[3]);
        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + eps);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v = scale * xv[j] * gv[j];
            d[lane + 32 * j] = gn_f2h_sat(q4x_sigmoid(zv[j]) * v);
        }
    }
}

static bool gated_norm_wave_enabled() {
    static const bool on = [] {
        const char * v = getenv("LUCE_GDN_NORM_LEGACY");
        return !(v && strcmp(v, "0") != 0);
    }();
    return on;
}

// Shared by the op and the parity unit test: `wave` selects the kernel family.
static void gated_rms_norm_f16_launch(bool wave, const float * x, const float * gamma, const float * z, half * dst,
        int ncols, int nh, int T, int64_t x_s1, int64_t x_s2, int64_t z_sh, int64_t z_st, int gamma_rows, float eps,
        cudaStream_t stream) {
    constexpr int ROWS = 4;
    const int64_t n_rows = (int64_t) nh * T;
    const unsigned blocks = (unsigned) ((n_rows + 8 * ROWS - 1) / (8 * ROWS));
    if (wave && gamma && ncols == 128) {
        gated_rms_norm_wave_kernel<ROWS><<<blocks, 256, 0, stream>>>(
            x, gamma, z, dst, ncols, nh, n_rows, x_s1, x_s2, z_sh, z_st, gamma_rows, eps);
    } else if (gamma && ncols <= 128) {
        gated_rms_norm2_f16_kernel<<<dim3((unsigned) ((nh + 1) / 2), (unsigned) T, 1), 256, 0, stream>>>(
            x, gamma, z, dst, ncols, nh, x_s1, x_s2, z_sh, z_st, gamma_rows, eps);
    } else if (gamma) {
        gated_rms_norm_f16_kernel<256, true><<<dim3(nh, T, 1), 256, 32 * sizeof(float), stream>>>(
            x, gamma, z, dst, ncols, x_s1, x_s2, z_sh, z_st, gamma_rows, eps);
    } else {
        gated_rms_norm_f16_kernel<256, false><<<dim3(nh, T, 1), 256, 0, stream>>>(
            x, nullptr, z, dst, ncols, x_s1, x_s2, z_sh, z_st, 1, eps);
    }
}

extern "C" { GGML_BACKEND_API int ggml_cuda_test_gated_rms_norm_f16(int wave,
        const float * x, const float * gamma, const float * z, void * dst,
        int ncols, int nh, int T, int64_t x_s1, int64_t x_s2, int64_t z_sh, int64_t z_st, int gamma_rows, float eps,
        void * raw_stream) {
    if (!x || !z || !dst || ncols <= 0 || ncols > 256 || nh <= 0 || T <= 0 || (gamma && gamma_rows <= 0)) return 0;
    gated_rms_norm_f16_launch(wave != 0, x, gamma, z, (half *) dst, ncols, nh, T, x_s1, x_s2, z_sh, z_st,
        gamma ? gamma_rows : 1, eps, (cudaStream_t) raw_stream);
    return cudaGetLastError() == cudaSuccess;
} }

void ggml_cuda_op_gated_rms_norm_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0], * gamma = dst->src[1], * z = dst->src[2];
    const int ncols = (int) x->ne[0], nh = (int) x->ne[1], T = (int) x->ne[2];
    GGML_ASSERT(ncols <= 256 && dst->type == GGML_TYPE_F16 && ggml_is_contiguous(dst));
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    // z is [ncols*nh, T] (head stride ncols) or a [ncols, nh, T] view with its own strides.
    const int64_t z_sh = z->ne[0] == ncols * nh ? ncols : (int64_t) (z->nb[1] / sizeof(float));
    const int64_t z_st = z->ne[0] == ncols * nh ? (int64_t) (z->nb[1] / sizeof(float)) : (int64_t) (z->nb[2] / sizeof(float));
    const int64_t x_s1 = (int64_t) (x->nb[1] / sizeof(float)), x_s2 = (int64_t) (x->nb[2] / sizeof(float));
    gated_rms_norm_f16_launch(gated_norm_wave_enabled(), (const float *) x->data, gamma ? (const float *) gamma->data : nullptr,
        (const float *) z->data, (half *) dst->data, ncols, nh, T, x_s1, x_s2, z_sh, z_st,
        gamma ? (int) (ggml_nelements(gamma) / ncols) : 1, eps, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}
