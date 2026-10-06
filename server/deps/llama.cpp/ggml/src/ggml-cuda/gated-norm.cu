#include "gated-norm.cuh"
#include "qwen4exp-common.cuh"

// same saturating conversion as mq_f2h_sat in mmb-q8f16.cuh
static __device__ __forceinline__ half gn_f2h_sat(const float v) {
    return __float2half(fabsf(v) > 65504.0f ? copysignf(65504.0f, v) : v);
}

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

void ggml_cuda_op_gated_rms_norm_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0], * gamma = dst->src[1], * z = dst->src[2];
    const int ncols = (int) x->ne[0], nh = (int) x->ne[1], T = (int) x->ne[2];
    GGML_ASSERT(ncols <= 256 && dst->type == GGML_TYPE_F16 && ggml_is_contiguous(dst));
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    // z is [ncols*nh, T] (head stride ncols) or a [ncols, nh, T] view with its own strides.
    const int64_t z_sh = z->ne[0] == ncols * nh ? ncols : (int64_t) (z->nb[1] / sizeof(float));
    const int64_t z_st = z->ne[0] == ncols * nh ? (int64_t) (z->nb[1] / sizeof(float)) : (int64_t) (z->nb[2] / sizeof(float));
    const dim3 grid((unsigned) nh, (unsigned) T, 1);
    const int64_t x_s1 = (int64_t) (x->nb[1] / sizeof(float)), x_s2 = (int64_t) (x->nb[2] / sizeof(float));
    if (gamma && ncols <= 128) {
        gated_rms_norm2_f16_kernel<<<dim3((unsigned) ((nh + 1) / 2), (unsigned) T, 1), 256, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) gamma->data, (const float *) z->data, (half *) dst->data, ncols, nh,
            x_s1, x_s2, z_sh, z_st, (int) (ggml_nelements(gamma) / ncols), eps);
    } else if (gamma) {
        gated_rms_norm_f16_kernel<256, true><<<grid, 256, 32 * sizeof(float), ctx.stream()>>>(
            (const float *) x->data, (const float *) gamma->data, (const float *) z->data, (half *) dst->data, ncols,
            x_s1, x_s2, z_sh, z_st, (int) (ggml_nelements(gamma) / ncols), eps);
    } else {
        gated_rms_norm_f16_kernel<256, false><<<grid, 256, 0, ctx.stream()>>>(
            (const float *) x->data, nullptr, (const float *) z->data, (half *) dst->data, ncols,
            x_s1, x_s2, z_sh, z_st, 1, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}
