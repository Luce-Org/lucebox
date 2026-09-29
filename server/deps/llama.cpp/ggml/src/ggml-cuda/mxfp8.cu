#include "mxfp8.cuh"
#include "dequantize.cuh"
#include <climits>
#include <cstdlib>

static constexpr int MXFP8_MMV_WARPS = 4;   // rows per thread block, one 32-lane warp per row

// Lane l of a row's warp owns codes [8l, 8l+8) of every 256-weight block (scale group l/4):
// one coalesced 256-byte code read per block, and 2 float4 activation reads per column.
template <int ncols>
static __global__ void __launch_bounds__(32*MXFP8_MMV_WARPS) mul_mat_vec_mxfp8(
        const block_mxfp8 * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int nblocks, const int nrows, const int64_t stride_row_x,
        const int64_t stride_col_y, const int64_t stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst) {
    const int lane = threadIdx.x;
    const int row  = blockIdx.x*MXFP8_MMV_WARPS + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    x   += (sample/sample_ratio)*stride_sample_x + (channel/channel_ratio)*stride_channel_x + row*stride_row_x;
    y   += sample*stride_sample_y + channel*stride_channel_y + 8*lane;
    dst += sample*stride_sample_dst + channel*stride_channel_dst + row;

    float acc[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        acc[j] = 0.0f;
    }
#pragma unroll 2
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint2 q = ((const uint2 *) x[ib].qs)[lane];
        const float d = mxfp8_scale(x[ib].e[lane/4]);
        float w[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            w[k]     = mxfp8_value((q.x >> (8*k)) & 0xff) * d;
            w[k + 4] = mxfp8_value((q.y >> (8*k)) & 0xff) * d;
        }
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float4 a = *(const float4 *) (y + j*stride_col_y + ib*QK_MXFP8);
            const float4 b = *(const float4 *) (y + j*stride_col_y + ib*QK_MXFP8 + 4);
            float s = acc[j];
            s = fmaf(w[0], a.x, s); s = fmaf(w[1], a.y, s); s = fmaf(w[2], a.z, s); s = fmaf(w[3], a.w, s);
            s = fmaf(w[4], b.x, s); s = fmaf(w[5], b.y, s); s = fmaf(w[6], b.z, s); s = fmaf(w[7], b.w, s);
            acc[j] = s;
        }
    }
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        acc[j] = warp_reduce_sum<32>(acc[j]);
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            dst[j*stride_col_dst] = acc[j];
        }
    }
}

static int mxfp8_mmv_max_ncols() {
    static const int value = [] {
        const char * e = std::getenv("LUCE_MXFP8_MMV_MAX_NCOLS");
        const int v = e ? std::atoi(e) : MXFP8_MMV_MAX_NCOLS;
        return v < 0 ? 0 : v > MXFP8_MMV_MAX_NCOLS ? MXFP8_MMV_MAX_NCOLS : v;
    }();
    return value;
}

bool ggml_cuda_mxfp8_mul_mat_vec_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const size_t bs = sizeof(block_mxfp8);
    return src0->type == GGML_TYPE_MXFP8 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= 1 && src1->ne[1] <= mxfp8_mmv_max_ncols() &&
        src0->ne[0] % QK_MXFP8 == 0 && src0->ne[0] == src1->ne[0] && src0->ne[1] <= INT_MAX &&
        src0->nb[0] == bs && src0->nb[1] % bs == 0 && src0->nb[2] % bs == 0 && src0->nb[3] % bs == 0 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0 &&
        (uintptr_t) src1->data % 16 == 0 && (uintptr_t) src0->data % 8 == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] &&
        dst->ne[2] == src1->ne[2] && dst->ne[3] == src1->ne[3] &&
        src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0 &&
        src1->ne[2] <= 65535 && src1->ne[3] <= 65535;
}

void ggml_cuda_mxfp8_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_mxfp8_mul_mat_vec_supported(src0, src1, dst));
    const size_t bs = sizeof(block_mxfp8);
    const int nrows = (int) src0->ne[1];
    const dim3 grid((nrows + MXFP8_MMV_WARPS - 1)/MXFP8_MMV_WARPS, (unsigned) src1->ne[2], (unsigned) src1->ne[3]);
    const dim3 block(32, MXFP8_MMV_WARPS, 1);
    const block_mxfp8 * x = (const block_mxfp8 *) src0->data;
    const float * y = (const float *) src1->data;
    float * d = (float *) dst->data;
    const int nblocks = (int) (src0->ne[0] / QK_MXFP8);
    const int64_t sx1 = src0->nb[1]/bs, sx2 = src0->nb[2]/bs, sx3 = src0->nb[3]/bs;
    const int64_t sy1 = src1->nb[1]/sizeof(float), sy2 = src1->nb[2]/sizeof(float), sy3 = src1->nb[3]/sizeof(float);
    const int64_t sd1 = dst->nb[1]/sizeof(float), sd2 = dst->nb[2]/sizeof(float), sd3 = dst->nb[3]/sizeof(float);
    const int cr = (int) (src1->ne[2]/src0->ne[2]), sr = (int) (src1->ne[3]/src0->ne[3]);
#define MXFP8_LAUNCH(n) mul_mat_vec_mxfp8<n><<<grid, block, 0, stream>>>(x, y, d, nblocks, nrows, sx1, sy1, sd1, cr, sx2, sy2, sd2, sr, sx3, sy3, sd3); break
    switch (src1->ne[1]) {
        case 1: MXFP8_LAUNCH(1);
        case 2: MXFP8_LAUNCH(2);
        case 3: MXFP8_LAUNCH(3);
        case 4: MXFP8_LAUNCH(4);
        case 5: MXFP8_LAUNCH(5);
        case 6: MXFP8_LAUNCH(6);
        case 7: MXFP8_LAUNCH(7);
        case 8: MXFP8_LAUNCH(8);
        default: GGML_ABORT("mxfp8 mmv: unsupported column count");
    }
#undef MXFP8_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
