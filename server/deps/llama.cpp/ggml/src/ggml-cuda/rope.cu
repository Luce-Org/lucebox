#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

// FP64 angle computation: avoids the FP32 "precision wall"
// (RoPE base must be < 1/eps_mach ~ 8.4e6, see arXiv:2602.10959).
// Empirically required for Qwen3.5 (freq_base=1e7).
//
// Returns the UNREDUCED angle  theta = (pos * theta_scale^exp_int)  in double.
// The mod-2pi reduction is deferred to rope_yarn(), right before cosf/sinf, so
// it happens AFTER all downstream angle scaling (freq_scale, freq_factor,
// ext_factor). Reducing here does NOT commute with that scaling and corrupts
// every config with freq_scale != 1.0 or freq_factor != 1.0 (freq_factor is
// applied by the callers as theta_base/freq_factor before rope_yarn()).
static __device__ __forceinline__ double rope_theta_fp64(int32_t p, float theta_scale, int exp_int) {
    // Dim 0: theta_scale^0 == 1 exactly.
    if (exp_int == 0) {
        return (double)p;
    }
    // Binary exponentiation instead of pow(): the libcall dominated the whole
    // rope kernel on RDNA4 (692 us vs 76 us per launch at n_tokens=512). Seven
    // double multiplies keep the large-freq_base precision (the entire point
    // of the fp64 path) to within 1 ulp of pow().
    if (exp_int < 0) {
        // `e >>= 1` on a negative int is an arithmetic shift and never
        // reaches 0, so the loop below would hang the GPU. No current caller
        // passes a negative exponent, but pow() was total over the domain and
        // this helper takes a plain int, so stay total too.
        return (double)p * pow((double)theta_scale, (double)exp_int);
    }
    double base = (double)theta_scale;
    double r    = 1.0;
    int    e    = exp_int;
    while (e) {
        if (e & 1) { r *= base; }
        base *= base;
        e >>= 1;
    }
    return (double)p * r;
}

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const double theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    double theta_interp = (double)freq_scale * theta_extrap;
    double theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    // FP64 mod-2pi reduction, deferred to here (see rope_theta_fp64) so it runs
    // AFTER all angle scaling and stays equivalence-preserving. Kept in double
    // to preserve the large-freq_base precision benefit before the trig calls.
    const double TAU = 6.2831853071795864769;
    theta -= TAU * floor(theta * (1.0 / TAU));
    cos_theta = cosf((float)theta) * mscale;
    sin_theta = sinf((float)theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const int            rot_offset) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    // rot_offset is 0 for the standard head rotation and ne00 - n_dims for
    // GGML_ROPE_TYPE_TAIL; j0 is the element's index inside the rotated span,
    // so a tail element sees exactly the angle it would see in an extracted
    // n_dims-wide tail tensor.
    if (i0 < rot_offset || i0 >= rot_offset + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }
    const int j0 = i0 - rot_offset;

    const double theta_base = rope_theta_fp64(pos[i2], theta_scale, j0/2);

    const float freq_factor = has_ff ? freq_factors[j0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, j0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

// A forward NORMAL (optionally TAIL) rope without frequency factors, as the
// fused rope kernels below run it: its launch parameters, read from the ROPE
// node the way ggml_cuda_op_rope reads them, and one rotated pair with
// rope_norm's arithmetic.
struct rope_norm_fwd {
    int n_dims;
    int rot_offset;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float theta_scale;
    rope_corr_dims corr_dims;
};

static rope_norm_fwd rope_norm_fwd_of(const ggml_tensor * rope, const int ncols) {
    const int32_t * op = (const int32_t *) rope->op_params;
    float freq_base, beta_fast, beta_slow;
    rope_norm_fwd r;
    r.n_dims = op[1];
    const int n_ctx_orig = op[4];
    memcpy(&freq_base,     op +  5, sizeof(float));
    memcpy(&r.freq_scale,  op +  6, sizeof(float));
    memcpy(&r.ext_factor,  op +  7, sizeof(float));
    memcpy(&r.attn_factor, op +  8, sizeof(float));
    memcpy(&beta_fast,     op +  9, sizeof(float));
    memcpy(&beta_slow,     op + 10, sizeof(float));
    r.rot_offset = (op[2] & GGML_ROPE_TYPE_TAIL) ? ncols - r.n_dims : 0;
    ggml_rope_yarn_corr_dims(r.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, r.corr_dims.v);
    r.theta_scale = powf(freq_base, -2.0f / r.n_dims);
    return r;
}

static __device__ __forceinline__ void rope_norm_fwd_pair(
        const rope_norm_fwd & r, const int32_t p, const int i0, const float x0, const float x1,
        float & o0, float & o1) {
    if (i0 < r.rot_offset || i0 >= r.rot_offset + r.n_dims) {
        o0 = x0;
        o1 = x1;
        return;
    }
    const int j0 = i0 - r.rot_offset;
    const double theta_base = rope_theta_fp64(p, r.theta_scale, j0/2);
    float cos_theta;
    float sin_theta;
    rope_yarn<true>(theta_base/1.0f, r.freq_scale, r.corr_dims, j0, r.ext_factor, r.attn_factor, cos_theta, sin_theta);
    o0 = x0 * cos_theta - x1 * sin_theta;
    o1 = x0 * sin_theta + x1 * cos_theta;
}

// RMS_NORM * w, then a NORMAL (optionally TAIL) forward ROPE, then the F32 ->
// F16 copy of the result, one row per block: rms_norm_f32<block_size, mul>'s
// arithmetic, rope_norm's per pair (no frequency factors) and the cast's
// rounding, so the F32 rope output and its F16 copy are bit-identical to the
// three launches they replace. Rows are [ncols] with positions pos[row / ne01].
template <int block_size>
static __global__ void rms_norm_rope_f16_kernel(
        const float * __restrict__ x, const int64_t x_stride,
        const float * __restrict__ w,
        float * __restrict__ rope_dst, half * __restrict__ dst_f16,
        const int ncols, const float eps, const int ne01,
        const int32_t * __restrict__ pos, const rope_norm_fwd r) {
    const int row = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    x += (int64_t) row * x_stride;
    rope_dst += (int64_t) row * ncols;
    dst_f16 += (int64_t) row * ncols;

    extern __shared__ float s_normed[];
    __shared__ float s_sum[32];

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += block_size) {
        s_normed[col] = scale * x[col] * w[col];
    }
    __syncthreads();

    const int i2 = row / ne01;
    for (int p = tid; p < ncols / 2; p += block_size) {
        const int i0 = 2 * p;
        float o0;
        float o1;
        rope_norm_fwd_pair(r, pos[i2], i0, s_normed[i0 + 0], s_normed[i0 + 1], o0, o1);
        rope_dst[i0 + 0] = o0;
        rope_dst[i0 + 1] = o1;
        dst_f16[i0 + 0] = __float2half(o0);
        dst_f16[i0 + 1] = __float2half(o1);
    }
}

// A forward NORMAL/TAIL rope (rope_norm's arithmetic, no frequency factors)
// that also writes its output's q8_1 form in the layout MMVQ quantizes a
// grouped view [G = D * heads_per_group, T, n_groups] of it to: element
// (d, head, t) -> row (head / hpg) * T + t, column (head % hpg) * D + d. A
// half-warp holds one 32-wide block (two values per thread); amax and the
// sum follow quantize_q8_1's 32-lane butterfly exactly.
static __global__ void rope_norm_q8_kernel(
        const float * __restrict__ x, float * __restrict__ dst, const int D, const int H, const int T,
        const int64_t s01, const int64_t s02,
        const int32_t * __restrict__ pos, const rope_norm_fwd r,
        block_q8_1 * __restrict__ q8, const int hpg) {
    const int row = (int) blockIdx.x;            // head + H * token
    const int p = (int) (blockIdx.y * blockDim.x + threadIdx.x);
    const int i0 = 2 * p;
    if (i0 >= D) return;                          // whole half-warps: D % 32 == 0
    const int head = row % H;
    const int t = row / H;
    const float x0 = x[(int64_t) head * s01 + (int64_t) t * s02 + i0 + 0];
    const float x1 = x[(int64_t) head * s01 + (int64_t) t * s02 + i0 + 1];
    float o0;
    float o1;
    rope_norm_fwd_pair(r, pos[t], i0, x0, x1, o0, o1);
    dst[(int64_t) row * D + i0 + 0] = o0;
    dst[(int64_t) row * D + i0 + 1] = o1;

    // quantize_q8_1: lane v of the block holds element v; here thread v/2.
    float amax = fmaxf(fabsf(o0), fabsf(o1));
    float v0 = o0;
    float v1 = o1;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 16));
        v0 += __shfl_xor_sync(0xffffffff, v0, off, 16);
        v1 += __shfl_xor_sync(0xffffffff, v1, off, 16);
    }
    const float sum = v0 + v1;
    const float  d = amax / 127.0f;
    const int8_t q0 = amax == 0.0f ? 0 : roundf(o0 / d);
    const int8_t q1 = amax == 0.0f ? 0 : roundf(o1 / d);
    const int G = D * hpg;
    const int g = head / hpg;
    const int64_t col = (int64_t) (head % hpg) * D + i0;
    const int64_t qi = ((int64_t) g * T + t) * G + col;
    block_q8_1 * yb = q8 + qi / QK8_1;
    yb->qs[qi % QK8_1 + 0] = q0;
    yb->qs[qi % QK8_1 + 1] = q1;
    if (qi % QK8_1 == 0) {
        yb->ds = make_half2(d, sum);
    }
}

void ggml_cuda_rope_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rope, void * q8, int hpg) {
    const ggml_tensor * x = rope->src[0];
    const int D = (int) x->ne[0];
    const int H = (int) x->ne[1];
    const int T = (int) x->ne[2];
    const dim3 grid(H * T, (D / 2 + 255) / 256, 1);
    rope_norm_q8_kernel<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) x->data, (float *) rope->data, D, H, T,
        x->nb[1] / sizeof(float), x->nb[2] / sizeof(float),
        (const int32_t *) rope->src[1]->data, rope_norm_fwd_of(rope, D),
        (block_q8_1 *) q8, hpg);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_rms_norm_rope_f16_supported(int ncols) {
#if defined(GGML_USE_HIP)
    return ncols % 2 == 0 && (size_t) ncols * sizeof(float) <= 32u * 1024u;
#else
    // Bit-exact on HIP only: nvcc builds with fast math and contracts the
    // rotation's products differently here than in rope_norm, so CUDA keeps
    // the three launches.
    GGML_UNUSED(ncols);
    return false;
#endif
}

void ggml_cuda_rms_norm_rope_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
                                 const ggml_tensor * mul, const ggml_tensor * rope, const ggml_tensor * cpy) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    float eps = 0.0f;
    memcpy(&eps, norm->op_params, sizeof(float));
    const int ncols = (int) x->ne[0];
    const int nrows = (int) ggml_nrows(x);
    const ggml_tensor * rope_src = rope->src[0];
    const int ne01 = (int) rope_src->ne[1];

    const rope_norm_fwd r = rope_norm_fwd_of(rope, ncols);

    const size_t shmem = (size_t) ncols * sizeof(float);
    const int64_t x_stride = x->nb[1] / sizeof(float);
    if (ncols < 1024) {
        rms_norm_rope_f16_kernel<256><<<nrows, 256, shmem, ctx.stream()>>>(
            (const float *) x->data, x_stride, (const float *) w->data,
            (float *) rope->data, (half *) cpy->data, ncols, eps, ne01,
            (const int32_t *) rope->src[1]->data, r);
    } else {
        rms_norm_rope_f16_kernel<1024><<<nrows, 1024, shmem, ctx.stream()>>>(
            (const float *) x->data, x_stride, (const float *) w->data,
            (float *) rope->data, (half *) cpy->data, ncols, eps, ne01,
            (const int32_t *) rope->src[1]->data, r);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 >= n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const double theta_base = rope_theta_fp64(pos[i2], theta_scale, i0/2);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

// cos/sin of pair i0 at token i2, shared by rope_multi and rope_multi_heads so both run the same angle code.
template <bool forward, bool has_ff>
static __device__ __forceinline__ void rope_multi_cs(const int i0, const int32_t * pos, const uint32_t i2, const int ne02,
        const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
        const float theta_scale, const float * freq_factors, const mrope_sections sections, const bool is_imrope,
        float & cos_theta, float & sin_theta) {
    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    double theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = rope_theta_fp64(pos[i2 + ne02 * 1], theta_scale, i0/2);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = rope_theta_fp64(pos[i2 + ne02 * 2], theta_scale, i0/2);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = rope_theta_fp64(pos[i2], theta_scale, i0/2);
        } else {
            theta_base = rope_theta_fp64(pos[i2 + ne02 * 3], theta_scale, i0/2);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = rope_theta_fp64(pos[i2], theta_scale, i0/2);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = rope_theta_fp64(pos[i2 + ne02 * 1], theta_scale, i0/2);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = rope_theta_fp64(pos[i2 + ne02 * 2], theta_scale, i0/2);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = rope_theta_fp64(pos[i2 + ne02 * 3], theta_scale, i0/2);
        }
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi(const T *            x,
                                  T *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    if (i0 >= n_dims) {
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];

        return;
    }

    float cos_theta;
    float sin_theta;
    rope_multi_cs<forward, has_ff>(i0, pos, i2, ne02, freq_scale, ext_factor, attn_factor, corr_dims,
        theta_scale, freq_factors, sections, is_imrope, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

// rope_multi with one block per (token, sample) and one thread per rotation pair: the angle is computed once and
// applied to every head (rope_multi recomputes it per head in one 256-thread block per row, half of them idle).
// dst strides are free, so a following PERMUTE + CONT is written directly. Same per-element math.
template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi_heads(const T * x, T * dst, const int ne00, const int ne01, const int ne02,
        const int s01, const int s02, const int s03, const int s1, const int s2, const int s3, const int n_dims,
        const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors,
        const mrope_sections sections, const bool is_imrope) {
    const uint32_t i2 = blockIdx.x, i3 = blockIdx.y;
    for (int j = threadIdx.x; j < ne00 / 2; j += blockDim.x) {
        const int i0 = 2 * j;
        if (i0 >= n_dims) {
            for (int i1 = threadIdx.y; i1 < ne01; i1 += blockDim.y) {
                const int idst = i1 * s1 + i2 * s2 + i3 * s3, ix = i1 * s01 + i2 * s02 + i3 * s03;
                dst[idst + i0 + 0] = x[ix + i0 + 0];
                dst[idst + i0 + 1] = x[ix + i0 + 1];
            }
            continue;
        }
        float cos_theta;
        float sin_theta;
        rope_multi_cs<forward, has_ff>(i0, pos, i2, ne02, freq_scale, ext_factor, attn_factor, corr_dims,
            theta_scale, freq_factors, sections, is_imrope, cos_theta, sin_theta);
        for (int i1 = threadIdx.y; i1 < ne01; i1 += blockDim.y) {
            const int idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3, ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
            const float x0 = x[ix + 0];
            const float x1 = x[ix + n_dims/2];

            // Explicit form of the contraction the compiler picks for rope_multi's two lines (bit-identical).
            dst[idst + 0]        = fmaf(x0, cos_theta, -__fmul_rn(x1, sin_theta));
            dst[idst + n_dims/2] = fmaf(x1, cos_theta, __fmul_rn(x0, sin_theta));
        }
    }
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    double theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = rope_theta_fp64(pos[i2], theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = rope_theta_fp64(pos[i2 + ne02], theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const int            rot_offset,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, rot_offset);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, rot_offset);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_neox<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    } else {
        rope_neox<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    }
}

template <bool forward, typename T>
static void rope_multi_cuda(const T *            x,
                            T *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    // The per-token kernel pins its rotation to the FMA form HIP compiles rope_multi to (bit-identical on AMD); CUDA
    // builds keep rope_multi so NVIDIA results do not move by an ulp. Other models keep rope_multi.
#if defined(GGML_USE_HIP)
    const bool per_token = ggml_cuda_qwen4exp_enabled() && ne02 >= 64;
#else
    const bool per_token = false;
#endif
    if (per_token) {   // prefill: one block per token (decode keeps rope_multi's per-row blocks)
        const int bx = std::min(128, std::max(32, ne00 / 2)), ne03 = nr / (ne01 * ne02);
        const dim3 hb(bx, 256 / bx, 1), hg(ne02, ne03, 1);
        if (freq_factors == nullptr) {
            rope_multi_heads<forward, false, T><<<hg, hb, 0, stream>>>(x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
        } else {
            rope_multi_heads<forward, true, T><<<hg, hb, 0, stream>>>(x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
        }
    } else {
        if (freq_factors == nullptr) {
            rope_multi<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
                x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
                attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
        } else {
            rope_multi<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
                x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
                attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
        }
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr,
                            const ggml_tensor *         perm_cont = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    size_t s3 = dst->nb[3] / ggml_type_size(dst->type);
    if (perm_cont != nullptr) {   // ROPE -> PERMUTE -> CONT: write the rotated rows straight into the CONT layout
        const ggml_tensor * perm = perm_cont->src[0];
        const int32_t * ax = (const int32_t *) perm->op_params;
        GGML_ASSERT(ax[0] == 0 && perm_cont->type == dst->type);
        dst_d = perm_cont->data;
        s1 = perm_cont->nb[ax[1]] / ggml_type_size(dst->type);
        s2 = perm_cont->nb[ax[2]] / ggml_type_size(dst->type);
        s3 = perm_cont->nb[ax[3]] / ggml_type_size(dst->type);
    }

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    mrope_sections sections;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_tail = mode & GGML_ROPE_TYPE_TAIL;
    const int  mode_base = mode & ~GGML_ROPE_TYPE_TAIL;
    const bool is_neox = mode_base & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode_base & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode_base == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode_base == GGML_ROPE_TYPE_VISION;
    GGML_ASSERT(!is_tail || mode_base == GGML_ROPE_TYPE_NORMAL);
    const int rot_offset = is_tail ? (int) ne00 - n_dims : 0;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, rot_offset, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, rot_offset, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, rot_offset, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

void ggml_cuda_op_rope_permuted(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * cont) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, nullptr, cont);
}
