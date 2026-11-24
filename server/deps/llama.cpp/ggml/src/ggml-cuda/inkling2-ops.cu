// Fused CUDA/HIP kernels for Inkling2-Loop and its bundled DSpark speculator.
// All kernels launch asynchronously on ctx.stream() with deterministic grid
// shapes and pool allocations, making every op 100% HIP-graph / CUDA-graph safe.

#include "inkling2-ops.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

static constexpr float kInvSqrt128 = 0.08838834764831844055f;
static constexpr float kLog2E      = 1.44269504088896340736f;
static constexpr float kLn2        = 0.69314718055994530942f;

static __device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float round_bf16_f32(float x) {
    uint32_t bits = __float_as_uint(x);
    bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    return __uint_as_float(bits);
}

template <typename T>
static __device__ __forceinline__ T i2_shfl_xor(T v, int lane_mask, int width = WARP_SIZE) {
#if defined(GGML_USE_HIP)
    return __shfl_xor(v, lane_mask, width);
#else
    return __shfl_xor_sync(0xFFFFFFFFu, v, lane_mask, width);
#endif
}

static __device__ __forceinline__ unsigned long long i2_ballot(int pred) {
#if defined(GGML_USE_HIP)
    return (unsigned long long) __ballot(pred);
#else
    return (unsigned long long) __ballot_sync(0xFFFFFFFFu, pred);
#endif
}

// ─── 1. Fused Block Attention Residuals (+ optional RMSNorm) ──────────────
// Mixes states[0..n_blocks-1] and cur (each RMS-normalized), weighted by
// softmax(query @ norm(state) / sqrt(n_embd)). Optionally applies final RMSNorm.
struct AttnResPtrs {
    const float * ptr[8];
    size_t        row_stride_elems[8];
};

static __global__ void inkling2_attn_res_mix_kernel(
        AttnResPtrs states,
        int n_states,
        const float * __restrict__ query,   // [n_embd, n_q]
        const float * __restrict__ norm_w,  // nullptr, [n_embd], or [n_embd, n_q]
        int norm_per_q,
        float * __restrict__ dst,           // [n_embd, n_q, T]
        int n_embd,
        int n_q,
        float eps) {
    const int t = blockIdx.x;
    const int q = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int nwarps = nthreads / WARP_SIZE;

    const float inv_sqrt_d = rsqrtf((float) n_embd);

    // Fast vectorized float4 + register-cached path for Inkling 2 (n_embd=2304, n_states<=5, parallel across (T, n_q)):
    // Reads query ONCE into registers, reads each state ONCE, packs ALL ss & dots into ONE __syncthreads()!
    if ((n_embd & 3) == 0 && n_embd <= 3072 && n_states <= 5 && nwarps <= 8) {
        const int n4 = n_embd >> 2;
        __shared__ float s_buf[10][8]; // [2 * n_states][nwarps]
        __shared__ float s_inv_rms[5];
        __shared__ float s_weights[5];
        __shared__ float s_out_red[8];

        const float4 * q4_ptr = reinterpret_cast<const float4 *>(query + (size_t) q * n_embd);
        float4 qv[3];
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            const int e4 = tid + it * 256;
            qv[it] = (e4 < n4) ? q4_ptr[e4] : make_float4(0.f, 0.f, 0.f, 0.f);
        }

        float4 sv[5][3];
        #pragma unroll
        for (int s = 0; s < 5; ++s) {
            if (s < n_states) {
                const float4 * sp4 = reinterpret_cast<const float4 *>(states.ptr[s] + (size_t) t * states.row_stride_elems[s]);
                float ss = 0.0f;
                float dq = 0.0f;
                #pragma unroll
                for (int it = 0; it < 3; ++it) {
                    const int e4 = tid + it * 256;
                    if (e4 < n4) {
                        const float4 v = sp4[e4];
                        sv[s][it] = v;
                        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                        dq += v.x * qv[it].x + v.y * qv[it].y + v.z * qv[it].z + v.w * qv[it].w;
                    }
                }
                ss = warp_reduce_sum(ss);
                dq = warp_reduce_sum(dq);
                if (lane == 0) {
                    s_buf[s][warp] = ss;
                    s_buf[n_states + s][warp] = dq;
                }
            }
        }
        __syncthreads();

        if (warp == 0) {
            const int n_tot = 2 * n_states;
            float reduced = 0.0f;
            if (lane < n_tot) {
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    if (w < nwarps) reduced += s_buf[lane][w];
                }
                if (lane < n_states) {
                    s_inv_rms[lane] = rsqrtf(reduced / (float) n_embd + eps);
                }
            }
            if (lane >= n_states && lane < n_tot) {
                const int s = lane - n_states;
                s_weights[s] = reduced * s_inv_rms[s] * inv_sqrt_d;
            }
            if (lane == 0) {
                float mx = s_weights[0];
                for (int s = 1; s < n_states; ++s) mx = fmaxf(mx, s_weights[s]);
                float sm = 0.0f;
                for (int s = 0; s < n_states; ++s) {
                    float w = expf(s_weights[s] - mx);
                    s_weights[s] = w;
                    sm += w;
                }
                const float inv_sm = 1.0f / sm;
                for (int s = 0; s < n_states; ++s) {
                    s_weights[s] *= inv_sm * s_inv_rms[s];
                }
            }
        }
        __syncthreads();

        float4 * out4 = reinterpret_cast<float4 *>(dst + ((size_t) t * n_q + q) * n_embd);
        float4 acc[3];
        float ss = 0.0f;
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            const int e4 = tid + it * 256;
            if (e4 < n4) {
                float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
                for (int s = 0; s < n_states; ++s) {
                    const float w = s_weights[s];
                    a.x += w * sv[s][it].x;
                    a.y += w * sv[s][it].y;
                    a.z += w * sv[s][it].z;
                    a.w += w * sv[s][it].w;
                }
                acc[it] = a;
                if (norm_w) {
                    ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
                } else {
                    out4[e4] = a;
                }
            }
        }
        if (norm_w) {
            ss = warp_reduce_sum(ss);
            if (lane == 0) s_out_red[warp] = ss;
            __syncthreads();
            if (warp == 0) {
                float v = (lane < nwarps) ? s_out_red[lane] : 0.0f;
                v = warp_reduce_sum(v);
                if (lane == 0) s_out_red[0] = rsqrtf(v / (float) n_embd + eps);
            }
            __syncthreads();
            const float irms = s_out_red[0];
            const float4 * nw4 = reinterpret_cast<const float4 *>(norm_w + (norm_per_q ? (size_t) q * n_embd : 0));
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                const int e4 = tid + it * 256;
                if (e4 < n4) {
                    const float4 nw = nw4[e4];
                    out4[e4] = make_float4(
                        acc[it].x * irms * nw.x,
                        acc[it].y * irms * nw.y,
                        acc[it].z * irms * nw.z,
                        acc[it].w * irms * nw.w);
                }
            }
        }
        return;
    }

    // Generic fallback for arbitrary n_embd / n_states
    __shared__ float s_reduce[32];
    __shared__ float s_inv_rms[8];
    __shared__ float s_weights[8];

    const float * q_ptr = query + (size_t) q * n_embd;
    for (int s = 0; s < n_states; ++s) {
        const float * sp = states.ptr[s] + (size_t) t * states.row_stride_elems[s];
        float ss = 0.0f;
        float dq = 0.0f;
        for (int e = tid; e < n_embd; e += nthreads) {
            const float v = sp[e];
            ss += v * v;
            dq += v * q_ptr[e];
        }
        ss = warp_reduce_sum(ss);
        if (lane == 0) s_reduce[warp] = ss;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < nwarps) ? s_reduce[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) s_inv_rms[s] = rsqrtf(v / (float) n_embd + eps);
        }
        __syncthreads();
        const float irms = s_inv_rms[s];
        dq = warp_reduce_sum(dq);
        if (lane == 0) s_reduce[warp] = dq;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < nwarps) ? s_reduce[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) s_weights[s] = v * irms * inv_sqrt_d;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = s_weights[0];
        for (int s = 1; s < n_states; ++s) mx = fmaxf(mx, s_weights[s]);
        float sm = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            float w = expf(s_weights[s] - mx);
            s_weights[s] = w;
            sm += w;
        }
        const float inv_sm = 1.0f / sm;
        for (int s = 0; s < n_states; ++s) {
            s_weights[s] *= inv_sm * s_inv_rms[s];
        }
    }
    __syncthreads();

    float * out_row = dst + ((size_t) t * n_q + q) * n_embd;
    if (!norm_w) {
        for (int e = tid; e < n_embd; e += nthreads) {
            float acc = 0.0f;
            for (int s = 0; s < n_states; ++s) {
                const float * sp = states.ptr[s] + (size_t) t * states.row_stride_elems[s];
                acc += s_weights[s] * sp[e];
            }
            out_row[e] = acc;
        }
    } else {
        float ss = 0.0f;
        for (int e = tid; e < n_embd; e += nthreads) {
            float acc = 0.0f;
            for (int s = 0; s < n_states; ++s) {
                const float * sp = states.ptr[s] + (size_t) t * states.row_stride_elems[s];
                acc += s_weights[s] * sp[e];
            }
            out_row[e] = acc;
            ss += acc * acc;
        }
        ss = warp_reduce_sum(ss);
        if (lane == 0) s_reduce[warp] = ss;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < nwarps) ? s_reduce[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) s_reduce[0] = rsqrtf(v / (float) n_embd + eps);
        }
        __syncthreads();
        const float irms = s_reduce[0];
        const float * nw = norm_w + (norm_per_q ? (size_t) q * n_embd : 0);
        for (int e = tid; e < n_embd; e += nthreads) {
            out_row[e] = out_row[e] * irms * nw[e];
        }
    }
}

// ─── 2. Fused Loop Controller Pre: [rms_norm(cur) | rms_norm(msg)] (3D batched!) ─
static __global__ void inkling2_loop_ctrl_pre_kernel(
        const float * __restrict__ cur, size_t cur_row_stride, size_t cur_grp_stride,
        const float * __restrict__ msg, size_t msg_row_stride, size_t msg_grp_stride,
        float * __restrict__ dst, size_t dst_row_stride, size_t dst_grp_stride,
        int n_embd, float eps) {
    const int t = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int nwarps = nthreads / WARP_SIZE;

    const float * cp = cur + (size_t) t * cur_row_stride + (size_t) g * cur_grp_stride;
    const float * mp = msg + (size_t) t * msg_row_stride + (size_t) g * msg_grp_stride;
    float * op = dst + (size_t) t * dst_row_stride + (size_t) g * dst_grp_stride;

    __shared__ float s_red[2][8];

    if ((n_embd & 3) == 0 && n_embd <= 3072 && nwarps <= 8) {
        const int n4 = n_embd >> 2;
        const float4 * cp4 = reinterpret_cast<const float4 *>(cp);
        const float4 * mp4 = reinterpret_cast<const float4 *>(mp);
        float4 * op4_c = reinterpret_cast<float4 *>(op);
        float4 * op4_m = reinterpret_cast<float4 *>(op + n_embd);

        float4 vc[3], vm[3];
        float ss_c = 0.0f, ss_m = 0.0f;
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            const int e4 = tid + it * 256;
            if (e4 < n4) {
                const float4 c4 = cp4[e4];
                const float4 m4 = mp4[e4];
                vc[it] = c4;
                vm[it] = m4;
                ss_c += c4.x * c4.x + c4.y * c4.y + c4.z * c4.z + c4.w * c4.w;
                ss_m += m4.x * m4.x + m4.y * m4.y + m4.z * m4.z + m4.w * m4.w;
            }
        }
        ss_c = warp_reduce_sum(ss_c);
        ss_m = warp_reduce_sum(ss_m);
        if (lane == 0) { s_red[0][warp] = ss_c; s_red[1][warp] = ss_m; }
        __syncthreads();
        if (warp == 0) {
            float v0 = (lane < nwarps) ? s_red[0][lane] : 0.0f;
            float v1 = (lane < nwarps) ? s_red[1][lane] : 0.0f;
            v0 = warp_reduce_sum(v0);
            v1 = warp_reduce_sum(v1);
            if (lane == 0) {
                s_red[0][0] = rsqrtf(v0 / (float) n_embd + eps);
                s_red[1][0] = rsqrtf(v1 / (float) n_embd + eps);
            }
        }
        __syncthreads();
        const float irms_c = s_red[0][0];
        const float irms_m = s_red[1][0];
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            const int e4 = tid + it * 256;
            if (e4 < n4) {
                op4_c[e4] = make_float4(vc[it].x * irms_c, vc[it].y * irms_c, vc[it].z * irms_c, vc[it].w * irms_c);
                op4_m[e4] = make_float4(vm[it].x * irms_m, vm[it].y * irms_m, vm[it].z * irms_m, vm[it].w * irms_m);
            }
        }
        return;
    }

    float ss_c = 0.0f, ss_m = 0.0f;
    for (int e = tid; e < n_embd; e += nthreads) {
        const float vc = cp[e];
        const float vm = mp[e];
        ss_c += vc * vc;
        ss_m += vm * vm;
    }
    ss_c = warp_reduce_sum(ss_c);
    ss_m = warp_reduce_sum(ss_m);
    if (lane == 0) { s_red[0][warp] = ss_c; s_red[1][warp] = ss_m; }
    __syncthreads();
    if (warp == 0) {
        float v0 = (lane < nwarps) ? s_red[0][lane] : 0.0f;
        float v1 = (lane < nwarps) ? s_red[1][lane] : 0.0f;
        v0 = warp_reduce_sum(v0);
        v1 = warp_reduce_sum(v1);
        if (lane == 0) {
            s_red[0][0] = rsqrtf(v0 / (float) n_embd + eps);
            s_red[1][0] = rsqrtf(v1 / (float) n_embd + eps);
        }
    }
    __syncthreads();
    const float irms_c = s_red[0][0];
    const float irms_m = s_red[1][0];
    for (int e = tid; e < n_embd; e += nthreads) {
        op[e]          = cp[e] * irms_c;
        op[n_embd + e] = mp[e] * irms_m;
    }
}

// ─── 3. Fused Loop Controller Post: (1-tanh(f))*cur + tanh(g)*msg (+RMSNorm, 3D batched!)
// Optional flags [T]: when flags != nullptr && (flags[t] & 1) == 0, skips message blend
// (pure RMSNorm on cur) so pass-0 and pass>0 rows run in ONE fused Core graph!
static __global__ void inkling2_loop_ctrl_post_kernel(
        const float   * __restrict__ cur, size_t cur_row_stride, size_t cur_grp_stride,
        const float   * __restrict__ msg, size_t msg_row_stride, size_t msg_grp_stride,
        const float   * __restrict__ gf,  size_t gf_row_stride,  size_t gf_grp_stride,
        const float   * __restrict__ norm_w, size_t nw_grp_stride,
        const int32_t * __restrict__ flags,
        float         * __restrict__ dst, size_t dst_row_stride, size_t dst_grp_stride,
        int n_embd, float eps) {
    const int t = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int nwarps = nthreads / WARP_SIZE;
    const bool do_blend = (flags == nullptr) || ((flags[t] & 1) != 0);

    __shared__ float s_red[8];
    const float * cp = cur + (size_t) t * cur_row_stride + (size_t) g * cur_grp_stride;
    const float * mp = msg + (size_t) t * msg_row_stride + (size_t) g * msg_grp_stride;
    const float * gp = gf  + (size_t) t * gf_row_stride  + (size_t) g * gf_grp_stride;
    const float * fp = gp + n_embd;
    const float * nw = norm_w ? (norm_w + (size_t) g * nw_grp_stride) : nullptr;
    float * op = dst + (size_t) t * dst_row_stride + (size_t) g * dst_grp_stride;

    if ((n_embd & 3) == 0 && n_embd <= 3072 && nwarps <= 8) {
        const int n4 = n_embd >> 2;
        const float4 * cp4 = reinterpret_cast<const float4 *>(cp);
        const float4 * mp4 = reinterpret_cast<const float4 *>(mp);
        const float4 * gp4 = reinterpret_cast<const float4 *>(gp);
        const float4 * fp4 = reinterpret_cast<const float4 *>(fp);
        float4 * op4 = reinterpret_cast<float4 *>(op);

        float4 vy[3];
        float ss = 0.0f;
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            const int e4 = tid + it * 256;
            if (e4 < n4) {
                const float4 c4 = cp4[e4];
                float4 y4 = c4;
                if (do_blend) {
                    const float4 m4 = mp4[e4];
                    const float4 g4 = gp4[e4];
                    const float4 f4 = fp4[e4];
                    y4.x = (c4.x - tanhf(f4.x) * c4.x) + tanhf(g4.x) * m4.x;
                    y4.y = (c4.y - tanhf(f4.y) * c4.y) + tanhf(g4.y) * m4.y;
                    y4.z = (c4.z - tanhf(f4.z) * c4.z) + tanhf(g4.z) * m4.z;
                    y4.w = (c4.w - tanhf(f4.w) * c4.w) + tanhf(g4.w) * m4.w;
                }
                vy[it] = y4;
                if (nw) {
                    ss += y4.x * y4.x + y4.y * y4.y + y4.z * y4.z + y4.w * y4.w;
                } else {
                    op4[e4] = y4;
                }
            }
        }
        if (nw) {
            ss = warp_reduce_sum(ss);
            if (lane == 0) s_red[warp] = ss;
            __syncthreads();
            if (warp == 0) {
                float v = (lane < nwarps) ? s_red[lane] : 0.0f;
                v = warp_reduce_sum(v);
                if (lane == 0) s_red[0] = rsqrtf(v / (float) n_embd + eps);
            }
            __syncthreads();
            const float irms = s_red[0];
            const float4 * nw4 = reinterpret_cast<const float4 *>(nw);
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                const int e4 = tid + it * 256;
                if (e4 < n4) {
                    const float4 w4 = nw4[e4];
                    op4[e4] = make_float4(
                        vy[it].x * irms * w4.x,
                        vy[it].y * irms * w4.y,
                        vy[it].z * irms * w4.z,
                        vy[it].w * irms * w4.w);
                }
            }
        }
        return;
    }

    float ss = 0.0f;
    for (int e = tid; e < n_embd; e += nthreads) {
        const float c = cp[e];
        float y = c;
        if (do_blend) {
            const float m = mp[e];
            const float gv = tanhf(gp[e]);
            const float fv = tanhf(fp[e]);
            y = (c - fv * c) + gv * m;
        }
        op[e] = y;
        ss += y * y;
    }
    if (nw) {
        ss = warp_reduce_sum(ss);
        if (lane == 0) s_red[warp] = ss;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < nwarps) ? s_red[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) s_red[0] = rsqrtf(v / (float) n_embd + eps);
        }
        __syncthreads();
        const float irms = s_red[0];
        for (int e = tid; e < n_embd; e += nthreads) {
            op[e] = op[e] * irms * nw[e];
        }
    }
}

// ─── 4. Fused Causal Short Conv (+ optional dynamic taps + optional residual addend!) ─
static __global__ void inkling2_sconv_kernel(
        const float * __restrict__ x, size_t x_stride,
        const float * __restrict__ w_static, // [4, n_ch]
        const float * __restrict__ state_in,
        float * __restrict__ state_out,
        const float * __restrict__ dyn_taps, size_t dyn_stride, // nullptr or [4*H_kv, T]
        const float * __restrict__ addend,   size_t add_stride, // nullptr or [n_ch, T]
        float * __restrict__ dst,
        int n_ch,
        int n_tokens,
        int state_stride,
        int state_chan_offset,
        int head_dim_dyn,
        int write_mode) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;

    const int sc = state_chan_offset + c;
    const float * st_in_c = state_in + (size_t) sc * 3;
    float h0 = st_in_c[0];
    float h1 = st_in_c[1];
    float h2 = st_in_c[2];

    const float * wc = w_static + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float scale = dyn_taps ? (1.0f / 3.0f) : rsqrtf(5.0f);
    const int head = (dyn_taps && head_dim_dyn > 0) ? (c / head_dim_dyn) : 0;

    for (int t = 0; t < n_tokens; ++t) {
        const float cur = x[(size_t) t * x_stride + c];
        float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
        if (dyn_taps) {
            const float * dp = dyn_taps + (size_t) t * dyn_stride + head * 4;
            // lag 0 = cur, lag 1 = h2, lag 2 = h1, lag 3 = h0
            acc += dp[0] * cur + dp[1] * h2 + dp[2] * h1 + dp[3] * h0;
        }
        float out_v = (cur + acc) * scale;
        if (addend) {
            out_v += addend[(size_t) t * add_stride + c];
        }
        dst[(size_t) t * n_ch + c] = out_v;
        h0 = h1;
        h1 = h2;
        h2 = cur;
        if (write_mode == 2 && state_out) {
            float * st_ring = state_out + ((size_t) t * state_stride + sc) * 3;
            st_ring[0] = h0;
            st_ring[1] = h1;
            st_ring[2] = h2;
        }
    }

    if (write_mode == 1 && state_out) {
        float * st_out_c = state_out + (size_t) sc * 3;
        st_out_c[0] = h0;
        st_out_c[1] = h1;
        st_out_c[2] = h2;
    }
}

// Parallel O(1) time-tiled ShortConv for prefill (T > 8, write_mode <= 1):
// ShortConv has a finite 4-tap receptive field (t-3..t), so 16-token tiles run 100% in parallel across all 64 CUs!
static __global__ void inkling2_sconv_parallel_kernel(
        const float * __restrict__ x, size_t x_stride,
        const float * __restrict__ w_static, // [4, n_ch]
        const float * __restrict__ state_in,
        float * __restrict__ state_out,
        const float * __restrict__ dyn_taps, size_t dyn_stride,
        const float * __restrict__ addend,   size_t add_stride,
        float * __restrict__ dst,
        int n_ch,
        int n_tokens,
        int state_chan_offset,
        int head_dim_dyn,
        int write_mode) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;

    const int t0 = blockIdx.y * 16;
    const int t1 = min(n_tokens, t0 + 16);
    const int sc = state_chan_offset + c;
    const float * st_in_c = state_in + (size_t) sc * 3;

    float h0 = (t0 >= 3) ? x[(size_t) (t0 - 3) * x_stride + c] : st_in_c[t0];
    float h1 = (t0 >= 2) ? x[(size_t) (t0 - 2) * x_stride + c] : st_in_c[t0 + 1];
    float h2 = (t0 >= 1) ? x[(size_t) (t0 - 1) * x_stride + c] : st_in_c[t0 + 2];

    const float * wc = w_static + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float scale = dyn_taps ? (1.0f / 3.0f) : rsqrtf(5.0f);
    const int head = (dyn_taps && head_dim_dyn > 0) ? (c / head_dim_dyn) : 0;

    for (int t = t0; t < t1; ++t) {
        const float cur = x[(size_t) t * x_stride + c];
        float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
        if (dyn_taps) {
            const float * dp = dyn_taps + (size_t) t * dyn_stride + head * 4;
            acc += dp[0] * cur + dp[1] * h2 + dp[2] * h1 + dp[3] * h0;
        }
        float out_v = (cur + acc) * scale;
        if (addend) {
            out_v += addend[(size_t) t * add_stride + c];
        }
        dst[(size_t) t * n_ch + c] = out_v;
        h0 = h1;
        h1 = h2;
        h2 = cur;
    }
    (void) state_out;
    (void) write_mode;
}

// Race-free post-tile state commit for parallel prefill (runs after all tiles finish reading state_in!).
static __global__ void inkling2_sconv_commit_state_kernel(
        const float * __restrict__ x, size_t x_stride,
        float * __restrict__ state_out,
        int n_ch, int n_tokens, int state_chan_offset) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    float * so = state_out + (size_t) (state_chan_offset + c) * 3;
    so[0] = x[(size_t) (n_tokens - 3) * x_stride + c];
    so[1] = x[(size_t) (n_tokens - 2) * x_stride + c];
    so[2] = x[(size_t) (n_tokens - 1) * x_stride + c];
}

// ─── 5. Fused Q/R Clip (+ optional scale, optional log-tau) / V RMSNorm ──
static __global__ void inkling2_clip_norm_kernel(
        const float * __restrict__ x, size_t x_stride,
        const float * __restrict__ scale_w, // nullptr or [head_dim, n_head]
        const int32_t * __restrict__ pos_buf, // nullptr or [T]
        float * __restrict__ dst,
        int head_dim, int n_head,
        int mode, float threshold, float eps,
        int log_floor, float log_alpha) {
    // 1 warp per (head, token)
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;

    const float * xp = x + (size_t) t * x_stride + (size_t) h * head_dim;
    const float * sp = scale_w ? (scale_w + (size_t) h * head_dim) : nullptr;
    float * op = dst + ((size_t) t * n_head + h) * head_dim;

    float tau = 1.0f;
    if (pos_buf && log_floor > 0 && log_alpha > 0.0f) {
        const float ratio = (float)(pos_buf[t] + 1) / (float) log_floor;
        if (ratio > 1.0f) {
            tau = 1.0f + log_alpha * logf(ratio);
        }
    }

    float ss = 0.0f;
    float vals[4];
    int cnt = 0;
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        float v = xp[d];
        if (sp) v *= sp[d];
        vals[cnt++] = v;
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);

    float mult = 1.0f;
    if (mode == 0) {
        const float nrm = sqrtf(ss) + eps;
        mult = fminf(1.0f, threshold / nrm) * tau;
    } else {
        mult = rsqrtf(ss / (float) head_dim + eps);
    }

    cnt = 0;
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        op[d] = vals[cnt++] * mult;
    }
}

// ─── 6. Fused K-clip / V-rmsnorm + Direct Write into KV Cache (F16/Q8_0) ─
// kv_cache layout: [head_dim, kv_cap, n_head_kv]
static __global__ void inkling2_kv_write_f16_kernel(
        const float * __restrict__ x, size_t x_stride,
        half * __restrict__ kv_cache, size_t kv_nb1_elems, size_t kv_nb2_elems,
        const int32_t * __restrict__ pos_buf,
        int head_dim, int n_head_kv, int kv_cap,
        int is_v, int swa_window, float threshold, float eps) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;

    const int32_t pos = pos_buf[t];
    if (pos < 0) return;
    const int slot = (swa_window > 0) ? (pos % kv_cap)
                   : (swa_window < 0) ? (pos + swa_window)
                   : pos;
    if (slot < 0 || slot >= kv_cap) return;

    const float * xp = x + (size_t) t * x_stride + (size_t) h * head_dim;
    half * op = kv_cache + (size_t) h * kv_nb2_elems + (size_t) slot * kv_nb1_elems;

    float ss = 0.0f;
    float vals[4];
    int cnt = 0;
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        const float v = xp[d];
        vals[cnt++] = v;
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);

    const float mult = is_v
        ? rsqrtf(ss / (float) head_dim + eps)
        : fminf(1.0f, threshold / (sqrtf(ss) + eps));

    cnt = 0;
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        op[d] = __float2half(vals[cnt++] * mult);
    }
}

static __global__ void inkling2_kv_write_q8_0_kernel(
        const float * __restrict__ x, size_t x_stride,
        char * __restrict__ kv_cache, size_t kv_nb1_bytes, size_t kv_nb2_bytes,
        const int32_t * __restrict__ pos_buf,
        int head_dim, int n_head_kv, int kv_cap,
        int is_v, int swa_window, float threshold, float eps) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x; // 0..31 -> 1 element per Q8_0 block (QK8_0 == 32)

    const int32_t pos = pos_buf[t];
    if (pos < 0) return;
    const int slot = (swa_window > 0) ? (pos % kv_cap)
                   : (swa_window < 0) ? (pos + swa_window)
                   : pos;
    if (slot < 0 || slot >= kv_cap) return;

    const float * xp = x + (size_t) t * x_stride + (size_t) h * head_dim;
    block_q8_0 * bp = reinterpret_cast<block_q8_0 *>(
        kv_cache + (size_t) h * kv_nb2_bytes + (size_t) slot * kv_nb1_bytes);

    const int n_blocks = head_dim / QK8_0; // 4 for D=128
    float ss = 0.0f;
    float vals[4];
    for (int b = 0; b < n_blocks; ++b) {
        const float v = xp[b * QK8_0 + lane];
        vals[b] = v;
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);

    const float mult = is_v
        ? rsqrtf(ss / (float) head_dim + eps)
        : fminf(1.0f, threshold / (sqrtf(ss) + eps));

    for (int b = 0; b < n_blocks; ++b) {
        const float v = vals[b] * mult;
        float amax = warp_reduce_max(fabsf(v));
        const float d = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t) __float2int_rn(v / d);
        bp[b].qs[lane] = q;
        if (lane == 0) bp[b].d = __float2half(d);
    }
}

// ─── 7. Fused Relative-Position Table: [rel_extent+1, n_head, T] ────────
static __global__ void inkling2_rel_table_kernel(
        const float * __restrict__ rel_bias, // [d_rel, rel_extent]
        const float * __restrict__ r_cur,    // [d_rel, n_head, T]
        const float * __restrict__ sink, size_t sink_stride, // [n_head, T]
        float * __restrict__ dst,            // [rel_extent+1, n_head, T]
        int d_rel, int rel_extent, int n_head) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ float s_r[32];
    if (tid < d_rel) {
        s_r[tid] = r_cur[((size_t) t * n_head + h) * d_rel + tid];
    }
    __syncthreads();

    const float s_val = sink[(size_t) t * sink_stride + h];
    float * op = dst + ((size_t) t * n_head + h) * (rel_extent + 1);

    for (int dist = tid; dist < rel_extent; dist += nthreads) {
        const float * rb = rel_bias + (size_t) dist * d_rel;
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            if (k < d_rel) dot += rb[k] * s_r[k];
        }
        op[dist] = dot - s_val;
    }
    if (tid == 0) {
        op[rel_extent] = -s_val;
    }
}

// ─── 8a. 1-Pass Single-Kernel SWA Attention (Zero Scratch, Zero Reduce Launch!) ──
template <bool KV_Q8_0, int GQA, int NWARPS>
static __global__ void inkling2_attn_rel_single_kernel(
        const float * __restrict__ q_ptr,     // [128, Hq, T]
        const char  * __restrict__ k_ptr,     // [128, kv_cap, Hk]
        const char  * __restrict__ v_ptr,     // [128, kv_cap, Hk]
        size_t k_nb1, size_t k_nb2,
        size_t v_nb1, size_t v_nb2,
        const float * __restrict__ rel_table, // [rel_extent+1, Hq, T]
        const int32_t * __restrict__ kv_meta, // {pos0, kv_len, kv_start, kv_end}
        const float * __restrict__ gate_ptr, size_t gate_stride,
        float * __restrict__ dst,             // [128*Hq, T] or [(128+2)*Hq, T]
        int Hq, int Hk, int kv_cap,
        int swa_window, int rel_extent,
        int partial_out, int include_sink) {
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..NWARPS-1
    const int kv_head = blockIdx.x;
    const int q_tok   = blockIdx.y;

    const int pos0     = kv_meta[0];
    const int kv_len   = kv_meta[1];
    const int kv_start = kv_meta[2];
    const int kv_end_r = kv_meta[3] > 0 ? kv_meta[3] : kv_len;

    const int q_pos     = pos0 + q_tok;
    const int eff_end   = min(kv_end_r, q_pos + 1);
    const int eff_start = (swa_window > 0) ? max(kv_start, q_pos - swa_window + 1) : kv_start;

    float q_reg[GQA][4];
    const float * rt_base[GQA];
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        const int h = kv_head * GQA + g;
        const float * qp = q_ptr + ((size_t) q_tok * Hq + h) * 128;
        q_reg[g][0] = qp[lane +  0] * kInvSqrt128;
        q_reg[g][1] = qp[lane + 32] * kInvSqrt128;
        q_reg[g][2] = qp[lane + 64] * kInvSqrt128;
        q_reg[g][3] = qp[lane + 96] * kInvSqrt128;
        rt_base[g] = rel_table + ((size_t) q_tok * Hq + h) * (rel_extent + 1);
    }

    float m_max[GQA];
    float l_sum[GQA];
    float o_acc[GQA][4];
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        if (include_sink && warp == 0) {
            m_max[g] = 0.0f;
            l_sum[g] = 1.0f;
        } else {
            m_max[g] = -INFINITY;
            l_sum[g] = 0.0f;
        }
        o_acc[g][0] = 0.0f; o_acc[g][1] = 0.0f; o_acc[g][2] = 0.0f; o_acc[g][3] = 0.0f;
    }

    const char * k_head = k_ptr + (size_t) kv_head * k_nb2;
    const char * v_head = v_ptr + (size_t) kv_head * v_nb2;

    for (int kp = eff_start + warp; kp < eff_end; kp += NWARPS) {
        const int slot = (swa_window > 0) ? (kp % kv_cap) : (kp - kv_start);
        float kv[4], vv[4];
        if constexpr (KV_Q8_0) {
            const block_q8_0 * kb = reinterpret_cast<const block_q8_0 *>(k_head + (size_t) slot * k_nb1);
            const block_q8_0 * vb = reinterpret_cast<const block_q8_0 *>(v_head + (size_t) slot * v_nb1);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                kv[b] = __half2float(kb[b].d) * (float) kb[b].qs[lane];
                vv[b] = __half2float(vb[b].d) * (float) vb[b].qs[lane];
            }
        } else {
            const half * kh = reinterpret_cast<const half *>(k_head + (size_t) slot * k_nb1);
            const half * vh = reinterpret_cast<const half *>(v_head + (size_t) slot * v_nb1);
            kv[0] = __half2float(kh[lane +  0]); kv[1] = __half2float(kh[lane + 32]);
            kv[2] = __half2float(kh[lane + 64]); kv[3] = __half2float(kh[lane + 96]);
            vv[0] = __half2float(vh[lane +  0]); vv[1] = __half2float(vh[lane + 32]);
            vv[2] = __half2float(vh[lane + 64]); vv[3] = __half2float(vh[lane + 96]);
        }
        const int dist = min(q_pos - kp, rel_extent);

        #pragma unroll
        for (int g = 0; g < GQA; ++g) {
            float dot = q_reg[g][0] * kv[0] + q_reg[g][1] * kv[1]
                      + q_reg[g][2] * kv[2] + q_reg[g][3] * kv[3];
            dot = warp_reduce_sum(dot);
            const float score = dot + rt_base[g][dist];
            const float m_new = fmaxf(m_max[g], score);
            const float alpha = isinf(m_max[g]) ? 0.0f : expf(m_max[g] - m_new);
            const float p     = expf(score - m_new);
            l_sum[g] = l_sum[g] * alpha + p;
            o_acc[g][0] = o_acc[g][0] * alpha + p * vv[0];
            o_acc[g][1] = o_acc[g][1] * alpha + p * vv[1];
            o_acc[g][2] = o_acc[g][2] * alpha + p * vv[2];
            o_acc[g][3] = o_acc[g][3] * alpha + p * vv[3];
            m_max[g] = m_new;
        }
    }

    __shared__ float s_o[NWARPS][GQA][128];
    __shared__ float s_m[NWARPS][GQA];
    __shared__ float s_l[NWARPS][GQA];

    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        s_o[warp][g][lane +  0] = o_acc[g][0];
        s_o[warp][g][lane + 32] = o_acc[g][1];
        s_o[warp][g][lane + 64] = o_acc[g][2];
        s_o[warp][g][lane + 96] = o_acc[g][3];
        if (lane == 0) {
            s_m[warp][g] = m_max[g];
            s_l[warp][g] = l_sum[g];
        }
    }
    __syncthreads();

    if (warp < GQA) {
        const int g = warp;
        float mx = s_m[0][g];
        #pragma unroll
        for (int w = 1; w < NWARPS; ++w) mx = fmaxf(mx, s_m[w][g]);
        float ltot = 0.0f;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
        if (!isinf(mx)) {
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                if (s_l[w][g] > 0.0f) {
                    const float sc = expf(s_m[w][g] - mx);
                    ltot += s_l[w][g] * sc;
                    o0 += s_o[w][g][lane +  0] * sc;
                    o1 += s_o[w][g][lane + 32] * sc;
                    o2 += s_o[w][g][lane + 64] * sc;
                    o3 += s_o[w][g][lane + 96] * sc;
                }
            }
        }
        const int h = kv_head * GQA + g;
        if (partial_out) {
            float * out = dst + ((size_t) q_tok * Hq + h) * 130;
            out[lane +  0] = o0; out[lane + 32] = o1; out[lane + 64] = o2; out[lane + 96] = o3;
            if (lane == 0) { out[128] = mx; out[129] = ltot; }
        } else {
            const float inv_l = ltot > 0.0f ? (1.0f / ltot) : 0.0f;
            float r0 = o0 * inv_l, r1 = o1 * inv_l, r2 = o2 * inv_l, r3 = o3 * inv_l;
            if (gate_ptr) {
                const float * gp = gate_ptr + (size_t) q_tok * gate_stride + (size_t) h * 128;
                r0 *= sigmoid_f32(gp[lane +  0]);
                r1 *= sigmoid_f32(gp[lane + 32]);
                r2 *= sigmoid_f32(gp[lane + 64]);
                r3 *= sigmoid_f32(gp[lane + 96]);
            }
            float * out = dst + ((size_t) q_tok * Hq + h) * 128;
            out[lane +  0] = r0; out[lane + 32] = r1; out[lane + 64] = r2; out[lane + 96] = r3;
        }
    }
}

// ─── 8b. Split-KV Inline Relative-Bias + Zero-Sink Attention (Global Layers) ──
template <bool KV_Q8_0, int GQA>
static __global__ void inkling2_attn_rel_split_kernel(
        const float * __restrict__ q_ptr,     // [128, Hq, T]
        const char  * __restrict__ k_ptr,     // [128, kv_cap, Hk]
        const char  * __restrict__ v_ptr,     // [128, kv_cap, Hk]
        size_t k_nb1, size_t k_nb2,
        size_t v_nb1, size_t v_nb2,
        const float * __restrict__ rel_table, // [rel_extent+1, Hq, T]
        const int32_t * __restrict__ kv_meta, // {pos0, kv_len, kv_start, kv_end}
        float * __restrict__ part_out,        // [(128+2), Hq, T, num_splits]
        int Hq, int Hk, int kv_cap,
        int swa_window, int rel_extent,
        int num_splits, int include_sink) {
    constexpr int NWARPS = 4;
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..3
    const int kv_head = blockIdx.x;
    const int q_tok   = blockIdx.y;
    const int split   = blockIdx.z;

    const int pos0     = kv_meta[0];
    const int kv_len   = kv_meta[1];
    const int kv_start = kv_meta[2];
    const int kv_end_r = kv_meta[3] > 0 ? kv_meta[3] : kv_len;

    const int q_pos = pos0 + q_tok;
    const int eff_end = min(kv_end_r, q_pos + 1);
    const int eff_start = (swa_window > 0) ? max(kv_start, q_pos - swa_window + 1) : kv_start;
    const int total_keys = max(0, eff_end - eff_start);

    const int keys_per_split = (total_keys + num_splits - 1) / num_splits;
    const int s_begin = eff_start + split * keys_per_split;
    const int s_end   = min(eff_end, s_begin + keys_per_split);

    // Load GQA query heads into registers (scaled by 1/sqrt(128))
    float q_reg[GQA][4];
    const float * rt_base[GQA];
    for (int g = 0; g < GQA; ++g) {
        const int h = kv_head * GQA + g;
        const float * qp = q_ptr + ((size_t) q_tok * Hq + h) * 128;
        q_reg[g][0] = qp[lane +  0] * kInvSqrt128;
        q_reg[g][1] = qp[lane + 32] * kInvSqrt128;
        q_reg[g][2] = qp[lane + 64] * kInvSqrt128;
        q_reg[g][3] = qp[lane + 96] * kInvSqrt128;
        rt_base[g] = rel_table + ((size_t) q_tok * Hq + h) * (rel_extent + 1);
    }

    float m_max[GQA];
    float l_sum[GQA];
    float o_acc[GQA][4];
    for (int g = 0; g < GQA; ++g) {
        // Zero-sink virtual key lives only in split 0, warp 0 when include_sink != 0
        if (include_sink && split == 0 && warp == 0) {
            m_max[g] = 0.0f;
            l_sum[g] = 1.0f;
        } else {
            m_max[g] = -INFINITY;
            l_sum[g] = 0.0f;
        }
        o_acc[g][0] = 0.0f;
        o_acc[g][1] = 0.0f;
        o_acc[g][2] = 0.0f;
        o_acc[g][3] = 0.0f;
    }

    const char * k_head = k_ptr + (size_t) kv_head * k_nb2;
    const char * v_head = v_ptr + (size_t) kv_head * v_nb2;

    for (int kp = s_begin + warp; kp < s_end; kp += NWARPS) {
        const int slot = (swa_window > 0) ? (kp % kv_cap) : (kp - kv_start);
        float kv[4];
        if constexpr (KV_Q8_0) {
            const block_q8_0 * kb = reinterpret_cast<const block_q8_0 *>(k_head + (size_t) slot * k_nb1);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                kv[b] = __half2float(kb[b].d) * (float) kb[b].qs[lane];
            }
        } else {
            const half * kh = reinterpret_cast<const half *>(k_head + (size_t) slot * k_nb1);
            kv[0] = __half2float(kh[lane +  0]);
            kv[1] = __half2float(kh[lane + 32]);
            kv[2] = __half2float(kh[lane + 64]);
            kv[3] = __half2float(kh[lane + 96]);
        }

        const int dist = min(q_pos - kp, rel_extent);

        float vv[4];
        if constexpr (KV_Q8_0) {
            const block_q8_0 * vb = reinterpret_cast<const block_q8_0 *>(v_head + (size_t) slot * v_nb1);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                vv[b] = __half2float(vb[b].d) * (float) vb[b].qs[lane];
            }
        } else {
            const half * vh = reinterpret_cast<const half *>(v_head + (size_t) slot * v_nb1);
            vv[0] = __half2float(vh[lane +  0]);
            vv[1] = __half2float(vh[lane + 32]);
            vv[2] = __half2float(vh[lane + 64]);
            vv[3] = __half2float(vh[lane + 96]);
        }

        #pragma unroll
        for (int g = 0; g < GQA; ++g) {
            float dot = q_reg[g][0] * kv[0] + q_reg[g][1] * kv[1]
                      + q_reg[g][2] * kv[2] + q_reg[g][3] * kv[3];
            dot = warp_reduce_sum(dot);
            const float score = dot + rt_base[g][dist];

            const float m_new = fmaxf(m_max[g], score);
            const float alpha = isinf(m_max[g]) ? 0.0f : expf(m_max[g] - m_new);
            const float p     = expf(score - m_new);
            l_sum[g] = l_sum[g] * alpha + p;
            o_acc[g][0] = o_acc[g][0] * alpha + p * vv[0];
            o_acc[g][1] = o_acc[g][1] * alpha + p * vv[1];
            o_acc[g][2] = o_acc[g][2] * alpha + p * vv[2];
            o_acc[g][3] = o_acc[g][3] * alpha + p * vv[3];
            m_max[g] = m_new;
        }
    }

    // Reduce across the 4 warps in shared memory
    __shared__ float s_o[NWARPS][GQA][128];
    __shared__ float s_m[NWARPS][GQA];
    __shared__ float s_l[NWARPS][GQA];

    for (int g = 0; g < GQA; ++g) {
        s_o[warp][g][lane +  0] = o_acc[g][0];
        s_o[warp][g][lane + 32] = o_acc[g][1];
        s_o[warp][g][lane + 64] = o_acc[g][2];
        s_o[warp][g][lane + 96] = o_acc[g][3];
        if (lane == 0) {
            s_m[warp][g] = m_max[g];
            s_l[warp][g] = l_sum[g];
        }
    }
    __syncthreads();

    if (warp < GQA) {
        const int g = warp;
        float mx = s_m[0][g];
        for (int w = 1; w < NWARPS; ++w) mx = fmaxf(mx, s_m[w][g]);
        float ltot = 0.0f;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
        if (!isinf(mx)) {
            for (int w = 0; w < NWARPS; ++w) {
                if (s_l[w][g] > 0.0f) {
                    const float sc = expf(s_m[w][g] - mx);
                    ltot += s_l[w][g] * sc;
                    o0 += s_o[w][g][lane +  0] * sc;
                    o1 += s_o[w][g][lane + 32] * sc;
                    o2 += s_o[w][g][lane + 64] * sc;
                    o3 += s_o[w][g][lane + 96] * sc;
                }
            }
        }
        const int h = kv_head * GQA + g;
        float * out = part_out + (((size_t) split * gridDim.y + q_tok) * Hq + h) * 130;
        out[lane +  0] = o0;
        out[lane + 32] = o1;
        out[lane + 64] = o2;
        out[lane + 96] = o3;
        if (lane == 0) {
            out[128] = mx;
            out[129] = ltot;
        }
    }
}

static __global__ void inkling2_attn_rel_reduce_splits_kernel(
        const float * __restrict__ part_in,  // [(128+2), Hq, T, num_splits]
        const float * __restrict__ gate_ptr, size_t gate_stride, // optional [128*Hq, T]
        float * __restrict__ dst,            // [128*Hq, T] or [(128+2)*Hq, T]
        int Hq, int T, int num_splits, int partial_out) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x; // 0..31

    const size_t split_stride = (size_t) T * Hq * 130;
    const size_t ht_off = ((size_t) t * Hq + h) * 130;

    float mx = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        const float m = part_in[(size_t) s * split_stride + ht_off + 128];
        mx = fmaxf(mx, m);
    }

    float ltot = 0.0f;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    if (!isinf(mx)) {
        for (int s = 0; s < num_splits; ++s) {
            const float * sp = part_in + (size_t) s * split_stride + ht_off;
            const float ls = sp[129];
            if (ls > 0.0f) {
                const float sc = expf(sp[128] - mx);
                ltot += ls * sc;
                o0 += sp[lane +  0] * sc;
                o1 += sp[lane + 32] * sc;
                o2 += sp[lane + 64] * sc;
                o3 += sp[lane + 96] * sc;
            }
        }
    }

    if (partial_out) {
        float * out = dst + ((size_t) t * Hq + h) * 130;
        out[lane +  0] = o0;
        out[lane + 32] = o1;
        out[lane + 64] = o2;
        out[lane + 96] = o3;
        if (lane == 0) {
            out[128] = mx;
            out[129] = ltot;
        }
    } else {
        const float inv_l = (ltot > 0.0f) ? (1.0f / ltot) : 0.0f;
        float * out = dst + ((size_t) t * Hq + h) * 128;
        if (gate_ptr) {
            const float * gp = gate_ptr + (size_t) t * gate_stride + (size_t) h * 128;
            out[lane +  0] = o0 * inv_l * sigmoid_f32(gp[lane +  0]);
            out[lane + 32] = o1 * inv_l * sigmoid_f32(gp[lane + 32]);
            out[lane + 64] = o2 * inv_l * sigmoid_f32(gp[lane + 64]);
            out[lane + 96] = o3 * inv_l * sigmoid_f32(gp[lane + 96]);
        } else {
            out[lane +  0] = o0 * inv_l;
            out[lane + 32] = o1 * inv_l;
            out[lane + 64] = o2 * inv_l;
            out[lane + 96] = o3 * inv_l;
        }
    }
}

// Tiled prefill kernel for T > 16: 1 block = (kv_head, q_tile of BM=8 tokens),
// 8 warps = 256 threads (1 warp per q_tok, each holding all GQA=3 heads -> 24 Q rows/block!).
// Loads BN=16 K & V rows into LDS once per tile -> 24x KV bandwidth reuse!
template <bool KV_Q8_0, int GQA>
static __global__ void inkling2_attn_rel_prefill_kernel(
        const float * __restrict__ q_ptr,     // [128, Hq, T]
        const char  * __restrict__ k_ptr,     // [128, kv_cap, Hk]
        const char  * __restrict__ v_ptr,     // [128, kv_cap, Hk]
        size_t k_nb1, size_t k_nb2,
        size_t v_nb1, size_t v_nb2,
        const float * __restrict__ rel_table, // [rel_extent+1, Hq, T]
        const int32_t * __restrict__ kv_meta, // {pos0, kv_len, kv_start, kv_end}
        const float * __restrict__ gate_ptr, size_t gate_stride,
        float * __restrict__ dst,             // [128*Hq, T] or [(128+2)*Hq, T]
        int Hq, int Hk, int T, int kv_cap,
        int swa_window, int rel_extent,
        int partial_out, int include_sink) {
    constexpr int BM = 8;
    constexpr int BN = 16;
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..7 (each warp = 1 query token)
    const int tid  = warp * 32 + lane; // 0..255

    const int kv_head = blockIdx.x;
    const int q_base  = blockIdx.y * BM;
    const int q_tok   = q_base + warp;
    const bool valid_q = (q_tok < T);

    const int pos0     = kv_meta[0];
    const int kv_len   = kv_meta[1];
    const int kv_start = kv_meta[2];
    const int kv_end_r = kv_meta[3] > 0 ? kv_meta[3] : kv_len;

    const int q_pos = pos0 + q_tok;
    const int tile_q_min = pos0 + q_base;
    const int tile_q_max = pos0 + min(T - 1, q_base + BM - 1);

    const int tile_k_begin = (swa_window > 0) ? max(kv_start, tile_q_min - swa_window + 1) : kv_start;
    const int tile_k_end   = min(kv_end_r, tile_q_max + 1);

    float q_reg[GQA][4];
    const float * rt_base[GQA];
    float m_max[GQA];
    float l_sum[GQA];
    float o_acc[GQA][4];

    for (int g = 0; g < GQA; ++g) {
        const int h = kv_head * GQA + g;
        if (valid_q) {
            const float * qp = q_ptr + ((size_t) q_tok * Hq + h) * 128;
            q_reg[g][0] = qp[lane +  0] * kInvSqrt128;
            q_reg[g][1] = qp[lane + 32] * kInvSqrt128;
            q_reg[g][2] = qp[lane + 64] * kInvSqrt128;
            q_reg[g][3] = qp[lane + 96] * kInvSqrt128;
            rt_base[g] = rel_table + ((size_t) q_tok * Hq + h) * (rel_extent + 1);
        }
        m_max[g] = include_sink ? 0.0f : -INFINITY;
        l_sum[g] = include_sink ? 1.0f : 0.0f;
        o_acc[g][0] = o_acc[g][1] = o_acc[g][2] = o_acc[g][3] = 0.0f;
    }

    __shared__ half s_K[BN][128];
    __shared__ half s_V[BN][128];

    const char * k_head = k_ptr + (size_t) kv_head * k_nb2;
    const char * v_head = v_ptr + (size_t) kv_head * v_nb2;

    for (int kb = tile_k_begin; kb < tile_k_end; kb += BN) {
        const int n_k = min(BN, tile_k_end - kb);
        // Cooperative load of n_k rows of K and V into LDS (256 threads: 16 rows x 16 = 256!)
        // Actually 8 warps, each warp loads 2 rows (warp*2 + 0, warp*2 + 1) across 32 lanes x 4 dims!
        for (int r = warp; r < n_k; r += BM) {
            const int kp = kb + r;
            const int slot = (swa_window > 0) ? (kp % kv_cap) : (kp - kv_start);
            if constexpr (KV_Q8_0) {
                const block_q8_0 * bk = reinterpret_cast<const block_q8_0 *>(k_head + (size_t) slot * k_nb1);
                const block_q8_0 * bv = reinterpret_cast<const block_q8_0 *>(v_head + (size_t) slot * v_nb1);
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    s_K[r][b * 32 + lane] = __float2half(__half2float(bk[b].d) * (float) bk[b].qs[lane]);
                    s_V[r][b * 32 + lane] = __float2half(__half2float(bv[b].d) * (float) bv[b].qs[lane]);
                }
            } else {
                const half * kh = reinterpret_cast<const half *>(k_head + (size_t) slot * k_nb1);
                const half * vh = reinterpret_cast<const half *>(v_head + (size_t) slot * v_nb1);
                s_K[r][lane +  0] = kh[lane +  0];
                s_K[r][lane + 32] = kh[lane + 32];
                s_K[r][lane + 64] = kh[lane + 64];
                s_K[r][lane + 96] = kh[lane + 96];
                s_V[r][lane +  0] = vh[lane +  0];
                s_V[r][lane + 32] = vh[lane + 32];
                s_V[r][lane + 64] = vh[lane + 64];
                s_V[r][lane + 96] = vh[lane + 96];
            }
        }
        __syncthreads();

        if (valid_q) {
            for (int r = 0; r < n_k; ++r) {
                const int kp = kb + r;
                const int d_pos = q_pos - kp;
                if (d_pos < 0 || (swa_window > 0 && d_pos >= swa_window)) continue;
                const int dist = min(d_pos, rel_extent);

                const float k0 = __half2float(s_K[r][lane +  0]);
                const float k1 = __half2float(s_K[r][lane + 32]);
                const float k2 = __half2float(s_K[r][lane + 64]);
                const float k3 = __half2float(s_K[r][lane + 96]);
                const float v0 = __half2float(s_V[r][lane +  0]);
                const float v1 = __half2float(s_V[r][lane + 32]);
                const float v2 = __half2float(s_V[r][lane + 64]);
                const float v3 = __half2float(s_V[r][lane + 96]);

                #pragma unroll
                for (int g = 0; g < GQA; ++g) {
                    float dot = q_reg[g][0] * k0 + q_reg[g][1] * k1
                              + q_reg[g][2] * k2 + q_reg[g][3] * k3;
                    dot = warp_reduce_sum(dot);
                    const float score = dot + rt_base[g][dist];
                    const float m_new = fmaxf(m_max[g], score);
                    const float alpha = isinf(m_max[g]) ? 0.0f : expf(m_max[g] - m_new);
                    const float p     = expf(score - m_new);
                    l_sum[g] = l_sum[g] * alpha + p;
                    o_acc[g][0] = o_acc[g][0] * alpha + p * v0;
                    o_acc[g][1] = o_acc[g][1] * alpha + p * v1;
                    o_acc[g][2] = o_acc[g][2] * alpha + p * v2;
                    o_acc[g][3] = o_acc[g][3] * alpha + p * v3;
                    m_max[g] = m_new;
                }
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        for (int g = 0; g < GQA; ++g) {
            const int h = kv_head * GQA + g;
            if (partial_out) {
                float * out = dst + ((size_t) q_tok * Hq + h) * 130;
                out[lane +  0] = o_acc[g][0];
                out[lane + 32] = o_acc[g][1];
                out[lane + 64] = o_acc[g][2];
                out[lane + 96] = o_acc[g][3];
                if (lane == 0) {
                    out[128] = m_max[g];
                    out[129] = l_sum[g];
                }
            } else {
                const float inv_l = (l_sum[g] > 0.0f) ? (1.0f / l_sum[g]) : 0.0f;
                float * out = dst + ((size_t) q_tok * Hq + h) * 128;
                if (gate_ptr) {
                    const float * gp = gate_ptr + (size_t) q_tok * gate_stride + (size_t) h * 128;
                    out[lane +  0] = o_acc[g][0] * inv_l * sigmoid_f32(gp[lane +  0]);
                    out[lane + 32] = o_acc[g][1] * inv_l * sigmoid_f32(gp[lane + 32]);
                    out[lane + 64] = o_acc[g][2] * inv_l * sigmoid_f32(gp[lane + 64]);
                    out[lane + 96] = o_acc[g][3] * inv_l * sigmoid_f32(gp[lane + 96]);
                } else {
                    out[lane +  0] = o_acc[g][0] * inv_l;
                    out[lane + 32] = o_acc[g][1] * inv_l;
                    out[lane + 64] = o_acc[g][2] * inv_l;
                    out[lane + 96] = o_acc[g][3] * inv_l;
                }
            }
        }
    }
    (void) tid;
}

#if defined(GGML_USE_HIP)
using i2_f16x8_t  = __attribute__((ext_vector_type(8)))  _Float16;
using i2_f16x16_t = __attribute__((ext_vector_type(16))) _Float16;
using i2_f32x8_t  = __attribute__((ext_vector_type(8)))  float;

static __device__ __forceinline__ int i2_k_swiz(const int ki, const int chunk16) {
    return chunk16 ^ (ki & 7);
}
static __device__ __forceinline__ int i2_v_swiz32(const int d, const int chunk16) {
    return chunk16 ^ ((d >> 1) & 3) ^ ((d >> 3) & 3);
}
static __device__ __forceinline__ uint32_t i2_pack_f16x2(const float a, const float b) {
    const __half2 h2 = __floats2half2_rn(a, b);
    return __builtin_bit_cast(uint32_t, h2);
}
static __device__ __forceinline__ uint4 i2_dequant_q8_0_quad(const block_q8_0 * bp, const int sub) {
    const float d = __half2float(bp->d);
    const uint16_t * qp = reinterpret_cast<const uint16_t *>(bp->qs + sub * 8);
    const uint32_t p0 = qp[0], p1 = qp[1], p2 = qp[2], p3 = qp[3];
    return make_uint4(
        i2_pack_f16x2(d * (float)(int8_t)(p0 & 0xFFu), d * (float)(int8_t)(p0 >> 8)),
        i2_pack_f16x2(d * (float)(int8_t)(p1 & 0xFFu), d * (float)(int8_t)(p1 >> 8)),
        i2_pack_f16x2(d * (float)(int8_t)(p2 & 0xFFu), d * (float)(int8_t)(p2 >> 8)),
        i2_pack_f16x2(d * (float)(int8_t)(p3 & 0xFFu), d * (float)(int8_t)(p3 >> 8)));
}
static __device__ __forceinline__ void i2_store_v_4rows32(
        _Float16 * KV_lds, const int d0, const int g,
        const uint4 r0, const uint4 r1, const uint4 r2, const uint4 r3) {
    const int ch16 = g >> 1;
    const int sub4 = (g & 1) * 4;
    auto write_pair = [&](const int u_base, const uint32_t w0, const uint32_t w1, const uint32_t w2, const uint32_t w3) {
        const uint2 lo = make_uint2((w0 & 0xFFFFu) | (w1 << 16), (w2 & 0xFFFFu) | (w3 << 16));
        const uint2 hi = make_uint2((w0 >> 16) | (w1 & 0xFFFF0000u), (w2 >> 16) | (w3 & 0xFFFF0000u));
        const int d_lo = d0 + u_base;
        const int d_hi = d0 + u_base + 1;
        *reinterpret_cast<uint2 *>(KV_lds + d_lo * 32 + i2_v_swiz32(d_lo, ch16) * 8 + sub4) = lo;
        *reinterpret_cast<uint2 *>(KV_lds + d_hi * 32 + i2_v_swiz32(d_hi, ch16) * 8 + sub4) = hi;
    };
    write_pair(0, r0.x, r1.x, r2.x, r3.x);
    write_pair(2, r0.y, r1.y, r2.y, r3.y);
    write_pair(4, r0.z, r1.z, r2.z, r3.z);
    write_pair(6, r0.w, r1.w, r2.w, r3.w);
}

// Native RDNA4 (gfx1201) & RDNA3.5 (gfx1151) Wave32 WMMA Inline-RelBias + Zero-Sink + Fused-Gate Prefill FlashAttention!
// BM=128 GQA-packed query rows per block (8 warps), BN=32 KV tile in XOR-swizzled 8 KB LDS -> 40+ TFLOPS!
template <bool KV_Q8_0, int GQA>
static __global__ void __launch_bounds__(256, 1)
inkling2_attn_rel_prefill_wmma_kernel(
        const float * __restrict__ q_ptr,     // [128, Hq, T]
        const char  * __restrict__ k_ptr,     // [128, kv_cap, Hk]
        const char  * __restrict__ v_ptr,     // [128, kv_cap, Hk]
        size_t k_nb1, size_t k_nb2,
        size_t v_nb1, size_t v_nb2,
        const float * __restrict__ rel_table, // [rel_extent+1, Hq, T]
        const int32_t * __restrict__ kv_meta, // {pos0, kv_len, kv_start, kv_end}
        const float * __restrict__ gate_ptr, size_t gate_stride,
        float * __restrict__ dst,             // [128*Hq, T] or [(128+2)*Hq, T]
        int Hq, int Hk, int T, int kv_cap,
        int swa_window, int rel_extent,
        int partial_out, int include_sink) {
#if defined(RDNA4) || defined(RDNA3) || defined(RDNA3_5)
    constexpr int D = 128;
    constexpr int BN = 32;
    constexpr int BM = 128;
    constexpr int ND = 8; // 128 / 16
    constexpr int NT = 2; // 32 / 16

    __shared__ __align__(16) _Float16 KV_lds[BN * D];

    const int lane = threadIdx.x;      // 0..31
    const int wave = threadIdx.y;      // 0..7
    const int tid  = wave * 32 + lane; // 0..255
    const int r    = lane & 15;        // 0..15
    const int hi   = lane >> 4;        // 0 or 1

    const int kv_head = blockIdx.x;
    const int q_base  = blockIdx.y * BM;
    const int total_q = T * GQA;

    const int pos0     = kv_meta[0];
    const int kv_len   = kv_meta[1];
    const int kv_start = kv_meta[2];
    const int kv_end_r = kv_meta[3] > 0 ? kv_meta[3] : kv_len;

    const int row_idx  = q_base + wave * 16 + r;
    const bool valid_q = (row_idx < total_q);
    const int q_tok    = valid_q ? (row_idx / GQA) : 0;
    const int head     = valid_q ? (kv_head * GQA + (row_idx - q_tok * GQA)) : 0;
    const int q_pos    = pos0 + q_tok;

    const int tile_qtok_min = q_base / GQA;
    const int tile_qtok_max = min(T - 1, (q_base + BM - 1) / GQA);
    const int tile_k_begin  = (swa_window > 0) ? max(kv_start, pos0 + tile_qtok_min - swa_window + 1) : kv_start;
    const int tile_k_end    = min(kv_end_r, pos0 + tile_qtok_max + 1);

    const float q_scale = kInvSqrt128 * kLog2E;
    i2_f16x8_t Q_reg[ND];
    const float * rt_base = nullptr;
    if (valid_q) {
        const float * qp = q_ptr + ((size_t) q_tok * Hq + head) * D;
        rt_base = rel_table + ((size_t) q_tok * Hq + head) * (rel_extent + 1);
#pragma unroll
        for (int kt = 0; kt < ND; ++kt) {
            const float4 f0 = *reinterpret_cast<const float4 *>(qp + kt * 16 + hi * 8 + 0);
            const float4 f1 = *reinterpret_cast<const float4 *>(qp + kt * 16 + hi * 8 + 4);
            const uint4 qu = make_uint4(
                i2_pack_f16x2(f0.x * q_scale, f0.y * q_scale),
                i2_pack_f16x2(f0.z * q_scale, f0.w * q_scale),
                i2_pack_f16x2(f1.x * q_scale, f1.y * q_scale),
                i2_pack_f16x2(f1.z * q_scale, f1.w * q_scale));
            Q_reg[kt] = __builtin_bit_cast(i2_f16x8_t, qu);
        }
    } else {
#pragma unroll
        for (int kt = 0; kt < ND; ++kt) {
            Q_reg[kt] = {(_Float16)0.f, (_Float16)0.f, (_Float16)0.f, (_Float16)0.f,
                         (_Float16)0.f, (_Float16)0.f, (_Float16)0.f, (_Float16)0.f};
        }
    }

    i2_f32x8_t O[ND];
#pragma unroll
    for (int dt = 0; dt < ND; ++dt) O[dt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_max = include_sink ? 0.0f : -INFINITY;
    float l_sum = include_sink ? 1.0f : 0.0f;

    const char * k_head = k_ptr + (size_t) kv_head * k_nb2;
    const char * v_head = v_ptr + (size_t) kv_head * v_nb2;

    for (int kb = tile_k_begin; kb < tile_k_end; kb += BN) {
        // 1. Load K[kb .. kb+31] into XOR-swizzled KV_lds[32][128]
        {
            constexpr int chunks_per_row = D / 8; // 16
            constexpr int total_chunks = BN * chunks_per_row; // 512
#pragma unroll 1
            for (int idx = tid; idx < total_chunks; idx += 256) {
                const int ki = idx / chunks_per_row;
                const int c  = idx - ki * chunks_per_row;
                const int kp = kb + ki;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (kp < tile_k_end) {
                    const int slot = (swa_window > 0) ? (kp % kv_cap) : (kp - kv_start);
                    if constexpr (!KV_Q8_0) {
                        v = *reinterpret_cast<const uint4 *>(k_head + (size_t) slot * k_nb1 + c * 16);
                    } else {
                        const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(k_head + (size_t) slot * k_nb1) + (c >> 2);
                        v = i2_dequant_q8_0_quad(bp, c & 3);
                    }
                }
                const int sc = i2_k_swiz(ki, c);
                *reinterpret_cast<uint4 *>(KV_lds + ki * D + sc * 8) = v;
            }
        }
        __syncthreads();

        // 2. S = K * Q^T via Wave32 WMMA
        i2_f32x8_t S[NT];
        S[0] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        S[1] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kt = 0; kt < ND; ++kt) {
#if defined(RDNA4)
            const i2_f16x8_t qb = Q_reg[kt];
            const int c_raw = kt * 2 + hi;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int ki = t * 16 + r;
                const int sc = i2_k_swiz(ki, c_raw);
                const uint4 ku = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + sc * 8);
                S[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(i2_f16x8_t, ku), qb, S[t]);
            }
            __builtin_amdgcn_sched_barrier(0);
#else
            const uint4 q_self = __builtin_bit_cast(uint4, Q_reg[kt]);
            const uint4 q_peer = make_uint4(
                __shfl_xor(q_self.x, 16, 32), __shfl_xor(q_self.y, 16, 32),
                __shfl_xor(q_self.z, 16, 32), __shfl_xor(q_self.w, 16, 32));
            const uint4 q0 = hi ? q_peer : q_self;
            const uint4 q1 = hi ? q_self : q_peer;
            const i2_f16x16_t qb = __builtin_bit_cast(i2_f16x16_t, (uint4[2]){q0, q1});
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int ki = t * 16 + r;
                const uint4 ku0 = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + i2_k_swiz(ki, kt * 2 + 0) * 8);
                const uint4 ku1 = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + i2_k_swiz(ki, kt * 2 + 1) * 8);
                S[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(i2_f16x16_t, (uint4[2]){ku0, ku1}), qb, S[t]);
            }
            __builtin_amdgcn_sched_barrier(0);
#endif
        }

        // 3. Inline causal + SWA mask + relative-position bias (ZERO kq_b memory!)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                const int kp = kb + t * 16 + hi * 8 + e;
#else
                const int kp = kb + t * 16 + 2 * e + hi;
#endif
                const int d_pos = q_pos - kp;
                if (!valid_q || kp >= tile_k_end || d_pos < 0 || (swa_window > 0 && d_pos >= swa_window)) {
                    S[t][e] = -INFINITY;
                } else {
                    const int dist = min(d_pos, rel_extent);
                    S[t][e] += rt_base[dist] * kLog2E;
                }
            }
        }

        // 4. Online Softmax (hardware v_exp_f32)
        float m_tile = m_max;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int e = 0; e < 8; ++e) m_tile = fmaxf(m_tile, S[t][e]);
        }
        m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16, 32));

        const bool valid_tile = isfinite(m_tile);
        const float alpha = (valid_tile && isfinite(m_max)) ? exp2f(m_max - m_tile) : 0.0f;
        if (valid_tile) {
            m_max = m_tile;
            l_sum *= alpha;
#pragma unroll
            for (int dt = 0; dt < ND; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) O[dt][e] *= alpha;
            }
        }

        i2_f16x8_t P[NT];
        float p_sum = 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float pv[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pv[e] = (valid_tile && isfinite(S[t][e])) ? exp2f(S[t][e] - m_max) : 0.0f;
                p_sum += pv[e];
            }
            const uint4 pu = make_uint4(
                i2_pack_f16x2(pv[0], pv[1]), i2_pack_f16x2(pv[2], pv[3]),
                i2_pack_f16x2(pv[4], pv[5]), i2_pack_f16x2(pv[6], pv[7]));
            P[t] = __builtin_bit_cast(i2_f16x8_t, pu);
        }
        p_sum += __shfl_xor(p_sum, 16, 32);
        l_sum += p_sum;

        __syncthreads();

        // 5. Load V[kb .. kb+31] transposed into XOR-swizzled KV_lds[128][32]
        {
            constexpr int chunks_per_row = D / 8; // 16
            constexpr int row_groups = BN / 4;    // 8
            constexpr int total_tasks = row_groups * chunks_per_row; // 128
            if (tid < total_tasks) {
                const int g   = tid / chunks_per_row;
                const int c   = tid - g * chunks_per_row;
                const int ki0 = g * 4;
                const int d0  = c * 8;
                auto load_v_row = [&](int kp) -> uint4 {
                    if (kp >= tile_k_end) return make_uint4(0u, 0u, 0u, 0u);
                    const int slot = (swa_window > 0) ? (kp % kv_cap) : (kp - kv_start);
                    if constexpr (!KV_Q8_0) {
                        return *reinterpret_cast<const uint4 *>(v_head + (size_t) slot * v_nb1 + c * 16);
                    } else {
                        const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(v_head + (size_t) slot * v_nb1) + (c >> 2);
                        return i2_dequant_q8_0_quad(bp, c & 3);
                    }
                };
                uint4 r0 = load_v_row(kb + ki0 + 0);
                uint4 r1 = load_v_row(kb + ki0 + 1);
                uint4 r2 = load_v_row(kb + ki0 + 2);
                uint4 r3 = load_v_row(kb + ki0 + 3);
                i2_store_v_4rows32(KV_lds, d0, g, r0, r1, r2, r3);
            }
        }
        __syncthreads();

        // 6. O += V^T * P via Wave32 WMMA
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#if defined(RDNA4)
            const i2_f16x8_t pt = P[t];
#pragma unroll
            for (int dt = 0; dt < ND; ++dt) {
                const int d = dt * 16 + r;
                const int sc = i2_v_swiz32(d, t * 2 + hi);
                const uint4 vu = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + sc * 8);
                O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(i2_f16x8_t, vu), pt, O[dt]);
                __builtin_amdgcn_sched_barrier(0);
            }
#else
            const uint4 p_self = __builtin_bit_cast(uint4, P[t]);
            const uint4 p_peer = make_uint4(
                __shfl_xor(p_self.x, 16, 32), __shfl_xor(p_self.y, 16, 32),
                __shfl_xor(p_self.z, 16, 32), __shfl_xor(p_self.w, 16, 32));
            const uint4 lo_s = hi ? p_peer : p_self;
            const uint4 hi_s = hi ? p_self : p_peer;
            auto ilo = [](uint32_t a, uint32_t b) -> uint32_t { return (a & 0xFFFFu) | (b << 16); };
            auto ihi = [](uint32_t a, uint32_t b) -> uint32_t { return (a >> 16) | (b & 0xFFFF0000u); };
            const uint4 p0 = make_uint4(ilo(lo_s.x, hi_s.x), ihi(lo_s.x, hi_s.x), ilo(lo_s.y, hi_s.y), ihi(lo_s.y, hi_s.y));
            const uint4 p1 = make_uint4(ilo(lo_s.z, hi_s.z), ihi(lo_s.z, hi_s.z), ilo(lo_s.w, hi_s.w), ihi(lo_s.w, hi_s.w));
            const i2_f16x16_t pt = __builtin_bit_cast(i2_f16x16_t, (uint4[2]){p0, p1});
#pragma unroll
            for (int dt = 0; dt < ND; ++dt) {
                const int d = dt * 16 + r;
                const uint4 v0 = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + i2_v_swiz32(d, t * 2 + 0) * 8);
                const uint4 v1 = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + i2_v_swiz32(d, t * 2 + 1) * 8);
                O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(i2_f16x16_t, (uint4[2]){v0, v1}), pt, O[dt]);
                __builtin_amdgcn_sched_barrier(0);
            }
#endif
        }
        __syncthreads();
    }

    // 7. Epilogue: partial (convert m_max log2 -> ln!) or normalized (+ fused sigmoid gate!)
    if (valid_q) {
        if (partial_out) {
            float * out = dst + ((size_t) q_tok * Hq + head) * 130;
#pragma unroll
            for (int dt = 0; dt < ND; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                    const int d = dt * 16 + hi * 8 + e;
#else
                    const int d = dt * 16 + 2 * e + hi;
#endif
                    out[d] = O[dt][e];
                }
            }
            if (hi == 0) {
                out[128] = isinf(m_max) ? -INFINITY : (m_max * kLn2);
                out[129] = l_sum;
            }
        } else {
            const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
            float * out = dst + ((size_t) q_tok * Hq + head) * D;
            const float * gp = gate_ptr ? (gate_ptr + (size_t) q_tok * gate_stride + (size_t) head * D) : nullptr;
#pragma unroll
            for (int dt = 0; dt < ND; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                    const int d = dt * 16 + hi * 8 + e;
#else
                    const int d = dt * 16 + 2 * e + hi;
#endif
                    float v = O[dt][e] * inv_l;
                    if (gp) v *= sigmoid_f32(gp[d]);
                    out[d] = v;
                }
            }
        }
    }
#endif
}
#endif // GGML_USE_HIP

// ─── 9. Combine Two Split-KV Partial Outputs (Dual-GPU Split-KV) ────────
static __global__ void inkling2_attn_split_combine_kernel(
        const float * __restrict__ part_a,
        const float * __restrict__ part_b,
        const float * __restrict__ gate_ptr, size_t gate_stride,
        float * __restrict__ dst,
        int head_dim, int n_head) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;
    const int stride_p = head_dim + 2;
    const float * pa = part_a + ((size_t) t * n_head + h) * stride_p;
    const float * pb = part_b + ((size_t) t * n_head + h) * stride_p;
    const float ma = pa[head_dim], la = pa[head_dim + 1];
    const float mb = pb[head_dim], lb = pb[head_dim + 1];
    const float mx = fmaxf(ma, mb);
    const float sa = (la > 0.0f && !isinf(mx)) ? expf(ma - mx) : 0.0f;
    const float sb = (lb > 0.0f && !isinf(mx)) ? expf(mb - mx) : 0.0f;
    const float inv_l = (la * sa + lb * sb > 0.0f) ? (1.0f / (la * sa + lb * sb)) : 0.0f;

    float * out = dst + ((size_t) t * n_head + h) * head_dim;
    const float * gp = gate_ptr ? (gate_ptr + (size_t) t * gate_stride + (size_t) h * head_dim) : nullptr;
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        float v = (pa[d] * sa + pb[d] * sb) * inv_l;
        if (gp) v *= sigmoid_f32(gp[d]);
        out[d] = v;
    }
}

// ─── 10. Fused SITU MoE Activation ──────────────────────────────────────
// gate_up: [2*n_ff, k, T] (first n_ff = gate, second n_ff = up) -> [n_ff, k, T]
static __global__ void inkling2_situ_kernel(
        const float * __restrict__ gate_up,
        float * __restrict__ dst,
        int n_ff, int n_rows, float inv_beta, float beta,
        float inv_lb, float lb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_ff * n_rows;
    if (idx >= total) return;
    const int row = idx / n_ff;
    const int col = idx - row * n_ff;
    const float g = gate_up[(size_t) row * (2 * n_ff) + col];
    const float u = gate_up[(size_t) row * (2 * n_ff) + n_ff + col];
    const float act = beta * tanhf(g * inv_beta) * sigmoid_f32(g);
    const float lin = (lb > 0.0f) ? (lb * tanhf(u * inv_lb)) : u;
    dst[idx] = act * lin;
}

// ─── 11. Fused Sigmoid MoE Router (1-Warp Register-Shuffle Top-K — 12x Faster!) ─
static __global__ void inkling2_sigmoid_topk_ids_kernel(
        const float * __restrict__ logits, // [n_exp, T]
        const float * __restrict__ bias,   // nullptr or [n_exp]
        int32_t * __restrict__ out_ids,    // [top_k, T]
        int n_exp, int top_k) {
    // 1 warp (32 lanes) per token! Zero __syncthreads(), zero LDS!
    const int t    = blockIdx.x;
    const int lane = threadIdx.x; // 0..31
    const float * lp = logits + (size_t) t * n_exp;
    int32_t * op = out_ids + (size_t) t * top_k;

    // Each lane holds up to 8 experts (supports n_exp <= 256)
    float sc[8];
    int   id[8];
    int   cnt = 0;
    for (int e = lane; e < n_exp && cnt < 8; e += WARP_SIZE) {
        float p = sigmoid_f32(lp[e]);
        if (bias) p += bias[e];
        sc[cnt] = p;
        id[cnt] = e;
        ++cnt;
    }

    for (int k = 0; k < top_k; ++k) {
        float best_v = -INFINITY;
        int   best_i = n_exp;
        int   best_j = -1;
        for (int j = 0; j < cnt; ++j) {
            if (sc[j] > best_v || (sc[j] == best_v && id[j] < best_i)) {
                best_v = sc[j];
                best_i = id[j];
                best_j = j;
            }
        }
        // Warp-wide max-index reduction (exact tie-break by smaller expert id)
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            const float ov = i2_shfl_xor(best_v, off, WARP_SIZE);
            const int   oi = i2_shfl_xor(best_i, off, WARP_SIZE);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            op[k] = best_i;
        }
        if (best_j >= 0 && id[best_j] == best_i) {
            sc[best_j] = -INFINITY;
        }
    }
}

static __global__ void inkling2_sigmoid_topk_wts_kernel(
        const float * __restrict__ logits, // [n_exp, T]
        const int32_t * __restrict__ ids,  // [top_k, T]
        float * __restrict__ out_wts,      // [top_k, T]
        int n_exp, int top_k, float route_scale, float eps) {
    // 1 warp per token: top_k lanes load & reduce in parallel!
    const int t    = blockIdx.x;
    const int lane = threadIdx.x;
    const float * lp   = logits + (size_t) t * n_exp;
    const int32_t * ip = ids    + (size_t) t * top_k;
    float * wp         = out_wts + (size_t) t * top_k;

    float p = (lane < top_k) ? sigmoid_f32(lp[ip[lane]]) : 0.0f;
    const float sm = warp_reduce_sum(p);
    if (lane < top_k) {
        wp[lane] = p * (route_scale / fmaxf(sm, eps));
    }
}

// ─── 12. Fused Final Logit Softcap + Unpadded Slice ─────────────────────
static __global__ void inkling2_softcap_slice_kernel(
        const float * __restrict__ logits, size_t src_stride,
        float * __restrict__ dst,
        int n_vocab_unpadded, int n_tokens, float softcap) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) n_vocab_unpadded * n_tokens;
    if (idx >= total) return;
    const int64_t t = idx / n_vocab_unpadded;
    const int64_t v = idx - t * n_vocab_unpadded;
    float x = logits[t * src_stride + v];
    if (softcap > 0.0f) {
        x = softcap * tanhf(x / softcap);
    }
    dst[idx] = x;
}

// ─── 13. DSpark: DFlare Weighted Tap Mix + RMSNorm ──────────────────────
static __global__ void inkling2_dspark_mix_norm_kernel(
        const float * __restrict__ taps,   // [n_embd, n_taps, T]
        const float * __restrict__ alpha,  // [n_taps]
        const float * __restrict__ weight, // [n_embd]
        float * __restrict__ dst,          // [n_embd, T]
        int n_embd, int n_taps, float eps) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int nwarps = nthreads / WARP_SIZE;

    __shared__ float s_red[32];
    float * op = dst + (size_t) t * n_embd;
    float ss = 0.0f;
    for (int e = tid; e < n_embd; e += nthreads) {
        float acc = 0.0f;
        for (int k = 0; k < n_taps; ++k) {
            acc += taps[((size_t) t * n_taps + k) * n_embd + e] * alpha[k];
        }
        op[e] = acc;
        ss += acc * acc;
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nwarps) ? s_red[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_red[0] = rsqrtf(v / (float) n_embd + eps);
    }
    __syncthreads();
    const float irms = s_red[0];
    for (int e = tid; e < n_embd; e += nthreads) {
        op[e] = op[e] * irms * weight[e];
    }
}

// ─── 14. DSpark: RNN Markov Gate (exact vLLM bf16-rounded semantics) ────
static __global__ void inkling2_dspark_markov_gate_kernel(
        const float * __restrict__ raw,   // [3*R, B]
        const float * __restrict__ state, // [R, B]
        float * __restrict__ dst,         // [2*R, B]: first R = new_state, second R = latent
        int rank) {
    const int b = blockIdx.x;
    const int c = threadIdx.x;
    if (c >= rank) return;
    const float * rp = raw + (size_t) b * (3 * rank);
    const float g_in = round_bf16_f32(rp[c]);
    const float c_in = round_bf16_f32(rp[rank + c]);
    const float o_in = round_bf16_f32(rp[2 * rank + c]);
    const float st   = state[(size_t) b * rank + c];

    const float gate   = round_bf16_f32(1.0f / (1.0f + expf(-g_in)));
    const float cand   = round_bf16_f32(tanhf(c_in));
    const float keep   = round_bf16_f32(gate * st);
    const float update = round_bf16_f32(round_bf16_f32(1.0f - gate) * cand);
    const float out_st = round_bf16_f32(keep + update);
    const float latent = round_bf16_f32(tanhf(o_in));

    float * op = dst + (size_t) b * (2 * rank);
    op[c]        = out_st;
    op[rank + c] = latent;
}

// ─── 15. DSpark: Add + Softcap + Unpadded Slice ─────────────────────────
static __global__ void inkling2_dspark_add_softcap_kernel(
        const float * __restrict__ base_logits, size_t base_stride,
        const float * __restrict__ bias,        size_t bias_stride,
        float * __restrict__ dst,
        int n_vocab_unpadded, int B, float softcap) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) n_vocab_unpadded * B;
    if (idx >= total) return;
    const int64_t b = idx / n_vocab_unpadded;
    const int64_t v = idx - b * n_vocab_unpadded;
    float x = round_bf16_f32(base_logits[b * base_stride + v] + bias[b * bias_stride + v]);
    if (softcap > 0.0f) {
        x = round_bf16_f32(softcap * tanhf(x / softcap));
    }
    dst[idx] = x;
}

// ─── 16. State Restore from Verify Ring ─────────────────────────────────
static __global__ void inkling2_state_restore_kernel(
        const float * __restrict__ state_ring, // [3, S, max_verify]
        float * __restrict__ state_live,       // [3, S]
        const int32_t * __restrict__ slot_idx, // [1]
        int64_t n_elems) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    const int32_t s = slot_idx[0];
    if (s < 0) return;
    state_live[i] = state_ring[(int64_t) s * n_elems + i];
}

// ─── 17. Dynamic Exact-Unique Work-Stealing MoE LUT (Dual-GPU Zero-Straggler) ─
// Deduplicates active experts in sel[n_used, T], sorts active unique experts by
// (freq desc, id asc), assigns exact ceil(n_unique * target_pct / 100) multi-token
// experts to target (lut[e]=e) and the remaining 1-token experts to Strix (lut[e]=-1).
static __global__ void inkling2_dynamic_split_lut_kernel(
        const int32_t * __restrict__ sel, int sel_ld,
        int32_t * __restrict__ out_lut,
        int n_used, int n_tokens, int n_expert, int target_pct) {
    __shared__ int32_t s_freq[256];
    __shared__ int32_t s_hist[32];
    __shared__ int32_t s_warp_cnt[8];
    __shared__ int32_t s_cut_f;
    __shared__ int32_t s_cut_rem;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;

    if (tid < n_expert) {
        s_freq[tid] = 0;
        out_lut[tid] = -1;
    }
    if (tid < 32) s_hist[tid] = 0;
    __syncthreads();

    const int total = n_used * n_tokens;
    for (int i = tid; i < total; i += blockDim.x) {
        const int t = i / n_used;
        const int k = i - t * n_used;
        const int32_t e = sel[t * sel_ld + k];
        if (e >= 0 && e < n_expert) {
            atomicAdd(&s_freq[e], 1);
        }
    }
    __syncthreads();

    // O(1) parallel frequency histogram (freq <= 31 for verify T <= 16 -> ~1.2 us total, 47x faster!)
    const int f = (tid < n_expert) ? min(s_freq[tid], 31) : 0;
    if (f > 0) {
        atomicAdd(&s_hist[f], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int n_unique = 0;
        for (int b = 1; b < 32; ++b) n_unique += s_hist[b];
        int n_hot = (n_unique * target_pct + 99) / 100;
        if (n_hot > n_unique) n_hot = n_unique;
        int cut_f = 32;
        int cut_rem = 0;
        if (n_hot > 0) {
            int acc = 0;
            for (int b = 31; b >= 1; --b) {
                if (s_hist[b] == 0) continue;
                if (acc + s_hist[b] <= n_hot) {
                    acc += s_hist[b];
                    cut_f = b;
                    cut_rem = 0;
                    if (acc == n_hot) break;
                } else {
                    cut_f = b;
                    cut_rem = n_hot - acc;
                    break;
                }
            }
        }
        s_cut_f = cut_f;
        s_cut_rem = cut_rem;
    }
    __syncthreads();

    const int cut_f   = s_cut_f;
    const int cut_rem = s_cut_rem;
    const bool is_tie = (tid < n_expert && f > 0 && f == cut_f && cut_rem > 0);
    const unsigned long long mask = i2_ballot(is_tie);
    if (lane == 0) s_warp_cnt[warp] = __popcll(mask);
    __syncthreads();

    if (tid < n_expert && f > 0) {
        if (f > cut_f || (f == cut_f && cut_rem == 0)) {
            out_lut[tid] = tid;
        } else if (is_tie) {
            int rank = 0;
            for (int w = 0; w < warp; ++w) rank += s_warp_cnt[w];
            const unsigned long long lower_mask = (1ULL << lane) - 1ULL;
            rank += __popcll(mask & lower_mask);
            if (rank < cut_rem) {
                out_lut[tid] = tid;
            }
        }
    }
}

// ─── 18. DSpark Fast Markov Step 1: 256 Buckets x Top-4 = 1024 Candidates ─
// Reads base_logits [V, B] ONCE in a single pass (<0.01 ms for all 7 rows!).
static __global__ void inkling2_dspark_topk_cands_kernel(
        const float * __restrict__ base_logits, size_t base_stride,
        int32_t * __restrict__ out_ids, // [1024, B]
        int n_vocab_unpadded) {
    // grid = (256 buckets, B rows), block = 256 threads
    const int bucket = blockIdx.x;
    const int row    = blockIdx.y;
    const int tid    = threadIdx.x;
    const int n_buckets = 256;

    const int chunk = (n_vocab_unpadded + n_buckets - 1) / n_buckets;
    const int v_beg = bucket * chunk;
    const int v_end = min(n_vocab_unpadded, v_beg + chunk);
    const float * bp = base_logits + (size_t) row * base_stride;

    // Each thread keeps top-4 in its striped subset of [v_beg, v_end)
    float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY;
    int   i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    for (int v = v_beg + tid; v < v_end; v += blockDim.x) {
        const float val = bp[v];
        if (val > v3) {
            if (val > v0) { v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = val; i0 = v; }
            else if (val > v1) { v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = val; i1 = v; }
            else if (val > v2) { v3 = v2; i3 = i2; v2 = val; i2 = v; }
            else { v3 = val; i3 = v; }
        }
    }

    __shared__ float s_val[256];
    __shared__ int   s_id[256];

    // 4 fast block-wide max reductions to extract top-4 of this bucket!
    int32_t * dst = out_ids + (size_t) row * 1024 + bucket * 4;
    for (int k = 0; k < 4; ++k) {
        s_val[tid] = v0;
        s_id[tid]  = i0;
        __syncthreads();
        for (int off = 128; off > 0; off >>= 1) {
            if (tid < off) {
                if (s_val[tid + off] > s_val[tid]) {
                    s_val[tid] = s_val[tid + off];
                    s_id[tid]  = s_id[tid + off];
                }
            }
            __syncthreads();
        }
        const int win_id = s_id[0];
        if (tid == 0) dst[k] = win_id;
        // Shift out winning element in the owning thread
        if (i0 == win_id && v0 == s_val[0]) {
            v0 = v1; i0 = i1;
            v1 = v2; i1 = i2;
            v2 = v3; i2 = i3;
            v3 = -INFINITY; i3 = 0;
        }
        __syncthreads();
    }
}

// ─── 19. DSpark Fast Markov Step 2: Score 1024 Candidates + Argmax ──────
// 256 blocks x 4 warps = 1024 warps (1 warp per candidate, 8 F16 dots/lane for R=256).
// Total data read: 1024 * 256 * 2 B = 512 KB (L2-resident!) -> ~8 us/step!
static __global__ void inkling2_dspark_topk_score_kernel(
        const int32_t * __restrict__ cand_ids,   // [1024]
        const float   * __restrict__ base_row,   // [V]
        const float   * __restrict__ latent,     // [R=256]
        const half    * __restrict__ fold_w,     // [R=256, V] F16
        const float   * __restrict__ fold_b,     // [V] F32
        float         * __restrict__ out_scores, // [1024]
        int rank, float softcap) {
    const int warp_in_blk = threadIdx.y; // 0..3
    const int lane        = threadIdx.x; // 0..31
    const int k = blockIdx.x * 4 + warp_in_blk;
    if (k >= 1024) return;

    const int32_t cid = cand_ids[k];
    const half * fw = fold_w + (size_t) cid * rank;
    float dot = 0.0f;
    for (int r = lane; r < rank; r += WARP_SIZE) {
        dot += __half2float(fw[r]) * latent[r];
    }
    dot = warp_reduce_sum(dot);
    if (lane == 0) {
        float x = round_bf16_f32(base_row[cid] + dot + fold_b[cid]);
        if (softcap > 0.0f) {
            x = round_bf16_f32(softcap * tanhf(x / softcap));
        }
        out_scores[k] = x;
    }
}

static __global__ void inkling2_dspark_topk_argmax_kernel(
        const int32_t * __restrict__ cand_ids, // [1024]
        const float   * __restrict__ scores,   // [1024]
        int32_t       * __restrict__ out_best) {
    // 1 block of 256 threads
    const int tid = threadIdx.x;
    float best_v = -INFINITY;
    int32_t best_id = 0;
    for (int k = tid; k < 1024; k += 256) {
        const float v = scores[k];
        const int32_t cid = cand_ids[k];
        if (v > best_v || (v == best_v && cid < best_id)) {
            best_v = v;
            best_id = cid;
        }
    }
    __shared__ float s_v[256];
    __shared__ int32_t s_id[256];
    s_v[tid] = best_v;
    s_id[tid] = best_id;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            if (s_v[tid + off] > s_v[tid] || (s_v[tid + off] == s_v[tid] && s_id[tid + off] < s_id[tid])) {
                s_v[tid] = s_v[tid + off];
                s_id[tid] = s_id[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_best[0] = s_id[0];
}

// ─── 20. Dual-GPU Masked MoE Sum (Zero-589KB-Copy Split Join!) ──────────
template <bool OUT_F16, bool PEER_F16>
static __global__ void inkling2_moe_masked_sum_kernel(
        const float   * __restrict__ down,     // [H, K, T]
        const float   * __restrict__ wsel,     // [K, T]
        const int32_t * __restrict__ mask_sel, // [K, T]
        const void    * __restrict__ peer_sum, // nullptr or [H, T] (F16 or F32)
        void          * __restrict__ dst,      // [H, T] (F16 or F32)
        int H, int K, int wsel_ld, int msel_ld) {
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float s_w[16];
    __shared__ int32_t s_m[16];
    if (threadIdx.x < K) {
        s_w[threadIdx.x] = wsel[(size_t) t * wsel_ld + threadIdx.x];
        s_m[threadIdx.x] = mask_sel[(size_t) t * msel_ld + threadIdx.x];
    }
    __syncthreads();
    if (e >= H) return;

    float acc = 0.0f;
    const float * dp = down + (size_t) t * K * H + e;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k < K && s_m[k] >= 0) {
            acc += s_w[k] * dp[(size_t) k * H];
        }
    }
    if (peer_sum) {
        const size_t pidx = (size_t) t * H + e;
        if constexpr (PEER_F16) {
            acc += __half2float(reinterpret_cast<const half *>(peer_sum)[pidx]);
        } else {
            acc += reinterpret_cast<const float *>(peer_sum)[pidx];
        }
    }
    const size_t oidx = (size_t) t * H + e;
    if constexpr (OUT_F16) {
        reinterpret_cast<half *>(dst)[oidx] = __float2half(acc);
    } else {
        reinterpret_cast<float *>(dst)[oidx] = acc;
    }
}

// ─── 21. DSpark Fused Full-Vocab 201k Q8_0/F16 Markov Argmax (~45 us/step!) ─
template <bool IS_Q8>
static __global__ void inkling2_dspark_markov_argmax_stage1_kernel(
        const float * __restrict__ base_row,   // [V_pad]
        const float * __restrict__ latent,     // [R=256]
        const char  * __restrict__ fold_w,     // [R=256, V_pad] Q8_0 or F16
        size_t fw_nb1,
        const float * __restrict__ fold_b,     // [V_pad]
        float       * __restrict__ part_val,   // [256]
        int32_t     * __restrict__ part_id,    // [256]
        int n_vocab, int rank) {
    // grid = 256 blocks, block = (32, 8) = 2048 warps streaming contiguous chunks through MALL!
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..7
    const int tid  = warp * WARP_SIZE + lane; // 0..255

    __shared__ float s_lat[256];
    if (tid < rank) {
        s_lat[tid] = latent[tid];
    }
    __syncthreads();

    const int chunk = (n_vocab + 255) >> 8;
    const int v_beg = blockIdx.x * chunk;
    const int v_end = min(n_vocab, v_beg + chunk);

    float best_v = -INFINITY;
    int32_t best_id = 0;

    if constexpr (IS_Q8) {
        // 4 lanes per block_q8_0 (8 blocks = 32 lanes): each lane loads 8 bytes (4x uint16_t = 256 B/warp coalesced!)
        const int blk = lane >> 2;
        const int sub = lane & 3;
        const float * lp = s_lat + blk * 32 + sub * 8;
        const float l0 = lp[0], l1 = lp[1], l2 = lp[2], l3 = lp[3];
        const float l4 = lp[4], l5 = lp[5], l6 = lp[6], l7 = lp[7];
        for (int v = v_beg + warp; v < v_end; v += 8) {
            const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(fold_w + (size_t) v * fw_nb1) + blk;
            const uint16_t * hp = reinterpret_cast<const uint16_t *>(bp->qs + sub * 8);
            const uint16_t p0 = hp[0], p1 = hp[1], p2 = hp[2], p3 = hp[3];
            float acc = (float)(int8_t)(p0 & 0xFFu) * l0 + (float)(int8_t)(p0 >> 8) * l1
                      + (float)(int8_t)(p1 & 0xFFu) * l2 + (float)(int8_t)(p1 >> 8) * l3
                      + (float)(int8_t)(p2 & 0xFFu) * l4 + (float)(int8_t)(p2 >> 8) * l5
                      + (float)(int8_t)(p3 & 0xFFu) * l6 + (float)(int8_t)(p3 >> 8) * l7;
            const float dot = warp_reduce_sum(acc * __half2float(bp->d));
            if (lane == 0) {
                const float sc = round_bf16_f32(base_row[v] + dot + fold_b[v]);
                if (sc > best_v || (sc == best_v && v < best_id)) {
                    best_v = sc;
                    best_id = v;
                }
            }
        }
    } else {
        const float * lp = s_lat + lane * 8;
        const float l0 = lp[0], l1 = lp[1], l2 = lp[2], l3 = lp[3];
        const float l4 = lp[4], l5 = lp[5], l6 = lp[6], l7 = lp[7];
        for (int v = v_beg + warp; v < v_end; v += 8) {
            const half2 * h2p = reinterpret_cast<const half2 *>(fold_w + (size_t) v * fw_nb1) + lane * 4;
            const float2 f0 = __half22float2(h2p[0]);
            const float2 f1 = __half22float2(h2p[1]);
            const float2 f2 = __half22float2(h2p[2]);
            const float2 f3 = __half22float2(h2p[3]);
            float acc = f0.x * l0 + f0.y * l1 + f1.x * l2 + f1.y * l3
                      + f2.x * l4 + f2.y * l5 + f3.x * l6 + f3.y * l7;
            const float dot = warp_reduce_sum(acc);
            if (lane == 0) {
                const float sc = round_bf16_f32(base_row[v] + dot + fold_b[v]);
                if (sc > best_v || (sc == best_v && v < best_id)) {
                    best_v = sc;
                    best_id = v;
                }
            }
        }
    }

    __shared__ float s_wv[8];
    __shared__ int32_t s_wi[8];
    if (lane == 0) {
        s_wv[warp] = best_v;
        s_wi[warp] = best_id;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float bv = s_wv[0];
        int32_t bi = s_wi[0];
        for (int w = 1; w < 8; ++w) {
            if (s_wv[w] > bv || (s_wv[w] == bv && s_wi[w] < bi)) {
                bv = s_wv[w];
                bi = s_wi[w];
            }
        }
        part_val[blockIdx.x] = bv;
        part_id[blockIdx.x]  = bi;
    }
}

static __global__ void inkling2_dspark_markov_argmax_stage2_kernel(
        const float   * __restrict__ part_val, // [256]
        const int32_t * __restrict__ part_id,  // [256]
        int32_t       * __restrict__ out_best) {
    const int tid = threadIdx.x;
    __shared__ float s_v[256];
    __shared__ int32_t s_i[256];
    s_v[tid] = part_val[tid];
    s_i[tid] = part_id[tid];
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            if (s_v[tid + off] > s_v[tid] || (s_v[tid + off] == s_v[tid] && s_i[tid + off] < s_i[tid])) {
                s_v[tid] = s_v[tid + off];
                s_i[tid] = s_i[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_best[0] = s_i[0];
}

// ─── 22. Fused SConv + K-clip/V-RMSNorm + Direct KV Cache Write (4-Warp LDS-Tiled!) ──
template <bool KV_Q8_0>
static __global__ void inkling2_sconv_kv_write_kernel(
        const float   * __restrict__ x, size_t x_stride,
        const float   * __restrict__ w_static, // [4, Hk*D]
        const float   * __restrict__ state_in,
        float         * __restrict__ state_out,
        const float   * __restrict__ dyn_taps, size_t dyn_stride,
        char          * __restrict__ kv_cache, size_t kv_nb1, size_t kv_nb2,
        const int32_t * __restrict__ pos_buf,
        int head_dim, int n_head_kv, int n_tokens, int kv_cap,
        int state_stride, int state_chan_offset,
        int is_v, int swa_window, float threshold, float eps, int write_mode) {
    // 4 warps (128 threads = 1 thread per head_dim channel!) per KV head:
    // Stage A: all 128 channels run conv recurrence in parallel in registers -> 4 KB LDS tile (8 tokens)
    // Stage B: 4 warps quantize/write tokens in parallel (4x higher occupancy & zero serial bottleneck!)
    const int h    = blockIdx.x;
    const int d    = threadIdx.x; // 0..127
    const int warp = d >> 5;      // 0..3
    const int lane = d & 31;      // 0..31
    const int c    = h * head_dim + d;
    const int sc   = state_chan_offset + c;
    const float scale = dyn_taps ? (1.0f / 3.0f) : rsqrtf(5.0f);

    const float * st = state_in + (size_t) sc * 3;
    float h0 = st[0], h1 = st[1], h2 = st[2];
    const float * wc = w_static + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];

    __shared__ float s_val[8][128];

    for (int t0 = 0; t0 < n_tokens; t0 += 8) {
        const int tile_n = min(8, n_tokens - t0);
        for (int dt = 0; dt < tile_n; ++dt) {
            const int t = t0 + dt;
            const float cur = x[(size_t) t * x_stride + c];
            float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
            if (dyn_taps) {
                const float * dp = dyn_taps + (size_t) t * dyn_stride + h * 4;
                acc += dp[0] * cur + dp[1] * h2 + dp[2] * h1 + dp[3] * h0;
            }
            s_val[dt][d] = (cur + acc) * scale;
            h0 = h1; h1 = h2; h2 = cur;
            if (write_mode == 2 && state_out) {
                float * sr = state_out + ((size_t) t * state_stride + sc) * 3;
                sr[0] = h0; sr[1] = h1; sr[2] = h2;
            }
        }
        __syncthreads();

        for (int dt = warp; dt < tile_n; dt += 4) {
            const int t = t0 + dt;
            const int32_t pos = pos_buf[t];
            if (pos >= 0) {
                const int slot = (swa_window > 0) ? (pos % kv_cap)
                               : (swa_window < 0) ? (pos + swa_window)
                               : pos;
                if (slot >= 0 && slot < kv_cap) {
                    const float v0 = s_val[dt][lane +  0];
                    const float v1 = s_val[dt][lane + 32];
                    const float v2 = s_val[dt][lane + 64];
                    const float v3 = s_val[dt][lane + 96];
                    const float ss = warp_reduce_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
                    const float mult = is_v
                        ? rsqrtf(ss / 128.0f + eps)
                        : fminf(1.0f, threshold / (sqrtf(ss) + eps));
                    char * dst_row = kv_cache + (size_t) h * kv_nb2 + (size_t) slot * kv_nb1;
                    if constexpr (KV_Q8_0) {
                        block_q8_0 * bp = reinterpret_cast<block_q8_0 *>(dst_row);
                        const float vs[4] = { v0 * mult, v1 * mult, v2 * mult, v3 * mult };
                        #pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            const float amax = warp_reduce_max(fabsf(vs[b]));
                            const float d_q = amax / 127.0f;
                            bp[b].qs[lane] = (amax == 0.0f) ? 0 : (int8_t) __float2int_rn(vs[b] / d_q);
                            if (lane == 0) bp[b].d = __float2half(d_q);
                        }
                    } else {
                        half * hp = reinterpret_cast<half *>(dst_row);
                        hp[lane +  0] = __float2half(v0 * mult);
                        hp[lane + 32] = __float2half(v1 * mult);
                        hp[lane + 64] = __float2half(v2 * mult);
                        hp[lane + 96] = __float2half(v3 * mult);
                    }
                }
            }
        }
        if (t0 + 8 < n_tokens) __syncthreads();
    }

    if (write_mode == 1 && state_out) {
        float * so = state_out + (size_t) sc * 3;
        so[0] = h0; so[1] = h1; so[2] = h2;
    }
}

// Parallel O(1) time-tiled SConv + K-clip/V-RMSNorm + Direct KV Cache Write (Prefill T > 8, write_mode <= 1):
// Launches grid = (H_kv, ceil(T/8)) across all 64 CUs instead of 8 serial blocks (~30x faster prefill!).
template <bool KV_Q8_0>
static __global__ void inkling2_sconv_kv_write_parallel_kernel(
        const float   * __restrict__ x, size_t x_stride,
        const float   * __restrict__ w_static,
        const float   * __restrict__ state_in,
        float         * __restrict__ state_out,
        const float   * __restrict__ dyn_taps, size_t dyn_stride,
        char          * __restrict__ kv_cache, size_t kv_nb1, size_t kv_nb2,
        const int32_t * __restrict__ pos_buf,
        int head_dim, int n_head_kv, int n_tokens, int kv_cap,
        int state_chan_offset,
        int is_v, int swa_window, float threshold, float eps, int write_mode) {
    const int h    = blockIdx.x;
    const int t0   = blockIdx.y * 8;
    const int tile_n = min(8, n_tokens - t0);
    const int d    = threadIdx.x; // 0..127
    const int warp = d >> 5;      // 0..3
    const int lane = d & 31;      // 0..31
    const int c    = h * head_dim + d;
    const int sc   = state_chan_offset + c;
    const float scale = dyn_taps ? (1.0f / 3.0f) : rsqrtf(5.0f);

    const float * st = state_in + (size_t) sc * 3;
    float h0 = (t0 >= 3) ? x[(size_t) (t0 - 3) * x_stride + c] : st[t0];
    float h1 = (t0 >= 2) ? x[(size_t) (t0 - 2) * x_stride + c] : st[t0 + 1];
    float h2 = (t0 >= 1) ? x[(size_t) (t0 - 1) * x_stride + c] : st[t0 + 2];
    const float * wc = w_static + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];

    __shared__ float s_val[8][128];

    for (int dt = 0; dt < tile_n; ++dt) {
        const int t = t0 + dt;
        const float cur = x[(size_t) t * x_stride + c];
        float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
        if (dyn_taps) {
            const float * dp = dyn_taps + (size_t) t * dyn_stride + h * 4;
            acc += dp[0] * cur + dp[1] * h2 + dp[2] * h1 + dp[3] * h0;
        }
        s_val[dt][d] = (cur + acc) * scale;
        h0 = h1; h1 = h2; h2 = cur;
    }
    __syncthreads();

    for (int dt = warp; dt < tile_n; dt += 4) {
        const int t = t0 + dt;
        const int32_t pos = pos_buf[t];
        if (pos >= 0) {
            const int slot = (swa_window > 0) ? (pos % kv_cap)
                           : (swa_window < 0) ? (pos + swa_window)
                           : pos;
            if (slot >= 0 && slot < kv_cap) {
                const float v0 = s_val[dt][lane +  0];
                const float v1 = s_val[dt][lane + 32];
                const float v2 = s_val[dt][lane + 64];
                const float v3 = s_val[dt][lane + 96];
                const float ss = warp_reduce_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
                const float mult = is_v
                    ? rsqrtf(ss / 128.0f + eps)
                    : fminf(1.0f, threshold / (sqrtf(ss) + eps));
                char * dst_row = kv_cache + (size_t) h * kv_nb2 + (size_t) slot * kv_nb1;
                if constexpr (KV_Q8_0) {
                    block_q8_0 * bp = reinterpret_cast<block_q8_0 *>(dst_row);
                    const float vs[4] = { v0 * mult, v1 * mult, v2 * mult, v3 * mult };
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const float amax = warp_reduce_max(fabsf(vs[b]));
                        const float d_q = amax / 127.0f;
                        bp[b].qs[lane] = (amax == 0.0f) ? 0 : (int8_t) __float2int_rn(vs[b] / d_q);
                        if (lane == 0) bp[b].d = __float2half(d_q);
                    }
                } else {
                    half * hp = reinterpret_cast<half *>(dst_row);
                    hp[lane +  0] = __float2half(v0 * mult);
                    hp[lane + 32] = __float2half(v1 * mult);
                    hp[lane + 64] = __float2half(v2 * mult);
                    hp[lane + 96] = __float2half(v3 * mult);
                }
            }
        }
    }

    (void) state_out;
    (void) write_mode;
}

// ─── 23. Fused R L2-Clip (+ scale + log-tau) + Relative-Position Table ──
static __global__ void inkling2_clip_r_rel_table_kernel(
        const float   * __restrict__ r_raw, size_t r_stride,
        const float   * __restrict__ r_clip_w, // nullptr or [d_rel, n_head]
        const int32_t * __restrict__ pos_buf,  // nullptr or [T]
        const float   * __restrict__ rel_bias, // [d_rel, rel_extent]
        const float   * __restrict__ sink, size_t sink_stride,
        float         * __restrict__ dst,      // [rel_extent+1, n_head, T]
        int d_rel, int rel_extent, int n_head,
        float threshold, float eps, int log_floor, float log_alpha) {
    const int h   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float s_r[32];
    if (tid < WARP_SIZE) {
        float v = 0.0f;
        if (tid < d_rel) {
            v = r_raw[(size_t) t * r_stride + (size_t) h * d_rel + tid];
            if (r_clip_w) v *= r_clip_w[(size_t) h * d_rel + tid];
        }
        const float ss = warp_reduce_sum(v * v);
        float tau = 1.0f;
        if (pos_buf && log_floor > 0 && log_alpha > 0.0f) {
            const float ratio = (float)(pos_buf[t] + 1) / (float) log_floor;
            if (ratio > 1.0f) tau = 1.0f + log_alpha * logf(ratio);
        }
        const float mult = fminf(1.0f, threshold / (sqrtf(ss) + eps)) * tau;
        if (tid < d_rel) s_r[tid] = v * mult;
    }
    __syncthreads();

    const float s_val = sink[(size_t) t * sink_stride + h];
    float * op = dst + ((size_t) t * n_head + h) * (rel_extent + 1);
    if (d_rel == 16) {
        const float4 r0 = reinterpret_cast<const float4 *>(s_r)[0];
        const float4 r1 = reinterpret_cast<const float4 *>(s_r)[1];
        const float4 r2 = reinterpret_cast<const float4 *>(s_r)[2];
        const float4 r3 = reinterpret_cast<const float4 *>(s_r)[3];
        const float4 * rb4 = reinterpret_cast<const float4 *>(rel_bias);
        for (int dist = tid; dist < rel_extent; dist += blockDim.x) {
            const float4 * b = rb4 + (size_t) dist * 4;
            const float4 b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
            const float dot = b0.x * r0.x + b0.y * r0.y + b0.z * r0.z + b0.w * r0.w
                            + b1.x * r1.x + b1.y * r1.y + b1.z * r1.z + b1.w * r1.w
                            + b2.x * r2.x + b2.y * r2.y + b2.z * r2.z + b2.w * r2.w
                            + b3.x * r3.x + b3.y * r3.y + b3.z * r3.z + b3.w * r3.w;
            op[dist] = dot - s_val;
        }
    } else {
        for (int dist = tid; dist < rel_extent; dist += blockDim.x) {
            const float * rb = rel_bias + (size_t) dist * d_rel;
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                if (k < d_rel) dot += rb[k] * s_r[k];
            }
            op[dist] = dot - s_val;
        }
    }
    if (tid == 0) op[rel_extent] = -s_val;
}

// ─── 24. 64-CU Weight-Stationary Q8_0 Router + Zero-Spill Packed Sigmoid Top-K (+ Split!) ─
template <int MAX_T>
static __global__ void inkling2_router_q8_gemv_kernel(
        const char  * __restrict__ w_q8, size_t w_nb1, // Q8_0 [n_embd, n_exp]
        const float * __restrict__ cur,  size_t cur_ld, // F32 [n_embd, T]
        float       * __restrict__ out_logits,          // F32 [n_exp, T]
        int n_embd, int n_exp, int T) {
    const int e    = blockIdx.x * 2 + threadIdx.y;
    const int lane = threadIdx.x;
    if (e >= n_exp) return;

    const int n_blk = n_embd >> 5;
    const int boff  = lane >> 2; // 0..7
    const int sub   = lane & 3;  // 0..3
    const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(w_q8 + (size_t) e * w_nb1);

    float dot[MAX_T];
    #pragma unroll
    for (int t = 0; t < MAX_T; ++t) dot[t] = 0.0f;

    for (int b0 = 0; b0 < n_blk; b0 += 8) {
        const int b = b0 + boff;
        if (b < n_blk) {
            const float d = __half2float(bp[b].d);
            const uint16_t * hp = reinterpret_cast<const uint16_t *>(bp[b].qs + sub * 8);
            const uint16_t p0 = hp[0], p1 = hp[1], p2 = hp[2], p3 = hp[3];
            const float q0 = d * (float)(int8_t)(p0 & 0xFFu), q1 = d * (float)(int8_t)(p0 >> 8);
            const float q2 = d * (float)(int8_t)(p1 & 0xFFu), q3 = d * (float)(int8_t)(p1 >> 8);
            const float q4 = d * (float)(int8_t)(p2 & 0xFFu), q5 = d * (float)(int8_t)(p2 >> 8);
            const float q6 = d * (float)(int8_t)(p3 & 0xFFu), q7 = d * (float)(int8_t)(p3 >> 8);
            const int xoff = b * 32 + sub * 8;
            #pragma unroll
            for (int t = 0; t < MAX_T; ++t) {
                if (t < T) {
                    const float4 * x4 = reinterpret_cast<const float4 *>(cur + (size_t) t * cur_ld + xoff);
                    const float4 a = x4[0];
                    const float4 c = x4[1];
                    dot[t] += q0 * a.x + q1 * a.y + q2 * a.z + q3 * a.w
                            + q4 * c.x + q5 * c.y + q6 * c.z + q7 * c.w;
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        if (t < T) {
            const float s = warp_reduce_sum(dot[t]);
            if (lane == 0) out_logits[(size_t) t * n_exp + e] = s;
        }
    }
}

// Register-only zero-spill warp Top-8 helper for n_exp == 128, top_k == 8 (eliminates scratch VRAM spills!)
static __device__ __forceinline__ void inkling2_warp_top8_128(
        const float * __restrict__ lp,
        const float * __restrict__ bias,
        int32_t * op_ids,
        int32_t * op_wts,
        float route_scale, float eps, int lane) {
    float p0 = sigmoid_f32(lp[lane +  0]);
    float p1 = sigmoid_f32(lp[lane + 32]);
    float p2 = sigmoid_f32(lp[lane + 64]);
    float p3 = sigmoid_f32(lp[lane + 96]);
    float s0 = bias ? (p0 + bias[lane +  0]) : p0;
    float s1 = bias ? (p1 + bias[lane + 32]) : p1;
    float s2 = bias ? (p2 + bias[lane + 64]) : p2;
    float s3 = bias ? (p3 + bias[lane + 96]) : p3;

    float my_chosen_p = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float bv = s0, bp = p0; int bi = lane;
        if (s1 > bv) { bv = s1; bp = p1; bi = lane + 32; }
        if (s2 > bv) { bv = s2; bp = p2; bi = lane + 64; }
        if (s3 > bv) { bv = s3; bp = p3; bi = lane + 96; }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = i2_shfl_xor(bv, off, WARP_SIZE);
            const float op = i2_shfl_xor(bp, off, WARP_SIZE);
            const int   oi = i2_shfl_xor(bi, off, WARP_SIZE);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov; bp = op; bi = oi;
            }
        }
        if (lane == k) {
            op_ids[k] = bi;
            my_chosen_p = bp;
        }
        if (bi == lane)           s0 = -INFINITY;
        else if (bi == lane + 32) s1 = -INFINITY;
        else if (bi == lane + 64) s2 = -INFINITY;
        else if (bi == lane + 96) s3 = -INFINITY;
    }
    const float sum_p = warp_reduce_sum(my_chosen_p);
    if (lane < 8) {
        const float inv = route_scale / fmaxf(sum_p, eps);
        op_wts[lane] = __float_as_int(my_chosen_p * inv);
    }
}

static __global__ void inkling2_sigmoid_topk_packed_kernel(
        const float   * __restrict__ logits, // F32 [n_exp, T]
        const float   * __restrict__ bias,   // nullptr or F32 [n_exp]
        const int32_t * __restrict__ flags,  // nullptr or I32 [T] (bit 1 = FLAG_PADDING -> reuse row 0 experts)
        int32_t       * __restrict__ dst,    // I32 [top_k, T, 2]
        int n_exp, int top_k, int n_tokens, float route_scale, float eps) {
    const int t    = blockIdx.x;
    const int lane = threadIdx.x;
    const int src_t = (flags && (flags[t] & 2) != 0) ? 0 : t;
    const float * lp = logits + (size_t) src_t * n_exp;
    int32_t * op_ids = dst + (size_t) t * top_k;
    int32_t * op_wts = dst + (size_t) n_tokens * top_k + (size_t) t * top_k;

    if (n_exp == 128 && top_k == 8) {
        inkling2_warp_top8_128(lp, bias, op_ids, op_wts, route_scale, eps, lane);
        return;
    }

    float sc[8], pr[8];
    int   id[8];
    int   cnt = 0;
    for (int e = lane; e < n_exp && cnt < 8; e += WARP_SIZE) {
        const float p = sigmoid_f32(lp[e]);
        pr[cnt] = p;
        sc[cnt] = bias ? (p + bias[e]) : p;
        id[cnt] = e;
        ++cnt;
    }

    float chosen_p[16];
    float sum_p = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        float best_v = -INFINITY;
        float best_p = 0.0f;
        int   best_i = n_exp;
        int   best_j = -1;
        for (int j = 0; j < cnt; ++j) {
            if (sc[j] > best_v || (sc[j] == best_v && id[j] < best_i)) {
                best_v = sc[j];
                best_p = pr[j];
                best_i = id[j];
                best_j = j;
            }
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            const float ov   = i2_shfl_xor(best_v, off, WARP_SIZE);
            const float op_v = i2_shfl_xor(best_p, off, WARP_SIZE);
            const int   oi   = i2_shfl_xor(best_i, off, WARP_SIZE);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_p = op_v;
                best_i = oi;
            }
        }
        if (lane == 0) {
            op_ids[k] = best_i;
            chosen_p[k] = best_p;
            sum_p += best_p;
        }
        if (best_j >= 0 && id[best_j] == best_i) {
            sc[best_j] = -INFINITY;
        }
    }
    if (lane == 0) {
        const float inv = route_scale / fmaxf(sum_p, eps);
        for (int k = 0; k < top_k; ++k) {
            op_wts[k] = __float_as_int(chosen_p[k] * inv);
        }
    }
}

// Fused 1-block Zero-Spill Top-8 + Dual-GPU O(1) Histogram/Ballot Split (supports up to T=64!):
// Emits I32 [8, T, 3]: slice 0 = hot_sel, slice 1 = cold_sel, slice 2 = __float_as_int(wsel).
static __global__ void inkling2_sigmoid_topk_split_kernel(
        const float   * __restrict__ logits, // F32 [128, T]
        const float   * __restrict__ bias,   // nullptr or F32 [128]
        const int32_t * __restrict__ flags,  // nullptr or I32 [T] (bit 1 = FLAG_PADDING)
        int32_t       * __restrict__ dst,    // I32 [8, T, 3]
        int n_tokens, int target_pct, float route_scale, float eps) {
    __shared__ int32_t s_ids[64 * 8];
    __shared__ int32_t s_freq[128];
    __shared__ int32_t s_hist[65];
    __shared__ int32_t s_warp_cnt[8];
    __shared__ uint8_t s_is_hot[128];
    __shared__ int32_t s_cut_f;
    __shared__ int32_t s_cut_rem;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int total  = n_tokens * 8;

    if (tid < 128) {
        s_freq[tid] = 0;
        s_is_hot[tid] = 0;
    }
    if (tid < 65) s_hist[tid] = 0;

    for (int row = warp; row < n_tokens; row += nwarps) {
        const int src_row = (flags && (flags[row] & 2) != 0) ? 0 : row;
        const float * lp = logits + (size_t) src_row * 128;
        int32_t * op_wts = dst + (size_t) 2 * total + (size_t) row * 8;
        inkling2_warp_top8_128(lp, bias, s_ids + row * 8, op_wts, route_scale, eps, lane);
    }
    __syncthreads();

    for (int i = tid; i < total; i += blockDim.x) {
        const int32_t e = s_ids[i];
        if (e >= 0 && e < 128) atomicAdd(&s_freq[e], 1);
    }
    __syncthreads();

    const int f = (tid < 128) ? min(s_freq[tid], 64) : 0;
    if (f > 0) atomicAdd(&s_hist[f], 1);
    __syncthreads();

    if (tid == 0) {
        int n_unique = 0;
        for (int b = 1; b < 65; ++b) n_unique += s_hist[b];
        int n_hot = (n_unique * target_pct + 99) / 100;
        if (n_hot > n_unique) n_hot = n_unique;
        int cut_f = 65, cut_rem = 0;
        if (n_hot > 0) {
            int acc = 0;
            for (int b = 64; b >= 1; --b) {
                if (s_hist[b] == 0) continue;
                if (acc + s_hist[b] <= n_hot) {
                    acc += s_hist[b];
                    cut_f = b;
                    cut_rem = 0;
                    if (acc == n_hot) break;
                } else {
                    cut_f = b;
                    cut_rem = n_hot - acc;
                    break;
                }
            }
        }
        s_cut_f = cut_f;
        s_cut_rem = cut_rem;
    }
    __syncthreads();

    const int cut_f   = s_cut_f;
    const int cut_rem = s_cut_rem;
    const bool is_tie = (tid < 128 && f > 0 && f == cut_f && cut_rem > 0);
    const unsigned long long mask = i2_ballot(is_tie);
    if (lane == 0 && warp < 4) s_warp_cnt[warp] = __popcll(mask);
    __syncthreads();

    if (tid < 128 && f > 0) {
        if (f > cut_f || (f == cut_f && cut_rem == 0)) {
            s_is_hot[tid] = 1;
        } else if (is_tie) {
            int rank = 0;
            for (int w = 0; w < warp; ++w) rank += s_warp_cnt[w];
            rank += __popcll(mask & ((1ULL << lane) - 1ULL));
            if (rank < cut_rem) s_is_hot[tid] = 1;
        }
    }
    __syncthreads();

    int32_t * hot_out  = dst;
    int32_t * cold_out = dst + total;
    for (int i = tid; i < total; i += blockDim.x) {
        const int32_t e = s_ids[i];
        const bool hot = (e >= 0 && e < 128 && s_is_hot[e] != 0);
        hot_out[i]  = hot ? e : -1;
        cold_out[i] = (!hot && e >= 0) ? e : -1;
    }
}

// ─── 25. Fused 64-CU Loop Controller Pre-Norm + Q8_0 Down-Projection ────
// Replaces loop_ctrl_pre + quantize_mmq + 2-CU ctrl_down (28.8 us) with 1 64-CU kernel (~4.5 us)!
static __global__ void inkling2_fused_loop_ctrl_pre_down_kernel(
        const float * __restrict__ cur, size_t cur_r, size_t cur_g,
        const float * __restrict__ msg, size_t msg_r, size_t msg_g,
        const char  * __restrict__ w_q8, size_t w_nb1, size_t w_nb2,
        float       * __restrict__ dst, size_t dst_r, size_t dst_g,
        int n_embd, int rank, float eps) {
    // grid = (T, G, rank/16), block = (32, 16) = 512 threads = 16 warps (1 warp per output row!)
    const int t     = blockIdx.x;
    const int g     = blockIdx.y;
    const int r_blk = blockIdx.z;
    const int lane  = threadIdx.x; // 0..31
    const int warp  = threadIdx.y; // 0..15
    const int tid   = warp * WARP_SIZE + lane; // 0..511

    __shared__ float s_cpre[6144]; // 2 * n_embd (4608 floats = 18 KB)
    __shared__ float s_red[2][16];

    const float * cp = cur + (size_t) t * cur_r + (size_t) g * cur_g;
    const float * mp = msg + (size_t) t * msg_r + (size_t) g * msg_g;

    float ss_c = 0.0f, ss_m = 0.0f;
    for (int e = tid; e < n_embd; e += 512) {
        const float cv = cp[e];
        const float mv = mp[e];
        s_cpre[e] = cv;
        s_cpre[n_embd + e] = mv;
        ss_c += cv * cv;
        ss_m += mv * mv;
    }
    ss_c = warp_reduce_sum(ss_c);
    ss_m = warp_reduce_sum(ss_m);
    if (lane == 0) {
        s_red[0][warp] = ss_c;
        s_red[1][warp] = ss_m;
    }
    __syncthreads();
    if (warp == 0) {
        float vc = (lane < 16) ? s_red[0][lane] : 0.0f;
        float vm = (lane < 16) ? s_red[1][lane] : 0.0f;
        vc = warp_reduce_sum(vc);
        vm = warp_reduce_sum(vm);
        if (lane == 0) {
            s_red[0][0] = rsqrtf(vc / (float) n_embd + eps);
            s_red[1][0] = rsqrtf(vm / (float) n_embd + eps);
        }
    }
    __syncthreads();

    const float irms_c = s_red[0][0];
    const float irms_m = s_red[1][0];
    for (int e = tid; e < n_embd; e += 512) {
        s_cpre[e]          *= irms_c;
        s_cpre[n_embd + e] *= irms_m;
    }
    __syncthreads();

    const int row = r_blk * 16 + warp;
    if (row >= rank) return;

    const int n_blk = (2 * n_embd) >> 5; // 144 for n_embd=2304
    const int boff  = lane >> 2;         // 0..7
    const int sub   = lane & 3;          // 0..3
    const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(
        w_q8 + (size_t) g * w_nb2 + (size_t) row * w_nb1);

    float dot = 0.0f;
    for (int b0 = 0; b0 < n_blk; b0 += 8) {
        const int b = b0 + boff;
        if (b < n_blk) {
            const uint16_t * hp = reinterpret_cast<const uint16_t *>(bp[b].qs + sub * 8);
            const uint16_t p0 = hp[0], p1 = hp[1], p2 = hp[2], p3 = hp[3];
            const float * lx = s_cpre + b * 32 + sub * 8;
            float acc = (float)(int8_t)(p0 & 0xFFu) * lx[0] + (float)(int8_t)(p0 >> 8) * lx[1]
                      + (float)(int8_t)(p1 & 0xFFu) * lx[2] + (float)(int8_t)(p1 >> 8) * lx[3]
                      + (float)(int8_t)(p2 & 0xFFu) * lx[4] + (float)(int8_t)(p2 >> 8) * lx[5]
                      + (float)(int8_t)(p3 & 0xFFu) * lx[6] + (float)(int8_t)(p3 >> 8) * lx[7];
            dot += acc * __half2float(bp[b].d);
        }
    }
    dot = warp_reduce_sum(dot);
    if (lane == 0) {
        dst[(size_t) g * dst_g + (size_t) t * dst_r + row] = dot;
    }
}

// ─── 26. Fused Dual-GPU Dynamic Split + Hot/Cold Route Masking (1 Launch!) ─
// Replaces dynamic_split_lut + moe_hot_routes(hot) + moe_hot_routes(cold) (3 launches -> 1 launch!)
static __global__ void inkling2_dynamic_split_routes_kernel(
        const int32_t * __restrict__ sel, int sel_ld,
        int32_t * __restrict__ out_routes, // [K, T, 2]: slice 0 = hot_sel, slice 1 = cold_sel
        int n_used, int n_tokens, int n_expert, int target_pct) {
    __shared__ int32_t s_freq[256];
    __shared__ int32_t s_hist[32];
    __shared__ int32_t s_warp_cnt[8];
    __shared__ uint8_t s_is_hot[256];
    __shared__ int32_t s_cut_f;
    __shared__ int32_t s_cut_rem;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;

    if (tid < n_expert) {
        s_freq[tid] = 0;
        s_is_hot[tid] = 0;
    }
    if (tid < 32) s_hist[tid] = 0;
    __syncthreads();

    const int total = n_used * n_tokens;
    for (int i = tid; i < total; i += blockDim.x) {
        const int t = i / n_used;
        const int k = i - t * n_used;
        const int32_t e = sel[t * sel_ld + k];
        if (e >= 0 && e < n_expert) {
            atomicAdd(&s_freq[e], 1);
        }
    }
    __syncthreads();

    const int f = (tid < n_expert) ? min(s_freq[tid], 31) : 0;
    if (f > 0) atomicAdd(&s_hist[f], 1);
    __syncthreads();

    if (tid == 0) {
        int n_unique = 0;
        for (int b = 1; b < 32; ++b) n_unique += s_hist[b];
        int n_hot = (n_unique * target_pct + 99) / 100;
        if (n_hot > n_unique) n_hot = n_unique;
        int cut_f = 32, cut_rem = 0;
        if (n_hot > 0) {
            int acc = 0;
            for (int b = 31; b >= 1; --b) {
                if (s_hist[b] == 0) continue;
                if (acc + s_hist[b] <= n_hot) {
                    acc += s_hist[b];
                    cut_f = b;
                    cut_rem = 0;
                    if (acc == n_hot) break;
                } else {
                    cut_f = b;
                    cut_rem = n_hot - acc;
                    break;
                }
            }
        }
        s_cut_f = cut_f;
        s_cut_rem = cut_rem;
    }
    __syncthreads();

    const int cut_f   = s_cut_f;
    const int cut_rem = s_cut_rem;
    const bool is_tie = (tid < n_expert && f > 0 && f == cut_f && cut_rem > 0);
    const unsigned long long mask = i2_ballot(is_tie);
    if (lane == 0) s_warp_cnt[warp] = __popcll(mask);
    __syncthreads();

    if (tid < n_expert && f > 0) {
        if (f > cut_f || (f == cut_f && cut_rem == 0)) {
            s_is_hot[tid] = 1;
        } else if (is_tie) {
            int rank = 0;
            for (int w = 0; w < warp; ++w) rank += s_warp_cnt[w];
            rank += __popcll(mask & ((1ULL << lane) - 1ULL));
            if (rank < cut_rem) s_is_hot[tid] = 1;
        }
    }
    __syncthreads();

    int32_t * hot_out  = out_routes;
    int32_t * cold_out = out_routes + total;
    for (int i = tid; i < total; i += blockDim.x) {
        const int t = i / n_used;
        const int k = i - t * n_used;
        const int32_t e = sel[t * sel_ld + k];
        const bool hot = (e >= 0 && e < n_expert && s_is_hot[e] != 0);
        hot_out[i]  = hot ? e : -1;
        cold_out[i] = (!hot && e >= 0) ? e : -1;
    }
}

// ─── 27. Fused K+V SConv + K-clip/V-RMSNorm + Direct KV Cache Write (1 Launch for Both K & V!) ─
template <bool KV_Q8_0>
static __global__ void inkling2_sconv_kv_write_kv_kernel(
        const float   * __restrict__ x_k, size_t xk_stride,
        const float   * __restrict__ x_v, size_t xv_stride,
        const float   * __restrict__ wk_static,
        const float   * __restrict__ wv_static,
        const float   * __restrict__ state_in,
        float         * __restrict__ state_out,
        char          * __restrict__ k_cache, size_t k_nb1, size_t k_nb2,
        char          * __restrict__ v_cache, size_t v_nb1, size_t v_nb2,
        const int32_t * __restrict__ pos_buf,
        int head_dim, int n_head_kv, int n_tokens, int kv_cap,
        int state_stride, int swa_window, float threshold, float eps, int write_mode) {
    const int is_v = (blockIdx.x >= n_head_kv) ? 1 : 0;
    const int h    = is_v ? (blockIdx.x - n_head_kv) : blockIdx.x;
    const int d    = threadIdx.x; // 0..127
    const int warp = d >> 5;
    const int lane = d & 31;
    const int c    = h * head_dim + d;
    const int sc   = (is_v ? n_head_kv * head_dim : 0) + c;
    const float scale = rsqrtf(5.0f);

    const float * x        = is_v ? x_v : x_k;
    const size_t  x_stride = is_v ? xv_stride : xk_stride;
    const float * w_static = is_v ? wv_static : wk_static;
    char        * kv_cache = is_v ? v_cache : k_cache;
    const size_t  kv_nb1   = is_v ? v_nb1 : k_nb1;
    const size_t  kv_nb2   = is_v ? v_nb2 : k_nb2;

    const float * st = state_in + (size_t) sc * 3;
    float h0 = st[0], h1 = st[1], h2 = st[2];
    const float * wc = w_static + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];

    __shared__ float s_val[8][128];

    for (int t0 = 0; t0 < n_tokens; t0 += 8) {
        const int tile_n = min(8, n_tokens - t0);
        for (int dt = 0; dt < tile_n; ++dt) {
            const int t = t0 + dt;
            const float cur = x[(size_t) t * x_stride + c];
            const float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
            s_val[dt][d] = (cur + acc) * scale;
            h0 = h1; h1 = h2; h2 = cur;
            if (write_mode == 2 && state_out) {
                float * sr = state_out + ((size_t) t * state_stride + sc) * 3;
                sr[0] = h0; sr[1] = h1; sr[2] = h2;
            }
        }
        __syncthreads();

        for (int dt = warp; dt < tile_n; dt += 4) {
            const int t = t0 + dt;
            const int32_t pos = pos_buf[t];
            if (pos >= 0) {
                const int slot = (swa_window > 0) ? (pos % kv_cap)
                               : (swa_window < 0) ? (pos + swa_window)
                               : pos;
                if (slot >= 0 && slot < kv_cap) {
                    const float v0 = s_val[dt][lane +  0];
                    const float v1 = s_val[dt][lane + 32];
                    const float v2 = s_val[dt][lane + 64];
                    const float v3 = s_val[dt][lane + 96];
                    const float ss = warp_reduce_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
                    const float mult = is_v
                        ? rsqrtf(ss / 128.0f + eps)
                        : fminf(1.0f, threshold / (sqrtf(ss) + eps));
                    char * dst_row = kv_cache + (size_t) h * kv_nb2 + (size_t) slot * kv_nb1;
                    if constexpr (KV_Q8_0) {
                        block_q8_0 * bp = reinterpret_cast<block_q8_0 *>(dst_row);
                        const float vs[4] = { v0 * mult, v1 * mult, v2 * mult, v3 * mult };
                        #pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            const float amax = warp_reduce_max(fabsf(vs[b]));
                            const float d_q = amax / 127.0f;
                            bp[b].qs[lane] = (amax == 0.0f) ? 0 : (int8_t) __float2int_rn(vs[b] / d_q);
                            if (lane == 0) bp[b].d = __float2half(d_q);
                        }
                    } else {
                        half * hp = reinterpret_cast<half *>(dst_row);
                        hp[lane +  0] = __float2half(v0 * mult);
                        hp[lane + 32] = __float2half(v1 * mult);
                        hp[lane + 64] = __float2half(v2 * mult);
                        hp[lane + 96] = __float2half(v3 * mult);
                    }
                }
            }
        }
        if (t0 + 8 < n_tokens) __syncthreads();
    }

    if (write_mode == 1 && state_out) {
        float * so = state_out + (size_t) sc * 3;
        so[0] = h0; so[1] = h1; so[2] = h2;
    }
}

// ─── 28. Fused MoE Masked Sum (+ Peer Join) + FFN Post-Conv + Residual Add (1 Launch!) ─
template <bool PEER_F16>
static __global__ void inkling2_moe_masked_sum_sconv_add_kernel(
        const float   * __restrict__ down,     // [H, K, T]
        const float   * __restrict__ wsel,     // [K, T]
        const int32_t * __restrict__ mask_sel, // [K, T]
        const void    * __restrict__ peer_sum, // nullptr or [H, T]
        const float   * __restrict__ w_static, // [4, H]
        const float   * __restrict__ state_in,
        float         * __restrict__ state_out,
        const float   * __restrict__ addend,   size_t add_stride, // [H, T]
        float         * __restrict__ dst,      // [H, T]
        int H, int K, int T, int wsel_ld, int msel_ld,
        int state_stride, int state_chan_offset, int write_mode) {
    __shared__ float   s_w[8][8];
    __shared__ int32_t s_m[8][8];
    const int tid = threadIdx.x;
    const int tk_tot = T * K;
    if (tid < tk_tot) {
        const int t = tid / K;
        const int k = tid - t * K;
        s_w[t][k] = wsel[(size_t) t * wsel_ld + k];
        s_m[t][k] = mask_sel[(size_t) t * msel_ld + k];
    }
    __syncthreads();

    const int c = blockIdx.x * blockDim.x + tid;
    if (c >= H) return;

    const int sc = state_chan_offset + c;
    const float * st_in = state_in + (size_t) sc * 3;
    float h0 = st_in[0], h1 = st_in[1], h2 = st_in[2];
    const float * wc = w_static + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float scale = rsqrtf(5.0f);

    for (int t = 0; t < T; ++t) {
        float cur = 0.0f;
        const float * dp = down + (size_t) t * K * H + c;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k < K && s_m[t][k] >= 0) {
                cur += s_w[t][k] * dp[(size_t) k * H];
            }
        }
        if (peer_sum) {
            const size_t pidx = (size_t) t * H + c;
            if constexpr (PEER_F16) {
                cur += __half2float(reinterpret_cast<const half *>(peer_sum)[pidx]);
            } else {
                cur += reinterpret_cast<const float *>(peer_sum)[pidx];
            }
        }
        const float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
        dst[(size_t) t * H + c] = (cur + acc) * scale + addend[(size_t) t * add_stride + c];
        h0 = h1; h1 = h2; h2 = cur;
        if (write_mode == 2 && state_out) {
            float * sr = state_out + ((size_t) t * state_stride + sc) * 3;
            sr[0] = h0; sr[1] = h1; sr[2] = h2;
        }
    }

    if (write_mode == 1 && state_out) {
        float * so = state_out + (size_t) sc * 3;
        so[0] = h0; so[1] = h1; so[2] = h2;
    }
}

// ─── 29. DSpark Fused Markov RNN Step (replaces get_rows+concat+gemv+add+gate = 7 ops -> 1!) ─
static __global__ void inkling2_dspark_markov_rnn_step_kernel(
        const int32_t * __restrict__ prev_tok,      // [1]
        const float   * __restrict__ state,         // [R]
        const half    * __restrict__ markov_w1,     // [R, V] F16
        const half    * __restrict__ rnn_proj_step, // [2*R, 3*R] F16
        const float   * __restrict__ rnn_ctx_s,     // [3*R] F32
        float         * __restrict__ dst,           // [2*R]: first R = new_state, second R = latent
        int rank) {
    // grid = (rank/8 = 32 blocks), block = (32, 8) = 256 threads (1 warp per output channel c)
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * WARP_SIZE + lane;
    const int c    = blockIdx.x * 8 + warp;

    __shared__ float s_in[512]; // [state(256) | markov_w1[prev](256)]
    const int32_t tok = prev_tok[0];
    if (tid < rank) {
        s_in[tid]        = state[tid];
        s_in[rank + tid] = __half2float(markov_w1[(size_t) tok * rank + tid]);
    }
    __syncthreads();
    if (c >= rank) return;

    const int n_in = 2 * rank;
    const half2 * wg = reinterpret_cast<const half2 *>(rnn_proj_step + (size_t) c * n_in);
    const half2 * wc = reinterpret_cast<const half2 *>(rnn_proj_step + (size_t) (rank + c) * n_in);
    const half2 * wo = reinterpret_cast<const half2 *>(rnn_proj_step + (size_t) (2 * rank + c) * n_in);
    const float2 * x2 = reinterpret_cast<const float2 *>(s_in);

    float dg = 0.0f, dc = 0.0f, do_ = 0.0f;
    const int n2 = n_in >> 1;
    for (int i = lane; i < n2; i += WARP_SIZE) {
        const float2 xv = x2[i];
        const float2 gv = __half22float2(wg[i]);
        const float2 cv = __half22float2(wc[i]);
        const float2 ov = __half22float2(wo[i]);
        dg  += gv.x * xv.x + gv.y * xv.y;
        dc  += cv.x * xv.x + cv.y * xv.y;
        do_ += ov.x * xv.x + ov.y * xv.y;
    }
    dg  = warp_reduce_sum(dg);
    dc  = warp_reduce_sum(dc);
    do_ = warp_reduce_sum(do_);

    if (lane == 0) {
        const float g_in   = round_bf16_f32(dg  + rnn_ctx_s[c]);
        const float c_in   = round_bf16_f32(dc  + rnn_ctx_s[rank + c]);
        const float o_in   = round_bf16_f32(do_ + rnn_ctx_s[2 * rank + c]);
        const float st     = s_in[c];
        const float gate   = round_bf16_f32(1.0f / (1.0f + expf(-g_in)));
        const float cand   = round_bf16_f32(tanhf(c_in));
        const float keep   = round_bf16_f32(gate * st);
        const float update = round_bf16_f32(round_bf16_f32(1.0f - gate) * cand);
        dst[c]        = round_bf16_f32(keep + update);
        dst[rank + c] = round_bf16_f32(tanhf(o_in));
    }
}

// ─── 30. DSpark On-Device Mask Builder (Zero CPU Fill / Zero PCIe Upload at 262K!) ─
static __global__ void inkling2_dspark_build_masks_kernel(
        const int32_t * __restrict__ draft_pos, // [B]
        half          * __restrict__ dst,       // [kv_span, B, 2]: slice 0 = global, slice 1 = local
        int kv_span, int B, int swa_window) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= kv_span) return;
    const int anchor_pos  = draft_pos[0];
    const int kv_total    = anchor_pos + B;
    const int local_start = max(0, anchor_pos - swa_window + 1);
    const half h_zero = __float2half(0.0f);
    const half h_ninf = __float2half(-INFINITY);
    const half vg = (k < kv_total) ? h_zero : h_ninf;
    const half vl = (k >= local_start && k < kv_total) ? h_zero : h_ninf;
    const size_t slice_elems = (size_t) kv_span * B;
    for (int q = 0; q < B; ++q) {
        dst[(size_t) q * kv_span + k] = vg;
        dst[slice_elems + (size_t) q * kv_span + k] = vl;
    }
}

// ─── 31. Fused Q-Clip + R-Clip + Relative-Position Table (Eliminates 48 Launches/Step!) ─
// Warp 0 clips R into s_r[0..d_rel-1] while Warp 1 clips Q (128 elems -> 4/lane) IN PARALLEL!
// Then all 256 threads compute the vectorized float4 rel_table.
// Output layout (flat F32): [Q: 128*Hq*T] followed by [rel_table: (rel_extent+1)*Hq*T].
static __global__ void inkling2_clip_qr_rel_table_kernel(
        const float   * __restrict__ q_raw, size_t q_stride,
        const float   * __restrict__ q_clip_w, // nullptr or [128, Hq]
        const float   * __restrict__ r_raw, size_t r_stride,
        const float   * __restrict__ r_clip_w, // nullptr or [d_rel, Hq]
        const int32_t * __restrict__ pos_buf,  // nullptr or [T]
        const float   * __restrict__ rel_bias, // [d_rel, rel_extent]
        const float   * __restrict__ sink, size_t sink_stride,
        float         * __restrict__ dst_q,    // [128, Hq, T]
        float         * __restrict__ dst_rt,   // [rel_extent+1, Hq, T]
        int head_dim, int d_rel, int rel_extent, int n_head,
        float q_thresh, float r_thresh, float eps, int log_floor, float log_alpha) {
    const int h   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    float tau = 1.0f;
    if (pos_buf && log_floor > 0 && log_alpha > 0.0f) {
        const float ratio = (float)(pos_buf[t] + 1) / (float) log_floor;
        if (ratio > 1.0f) tau = 1.0f + log_alpha * logf(ratio);
    }

    __shared__ float s_r[32];
    if (tid < WARP_SIZE) {
        // Warp 0: R L2-clip + scale + log-tau
        float v = 0.0f;
        if (tid < d_rel) {
            v = r_raw[(size_t) t * r_stride + (size_t) h * d_rel + tid];
            if (r_clip_w) v *= r_clip_w[(size_t) h * d_rel + tid];
        }
        const float ss = warp_reduce_sum(v * v);
        const float mult = fminf(1.0f, r_thresh / (sqrtf(ss) + eps)) * tau;
        if (tid < d_rel) s_r[tid] = v * mult;
    } else if (tid < 2 * WARP_SIZE) {
        // Warp 1: Q L2-clip + scale + log-tau IN PARALLEL WITH WARP 0 (zero extra wall time!)
        const int lane = tid - WARP_SIZE;
        const float * qp = q_raw + (size_t) t * q_stride + (size_t) h * head_dim;
        const float * sp = q_clip_w ? (q_clip_w + (size_t) h * head_dim) : nullptr;
        float * oq = dst_q + ((size_t) t * n_head + h) * head_dim;
        float ss = 0.0f;
        float vals[4];
        int cnt = 0;
        for (int d = lane; d < head_dim; d += WARP_SIZE) {
            float v = qp[d];
            if (sp) v *= sp[d];
            vals[cnt++] = v;
            ss += v * v;
        }
        ss = warp_reduce_sum(ss);
        const float mult = fminf(1.0f, q_thresh / (sqrtf(ss) + eps)) * tau;
        cnt = 0;
        for (int d = lane; d < head_dim; d += WARP_SIZE) {
            oq[d] = vals[cnt++] * mult;
        }
    }
    __syncthreads();

    const float s_val = sink[(size_t) t * sink_stride + h];
    float * op = dst_rt + ((size_t) t * n_head + h) * (rel_extent + 1);
    if (d_rel == 16) {
        const float4 r0 = reinterpret_cast<const float4 *>(s_r)[0];
        const float4 r1 = reinterpret_cast<const float4 *>(s_r)[1];
        const float4 r2 = reinterpret_cast<const float4 *>(s_r)[2];
        const float4 r3 = reinterpret_cast<const float4 *>(s_r)[3];
        const float4 * rb4 = reinterpret_cast<const float4 *>(rel_bias);
        for (int dist = tid; dist < rel_extent; dist += blockDim.x) {
            const float4 * b = rb4 + (size_t) dist * 4;
            const float4 b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
            const float dot = b0.x * r0.x + b0.y * r0.y + b0.z * r0.z + b0.w * r0.w
                            + b1.x * r1.x + b1.y * r1.y + b1.z * r1.z + b1.w * r1.w
                            + b2.x * r2.x + b2.y * r2.y + b2.z * r2.z + b2.w * r2.w
                            + b3.x * r3.x + b3.y * r3.y + b3.z * r3.z + b3.w * r3.w;
            op[dist] = dot - s_val;
        }
    } else {
        for (int dist = tid; dist < rel_extent; dist += blockDim.x) {
            const float * rb = rel_bias + (size_t) dist * d_rel;
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                if (k < d_rel) dot += rb[k] * s_r[k];
            }
            op[dist] = dot - s_val;
        }
    }
    if (tid == 0) op[rel_extent] = -s_val;
}

// ─── 32. Exact GPU Top-Candidate Extractor for Fast Non-Greedy Sampling (temp > 0) ─
// Stage 1: finds per-block max across V=201k and zeroes per-row atomic accumulators.
static __global__ void inkling2_top_cands_stage1_kernel(
        const float * __restrict__ logits, // [V, T]
        float       * __restrict__ blk_max, // [128, T]
        int32_t     * __restrict__ out_buf, // [516, T]
        int n_vocab) {
    const int b   = blockIdx.x; // 0..127
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;

    if (b == 0 && tid == 0) {
        int32_t * row = out_buf + (size_t) t * 516;
        row[0] = 0; // count
        row[2] = 0; // float 0.0f bit pattern for Z atomicAdd
    }

    const int chunk = (n_vocab + 127) >> 7;
    const int v0 = b * chunk;
    const int v1 = min(n_vocab, v0 + chunk);
    const float * lp = logits + (size_t) t * n_vocab;

    float mx = -INFINITY;
    for (int v = v0 + tid; v < v1; v += blockDim.x) {
        mx = fmaxf(mx, lp[v]);
    }
    mx = warp_reduce_max(mx);
    __shared__ float s_w[8];
    if (lane == 0) s_w[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_w[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) blk_max[(size_t) t * 128 + b] = v;
    }
}

// Stage 2: computes exact full-vocab Z = sum exp((x - max)/temp) and extracts all
// candidates with x >= max - 12*temp (up to 256/row -> 2 KB/row instead of 804 KB/row!).
static __global__ void inkling2_top_cands_stage2_kernel(
        const float * __restrict__ logits,  // [V, T]
        const float * __restrict__ blk_max, // [128, T]
        int32_t     * __restrict__ out_buf, // [516, T]
        int n_vocab, float temp) {
    const int b   = blockIdx.x; // 0..127
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;

    __shared__ float s_max;
    __shared__ float s_w[8];
    if (warp == 0) {
        const float * bm = blk_max + (size_t) t * 128;
        float mx = fmaxf(fmaxf(bm[lane], bm[lane + 32]), fmaxf(bm[lane + 64], bm[lane + 96]));
        mx = warp_reduce_max(mx);
        if (lane == 0) s_max = mx;
    }
    __syncthreads();

    const float gmax     = s_max;
    const float inv_temp = 1.0f / fmaxf(temp, 1e-6f);
    const float cut      = gmax - 12.0f * fmaxf(temp, 1e-6f);
    int32_t * row        = out_buf + (size_t) t * 516;
    if (b == 0 && tid == 0) {
        row[1] = __float_as_int(gmax);
    }

    const int chunk = (n_vocab + 127) >> 7;
    const int v0 = b * chunk;
    const int v1 = min(n_vocab, v0 + chunk);
    const float * lp = logits + (size_t) t * n_vocab;

    float z_acc = 0.0f;
    for (int v = v0 + tid; v < v1; v += blockDim.x) {
        const float x = lp[v];
        const float scaled = (x - gmax) * inv_temp;
        if (scaled > -40.0f) {
            z_acc += expf(scaled);
        }
        if (x >= cut) {
            const int slot = atomicAdd(&row[0], 1);
            if (slot < 256) {
                row[4 + 2 * slot + 0] = __float_as_int(x);
                row[4 + 2 * slot + 1] = v;
            }
        }
    }
    z_acc = warp_reduce_sum(z_acc);
    if (lane == 0) s_w[warp] = z_acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_w[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0 && v > 0.0f) {
            atomicAdd(reinterpret_cast<float *>(&row[2]), v);
        }
    }
}

// ─── 33. Tree 4-Tap ShortConv (+ optional dynamic taps + optional residual addend) ──
// Packed meta [m_stride, T] per row t:
//   [0]=slot, [1]=pos, [2]=plane, [3]=flags, [4]=extra,
//   [5..7]=hist_row[3] (>=0 -> x[hr], <0 -> use hist_src),
//   [8..10]=hist_src[3] (>=0 -> tree_conv[plane, hs], <0 -> prefix_conv[plane, -(hs+1)]),
//   [11..11+W-1]=path[W].
static __global__ void inkling2_tree_sconv_kernel(
        const float   * __restrict__ x, size_t x_stride,
        const float   * __restrict__ w_static,    // [4, C]
        const float   * __restrict__ prefix_conv, // [3, S_ch, P]
        float         * __restrict__ tree_conv,   // [S_ch, S_max, P]
        const int32_t * __restrict__ meta,        // [m_stride, T]
        const float   * __restrict__ dyn_taps, size_t dyn_stride,
        const float   * __restrict__ addend,   size_t add_stride,
        float         * __restrict__ dst,         // [C, T]
        int n_ch, int n_tokens, int s_ch, int s_max,
        int chan_offset, int head_dim_dyn, int m_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= n_ch || t >= n_tokens) return;

    const int32_t * mt = meta + (size_t) t * m_stride;
    const int slot  = mt[0];
    const int plane = mt[2];
    const int sc    = chan_offset + c;
    const float cur = x[(size_t) t * x_stride + c];

    const float * pc_plane = prefix_conv + (size_t) plane * s_ch * 3 + (size_t) sc * 3;
    float * tc_plane       = tree_conv   + (size_t) plane * s_max * s_ch;

    float hist[3];
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int hr = mt[5 + j];
        if (hr >= 0) {
            hist[j] = x[(size_t) hr * x_stride + c];
        } else {
            const int hs = mt[8 + j];
            hist[j] = (hs >= 0) ? tc_plane[(size_t) hs * s_ch + sc] : pc_plane[-(hs + 1)];
        }
    }

    const float * wc = w_static + (size_t) c * 4;
    float acc = wc[0] * hist[0] + wc[1] * hist[1] + wc[2] * hist[2] + wc[3] * cur;
    if (dyn_taps) {
        const int h = c / head_dim_dyn;
        const float * dp = dyn_taps + (size_t) t * dyn_stride + h * 4;
        acc += dp[0] * cur + dp[1] * hist[2] + dp[2] * hist[1] + dp[3] * hist[0];
    }
    const float scale = dyn_taps ? (1.0f / 3.0f) : rsqrtf(5.0f);
    float y = (cur + acc) * scale;
    if (addend) y += addend[(size_t) t * add_stride + c];
    dst[(size_t) t * n_ch + c] = y;
    tc_plane[(size_t) slot * s_ch + sc] = cur;
}

// ─── 34. Fused Tree 4-Tap ShortConv + K-Clip / V-RMSNorm + Tree KV Write ──
template <bool KV_Q8_0>
static __global__ void inkling2_tree_sconv_kv_write_kernel(
        const float   * __restrict__ x, size_t x_stride,
        const float   * __restrict__ w_static,    // [4, Hk*128]
        const float   * __restrict__ prefix_conv, // [3, S_ch, P]
        float         * __restrict__ tree_conv,   // [S_ch, S_max, P]
        const int32_t * __restrict__ meta,        // [m_stride, T]
        const float   * __restrict__ dyn_taps, size_t dyn_stride,
        char          * __restrict__ tree_kv, size_t kv_nb1, size_t kv_nb2, size_t kv_nb3,
        int n_head_kv, int s_ch, int s_max, int chan_offset,
        int is_v, int m_stride, float threshold, float eps) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x; // 0..127
    const int c = h * 128 + d;
    const int sc = chan_offset + c;

    const int32_t * mt = meta + (size_t) t * m_stride;
    const int slot  = mt[0];
    const int plane = mt[2];
    const float cur = x[(size_t) t * x_stride + c];

    const float * pc_plane = prefix_conv + (size_t) plane * s_ch * 3 + (size_t) sc * 3;
    float * tc_plane       = tree_conv   + (size_t) plane * s_max * s_ch;

    float hist[3];
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int hr = mt[5 + j];
        if (hr >= 0) {
            hist[j] = x[(size_t) hr * x_stride + c];
        } else {
            const int hs = mt[8 + j];
            hist[j] = (hs >= 0) ? tc_plane[(size_t) hs * s_ch + sc] : pc_plane[-(hs + 1)];
        }
    }

    const float * wc = w_static + (size_t) c * 4;
    float acc = wc[0] * hist[0] + wc[1] * hist[1] + wc[2] * hist[2] + wc[3] * cur;
    if (dyn_taps) {
        const float * dp = dyn_taps + (size_t) t * dyn_stride + h * 4;
        acc += dp[0] * cur + dp[1] * hist[2] + dp[2] * hist[1] + dp[3] * hist[0];
    }
    const float scale = dyn_taps ? (1.0f / 3.0f) : rsqrtf(5.0f);
    __shared__ float s_val[128];
    s_val[d] = (cur + acc) * scale;
    tc_plane[(size_t) slot * s_ch + sc] = cur;
    __syncthreads();

    if (d < WARP_SIZE) {
        const int lane = d;
        const float v0 = s_val[lane +  0];
        const float v1 = s_val[lane + 32];
        const float v2 = s_val[lane + 64];
        const float v3 = s_val[lane + 96];
        const float ss = warp_reduce_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
        const float mult = is_v
            ? rsqrtf(ss / 128.0f + eps)
            : fminf(1.0f, threshold / (sqrtf(ss) + eps));
        char * dst_row = tree_kv + (size_t) plane * kv_nb3 + (size_t) h * kv_nb2 + (size_t) slot * kv_nb1;
        if constexpr (KV_Q8_0) {
            block_q8_0 * bp = reinterpret_cast<block_q8_0 *>(dst_row);
            const float vs[4] = { v0 * mult, v1 * mult, v2 * mult, v3 * mult };
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float amax = warp_reduce_max(fabsf(vs[b]));
                const float d_q = amax / 127.0f;
                bp[b].qs[lane] = (amax == 0.0f) ? 0 : (int8_t) __float2int_rn(vs[b] / d_q);
                if (lane == 0) bp[b].d = __float2half(d_q);
            }
        } else {
            half * hp = reinterpret_cast<half *>(dst_row);
            hp[lane +  0] = __float2half(v0 * mult);
            hp[lane + 32] = __float2half(v1 * mult);
            hp[lane + 64] = __float2half(v2 * mult);
            hp[lane + 96] = __float2half(v3 * mult);
        }
    }
}

// ─── 35. Tree-Aware Inline Relative-Bias + Zero-Sink + Gate Wave32 Attention ──
template <bool KV_Q8_0, int GQA, int NWARPS>
static __global__ void inkling2_tree_attn_rel_kernel(
        const float   * __restrict__ q_ptr,          // [128, Hq, T]
        const char    * __restrict__ prefix_k,       // [128, cap, Hk, P]
        const char    * __restrict__ prefix_v,       // [128, cap, Hk, P]
        size_t pk_nb1, size_t pk_nb2, size_t pk_nb3,
        size_t pv_nb1, size_t pv_nb2, size_t pv_nb3,
        const char    * __restrict__ tree_k,         // [128, S_max, Hk, P]
        const char    * __restrict__ tree_v,         // [128, S_max, Hk, P]
        size_t tk_nb1, size_t tk_nb2, size_t tk_nb3,
        size_t tv_nb1, size_t tv_nb2, size_t tv_nb3,
        const float   * __restrict__ rel_table,      // [rel_extent+1, Hq, T]
        const int32_t * __restrict__ meta,           // [m_stride, T]
        const int32_t * __restrict__ prefix_len_buf, // [1] or [P] (per-plane for multi-slot batching)
        const float   * __restrict__ gate_ptr, size_t gate_stride,
        float         * __restrict__ dst,            // [128*Hq, T]
        int Hq, int prefix_cap, int swa_window, int rel_extent,
        int m_stride, int path_w, int prefix_len_n) {
    const int lane    = threadIdx.x; // 0..31
    const int warp    = threadIdx.y; // 0..NWARPS-1
    const int kv_head = blockIdx.x;
    const int q_tok   = blockIdx.y;

    const int32_t * mt  = meta + (size_t) q_tok * m_stride;
    const int q_pos     = mt[1];
    const int plane     = mt[2];
    const int pref_len  = (prefix_len_n > 1) ? prefix_len_buf[plane] : prefix_len_buf[0];
    const int eff_end   = q_pos + 1;
    const int eff_start = (swa_window > 0) ? max(0, q_pos - swa_window + 1) : 0;

    float q_reg[GQA][4];
    const float * rt_base[GQA];
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        const int h = kv_head * GQA + g;
        const float * qp = q_ptr + ((size_t) q_tok * Hq + h) * 128;
        q_reg[g][0] = qp[lane +  0] * kInvSqrt128;
        q_reg[g][1] = qp[lane + 32] * kInvSqrt128;
        q_reg[g][2] = qp[lane + 64] * kInvSqrt128;
        q_reg[g][3] = qp[lane + 96] * kInvSqrt128;
        rt_base[g] = rel_table + ((size_t) q_tok * Hq + h) * (rel_extent + 1);
    }

    float m_max[GQA];
    float l_sum[GQA];
    float o_acc[GQA][4];
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        if (warp == 0) {
            m_max[g] = 0.0f;
            l_sum[g] = 1.0f;
        } else {
            m_max[g] = -INFINITY;
            l_sum[g] = 0.0f;
        }
        o_acc[g][0] = 0.0f; o_acc[g][1] = 0.0f; o_acc[g][2] = 0.0f; o_acc[g][3] = 0.0f;
    }

    const char * pk_head = prefix_k + (size_t) plane * pk_nb3 + (size_t) kv_head * pk_nb2;
    const char * pv_head = prefix_v + (size_t) plane * pv_nb3 + (size_t) kv_head * pv_nb2;
    const char * tk_head = tree_k   + (size_t) plane * tk_nb3 + (size_t) kv_head * tk_nb2;
    const char * tv_head = tree_v   + (size_t) plane * tv_nb3 + (size_t) kv_head * tv_nb2;

    for (int kp = eff_start + warp; kp < eff_end; kp += NWARPS) {
        const char * k_row = nullptr;
        const char * v_row = nullptr;
        if (kp < pref_len) {
            const int pslot = (swa_window > 0) ? (kp % prefix_cap) : kp;
            k_row = pk_head + (size_t) pslot * pk_nb1;
            v_row = pv_head + (size_t) pslot * pv_nb1;
        } else {
            const int e = kp - pref_len;
            if (e < 0 || e >= path_w) continue;
            const int tslot = mt[11 + e];
            if (tslot < 0) continue;
            k_row = tk_head + (size_t) tslot * tk_nb1;
            v_row = tv_head + (size_t) tslot * tv_nb1;
        }

        float kv[4], vv[4];
        if constexpr (KV_Q8_0) {
            const block_q8_0 * kb = reinterpret_cast<const block_q8_0 *>(k_row);
            const block_q8_0 * vb = reinterpret_cast<const block_q8_0 *>(v_row);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                kv[b] = __half2float(kb[b].d) * (float) kb[b].qs[lane];
                vv[b] = __half2float(vb[b].d) * (float) vb[b].qs[lane];
            }
        } else {
            const half * kh = reinterpret_cast<const half *>(k_row);
            const half * vh = reinterpret_cast<const half *>(v_row);
            kv[0] = __half2float(kh[lane +  0]); kv[1] = __half2float(kh[lane + 32]);
            kv[2] = __half2float(kh[lane + 64]); kv[3] = __half2float(kh[lane + 96]);
            vv[0] = __half2float(vh[lane +  0]); vv[1] = __half2float(vh[lane + 32]);
            vv[2] = __half2float(vh[lane + 64]); vv[3] = __half2float(vh[lane + 96]);
        }
        const int dist = min(q_pos - kp, rel_extent);

        #pragma unroll
        for (int g = 0; g < GQA; ++g) {
            float dot = q_reg[g][0] * kv[0] + q_reg[g][1] * kv[1]
                      + q_reg[g][2] * kv[2] + q_reg[g][3] * kv[3];
            dot = warp_reduce_sum(dot);
            const float score = dot + rt_base[g][dist];
            const float m_new = fmaxf(m_max[g], score);
            const float alpha = isinf(m_max[g]) ? 0.0f : expf(m_max[g] - m_new);
            const float p     = expf(score - m_new);
            l_sum[g] = l_sum[g] * alpha + p;
            o_acc[g][0] = o_acc[g][0] * alpha + p * vv[0];
            o_acc[g][1] = o_acc[g][1] * alpha + p * vv[1];
            o_acc[g][2] = o_acc[g][2] * alpha + p * vv[2];
            o_acc[g][3] = o_acc[g][3] * alpha + p * vv[3];
            m_max[g] = m_new;
        }
    }

    __shared__ float s_o[NWARPS][GQA][128];
    __shared__ float s_m[NWARPS][GQA];
    __shared__ float s_l[NWARPS][GQA];

    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        s_o[warp][g][lane +  0] = o_acc[g][0];
        s_o[warp][g][lane + 32] = o_acc[g][1];
        s_o[warp][g][lane + 64] = o_acc[g][2];
        s_o[warp][g][lane + 96] = o_acc[g][3];
        if (lane == 0) {
            s_m[warp][g] = m_max[g];
            s_l[warp][g] = l_sum[g];
        }
    }
    __syncthreads();

    if (warp < GQA) {
        const int g = warp;
        float mx = s_m[0][g];
        #pragma unroll
        for (int w = 1; w < NWARPS; ++w) mx = fmaxf(mx, s_m[w][g]);
        float ltot = 0.0f;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
        if (!isinf(mx)) {
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                if (s_l[w][g] > 0.0f) {
                    const float sc = expf(s_m[w][g] - mx);
                    ltot += s_l[w][g] * sc;
                    o0 += s_o[w][g][lane +  0] * sc;
                    o1 += s_o[w][g][lane + 32] * sc;
                    o2 += s_o[w][g][lane + 64] * sc;
                    o3 += s_o[w][g][lane + 96] * sc;
                }
            }
        }
        const int h = kv_head * GQA + g;
        const float inv_l = ltot > 0.0f ? (1.0f / ltot) : 0.0f;
        float r0 = o0 * inv_l, r1 = o1 * inv_l, r2 = o2 * inv_l, r3 = o3 * inv_l;
        if (gate_ptr) {
            const float * gp = gate_ptr + (size_t) q_tok * gate_stride + (size_t) h * 128;
            r0 *= sigmoid_f32(gp[lane +  0]);
            r1 *= sigmoid_f32(gp[lane + 32]);
            r2 *= sigmoid_f32(gp[lane + 64]);
            r3 *= sigmoid_f32(gp[lane + 96]);
        }
        float * out = dst + ((size_t) q_tok * Hq + h) * 128;
        out[lane +  0] = r0; out[lane + 32] = r1; out[lane + 64] = r2; out[lane + 96] = r3;
    }
}

// ─── 36. Tree Commit: Copy Accepted Path Slots from Tree -> Prefix KV & Conv ──
static __global__ void inkling2_tree_commit_kv_kernel(
        char          * __restrict__ prefix_k, size_t pk_nb1, size_t pk_nb2, size_t pk_nb3,
        char          * __restrict__ prefix_v, size_t pv_nb1, size_t pv_nb2, size_t pv_nb3,
        const char    * __restrict__ tree_k,   size_t tk_nb1, size_t tk_nb2, size_t tk_nb3,
        const char    * __restrict__ tree_v,   size_t tv_nb1, size_t tv_nb2, size_t tv_nb3,
        const int32_t * __restrict__ commit_slots,
        const int32_t * __restrict__ commit_meta, // {prefix_len, n_commit}
        int prefix_cap, int swa_window, int row_u32) {
    const int h     = blockIdx.x;
    const int plane = blockIdx.y;
    const int tid   = threadIdx.x;
    const int pref_len = commit_meta[0];
    const int n_commit = commit_meta[1];

    char * pk_h = prefix_k + (size_t) plane * pk_nb3 + (size_t) h * pk_nb2;
    char * pv_h = prefix_v + (size_t) plane * pv_nb3 + (size_t) h * pv_nb2;
    const char * tk_h = tree_k + (size_t) plane * tk_nb3 + (size_t) h * tk_nb2;
    const char * tv_h = tree_v + (size_t) plane * tv_nb3 + (size_t) h * tv_nb2;

    for (int i = 0; i < n_commit; ++i) {
        const int pos = pref_len + i;
        const int dst_slot = (swa_window > 0) ? (pos % prefix_cap) : pos;
        const int src_slot = commit_slots[i];
        uint32_t * d_k = reinterpret_cast<uint32_t *>(pk_h + (size_t) dst_slot * pk_nb1);
        uint32_t * d_v = reinterpret_cast<uint32_t *>(pv_h + (size_t) dst_slot * pv_nb1);
        const uint32_t * s_k = reinterpret_cast<const uint32_t *>(tk_h + (size_t) src_slot * tk_nb1);
        const uint32_t * s_v = reinterpret_cast<const uint32_t *>(tv_h + (size_t) src_slot * tv_nb1);
        for (int u = tid; u < row_u32; u += blockDim.x) {
            d_k[u] = s_k[u];
            d_v[u] = s_v[u];
        }
    }
}

static __global__ void inkling2_tree_commit_conv_kernel(
        float         * __restrict__ prefix_conv, // [3, S_ch, P]
        const float   * __restrict__ tree_conv,   // [S_ch, S_max, P]
        const int32_t * __restrict__ commit_slots,
        const int32_t * __restrict__ commit_meta, // {prefix_len, n_commit}
        int s_ch, int s_max) {
    const int c     = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = blockIdx.y;
    if (c >= s_ch) return;
    const int n_commit = commit_meta[1];
    if (n_commit <= 0) return;

    float * pc = prefix_conv + ((size_t) plane * s_ch + c) * 3;
    const float * tc_plane = tree_conv + (size_t) plane * s_max * s_ch;
    float h0 = pc[0], h1 = pc[1], h2 = pc[2];
    const int start = max(0, n_commit - 3);
    for (int i = start; i < n_commit; ++i) {
        const int slot = commit_slots[i];
        h0 = h1;
        h1 = h2;
        h2 = tc_plane[(size_t) slot * s_ch + c];
    }
    pc[0] = h0;
    pc[1] = h1;
    pc[2] = h2;
}

// ─── 37. Batched 2DSD Vanilla DSpark Markov Argmax Across L Leaves ──
template <bool IS_Q8>
static __global__ void inkling2_2dsd_markov_stage1_kernel(
        const float   * __restrict__ base_logits, size_t base_stride, // [V, L]
        const int32_t * __restrict__ prev_tok,                        // [L]
        const half    * __restrict__ markov_w1,   size_t w1_stride,   // [R=256, V] F16
        const char    * __restrict__ markov_w2,   size_t w2_nb1,      // [R=256, V] Q8_0 or F16
        float         * __restrict__ part_val,                        // [256, L]
        int32_t       * __restrict__ part_id,                         // [256, L]
        int n_vocab, int rank) {
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..7
    const int tid  = warp * WARP_SIZE + lane; // 0..255
    const int leaf = blockIdx.y;

    __shared__ float s_lat[256];
    if (tid < rank) {
        const int tok = max(0, min(n_vocab - 1, prev_tok[leaf]));
        s_lat[tid] = __half2float(markov_w1[(size_t) tok * w1_stride + tid]);
    }
    __syncthreads();

    const float * base_row = base_logits + (size_t) leaf * base_stride;
    const int chunk = (n_vocab + 255) >> 8;
    const int v_beg = blockIdx.x * chunk;
    const int v_end = min(n_vocab, v_beg + chunk);

    float best_v = -INFINITY;
    int32_t best_id = 0;

    if constexpr (IS_Q8) {
        const int blk = lane >> 2;
        const int sub = lane & 3;
        const float * lp = s_lat + blk * 32 + sub * 8;
        const float l0 = lp[0], l1 = lp[1], l2 = lp[2], l3 = lp[3];
        const float l4 = lp[4], l5 = lp[5], l6 = lp[6], l7 = lp[7];
        for (int v = v_beg + warp; v < v_end; v += 8) {
            const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(markov_w2 + (size_t) v * w2_nb1) + blk;
            const uint16_t * hp = reinterpret_cast<const uint16_t *>(bp->qs + sub * 8);
            const uint16_t p0 = hp[0], p1 = hp[1], p2 = hp[2], p3 = hp[3];
            float acc = (float)(int8_t)(p0 & 0xFFu) * l0 + (float)(int8_t)(p0 >> 8) * l1
                      + (float)(int8_t)(p1 & 0xFFu) * l2 + (float)(int8_t)(p1 >> 8) * l3
                      + (float)(int8_t)(p2 & 0xFFu) * l4 + (float)(int8_t)(p2 >> 8) * l5
                      + (float)(int8_t)(p3 & 0xFFu) * l6 + (float)(int8_t)(p3 >> 8) * l7;
            const float dot = warp_reduce_sum(acc * __half2float(bp->d));
            if (lane == 0) {
                const float sc = base_row[v] + dot;
                if (sc > best_v || (sc == best_v && v < best_id)) {
                    best_v = sc;
                    best_id = v;
                }
            }
        }
    } else {
        const float * lp = s_lat + lane * 8;
        const float l0 = lp[0], l1 = lp[1], l2 = lp[2], l3 = lp[3];
        const float l4 = lp[4], l5 = lp[5], l6 = lp[6], l7 = lp[7];
        for (int v = v_beg + warp; v < v_end; v += 8) {
            const half2 * h2p = reinterpret_cast<const half2 *>(markov_w2 + (size_t) v * w2_nb1) + lane * 4;
            const float2 f0 = __half22float2(h2p[0]);
            const float2 f1 = __half22float2(h2p[1]);
            const float2 f2 = __half22float2(h2p[2]);
            const float2 f3 = __half22float2(h2p[3]);
            float acc = f0.x * l0 + f0.y * l1 + f1.x * l2 + f1.y * l3
                      + f2.x * l4 + f2.y * l5 + f3.x * l6 + f3.y * l7;
            const float dot = warp_reduce_sum(acc);
            if (lane == 0) {
                const float sc = base_row[v] + dot;
                if (sc > best_v || (sc == best_v && v < best_id)) {
                    best_v = sc;
                    best_id = v;
                }
            }
        }
    }

    __shared__ float s_wv[8];
    __shared__ int32_t s_wi[8];
    if (lane == 0) {
        s_wv[warp] = best_v;
        s_wi[warp] = best_id;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float bv = s_wv[0];
        int32_t bi = s_wi[0];
        for (int w = 1; w < 8; ++w) {
            if (s_wv[w] > bv || (s_wv[w] == bv && s_wi[w] < bi)) {
                bv = s_wv[w];
                bi = s_wi[w];
            }
        }
        part_val[(size_t) leaf * 256 + blockIdx.x] = bv;
        part_id[(size_t) leaf * 256 + blockIdx.x]  = bi;
    }
}

static __global__ void inkling2_2dsd_markov_stage2_kernel(
        const float   * __restrict__ part_val, // [256, L]
        const int32_t * __restrict__ part_id,  // [256, L]
        int32_t       * __restrict__ out_best) { // [L]
    const int tid  = threadIdx.x;
    const int leaf = blockIdx.x;
    __shared__ float s_v[256];
    __shared__ int32_t s_i[256];
    s_v[tid] = part_val[(size_t) leaf * 256 + tid];
    s_i[tid] = part_id[(size_t) leaf * 256 + tid];
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            if (s_v[tid + off] > s_v[tid] || (s_v[tid + off] == s_v[tid] && s_i[tid + off] < s_i[tid])) {
                s_v[tid] = s_v[tid + off];
                s_i[tid] = s_i[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_best[leaf] = s_i[0];
}

// ─── 38. Vectorized 2DSD Confidence Head: sigmoid(dot([h_spec, w1[prev]], w) + b) ──
static __global__ void inkling2_2dsd_conf_head_kernel(
        const float   * __restrict__ h_spec, size_t h_stride,   // [H, M]
        const int32_t * __restrict__ before_toks,               // [M]
        const half    * __restrict__ markov_w1, size_t w1_stride, // [R, V] F16
        const float   * __restrict__ conf_w,                    // [H + R]
        const float   * __restrict__ conf_b,                    // [1]
        float         * __restrict__ dst,                       // [M]
        int H, int R, int n_vocab) {
    const int m   = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;

    const float * hp = h_spec + (size_t) m * h_stride;
    const int tok = max(0, min(n_vocab - 1, before_toks[m]));
    const half * ep = markov_w1 + (size_t) tok * w1_stride;

    float acc = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        acc += hp[i] * conf_w[i];
    }
    for (int i = tid; i < R; i += blockDim.x) {
        acc += __half2float(ep[i]) * conf_w[H + i];
    }
    acc = warp_reduce_sum(acc);
    __shared__ float s_red[8];
    if (lane == 0) s_red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nwarps) ? s_red[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            dst[m] = sigmoid_f32(v + conf_b[0]);
        }
    }
}

// ─── 39. On-Device Multi-Leaf Loop-Context Drafter Attention Mask ─────────
static __global__ void inkling2_2dsd_draft_mask_kernel(
        const int32_t * __restrict__ draft_meta,
        half          * __restrict__ dst, // [kv_span, B * L_b]
        int kv_span, int B) {
    const int k    = blockIdx.x * blockDim.x + threadIdx.x;
    const int leaf = blockIdx.y;
    if (k >= kv_span) return;

    const int pref_len = draft_meta[0];
    const int L_b      = draft_meta[1];
    const int S_b      = draft_meta[2];
    const int slen     = draft_meta[3 + leaf];

    const int span_beg = pref_len + leaf * S_b;
    const int span_end = span_beg + slen;
    const int blk_beg  = pref_len + L_b * S_b + leaf * B;
    const int blk_end  = blk_beg + B;

    const bool vis = (k < pref_len)
                  || (k >= span_beg && k < span_end)
                  || (k >= blk_beg  && k < blk_end);
    const half val = vis ? __float2half(0.0f) : __float2half(-INFINITY);
    const int q0 = leaf * B;
    for (int q = 0; q < B; ++q) {
        dst[(size_t) (q0 + q) * kv_span + k] = val;
    }
}

// ─── 40. Batched 2DSD Top-1024 Candidate Vanilla Markov Argmax (32-Block Full-GPU Parallel!) ─
template <bool IS_Q8>
static __global__ void inkling2_2dsd_markov_topk_stage1_kernel(
        const int32_t * __restrict__ cand_ids,    size_t cand_stride, // [1024, L]
        const float   * __restrict__ base_logits, size_t base_stride, // [V, L]
        const int32_t * __restrict__ prev_tok,                        // [L]
        const half    * __restrict__ markov_w1,   size_t w1_stride,   // [256, V] F16
        const char    * __restrict__ markov_w2,   size_t w2_nb1,      // [256, V] Q8_0 or F16
        float         * __restrict__ part_val,                        // [32, L]
        int32_t       * __restrict__ part_id,                         // [32, L]
        int n_vocab, int n_cands) {
    // grid = (32, L, 1), block = (32, 8, 1) -> 32 CUs/leaf, only 4 candidates/warp (25x faster!)
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..7
    const int tid  = warp * WARP_SIZE + lane; // 0..255
    const int leaf = blockIdx.y;

    __shared__ float s_lat[256];
    const int tok = max(0, min(n_vocab - 1, prev_tok[leaf]));
    s_lat[tid] = __half2float(markov_w1[(size_t) tok * w1_stride + tid]);
    __syncthreads();

    const int32_t * cp       = cand_ids    + (size_t) leaf * cand_stride;
    const float   * base_row = base_logits + (size_t) leaf * base_stride;
    const int chunk = (n_cands + 31) >> 5;
    const int k_beg = blockIdx.x * chunk;
    const int k_end = min(n_cands, k_beg + chunk);

    float best_v = -INFINITY;
    int32_t best_id = 0;

    if constexpr (IS_Q8) {
        const int blk = lane >> 2;
        const int sub = lane & 3;
        const float * lp = s_lat + blk * 32 + sub * 8;
        const float l0 = lp[0], l1 = lp[1], l2 = lp[2], l3 = lp[3];
        const float l4 = lp[4], l5 = lp[5], l6 = lp[6], l7 = lp[7];
        for (int k = k_beg + warp; k < k_end; k += 8) {
            const int32_t v = cp[k];
            const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(markov_w2 + (size_t) v * w2_nb1) + blk;
            const uint16_t * hp = reinterpret_cast<const uint16_t *>(bp->qs + sub * 8);
            const uint16_t p0 = hp[0], p1 = hp[1], p2 = hp[2], p3 = hp[3];
            float acc = (float)(int8_t)(p0 & 0xFFu) * l0 + (float)(int8_t)(p0 >> 8) * l1
                      + (float)(int8_t)(p1 & 0xFFu) * l2 + (float)(int8_t)(p1 >> 8) * l3
                      + (float)(int8_t)(p2 & 0xFFu) * l4 + (float)(int8_t)(p2 >> 8) * l5
                      + (float)(int8_t)(p3 & 0xFFu) * l6 + (float)(int8_t)(p3 >> 8) * l7;
            const float dot = warp_reduce_sum(acc * __half2float(bp->d));
            if (lane == 0) {
                const float sc = base_row[v] + dot;
                if (sc > best_v || (sc == best_v && v < best_id)) {
                    best_v = sc;
                    best_id = v;
                }
            }
        }
    } else {
        const float * lp = s_lat + lane * 8;
        const float l0 = lp[0], l1 = lp[1], l2 = lp[2], l3 = lp[3];
        const float l4 = lp[4], l5 = lp[5], l6 = lp[6], l7 = lp[7];
        for (int k = k_beg + warp; k < k_end; k += 8) {
            const int32_t v = cp[k];
            const half2 * h2p = reinterpret_cast<const half2 *>(markov_w2 + (size_t) v * w2_nb1) + lane * 4;
            const float2 f0 = __half22float2(h2p[0]);
            const float2 f1 = __half22float2(h2p[1]);
            const float2 f2 = __half22float2(h2p[2]);
            const float2 f3 = __half22float2(h2p[3]);
            float acc = f0.x * l0 + f0.y * l1 + f1.x * l2 + f1.y * l3
                      + f2.x * l4 + f2.y * l5 + f3.x * l6 + f3.y * l7;
            const float dot = warp_reduce_sum(acc);
            if (lane == 0) {
                const float sc = base_row[v] + dot;
                if (sc > best_v || (sc == best_v && v < best_id)) {
                    best_v = sc;
                    best_id = v;
                }
            }
        }
    }

    __shared__ float s_wv[8];
    __shared__ int32_t s_wi[8];
    if (lane == 0) {
        s_wv[warp] = best_v;
        s_wi[warp] = best_id;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float bv = s_wv[0];
        int32_t bi = s_wi[0];
        for (int w = 1; w < 8; ++w) {
            if (s_wv[w] > bv || (s_wv[w] == bv && s_wi[w] < bi)) {
                bv = s_wv[w];
                bi = s_wi[w];
            }
        }
        part_val[(size_t) leaf * 32 + blockIdx.x] = bv;
        part_id[(size_t) leaf * 32 + blockIdx.x]  = bi;
    }
}

static __global__ void inkling2_2dsd_markov_topk_stage2_kernel(
        const float   * __restrict__ part_val, // [32, L]
        const int32_t * __restrict__ part_id,  // [32, L]
        int32_t       * __restrict__ out_best) { // [L]
    // 1 warp per leaf, zero __syncthreads(), 5 warp shuffles!
    const int lane = threadIdx.x;
    const int leaf = blockIdx.x;
    float   bv = part_val[(size_t) leaf * 32 + lane];
    int32_t bi = part_id[(size_t) leaf * 32 + lane];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float   ov = i2_shfl_xor(bv, off);
        const int32_t oi = i2_shfl_xor(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) out_best[leaf] = bi;
}

// ─── 41. Fused SWA Tree K+V ShortConv + KV-Write (1 Launch for Both K and V!) ──
template <bool KV_Q8_0>
static __global__ void inkling2_tree_sconv_kv_write_kv_kernel(
        const float   * __restrict__ x_k, size_t xk_stride,
        const float   * __restrict__ x_v, size_t xv_stride,
        const float   * __restrict__ wk_static,
        const float   * __restrict__ wv_static,
        const float   * __restrict__ prefix_conv, // [3, S_ch, P]
        float         * __restrict__ tree_conv,   // [S_ch, S_max, P]
        const int32_t * __restrict__ meta,        // [m_stride, T]
        char          * __restrict__ tree_k, size_t tk_nb1, size_t tk_nb2, size_t tk_nb3,
        char          * __restrict__ tree_v, size_t tv_nb1, size_t tv_nb2, size_t tv_nb3,
        int n_head_kv, int s_ch, int s_max, int m_stride, float threshold, float eps) {
    const int is_v = (blockIdx.x >= n_head_kv) ? 1 : 0;
    const int h    = is_v ? (blockIdx.x - n_head_kv) : blockIdx.x;
    const int t    = blockIdx.y;
    const int d    = threadIdx.x; // 0..127
    const int c    = h * 128 + d;
    const int sc   = (is_v ? n_head_kv * 128 : 0) + c;

    const float * x        = is_v ? x_v : x_k;
    const size_t  x_stride = is_v ? xv_stride : xk_stride;
    const float * w_static = is_v ? wv_static : wk_static;
    char        * tree_kv  = is_v ? tree_v : tree_k;
    const size_t  kv_nb1   = is_v ? tv_nb1 : tk_nb1;
    const size_t  kv_nb2   = is_v ? tv_nb2 : tk_nb2;
    const size_t  kv_nb3   = is_v ? tv_nb3 : tk_nb3;

    const int32_t * mt = meta + (size_t) t * m_stride;
    const int slot  = mt[0];
    const int plane = mt[2];
    const float cur = x[(size_t) t * x_stride + c];

    const float * pc_plane = prefix_conv + (size_t) plane * s_ch * 3 + (size_t) sc * 3;
    float * tc_plane       = tree_conv   + (size_t) plane * s_max * s_ch;

    float hist[3];
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int hr = mt[5 + j];
        if (hr >= 0) {
            hist[j] = x[(size_t) hr * x_stride + c];
        } else {
            const int hs = mt[8 + j];
            hist[j] = (hs >= 0) ? tc_plane[(size_t) hs * s_ch + sc] : pc_plane[-(hs + 1)];
        }
    }

    const float * wc = w_static + (size_t) c * 4;
    const float acc  = wc[0] * hist[0] + wc[1] * hist[1] + wc[2] * hist[2] + wc[3] * cur;
    __shared__ float s_val[128];
    s_val[d] = (cur + acc) * rsqrtf(5.0f);
    tc_plane[(size_t) slot * s_ch + sc] = cur;
    __syncthreads();

    if (d < WARP_SIZE) {
        const int lane = d;
        const float v0 = s_val[lane +  0];
        const float v1 = s_val[lane + 32];
        const float v2 = s_val[lane + 64];
        const float v3 = s_val[lane + 96];
        const float ss = warp_reduce_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
        const float mult = is_v
            ? rsqrtf(ss / 128.0f + eps)
            : fminf(1.0f, threshold / (sqrtf(ss) + eps));
        char * dst_row = tree_kv + (size_t) plane * kv_nb3 + (size_t) h * kv_nb2 + (size_t) slot * kv_nb1;
        if constexpr (KV_Q8_0) {
            block_q8_0 * bp = reinterpret_cast<block_q8_0 *>(dst_row);
            const float vs[4] = { v0 * mult, v1 * mult, v2 * mult, v3 * mult };
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float amax = warp_reduce_max(fabsf(vs[b]));
                const float d_q = amax / 127.0f;
                bp[b].qs[lane] = (amax == 0.0f) ? 0 : (int8_t) __float2int_rn(vs[b] / d_q);
                if (lane == 0) bp[b].d = __float2half(d_q);
            }
        } else {
            half * hp = reinterpret_cast<half *>(dst_row);
            hp[lane +  0] = __float2half(v0 * mult);
            hp[lane + 32] = __float2half(v1 * mult);
            hp[lane + 64] = __float2half(v2 * mult);
            hp[lane + 96] = __float2half(v3 * mult);
        }
    }
}

// ─── 42. Fused MoE Masked-Sum (+ Peer Join) + Tree FFN Post-Conv + Residual Add ──
template <bool PEER_F16>
static __global__ void inkling2_moe_masked_sum_tree_sconv_add_kernel(
        const float   * __restrict__ down,        // [H, K, T]
        const float   * __restrict__ wsel,        // [K, T]
        const int32_t * __restrict__ mask_sel,    // [K, T]
        const void    * __restrict__ peer_sum,    // nullptr or [H, T] (F16 or F32)
        const float   * __restrict__ w_static,    // [4, H]
        const float   * __restrict__ prefix_conv, // [3, S_ch, P]
        float         * __restrict__ tree_conv,   // [S_ch, S_max, P]
        const int32_t * __restrict__ meta,        // [m_stride, T]
        const float   * __restrict__ addend, size_t add_stride, // [H, T]
        float         * __restrict__ dst,         // [H, T]
        int H, int K, int T, int wsel_ld, int msel_ld,
        int s_ch, int s_max, int chan_offset, int m_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= H || t >= T) return;

    auto eval_moe = [&](int row) -> float {
        const float   * dp = down     + (size_t) row * K * H + c;
        const float   * wp = wsel     + (size_t) row * wsel_ld;
        const int32_t * mp = mask_sel + (size_t) row * msel_ld;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k < K && mp[k] >= 0) {
                acc += wp[k] * dp[(size_t) k * H];
            }
        }
        if (peer_sum) {
            const size_t pidx = (size_t) row * H + c;
            if constexpr (PEER_F16) {
                acc += __half2float(reinterpret_cast<const half *>(peer_sum)[pidx]);
            } else {
                acc += reinterpret_cast<const float *>(peer_sum)[pidx];
            }
        }
        return acc;
    };

    const int32_t * mt = meta + (size_t) t * m_stride;
    const int slot  = mt[0];
    const int plane = mt[2];
    const int sc    = chan_offset + c;

    const float * pc_plane = prefix_conv + (size_t) plane * s_ch * 3 + (size_t) sc * 3;
    float * tc_plane       = tree_conv   + (size_t) plane * s_max * s_ch;

    const float cur = eval_moe(t);
    float hist[3];
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int hr = mt[5 + j];
        if (hr >= 0) {
            hist[j] = eval_moe(hr);
        } else {
            const int hs = mt[8 + j];
            hist[j] = (hs >= 0) ? tc_plane[(size_t) hs * s_ch + sc] : pc_plane[-(hs + 1)];
        }
    }

    const float * wc = w_static + (size_t) c * 4;
    const float acc  = wc[0] * hist[0] + wc[1] * hist[1] + wc[2] * hist[2] + wc[3] * cur;
    dst[(size_t) t * H + c] = (cur + acc) * rsqrtf(5.0f) + addend[(size_t) t * add_stride + c];
    tc_plane[(size_t) slot * s_ch + sc] = cur;
}

// ─── 43. Prefill Coda Plane Clone (Depth-3 Plane P-1 -> Planes 0..P-2 in ~40 us!) ─
static __global__ void inkling2_tree_prefill_clone_kv_kernel(
        char * __restrict__ prefix_k, size_t pk_nb1, size_t pk_nb2, size_t pk_nb3,
        char * __restrict__ prefix_v, size_t pv_nb1, size_t pv_nb2, size_t pv_nb3,
        int pos0, int n_tokens, int prefix_cap, int swa_window, int P, int row_u32) {
    const int h   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;
    if (t >= n_tokens || P <= 1) return;
    const int pos  = pos0 + t;
    const int slot = (swa_window > 0) ? (pos % prefix_cap) : pos;
    if (slot < 0 || slot >= prefix_cap) return;

    const int src_p = P - 1;
    const uint32_t * sk = reinterpret_cast<const uint32_t *>(
        prefix_k + (size_t) src_p * pk_nb3 + (size_t) h * pk_nb2 + (size_t) slot * pk_nb1);
    const uint32_t * sv = reinterpret_cast<const uint32_t *>(
        prefix_v + (size_t) src_p * pv_nb3 + (size_t) h * pv_nb2 + (size_t) slot * pv_nb1);
    for (int i = tid; i < row_u32; i += blockDim.x) {
        const uint32_t vk = sk[i];
        const uint32_t vv = sv[i];
        for (int p = 0; p < src_p; ++p) {
            uint32_t * dk = reinterpret_cast<uint32_t *>(
                prefix_k + (size_t) p * pk_nb3 + (size_t) h * pk_nb2 + (size_t) slot * pk_nb1);
            uint32_t * dv = reinterpret_cast<uint32_t *>(
                prefix_v + (size_t) p * pv_nb3 + (size_t) h * pv_nb2 + (size_t) slot * pv_nb1);
            dk[i] = vk;
            dv[i] = vv;
        }
    }
}

static __global__ void inkling2_tree_prefill_clone_conv_kernel(
        float * __restrict__ prefix_conv, // [3, s_ch, P]
        int s_ch, int P) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= s_ch || P <= 1) return;
    const int src_p = P - 1;
    const float * sp = prefix_conv + ((size_t) src_p * s_ch + c) * 3;
    const float v0 = sp[0], v1 = sp[1], v2 = sp[2];
    for (int p = 0; p < src_p; ++p) {
        float * dp = prefix_conv + ((size_t) p * s_ch + c) * 3;
        dp[0] = v0; dp[1] = v1; dp[2] = v2;
    }
}

} // namespace

void ggml_cuda_op_inkling2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int subop = ggml_get_op_params_i32(dst, 0);
    cudaStream_t stream = ctx.stream();

    switch (subop) {
        case GGML_INKLING2_ATTN_RES_MIX: {
            const ggml_tensor * cur    = dst->src[0];
            const ggml_tensor * query  = dst->src[1];
            const ggml_tensor * norm_w = dst->src[2];
            const int n_blocks = ggml_get_op_params_i32(dst, 1);
            const int n_q      = ggml_get_op_params_i32(dst, 2);
            const float eps    = ggml_get_op_params_f32(dst, 3);
            const int n_embd   = (int) cur->ne[0];
            const int n_tokens = (int) cur->ne[1];
            GGML_ASSERT(n_blocks + 1 <= 8 && n_q <= 4);

            AttnResPtrs ptrs{};
            for (int i = 0; i < n_blocks; ++i) {
                ptrs.ptr[i] = (const float *) dst->src[3 + i]->data;
                ptrs.row_stride_elems[i] = dst->src[3 + i]->nb[1] / sizeof(float);
            }
            ptrs.ptr[n_blocks] = (const float *) cur->data;
            ptrs.row_stride_elems[n_blocks] = cur->nb[1] / sizeof(float);

            const int norm_per_q = (norm_w && norm_w->ne[1] == n_q && n_q > 1) ? 1 : 0;
            dim3 grid(n_tokens, n_q, 1);
            inkling2_attn_res_mix_kernel<<<grid, 256, 0, stream>>>(
                ptrs, n_blocks + 1,
                (const float *) query->data,
                norm_w ? (const float *) norm_w->data : nullptr,
                norm_per_q,
                (float *) dst->data,
                n_embd, n_q, eps);
            break;
        }
        case GGML_INKLING2_LOOP_CTRL_PRE: {
            const ggml_tensor * cur = dst->src[0];
            const ggml_tensor * msg = dst->src[1];
            const float eps = ggml_get_op_params_f32(dst, 1);
            const int n_embd   = (int) cur->ne[0];
            const int n_tokens = (int) msg->ne[1];
            const int n_groups = (int) msg->ne[2];
            dim3 grid(n_tokens, n_groups, 1);
            inkling2_loop_ctrl_pre_kernel<<<grid, 256, 0, stream>>>(
                (const float *) cur->data, cur->nb[1] / sizeof(float), (cur->ne[2] > 1) ? (cur->nb[2] / sizeof(float)) : 0,
                (const float *) msg->data, msg->nb[1] / sizeof(float), (msg->ne[2] > 1) ? (msg->nb[2] / sizeof(float)) : 0,
                (float *) dst->data, dst->nb[1] / sizeof(float), (dst->ne[2] > 1) ? (dst->nb[2] / sizeof(float)) : 0,
                n_embd, eps);
            break;
        }
        case GGML_INKLING2_LOOP_CTRL_POST: {
            const ggml_tensor * cur    = dst->src[0];
            const ggml_tensor * msg    = dst->src[1];
            const ggml_tensor * gf     = dst->src[2];
            const ggml_tensor * norm_w = dst->src[3];
            const ggml_tensor * flags  = dst->src[4];
            const float eps = ggml_get_op_params_f32(dst, 1);
            const int n_embd   = (int) cur->ne[0];
            const int n_tokens = (int) msg->ne[1];
            const int n_groups = (int) msg->ne[2];
            dim3 grid(n_tokens, n_groups, 1);
            inkling2_loop_ctrl_post_kernel<<<grid, 256, 0, stream>>>(
                (const float *) cur->data, cur->nb[1] / sizeof(float), (cur->ne[2] > 1) ? (cur->nb[2] / sizeof(float)) : 0,
                (const float *) msg->data, msg->nb[1] / sizeof(float), (msg->ne[2] > 1) ? (msg->nb[2] / sizeof(float)) : 0,
                (const float *) gf->data,  gf->nb[1]  / sizeof(float), (gf->ne[2]  > 1) ? (gf->nb[2]  / sizeof(float)) : 0,
                norm_w ? (const float *) norm_w->data : nullptr,
                (norm_w && norm_w->ne[1] > 1) ? (norm_w->nb[1] / sizeof(float)) : 0,
                flags ? (const int32_t *) flags->data : nullptr,
                (float *) dst->data, dst->nb[1] / sizeof(float), (dst->ne[2] > 1) ? (dst->nb[2] / sizeof(float)) : 0,
                n_embd, eps);
            break;
        }
        case GGML_INKLING2_SCONV: {
            const ggml_tensor * x         = dst->src[0];
            const ggml_tensor * w_static  = dst->src[1];
            const ggml_tensor * state_in  = dst->src[2];
            ggml_tensor       * state_out = dst->src[3];
            const ggml_tensor * dyn_taps  = dst->src[4];
            const ggml_tensor * addend    = dst->src[5];
            const int state_chan_offset = ggml_get_op_params_i32(dst, 1);
            const int head_dim_dyn      = ggml_get_op_params_i32(dst, 2);
            const int write_mode        = ggml_get_op_params_i32(dst, 3);
            const int n_ch     = (int) x->ne[0];
            const int n_tokens = (int) x->ne[1];
            const int state_stride = (int) state_in->ne[1];
            if (n_tokens > 8 && write_mode <= 1) {
                dim3 grid((n_ch + 255) / 256, (n_tokens + 15) / 16, 1);
                inkling2_sconv_parallel_kernel<<<grid, 256, 0, stream>>>(
                    (const float *) x->data, x->nb[1] / sizeof(float),
                    (const float *) w_static->data,
                    (const float *) state_in->data,
                    state_out ? (float *) state_out->data : nullptr,
                    dyn_taps ? (const float *) dyn_taps->data : nullptr,
                    dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                    addend ? (const float *) addend->data : nullptr,
                    addend ? (addend->nb[1] / sizeof(float)) : 0,
                    (float *) dst->data,
                    n_ch, n_tokens, state_chan_offset, head_dim_dyn, write_mode);
                if (write_mode == 1 && state_out) {
                    inkling2_sconv_commit_state_kernel<<<(n_ch + 255) / 256, 256, 0, stream>>>(
                        (const float *) x->data, x->nb[1] / sizeof(float),
                        (float *) state_out->data, n_ch, n_tokens, state_chan_offset);
                }
            } else {
                const int blocks = (n_ch + 255) / 256;
                inkling2_sconv_kernel<<<blocks, 256, 0, stream>>>(
                    (const float *) x->data, x->nb[1] / sizeof(float),
                    (const float *) w_static->data,
                    (const float *) state_in->data,
                    state_out ? (float *) state_out->data : nullptr,
                    dyn_taps ? (const float *) dyn_taps->data : nullptr,
                    dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                    addend ? (const float *) addend->data : nullptr,
                    addend ? (addend->nb[1] / sizeof(float)) : 0,
                    (float *) dst->data,
                    n_ch, n_tokens, state_stride, state_chan_offset, head_dim_dyn, write_mode);
            }
            break;
        }
        case GGML_INKLING2_CLIP_NORM: {
            const ggml_tensor * x       = dst->src[0];
            const ggml_tensor * scale_w = dst->src[1];
            const ggml_tensor * pos_buf = dst->src[2];
            const int head_dim   = ggml_get_op_params_i32(dst, 1);
            const int n_head     = ggml_get_op_params_i32(dst, 2);
            const int mode       = ggml_get_op_params_i32(dst, 3);
            const float thresh   = ggml_get_op_params_f32(dst, 4);
            const float eps      = ggml_get_op_params_f32(dst, 5);
            const int log_floor  = ggml_get_op_params_i32(dst, 6);
            const float log_a    = ggml_get_op_params_f32(dst, 7);
            const int n_tokens   = (int) x->ne[1];
            dim3 grid(n_head, n_tokens, 1);
            inkling2_clip_norm_kernel<<<grid, WARP_SIZE, 0, stream>>>(
                (const float *) x->data, x->nb[1] / sizeof(float),
                scale_w ? (const float *) scale_w->data : nullptr,
                pos_buf ? (const int32_t *) pos_buf->data : nullptr,
                (float *) dst->data,
                head_dim, n_head, mode, thresh, eps, log_floor, log_a);
            break;
        }
        case GGML_INKLING2_KV_WRITE: {
            const ggml_tensor * x       = dst->src[0];
            ggml_tensor       * kv      = dst->src[1];
            const ggml_tensor * pos_buf = dst->src[2];
            const int head_dim   = ggml_get_op_params_i32(dst, 1);
            const int n_head_kv  = ggml_get_op_params_i32(dst, 2);
            const int is_v       = ggml_get_op_params_i32(dst, 3);
            const int swa_window = ggml_get_op_params_i32(dst, 4);
            const float thresh   = ggml_get_op_params_f32(dst, 5);
            const float eps      = ggml_get_op_params_f32(dst, 6);
            const int n_tokens   = (int) x->ne[1];
            const int kv_cap     = (int) kv->ne[1];
            dim3 grid(n_head_kv, n_tokens, 1);
            if (kv->type == GGML_TYPE_Q8_0) {
                inkling2_kv_write_q8_0_kernel<<<grid, WARP_SIZE, 0, stream>>>(
                    (const float *) x->data, x->nb[1] / sizeof(float),
                    (char *) kv->data, kv->nb[1], kv->nb[2],
                    (const int32_t *) pos_buf->data,
                    head_dim, n_head_kv, kv_cap, is_v, swa_window, thresh, eps);
            } else {
                GGML_ASSERT(kv->type == GGML_TYPE_F16);
                inkling2_kv_write_f16_kernel<<<grid, WARP_SIZE, 0, stream>>>(
                    (const float *) x->data, x->nb[1] / sizeof(float),
                    (half *) kv->data, kv->nb[1] / sizeof(half), kv->nb[2] / sizeof(half),
                    (const int32_t *) pos_buf->data,
                    head_dim, n_head_kv, kv_cap, is_v, swa_window, thresh, eps);
            }
            break;
        }
        case GGML_INKLING2_REL_TABLE: {
            const ggml_tensor * rel_bias = dst->src[0];
            const ggml_tensor * r_cur    = dst->src[1];
            const ggml_tensor * sink     = dst->src[2];
            const int d_rel      = (int) rel_bias->ne[0];
            const int rel_extent = (int) rel_bias->ne[1];
            const int n_head     = (int) r_cur->ne[1];
            const int n_tokens   = (int) r_cur->ne[2];
            dim3 grid(n_head, n_tokens, 1);
            inkling2_rel_table_kernel<<<grid, 256, 0, stream>>>(
                (const float *) rel_bias->data,
                (const float *) r_cur->data,
                (const float *) sink->data, sink->nb[1] / sizeof(float),
                (float *) dst->data,
                d_rel, rel_extent, n_head);
            break;
        }
        case GGML_INKLING2_ATTN_REL: {
            const ggml_tensor * q         = dst->src[0];
            const ggml_tensor * k_cache   = dst->src[1];
            const ggml_tensor * v_cache   = dst->src[2];
            const ggml_tensor * rel_table = dst->src[3];
            const ggml_tensor * kv_meta   = dst->src[4];
            const ggml_tensor * gate_proj = dst->src[7];
            const int swa_window   = ggml_get_op_params_i32(dst, 1);
            const int rel_extent   = ggml_get_op_params_i32(dst, 2);
            const int partial_out  = ggml_get_op_params_i32(dst, 3);
            const int include_sink = ggml_get_op_params_i32(dst, 4);

            const int head_dim = (int) q->ne[0];
            const int Hq       = (int) q->ne[1];
            const int T        = (int) q->ne[2];
            const int kv_cap   = (int) k_cache->ne[1];
            const int Hk       = (int) k_cache->ne[2];
            GGML_ASSERT(head_dim == 128 && Hq == Hk * 3);
            const bool is_q8 = (k_cache->type == GGML_TYPE_Q8_0);
            const float * gate_ptr = gate_proj ? (const float *) gate_proj->data : nullptr;
            const size_t gate_stride = gate_proj ? (gate_proj->nb[1] / sizeof(float)) : 0;

            if (T > 16) {
#if defined(GGML_USE_HIP)
                const int cc = ggml_cuda_info().devices[ctx.device].cc;
                const bool has_wmma = (cc >= GGML_CUDA_CC_RDNA3);
                if (has_wmma) {
                    constexpr int BM = 128;
                    const int total_q = T * 3; // GQA=3
                    dim3 grid(Hk, (total_q + BM - 1) / BM, 1);
                    dim3 block(WARP_SIZE, 8, 1);
                    if (is_q8) {
                        inkling2_attn_rel_prefill_wmma_kernel<true, 3><<<grid, block, 0, stream>>>(
                            (const float *) q->data,
                            (const char *) k_cache->data, (const char *) v_cache->data,
                            k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                            (const float *) rel_table->data,
                            (const int32_t *) kv_meta->data,
                            gate_ptr, gate_stride, (float *) dst->data,
                            Hq, Hk, T, kv_cap, swa_window, rel_extent, partial_out, include_sink);
                    } else {
                        inkling2_attn_rel_prefill_wmma_kernel<false, 3><<<grid, block, 0, stream>>>(
                            (const float *) q->data,
                            (const char *) k_cache->data, (const char *) v_cache->data,
                            k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                            (const float *) rel_table->data,
                            (const int32_t *) kv_meta->data,
                            gate_ptr, gate_stride, (float *) dst->data,
                            Hq, Hk, T, kv_cap, swa_window, rel_extent, partial_out, include_sink);
                    }
                    break;
                }
#endif
                constexpr int BM = 8;
                dim3 grid(Hk, (T + BM - 1) / BM, 1);
                dim3 block(WARP_SIZE, BM, 1);
                if (is_q8) {
                    inkling2_attn_rel_prefill_kernel<true, 3><<<grid, block, 0, stream>>>(
                        (const float *) q->data,
                        (const char *) k_cache->data, (const char *) v_cache->data,
                        k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                        (const float *) rel_table->data,
                        (const int32_t *) kv_meta->data,
                        gate_ptr, gate_stride, (float *) dst->data,
                        Hq, Hk, T, kv_cap, swa_window, rel_extent, partial_out, include_sink);
                } else {
                    inkling2_attn_rel_prefill_kernel<false, 3><<<grid, block, 0, stream>>>(
                        (const float *) q->data,
                        (const char *) k_cache->data, (const char *) v_cache->data,
                        k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                        (const float *) rel_table->data,
                        (const int32_t *) kv_meta->data,
                        gate_ptr, gate_stride, (float *) dst->data,
                        Hq, Hk, T, kv_cap, swa_window, rel_extent, partial_out, include_sink);
                }
            } else if (swa_window > 0) {
                // 1-Pass Single-Kernel SWA Attention: ZERO scratch buffer, ZERO reduce_splits launch!
                dim3 grid(Hk, T, 1);
                if (T <= 2) {
                    dim3 block(WARP_SIZE, 8, 1);
                    if (is_q8) {
                        inkling2_attn_rel_single_kernel<true, 3, 8><<<grid, block, 0, stream>>>(
                            (const float *) q->data,
                            (const char *) k_cache->data, (const char *) v_cache->data,
                            k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                            (const float *) rel_table->data, (const int32_t *) kv_meta->data,
                            gate_ptr, gate_stride, (float *) dst->data,
                            Hq, Hk, kv_cap, swa_window, rel_extent, partial_out, include_sink);
                    } else {
                        inkling2_attn_rel_single_kernel<false, 3, 8><<<grid, block, 0, stream>>>(
                            (const float *) q->data,
                            (const char *) k_cache->data, (const char *) v_cache->data,
                            k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                            (const float *) rel_table->data, (const int32_t *) kv_meta->data,
                            gate_ptr, gate_stride, (float *) dst->data,
                            Hq, Hk, kv_cap, swa_window, rel_extent, partial_out, include_sink);
                    }
                } else {
                    dim3 block(WARP_SIZE, 4, 1);
                    if (is_q8) {
                        inkling2_attn_rel_single_kernel<true, 3, 4><<<grid, block, 0, stream>>>(
                            (const float *) q->data,
                            (const char *) k_cache->data, (const char *) v_cache->data,
                            k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                            (const float *) rel_table->data, (const int32_t *) kv_meta->data,
                            gate_ptr, gate_stride, (float *) dst->data,
                            Hq, Hk, kv_cap, swa_window, rel_extent, partial_out, include_sink);
                    } else {
                        inkling2_attn_rel_single_kernel<false, 3, 4><<<grid, block, 0, stream>>>(
                            (const float *) q->data,
                            (const char *) k_cache->data, (const char *) v_cache->data,
                            k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                            (const float *) rel_table->data, (const int32_t *) kv_meta->data,
                            gate_ptr, gate_stride, (float *) dst->data,
                            Hq, Hk, kv_cap, swa_window, rel_extent, partial_out, include_sink);
                    }
                }
            } else {
                // Global layers: Split-KV FlashDecoding (HIP-graph invariant across 128K-262K context!)
                const int num_splits = (T <= 2 ? 16 : (T <= 8 ? 8 : 4));
                ggml_cuda_pool_alloc<float> part_buf(ctx.pool(), (size_t) num_splits * T * Hq * 130);
                dim3 grid(Hk, T, num_splits);
                dim3 block(WARP_SIZE, 4, 1);
                if (is_q8) {
                    inkling2_attn_rel_split_kernel<true, 3><<<grid, block, 0, stream>>>(
                        (const float *) q->data,
                        (const char *) k_cache->data, (const char *) v_cache->data,
                        k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                        (const float *) rel_table->data,
                        (const int32_t *) kv_meta->data,
                        part_buf.get(), Hq, Hk, kv_cap, swa_window, rel_extent, num_splits, include_sink);
                } else {
                    inkling2_attn_rel_split_kernel<false, 3><<<grid, block, 0, stream>>>(
                        (const float *) q->data,
                        (const char *) k_cache->data, (const char *) v_cache->data,
                        k_cache->nb[1], k_cache->nb[2], v_cache->nb[1], v_cache->nb[2],
                        (const float *) rel_table->data,
                        (const int32_t *) kv_meta->data,
                        part_buf.get(), Hq, Hk, kv_cap, swa_window, rel_extent, num_splits, include_sink);
                }
                dim3 rgrid(Hq, T, 1);
                inkling2_attn_rel_reduce_splits_kernel<<<rgrid, WARP_SIZE, 0, stream>>>(
                    part_buf.get(), gate_ptr, gate_stride, (float *) dst->data,
                    Hq, T, num_splits, partial_out);
            }
            break;
        }
        case GGML_INKLING2_ATTN_SPLIT_COMBINE: {
            const ggml_tensor * part_a   = dst->src[0];
            const ggml_tensor * part_b   = dst->src[1];
            const ggml_tensor * gate_proj = dst->src[2];
            const int head_dim = ggml_get_op_params_i32(dst, 1);
            const int n_head   = ggml_get_op_params_i32(dst, 2);
            const int T        = (int) part_a->ne[1];
            dim3 grid(n_head, T, 1);
            inkling2_attn_split_combine_kernel<<<grid, WARP_SIZE, 0, stream>>>(
                (const float *) part_a->data,
                (const float *) part_b->data,
                gate_proj ? (const float *) gate_proj->data : nullptr,
                gate_proj ? (gate_proj->nb[1] / sizeof(float)) : 0,
                (float *) dst->data, head_dim, n_head);
            break;
        }
        case GGML_INKLING2_SITU: {
            const ggml_tensor * gu = dst->src[0];
            const float beta = ggml_get_op_params_f32(dst, 1);
            const float lb   = ggml_get_op_params_f32(dst, 2);
            const int n_ff   = (int) (gu->ne[0] / 2);
            const int n_rows = (int) (gu->ne[1] * gu->ne[2]);
            const int total  = n_ff * n_rows;
            const int blocks = (total + 255) / 256;
            inkling2_situ_kernel<<<blocks, 256, 0, stream>>>(
                (const float *) gu->data, (float *) dst->data,
                n_ff, n_rows, 1.0f / beta, beta, lb > 0.0f ? (1.0f / lb) : 0.0f, lb);
            break;
        }
        case GGML_INKLING2_SIGMOID_TOPK_IDS: {
            const ggml_tensor * logits = dst->src[0];
            const ggml_tensor * bias   = dst->src[1];
            const int top_k = ggml_get_op_params_i32(dst, 1);
            const int n_exp = (int) logits->ne[0];
            const int T     = (int) logits->ne[1];
            inkling2_sigmoid_topk_ids_kernel<<<T, WARP_SIZE, 0, stream>>>(
                (const float *) logits->data,
                bias ? (const float *) bias->data : nullptr,
                (int32_t *) dst->data, n_exp, top_k);
            break;
        }
        case GGML_INKLING2_SIGMOID_TOPK_WTS: {
            const ggml_tensor * logits = dst->src[0];
            const ggml_tensor * ids    = dst->src[1];
            const float route_scale = ggml_get_op_params_f32(dst, 1);
            const float eps         = ggml_get_op_params_f32(dst, 2);
            const int n_exp = (int) logits->ne[0];
            const int top_k = (int) ids->ne[0];
            const int T     = (int) ids->ne[1];
            inkling2_sigmoid_topk_wts_kernel<<<T, WARP_SIZE, 0, stream>>>(
                (const float *) logits->data, (const int32_t *) ids->data,
                (float *) dst->data, n_exp, top_k, route_scale, eps);
            break;
        }
        case GGML_INKLING2_SOFTCAP_SLICE: {
            const ggml_tensor * logits = dst->src[0];
            const float softcap = ggml_get_op_params_f32(dst, 1);
            const int n_vocab = (int) dst->ne[0];
            const int T       = (int) dst->ne[1];
            const int64_t total = (int64_t) n_vocab * T;
            const int blocks = (int) ((total + 255) / 256);
            inkling2_softcap_slice_kernel<<<blocks, 256, 0, stream>>>(
                (const float *) logits->data, logits->nb[1] / sizeof(float),
                (float *) dst->data, n_vocab, T, softcap);
            break;
        }
        case GGML_INKLING2_DSPARK_MIX_NORM: {
            const ggml_tensor * taps   = dst->src[0];
            const ggml_tensor * alpha  = dst->src[1];
            const ggml_tensor * weight = dst->src[2];
            const float eps = ggml_get_op_params_f32(dst, 1);
            const int n_embd = (int) taps->ne[0];
            const int n_taps = (int) taps->ne[1];
            const int T      = (int) taps->ne[2];
            inkling2_dspark_mix_norm_kernel<<<T, 256, 0, stream>>>(
                (const float *) taps->data,
                (const float *) alpha->data,
                (const float *) weight->data,
                (float *) dst->data, n_embd, n_taps, eps);
            break;
        }
        case GGML_INKLING2_DSPARK_MARKOV_GATE: {
            const ggml_tensor * raw   = dst->src[0];
            const ggml_tensor * state = dst->src[1];
            const int rank = (int) state->ne[0];
            const int B    = (int) state->ne[1];
            inkling2_dspark_markov_gate_kernel<<<B, rank, 0, stream>>>(
                (const float *) raw->data, (const float *) state->data,
                (float *) dst->data, rank);
            break;
        }
        case GGML_INKLING2_DSPARK_ADD_SOFTCAP: {
            const ggml_tensor * base_logits = dst->src[0];
            const ggml_tensor * bias        = dst->src[1];
            const float softcap = ggml_get_op_params_f32(dst, 1);
            const int n_vocab = (int) dst->ne[0];
            const int B       = (int) dst->ne[1];
            const int64_t total = (int64_t) n_vocab * B;
            const int blocks = (int) ((total + 255) / 256);
            inkling2_dspark_add_softcap_kernel<<<blocks, 256, 0, stream>>>(
                (const float *) base_logits->data, base_logits->nb[1] / sizeof(float),
                (const float *) bias->data, bias->nb[1] / sizeof(float),
                (float *) dst->data, n_vocab, B, softcap);
            break;
        }
        case GGML_INKLING2_STATE_RESTORE: {
            const ggml_tensor * ring = dst->src[0];
            ggml_tensor       * live = dst->src[1];
            const ggml_tensor * slot = dst->src[2];
            const int64_t n_elems = ggml_nelements(live);
            const int blocks = (int) ((n_elems + 255) / 256);
            inkling2_state_restore_kernel<<<blocks, 256, 0, stream>>>(
                (const float *) ring->data, (float *) live->data,
                (const int32_t *) slot->data, n_elems);
            break;
        }
        case GGML_INKLING2_DYNAMIC_SPLIT_LUT: {
            const ggml_tensor * sel = dst->src[0];
            const int n_expert   = ggml_get_op_params_i32(dst, 1);
            const int target_pct = ggml_get_op_params_i32(dst, 2);
            const int n_used     = (int) sel->ne[0];
            const int n_tokens   = (int) sel->ne[1];
            const int sel_ld     = (int) (sel->nb[1] / sizeof(int32_t));
            inkling2_dynamic_split_lut_kernel<<<1, 128, 0, stream>>>(
                (const int32_t *) sel->data, sel_ld, (int32_t *) dst->data,
                n_used, n_tokens, n_expert, target_pct);
            break;
        }
        case GGML_INKLING2_DSPARK_TOPK_CANDS: {
            const ggml_tensor * base_logits = dst->src[0];
            const int n_vocab = ggml_get_op_params_i32(dst, 1);
            const int B       = (int) base_logits->ne[1];
            dim3 grid(256, B, 1);
            inkling2_dspark_topk_cands_kernel<<<grid, 256, 0, stream>>>(
                (const float *) base_logits->data, base_logits->nb[1] / sizeof(float),
                (int32_t *) dst->data, n_vocab);
            break;
        }
        case GGML_INKLING2_DSPARK_TOPK_SCORE: {
            const ggml_tensor * cand_ids    = dst->src[0];
            const ggml_tensor * base_row    = dst->src[1];
            const ggml_tensor * latent      = dst->src[2];
            const ggml_tensor * fold_weight = dst->src[3];
            const ggml_tensor * fold_bias   = dst->src[4];
            const float softcap = ggml_get_op_params_f32(dst, 1);
            const int rank = (int) latent->ne[0];
            ggml_cuda_pool_alloc<float> score_buf(ctx.pool(), 1024);
            dim3 grid(256, 1, 1);
            dim3 block(WARP_SIZE, 4, 1);
            inkling2_dspark_topk_score_kernel<<<grid, block, 0, stream>>>(
                (const int32_t *) cand_ids->data,
                (const float *) base_row->data,
                (const float *) latent->data,
                (const half *) fold_weight->data,
                (const float *) fold_bias->data,
                score_buf.get(), rank, softcap);
            inkling2_dspark_topk_argmax_kernel<<<1, 256, 0, stream>>>(
                (const int32_t *) cand_ids->data, score_buf.get(), (int32_t *) dst->data);
            break;
        }
        case GGML_INKLING2_MOE_MASKED_SUM: {
            const ggml_tensor * down     = dst->src[0];
            const ggml_tensor * wsel     = dst->src[1];
            const ggml_tensor * mask_sel = dst->src[2];
            const ggml_tensor * peer_sum = dst->src[3];
            const int out_f16 = ggml_get_op_params_i32(dst, 1);
            const int H = (int) down->ne[0];
            const int K = (int) down->ne[1];
            const int T = (int) down->ne[2];
            const int wsel_ld = (int) (wsel->nb[1] / sizeof(float));
            const int msel_ld = (int) (mask_sel->nb[1] / sizeof(int32_t));
            const bool peer_f16 = (peer_sum && peer_sum->type == GGML_TYPE_F16);
            dim3 grid((H + 255) / 256, T, 1);
            if (out_f16) {
                inkling2_moe_masked_sum_kernel<true, false><<<grid, 256, 0, stream>>>(
                    (const float *) down->data, (const float *) wsel->data, (const int32_t *) mask_sel->data,
                    peer_sum ? peer_sum->data : nullptr, dst->data, H, K, wsel_ld, msel_ld);
            } else if (peer_f16) {
                inkling2_moe_masked_sum_kernel<false, true><<<grid, 256, 0, stream>>>(
                    (const float *) down->data, (const float *) wsel->data, (const int32_t *) mask_sel->data,
                    peer_sum->data, dst->data, H, K, wsel_ld, msel_ld);
            } else {
                inkling2_moe_masked_sum_kernel<false, false><<<grid, 256, 0, stream>>>(
                    (const float *) down->data, (const float *) wsel->data, (const int32_t *) mask_sel->data,
                    peer_sum ? peer_sum->data : nullptr, dst->data, H, K, wsel_ld, msel_ld);
            }
            break;
        }
        case GGML_INKLING2_DSPARK_MARKOV_ARGMAX: {
            const ggml_tensor * base_row    = dst->src[0];
            const ggml_tensor * latent      = dst->src[1];
            const ggml_tensor * fold_weight = dst->src[2];
            const ggml_tensor * fold_bias   = dst->src[3];
            const int n_vocab = ggml_get_op_params_i32(dst, 1);
            const int rank    = (int) latent->ne[0];
            ggml_cuda_pool_alloc<float>   part_v(ctx.pool(), 256);
            ggml_cuda_pool_alloc<int32_t> part_i(ctx.pool(), 256);
            dim3 grid(256, 1, 1);
            dim3 block(WARP_SIZE, 8, 1);
            if (fold_weight->type == GGML_TYPE_Q8_0) {
                inkling2_dspark_markov_argmax_stage1_kernel<true><<<grid, block, 0, stream>>>(
                    (const float *) base_row->data, (const float *) latent->data,
                    (const char *) fold_weight->data, fold_weight->nb[1],
                    (const float *) fold_bias->data, part_v.get(), part_i.get(), n_vocab, rank);
            } else {
                inkling2_dspark_markov_argmax_stage1_kernel<false><<<grid, block, 0, stream>>>(
                    (const float *) base_row->data, (const float *) latent->data,
                    (const char *) fold_weight->data, fold_weight->nb[1],
                    (const float *) fold_bias->data, part_v.get(), part_i.get(), n_vocab, rank);
            }
            inkling2_dspark_markov_argmax_stage2_kernel<<<1, 256, 0, stream>>>(
                part_v.get(), part_i.get(), (int32_t *) dst->data);
            break;
        }
        case GGML_INKLING2_SCONV_KV_WRITE: {
            const ggml_tensor * x         = dst->src[0];
            const ggml_tensor * w_static  = dst->src[1];
            const ggml_tensor * state_in  = dst->src[2];
            ggml_tensor       * state_out = dst->src[3];
            const ggml_tensor * dyn_taps  = dst->src[4];
            ggml_tensor       * kv_cache  = dst->src[5];
            const ggml_tensor * pos_buf   = dst->src[6];
            const int state_chan_offset = ggml_get_op_params_i32(dst, 1);
            const int head_dim          = ggml_get_op_params_i32(dst, 2);
            const int n_head_kv         = ggml_get_op_params_i32(dst, 3);
            const int is_v              = ggml_get_op_params_i32(dst, 4);
            const int swa_window        = ggml_get_op_params_i32(dst, 5);
            const float threshold       = ggml_get_op_params_f32(dst, 6);
            const float eps             = ggml_get_op_params_f32(dst, 7);
            const int write_mode        = ggml_get_op_params_i32(dst, 8);
            const int n_tokens     = (int) x->ne[1];
            const int kv_cap       = (int) kv_cache->ne[1];
            const int state_stride = (int) state_in->ne[1];
            GGML_ASSERT(head_dim == 128);
            if (n_tokens > 8 && write_mode <= 1) {
                dim3 grid(n_head_kv, (n_tokens + 7) / 8, 1);
                if (kv_cache->type == GGML_TYPE_Q8_0) {
                    inkling2_sconv_kv_write_parallel_kernel<true><<<grid, 128, 0, stream>>>(
                        (const float *) x->data, x->nb[1] / sizeof(float),
                        (const float *) w_static->data, (const float *) state_in->data,
                        state_out ? (float *) state_out->data : nullptr,
                        dyn_taps ? (const float *) dyn_taps->data : nullptr,
                        dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                        (char *) kv_cache->data, kv_cache->nb[1], kv_cache->nb[2],
                        (const int32_t *) pos_buf->data,
                        head_dim, n_head_kv, n_tokens, kv_cap, state_chan_offset,
                        is_v, swa_window, threshold, eps, write_mode);
                } else {
                    inkling2_sconv_kv_write_parallel_kernel<false><<<grid, 128, 0, stream>>>(
                        (const float *) x->data, x->nb[1] / sizeof(float),
                        (const float *) w_static->data, (const float *) state_in->data,
                        state_out ? (float *) state_out->data : nullptr,
                        dyn_taps ? (const float *) dyn_taps->data : nullptr,
                        dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                        (char *) kv_cache->data, kv_cache->nb[1], kv_cache->nb[2],
                        (const int32_t *) pos_buf->data,
                        head_dim, n_head_kv, n_tokens, kv_cap, state_chan_offset,
                        is_v, swa_window, threshold, eps, write_mode);
                }
                if (write_mode == 1 && state_out) {
                    const int n_ch = n_head_kv * head_dim;
                    inkling2_sconv_commit_state_kernel<<<(n_ch + 255) / 256, 256, 0, stream>>>(
                        (const float *) x->data, x->nb[1] / sizeof(float),
                        (float *) state_out->data, n_ch, n_tokens, state_chan_offset);
                }
            } else if (kv_cache->type == GGML_TYPE_Q8_0) {
                inkling2_sconv_kv_write_kernel<true><<<n_head_kv, 128, 0, stream>>>(
                    (const float *) x->data, x->nb[1] / sizeof(float),
                    (const float *) w_static->data, (const float *) state_in->data,
                    state_out ? (float *) state_out->data : nullptr,
                    dyn_taps ? (const float *) dyn_taps->data : nullptr,
                    dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                    (char *) kv_cache->data, kv_cache->nb[1], kv_cache->nb[2],
                    (const int32_t *) pos_buf->data,
                    head_dim, n_head_kv, n_tokens, kv_cap, state_stride, state_chan_offset,
                    is_v, swa_window, threshold, eps, write_mode);
            } else {
                inkling2_sconv_kv_write_kernel<false><<<n_head_kv, 128, 0, stream>>>(
                    (const float *) x->data, x->nb[1] / sizeof(float),
                    (const float *) w_static->data, (const float *) state_in->data,
                    state_out ? (float *) state_out->data : nullptr,
                    dyn_taps ? (const float *) dyn_taps->data : nullptr,
                    dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                    (char *) kv_cache->data, kv_cache->nb[1], kv_cache->nb[2],
                    (const int32_t *) pos_buf->data,
                    head_dim, n_head_kv, n_tokens, kv_cap, state_stride, state_chan_offset,
                    is_v, swa_window, threshold, eps, write_mode);
            }
            break;
        }
        case GGML_INKLING2_CLIP_R_REL_TABLE: {
            const ggml_tensor * r_raw    = dst->src[0];
            const ggml_tensor * r_clip_w = dst->src[1];
            const ggml_tensor * pos_buf  = dst->src[2];
            const ggml_tensor * rel_bias = dst->src[3];
            const ggml_tensor * sink     = dst->src[4];
            const int d_rel      = ggml_get_op_params_i32(dst, 1);
            const int n_head     = ggml_get_op_params_i32(dst, 2);
            const float thresh   = ggml_get_op_params_f32(dst, 3);
            const float eps      = ggml_get_op_params_f32(dst, 4);
            const int log_floor  = ggml_get_op_params_i32(dst, 5);
            const float log_a    = ggml_get_op_params_f32(dst, 6);
            const int rel_extent = (int) rel_bias->ne[1];
            const int n_tokens   = (int) r_raw->ne[1];
            dim3 grid(n_head, n_tokens, 1);
            inkling2_clip_r_rel_table_kernel<<<grid, 256, 0, stream>>>(
                (const float *) r_raw->data, r_raw->nb[1] / sizeof(float),
                r_clip_w ? (const float *) r_clip_w->data : nullptr,
                pos_buf ? (const int32_t *) pos_buf->data : nullptr,
                (const float *) rel_bias->data,
                (const float *) sink->data, sink->nb[1] / sizeof(float),
                (float *) dst->data, d_rel, rel_extent, n_head, thresh, eps, log_floor, log_a);
            break;
        }
        case 24: { // FUSED_ROUTER_Q8_TOPK (when src[1]==nullptr, src[0] is precomputed logits from fast MMQ!)
            const ggml_tensor * src0 = dst->src[0];
            const ggml_tensor * cur  = dst->src[1];
            const ggml_tensor * bias = dst->src[2];
            const int top_k         = ggml_get_op_params_i32(dst, 1);
            const float route_scale = ggml_get_op_params_f32(dst, 2);
            const float eps         = ggml_get_op_params_f32(dst, 3);
            const int target_pct    = (dst->ne[2] == 3) ? ggml_get_op_params_i32(dst, 4) : 0;

            const float * logits_ptr = nullptr;
            int n_exp = 0, T = 0;
            ggml_cuda_pool_alloc<float> tmp_logits;
            if (cur == nullptr) {
                // src0 is already computed F32 logits [n_exp, T] from fast luce_mmq_twin_rows32!
                logits_ptr = (const float *) src0->data;
                n_exp = (int) src0->ne[0];
                T     = (int) src0->ne[1];
            } else {
                const int n_embd = (int) src0->ne[0];
                n_exp = (int) src0->ne[1];
                T     = (int) cur->ne[1];
                tmp_logits.alloc(ctx.pool(), (size_t) n_exp * T);
                const int n_blocks = (n_exp + 1) / 2;
                dim3 block(WARP_SIZE, 2, 1);
                if (T == 1) {
                    inkling2_router_q8_gemv_kernel<1><<<n_blocks, block, 0, stream>>>(
                        (const char *) src0->data, src0->nb[1],
                        (const float *) cur->data, cur->nb[1] / sizeof(float),
                        tmp_logits.get(), n_embd, n_exp, 1);
                } else if (T <= 8) {
                    inkling2_router_q8_gemv_kernel<8><<<n_blocks, block, 0, stream>>>(
                        (const char *) src0->data, src0->nb[1],
                        (const float *) cur->data, cur->nb[1] / sizeof(float),
                        tmp_logits.get(), n_embd, n_exp, T);
                } else {
                    inkling2_router_q8_gemv_kernel<16><<<n_blocks, block, 0, stream>>>(
                        (const char *) src0->data, src0->nb[1],
                        (const float *) cur->data, cur->nb[1] / sizeof(float),
                        tmp_logits.get(), n_embd, n_exp, T);
                }
                logits_ptr = tmp_logits.get();
            }
            const ggml_tensor * flags = dst->src[3];
            if (target_pct > 0 && n_exp == 128 && top_k == 8 && T <= 64) {
                const int nthr = std::min(1024, std::max(128, T * WARP_SIZE));
                inkling2_sigmoid_topk_split_kernel<<<1, nthr, 0, stream>>>(
                    logits_ptr, bias ? (const float *) bias->data : nullptr,
                    flags ? (const int32_t *) flags->data : nullptr,
                    (int32_t *) dst->data, T, target_pct, route_scale, eps);
            } else {
                inkling2_sigmoid_topk_packed_kernel<<<T, WARP_SIZE, 0, stream>>>(
                    logits_ptr, bias ? (const float *) bias->data : nullptr,
                    flags ? (const int32_t *) flags->data : nullptr,
                    (int32_t *) dst->data, n_exp, top_k, T, route_scale, eps);
            }
            break;
        }
        case 25: { // FUSED_LOOP_CTRL_PRE_DOWN
            const ggml_tensor * cur  = dst->src[0];
            const ggml_tensor * msg  = dst->src[1];
            const ggml_tensor * w_q8 = dst->src[2];
            const float eps = ggml_get_op_params_f32(dst, 1);
            const int n_embd   = (int) cur->ne[0];
            const int rank     = (int) w_q8->ne[1];
            const int n_tokens = (int) msg->ne[1];
            const int n_groups = (int) msg->ne[2];
            dim3 grid(n_tokens, n_groups, (rank + 15) / 16);
            dim3 block(WARP_SIZE, 16, 1);
            inkling2_fused_loop_ctrl_pre_down_kernel<<<grid, block, 0, stream>>>(
                (const float *) cur->data, cur->nb[1] / sizeof(float), (cur->ne[2] > 1) ? (cur->nb[2] / sizeof(float)) : 0,
                (const float *) msg->data, msg->nb[1] / sizeof(float), (msg->ne[2] > 1) ? (msg->nb[2] / sizeof(float)) : 0,
                (const char *) w_q8->data, w_q8->nb[1], (w_q8->ne[2] > 1) ? w_q8->nb[2] : 0,
                (float *) dst->data, dst->nb[1] / sizeof(float), (dst->ne[2] > 1) ? (dst->nb[2] / sizeof(float)) : 0,
                n_embd, rank, eps);
            break;
        }
        case 26: { // FUSED_DYNAMIC_SPLIT_ROUTES
            const ggml_tensor * sel = dst->src[0];
            const int n_expert   = ggml_get_op_params_i32(dst, 1);
            const int target_pct = ggml_get_op_params_i32(dst, 2);
            const int n_used     = (int) sel->ne[0];
            const int n_tokens   = (int) sel->ne[1];
            const int sel_ld     = (int) (sel->nb[1] / sizeof(int32_t));
            inkling2_dynamic_split_routes_kernel<<<1, 128, 0, stream>>>(
                (const int32_t *) sel->data, sel_ld, (int32_t *) dst->data,
                n_used, n_tokens, n_expert, target_pct);
            break;
        }
        case 27: { // FUSED_SCONV_KV_WRITE_KV (1 launch for both K and V on SWA layers!)
            const ggml_tensor * x_k       = dst->src[0];
            const ggml_tensor * x_v       = dst->src[1];
            const ggml_tensor * wk_static = dst->src[2];
            const ggml_tensor * wv_static = dst->src[3];
            const ggml_tensor * state_in  = dst->src[4];
            ggml_tensor       * state_out = dst->src[5];
            ggml_tensor       * k_cache   = dst->src[6];
            ggml_tensor       * v_cache   = dst->src[7];
            const ggml_tensor * pos_buf   = dst->src[8];
            const int head_dim    = ggml_get_op_params_i32(dst, 1);
            const int n_head_kv   = ggml_get_op_params_i32(dst, 2);
            const int swa_window  = ggml_get_op_params_i32(dst, 3);
            const float threshold = ggml_get_op_params_f32(dst, 4);
            const float eps       = ggml_get_op_params_f32(dst, 5);
            const int write_mode  = ggml_get_op_params_i32(dst, 6);
            const int n_tokens     = (int) x_k->ne[1];
            const int kv_cap       = (int) k_cache->ne[1];
            const int state_stride = (int) state_in->ne[1];
            GGML_ASSERT(head_dim == 128);
            if (k_cache->type == GGML_TYPE_Q8_0) {
                inkling2_sconv_kv_write_kv_kernel<true><<<n_head_kv * 2, 128, 0, stream>>>(
                    (const float *) x_k->data, x_k->nb[1] / sizeof(float),
                    (const float *) x_v->data, x_v->nb[1] / sizeof(float),
                    (const float *) wk_static->data, (const float *) wv_static->data,
                    (const float *) state_in->data, state_out ? (float *) state_out->data : nullptr,
                    (char *) k_cache->data, k_cache->nb[1], k_cache->nb[2],
                    (char *) v_cache->data, v_cache->nb[1], v_cache->nb[2],
                    (const int32_t *) pos_buf->data,
                    head_dim, n_head_kv, n_tokens, kv_cap, state_stride,
                    swa_window, threshold, eps, write_mode);
            } else {
                inkling2_sconv_kv_write_kv_kernel<false><<<n_head_kv * 2, 128, 0, stream>>>(
                    (const float *) x_k->data, x_k->nb[1] / sizeof(float),
                    (const float *) x_v->data, x_v->nb[1] / sizeof(float),
                    (const float *) wk_static->data, (const float *) wv_static->data,
                    (const float *) state_in->data, state_out ? (float *) state_out->data : nullptr,
                    (char *) k_cache->data, k_cache->nb[1], k_cache->nb[2],
                    (char *) v_cache->data, v_cache->nb[1], v_cache->nb[2],
                    (const int32_t *) pos_buf->data,
                    head_dim, n_head_kv, n_tokens, kv_cap, state_stride,
                    swa_window, threshold, eps, write_mode);
            }
            break;
        }
        case 28: { // FUSED_MOE_MASKED_SUM_SCONV_ADD
            const ggml_tensor * down      = dst->src[0];
            const ggml_tensor * wsel      = dst->src[1];
            const ggml_tensor * mask_sel  = dst->src[2];
            const ggml_tensor * peer_sum  = dst->src[3];
            const ggml_tensor * w_static  = dst->src[4];
            const ggml_tensor * state_in  = dst->src[5];
            ggml_tensor       * state_out = dst->src[6];
            const ggml_tensor * addend    = dst->src[7];
            const int state_chan_offset = ggml_get_op_params_i32(dst, 1);
            const int write_mode        = ggml_get_op_params_i32(dst, 2);
            const int H = (int) down->ne[0];
            const int K = (int) down->ne[1];
            const int T = (int) down->ne[2];
            const int wsel_ld      = (int) (wsel->nb[1] / sizeof(float));
            const int msel_ld      = (int) (mask_sel->nb[1] / sizeof(int32_t));
            const int state_stride = (int) state_in->ne[1];
            const bool peer_f16    = (peer_sum && peer_sum->type == GGML_TYPE_F16);
            const int blocks = (H + 63) / 64;
            if (peer_f16) {
                inkling2_moe_masked_sum_sconv_add_kernel<true><<<blocks, 64, 0, stream>>>(
                    (const float *) down->data, (const float *) wsel->data, (const int32_t *) mask_sel->data,
                    peer_sum->data, (const float *) w_static->data, (const float *) state_in->data,
                    state_out ? (float *) state_out->data : nullptr,
                    (const float *) addend->data, addend->nb[1] / sizeof(float),
                    (float *) dst->data, H, K, T, wsel_ld, msel_ld,
                    state_stride, state_chan_offset, write_mode);
            } else {
                inkling2_moe_masked_sum_sconv_add_kernel<false><<<blocks, 64, 0, stream>>>(
                    (const float *) down->data, (const float *) wsel->data, (const int32_t *) mask_sel->data,
                    peer_sum ? peer_sum->data : nullptr, (const float *) w_static->data, (const float *) state_in->data,
                    state_out ? (float *) state_out->data : nullptr,
                    (const float *) addend->data, addend->nb[1] / sizeof(float),
                    (float *) dst->data, H, K, T, wsel_ld, msel_ld,
                    state_stride, state_chan_offset, write_mode);
            }
            break;
        }
        case 29: { // FUSED_DSPARK_MARKOV_RNN_STEP
            const ggml_tensor * prev_tok      = dst->src[0];
            const ggml_tensor * state         = dst->src[1];
            const ggml_tensor * markov_w1     = dst->src[2];
            const ggml_tensor * rnn_proj_step = dst->src[3];
            const ggml_tensor * rnn_ctx_s     = dst->src[4];
            const int rank = (int) state->ne[0];
            dim3 grid((rank + 7) / 8, 1, 1);
            dim3 block(WARP_SIZE, 8, 1);
            inkling2_dspark_markov_rnn_step_kernel<<<grid, block, 0, stream>>>(
                (const int32_t *) prev_tok->data,
                (const float *) state->data,
                (const half *) markov_w1->data,
                (const half *) rnn_proj_step->data,
                (const float *) rnn_ctx_s->data,
                (float *) dst->data, rank);
            break;
        }
        case 30: { // DSPARK_BUILD_MASKS
            const ggml_tensor * draft_pos = dst->src[0];
            const int kv_span    = (int) dst->ne[0];
            const int B          = (int) dst->ne[1];
            const int swa_window = ggml_get_op_params_i32(dst, 1);
            const int blocks = (kv_span + 255) / 256;
            inkling2_dspark_build_masks_kernel<<<blocks, 256, 0, stream>>>(
                (const int32_t *) draft_pos->data, (half *) dst->data, kv_span, B, swa_window);
            break;
        }
        case 31: { // FUSED_CLIP_QR_REL_TABLE (Q-clip + R-clip + rel_table in ONE launch!)
            const ggml_tensor * q_raw    = dst->src[0];
            const ggml_tensor * q_clip_w = dst->src[1];
            const ggml_tensor * r_raw    = dst->src[2];
            const ggml_tensor * r_clip_w = dst->src[3];
            const ggml_tensor * pos_buf  = dst->src[4];
            const ggml_tensor * rel_bias = dst->src[5];
            const ggml_tensor * sink     = dst->src[6];
            const int head_dim   = ggml_get_op_params_i32(dst, 1);
            const int d_rel      = ggml_get_op_params_i32(dst, 2);
            const int n_head     = ggml_get_op_params_i32(dst, 3);
            const float q_thresh = ggml_get_op_params_f32(dst, 4);
            const float r_thresh = ggml_get_op_params_f32(dst, 5);
            const float eps      = ggml_get_op_params_f32(dst, 6);
            const int log_floor  = ggml_get_op_params_i32(dst, 7);
            const float log_a    = ggml_get_op_params_f32(dst, 8);
            const int rel_extent = (int) rel_bias->ne[1];
            const int n_tokens   = (int) q_raw->ne[1];
            float * dst_q  = (float *) dst->data;
            float * dst_rt = dst_q + (size_t) head_dim * n_head * n_tokens;
            dim3 grid(n_head, n_tokens, 1);
            inkling2_clip_qr_rel_table_kernel<<<grid, 256, 0, stream>>>(
                (const float *) q_raw->data, q_raw->nb[1] / sizeof(float),
                q_clip_w ? (const float *) q_clip_w->data : nullptr,
                (const float *) r_raw->data, r_raw->nb[1] / sizeof(float),
                r_clip_w ? (const float *) r_clip_w->data : nullptr,
                pos_buf ? (const int32_t *) pos_buf->data : nullptr,
                (const float *) rel_bias->data,
                (const float *) sink->data, sink->nb[1] / sizeof(float),
                dst_q, dst_rt,
                head_dim, d_rel, rel_extent, n_head,
                q_thresh, r_thresh, eps, log_floor, log_a);
            break;
        }
        case 32: { // EXACT_GPU_TOP_CANDS (for ultra-fast non-greedy sampling!)
            const ggml_tensor * logits = dst->src[0];
            const float temp = ggml_get_op_params_f32(dst, 1);
            const int n_vocab = (int) logits->ne[0];
            const int T       = (int) logits->ne[1];
            ggml_cuda_pool_alloc<float> blk_max(ctx.pool(), (size_t) 128 * T);
            dim3 grid(128, T, 1);
            inkling2_top_cands_stage1_kernel<<<grid, 256, 0, stream>>>(
                (const float *) logits->data, blk_max.get(), (int32_t *) dst->data, n_vocab);
            inkling2_top_cands_stage2_kernel<<<grid, 256, 0, stream>>>(
                (const float *) logits->data, blk_max.get(), (int32_t *) dst->data, n_vocab, temp);
            break;
        }
        case 33: { // TREE_SCONV (+ optional dynamic taps + optional addend)
            const ggml_tensor * x           = dst->src[0];
            const ggml_tensor * w_static    = dst->src[1];
            const ggml_tensor * prefix_conv = dst->src[2];
            ggml_tensor       * tree_conv   = dst->src[3];
            const ggml_tensor * meta        = dst->src[4];
            const ggml_tensor * dyn_taps    = dst->src[5];
            const ggml_tensor * addend      = dst->src[6];
            const int chan_offset  = ggml_get_op_params_i32(dst, 1);
            const int head_dim_dyn = ggml_get_op_params_i32(dst, 2);
            const int m_stride     = ggml_get_op_params_i32(dst, 3);
            const int n_ch     = (int) x->ne[0];
            const int n_tokens = (int) x->ne[1];
            const int s_ch     = (int) tree_conv->ne[0];
            const int s_max    = (int) tree_conv->ne[1];
            dim3 grid((n_ch + 255) / 256, n_tokens, 1);
            inkling2_tree_sconv_kernel<<<grid, 256, 0, stream>>>(
                (const float *) x->data, x->nb[1] / sizeof(float),
                (const float *) w_static->data,
                (const float *) prefix_conv->data,
                (float *) tree_conv->data,
                (const int32_t *) meta->data,
                dyn_taps ? (const float *) dyn_taps->data : nullptr,
                dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                addend ? (const float *) addend->data : nullptr,
                addend ? (addend->nb[1] / sizeof(float)) : 0,
                (float *) dst->data,
                n_ch, n_tokens, s_ch, s_max, chan_offset, head_dim_dyn, m_stride);
            break;
        }
        case 34: { // TREE_SCONV_KV_WRITE
            const ggml_tensor * x           = dst->src[0];
            const ggml_tensor * w_static    = dst->src[1];
            const ggml_tensor * prefix_conv = dst->src[2];
            ggml_tensor       * tree_conv   = dst->src[3];
            const ggml_tensor * meta        = dst->src[4];
            const ggml_tensor * dyn_taps    = dst->src[5];
            ggml_tensor       * tree_kv     = dst->src[6];
            const int chan_offset = ggml_get_op_params_i32(dst, 1);
            const int head_dim    = ggml_get_op_params_i32(dst, 2);
            const int n_head_kv   = ggml_get_op_params_i32(dst, 3);
            const int is_v        = ggml_get_op_params_i32(dst, 4);
            const int m_stride    = ggml_get_op_params_i32(dst, 5);
            const float threshold = ggml_get_op_params_f32(dst, 6);
            const float eps       = ggml_get_op_params_f32(dst, 7);
            const int n_tokens = (int) x->ne[1];
            const int s_ch     = (int) tree_conv->ne[0];
            const int s_max    = (int) tree_conv->ne[1];
            GGML_ASSERT(head_dim == 128);
            dim3 grid(n_head_kv, n_tokens, 1);
            if (tree_kv->type == GGML_TYPE_Q8_0) {
                inkling2_tree_sconv_kv_write_kernel<true><<<grid, 128, 0, stream>>>(
                    (const float *) x->data, x->nb[1] / sizeof(float),
                    (const float *) w_static->data,
                    (const float *) prefix_conv->data,
                    (float *) tree_conv->data,
                    (const int32_t *) meta->data,
                    dyn_taps ? (const float *) dyn_taps->data : nullptr,
                    dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                    (char *) tree_kv->data, tree_kv->nb[1], tree_kv->nb[2], tree_kv->nb[3],
                    n_head_kv, s_ch, s_max, chan_offset, is_v, m_stride, threshold, eps);
            } else {
                GGML_ASSERT(tree_kv->type == GGML_TYPE_F16);
                inkling2_tree_sconv_kv_write_kernel<false><<<grid, 128, 0, stream>>>(
                    (const float *) x->data, x->nb[1] / sizeof(float),
                    (const float *) w_static->data,
                    (const float *) prefix_conv->data,
                    (float *) tree_conv->data,
                    (const int32_t *) meta->data,
                    dyn_taps ? (const float *) dyn_taps->data : nullptr,
                    dyn_taps ? (dyn_taps->nb[1] / sizeof(float)) : 0,
                    (char *) tree_kv->data, tree_kv->nb[1], tree_kv->nb[2], tree_kv->nb[3],
                    n_head_kv, s_ch, s_max, chan_offset, is_v, m_stride, threshold, eps);
            }
            break;
        }
        case 35: { // TREE_ATTN_REL
            const ggml_tensor * q              = dst->src[0];
            const ggml_tensor * prefix_k       = dst->src[1];
            const ggml_tensor * prefix_v       = dst->src[2];
            const ggml_tensor * tree_k         = dst->src[3];
            const ggml_tensor * tree_v         = dst->src[4];
            const ggml_tensor * rel_table      = dst->src[5];
            const ggml_tensor * meta           = dst->src[6];
            const ggml_tensor * prefix_len_buf = dst->src[7];
            const ggml_tensor * gate           = dst->src[8];
            const int swa_window = ggml_get_op_params_i32(dst, 1);
            const int rel_extent = ggml_get_op_params_i32(dst, 2);
            const int m_stride   = ggml_get_op_params_i32(dst, 3);
            const int path_w     = ggml_get_op_params_i32(dst, 4);
            const int Hq         = (int) q->ne[1];
            const int T          = (int) q->ne[2];
            const int Hk         = (int) prefix_k->ne[2];
            const int prefix_cap = (int) prefix_k->ne[1];
            const int gqa        = Hq / Hk;
            GGML_ASSERT(q->ne[0] == 128 && (gqa == 3 || gqa == 1 || gqa == 2 || gqa == 4));
            dim3 grid(Hk, T, 1);
            dim3 block(WARP_SIZE, 8, 1);
            auto launch_tree_attn = [&](auto q8_tag, auto gqa_tag) {
                constexpr bool IS_Q8 = decltype(q8_tag)::value;
                constexpr int  G     = decltype(gqa_tag)::value;
                inkling2_tree_attn_rel_kernel<IS_Q8, G, 8><<<grid, block, 0, stream>>>(
                    (const float *) q->data,
                    (const char *) prefix_k->data, (const char *) prefix_v->data,
                    prefix_k->nb[1], prefix_k->nb[2], prefix_k->nb[3],
                    prefix_v->nb[1], prefix_v->nb[2], prefix_v->nb[3],
                    (const char *) tree_k->data, (const char *) tree_v->data,
                    tree_k->nb[1], tree_k->nb[2], tree_k->nb[3],
                    tree_v->nb[1], tree_v->nb[2], tree_v->nb[3],
                    (const float *) rel_table->data,
                    (const int32_t *) meta->data,
                    (const int32_t *) prefix_len_buf->data,
                    gate ? (const float *) gate->data : nullptr,
                    gate ? (gate->nb[1] / sizeof(float)) : 0,
                    (float *) dst->data,
                    Hq, prefix_cap, swa_window, rel_extent, m_stride, path_w,
                    (int) prefix_len_buf->ne[0]);
            };
            if (prefix_k->type == GGML_TYPE_Q8_0) {
                if      (gqa == 3) launch_tree_attn(std::true_type{}, std::integral_constant<int, 3>{});
                else if (gqa == 4) launch_tree_attn(std::true_type{}, std::integral_constant<int, 4>{});
                else if (gqa == 2) launch_tree_attn(std::true_type{}, std::integral_constant<int, 2>{});
                else               launch_tree_attn(std::true_type{}, std::integral_constant<int, 1>{});
            } else {
                if      (gqa == 3) launch_tree_attn(std::false_type{}, std::integral_constant<int, 3>{});
                else if (gqa == 4) launch_tree_attn(std::false_type{}, std::integral_constant<int, 4>{});
                else if (gqa == 2) launch_tree_attn(std::false_type{}, std::integral_constant<int, 2>{});
                else               launch_tree_attn(std::false_type{}, std::integral_constant<int, 1>{});
            }
            break;
        }
        case 36: { // TREE_COMMIT (commit accepted slots from tree -> prefix KV & conv)
            ggml_tensor       * prefix_k     = dst->src[0];
            ggml_tensor       * prefix_v     = dst->src[1];
            const ggml_tensor * tree_k       = dst->src[2];
            const ggml_tensor * tree_v       = dst->src[3];
            ggml_tensor       * prefix_conv  = dst->src[4];
            const ggml_tensor * tree_conv    = dst->src[5];
            const ggml_tensor * commit_slots = dst->src[6];
            const ggml_tensor * commit_meta  = dst->src[7];
            const int swa_window = ggml_get_op_params_i32(dst, 1);
            const int prefix_cap = (int) prefix_k->ne[1];
            const int Hk         = (int) prefix_k->ne[2];
            const int P          = (int) prefix_k->ne[3];
            const int row_u32    = (int) (prefix_k->nb[1] / sizeof(uint32_t));
            const int s_ch       = (int) tree_conv->ne[0];
            const int s_max      = (int) tree_conv->ne[1];
            GGML_ASSERT((prefix_k->nb[1] % sizeof(uint32_t)) == 0 && prefix_k->nb[1] == tree_k->nb[1]);
            dim3 kv_grid(Hk, P, 1);
            inkling2_tree_commit_kv_kernel<<<kv_grid, 64, 0, stream>>>(
                (char *) prefix_k->data, prefix_k->nb[1], prefix_k->nb[2], prefix_k->nb[3],
                (char *) prefix_v->data, prefix_v->nb[1], prefix_v->nb[2], prefix_v->nb[3],
                (const char *) tree_k->data, tree_k->nb[1], tree_k->nb[2], tree_k->nb[3],
                (const char *) tree_v->data, tree_v->nb[1], tree_v->nb[2], tree_v->nb[3],
                (const int32_t *) commit_slots->data,
                (const int32_t *) commit_meta->data,
                prefix_cap, swa_window, row_u32);
            dim3 conv_grid((s_ch + 255) / 256, P, 1);
            inkling2_tree_commit_conv_kernel<<<conv_grid, 256, 0, stream>>>(
                (float *) prefix_conv->data,
                (const float *) tree_conv->data,
                (const int32_t *) commit_slots->data,
                (const int32_t *) commit_meta->data,
                s_ch, s_max);
            break;
        }
        case 37: { // 2DSD_MARKOV_STEP (batched vanilla DSpark argmax across L leaves)
            const ggml_tensor * base_logits = dst->src[0];
            const ggml_tensor * prev_tok    = dst->src[1];
            const ggml_tensor * markov_w1   = dst->src[2];
            const ggml_tensor * markov_w2   = dst->src[3];
            const int n_vocab = ggml_get_op_params_i32(dst, 1);
            const int L       = (int) prev_tok->ne[0];
            const int rank    = (int) markov_w1->ne[0];
            GGML_ASSERT(rank == 256 && markov_w1->type == GGML_TYPE_F16);
            ggml_cuda_pool_alloc<float>   part_v(ctx.pool(), (size_t) 256 * L);
            ggml_cuda_pool_alloc<int32_t> part_i(ctx.pool(), (size_t) 256 * L);
            dim3 grid1(256, L, 1);
            dim3 block1(WARP_SIZE, 8, 1);
            if (markov_w2->type == GGML_TYPE_Q8_0) {
                inkling2_2dsd_markov_stage1_kernel<true><<<grid1, block1, 0, stream>>>(
                    (const float *) base_logits->data, base_logits->nb[1] / sizeof(float),
                    (const int32_t *) prev_tok->data,
                    (const half *) markov_w1->data, markov_w1->nb[1] / sizeof(half),
                    (const char *) markov_w2->data, markov_w2->nb[1],
                    part_v.get(), part_i.get(), n_vocab, rank);
            } else {
                GGML_ASSERT(markov_w2->type == GGML_TYPE_F16);
                inkling2_2dsd_markov_stage1_kernel<false><<<grid1, block1, 0, stream>>>(
                    (const float *) base_logits->data, base_logits->nb[1] / sizeof(float),
                    (const int32_t *) prev_tok->data,
                    (const half *) markov_w1->data, markov_w1->nb[1] / sizeof(half),
                    (const char *) markov_w2->data, markov_w2->nb[1],
                    part_v.get(), part_i.get(), n_vocab, rank);
            }
            inkling2_2dsd_markov_stage2_kernel<<<L, 256, 0, stream>>>(
                part_v.get(), part_i.get(), (int32_t *) dst->data);
            break;
        }
        case 38: { // 2DSD_CONF_HEAD (vectorized confidence head over M = 7 * L slots)
            const ggml_tensor * h_spec      = dst->src[0];
            const ggml_tensor * before_toks = dst->src[1];
            const ggml_tensor * markov_w1   = dst->src[2];
            const ggml_tensor * conf_w      = dst->src[3];
            const ggml_tensor * conf_b      = dst->src[4];
            const int H       = (int) h_spec->ne[0];
            const int M       = (int) h_spec->ne[1];
            const int R       = (int) markov_w1->ne[0];
            const int n_vocab = (int) markov_w1->ne[1];
            inkling2_2dsd_conf_head_kernel<<<M, 256, 0, stream>>>(
                (const float *) h_spec->data, h_spec->nb[1] / sizeof(float),
                (const int32_t *) before_toks->data,
                (const half *) markov_w1->data, markov_w1->nb[1] / sizeof(half),
                (const float *) conf_w->data,
                (const float *) conf_b->data,
                (float *) dst->data, H, R, n_vocab);
            break;
        }
        case 39: { // 2DSD_DRAFT_MASK (multi-leaf loop-context bidirectional mask)
            const ggml_tensor * draft_meta = dst->src[0];
            const int kv_span = (int) dst->ne[0];
            const int B       = ggml_get_op_params_i32(dst, 1);
            const int L_b     = (int) dst->ne[1] / B;
            dim3 grid((kv_span + 255) / 256, L_b, 1);
            inkling2_2dsd_draft_mask_kernel<<<grid, 256, 0, stream>>>(
                (const int32_t *) draft_meta->data, (half *) dst->data, kv_span, B);
            break;
        }
        case 40: { // 2DSD_MARKOV_TOPK_STEP (32-block/leaf parallel top-1024 vanilla Markov argmax)
            const ggml_tensor * cand_ids    = dst->src[0];
            const ggml_tensor * base_logits = dst->src[1];
            const ggml_tensor * prev_tok    = dst->src[2];
            const ggml_tensor * markov_w1   = dst->src[3];
            const ggml_tensor * markov_w2   = dst->src[4];
            const int n_vocab = ggml_get_op_params_i32(dst, 1);
            const int n_cands = (int) cand_ids->ne[0];
            const int L       = (int) prev_tok->ne[0];
            GGML_ASSERT(markov_w1->ne[0] == 256 && markov_w1->type == GGML_TYPE_F16);
            ggml_cuda_pool_alloc<float>   part_v(ctx.pool(), (size_t) 32 * L);
            ggml_cuda_pool_alloc<int32_t> part_i(ctx.pool(), (size_t) 32 * L);
            dim3 grid1(32, L, 1);
            dim3 block1(WARP_SIZE, 8, 1);
            if (markov_w2->type == GGML_TYPE_Q8_0) {
                inkling2_2dsd_markov_topk_stage1_kernel<true><<<grid1, block1, 0, stream>>>(
                    (const int32_t *) cand_ids->data, cand_ids->nb[1] / sizeof(int32_t),
                    (const float *) base_logits->data, base_logits->nb[1] / sizeof(float),
                    (const int32_t *) prev_tok->data,
                    (const half *) markov_w1->data, markov_w1->nb[1] / sizeof(half),
                    (const char *) markov_w2->data, markov_w2->nb[1],
                    part_v.get(), part_i.get(), n_vocab, n_cands);
            } else {
                GGML_ASSERT(markov_w2->type == GGML_TYPE_F16);
                inkling2_2dsd_markov_topk_stage1_kernel<false><<<grid1, block1, 0, stream>>>(
                    (const int32_t *) cand_ids->data, cand_ids->nb[1] / sizeof(int32_t),
                    (const float *) base_logits->data, base_logits->nb[1] / sizeof(float),
                    (const int32_t *) prev_tok->data,
                    (const half *) markov_w1->data, markov_w1->nb[1] / sizeof(half),
                    (const char *) markov_w2->data, markov_w2->nb[1],
                    part_v.get(), part_i.get(), n_vocab, n_cands);
            }
            inkling2_2dsd_markov_topk_stage2_kernel<<<L, WARP_SIZE, 0, stream>>>(
                part_v.get(), part_i.get(), (int32_t *) dst->data);
            break;
        }
        case 41: { // TREE_SCONV_KV_WRITE_KV (1 launch for both K and V on SWA layers!)
            const ggml_tensor * x_k         = dst->src[0];
            const ggml_tensor * x_v         = dst->src[1];
            const ggml_tensor * wk_static   = dst->src[2];
            const ggml_tensor * wv_static   = dst->src[3];
            const ggml_tensor * prefix_conv = dst->src[4];
            ggml_tensor       * tree_conv   = dst->src[5];
            const ggml_tensor * meta        = dst->src[6];
            ggml_tensor       * tree_k      = dst->src[7];
            ggml_tensor       * tree_v      = dst->src[8];
            const int head_dim    = ggml_get_op_params_i32(dst, 1);
            const int n_head_kv   = ggml_get_op_params_i32(dst, 2);
            const int m_stride    = ggml_get_op_params_i32(dst, 3);
            const float threshold = ggml_get_op_params_f32(dst, 4);
            const float eps       = ggml_get_op_params_f32(dst, 5);
            const int n_tokens = (int) x_k->ne[1];
            const int s_ch     = (int) tree_conv->ne[0];
            const int s_max    = (int) tree_conv->ne[1];
            GGML_ASSERT(head_dim == 128);
            dim3 grid(n_head_kv * 2, n_tokens, 1);
            if (tree_k->type == GGML_TYPE_Q8_0) {
                inkling2_tree_sconv_kv_write_kv_kernel<true><<<grid, 128, 0, stream>>>(
                    (const float *) x_k->data, x_k->nb[1] / sizeof(float),
                    (const float *) x_v->data, x_v->nb[1] / sizeof(float),
                    (const float *) wk_static->data, (const float *) wv_static->data,
                    (const float *) prefix_conv->data, (float *) tree_conv->data,
                    (const int32_t *) meta->data,
                    (char *) tree_k->data, tree_k->nb[1], tree_k->nb[2], tree_k->nb[3],
                    (char *) tree_v->data, tree_v->nb[1], tree_v->nb[2], tree_v->nb[3],
                    n_head_kv, s_ch, s_max, m_stride, threshold, eps);
            } else {
                GGML_ASSERT(tree_k->type == GGML_TYPE_F16);
                inkling2_tree_sconv_kv_write_kv_kernel<false><<<grid, 128, 0, stream>>>(
                    (const float *) x_k->data, x_k->nb[1] / sizeof(float),
                    (const float *) x_v->data, x_v->nb[1] / sizeof(float),
                    (const float *) wk_static->data, (const float *) wv_static->data,
                    (const float *) prefix_conv->data, (float *) tree_conv->data,
                    (const int32_t *) meta->data,
                    (char *) tree_k->data, tree_k->nb[1], tree_k->nb[2], tree_k->nb[3],
                    (char *) tree_v->data, tree_v->nb[1], tree_v->nb[2], tree_v->nb[3],
                    n_head_kv, s_ch, s_max, m_stride, threshold, eps);
            }
            break;
        }
        case 42: { // FUSED_MOE_MASKED_SUM_TREE_SCONV_ADD
            const ggml_tensor * down        = dst->src[0];
            const ggml_tensor * wsel        = dst->src[1];
            const ggml_tensor * mask_sel    = dst->src[2];
            const ggml_tensor * peer_sum    = dst->src[3];
            const ggml_tensor * w_static    = dst->src[4];
            const ggml_tensor * prefix_conv = dst->src[5];
            ggml_tensor       * tree_conv   = dst->src[6];
            const ggml_tensor * meta        = dst->src[7];
            const ggml_tensor * addend      = dst->src[8];
            const int chan_offset = ggml_get_op_params_i32(dst, 1);
            const int m_stride    = ggml_get_op_params_i32(dst, 2);
            const int H = (int) down->ne[0];
            const int K = (int) down->ne[1];
            const int T = (int) down->ne[2];
            const int wsel_ld = (int) (wsel->nb[1] / sizeof(float));
            const int msel_ld = (int) (mask_sel->nb[1] / sizeof(int32_t));
            const int s_ch    = (int) tree_conv->ne[0];
            const int s_max   = (int) tree_conv->ne[1];
            const bool peer_f16 = (peer_sum && peer_sum->type == GGML_TYPE_F16);
            dim3 grid((H + 255) / 256, T, 1);
            if (peer_f16) {
                inkling2_moe_masked_sum_tree_sconv_add_kernel<true><<<grid, 256, 0, stream>>>(
                    (const float *) down->data, (const float *) wsel->data, (const int32_t *) mask_sel->data,
                    peer_sum->data, (const float *) w_static->data,
                    (const float *) prefix_conv->data, (float *) tree_conv->data,
                    (const int32_t *) meta->data,
                    (const float *) addend->data, addend->nb[1] / sizeof(float),
                    (float *) dst->data, H, K, T, wsel_ld, msel_ld, s_ch, s_max, chan_offset, m_stride);
            } else {
                inkling2_moe_masked_sum_tree_sconv_add_kernel<false><<<grid, 256, 0, stream>>>(
                    (const float *) down->data, (const float *) wsel->data, (const int32_t *) mask_sel->data,
                    peer_sum ? peer_sum->data : nullptr, (const float *) w_static->data,
                    (const float *) prefix_conv->data, (float *) tree_conv->data,
                    (const int32_t *) meta->data,
                    (const float *) addend->data, addend->nb[1] / sizeof(float),
                    (float *) dst->data, H, K, T, wsel_ld, msel_ld, s_ch, s_max, chan_offset, m_stride);
            }
            break;
        }
        case 43: { // TREE_PREFILL_CLONE_CODA (clone depth-3 plane P-1 KV & conv -> planes 0..P-2)
            ggml_tensor * prefix_k    = dst->src[0];
            ggml_tensor * prefix_v    = dst->src[1];
            ggml_tensor * prefix_conv = dst->src[2];
            const int pos0       = ggml_get_op_params_i32(dst, 1);
            const int n_tokens   = ggml_get_op_params_i32(dst, 2);
            const int swa_window = ggml_get_op_params_i32(dst, 3);
            const int prefix_cap = (int) prefix_k->ne[1];
            const int Hk         = (int) prefix_k->ne[2];
            const int P          = (int) prefix_k->ne[3];
            const int row_u32    = (int) (prefix_k->nb[1] / sizeof(uint32_t));
            const int s_ch       = (int) prefix_conv->ne[1];
            GGML_ASSERT((prefix_k->nb[1] % sizeof(uint32_t)) == 0);
            if (P > 1 && n_tokens > 0) {
                dim3 kv_grid(Hk, n_tokens, 1);
                inkling2_tree_prefill_clone_kv_kernel<<<kv_grid, 64, 0, stream>>>(
                    (char *) prefix_k->data, prefix_k->nb[1], prefix_k->nb[2], prefix_k->nb[3],
                    (char *) prefix_v->data, prefix_v->nb[1], prefix_v->nb[2], prefix_v->nb[3],
                    pos0, n_tokens, prefix_cap, swa_window, P, row_u32);
                inkling2_tree_prefill_clone_conv_kernel<<<(s_ch + 255) / 256, 256, 0, stream>>>(
                    (float *) prefix_conv->data, s_ch, P);
            }
            break;
        }
        default:
            GGML_ABORT("ggml_cuda_op_inkling2: unknown subop %d", subop);
    }
}
