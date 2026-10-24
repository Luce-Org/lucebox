#include "gdn-conv.cuh"
#include "qwen4exp-common.cuh"
#include "unary.cuh"
#include <cstdlib>
#include <cstring>

template <int TT>
static __global__ void __launch_bounds__(256) gdn_conv_direct_kernel(const float * __restrict__ saved_state, const float * __restrict__ x,
        const float * __restrict__ w, float * __restrict__ y, const int C, const int T, const int state_stride, const int c0) {
    const int c  = c0 + blockIdx.x * 256 + threadIdx.x;
    const int t0 = blockIdx.y * TT;
    if (c >= C) return;
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wr[j] = w[c * 4 + j];
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) { const int jp = t0 + k; xw[k] = (jp < 3) ? saved_state[(size_t) c * state_stride + jp] : x[(size_t) (jp - 3) * C + c]; }
    const int tend = min(T, t0 + TT);
    for (int t = t0; t < tend; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumf += xw[j] * wr[j];
        y[(size_t) t * C + c] = ggml_cuda_op_silu_single(sumf);
        xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
        const int jp = t + 4; xw[3] = (jp < 3) ? saved_state[(size_t) c * state_stride + jp] : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
    }
}

// SiLU(conv) with the GDN q/k L2 norm folded in: channels [c0, c0 + 256*gridDim.x) are two 128-channel heads per
// block, normalised per token as rms_norm(eps) * post_scale + bias. Threads 32w..32w+31 of a head are warp w of
// rms_norm_scale_f32's 128-thread block, so the per-warp xor-tree partials and their (p0+p2)+(p1+p3) combination
// (block_reduce<SUM,128>'s second stage over [p0..p3, 0, ...]) match that kernel bit for bit; the normalised value
// is formed by the same __fmul_rn / post_scale * v + bias expressions. Tokens are processed in sub-tiles of ST so one
// barrier publishes ST tokens' partials (double-buffered: the slot rewritten here was last read before the previous
// barrier). The conv arithmetic and the store order are those of gdn_conv_direct_kernel.
template <int TT, int ST>
static __global__ void __launch_bounds__(256) gdn_conv_direct_norm_kernel(const float * __restrict__ saved_state,
        const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y, const int C, const int T,
        const int state_stride, const float eps, const float post_scale, const float bias) {
    constexpr int ncols = 128;
    static_assert(TT % ST == 0, "sub-tiles must divide the token tile");
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int t0 = blockIdx.y * TT;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, hb = warp & 4;   // hb: first warp of this head
    __shared__ float s_part[2][ST][8];
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wr[j] = w[c * 4 + j];
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) { const int jp = t0 + k; xw[k] = (jp < 3) ? saved_state[(size_t) c * state_stride + jp] : x[(size_t) (jp - 3) * C + c]; }
    const float b = 0.0f;
    const int tend = min(T, t0 + TT);
    for (int ts = t0; ts < tend; ts += ST) {
        const int buf = ((ts - t0) / ST) & 1;
        const int n = min(ST, tend - ts);
        float yb[ST];
#pragma unroll
        for (int i = 0; i < ST; ++i) {
            if (i < n) {
                float sumf = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) sumf += xw[j] * wr[j];
                sumf += b;
                yb[i] = ggml_cuda_op_silu_single(sumf);
                float tmp;
                {
                    // fast-honor-pragmas would contract the square into the reduction's first add; the reference
                    // rms_norm_scale_f32 kernel keeps it a plain multiply.
#pragma clang fp contract(off)
                    tmp = yb[i] * yb[i];
                }
                tmp = warp_reduce_sum(tmp);
                if (lane == 0) s_part[buf][i][warp] = tmp;
                xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
                const int jp = ts + i + 4; xw[3] = (jp < 3) ? saved_state[(size_t) c * state_stride + jp] : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < ST; ++i) {
            if (i < n) {
                const float p0 = s_part[buf][i][hb], p1 = s_part[buf][i][hb + 1], p2 = s_part[buf][i][hb + 2], p3 = s_part[buf][i][hb + 3];
                const float sum = (p0 + p2) + (p1 + p3);
                float scale;
                {
#pragma clang fp contract(off)   // sum / 128 becomes a multiply; keep it out of an fma with eps
                    scale = rsqrtf(sum / ncols + eps);
                }
                const float normalized = __fmul_rn(scale, yb[i]);
                y[(size_t) (ts + i) * C + c] = post_scale * normalized + bias;
            }
        }
    }
}

static bool gdn_conv_qknorm_enabled() {
    static const bool on = [] {
        const char * v = getenv("LUCE_GDN_NORM_LEGACY");
        return !(v && strcmp(v, "0") != 0);
    }();
    return on;
}

// Plain conv over all C channels, or the split launch: norm kernel over [0, qk_channels), plain conv over the rest.
static void gdn_conv_direct_launch(const float * saved_state, const float * x, const float * w, float * y, int C, int T,
        int state_stride, int qk_channels, float eps, float post_scale, float bias, cudaStream_t stream) {
    constexpr int TT = 128;
    const unsigned ty = (unsigned) ((T + TT - 1) / TT);
    if (qk_channels > 0) {
        gdn_conv_direct_norm_kernel<TT, 16><<<dim3((unsigned) (qk_channels / 256), ty), 256, 0, stream>>>(
            saved_state, x, w, y, C, T, state_stride, eps, post_scale, bias);
    }
    if (C > qk_channels) {
        gdn_conv_direct_kernel<TT><<<dim3((unsigned) ((C - qk_channels) / 256), ty), 256, 0, stream>>>(
            saved_state, x, w, y, C, T, state_stride, qk_channels);
    }
}

extern "C" { GGML_BACKEND_API int ggml_cuda_test_gdn_conv_direct(int norm, const float * saved_state, const float * x,
        const float * w, float * y, int C, int T, int state_stride, int qk_channels, float eps, float post_scale,
        float bias, void * stream) {
    if (!saved_state || !x || !w || !y || C <= 0 || C % 256 != 0 || T <= 0 || state_stride < 3 ||
        qk_channels < 0 || qk_channels > C || qk_channels % 256 != 0) return 0;
    gdn_conv_direct_launch(saved_state, x, w, y, C, T, state_stride, norm ? qk_channels : 0, eps, post_scale, bias,
        (cudaStream_t) stream);
    return cudaGetLastError() == cudaSuccess;
} }

static bool gdn_conv_check(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) {
    if (i < 0 || i + 2 >= cgraph->n_nodes) return false;
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) return false;
    const ggml_tensor * st = cc->src[0]; const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32) return false;
    if (tr->op != GGML_OP_TRANSPOSE || !tr->view_src) return false;
    const ggml_tensor * x = tr->view_src;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) return false;
    const int64_t C = x->ne[0], T = x->ne[1];
    if (st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 || !ggml_is_contiguous(st)) return false;
    if (tr->ne[0] != T || tr->ne[1] != C || tr->view_offs != 0) return false;
    if (cc->ne[0] != T + 3 || cc->ne[1] != C || cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc)) return false;
    if (T < 256 || C % 256 != 0) return false;
    if (i + 1 >= cgraph->n_nodes || cgraph->nodes[i + 1]->op != GGML_OP_VIEW || cgraph->nodes[i + 1]->view_src != cc) return false;
    int64_t tail_from = T + 3; int conv = -1;
    for (int n = i + 1; n < cgraph->n_nodes && n < i + 64; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_SSM_CONV && t->src[0] == cc) { conv = n; break; }
        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == cc || t->src[s]->view_src == cc) reads = true;
        if (!reads) continue;
        if (t->op == GGML_OP_VIEW) {
            if (t->ne[0] != 3 || t->ne[1] != C || t->nb[1] != cc->nb[1] || t->view_offs % sizeof(float)) return false;
            const int64_t col = (int64_t) (t->view_offs / sizeof(float)); if (col < 0 || col + 3 > T + 3) return false;
            tail_from = std::min(tail_from, col); continue;
        }
        if (t->op == GGML_OP_CONT && t->src[0]->op == GGML_OP_VIEW && t->src[0]->view_src == cc) continue;
        return false;
    }
    if (conv < 0) return false;
    const ggml_tensor * cv = cgraph->nodes[conv];
    const ggml_tensor * w = cv->src[1];
    if (cv->type != GGML_TYPE_F32 || !w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || !ggml_is_contiguous(w)) return false;
    if (cv->ne[0] != C || cv->ne[1] != T || cv->ne[2] != 1 || !ggml_is_contiguous(cv)) return false;
    if (conv + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * su = cgraph->nodes[conv + 1];
    if (su->op != GGML_OP_UNARY || ggml_get_unary_op(su) != GGML_UNARY_OP_SILU || su->src[0] != cv || su->type != GGML_TYPE_F32 || !ggml_is_contiguous(su)) return false;
    for (int n = conv + 2; n < cgraph->n_nodes; ++n) { const ggml_tensor * t = cgraph->nodes[n];
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == cv || t->src[s]->view_src == cv) return false; }
    m.concat_idx = i; m.conv_idx = conv; m.input = tr; m.state = st; m.concat = cc; m.w = w; m.conv_out = cgraph->nodes[conv + 1];
    m.C = C; m.T = T; m.tail_from = tail_from;
    // Folded q/k norm request (qwen4exp graph): op_params[1] = channels, [2] eps, [3] scale, [4] bias.
    m.qk_channels = ggml_get_op_params_i32(cv, 1);
    if (m.qk_channels != 0) {
        if (!gdn_conv_qknorm_enabled() || m.qk_channels < 0 || m.qk_channels > C || m.qk_channels % 256 != 0) return false;
        m.norm_eps = ggml_get_op_params_f32(cv, 2); m.norm_scale = ggml_get_op_params_f32(cv, 3); m.norm_bias = ggml_get_op_params_f32(cv, 4);
    }
    return true;
}

bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) { return gdn_conv_check(cgraph, i, m); }

bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m) {
    if (j < 1 || cgraph->nodes[j]->op != GGML_OP_SSM_CONV) return false;
    const ggml_tensor * cc = cgraph->nodes[j]->src[0];
    for (int i = j - 1; i >= 0 && i >= j - 64; --i) if (cgraph->nodes[i] == cc) return gdn_conv_check(cgraph, i, m) && m.conv_idx == j;
    return false;
}

void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    const int ncols = (int) (m.T + 3 - m.tail_from);
    const int64_t n = m.C * (3 + ncols);
    q4x_conv_concat_tail<3><<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.input->data,
        (float *) m.concat->data, (int) m.C, (int) m.T, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    gdn_conv_direct_launch((const float *) m.concat->data, (const float *) m.input->data, (const float *) m.w->data,
        (float *) m.conv_out->data, (int) m.C, (int) m.T, (int) (m.concat->nb[1] / sizeof(float)),
        m.qk_channels, m.norm_eps, m.norm_scale, m.norm_bias, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}
