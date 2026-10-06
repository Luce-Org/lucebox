#pragma once
#include "qsa-decode-wmma.cuh"

#if defined(__gfx1151__) || !defined(__HIP_DEVICE_COMPILE__)
static __global__ __launch_bounds__(256) void qsa_decode_merge(
        const float * partial, float * out, int splits) {
    const int row = blockIdx.x, d = threadIdx.x;
    const float * src = partial + row*splits*258;
    float maximum = -INFINITY;
    for (int s = 0; s < splits; ++s) { maximum = fmaxf(maximum, src[s*258 + 256]); }
    float sum = 0.0f, value = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float count = src[s*258 + 257];
        const float factor = count > 0.0f ? expf(src[s*258 + 256] - maximum) : 0.0f;
        value += src[s*258 + d]*factor;
        sum += count*factor;
    }
    out[row*256 + d] = sum > 0.0f ? value/sum : 0.0f;
}
#endif

bool ggml_cuda_flash_attn_ext_qsa_decode_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ctx.device].cc)) { return false; }
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * ids = dst->src[5];
    if (!q || !k || !v || !ids || dst->src[4] || dst->src[6] || dst->src[7]) { return false; }
    float bias, cap;
    memcpy(&bias, (const char *) dst->op_params + 4, 4);
    memcpy(&cap, (const char *) dst->op_params + 8, 4);
    if (bias != 0.0f || cap != 0.0f || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 || q->ne[1] < 1 || q->ne[1] > 127 ||
        k->ne[1] < 1 || k->ne[1] > 262144 || v->ne[1] != k->ne[1] ||
        k->ne[2] < 1 || q->ne[2] != 12*k->ne[2] || v->ne[2] != k->ne[2] ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        q->nb[0] != 4 || k->nb[0] != 2 || v->nb[0] != 2 || ids->nb[0] != 4 ||
        ids->ne[0] < 1 || ids->ne[0] > 2560 || ids->ne[1] < q->ne[1] || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        k->nb[1] % 16 || k->nb[2] % 16 || uintptr_t(k->data) % 16 || !ggml_is_contiguous(dst)) { return false; }
    return !m || (m->type == GGML_TYPE_F16 && m->nb[0] == 2 && m->ne[0] >= k->ne[1] &&
                  m->ne[1] >= q->ne[1] && m->ne[2] == 1 && m->ne[3] == 1);
}

// The WMMA kernel takes one KV head's 12 query heads per block (qwen4exp's GQA, which the graph requires for QSA).
void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
#else
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * ids = dst->src[5];
    const int ns = ids->ne[0], nq = q->ne[1], nh = q->ne[2];
    const int splits = (ns + 63)/64;
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), size_t(nq)*nh*splits*258);
    qsa_decode_wmma_partial<<<dim3(k->ne[2], nq, splits), 256, 0, ctx.stream()>>>(
        (const char *) q->data, (const char *) k->data, (const char *) v->data,
        m ? (const char *) m->data : nullptr, (const char *) ids->data,
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], m ? m->nb[1] : 0, ids->nb[1],
        k->ne[1], ns, nh, splits, scale, partial.get());
    qsa_decode_merge<<<nq*nh, 256, 0, ctx.stream()>>>(partial.get(), (float *) dst->data, splits);
    CUDA_CHECK(cudaGetLastError());
#endif
}
