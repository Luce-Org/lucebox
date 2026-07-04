#include "turbo-wht.cuh"
#include "tq3-quant.cuh"
#include "cpy-utils.cuh"

// Each thread independently transforms one 128-element group.
// Supports non-contiguous src via separate src/dst strides (dim0 must be
// contiguous in both). This lets us skip ggml_cont before turbo_wht when
// the input comes from ggml_permute with dim0 unchanged.
static __global__ void k_turbo_wht(
        const char * __restrict__ src_base,
        char       * __restrict__ dst_base,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t src_nb1,
        const int64_t src_nb2,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int64_t total_groups,
        const int64_t groups_per_row,
        int           direction) {
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_groups) return;

    const int64_t g   = gid % groups_per_row;
    const int64_t rem = gid / groups_per_row;
    const int64_t i01 = rem % ne01;
    const int64_t i02 = rem / ne01;

    const float * row = (const float *)(src_base + i01 * src_nb1 + i02 * src_nb2) + g * QK_TQ3_0_GROUP;
    float * out_row   = (float *)(dst_base + i01 * dst_nb1 + i02 * dst_nb2) + g * QK_TQ3_0_GROUP;

    float x[128];
    for (int i = 0; i < 128; i++) x[i] = row[i];

    if (direction == 0) {
        tq3_rotate_forward(x);
    } else {
        tq3_rotate_inverse(x);
    }

    for (int i = 0; i < 128; i++) out_row[i] = x[i];
}

// Fused kernel: FWHT-rotate a non-contiguous F32 source and quantize directly
// to Q4_0 (or Q8_0). Eliminates the intermediate F32 buffer and two kernel
// launches (cont + turbo_wht) that precede the cpy(F32→Qx) cache write.
// Each thread processes one 128-element FWHT group = 4 Q4_0 blocks (or 4 Q8_0 blocks).
template <typename block_t, int QK, void (*quantize_block)(const float *, block_t *)>
static __global__ void k_turbo_wht_quantize(
        const char * __restrict__ src_base,
        char       * __restrict__ dst_base,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t src_nb1,
        const int64_t src_nb2,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int64_t total_groups,
        const int64_t groups_per_row) {
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_groups) return;

    const int64_t g   = gid % groups_per_row;
    const int64_t rem = gid / groups_per_row;
    const int64_t i01 = rem % ne01;
    const int64_t i02 = rem / ne01;

    const float * row = (const float *)(src_base + i01 * src_nb1 + i02 * src_nb2) + g * QK_TQ3_0_GROUP;

    float x[128];
    for (int i = 0; i < 128; i++) x[i] = row[i];
    tq3_rotate_forward(x);

    // Quantize 128 elements = QK_TQ3_0_GROUP/QK blocks
    constexpr int BLOCKS_PER_GROUP = QK_TQ3_0_GROUP / QK;
    const int64_t dst_block_offset = (i01 * dst_nb1 + i02 * dst_nb2);
    block_t * dst_row = (block_t *)(dst_base + dst_block_offset);
    for (int b = 0; b < BLOCKS_PER_GROUP; b++) {
        quantize_block(x + b * QK, dst_row + g * BLOCKS_PER_GROUP + b);
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    // dim0 must be contiguous (element stride = sizeof(float))
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    GGML_ASSERT(ne00 % QK_TQ3_0_GROUP == 0);

    const int64_t groups_per_row = ne00 / QK_TQ3_0_GROUP;
    const int64_t total_groups   = groups_per_row * ne01 * ne02;

    constexpr int THREADS_PER_BLOCK = 128;
    const int n_blocks = (int)((total_groups + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Destination strides are always contiguous
    const int64_t dst_nb1 = ne00 * sizeof(float);
    const int64_t dst_nb2 = ne00 * ne01 * sizeof(float);

    k_turbo_wht<<<n_blocks, THREADS_PER_BLOCK, 0, ctx.stream()>>>(
        (const char *)src0->data, (char *)dst->data,
        ne00, ne01, ne02,
        src0->nb[1], src0->nb[2],
        dst_nb1, dst_nb2,
        total_groups, groups_per_row,
        direction);
}

void ggml_cuda_op_turbo_wht_quantize(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor * src, ggml_tensor * dst,
                                      ggml_type quant_type) {
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(src->nb[0] == sizeof(float));

    const int64_t ne00 = src->ne[0];
    const int64_t ne01 = src->ne[1];
    const int64_t ne02 = src->ne[2];
    GGML_ASSERT(ne00 % QK_TQ3_0_GROUP == 0);

    const int64_t groups_per_row = ne00 / QK_TQ3_0_GROUP;
    const int64_t total_groups   = groups_per_row * ne01 * ne02;

    constexpr int THREADS_PER_BLOCK = 128;
    const int n_blocks = (int)((total_groups + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    switch (quant_type) {
        case GGML_TYPE_Q4_0:
            k_turbo_wht_quantize<block_q4_0, QK4_0, quantize_f32_q4_0_block>
                <<<n_blocks, THREADS_PER_BLOCK, 0, ctx.stream()>>>(
                    (const char *)src->data, (char *)dst->data,
                    ne00, ne01, ne02,
                    src->nb[1], src->nb[2],
                    dst->nb[1], dst->nb[2],
                    total_groups, groups_per_row);
            break;
        case GGML_TYPE_Q8_0:
            k_turbo_wht_quantize<block_q8_0, QK8_0, quantize_f32_q8_0_block>
                <<<n_blocks, THREADS_PER_BLOCK, 0, ctx.stream()>>>(
                    (const char *)src->data, (char *)dst->data,
                    ne00, ne01, ne02,
                    src->nb[1], src->nb[2],
                    dst->nb[1], dst->nb[2],
                    total_groups, groups_per_row);
            break;
        default:
            GGML_ASSERT(false && "unsupported quant type for rotated quantize");
    }
}
