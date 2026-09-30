// DeepSeek V4.1 candidate blocks (model.py select_candidate_blocks): the
// coarse block pre-selection the index sources after candidate_source_layer
// restrict their top-k to. See deepseek4_build_indexer_topk.
#include "deepseek4_internal.h"

namespace luce::common {

ggml_tensor * deepseek4_candidate_blocks(
        ggml_context * ctx, ggml_tensor * scores, ggml_tensor * positions,
        int ratio, int topk_blocks, int block_size) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32 && ggml_is_contiguous(scores));
    GGML_ASSERT(positions->type == GGML_TYPE_I32 && positions->ne[0] == scores->ne[1]);
    // Powers of two keep the frontier arithmetic below exact in F32.
    GGML_ASSERT(ratio > 0 && (ratio & (ratio - 1)) == 0);
    GGML_ASSERT(block_size > 0 && (block_size & (block_size - 1)) == 0);
    const int64_t n_tokens = scores->ne[1];
    const int64_t n_full = scores->ne[0] / block_size;
    GGML_ASSERT(n_full + 1 > topk_blocks);
    // A full block scores as its best row; one the query cannot reach yet
    // scores -1e30 like its rows. The partial block at the end (if any) is
    // reachable only by the queries whose newest row it holds, and those pin
    // it below, so its column starts at -1e30.
    ggml_tensor * blocks = ggml_pool_2d(ctx, scores, GGML_OP_POOL_MAX,
                                        block_size, 1, block_size, 1, 0.0f, 0.0f);
    ggml_tensor * pos = ggml_cast(ctx, positions, GGML_TYPE_F32);
    ggml_tensor * partial = ggml_scale_bias(ctx, pos, 0.0f, -1.0e30f);
    blocks = ggml_concat(ctx, blocks, ggml_reshape_2d(ctx, partial, 1, n_tokens), 0);
    // The block of the query's newest row, ((pos + 1) / ratio - 1) / block_size,
    // is pinned in: it is only partly filled and could lose to an older block.
    ggml_tensor * visible = ggml_floor(ctx, ggml_scale_bias(ctx, pos, 1.0f / ratio, 1.0f / ratio));
    ggml_tensor * newest = ggml_floor(ctx, ggml_scale_bias(
        ctx, visible, 1.0f / block_size, -1.0f / block_size));
    ggml_tensor * pin = ggml_scale_bias(ctx, pos, 0.0f, 1.0e30f);
    blocks = ggml_set_rows(ctx, ggml_reshape_3d(ctx, blocks, 1, n_full + 1, n_tokens),
                           ggml_reshape_3d(ctx, pin, 1, 1, n_tokens),
                           ggml_reshape_2d(ctx, ggml_cast(ctx, newest, GGML_TYPE_I32), 1, n_tokens));
    return ggml_top_k(ctx, ggml_reshape_2d(ctx, blocks, n_full + 1, n_tokens), topk_blocks);
}

ggml_tensor * deepseek4_restrict_to_candidate_blocks(
        ggml_context * ctx, ggml_tensor * scores, ggml_tensor * candidates, int block_size) {
    GGML_ASSERT(candidates->type == GGML_TYPE_I32 && ggml_is_contiguous(candidates));
    GGML_ASSERT(candidates->ne[1] == scores->ne[1]);
    const int64_t n_blocks = candidates->ne[0], n_tokens = candidates->ne[1];
    // Rows block * block_size + j of every candidate block (exact in F32);
    // the mask keeps the scores there, rows past n_comp are ignored.
    ggml_tensor * first = ggml_scale(ctx, ggml_cast(ctx, candidates, GGML_TYPE_F32), (float) block_size);
    first = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, first, 1, n_blocks, n_tokens),
                           block_size, n_blocks, n_tokens, 1);
    ggml_tensor * rows = ggml_add(ctx, first, ggml_arange(ctx, 0.0f, (float) block_size, 1.0f));
    rows = ggml_cast(ctx, ggml_reshape_2d(ctx, rows, block_size * n_blocks, n_tokens), GGML_TYPE_I32);
    return ggml_ds4_indexer_mask(ctx, scores, rows, 0);
}

}  // namespace luce::common
