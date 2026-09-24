#include "deepseek4_paged_cache.h"

#include "deepseek4_page_layout.h"

#ifndef LUCE_DS4_PLAN_ONLY
#include "deepseek4_internal.h"
#endif

#include <cstdio>
#include <limits>
#include <memory>

namespace luce::common {
namespace {
bool add_mul(uint64_t & dst, uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
    const uint64_t v = a * b;
    if (dst > std::numeric_limits<uint64_t>::max() - v) return false;
    dst += v;
    return true;
}
}

DeepSeek4LayerGeometry deepseek4_layer_geometry_from_ratio(
        uint32_t ratio, int64_t head_dim, int64_t raw_rows, int64_t indexer_head_dim,
        bool is_kv_source, bool is_index_source) {
    DeepSeek4LayerGeometry g;
    g.ratio = ratio;
    g.head_dim = head_dim;
    g.raw_rows = raw_rows;
    g.is_kv_source = ratio > 0 && is_kv_source;
    g.is_index_source = ratio > 0 && is_index_source;
    g.has_comp = g.is_kv_source;
    if (g.has_comp) {
        // V4's ratio-4 CSA keeps the previous and current windows at double
        // width; every other pooling compressor keeps one window of `ratio`
        // rows; the ratio-1 latent is emitted per token and needs no state.
        if (ratio == 4) {
            g.comp_width = 2 * head_dim;
            g.comp_state_rows = 2 * (int64_t) ratio;
        } else if (ratio >= 2) {
            g.comp_width = head_dim;
            g.comp_state_rows = (int64_t) ratio;
        }
    }
    g.has_index = g.has_comp && g.is_index_source;
    if (g.has_index) {
        g.index_dim = indexer_head_dim;
        if (ratio == 4) {
            // V4 indexer compressor: width 2 * indexer head dim, same double buffer.
            g.index_state_width = 2 * indexer_head_dim;
            g.index_state_rows = 2 * (int64_t) ratio;
        }
    }
    return g;
}

bool plan_deepseek4_paged_pool_blocks(uint32_t max_ctx, uint32_t slots,
                                      uint64_t requested_tokens,
                                      uint32_t & physical_blocks) {
    physical_blocks = 0;
    if (!max_ctx || !slots) return false;
    const uint64_t blocks = requested_tokens
        ? 1 + (requested_tokens - 1) / DS4_PAGE_TOKENS
        : (1 + (uint64_t(max_ctx) - 1) / DS4_PAGE_TOKENS) * slots;
    if (!blocks) return false;
    if (blocks > UINT32_MAX / DS4_PAGE_TOKENS) return false;
    physical_blocks = static_cast<uint32_t>(blocks);
    return true;
}

bool prepare_deepseek4_gathered_lane_rows(
        const int32_t * slots, const int64_t * positions, uint32_t lanes,
        const int32_t * block_tables, uint32_t block_table_stride,
        uint32_t physical_blocks, uint32_t ratio,
        std::vector<DeepSeek4GatheredLaneRows> & out) {
    if (!slots || !positions || !block_tables || !block_table_stride ||
        !physical_blocks || (ratio != 0 && !ds4_compress_ratio_pages(ratio))) {
        return false;
    }
    std::vector<DeepSeek4GatheredLaneRows> prepared(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        auto & rows = prepared[lane];
        rows.slot = slots[lane];
        if (rows.slot < 0) continue; // Padding must remain entirely passive.
        // Padding lanes do not have a validated position. Keep the default
        // zero so inverse RoPE and compressor bookkeeping cannot inherit an
        // arbitrary positions[] value once callers use constant-width steps.
        rows.position = positions[lane];
        if (rows.position < 0) return false;
        const uint64_t pos = static_cast<uint64_t>(rows.position);
        // The current row is appended in-graph, so retain at most the 127
        // preceding rows that can coexist with it in the 128-row SWA window.
        const uint64_t first_raw = pos >= DS4_PAGE_TOKENS
            ? pos - DS4_PAGE_TOKENS + 1 : 0;
        rows.raw_history.reserve(static_cast<size_t>(pos - first_raw));
        for (uint64_t p = first_raw; p < pos; ++p) {
            rows.raw_history.push_back(
                int64_t(rows.slot) * DS4_PAGE_TOKENS + ds4_raw_ring_row(p));
        }
        rows.raw_history_valid =
            static_cast<uint32_t>(rows.raw_history.size());
        rows.raw_scatter = int64_t(rows.slot) * DS4_PAGE_TOKENS +
                           ds4_raw_ring_row(pos);
        // Validate all active pages even when this layer has no compressor.
        // Bound the position before reserving compressed history.
        const uint64_t current_logical_block = pos / DS4_PAGE_TOKENS;
        if (current_logical_block >= block_table_stride) return false;
        const int32_t * lane_table = block_tables + size_t(lane) * block_table_stride;
        for (uint64_t block = 0; block <= current_logical_block; ++block) {
            if (lane_table[block] < 0 || uint32_t(lane_table[block]) >= physical_blocks) {
                return false;
            }
        }
        if (!ratio) continue;
        const int32_t current_physical = lane_table[current_logical_block];

        // Every completed group before the current token contributes one
        // chronological row.  Looking up each logical page (rather than
        // assuming contiguous physical pages) is the reference behaviour.
        const uint64_t completed = pos / ratio;
        rows.compressed_history.reserve(static_cast<size_t>(completed));
        for (uint64_t group = 0; group < completed; ++group) {
            const uint64_t end_token = group * ratio + ratio - 1;
            const uint64_t logical_block = end_token / DS4_PAGE_TOKENS;
            const int32_t physical = lane_table[logical_block];
            uint64_t row = 0; bool emitted = false;
            if (!ds4_compressed_page_row(end_token, uint32_t(physical), ratio,
                                         row, emitted) || !emitted ||
                row > uint64_t(INT64_MAX)) return false;
            rows.compressed_history.push_back(static_cast<int64_t>(row));
        }
        rows.compressed_history_valid =
            static_cast<uint32_t>(rows.compressed_history.size());
        uint64_t scatter = 0;
        if (!ds4_compressed_page_row(pos, uint32_t(current_physical), ratio, scatter,
                                     rows.compressed_emitted) ||
            scatter > uint64_t(INT64_MAX)) return false;
        if (rows.compressed_emitted) rows.compressed_scatter = int64_t(scatter);
    }
    out = std::move(prepared);
    return true;
}

bool plan_deepseek4_paged_cache(uint32_t head_dim, uint32_t indexer_head_dim,
                                uint32_t slots, uint32_t max_ctx,
                                uint32_t physical_blocks,
                                const std::vector<DeepSeek4LayerGeometry> & layers,
                                DeepSeek4PagedCachePlan & out) {
    DeepSeek4PagedCachePlan p;
    if (!head_dim || !indexer_head_dim || !slots || !max_ctx ||
        !physical_blocks || layers.empty() ||
        physical_blocks > UINT32_MAX / DS4_PAGE_TOKENS) return false;
    p.slots = slots; p.max_ctx = max_ctx; p.physical_blocks = physical_blocks;
    p.max_blocks_per_sequence = 1 + (max_ctx - 1) / DS4_PAGE_TOKENS;
    p.ratios.resize(layers.size());
    p.physical_rows.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        const DeepSeek4LayerGeometry & g = layers[i];
        const uint32_t r = g.ratio;
        p.ratios[i] = r;
        if (r != 0 && !ds4_compress_ratio_pages(r)) return false;
        if (uint64_t(g.head_dim) != head_dim || uint64_t(g.raw_rows) != DS4_PAGE_TOKENS) return false;
        if (!add_mul(p.raw_bytes, uint64_t(head_dim) * DS4_PAGE_TOKENS * 2, slots)) return false;
        if (!g.has_comp) continue;
        uint64_t rows = 0;
        if (!ds4_compressed_page_capacity(physical_blocks, r, rows)) return false;
        p.physical_rows[i] = rows;
        if (!add_mul(p.compressed_bytes, uint64_t(head_dim) * 2, rows)) return false;
        if (g.has_comp_state() &&
            !add_mul(p.state_bytes, uint64_t(g.comp_width) * uint64_t(g.comp_state_rows) * 8, slots)) {
            return false; // KV + score F32
        }
        if (g.has_index) {
            if (uint64_t(g.index_dim) != indexer_head_dim) return false;
            if (!add_mul(p.compressed_bytes, uint64_t(indexer_head_dim) * 2, rows)) return false;
            if (g.has_index_state() &&
                !add_mul(p.state_bytes,
                         uint64_t(g.index_state_width) * uint64_t(g.index_state_rows) * 8, slots)) {
                return false;
            }
        }
    }
    p.total_persistent_bytes = p.raw_bytes;
    if (p.total_persistent_bytes > UINT64_MAX - p.compressed_bytes) return false;
    p.total_persistent_bytes += p.compressed_bytes;
    if (p.total_persistent_bytes > UINT64_MAX - p.state_bytes) return false;
    p.total_persistent_bytes += p.state_bytes;
    out = std::move(p);
    return true;
}

bool plan_deepseek4_paged_cache(uint32_t head_dim, uint32_t indexer_head_dim,
                                uint32_t slots, uint32_t max_ctx,
                                uint32_t physical_blocks,
                                const std::vector<uint32_t> & ratios,
                                DeepSeek4PagedCachePlan & out) {
    std::vector<DeepSeek4LayerGeometry> layers;
    layers.reserve(ratios.size());
    for (uint32_t r : ratios) {
        layers.push_back(deepseek4_layer_geometry_from_ratio(
            r, head_dim, DS4_PAGE_TOKENS, indexer_head_dim,
            /*is_kv_source=*/r > 0, /*is_index_source=*/r == 4));
    }
    return plan_deepseek4_paged_cache(head_dim, indexer_head_dim, slots, max_ctx,
                                      physical_blocks, layers, out);
}

#ifndef LUCE_DS4_PLAN_ONLY
bool create_deepseek4_paged_cache(ggml_backend_t backend,
                                  const DeepSeek4Weights & w, uint32_t slots,
                                  uint32_t max_ctx, uint32_t physical_blocks,
                                  DeepSeek4PagedCache & out) {
    free_deepseek4_paged_cache(out);
    DeepSeek4PagedCachePlan plan;
    const std::vector<DeepSeek4LayerGeometry> geometry = deepseek4_layer_geometries(w);
    if (!backend || w.n_layer <= 0 || geometry.size() != size_t(w.n_layer) ||
        !plan_deepseek4_paged_cache(w.head_dim, w.n_indexer_head_dim, slots,
                                    max_ctx, physical_blocks, geometry, plan)) return false;
    try { out.pool = std::make_unique<PagedKvPool>(physical_blocks, slots, DS4_PAGE_TOKENS); }
    catch (...) { free_deepseek4_paged_cache(out); return false; }
    out.plan = plan;
    out.layers.resize(w.n_layer);
    ggml_init_params ip{ggml_tensor_overhead() * size_t(w.n_layer * 9 + 5) + 4096, nullptr, true};
    out.ctx = ggml_init(ip);
    if (!out.ctx) { free_deepseek4_paged_cache(out); return false; }
    // Readers of a shared compressed cache (V4.1) keep their ratio but own no
    // rows; bind through ds4_comp_cache() to reach the source's tensors.
    for (int il = 0; il < w.n_layer; ++il) {
        auto & l = out.layers[il]; const DeepSeek4LayerGeometry & g = geometry[il];
        l.ratio = g.ratio; l.physical_rows = plan.physical_rows[il];
        l.raw_kv = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16, w.head_dim, DS4_PAGE_TOKENS, slots);
        if (!g.has_comp) continue;
        l.comp_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, w.head_dim, l.physical_rows);
        if (g.has_comp_state()) {
            l.attn_compressor.state_kv = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, g.comp_width, g.comp_state_rows, slots);
            l.attn_compressor.state_score = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, g.comp_width, g.comp_state_rows, slots);
        }
        if (g.has_index) {
            l.index_comp_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, g.index_dim, l.physical_rows);
            if (g.has_index_state()) {
                l.indexer_compressor.state_kv = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, g.index_state_width, g.index_state_rows, slots);
                l.indexer_compressor.state_score = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, g.index_state_width, g.index_state_rows, slots);
            }
        }
    }
    uint64_t raw_bytes = 0, compressed_bytes = 0, state_bytes = 0;
    for (const auto & layer : out.layers) {
        raw_bytes += ggml_nbytes(layer.raw_kv);
        for (const auto * tensor : {layer.comp_kv, layer.index_comp_kv}) {
            if (tensor) compressed_bytes += ggml_nbytes(tensor);
        }
        for (const auto * tensor : {layer.attn_compressor.state_kv,
                                   layer.attn_compressor.state_score,
                                   layer.indexer_compressor.state_kv,
                                   layer.indexer_compressor.state_score}) {
            if (tensor) state_bytes += ggml_nbytes(tensor);
        }
    }
    if (raw_bytes != plan.raw_bytes || compressed_bytes != plan.compressed_bytes ||
        state_bytes != plan.state_bytes) {
        std::fprintf(stderr, "[deepseek4] paged cache tensor sizes disagree with allocation plan\n");
        free_deepseek4_paged_cache(out);
        return false;
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) { free_deepseek4_paged_cache(out); return false; }
    ggml_backend_buffer_clear(out.buf, 0);
    return true;
}

void reset_deepseek4_paged_slot(DeepSeek4PagedCache & c, uint32_t slot) {
    if (!c.buf || slot >= c.plan.slots) return;
    auto clear_slot = [slot](ggml_tensor * tensor) {
        if (!tensor || tensor->ne[2] <= (int64_t) slot) return;
        const size_t bytes = tensor->nb[2];
        std::vector<uint8_t> zeros(bytes, 0);
        ggml_backend_tensor_set(tensor, zeros.data(), (size_t) slot * bytes,
                                bytes);
    };
    for (DeepSeek4PagedLayerCache & layer : c.layers) {
        clear_slot(layer.attn_compressor.state_kv);
        clear_slot(layer.attn_compressor.state_score);
        clear_slot(layer.indexer_compressor.state_kv);
        clear_slot(layer.indexer_compressor.state_score);
    }
}

void free_deepseek4_paged_cache(DeepSeek4PagedCache & c) {
    deepseek4_release_paged_gathered_runtime(c);
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.pool.reset(); c.layers.clear(); c.plan = {};
}
#endif
} // namespace luce::common
