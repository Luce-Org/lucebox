#pragma once

#include <cstddef>

namespace dflash::common {

struct Qwen3DrafterBufferPlan {
    std::size_t rope_k_buffers;
    std::size_t value_buffers;
    std::size_t rope_q_tail_buffers;
    bool reuse_current_layer_kv;

    std::size_t layer_cache_index(int layer) const {
        return reuse_current_layer_kv ? 0u : static_cast<std::size_t>(layer);
    }
};

inline Qwen3DrafterBufferPlan qwen3_drafter_buffer_plan(
        bool nope_tail, int n_layer) {
    const std::size_t layers = n_layer > 0 ? (std::size_t)n_layer : 0u;
    return {
        nope_tail ? (layers > 0 ? 1u : 0u) : layers,
        layers > 0 ? 1u : 0u,
        nope_tail ? 0u : layers,
        nope_tail,
    };
}

} // namespace dflash::common
