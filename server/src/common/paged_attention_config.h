// Paged K/V block geometry.
//
// Compatibility rules for --paged-attention live in feature_gate.cpp and the
// capability table in model_capabilities.h; only the sizing arithmetic that
// the allocator, the backend, and the gate all need is here. The one rule
// that cannot move is the 256-wide K/V head check in Qwen35Backend::init():
// it needs loaded tensor dimensions, which GgufModelInfo does not carry.

#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>

namespace luce::common {

constexpr int PAGED_BLOCK_SIZE = 16;
inline constexpr int DEEPSEEK4_MAX_PAGED_SEQUENCES = 6;
// Monolithic exact serving packs chronological prompt rows into the gathered
// graph, sharing this row budget between decode and prompt sequences.
inline constexpr int DEEPSEEK4_MAX_GATHERED_ROWS = 16;
// Routed-expert part width for packed steps: the RDNA3 MUL_MAT_ID MMVQ ceiling.
inline constexpr int DEEPSEEK4_MAX_MOE_ROWS_PER_PART = 8;

// Expert-owner switches paged DeepSeek V4.1 serving is not qualified with:
// with them it has faulted the GPU (server/docs/DS41.md). Under
// --paged-attention a launch profile's default for one of them is left out;
// one the operator sets is refused by the backend.
inline constexpr const char * DEEPSEEK41_PAGED_UNQUALIFIED_ENV[] = {
    "LUCE_DS4_TP_GROUPED_MMVQ", "LUCE_MOE_TP_GROUPED_MMVQ", "LUCE_MMID_GROUPED",
    "LUCE_DS4_TP_NATIVE_ROUTE_WIDTH", "LUCE_DS4_TP_MASKED_ROUTES",
    "LUCE_DS4_TP_BATCH_SPLIT_COPIES", "GGML_CUDA_BATCH_PEER_COPIES", "GGML_BATCH_PEER_COPIES",
    "LUCE_DS4_TP_DEVICE_JOIN", "LUCE_MOE_TP_DEVICE_JOIN", "LUCE_DS4_TP_DEVICE_JOIN_SPLIT",
    "LUCE_DS4_TP_FUSED_HC_JOIN", "LUCE_DS4_TP_ROUTE_PREFORK", "LUCE_MOE_TP_ROUTE_PREFORK",
    "LUCE_DS4_TP_COARSE_OWNER", "LUCE_MOE_TP_COARSE_OWNER", "LUCE_DS4_TP_MAIN_ROUTE_WEIGHTS",
    "LUCE_CUDA_MMVQ_MOE_ROWS_PER_BLOCK",
};

inline bool deepseek41_paged_unqualified_env(std::string_view name) {
    return std::any_of(std::begin(DEEPSEEK41_PAGED_UNQUALIFIED_ENV), std::end(DEEPSEEK41_PAGED_UNQUALIFIED_ENV),
                       [&](const char * listed) { return name == listed; });
}

constexpr int paged_block_count(int max_ctx) {
    return (max_ctx + PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;
}

constexpr int paged_token_capacity(int max_ctx) {
    return paged_block_count(max_ctx) * PAGED_BLOCK_SIZE;
}

constexpr int64_t paged_kv_address_cap() {
    return ((int64_t)INT32_MAX - PAGED_BLOCK_SIZE) / PAGED_BLOCK_SIZE *
        PAGED_BLOCK_SIZE;
}

// Inputs for sizing a concurrent paged pool from memory that remains after
// model weights have loaded. `fixed_cache_bytes` covers cache allocations that
// do not shrink with the pool (per-slot recurrent state and metadata);
// `reserve_bytes` leaves room for runtime graph buffers.
struct PagedKvAutoBudget {
    int64_t free_bytes        = 0;
    int64_t fixed_cache_bytes = 0;
    int64_t reserve_bytes     = 0;
    int64_t bytes_per_token   = 0;
};

// Return a whole-block physical capacity, capped by the old
// n_slots * max_ctx policy and by the pool's signed-int tensor address space.
// Zero means the supplied memory budget cannot hold even one block.
inline int64_t paged_kv_auto_pool_tokens(int max_ctx, int n_slots,
                                         const PagedKvAutoBudget & budget) {
    if (max_ctx < 1 || n_slots < 1 || budget.free_bytes <= 0 ||
        budget.bytes_per_token <= 0) {
        return 0;
    }
    const int64_t usable = std::max<int64_t>(
        0, budget.free_bytes - budget.fixed_cache_bytes -
               budget.reserve_bytes);
    int64_t tokens = usable / budget.bytes_per_token;
    tokens = (tokens / PAGED_BLOCK_SIZE) * PAGED_BLOCK_SIZE;
    const int64_t logical_cap =
        (int64_t)n_slots * (int64_t)paged_token_capacity(max_ctx);
    const int64_t address_cap = paged_kv_address_cap();
    return std::max<int64_t>(
        0, std::min(tokens, std::min(logical_cap, address_cap)));
}

}  // namespace luce::common
