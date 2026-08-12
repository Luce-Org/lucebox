// Paged K/V block geometry.
//
// Compatibility rules for --paged-attention live in feature_gate.cpp and the
// capability table in model_capabilities.h; only the sizing arithmetic that
// the allocator, the backend, and the gate all need is here. The one rule
// that cannot move is the 256-wide K/V head check in Qwen35Backend::init():
// it needs loaded tensor dimensions, which GgufModelInfo does not carry.

#pragma once

namespace dflash::common {

constexpr int PAGED_BLOCK_SIZE = 16;

constexpr int paged_block_count(int max_ctx) {
    return (max_ctx + PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;
}

constexpr int paged_token_capacity(int max_ctx) {
    return paged_block_count(max_ctx) * PAGED_BLOCK_SIZE;
}

}  // namespace dflash::common
