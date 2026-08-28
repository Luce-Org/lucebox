#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace dflash::common {

// Host-only description of the selector tensors. Keeping validation in terms
// of dimensions makes it usable both while GGUF tensor descriptors are being
// loaded and when a concrete target lm_head is attached to the batched path.
struct DFlash2SelectorLayout {
    int rank = 0;
    int top_k = 0;
    int64_t hproj_rank = 0;
    int64_t pred_rank = 0;
    int64_t pred_vocab = 0;
    int64_t succ_rank = 0;
    int64_t succ_vocab = 0;
    std::optional<int64_t> target_output_vocab;
    std::optional<int64_t> target_declared_vocab;
};

inline bool validate_dflash2_selector_layout(
        const DFlash2SelectorLayout & layout, std::string & error) {
    error.clear();
    if (layout.rank <= 0) {
        error = "DFlash 2 selector rank must be positive (got " +
            std::to_string(layout.rank) + ")";
        return false;
    }
    if (layout.top_k <= 0) {
        error = "DFlash 2 selector top_k must be positive (got " +
            std::to_string(layout.top_k) + ")";
        return false;
    }
    if (layout.hproj_rank != layout.rank ||
        layout.pred_rank != layout.rank ||
        layout.succ_rank != layout.rank) {
        error = "DFlash 2 selector rank mismatch: metadata=" +
            std::to_string(layout.rank) + " hproj=" +
            std::to_string(layout.hproj_rank) + " pred_cb=" +
            std::to_string(layout.pred_rank) + " succ_cb=" +
            std::to_string(layout.succ_rank);
        return false;
    }
    if (layout.pred_vocab <= 0 || layout.succ_vocab <= 0) {
        error = "DFlash 2 selector codebook vocab must be positive: pred_cb=" +
            std::to_string(layout.pred_vocab) + " succ_cb=" +
            std::to_string(layout.succ_vocab);
        return false;
    }
    if (layout.pred_vocab != layout.succ_vocab) {
        error = "DFlash 2 selector codebook vocab mismatch: pred_cb=" +
            std::to_string(layout.pred_vocab) + " succ_cb=" +
            std::to_string(layout.succ_vocab);
        return false;
    }
    if (layout.top_k > layout.pred_vocab) {
        error = "DFlash 2 selector top_k=" + std::to_string(layout.top_k) +
            " exceeds codebook vocab=" + std::to_string(layout.pred_vocab);
        return false;
    }
    if (layout.target_output_vocab && *layout.target_output_vocab <= 0) {
        error = "DFlash 2 target output/lm_head vocab must be positive (got " +
            std::to_string(*layout.target_output_vocab) + ")";
        return false;
    }
    if (layout.target_declared_vocab && *layout.target_declared_vocab <= 0) {
        error = "DFlash 2 target.n_vocab must be positive (got " +
            std::to_string(*layout.target_declared_vocab) + ")";
        return false;
    }
    if (layout.target_output_vocab &&
        layout.target_declared_vocab &&
        *layout.target_output_vocab != *layout.target_declared_vocab) {
        error = "DFlash 2 target vocab mismatch: output/lm_head=" +
            std::to_string(*layout.target_output_vocab) + " target.n_vocab=" +
            std::to_string(*layout.target_declared_vocab);
        return false;
    }
    const auto validate_target_bounds = [&] (
            const std::optional<int64_t> & vocab,
            const char * source) {
        if (!vocab) {
            return true;
        }
        if (layout.top_k > *vocab) {
            error = "DFlash 2 selector top_k=" +
                std::to_string(layout.top_k) + " exceeds target vocab (" +
                source + ")=" + std::to_string(*vocab);
            return false;
        }
        if (layout.pred_vocab < *vocab) {
            error = "DFlash 2 selector codebook is too small: codebook=" +
                std::to_string(layout.pred_vocab) + " " + source + "=" +
                std::to_string(*vocab);
            return false;
        }
        return true;
    };
    if (!validate_target_bounds(
            layout.target_output_vocab, "target output/lm_head") ||
        !validate_target_bounds(
            layout.target_declared_vocab, "target.n_vocab")) {
        return false;
    }
    return true;
}

}  // namespace dflash::common
