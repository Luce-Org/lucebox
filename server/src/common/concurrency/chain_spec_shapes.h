#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::common {

// The scheduler's per-lane switch for a speculative burst: the budget is settled, no grammar masks the
// position, and the request did not ask for AR decode (`luce_force_ar_decode`, the C-O2 oracle's T2).
inline bool step_allows_speculation(bool budget_settled, bool constrained, bool force_ar) {
    return budget_settled && !constrained && !force_ar;
}

// A fixed DFlash2 chain can run only when the engine was handed an enabled chain
// config whose width is the drafter's block (2..16), each slot's scratch stride
// holds a whole chain, and the target feature ring exists.
inline bool chain_config_ready(bool enabled, int width, int draft_block, int scratch_stride, bool feat_ring_ok) {
    return enabled && width > 1 && width <= 16 && width == draft_block &&
           scratch_stride >= width && feat_ring_ok;
}

inline int chain_decode_bucket_width(int lanes) {
    static constexpr int buckets[] = {
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
    };
    if (lanes <= 0) return 0;
    for (int bucket : buckets) {
        if (bucket >= lanes) return bucket;
    }
    return 64;
}

inline int chain_draft_bucket_width(int lanes) {
    // Exact C=5 is stable for the draft graph. Target verification keeps the
    // shared decode bucket because changing its MMQ shape changes numerics.
    return lanes == 5 ? 5 : chain_decode_bucket_width(lanes);
}

// draft_tokens[0] is the already-pending root. The target posterior at row N
// verifies draft_tokens[N + 1], so accepted tokens always form a prefix.
inline size_t chain_verified_prefix(
        const std::vector<int32_t> & draft_tokens,
        const int32_t * posterior,
        size_t posterior_count) {
    if (draft_tokens.empty()) return 0;
    size_t accepted = 1;
    while (accepted < draft_tokens.size() &&
           accepted - 1 < posterior_count &&
           posterior[accepted - 1] == draft_tokens[accepted]) {
        ++accepted;
    }
    return accepted;
}

// The pending root at path[0] was sampled by the preceding target step, so
// the ordinary sampler has already applied the min-token EOS floor to it.
// Accepted children would bypass that sampler. Stop before an EOS that is
// still below the floor so replay samples a replacement from the kept
// tip's exact logits. Once the floor is met, keep the EOS itself but discard
// deeper accepted tokens that the scheduler would hide after retirement.
template <typename IsEos>
inline size_t chain_min_tokens_safe_prefix(
        const int32_t * path,
        size_t path_size,
        int generated_tokens_before_root,
        int min_tokens,
        IsEos is_eos) {
    const int generated = std::max(0, generated_tokens_before_root);
    for (size_t child = 1; child < path_size; ++child) {
        if (!is_eos(path[child])) continue;
        return generated + static_cast<int>(child) < min_tokens
            ? child : child + 1;
    }
    return path_size;
}

}  // namespace luce::common
