#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::common {

// Thinking-budget configuration for one request. When the remaining
// generation window falls to hard_limit_remaining while the model is still
// thinking, close_token_ids replace the next emitted tokens in order so
// reusable KV state sees the replacement.
struct BudgetHook {
    // The sequence the budget injects (the terminator hint, which usually
    // ends with the marker). Empty disables the hook.
    std::vector<int32_t> close_token_ids;
    // How the model itself ends thinking, e.g. `</think>`. A natural close
    // is recognized only by this marker, never by the hint: the two differ
    // whenever a hint carries a lead-in or a trailing transition. Empty
    // disables natural-close detection.
    std::vector<int32_t> marker_token_ids;
    int hard_limit_remaining = 0;
};

// The reply reserve comes out of the request's output cap. When the cap is at
// most the configured reserve, the reply gets half of it and thinking the
// other half; the full reserve would leave no room to think at all.
inline int reply_reserve_within(int reserve, int output_cap) {
    const int half = output_cap / 2;
    return reserve < half ? reserve : half;
}

// One controller per generation, shared by the seed, speculative blocks and
// every AR tail. The decoder supplies ACCEPTED output candidates exactly once,
// before committing them to history/KV or emitting them. Rejected drafts and
// prompt/replayed tokens must never advance the controller.
//
// Model-specific behavior is data in BudgetHook, not a branch in this class.
// The hook must outlive the controller. A default controller disables the
// policy; construct with a hook and the effective output cap before decoding.
class ThinkingBudget {
public:
    struct Decision {
        int32_t token;
        // End a speculative block here, discard its suffix and repair KV.
        // Starting a forced close cuts even when the candidate already equals
        // close[0]. Later matching forced drafts can be retained.
        bool cut;
    };

    ThinkingBudget() = default;
    ThinkingBudget(const BudgetHook & hook, int n_gen)
        : ThinkingBudget(&hook, n_gen) {}
    ThinkingBudget(const BudgetHook * hook, int n_gen)
        : hook_(hook), remaining_(n_gen) {
        if (hook_ && !hook_->close_token_ids.empty() &&
            hook_->marker_token_ids.size() > 1) {
            marker_tail_.reserve(hook_->marker_token_ids.size() - 1);
        }
    }
    ThinkingBudget(BudgetHook &&, int) = delete;

    int remaining() const { return remaining_; }
    bool naturally_closed() const { return closed_; }
    bool forced_close() const { return forced_; }
    bool injecting() const {
        return forced_ && inject_pos_ < hook_->close_token_ids.size();
    }
    bool settled() const {
        return !hook_ || hook_->close_token_ids.empty() || closed_ ||
               (forced_ && !injecting());
    }

    // Decoders that cannot replace a token inside a verified block hand off
    // to AR before such a block could reach the boundary. The same controller
    // follows the handoff; a natural close disables this restriction.
    bool needs_ar_tail(int max_block_tokens) const {
        return !settled() &&
               remaining_ - hook_->hard_limit_remaining <= max_block_tokens;
    }

    std::size_t pending_close_tokens() const {
        return injecting() ? hook_->close_token_ids.size() - inject_pos_ : 0;
    }
    // Used by speculative decoders that verify the pending close as forced
    // drafts. offset must be less than pending_close_tokens().
    int32_t pending_close_token(std::size_t offset) const {
        return hook_->close_token_ids[inject_pos_ + offset];
    }

    // Includes this candidate in output accounting, even after the hook has
    // settled. The returned token, not the original candidate, is the token
    // the decoder must emit and use for its next forward pass.
    Decision apply(int32_t candidate) {
        const int remaining = remaining_--;
        if (settled()) return {candidate, false};
        if (forced_) {
            const int32_t token = hook_->close_token_ids[inject_pos_++];
            return {token, token != candidate};
        }
        if (completes_marker(candidate)) {
            closed_ = true;
            marker_tail_.clear();
            return {candidate, false};
        }
        if (remaining > hook_->hard_limit_remaining) {
            remember(candidate);
            return {candidate, false};
        }
        forced_ = true;
        inject_pos_ = 1;
        marker_tail_.clear();
        return {hook_->close_token_ids.front(), true};
    }

    // Walk an accepted block in emit order. A cut replaces its last retained
    // token and excludes the suffix from both output and budget accounting.
    // The caller restores/replays KV to match the retained prefix.
    int apply_block(int32_t * tokens, int n) {
        for (int i = 0; i < n; ++i) {
            if (settled()) {
                remaining_ -= n - i;
                return n;
            }
            const Decision decision = apply(tokens[i]);
            tokens[i] = decision.token;
            if (decision.cut) return i + 1;
        }
        return n;
    }

private:
    bool completes_marker(int32_t token) const {
        const auto & marker = hook_->marker_token_ids;
        if (marker.empty() || token != marker.back()) return false;
        const std::size_t need = marker.size() - 1;
        if (marker_tail_.size() < need) return false;
        for (std::size_t i = 0; i < need; ++i) {
            if (marker_tail_[i] != marker[i]) return false;
        }
        return true;
    }

    void remember(int32_t token) {
        const auto size = hook_->marker_token_ids.size();
        if (size < 2) return;
        // Discard before appending so the once-reserved capacity never grows.
        if (marker_tail_.size() == size - 1) {
            marker_tail_.erase(marker_tail_.begin());
        }
        marker_tail_.push_back(token);
    }

    const BudgetHook * hook_ = nullptr;
    int remaining_ = 0;
    bool closed_ = false;
    bool forced_ = false;
    std::size_t inject_pos_ = 0;
    std::vector<int32_t> marker_tail_;
};

// Degenerate-decode watchdog for the reply after a forced close. Returns the
// period of a verbatim loop at the end of `tokens`, or 0. The aime2025-02 case
// at think_max=4k produces a ~50-token phrase that repeats verbatim until
// max_tokens — pure waste.
//
// A loop counts only once the last kLoopTokens tokens all repeat with period
// P. One repeat is not a loop: replies repeat a line on purpose (a test that
// reads the same range twice to check a cache, identical table rows or list
// items), and cutting there truncated a tool call mid-argument. A real loop
// runs until max_tokens, so waiting kLoopTokens costs a few hundred tokens of
// the reply reserve, while legitimate repeats span far fewer tokens.
//
// Sweep contiguous periods 12..80: fixed periods missed the aime02 loop
// (period ~50), and shorter cycles show up as one of their multiples.
inline int post_close_loop_period(const std::vector<int32_t> & tokens) {
    constexpr int kMinPeriod = 12, kMaxPeriod = 80;
    constexpr int kLoopTokens = 256;
    static_assert(kLoopTokens >= 3 * kMaxPeriod,
                  "a loop must span at least three periods");
    if (tokens.size() < (std::size_t) kLoopTokens) return 0;
    const auto tail = tokens.end() - kLoopTokens;
    for (int P = kMinPeriod; P <= kMaxPeriod; P++) {
        // Every tail token equals the token P before it.
        if (std::equal(tail + P, tokens.end(), tail)) return P;
    }
    return 0;
}

}  // namespace luce::common
