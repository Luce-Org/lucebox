#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
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
    // Thinking that goes round in circles (greedy decoding's failure: the
    // same draft rewritten again and again, or an exact loop): once one span
    // of loop_span tokens has appeared loop_repeats times, it ends the way
    // the hard limit does, when the decoder arms the close (arm_loop_close).
    // 0 repeats disables.
    int loop_span = 64;
    int loop_repeats = 0;
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
        if (hook_ && !hook_->close_token_ids.empty() &&
            hook_->loop_repeats > 1 && hook_->loop_span > 0) {
            loop_ring_.assign((size_t)hook_->loop_span, 0);
            loop_drop_ = 1;
            for (int i = 1; i < hook_->loop_span; ++i) loop_drop_ *= kLoopBase;
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
               (looping_ ||
                remaining_ - hook_->hard_limit_remaining <= max_block_tokens);
    }

    // Thinking repeats itself (see BudgetHook::loop_repeats) and has not
    // closed yet.
    bool looping() const { return looping_ && !settled(); }
    // The next accepted token starts the close. A decoder that cannot
    // replace a token inside a verified block arms it before a step that
    // emits a single token.
    void arm_loop_close() {
        if (looping()) loop_armed_ = true;
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
        if (remaining > hook_->hard_limit_remaining && !loop_armed_) {
            remember(candidate);
            if (!looping_ && !loop_ring_.empty()) looping_ = repeats(candidate);
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

    // Counts each loop_span-token window of thinking by a rolling hash.
    bool repeats(int32_t token) {
        const size_t span = loop_ring_.size();
        const size_t at = (size_t)(loop_seen_ % span);
        if (loop_seen_ >= (long long)span) loop_hash_ -= (uint64_t)loop_ring_[at] * loop_drop_;
        loop_hash_ = loop_hash_ * kLoopBase + (uint64_t)(uint32_t)token;
        loop_ring_[at] = token;
        if (++loop_seen_ < (long long)span) return false;
        return ++loop_windows_[loop_hash_] >= hook_->loop_repeats;
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
    static constexpr uint64_t kLoopBase = 0x9E3779B97F4A7C15ull;
    std::vector<int32_t> loop_ring_;  // the last loop_span thinking tokens
    std::unordered_map<uint64_t, int> loop_windows_;
    uint64_t loop_hash_ = 0;
    uint64_t loop_drop_ = 0;  // kLoopBase^(loop_span - 1)
    long long loop_seen_ = 0;
    bool looping_ = false;
    bool loop_armed_ = false;
};

}  // namespace luce::common
