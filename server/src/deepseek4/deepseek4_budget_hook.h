#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce {
namespace deepseek4 {

// Level-2 thinking-budget force-close, as applied to one sampled token.
//
// Header-only and free of backend state so the rule can be tested without a model or a GPU;
// the AR decode loop is the only caller.
//
// Contract: when the remaining window falls to the reserved reply budget, steer the stream
// into `close_ids` one token per step, then STOP INTERVENING so the model spends the rest of
// the window on a visible answer. Overriding the sampled token (rather than appending tokens
// out of band) keeps every emitted token on the normal decode path, so the next step's forward
// pass sees it and KV state stays consistent.
//
// The previous DeepSeek4 implementation pushed the whole close sequence and broke out of the
// loop instead. Two consequences, both measured on DeepSeek-V4-Flash-0731: no forward ever ran
// over the injected tokens, and generation ended immediately, so the reserved reply budget was
// never usable. Completions came to exactly `thinking_ceiling + close_ids.size()` tokens for
// sequences of length 1, 3 and 23 -- zero tokens of answer every time -- while finish_reason
// still reported "stop", which a client cannot distinguish from a real completion. One item
// returned 22,706 characters of reasoning and 1 character of answer. qwen35_backend already
// used the override-and-continue shape this restores.
//
//   close_ids        the close sequence; empty disables the hook entirely
//   remaining        n_gen - generated, i.e. window left INCLUDING this token
//   hard_limit       reserved reply budget that triggers the close
//   sampled          the token the sampler chose
//   started          [in,out] false until the close sequence begins
//   inject_pos       [in,out] index of the next close token to emit
//   forced_close     [out] set true on the step the hook first fires; never cleared here
//
// Returns the token that should actually be emitted.
inline int32_t budget_hook_apply(const std::vector<int32_t> & close_ids,
                                 int remaining,
                                 int hard_limit,
                                 int32_t sampled,
                                 bool & started,
                                 std::size_t & inject_pos,
                                 bool & forced_close) {
    if (close_ids.empty()) {
        return sampled;                       // hook disabled
    }
    if (started) {
        if (inject_pos < close_ids.size()) {
            return close_ids[inject_pos++];   // continue the sequence
        }
        return sampled;                       // sequence done: the model answers freely
    }
    if (remaining > hard_limit) {
        return sampled;                       // still inside the thinking window
    }
    started = true;
    inject_pos = 1;
    forced_close = true;
    // Unconditional override. When the model itself sampled close[0] at the boundary this is
    // a no-op by value -- the emitted token is identical -- so "consume the model's own close
    // token" falls out of override semantics without a comparison. forced_close still reports
    // that the hook fired: it marks that the BUDGET decided the close happened here, which is
    // true whether or not the model concurrently agreed.
    return close_ids.front();
}

// The same rule applied to one DSpark speculative step, so thinking requests
// keep speculative decode instead of routing to AR.
//
// A step emits `accept` tokens: the kept candidates draft[1..accept-1], then
// `bonus`. `first_remaining` is the window left (n_gen - generated) at the
// step's first emitted token. Where the AR rule would override emitted token
// i, the step is truncated there and the close token becomes the bonus (the
// next step's seed). While the rest of the close sequence is pending
// (spec_budget_hook_forcing), the caller drafts it verbatim, verifies it and
// keeps every forced candidate; `n_forced` is how many it verified, and only
// the bonus after them is checked here. Every emitted token then equals what
// budget_hook_apply would emit token by token.
struct SpecBudgetHookState {
    bool started = false;
    std::size_t inject_pos = 0;
    bool forced_close = false;
};

// True while the next step must draft the close sequence instead of asking the
// drafter.
inline bool spec_budget_hook_forcing(const std::vector<int32_t> & close_ids,
                                     const SpecBudgetHookState & st) {
    return !close_ids.empty() && st.started && st.inject_pos < close_ids.size();
}

inline void spec_budget_hook_step(const std::vector<int32_t> & close_ids,
                                  int first_remaining,
                                  int hard_limit,
                                  bool forcing,
                                  int n_forced,
                                  int & accept,
                                  int32_t & bonus,
                                  SpecBudgetHookState & st) {
    if (close_ids.empty()) return;
    if (forcing) st.inject_pos += (std::size_t) n_forced;
    for (int i = forcing ? accept : 1; i <= accept; i++) {
        const int remaining = first_remaining - (i - 1);
        if (st.started) {
            if (st.inject_pos < close_ids.size()) {
                if (i < accept) accept = i;
                bonus = close_ids[st.inject_pos++];
            }
            return;
        }
        if (remaining <= hard_limit) {
            st.started = true;
            st.forced_close = true;
            st.inject_pos = 1;
            if (i < accept) accept = i;
            bonus = close_ids.front();
            return;
        }
    }
}

}  // namespace deepseek4
}  // namespace luce
