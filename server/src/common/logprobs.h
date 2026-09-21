// Per-token logprob support (opt-in, QA/fidelity tooling).
//
// Semantics: log-softmax over the raw target logits at the committed
// position — before repetition/frequency/presence penalties and before
// temperature — computed in double precision (max-subtract + log-sum-exp).
// The committed token's value is reported even when it falls outside the
// requested top-K (e.g. a budget-hook substitution or a sampled draw).
//
// This differs from OpenAI's sampled-distribution logprobs; see
// docs/API.md "Logprobs".

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace dflash::common {

struct TokenLogprob {
    int32_t token = -1;
    float   logprob = 0.0f;
    // (token, logprob) sorted by descending logit; ties resolve to the lower
    // token id — the same rule argmax uses. Empty when top_k == 0.
    std::vector<std::pair<int32_t, float>> top;
};

// Compute the logprob of `token` under softmax(logits) plus the top_k
// highest-logit alternatives. top_k <= 0 yields an empty `top` list; a
// top_k above vocab clamps to vocab. Returns a default-constructed record
// (logprob 0) when arguments are out of range.
TokenLogprob compute_token_logprob(const float * logits, int vocab,
                                   int32_t token, int top_k);

} // namespace dflash::common
