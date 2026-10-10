// Per-token log-probabilities reported to API clients (OpenAI `logprobs`).
//
// Backends compute them from the raw logits row a token was chosen from, so
// the values describe the model's distribution before temperature, top-k/p
// and penalties. The HTTP layer turns token ids into text.

#pragma once

#include <cstdint>
#include <vector>

namespace luce::common {

// OpenAI caps top_logprobs at 20.
inline constexpr int kMaxTopLogprobs = 20;

struct TokenLogprob {
    int32_t token = -1;
    float logprob = 0.0f;
};

struct TokenLogprobs {
    TokenLogprob chosen;
    // Most likely tokens, descending; equal logits keep the lower id first.
    std::vector<TokenLogprob> top;
};

// Log-softmax of `logits[0..vocab)`, returning `chosen`'s log-probability
// and the `top_n` most likely tokens (clamped to [0, kMaxTopLogprobs]).
TokenLogprobs compute_token_logprobs(const float * logits, int vocab,
                                     int32_t chosen, int top_n);

}  // namespace luce::common
