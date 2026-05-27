// Shared CPU sampler chain used by both target arches.
//
// dflash::common daemon protocol embeds optional sampler params as a tail on each
// generate command: ` samp=temp,top_p,top_k,rep_pen,seed[,freq_pen,pres_pen]`.
// parse_sampler_token strips the tail in place and fills a SamplerCfg;
// sample_logits applies the chain:
//   rep_penalty -> freq/pres_penalty -> top_k -> softmax(temp) -> top_p -> draw.
//
// All backends (qwen35, qwen3, gemma4, laguna) include this header to keep
// sampling behaviour identical across arches.

#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct SamplerCfg {
    float    temp       = 0.0f;
    float    top_p      = 1.0f;
    int      top_k      = 0;
    float    rep_pen    = 1.0f;     // multiplicative repetition penalty (HF-style)
    int      rep_window = 256;
    uint64_t seed       = 0;

    // OpenAI-style additive penalties (applied per-token to logits before softmax).
    // frequency_penalty: subtract freq_pen * count(token_in_history) from logit.
    // presence_penalty:  subtract pres_pen * 1(token_appeared_in_history) from logit.
    // Range: [-2.0, 2.0], default 0.0 (no effect).
    float    freq_pen   = 0.0f;
    float    pres_pen   = 0.0f;

    // True when any logit modifier is active (penalties or stochastic sampling).
    // Backends should use CPU sample_logits() path when this returns true.
    bool needs_logit_processing() const {
        return temp > 0.0f || rep_pen > 1.0f || freq_pen != 0.0f || pres_pen != 0.0f;
    }
};

// Returns the chosen token id. cfg.temp == 0 -> caller should use argmax;
// the chain assumes a positive temperature and falls back to a small floor.
int sample_logits(const float * logits_in,
                  int vocab,
                  const SamplerCfg & cfg,
                  const std::vector<int32_t> & history,
                  std::mt19937_64 & rng);

// Strip ` samp=...` tail from `line` (in place); return true when one was
// parsed. Out-of-band fields default to a permissive greedy-equivalent (top_p=1,
// top_k=0, rep_pen=1, seed=0).
bool parse_sampler_token(std::string & line, SamplerCfg & out);

}  // namespace dflash::common
