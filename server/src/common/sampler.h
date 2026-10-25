// Shared CPU sampler chain used by both target arches.
//
// luce::common daemon protocol embeds optional sampler params as a tail on each
// generate command: ` samp=temp,top_p,top_k,rep_pen,seed[,freq_pen,pres_pen]`.
// parse_sampler_token strips the tail in place and fills a SamplerCfg;
// sample_logits applies the chain:
//   rep_penalty -> freq/pres_penalty -> top_k -> softmax(temp) -> top_p -> draw.
//
// All backends (qwen35, qwen3, gemma4, laguna) include this header to keep
// sampling behaviour identical across arches.

#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

class OutputConstraint;

struct SamplerCfg {
    float    temp       = 0.0f;
    float    top_p      = 1.0f;
    int      top_k      = 0;
    float    rep_pen    = 1.0f;     // multiplicative repetition penalty (HF-style)
    int      rep_window = 256;
    uint64_t seed       = 0;
    // The request named a seed. Lets an explicit seed 0 stay reproducible;
    // transports that carry only `seed` (daemon samp= tail, cluster wire)
    // leave it false, so there seed 0 still means "no seed".
    bool     has_seed   = false;

    // OpenAI-style additive penalties (applied per-token to logits before softmax).
    // frequency_penalty: subtract freq_pen * count(token_in_history) from logit.
    // presence_penalty:  subtract pres_pen * 1(token_appeared_in_history) from logit.
    // Range: [-2.0, 2.0], default 0.0 (no effect).
    float    freq_pen   = 0.0f;
    float    pres_pen   = 0.0f;

    // Structured output (common/output_constraint.h): the request's tokens
    // follow a grammar. Shared by every copy of the config, so the sampler
    // and the commit point see one position.
    std::shared_ptr<OutputConstraint> constraint;

    // True when any logit modifier is active (penalties, stochastic sampling
    // or a grammar). Backends should use CPU sample_logits() path when this
    // returns true.
    bool needs_logit_processing() const {
        return temp > 0.0f || rep_pen > 1.0f || freq_pen != 0.0f || pres_pen != 0.0f ||
               constraint != nullptr;
    }

    // True when draws must be reproducible from `seed`. Otherwise (OpenAI
    // semantics) every request samples from a fresh random stream.
    bool seeded() const { return has_seed || seed != 0; }
};

// The seed a request's sampling stream starts from: `seed` when seeded(),
// otherwise 64 fresh bits. Every sampled request is seeded with it, so an
// unseeded request never continues the stream an earlier request left behind
// (a seeded request before it would otherwise make it repeat).
inline uint64_t request_rng_seed(const SamplerCfg & cfg) {
    if (cfg.seeded()) return cfg.seed;
    std::random_device rd;
    return ((uint64_t)rd() << 32) ^ (uint64_t)rd();
}

// Returns the chosen token id. cfg.temp == 0 -> caller should use argmax;
// the chain assumes a positive temperature and falls back to a small floor.
// With cfg.constraint the token is one the grammar allows at its position.
int sample_logits(const float * logits_in,
                  int vocab,
                  const SamplerCfg & cfg,
                  const std::vector<int32_t> & history,
                  std::mt19937_64 & rng);

// The distribution sample_logits draws from, after the whole chain (penalties,
// top_k, temperature, top_p) as (probability, token) pairs summing to 1. With
// temp <= 0 it is the single post-penalty argmax. Speculative sampling uses it
// to accept a drafted token with the target's own probability. A constraint
// masks the distribution at its current position only.
void sampler_distribution(const float * logits_in,
                          int vocab,
                          const SamplerCfg & cfg,
                          const std::vector<int32_t> & history,
                          std::vector<std::pair<float, int>> & out);

// Inverse-CDF draw over (weight, token) pairs; weights need not be normalized.
int sampler_draw(const std::vector<std::pair<float, int>> & cand, double r_uniform);

// Strip ` samp=...` tail from `line` (in place); return true when one was
// parsed. Out-of-band fields default to a permissive greedy-equivalent (top_p=1,
// top_k=0, rep_pen=1, seed=0).
bool parse_sampler_token(std::string & line, SamplerCfg & out);

}  // namespace luce::common
