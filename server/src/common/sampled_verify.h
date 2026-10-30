#pragma once

#include "common/sampler.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace luce::common {

// Sampled verify (LUCE_SAMPLED_VERIFY=1) of a deterministic chain draft. The
// target draws each position from its own sampler; a drafted token is kept
// while it equals the draw. Every committed token is then a single target
// draw at its position, so the output follows autoregressive sampling
// exactly, and the draft only decides how far one verify goes.

// Whether a lane's sampler lets it join a batched chain round. A greedy lane
// verifies by argmax; a sampled lane only with sampled verify on. A grammar
// masks one position at a time, which a block verify cannot honour (the
// scheduler already keeps such lanes out).
inline bool chain_sampler_spec_capable(const SamplerCfg & cfg,
                                       bool sampled_verify) {
    if (!cfg.needs_logit_processing()) return true;
    return sampled_verify && !cfg.constraint;
}

struct SampledVerifyWalk {
    int     accepted = 1;  // the root plus the drafts that matched a draw
    int32_t bonus = -1;    // the draw that ended the walk; -1 when all matched
};

// rows[i] holds the target's logits after tokens[0..i]; tokens[0] is the root,
// already a target sample. Row i is drawn and tokens[i + 1] kept while it
// equals the draw, so a walk that ends at a bonus drew rows 0..accepted - 1.
// stops(token, position) ends the walk at a draw the caller will rewrite at
// that position (an EOS the min-token floor replaces), matching or not, so
// the rewrite applies once, as in autoregressive decoding. history is the
// penalty history before the root; the walk appends the root and each kept
// draft, so every draw sees the history autoregressive sampling would.
template <typename Stops>
inline SampledVerifyWalk sampled_verify_walk(
        const float * rows, int vocab, int width, const int32_t * tokens,
        const SamplerCfg & cfg, std::vector<int32_t> & history,
        std::mt19937_64 & rng, Stops stops) {
    SampledVerifyWalk walk;
    history.push_back(tokens[0]);
    for (int i = 0; i + 1 < width; ++i) {
        const int32_t token = sample_logits(
            rows + static_cast<size_t>(i) * static_cast<size_t>(vocab), vocab,
            cfg, history, rng);
        if (token != tokens[i + 1] || stops(token, i + 1)) {
            walk.bonus = token;
            break;
        }
        history.push_back(token);
        ++walk.accepted;
    }
    return walk;
}

inline SampledVerifyWalk sampled_verify_walk(
        const float * rows, int vocab, int width, const int32_t * tokens,
        const SamplerCfg & cfg, std::vector<int32_t> & history,
        std::mt19937_64 & rng) {
    return sampled_verify_walk(rows, vocab, width, tokens, cfg, history, rng,
                               [](int32_t, int) { return false; });
}

// The token after the first `kept` committed tokens (root included), which
// the next round feeds as its root. When the walk drew the row of the kept
// tip, that draw is the sample there, and drawing again would spend a second
// uniform on the position:
// - kept < accepted (a clamp cut the walk's prefix): the draw at row
//   kept - 1 matched tokens[kept];
// - kept == accepted with a bonus: the bonus (fast rollback leaves it
//   uncommitted).
// Otherwise nothing was drawn at the tip: every draft matched, or the bonus
// itself was committed (kept == accepted + 1), and draw(kept - 1) samples the
// tip's row once. The caller applies the min-token floor to the result.
template <typename Draw>
inline int32_t sampled_verify_next(const SampledVerifyWalk & walk,
                                   const int32_t * tokens, int kept,
                                   Draw draw) {
    if (kept <= 0) return -1;
    if (kept < walk.accepted) return tokens[kept];
    if (kept == walk.accepted && walk.bonus >= 0) return walk.bonus;
    return draw(kept - 1);
}

}  // namespace luce::common
