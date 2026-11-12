#pragma once

#include "common/output_constraint.h"
#include "common/sampler.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace luce::common {

// Sampled verify of a deterministic chain draft. The target draws each
// position from its own sampler; a drafted token is kept while it equals the
// draw. Every committed token is then a single target draw at its position,
// so the output follows autoregressive sampling exactly, and the draft only
// decides how far one verify goes.

// LUCE_SAMPLED_VERIFY: 1 turns sampled verify on, 0 turns it off (sampled
// requests then decode one token at a time); unset or any other value keeps
// the backend's default, which is on where it was measured.
inline bool sampled_verify_switch(bool default_on) {
    const char * e = std::getenv("LUCE_SAMPLED_VERIFY");
    if (e != nullptr && std::strcmp(e, "1") == 0) return true;
    if (e != nullptr && std::strcmp(e, "0") == 0) return false;
    return default_on;
}

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
// cfg.constraint is at the position of tokens[0] (the commit point has not
// kept it yet); the walk draws from a snapshot it advances over the root and
// each kept draft, so every draw sees the grammar state decoding would.
//
// draw(i, cfg, history, rng) samples row i as sample_logits would, or returns
// -1 when the row cannot be read (the walk then returns accepted 0). The walk
// draws a row only when it reaches it, so a caller can leave the rows after
// the first mismatch on the device, or draw from each row's top candidates.
template <typename Draw, typename Stops>
inline SampledVerifyWalk sampled_verify_walk_draws(
        Draw draw, int width, const int32_t * tokens,
        const SamplerCfg & cfg, std::vector<int32_t> & history,
        std::mt19937_64 & rng, Stops stops) {
    SampledVerifyWalk walk;
    SamplerCfg round = cfg;
    if (round.constraint) {
        round.constraint = cfg.constraint->snapshot();
        round.constraint->accept(tokens[0]);
    }
    history.push_back(tokens[0]);
    for (int i = 0; i + 1 < width; ++i) {
        const int32_t token = draw(i, static_cast<const SamplerCfg &>(round), history, rng);
        if (token < 0) return SampledVerifyWalk{0, -1};
        if (token != tokens[i + 1] || stops(token, i + 1)) {
            walk.bonus = token;
            break;
        }
        history.push_back(token);
        if (round.constraint) round.constraint->accept(token);
        ++walk.accepted;
    }
    return walk;
}

// The same walk over rows already on the host: rows[i] at rows + i * vocab.
template <typename Stops>
inline SampledVerifyWalk sampled_verify_walk(
        const float * rows, int vocab, int width, const int32_t * tokens,
        const SamplerCfg & cfg, std::vector<int32_t> & history,
        std::mt19937_64 & rng, Stops stops) {
    return sampled_verify_walk_draws(
        [rows, vocab](int i, const SamplerCfg & round, std::vector<int32_t> & h,
                      std::mt19937_64 & r) {
            return sample_logits(
                rows + static_cast<size_t>(i) * static_cast<size_t>(vocab),
                vocab, round, h, r);
        },
        width, tokens, cfg, history, rng, stops);
}

inline SampledVerifyWalk sampled_verify_walk(
        const float * rows, int vocab, int width, const int32_t * tokens,
        const SamplerCfg & cfg, std::vector<int32_t> & history,
        std::mt19937_64 & rng) {
    return sampled_verify_walk(rows, vocab, width, tokens, cfg, history, rng,
                               [](int32_t, int) { return false; });
}

// Greedy verify of a chain draft under a grammar (cfg.greedy_under_constraint(),
// cfg.constraint at the position of the root tokens[0]). argmax[i] is the
// target's argmax of row i, the logits after tokens[0..i]. Where the grammar
// rejects it, row(i, vocab) returns the row's logits (null when they cannot
// be read) and the sampler picks the token as autoregressive decoding does.
// The walk rewrites argmax[i] with that token for every row it reaches (the
// last one included) and returns how many tokens it keeps, the root and the
// drafts that equal it, as the unconstrained greedy accept does; 0 when a
// row could not be read.
template <typename Row>
inline int constrained_greedy_walk(int32_t * argmax, int width, const int32_t * tokens,
                                   const SamplerCfg & cfg, Row row) {
    SamplerCfg round = cfg;
    round.constraint = cfg.constraint->snapshot();
    OutputConstraint & grammar = *round.constraint;
    grammar.accept(tokens[0]);
    static const std::vector<int32_t> no_history;  // greedy: no penalties
    std::mt19937_64 unused_rng;                     // greedy: no draws
    int accepted = 1;
    for (int i = 0; i < width; ++i) {
        if (!grammar.allows(argmax[i])) {
            int vocab = 0;
            const float * logits = row(i, vocab);
            if (logits == nullptr) return 0;
            argmax[i] = sample_logits(logits, vocab, round, no_history, unused_rng);
        }
        if (i + 1 == width || tokens[i + 1] != argmax[i]) break;
        grammar.accept(argmax[i]);
        ++accepted;
    }
    return accepted;
}

// The next root after a forward whose argmax is `tok`, with cfg.constraint at
// the root's position. Greedy under a grammar keeps the argmax where the
// grammar allows it and takes the sampler's pick from row() (the same
// logits) where it does not, as constrained AR decoding does; a root the
// grammar rejects would mark it dead at the next feed. Returns -1 when the row
// cannot be read. Every place that seeds a root from an argmax calls this:
// the chain step's replay seed and the AR hop's.
template <typename Row>
inline int32_t constrained_next_root(int32_t tok, const SamplerCfg & cfg,
                                     std::vector<int32_t> & history,
                                     std::mt19937_64 & rng, Row row) {
    if (!cfg.greedy_under_constraint() || cfg.constraint->allows(tok)) return tok;
    int vocab = 0;
    const float * logits = row(vocab);
    if (logits == nullptr) return -1;
    return sample_logits(logits, vocab, cfg, history, rng);
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
// tip's row once. Any min-token floor on the result is the caller's.
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
