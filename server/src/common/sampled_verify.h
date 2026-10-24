#pragma once

#include "sampler.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace luce::common {

// Single-chain sampled-verify (LUCE_SAMPLED_VERIFY=1): which drafts a verify
// block commits, and what the next step seeds with. The committed tokens must
// be distributed exactly as AR sampling of the target.

struct SampledVerifyWalk {
    int     accept_n  = 1;   // the seed plus the drafts that matched a draw
    int32_t miss_draw = -1;  // the draw at the miss row; -1 when every draft matched
};

// Row i of `rows` (v_len rows of `vocab` logits) is the target after
// draft_tok[0..i]. The walk draws row i and accepts draft_tok[i + 1] while it
// equals the draw, so a miss stops it at row accept_n - 1. `history` is the
// penalty history AR would use before the seed (the generated tokens); the
// walk appends the seed and each accepted draft, so every draw sees the
// history AR would have at that position.
inline SampledVerifyWalk sampled_verify_walk(const float * rows, int vocab,
                                             const std::vector<int32_t> & draft_tok,
                                             int v_len, const SamplerCfg & sampler,
                                             std::vector<int32_t> & history,
                                             std::mt19937_64 & rng) {
    SampledVerifyWalk walk;
    history.push_back(draft_tok[0]);
    for (int i = 0; i < v_len - 1; i++) {
        const int32_t s = sample_logits(rows + (size_t)i * vocab, vocab, sampler,
                                        history, rng);
        if (draft_tok[(size_t)i + 1] != s) {
            walk.miss_draw = s;
            break;
        }
        walk.accept_n++;
        history.push_back(s);
    }
    return walk;
}

// The token the next step seeds with when the walk already drew at the tip
// row. When the step commits only the accepted prefix (fast rollback, or the
// budget clamp dropped the bonus), the next seed comes from row commit_n - 1.
// If that is the miss row, its draw is already a target sample there and must
// be the seed: a fresh draw would give the rejected draft a second chance,
// P(d) = p(d) + (1 - p(d)) p(d). Returns -1 otherwise and the caller draws the
// tip row fresh: every draft was accepted or the bonus was committed (the tip
// row was never drawn), or the budget cut the chain before the miss row (the
// budget is then exhausted and the seed is never emitted).
inline int32_t sampled_verify_carried_seed(int32_t miss_draw,
                                           int accept_n,
                                           int commit_n) {
    if (miss_draw < 0 || commit_n != accept_n) return -1;
    return miss_draw;
}

}  // namespace luce::common
