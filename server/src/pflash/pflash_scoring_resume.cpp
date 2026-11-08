#include "pflash/pflash_scoring_resume.h"

#include <algorithm>

namespace luce::pflash {

static_assert(kScoringResumeGrid % kScoringUbatch == 0, "the resume grid must hold whole ubatches");

int pflash_scoring_checkpoint(int S) {
    const int cut = S - kScoringCutBack;
    if (cut < kScoringResumeGrid) return 0;
    return cut / kScoringResumeGrid * kScoringResumeGrid;
}

int pflash_scoring_proj_begin(int) { return 0; }

bool pflash_scoring_proj_rows_cover(int, int, int) { return false; }

int pflash_scoring_exact_rows_end(int S) {
    const int cut = S - kScoringCutBack;
    if (cut < kScoringUbatch) return 0;
    return cut / kScoringUbatch * kScoringUbatch;
}

int pflash_scoring_resume(int checkpoint, int shared, int S, int cap) {
    if (checkpoint <= 0 || checkpoint % kScoringResumeGrid != 0) return 0;
    if (checkpoint > shared || checkpoint > S - kScoringCutBack || checkpoint > cap) return 0;
    return checkpoint;
}

PFlashResumePick pflash_pick_resume(const std::vector<PFlashResumeCandidate> & sessions,
                                    const std::vector<int32_t> & ids, int cap) {
    const int S = (int) ids.size();
    const int S_padded = (S + 255) / 256 * 256;
    PFlashResumePick best;
    for (size_t i = 0; i < sessions.size(); ++i) {
        const PFlashResumeCandidate & c = sessions[i];
        if (!c.usable || !c.ids || c.ids->empty()) continue;
        const size_t n = std::min(c.ids->size(), ids.size());
        const int shared = (int) (std::mismatch(c.ids->begin(), c.ids->begin() + (long) n, ids.begin()).first -
                                  c.ids->begin());
        int r = pflash_scoring_resume(c.checkpoint, shared, S, cap);
        if (r > 0 && c.rows_cover) r = c.rows_cover(r, shared);
        if (r <= 0) continue;
        const bool grow = c.capacity < S_padded;
        if (r > best.resume || (r == best.resume && best.needs_grow && !grow)) {
            best = {(int) i, r, shared, grow};
        }
    }
    return best;
}

std::vector<int> pflash_span_tokens(int begin, int end) {
    std::vector<int> tokens;
    for (int t = begin; t < end; ++t) tokens.push_back(t);
    return tokens;
}

bool PFlashExactRowSets::put(const std::vector<int> & tokens, const float * rows, int hidden, int exact_end) {
    if (tokens.empty() || hidden <= 0 || !rows || tokens.front() < 0 || tokens.back() >= exact_end) return false;
    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] <= tokens[i - 1]) return false;
    }
    sets_.erase(std::remove_if(sets_.begin(), sets_.end(),
                               [&tokens](const Set & s) { return s.tokens == tokens; }),
                sets_.end());
    Set set;
    set.tokens = tokens;
    set.rows.assign(rows, rows + (size_t) hidden * tokens.size());
    sets_.push_back(std::move(set));
    while (sets_.size() > max_sets_) sets_.erase(sets_.begin());
    return true;
}

bool PFlashExactRowSets::lookup(const std::vector<int> & tokens, int hidden, std::vector<float> & rows) const {
    for (const auto & s : sets_) {
        if (s.tokens == tokens && s.rows.size() == (size_t) hidden * tokens.size()) {
            rows = s.rows;
            return true;
        }
    }
    return false;
}

int PFlashExactRowSets::invalidate_from(int position) {
    const size_t before = sets_.size();
    sets_.erase(std::remove_if(sets_.begin(), sets_.end(),
                               [position](const Set & s) { return s.tokens.back() >= position; }),
                sets_.end());
    return (int) (before - sets_.size());
}

int pflash_session_capacity(int S, int ctx_limit) {
    const auto align256 = [](long v) { return (v + 255) / 256 * 256; };
    const long need = align256(std::max(S, 0));
    long want = (long) S + S / 2 + 4096;
    if (ctx_limit > 0) want = std::min(want, align256(ctx_limit));
    return (int) std::max(need, want);
}

int pflash_session_ctx_limit(int drafter_ctx, int engine_max_ctx) {
    if (drafter_ctx > 0 && engine_max_ctx > 0) return std::min(drafter_ctx, engine_max_ctx);
    return std::max(std::max(drafter_ctx, engine_max_ctx), 0);
}

} // namespace luce::pflash
