#pragma once

// Exact drafter-session resume for the strict scorer (qwen35_drafter.cpp).
//
// A fresh scoring call over S tokens runs blocks 0..14 in 1024-token ubatches from token 0, cut once at S - 64, and
// projects keys / probe / request logits in 8192-token chunks from token 0. A resumed call reproduces those values
// bit for bit only when it uses the same partition for every token it recomputes and reads stored values that a
// fresh call would also have computed in that partition:
//   * it resumes at a multiple of kScoringResumeGrid (1024 = the ubatch), so its ubatches fall on the fresh call's
//     grid, and at or before S - 64, so the cut stays where the fresh call has it;
//   * the session keeps one recurrent-state snapshot, at pflash_scoring_checkpoint(S) of the call that stored it;
//   * stored block-14 rows are exact only below pflash_scoring_exact_rows_end(S) of the call that computed them
//     (they sit in full grid ubatches there; later rows met the S - 64 cut). That bound equals the checkpoint;
//   * the projection chunks (kScoringProjChunk, 8192) stay the fresh call's: a resume inside a chunk re-projects
//     that whole chunk from pflash_scoring_proj_begin(resume), taking the block-14 rows ahead of the resume point
//     from the session (it keeps [proj_begin(checkpoint), checkpoint), at most one chunk) and the rest from this
//     call. Values the old call projected in a shorter, cut-off chunk are overwritten, never reused.
// Nothing here approximates: a value that is not exactly available sends the call back to token 0.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace luce::pflash {

constexpr int kScoringResumeGrid = 1024;
constexpr int kScoringProjChunk = 8192;   // keys / probe / request logits are projected in chunks this long from 0
constexpr int kScoringUbatch = 1024;
constexpr int kScoringCutBack = 64;

// The snapshot position a call over ``S`` tokens keeps: the last multiple of the grid at or before S - 64 (0 when
// there is none: nothing to resume from).
int pflash_scoring_checkpoint(int S);

// Start of the projection chunk that holds a resume at ``resume`` (a multiple of kScoringProjChunk, <= resume).
int pflash_scoring_proj_begin(int resume);

// True when the block-14 rows [rows_begin, rows_begin + rows) a session kept cover what a resume at ``resume``
// re-projects ahead of the resume point: exactly [pflash_scoring_proj_begin(resume), resume).
bool pflash_scoring_proj_rows_cover(int resume, int rows_begin, int rows);

// Rows this call computes at positions below this are exact for any later prompt that shares them.
int pflash_scoring_exact_rows_end(int S);

// Where a call over ``S`` tokens resumes from a session whose snapshot is ``checkpoint`` and that shares
// ``shared`` leading tokens with it: the checkpoint when it is on the grid, > 0, <= shared, <= S - 64 and
// <= ``cap``; else 0 (a fresh call).
int pflash_scoring_resume(int checkpoint, int shared, int S, int cap);

// One stored session as the resume pick sees it. ``rows_cover(r, shared)`` returns r when the stored rows the call
// needs (query windows, probe / subunit / request rows) cover a resume at r, else 0; empty = no row condition.
struct PFlashResumeCandidate {
    const std::vector<int32_t> * ids = nullptr;   // prompt tokens the session covers
    int checkpoint = 0;
    int capacity = 0;                              // tokens the session can hold
    bool usable = true;                            // false: skip (wrong head, nothing stored)
    std::function<int(int r, int shared)> rows_cover;
};

struct PFlashResumePick {
    int index = -1;          // the session to resume from, -1 = fresh call
    int resume = 0;
    int shared = 0;          // leading tokens it shares with the prompt
    bool needs_grow = false; // its capacity is below the prompt rounded up to 256: grow it, do not discard it
};

// The session a call over ``ids`` resumes from: the largest resume point (pflash_scoring_resume, then rows_cover);
// at equal points a session that fits wins. A session too small for the prompt is still a candidate (needs_grow).
PFlashResumePick pflash_pick_resume(const std::vector<PFlashResumeCandidate> & sessions,
                                    const std::vector<int32_t> & ids, int cap);

// Token capacity of a new or regrown scoring session for an ``S``-token prompt: S + S/2 + 4096 (headroom, so later
// turns of the conversation append without reallocating), but never past ``ctx_limit`` rounded up to 256 (the
// longest prompt the session can be asked to hold: the target's max context and the drafter's trained context;
// <= 0 = no limit) and never below S rounded up to 256 (what this call needs).
int pflash_session_capacity(int S, int ctx_limit);

// The session token limit from the drafter's trained context and the serving engine's max context: the smaller of
// the two that are > 0, or 0 when neither is.
int pflash_session_ctx_limit(int drafter_ctx, int engine_max_ctx);

// The token positions of the window [begin, end), ascending; empty when end <= begin.
std::vector<int> pflash_span_tokens(int begin, int end);

// Exact F32 block-14 rows of query token sets (the query step's candidate and END sets), kept per session for later
// turns whose activations are gone. Only sets whose last token is below the storing call's exact-rows end are put.
class PFlashExactRowSets {
public:
    explicit PFlashExactRowSets(size_t max_sets = 8) : max_sets_(max_sets) {}

    // Store ``rows`` ([hidden, tokens.size()]) for the ascending ``tokens``; replaces an equal set, evicts the
    // least recently stored past max_sets. False (nothing stored) when tokens are empty, not ascending, reach
    // ``exact_end``, or rows have the wrong size.
    bool put(const std::vector<int> & tokens, const float * rows, int hidden, int exact_end);
    // The rows of exactly this token set, when stored.
    bool lookup(const std::vector<int> & tokens, int hidden, std::vector<float> & rows) const;
    // Drop every set with a token at or after ``position`` (the prompt diverged there). Returns how many.
    int invalidate_from(int position);
    void clear() { sets_.clear(); }
    size_t size() const { return sets_.size(); }

private:
    struct Set {
        std::vector<int> tokens;
        std::vector<float> rows;
    };
    size_t max_sets_;
    std::vector<Set> sets_;   // oldest first
};

} // namespace luce::pflash
