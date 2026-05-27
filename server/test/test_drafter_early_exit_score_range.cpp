// Unit tests for dflash::common::compute_score_range().
// Plain int main(), no frameworks.
//
// Verifies that SCORE_LAYERS is interpreted relative to fwd_layer_limit
// (the early-exit boundary) rather than the full model depth, so that
// early_exit_n=7 + score_layers=7 produces the non-empty range [0,7)
// instead of the phantom-empty [7,7) the old inline code produced.

#include "score_range.h"

#include <cstdio>
#include <cstdlib>

// REQUIRE survives -DNDEBUG (bare assert does not).
#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s line %d: %s\n", __FILE__, __LINE__, #cond); \
        std::exit(1); \
    } } while (0)

using dflash::common::ScoreRange;
using dflash::common::compute_score_range;

// T1 — The exact bug scenario: early_exit_n=7, score_layers=7, n_layer=28.
// OLD code: start = min(28-7, 7) = 7, end = 7 → empty loop.
// NEW code: effective_n=7, want=min(7,7)=7, start=7-7=0, end=7 → [0,7).
static void t1_bug_scenario() {
    ScoreRange r = compute_score_range(/*n_layer=*/28,
                                       /*score_layers=*/7,
                                       /*fwd_layer_limit=*/7);
    REQUIRE(r.start == 0 && "score_layer_start must be 0");
    REQUIRE(r.end   == 7 && "score_layer_end must equal fwd_layer_limit");
    REQUIRE(!r.empty()   && "range must be non-empty");
    REQUIRE(r.count() == 7);
    printf("T1 pass: early_exit_n=7 score_layers=7 n_layer=28 -> [%d,%d)\n",
           r.start, r.end);
}

// T2 — No early exit (fwd_layer_limit == n_layer).
// score_layers=7 should pick the last 7 layers [21,28).
static void t2_no_early_exit() {
    ScoreRange r = compute_score_range(28, 7, 28);
    REQUIRE(r.start == 21);
    REQUIRE(r.end   == 28);
    REQUIRE(!r.empty());
    REQUIRE(r.count() == 7);
    printf("T2 pass: no early exit score_layers=7 -> [%d,%d)\n", r.start, r.end);
}

// T3 — score_layers == -1 (all layers) with no early exit.
static void t3_all_layers_no_exit() {
    ScoreRange r = compute_score_range(28, -1, 28);
    REQUIRE(r.start == 0);
    REQUIRE(r.end   == 28);
    REQUIRE(!r.empty());
    printf("T3 pass: score_layers=-1 no exit -> [%d,%d)\n", r.start, r.end);
}

// T4 — All layers, with early exit at 14.
static void t4_all_layers_with_exit() {
    ScoreRange r = compute_score_range(28, -1, 14);
    REQUIRE(r.start == 0);
    REQUIRE(r.end   == 14);
    REQUIRE(!r.empty());
    printf("T4 pass: score_layers=-1 early_exit=14 -> [%d,%d)\n", r.start, r.end);
}

// T5 — SCORE_LAYERS larger than fwd_layer_limit: clamp to [0, fwd_layer_limit).
static void t5_score_layers_exceeds_exit() {
    // score_layers=14 but only 7 computed: want = min(14,7) = 7, start=0
    ScoreRange r = compute_score_range(28, 14, 7);
    REQUIRE(r.start == 0);
    REQUIRE(r.end   == 7);
    REQUIRE(!r.empty());
    printf("T5 pass: score_layers=14 early_exit=7 -> [%d,%d)\n", r.start, r.end);
}

// T6 — SCORE_LAYERS == n_layer (all layers) with no early exit.
static void t6_score_layers_equals_n_layer() {
    ScoreRange r = compute_score_range(28, 28, 28);
    // score_layers == n_layer → condition (score_layers < n_layer) is false → start=0
    REQUIRE(r.start == 0);
    REQUIRE(r.end   == 28);
    REQUIRE(!r.empty());
    printf("T6 pass: score_layers=n_layer=28 -> [%d,%d)\n", r.start, r.end);
}

// T7 — early_exit_n == 14, score_layers == 7: should produce [7,14).
static void t7_partial_exit_partial_score() {
    ScoreRange r = compute_score_range(28, 7, 14);
    REQUIRE(r.start == 7);
    REQUIRE(r.end   == 14);
    REQUIRE(!r.empty());
    REQUIRE(r.count() == 7);
    printf("T7 pass: early_exit=14 score_layers=7 -> [%d,%d)\n", r.start, r.end);
}

int main() {
    t1_bug_scenario();
    t2_no_early_exit();
    t3_all_layers_no_exit();
    t4_all_layers_with_exit();
    t5_score_layers_exceeds_exit();
    t6_score_layers_equals_n_layer();
    t7_partial_exit_partial_score();
    printf("\nAll score_range tests passed.\n");
    return 0;
}
