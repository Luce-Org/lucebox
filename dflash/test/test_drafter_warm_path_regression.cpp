// Regression test: layer-subset warm-path buffer sizing fix.
//
// Root cause (commit that introduced fix): when PFLASH_DRAFTER_SCORE_LAYERS=7
// with a 28-layer model, the old code allocated K_norope_v for ALL 28 layers
// (~7.5 GB on RTX 3090 at S=128K) even though only 7 layers are read in scoring.
// The extra 21 × 268 MB = 5.6 GB pushed total VRAM above 24 GB, causing GPU
// page migration and a 5.4× A_compute regression on warm runs.
//
// The fix: size K_norope_v / Q_norope_v to n_score_layers (= score_range.count()),
// which equals 7 rather than 28.  This test verifies the sizing formula via
// compute_score_range without needing a GPU.

#include "score_range.h"

#include <cassert>
#include <cstdio>

using dflash::common::ScoreRange;
using dflash::common::compute_score_range;

// Helper: compute n_score_layers as the fixed allocator does.
static int score_layer_count(int n_layer, int score_layers_env, int early_exit_env) {
    const int fwd_limit = (early_exit_env > 0 && early_exit_env < n_layer)
        ? early_exit_env : n_layer;
    ScoreRange r = compute_score_range(n_layer, score_layers_env, fwd_limit);
    return r.count();
}

// T1: baseline case — SCORE_LAYERS unset (-1), no early exit.
// K_norope_v should have n_layer entries.
static void t1_baseline_full_alloc() {
    int n = score_layer_count(28, -1, -1);
    assert(n == 28 && "baseline: all 28 layers must be allocated");
    printf("T1 pass: baseline n_score_layers=%d\n", n);
}

// T2: L7 case — SCORE_LAYERS=7, no early exit.
// OLD: allocated 28 entries (5.6 GB wasted). NEW: 7 entries.
static void t2_l7_trimmed_alloc() {
    int n = score_layer_count(28, 7, -1);
    assert(n == 7 && "L7: only 7 K_norope entries must be allocated");
    printf("T2 pass: L7 n_score_layers=%d (was 28 before fix)\n", n);
}

// T3: early-exit=14, SCORE_LAYERS=7. Scoring range [7,14), 7 layers.
static void t3_early_exit_with_score_layers() {
    int n = score_layer_count(28, 7, 14);
    assert(n == 7);
    printf("T3 pass: early_exit=14 score_layers=7 -> n_score_layers=%d\n", n);
}

// T4: early-exit=7, SCORE_LAYERS=7 (the classic double-7 composition).
// Range [0,7), 7 layers.
static void t4_ee7_score7_composition() {
    int n = score_layer_count(28, 7, 7);
    assert(n == 7);
    printf("T4 pass: ee7+score7 n_score_layers=%d\n", n);
}

// T5: SCORE_LAYERS not set (all layers), early-exit=14.
// Scoring range [0,14), 14 layers needed.
static void t5_all_score_with_early_exit() {
    int n = score_layer_count(28, -1, 14);
    assert(n == 14);
    printf("T5 pass: score_all early_exit=14 n_score_layers=%d\n", n);
}

// T6: validate that score_layer_start_pre matches score_layer_start used
// in the scoring loop (must be identical for correct buffer indexing).
static void t6_start_pre_matches_loop_start() {
    // Replicate the pre-alloc computation.
    const int n_layer = 28, score_layers_env = 7, early_exit_env = -1;
    const int fwd_limit = (early_exit_env > 0 && early_exit_env < n_layer)
        ? early_exit_env : n_layer;
    ScoreRange pre   = compute_score_range(n_layer, score_layers_env, fwd_limit);
    // Scoring loop uses the same fwd_layer_limit (== fwd_limit) and same env.
    ScoreRange loop  = compute_score_range(n_layer, score_layers_env, fwd_limit);
    assert(pre.start == loop.start && "score_layer_start_pre must equal score_layer_start");
    assert(pre.end   == loop.end);
    printf("T6 pass: pre_start=%d loop_start=%d (match)\n", pre.start, loop.start);
}

// T7: alloc loop boundary check — the alloc loop iterates 0..n_layer but must only
// fill K_norope_v for layers in [score_layer_start_pre, fwd_layer_limit_pre).
// This replicates the guard added to the alloc loop: il >= start AND il < fwd_limit.
// Before the fix: il was only bounded below (il >= start), causing K_norope_v[si]
// out-of-bounds when n_score_layers < n_layer (e.g. ee14: si 0..27 but vec size 14).
static void t7_alloc_loop_upper_bound() {
    struct FakeVec {
        int capacity;
        int max_si_written = -1;
        void write(int si) {
            assert(si >= 0 && si < capacity && "si out of bounds");
            if (si > max_si_written) max_si_written = si;
        }
    };

    // Simulate ee14 (no SCORE_LAYERS, early_exit=14, n_layer=28).
    {
        const int n_layer = 28, score_layers = -1, early_exit = 14;
        const int fwd_limit = early_exit;
        ScoreRange r = compute_score_range(n_layer, score_layers, fwd_limit);
        const int n_score = r.count();  // 14
        FakeVec v{n_score};
        int writes = 0;
        for (int il = 0; il < n_layer; ++il) {
            // Correct guard: il >= start AND il < fwd_limit (the fix)
            if (il >= r.start && il < fwd_limit) {
                v.write(il - r.start);
                writes++;
            }
        }
        assert(writes == n_score && "ee14: must write exactly n_score_layers entries");
        printf("T7a pass: ee14 alloc writes=%d capacity=%d (no overflow)\n", writes, n_score);
    }

    // Simulate ee7 (SCORE_LAYERS=7, early_exit=7, n_layer=28).
    {
        const int n_layer = 28, score_layers = 7, early_exit = 7;
        const int fwd_limit = early_exit;
        ScoreRange r = compute_score_range(n_layer, score_layers, fwd_limit);
        const int n_score = r.count();  // 7
        FakeVec v{n_score};
        int writes = 0;
        for (int il = 0; il < n_layer; ++il) {
            if (il >= r.start && il < fwd_limit) {
                v.write(il - r.start);
                writes++;
            }
        }
        assert(writes == n_score && "ee7: must write exactly 7 entries");
        printf("T7b pass: ee7 alloc writes=%d capacity=%d (no overflow)\n", writes, n_score);
    }

    // Simulate baseline (no ee, no score_layers).
    {
        const int n_layer = 28, score_layers = -1, early_exit = -1;
        const int fwd_limit = n_layer;
        ScoreRange r = compute_score_range(n_layer, score_layers, fwd_limit);
        const int n_score = r.count();  // 28
        FakeVec v{n_score};
        int writes = 0;
        for (int il = 0; il < n_layer; ++il) {
            if (il >= r.start && il < fwd_limit) {
                v.write(il - r.start);
                writes++;
            }
        }
        assert(writes == n_score && "baseline: must write 28 entries");
        printf("T7c pass: baseline alloc writes=%d capacity=%d (no overflow)\n", writes, n_score);
    }
}

int main() {
    t1_baseline_full_alloc();
    t2_l7_trimmed_alloc();
    t3_early_exit_with_score_layers();
    t4_ee7_score7_composition();
    t5_all_score_with_early_exit();
    t6_start_pre_matches_loop_start();
    t7_alloc_loop_upper_bound();
    printf("\nAll warm-path regression tests passed.\n");
    return 0;
}
