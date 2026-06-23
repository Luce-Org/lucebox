// Unit tests for plan_warm_prefill() — pure planner that unifies warm-prefill
// delta slicing and drafter-history reconstruction.
//
// No CUDA, no ggml: links against the header-only helper only.
// Run: test_warm_prefill (exits 0 on pass, 1 on any failure).
#include "common/warm_prefill.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

// Helper: equality check with a name
static void check_eq(const std::vector<int32_t> & a,
                     const std::vector<int32_t> & b,
                     const char * msg) {
    check(a == b, msg);
}

int main() {
    using dflash::common::plan_warm_prefill;
    using dflash::common::WarmPrefillPlan;

    const std::vector<int32_t> full = {10, 11, 20, 21, 22};

    // ── cached_prefix_len == 0: cold start ──────────────────────────────────
    // delta == full_prompt, drafter_history == full_prompt
    {
        const WarmPrefillPlan p = plan_warm_prefill(full, 0);
        check_eq(p.delta, full, "cold: delta == full_prompt");
        check_eq(p.drafter_history, full, "cold: drafter_history == full_prompt");
    }

    // ── cached_prefix_len == N (0 < N < len): partial cache hit ─────────────
    // delta == full[N:], drafter_history == full[0:N] + delta == full_prompt
    {
        const WarmPrefillPlan p = plan_warm_prefill(full, 2);
        check_eq(p.delta, (std::vector<int32_t>{20, 21, 22}),
                 "partial: delta == full[2:]");
        check_eq(p.drafter_history, full,
                 "partial: drafter_history == full_prompt");
    }

    // Another mid-point (N=4, last token to prefill)
    {
        const WarmPrefillPlan p = plan_warm_prefill(full, 4);
        check_eq(p.delta, (std::vector<int32_t>{22}),
                 "partial N=4: delta == full[4:]");
        check_eq(p.drafter_history, full,
                 "partial N=4: drafter_history == full_prompt");
    }

    // ── cached_prefix_len == len: full cache hit, nothing to prefill ─────────
    // delta empty, drafter_history == full_prompt
    {
        const WarmPrefillPlan p = plan_warm_prefill(full, (int)full.size());
        check(p.delta.empty(), "full hit: delta is empty");
        check_eq(p.drafter_history, full,
                 "full hit: drafter_history == full_prompt");
    }

    // ── out-of-range / negative: must throw (mirror restore_prompt_delta) ────
    {
        bool threw_negative = false;
        try { plan_warm_prefill(full, -1); }
        catch (const std::invalid_argument &) { threw_negative = true; }
        check(threw_negative, "negative cached_prefix_len throws invalid_argument");
    }
    {
        bool threw_over = false;
        try { plan_warm_prefill(full, (int)full.size() + 1); }
        catch (const std::out_of_range &) { threw_over = true; }
        check(threw_over, "cached_prefix_len > prompt.size() throws out_of_range");
    }

    if (failures == 0) {
        std::printf("ok: all warm_prefill tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
