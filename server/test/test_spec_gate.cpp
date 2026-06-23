// Unit tests for spec_gate_active() and spec_gate_ema_update() — pure timing
// gate logic extracted from qwen35_backend and qwen35moe_backend.
//
// No CUDA, no ggml: links against the header-only helper only.
// Run: test_spec_gate (exits 0 on pass, 1 on any failure).
#include "common/spec_gate.h"

#include <cstdio>

static int failures = 0;

static void check(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

int main() {
    using dflash::common::SpecGateConfig;
    using dflash::common::SpecGateState;
    using dflash::common::spec_gate_active;
    using dflash::common::spec_gate_ema_update;

    // ── Config with all features enabled ────────────────────────────────────
    const SpecGateConfig cfg_on  { /*enabled=*/true,  /*margin=*/1.0, /*sustain=*/3, /*warmup=*/2 };
    const SpecGateConfig cfg_off { /*enabled=*/false, /*margin=*/1.0, /*sustain=*/3, /*warmup=*/2 };

    // ── Gate disabled: always inactive ──────────────────────────────────────
    {
        SpecGateState st;
        st.ema_ratio      = 0.5;  // below margin
        st.gate_low_streak = 3;   // at sustain
        // n_draft_steps >= warmup
        check(!spec_gate_active(cfg_off, st, /*n_draft_steps=*/5, /*sampled_verify=*/false),
              "disabled cfg: never active even when EMA is low");
        check(!spec_gate_active(cfg_off, st, /*n_draft_steps=*/5, /*sampled_verify=*/true),
              "disabled cfg + sampled_verify: never active");
    }

    // ── sampled_verify suppresses the gate even when enabled ────────────────
    {
        SpecGateState st;
        st.ema_ratio       = 0.5;
        st.gate_low_streak = 5;
        check(!spec_gate_active(cfg_on, st, /*n_draft_steps=*/5, /*sampled_verify=*/true),
              "sampled_verify=true: gate suppressed");
    }

    // ── Warmup period: gate is inactive regardless of EMA ───────────────────
    // kGateWarmup=2: must have n_draft_steps >= 2 before gate can fire.
    {
        SpecGateState st;
        st.ema_ratio       = 0.1;  // well below margin
        st.gate_low_streak = 10;   // above sustain
        check(!spec_gate_active(cfg_on, st, /*n_draft_steps=*/0, /*sampled_verify=*/false),
              "warmup n=0: gate inactive");
        check(!spec_gate_active(cfg_on, st, /*n_draft_steps=*/1, /*sampled_verify=*/false),
              "warmup n=1: gate inactive (< kGateWarmup=2)");
        check(spec_gate_active(cfg_on, st, /*n_draft_steps=*/2, /*sampled_verify=*/false),
              "warmup n=2: gate fires (== kGateWarmup=2)");
    }

    // ── Sustain: streak must reach kGateSustain ──────────────────────────────
    {
        SpecGateState st;
        st.ema_ratio = 0.1;

        st.gate_low_streak = 2;
        check(!spec_gate_active(cfg_on, st, /*n_draft_steps=*/5, /*sampled_verify=*/false),
              "streak=2 < sustain=3: gate inactive");

        st.gate_low_streak = 3;
        check(spec_gate_active(cfg_on, st, /*n_draft_steps=*/5, /*sampled_verify=*/false),
              "streak=3 == sustain=3: gate fires");

        st.gate_low_streak = 10;
        check(spec_gate_active(cfg_on, st, /*n_draft_steps=*/5, /*sampled_verify=*/false),
              "streak=10 > sustain=3: gate fires");
    }

    // ── EMA above margin: streak resets, gate does not fire ─────────────────
    {
        SpecGateState st;
        st.ema_ratio       = 1.5;  // above margin=1.0
        st.gate_low_streak = 10;   // high but EMA is OK -> gate should be inactive
        // gate_active is based purely on streak >= sustain AND ema logic is
        // captured through the streak; streak is updated by ema_update not here.
        // When streak >= sustain the gate fires regardless of current ema_ratio
        // (streak is the accumulator). This test verifies a streak=0 case.
        st.gate_low_streak = 0;
        check(!spec_gate_active(cfg_on, st, /*n_draft_steps=*/5, /*sampled_verify=*/false),
              "streak=0 (EMA was good): gate inactive");
    }

    // ── EMA update: streak increments when ema_ratio < margin ───────────────
    {
        SpecGateState st;
        st.ema_ratio       = 2.0;  // init optimistic
        st.gate_low_streak = 0;
        double t_ar = 0.1;  // 100ms per AR token

        // Step 1: spec was slower than AR → ratio < 1.0 → streak++
        // step_wall=0.5s, commit_n=1 → ratio = 1*0.1/0.5 = 0.2
        spec_gate_ema_update(cfg_on, st, /*commit_n=*/1, /*step_wall=*/0.5, t_ar);
        // new ema = 0.5*2.0 + 0.5*0.2 = 1.1 — above margin=1.0 → streak stays 0
        check(st.gate_low_streak == 0, "ema_update: streak stays 0 when ema above margin");

        // Step 2: spec very slow → ratio = 1*0.1/2.0 = 0.05
        spec_gate_ema_update(cfg_on, st, /*commit_n=*/1, /*step_wall=*/2.0, t_ar);
        // new ema = 0.5*1.1 + 0.5*0.05 = 0.575 — below margin=1.0 → streak++
        check(st.gate_low_streak == 1, "ema_update: streak=1 after ema drops below margin");

        // Three more slow steps → streak reaches 4
        spec_gate_ema_update(cfg_on, st, 1, 2.0, t_ar);
        spec_gate_ema_update(cfg_on, st, 1, 2.0, t_ar);
        spec_gate_ema_update(cfg_on, st, 1, 2.0, t_ar);
        check(st.gate_low_streak == 4, "ema_update: streak=4 after 4 slow steps");
        check(spec_gate_active(cfg_on, st, 5, false),
              "gate fires after streak >= sustain=3");
    }

    // ── EMA update: streak resets when ema_ratio >= margin ──────────────────
    {
        SpecGateState st;
        st.ema_ratio       = 0.3;
        st.gate_low_streak = 5;
        double t_ar = 0.1;

        // Fast step: commit_n=10, step_wall=0.1 → ratio = 10*0.1/0.1 = 10.0
        spec_gate_ema_update(cfg_on, st, 10, 0.1, t_ar);
        // new ema = 0.5*0.3 + 0.5*10.0 = 5.15 → above margin → streak resets
        check(st.gate_low_streak == 0, "ema_update: streak resets on good step");
    }

    // ── EMA update: t_ar=0 guard (ratio defaults to 2.0) ────────────────────
    {
        SpecGateState st;
        st.ema_ratio       = 2.0;
        st.gate_low_streak = 0;
        // t_ar=0 → ratio stays 2.0 (guard), streak should not increment
        spec_gate_ema_update(cfg_on, st, 1, 0.5, /*t_ar=*/0.0);
        check(st.gate_low_streak == 0, "t_ar=0: ratio defaults to 2.0, streak stays 0");
    }

    // ── EMA update: step_wall=0 guard ────────────────────────────────────────
    {
        SpecGateState st;
        st.ema_ratio       = 2.0;
        st.gate_low_streak = 0;
        spec_gate_ema_update(cfg_on, st, 1, /*step_wall=*/0.0, /*t_ar=*/0.1);
        check(st.gate_low_streak == 0, "step_wall=0: ratio defaults to 2.0, streak stays 0");
    }

    if (failures == 0) {
        std::printf("ok: all spec_gate tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
