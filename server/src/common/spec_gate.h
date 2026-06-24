#pragma once
// Pure timing gate for self-calibrating spec-decode.
//
// Backends keep their own env-var reads (SpecGateConfig is constructed once
// in each decode function) and their own EMA state members.  Only the pure
// decision and EMA-update expressions are shared here.
//
// No CUDA, no ggml, no std::filesystem — header-only, testable standalone.

namespace dflash::common {

// Immutable configuration read once per decode call from env vars.
struct SpecGateConfig {
    bool   enabled;  // DFLASH_SPEC_GATE (alias DFLASH_ENTROPY_GATE), default true
    double margin;   // DFLASH_SPEC_GATE_MARGIN,  default 1.0
    int    sustain;  // DFLASH_SPEC_GATE_SUSTAIN, default 3
    int    warmup;   // DFLASH_SPEC_GATE_WARMUP,  default 2
};

// Mutable EMA state.  Promoted to backend member so state persists across turns.
struct SpecGateState {
    double ema_ratio      = 0.0; // EMA of realized speedup; pessimistic init (gates fast on losers)
    int    gate_low_streak = 0;
    int    n_ema_updates   = 0;  // total updates; gates hard-floor until warmup complete
};

// Returns true when the gate decides to floor to AR decode.
//
// sampled_verify — must be false to allow gating; sampled acceptance is
//   distribution-preserving so timing routing would change the distribution.
// n_draft_steps  — number of spec steps completed so far (gate is silent
//   during the warmup window, i.e. when n_draft_steps < cfg.warmup).
//
// Two floor paths:
//   streak path  — ema_ratio < margin for sustain consecutive steps (borderline)
//   hard path    — ema_ratio < 0.5 * margin after warmup steps (clear 2×-slower loser)
inline bool spec_gate_active(const SpecGateConfig & cfg,
                             const SpecGateState  & st,
                             int                    n_draft_steps,
                             bool                   sampled_verify) {
    if (!cfg.enabled || sampled_verify || n_draft_steps < cfg.warmup) return false;
    if (st.gate_low_streak >= cfg.sustain)                              return true;  // streak path
    if (st.n_ema_updates >= cfg.warmup && st.ema_ratio < 0.5 * cfg.margin) return true;  // hard path
    return false;
}

// Update EMA and low-streak counter after each spec step.
//
// commit_n  — tokens committed this step (accepted + any injected prefix)
// step_wall — wall time for this step in seconds
// t_ar      — measured AR per-token wall time in seconds (0 = not yet known)
inline void spec_gate_ema_update(const SpecGateConfig & cfg,
                                 SpecGateState        & st,
                                 int                    commit_n,
                                 double                 step_wall,
                                 double                 t_ar) {
    const double ratio = (t_ar > 0.0 && step_wall > 0.0)
        ? ((double)commit_n * t_ar / step_wall)
        : 1.0;  // neutral when timing unavailable
    st.ema_ratio = 0.5 * st.ema_ratio + 0.5 * ratio;
    st.n_ema_updates++;
    if (st.ema_ratio < cfg.margin) st.gate_low_streak++;
    else                           st.gate_low_streak = 0;
}

} // namespace dflash::common
