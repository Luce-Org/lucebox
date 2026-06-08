// C2 gate predicate — pure function, no GPU/model deps.
// Extracted from qwen35_backend.cpp for testability.
//
// Reasoning: when pflash compresses a 128K prompt to ~11K tokens, the
// target KV at decode time = 11K (small). T_target is fast (small KV),
// T_draft ≈ constant. r = T_draft/T_target ≈ 1, so spec-decode does NOT
// win over AR. Empirical: D_composition 128K: AR=27.5 tok/s, spec=5.74 tok/s.
// Gate correctly blocks spec-decode when eff_fa_window > 2*fa_window_cfg.
#pragma once

namespace dflash::common {

// Spec-decode budget reference. fa_window=0 is full attention (required for
// tool calls — a finite window drops the system prompt). But the spec-decode
// admission budget must NOT collapse to 0; it is decoupled from the AR window.
constexpr int kSpecCompressFaRef = 2048;

// Spec-decode only wins on short, high-accept contexts (empirically lost to AR
// by ~17K on agentic; won ~6K); 8192 is a conservative default — tunable per workload.
inline constexpr int kSpecMaxUncompressedCtx = 8192;

// Resolve the fa_window reference used by the spec-decode admission math.
// Production default fa_window=0 → use kSpecCompressFaRef so the gate/ladder
// bands (2x, 1.5x) yield the intended 4096 ceiling. A passed --fa-window>0
// is honored verbatim (preserves prior behavior/tests).
inline int spec_fa_ref(int fa_window_cfg) {
    return fa_window_cfg > 0 ? fa_window_cfg : kSpecCompressFaRef;
}

// Returns true if spec-decode should be attempted.
//   fa_window_override: 0 = no pflash; else = compressed_prompt_size + 256
//   fa_window_cfg     : cfg_.fa_window (default 2048)
//   kv_committed      : KV position after prefill (real context depth)
//
// Compressed (pflash active): keep existing budget gate.
// Uncompressed warm/cold turn: gate on real context length.
// Spec-decode loses to AR on long context (accept collapses); force AR there.
inline bool c2_spec_decode_permitted(int fa_window_override,
                                     int fa_window_cfg,
                                     int kv_committed) {
    if (fa_window_override > 0) {
        return fa_window_override <= 2 * fa_window_cfg;
    }
    return kv_committed < kSpecMaxUncompressedCtx;
}

} // namespace dflash::common
