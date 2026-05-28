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

// Returns true if spec-decode should be attempted.
//   fa_window_override: 0 = no pflash; else = compressed_prompt_size + 256
//   fa_window_cfg     : cfg_.fa_window (default 2048)
//   kv_committed      : KV position after prefill (unused; kept for future use)
//
// Gate: permit spec-decode when eff_fa_window <= 2 * fa_window_cfg.
// For uncompressed (override==0): always permit.
// For pflash-compressed: permit only when compressed_size <= 3840 tokens.
// At compressed_size > 3840, target KV is large enough that AR is faster
// than spec-decode (empirically: D_composition 128K AR=27.5 vs spec=5.74 tok/s).
inline bool c2_spec_decode_permitted(int fa_window_override,
                                     int fa_window_cfg,
                                     int kv_committed) {
    (void)kv_committed;
    return (fa_window_override == 0)
        || (fa_window_override <= 2 * fa_window_cfg);
}

} // namespace dflash::common
