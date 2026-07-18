#pragma once

namespace dflash::common {

// Numerics/performance policy for model-specific prefill attention. Exact is
// the tokenwise reference. Dense changes the execution/reduction topology and
// Sparse additionally prunes model-specific cache entries; neither optimized
// mode promises byte-identical logits or generated tokens.
enum class PrefillAttentionMode {
    Exact,
    Dense,
    Sparse,
};

inline const char * prefill_attention_mode_name(PrefillAttentionMode mode) {
    switch (mode) {
        case PrefillAttentionMode::Exact:  return "exact";
        case PrefillAttentionMode::Dense:  return "dense";
        case PrefillAttentionMode::Sparse: return "sparse";
    }
    return "unknown";
}

inline bool prefill_attention_mode_is_approximate(PrefillAttentionMode mode) {
    return mode != PrefillAttentionMode::Exact;
}

} // namespace dflash::common
