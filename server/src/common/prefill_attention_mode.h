#pragma once

namespace dflash::common {

// Numerics/performance policy for model-specific prefill attention. Exact is
// the tokenwise reference, Dense selects a model's fused dense kernel, and
// Sparse may prune model-specific cache entries.
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

} // namespace dflash::common
