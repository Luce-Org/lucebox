#pragma once

#include <cstdint>

// Grouped columns share each Q/K load across four state columns. On Ampere,
// that reuse wins for prefill-sized launches but its larger register footprint
// loses at decode and other short batch sizes. Diagnostic overrides retain the
// existing force-over-disable precedence.
inline bool ggml_cuda_gdn_use_grouped_columns(
        bool ampere_nvidia,
        int64_t n_tokens,
        bool force_grouped_columns,
        bool disable_grouped_columns) {
    const bool ampere_prefill = ampere_nvidia && n_tokens >= 512;
    return force_grouped_columns ||
        (!disable_grouped_columns && (!ampere_nvidia || ampere_prefill));
}
