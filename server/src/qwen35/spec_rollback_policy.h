#pragma once

// Pure K/V write-path rule for the Qwen3.5 target graph, kept free of
// ggml/GPU state so the KVFlash x delta-capture interaction is unit-testable
// on the host.

namespace luce::qwen35 {

// Whether build_target_step() writes K/V through the slot-mapped set_rows
// path (`kv_write_rows`). Mirrors the gate in graph_builders.cpp.
inline bool use_kv_write_rows(bool paged_attention, bool no_kvpad,
                              bool capture_delta_intermediate,
                              bool kvflash_mask, int fa_window,
                              int n_tokens, bool with_mask, bool capture) {
    // KVFlash: set_rows writes K/V in the full-attention layers, the capture
    // is in the DeltaNet layers; they share no tensor, so both are allowed.
    return paged_attention ||
           (!no_kvpad &&
            (kvflash_mask
                 ? (fa_window == 0)
                 : (!capture_delta_intermediate && n_tokens == 1 &&
                    fa_window == 0 && !with_mask && !capture)));
}

}  // namespace luce::qwen35
