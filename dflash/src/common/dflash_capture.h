// dflash_capture.h — DFlash capture-layer index helper (target-agnostic).
//
// Maps an absolute target-layer index to its position in the capture-layer
// list (or -1 if the layer is not captured). The capture-layer list comes
// from the target architecture (e.g. Qwen35TargetWeights::capture_layer_ids
// or DFlashTarget::capture_layer_ids()).

#pragma once

namespace dflash27b {

// Linear search for layer_idx in capture_layer_ids[0..n_capture_layers).
// Returns the capture index (0..n_capture_layers-1) on hit, -1 on miss.
int target_capture_index(const int * capture_layer_ids,
                         int n_capture_layers,
                         int layer_idx);

}  // namespace dflash27b
