// dflash_capture.h — DFlash capture-layer index helper (target-agnostic).
//
// Maps an absolute target-layer index to its position in the capture-layer
// list (or -1 if the layer is not captured). The capture-layer list comes
// from the target architecture (e.g. Qwen35TargetWeights::capture_layer_ids
// or DFlashTarget::capture_layer_ids()).

#pragma once

#include <vector>

namespace luce::common {

// Linear search for layer_idx in capture_layer_ids[0..n_capture_layers).
// Returns the capture index (0..n_capture_layers-1) on hit, -1 on miss.
int target_capture_index(const int * capture_layer_ids,
                         int n_capture_layers,
                         int layer_idx);

// Adopt a drafter's explicit capture layers (GGUF dflash.target_layer_ids)
// into a target capture table of `max_slots` entries. The count follows the
// drafter (its fc width), not the 5-layer default: capturing different layers
// than the drafter was trained on silently destroys acceptance. Returns false
// and leaves the table untouched when the list is empty, longer than the
// table, or names a layer outside [0, target_n_layer).
bool adopt_drafter_capture_layers(const std::vector<int> & drafter_ids,
                                  int target_n_layer,
                                  int max_slots,
                                  int * capture_layer_ids,
                                  int & n_capture_layers);

}  // namespace luce::common
