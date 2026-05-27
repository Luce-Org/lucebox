// Layer-split utilities for multi-GPU model serving.
//
// Provides pure functions for computing how to distribute model layers
// across multiple GPUs and validating device configurations.

#pragma once

#include "placement/placement_config.h"

#include <string>
#include <utility>
#include <vector>

namespace dflash::common {

// Compute [begin, end) layer ranges for each GPU shard.
// If weights is empty, splits layers equally.
// If weights has entries, distributes proportionally (at least 1 layer per GPU).
// Returns empty vector on error (n_layer <= 0 or n_gpus <= 0).
std::vector<std::pair<int,int>> compute_layer_ranges(
    int n_layer,
    int n_gpus,
    const std::vector<double> & weights);

// Validate a DevicePlacement against system constraints.
// Returns empty string on success, error description on failure.
std::string validate_device_placement(
    const DevicePlacement & dp,
    int cuda_device_count);

}  // namespace dflash::common
