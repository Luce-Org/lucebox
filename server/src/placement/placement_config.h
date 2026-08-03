// Device placement configuration for model backends.
//
// Describes which backend device(s) to use for a model. Supports:
//   - Single-GPU: backend + gpu fields, exposed as cuda:0 / hip:0 / auto:0
//   - Multi-GPU layer-split: per-shard backend + layer_split_gpus + optional weights
//   - Peer access between GPUs

#pragma once

#include "placement_backend.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace dflash::common {

enum class TargetSplitMode {
    Layer,
    Tensor,
};

inline const char * target_split_mode_name(TargetSplitMode mode) {
    return mode == TargetSplitMode::Tensor ? "tensor" : "layer";
}

inline bool parse_target_split_mode(const std::string & value,
                                    TargetSplitMode & out) {
    if (value == "layer") {
        out = TargetSplitMode::Layer;
        return true;
    }
    if (value == "tensor") {
        out = TargetSplitMode::Tensor;
        return true;
    }
    return false;
}

struct DevicePlacement {
    PlacementBackend backend = PlacementBackend::Auto;
    int gpu = 0;                              // primary GPU (single-GPU mode)
    TargetSplitMode split_mode = TargetSplitMode::Layer;

    // Multi-GPU layer-split. Empty = single GPU mode.
    std::vector<PlacementBackend> layer_split_backends; // backend for each shard
    std::vector<int>              layer_split_gpus;     // GPU IDs for each shard
    std::vector<double>           layer_split_weights;  // proportional layer distribution (optional)

    bool peer_access = false;                 // enable CUDA/HIP peer access between GPUs
    int  max_ctx     = 8192;                  // max KV cache context length

    bool is_multi_device() const { return layer_split_gpus.size() > 1; }
    bool is_layer_split() const {
        return is_multi_device() && split_mode == TargetSplitMode::Layer;
    }
    bool is_tensor_parallel() const {
        return is_multi_device() && split_mode == TargetSplitMode::Tensor;
    }
    bool is_mixed_layer_split() const {
        if (layer_split_backends.size() <= 1) return false;
        const PlacementBackend first = layer_split_backends[0];
        for (PlacementBackend backend : layer_split_backends) {
            if (backend != first) return true;
        }
        return false;
    }
    PlacementBackend layer_split_backend(size_t idx) const {
        return idx < layer_split_backends.size() ? layer_split_backends[idx] : backend;
    }

    int primary_gpu() const {
        return layer_split_gpus.empty() ? gpu : layer_split_gpus[0];
    }
};

inline std::string placement_device_name(const DevicePlacement & device) {
    return std::string(placement_backend_name(device.backend)) + ":" +
           std::to_string(device.primary_gpu());
}

inline bool parse_placement_device(const std::string & value,
                                   DevicePlacement & out) {
    const std::size_t sep = value.find(':');
    if (sep == std::string::npos || sep == 0 || sep + 1 >= value.size()) {
        return false;
    }

    PlacementBackend backend = PlacementBackend::Auto;
    if (!parse_placement_backend(value.substr(0, sep), backend)) {
        return false;
    }

    const std::string gpu_text = value.substr(sep + 1);
    char * end = nullptr;
    long gpu = std::strtol(gpu_text.c_str(), &end, 10);
    if (end == gpu_text.c_str() || *end != '\0' || gpu < 0) {
        return false;
    }

    out.backend = backend;
    out.gpu = static_cast<int>(gpu);
    out.layer_split_backends.clear();
    out.layer_split_gpus.clear();
    out.layer_split_weights.clear();
    return true;
}

inline bool parse_placement_device_list(const std::string & value,
                                        DevicePlacement & out) {
    if (value.empty()) return false;

    std::vector<PlacementBackend> backends;
    std::vector<int> gpus;
    PlacementBackend backend = PlacementBackend::Auto;
    bool have_backend = false;

    std::size_t begin = 0;
    while (begin < value.size()) {
        const std::size_t end = value.find(',', begin);
        const std::string item = value.substr(
            begin,
            end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty()) return false;

        DevicePlacement parsed;
        if (!parse_placement_device(item, parsed)) return false;
        if (!have_backend) {
            backend = parsed.backend;
            have_backend = true;
        }
        backends.push_back(parsed.backend);
        gpus.push_back(parsed.gpu);

        if (end == std::string::npos) break;
        begin = end + 1;
    }

    if (gpus.empty()) return false;
    out.backend = backend;
    out.gpu = gpus[0];
    out.layer_split_backends =
        gpus.size() > 1 ? backends : std::vector<PlacementBackend>{};
    out.layer_split_gpus = gpus.size() > 1 ? gpus : std::vector<int>{};
    out.layer_split_weights.clear();
    return true;
}

// Validate a DevicePlacement against system constraints.
// If device_count is negative, only validates structural constraints that do
// not require querying the runtime-visible GPU count.
// Returns empty string on success, error description on failure.
std::string validate_device_placement(
    const DevicePlacement & dp,
    int device_count);

}  // namespace dflash::common
