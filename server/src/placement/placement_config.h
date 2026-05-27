// Device placement configuration for model backends.
//
// Describes which backend device(s) to use for a model. Supports:
//   - Single-GPU: backend + gpu fields, exposed as cuda:0 / hip:0 / auto:0
//   - Multi-GPU layer-split: one backend + layer_split_gpus + optional weights
//   - Peer access between GPUs

#pragma once

#include "placement_backend.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace dflash::common {

struct DevicePlacement {
    PlacementBackend backend = PlacementBackend::Auto;
    int gpu = 0;                              // primary GPU (single-GPU mode)

    // Multi-GPU layer-split. Empty = single GPU mode.
    std::vector<int>    layer_split_gpus;     // GPU IDs for each shard
    std::vector<double> layer_split_weights;  // proportional layer distribution (optional)

    bool peer_access = false;                 // enable CUDA/HIP peer access between GPUs
    int  max_ctx     = 8192;                  // max KV cache context length

    bool is_layer_split() const { return layer_split_gpus.size() > 1; }

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
    out.layer_split_gpus.clear();
    out.layer_split_weights.clear();
    return true;
}

inline bool parse_placement_device_list(const std::string & value,
                                        DevicePlacement & out) {
    if (value.empty()) return false;

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
        } else if (parsed.backend != backend) {
            return false;
        }
        gpus.push_back(parsed.gpu);

        if (end == std::string::npos) break;
        begin = end + 1;
    }

    if (gpus.empty()) return false;
    out.backend = backend;
    out.gpu = gpus[0];
    out.layer_split_gpus = gpus.size() > 1 ? gpus : std::vector<int>{};
    out.layer_split_weights.clear();
    return true;
}

}  // namespace dflash::common
