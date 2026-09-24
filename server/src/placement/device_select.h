// Target device discovery and `--target-device auto` selection.
//
// Enumeration talks to the compiled GPU runtime; the selection policy is a
// pure function over the enumerated facts so it can be tested without a GPU.

#pragma once

#include "ggml.h"
#include "placement_backend.h"
#include "placement_config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

struct GpuDeviceInfo {
    int         index = -1;
    std::string name;
    std::string arch;          // gfx arch on HIP, sm_XY on CUDA
    bool        integrated = false;
    uint64_t    total_bytes = 0;
    uint64_t    free_bytes = 0;    // filled only when queried
};

// Devices of the compiled backend, in runtime order (the order hip:N and
// cuda:N refer to). Reading free memory opens a runtime context on every
// device, so it is opt-in; selection uses total memory only.
std::vector<GpuDeviceInfo> enumerate_gpu_devices(bool query_free = false);

// Bytes the target weights need: the GGUF file size, summed over every part
// of a split GGUF. Returns 0 when the file or any of its parts cannot be read.
uint64_t gguf_model_bytes(const std::string & path);

// Bytes the KV cache of one max_ctx sequence needs, for models whose cache
// follows from the GGUF header: the Qwen3.5/3.6 hybrids, where only the
// full-attention layers carry KV (kv_reservation_bytes_per_token), with the
// cache types the backend resolves from the overrides (GGML_TYPE_COUNT: none)
// and LUCE_KV_*. Returns 0 for other families, whose backends size their own
// caches (DeepSeek4 compresses it, Gemma4 uses a sliding window), and for
// unreadable files: the flat margin below is all they get.
uint64_t gguf_kv_cache_bytes(const std::string & path, int max_ctx,
                             ggml_type cache_type_k = GGML_TYPE_COUNT,
                             ggml_type cache_type_v = GGML_TYPE_COUNT);

// Weights, the estimated KV cache (0 when unknown), and a margin for compute
// buffers, runtime overhead and any cache the estimate does not cover: a tenth
// of the weights, kept between 2 and 4 GiB.
inline uint64_t auto_device_required_bytes(uint64_t model_bytes, uint64_t kv_bytes = 0) {
    const uint64_t min_margin = 2ull << 30;
    const uint64_t max_margin = 4ull << 30;
    uint64_t margin = model_bytes / 10;
    if (margin < min_margin) margin = min_margin;
    if (margin > max_margin) margin = max_margin;
    return model_bytes + kv_bytes + margin;
}

struct AutoDeviceChoice {
    int         index = -1;    // -1: no device available
    bool        fits = false;  // the model fits resident on the choice
    std::string reason;
};

// Prefer a device the whole model fits on, discrete GPUs before integrated
// ones (they have their own memory bandwidth), then runtime order. When no
// device fits, take the largest one: backends with expert or layer offload
// can still run there, and it gives the monolithic loaders the best chance.
inline AutoDeviceChoice choose_auto_target_device(
    const std::vector<GpuDeviceInfo> & devices,
    uint64_t model_bytes,
    uint64_t kv_bytes = 0)
{
    AutoDeviceChoice choice;
    if (devices.empty()) {
        choice.reason = "no GPU devices found";
        return choice;
    }
    const uint64_t need = auto_device_required_bytes(model_bytes, kv_bytes);
    const GpuDeviceInfo * best = nullptr;
    for (const GpuDeviceInfo & device : devices) {
        if (device.total_bytes < need) continue;
        if (!best || (best->integrated && !device.integrated)) best = &device;
    }
    if (best) {
        choice.index = best->index;
        choice.fits = true;
        choice.reason = best->integrated
            ? "first integrated GPU that fits the model"
            : "first discrete GPU that fits the model";
        return choice;
    }
    for (const GpuDeviceInfo & device : devices) {
        if (!best || device.total_bytes > best->total_bytes) best = &device;
    }
    choice.index = best->index;
    choice.reason = "no GPU holds the model with headroom; using the largest one";
    return choice;
}

// Draft placement before architecture defaults apply, with one precedence:
// an explicit --draft-device, then the architecture's own default (DeepSeek4
// reads LUCE_DS4_DRAFT_GPU/_BACKEND), then the target GPU.
//  - An explicit placement becomes concrete: auto:N names GPU N of the
//    compiled backend, so no backend mistakes it for "unspecified".
//  - An unplaced drafter keeps the auto backend, which backends read as
//    "unspecified". With an auto-placed target it takes the target's GPU
//    index, for backends that read only the index.
inline DevicePlacement resolve_draft_placement(DevicePlacement draft,
                                               bool draft_explicit,
                                               const DevicePlacement & target,
                                               bool target_auto) {
    if (draft_explicit) {
        if (draft.backend == PlacementBackend::Auto) {
            draft.backend = compiled_placement_backend();
        }
        return draft;
    }
    if (target_auto) {
        draft.backend = PlacementBackend::Auto;
        draft.gpu = target.gpu;
    }
    return draft;
}

}  // namespace luce::common
