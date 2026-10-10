// Target device discovery and `--target-device auto` selection.
//
// Enumeration talks to the compiled GPU runtime; the selection policy is a
// pure function over the enumerated facts so it can be tested without a GPU.

#pragma once

#include "common/backend_args.h"
#include "common/model_capabilities.h"
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
    // Another process runs compute work here (filled only when queried; HIP
    // on Linux reads the KFD queues, other runtimes leave it false).
    bool        busy = false;
    std::string busy_by;       // e.g. "pid 1234 (luce_server)"
};

// Devices of the compiled backend, in runtime order (the order hip:N and
// cuda:N refer to). Reading free memory and other processes' use opens a
// runtime context on every device, so it is opt-in; choose_auto_target_device
// uses total memory only.
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

// How an architecture spreads one model over several local GPUs.
enum class MultiGpuTechnique {
    None,            // one device only
    LayerSplit,      // --target-devices: contiguous layer ranges per GPU
    ExpertParallel,  // --expert-device: dense work and hot experts on the
                     // main GPU, the remaining routed experts on a second one
};

inline const char * multi_gpu_technique_name(MultiGpuTechnique technique) {
    switch (technique) {
        case MultiGpuTechnique::LayerSplit:     return "layer split";
        case MultiGpuTechnique::ExpertParallel: return "expert parallel";
        case MultiGpuTechnique::None:           break;
    }
    return "single device";
}

// The qualified multi-GPU path of each family: DeepSeek V4 / V4.1 pair a
// discrete GPU with Strix Halo through expert parallelism, other families
// with a layer-split adapter split by layer.
inline MultiGpuTechnique arch_multi_gpu_technique(const std::string & arch) {
    if (arch_is_deepseek4_family(arch)) return MultiGpuTechnique::ExpertParallel;
    if (arch_supports_layer_split(arch)) return MultiGpuTechnique::LayerSplit;
    return MultiGpuTechnique::None;
}

// Auto placement must not replace a working single-device feature with an
// adapter that rejects or ignores it. Explicit split requests are still
// validated by the normal feature gate.
inline MultiGpuTechnique auto_multi_gpu_technique(
    const std::string & arch, const BackendArgs & args)
{
    // A separately placed vision encoder is supported only by the
    // single-target DeepSeek layout, not by its expert-parallel path.
    if (args.mmproj_device.has_value() || args.remote_target_shard.enabled() ||
        args.device.split_mode != TargetSplitMode::Layer) {
        return MultiGpuTechnique::None;
    }
    const MultiGpuTechnique technique = arch_multi_gpu_technique(arch);
    const ArchCapabilities * caps = find_arch_capabilities(arch);
    if (technique == MultiGpuTechnique::LayerSplit &&
        (args.mmproj_path.has_value() || args.paged_attention || args.max_concurrency > 1 ||
         (args.draft_path && caps->decode_draft == kMono) ||
         (args.ddtree_mode && caps->ddtree == kMono) ||
         (args.verify_width && caps->verify_width == kMono) ||
         (args.draft_block_size && caps->draft_block_size == kMono) ||
         (args.fa_window && caps->fa_window == kMono) ||
         (args.draft_swa_window && caps->draft_swa == kMono))) {
        return MultiGpuTechnique::None;
    }
    return technique;
}

struct AutoPlacementRequest {
    uint64_t model_bytes = 0;
    uint64_t kv_bytes = 0;
    MultiGpuTechnique technique = MultiGpuTechnique::None;
    // The backend runs a model larger than its device (expert offload), so
    // the roomiest available device is still worth trying when none fits.
    bool can_offload = false;
    // The paged DeepSeek backend only implements expert parallelism for
    // a local HIP gfx1201 primary paired with a gfx1151 secondary.
    bool paged_expert_parallel = false;
};

struct AutoPlacement {
    MultiGpuTechnique technique = MultiGpuTechnique::None;
    // Single device: one index. LayerSplit: shards in runtime order.
    // ExpertParallel: {main, expert}. Empty: no placement.
    std::vector<int>    gpus;
    std::vector<double> layer_weights;  // LayerSplit: free memory per shard
    bool        fits = false;
    std::string reason;
};

// Live placement requires queried free memory. Zero is an exhausted device,
// not a missing measurement: falling back to total memory would select it.
inline uint64_t auto_device_capacity(const GpuDeviceInfo & device) {
    return device.free_bytes;
}

// `--target-device auto` over live device state. Devices another process
// computes on are skipped, and capacity is free memory.
//  1. A discrete GPU that holds the model alone: the fastest placement
//     (Qwen 27B decodes 36 tok/s on an R9700, 15 split with Strix Halo).
//  2. Otherwise, with two or more available GPUs and a multi-GPU technique,
//     spread the model over them when their free memory holds it together:
//     subject to the launch's feature and topology constraints.
//  3. Otherwise one available GPU that holds it, discrete first.
//  4. Otherwise the roomiest available GPU when the backend can offload;
//     no placement when it cannot, or when every GPU is busy.
inline AutoPlacement choose_auto_placement(
    const std::vector<GpuDeviceInfo> & devices,
    const AutoPlacementRequest & request)
{
    AutoPlacement placement;
    std::vector<GpuDeviceInfo> available;
    std::string busy;
    for (const GpuDeviceInfo & device : devices) {
        if (device.busy) {
            busy += (busy.empty() ? "" : ", ") + std::to_string(device.index) + " (" +
                    (device.busy_by.empty() ? "another process" : device.busy_by) + ")";
            continue;
        }
        if (auto_device_capacity(device) == 0) continue;
        GpuDeviceInfo free_view = device;
        free_view.total_bytes = auto_device_capacity(device);
        available.push_back(free_view);
    }
    const std::string busy_note = busy.empty() ? "" : "; skipped busy GPU " + busy;
    if (available.empty()) {
        placement.reason = devices.empty() ? "no GPU devices found"
                                           : "no GPU has available memory" + busy_note;
        return placement;
    }

    const uint64_t need = auto_device_required_bytes(request.model_bytes, request.kv_bytes);
    const GpuDeviceInfo * fastest = nullptr;  // the roomiest discrete GPU
    for (const GpuDeviceInfo & device : available) {
        if (device.integrated) continue;
        if (!fastest || device.total_bytes > fastest->total_bytes) fastest = &device;
    }
    if (fastest && fastest->total_bytes >= need) {
        placement.gpus = {fastest->index};
        placement.fits = true;
        placement.reason = "a discrete GPU holds the model alone" + busy_note;
        return placement;
    }

    if (request.technique != MultiGpuTechnique::None && available.size() >= 2) {
        uint64_t combined = 0;
        for (const GpuDeviceInfo & device : available) combined += device.total_bytes;
        if (request.technique == MultiGpuTechnique::LayerSplit && combined >= need) {
            placement.technique = MultiGpuTechnique::LayerSplit;
            for (const GpuDeviceInfo & device : available) {
                placement.gpus.push_back(device.index);
                placement.layer_weights.push_back((double) device.total_bytes);
            }
        }
        // Expert parallelism keeps the dense work on the discrete GPU and
        // gives the roomiest other GPU the experts that do not fit there.
        if (request.technique == MultiGpuTechnique::ExpertParallel && fastest) {
            const GpuDeviceInfo * expert = nullptr;
            for (const GpuDeviceInfo & device : available) {
                if (&device == fastest) continue;
                if (!expert || device.total_bytes > expert->total_bytes) expert = &device;
            }
            const bool paged_pair_supported = !request.paged_expert_parallel ||
                (compiled_placement_backend() == PlacementBackend::Hip &&
                 fastest->arch == "gfx1201" && expert->arch == "gfx1151");
            if (paged_pair_supported && fastest->total_bytes + expert->total_bytes >= need) {
                placement.technique = MultiGpuTechnique::ExpertParallel;
                placement.gpus = {fastest->index, expert->index};
            }
        }
        if (!placement.gpus.empty()) {
            placement.fits = true;
            placement.reason = std::string(multi_gpu_technique_name(placement.technique)) +
                               " over the available GPUs" + busy_note;
            return placement;
        }
    }

    const AutoDeviceChoice single = choose_auto_target_device(
        available, request.model_bytes, request.kv_bytes);
    if (!single.fits && !request.can_offload) {
        placement.reason = "no available GPU has room for the model" + busy_note;
        return placement;
    }
    placement.gpus = {single.index};
    placement.fits = single.fits;
    placement.reason = (single.fits ? single.reason
                                    : "no available GPU holds the model; using the roomiest") +
                       busy_note;
    return placement;
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
