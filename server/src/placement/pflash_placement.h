// PFlash drafter placement resolution for native server/runtime code.

#pragma once

#include "placement_config.h"
#include "remote_draft_config.h"

#include <optional>

namespace luce::common {

struct PFlashDrafterPlacement {
    PlacementBackend target_backend = PlacementBackend::Auto;
    PlacementBackend drafter_backend = PlacementBackend::Auto;
    int drafter_gpu = 0;
    bool remote_drafter = false;
    RemoteDraftConfig remote;
};

inline PFlashDrafterPlacement resolve_pflash_drafter_placement(
        const DevicePlacement & target_device,
        const DevicePlacement & drafter_device,
        const RemoteDraftConfig & remote,
        bool pflash_enabled) {
    PFlashDrafterPlacement out;
    const PlacementBackend compiled = compiled_placement_backend();
    out.target_backend = target_device.backend == PlacementBackend::Auto
        ? compiled : target_device.backend;
    out.drafter_backend = drafter_device.backend == PlacementBackend::Auto
        ? out.target_backend : drafter_device.backend;
    out.drafter_gpu = drafter_device.gpu;
    out.remote_drafter = pflash_enabled &&
                         out.target_backend != out.drafter_backend;
    if (out.remote_drafter) {
        out.remote = remote;
    }
    return out;
}

// The PFlash drafter's device: its own (--prefill-drafter-device) when given,
// else the decode drafter's (--draft-device).
inline DevicePlacement pflash_drafter_device(
        const DevicePlacement & draft,
        const std::optional<DevicePlacement> & own) {
    return own ? *own : draft;
}

inline PlacementBackend resolved_placement_backend(PlacementBackend backend) {
    return backend == PlacementBackend::Auto ? compiled_placement_backend() : backend;
}

// Why the PFlash drafter cannot have a device of its own with this target, or
// nullptr when it can. It scores in process on the target's backend: another
// backend would need the IPC drafter, a layer-split target loads its drafter on
// its first GPU, and a target placed by --target-device auto is chosen only
// after the paged pool and parking are sized.
inline const char * pflash_drafter_device_refusal(
        const DevicePlacement & target,
        bool target_auto,
        const std::optional<DevicePlacement> & own) {
    if (!own) {
        return nullptr;
    }
    if (target_auto) {
        return "needs an explicit --target-device, not auto";
    }
    if (target.is_multi_device()) {
        return "is not supported with --target-devices";
    }
    if (resolved_placement_backend(target.backend) != resolved_placement_backend(own->backend)) {
        return "must be on the target's backend (use --draft-device for an IPC drafter)";
    }
    return nullptr;
}

// Whether the PFlash drafter scores on the target's GPU, where it needs room
// beside the target: concurrent serving reserves it in the paged pool, and a
// single request parks the target. Without its own device it follows the decode
// drafter, which concurrent serving keeps on the target's GPU.
inline bool pflash_drafter_beside_target(
        const DevicePlacement & target,
        const std::optional<DevicePlacement> & own) {
    if (!own) {
        return true;
    }
    return resolved_placement_backend(target.backend) == resolved_placement_backend(own->backend) &&
           target.gpu == own->gpu;
}

}  // namespace luce::common
