// PFlash drafter placement resolution for native server/runtime code.

#pragma once

#include "placement_config.h"
#include "remote_draft_config.h"

namespace dflash::common {

struct PFlashDrafterPlacement {
    PlacementBackend target_backend = PlacementBackend::Auto;
    PlacementBackend drafter_backend = PlacementBackend::Auto;
    int drafter_gpu = 0;
    bool remote_drafter = false;
    RemoteDraftConfig remote;
};

inline bool pflash_drafter_placement_used(bool pflash_enabled,
                                          bool has_decode_draft) {
    return pflash_enabled || has_decode_draft;
}

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

}  // namespace dflash::common
