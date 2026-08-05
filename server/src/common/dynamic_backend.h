// Runtime backend selection for configurations that contain more than one
// GPU vendor in a single process.

#pragma once

#include "placement/placement_backend.h"

#include "ggml-backend.h"

#include <string>

namespace dflash::common {

struct BackendPairCapabilities {
    // Backends built by the same runtime implementation have compatible
    // device pointers, streams, and events. Different vendors must exchange
    // tensors through backend-neutral staging.
    bool same_runtime = false;
    bool native_gpu_handoff = false;
};

// Initialize one device from the requested backend. Ordinary builds keep
// using their linked backend; mixed builds load the isolated peer module on
// first use.
ggml_backend_t init_placement_backend(PlacementBackend backend,
                                      int device,
                                      std::string * error = nullptr);

// Resolve the vendor represented by an initialized ggml backend.
PlacementBackend placement_backend_of(ggml_backend_t backend);

// Describe the operations that are safe between two initialized backends.
// This deliberately keys off backend identity rather than vendor names so the
// scheduling code also works for future runtime modules.
BackendPairCapabilities backend_pair_capabilities(ggml_backend_t first,
                                                  ggml_backend_t second);

}  // namespace dflash::common
