// Snapshot backend selection utility.
//
// Prefix cache snapshots can be large (full KV cache copies). On discrete
// GPUs (CUDA, HIP) keeping them in VRAM wastes scarce device memory and can
// cause performance degradation when VRAM spills to system RAM.
//
// On unified-memory platforms (Metal, AMD iGPU/HALO) the GPU buffer IS
// host memory, so there is no benefit from a separate CPU backend.
//
// This header provides helpers to select the appropriate backend for
// snapshot storage based on the compute backend's memory properties.

#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>

namespace dflash::common {

// Select or create a backend for prefix cache snapshot storage.
//
// If the compute backend's default buffer type is host-accessible (unified
// memory — e.g. Metal, AMD iGPU), returns compute_backend itself since GPU
// and CPU share the same physical memory and no copy is needed.
//
// If the buffer type is NOT host-accessible (discrete VRAM — e.g. CUDA,
// HIP), allocates and returns a CPU backend so snapshots reside in system
// RAM, freeing VRAM for model weights and KV cache.
//
// Returns nullptr on allocation failure (caller should treat as fatal).
inline ggml_backend_t create_snapshot_backend(ggml_backend_t compute_backend) {
    auto buft = ggml_backend_get_default_buffer_type(compute_backend);
    if (ggml_backend_buft_is_host(buft)) {
        // Unified memory — snapshots can stay on compute backend.
        return compute_backend;
    }
    // Discrete VRAM — allocate a CPU backend for snapshot storage.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "[snapshot] CPU backend init failed\n");
    }
    return cpu;
}

// Free the snapshot backend if it was separately allocated.
// Safe to call with nullptr. Does NOT free the compute backend.
inline void free_snapshot_backend(ggml_backend_t snap_backend,
                                  ggml_backend_t compute_backend) {
    if (snap_backend && snap_backend != compute_backend) {
        ggml_backend_free(snap_backend);
    }
}

}  // namespace dflash::common
