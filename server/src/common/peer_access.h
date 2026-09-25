// CUDA peer-access helpers for multi-GPU inference.
//
// Provides enable_peer_access_pair(), cross_device_peer_memcpy_ok(), and
// copy_peer_async() — used by DraftFeatureMirror and the speculative-decode
// loop to move data between target and draft GPUs.
//
// Global state (opt-in flag + cache) is accessed through free functions.
// Long-term these should move into a runtime context object.

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "gpu_runtime_compat.h"

namespace luce::common {

// ── global state ────────────────────────────────────────────────
// Set from argv: opt into cudaMemcpyPeerAsync for cross-device copies.
extern bool g_peer_access_opt_in;
extern std::unordered_map<std::uint64_t, bool> g_peer_pair_ok_cache;

// ── functions ───────────────────────────────────────────────────
bool enable_peer_access_one_way(int device, int peer);
bool enable_peer_access_pair(int a, int b);
bool cross_device_peer_memcpy_ok(int src_device, int dst_device);

// A per-thread, per-device non-blocking stream for helper copies and
// conversions. Work on the legacy default stream and cudaDeviceSynchronize()
// interact with every stream of the process on HIP, so they fail while
// another thread captures a graph (several models served by one process each
// capture on their own streams). Callers synchronize this stream instead.
cudaStream_t luce_copy_stream(int device);
bool luce_copy_stream_sync(int device);

// Copy `bytes` from src_device memory to dst_device memory and wait for it.
// Uses P2P when available and opted-in, otherwise falls back to host staging.
// Without a stream the copy runs on luce_copy_stream(dst_device).
bool copy_peer_async(void * dst, int dst_device,
                     const void * src, int src_device,
                     size_t bytes,
                     cudaStream_t stream = nullptr);

}  // namespace luce::common
