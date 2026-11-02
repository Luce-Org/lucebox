// Memory the GPU driver is holding for reuse.
//
// When a process frees GPU buffers that live in system RAM (GTT, and every
// allocation on an integrated GPU), the kernel's TTM layer keeps the pages in a
// pool instead of returning them. The next GPU allocation is served from that
// pool first, and the kernel shrinks it under memory pressure. /proc/meminfo
// does not count the pool in MemAvailable, so after one large run MemAvailable
// can read tens of GiB low even though that memory is there for the taking.
//
// The pool size itself is only visible to root (debugfs). This estimates it from
// what any process can read: RAM that /proc/meminfo attributes to nothing,
// minus the GPU buffers that are live right now (amdgpu sysfs). Measured on a
// Strix Halo + R9700 box it tracks the kernel counter within 0.8 GiB with the
// pool empty, full (61 GiB), and while 88 GiB of GPU buffers are live.
#pragma once

#include <cstdint>

namespace luce::common {

// 0 when nothing can be read (non-Linux, no amdgpu). The result already has a
// 1 GiB margin taken off for pages owned by other drivers.
uint64_t reclaimable_gpu_page_pool_bytes();

// Same estimate from caller-supplied text, for tests.
uint64_t reclaimable_gpu_page_pool_bytes(const char * meminfo_text, uint64_t live_gpu_host_bytes);

// The pool where a host-shared device's free memory (ggml_backend_dev_memory)
// leaves it out: with hugetlbfs pages configured that figure is the free huge
// pages alone. Elsewhere it already counts the pool, where it can be estimated,
// and this is 0.
uint64_t gpu_page_pool_bytes_beside_device_free();

// Same from caller-supplied text, for tests.
uint64_t gpu_page_pool_bytes_beside_device_free(const char * meminfo_text, uint64_t live_gpu_host_bytes);

} // namespace luce::common
