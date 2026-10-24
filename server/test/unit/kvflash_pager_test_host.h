// Shared setup for the CPU-backend KvFlashPager tests. The cache tensors live
// in host memory, so no case here moves bytes over a GPU stream; the only GPU
// dependency is the pager's pinned host backing (cudaMallocHost), which fails
// on a runner with a stub driver or no device. TestPager falls back to
// pageable host memory there, so every logic case (serde, truncate, pins,
// scores) runs on hosted CI exactly as on a GPU runner.
#pragma once
#include "../../src/common/kvflash_pager.h"

namespace luce::common {

inline bool pager_pinned_host_available() {
#ifdef KVFLASH_HAS_ASYNC_DMA
    void * p = nullptr;
    if (cudaMallocHost(&p, 256) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    cudaFreeHost(p);
#endif
    return true;
}

struct TestPager : KvFlashPager {
    TestPager() { set_pinned_host_backing(pager_pinned_host_available()); }
};

}  // namespace luce::common
