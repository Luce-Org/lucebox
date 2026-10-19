// Shared guard for the CPU-backend KvFlashPager tests that page chunks out.
// On GPU builds the pager's host backing is pinned (cudaMallocHost), which
// fails on a runner without a driver or device; those cases skip there and
// run on the GPU runners.
#pragma once
#include "../../src/common/kvflash_pager.h"

namespace luce::common {

inline bool pager_host_backing_available() {
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

}  // namespace luce::common

#define SKIP_WITHOUT_PAGER_HOST_BACKING() \
    do { if (!luce::common::pager_host_backing_available()) SKIP("pinned host memory unavailable (no GPU runtime)"); } while (0)
