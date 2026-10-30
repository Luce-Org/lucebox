#include "gpu_page_pool.h"

// The estimate lives with the RAM check for managed allocations on integrated
// GPUs, which needs it inside ggml-cuda and cannot depend on server code.
#include "../../deps/llama.cpp/ggml/src/ggml-cuda/uma-budget.h"

#include <string>

namespace luce::common {

uint64_t reclaimable_gpu_page_pool_bytes(const char * meminfo_text, uint64_t live_gpu_host_bytes) {
    return ggml_cuda_uma_page_pool_bytes(meminfo_text, live_gpu_host_bytes);
}

uint64_t reclaimable_gpu_page_pool_bytes() {
#if defined(__linux__)
    std::string meminfo;
    uint64_t live = 0;
    if (!ggml_cuda_uma_read_meminfo(&meminfo) || !ggml_cuda_uma_live_gpu_host_bytes(&live)) return 0;
    return ggml_cuda_uma_page_pool_bytes(meminfo.c_str(), live);
#else
    return 0;
#endif
}

} // namespace luce::common
