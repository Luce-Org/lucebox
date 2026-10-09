#include "host_stream_flag.h"

#if defined(LUCE_BACKEND_HIP)
#include <hip/hip_runtime.h>
#endif

namespace luce::common {

#if defined(LUCE_BACKEND_HIP)

HostStreamFlag::~HostStreamFlag() {
    if (host_) (void) hipHostFree(host_);
}

bool HostStreamFlag::init() {
    if (host_) return true;
    uint32_t * host = nullptr;
    if (hipHostMalloc((void **) &host, 64, hipHostMallocDefault) != hipSuccess) return false;
    *host = 0;
    uint32_t * dev = nullptr;
    if (hipHostGetDevicePointer((void **) &dev, host, 0) != hipSuccess) {
        (void) hipHostFree(host);
        return false;
    }
    host_ = host;
    dev_ = dev;
    return true;
}

bool HostStreamFlag::enqueue_wait(void * stream, uint32_t value) const {
    return dev_ && hipStreamWaitValue32((hipStream_t) stream, dev_, value,
                                        hipStreamWaitValueEq, 0xffffffffu) == hipSuccess;
}

void HostStreamFlag::signal(uint32_t value) const {
    if (host_) __atomic_store_n(host_, value, __ATOMIC_RELEASE);
}

bool HostStreamFlag::copy_to_device(void * stream, void * dst, const void * src, size_t bytes) {
    return hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, (hipStream_t) stream) == hipSuccess;
}

#else

HostStreamFlag::~HostStreamFlag() = default;
bool HostStreamFlag::init() { return false; }
bool HostStreamFlag::enqueue_wait(void *, uint32_t) const { return false; }
void HostStreamFlag::signal(uint32_t) const {}
bool HostStreamFlag::copy_to_device(void *, void *, const void *, size_t) { return false; }

#endif

}  // namespace luce::common
