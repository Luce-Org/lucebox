// Cross-GPU device memory copy via pinned host staging (no CUDA peer access).

#include "internal.h"

#include <cuda_runtime.h>

#include <mutex>

namespace {

std::mutex g_pin_mu;
void *     g_pin     = nullptr;
size_t     g_pin_cap = 0;

bool synchronize_on_device(int dev, cudaStream_t stream) {
    if (cudaSetDevice(dev) != cudaSuccess) {
        return false;
    }
    if (stream) {
        return cudaStreamSynchronize(stream) == cudaSuccess;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

} // namespace

bool dflash_cuda_copy_between_devices(int src_dev, const void * src,
                                      int dst_dev, void * dst, size_t nbytes,
                                      cudaStream_t src_stream,
                                      cudaStream_t dst_stream) {
    if (nbytes == 0) {
        return true;
    }
    if (src_dev == dst_dev) {
        cudaStream_t stream = dst_stream ? dst_stream : src_stream;
        cudaError_t err = cudaSetDevice(src_dev);
        if (err != cudaSuccess) {
            return false;
        }
        err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            return false;
        }
        if (stream) {
            return cudaStreamSynchronize(stream) == cudaSuccess;
        }
        return cudaDeviceSynchronize() == cudaSuccess;
    }

    std::lock_guard<std::mutex> lock(g_pin_mu);
    if (g_pin_cap < nbytes) {
        if (g_pin) {
            cudaFreeHost(g_pin);
            g_pin     = nullptr;
            g_pin_cap = 0;
        }
        cudaError_t err = cudaMallocHost(&g_pin, nbytes);
        if (err != cudaSuccess) {
            return false;
        }
        g_pin_cap = nbytes;
    }
    void * pin = g_pin;

    cudaError_t err = cudaSetDevice(src_dev);
    if (err != cudaSuccess) {
        return false;
    }
    cudaEvent_t d2h_done = nullptr;
    err = cudaEventCreateWithFlags(&d2h_done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaMemcpyAsync(pin, src, nbytes, cudaMemcpyDeviceToHost, src_stream);
    if (err != cudaSuccess) {
        cudaEventDestroy(d2h_done);
        return false;
    }
    err = cudaEventRecord(d2h_done, src_stream);
    if (err != cudaSuccess) {
        cudaEventDestroy(d2h_done);
        return false;
    }
    err = cudaSetDevice(dst_dev);
    if (err != cudaSuccess) {
        synchronize_on_device(src_dev, src_stream);
        cudaEventDestroy(d2h_done);
        return false;
    }
    err = cudaStreamWaitEvent(dst_stream, d2h_done, 0);
    if (err != cudaSuccess) {
        synchronize_on_device(src_dev, src_stream);
        cudaEventDestroy(d2h_done);
        return false;
    }
    err = cudaMemcpyAsync(dst, pin, nbytes, cudaMemcpyHostToDevice, dst_stream);
    if (err != cudaSuccess) {
        synchronize_on_device(dst_dev, dst_stream);
        cudaEventDestroy(d2h_done);
        return false;
    }
    if (!synchronize_on_device(dst_dev, dst_stream)) {
        cudaEventDestroy(d2h_done);
        return false;
    }
    err = cudaEventDestroy(d2h_done);
    if (err != cudaSuccess) {
        return false;
    }
    return true;
}
