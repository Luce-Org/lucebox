#pragma once

#if defined(DFLASH27B_BACKEND_HIP)
#if defined(__HIPCC__)
#include "../deps/llama.cpp/ggml/src/ggml-cuda/vendors/hip.h"
using __nv_bfloat16 = __hip_bfloat16;
#ifndef cudaEventCreate
#define cudaEventCreate hipEventCreate
#endif
#ifndef cudaEventElapsedTime
#define cudaEventElapsedTime hipEventElapsedTime
#endif
// HIP has no masked ballot. __ballot(p) ballots every active lane of the
// wavefront and returns a 64-bit mask (wavefronts are 64 lanes wide on
// GCN/CDNA), whereas CUDA's __ballot_sync(mask, p) returns a 32-bit mask over
// only the lanes in `mask`. Two porting hazards follow:
//   1. Width: the result must be stored in a 64-bit sink on wave64 archs, or
//      lanes 32..63 are truncated away. We cast to unsigned long long so the
//      full width is explicit at the shim boundary; call sites that assign to a
//      32-bit type still truncate and are only correct on wave32 (e.g. the
//      rocWMMA flashprefill kernel, which is gated wave32-only).
//   2. Mask: the CUDA active-lane mask is dropped. That is only equivalent when
//      every lane in `mask` is active, which holds at all current call sites
//      (they pass a full 0xffffffff mask).
#ifndef __ballot_sync
#define __ballot_sync(mask, predicate) \
    (static_cast<unsigned long long>(__ballot(predicate)))
#endif
#else
#include <hip/hip_runtime.h>

#define cudaEvent_t hipEvent_t
#define cudaStream_t hipStream_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetDevice hipGetDevice
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#endif
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
