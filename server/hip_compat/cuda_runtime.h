// HIP compatibility shim: maps <cuda_runtime.h> to HIP equivalents.
// Included transparently when building with -I hip_compat on ROCm.
#pragma once

// hip/hip_runtime.h requires exactly one of __HIP_PLATFORM_AMD__ or
// __HIP_PLATFORM_NVIDIA__ to be defined. hipcc sets it automatically;
// g++ (used for plain CXX sources in the dflash build) does not.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#  define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Type aliases
using cudaStream_t          = hipStream_t;
using cudaEvent_t           = hipEvent_t;
using cudaError_t           = hipError_t;
using cudaMemcpyKind        = hipMemcpyKind;
using cudaDeviceProp        = hipDeviceProp_t;

// Memcpy kind constants
#define cudaMemcpyHostToHost        hipMemcpyHostToHost
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice
#define cudaMemcpyDefault           hipMemcpyDefault

// Error codes
#define cudaSuccess                 hipSuccess
#define cudaErrorInvalidValue       hipErrorInvalidValue

// Memory functions
#define cudaMalloc                  hipMalloc
#define cudaMallocHost              hipHostMalloc
#define cudaFree                    hipFree
#define cudaFreeHost                hipHostFree
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemcpy2DAsync           hipMemcpy2DAsync
#define cudaMemcpyPeerAsync         hipMemcpyPeerAsync
#define cudaMemset                  hipMemset
#define cudaMemsetAsync             hipMemsetAsync

// Stream functions
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaStreamDefault           hipStreamDefault
#define cudaStreamNonBlocking       hipStreamNonBlocking

// Device functions
#define cudaGetDevice               hipGetDevice
#define cudaSetDevice               hipSetDevice
#define cudaDeviceSynchronize       hipDeviceSynchronize
#define cudaGetDeviceProperties     hipGetDeviceProperties
#define cudaDeviceReset             hipDeviceReset

// Event functions
#define cudaEventCreate             hipEventCreate
#define cudaEventDestroy            hipEventDestroy
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventElapsedTime        hipEventElapsedTime
#define cudaEventCreateWithFlags    hipEventCreateWithFlags
#define cudaEventDisableTiming      hipEventDisableTiming

// Kernel attribute
#define cudaFuncSetAttribute        hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// Error checking
#define cudaGetLastError            hipGetLastError
#define cudaGetErrorString          hipGetErrorString

// Launch bounds
#define __launch_bounds__           __launch_bounds__

// Stream capture status (added CUDA 10.0 — ROCm compat headers may omit this)
#define cudaStreamCaptureStatus             hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone         hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive       hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated  hipStreamCaptureStatusInvalidated
#define cudaStreamIsCapturing               hipStreamIsCapturing

// Peer device access
#define cudaDeviceCanAccessPeer             hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess          hipDeviceEnablePeerAccess
#define cudaErrorPeerAccessAlreadyEnabled   hipErrorPeerAccessAlreadyEnabled

// Device count
#define cudaGetDeviceCount                  hipGetDeviceCount
