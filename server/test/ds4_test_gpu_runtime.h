// One place the ds4 kernel tests pick up a GPU runtime, so they build for CUDA as well as HIP.
//
// The kernels themselves were always backend-general: rocmfp{2,3}_mix.cuh declares its
// launchers with `cudaStream_t`, and ggml's vendors/hip.h maps the cuda* spellings onto hip*
// when GGML_USE_HIP is set. Only the TESTS were HIP-only -- they included <hip/hip_runtime.h>
// directly and called hipDeviceSynchronize/hipMalloc/hipMemGetInfo -- which is why the
// qtype-105/106 numerics were verified on gfx1151 and nowhere else, and why a failure on a
// CUDA box could not be attributed between artifact, loader and kernel.
//
// Selection mirrors common.cuh exactly rather than inventing a second rule, so the mapping has
// one definition. Tests then use the cuda* spelling throughout and compile for both.

#pragma once

#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#ifndef cudaErrorNoDevice
#define cudaErrorNoDevice hipErrorNoDevice
#endif
#elif defined(GGML_USE_MUSA)
#include "vendors/musa.h"
#else
#include "vendors/cuda.h"
#endif
