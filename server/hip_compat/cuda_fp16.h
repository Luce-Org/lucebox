// HIP compatibility shim for <cuda_fp16.h>
#pragma once
#include <hip/hip_fp16.h>

// __half is the same name in HIP — no alias needed.
// Intrinsics like __half2float, __float2half, __hadd, etc. are available directly.
