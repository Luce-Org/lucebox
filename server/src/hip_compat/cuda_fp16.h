#pragma once

// Compatibility include for vendored ggml sources compiled through HIP.
// Some submodule snapshots include <cuda_fp16.h> in shared CUDA/HIP sources;
// on AMD HIP builds that must resolve to hip_fp16 instead of the CUDA SDK.

#include <hip/hip_fp16.h>
