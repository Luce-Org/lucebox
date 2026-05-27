// HIP compatibility shim for <mma.h> (NVIDIA WMMA).
//
// Phase 1: empty — flashprefill_kernels.cu is excluded from the Phase 1 build
//          (DFLASH27B_HAVE_FLASHPREFILL not defined), so this file is never reached.
//
// Phase 2: replace nvcuda::wmma with rocwmma. Add:
//   #include <rocwmma/rocwmma.hpp>
//   namespace nvcuda { namespace wmma = rocwmma; }  // approximate alias
//   Then fix the accumulator fragment register layout in sparse_flash_forward_kernel_bf16
//   (lines 408-443 of flashprefill_kernels.cu) to match AMD's m16n16k16 layout.
//
// NOTE: a namespace alias is not sufficient — the fragment register layouts differ
// between NVIDIA sm_80 and AMD gfx1151. The manual row/col extraction code in
// kernel 4 must be rewritten per the rocWMMA accumulator layout docs.
#pragma once
