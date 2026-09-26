// Whether the loaded ggml GPU backend allocates compute scratch from its VMM
// pool — the pool whose cuMemSetAccess failed under skip-park fragmentation
// (placement/skip_park_guard.h).

#pragma once

namespace luce::common {

// Reads the GPU backend registry's feature list: ggml-cuda (also compiled as
// ggml-hip) reports NO_VMM when built without the VMM pool, which is the HIP
// default (GGML_HIP_NO_VMM=ON). Returns true when no GPU backend can be
// inspected, so an unknown build keeps the guard.
bool gpu_backend_uses_vmm_pool();

}  // namespace luce::common
