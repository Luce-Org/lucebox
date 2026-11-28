// Footprint-aware guard: downgrade --prefill-skip-park on <32GB-class GPUs at max_ctx>65536.
#pragma once
#include <cstddef>

namespace luce::common {

// Real 32 GiB GPUs (e.g., R9700: 32624 MiB, RTX 5090: 32607 MiB) report ~31.8 GiB
// totalGlobalMem after firmware/display carveout, so use 30 GiB as the 32GB-class floor.
inline constexpr size_t kSkipParkBigGpuMinBytes = 30ull * 1024 * 1024 * 1024;

// Returns false only when dual-residency is unsafe (VMM VA-fragmentation risk).
inline bool skip_park_allowed(bool requested, size_t total_vram_bytes, int max_ctx) {
    return requested && (total_vram_bytes >= kSkipParkBigGpuMinBytes || max_ctx <= 65536);
}

}  // namespace luce::common
