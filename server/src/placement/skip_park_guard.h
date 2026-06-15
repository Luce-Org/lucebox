// Footprint-aware guard: downgrade --prefill-skip-park on <32GB GPUs at max_ctx>65536.
#pragma once
#include <cstddef>

namespace dflash::common {

// Returns false only when dual-residency is unsafe (VMM VA-fragmentation risk).
inline bool skip_park_allowed(bool requested, size_t total_vram_bytes, int max_ctx) {
    return requested && (total_vram_bytes >= 32ull*1024*1024*1024 || max_ctx <= 65536);
}

}  // namespace dflash::common
