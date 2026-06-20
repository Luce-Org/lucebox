// KVFlash placement KV-reservation rule (architecture-agnostic, header-only).
//
// Any MoE / weight-offload backend that places experts against a VRAM budget
// must decide how much KV to reserve.  Reserving for `max_ctx` forces experts
// cold at high max_ctx even when KVFlash bounds the *resident* KV to a fixed
// pool.  This helper centralises the rule so every backend (qwen35moe today,
// DeepSeek-V4 / future MoE next) inherits the "pool bounds the expert-placement
// cliff" win without re-deriving the byte math.
#pragma once

#include <cstdint>

namespace dflash::common {

struct KvfPlacementDecision {
    uint64_t kv_total       = 0;      // bytes to reserve for the KV cache
    int      kv_ctx         = 0;      // tokens the reservation covers (pool or max_ctx)
    bool     all_hot_full_kv = false; // would ALL experts be hot with the FULL max_ctx KV?
    bool     pool_reduced   = false;  // did we reserve for the pool instead of max_ctx?
};

// Decide the KV reservation for VRAM-budget expert placement.
//
// kvf_pool: resident KVFlash pool in tokens (0 = KVFlash inactive).
// all_hot_full_kv reports whether the full max_ctx KV already fits all experts
//   hot — i.e. KVFlash is redundant; the caller's gate uses it to disable the
//   pool when unneeded (so a pool that is *itself* keeping experts hot is never
//   disabled).
// When KVFlash is active AND the full reservation would force experts cold, the
//   reservation is reduced to the pool so experts stay hot.
inline KvfPlacementDecision kvflash_placement_decision(
    uint64_t kv_bytes_per_tok, int max_ctx, int kvf_pool,
    uint64_t gpu_total, uint64_t core_bytes, uint64_t total_expert_bytes,
    uint64_t warm_bytes, uint64_t safety_bytes, uint64_t draft_bytes)
{
    KvfPlacementDecision d;
    const uint64_t kv_full = kv_bytes_per_tok * (uint64_t)max_ctx;
    const uint64_t fixed   = core_bytes + warm_bytes + safety_bytes + draft_bytes;

    uint64_t eb_full = 0;
    if (gpu_total > fixed + kv_full) eb_full = gpu_total - fixed - kv_full;
    d.all_hot_full_kv = (eb_full >= total_expert_bytes);

    d.kv_ctx   = max_ctx;
    d.kv_total = kv_full;
    if (kvf_pool > 0 && kvf_pool < max_ctx && !d.all_hot_full_kv) {
        d.kv_ctx       = kvf_pool;
        d.kv_total     = kv_bytes_per_tok * (uint64_t)kvf_pool;
        d.pool_reduced = true;
    }
    return d;
}

} // namespace dflash::common
