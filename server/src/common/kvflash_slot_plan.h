// The pure half of the slot pager: slice geometry and host-budget admission with its ledger. No ggml, no device
// code, so it compiles and tests without the pager. The pager over a slice is kvflash_slot_pager.h; the step's pager
// work is kvflash_step_driver.h.

#pragma once

#include "kvflash_config.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace luce::common {

// ── Slice geometry ─────────────────────────────────────────────────────

struct KvFlashSlice {
    int     slot = 0;
    int     block_lo = 0, block_hi = 0;   // [lo, hi) in arena blocks (block = chunk = chunk_tokens)
    int64_t row_lo = 0, row_hi = 0;       // [lo, hi) in arena rows
};

inline KvFlashSlice kvflash_slice(int slot, int quota_chunks, int chunk_tokens = kKvFlashChunkTokens) {
    KvFlashSlice s;
    s.slot = slot;
    s.block_lo = slot * quota_chunks;
    s.block_hi = s.block_lo + quota_chunks;
    s.row_lo = (int64_t)s.block_lo * chunk_tokens;
    s.row_hi = (int64_t)s.block_hi * chunk_tokens;
    return s;
}

// Table entries per chunk when the kernel's physical block is `block_tokens`
// (the WMMA kernel needs 16-token blocks, so a 64-token chunk is 4 entries).
// 0 when the block does not divide the chunk.
inline int kvflash_table_entries(int chunk_tokens, int block_tokens) {
    return block_tokens > 0 && chunk_tokens % block_tokens == 0
               ? chunk_tokens / block_tokens : 0;
}

// Rows of the shared KV allocation: every slice plus one scratch block.
inline int64_t kvflash_arena_rows(int n_slots, int quota_chunks, int chunk_tokens = kKvFlashChunkTokens) {
    return ((int64_t)n_slots * quota_chunks + 1) * chunk_tokens;
}

// "auto" sizes ONE pool from the free memory; the arena holds N of them, so each slot
// gets its share of (free - reserve). The reserve (compute buffers) is held once.
inline KvFlashAutoBudget kvflash_budget_per_slot(KvFlashAutoBudget b, int n_slots) {
    if (n_slots > 1 && b.free_bytes > b.reserve_bytes)
        b.free_bytes = b.reserve_bytes + (b.free_bytes - b.reserve_bytes) / n_slots;
    return b;
}

// The arena a KVFlash slot engine allocates: `rows` is ctx_alloc, `pool_tokens`
// the N x Q x chunk the pool is derived as. --kv-pool-tokens is never a size
// here; given and different, it is refused with `error` set (rows 0).
// Layout: [N slices][N x chain_scratch_stride chain scratch rows][one dead-row block].
// `scratch_base` is the first chain scratch row (the pool's end, 16-aligned) and `dead_row`
// the first row of the dead block; with no chain stride they coincide at the pool's end.
struct KvFlashArenaSizing { int64_t rows = 0, pool_tokens = 0, scratch_base = 0, dead_row = 0; std::string error; };
inline KvFlashArenaSizing kvflash_arena_sizing(int n_slots, int quota_chunks, int64_t kv_pool_tokens,
                                               int chunk_tokens = kKvFlashChunkTokens, int chain_scratch_stride = 0) {
    KvFlashArenaSizing r;
    if (n_slots < 1 || quota_chunks < 1) { r.error = "KVFlash arena needs at least one slot and one chunk of quota"; return r; }
    r.pool_tokens = (int64_t)n_slots * quota_chunks * chunk_tokens;
    if (kv_pool_tokens != 0 && kv_pool_tokens != r.pool_tokens) {
        r.error = "--kv-pool-tokens " + std::to_string(kv_pool_tokens) + " conflicts with KVFlash: the arena is " +
                  std::to_string(n_slots) + " slots x " + std::to_string(quota_chunks) + " chunks x " +
                  std::to_string(chunk_tokens) + " = " + std::to_string(r.pool_tokens) + " tokens; omit the flag";
        r.pool_tokens = 0;
        return r;
    }
    if (chain_scratch_stride < 0 || (chain_scratch_stride > 0 && (chain_scratch_stride % 16 != 0 || r.pool_tokens % 16 != 0))) {
        r.error = "KVFlash chain scratch stride " + std::to_string(chain_scratch_stride) + " must be a non-negative multiple of 16";
        r.pool_tokens = 0;
        return r;
    }
    r.scratch_base = r.pool_tokens;
    r.dead_row = r.pool_tokens + (int64_t)n_slots * chain_scratch_stride;
    r.rows = kvflash_arena_rows(n_slots, quota_chunks, chunk_tokens) + (int64_t)n_slots * chain_scratch_stride;
    if (r.rows > INT32_MAX) { r.error = "KVFlash arena exceeds INT32_MAX rows"; r.rows = r.pool_tokens = r.scratch_base = r.dead_row = 0; }
    return r;
}

// Headroom: page-outs per step are capped at `headroom_blocks`, so the
// slot's usable quota is Q - headroom and a page-in never waits on a block
// freed in the same step. The usable quota must still leave an evictable block.
inline bool kvflash_headroom_ok(int quota_chunks, int headroom_blocks, const KvFlashConfig & cfg = {}) {
    return headroom_blocks >= 0 &&
           (quota_chunks - headroom_blocks) * cfg.chunk_tokens >= kvflash_min_pool_tokens(cfg);
}

// A quota must leave an evictable block (the pager's own floor).
inline bool kvflash_quota_ok(int quota_chunks, const KvFlashConfig & cfg = {}) {
    return quota_chunks * cfg.chunk_tokens >= kvflash_min_pool_tokens(cfg);
}

// Quota in chunks for a pool of `pool_tokens`; 0 when it is not a whole number of chunks.
inline int kvflash_quota_chunks(int pool_tokens, int chunk_tokens = kKvFlashChunkTokens) {
    return pool_tokens > 0 && pool_tokens % chunk_tokens == 0 ? pool_tokens / chunk_tokens : 0;
}

// ── Admission ───────────────────────────────────

// HostTooSmall is permanent: the request alone exceeds the whole host budget,
// so queueing it as BusyHost would wait forever.
enum class KvFlashAdmit { Ok, BusySlot, BusyHost, PromptTooLong, BadQuota, HostTooSmall };

struct KvFlashAdmitInput {
    bool    slot_free = false;               // slot, its slice and its state slab are all free
    int     quota_chunks = 0, headroom_blocks = 0;   // usable = quota - headroom must stay above the pager floor
    int64_t prompt_tokens = 0, max_new_tokens = 0, max_ctx = 0;
    int     chunk_tokens = kKvFlashChunkTokens;
    int64_t chunk_bytes = 0;                 // host backing per chunk
    int64_t ladder_bytes = 0;                // rungs x live state this slot will hold
    int64_t host_used_bytes = 0, host_budget_bytes = 0;
};

// Host bytes this request reserves: backing for every chunk of prompt + new
// tokens plus its ladder. ponytail: backing is the upper bound (every chunk evicted at least
// once over a long run), not chunks - Q; the ceiling is admitting fewer
// sessions than fit when most never evict.
inline int64_t kvflash_host_need_bytes(const KvFlashAdmitInput & in) {
    const int64_t tokens = in.prompt_tokens + in.max_new_tokens;
    const int64_t chunks = (tokens + in.chunk_tokens - 1) / in.chunk_tokens;
    return chunks * in.chunk_bytes + in.ladder_bytes;
}

// The ladder charge of one request (KvFlashAdmitInput::ladder_bytes): the R state snapshot of a recall turn and
// the rungs of a sticky session (checkpoint, trunk and the third, kvflash_slot_turn.h rule v5), one slab each.
constexpr int kKvFlashLadderRungs = 3;
inline int64_t kvflash_ladder_bytes(int64_t slab_bytes, bool recall, bool sticky) {
    return slab_bytes * ((recall ? 1 : 0) + (sticky ? kKvFlashLadderRungs : 0));
}

// A permanent refusal outranks a transient busy. The host bound does not
// depend on the headroom: it counts every chunk, resident or not.
inline KvFlashAdmit kvflash_admit(const KvFlashAdmitInput & in) {
    KvFlashConfig cfg; cfg.chunk_tokens = in.chunk_tokens;
    if (!kvflash_headroom_ok(in.quota_chunks, in.headroom_blocks, cfg)) return KvFlashAdmit::BadQuota;
    if (in.prompt_tokens > in.max_ctx) return KvFlashAdmit::PromptTooLong;
    const int64_t alone = kvflash_host_need_bytes(in);
    if (alone > in.host_budget_bytes) return KvFlashAdmit::HostTooSmall;
    if (!in.slot_free) return KvFlashAdmit::BusySlot;
    if (in.host_used_bytes + alone > in.host_budget_bytes) return KvFlashAdmit::BusyHost;
    return KvFlashAdmit::Ok;
}

// Host RESERVATIONS, apart from allocated bytes: admit takes a request's
// worst-case backing (kvflash_host_need_bytes), release gives it back and is
// called on EVERY exit (finish, cancel, error, abort_step). Not thread-safe:
// the engine thread owns it.
class KvFlashHostLedger {
public:
    explicit KvFlashHostLedger(int64_t budget_bytes) : budget_(budget_bytes) {}

    // `in.host_*` are overwritten from the ledger.
    // A request id already holding a reservation is refused (BusySlot), never charged twice.
    KvFlashAdmit admit(uint64_t request_id, KvFlashAdmitInput in) {
        if (reserved_.count(request_id)) return KvFlashAdmit::BusySlot;
        in.host_budget_bytes = budget_;
        in.host_used_bytes = used_;
        const KvFlashAdmit v = kvflash_admit(in);
        if (v == KvFlashAdmit::Ok) {
            const int64_t need = kvflash_host_need_bytes(in);
            reserved_[request_id] = need;
            used_ += need;
        }
        return v;
    }
    // What admit() would answer for a request that holds no reservation yet, charging nothing.
    KvFlashAdmit would_admit_new(KvFlashAdmitInput in) const {
        in.host_budget_bytes = budget_;
        in.host_used_bytes = used_;
        return kvflash_admit(in);
    }
    void release(uint64_t request_id) {   // idempotent
        const auto it = reserved_.find(request_id);
        if (it == reserved_.end()) return;
        used_ -= it->second;
        reserved_.erase(it);
    }
    int64_t used_bytes() const { return used_; }
    int64_t budget_bytes() const { return budget_; }
    int64_t reserved_of(uint64_t request_id) const { const auto it = reserved_.find(request_id); return it == reserved_.end() ? 0 : it->second; }

protected:   // the test fixture counts reservations
    int64_t budget_, used_ = 0;
    std::unordered_map<uint64_t, int64_t> reserved_;
};

} // namespace luce::common
