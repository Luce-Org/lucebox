// KvFlashSlotPager: N KVFlash sessions on one GPU
// The KV rows stay ONE allocation of N x Q x 64 rows
// plus a scratch block; slot s owns rows [sQ*64, (s+1)Q*64) as a private free
// list. A slot pager is the existing KvFlashPager confined to its slice
// (KvFlashConfig::row_base), not a fork: victims, pins, truncate, serialize
// and reset are the single-session code, so they cannot cross a slice.
// The slice geometry, admission and step plan are kvflash_slot_plan.h; the arena
// stream shared by all slots is kvflash_arena.h.

#pragma once

#include "kvflash_pager.h"
#include "kvflash_slot_plan.h"

#include <vector>

namespace luce::common {

// ── The slot pager ─────────────────────────────────────────────────────

class KvFlashSlotPager : public KvFlashPager {
public:
    // Attach slot `slot` of `n_slots` with a quota of `quota_chunks` chunks.
    // cfg.pool_tokens and cfg.row_base are derived; the K/V tensors are the
    // shared arena (>= kvflash_arena_rows rows, or none for the no-tensor path).
    bool attach_slice(int slot, int n_slots, int quota_chunks, KvFlashConfig cfg,
                      const std::vector<ggml_tensor *> & attn_k,
                      const std::vector<ggml_tensor *> & attn_v) {
        if (slot < 0 || slot >= n_slots || !kvflash_quota_ok(quota_chunks, cfg)) return false;
        const KvFlashSlice sl = kvflash_slice(slot, quota_chunks, cfg.chunk_tokens);
        cfg.pool_tokens = quota_chunks * cfg.chunk_tokens;
        cfg.row_base = (int)sl.row_lo;
        // The scratch block must exist too, or the arena is not the one the
        // table was sized for.
        if (!attn_k.empty() && (int64_t)attn_k[0]->ne[1] < kvflash_arena_rows(n_slots, quota_chunks, cfg.chunk_tokens))
            return false;
        if (!attach(cfg, attn_k, attn_v)) return false;
        set_deferred_free(true);   // a paged-out block is reusable only after completion
        slice_ = sl;   // only after the attach took: a refusal leaves the old slice and pager consistent
        return true;
    }

    const KvFlashSlice & slice() const { return slice_; }

    // One block-table row of `n_entries` logical chunks: the arena block of a
    // resident chunk, -1 (null) for an evicted or not yet materialized one.
    // Only this slot's pager writes its column; entries never leave the slice.
    void fill_block_table(int32_t * dst, int n_entries) const {
        for (int c = 0; c < n_entries; c++) dst[c] = block_of(c);
    }

    // The same column for a kernel whose physical block is `block_tokens`:
    // chunk c with arena block B becomes the entries B*k + 0..k-1 (k =
    // kvflash_table_entries), four -1 when the chunk is not resident. `dst`
    // holds n_chunks*k entries. A block size that does not divide the chunk
    // writes nothing. block_tokens == chunk_tokens is the 2-argument form.
    void fill_block_table(int32_t * dst, int n_chunks, int block_tokens) const {
        const int k = kvflash_table_entries(chunk_tokens(), block_tokens);
        for (int c = 0; c < n_chunks; c++) {
            const int b = block_of(c);
            for (int j = 0; j < k; j++) dst[(size_t)c * k + j] = b < 0 ? -1 : b * k + j;
        }
    }

private:
    // The slice descriptor and the pager must change together; attach_slice
    // is the only way to attach a slot pager.
    using KvFlashPager::attach;
    KvFlashSlice slice_;
};

}  // namespace luce::common
