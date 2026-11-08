// KvFlashPager — KVFlash core: a bounded resident pool for the
// full-attention KV cache (see optimizations/kvflash/).
//
// Lookahead-sparse-attention-style (FlashMemory, arXiv 2606.09079)
// decode-time KV residency for the qwen35 target: the cache tensors are
// allocated at POOL size (a fraction of the logical context), and this
// class owns the mapping from logical token positions to physical pool
// slots. Chunks (64 logical tokens) that fall cold are paged out to a
// host backing store and their slots are reused; paged-out chunks remain
// recallable bit-exact. GPU footprint is a hard O(pool) bound regardless
// of logical context length.
//
// Policy-agnostic by design: with no scorer, eviction is LRU over
// unprotected chunks (recency-only memory). A KvFlashScorer plugged into
// `score_hook` upgrades eviction and reselect() to relevance-driven
// residency; with pflash enabled, its drafter attaches automatically
// (KvFlashDrafterScorer) and recalls cold context the generation needs.
//
// Correctness notes (why relocating rows is legal):
//  * RoPE is baked into K rows at write time from the `positions` input,
//    so a row's physical slot is semantically irrelevant.
//  * Attention runs over the whole pool with a slot-validity mask
//    (resident = 0, free/paged-out = -inf). The mask must be re-uploaded
//    before EVERY compute: input tensors live in the gallocr compute
//    buffer whose regions are reused during graph execution.
//  * Freed slots are additionally zeroed (defense in depth; a zero K row
//    contributes exp(-max) ~ 0, the same assumption the production
//    stride-256 padded span relies on in maskless mode).
//  * The FWHT K-rotation and KV quantization operate per-row; page-out /
//    page-in moves raw quantized bytes and is therefore bit-exact.
//
// Scope: full-attention layers only. DeltaNet/conv recurrent state is
// fixed-size, position-dependent in-place state and is never paged.
//
// Async DMA (KVFLASH_HAS_ASYNC_DMA):
//  When built with a CUDA or HIP backend, page-out (D2H) and page-in
//  (H2D) transfers run on a dedicated page_stream_ rather than blocking
//  the decode thread via ggml_backend_tensor_get/set.  Host backing
//  buffers are cudaMallocHost-pinned for maximum PCIe throughput.
//  alloc_span() synchronises page_stream_ only if a page-in was issued
//  (H2D must complete before the next attention forward).  Page-outs are
//  queued, but slot_for() does not hand out the block they freed until the
//  compute stream is ordered after them: a device-side wait if the backend
//  registered its stream (CUDA), otherwise a host wait.  So the nested
//  default-path allocation can sync after a page-out, and the new K/V write
//  cannot overtake the copy or the zeroing.  On the same stream, page-out D2H
//  always precedes the slot zero (cudaMemsetAsync), which in turn precedes any
//  page-in H2D, so host_data is coherent by the time it is read back.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "fnv.h"
#include "kvflash_pool_flag.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

// Async DMA via a dedicated copy stream — enabled when a GPU runtime is
// available.  gpu_runtime_compat.h maps cuda* → hip* on HIP builds and
// pulls in <cuda_runtime.h> on CUDA builds.
#if defined(LUCE_BACKEND_CUDA) || defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "gpu_runtime_compat.h"
#define KVFLASH_HAS_ASYNC_DMA 1
// ponytail: the device-side fence re-records one event per fence, which CUDA
// allows and HIP documents as undefined while an earlier record is pending. HIP
// builds therefore stay on the host wait (one stall per reused block after a
// page-out on AMD). Upgrade: an event per pending record, or a HIP-verified reuse.
#if !defined(LUCE_BACKEND_HIP) && !defined(GGML_USE_HIP)
#define KVFLASH_EVENT_FENCE 1
#endif
#endif

// serialize()/deserialize() write and read integers and floats in host byte
// order, so the blob is little-endian only on little-endian hosts. There is no
// byte-swapping encoder; refuse to build where the blob would not be portable.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "KvFlashPager blobs are written in host byte order and are defined as little-endian");
#endif

namespace luce::common {

// Largest multiple of chunk_tokens that is <= pos. Aligns a snapshot boundary
// (serialize(max_chunks)) to a whole number of pager chunks.
inline int floor_to_chunk(int pos, int chunk_tokens) {
    return (pos / chunk_tokens) * chunk_tokens;
}

struct KvFlashConfig {
    int chunk_tokens       = 64;  // logical tokens per page
    int pool_tokens        = 0;   // resident pool capacity (multiple of chunk_tokens)
    int sink_chunks        = 1;   // leading chunks never evicted (attention sinks)
    int tail_window_chunks = 4;   // trailing chunks never evicted (local window)
    int row_base           = 0;   // first row of this pool inside a shared KV allocation (slot slice)
};

struct KvFlashStats {
    int64_t page_outs  = 0;
    int64_t page_ins   = 0;
    int64_t host_bytes = 0;   // backing store currently held on host
    int64_t moved_bytes = 0;  // cumulative D2H+H2D traffic
    int64_t d2h_skipped = 0;  // page-outs whose D2H was skipped: host copy already equal to the block
};

class KvFlashPager {
public:
    // `attn_k` / `attn_v` are the per-full-attention-layer cache tensors,
    // each [head_dim, pool_tokens, n_head_kv]. All must share dims/types
    // within their K/V group.
    // Minimum pool for a config: sinks + trailing window stay resident
    // unconditionally, so at least 2 more chunks are required (1 evictable
    // victim + the partially filled append head) or eviction deadlocks and
    // slot_for() starts failing once the pool fills.
    static int min_pool_tokens(const KvFlashConfig & cfg) {
        return (cfg.sink_chunks + cfg.tail_window_chunks + 2) * cfg.chunk_tokens;
    }

    ~KvFlashPager() { cleanup_(); }

    bool attach(const KvFlashConfig & cfg,
                const std::vector<ggml_tensor *> & attn_k,
                const std::vector<ggml_tensor *> & attn_v) {
        if (cfg.pool_tokens <= 0 || cfg.pool_tokens % cfg.chunk_tokens != 0) return false;
        if (cfg.row_base < 0 || cfg.row_base % cfg.chunk_tokens != 0) return false;
        if (cfg.pool_tokens < min_pool_tokens(cfg)) {
            std::fprintf(stderr,
                "kvflash: pool %d < minimum %d (%d sink + %d tail chunks must "
                "leave an evictable block)\n",
                cfg.pool_tokens, min_pool_tokens(cfg),
                cfg.sink_chunks, cfg.tail_window_chunks);
            return false;
        }
        if (attn_k.size() != attn_v.size()) return false;
        // Reject a tensor that cannot hold the pool BEFORE touching any state,
        // so a failed attach leaves the previous attachment intact.
        const int64_t need_rows = (int64_t)cfg.row_base + cfg.pool_tokens;
        for (size_t l = 0; l < attn_k.size(); l++)
            if ((int64_t)attn_k[l]->ne[1] < need_rows || (int64_t)attn_v[l]->ne[1] < need_rows) return false;

        cleanup_();   // release pinned buffers + stream from any prior attach

        cfg_ = cfg;
        row_base_ = cfg.row_base;
        attn_k_ = attn_k;
        attn_v_ = attn_v;
        n_blocks_ = cfg.pool_tokens / cfg.chunk_tokens;
        if (!attn_k.empty()) {
            const ggml_tensor * K0 = attn_k[0];
            n_head_kv_ = (int)K0->ne[2];

            // Per-(tensor, head) contiguous segment of chunk_tokens rows.
            k_seg_bytes_ = (size_t)cfg.chunk_tokens * K0->nb[1];
            v_seg_bytes_ = (size_t)cfg.chunk_tokens * attn_v[0]->nb[1];
            chunk_bytes_ = (k_seg_bytes_ + v_seg_bytes_) * (size_t)n_head_kv_ * attn_k.size();
            zero_buf_.assign(std::max(k_seg_bytes_, v_seg_bytes_), 0);
        } else {
            n_head_kv_ = 0;
            k_seg_bytes_ = 0;
            v_seg_bytes_ = 0;
            chunk_bytes_ = 0;
            zero_buf_.clear();
        }

        free_blocks_.clear();
        pending_free_.clear();
        for (int b = n_blocks_ - 1; b >= 0; b--) free_blocks_.push_back(b);
        chunks_.clear();
        pinned_.clear();   // pins belong to the previous pool geometry
        stats_ = {};
        clock_ = 0;
        has_pending_page_in_ = false;

        generation_++;
#ifdef KVFLASH_HAS_ASYNC_DMA
        if (ext_page_stream_) {
            page_stream_ = ext_page_stream_;   // the arena's stream; never ours to destroy
        } else if (cudaStreamCreate(&page_stream_) != cudaSuccess) {
            std::fprintf(stderr, "[kvflash] cudaStreamCreate failed; paging "
                         "falls back to the (blocking) default stream\n");
            page_stream_ = nullptr;
        }
#ifdef KVFLASH_EVENT_FENCE
        if (cudaEventCreateWithFlags(&page_event_, cudaEventDisableTiming) != cudaSuccess) {
            page_event_ = nullptr;   // set_compute_stream() then keeps the host wait
            cudaGetLastError();
        }
#endif
#endif
        return true;
    }

    // Host backing is pinned (cudaMallocHost) on GPU builds. Pass false to use
    // pageable memory instead: slower DMA, but works where pinned allocation is
    // unavailable (hosted CI with a stub driver, CPU-backend tests). Set it
    // before the first page_out/deserialize; each chunk remembers how its own
    // buffer was allocated. No effect on builds without async DMA.
    void set_pinned_host_backing(bool pinned) { pinned_host_ = pinned; }

    // Optional: custom block hand-out order (e.g. shuffled placement in
    // relocation tests). `order[i]` = i-th block to hand out.
    // `order` holds ARENA block ids (block_of() space) of this pool, each at
    // most once; anything else is refused and the free list is left alone.
    bool set_block_order(const std::vector<int> & order) {
        std::vector<int> local;
        std::vector<char> seen((size_t)n_blocks_, 0);
        const int lo = row_base_ / cfg_.chunk_tokens;
        for (int b : order) {
            const int l = b - lo;
            if (l < 0 || l >= n_blocks_ || seen[(size_t)l]) return false;
            seen[(size_t)l] = 1;
            local.push_back(l);
        }
        free_blocks_.assign(local.rbegin(), local.rend());
        return true;
    }

    // Drop all mappings and host backing (new request / cache reset).
    // Cumulative stats are kept; the epoch advances so cached masks refill.
    void reset() {
#ifdef KVFLASH_HAS_ASYNC_DMA
        sync_page_stream_();
        for (auto & st : chunks_) free_host_buffer_(st);
#endif
        chunks_.clear();
        free_blocks_.clear();
        pending_free_.clear();
        generation_++;   // ops planned against the previous request are stale
        for (int b = n_blocks_ - 1; b >= 0; b--) free_blocks_.push_back(b);
        stats_.host_bytes = 0;
        cur_chunk_ = 0;
        epoch_++;
        has_pending_page_in_ = false;
        unpin_all();   // pins are per-request; the backend re-applies them
        // Drop any relevance scoring from the previous request: a stale
        // score_hook could otherwise select an in-flight chunk of a multi-chunk
        // prefill ubatch as an eviction victim. It is re-installed by the
        // decode-time reselect path, so pooled prefill falls back to LRU, where
        // the just-allocated (newest) chunks are never victims.
        score_hook = nullptr;
    }

    // Drop every logical position >= P: chunks entirely >= P give their pool
    // block back to the free list and lose their host copy and score; the
    // chunk containing P keeps its block (or host copy) with rows >= P zeroed.
    // Pins on dropped chunks are removed, so a pin range is clipped to P. The
    // result matches a pager that only ever saw P tokens. P <= 0 empties the
    // pager (pins kept clipped to nothing); P at or beyond the materialized
    // length is a no-op. Valid for any P because full-attention KV at a
    // position does not depend on later tokens.
    void truncate(int64_t P) {
        if (!attached()) return;
        if (P < 0) P = 0;
        if (P >= (int64_t)chunks_.size() * cfg_.chunk_tokens) return;
        generation_++;          // page-outs planned against the longer sequence are stale
        synchronize_paging();   // queued page DMA must not touch what is freed below
        const int keep = (int)((P + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens);
        for (int c = (int)chunks_.size() - 1; c >= keep; c--) {
            ChunkState & st = chunks_[(size_t)c];
            if (st.block >= 0) {
                zero_block(st.block);
                free_blocks_.push_back(st.block);
            }
            if (st.on_host && has_tensor_storage()) {
                release_host_(st);
                stats_.host_bytes -= (int64_t)chunk_bytes_;
            }
        }
        chunks_.resize((size_t)keep);
        if (pinned_.size() > (size_t)keep) pinned_.resize((size_t)keep);
        if (keep > 0 && P % cfg_.chunk_tokens != 0) {
            zero_chunk_rows_(keep - 1, (int)(P % cfg_.chunk_tokens));
        }
        cur_chunk_ = keep > 0 ? keep - 1 : 0;
        epoch_++;
        synchronize_paging();   // zeroing is queued on page_stream_
        if (keep <= n_blocks_) relayout_identity_(keep);
        else                   refill_after_truncate_();
        synchronize_paging();   // recalls are queued on page_stream_
    }

    // The retained prefix fits the pool: a pager that only ever saw it holds
    // every chunk resident in block c, and pool attention reduces over physical
    // slots in order, so the block map must match or the rounding differs. Move
    // every retained chunk to block c (resident ones from wherever the history
    // left them, host-only ones from their host copy), then rebuild the free
    // stack so later appends get keep, keep+1, ... Moves go through a host
    // scratch buffer, one chunk at a time, in an order that never overwrites a
    // block still holding a chunk to move (a cycle parks one chunk first).
    void relayout_identity_(int keep) {
        std::vector<int> occ((size_t)n_blocks_, -1);   // block -> retained resident chunk
        for (int c = 0; c < keep; c++)
            if (chunks_[(size_t)c].block >= 0) occ[(size_t)chunks_[(size_t)c].block] = c;
        std::vector<int> vacated;                      // blocks >= keep that held a retained chunk
        for (int b = keep; b < n_blocks_; b++)
            if (occ[(size_t)b] >= 0) vacated.push_back(b);
        std::vector<uint8_t> buf, held;
        if (has_tensor_storage()) { buf.resize(chunk_bytes_); held.resize(chunk_bytes_); }
        auto move_to = [&](int c) {      // resident chunk c -> block c (free), via buf
            ChunkState & st = chunks_[(size_t)c];
            copy_block_(st.block, buf.data(), /*to_host=*/true);
            synchronize_paging();
            copy_block_(c, buf.data(), /*to_host=*/false);
            synchronize_paging();
            occ[(size_t)st.block] = -1;
            st.block = c;
            occ[(size_t)c] = c;
        };
        auto movable = [&](int d) {
            const int b = chunks_[(size_t)d].block;
            return b >= 0 && b != d && occ[(size_t)d] < 0;
        };
        for (;;) {
            bool progress = false, pending = false;
            for (int c = 0; c < keep; c++) {
                const int b = chunks_[(size_t)c].block;
                if (b < 0 || b == c) continue;
                pending = true;
                if (movable(c)) { move_to(c); progress = true; }
            }
            if (progress) continue;
            if (!pending) break;
            // Only cycles remain: park one chunk's bytes, which frees its block.
            int c = 0;
            while (chunks_[(size_t)c].block < 0 || chunks_[(size_t)c].block == c) c++;
            ChunkState & st = chunks_[(size_t)c];
            copy_block_(st.block, held.data(), /*to_host=*/true);
            synchronize_paging();
            occ[(size_t)st.block] = -1;
            st.block = -1;
            for (bool again = true; again;) {   // unwind the cycle into the freed block
                again = false;
                for (int d = 0; d < keep; d++)
                    if (movable(d)) { move_to(d); again = true; }
            }
            copy_block_(c, held.data(), /*to_host=*/false);
            synchronize_paging();
            st.block = c;
            occ[(size_t)c] = c;
        }
        for (int b : vacated) zero_block(b);
        for (int c = 0; c < keep; c++) {         // host-only chunks: recall into their own block
            ChunkState & st = chunks_[(size_t)c];
            if (st.block >= 0 || !st.on_host) continue;
            st.block = c;
            st.dirty = false;
            copy_chunk(c, c, /*to_host=*/false);
            stats_.page_ins++;
            stats_.moved_bytes += chunk_bytes_;
        }
        synchronize_paging();
        free_blocks_.clear();
        for (int b = n_blocks_ - 1; b >= keep; b--) free_blocks_.push_back(b);
        for (auto & st : chunks_) st.last_use = ++clock_;
        epoch_++;
    }

    // After truncate the pool holds what the dropped tail left behind: after a
    // long decode that is the NEWEST chunks, with older prompt chunks on host
    // and free blocks to spare. A pager that only ever saw P tokens would hold
    // the best ones that fit (recency, or score when a scorer is attached), so
    // recall host chunks into the free blocks the same way, then renumber the
    // use clock in logical order (as appending produces it) so later evictions
    // pick the same victims. Residency is exact either way (host copies are
    // bit-exact); this only changes which chunks decode and replay attend over.
    void refill_after_truncate_() {
        std::vector<int> host;
        for (int c = 0; c < (int)chunks_.size(); c++)
            if (chunks_[(size_t)c].block < 0 && chunks_[(size_t)c].on_host) host.push_back(c);
        // Protected chunks (sink, tail window, pins) come back first, exactly
        // as a fresh pager would hold them; only the rest is ranked.
        auto prot = [&](int c) {
            return c < cfg_.sink_chunks || c > cur_chunk_ - 1 - cfg_.tail_window_chunks ||
                   is_pinned(c);
        };
        auto mid = std::stable_partition(host.begin(), host.end(), prot);
        if (score_hook)
            std::stable_sort(mid, host.end(),
                             [&](int a, int b) { return score_hook(a) > score_hook(b); });
        else
            std::reverse(mid, host.end());                   // newest first
        std::vector<int> take(host.begin(),
                              host.begin() + std::min(host.size(), free_blocks_.size()));
        std::sort(take.begin(), take.end());                  // recall oldest first
        for (int c : take) page_in(c);
        for (auto & st : chunks_) st.last_use = 0;
        for (int c = 0; c < (int)chunks_.size(); c++) chunks_[(size_t)c].last_use = ++clock_;
    }

    // Zero every currently-free block. reset() drops mappings but leaves the
    // previous request's bytes in place; maskless consumers (the qwen35moe
    // pipelined decode reads the whole padded pool span with no slot mask)
    // need stale rows to dequantise to ~zero contribution. Masked consumers
    // don't need this but it is cheap (pool-sized memset, sub-ms).
    void zero_free_blocks() {
        for (int b : free_blocks_) zero_block(b);
#ifdef KVFLASH_HAS_ASYNC_DMA
        sync_page_stream_();
#endif
    }

    bool attached() const { return n_blocks_ > 0; }
    int pool_tokens() const { return cfg_.pool_tokens; }
    int chunk_tokens() const { return cfg_.chunk_tokens; }

    // Optional external relevance score; higher = keep. Falls back to LRU.
    std::function<float(int /*chunk*/)> score_hook;

    // Pin the chunks overlapping logical token range [tok_lo, tok_hi)
    // (half-open). Pinned chunks are never eviction victims and are always
    // kept by reselect(), on top of the sink/tail protections. Pins are
    // cleared by reset() (and so by deserialize()). Best-effort deadlock
    // guard: if sink + tail + pinned + 2 would exceed the pool the whole span
    // is refused with a warning and no pin is set. Empty/inverted ranges are
    // no-ops.
    void pin_range(int64_t tok_lo, int64_t tok_hi) { pin_chunks_(tok_lo, tok_hi, kPinHard); }
    void pin_chunks_(int64_t tok_lo, int64_t tok_hi, uint8_t kind) {
        if (!attached() || tok_lo < 0 || tok_hi <= tok_lo) return;
        // Validate in 64 bits BEFORE narrowing to a chunk index: a position past
        // 2^31 chunks would wrap negative and index pinned_ out of bounds, and a
        // span longer than the pool could never be pinned anyway (so the scan
        // below stays O(pool) and pinned_ stays small).
        const int64_t lo64 = tok_lo / cfg_.chunk_tokens;
        const int64_t hi64 = (tok_hi - 1) / cfg_.chunk_tokens;      // inclusive
        if (hi64 >= kMaxPinChunks || hi64 - lo64 + 1 > n_blocks_) {
            std::fprintf(stderr,
                "[kvflash] pin_range [%lld,%lld) refused: chunks [%lld,%lld] "
                "outside the pinnable range (< %d chunks, span <= %d)\n",
                (long long)tok_lo, (long long)tok_hi, (long long)lo64,
                (long long)hi64, kMaxPinChunks, n_blocks_);
            return;
        }
        const int c_lo = (int)lo64;
        const int c_hi = (int)hi64;
        int pinned = 0, fresh = 0;
        for (uint8_t f : pinned_) pinned += f != 0;
        for (int c = c_lo; c <= c_hi; c++) fresh += !is_pinned(c);
        if (cfg_.sink_chunks + cfg_.tail_window_chunks + pinned + fresh + 2 > n_blocks_) {
            std::fprintf(stderr,
                "[kvflash] pin_range [%lld,%lld) refused: sink=%d tail=%d "
                "pinned=%d new=%d pool_blocks=%d would deadlock eviction\n",
                (long long)tok_lo, (long long)tok_hi, cfg_.sink_chunks,
                cfg_.tail_window_chunks, pinned, fresh, n_blocks_);
            return;
        }
        if (c_hi + 1 > (int)pinned_.size()) pinned_.resize((size_t)c_hi + 1, 0);
        for (int c = c_lo; c <= c_hi; c++)
            if (kind == kPinHard || !pinned_[(size_t)c]) pinned_[(size_t)c] = kind;
    }
    // Pin the chunks overlapping [tok_lo, tok_hi) until end_step(). Same
    // guard and refusal as pin_range, and the step pins count against it. A
    // chunk already pinned by pin_range stays hard-pinned.
    void pin_step_range(int64_t tok_lo, int64_t tok_hi) { pin_chunks_(tok_lo, tok_hi, kPinStep); }
    // Step boundary: step pins expire, pin_range pins stay.
    void end_step() {
        for (auto & f : pinned_) if (f == kPinStep) f = 0;
    }
    bool is_pinned(int c) const {
        return c >= 0 && c < (int)pinned_.size() && pinned_[(size_t)c] != 0;
    }
    void unpin_all() { std::fill(pinned_.begin(), pinned_.end(), 0); }

    // Allocate slots for [kv_start, kv_start + n_tok) ahead of a forward
    // step (evicting LRU/low-score chunks as needed). False — with a
    // diagnostic — if the pool has no evictable block left.
    //
    // When KVFLASH_HAS_ASYNC_DMA: page-out D2H and slot-zero transfers are
    // queued on page_stream_ and left running.  If any page-in (H2D) was
    // issued, page_stream_ is synchronised here so that recalled KV data is
    // resident before the next attention forward.
    bool alloc_span(int kv_start, int n_tok) {
        for (int i = 0; i < n_tok; ++i) {
            if (slot_for(kv_start + i) < 0) {
                std::fprintf(stderr, "[kvflash] no pool slot at pos %d "
                                     "(pool %d exhausted)\n",
                             kv_start + i, cfg_.pool_tokens);
                return false;
            }
        }
        if (has_pending_page_in_ && !batch_submit_) synchronize_paging();   // batch: the step's one Fence waits
        return true;
    }

    // Physical pool slot for logical position `pos`. Allocates (and, when
    // the pool is full, evicts) at chunk granularity. Call once per
    // appended token, in logical order.
    // A slot_for() is a row write request: the chunk's host copy (if any) is stale
    // from here on, so the next page_out copies.
    int slot_for(int64_t pos) { return slot_for_(pos, /*write=*/true); }

    // `write` false is a pure recall (page_in, deserialize): a chunk restored from
    // host stays clean, so page_out can skip the D2H while no row is written.
    int slot_for_(int64_t pos, bool write) {
        const int c = (int)(pos / cfg_.chunk_tokens);
        // cur_chunk_ tracks the append head only; a page_in of an older
        // chunk must not shrink the protected tail window. It must advance
        // BEFORE eviction (so the victim search protects the new tail), but
        // a failed allocation must roll it back or the next eviction's tail
        // window is computed from a chunk that never materialized.
        const int prev_cur_chunk = cur_chunk_;
        if (c > cur_chunk_) cur_chunk_ = c;
        if ((int)chunks_.size() <= c) chunks_.resize(c + 1);
        ChunkState & st = chunks_[c];
        if (st.block < 0) {
            if (!ensure_free_block()) {
                cur_chunk_ = prev_cur_chunk;
                return -1;
            }
            st.block = free_blocks_.back();
            free_blocks_.pop_back();
            epoch_++;
            st.dirty = write || !st.on_host;
            if (st.on_host) {              // recall: restore paged-out bytes
                copy_chunk(c, st.block, /*to_host=*/false);
                stats_.page_ins++;
                stats_.moved_bytes += chunk_bytes_;
#ifdef KVFLASH_HAS_ASYNC_DMA
                has_pending_page_in_ = true;
#endif
            }
            // The caller uses this block on the compute stream next.
            fence_page_work_();
        }
        else if (write) st.dirty = true;
        st.last_use = ++clock_;
        return row_base_ + st.block * cfg_.chunk_tokens + (int)(pos % cfg_.chunk_tokens);
    }

    // Force a chunk out of the pool (host backing + zeroed slots). A pinned
    // chunk is refused: its rows belong to the step (or request) that pinned it.
    bool page_out(int c) {
        if (c >= (int)chunks_.size() || chunks_[c].block < 0 || is_pinned(c)) return false;
        ChunkState & st = chunks_[c];
        if (has_tensor_storage() && !st.on_host) {
#ifdef KVFLASH_HAS_ASYNC_DMA
            if (!alloc_host_buffer_(st)) return false;
#else
            st.host_data.resize(chunk_bytes_);
#endif
            stats_.host_bytes += (int64_t)chunk_bytes_;
        }
        // A chunk recalled from host and not written since holds the bytes its host
        // copy already has (every row write goes through slot_for(), which sets dirty).
        if (has_tensor_storage() && st.on_host && !st.dirty) stats_.d2h_skipped++;
        else                                                  copy_chunk(c, st.block, /*to_host=*/true);
        zero_block(st.block);
        st.on_host = true;
        st.dirty = false;
        if (deferred_free_) pending_free_.push_back({st.block, ++record_seq_});
        else                free_blocks_.push_back(st.block);
        st.block = -1;
        epoch_++;
        stats_.page_outs++;
        stats_.moved_bytes += chunk_bytes_;
        return true;
    }

    // Page-out for an op planned against `generation` (generation() when it was
    // planned): a request that ended or restarted in between (reset/attach)
    // makes the op stale, and a stale op must not evict the new request's chunk.
    bool page_out_checked(int c, uint64_t generation) {
        return generation == generation_ && evictable_(c) && page_out(c);
    }
    uint64_t generation() const { return generation_; }

    // Failed step: step pins expire and every queued copy settles (the host
    // wait), so deferred frees complete and nothing in flight touches a block
    // the caller releases next. Idempotent; hard pins stay.
    void abort_step() {
        end_step();
        synchronize_paging();
        generation_++;   // page-outs planned before the abort are stale
    }

    // Arena mode (KvFlashArena): copies are only queued. slot_for/alloc_span do
    // not wait or fence; the step plan's one Fence orders the compute stream
    // after them, and arena completion reclaims the blocks.
    void set_batch_submit(bool on) { batch_submit_ = on; }
    void * page_stream() const {
#ifdef KVFLASH_HAS_ASYNC_DMA
        return page_stream_;
#else
        return nullptr;
#endif
    }
    // Use the arena's page stream. Before attach(); never destroyed here.
    void set_external_page_stream(void * stream) {
#ifdef KVFLASH_HAS_ASYNC_DMA
        ext_page_stream_ = (cudaStream_t)stream;
#else
        (void)stream;
#endif
    }

    // Recall a chunk into the pool (used by reselect / tests).
    bool page_in(int c) {
        if (c >= (int)chunks_.size() || !chunks_[c].on_host || chunks_[c].block >= 0) return false;
        return slot_for_((int64_t)c * cfg_.chunk_tokens, /*write=*/false) >= 0;
    }

    // Optional fast path (CUDA). The stream that consumes the pool (the
    // backend's compute stream) is non-blocking, so it is NOT implicitly
    // ordered with page_stream_. By default slot_for() therefore waits on the
    // host for the queued page work before it hands out a block. With the
    // compute stream registered here it instead makes that stream wait on the
    // device, with no host stall for that fence (recall in alloc_span() and
    // reselect() still waits on the host). Call after attach(), which clears
    // it. Only register a stream that runs EVERY graph touching the pool: a
    // pool spread over several devices (layer split) has one compute stream
    // per device and must stay on the default. Ignored on HIP builds.
    void set_compute_stream(void * stream) {
#ifdef KVFLASH_EVENT_FENCE
        compute_stream_ = (cudaStream_t)stream;
#else
        (void)stream;
#endif
    }

    // Block until queued page DMA on page_stream_ completes. Hot paths
    // (alloc_span/reselect) batch this internally after their recalls; callers
    // that issue page_in()/page_out() directly and then read device KV (tests,
    // tools) must call this first. No-op when async DMA is compiled out.
    void synchronize_paging() {
#ifdef KVFLASH_HAS_ASYNC_DMA
        sync_page_stream_();
#endif
        has_pending_page_in_ = false;
        complete(record_seq_);   // every queued D2H has landed
    }

    // Deferred free: with deferral on, a page-out's block goes to a pending
    // list and is allocatable only after complete(record) for a record >= its
    // own. Off by default: blocks are reusable at once and slot_for() fences
    // instead. With deferral on, an alloc that finds only pending blocks fails
    // (no extra victim is taken) and an eviction on demand queues its block and
    // fails once; the caller runs page-outs a step ahead, completes, then
    // allocates.
    // ponytail: completion is the caller's explicit complete() (tests) or
    // synchronize_paging() (the host wait, HIP's step-end sync). The CUDA event
    // query per record is a later device step.
    void set_deferred_free(bool on) { deferred_free_ = on; }
    uint64_t last_record() const { return record_seq_; }       // record of the latest page-out
    int pending_free_blocks() const { return (int)pending_free_.size(); }
    void complete(uint64_t record) {
        size_t keep = 0;
        for (const auto & pf : pending_free_) {
            if (pf.second <= record) free_blocks_.push_back(pf.first);
            else pending_free_[keep++] = pf;
        }
        pending_free_.resize(keep);
    }

    bool is_resident(int c) const {
        return c < (int)chunks_.size() && chunks_[c].block >= 0;
    }

    // True while every materialized chunk still sits in its identity block
    // (chunk c in block c, nothing paged out). This is the layout contract
    // identity-copy snapshots rely on; it holds from reset() until the
    // first eviction of the CURRENT request (cumulative stats do not).
    // Arena identity: false at any nonzero row_base, because "chunk c in
    // block c" then names another slot's rows.
    bool is_identity() const {
        if (row_base_ != 0) return false;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            if (chunks_[c].block >= 0 && chunks_[c].block != c) return false;
            if (chunks_[c].block < 0 && chunks_[c].on_host) return false;
        }
        return true;
    }

    // True iff every chunk intersecting [0, n_tok) is resident in its identity
    // block (block_of(c) == c). Expresses "the logical prefix [0, n_tok) is a
    // contiguous, identity-mapped, materialized span" — the exact precondition
    // the non-paged tree-verify graph relies on. Stronger than is_identity(),
    // which is also true for an empty / not-yet-materialized pager.
    bool identity_prefix_covers(int n_tok) const {
        if (row_base_ != 0) return false;   // arena identity only
        if (n_tok <= 0) return true;
        const int nc = (n_tok + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens;
        if (nc > (int)chunks_.size()) return false;
        for (int c = 0; c < nc; c++)
            if (chunks_[c].block != c) return false;
        return true;
    }
    // ARENA block holding chunk c (row_base()/chunk_tokens + pool-local
    // block), -1 if not resident: the block-table entry and the tensor row
    // block, in the same space as slot_for()/slot_of().
    int block_of(int c) const {
        if (c < 0 || c >= (int)chunks_.size() || chunks_[c].block < 0) return -1;
        return row_base_ / cfg_.chunk_tokens + chunks_[c].block;
    }

    // First arena row of this pool (0 unless it is a slot slice).
    int row_base() const { return row_base_; }

    // Everything that reaches a tensor, graph, mask or snapshot is in ARENA
    // space: slot_for()/slot_of() rows, block_of() blocks, and the columns of
    // fill_slot_pos()/fill_slot_mask() (dst holds arena_tokens() entries; the
    // columns outside this slice stay empty). Pool-local ids are internal.
    int arena_tokens() const { return row_base_ + cfg_.pool_tokens; }

    // Const lookup (no alloc / LRU touch): physical slot currently holding
    // logical `pos`, or -1 if its chunk is not resident. Callers that may
    // need an allocation must use slot_for() beforehand.
    int slot_of(int64_t pos) const {
        const int c = (int)(pos / cfg_.chunk_tokens);
        if (c >= (int)chunks_.size() || chunks_[c].block < 0) return -1;
        return row_base_ + chunks_[c].block * cfg_.chunk_tokens + (int)(pos % cfg_.chunk_tokens);
    }

    // Logical position held by each pool slot, -1 for free blocks. `dst`
    // must hold arena_tokens() entries. Lets callers build masks that need
    // POSITION semantics in slot space (causal / sliding-window): the
    // mask condition is evaluated on dst[slot] instead of the column index.
    void fill_slot_pos(int32_t * dst) const {
        for (int i = 0; i < arena_tokens(); i++) dst[i] = -1;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            if (chunks_[c].block < 0) continue;
            int32_t * p = dst + (size_t)row_base_ + (size_t)chunks_[c].block * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; i++)
                p[i] = (int32_t)c * cfg_.chunk_tokens + i;
        }
    }
    const KvFlashStats & stats() const { return stats_; }
    size_t chunk_bytes() const { return chunk_bytes_; }
    int resident_blocks() const { return n_blocks_ - (int)free_blocks_.size() - (int)pending_free_.size(); }
    int n_chunks() const { return (int)chunks_.size(); }
    int cur_chunk() const { return cur_chunk_; }
    int n_pool_blocks() const { return n_blocks_; }
    // Bytes of one (layer, kv_head) K segment: chunk_tokens contiguous rows.
    size_t k_seg_bytes() const { return k_seg_bytes_; }
    // Host copy of chunk c's K rows for (layer, kv_head), or nullptr when the
    // chunk is resident (read the pool tensor instead) or has no host copy.
    // Host layout (see copy_chunk): per layer, K for all heads then V for all
    // heads, each segment k_seg_bytes_ / v_seg_bytes_.
    const uint8_t * host_k_seg(int c, int layer, int head) const {
        if (!has_tensor_storage() || c < 0 || c >= (int)chunks_.size()) return nullptr;
        const ChunkState & st = chunks_[(size_t)c];
        if (st.block >= 0 || !st.on_host) return nullptr;
        if (layer < 0 || layer >= (int)attn_k_.size() || head < 0 || head >= n_head_kv_) return nullptr;
        const size_t off = (size_t)layer * n_head_kv_ * (k_seg_bytes_ + v_seg_bytes_) +
                           (size_t)head * k_seg_bytes_;
#ifdef KVFLASH_HAS_ASYNC_DMA
        return (const uint8_t *)st.host_data + off;
#else
        return st.host_data.data() + off;
#endif
    }

    // Bumped on every residency change (alloc / page_out / page_in).
    // Callers cache the slot mask and refill only when the epoch moves.
    uint64_t epoch() const { return epoch_; }

    // F16 slot-validity mask for one query row: 0 for slots belonging to a
    // resident chunk, -inf for free / paged-out blocks. `dst` must hold
    // arena_tokens() entries. Used as the FA mask so non-resident slots are
    // excluded exactly instead of via the zero-row ~exp(-max) approximation.
    // valid_upto bounds the logical positions that may be attended: rows of a
    // resident chunk at positions >= valid_upto (rejected drafts, a previous
    // turn's tokens above a rewind) are masked off. A decode step at position p
    // passes p + 1: it attends the rows below p and its own row. The default
    // leaves every resident row visible, as before.
    void fill_slot_mask(uint16_t * dst, int64_t valid_upto = INT64_MAX) const {
        constexpr uint16_t F16_ZERO = 0x0000, F16_NEG_INF = 0xFC00;
        for (int i = 0; i < arena_tokens(); i++) dst[i] = F16_NEG_INF;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            if (chunks_[c].block < 0) continue;
            uint16_t * p = dst + (size_t)row_base_ + (size_t)chunks_[c].block * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; i++)
                if ((int64_t)c * cfg_.chunk_tokens + i < valid_upto) p[i] = F16_ZERO;
        }
    }

    // Reselect to the recency set (sink + the newest chunks that fit), then drop the
    // hook so later evictions are plain LRU. Returns the number of page events.
    int reselect_recent() {
        score_hook = [](int c) { return (float)c; };
        const int events = reselect(0);
        score_hook = nullptr;
        return events;
    }

    // Lookahead reselect (FlashMemory τ-step): rebuild the resident set as
    // the top-pool chunks by score_hook among ALL known chunks (resident or
    // host-backed). Sinks and the trailing window are always kept. Returns
    // the number of page events. Call between decode steps.
    // tail_override >= 0 replaces cfg tail_window_chunks for this call only.
    // Score ties keep already-resident chunks, so a tail covering every chunk is a no-op.
    int reselect(int tail_override = -1) {
        if (!score_hook) return 0;
        const int tail = tail_override >= 0 ? tail_override : cfg_.tail_window_chunks;
        struct Cand { int c; float s; bool res; };
        std::vector<Cand> cands;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            const ChunkState & st = chunks_[c];
            if (st.block < 0 && !st.on_host) continue;     // never materialized
            const bool prot = c < cfg_.sink_chunks ||
                              c > cur_chunk_ - 1 - tail ||
                              is_pinned(c);
            cands.push_back({c, prot ? 3.4e38f : score_hook(c), st.block >= 0});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand & a, const Cand & b) {
                      return a.s != b.s ? a.s > b.s : a.res > b.res;   // ties keep resident chunks
                  });
        std::vector<uint8_t> want(chunks_.size(), 0);
        for (int i = 0; i < (int)cands.size() && i < n_blocks_; i++) want[cands[i].c] = 1;

        int events = 0;
        for (int c = 0; c < (int)chunks_.size(); c++) {       // out first: frees blocks
            if (!want[c] && chunks_[c].block >= 0) { page_out(c); events++; }
        }
        for (int c = 0; c < (int)chunks_.size(); c++) {
            if (want[c] && chunks_[c].block < 0 && chunks_[c].on_host) {
                if (page_in(c)) events++;
            }
        }
        if (has_pending_page_in_) synchronize_paging();
        return events;
    }

    // Resident, and not sink, tail window or pinned: what an eviction may take.
    bool evictable_(int c) const {
        return c >= 0 && c < (int)chunks_.size() && chunks_[c].block >= 0 && c >= cfg_.sink_chunks &&
               c <= cur_chunk_ - 1 - cfg_.tail_window_chunks && !is_pinned(c);
    }
    // Up to n eviction victims, worst first: evictable chunks ordered by score
    // (score_hook, lowest first) or by oldest use (LRU fallback); ties go to
    // the lower chunk id.
    std::vector<int> lru_victims(int n) const {
        std::vector<std::pair<std::pair<float, uint64_t>, int>> cand;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            if (!evictable_(c)) continue;
            cand.push_back({score_hook ? std::make_pair(score_hook(c), uint64_t(0))
                                       : std::make_pair(0.f, chunks_[c].last_use), c});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        std::vector<int> out;
        for (size_t i = 0; i < cand.size() && (int)i < n; i++) out.push_back(cand[i].second);
        return out;
    }

    // Debug: resident chunk ids as "a-b,c" ranges, and a hash of the score
    // ledger of chunks < upto (LUCE_KVFLASH_DEBUG_RESIDENT).
    std::string debug_resident(int upto) const {
        std::string out;
        char buf[48];
        for (int c = 0, n = (int)chunks_.size(); c < n;) {
            if (chunks_[(size_t)c].block < 0) { c++; continue; }
            int e = c;
            while (e + 1 < n && chunks_[(size_t)e + 1].block >= 0) e++;
            std::snprintf(buf, sizeof buf, "%s%d-%d", out.empty() ? "" : ",", c, e);
            out += buf;
            c = e + 1;
        }
        uint64_t h = kFnvBasis;
        for (int c = 0; c < upto && c < (int)chunks_.size(); c++) {
            uint32_t b; const float sc = chunks_[(size_t)c].score;
            std::memcpy(&b, &sc, 4);
            h = fnv1a_mix(h, b);
        }
        std::snprintf(buf, sizeof buf, " nchunks=%d ledger_hash=%016llx", (int)chunks_.size(), (unsigned long long)h);
        return out + buf;
    }

    // Per-chunk relevance ledger (QK scorer): set before serialize() so the
    // scores round-trip. -inf = never scored.
    void set_chunk_score(int c, float s) {
        if (c >= 0 && c < (int)chunks_.size()) chunks_[(size_t)c].score = s;
    }
    float chunk_score(int c) const {
        return c >= 0 && c < (int)chunks_.size()
            ? chunks_[(size_t)c].score : -std::numeric_limits<float>::infinity();
    }

    // ggml_type of the K cache recorded in the most recent successfully
    // deserialized blob.
    uint32_t serialized_kv_k_type() const { return last_serialized_kv_k_type_; }

    // Snapshot chunks [0, max_chunks) (all when max_chunks < 0 or beyond
    // n_chunks) into a flat blob; pair with deserialize(). Resident chunks are
    // read from the pool tensors, paged-out chunks from their host backing
    // (pinned host_data on async-DMA builds), never-written chunks are zero.
    //
    // Layout, magic "KVFLASH3" (u64); all integers and floats are written in host
    // byte order, which the static_assert at the top of this file pins to little-endian:
    //   9 x uint32 header: nc, chunk_tokens, n_head_kv, k_seg_bytes,
    //                      v_seg_bytes, chunk_bytes, ggml_type(K), has_ledger(=1),
    //                      ggml_type(V)
    //   nc x chunk_bytes   KV bytes, segment order of copy_chunk()
    //   nc x 8 bytes       ledger: u8 was_resident, u8 materialized, 2 pad, f32 score
    //                      (a hole left by a failed allocation is not materialized)
    //   pool layout:       u32 n_blocks, u32 n_free, u64 use clock,
    //                      nc x (i32 block or -1, u32 0, u64 LRU rank, 0 if not resident),
    //                      n_free x i32 free blocks in free-list order (back = next)
    // The pool layout lets deserialize() put every chunk back in its own block
    // with its LRU stamp, so a restored pager is the pager that was saved:
    // attention then reads the same slots in the same order as a cold prefill,
    // and later evictions pick the same victims. Chunks past max_chunks give
    // their blocks to the free list. Not serialized: pins.
    //
    // KVFLASH2 is the same blob without the pool layout: blob_header() +
    // read_chunk(0..nc) + ledger_blob(), streamed by serialize_to(). A writer
    // that keeps chunks apart (a slot bundle rewrites only the chunks a turn
    // changed) assembles it; deserialize() takes both and restores a KVFLASH2
    // blob in identity block order.
    std::vector<uint8_t> serialize(int max_chunks = -1) const {
        synchronize_page_stream_();   // queued D2H must land in host_data first
        const int nc = serialized_chunks_(max_chunks);
        const size_t kv_bytes = (size_t)nc * chunk_bytes_;
        std::vector<int32_t> free_list(free_blocks_.begin(), free_blocks_.end());
        // A block awaiting its page-out record holds no resident chunk: free in the layout.
        for (const auto & pf : pending_free_) free_list.insert(free_list.begin(), pf.first);
        for (int c = (int)chunks_.size() - 1; c >= nc; --c)   // dropped tail: handed out last
            if (chunks_[(size_t)c].block >= 0) free_list.insert(free_list.begin(), chunks_[(size_t)c].block);
        const size_t ledger_end = kBlobHdr + kv_bytes + (size_t)nc * kLedgerEntry;
        // LRU stamps as dense ranks among the resident chunks (0 elsewhere):
        // eviction only compares stamps and new ones always exceed the clock,
        // so ranks behave exactly like the raw clock values, and two pagers
        // that would evict alike serialize alike.
        std::vector<uint64_t> stamps;
        for (int c = 0; c < nc; ++c)
            if (chunks_[(size_t)c].block >= 0) stamps.push_back(chunks_[(size_t)c].last_use);
        std::sort(stamps.begin(), stamps.end());
        stamps.erase(std::unique(stamps.begin(), stamps.end()), stamps.end());
        auto rank_of = [&](uint64_t u) {
            return (uint64_t)(std::lower_bound(stamps.begin(), stamps.end(), u) - stamps.begin()) + 1;
        };
        std::vector<uint8_t> out(ledger_end + kPoolHdr + (size_t)nc * kPoolEntry +
                                 free_list.size() * sizeof(int32_t), 0);
        const std::vector<uint8_t> head = blob_header(max_chunks);
        std::memcpy(out.data(), head.data(), kBlobHdr);
        std::memcpy(out.data(), &kBlobMagic3, 8);   // the pool layout below makes it KVFLASH3
        for (int c = 0; c < nc; ++c) read_chunk(c, out.data() + kBlobHdr + (size_t)c * chunk_bytes_);
        const std::vector<uint8_t> ledger = ledger_blob(max_chunks);
        std::memcpy(out.data() + kBlobHdr + kv_bytes, ledger.data(), ledger.size());
        for (int c = 0; c < nc; ++c) {
            const ChunkState & st = chunks_[(size_t)c];
            uint8_t * pe = out.data() + ledger_end + kPoolHdr + (size_t)c * kPoolEntry;
            const int32_t blk = st.block;
            const uint64_t rank = st.block >= 0 ? rank_of(st.last_use) : 0;
            std::memcpy(pe, &blk, 4);
            std::memcpy(pe + 8, &rank, 8);
        }
        uint8_t * ph = out.data() + ledger_end;
        const uint32_t pool_hdr[2] = {(uint32_t)n_blocks_, (uint32_t)free_list.size()};
        const uint64_t clock = stamps.size();
        std::memcpy(ph, pool_hdr, 8);
        std::memcpy(ph + 8, &clock, 8);
        if (!free_list.empty())
            std::memcpy(ph + kPoolHdr + (size_t)nc * kPoolEntry, free_list.data(),
                        free_list.size() * sizeof(int32_t));
        return out;
    }

    // KVFLASH2 streamed through sink(bytes, n) -> bool, one chunk of staging
    // instead of the whole blob (a disk bundle writes it straight to its file).
    // False when the sink refuses.
    template <class Sink>
    bool serialize_to(Sink && sink, int max_chunks = -1) const {
        synchronize_page_stream_();   // queued D2H must land in host_data first
        const int nc = serialized_chunks_(max_chunks);
        const std::vector<uint8_t> head = blob_header(max_chunks);
        if (!sink(head.data(), head.size())) return false;
        std::vector<uint8_t> chunk(chunk_bytes_);
        for (int c = 0; c < nc; ++c) {
            read_chunk(c, chunk.data());
            if (!sink(chunk.data(), chunk_bytes_)) return false;
        }
        const std::vector<uint8_t> ledger = ledger_blob(max_chunks);
        return sink(ledger.data(), ledger.size());
    }

    // The three parts of a KVFLASH2 blob: blob_header() + read_chunk(0..nc) + ledger_blob().
    std::vector<uint8_t> blob_header(int max_chunks = -1) const {
        std::vector<uint8_t> head(kBlobHdr);
        std::memcpy(head.data(), &kBlobMagic2, 8);
        const uint32_t hdr[9] = {(uint32_t)serialized_chunks_(max_chunks), (uint32_t)cfg_.chunk_tokens,
            (uint32_t)n_head_kv_, (uint32_t)k_seg_bytes_, (uint32_t)v_seg_bytes_,
            (uint32_t)chunk_bytes_, live_k_type_(), 1u, live_v_type_()};
        std::memcpy(head.data() + 8, hdr, sizeof(hdr));
        return head;
    }
    // chunk_bytes() into dst: a resident chunk from the pool tensors, a paged-out one from its
    // host backing, a never-written one zeros. After synchronize_paging().
    void read_chunk(int c, uint8_t * dst) const {
        const ChunkState & st = chunks_[(size_t)c];
        std::memset(dst, 0, chunk_bytes_);
        if (st.block >= 0 && has_tensor_storage()) {
            for (size_t l = 0; l < attn_k_.size(); ++l) {
                for (int kv = 0; kv < 2; ++kv) {
                    ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                    const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                    for (int h = 0; h < n_head_kv_; ++h) {
                        const size_t off = block_off_(st.block, t) + (size_t)h * t->nb[2];
                        ggml_backend_tensor_get(t, dst, off, seg);
                        dst += seg;
                    }
                }
            }
        } else if (st.block < 0 && st.on_host && has_tensor_storage()) {
#ifdef KVFLASH_HAS_ASYNC_DMA
            std::memcpy(dst, st.host_data, chunk_bytes_);
#else
            std::memcpy(dst, st.host_data.data(), chunk_bytes_);
#endif
        }
    }
    // nc x 8 bytes: u8 was_resident, u8 materialized, 2 pad, f32 score.
    std::vector<uint8_t> ledger_blob(int max_chunks = -1) const {
        const int nc = serialized_chunks_(max_chunks);
        std::vector<uint8_t> ledger((size_t)nc * kLedgerEntry, 0);
        for (int c = 0; c < nc; ++c) {
            const ChunkState & st = chunks_[(size_t)c];
            uint8_t * lp = ledger.data() + (size_t)c * kLedgerEntry;
            lp[0] = st.block >= 0 ? 1u : 0u;
            lp[1] = (st.block >= 0 || st.on_host) ? 1u : 0u;
            std::memcpy(lp + 4, &st.score, 4);
        }
        return ledger;
    }

    // Replace the pager state with a serialize() (KVFLASH3) or serialize_to()
    // (KVFLASH2) blob. The blob is fully validated (magic, geometry, K and V
    // dtype, length, resident count <= pool, pool layout) BEFORE any state is
    // touched: a rejected blob returns false and leaves the pager as it was.
    // On success: reset() semantics (pins cleared), chunks that were resident
    // get a pool block (bytes restored), the rest stay host-backed and
    // recallable via page_in(), and scores are restored. When a KVFLASH3 blob
    // was saved from a pool with this block count, every resident chunk gets
    // its saved block, LRU stamp, use clock and free-list order back, so the
    // pager is exactly the one that was saved; otherwise (another pool size,
    // or a KVFLASH2 blob) only the logical residency set is kept, in identity
    // block order. Callers must rebuild slot masks (epoch advances). Returns
    // false (pager reset/empty) only if a pinned host buffer cannot be
    // allocated.
    // expect_chunks >= 0: the blob must hold exactly that many chunks (a bundle
    // adoption admitted that count); checked with the header, before any state moves.
    bool deserialize(const uint8_t * data, size_t n, int expect_chunks = -1) {
        if (!attached() || !data || n < kBlobHdr) return false;
        uint64_t magic = 0;
        std::memcpy(&magic, data, 8);
        const bool pooled = magic == kBlobMagic3;   // pool layout trailer present
        if (!pooled && magic != kBlobMagic2) return false;
        uint32_t h[9];
        std::memcpy(h, data + 8, sizeof(h));
        const size_t nc = h[0];
        if (expect_chunks >= 0 && nc != (size_t)expect_chunks) return false;
        if ((int)h[1] != cfg_.chunk_tokens || (int)h[2] != n_head_kv_ ||
            h[3] != k_seg_bytes_ || h[4] != v_seg_bytes_ || h[5] != chunk_bytes_ ||
            h[6] != live_k_type_() || h[7] != 1u || h[8] != live_v_type_()) {
            return false;
        }
        // Overflow-safe length checks: nc and n_free come from the (untrusted) blob.
        const size_t pool_hdr_bytes = pooled ? kPoolHdr : 0, pool_entry = pooled ? kPoolEntry : 0;
        if (n - kBlobHdr < pool_hdr_bytes) return false;
        if (nc > (n - kBlobHdr - pool_hdr_bytes) / (chunk_bytes_ + kLedgerEntry + pool_entry)) return false;
        const uint8_t * kv = data + kBlobHdr;
        const uint8_t * lp = kv + nc * chunk_bytes_;
        size_t n_resident = 0;
        for (size_t c = 0; c < nc; ++c) {
            const uint8_t res = lp[c * kLedgerEntry], mat = lp[c * kLedgerEntry + 1];
            if (mat > 1 || (res != 0 && mat == 0)) return false;   // resident implies materialized
            n_resident += res != 0;
        }
        if (n_resident > (size_t)n_blocks_) return false;
        // Exact layout only from a KVFLASH3 blob into a pool of the saved size;
        // the saved blocks and free list must then be a permutation of the pool's blocks.
        const uint8_t * ph = lp + nc * kLedgerEntry;
        const uint8_t * pe = ph + kPoolHdr;
        const uint8_t * fl = pe + nc * kPoolEntry;
        size_t n_free = 0;
        bool exact = false;
        if (pooled) {
            uint32_t pool_hdr[2];
            std::memcpy(pool_hdr, ph, 8);
            n_free = pool_hdr[1];
            const size_t fixed = kBlobHdr + nc * (chunk_bytes_ + kLedgerEntry + kPoolEntry) + kPoolHdr;
            if (n_free > (n - fixed) / sizeof(int32_t)) return false;
            exact = (int64_t)pool_hdr[0] == (int64_t)n_blocks_;
        }
        if (exact) {
            if (n_resident + n_free != (size_t)n_blocks_) return false;
            std::vector<uint8_t> used((size_t)n_blocks_, 0);
            auto take = [&](int32_t b) {
                if (b < 0 || b >= n_blocks_ || used[(size_t)b]) return false;
                used[(size_t)b] = 1;
                return true;
            };
            for (size_t c = 0; c < nc; ++c) {
                int32_t b;
                std::memcpy(&b, pe + c * kPoolEntry, 4);
                if (lp[c * kLedgerEntry] != 0 ? !take(b) : b != -1) return false;
            }
            for (size_t i = 0; i < n_free; ++i) {
                int32_t b;
                std::memcpy(&b, fl + i * sizeof(int32_t), 4);
                if (!take(b)) return false;
            }
        }

        reset();
        last_serialized_kv_k_type_ = h[6];
        chunks_.resize(nc);
        for (size_t c = 0; c < nc; ++c) {
            ChunkState & st = chunks_[c];
            const bool materialized = lp[c * kLedgerEntry + 1] != 0;
            if (has_tensor_storage() && materialized) {
                const uint8_t * src = kv + c * chunk_bytes_;
#ifdef KVFLASH_HAS_ASYNC_DMA
                if (!alloc_host_buffer_(st)) {
                    reset();   // frees the host buffers allocated so far
                    return false;
                }
                std::memcpy(st.host_data, src, chunk_bytes_);
#else
                st.host_data.assign(src, src + chunk_bytes_);
#endif
                stats_.host_bytes += (int64_t)chunk_bytes_;
            }
            st.on_host = materialized;   // a hole stays a hole, not recallable zero KV
            std::memcpy(&st.score, lp + c * kLedgerEntry + 4, 4);
            if (exact) continue;
            // slot_for()'s recall branch copies host_data into the new block.
            if (lp[c * kLedgerEntry] != 0) slot_for_((int64_t)c * cfg_.chunk_tokens, /*write=*/false);
        }
        if (exact) {
            free_blocks_.resize(n_free);
            if (n_free > 0) std::memcpy(free_blocks_.data(), fl, n_free * sizeof(int32_t));
            for (size_t c = 0; c < nc; ++c) {
                ChunkState & st = chunks_[c];
                int32_t b;
                std::memcpy(&b, pe + c * kPoolEntry, 4);
                std::memcpy(&st.last_use, pe + c * kPoolEntry + 8, 8);
                if (b < 0) continue;
                st.block = b;
                if (st.on_host) {   // recall into the saved block, as slot_for() would
                    st.dirty = false;
                    copy_chunk((int)c, b, /*to_host=*/false);
                    stats_.page_ins++;
                    stats_.moved_bytes += chunk_bytes_;
#ifdef KVFLASH_HAS_ASYNC_DMA
                    has_pending_page_in_ = true;
#endif
                }
            }
            std::memcpy(&clock_, ph + 8, 8);
            epoch_++;
        }
        if (nc > 0 && cur_chunk_ < (int)nc - 1) cur_chunk_ = (int)nc - 1;
        if (has_pending_page_in_) synchronize_paging();   // async recalls settle
        return true;
    }

private:
    static constexpr uint64_t kBlobMagic2 = 0x4b56464c41534832ULL;  // "KVFLASH2": header, chunks, ledger
    static constexpr uint64_t kBlobMagic3 = 0x4b56464c41534833ULL;  // "KVFLASH3": KVFLASH2 + pool layout
    static constexpr size_t kBlobHdr = sizeof(uint64_t) + 9 * sizeof(uint32_t);
    static constexpr size_t kLedgerEntry = 8;
    static constexpr size_t kPoolHdr = 2 * sizeof(uint32_t) + sizeof(uint64_t);
    static constexpr size_t kPoolEntry = 16;
    static constexpr uint8_t kPinHard = 1, kPinStep = 2;   // pinned_ values; 0 = unpinned
    static constexpr int kMaxPinChunks = 1 << 22;   // pin flags stay <= 4 MiB

    int serialized_chunks_(int max_chunks) const {
        return (max_chunks >= 0 && max_chunks < (int)chunks_.size()) ? max_chunks : (int)chunks_.size();
    }

    uint32_t live_k_type_() const {
        return !attn_k_.empty() && attn_k_[0] ? (uint32_t)attn_k_[0]->type
                                              : (uint32_t)GGML_TYPE_F16;
    }
    uint32_t live_v_type_() const {
        return !attn_v_.empty() && attn_v_[0] ? (uint32_t)attn_v_[0]->type
                                              : (uint32_t)GGML_TYPE_F16;
    }
    void synchronize_page_stream_() const {
#ifdef KVFLASH_HAS_ASYNC_DMA
        cudaStreamSynchronize(page_stream_);
#endif
    }

    struct ChunkState {
        int      block    = -1;       // pool block index, -1 = not resident
        bool     on_host  = false;    // backing store holds valid bytes
        bool     dirty    = true;     // resident block may differ from its host copy (a row write, or no copy yet)
        uint64_t last_use = 0;
        float    score    = -std::numeric_limits<float>::infinity();  // QK ledger
#ifdef KVFLASH_HAS_ASYNC_DMA
        void *   host_data = nullptr; // pinned (cudaMallocHost) unless host_pinned is false
        bool     host_pinned = true;  // how host_data was allocated, so it is freed the same way
#else
        std::vector<uint8_t> host_data;
#endif
    };

    bool ensure_free_block() {
        if (!free_blocks_.empty()) return true;
        if (!pending_free_.empty()) return false;   // a block is in flight; take no second victim
        const std::vector<int> v = lru_victims(1);
        return !v.empty() && page_out(v[0]) && !free_blocks_.empty();
    }

#ifdef KVFLASH_HAS_ASYNC_DMA
    bool alloc_host_buffer_(ChunkState & st) {
        if (pinned_host_) {
            const cudaError_t err = cudaMallocHost(&st.host_data, chunk_bytes_);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "[kvflash] cudaMallocHost failed: %s\n",
                             cudaGetErrorString(err));
                st.host_data = nullptr;
                return false;
            }
        } else if (!(st.host_data = std::malloc(chunk_bytes_))) {
            return false;
        }
        st.host_pinned = pinned_host_;
        return true;
    }
    void free_host_buffer_(ChunkState & st) {
        if (!st.host_data) return;
        if (!st.host_pinned) {
            std::free(st.host_data);
        } else {
            const cudaError_t err = cudaFreeHost(st.host_data);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "[kvflash] cudaFreeHost failed: %s\n",
                             cudaGetErrorString(err));
            }
        }
        st.host_data = nullptr;
    }

    // True when the tensor lives in host memory (CPU backend / partial offload):
    // cudaMemcpyAsync/cudaMemsetAsync on such pointers are invalid.
    static bool is_host_tensor_(const ggml_tensor * t) {
        return t->buffer && ggml_backend_buffer_is_host(t->buffer);
    }
#endif

    // Move one chunk between pool slots and host backing.
    // Segment order is fixed (layer-major, K then V, head-minor).
    // When KVFLASH_HAS_ASYNC_DMA: transfers are issued on page_stream_
    // (async); the caller is responsible for synchronising before the data
    // is consumed.  The stream serialises D2H and H2D within a single
    // alloc_span(), so host_data is coherent before any H2D read starts.
    void copy_chunk(int c, int block, bool to_host) {
        ChunkState & st = chunks_[c];
#ifdef KVFLASH_HAS_ASYNC_DMA
        copy_block_(block, (uint8_t *)st.host_data, to_host);
#else
        copy_block_(block, st.host_data.data(), to_host);
#endif
    }

    // One pool block <-> a chunk_bytes_ host buffer (same layout as a chunk's
    // host copy). `host` must stay valid until synchronize_paging().
    void copy_block_(int block, uint8_t * host, bool to_host) {
        if (!has_tensor_storage()) return;
#ifdef KVFLASH_HAS_ASYNC_DMA
        size_t host_off = 0;
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                uint8_t * dev0 = (uint8_t *)t->data + block_off_(block, t);
                uint8_t * host0 = host + host_off;
                host_off += seg * (size_t)n_head_kv_;
                if (is_host_tensor_(t)) {  // CPU-backend tensor: plain memcpy
                    for (int h = 0; h < n_head_kv_; h++) {
                        uint8_t * d = dev0 + (size_t)h * t->nb[2];
                        uint8_t * hp = host0 + (size_t)h * seg;
                        if (to_host) std::memcpy(hp, d, seg);
                        else         std::memcpy(d, hp, seg);
                    }
                    continue;
                }
                // One strided copy per (layer, K/V): heads sit nb[2] apart on the
                // device and seg apart in the host segment. Same bytes as one
                // copy per head, n_head_kv times fewer launches.
                const cudaError_t err = to_host
                    ? cudaMemcpy2DAsync(host0, seg, dev0, t->nb[2], seg, (size_t)n_head_kv_,
                                        cudaMemcpyDeviceToHost, page_stream_)
                    : cudaMemcpy2DAsync(dev0, t->nb[2], host0, seg, seg, (size_t)n_head_kv_,
                                        cudaMemcpyHostToDevice, page_stream_);
                if (err != cudaSuccess) {
                    std::fprintf(stderr, "[kvflash] cudaMemcpy2DAsync(%s) failed: %s\n",
                                 to_host ? "D2H" : "H2D", cudaGetErrorString(err));
                }
            }
        }
        page_work_queued_ = true;
#else
        uint8_t * p = host;
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                for (int h = 0; h < n_head_kv_; h++) {
                    const size_t off = block_off_(block, t) + (size_t)h * t->nb[2];
                    if (to_host) ggml_backend_tensor_get(t, p, off, seg);
                    else         ggml_backend_tensor_set(t, p, off, seg);
                    p += seg;
                }
            }
        }
#endif
    }

    // Zero one pool block (defense-in-depth: stale K rows → exp(−max) ≈ 0
    // in maskless mode).  On CUDA/HIP builds, the memset is async on
    // page_stream_ so it is serialised after any preceding D2H for the same
    // block and before any subsequent H2D that reuses the slot.
    void zero_block(int block) {
        if (!has_tensor_storage()) return;
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                const size_t off0 = block_off_(block, t);
#ifdef KVFLASH_HAS_ASYNC_DMA
                uint8_t * dev0 = (uint8_t *)t->data + off0;
                if (is_host_tensor_(t)) {
                    for (int h = 0; h < n_head_kv_; h++) std::memset(dev0 + (size_t)h * t->nb[2], 0, seg);
                    continue;
                }
                const cudaError_t err = cudaMemset2DAsync(dev0, t->nb[2], 0, seg, (size_t)n_head_kv_, page_stream_);
                if (err != cudaSuccess) {
                    std::fprintf(stderr, "[kvflash] cudaMemset2DAsync failed: %s\n",
                                 cudaGetErrorString(err));
                }
#else
                for (int h = 0; h < n_head_kv_; h++)
                    ggml_backend_tensor_set(t, zero_buf_.data(), off0 + (size_t)h * t->nb[2], seg);
#endif
            }
        }
#ifdef KVFLASH_HAS_ASYNC_DMA
        page_work_queued_ = true;
#endif
    }

#ifdef KVFLASH_HAS_ASYNC_DMA
    // A failed stream or event call means ordering can no longer be guaranteed
    // (sticky error), and continuing could hand out an unfenced block: abort, like
    // CUDA_CHECK in ggml-cuda.
    void check_(cudaError_t err, const char * call) const {
        if (err != cudaSuccess)
            GGML_ABORT("[kvflash] %s failed: %s (pool %d tokens, chunk %d tokens)",
                       call, cudaGetErrorString(err), cfg_.pool_tokens, cfg_.chunk_tokens);
    }

    // Host wait for the queued page work. Host (CPU-backend) tensors are paged
    // with memcpy, so with no page stream and no device tensor nothing was
    // queued; with device tensors but no page stream the work ran on the
    // default stream.
    void sync_page_stream_() {
        if (page_stream_ || has_device_tensor_())
            check_(cudaStreamSynchronize(page_stream_), "cudaStreamSynchronize(page)");
        page_work_queued_ = false;
    }

    bool has_device_tensor_() const {
        for (const ggml_tensor * t : attn_k_) if (t && !is_host_tensor_(t)) return true;
        for (const ggml_tensor * t : attn_v_) if (t && !is_host_tensor_(t)) return true;
        return false;
    }
#endif

    // Order a block about to be reused after the page work queued on
    // page_stream_ (the page-out copy and zeroing of its old contents, or the
    // recall copy into it); otherwise the compute stream's write of the new
    // row can land before the copy (the archive then holds the new bytes) or
    // before the zeroing (the new row is wiped).
    // ponytail: without a registered compute stream this is a host wait, one
    // stall per new chunk after a page-out; registering the stream removes it.
    void fence_page_work_() {
#ifdef KVFLASH_HAS_ASYNC_DMA
        if (!page_work_queued_ || batch_submit_) return;
#ifdef KVFLASH_EVENT_FENCE
        if (compute_stream_ && page_event_) {
            check_(cudaEventRecord(page_event_, page_stream_), "cudaEventRecord(page)");
            check_(cudaStreamWaitEvent(compute_stream_, page_event_, 0), "cudaStreamWaitEvent(compute)");
            page_work_queued_ = false;
            return;
        }
#endif
        sync_page_stream_();
#endif
    }

    // Free one chunk's host backing store.
    void release_host_(ChunkState & st) {
#ifdef KVFLASH_HAS_ASYNC_DMA
        free_host_buffer_(st);
#else
        std::vector<uint8_t>().swap(st.host_data);
#endif
        st.on_host = false;
    }

    // Zero rows [row0, chunk_tokens) of chunk c, wherever its bytes live: the
    // pool block when resident, else the host copy. (A resident chunk's host
    // copy is stale by construction and overwritten on the next page_out.)
    void zero_chunk_rows_(int c, int row0) {
        if (!has_tensor_storage()) return;
        ChunkState & st = chunks_[(size_t)c];
        size_t host_off = 0;
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                const size_t row = seg / (size_t)cfg_.chunk_tokens;
                const size_t n = seg - (size_t)row0 * row;
                for (int h = 0; h < n_head_kv_; h++) {
                    if (st.block >= 0) {
                        st.dirty = true;   // device rows change; the host copy no longer equals the block
                        const size_t off = block_off_(st.block, t) +
                                           (size_t)h * t->nb[2] + (size_t)row0 * row;
#ifdef KVFLASH_HAS_ASYNC_DMA
                        if (is_host_tensor_(t)) std::memset((uint8_t *)t->data + off, 0, n);
                        else cudaMemsetAsync((uint8_t *)t->data + off, 0, n, page_stream_);
#else
                        ggml_backend_tensor_set(t, zero_buf_.data(), off, n);
#endif
                    } else if (st.on_host) {
#ifdef KVFLASH_HAS_ASYNC_DMA
                        std::memset((uint8_t *)st.host_data + host_off + (size_t)row0 * row, 0, n);
#else
                        std::memset(st.host_data.data() + host_off + (size_t)row0 * row, 0, n);
#endif
                    }
                    host_off += seg;
                }
            }
        }
    }

    // Release page_stream_ and all pinned host_data buffers.
    // Safe to call on a never-attached or already-cleaned-up instance.
    // Does NOT touch pool state (chunks_, free_blocks_) — that is the
    // caller's responsibility (reset() or destructor via caller).
    void cleanup_() {
#ifdef KVFLASH_HAS_ASYNC_DMA
        cudaStreamSynchronize(page_stream_);  // real stream, or default stream if create failed
        if (page_stream_ && page_stream_ != ext_page_stream_) cudaStreamDestroy(page_stream_);
        page_stream_ = nullptr;
#ifdef KVFLASH_EVENT_FENCE
        if (page_event_) {
            cudaEventDestroy(page_event_);
            page_event_ = nullptr;
        }
        compute_stream_ = nullptr;   // a reattach must not wait on a stream the caller has since destroyed
#endif
        page_work_queued_ = false;
        // Free pinned host buffers unconditionally: page_out() may have
        // allocated them even when stream creation failed (default-stream path).
        for (auto & st : chunks_) free_host_buffer_(st);
#endif
        has_pending_page_in_ = false;
    }

    // Byte offset of pool-local `block` in tensor t: the pool may be a slice
    // of a larger allocation (row_base_), and block ids stay pool-local.
    size_t block_off_(int block, const ggml_tensor * t) const {
        return ((size_t)row_base_ + (size_t)block * cfg_.chunk_tokens) * t->nb[1];
    }

    bool has_tensor_storage() const {
        return !attn_k_.empty() && chunk_bytes_ > 0;
    }

    KvFlashConfig cfg_;
    std::vector<ggml_tensor *> attn_k_, attn_v_;
    std::vector<ChunkState> chunks_;
    std::vector<int> free_blocks_;
    std::vector<std::pair<int, uint64_t>> pending_free_;   // (block, page-out record), deferred mode
    uint64_t record_seq_ = 0;
    bool deferred_free_ = false;
    bool batch_submit_ = false;
    uint64_t generation_ = 0;
    std::vector<uint8_t> zero_buf_;   // used by zero_block() in non-CUDA builds
    std::vector<uint8_t> pinned_;     // per-chunk pin flag; empty = no pins
    KvFlashStats stats_;
    size_t k_seg_bytes_ = 0, v_seg_bytes_ = 0, chunk_bytes_ = 0;
    int n_blocks_ = 0, n_head_kv_ = 0, cur_chunk_ = 0, row_base_ = 0;
    uint64_t clock_ = 0;
    uint64_t epoch_ = 0;
    uint32_t last_serialized_kv_k_type_ = (uint32_t)GGML_TYPE_F16;

#ifdef KVFLASH_HAS_ASYNC_DMA
    cudaStream_t page_stream_ = nullptr;
    cudaStream_t ext_page_stream_ = nullptr;   // arena-owned; see set_external_page_stream()
    bool         page_work_queued_ = false;   // work on page_stream_ no consumer has waited for
#ifdef KVFLASH_EVENT_FENCE
    cudaStream_t compute_stream_ = nullptr;   // consumer of the pool; see set_compute_stream()
    cudaEvent_t  page_event_ = nullptr;
#endif
#endif
    bool has_pending_page_in_ = false;
    bool pinned_host_ = true;
};

// ── Shared backend helpers ─────────────────────────────────────────────
//
// Every backend integration needs the same three steps: read the pool size
// from the env, allocate slots ahead of each forward (alloc_span above),
// and build slot-space inputs for the graph. The first and last live here
// so the per-arch code reduces to wiring.

// VRAM budget for "auto" pool sizing. Backends fill this AFTER the target
// weights are on the GPU and BEFORE the cache is allocated, so free_bytes
// reflects what the pool can actually use.
struct KvFlashAutoBudget {
    int64_t free_bytes      = 0;   // device free memory right now
    int64_t reserve_bytes   = 0;   // compute buffers + (if expected) drafter
    int64_t bytes_per_token = 0;   // pooled attention KV density for this model
    // Decode cost grows with the FA span (= the pool), so cap the auto pool
    // where speed stays near the small-pool point. Measured on the 27B/3090:
    // 1K pool 39.6 tok/s, 4K 38.7; 16K extrapolates to ~31-33, still 1.7-2.4x
    // the full cache at 128-256K. Override: LUCE_KVFLASH_MAX_POOL.
    int     speed_cap_tokens = 16384;
};

// The compatibility gate can reject a fixed KVFlash pool before model setup.
// "auto" is deliberately excluded: only the backend's VRAM-aware sizing can
// determine whether an automatic pool will actually be active.
inline bool kvflash_fixed_pool_requested(const char * value) {
    return kvflash_pool_requested(value) &&
           std::strcmp(value, "auto") != 0;
}

// Pool size from LUCE_KVFLASH for a backend with `cfg` protections:
// 0 = off; otherwise rounded to a 256 multiple, floored at
// min_pool_tokens(cfg) (eviction must keep a victim) and clamped to
// `max_ctx` (a pool larger than the logical context is meaningless), with
// warnings on both adjustments.
//
// The literal value "auto" sizes the pool from the GPU, not from a fixed
// fraction: take half of (free VRAM - reserve), convert to tokens at the
// model's KV density, then cap at the speed point and max_ctx. Big pools
// avoid relevance-crowding (more resident chunks = fewer forced evictions
// of useful context); the speed cap keeps decode near the flat optimum.
// Falls back to max_ctx/4 (scorer expected) or /2 (LRU) when the backend
// supplies no budget.
inline int kvflash_pool_from_env(int max_ctx, const KvFlashConfig & cfg = {},
                                 bool scorer_expected = false,
                                 const KvFlashAutoBudget & budget = {}) {
    const char * env = std::getenv("LUCE_KVFLASH");
    if (!env) return 0;
    int tokens;
    if (std::strcmp(env, "auto") == 0) {
        int speed_cap = budget.speed_cap_tokens;
        if (const char * mp = std::getenv("LUCE_KVFLASH_MAX_POOL")) {
            speed_cap = std::max(256, std::atoi(mp));
        }
        if (budget.bytes_per_token > 0 && budget.free_bytes > 0) {
            const int64_t usable =
                std::max<int64_t>(0, budget.free_bytes - budget.reserve_bytes) / 2;
            const int64_t vram_tokens = usable / budget.bytes_per_token;
            tokens = (int)std::min<int64_t>(vram_tokens,
                                            std::min(max_ctx, speed_cap));
            std::fprintf(stderr,
                "[kvflash] auto pool: %d tokens (free %.1f GiB - reserve %.1f GiB, "
                "%.1f KiB/token, caps: speed %d / max_ctx %d)\n",
                tokens, budget.free_bytes / 1073741824.0,
                budget.reserve_bytes / 1073741824.0,
                budget.bytes_per_token / 1024.0, speed_cap, max_ctx);
        } else {
            tokens = max_ctx / (scorer_expected ? 4 : 2);
            std::fprintf(stderr, "[kvflash] auto pool: %d tokens (%d%% of max_ctx %d, "
                                 "no VRAM budget supplied)\n",
                         tokens, scorer_expected ? 25 : 50, max_ctx);
        }
    } else {
        tokens = std::atoi(env);
    }
    if (tokens <= 0) return 0;
    tokens = ((tokens + 255) / 256) * 256;
    const int floor_tokens =
        ((KvFlashPager::min_pool_tokens(cfg) + 255) / 256) * 256;
    if (tokens < floor_tokens) {
        std::fprintf(stderr, "[kvflash] requested pool %d < minimum %d "
                             "(%d sink + %d tail chunks must leave an "
                             "evictable block); raising\n",
                     tokens, floor_tokens, cfg.sink_chunks, cfg.tail_window_chunks);
        tokens = floor_tokens;
    }
    if (tokens > max_ctx) {
        std::fprintf(stderr, "[kvflash] requested pool %d > max_ctx %d; clamping "
                             "(raise --max-ctx for a larger pool)\n",
                     tokens, max_ctx);
        tokens = (max_ctx / 256) * 256;
    }
    return tokens;
}

// Residency policy from LUCE_KVFLASH_POLICY (--kvflash-policy): "lru"
// forces recency-only paging (no drafter probe, no scorer); anything else
// (default "drafter") means scored residency when a drafter is available.
inline bool kvflash_policy_is_lru() {
    const char * env = std::getenv("LUCE_KVFLASH_POLICY");
    return env && std::strcmp(env, "lru") == 0;
}

// "qk": target-QK residency scoring (kvflash_qk.h) — pooled post-RoPE keys
// vs the current decode query, no drafter involved.
inline bool kvflash_policy_is_qk() {
    const char * env = std::getenv("LUCE_KVFLASH_POLICY");
    return env && std::strcmp(env, "qk") == 0;
}

// Locate the Qwen3.5-0.8B residency drafter: the explicit override
// (LUCE_KVFLASH_DRAFTER, set from --prefill-drafter), then the
// well-known locations next to the target model, then the appliance path.
// Returns "" when nothing is readable (callers fall back to LRU, loudly).
inline std::string kvflash_find_drafter(const char * target_path) {
    if (kvflash_policy_is_lru()) return "";
    if (const char * dp = std::getenv("LUCE_KVFLASH_DRAFTER")) return dp;
    if (!target_path) return "";
    std::string dir(target_path);
    const size_t slash = dir.find_last_of('/');
    dir = (slash == std::string::npos) ? "." : dir.substr(0, slash);
    const std::string candidates[] = {
        dir + "/Qwen3.5-0.8B-BF16.gguf",
        dir + "/drafter/Qwen3.5-0.8B-BF16.gguf",
        dir + "/draft/Qwen3.5-0.8B-BF16.gguf",
        "/opt/lucebox/models/drafter/Qwen3.5-0.8B-BF16.gguf",
    };
    for (const std::string & c : candidates) {
        if (std::FILE * f = std::fopen(c.c_str(), "rb")) {
            std::fclose(f);
            std::fprintf(stderr, "[kvflash] found residency drafter: %s\n", c.c_str());
            return c;
        }
    }
    return "";
}

// Serial-only consumers (identity sync, the single-mask upload) call this:
// a slot slice has no row-zero identity, so they stop with a clear message.
inline void kvflash_require_base_zero(const KvFlashPager & pager, const char * who) {
    if (pager.row_base() == 0) return;
    std::fprintf(stderr, "[kvflash] %s supports only a row_base 0 pager (got %d)\n", who, pager.row_base());
    std::abort();
}

// Slot-space step inputs for masked consumers: the K/V append row for each
// of this step's tokens, plus F32 causal (`mfull`) and sliding-window
// (`mswa`, optional) masks of width `mk_w` whose conditions are evaluated
// on the POSITION each pool slot holds (free slots stay -inf). The caller
// must have alloc_span()'d [kv_start, kv_start + n_tok) first. The pager
// zeroes freed slots, but the mask is what keeps relocation exact.
inline bool kvflash_fill_rows_and_masks(
    const KvFlashPager & pager,
    int kv_start, int n_tok, int mk_w, int swa_window,
    std::vector<int32_t> & rows,
    std::vector<float> * mfull, std::vector<float> * mswa) {
    rows.resize((size_t)n_tok);
    for (int i = 0; i < n_tok; ++i) {
        const int s = pager.slot_of(kv_start + i);   // arena row, the same space as the masks
        if (s < 0) {
            std::fprintf(stderr, "[kvflash] no pool slot at pos %d "
                                 "(alloc_span not called?)\n", kv_start + i);
            return false;
        }
        rows[(size_t)i] = s;
    }
    if (!mfull) return true;
    std::vector<int32_t> spos((size_t)pager.arena_tokens(), -1);
    pager.fill_slot_pos(spos.data());
    mfull->assign((size_t)mk_w * n_tok, -INFINITY);
    if (mswa) mswa->assign((size_t)mk_w * n_tok, -INFINITY);
    const int s_hi = std::min(mk_w, (int)spos.size());
    for (int q = 0; q < n_tok; ++q) {
        const int abs_q = kv_start + q;
        const int win_lo = std::max(0, abs_q - swa_window + 1);
        for (int s = 0; s < s_hi; ++s) {
            const int p = spos[(size_t)s];
            if (p < 0 || p > abs_q) continue;
            (*mfull)[(size_t)q * mk_w + s] = 0.0f;
            if (mswa && p >= win_lo) (*mswa)[(size_t)q * mk_w + s] = 0.0f;
        }
    }
    return true;
}

// RAII: restores the pager's score_hook to what it was at construction. A
// request that installs its own hook (prefill reselect) holds one so the hook
// of the arm's policy, or none (LRU), is back for the next request.
class ScopedScoreHook {
public:
    explicit ScopedScoreHook(KvFlashPager & p) : p_(p), saved_(p.score_hook) {}
    ~ScopedScoreHook() { p_.score_hook = std::move(saved_); }
    ScopedScoreHook(const ScopedScoreHook &) = delete;
    ScopedScoreHook & operator=(const ScopedScoreHook &) = delete;
private:
    KvFlashPager & p_;
    std::function<float(int)> saved_;
};

} // namespace luce::common
