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
};

struct KvFlashStats {
    int64_t page_outs  = 0;
    int64_t page_ins   = 0;
    int64_t host_bytes = 0;   // backing store currently held on host
    int64_t moved_bytes = 0;  // cumulative D2H+H2D traffic
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
        if (cfg.pool_tokens < min_pool_tokens(cfg)) {
            std::fprintf(stderr,
                "kvflash: pool %d < minimum %d (%d sink + %d tail chunks must "
                "leave an evictable block)\n",
                cfg.pool_tokens, min_pool_tokens(cfg),
                cfg.sink_chunks, cfg.tail_window_chunks);
            return false;
        }
        if (attn_k.size() != attn_v.size()) return false;

        cleanup_();   // release pinned buffers + stream from any prior attach

        cfg_ = cfg;
        attn_k_ = attn_k;
        attn_v_ = attn_v;
        n_blocks_ = cfg.pool_tokens / cfg.chunk_tokens;
        if (!attn_k.empty()) {
            const ggml_tensor * K0 = attn_k[0];
            if ((int)K0->ne[1] < cfg.pool_tokens) return false;
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
        for (int b = n_blocks_ - 1; b >= 0; b--) free_blocks_.push_back(b);
        chunks_.clear();
        pinned_.clear();   // pins belong to the previous pool geometry
        stats_ = {};
        clock_ = 0;
        has_pending_page_in_ = false;

#ifdef KVFLASH_HAS_ASYNC_DMA
        if (cudaStreamCreate(&page_stream_) != cudaSuccess) {
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
    void set_block_order(const std::vector<int> & order) {
        free_blocks_.assign(order.rbegin(), order.rend());
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
        refill_after_truncate_();
        synchronize_paging();   // recalls are queued on page_stream_
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
    void pin_range(int64_t tok_lo, int64_t tok_hi) {
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
        for (uint8_t f : pinned_) pinned += f;
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
        for (int c = c_lo; c <= c_hi; c++) pinned_[(size_t)c] = 1;
    }
    bool is_pinned(int c) const {
        return c >= 0 && c < (int)pinned_.size() && pinned_[(size_t)c];
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
        if (has_pending_page_in_) synchronize_paging();
        return true;
    }

    // Physical pool slot for logical position `pos`. Allocates (and, when
    // the pool is full, evicts) at chunk granularity. Call once per
    // appended token, in logical order.
    int slot_for(int64_t pos) {
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
        st.last_use = ++clock_;
        return st.block * cfg_.chunk_tokens + (int)(pos % cfg_.chunk_tokens);
    }

    // Force a chunk out of the pool (host backing + zeroed slots).
    bool page_out(int c) {
        if (c >= (int)chunks_.size() || chunks_[c].block < 0) return false;
        ChunkState & st = chunks_[c];
        if (has_tensor_storage() && !st.on_host) {
#ifdef KVFLASH_HAS_ASYNC_DMA
            if (!alloc_host_buffer_(st)) return false;
#else
            st.host_data.resize(chunk_bytes_);
#endif
            stats_.host_bytes += (int64_t)chunk_bytes_;
        }
        copy_chunk(c, st.block, /*to_host=*/true);
        zero_block(st.block);
        st.on_host = true;
        free_blocks_.push_back(st.block);
        st.block = -1;
        epoch_++;
        stats_.page_outs++;
        stats_.moved_bytes += chunk_bytes_;
        return true;
    }

    // Recall a chunk into the pool (used by reselect / tests).
    bool page_in(int c) {
        if (c >= (int)chunks_.size() || !chunks_[c].on_host || chunks_[c].block >= 0) return false;
        return slot_for((int64_t)c * cfg_.chunk_tokens) >= 0;
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
    }

    bool is_resident(int c) const {
        return c < (int)chunks_.size() && chunks_[c].block >= 0;
    }

    // True while every materialized chunk still sits in its identity block
    // (chunk c in block c, nothing paged out). This is the layout contract
    // identity-copy snapshots rely on; it holds from reset() until the
    // first eviction of the CURRENT request (cumulative stats do not).
    bool is_identity() const {
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
        if (n_tok <= 0) return true;
        const int nc = (n_tok + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens;
        if (nc > (int)chunks_.size()) return false;
        for (int c = 0; c < nc; c++)
            if (chunks_[c].block != c) return false;
        return true;
    }
    int block_of(int c) const {
        return c < (int)chunks_.size() ? chunks_[c].block : -1;
    }

    // Const lookup (no alloc / LRU touch): physical slot currently holding
    // logical `pos`, or -1 if its chunk is not resident. Callers that may
    // need an allocation must use slot_for() beforehand.
    int slot_of(int64_t pos) const {
        const int c = (int)(pos / cfg_.chunk_tokens);
        if (c >= (int)chunks_.size() || chunks_[c].block < 0) return -1;
        return chunks_[c].block * cfg_.chunk_tokens + (int)(pos % cfg_.chunk_tokens);
    }

    // Logical position held by each pool slot, -1 for free blocks. `dst`
    // must hold pool_tokens entries. Lets callers build masks that need
    // POSITION semantics in slot space (causal / sliding-window): the
    // mask condition is evaluated on dst[slot] instead of the column index.
    void fill_slot_pos(int32_t * dst) const {
        for (int i = 0; i < cfg_.pool_tokens; i++) dst[i] = -1;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            if (chunks_[c].block < 0) continue;
            int32_t * p = dst + (size_t)chunks_[c].block * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; i++)
                p[i] = (int32_t)c * cfg_.chunk_tokens + i;
        }
    }
    const KvFlashStats & stats() const { return stats_; }
    int resident_blocks() const { return n_blocks_ - (int)free_blocks_.size(); }
    int n_chunks() const { return (int)chunks_.size(); }

    // Bumped on every residency change (alloc / page_out / page_in).
    // Callers cache the slot mask and refill only when the epoch moves.
    uint64_t epoch() const { return epoch_; }

    // F16 slot-validity mask for one query row: 0 for slots belonging to a
    // resident chunk, -inf for free / paged-out blocks. `dst` must hold
    // pool_tokens entries. Used as the FA mask so non-resident slots are
    // excluded exactly instead of via the zero-row ~exp(-max) approximation.
    void fill_slot_mask(uint16_t * dst) const {
        constexpr uint16_t F16_ZERO = 0x0000, F16_NEG_INF = 0xFC00;
        for (int i = 0; i < cfg_.pool_tokens; i++) dst[i] = F16_NEG_INF;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            if (chunks_[c].block < 0) continue;
            uint16_t * p = dst + (size_t)chunks_[c].block * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; i++) p[i] = F16_ZERO;
        }
    }

    // Lookahead reselect (FlashMemory τ-step): rebuild the resident set as
    // the top-pool chunks by score_hook among ALL known chunks (resident or
    // host-backed). Sinks and the trailing window are always kept. Returns
    // the number of page events. Call between decode steps.
    int reselect() {
        if (!score_hook) return 0;
        struct Cand { int c; float s; };
        std::vector<Cand> cands;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            const ChunkState & st = chunks_[c];
            if (st.block < 0 && !st.on_host) continue;     // never materialized
            const bool prot = c < cfg_.sink_chunks ||
                              c > cur_chunk_ - 1 - cfg_.tail_window_chunks ||
                              is_pinned(c);
            cands.push_back({c, prot ? 3.4e38f : score_hook(c)});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand & a, const Cand & b) { return a.s > b.s; });
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
        uint64_t h = 1469598103934665603ULL;
        for (int c = 0; c < upto && c < (int)chunks_.size(); c++) {
            uint32_t b; const float sc = chunks_[(size_t)c].score;
            std::memcpy(&b, &sc, 4);
            h = (h ^ b) * 1099511628211ULL;
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
    // Layout, magic "KVFLASH2" (u64); all integers and floats are written in host
    // byte order, which the static_assert at the top of this file pins to little-endian:
    //   9 x uint32 header: nc, chunk_tokens, n_head_kv, k_seg_bytes,
    //                      v_seg_bytes, chunk_bytes, ggml_type(K), has_ledger(=1),
    //                      ggml_type(V)
    //   nc x chunk_bytes   KV bytes, segment order of copy_chunk()
    //   nc x 8 bytes       ledger: u8 was_resident, u8 materialized, 2 pad, f32 score
    //                      (a hole left by a failed allocation is not materialized)
    // Not serialized: physical block numbers, LRU order, pins.
    std::vector<uint8_t> serialize(int max_chunks = -1) const {
        synchronize_page_stream_();   // queued D2H must land in host_data first
        const int nc = (max_chunks >= 0 && max_chunks < (int)chunks_.size())
                       ? max_chunks : (int)chunks_.size();
        const size_t kv_bytes = (size_t)nc * chunk_bytes_;
        std::vector<uint8_t> out(kBlobHdr + kv_bytes + (size_t)nc * kLedgerEntry, 0);
        uint8_t * p = out.data();
        std::memcpy(p, &kBlobMagic, 8); p += 8;
        const uint32_t hdr[9] = {(uint32_t)nc, (uint32_t)cfg_.chunk_tokens,
            (uint32_t)n_head_kv_, (uint32_t)k_seg_bytes_, (uint32_t)v_seg_bytes_,
            (uint32_t)chunk_bytes_, live_k_type_(), 1u, live_v_type_()};
        std::memcpy(p, hdr, sizeof(hdr));
        for (int c = 0; c < nc; ++c) {
            const ChunkState & st = chunks_[(size_t)c];
            uint8_t * dst = out.data() + kBlobHdr + (size_t)c * chunk_bytes_;
            if (st.block >= 0 && has_tensor_storage()) {
                for (size_t l = 0; l < attn_k_.size(); ++l) {
                    for (int kv = 0; kv < 2; ++kv) {
                        ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                        const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                        for (int h = 0; h < n_head_kv_; ++h) {
                            const size_t off = (size_t)st.block * cfg_.chunk_tokens * t->nb[1] +
                                               (size_t)h * t->nb[2];
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
            uint8_t * lp = out.data() + kBlobHdr + kv_bytes + (size_t)c * kLedgerEntry;
            lp[0] = st.block >= 0 ? 1u : 0u;
            lp[1] = (st.block >= 0 || st.on_host) ? 1u : 0u;
            std::memcpy(lp + 4, &st.score, 4);
        }
        return out;
    }

    // Replace the pager state with a serialize() blob. The blob is fully
    // validated (magic, geometry, K and V dtype, length, resident count <= pool)
    // BEFORE any state is touched: a rejected blob returns false and leaves
    // the pager as it was. On success: reset() semantics (pins cleared),
    // chunks that were resident get a pool block (identity order, bytes
    // restored), the rest stay host-backed and recallable via page_in(), and
    // scores are restored. Pool slot numbers and LRU order are NOT preserved,
    // only the logical residency set. Callers must rebuild slot masks (epoch
    // advances). Returns false (pager reset/empty) only if a pinned host
    // buffer cannot be allocated.
    bool deserialize(const uint8_t * data, size_t n) {
        if (!attached() || !data || n < kBlobHdr) return false;
        uint64_t magic = 0;
        std::memcpy(&magic, data, 8);
        if (magic != kBlobMagic) return false;
        uint32_t h[9];
        std::memcpy(h, data + 8, sizeof(h));
        const size_t nc = h[0];
        if ((int)h[1] != cfg_.chunk_tokens || (int)h[2] != n_head_kv_ ||
            h[3] != k_seg_bytes_ || h[4] != v_seg_bytes_ || h[5] != chunk_bytes_ ||
            h[6] != live_k_type_() || h[7] != 1u || h[8] != live_v_type_()) {
            return false;
        }
        // Overflow-safe length check: nc comes from the (untrusted) blob.
        if (nc > (n - kBlobHdr) / (chunk_bytes_ + kLedgerEntry)) return false;
        const uint8_t * kv = data + kBlobHdr;
        const uint8_t * lp = kv + nc * chunk_bytes_;
        size_t n_resident = 0;
        for (size_t c = 0; c < nc; ++c) {
            const uint8_t res = lp[c * kLedgerEntry], mat = lp[c * kLedgerEntry + 1];
            if (mat > 1 || (res != 0 && mat == 0)) return false;   // resident implies materialized
            n_resident += res != 0;
        }
        if (n_resident > (size_t)n_blocks_) return false;

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
            // slot_for()'s recall branch copies host_data into the new block.
            if (lp[c * kLedgerEntry] != 0) slot_for((int64_t)c * cfg_.chunk_tokens);
        }
        if (nc > 0 && cur_chunk_ < (int)nc - 1) cur_chunk_ = (int)nc - 1;
        if (has_pending_page_in_) synchronize_paging();   // async recalls settle
        return true;
    }

private:
    static constexpr uint64_t kBlobMagic = 0x4b56464c41534832ULL;  // "KVFLASH2"
    static constexpr size_t kBlobHdr = sizeof(uint64_t) + 9 * sizeof(uint32_t);
    static constexpr size_t kLedgerEntry = 8;
    static constexpr int kMaxPinChunks = 1 << 22;   // pin flags stay <= 4 MiB

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
        // Victim: unprotected resident chunk with the lowest score
        // (score_hook) or the oldest use (LRU fallback).
        int victim = -1;
        float v_score = 0.f;
        uint64_t v_use = 0;
        for (int c = 0; c < (int)chunks_.size(); c++) {
            if (chunks_[c].block < 0) continue;
            if (c < cfg_.sink_chunks) continue;
            if (c > cur_chunk_ - 1 - cfg_.tail_window_chunks) continue;
            if (is_pinned(c)) continue;
            if (score_hook) {
                const float s = score_hook(c);
                if (victim < 0 || s < v_score) { victim = c; v_score = s; }
            } else {
                if (victim < 0 || chunks_[c].last_use < v_use) { victim = c; v_use = chunks_[c].last_use; }
            }
        }
        return victim >= 0 && page_out(victim);
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
        if (!has_tensor_storage()) return;
        ChunkState & st = chunks_[c];
#ifdef KVFLASH_HAS_ASYNC_DMA
        size_t host_off = 0;
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                for (int h = 0; h < n_head_kv_; h++) {
                    const size_t dev_off =
                        (size_t)block * cfg_.chunk_tokens * t->nb[1] +
                        (size_t)h * t->nb[2];
                    void * host_ptr = (uint8_t *)st.host_data + host_off;
                    void * dev_ptr  = (uint8_t *)t->data + dev_off;
                    if (is_host_tensor_(t)) {  // CPU-backend tensor: plain memcpy
                        if (to_host) std::memcpy(host_ptr, dev_ptr, seg);
                        else         std::memcpy(dev_ptr, host_ptr, seg);
                        host_off += seg;
                        continue;
                    }
                    cudaError_t err;
                    if (to_host)
                        err = cudaMemcpyAsync(host_ptr, dev_ptr, seg,
                                        cudaMemcpyDeviceToHost, page_stream_);
                    else
                        err = cudaMemcpyAsync(dev_ptr, host_ptr, seg,
                                        cudaMemcpyHostToDevice, page_stream_);
                    if (err != cudaSuccess) {
                        std::fprintf(stderr, "[kvflash] cudaMemcpyAsync(%s) failed: %s\n",
                                     to_host ? "D2H" : "H2D", cudaGetErrorString(err));
                    }
                    host_off += seg;
                }
            }
        }
        page_work_queued_ = true;
#else
        uint8_t * p = (uint8_t *)st.host_data.data();
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                for (int h = 0; h < n_head_kv_; h++) {
                    const size_t off = (size_t)block * cfg_.chunk_tokens * t->nb[1] + (size_t)h * t->nb[2];
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
                for (int h = 0; h < n_head_kv_; h++) {
                    const size_t off = (size_t)block * cfg_.chunk_tokens * t->nb[1] + (size_t)h * t->nb[2];
#ifdef KVFLASH_HAS_ASYNC_DMA
                    if (is_host_tensor_(t)) { std::memset((uint8_t *)t->data + off, 0, seg); continue; }
                    cudaError_t err = cudaMemsetAsync((uint8_t *)t->data + off, 0, seg, page_stream_);
                    if (err != cudaSuccess) {
                        std::fprintf(stderr, "[kvflash] cudaMemsetAsync failed: %s\n",
                                     cudaGetErrorString(err));
                    }
#else
                    ggml_backend_tensor_set(t, zero_buf_.data(), off, seg);
#endif
                }
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

    // Host wait for the queued page work. With no tensor storage nothing was
    // queued; with device tensors but no page stream the work ran on the
    // default stream.
    void sync_page_stream_() {
        if (page_stream_ || !attn_k_.empty())
            check_(cudaStreamSynchronize(page_stream_), "cudaStreamSynchronize(page)");
        page_work_queued_ = false;
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
        if (!page_work_queued_) return;
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
                        const size_t off = (size_t)st.block * cfg_.chunk_tokens * t->nb[1] +
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
        if (page_stream_) {
            cudaStreamDestroy(page_stream_);
            page_stream_ = nullptr;
        }
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

    bool has_tensor_storage() const {
        return !attn_k_.empty() && chunk_bytes_ > 0;
    }

    KvFlashConfig cfg_;
    std::vector<ggml_tensor *> attn_k_, attn_v_;
    std::vector<ChunkState> chunks_;
    std::vector<int> free_blocks_;
    std::vector<uint8_t> zero_buf_;   // used by zero_block() in non-CUDA builds
    std::vector<uint8_t> pinned_;     // per-chunk pin flag; empty = no pins
    KvFlashStats stats_;
    size_t k_seg_bytes_ = 0, v_seg_bytes_ = 0, chunk_bytes_ = 0;
    int n_blocks_ = 0, n_head_kv_ = 0, cur_chunk_ = 0;
    uint64_t clock_ = 0;
    uint64_t epoch_ = 0;
    uint32_t last_serialized_kv_k_type_ = (uint32_t)GGML_TYPE_F16;

#ifdef KVFLASH_HAS_ASYNC_DMA
    cudaStream_t page_stream_ = nullptr;
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

// Whether the operator requested any KVFlash pool. "auto" is a real request
// even though its final size is not known until the backend measures VRAM.
inline bool kvflash_pool_requested(const char * value) {
    return value != nullptr &&
           (std::strcmp(value, "auto") == 0 || std::atoi(value) > 0);
}

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
        const int s = pager.slot_of(kv_start + i);
        if (s < 0) {
            std::fprintf(stderr, "[kvflash] no pool slot at pos %d "
                                 "(alloc_span not called?)\n", kv_start + i);
            return false;
        }
        rows[(size_t)i] = s;
    }
    if (!mfull) return true;
    std::vector<int32_t> spos((size_t)pager.pool_tokens(), -1);
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

} // namespace luce::common
