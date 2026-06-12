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

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace dflash::common {

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
        if (attn_k.empty() || attn_k.size() != attn_v.size()) return false;
        cfg_ = cfg;
        attn_k_ = attn_k;
        attn_v_ = attn_v;
        n_blocks_ = cfg.pool_tokens / cfg.chunk_tokens;
        const ggml_tensor * K0 = attn_k[0];
        if ((int)K0->ne[1] < cfg.pool_tokens) return false;
        n_head_kv_ = (int)K0->ne[2];

        // Per-(tensor, head) contiguous segment of chunk_tokens rows.
        k_seg_bytes_ = (size_t)cfg.chunk_tokens * K0->nb[1];
        v_seg_bytes_ = (size_t)cfg.chunk_tokens * attn_v[0]->nb[1];
        chunk_bytes_ = (k_seg_bytes_ + v_seg_bytes_) * (size_t)n_head_kv_ * attn_k.size();
        zero_buf_.assign(std::max(k_seg_bytes_, v_seg_bytes_), 0);

        free_blocks_.clear();
        for (int b = n_blocks_ - 1; b >= 0; b--) free_blocks_.push_back(b);
        chunks_.clear();
        stats_ = {};
        clock_ = 0;
        return true;
    }

    // Optional: custom block hand-out order (e.g. shuffled placement in
    // relocation tests). `order[i]` = i-th block to hand out.
    void set_block_order(const std::vector<int> & order) {
        free_blocks_.assign(order.rbegin(), order.rend());
    }

    // Drop all mappings and host backing (new request / cache reset).
    // Cumulative stats are kept; the epoch advances so cached masks refill.
    void reset() {
        chunks_.clear();
        free_blocks_.clear();
        for (int b = n_blocks_ - 1; b >= 0; b--) free_blocks_.push_back(b);
        stats_.host_bytes = 0;
        cur_chunk_ = 0;
        epoch_++;
    }

    // Zero every currently-free block. reset() drops mappings but leaves the
    // previous request's bytes in place; maskless consumers (the qwen35moe
    // pipelined decode reads the whole padded pool span with no slot mask)
    // need stale rows to dequantise to ~zero contribution. Masked consumers
    // don't need this but it is cheap (pool-sized memset, sub-ms).
    void zero_free_blocks() {
        for (int b : free_blocks_) zero_block(b);
    }

    bool attached() const { return n_blocks_ > 0; }
    int pool_tokens() const { return cfg_.pool_tokens; }
    int chunk_tokens() const { return cfg_.chunk_tokens; }

    // Optional external relevance score; higher = keep. Falls back to LRU.
    std::function<float(int /*chunk*/)> score_hook;

    // Physical pool slot for logical position `pos`. Allocates (and, when
    // the pool is full, evicts) at chunk granularity. Call once per
    // appended token, in logical order.
    int slot_for(int64_t pos) {
        const int c = (int)(pos / cfg_.chunk_tokens);
        // cur_chunk_ tracks the append head only; a page_in of an older
        // chunk must not shrink the protected tail window.
        if (c > cur_chunk_) cur_chunk_ = c;
        if ((int)chunks_.size() <= c) chunks_.resize(c + 1);
        ChunkState & st = chunks_[c];
        if (st.block < 0) {
            if (!ensure_free_block()) return -1;
            st.block = free_blocks_.back();
            free_blocks_.pop_back();
            epoch_++;
            if (st.on_host) {              // recall: restore paged-out bytes
                copy_chunk(c, st.block, /*to_host=*/false);
                stats_.page_ins++;
                stats_.moved_bytes += chunk_bytes_;
            }
        }
        st.last_use = ++clock_;
        return st.block * cfg_.chunk_tokens + (int)(pos % cfg_.chunk_tokens);
    }

    // Force a chunk out of the pool (host backing + zeroed slots).
    bool page_out(int c) {
        if (c >= (int)chunks_.size() || chunks_[c].block < 0) return false;
        ChunkState & st = chunks_[c];
        if (!st.on_host) {
            st.host_data.resize(chunk_bytes_);
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

    bool is_resident(int c) const {
        return c < (int)chunks_.size() && chunks_[c].block >= 0;
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
                              c > cur_chunk_ - 1 - cfg_.tail_window_chunks;
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
        return events;
    }

private:
    struct ChunkState {
        int      block = -1;       // pool block index, -1 = not resident
        bool     on_host = false;  // backing store holds valid bytes
        uint64_t last_use = 0;
        std::vector<uint8_t> host_data;
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
            if (score_hook) {
                const float s = score_hook(c);
                if (victim < 0 || s < v_score) { victim = c; v_score = s; }
            } else {
                if (victim < 0 || chunks_[c].last_use < v_use) { victim = c; v_use = chunks_[c].last_use; }
            }
        }
        return victim >= 0 && page_out(victim);
    }

    // Move one chunk between pool slots and host backing. Segment order is
    // fixed (layer-major, K then V, head-minor) so offsets are stable.
    void copy_chunk(int c, int block, bool to_host) {
        ChunkState & st = chunks_[c];
        uint8_t * p = st.host_data.data();
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
    }

    void zero_block(int block) {
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                for (int h = 0; h < n_head_kv_; h++) {
                    const size_t off = (size_t)block * cfg_.chunk_tokens * t->nb[1] + (size_t)h * t->nb[2];
                    ggml_backend_tensor_set(t, zero_buf_.data(), off, seg);
                }
            }
        }
    }

    KvFlashConfig cfg_;
    std::vector<ggml_tensor *> attn_k_, attn_v_;
    std::vector<ChunkState> chunks_;
    std::vector<int> free_blocks_;
    std::vector<uint8_t> zero_buf_;
    KvFlashStats stats_;
    size_t k_seg_bytes_ = 0, v_seg_bytes_ = 0, chunk_bytes_ = 0;
    int n_blocks_ = 0, n_head_kv_ = 0, cur_chunk_ = 0;
    uint64_t clock_ = 0;
    uint64_t epoch_ = 0;
};

} // namespace dflash::common
