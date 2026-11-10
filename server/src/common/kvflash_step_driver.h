// KvFlashStepDriver: what the seq engine runs around the graph of one step. kvflash_preflight_step (D2) decides
// which slots start the step; run_step pins the chunks of their rows and allocates them.
//
// A step has no fence: a slot runs only when the blocks it needs are free and their page-out has completed, so its
// allocation queues no copy. Page-outs are queued a step ahead (settle, or the idle step of a slot that does not
// fit) and the next run_step waits for them once on the host before any block is handed out.
//
// Engine use: run_step before the graph; abort_step() on any failure after it;
// settle() after the argmax readback sync queues the page-outs that keep every
// live slot at its usable quota (Q - headroom), so the next step finds the
// free blocks it needs, and a prefill cut is not paced
// at one eviction per step. The step pins protect the step's own chunks and
// outlive the graph: the engine calls finish_step() after the readback. A
// slot's table column is re-uploaded only when table_stale(slot).
// Engine thread only; pagers and arena must outlive it.

#pragma once

#include "kvflash_arena.h"
#include "kvflash_slot_plan.h"
#include "kvflash_slot_pager.h"

#include <algorithm>
#include <vector>

namespace luce::common {

// D2: a slot starts a step only if the blocks it needs are free AND completed.
struct KvFlashPreflightSlot { int slot; int free_completed_blocks; int need_blocks; };

inline std::vector<int> kvflash_preflight_step(const std::vector<KvFlashPreflightSlot> & slots) {
    std::vector<int> ran;
    for (const auto & s : slots) if (s.free_completed_blocks >= s.need_blocks) ran.push_back(s.slot);
    return ran;
}

struct KvFlashStepWork { int slot; int64_t kv_start; int tokens; };   // rows the slot appends this step

class KvFlashStepDriver final {
public:
    KvFlashStepDriver(std::vector<KvFlashSlotPager *> pagers, KvFlashArena & arena, int headroom_blocks = 8)
        : pagers_(std::move(pagers)), arena_(arena), headroom_(headroom_blocks), uploaded_(pagers_.size(), ~uint64_t(0)) {}

    struct Result { bool ok = true; std::vector<int> ran; };

    // Settle the previous step's page-outs, preflight, then pin and allocate the rows of every slot that fits. A slot
    // that does not fit idles this step; the page-outs for the blocks it lacks
    // are queued now and the next call settles them (the recovery step).
    // ok=false: a page-out or an allocation failed and abort_step already ran.
    Result run_step(const std::vector<KvFlashStepWork> & work) {
        Result r;
        arena_.complete_all();
        std::vector<KvFlashPreflightSlot> pre;
        std::vector<KvFlashStepWork> rows;
        for (const auto & w : work) {
            if (w.tokens <= 0) continue;
            const KvFlashSlotPager & p = *pagers_[(size_t)w.slot];
            int new_chunks = 0;
            for (int c = first_chunk_(p, w); c <= last_chunk_(p, w); c++) new_chunks += p.block_of(c) < 0;
            // D2: a slot runs when its new chunks are free. A full slice is the normal state:
            // settle pages out what the next step allocates, a step ahead.
            pre.push_back({w.slot, p.n_pool_blocks() - p.resident_blocks() - p.pending_free_blocks(), new_chunks});
            rows.push_back(w);
        }
        r.ran = kvflash_preflight_step(pre);
        std::vector<KvFlashStepWork> fit;
        for (size_t i = 0; i < rows.size(); i++) {
            if (std::find(r.ran.begin(), r.ran.end(), rows[i].slot) != r.ran.end()) { fit.push_back(rows[i]); continue; }
            KvFlashSlotPager & p = *pagers_[(size_t)rows[i].slot];
            // D2: the slot pages out what this step lacks and runs when it has settled
            if (queue_page_outs_(p, pre[i].need_blocks - pre[i].free_completed_blocks - p.pending_free_blocks()) < 0) {
                abort_step();
                r.ok = false;
                return r;
            }
        }
        for (const auto & w : fit) {
            KvFlashSlotPager & p = *pagers_[(size_t)w.slot];
            const int64_t ct = p.chunk_tokens();
            for (int c = first_chunk_(p, w); c <= last_chunk_(p, w); c++) p.pin_step_range(c * ct, (c + 1) * ct);
        }
        for (const auto & w : fit) {
            if (pagers_[(size_t)w.slot]->alloc_span((int)w.kv_start, w.tokens)) continue;
            abort_step();   // D3: the first failure aborts the step, nothing runs after it
            r.ok = false;
            break;
        }
        return r;
    }

    // The step pins stay until here, after the readback.
    void finish_step() { for (KvFlashSlotPager * p : pagers_) p->end_step(); }
    void abort_step() { for (KvFlashSlotPager * p : pagers_) p->abort_step(); }

    int headroom() const { return headroom_; }

    // The one host wait that makes every queued page-out's block allocatable (before a truncate).
    void complete_all() { arena_.complete_all(); }

    // After the readback: bring each listed slot down to Q - headroom resident
    // blocks (LRU victims, async). Returns the page-outs queued, -1 if one failed (abort_step ran).
    int settle(const std::vector<int> & live_slots) {
        int queued = 0;
        for (int slot : live_slots) {
            KvFlashSlotPager & p = *pagers_[(size_t)slot];
            const int room = p.n_pool_blocks() - p.resident_blocks();
            const int n = queue_page_outs_(p, headroom_ - room);
            if (n < 0) { abort_step(); return -1; }
            queued += n;
        }
        return queued;
    }

    // The slot's column changed since it was last written.
    bool table_stale(int slot) const { return pagers_[(size_t)slot]->epoch() != uploaded_[(size_t)slot]; }
    void write_table(int slot, int32_t * dst, int n_chunks, int block_tokens) {
        pagers_[(size_t)slot]->fill_block_table(dst, n_chunks, block_tokens);
        uploaded_[(size_t)slot] = pagers_[(size_t)slot]->epoch();
    }

private:
    static int first_chunk_(const KvFlashSlotPager & p, const KvFlashStepWork & w) { return (int)(w.kv_start / p.chunk_tokens()); }
    static int last_chunk_(const KvFlashSlotPager & p, const KvFlashStepWork & w) {
        return (int)((w.kv_start + w.tokens - 1) / p.chunk_tokens());
    }

    // Page-outs queued for n blocks; -1 when one fails (a host buffer could not be allocated).
    static int queue_page_outs_(KvFlashSlotPager & p, int n) {
        int queued = 0;
        const uint64_t planned = p.generation();   // the victims are picked against this state
        if (n > 0) for (int c : p.lru_victims(n)) { if (!p.page_out_checked(c, planned)) return -1; queued++; }
        return queued;
    }

    std::vector<KvFlashSlotPager *> pagers_;
    KvFlashArena & arena_;
    int headroom_;
    std::vector<uint64_t> uploaded_;
};

}  // namespace luce::common
