// KvFlashSlotKv: the SlotKv allocator over N attached slot
// pagers (one arena, one slice per slot). SeqSlotManager runs on it unchanged:
//   block        = one 64-token chunk of the slot's slice
//   rows         = arena rows from the slot pager's slot_for (inside the slice)
//   capacity     = the host reservation taken at acquire (KvFlashHostLedger)
//   append       = allocate the span; a full slice queues a page-out and returns
//                  BlocksExhausted until the arena completes it (the step plan's
//                  CompleteOut), then the retry succeeds
//   rollback/clear = pager truncate / reset; release = reset + ledger release
//   parkable     = false: a slot is never detached or recomputed
// Block tables are NOT this class's output: the engine writes each slot's
// column from fill_block_table (4 entries per chunk), so PagedKvAppendResult's
// physical_block is the arena chunk id and means nothing to the graph.
// Owns nothing: pagers and ledger must outlive it. Engine thread only.

#pragma once

#include "slot_kv.h"
#include "../kvflash_slot_pager.h"

#include <algorithm>
#include <vector>

namespace luce::common {

struct KvFlashSlotKvConfig {
    int quota_chunks = 0, headroom_blocks = 8;
    int64_t max_ctx = 0;
    int64_t chunk_bytes = 0;     // host backing per chunk; 0 = ask the pager
};

class KvFlashSlotKv final : public SlotKv {
public:
    KvFlashSlotKv(std::vector<KvFlashSlotPager *> pagers, KvFlashHostLedger & ledger, KvFlashSlotKvConfig cfg)
        : pagers_(std::move(pagers)), ledger_(ledger), cfg_(cfg), slots_(pagers_.size()) {}

    KvFlashSlotPager & pager(int slot) { return *pagers_[(size_t)slot]; }
    uint32_t block_size() const override { return (uint32_t)pagers_.front()->chunk_tokens(); }
    // Logical capacity: eviction lets a slot hold max_ctx tokens in a quota-sized slice.
    uint32_t physical_block_count() const override { return (uint32_t)pagers_.size() * logical_blocks_(); }
    uint32_t max_sequences() const override { return (uint32_t)pagers_.size(); }
    bool parkable() const override { return false; }
    uint32_t free_block_count() const override {
        uint32_t used = 0;
        for (const Slot & s : slots_) if (s.live) used += s.capacity_blocks;
        return physical_block_count() - used;
    }

    // The most tokens the NEXT acquire's request can hold, prompt + generation bound (one-shot, 0 after it):
    // host backing and slot capacity cover it. Without it they cover the acquire's token_capacity.
    void set_next_total_tokens(int64_t n) { next_total_ = std::max<int64_t>(0, n); }
    const KvFlashHostLedger & ledger() const { return ledger_; }

    PagedKvStatus acquire_reserved(PagedKvRequestId id, uint32_t token_capacity, PagedKvSequenceHandle & out) override {
        const int64_t total = next_total_;
        next_total_ = 0;
        int slot = -1;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].live && slots_[i].request == id) return PagedKvStatus::DuplicateRequest;
            if (!slots_[i].live && slot < 0) slot = (int)i;
        }
        if (slot < 0) return PagedKvStatus::SequenceSlotsExhausted;
        KvFlashSlotPager & p = *pagers_[(size_t)slot];
        const KvFlashAdmitInput in = admit_input_(p, charged_tokens_(token_capacity, total));
        const KvFlashAdmit v = ledger_.admit(id, in);
        if (v != KvFlashAdmit::Ok) return admit_status_(v);
        Slot & s = slots_[(size_t)slot];
        s = Slot{};
        p.reset();                // also drops a score hook left by the previous request
        s.live = true; s.request = id; s.generation = ++generation_;
        s.capacity_blocks = (uint32_t)((in.prompt_tokens + block_size() - 1) / block_size());
        out.slot = (uint32_t)slot; out.generation = s.generation;
        return PagedKvStatus::Ok;
    }

    PagedKvStatus reserve_capacity(PagedKvSequenceHandle h, uint32_t token_capacity) override {
        Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        // The host reservation is fixed at acquire; growth beyond it would need a ledger top-up.
        // The manager asks for cur_pos + one decode page; the page past the reservation is the rolling head, not backed.
        return (token_capacity + block_size() - 1) / block_size() <= s->capacity_blocks + 1 ? PagedKvStatus::Ok
                                                                                      : PagedKvStatus::BlocksExhausted;
    }

    PagedKvAppendResult append(PagedKvSequenceHandle h, uint32_t n, bool only_first_last = false) override {
        PagedKvAppendResult r;
        Slot * s = live_(h);
        if (!s) { r.status = PagedKvStatus::StaleHandle; return r; }
        if ((int64_t)s->len + n > (int64_t)s->capacity_blocks * block_size()) { r.status = PagedKvStatus::BlocksExhausted; return r; }
        KvFlashSlotPager & p = *pagers_[h.slot];
        std::vector<PagedKvWriteSlot> ws;
        ws.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            const int row = p.slot_for((int64_t)s->len + i);
            if (row < 0) {                       // out of blocks mid-span: len is not advanced; chunks already taken stay and the
                                                 // retry reuses them. No truncate: it would drop the queued page-out.
                r.status = PagedKvStatus::BlocksExhausted;
                return r;
            }
            PagedKvWriteSlot w;
            w.logical_position = s->len + i;
            w.physical_block = (uint32_t)p.block_of((int)(w.logical_position / block_size()));
            w.block_offset = w.logical_position % block_size();
            w.physical_token_index = (uint64_t)row;
            ws.push_back(w);
        }
        s->len += n;
        r.token_count = n;
        if (n) { r.first = ws.front(); r.last = ws.back(); }
        if (!only_first_last) r.write_slots = std::move(ws);
        return r;
    }

    PagedKvStatus rollback_append(PagedKvSequenceHandle h, uint32_t n) override {
        Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        if (n > s->len) return PagedKvStatus::InvalidArgument;
        s->len -= n;
        pagers_[h.slot]->truncate((int64_t)s->len);
        return PagedKvStatus::Ok;
    }

    PagedKvStatus clear(PagedKvSequenceHandle h) override {
        Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        s->len = 0;
        pagers_[h.slot]->reset();
        return PagedKvStatus::Ok;
    }

    PagedKvStatus release(PagedKvSequenceHandle h) override {
        Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        pagers_[h.slot]->reset();
        ledger_.release(s->request);
        s->live = false;
        return PagedKvStatus::Ok;
    }

    PagedKvStatus owned_block_count(PagedKvSequenceHandle h, uint32_t & out) const override {
        const Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        out = s->capacity_blocks;   // logical blocks reserved, what free_block_count() counts; evicted history is still owned
        return PagedKvStatus::Ok;
    }

private:
    struct Slot {
        bool live = false;
        PagedKvRequestId request = 0;
        uint64_t generation = 0;
        uint32_t len = 0, capacity_blocks = 0;
    };
    // The tokens an acquire is charged for. The rolling decode page may spill one block past max_ctx; anything more is a prompt too
    // long. `total`, when given, is the authoritative bound (the manager's extra decode page is not charged), clamped to max_ctx;
    // an overlong request keeps its own size so the ledger refuses it.
    int64_t charged_tokens_(int64_t token_capacity, int64_t total) const {
        const int64_t held = token_capacity <= cfg_.max_ctx + (int64_t)block_size() ? std::min<int64_t>(token_capacity, cfg_.max_ctx)
                                                                                   : token_capacity;
        return total > 0 && held <= cfg_.max_ctx ? std::min<int64_t>(total, cfg_.max_ctx) : held;
    }
    KvFlashAdmitInput admit_input_(const KvFlashSlotPager & p, int64_t tokens) const {
        KvFlashAdmitInput in;
        in.slot_free = true;
        in.quota_chunks = cfg_.quota_chunks; in.headroom_blocks = cfg_.headroom_blocks;
        in.prompt_tokens = tokens; in.max_new_tokens = 0; in.max_ctx = cfg_.max_ctx;
        in.chunk_tokens = p.chunk_tokens();
        in.chunk_bytes = cfg_.chunk_bytes ? cfg_.chunk_bytes : (int64_t)p.chunk_bytes();
        return in;
    }
    static PagedKvStatus admit_status_(KvFlashAdmit v) {
        switch (v) {
            case KvFlashAdmit::Ok: return PagedKvStatus::Ok;
            case KvFlashAdmit::BusySlot: return PagedKvStatus::SequenceSlotsExhausted;
            case KvFlashAdmit::BusyHost: return PagedKvStatus::BlocksExhausted;
            case KvFlashAdmit::HostTooSmall: return PagedKvStatus::CapacityExceeded;   // permanent: the request alone exceeds the whole host budget
            default: return PagedKvStatus::InvalidArgument;   // permanent: prompt too long, bad quota
        }
    }
    uint32_t logical_blocks_() const {
        return (uint32_t)((cfg_.max_ctx + block_size() - 1) / block_size());
    }
    Slot * live_(PagedKvSequenceHandle h) {
        return h.slot < slots_.size() && slots_[h.slot].live && slots_[h.slot].generation == h.generation ? &slots_[h.slot] : nullptr;
    }
    const Slot * live_(PagedKvSequenceHandle h) const { return const_cast<KvFlashSlotKv *>(this)->live_(h); }

    std::vector<KvFlashSlotPager *> pagers_;
    KvFlashHostLedger & ledger_;
    KvFlashSlotKvConfig cfg_;
    std::vector<Slot> slots_;
    uint64_t generation_ = 0;
    int64_t next_total_ = 0;
};

}  // namespace luce::common
