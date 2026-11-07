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
//   park         = sticky session: the slot leaves the live set with its rows, pager and host reservation; a resume
//                  (set_next_resume) continues it at its length, any other acquire there drops it
// Block tables are NOT this class's output: the engine writes each slot's
// column from fill_block_table (4 entries per chunk), so PagedKvAppendResult's
// physical_block is the arena chunk id and means nothing to the graph.
// Owns nothing: pagers and ledger must outlive it. Engine thread only.

#pragma once

#include "slot_kv.h"
#include "../kvflash_pinned_head.h"
#include "../kvflash_slot_pager.h"
#include "../kvflash_slot_turn.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>

namespace luce::common {

struct KvFlashSlotKvConfig {
    int quota_chunks = 0, headroom_blocks = 8;
    int64_t max_ctx = 0;
    int64_t chunk_bytes = 0;     // host backing per chunk; 0 = ask the pager
    int64_t ladder_bytes = 0;    // per request
    int64_t stage_chunk_bytes = 0;   // per request, per chunk of its tokens (the park staging)
};

class KvFlashSlotKv final : public SlotKv {
public:
    KvFlashSlotKv(std::vector<KvFlashSlotPager *> pagers, KvFlashHostLedger & ledger, KvFlashSlotKvConfig cfg)
        : pagers_(std::move(pagers)), ledger_(ledger), cfg_(cfg), slots_(pagers_.size()) {}

    KvFlashSlotPager & pager(int slot) { return *pagers_[(size_t)slot]; }
    // The pinned heads' bindings. Every operation below that resets, truncates or replaces a slot's pager ends
    // with the binder's re-pin for that slot, so no engine path has to remember it. Null (the default): the
    // allocator is what it was.
    void set_binder(KvFlashHeadBinder * b) { binder_ = b; }
    uint32_t block_size() const override { return (uint32_t)pagers_.front()->chunk_tokens(); }
    // Logical capacity: eviction lets a slot hold max_ctx tokens in a quota-sized slice.
    uint32_t physical_block_count() const override { return (uint32_t)pagers_.size() * logical_blocks_(); }
    uint32_t max_sequences() const override { return (uint32_t)pagers_.size(); }
    bool parkable() const override { return false; }
    uint32_t free_block_count() const override {
        uint32_t used = 0;
        for (const Slot & s : slots_) if (s.live || s.parked) used += s.capacity_blocks;
        return physical_block_count() - used;
    }

    // The most tokens the NEXT acquire's request can hold, prompt + generation bound (one-shot, 0 after it):
    // host backing and slot capacity cover it. Without it they cover the acquire's token_capacity.
    void set_next_total_tokens(int64_t n) { next_total_ = std::max<int64_t>(0, n); }
    // Sticky sessions: the NEXT acquire takes this slot (one-shot). A parked session there is
    // dropped, unless set_next_resume names it: then the acquire keeps its rows and continues at `len`,
    // which must be the parked length, re-admitting the ledger for the new total.
    void set_next_slot(int slot) { next_slot_ = slot; next_resume_ = -1; }
    void set_next_resume(int slot, uint32_t len) { next_slot_ = slot; next_resume_ = (int64_t)len; }
    // Is the slot's last park still queued or being written? Its staging is part of the reservation, so a resume must not shrink it yet.
    void set_park_in_flight(std::function<bool(int)> f) { park_in_flight_ = std::move(f); }
    const KvFlashHostLedger & ledger() const { return ledger_; }
    int64_t reserved_bytes(int slot) const { return slot >= 0 && slot < (int)slots_.size() ? ledger_.reserved_of(slots_[(size_t)slot].request) : 0; }
    bool parked(int slot) const { return slot >= 0 && slot < (int)slots_.size() && slots_[(size_t)slot].parked; }
    void drop_parked(int slot) {
        if (!parked(slot)) return;
        Slot & s = slots_[(size_t)slot];
        ledger_.release(s.request);
        pagers_[(size_t)slot]->reset();
        s = Slot{};
        repin_(slot);
    }
    // A session restored from disk parks in a free slot: `len` tokens of rows from a KvFlashPager::serialize() blob, a
    // host reservation for them. The blob must hold exactly the chunks of `len` tokens; a refused blob leaves the slot
    // free and the ledger untouched.
    PagedKvStatus adopt_parked(int slot, PagedKvRequestId id, uint32_t len, const uint8_t * blob, size_t n) {
        if (slot < 0 || slot >= (int)slots_.size() || slots_[(size_t)slot].live || slots_[(size_t)slot].parked || len == 0)
            return PagedKvStatus::InvalidArgument;
        settle_floors_();
        KvFlashSlotPager & p = *pagers_[(size_t)slot];
        const KvFlashAdmitInput in = admit_input_(p, (int64_t)len);
        const KvFlashAdmit v = ledger_.admit(id, in);
        if (v != KvFlashAdmit::Ok) return admit_status_(v);
        const uint32_t blocks = (len + block_size() - 1) / block_size();
        if (!p.deserialize(blob, n, (int)blocks)) {   // the count is checked before any state moves
            p.reset();
            ledger_.release(id);
            repin_(slot);
            return PagedKvStatus::InvalidArgument;
        }
        Slot & s = slots_[(size_t)slot];
        s = Slot{};
        s.parked = true; s.request = id; s.admit_in = in; s.len = len; s.capacity_blocks = blocks; s.generation = ++generation_;
        repin_(slot);
        return PagedKvStatus::Ok;
    }
    // What adopt_parked(slot, id, len, ...) would answer on the ledger, with nothing charged and no blob needed: the
    // host-budget probe of a returning session before its bundle is read. InvalidArgument where adopt_parked would
    // refuse the slot or the length, so a caller treats anything but BlocksExhausted as "cannot decide".
    PagedKvStatus adopt_probe(int slot, PagedKvRequestId id, uint32_t len) {
        if (slot < 0 || slot >= (int)slots_.size() || slots_[(size_t)slot].live || slots_[(size_t)slot].parked || len == 0)
            return PagedKvStatus::InvalidArgument;
        settle_floors_();
        return admit_status_(ledger_.would_admit(id, admit_input_(*pagers_[(size_t)slot], (int64_t)len)));
    }
    // Does a request of `total` tokens (prompt + generation bound) alone exceed the whole host budget? Permanent: no eviction
    // makes room, and a cold admit needs the same bytes. Asked before a resume; charges and moves nothing.
    // The charge is the one acquire_reserved makes (charged_tokens_), so a total of max_ctx + 1 is charged max_ctx here too. A prompt
    // past max_ctx never gets here as a ledger refusal: SeqSlotManager::admit_at refuses it first ("prompt exceeds max_ctx").
    bool exceeds_host_budget(int slot, int64_t total) const {
        if (slot < 0 || slot >= (int)slots_.size() || total <= 0) return false;
        return ledger_.would_admit_new(admit_input_(*pagers_[(size_t)slot], charged_tokens_(0, total))) == KvFlashAdmit::HostTooSmall;
    }
    // A parked session drops its rows past `len` (the resume at a checkpoint below its length).
    bool truncate_parked(int slot, uint32_t len) {
        if (!parked(slot) || len > slots_[(size_t)slot].len) return false;
        slots_[(size_t)slot].len = len;
        pagers_[(size_t)slot]->truncate((int64_t)len);
        repin_(slot);
        return true;
    }

    // A resume's reservation floor (the staging of its last park, still queued then) ends with the writer: the session is charged its own
    // need from the next admission on, whether it is live or parked. Idempotent: a settled slot has no floor left.
    void settle_floors_() {
        if (!park_in_flight_) return;
        for (size_t i = 0; i < slots_.size(); ++i) {
            Slot & s = slots_[i];
            if ((!s.live && !s.parked) || s.admit_in.reserve_floor_bytes == 0 || park_in_flight_((int)i)) continue;
            s.admit_in.reserve_floor_bytes = 0;
            ledger_.release(s.request);
            ledger_.admit(s.request, s.admit_in);   // never larger than what it held
        }
    }

    PagedKvStatus acquire_reserved(PagedKvRequestId id, uint32_t token_capacity, PagedKvSequenceHandle & out) override {
        settle_floors_();
        const int64_t total = next_total_, resume = next_resume_;
        int slot = next_slot_;
        next_total_ = 0; next_slot_ = -1; next_resume_ = -1;
        for (const Slot & s : slots_) if (s.live && s.request == id) return PagedKvStatus::DuplicateRequest;
        if (slot >= 0) {
            if (slot >= (int)slots_.size() || slots_[(size_t)slot].live) return PagedKvStatus::InvalidArgument;
            if (resume >= 0) {
                if (!slots_[(size_t)slot].parked || (int64_t)slots_[(size_t)slot].len != resume) return PagedKvStatus::InvalidArgument;
            } else {
                drop_parked(slot);
            }
        } else {
            for (size_t i = 0; i < slots_.size() && slot < 0; ++i) if (!slots_[i].live && !slots_[i].parked) slot = (int)i;
        }
        if (slot < 0) return PagedKvStatus::SequenceSlotsExhausted;
        KvFlashSlotPager & p = *pagers_[(size_t)slot];
        KvFlashAdmitInput in = admit_input_(p, charged_tokens_(token_capacity, total));
        Slot & s = slots_[(size_t)slot];
        // the slot's last park may still be queued: its staging is inside the reservation, which then does not shrink yet
        if (resume >= 0 && park_in_flight_ && park_in_flight_(slot)) in.reserve_floor_bytes = ledger_.reserved_of(s.request);
        if (resume >= 0) ledger_.release(s.request);   // the parked reservation makes room for the new total
        const KvFlashAdmit v = ledger_.admit(id, in);
        if (v != KvFlashAdmit::Ok) {
            if (resume >= 0) ledger_.admit(s.request, s.admit_in);   // the session stays parked as it was
            return admit_status_(v);
        }
        if (resume >= 0) {
            s.parked = false;             // rows and pager stay
        } else {
            s = Slot{};
            p.reset();                // also drops a score hook left by the previous request
            repin_(slot);
        }
        s.live = true; s.request = id; s.generation = ++generation_; s.admit_in = in;
        s.capacity_blocks = (uint32_t)((in.prompt_tokens + block_size() - 1) / block_size());
        s.base_page_outs = p.stats().page_outs; s.base_page_ins = p.stats().page_ins;   // a resume keeps its pager: the counts are this request's
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

    // Pages the live (or just parked) request pushed out since it was admitted; 0 for a free or unknown slot.
    int64_t page_outs(int slot) const {
        if (slot < 0 || slot >= (int)slots_.size() || (!slots_[(size_t)slot].live && !slots_[(size_t)slot].parked)) return 0;
        return pagers_[(size_t)slot]->stats().page_outs - slots_[(size_t)slot].base_page_outs;
    }
    // Rows a live slot may still append inside its reservation (0 for a slot that is free, parked or out of range).
    // A chain round accepts up to its width, which can pass the last reserved block when the request is nearly done.
    int64_t append_room(int slot) const {
        if (slot < 0 || slot >= (int)slots_.size() || !slots_[(size_t)slot].live) return 0;
        const Slot & s = slots_[(size_t)slot];
        return (int64_t)s.capacity_blocks * block_size() - (int64_t)s.len;
    }

    // A chain round plans rows for its whole width before verify and accepts a prefix: a chunk opened only by rows that
    // were not accepted is dropped, so the pager holds exactly ceil(len/block) chunks (a park records n_chunks and an
    // adopt checks that count; the next crossing into the chunk must allocate and upload afresh). True when it trimmed.
    bool trim_to_len(int slot) {
        if (slot < 0 || slot >= (int)slots_.size() || !slots_[(size_t)slot].live) return false;
        const uint32_t len = slots_[(size_t)slot].len;
        KvFlashSlotPager & p = *pagers_[(size_t)slot];
        const int keep = (int)((len + block_size() - 1) / block_size());
        if (len == 0 || p.n_chunks() <= keep) return false;
        p.truncate((int64_t)len);
        repin_(slot);
        return true;
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
        repin_((int)h.slot);
        return PagedKvStatus::Ok;
    }

    PagedKvStatus clear(PagedKvSequenceHandle h) override {
        Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        s->len = 0;
        pagers_[h.slot]->reset();
        repin_((int)h.slot);
        return PagedKvStatus::Ok;
    }

    PagedKvStatus release(PagedKvSequenceHandle h) override {
        Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        {
            const KvFlashStats & st = pagers_[h.slot]->stats();   // stats are cumulative per pager; reset() clears them, so read before
            std::fprintf(stderr, "[kvflash] slot %u release: len=%u page_outs=%lld page_ins=%lld resident=%d\n",
                         h.slot, s->len, (long long)(st.page_outs - s->base_page_outs),
                         (long long)(st.page_ins - s->base_page_ins), pagers_[h.slot]->resident_blocks());
        }
        pagers_[h.slot]->reset();
        ledger_.release(s->request);
        s->live = false;
        repin_((int)h.slot);
        return PagedKvStatus::Ok;
    }

    // Sticky session: the slot leaves the live set with its rows, pager state and
    // host reservation; a resume at the same slot continues at len.
    PagedKvStatus park(PagedKvSequenceHandle h) override {
        Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        pagers_[h.slot]->score_hook = nullptr;   // the turn's scores die with the request
        s->live = false;
        s->parked = true;
        return PagedKvStatus::Ok;
    }

    PagedKvStatus owned_block_count(PagedKvSequenceHandle h, uint32_t & out) const override {
        const Slot * s = live_(h);
        if (!s) return PagedKvStatus::StaleHandle;
        out = s->capacity_blocks;   // logical blocks reserved, what free_block_count() counts; evicted history is still owned
        return PagedKvStatus::Ok;
    }

private:
    void repin_(int slot) { if (binder_) binder_->ensure(slot, *pagers_[(size_t)slot]); }
    struct Slot {
        bool live = false, parked = false;
        PagedKvRequestId request = 0;
        KvFlashAdmitInput admit_in;   // the ledger reservation, re-admitted when a resume is refused
        uint64_t generation = 0;
        uint32_t len = 0, capacity_blocks = 0;
        int64_t base_page_outs = 0, base_page_ins = 0;
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
        in.ladder_bytes = cfg_.ladder_bytes;
        in.stage_chunk_bytes = cfg_.stage_chunk_bytes;
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
    std::function<bool(int)> park_in_flight_;
    KvFlashHostLedger & ledger_;
    KvFlashSlotKvConfig cfg_;
    std::vector<Slot> slots_;
    uint64_t generation_ = 0;
    int64_t next_total_ = 0, next_resume_ = -1;
    int next_slot_ = -1;
    KvFlashHeadBinder * binder_ = nullptr;
};

// What the engine's adopt makes of the SlotKv's answer: a full host budget is Busy (the caller evicts and retries), any other refusal is final.
inline KvFlashAdopt kvflash_adopt_of(PagedKvStatus st) {
    return st == PagedKvStatus::Ok ? KvFlashAdopt::Ok : st == PagedKvStatus::BlocksExhausted ? KvFlashAdopt::Busy : KvFlashAdopt::Refused;
}

}  // namespace luce::common
