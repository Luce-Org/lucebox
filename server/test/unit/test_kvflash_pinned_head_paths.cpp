// KVFlash pinned head: every engine path keeps the head's pins, CPU only. The allocator (KvFlashSlotKv over slot
// pagers) owns the re-pin through the binder it is given, so these tests never call the re-pin themselves: they
// drive the operations each engine path performs and read pins and residency.
//
//   capture-on-miss   cold prefill to the cut, capture (binds + pins), the rest of the prompt
//   attach            attach blob + adopt, resume admit at k chunks, prefill the suffix
//   restart           adopt of a full session blob, then the bound session's rebind
//   rung rewind       truncate_parked to a checkpoint
//   mode-4 rewind     rewind_prefill of a live turn to R
//   failed admission  a resume or cold admit the host ledger refuses
//   cold slot reuse   a bound parked session's slot taken by a new session
//   recompute/clear   clear / release / drop of the slot
// and the admit through a recording engine: a second session with the same head attaches and prefills only from it.

#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"
#include "kvflash_slot_world.h"
#include "kvflash_test_seams.h"

#include "../../src/common/concurrency/kvflash_slot_kv.h"
#include "../../src/common/concurrency/seq_slot_manager.h"
#include "../../src/common/kvflash_arena.h"
#include "../../src/common/kvflash_pinned_head.h"
#include "../../src/common/kvflash_slot_bundle.h"
#include "../../src/common/kvflash_session_store.h"

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPinnedHeadPathsFixture {};

constexpr int kN = 2, kQ = 12, kChunk = 64, kK = 3, kHead = kK * kChunk + 10;   // kHead: the shared system+tools head of prompt_of
constexpr int64_t kMaxCtx = 4096, kChunkBytes = 1000;

struct Slot : KvFlashSlotPager {
    Slot() { set_pinned_host_backing(pager_pinned_host_available()); }
};

struct World {
    KvFlashTestArena arena;
    std::vector<std::unique_ptr<Slot>> s;
    KvFlashTestLedger ledger;
    std::unique_ptr<KvFlashSlotKv> kv;
    KvFlashHeadBinder binder{kN};
    std::map<uint64_t, KvFlashHead> heads;                      // what the engine keeps host-side, keyed by hash
    explicit World(int64_t host_budget = 1000000) : ledger(host_budget) {
        arena.set_host_wait([] {});
        std::vector<KvFlashSlotPager *> raw;
        for (int i = 0; i < kN; i++) {
            s.push_back(std::make_unique<Slot>());
            KvFlashConfig c; c.chunk_tokens = kChunk; c.sink_chunks = 1; c.tail_window_chunks = 2;
            arena.add(*s.back());
            if (!s.back()->attach_slice(i, kN, kQ, c, {}, {})) throw std::runtime_error("attach failed");
            raw.push_back(s.back().get());
        }
        KvFlashSlotKvConfig cfg; cfg.quota_chunks = kQ; cfg.headroom_blocks = 2; cfg.max_ctx = kMaxCtx; cfg.chunk_bytes = kChunkBytes;
        kv = std::make_unique<KvFlashSlotKv>(raw, ledger, cfg);
        kv->set_binder(&binder);                                  // the production wiring, nothing else
    }
};

std::vector<int32_t> prompt_of(int n, int salt = 0) {
    std::vector<int32_t> v((size_t)n);
    for (int i = 0; i < n; i++) v[(size_t)i] = i < kHead ? i * 3 + 5 : 100000 + salt * 1000 + i;   // a shared head, a tail per session
    return v;
}

SeqEngine::AdmitResult admit_cold(World & w, SeqSlotManager & mgr, uint64_t id, const std::vector<int32_t> & prompt, int slot = -1) {
    w.kv->set_next_total_tokens((int64_t)prompt.size() + 60);
    if (slot >= 0) w.kv->set_next_slot(slot);
    return mgr.admit(id, prompt, SamplerCfg{});
}

// Prefill `n` more tokens in the pool's own cuts, completing the page-outs the full slice queues.
bool prefill(World & w, SeqSlotManager & mgr, int slot, int n) {
    while (n > 0) {
        const int c = std::min(n, kChunk);
        auto r = mgr.append_prefill(slot, c);
        if (!r.ok) { if (!w.arena.complete_all()) return false; r = mgr.append_prefill(slot, c); if (!r.ok) return false; }
        n -= c;
    }
    return true;
}

void must(bool ok) { if (!ok) throw std::runtime_error("paths test precondition failed"); }

// Pins and residency of chunks [0, k) on `slot`, and what recall reads.
bool bound(World & w, int slot, int k) {
    bool ok = w.binder.binding(slot).k == k && w.binder.pins(slot) == k;
    for (int c = 0; c < k; c++) ok = ok && w.s[(size_t)slot]->is_pinned(c) && w.s[(size_t)slot]->is_resident(c);
    return ok;
}
// A restored pager holds the pins it parked with (they ride in the pool layout); the binder knows nothing of the slot until the rebind.
bool restored_unbound(World & w, int slot, int k) {
    bool ok = w.binder.binding(slot).k == 0 && w.binder.pins(slot) == 0;
    for (int c = 0; c < k; c++) ok = ok && w.s[(size_t)slot]->is_pinned(c);
    return ok;
}
bool unbound(World & w, int slot, int k_was) {
    bool ok = w.binder.binding(slot).k == 0 && w.binder.pins(slot) == 0;
    for (int c = 0; c < k_was; c++) ok = ok && !w.s[(size_t)slot]->is_pinned(c);
    return ok;
}
// Later pressure: fill the slice past its pool; the bound chunks must survive every eviction.
bool pressure(World & w, SeqSlotManager & mgr, int slot, int k) {
    const int len = mgr.slot(slot).cur_pos;
    return prefill(w, mgr, slot, (kQ + 4) * kChunk - len) && bound(w, slot, k);
}

struct Captured { uint64_t hash = 0; int slot = -1; std::vector<int32_t> prompt; KvFlashHead head; };
// A session that prefilled its first kK chunks, captured the entry and went on to a full prompt.
Captured capture_on_miss(World & w, SeqSlotManager & mgr, uint64_t id, int total, int salt = 0) {
    Captured c;
    c.prompt = prompt_of(total, salt);
    const auto r = admit_cold(w, mgr, id, c.prompt);
    must(r.status == SeqEngine::AdmitResult::Status::admitted);
    c.slot = r.slot;
    must(prefill(w, mgr, c.slot, kK * kChunk));
    c.head = kvflash_head_capture(w.binder, c.slot, w.kv->pager(c.slot), c.prompt, kK, kK, {1, 2, 3});
    must(c.head.k == kK);
    c.hash = c.head.hash;
    w.heads.emplace(c.hash, c.head);
    return c;
}
}  // namespace

TEST_CASE(KvflashPinnedHeadPathsFixture, capture_on_miss_pins_the_prefix_and_it_survives_the_rest_of_the_prompt) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured c = capture_on_miss(w, mgr, 1, 3000);
    CHECK(bound(w, c.slot, kK));
    CHECK(pressure(w, mgr, c.slot, kK));
}

TEST_CASE(KvflashPinnedHeadPathsFixture, attach_pins_the_prefix_resumes_at_64k_and_it_survives_the_suffix) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    const int b_slot = 1 - a.slot;
    REQUIRE(kvflash_head_attach(*w.kv, w.binder, b_slot, 0x77, w.heads.at(a.hash)));
    CHECK(w.heads.at(a.hash).slab == std::vector<uint8_t>({1, 2, 3}));
    CHECK(bound(w, b_slot, kK));                                 // the adopt itself left it pinned
    const auto p = prompt_of(2600, 1);
    w.kv->set_next_total_tokens((int64_t)p.size() + 60);
    w.kv->set_next_resume(b_slot, kK * kChunk);
    const auto r = mgr.admit_at(b_slot, 2, p, SamplerCfg{}, kK * kChunk);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(mgr.slot(b_slot).cur_pos == kK * kChunk);              // prefills only from E_c
    CHECK(bound(w, b_slot, kK));
    CHECK(pressure(w, mgr, b_slot, kK));
}

TEST_CASE(KvflashPinnedHeadPathsFixture, an_attach_the_pager_cannot_pin_falls_back_to_cold_and_leaves_the_slot_free) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    // k = 9: sink 1 + tail 2 + 9 + 2 > 12, the pager refuses the span
    KvFlashHead big = a.head;
    big.k = 9; big.hash = 0x99;
    big.hdr[8] = 9; big.hdr[9] = big.hdr[10] = big.hdr[11] = 0;   // the header of the captured pager says 3 chunks: patch the count
    big.chunks.clear();
    CHECK(!kvflash_head_attach(*w.kv, w.binder, 1 - a.slot, 0x78, big));
    CHECK(!w.kv->parked(1 - a.slot));
    CHECK(w.s[(size_t)(1 - a.slot)]->n_chunks() == 0);
    CHECK(unbound(w, 1 - a.slot, 9));
    CHECK(w.ledger.reservation_count() == 1);                    // only the live capturing session's
}

TEST_CASE(KvflashPinnedHeadPathsFixture, restart_adopts_a_full_session_and_the_rebind_pins_it_again) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 1200);
    REQUIRE(prefill(w, mgr, a.slot, 1200 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    CHECK(bound(w, a.slot, kK));                                 // parked-gpu: still bound
    const std::vector<uint8_t> blob = w.s[(size_t)a.slot]->serialize();
    // "restart": the pagers are fresh, the binding comes from meta.json
    World w2;
    REQUIRE(w2.kv->adopt_parked(a.slot, 0x99, 1200, blob.data(), blob.size()) == PagedKvStatus::Ok);
    CHECK(restored_unbound(w2, a.slot, kK));                      // deserialize put the pins back, nothing bound yet
    CHECK(kvflash_head_rebind(*w2.kv, w2.binder, a.slot, hist, a.hash, kK));
    CHECK(bound(w2, a.slot, kK));
    // a binding that names another prefix than the history's is not honoured
    World w3;
    REQUIRE(w3.kv->adopt_parked(a.slot, 0x99, 1200, blob.data(), blob.size()) == PagedKvStatus::Ok);
    auto other = hist;
    other[5] ^= 1;
    CHECK(!kvflash_head_rebind(*w3.kv, w3.binder, a.slot, other, a.hash, kK));
    CHECK(unbound(w3, a.slot, kK));
}

// A parked session that went through evictions (its pool layout is not the identity and its LRU order is not the chunk
// order) comes back from a bundle (header, chunks and ledger, and the pool layout the head carries) as the pager that
// parked, with the pins the rebind puts back: the restored slot evicts the victims the live slot would.
TEST_CASE(KvflashPinnedHeadPathsFixture, a_restart_restores_the_parked_layout_lru_order_and_pins) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 1200);
    REQUIRE(prefill(w, mgr, a.slot, 1200 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    KvFlashSlotPager & live = *w.s[(size_t)a.slot];
    REQUIRE(live.resident_blocks() < live.n_chunks());           // it evicted: not every chunk is in the pool
    std::vector<int> identity;
    for (int c = 0; c < live.n_chunks(); c++) if (live.is_resident(c)) identity.push_back(c);
    REQUIRE(live.lru_victims(kQ) != std::vector<int>(identity.begin() + kK, identity.end()));   // the LRU order is not the chunk order

    KvFlashSlotHead h;
    kvflash_head_take_pager(h, live);
    std::vector<uint8_t> blob;                                    // what the bundle assembles from the head and the log
    live.serialize_to([&](const uint8_t * d, size_t n) { blob.insert(blob.end(), d, d + n); return true; });
    std::memcpy(blob.data(), kKvFlashPagerMagic3, 8);
    blob.insert(blob.end(), h.pool.begin(), h.pool.end());

    World w2;                                                     // the restarted server
    REQUIRE(w2.kv->adopt_parked(a.slot, 0x99, 1200, blob.data(), blob.size()) == PagedKvStatus::Ok);
    KvFlashSlotPager & back = *w2.s[(size_t)a.slot];
    CHECK(back.serialize() == live.serialize());                  // block table, LRU ranks, use clock, free list order
    CHECK(restored_unbound(w2, a.slot, kK));                      // the pins are in the layout: the binding comes from meta.json
    REQUIRE(kvflash_head_rebind(*w2.kv, w2.binder, a.slot, hist, a.hash, kK));
    CHECK(bound(w2, a.slot, kK));
    for (int c = 0; c < live.n_chunks(); c++) CHECK(back.is_pinned(c) == live.is_pinned(c));
    CHECK(back.lru_victims(kQ) == live.lru_victims(kQ));
}

TEST_CASE(KvflashPinnedHeadPathsFixture, a_rung_rewind_of_a_parked_session_keeps_the_binding_down_to_the_prefix_and_releases_below) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 1200);
    REQUIRE(prefill(w, mgr, a.slot, 1200 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    REQUIRE(w.kv->truncate_parked(a.slot, 700));
    CHECK(bound(w, a.slot, kK));
    REQUIRE(w.kv->truncate_parked(a.slot, kK * kChunk));         // exactly the prefix: still held
    CHECK(bound(w, a.slot, kK));
    REQUIRE(w.kv->truncate_parked(a.slot, kChunk + 10));         // below the attached prefix
    CHECK(unbound(w, a.slot, kK));
}

TEST_CASE(KvflashPinnedHeadPathsFixture, a_mode_4_rewind_keeps_the_pins_and_the_recall_count_and_below_the_prefix_releases) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    REQUIRE(prefill(w, mgr, a.slot, 1500 - kK * kChunk));
    REQUIRE(mgr.rewind_prefill(a.slot, 900));                    // R: the first pass is rewound
    CHECK(bound(w, a.slot, kK));                                 // what kvflash_prefill_apply_on reads is still k
    CHECK(pressure(w, mgr, a.slot, kK));
    REQUIRE(mgr.rewind_prefill(a.slot, kChunk));                 // a rewind below the attached prefix
    CHECK(unbound(w, a.slot, kK));
}

TEST_CASE(KvflashPinnedHeadPathsFixture, a_failed_admission_leaves_no_stale_binding_or_pins) {
    World w(/*host budget: 14 chunks of backing*/ 14 * kChunkBytes);
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 640);
    REQUIRE(prefill(w, mgr, a.slot, 640 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    // a resume the host ledger refuses: the session stays parked and bound as it was
    w.kv->set_next_total_tokens(20 * kChunk);
    w.kv->set_next_resume(a.slot, 640);
    CHECK(mgr.admit_at(a.slot, 2, prompt_of(900), SamplerCfg{}, 640).status != SeqEngine::AdmitResult::Status::admitted);
    CHECK(w.kv->parked(a.slot));
    CHECK(bound(w, a.slot, kK));
    // a cold admit into the same slot that the ledger refuses: the parked session was dropped, its binding with it
    w.kv->set_next_total_tokens(20 * kChunk);
    w.kv->set_next_slot(a.slot);
    CHECK(mgr.admit_at(a.slot, 3, prompt_of(900), SamplerCfg{}, 0).status != SeqEngine::AdmitResult::Status::admitted);
    CHECK(!w.kv->parked(a.slot));
    CHECK(unbound(w, a.slot, kK));
}

TEST_CASE(KvflashPinnedHeadPathsFixture, cold_slot_reuse_releases_the_old_binding_and_the_new_session_binds_on_its_own) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 1200);
    REQUIRE(prefill(w, mgr, a.slot, 1200 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    const auto r = admit_cold(w, mgr, 2, prompt_of(2000, 4), a.slot);   // the slot's session is replaced
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(r.slot == a.slot);
    CHECK(unbound(w, a.slot, kK));
    REQUIRE(prefill(w, mgr, a.slot, kK * kChunk));
    const auto cap = kvflash_head_capture(w.binder, a.slot, w.kv->pager(a.slot), prompt_of(2000, 4), kK, kK, {1});
    CHECK(cap.hash == a.hash);                                   // same head, the new session binds to it
    CHECK(bound(w, a.slot, kK));
    CHECK(pressure(w, mgr, a.slot, kK));
}

TEST_CASE(KvflashPinnedHeadPathsFixture, clear_release_and_drop_release_each_slots_binding_and_leave_the_other_slots_alone) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    REQUIRE(kvflash_head_attach(*w.kv, w.binder, 1 - a.slot, 0x77, w.heads.at(a.hash)));
    // the live slot is cleared (a recompute starts over): its binding goes, the parked slot's stays
    CHECK(w.kv->clear(mgr.slot(a.slot).handle) == PagedKvStatus::Ok);
    CHECK(unbound(w, a.slot, kK));
    mgr.retire(a.slot);                                          // release: reset + ledger release
    CHECK(unbound(w, a.slot, kK));
    CHECK(bound(w, 1 - a.slot, kK));
    w.kv->drop_parked(1 - a.slot);                               // evict / drop of the other
    CHECK(unbound(w, 1 - a.slot, kK));
}

// ---- the admit through a recording engine (the style of the scheduler's MetaRecordingEngine) ----------------
namespace {
// What Qwen35SeqEngine::admit_bounded does about the head, on the CPU allocator: decide, attach or cold
// admit, capture at the cut. It records where each prefill started.
struct RecordingEngine {
    World & w;
    SeqSlotManager & mgr;
    struct Admit { int slot = -1; int prefill_from = -1; KvFlashPreambleDecision::Kind kind = KvFlashPreambleDecision::Cold; };
    std::vector<Admit> log;
    int e_role_of(const std::vector<int32_t> & ids) const { (void)ids; return kHead; }   // the system block's end in prompt_of
    Admit admit(uint64_t id, const std::vector<int32_t> & prompt) {
        Admit a;
        const int e_role = e_role_of(prompt);
        const auto d = kvflash_head_decide(prompt, e_role / kChunk * kChunk, kChunk, kChunk, /*k_max=*/kK,
                                           [&](uint64_t h, int) { return w.heads.count(h) > 0; });
        a.kind = d.kind;
        int slot = 0;
        while (slot < kN && (mgr.is_active(slot) || w.kv->parked(slot))) slot++;
        if (slot == kN) return a;
        a.slot = slot;
        if (d.kind == KvFlashPreambleDecision::Attach) {
            const auto hashes = kvflash_head_hashes(prompt, d.k, kChunk);
            if (kvflash_head_attach(*w.kv, w.binder, slot, 0x1000 + (uint64_t)slot, w.heads.at(hashes.back()))) {
                w.kv->set_next_total_tokens((int64_t)prompt.size() + 60);
                w.kv->set_next_resume(slot, (uint32_t)d.k * kChunk);
                const auto r = mgr.admit_at(slot, id, prompt, SamplerCfg{}, d.k * kChunk);
                if (r.status == SeqEngine::AdmitResult::Status::admitted) { a.prefill_from = d.k * kChunk; log.push_back(a); return a; }
                w.kv->drop_parked(slot);
            }
        }
        const auto r = admit_cold(w, mgr, id, prompt, slot);
        a.slot = r.slot; a.prefill_from = 0;
        if (d.kind == KvFlashPreambleDecision::Capture) {   // the capture cut at k chunks: prefill to it, capture, go on
            prefill(w, mgr, r.slot, d.k * kChunk);
            const KvFlashHead h = kvflash_head_capture(w.binder, r.slot, w.kv->pager(r.slot), prompt, d.k, d.k, {1, 2, 3});
            if (h.k > 0) w.heads.emplace(h.hash, h);
        }
        log.push_back(a);
        return a;
    }
};
}  // namespace

TEST_CASE(KvflashPinnedHeadPathsFixture, a_second_session_with_the_same_system_and_tools_attaches_and_prefills_only_from_e_c) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    RecordingEngine eng{w, mgr, {}};
    const auto first = eng.admit(1, prompt_of(1500, 1));         // E_role = 202 -> E_c = 192: three chunks, captured on miss
    CHECK(first.kind == KvFlashPreambleDecision::Capture);
    CHECK(first.prefill_from == 0);
    CHECK(w.heads.size() == 1);
    const auto second = eng.admit(2, prompt_of(1700, 2));        // same head, another suffix
    CHECK(second.kind == KvFlashPreambleDecision::Attach);
    CHECK(second.prefill_from == kK * kChunk);                    // only from E_c
    CHECK(second.slot != first.slot);
    CHECK(mgr.slot(second.slot).cur_pos == kK * kChunk);
    CHECK(w.binder.binding(second.slot).k == kK);
    CHECK(w.binder.pins(second.slot) == kK);
    CHECK(w.binder.binding(first.slot).hash == w.binder.binding(second.slot).hash);
    // a session with another system prompt shares nothing: cold, and it captures its own entry 
    auto other = prompt_of(1500, 3);
    for (int i = 0; i < 20; i++) other[(size_t)i] += 77;
    mgr.retire(first.slot);                                      // a free slot for it
    const auto third = eng.admit(3, other);
    CHECK(third.kind == KvFlashPreambleDecision::Capture);
    CHECK(third.prefill_from == 0);
    CHECK(w.heads.size() == 2);
}

// The report's cached tokens come from AdmitResult::resumed_tokens (scheduler: admit_cached_tokens): an admit that starts
// the prompt at an attached head of L tokens says so, as a parked resume does; a cold admit says 0.
TEST_CASE(KvflashPinnedHeadPathsFixture, an_admit_on_an_attached_head_reports_the_heads_tokens_as_resumed) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    const int b_slot = 1 - a.slot;
    REQUIRE(kvflash_head_attach(*w.kv, w.binder, b_slot, 0x77, w.heads.at(a.hash)));
    const auto p = prompt_of(2600, 1);
    w.kv->set_next_total_tokens((int64_t)p.size() + 60);
    w.kv->set_next_resume(b_slot, kK * kChunk);
    const auto r = mgr.admit_at(b_slot, 2, p, SamplerCfg{}, kK * kChunk);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(r.resumed_tokens == kK * kChunk);
    mgr.retire(a.slot);
    CHECK(admit_cold(w, mgr, 3, prompt_of(1500, 2), a.slot).resumed_tokens == 0);
}

// ---- start-up order: init_bundles_ adopts, then init_heads_ installs a fresh binder and head_k_max -------------------

namespace {
// A parked session with a pinned head, as the engine's park left it: its blob, history, hash and the slot it parked in.
struct Parked { std::vector<uint8_t> blob; std::vector<int32_t> hist; uint64_t hash = 0; int slot = -1; };
Parked parked_with_head() {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 1200);
    must(prefill(w, mgr, a.slot, 1200 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    Parked p;
    must(mgr.park(a.slot, p.hist));
    p.blob = w.s[(size_t)a.slot]->serialize();
    p.hash = a.hash; p.slot = a.slot;
    return p;
}
// What the constructor does: adopt (no binder yet: no rebind), init_heads_ (a fresh binder), then the one pass.
int startup(World & w, const Parked & p, int slot, uint64_t hash, int k, const std::vector<int32_t> & hist) {
    must(w.kv->adopt_parked(slot, 0x99, 1200, p.blob.data(), p.blob.size()) == PagedKvStatus::Ok);
    w.binder = KvFlashHeadBinder(kN);
    return kvflash_rebind_restored(*w.kv, w.binder, kK, {{slot, &hist, kvflash_hex16(hash), k}});
}
}

TEST_CASE(KvflashPinnedHeadPathsFixture, a_session_restored_at_startup_has_its_head_bound_with_slot_pins_equal_to_the_pager_pins) {
    const Parked p = parked_with_head();
    World w;
    CHECK(startup(w, p, p.slot, p.hash, kK, p.hist) == 1);
    CHECK(bound(w, p.slot, kK));                                  // binder k, slot pins and the pager's hard pins all agree
    CHECK(w.binder.binding(p.slot).hash == p.hash);
    CHECK(kvflash_pinned_prefix(*w.s[(size_t)p.slot]) == kK);
}

TEST_CASE(KvflashPinnedHeadPathsFixture, the_binding_a_restored_session_parks_with_is_the_one_it_restored_with) {
    const Parked p = parked_with_head();
    for (bool rebind : {true, false}) {
        World w;
        if (rebind) {
            REQUIRE(startup(w, p, p.slot, p.hash, kK, p.hist) == 1);
        } else {
            REQUIRE(w.kv->adopt_parked(p.slot, 0x99, 1200, p.blob.data(), p.blob.size()) == PagedKvStatus::Ok);
            w.binder = KvFlashHeadBinder(kN);                    // init_heads_ without kvflash_rebind_restored
            REQUIRE(restored_unbound(w, p.slot, kK));
        }
        SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
        auto prompt = p.hist;
        prompt.push_back(42);                                   // resume the restored history, then park again
        w.kv->set_next_total_tokens((int64_t)prompt.size() + 60);
        w.kv->set_next_resume(p.slot, (uint32_t)p.hist.size());
        REQUIRE(mgr.admit_at(p.slot, 2, prompt, SamplerCfg{}, (int)p.hist.size()).status == SeqEngine::AdmitResult::Status::admitted);
        REQUIRE(prefill(w, mgr, p.slot, 1));
        mgr.commit_prefill(p.slot);
        std::vector<int32_t> hist;
        REQUIRE(mgr.park(p.slot, hist));

        KvFlashParkStats st;
        st.tokens = hist.size();
        // Mirror server/src/qwen35/concurrency/qwen35_seq_engine.cpp:1496 (inline park-time binding).
        if (const KvFlashHeadBinding hb = w.binder.binding(p.slot); hb.k > 0) { st.preamble = kvflash_hex16(hb.hash); st.preamble_chunks = hb.k; }
        const std::string root = (std::filesystem::temp_directory_path() / "kvflash_pinned_head_repark").string();
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root + "/sessions");
        KvFlashSessionBook book;
        book.init(kN);
        book.enable_store(KvFlashSessionStore(root), 0x99);
        book.begin_session(p.slot, "claude", "S", 1000);
        REQUIRE(book.publish_park(p.slot, st, 1001));
        std::vector<uint8_t> meta;
        REQUIRE(kvflash_bundle_read(book.store().dir(book[p.slot].key) + "/meta.json", meta, kKvFlashMetaMax));
        KvFlashSessionEntry persisted;
        REQUIRE(kvflash_meta_unpack(std::string_view((const char *)meta.data(), meta.size()), persisted));
        if (rebind) {
            CHECK(persisted.preamble_chunks == kK);
            CHECK(persisted.preamble == kvflash_hex16(p.hash));
        } else {
            CHECK(persisted.preamble_chunks == 0);
            CHECK(persisted.preamble.empty());
        }
        std::filesystem::remove_all(root);
    }
}

TEST_CASE(KvflashPinnedHeadPathsFixture, a_session_restored_into_the_other_slot_index_rebinds_too) {
    const Parked p = parked_with_head();
    World w;
    const int other = 1 - p.slot;
    CHECK(startup(w, p, other, p.hash, kK, p.hist) == 1);
    CHECK(bound(w, other, kK));
    CHECK(unbound(w, p.slot, kK));                                // the other slot is untouched
}

TEST_CASE(KvflashPinnedHeadPathsFixture, a_rebind_that_fails_leaves_the_restored_session_unbound_and_unpinned) {
    const Parked p = parked_with_head();
    World w;
    auto other = p.hist;
    other[5] ^= 1;                                                // the history no longer starts with the head
    CHECK(startup(w, p, p.slot, p.hash, kK, other) == 0);
    CHECK(unbound(w, p.slot, kK));                                // not half-bound: no binding, no pins, the pager's pins released
}
