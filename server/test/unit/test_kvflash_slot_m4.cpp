// Unit test for the slot engine's mode-4 single pass (no GPU): the path gate, the turn conversion (one pass, apply at
// the resume point, no rewind) and the tail capture of a tracked turn.

#include "CppUnitTestFramework.hpp"

#include "kvflash_slot_m4.h"
#include "kvflash_slot_turn.h"
#include "kvflash_uq.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotM4Fixture {};
constexpr int CT = 64;
std::vector<int32_t> seq(int n) { std::vector<int32_t> v((size_t)n); for (int i = 0; i < n; i++) v[(size_t)i] = i; return v; }
std::vector<int32_t> extend(std::vector<int32_t> v, int n) { for (int i = 0; i < n; i++) v.push_back(1000000 + i); return v; }
KvFlashM4Last last_scored() {
    KvFlashM4Last l;
    l.valid = true; l.uq_hi = 900; l.uq_hash = 7; l.ids = seq(6400);
    l.scores.assign(90, 0.5f); l.q_tail.assign(8, 1.0f);
    return l;
}
KvFlashM4Ask ask() { KvFlashM4Ask a; a.uq_hi = 900; a.uq_hash = 7; a.uq_stored = true; a.kv_offset = 6336; a.chunk_tokens = CT; a.first_window = 80; return a; }
using Skip = KvFlashRecallSkip;
KvFlashSlotTurn first_pass_turn() {
    KvFlashSlotTurn t;
    t.phase = KvFlashTurnPhase::FirstPass; t.m4 = true;
    t.kv_offset = 6336; t.r = 6500; t.end = 6700; t.reserve_chunks = 5; t.ckpt = 6690; t.end_rung = true; t.header_rung = true;
    t.cuts = {6500, 6600, 6700};
    return t;
}
}

TEST_CASE(KvflashSlotM4Fixture, the_path_is_the_serial_decision_when_enabled) {
    const auto ids = extend(seq(6400), 300);
    CHECK(kvflash_slot_m4_path(true, 1, true, Skip::None, last_scored(), ask(), ids) == KvFlashM4Path::Reuse);
    KvFlashM4Ask a = ask(); a.first_window = 91;
    CHECK(kvflash_slot_m4_path(true, 1, true, Skip::None, last_scored(), a, ids) == KvFlashM4Path::Prescore);
}
TEST_CASE(KvflashSlotM4Fixture, anything_off_or_unfit_runs_the_full_path) {
    const auto ids = extend(seq(6400), 300);
    CHECK(kvflash_slot_m4_path(false, 1, true, Skip::None, last_scored(), ask(), ids) == KvFlashM4Path::Full);       // LUCE_KVFLASH_M4_SINGLE=0
    CHECK(kvflash_slot_m4_path(true, 0, true, Skip::None, last_scored(), ask(), ids) == KvFlashM4Path::Full);        // no recall blocks
    CHECK(kvflash_slot_m4_path(true, 1, false, Skip::None, last_scored(), ask(), ids) == KvFlashM4Path::Full);       // not a first-pass turn
    CHECK(kvflash_slot_m4_path(true, 1, true, Skip::Fits, last_scored(), ask(), ids) == KvFlashM4Path::Full);        // gate at the resume point
    CHECK(kvflash_slot_m4_path(true, 1, true, Skip::PoolTooSmall, last_scored(), ask(), ids) == KvFlashM4Path::Full);
    CHECK(kvflash_slot_m4_path(true, 1, true, Skip::NoSealedChunk, last_scored(), ask(), ids) == KvFlashM4Path::Full);
    CHECK(kvflash_slot_m4_path(true, 1, true, Skip::None, KvFlashM4Last{}, ask(), ids) == KvFlashM4Path::Full);      // no record
}
// A side request with another last human question runs in the session's slot (a regenerate, a title call, a probe) and the
// session then goes on from before it: its own question's rows are still in the slot, so the turn is not Full. With one
// store per slot the side question replaced them and every later turn of the question ran two passes without its rows.
TEST_CASE(KvflashSlotM4Fixture, a_side_question_keeps_the_first_questions_rows) {
    auto store = [](int hi, uint64_t hash) { KvFlashUqStore s; s.lo = hi - 2; s.hi = hi; s.n = 2; s.hash = hash; s.rows.assign(8, 1.0f); return s; };
    KvFlashUqSlots stores(1);
    kvflash_uq_slot_put(stores, 0, store(900, 7));     // question A: the key ask() and last_scored() carry
    kvflash_uq_slot_put(stores, 0, store(1000, 8));    // question B, the side request's
    KvFlashM4Ask a = ask();
    a.uq_stored = kvflash_uq_slot_get(stores, 0, a.uq_hi, a.uq_hash) != nullptr;
    CHECK(a.uq_stored);
    CHECK(kvflash_slot_m4_path(true, 1, true, Skip::None, last_scored(), a, extend(seq(6400), 300)) != KvFlashM4Path::Full);
}
TEST_CASE(KvflashSlotM4Fixture, single_turn_is_one_pass_applied_at_the_resume_point) {
    KvFlashSlotTurn t = first_pass_turn();
    kvflash_slot_turn_single(t, CT);
    CHECK(!t.recall());
    CHECK(!t.first_pass());
    CHECK(!t.snapshot_at_r());
    CHECK(t.apply_at == 6336);
    CHECK(t.cuts.empty());
    CHECK(!t.end_rung);
    CHECK(!t.header_rung);
    CHECK(t.track_tail);
    CHECK(t.reserve_chunks == (6700 - 6336 + CT - 1) / CT + 1);   // room for the one pass, not the replay
    CHECK(t.ckpt == 6690);                                          // the LRU turn's checkpoint rule
}
TEST_CASE(KvflashSlotM4Fixture, a_tracked_turn_captures_its_tail_without_a_rewind) {
    KvFlashSlotTurn t = first_pass_turn();
    kvflash_slot_turn_single(t, CT);
    const KvFlashTurnStep mid = kvflash_slot_turn_step(t, 6336, 300, 64);
    CHECK(!mid.capture);
    CHECK(mid.q_rows == 0);
    const KvFlashTurnStep last = kvflash_slot_turn_step(t, 6600, 100, 64);
    CHECK(last.capture);
    CHECK(last.q_rows == 64);
    CHECK(!last.snapshot);
    // a tracked LRU turn (the prompt fits: no recall) captures too; an untracked one does not
    KvFlashSlotTurn lru; lru.end = 700; lru.track_tail = true;
    CHECK(kvflash_slot_turn_step(lru, 600, 100, 64).q_rows == 64);
    lru.track_tail = false;
    CHECK(kvflash_slot_turn_step(lru, 600, 100, 64).q_rows == 0);
    CHECK(!kvflash_slot_turn_step(lru, 600, 100, 64).capture);
}
TEST_CASE(KvflashSlotM4Fixture, the_first_pass_still_snapshots_and_rewinds) {
    KvFlashSlotTurn t = first_pass_turn();
    CHECK(kvflash_slot_turn_step(t, 6400, 100, 64).snapshot);
    CHECK(kvflash_slot_turn_step(t, 6600, 100, 64).capture);
}
TEST_CASE(KvflashSlotM4Fixture, the_record_needs_a_question) {
    CHECK(!kvflash_slot_m4_record(seq(10), 0, 0, {}, {}).valid);
    const KvFlashM4Last r = kvflash_slot_m4_record(seq(10), 5, 9, {1.f}, {2.f});
    CHECK(r.valid);
    CHECK(r.uq_hi == 5);
    CHECK(r.uq_hash == 9);
    CHECK(r.ids.size() == 10u);
    CHECK(r.scores.size() == 1u);
    CHECK(r.q_tail.size() == 1u);
}
