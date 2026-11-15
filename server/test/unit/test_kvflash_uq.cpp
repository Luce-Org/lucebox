// Human-turn / capture-plan / per-session store helpers of LUCE_KVFLASH_UQ_QUERIES (kvflash_uq.h), the CPU selector
// with an extra query, and the pure GPU-scorer guard.

#include "CppUnitTestFramework.hpp"

#include "kvflash_exact_fast.h"
#include "kvflash_gpu_mass.h"
#include "kvflash_mass_synth.h"
#include "kvflash_slot_plan.h"
#include "kvflash_slot_turn.h"
#include "kvflash_uq.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashUqFixture {};

// Token ids used by the synthetic prompts.
constexpr int32_t S = 100, U = 101, SYS = 102, NL = 103, END = 104, TR = 105, A = 106;

void must(bool ok) { if (!ok) throw std::runtime_error("uq helper precondition failed"); }
}  // namespace

// ---- last human turn --------------------------------------------------------
TEST_CASE(KvflashUqFixture, last_human_turn_picks_the_last_non_tool_user_turn) {
    //            0  1    2   3 4 5  6   7  8 9  10 11 12 13 14
    std::vector<int32_t> ids = {S, SYS, NL, 7, END, NL, S, U, NL, 21, 22, 23, END, NL, S};
    ids.insert(ids.end(), {A, NL, 31, END, NL});                      // assistant turn (idx 14..)
    const auto t = kvflash_last_human_turn(ids, S, U, NL, END, TR);
    REQUIRE(t.valid());
    CHECK(t.lo == 9);
    CHECK(t.hi == 12);
}
TEST_CASE(KvflashUqFixture, last_human_turn_skips_tool_results) {
    // human question, assistant tool call, user turn that is a tool result, assistant header
    const std::vector<int32_t> ids = {S, U, NL, 21, 22, END, NL,
                                      S, A, NL, 31, END, NL,
                                      S, U, NL, TR, 41, END, NL,
                                      S, A, NL};
    const auto t = kvflash_last_human_turn(ids, S, U, NL, END, TR);
    REQUIRE(t.valid());
    CHECK(t.lo == 3);
    CHECK(t.hi == 5);
}
TEST_CASE(KvflashUqFixture, last_human_turn_several_human_turns_gives_the_last) {
    const std::vector<int32_t> ids = {S, U, NL, 1, END, NL, S, A, NL, 2, END, NL, S, U, NL, 3, 4, 5, END, NL, S, A, NL};
    const auto t = kvflash_last_human_turn(ids, S, U, NL, END, TR);
    CHECK(t.lo == 15);
    CHECK(t.hi == 18);
}
TEST_CASE(KvflashUqFixture, last_human_turn_none) {
    CHECK(!kvflash_last_human_turn({}, S, U, NL, END, TR).valid());
    CHECK(!kvflash_last_human_turn({S, SYS, NL, 7, END, NL}, S, U, NL, END, TR).valid());           // no user turn
    CHECK(!kvflash_last_human_turn({S, U, NL, TR, 9, END}, S, U, NL, END, TR).valid());             // only a tool result
    CHECK(!kvflash_last_human_turn({S, U, NL, 9, 9}, S, U, NL, END, TR).valid());                   // unterminated
    CHECK(!kvflash_last_human_turn({S, U, NL, 9, END}, -1, U, NL, END, TR).valid());                // unknown ids
    // Unknown tool_response id: tool results cannot be told apart, so there is no answer.
    CHECK(!kvflash_last_human_turn({S, U, NL, TR, 9, END}, S, U, NL, END, -1).valid());
    CHECK(!kvflash_last_human_turn({S, U, NL, 9, END}, S, U, NL, END, -1).valid());
}

// ---- capture plan, batch take, store ---------------------------------------
TEST_CASE(KvflashUqFixture, capture_plan_covers_the_window_or_says_why_not) {
    const KvFlashHumanTurn t{100, 300};
    auto p = kvflash_uq_plan(t, 64, 0, 1000);
    CHECK(p.capture); CHECK(p.n == 64); CHECK(p.win_lo == 236); CHECK(p.cut == 300);
    p = kvflash_uq_plan(KvFlashHumanTurn{100, 130}, 64, 0, 1000);   // short question: only its own tokens
    CHECK(p.capture); CHECK(p.n == 30); CHECK(p.win_lo == 100);
    p = kvflash_uq_plan(t, 64, 250, 1000);                           // window starts before the prefill
    CHECK(!p.capture);
    p = kvflash_uq_plan(t, 64, 0, 299);                              // prefill ends before the question does
    CHECK(!p.capture);
    p = kvflash_uq_plan(t, 0, 0, 1000);                              // N = 0: feature off
    CHECK(!p.capture); CHECK(p.n == 0);
    p = kvflash_uq_plan(KvFlashHumanTurn{}, 64, 0, 1000);            // no human turn
    CHECK(!p.capture);
}
TEST_CASE(KvflashUqFixture, batch_take_intersects_window_with_captured_rows) {
    // Batch [512, 1024) captured its last 64 rows = positions [960, 1024); window [980, 1024).
    auto t = kvflash_uq_batch_take(512, 512, 64, 980, 1024);
    CHECK(t.first_row == 20); CHECK(t.count == 44);
    // Window ends inside the batch (forced cut at hi): batch [768, 900) with hi = 900.
    t = kvflash_uq_batch_take(768, 132, 64, 850, 900);
    CHECK(t.first_row == 14); CHECK(t.count == 50);
    // No overlap.
    CHECK(kvflash_uq_batch_take(0, 512, 64, 980, 1024).count == 0);
    CHECK(kvflash_uq_batch_take(1024, 512, 64, 980, 1024).count == 0);
    // Short batch: captured rows = batch tokens.
    t = kvflash_uq_batch_take(1000, 24, 24, 980, 1024);
    CHECK(t.first_row == 0); CHECK(t.count == 24);
}
namespace {
KvFlashUqStore mk(int hi, uint64_t hash) {
    KvFlashUqStore st; st.lo = hi - 2; st.hi = hi; st.n = 2; st.hash = hash; st.rows.assign(8, (float)hi);
    return st;
}
}
TEST_CASE(KvflashUqFixture, store_lru_keyed_by_hi_and_prefix_hash) {
    std::vector<int32_t> ids(1000);
    for (int i = 0; i < 1000; i++) ids[(size_t)i] = i % 97;
    auto other = ids; other[299] ^= 1;
    auto later = ids; later[300] ^= 1;
    CHECK(kvflash_uq_hash(ids, 300) != kvflash_uq_hash(other, 300));    // ids[0,hi) differs
    CHECK(kvflash_uq_hash(ids, 300) == kvflash_uq_hash(later, 300));    // only ids[0,hi) matters
    std::vector<KvFlashUqStore> v;
    CHECK(kvflash_uq_find(v, 300, 1) == -1);
    kvflash_uq_put(v, mk(300, kvflash_uq_hash(ids, 300)), 8);
    CHECK(kvflash_uq_find(v, 300, kvflash_uq_hash(later, 300)) == 0);
    CHECK(kvflash_uq_find(v, 301, kvflash_uq_hash(ids, 300)) == -1);    // different hi
    CHECK(kvflash_uq_find(v, 300, kvflash_uq_hash(other, 300)) == -1);  // different prefix
    kvflash_uq_erase(v, 300, kvflash_uq_hash(ids, 300));
    CHECK(v.empty());
}
TEST_CASE(KvflashUqFixture, store_lru_keeps_other_conversations_and_evicts_the_oldest) {
    std::vector<KvFlashUqStore> v;
    for (int i = 1; i <= 3; i++) CHECK(kvflash_uq_put(v, mk(100 * i, (uint64_t)i), 3).empty());
    CHECK(v.size() == 3);
    CHECK(v[0].hi == 300 && v[2].hi == 100);
    kvflash_uq_touch(v, kvflash_uq_find(v, 100, 1));                    // conversation A is used again
    CHECK(v[0].hi == 100);
    const auto ev = kvflash_uq_put(v, mk(400, 4), 3);                   // a fourth: the least recent (200) goes
    REQUIRE(ev.size() == 1);
    CHECK(ev[0] == 200);
    CHECK(kvflash_uq_find(v, 100, 1) >= 0);                             // A survived the switch to B and C
    CHECK(kvflash_uq_find(v, 200, 2) == -1);
    kvflash_uq_put(v, mk(100, 1), 3);                                   // same key replaces, no growth
    CHECK(v.size() == 3);
}
TEST_CASE(KvflashUqFixture, a_slot_keeps_the_first_questions_store_after_a_side_question) {
    KvFlashUqSlots s(2);
    CHECK(kvflash_uq_slot_put(s, 0, mk(100, 0xa)).empty());             // the session's question A
    CHECK(kvflash_uq_slot_put(s, 0, mk(200, 0xb)).empty());             // a side request's question B, same slot
    const KvFlashUqStore * a = kvflash_uq_slot_get(s, 0, 100, 0xa);
    REQUIRE(a != nullptr);                                              // A is still there
    CHECK(a->hi == 100);
    CHECK(kvflash_uq_slot_get(s, 0, 200, 0xb) != nullptr);
    CHECK(s[1].empty());                                                // the other slot: untouched, and never read
    CHECK(kvflash_uq_slot_get(s, 1, 100, 0xa) == nullptr);
}
TEST_CASE(KvflashUqFixture, a_slots_stores_evict_the_least_recent_past_the_cap) {
    KvFlashUqSlots s(2);
    const int cap = (int)kKvFlashUqSlotMaxStores;
    for (int i = 1; i <= cap; i++) CHECK(kvflash_uq_slot_put(s, 0, mk(100 * i, (uint64_t)i)).empty());
    CHECK(kvflash_uq_slot_get(s, 0, 100, 1) != nullptr);                // the oldest is used again: 200 is now the least recent
    const auto ev = kvflash_uq_slot_put(s, 0, mk(100 * (cap + 1), (uint64_t)(cap + 1)));
    REQUIRE(ev.size() == 1);
    CHECK(ev[0] == 200);
    CHECK((int)s[0].size() == cap);
    CHECK(kvflash_uq_slot_get(s, 0, 100, 1) != nullptr);
    CHECK(kvflash_uq_slot_get(s, 0, 200, 2) == nullptr);
    CHECK(s[1].empty());
}
TEST_CASE(KvflashUqFixture, store_lru_with_max_zero_keeps_no_store) {
    std::vector<KvFlashUqStore> v;
    const auto ev = kvflash_uq_put(v, mk(100, 1), 0);
    CHECK(v.empty());
    REQUIRE(ev.size() == 1);
    CHECK(ev[0] == 100);
    kvflash_uq_put(v, mk(200, 2), 1);
    CHECK(kvflash_uq_put(v, mk(300, 3), 0) == std::vector<int>({200, 300}));
    CHECK(v.empty());
}
TEST_CASE(KvflashUqFixture, cut_follows_the_question_not_the_store) {
    const KvFlashHumanTurn t{100, 300};
    CHECK(kvflash_uq_cut(t, 64, 0, 1000) == 300);
    CHECK(kvflash_uq_cut(t, 64, 250, 1000) == 300);        // suffix prefill that still reaches the question end
    CHECK(kvflash_uq_cut(t, 64, 300, 1000) == -1);        // already behind the prefill start
    CHECK(kvflash_uq_cut(t, 64, 0, 300) == -1);           // the batch ends there anyway
    CHECK(kvflash_uq_cut(t, 64, 0, 200) == -1);
    CHECK(kvflash_uq_cut(t, 0, 0, 1000) == -1);           // off
    CHECK(kvflash_uq_cut(KvFlashHumanTurn{}, 64, 0, 1000) == -1);
}
TEST_CASE(KvflashUqFixture, a_store_holds_a_question_by_its_end_and_prefix_hash) {
    const KvFlashUqStore st = mk(300, 7);
    CHECK(kvflash_uq_holds(st, 300, 7));
    CHECK(!kvflash_uq_holds(st, 300, 8));                  // another prompt prefix
    CHECK(!kvflash_uq_holds(st, 364, 7));                  // another question
    CHECK(!kvflash_uq_holds(KvFlashUqStore{}, -1, 0));     // nothing stored, no question
}
TEST_CASE(KvflashUqFixture, cut_rows_are_the_cuts_last_rows_inside_the_window) {
    auto p = kvflash_uq_plan(KvFlashHumanTurn{1000, 1100}, 64, 0, 5000);   // window [1036, 1100)
    CHECK(kvflash_uq_cut_rows(p, 512, 512) == 0);          // [512, 1024): before the window
    CHECK(kvflash_uq_cut_rows(p, 1024, 76) == 64);         // [1024, 1100): the whole window
    CHECK(kvflash_uq_cut_rows(p, 1024, 30) == 18);         // [1024, 1054): its first 18 rows
    CHECK(kvflash_uq_cut_rows(p, 1054, 46) == 46);         // [1054, 1100): the rest
    CHECK(kvflash_uq_cut_rows(p, 1100, 512) == 0);         // behind it
    CHECK(kvflash_uq_cut_rows(p, 1024, 512) == 0);         // a cut over the question's end: its last rows are not the window
    p.capture = false;                                     // stored, or not in this prefill
    CHECK(kvflash_uq_cut_rows(p, 1024, 76) == 0);
}
TEST_CASE(KvflashUqFixture, rewind_below_the_window_recaptures) {
    const auto p = kvflash_uq_plan(KvFlashHumanTurn{100, 300}, 64, 0, 1000);   // window [236, 300)
    CHECK(kvflash_uq_rewind_recaptures(p, 97));            // start of the human turn
    CHECK(kvflash_uq_rewind_recaptures(p, 236));
    CHECK(!kvflash_uq_rewind_recaptures(p, 237));          // rewind stays after the window start
    CHECK(!kvflash_uq_rewind_recaptures(kvflash_uq_plan(KvFlashHumanTurn{}, 64, 0, 1000), 0));
}
TEST_CASE(KvflashUqFixture, a_capped_rewind_inside_the_window_moves_to_the_window_end) {
    // kvflash_mode4_plan's tail cap can put R inside the question's window. Replayed from there, the window's tail would
    // be recomputed under the new pool while its stored rows keep the old pool's: R moves to the window's end instead.
    const auto p = kvflash_uq_plan(KvFlashHumanTurn{100, 300}, 64, 0, 1000);   // window [236, 300)
    CHECK(kvflash_uq_snap_rewind(p, 237) == 300);
    CHECK(kvflash_uq_snap_rewind(p, 299) == 300);
    CHECK(kvflash_uq_snap_rewind(p, 236) == 236);          // at the window start: the whole window is recaptured
    CHECK(kvflash_uq_snap_rewind(p, 97) == 97);
    CHECK(kvflash_uq_snap_rewind(p, 300) == 300);          // at or after the end: the window is not replayed
    CHECK(kvflash_uq_snap_rewind(p, 500) == 500);
    CHECK(kvflash_uq_snap_rewind(kvflash_uq_plan(KvFlashHumanTurn{}, 64, 0, 1000), 250) == 250);   // UQ off
}

// ---- scorer: an extra aligned query raises that chunk -----------------------
TEST_CASE(KvflashUqFixture, the_question_rows_join_by_the_store_and_selector_alone_and_raise_their_chunk) {
    SynthCfg c;
    c.L = 2; c.HK = 2; c.G = 3; c.D = 64; c.CT = 16; c.NB = 24; c.KT = GGML_TYPE_Q8_0;
    c.sealed = 100; c.N = 8; c.keep_newest = 4; c.tail = 16;
    Synth s;
    s.build(c);
    REQUIRE(s.n_hot >= 8 && s.n_hot < 13);              // hot j and j + 8 would share query j % 8; j = 5 has no partner
    const int hot = 5;                                  // planted chunk 10 + 7*5 = 45, driven by query 5 alone
    const int chunk = 10 + 7 * hot;
    KvFlashExactKeys keys;
    keys.n_chunks = c.sealed; keys.chunk_tokens = c.CT; keys.type = c.KT;
    keys.seg = [&](int ch, int l, int h) -> const uint8_t * {
        return s.blk[(size_t)ch] >= 0 ? s.pool_seg(s.blk[(size_t)ch], l, h) : s.host_seg(ch, l, h);
    };
    auto select = [&](const std::vector<const float *> & qs) {
        std::vector<std::vector<float>> outs;
        std::vector<KvFlashMassPlanes> planes;
        must(kvflash_exact_mass_scores_fast_multi(keys, s.dims(), qs, outs, 0, nullptr, &planes));
        return kvflash_select_scores(planes);
    };
    // Base queries: all but #5. The stored user-question row is query #5.
    std::vector<const float *> base;
    for (int n = 0; n < c.N; n++) if (n != hot) base.push_back(s.qs.data() + (size_t)n * s.QS);
    KvFlashUqStore st;
    st.n = 1; st.rows.assign(s.qs.begin() + (long)hot * (long)s.QS, s.qs.begin() + (long)(hot + 1) * (long)s.QS);

    std::vector<const float *> with_uq = base;
    CHECK(kvflash_uq_join(with_uq, &st, "", s.QS, base.size()) == nullptr);
    REQUIRE(with_uq.size() == base.size() + 1);
    const auto sel_base = select(base), sel_uq = select(with_uq);
    CHECK(sel_uq[(size_t)chunk] > sel_base[(size_t)chunk] * 1.5f);
    // Monotone: max over queries never lowers any chunk.
    for (size_t i = 0; i < sel_base.size(); i++) CHECK(sel_uq[i] >= sel_base[i]);
    // No store, an empty one, rows of another shape or no selector queries: the reason is named and the query list
    // is untouched, byte for byte. The store and the selector decide alone: nothing about the device does.
    KvFlashUqStore none;
    std::vector<const float *> same = base;
    CHECK(std::string(kvflash_uq_join(same, &none, "", s.QS, base.size())) == "no-store");
    CHECK(std::string(kvflash_uq_join(same, nullptr, "window-not-in-prefill", s.QS, base.size())) == "window-not-in-prefill");
    CHECK(std::string(kvflash_uq_join(same, &st, "", s.QS + 1, base.size())) == "no-store");
    CHECK(std::string(kvflash_uq_join(same, &st, "", s.QS, 0)) == "no-v2-selector");
    CHECK((same == base));
}

// ---- GPU scorer VRAM guard (pure) ---------------------------------------------
TEST_CASE(KvflashUqFixture, gpu_mass_fits_guard_matches_the_scorer_rule) {
    const size_t MiB = (size_t)1 << 20, margin = 512 * MiB;
    CHECK(kvflash_gpu_mass_fits(100 * MiB, 1000 * MiB, 8000 * MiB, margin));
    CHECK(!kvflash_gpu_mass_fits(600 * MiB, 1000 * MiB, 8000 * MiB, margin));        // 600 + 512 > 1000
    CHECK(kvflash_gpu_mass_fits(9999 * MiB, 0, 0, margin));                          // unknown total: no guard
    // 64 queries more cost real bytes: 128 queries need more than 64 on the same geometry.
    KvFlashQkDims d; d.n_layers = 16; d.n_q_heads = 24; d.n_kv_heads = 4; d.head_dim = 256;
    CHECK(kvflash_gpu_mass_bytes(d, GGML_TYPE_Q4_0, 64, 256, 300, 128) > kvflash_gpu_mass_bytes(d, GGML_TYPE_Q4_0, 64, 256, 300, 64));
}

// ---- fail closed on missing token ids ------------------------------------------
TEST_CASE(KvflashUqFixture, uq_gate_requires_every_id) {
    CHECK(kvflash_uq_gate(S, U, END, TR).empty());
    CHECK(kvflash_uq_gate(S, U, END, -1) == "<tool_response> is not a single token");
    CHECK(kvflash_uq_gate(-1, U, END, TR) == "<|im_start|> id missing");
    CHECK(kvflash_uq_gate(S, -1, END, TR) == "user id missing");
    CHECK(kvflash_uq_gate(S, U, -1, TR) == "<|im_end|> id missing");
}

// ---- q_tail after a rewind (selector-v2 queries of the next turn) ---------------
namespace {
// One query row per position, every float of the row equal to the position.
std::vector<float> q_rows(int lo, int hi, size_t qsize) {
    std::vector<float> v;
    for (int p = lo; p < hi; p++) v.insert(v.end(), qsize, (float)p);
    return v;
}
std::vector<int> q_positions(const std::vector<float> & t, size_t qsize) {
    std::vector<int> o;
    for (size_t i = 0; i < t.size(); i += qsize) o.push_back((int)t[i]);
    return o;
}
}  // namespace

TEST_CASE(KvflashUqFixture, q_tail_trim_drops_rows_at_or_after_the_rewind_point) {
    const size_t q = 3;
    auto t = q_rows(92, 100, q);                                   // end=100, R=96
    kvflash_q_tail_trim(t, q, 100, 96);
    CHECK((q_positions(t, q) == std::vector<int>{92, 93, 94, 95}));
    t = q_rows(92, 100, q); kvflash_q_tail_trim(t, q, 100, 90);    // every row is >= R
    CHECK(t.empty());
    t = q_rows(92, 100, q); kvflash_q_tail_trim(t, q, 100, 100);   // R == end: nothing to drop
    CHECK(t.size() == 8 * q);
    t = q_rows(92, 100, q); kvflash_q_tail_trim(t, q, 100, 0);
    CHECK(t.empty());
    t = q_rows(97, 100, q); kvflash_q_tail_trim(t, q, 100, 98);    // tail shorter than the window
    CHECK((q_positions(t, q) == std::vector<int>{97}));
}
TEST_CASE(KvflashUqFixture, q_tail_trim_clears_a_ragged_tail) {
    std::vector<float> t(7, 1.f);
    kvflash_q_tail_trim(t, 3, 100, 96);
    CHECK(t.empty());
    t.assign(6, 1.f);
    kvflash_q_tail_trim(t, 0, 100, 96);
    CHECK(t.empty());
}
TEST_CASE(KvflashUqFixture, q_tail_after_rewind_is_pass1_below_r_then_pass2_rows) {
    const size_t q = 3; const int keep = 8, end = 100;
    // Pass 1 batches of 4 tokens, the last two hold the final 8 positions.
    std::vector<float> tail;
    for (int lo = 88; lo < end; lo += 4) kvflash_q_tail_append(tail, q_rows(lo, lo + 4, q), q, keep);
    CHECK((q_positions(tail, q) == std::vector<int>{92, 93, 94, 95, 96, 97, 98, 99}));
    // Rewind to R=96: pass 1 rows >= R are stale and dropped, pass 2 re-captures [96,100) under the new pool.
    kvflash_q_tail_trim(tail, q, end, 96);
    CHECK((q_positions(tail, q) == std::vector<int>{92, 93, 94, 95}));
    std::vector<float> pass2 = q_rows(96, 100, q);
    for (float & f : pass2) f += 1000.f;                           // marks "new pool"
    kvflash_q_tail_append(tail, pass2, q, keep);
    REQUIRE(tail.size() == 8 * q);
    for (size_t r = 0; r < 8; r++) {
        const float expect = r < 4 ? (float)(92 + r) : 1000.f + (float)(96 + r - 4);
        CHECK(tail[r * q] == expect);
    }
    // Rewind further than the window: the whole tail goes and pass 2 refills it.
    tail.clear();
    for (int lo = 88; lo < end; lo += 4) kvflash_q_tail_append(tail, q_rows(lo, lo + 4, q), q, keep);
    kvflash_q_tail_trim(tail, q, end, 40);
    CHECK(tail.empty());
}

// The fused scorer's device workspace stays allocated between calls and is not the archive: when a call is refused,
// releasing it must be tried before the CPU scorer answers.
TEST_CASE(KvflashUqFixture, gpu_mass_release_ws_helps_only_when_it_turns_a_refusal_into_a_fit) {
    const size_t MiB = (size_t)1 << 20, margin = 512 * MiB, total = 24000 * MiB;
    // a refusal like the M0 log lines (needs 2275 +512, 2600 free): a 256 MiB workspace held turns it into a fit
    CHECK(kvflash_gpu_mass_release_ws_helps(2275 * MiB, 2600 * MiB, total, margin, 256 * MiB));
    CHECK(!kvflash_gpu_mass_release_ws_helps(2275 * MiB, 2600 * MiB, total, margin, 0));          // nothing held
    CHECK(!kvflash_gpu_mass_release_ws_helps(2275 * MiB, 2800 * MiB, total, margin, 256 * MiB)); // already fits
    CHECK(!kvflash_gpu_mass_release_ws_helps(2275 * MiB, 2000 * MiB, total, margin, 256 * MiB)); // still short
    CHECK(kvflash_gpu_mass_release_ws_helps(2275 * MiB, 2531 * MiB, total, margin, 256 * MiB));  // exact boundary: 2275+512 == 2787 == 2531+256
    CHECK(!kvflash_gpu_mass_release_ws_helps(2275 * MiB, 2530 * MiB, total, margin, 256 * MiB)); // one MiB short
    CHECK(!kvflash_gpu_mass_release_ws_helps(9999 * MiB, 0, 0, margin, 256 * MiB));              // unknown total: no guard, never refused
}

// The live question stores are host memory no request reserved (24 MiB a store on qwen35-27B, 4 a slot): the list's bytes
// are held on the host ledger, put by put, and given back when a store leaves.
namespace {
KvFlashUqStore mkb(int hi, uint64_t hash, size_t floats) {
    KvFlashUqStore st; st.lo = hi - 1; st.hi = hi; st.n = 1; st.hash = hash; st.rows.assign(floats, (float)hi);
    return st;
}
}  // namespace
TEST_CASE(KvflashUqFixture, the_host_ledger_holds_bytes_outside_a_reservation_up_to_its_budget) {
    KvFlashHostLedger lg(100);
    CHECK(lg.hold(60));
    CHECK(lg.used_bytes() == 60);
    CHECK(!lg.hold(41));                                                // past the budget: refused, nothing charged
    CHECK(lg.used_bytes() == 60);
    CHECK(lg.hold(40));
    CHECK(lg.hold(-100));                                               // a release is never refused
    CHECK(lg.used_bytes() == 0);
}
// A request that fits the budget alone, a ledger with no reservation, and question stores no session owns.
KvFlashAdmitInput uq_admit_in() {
    KvFlashAdmitInput in;
    in.slot_free = true; in.quota_chunks = 352; in.headroom_blocks = 8;
    in.prompt_tokens = 1000; in.max_new_tokens = 24; in.max_ctx = 131072; in.chunk_tokens = 64; in.chunk_bytes = 17 * 1024 * 1024 / 8;
    return in;
}
TEST_CASE(KvflashUqFixture, a_request_that_fits_the_budget_alone_is_not_busy_behind_an_idle_slots_question_stores) {
    const KvFlashAdmitInput in = uq_admit_in();
    KvFlashHostLedger lg(kvflash_host_need_bytes(in));                  // the budget fits this request alone
    auto hold = [&](int64_t d) { return lg.hold(d); };
    KvFlashUqSlots slots(2);
    CHECK(kvflash_uq_put_held(slots[1], mkb(8, 1, 64), 4, hold));       // an earlier request's capture; that request is gone
    CHECK(kvflash_uq_put_held(slots[1], mkb(9, 2, 36), 4, hold));
    CHECK(lg.admit(1, in) == KvFlashAdmit::BusyHost);                   // not HostTooSmall: deferred, and nothing live to finish
    const KvFlashUqReclaimed r = kvflash_uq_reclaim_idle(slots, {false, false}, hold);
    CHECK(r.stores == 2);
    CHECK(r.bytes == 400);
    CHECK(slots[1].empty());
    CHECK(lg.used_bytes() == 0);
    CHECK(lg.admit(1, in) == KvFlashAdmit::Ok);                         // the retried admit
}
TEST_CASE(KvflashUqFixture, a_reclaim_never_touches_the_stores_of_a_slot_with_a_live_request_or_a_parked_session) {
    KvFlashHostLedger lg(1000);
    auto hold = [&](int64_t d) { return lg.hold(d); };
    KvFlashUqSlots slots(3);
    CHECK(kvflash_uq_put_held(slots[0], mkb(100, 1, 10), 4, hold));     // slot 0: a live request
    CHECK(kvflash_uq_put_held(slots[1], mkb(100, 1, 20), 4, hold));     // slot 1: a parked session
    CHECK(kvflash_uq_put_held(slots[2], mkb(100, 1, 5), 4, hold));      // slot 2: nobody
    const KvFlashUqReclaimed r = kvflash_uq_reclaim_idle(slots, {true, true, false}, hold);
    CHECK(r.stores == 1);
    CHECK(r.bytes == 20);
    CHECK(slots[0].size() == 1u);
    CHECK(slots[1].size() == 1u);
    CHECK(slots[2].empty());
    CHECK(lg.used_bytes() == 120);
}
TEST_CASE(KvflashUqFixture, with_nothing_to_reclaim_the_admit_stays_busy_and_is_not_retried) {
    const KvFlashAdmitInput in = uq_admit_in();
    KvFlashHostLedger lg(kvflash_host_need_bytes(in));
    auto hold = [&](int64_t d) { return lg.hold(d); };
    KvFlashUqSlots slots(2);
    CHECK(kvflash_uq_put_held(slots[1], mkb(8, 1, 64), 4, hold));       // slot 1's session is parked: its stores stay
    CHECK(lg.admit(1, in) == KvFlashAdmit::BusyHost);
    const KvFlashUqReclaimed r = kvflash_uq_reclaim_idle(slots, {false, true}, hold);
    CHECK(r.stores == 0);                                               // 0 is the engine's "do not retry": the admit is deferred
    CHECK(r.bytes == 0);
    CHECK(slots[1].size() == 1u);
    CHECK(lg.used_bytes() == 256);
    CHECK(lg.admit(1, in) == KvFlashAdmit::BusyHost);
    CHECK(kvflash_uq_reclaim_idle(slots, {false, true}, hold).stores == 0);   // and a second ask frees nothing either
}
// A parked session's next turn: the parked reservation makes room, the new total is asked; refused, the parked one is back.
struct UqResume {
    KvFlashAdmitInput parked = uq_admit_in(), turn = uq_admit_in();
    KvFlashHostLedger lg;
    UqResume() : lg((turn.prompt_tokens = 1200, kvflash_host_need_bytes(turn))) { parked.max_new_tokens = 0; }   // the budget fits the resumed turn alone
    KvFlashAdmit ask() { lg.release(1); const KvFlashAdmit v = lg.admit(2, turn); if (v != KvFlashAdmit::Ok) lg.admit(1, parked); return v; }
};
TEST_CASE(KvflashUqFixture, a_resume_that_fits_the_budget_alone_is_not_busy_behind_an_idle_slots_question_stores) {
    UqResume u;
    auto hold = [&](int64_t d) { return u.lg.hold(d); };
    CHECK(u.lg.admit(1, u.parked) == KvFlashAdmit::Ok);                 // session A parked in slot 0
    KvFlashUqSlots slots(2);
    CHECK(kvflash_uq_put_held(slots[1], mkb(8, 1, 64), 4, hold));       // slot 1: a request that retired tainted, its stores stay
    CHECK(u.ask() == KvFlashAdmit::BusyHost);                           // no other parked session: nothing to evict, nothing live to finish
    int asks = 0;
    auto reclaim = [&] { asks++; return kvflash_uq_reclaim_idle(slots, {true, false}, hold).stores; };
    CHECK(kvflash_resume_refused(true, false, false, reclaim) == KvFlashResumeRefused::reclaimed_retry);
    CHECK(slots[1].empty());
    CHECK(u.ask() == KvFlashAdmit::Ok);                                 // the one more try
    CHECK(kvflash_resume_refused(true, false, false, reclaim) == KvFlashResumeRefused::wait_parked);   // a second refusal waits
    CHECK(asks == 2);
}
TEST_CASE(KvflashUqFixture, a_busy_resume_with_nothing_to_reclaim_waits_parked_and_is_not_retried) {
    UqResume u;
    auto hold = [&](int64_t d) { return u.lg.hold(d); };
    CHECK(u.lg.admit(1, u.parked) == KvFlashAdmit::Ok);
    KvFlashUqSlots slots(2);
    CHECK(kvflash_uq_put_held(slots[0], mkb(8, 1, 64), 4, hold));       // the parked session's own stores: kept
    CHECK(u.ask() == KvFlashAdmit::BusyHost);
    int asks = 0;
    auto reclaim = [&] { asks++; return kvflash_uq_reclaim_idle(slots, {true, false}, hold).stores; };
    CHECK(kvflash_resume_refused(true, false, false, reclaim) == KvFlashResumeRefused::wait_parked);
    CHECK(asks == 1);
    CHECK(slots[0].size() == 1u);
    // the stores are asked for only when the resume would wait: an eviction, a cold drop and a refusal never reclaim
    CHECK(kvflash_resume_refused(true, true, false, reclaim) == KvFlashResumeRefused::evict_other_retry);
    CHECK(kvflash_resume_refused(false, false, false, reclaim) == KvFlashResumeRefused::drop_cold);
    CHECK(kvflash_resume_refused(true, false, true, reclaim) == KvFlashResumeRefused::refuse);
    CHECK(asks == 1);
}
TEST_CASE(KvflashUqFixture, a_put_holds_the_stores_bytes_and_an_lru_eviction_releases_the_evicted_ones) {
    KvFlashHostLedger lg(1000);
    auto hold = [&](int64_t d) { return lg.hold(d); };
    std::vector<KvFlashUqStore> v;
    CHECK(kvflash_uq_put_held(v, mkb(100, 1, 10), 2, hold));
    CHECK(lg.used_bytes() == 40);
    CHECK(kvflash_uq_put_held(v, mkb(200, 2, 20), 2, hold));
    CHECK(lg.used_bytes() == 120);
    std::vector<int> ev;
    CHECK(kvflash_uq_put_held(v, mkb(300, 3, 5), 2, hold, &ev));        // past the cap: 100 (40 bytes) goes
    CHECK(ev == std::vector<int>({100}));
    CHECK(lg.used_bytes() == 100);
    CHECK(kvflash_uq_put_held(v, mkb(200, 2, 1), 2, hold));             // the same key: the old rows are released
    CHECK(lg.used_bytes() == 24);
    CHECK(lg.used_bytes() == kvflash_uq_bytes(v));
    CHECK(kvflash_uq_put_held(v, mkb(400, 4, 3), 0, hold));             // max 0 keeps none
    CHECK(v.empty());
    CHECK(lg.used_bytes() == 0);
}
TEST_CASE(KvflashUqFixture, a_clear_releases_every_store_of_the_list_and_nothing_else) {
    KvFlashHostLedger lg(1000);
    auto hold = [&](int64_t d) { return lg.hold(d); };
    KvFlashUqSlots s(2);
    CHECK(kvflash_uq_put_held(s[0], mkb(100, 1, 10), 4, hold));
    CHECK(kvflash_uq_put_held(s[0], mkb(200, 2, 20), 4, hold));
    CHECK(kvflash_uq_put_held(s[1], mkb(100, 1, 7), 4, hold));
    kvflash_uq_clear_held(s[0], hold);
    CHECK(s[0].empty());
    CHECK(lg.used_bytes() == 28);                                       // the other slot's store stays charged
    kvflash_uq_clear_held(s[0], hold);                                  // idempotent
    CHECK(lg.used_bytes() == 28);
}
TEST_CASE(KvflashUqFixture, a_put_the_ledger_refuses_keeps_the_old_stores_and_does_not_throw) {
    KvFlashHostLedger lg(100);
    auto hold = [&](int64_t d) { return lg.hold(d); };
    std::vector<KvFlashUqStore> v;
    CHECK(kvflash_uq_put_held(v, mkb(100, 1, 10), 4, hold));
    std::vector<int> ev{7};
    CHECK(!kvflash_uq_put_held(v, mkb(200, 2, 20), 4, hold, &ev));      // 40 + 80 > 100
    REQUIRE(v.size() == 1u);
    CHECK(v[0].hi == 100);
    CHECK(v[0].rows.size() == 10u);
    CHECK(ev == std::vector<int>({7}));                                 // nothing was evicted
    CHECK(lg.used_bytes() == 40);
    CHECK(kvflash_uq_put_held(v, mkb(200, 2, 20), 1, hold));            // the eviction makes the room: charged by the net change
    CHECK(lg.used_bytes() == 80);
    KvFlashHostLedger none(0);
    std::vector<KvFlashUqStore> w;
    CHECK(!kvflash_uq_put_held(w, mkb(100, 1, 10), 4, [&](int64_t d) { return none.hold(d); }));
    CHECK(w.empty());
}
