// Human-turn / capture-plan / per-session store helpers of LUCE_KVFLASH_UQ_QUERIES (kvflash_uq.h), the CPU selector
// with an extra query, and the pure GPU-scorer guard.

#include "CppUnitTestFramework.hpp"

#include "kvflash_exact_fast.h"
#include "kvflash_gpu_mass.h"
#include "kvflash_mass_synth.h"
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

// A bundle save keeps a slot's rows only when they are the session's own: the slot's store outlives the session that captured it.
TEST_CASE(KvflashUqFixture, a_store_belongs_to_the_history_it_was_captured_from) {
    const std::vector<int32_t> ids{5, 6, 7, 8, 9, 10, 11, 12};
    KvFlashUqStore s;
    s.lo = 4; s.hi = 7; s.n = 2; s.hash = kvflash_uq_hash(ids, 7); s.rows = {1.f, 2.f, 3.f, 4.f};
    CHECK(kvflash_uq_belongs(s, ids));
    auto other = ids; other[2] = 99;                                                       // another session's tokens before the window's end
    CHECK(!kvflash_uq_belongs(s, other));
    other = ids; other[7] = 99;                                                            // tokens after it do not matter (the session grew)
    CHECK(kvflash_uq_belongs(s, other));
    CHECK(!kvflash_uq_belongs(s, std::vector<int32_t>(ids.begin(), ids.begin() + 6)));   // a history shorter than the window's end
    KvFlashUqStore t = s; t.n = 4;                    CHECK(!kvflash_uq_belongs(t, ids));   // more rows than the window holds
    t = s; t.rows.pop_back();                         CHECK(!kvflash_uq_belongs(t, ids));   // ragged rows
    CHECK(!kvflash_uq_belongs(KvFlashUqStore{}, ids));
}
