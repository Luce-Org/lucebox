#include "CppUnitTestFramework.hpp"
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include "../../src/common/dflash_target.h"
#include "../../src/common/kvflash_pager.h"
#include "../../src/common/kvflash_refusals.h"
#include "../../src/qwen35/prefill_helpers.h"
#include "../../src/qwen35/qwen35_daemon.h"
#include "../../src/qwen35moe/qwen35moe_daemon.h"
#include <cstdlib>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "kvflash_pager_test_host.h"

namespace {
struct KvflashSpecSerialFixture {};
}

using namespace luce::common;

// F1: DDTree commits every accepted token before it checks termination and
// never settles, so on EOS/cancel/budget end it finishes one position past AR.
// --ddtree with KVFlash fails closed. Launchers do add it (the container
// entrypoint, harness/clients/common.sh), so the check sits in luce_server
// startup and in both legacy daemons.
TEST_CASE(KvflashSpecSerialFixture, ddtree_with_kvflash_is_refused) {
    CHECK(!kvflash_ddtree_refusal(true, kvflash_pool_requested("4096")).empty());   // --kvflash 4096 or LUCE_KVFLASH
    CHECK(!kvflash_ddtree_refusal(true, kvflash_pool_requested("auto")).empty());
}

TEST_CASE(KvflashSpecSerialFixture, ddtree_alone_or_kvflash_alone_is_accepted) {
    CHECK(kvflash_ddtree_refusal(true, kvflash_pool_requested(nullptr)).empty());
    CHECK(kvflash_ddtree_refusal(true, kvflash_pool_requested("0")).empty());       // pool 0 = off
    CHECK(kvflash_ddtree_refusal(false, kvflash_pool_requested("4096")).empty());
}

TEST_CASE(KvflashSpecSerialFixture, ddtree_refusal_is_one_line_naming_both_flags) {
    const std::string m = kvflash_ddtree_refusal(true, true);
    CHECK(m.find("--ddtree") != std::string::npos);
    CHECK(m.find("--kvflash") != std::string::npos);
    CHECK(m.find('\n') == std::string::npos);
}

// F2: a plain-decode (adaptive one-token) step used to skip the snapshot when
// the budget was settled and ample. io.emit can still end the request on that
// step (cancel, stop string), and the repair restores the snapshot. Every
// terminal one-token step must repair to AR's state: the emitted token is
// pending, cur_pos stays at the step start. The loop now snapshots every step.
TEST_CASE(KvflashSpecSerialFixture, one_token_step_cancelled_after_emit_ends_in_ars_state) {
    const int committed = 700;
    // The step verified the seed (pending at 700) and emitted it; the request
    // ends there. AR ends in the same place: the emitted token is pending, its
    // row not consumed, cur_pos 700.
    const auto s = spec_step_settle(committed, /*commit_n=*/1, /*emitted=*/1,
                                    /*budget_close_fired=*/false, /*terminal=*/true);
    CHECK(s.repair);                      // needs the pre-step snapshot
    CHECK(s.replay_n == 0);
    CHECK(s.new_committed == committed);  // not committed + 1
}

TEST_CASE(KvflashSpecSerialFixture, one_token_step_that_keeps_going_needs_no_repair) {
    const auto s = spec_step_settle(700, 1, 1, false, false);
    CHECK(!s.repair);
    CHECK(s.new_committed == 701);
}

// F3/F4/F5: a verify span never contains a reselect point except at its first
// row, and never reaches into a chunk that is not allocated yet except p0's
// own. The step starting at the reselect point then reselects first, exactly
// where AR does, from the query of the last row before it.
namespace {
const std::function<bool(int)> kNoneAllocated = [](int) { return false; };
const std::function<bool(int)> kAllAllocated = [](int) { return true; };
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_without_kvflash_or_points_is_the_proposal) {
    CHECK(kvflash_verify_span_cap(100, 16, {}, 0, kNoneAllocated) == 16);
    CHECK(kvflash_verify_span_cap(100, 1, {}, 0, kNoneAllocated) == 1);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_stops_before_a_reselect_point) {
    // point at p0 + 5: rows 0..4 only; the row at the point starts the next step.
    CHECK(kvflash_verify_span_cap(60, 16, {65}, 0, kAllAllocated) == 5);
    // point at p0 + n: the whole span fits, the point is the next step's first row.
    CHECK(kvflash_verify_span_cap(60, 5, {65}, 0, kAllAllocated) == 5);
    // point at p0 + 1: a single row.
    CHECK(kvflash_verify_span_cap(60, 16, {61}, 0, kAllAllocated) == 1);
    // point beyond the span: no clamp.
    CHECK(kvflash_verify_span_cap(60, 4, {65}, 0, kAllAllocated) == 4);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_a_point_at_p0_is_the_first_row_and_does_not_clamp) {
    // The step reselects before row 0, then runs the whole span.
    CHECK(kvflash_verify_span_cap(64, 16, {64}, 0, kAllAllocated) == 16);
    // Points before p0 are history.
    CHECK(kvflash_verify_span_cap(70, 16, {64}, 0, kAllAllocated) == 16);
    // The nearest point after p0 wins.
    CHECK(kvflash_verify_span_cap(64, 16, {64, 70, 80}, 0, kAllAllocated) == 6);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_n_one_is_never_clamped_below_one) {
    CHECK(kvflash_verify_span_cap(60, 1, {61}, 64, kNoneAllocated) == 1);
    CHECK(kvflash_verify_span_cap(447, 1, {}, 64, kNoneAllocated) == 1);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_does_not_allocate_a_chunk_ahead_of_ar) {
    // 64-token chunks, 448-token pool = 7 chunks (0..6). p0 = 447 is the last
    // row of chunk 6, chunk 7 is not allocated: AR would not touch it for one
    // more token, so the span is one row.
    const std::function<bool(int)> upto6 = [](int c) { return c <= 6; };
    CHECK(kvflash_verify_span_cap(447, 16, {}, 64, upto6) == 1);
    // p0 at a chunk start: its own chunk is allocated by the first row, as AR does.
    CHECK(kvflash_verify_span_cap(448, 16, {}, 64, upto6) == 16);
    // p0 mid-chunk, the span ends inside the chunk: no clamp.
    CHECK(kvflash_verify_span_cap(400, 16, {}, 64, upto6) == 16);
    // The next chunk is already allocated (rows from an earlier rejected span): no clamp.
    const std::function<bool(int)> upto7 = [](int c) { return c <= 7; };
    CHECK(kvflash_verify_span_cap(447, 16, {}, 64, upto7) == 16);
    // The crossing needs a new chunk 3 rows in.
    CHECK(kvflash_verify_span_cap(445, 16, {}, 64, upto6) == 3);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_takes_the_tighter_of_point_and_chunk) {
    const std::function<bool(int)> upto6 = [](int c) { return c <= 6; };
    CHECK(kvflash_verify_span_cap(445, 16, {447}, 64, upto6) == 2);
    CHECK(kvflash_verify_span_cap(445, 16, {460}, 64, upto6) == 3);
}

// Reselect points as positions: AR reselects after emitting generated token r
// (r % tau == 0), before forwarding it; token r sits at p0 + (r - g0 - 1) for a
// step at p0 that has emitted g0 tokens (the seed is token g0 + 1).
TEST_CASE(KvflashSpecSerialFixture, reselect_positions_map_generated_index_to_position) {
    // hist0 100, tau 64: r = 64 is the only point up to 70. g0 = 60, p0 = 160.
    CHECK(kvflash_reselect_positions(100, 60, 160, 16, 64) == std::vector<int>({163}));
    // Nothing in range.
    CHECK(kvflash_reselect_positions(100, 0, 100, 16, 64).empty());
    // r == g0 + 1 is the seed's own point: handled before the step, not listed.
    CHECK(kvflash_reselect_positions(100, 63, 163, 16, 64).empty());
    CHECK(kvflash_reselect_positions(100, 0, 100, 16, 0).empty());   // tau off
}

// q_cap holds the query of the last consumed row. A spec step stages the
// query of every verify row; after acceptance only the accepted rows count,
// so q_cap must equal what AR holds after the same number of steps: the query
// of position (new_committed - 1). Rejected rows never reach q_cap.
TEST_CASE(KvflashSpecSerialFixture, q_stage_row_picks_the_last_consumed_position) {
    // verify at p0 = 100, 8 rows (positions 100..107).
    CHECK(kvflash_q_stage_row(100, 8, 101) == 0);      // 1 row consumed: new_committed 101
    CHECK(kvflash_q_stage_row(100, 8, 104) == 3);      // accepted 4
    CHECK(kvflash_q_stage_row(100, 8, 108) == 7);      // whole block
    CHECK(kvflash_q_stage_row(100, 8, 100) == -1);     // nothing consumed: q_cap stays
    CHECK(kvflash_q_stage_row(100, 8, 109) == -1);     // beyond the staged rows
    CHECK(kvflash_q_stage_row(-1, 0, 50) == -1);       // nothing staged
}

TEST_CASE(KvflashSpecSerialFixture, q_cap_after_k_of_n_accepted_equals_q_cap_after_k_ar_steps) {
    const int n = 8, per_row = 6, p0 = 100;
    // AR: one query per step; q_cap after step at position p holds q(p).
    auto query = [&](int pos, int i) { return (float)(pos * 10 + i); };
    std::vector<float> stage((size_t)n * per_row);
    for (int r = 0; r < n; r++) for (int i = 0; i < per_row; i++) stage[(size_t)r * per_row + i] = query(p0 + r, i);
    for (int k = 1; k <= n; k++) {
        std::vector<float> ar_q_cap(per_row);
        for (int i = 0; i < per_row; i++) ar_q_cap[i] = query(p0 + k - 1, i);   // after k AR steps
        std::vector<float> spec_q_cap(per_row, -1.0f);
        const int row = kvflash_q_stage_row(p0, n, p0 + k);
        CHECK(row == k - 1);
        std::copy(stage.begin() + (size_t)row * per_row, stage.begin() + (size_t)(row + 1) * per_row,
                  spec_q_cap.begin());
        CHECK(spec_q_cap == ar_q_cap);
    }
}

// F5: the seed's own chunk is exempt from the span cap (the first row always
// allocates it, as AR does), but AR allocates it only after it emitted the
// seed. A request that ends on that emission leaves the pager untouched in AR;
// a spec step that verifies first allocates (and evicts) before the emit loop,
// and the terminal repair restores recurrent state only.
namespace {
constexpr int kF5Pool = 512, kF5Chunk = 64;

struct F5Pager {
    ggml_context *        ctx = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    TestPager             pager;

    F5Pager() {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < 2; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kF5Pool, 2));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kF5Pool, 2));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        KvFlashConfig c;
        c.pool_tokens = kF5Pool;
        c.chunk_tokens = kF5Chunk;
        c.sink_chunks = 1;
        c.tail_window_chunks = 4;
        pager.attach(c, k, v);
    }
    ~F5Pager() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    F5Pager(const F5Pager &) = delete;
    F5Pager & operator=(const F5Pager &) = delete;

    // Full pool: chunks 0..7 resident, the next position opens chunk 8.
    void fill() { pager.alloc_span(0, kF5Pool); }
    std::string state() const {
        return pager.debug_resident(16) + " cur=" + std::to_string(pager.cur_chunk());
    }
    bool alloc_row(int pos) { return pager.slot_for(pos) >= 0; }
};
}

TEST_CASE(KvflashSpecSerialFixture, terminal_seed_at_a_new_chunk_leaves_the_pager_as_ar_does) {
    const int p0 = kF5Pool;   // first row of chunk 8, pool full
    int remaining = 1;        // the seed is the last token the request may emit

    F5Pager ar;               // AR: emit, stop, never forward
    ar.fill();
    const std::string ar_before = ar.state();
    remaining--;
    if (remaining > 0) ar.alloc_row(p0);
    CHECK(ar.state() == ar_before);

    // The old spec order: verify (allocate) first, emit after.
    F5Pager old_order;
    old_order.fill();
    old_order.alloc_row(p0);
    CHECK(old_order.state() != ar_before);   // the hazard: allocation + eviction stay

    // The spec step as the loop runs it.
    F5Pager spec;
    spec.fill();
    remaining = 1;
    const bool ar_hop = kvflash_seed_takes_ar_step(
        p0, kF5Chunk, [&](int c) { return spec.pager.is_resident(c); });
    if (ar_hop) {
        remaining--;
        if (remaining <= 0) { /* ends here, as AR */ }
        else spec.alloc_row(p0);
    } else {
        spec.alloc_row(p0);   // verify
        remaining--;
    }
    CHECK(spec.state() == ar_before);
}

TEST_CASE(KvflashSpecSerialFixture, seed_takes_the_ar_step_only_where_it_opens_an_unallocated_chunk) {
    const std::function<bool(int)> upto7 = [](int c) { return c <= 7; };
    CHECK(kvflash_seed_takes_ar_step(512, 64, upto7));     // opens chunk 8
    CHECK(!kvflash_seed_takes_ar_step(511, 64, upto7));    // last row of chunk 7
    CHECK(!kvflash_seed_takes_ar_step(100, 64, upto7));    // mid chunk
    CHECK(!kvflash_seed_takes_ar_step(512, 64, [](int) { return true; }));  // already allocated
    CHECK(!kvflash_seed_takes_ar_step(512, 0, upto7));     // no KVFlash
}

// F1: the legacy daemon (test_dflash --daemon --ddtree, launched by
// pflash/dflash_client.py with LUCE_KVFLASH inherited) reaches the backend
// without passing luce_server's startup check. It must refuse the same
// combination with exit 2 before it loads anything: the bogus target path
// would make init() return 1, so 2 proves the refusal ran first.
namespace {
struct ScopedEnv {
    std::string name, old; bool had;
    ScopedEnv(const char * n, const char * v) : name(n) {
        const char * o = std::getenv(n);
        had = o != nullptr; if (had) old = o;
        if (v) setenv(n, v, 1); else unsetenv(n);
    }
    ~ScopedEnv() { if (had) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};
Qwen35DaemonArgs daemon_args(bool ddtree, const char * draft = nullptr) {
    Qwen35DaemonArgs a;
    a.target_path = "/nonexistent/no-such-target.gguf";
    a.draft_path = draft;
    a.ddtree_mode = ddtree;
    return a;
}
}

TEST_CASE(KvflashSpecSerialFixture, legacy_daemon_refuses_ddtree_with_kvflash_before_init) {
    ScopedEnv env("LUCE_KVFLASH", "4096");
    CHECK(run_qwen35_daemon(daemon_args(true)) == 2);
    CHECK(run_qwen35moe_daemon(daemon_args(true)) == 2);
}

TEST_CASE(KvflashSpecSerialFixture, legacy_daemon_without_the_combination_is_not_refused) {
    {
        ScopedEnv env("LUCE_KVFLASH", "4096");           // kvflash alone: reaches init, which fails (1)
        CHECK(run_qwen35_daemon(daemon_args(false)) == 1);
    }
    {
        ScopedEnv env("LUCE_KVFLASH", nullptr);          // ddtree alone
        CHECK(run_qwen35_daemon(daemon_args(true)) == 1);
    }
}

// P2: reselection scores the history AR has at that point, and AR applies the
// thinking budget before it records a token. A forced close that coincides
// with a reselect point replaces the seed, so the scorer must see close[0].
TEST_CASE(KvflashSpecSerialFixture, reselect_history_holds_the_budgeted_seed_like_ar) {
    BudgetHook hook;
    hook.close_token_ids = {9, 8};
    hook.marker_token_ids = {8};
    hook.hard_limit_remaining = 3;
    ThinkingBudget spec_budget(hook, 3);     // remaining 3: the next token is force-closed
    ThinkingBudget ar_budget(hook, 3);

    const int32_t seed = 5;
    const int32_t ar_recorded = ar_budget.apply(seed).token;   // AR: apply, then record, then reselect
    CHECK(ar_recorded == 9);
    CHECK(kvflash_reselect_seed_token(spec_budget, seed) == ar_recorded);
    // The real accounting stays at emission.
    CHECK(spec_budget.remaining() == 3);
    CHECK(!spec_budget.forced_close());
    CHECK(spec_budget.apply(seed).token == 9);
}

TEST_CASE(KvflashSpecSerialFixture, reselect_history_keeps_the_seed_when_the_budget_does_not_replace_it) {
    BudgetHook hook;
    hook.close_token_ids = {9, 8};
    hook.marker_token_ids = {8};
    hook.hard_limit_remaining = 3;
    ThinkingBudget b(hook, 100);
    CHECK(kvflash_reselect_seed_token(b, 5) == 5);
    CHECK(b.remaining() == 100);
    ThinkingBudget off;                      // no hook
    CHECK(kvflash_reselect_seed_token(off, 5) == 5);
    ThinkingBudget closing(hook, 100);       // the natural marker passes through unchanged
    CHECK(kvflash_reselect_seed_token(closing, 8) == 8);
}

// The AR hop ends before the span computation, so a step that starts after it
// sees the same inputs as any other step: g0 counts the tokens emitted before
// the seed, the seed is not in out_tokens. Codex's counterexample: hist0 = 450,
// seed at the chunk start 512, 62 generated, tau 64. The seed is token 63; with
// the seed counted twice g0 would be 63 and the reselect at g = 64 would fall
// on a different row.
TEST_CASE(KvflashSpecSerialFixture, span_cap_inputs_at_a_chunk_start_are_those_of_a_step_without_the_hop) {
    const size_t hist0 = 450;
    const int p0 = 512, generated = 62, tau = 64, v_len = 16;
    const std::function<bool(int)> upto7 = [](int c) { return c <= 7; };
    // out_tokens holds `generated` tokens; the seed is pending, not in it.
    const int cap = kvflash_step_span_cap(hist0, /*entry=*/0, (size_t)generated, p0, v_len,
                                          tau, 64, upto7);
    // Reference: the pieces the step computes by hand.
    const int g0 = generated;
    const int ref = kvflash_verify_span_cap(
        p0, v_len, kvflash_reselect_positions(hist0, g0, p0, v_len, tau), 64, upto7);
    CHECK(cap == ref);
    // Counting the seed twice moves the reselect point: not equal.
    const int shifted = kvflash_verify_span_cap(
        p0, v_len, kvflash_reselect_positions(hist0, g0 + 1, p0, v_len, tau), 64, upto7);
    CHECK(shifted != ref);
    // tau(g) = max(64, (450 + g) / 45): g = 64 reselects at row p0 + 1.
    CHECK(ref == 1);
}

// Production reads: no path appends the seed to out_tokens before the span
// computation of a spec step. The hop block pushes and then leaves the step
// (forward + continue, or a return); nothing between its end and the span
// computation may push the seed (last_tok), with or without the budget applied.
TEST_CASE(KvflashSpecSerialFixture, no_production_path_appends_the_seed_before_the_span_computation) {
    std::string path = __FILE__;
    path = path.substr(0, path.rfind('/')) + "/../../src/qwen35/qwen35_backend.cpp";
    std::ifstream f(path);
    REQUIRE(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();
    CHECK(src.find("seed_pre_emitted") == std::string::npos);
    const size_t end_hop = src.find("// end of the AR hop");
    const size_t span = src.find("kvflash_step_span_cap(");
    REQUIRE(end_hop != std::string::npos);
    REQUIRE(span != std::string::npos);
    REQUIRE(end_hop < span);
    // (The DDTree branch between them pushes its accepted path and bonus and
    // leaves the iteration; KVFlash refuses DDTree. The seed is last_tok.)
    const std::string between = src.substr(end_hop, span - end_hop);
    CHECK(between.find("out_tokens.push_back(last_tok") == std::string::npos);
    CHECK(between.find("apply(last_tok") == std::string::npos);
}

// The history the reselect scores, built by the one helper the step calls:
// generated tokens, then the pending seed as AR records it (budget applied),
// cut to g tokens. Previously the lambda in do_spec_decode did this inline.
TEST_CASE(KvflashSpecSerialFixture, reselect_history_is_generated_then_the_budgeted_seed_cut_to_g) {
    BudgetHook hook;
    hook.close_token_ids = {9, 8};
    hook.marker_token_ids = {8};
    hook.hard_limit_remaining = 3;
    ThinkingBudget closing(hook, 3);        // the next token is force-closed
    const std::vector<int32_t> out = {100, 101, 102};
    std::vector<int32_t> hist = {1, 2, 3, 4};   // base 4 = prompt
    // g == g_seed: the seed is token 4 of the generation, replaced by close[0].
    kvflash_reselect_history(hist, 4, out, 0, /*g=*/4, /*g_seed=*/4, closing, /*last_tok=*/5);
    CHECK((hist == std::vector<int32_t>{1, 2, 3, 4, 100, 101, 102, 9}));
    CHECK(closing.remaining() == 3);        // accounting stays at emission
    // g inside the emitted tokens: no seed, cut to g.
    kvflash_reselect_history(hist, 4, out, 0, /*g=*/2, /*g_seed=*/4, closing, 5);
    CHECK((hist == std::vector<int32_t>{1, 2, 3, 4, 100, 101}));
    // No pending seed: nothing appended.
    kvflash_reselect_history(hist, 4, out, 0, /*g=*/3, /*g_seed=*/4, closing, -1);
    CHECK((hist == std::vector<int32_t>{1, 2, 3, 4, 100, 101, 102}));
}

// Pass 6, AR handoff: hist0 = 450, 62 outputs, committed = 512, tau = 64. A
// forced-close hop emits generated token 63 and hands off; AR emits token 64
// next and must reselect there. Counting from AR's entry it would see 1.
TEST_CASE(KvflashSpecSerialFixture, ar_handoff_counts_reselect_from_the_request_origin) {
    const size_t hist0 = 450, origin = 0;
    std::vector<int32_t> out(63, 7);                       // 62 + the hop's token
    const size_t entry_size = out.size();
    out.push_back(8);                                      // AR's first token: global 64
    const int g_origin = (int)(out.size() - origin), g_entry = (int)(out.size() - entry_size);
    CHECK(g_origin == 64);
    CHECK(g_entry == 1);   // the old local count
    CHECK(!kvflash_reselect_points(hist0, 63, g_origin, 64).empty());
    CHECK(kvflash_reselect_points(hist0, 0, g_entry, 64).empty());
    // the history AR scores holds every emitted token
    std::vector<int32_t> hist(hist0, 1);
    kvflash_history_set_generated(hist, hist0, out, origin);
    CHECK(hist.size() == hist0 + 64);
}

// Production reads: every AR handoff inside the spec loop goes through the one
// ar_tail lambda, which syncs the history and passes the origin; the hop's
// decision goes through kvflash_seed_takes_ar_step.
TEST_CASE(KvflashSpecSerialFixture, spec_loop_handoffs_carry_origin_and_hop_uses_the_action_helper) {
    std::string path = __FILE__;
    path = path.substr(0, path.rfind('/')) + "/../../src/qwen35/qwen35_backend.cpp";
    std::ifstream f(path);
    REQUIRE(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();
    const size_t guard = src.find("history_guard{this, out_tokens");
    REQUIRE(guard != std::string::npos);
    const std::string tail = src.substr(guard);
    size_t calls = 0;
    for (size_t at = tail.find("do_ar_decode("); at != std::string::npos; at = tail.find("do_ar_decode(", at + 1)) {
        // skip comment mentions
        const size_t ls = tail.rfind('\n', at);
        if (tail.compare(tail.find_first_not_of(' ', ls + 1), 2, "//") != 0) calls++;
    }
    const size_t def_end = tail.find("bool Qwen35Backend::sync_remote_draft_features");
    (void)def_end;
    CHECK(calls == 1);
    CHECK(tail.find("history_guard.entry);") != std::string::npos);
    CHECK(tail.find("kvflash_seed_takes_ar_step(") != std::string::npos);
    CHECK(src.find("kvflash_ar_reselect_g(") != std::string::npos);
}

// Pass 7: the EOS floor (LUCE_MIN_TOKENS, default off) with KVFlash is refused
// at startup. Its tool-injection replay forwards tokens without reselect checks
// and the step-start reselect can score an EOS the floor then discards.
TEST_CASE(KvflashSpecSerialFixture, min_tokens_with_kvflash_is_refused) {
    CHECK(!kvflash_min_tokens_refusal(true, "8").empty());
    CHECK(kvflash_min_tokens_refusal(true, "0").empty());
    CHECK(kvflash_min_tokens_refusal(true, "").empty());
    CHECK(kvflash_min_tokens_refusal(true, nullptr).empty());
    CHECK(kvflash_min_tokens_refusal(false, "8").empty());
}

TEST_CASE(KvflashSpecSerialFixture, legacy_daemon_refuses_min_tokens_with_kvflash_before_init) {
    ScopedEnv kv("LUCE_KVFLASH", "4096");
    ScopedEnv mt("LUCE_MIN_TOKENS", "8");
    CHECK(run_qwen35_daemon(daemon_args(false, "/nonexistent/draft.gguf")) == 2);
    CHECK(run_qwen35moe_daemon(daemon_args(false, "/nonexistent/draft.gguf")) == 2);
    // No draft model: AR with KVFlash and the floor is unchanged, it reaches init (1).
    CHECK(run_qwen35_daemon(daemon_args(false)) == 1);
}

// The AR step uploads its graph inputs (token embedding, M-RoPE positions) after
// the build: build_target_step frees and reallocates the step graph, so inputs
// written before it land in the old graph's buffer. Repeated AR steps share a
// shape and hid it; the first AR step after a prefill or a verify block (any
// spec-to-AR handoff, and AR itself after prefill) read stale positions and
// wrote a K row with wrong rope dims (max abs diff 20.6 on K against 0.35 for
// numerics, measured on the 3090).
TEST_CASE(KvflashSpecSerialFixture, ar_step_uploads_its_inputs_after_the_graph_build) {
    std::string path = __FILE__;
    path = path.substr(0, path.rfind('/')) + "/../../src/qwen35/qwen35_backend.cpp";
    std::ifstream f(path);
    REQUIRE(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();
    const size_t def = src.find("bool Qwen35Backend::do_ar_decode(");
    REQUIRE(def != std::string::npos);
    const std::string body = src.substr(def);
    const size_t build = body.find("build_target_step(sg_");
    const size_t embed = body.find("tensor_set(sg_.inp_embed");
    const size_t positions = body.find("tensor_set(sg_.positions");
    REQUIRE(build != std::string::npos);
    REQUIRE(embed != std::string::npos);
    REQUIRE(positions != std::string::npos);
    CHECK(build < embed);
    CHECK(build < positions);
}

// AR decode builds its attention mask from fill_slot_mask. Without a position
// bound it exposes every row of a resident chunk, including rows above cur_pos
// that hold rejected drafts or an earlier turn's tokens.
TEST_CASE(KvflashSpecSerialFixture, slot_mask_hides_rows_above_the_decode_position) {
    TestPager p;
    KvFlashConfig c;
    c.pool_tokens = 512;
    c.chunk_tokens = 64;
    c.sink_chunks = 1;
    c.tail_window_chunks = 4;
    REQUIRE(p.attach(c, {}, {}));
    p.alloc_span(0, 3 * 64);                               // chunks 0..2 resident
    const int cur_pos = 2 * 64 + 10;                       // inside the tail chunk
    std::vector<uint16_t> open((size_t)p.arena_tokens()), cut((size_t)p.arena_tokens());
    p.fill_slot_mask(open.data());
    const int tail_slot = p.slot_of(2 * 64);
    CHECK(open[(size_t)tail_slot + 40] == 0);              // unbounded: the defect
    p.fill_slot_mask(cut.data(), cur_pos + 1);             // AR at cur_pos attends rows <= cur_pos
    for (int i = 0; i < 64; i++) {
        const bool visible = 2 * 64 + i <= cur_pos;
        CHECK(cut[(size_t)tail_slot + i] == (visible ? 0 : 0xFC00));
    }
    CHECK(cut[(size_t)p.slot_of(0) + 5] == 0);             // older chunks stay visible
    CHECK(cut[(size_t)p.slot_of(64) + 63] == 0);
}

// A verify block that cannot get a pool slot is not a spec failure: the caller
// finishes the request with plain decode. kvflash_verify_slots reports the row
// that failed so verify_batch can raise verify_pool_exhausted().
namespace {
struct FullAt {                                            // slot_for fails from `limit` on
    int64_t limit;
    int slot_for(int64_t pos) { return pos < limit ? (int)pos : -1; }
};
}  // namespace

TEST_CASE(KvflashSpecSerialFixture, verify_slots_reports_pool_exhaustion_at_the_failing_row) {
    std::vector<int> slots;
    int failed = -1;
    FullAt room{1000};
    CHECK(kvflash_verify_slots(room, 100, 16, slots, &failed));
    CHECK(slots.size() == 16);
    CHECK(failed == -1);
    FullAt full{105};
    CHECK(!kvflash_verify_slots(full, 100, 16, slots, &failed));
    CHECK(failed == 105);
}

namespace {
struct StubTarget : DFlashTarget {
    bool verify_batch(const std::vector<int32_t> &, int, int &, std::vector<int32_t> *, bool) override { return false; }
    bool snapshot_kv() override { return true; }
    bool restore_kv() override { return true; }
    bool is_eos(int) const override { return false; }
    bool embed_tokens(const int32_t *, int, float *) const override { return false; }
    bool project_hidden_to_tokens(const float *, int, std::vector<int32_t> &) override { return false; }
    int hidden_size() const override { return 0; }
    int mask_token_id() const override { return 0; }
    const std::vector<int> & capture_layer_ids() const override { return ids_; }
    std::vector<int> ids_;
};
}

// Backends that never page keep the default: a failed verify is a real failure.
TEST_CASE(KvflashSpecSerialFixture, a_target_without_a_pager_never_reports_pool_exhaustion) {
    StubTarget t;
    CHECK(!t.verify_pool_exhausted());
}
