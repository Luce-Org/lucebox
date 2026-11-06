// GDN single commit: recurrent-state bookkeeping (qwen35/gdn_single_commit.h). GPU-free.
//
// The gating, rollback and restore plans and the storage exchange are checked
// directly. A toy single-layer model then replays the spec-decode protocol
// (snapshot_kv, verify_batch, rollback_to, restore_kv and the replays of
// qwen35_backend.cpp do_spec_decode, plus request resets) twice, once with
// the parent's semantics and once with the GDN single commit's, and requires
// the committed recurrent state to match after every call.

#include "CppUnitTestFramework.hpp"

#include "qwen35/gdn_single_commit.h"

#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace gp = luce::common::gdn_commit;

struct GdnSingleCommit {};

gp::PathFacts eligible_path() {
    gp::PathFacts f;
    f.fast_rollback = true;
    f.have_state = true;
    f.intermediates_f32 = true;
    return f;
}

// ── Toy device: one delta layer, contents are hashes ──────────────────────

using u64 = uint64_t;
using Conv = std::array<u64, 3>;   // the K-1 = 3 most recent conv inputs

u64 mix(u64 a, u64 b) {
    a ^= b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2);
    a *= 0xff51afd7ed558ccdull;
    return a ^ (a >> 33);
}

// What a verify computes from (state, conv): inter[t] is the state after
// token t, ci the conv window [history | tokens].
struct VerifyOut {
    std::vector<u64> inter;
    std::vector<u64> ci;
};

VerifyOut run_tokens(u64 state, const Conv & conv, const std::vector<u64> & tokens) {
    VerifyOut o;
    o.ci.assign(conv.begin(), conv.end());
    for (u64 t : tokens) {
        state = mix(state, t);
        o.inter.push_back(state);
        o.ci.push_back(t);
    }
    return o;
}

Conv window_at(const std::vector<u64> & ci, int offset) {
    return {ci[(size_t)offset], ci[(size_t)offset + 1], ci[(size_t)offset + 2]};
}

// The parent's Qwen35DFlashTarget, reduced to its effect on one layer.
struct ParentTarget {
    u64 live = 1, snap = 0;
    Conv conv{2, 3, 4}, conv_snap{};
    VerifyOut cap;
    int q = 0;
    int ssm_copies = 0;   // SSM-state copies, the in-graph write included

    void snapshot() { snap = live; conv_snap = conv; ssm_copies++; }
    void verify(const std::vector<u64> & tokens, bool capture) {
        VerifyOut o = run_tokens(live, conv, tokens);
        live = o.inter.back();
        ssm_copies++;
        conv = window_at(o.ci, (int)tokens.size());
        if (capture) { cap = o; q = (int)tokens.size(); }
    }
    // A copy that fails after `layers_done` layers: the parent leaves a
    // partial write in live.
    bool rollback(int commit_n, bool fail = false) {
        if (commit_n >= q) return true;
        if (fail) { live = 0xdeadull; return false; }
        live = cap.inter[(size_t)commit_n - 1];
        conv = window_at(cap.ci, commit_n);
        ssm_copies++;
        return true;
    }
    void restore() { live = snap; conv = conv_snap; ssm_copies++; }
    void reset() { live = 0; conv = {0, 0, 0}; }   // base_buf clear
};

// The GDN single commit's target: the same calls, driven by gdn_single_commit.h, with
// the two physical SSM buffers of the real cache (0 = base_buf, 1 =
// rollback_buf) and TargetCache::ssm_state_swapped.
struct CommitTarget {
    bool keep_base_on = false;
    bool swap_on = false;
    bool single_replay_on = false;
    gp::StateBook book;
    u64 buf[2] = {1, 0};
    bool swapped = false;
    Conv conv{2, 3, 4}, conv_snap{};
    VerifyOut cap;
    // SINGLE_REPLAY: the tokens of the last replay-log verify (its log).
    std::vector<u64> replay_log;
    int ssm_copies = 0;
    int replay_commits = 0;

    u64 & live() { return buf[swapped ? 1 : 0]; }
    u64 & snap() { return buf[swapped ? 0 : 1]; }

    void snapshot() {
        gp::drop_pending(book);                      // gdn_drop_uncommitted
        const bool swap = swap_on && keep_base_on;   // swap_snapshot_block_reason
        gp::on_snapshot(book, swap);
        conv_snap = conv;
        if (!swap) { snap() = live(); ssm_copies++; }
    }
    void materialize() {                             // gdn_materialize_snapshot
        if (!book.snapshot_deferred) return;
        snap() = live();
        ssm_copies++;
        gp::materialized(book);
    }
    // `fail_early`: build or upload failure before any compute.
    void verify(const std::vector<u64> & tokens, bool capture, bool fail_early = false) {
        const int n = (int)tokens.size();
        const bool kb = gp::keep_base_verify(keep_base_on, capture, n);
        // gdn_single_replay_ids on the eligible path (grouped-cols route).
        gp::ReplayFacts facts;
        facts.grouped_cols = true;
        facts.n_tokens = n;
        facts.slot_map = true;
        const bool replay = kb && single_replay_on &&
                            gp::single_replay_block_reason(facts) == nullptr;
        gp::drop_pending(book);
        if (gp::must_materialize(book, kb)) materialize();
        if (fail_early) return;
        VerifyOut o = run_tokens(live(), conv, tokens);
        if (!kb) { live() = o.inter.back(); ssm_copies++; }
        conv = window_at(o.ci, n);
        if (capture) {
            cap = o;
            // A replay-log verify writes no checkpoints: whatever the
            // buffer held before is stale.
            if (replay) cap.inter.assign(cap.inter.size(), 0xbadull);
        }
        if (replay) replay_log = tokens;
        if (kb) gp::on_keep_base_verify(book, n, replay);
    }
    // q_len is cur_pos - base_pos: the last verify's width.
    bool rollback(int commit_n, int q_len, bool fail = false) {
        const gp::RollbackPlan p = gp::plan_rollback(book, commit_n, q_len);
        if (!p.valid) return false;
        if (p.shortcut) return true;
        u64 & dst = p.into_spare ? snap() : live();
        if (fail) { dst = 0xdeadull; return false; }   // book left alone
        if (p.from_replay) {
            // ggml_backend_cuda_gdn_chain_commit: the first ssm_index + 1
            // transitions applied to the live state, into dst.
            const std::vector<u64> prefix(replay_log.begin(),
                                          replay_log.begin() + p.ssm_index + 1);
            dst = run_tokens(live(), conv_snap, prefix).inter.back();
            replay_commits++;
        } else {
            dst = cap.inter[(size_t)p.ssm_index];
            ssm_copies++;
        }
        if (p.copy_conv) conv = window_at(cap.ci, commit_n);
        if (p.into_spare) swapped = !swapped;            // exchange_pairs
        gp::finish_rollback(book, p);
        return true;
    }
    void restore() {
        if (gp::plan_restore(book) == gp::RestoreSsm::kCopyFromSnap) {
            live() = snap();
            ssm_copies++;
        }
        conv = conv_snap;
    }
    void finish() {
        gp::drop_pending(book);
        gp::on_finish(book);
    }
    void reset() {                                   // reset_recurrent_state
        if (swapped) swapped = false;                // unswap_pairs
        buf[0] = 0;                                  // base_buf clear
        conv = {0, 0, 0};
    }
};

std::vector<u64> tokens_of(std::mt19937_64 & rng, int n) {
    std::vector<u64> t((size_t)n);
    for (u64 & x : t) x = rng();
    return t;
}

// Drives both targets through do_spec_decode's call sequences and checks
// the committed state after every call that commits.
struct Protocol {
    ParentTarget parent;
    CommitTarget commit;
    std::mt19937_64 rng;
    int checks = 0;
    bool ok = true;

    Protocol(uint64_t seed, bool keep_base, bool swap, bool single_replay = false)
        : rng(seed) {
        commit.keep_base_on = keep_base;
        commit.swap_on = swap;
        commit.single_replay_on = single_replay;
    }

    void expect_same(const char * where) {
        ++checks;
        if (parent.live != commit.live() || parent.conv != commit.conv) {
            ok = false;
            std::fprintf(stderr, "[gdn-commit] state diverged after %s\n", where);
        }
    }
    void snapshot() { parent.snapshot(); commit.snapshot(); }
    void verify(const std::vector<u64> & t, bool capture) {
        parent.verify(t, capture);
        commit.verify(t, capture);
    }
    void restore_and_replay(const std::vector<u64> & t, int n) {
        parent.restore(); commit.restore();
        expect_same("restore");
        if (n > 0) {
            std::vector<u64> prefix(t.begin(), t.begin() + n);
            verify(prefix, /*capture=*/false);
            expect_same("replay");
        }
    }

    // One spec step: snapshot, capture verify, then rollback (which may fail
    // and fall back), or the slow path, then an optional emit-phase repair.
    void spec_step() {
        const int v_len = 1 + (int)(rng() % 16);
        const std::vector<u64> t = tokens_of(rng, v_len);
        snapshot();
        verify(t, /*capture=*/true);
        const int accept = 1 + (int)(rng() % v_len);
        const uint64_t path = rng() % 8;
        if (path < 5) {
            parent.rollback(accept);
            if (!commit.rollback(accept, v_len)) {
                ok = false;
                std::fprintf(stderr, "[gdn-commit] rollback_to refused\n");
            }
            expect_same("rollback_to");
        } else if (path < 6 && accept < v_len) {
            // A device copy fails part way: the loop restores and replays.
            parent.rollback(accept, /*fail=*/true);
            commit.rollback(accept, v_len, /*fail=*/true);
            restore_and_replay(t, accept);
        } else {
            restore_and_replay(t, accept);
        }
        if (rng() % 3 == 0) {            // EOS, budget close or floor_to_ar
            restore_and_replay(t, (int)(rng() % (accept + 1)));
            if (rng() % 2 == 0) restore_and_replay(t, (int)(rng() % (accept + 1)));
        }
    }

    // AR-burst seed step: one-token capture verify, the state is used as the
    // verify left it; a snapshot only when an emit hook is armed.
    void ar_step() {
        const std::vector<u64> t = tokens_of(rng, 1);
        const bool snap = rng() % 2 == 0;
        if (snap) snapshot();
        verify(t, /*capture=*/true);
        expect_same("AR seed verify");
        if (snap && rng() % 3 == 0) restore_and_replay(t, (int)(rng() % 2));
    }

    // verify failed (before or after compute) or its logits could not be
    // read: restore and give up the request.
    void failed_step() {
        const std::vector<u64> t = tokens_of(rng, 2 + (int)(rng() % 15));
        snapshot();
        if (rng() % 2 == 0) {
            commit.verify(t, /*capture=*/true, /*fail_early=*/true);
        } else {
            verify(t, /*capture=*/true);
        }
        parent.restore(); commit.restore();
        expect_same("restore after a failed verify");
    }

    // A new request: finish (or abort) the decode, zero the recurrent state.
    void request_boundary() {
        if (rng() % 2 == 0) commit.finish();
        parent.reset(); commit.reset();
        expect_same("reset_recurrent_state");
    }

    void run(int steps) {
        for (int step = 0; step < steps && ok; ++step) {
            const uint64_t kind = rng() % 16;
            if (kind < 10) spec_step();
            else if (kind < 13) ar_step();
            else if (kind < 15) failed_step();
            else request_boundary();
        }
    }
};

ggml_tensor tensor_at(void * data, ggml_backend_buffer * buffer, void * extra) {
    ggml_tensor t{};
    t.type = GGML_TYPE_F32;
    t.data = data;
    t.buffer = buffer;
    t.extra = extra;
    t.ne[0] = 4; t.ne[1] = t.ne[2] = t.ne[3] = 1;
    return t;
}

}  // namespace

TEST_CASE(GdnSingleCommit, every_part_is_on_by_default) {
    const gp::Parts defaults;
    CHECK(defaults.keep_base && defaults.swap_snapshot && defaults.single_replay);
    CHECK(gp::parts().keep_base && gp::parts().swap_snapshot && gp::parts().single_replay);
}

TEST_CASE(GdnSingleCommit, keep_base_block_reasons) {
    CHECK(gp::keep_base_block_reason(eligible_path()) == nullptr);

    gp::PathFacts f = eligible_path();
    f.fast_rollback = false;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    f = eligible_path(); f.kvflash = true;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    f = eligible_path(); f.specla = true;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    f = eligible_path(); f.tensor_parallel = true;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    f = eligible_path(); f.n_seq_slots = 2;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    f = eligible_path(); f.bailingmoe3 = true;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    f = eligible_path(); f.moe = true;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    f = eligible_path(); f.ddtree = true;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    f = eligible_path(); f.have_state = false;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
    // F16 checkpoints (LUCE_SINGLE_CHAIN_CHECKPOINT_F32=0) are rounded.
    f = eligible_path(); f.intermediates_f32 = false;
    CHECK(gp::keep_base_block_reason(f) != nullptr);
}

TEST_CASE(GdnSingleCommit, swap_snapshot_needs_keep_base_and_exchangeable_pairs) {
    CHECK(gp::swap_snapshot_block_reason(true, eligible_path(), true) == nullptr);
    CHECK(gp::swap_snapshot_block_reason(false, eligible_path(), true) != nullptr);
    CHECK(gp::swap_snapshot_block_reason(true, eligible_path(), false) != nullptr);
    gp::PathFacts f = eligible_path();
    f.intermediates_f32 = false;
    CHECK(gp::swap_snapshot_block_reason(true, f, true) != nullptr);
    f = eligible_path(); f.tensor_parallel = true;
    CHECK(gp::swap_snapshot_block_reason(true, f, true) != nullptr);
}

TEST_CASE(GdnSingleCommit, keep_base_only_for_multi_token_capture_verifies) {
    CHECK(gp::keep_base_verify(true, true, 16));
    CHECK(gp::keep_base_verify(true, true, 2));
    // The AR-burst seed step uses the state as verify left it.
    CHECK(!gp::keep_base_verify(true, true, 1));
    // Replays (no capture) and ineligible paths keep the parent write.
    CHECK(!gp::keep_base_verify(true, false, 16));
    CHECK(!gp::keep_base_verify(false, true, 16));
}

TEST_CASE(GdnSingleCommit, rollback_plan_without_pending_commit_is_the_parent) {
    const gp::StateBook book;
    for (int q = 1; q <= 16; ++q) {
        for (int c = 1; c <= q + 1; ++c) {
            const gp::RollbackPlan p = gp::plan_rollback(book, c, q);
            CHECK(p.valid);
            CHECK_EQUAL(p.shortcut, c >= q);
            CHECK_EQUAL(p.ssm_index, c - 1);
            CHECK(p.copy_conv);
            CHECK(!p.into_spare);
        }
    }
}

TEST_CASE(GdnSingleCommit, rollback_plan_commits_every_count_after_keep_base) {
    for (int q = 2; q <= 16; ++q) {
        for (bool deferred : {false, true}) {
            gp::StateBook book;
            book.commit_pending = true;
            book.pending_q_len = q;
            book.snapshot_deferred = deferred;
            for (int c = 1; c <= q; ++c) {
                const gp::RollbackPlan p = gp::plan_rollback(book, c, q);
                CHECK(p.valid);
                CHECK(!p.shortcut);
                CHECK_EQUAL(p.ssm_index, c - 1);
                // All accepted: the verify already left the last conv window.
                CHECK_EQUAL(p.copy_conv, c < q);
                CHECK_EQUAL(p.into_spare, deferred);
            }
            // cur_pos moved since the verify: refuse, the caller replays.
            CHECK(!gp::plan_rollback(book, 1, q + 1).valid);
            gp::finish_rollback(book, gp::plan_rollback(book, 1, q));
            CHECK(!book.commit_pending);
            CHECK(!book.snapshot_deferred);
        }
    }
}

TEST_CASE(GdnSingleCommit, deferred_snapshot_is_taken_before_any_live_write) {
    gp::StateBook book;
    gp::on_snapshot(book, /*swap=*/true);
    CHECK(book.snapshot_deferred);
    // The keep-base verify leaves live alone.
    CHECK(!gp::must_materialize(book, /*keeps_live_state=*/true));
    // A rollback without a keep-base verify would overwrite the only P.
    CHECK(!gp::plan_rollback(book, 2, 4).valid);
    // Restore: live is P, the snapshot stays deferred for the replay.
    CHECK(gp::plan_restore(book) == gp::RestoreSsm::kKeepLive);
    CHECK(book.snapshot_deferred);
    CHECK(gp::must_materialize(book, /*keeps_live_state=*/false));
    gp::materialized(book);
    CHECK(gp::plan_restore(book) == gp::RestoreSsm::kCopyFromSnap);
    gp::on_snapshot(book, /*swap=*/true);
    gp::on_finish(book);
    CHECK(!book.snapshot_deferred);
    gp::on_snapshot(book, /*swap=*/false);
    CHECK(!book.snapshot_deferred);
}

TEST_CASE(GdnSingleCommit, exchange_trades_storage_and_keeps_identity) {
    float a_mem[4] = {}, b_mem[4] = {};
    auto * buf_a = reinterpret_cast<ggml_backend_buffer *>(0x1000);
    auto * buf_b = reinterpret_cast<ggml_backend_buffer *>(0x2000);
    int extra_a = 0, extra_b = 0;
    ggml_tensor a = tensor_at(a_mem, buf_a, &extra_a);
    ggml_tensor b = tensor_at(b_mem, buf_b, &extra_b);
    std::vector<ggml_tensor *> live{&a, nullptr};
    std::vector<ggml_tensor *> spare{&b, nullptr};
    bool swapped = false;

    gp::unswap_pairs(live, spare, swapped);          // no-op when canonical
    CHECK(a.data == a_mem);
    CHECK(!swapped);

    gp::exchange_pairs(live, spare, swapped);
    CHECK(swapped);
    CHECK(live[0] == &a);                            // identity kept
    CHECK(a.data == b_mem);
    CHECK(a.buffer == buf_b);
    CHECK(a.extra == &extra_b);
    CHECK(b.data == a_mem);
    CHECK(b.buffer == buf_a);
    CHECK(b.extra == &extra_a);
    CHECK_EQUAL(a.ne[0], (int64_t)4);

    gp::unswap_pairs(live, spare, swapped);
    CHECK(!swapped);
    CHECK(a.data == a_mem);
    CHECK(a.buffer == buf_a);
    CHECK(b.data == b_mem);
    CHECK(b.buffer == buf_b);
}

TEST_CASE(GdnSingleCommit, keep_base_protocol_matches_parent_state) {
    for (uint64_t seed = 1; seed <= 64; ++seed) {
        Protocol run(seed, /*keep_base=*/true, /*swap=*/false);
        run.run(300);
        CHECK(run.ok);
        CHECK(run.checks > 300);
        CHECK(!run.commit.swapped);
    }
}

TEST_CASE(GdnSingleCommit, swap_snapshot_protocol_matches_parent_state) {
    int swapped_runs = 0;
    for (uint64_t seed = 1; seed <= 64; ++seed) {
        Protocol run(seed, /*keep_base=*/true, /*swap=*/true);
        run.run(300);
        CHECK(run.ok);
        CHECK(run.checks > 300);
        if (run.commit.swapped) swapped_runs++;
    }
    CHECK(swapped_runs > 0);   // runs end on either parity
}

TEST_CASE(GdnSingleCommit, swap_without_keep_base_is_the_parent) {
    for (uint64_t seed = 1; seed <= 16; ++seed) {
        Protocol run(seed, /*keep_base=*/false, /*swap=*/true);
        run.run(200);
        CHECK(run.ok);
        CHECK(!run.commit.swapped);
        CHECK_EQUAL(run.commit.ssm_copies, run.parent.ssm_copies);
    }
}

TEST_CASE(GdnSingleCommit, each_accept_count_commits_with_one_ssm_copy) {
    // commit_n = 1 .. 16 after a 16-token verify, the all-accepted case
    // included. Parent: snapshot copy, in-graph write, rollback copy (none
    // when all accepted). KEEP_BASE: snapshot copy and the rollback copy.
    // KEEP_BASE and SWAP_SNAPSHOT: the rollback copy only.
    for (int c = 1; c <= 16; ++c) {
        for (int mode = 0; mode < 2; ++mode) {
            Protocol run(1000 + (uint64_t)c, /*keep_base=*/true, /*swap=*/mode == 1);
            const std::vector<u64> t = tokens_of(run.rng, 16);
            run.snapshot();
            run.verify(t, /*capture=*/true);
            CHECK(run.commit.book.commit_pending);
            run.parent.rollback(c);
            REQUIRE(run.commit.rollback(c, 16));
            run.expect_same("rollback_to");
            CHECK(!run.commit.book.commit_pending);
            CHECK(run.ok);
            CHECK_EQUAL(run.parent.ssm_copies, c < 16 ? 3 : 2);
            CHECK_EQUAL(run.commit.ssm_copies, mode == 1 ? 1 : 2);
            // The pre-step state survives in the snapshot for a repair.
            run.restore_and_replay(t, c - 1);
            CHECK(run.ok);
        }
    }
}

// ── SINGLE_REPLAY ───────────────────────────────────────────────────────

TEST_CASE(GdnSingleCommit, single_replay_inherits_keep_base_and_needs_the_grouped_route) {
    CHECK(gp::single_replay_inherited_reason(true, nullptr) == nullptr);
    CHECK(gp::single_replay_inherited_reason(false, nullptr) != nullptr);
    CHECK(gp::single_replay_inherited_reason(true, "DDTree mode") != nullptr);
    CHECK(std::strcmp(gp::single_replay_inherited_reason(true, "DDTree mode"),
                      "DDTree mode") == 0);

    gp::ReplayFacts r;
    r.grouped_cols = true;
    r.n_tokens = 16;
    r.slot_map = true;
    CHECK(gp::single_replay_block_reason(r) == nullptr);
    r.n_tokens = 2;
    CHECK(gp::single_replay_block_reason(r) == nullptr);
    // The scalar kernel computes g*s + k*delta: not the commit's update.
    r.grouped_cols = false;
    CHECK(gp::single_replay_block_reason(r) != nullptr);
    r.grouped_cols = true;
    r.n_tokens = gp::kSingleReplayMaxTokens + 1;
    CHECK(gp::single_replay_block_reason(r) != nullptr);
    r.n_tokens = 16;
    r.slot_map = false;
    CHECK(gp::single_replay_block_reason(r) != nullptr);
}

TEST_CASE(GdnSingleCommit, replay_plan_commits_from_the_log) {
    for (int q = 2; q <= 16; ++q) {
        for (bool deferred : {false, true}) {
            gp::StateBook book;
            gp::on_keep_base_verify(book, q, /*replay=*/true);
            book.snapshot_deferred = deferred;
            for (int c = 1; c <= q; ++c) {
                const gp::RollbackPlan p = gp::plan_rollback(book, c, q);
                CHECK(p.valid);
                CHECK(!p.shortcut);
                CHECK(p.from_replay);
                CHECK_EQUAL(p.ssm_index, c - 1);    // c transitions
                CHECK_EQUAL(p.copy_conv, c < q);
                // Out of place into the spare with a deferred snapshot, else
                // in place after the parent's snapshot copy.
                CHECK_EQUAL(p.into_spare, deferred);
            }
            CHECK(!gp::plan_rollback(book, 1, q + 1).valid);
            gp::finish_rollback(book, gp::plan_rollback(book, 1, q));
            CHECK(!book.commit_pending);
            CHECK(!book.replay_pending);
        }
    }
    // A checkpoint verify after a replay one plans a copy again.
    gp::StateBook book;
    gp::on_keep_base_verify(book, 8, /*replay=*/true);
    gp::on_keep_base_verify(book, 8, /*replay=*/false);
    CHECK(!gp::plan_rollback(book, 3, 8).from_replay);
    // Restore and a dropped commit forget the log.
    gp::on_keep_base_verify(book, 8, /*replay=*/true);
    (void)gp::plan_restore(book);
    CHECK(!book.commit_pending);
    CHECK(!book.replay_pending);
    gp::on_keep_base_verify(book, 8, /*replay=*/true);
    gp::drop_pending(book);
    CHECK(!book.replay_pending);
    CHECK(!gp::plan_rollback(book, 3, 8).from_replay);
}

TEST_CASE(GdnSingleCommit, single_replay_protocol_matches_parent_state) {
    int commits = 0;
    for (uint64_t seed = 1; seed <= 64; ++seed) {
        Protocol run(seed, /*keep_base=*/true, /*swap=*/false, /*single_replay=*/true);
        run.run(300);
        CHECK(run.ok);
        CHECK(run.checks > 300);
        CHECK(!run.commit.swapped);
        commits += run.commit.replay_commits;
    }
    CHECK(commits > 0);
}

TEST_CASE(GdnSingleCommit, single_replay_with_swap_protocol_matches_parent_state) {
    int commits = 0;
    int swapped_runs = 0;
    for (uint64_t seed = 1; seed <= 64; ++seed) {
        Protocol run(seed, /*keep_base=*/true, /*swap=*/true, /*single_replay=*/true);
        run.run(300);
        CHECK(run.ok);
        CHECK(run.checks > 300);
        commits += run.commit.replay_commits;
        if (run.commit.swapped) swapped_runs++;
    }
    CHECK(commits > 0);
    CHECK(swapped_runs > 0);
}

TEST_CASE(GdnSingleCommit, single_replay_without_keep_base_is_the_parent) {
    for (uint64_t seed = 1; seed <= 16; ++seed) {
        Protocol run(seed, /*keep_base=*/false, /*swap=*/false, /*single_replay=*/true);
        run.run(200);
        CHECK(run.ok);
        CHECK_EQUAL(run.commit.replay_commits, 0);
        CHECK_EQUAL(run.commit.ssm_copies, run.parent.ssm_copies);
    }
}

TEST_CASE(GdnSingleCommit, each_accept_count_commits_from_the_replay_log) {
    // commit_n = 1 .. 16 after a 16-token verify, the all-accepted case
    // included. SINGLE_REPLAY: no checkpoint copy; the snapshot copy only
    // without SWAP_SNAPSHOT, nothing with it; one replay commit.
    for (int c = 1; c <= 16; ++c) {
        for (int mode = 0; mode < 2; ++mode) {
            Protocol run(2000 + (uint64_t)c, /*keep_base=*/true, /*swap=*/mode == 1,
                         /*single_replay=*/true);
            const std::vector<u64> t = tokens_of(run.rng, 16);
            run.snapshot();
            run.verify(t, /*capture=*/true);
            CHECK(run.commit.book.commit_pending);
            CHECK(run.commit.book.replay_pending);
            run.parent.rollback(c);
            REQUIRE(run.commit.rollback(c, 16));
            run.expect_same("rollback_to");
            CHECK(!run.commit.book.commit_pending);
            CHECK(!run.commit.book.replay_pending);
            CHECK(run.ok);
            CHECK_EQUAL(run.commit.ssm_copies, mode == 1 ? 0 : 1);
            CHECK_EQUAL(run.commit.replay_commits, 1);
            run.restore_and_replay(t, c - 1);
            CHECK(run.ok);
        }
    }
}
