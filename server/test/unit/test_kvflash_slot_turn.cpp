// Pure unit tests for the C1.4 per-slot turn bookkeeping (kvflash_slot_turn.h):
// mode-4 plan on/off, first-pass cuts, per-step snapshot/capture flags and
// the sticky checkpoints, the rung ladder and the park point. No ggml, no GPU.
#include "CppUnitTestFramework.hpp"
#include "kvflash_test_seams.h"
#include "../../src/common/kvflash_slot_turn.h"
#include "../../src/common/kvflash_uq.h"

#include <array>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotTurnFixture {};

constexpr int32_t kImStart = 7, kUser = 9, kCt = 64, kQ = 32, kU = 512, kT = 64;

// A prompt of n tokens whose last user turn starts at `user_at` (-1: none).
std::vector<int32_t> prompt_with_user(int n, int user_at) {
    std::vector<int32_t> p((size_t)n, 1);
    if (user_at >= 0 && user_at + 1 < n) { p[(size_t)user_at] = kImStart; p[(size_t)user_at + 1] = kUser; }
    return p;
}
}  // namespace

TEST_CASE(KvflashSlotTurnFixture, cuts_force_r_then_u_cuts_and_a_last_cut_of_at_least_t_queries) {
    // r at 100 lands on a cut end; 100..2020 is 1920 tokens: 512,512,512,384 >= 64 untouched
    CHECK((kvflash_slot_turn_cuts(0, 100, 2020, kU, kT) == std::vector<int>{100, 612, 1124, 1636, 2020}));
    // 0..520 from r: the 8-token last cut would miss queries; its predecessor shrinks to end - 64
    CHECK((kvflash_slot_turn_cuts(0, 0, 520, kU, kT) == std::vector<int>{456, 520}));
    // a turn shorter than T is one cut and captures what there is
    CHECK((kvflash_slot_turn_cuts(1000, 1010, 1040, kU, kT) == std::vector<int>{1010, 1040}));
    // r == kv_offset: no cut before r
    CHECK((kvflash_slot_turn_cuts(1000, 1000, 1600, kU, kT) == std::vector<int>{1512, 1600}));
    // r before kv_offset clamps to it; r at end gives a first pass with no replay tokens
    CHECK((kvflash_slot_turn_cuts(1000, 10, 1600, kU, kT) == std::vector<int>{1512, 1600}));
    CHECK((kvflash_slot_turn_cuts(0, 600, 600, kU, kT) == std::vector<int>{512, 600}));
    CHECK(kvflash_slot_turn_cuts(600, 600, 600, kU, kT).empty());
}

TEST_CASE(KvflashSlotTurnFixture, plan_runs_mode4_only_with_the_profile_and_a_pool_that_does_not_fit) {
    const auto prompt = prompt_with_user(5000, 4800);          // 79 chunks > 32 blocks
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(t.first_pass());
    CHECK(t.r == 4800);
    CHECK(t.end == 5000);
    CHECK(t.reserve_chunks == (5000 - 4800 + kCt - 1) / kCt + 1);
    CHECK(t.snapshot_at_r());
    CHECK(!t.cuts.empty());
    CHECK(t.cuts.back() == 5000);
    CHECK(std::find(t.cuts.begin(), t.cuts.end(), 4800) != t.cuts.end());

    const KvFlashSlotTurn off = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false);
    CHECK(!off.recall());
    CHECK(off.end == 5000);
    CHECK(off.cuts.empty());

    // the user turn starts inside the first chunk: nothing sealed to score, LRU turn
    const KvFlashSlotTurn early = kvflash_slot_turn_plan(prompt_with_user(5000, 40), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(!early.recall());
    CHECK(early.r == -1);
    // 20 chunks + reserve fit 32 blocks: LRU turn
    const KvFlashSlotTurn fits = kvflash_slot_turn_plan(prompt_with_user(1280, 1100), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(!fits.recall());
    CHECK(fits.r == -1);
}

TEST_CASE(KvflashSlotTurnFixture, plan_recalls_the_tail_of_a_cold_turn_too_long_to_replay) {
    // The Modal needle smoke: one 81,430-token cold turn, its user turn at token 17. Uncapped it ran LRU only
    // ("skip (81430 tokens fit 256 blocks or no turn)") and answered a code it had never kept.
    const std::vector<int32_t> prompt = prompt_with_user(81430, 17);
    const int cap = kvflash_max_turn_tokens(kCt, 256, 18, 60);
    CHECK(!kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, 256, kU, kT, true).recall());
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, 256, kU, kT, true, -1, false, cap);
    CHECK(t.first_pass());
    CHECK(t.r == 81430 - cap);
    CHECK(t.reserve_chunks == 256 - 18 - 60 - 2);
    // a turn within the cap keeps its user-turn R
    CHECK(kvflash_slot_turn_plan(prompt_with_user(20000, 15000), 0, kImStart, kUser, kCt, 256, kU, kT, true, -1, false, cap).r == 15000);
}

TEST_CASE(KvflashSlotTurnFixture, plan_on_a_resumed_session_rewinds_to_the_new_user_turn_only) {
    auto prompt = prompt_with_user(6000, 5900);
    prompt[(size_t)100] = kImStart; prompt[(size_t)101] = kUser;   // an older user turn inside the session
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 5880, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(t.first_pass());
    CHECK(t.kv_offset == 5880);
    CHECK(t.r == 5900);
    CHECK((t.cuts == std::vector<int>{5900, 6000}));
    // no user turn past the resume point: R is the resume point, nothing to snapshot
    const KvFlashSlotTurn none = kvflash_slot_turn_plan(prompt_with_user(6000, 100), 5880, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(none.first_pass());
    CHECK(none.r == 5880);
    CHECK(!none.snapshot_at_r());
    // nothing to prefill
    CHECK(!kvflash_slot_turn_plan(prompt, 6000, kImStart, kUser, kCt, kQ, kU, kT, true).recall());
}

TEST_CASE(KvflashSlotTurnFixture, next_cut_follows_the_plan_in_the_first_pass_and_u_cuts_in_the_replay) {
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    int pos = 0;
    std::vector<int> ends;
    while (int n = kvflash_slot_turn_next_cut(t, pos, kU)) { pos += n; ends.push_back(pos); }
    CHECK(ends == t.cuts);
    CHECK(kvflash_slot_turn_next_cut(t, 5000, kU) == 0);
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);          // 4800..5000 in one cut
    CHECK(kvflash_slot_turn_next_cut(t, 4000, kU) == kvflash_next_cut(4000, 5000, kU) - 4000);
    KvFlashSlotTurn lru;
    lru.end = 1300;
    CHECK(kvflash_slot_turn_next_cut(lru, 0, kU) == kvflash_next_cut(0, 1300, kU));
}

// G-D: a turn that ends in prefill (max_tokens 1: no decode step) parks the state after its last prompt
// token. With a tail the turn's final cut leaves the last `tail` tokens to a cut of their own, so the engine
// can stash the state before them and park that far short.
TEST_CASE(KvflashSlotTurnFixture, a_tail_cut_leaves_the_last_tokens_to_a_cut_of_their_own_except_in_the_first_pass) {
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU, 1) == 200);      // the first pass keeps its capture cut whole
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU, 1) == 199);      // the replay stops one short of the end
    CHECK(kvflash_slot_turn_next_cut(t, 4999, kU, 1) == 1);        // and the last token is its own cut
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU, 0) == 200);     // off: unchanged
    CHECK(kvflash_slot_turn_next_cut(t, 4000, kU, 1) == kvflash_slot_turn_next_cut(t, 4000, kU));   // a cut that does not end the turn is untouched
    KvFlashSlotTurn lru;                                              // an LRU turn: U cuts, the last one short by a token
    lru.end = 1300;
    int pos = 0, cuts = 0, last = 0;
    while (int n = kvflash_slot_turn_next_cut(lru, pos, kU, 1)) { pos += n; ++cuts; last = n; }
    CHECK(pos == 1300);
    CHECK(last == 1);
    CHECK(cuts == 4);                                                  // 512, 512, 275, 1
    lru.end = 1;                                                       // a one-token turn is one cut
    CHECK(kvflash_slot_turn_next_cut(lru, 0, kU, 1) == 1);
    lru.end = 1300; lru.ckpt = 1297;                                   // the header cut still stops at the checkpoint, then 2 and 1
    pos = 0;
    std::vector<int> ends;
    while (int n = kvflash_slot_turn_next_cut(lru, pos, kU, 1)) { pos += n; ends.push_back(pos); }
    CHECK((ends == std::vector<int>{512, 1024, 1297, 1299, 1300}));
    // a tail of 4 (R's replay window): the last four tokens are one cut, whatever U cuts came before
    KvFlashSlotTurn t4;
    t4.end = 1300;
    pos = 0;
    ends.clear();
    while (int n = kvflash_slot_turn_next_cut(t4, pos, kU, 4)) { pos += n; ends.push_back(pos); }
    CHECK((ends == std::vector<int>{512, 1024, 1296, 1300}));
    t4.end = 3;                                                        // a turn shorter than the tail is one cut
    CHECK(kvflash_slot_turn_next_cut(t4, 0, kU, 4) == 3);
    t4.end = 1300; t4.ckpt = 1297;                                     // the header cut wins: 1297 is within the tail, so 3 tokens follow whole
    pos = 0; ends.clear();
    while (int n = kvflash_slot_turn_next_cut(t4, pos, kU, 4)) { pos += n; ends.push_back(pos); }
    CHECK((ends == std::vector<int>{512, 1024, 1297, 1300}));
}

TEST_CASE(KvflashSlotTurnFixture, the_step_ending_at_r_snapshots_and_the_step_ending_the_first_pass_captures) {
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    int pos = 0;
    int snaps = 0, caps = 0;
    while (int n = kvflash_slot_turn_next_cut(t, pos, kU)) {
        const KvFlashTurnStep s = kvflash_slot_turn_step(t, pos, n);
        if (s.snapshot) { snaps++; CHECK(pos + n == 4800); }
        if (s.capture) { caps++; CHECK(pos + n == 5000); CHECK(n >= kT); }
        pos += n;
    }
    CHECK(snaps == 1);
    CHECK(caps == 1);
    KvFlashSlotTurn replay = t;
    replay.phase = KvFlashTurnPhase::Replay;
    CHECK(!kvflash_slot_turn_step(replay, 4800, 200).capture);
    CHECK(!kvflash_slot_turn_step(replay, 4288, 512).snapshot);
}

TEST_CASE(KvflashSlotTurnFixture, the_checkpoint_is_the_last_im_start_and_cuts_stop_there_except_in_the_first_pass) {
    auto prompt = prompt_with_user(5000, 4800);
    prompt[4990] = kImStart;                                           // the assistant header of the generation prompt
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(t.ckpt == 4990);
    CHECK(t.first_pass());
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);            // the first pass keeps its capture cut whole
    CHECK(!kvflash_slot_turn_step(t, 4800, 200).ckpt);
    t.phase = KvFlashTurnPhase::Replay;
    // a recall turn's sticky checkpoint is its R snapshot (a first-pass state, the one the cold arm
    // has at R): the replay does not stop at the header and takes no snapshot of its own
    CHECK(kvflash_sticky_ckpt_pos(t) == t.r);
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);
    CHECK(!kvflash_slot_turn_step(t, 4800, 200).ckpt);
    CHECK(!kvflash_slot_turn_step(t, 4800, 190).ckpt);
    const KvFlashSlotTurn lru = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false);
    CHECK(lru.ckpt == 4990);
    CHECK(kvflash_sticky_ckpt_pos(lru) == 4990);                        // an LRU turn: the header, from the only pass
    int pos = 0, n_ckpt = 0;
    while (int n = kvflash_slot_turn_next_cut(lru, pos, kU)) { n_ckpt += kvflash_slot_turn_step(lru, pos, n).ckpt; pos += n; CHECK(n <= kU); }
    CHECK(pos == 5000);
    CHECK(n_ckpt == 1);
    // resumed at the header itself: nothing before it to checkpoint in this turn
    CHECK(kvflash_slot_turn_plan(prompt, 4990, kImStart, kUser, kCt, kQ, kU, kT, true).ckpt == -1);
    CHECK(kvflash_sticky_ckpt_pos(kvflash_slot_turn_plan(prompt, 4990, kImStart, kUser, kCt, kQ, kU, kT, false)) == -1);
    CHECK(kvflash_slot_turn_plan(prompt_with_user(300, -1), 0, kImStart, kUser, kCt, kQ, kU, kT, true).ckpt == -1);
}

TEST_CASE(KvflashSlotTurnFixture, resume_point_is_the_history_end_else_the_prompt_end_checkpoint_else_cold) {
    CHECK(kvflash_resume_point(4535, 4535, 4470, -1, 4674) == 4535);     // the reply re-tokenized as generated
    CHECK(kvflash_resume_point(4500, 4535, 4470, -1, 4674) == 4470);     // it did not: back to the prompt end
    CHECK(kvflash_resume_point(4535, 4535, 4470, -1, 4535) == 4470);     // nothing new past the history: replay the reply
    CHECK(kvflash_resume_point(4535, 4535, -1, -1, 4535) == -1);         // no checkpoint: cold
    CHECK(kvflash_resume_point(4000, 4535, 4470, -1, 4674) == -1);       // edited before the checkpoint: cold
    CHECK(kvflash_resume_point(0, 4535, 4470, -1, 4674) == -1);
    CHECK(kvflash_resume_point(10, 10, 10, -1, 20) == 10);
    CHECK(kvflash_resume_point(10, 0, 0, -1, 20) == -1);
}

TEST_CASE(KvflashSlotTurnFixture, need_chunks_is_what_the_next_cut_adds_past_the_allocated_chunks) {
    CHECK(kvflash_slot_need_chunks(5022, 142, kCt, 79) == 2);        // chunk 78 is partial; 79 and 80 are new
    CHECK(kvflash_slot_need_chunks(5164, 16, kCt, 81) == 0);         // decode inside chunk 80
    CHECK(kvflash_slot_need_chunks(5180, 16, kCt, 81) == 1);         // the next 16 tokens cross into chunk 81
    CHECK(kvflash_slot_need_chunks(0, 512, kCt, 0) == 8);            // a full first cut
    CHECK(kvflash_slot_need_chunks(100, 0, kCt, 2) == 0);
    CHECK(kvflash_slot_need_chunks(100, 10, 0, 2) == 0);
}

// C1.5 F1: two rungs per session, trunk below ckpt. The resume takes the highest rung the prompt
// covers (Rule B); a new checkpoint moves the older one to the trunk (Rule A).
TEST_CASE(KvflashSlotTurnFixture, resume_point_takes_the_highest_rung_the_prompt_covers) {
    // history 4535, ckpt 4470 (the header), trunk 4000 (the previous header)
    CHECK(kvflash_resume_point(4535, 4535, 4470, 4000, 4674) == 4535);   // extends the history: len
    CHECK(kvflash_resume_point(4500, 4535, 4470, 4000, 4674) == 4470);   // diverges in the reply: the header
    CHECK(kvflash_resume_point(4200, 4535, 4470, 4000, 4674) == 4000);   // diverges between the rungs: the trunk
    CHECK(kvflash_resume_point(3999, 4535, 4470, 4000, 4674) == -1);     // below both: cold
    CHECK(kvflash_resume_point(4535, 4535, 4470, 4000, 4535) == 4470);   // identical prompt: the rung below prompt_len
    CHECK(kvflash_resume_point(4470, 4535, 4470, 4000, 4470) == 4000);   // a prompt ending at the header: the trunk
    CHECK(kvflash_resume_point(4200, 4535, -1, 4000, 4674) == 4000);     // no ckpt, a trunk
    CHECK(kvflash_resume_point(4535, 4535, -1, -1, 4535) == -1);
    CHECK(kvflash_resume_point(4500, 4535, 4000, 4470, 4674) == 4470);   // the largest wins whatever the argument order
}

// G-D: park one token short. The slot keeps the state before its last fed token (the stash) and
// parks there, so the history is len-1 tokens and an identical prompt still has a token to prefill.
TEST_CASE(KvflashSlotTurnFixture, park_point_is_one_token_short_when_the_stash_holds_that_state) {
    CHECK(kvflash_park_point(4535, 4534, 4470, 4000) == 4534);   // the stash is the state before the last fed token
    CHECK(kvflash_park_point(4535, -1, 4470, 4000) == 4535);     // no stash (chain round, no decode step): park at len
    CHECK(kvflash_park_point(4535, 4530, 4470, 4000) == 4535);   // a stale stash (not len-1): park at len
    CHECK(kvflash_park_point(4535, 4534, 4535, 4000) == 4535);   // a rung at len: parking below it would orphan the rung
    CHECK(kvflash_park_point(4535, 4534, 4534, -1) == 4534);     // a rung at len-1 stays valid: it is the state at L
    CHECK(kvflash_park_point(1, 0, -1, -1) == 1);                // nothing below len-1 worth parking
    CHECK(kvflash_park_point(4535, 4531, 4470, 4000, 4) == 4531);   // a tail of 4: a state up to 4 tokens back is parked as it is
    CHECK(kvflash_park_point(4535, 4534, 4470, 4000, 4) == 4534);   // and a decode turn's one-back state too
    CHECK(kvflash_park_point(4535, 4530, 4470, 4000, 4) == 4535);   // more than the tail back: not a park point
    CHECK(kvflash_park_point(4535, 4531, 4470, 4000, 1) == 4535);   // tail 1 (the default): only len-1
    CHECK(kvflash_park_point(4535, 4531, 4532, 4000, 4) == 4535);   // a rung above the state: park at len
    CHECK(kvflash_park_point(4535, 4535, 4470, 4000, 4) == 4535);   // a stash at len is no stash
}

TEST_CASE(KvflashSlotTurnFixture, a_same_prompt_resumes_at_the_park_point_and_prefills_one_token) {
    const int len = 4535, P = kvflash_park_point(len, len - 1, 4470, 4000);
    // the prompt equals the history of len tokens; the session holds the first P of them
    const int L = kvflash_resume_point(P, P, 4470, 4000, len);
    CHECK(L == len - 1);
    CHECK(len - L == 1);                                          // exactly the pending token
    CHECK(kvflash_resume_point(P, P, -1, -1, len) == len - 1);    // with no rung at all (it was cold before)
    // without the stash the same prompt rewound to a rung (the G-D failure)
    CHECK(kvflash_resume_point(len, len, 4470, 4000, len) == 4470);
}

TEST_CASE(KvflashSlotTurnFixture, a_continuation_resumes_at_the_park_point_and_the_rung_order_holds_below_it) {
    const int len = 4535, P = kvflash_park_point(len, len - 1, 4470, 4000);
    CHECK(kvflash_resume_point(P, P, 4470, 4000, 4674) == P);     // re-rendered reply shares the history: resume at P, 140 new tokens
    CHECK(kvflash_resume_point(4500, P, 4470, 4000, 4674) == 4470);   // diverges in the reply: the header, as before
    CHECK(kvflash_resume_point(4200, P, 4470, 4000, 4674) == 4000);   // between the rungs: the trunk, as before
    CHECK(kvflash_resume_point(3999, P, 4470, 4000, 4674) == -1);     // below both: cold, as before
    const KvFlashRungs r = kvflash_rungs_after_resume(4470, 4000, P);  // at P both rungs stay
    CHECK(r.ckpt == 4470); CHECK(r.trunk == 4000);
}

TEST_CASE(KvflashSlotTurnFixture, the_rungs_after_a_resume_are_the_one_at_L_and_the_one_below_it) {
    KvFlashRungs r = kvflash_rungs_after_resume(4470, 4000, 4535);       // at len: both kept
    CHECK(r.ckpt == 4470); CHECK(r.trunk == 4000);
    r = kvflash_rungs_after_resume(4470, 4000, 4470);                    // at the ckpt: it is the fallback, the trunk stays
    CHECK(r.ckpt == 4470); CHECK(r.trunk == 4000);
    r = kvflash_rungs_after_resume(4470, 4000, 4000);                    // at the trunk: it becomes the fallback, the ckpt is gone
    CHECK(r.ckpt == 4000); CHECK(r.trunk == -1);
    r = kvflash_rungs_after_resume(4470, -1, 4470);
    CHECK(r.ckpt == 4470); CHECK(r.trunk == -1);
    r = kvflash_rungs_after_resume(-1, -1, 4535);
    CHECK(r.ckpt == -1); CHECK(r.trunk == -1);
    r = kvflash_rungs_after_resume(4470, 4000, 4200);                    // L between the rungs (not a rung: the engine never does this): the trunk is the fallback
    CHECK(r.ckpt == 4000); CHECK(r.trunk == -1);
}

// C1.5 F7: on a recall turn the replay stops at the header and checkpoints there when end - R
// reaches LUCE_KVFLASH_END_RUNG_MIN; the first pass and its capture cut are unchanged.
TEST_CASE(KvflashSlotTurnFixture, the_end_rung_stops_the_replay_at_the_header_when_the_policy_allows_it) {
    auto prompt = prompt_with_user(5000, 4800);
    prompt[4990] = kImStart;                                           // the header inside the last U cut
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, 100);
    CHECK(t.first_pass());
    CHECK(t.end_rung);                                                 // end - r = 200 >= 100
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);            // the first pass keeps its capture cut whole
    CHECK(!kvflash_slot_turn_step(t, 4800, 200).ckpt);
    CHECK(kvflash_slot_turn_step(t, 4800, 200).capture);
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 190);            // the replay stops at the header
    CHECK(kvflash_slot_turn_step(t, 4800, 190).ckpt);
    CHECK(kvflash_slot_turn_next_cut(t, 4990, kU) == 10);
    CHECK(!kvflash_slot_turn_step(t, 4990, 10).ckpt);
    CHECK(kvflash_slot_turn_next_cut(t, 4000, kU) == std::min(kvflash_next_cut(4000, 5000, kU), 4990) - 4000);
    CHECK(!kvflash_slot_turn_step(t, 4000, 512).ckpt);
    // the same turn with a higher threshold, or with the policy unset: today's replay
    for (int min : {201, 2048, -1}) {
        KvFlashSlotTurn off = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, min);
        off.phase = KvFlashTurnPhase::Replay;
        CHECK(!off.end_rung);
        CHECK(kvflash_slot_turn_next_cut(off, 4800, kU) == 200);
        CHECK(!kvflash_slot_turn_step(off, 4800, 190).ckpt);
    }
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, 0).end_rung);      // 0: always
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, 200).end_rung);    // at the threshold
    // no header after R (the question ends the prompt): nothing to rung
    CHECK(!kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true, 0).end_rung);
    // an LRU turn ignores the policy: its header checkpoint is there anyway
    const KvFlashSlotTurn lru = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false, 0);
    CHECK(!lru.end_rung);
    CHECK(kvflash_slot_turn_step(lru, 4800, 190).ckpt);
}

TEST_CASE(KvflashSlotTurnFixture, mode5_plan_applies_the_drafters_picks_at_r_with_no_first_pass_and_no_r_snapshot) {
    const auto prompt = prompt_with_user(5000, 4800);          // 79 chunks > 32 blocks
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, -1, true);
    CHECK(!t.recall());                                         // one prefill: the LRU turn's checkpoint rule holds
    CHECK(t.apply_at == 4800);
    CHECK(t.reserve_chunks == (5000 - 4800 + kCt - 1) / kCt + 1);
    CHECK(t.cuts.empty());
    CHECK(!t.snapshot_at_r());
    CHECK(t.ckpt_pass());
    CHECK(!kvflash_slot_turn_step(t, 4500, 300).capture);
    CHECK(!kvflash_slot_turn_step(t, 4500, 300).snapshot);
    // the prefill stops at R, then runs plain cuts
    CHECK(kvflash_slot_turn_next_cut(t, 4700, kU) == 100);
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);
    CHECK(!kvflash_slot_turn_apply_due(t, 4799));
    CHECK(kvflash_slot_turn_apply_due(t, 4800));
    CHECK(kvflash_slot_turn_apply_due(t, 4900));

    // resumed at R: due at once; no apply when the profile is off, the pool fits or nothing is sealed before R
    CHECK(kvflash_slot_turn_apply_due(kvflash_slot_turn_plan(prompt, 4800, kImStart, kUser, kCt, kQ, kU, kT, true, -1, true), 4800));
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false, -1, true).apply_at == -1);
    CHECK(kvflash_slot_turn_plan(prompt_with_user(1000, 900), 0, kImStart, kUser, kCt, kQ, kU, kT, true, -1, true).apply_at == -1);
    CHECK(kvflash_slot_turn_plan(prompt_with_user(5000, 40), 0, kImStart, kUser, kCt, kQ, kU, kT, true, -1, true).apply_at == -1);
    // mode 4 plans never carry an apply point
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true).apply_at == -1);
    CHECK(!kvflash_slot_turn_apply_due(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true), 5000));
}

TEST_CASE(KvflashSlotTurnFixture, park_cfg_tail_on_by_default_decode_stash_off_until_asked) {
    auto c = kvflash_park_cfg(true, nullptr, nullptr, nullptr);
    CHECK(c.tail_on); CHECK(!c.decode_on); CHECK(c.tail == 4); CHECK(c.any());
    c = kvflash_park_cfg(true, nullptr, "1", "1");
    CHECK(c.decode_on); CHECK(c.tail == 1);
    CHECK(!kvflash_park_cfg(true, nullptr, nullptr, "0").decode_on);
    CHECK(kvflash_park_cfg(true, nullptr, "999", nullptr).tail == 64);
    CHECK(kvflash_park_cfg(true, nullptr, "0", nullptr).tail == 1);
    c = kvflash_park_cfg(true, "0", nullptr, "1");           // PARK_SHORT=0 beats everything
    CHECK(!c.any());
    CHECK(!kvflash_park_cfg(false, nullptr, nullptr, "1").any());   // no sticky sessions, no parking
}

TEST_CASE(KvflashSlotTurnFixture, the_query_tail_accumulates_the_last_64_prompt_queries_across_cuts_like_the_serial_engine) {
    KvFlashSlotTurn t;
    t.phase = KvFlashTurnPhase::FirstPass; t.end = 74;
    // cuts of 40 / 30 / 4 tokens: every one holds some of the last 64 prompt tokens (positions 10..73)
    CHECK(kvflash_slot_turn_step(t, 0, 40, 64).q_rows == 30);
    CHECK(kvflash_slot_turn_step(t, 40, 30, 64).q_rows == 30);
    CHECK(kvflash_slot_turn_step(t, 70, 4, 64).q_rows == 4);
    CHECK(kvflash_slot_turn_step(t, 70, 4, 64).capture);
    CHECK(!kvflash_slot_turn_step(t, 40, 30, 64).capture);
    CHECK(kvflash_slot_turn_step(t, 0, 10, 64).q_rows == 0);          // wholly before the window
    CHECK(kvflash_slot_turn_step(t, 0, 40, 0).q_rows == 0);           // off
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_step(t, 70, 4, 64).q_rows == 0);          // only the first pass captures
    // one query per position (row_size 2: the position twice); the three cuts hand over their last q_rows rows
    std::vector<float> tail;
    auto cut = [&](int p, int n, int q_rows) {
        std::vector<float> rows;
        for (int i = n - q_rows; i < n; i++) { rows.push_back((float)(p + i)); rows.push_back((float)(p + i)); }
        kvflash_q_tail_push(tail, rows.data(), q_rows, 2, 64);
    };
    cut(0, 40, 30); cut(40, 30, 30); cut(70, 4, 4);
    CHECK(tail.size() == 64 * 2);
    for (int i = 0; i < 64; i++) { CHECK(tail[(size_t)2 * i] == (float)(10 + i)); CHECK(tail[(size_t)2 * i + 1] == (float)(10 + i)); }
    // a longer history keeps only the newest 64
    std::vector<float> one(2 * 100); for (int i = 0; i < 100; i++) one[2 * i] = one[2 * i + 1] = (float)i;
    tail.clear(); kvflash_q_tail_push(tail, one.data(), 100, 2, 64);
    CHECK(tail.size() == 128); CHECK(tail[0] == 36.0f); CHECK(tail.back() == 99.0f);
}

TEST_CASE(KvflashSlotTurnFixture, the_plan_budgets_the_usable_quota_not_the_full_slice) {
    // settle() holds a live slot at Q - headroom resident, so a prompt that fits Q but not Q - headroom must recall.
    constexpr int kHeadroom = 8;
    CHECK(kvflash_usable_blocks(kQ, kHeadroom) == kQ - kHeadroom);
    CHECK(kvflash_usable_blocks(4, kHeadroom) == 0);     // never negative
    const int end = (kQ - 6) * kCt;                      // 26 chunks + 4 reserve: fits 32, not 24
    const auto prompt = prompt_with_user(end, end - 200);
    CHECK(!kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true).first_pass());
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kvflash_usable_blocks(kQ, kHeadroom), kU, kT, true).first_pass());
}

// ---- v5: the third rung (S1's numbers: T2 parks ckpt 3671 / trunk 2047; P1 is a branch above 3671; P2 branches again) ----
namespace {
using L3 = KvFlashLadder;
bool operator==(const KvFlashLadder & a, const KvFlashLadder & b) { return a.ckpt == b.ckpt && a.trunk == b.trunk && a.extra == b.extra && a.pinned == b.pinned; }
// One turn of the pure rule: resume at the highest rung the prompt covers, then the prompt's own checkpoint at p-7.
struct V5Turn { KvFlashLadder lad; int len = 0; int L = -1; KvFlashLadderMove at_resume, at_ckpt; };
V5Turn v5_turn(const KvFlashLadder & lad, int len, int lcp, int p) {
    V5Turn t;
    t.L = kvflash_resume_point(lcp, len, lad.ckpt, lad.trunk, p, lad.extra);
    t.at_resume = kvflash_ladder_resume(lad, t.L, len);
    t.at_ckpt = kvflash_ladder_ckpt(t.at_resume.to, p - 7);
    t.lad = t.at_ckpt.to; t.len = p + 2;
    return t;
}
}  // namespace

TEST_CASE(KvflashSlotTurnFixture, v5_a_resume_point_may_be_the_third_rung) {
    CHECK(kvflash_resume_point(3678, 3851, 3842, 3837, 3844, 3671) == 3671);   // above the third: the trunk is not covered, the third is
    CHECK(kvflash_resume_point(3678, 3851, 3842, 3837, 3844) == -1);
    CHECK(kvflash_resume_point(3678, 3851, 3842, 3671, 3844, 2047) == 3671);   // the highest covered rung wins
    CHECK(kvflash_resume_point(3678, 3851, 3842, 3837, 3671, 3671) == -1);     // never at the prompt's own end
}

TEST_CASE(KvflashSlotTurnFixture, v5_a_branch_below_the_checkpoint_pins_its_rung_and_the_next_trunk_turn_returns_to_it) {
    KvFlashLadder t2{3671, 2047, -1, false};                       // after T2
    auto p1 = v5_turn(t2, 3680, 3674, 3849);                       // P1: at the checkpoint, no branch event
    CHECK(p1.L == 3671); CHECK(!p1.at_resume.to.pinned);
    CHECK((p1.lad == L3{3842, 3671, 2047, false}));                // the demoted rung rides in the third slot
    auto p2 = v5_turn(p1.lad, p1.len, 3678, 3844);                 // P2 branches below P1's checkpoint
    CHECK(p2.L == 3671);                                           // 173 tokens to prefill, not 1797
    CHECK((p2.at_resume.to == L3{3671, -1, 3671, true}));
    CHECK(p2.at_resume.ckpt_from == kRungTrunk); CHECK(p2.at_resume.extra_from == kRungTrunk);   // one slab, two copies
    CHECK((p2.lad == L3{3837, -1, 3671, true}));                   // the pinned rung leaves the ladder; the ckpt placed is live
    CHECK(p2.at_ckpt.ckpt_from == kRungLive); CHECK(p2.at_ckpt.extra_from == kRungExtra);
    auto t3 = v5_turn(p2.lad, p2.len, 3678, 5291);
    CHECK(t3.L == 3671);                                           // T3 at T2's header rung (the base resumes below it)
}

TEST_CASE(KvflashSlotTurnFixture, v5_a_busy_retry_decides_once_the_first_attempt_cut_the_ladder_and_the_retry_changes_nothing) {
    const KvFlashLadder p1{3842, 3671, 2047, false};
    const auto first = kvflash_ladder_resume(p1, 3671, 3851);      // P2's attempt: the event
    REQUIRE((first.to == L3{3671, -1, 3671, true}));
    const auto retry = kvflash_ladder_resume(first.to, 3671, 3671);   // the history was cut to 3671: the retry resumes at its end
    CHECK((retry.to == first.to));
    CHECK(retry.ckpt_from == kRungCkpt); CHECK(retry.extra_from == kRungExtra);   // from the rungs it holds
    const auto third = kvflash_ladder_resume(retry.to, 3671, 3671);
    CHECK((third.to == first.to));
}

TEST_CASE(KvflashSlotTurnFixture, v5_another_request_after_the_event_keeps_the_pin_when_it_resumes_at_or_above_it_and_drops_it_below) {
    const KvFlashLadder ev{3671, -1, 3671, true};                  // after P2's event, before its checkpoint
    CHECK((kvflash_ladder_resume(ev, 3671, 3671).to.extra == 3671));
    const KvFlashLadder after{3837, 3000, 3671, true};
    CHECK((kvflash_ladder_resume(after, 3837, 3851).to == after));        // a whole-history resume changes nothing
    CHECK((kvflash_ladder_resume(after, 3000, 3851).to == L3{3000, -1, 3000, true}));   // a deeper branch: the pin moves down, the old one goes
    const KvFlashLadder unpinned{3842, 3671, 2047, false};
    CHECK((kvflash_ladder_resume(unpinned, 3842, 3851).to == unpinned));   // at the checkpoint: no event, the demoted rung stays
}

TEST_CASE(KvflashSlotTurnFixture, v5_the_chain_T4_P3_T5_keeps_a_rung_at_the_end_of_every_prompt_like_the_base) {
    KvFlashLadder lad{5280, 3671, 2047, false};
    int len = 5289;
    auto t4 = v5_turn(lad, len, 5287, 6900);    CHECK(t4.L == 5280);
    auto p3 = v5_turn(t4.lad, t4.len, 6896, 6925);   CHECK(p3.L == 6893);
    auto t5 = v5_turn(p3.lad, p3.len, 6900, 8500);   CHECK(t5.L == 6893);
}

TEST_CASE(KvflashSlotTurnFixture, v5_a_ckpt_never_leaves_a_duplicate_or_a_rung_at_the_prompt_end) {
    const auto m = kvflash_ladder_ckpt(L3{3000, 2000, 1000, false}, 4000);
    CHECK((m.to == L3{4000, 3000, 2000, false}));
    CHECK(m.ckpt_from == kRungLive); CHECK(m.trunk_from == kRungCkpt); CHECK(m.extra_from == kRungTrunk);   // the demotion
    CHECK((kvflash_ladder_ckpt(L3{3000, 2000, 1000, true}, 4000).to == L3{4000, 3000, 1000, true}));        // pinned: stays, the trunk below goes
    CHECK((kvflash_ladder_ckpt(L3{3671, -1, 3671, true}, 3837).to == L3{3837, -1, 3671, true}));            // the checkpoint at the pin is the pin
}

// ---- v5: the slab ops of a ladder move never hold more than the three slabs the admission charge covers ----
namespace {
// Runs the ops on labelled slabs: slab i = label i; returns the labels the three rungs end up with (-1 none).
std::array<int, 3> run_ops(const KvFlashLadderMove & m) {
    std::array<int, 4> held{0, 1, 2, 3};
    std::array<int, 3> out{-1, -1, -1};
    for (const KvFlashSlabOp & op : kvflash_ladder_ops(m)) {
        if (op.kind == KvFlashSlabOp::Free) held[(size_t)op.from] = -1;
        else if (op.kind == KvFlashSlabOp::Copy) out[(size_t)op.to] = held[(size_t)op.from];
        else { out[(size_t)op.to] = held[(size_t)op.from]; held[(size_t)op.from] = -1; }
    }
    return out;
}
int held_mask(const KvFlashLadder & l, bool live) { return (l.ckpt > 0) | (l.trunk > 0) << 1 | (l.extra > 0) << 2 | (int)live << 3; }
}  // namespace

TEST_CASE(KvflashSlotTurnFixture, v5_the_branch_copy_frees_the_dead_slabs_first_and_never_holds_a_fourth) {
    const KvFlashLadder codex{3842, 3671, 2047, false};              // Codex's sequence: resume at 3671 copies the trunk into the checkpoint
    const KvFlashLadderMove m = kvflash_ladder_resume(codex, 3671, 3851);
    const auto ops = kvflash_ladder_ops(m);
    REQUIRE(ops.size() == 4u);
    CHECK((ops[0].kind == KvFlashSlabOp::Free && ops[0].from == kRungCkpt));       // the old checkpoint and the old third go before the copy
    CHECK((ops[1].kind == KvFlashSlabOp::Free && ops[1].from == kRungExtra));
    CHECK((ops[2].kind == KvFlashSlabOp::Copy && ops[2].from == kRungTrunk && ops[2].to == kRungCkpt));
    CHECK((ops[3].kind == KvFlashSlabOp::Move && ops[3].from == kRungTrunk && ops[3].to == kRungExtra));
    CHECK(kvflash_ladder_peak(held_mask(codex, false), ops, false) == 3);
    const auto got = run_ops(m);                                     // the rungs end up with the slabs the move names
    CHECK((got == std::array<int, 3>{kRungTrunk, -1, kRungTrunk}));
}

TEST_CASE(KvflashSlotTurnFixture, v5_no_ladder_move_of_the_reachable_states_holds_more_than_three_slabs) {
    const KvFlashLadder states[] = {{3842, 3671, 2047, false}, {3837, -1, 3671, true}, {3837, 3000, 3671, true}, {900, -1, 900, true}, {4000, 3000, 2000, false},
                                    {5000, 4000, 3000, false}, {3671, 2047, -1, false}, {3671, -1, -1, false}};
    for (const KvFlashLadder & s : states) {
        const int len = std::max({s.ckpt, s.trunk, s.extra}) + 9;
        for (int L : {s.ckpt, s.trunk, s.extra, len}) {
            if (L <= 0) continue;
            const KvFlashLadderMove m = kvflash_ladder_resume(s, L, len);
            const auto ops = kvflash_ladder_ops(m);
            CHECK(kvflash_ladder_peak(held_mask(s, false), ops, false) <= 3);
            const auto got = run_ops(m);                             // every rung of the move gets exactly its source
            CHECK(got[0] == (m.to.ckpt > 0 ? m.ckpt_from : -1)); CHECK(got[1] == (m.to.trunk > 0 ? m.trunk_from : -1)); CHECK(got[2] == (m.to.extra > 0 ? m.extra_from : -1));
            const KvFlashLadderMove c = kvflash_ladder_ckpt(m.to, len + 100);       // then the next checkpoint: the snapshot is taken after the frees
            CHECK(kvflash_ladder_peak(held_mask(m.to, false), kvflash_ladder_ops(c), true) <= 3);
        }
    }
}

// ---- LUCE_KVFLASH_UQ_QUERIES in the slot engine: the question's end is a cut, its last rows are captured ----
namespace {
// Cut ends of the pass from `from` to the prompt end.
std::vector<int> pass_ends(const KvFlashSlotTurn & t, int from) {
    std::vector<int> e;
    for (int p = from; p < t.end;) {
        const int n = kvflash_slot_turn_next_cut(t, p, kU);
        if (n <= 0) break;
        e.push_back(p += n);
    }
    return e;
}
}  // namespace

TEST_CASE(KvflashSlotTurnFixture, the_questions_end_is_a_cut_end_of_the_first_pass_the_replay_and_an_lru_turn) {
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    REQUIRE(t.first_pass());
    const std::vector<int> grid = pass_ends(t, 0);
    CHECK(std::find(grid.begin(), grid.end(), 4990) == grid.end());
    kvflash_slot_turn_force_cut(t, 4990);                       // the question ends 10 tokens before the prompt end
    std::vector<int> want = grid;
    want.insert(want.end() - 1, 4990);
    CHECK(pass_ends(t, 0) == want);
    // the last 64 prompt queries still add up over the two cuts the question's end made of the last one
    CHECK(kvflash_slot_turn_step(t, 4800, 190, kT).q_rows == 54);
    CHECK(kvflash_slot_turn_step(t, 4990, 10, kT).q_rows == 10);
    CHECK(kvflash_slot_turn_step(t, 4990, 10, kT).capture);
    CHECK(kvflash_slot_turn_step(t, 4608, 192, kT).snapshot);   // R is still a cut end
    t.phase = KvFlashTurnPhase::Replay;
    CHECK((pass_ends(t, 4800) == std::vector<int>{4990, 5000}));
    // a cut end already there is not doubled; -1 (no question inside the prefill) changes nothing
    KvFlashSlotTurn same = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    kvflash_slot_turn_force_cut(same, 4800);
    CHECK(pass_ends(same, 0) == grid);
    kvflash_slot_turn_force_cut(same, -1);
    CHECK(pass_ends(same, 0) == grid);
    // an LRU turn: the U grid, the checkpoint at the last <|im_start|>, the question's end, the prompt end
    KvFlashSlotTurn lru = kvflash_slot_turn_plan(prompt_with_user(1280, 1100), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    REQUIRE(!lru.recall());
    kvflash_slot_turn_force_cut(lru, 1200);
    CHECK((pass_ends(lru, 0) == std::vector<int>{512, 1024, 1100, 1200, 1280}));
    CHECK((pass_ends(lru, 1200) == std::vector<int>{1280}));    // resumed at the cut: it is behind
}

TEST_CASE(KvflashSlotTurnFixture, the_window_rows_of_the_cuts_add_up_to_the_plan_in_every_pass) {
    struct Case { int n_prompt, user_at, kv_offset, q_lo, q_hi; };
    for (const Case & c : std::vector<Case>{
             {5000, 4800, 0, 4803, 4990},      // the question is the last user turn (a first request that does not fit)
             {5000, 4800, 0, 900, 1030},       // far back, its window [966, 1030) straddles the U grid at 1024
             {5000, 4800, 0, 1000, 1020},      // 20 tokens: the window is the whole question
             {1280, 1100, 0, 1103, 1270},      // an LRU turn (the prompt fits)
             {9000, 8800, 4096, 5000, 5200},   // resumed below the question
         }) {
        const KvFlashHumanTurn q{c.q_lo, c.q_hi};
        KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(c.n_prompt, c.user_at), c.kv_offset, kImStart, kUser, kCt, kQ, kU, kT, true);
        kvflash_slot_turn_force_cut(t, kvflash_uq_cut(q, 64, c.kv_offset, t.end));
        KvFlashUqPlan up = kvflash_uq_plan(q, 64, c.kv_offset, t.end);
        REQUIRE(up.capture);
        auto rows_over = [&](int from) {
            int rows = 0, pos = from;
            for (int e : pass_ends(t, from)) { rows += kvflash_uq_cut_rows(up, pos, e - pos); pos = e; }
            return rows;
        };
        CHECK(rows_over(c.kv_offset) == up.n);
        const std::vector<int> ends = pass_ends(t, c.kv_offset);
        CHECK(std::find(ends.begin(), ends.end(), c.q_hi) != ends.end());
        if (t.first_pass()) {
            int q_rows = 0, pos = c.kv_offset;
            for (int e : ends) { q_rows += kvflash_slot_turn_step(t, pos, e - pos, kT).q_rows; pos = e; }
            CHECK(q_rows == kT);
            // the replay recaptures a window it reaches back over, and only then
            const bool again = kvflash_uq_rewind_recaptures(up, t.r);
            CHECK(again == (c.q_lo > t.r));
            t.phase = KvFlashTurnPhase::Replay;
            CHECK(rows_over(t.r) == (again ? up.n : 0));
        }
    }
}
