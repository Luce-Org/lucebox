// KVFlash 2.0 sessions P2: the reachable rows of the admit table (SESSIONS_DESIGN.md 4.2: resume, summary-cold, cold), pure, no GPU.
// The two opencode 1.19.0 compaction requests (synthetic token shapes, no session content) and the
// pick / eligibility / resume-point inputs the engine feeds the plan.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_admit_plan.h"

using namespace luce::common;
using A = KvFlashAdmitAction;

namespace {
struct KvflashAdmitPlanFixture {};

constexpr int32_t IM = 1, SYS = 2, USR = 3, AST = 4;
constexpr int E_ROLE = 1150;   // the design's example: E_c 1088 = 17 chunks

std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t> & b) { a.insert(a.end(), b.begin(), b.end()); return a; }
std::vector<int32_t> run(int32_t base, int n) { std::vector<int32_t> v((size_t)n); for (int i = 0; i < n; ++i) v[(size_t)i] = base + i; return v; }
// <|im_start|>system ...tools... (E_ROLE tokens), then a turn: <|im_start|>user text
std::vector<int32_t> preamble() { return cat({IM, SYS}, run(100, E_ROLE - 2)); }
std::vector<int32_t> turn(int32_t role, int32_t text_base, int n) { return cat({IM, role}, run(text_base, n)); }

struct Parked { std::vector<int32_t> history; const char * client; const char * sid; uint64_t tick; int ckpt = 0, trunk = 0; };

// What Qwen35SeqEngine::admit_bounded computes before it asks for the plan.
KvFlashAdmitPlan plan_of(const std::vector<Parked> & parked, const std::string & client, const std::string & sid, const std::vector<int32_t> & prompt) {
    std::vector<KvFlashOwnedSession> list;
    for (size_t i = 0; i < parked.size(); ++i) list.push_back({{(int)i, &parked[i].history, parked[i].tick}, parked[i].client, parked[i].sid});
    const int e_role = kvflash_preamble_end(prompt, IM, SYS);
    const KvFlashSessionPick pick = kvflash_pick_parked_owner(list, client, sid, prompt, e_role);
    KvFlashAdmitIn in;
    in.e_role = e_role;
    if (pick.exact()) {
        const Parked & p = parked[(size_t)pick.slot];
        in.pick_slot = pick.slot; in.lcp = pick.lcp;
        in.L = kvflash_resume_point(pick.lcp, (int)p.history.size(), p.ckpt, p.trunk, (int)prompt.size());
    }
    in.same_sid_slot = kvflash_same_sid_slot(list, client, sid, pick.slot);
    return kvflash_admit_plan(in);
}

// A session that has had two turns: preamble + user1 + assistant1 + user2, parked after the second reply.
std::vector<int32_t> parked_history() { return cat(cat(cat(preamble(), turn(USR, 1000, 300)), turn(AST, 2000, 40)), turn(USR, 3000, 300)); }
}  // namespace

// ── the rows of 4.2 ─────────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashAdmitPlanFixture, row1_an_eligible_pick_past_the_preamble_with_a_resume_point_resumes) {
    const std::vector<Parked> parked = {{parked_history(), "opencode", "ses_a", 1}};
    const auto prompt = cat(parked[0].history, turn(AST, 4000, 40));
    const KvFlashAdmitPlan p = plan_of(parked, "opencode", "ses_a", prompt);
    CHECK(p.action == A::resume);
    CHECK(p.slot == 0);
    CHECK(p.L == (int)parked[0].history.size());
}

TEST_CASE(KvflashAdmitPlanFixture, row3_no_system_block_under_a_parked_sid_is_cold_beside_it) {
    const std::vector<Parked> parked = {{parked_history(), "opencode", "ses_a", 1}, {parked_history(), "opencode", "ses_b", 2}};
    const auto prompt = turn(USR, 5000, 800);   // no system block: E_role 0
    const KvFlashAdmitPlan p = plan_of(parked, "opencode", "ses_a", prompt);
    CHECK(p.action == A::summary_cold);
    CHECK(p.slot == 0);                  // the slot to keep
    CHECK(p.L == 0);
    // the same request from a sid with no parked instance is an ordinary cold admit
    CHECK(plan_of(parked, "opencode", "ses_c", prompt).action == A::cold);
}

TEST_CASE(KvflashAdmitPlanFixture, a_parked_sid_that_cannot_resume_and_no_parked_session_at_all_admit_cold) {
    const std::vector<Parked> parked = {{parked_history(), "opencode", "ses_a", 1, (int)parked_history().size() - 340}};
    // the post-summary request: same system block, another first user turn, then the summary and the tail
    const auto prompt = cat(cat(preamble(), turn(USR, 7000, 20)), turn(AST, 7100, 600));
    const KvFlashAdmitPlan p = plan_of(parked, "opencode", "ses_a", prompt);
    CHECK(p.action == A::cold);
    CHECK(p.slot == -1);
    CHECK(plan_of({}, "opencode", "ses_a", prompt).action == A::cold);
}

// ── edges of the rows ───────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashAdmitPlanFixture, row1_is_lcp_past_the_preamble_and_a_resume_point_even_when_that_point_is_at_or_before_the_preamble_end) {
    // spec row 1: lcp > E_role and L > 0. The pick shares E_role + 2 tokens and holds a rung at 200 (inside the system
    // block): the saved resume point is valid and the session resumes there
    const std::vector<Parked> parked = {{parked_history(), "opencode", "ses_a", 1, 200, 0}};
    const auto prompt = cat(cat(preamble(), turn(USR, 7000, 20)), turn(AST, 7100, 600));
    KvFlashAdmitPlan p = plan_of(parked, "opencode", "ses_a", prompt);
    CHECK(p.action == A::resume);
    CHECK(p.L == 200);
    // the design's example: E_role 1150, lcp 1152, a rung exactly at 1150 resumes at 1150
    const std::vector<Parked> at_end = {{parked_history(), "opencode", "ses_a", 1, E_ROLE, 0}};
    p = plan_of(at_end, "opencode", "ses_a", prompt);
    CHECK(p.action == A::resume);
    CHECK(p.L == E_ROLE);
}

TEST_CASE(KvflashAdmitPlanFixture, a_pick_whose_lcp_does_not_pass_the_preamble_is_not_row1_whatever_its_resume_point) {
    // lcp == E_role: the request diverges at the first token after the system block (a compaction): never resume
    KvFlashAdmitIn in;
    in.e_role = E_ROLE; in.pick_slot = 0; in.lcp = E_ROLE; in.L = 200; in.same_sid_slot = 0;
    CHECK(kvflash_admit_plan(in).action == A::cold);
    in.lcp = E_ROLE + 1;
    CHECK(kvflash_admit_plan(in).action == A::resume);
    in.L = -1;                                              // no resume point: cold again
    CHECK(kvflash_admit_plan(in).action == A::cold);
}

TEST_CASE(KvflashAdmitPlanFixture, the_gap4_pending_token_history_one_short_of_the_prompt_resumes) {
    std::vector<int32_t> h = parked_history();
    h.pop_back();                                           // parked one token short of its KV length (the stash case)
    const std::vector<Parked> parked = {{h, "opencode", "ses_a", 1}};
    const KvFlashAdmitPlan p = plan_of(parked, "opencode", "ses_a", cat(h, {99}));
    CHECK(p.action == A::resume);
    CHECK(p.L == (int)h.size());
}

TEST_CASE(KvflashAdmitPlanFixture, a_foreign_explicit_session_is_not_resumed_by_content) {
    const std::vector<Parked> parked = {{parked_history(), "opencode", "ses_a", 1}};
    const auto prompt = cat(parked[0].history, turn(AST, 4000, 40));   // byte-identical continuation of A's transcript
    const KvFlashAdmitPlan p = plan_of(parked, "opencode", "ses_b", prompt);
    CHECK(p.action == A::cold);
    CHECK(p.slot == -1);
    CHECK(plan_of(parked, "claude", "ses_a", prompt).action == A::cold);   // the same sid of another client is another session
}

TEST_CASE(KvflashAdmitPlanFixture, identical_transcripts_under_two_sids_resume_each_its_own_instance_whatever_the_parked_ages) {
    for (const bool a_older : {true, false}) {
        const std::vector<Parked> parked = {{parked_history(), "opencode", "ses_a", a_older ? 1u : 9u}, {parked_history(), "opencode", "ses_b", a_older ? 9u : 1u}};
        const auto prompt = cat(parked_history(), turn(AST, 4000, 40));
        const KvFlashAdmitPlan pa = plan_of(parked, "opencode", "ses_a", prompt), pb = plan_of(parked, "opencode", "ses_b", prompt);
        CHECK(pa.action == A::resume); CHECK(pa.slot == 0);
        CHECK(pb.action == A::resume); CHECK(pb.slot == 1);
        CHECK(plan_of(parked, "opencode", "ses_b", turn(USR, 5000, 800)).slot == 1);
    }
}

TEST_CASE(KvflashAdmitPlanFixture, the_same_sid_slot_is_the_pick_else_the_newest_and_never_an_anonymous_request) {
    const std::vector<int32_t> h = {1, 2, 3};
    const std::vector<KvFlashOwnedSession> list = {{{0, &h, 5}, "opencode", "s"}, {{1, &h, 9}, "opencode", "s"}, {{2, &h, 99}, "opencode", "t"}, {{3, &h, 7}, "hash", "h:1"}};
    CHECK(kvflash_same_sid_slot(list, "opencode", "s", -1) == 1);
    CHECK(kvflash_same_sid_slot(list, "opencode", "s", 0) == 0);
    CHECK(kvflash_same_sid_slot(list, "opencode", "s", 2) == 1);      // a pick that is another sid's is not the sid's
    CHECK(kvflash_same_sid_slot(list, "hash", "h:1", 3) == -1);       // stateless prefix reuse owns nothing
    CHECK(kvflash_same_sid_slot(list, "opencode", "none", -1) == -1);
}

// ── the two opencode 1.19.0 compaction requests (4.3), synthetic shapes ─────────────────────

TEST_CASE(KvflashAdmitPlanFixture, the_two_opencode_compaction_requests_resolve_to_row_3_and_cold) {
    const KvFlashHeaders h = kvflash_parse_headers("X-Opencode-Session-Id: ses_compact\r\nx-session-affinity: ses_compact\r\n");
    const KvFlashSessionId who = kvflash_resolve_session(h, "");
    REQUIRE(who.client == "opencode");
    REQUIRE(who.sid == "ses_compact");
    // the session before the compaction: system+tools, a long conversation, parked with its prompt-end rung deep inside it
    std::vector<int32_t> conv = preamble();
    for (int t = 0; t < 8; ++t) conv = cat(cat(conv, turn(USR, 1000 + 1000 * t, 700)), turn(AST, 1500 + 1000 * t, 120));
    const std::vector<Parked> parked = {{conv, "opencode", "ses_compact", 1, (int)conv.size() - 120, (int)conv.size() - 900}};
    // (1) summary generation: tools {}, system [], one user message = the summarization prompt + the rendered conversation
    const auto summary_req = turn(USR, 20000, 2500);
    CHECK(kvflash_preamble_end(summary_req, IM, SYS) == 0);
    const KvFlashAdmitPlan s = plan_of(parked, who.client, who.sid, summary_req);
    CHECK(s.action == A::summary_cold);
    CHECK(s.slot == 0);
    // (2) the first post-summary request: the system block, "What did we do so far?", the summary, the verbatim tail, the new turn
    const auto post_req = cat(cat(cat(cat(preamble(), turn(USR, 30000, 8)), turn(AST, 31000, 900)), turn(USR, 32000, 400)), turn(AST, 33000, 100));
    CHECK(kvflash_preamble_end(post_req, IM, SYS) == E_ROLE);
    const KvFlashAdmitPlan q = plan_of(parked, who.client, who.sid, post_req);
    CHECK(q.action == A::cold);
    CHECK(q.slot == -1);
}

// A summary-cold request keeps its sid's parked slot; with nothing live to wait for, a refusal there can never clear, so the keep is given up.
TEST_CASE(KvflashAdmitPlanFixture, cold_slot_gives_up_the_kept_session_when_nothing_is_live) {
    std::vector<int> asked;
    auto only_slot0 = [&](int keep) { asked.push_back(keep); return keep == 0 ? -1 : 0; };   // one parked slot, and it is the kept one
    CHECK(kvflash_evict_for_cold(only_slot0, 0, false) == 0);
    CHECK(asked == (std::vector<int>{0, -1}));
    asked.clear();
    CHECK(kvflash_evict_for_cold(only_slot0, 0, true) == -1);    // a live request will retire: wait, keep the session
    CHECK(asked == (std::vector<int>{0}));
    asked.clear();
    CHECK(kvflash_evict_for_cold(only_slot0, -1, false) == 0);   // no keep: one ask
    CHECK(asked == (std::vector<int>{-1}));
}
