// Pure unit tests for the session identity and the slot owners (kvflash_session_store.h): the headers
// and the body name a session, the owner-restricted resume pick, the resume wait, the owner book and the
// admission gate. No engine, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_session_store.h"

#include <set>

using namespace luce::common;

namespace {
struct KvflashSessionStoreFixture {};

KvFlashOwnedSession owned(int slot, const std::vector<int32_t> * h, uint64_t tick, const char * client, const char * sid) {
    return {{slot, h, tick}, client, sid};
}

// What the engine's parked_list_ hands the pick: every slot's owner from the book.
std::vector<KvFlashOwnedSession> owners_of(const KvFlashSessionBook & book, const std::vector<int32_t> & hist) {
    static std::vector<int32_t> keep;   // the pick borrows the history
    keep = hist;
    std::vector<KvFlashOwnedSession> v;
    for (int i = 0; i < book.slots(); ++i) v.push_back({{i, &keep, (uint64_t)(i + 1)}, book[i].client, book[i].sid});
    return v;
}
}  // namespace

// ── identity ────────────────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, identity_resolves_opencode_then_affinity_then_claude_then_generic_then_body) {
    const KvFlashHeaders all = kvflash_parse_headers(
        "X-Session-Id: generic\r\nX-Claude-Code-Session-Id: claude\r\nx-session-affinity: aff\r\nx-opencode-session-id: ses_oc\r\n");
    KvFlashSessionId id = kvflash_resolve_session(all, "body");
    CHECK(id.sid == "ses_oc");
    CHECK(id.client == "opencode");

    id = kvflash_resolve_session(kvflash_parse_headers("X-Session-Id: generic\r\nX-Claude-Code-Session-Id: claude\r\nX-Session-Affinity: aff\r\n"), "body");
    CHECK(id.sid == "aff");                                   // opencode's other header
    CHECK(id.client == "opencode");

    id = kvflash_resolve_session(kvflash_parse_headers("X-Session-Id: generic\r\nX-Claude-Code-Session-Id: claude\r\n"), "body");
    CHECK(id.sid == "claude");
    CHECK(id.client == "claude");

    id = kvflash_resolve_session(kvflash_parse_headers("X-Session-Id: generic\r\n"), "body");
    CHECK(id.sid == "generic");
    CHECK(id.client == "header");

    id = kvflash_resolve_session(kvflash_parse_headers("Content-Type: application/json\r\n"), "body");
    CHECK(id.sid == "body");
    CHECK(id.client == "body");

    // parent ids are a relation, not an identity
    id = kvflash_resolve_session(kvflash_parse_headers("x-parent-session-id: p\r\nx-opencode-parent-session-id: q\r\n"), "");
    CHECK(id.empty());
    CHECK(id.client.empty());

    // an empty header value is no identity: the next source answers
    id = kvflash_resolve_session(kvflash_parse_headers("x-opencode-session-id:   \r\nX-Claude-Code-Session-Id: claude\r\n"), "");
    CHECK(id.sid == "claude");
}

TEST_CASE(KvflashSessionStoreFixture, headers_parse_lower_cased_trimmed_first_value_wins_and_both_line_endings) {
    const KvFlashHeaders h = kvflash_parse_headers(
        "Host: example\r\nX-OpenCode-Session-ID:  ses_1 \r\nCONTENT-LENGTH: 12\r\nx-opencode-session-id: ses_2\r\nbroken line\r\nEmpty:\r\n");
    CHECK(h.at("host") == "example");
    CHECK(h.at("x-opencode-session-id") == "ses_1");   // a duplicate keeps the first
    CHECK(h.at("content-length") == "12");
    CHECK(h.at("empty") == "");
    CHECK(h.count("broken line") == 0);
    CHECK(h.count("X-OpenCode-Session-ID") == 0);      // names are stored lower-cased only
    CHECK(h.size() == 4);

    const KvFlashHeaders lf = kvflash_parse_headers("A: 1\nB: 2\n");
    CHECK(lf.at("a") == "1");
    CHECK(lf.at("b") == "2");
    CHECK(kvflash_parse_headers("").empty());
}

TEST_CASE(KvflashSessionStoreFixture, content_matching_never_overrides_an_explicit_sid_on_a_parked_session) {
    const std::vector<int32_t> ha = {1, 2, 3, 4}, hb = {1, 2, 3, 4, 5}, hm = {1, 2, 3, 4, 5, 6}, hn = {1, 2, 9};
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7};
    const std::vector<KvFlashOwnedSession> parked = {
        owned(0, &ha, 1, "opencode", "ses_a"), owned(1, &hb, 2, "opencode", "ses_b"), owned(2, &hm, 3, "migrated", "migrated:1")};
    // ses_a takes only ses_a, from the first shared token, even though ses_b and the migrated one share more
    KvFlashSessionPick p = kvflash_pick_parked_owner(parked, "opencode", "ses_a", prompt, 0);
    CHECK(p.slot == 0);
    CHECK(p.lcp == 4);
    CHECK(kvflash_pick_parked_owner(parked, "claude", "ses_a", prompt, 0).slot == 2);   // same sid, other client: not the session; the ownerless one is
    // a sid with no session of its own takes the ownerless one past the preamble, never another's
    p = kvflash_pick_parked_owner(parked, "opencode", "ses_c", prompt, 2);
    CHECK(p.slot == 2);
    CHECK(kvflash_pick_parked_owner(parked, "opencode", "ses_c", {1, 2, 8}, 2).slot < 0);    // preamble only: lcp 2 is not > e_role 2
    CHECK(kvflash_pick_parked_owner({owned(0, &ha, 1, "opencode", "ses_a")}, "opencode", "ses_c", prompt, 0).slot < 0);
    // an anonymous request takes ownerless sessions only
    CHECK(kvflash_pick_parked_owner(parked, "anon", "anon:9", prompt, 0).slot == 2);
    CHECK(kvflash_pick_parked_owner({owned(0, &ha, 1, "opencode", "ses_a")}, "anon", "anon:9", prompt, 0).slot < 0);
    // two anonymous sessions: content decides, as today
    const std::vector<KvFlashOwnedSession> anon = {owned(0, &ha, 1, "anon", "anon:1"), owned(1, &hn, 2, "anon", "anon:2")};
    CHECK(kvflash_pick_parked_owner(anon, "anon", "anon:3", prompt, 0).slot == 0);
}

// The owner of a slot is kept in memory.
TEST_CASE(KvflashSessionStoreFixture, a_slot_keeps_its_owner_and_a_matching_b_or_anon_cannot_take_it) {
    KvFlashSessionBook book;
    book.init(2);
    book.begin_session(0, "opencode", "A");
    CHECK(book[0].sid == "A");
    CHECK(book[0].client == "opencode");
    const std::vector<int32_t> hist = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto owners = owners_of(book, hist);
    // an explicit B whose prompt matches A's history, and an anonymous request, are not A
    CHECK(kvflash_pick_parked_owner(owners, "opencode", "B", hist, 0).slot != 0);
    CHECK(kvflash_pick_parked_owner(owners, "anon", "anon:9", hist, 0).slot != 0);
    CHECK(kvflash_pick_parked_owner(owners, "opencode", "A", hist, 0).slot == 0);   // A itself resumes
    // the slot leaving its session (a drop) resets the owner too
    book.reset(0);
    CHECK(book[0].sid.empty());
    CHECK(book[0].client.empty());
}

// One live owner per explicit identity: a second request for it waits.
TEST_CASE(KvflashSessionStoreFixture, a_live_identity_is_busy_for_a_second_request) {
    KvFlashSessionBook book;
    book.init(2);
    book.begin_session(0, "opencode", "A");                                      // A decodes in slot 0
    std::vector<bool> active = {true, false};
    CHECK(book.identity_busy("opencode", "A", active));                          // a second request for A is deferred, never a second owner
    CHECK(!book.identity_busy("opencode", "B", active));
    CHECK(!book.identity_busy("claude", "A", active));                           // the pair is the identity
    CHECK(!book.identity_busy("anon", "anon:7", active));
    active[0] = false;                                                           // A finished: parked, resumable
    CHECK(!book.identity_busy("opencode", "A", active));
}

TEST_CASE(KvflashSessionStoreFixture, the_admission_gate_refuses_an_oversized_identity_and_a_running_one) {
    KvFlashSessionBook book;
    book.init(2);
    book.begin_session(0, "claude", "S");
    const std::vector<bool> active = {true, false};
    using V = KvFlashAdmitVerdict;
    CHECK(kvflash_admit_gate(book, {std::string(kKvFlashSidMax + 1, 'x'), "body"}, active) == V::invalid_identity);
    CHECK(kvflash_admit_gate(book, {"S", "claude"}, active) == V::identity_busy);   // its slot runs a request
    CHECK(kvflash_admit_gate(book, {"S", "claude"}, {false, false}) == V::proceed);   // parked: it resumes
    CHECK(kvflash_admit_gate(book, {"other", "claude"}, active) == V::proceed);
    CHECK(kvflash_admit_gate(book, KvFlashSessionId{}, active) == V::proceed);
}

// A resume hands an ownerless session to the explicit client that took it, and to no one else.
TEST_CASE(KvflashSessionStoreFixture, only_an_explicit_request_takes_over_an_ownerless_session) {
    KvFlashSessionBook book;
    book.init(1);
    book.begin_session(0, "anon", "anon:3");
    CHECK(!book.promote_owner(0, "anon", "anon:9"));                              // an anonymous request owns nothing
    CHECK((book[0].client == "anon" && book[0].sid == "anon:3"));
    CHECK(book.promote_owner(0, "opencode", "ses_1"));
    CHECK((book[0].client == "opencode" && book[0].sid == "ses_1"));
    CHECK(!book.promote_owner(0, "claude", "other"));                             // a stated identity is never taken over by content
    CHECK((book[0].client == "opencode" && book[0].sid == "ses_1"));
}

TEST_CASE(KvflashSessionStoreFixture, an_oversized_identity_is_invalid_and_no_identity_is_valid) {
    KvFlashSessionId ok{std::string(kKvFlashSidMax, 'x'), "body"};
    KvFlashSessionId big{std::string(kKvFlashSidMax + 1, 'x'), "body"};
    KvFlashSessionId huge{std::string(2 << 20, 'x'), "header"};
    CHECK(kvflash_identity_valid(ok));
    CHECK(!kvflash_identity_valid(big));
    CHECK(!kvflash_identity_valid(huge));
    CHECK(kvflash_identity_valid(KvFlashSessionId{}));                           // no identity: anonymous, valid
}

// A parked session whose resume wait holds for the request is picked again by that request only.
TEST_CASE(KvflashSessionStoreFixture, a_waiting_resume_is_picked_again_by_the_request_that_waits) {
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    const std::vector<int32_t> history = {1, 2, 3, 4};               // A, parked in slot 0: the prompt extends it completely
    const KvFlashSessionId who{"anon:9", "anon"};
    const KvFlashResumeWait wait = kvflash_resume_wait_make(/*L=*/4, /*lcp=*/4, prompt, who.client, who.sid);
    const KvFlashSessionPick pick = kvflash_pick_with_wait({}, 0, (int)history.size(), wait, prompt, who.client, who.sid);
    CHECK(pick.slot == 0);
    CHECK(pick.lcp == 4);
    // a request that is not the waiting one, or whose prompt does not start with the waiting prompt's prefix, is not the wait's
    CHECK(!kvflash_pick_with_wait({}, 0, (int)history.size(), wait, prompt, "anon", "anon:1").exact());
    CHECK(!kvflash_pick_with_wait({}, 0, (int)history.size(), wait, {1, 2, 9, 4, 5}, who.client, who.sid).exact());
}

// A second request under a running identity must not become the scheduler's deferred head of line: it admits as its own anonymous session.
TEST_CASE(KvflashSessionStoreFixture, a_busy_identity_admits_the_second_request_as_anonymous) {
    KvFlashSessionBook book;
    book.init(2);
    book.begin_session(0, "claude", "S");
    const std::vector<bool> active{true, false};
    const KvFlashSessionId same = kvflash_admit_identity(book, {"S", "claude"}, active, 42);
    CHECK(same.client == "anon");
    CHECK(same.sid == "anon:42");
    CHECK(kvflash_admit_identity(book, {"other", "claude"}, active, 43).sid == "other");   // not busy: unchanged
    CHECK(kvflash_admit_identity(book, {"S", "claude"}, {false, false}, 44).sid == "S");   // parked: it resumes
    CHECK(kvflash_admit_identity(book, {std::string(kKvFlashSidMax + 1, 'x'), "body"}, active, 45).sid.size() == kKvFlashSidMax + 1);   // the gate still refuses it
}

// A client-chosen id the engine cannot hold or log safely is no identity: the request is served without a session, never refused.
TEST_CASE(KvflashSessionStoreFixture, an_oversized_or_control_character_id_is_no_identity) {
    CHECK(kvflash_bound_identity({"S", "claude"}).sid == "S");
    CHECK(kvflash_bound_identity({std::string(kKvFlashSidMax, 'x'), "header"}).sid.size() == kKvFlashSidMax);
    CHECK(kvflash_bound_identity({std::string(kKvFlashSidMax + 1, 'x'), "header"}).empty());
    CHECK(kvflash_bound_identity({std::string(kKvFlashSidMax + 1, 'x'), "header"}).client.empty());
    CHECK(kvflash_bound_identity({"x\n[server] chat DONE ok=true", "body"}).empty());   // forges a log line
    CHECK(kvflash_bound_identity({"a\x1b[31mb", "body"}).empty());
}
