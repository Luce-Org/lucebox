// Pure unit tests for the KVFlash sessions (kvflash_session_store.h): the headers and the body name a
// session, the owner-restricted resume pick, the resume wait, the owner book and the admission gate; and
// the session store on disk (<kv-cache-dir>/sessions/<key>/): meta.json, index.json, the scan that
// rebuilds the index from the directories, evict-not-delete, adoption, and the DiskStore tenant that
// counts the store in the one cache budget. Real bundle directories written through
// kvflash_bundle_commit, no engine, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_session_store.h"
#include "../../src/common/kvflash_sessions_plan.h"
#include "../../src/server/kv/disk_store.h"
#include "../../src/server/kv/session_tenant.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

using namespace luce::common;
namespace fs = std::filesystem;

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

// ── disk fixtures ───────────────────────────────────────────────────────────────────────────

namespace {
constexpr size_t CB = 8;
const size_t RB = kvflash_record_bytes(CB);
constexpr uint64_t ID = 0x5e55;
constexpr size_t HEAD_MAX = 1 << 20;

std::vector<uint8_t> hdr44() { std::vector<uint8_t> h(kKvFlashPagerHdr); for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t)(0xa0 + i); return h; }

// A whole park of `n` chunks whose history is `hist`, written to `dir` (created).
bool park_session(const std::string & dir, const std::vector<int32_t> & hist, int n, uint8_t fill, uint64_t identity = ID, int crash_step = 0) {
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = 1; c.log_gen = 1; c.new_log = true; c.new_dir = true; c.record_bytes = RB;
    KvFlashSlotHead h;
    h.identity = identity; h.gen = 1; h.log_gen = 1; h.end_gen = 2; h.history = hist; h.pager_hdr = hdr44(); h.ledger.assign((size_t)n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h.ledger[(size_t)i * 8] = 1; h.ledger[(size_t)i * 8 + 1] = 1;
        c.records.push_back({(uint32_t)i, std::vector<uint8_t>(CB, (uint8_t)(fill + i))});
        h.map.push_back(kKvFlashLogHeader + (uint64_t)i * RB);
    }
    h.log_bytes = kKvFlashLogHeader + (uint64_t)n * RB;
    c.files = {{kvflash_bundle_slab_name(2), kvflash_slab_pack({{fill, 1}}, {{fill, 2}})}};
    c.head = kvflash_head_pack(h);
    return kvflash_bundle_commit(c, crash_step);
}

KvFlashBundleLimits lim() { return KvFlashBundleLimits{1 << 20, 1 << 20, 1 << 20}; }

bool loads(const std::string & dir, std::vector<int32_t> * hist = nullptr) {
    KvFlashBundleLoaded l;
    std::string why;
    if (!kvflash_bundle_load(dir, ID, CB, lim(), l, why)) return false;
    if (hist) *hist = l.head.history;
    return true;
}

// A directory of its own under TMPDIR, empty.
std::string fresh_root(const char * name) {
    const std::string root = (fs::temp_directory_path() / name).string();
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

std::set<std::string> names(const std::string & dir) {
    std::set<std::string> out;
    for (const auto & e : fs::directory_iterator(dir)) out.insert(e.path().filename().string());
    return out;
}

std::string slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

KvFlashSessionEntry entry(const std::string & sid, int64_t last_used, const std::string & state = "parked-gpu") {
    KvFlashSessionEntry e;
    e.key = kvflash_sid_key("header", sid); e.sid = sid; e.client = "header"; e.state = state;
    e.created = 100; e.last_used = last_used; e.tokens = 5; e.identity = ID;
    return e;
}

// Makes one I/O step fail (or just records it) for the scope of a test.
struct IoFault {
    explicit IoFault(KvFlashIoFault f) { kvflash_io_fault() = std::move(f); }
    ~IoFault() { kvflash_io_fault() = nullptr; }
};
bool ends_with(const std::string & s, const std::string & suffix) { return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0; }

// A session the way the engine leaves it: its directory with a head and its meta.json, listed in `store`.
KvFlashSessionEntry commit_session(KvFlashSessionStore & store, const std::string & client, const std::string & sid,
                                   const std::vector<int32_t> & hist, uint8_t fill, int64_t last_used) {
    KvFlashSessionEntry e;
    e.key = kvflash_sid_key(client, sid); e.sid = sid; e.client = client; e.state = "disk";
    e.created = 100; e.last_used = last_used; e.tokens = hist.size(); e.identity = ID;
    fs::create_directories(store.dir(e.key));
    if (!park_session(store.dir(e.key), hist, 1, fill) || (store.upsert(e), !store.write_meta(e.key))) throw std::runtime_error("commit_session");
    return e;
}

// The book of a restart: the sessions the directories hold, scanned.
void rescan(KvFlashSessionStore & store) {
    KvFlashScanReport rep;
    store.load(kvflash_session_scan(store.root(), ID, RB, HEAD_MAX, rep));
}

// The session in `key` loaded into slot 0 of `book`, the way the engine's restore does.
std::vector<int32_t> adopt_into_slot(KvFlashSessionBook & book, const std::string & key, int slot = 0) {
    KvFlashBundleLoaded l;
    std::string why;
    if (!kvflash_bundle_load(book.store().dir(key), ID, CB, lim(), l, why)) throw std::runtime_error("adopt_into_slot: " + why);
    return book.adopt(slot, key, std::move(l.head));
}
}  // namespace

// ── session keys ────────────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, the_directory_key_is_fnv_hex16_of_the_pair_and_a_taken_key_gets_a_deterministic_variant) {
    CHECK(kvflash_sid_key("header", "ses_1").size() == 16);
    CHECK(kvflash_sid_key("header", "ses_1") != kvflash_sid_key("header", "ses_2"));
    CHECK(kvflash_sid_key("header", "ses_1") != kvflash_sid_key("opencode", "ses_1"));   // the same sid under two clients is two sessions
    std::set<std::string> taken;
    const std::string k0 = kvflash_session_dir_key("header", "ses_1", taken);
    CHECK(k0 == kvflash_sid_key("header", "ses_1"));
    taken.insert(k0);
    const std::string k1 = kvflash_session_dir_key("header", "ses_1", taken);
    CHECK(k1 != k0);
    CHECK(k1.size() == 16);
    taken.insert(k1);
    const std::string k2 = kvflash_session_dir_key("header", "ses_1", taken);
    CHECK(k2 != k0);
    CHECK(k2 != k1);
    CHECK(kvflash_session_dir_key("header", "ses_1", std::set<std::string>{k0}) == k1);   // same input, same answer
}

// ── meta and index ──────────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, meta_round_trips_and_escapes_a_hostile_sid) {
    KvFlashSessionEntry e = entry("ses \"q\" \\ \n\t \x01 é", 12345);
    e.client = "opencode"; e.state = "disk"; e.archive_of = "abc"; e.archive = "def";
    e.bytes_live = 1234567890123ULL; e.bytes_dead = 7; e.tokens = 99;
    const std::string json = kvflash_meta_pack(e);
    CHECK(json.find('\n') == std::string::npos);   // one line, whatever the sid holds
    KvFlashSessionEntry b;
    REQUIRE(kvflash_meta_unpack(json, b));
    CHECK(b.sid == e.sid);
    CHECK(b.client == "opencode");
    CHECK(b.state == "disk");
    CHECK(b.archive_of == "abc");
    CHECK(b.archive == "def");
    CHECK(b.created == 100);
    CHECK(b.last_used == 12345);
    CHECK(b.tokens == 99);
    CHECK(b.bytes_live == 1234567890123ULL);
    CHECK(b.bytes_dead == 7);
    CHECK(b.identity == ID);
    CHECK(!kvflash_meta_unpack("", b));
    CHECK(!kvflash_meta_unpack("{\"sid\": \"x\"", b));    // torn
    CHECK(!kvflash_meta_unpack("{\"client\": \"x\"}", b));   // no sid
}

TEST_CASE(KvflashSessionStoreFixture, writes_are_atomic_replaces_and_leave_no_tmp) {
    const std::string root = fresh_root("kvflash_s10b_atomic");
    KvFlashSessionStore store(root);
    REQUIRE(fs::create_directories(store.dir("k1")));
    KvFlashSessionEntry e = entry("s1", 10);
    e.key = "k1";
    store.upsert(e);
    REQUIRE(store.write_meta("k1"));
    REQUIRE(store.write_index());
    const std::string m1 = slurp(root + "/sessions/k1/meta.json"), i1 = slurp(root + "/sessions/index.json");
    CHECK(!m1.empty());
    CHECK(i1.find("\"s1\"") != std::string::npos);
    e.last_used = 20; e.tokens = 77;
    store.upsert(e);
    REQUIRE(store.write_meta("k1"));
    REQUIRE(store.write_index());
    const std::string m2 = slurp(root + "/sessions/k1/meta.json");
    CHECK(m2 != m1);
    CHECK(m2.find("77") != std::string::npos);
    for (const auto & f : names(root + "/sessions/k1")) CHECK(f == "meta.json");
    for (const auto & f : names(root + "/sessions")) CHECK((f == "k1" || f == "index.json"));
    for (const auto & f : names(root)) CHECK(f == "sessions");   // nothing of the store outside its own directory
    // a write killed before its rename leaves only a tmp: the old file is whole, and a scan ignores the tmp
    std::ofstream(root + "/sessions/k1/meta.json.tmp.4242") << "{\"sid\": \"torn";
    CHECK(slurp(root + "/sessions/k1/meta.json") == m2);
    fs::remove_all(root);
}

// ── scan: the index is a cache of the directories ───────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, scan_rebuilds_the_index_from_head_and_meta_with_a_missing_meta_or_a_missing_head) {
    const std::string root = fresh_root("kvflash_s10b_scan");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    // a: head + meta. b: head, no meta. c: meta, no head (a first park killed before its commit). d: another server's head.
    for (const char * k : {"aaaa", "bbbb", "cccc", "dddd"}) fs::create_directories(store.dir(k));
    REQUIRE(park_session(store.dir("aaaa"), {1, 2, 3, 4, 5, 6}, 2, 0x10));
    REQUIRE(park_session(store.dir("bbbb"), {7, 8, 9}, 1, 0x20));
    REQUIRE(park_session(store.dir("dddd"), {1}, 1, 0x30, /*identity=*/ID + 1));
    KvFlashSessionEntry a = entry("sid-a", 500, "parked-gpu");
    a.key = "aaaa"; a.tokens = 4;   // stale: the head holds 6 tokens
    KvFlashSessionEntry c = entry("sid-c", 600);
    c.key = "cccc";
    store.upsert(a); store.upsert(c);
    REQUIRE(store.write_meta("aaaa"));
    REQUIRE(store.write_meta("cccc"));
    std::ofstream(store.dir("cccc") + "/chunks.1.log") << "uncommitted";

    KvFlashScanReport rep;
    const std::vector<KvFlashSessionEntry> es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(es.size() == 3);                       // a, b and d (listed as foreign: another server's, never erased)
    const KvFlashSessionEntry * ea = nullptr, * eb = nullptr, * ed = nullptr;
    for (const auto & e : es) { if (e.key == "aaaa") ea = &e; if (e.key == "bbbb") eb = &e; if (e.key == "dddd") ed = &e; }
    REQUIRE(ed != nullptr);
    CHECK(ed->state == "foreign");
    CHECK(ed->identity == ID + 1);
    REQUIRE(ea != nullptr);
    REQUIRE(eb != nullptr);
    CHECK(ea->sid == "sid-a");
    CHECK(ea->tokens == 6);                       // the head is the truth, not the meta
    CHECK(ea->last_used == 500);
    CHECK(ea->state == "disk");                    // nothing is in a slot at start
    CHECK(ea->history == (std::vector<int32_t>{1, 2, 3, 4, 5, 6}));
    CHECK(ea->bytes_live > 0);
    CHECK(eb->state == "foreign");                 // no meta: whose it is cannot be proven, so it is listed and never resumed
    CHECK(eb->client == "foreign");
    CHECK(eb->history.empty());
    CHECK(fs::exists(store.dir("bbbb") + "/head"));
    CHECK(!fs::exists(store.dir("cccc")));         // meta without a head: nothing was committed
    CHECK(fs::exists(store.dir("dddd") + "/head"));   // another server's bundle is skipped, never erased
    CHECK(std::set<std::string>(rep.skipped.begin(), rep.skipped.end()) == (std::set<std::string>{"bbbb", "dddd"}));
    REQUIRE(rep.removed.size() == 1);
    CHECK(rep.removed[0] == "cccc");
    // a name that is not a session key (a dot in it) is not a session, and index.json is no directory
    fs::create_directories(root + "/sessions/aaaa.skipped");
    REQUIRE(store.write_index());
    KvFlashScanReport rep2;
    CHECK(kvflash_session_scan(root, ID, RB, HEAD_MAX, rep2).size() == 3);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_crash_between_the_bundle_data_and_the_index_loses_nothing_that_was_committed) {
    for (int step = 1; step <= 5; ++step) {
        const std::string root = fresh_root("kvflash_s10b_crash");
        fs::create_directories(root + "/sessions");
        KvFlashSessionStore store(root);
        // the previous park of session "old" is whole; a first park of "new" is killed after `step`
        REQUIRE(fs::create_directories(store.dir("old")));
        REQUIRE(park_session(store.dir("old"), {1, 2, 3}, 1, 0x10));
        KvFlashSessionEntry o = entry("sid-old", 10);
        o.key = "old";
        store.upsert(o);
        REQUIRE(store.write_meta("old"));
        REQUIRE(store.write_index());
        KvFlashSessionEntry n = entry("sid-new", 20);
        n.key = "new";
        REQUIRE(fs::create_directories(store.dir("new")));
        store.upsert(n);
        REQUIRE(store.write_meta("new"));                       // staged like the engine's park: the meta goes before the data
        CHECK(!park_session(store.dir("new"), {4, 5}, 1, 0x20, ID, step));   // killed after this step; index.json is stale

        KvFlashScanReport rep;
        const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
        std::set<std::string> keys;
        for (const auto & e : es) keys.insert(e.key);
        CHECK(keys.count("old") == 1);
        CHECK(std::none_of(es.begin(), es.end(), [](const auto & e) { return e.state == "foreign"; }));
        CHECK(loads(store.dir("old")));
        if (step < 4) {   // before the head rename: nothing of "new" was committed
            CHECK(keys.size() == 1);
            CHECK(!fs::exists(store.dir("new")));
        } else {          // head renamed: "new" is whole and listed, its meta says who it is
            CHECK(keys.size() == 2);
            for (const auto & e : es) if (e.key == "new") { CHECK(e.sid == "sid-new"); CHECK(e.tokens == 2); }
            CHECK(loads(store.dir("new")));
        }
        fs::remove_all(root);
    }
}

TEST_CASE(KvflashSessionStoreFixture, the_restart_cleanup_keeps_meta_json_and_cuts_what_the_head_does_not_name) {
    const std::string root = fresh_root("kvflash_s10b_clean");
    fs::create_directories(root + "/sessions");
    std::ofstream(root + "/sessions/index.json") << "[]";
    REQUIRE(park_session(root + "/sessions/k1", {1, 2}, 1, 0x10));
    std::ofstream(root + "/sessions/k1/meta.json") << "{}";
    std::ofstream(root + "/sessions/k1/chunks.9.log") << "stale";
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(root + "/sessions/k1", ID, CB, lim(), l, why));
    kvflash_bundle_clean(root + "/sessions/k1", &l.head);
    CHECK(fs::exists(root + "/sessions/k1/meta.json"));
    CHECK(!fs::exists(root + "/sessions/k1/chunks.9.log"));
    CHECK(fs::exists(root + "/sessions/k1/head"));
    CHECK(fs::exists(root + "/sessions/index.json"));       // the cleanup of a session directory never reaches the store's files above it
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, r15_a_stat_error_on_the_head_is_unknown_and_preserved_by_the_scan) {
    const std::string root = fresh_root("kvflash_s10b_stat");
    KvFlashSessionStore seed(root);
    const auto a = commit_session(seed, "header", "A", {1, 2, 3}, 0x10, 100);
    KvFlashScanReport rep;
    std::vector<KvFlashSessionEntry> es;
    {
        IoFault f([](std::string_view o, const std::string & path) { return o == "stat" && ends_with(path, "/head"); });
        es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    }
    CHECK(fs::exists(seed.dir(a.key) + "/head"));                            // not removed as an empty directory
    CHECK(rep.removed.empty());
    REQUIRE(es.size() == 1);
    CHECK(es[0].state == "foreign");
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, r15_the_scan_marks_a_corrupt_log_a_missing_or_malformed_slab_foreign) {
    const std::string root = fresh_root("kvflash_s10b_corrupt");
    KvFlashSessionStore seed(root);
    const auto good = commit_session(seed, "header", "good", {1, 2, 3}, 0x10, 500);
    const auto bad_log = commit_session(seed, "header", "bad_log", {4, 5, 6}, 0x20, 100);
    const auto no_slab = commit_session(seed, "header", "no_slab", {7, 8, 9}, 0x30, 100);
    const auto bad_slab = commit_session(seed, "header", "bad_slab", {2, 4, 6}, 0x40, 100);
    const auto short_log = commit_session(seed, "header", "short_log", {3, 6, 9}, 0x50, 100);
    {   // a flipped byte inside a record's payload
        std::fstream f(seed.dir(bad_log.key) + "/chunks.1.log", std::ios::in | std::ios::out | std::ios::binary);
        f.seekp((std::streamoff)(kKvFlashLogHeader + kKvFlashRecordTag + 1));
        f.put('\x7f');
    }
    fs::remove(seed.dir(no_slab.key) + "/" + kvflash_bundle_slab_name(2));
    { std::ofstream f(seed.dir(bad_slab.key) + "/" + kvflash_bundle_slab_name(2), std::ios::binary | std::ios::trunc); f << "xx"; }
    fs::resize_file(seed.dir(short_log.key) + "/chunks.1.log", kKvFlashLogHeader + 4);
    KvFlashScanReport rep;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(es.size() == 5);
    for (const auto & e : es) {
        if (e.key == good.key) CHECK(e.state == "disk");
        else { CHECK(e.state == "foreign"); CHECK(e.history.empty()); CHECK(fs::exists(seed.dir(e.key) + "/head")); }
    }
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, r01_a_named_record_whose_meta_cannot_be_read_stays_foreign_and_no_identity_can_take_it) {
    const std::string root = fresh_root("kvflash_s10b_torn_meta");
    KvFlashSessionStore seed(root);
    const auto a = commit_session(seed, "opencode", "A", {1, 2, 3, 4, 5, 6}, 0x10, 100);
    std::ofstream(seed.dir(a.key) + "/meta.json", std::ios::trunc) << "{\"sid\": \"A\"";          // torn
    KvFlashSessionStore store(root);
    rescan(store);
    REQUIRE(store.entries().size() == 1);
    CHECK(store.entries()[0].state == "foreign");
    CHECK(store.pick_disk("opencode", "B", {1, 2, 3, 4, 5, 6, 7}, 0) == nullptr);                 // explicit B cannot take it by content
    CHECK(store.pick_disk("anon", "anon:1", {1, 2, 3, 4, 5, 6, 7}, 0) == nullptr);                // nor an anonymous request
    CHECK(fs::exists(store.dir(a.key) + "/head"));
    fs::remove_all(root);
}

// ── evict-not-delete, find by sid, adoption ─────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, a_full_house_evicts_the_oldest_to_disk_keeps_its_bundle_and_the_index_lists_it) {
    const std::string root = fresh_root("kvflash_s10b_evict");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    const std::vector<int32_t> ha = {1, 2, 3}, hb = {4, 5, 6, 7};
    for (const auto & [sid, last, hist, fill] : std::vector<std::tuple<std::string, int64_t, std::vector<int32_t>, uint8_t>>{
             {"sid-a", 10, ha, 0x10}, {"sid-b", 20, hb, 0x20}}) {
        KvFlashSessionEntry e = entry(sid, last);
        REQUIRE(fs::create_directories(store.dir(e.key)));
        REQUIRE(park_session(store.dir(e.key), hist, 1, fill));
        store.upsert(e);
        REQUIRE(store.write_meta(e.key));
    }
    // the two slots are full and a third session arrives: the oldest parked one goes to disk
    const std::vector<KvFlashParkedSession> parked = {{0, &ha, 1}, {1, &hb, 2}};   // slot, history, park tick: the engine's own pick
    CHECK(kvflash_oldest_session(parked) == 0);
    const std::string key_a = kvflash_sid_key("header", "sid-a");
    REQUIRE(store.evict(key_a, ha));
    CHECK(store.find_key(key_a)->state == "disk");
    CHECK(store.find_key(key_a)->history == ha);
    CHECK(store.find_key(kvflash_sid_key("header", "sid-b"))->state == "parked-gpu");
    CHECK(fs::exists(store.dir(key_a) + "/head"));                   // the bundle is kept
    CHECK(loads(store.dir(key_a)));
    const std::string idx = slurp(root + "/sessions/index.json");
    CHECK(idx.find("\"sid-a\"") != std::string::npos);                // and the index still lists it
    CHECK(idx.find("\"disk\"") != std::string::npos);
    CHECK(idx.find("\"sid-b\"") != std::string::npos);
    CHECK(slurp(store.dir(key_a) + "/meta.json").find("\"disk\"") != std::string::npos);
    KvFlashScanReport rep;                                           // a restart: a scan lists both, both on disk
    CHECK(kvflash_session_scan(root, ID, RB, HEAD_MAX, rep).size() == 2);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_session_evicted_to_disk_is_found_again_by_sid_and_restored) {
    const std::string root = fresh_root("kvflash_s10b_restore");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    const std::vector<int32_t> ha = {1, 2, 3, 4};
    KvFlashSessionEntry a = entry("sid-a", 10);
    REQUIRE(fs::create_directories(store.dir(a.key)));
    REQUIRE(park_session(store.dir(a.key), ha, 1, 0x10));
    store.upsert(a);
    REQUIRE(store.write_meta(a.key));
    REQUIRE(store.evict(a.key, ha));

    const std::vector<int32_t> prompt = {1, 2, 3, 4, 8, 9};
    const KvFlashSessionEntry * hit = store.pick_disk("header", "sid-a", prompt, 0);
    REQUIRE(hit != nullptr);
    CHECK(hit->key == a.key);
    CHECK(store.pick_disk("header", "sid-other", prompt, 0) == nullptr);   // another sid does not take it
    CHECK(store.pick_disk("opencode", "sid-a", prompt, 0) == nullptr);     // nor does the same sid of another client
    CHECK(store.pick_disk("anon", "anon:1", prompt, 0) == nullptr);        // and no sid never matches a real one
    std::vector<int32_t> hist;
    CHECK(loads(store.dir(a.key), &hist));
    CHECK(hist == ha);
    REQUIRE(store.adopt(a.key));                                      // restored into a slot
    CHECK(store.find_key(a.key)->state == "parked-gpu");
    CHECK(store.find_key(a.key)->history.empty());                    // a slot session's history lives in the engine
    CHECK(store.pick_disk("header", "sid-a", prompt, 0) == nullptr);  // no longer on disk

    // the same sid, a second instance (the first not resumable): two dirs, the newest answers
    KvFlashSessionEntry b = entry("sid-a", 50);
    b.key = store.new_key("header", "sid-a");
    CHECK(b.key != a.key);
    REQUIRE(fs::create_directories(store.dir(b.key)));
    REQUIRE(park_session(store.dir(b.key), {1, 2, 3}, 1, 0x40));
    store.upsert(b);
    REQUIRE(store.evict(b.key, {1, 2, 3}));
    REQUIRE(store.evict(a.key, ha));
    REQUIRE(store.pick_disk("header", "sid-a", prompt, 0) != nullptr);
    CHECK(store.pick_disk("header", "sid-a", prompt, 0)->key == b.key);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, an_ownerless_disk_session_is_found_by_the_longest_prefix_beyond_the_preamble) {
    const std::string root = fresh_root("kvflash_s10b_ownerless");
    KvFlashSessionStore store(root);
    KvFlashSessionEntry m1 = entry("anon:1", 10, "disk"), m2 = entry("anon:2", 20, "disk"), r = entry("real", 30, "disk");
    m1.client = m2.client = "anon";
    m1.history = {1, 2, 3, 4}; m2.history = {1, 2, 9}; r.history = {1, 2, 3, 4, 5};
    store.upsert(m1); store.upsert(m2); store.upsert(r);
    const KvFlashSessionEntry * hit = store.pick_disk("opencode", "ses_new", {1, 2, 3, 4, 8}, 0);
    REQUIRE(hit != nullptr);
    CHECK(hit->sid == "anon:1");                                      // the longest prefix, and only among ownerless ones
    CHECK(store.pick_disk("opencode", "ses_new", {7, 7}, 0) == nullptr);
    CHECK(store.pick_disk("header", "real", {7, 7}, 0) == nullptr);   // a sid match still needs a shared prefix: no load for nothing
    CHECK(store.pick_disk("header", "real", {1, 2, 3, 4, 5, 6}, 0)->sid == "real");
    CHECK(store.pick_disk("opencode", "ses_new", {1, 2, 8}, 2) == nullptr);   // a shared preamble alone (lcp <= e_role) is not a conversation
    CHECK(store.pick_disk("anon", "anon:4", {1, 2, 3, 8}, 2)->sid == "anon:1");
    CHECK(store.pick_disk("anon", "anon:4", {1, 2, 3, 4, 5, 6}, 0)->sid == "anon:1");   // an anonymous request never takes a session with a sid
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, content_matching_never_overrides_an_explicit_sid_on_a_parked_or_a_disk_session) {
    const std::vector<int32_t> ha = {1, 2, 3, 4}, hb = {1, 2, 3, 4, 5}, hm = {1, 2, 3, 4, 5, 6}, hn = {1, 2, 9};
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7};
    const std::vector<KvFlashOwnedSession> parked = {
        owned(0, &ha, 1, "opencode", "ses_a"), owned(1, &hb, 2, "opencode", "ses_b"), owned(2, &hm, 3, "anon", "anon:1")};
    // ses_a takes only ses_a, from the first shared token, even though ses_b and the ownerless one share more
    KvFlashSessionPick p = kvflash_pick_parked_owner(parked, "opencode", "ses_a", prompt, 0);
    CHECK(p.slot == 0);
    CHECK(p.lcp == 4);
    CHECK(kvflash_pick_parked_owner(parked, "claude", "ses_a", prompt, 0).slot == 2);   // same sid, other client: not the session; the ownerless one is
    CHECK(kvflash_pick_parked_owner(parked, "anon", "anon:9", prompt, 0).slot == 2);
    const std::vector<KvFlashOwnedSession> anon = {owned(0, &ha, 1, "anon", "anon:1"), owned(1, &hn, 2, "anon", "anon:2")};
    CHECK(kvflash_pick_parked_owner(anon, "anon", "anon:3", prompt, 0).slot == 0);
    // the disk tier follows the same rules
    KvFlashSessionStore store(fresh_root("kvflash_s10b_elig"));
    KvFlashSessionEntry b = entry("ses_b", 10, "disk");
    b.client = "opencode"; b.history = {1, 2, 3};
    store.upsert(b);
    CHECK(store.pick_disk("opencode", "ses_a", {1, 2, 3, 4}, 0) == nullptr);
    CHECK(store.pick_disk("anon", "anon:1", {1, 2, 3, 4}, 0) == nullptr);
    CHECK(store.pick_disk("opencode", "ses_b", {1, 2, 3, 4}, 0) != nullptr);
    fs::remove_all(store.root());
}

TEST_CASE(KvflashSessionStoreFixture, cleanup_adopt_and_a_lost_index_keep_the_raw_sid_links_and_timestamps) {
    const std::string root = fresh_root("kvflash_s10b_links");
    KvFlashSessionStore store(root);
    KvFlashSessionEntry e = entry("ses \"odd\" \xc3\xa9", 777);
    e.client = "claude"; e.archive_of = "feedfeedfeedfeed"; e.archive = "beefbeefbeefbeef"; e.created = 55;
    REQUIRE(fs::create_directories(store.dir(e.key)));
    REQUIRE(park_session(store.dir(e.key), {1, 2, 3}, 1, 0x10));
    store.upsert(e);
    REQUIRE(store.write_meta(e.key));
    REQUIRE(store.evict(e.key, {1, 2, 3}));
    KvFlashSessionEntry succ = entry("ses \"odd\" \xc3\xa9", 800);   // its successor instance, whose directory the link names
    succ.client = "claude"; succ.key = "beefbeefbeefbeef"; succ.archive_of = e.key;
    REQUIRE(fs::create_directories(store.dir(succ.key)));
    REQUIRE(park_session(store.dir(succ.key), {9}, 1, 0x20));
    store.upsert(succ);
    REQUIRE(store.write_meta(succ.key));
    // adopt (restore), then the restart cleanup of the directory, then the index is lost
    REQUIRE(store.adopt(e.key));
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(store.dir(e.key), ID, CB, lim(), l, why));
    kvflash_bundle_clean(store.dir(e.key), &l.head);
    REQUIRE(store.write_index());
    fs::remove(root + "/sessions/index.json");
    KvFlashScanReport rep;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(es.size() == 2);
    CHECK(es[1].key == e.key);                    // the older one
    CHECK(es[1].sid == e.sid);                    // the raw sid, not a hash of it
    CHECK(es[1].client == "claude");
    CHECK(es[1].archive_of == "feedfeedfeedfeed");
    CHECK(es[1].archive == "beefbeefbeefbeef");
    CHECK(es[1].created == 55);
    CHECK(es[1].last_used == 777);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_park_is_durable_only_when_committed_and_a_session_without_a_committed_head_is_never_marked_disk) {
    // pending, failed, rewrite, head generation
    CHECK(kvflash_session_park_durable(true, false, false, 0));    // the writer acknowledged the staged park
    CHECK(!kvflash_session_park_durable(true, true, false, 3));    // the staged park failed: an older head is not this session
    CHECK(!kvflash_session_park_durable(false, false, true, 3));   // the park was skipped (slab snapshot failed)
    CHECK(!kvflash_session_park_durable(false, false, false, 0));  // nothing ever committed
    CHECK(kvflash_session_park_durable(false, false, false, 3));   // restored from disk and untouched
    // the store refuses to list as disk a session whose directory holds no committed head
    const std::string root = fresh_root("kvflash_s10b_nohead");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    KvFlashSessionEntry e = entry("ses_x", 10);
    REQUIRE(fs::create_directories(store.dir(e.key)));
    std::ofstream(store.dir(e.key) + "/chunks.1.log") << "uncommitted";
    store.upsert(e);
    CHECK(!store.evict(e.key, {1, 2}));
    CHECK(store.find_key(e.key)->state == "parked-gpu");
    CHECK(fs::exists(store.dir(e.key) + "/chunks.1.log"));         // and nothing was removed
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_slot_reused_by_another_session_starts_its_bundle_bookkeeping_over_a_b_a) {
    const std::string root = fresh_root("kvflash_s10b_reuse");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    const std::string ka = kvflash_sid_key("header", "A"), kb = kvflash_sid_key("header", "B");
    // slot 0 hosts A: its bookkeeping holds A's committed head and generations
    KvFlashSlotBundle slot0 = KvFlashSlotBundle::for_session(ka, "A", "header", 10);
    REQUIRE(fs::create_directories(store.dir(ka)));
    REQUIRE(park_session(store.dir(ka), {1, 2, 3}, 2, 0x10));
    {
        std::vector<uint8_t> bytes;
        REQUIRE(kvflash_bundle_read(store.dir(ka) + "/head", bytes));
        REQUIRE(kvflash_head_unpack(bytes.data(), bytes.size(), ID, slot0.head));
        slot0.next_gen = 5; slot0.resume_L = 64; slot0.pending = true; slot0.debt = {"chunks.9.log"};
    }
    const std::string a_log = slurp(store.dir(ka) + "/chunks.1.log"), a_head = slurp(store.dir(ka) + "/head");
    CHECK(!slot0.fresh_park(false));                                 // A's bookkeeping as it stands would append to A's log
    // B takes the slot: nothing of A's carries over
    slot0 = KvFlashSlotBundle::for_session(kb, "B", "header", 20);
    CHECK(slot0.key == kb);
    CHECK(slot0.sid == "B");
    CHECK(slot0.head.gen == 0);
    CHECK(slot0.staged.gen == 0);
    CHECK(slot0.next_gen == 1);
    CHECK(slot0.resume_L == 0);
    CHECK(!slot0.pending);
    CHECK(!slot0.rewrite);
    CHECK(slot0.debt.empty());
    CHECK(slot0.fresh_park(false));                                   // B's first park writes a fresh log in B's own directory
    REQUIRE(fs::create_directories(store.dir(kb)));
    REQUIRE(park_session(store.dir(kb), {7, 8}, 1, 0x30));
    CHECK(slurp(store.dir(ka) + "/chunks.1.log") == a_log);           // A's log and head are byte-for-byte what A parked
    CHECK(slurp(store.dir(ka) + "/head") == a_head);
    // A comes back (adopted from its directory): its own head and counters, none of B's
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(store.dir(ka), ID, CB, lim(), l, why));
    const uint32_t top = std::max({l.head.gen, l.head.end_gen, l.head.ckpt_gen, l.head.trunk_gen});
    slot0 = KvFlashSlotBundle::adopted(ka, "A", "header", 10, std::move(l.head));
    CHECK(slot0.key == ka);
    CHECK(slot0.sid == "A");
    CHECK(slot0.head.history == (std::vector<int32_t>{1, 2, 3}));
    CHECK(slot0.next_gen == top + 1);                                 // past every file the head names
    CHECK(!slot0.pending);
    CHECK(slot0.debt.empty());
    CHECK(!slot0.fresh_park(false));                                  // A's next park appends to A's log, in A's directory
    CHECK(slurp(store.dir(kb) + "/head") != a_head);
    fs::remove_all(root);
}

// ── the book over the store ─────────────────────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, r02_a_slot_keeps_its_owner_with_persistence_off_and_a_matching_b_or_anon_cannot_take_it) {
    KvFlashSessionBook book;                  // no enable_store: sessions are not kept on disk
    book.init(2);
    REQUIRE(!book.persistent());
    book.begin_session(0, "opencode", "A", 1000);
    CHECK(book[0].sid == "A");
    CHECK(book[0].client == "opencode");
    CHECK(book[0].key.empty());               // no directory, no store entry
    CHECK(book.store().entries().empty());
    const std::vector<int32_t> hist = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto owners = owners_of(book, hist);
    CHECK(kvflash_pick_parked_owner(owners, "opencode", "B", hist, 0).slot != 0);
    CHECK(kvflash_pick_parked_owner(owners, "anon", "anon:9", hist, 0).slot != 0);
    CHECK(kvflash_pick_parked_owner(owners, "opencode", "A", hist, 0).slot == 0);   // A itself resumes
    book.reset(0);
    CHECK(book[0].sid.empty());
    CHECK(book[0].client.empty());
    CHECK(book[0].fresh_park(false));
}

TEST_CASE(KvflashSessionStoreFixture, r02_a_dropped_session_resets_the_owner_with_persistence_off_and_on) {
    const std::string root = fresh_root("kvflash_s10b_r02");
    KvFlashSessionBook off, on;
    off.init(1); on.init(1);
    on.enable_store(KvFlashSessionStore(root), ID);
    for (KvFlashSessionBook * b : {&off, &on}) {
        b->begin_session(0, "claude", "S", 5);
        CHECK(((*b)[0].sid == "S" && (*b)[0].client == "claude"));
        b->reset(0);
        CHECK(((*b)[0].sid.empty() && (*b)[0].client.empty() && (*b)[0].key.empty()));
    }
    fs::remove_all(root);
}

// R04: startup adoption, eviction without a turn, disk resume; A -> B -> A.
TEST_CASE(KvflashSessionStoreFixture, r04_an_adopted_session_evicted_without_a_turn_keeps_its_history_and_resumes_from_disk_a_b_a) {
    const std::string root = fresh_root("kvflash_s10b_r04");
    {
        KvFlashSessionStore seed(root);
        commit_session(seed, "header", "A", {1, 2, 3, 4}, 0x10, 200);
        commit_session(seed, "header", "B", {9, 8, 7}, 0x20, 100);
    }
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    rescan(book.store());
    REQUIRE(book.store().entries().size() == 2);
    const std::string ka = kvflash_sid_key("header", "A"), kb = kvflash_sid_key("header", "B");
    auto adopt = [&](const std::string & key, const std::vector<int32_t> & want) {
        CHECK(adopt_into_slot(book, key) == want);                             // the parked session starts with its history ...
        CHECK(book[0].head.history == want);                                   // ... and the bookkeeping keeps it for the eviction
    };
    adopt(ka, {1, 2, 3, 4});
    REQUIRE(book.evict(0, false));                                              // slot pressure, before any turn
    const KvFlashSessionEntry * a = book.store().find_key(ka);
    REQUIRE(a != nullptr);
    CHECK(a->state == "disk");
    CHECK(a->tokens == 4);
    CHECK(a->history == (std::vector<int32_t>{1, 2, 3, 4}));
    const KvFlashSessionEntry * pick = book.store().pick_disk("header", "A", {1, 2, 3, 4, 5}, 0);
    REQUIRE(pick != nullptr);
    CHECK(pick->key == ka);                                                     // the disk picker finds it again
    adopt(kb, {9, 8, 7});                                                       // B takes the slot
    REQUIRE(book.evict(0, false));
    adopt(ka, {1, 2, 3, 4});                                                    // and A comes back, with its history
    REQUIRE(book.evict(0, false));
    CHECK(book.store().find_key(ka)->history == (std::vector<int32_t>{1, 2, 3, 4}));
    CHECK(book.store().find_key(kb)->history == (std::vector<int32_t>{9, 8, 7}));
    CHECK(book.store().pick_disk("header", "B", {9, 8, 7, 1}, 0) != nullptr);
    fs::remove_all(root);
}

// R03: one owner per identity; a restore has four outcomes; a denied claim is busy.
TEST_CASE(KvflashSessionStoreFixture, r03_a_live_identity_is_busy_for_a_second_request_and_a_denied_resume_claim_is_busy) {
    const std::string root = fresh_root("kvflash_s10b_r03");
    {
        KvFlashSessionStore seed(root);
        commit_session(seed, "opencode", "D", {1, 2, 3}, 0x10, 100);
    }
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    rescan(book.store());
    book.begin_session(0, "opencode", "A", 1000);                               // A decodes in slot 0
    std::vector<bool> active = {true, false};
    CHECK(book.identity_busy("opencode", "A", active));                          // a second request for A is deferred, never a second owner
    CHECK(!book.identity_busy("opencode", "B", active));
    CHECK(!book.identity_busy("claude", "A", active));                           // the pair is the identity
    CHECK(!book.identity_busy("anon", "anon:7", active));
    CHECK(book.plan_restore("opencode", "A", {1, 2}, 0, active).pick == KvFlashRestorePick::busy);
    active[0] = false;                                                           // A finished: parked, resumable
    CHECK(!book.identity_busy("opencode", "A", active));
    auto plan = book.plan_restore("opencode", "D", {1, 2, 3, 4}, 0, active);
    CHECK(plan.pick == KvFlashRestorePick::load);
    CHECK(plan.key == kvflash_sid_key("opencode", "D"));
    book.resume_claim = [](const std::string &) { return false; };
    CHECK(book.plan_restore("opencode", "D", {1, 2, 3, 4}, 0, active).pick == KvFlashRestorePick::busy);
    book.resume_claim = [](const std::string &) { return true; };
    CHECK(book.plan_restore("opencode", "D", {1, 2, 3, 4}, 0, active).pick == KvFlashRestorePick::load);
    CHECK(book.plan_restore("opencode", "Z", {5, 6}, 0, active).pick == KvFlashRestorePick::none);   // nothing on disk for it: absent
    fs::remove_all(root);
}

// R01: the identity meta is durable before the first head is handed to the writer.
TEST_CASE(KvflashSessionStoreFixture, r01_a_failed_identity_meta_write_or_fsync_blocks_the_park_and_keeps_the_owner) {
    for (const char * op : {"write", "fsync"}) {
        const std::string root = fresh_root("kvflash_s10b_r01a");
        KvFlashSessionBook book;
        book.init(1);
        book.enable_store(KvFlashSessionStore(root), ID);
        fs::create_directories(root + "/sessions");
        book.begin_session(0, "opencode", "A", 1000);
        const std::string key = book[0].key;
        const KvFlashParkStats st{3, 100, 0};
        {
            IoFault f([&](std::string_view o, const std::string & path) { return o == op && (std::string_view(op) == "fsync" ? ends_with(path, key) : ends_with(path, "meta.json")); });
            CHECK(!book.publish_park(0, st, 1001));                              // the writer is not handed this park
        }
        CHECK(book[0].sid == "A");                                               // the owner stays
        CHECK(book[0].client == "opencode");
        CHECK(book[0].rewrite);                                                  // the park is owed, whole
        CHECK(!book.durable(0, false));                                          // not durable: slot pressure keeps the session in memory
        CHECK(!book.evict(0, false));
        CHECK(book.store().find_key(key)->state == "parked-gpu");
        CHECK(book.publish_park(0, st, 1002));                                   // the fault gone, the next park publishes the identity first
        REQUIRE(park_session(book.store().dir(key), {1, 2, 3}, 1, 0x10));
        KvFlashScanReport rep;                                                   // restart: the record is A's, nobody else's
        const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
        REQUIRE(es.size() == 1);
        CHECK(es[0].sid == "A");
        CHECK(es[0].client == "opencode");
        CHECK(es[0].state == "disk");
        fs::remove_all(root);
    }
}

TEST_CASE(KvflashSessionStoreFixture, r01_a_later_mutable_meta_failure_does_not_block_a_park_whose_identity_is_already_durable) {
    const std::string root = fresh_root("kvflash_s10b_r01b");
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    book.begin_session(0, "claude", "S", 1000);
    REQUIRE(book.publish_park(0, KvFlashParkStats{3, 10, 0}, 1001));
    IoFault f([](std::string_view o, const std::string & path) { return o == "write" && ends_with(path, "meta.json"); });
    CHECK(book.publish_park(0, KvFlashParkStats{5, 10, 0}, 1002));             // last_used/tokens are mutable: the head is the truth
    CHECK(!book[0].rewrite);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, p1b_a_park_publishes_the_preamble_binding_and_a_later_park_without_one_clears_it) {
    const std::string root = fresh_root("kvflash_s10b_binding");
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    book.begin_session(0, "claude", "S", 1000);
    KvFlashParkStats st{3, 10, 0};
    st.preamble = "00000000000000ab"; st.preamble_chunks = 4;
    REQUIRE(book.publish_park(0, st, 1001));
    const KvFlashSessionEntry * e = book.store().find_key(book[0].key);
    REQUIRE(e != nullptr);
    CHECK(e->preamble == "00000000000000ab");
    CHECK(e->preamble_chunks == 4);
    CHECK(book.publish_park(0, KvFlashParkStats{5, 10, 0}, 1002));   // the binding was released: the meta forgets it
    e = book.store().find_key(book[0].key);
    REQUIRE(e != nullptr);
    CHECK(e->preamble.empty());
    CHECK(e->preamble_chunks == 0);
    fs::remove_all(root);
}

// F01: an ownerless session an explicit request takes over is a new identity publication: the owner change is durable before a head.
TEST_CASE(KvflashSessionStoreFixture, f01_promoting_an_ownerless_session_to_an_explicit_identity_needs_its_owner_meta_durable_before_a_head) {
    const std::string root = fresh_root("kvflash_s10b_f01");
    KvFlashSessionEntry anon;
    {
        KvFlashSessionStore seed(root);
        anon = commit_session(seed, "anon", "anon:7", {1, 2, 3, 4}, 0x10, 100);
    }
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    rescan(book.store());
    adopt_into_slot(book, anon.key);
    CHECK(book[0].identity_durable);                                             // restored from its directory
    book.promote_owner(0, "claude", "S");
    CHECK(book[0].client == "claude");
    CHECK(!book[0].identity_durable);                                            // the new owner is on no disk yet
    {
        IoFault f([](std::string_view o, const std::string & path) { return o == "write" && ends_with(path, "meta.json"); });
        CHECK(!book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1001));        // no head may follow an owner nobody can prove
    }
    CHECK(book[0].rewrite);
    CHECK(!book.durable(0, false));
    CHECK(!book.evict(0, false));
    KvFlashScanReport r2;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, r2);            // a restart before the owner landed: still the old ownerless record
    REQUIRE(es.size() == 1);
    CHECK(es[0].client == "anon");
    CHECK(book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1002));              // the fault gone, the owner lands and the park may go
    CHECK(book[0].identity_durable);
    KvFlashScanReport r3;
    const auto after = kvflash_session_scan(root, ID, RB, HEAD_MAX, r3);
    REQUIRE(after.size() == 1);
    CHECK(after[0].client == "claude");
    CHECK(after[0].sid == "S");
    fs::remove_all(root);
}

// R17b: a parked session whose resume wait holds for the request is resumed before any other disk session is restored.
TEST_CASE(KvflashSessionStoreFixture, r17b_a_waiting_resume_restores_no_other_disk_session) {
    const std::string root = fresh_root("kvflash_s10b_wait_gate");
    {
        KvFlashSessionStore seed(root);
        commit_session(seed, "anon", "anon:5", {1, 2, 3}, 0x10, 100);   // B, ownerless, on disk, shares the prefix
    }
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    rescan(book.store());
    const std::vector<bool> active = {false, true};
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    const KvFlashSessionId who{"anon:9", "anon"};
    int restores = 0;
    auto restore = [&] {   // the engine's restore: plan, then read the bundle
        if (book.plan_restore(who.client, who.sid, prompt, 0, active).pick == KvFlashRestorePick::load) ++restores;
        return KvFlashRestore::absent;
    };
    using V = KvFlashAdmitVerdict;
    CHECK(kvflash_admit_gate(book, who, active, false, restore) == V::proceed);
    CHECK(restores == 1);                                             // nothing waits: B is restored
    restores = 0;
    CHECK(kvflash_admit_gate(book, who, active, true, restore) == V::proceed);
    CHECK(restores == 0);                                             // A's wait holds: no bundle read
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, r01_an_oversized_meta_is_refused_instead_of_published_unreadable) {
    const std::string root = fresh_root("kvflash_s10b_r01c");
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    book.begin_session(0, "body", std::string(2 << 20, 'x'), 1);
    CHECK(!book.publish_park(0, KvFlashParkStats{1, 1, 0}, 2));
    fs::remove_all(root);
}

// R17: a park the disk refused leaves the last acknowledged head, listed as disk.
TEST_CASE(KvflashSessionStoreFixture, a_refused_park_keeps_the_last_committed_bundle_and_the_next_turn_restores_it) {
    const std::string root = fresh_root("kvflash_s10b_refused_park");
    {
        KvFlashSessionStore seed(root);
        commit_session(seed, "opencode", "A", {1, 2, 3, 4}, 0x10, 200);
    }
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    rescan(book.store());
    const std::string key = kvflash_sid_key("opencode", "A");
    adopt_into_slot(book, key);                                              // turn 1's park is committed (gen != 0)
    REQUIRE(book[0].head.gen != 0);
    book[0].rewrite = true;                                                  // turn 2's park was refused: the session is newer than its head
    CHECK(!book.durable(0, false));
    CHECK(!book.evict(0, false));                                            // the ordinary eviction refuses it
    REQUIRE(book.release_to_last_commit(0));                                 // the host-drop fallback
    CHECK(fs::exists(book.store().dir(key) + "/head"));                      // the directory is still there
    const KvFlashSessionEntry * e = book.store().find_key(key);
    REQUIRE(e != nullptr);                                                   // its store entry and its owner too
    CHECK(e->state == "disk");
    CHECK(e->sid == "A");
    CHECK(e->client == "opencode");
    CHECK(e->history == (std::vector<int32_t>{1, 2, 3, 4}));                 // the older head's history, not turn 2's
    CHECK(book[0].key.empty());                                              // the slot is free for the next session
    book.begin_session(0, "opencode", "B", 300);                             // slot reuse
    CHECK(fs::exists(book.store().dir(key) + "/head"));
    const std::vector<bool> active(2, false);
    const auto plan = book.plan_restore("opencode", "A", {1, 2, 3, 4, 5, 6, 7, 8}, 0, active);
    CHECK(plan.pick == KvFlashRestorePick::load);
    CHECK(plan.key == key);
    CHECK(!book.release_to_last_commit(0));                                  // a session whose park never committed has nothing to keep
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_release_keeps_the_committed_bundle_when_the_index_write_fails) {
    const std::string root = fresh_root("kvflash_s10b_release_io");
    {
        KvFlashSessionStore seed(root);
        commit_session(seed, "opencode", "A", {1, 2, 3, 4}, 0x10, 200);
    }
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    rescan(book.store());
    const std::string key = kvflash_sid_key("opencode", "A");
    adopt_into_slot(book, key);
    REQUIRE(book[0].head.gen != 0);
    fs::remove(root + "/sessions/index.json");
    fs::create_directory(root + "/sessions/index.json");                   // the index cannot be replaced: every write_index fails
    REQUIRE(!book.store().write_index());
    CHECK(book.release_to_last_commit(0));                                 // released: the caller must not drop the directory
    CHECK(fs::exists(book.store().dir(key) + "/head"));
    CHECK(book[0].key.empty());
    const KvFlashSessionEntry * e = book.store().find_key(key);
    REQUIRE(e != nullptr);
    CHECK(e->state == "disk");
    CHECK(e->history == (std::vector<int32_t>{1, 2, 3, 4}));
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_release_and_restore_keep_a_failed_owner_publication_not_durable) {
    const std::string root = fresh_root("kvflash_s10b_owner_release");
    KvFlashSessionEntry anon;
    {
        KvFlashSessionStore seed(root);
        anon = commit_session(seed, "anon", "anon:7", {1, 2, 3, 4}, 0x10, 100);
    }
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    rescan(book.store());
    adopt_into_slot(book, anon.key);
    book.promote_owner(0, "claude", "S");
    {
        IoFault f([](std::string_view o, const std::string & path) { return o == "write" && ends_with(path, "meta.json"); });
        CHECK(!book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1001));
        REQUIRE(book.release_to_last_commit(0));                                 // the park was refused, the committed head stays
        adopt_into_slot(book, anon.key);                                         // the same process restores it while the fault persists
        CHECK(!book[0].identity_durable);                                        // the owner is still on no disk
        CHECK(!book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1002));         // so no head may follow
        CHECK(book[0].rewrite);
    }
    CHECK(book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1003));              // the fault gone, the owner lands
    CHECK(book[0].identity_durable);
    KvFlashScanReport r2;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, r2);
    REQUIRE(es.size() == 1);
    CHECK(es[0].client == "claude");
    fs::remove_all(root);
}

// ── the DiskStore tenant ────────────────────────────────────────────────────────────────────

namespace {
// A DiskStore over `root` with a clock and a roomy disk, so only the configured budget and TTL decide.
std::shared_ptr<DiskStore> open_disk(const std::string & root, uint64_t budget, uint64_t ttl, uint64_t * clock) {
    DiskCacheConfig cfg;
    cfg.cache_dir = root; cfg.budget_bytes = (size_t)budget; cfg.ttl_seconds = ttl;
    bool failed = false;
    return DiskStore::open(cfg, [clock] { return *clock; }, [] { return (uint64_t)1 << 40; }, failed);
}
}  // namespace

TEST_CASE(KvflashSessionStoreFixture, the_tenant_counts_the_sessions_and_the_oldest_disk_session_goes_first_when_the_budget_needs_room) {
    const std::string root = fresh_root("kvflash_s10b_tenant_lru");
    KvFlashSessionStore store(root);
    commit_session(store, "header", "old", {1, 2, 3}, 0x10, 100);
    commit_session(store, "header", "mid", {4, 5, 6}, 0x20, 200);
    commit_session(store, "header", "new", {7, 8, 9}, 0x30, 300);
    rescan(store);
    const uint64_t each = store.ledger()->usage() / 3;
    REQUIRE(each > 0);
    uint64_t clock = 1000;
    auto disk = open_disk(root, each * 3 + each / 2, 0, &clock);
    REQUIRE(disk != nullptr);
    const uint64_t id = attach_session_store(disk, store);
    CHECK(disk->tenant_bytes() == store.ledger()->usage());
    REQUIRE(disk->reserve(each));                                   // 4 of 3.5 units: the oldest session makes room
    CHECK(!fs::exists(store.dir(kvflash_sid_key("header", "old"))));
    CHECK(fs::exists(store.dir(kvflash_sid_key("header", "mid"))));
    CHECK(store.find_key(kvflash_sid_key("header", "old")) == nullptr);   // the owner thread learns of it at its next look
    CHECK(store.find_key(kvflash_sid_key("header", "mid")) != nullptr);
    disk->release(each);
    disk->detach_tenant(id);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_session_in_a_slot_a_foreign_bundle_and_a_claimed_session_are_never_evicted) {
    const std::string root = fresh_root("kvflash_s10b_tenant_protect");
    KvFlashSessionStore store(root);
    commit_session(store, "header", "gpu", {1, 2, 3}, 0x10, 100);   // oldest, but in a slot
    commit_session(store, "header", "disk", {4, 5, 6}, 0x20, 200);
    commit_session(store, "header", "claimed", {7, 8, 9}, 0x30, 300);
    REQUIRE(park_session(store.dir("foreign"), {1}, 1, 0x40, ID + 1));   // another server's: counted, never touched
    rescan(store);
    const std::string kg = kvflash_sid_key("header", "gpu"), kd = kvflash_sid_key("header", "disk"), kc = kvflash_sid_key("header", "claimed");
    REQUIRE(store.adopt(kg));                                       // it is in a slot
    REQUIRE(store.claim(kc));                                       // a restore is reading it
    const uint64_t used = store.ledger()->usage();
    uint64_t clock = 1000;
    auto disk = open_disk(root, used, 0, &clock);
    const uint64_t id = attach_session_store(disk, store);
    REQUIRE(disk->reserve(1));                                      // the budget is full: one session must go, and only "disk" may
    CHECK(fs::exists(store.dir(kg)));
    CHECK(!fs::exists(store.dir(kd)));
    CHECK(fs::exists(store.dir(kc)));
    CHECK(fs::exists(store.dir("foreign") + "/head"));
    disk->release(1);
    CHECK(!disk->reserve(used));                                    // nothing else may go: refused, and nothing is held for it
    CHECK(fs::exists(store.dir(kg)));
    CHECK(fs::exists(store.dir("foreign") + "/head"));
    disk->detach_tenant(id);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_session_the_budget_evicted_cannot_be_claimed_for_a_restore) {
    const std::string root = fresh_root("kvflash_s10b_tenant_claim");
    KvFlashSessionStore store(root);
    commit_session(store, "header", "A", {1, 2, 3}, 0x10, 100);
    rescan(store);
    const std::string ka = kvflash_sid_key("header", "A");
    REQUIRE(store.pick_disk("header", "A", {1, 2, 3, 4}, 0) != nullptr);   // the pick found it ...
    uint64_t clock = 1000;
    auto disk = open_disk(root, store.ledger()->usage(), 0, &clock);
    const uint64_t id = attach_session_store(disk, store);
    REQUIRE(disk->reserve(1));                                              // ... and the budget took it before the restore claimed it
    CHECK(!store.claim(ka));
    CHECK(!store.adopt(ka));
    CHECK(store.pick_disk("header", "A", {1, 2, 3, 4}, 0) == nullptr);
    disk->release(1);
    disk->detach_tenant(id);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, the_one_ttl_expires_unused_disk_sessions_and_leaves_the_ones_in_use) {
    const std::string root = fresh_root("kvflash_s10b_tenant_ttl");
    KvFlashSessionStore store(root);
    commit_session(store, "header", "stale", {1, 2, 3}, 0x10, 500);
    commit_session(store, "header", "fresh", {4, 5, 6}, 0x20, 950);
    commit_session(store, "header", "slot", {7, 8, 9}, 0x30, 400);
    rescan(store);
    REQUIRE(store.adopt(kvflash_sid_key("header", "slot")));
    uint64_t clock = 1000;
    auto disk = open_disk(root, (uint64_t)1 << 30, 100, &clock);     // 100 s
    const uint64_t id = attach_session_store(disk, store);           // attaching applies the budget and the TTL at once
    CHECK(!fs::exists(store.dir(kvflash_sid_key("header", "stale"))));   // 500 s idle
    CHECK(fs::exists(store.dir(kvflash_sid_key("header", "fresh"))));    // 50 s idle
    CHECK(fs::exists(store.dir(kvflash_sid_key("header", "slot"))));     // idle on disk, but attached to a slot
    disk->detach_tenant(id);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_park_the_one_budget_cannot_hold_is_refused_owed_whole_and_a_fitting_one_holds_nothing_after) {
    const std::string root = fresh_root("kvflash_s10b_tenant_park");
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    uint64_t clock = 1000;
    auto disk = open_disk(root, 1000, 0, &clock);
    const uint64_t id = attach_session_store(disk, book.store());
    book.begin_session(0, "claude", "S", 1000);
    CHECK(!book.publish_park(0, KvFlashParkStats{3, 5000, 0}, 1001));   // 5000 bytes in a 1000-byte budget
    CHECK(book[0].rewrite);                                              // owed whole, nothing submitted
    CHECK(disk->tenant_bytes() == 0);
    CHECK(book.publish_park(0, KvFlashParkStats{3, 600, 0}, 1002));
    CHECK(disk->tenant_bytes() == 600);
    CHECK(book.publish_park(0, KvFlashParkStats{4, 900, 0}, 1003));     // growth of 300 over what the ledger holds
    CHECK(disk->tenant_bytes() == 900);
    CHECK(disk->reserve(100));                                          // no hold is left over from the parks: 900 + 100 fits
    disk->release(100);
    disk->detach_tenant(id);
    fs::remove_all(root);
}

// A restore that does not happen (no room, a refused bundle) gives its claim back: the session is on disk and the budget may take it.
TEST_CASE(KvflashSessionStoreFixture, a_restore_that_does_not_happen_gives_its_claim_back_to_the_budget) {
    const std::string root = fresh_root("kvflash_s10c_unclaim");
    KvFlashSessionStore store(root);
    const auto a = commit_session(store, "opencode", "A", {1, 2, 3}, 0x10, 100);
    uint64_t last = 0;
    CHECK(store.ledger()->oldest(last));                  // on disk and unclaimed: it may go
    CHECK(store.claim(a.key));
    CHECK(!store.ledger()->oldest(last));                 // claimed by a restore: it may not
    store.unclaim(a.key);
    CHECK(store.ledger()->oldest(last));
    store.unclaim("no-such-key");                         // nothing to give back
    CHECK(store.claim(a.key));
    CHECK(store.adopt(a.key));                            // adopted: in a slot, the state decides, not the claim
    store.unclaim(a.key);
    CHECK(!store.ledger()->oldest(last));
    fs::remove_all(root);
}

// --kv-sessions: on by default whenever KVFlash and a disk dir are on; the old bundle-dir variable stays one release, with a warning.
TEST_CASE(KvflashSessionStoreFixture, sessions_are_on_whenever_kvflash_and_a_disk_dir_are_on_and_the_flag_can_turn_them_off) {
    auto plan = [](bool kvflash, KvFlashSessionsFlag flag, const std::string & dir, const std::string & env = "") {
        return kvflash_sessions_plan(kvflash, flag, dir, env);
    };
    const auto on = plan(true, KvFlashSessionsFlag::automatic, "/cache");
    CHECK(on.on); CHECK(on.root == "/cache"); CHECK(on.note.empty());
    CHECK(plan(true, KvFlashSessionsFlag::on, "/cache").on);
    CHECK(!plan(false, KvFlashSessionsFlag::automatic, "/cache").on);              // no KVFlash: no slots to park
    CHECK(!plan(false, KvFlashSessionsFlag::on, "/cache").on);
    const auto off = plan(true, KvFlashSessionsFlag::off, "/cache");
    CHECK(!off.on); CHECK(off.root.empty()); CHECK(!off.note.empty());
    const auto nodir = plan(true, KvFlashSessionsFlag::automatic, "");
    CHECK(!nodir.on); CHECK(nodir.note.empty());                                     // the default with no disk cache is quiet
    const auto asked = plan(true, KvFlashSessionsFlag::on, "");
    CHECK(!asked.on); CHECK(asked.note.find("--kv-cache-dir") != std::string::npos); // an explicit ask that cannot be met says why
    const auto env = plan(true, KvFlashSessionsFlag::automatic, "/cache", "/old/bundles");
    CHECK(env.on); CHECK(env.root == "/old/bundles");
    CHECK(env.note.find("LUCE_KVFLASH_SLOT_BUNDLE_DIR") != std::string::npos);     // the warning names the variable and the way out
    CHECK(env.note.find("--kv-cache-dir") != std::string::npos);
    CHECK(plan(true, KvFlashSessionsFlag::automatic, "", "/old/bundles").on);       // the override needs no cache dir
    CHECK(!plan(true, KvFlashSessionsFlag::off, "/cache", "/old/bundles").on);      // an explicit off wins over the variable
    CHECK(!plan(false, KvFlashSessionsFlag::automatic, "/cache", "/old/bundles").on);
    KvFlashSessionsFlag f = KvFlashSessionsFlag::automatic;
    CHECK(kvflash_sessions_flag_parse("on", f) && f == KvFlashSessionsFlag::on);
    CHECK(kvflash_sessions_flag_parse("off", f) && f == KvFlashSessionsFlag::off);
    CHECK(!kvflash_sessions_flag_parse("auto", f) && f == KvFlashSessionsFlag::off);   // only on|off; a bad value leaves it
}
