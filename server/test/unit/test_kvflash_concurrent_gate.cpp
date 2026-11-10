// C1.3-02: the pure decisions around N KVFlash sessions on one engine:
// when the seq engine is built (including explicit --max-concurrency 1),
// which architectures and kernels may run it, and the LRU-only profile that
// replaces the serial recall defaults. No GPU, no startup.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_concurrent_gate.h"
#include "../../src/common/kvflash_arena.h"
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashConcurrentGateFixture {};
}  // namespace

TEST_CASE(KvflashConcurrentGateFixture, seq_engine_is_wanted_for_n_gt_1_explicit_1_or_a_kvflash_arena_when_paged) {
    SeqRouteInput in;
    in.paged = true;
    CHECK(!seq_engine_wanted(in));                       // default N=1: the single-sequence path, as before
    in.n_slots = 2;
    CHECK(seq_engine_wanted(in));
    in.n_slots = 1; in.max_concurrency_set = true;       // explicit --max-concurrency 1: public routing unchanged without the switch
    CHECK(!seq_engine_wanted(in));
    in.test_switch = true;
    CHECK(seq_engine_wanted(in));
    in.test_switch = false;
    in.max_concurrency_set = false; in.kvflash_arena = true;
    CHECK(seq_engine_wanted(in));
    in.paged = false;                                    // never without paged attention
    CHECK(!seq_engine_wanted(in));
    in.n_slots = 8; in.max_concurrency_set = true;
    CHECK(!seq_engine_wanted(in));
}

TEST_CASE(KvflashConcurrentGateFixture, only_dense_qwen35_may_run_the_concurrent_arena) {
    KvFlashConcurrentRequest r;
    r.arch = "qwen35";
    CHECK(!kvflash_concurrent_refusal(r).has_value());
    for (const char * a : {"qwen35moe", "qwen3", "gemma4", "laguna", "deepseek4", ""}) {
        r.arch = a;
        const auto m = kvflash_concurrent_refusal(r);
        REQUIRE(m.has_value());
        CHECK(m->find("qwen35") != std::string::npos);
    }
}

TEST_CASE(KvflashConcurrentGateFixture, wmma_a_draft_and_tree_verification_are_refused_with_a_message_each) {
    auto base = [] { KvFlashConcurrentRequest r; r.arch = "qwen35"; return r; };
    {
        auto r = base(); r.wmma_env = true;               // until the kernel excludes null blocks
        const auto m = kvflash_concurrent_refusal(r);
        REQUIRE(m.has_value());
        CHECK(m->find("PAGED_WMMA") != std::string::npos);
        r.wmma_null_exclusion_ready = true;               // what qwen35_backend.cpp sets: the WMMA kernel skips all-null chunks
        CHECK(!kvflash_concurrent_refusal(r).has_value());
    }
    { auto r = base(); r.draft = true;  const auto m = kvflash_concurrent_refusal(r); REQUIRE(m.has_value()); CHECK(m->find("draft") != std::string::npos); }
    { auto r = base(); r.tree = true;   const auto m = kvflash_concurrent_refusal(r); REQUIRE(m.has_value()); CHECK(m->find("tree") != std::string::npos); }
}

namespace {
// A DFlash2 chain request that satisfies every C-0 condition; each test below breaks one.
KvFlashConcurrentRequest chain_request() {
    KvFlashConcurrentRequest r;
    r.arch = "qwen35";
    r.draft = true;
    r.chain_spec_switch = true;
    r.draft_local_same_device = true;
    r.draft_block = 16;
    r.fa_window = 0;
    return r;
}
}  // namespace

TEST_CASE(KvflashConcurrentGateFixture, a_dflash2_chain_is_allowed_only_with_the_chain_spec_switch_and_every_chain_condition) {
    CHECK(!kvflash_concurrent_refusal(chain_request()).has_value());
    for (int b : {2, 4, 8, 16}) { auto r = chain_request(); r.draft_block = b; CHECK(!kvflash_concurrent_refusal(r).has_value()); }
    auto refused = [&](KvFlashConcurrentRequest r, const char * what) {
        const auto m = kvflash_concurrent_refusal(r);
        REQUIRE(m.has_value());
        CHECK(m->find(what) != std::string::npos);
    };
    { auto r = chain_request(); r.chain_spec_switch = false;       refused(r, "LUCE_KVFLASH_CHAIN_SPEC"); }
    { auto r = chain_request(); r.tree = true;                     refused(r, "tree"); }
    { auto r = chain_request(); r.draft_local_same_device = false; refused(r, "draft"); }
    { auto r = chain_request(); r.draft_block = 17;                refused(r, "draft"); }
    { auto r = chain_request(); r.draft_block = 1;                 refused(r, "draft"); }
    { auto r = chain_request(); r.draft_block = 0;                 refused(r, "draft"); }
    { auto r = chain_request(); r.fa_window = 4096;                refused(r, "draft"); }
    { auto r = chain_request(); r.arch = "qwen35moe";              refused(r, "qwen35"); }
    { auto r = chain_request(); r.wmma_env = true;                 refused(r, "PAGED_WMMA"); }
    // the switch without a draft changes nothing
    { KvFlashConcurrentRequest r; r.arch = "qwen35"; r.chain_spec_switch = true; CHECK(!kvflash_concurrent_refusal(r).has_value()); }
}

TEST_CASE(KvflashConcurrentGateFixture, the_chain_spec_switch_reads_exactly_LUCE_KVFLASH_CHAIN_SPEC_equals_1) {
    const char * old = std::getenv("LUCE_KVFLASH_CHAIN_SPEC");
    const std::string saved = old ? old : "";
    unsetenv("LUCE_KVFLASH_CHAIN_SPEC");
    CHECK(!kvflash_chain_spec_switch());
    setenv("LUCE_KVFLASH_CHAIN_SPEC", "0", 1);  CHECK(!kvflash_chain_spec_switch());
    setenv("LUCE_KVFLASH_CHAIN_SPEC", "true", 1); CHECK(!kvflash_chain_spec_switch());
    setenv("LUCE_KVFLASH_CHAIN_SPEC", "1", 1);  CHECK(kvflash_chain_spec_switch());
    if (old) setenv("LUCE_KVFLASH_CHAIN_SPEC", saved.c_str(), 1); else unsetenv("LUCE_KVFLASH_CHAIN_SPEC");
}

TEST_CASE(KvflashConcurrentGateFixture, the_concurrent_profile_is_lru_only_by_default_and_rejects_c2_requests) {
    KvFlashProfileRequest p;                              // plain --kvflash: serial defaults would install QK, recall, ...
    const KvFlashProfile ok = kvflash_resolve_concurrent_profile(p);
    CHECK(ok.ok);
    CHECK(ok.policy == "lru-only");
    CHECK(ok.suppress_serial_defaults);                   // the recall env defaults must not be installed
    CHECK(!ok.log_line.empty());
    CHECK(ok.log_line.find("lru-only") != std::string::npos);
    for (int i = 0; i < 3; i++) {                         // C2 pieces, alone or on top of a full recall request
        for (int with_recall = 0; with_recall < 2; with_recall++) {
            KvFlashProfileRequest q;
            if (with_recall) { q.explicit_qk_policy = q.explicit_recall = q.exact_scorer = true; q.reselect_mode = 4; q.select_queries = 64; }
            switch (i) {
                case 0: q.explicit_uq = true; break;
                case 1: q.explicit_tau = true; break;
                case 2: q.explicit_preamble_pin = true; break;
            }
            const KvFlashProfile bad = kvflash_resolve_concurrent_profile(q);
            CHECK(!bad.ok);
            CHECK(!bad.error.empty());
            CHECK(bad.error.find("C2") != std::string::npos);
        }
    }
}

TEST_CASE(KvflashConcurrentGateFixture, the_recall_profile_needs_every_explicit_piece_and_installs_no_default) {
    KvFlashProfileRequest full;                           // C1.4: qk + mode 4 + exact scorer + recall blocks
    full.explicit_qk_policy = full.explicit_recall = full.exact_scorer = true;
    full.reselect_mode = 4; full.select_queries = 64;
    const KvFlashProfile ok = kvflash_resolve_concurrent_profile(full);
    CHECK(ok.ok);
    CHECK(ok.policy == "recall");
    CHECK(ok.suppress_serial_defaults);
    CHECK(ok.log_line.find("recall") != std::string::npos);
    for (int i = 0; i < 7; i++) {                         // a partial request is refused, not completed with defaults
        KvFlashProfileRequest q = full;
        switch (i) {
            case 0: q = {}; q.explicit_qk_policy = true; break;
            case 1: q = {}; q.explicit_recall = true; break;
            case 2: q = {}; q.exact_scorer = true; break;
            case 3: q = {}; q.reselect_mode = 4; break;
            case 4: q.reselect_mode = 3; break;                // mode 3 is not per-slot recall
            case 5: q.exact_scorer = false; break;
            case 6: q.select_queries = 0; break;
        }
        const KvFlashProfile bad = kvflash_resolve_concurrent_profile(q);
        CHECK(!bad.ok);
        CHECK(bad.error.find("together") != std::string::npos);
    }
}

TEST_CASE(KvflashConcurrentGateFixture, mode5_takes_the_pflash2_scorer_in_place_of_the_exact_scorer_and_its_queries) {
    KvFlashProfileRequest m5;                             // qk + mode 5 + pflash2 + recall blocks; no select queries
    m5.explicit_qk_policy = m5.explicit_recall = m5.pflash2_scorer = true;
    m5.reselect_mode = 5;
    const KvFlashProfile ok = kvflash_resolve_concurrent_profile(m5);
    CHECK(ok.ok);
    CHECK(ok.policy == "recall");
    CHECK(ok.suppress_serial_defaults);
    for (int i = 0; i < 4; i++) {
        KvFlashProfileRequest q = m5;
        switch (i) {
            case 0: q.pflash2_scorer = false; break;       // mode 5 without its scorer
            case 1: q.explicit_recall = false; break;
            case 2: q.explicit_qk_policy = false; break;
            case 3: q.reselect_mode = 4; break;            // pflash2 under mode 4 is not a profile
        }
        const KvFlashProfile bad = kvflash_resolve_concurrent_profile(q);
        CHECK(!bad.ok);
        CHECK(bad.error.find("together") != std::string::npos);
    }
}

TEST_CASE(KvflashConcurrentGateFixture, qwen35_with_no_recall_env_installs_the_c14_recall_profile_as_defaults) {
    KvFlashProfileRequest p;
    p.arch = "qwen35";
    const KvFlashProfile ok = kvflash_resolve_concurrent_profile(p);
    CHECK(ok.ok);
    CHECK(ok.policy == "recall");
    CHECK(ok.log_line.find("C1.4 defaults") != std::string::npos);
    const std::vector<std::pair<std::string, std::string>> want = {
        {"LUCE_KVFLASH_POLICY", "qk"},
        {"LUCE_KVFLASH_PREFILL_RESELECT", "4"},
        {"LUCE_KVFLASH_RESELECT_SCORER", "exact"},
        {"LUCE_KVFLASH_RECALL_BLOCKS", "6"},
        {"LUCE_KVFLASH_SCORE_POOL_RECENT", "1"},
        {"LUCE_KVFLASH_SELECT_UNIT", "2"},
        {"LUCE_KVFLASH_UQ_QUERIES", "0"},                  // uq is refused in concurrent mode
    };
    CHECK(ok.env_defaults == want);
}

namespace {
using EnvMap = std::map<std::string, std::string>;
KvFlashLookup lookup_of(const EnvMap & m) {
    return [&m](const char * k) -> const char * { auto it = m.find(k); return it == m.end() ? nullptr : it->second.c_str(); };
}
// What the server's env holds after the defaults went in: a set variable keeps its value.
EnvMap merged_after_defaults(const EnvMap & user) {
    EnvMap env = user;
    const KvFlashProfile p = kvflash_resolve_concurrent_profile(kvflash_concurrent_request("qwen35", "", false, false, lookup_of(user)));
    for (const auto & kv : p.env_defaults) env.emplace(kv.first, kv.second);
    return env;
}
std::string merged_error(const EnvMap & user, const std::string & cli_policy = "") {
    const EnvMap env = merged_after_defaults(user);
    return kvflash_merged_profile_error("qwen35", cli_policy, false, false, lookup_of(env));
}
}  // namespace

TEST_CASE(KvflashConcurrentGateFixture, merged_defaults_resolve_to_recall_through_the_explicit_branch) {
    CHECK(merged_error({}).empty());
    CHECK(merged_error({{"LUCE_KVFLASH_LOG_RESIDENT", "1"}}).empty());     // an unrelated variable changes nothing
    CHECK(merged_error({{"LUCE_KVFLASH_POLICY", "qk"}, {"LUCE_KVFLASH_PREFILL_RESELECT", "4"},
                        {"LUCE_KVFLASH_RESELECT_SCORER", "exact"}, {"LUCE_KVFLASH_RECALL_BLOCKS", "3"}}).empty());   // a coherent explicit user value is kept
}

TEST_CASE(KvflashConcurrentGateFixture, a_user_value_that_breaks_the_merged_profile_is_refused) {
    CHECK(!merged_error({{"LUCE_KVFLASH_POLICY", "drafter"}}).empty());
    CHECK(!merged_error({}, "drafter").empty());                           // --kvflash-policy applied later
    CHECK(!merged_error({{"LUCE_KVFLASH_PREFILL_RESELECT", "3"}}).empty());
    // Every recall variable kept at an off value: the merged request falls into the defaults branch again.
    const EnvMap all_off = {{"LUCE_KVFLASH_POLICY", "drafter"}, {"LUCE_KVFLASH_PREFILL_RESELECT", "0"},
                            {"LUCE_KVFLASH_RESELECT_SCORER", "cosine"},
                            {"LUCE_KVFLASH_RECALL_BLOCKS", "0"}};
    CHECK(!merged_error(all_off).empty());
    CHECK(!merged_error({{"LUCE_KVFLASH_PREFILL_RESELECT", "0"}, {"LUCE_KVFLASH_RECALL_BLOCKS", "0"}}, "drafter").empty());
    // The refusal follows installed defaults, so it must not say none were installed.
    CHECK(merged_error({{"LUCE_KVFLASH_PREFILL_RESELECT", "3"}}).find("no defaults are installed") == std::string::npos);
}

TEST_CASE(KvflashConcurrentGateFixture, multi_model_qwen35_with_no_env_is_lru_only_and_installs_nothing) {
    const EnvMap none;
    const KvFlashProfile p = kvflash_resolve_concurrent_profile(kvflash_concurrent_request("qwen35", "", false, true, lookup_of(none)));
    CHECK(p.ok);
    CHECK(p.policy == "lru-only");
    CHECK(p.env_defaults.empty());
    CHECK(p.log_line.find("single-model") != std::string::npos);
}

TEST_CASE(KvflashConcurrentGateFixture, multi_model_with_the_full_explicit_mode4_env_still_runs_recall) {
    const EnvMap env = {{"LUCE_KVFLASH_POLICY", "qk"}, {"LUCE_KVFLASH_PREFILL_RESELECT", "4"},
                        {"LUCE_KVFLASH_RESELECT_SCORER", "exact"}, {"LUCE_KVFLASH_RECALL_BLOCKS", "6"}};
    const KvFlashProfile p = kvflash_resolve_concurrent_profile(kvflash_concurrent_request("qwen35", "", false, true, lookup_of(env)));
    CHECK(p.ok);
    CHECK(p.policy == "recall");
    CHECK(p.env_defaults.empty());
}

TEST_CASE(KvflashConcurrentGateFixture, the_request_builder_reads_cli_over_env_and_mirrors_env_set_on) {
    EnvMap env = {{"LUCE_KVFLASH_POLICY", "qk"}, {"LUCE_KVFLASH_RECALL_BLOCKS", ""}, {"LUCE_KVFLASH_UQ_QUERIES", "0"},
                  {"LUCE_KVFLASH_PREFILL_RESELECT", "4"}, {"LUCE_KVFLASH_RESELECT_SCORER", "exact"}};
    KvFlashProfileRequest r = kvflash_concurrent_request("qwen35", "", false, false, lookup_of(env));
    CHECK(r.arch == "qwen35");
    CHECK(r.explicit_qk_policy);
    CHECK(!r.explicit_lru_policy);
    CHECK(!r.explicit_recall);                 // empty counts as unset
    CHECK(!r.explicit_uq);                     // "0" counts as unset
    CHECK(r.reselect_mode == 4);
    CHECK(r.exact_scorer);
    CHECK(!r.pflash2_scorer);
    CHECK(r.select_queries == kQueryCaptureTokens);   // mode 4 keeps the capture window, as the backend derives it
    CHECK(!r.multi_model);
    r = kvflash_concurrent_request("qwen35", "lru", true, true, lookup_of(env));
    CHECK(r.explicit_lru_policy);              // the CLI policy wins over the env policy
    CHECK(!r.explicit_qk_policy);
    CHECK(r.explicit_tau);
    CHECK(r.multi_model);
    env["LUCE_KVFLASH_RECALL_BLOCKS"] = "6";
    env["LUCE_KVFLASH_PREFILL_RESELECT"] = "0";
    env["LUCE_KVFLASH_PREAMBLE_ROWS"] = "6144";
    r = kvflash_concurrent_request("qwen35", "", false, false, lookup_of(env));
    CHECK(r.explicit_recall);
    CHECK(r.select_queries == 0);              // no mode 4, no queries
    CHECK(r.explicit_preamble_pin);
}

TEST_CASE(KvflashConcurrentGateFixture, explicit_lru_policy_keeps_lru_only_and_installs_no_defaults) {
    KvFlashProfileRequest p;
    p.arch = "qwen35";
    p.explicit_lru_policy = true;
    const KvFlashProfile ok = kvflash_resolve_concurrent_profile(p);
    CHECK(ok.ok);
    CHECK(ok.policy == "lru-only");
    CHECK(ok.env_defaults.empty());
    CHECK(ok.log_line.find("lru-only") != std::string::npos);
}

TEST_CASE(KvflashConcurrentGateFixture, explicit_full_mode4_and_mode5_install_no_defaults) {
    KvFlashProfileRequest m4;
    m4.arch = "qwen35";
    m4.explicit_qk_policy = m4.explicit_recall = m4.exact_scorer = true;
    m4.reselect_mode = 4; m4.select_queries = 64;
    KvFlashProfile ok = kvflash_resolve_concurrent_profile(m4);
    CHECK(ok.ok);
    CHECK(ok.policy == "recall");
    CHECK(ok.env_defaults.empty());
    CHECK(ok.log_line.find("explicit env") != std::string::npos);
    KvFlashProfileRequest m5;
    m5.arch = "qwen35";
    m5.explicit_qk_policy = m5.explicit_recall = m5.pflash2_scorer = true;
    m5.reselect_mode = 5;
    ok = kvflash_resolve_concurrent_profile(m5);
    CHECK(ok.ok);
    CHECK(ok.policy == "recall");
    CHECK(ok.env_defaults.empty());
}

TEST_CASE(KvflashConcurrentGateFixture, qwen35_partial_and_uq_requests_are_still_refused) {
    KvFlashProfileRequest part;
    part.arch = "qwen35";
    part.explicit_recall = true;
    KvFlashProfile bad = kvflash_resolve_concurrent_profile(part);
    CHECK(!bad.ok);
    CHECK(bad.error.find("together") != std::string::npos);
    CHECK(bad.env_defaults.empty());
    KvFlashProfileRequest uq;
    uq.arch = "qwen35";
    uq.explicit_uq = true;
    bad = kvflash_resolve_concurrent_profile(uq);
    CHECK(!bad.ok);
    CHECK(bad.error.find("C2") != std::string::npos);
}

TEST_CASE(KvflashConcurrentGateFixture, other_architectures_stay_lru_only_with_no_defaults) {
    for (const char * a : {"qwen35moe", "qwen3", ""}) {
        KvFlashProfileRequest p;
        p.arch = a;
        const KvFlashProfile ok = kvflash_resolve_concurrent_profile(p);
        CHECK(ok.ok);
        CHECK(ok.policy == "lru-only");
        CHECK(ok.env_defaults.empty());
    }
}

TEST_CASE(KvflashConcurrentGateFixture, test_switch_opens_only_on_exactly_1) {
    const char * old = std::getenv("LUCE_KVFLASH_CONCURRENT_TEST");
    unsetenv("LUCE_KVFLASH_CONCURRENT_TEST");
    CHECK(!luce::common::kvflash_concurrent_test_switch());
    setenv("LUCE_KVFLASH_CONCURRENT_TEST", "0", 1);
    CHECK(!luce::common::kvflash_concurrent_test_switch());
    setenv("LUCE_KVFLASH_CONCURRENT_TEST", "1", 1);
    CHECK(luce::common::kvflash_concurrent_test_switch());
    if (old) setenv("LUCE_KVFLASH_CONCURRENT_TEST", old, 1); else unsetenv("LUCE_KVFLASH_CONCURRENT_TEST");
}

TEST_CASE(KvflashConcurrentGateFixture, wmma_is_active_when_the_launcher_would_run_it_not_only_for_a_literal_1) {
    // The launcher: on unless LUCE_PAGED_WMMA parses to 0, and only on an RDNA4 device.
    CHECK(kvflash_paged_wmma_active(nullptr, true));     // unset: the kernel is on by default
    CHECK(kvflash_paged_wmma_active("1", true));
    CHECK(kvflash_paged_wmma_active("2", true));
    CHECK(!kvflash_paged_wmma_active("", true));         // atoi("") == 0: the launcher reads it as off
    CHECK(!kvflash_paged_wmma_active("0", true));        // the opt-out
    CHECK(!kvflash_paged_wmma_active(nullptr, false));   // gfx1151 and NVIDIA never take the kernel
    CHECK(!kvflash_paged_wmma_active("1", false));
    KvFlashConcurrentRequest r; r.arch = "qwen35";
    r.wmma_env = kvflash_paged_wmma_active(nullptr, true);
    const auto m = kvflash_concurrent_refusal(r);
    REQUIRE(m.has_value());
    CHECK(m->find("LUCE_PAGED_WMMA=0") != std::string::npos);   // the message names the way out
}

TEST_CASE(KvflashConcurrentGateFixture, the_page_stream_must_exist_on_an_async_dma_build) {
    CHECK(kvflash_arena_stream_usable(true, true));
    CHECK(!kvflash_arena_stream_usable(true, false));    // a failed cudaStreamCreate would drop the completion wait
    CHECK(kvflash_arena_stream_usable(false, false));    // no async DMA: nothing to wait for
}

TEST_CASE(KvflashConcurrentGateFixture, host_mib_is_validated_not_used_as_read) {
    CHECK(kvflash_parse_host_mib(nullptr) == std::optional<int64_t>(16384));
    CHECK(kvflash_parse_host_mib("") == std::optional<int64_t>(16384));
    CHECK(kvflash_parse_host_mib("2048") == std::optional<int64_t>(2048));
    CHECK(!kvflash_parse_host_mib("0").has_value());
    CHECK(!kvflash_parse_host_mib("-5").has_value());
    CHECK(!kvflash_parse_host_mib("abc").has_value());
    CHECK(!kvflash_parse_host_mib("12x").has_value());
}
