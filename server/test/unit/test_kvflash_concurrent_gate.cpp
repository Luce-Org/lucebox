// Pure decisions around N KVFlash sessions on one seq engine:
// when the seq engine is built, which architectures and chains may run it,
// and when concurrent per-slot recall runs vs LRU-only. No GPU, no startup.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_concurrent_gate.h"
#include "../../src/common/kvflash_arena.h"

using namespace luce::common;

namespace {
struct KvflashConcurrentGateFixture {};

KvFlashConcurrentRequest chain_request() {
    KvFlashConcurrentRequest r;
    r.arch = "qwen35";
    r.draft = true;
    r.draft_local_same_device = true;
    r.draft_block = 16;
    r.fa_window = 0;
    return r;
}
}  // namespace

TEST_CASE(KvflashConcurrentGateFixture, seq_engine_is_wanted_for_n_gt_1_or_a_kvflash_arena_when_paged) {
    SeqRouteInput in;
    in.paged = true;
    CHECK(!seq_engine_wanted(in));
    in.n_slots = 2;
    CHECK(seq_engine_wanted(in));
    in.n_slots = 1;
    in.kvflash_arena = true;
    CHECK(seq_engine_wanted(in));
    in.paged = false;
    CHECK(!seq_engine_wanted(in));
    in.n_slots = 8;
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

TEST_CASE(KvflashConcurrentGateFixture, a_dflash2_chain_is_allowed_when_every_chain_condition_holds) {
    CHECK(!kvflash_concurrent_refusal(chain_request()).has_value());
    for (int b : {2, 4, 8, 16}) {
        auto r = chain_request();
        r.draft_block = b;
        CHECK(!kvflash_concurrent_refusal(r).has_value());
    }
    auto refused = [&](KvFlashConcurrentRequest r, const char * what) {
        const auto m = kvflash_concurrent_refusal(r);
        REQUIRE(m.has_value());
        CHECK(m->find(what) != std::string::npos);
    };
    { auto r = chain_request(); r.tree = true;                     refused(r, "tree"); }
    { auto r = chain_request(); r.draft_local_same_device = false; refused(r, "draft"); }
    { auto r = chain_request(); r.draft_block = 17;                refused(r, "draft"); }
    { auto r = chain_request(); r.draft_block = 1;                 refused(r, "draft"); }
    { auto r = chain_request(); r.draft_block = 0;                 refused(r, "draft"); }
    { auto r = chain_request(); r.fa_window = 4096;                refused(r, "draft"); }
    { auto r = chain_request(); r.arch = "qwen35moe";              refused(r, "qwen35"); }
}

TEST_CASE(KvflashConcurrentGateFixture, concurrent_recall_is_default_on_single_model_qwen35_unless_lru) {
    CHECK(kvflash_concurrent_use_recall("qwen35", "", "", false));
    CHECK(kvflash_concurrent_use_recall("qwen35", "qk", "lru", false));     // CLI overrides env
    CHECK(!kvflash_concurrent_use_recall("qwen35", "lru", "", false));
    CHECK(!kvflash_concurrent_use_recall("qwen35", "", "lru", false));
    CHECK(!kvflash_concurrent_use_recall("qwen35", "", "", true));          // multi-model stays LRU-only
    for (const char * a : {"qwen35moe", "qwen3", ""}) {
        CHECK(!kvflash_concurrent_use_recall(a, "", "", false));
    }
}

TEST_CASE(KvflashConcurrentGateFixture, the_page_stream_must_exist_on_an_async_dma_build) {
    CHECK(kvflash_arena_stream_usable(true, true));
    CHECK(!kvflash_arena_stream_usable(true, false));
    CHECK(kvflash_arena_stream_usable(false, false));
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
