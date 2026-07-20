#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_routing_stats.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace dflash::common;

#define TEST_ASSERT(cond) do { \
    auto _cpputf_exception = CppUnitTestFramework::Assert::IsTrue(static_cast<bool>(cond), #cond); \
    if (_cpputf_exception) { \
        throw *_cpputf_exception; \
    } \
} while (0)
static void expect(bool cond, const char * msg) { (void) msg; TEST_ASSERT(cond); }

namespace {
struct Qwen35MoeRoutingStatsFixture {};
}

TEST_CASE(Qwen35MoeRoutingStatsFixture, moe_routing_stats_suite) {
    MoeHybridRoutingStats stats;
    expect(stats.init(2, 4, 2), "init");
    expect(stats.matches(2, 4, 2), "matches after init");

    const int32_t layer0_a[] = {2, 1};
    const int32_t layer0_b[] = {2, 3};
    const int32_t layer1_a[] = {0, 0};

    expect(stats.observe(0, layer0_a, 2), "observe layer0_a");
    expect(stats.observe(0, layer0_b, 2), "observe layer0_b");
    expect(stats.observe(1, layer1_a, 2), "observe layer1_a");

    expect(stats.count(0, 2) == 2, "layer0 expert2 count");
    expect(stats.count(0, 1) == 1, "layer0 expert1 count");
    expect(stats.count(0, 3) == 1, "layer0 expert3 count");
    expect(stats.count(1, 0) == 2, "layer1 expert0 count");
    expect(stats.layer_totals[0] == 4, "layer0 total");
    expect(stats.layer_totals[1] == 2, "layer1 total");

    std::vector<int> ranked0 = stats.ranked_experts(0);
    expect(ranked0.size() == 4, "ranked size");
    expect(ranked0[0] == 2, "ranked leader");

    std::vector<int> hot0 = stats.hot_experts(0, 2);
    expect(hot0.size() == 2, "hot size");
    expect(hot0[0] == 2, "hot leader");

    const auto tmp = std::filesystem::temp_directory_path() / "moe-hybrid-routing-stats-test.csv";
    std::string err;
    expect(stats.save_csv(tmp.string(), &err), err.c_str());

    MoeHybridRoutingStats loaded;
    expect(MoeHybridRoutingStats::load_csv(tmp.string(), loaded, &err), err.c_str());
    expect(loaded.matches(2, 4, 2), "loaded matches dims");
    expect(loaded.count(0, 2) == 2, "loaded count");
    expect(loaded.layer_totals[1] == 2, "loaded total");

    std::filesystem::remove(tmp);
    std::printf("OK\n");
}
