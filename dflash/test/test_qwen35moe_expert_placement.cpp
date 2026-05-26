#include "qwen35moe_expert_placement.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace dflash::common;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

int main() {
    Qwen35MoeRoutingStats stats;
    stats.n_layer = 2;
    stats.n_expert = 4;
    stats.n_expert_used = 2;
    stats.counts = {
        100, 80, 60, 40,   // layer 0
        100, 1, 1, 1       // layer 1
    };
    stats.layer_totals = {280, 103};

    Qwen35MoeExpertPlacement placement;
    std::string err;
    expect(Qwen35MoeExpertPlacement::build_from_stats(stats, /*total_hot_budget=*/4,
                                                      /*min_hot_per_layer=*/1,
                                                      placement, &err), err.c_str());
    expect(placement.n_layer == 2, "n_layer");
    expect(placement.hot_counts.size() == 2, "hot_counts size");
    expect(placement.hot_counts[0] == 3, "layer0 got extra hot slots");
    expect(placement.hot_counts[1] == 1, "layer1 kept minimum hot slot");
    expect(placement.is_hot(0, 0), "layer0 expert0 hot");
    expect(placement.is_hot(0, 1), "layer0 expert1 hot");
    expect(placement.is_hot(0, 2), "layer0 expert2 hot");
    expect(!placement.is_hot(0, 3), "layer0 expert3 cold");
    expect(placement.is_hot(1, 0), "layer1 expert0 hot");
    expect(!placement.is_hot(1, 1), "layer1 expert1 cold");

    TargetWeights w;
    w.is_moe = true;
    w.n_layer = 2;
    w.n_expert = 4;
    w.n_expert_used = 2;
    expect(placement.matches(w), "placement matches weights");

    const auto tmp = std::filesystem::temp_directory_path() / "qwen35moe-placement-test.json";
    expect(placement.save_json(tmp.string(), &err), err.c_str());
    Qwen35MoeExpertPlacement loaded;
    expect(Qwen35MoeExpertPlacement::load_json(tmp.string(), loaded, &err), err.c_str());
    expect(loaded.hot_counts == placement.hot_counts, "loaded hot counts");
    expect(loaded.hot_expert_ids == placement.hot_expert_ids, "loaded hot ids");
    std::filesystem::remove(tmp);

    std::printf("OK\n");
    return 0;
}
