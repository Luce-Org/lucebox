#include "CppUnitTestFramework.hpp"
#include "common/draft_swa.h"
#include "internal.h"

#include <initializer_list>
#include <vector>

using namespace dflash::common;

namespace {

struct DraftSwaFixture {};

DraftWeights make_weights(std::initializer_list<bool> pattern,
                          bool pattern_loaded,
                          int window) {
    DraftWeights weights;
    weights.n_layer = static_cast<int>(pattern.size());
    weights.layers.resize(pattern.size());
    weights.swa_window = window;
    weights.swa_pattern_loaded = pattern_loaded;
    size_t index = 0;
    for (bool is_swa : pattern) {
        weights.layers[index++].is_swa = is_swa;
    }
    return weights;
}

std::vector<bool> layer_pattern(const DraftWeights & weights) {
    std::vector<bool> pattern;
    pattern.reserve(weights.layers.size());
    for (const DraftLayer & layer : weights.layers) {
        pattern.push_back(layer.is_swa);
    }
    return pattern;
}

} // namespace

TEST_CASE(DraftSwaFixture, override_preserves_loaded_mixed_pattern) {
    DraftWeights weights = make_weights({true, false, true, false}, true, 1024);

    const DraftSwaOverrideResult result =
        apply_draft_swa_window_override(weights, 2048);

    CHECK(layer_pattern(weights) == std::vector<bool>({true, false, true, false}));
    CHECK(weights.swa_window == 2048);
    CHECK(result.effective_window == 2048);
    CHECK(result.swa_layers == 2);
    CHECK(result.total_layers == 4);
    CHECK(!result.inferred_legacy_pattern);
}

TEST_CASE(DraftSwaFixture, override_preserves_loaded_all_full_pattern) {
    DraftWeights weights = make_weights({false, false, false}, true, 1024);

    const DraftSwaOverrideResult result =
        apply_draft_swa_window_override(weights, 2048);

    CHECK(layer_pattern(weights) == std::vector<bool>({false, false, false}));
    CHECK(result.swa_layers == 0);
    CHECK(!result.inferred_legacy_pattern);
}

TEST_CASE(DraftSwaFixture, override_infers_legacy_pattern_when_metadata_is_missing) {
    DraftWeights weights = make_weights({false, false, false, false, false}, false, 0);

    const DraftSwaOverrideResult result =
        apply_draft_swa_window_override(weights, 2048);

    CHECK(layer_pattern(weights) == std::vector<bool>({true, true, true, true, false}));
    CHECK(result.swa_layers == 4);
    CHECK(result.total_layers == 5);
    CHECK(result.inferred_legacy_pattern);
}

TEST_CASE(DraftSwaFixture, nonpositive_override_is_a_noop) {
    DraftWeights weights = make_weights({false, false, false}, false, 0);

    const DraftSwaOverrideResult result =
        apply_draft_swa_window_override(weights, 0);

    CHECK(layer_pattern(weights) == std::vector<bool>({false, false, false}));
    CHECK(weights.swa_window == 0);
    CHECK(result.swa_layers == 0);
    CHECK(!result.inferred_legacy_pattern);
}

TEST_CASE(DraftSwaFixture, override_is_idempotent) {
    DraftWeights weights = make_weights({false, false, false}, false, 0);
    apply_draft_swa_window_override(weights, 2048);
    const std::vector<bool> first_pattern = layer_pattern(weights);

    const DraftSwaOverrideResult result =
        apply_draft_swa_window_override(weights, 4096);

    CHECK(layer_pattern(weights) == first_pattern);
    CHECK(weights.swa_window == 4096);
    CHECK(result.swa_layers == 2);
    CHECK(!result.inferred_legacy_pattern);
}
