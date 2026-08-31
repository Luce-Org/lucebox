#pragma once

namespace dflash::common {

struct DraftWeights;

struct DraftSwaOverrideResult {
    int  effective_window = 0;
    int  swa_layers = 0;
    int  total_layers = 0;
    bool inferred_legacy_pattern = false;
};

DraftSwaOverrideResult apply_draft_swa_window_override(
    DraftWeights & weights, int requested_window);

} // namespace dflash::common
