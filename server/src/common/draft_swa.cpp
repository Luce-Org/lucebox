#include "draft_swa.h"

#include "internal.h"

namespace dflash::common {

DraftSwaOverrideResult apply_draft_swa_window_override(
        DraftWeights & weights, int requested_window) {
    DraftSwaOverrideResult result;
    result.effective_window = weights.swa_window;
    result.total_layers = static_cast<int>(weights.layers.size());

    bool any_swa = false;
    for (const DraftLayer & layer : weights.layers) {
        if (layer.is_swa) {
            any_swa = true;
            ++result.swa_layers;
        }
    }

    if (requested_window <= 0) {
        return result;
    }

    if (!weights.swa_pattern_loaded && !any_swa && weights.layers.size() > 1) {
        for (size_t index = 0; index + 1 < weights.layers.size(); ++index) {
            weights.layers[index].is_swa = true;
        }
        result.swa_layers = result.total_layers - 1;
        result.inferred_legacy_pattern = true;
    }

    weights.swa_window = requested_window;
    result.effective_window = requested_window;
    return result;
}

} // namespace dflash::common
