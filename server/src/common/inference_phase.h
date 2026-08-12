#pragma once

#include <cstdint>

namespace dflash::common {

enum class InferencePhase : int32_t {
    Unspecified = 0,
    Exact = 1,
    Dense = 2,
    Sparse = 3,
    Decode = 4,
    Verify = 5,
    ReferenceExact = 6,
    Sequential = 7,
    Batched = 8,
};

constexpr int32_t inference_phase_wire_value(InferencePhase phase) {
    return static_cast<int32_t>(phase);
}

constexpr bool inference_phase_from_wire_value(int32_t value,
                                                InferencePhase & phase) {
    if (value < inference_phase_wire_value(InferencePhase::Unspecified) ||
        value > inference_phase_wire_value(InferencePhase::Batched)) {
        return false;
    }
    phase = static_cast<InferencePhase>(value);
    return true;
}

} // namespace dflash::common
