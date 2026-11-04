// usage.luce: per-response output ids and decode step counts for Lucepilot.
//
// With LUCE_PILOT_USAGE on (=1; "0" or unset is off) the OpenAI chat
// completion response, streamed or not, carries in its usage object
//
//   "luce": {"output_ids": [generated token ids in order, prompt excluded],
//            "steps": ..., "draft_tokens": ..., "accepted_tokens": ...,
//            "target_forwards": ...}
//
// with the counts of DecodeStepCounts (the [spec-decode] stderr lines' steps=,
// accepted=A/O and target_forwards=). A backend that does not count reports
// them as null. With the switch off the field does not exist.

#pragma once

#include "common/generation_types.h"
#include "common/pilot_switch.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <vector>

namespace luce::common {

inline constexpr const char * kPilotUsageSwitch = "LUCE_PILOT_USAGE";

// Read once per process.
inline bool pilot_usage_enabled() {
    static const luce::pilot::Switch usage(kPilotUsageSwitch);
    return usage.on();
}

inline nlohmann::json pilot_usage_json(const std::vector<int32_t> & output_ids,
                                       const DecodeStepCounts & counts) {
    auto count = [&](int value) {
        return counts.reported ? nlohmann::json(value) : nlohmann::json(nullptr);
    };
    nlohmann::json luce = nlohmann::json::object();
    luce["output_ids"] = output_ids;
    luce["steps"] = count(counts.steps);
    luce["draft_tokens"] = count(counts.draft_tokens);
    luce["accepted_tokens"] = count(counts.accepted_tokens);
    luce["target_forwards"] = count(counts.target_forwards);
    return luce;
}

// Adds usage["luce"] when `enabled`; leaves `usage` untouched otherwise.
inline void attach_pilot_usage(nlohmann::json & usage, bool enabled,
                               const std::vector<int32_t> & output_ids,
                               const DecodeStepCounts & counts) {
    if (!enabled) return;
    usage["luce"] = pilot_usage_json(output_ids, counts);
}

}  // namespace luce::common
