// Ordered expert-target config parsing and target materialization helpers.
//
// This is the planning-side adapter for sparse MoE backends that want an
// ordered target list (for example "cuda:0,cuda:1,cpu") without exposing
// tier-specific naming. It parses user config, optionally auto-discovers
// device capacities, and emits ExpertSplitTarget entries for the planner.

#pragma once

#include "expert_split_plan.h"
#include "placement/placement_backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct ExpertSplitTargetSpec {
    PlacementBackend backend = PlacementBackend::Auto;
    int device_id = -1;
    bool auto_capacity = true;
    uint64_t capacity_bytes = 0;
    bool unlimited = false;
};

bool parse_expert_split_target_list(const std::string & value,
                                    std::vector<ExpertSplitTargetSpec> & out,
                                    std::string * err = nullptr);

bool parse_expert_split_capacity_overrides(const std::string & value,
                                           std::vector<uint64_t> & out_bytes,
                                           std::string * err = nullptr);

bool build_expert_split_targets(const std::vector<ExpertSplitTargetSpec> & specs,
                                uint64_t primary_capacity_bytes,
                                std::vector<ExpertSplitTarget> & out,
                                std::string * err = nullptr);

bool resolve_expert_split_targets_from_env(const char * targets_env_name,
                                           const char * caps_env_name,
                                           uint64_t primary_capacity_bytes,
                                           std::vector<ExpertSplitTarget> & out,
                                           std::string * err = nullptr);

bool validate_primary_expert_split_target(
    const std::vector<ExpertSplitTarget> & targets,
    PlacementBackend local_backend,
    int local_device_id,
    std::string * err = nullptr);

}  // namespace dflash::common
