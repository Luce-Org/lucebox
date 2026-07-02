#include "expert_split_plan.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

namespace dflash::common {

namespace {

struct RankedUnitRef {
    size_t unit_index = 0;
    double value = 0.0;
};

uint64_t usable_bytes(const ExpertSplitTarget & target) {
    if (target.unlimited) {
        return std::numeric_limits<uint64_t>::max();
    }
    if (target.capacity_bytes <= target.reserved_bytes) {
        return 0;
    }
    return target.capacity_bytes - target.reserved_bytes;
}

bool better_ranked_unit(const ExpertSplitUnit & lhs,
                        double lhs_value,
                        const ExpertSplitUnit & rhs,
                        double rhs_value) {
    if (lhs_value != rhs_value) return lhs_value > rhs_value;
    if (lhs.score != rhs.score) return lhs.score > rhs.score;
    if (lhs.bytes != rhs.bytes) return lhs.bytes < rhs.bytes;
    if (lhs.layer != rhs.layer) return lhs.layer < rhs.layer;
    return lhs.expert < rhs.expert;
}

bool can_fit(const ExpertSplitTarget & target,
             uint64_t used,
             uint64_t bytes) {
    if (target.unlimited) return true;
    const uint64_t avail = usable_bytes(target);
    return used <= avail && bytes <= avail - used;
}

int floor_for_target(const ExpertSplitConfig & cfg, size_t target_index) {
    if (target_index >= cfg.min_per_layer_by_target.size()) {
        return 0;
    }
    return std::max(0, cfg.min_per_layer_by_target[target_index]);
}

void assign_unit(ExpertSplitPlan & plan,
                 const ExpertSplitUnit & unit,
                 int target_index) {
    ExpertSplitAssignment & dst = plan.at(unit.layer, unit.expert);
    dst.target_index = target_index;
    dst.bytes = unit.bytes;
    dst.score = unit.score;
    if (target_index >= 0 &&
        (size_t) target_index < plan.target_used_bytes.size()) {
        plan.target_used_bytes[(size_t) target_index] += unit.bytes;
    }
}

int choose_first_fit_target(const ExpertSplitPlan & plan,
                            const ExpertSplitUnit & unit) {
    for (size_t i = 0; i < plan.targets.size(); ++i) {
        if (can_fit(plan.targets[i], plan.target_used_bytes[i], unit.bytes)) {
            return (int) i;
        }
    }
    return -1;
}

bool place_target_floor(ExpertSplitPlan & plan,
                        const ExpertSplitConfig & cfg,
                        const std::vector<std::vector<size_t>> & units_by_layer,
                        const std::vector<ExpertSplitUnit> & units,
                        std::vector<uint8_t> & assigned,
                        size_t target_index,
                        std::string * err) {
    const int floor = floor_for_target(cfg, target_index);
    if (floor <= 0) return true;

    for (int layer = 0; layer < cfg.n_layer; ++layer) {
        std::vector<size_t> candidates = units_by_layer[(size_t) layer];
        std::stable_sort(candidates.begin(), candidates.end(),
            [&](size_t lhs, size_t rhs) {
                const ExpertSplitUnit & lu = units[lhs];
                const ExpertSplitUnit & ru = units[rhs];
                const double lv = lu.bytes == 0 ? 0.0 : lu.score / (double) lu.bytes;
                const double rv = ru.bytes == 0 ? 0.0 : ru.score / (double) ru.bytes;
                return better_ranked_unit(lu, lv, ru, rv);
            });

        int placed = 0;
        for (size_t unit_index : candidates) {
            if (placed >= floor) break;
            if (assigned[unit_index]) continue;
            const ExpertSplitUnit & unit = units[unit_index];
            if (unit.bytes == 0) continue;
            if (!can_fit(plan.targets[target_index],
                         plan.target_used_bytes[target_index],
                         unit.bytes)) {
                continue;
            }
            assign_unit(plan, unit, (int) target_index);
            assigned[unit_index] = 1;
            placed++;
        }
        if (placed < floor) {
            if (err) {
                std::ostringstream ss;
                ss << "target " << target_index << " floor could not fit"
                   << " for layer " << layer
                   << " requested=" << floor
                   << " placed=" << placed;
                *err = ss.str();
            }
            return false;
        }
    }

    return true;
}

bool validate_units(const ExpertSplitConfig & cfg,
                    const std::vector<ExpertSplitUnit> & units,
                    std::string * err) {
    if (cfg.n_layer <= 0 || cfg.n_expert <= 0) {
        if (err) *err = "expert split dimensions must be positive";
        return false;
    }

    std::vector<uint8_t> seen((size_t) cfg.n_layer * (size_t) cfg.n_expert, 0);
    for (const ExpertSplitUnit & unit : units) {
        if (unit.layer < 0 || unit.layer >= cfg.n_layer ||
            unit.expert < 0 || unit.expert >= cfg.n_expert) {
            if (err) *err = "expert split unit out of range";
            return false;
        }
        const size_t idx =
            (size_t) unit.layer * (size_t) cfg.n_expert + (size_t) unit.expert;
        if (seen[idx]) {
            if (err) *err = "duplicate expert split unit";
            return false;
        }
        seen[idx] = 1;
    }

    if (cfg.require_full_grid) {
        const size_t expected = (size_t) cfg.n_layer * (size_t) cfg.n_expert;
        if (units.size() != expected) {
            if (err) *err = "expert split units do not cover full layer/expert grid";
            return false;
        }
        if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
            if (err) *err = "expert split units missing from full layer/expert grid";
            return false;
        }
    }

    return true;
}

bool validate_targets(const std::vector<ExpertSplitTarget> & targets,
                      std::string * err) {
    for (const ExpertSplitTarget & target : targets) {
        if (!target.unlimited && target.capacity_bytes < target.reserved_bytes) {
            if (err) *err = "expert split target reserved bytes exceed capacity";
            return false;
        }
    }
    return true;
}

}  // namespace

uint64_t ExpertSplitTarget::usable_bytes() const {
    return dflash::common::usable_bytes(*this);
}

bool ExpertSplitPlan::matches(int n_layer_, int n_expert_) const {
    return n_layer == n_layer_ && n_expert == n_expert_ &&
           (int) assignments.size() == n_layer * n_expert;
}

int ExpertSplitPlan::index(int layer, int expert) const {
    if (layer < 0 || layer >= n_layer || expert < 0 || expert >= n_expert) {
        return -1;
    }
    return layer * n_expert + expert;
}

const ExpertSplitAssignment & ExpertSplitPlan::at(int layer, int expert) const {
    return assignments[(size_t) index(layer, expert)];
}

ExpertSplitAssignment & ExpertSplitPlan::at(int layer, int expert) {
    return assignments[(size_t) index(layer, expert)];
}

int ExpertSplitPlan::count_on_target(int target_index) const {
    int out = 0;
    for (const ExpertSplitAssignment & assignment : assignments) {
        if (assignment.target_index == target_index) out++;
    }
    return out;
}

int ExpertSplitPlan::layer_count_on_target(int layer, int target_index) const {
    if (layer < 0 || layer >= n_layer) return 0;
    int out = 0;
    for (int expert = 0; expert < n_expert; ++expert) {
        if (at(layer, expert).target_index == target_index) out++;
    }
    return out;
}

bool build_expert_split_plan(const ExpertSplitConfig & cfg,
                             const std::vector<ExpertSplitTarget> & targets,
                             const std::vector<ExpertSplitUnit> & units,
                             ExpertSplitPlan & out,
                             std::string * err) {
    if (!validate_units(cfg, units, err) ||
        !validate_targets(targets, err)) {
        return false;
    }

    std::vector<ExpertSplitTarget> plan_targets = targets;
    const bool has_cpu_fallback = std::any_of(plan_targets.begin(), plan_targets.end(),
        [](const ExpertSplitTarget & target) {
            return target.backend == "cpu";
        });
    if (!has_cpu_fallback && cfg.allow_implicit_cpu_fallback) {
        plan_targets.push_back({"cpu", "cpu", -1, 0, 0, true});
    }
    if (plan_targets.empty()) {
        if (err) *err = "expert split requires at least one target";
        return false;
    }

    ExpertSplitPlan plan;
    plan.n_layer = cfg.n_layer;
    plan.n_expert = cfg.n_expert;
    plan.targets = std::move(plan_targets);
    plan.target_used_bytes.assign(plan.targets.size(), 0);
    plan.assignments.assign((size_t) cfg.n_layer * (size_t) cfg.n_expert, {});

    std::vector<ExpertSplitUnit> sorted_units = units;
    std::stable_sort(sorted_units.begin(), sorted_units.end(),
        [](const ExpertSplitUnit & lhs, const ExpertSplitUnit & rhs) {
            if (lhs.layer != rhs.layer) return lhs.layer < rhs.layer;
            return lhs.expert < rhs.expert;
        });

    std::vector<std::vector<size_t>> units_by_layer((size_t) cfg.n_layer);
    for (size_t i = 0; i < sorted_units.size(); ++i) {
        units_by_layer[(size_t) sorted_units[i].layer].push_back(i);
    }
    std::vector<uint8_t> assigned(sorted_units.size(), 0);

    for (size_t target_index = 0; target_index < plan.targets.size(); ++target_index) {
        if (!place_target_floor(plan, cfg, units_by_layer, sorted_units, assigned,
                                target_index, err)) {
            return false;
        }
    }

    std::vector<RankedUnitRef> ranked;
    ranked.reserve(sorted_units.size());
    for (size_t i = 0; i < sorted_units.size(); ++i) {
        if (assigned[i]) continue;
        const ExpertSplitUnit & unit = sorted_units[i];
        if (unit.bytes == 0) continue;
        ranked.push_back({i, unit.score / (double) unit.bytes});
    }
    std::stable_sort(ranked.begin(), ranked.end(),
        [&](const RankedUnitRef & lhs, const RankedUnitRef & rhs) {
            return better_ranked_unit(sorted_units[lhs.unit_index], lhs.value,
                                      sorted_units[rhs.unit_index], rhs.value);
        });

    for (const RankedUnitRef & ref : ranked) {
        if (assigned[ref.unit_index]) continue;
        const ExpertSplitUnit & unit = sorted_units[ref.unit_index];
        const int target_index = choose_first_fit_target(plan, unit);
        if (target_index < 0) {
            if (err) {
                std::ostringstream ss;
                ss << "expert split capacity exhausted while placing layer "
                   << unit.layer << " expert " << unit.expert
                   << " bytes=" << unit.bytes;
                *err = ss.str();
            }
            return false;
        }
        assign_unit(plan, unit, target_index);
        assigned[ref.unit_index] = 1;
    }

    out = std::move(plan);
    return true;
}

}  // namespace dflash::common
