#include "heterogeneous_stage_planner.h"

#include <algorithm>
#include <cmath>

namespace dflash::common {

namespace {

HeterogeneousStagePlan unsplit_plan(int total_width, int alignment) {
    HeterogeneousStagePlan plan;
    plan.total_width = std::max(0, total_width);
    plan.main_width = plan.total_width;
    plan.alignment = std::max(1, alignment);
    return plan;
}

}  // namespace

HeterogeneousStagePlan plan_heterogeneous_stage_width(
        int total_width,
        int alignment,
        double peer_fraction) {
    alignment = std::max(1, alignment);
    HeterogeneousStagePlan plan = unsplit_plan(total_width, alignment);
    if (total_width <= 0 || total_width % alignment != 0 ||
        !std::isfinite(peer_fraction) || peer_fraction < 0.0 ||
        peer_fraction > 1.0) {
        return plan;
    }

    if (peer_fraction == 0.0) return plan;
    if (peer_fraction == 1.0) {
        plan.main_width = 0;
        plan.peer_width = total_width;
        plan.peer_fraction = 1.0;
        return plan;
    }

    const double requested_units =
        peer_fraction * (double) total_width / (double) alignment;
    const int total_units = total_width / alignment;
    if (total_units < 2) return plan;
    int peer_units = std::clamp(
        (int) std::llround(requested_units), 1, total_units - 1);

    plan.peer_width = peer_units * alignment;
    plan.main_width = total_width - plan.peer_width;
    plan.peer_fraction =
        (double) plan.peer_width / (double) plan.total_width;
    return plan;
}

HeterogeneousStagePlan plan_balanced_heterogeneous_stage_width(
        int total_width,
        int alignment,
        double main_rate,
        double peer_rate,
        double main_fixed_work,
        double peer_fixed_work) {
    if (total_width <= 0 || alignment <= 0 ||
        total_width % alignment != 0 ||
        !std::isfinite(main_rate) || !std::isfinite(peer_rate) ||
        !std::isfinite(main_fixed_work) || !std::isfinite(peer_fixed_work) ||
        main_rate <= 0.0 || peer_rate <= 0.0 ||
        main_fixed_work < 0.0 || peer_fixed_work < 0.0) {
        return unsplit_plan(total_width, alignment);
    }

    // Solve (main_fixed + total - peer) / main_rate ==
    //       (peer_fixed + peer) / peer_rate.
    const double peer_work = std::clamp(
        (peer_rate * (main_fixed_work + (double) total_width) -
         main_rate * peer_fixed_work) /
        (main_rate + peer_rate), 0.0, (double) total_width);
    return plan_heterogeneous_stage_width(
        total_width, alignment, peer_work / (double) total_width);
}

}  // namespace dflash::common
