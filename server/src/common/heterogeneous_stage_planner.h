// Generic work splitter for two heterogeneous execution owners.
//
// The planner is deliberately independent of ggml, model architecture, and
// device vendor.  Callers describe a splittable width and the granularity
// required by their kernels or quantization format.

#pragma once

namespace dflash::common {

struct HeterogeneousStagePlan {
    int total_width = 0;
    int main_width = 0;
    int peer_width = 0;
    int alignment = 1;
    double peer_fraction = 0.0;

    bool split() const {
        return total_width > 0 && main_width > 0 && peer_width > 0 &&
               main_width + peer_width == total_width;
    }

    bool valid() const {
        return total_width > 0 && main_width >= 0 && peer_width >= 0 &&
               main_width + peer_width == total_width;
    }

    bool uses_peer() const {
        return valid() && peer_width > 0;
    }
};

// Plan an explicit partition. The result is rounded to the closest legal peer
// width. Fractions 0 and 1 assign the whole stage to one owner; fractions in
// between leave at least one aligned unit on each. An invalid request returns
// an unsplit plan owned entirely by `main`.
HeterogeneousStagePlan plan_heterogeneous_stage_width(
    int total_width,
    int alignment,
    double peer_fraction);

// Balance a stage from owner throughput and already-scheduled fixed work.
// Work and rates may use any consistent unit (bytes and bytes/us, FLOPs and
// FLOPs/us, or calibrated route-equivalents).  The selected split minimizes
// the estimated maximum of the two owner completion times before alignment.
HeterogeneousStagePlan plan_balanced_heterogeneous_stage_width(
    int total_width,
    int alignment,
    double main_rate,
    double peer_rate,
    double main_fixed_work = 0.0,
    double peer_fixed_work = 0.0);

}  // namespace dflash::common
