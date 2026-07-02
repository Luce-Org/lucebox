// Compatibility shim for the old cold-FFN name.
//
// New code should include moe_expert_compute.h directly. The aliases keep
// existing call sites and downstream branches buildable while the compute
// abstraction moves from "cold expert fallback" to neutral MoE expert compute.
#pragma once

#include "moe_expert_compute.h"

namespace dflash::common {

using ColdFfnLayer = MoeExpertLayer;
using ColdFfnCompute = MoeExpertCompute;

inline std::unique_ptr<ColdFfnCompute> make_cpu_cold_ffn_compute(int n_ff_max) {
    return make_cpu_moe_expert_compute(n_ff_max);
}

}  // namespace dflash::common
