#pragma once

#include "tool_speculation.h"

#include <memory>
#include <string>

namespace dflash::common {

// Benchmark-only trusted executor used to qualify same-process HIP sharing.
// It accepts the allowlisted `benchmark_hip_sgemm` tool with
// {"iterations": N}. The matrix size is fixed at startup so allocation and
// warmup stay outside measured requests.
std::shared_ptr<ToolSpeculationExecutor>
create_hip_sgemm_tool_speculation_executor(
    int device,
    int matrix_size,
    int & total_compute_units,
    std::string & error);

}  // namespace dflash::common
