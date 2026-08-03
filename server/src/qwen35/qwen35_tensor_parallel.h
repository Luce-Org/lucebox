#pragma once

#include "internal.h"
#include "placement/placement_config.h"

#include "ggml-backend.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace dflash::common {

ggml_backend_meta_split_state qwen35_tensor_parallel_split_state(
    const ggml_tensor * tensor,
    const TargetWeights & weights,
    std::size_t device_count);

class Qwen35TensorParallelContext {
public:
    static std::unique_ptr<Qwen35TensorParallelContext> create(
        const DevicePlacement & placement,
        TargetWeights & weights);

    ggml_backend_t init_backend() const;
    std::size_t device_count() const { return devices_.size(); }

private:
    Qwen35TensorParallelContext(TargetWeights & weights,
                                std::vector<ggml_backend_dev_t> devices);

    static ggml_backend_meta_split_state split_state(
        const ggml_tensor * tensor,
        void * userdata);

    TargetWeights * weights_ = nullptr;
    std::vector<ggml_backend_dev_t> devices_;
    ggml_backend_dev_t meta_device_ = nullptr;
};

}  // namespace dflash::common
