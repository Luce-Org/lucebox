#include "qwen35_tensor_parallel.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace dflash::common {
namespace {

using SplitState = ggml_backend_meta_split_state;
using SplitAxis = ggml_backend_meta_split_axis;

struct TensorSplit {
    SplitAxis axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    int layer = 0;
    std::vector<std::pair<int64_t, uint32_t>> segments;
    std::vector<int64_t> granularities;
};

bool starts_with(const std::string & value, const char * prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string & value, const char * suffix) {
    const std::size_t n = std::strlen(suffix);
    return value.size() >= n && value.compare(value.size() - n, n, suffix) == 0;
}

int parse_layer(const std::string & name) {
    if (starts_with(name, "blk.")) {
        const std::size_t end = name.find('.', 4);
        return end == std::string::npos ? 0 : std::atoi(name.substr(4, end - 4).c_str());
    }
    const std::size_t end = name.find_last_not_of("0123456789");
    return end == std::string::npos || end + 1 == name.size()
        ? 0
        : std::atoi(name.c_str() + end + 1);
}

bool is_delta_layer(const TargetWeights & w, int layer) {
    return ((layer + 1) % w.full_attention_interval) != 0;
}

int effective_layer(const TargetWeights & w, int layer) {
    int result = 0;
    const bool delta = is_delta_layer(w, layer);
    for (int i = 0; i < layer; ++i) {
        result += is_delta_layer(w, i) == delta;
    }
    return result;
}

SplitState mirrored_state() {
    SplitState state{};
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    state.nr[0] = 1;
    state.n_segments = 1;
    return state;
}

TensorSplit classify_tensor(const ggml_tensor * tensor,
                            const TargetWeights & w) {
    const std::string name = tensor->name;
    TensorSplit result;
    result.layer = parse_layer(name);

    auto one_segment = [&](SplitAxis axis, int64_t granularity) {
        result.axis = axis;
        result.segments = {{tensor->ne[axis], 1}};
        result.granularities = {granularity};
    };

    const int64_t block = ggml_blck_size(tensor->type);
    const int64_t perf_block = std::lcm<int64_t>(block, 128);
    const int64_t n_gqa = w.n_head / w.n_head_kv;
    const int64_t group_q_dim = n_gqa * w.n_embd_head_k;
    const int64_t q_granularity = std::lcm<int64_t>(group_q_dim, perf_block);
    const int64_t kv_granularity = q_granularity / n_gqa;
    const int64_t key_dim = (int64_t) w.ssm_d_state * w.ssm_n_group;
    const int64_t value_head_ratio = w.ssm_dt_rank / w.ssm_n_group;

    // Lucebox builds ARGMAX into the target graph. Keep the LM head mirrored
    // until the meta backend has a distributed max-with-index reduction.
    if (ends_with(name, ".attn_q.weight")) {
        one_segment(GGML_BACKEND_SPLIT_AXIS_1,
                    std::lcm<int64_t>(2 * group_q_dim, perf_block));
    } else if (ends_with(name, ".attn_k.weight") ||
               ends_with(name, ".attn_v.weight")) {
        one_segment(GGML_BACKEND_SPLIT_AXIS_1, kv_granularity);
    } else if (ends_with(name, ".attn_output.weight")) {
        one_segment(GGML_BACKEND_SPLIT_AXIS_0, q_granularity);
    } else if (ends_with(name, ".attn_qkv.weight") ||
               ends_with(name, ".ssm_conv1d.weight")) {
        result.axis = GGML_BACKEND_SPLIT_AXIS_1;
        result.segments = {{key_dim, (uint32_t) (2 + value_head_ratio)}};
        result.granularities = {std::lcm<int64_t>(perf_block, w.ssm_d_state)};
    } else if (ends_with(name, ".attn_gate.weight")) {
        result.axis = GGML_BACKEND_SPLIT_AXIS_1;
        result.segments = {{key_dim, (uint32_t) value_head_ratio}};
        result.granularities = {std::lcm<int64_t>(perf_block, w.ssm_d_state)};
    } else if (ends_with(name, ".ssm_out.weight")) {
        result.axis = GGML_BACKEND_SPLIT_AXIS_0;
        result.segments = {{key_dim, (uint32_t) value_head_ratio}};
        result.granularities = {std::lcm<int64_t>(perf_block, w.ssm_d_state)};
    } else if (ends_with(name, ".ssm_alpha.weight") ||
               ends_with(name, ".ssm_beta.weight")) {
        result.axis = GGML_BACKEND_SPLIT_AXIS_1;
        result.segments = {{w.ssm_n_group, (uint32_t) value_head_ratio}};
        result.granularities = {1};
    } else if (ends_with(name, ".ssm_a") || ends_with(name, ".ssm_dt.bias")) {
        result.axis = GGML_BACKEND_SPLIT_AXIS_0;
        result.segments = {{w.ssm_n_group, (uint32_t) value_head_ratio}};
        result.granularities = {1};
    } else if (ends_with(name, ".ffn_gate.weight") ||
               ends_with(name, ".ffn_up.weight")) {
        one_segment(GGML_BACKEND_SPLIT_AXIS_1, perf_block);
    } else if (ends_with(name, ".ffn_down.weight")) {
        one_segment(GGML_BACKEND_SPLIT_AXIS_0, perf_block);
    } else if (starts_with(name, "cache_k_") || starts_with(name, "cache_v_") ||
               starts_with(name, "snap_cache_k_") || starts_with(name, "snap_cache_v_")) {
        one_segment(GGML_BACKEND_SPLIT_AXIS_2, 1);
    } else if (starts_with(name, "ssm_state_") ||
               starts_with(name, "ssm_state_snap_") ||
               starts_with(name, "ssm_intermediate_") ||
               starts_with(name, "snap_ssm_state_")) {
        result.axis = GGML_BACKEND_SPLIT_AXIS_2;
        result.segments = {{w.ssm_n_group, (uint32_t) value_head_ratio}};
        result.granularities = {1};
    } else if (starts_with(name, "conv_state_") ||
               starts_with(name, "conv_state_snap_") ||
               starts_with(name, "conv_input_cache_") ||
               starts_with(name, "snap_conv_state_")) {
        result.axis = GGML_BACKEND_SPLIT_AXIS_1;
        result.segments = {{key_dim, (uint32_t) (2 + value_head_ratio)}};
        result.granularities = {w.ssm_d_state};
    } else if (name == "q_cap") {
        one_segment(GGML_BACKEND_SPLIT_AXIS_1, n_gqa);
    }

    return result;
}

SplitState build_split_state(const ggml_tensor * tensor,
                             const TargetWeights & w,
                             std::size_t n_devices) {
    const TensorSplit split = classify_tensor(tensor, w);
    if (split.axis < 0 || split.axis >= GGML_MAX_DIMS) {
        return mirrored_state();
    }

    SplitState state{};
    state.axis = split.axis;
    state.n_segments = (uint32_t) split.segments.size();
    const std::size_t rotation =
        (std::size_t) effective_layer(w, split.layer) % n_devices;

    int64_t expected = 0;
    for (std::size_t s = 0; s < split.segments.size(); ++s) {
        const int64_t segment_size = split.segments[s].first;
        const uint32_t repeat = split.segments[s].second;
        const int64_t granularity = split.granularities[s];
        int64_t low = 0;
        for (std::size_t j = 0; j < n_devices; ++j) {
            int64_t high = j + 1 == n_devices
                ? segment_size
                : segment_size * (int64_t) (j + 1) / (int64_t) n_devices;
            high -= high % granularity;
            state.ne[s * n_devices + (j + rotation) % n_devices] = high - low;
            low = high;
        }
        state.nr[s] = repeat;
        expected += segment_size * repeat;
    }

    if (expected != tensor->ne[split.axis]) {
        std::fprintf(stderr,
            "[tp] split layout mismatch tensor=%s axis=%d expected=%lld actual=%lld\n",
            tensor->name, (int) split.axis,
            (long long) expected, (long long) tensor->ne[split.axis]);
        return mirrored_state();
    }
    return state;
}

}  // namespace

ggml_backend_meta_split_state qwen35_tensor_parallel_split_state(
        const ggml_tensor * tensor,
        const TargetWeights & weights,
        std::size_t device_count) {
    return build_split_state(tensor, weights, device_count);
}

Qwen35TensorParallelContext::Qwen35TensorParallelContext(
        TargetWeights & weights,
        std::vector<ggml_backend_dev_t> devices)
    : weights_(&weights), devices_(std::move(devices)) {
    meta_device_ = ggml_backend_meta_device(
        devices_.data(), devices_.size(), split_state, this);
}

std::unique_ptr<Qwen35TensorParallelContext>
Qwen35TensorParallelContext::create(const DevicePlacement & placement,
                                    TargetWeights & weights) {
#if !defined(DFLASH27B_BACKEND_CUDA)
    (void) placement;
    (void) weights;
    std::fprintf(stderr, "[tp] tensor parallelism requires a CUDA build\n");
    return nullptr;
#else
    ggml_backend_reg_t cuda_reg = ggml_backend_cuda_reg();
    if (!cuda_reg) {
        std::fprintf(stderr, "[tp] CUDA backend registry is unavailable\n");
        return nullptr;
    }

    std::vector<ggml_backend_dev_t> devices;
    devices.reserve(placement.layer_split_gpus.size());
    const std::size_t available = ggml_backend_reg_dev_count(cuda_reg);
    for (int gpu : placement.layer_split_gpus) {
        if (gpu < 0 || (std::size_t) gpu >= available) {
            std::fprintf(stderr, "[tp] CUDA device %d is unavailable\n", gpu);
            return nullptr;
        }
        ggml_backend_dev_t device = ggml_backend_reg_dev_get(cuda_reg, (std::size_t) gpu);
        if (!device) {
            std::fprintf(stderr, "[tp] failed to resolve CUDA device %d\n", gpu);
            return nullptr;
        }
        devices.push_back(device);
    }

    auto result = std::unique_ptr<Qwen35TensorParallelContext>(
        new Qwen35TensorParallelContext(weights, std::move(devices)));
    if (!result->meta_device_) {
        std::fprintf(stderr, "[tp] failed to create GGML meta device\n");
        return nullptr;
    }
    return result;
#endif
}

ggml_backend_t Qwen35TensorParallelContext::init_backend() const {
    return meta_device_ ? ggml_backend_dev_init(meta_device_, nullptr) : nullptr;
}

ggml_backend_meta_split_state Qwen35TensorParallelContext::split_state(
        const ggml_tensor * tensor,
        void * userdata) {
    auto * context = static_cast<Qwen35TensorParallelContext *>(userdata);
    if (!context || !context->weights_ || context->devices_.empty()) {
        return mirrored_state();
    }
    return qwen35_tensor_parallel_split_state(
        tensor, *context->weights_, context->devices_.size());
}

}  // namespace dflash::common
