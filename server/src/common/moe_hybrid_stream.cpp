#include "moe_hybrid_stream.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace dflash::common {
namespace {

// Allocate host staging through the selected ggml backend module. This is
// essential in a mixed HIP+CUDA process: memory pinned by the linked HIP
// runtime is not CUDA-pinned memory for a dynamically loaded CUDA backend.
// Keeping allocation behind the backend device also makes the SSD scheduler
// independent of which GPU vendor owns a route partition.
class BackendHostAllocator {
public:
    bool init(ggml_backend_t backend, std::string * err) {
        const ggml_backend_dev_t device = backend
            ? ggml_backend_get_device(backend) : nullptr;
        buffer_type_ = device
            ? ggml_backend_dev_host_buffer_type(device) : nullptr;
        if (!buffer_type_) buffer_type_ = ggml_backend_cpu_buffer_type();
        if (!buffer_type_) {
            if (err) *err = "stream backend has no host staging buffer type";
            return false;
        }
        return true;
    }

    bool allocate(void ** ptr, size_t bytes) {
        if (!ptr || bytes == 0 || !buffer_type_) return false;
        *ptr = nullptr;
        ggml_backend_buffer_t buffer =
            ggml_backend_buft_alloc_buffer(buffer_type_, bytes);
        if (!buffer) return false;
        void * base = ggml_backend_buffer_get_base(buffer);
        if (!base) {
            ggml_backend_buffer_free(buffer);
            return false;
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto [_, inserted] = buffers_.emplace(base, buffer);
            if (!inserted) {
                ggml_backend_buffer_free(buffer);
                return false;
            }
        } catch (const std::bad_alloc &) {
            ggml_backend_buffer_free(buffer);
            return false;
        }
        *ptr = base;
        return true;
    }

    void release(void * ptr) {
        if (!ptr) return;
        ggml_backend_buffer_t buffer = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = buffers_.find(ptr);
            if (found == buffers_.end()) return;
            buffer = found->second;
            buffers_.erase(found);
        }
        ggml_backend_buffer_free(buffer);
    }

    static bool allocate_callback(void ** ptr, size_t bytes, void * opaque) {
        return opaque && static_cast<BackendHostAllocator *>(opaque)->allocate(
            ptr, bytes);
    }

    static void free_callback(void * ptr, void * opaque) {
        if (opaque) static_cast<BackendHostAllocator *>(opaque)->release(ptr);
    }

    ~BackendHostAllocator() {
        for (;;) {
            ggml_backend_buffer_t buffer = nullptr;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (buffers_.empty()) break;
                const auto found = buffers_.begin();
                buffer = found->second;
                buffers_.erase(found);
            }
            ggml_backend_buffer_free(buffer);
        }
    }

private:
    ggml_backend_buffer_type_t buffer_type_ = nullptr;
    std::mutex mutex_;
    std::unordered_map<void *, ggml_backend_buffer_t> buffers_;
};

int env_bounded_int(const char * name, int fallback, int lo, int hi) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < lo || parsed > hi) return fallback;
    return (int) parsed;
}

size_t env_mib(const char * name, size_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0] || value[0] == '-') return fallback;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    constexpr size_t kMiB = 1024 * 1024;
    if (end == value || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max() / kMiB) {
        return fallback;
    }
    return (size_t) parsed * kMiB;
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

bool checked_mul_size(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool valid_ggml_type(ggml_type type) {
    return type >= 0 && type < GGML_TYPE_COUNT;
}

bool same_stream_spec(const MoeStreamExpertSpec & a,
                      const MoeStreamExpertSpec & b) {
    return a.input_dim == b.input_dim &&
           a.intermediate_dim == b.intermediate_dim &&
           a.output_dim == b.output_dim &&
           a.gate_type == b.gate_type &&
           a.up_type == b.up_type &&
           a.down_type == b.down_type &&
           a.gate_up_type == b.gate_up_type &&
           a.fused_gate_up == b.fused_gate_up &&
           a.gated_activation == b.gated_activation &&
           a.swiglu_clamp == b.swiglu_clamp &&
           a.situ_beta == b.situ_beta &&
           a.situ_linear_beta == b.situ_linear_beta &&
           a.gate_scale == b.gate_scale &&
           a.up_scale == b.up_scale &&
           a.down_scale == b.down_scale &&
           a.gate_up_scale == b.gate_up_scale;
}

uint64_t device_key(int layer, int expert) {
    return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
}

} // namespace

MoeStreamConfig MoeStreamConfig::from_env() {
    MoeStreamConfig config;
    config.nvme = MoeNvmeConfig::from_env(config.nvme);
    config.device_slots = env_bounded_int(
        "DFLASH_MOE_NVME_DEVICE_SLOTS", config.device_slots, 2, 8);
    config.graph_cache_entries = env_bounded_int(
        "DFLASH_MOE_NVME_GRAPH_CACHE", config.graph_cache_entries, 0, 64);
    config.fused_decode = env_bounded_int(
        "DFLASH_MOE_NVME_FUSED_DECODE", config.fused_decode ? 1 : 0, 0, 1) != 0;
    config.cache_first_decode = env_bounded_int(
        "DFLASH_MOE_NVME_CACHE_FIRST",
        config.cache_first_decode ? 1 : 0, 0, 1) != 0;
    config.device_cache_bytes = env_mib(
        "DFLASH_MOE_NVME_DEVICE_CACHE_MB", config.device_cache_bytes);
    config.prefill_threshold = env_bounded_int(
        "DFLASH_MOE_NVME_PREFILL_THRESHOLD", config.prefill_threshold, 1, 4096);
    return config;
}

MoeStreamDualOwnerPolicy MoeStreamDualOwnerPolicy::from_env() {
    MoeStreamDualOwnerPolicy policy;
    policy.primary_share_per_mille = env_bounded_int(
        "DFLASH_MOE_PRIMARY_SHARE_PER_MILLE",
        policy.primary_share_per_mille, 0, 1000);
    return policy;
}

bool make_moe_stream_expert_spec(
        const MoeHybridConfig & cfg,
        const MoeLayerDesc & desc,
        const LayerExpertRegions & regions,
        MoeStreamExpertSpec & out,
        std::string * err) {
    out = {};
    const int expert_dim = cfg.expert_embd();
    if (expert_dim <= 0 || cfg.n_ff_exp <= 0) {
        if (err) *err = "streamed expert dimensions must be positive";
        return false;
    }
    out.input_dim = expert_dim;
    out.intermediate_dim = cfg.n_ff_exp;
    out.output_dim = expert_dim;
    out.fused_gate_up = regions.fused_gate_up;
    out.gated_activation = cfg.gated_activation;
    out.swiglu_clamp = cfg.swiglu_clamp;
    out.situ_beta = cfg.situ_beta;
    out.situ_linear_beta = cfg.situ_linear_beta;
    out.gate_scale = desc.ffn_gate_exps_s;
    out.up_scale = desc.ffn_up_exps_s;
    out.down_scale = desc.ffn_down_exps_s;
    out.gate_up_scale = desc.ffn_gate_up_exps_s;

    if (out.fused_gate_up) {
        if (!desc.ffn_gate_up_exps || !desc.ffn_down_exps) {
            if (err) *err = "fused streamed expert is missing gate_up or down metadata";
            return false;
        }
        out.gate_up_type = desc.ffn_gate_up_exps->type;
        out.down_type = desc.ffn_down_exps->type;
    } else {
        if (!desc.ffn_gate_exps || !desc.ffn_up_exps || !desc.ffn_down_exps) {
            if (err) *err = "streamed expert is missing gate, up, or down metadata";
            return false;
        }
        out.gate_type = desc.ffn_gate_exps->type;
        out.up_type = desc.ffn_up_exps->type;
        out.down_type = desc.ffn_down_exps->type;
    }
    return true;
}

bool validate_moe_stream_expert_layout(
        const MoeStreamExpertSpec & spec,
        const MoeExpertIoLayout & layout,
        std::string * err) {
    if (spec.input_dim <= 0 || spec.intermediate_dim <= 0 ||
        spec.output_dim <= 0 || !valid_ggml_type(spec.down_type) ||
        (spec.fused_gate_up
            ? !valid_ggml_type(spec.gate_up_type)
            : (!valid_ggml_type(spec.gate_type) || !valid_ggml_type(spec.up_type)))) {
        if (err) *err = "invalid streamed expert shape or tensor type";
        return false;
    }
    if (spec.swiglu_clamp < 0.0f ||
        !std::isfinite(spec.swiglu_clamp) ||
        !std::isfinite(spec.gate_scale) ||
        !std::isfinite(spec.up_scale) ||
        !std::isfinite(spec.down_scale) ||
        !std::isfinite(spec.gate_up_scale) ||
        (spec.gated_activation != MoeGatedActivation::SwiGlu &&
         spec.gated_activation != MoeGatedActivation::Situ) ||
        (spec.gated_activation == MoeGatedActivation::Situ &&
         (spec.situ_beta <= 0.0f || spec.situ_linear_beta <= 0.0f ||
          !std::isfinite(spec.situ_beta) ||
          !std::isfinite(spec.situ_linear_beta)))) {
        if (err) *err = "invalid streamed expert activation parameters";
        return false;
    }
    if (layout.fused_gate_up != spec.fused_gate_up) {
        if (err) *err = "streamed storage and compute disagree about fused gate/up";
        return false;
    }

    auto expected_bytes = [&](ggml_type type, int64_t columns,
                              int64_t rows, size_t & bytes) -> bool {
        if (!valid_ggml_type(type) || columns <= 0 || rows <= 0) return false;
        const size_t row = ggml_row_size(type, columns);
        return checked_mul_size(row, (size_t) rows, bytes);
    };
    auto require_component = [&](MoeExpertComponentKind kind, size_t expected,
                                 const char * label) -> bool {
        const MoeExpertComponentLayout * component = layout.component(kind);
        if (!component || component->bytes != expected) {
            if (err) {
                *err = std::string("streamed ") + label +
                    " bytes do not match tensor type/shape";
            }
            return false;
        }
        return true;
    };

    size_t down_bytes = 0;
    if (!expected_bytes(spec.down_type, spec.intermediate_dim,
                        spec.output_dim, down_bytes)) {
        if (err) *err = "streamed down tensor size overflow";
        return false;
    }
    if (spec.fused_gate_up) {
        if (spec.intermediate_dim > std::numeric_limits<int>::max() / 2) {
            if (err) *err = "streamed fused intermediate dimension overflow";
            return false;
        }
        size_t gate_up_bytes = 0;
        if (!expected_bytes(spec.gate_up_type, spec.input_dim,
                            2LL * spec.intermediate_dim, gate_up_bytes) ||
            !require_component(MoeExpertComponentKind::FusedGateUp,
                               gate_up_bytes, "gate_up")) {
            return false;
        }
    } else {
        size_t gate_bytes = 0;
        size_t up_bytes = 0;
        if (!expected_bytes(spec.gate_type, spec.input_dim,
                            spec.intermediate_dim, gate_bytes) ||
            !expected_bytes(spec.up_type, spec.input_dim,
                            spec.intermediate_dim, up_bytes) ||
            !require_component(MoeExpertComponentKind::Gate, gate_bytes, "gate") ||
            !require_component(MoeExpertComponentKind::Up, up_bytes, "up")) {
            return false;
        }
    }
    return require_component(MoeExpertComponentKind::Down, down_bytes, "down");
}

namespace {

struct StreamExpertDeviceBinding {
    const void * gate = nullptr;
    const void * up = nullptr;
    const void * down = nullptr;
    size_t gate_alloc_bytes = 0;
    size_t up_alloc_bytes = 0;
    size_t down_alloc_bytes = 0;
};

bool bind_stream_tensor(ggml_backend_buffer_t buffer,
                        ggml_tensor * tensor,
                        const void * data,
                        size_t available_bytes,
                        const char * label,
                        std::string * err) {
    if (!buffer || !tensor || !data) {
        if (err) *err = std::string("invalid streamed ") +
                        label + " tensor binding";
        return false;
    }
    const size_t required_bytes =
        ggml_backend_buffer_get_alloc_size(buffer, tensor);
    if (available_bytes < required_bytes) {
        if (err) *err = std::string("streamed ") + label +
            " device allocation is smaller than the backend's padded "
            "tensor requirement";
        return false;
    }
    const size_t alignment = ggml_backend_buffer_get_alignment(buffer);
    if (alignment != 0 && (uintptr_t) data % alignment != 0) {
        if (err) *err = std::string("streamed ") + label +
                        " tensor is not aligned for the compute backend";
        return false;
    }
    if (ggml_backend_tensor_alloc(
            buffer, tensor, const_cast<void *>(data)) != GGML_STATUS_SUCCESS) {
        if (err) *err = std::string("failed to bind streamed ") + label +
                        " tensor to expert cache";
        return false;
    }
    return true;
}

ggml_tensor * scale_stream_tensor(ggml_context * ctx,
                                  ggml_tensor * value,
                                  float scale) {
    return scale == 1.0f ? value : ggml_scale(ctx, value, scale);
}

ggml_tensor * apply_stream_gated_activation(
        ggml_context * ctx,
        const MoeStreamExpertSpec & spec,
        ggml_tensor * gate,
        ggml_tensor * up) {
    if (spec.gated_activation == MoeGatedActivation::Situ) {
        ggml_tensor * nonlinear = ggml_scale(ctx, gate, 1.0f / spec.situ_beta);
        nonlinear = ggml_tanh(ctx, nonlinear);
        nonlinear = ggml_scale(ctx, nonlinear, spec.situ_beta);
        nonlinear = ggml_mul(ctx, nonlinear, ggml_sigmoid(ctx, gate));
        ggml_tensor * linear = ggml_scale(
            ctx, up, 1.0f / spec.situ_linear_beta);
        linear = ggml_tanh(ctx, linear);
        linear = ggml_scale(ctx, linear, spec.situ_linear_beta);
        return ggml_mul(ctx, nonlinear, linear);
    }
    if (spec.swiglu_clamp > 0.0f) {
        return ggml_swiglu_ds4_split(ctx, gate, up, spec.swiglu_clamp);
    }
    return ggml_swiglu_split(ctx, gate, up);
}

ggml_tensor * build_stream_expert_branch(
        ggml_context * ctx,
        const MoeStreamExpertSpec & spec,
        int batch,
        ggml_tensor * input,
        ggml_tensor * gate,
        ggml_tensor * up,
        ggml_tensor * down,
        ggml_tensor * gate_up) {
    ggml_tensor * activated = nullptr;
    if (gate_up) {
        ggml_tensor * combined = scale_stream_tensor(
            ctx, ggml_mul_mat(ctx, gate_up, input), spec.gate_up_scale);
        ggml_tensor * gate_part = ggml_view_2d(
            ctx, combined, spec.intermediate_dim, batch,
            combined->nb[1], 0);
        ggml_tensor * up_part = ggml_view_2d(
            ctx, combined, spec.intermediate_dim, batch,
            combined->nb[1],
            (size_t) spec.intermediate_dim * sizeof(float));
        activated = apply_stream_gated_activation(
            ctx, spec, ggml_cont(ctx, gate_part), ggml_cont(ctx, up_part));
    } else {
        ggml_tensor * gate_value = scale_stream_tensor(
            ctx, ggml_mul_mat(ctx, gate, input), spec.gate_scale);
        ggml_tensor * up_value = scale_stream_tensor(
            ctx, ggml_mul_mat(ctx, up, input), spec.up_scale);
        activated = apply_stream_gated_activation(
            ctx, spec, gate_value, up_value);
    }
    return scale_stream_tensor(
        ctx, ggml_mul_mat(ctx, down, activated), spec.down_scale);
}

class PersistentStreamExpertGraph {
public:
    ~PersistentStreamExpertGraph() { destroy(); }

    bool matches(const MoeStreamExpertSpec & spec, int batch) const {
        return batch_ == batch && same_stream_spec(spec_, spec);
    }

    bool build(ggml_backend_t backend,
               ggml_backend_buffer_t expert_buffer,
               const MoeStreamExpertSpec & spec,
               int batch,
               const void * gate_data,
               const void * up_data,
               const void * down_data,
               size_t gate_alloc_bytes,
               size_t up_alloc_bytes,
               size_t down_alloc_bytes,
               std::string * err) {
        destroy();
        if (!backend || !expert_buffer || batch <= 0 || !gate_data || !down_data ||
            (!spec.fused_gate_up && !up_data)) {
            if (err) *err = "invalid persistent streamed-expert graph arguments";
            return false;
        }
        backend_ = backend;
        spec_ = spec;
        batch_ = batch;

        ggml_init_params params{};
        params.mem_size = 4 * 1024 * 1024;
        params.no_alloc = true;
        ctx_ = ggml_init(params);
        if (!ctx_) {
            if (err) *err = "ggml_init failed for persistent streamed expert";
            return false;
        }

        input_ = ggml_new_tensor_2d(
            ctx_, GGML_TYPE_F32, spec.input_dim, batch);
        ggml_set_input(input_);
        if (spec.fused_gate_up) {
            gate_up_ = ggml_new_tensor_2d(
                ctx_, spec.gate_up_type, spec.input_dim,
                2LL * spec.intermediate_dim);
            down_ = ggml_new_tensor_2d(
                ctx_, spec.down_type, spec.intermediate_dim, spec.output_dim);
            ggml_set_input(gate_up_);
            ggml_set_input(down_);
        } else {
            gate_ = ggml_new_tensor_2d(
                ctx_, spec.gate_type, spec.input_dim, spec.intermediate_dim);
            up_ = ggml_new_tensor_2d(
                ctx_, spec.up_type, spec.input_dim, spec.intermediate_dim);
            down_ = ggml_new_tensor_2d(
                ctx_, spec.down_type, spec.intermediate_dim, spec.output_dim);
            ggml_set_input(gate_);
            ggml_set_input(up_);
            ggml_set_input(down_);
        }

        if (spec.fused_gate_up) {
            if (!bind_stream_tensor(expert_buffer, gate_up_, gate_data,
                                    gate_alloc_bytes, "gate_up", err) ||
                !bind_stream_tensor(expert_buffer, down_, down_data,
                                    down_alloc_bytes, "down", err)) {
                return false;
            }
        } else if (!bind_stream_tensor(expert_buffer, gate_, gate_data,
                                       gate_alloc_bytes, "gate", err) ||
                   !bind_stream_tensor(expert_buffer, up_, up_data,
                                       up_alloc_bytes, "up", err) ||
                   !bind_stream_tensor(expert_buffer, down_, down_data,
                                       down_alloc_bytes, "down", err)) {
            return false;
        }

        output_ = build_stream_expert_branch(
            ctx_, spec, batch, input_, gate_, up_, down_, gate_up_);
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_, 512, false);
        ggml_build_forward_expand(graph_, output_);
        alloc_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        if (!alloc_ || !ggml_gallocr_alloc_graph(alloc_, graph_)) {
            if (err) *err = "persistent streamed-expert graph allocation failed";
            return false;
        }
        return true;
    }

    bool launch(const void * gate_data,
                const void * up_data,
                const void * down_data,
                const float * input,
                std::string * err) {
        if (!valid() || !gate_data || !down_data || !input ||
            (!spec_.fused_gate_up && !up_data)) {
            if (err) *err = "persistent streamed-expert graph is not ready";
            return false;
        }
        if (gate_up_) gate_up_->data = const_cast<void *>(gate_data);
        else gate_->data = const_cast<void *>(gate_data);
        if (up_) up_->data = const_cast<void *>(up_data);
        down_->data = const_cast<void *>(down_data);
        size_t input_values = 0;
        if (!checked_mul_size((size_t) spec_.input_dim,
                              (size_t) batch_, input_values)) {
            if (err) *err = "streamed expert input size overflow";
            return false;
        }
        ggml_backend_tensor_set(
            input_, input, 0, input_values * sizeof(float));
        if (ggml_backend_graph_compute_async(backend_, graph_) !=
            GGML_STATUS_SUCCESS) {
            if (err) *err = "persistent streamed-expert graph launch failed";
            return false;
        }
        return true;
    }

    bool finish(std::vector<float> & output, std::string * err) {
        if (!valid()) {
            if (err) *err = "persistent streamed-expert graph is not ready";
            return false;
        }
        ggml_backend_synchronize(backend_);
        size_t output_values = 0;
        if (!checked_mul_size((size_t) spec_.output_dim,
                              (size_t) batch_, output_values)) {
            if (err) *err = "streamed expert output size overflow";
            return false;
        }
        output.resize(output_values);
        ggml_backend_tensor_get(
            output_, output.data(), 0, output_values * sizeof(float));
        return true;
    }

    void destroy() {
        if (alloc_) ggml_gallocr_free(alloc_);
        alloc_ = nullptr;
        if (ctx_) ggml_free(ctx_);
        ctx_ = nullptr;
        graph_ = nullptr;
        input_ = gate_ = up_ = down_ = gate_up_ = output_ = nullptr;
        backend_ = nullptr;
        batch_ = 0;
    }

    bool valid() const {
        return backend_ && ctx_ && graph_ && alloc_ && input_ && output_;
    }

    uint64_t last_touch = 0;

private:
    ggml_backend_t backend_ = nullptr;
    MoeStreamExpertSpec spec_{};
    int batch_ = 0;
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t alloc_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * gate_ = nullptr;
    ggml_tensor * up_ = nullptr;
    ggml_tensor * down_ = nullptr;
    ggml_tensor * gate_up_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

// Single-token decode selects several independent experts and then computes a
// weighted sum. Keeping those branches in one persistent graph removes the
// host boundary between experts: one graph submission, one synchronization,
// and one output copy per routed MoE layer.
class PersistentStreamMoEDecodeGraph {
public:
    ~PersistentStreamMoEDecodeGraph() { destroy(); }

    bool matches(const MoeStreamExpertSpec & spec, int expert_count) const {
        return expert_count_ == expert_count && same_stream_spec(spec_, spec);
    }

    bool build(ggml_backend_t backend,
               ggml_backend_buffer_t expert_buffer,
               const MoeStreamExpertSpec & spec,
               const std::vector<StreamExpertDeviceBinding> & bindings,
               std::string * err) {
        destroy();
        if (!backend || !expert_buffer || bindings.size() < 2) {
            if (err) *err = "invalid fused streamed-MoE graph arguments";
            return false;
        }
        backend_ = backend;
        spec_ = spec;
        expert_count_ = (int) bindings.size();

        ggml_init_params params{};
        params.mem_size = 8 * 1024 * 1024;
        params.no_alloc = true;
        ctx_ = ggml_init(params);
        if (!ctx_) {
            if (err) *err = "ggml_init failed for fused streamed-MoE decode";
            return false;
        }

        input_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, spec.input_dim, 1);
        route_weights_ = ggml_new_tensor_1d(
            ctx_, GGML_TYPE_F32, expert_count_);
        ggml_set_input(input_);
        ggml_set_input(route_weights_);

        tensors_.resize(bindings.size());
        expert_outputs_ = ggml_new_tensor_3d(
            ctx_, GGML_TYPE_F32, spec.output_dim, expert_count_, 1);
        copy_nodes_.reserve(bindings.size());
        for (size_t i = 0; i < bindings.size(); ++i) {
            ExpertTensors & tensors = tensors_[i];
            if (spec.fused_gate_up) {
                tensors.gate_up = ggml_new_tensor_2d(
                    ctx_, spec.gate_up_type, spec.input_dim,
                    2LL * spec.intermediate_dim);
                tensors.down = ggml_new_tensor_2d(
                    ctx_, spec.down_type, spec.intermediate_dim,
                    spec.output_dim);
                ggml_set_input(tensors.gate_up);
                ggml_set_input(tensors.down);
                if (!bind_stream_tensor(
                        expert_buffer, tensors.gate_up, bindings[i].gate,
                        bindings[i].gate_alloc_bytes, "gate_up", err) ||
                    !bind_stream_tensor(
                        expert_buffer, tensors.down, bindings[i].down,
                        bindings[i].down_alloc_bytes, "down", err)) {
                    return false;
                }
            } else {
                tensors.gate = ggml_new_tensor_2d(
                    ctx_, spec.gate_type, spec.input_dim,
                    spec.intermediate_dim);
                tensors.up = ggml_new_tensor_2d(
                    ctx_, spec.up_type, spec.input_dim,
                    spec.intermediate_dim);
                tensors.down = ggml_new_tensor_2d(
                    ctx_, spec.down_type, spec.intermediate_dim,
                    spec.output_dim);
                ggml_set_input(tensors.gate);
                ggml_set_input(tensors.up);
                ggml_set_input(tensors.down);
                if (!bind_stream_tensor(
                        expert_buffer, tensors.gate, bindings[i].gate,
                        bindings[i].gate_alloc_bytes, "gate", err) ||
                    !bind_stream_tensor(
                        expert_buffer, tensors.up, bindings[i].up,
                        bindings[i].up_alloc_bytes, "up", err) ||
                    !bind_stream_tensor(
                        expert_buffer, tensors.down, bindings[i].down,
                        bindings[i].down_alloc_bytes, "down", err)) {
                    return false;
                }
            }

            ggml_tensor * branch = build_stream_expert_branch(
                ctx_, spec, 1, input_, tensors.gate, tensors.up,
                tensors.down, tensors.gate_up);
            ggml_tensor * destination = ggml_view_2d(
                ctx_, expert_outputs_, spec.output_dim, 1,
                expert_outputs_->nb[1], i * expert_outputs_->nb[1]);
            ggml_tensor * copy = ggml_cpy(ctx_, branch, destination);
            ggml_set_output(copy);
            copy_nodes_.push_back(copy);
        }

        output_ = ggml_laguna_moe_combine(
            ctx_, expert_outputs_, route_weights_);
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(
            ctx_, std::max<size_t>(512, bindings.size() * 256), false);
        for (ggml_tensor * copy : copy_nodes_) {
            ggml_build_forward_expand(graph_, copy);
        }
        ggml_build_forward_expand(graph_, output_);
        alloc_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        if (!alloc_ || !ggml_gallocr_alloc_graph(alloc_, graph_)) {
            if (err) *err = "fused streamed-MoE graph allocation failed";
            return false;
        }
        return true;
    }

    bool launch(const std::vector<StreamExpertDeviceBinding> & bindings,
                const float * input,
                const float * route_weights,
                std::string * err) {
        if (!valid() || !input || !route_weights ||
            bindings.size() != tensors_.size()) {
            if (err) *err = "fused streamed-MoE graph is not ready";
            return false;
        }
        for (size_t i = 0; i < bindings.size(); ++i) {
            ExpertTensors & tensors = tensors_[i];
            if (tensors.gate_up) {
                tensors.gate_up->data = const_cast<void *>(bindings[i].gate);
            } else {
                tensors.gate->data = const_cast<void *>(bindings[i].gate);
                tensors.up->data = const_cast<void *>(bindings[i].up);
            }
            tensors.down->data = const_cast<void *>(bindings[i].down);
        }
        ggml_backend_tensor_set(
            input_, input, 0, (size_t) spec_.input_dim * sizeof(float));
        ggml_backend_tensor_set(
            route_weights_, route_weights, 0,
            bindings.size() * sizeof(float));
        if (ggml_backend_graph_compute_async(backend_, graph_) !=
            GGML_STATUS_SUCCESS) {
            if (err) *err = "fused streamed-MoE graph launch failed";
            return false;
        }
        return true;
    }

    bool finish(std::vector<float> & output, std::string * err) {
        if (!valid()) {
            if (err) *err = "fused streamed-MoE graph is not ready";
            return false;
        }
        ggml_backend_synchronize(backend_);
        output.resize((size_t) spec_.output_dim);
        ggml_backend_tensor_get(
            output_, output.data(), 0,
            output.size() * sizeof(float));
        return true;
    }

    void destroy() {
        if (alloc_) ggml_gallocr_free(alloc_);
        alloc_ = nullptr;
        if (ctx_) ggml_free(ctx_);
        ctx_ = nullptr;
        graph_ = nullptr;
        input_ = route_weights_ = expert_outputs_ = output_ = nullptr;
        tensors_.clear();
        copy_nodes_.clear();
        backend_ = nullptr;
        expert_count_ = 0;
    }

    bool valid() const {
        return backend_ && ctx_ && graph_ && alloc_ && input_ &&
               route_weights_ && expert_outputs_ && output_;
    }

    uint64_t last_touch = 0;

private:
    struct ExpertTensors {
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        ggml_tensor * gate_up = nullptr;
    };

    ggml_backend_t backend_ = nullptr;
    MoeStreamExpertSpec spec_{};
    int expert_count_ = 0;
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t alloc_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * route_weights_ = nullptr;
    ggml_tensor * expert_outputs_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<ExpertTensors> tensors_;
    std::vector<ggml_tensor *> copy_nodes_;
};

} // namespace

struct MoeHybridStreamEngine::Runtime {
    struct DeviceComponentLayout {
        MoeExpertComponentKind kind = MoeExpertComponentKind::Gate;
        size_t offset = 0;
        size_t logical_bytes = 0;
        size_t alloc_bytes = 0;
    };

    struct DeviceExpertLayout {
        MoeStreamExpertSpec spec{};
        DeviceComponentLayout components[3]{};
        int component_count = 0;
        size_t bytes = 0;
        bool configured = false;

        const DeviceComponentLayout * component(
                MoeExpertComponentKind kind) const {
            for (int i = 0; i < component_count; ++i) {
                if (components[i].kind == kind) return &components[i];
            }
            return nullptr;
        }
    };

    struct DeviceSlot {
        void * data = nullptr;
        ggml_tensor * transfer_tensor = nullptr;
        ggml_backend_event_t ready = nullptr;
        bool pending = false;
        bool valid = false;
        bool cache_managed = false;
        bool pinned = false;
        int compute_users = 0;
        MoeExpertKey key{};
        uint64_t frequency = 0;
        uint64_t last_touch = 0;
        MoeNvmeLease host_lease;
        MoeExpertIoLayout layout{};
        DeviceExpertLayout device_layout{};
    };

    ggml_backend_t backend = nullptr;
    // A second backend instance on the same device owns the upload stream.
    // Its interface comes from the same module as `backend`, so this works for
    // both the linked runtime and an isolated CUDA/HIP peer module.
    ggml_backend_t transfer_backend = nullptr;
    size_t max_expert_bytes = 0;
    MoeStreamConfig config{};
    BackendHostAllocator host_allocator;
    // Quantized CUDA/HIP kernels may read backend-added row padding. Keep one
    // immutable backend-pinned zero source and upload only those padding bytes
    // with each expert. Clearing a capacity-sized cache eagerly can fault tens
    // of GiB of APU managed memory before the first request.
    void * zero_padding = nullptr;
    size_t zero_padding_bytes = 0;
    std::unique_ptr<MoeNvmeScheduler> io;
    ggml_backend_buffer_t device_pool_buffer = nullptr;
    void * device_pool = nullptr;
    ggml_context * device_slot_ctx = nullptr;
    size_t device_stride = 0;
    size_t device_pool_bytes = 0;
    std::vector<DeviceSlot> device_slots;
    std::unordered_map<uint64_t, int> device_index;
    std::unordered_map<int, DeviceExpertLayout> layer_device_layouts;
    uint64_t device_clock = 0;
    uint64_t device_cache_hits = 0;
    uint64_t device_cache_misses = 0;
    uint64_t device_cache_evictions = 0;
    size_t pinned_experts = 0;
    int active_slot = -1;
    std::vector<std::unique_ptr<PersistentStreamExpertGraph>> graph_cache;
    std::vector<std::unique_ptr<PersistentStreamMoEDecodeGraph>>
        fused_decode_graph_cache;
    uint64_t graph_clock = 0;
    MoeStreamComputeStats compute_stats{};
    std::mutex compute_mutex;
};

template <typename RuntimeT>
void release_device_cache(RuntimeT & runtime) {
    if (runtime.backend) ggml_backend_synchronize(runtime.backend);
    if (runtime.transfer_backend) {
        ggml_backend_synchronize(runtime.transfer_backend);
    }
    runtime.graph_cache.clear();
    runtime.fused_decode_graph_cache.clear();
    for (auto & slot : runtime.device_slots) {
        slot.host_lease.reset();
        if (slot.ready) ggml_backend_event_free(slot.ready);
        slot.ready = nullptr;
        slot.data = nullptr;
        slot.transfer_tensor = nullptr;
        slot.pending = false;
    }
    runtime.device_slots.clear();
    runtime.device_index.clear();
    runtime.pinned_experts = 0;
    runtime.active_slot = -1;
    if (runtime.device_pool_buffer) {
        ggml_backend_buffer_free(runtime.device_pool_buffer);
    }
    runtime.device_pool_buffer = nullptr;
    runtime.device_pool = nullptr;
    if (runtime.device_slot_ctx) ggml_free(runtime.device_slot_ctx);
    runtime.device_slot_ctx = nullptr;
    runtime.device_stride = 0;
    runtime.device_pool_bytes = 0;
}

template <typename RuntimeT>
bool ensure_zero_padding(RuntimeT & runtime, size_t required_bytes,
                         std::string * err) {
    if (required_bytes == 0 || required_bytes <= runtime.zero_padding_bytes) {
        return true;
    }
    const size_t allocation_bytes = align_up(required_bytes, 256);
    if (allocation_bytes == 0) {
        if (err) *err = "SSD quantization padding size overflow";
        return false;
    }

    void * replacement = nullptr;
    if (!runtime.host_allocator.allocate(&replacement, allocation_bytes)) {
        if (err) *err = "failed to allocate SSD quantization padding staging";
        return false;
    }
    std::memset(replacement, 0, allocation_bytes);

    // The previous immutable source may still be referenced by queued H2D
    // copies. Growth normally happens once, on the first numerical layout, so
    // synchronize only before replacement and never on the steady-state path.
    if (runtime.zero_padding) {
        ggml_backend_synchronize(runtime.transfer_backend);
        runtime.host_allocator.release(runtime.zero_padding);
    }
    runtime.zero_padding = replacement;
    runtime.zero_padding_bytes = allocation_bytes;
    return true;
}

template <typename RuntimeT>
bool allocate_device_cache(RuntimeT & runtime, std::string * err,
                           size_t minimum_stride = 0) {
    const size_t logical_stride =
        std::max(runtime.max_expert_bytes, minimum_stride);
    runtime.device_stride = align_up(logical_stride, 256);
    if (runtime.device_stride == 0) {
        if (err) *err = "SSD device-cache stride overflow";
        return false;
    }

    size_t desired_slots = (size_t) std::max(2, runtime.config.device_slots);
    if (runtime.config.device_cache_bytes > 0) {
        desired_slots = std::max(
            desired_slots, runtime.config.device_cache_bytes / runtime.device_stride);
    }
    constexpr size_t kMaxDeviceSlots = 65536;
    desired_slots = std::min(desired_slots, kMaxDeviceSlots);
    desired_slots = std::min(
        desired_slots, std::numeric_limits<size_t>::max() / runtime.device_stride);

    // A large contiguous allocation keeps address arithmetic cheap and avoids
    // thousands of allocator objects. If the planner's free-memory snapshot
    // raced another allocation, converge to a smaller usable cache instead of
    // failing model startup.
    size_t attempt_slots = desired_slots;
    ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(runtime.backend);
    while (attempt_slots >= 2) {
        const size_t bytes = attempt_slots * runtime.device_stride;
        runtime.device_pool_buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
        if (runtime.device_pool_buffer) {
            ggml_backend_buffer_set_usage(
                runtime.device_pool_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            runtime.device_pool =
                ggml_backend_buffer_get_base(runtime.device_pool_buffer);
            runtime.device_pool_bytes = bytes;
            break;
        }
        if (attempt_slots == 2) break;
        attempt_slots = std::max<size_t>(2, attempt_slots * 3 / 4);
    }
    if (!runtime.device_pool) {
        if (err) *err = "failed to allocate SSD GPU expert cache";
        return false;
    }

    try {
        runtime.device_slots.resize(attempt_slots);
    } catch (const std::bad_alloc &) {
        ggml_backend_buffer_free(runtime.device_pool_buffer);
        runtime.device_pool_buffer = nullptr;
        runtime.device_pool = nullptr;
        runtime.device_pool_bytes = 0;
        if (err) *err = "failed to allocate SSD GPU cache metadata";
        return false;
    }
    auto * base = static_cast<uint8_t *>(runtime.device_pool);
    if (runtime.device_stride >
            static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        attempt_slots >
        (std::numeric_limits<size_t>::max() - 1024) /
            ggml_tensor_overhead()) {
        if (err) *err = "SSD GPU cache tensor size overflow";
        release_device_cache(runtime);
        return false;
    }
    ggml_init_params params{};
    params.mem_size = attempt_slots * ggml_tensor_overhead() + 1024;
    params.no_alloc = true;
    runtime.device_slot_ctx = ggml_init(params);
    if (!runtime.device_slot_ctx) {
        if (err) *err = "failed to allocate SSD GPU cache tensor metadata";
        release_device_cache(runtime);
        return false;
    }
    for (size_t i = 0; i < runtime.device_slots.size(); ++i) {
        auto & slot = runtime.device_slots[i];
        slot.data = base + i * runtime.device_stride;
        slot.transfer_tensor = ggml_new_tensor_1d(
            runtime.device_slot_ctx, GGML_TYPE_I8,
            (int64_t) runtime.device_stride);
        if (!slot.transfer_tensor ||
            ggml_backend_buffer_get_alloc_size(
                runtime.device_pool_buffer, slot.transfer_tensor) >
                runtime.device_stride ||
            ggml_backend_tensor_alloc(
                runtime.device_pool_buffer, slot.transfer_tensor,
                slot.data) != GGML_STATUS_SUCCESS) {
            if (err) *err = "failed to bind SSD GPU cache transfer tensor";
            release_device_cache(runtime);
            return false;
        }
    }
    return true;
}

template <typename RuntimeT>
bool build_device_expert_layout(
        RuntimeT & runtime,
        const MoeStreamExpertSpec & spec,
        typename RuntimeT::DeviceExpertLayout & out,
        std::string * err) {
    out = typename RuntimeT::DeviceExpertLayout{};
    if (spec.input_dim <= 0 || spec.intermediate_dim <= 0 ||
        spec.output_dim <= 0 || !valid_ggml_type(spec.down_type) ||
        (spec.fused_gate_up
            ? !valid_ggml_type(spec.gate_up_type)
            : (!valid_ggml_type(spec.gate_type) ||
               !valid_ggml_type(spec.up_type)))) {
        if (err) *err = "invalid streamed expert shape or tensor type";
        return false;
    }

    ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(runtime.backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        if (err) *err = "streamed expert backend alignment is invalid";
        return false;
    }

    ggml_init_params params{};
    params.mem_size = 128 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        if (err) *err = "ggml_init failed for streamed device layout";
        return false;
    }

    size_t cursor = 0;
    auto add = [&](MoeExpertComponentKind kind, ggml_type type,
                   int64_t columns, int64_t rows, const char * label) -> bool {
        if (out.component_count >= 3 || columns <= 0 || rows <= 0 ||
            columns % ggml_blck_size(type) != 0) {
            if (err) *err = std::string("invalid streamed ") + label +
                            " tensor dimensions";
            return false;
        }
        ggml_tensor * tensor =
            ggml_new_tensor_2d(ctx, type, columns, rows);
        const size_t logical_bytes = ggml_nbytes(tensor);
        const size_t alloc_bytes =
            ggml_backend_buft_get_alloc_size(buft, tensor);
        const size_t offset = align_up(cursor, alignment);
        if (offset == 0 && cursor != 0) {
            if (err) *err = "streamed device component alignment overflow";
            return false;
        }
        if (alloc_bytes < logical_bytes ||
            offset > std::numeric_limits<size_t>::max() - alloc_bytes) {
            if (err) *err = "streamed device component size overflow";
            return false;
        }
        out.components[out.component_count++] = {
            kind, offset, logical_bytes, alloc_bytes};
        cursor = offset + alloc_bytes;
        return true;
    };

    bool ok = true;
    if (spec.fused_gate_up) {
        ok = spec.intermediate_dim <= std::numeric_limits<int>::max() / 2 &&
             add(MoeExpertComponentKind::FusedGateUp, spec.gate_up_type,
                 spec.input_dim, 2LL * spec.intermediate_dim, "gate_up") &&
             add(MoeExpertComponentKind::Down, spec.down_type,
                 spec.intermediate_dim, spec.output_dim, "down");
    } else {
        ok = add(MoeExpertComponentKind::Gate, spec.gate_type,
                 spec.input_dim, spec.intermediate_dim, "gate") &&
             add(MoeExpertComponentKind::Up, spec.up_type,
                 spec.input_dim, spec.intermediate_dim, "up") &&
             add(MoeExpertComponentKind::Down, spec.down_type,
                 spec.intermediate_dim, spec.output_dim, "down");
    }
    if (ok) {
        out.bytes = align_up(cursor, std::max<size_t>(256, alignment));
        if (out.bytes == 0) {
            if (err) *err = "streamed device expert stride overflow";
            ok = false;
        }
    }
    ggml_free(ctx);
    if (!ok) return false;
    out.spec = spec;
    out.configured = true;
    return true;
}

template <typename RuntimeT>
bool prepare_device_expert_layout(RuntimeT & runtime, int layer,
                                  const MoeStreamExpertSpec & spec,
                                  std::string * err) {
    auto existing = runtime.layer_device_layouts.find(layer);
    if (existing != runtime.layer_device_layouts.end()) {
        if (!same_stream_spec(existing->second.spec, spec)) {
            if (err) *err = "streamed expert specification changed within one layer";
            return false;
        }
        return true;
    }

    typename RuntimeT::DeviceExpertLayout layout;
    if (!build_device_expert_layout(runtime, spec, layout, err)) return false;
    const bool has_compact_cached_slot = std::any_of(
        runtime.device_slots.begin(), runtime.device_slots.end(),
        [&](const auto & slot) {
            return slot.valid && slot.key.layer == layer &&
                   !slot.device_layout.configured;
        });
    if (layout.bytes > runtime.device_stride || has_compact_cached_slot) {
        const size_t required_stride =
            std::max(layout.bytes, runtime.device_stride);
        release_device_cache(runtime);
        if (!allocate_device_cache(runtime, err, required_stride)) return false;
    }
    runtime.layer_device_layouts.emplace(layer, layout);
    return true;
}

MoeHybridStreamEngine::MoeHybridStreamEngine() = default;
MoeHybridStreamEngine::~MoeHybridStreamEngine() { destroy(); }
MoeHybridStreamEngine::MoeHybridStreamEngine(MoeHybridStreamEngine &&) noexcept = default;
MoeHybridStreamEngine & MoeHybridStreamEngine::operator=(MoeHybridStreamEngine &&) noexcept = default;

bool MoeHybridStreamEngine::init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
                                 std::string * err) {
    return init(gpu_backend, max_expert_bytes, MoeStreamConfig::from_env(), err);
}

bool MoeHybridStreamEngine::init(ggml_backend_t gpu_backend,
                                 size_t max_expert_bytes,
                                 const MoeStreamConfig & config,
                                 std::string * err) {
    destroy();
    if (!gpu_backend || max_expert_bytes == 0) {
        if (err) *err = "invalid arguments to stream engine init";
        return false;
    }

    std::unique_ptr<Runtime> runtime(new (std::nothrow) Runtime);
    if (!runtime) {
        if (err) *err = "failed to allocate stream runtime";
        return false;
    }
    runtime->backend = gpu_backend;
    ggml_backend_dev_t device = ggml_backend_get_device(gpu_backend);
    if (!device) {
        if (err) *err = "failed to resolve SSD stream backend device";
        return false;
    }
    runtime->transfer_backend = ggml_backend_dev_init(device, nullptr);
    if (!runtime->transfer_backend) {
        if (err) *err = "failed to create SSD upload backend stream";
        return false;
    }
    if (!runtime->host_allocator.init(runtime->transfer_backend, err)) {
        ggml_backend_free(runtime->transfer_backend);
        runtime->transfer_backend = nullptr;
        return false;
    }
    runtime->max_expert_bytes = max_expert_bytes;
    runtime->config = config;
    runtime->config.device_slots = std::max(2, runtime->config.device_slots);
    runtime->io.reset(new (std::nothrow) MoeNvmeScheduler);
    if (!runtime->io) {
        if (err) *err = "failed to allocate SSD scheduler";
        ggml_backend_free(runtime->transfer_backend);
        runtime->transfer_backend = nullptr;
        return false;
    }
    if (!runtime->io->init(runtime->config.nvme, max_expert_bytes,
                           BackendHostAllocator::allocate_callback,
                           BackendHostAllocator::free_callback,
                           &runtime->host_allocator, err)) {
        runtime->io.reset();
        ggml_backend_free(runtime->transfer_backend);
        runtime->transfer_backend = nullptr;
        return false;
    }
    if (!allocate_device_cache(*runtime, err)) {
        // Install the partial runtime so destroy() releases every resource.
        runtime_ = std::move(runtime);
        destroy();
        return false;
    }
    runtime_ = std::move(runtime);
    return true;
}

bool MoeHybridStreamEngine::init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
                                 const MoeHybridStorage & storage,
                                 std::string * err) {
    return init(gpu_backend, max_expert_bytes, storage, MoeStreamConfig::from_env(), err);
}

bool MoeHybridStreamEngine::init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
                                 const MoeHybridStorage & storage,
                                 const MoeStreamConfig & config,
                                 std::string * err) {
    if (!init(gpu_backend, max_expert_bytes, config, err)) return false;
    if (!bind_storage(storage, err)) {
        destroy();
        return false;
    }
    return true;
}

bool MoeHybridStreamEngine::bind_storage(const MoeHybridStorage & storage,
                                         std::string * err) {
    if (!runtime_ || !runtime_->io || !runtime_->io->is_initialized()) {
        if (err) *err = "stream engine is not initialized";
        return false;
    }
    return runtime_->io->bind_source(
        {storage.mmap_data, storage.mmap_size, storage.mmap_fd},
        storage.layer_regions, err);
}

bool MoeHybridStreamEngine::bind_sources(
        const std::vector<MoeNvmeSource> & sources,
        const std::vector<LayerExpertRegions> & layer_regions,
        std::string * err) {
    if (!runtime_ || !runtime_->io || !runtime_->io->is_initialized()) {
        if (err) *err = "stream engine is not initialized";
        return false;
    }
    return runtime_->io->bind_sources(sources, layer_regions, err);
}

bool MoeHybridStreamEngine::is_ready() const {
    return runtime_ && runtime_->backend && runtime_->io &&
           runtime_->io->is_initialized() && runtime_->transfer_backend &&
           !runtime_->device_slots.empty();
}

bool MoeHybridStreamEngine::is_bound() const {
    return is_ready() && runtime_->io->is_bound();
}

void MoeHybridStreamEngine::destroy() {
    if (!runtime_) return;
    const size_t device_cache_slot_count = runtime_->device_slots.size();
    const size_t device_cache_byte_count = runtime_->device_pool_bytes;
    if (runtime_->backend) ggml_backend_synchronize(runtime_->backend);
    if (runtime_->transfer_backend) {
        ggml_backend_synchronize(runtime_->transfer_backend);
    }
    runtime_->graph_cache.clear();
    runtime_->fused_decode_graph_cache.clear();
    for (Runtime::DeviceSlot & slot : runtime_->device_slots) {
        slot.host_lease.reset();
        if (slot.ready) ggml_backend_event_free(slot.ready);
        slot.ready = nullptr;
        slot.data = nullptr;
        slot.transfer_tensor = nullptr;
        slot.pending = false;
    }
    runtime_->device_slots.clear();
    runtime_->device_index.clear();
    if (runtime_->device_pool_buffer) {
        ggml_backend_buffer_free(runtime_->device_pool_buffer);
    }
    runtime_->device_pool_buffer = nullptr;
    runtime_->device_pool = nullptr;
    if (runtime_->device_slot_ctx) ggml_free(runtime_->device_slot_ctx);
    runtime_->device_slot_ctx = nullptr;
    if (runtime_->io) {
        const MoeNvmeStats stats = runtime_->io->stats();
        if (stats.requests != 0 || stats.read_ops != 0 || stats.errors != 0) {
            const double payload_gib = (double) stats.payload_bytes /
                                       (1024.0 * 1024.0 * 1024.0);
            const double physical_gib = (double) stats.physical_bytes /
                                        (1024.0 * 1024.0 * 1024.0);
            const double read_seconds = (double) stats.active_io_ns / 1.0e9;
            const double read_gib_s = read_seconds > 0.0
                ? physical_gib / read_seconds : 0.0;
            const double hit_rate = stats.requests > 0
                ? 100.0 * (double) stats.cache_hits / (double) stats.requests : 0.0;
            const double mean_wait_ms = stats.demand_requests > 0
                ? ((double) stats.wait_ns / 1.0e6) /
                  (double) stats.demand_requests : 0.0;
            std::fprintf(stderr,
                "[moe-nvme] io=%s requests=%llu reads=%llu "
                "payload=%.3f GiB physical=%.3f GiB active-io-rate=%.3f GiB/s "
                "cache-hit=%.1f%% mean-demand-wait=%.3f ms "
                "dedupe=%llu upgrades=%llu dropped-prefetch=%llu "
                "timeouts=%llu errors=%llu "
                "device-cache=%.1f MiB slots=%zu pinned=%zu "
                "hits=%llu misses=%llu evictions=%llu "
                "graphs=%llu graph-hits=%llu graph-evictions=%llu launches=%llu "
                "fused-decode-launches=%llu fused-decode-experts=%llu "
                "cache-first-reorders=%llu cache-first-experts=%llu\n",
                runtime_->io->effective_backend_name(),
                (unsigned long long) stats.requests,
                (unsigned long long) stats.read_ops,
                payload_gib, physical_gib, read_gib_s, hit_rate, mean_wait_ms,
                (unsigned long long) stats.inflight_deduplications,
                (unsigned long long) stats.demand_upgrades,
                (unsigned long long) stats.prefetch_drops,
                (unsigned long long) stats.demand_timeouts,
                (unsigned long long) stats.errors,
                device_cache_byte_count / 1024.0 / 1024.0,
                device_cache_slot_count,
                runtime_->pinned_experts,
                (unsigned long long) runtime_->device_cache_hits,
                (unsigned long long) runtime_->device_cache_misses,
                (unsigned long long) runtime_->device_cache_evictions,
                (unsigned long long) runtime_->compute_stats.graph_builds,
                (unsigned long long) runtime_->compute_stats.graph_cache_hits,
                (unsigned long long) runtime_->compute_stats.graph_evictions,
                (unsigned long long) runtime_->compute_stats.graph_launches,
                (unsigned long long) runtime_->compute_stats.fused_decode_launches,
                (unsigned long long) runtime_->compute_stats.fused_decode_experts,
                (unsigned long long) runtime_->compute_stats.cache_first_reorders,
                (unsigned long long) runtime_->compute_stats.cache_first_experts);
        }
        runtime_->io->destroy();
    }
    runtime_->io.reset();
    if (runtime_->zero_padding) {
        runtime_->host_allocator.release(runtime_->zero_padding);
        runtime_->zero_padding = nullptr;
        runtime_->zero_padding_bytes = 0;
    }
    if (runtime_->transfer_backend) {
        ggml_backend_free(runtime_->transfer_backend);
        runtime_->transfer_backend = nullptr;
    }
    runtime_.reset();
}

void MoeHybridStreamEngine::request_experts(int layer, const int32_t * expert_ids,
                                             int count, MoeNvmePriority priority) {
    if (!is_bound() || !expert_ids || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        if (expert_ids[i] < 0) continue;
        const uint64_t key = device_key(layer, expert_ids[i]);
        const auto cached = runtime_->device_index.find(key);
        if (cached != runtime_->device_index.end()) {
            const int slot_index = cached->second;
            if (slot_index >= 0 && slot_index < (int) runtime_->device_slots.size()) {
                const Runtime::DeviceSlot & slot =
                    runtime_->device_slots[(size_t) slot_index];
                if (slot.valid && slot.key.layer == layer &&
                    slot.key.expert == expert_ids[i]) {
                    continue;
                }
            }
            runtime_->device_index.erase(cached);
        }
        (void) runtime_->io->request(layer, expert_ids[i], priority, nullptr);
    }
}

void MoeHybridStreamEngine::prefetch_cold_experts(
    const void * mmap_data, size_t mmap_size,
    const LayerExpertRegions & regions,
    const int32_t * cold_expert_ids, int n_cold) {
    if (!mmap_data || mmap_size == 0 || !cold_expert_ids || n_cold <= 0) return;

#if !defined(_WIN32)
    const size_t page_size = (size_t) std::max<long>(1, ::sysconf(_SC_PAGESIZE));
    auto advise = [&](size_t offset, size_t bytes) {
        if (offset > mmap_size || bytes > mmap_size - offset || bytes == 0) return;
        const size_t aligned = (offset / page_size) * page_size;
        const size_t length = bytes + (offset - aligned);
        (void) ::madvise(
            const_cast<uint8_t *>(static_cast<const uint8_t *>(mmap_data)) + aligned,
            length, MADV_WILLNEED);
    };
    for (int i = 0; i < n_cold; ++i) {
        const int expert = cold_expert_ids[i];
        if (expert < 0) continue;
        if (regions.fused_gate_up) {
            advise(regions.gate_up_exps.offset + (size_t) expert * regions.expert_bytes_gate_up,
                   regions.expert_bytes_gate_up);
        } else {
            advise(regions.gate_exps.offset + (size_t) expert * regions.expert_bytes_gate,
                   regions.expert_bytes_gate);
            advise(regions.up_exps.offset + (size_t) expert * regions.expert_bytes_up,
                   regions.expert_bytes_up);
        }
        advise(regions.down_exps.offset + (size_t) expert * regions.expert_bytes_down,
               regions.expert_bytes_down);
    }
#else
    (void) regions;
#endif
}

bool MoeHybridStreamEngine::stage_expert_async(int layer, int expert_id,
                                                int device_slot,
                                                std::string * err) {
    if (!is_bound()) {
        if (err) *err = "stream engine has no bound SSD model source";
        return false;
    }
    if (device_slot < 0 || device_slot >= (int) runtime_->device_slots.size()) {
        if (err) *err = "SSD device slot is out of range";
        return false;
    }
    Runtime::DeviceSlot & dst = runtime_->device_slots[(size_t) device_slot];
    if (dst.pinned) {
        if (err) *err = "SSD device slot is pinned by the warm expert set";
        return false;
    }
    if (dst.compute_users != 0) {
        if (err) *err = "SSD device slot is still in use by expert compute";
        return false;
    }
    if (dst.pending) {
        ggml_backend_event_synchronize(dst.ready);
        dst.pending = false;
        dst.host_lease.reset();
    }
    if (dst.cache_managed && dst.valid) {
        runtime_->device_index.erase(device_key(dst.key.layer, dst.key.expert));
    }
    dst.valid = false;
    dst.cache_managed = false;
    dst.pinned = false;
    dst.key = {};
    dst.layout = MoeExpertIoLayout{};
    dst.device_layout = Runtime::DeviceExpertLayout{};

    if (!dst.ready) {
        dst.ready = ggml_backend_event_new(
            ggml_backend_get_device(runtime_->transfer_backend));
        if (!dst.ready) {
            if (err) *err = "stream backend does not support upload events";
            return false;
        }
    }
    if (!dst.transfer_tensor) {
        if (err) *err = "SSD device slot has no transfer tensor";
        return false;
    }

    MoeNvmeLease lease;
    if (!runtime_->io->acquire(layer, expert_id, lease, err)) return false;
    if (lease.layout().payload_bytes > runtime_->max_expert_bytes) {
        if (err) *err = "streamed expert exceeds GPU device slot";
        return false;
    }
    const auto prepared = runtime_->layer_device_layouts.find(layer);
    if (prepared != runtime_->layer_device_layouts.end()) {
        const Runtime::DeviceExpertLayout & device_layout = prepared->second;
        if (!device_layout.configured ||
            device_layout.bytes > runtime_->device_stride) {
            if (err) *err = "prepared streamed expert exceeds GPU device stride";
            return false;
        }
        size_t maximum_padding = 0;
        for (int i = 0; i < device_layout.component_count; ++i) {
            const Runtime::DeviceComponentLayout & component =
                device_layout.components[i];
            maximum_padding = std::max(
                maximum_padding,
                component.alloc_bytes - component.logical_bytes);
        }
        if (!ensure_zero_padding(*runtime_, maximum_padding, err)) return false;

        for (int i = 0; i < device_layout.component_count; ++i) {
            const Runtime::DeviceComponentLayout & device_component =
                device_layout.components[i];
            const MoeExpertComponentLayout * io_component =
                lease.layout().component(device_component.kind);
            if (!io_component ||
                io_component->bytes != device_component.logical_bytes) {
                if (err) *err = "streamed device and storage components disagree";
                return false;
            }

            const uint8_t * source = nullptr;
            for (int span_index = 0;
                 span_index < lease.layout().span_count; ++span_index) {
                const MoeExpertIoSpan & span =
                    lease.layout().spans[span_index];
                if (io_component->device_offset < span.device_offset) continue;
                const size_t delta =
                    io_component->device_offset - span.device_offset;
                if (delta <= span.bytes &&
                    io_component->bytes <= span.bytes - delta) {
                    source = lease.data() + span.buffer_offset + delta;
                    break;
                }
            }
            if (!source) {
                if (err) *err = "streamed component is not contained in its I/O span";
                return false;
            }

            ggml_backend_tensor_set_async(
                runtime_->transfer_backend, dst.transfer_tensor,
                source, device_component.offset,
                device_component.logical_bytes);
            const size_t padding_bytes =
                device_component.alloc_bytes - device_component.logical_bytes;
            if (padding_bytes > 0) {
                ggml_backend_tensor_set_async(
                    runtime_->transfer_backend, dst.transfer_tensor,
                    runtime_->zero_padding,
                    device_component.offset + device_component.logical_bytes,
                    padding_bytes);
            }
        }
        dst.device_layout = device_layout;
    } else {
        // Storage-only callers do not supply a compute specification. Preserve
        // their compact byte-for-byte staging contract; numerical evaluation
        // always registers an exact backend-padded layout before reaching here.
        for (int i = 0; i < lease.layout().span_count; ++i) {
            const MoeExpertIoSpan & span = lease.layout().spans[i];
            ggml_backend_tensor_set_async(
                runtime_->transfer_backend, dst.transfer_tensor,
                lease.data() + span.buffer_offset,
                span.device_offset, span.bytes);
        }
        dst.device_layout.component_count = lease.layout().component_count;
        dst.device_layout.bytes = lease.layout().payload_bytes;
        for (int i = 0; i < lease.layout().component_count; ++i) {
            const MoeExpertComponentLayout & component =
                lease.layout().components[i];
            dst.device_layout.components[i] = {
                component.kind, component.device_offset,
                component.bytes, component.bytes};
        }
    }
    ggml_backend_event_record(dst.ready, runtime_->transfer_backend);
    dst.layout = lease.layout();
    dst.host_lease = std::move(lease);
    dst.pending = true;
    return true;
}

bool MoeHybridStreamEngine::stage_expert_cached_async(
        int layer, int expert_id, int * device_slot, std::string * err) {
    if (!device_slot) {
        if (err) *err = "SSD cache stage requires an output slot";
        return false;
    }
    *device_slot = -1;
    if (!is_bound()) {
        if (err) *err = "stream engine has no bound SSD model source";
        return false;
    }

    const uint64_t key = device_key(layer, expert_id);
    auto cached = runtime_->device_index.find(key);
    if (cached != runtime_->device_index.end()) {
        const int index = cached->second;
        if (index >= 0 && index < (int) runtime_->device_slots.size()) {
            Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) index];
            if (slot.valid && slot.cache_managed &&
                slot.key.layer == layer && slot.key.expert == expert_id) {
                ++runtime_->device_cache_hits;
                ++slot.frequency;
                slot.last_touch = ++runtime_->device_clock;
                *device_slot = index;
                return true;
            }
        }
        runtime_->device_index.erase(cached);
    }

    ++runtime_->device_cache_misses;
    int victim = -1;
    for (size_t i = 0; i < runtime_->device_slots.size(); ++i) {
        const Runtime::DeviceSlot & slot = runtime_->device_slots[i];
        if (!slot.valid && !slot.pending && slot.compute_users == 0) {
            victim = (int) i;
            break;
        }
    }
    if (victim < 0) {
        uint64_t best_score = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < runtime_->device_slots.size(); ++i) {
            const Runtime::DeviceSlot & slot = runtime_->device_slots[i];
            if (!slot.valid || slot.pending || slot.pinned ||
                slot.compute_users != 0) continue;
            const uint64_t age = runtime_->device_clock >= slot.last_touch
                ? runtime_->device_clock - slot.last_touch : 0;
            const uint64_t recency = age < 65535 ? 65535 - age : 0;
            const uint64_t score = (slot.frequency << 16) | recency;
            if (score < best_score) {
                best_score = score;
                victim = (int) i;
            }
        }
    }
    if (victim < 0) {
        if (err) *err = "all SSD GPU expert-cache slots are busy";
        return false;
    }

    const bool evicting = runtime_->device_slots[(size_t) victim].valid;
    if (!stage_expert_async(layer, expert_id, victim, err)) return false;
    Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) victim];
    slot.valid = true;
    slot.cache_managed = true;
    slot.key = {(int32_t) layer, (int32_t) expert_id};
    slot.frequency = 1;
    slot.last_touch = ++runtime_->device_clock;
    runtime_->device_index[key] = victim;
    if (evicting) ++runtime_->device_cache_evictions;
    *device_slot = victim;
    return true;
}

bool MoeHybridStreamEngine::activate_device_slot(int device_slot,
                                                  std::string * err) {
    if (!is_ready() || device_slot < 0 ||
        device_slot >= (int) runtime_->device_slots.size()) {
        if (err) *err = "SSD device slot is out of range";
        return false;
    }
    Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) device_slot];
    if (slot.pending) {
        ggml_backend_event_synchronize(slot.ready);
        slot.pending = false;
        slot.host_lease.reset();
    }
    if (slot.layout.component_count < 2) {
        if (err) *err = "SSD device slot has no complete expert";
        return false;
    }
    if (slot.cache_managed) {
        if (slot.compute_users != 0) {
            if (err) *err = "cached expert slot is already executing";
            return false;
        }
        ++slot.compute_users;
        ++slot.frequency;
        slot.last_touch = ++runtime_->device_clock;
    }
    runtime_->active_slot = device_slot;
    return true;
}

void MoeHybridStreamEngine::release_device_slot(int device_slot) {
    if (!runtime_ || device_slot < 0 ||
        device_slot >= (int) runtime_->device_slots.size()) {
        return;
    }
    Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) device_slot];
    if (slot.cache_managed && slot.compute_users > 0) --slot.compute_users;
    if (runtime_->active_slot == device_slot) runtime_->active_slot = -1;
}

int MoeHybridStreamEngine::device_slot_count() const {
    return runtime_ ? (int) runtime_->device_slots.size() : 0;
}

size_t MoeHybridStreamEngine::device_cache_bytes() const {
    return runtime_ ? runtime_->device_pool_bytes : 0;
}

size_t MoeHybridStreamEngine::pinned_expert_count() const {
    return runtime_ ? runtime_->pinned_experts : 0;
}

bool MoeHybridStreamEngine::warm_and_pin_device_cache(
        const std::vector<MoeStreamExpertSpec> & layer_specs,
        const std::vector<MoeStreamCacheWarmEntry> & entries,
        int reserve_slots,
        MoeStreamCacheWarmStats * stats,
        std::string * err) {
    MoeStreamCacheWarmStats local;
    local.requested = entries.size();
    if (stats) *stats = local;
    if (!is_bound()) {
        if (err) *err = "cannot warm an unbound streamed expert cache";
        return false;
    }
    reserve_slots = std::max(2, reserve_slots);

    std::lock_guard<std::mutex> compute_guard(runtime_->compute_mutex);

    std::vector<MoeStreamCacheWarmEntry> candidates = entries;
    std::stable_sort(candidates.begin(), candidates.end(),
        [](const MoeStreamCacheWarmEntry & a,
           const MoeStreamCacheWarmEntry & b) {
            if (a.frequency != b.frequency) return a.frequency > b.frequency;
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.expert < b.expert;
        });
    std::unordered_set<uint64_t> seen;
    std::vector<MoeStreamCacheWarmEntry> unique;
    unique.reserve(candidates.size());
    for (const MoeStreamCacheWarmEntry & candidate : candidates) {
        if (candidate.layer < 0 || candidate.expert < 0 ||
            (size_t) candidate.layer >= layer_specs.size()) {
            if (err) *err = "streamed cache warm entry is out of range";
            return false;
        }
        const uint64_t key = device_key(candidate.layer, candidate.expert);
        if (seen.insert(key).second) unique.push_back(candidate);
    }

    // Establish every backend-padded layer layout before loading data. If one
    // format needs a larger device stride, the cache is resized once while it
    // is still empty instead of invalidating an already-warmed prefix.
    std::unordered_set<int32_t> prepared_layers;
    for (const MoeStreamCacheWarmEntry & candidate : unique) {
        if (!prepared_layers.insert(candidate.layer).second) continue;
        if (!prepare_device_expert_layout(
                *runtime_, candidate.layer,
                layer_specs[(size_t) candidate.layer], err)) {
            return false;
        }
    }

    const size_t slot_count = runtime_->device_slots.size();
    const size_t pin_capacity = slot_count > (size_t) reserve_slots
        ? slot_count - (size_t) reserve_slots : 0;
    for (const MoeStreamCacheWarmEntry & candidate : unique) {
        const uint64_t key = device_key(candidate.layer, candidate.expert);
        auto found = runtime_->device_index.find(key);
        if (found != runtime_->device_index.end()) {
            const int slot_index = found->second;
            if (slot_index >= 0 && slot_index < (int) slot_count) {
                Runtime::DeviceSlot & resident =
                    runtime_->device_slots[(size_t) slot_index];
                if (resident.valid && resident.pinned) {
                    ++local.already_resident;
                    continue;
                }
            }
        }
        if (runtime_->pinned_experts >= pin_capacity) {
            ++local.capacity_drops;
            continue;
        }

        const bool was_resident = found != runtime_->device_index.end();
        int device_slot = -1;
        if (!stage_expert_cached_async(
                candidate.layer, candidate.expert, &device_slot, err) ||
            !activate_device_slot(device_slot, err)) {
            if (stats) *stats = local;
            return false;
        }
        release_device_slot(device_slot);
        Runtime::DeviceSlot & slot =
            runtime_->device_slots[(size_t) device_slot];
        slot.pinned = true;
        slot.frequency = std::max<uint64_t>(slot.frequency, 2);
        ++runtime_->pinned_experts;
        ++local.admitted;
        if (was_resident) ++local.already_resident;
    }
    if (stats) *stats = local;
    return true;
}

ggml_backend_t MoeHybridStreamEngine::compute_backend() const {
    return runtime_ ? runtime_->backend : nullptr;
}

bool MoeHybridStreamEngine::stream_expert_sync(int layer, int expert_id,
                                                std::string * err) {
    if (!stage_expert_async(layer, expert_id, 0, err)) return false;
    return activate_device_slot(0, err);
}

bool MoeHybridStreamEngine::stream_expert_sync(
    const void * mmap_data, size_t mmap_size,
    const LayerExpertRegions & regions, int expert_id,
    ggml_backend_t gpu_backend, std::string * err) {
    (void) gpu_backend;
    if (!is_ready()) {
        if (err) *err = "stream engine is not initialized";
        return false;
    }
    if (!is_bound()) {
        std::vector<LayerExpertRegions> one_layer{regions};
        if (!runtime_->io->bind_source({mmap_data, mmap_size, -1}, one_layer, err)) return false;
    }
    return stream_expert_sync(0, expert_id, err);
}

const void * MoeHybridStreamEngine::scratch_gate_data() const {
    if (!runtime_ || runtime_->active_slot < 0) return nullptr;
    const Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) runtime_->active_slot];
    const MoeExpertComponentKind kind = slot.layout.fused_gate_up
        ? MoeExpertComponentKind::FusedGateUp : MoeExpertComponentKind::Gate;
    const Runtime::DeviceComponentLayout * component =
        slot.device_layout.component(kind);
    return component
        ? static_cast<const uint8_t *>(slot.data) + component->offset
        : nullptr;
}

const void * MoeHybridStreamEngine::scratch_up_data() const {
    if (!runtime_ || runtime_->active_slot < 0) return nullptr;
    const Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) runtime_->active_slot];
    if (slot.layout.fused_gate_up) return nullptr;
    const Runtime::DeviceComponentLayout * component =
        slot.device_layout.component(MoeExpertComponentKind::Up);
    return component
        ? static_cast<const uint8_t *>(slot.data) + component->offset
        : nullptr;
}

const void * MoeHybridStreamEngine::scratch_down_data() const {
    if (!runtime_ || runtime_->active_slot < 0) return nullptr;
    const Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) runtime_->active_slot];
    const Runtime::DeviceComponentLayout * component =
        slot.device_layout.component(MoeExpertComponentKind::Down);
    return component
        ? static_cast<const uint8_t *>(slot.data) + component->offset
        : nullptr;
}

size_t MoeHybridStreamEngine::scratch_gate_bytes() const {
    if (!runtime_ || runtime_->active_slot < 0) return 0;
    const Runtime::DeviceSlot & slot =
        runtime_->device_slots[(size_t) runtime_->active_slot];
    const MoeExpertComponentKind kind = slot.layout.fused_gate_up
        ? MoeExpertComponentKind::FusedGateUp : MoeExpertComponentKind::Gate;
    const Runtime::DeviceComponentLayout * component =
        slot.device_layout.component(kind);
    return component ? component->logical_bytes : 0;
}

size_t MoeHybridStreamEngine::scratch_up_bytes() const {
    if (!runtime_ || runtime_->active_slot < 0) return 0;
    const Runtime::DeviceSlot & slot =
        runtime_->device_slots[(size_t) runtime_->active_slot];
    if (slot.layout.fused_gate_up) return 0;
    const Runtime::DeviceComponentLayout * component =
        slot.device_layout.component(MoeExpertComponentKind::Up);
    return component ? component->logical_bytes : 0;
}

size_t MoeHybridStreamEngine::scratch_down_bytes() const {
    if (!runtime_ || runtime_->active_slot < 0) return 0;
    const Runtime::DeviceSlot & slot =
        runtime_->device_slots[(size_t) runtime_->active_slot];
    const Runtime::DeviceComponentLayout * component =
        slot.device_layout.component(MoeExpertComponentKind::Down);
    return component ? component->logical_bytes : 0;
}

size_t MoeHybridStreamEngine::pinned_bytes() const {
    return runtime_ && runtime_->io ? runtime_->io->total_host_bytes() : 0;
}

size_t MoeHybridStreamEngine::scratch_bytes() const {
    return runtime_ ? runtime_->device_pool_bytes : 0;
}

const char * MoeHybridStreamEngine::io_backend_name() const {
    return runtime_ && runtime_->io ? runtime_->io->effective_backend_name() : "uninitialized";
}

MoeNvmeStats MoeHybridStreamEngine::io_stats() const {
    return runtime_ && runtime_->io ? runtime_->io->stats() : MoeNvmeStats{};
}

MoeStreamComputeStats MoeHybridStreamEngine::compute_stats() const {
    return runtime_ ? runtime_->compute_stats : MoeStreamComputeStats{};
}

static bool eval_moe_cold_experts_streaming_reference(
    MoeHybridStreamEngine &         engine,
    ggml_backend_t                  gpu_backend,
    const void *                    mmap_data,
    size_t                          mmap_size,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    const LayerExpertRegions &      regions,
    const MoeHybridLayerStorage &   storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    int                             layer) {

    // The streamed tier can intentionally target a different owner from the
    // caller (Strix for Lucebox, while hot/dense work stays on the R9700).
    // Always build and launch the expert graph on the device that owns the
    // stream slots.
    if (engine.compute_backend()) gpu_backend = engine.compute_backend();

    const int n_embd = cfg.expert_embd();
    const int n_ff_exp = cfg.n_ff_exp;
    const int n_used = cfg.n_expert_used;
    const int total_slots = n_used * n_tokens;

    if (cfg.gated_activation == MoeGatedActivation::Situ &&
        (cfg.situ_beta <= 0.0f || cfg.situ_linear_beta <= 0.0f)) {
        if (err) *err = "SiTU activation scales must be positive";
        return false;
    }

    out.assign((size_t) n_embd * (size_t) n_tokens, 0.0f);
    if (!engine.is_ready()) {
        if (err) *err = "stream engine is not ready";
        return false;
    }
    if (!engine.is_bound() && (!mmap_data || mmap_size == 0)) {
        if (err) *err = "mmap is not available";
        return false;
    }

    std::vector<bool> cold_needed((size_t) cfg.n_expert, false);
    for (int i = 0; i < total_slots; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= cfg.n_expert) continue;
        if (selected_weights[i] == 0.0f) continue;
        if (storage.hot_local_by_global[(size_t) gid] < 0) cold_needed[(size_t) gid] = true;
    }

    std::vector<int32_t> unique_cold;
    for (int expert = 0; expert < cfg.n_expert; ++expert) {
        if (cold_needed[(size_t) expert]) unique_cold.push_back((int32_t) expert);
    }
    if (unique_cold.empty()) return true;

    const bool cache_pipeline = engine.is_bound();

    // Admit every actual route before compute. io_uring sees the whole batch;
    // the thread fallback obtains enough outstanding reads to saturate NVMe.
    if (cache_pipeline) {
        engine.request_experts(layer, unique_cold.data(), (int) unique_cold.size(),
                               MoeNvmePriority::Demand);
    }

    int staged_device_slot = 0;
    if (cache_pipeline) {
        if (!engine.stage_expert_cached_async(
                layer, unique_cold[0], &staged_device_slot, err)) return false;
    } else {
        if (!engine.stream_expert_sync(mmap_data, mmap_size, regions,
                                       unique_cold[0], gpu_backend, err)) return false;
    }

    for (size_t cold_index = 0; cold_index < unique_cold.size(); ++cold_index) {
        const int32_t cold_eid = unique_cold[cold_index];
        const int current_device_slot = staged_device_slot;
        if (cache_pipeline &&
            !engine.activate_device_slot(current_device_slot, err)) return false;
        auto release_current = [&]() {
            if (cache_pipeline) engine.release_device_slot(current_device_slot);
        };

        struct TokenHit { int token; float weight; };
        std::vector<TokenHit> hits;
        hits.reserve((size_t) n_tokens);
        for (int token = 0; token < n_tokens; ++token) {
            for (int k = 0; k < n_used; ++k) {
                const int slot = token * n_used + k;
                if (selected_ids[slot] != cold_eid) continue;
                if (selected_weights[slot] != 0.0f) {
                    hits.push_back({token, selected_weights[slot]});
                }
                break;
            }
        }
        if (hits.empty()) {
            release_current();
            continue;
        }

        const int batch = (int) hits.size();
        std::vector<float> batch_input((size_t) n_embd * (size_t) batch);
        for (int i = 0; i < batch; ++i) {
            const float * src = cur_host + (size_t) hits[(size_t) i].token * (size_t) n_embd;
            std::memcpy(batch_input.data() + (size_t) i * (size_t) n_embd,
                        src, sizeof(float) * (size_t) n_embd);
        }

        ggml_init_params ip{};
        ip.mem_size = 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            if (err) *err = "ggml_init failed in SSD streaming eval";
            release_current();
            return false;
        }

        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch);
        ggml_set_input(inp);
        ggml_tensor * gate_t = nullptr;
        ggml_tensor * up_t = nullptr;
        ggml_tensor * down_t = nullptr;
        ggml_tensor * gate_up_t = nullptr;
        if (regions.fused_gate_up) {
            gate_up_t = ggml_new_tensor_2d(ctx, desc.ffn_gate_up_exps->type,
                                            n_embd, 2 * n_ff_exp);
            down_t = ggml_new_tensor_2d(ctx, desc.ffn_down_exps->type,
                                        n_ff_exp, n_embd);
            ggml_set_input(gate_up_t);
            ggml_set_input(down_t);
        } else {
            gate_t = ggml_new_tensor_2d(ctx, desc.ffn_gate_exps->type,
                                        n_embd, n_ff_exp);
            up_t = ggml_new_tensor_2d(ctx, desc.ffn_up_exps->type,
                                      n_embd, n_ff_exp);
            down_t = ggml_new_tensor_2d(ctx, desc.ffn_down_exps->type,
                                        n_ff_exp, n_embd);
            ggml_set_input(gate_t);
            ggml_set_input(up_t);
            ggml_set_input(down_t);
        }

        auto apply_gated_activation = [&](ggml_tensor * gate,
                                          ggml_tensor * up) -> ggml_tensor * {
            if (cfg.gated_activation == MoeGatedActivation::Situ) {
                ggml_tensor * nonlinear = ggml_scale(ctx, gate, 1.0f / cfg.situ_beta);
                nonlinear = ggml_tanh(ctx, nonlinear);
                nonlinear = ggml_scale(ctx, nonlinear, cfg.situ_beta);
                nonlinear = ggml_mul(ctx, nonlinear, ggml_sigmoid(ctx, gate));
                ggml_tensor * linear = ggml_scale(
                    ctx, up, 1.0f / cfg.situ_linear_beta);
                linear = ggml_tanh(ctx, linear);
                linear = ggml_scale(ctx, linear, cfg.situ_linear_beta);
                return ggml_mul(ctx, nonlinear, linear);
            }
            if (cfg.swiglu_clamp > 0.0f) {
                return ggml_swiglu_ds4_split(ctx, gate, up, cfg.swiglu_clamp);
            }
            return ggml_swiglu_split(ctx, gate, up);
        };

        ggml_tensor * gated = nullptr;
        if (gate_up_t) {
            ggml_tensor * gate_up_out = ggml_mul_mat(ctx, gate_up_t, inp);
            if (desc.ffn_gate_up_exps_s != 1.0f) {
                gate_up_out = ggml_scale(ctx, gate_up_out, desc.ffn_gate_up_exps_s);
            }
            ggml_tensor * gate_part = ggml_view_2d(
                ctx, gate_up_out, n_ff_exp, batch, gate_up_out->nb[1], 0);
            ggml_tensor * up_part = ggml_view_2d(
                ctx, gate_up_out, n_ff_exp, batch, gate_up_out->nb[1],
                (size_t) n_ff_exp * sizeof(float));
            gate_part = ggml_cont(ctx, gate_part);
            up_part = ggml_cont(ctx, up_part);
            gated = apply_gated_activation(gate_part, up_part);
        } else {
            ggml_tensor * gate = ggml_mul_mat(ctx, gate_t, inp);
            if (desc.ffn_gate_exps_s != 1.0f) gate = ggml_scale(ctx, gate, desc.ffn_gate_exps_s);
            ggml_tensor * up = ggml_mul_mat(ctx, up_t, inp);
            if (desc.ffn_up_exps_s != 1.0f) up = ggml_scale(ctx, up, desc.ffn_up_exps_s);
            gated = apply_gated_activation(gate, up);
        }
        ggml_tensor * expert_out = ggml_mul_mat(ctx, down_t, gated);
        if (desc.ffn_down_exps_s != 1.0f) {
            expert_out = ggml_scale(ctx, expert_out, desc.ffn_down_exps_s);
        }

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
        ggml_set_output(expert_out);
        ggml_build_forward_expand(graph, expert_out);
        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(gpu_backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
            if (err) *err = "SSD streaming eval graph allocation failed";
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            release_current();
            return false;
        }

        ggml_backend_tensor_set(inp, batch_input.data(), 0,
                                sizeof(float) * (size_t) n_embd * (size_t) batch);
        if (gate_up_t) {
            gate_up_t->data = const_cast<void *>(engine.scratch_gate_data());
            down_t->data = const_cast<void *>(engine.scratch_down_data());
        } else {
            gate_t->data = const_cast<void *>(engine.scratch_gate_data());
            up_t->data = const_cast<void *>(engine.scratch_up_data());
            down_t->data = const_cast<void *>(engine.scratch_down_data());
        }

        const ggml_status status = ggml_backend_graph_compute_async(gpu_backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            if (err) *err = "SSD streaming expert compute launch failed";
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            release_current();
            return false;
        }

        // Compute N is now running. Wait for the already-issued disk read of
        // N+1 and enqueue its H2D into a different device slot.
        if (cold_index + 1 < unique_cold.size() && cache_pipeline) {
            int next_slot = -1;
            if (!engine.stage_expert_cached_async(
                    layer, unique_cold[cold_index + 1], &next_slot, err)) {
                ggml_backend_synchronize(gpu_backend);
                release_current();
                ggml_gallocr_free(alloc);
                ggml_free(ctx);
                return false;
            }
            staged_device_slot = next_slot;
        }

        ggml_backend_synchronize(gpu_backend);
        std::vector<float> batch_result((size_t) n_embd * (size_t) batch);
        ggml_backend_tensor_get(expert_out, batch_result.data(), 0,
                                sizeof(float) * (size_t) n_embd * (size_t) batch);
        for (int i = 0; i < batch; ++i) {
            const float weight = hits[(size_t) i].weight;
            float * dst = out.data() + (size_t) hits[(size_t) i].token * (size_t) n_embd;
            const float * src = batch_result.data() + (size_t) i * (size_t) n_embd;
            for (int j = 0; j < n_embd; ++j) dst[j] += weight * src[(size_t) j];
        }
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        release_current();

        if (cold_index + 1 < unique_cold.size() && !cache_pipeline) {
            if (!engine.stream_expert_sync(mmap_data, mmap_size, regions,
                                           unique_cold[cold_index + 1],
                                           gpu_backend, err)) return false;
        }
    }
    return true;
}

bool eval_moe_streamed_experts(
        MoeHybridStreamEngine & engine,
        const MoeStreamExpertSpec & spec,
        const MoeStreamRouteBatch & batch,
        std::vector<float> & out,
        std::string * err) {
    if (!engine.runtime_ || !engine.is_bound()) {
        if (err) *err = "streamed expert evaluation requires a bound model source";
        return false;
    }
    if (batch.layer < 0 || batch.n_expert <= 0 || batch.top_k <= 0 ||
        batch.top_k > batch.n_expert ||
        batch.n_tokens <= 0 || !batch.inputs || !batch.selected_ids ||
        !batch.selected_weights || spec.input_dim <= 0 ||
        spec.output_dim <= 0) {
        if (err) *err = "invalid model-neutral streamed route batch";
        return false;
    }
    if (batch.resident_local_by_global &&
        batch.resident_map_size < (size_t) batch.n_expert) {
        if (err) *err = "streamed route residency map is smaller than n_expert";
        return false;
    }
    size_t output_values = 0;
    if (!checked_mul_size((size_t) spec.output_dim,
                          (size_t) batch.n_tokens, output_values)) {
        if (err) *err = "streamed route output size overflow";
        return false;
    }
    out.assign(output_values, 0.0f);

    auto & runtime = *engine.runtime_;
    std::lock_guard<std::mutex> compute_guard(runtime.compute_mutex);
    if (!prepare_device_expert_layout(
            runtime, batch.layer, spec, err)) {
        return false;
    }

    size_t route_slots = 0;
    if (!checked_mul_size((size_t) batch.top_k,
                          (size_t) batch.n_tokens, route_slots)) {
        if (err) *err = "streamed route slot count overflow";
        return false;
    }
    std::vector<bool> needed((size_t) batch.n_expert, false);
    for (size_t i = 0; i < route_slots; ++i) {
        const int32_t expert = batch.selected_ids[i];
        if (expert < 0) continue;
        if (expert >= batch.n_expert) {
            if (err) *err = "native router selected an out-of-range expert";
            return false;
        }
        if (!std::isfinite(batch.selected_weights[i])) {
            if (err) *err = "native router produced a non-finite expert weight";
            return false;
        }
        if (batch.selected_weights[i] == 0.0f) continue;
        if (batch.resident_local_by_global &&
            batch.resident_local_by_global[(size_t) expert] >= 0) {
            continue;
        }
        needed[(size_t) expert] = true;
    }

    std::vector<int32_t> unique_experts;
    for (int expert = 0; expert < batch.n_expert; ++expert) {
        if (needed[(size_t) expert]) unique_experts.push_back((int32_t) expert);
    }
    if (unique_experts.empty()) return true;

    engine.request_experts(batch.layer, unique_experts.data(),
                           (int) unique_experts.size(),
                           MoeNvmePriority::Demand);

    // Decode is latency-sensitive and normally selects several experts for a
    // single token. If every selected expert is already device-resident, keep
    // the complete fork/join on the GPU. A miss retains the pipelined path so
    // expert N compute can overlap the upload of N+1. This removes the hot-path
    // host boundary without changing routing, formats, or prefill behavior.
    bool all_selected_resident = runtime.config.fused_decode &&
        batch.n_tokens == 1 && unique_experts.size() > 1;
    if (all_selected_resident) {
        for (const int32_t expert : unique_experts) {
            const auto found = runtime.device_index.find(
                device_key(batch.layer, expert));
            if (found == runtime.device_index.end() || found->second < 0 ||
                found->second >= (int) runtime.device_slots.size()) {
                all_selected_resident = false;
                break;
            }
            const auto & slot =
                runtime.device_slots[(size_t) found->second];
            if (!slot.valid || !slot.cache_managed ||
                slot.compute_users != 0 || slot.key.layer != batch.layer ||
                slot.key.expert != expert) {
                all_selected_resident = false;
                break;
            }
        }
    }
    if (all_selected_resident) {
        struct ActiveSlotSet {
            MoeHybridStreamEngine & engine;
            std::vector<int> slots;
            ~ActiveSlotSet() {
                for (auto it = slots.rbegin(); it != slots.rend(); ++it) {
                    engine.release_device_slot(*it);
                }
            }
        } active{engine, {}};
        active.slots.reserve(unique_experts.size());

        std::vector<StreamExpertDeviceBinding> bindings;
        std::vector<float> route_weights;
        bindings.reserve(unique_experts.size());
        route_weights.reserve(unique_experts.size());
        for (const int32_t expert : unique_experts) {
            float combined_weight = 0.0f;
            for (int rank = 0; rank < batch.top_k; ++rank) {
                if (batch.selected_ids[(size_t) rank] == expert) {
                    combined_weight += batch.selected_weights[(size_t) rank];
                }
            }
            if (!std::isfinite(combined_weight)) {
                if (err) *err = "combined expert route weight overflowed";
                return false;
            }

            int slot_index = -1;
            if (!engine.stage_expert_cached_async(
                    batch.layer, expert, &slot_index, err) ||
                !engine.activate_device_slot(slot_index, err)) {
                return false;
            }
            active.slots.push_back(slot_index);
            const auto & slot = runtime.device_slots[(size_t) slot_index];
            if (!validate_moe_stream_expert_layout(spec, slot.layout, err)) {
                return false;
            }
            const MoeExpertComponentKind gate_kind = spec.fused_gate_up
                ? MoeExpertComponentKind::FusedGateUp
                : MoeExpertComponentKind::Gate;
            const auto * gate_component =
                slot.device_layout.component(gate_kind);
            const auto * up_component = spec.fused_gate_up
                ? nullptr
                : slot.device_layout.component(MoeExpertComponentKind::Up);
            const auto * down_component =
                slot.device_layout.component(MoeExpertComponentKind::Down);
            if (!gate_component || !down_component ||
                (!spec.fused_gate_up && !up_component)) {
                if (err) *err =
                    "streamed device layout is missing an expert component";
                return false;
            }
            const auto * base = static_cast<const uint8_t *>(slot.data);
            bindings.push_back({
                base + gate_component->offset,
                up_component ? base + up_component->offset : nullptr,
                base + down_component->offset,
                gate_component->alloc_bytes,
                up_component ? up_component->alloc_bytes : 0,
                down_component->alloc_bytes,
            });
            route_weights.push_back(combined_weight);
        }

        std::unique_ptr<PersistentStreamMoEDecodeGraph> ephemeral;
        PersistentStreamMoEDecodeGraph * graph = nullptr;
        const uint64_t touch = ++runtime.graph_clock;
        if (runtime.config.graph_cache_entries > 0) {
            for (auto & candidate : runtime.fused_decode_graph_cache) {
                if (candidate && candidate->matches(
                        spec, (int) bindings.size())) {
                    candidate->last_touch = touch;
                    ++runtime.compute_stats.graph_cache_hits;
                    graph = candidate.get();
                    break;
                }
            }
        }
        if (!graph) {
            std::unique_ptr<PersistentStreamMoEDecodeGraph> built(
                new (std::nothrow) PersistentStreamMoEDecodeGraph);
            if (!built) {
                if (err) *err =
                    "failed to allocate fused streamed-MoE graph";
                return false;
            }
            if (!built->build(runtime.backend, runtime.device_pool_buffer,
                              spec, bindings, err)) {
                return false;
            }
            built->last_touch = touch;
            ++runtime.compute_stats.graph_builds;
            if (runtime.config.graph_cache_entries <= 0) {
                graph = built.get();
                ephemeral = std::move(built);
            } else {
                if ((int) runtime.fused_decode_graph_cache.size() >=
                    runtime.config.graph_cache_entries) {
                    auto victim = std::min_element(
                        runtime.fused_decode_graph_cache.begin(),
                        runtime.fused_decode_graph_cache.end(),
                        [](const auto & a, const auto & b) {
                            return a->last_touch < b->last_touch;
                        });
                    if (victim != runtime.fused_decode_graph_cache.end()) {
                        runtime.fused_decode_graph_cache.erase(victim);
                        ++runtime.compute_stats.graph_evictions;
                    }
                }
                graph = built.get();
                runtime.fused_decode_graph_cache.push_back(std::move(built));
            }
        }
        if (!graph->launch(
                bindings, batch.inputs, route_weights.data(), err) ||
            !graph->finish(out, err)) {
            return false;
        }
        ++runtime.compute_stats.graph_launches;
        ++runtime.compute_stats.fused_decode_launches;
        runtime.compute_stats.fused_decode_experts += bindings.size();
        return true;
    }

    // The weighted expert sum is order-independent mathematically. Execute
    // device-resident experts first so their compute overlaps every admitted
    // SSD miss instead of blocking on the first cold expert encountered by ID.
    // Contributions are accumulated later in the original deterministic order
    // to preserve the previous floating-point result.
    std::vector<int32_t> execution_experts = unique_experts;
    bool cache_first_reordered = false;
    if (runtime.config.cache_first_decode && batch.n_tokens == 1 &&
        execution_experts.size() > 1) {
        const auto is_device_resident = [&](int32_t expert) {
            const auto found = runtime.device_index.find(
                device_key(batch.layer, expert));
            if (found == runtime.device_index.end() || found->second < 0 ||
                found->second >= (int) runtime.device_slots.size()) {
                return false;
            }
            const auto & slot =
                runtime.device_slots[(size_t) found->second];
            return slot.valid && slot.cache_managed &&
                slot.compute_users == 0 && slot.key.layer == batch.layer &&
                slot.key.expert == expert;
        };
        const auto cold_begin = std::stable_partition(
            execution_experts.begin(), execution_experts.end(),
            is_device_resident);
        const size_t resident_count = (size_t) std::distance(
            execution_experts.begin(), cold_begin);
        cache_first_reordered = execution_experts != unique_experts;
        if (cache_first_reordered) {
            ++runtime.compute_stats.cache_first_reorders;
            runtime.compute_stats.cache_first_experts += resident_count;
        }
    }

    std::vector<float> ordered_contributions;
    if (cache_first_reordered) {
        size_t contribution_values = 0;
        if (!checked_mul_size(output_values, unique_experts.size(),
                              contribution_values)) {
            if (err) *err = "streamed expert contribution size overflow";
            return false;
        }
        ordered_contributions.assign(contribution_values, 0.0f);
    }

    int staged_slot = -1;
    if (!engine.stage_expert_cached_async(
            batch.layer, execution_experts[0], &staged_slot, err)) {
        return false;
    }

    auto acquire_graph = [&](int graph_batch,
                             std::unique_ptr<PersistentStreamExpertGraph> & ephemeral,
                             PersistentStreamExpertGraph ** graph_out) -> bool {
        *graph_out = nullptr;
        const uint64_t touch = ++runtime.graph_clock;
        if (runtime.config.graph_cache_entries > 0) {
            for (auto & candidate : runtime.graph_cache) {
                if (candidate && candidate->matches(spec, graph_batch)) {
                    candidate->last_touch = touch;
                    ++runtime.compute_stats.graph_cache_hits;
                    *graph_out = candidate.get();
                    return true;
                }
            }
        }

        const int active = runtime.active_slot;
        if (active < 0 || active >= (int) runtime.device_slots.size()) {
            if (err) *err = "no active streamed expert slot for graph build";
            return false;
        }
        const MoeExpertIoLayout & layout =
            runtime.device_slots[(size_t) active].layout;
        if (!validate_moe_stream_expert_layout(spec, layout, err)) return false;
        const auto & device_layout =
            runtime.device_slots[(size_t) active].device_layout;
        const MoeExpertComponentKind gate_kind = spec.fused_gate_up
            ? MoeExpertComponentKind::FusedGateUp
            : MoeExpertComponentKind::Gate;
        const auto * gate_component =
            device_layout.component(gate_kind);
        const auto * up_component =
            spec.fused_gate_up
                ? nullptr
                : device_layout.component(MoeExpertComponentKind::Up);
        const auto * down_component =
            device_layout.component(MoeExpertComponentKind::Down);
        if (!gate_component || !down_component ||
            (!spec.fused_gate_up && !up_component)) {
            if (err) *err = "streamed device layout is missing an expert component";
            return false;
        }

        std::unique_ptr<PersistentStreamExpertGraph> built(
            new (std::nothrow) PersistentStreamExpertGraph);
        if (!built) {
            if (err) *err = "failed to allocate persistent streamed-expert graph";
            return false;
        }
        if (!built->build(runtime.backend, runtime.device_pool_buffer,
                          spec, graph_batch,
                          engine.scratch_gate_data(),
                          engine.scratch_up_data(),
                          engine.scratch_down_data(),
                          gate_component->alloc_bytes,
                          up_component ? up_component->alloc_bytes : 0,
                          down_component->alloc_bytes,
                          err)) {
            return false;
        }
        built->last_touch = touch;
        ++runtime.compute_stats.graph_builds;

        if (runtime.config.graph_cache_entries <= 0) {
            *graph_out = built.get();
            ephemeral = std::move(built);
            return true;
        }
        if ((int) runtime.graph_cache.size() >=
            runtime.config.graph_cache_entries) {
            auto victim = std::min_element(
                runtime.graph_cache.begin(), runtime.graph_cache.end(),
                [](const auto & a, const auto & b) {
                    return a->last_touch < b->last_touch;
                });
            if (victim != runtime.graph_cache.end()) {
                runtime.graph_cache.erase(victim);
                ++runtime.compute_stats.graph_evictions;
            }
        }
        *graph_out = built.get();
        runtime.graph_cache.push_back(std::move(built));
        return true;
    };

    struct TokenHit { int token; float weight; };
    std::vector<TokenHit> hits;
    hits.reserve((size_t) batch.n_tokens);
    std::vector<float> compact_input;
    std::vector<float> result;

    for (size_t expert_index = 0;
         expert_index < execution_experts.size(); ++expert_index) {
        const int current_slot = staged_slot;
        if (!engine.activate_device_slot(current_slot, err)) return false;
        auto release_current = [&]() {
            engine.release_device_slot(current_slot);
        };

        hits.clear();
        const int32_t expert = execution_experts[expert_index];
        for (int token = 0; token < batch.n_tokens; ++token) {
            float combined_weight = 0.0f;
            for (int rank = 0; rank < batch.top_k; ++rank) {
                const size_t route =
                    (size_t) token * (size_t) batch.top_k + (size_t) rank;
                if (batch.selected_ids[route] != expert) continue;
                combined_weight += batch.selected_weights[route];
            }
            if (!std::isfinite(combined_weight)) {
                if (err) *err = "combined expert route weight overflowed";
                release_current();
                return false;
            }
            if (combined_weight != 0.0f) {
                hits.push_back({token, combined_weight});
            }
        }
        if (hits.empty()) {
            release_current();
            continue;
        }

        size_t input_values = 0;
        if (!checked_mul_size((size_t) spec.input_dim, hits.size(),
                              input_values)) {
            if (err) *err = "streamed compact input size overflow";
            release_current();
            return false;
        }
        compact_input.resize(input_values);
        for (size_t i = 0; i < hits.size(); ++i) {
            const float * src = batch.inputs +
                (size_t) hits[i].token * (size_t) spec.input_dim;
            std::memcpy(compact_input.data() + i * (size_t) spec.input_dim,
                        src, sizeof(float) * (size_t) spec.input_dim);
        }

        std::unique_ptr<PersistentStreamExpertGraph> ephemeral;
        PersistentStreamExpertGraph * graph = nullptr;
        if (!acquire_graph((int) hits.size(), ephemeral, &graph)) {
            release_current();
            return false;
        }
        if (!validate_moe_stream_expert_layout(
                spec, runtime.device_slots[(size_t) current_slot].layout, err) ||
            !graph->launch(engine.scratch_gate_data(),
                           engine.scratch_up_data(),
                           engine.scratch_down_data(),
                           compact_input.data(), err)) {
            release_current();
            return false;
        }
        ++runtime.compute_stats.graph_launches;

        // Compute N is running while the already-admitted read for N+1 is
        // acquired and uploaded into a different, eviction-protected slot.
        if (expert_index + 1 < execution_experts.size()) {
            int next_slot = -1;
            if (!engine.stage_expert_cached_async(
                    batch.layer, execution_experts[expert_index + 1],
                    &next_slot, err)) {
                ggml_backend_synchronize(runtime.backend);
                release_current();
                return false;
            }
            staged_slot = next_slot;
        }

        if (!graph->finish(result, err)) {
            release_current();
            return false;
        }
        const size_t original_expert_index = cache_first_reordered
            ? (size_t) std::distance(
                unique_experts.begin(),
                std::lower_bound(
                    unique_experts.begin(), unique_experts.end(), expert))
            : 0;
        for (size_t i = 0; i < hits.size(); ++i) {
            float * dst = cache_first_reordered
                ? ordered_contributions.data() +
                    original_expert_index * output_values +
                    (size_t) hits[i].token * (size_t) spec.output_dim
                : out.data() +
                    (size_t) hits[i].token * (size_t) spec.output_dim;
            const float * src = result.data() +
                i * (size_t) spec.output_dim;
            const float weight = hits[i].weight;
            for (int j = 0; j < spec.output_dim; ++j) {
                dst[j] += weight * src[(size_t) j];
            }
        }
        release_current();
    }
    if (cache_first_reordered) {
        for (size_t expert_index = 0;
             expert_index < unique_experts.size(); ++expert_index) {
            const float * contribution = ordered_contributions.data() +
                expert_index * output_values;
            for (size_t value = 0; value < output_values; ++value) {
                out[value] += contribution[value];
            }
        }
    }
    return true;
}

namespace {

uint32_t stream_owner_hash(int layer, int expert) {
    uint32_t value = (uint32_t) expert + 0x9e3779b9U;
    value ^= (uint32_t) layer * 0x85ebca6bU;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

} // namespace

bool moe_stream_primary_owns_expert(
        const MoeStreamDualOwnerPolicy & policy,
        int layer,
        int expert) {
    if (layer < 0 || expert < 0 ||
        policy.primary_share_per_mille < 0 ||
        policy.primary_share_per_mille > 1000) {
        return false;
    }
    if (policy.primary_placement) {
        return policy.primary_placement->is_hot(layer, expert);
    }
    return stream_owner_hash(layer, expert) % 1000U <
           (uint32_t) policy.primary_share_per_mille;
}

bool partition_moe_stream_routes(
        const MoeStreamRouteBatch & batch,
        const MoeStreamDualOwnerPolicy & policy,
        std::vector<float> & primary_weights,
        std::vector<float> & secondary_weights,
        MoeStreamDualOwnerStats * stats,
        std::string * err) {
    if (batch.layer < 0 || batch.n_expert <= 0 || batch.top_k <= 0 ||
        batch.top_k > batch.n_expert || batch.n_tokens <= 0 ||
        !batch.selected_ids || !batch.selected_weights ||
        policy.primary_share_per_mille < 0 ||
        policy.primary_share_per_mille > 1000) {
        if (err) *err = "invalid dual-owner route batch or policy";
        return false;
    }
    if (policy.primary_placement &&
        (policy.primary_placement->n_layer <= batch.layer ||
         policy.primary_placement->n_expert != batch.n_expert)) {
        if (err) *err = "dual-owner placement does not match routed batch";
        return false;
    }

    size_t route_slots = 0;
    if (!checked_mul_size((size_t) batch.top_k,
                          (size_t) batch.n_tokens, route_slots)) {
        if (err) *err = "dual-owner route slot count overflow";
        return false;
    }
    primary_weights.assign(route_slots, 0.0f);
    secondary_weights.assign(route_slots, 0.0f);

    // -1 means unseen, 0 secondary, 1 primary. Duplicate appearances of one
    // expert in a batch must retain one owner so its cache remains coherent.
    std::vector<int8_t> owner((size_t) batch.n_expert, -1);
    std::vector<int32_t> unique_experts;
    unique_experts.reserve(std::min(route_slots, (size_t) batch.n_expert));
    int primary_experts = 0;
    int secondary_experts = 0;
    for (size_t route = 0; route < route_slots; ++route) {
        const int32_t expert = batch.selected_ids[route];
        const float weight = batch.selected_weights[route];
        if (expert < 0 || weight == 0.0f) continue;
        if (expert >= batch.n_expert || !std::isfinite(weight)) {
            if (err) *err = "native router produced an invalid dual-owner route";
            return false;
        }
        if (owner[(size_t) expert] >= 0) continue;
        const bool primary_owner = moe_stream_primary_owns_expert(
            policy, batch.layer, expert);
        owner[(size_t) expert] = primary_owner ? 1 : 0;
        unique_experts.push_back(expert);
        if (primary_owner) ++primary_experts;
        else ++secondary_experts;
    }

    int primary_routes = 0;
    int secondary_routes = 0;
    for (size_t route = 0; route < route_slots; ++route) {
        const int32_t expert = batch.selected_ids[route];
        const float weight = batch.selected_weights[route];
        if (expert < 0 || weight == 0.0f) continue;
        if (owner[(size_t) expert] == 1) {
            primary_weights[route] = weight;
            ++primary_routes;
        } else {
            secondary_weights[route] = weight;
            ++secondary_routes;
        }
    }
    if (stats) {
        stats->primary_routes = primary_routes;
        stats->secondary_routes = secondary_routes;
        stats->primary_experts = primary_experts;
        stats->secondary_experts = secondary_experts;
    }
    return true;
}

struct MoeStreamDualOwnerExecutor::Runtime {
    MoeHybridStreamEngine * primary = nullptr;
    MoeHybridStreamEngine * secondary = nullptr;
    std::mutex call_mutex;
    std::mutex work_mutex;
    std::condition_variable work_cv;
    std::condition_variable done_cv;
    std::thread worker;
    bool stop = false;
    bool pending = false;
    bool done = false;

    const MoeStreamExpertSpec * job_spec = nullptr;
    MoeStreamRouteBatch job_batch;
    std::vector<float> * job_out = nullptr;
    std::string * job_error = nullptr;
    bool job_ok = false;
    uint64_t job_us = 0;

    void worker_loop() {
        using Clock = std::chrono::steady_clock;
        for (;;) {
            const MoeStreamExpertSpec * spec = nullptr;
            MoeStreamRouteBatch batch;
            std::vector<float> * out = nullptr;
            std::string * error = nullptr;
            {
                std::unique_lock<std::mutex> lock(work_mutex);
                work_cv.wait(lock, [&]() { return stop || pending; });
                if (stop) return;
                spec = job_spec;
                batch = job_batch;
                out = job_out;
                error = job_error;
                pending = false;
            }

            const auto start = Clock::now();
            bool ok = false;
            try {
                ok = eval_moe_streamed_experts(
                    *secondary, *spec, batch, *out, error);
            } catch (const std::exception & ex) {
                *error = ex.what();
            } catch (...) {
                *error = "unknown secondary-owner exception";
            }
            const uint64_t elapsed_us = (uint64_t)
                std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - start).count();
            {
                std::lock_guard<std::mutex> lock(work_mutex);
                job_ok = ok;
                job_us = elapsed_us;
                done = true;
            }
            done_cv.notify_one();
        }
    }
};

MoeStreamDualOwnerExecutor::MoeStreamDualOwnerExecutor() = default;

MoeStreamDualOwnerExecutor::~MoeStreamDualOwnerExecutor() {
    destroy();
}

bool MoeStreamDualOwnerExecutor::init(
        MoeHybridStreamEngine & primary,
        MoeHybridStreamEngine & secondary,
        std::string * err) {
    destroy();
    if (!primary.is_bound() || !secondary.is_bound() ||
        !primary.compute_backend() || !secondary.compute_backend() ||
        primary.compute_backend() == secondary.compute_backend()) {
        if (err) *err = "dual-owner streaming requires two bound GPU backends";
        return false;
    }

    auto runtime = std::make_unique<Runtime>();
    runtime->primary = &primary;
    runtime->secondary = &secondary;
    try {
        Runtime * worker_runtime = runtime.get();
        runtime->worker = std::thread(
            [worker_runtime]() { worker_runtime->worker_loop(); });
    } catch (const std::exception & ex) {
        if (err) {
            *err = std::string("failed to start secondary owner worker: ") +
                ex.what();
        }
        return false;
    }
    runtime_ = std::move(runtime);
    return true;
}

bool MoeStreamDualOwnerExecutor::is_ready() const {
    return runtime_ != nullptr;
}

void MoeStreamDualOwnerExecutor::destroy() {
    if (!runtime_) return;
    auto runtime = std::move(runtime_);
    std::lock_guard<std::mutex> call_guard(runtime->call_mutex);
    {
        std::lock_guard<std::mutex> lock(runtime->work_mutex);
        runtime->stop = true;
    }
    runtime->work_cv.notify_one();
    if (runtime->worker.joinable()) runtime->worker.join();
}

bool MoeStreamDualOwnerExecutor::eval(
        const MoeStreamExpertSpec & spec,
        const MoeStreamRouteBatch & batch,
        const MoeStreamDualOwnerPolicy & policy,
        std::vector<float> & out,
        MoeStreamDualOwnerStats * stats,
        std::string * err) {
    if (!runtime_) {
        if (err) *err = "dual-owner executor is not initialized";
        return false;
    }
    Runtime & runtime = *runtime_;
    std::lock_guard<std::mutex> call_guard(runtime.call_mutex);
    if (!runtime.primary->is_bound() || !runtime.secondary->is_bound()) {
        if (err) *err = "dual-owner stream engine was destroyed";
        return false;
    }

    MoeStreamDualOwnerStats local_stats;
    std::vector<float> primary_weights;
    std::vector<float> secondary_weights;
    if (!partition_moe_stream_routes(
            batch, policy, primary_weights, secondary_weights,
            &local_stats, err)) {
        return false;
    }

    // A stable placement may legitimately route this token to only one GPU.
    // Bypass the worker rendezvous in that case; forcing synthetic work onto
    // both owners would hurt cache locality and small-top-k decode latency.
    if (local_stats.primary_routes == 0 ||
        local_stats.secondary_routes == 0) {
        const bool use_primary = local_stats.primary_routes != 0;
        MoeHybridStreamEngine & owner = use_primary
            ? *runtime.primary : *runtime.secondary;
        MoeStreamRouteBatch owner_batch = batch;
        owner_batch.selected_weights = use_primary
            ? primary_weights.data() : secondary_weights.data();
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        std::string owner_error;
        bool ok = false;
        try {
            ok = eval_moe_streamed_experts(
                owner, spec, owner_batch, out, &owner_error);
        } catch (const std::exception & ex) {
            owner_error = ex.what();
        } catch (...) {
            owner_error = "unknown single-owner exception";
        }
        const uint64_t elapsed_us = (uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start).count();
        local_stats.wall_us = elapsed_us;
        if (use_primary) local_stats.primary_us = elapsed_us;
        else local_stats.secondary_us = elapsed_us;
        if (stats) *stats = local_stats;
        if (!ok && err) {
            *err = std::string(use_primary ? "primary" : "secondary") +
                " owner failed: " + owner_error;
        }
        return ok;
    }

    MoeStreamRouteBatch primary_batch = batch;
    MoeStreamRouteBatch secondary_batch = batch;
    primary_batch.selected_weights = primary_weights.data();
    secondary_batch.selected_weights = secondary_weights.data();
    std::vector<float> primary_out;
    std::vector<float> secondary_out;
    std::string primary_error;
    std::string secondary_error;
    using Clock = std::chrono::steady_clock;
    const auto wall_start = Clock::now();

    {
        std::lock_guard<std::mutex> lock(runtime.work_mutex);
        runtime.job_spec = &spec;
        runtime.job_batch = secondary_batch;
        runtime.job_out = &secondary_out;
        runtime.job_error = &secondary_error;
        runtime.job_ok = false;
        runtime.job_us = 0;
        runtime.done = false;
        runtime.pending = true;
    }
    runtime.work_cv.notify_one();

    bool primary_ok = false;
    const auto primary_start = Clock::now();
    try {
        primary_ok = eval_moe_streamed_experts(
            *runtime.primary, spec, primary_batch,
            primary_out, &primary_error);
    } catch (const std::exception & ex) {
        primary_error = ex.what();
    } catch (...) {
        primary_error = "unknown primary-owner exception";
    }
    local_stats.primary_us = (uint64_t)
        std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - primary_start).count();

    bool secondary_ok = false;
    {
        std::unique_lock<std::mutex> lock(runtime.work_mutex);
        runtime.done_cv.wait(lock, [&]() { return runtime.done; });
        secondary_ok = runtime.job_ok;
        local_stats.secondary_us = runtime.job_us;
    }
    local_stats.wall_us = (uint64_t)
        std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - wall_start).count();

    if (!primary_ok || !secondary_ok) {
        if (err) {
            *err = !primary_ok
                ? std::string("primary owner failed: ") + primary_error
                : std::string("secondary owner failed: ") + secondary_error;
        }
        return false;
    }
    if (primary_out.size() != secondary_out.size()) {
        if (err) *err = "dual-owner partial sizes do not match";
        return false;
    }
    out.resize(primary_out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = primary_out[i] + secondary_out[i];
    }
    if (stats) *stats = local_stats;
    return true;
}

bool eval_moe_streamed_experts_dual_owner(
        MoeHybridStreamEngine & primary,
        MoeHybridStreamEngine & secondary,
        const MoeStreamExpertSpec & spec,
        const MoeStreamRouteBatch & batch,
        const MoeStreamDualOwnerPolicy & policy,
        std::vector<float> & out,
        MoeStreamDualOwnerStats * stats,
        std::string * err) {
    MoeStreamDualOwnerExecutor executor;
    if (!executor.init(primary, secondary, err)) return false;
    return executor.eval(spec, batch, policy, out, stats, err);
}

bool eval_moe_cold_experts_streaming(
        MoeHybridStreamEngine & engine,
        ggml_backend_t gpu_backend,
        const void * mmap_data,
        size_t mmap_size,
        const MoeHybridConfig & cfg,
        const MoeLayerDesc & desc,
        const LayerExpertRegions & regions,
        const MoeHybridLayerStorage & storage,
        const float * cur_host,
        const int32_t * selected_ids,
        const float * selected_weights,
        int n_tokens,
        std::vector<float> & out,
        std::string * err,
        int layer) {
    const char * reference = std::getenv("DFLASH_MOE_NVME_REFERENCE_EVAL");
    if (reference && (std::strcmp(reference, "1") == 0 ||
                      std::strcmp(reference, "on") == 0 ||
                      std::strcmp(reference, "true") == 0)) {
        return eval_moe_cold_experts_streaming_reference(
            engine, gpu_backend, mmap_data, mmap_size, cfg, desc, regions,
            storage, cur_host, selected_ids, selected_weights, n_tokens,
            out, err, layer);
    }

    int bound_layer = layer;
    if (!engine.is_bound()) {
        if (!mmap_data || mmap_size == 0 ||
            !engine.bind_sources({{mmap_data, mmap_size, -1}}, {regions}, err)) {
            if (err && err->empty()) *err = "stream engine has no model source";
            return false;
        }
        bound_layer = 0;
    }

    MoeStreamExpertSpec spec;
    if (!make_moe_stream_expert_spec(cfg, desc, regions, spec, err)) return false;
    MoeStreamRouteBatch route_batch;
    route_batch.layer = bound_layer;
    route_batch.n_expert = cfg.n_expert;
    route_batch.top_k = cfg.n_expert_used;
    route_batch.n_tokens = n_tokens;
    route_batch.inputs = cur_host;
    route_batch.selected_ids = selected_ids;
    route_batch.selected_weights = selected_weights;
    route_batch.resident_local_by_global =
        storage.hot_local_by_global.empty()
            ? nullptr : storage.hot_local_by_global.data();
    route_batch.resident_map_size = storage.hot_local_by_global.size();
    return eval_moe_streamed_experts(engine, spec, route_batch, out, err);
}

} // namespace dflash::common
