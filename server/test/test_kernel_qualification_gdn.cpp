#include "kernel_qualification.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace kq = lucebox::kernel_qualification;

namespace {

constexpr int S_V = 128;
constexpr int N_HEAD = 4;
constexpr int N_SEQS = 2;
constexpr int N_STATE_SLOTS = 2;
constexpr int N_STEPS = 3;
constexpr double OUTPUT_TOLERANCE = 5.0e-5;
constexpr double STATE_TOLERANCE = 2.0e-4;
constexpr size_t QKV_ELEMENTS = (size_t) S_V*N_HEAD*N_SEQS;
constexpr size_t GATE_ELEMENTS = (size_t) N_HEAD*N_SEQS;
constexpr size_t STATE_ELEMENTS =
    (size_t) S_V*S_V*N_HEAD*N_STATE_SLOTS;
constexpr size_t GUARD_ELEMENTS = 64;

bool set_environment(const char * name, const char * value) {
#ifdef _WIN32
    return _putenv_s(name, value ? value : "") == 0;
#else
    return value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0;
#endif
}

class EnvironmentValue {
public:
    EnvironmentValue(const char * name, const char * value) : name_(name) {
        const char * previous = std::getenv(name);
        if (previous) {
            had_previous_ = true;
            previous_ = previous;
        }
        valid_ = set_environment(name, value);
    }

    ~EnvironmentValue() {
        set_environment(name_.c_str(),
                        had_previous_ ? previous_.c_str() : nullptr);
    }

    bool valid() const { return valid_; }

    EnvironmentValue(const EnvironmentValue &) = delete;
    EnvironmentValue & operator=(const EnvironmentValue &) = delete;

private:
    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
    bool valid_ = false;
};

struct StepInput {
    std::vector<float> q = std::vector<float>(QKV_ELEMENTS);
    std::vector<float> k = std::vector<float>(QKV_ELEMENTS);
    std::vector<float> v = std::vector<float>(QKV_ELEMENTS);
    std::vector<float> g = std::vector<float>(GATE_ELEMENTS);
    std::vector<float> beta = std::vector<float>(GATE_ELEMENTS);
};

struct StepOutput {
    bool computed = false;
    std::string reason;
    std::vector<float> output = std::vector<float>(QKV_ELEMENTS);
    std::vector<float> state = std::vector<float>(STATE_ELEMENTS);
    std::vector<float> state_guard_before = std::vector<float>(GUARD_ELEMENTS);
    std::vector<float> state_guard_after = std::vector<float>(GUARD_ELEMENTS);
    std::vector<float> output_guard_before = std::vector<float>(GUARD_ELEMENTS);
    std::vector<float> output_guard_after = std::vector<float>(GUARD_ELEMENTS);
};

struct PathRun {
    bool computed = false;
    std::string reason;
    size_t scalar_launches = 0;
    size_t grouped_launches = 0;
    std::vector<StepOutput> steps;
};

void fill_uniform(
        std::mt19937 & rng,
        float low,
        float high,
        std::vector<float> & values) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float & value : values) value = distribution(rng);
}

std::vector<StepInput> make_inputs() {
    std::mt19937 rng(20260901);
    std::vector<StepInput> steps(N_STEPS);
    for (StepInput & step : steps) {
        fill_uniform(rng, -0.25f, 0.25f, step.q);
        fill_uniform(rng, -0.25f, 0.25f, step.k);
        fill_uniform(rng, -0.25f, 0.25f, step.v);
        fill_uniform(rng, -1.0f, -0.02f, step.g);
        fill_uniform(rng, 0.05f, 0.95f, step.beta);

        for (int sequence = 0; sequence < N_SEQS; ++sequence) {
            for (int head = 0; head < N_HEAD; ++head) {
                const size_t base =
                    ((size_t) sequence*N_HEAD + head)*S_V;
                for (std::vector<float> * values : {&step.q, &step.k}) {
                    float norm2 = 0.0f;
                    for (int i = 0; i < S_V; ++i) {
                        const float value = (*values)[base + i];
                        norm2 += value*value;
                    }
                    const float inverse_norm = 1.0f/std::sqrt(norm2);
                    for (int i = 0; i < S_V; ++i) {
                        (*values)[base + i] *= inverse_norm;
                    }
                }
            }
        }
    }
    return steps;
}

std::vector<float> make_initial_state() {
    std::mt19937 rng(20260902);
    std::vector<float> state(STATE_ELEMENTS);
    fill_uniform(rng, -0.06f, 0.06f, state);
    return state;
}

std::vector<float> make_guard(float offset) {
    std::vector<float> guard(GUARD_ELEMENTS);
    for (size_t i = 0; i < guard.size(); ++i) {
        guard[i] = offset + static_cast<float>(i)*0.125f;
    }
    return guard;
}

size_t padded_allocation_size(const ggml_tensor * tensor) {
    const size_t alignment =
        ggml_backend_buffer_get_alignment(tensor->buffer);
    const size_t size =
        ggml_backend_buffer_get_alloc_size(tensor->buffer, tensor);
    return (size + alignment - 1) & ~(alignment - 1);
}

bool guards_are_adjacent(
        const ggml_tensor * before,
        const ggml_tensor * guarded,
        const ggml_tensor * after) {
    if (!before->buffer || before->buffer != guarded->buffer ||
        guarded->buffer != after->buffer ||
        !before->data || !guarded->data || !after->data) {
        return false;
    }
    const uintptr_t before_address =
        reinterpret_cast<uintptr_t>(before->data);
    const uintptr_t guarded_address =
        reinterpret_cast<uintptr_t>(guarded->data);
    const uintptr_t after_address =
        reinterpret_cast<uintptr_t>(after->data);
    return before_address + padded_allocation_size(before) ==
            guarded_address &&
        guarded_address + padded_allocation_size(guarded) == after_address;
}

StepOutput run_step(
        ggml_backend_t backend,
        const StepInput & input,
        const std::vector<float> & state_input) {
    StepOutput result;
    ggml_init_params params{};
    params.mem_size = 4*1024*1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (!context) return result;

    ggml_tensor * q = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, S_V, N_HEAD, 1, N_SEQS);
    ggml_tensor * k = ggml_dup_tensor(context, q);
    ggml_tensor * v = ggml_dup_tensor(context, q);
    ggml_tensor * g = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, 1, N_HEAD, 1, N_SEQS);
    ggml_tensor * beta = ggml_dup_tensor(context, g);
    ggml_tensor * state_guard_before =
        ggml_new_tensor_1d(context, GGML_TYPE_F32, GUARD_ELEMENTS);
    ggml_tensor * state = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, S_V, S_V, N_HEAD, N_STATE_SLOTS);
    ggml_tensor * state_guard_after =
        ggml_new_tensor_1d(context, GGML_TYPE_F32, GUARD_ELEMENTS);
    ggml_tensor * active_slots =
        ggml_new_tensor_1d(context, GGML_TYPE_I32, N_SEQS);
    ggml_tensor * output_guard_before =
        ggml_new_tensor_1d(context, GGML_TYPE_F32, GUARD_ELEMENTS);
    for (ggml_tensor * tensor :
         {q, k, v, g, beta, state_guard_before, state,
          state_guard_after, active_slots, output_guard_before}) {
        ggml_set_input(tensor);
    }

    ggml_tensor * output = ggml_gated_delta_net_active_inplace(
        context, q, k, v, g, beta, state, active_slots);
    ggml_gated_delta_net_set_skip_intermediate(output, true);
    ggml_set_output(output);
    ggml_tensor * output_guard_after =
        ggml_new_tensor_1d(context, GGML_TYPE_F32, GUARD_ELEMENTS);
    ggml_set_input(output_guard_after);

    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) {
        ggml_free(context);
        return result;
    }
    if (!guards_are_adjacent(
            state_guard_before, state, state_guard_after) ||
        !guards_are_adjacent(
            output_guard_before, output, output_guard_after)) {
        result.reason =
            "allocation guards are not adjacent to state/output";
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return result;
    }

    const std::vector<float> expected_state_guard_before = make_guard(1000.0f);
    const std::vector<float> expected_state_guard_after = make_guard(-1000.0f);
    const std::vector<float> expected_output_guard_before = make_guard(2000.0f);
    const std::vector<float> expected_output_guard_after = make_guard(-2000.0f);
    const std::array<int32_t, N_SEQS> slot_ids{1, 0};
    ggml_backend_tensor_set(
        state_guard_before, expected_state_guard_before.data(), 0,
        expected_state_guard_before.size()*sizeof(float));
    ggml_backend_tensor_set(q, input.q.data(), 0, input.q.size()*sizeof(float));
    ggml_backend_tensor_set(k, input.k.data(), 0, input.k.size()*sizeof(float));
    ggml_backend_tensor_set(v, input.v.data(), 0, input.v.size()*sizeof(float));
    ggml_backend_tensor_set(g, input.g.data(), 0, input.g.size()*sizeof(float));
    ggml_backend_tensor_set(
        beta, input.beta.data(), 0, input.beta.size()*sizeof(float));
    ggml_backend_tensor_set(
        state, state_input.data(), 0, state_input.size()*sizeof(float));
    ggml_backend_tensor_set(
        state_guard_after, expected_state_guard_after.data(), 0,
        expected_state_guard_after.size()*sizeof(float));
    ggml_backend_tensor_set(
        active_slots, slot_ids.data(), 0, slot_ids.size()*sizeof(int32_t));
    ggml_backend_tensor_set(
        output_guard_before, expected_output_guard_before.data(), 0,
        expected_output_guard_before.size()*sizeof(float));
    ggml_backend_tensor_set(
        output_guard_after, expected_output_guard_after.data(), 0,
        expected_output_guard_after.size()*sizeof(float));

    result.computed =
        ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (result.computed) {
        ggml_backend_tensor_get(
            output, result.output.data(), 0, result.output.size()*sizeof(float));
        ggml_backend_tensor_get(
            state, result.state.data(), 0, result.state.size()*sizeof(float));
        ggml_backend_tensor_get(
            state_guard_before, result.state_guard_before.data(), 0,
            result.state_guard_before.size()*sizeof(float));
        ggml_backend_tensor_get(
            state_guard_after, result.state_guard_after.data(), 0,
            result.state_guard_after.size()*sizeof(float));
        ggml_backend_tensor_get(
            output_guard_before, result.output_guard_before.data(), 0,
            result.output_guard_before.size()*sizeof(float));
        ggml_backend_tensor_get(
            output_guard_after, result.output_guard_after.data(), 0,
            result.output_guard_after.size()*sizeof(float));
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return result;
}

PathRun run_steps(
        ggml_backend_t backend,
        const std::vector<StepInput> & inputs,
        const std::vector<float> & initial_state) {
    PathRun result;
    std::vector<float> state = initial_state;
    for (const StepInput & input : inputs) {
        StepOutput step = run_step(backend, input, state);
        if (!step.computed) {
            result.reason = step.reason.empty()
                ? "backend graph compute failed"
                : step.reason;
            return result;
        }
        state = step.state;
        result.steps.push_back(std::move(step));
    }
    result.computed = true;
    return result;
}

PathRun run_path(
        ggml_backend_t backend,
        const std::vector<StepInput> & inputs,
        const std::vector<float> & initial_state,
        bool grouped) {
    EnvironmentValue force_grouped(
        "DFLASH_GDN_FORCE_GROUPED_COLS", grouped ? "1" : nullptr);
    EnvironmentValue disable_grouped(
        "DFLASH_GDN_NO_GROUPED_COLS", grouped ? nullptr : "1");
    if (!force_grouped.valid() || !disable_grouped.valid()) {
        PathRun result;
        result.reason = "failed to select GDN route";
        return result;
    }

    const size_t scalar_before =
        ggml_backend_cuda_get_gdn_scalar_launch_count();
    const size_t grouped_before =
        ggml_backend_cuda_get_gdn_grouped_cols_launch_count();
    PathRun result = run_steps(backend, inputs, initial_state);
    result.scalar_launches =
        ggml_backend_cuda_get_gdn_scalar_launch_count() - scalar_before;
    result.grouped_launches =
        ggml_backend_cuda_get_gdn_grouped_cols_launch_count() - grouped_before;
    return result;
}

kq::Metric failed_metric(const std::string & reason) {
    kq::Metric metric;
    metric.name = "compute";
    metric.finite = false;
    metric.reason = reason;
    return metric;
}

kq::CaseResult qualify_gdn(ggml_backend_t backend) {
    const std::vector<StepInput> inputs = make_inputs();
    const std::vector<float> initial_state = make_initial_state();
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    PathRun oracle;
    if (cpu_backend) {
        oracle = run_steps(cpu_backend, inputs, initial_state);
    } else {
        oracle.reason = "backend unavailable";
    }
    if (cpu_backend) ggml_backend_free(cpu_backend);
    const bool previous_graphs_disabled =
        ggml_backend_cuda_set_graphs_disabled_override(true);
    const PathRun reference = run_path(backend, inputs, initial_state, false);
    const PathRun candidate = run_path(backend, inputs, initial_state, true);
    ggml_backend_cuda_set_graphs_disabled_override(previous_graphs_disabled);

    const bool route_matched = reference.computed && candidate.computed &&
        reference.scalar_launches == N_STEPS &&
        reference.grouped_launches == 0 &&
        candidate.scalar_launches == 0 &&
        candidate.grouped_launches == N_STEPS;
    const std::string observed =
        "reference.scalar=" + std::to_string(reference.scalar_launches) +
        ",reference.grouped_cols=" +
        std::to_string(reference.grouped_launches) +
        ",candidate.scalar=" + std::to_string(candidate.scalar_launches) +
        ",candidate.grouped_cols=" +
        std::to_string(candidate.grouped_launches);
    kq::Route route{
        "reference.scalar=" + std::to_string(N_STEPS) +
            ",candidate.grouped_cols=" + std::to_string(N_STEPS),
        observed, route_matched};

    if (!oracle.computed || !reference.computed || !candidate.computed) {
        const std::string reason = !oracle.computed
            ? "CPU oracle: " + oracle.reason
            : !reference.computed
            ? "reference: " + reference.reason
            : "candidate: " + candidate.reason;
        return kq::evaluate(
            "gdn.active_inplace.eager.gpu_routes_vs_cpu",
            {failed_metric(reason)}, std::move(route));
    }

    std::vector<kq::Metric> metrics;
    const std::vector<float> expected_state_guard_before = make_guard(1000.0f);
    const std::vector<float> expected_state_guard_after = make_guard(-1000.0f);
    const std::vector<float> expected_output_guard_before = make_guard(2000.0f);
    const std::vector<float> expected_output_guard_after = make_guard(-2000.0f);
    for (size_t step = 0; step < inputs.size(); ++step) {
        const std::string prefix = "step" + std::to_string(step) + ".";
        metrics.push_back(kq::compare_f32(
            prefix + "scalar.output", reference.steps[step].output,
            oracle.steps[step].output, OUTPUT_TOLERANCE));
        metrics.push_back(kq::compare_f32(
            prefix + "scalar.state", reference.steps[step].state,
            oracle.steps[step].state, STATE_TOLERANCE));
        metrics.push_back(kq::compare_f32(
            prefix + "grouped.output", candidate.steps[step].output,
            oracle.steps[step].output, OUTPUT_TOLERANCE));
        metrics.push_back(kq::compare_f32(
            prefix + "grouped.state", candidate.steps[step].state,
            oracle.steps[step].state, STATE_TOLERANCE));
        metrics.push_back(kq::compare_f32(
            prefix + "reference.state_guard_before",
            reference.steps[step].state_guard_before,
            expected_state_guard_before, 0.0));
        metrics.push_back(kq::compare_f32(
            prefix + "reference.state_guard_after",
            reference.steps[step].state_guard_after,
            expected_state_guard_after, 0.0));
        metrics.push_back(kq::compare_f32(
            prefix + "reference.output_guard_before",
            reference.steps[step].output_guard_before,
            expected_output_guard_before, 0.0));
        metrics.push_back(kq::compare_f32(
            prefix + "reference.output_guard_after",
            reference.steps[step].output_guard_after,
            expected_output_guard_after, 0.0));
        metrics.push_back(kq::compare_f32(
            prefix + "candidate.state_guard_before",
            candidate.steps[step].state_guard_before,
            expected_state_guard_before, 0.0));
        metrics.push_back(kq::compare_f32(
            prefix + "candidate.state_guard_after",
            candidate.steps[step].state_guard_after,
            expected_state_guard_after, 0.0));
        metrics.push_back(kq::compare_f32(
            prefix + "candidate.output_guard_before",
            candidate.steps[step].output_guard_before,
            expected_output_guard_before, 0.0));
        metrics.push_back(kq::compare_f32(
            prefix + "candidate.output_guard_after",
            candidate.steps[step].output_guard_after,
            expected_output_guard_after, 0.0));
    }
    return kq::evaluate(
        "gdn.active_inplace.eager.gpu_routes_vs_cpu",
        std::move(metrics), std::move(route));
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        const std::vector<kq::CaseResult> cases{kq::unsupported(
            "gdn.active_inplace.eager.gpu_routes_vs_cpu",
            "CUDA/HIP backend unavailable")};
        kq::write_json(std::cout, "unavailable", cases);
        return kq::exit_code(cases);
    }

    char description[256] = {};
    ggml_backend_cuda_get_device_description(
        0, description, sizeof(description));
    if (!ggml_backend_cuda_supports_gdn_grouped_cols(0)) {
        const std::vector<kq::CaseResult> cases{kq::unsupported(
            "gdn.active_inplace.eager.gpu_routes_vs_cpu",
            "grouped-column GDN route unsupported on this device")};
        kq::write_json(std::cout, description, cases);
        ggml_backend_free(backend);
        return kq::exit_code(cases);
    }

    const std::vector<kq::CaseResult> cases{qualify_gdn(backend)};
    kq::write_json(std::cout, description, cases);
    ggml_backend_free(backend);
    return kq::exit_code(cases);
}
