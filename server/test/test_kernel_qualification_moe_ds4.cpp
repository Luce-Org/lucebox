#include "kernel_qualification.h"

#include "ds4_test_gpu_runtime.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "rocmfpx.h"

// Keep this representative rather than reproducing the large standalone test
// matrices. The MoE case qualifies one fused mixed-expert gate/up launch with
// invalid route sentinels and allocation guards. The DeepSeek4 case qualifies
// one speculative-width grouped MMID launch against the disabled-grouped path.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#else
#include <cerrno>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

bool ggml_cuda_rocmfp2_mix_mul_mat_id(
    const void * vx, const float * src1, const int32_t * ids, float * dst,
    int in, int out, int n_expert_used, int n_tokens, int ne11,
    int64_t ids_s0, int64_t ids_s1,
    int64_t src1_s1, int64_t src1_s2,
    int64_t dst_s1, int64_t dst_s2, cudaStream_t stream);

bool ggml_cuda_rocmfp2_mix_mul_mat_id_glu(
    const void * vx_up, const void * vx_gate,
    const float * src1, const int32_t * ids, float * dst,
    int in, int out, int n_expert_used, int n_tokens, int ne11,
    int64_t ids_s0, int64_t ids_s1,
    int64_t src1_s1, int64_t src1_s2,
    int64_t dst_s1, int64_t dst_s2,
    float glu_limit, cudaStream_t stream);

namespace kq = lucebox::kernel_qualification;

namespace {

constexpr const char * MOE_CASE = "moe.rocmfp2_mix.gate_up_glu";
constexpr const char * DS4_CASE = "deepseek4.grouped_mmid.rocmfp2";

kq::Metric failed_metric(const std::string & reason) {
    kq::Metric metric;
    metric.name = "compute";
    metric.finite = false;
    metric.reason = reason;
    return metric;
}

kq::CaseResult failed_case(
        const char * name,
        const std::string & reason,
        kq::Route route) {
    return kq::evaluate(name, {failed_metric(reason)}, std::move(route));
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }

    ~DeviceBuffer() {
        if (data_) cudaFree(data_);
    }

    bool allocate(size_t count) {
        if (data_) return false;
        count_ = count;
        return cudaMalloc(reinterpret_cast<void **>(&data_),
                          count*sizeof(T)) == cudaSuccess;
    }

    bool copy_from(const std::vector<T> & values) {
        return values.size() <= count_ &&
            cudaMemcpy(data_, values.data(), values.size()*sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess;
    }

    bool copy_to(std::vector<T> & values) const {
        return values.size() <= count_ &&
            cudaMemcpy(values.data(), data_, values.size()*sizeof(T),
                       cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    T * data() { return data_; }
    const T * data() const { return data_; }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;

private:
    T * data_ = nullptr;
    size_t count_ = 0;
};

class MixRegistration {
public:
    MixRegistration() = default;
    ~MixRegistration() {
        if (base_) ggml_cuda_rocmfp2_mix_unregister(base_);
    }

    bool register_host(
            const void * base,
            size_t expert_stride,
            int n_experts,
            int out,
            int in,
            const std::vector<uint16_t> & codebooks,
            const std::vector<uint8_t> & modes) {
        if (!ggml_cuda_rocmfp2_mix_register_host(
                base, expert_stride, n_experts, out, in,
                codebooks.data(), modes.data())) {
            return false;
        }
        base_ = base;
        return true;
    }

    MixRegistration(const MixRegistration &) = delete;
    MixRegistration & operator=(const MixRegistration &) = delete;

private:
    const void * base_ = nullptr;
};

uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = ((bits >> 16) & 1U) + 0x7FFFU;
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

float host_swiglu_ds4(float gate, float up, float limit) {
    gate = std::fmin(gate, limit);
    up = std::fmax(std::fmin(up, limit), -limit);
    return gate/(1.0f + std::exp(-gate))*up;
}

std::vector<float> make_guard(float offset, size_t count) {
    std::vector<float> guard(count);
    for (size_t i = 0; i < count; ++i) {
        guard[i] = offset + static_cast<float>(i)*0.125f;
    }
    return guard;
}

kq::CaseResult qualify_moe_gate_up_glu() {
    constexpr int qk = 32;
    constexpr int block_bytes = 10;
    constexpr int levels = 4;
    constexpr int in = 256;
    constexpr int out = 64;
    constexpr int n_experts = 6;
    constexpr int n_used = 3;
    constexpr int n_tokens = 2;
    constexpr size_t guard_elements = 32;
    constexpr float limit = 7.0f;
    const size_t row_bytes = static_cast<size_t>(in/qk)*block_bytes;
    const size_t expert_bytes = static_cast<size_t>(out)*row_bytes;
    const size_t weight_bytes = expert_bytes*n_experts;
    const size_t input_elements = static_cast<size_t>(in)*n_tokens;
    const size_t output_elements = static_cast<size_t>(out)*n_used*n_tokens;

    std::mt19937 rng(20260903);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    std::vector<uint8_t> weights_up(weight_bytes);
    std::vector<uint8_t> weights_gate(weight_bytes);
    for (uint8_t & value : weights_up) {
        value = static_cast<uint8_t>(byte_distribution(rng));
    }
    for (uint8_t & value : weights_gate) {
        value = static_cast<uint8_t>(byte_distribution(rng));
    }
    for (size_t block = 0; block < weight_bytes/block_bytes; ++block) {
        for (std::vector<uint8_t> * weights : {&weights_up, &weights_gate}) {
            (*weights)[block*block_bytes + 8] = static_cast<uint8_t>(
                0x30U | ((*weights)[block*block_bytes + 8] & 0x80U));
            (*weights)[block*block_bytes + 9] = static_cast<uint8_t>(
                0x30U | ((*weights)[block*block_bytes + 9] & 0x80U));
        }
    }

    std::vector<uint16_t> books_up(n_experts*2*levels);
    std::vector<uint16_t> books_gate(n_experts*2*levels);
    for (size_t i = 0; i < books_up.size(); ++i) {
        books_up[i] = f32_to_bf16(-1.0f + 0.37f*static_cast<float>(i % 7));
        books_gate[i] = f32_to_bf16(0.5f - 0.21f*static_cast<float>(i % 5));
    }
    const std::vector<uint8_t> modes(n_experts, 1);

    std::vector<float> input(input_elements);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = -0.75f + 0.03f*static_cast<float>(i % 51);
    }
    const std::vector<int32_t> ids{-1, 1, 3, n_experts, 5, 2};
    const std::vector<float> guard_before = make_guard(1000.0f, guard_elements);
    const std::vector<float> guard_after = make_guard(-1000.0f, guard_elements);
    std::vector<float> guarded_output(
        guard_elements + output_elements + guard_elements);
    std::copy(guard_before.begin(), guard_before.end(), guarded_output.begin());
    std::copy(guard_after.begin(), guard_after.end(),
              guarded_output.begin() + guard_elements + output_elements);

    DeviceBuffer<uint8_t> device_up(weight_bytes);
    DeviceBuffer<uint8_t> device_gate(weight_bytes);
    DeviceBuffer<float> device_input(input_elements);
    DeviceBuffer<int32_t> device_ids(ids.size());
    DeviceBuffer<float> device_up_output(output_elements);
    DeviceBuffer<float> device_gate_output(output_elements);
    DeviceBuffer<float> device_guarded_output(guarded_output.size());
    const bool allocated = device_up.data() && device_gate.data() &&
        device_input.data() && device_ids.data() && device_up_output.data() &&
        device_gate_output.data() && device_guarded_output.data();
    const bool copied = allocated && device_up.copy_from(weights_up) &&
        device_gate.copy_from(weights_gate) && device_input.copy_from(input) &&
        device_ids.copy_from(ids) && device_guarded_output.copy_from(guarded_output);
    if (!copied) {
        return failed_case(
            MOE_CASE, "device allocation or copy failed",
            {"unfused.up=1,unfused.gate=1,fused=1", "setup_failed", false});
    }

    MixRegistration up_registration;
    MixRegistration gate_registration;
    if (!up_registration.register_host(
            device_up.data(), expert_bytes, n_experts, out, in,
            books_up, modes) ||
        !gate_registration.register_host(
            device_gate.data(), expert_bytes, n_experts, out, in,
            books_gate, modes)) {
        return failed_case(
            MOE_CASE, "mixed expert registration failed",
            {"unfused.up=1,unfused.gate=1,fused=1", "registration_failed", false});
    }

    constexpr int64_t ids_s0 = 1;
    constexpr int64_t ids_s1 = n_used;
    constexpr int64_t input_s1 = 0;
    constexpr int64_t input_s2 = in;
    constexpr int64_t output_s1 = out;
    constexpr int64_t output_s2 = static_cast<int64_t>(out)*n_used;
    const bool up_launched = ggml_cuda_rocmfp2_mix_mul_mat_id(
        device_up.data(), device_input.data(), device_ids.data(),
        device_up_output.data(), in, out, n_used, n_tokens, 1,
        ids_s0, ids_s1, input_s1, input_s2, output_s1, output_s2, nullptr);
    const bool gate_launched = ggml_cuda_rocmfp2_mix_mul_mat_id(
        device_gate.data(), device_input.data(), device_ids.data(),
        device_gate_output.data(), in, out, n_used, n_tokens, 1,
        ids_s0, ids_s1, input_s1, input_s2, output_s1, output_s2, nullptr);
    float * candidate = device_guarded_output.data() + guard_elements;
    const bool fused_launched = ggml_cuda_rocmfp2_mix_mul_mat_id_glu(
        device_up.data(), device_gate.data(), device_input.data(),
        device_ids.data(), candidate, in, out, n_used, n_tokens, 1,
        ids_s0, ids_s1, input_s1, input_s2, output_s1, output_s2,
        limit, nullptr);
    const bool synchronized = cudaDeviceSynchronize() == cudaSuccess;
    const bool route_matched =
        up_launched && gate_launched && fused_launched && synchronized;
    const std::string observed =
        "unfused.up=" + std::to_string(up_launched ? 1 : 0) +
        ",unfused.gate=" + std::to_string(gate_launched ? 1 : 0) +
        ",fused=" + std::to_string(fused_launched ? 1 : 0) +
        ",sync=" + std::to_string(synchronized ? 1 : 0);
    kq::Route route{
        "unfused.up=1,unfused.gate=1,fused=1,sync=1",
        observed, route_matched};
    if (!route_matched) {
        return failed_case(MOE_CASE, "kernel launch failed", std::move(route));
    }

    std::vector<float> up_output(output_elements);
    std::vector<float> gate_output(output_elements);
    if (!device_up_output.copy_to(up_output) ||
        !device_gate_output.copy_to(gate_output) ||
        !device_guarded_output.copy_to(guarded_output)) {
        return failed_case(MOE_CASE, "device result copy failed", std::move(route));
    }
    std::vector<float> fused_output(
        guarded_output.begin() + guard_elements,
        guarded_output.begin() + guard_elements + output_elements);
    std::vector<float> reference(output_elements);
    for (size_t i = 0; i < reference.size(); ++i) {
        reference[i] = host_swiglu_ds4(gate_output[i], up_output[i], limit);
    }
    std::vector<float> actual_guard_before(
        guarded_output.begin(), guarded_output.begin() + guard_elements);
    std::vector<float> actual_guard_after(
        guarded_output.begin() + guard_elements + output_elements,
        guarded_output.end());
    std::vector<float> sentinel_output;
    for (size_t route = 0; route < ids.size(); ++route) {
        if (ids[route] >= 0 && ids[route] < n_experts) continue;
        const size_t offset = route*static_cast<size_t>(out);
        sentinel_output.insert(
            sentinel_output.end(),
            fused_output.begin() + offset,
            fused_output.begin() + offset + out);
    }
    const std::vector<float> sentinel_reference(sentinel_output.size(), 0.0f);

    return kq::evaluate(
        MOE_CASE,
        {
            kq::compare_f32("output", fused_output, reference, 1.0e-4),
            kq::compare_f32(
                "invalid_route_zero", sentinel_output, sentinel_reference, 0.0),
            kq::compare_f32(
                "guard_before", actual_guard_before, guard_before, 0.0),
            kq::compare_f32(
                "guard_after", actual_guard_after, guard_after, 0.0),
        },
        std::move(route));
}

struct GraphRun {
    bool computed = false;
    size_t grouped_launches = 0;
    std::vector<float> output;
};

GraphRun run_grouped_mmid_graph(ggml_backend_t backend) {
    constexpr int k_dim = 256;
    constexpr int n_rows = 128;
    constexpr int n_experts = 32;
    constexpr int top_k = 8;
    constexpr int width = 8;
    GraphRun run;

    ggml_init_params params{16*1024*1024, nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) return run;
    ggml_tensor * weights = ggml_new_tensor_3d(
        context, GGML_TYPE_Q2_0_ROCMFP2, k_dim, n_rows, n_experts);
    ggml_tensor * input = ggml_new_tensor_3d(
        context, GGML_TYPE_F32, k_dim, 1, width);
    ggml_tensor * ids = ggml_new_tensor_2d(
        context, GGML_TYPE_I32, top_k, width);
    for (ggml_tensor * tensor : {weights, input, ids}) ggml_set_input(tensor);
    ggml_tensor * result = ggml_mul_mat_id(context, weights, input, ids);
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, result);

    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) {
        ggml_gallocr_free(allocator);
        ggml_free(context);
        return run;
    }

    std::mt19937 rng(20260904);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> weights_f(
        static_cast<size_t>(k_dim)*n_rows*n_experts);
    std::vector<float> input_f(static_cast<size_t>(k_dim)*width);
    for (float & value : weights_f) value = distribution(rng);
    for (float & value : input_f) value = distribution(rng);
    std::vector<uint8_t> weights_q(ggml_nbytes(weights));
    const size_t quantized = rocmfpx_quantize_fp2(
        weights_f.data(), weights_q.data(), n_rows*n_experts, k_dim, nullptr);
    std::vector<int32_t> ids_f(static_cast<size_t>(top_k)*width);
    for (int token = 0; token < width; ++token) {
        for (int slot = 0; slot < top_k; ++slot) {
            ids_f[static_cast<size_t>(token)*top_k + slot] =
                (token*3 + slot*5) % n_experts;
        }
    }

    if (quantized == weights_q.size()) {
        ggml_backend_tensor_set(weights, weights_q.data(), 0, weights_q.size());
        ggml_backend_tensor_set(
            input, input_f.data(), 0, input_f.size()*sizeof(float));
        ggml_backend_tensor_set(
            ids, ids_f.data(), 0, ids_f.size()*sizeof(int32_t));
        const size_t grouped_before =
            ggml_backend_cuda_get_mmvq_mmid_grouped_launch_count();
        const bool graphs_were_disabled =
            ggml_backend_cuda_set_graphs_disabled_override(true);
        run.computed =
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_cuda_set_graphs_disabled_override(graphs_were_disabled);
        run.grouped_launches =
            ggml_backend_cuda_get_mmvq_mmid_grouped_launch_count() -
            grouped_before;
        if (run.computed) {
            run.output.resize(ggml_nelements(result));
            ggml_backend_tensor_get(
                result, run.output.data(), 0, run.output.size()*sizeof(float));
        }
    }

    ggml_gallocr_free(allocator);
    ggml_free(context);
    return run;
}

bool write_binary(const std::string & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size()*sizeof(float)));
    return output.good();
}

bool write_count(const std::string & path, size_t count) {
    std::ofstream output(path, std::ios::trunc);
    output << count << '\n';
    return output.good();
}

std::vector<float> read_binary(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streamsize bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
        return {};
    }
    input.seekg(0);
    std::vector<float> values(static_cast<size_t>(bytes)/sizeof(float));
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return input ? values : std::vector<float>{};
}

size_t read_count(const std::string & path) {
    std::ifstream input(path);
    size_t count = 0;
    return input >> count ? count : static_cast<size_t>(-1);
}

std::string shell_quote(const std::string & value) {
#if defined(_WIN32)
    std::string quoted = "\"";
    for (char ch : value) {
        if (ch == '"') quoted += '\\';
        quoted += ch;
    }
    return quoted + "\"";
#else
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') quoted += "'\\''";
        else quoted += ch;
    }
    return quoted + "'";
#endif
}

std::string child_command(
        const std::string & executable,
        bool grouped,
        const std::string & output_path,
        const std::string & count_path) {
#if defined(_WIN32)
    return "set \"DFLASH_MMID_GROUPED_TYPES=15\" && set \"DFLASH_MMID_GROUPED=" +
        std::string(grouped ? "1" : "0") + "\" && " + shell_quote(executable) +
        " --child " + shell_quote(output_path) + " " + shell_quote(count_path);
#else
    return "DFLASH_MMID_GROUPED_TYPES=15 DFLASH_MMID_GROUPED=" +
        std::string(grouped ? "1" : "0") + " " + shell_quote(executable) +
        " --child " + shell_quote(output_path) + " " + shell_quote(count_path);
#endif
}

int run_command_with_timeout(
        const std::string & command,
        std::chrono::seconds timeout) {
#if defined(_WIN32)
    std::string command_line = "cmd.exe /S /C " + command;
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessA(
            nullptr, command_line.data(), nullptr, nullptr, FALSE,
            CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process)) {
        return 1;
    }
    const DWORD wait_status = WaitForSingleObject(
        process.hProcess, static_cast<DWORD>(timeout.count()*1000));
    int status = 1;
    if (wait_status == WAIT_TIMEOUT) {
        TerminateProcess(process.hProcess, 124);
        WaitForSingleObject(process.hProcess, 5000);
        status = 124;
    } else if (wait_status == WAIT_OBJECT_0) {
        DWORD exit_code = 1;
        if (GetExitCodeProcess(process.hProcess, &exit_code)) {
            status = static_cast<int>(exit_code);
        }
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return status;
#else
    const pid_t child = fork();
    if (child < 0) return 1;
    if (child == 0) {
        execl("/bin/sh", "sh", "-c", command.c_str(), nullptr);
        _exit(127);
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    int wait_status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t waited = waitpid(child, &wait_status, WNOHANG);
        if (waited == child) {
            return WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : 1;
        }
        if (waited < 0 && errno != EINTR) return 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill(child, SIGKILL);
    while (waitpid(child, &wait_status, 0) < 0 && errno == EINTR) {}
    return 124;
#endif
}

class TemporaryFiles {
public:
    explicit TemporaryFiles(std::string prefix) : prefix_(std::move(prefix)) {}
    ~TemporaryFiles() {
        for (const std::string & suffix : {
                 "_reference.bin", "_reference.count",
                 "_candidate.bin", "_candidate.count"}) {
            std::remove((prefix_ + suffix).c_str());
        }
    }

    std::string path(const char * suffix) const { return prefix_ + suffix; }

private:
    std::string prefix_;
};

bool grouped_supported_device() {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return false;
#if defined(__HIP_PLATFORM_AMD__)
    const std::string arch = properties.gcnArchName;
    return arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
#else
    return properties.major*10 + properties.minor >= 75;
#endif
}

kq::CaseResult qualify_deepseek4_grouped_mmid(const std::string & executable) {
    if (!grouped_supported_device()) {
        return kq::unsupported(
            DS4_CASE, "grouped MMID requires NVIDIA Turing+ or AMD RDNA3/RDNA4");
    }
#if defined(_WIN32)
    const long long pid = static_cast<long long>(_getpid());
#else
    const long long pid = static_cast<long long>(getpid());
#endif
    TemporaryFiles files("kernel_qualification_ds4_" + std::to_string(pid));
    const std::string reference_output = files.path("_reference.bin");
    const std::string reference_count = files.path("_reference.count");
    const std::string candidate_output = files.path("_candidate.bin");
    const std::string candidate_count = files.path("_candidate.count");
    constexpr std::chrono::seconds child_timeout(120);
    const int reference_status = run_command_with_timeout(
        child_command(executable, false, reference_output, reference_count),
        child_timeout);
    const int candidate_status = run_command_with_timeout(
        child_command(executable, true, candidate_output, candidate_count),
        child_timeout);
    const std::vector<float> reference = read_binary(reference_output);
    const std::vector<float> candidate = read_binary(candidate_output);
    const size_t reference_grouped = read_count(reference_count);
    const size_t candidate_grouped = read_count(candidate_count);
    const bool route_matched = reference_status == 0 && candidate_status == 0 &&
        reference_grouped == 0 && candidate_grouped == 1;
    const std::string observed =
        "reference.grouped=" + std::to_string(reference_grouped) +
        ",candidate.grouped=" + std::to_string(candidate_grouped) +
        ",reference.status=" + std::to_string(reference_status) +
        ",candidate.status=" + std::to_string(candidate_status);
    kq::Route route{
        "reference.grouped=0,candidate.grouped=1,status=0",
        observed, route_matched};
    if (!route_matched || reference.empty() || candidate.empty()) {
        return failed_case(
            DS4_CASE, "reference or grouped child failed", std::move(route));
    }
    return kq::evaluate(
        DS4_CASE,
        {kq::compare_f32_nmse("output", candidate, reference, 5.0e-4)},
        std::move(route));
}

int run_child(const char * output_path, const char * count_path) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 1;
    const GraphRun run = run_grouped_mmid_graph(backend);
    const bool wrote = run.computed && write_binary(output_path, run.output) &&
        write_count(count_path, run.grouped_launches);
    ggml_backend_free(backend);
    return wrote ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--child") == 0) {
        return run_child(argv[2], argv[3]);
    }
    if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [--child OUTPUT COUNT]\n";
        return 2;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        const std::vector<kq::CaseResult> cases{
            kq::unsupported(MOE_CASE, "CUDA/HIP backend unavailable"),
            kq::unsupported(DS4_CASE, "CUDA/HIP backend unavailable"),
        };
        kq::write_json(std::cout, "unavailable", cases);
        return kq::exit_code(cases);
    }

    char description[256] = {};
    ggml_backend_cuda_get_device_description(0, description, sizeof(description));
    const std::vector<kq::CaseResult> cases{
        qualify_moe_gate_up_glu(),
        qualify_deepseek4_grouped_mmid(argv[0]),
    };
    kq::write_json(std::cout, description, cases);
    return kq::exit_code(cases);
}
