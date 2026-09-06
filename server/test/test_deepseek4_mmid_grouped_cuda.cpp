#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "rocmfpx.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

static bool run_case(
        ggml_backend_t backend,
        ggml_type type,
        int width,
        bool fused_ds4,
        bool write_output,
        std::ofstream & output) {
    int k_dim = 256;
    int n_rows = 128;
    int n_experts = 32;
    int top_k = 8;
    const auto env_positive = [](const char * name, int fallback) {
        const char * raw = std::getenv(name);
        const int parsed = raw ? std::atoi(raw) : 0;
        return parsed > 0 ? parsed : fallback;
    };
    if (std::getenv("DFLASH_MMID_BENCH_ITERS")) {
        k_dim = env_positive("DFLASH_MMID_BENCH_K", k_dim);
        n_rows = env_positive("DFLASH_MMID_BENCH_ROWS", n_rows);
        n_experts = env_positive("DFLASH_MMID_BENCH_EXPERTS", n_experts);
        top_k = env_positive("DFLASH_MMID_BENCH_TOP_K", top_k);
    }
    if (top_k > n_experts) {
        std::fprintf(stderr, "top_k=%d exceeds n_experts=%d\n", top_k, n_experts);
        return false;
    }

    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, type, k_dim, n_rows, n_experts);
    ggml_tensor * gate_weights =
        fused_ds4 ? ggml_new_tensor_3d(ctx, type, k_dim, n_rows, n_experts) : nullptr;
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_dim, 1, width);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, width);
    ggml_set_input(weights);
    if (gate_weights != nullptr) {
        ggml_set_input(gate_weights);
    }
    ggml_set_input(input);
    ggml_set_input(ids);

    ggml_tensor * result = ggml_mul_mat_id(ctx, weights, input, ids);
    if (gate_weights != nullptr) {
        ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weights, input, ids);
        result = ggml_swiglu_ds4_split(ctx, gate, result, 1.5f);
    }
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    const bool benchmark = std::getenv("DFLASH_MMID_BENCH_ITERS") != nullptr;
    std::mt19937 rng(20260713u + (unsigned) type * 97u + (unsigned) width);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> weights_f;
    if (!benchmark) {
        weights_f.resize((size_t) k_dim * n_rows * n_experts);
    }
    std::vector<float> input_f((size_t) k_dim * width);
    for (float & value : weights_f) {
        value = dist(rng);
    }
    for (float & value : input_f) {
        value = dist(rng);
    }

    std::vector<uint8_t> weights_q(ggml_nbytes(weights));
    // A zero-filled quantized tensor is valid and exercises the identical GPU
    // load/dequantize path without constructing multi-gigabyte F32 weights for
    // realistic expert-count benchmarks. Correctness runs still use quantized
    // randomized weights.
    const size_t quantized = benchmark ? weights_q.size() :
        type == GGML_TYPE_Q2_0_ROCMFP2
            ? rocmfpx_quantize_fp2(
                  weights_f.data(), weights_q.data(), n_rows * n_experts,
                  k_dim, nullptr)
        : type == GGML_TYPE_Q3_0_ROCMFPX
            ? rocmfpx_quantize_fp3(
                  weights_f.data(), weights_q.data(), n_rows * n_experts,
                  k_dim, nullptr)
            : ggml_quantize_chunk(
                  type, weights_f.data(), weights_q.data(), 0,
                  n_rows * n_experts, k_dim, nullptr);
    if (quantized != weights_q.size()) {
        std::fprintf(stderr, "quantize size mismatch type=%s got=%zu expected=%zu\n",
                     ggml_type_name(type), quantized, weights_q.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> ids_h((size_t) top_k * width);
    const bool masked_owner_routes =
        width >= 32 &&
        (type == GGML_TYPE_Q2_0_ROCMFP2 || type == GGML_TYPE_Q3_0_ROCMFPX);
    for (int token = 0; token < width; ++token) {
        for (int slot = 0; slot < top_k; ++slot) {
            // Exercise the owner-split contract as well as dense routing:
            // negative IDs are masked routes and their output lanes must be
            // exactly zero rather than stale allocator contents.
            ids_h[(size_t) token * top_k + slot] =
                masked_owner_routes && (token + slot) % 5 == 0
                    ? -1
                    : (token * 3 + slot * 5) % n_experts;
        }
    }

    ggml_backend_tensor_set(weights, weights_q.data(), 0, weights_q.size());
    if (gate_weights != nullptr) {
        ggml_backend_tensor_set(gate_weights, weights_q.data(), 0, weights_q.size());
    }
    ggml_backend_tensor_set(input, input_f.data(), 0, input_f.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_h.data(), 0, ids_h.size() * sizeof(int32_t));
    ggml_backend_synchronize(backend);

    ggml_status status = ggml_backend_graph_compute(backend, graph);
    int benchmark_iterations = 0;
    if (const char * raw = std::getenv("DFLASH_MMID_BENCH_ITERS")) {
        benchmark_iterations = std::max(0, std::atoi(raw));
    }
    if (status == GGML_STATUS_SUCCESS && benchmark_iterations > 0) {
        ggml_backend_synchronize(backend);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < benchmark_iterations && status == GGML_STATUS_SUCCESS; ++i) {
            status = ggml_backend_graph_compute(backend, graph);
        }
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        const double average_us = std::chrono::duration<double, std::micro>(
            end - start).count() / benchmark_iterations;
        std::printf(
            "[mmid-grouped-test] benchmark type=%s width=%d experts=%d top_k=%d "
            "k=%d rows=%d iterations=%d average_us=%.3f\n",
            ggml_type_name(type), width, n_experts, top_k, k_dim, n_rows,
            benchmark_iterations, average_us);
    }
    std::vector<float> result_h(ggml_nelements(result));
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(result, result_h.data(), 0, result_h.size() * sizeof(float));
        for (int token = 0; token < width; ++token) {
            for (int slot = 0; slot < top_k; ++slot) {
                if (ids_h[(size_t) token * top_k + slot] >= 0) {
                    continue;
                }
                for (int row = 0; row < n_rows; ++row) {
                    const size_t index =
                        ((size_t) token * top_k + slot) * n_rows + row;
                    if (result_h[index] != 0.0f || std::signbit(result_h[index])) {
                        std::fprintf(stderr,
                                     "masked route is not +0 token=%d slot=%d row=%d value=%g\n",
                                     token, slot, row, result_h[index]);
                        ggml_gallocr_free(alloc);
                        ggml_free(ctx);
                        return false;
                    }
                }
            }
        }
        if (write_output) {
            output.write(reinterpret_cast<const char *>(result_h.data()),
                         (std::streamsize) (result_h.size() * sizeof(float)));
        }
    }

    std::printf("[mmid-grouped-test] type=%s width=%d fused_ds4=%d pairs=%d status=%d bytes=%zu\n",
                ggml_type_name(type), width, fused_ds4 ? 1 : 0, width * top_k, (int) status,
                result_h.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS && output.good();
}

static int run_child(const char * mode, const char * output_path) {
    const bool grouped = std::strcmp(mode, "grouped") == 0;
    const bool masked_fused = std::strcmp(mode, "masked-fused") == 0;
    if (!grouped && !masked_fused && std::strcmp(mode, "legacy") != 0) {
        return 2;
    }
#if defined(_WIN32)
    if (!grouped && !masked_fused) {
        _putenv_s("GGML_CUDA_DISABLE_FUSION", "1");
    } else {
        _putenv_s("GGML_CUDA_DISABLE_FUSION", "");
    }
#else
    if (!grouped && !masked_fused) {
        setenv("GGML_CUDA_DISABLE_FUSION", "1", 1);
    } else {
        unsetenv("GGML_CUDA_DISABLE_FUSION");
    }
#endif

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "GPU backend unavailable\n");
        return 1;
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (masked_fused) {
        const bool ok = output.good() && run_case(
            backend, GGML_TYPE_Q2_0_ROCMFP2, 32, true, true, output);
        output.close();
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    }
    const ggml_type types[] = {
        GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_K, GGML_TYPE_Q2_0_ROCMFP2, GGML_TYPE_Q3_0_ROCMFPX,
        GGML_TYPE_Q4_0_ROCMFP4_FAST,
    };
    int width_filter = 0;
    if (const char * raw = std::getenv("DFLASH_MMID_TEST_WIDTH")) {
        width_filter = std::max(0, std::atoi(raw));
    }
    const std::vector<int> widths = width_filter > 0
        ? std::vector<int>{width_filter}
        : std::vector<int>{2, 4, 8, 9, 16, 32, 48, 64};
    bool ok = output.good();
    for (ggml_type type : types) {
        if (type == GGML_TYPE_Q4_0_ROCMFP4_FAST && width_filter == 0) {
            continue;
        }
        for (int width : widths) {
            if ((width_filter > 0 && width != width_filter) ||
                (width >= 32 &&
                type != GGML_TYPE_Q2_0_ROCMFP2 &&
                type != GGML_TYPE_Q3_0_ROCMFPX &&
                type != GGML_TYPE_Q4_0_ROCMFP4_FAST)) {
                continue;
            }
            ok = run_case(backend, type, width, false, true, output) && ok;
            if (width < 32 && width_filter == 0) {
                ok = run_case(backend, type, width, true, true, output) && ok;
            }
        }
    }
    output.close();
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}

static std::vector<char> read_file(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamsize size = input.tellg();
    input.seekg(0);
    std::vector<char> data((size_t) size);
    input.read(data.data(), size);
    return data;
}

static size_t count_records(const std::vector<char> & data, const char * needle) {
    const std::string text(data.begin(), data.end());
    size_t count = 0;
    size_t offset = 0;
    while ((offset = text.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += std::strlen(needle);
    }
    return count;
}

static bool has_mmvq_record(
        const std::vector<char> & log,
        ggml_type type,
        int width,
        const char * variant = nullptr) {
    const std::string type_field = "type=" + std::string(ggml_type_name(type)) + " ";
    const std::string width_field = "width=" + std::to_string(width) + " ";
    const std::string variant_field = variant ? "variant=" + std::string(variant) : std::string();
    std::istringstream lines(std::string(log.begin(), log.end()));
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find("[dflash-mmid] event=mmvq ") == std::string::npos ||
            line.find(type_field) == std::string::npos ||
            line.find(width_field) == std::string::npos) {
            continue;
        }
        if (!variant || line.find(variant_field) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool compare_case_outputs(
        const std::vector<char> & legacy,
        const std::vector<char> & grouped,
        size_t offset,
        size_t case_bytes,
        bool require_exact) {
    if (offset + case_bytes > legacy.size() || offset + case_bytes > grouped.size()) {
        return false;
    }
    if (require_exact) {
        return std::equal(
            legacy.begin() + (std::ptrdiff_t) offset,
            legacy.begin() + (std::ptrdiff_t) (offset + case_bytes),
            grouped.begin() + (std::ptrdiff_t) offset);
    }

    double squared_error = 0.0;
    double reference_power = 0.0;
    bool finite = true;
    for (size_t byte = 0; byte < case_bytes; byte += sizeof(float)) {
        float expected;
        float actual;
        std::memcpy(&expected, legacy.data() + offset + byte, sizeof(float));
        std::memcpy(&actual, grouped.data() + offset + byte, sizeof(float));
        finite = finite && std::isfinite(expected) && std::isfinite(actual);
        const double error = (double) actual - expected;
        squared_error += error * error;
        reference_power += (double) expected * expected;
    }

    // The unfused legacy graph is an independent numerical oracle for fused
    // DS4 cases, and legacy MMQ is the oracle for cases that did not previously
    // fit the architecture's MMVQ ceiling. Match the repository's MMQ tolerance
    // for the resulting reduction-order and fused-operation differences.
    constexpr double max_nmse = 5e-4;
    const double nmse = squared_error / std::max(reference_power, 1e-30);
    return finite && nmse <= max_nmse;
}

static bool grouped_supported_device() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }
#if defined(__HIP_PLATFORM_AMD__)
    const std::string arch = prop.gcnArchName;
    return arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
#else
    return prop.major * 10 + prop.minor >= 75;
#endif
}

struct CombineRun {
    std::vector<float> output;
    double average_us = 0.0;
    bool ok = false;
};

static CombineRun run_combine_graph(
        ggml_backend_t backend,
        bool fused,
        int n_embd,
        int n_used,
        int n_tokens,
        const std::vector<float> & experts_h,
        const std::vector<float> & weights_h,
        int iterations) {
    CombineRun run;
    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return run;
    }

    ggml_tensor * experts = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, n_embd, n_used, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(experts);
    ggml_set_input(weights);

    ggml_tensor * result = nullptr;
    if (fused) {
        result = ggml_laguna_moe_combine(ctx, experts, weights);
    } else {
        ggml_tensor * weights_3d = ggml_reshape_3d(
            ctx, weights, 1, n_used, n_tokens);
        result = ggml_mul(ctx, experts, weights_3d);
        result = ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
        result = ggml_sum_rows(ctx, result);
        result = ggml_reshape_2d(ctx, result, n_embd, n_tokens);
    }
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return run;
    }

    ggml_backend_tensor_set(
        experts, experts_h.data(), 0, experts_h.size() * sizeof(float));
    ggml_backend_tensor_set(
        weights, weights_h.data(), 0, weights_h.size() * sizeof(float));
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations && status == GGML_STATUS_SUCCESS; ++i) {
        status = ggml_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();

    run.output.resize((size_t)n_embd * n_tokens);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(
            result, run.output.data(), 0, run.output.size() * sizeof(float));
        run.average_us = std::chrono::duration<double, std::micro>(
            end - start).count() / std::max(iterations, 1);
        run.ok = true;
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return run;
}

static bool run_combine_parity_and_benchmark(ggml_backend_t backend) {
    const char * bench_raw = std::getenv("DFLASH_MOE_COMBINE_BENCH");
    const bool benchmark = bench_raw && *bench_raw && std::strcmp(bench_raw, "0") != 0;
    const int n_embd = benchmark ? 4096 : 260;
    const int n_used = 6;
    int n_tokens = benchmark ? 3072 : 33;
    if (const char * raw = std::getenv("DFLASH_MOE_COMBINE_BENCH_TOKENS")) {
        const int requested = std::atoi(raw);
        if (requested > 0) n_tokens = requested;
    }
    const int iterations = benchmark ? 20 : 2;

    std::vector<float> experts((size_t)n_embd * n_used * n_tokens);
    std::vector<float> weights((size_t)n_used * n_tokens);
    for (size_t i = 0; i < experts.size(); ++i) {
        experts[i] = ((int)(i % 251) - 125) * (1.0f / 127.0f);
    }
    for (int t = 0; t < n_tokens; ++t) {
        float total = 0.0f;
        for (int e = 0; e < n_used; ++e) {
            float value = (float)(e + 1 + t % 7);
            if ((t + e) % 11 == 0) value = 0.0f;
            weights[(size_t)t * n_used + e] = value;
            total += value;
        }
        for (int e = 0; e < n_used; ++e) {
            weights[(size_t)t * n_used + e] /= total;
        }
    }

    std::vector<float> expected((size_t)n_embd * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_embd; ++h) {
            float sum = 0.0f;
            for (int e = 0; e < n_used; ++e) {
                const float weight = weights[(size_t)t * n_used + e];
                if (weight == 0.0f) {
                    if (e == 0) sum = 0.0f;
                    continue;
                }
                const float product =
                    experts[((size_t)t * n_used + e) * n_embd + h] * weight;
                sum = e == 0 ? product : sum + product;
            }
            expected[(size_t)t * n_embd + h] = sum;
        }
    }

    CombineRun legacy;
    if (benchmark) {
        legacy = run_combine_graph(
            backend, false, n_embd, n_used, n_tokens,
            experts, weights, iterations);
    }
    const CombineRun fused = run_combine_graph(
        backend, true, n_embd, n_used, n_tokens,
        experts, weights, iterations);
    if (!fused.ok || fused.output.size() != expected.size() ||
        (benchmark && (!legacy.ok || legacy.output.size() != expected.size()))) {
        return false;
    }

    double fused_squared_error = 0.0;
    double legacy_squared_error = 0.0;
    double reference_power = 0.0;
    float fused_max_abs_error = 0.0f;
    size_t fused_exact = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::memcmp(&expected[i], &fused.output[i], sizeof(float)) == 0) {
            ++fused_exact;
        }
        const float fused_error = fused.output[i] - expected[i];
        fused_max_abs_error = std::max(
            fused_max_abs_error, std::fabs(fused_error));
        fused_squared_error += (double)fused_error * fused_error;
        if (benchmark) {
            const float legacy_error = legacy.output[i] - expected[i];
            legacy_squared_error += (double)legacy_error * legacy_error;
        }
        reference_power += (double)expected[i] * expected[i];
    }
    const double fused_nmse =
        fused_squared_error / std::max(reference_power, 1e-30);
    const double legacy_nmse = benchmark
        ? legacy_squared_error / std::max(reference_power, 1e-30) : 0.0;
    const double speedup = benchmark && fused.average_us > 0.0
        ? legacy.average_us / fused.average_us : 0.0;
    if (!std::isfinite(fused_nmse) || fused_nmse > 1e-12) {
        for (size_t i = 0; i < std::min<size_t>(expected.size(), 8); ++i) {
            std::fprintf(stderr,
                "combine mismatch[%zu] expected=%g fused=%g\n",
                i, expected[i], fused.output[i]);
        }
    }
    if (benchmark) {
        std::printf(
            "[mmid-grouped-test] combine n_embd=%d n_used=%d n_tokens=%d "
            "legacy_us=%.3f fused_us=%.3f speedup=%.3fx "
            "fused_max_abs=%g fused_nmse=%g legacy_nmse=%g\n",
            n_embd, n_used, n_tokens, legacy.average_us, fused.average_us,
            speedup, fused_max_abs_error, fused_nmse, legacy_nmse);
    } else {
        std::printf(
            "[mmid-grouped-test] combine-vec4 exact=%zu/%zu "
            "max_abs=%g nmse=%g\n",
            fused_exact, expected.size(), fused_max_abs_error, fused_nmse);
    }
    return std::isfinite(fused_nmse) && fused_nmse <= 1e-12;
}

static std::string shell_quote(const std::string & value) {
#if defined(_WIN32)
    std::string quoted = "\"";
    for (const char c : value) {
        if (c == '"') {
            quoted += '\\';
        }
        quoted += c;
    }
    return quoted + "\"";
#else
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    return quoted + "'";
#endif
}

static std::string child_command(
        const std::string & executable,
        const char * mode,
        const std::string & output_path,
        const std::string & log_path) {
#if defined(_WIN32)
    return "set \"DFLASH_MMID_TELEMETRY=1\" && set \"DFLASH_MMID_GROUPED_TYPES=15\" && "
        "set \"DFLASH_CUDA_MMVQ_MOE_FP2_PACKED32=1\" && "
        "set \"DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24=1\" && "
        "set \"DFLASH_MMID_GROUPED=" +
        std::string(std::strcmp(mode, "grouped") == 0 ? "1" : "0") + "\" && " +
        shell_quote(executable) + " --child " + mode + " " + shell_quote(output_path) +
        " 2>" + shell_quote(log_path);
#else
    return "DFLASH_MMID_TELEMETRY=1 DFLASH_MMID_GROUPED_TYPES=15 "
        "DFLASH_CUDA_MMVQ_MOE_FP2_PACKED32=1 DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24=1 "
        "DFLASH_MMID_GROUPED=" +
        std::string(std::strcmp(mode, "grouped") == 0 ? "1" : "0") + " " +
        shell_quote(executable) + " --child " + mode + " " + shell_quote(output_path) +
        " 2>" + shell_quote(log_path);
#endif
}

int main(int argc, char ** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--child") == 0) {
        return run_child(argv[2], argv[3]);
    }
    const bool combine_only =
        argc == 2 && std::strcmp(argv[1], "--combine-only") == 0;
    if (argc != 1 && !combine_only) {
        std::fprintf(stderr,
                     "usage: %s [--combine-only|--child legacy|grouped|masked-fused OUTPUT]\n",
                     argv[0]);
        return 2;
    }

    if (!grouped_supported_device()) {
        std::printf("[mmid-grouped-test] SKIP: grouped MMID requires NVIDIA Turing+ or AMD RDNA3/RDNA4\n");
        return 77;
    }

#if defined(_WIN32)
    if (!std::getenv("DFLASH_MOE_COMBINE_VEC4")) {
        _putenv_s("DFLASH_MOE_COMBINE_VEC4", "1");
    }
#else
    setenv("DFLASH_MOE_COMBINE_VEC4", "1", 0);
#endif

    ggml_backend_t combine_backend = ggml_backend_cuda_init(0);
    if (combine_backend == nullptr) {
        std::fprintf(stderr, "GPU backend unavailable for combine parity\n");
        return 1;
    }
    const bool combine_parity = run_combine_parity_and_benchmark(combine_backend);
    ggml_backend_free(combine_backend);
    if (!combine_parity) {
        std::fprintf(stderr, "grouped MoE fused-combine parity failed\n");
        return 1;
    }
    if (combine_only) {
        return 0;
    }

#if defined(_WIN32)
    const long long pid = (long long) _getpid();
#else
    const long long pid = (long long) getpid();
#endif
    const std::string prefix = "mmid_grouped_" + std::to_string(pid);
    const std::string legacy_path = prefix + "_legacy.bin";
    const std::string grouped_path = prefix + "_enabled.bin";
    const std::string legacy_log_path = prefix + "_legacy.log";
    const std::string grouped_log_path = prefix + "_enabled.log";
    const std::string masked_fused_path = prefix + "_masked_fused.bin";
    const std::string masked_fused_log_path = prefix + "_masked_fused.log";
    const std::string executable = argv[0];
    const std::string legacy_cmd =
        child_command(executable, "legacy", legacy_path, legacy_log_path);
    const std::string grouped_cmd =
        child_command(executable, "grouped", grouped_path, grouped_log_path);
    const std::string masked_fused_cmd = child_command(
        executable, "masked-fused", masked_fused_path, masked_fused_log_path);

    const int legacy_status = std::system(legacy_cmd.c_str());
    const int grouped_status = std::system(grouped_cmd.c_str());
    const int masked_fused_status = std::system(masked_fused_cmd.c_str());
    const std::vector<char> legacy = read_file(legacy_path.c_str());
    const std::vector<char> grouped = read_file(grouped_path.c_str());
    const std::vector<char> masked_fused = read_file(masked_fused_path.c_str());
    const std::vector<char> legacy_log = read_file(legacy_log_path.c_str());
    const std::vector<char> grouped_log = read_file(grouped_log_path.c_str());
    const size_t legacy_grouped = count_records(legacy_log, "variant=grouped");
    const size_t grouped_grouped = count_records(grouped_log, "variant=grouped");

    const ggml_type types[] = {
        GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_K, GGML_TYPE_Q2_0_ROCMFP2, GGML_TYPE_Q3_0_ROCMFPX,
    };
    const int widths[] = {2, 4, 8, 9, 16, 32, 48, 64};
    size_t offset = 0;
    size_t compared_bytes = 0;
    int compared_cases = 0;
    int exact_cases = 0;
    int tolerant_cases = 0;
    bool output_parity = legacy.size() == grouped.size() && !legacy.empty();
    bool grouped_dispatch = true;
    for (ggml_type type : types) {
        for (int width : widths) {
            if (width >= 32 &&
                type != GGML_TYPE_Q2_0_ROCMFP2 &&
                type != GGML_TYPE_Q3_0_ROCMFPX) {
                continue;
            }
            const bool legacy_mmvq = has_mmvq_record(legacy_log, type, width);
            for (bool fused_ds4 : {false, true}) {
                if (width >= 32 && fused_ds4) {
                    continue;
                }
                const size_t case_bytes = (size_t) 128 * 8 * width * sizeof(float);
                const bool require_exact = legacy_mmvq && !fused_ds4;
                if (width <= 16) {
                    grouped_dispatch =
                        has_mmvq_record(grouped_log, type, width, "grouped") && grouped_dispatch;
                }
                if (output_parity) {
                    output_parity = compare_case_outputs(
                        legacy, grouped, offset, case_bytes, require_exact);
                    if (!output_parity) {
                        std::fprintf(stderr,
                                     "output mismatch type=%s width=%d fused_ds4=%d exact=%d\n",
                                     ggml_type_name(type), width, fused_ds4 ? 1 : 0,
                                     require_exact ? 1 : 0);
                    }
                }
                compared_bytes += case_bytes;
                ++compared_cases;
                exact_cases += require_exact ? 1 : 0;
                tolerant_cases += require_exact ? 0 : 1;
                offset += case_bytes;
            }
        }
    }
    output_parity = output_parity && offset == legacy.size() && compared_cases == 76;
    const size_t masked_case_bytes = (size_t) 128 * 8 * 32 * sizeof(float);
    const bool masked_fused_zero = masked_fused.size() == masked_case_bytes;
    const bool pass = legacy_status == 0 && grouped_status == 0 &&
        masked_fused_status == 0 && masked_fused_zero &&
        output_parity && grouped_dispatch && legacy_grouped == 0 && grouped_grouped == 105;
    if (pass) {
        std::remove(legacy_path.c_str());
        std::remove(grouped_path.c_str());
        std::remove(legacy_log_path.c_str());
        std::remove(grouped_log_path.c_str());
        std::remove(masked_fused_path.c_str());
        std::remove(masked_fused_log_path.c_str());
    }
    std::printf("[mmid-grouped-test] legacy_status=%d grouped_status=%d bytes=%zu "
                "compared_cases=%d exact_cases=%d tolerant_cases=%d compared_bytes=%zu "
                "legacy_grouped=%zu grouped_grouped=%zu masked_fused_zero=%s "
                "parity=%s\n",
                legacy_status, grouped_status, legacy.size(), compared_cases, exact_cases,
                tolerant_cases, compared_bytes, legacy_grouped, grouped_grouped,
                masked_fused_zero ? "PASS" : "FAIL",
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
