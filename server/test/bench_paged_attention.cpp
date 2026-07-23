#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

constexpr int D = 256;
constexpr int N_HEAD = 24;
constexpr int N_HEAD_KV = 4;
constexpr int BLOCK_SIZE = 16;
constexpr int PROTOTYPE_ROWS = 256;
constexpr int CONTIGUOUS_CONTEXT_ALIGNMENT = 256;
constexpr int QWEN_FULL_ATTENTION_LAYERS = 16;
constexpr size_t MAX_RAGGED_SEQUENCES = 64;
constexpr double TARGET_SAMPLE_SECONDS = 0.20;

struct Options {
    int context = 4096;
    int warmup = 10;
    int samples = 7;
    int iterations = 0;
    ggml_type k_type = GGML_TYPE_Q4_0;
    ggml_type v_type = GGML_TYPE_Q4_0;
    std::vector<int> ragged_contexts;
};

struct TimingStats {
    int iterations = 0;
    double median_us = 0.0;
    double p95_us = 0.0;
    double aggregate_queries_s = 0.0;
};

struct BenchResult {
    int n_seq = 0;
    int padded_context = 0;
    int64_t live_tokens = 0;
    int64_t padded_tokens = 0;
    int64_t paged_pool_tokens = 0;
    TimingStats contiguous;
    TimingStats paged;
    uint64_t contiguous_kv_bytes = 0;
    uint64_t paged_kv_bytes = 0;
    uint64_t paged_metadata_bytes = 0;
    double contiguous_kv_mib = 0.0;
    double paged_kv_mib = 0.0;
    double paged_metadata_kib = 0.0;
    double max_abs_error = 0.0;
    double max_tolerance_ratio = 0.0;
    double rmse = 0.0;
};

struct CachePair {
    std::vector<uint8_t> contiguous;
    std::vector<uint8_t> paged;
};

struct ContextGuard {
    ggml_context * value = nullptr;
    ~ContextGuard() {
        if (value) ggml_free(value);
    }
};

struct AllocatorGuard {
    ggml_gallocr_t value = nullptr;
    ~AllocatorGuard() {
        if (value) ggml_gallocr_free(value);
    }
};

void print_usage(const char * program) {
    std::fprintf(stderr,
        "Usage: %s [--context N] [--warmup N] [--samples N]\n"
        "          [--iterations N] [--k-type TYPE] [--v-type TYPE]\n"
        "          [--ragged-contexts N1,N2,...]\n"
        "  TYPE: f16, q4_0, or q8_0\n"
        "  --ragged-contexts runs one ragged case against a per-sequence\n"
        "    contiguous cache padded to a 256-token maximum context.\n"
        "    Between 2 and %zu sequence lengths are accepted.\n"
        "  --iterations 0 automatically targets %.0f ms timing windows.\n",
        program, MAX_RAGGED_SEQUENCES,
        TARGET_SAMPLE_SECONDS * 1000.0);
}

bool parse_positive_int(const char * text, int & value, bool allow_zero) {
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (!text[0] || !end || *end != '\0' ||
        parsed < (allow_zero ? 0 : 1) || parsed > INT32_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_type(const char * text, ggml_type & type) {
    if (std::strcmp(text, "f16") == 0) {
        type = GGML_TYPE_F16;
        return true;
    }
    if (std::strcmp(text, "q4_0") == 0) {
        type = GGML_TYPE_Q4_0;
        return true;
    }
    if (std::strcmp(text, "q8_0") == 0) {
        type = GGML_TYPE_Q8_0;
        return true;
    }
    return false;
}

bool parse_context_list(
        const char * text,
        std::vector<int> & contexts) {
    contexts.clear();
    const char * cursor = text;
    while (cursor && cursor[0]) {
        int context = 0;
        char * end = nullptr;
        const long parsed = std::strtol(cursor, &end, 10);
        if (end == cursor || parsed < 1 || parsed > INT32_MAX ||
            (*end != ',' && *end != '\0') ||
            contexts.size() >= MAX_RAGGED_SEQUENCES) {
            contexts.clear();
            return false;
        }
        context = static_cast<int>(parsed);
        contexts.push_back(context);
        if (*end == '\0') {
            if (contexts.size() >= 2) return true;
            contexts.clear();
            return false;
        }
        cursor = end + 1;
    }
    contexts.clear();
    return false;
}

bool parse_options(int argc, char ** argv, Options & options) {
    for (int i = 1; i < argc; ++i) {
        const auto next = [&]() -> const char * {
            return i + 1 < argc ? argv[++i] : nullptr;
        };
        if (std::strcmp(argv[i], "--context") == 0) {
            const char * value = next();
            if (!value ||
                !parse_positive_int(value, options.context, false)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            const char * value = next();
            if (!value ||
                !parse_positive_int(value, options.warmup, true)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--samples") == 0) {
            const char * value = next();
            if (!value ||
                !parse_positive_int(value, options.samples, false)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--iterations") == 0) {
            const char * value = next();
            if (!value ||
                !parse_positive_int(value, options.iterations, true)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--k-type") == 0) {
            const char * value = next();
            if (!value || !parse_type(value, options.k_type)) return false;
        } else if (std::strcmp(argv[i], "--v-type") == 0) {
            const char * value = next();
            if (!value || !parse_type(value, options.v_type)) return false;
        } else if (std::strcmp(argv[i], "--ragged-contexts") == 0) {
            const char * value = next();
            if (!value ||
                !parse_context_list(value, options.ragged_contexts)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> make_prototype_rows(
        ggml_type type,
        float phase) {
    const size_t row_bytes = ggml_row_size(type, D);
    if (row_bytes == 0) return {};

    std::vector<float> source(
        static_cast<size_t>(PROTOTYPE_ROWS) * D);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] =
            std::sin(static_cast<float>(i) * 0.011f + phase) * 0.25f;
    }

    std::vector<uint8_t> result(
        static_cast<size_t>(PROTOTYPE_ROWS) * row_bytes);
    const size_t written = ggml_quantize_chunk(
        type, source.data(), result.data(), 0, PROTOTYPE_ROWS, D, nullptr);
    return written == result.size() ? result : std::vector<uint8_t>{};
}

bool cache_size_fits(ggml_type type, int64_t rows) {
    const size_t row_bytes = ggml_row_size(type, D);
    return rows > 0 && row_bytes > 0 &&
           static_cast<uint64_t>(rows) <= SIZE_MAX / row_bytes;
}

CachePair make_cache_pair(
        ggml_type type,
        const std::vector<int> & kv_seq_lens,
        int contiguous_context,
        int max_blocks,
        int64_t pool_tokens,
        const std::vector<int32_t> & block_table,
        float phase) {
    const int n_seq = static_cast<int>(kv_seq_lens.size());
    const int64_t contiguous_rows =
        static_cast<int64_t>(contiguous_context) * N_HEAD_KV * n_seq;
    const int64_t paged_rows = pool_tokens * N_HEAD_KV;
    if (!cache_size_fits(type, contiguous_rows) ||
        !cache_size_fits(type, paged_rows)) {
        return {};
    }

    const size_t row_bytes = ggml_row_size(type, D);
    const std::vector<uint8_t> prototype =
        make_prototype_rows(type, phase);
    if (prototype.empty()) return {};

    CachePair result;
    result.contiguous.resize(
        static_cast<size_t>(contiguous_rows) * row_bytes);
    result.paged.resize(
        static_cast<size_t>(paged_rows) * row_bytes);

    // Generate one canonical logical row and place the identical quantized
    // bytes in both the contiguous tensor and the paged physical pool.
    for (int seq = 0; seq < n_seq; ++seq) {
        for (int kv_head = 0; kv_head < N_HEAD_KV; ++kv_head) {
            for (int token = 0; token < kv_seq_lens[seq]; ++token) {
                const int64_t logical_row =
                    (static_cast<int64_t>(seq) * N_HEAD_KV + kv_head) *
                        contiguous_context +
                    token;
                const int32_t physical_block =
                    block_table[
                        static_cast<size_t>(seq) * max_blocks +
                        token / BLOCK_SIZE];
                const int64_t physical_token =
                    static_cast<int64_t>(physical_block) * BLOCK_SIZE +
                    token % BLOCK_SIZE;
                const int64_t paged_row =
                    static_cast<int64_t>(kv_head) * pool_tokens +
                    physical_token;
                // Keep sequence and head identity in the source pattern.
                // Using logical_row % PROTOTYPE_ROWS would collapse both at
                // common contexts such as 4096.
                const size_t prototype_row =
                    static_cast<size_t>(
                        (static_cast<int64_t>(token) * 17 +
                         (seq * N_HEAD_KV + kv_head) * 7) %
                        PROTOTYPE_ROWS);

                std::memcpy(
                    result.contiguous.data() +
                        static_cast<size_t>(logical_row) * row_bytes,
                    prototype.data() + prototype_row * row_bytes,
                    row_bytes);
                std::memcpy(
                    result.paged.data() +
                        static_cast<size_t>(paged_row) * row_bytes,
                    prototype.data() + prototype_row * row_bytes,
                    row_bytes);
            }
        }
    }
    return result;
}

bool make_block_table(
        const std::vector<int> & kv_seq_lens,
        int max_blocks,
        int64_t & pool_tokens,
        std::vector<int32_t> & block_table) {
    const int n_seq = static_cast<int>(kv_seq_lens.size());
    if (n_seq <= 0 ||
        static_cast<uint64_t>(max_blocks) * n_seq >
            SIZE_MAX / sizeof(int32_t)) {
        return false;
    }

    std::vector<int> sequence_blocks(n_seq);
    int64_t physical_blocks = 0;
    for (int seq = 0; seq < n_seq; ++seq) {
        sequence_blocks[seq] = static_cast<int>(
            (static_cast<int64_t>(kv_seq_lens[seq]) +
             BLOCK_SIZE - 1) /
            BLOCK_SIZE);
        physical_blocks += sequence_blocks[seq];
    }
    pool_tokens = physical_blocks * BLOCK_SIZE;
    if (physical_blocks <= 0 || physical_blocks > INT32_MAX ||
        pool_tokens <= 0 || pool_tokens > INT32_MAX) {
        return false;
    }

    block_table.assign(
        static_cast<size_t>(max_blocks) * n_seq, -1);
    int32_t physical = 0;
    for (int logical = 0; logical < max_blocks; ++logical) {
        // Round-robin by logical block keeps live blocks from different
        // sequences interleaved while assigning a compact, unique pool.
        for (int rank = 0; rank < n_seq; ++rank) {
            const int seq = (logical + rank) % n_seq;
            if (logical >= sequence_blocks[seq]) continue;
            block_table[
                static_cast<size_t>(seq) * max_blocks + logical] =
                physical++;
        }
    }
    return physical == physical_blocks;
}

bool compute_iterations(
        ggml_backend_t backend,
        ggml_cgraph * graph,
        int iterations,
        double & elapsed_seconds) {
    ggml_backend_synchronize(backend);
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (ggml_backend_graph_compute_async(backend, graph) !=
            GGML_STATUS_SUCCESS) {
            ggml_backend_synchronize(backend);
            return false;
        }
    }
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();
    elapsed_seconds =
        std::chrono::duration<double>(end - begin).count();
    return true;
}

bool warm_graph(
        ggml_backend_t backend,
        ggml_cgraph * graph,
        int warmup) {
    if (warmup <= 0) return true;
    double ignored_seconds = 0.0;
    return compute_iterations(
        backend, graph, warmup, ignored_seconds);
}

bool probe_graph(
        ggml_backend_t backend,
        ggml_cgraph * graph,
        double & seconds_per_step) {
    constexpr int PROBE_ITERATIONS = 5;
    double elapsed_seconds = 0.0;
    if (!compute_iterations(
            backend, graph, PROBE_ITERATIONS, elapsed_seconds)) {
        return false;
    }
    seconds_per_step =
        std::max(elapsed_seconds / PROBE_ITERATIONS, 1.0e-6);
    return true;
}

bool collect_timing(
        ggml_backend_t backend,
        ggml_cgraph * graph,
        const Options & options,
        int iterations,
        int n_seq,
        TimingStats & result) {
    if (!warm_graph(backend, graph, options.warmup)) return false;

    std::vector<double> sample_us;
    sample_us.reserve(options.samples);
    for (int sample = 0; sample < options.samples; ++sample) {
        double elapsed_seconds = 0.0;
        if (!compute_iterations(
                backend, graph, iterations, elapsed_seconds)) {
            return false;
        }
        sample_us.push_back(
            elapsed_seconds * 1.0e6 / static_cast<double>(iterations));
    }

    std::sort(sample_us.begin(), sample_us.end());
    const size_t p95_index = std::min(
        sample_us.size() - 1,
        static_cast<size_t>(
            std::ceil(sample_us.size() * 0.95) - 1.0));
    result.iterations = iterations;
    const size_t upper_middle = sample_us.size() / 2;
    result.median_us =
        sample_us.size() % 2 != 0
            ? sample_us[upper_middle]
            : (sample_us[upper_middle - 1] +
               sample_us[upper_middle]) /
                  2.0;
    result.p95_us = sample_us[p95_index];
    result.aggregate_queries_s =
        static_cast<double>(n_seq) * 1.0e6 / result.median_us;
    return true;
}

bool compare_outputs(
        ggml_tensor * paged_output,
        ggml_tensor * contiguous_output,
        int n_seq,
        BenchResult & result) {
    std::vector<float> paged(ggml_nelements(paged_output));
    std::vector<float> contiguous(ggml_nelements(contiguous_output));
    ggml_backend_tensor_get(
        paged_output, paged.data(), 0,
        paged.size() * sizeof(paged[0]));
    ggml_backend_tensor_get(
        contiguous_output, contiguous.data(), 0,
        contiguous.size() * sizeof(contiguous[0]));

    double squared_error = 0.0;
    bool ok = true;
    for (int seq = 0; seq < n_seq; ++seq) {
        for (int head = 0; head < N_HEAD; ++head) {
            for (int d = 0; d < D; ++d) {
                const size_t paged_index =
                    (static_cast<size_t>(head) * n_seq + seq) * D + d;
                const size_t contiguous_index =
                    (static_cast<size_t>(seq) * N_HEAD + head) * D + d;
                const float paged_value = paged[paged_index];
                const float contiguous_value =
                    contiguous[contiguous_index];
                if (!std::isfinite(paged_value) ||
                    !std::isfinite(contiguous_value)) {
                    return false;
                }

                const double abs_error =
                    std::fabs(
                        static_cast<double>(paged_value) -
                        contiguous_value);
                result.max_abs_error =
                    std::max(result.max_abs_error, abs_error);
                squared_error += abs_error * abs_error;

                const double tolerance =
                    1.0e-4 +
                    1.0e-3 *
                        std::fabs(static_cast<double>(contiguous_value));
                const double tolerance_ratio = abs_error / tolerance;
                result.max_tolerance_ratio =
                    std::max(
                        result.max_tolerance_ratio,
                        tolerance_ratio);
                ok = ok && tolerance_ratio <= 1.0;
            }
        }
    }
    result.rmse =
        std::sqrt(
            squared_error /
            static_cast<double>(D * N_HEAD * n_seq));
    return ok;
}

bool run_case(
        ggml_backend_t backend,
        const Options & options,
        const std::vector<int> & kv_seq_lens,
        int contiguous_context,
        bool use_padding_mask,
        bool contiguous_first,
        BenchResult & result) {
    if (kv_seq_lens.empty() ||
        kv_seq_lens.size() > MAX_RAGGED_SEQUENCES) {
        return false;
    }
    const int n_seq = static_cast<int>(kv_seq_lens.size());
    const int max_context =
        *std::max_element(kv_seq_lens.begin(), kv_seq_lens.end());
    const int64_t max_blocks_64 =
        (static_cast<int64_t>(max_context) + BLOCK_SIZE - 1) /
        BLOCK_SIZE;
    if (max_blocks_64 <= 0 || max_blocks_64 > INT32_MAX ||
        contiguous_context < max_context) {
        std::fprintf(
            stderr, "context layout exceeds attention kernel limits\n");
        return false;
    }
    const int max_blocks = static_cast<int>(max_blocks_64);

    int64_t pool_tokens = 0;
    std::vector<int32_t> block_table;
    if (!make_block_table(
            kv_seq_lens, max_blocks, pool_tokens, block_table)) {
        std::fprintf(
            stderr, "paged block table exceeds attention kernel limits\n");
        return false;
    }

    const CachePair k_data =
        make_cache_pair(
            options.k_type, kv_seq_lens, contiguous_context, max_blocks,
            pool_tokens, block_table, 0.1f);
    const CachePair v_data =
        make_cache_pair(
            options.v_type, kv_seq_lens, contiguous_context, max_blocks,
            pool_tokens, block_table, 0.7f);
    if (k_data.contiguous.empty() || k_data.paged.empty() ||
        v_data.contiguous.empty() || v_data.paged.empty()) {
        return false;
    }

    std::vector<float> paged_q(
        static_cast<size_t>(D) * n_seq * N_HEAD);
    std::vector<float> contiguous_q(paged_q.size());
    for (int seq = 0; seq < n_seq; ++seq) {
        for (int head = 0; head < N_HEAD; ++head) {
            for (int d = 0; d < D; ++d) {
                const size_t canonical_index =
                    (static_cast<size_t>(seq) * N_HEAD + head) * D + d;
                const float value =
                    std::cos(
                        static_cast<float>(canonical_index) * 0.013f) *
                    0.20f;
                contiguous_q[canonical_index] = value;
                paged_q[
                    (static_cast<size_t>(head) * n_seq + seq) * D + d] =
                    value;
            }
        }
    }

    ggml_init_params params{};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ContextGuard ctx{ggml_init(params)};
    if (!ctx.value) return false;

    ggml_tensor * q_paged =
        ggml_new_tensor_3d(
            ctx.value, GGML_TYPE_F32, D, n_seq, N_HEAD);
    ggml_tensor * k_paged =
        ggml_new_tensor_3d(
            ctx.value, options.k_type, D, pool_tokens, N_HEAD_KV);
    ggml_tensor * v_paged =
        ggml_new_tensor_3d(
            ctx.value, options.v_type, D, pool_tokens, N_HEAD_KV);
    ggml_tensor * table =
        ggml_new_tensor_2d(
            ctx.value, GGML_TYPE_I32, max_blocks, n_seq);
    ggml_tensor * kv_seq_lens_tensor =
        ggml_new_tensor_1d(ctx.value, GGML_TYPE_I32, n_seq);

    ggml_tensor * q_contiguous =
        ggml_new_tensor_4d(
            ctx.value, GGML_TYPE_F32, D, 1, N_HEAD, n_seq);
    ggml_tensor * k_contiguous =
        ggml_new_tensor_4d(
            ctx.value, options.k_type, D, contiguous_context,
            N_HEAD_KV, n_seq);
    ggml_tensor * v_contiguous =
        ggml_new_tensor_4d(
            ctx.value, options.v_type, D, contiguous_context,
            N_HEAD_KV, n_seq);
    ggml_tensor * padding_mask =
        use_padding_mask
            ? ggml_new_tensor_4d(
                  ctx.value, GGML_TYPE_F16, contiguous_context, 1, 1,
                  n_seq)
            : nullptr;

    for (ggml_tensor * input : {
            q_paged, k_paged, v_paged, table, kv_seq_lens_tensor,
            q_contiguous, k_contiguous, v_contiguous}) {
        ggml_set_input(input);
    }
    if (padding_mask) ggml_set_input(padding_mask);

    ggml_tensor * paged_output = ggml_paged_attn(
        ctx.value, q_paged, k_paged, v_paged, table, kv_seq_lens_tensor,
        1.0f / std::sqrt(static_cast<float>(D)), BLOCK_SIZE);
    ggml_tensor * contiguous_output = ggml_flash_attn_ext(
        ctx.value, q_contiguous, k_contiguous, v_contiguous,
        padding_mask, 1.0f / std::sqrt(static_cast<float>(D)),
        0.0f, 0.0f);
    ggml_set_output(paged_output);
    ggml_set_output(contiguous_output);

    if (!ggml_backend_supports_op(backend, paged_output) ||
        !ggml_backend_supports_op(backend, contiguous_output)) {
        std::fprintf(
            stderr,
            "GPU backend does not support K=%s V=%s for both layouts\n",
            ggml_type_name(options.k_type),
            ggml_type_name(options.v_type));
        return false;
    }

    ggml_cgraph * paged_graph = ggml_new_graph(ctx.value);
    ggml_build_forward_expand(paged_graph, paged_output);
    ggml_cgraph * contiguous_graph = ggml_new_graph(ctx.value);
    ggml_build_forward_expand(contiguous_graph, contiguous_output);
    ggml_cgraph * allocation_graph = ggml_new_graph(ctx.value);
    ggml_build_forward_expand(allocation_graph, paged_output);
    ggml_build_forward_expand(allocation_graph, contiguous_output);

    AllocatorGuard allocator{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))};
    if (!allocator.value ||
        !ggml_gallocr_alloc_graph(allocator.value, allocation_graph)) {
        return false;
    }

    ggml_backend_tensor_set(
        q_paged, paged_q.data(), 0,
        paged_q.size() * sizeof(paged_q[0]));
    ggml_backend_tensor_set(
        k_paged, k_data.paged.data(), 0, k_data.paged.size());
    ggml_backend_tensor_set(
        v_paged, v_data.paged.data(), 0, v_data.paged.size());
    ggml_backend_tensor_set(
        table, block_table.data(), 0,
        block_table.size() * sizeof(block_table[0]));
    ggml_backend_tensor_set(
        kv_seq_lens_tensor, kv_seq_lens.data(), 0,
        kv_seq_lens.size() * sizeof(kv_seq_lens[0]));
    ggml_backend_tensor_set(
        q_contiguous, contiguous_q.data(), 0,
        contiguous_q.size() * sizeof(contiguous_q[0]));
    ggml_backend_tensor_set(
        k_contiguous, k_data.contiguous.data(), 0,
        k_data.contiguous.size());
    ggml_backend_tensor_set(
        v_contiguous, v_data.contiguous.data(), 0,
        v_data.contiguous.size());
    if (padding_mask) {
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t negative_infinity =
            ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask_data(
            static_cast<size_t>(contiguous_context) * n_seq,
            negative_infinity);
        for (int seq = 0; seq < n_seq; ++seq) {
            std::fill_n(
                mask_data.begin() +
                    static_cast<size_t>(seq) * contiguous_context,
                kv_seq_lens[seq], zero);
        }
        ggml_backend_tensor_set(
            padding_mask, mask_data.data(), 0,
            mask_data.size() * sizeof(mask_data[0]));
    }
    ggml_backend_synchronize(backend);

    if (ggml_backend_graph_compute(backend, paged_graph) !=
            GGML_STATUS_SUCCESS ||
        ggml_backend_graph_compute(backend, contiguous_graph) !=
            GGML_STATUS_SUCCESS ||
        !compare_outputs(
            paged_output, contiguous_output, n_seq, result)) {
        std::fprintf(
            stderr,
            "layout parity failed at n_seq=%d: max_abs=%.6g "
            "max_tolerance_ratio=%.6g rmse=%.6g\n",
            n_seq, result.max_abs_error,
            result.max_tolerance_ratio, result.rmse);
        return false;
    }

    int paged_iterations = options.iterations;
    int contiguous_iterations = options.iterations;
    if (options.iterations == 0) {
        double paged_seconds_per_step = 0.0;
        double contiguous_seconds_per_step = 0.0;
        if (!warm_graph(backend, paged_graph, options.warmup) ||
            !probe_graph(
                backend, paged_graph, paged_seconds_per_step) ||
            !warm_graph(backend, contiguous_graph, options.warmup) ||
            !probe_graph(
                backend, contiguous_graph,
                contiguous_seconds_per_step)) {
            return false;
        }
        paged_iterations = std::clamp(
            static_cast<int>(
                std::ceil(
                    TARGET_SAMPLE_SECONDS /
                    paged_seconds_per_step)),
            1, 100000);
        contiguous_iterations = std::clamp(
            static_cast<int>(
                std::ceil(
                    TARGET_SAMPLE_SECONDS /
                    contiguous_seconds_per_step)),
            1, 100000);
    }

    // Keep each layout's sample windows together so ggml's CUDA-graph cache
    // remains stable. Alternate the order across cases to reduce clock and
    // thermal bias without repeatedly invalidating graph capture.
    bool timing_ok = false;
    if (contiguous_first) {
        timing_ok =
            collect_timing(
                backend, contiguous_graph, options,
                contiguous_iterations,
                n_seq, result.contiguous) &&
            collect_timing(
                backend, paged_graph, options, paged_iterations,
                n_seq, result.paged);
    } else {
        timing_ok =
            collect_timing(
                backend, paged_graph, options, paged_iterations,
                n_seq, result.paged) &&
            collect_timing(
                backend, contiguous_graph, options,
                contiguous_iterations,
                n_seq, result.contiguous);
    }
    if (!timing_ok) return false;

    result.n_seq = n_seq;
    result.padded_context = contiguous_context;
    result.live_tokens = 0;
    for (int context : kv_seq_lens) {
        result.live_tokens += context;
    }
    result.padded_tokens =
        static_cast<int64_t>(contiguous_context) * n_seq;
    result.paged_pool_tokens = pool_tokens;
    result.contiguous_kv_bytes =
        ggml_nbytes(k_contiguous) + ggml_nbytes(v_contiguous);
    result.paged_kv_bytes =
        ggml_nbytes(k_paged) + ggml_nbytes(v_paged);
    result.paged_metadata_bytes =
        ggml_nbytes(table) + ggml_nbytes(kv_seq_lens_tensor);
    result.contiguous_kv_mib =
        static_cast<double>(result.contiguous_kv_bytes) /
        (1024.0 * 1024.0);
    result.paged_kv_mib =
        static_cast<double>(result.paged_kv_bytes) /
        (1024.0 * 1024.0);
    result.paged_metadata_kib =
        static_cast<double>(result.paged_metadata_bytes) / 1024.0;
    return true;
}

bool padded_context_for(
        const std::vector<int> & contexts,
        int & padded_context) {
    if (contexts.empty()) return false;
    const int max_context =
        *std::max_element(contexts.begin(), contexts.end());
    const int64_t aligned =
        ((static_cast<int64_t>(max_context) +
          CONTIGUOUS_CONTEXT_ALIGNMENT - 1) /
         CONTIGUOUS_CONTEXT_ALIGNMENT) *
        CONTIGUOUS_CONTEXT_ALIGNMENT;
    if (aligned <= 0 || aligned > INT32_MAX) return false;
    padded_context = static_cast<int>(aligned);
    return true;
}

std::string join_contexts(const std::vector<int> & contexts) {
    std::string joined;
    for (size_t i = 0; i < contexts.size(); ++i) {
        if (i != 0) joined.push_back(',');
        joined += std::to_string(contexts[i]);
    }
    return joined;
}

void print_ragged_result(
        const Options & options,
        const BenchResult & result) {
    const bool cuda_graphs_enabled =
        std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr;
    const std::string contexts =
        join_contexts(options.ragged_contexts);
    const double step_time_ratio =
        result.paged.median_us / result.contiguous.median_us;
    const double throughput_ratio =
        result.paged.aggregate_queries_s /
        result.contiguous.aggregate_queries_s;
    const double memory_ratio =
        static_cast<double>(result.contiguous_kv_bytes) /
        static_cast<double>(result.paged_kv_bytes);
    const double memory_saving =
        (1.0 -
         static_cast<double>(result.paged_kv_bytes) /
             static_cast<double>(result.contiguous_kv_bytes)) *
        100.0;
    const uint64_t projected_contiguous_bytes =
        result.contiguous_kv_bytes * QWEN_FULL_ATTENTION_LAYERS;
    const uint64_t projected_paged_bytes =
        result.paged_kv_bytes * QWEN_FULL_ATTENTION_LAYERS;

    std::printf(
        "# padded per-sequence contiguous vs paged GPU attention; "
        "ragged decode, not whole-model throughput\n"
        "# D=%d,Hq=%d,Hkv=%d,block=%d,contexts=%s,"
        "contiguous_context_pad=%d,k=%s,v=%s,warmup=%d,samples=%d,"
        "cuda_graphs_env=%s\n"
        "# the Qwen projection multiplies K/V storage by %d full-attention "
        "layers; recurrent state and all other model memory are excluded\n"
        "# paged_step_time_overhead_pct > 0 means paged is slower; "
        "paged_over_padded_contiguous_query_throughput > 1 means paged "
        "is faster\n",
        D, N_HEAD, N_HEAD_KV, BLOCK_SIZE, contexts.c_str(),
        result.padded_context,
        ggml_type_name(options.k_type),
        ggml_type_name(options.v_type),
        options.warmup, options.samples,
        cuda_graphs_enabled ? "allowed" : "disabled",
        QWEN_FULL_ATTENTION_LAYERS);
    std::printf(
        "case,n_seq,contexts,live_tokens,"
        "padded_contiguous_tokens,paged_pool_tokens,"
        "padded_contiguous_iterations,paged_iterations,"
        "padded_contiguous_median_window_mean_us_per_step,"
        "paged_median_window_mean_us_per_step,"
        "padded_contiguous_p95_window_mean_us_per_step,"
        "paged_p95_window_mean_us_per_step,"
        "paged_over_padded_contiguous_step_time,"
        "paged_step_time_overhead_pct,"
        "padded_contiguous_aggregate_attention_queries_s,"
        "paged_aggregate_attention_queries_s,"
        "paged_over_padded_contiguous_query_throughput,"
        "padded_contiguous_kv_bytes_one_layer,"
        "paged_kv_bytes_one_layer,"
        "padded_contiguous_kv_mib_one_layer,"
        "paged_kv_mib_one_layer,"
        "padded_contiguous_kv_bytes_qwen16,"
        "paged_kv_bytes_qwen16,"
        "padded_contiguous_kv_gib_qwen16,"
        "paged_kv_gib_qwen16,"
        "padded_contiguous_over_paged_kv_ratio,"
        "paged_kv_saving_pct,paged_metadata_bytes,"
        "max_abs_error,max_tolerance_ratio,rmse\n");
    std::printf(
        "ragged,%d,\"%s\",%lld,%lld,%lld,%d,%d,"
        "%.3f,%.3f,%.3f,%.3f,%.6f,%.3f,%.2f,%.2f,%.6f,"
        "%llu,%llu,%.2f,%.2f,%llu,%llu,%.3f,%.3f,%.6f,%.3f,%llu,"
        "%.8g,%.8g,%.8g\n",
        result.n_seq, contexts.c_str(),
        static_cast<long long>(result.live_tokens),
        static_cast<long long>(result.padded_tokens),
        static_cast<long long>(result.paged_pool_tokens),
        result.contiguous.iterations, result.paged.iterations,
        result.contiguous.median_us, result.paged.median_us,
        result.contiguous.p95_us, result.paged.p95_us,
        step_time_ratio, (step_time_ratio - 1.0) * 100.0,
        result.contiguous.aggregate_queries_s,
        result.paged.aggregate_queries_s, throughput_ratio,
        static_cast<unsigned long long>(result.contiguous_kv_bytes),
        static_cast<unsigned long long>(result.paged_kv_bytes),
        result.contiguous_kv_mib, result.paged_kv_mib,
        static_cast<unsigned long long>(projected_contiguous_bytes),
        static_cast<unsigned long long>(projected_paged_bytes),
        static_cast<double>(projected_contiguous_bytes) /
            (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(projected_paged_bytes) /
            (1024.0 * 1024.0 * 1024.0),
        memory_ratio, memory_saving,
        static_cast<unsigned long long>(result.paged_metadata_bytes),
        result.max_abs_error, result.max_tolerance_ratio, result.rmse);
}

}  // namespace

int main(int argc, char ** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "GPU backend unavailable\n");
        return 1;
    }

    std::vector<BenchResult> results;
    bool ok = true;
    try {
        if (!options.ragged_contexts.empty()) {
            int padded_context = 0;
            BenchResult result;
            if (!padded_context_for(
                    options.ragged_contexts, padded_context) ||
                !run_case(
                    backend, options, options.ragged_contexts,
                    padded_context, true, true, result)) {
                std::fprintf(
                    stderr, "ragged attention benchmark failed\n");
                ok = false;
            } else {
                results.push_back(result);
            }
        } else {
            const std::array<int, 4> sequence_counts = {1, 2, 4, 8};
            results.reserve(sequence_counts.size());
            for (size_t case_index = 0;
                 case_index < sequence_counts.size();
                 ++case_index) {
                const int n_seq = sequence_counts[case_index];
                const std::vector<int> contexts(
                    n_seq, options.context);
                BenchResult result;
                if (!run_case(
                        backend, options, contexts, options.context,
                        false, case_index % 2 == 0, result)) {
                    std::fprintf(
                        stderr,
                        "attention benchmark failed at n_seq=%d\n",
                        n_seq);
                    ok = false;
                    break;
                }
                results.push_back(result);
            }
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "benchmark setup failed: %s\n", error.what());
        ok = false;
    }

    if (ok) {
        if (!options.ragged_contexts.empty()) {
            print_ragged_result(options, results.front());
            ggml_backend_free(backend);
            return 0;
        }

        const bool cuda_graphs_enabled =
            std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr;
        std::printf(
            "# native contiguous vs paged GPU attention; "
            "not whole-model throughput\n"
            "# D=%d,Hq=%d,Hkv=%d,block=%d,context=%d,k=%s,v=%s,"
            "warmup=%d,samples=%d,cuda_graphs_env=%s\n"
            "# paged_step_time_overhead_pct > 0 means paged is slower; "
            "paged_over_contiguous_query_throughput > 1 means paged "
            "is faster\n",
            D, N_HEAD, N_HEAD_KV, BLOCK_SIZE, options.context,
            ggml_type_name(options.k_type),
            ggml_type_name(options.v_type),
            options.warmup, options.samples,
            cuda_graphs_enabled ? "allowed" : "disabled");
        std::printf(
            "n_seq,context,k_type,v_type,"
            "contiguous_iterations,paged_iterations,"
            "contiguous_median_window_mean_us_per_step,"
            "paged_median_window_mean_us_per_step,"
            "contiguous_p95_window_mean_us_per_step,"
            "paged_p95_window_mean_us_per_step,"
            "paged_over_contiguous_step_time,"
            "paged_step_time_overhead_pct,"
            "contiguous_aggregate_attention_queries_s,"
            "paged_aggregate_attention_queries_s,"
            "paged_over_contiguous_query_throughput,"
            "contiguous_scaling_vs_b1,"
            "paged_scaling_vs_b1,"
            "contiguous_batch_efficiency,"
            "paged_batch_efficiency,"
            "contiguous_kv_mib,paged_kv_mib,"
            "paged_metadata_kib,"
            "max_abs_error,max_tolerance_ratio,rmse\n");

        const double contiguous_baseline =
            results.front().contiguous.aggregate_queries_s;
        const double paged_baseline =
            results.front().paged.aggregate_queries_s;
        for (const BenchResult & result : results) {
            const double contiguous_scaling =
                result.contiguous.aggregate_queries_s /
                contiguous_baseline;
            const double paged_scaling =
                result.paged.aggregate_queries_s / paged_baseline;
            const double step_time_ratio =
                result.paged.median_us /
                result.contiguous.median_us;
            const double throughput_ratio =
                result.paged.aggregate_queries_s /
                result.contiguous.aggregate_queries_s;
            std::printf(
                "%d,%d,%s,%s,%d,%d,"
                "%.3f,%.3f,%.3f,%.3f,"
                "%.6f,%.3f,%.2f,%.2f,%.6f,"
                "%.4f,%.4f,%.4f,%.4f,"
                "%.2f,%.2f,%.3f,"
                "%.8g,%.8g,%.8g\n",
                result.n_seq, options.context,
                ggml_type_name(options.k_type),
                ggml_type_name(options.v_type),
                result.contiguous.iterations,
                result.paged.iterations,
                result.contiguous.median_us,
                result.paged.median_us,
                result.contiguous.p95_us,
                result.paged.p95_us,
                step_time_ratio,
                (step_time_ratio - 1.0) * 100.0,
                result.contiguous.aggregate_queries_s,
                result.paged.aggregate_queries_s,
                throughput_ratio,
                contiguous_scaling,
                paged_scaling,
                contiguous_scaling / result.n_seq,
                paged_scaling / result.n_seq,
                result.contiguous_kv_mib,
                result.paged_kv_mib,
                result.paged_metadata_kib,
                result.max_abs_error,
                result.max_tolerance_ratio,
                result.rmse);
        }
    }

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
