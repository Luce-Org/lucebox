// Correctness coverage for the CUDA/HIP concat fast path used by Qwen's
// DeltaNet convolution staging:
//
//   contiguous [prefix, channels, sequences]
//     + transpose(contiguous [channels, tokens, sequences])
//     -> contiguous [prefix + tokens, channels, sequences].
//
// The operation is a copy, so every output bit must match the host reference.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct ConcatCase {
    const char * name;
    int prefix;
    int tokens;
    int channels;
    int sequences;
    bool expect_fast_path;
};

float tagged_value(uint32_t tag, size_t index) {
    // Finite, non-NaN values with deterministic and varied payload bits.
    const uint32_t bits = tag | (uint32_t) (index % 0x007fffffu);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool run_case(ggml_backend_t backend, const ConcatCase & tc) {
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "%s: ggml_init failed\n", tc.name);
        return false;
    }

    ggml_tensor * history = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, tc.prefix, tc.channels, tc.sequences);
    ggml_tensor * qkv = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, tc.channels, tc.tokens, tc.sequences);
    ggml_set_input(history);
    ggml_set_input(qkv);

    ggml_tensor * qkv_transposed = ggml_transpose(ctx, qkv);
    ggml_tensor * output = ggml_concat(ctx, history, qkv_transposed, 0);
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    bool ok = allocator && ggml_gallocr_alloc_graph(allocator, graph);
    bool output_available = false;
    bool dispatch_matches = false;
    bool data_match = false;
    const size_t fast_path_before =
        ggml_backend_cuda_get_concat_transpose_f32_count();

    const size_t history_elements =
        (size_t) tc.prefix * tc.channels * tc.sequences;
    const size_t qkv_elements =
        (size_t) tc.channels * tc.tokens * tc.sequences;
    const size_t output_elements =
        (size_t) (tc.prefix + tc.tokens) * tc.channels * tc.sequences;
    std::vector<float> history_data(history_elements);
    std::vector<float> qkv_data(qkv_elements);
    std::vector<float> expected(output_elements);
    std::vector<float> actual(output_elements);

    for (size_t i = 0; i < history_data.size(); ++i) {
        history_data[i] = tagged_value(0xbf000000u, i);
    }
    for (size_t i = 0; i < qkv_data.size(); ++i) {
        qkv_data[i] = tagged_value(0x3f000000u, i);
    }

    for (int s = 0; s < tc.sequences; ++s) {
        for (int c = 0; c < tc.channels; ++c) {
            const size_t output_row =
                ((size_t) s * tc.channels + c) * (tc.prefix + tc.tokens);
            const size_t history_row =
                ((size_t) s * tc.channels + c) * tc.prefix;
            for (int p = 0; p < tc.prefix; ++p) {
                expected[output_row + p] = history_data[history_row + p];
            }
            for (int t = 0; t < tc.tokens; ++t) {
                const size_t qkv_index =
                    ((size_t) s * tc.tokens + t) * tc.channels + c;
                expected[output_row + tc.prefix + t] = qkv_data[qkv_index];
            }
        }
    }

    if (ok) {
        ggml_backend_tensor_set(
            history, history_data.data(), 0, history_data.size() * sizeof(float));
        ggml_backend_tensor_set(
            qkv, qkv_data.data(), 0, qkv_data.size() * sizeof(float));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        const size_t fast_path_after =
            ggml_backend_cuda_get_concat_transpose_f32_count();
        dispatch_matches = ok && fast_path_after ==
            fast_path_before + (tc.expect_fast_path ? 1 : 0);
    }
    if (ok) {
        ggml_backend_tensor_get(
            output, actual.data(), 0, actual.size() * sizeof(float));
        output_available = true;
        data_match = std::memcmp(
            actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
        ok = data_match && dispatch_matches;
    }

    if (!ok && !output_available) {
        std::fprintf(stderr, "%s: graph allocation or compute failed\n", tc.name);
    } else if (!data_match) {
        size_t mismatch = 0;
        while (mismatch < actual.size() &&
               std::memcmp(&actual[mismatch], &expected[mismatch],
                           sizeof(float)) == 0) {
            ++mismatch;
        }
        if (mismatch < actual.size()) {
            uint32_t actual_bits;
            uint32_t expected_bits;
            std::memcpy(&actual_bits, &actual[mismatch], sizeof(actual_bits));
            std::memcpy(&expected_bits, &expected[mismatch], sizeof(expected_bits));
            std::fprintf(
                stderr,
                "%s: mismatch at %zu actual=0x%08x expected=0x%08x\n",
                tc.name, mismatch, actual_bits, expected_bits);
        }
    } else if (!dispatch_matches) {
        std::fprintf(stderr, "%s: unexpected concat dispatch path\n", tc.name);
    } else {
        std::printf(
            "concat transpose %-20s P=%d T=%d C=%d S=%d PASS\n",
            tc.name, tc.prefix, tc.tokens, tc.channels, tc.sequences);
    }

    if (allocator) {
        ggml_gallocr_free(allocator);
    }
    ggml_free(ctx);
    return ok;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "GPU backend unavailable\n");
        return 1;
    }

    // Exercise decode, both tile boundaries, both tails, multiple sequences,
    // the actual Qwen channel width, and the strict prefix>32 fallback.
    const ConcatCase cases[] = {
        {"decode_tail",       3,   1,    65, 3, true},
        {"tiny",              1,   2,    33, 2, true},
        {"token_tail",        3,  31,    63, 1, true},
        {"exact_tile",        7,  32,    64, 2, true},
        {"both_tails",        3,  33,    65, 3, true},
        {"wide_tail",         3, 511,   130, 1, true},
        {"max_prefix",       32, 513,   129, 1, true},
        {"qwen_prefill",      3, 512, 10240, 1, true},
        {"prefix_fallback",  33,  17,    35, 2, false},
    };

    bool ok = true;
    for (const ConcatCase & tc : cases) {
        ok = run_case(backend, tc) && ok;
    }

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
