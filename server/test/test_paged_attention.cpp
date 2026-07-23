#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int D = 256;
constexpr int N_HEAD = 24;
constexpr int N_HEAD_KV = 4;
constexpr int BLOCK_SIZE = 16;
constexpr int MAX_BLOCKS = 3;
constexpr int N_SEQ = 6;
constexpr int PHYSICAL_BLOCKS = MAX_BLOCKS * N_SEQ;
constexpr int POOL_TOKENS = PHYSICAL_BLOCKS * BLOCK_SIZE;

const std::array<int32_t, N_SEQ> KV_SEQ_LENS = {1, 15, 16, 17, 31, 33};

// Rows are sequences. Every live block is private, and deliberately
// shuffled so a contiguous-cache implementation cannot pass this test.
// Two in-range entries are deliberately invalid — one negative, one past
// the physical pool — to pin down the kernel's bounds guard: it must skip
// those blocks (matching the oracle) instead of reading out of bounds.
const std::array<int32_t, MAX_BLOCKS * N_SEQ> BLOCK_TABLE = {
    11,  2, 15,
     4, 17,  7,
     9,  0, 13,
     5, -1,  1,   // token 16 of seq 3 maps to a negative block
    14,  8,  3,
    10, 99, 12,   // tokens 16-31 of seq 5 map past the pool
};

bool block_is_valid(int32_t block) {
    return block >= 0 && block < PHYSICAL_BLOCKS;
}

std::vector<uint8_t> quantize_rows(ggml_type type,
                                   const std::vector<float> & values,
                                   int64_t rows) {
    std::vector<uint8_t> result(
        ggml_row_size(type, D) * static_cast<size_t>(rows));
    const size_t written = ggml_quantize_chunk(
        type, values.data(), result.data(), 0, rows, D, nullptr);
    if (written != result.size()) {
        std::fprintf(stderr,
            "quantize size mismatch for %s: %zu != %zu\n",
            ggml_type_name(type), written, result.size());
        return {};
    }
    return result;
}

std::vector<float> dequantize_rows(ggml_type type,
                                   const std::vector<uint8_t> & values,
                                   int64_t rows) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) return {};

    const size_t row_bytes = ggml_row_size(type, D);
    std::vector<float> result(static_cast<size_t>(rows) * D);
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(values.data() + row * row_bytes,
                         result.data() + row * D, D);
    }
    return result;
}

std::vector<float> reference_attention(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v) {
    std::vector<float> output(q.size(), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int q_per_kv = N_HEAD / N_HEAD_KV;

    for (int seq = 0; seq < N_SEQ; ++seq) {
        const int kv_seq_len = KV_SEQ_LENS[seq];
        for (int head = 0; head < N_HEAD; ++head) {
            const int kv_head = head / q_per_kv;
            const float * q_row =
                q.data() + (static_cast<size_t>(head) * N_SEQ + seq) * D;

            std::vector<float> scores(kv_seq_len);
            float max_score = -INFINITY;
            for (int token = 0; token < kv_seq_len; ++token) {
                const int block =
                    BLOCK_TABLE[seq * MAX_BLOCKS + token / BLOCK_SIZE];
                if (!block_is_valid(block)) {
                    // Mirrors the kernel: invalid blocks contribute nothing.
                    scores[token] = -INFINITY;
                    continue;
                }
                const int physical = block * BLOCK_SIZE + token % BLOCK_SIZE;
                const float * k_row =
                    k.data() +
                    (static_cast<size_t>(kv_head) * POOL_TOKENS + physical) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += q_row[d] * k_row[d];
                scores[token] = dot * scale;
                max_score = std::max(max_score, scores[token]);
            }

            float denominator = 0.0f;
            for (float & score : scores) {
                score = std::exp(score - max_score);
                denominator += score;
            }

            float * out_row =
                output.data() + (static_cast<size_t>(head) * N_SEQ + seq) * D;
            for (int token = 0; token < kv_seq_len; ++token) {
                const int block =
                    BLOCK_TABLE[seq * MAX_BLOCKS + token / BLOCK_SIZE];
                if (!block_is_valid(block)) continue;
                const int physical = block * BLOCK_SIZE + token % BLOCK_SIZE;
                const float * v_row =
                    v.data() +
                    (static_cast<size_t>(kv_head) * POOL_TOKENS + physical) * D;
                const float probability = scores[token] / denominator;
                for (int d = 0; d < D; ++d) {
                    out_row[d] += probability * v_row[d];
                }
            }
        }
    }
    return output;
}

bool run_case(ggml_backend_t backend, ggml_type k_type, ggml_type v_type) {
    ggml_init_params params{};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_SEQ, N_HEAD);
    ggml_tensor * k =
        ggml_new_tensor_3d(ctx, k_type, D, POOL_TOKENS, N_HEAD_KV);
    ggml_tensor * v =
        ggml_new_tensor_3d(ctx, v_type, D, POOL_TOKENS, N_HEAD_KV);
    ggml_tensor * table =
        ggml_new_tensor_2d(ctx, GGML_TYPE_I32, MAX_BLOCKS, N_SEQ);
    ggml_tensor * kv_seq_lens =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_SEQ);
    for (ggml_tensor * input : {q, k, v, table, kv_seq_lens}) {
        ggml_set_input(input);
    }

    ggml_tensor * output = ggml_paged_attn(
        ctx, q, k, v, table, kv_seq_lens,
        1.0f / std::sqrt(static_cast<float>(D)), BLOCK_SIZE);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) {
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> q_data(static_cast<size_t>(D) * N_SEQ * N_HEAD);
    std::vector<float> k_source(
        static_cast<size_t>(D) * POOL_TOKENS * N_HEAD_KV);
    std::vector<float> v_source(k_source.size());
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.25f;
    }
    for (size_t i = 0; i < k_source.size(); ++i) {
        k_source[i] = std::cos(static_cast<float>(i) * 0.007f) * 0.20f;
        v_source[i] = std::sin(static_cast<float>(i) * 0.011f + 0.3f) * 0.30f;
    }

    const int64_t cache_rows = static_cast<int64_t>(POOL_TOKENS) * N_HEAD_KV;
    const std::vector<uint8_t> k_data =
        quantize_rows(k_type, k_source, cache_rows);
    const std::vector<uint8_t> v_data =
        quantize_rows(v_type, v_source, cache_rows);
    const std::vector<float> k_reference =
        dequantize_rows(k_type, k_data, cache_rows);
    const std::vector<float> v_reference =
        dequantize_rows(v_type, v_data, cache_rows);
    bool ok = !k_data.empty() && !v_data.empty() &&
              !k_reference.empty() && !v_reference.empty();

    if (ok) {
        ggml_backend_tensor_set(q, q_data.data(), 0,
                                q_data.size() * sizeof(q_data[0]));
        ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size());
        ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size());
        ggml_backend_tensor_set(table, BLOCK_TABLE.data(), 0,
                                BLOCK_TABLE.size() * sizeof(BLOCK_TABLE[0]));
        ggml_backend_tensor_set(
            kv_seq_lens, KV_SEQ_LENS.data(), 0,
            KV_SEQ_LENS.size() * sizeof(KV_SEQ_LENS[0]));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }

    float max_abs_error = INFINITY;
    if (ok) {
        std::vector<float> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(actual[0]));
        const std::vector<float> expected =
            reference_attention(q_data, k_reference, v_reference);
        max_abs_error = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) {
                ok = false;
                break;
            }
            max_abs_error =
                std::max(max_abs_error, std::fabs(actual[i] - expected[i]));
        }
        ok = ok && max_abs_error < 3.0e-3f;
    }

    std::printf("paged attention K=%-4s V=%-4s max_abs=%.6g %s\n",
                ggml_type_name(k_type), ggml_type_name(v_type),
                max_abs_error, ok ? "PASS" : "FAIL");
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "GPU backend unavailable\n");
        return 1;
    }

    const ggml_type types[] = {
        GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
    };
    bool ok = true;
    for (ggml_type k_type : types) {
        for (ggml_type v_type : types) {
            ok = run_case(backend, k_type, v_type) && ok;
        }
    }

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
