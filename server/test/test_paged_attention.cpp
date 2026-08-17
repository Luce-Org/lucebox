#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

using namespace CppUnitTestFramework;

namespace {

constexpr int D = 256;
constexpr int N_HEAD = 24;
constexpr int N_HEAD_KV = 4;
constexpr int BLOCK_SIZE = 16;
constexpr float MAX_ABS_ERROR = 5.0e-4f;

struct TestCase {
    const char * name;
    int max_blocks;
    std::vector<int32_t> kv_seq_lens;
    bool corrupt_blocks;
};

int clamped_seq_len(const TestCase & test_case, int seq) {
    return std::max(
        0, std::min(
               test_case.kv_seq_lens[seq],
               test_case.max_blocks * BLOCK_SIZE));
}

int count_physical_blocks(const TestCase & test_case) {
    int result = 0;
    for (size_t seq = 0; seq < test_case.kv_seq_lens.size(); ++seq) {
        const int kv_seq_len =
            clamped_seq_len(test_case, static_cast<int>(seq));
        result += (kv_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    return result;
}

bool block_is_valid(int32_t block, int physical_blocks) {
    return block >= 0 && block < physical_blocks;
}

std::vector<int32_t> make_block_table(
        const TestCase & test_case,
        int physical_blocks) {
    const int n_seq = static_cast<int>(test_case.kv_seq_lens.size());
    std::vector<int32_t> result(
        static_cast<size_t>(test_case.max_blocks) * n_seq, -1);

    // 37 is coprime with the test cases' live-block counts, so this maps every
    // logical page to a unique shuffled physical page.
    int ordinal = 0;
    for (int seq = 0; seq < n_seq; ++seq) {
        const int kv_seq_len = clamped_seq_len(test_case, seq);
        const int n_blocks =
            (kv_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int logical = 0; logical < n_blocks; ++logical) {
            result[seq * test_case.max_blocks + logical] =
                (ordinal * 37 + 11) % physical_blocks;
            ++ordinal;
        }
    }

    std::set<int32_t> assigned(result.begin(), result.end());
    assigned.erase(-1);
    GGML_ASSERT(static_cast<int>(assigned.size()) == physical_blocks);

    if (test_case.corrupt_blocks) {
        // Two in-range entries are deliberately invalid — one negative, one
        // past the physical pool — to pin down the physical-block bounds
        // guard. Sequence 9 spans 65 blocks, so its corrupt page also
        // exercises the partitioned long-context variant.
        result[4 * test_case.max_blocks + 1] = -1;
        result[9 * test_case.max_blocks + 7] = physical_blocks + 9;
    }
    return result;
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
        const TestCase & test_case,
        const std::vector<int32_t> & block_table,
        int pool_tokens,
        int physical_blocks,
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<int32_t> * active_slot_ids = nullptr,
        const std::vector<int32_t> * query_positions = nullptr) {
    std::vector<float> output(q.size(), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int q_per_kv = N_HEAD / N_HEAD_KV;
    const int physical_n_seq = static_cast<int>(test_case.kv_seq_lens.size());
    const int n_seq = active_slot_ids
        ? static_cast<int>(active_slot_ids->size())
        : physical_n_seq;

    for (int seq = 0; seq < n_seq; ++seq) {
        const int physical_seq = active_slot_ids ? (*active_slot_ids)[seq] : seq;
        // Mirrors the kernel: out-of-range slot ids and negative positions
        // are padding rows and leave zero output.
        if (physical_seq < 0 || physical_seq >= physical_n_seq) continue;
        int kv_seq_len = clamped_seq_len(test_case, physical_seq);
        if (query_positions && (*query_positions)[seq] < kv_seq_len) {
            // The inclusive causal clamp: row seq attends its sequence's
            // cached tokens [0, position].
            kv_seq_len = (*query_positions)[seq] + 1;
        }
        for (int head = 0; head < N_HEAD; ++head) {
            const int kv_head = head / q_per_kv;
            const float * q_row =
                q.data() + (static_cast<size_t>(head) * n_seq + seq) * D;

            std::vector<float> scores(kv_seq_len);
            float max_score = -INFINITY;
            for (int token = 0; token < kv_seq_len; ++token) {
                const int block =
                    block_table[
                        physical_seq * test_case.max_blocks + token / BLOCK_SIZE];
                if (!block_is_valid(block, physical_blocks)) {
                    // Mirrors the kernel: invalid blocks contribute nothing.
                    scores[token] = -INFINITY;
                    continue;
                }
                const int physical = block * BLOCK_SIZE + token % BLOCK_SIZE;
                const float * k_row =
                    k.data() +
                    (static_cast<size_t>(kv_head) * pool_tokens + physical) * D;
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
                output.data() + (static_cast<size_t>(head) * n_seq + seq) * D;
            for (int token = 0; token < kv_seq_len; ++token) {
                const int block =
                    block_table[
                        physical_seq * test_case.max_blocks + token / BLOCK_SIZE];
                if (!block_is_valid(block, physical_blocks)) continue;
                const int physical = block * BLOCK_SIZE + token % BLOCK_SIZE;
                const float * v_row =
                    v.data() +
                    (static_cast<size_t>(kv_head) * pool_tokens + physical) * D;
                const float probability = scores[token] / denominator;
                for (int d = 0; d < D; ++d) {
                    out_row[d] += probability * v_row[d];
                }
            }
        }
    }
    return output;
}

bool run_case(ggml_backend_t backend,
              const TestCase & test_case,
              ggml_type k_type,
              ggml_type v_type,
              const std::vector<int32_t> * active_slot_ids = nullptr,
              const std::vector<int32_t> * query_positions = nullptr) {
    const int physical_n_seq = static_cast<int>(test_case.kv_seq_lens.size());
    const int n_seq = active_slot_ids
        ? static_cast<int>(active_slot_ids->size())
        : physical_n_seq;
    // Ragged causal positions ride on the compact row -> slot mapping.
    GGML_ASSERT(!query_positions ||
                (active_slot_ids &&
                 query_positions->size() == active_slot_ids->size()));
    const int physical_blocks = count_physical_blocks(test_case);
    const int pool_tokens = physical_blocks * BLOCK_SIZE;
    const std::vector<int32_t> block_table =
        make_block_table(test_case, physical_blocks);

    ggml_init_params params{};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_seq, N_HEAD);
    ggml_tensor * k =
        ggml_new_tensor_3d(ctx, k_type, D, pool_tokens, N_HEAD_KV);
    ggml_tensor * v =
        ggml_new_tensor_3d(ctx, v_type, D, pool_tokens, N_HEAD_KV);
    ggml_tensor * table =
        ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, test_case.max_blocks, physical_n_seq);
    ggml_tensor * kv_seq_lens =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, physical_n_seq);
    for (ggml_tensor * input : {q, k, v, table, kv_seq_lens}) {
        ggml_set_input(input);
    }

    ggml_tensor * active = nullptr;
    if (active_slot_ids) {
        active = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seq);
        ggml_set_input(active);
    }
    ggml_tensor * positions = nullptr;
    if (query_positions) {
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seq);
        ggml_set_input(positions);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int max_kv_seq_len = *std::max_element(
        test_case.kv_seq_lens.begin(), test_case.kv_seq_lens.end());
    ggml_tensor * output = ggml_paged_attn_ext(
        ctx, q, k, v, table, kv_seq_lens, active, positions,
        scale, BLOCK_SIZE, max_kv_seq_len);
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

    std::vector<float> q_data(static_cast<size_t>(D) * n_seq * N_HEAD);
    std::vector<float> k_source(
        static_cast<size_t>(D) * pool_tokens * N_HEAD_KV);
    std::vector<float> v_source(k_source.size());
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.25f;
    }
    for (size_t i = 0; i < k_source.size(); ++i) {
        k_source[i] = std::cos(static_cast<float>(i) * 0.007f) * 0.20f;
        v_source[i] = std::sin(static_cast<float>(i) * 0.011f + 0.3f) * 0.30f;
    }

    const int64_t cache_rows = static_cast<int64_t>(pool_tokens) * N_HEAD_KV;
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
        ggml_backend_tensor_set(
            table, block_table.data(), 0,
            block_table.size() * sizeof(block_table[0]));
        ggml_backend_tensor_set(
            kv_seq_lens, test_case.kv_seq_lens.data(), 0,
            test_case.kv_seq_lens.size() *
                sizeof(test_case.kv_seq_lens[0]));
        if (active_slot_ids) {
            ggml_backend_tensor_set(
                active, active_slot_ids->data(), 0,
                active_slot_ids->size() * sizeof((*active_slot_ids)[0]));
        }
        if (query_positions) {
            ggml_backend_tensor_set(
                positions, query_positions->data(), 0,
                query_positions->size() * sizeof((*query_positions)[0]));
        }
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }

    float max_abs_error = INFINITY;
    if (ok) {
        std::vector<float> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(actual[0]));
        const std::vector<float> expected =
            reference_attention(
                test_case, block_table, pool_tokens, physical_blocks,
                q_data, k_reference, v_reference, active_slot_ids,
                query_positions);
        max_abs_error = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) {
                ok = false;
                break;
            }
            max_abs_error =
                std::max(max_abs_error, std::fabs(actual[i] - expected[i]));
        }
        ok = ok && max_abs_error < MAX_ABS_ERROR;
    }

    std::printf("paged attention %-11s K=%-4s V=%-4s active=%s pos=%s max_abs=%.6g %s\n",
                test_case.name, ggml_type_name(k_type), ggml_type_name(v_type),
                active_slot_ids ? "yes" : "no",
                query_positions ? "yes" : "no",
                max_abs_error, ok ? "PASS" : "FAIL");
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

bool rejects_unlaunchable_gqa(ggml_backend_t backend) {
    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    // Ratio 34 cannot fit the one-head-per-warp fallback in a CUDA block and
    // is not divisible by either wider specialization (3 or 6 heads/warp).
    // The support query must reject it before dispatch indexes the occupancy
    // cache or attempts an oversized launch.
    ggml_tensor * q =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 34);
    ggml_tensor * k =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, BLOCK_SIZE, 1);
    ggml_tensor * v =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, BLOCK_SIZE, 1);
    ggml_tensor * table =
        ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_tensor * kv_seq_lens =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * output = ggml_paged_attn_ext(
        ctx, q, k, v, table, kv_seq_lens, nullptr, nullptr,
        1.0f / std::sqrt(static_cast<float>(D)), BLOCK_SIZE, 1);

    const bool rejected = !ggml_backend_supports_op(backend, output);
    std::printf("paged attention unlaunchable GQA support %s\n",
                rejected ? "PASS" : "FAIL");
    ggml_free(ctx);
    return rejected;
}

}  // namespace

struct PagedAttention : CommonFixture {
    using CommonFixture::CommonFixture;

void run_paged_attention_case(const TestCase & test_case) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);

    const ggml_type types[] = {
        GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
    };
    CHECK(rejects_unlaunchable_gqa(backend));
    for (ggml_type k_type : types) {
        for (ggml_type v_type : types) {
            CHECK(run_case(backend, test_case, k_type, v_type));
        }
    }

    ggml_backend_free(backend);
}

void run_active_slot_case(const TestCase & test_case,
                          const std::vector<int32_t> & active_slot_ids) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    CHECK(run_case(backend, test_case, GGML_TYPE_F16, GGML_TYPE_F16,
                   &active_slot_ids));
    ggml_backend_free(backend);
}

void run_ragged_case(const TestCase & test_case,
                     const std::vector<int32_t> & active_slot_ids,
                     const std::vector<int32_t> & query_positions) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    CHECK(run_case(backend, test_case, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
                   &active_slot_ids, &query_positions));
    ggml_backend_free(backend);
}

};
TEST_CASE(PagedAttention, PartitionedPathMatchesReference) {
    run_paged_attention_case({
        "partitioned",
        65,
        {-7, 1, 15, 16, 17, 31, 33, 257, 511, 1025, 2000},
        true,
    });
}

TEST_CASE(PagedAttention, DirectPathMatchesReference) {
    luce_test::ScopedEnvVar force_partitions(
        "GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS",
        "1"
    );
    run_paged_attention_case({
        "direct",
        64,
        {0, 1, 15, 16, 17, 257, 511},
        false,
    });
}

TEST_CASE(PagedAttention, ActiveSlotsMatchReference) {
    run_active_slot_case({
        "active-slots",
        65,
        {-7, 1, 15, 16, 17, 31, 33, 257, 511, 1025, 2000},
        true,
    }, {10, 9, -1, 4});
}

TEST_CASE(PagedAttention, CompactThreeSlotBucketMatchesReference) {
    // Three physical slots round up to a four-row graph bucket. The final row
    // is padding and maps to no physical block-table column.
    run_active_slot_case({
        "compact-3",
        2,
        {1, 17, 31},
        false,
    }, {2, 1, 0, -1});
}

TEST_CASE(PagedAttention, RaggedCausalPositionsMatchReference) {
    // Interleaved query rows from two sequences attend the paged pool causally
    // through per-row positions. Sequence 1 spans 65 logical blocks, so its
    // full-prefix row also exercises the partitioned path.
    const std::vector<int32_t> active_slot_ids{1, 0, 1, 0, 0, -1};
    // First token, short prefix, multi-partition full prefix, block-aligned
    // prefix, an over-range clamp, and a padding row.
    const std::vector<int32_t> query_positions{
        0, 4, 1024, 31, INT32_MAX, -1,
    };
    run_ragged_case({
        "ragged",
        65,
        {33, 1025},
        false,
    }, active_slot_ids, query_positions);
}
