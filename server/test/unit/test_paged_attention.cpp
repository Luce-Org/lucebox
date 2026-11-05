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
#include <numeric>
#include <set>
#include <utility>
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
    // Chain-over-holes cases (C-3). 0 none, 1 partial-null partitions, 2 whole-null partitions
    // (128 tokens), 3 an all-null table, 4 modes 1, 2, 3 on slots 0, 1, 2 (then repeating).
    // The last logical block of a slot stays mapped in modes 1 and 2: a KVFlash tail is resident.
    int  gap_mode = 0;
    bool poison_stale = false;    // pool rows no table entry names hold large K/V, as an evicted chunk's would
    bool poison_tail = false;     // rows above the committed length inside the mapped tail block hold large K/V, as a rejected draft leaves them
};

struct TreeMetadata {
    int width;
    int scratch_stride;
    std::vector<int32_t> parent_ids;
    std::vector<int32_t> tree_sizes;
    int ar_rows = 0;
    bool chain = false;   // every tree is a chain (parent n-1): build the chain op, which the WMMA route may take
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

bool tree_visible(
        const TreeMetadata & tree,
        int tree_seq,
        int query_node,
        int candidate) {
    const int tree_size = tree.tree_sizes[tree_seq];
    if (tree_size < 0 || tree_size > tree.width ||
        query_node < 0 || query_node >= tree_size ||
        candidate < 0 || candidate >= tree_size) {
        return false;
    }

    bool visible = false;
    int current = query_node;
    for (int depth = 0; depth < tree_size; ++depth) {
        if (current == candidate) {
            visible = true;
        }
        if (current < 0 || current >= tree_size) {
            return false;
        }
        const int parent =
            tree.parent_ids[tree_seq * tree.width + current];
        if (parent == -1) {
            return visible;
        }
        if (parent < 0 || parent >= current) {
            return false;
        }
        current = parent;
    }
    return false;
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
    if (test_case.gap_mode != 0) {
        for (int seq = 0; seq < n_seq; ++seq) {
            const int mode = test_case.gap_mode == 4 ? 1 + seq % 3 : test_case.gap_mode;
            const int n_blocks = (clamped_seq_len(test_case, seq) + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (int logical = 0; logical < n_blocks; ++logical) {
                const bool is_tail = logical == n_blocks - 1;
                const int partition_block = logical % 8;   // 8 blocks of 16 = one 128-token partition
                const bool null_it =
                    mode == 3 ||
                    (!is_tail && mode == 1 && (partition_block == 1 || partition_block == 2 || partition_block == 5)) ||
                    (!is_tail && mode == 2 && (logical / 8) % 2 == 1);
                if (null_it) result[seq * test_case.max_blocks + logical] = -1;
            }
        }
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
        const std::vector<int32_t> * query_positions = nullptr,
        const TreeMetadata * tree = nullptr,
        int tree_scratch_base = 0) {
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
        const bool tree_query = tree && seq >= tree->ar_rows;
        const int tree_seq = tree_query
            ? (seq - tree->ar_rows) / tree->width : 0;
        const int query_node = tree_query
            ? (seq - tree->ar_rows) % tree->width : -1;
        int kv_seq_len = clamped_seq_len(test_case, physical_seq);
        if (query_positions && !tree_query) {
            const int32_t query_position = (*query_positions)[seq];
            if (query_position < 0) continue;
            // The inclusive causal clamp: row seq attends its sequence's
            // cached tokens [0, position].
            if (query_position < kv_seq_len) {
                kv_seq_len = query_position + 1;
            }
        }
        const int tree_size = tree_query ? tree->tree_sizes[tree_seq] : 0;
        if (tree_query &&
            (tree_size < 0 || tree_size > tree->width ||
             query_node >= tree_size)) {
            continue;
        }

        std::vector<int32_t> physical_rows;
        physical_rows.reserve(
            kv_seq_len + (tree_query ? tree->width : 0));
        for (int token = 0; token < kv_seq_len; ++token) {
            const int block =
                block_table[
                    physical_seq * test_case.max_blocks +
                    token / BLOCK_SIZE];
            physical_rows.push_back(
                block_is_valid(block, physical_blocks)
                    ? block * BLOCK_SIZE + token % BLOCK_SIZE
                    : -1);
        }
        if (tree_query) {
            for (int candidate = 0; candidate < tree->width; ++candidate) {
                physical_rows.push_back(
                    tree_visible(*tree, tree_seq, query_node, candidate)
                        ? tree_scratch_base +
                              physical_seq * tree->scratch_stride + candidate
                        : -1);
            }
        }
        for (int head = 0; head < N_HEAD; ++head) {
            const int kv_head = head / q_per_kv;
            const float * q_row =
                q.data() + (static_cast<size_t>(head) * n_seq + seq) * D;

            std::vector<float> scores(physical_rows.size(), -INFINITY);
            float max_score = -INFINITY;
            for (size_t row = 0; row < physical_rows.size(); ++row) {
                const int physical = physical_rows[row];
                if (physical < 0) continue;
                const float * k_row =
                    k.data() +
                    (static_cast<size_t>(kv_head) * pool_tokens + physical) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += q_row[d] * k_row[d];
                scores[row] = dot * scale;
                max_score = std::max(max_score, scores[row]);
            }

            float denominator = 0.0f;
            for (float & score : scores) {
                if (!std::isfinite(score)) {
                    score = 0.0f;
                    continue;
                }
                score = std::exp(score - max_score);
                denominator += score;
            }

            float * out_row =
                output.data() + (static_cast<size_t>(head) * n_seq + seq) * D;
            for (size_t row = 0; row < physical_rows.size(); ++row) {
                const int physical = physical_rows[row];
                if (physical < 0 || denominator == 0.0f) continue;
                const float * v_row =
                    v.data() +
                    (static_cast<size_t>(kv_head) * pool_tokens + physical) * D;
                const float probability = scores[row] / denominator;
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
              const std::vector<int32_t> * query_positions = nullptr,
              const TreeMetadata * tree = nullptr) {
    const int physical_n_seq = static_cast<int>(test_case.kv_seq_lens.size());
    const int n_seq = active_slot_ids
        ? static_cast<int>(active_slot_ids->size())
        : physical_n_seq;
    // Ragged causal positions ride on the compact row -> slot mapping.
    GGML_ASSERT(!query_positions ||
                (active_slot_ids &&
                 query_positions->size() == active_slot_ids->size()));
    GGML_ASSERT(!tree || active_slot_ids);
    if (tree) {
        GGML_ASSERT(tree->width > 0);
        GGML_ASSERT(tree->scratch_stride >= tree->width);
        GGML_ASSERT(tree->ar_rows >= 0);
        GGML_ASSERT(tree->ar_rows == 0 || query_positions);
        GGML_ASSERT(
            tree->parent_ids.size() ==
            static_cast<size_t>(tree->width) * tree->tree_sizes.size());
        GGML_ASSERT(
            n_seq == tree->ar_rows +
            tree->width * static_cast<int>(tree->tree_sizes.size()));
    }
    const int physical_blocks = count_physical_blocks(test_case);
    const int tree_scratch_base = physical_blocks * BLOCK_SIZE;
    const int pool_tokens = tree
        ? tree_scratch_base + physical_n_seq * tree->scratch_stride
        : tree_scratch_base;
    GGML_ASSERT(pool_tokens % BLOCK_SIZE == 0);
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
    ggml_tensor * parents = nullptr;
    ggml_tensor * sizes = nullptr;
    if (tree) {
        parents = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, tree->width, tree->tree_sizes.size());
        sizes = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, tree->tree_sizes.size());
        ggml_set_input(parents);
        ggml_set_input(sizes);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int max_kv_seq_len = *std::max_element(
        test_case.kv_seq_lens.begin(), test_case.kv_seq_lens.end());
    ggml_tensor * output = tree
        ? (tree->chain ? ggml_paged_attn_ext_chain : ggml_paged_attn_ext_tree)(
            ctx, q, k, v, table, kv_seq_lens, active,
            positions, scale, BLOCK_SIZE, max_kv_seq_len, parents, sizes,
            tree_scratch_base, tree->scratch_stride)
        : ggml_paged_attn_ext(
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

    if (test_case.poison_stale) {
        std::vector<char> mapped(static_cast<size_t>(physical_blocks), 0);
        for (int32_t entry : block_table) {
            if (block_is_valid(entry, physical_blocks)) mapped[static_cast<size_t>(entry)] = 1;
        }
        for (int block = 0; block < physical_blocks; ++block) {
            if (mapped[static_cast<size_t>(block)]) continue;
            for (int head = 0; head < N_HEAD_KV; ++head) {
                for (int row = block * BLOCK_SIZE; row < (block + 1) * BLOCK_SIZE; ++row) {
                    for (int d = 0; d < D; ++d) {
                        const size_t at = (static_cast<size_t>(head) * pool_tokens + row) * D + d;
                        k_source[at] = 2.0f;
                        v_source[at] = 50.0f;
                    }
                }
            }
        }
    }
    if (test_case.poison_tail) {
        for (size_t seq = 0; seq < test_case.kv_seq_lens.size(); ++seq) {
            const int len = clamped_seq_len(test_case, static_cast<int>(seq));
            const int tail = (len - 1) / BLOCK_SIZE;
            const int32_t entry = block_table[seq * test_case.max_blocks + tail];
            if (!block_is_valid(entry, physical_blocks)) continue;
            for (int head = 0; head < N_HEAD_KV; ++head) {
                for (int row = entry * BLOCK_SIZE + (len - tail * BLOCK_SIZE); row < (entry + 1) * BLOCK_SIZE; ++row) {
                    for (int d = 0; d < D; ++d) {
                        const size_t at = (static_cast<size_t>(head) * pool_tokens + row) * D + d;
                        k_source[at] = 2.0f;
                        v_source[at] = 50.0f;
                    }
                }
            }
        }
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
        if (tree) {
            ggml_backend_tensor_set(
                parents, tree->parent_ids.data(), 0,
                tree->parent_ids.size() * sizeof(tree->parent_ids[0]));
            ggml_backend_tensor_set(
                sizes, tree->tree_sizes.data(), 0,
                tree->tree_sizes.size() * sizeof(tree->tree_sizes[0]));
        }
        // Poison the output first, so a row the kernel fails to write shows
        // up as a mismatch instead of whatever the allocator left there.
        const std::vector<float> poison(ggml_nelements(output), 1.0e30f);
        ggml_backend_tensor_set(output, poison.data(), 0, ggml_nbytes(output));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }

    float max_abs_error = INFINITY;
    int drifted_runs = 0;   // the same graph on the same inputs must give the same bits every launch
    if (ok) {
        std::vector<float> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(actual[0]));
        std::vector<float> again(actual.size());
        for (int rep = 0; rep < 31 && ok; ++rep) {
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(output, again.data(), 0, again.size() * sizeof(again[0]));
            if (std::memcmp(again.data(), actual.data(), actual.size() * sizeof(actual[0])) != 0) ++drifted_runs;
        }
        if (drifted_runs) { std::printf("paged attention %s: %d of 31 relaunches differ bitwise\n", test_case.name, drifted_runs); ok = false; }
        const std::vector<float> expected =
            reference_attention(
                test_case, block_table, pool_tokens, physical_blocks,
                q_data, k_reference, v_reference, active_slot_ids,
                query_positions, tree, tree ? tree_scratch_base : 0);
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

    std::printf("paged attention %-11s K=%-4s V=%-4s active=%s pos=%s tree=%s max_abs=%.6g %s\n",
                test_case.name, ggml_type_name(k_type), ggml_type_name(v_type),
                active_slot_ids ? "yes" : "no",
                query_positions ? "yes" : "no",
                tree ? "yes" : "no",
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

void run_tree_case() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    const TestCase tree_case{"tree", 65, {1025, 17, 257}, false};
    const TreeMetadata tree_metadata{
        16, 16,
        {
            -1, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14,
            -1, 0, 1, 2, 3, 4, 5, 6,
            7, -1, -1, -1, -1, -1, -1, -1,
        },
        {16, 9},
    };
    const std::vector<int32_t> tree_slot_ids{
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2,
    };
    CHECK(run_case(backend, tree_case, GGML_TYPE_F16, GGML_TYPE_F16,
                   &tree_slot_ids, nullptr, &tree_metadata));
    CHECK(run_case(backend, tree_case, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
                   &tree_slot_ids, nullptr, &tree_metadata));
    CHECK(run_case(backend, tree_case, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
                   &tree_slot_ids, nullptr, &tree_metadata));
    // The same chains through the chain op (the WMMA route on gfx1201).
    TreeMetadata chain_metadata = tree_metadata;
    chain_metadata.chain = true;
    for (const auto & kv : {std::make_pair(GGML_TYPE_F16, GGML_TYPE_F16),
                            std::make_pair(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0),
                            std::make_pair(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)}) {
        CHECK(run_case(backend, tree_case, kv.first, kv.second,
                       &tree_slot_ids, nullptr, &chain_metadata));
    }
    ggml_backend_free(backend);
}

void run_mixed_tree_case() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    const TestCase mixed_case{"mixed-tree", 8, {33, 65, 17}, false};
    const TreeMetadata tree_metadata{
        16, 16,
        {
            -1, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14,
        },
        {16},
        2,
    };
    std::vector<int32_t> query_slots{1, 0};
    query_slots.insert(query_slots.end(), 16, 2);
    std::vector<int32_t> query_positions{7, 15};
    query_positions.insert(query_positions.end(), 16, -1);
    CHECK(run_case(backend, mixed_case, GGML_TYPE_F16, GGML_TYPE_F16,
                   &query_slots, &query_positions, &tree_metadata));
    TreeMetadata chain_metadata = tree_metadata;
    chain_metadata.chain = true;
    CHECK(run_case(backend, mixed_case, GGML_TYPE_F16, GGML_TYPE_F16,
                   &query_slots, &query_positions, &chain_metadata));
    CHECK(run_case(backend, mixed_case, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                   &query_slots, &query_positions, &chain_metadata));
    ggml_backend_free(backend);
}

void run_cyclic_tree_case() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    const TestCase tree_case{"cyclic-tree", 4, {17}, false};
    const TreeMetadata tree_metadata{
        4, 16,
        {-1, 2, 1, -1},
        {3},
    };
    const std::vector<int32_t> tree_slot_ids{0, 0, 0, 0};
    CHECK(run_case(backend, tree_case, GGML_TYPE_F16, GGML_TYPE_F16,
                   &tree_slot_ids, nullptr, &tree_metadata));
    ggml_backend_free(backend);
}

// Stage C-3: chain verify (a tree with parent node-1) over KVFlash-style gapped block tables.
// Every logical length stays the full logical length (as seq_lens_ does in a KVFlash arena); the
// table carries null entries, the pool rows behind unmapped blocks hold poison (stale data of an
// evicted chunk), and the chain's scratch rows sit beyond every mapped block.
void run_gapped_chain_case(const TestCase & test_case, int width,
                           std::vector<std::pair<ggml_type, ggml_type>> kv_types = {
                               {GGML_TYPE_F16, GGML_TYPE_F16}, {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}}) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    const int n_seq = static_cast<int>(test_case.kv_seq_lens.size());
    TreeMetadata meta{width, 16, {}, std::vector<int32_t>(static_cast<size_t>(n_seq), width)};
    std::vector<int32_t> slot_ids;
    for (int s = 0; s < n_seq; ++s) {
        for (int n = 0; n < width; ++n) {
            meta.parent_ids.push_back(n - 1);   // a chain: node n's parent is n-1, the root's is -1
            slot_ids.push_back(s);
        }
    }
    for (const auto & kv : kv_types) {
        CHECK(run_case(backend, test_case, kv.first, kv.second, &slot_ids, nullptr, &meta));
    }
    meta.chain = true;   // the same gapped chains through the chain op
    for (const auto & kv : kv_types) {
        CHECK(run_case(backend, test_case, kv.first, kv.second, &slot_ids, nullptr, &meta));
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
                     const std::vector<int32_t> & query_positions,
                     std::vector<std::pair<ggml_type, ggml_type>> kv_types = {{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}}) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    for (const auto & kv : kv_types) {
        CHECK(run_case(backend, test_case, kv.first, kv.second, &active_slot_ids, &query_positions));
    }
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

TEST_CASE(PagedAttention, CompactThreeSlotBucketDirectPathMatchesReference) {
    // Same padded bucket with one partition: the direct path must zero the
    // padding row itself. The variable is read once per process, so this
    // holds when ctest runs the case alone.
    luce_test::ScopedEnvVar force_partitions(
        "GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS",
        "1"
    );
    run_active_slot_case({
        "compact-3-direct",
        2,
        {1, 17, 31},
        false,
    }, {2, 1, 0, -1});
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

TEST_CASE(PagedAttention, PackedTreesMatchReference) {
    run_tree_case();
}

TEST_CASE(PagedAttention, CompactArAndFixedChainMatchReference) {
    run_mixed_tree_case();
}

TEST_CASE(PagedAttention, CyclicParentMetadataIsMasked) {
    run_cyclic_tree_case();
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

// A KVFlash slot's table has holes where chunks are paged out. Prefill and decode rows (no tree) attend around them
// on every route: slot 0 has partial-null partitions, slot 1 whole-null ones, slot 2 an all-null table. A hole that
// enters the softmax as a zero-score token dilutes the row: position 47 of slot 0 sees 16 mapped tokens and 32 holes.
TEST_CASE(PagedAttention, RaggedRowsOverNullBlocksMatchReference) {
    const std::vector<int32_t> active_slot_ids{0, 1, 2, 0, 1, 0, 0, -1};
    const std::vector<int32_t> query_positions{1024, 516, 256, 300, 130, 47, INT32_MAX, -1};
    run_ragged_case({"ragged-gap", 130, {1025, 517, 257}, false, 4, true, true}, active_slot_ids, query_positions,
                    {{GGML_TYPE_F16, GGML_TYPE_F16}, {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}, {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}});
}

// gap_mode: 1 partial-null partitions, 2 whole-null partitions, 3 an all-null table, 4 one mode per slot.
// poison_stale: unmapped pool blocks hold large K/V, so a read through a null entry fails the case.
TEST_CASE(PagedAttention, ChainVerifyWidth16OverPartialNullPartitionsMatchesReference) {
    run_gapped_chain_case({"gap-partial16", 130, {1025, 517, 257}, false, 1, true, true}, 16);
}
TEST_CASE(PagedAttention, ChainVerifyWidth8OverPartialNullPartitionsMatchesReference) {
    run_gapped_chain_case({"gap-partial8", 130, {1025, 517, 257}, false, 1, true, true}, 8);
}
TEST_CASE(PagedAttention, ChainVerifyWidth16OverWholeNullPartitionsMatchesReference) {
    run_gapped_chain_case({"gap-whole16", 130, {1025, 517, 257}, false, 2, true, true}, 16);
}
TEST_CASE(PagedAttention, ChainVerifyWidth8OverWholeNullPartitionsMatchesReference) {
    run_gapped_chain_case({"gap-whole8", 130, {1025, 517, 257}, false, 2, true, true}, 8);
}
TEST_CASE(PagedAttention, ChainVerifyOverAnAllNullTableReadsOnlyItsScratchRows) {
    run_gapped_chain_case({"gap-allnull16", 130, {1025, 517, 257}, false, 3, true, true}, 16);
    run_gapped_chain_case({"gap-allnull8", 130, {1025, 517, 257}, false, 3, true, true}, 8);
}
TEST_CASE(PagedAttention, ChainVerifyWithADifferentGapPatternPerSlotMatchesReference) {
    // slot 0 partial-null, slot 1 whole-null partitions, slot 2 all-null, in one verify batch
    run_gapped_chain_case({"gap-mixed16", 130, {1025, 517, 257}, false, 4, true, true}, 16);
    run_gapped_chain_case({"gap-mixed8", 130, {1025, 517, 257}, false, 4, true, true}, 8);
}
TEST_CASE(PagedAttention, ChainVerifyOverGapsWithMixedKvTypesMatchesReference) {
    run_gapped_chain_case({"gap-q8q4", 130, {1025, 517, 257}, false, 1, true, true}, 16,
                          {{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}, {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}});
}

// ── Batch-composition invariance ────────────────────────────────────────────
// A row's attention bits must not depend on what else is in the batch. The
// KVFlash scorer picks between near-tie chunk scores, so a kernel whose
// context split follows the batch (neighbour lengths, n_rows, the 256-rounded
// launch length) turns two-lane scheduling noise into different picks
// (pr_series/rc3_spec/WMMA_DETERMINISM.md: 31/55 requests differ run to run).
// Row A, a decode row over a long committed prefix, is computed alone and then
// inside batches of very different shapes; every A output must be
// memcmp-identical to the alone run.
namespace {

struct InvarianceRow {
    int slot;   // block-table column
    int pos;    // causal position; -1 is a dead row
};

struct InvariancePool {
    ggml_type kv_type;
    int max_blocks;                 // block-table width: the config constant
    int pool_tokens;
    std::vector<int32_t> table;     // [max_blocks x n_slots]
    std::vector<int32_t> seq_lens;  // per slot
    std::vector<uint8_t> kv_bytes;  // one pool, used for K and for V
};

uint32_t invariance_rng = 0x9e3779b9u;
float invariance_next() {
    invariance_rng = invariance_rng * 1664525u + 1013904223u;
    return ((int32_t) (invariance_rng >> 8) - 8388608) / 8388608.0f;
}

// Slots: 0 = A (a_len, optional KVFlash holes), 1 = long neighbour, 2 = short
// neighbour. The neighbours alias A's physical blocks at an offset, so one pool
// serves all three.
InvariancePool make_invariance_pool(ggml_type kv_type, int a_len, int long_len,
                                    int short_len, bool holes) {
    InvariancePool pool;
    pool.kv_type = kv_type;
    pool.max_blocks = 8192;
    const int pool_blocks = (std::max(a_len, long_len) + BLOCK_SIZE - 1) / BLOCK_SIZE + 4;
    pool.pool_tokens = pool_blocks * BLOCK_SIZE;
    pool.seq_lens = {a_len, long_len, short_len};
    pool.table.assign((size_t) pool.max_blocks * 3, -1);
    for (int s = 0; s < 3; ++s) {
        const int n_blocks = (pool.seq_lens[s] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int b = 0; b < n_blocks; ++b) {
            if (s == 0 && holes && (b / 4) % 3 == 1) continue;   // whole 64-token chunks missing
            pool.table[(size_t) s * pool.max_blocks + b] = (b + 5 * s) % pool_blocks;
        }
    }
    const size_t rows = (size_t) pool.pool_tokens * N_HEAD_KV;
    std::vector<float> src(rows * D);
    for (float & x : src) x = invariance_next();
    pool.kv_bytes = quantize_rows(kv_type, src, (int64_t) rows);
    return pool;
}

// Runs one batch and returns row `a_row`'s output (N_HEAD x D floats), empty on
// failure. Q for a row is a function of its slot only, so A's query is the same
// in every batch.
std::vector<float> run_invariance_batch(ggml_backend_t backend, const InvariancePool & pool,
                                        const std::vector<InvarianceRow> & rows,
                                        int max_kv_seq_len, int a_row, size_t * wmma_launches) {
    const int n_rows = (int) rows.size();
    ggml_init_params params{};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return {};

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_rows, N_HEAD);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, pool.kv_type, D, pool.pool_tokens, N_HEAD_KV);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, pool.kv_type, D, pool.pool_tokens, N_HEAD_KV);
    ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, pool.max_blocks, 3);
    ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ggml_tensor * active = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
    for (ggml_tensor * t : {q, k, v, table, seq_lens, active, positions}) ggml_set_input(t);
    ggml_tensor * output = ggml_paged_attn_ext(
        ctx, q, k, v, table, seq_lens, active, positions,
        1.0f / std::sqrt((float) D), BLOCK_SIZE, max_kv_seq_len);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) {
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return {};
    }

    std::vector<float> q_data((size_t) D * n_rows * N_HEAD);
    std::vector<int32_t> active_ids(n_rows), pos(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        active_ids[r] = rows[r].slot;
        pos[r] = rows[r].pos;
        for (int h = 0; h < N_HEAD; ++h) {
            for (int d = 0; d < D; ++d) {
                q_data[((size_t) h * n_rows + r) * D + d] =
                    std::sin((float) (rows[r].slot * 7919 + h * 263 + d) * 0.013f) * 0.25f;
            }
        }
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, pool.kv_bytes.data(), 0, pool.kv_bytes.size());
    ggml_backend_tensor_set(v, pool.kv_bytes.data(), 0, pool.kv_bytes.size());
    ggml_backend_tensor_set(table, pool.table.data(), 0, pool.table.size() * sizeof(int32_t));
    ggml_backend_tensor_set(seq_lens, pool.seq_lens.data(), 0, pool.seq_lens.size() * sizeof(int32_t));
    ggml_backend_tensor_set(active, active_ids.data(), 0, active_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    {
        std::vector<float> sentinel(ggml_nelements(output), std::nanf(""));
        ggml_backend_tensor_set(output, sentinel.data(), 0, ggml_nbytes(output));
    }

    const size_t before = ggml_backend_cuda_get_paged_attn_wmma256_launch_count();
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (wmma_launches) *wmma_launches = ggml_backend_cuda_get_paged_attn_wmma256_launch_count() - before;

    std::vector<float> a_out;
    if (ok) {
        std::vector<float> all(ggml_nelements(output));
        ggml_backend_tensor_get(output, all.data(), 0, ggml_nbytes(output));
        a_out.resize((size_t) N_HEAD * D);
        for (int h = 0; h < N_HEAD; ++h) {
            std::memcpy(a_out.data() + (size_t) h * D,
                        all.data() + ((size_t) h * n_rows + a_row) * D, D * sizeof(float));
        }
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return a_out;
}

int round_up(int x, int m) { return (x + m - 1) / m * m; }

// One A length and KV type: A alone is the reference; every other batch must
// reproduce A's bytes. Returns the number of batches whose A output differs.
int invariance_mismatches(ggml_backend_t backend, ggml_type kv_type, int a_len, bool holes,
                          size_t * wmma_launches) {
    const int long_len = 100352, short_len = 300;
    const InvariancePool pool = make_invariance_pool(kv_type, a_len, long_len, short_len, holes);
    const InvarianceRow A{0, a_len - 1}, L{1, long_len - 1}, S{2, short_len - 1}, X{2, -1};
    struct Batch {
        const char * name;
        std::vector<InvarianceRow> rows;
        int a_row;
        int max_kv_seq_len;
    };
    const Batch batches[] = {
        {"alone",            {A},                         0, round_up(a_len, 256)},
        {"short-before",     {S, A},                      1, round_up(a_len, 256)},
        {"long-after",       {A, L},                      0, round_up(long_len, 256)},
        {"long-before",      {L, A},                      1, round_up(long_len, 256)},
        {"mixed-group",      {L, S, L, A, S},             3, round_up(long_len, 256)},
        {"dead-rows",        {A, X, X, X, X, X, X},       0, round_up(a_len, 256)},
        {"launch-4096",      {A},                         0, round_up(a_len, 4096)},
        {"launch-table",     {A},                         0, 8192 * BLOCK_SIZE},
        {"second-group",     {L, L, L, L, A, S, S},       4, round_up(long_len, 256)},
        {"three-short",      {S, S, S, A},                3, round_up(a_len, 256)},
    };
    std::vector<float> reference;
    int mismatches = 0;
    for (const Batch & b : batches) {
        size_t launches = 0;
        const std::vector<float> out =
            run_invariance_batch(backend, pool, b.rows, b.max_kv_seq_len, b.a_row, &launches);
        bool finite = !out.empty();
        for (float x : out) finite = finite && std::isfinite(x);
        if (reference.empty()) {
            reference = out;
            if (wmma_launches) *wmma_launches = launches;
        }
        const bool same = finite && out.size() == reference.size() &&
            std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0;
        size_t n_diff = 0;
        for (size_t i = 0; i < std::min(out.size(), reference.size()); ++i) {
            n_diff += std::memcmp(&out[i], &reference[i], sizeof(float)) != 0;
        }
        std::printf("paged attention invariance K/V=%-4s A=%-6d holes=%d %-13s rows=%zu max_kv=%-6d %s%s\n",
                    ggml_type_name(kv_type), a_len, holes ? 1 : 0, b.name, b.rows.size(),
                    b.max_kv_seq_len, same ? "SAME" : "DIFF", finite ? "" : " (non-finite)");
        if (!same) {
            ++mismatches;
            std::printf("    %zu of %zu floats differ from the alone run\n", n_diff, reference.size());
        }
    }
    return mismatches;
}

}  // namespace

TEST_CASE(PagedAttention, RowOutputBitsIndependentOfBatch) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    for (ggml_type kv_type : {GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
        for (bool holes : {false, true}) {
            for (int a_len : {70000, 20480, 4096}) {
                const int mismatches =
                    invariance_mismatches(backend, kv_type, a_len, holes, nullptr);
                // Both routes (WMMA on gfx12, V_DOT2 on the 3090 and with
                // LUCE_PAGED_WMMA=0) cut the context at fixed absolute blocks,
                // so every length and batch shape must reproduce A's bits.
                CHECK_EQUAL(0, mismatches);
            }
        }
    }
    ggml_backend_free(backend);
}

// ── Chain verify vs sequential AR ───────────────────────────────────────────
// Spec decode must equal AR in bits. A chain verify of width k over a slot with
// L committed tokens must give row n the bytes of the AR call at position L+n:
// one query row, kv_seq_lens = L+n+1, the candidate rows read from the block
// table. The chain call sees only the L committed rows through the table (the
// rows above L hold poison there, as a rejected draft leaves them) and the
// candidates' K/V in the scratch slab. On gfx1201 the AR call runs the WMMA
// kernel, so a chain verify that drops to V_DOT2 differs in bits. The 3090 runs
// V_DOT2 for both, so there this checks the layout and the scratch addressing,
// not the route.
namespace {

struct ChainPool {
    int committed;
    int width;
    int max_blocks;
    int pool_tokens;
    int scratch_base;
    int scratch_stride;
    std::vector<int32_t> table;
    std::vector<uint8_t> k_ar, v_ar;       // candidate rows at their table positions
    std::vector<uint8_t> k_chain, v_chain; // poison there, the candidates in the scratch slab
};

ChainPool make_chain_pool(ggml_type kv_type, int committed, int width, int max_blocks) {
    ChainPool p;
    p.committed = committed;
    p.width = width;
    p.max_blocks = max_blocks;
    p.scratch_stride = 16;
    const int live_blocks = (committed + width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int pool_blocks = live_blocks + 3;
    p.scratch_base = pool_blocks * BLOCK_SIZE;
    p.pool_tokens = p.scratch_base + p.scratch_stride;
    p.table.assign(max_blocks, -1);
    int stride = 7;   // any stride coprime with the pool size maps the logical blocks to distinct shuffled ones
    while (std::gcd(stride, pool_blocks) != 1) ++stride;
    for (int b = 0; b < live_blocks; ++b) p.table[b] = (b * stride + 2) % pool_blocks;
    {   // the shuffle must be a permutation of distinct blocks
        std::set<int32_t> seen(p.table.begin(), p.table.begin() + live_blocks);
        GGML_ASSERT((int) seen.size() == live_blocks);
    }
    const auto row_of = [&](int token) {
        return p.table[token / BLOCK_SIZE] * BLOCK_SIZE + token % BLOCK_SIZE;
    };
    const size_t rows = (size_t) p.pool_tokens * N_HEAD_KV;
    std::vector<float> k_src(rows * D), v_src(rows * D);
    for (float & x : k_src) x = invariance_next() * 0.5f;
    for (float & x : v_src) x = invariance_next() * 0.5f;
    std::vector<float> k_poison = k_src, v_poison = v_src;
    for (int head = 0; head < N_HEAD_KV; ++head) {
        for (int j = 0; j < width; ++j) {
            const size_t at = ((size_t) head * p.pool_tokens + row_of(committed + j)) * D;
            const size_t scratch = ((size_t) head * p.pool_tokens + p.scratch_base + j) * D;
            for (int d = 0; d < D; ++d) {
                k_src[scratch + d] = k_src[at + d];
                v_src[scratch + d] = v_src[at + d];
                k_poison[at + d] = 3.0f;
                v_poison[at + d] = 60.0f;
            }
        }
    }
    // The chain pool carries the candidates in scratch; the AR pool keeps them in the table.
    std::vector<float> k_chain_src = k_poison, v_chain_src = v_poison;
    for (int head = 0; head < N_HEAD_KV; ++head) {
        for (int j = 0; j < width; ++j) {
            const size_t scratch = ((size_t) head * p.pool_tokens + p.scratch_base + j) * D;
            for (int d = 0; d < D; ++d) {
                k_chain_src[scratch + d] = k_src[scratch + d];
                v_chain_src[scratch + d] = v_src[scratch + d];
            }
        }
    }
    p.k_ar = quantize_rows(kv_type, k_src, (int64_t) rows);
    p.v_ar = quantize_rows(kv_type, v_src, (int64_t) rows);
    p.k_chain = quantize_rows(kv_type, k_chain_src, (int64_t) rows);
    p.v_chain = quantize_rows(kv_type, v_chain_src, (int64_t) rows);
    return p;
}

// n_rows == 1: the AR decode call at kv_len. n_rows == width: the chain verify.
// q is [head][row][D]. Returns the output in the same layout, empty on failure.
std::vector<float> run_chain_or_ar(ggml_backend_t backend, ggml_type kv_type, const ChainPool & p,
                                   bool chain, int kv_len, int max_kv_seq_len,
                                   const std::vector<float> & q_data, int n_rows,
                                   size_t * wmma_launches) {
    ggml_init_params params{};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return {};
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_rows, N_HEAD);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, kv_type, D, p.pool_tokens, N_HEAD_KV);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, kv_type, D, p.pool_tokens, N_HEAD_KV);
    ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, p.max_blocks, 1);
    ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * active = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
    ggml_tensor * parents = chain ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, p.width, 1) : nullptr;
    ggml_tensor * sizes = chain ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1) : nullptr;
    for (ggml_tensor * t : {q, k, v, table, seq_lens, active, parents, sizes}) {
        if (t) ggml_set_input(t);
    }
    const float scale = 1.0f / std::sqrt((float) D);
    ggml_tensor * output = chain
        ? ggml_paged_attn_ext_chain(ctx, q, k, v, table, seq_lens, active, nullptr, scale, BLOCK_SIZE,
                                    max_kv_seq_len, parents, sizes, p.scratch_base, p.scratch_stride)
        : ggml_paged_attn_ext(ctx, q, k, v, table, seq_lens, active, nullptr, scale, BLOCK_SIZE,
                              max_kv_seq_len);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<float> result;
    if (ggml_gallocr_alloc_graph(allocator, graph)) {
        const std::vector<uint8_t> & kb = chain ? p.k_chain : p.k_ar;
        const std::vector<uint8_t> & vb = chain ? p.v_chain : p.v_ar;
        ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
        ggml_backend_tensor_set(k, kb.data(), 0, kb.size());
        ggml_backend_tensor_set(v, vb.data(), 0, vb.size());
        ggml_backend_tensor_set(table, p.table.data(), 0, p.table.size() * sizeof(int32_t));
        ggml_backend_tensor_set(seq_lens, &kv_len, 0, sizeof(int32_t));
        const std::vector<int32_t> slots(n_rows, 0);
        ggml_backend_tensor_set(active, slots.data(), 0, slots.size() * sizeof(int32_t));
        if (chain) {
            std::vector<int32_t> parent_ids(p.width);
            for (int n = 0; n < p.width; ++n) parent_ids[n] = n - 1;
            const int32_t size = p.width;
            ggml_backend_tensor_set(parents, parent_ids.data(), 0, parent_ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(sizes, &size, 0, sizeof(int32_t));
        }
        const size_t before = ggml_backend_cuda_get_paged_attn_wmma256_launch_count();
        if (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS) {
            result.resize(ggml_nelements(output));
            ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
        }
        if (wmma_launches) *wmma_launches = ggml_backend_cuda_get_paged_attn_wmma256_launch_count() - before;
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return result;
}

// Returns the number of chain rows whose bytes differ from the AR call at the same position.
int chain_vs_ar_mismatches(ggml_backend_t backend, ggml_type kv_type, int committed, int width,
                           int max_blocks, bool expect_wmma) {
    const ChainPool pool = make_chain_pool(kv_type, committed, width, max_blocks);
    std::vector<float> q_data((size_t) D * width * N_HEAD);
    for (float & x : q_data) x = invariance_next() * 0.5f;
    size_t chain_launches = 0;
    const std::vector<float> chain_out = run_chain_or_ar(
        backend, kv_type, pool, true, committed, round_up(committed, 256), q_data, width, &chain_launches);
    int mismatches = chain_out.empty() ? width : 0;
    for (int n = 0; n < width && !chain_out.empty(); ++n) {
        std::vector<float> q_row((size_t) D * N_HEAD);
        for (int h = 0; h < N_HEAD; ++h) {
            std::memcpy(q_row.data() + (size_t) h * D, q_data.data() + ((size_t) h * width + n) * D, D * sizeof(float));
        }
        size_t ar_launches = 0;
        const std::vector<float> ar = run_chain_or_ar(
            backend, kv_type, pool, false, committed + n + 1, round_up(committed + n + 1, 256), q_row, 1, &ar_launches);
        bool same = ar.size() == (size_t) N_HEAD * D;
        for (int h = 0; h < N_HEAD && same; ++h) {
            same = std::memcmp(ar.data() + (size_t) h * D, chain_out.data() + ((size_t) h * width + n) * D,
                               D * sizeof(float)) == 0;
        }
        mismatches += same ? 0 : 1;
        if (n == 0 || !same) {
            std::printf("paged attention chain-vs-AR K/V=%-4s L=%-5d k=%-2d row %-2d %s (AR wmma=%zu chain wmma=%zu)\n",
                        ggml_type_name(kv_type), committed, width, n, same ? "SAME" : "DIFF",
                        ar_launches, chain_launches);
        }
        // The route must be the same one: a chain that falls to V_DOT2 while AR runs WMMA is the bug.
        if ((ar_launches > 0) != (chain_launches > 0)) ++mismatches;
    }
    if (expect_wmma && chain_launches == 0) ++mismatches;
    return mismatches;
}

}  // namespace

TEST_CASE(PagedAttention, ChainVerifyRowsEqualSequentialArBits) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE_NOT_NULL(backend);
    for (ggml_type kv_type : {GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
        // Where the WMMA route is eligible (RDNA4, LUCE_PAGED_WMMA not 0) the chain verify must run it too.
        const bool expect_wmma = ggml_backend_cuda_paged_attn_wmma_route(
            0, /*tree_mode=*/false, D, N_HEAD, N_HEAD_KV, kv_type, kv_type, BLOCK_SIZE);
        // L odd and even, on and off a block edge, one and several partitions, a 130-block table.
        for (int committed : {33, 64, 1025, 2047}) {
            for (int width : {16, 8}) {
                CHECK_EQUAL(0, chain_vs_ar_mismatches(backend, kv_type, committed, width, 130, expect_wmma));
            }
        }
    }
    ggml_backend_free(backend);
}
