#include "common/dflash_draft_kv.h"
#include "common/draft_swa.h"
#include "internal.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

using namespace dflash::common;

namespace {

constexpr int N_LANES = 3;
constexpr int CAPACITY = 160;
constexpr int SWA_WINDOW = 64;
constexpr uint16_t F16_ZERO = 0x0000;
constexpr uint16_t F16_NEG_INF = 0xFC00;
constexpr float MAX_ABS_ERROR = 5.0e-4f;

struct Resources {
    ggml_backend_t backend = nullptr;
    DraftWeights weights;
    std::array<DraftKvState, N_LANES> states;
    DraftKvBatchGraph batch;

    ~Resources() {
        draft_kv_batch_free(batch);
        for (DraftKvState & state : states) {
            draft_kv_free(state);
        }
        free_draft_weights(weights);
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

std::vector<bool> layer_pattern(const DraftWeights & weights) {
    std::vector<bool> pattern;
    pattern.reserve(weights.layers.size());
    for (const DraftLayer & layer : weights.layers) {
        pattern.push_back(layer.is_swa);
    }
    return pattern;
}

bool check_lane_mask(const DraftKvState & state, int committed) {
    const int window_start = committed - SWA_WINDOW;
    for (int query = 0; query < state.q_len; ++query) {
        const size_t row = static_cast<size_t>(query) * state.kv_total;
        for (int slot = 0; slot < state.cap; ++slot) {
            const int position = state.slot_pos[static_cast<size_t>(slot)];
            const bool visible =
                position >= window_start && position < committed;
            const uint16_t expected = visible ? F16_ZERO : F16_NEG_INF;
            if (state.mask_hbuf[row + static_cast<size_t>(slot)] != expected) {
                std::fprintf(stderr,
                    "lane mask mismatch committed=%d query=%d slot=%d pos=%d\n",
                    committed, query, slot, position);
                return false;
            }
        }
        for (int noise = 0; noise < state.q_len; ++noise) {
            const uint16_t expected = noise <= query ? F16_ZERO : F16_NEG_INF;
            if (state.mask_hbuf[
                    row + static_cast<size_t>(state.cap + noise)] != expected) {
                std::fprintf(stderr,
                    "lane causal mask mismatch committed=%d query=%d noise=%d\n",
                    committed, query, noise);
                return false;
            }
        }
    }
    return true;
}

float max_abs_diff(const std::vector<float> & lhs,
                   const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        return INFINITY;
    }
    float max_diff = 0.0f;
    for (size_t index = 0; index < lhs.size(); ++index) {
        if (!std::isfinite(lhs[index]) || !std::isfinite(rhs[index])) {
            return INFINITY;
        }
        max_diff = std::max(max_diff, std::fabs(lhs[index] - rhs[index]));
    }
    return max_diff;
}

struct AppendStateFamilies {
    std::array<DraftKvState, N_LANES> reference;
    std::array<DraftKvState, N_LANES> packed;

    ~AppendStateFamilies() {
        for (DraftKvState & state : reference) {
            draft_kv_free(state);
        }
        for (DraftKvState & state : packed) {
            draft_kv_free(state);
        }
    }
};

bool compute_append_graph(
        ggml_backend_t backend,
        const DraftWeights & weights,
        const std::vector<DraftKvAppendLane> & lanes) {
    std::vector<uint8_t> arena(
        16u * 1024 * 1024 * std::max<size_t>(lanes.size(), 1));
    ggml_init_params params{};
    params.mem_size = arena.size();
    params.mem_buffer = arena.data();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(
        ctx, 4096 * std::max<size_t>(lanes.size(), 1), false);
    bool ok = build_draft_kv_appends(ctx, graph, weights, lanes);
    ggml_gallocr_t allocator = nullptr;
    if (ok) {
        allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        ok = allocator &&
             ggml_gallocr_alloc_graph(allocator, graph) &&
             ggml_backend_graph_compute(backend, graph) ==
                 GGML_STATUS_SUCCESS;
    }
    if (allocator) {
        ggml_gallocr_free(allocator);
    }
    ggml_free(ctx);
    return ok;
}

void fill_append_inputs(
        DraftKvState & state,
        int lane,
        int append_count) {
    const size_t feature_elements =
        static_cast<size_t>(state.fc_in) * state.a_step;
    std::vector<float> features(feature_elements);
    for (int column = 0; column < state.a_step; ++column) {
        const int feature_column =
            column < append_count ? column : append_count;
        for (int row = 0; row < state.fc_in; ++row) {
            const size_t index =
                static_cast<size_t>(column) * state.fc_in + row;
            features[index] =
                0.025f * static_cast<float>(lane + 1) +
                0.00001f * static_cast<float>(
                    (row + 17 * feature_column) % 997);
        }
    }

    std::vector<int32_t> positions(static_cast<size_t>(state.a_step));
    std::vector<int32_t> rows(static_cast<size_t>(state.a_step));
    for (int column = 0; column < state.a_step; ++column) {
        if (column < append_count) {
            const int position = 11 + 37 * lane + column;
            positions[static_cast<size_t>(column)] = position;
            rows[static_cast<size_t>(column)] = position % state.cap;
        } else {
            positions[static_cast<size_t>(column)] = 0;
            rows[static_cast<size_t>(column)] = state.trash_slot;
        }
    }

    ggml_backend_tensor_set(
        state.ap_feat, features.data(), 0,
        features.size() * sizeof(float));
    ggml_backend_tensor_set(
        state.ap_pos, positions.data(), 0,
        positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(
        state.ap_rows, rows.data(), 0,
        rows.size() * sizeof(int32_t));
}

bool compare_cache_tensor(
        const ggml_tensor * reference,
        const ggml_tensor * packed,
        int lane,
        int layer,
        int trash_row,
        const char * kind) {
    if (reference->ne[0] != packed->ne[0] ||
        reference->ne[1] != packed->ne[1]) {
        return false;
    }

    const size_t elements = static_cast<size_t>(ggml_nelements(reference));
    std::vector<ggml_fp16_t> reference_values(elements);
    std::vector<ggml_fp16_t> packed_values(elements);
    ggml_backend_tensor_get(
        reference, reference_values.data(), 0,
        elements * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(
        packed, packed_values.data(), 0,
        elements * sizeof(ggml_fp16_t));

    const size_t row_width = static_cast<size_t>(reference->ne[0]);
    if (trash_row < 0 ||
        trash_row >= reference->ne[1]) {
        return false;
    }
    float max_error = 0.0f;
    float trash_error = 0.0f;
    size_t max_index = 0;
    for (size_t index = 0; index < elements; ++index) {
        const float reference_value =
            ggml_fp16_to_fp32(reference_values[index]);
        const float packed_value =
            ggml_fp16_to_fp32(packed_values[index]);
        if (!std::isfinite(reference_value) ||
            !std::isfinite(packed_value)) {
            return false;
        }
        const float error =
            std::fabs(reference_value - packed_value);
        if (index / row_width == static_cast<size_t>(trash_row)) {
            trash_error = std::max(trash_error, error);
        }
        if (error > max_error) {
            max_error = error;
            max_index = index;
        }
    }
    if (max_error > MAX_ABS_ERROR) {
        std::fprintf(
            stderr,
            "draft append cache mismatch lane=%d layer=%d kind=%s "
            "row=%zu trash=%d max_abs=%.6g trash_abs=%.6g\n",
            lane, layer, kind, max_index / row_width, trash_row,
            max_error, trash_error);
        return false;
    }
    return true;
}

bool check_trash_row(
        const ggml_tensor * cache,
        int trash_row,
        bool expect_written,
        int lane,
        int layer,
        const char * family,
        const char * kind) {
    const size_t row_width = static_cast<size_t>(cache->ne[0]);
    std::vector<ggml_fp16_t> values(row_width);
    ggml_backend_tensor_get(
        cache, values.data(),
        static_cast<size_t>(trash_row) * row_width * sizeof(ggml_fp16_t),
        row_width * sizeof(ggml_fp16_t));

    bool written = false;
    for (ggml_fp16_t value : values) {
        const float converted = ggml_fp16_to_fp32(value);
        if (!std::isfinite(converted)) {
            return false;
        }
        written = written || converted != 0.0f;
    }
    if (written != expect_written) {
        std::fprintf(
            stderr,
            "draft append trash mismatch family=%s lane=%d layer=%d "
            "kind=%s expected_written=%d actual_written=%d\n",
            family, lane, layer, kind,
            expect_written ? 1 : 0, written ? 1 : 0);
        return false;
    }
    return true;
}

bool check_packed_append_caches(
        ggml_backend_t backend,
        const DraftWeights & weights) {
    AppendStateFamilies families;
    for (int lane = 0; lane < N_LANES; ++lane) {
        if (!draft_kv_init_batched(
                families.reference[static_cast<size_t>(lane)],
                weights, backend, CAPACITY) ||
            !draft_kv_init_batched(
                families.packed[static_cast<size_t>(lane)],
                weights, backend, CAPACITY)) {
            std::fprintf(
                stderr,
                "draft append qualification: lane %d state init failed\n",
                lane);
            return false;
        }
    }

    const int append_width = families.reference.front().a_step;
    const std::array<int, N_LANES> append_counts{
        0, append_width / 2, append_width,
    };
    for (int lane = 0; lane < N_LANES; ++lane) {
        fill_append_inputs(
            families.reference[static_cast<size_t>(lane)],
            lane, append_counts[static_cast<size_t>(lane)]);
        fill_append_inputs(
            families.packed[static_cast<size_t>(lane)],
            lane, append_counts[static_cast<size_t>(lane)]);
    }

    for (DraftKvState & state : families.reference) {
        const std::vector<DraftKvAppendLane> lane{
            {&state.cache, state.ap_feat, state.ap_pos, state.ap_rows},
        };
        if (!compute_append_graph(backend, weights, lane)) {
            std::fprintf(
                stderr,
                "draft append qualification: reference graph failed\n");
            return false;
        }
    }

    std::vector<DraftKvAppendLane> packed_lanes;
    packed_lanes.reserve(N_LANES);
    for (DraftKvState & state : families.packed) {
        packed_lanes.push_back(
            {&state.cache, state.ap_feat, state.ap_pos, state.ap_rows});
    }
    if (!compute_append_graph(backend, weights, packed_lanes)) {
        std::fprintf(
            stderr,
            "draft append qualification: packed graph failed\n");
        return false;
    }

    for (int lane = 0; lane < N_LANES; ++lane) {
        const DraftKvState & reference =
            families.reference[static_cast<size_t>(lane)];
        const DraftKvState & packed =
            families.packed[static_cast<size_t>(lane)];
        const bool expect_trash_written =
            append_counts[static_cast<size_t>(lane)] < append_width;
        for (int layer = 0; layer < weights.n_layer; ++layer) {
            if (!compare_cache_tensor(
                    reference.cache.k[static_cast<size_t>(layer)],
                    packed.cache.k[static_cast<size_t>(layer)],
                    lane, layer, reference.trash_slot, "K") ||
                !compare_cache_tensor(
                    reference.cache.v[static_cast<size_t>(layer)],
                    packed.cache.v[static_cast<size_t>(layer)],
                    lane, layer, reference.trash_slot, "V")) {
                return false;
            }
            if (!check_trash_row(
                    reference.cache.k[static_cast<size_t>(layer)],
                    reference.trash_slot, expect_trash_written,
                    lane, layer, "reference", "K") ||
                !check_trash_row(
                    reference.cache.v[static_cast<size_t>(layer)],
                    reference.trash_slot, expect_trash_written,
                    lane, layer, "reference", "V") ||
                !check_trash_row(
                    packed.cache.k[static_cast<size_t>(layer)],
                    packed.trash_slot, expect_trash_written,
                    lane, layer, "packed", "K") ||
                !check_trash_row(
                    packed.cache.v[static_cast<size_t>(layer)],
                    packed.trash_slot, expect_trash_written,
                    lane, layer, "packed", "V")) {
                return false;
            }
        }
    }

    std::printf(
        "draft append packed cache qualification passed "
        "counts=0,%d,%d\n",
        append_width / 2, append_width);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 1) {
        std::fprintf(stderr,
            "usage: %s <dflash2.gguf> [gpu] "
            "[--append-only|--dynconv-only]\n", argv[0]);
        return 77;
    }
    if (argc > 4) {
        return 2;
    }

    int gpu = 0;
    bool gpu_set = false;
    bool append_only = false;
    bool dynconv_only = false;
    for (int arg = 2; arg < argc; ++arg) {
        if (std::strcmp(argv[arg], "--append-only") == 0) {
            append_only = true;
        } else if (std::strcmp(argv[arg], "--dynconv-only") == 0) {
            dynconv_only = true;
        } else if (gpu_set) {
            return 2;
        } else {
            gpu = std::atoi(argv[arg]);
            gpu_set = true;
        }
    }
    if (append_only && dynconv_only) {
        return 2;
    }
    Resources resources;
    resources.backend = ggml_backend_cuda_init(gpu);
    if (!resources.backend) {
        std::fprintf(stderr, "draft SWA qualification: GPU %d unavailable\n", gpu);
        return 1;
    }
    if (!load_draft_gguf(argv[1], resources.backend, resources.weights)) {
        std::fprintf(stderr, "draft SWA qualification: %s\n",
                     dflash27b_last_error());
        return 1;
    }
    if (dynconv_only &&
        (resources.weights.conv_kernel_size <= 0 ||
         resources.weights.conv_group_size <= 0 ||
         !std::all_of(resources.weights.layers.begin(), resources.weights.layers.end(),
             [](const DraftLayer & layer) {
                 return layer.attn_conv.present() && layer.mlp_conv.present();
             }))) {
        std::fprintf(stderr, "draft SWA qualification: dynamic convolution absent\n");
        return 1;
    }
    if (!check_packed_append_caches(
            resources.backend, resources.weights)) {
        return 1;
    }
    if (append_only) {
        return 0;
    }

    if (!dynconv_only) {
        const std::vector<bool> trained_pattern = layer_pattern(resources.weights);
        if (!resources.weights.swa_pattern_loaded) {
            std::fprintf(stderr,
                "draft SWA qualification: GGUF has no SWA pattern\n");
            return 1;
        }
        const DraftSwaOverrideResult swa =
            apply_draft_swa_window_override(resources.weights, SWA_WINDOW);
        if (layer_pattern(resources.weights) != trained_pattern ||
            swa.effective_window != SWA_WINDOW || swa.swa_layers == 0) {
            std::fprintf(stderr,
                "draft SWA qualification: override changed the trained pattern\n");
            return 1;
        }
    }

    DraftKvState batched_state;
    if (!draft_kv_init_batched(
            batched_state, resources.weights, resources.backend, CAPACITY) ||
        batched_state.gf || batched_state.g_ctx || batched_state.galloc ||
        !batched_state.meta_arena.empty()) {
        std::fprintf(stderr,
            "draft SWA qualification: batched init allocated a single-lane graph\n");
        draft_kv_free(batched_state);
        return 1;
    }
    draft_kv_free(batched_state);

    const std::array<int, N_LANES> committed = {65, 81, 133};
    const size_t hidden_elements =
        static_cast<size_t>(resources.weights.n_embd) *
        static_cast<size_t>(resources.weights.block_size);
    std::array<std::vector<float>, N_LANES> single_hidden;
    DraftFeatureMirror unused_ring;

    for (int lane = 0; lane < N_LANES; ++lane) {
        DraftKvState & state = resources.states[static_cast<size_t>(lane)];
        if (!draft_kv_init(state, resources.weights, resources.backend,
                           CAPACITY, nullptr)) {
            std::fprintf(stderr, "draft SWA qualification: lane %d init failed\n", lane);
            return 1;
        }
        for (int position = 0; position < committed[static_cast<size_t>(lane)];
             ++position) {
            state.slot_pos[static_cast<size_t>(position % CAPACITY)] = position;
        }
        state.next_pos = committed[static_cast<size_t>(lane)];
        if (!draft_kv_begin_step(
                state, resources.weights, resources.backend, unused_ring,
                committed[static_cast<size_t>(lane)]) ||
            (!dynconv_only &&
             !check_lane_mask(state, committed[static_cast<size_t>(lane)]))) {
            return 1;
        }

        std::vector<float> embedding(hidden_elements);
        for (size_t index = 0; index < embedding.size(); ++index) {
            const size_t token = index / resources.weights.n_embd;
            const size_t channel = index % resources.weights.n_embd;
            embedding[index] =
                0.01f * static_cast<float>(lane + 1) +
                0.0001f * static_cast<float>(
                    (channel + 17 * (lane + 3) * (token + 1)) % 127);
        }
        ggml_backend_tensor_set(state.inp_embed, embedding.data(), 0,
                                embedding.size() * sizeof(float));
        if (ggml_backend_graph_compute(resources.backend, state.gf) !=
            GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                         "draft SWA qualification: lane %d compute failed\n", lane);
            return 1;
        }
        single_hidden[static_cast<size_t>(lane)].resize(hidden_elements);
        ggml_backend_tensor_get(
            state.hidden_states,
            single_hidden[static_cast<size_t>(lane)].data(), 0,
            hidden_elements * sizeof(float));
    }

    if (max_abs_diff(single_hidden[0], single_hidden[1]) <= MAX_ABS_ERROR) {
        std::fprintf(stderr,
            "draft SWA qualification: distinct lane inputs collapsed\n");
        return 1;
    }

    std::vector<DraftKvState *> lane_states;
    lane_states.reserve(N_LANES);
    for (DraftKvState & state : resources.states) {
        lane_states.push_back(&state);
    }
    std::vector<std::vector<float>> packed_hidden;
    if (!draft_kv_batch_compute(resources.batch, resources.weights,
                                resources.backend, lane_states, packed_hidden)) {
        return 1;
    }

    for (int lane = 0; lane < N_LANES; ++lane) {
        const float error = max_abs_diff(
            single_hidden[static_cast<size_t>(lane)],
            packed_hidden[static_cast<size_t>(lane)]);
        std::printf("draft SWA lane=%d committed=%d max_abs=%.6g\n",
                    lane, committed[static_cast<size_t>(lane)], error);
        if (error > MAX_ABS_ERROR) {
            std::fprintf(stderr,
                "draft SWA qualification: lane %d exceeds tolerance %.6g\n",
                lane, MAX_ABS_ERROR);
            return 1;
        }
    }

    std::printf("draft %s three-lane qualification passed\n",
                dynconv_only ? "dynamic-convolution" : "SWA post-window");
    return 0;
}
