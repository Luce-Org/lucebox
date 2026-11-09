// Bit parity of the stream-ordered tree commit
// (ggml_backend_cuda_tree_commit_transaction_async) with the synchronous
// ggml_backend_cuda_tree_commit_transaction.
//
// One fixed chain round's commit is set up with random contents: recurrent
// layers of mixed shapes (the 128-wide scalar-gate tile, a 64-wide scalar
// gate, a 32-wide vector gate), three K/V caches with different row sizes and
// head counts, and a feature ring, for four lanes with a parked lane (slot -1),
// unwritten rows (-1) and varied accepted prefixes. The synchronous commit
// runs on one copy, the async one on another (its index tensors uploaded
// through a pinned stage on the backend stream just before, as the engine
// does), and every destination must match byte for byte.
//
// Further cases: a second commit with new indices on the same tensors (the
// cached placement), two layers sharing one recurrent state (destinations
// overlap, so the commit runs layer by layer in argument order), and a
// rejected preflight that must leave every tensor untouched.
#include "CppUnitTestFramework.hpp"
#include "common/pinned_stage.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using luce::common::PinnedStage;

constexpr int kTreeWidth = 4;     // tokens per lane
constexpr int kLanes = 4;         // tree bucket
constexpr int kStateSlots = 6;
constexpr int kConvWindow = 3;
constexpr int kConvChannels = 96;
constexpr int kScratchBase = 40;  // committed rows live below it
constexpr int kScratchStride = kTreeWidth;
constexpr int kCacheRows = kScratchBase + kStateSlots * kScratchStride;
constexpr int kFeatureWidth = 40;
constexpr int kFeatureSourceRows = kLanes * kTreeWidth + 2;  // two AR rows
constexpr int kFeatureCapacity = 48;

struct LayerSpec {
    int state_size;
    int heads;
    bool scalar_gate;  // replay log rows of 2S+1, else 3S
};

struct CacheSpec {
    ggml_type type;
    int row_elements;
    int heads;
};

// The host's copy of one round's index tensors.
struct Indices {
    std::vector<int64_t> commit_rows;
    std::vector<int32_t> accepted;
    std::vector<int32_t> slots;
    std::vector<int32_t> feature_rows;
};

Indices make_indices(const std::vector<int32_t> & slots,
                     const std::vector<int32_t> & accepted,
                     unsigned seed) {
    Indices indices;
    indices.slots = slots;
    indices.accepted = accepted;
    std::mt19937 rng(seed);
    // Distinct committed rows below the scratch region, some left unwritten.
    std::vector<int64_t> rows(kScratchBase);
    for (int row = 0; row < kScratchBase; ++row) rows[(size_t) row] = row;
    std::shuffle(rows.begin(), rows.end(), rng);
    size_t next = 0;
    indices.commit_rows.assign((size_t) kLanes * kTreeWidth, -1);
    for (int lane = 0; lane < kLanes; ++lane) {
        if (slots[(size_t) lane] < 0) continue;
        for (int node = 0; node < accepted[(size_t) lane]; ++node) {
            const bool skip = (lane + node) % 5 == 4;
            indices.commit_rows[(size_t) lane * kTreeWidth + node] =
                skip ? -1 : rows[next++];
        }
    }
    std::vector<int32_t> feature(kFeatureCapacity);
    for (int row = 0; row < kFeatureCapacity; ++row) feature[(size_t) row] = row;
    std::shuffle(feature.begin(), feature.end(), rng);
    indices.feature_rows.assign(kFeatureSourceRows, -1);
    for (int row = 0; row < kFeatureSourceRows; ++row) {
        if (row % 3 != 2) indices.feature_rows[(size_t) row] = feature[(size_t) row];
    }
    return indices;
}

// Every tensor of one commit in one device buffer, with the initial bytes of
// each so a case can start over from the same contents.
struct Round {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> replay_logs, states, conv_inputs, conv_states;
    std::vector<ggml_tensor *> caches;
    ggml_tensor * feature_source = nullptr;
    ggml_tensor * feature_destination = nullptr;
    ggml_tensor * feature_rows = nullptr;
    ggml_tensor * commit_rows = nullptr;
    ggml_tensor * accepted = nullptr;
    ggml_tensor * slots = nullptr;
    std::vector<ggml_tensor *> all;
    std::vector<std::vector<uint8_t>> initial;

    // shared_state: layer 1 commits into layer 0's state and conv window.
    Round(ggml_backend_t backend, const std::vector<LayerSpec> & layers,
          const std::vector<CacheSpec> & cache_specs, bool shared_state) {
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * 128;
        params.no_alloc = true;
        ctx = ggml_init(params);
        for (size_t index = 0; index < layers.size(); ++index) {
            const LayerSpec & layer = layers[index];
            const int width = layer.scalar_gate
                ? 2 * layer.state_size + 1 : 3 * layer.state_size;
            replay_logs.push_back(ggml_new_tensor_4d(
                ctx, GGML_TYPE_F32, width, layer.heads, kTreeWidth, kLanes));
            conv_inputs.push_back(ggml_new_tensor_4d(
                ctx, GGML_TYPE_F32, kConvWindow + kTreeWidth, kConvChannels,
                kLanes, 1));
            if (shared_state && index == 1) {
                states.push_back(states[0]);
                conv_states.push_back(conv_states[0]);
                continue;
            }
            states.push_back(ggml_new_tensor_4d(
                ctx, GGML_TYPE_F32, layer.state_size, layer.state_size,
                layer.heads, kStateSlots));
            conv_states.push_back(ggml_new_tensor_4d(
                ctx, GGML_TYPE_F32, kConvWindow, kConvChannels, kStateSlots, 1));
        }
        for (const CacheSpec & spec : cache_specs) {
            caches.push_back(ggml_new_tensor_4d(
                ctx, spec.type, spec.row_elements, kCacheRows, spec.heads, 1));
        }
        feature_source = ggml_new_tensor_2d(
            ctx, GGML_TYPE_BF16, kFeatureWidth, kFeatureSourceRows);
        feature_destination = ggml_new_tensor_2d(
            ctx, GGML_TYPE_BF16, kFeatureWidth, kFeatureCapacity);
        feature_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kFeatureSourceRows);
        commit_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, kTreeWidth, kLanes);
        accepted = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kLanes);
        slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kLanes);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
             t = ggml_get_next_tensor(ctx, t)) {
            all.push_back(t);
        }
    }
    ~Round() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    Round(const Round &) = delete;
    Round & operator=(const Round &) = delete;

    // Random contents. Recurrent values stay small so four steps stay finite.
    void fill(unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> value(-0.9f, 0.9f);
        std::uniform_int_distribution<int> byte(0, 255);
        initial.clear();
        for (ggml_tensor * t : all) {
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            if (t->type == GGML_TYPE_F32) {
                for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
                    const float f = value(rng);
                    std::memcpy(bytes.data() + i, &f, 4);
                }
            } else {
                for (uint8_t & b : bytes) b = (uint8_t) byte(rng);
            }
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
            initial.push_back(std::move(bytes));
        }
    }

    void restore() {
        for (size_t i = 0; i < all.size(); ++i) {
            ggml_backend_tensor_set(all[i], initial[i].data(), 0, initial[i].size());
        }
    }

    std::vector<std::vector<uint8_t>> snapshot() const {
        std::vector<std::vector<uint8_t>> out;
        for (ggml_tensor * t : all) {
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
            out.push_back(std::move(bytes));
        }
        return out;
    }

    void upload_blocking(const Indices & indices) {
        ggml_backend_tensor_set(commit_rows, indices.commit_rows.data(), 0,
                                indices.commit_rows.size() * sizeof(int64_t));
        ggml_backend_tensor_set(accepted, indices.accepted.data(), 0,
                                indices.accepted.size() * sizeof(int32_t));
        ggml_backend_tensor_set(slots, indices.slots.data(), 0,
                                indices.slots.size() * sizeof(int32_t));
        ggml_backend_tensor_set(feature_rows, indices.feature_rows.data(), 0,
                                indices.feature_rows.size() * sizeof(int32_t));
    }

    bool commit_sync(const Indices & indices) {
        upload_blocking(indices);
        return ggml_backend_cuda_tree_commit_transaction(
            caches.data(), (int) caches.size(),
            feature_source, feature_destination, feature_rows,
            replay_logs.data(), states.data(), conv_inputs.data(),
            conv_states.data(), (int) replay_logs.size(),
            commit_rows, accepted, slots, kScratchBase, kScratchStride);
    }

    // The engine's sequence: indices through a pinned stage on the backend
    // stream, the commit queued behind them, one wait at the end.
    bool commit_async(ggml_backend_t backend, PinnedStage & stage,
                      const Indices & indices) {
        stage.reset();
        stage.set(backend, commit_rows, indices.commit_rows.data(), 0,
                  indices.commit_rows.size() * sizeof(int64_t));
        stage.set(backend, accepted, indices.accepted.data(), 0,
                  indices.accepted.size() * sizeof(int32_t));
        stage.set(backend, slots, indices.slots.data(), 0,
                  indices.slots.size() * sizeof(int32_t));
        stage.set(backend, feature_rows, indices.feature_rows.data(), 0,
                  indices.feature_rows.size() * sizeof(int32_t));
        const ggml_cuda_tree_commit_host_maps maps{
            indices.commit_rows.data(), indices.accepted.data(),
            indices.slots.data(), indices.feature_rows.data(),
        };
        const bool ok = ggml_backend_cuda_tree_commit_transaction_async(
            backend, caches.data(), (int) caches.size(),
            feature_source, feature_destination, feature_rows,
            replay_logs.data(), states.data(), conv_inputs.data(),
            conv_states.data(), (int) replay_logs.size(),
            commit_rows, accepted, slots, kScratchBase, kScratchStride, &maps);
        ggml_backend_synchronize(backend);
        return ok;
    }
};

const std::vector<LayerSpec> kMixedLayers = {
    {128, 2, true}, {128, 2, true}, {128, 2, true},  // one tile launch
    {64, 3, true},                                   // generic, scalar gate
    {32, 2, false},                                  // generic, vector gate
    {128, 2, true},                                  // a new run of 128
};
const std::vector<CacheSpec> kCaches = {
    {GGML_TYPE_F16, 128, 2},
    {GGML_TYPE_F16, 96, 4},
    {GGML_TYPE_F32, 40, 1},
};

struct TreeCommitAsyncFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;

    template <typename F> void each_gpu(F && f) {
        const int n_dev = ggml_backend_cuda_get_device_count();
        if (n_dev <= 0) SKIP("no GPU");
        for (int dev = 0; dev < n_dev; ++dev) {
            ggml_backend_ptr be(ggml_backend_cuda_init(dev));
            REQUIRE(be != nullptr);
            f(be.get());
        }
    }

    // The synchronous commit and the async one from the same start: every
    // tensor byte for byte.
    void check_parity(ggml_backend_t backend, Round & round, PinnedStage & stage,
                      const Indices & indices, const char * what) {
        round.restore();
        REQUIRE(round.commit_sync(indices));
        const auto expected = round.snapshot();
        round.restore();
        REQUIRE(round.commit_async(backend, stage, indices));
        const auto actual = round.snapshot();
        REQUIRE(expected.size() == actual.size());
        int differing = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (expected[i] != actual[i]) {
                ++differing;
                std::printf("  %s: tensor %zu (%s) differs\n", what, i,
                            ggml_type_name(round.all[i]->type));
            }
        }
        std::printf("  [%s] %s\n", differing == 0 ? "PASS" : "FAIL", what);
        CHECK(differing == 0);
    }
};

}  // namespace

TEST_CASE(TreeCommitAsyncFixture, matches_sync_commit) {
    each_gpu([&](ggml_backend_t backend) {
        Round round(backend, kMixedLayers, kCaches, false);
        REQUIRE(round.buffer != nullptr);
        round.fill(11);
        PinnedStage stage;
        REQUIRE(stage.reserve(backend, 1 << 16));

        const Indices first = make_indices({2, -1, 0, 5}, {4, 2, 1, 3}, 5);
        check_parity(backend, round, stage, first, "mixed layers and caches");
        // Same tensors, new indices: the placement is cached, the values are
        // not.
        const Indices second = make_indices({5, 0, -1, 3}, {1, 4, 0, 2}, 6);
        check_parity(backend, round, stage, second, "cached placement");
        check_parity(backend, round, stage, first, "cached placement again");
    });
}

TEST_CASE(TreeCommitAsyncFixture, shared_state_commits_in_layer_order) {
    each_gpu([&](ggml_backend_t backend) {
        const std::vector<LayerSpec> layers = {
            {128, 2, true}, {128, 2, true}, {64, 3, true}};
        Round round(backend, layers, kCaches, true);
        REQUIRE(round.buffer != nullptr);
        round.fill(23);
        PinnedStage stage;
        REQUIRE(stage.reserve(backend, 1 << 16));
        const Indices indices = make_indices({1, 4, 3, -1}, {3, 4, 2, 1}, 9);
        check_parity(backend, round, stage, indices, "shared recurrent state");
    });
}

TEST_CASE(TreeCommitAsyncFixture, rejected_preflight_writes_nothing) {
    each_gpu([&](ggml_backend_t backend) {
        Round round(backend, kMixedLayers, kCaches, false);
        REQUIRE(round.buffer != nullptr);
        round.fill(31);
        PinnedStage stage;
        REQUIRE(stage.reserve(backend, 1 << 16));
        const auto before = round.snapshot();

        // Too long an accepted prefix, then a slot used by two lanes, then a
        // feature row past the ring: each is refused before any launch.
        Indices bad = make_indices({2, -1, 0, 5}, {4, 2, 1, 3}, 5);
        bad.accepted[0] = kTreeWidth + 1;
        CHECK(!round.commit_async(backend, stage, bad));
        bad = make_indices({2, -1, 2, 5}, {4, 2, 1, 3}, 5);
        CHECK(!round.commit_async(backend, stage, bad));
        bad = make_indices({2, -1, 0, 5}, {4, 2, 1, 3}, 5);
        bad.feature_rows[0] = kFeatureCapacity;
        CHECK(!round.commit_async(backend, stage, bad));

        // The refused calls uploaded their indices but committed nothing:
        // every other tensor holds its first bytes.
        const auto after = round.snapshot();
        for (size_t i = 0; i < before.size(); ++i) {
            const ggml_tensor * t = round.all[i];
            if (t == round.commit_rows || t == round.accepted ||
                t == round.slots || t == round.feature_rows) {
                continue;
            }
            CHECK(before[i] == after[i]);
        }
    });
}
