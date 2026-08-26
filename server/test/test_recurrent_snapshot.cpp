#include "CppUnitTestFramework.hpp"
#include "internal.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

using namespace CppUnitTestFramework;

using dflash::common::PrefixSnapshot;
using dflash::common::TargetCache;
using dflash::common::estimate_paged_target_cache_snapshot_bytes;
using dflash::common::free_prefix_snapshot;
using dflash::common::replace_paged_target_cache;
using dflash::common::restore_paged_target_cache;
using dflash::common::restore_ssm_state;
using dflash::common::snapshot_paged_target_cache;
using dflash::common::snapshot_ssm_state;

namespace {
struct RecurrentSnapshotFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

static void set_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    ggml_backend_tensor_set(tensor, values.data(), 0,
                            values.size() * sizeof(float));
}

static std::vector<float> get_tensor(const ggml_tensor * tensor) {
    std::vector<float> values((size_t)ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0,
                            values.size() * sizeof(float));
    return values;
}

TEST_CASE(RecurrentSnapshotFixture, snapshot_and_restore_recurrent_state) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) SKIP("CPU backend is unavailable");

    ggml_init_params params{};
    params.mem_size = 8 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    if (!ctx) {
        ggml_backend_free(backend);
        SKIP("could not initialize ggml context");
    }

    ggml_tensor * ssm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * ssm_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
    ggml_tensor * conv_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        SKIP("could not allocate CPU backend tensors");
    }

    TargetCache cache;
    cache.ssm_state = {ssm, nullptr};
    cache.ssm_state_snap = {ssm_snap, nullptr};
    cache.conv_state = {conv, nullptr};
    cache.conv_state_snap = {conv_snap, nullptr};

    const std::vector<float> ssm_original = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    const std::vector<float> conv_original = {
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    };
    const std::vector<float> ssm_mutated(ssm_original.size(), -1.0f);
    const std::vector<float> conv_mutated(conv_original.size(), -2.0f);

    CHECK(ggml_nelements(ssm) == (int64_t) ssm_original.size());
    CHECK(ggml_nelements(conv) == (int64_t) conv_original.size());
    set_tensor(ssm, ssm_original);
    set_tensor(conv, conv_original);
    CHECK(snapshot_ssm_state(cache, backend));
    set_tensor(ssm, ssm_mutated);
    set_tensor(conv, conv_mutated);
    CHECK(restore_ssm_state(cache, backend));
    CHECK(get_tensor(ssm) == ssm_original);
    CHECK(get_tensor(conv) == conv_original);

    // A partial shard may omit a complete recurrent-state quartet, but an
    // asymmetric quartet must fail validation before any copy is queued.
    set_tensor(ssm, ssm_mutated);
    cache.conv_state_snap[0] = nullptr;
    CHECK(!snapshot_ssm_state(cache, backend));
    CHECK(get_tensor(ssm_snap) == ssm_original);
    cache.conv_state_snap[0] = conv_snap;

    CHECK(!snapshot_ssm_state(cache, nullptr));
    cache.ssm_state_snap.pop_back();
    CHECK(!restore_ssm_state(cache, backend));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

TEST_CASE(RecurrentSnapshotFixture, copied_paged_prefix_uses_fresh_pages) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) SKIP("CPU backend is unavailable");

    ggml_init_params params{};
    params.mem_size = 12 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    if (!ctx) {
        ggml_backend_free(backend);
        SKIP("could not initialize ggml context");
    }

    ggml_tensor * key =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 64, 2);
    ggml_tensor * value =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 64, 2);
    ggml_tensor * ssm =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 2, 2);
    ggml_tensor * conv =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2);
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        SKIP("could not allocate CPU backend tensors");
    }

    std::vector<float> keys((size_t)ggml_nelements(key));
    std::vector<float> values((size_t)ggml_nelements(value));
    std::vector<float> states((size_t)ggml_nelements(ssm));
    std::vector<float> convs((size_t)ggml_nelements(conv));
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = (float)i + 1.0f;
        values[i] = (float)i + 1001.0f;
    }
    for (size_t i = 0; i < states.size(); ++i) states[i] = (float)i + 2001.0f;
    for (size_t i = 0; i < convs.size(); ++i) convs[i] = (float)i + 3001.0f;
    set_tensor(key, keys);
    set_tensor(value, values);
    set_tensor(ssm, states);
    set_tensor(conv, convs);

    TargetCache cache;
    cache.backend = backend;
    cache.max_ctx = 64;
    cache.n_seq_slots = 2;
    cache.kv_k_type = GGML_TYPE_F32;
    cache.attn_k = {key};
    cache.attn_v = {value};
    cache.ssm_state = {ssm};
    cache.conv_state = {conv};

    const size_t estimated_bytes =
        estimate_paged_target_cache_snapshot_bytes(
            cache, /*token_count=*/20);
    CHECK(estimated_bytes > 0);

    PrefixSnapshot snap;
    const std::vector<uint32_t> source_blocks = {2, 0};
    CHECK(snapshot_paged_target_cache(
        cache, /*seq_slot=*/1, source_blocks,
        /*block_size=*/16, /*token_count=*/20, snap));
    CHECK(snap.layout == PrefixSnapshot::Layout::paged &&
          snap.cur_pos == 20);
    CHECK(ggml_backend_buffer_get_size(snap.buf) == estimated_bytes);
    // Paged copies use get/set with snapshot tensor data as host staging.
    // Keep that storage on a true CPU buffer, including on unified-memory
    // compute backends.
    CHECK(ggml_backend_buffer_get_type(snap.buf) ==
          ggml_backend_cpu_buffer_type());

    // An incomplete per-layer pair is invalid topology and must not replace
    // the committed payload.
    cache.attn_v[0] = nullptr;
    CHECK(!replace_paged_target_cache(
        cache, /*seq_slot=*/1, source_blocks,
        /*block_size=*/16, /*token_count=*/20, snap));
    cache.attn_v[0] = value;
    CHECK(snap.layout == PrefixSnapshot::Layout::paged &&
          snap.cur_pos == 20);

    // A failed shape-changing replacement must leave the incumbent payload
    // intact. Block 4 begins beyond this cache's 64 physical rows, so the
    // staged copy fails after allocating a differently-sized candidate.
    const std::vector<uint32_t> invalid_source_blocks = {4, 0};
    CHECK(!replace_paged_target_cache(
        cache, /*seq_slot=*/1, invalid_source_blocks,
        /*block_size=*/16, /*token_count=*/17, snap));
    CHECK(snap.layout == PrefixSnapshot::Layout::paged &&
          snap.cur_pos == 20);

    set_tensor(key, std::vector<float>(keys.size(), 0.0f));
    set_tensor(value, std::vector<float>(values.size(), 0.0f));
    set_tensor(ssm, std::vector<float>(states.size(), 0.0f));
    set_tensor(conv, std::vector<float>(convs.size(), 0.0f));
    const std::vector<uint32_t> destination_blocks = {1, 3};
    CHECK(restore_paged_target_cache(
        snap, cache, /*seq_slot=*/0, destination_blocks,
        /*block_size=*/16));

    const auto restored_keys = get_tensor(key);
    const auto restored_values = get_tensor(value);
    for (int head = 0; head < 2; ++head) {
        for (int logical = 0; logical < 20; ++logical) {
            const int source_row = logical < 16 ? 32 + logical : logical - 16;
            const int destination_row = logical < 16 ? 16 + logical : 48 + logical - 16;
            for (int element = 0; element < 2; ++element) {
                const size_t src = (size_t)head * 128 +
                    (size_t)source_row * 2 + element;
                const size_t dst = (size_t)head * 128 +
                    (size_t)destination_row * 2 + element;
                CHECK(restored_keys[dst] == keys[src]);
                CHECK(restored_values[dst] == values[src]);
            }
        }
    }
    const auto restored_states = get_tensor(ssm);
    const auto restored_convs = get_tensor(conv);
    CHECK(std::equal(states.begin() + states.size() / 2, states.end(),
                     restored_states.begin()));
    CHECK(std::equal(convs.begin() + convs.size() / 2, convs.end(),
                     restored_convs.begin()));

    free_prefix_snapshot(snap);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
