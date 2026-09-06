#include "common/io_utils.h"
#include "placement/placement_config.h"
#include "placement/pflash_placement.h"
#include "placement/draft_residency.h"
#include "common/backend_precision.h"
#include "common/layer_split_backend.h"
#include "common/layer_split_kvflash.h"
#include "common/layer_split_utils.h"
#include "common/kvflash_pager.h"
#include "support/environment.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cmath>
#include <string>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

TEST_CASE(ServerUnitFixture, test_pflash_placement_same_backend_local) {
    DevicePlacement target;
    target.backend = compiled_placement_backend();
    target.gpu = 0;
    DevicePlacement drafter;
    drafter.backend = target.backend;
    drafter.gpu = 2;
    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";

    auto placement = resolve_pflash_drafter_placement(
        target, drafter, remote, /*pflash_enabled=*/true);
    TEST_ASSERT(placement.target_backend == target.backend);
    TEST_ASSERT(placement.drafter_backend == target.backend);
    TEST_ASSERT(placement.drafter_gpu == 2);
    TEST_ASSERT(!placement.remote_drafter);
    TEST_ASSERT(!placement.remote.enabled());
}

TEST_CASE(ServerUnitFixture, test_pflash_placement_mixed_backend_remote) {
    DevicePlacement target;
    target.backend = PlacementBackend::Cuda;
    target.gpu = 0;
    DevicePlacement drafter;
    drafter.backend = PlacementBackend::Hip;
    drafter.gpu = 1;
    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";
    remote.work_dir = "/tmp/pflash-ipc";

    auto placement = resolve_pflash_drafter_placement(
        target, drafter, remote, /*pflash_enabled=*/true);
    TEST_ASSERT(placement.target_backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.drafter_backend == PlacementBackend::Hip);
    TEST_ASSERT(placement.drafter_gpu == 1);
    TEST_ASSERT(placement.remote_drafter);
    TEST_ASSERT(placement.remote.enabled());
    TEST_ASSERT(placement.remote.work_dir == "/tmp/pflash-ipc");
}

TEST_CASE(ServerUnitFixture, test_pflash_placement_auto_draft_follows_target) {
    DevicePlacement target;
    target.backend = PlacementBackend::Hip;
    target.gpu = 0;
    DevicePlacement drafter;
    drafter.backend = PlacementBackend::Auto;
    drafter.gpu = 3;
    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";

    auto placement = resolve_pflash_drafter_placement(
        target, drafter, remote, /*pflash_enabled=*/true);
    TEST_ASSERT(placement.target_backend == PlacementBackend::Hip);
    TEST_ASSERT(placement.drafter_backend == PlacementBackend::Hip);
    TEST_ASSERT(placement.drafter_gpu == 3);
    TEST_ASSERT(!placement.remote_drafter);
}

TEST_CASE(ServerUnitFixture, test_pflash_placement_disabled_never_remote) {
    DevicePlacement target;
    target.backend = PlacementBackend::Cuda;
    DevicePlacement drafter;
    drafter.backend = PlacementBackend::Hip;
    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";

    auto placement = resolve_pflash_drafter_placement(
        target, drafter, remote, /*pflash_enabled=*/false);
    TEST_ASSERT(placement.target_backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.drafter_backend == PlacementBackend::Hip);
    TEST_ASSERT(!placement.remote_drafter);
    TEST_ASSERT(!placement.remote.enabled());
}

TEST_CASE(ServerUnitFixture, test_pflash_placement_usage_gate) {
    TEST_ASSERT(!pflash_drafter_placement_used(
        /*pflash_enabled=*/false, /*has_decode_draft=*/false));
    TEST_ASSERT(pflash_drafter_placement_used(
        /*pflash_enabled=*/false, /*has_decode_draft=*/true));
    TEST_ASSERT(pflash_drafter_placement_used(
        /*pflash_enabled=*/true, /*has_decode_draft=*/false));
    TEST_ASSERT(pflash_drafter_placement_used(
        /*pflash_enabled=*/true, /*has_decode_draft=*/true));
}

TEST_CASE(ServerUnitFixture, test_draft_residency_parse) {
    DraftResidencyPolicy policy = DraftResidencyPolicy::Auto;
    TEST_ASSERT(parse_draft_residency_policy("auto", policy));
    TEST_ASSERT(policy == DraftResidencyPolicy::Auto);
    TEST_ASSERT(parse_draft_residency_policy("persistent", policy));
    TEST_ASSERT(policy == DraftResidencyPolicy::Persistent);
    TEST_ASSERT(parse_draft_residency_policy("request-scoped", policy));
    TEST_ASSERT(policy == DraftResidencyPolicy::RequestScoped);
    TEST_ASSERT(parse_draft_residency_policy("request_scoped", policy));
    TEST_ASSERT(policy == DraftResidencyPolicy::RequestScoped);
    TEST_ASSERT(!parse_draft_residency_policy("request", policy));
}

TEST_CASE(ServerUnitFixture, test_draft_residency_pflash_auto) {
    auto action = resolve_draft_residency_action(
        DraftResidencyPolicy::Auto,
        DraftResidencyContext{
            DraftResidencyUse::PFlashCompress,
            /*low_vram_hint=*/false,
            /*has_decode_draft=*/false,
        });
    TEST_ASSERT(action == DraftResidencyAction::ReleaseAfterUse);

    action = resolve_draft_residency_action(
        DraftResidencyPolicy::Auto,
        DraftResidencyContext{
            DraftResidencyUse::PFlashCompress,
            /*low_vram_hint=*/true,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::ReleaseAfterUse);
}

TEST_CASE(ServerUnitFixture, test_draft_residency_dflash_auto_and_request_scoped) {
    auto action = resolve_draft_residency_action(
        DraftResidencyPolicy::Auto,
        DraftResidencyContext{
            DraftResidencyUse::DFlashDecode,
            /*low_vram_hint=*/false,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::KeepLoaded);

    action = resolve_draft_residency_action(
        DraftResidencyPolicy::Auto,
        DraftResidencyContext{
            DraftResidencyUse::DFlashDecode,
            /*low_vram_hint=*/true,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::ReleaseAfterUse);

    action = resolve_draft_residency_action(
        DraftResidencyPolicy::RequestScoped,
        DraftResidencyContext{
            DraftResidencyUse::DFlashDecode,
            /*low_vram_hint=*/false,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::ReleaseAfterUse);

    action = resolve_draft_residency_action(
        DraftResidencyPolicy::Persistent,
        DraftResidencyContext{
            DraftResidencyUse::DFlashDecode,
            /*low_vram_hint=*/true,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::KeepLoaded);
}

// ═══════════════════════════════════════════════════════════════════════
// Placement config tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_parse_target_device_list_same_backend) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1", placement));
    TEST_ASSERT(placement.backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.gpu == 0);
    TEST_ASSERT(placement.is_layer_split());
    TEST_ASSERT(!placement.is_mixed_layer_split());
    TEST_ASSERT(placement.layer_split_backends.size() == 2);
    TEST_ASSERT(placement.layer_split_backends[0] == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_backends[1] == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_gpus.size() == 2);
    TEST_ASSERT(placement.layer_split_gpus[0] == 0);
    TEST_ASSERT(placement.layer_split_gpus[1] == 1);
    TEST_ASSERT(placement.layer_split_weights.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_target_device_list_mixed_backend) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,hip:1", placement));
    TEST_ASSERT(placement.backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.gpu == 0);
    TEST_ASSERT(placement.is_layer_split());
    TEST_ASSERT(placement.is_mixed_layer_split());
    TEST_ASSERT(placement.layer_split_backends.size() == 2);
    TEST_ASSERT(placement.layer_split_backends[0] == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_backends[1] == PlacementBackend::Hip);
    TEST_ASSERT(placement.layer_split_backend(0) == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_backend(1) == PlacementBackend::Hip);
    TEST_ASSERT(placement.layer_split_gpus.size() == 2);
    TEST_ASSERT(placement.layer_split_gpus[0] == 0);
    TEST_ASSERT(placement.layer_split_gpus[1] == 1);
}

TEST_CASE(ServerUnitFixture, test_parse_target_device_list_mixed_backend_multi_remote) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,hip:0,hip:1", placement));
    TEST_ASSERT(placement.backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.gpu == 0);
    TEST_ASSERT(placement.is_layer_split());
    TEST_ASSERT(placement.is_mixed_layer_split());
    TEST_ASSERT(placement.layer_split_backends.size() == 3);
    TEST_ASSERT(placement.layer_split_backends[0] == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_backends[1] == PlacementBackend::Hip);
    TEST_ASSERT(placement.layer_split_backends[2] == PlacementBackend::Hip);
    TEST_ASSERT(placement.layer_split_gpus.size() == 3);
    TEST_ASSERT(placement.layer_split_gpus[0] == 0);
    TEST_ASSERT(placement.layer_split_gpus[1] == 0);
    TEST_ASSERT(placement.layer_split_gpus[2] == 1);
}

TEST_CASE(ServerUnitFixture, test_parse_target_device_list_single_gpu_is_not_layer_split) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("hip:2", placement));
    TEST_ASSERT(placement.backend == PlacementBackend::Hip);
    TEST_ASSERT(placement.gpu == 2);
    TEST_ASSERT(!placement.is_layer_split());
    TEST_ASSERT(!placement.is_mixed_layer_split());
    TEST_ASSERT(placement.layer_split_backends.empty());
    TEST_ASSERT(placement.layer_split_gpus.empty());
}

TEST_CASE(ServerUnitFixture, test_validate_layer_split_weights_shape) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1", placement));

    placement.layer_split_weights = {1.0};
    TEST_ASSERT(!validate_device_placement(placement, -1).empty());

    placement.layer_split_weights = {1.0, 0.0};
    TEST_ASSERT(!validate_device_placement(placement, -1).empty());

    placement.layer_split_weights = {1.0, 2.0};
    TEST_ASSERT(validate_device_placement(placement, -1).empty());
}

TEST_CASE(ServerUnitFixture, test_target_shard_plan_same_backend_split) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1,cuda:2", placement));

    MixedLayerSplitPlan plan;
    TEST_ASSERT(compute_target_shard_layer_split_plan(
        placement, PlacementBackend::Cuda, plan, "test-target-shard"));
    TEST_ASSERT(plan.remote_begin == 1);
    TEST_ASSERT(plan.remote_backend == PlacementBackend::Cuda);
}

TEST_CASE(ServerUnitFixture, test_target_shard_plan_mixed_backend_split) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1,hip:0,hip:1", placement));

    MixedLayerSplitPlan plan;
    TEST_ASSERT(compute_target_shard_layer_split_plan(
        placement, PlacementBackend::Cuda, plan, "test-target-shard"));
    TEST_ASSERT(plan.remote_begin == 2);
    TEST_ASSERT(plan.remote_backend == PlacementBackend::Hip);
}

TEST_CASE(ServerUnitFixture, test_target_shard_plan_rejects_bad_local_backend) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,hip:0", placement));

    MixedLayerSplitPlan plan;
    TEST_ASSERT(!compute_target_shard_layer_split_plan(
        placement, PlacementBackend::Hip, plan, "test-target-shard"));
}

static bool kvflash_test_sync_identity(KvFlashPager & pager, int committed) {
    return layer_split_kvflash_sync_identity(
        pager, committed, pager.pool_tokens(), "test-target-split");
}

TEST_CASE(ServerUnitFixture, test_kvflash_pager_identity_sync_contract) {
    KvFlashConfig cfg;
    cfg.pool_tokens = 512;

    KvFlashPager local;
    KvFlashPager remote;
    TEST_ASSERT(local.attach(cfg, {}, {}));
    TEST_ASSERT(remote.attach(cfg, {}, {}));

    TEST_ASSERT(kvflash_test_sync_identity(local, 256));
    TEST_ASSERT(kvflash_test_sync_identity(remote, 256));
    TEST_ASSERT(local.slot_of(255) == remote.slot_of(255));

    TEST_ASSERT(kvflash_test_sync_identity(local, cfg.pool_tokens));
    TEST_ASSERT(local.slot_of(cfg.pool_tokens - 1) == cfg.pool_tokens - 1);
    TEST_ASSERT(local.is_identity());

    const int relocated = local.slot_for(cfg.pool_tokens);
    TEST_ASSERT(relocated >= 0);
    TEST_ASSERT(relocated != cfg.pool_tokens);
    TEST_ASSERT(!local.is_identity());

    TEST_ASSERT(kvflash_test_sync_identity(local, 128));
    TEST_ASSERT(local.slot_of(127) == 127);
}

TEST_CASE(ServerUnitFixture, test_layer_split_kvflash_history_contract) {
    std::vector<int32_t> history;
    layer_split_kvflash_sync_history(history, {1, 2, 3}, 0);
    TEST_ASSERT((history == std::vector<int32_t>{1, 2, 3}));

    layer_split_kvflash_sync_history(history, {4, 5}, 3);
    TEST_ASSERT((history == std::vector<int32_t>{1, 2, 3, 4, 5}));

    layer_split_kvflash_sync_history(history, {9}, 2);
    TEST_ASSERT((history == std::vector<int32_t>{1, 2, 9}));

    layer_split_kvflash_sync_history(history, {7}, 5);
    TEST_ASSERT(history.size() == 6);
    TEST_ASSERT(history[0] == 1);
    TEST_ASSERT(history[1] == 2);
    TEST_ASSERT(history[2] == 9);
    TEST_ASSERT(history[3] == 0);
    TEST_ASSERT(history[4] == 0);
    TEST_ASSERT(history[5] == 7);

    std::vector<std::vector<int32_t>> snapshots(2);
    layer_split_kvflash_save_history_snapshot(history, 4, snapshots[1]);
    TEST_ASSERT((snapshots[1] == std::vector<int32_t>{1, 2, 9, 0}));

    history = {8, 8, 8};
    layer_split_kvflash_restore_history(history, snapshots, 1, 6);
    TEST_ASSERT((history == std::vector<int32_t>{1, 2, 9, 0, 0, 0}));

    layer_split_kvflash_restore_history(history, snapshots, 0, 3);
    TEST_ASSERT((history == std::vector<int32_t>{0, 0, 0}));
}

TEST_CASE(ServerUnitFixture, test_backend_precision_cuda_sm_policy) {
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(90) == GGML_TYPE_BF16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(80) == GGML_TYPE_BF16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(75) == GGML_TYPE_F16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(70) == GGML_TYPE_F16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(60) == GGML_TYPE_F16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(62) == GGML_TYPE_F32);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(61) == GGML_TYPE_F32);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(52) == GGML_TYPE_F32);
}

TEST_CASE(ServerUnitFixture, test_backend_precision_hip_arch_policy) {
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx90a") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx942") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx950") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx1100") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx1200") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx906") == GGML_TYPE_F16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx1030") == GGML_TYPE_F16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx803") == GGML_TYPE_F32);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("") == GGML_TYPE_F32);
}

TEST_CASE(ServerUnitFixture, test_backend_precision_activation_type_combine) {
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_BF16, GGML_TYPE_BF16) == GGML_TYPE_BF16);
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_BF16, GGML_TYPE_F16) == GGML_TYPE_F16);
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_F16, GGML_TYPE_BF16) == GGML_TYPE_F16);
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_F16, GGML_TYPE_F32) == GGML_TYPE_F32);
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_F32, GGML_TYPE_BF16) == GGML_TYPE_F32);
}

struct MockLayerSplitAdapter : LayerSplitAdapter {
    int max_ctx = 128;
    bool reset_called = false;
    int saved_slot = -1;
    int saved_pos = 0;
    int restored_slot = -1;
    int current_pos = 0;
    int current_last = -1;
    std::vector<int> prefill_bases;
    std::vector<int> prefill_sizes;
    int dflash_base = -1;
    int dflash_last = -1;
    std::vector<int32_t> emitted_tokens;
    bool dflash_enabled = false;
    bool dflash_called = false;
    bool sampling_enabled = false;
    bool kvflash_enabled = false;
    bool mixed_backend_enabled = false;
    int shutdown_calls = 0;
    ModelBackend::CompressRequest last_compress_req;
    int prefill_chunk = 0;
    std::function<void()> on_prefill;

    const char * name() const override { return "mock"; }
    bool init() override { return true; }
    int max_context() const override { return max_ctx; }
    void reset_request_state() override {
        reset_called = true;
        current_pos = 0;
        current_last = -1;
    }
    int prefill_chunk_tokens() const override { return prefill_chunk; }
    bool prefill(const std::vector<int32_t> & prompt,
                 int base_pos, int & last_tok) override {
        prefill_bases.push_back(base_pos);
        prefill_sizes.push_back((int)prompt.size());
        current_pos = base_pos + (int)prompt.size();
        current_last = prompt.empty() ? current_last : prompt.back();
        last_tok = current_last;
        if (on_prefill) on_prefill();
        return true;
    }
    bool decode_ar(int last_tok, int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io) override {
        (void)history_prefix;
        if (committed != current_pos) return false;
        for (int i = 0; i < n_gen; ++i) {
            int32_t tok = last_tok + i + 1;
            out_tokens.push_back(tok);
            emitted_tokens.push_back(tok);
            io.emit(tok);
        }
        io.emit(-1);
        return true;
    }
    bool can_dflash_decode() const override { return dflash_enabled; }
    bool supports_cpu_sampling() const override { return sampling_enabled; }
    bool supports_kvflash() const override { return kvflash_enabled; }
    bool supports_mixed_backend_layer_split() const override {
        return mixed_backend_enabled;
    }
    bool decode_dflash(const std::vector<int32_t> & prompt, int base_pos,
                       int last_tok, int n_gen, std::vector<int32_t> & out_tokens,
                       const DaemonIO & io, float & accept_rate_out) override {
        (void)prompt;
        accept_rate_out = 0.0f;
        dflash_called = true;
        dflash_base = base_pos;
        dflash_last = last_tok;
        for (int i = 0; i < n_gen; ++i) {
            int32_t tok = last_tok + i + 10;
            out_tokens.push_back(tok);
            emitted_tokens.push_back(tok);
            io.emit(tok);
        }
        io.emit(-1);
        return true;
    }
    void free_drafter() override {}
    bool snapshot_save(int slot) override {
        saved_slot = slot;
        saved_pos = current_pos;
        return true;
    }
    bool snapshot_used(int slot) const override {
        return slot == saved_slot && saved_pos > 0;
    }
    int snapshot_cur_pos(int slot) const override {
        return snapshot_used(slot) ? saved_pos : 0;
    }
    bool snapshot_restore(int slot) override {
        if (!snapshot_used(slot)) return false;
        restored_slot = slot;
        current_pos = saved_pos;
        current_last = saved_pos;
        return true;
    }
    int current_last_token() const override { return current_last; }
    const char * default_compress_drafter_path() const override {
        return "/tmp/default-layer-split-drafter.gguf";
    }
    ModelBackend::CompressResult
    compress(const ModelBackend::CompressRequest & req) override {
        last_compress_req = req;
        ModelBackend::CompressResult result;
        result.ok = true;
        result.compressed_ids = {77, 88};
        return result;
    }
    void shutdown() override { shutdown_calls++; }
};

TEST_CASE(ServerUnitFixture, test_layer_split_backend_inline_snapshot_and_restore_delta) {
    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

    GenerateRequest req;
    req.prompt = {10, 11, 12, 13};
    req.n_gen = 1;
    req.snap_slot = 2;
    req.snap_pos = 3;
    DaemonIO io;
    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(raw->reset_called);
    TEST_ASSERT(raw->saved_slot == 2);
    TEST_ASSERT(raw->saved_pos == 3);
    TEST_ASSERT(raw->prefill_bases.size() == 2);
    TEST_ASSERT(raw->prefill_bases[0] == 0);
    TEST_ASSERT(raw->prefill_sizes[0] == 3);
    TEST_ASSERT(raw->prefill_bases[1] == 3);
    TEST_ASSERT(raw->prefill_sizes[1] == 1);
    TEST_ASSERT(backend.snapshot_used(2));
    TEST_ASSERT(backend.snapshot_cur_pos(2) == 3);

    raw->reset_called = false;
    raw->prefill_bases.clear();
    raw->prefill_sizes.clear();
    raw->dflash_enabled = true;
    GenerateRequest restore_req;
    restore_req.prompt = {10, 11, 12, 99};
    restore_req.n_gen = 1;
    GenerateResult restored = backend.restore_and_generate(2, restore_req, io);

    TEST_ASSERT(restored.ok());
    TEST_ASSERT(raw->dflash_called);
    TEST_ASSERT(raw->restored_slot == 2);
    TEST_ASSERT(!raw->reset_called);
    TEST_ASSERT(raw->prefill_bases.size() == 1);
    TEST_ASSERT(raw->prefill_bases[0] == 3);
    TEST_ASSERT(raw->prefill_sizes[0] == 1);
    TEST_ASSERT(raw->dflash_base == 3);
    TEST_ASSERT(raw->dflash_last == 99);
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_sampling_capability_gate) {
    {
        auto * raw = new MockLayerSplitAdapter();
        LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

        GenerateRequest req;
        req.prompt = {10, 11};
        req.n_gen = 1;
        req.do_sample = true;
        req.sampler.temp = 0.8f;
        DaemonIO io;
        GenerateResult result = backend.generate(req, io);

        TEST_ASSERT(!result.ok());
        TEST_ASSERT(result.error->code == GenerateErrorCode::SamplingUnsupported);
        TEST_ASSERT(result.error_code() == "sampling_unsupported");
    }

    {
        auto * raw = new MockLayerSplitAdapter();
        raw->sampling_enabled = true;
        LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

        GenerateRequest req;
        req.prompt = {10, 11};
        req.n_gen = 1;
        req.do_sample = true;
        req.sampler.temp = 0.8f;
        DaemonIO io;
        GenerateResult result = backend.generate(req, io);

        TEST_ASSERT(result.ok());
        TEST_ASSERT(result.tokens.size() == 1);
        TEST_ASSERT(result.tokens[0] == 12);
    }
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_chunks_prefill_by_adapter_limit) {
    auto * raw = new MockLayerSplitAdapter();
    raw->prefill_chunk = 3;
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

    GenerateRequest req;
    req.prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    req.n_gen = 1;
    DaemonIO io;
    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(raw->prefill_bases.size() == 3);
    TEST_ASSERT(raw->prefill_sizes.size() == 3);
    TEST_ASSERT(raw->prefill_bases[0] == 0);
    TEST_ASSERT(raw->prefill_sizes[0] == 3);
    TEST_ASSERT(raw->prefill_bases[1] == 3);
    TEST_ASSERT(raw->prefill_sizes[1] == 3);
    TEST_ASSERT(raw->prefill_bases[2] == 6);
    TEST_ASSERT(raw->prefill_sizes[2] == 2);
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_cancels_between_prefill_chunks) {
    auto * raw = new MockLayerSplitAdapter();
    raw->prefill_chunk = 3;
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

    bool cancel = false;
    raw->on_prefill = [&cancel]() { cancel = true; };
    DaemonIO io;
    io.should_cancel = [&cancel]() { return cancel; };

    GenerateRequest req;
    req.prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    req.n_gen = 4;
    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(io.is_cancelled());
    TEST_ASSERT(raw->prefill_bases.size() == 1);
    TEST_ASSERT(raw->prefill_sizes.size() == 1);
    TEST_ASSERT(raw->prefill_sizes[0] == 3);
    TEST_ASSERT(raw->emitted_tokens.empty());
}

TEST_CASE(ServerUnitFixture, test_layer_split_compress_nopark_uses_default_drafter_path) {
    const std::string ids_path = "/tmp/dflash_test_layer_split_compress_ids.bin";
    unlink(ids_path.c_str());
    TEST_ASSERT(write_int32_file(ids_path, {1, 2, 3, 4}));

    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};
    DaemonIO io;

    const std::string cmd = "compress " + ids_path + " 250 nopark";
    TEST_ASSERT(backend.handle_compress(cmd, io));
    TEST_ASSERT(raw->last_compress_req.skip_park);
    TEST_ASSERT(std::abs(raw->last_compress_req.keep_ratio - 0.25f) < 1e-5f);
    TEST_ASSERT(raw->last_compress_req.input_ids.size() == 4);
    TEST_ASSERT(raw->last_compress_req.drafter_path ==
                "/tmp/default-layer-split-drafter.gguf");

    unlink(ids_path.c_str());
}

TEST_CASE(ServerUnitFixture, test_layer_split_compress_rejects_bad_keep_ratio) {
    const std::string ids_path = "/tmp/dflash_test_layer_split_compress_bad.bin";
    unlink(ids_path.c_str());
    TEST_ASSERT(write_int32_file(ids_path, {1, 2, 3, 4}));

    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};
    DaemonIO io;

    const std::string cmd = "compress " + ids_path + " 1250 nopark";
    TEST_ASSERT(!backend.handle_compress(cmd, io));
    TEST_ASSERT(raw->last_compress_req.input_ids.empty());

    unlink(ids_path.c_str());
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_shutdown_is_idempotent) {
    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};
    backend.shutdown();
    backend.shutdown();
    TEST_ASSERT(raw->shutdown_calls == 1);
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_capability_proxy) {
    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

    TEST_ASSERT(!backend.supports_kvflash());
    TEST_ASSERT(!backend.supports_mixed_backend_layer_split());

    raw->kvflash_enabled = true;
    raw->mixed_backend_enabled = true;
    TEST_ASSERT(backend.supports_kvflash());
    TEST_ASSERT(backend.supports_mixed_backend_layer_split());
}
