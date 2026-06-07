// Qwen35 layer-split adapter.

#include "qwen35_layer_split_adapter.h"

#include "common/backend_precision.h"
#include "common/dflash_spec_decode.h"
#include "common/gguf_inspect.h"
#include "common/layer_split_utils.h"
#include "common/sampler.h"
#include "common/layer_split_runtime.h"
#include "common/snapshot_backend.h"
#include "qwen35/layer_split_forward.h"
#include "qwen35/qwen35_layer_split_dflash_target.h"
#include "qwen3/qwen3_drafter.h"

#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace dflash::common {

Qwen35LayerSplitAdapter::Qwen35LayerSplitAdapter(
        const Qwen35LayerSplitAdapterConfig & cfg)
    : cfg_(cfg) {}

Qwen35LayerSplitAdapter::~Qwen35LayerSplitAdapter() { shutdown(); }

bool Qwen35LayerSplitAdapter::init() {
    if (cfg_.device.is_mixed_layer_split()) {
        return init_mixed_target_split();
    }

    const LayerSplitRuntimeInit runtime_cfg{
        cfg_.target_path,
        &cfg_.device,
        "target-split",
    };
    if (!init_layer_split_runtime(runtime_cfg, shards_, snapshot_backends_)) {
        return false;
    }

    std::vector<ggml_backend_t> shard_backends;
    shard_backends.reserve(shards_.size());
    for (const auto & shard : shards_) shard_backends.push_back(shard.backend);
    const BackendActivationPolicy activation_policy =
        select_common_activation_precision_policy(
            shard_backends, /*force_f32=*/cfg_.run_dflash,
            "LUCEBOX_LAYER_SPLIT_ACT_TYPE");
    activation_type_ = activation_policy.activation_type;
    std::fprintf(stderr, "[target-split] activation=%s (%s",
                 backend_precision_type_name(activation_type_),
                 activation_policy.reason.c_str());
    if (!activation_policy.runtime_arch.empty()) {
        std::fprintf(stderr, ", arch=%s", activation_policy.runtime_arch.c_str());
    } else if (activation_policy.cuda_sm > 0) {
        std::fprintf(stderr, ", sm=%d", activation_policy.cuda_sm);
    }
    std::fprintf(stderr, ")\n");

    for (auto & shard : shards_) {
        const TargetLoadPlan plan =
            make_layer_split_load_plan<TargetLoadPlan>(shard, &shard == &shards_.back());
        if (!load_target_gguf_partial(cfg_.target_path, shard.backend, plan,
                                      shard.weights) ||
            !create_target_cache_partial(shard.weights, cfg_.device.max_ctx,
                                         cfg_.max_verify_tokens, shard.backend,
                                         shard.cache,
                                         /*prefill_only=*/!cfg_.run_dflash,
                                         shard.layer_begin, shard.layer_end,
                                         /*allocate_target_feat=*/false)) {
            std::fprintf(stderr, "[target-split] load/cache gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            return false;
        }
        std::fprintf(stderr, "[target-split] gpu=%d layers=[%d,%d)\n",
                     shard.gpu, shard.layer_begin, shard.layer_end);
    }

    if (cfg_.draft_path && cfg_.run_dflash && !load_draft()) {
        return false;
    }
    prefix_snapshots_.resize(PREFIX_SLOTS);
    for (auto & slot : prefix_snapshots_) {
        slot.resize(shards_.size());
    }
    snapshot_prefill_logits_.resize(PREFIX_SLOTS);
    snapshot_prefill_logit_tensors_.resize(PREFIX_SLOTS);
    disk_snapshot_contexts_.assign(PREFIX_SLOTS, nullptr);
    disk_snapshot_buffers_.assign(PREFIX_SLOTS, nullptr);
    disk_snapshot_backends_.assign(PREFIX_SLOTS, nullptr);
    draft_feature_snapshots_.resize(PREFIX_SLOTS);

    return true;
}

bool Qwen35LayerSplitAdapter::init_mixed_target_split() {
    if (!cfg_.remote_target_shard.enabled() ||
        cfg_.device.layer_split_gpus.size() < 2) {
        std::fprintf(stderr,
            "[target-split] mixed target split requires at least two shards and "
            "remote target shard IPC\n");
        return false;
    }
    if (cfg_.device.layer_split_backend(0) != compiled_placement_backend()) {
        std::fprintf(stderr,
            "[target-split] first mixed shard must match compiled backend\n");
        return false;
    }
    size_t remote_begin = 0;
    while (remote_begin < cfg_.device.layer_split_gpus.size() &&
           cfg_.device.layer_split_backend(remote_begin) == compiled_placement_backend()) {
        ++remote_begin;
    }
    if (remote_begin == 0 || remote_begin >= cfg_.device.layer_split_gpus.size()) {
        std::fprintf(stderr,
            "[target-split] mixed target split requires one local backend group "
            "followed by one remote backend group\n");
        return false;
    }
    const PlacementBackend remote_backend = cfg_.device.layer_split_backend(remote_begin);
    for (size_t i = remote_begin; i < cfg_.device.layer_split_gpus.size(); ++i) {
        if (cfg_.device.layer_split_backend(i) != remote_backend) {
            std::fprintf(stderr,
                "[target-split] mixed target split supports one backend boundary only\n");
            return false;
        }
    }

    const auto info = inspect_gguf_model_info(cfg_.target_path);
    const int n_layer = info.n_layer;
    if (n_layer <= 0) {
        std::fprintf(stderr, "[target-split] failed to inspect target layer count\n");
        return false;
    }
    const auto ranges = compute_layer_ranges(
        n_layer, (int)cfg_.device.layer_split_gpus.size(),
        cfg_.device.layer_split_weights);
    if (ranges.size() != cfg_.device.layer_split_gpus.size()) {
        std::fprintf(stderr,
            "[target-split] bad mixed layer split for %zu GPUs and %d layers\n",
            cfg_.device.layer_split_gpus.size(), n_layer);
        return false;
    }

    shards_.resize(remote_begin);
    for (size_t i = 0; i < remote_begin; ++i) {
        Qwen35LayerSplitShard & local = shards_[i];
        local.placement_backend = cfg_.device.layer_split_backend(i);
        local.gpu = cfg_.device.layer_split_gpus[i];
        local.layer_begin = ranges[i].begin;
        local.layer_end = ranges[i].end;
        local.backend = ggml_backend_cuda_init(local.gpu);
        if (!local.backend) {
            std::fprintf(stderr, "[target-split] local backend init failed gpu=%d\n",
                         local.gpu);
            return false;
        }
    }

    for (auto & local : shards_) {
        const TargetLoadPlan local_plan =
            make_layer_split_load_plan<TargetLoadPlan>(local, /*is_last_shard=*/false);
        if (!load_target_gguf_partial(cfg_.target_path, local.backend, local_plan,
                                      local.weights) ||
            !create_target_cache_partial(local.weights, cfg_.device.max_ctx,
                                         cfg_.max_verify_tokens, local.backend,
                                         local.cache,
                                         /*prefill_only=*/!cfg_.run_dflash,
                                         local.layer_begin, local.layer_end,
                                         /*allocate_target_feat=*/false)) {
            std::fprintf(stderr, "[target-split] mixed local load/cache gpu=%d: %s\n",
                         local.gpu, dflash27b_last_error());
            return false;
        }
    }

    std::vector<int> remote_gpus;
    std::vector<int> remote_layer_begins;
    std::vector<int> remote_layer_ends;
    for (size_t i = remote_begin; i < cfg_.device.layer_split_gpus.size(); ++i) {
        remote_gpus.push_back(cfg_.device.layer_split_gpus[i]);
        remote_layer_begins.push_back(ranges[i].begin);
        remote_layer_ends.push_back(ranges[i].end);
    }
    if (!remote_target_shard_.start(
            cfg_.remote_target_shard.ipc_bin, cfg_.target_path, remote_gpus,
            remote_layer_begins, remote_layer_ends, cfg_.device.max_ctx,
            cfg_.max_verify_tokens, cfg_.kq_stride_pad, cfg_.fa_window,
            shards_.front().weights.n_embd, shards_.front().weights.n_vocab,
            std::max(1, cfg_.device.max_ctx),
            cfg_.remote_target_shard.work_dir,
            cfg_.run_dflash)) {
        std::fprintf(stderr,
            "[target-split] remote target shard start failed layers=[%d,%d)\n",
            remote_layer_begins.front(), remote_layer_ends.back());
        return false;
    }

    if (cfg_.draft_path && cfg_.run_dflash && !load_draft()) {
        return false;
    }

    for (const auto & local : shards_) {
        std::fprintf(stderr, "[target-split] local %s:%d layers=[%d,%d)\n",
                     placement_backend_name(local.placement_backend),
                     local.gpu, local.layer_begin, local.layer_end);
    }
    for (size_t i = 0; i < remote_gpus.size(); ++i) {
        std::fprintf(stderr, "[target-split] remote %s:%d layers=[%d,%d)\n",
                     placement_backend_name(remote_backend),
                     remote_gpus[i], remote_layer_begins[i], remote_layer_ends[i]);
    }

    prefix_snapshots_.resize(PREFIX_SLOTS);
    for (auto & slot : prefix_snapshots_) {
        slot.resize(shards_.size());
    }
    snapshot_backends_.assign(shards_.size(), nullptr);
    for (size_t i = 0; i < shards_.size(); ++i) {
        snapshot_backends_[i] = create_snapshot_backend(shards_[i].backend);
        if (!snapshot_backends_[i]) {
            std::fprintf(stderr,
                "[target-split] mixed snapshot backend init failed gpu=%d\n",
                shards_[i].gpu);
            return false;
        }
    }
    snapshot_prefill_logits_.resize(PREFIX_SLOTS);
    snapshot_prefill_logit_tensors_.resize(PREFIX_SLOTS);
    disk_snapshot_contexts_.assign(PREFIX_SLOTS, nullptr);
    disk_snapshot_buffers_.assign(PREFIX_SLOTS, nullptr);
    disk_snapshot_backends_.assign(PREFIX_SLOTS, nullptr);
    draft_feature_snapshots_.resize(PREFIX_SLOTS);
    return true;
}

bool Qwen35LayerSplitAdapter::load_draft() {
    if (cfg_.remote_draft.enabled()) {
        const int cap = cfg_.remote_draft.ring_cap > 0
            ? std::min(cfg_.remote_draft.ring_cap, cfg_.device.max_ctx)
            : std::min(cfg_.device.max_ctx, cfg_.draft_ctx_max);
        if (!remote_draft_.start(cfg_.remote_draft.ipc_bin, cfg_.draft_path,
                                 cfg_.draft_gpu, cap,
                                 cfg_.remote_draft.work_dir)) {
            std::fprintf(stderr,
                "[target-split] remote draft start failed gpu=%d\n",
                cfg_.draft_gpu);
            return false;
        }
        draft_weights_.n_embd = DFLASH27B_TARGET_HIDDEN;
        draft_weights_.block_size = DFLASH27B_DRAFT_BLOCK_SIZE;
        draft_weights_.n_target_layers = DFLASH27B_DRAFT_N_TARGET_LAYERS;
        if (cfg_.draft_swa_window > 0) {
            draft_weights_.swa_window = cfg_.draft_swa_window;
        }
        std::fprintf(stderr,
            "[target-split] remote draft ready gpu=%d cap=%d\n",
            cfg_.draft_gpu, cap);
        return true;
    }

    for (auto & shard : shards_) {
        if (shard.gpu == cfg_.draft_gpu) {
            draft_backend_ = shard.backend;
            break;
        }
    }
    if (!draft_backend_) {
        draft_backend_ = ggml_backend_cuda_init(cfg_.draft_gpu);
        if (!draft_backend_) {
            std::fprintf(stderr,
                "[target-split] draft backend init failed gpu=%d\n",
                cfg_.draft_gpu);
            return false;
        }
        draft_backend_owned_ = true;
    }

    std::string draft_path(cfg_.draft_path ? cfg_.draft_path : "");
    const bool draft_ok = draft_path.size() >= 5 &&
            draft_path.substr(draft_path.size() - 5) == ".gguf"
        ? load_draft_gguf(cfg_.draft_path, draft_backend_, draft_weights_,
                          &shards_.front().weights)
        : load_draft_safetensors(cfg_.draft_path, draft_backend_,
                                 draft_weights_, &shards_.front().weights);
    if (!draft_ok) {
        std::fprintf(stderr, "[target-split] draft load gpu=%d: %s\n",
                     cfg_.draft_gpu, dflash27b_last_error());
        return false;
    }
    if (cfg_.draft_swa_window > 0) {
        draft_weights_.swa_window = cfg_.draft_swa_window;
    }

    const int cap = std::min(cfg_.device.max_ctx, cfg_.draft_ctx_max);
    if (!draft_feature_mirror_init(feature_ring_, draft_backend_,
                                   cfg_.draft_gpu, cfg_.draft_gpu, cap,
                                   draft_weights_.n_target_layers,
                                   draft_weights_.n_embd)) {
        std::fprintf(stderr,
            "[target-split] draft feature ring init failed gpu=%d\n",
            cfg_.draft_gpu);
        return false;
    }
    return true;
}

void Qwen35LayerSplitAdapter::begin_request(const GenerateRequest & req) {
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }
}

void Qwen35LayerSplitAdapter::reset_request_state() {
    for (auto & shard : shards_) reset_target_cache(shard.cache);
    if (use_mixed_target_split() &&
        !remote_target_shard_.reset_request_state()) {
        std::fprintf(stderr,
            "[target-split] remote shard reset_request_state failed\n");
    }
    prefill_last_logits_.clear();
}

int Qwen35LayerSplitAdapter::prefill_chunk_tokens() const {
    return cfg_.chunk > 0 ? cfg_.chunk : 0;
}

bool Qwen35LayerSplitAdapter::prefill(const std::vector<int32_t> & prompt,
                                      int base_pos, int & last_tok) {
    if (prompt.empty()) return false;
    if (base_pos < 0 || base_pos + (int)prompt.size() > cfg_.device.max_ctx) {
        std::fprintf(stderr,
            "[target-split] prompt range [%d,%zu) exceeds max_ctx (%d)\n",
            base_pos, (size_t)base_pos + prompt.size(), cfg_.device.max_ctx);
        return false;
    }
    int ubatch = prompt.size() > 2048 ? 384 : 16;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        ubatch = std::max(1, std::atoi(s));
    }
    if (use_mixed_target_split()) {
        return run_qwen35_mixed_layer_split_forward(
            shards_, remote_target_shard_, shards_.front().weights,
            prompt, base_pos, ubatch, last_tok,
            cfg_.kq_stride_pad, /*fa_window=*/0,
            /*argmax_out=*/nullptr,
            &prefill_last_logits_,
            (cfg_.run_dflash && !remote_draft_.active()) ? &feature_ring_ : nullptr,
            remote_draft_.active() ? &remote_draft_ : nullptr);
    }
    return run_qwen35_layer_split_forward(
        shards_, shards_.front().weights, prompt, base_pos, ubatch, last_tok,
        cfg_.kq_stride_pad, /*fa_window=*/0,
        (cfg_.run_dflash && !remote_draft_.active()) ? &feature_ring_ : nullptr,
        /*argmax_out=*/nullptr,
        &prefill_last_logits_,
        cfg_.run_dflash ? &remote_draft_ : nullptr,
        activation_type_);
}

bool Qwen35LayerSplitAdapter::snapshot_slot_valid(int slot) const {
    return slot >= 0 && slot < PREFIX_SLOTS &&
           prefix_snapshots_.size() == (size_t)PREFIX_SLOTS &&
           !shards_.empty();
}

bool Qwen35LayerSplitAdapter::snapshot_save(int slot) {
    if (!snapshot_slot_valid(slot)) return false;
    if (snapshot_backends_.size() != shards_.size()) return false;
    snapshot_free(slot);
    auto & snaps = prefix_snapshots_[(size_t)slot];
    if (snaps.size() != shards_.size()) snaps.resize(shards_.size());
    for (size_t i = 0; i < shards_.size(); ++i) {
        if (!snapshot_target_cache(shards_[i].weights, shards_[i].cache,
                                   snapshot_backends_[i], snaps[i])) {
            for (size_t j = 0; j <= i && j < snaps.size(); ++j) {
                free_prefix_snapshot(snaps[j]);
            }
            return false;
        }
    }
    if (use_mixed_target_split() && !remote_target_shard_.snapshot_save(slot)) {
        snapshot_free(slot);
        return false;
    }
    if (snapshot_prefill_logits_.size() != (size_t)PREFIX_SLOTS) return false;
    snapshot_prefill_logits_[(size_t)slot] = prefill_last_logits_;
    if (!snapshot_draft_features(slot)) {
        snapshot_free(slot);
        return false;
    }
    if (!use_mixed_target_split() && !rebuild_disk_snapshot(slot)) {
        snapshot_free(slot);
        return false;
    }
    return true;
}

void Qwen35LayerSplitAdapter::snapshot_free(int slot) {
    if (!snapshot_slot_valid(slot)) return;
    ggml_context * disk_ctx = nullptr;
    ggml_backend_buffer_t disk_buf = nullptr;
    ggml_backend_t disk_backend = nullptr;
    if (disk_snapshot_contexts_.size() == (size_t)PREFIX_SLOTS &&
        slot < (int)disk_snapshot_contexts_.size()) {
        disk_ctx = disk_snapshot_contexts_[(size_t)slot];
        disk_buf = disk_snapshot_buffers_[(size_t)slot];
        disk_snapshot_contexts_[(size_t)slot] = nullptr;
        disk_snapshot_buffers_[(size_t)slot] = nullptr;
        if (disk_snapshot_backends_.size() == (size_t)PREFIX_SLOTS &&
            disk_snapshot_backends_[(size_t)slot]) {
            disk_backend = disk_snapshot_backends_[(size_t)slot];
            disk_snapshot_backends_[(size_t)slot] = nullptr;
        }
    }
    for (auto & snap : prefix_snapshots_[(size_t)slot]) {
        if (disk_ctx && snap.ctx == disk_ctx) {
            snap = PrefixSnapshot{};
        } else {
            free_prefix_snapshot(snap);
        }
    }
    if (disk_buf) ggml_backend_buffer_free(disk_buf);
    if (disk_ctx) ggml_free(disk_ctx);
    if (disk_backend) ggml_backend_free(disk_backend);
    if (snapshot_prefill_logits_.size() == (size_t)PREFIX_SLOTS) {
        snapshot_prefill_logits_[(size_t)slot].clear();
    }
    if (snapshot_prefill_logit_tensors_.size() == (size_t)PREFIX_SLOTS) {
        snapshot_prefill_logit_tensors_[(size_t)slot].clear();
    }
    if (use_mixed_target_split()) {
        remote_target_shard_.snapshot_free(slot);
    }
    free_draft_feature_snapshot(slot);
}

bool Qwen35LayerSplitAdapter::snapshot_used(int slot) const {
    if (!snapshot_slot_valid(slot)) return false;
    const auto & snaps = prefix_snapshots_[(size_t)slot];
    if (snaps.size() != shards_.size()) return false;
    for (const auto & snap : snaps) {
        if (!snap.ctx) return false;
    }
    if (snapshot_prefill_logits_.size() != (size_t)PREFIX_SLOTS ||
        snapshot_prefill_logits_[(size_t)slot].empty()) {
        return false;
    }
    if (cfg_.run_dflash && cfg_.draft_path) {
        if (draft_feature_snapshots_.size() != (size_t)PREFIX_SLOTS) return false;
        const auto & draft_snap = draft_feature_snapshots_[(size_t)slot];
        if (draft_snap.cur_pos <= 0 || draft_snap.n_tokens <= 0 ||
            draft_snap.data.empty()) return false;
    }
    return true;
}

int Qwen35LayerSplitAdapter::snapshot_cur_pos(int slot) const {
    if (!snapshot_used(slot)) return 0;
    return prefix_snapshots_[(size_t)slot].front().cur_pos;
}

bool Qwen35LayerSplitAdapter::snapshot_restore(int slot) {
    if (!snapshot_used(slot)) return false;
    auto & snaps = prefix_snapshots_[(size_t)slot];
    const int cur_pos = snaps.front().cur_pos;
    for (size_t i = 0; i < shards_.size(); ++i) {
        if (snaps[i].cur_pos != cur_pos ||
            !restore_target_cache(snaps[i], shards_[i].cache)) {
            return false;
        }
    }
    if (use_mixed_target_split() &&
        !remote_target_shard_.snapshot_restore(slot)) {
        return false;
    }
    if (snapshot_prefill_logits_.size() != (size_t)PREFIX_SLOTS) return false;
    prefill_last_logits_ = snapshot_prefill_logits_[(size_t)slot];
    if (!restore_draft_features(slot)) return false;
    return true;
}

bool Qwen35LayerSplitAdapter::rebuild_disk_snapshot(int slot) {
    if (!snapshot_slot_valid(slot) || use_mixed_target_split()) return false;
    if (disk_snapshot_contexts_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_buffers_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_backends_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }
    if (disk_snapshot_buffers_[(size_t)slot]) {
        ggml_backend_buffer_free(disk_snapshot_buffers_[(size_t)slot]);
        disk_snapshot_buffers_[(size_t)slot] = nullptr;
    }
    if (disk_snapshot_contexts_[(size_t)slot]) {
        ggml_free(disk_snapshot_contexts_[(size_t)slot]);
        disk_snapshot_contexts_[(size_t)slot] = nullptr;
    }
    if (disk_snapshot_backends_[(size_t)slot]) {
        ggml_backend_free(disk_snapshot_backends_[(size_t)slot]);
        disk_snapshot_backends_[(size_t)slot] = nullptr;
    }

    const auto & snaps = prefix_snapshots_[(size_t)slot];
    const bool has_dflash_features = cfg_.run_dflash && cfg_.draft_path;
    const DraftFeatureSnapshot * draft_snap = nullptr;
    if (has_dflash_features) {
        if (draft_feature_snapshots_.size() != (size_t)PREFIX_SLOTS) return false;
        draft_snap = &draft_feature_snapshots_[(size_t)slot];
        if (draft_snap->cur_pos <= 0 || draft_snap->start_pos < 0 ||
            draft_snap->n_tokens <= 0 || draft_snap->cap <= 0 ||
            draft_snap->n_target_layers <= 0 || draft_snap->hidden_size <= 0 ||
            draft_snap->data.empty()) {
            return false;
        }
    }

    size_t n_tensors = 1;  // prefill logits
    if (has_dflash_features) n_tensors += 2;  // metadata + feature rows
    for (const auto & snap : snaps) {
        if (!snap.ctx) return false;
        for (ggml_tensor * t = ggml_get_first_tensor(snap.ctx); t;
             t = ggml_get_next_tensor(snap.ctx, t)) {
            n_tensors++;
        }
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (n_tensors + 8) + 4096;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    struct CopyPair {
        ggml_tensor * src = nullptr;
        ggml_tensor * dst = nullptr;
    };
    std::vector<CopyPair> copies;
    copies.reserve(n_tensors);

    for (size_t shard_idx = 0; shard_idx < snaps.size(); ++shard_idx) {
        const auto & snap = snaps[shard_idx];
        for (ggml_tensor * src = ggml_get_first_tensor(snap.ctx); src;
             src = ggml_get_next_tensor(snap.ctx, src)) {
            ggml_tensor * dst = ggml_dup_tensor(ctx, src);
            if (!dst) {
                ggml_free(ctx);
                return false;
            }
            const std::string name =
                "ls" + std::to_string(shard_idx) + "_" + src->name;
            ggml_set_name(dst, name.c_str());
            copies.push_back({src, dst});
        }
    }

    const auto & logits = snapshot_prefill_logits_[(size_t)slot];
    if (logits.empty()) {
        ggml_free(ctx);
        return false;
    }
    ggml_tensor * logits_t =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)logits.size());
    if (!logits_t) {
        ggml_free(ctx);
        return false;
    }
    ggml_set_name(logits_t, "snap_prefill_logits");

    ggml_tensor * draft_meta_t = nullptr;
    ggml_tensor * draft_data_t = nullptr;
    if (has_dflash_features && draft_snap) {
        draft_meta_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);
        if (!draft_meta_t) {
            ggml_free(ctx);
            return false;
        }
        ggml_set_name(draft_meta_t, "dflash_feature_meta");
        draft_data_t = ggml_new_tensor_1d(
            ctx, GGML_TYPE_F32, (int64_t)draft_snap->data.size());
        if (!draft_data_t) {
            ggml_free(ctx);
            return false;
        }
        ggml_set_name(draft_data_t, "dflash_feature_data");
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) {
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return false;
    }

    std::vector<uint8_t> tmp(4 * 1024 * 1024);
    for (const CopyPair & cp : copies) {
        const size_t nbytes = ggml_nbytes(cp.src);
        size_t offset = 0;
        while (offset < nbytes) {
            const size_t chunk = std::min(tmp.size(), nbytes - offset);
            ggml_backend_tensor_get(cp.src, tmp.data(), offset, chunk);
            ggml_backend_tensor_set(cp.dst, tmp.data(), offset, chunk);
            offset += chunk;
        }
    }
    ggml_backend_tensor_set(logits_t, logits.data(), 0,
                            sizeof(float) * logits.size());
    if (draft_meta_t && draft_data_t && draft_snap) {
        const int32_t meta[5] = {
            draft_snap->cur_pos,
            draft_snap->start_pos,
            draft_snap->n_tokens,
            draft_snap->cap,
            draft_snap->n_target_layers,
        };
        ggml_backend_tensor_set(draft_meta_t, meta, 0, sizeof(meta));
        ggml_backend_tensor_set(draft_data_t, draft_snap->data.data(), 0,
                                sizeof(float) * draft_snap->data.size());
    }

    disk_snapshot_contexts_[(size_t)slot] = ctx;
    disk_snapshot_buffers_[(size_t)slot] = buf;
    disk_snapshot_backends_[(size_t)slot] = cpu;
    return true;
}

ModelBackend::SnapshotRef Qwen35LayerSplitAdapter::snapshot_ref(int slot) const {
    ModelBackend::SnapshotRef ref;
    if (use_mixed_target_split()) return ref;
    if (!snapshot_used(slot)) return ref;
    if (slot < 0 || slot >= (int)disk_snapshot_contexts_.size()) return ref;
    ref.ctx = disk_snapshot_contexts_[(size_t)slot];
    ref.buf = disk_snapshot_buffers_[(size_t)slot];
    ref.cur_pos = snapshot_cur_pos(slot);
    ref.last_tok = prefix_snapshots_[(size_t)slot].empty()
        ? current_last_token()
        : prefix_snapshots_[(size_t)slot].front().last_tok;
    return ref;
}

bool Qwen35LayerSplitAdapter::snapshot_adopt(int slot,
                                             ggml_context * ctx,
                                             ggml_backend_buffer_t buf,
                                             int cur_pos,
                                             int32_t last_tok) {
    if (use_mixed_target_split() || !snapshot_slot_valid(slot) || !ctx ||
        !buf || cur_pos <= 0) {
        return false;
    }
    snapshot_free(slot);
    auto & snaps = prefix_snapshots_[(size_t)slot];
    if (snaps.size() != shards_.size()) snaps.resize(shards_.size());
    if (snapshot_prefill_logits_.size() != (size_t)PREFIX_SLOTS ||
        snapshot_prefill_logit_tensors_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_contexts_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_buffers_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_backends_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }

    ggml_tensor * logits_tensor = nullptr;
    ggml_tensor * dflash_feature_meta = nullptr;
    ggml_tensor * dflash_feature_data = nullptr;

    auto fail = [&]() {
        for (auto & snap : snaps) snap = PrefixSnapshot{};
        snapshot_prefill_logits_[(size_t)slot].clear();
        snapshot_prefill_logit_tensors_[(size_t)slot].clear();
        free_draft_feature_snapshot(slot);
        return false;
    };

    for (size_t shard_idx = 0; shard_idx < shards_.size(); ++shard_idx) {
        auto & snap = snaps[shard_idx];
        snap.attn_k_snap.assign(shards_[shard_idx].cache.attn_k.size(), nullptr);
        snap.attn_v_snap.assign(shards_[shard_idx].cache.attn_v.size(), nullptr);
        snap.ssm_state_snap.assign(shards_[shard_idx].cache.ssm_state.size(), nullptr);
        snap.conv_state_snap.assign(shards_[shard_idx].cache.conv_state.size(), nullptr);
        snap.target_feat_snap = nullptr;
        snap.cur_pos = cur_pos;
        snap.last_tok = last_tok;
        snap.kv_k_type = shards_[shard_idx].cache.kv_k_type;
        snap.max_ctx = shards_[shard_idx].cache.max_ctx;
        snap.target_feat_cap = shards_[shard_idx].cache.target_feat_cap;
    }
    snapshot_prefill_logits_[(size_t)slot].clear();
    snapshot_prefill_logit_tensors_[(size_t)slot].assign(shards_.size(), nullptr);

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->name[0]) continue;
        if (std::strcmp(t->name, "snap_prefill_logits") == 0) {
            logits_tensor = t;
            continue;
        }
        if (std::strcmp(t->name, "dflash_feature_meta") == 0) {
            dflash_feature_meta = t;
            continue;
        }
        if (std::strcmp(t->name, "dflash_feature_data") == 0) {
            dflash_feature_data = t;
            continue;
        }
        int shard_idx = -1;
        int idx = -1;
        if (std::sscanf(t->name, "ls%d_snap_cache_k_%d", &shard_idx, &idx) == 2 &&
            shard_idx >= 0 && shard_idx < (int)shards_.size() &&
            idx >= 0 && idx < (int)snaps[(size_t)shard_idx].attn_k_snap.size()) {
            snaps[(size_t)shard_idx].attn_k_snap[(size_t)idx] = t;
        } else if (std::sscanf(t->name, "ls%d_snap_cache_v_%d", &shard_idx, &idx) == 2 &&
                   shard_idx >= 0 && shard_idx < (int)shards_.size() &&
                   idx >= 0 && idx < (int)snaps[(size_t)shard_idx].attn_v_snap.size()) {
            snaps[(size_t)shard_idx].attn_v_snap[(size_t)idx] = t;
        } else if (std::sscanf(t->name, "ls%d_snap_ssm_state_%d", &shard_idx, &idx) == 2 &&
                   shard_idx >= 0 && shard_idx < (int)shards_.size() &&
                   idx >= 0 && idx < (int)snaps[(size_t)shard_idx].ssm_state_snap.size()) {
            snaps[(size_t)shard_idx].ssm_state_snap[(size_t)idx] = t;
        } else if (std::sscanf(t->name, "ls%d_snap_conv_state_%d", &shard_idx, &idx) == 2 &&
                   shard_idx >= 0 && shard_idx < (int)shards_.size() &&
                   idx >= 0 && idx < (int)snaps[(size_t)shard_idx].conv_state_snap.size()) {
            snaps[(size_t)shard_idx].conv_state_snap[(size_t)idx] = t;
        } else if (std::sscanf(t->name, "ls%d_snap_target_feat", &shard_idx) == 1 &&
                   shard_idx >= 0 && shard_idx < (int)shards_.size()) {
            snaps[(size_t)shard_idx].target_feat_snap = t;
        } else if (std::sscanf(t->name, "ls%d_snap_prefill_logits", &shard_idx) == 1 &&
                   shard_idx >= 0 && shard_idx < (int)shards_.size()) {
            snapshot_prefill_logit_tensors_[(size_t)slot][(size_t)shard_idx] = t;
            logits_tensor = t;
        }
    }

    for (size_t shard_idx = 0; shard_idx < shards_.size(); ++shard_idx) {
        auto & snap = snaps[shard_idx];
        for (size_t i = 0; i < snap.attn_k_snap.size(); ++i) {
            const bool cache_has_kv =
                shards_[shard_idx].cache.attn_k[i] || shards_[shard_idx].cache.attn_v[i];
            if (cache_has_kv && (!snap.attn_k_snap[i] || !snap.attn_v_snap[i])) {
                return fail();
            }
        }
        for (size_t i = 0; i < snap.ssm_state_snap.size(); ++i) {
            const bool cache_has_state =
                shards_[shard_idx].cache.ssm_state[i] || shards_[shard_idx].cache.conv_state[i];
            if (cache_has_state && (!snap.ssm_state_snap[i] || !snap.conv_state_snap[i])) {
                return fail();
            }
        }
        if (shards_[shard_idx].cache.target_feat && !snap.target_feat_snap) {
            return fail();
        }
    }

    if (!logits_tensor || logits_tensor->ne[0] <= 0) {
        return fail();
    }
    snapshot_prefill_logits_[(size_t)slot].assign((size_t)logits_tensor->ne[0], 0.0f);
    ggml_backend_tensor_get(logits_tensor,
                            snapshot_prefill_logits_[(size_t)slot].data(),
                            0,
                            sizeof(float) *
                                snapshot_prefill_logits_[(size_t)slot].size());

    if (cfg_.run_dflash && cfg_.draft_path) {
        if (!dflash_feature_meta || !dflash_feature_data ||
            dflash_feature_meta->type != GGML_TYPE_I32 ||
            dflash_feature_data->type != GGML_TYPE_F32 ||
            dflash_feature_meta->ne[0] < 5 ||
            dflash_feature_data->ne[0] <= 0) {
            return fail();
        }
        int32_t meta[5] = {};
        ggml_backend_tensor_get(dflash_feature_meta, meta, 0, sizeof(meta));
        auto & draft_snap = draft_feature_snapshots_[(size_t)slot];
        draft_snap.cur_pos = meta[0];
        draft_snap.start_pos = meta[1];
        draft_snap.n_tokens = meta[2];
        draft_snap.cap = meta[3];
        draft_snap.n_target_layers = meta[4];
        const int hidden = remote_draft_.active() ? remote_draft_.hidden_size()
                                                  : feature_ring_.hidden_size;
        if (draft_snap.cur_pos <= 0 || draft_snap.start_pos < 0 ||
            draft_snap.n_tokens <= 0 || draft_snap.cap <= 0 ||
            draft_snap.n_target_layers <= 0 || hidden <= 0) {
            return fail();
        }
        const size_t expected =
            (size_t)draft_snap.n_tokens *
            (size_t)draft_snap.n_target_layers *
            (size_t)hidden;
        if ((size_t)dflash_feature_data->ne[0] != expected) {
            return fail();
        }
        draft_snap.hidden_size = hidden;
        draft_snap.data.assign(expected, 0.0f);
        ggml_backend_tensor_get(dflash_feature_data, draft_snap.data.data(), 0,
                                sizeof(float) * draft_snap.data.size());
    } else {
        free_draft_feature_snapshot(slot);
    }

    for (auto & snap : snaps) {
        snap.ctx = ctx;
        snap.buf = buf;
    }
    disk_snapshot_contexts_[(size_t)slot] = ctx;
    disk_snapshot_buffers_[(size_t)slot] = buf;
    disk_snapshot_backends_[(size_t)slot] = nullptr;
    std::fprintf(stderr,
                 "[target-split] adopted disk snapshot slot=%d shards=%zu pos=%d\n",
                 slot, shards_.size(), cur_pos);
    return true;
}

bool Qwen35LayerSplitAdapter::snapshot_draft_features(int slot) {
    if (!cfg_.run_dflash || !cfg_.draft_path) {
        free_draft_feature_snapshot(slot);
        return true;
    }
    if (!snapshot_slot_valid(slot) ||
        draft_feature_snapshots_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }

    const auto & snaps = prefix_snapshots_[(size_t)slot];
    if (snaps.empty() || !snaps.front().ctx) return false;
    const int cur_pos = snaps.front().cur_pos;
    if (cur_pos <= 0) return false;
    const int ring_cap = remote_draft_.active() ? remote_draft_.ring_cap() : feature_ring_.cap;
    const int n_layers = remote_draft_.active() ? remote_draft_.n_target_layers()
                                                : feature_ring_.n_target_layers;
    const int hidden = remote_draft_.active() ? remote_draft_.hidden_size()
                                              : feature_ring_.hidden_size;
    if (ring_cap <= 0 || n_layers <= 0 || hidden <= 0) return false;
    const int n_tokens = std::min(cur_pos, ring_cap);
    const int start_pos = cur_pos - n_tokens;
    if (n_tokens <= 0) return false;

    auto & snap = draft_feature_snapshots_[(size_t)slot];
    snap.cur_pos = cur_pos;
    snap.start_pos = start_pos;
    snap.n_tokens = n_tokens;
    snap.cap = ring_cap;
    snap.n_target_layers = n_layers;
    snap.hidden_size = hidden;
    snap.data.clear();
    snap.data.resize((size_t)n_tokens * (size_t)n_layers * (size_t)hidden);

    if (remote_draft_.active()) {
        return remote_draft_.get_feature_range(start_pos, n_tokens, snap.data);
    }

    return copy_feature_ring_range_to_host_f32(
        feature_ring_, start_pos, n_tokens, snap.data);
}

void Qwen35LayerSplitAdapter::free_draft_feature_snapshot(int slot) {
    if (slot < 0 || draft_feature_snapshots_.size() != (size_t)PREFIX_SLOTS ||
        slot >= (int)draft_feature_snapshots_.size()) {
        return;
    }
    draft_feature_snapshots_[(size_t)slot] = DraftFeatureSnapshot{};
}

bool Qwen35LayerSplitAdapter::restore_draft_features(int slot) {
    if (!cfg_.run_dflash || !cfg_.draft_path) return true;
    if (slot < 0 || draft_feature_snapshots_.size() != (size_t)PREFIX_SLOTS ||
        slot >= (int)draft_feature_snapshots_.size()) {
        return false;
    }

    const auto & snap = draft_feature_snapshots_[(size_t)slot];
    if (snap.cur_pos <= 0 || snap.start_pos < 0 || snap.n_tokens <= 0 ||
        snap.cap <= 0 || snap.n_target_layers <= 0 || snap.hidden_size <= 0 ||
        snap.data.empty()) {
        return false;
    }

    if (remote_draft_.active()) {
        if (snap.cap != remote_draft_.ring_cap() ||
            snap.n_target_layers != remote_draft_.n_target_layers() ||
            snap.hidden_size != remote_draft_.hidden_size()) {
            return false;
        }
        return remote_draft_.set_feature_range(snap.start_pos, snap.n_tokens, snap.data);
    }

    if (!feature_ring_.target_feat ||
        snap.cap != feature_ring_.cap ||
        snap.n_target_layers != feature_ring_.n_target_layers ||
        snap.hidden_size != feature_ring_.hidden_size) {
        return false;
    }
    return copy_host_f32_to_feature_ring_range(
        feature_ring_, snap.start_pos, snap.n_tokens, snap.data);
}

int Qwen35LayerSplitAdapter::current_last_token() const {
    if (shards_.empty()) return -1;
    return shards_.front().cache.last_tok;
}

bool Qwen35LayerSplitAdapter::decode_ar(
        int last_tok, int committed, int n_gen,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io) {
    if (n_gen <= 0) return true;
    const auto & w = shards_.front().weights;
    const int vocab = w.n_vocab;
    return run_layer_split_ar_decode(
        last_tok, committed, n_gen, vocab, prefill_last_logits_, sampler_,
        sampler_rng_,
        [&](const std::vector<int32_t> & one, int pos, int & next_tok,
            std::vector<float> * logits_out) {
            if (use_mixed_target_split()) {
                return run_qwen35_mixed_layer_split_forward(
                    shards_, remote_target_shard_, shards_.front().weights,
                    one, pos, 1, next_tok,
                    cfg_.kq_stride_pad, cfg_.fa_window,
                    /*argmax_out=*/nullptr,
                    logits_out,
                    (cfg_.run_dflash && !remote_draft_.active()) ? &feature_ring_ : nullptr,
                    remote_draft_.active() ? &remote_draft_ : nullptr);
            }
            return run_qwen35_layer_split_forward(
                shards_, shards_.front().weights, one, pos, 1, next_tok,
                cfg_.kq_stride_pad, cfg_.fa_window,
                (cfg_.run_dflash && !remote_draft_.active()) ? &feature_ring_ : nullptr,
                /*argmax_out=*/nullptr,
                logits_out,
                cfg_.run_dflash ? &remote_draft_ : nullptr,
                activation_type_);
        },
        [&](int tok) { return is_eos_tok(tok, w); },
        out_tokens, io);
}

bool Qwen35LayerSplitAdapter::can_dflash_decode() const {
    return cfg_.run_dflash && cfg_.draft_path && !sampler_.needs_logit_processing();
}

bool Qwen35LayerSplitAdapter::decode_dflash(
        const std::vector<int32_t> & prompt, int base_pos, int last_tok, int n_gen,
        std::vector<int32_t> & out_tokens, const DaemonIO & io,
        float & accept_rate_out) {
    accept_rate_out = 0.0f;
    const bool use_remote_draft = remote_draft_.active();
    Qwen35LayerSplitDFlashTarget target(
        shards_, use_remote_draft ? nullptr : &feature_ring_,
        cfg_.kq_stride_pad, cfg_.fa_window,
        use_remote_draft ? &remote_draft_ : nullptr,
        use_mixed_target_split() ? &remote_target_shard_ : nullptr);
    DaemonIO collect_io = io.with_token_callback([&](int32_t tok) -> bool {
        out_tokens.push_back(tok);
        return true;
    });
    double accept_rate = 0.0;
    const bool ok = run_dflash_spec_decode(
        target, draft_weights_, draft_backend_, feature_ring_, prompt, n_gen,
        last_tok, /*out_path=*/nullptr, cfg_.draft_ctx_max, collect_io,
        use_remote_draft ? &remote_draft_ : nullptr, /*hint_tokens=*/nullptr, base_pos,
        &accept_rate);
    accept_rate_out = (float)accept_rate;
    return ok;
}

const char * Qwen35LayerSplitAdapter::default_compress_drafter_path() const {
    return "/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf";
}

ModelBackend::CompressResult
Qwen35LayerSplitAdapter::compress(const ModelBackend::CompressRequest & req) {
    ModelBackend::CompressResult result;
    if (req.input_ids.empty() || req.drafter_path.empty()) return result;

    for (auto & shard : shards_) ggml_backend_synchronize(shard.backend);
    if (draft_backend_) ggml_backend_synchronize(draft_backend_);

    if (!pflash_drafter_loaded_) {
        std::fprintf(stderr, "[target-split][compress] loading drafter from %s ...\n",
                     req.drafter_path.c_str());
        if (!load_drafter(req.drafter_path, /*gpu_layers=*/999,
                          pflash_drafter_)) {
            std::fprintf(stderr,
                         "[target-split][compress] drafter init failed: %s\n",
                         dflash27b_last_error());
            return result;
        }
        pflash_drafter_loaded_ = true;
        std::fprintf(stderr, "[target-split][compress] drafter ready\n");
    }

    result.compressed_ids = drafter_score_and_compress(
        pflash_drafter_, req.input_ids, req.keep_ratio);
    result.ok = !result.compressed_ids.empty();
    if (result.ok) {
        std::fprintf(stderr, "[target-split][compress] %zu -> %zu tokens\n",
                     req.input_ids.size(), result.compressed_ids.size());
    }
    return result;
}

void Qwen35LayerSplitAdapter::free_drafter() {
    remote_draft_.close();
    if (pflash_drafter_loaded_) {
        dflash::common::free_drafter(pflash_drafter_);
        pflash_drafter_loaded_ = false;
    }
    step_graph_destroy(draft_sg_);
    step_graph_destroy(proj_sg_);
}

DFlashTarget * Qwen35LayerSplitAdapter::dflash_target() {
    if (!dflash_target_) {
        dflash_target_ = std::make_unique<Qwen35LayerSplitDFlashTarget>(
            shards_,
            (cfg_.run_dflash && !remote_draft_.active()) ? &feature_ring_ : nullptr,
            cfg_.kq_stride_pad, cfg_.fa_window,
            remote_draft_.active() ? &remote_draft_ : nullptr,
            use_mixed_target_split() ? &remote_target_shard_ : nullptr);
    }
    return dflash_target_.get();
}

void Qwen35LayerSplitAdapter::shutdown() {
    dflash_target_.reset();
    free_drafter();
    for (int slot = 0; slot < (int)prefix_snapshots_.size(); ++slot) {
        snapshot_free(slot);
    }
    remote_target_shard_.close();
    draft_feature_mirror_free(feature_ring_);
    free_draft_weights(draft_weights_);
    prefix_snapshots_.clear();
    snapshot_prefill_logits_.clear();
    snapshot_prefill_logit_tensors_.clear();
    disk_snapshot_contexts_.clear();
    disk_snapshot_buffers_.clear();
    disk_snapshot_backends_.clear();
    draft_feature_snapshots_.clear();
    auto shard_metas = layer_split_shard_metas(shards_);
    free_layer_split_snapshot_backends(shard_metas, snapshot_backends_);
    if (draft_backend_owned_ && draft_backend_) {
        ggml_backend_free(draft_backend_);
    }
    draft_backend_ = nullptr;
    draft_backend_owned_ = false;
    free_qwen35_layer_split_shards(shards_);
    shards_.clear();
}

}  // namespace dflash::common
