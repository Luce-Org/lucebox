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
#include "qwen3/qwen3_kvflash_scorer.h"
#include "kv_quant.h"

#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace dflash::common {

Qwen35LayerSplitAdapter::Qwen35LayerSplitAdapter(
        const Qwen35LayerSplitAdapterConfig & cfg)
    : cfg_(cfg) {}

Qwen35LayerSplitAdapter::~Qwen35LayerSplitAdapter() { shutdown(); }

bool Qwen35LayerSplitAdapter::init() {
    if (cfg_.device.is_layer_split() && cfg_.remote_target_shard.enabled()) {
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
                                      shard.weights)) {
            std::fprintf(stderr, "[target-split] load gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            return false;
        }
    }
    kvflash_read_config();
    if (kvflash_active() && cfg_.fa_window > 0) {
        std::fprintf(stderr,
            "[target-split][kvflash] fa_window must be 0 because pool slots "
            "need full slot-space masks\n");
        return false;
    }

    for (auto & shard : shards_) {
        if (!create_target_cache_partial(shard.weights, cfg_.device.max_ctx,
                                         cfg_.max_verify_tokens, shard.backend,
                                         shard.cache,
                                         /*prefill_only=*/!cfg_.run_dflash,
                                         shard.layer_begin, shard.layer_end,
                                         /*allocate_target_feat=*/false,
                                         kvflash_tokens_)) {
            std::fprintf(stderr, "[target-split] cache gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            return false;
        }
        std::fprintf(stderr, "[target-split] gpu=%d layers=[%d,%d)\n",
                     shard.gpu, shard.layer_begin, shard.layer_end);
    }
    if (!kvflash_attach()) return false;

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
    if (kvflash_active()) {
        kvflash_history_snapshots_.resize(PREFIX_SLOTS);
    }

    return true;
}

void Qwen35LayerSplitAdapter::kvflash_read_config() {
    if (!std::getenv("DFLASH_KVFLASH") || shards_.empty()) return;
    const bool target_shard_split =
        cfg_.device.is_layer_split() && cfg_.remote_target_shard.enabled();
    kvflash_drafter_path_ = target_shard_split
        ? std::string{}
        : kvflash_find_drafter(cfg_.target_path);

    ggml_type kv_k = GGML_TYPE_Q8_0;
    ggml_type kv_v = GGML_TYPE_Q8_0;
    dflash::resolve_kv_types(kv_k, kv_v);

    int64_t min_free = std::numeric_limits<int64_t>::max();
    int64_t max_bytes_per_token = 0;
    for (const auto & shard : shards_) {
        size_t gpu_free = 0, gpu_total = 0;
        if (ggml_backend_dev_t dev = ggml_backend_get_device(shard.backend)) {
            ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
        }
        min_free = std::min<int64_t>(min_free, (int64_t)gpu_free);

        int owned_full = 0;
        for (int il = shard.layer_begin; il < shard.layer_end; ++il) {
            if (((il + 1) % shard.weights.full_attention_interval) == 0) {
                ++owned_full;
            }
        }
        const int64_t bpt = (int64_t)owned_full * shard.weights.n_head_kv *
            (int64_t)(ggml_row_size(kv_k, shard.weights.n_embd_head_k) +
                      ggml_row_size(kv_v, shard.weights.n_embd_head_v));
        max_bytes_per_token = std::max<int64_t>(max_bytes_per_token, bpt);
    }
    if (min_free == std::numeric_limits<int64_t>::max()) min_free = 0;

    KvFlashAutoBudget budget;
    budget.free_bytes = min_free;
    budget.bytes_per_token = max_bytes_per_token;
    budget.reserve_bytes = (int64_t)(1.5 * 1073741824.0) +
        (kvflash_drafter_path_.empty() ? 0 : (int64_t)(1.7 * 1073741824.0));
    kvflash_tokens_ = kvflash_pool_from_env(
        cfg_.device.max_ctx, KvFlashConfig{},
        !kvflash_drafter_path_.empty(), budget);
    if (kvflash_tokens_ > 0) {
        const char * tau = std::getenv("DFLASH_KVFLASH_TAU");
        kvflash_tau_ = std::max(1, tau ? std::atoi(tau) : 64);
    }
}

bool Qwen35LayerSplitAdapter::kvflash_attach() {
    if (!kvflash_active()) return true;
    std::vector<ggml_tensor *> full_k;
    std::vector<ggml_tensor *> full_v;
    const int n_full = shards_.front().weights.n_layer /
        shards_.front().weights.full_attention_interval;
    for (int i = 0; i < n_full; ++i) {
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
        for (auto & shard : shards_) {
            if (i < (int)shard.cache.attn_k.size() && shard.cache.attn_k[(size_t)i]) {
                k = shard.cache.attn_k[(size_t)i];
                v = shard.cache.attn_v[(size_t)i];
                break;
            }
        }
        if (k && v) {
            full_k.push_back(k);
            full_v.push_back(v);
        }
    }
    KvFlashConfig pc;
    pc.pool_tokens = kvflash_tokens_;
    if (!kvflash_pager_.attach(pc, full_k, full_v)) {
        std::fprintf(stderr,
            "[target-split][kvflash] pager attach failed pool=%d layers=%zu\n",
            kvflash_tokens_, full_k.size());
        return false;
    }
    std::printf("[target-split][kvflash] resident pool %d tokens over %zu "
                "full-attn layers (logical max_ctx %d), tau=%d, policy=%s\n",
                kvflash_tokens_, full_k.size(), cfg_.device.max_ctx,
                kvflash_tau_,
                !kvflash_drafter_path_.empty()
                    ? "drafter (attaches on first reselect)"
                    : "lru (recency-only: no Qwen3-0.6B drafter found)");
    std::fflush(stdout);
    return true;
}

bool Qwen35LayerSplitAdapter::kvflash_sync_identity(int committed) {
    if (!kvflash_active()) return true;
    if (!layer_split_kvflash_sync_identity(
            kvflash_pager_, committed, kvflash_tokens_, "target-split")) {
        return false;
    }
    if (use_mixed_target_split() &&
        !remote_target_shard_.kvflash_sync_identity(committed)) {
        std::fprintf(stderr,
            "[target-split][kvflash] remote identity sync failed pos=%d\n",
            committed);
        return false;
    }
    return true;
}

void Qwen35LayerSplitAdapter::kvflash_sync_history(
        const std::vector<int32_t> & tokens, int base_pos) {
    if (!kvflash_active()) return;
    layer_split_kvflash_sync_history(kvflash_history_, tokens, base_pos);
}

void Qwen35LayerSplitAdapter::kvflash_maybe_reselect(int generated) {
    if (!kvflash_active() || kvflash_tau_ <= 0) return;
    if (use_mixed_target_split()) return;
    const int tau = std::max<int>(kvflash_tau_, (int)(kvflash_history_.size() / 45));
    if (generated % tau != 0) return;
    if (!kvflash_scorer_) {
        if (kvflash_drafter_path_.empty() || kvflash_drafter_failed_) return;
        if (!kvflash_drafter_loaded_) {
            for (auto & shard : shards_) ggml_backend_synchronize(shard.backend);
            std::fprintf(stderr,
                "[target-split][kvflash] loading residency drafter: %s\n",
                kvflash_drafter_path_.c_str());
            if (!load_drafter(kvflash_drafter_path_, /*gpu_layers=*/999,
                              shards_.front().gpu, kvflash_drafter_)) {
                std::fprintf(stderr,
                    "[target-split][kvflash] drafter load failed (%s); "
                    "staying on LRU residency\n",
                    dflash27b_last_error());
                kvflash_drafter_failed_ = true;
                return;
            }
            kvflash_drafter_loaded_ = true;
        }
        kvflash_scorer_ =
            std::make_unique<KvFlashDrafterScorer>(&kvflash_drafter_);
        std::fprintf(stderr,
            "[target-split][kvflash] drafter scorer attached (tau=%d)\n",
            kvflash_tau_);
    }
    if (!kvflash_scorer_->score_chunks(
            kvflash_history_, kvflash_pager_.chunk_tokens(), kvflash_scores_)) {
        return;
    }
    kvflash_pager_.score_hook = [this](int c) {
        return c < (int)kvflash_scores_.size() ? kvflash_scores_[(size_t)c] : 1e30f;
    };
    const int events = kvflash_pager_.reselect();
    if (events > 0) {
        std::fprintf(stderr,
            "[target-split][kvflash] reselect @gen=%d: %d page events\n",
            generated, events);
    }
}

bool Qwen35LayerSplitAdapter::init_mixed_target_split() {
    if (!cfg_.remote_target_shard.enabled() ||
        cfg_.device.layer_split_gpus.size() < 2) {
        std::fprintf(stderr,
            "[target-split] mixed target split requires at least two shards and "
            "remote target shard IPC\n");
        return false;
    }
    MixedLayerSplitPlan mixed_plan;
    if (!compute_target_shard_layer_split_plan(
            cfg_.device, compiled_placement_backend(), mixed_plan,
            "target-split")) {
        return false;
    }
    const size_t remote_begin = mixed_plan.remote_begin;
    const PlacementBackend remote_backend = mixed_plan.remote_backend;

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
                                      local.weights)) {
            std::fprintf(stderr, "[target-split] mixed local load gpu=%d: %s\n",
                         local.gpu, dflash27b_last_error());
            return false;
        }
    }

    kvflash_read_config();
    if (kvflash_active() && cfg_.fa_window > 0) {
        std::fprintf(stderr,
            "[target-split][kvflash] fa_window must be 0 because pool slots "
            "need full slot-space masks\n");
        return false;
    }

    for (auto & local : shards_) {
        if (!create_target_cache_partial(local.weights, cfg_.device.max_ctx,
                                         cfg_.max_verify_tokens, local.backend,
                                         local.cache,
                                         /*prefill_only=*/!cfg_.run_dflash,
                                         local.layer_begin, local.layer_end,
                                         /*allocate_target_feat=*/false,
                                         kvflash_tokens_)) {
            std::fprintf(stderr, "[target-split] mixed local cache gpu=%d: %s\n",
                         local.gpu, dflash27b_last_error());
            return false;
        }
    }
    if (!kvflash_attach()) return false;

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
            cfg_.run_dflash,
            kvflash_tokens_)) {
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
    if (kvflash_active()) {
        kvflash_history_snapshots_.resize(PREFIX_SLOTS);
    }
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
    if (kvflash_active()) {
        kvflash_pager_.reset();
        kvflash_history_.clear();
        kvflash_scores_.clear();
        kvflash_pager_.score_hook = nullptr;
    }
    if (use_mixed_target_split() &&
        !remote_target_shard_.reset_request_state()) {
        std::fprintf(stderr,
            "[target-split] remote shard reset_request_state failed\n");
    }
    prefill_last_logits_.clear();
}

int Qwen35LayerSplitAdapter::prefill_chunk_tokens() const {
    if (kvflash_active()) {
        return kvflash_pager_.chunk_tokens();
    }
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
    int ubatch = cfg_.chunk > 0 ? cfg_.chunk : 512;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        ubatch = std::max(1, std::atoi(s));
    }
    if (use_mixed_target_split()) {
        const bool ok = run_qwen35_mixed_layer_split_forward(
            shards_, remote_target_shard_, shards_.front().weights,
            prompt, base_pos, ubatch, last_tok,
            cfg_.kq_stride_pad, /*fa_window=*/0,
            /*argmax_out=*/nullptr,
            &prefill_last_logits_,
            (cfg_.run_dflash && !remote_draft_.active()) ? &feature_ring_ : nullptr,
            remote_draft_.active() ? &remote_draft_ : nullptr,
            kvflash_active() ? &kvflash_pager_ : nullptr);
        if (ok && kvflash_active()) {
            kvflash_sync_history(prompt, base_pos);
            kvflash_pager_.zero_free_blocks();
        }
        return ok;
    }
    const bool ok = run_qwen35_layer_split_forward(
        shards_, shards_.front().weights, prompt, base_pos, ubatch, last_tok,
        cfg_.kq_stride_pad, /*fa_window=*/0,
        (cfg_.run_dflash && !remote_draft_.active()) ? &feature_ring_ : nullptr,
        /*argmax_out=*/nullptr,
        &prefill_last_logits_,
        cfg_.run_dflash ? &remote_draft_ : nullptr,
        activation_type_,
        kvflash_active() ? &kvflash_pager_ : nullptr);
    if (ok && kvflash_active()) {
        kvflash_sync_history(prompt, base_pos);
        kvflash_pager_.zero_free_blocks();
    }
    return ok;
}

bool Qwen35LayerSplitAdapter::snapshot_slot_valid(int slot) const {
    return slot >= 0 && slot < PREFIX_SLOTS &&
           prefix_snapshots_.size() == (size_t)PREFIX_SLOTS &&
           !shards_.empty();
}

bool Qwen35LayerSplitAdapter::snapshot_save(int slot) {
    if (!snapshot_slot_valid(slot)) return false;
    if (snapshot_backends_.size() != shards_.size()) return false;
    const int cur_pos = shards_.empty() ? 0 : shards_.front().cache.cur_pos;
    if (kvflash_active() &&
        (cur_pos > kvflash_tokens_ || !kvflash_pager_.is_identity())) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                "[target-split][kvflash] snapshot skipped: pooled layout "
                "needs page-table serialization\n");
            warned = true;
        }
        return false;
    }
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
    if (kvflash_active() &&
        kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS) {
        layer_split_kvflash_save_history_snapshot(
            kvflash_history_, cur_pos, kvflash_history_snapshots_[(size_t)slot]);
    }
    if (!snapshot_draft_features(slot)) {
        snapshot_free(slot);
        return false;
    }
    if (!rebuild_disk_snapshot(slot)) {
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
    if (kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS) {
        kvflash_history_snapshots_[(size_t)slot].clear();
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
    if (kvflash_active()) {
        if (!kvflash_sync_identity(cur_pos)) return false;
        layer_split_kvflash_restore_history(
            kvflash_history_, kvflash_history_snapshots_, slot, cur_pos);
    }
    if (!restore_draft_features(slot)) return false;
    return true;
}

bool Qwen35LayerSplitAdapter::rebuild_disk_snapshot(int slot) {
    if (!snapshot_slot_valid(slot)) return false;
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
    const bool mixed_target_split = use_mixed_target_split();
    Qwen35TargetShardSnapshotData remote_snapshot;
    if (mixed_target_split) {
        if (!remote_target_shard_.snapshot_export(slot, remote_snapshot) ||
            remote_snapshot.shard_count <= 0 ||
            remote_snapshot.cur_pos <= 0 ||
            remote_snapshot.tensors.empty() ||
            remote_snapshot.logits.empty()) {
            return false;
        }
        if (snaps.empty() || !snaps.front().ctx ||
            remote_snapshot.cur_pos != snaps.front().cur_pos) {
            return false;
        }
        for (const auto & t : remote_snapshot.tensors) {
            if (t.shard < 0 || t.shard >= remote_snapshot.shard_count ||
                t.name.empty() || t.name.size() >= (size_t)GGML_MAX_NAME ||
                t.type >= GGML_TYPE_COUNT || t.data.empty()) {
                return false;
            }
        }
    }
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
    n_tensors += remote_snapshot.tensors.size();

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
    struct RemoteCopy {
        const Qwen35TargetShardSnapshotTensor * src = nullptr;
        ggml_tensor * dst = nullptr;
    };
    std::vector<RemoteCopy> remote_copies;
    remote_copies.reserve(remote_snapshot.tensors.size());

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
    for (const auto & src : remote_snapshot.tensors) {
        ggml_tensor * dst = ggml_new_tensor(
            ctx, (ggml_type)src.type, 4, src.ne);
        if (!dst) {
            ggml_free(ctx);
            return false;
        }
        if (ggml_nbytes(dst) != src.data.size()) {
            ggml_free(ctx);
            return false;
        }
        const std::string name =
            "ls" + std::to_string(snaps.size() + (size_t)src.shard) +
            "_" + src.name;
        ggml_set_name(dst, name.c_str());
        remote_copies.push_back({&src, dst});
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
    for (const RemoteCopy & cp : remote_copies) {
        ggml_backend_tensor_set(cp.dst, cp.src->data.data(), 0,
                                cp.src->data.size());
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
    if (!snapshot_used(slot)) return ref;
    if (slot < 0 || slot >= (int)disk_snapshot_contexts_.size()) return ref;
    if (!disk_snapshot_contexts_[(size_t)slot] ||
        !disk_snapshot_buffers_[(size_t)slot]) {
        return ref;
    }
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
    const bool mixed_target_split = use_mixed_target_split();
    if (!snapshot_slot_valid(slot) || !ctx || !buf || cur_pos <= 0) {
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
    if (kvflash_active() &&
        kvflash_history_snapshots_.size() != (size_t)PREFIX_SLOTS) {
        kvflash_history_snapshots_.resize(PREFIX_SLOTS);
    }

    ggml_tensor * logits_tensor = nullptr;
    ggml_tensor * dflash_feature_meta = nullptr;
    ggml_tensor * dflash_feature_data = nullptr;
    Qwen35TargetShardSnapshotData remote_import;
    std::vector<int> remote_tensor_counts;
    if (mixed_target_split) {
        const size_t local_shard_count = shards_.size();
        const size_t total_shard_count = cfg_.device.layer_split_gpus.size();
        if (total_shard_count <= local_shard_count ||
            total_shard_count - local_shard_count >
                (size_t)std::numeric_limits<int>::max()) {
            return false;
        }
        remote_import.shard_count =
            (int)(total_shard_count - local_shard_count);
        remote_import.cur_pos = cur_pos;
        remote_import.last_tok = last_tok;
        remote_tensor_counts.assign((size_t)remote_import.shard_count, 0);
    }

    auto fail = [&]() {
        for (auto & snap : snaps) snap = PrefixSnapshot{};
        snapshot_prefill_logits_[(size_t)slot].clear();
        snapshot_prefill_logit_tensors_[(size_t)slot].clear();
        if (kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS) {
            kvflash_history_snapshots_[(size_t)slot].clear();
        }
        free_draft_feature_snapshot(slot);
        if (mixed_target_split) {
            remote_target_shard_.snapshot_free(slot);
        }
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
    if (kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS) {
        kvflash_history_snapshots_[(size_t)slot].clear();
    }

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
        } else if (mixed_target_split) {
            const char * name = t->name;
            if (name[0] == 'l' && name[1] == 's') {
                char * end = nullptr;
                const long parsed = std::strtol(name + 2, &end, 10);
                if (end && *end == '_' && parsed >= (long)shards_.size() &&
                    parsed < (long)shards_.size() + remote_import.shard_count) {
                    const int remote_shard = (int)parsed - (int)shards_.size();
                    const char * raw_name = end + 1;
                    if (!raw_name[0] || t->type >= GGML_TYPE_COUNT) {
                        return fail();
                    }
                    Qwen35TargetShardSnapshotTensor remote_tensor;
                    remote_tensor.shard = remote_shard;
                    remote_tensor.name = raw_name;
                    remote_tensor.type = (uint32_t)t->type;
                    for (int d = 0; d < 4; ++d) {
                        remote_tensor.ne[d] = t->ne[d];
                    }
                    const size_t nbytes = ggml_nbytes(t);
                    if (nbytes == 0) return fail();
                    remote_tensor.data.assign(nbytes, 0);
                    ggml_backend_tensor_get(t, remote_tensor.data.data(), 0, nbytes);
                    remote_import.tensors.push_back(std::move(remote_tensor));
                    remote_tensor_counts[(size_t)remote_shard]++;
                }
            }
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
    if (mixed_target_split) {
        if (remote_import.tensors.empty()) {
            return fail();
        }
        if (remote_tensor_counts.size() != (size_t)remote_import.shard_count) {
            return fail();
        }
        for (int count : remote_tensor_counts) {
            if (count <= 0) return fail();
        }
        remote_import.logits = snapshot_prefill_logits_[(size_t)slot];
        if (!remote_target_shard_.snapshot_import(slot, remote_import)) {
            return fail();
        }
    }

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
    if (kvflash_active()) {
        kvflash_history_snapshots_[(size_t)slot].assign((size_t)cur_pos, 0);
    }

    for (auto & snap : snaps) {
        snap.ctx = ctx;
        snap.buf = buf;
    }
    disk_snapshot_contexts_[(size_t)slot] = ctx;
    disk_snapshot_buffers_[(size_t)slot] = buf;
    disk_snapshot_backends_[(size_t)slot] = nullptr;
    if (mixed_target_split) {
        std::fprintf(stderr,
                     "[target-split] adopted disk snapshot slot=%d local_shards=%zu remote_shards=%d pos=%d\n",
                     slot, shards_.size(), remote_import.shard_count, cur_pos);
    } else {
        std::fprintf(stderr,
                     "[target-split] adopted disk snapshot slot=%d shards=%zu pos=%d\n",
                     slot, shards_.size(), cur_pos);
    }
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
        const std::vector<int32_t> & history_prefix,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io) {
    if (n_gen <= 0) return true;
    const auto & w = shards_.front().weights;
    const int vocab = w.n_vocab;
    const bool ok = run_layer_split_ar_decode(
        last_tok, committed, n_gen, vocab, prefill_last_logits_, sampler_,
        sampler_rng_, history_prefix,
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
                    remote_draft_.active() ? &remote_draft_ : nullptr,
                    kvflash_active() ? &kvflash_pager_ : nullptr);
            }
            return run_qwen35_layer_split_forward(
                shards_, shards_.front().weights, one, pos, 1, next_tok,
                cfg_.kq_stride_pad, cfg_.fa_window,
                (cfg_.run_dflash && !remote_draft_.active()) ? &feature_ring_ : nullptr,
                /*argmax_out=*/nullptr,
                logits_out,
                cfg_.run_dflash ? &remote_draft_ : nullptr,
                activation_type_,
                kvflash_active() ? &kvflash_pager_ : nullptr);
        },
        [&](int tok) { return is_eos_tok(tok, w); },
        out_tokens, io);
    if (ok && kvflash_active()) {
        kvflash_sync_history(out_tokens, committed);
        kvflash_maybe_reselect((int)out_tokens.size());
    }
    return ok;
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
        use_mixed_target_split() ? &remote_target_shard_ : nullptr,
        kvflash_active() ? &kvflash_pager_ : nullptr);
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
    if (ok && kvflash_active()) {
        kvflash_sync_history(out_tokens, base_pos + (int)prompt.size());
        kvflash_maybe_reselect((int)out_tokens.size());
    }
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
    kvflash_scorer_.reset();
    if (kvflash_drafter_loaded_) {
        dflash::common::free_drafter(kvflash_drafter_);
        kvflash_drafter_loaded_ = false;
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
            use_mixed_target_split() ? &remote_target_shard_ : nullptr,
            kvflash_active() ? &kvflash_pager_ : nullptr);
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
    kvflash_history_snapshots_.clear();
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
