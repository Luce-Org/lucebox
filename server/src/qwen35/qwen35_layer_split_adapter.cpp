// Qwen35 layer-split adapter.

#include "qwen35_layer_split_adapter.h"

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

#include <algorithm>
#include <cstdio>
#include <cstdlib>

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
        cfg_.run_dflash ? &remote_draft_ : nullptr);
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
    return true;
}

void Qwen35LayerSplitAdapter::snapshot_free(int slot) {
    if (!snapshot_slot_valid(slot)) return;
    for (auto & snap : prefix_snapshots_[(size_t)slot]) {
        free_prefix_snapshot(snap);
    }
    if (snapshot_prefill_logits_.size() == (size_t)PREFIX_SLOTS) {
        snapshot_prefill_logits_[(size_t)slot].clear();
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
                cfg_.run_dflash ? &feature_ring_ : nullptr,
                /*argmax_out=*/nullptr,
                logits_out);
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
    remote_target_shard_.close();
    draft_feature_mirror_free(feature_ring_);
    free_draft_weights(draft_weights_);
    for (auto & slot : prefix_snapshots_) {
        for (auto & snap : slot) free_prefix_snapshot(snap);
    }
    prefix_snapshots_.clear();
    snapshot_prefill_logits_.clear();
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
