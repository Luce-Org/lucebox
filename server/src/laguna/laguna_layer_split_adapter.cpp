// Laguna target layer-split adapter.

#include "laguna_layer_split_adapter.h"

#include "common/backend_precision.h"
#include "common/dflash_layer_split_runtime.h"
#include "common/gguf_inspect.h"
#include "common/layer_split_utils.h"
#include "common/sampler.h"
#include "common/layer_split_runtime.h"
#include "common/target_shard_ipc_daemon.h"
#include "dflash27b.h"
#include "placement/placement_backend.h"
#include "qwen3/qwen3_kvflash_scorer.h"

#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

namespace {

static bool tensor_ready(const ggml_tensor * t) {
    return t && t->buffer;
}

}  // namespace

LagunaLayerSplitAdapter::LagunaLayerSplitAdapter(
        const LagunaLayerSplitAdapterConfig & cfg)
    : cfg_(cfg) {}

LagunaLayerSplitAdapter::~LagunaLayerSplitAdapter() { shutdown(); }

bool LagunaLayerSplitAdapter::init() {
    if (cfg_.device.is_layer_split() && cfg_.remote_target_shard.enabled()) {
        return init_mixed_target_split();
    }

    const LayerSplitRuntimeInit runtime_cfg{
        cfg_.target_path,
        &cfg_.device,
        "laguna-target-split",
    };
    if (!init_layer_split_runtime(runtime_cfg, shards_, snapshot_backends_)) {
        return false;
    }

    std::vector<ggml_backend_t> shard_backends;
    shard_backends.reserve(shards_.size());
    for (const auto & shard : shards_) shard_backends.push_back(shard.backend);
    const BackendActivationPolicy activation_policy =
        select_common_activation_precision_policy(
            shard_backends, /*force_f32=*/false,
            "LUCEBOX_LAYER_SPLIT_ACT_TYPE");
    activation_type_ = activation_policy.activation_type;
    std::fprintf(stderr, "[laguna-target-split] activation=%s (%s",
                 backend_precision_type_name(activation_type_),
                 activation_policy.reason.c_str());
    if (!activation_policy.runtime_arch.empty()) {
        std::fprintf(stderr, ", arch=%s", activation_policy.runtime_arch.c_str());
    } else if (activation_policy.cuda_sm > 0) {
        std::fprintf(stderr, ", sm=%d", activation_policy.cuda_sm);
    }
    std::fprintf(stderr, ")\n");

    for (size_t i = 0; i < shards_.size(); ++i) {
        auto & shard = shards_[i];
        const TargetLoadPlan plan =
            make_layer_split_load_plan<TargetLoadPlan>(shard, i + 1 == shards_.size());
        if (!load_target_gguf_laguna_partial(
                cfg_.target_path, shard.backend, plan, shard.weights)) {
            std::fprintf(stderr,
                "[laguna-target-split] load gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
            return false;
        }
    }

    kvflash_read_config();

    for (size_t i = 0; i < shards_.size(); ++i) {
        auto & shard = shards_[i];
        if (!create_laguna_target_cache_partial(
                shard.weights, cfg_.device.max_ctx, shard.backend,
                shard.layer_begin, shard.layer_end, shard.cache,
                kvflash_tokens_)) {
            std::fprintf(stderr,
                "[laguna-target-split] cache gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
            return false;
        }
        std::fprintf(stderr, "[laguna-target-split] gpu=%d layers=[%d,%d)\n",
                     shard.gpu, shard.layer_begin, shard.layer_end);
    }
    if (!kvflash_attach()) return false;

    snapshots_.resize(PREFIX_SLOTS);
    for (auto & slot : snapshots_) {
        slot.shards.resize(shards_.size());
    }
    snapshot_prefill_logit_tensors_.resize(PREFIX_SLOTS);
    disk_snapshot_contexts_.assign(PREFIX_SLOTS, nullptr);
    disk_snapshot_buffers_.assign(PREFIX_SLOTS, nullptr);
    disk_snapshot_backends_.assign(PREFIX_SLOTS, nullptr);
    kvflash_history_snapshots_.resize(PREFIX_SLOTS);
    return true;
}

bool LagunaLayerSplitAdapter::init_mixed_target_split() {
    if (!cfg_.remote_target_shard.enabled() ||
        cfg_.device.layer_split_gpus.size() < 2) {
        std::fprintf(stderr,
            "[laguna-target-split] mixed target split requires remote target shard IPC\n");
        return false;
    }
    MixedLayerSplitPlan mixed_plan;
    if (!compute_target_shard_layer_split_plan(
            cfg_.device, compiled_placement_backend(), mixed_plan,
            "laguna-target-split")) {
        return false;
    }

    const auto info = inspect_gguf_model_info(cfg_.target_path);
    const int n_layer = info.n_layer;
    if (n_layer <= 0) {
        std::fprintf(stderr,
            "[laguna-target-split] failed to inspect target layer count\n");
        return false;
    }
    const auto ranges = compute_layer_ranges(
        n_layer, (int)cfg_.device.layer_split_gpus.size(),
        cfg_.device.layer_split_weights);
    if (ranges.size() != cfg_.device.layer_split_gpus.size()) {
        std::fprintf(stderr,
            "[laguna-target-split] bad mixed layer split for %zu GPUs and %d layers\n",
            cfg_.device.layer_split_gpus.size(), n_layer);
        return false;
    }

    const size_t remote_begin = mixed_plan.remote_begin;
    shards_.resize(remote_begin);
    for (size_t i = 0; i < remote_begin; ++i) {
        auto & shard = shards_[i];
        shard.placement_backend = cfg_.device.layer_split_backend(i);
        shard.gpu = cfg_.device.layer_split_gpus[i];
        shard.layer_begin = ranges[i].begin;
        shard.layer_end = ranges[i].end;
        shard.backend = ggml_backend_cuda_init(shard.gpu);
        if (!shard.backend) {
            std::fprintf(stderr,
                "[laguna-target-split] local backend init failed gpu=%d\n",
                shard.gpu);
            return false;
        }
    }

    std::vector<ggml_backend_t> shard_backends;
    shard_backends.reserve(shards_.size());
    for (const auto & shard : shards_) shard_backends.push_back(shard.backend);
    const BackendActivationPolicy activation_policy =
        select_common_activation_precision_policy(
            shard_backends, /*force_f32=*/false,
            "LUCEBOX_LAYER_SPLIT_ACT_TYPE");
    activation_type_ = activation_policy.activation_type;

    for (auto & shard : shards_) {
        const TargetLoadPlan plan =
            make_layer_split_load_plan<TargetLoadPlan>(
                shard, /*is_last_shard=*/false);
        if (!load_target_gguf_laguna_partial(
                cfg_.target_path, shard.backend, plan, shard.weights)) {
            std::fprintf(stderr,
                "[laguna-target-split] mixed local load gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
            return false;
        }
    }

    kvflash_read_config();
    for (auto & shard : shards_) {
        if (!create_laguna_target_cache_partial(
                shard.weights, cfg_.device.max_ctx, shard.backend,
                shard.layer_begin, shard.layer_end, shard.cache,
                kvflash_tokens_)) {
            std::fprintf(stderr,
                "[laguna-target-split] mixed local cache gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
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

    const LagunaTargetWeights & ref = shards_.front().weights;
    TargetShardIpcLaunchConfig launch;
    launch.mode = BackendIpcMode::LagunaTargetShard;
    launch.bin = cfg_.remote_target_shard.ipc_bin;
    launch.target_path = cfg_.target_path ? cfg_.target_path : "";
    launch.gpus = remote_gpus;
    launch.layer_begins = remote_layer_begins;
    launch.layer_ends = remote_layer_ends;
    launch.max_ctx = cfg_.device.max_ctx;
    launch.hidden = ref.n_embd;
    launch.vocab = (int)ref.embedder.n_vocab;
    launch.max_tokens = std::max(1, cfg_.device.max_ctx);
    launch.work_dir = cfg_.remote_target_shard.work_dir;
    launch.kvflash_pool_tokens = kvflash_tokens_;
    if (!remote_target_shard_.start(launch)) {
        std::fprintf(stderr,
            "[laguna-target-split] remote target shard start failed layers=[%d,%d)\n",
            remote_layer_begins.front(), remote_layer_ends.back());
        return false;
    }

    snapshots_.resize(PREFIX_SLOTS);
    for (auto & slot : snapshots_) slot.shards.resize(shards_.size());
    snapshot_prefill_logit_tensors_.resize(PREFIX_SLOTS);
    disk_snapshot_contexts_.assign(PREFIX_SLOTS, nullptr);
    disk_snapshot_buffers_.assign(PREFIX_SLOTS, nullptr);
    disk_snapshot_backends_.assign(PREFIX_SLOTS, nullptr);
    auto shard_metas = layer_split_shard_metas(shards_);
    if (!init_layer_split_snapshot_backends(
            shard_metas, snapshot_backends_, "laguna-target-split")) {
        return false;
    }
    kvflash_history_snapshots_.resize(PREFIX_SLOTS);
    return true;
}

KvFlashConfig LagunaLayerSplitAdapter::kvflash_config() const {
    KvFlashConfig pc;
    if (!shards_.empty() && shards_.front().weights.sliding_window > 0) {
        pc.tail_window_chunks =
            std::max(4, (shards_.front().weights.sliding_window +
                         pc.chunk_tokens - 1) / pc.chunk_tokens + 1);
    }
    return pc;
}

void LagunaLayerSplitAdapter::kvflash_read_config() {
    if (!std::getenv("DFLASH_KVFLASH") || shards_.empty()) return;
    kvflash_drafter_path_ = kvflash_find_drafter(cfg_.target_path);

    int64_t min_free = std::numeric_limits<int64_t>::max();
    int64_t max_bytes_per_token = 0;
    const LagunaTargetWeights & ref = shards_.front().weights;
    for (const auto & shard : shards_) {
        size_t gpu_free = 0, gpu_total = 0;
        if (ggml_backend_dev_t dev = ggml_backend_get_device(shard.backend)) {
            ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
        }
        min_free = std::min<int64_t>(min_free, (int64_t)gpu_free);

        const int owned_layers =
            std::max(0, shard.layer_end - shard.layer_begin);
        const int64_t bpt = (int64_t)owned_layers * ref.n_head_kv * 2 *
            (int64_t)ggml_row_size(GGML_TYPE_Q8_0, ref.head_dim);
        max_bytes_per_token = std::max<int64_t>(max_bytes_per_token, bpt);
    }
    if (min_free == std::numeric_limits<int64_t>::max()) min_free = 0;

    KvFlashAutoBudget budget;
    budget.free_bytes = min_free;
    budget.bytes_per_token = max_bytes_per_token;
    budget.reserve_bytes = (int64_t)(1.5 * 1073741824.0) +
        (kvflash_drafter_path_.empty() ? 0 : (int64_t)(1.7 * 1073741824.0));
    kvflash_tokens_ = kvflash_pool_from_env(
        cfg_.device.max_ctx, kvflash_config(),
        !kvflash_drafter_path_.empty(), budget);
    if (kvflash_tokens_ > 0) {
        const char * tau = std::getenv("DFLASH_KVFLASH_TAU");
        kvflash_tau_ = std::max(1, tau ? std::atoi(tau) : 64);
    }
}

bool LagunaLayerSplitAdapter::kvflash_attach() {
    if (!kvflash_active()) return true;
    std::vector<ggml_tensor *> all_k;
    std::vector<ggml_tensor *> all_v;
    const int n_layer = shards_.empty() ? 0 : shards_.front().weights.n_layer;
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
        for (auto & shard : shards_) {
            if (il < (int)shard.cache.attn_k.size() &&
                shard.cache.attn_k[(size_t)il]) {
                k = shard.cache.attn_k[(size_t)il];
                v = shard.cache.attn_v[(size_t)il];
                break;
            }
        }
        if (k && v) {
            all_k.push_back(k);
            all_v.push_back(v);
        }
    }
    KvFlashConfig pc = kvflash_config();
    pc.pool_tokens = kvflash_tokens_;
    if (!kvflash_pager_.attach(pc, all_k, all_v)) {
        std::fprintf(stderr,
            "[laguna-target-split][kvflash] pager attach failed pool=%d layers=%zu\n",
            kvflash_tokens_, all_k.size());
        return false;
    }
    std::printf("[laguna-target-split][kvflash] resident pool %d tokens over "
                "%zu layers (logical max_ctx %d), tau=%d, policy=%s, "
                "swa_tail=%d chunks\n",
                kvflash_tokens_, all_k.size(), cfg_.device.max_ctx,
                kvflash_tau_,
                !kvflash_drafter_path_.empty()
                    ? "drafter/cross-tok (attaches on first reselect)"
                    : "lru (recency-only: no Qwen3-0.6B drafter found)",
                pc.tail_window_chunks);
    std::fflush(stdout);
    return true;
}

bool LagunaLayerSplitAdapter::kvflash_sync_identity(int committed) {
    if (!kvflash_active()) return true;
    if (!layer_split_kvflash_sync_identity(
            kvflash_pager_, committed, kvflash_tokens_, "laguna-target-split")) {
        return false;
    }
    if (use_mixed_target_split() &&
        !remote_target_shard_.kvflash_sync_identity(committed)) {
        std::fprintf(stderr,
            "[laguna-target-split][kvflash] remote identity sync failed pos=%d\n",
            committed);
        return false;
    }
    return true;
}

void LagunaLayerSplitAdapter::kvflash_sync_history(
        const std::vector<int32_t> & tokens, int base_pos) {
    if (!kvflash_active()) return;
    layer_split_kvflash_sync_history(kvflash_history_, tokens, base_pos);
}

void LagunaLayerSplitAdapter::kvflash_maybe_reselect(int generated) {
    if (!kvflash_active() || kvflash_tau_ <= 0) return;
    if (use_mixed_target_split()) return;
    const int tau = std::max<int>(kvflash_tau_, (int)(kvflash_history_.size() / 45));
    if (generated % tau != 0) return;
    if (!kvflash_scorer_) {
        if (kvflash_drafter_path_.empty() || kvflash_drafter_failed_) return;
        if (!kvflash_drafter_loaded_) {
            for (auto & shard : shards_) ggml_backend_synchronize(shard.backend);
            std::fprintf(stderr,
                "[laguna-target-split][kvflash] loading residency drafter: %s\n",
                kvflash_drafter_path_.c_str());
            if (!load_drafter(kvflash_drafter_path_, /*gpu_layers=*/999,
                              shards_.front().gpu, kvflash_drafter_)) {
                std::fprintf(stderr,
                    "[laguna-target-split][kvflash] drafter load failed (%s); "
                    "staying on LRU residency\n",
                    dflash27b_last_error());
                kvflash_drafter_failed_ = true;
                return;
            }
            kvflash_drafter_loaded_ = true;
        }
        kvflash_scorer_ = std::make_unique<KvFlashCrossTokScorer>(
            &kvflash_drafter_, cfg_.target_path, kvflash_drafter_path_);
        std::fprintf(stderr,
            "[laguna-target-split][kvflash] cross-tokenizer drafter scorer "
            "attached (tau=%d)\n", kvflash_tau_);
    }
    if (!kvflash_scorer_->score_chunks(
            kvflash_history_, kvflash_pager_.chunk_tokens(), kvflash_scores_)) {
        return;
    }
    kvflash_pager_.score_hook = [this](int c) {
        return c < (int)kvflash_scores_.size() ? kvflash_scores_[(size_t)c] : 1e30f;
    };
    const int events = kvflash_pager_.reselect();
    kvflash_pager_.score_hook = nullptr;
    if (events > 0) {
        std::fprintf(stderr,
            "[laguna-target-split][kvflash] reselect @gen=%d: %d page events\n",
            generated, events);
    }
}

void LagunaLayerSplitAdapter::begin_request(const GenerateRequest & req) {
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }
}

void LagunaLayerSplitAdapter::reset_request_state() {
    for (auto & shard : shards_) {
        reset_laguna_target_cache(shard.cache);
    }
    if (kvflash_active()) {
        kvflash_pager_.reset();
        kvflash_history_.clear();
        kvflash_scores_.clear();
        kvflash_pager_.score_hook = nullptr;
    }
    if (use_mixed_target_split() &&
        !remote_target_shard_.reset_request_state()) {
        std::fprintf(stderr,
            "[laguna-target-split] remote shard reset_request_state failed\n");
    }
    prefill_last_logits_.clear();
}

bool LagunaLayerSplitAdapter::run_forward(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<float> * logits_out) {
    if (shards_.empty() || tokens.empty()) return false;
    const LagunaTargetWeights & ref = shards_.front().weights;
    const int hidden = ref.n_embd;
    const int n_tokens_total = (int)tokens.size();
    int ubatch = cfg_.chunk > 0 ? cfg_.chunk : 2048;
    if (const char * e = std::getenv("DFLASH_LAGUNA_LAYER_SPLIT_UBATCH")) {
        ubatch = std::max(1, std::atoi(e));
    }

    if (base_pos < 0 || base_pos + n_tokens_total > cfg_.device.max_ctx) {
        std::fprintf(stderr,
            "[laguna-target-split] range [%d,%d) exceeds max_ctx=%d\n",
            base_pos, base_pos + n_tokens_total, cfg_.device.max_ctx);
        return false;
    }

    ActivationPair acts;
    if (!activation_pair_init(acts, shards_.front().backend, hidden,
                              n_tokens_total, activation_type_)) {
        std::fprintf(stderr, "[laguna-target-split] activation alloc failed gpu=%d\n",
                     shards_.front().gpu);
        return false;
    }

    {
        constexpr int kEmbedBatch = 4096;
        std::vector<float> emb((size_t)hidden * std::min(kEmbedBatch, n_tokens_total));
        for (int i = 0; i < n_tokens_total; i += kEmbedBatch) {
            const int n = std::min(kEmbedBatch, n_tokens_total - i);
            if ((int)emb.size() < hidden * n) emb.resize((size_t)hidden * n);
            if (!ref.embedder.embed(tokens.data() + i, n, emb.data())) {
                activation_pair_free(acts);
                return false;
            }
            const size_t off = (size_t)i * acts.a->nb[1];
            if (!set_activation_tensor_from_f32(
                    acts.a, emb.data(), off, (size_t)hidden * (size_t)n)) {
                std::fprintf(stderr,
                    "[laguna-target-split] unsupported activation type: %s\n",
                    ggml_type_name(acts.a->type));
                activation_pair_free(acts);
                return false;
            }
        }
    }

    ggml_tensor * act_in = acts.a;
    ggml_tensor * act_out = acts.b;
    LagunaLayerSplitShard * current_shard = &shards_.front();
    for (int il = 0; il < ref.n_layer; ++il) {
        LagunaLayerSplitShard * shard = find_layer_split_shard(shards_, il);
        if (!shard) {
            std::fprintf(stderr,
                "[laguna-target-split] missing owner for layer %d\n", il);
            activation_pair_free(acts);
            return false;
        }
        if (shard != current_shard) {
            ActivationPair next_acts;
            if (!activation_pair_init(next_acts, shard->backend, hidden,
                                      n_tokens_total, activation_type_)) {
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_synchronize(current_shard->backend);
            ggml_backend_tensor_copy(act_in, next_acts.a);
            ggml_backend_synchronize(shard->backend);
            activation_pair_free(acts);
            acts = next_acts;
            act_in = acts.a;
            act_out = acts.b;
            current_shard = shard;
        }

        for (int start = 0; start < n_tokens_total;) {
            const int n = std::min(ubatch, n_tokens_total - start);
            const int kv_start = base_pos + start;
            if (kvflash_active() && !kvflash_pager_.alloc_span(kv_start, n)) {
                activation_pair_free(acts);
                return false;
            }
            if (!build_laguna_layer_step(
                    shard->layer_graph, shard->weights, shard->cache,
                    shard->backend, il, act_in, act_out, start, n, kv_start,
                    kvflash_active() ? &kvflash_pager_ : nullptr)) {
                std::fprintf(stderr,
                    "[laguna-target-split] build layer=%d @%d gpu=%d\n",
                    il, start, shard->gpu);
                activation_pair_free(acts);
                return false;
            }

            std::vector<int32_t> pos((size_t)n);
            for (int i = 0; i < n; ++i) pos[(size_t)i] = kv_start + i;
            if (!tensor_ready(shard->layer_graph.positions)) {
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_tensor_set(shard->layer_graph.positions, pos.data(), 0,
                                    sizeof(int32_t) * pos.size());

            if (kvflash_active()) {
                std::vector<int32_t> rows;
                std::vector<float> mfull;
                std::vector<float> mswa;
                const ggml_tensor * mask_ref =
                    shard->layer_graph.attn_mask ? shard->layer_graph.attn_mask
                                                 : shard->layer_graph.attn_mask_swa;
                if (!mask_ref) {
                    activation_pair_free(acts);
                    return false;
                }
                if (!kvflash_fill_rows_and_masks(
                        kvflash_pager_, kv_start, n,
                        (int)mask_ref->ne[0],
                        ref.sliding_window, rows, &mfull, &mswa)) {
                    activation_pair_free(acts);
                    return false;
                }
                if (tensor_ready(shard->layer_graph.kv_idx)) {
                    ggml_backend_tensor_set(shard->layer_graph.kv_idx,
                                            rows.data(), 0,
                                            ggml_nbytes(shard->layer_graph.kv_idx));
                }
                if (tensor_ready(shard->layer_graph.attn_mask)) {
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask,
                                            mfull.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask));
                }
                if (tensor_ready(shard->layer_graph.attn_mask_swa)) {
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask_swa,
                                            mswa.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask_swa));
                }
            } else {
                const int kv_len = kv_start + n;
                std::vector<float> mfull((size_t)kv_len * n, -INFINITY);
                for (int q = 0; q < n; ++q) {
                    const int abs_q = kv_start + q;
                    for (int k = 0; k <= abs_q && k < kv_len; ++k) {
                        mfull[(size_t)q * kv_len + k] = 0.0f;
                    }
                }
                if (tensor_ready(shard->layer_graph.kv_idx)) {
                    ggml_backend_tensor_set(shard->layer_graph.kv_idx,
                                            pos.data(), 0,
                                            ggml_nbytes(shard->layer_graph.kv_idx));
                }
                if (tensor_ready(shard->layer_graph.attn_mask)) {
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask,
                                            mfull.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask));
                }

                std::vector<float> mswa((size_t)kv_len * n, -INFINITY);
                const int W = ref.sliding_window;
                for (int q = 0; q < n; ++q) {
                    const int abs_q = kv_start + q;
                    const int win_lo = std::max(0, abs_q - W + 1);
                    for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                        mswa[(size_t)q * kv_len + k] = 0.0f;
                    }
                }
                if (tensor_ready(shard->layer_graph.attn_mask_swa)) {
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask_swa,
                                            mswa.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask_swa));
                }
            }

            auto st = ggml_backend_graph_compute(shard->backend,
                                                 shard->layer_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr,
                    "[laguna-target-split] compute layer=%d @%d gpu=%d status=%d\n",
                    il, start, shard->gpu, (int)st);
                activation_pair_free(acts);
                return false;
            }
            start += n;
        }
        std::swap(act_in, act_out);
    }

    std::vector<int32_t> argmax;
    LagunaLayerSplitShard & last = shards_.back();
    const bool ok = compute_laguna_split_projection(
        last.backend, last.weights, act_in,
        n_tokens_total - 1, 1, &argmax, logits_out);
    activation_pair_free(acts);
    if (!ok || argmax.empty()) return false;
    last_tok = argmax.back();
    for (auto & shard : shards_) {
        shard.cache.cur_pos = base_pos + n_tokens_total;
        shard.cache.last_tok = last_tok;
    }
    return true;
}

bool LagunaLayerSplitAdapter::run_mixed_forward(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<float> * logits_out) {
    if (!use_mixed_target_split() || tokens.empty()) return false;
    const LagunaTargetWeights & ref = shards_.front().weights;
    const int hidden = ref.n_embd;
    const int n_tokens_total = (int)tokens.size();
    int ubatch = cfg_.chunk > 0 ? cfg_.chunk : 2048;
    if (const char * e = std::getenv("DFLASH_LAGUNA_LAYER_SPLIT_UBATCH")) {
        ubatch = std::max(1, std::atoi(e));
    }
    if (base_pos < 0 || base_pos + n_tokens_total > cfg_.device.max_ctx) {
        std::fprintf(stderr,
            "[laguna-target-split] mixed range [%d,%d) exceeds max_ctx=%d\n",
            base_pos, base_pos + n_tokens_total, cfg_.device.max_ctx);
        return false;
    }

    ActivationPair acts;
    if (!activation_pair_init(acts, shards_.front().backend, hidden,
                              n_tokens_total, activation_type_)) {
        std::fprintf(stderr,
            "[laguna-target-split] mixed activation alloc failed gpu=%d\n",
            shards_.front().gpu);
        return false;
    }

    {
        constexpr int kEmbedBatch = 4096;
        std::vector<float> emb((size_t)hidden * std::min(kEmbedBatch, n_tokens_total));
        for (int i = 0; i < n_tokens_total; i += kEmbedBatch) {
            const int n = std::min(kEmbedBatch, n_tokens_total - i);
            if ((int)emb.size() < hidden * n) emb.resize((size_t)hidden * n);
            if (!ref.embedder.embed(tokens.data() + i, n, emb.data())) {
                activation_pair_free(acts);
                return false;
            }
            const size_t off = (size_t)i * acts.a->nb[1];
            if (!set_activation_tensor_from_f32(
                    acts.a, emb.data(), off, (size_t)hidden * (size_t)n)) {
                activation_pair_free(acts);
                return false;
            }
        }
    }

    ggml_tensor * act_in = acts.a;
    ggml_tensor * act_out = acts.b;
    LagunaLayerSplitShard * current_shard = &shards_.front();
    for (int il = shards_.front().layer_begin; il < shards_.back().layer_end; ++il) {
        LagunaLayerSplitShard * shard = find_layer_split_shard(shards_, il);
        if (!shard) {
            std::fprintf(stderr,
                "[laguna-target-split] mixed missing local owner for layer %d\n",
                il);
            activation_pair_free(acts);
            return false;
        }
        if (shard != current_shard) {
            ActivationPair next_acts;
            if (!activation_pair_init(next_acts, shard->backend, hidden,
                                      n_tokens_total, activation_type_)) {
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_synchronize(current_shard->backend);
            ggml_backend_tensor_copy(act_in, next_acts.a);
            ggml_backend_synchronize(shard->backend);
            activation_pair_free(acts);
            acts = next_acts;
            act_in = acts.a;
            act_out = acts.b;
            current_shard = shard;
        }

        for (int start = 0; start < n_tokens_total;) {
            const int n = std::min(ubatch, n_tokens_total - start);
            const int kv_start = base_pos + start;
            if (kvflash_active() && !kvflash_pager_.alloc_span(kv_start, n)) {
                activation_pair_free(acts);
                return false;
            }
            if (!build_laguna_layer_step(
                    shard->layer_graph, shard->weights, shard->cache,
                    shard->backend, il, act_in, act_out, start, n, kv_start,
                    kvflash_active() ? &kvflash_pager_ : nullptr)) {
                std::fprintf(stderr,
                    "[laguna-target-split] mixed build layer=%d @%d gpu=%d\n",
                    il, start, shard->gpu);
                activation_pair_free(acts);
                return false;
            }

            std::vector<int32_t> pos((size_t)n);
            for (int i = 0; i < n; ++i) pos[(size_t)i] = kv_start + i;
            if (!tensor_ready(shard->layer_graph.positions)) {
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_tensor_set(shard->layer_graph.positions, pos.data(), 0,
                                    sizeof(int32_t) * pos.size());

            if (kvflash_active()) {
                std::vector<int32_t> rows;
                std::vector<float> mfull;
                std::vector<float> mswa;
                const ggml_tensor * mask_ref =
                    shard->layer_graph.attn_mask ? shard->layer_graph.attn_mask
                                                 : shard->layer_graph.attn_mask_swa;
                if (!mask_ref ||
                    !kvflash_fill_rows_and_masks(
                        kvflash_pager_, kv_start, n,
                        (int)mask_ref->ne[0],
                        ref.sliding_window, rows, &mfull, &mswa)) {
                    activation_pair_free(acts);
                    return false;
                }
                if (tensor_ready(shard->layer_graph.kv_idx)) {
                    ggml_backend_tensor_set(shard->layer_graph.kv_idx,
                                            rows.data(), 0,
                                            ggml_nbytes(shard->layer_graph.kv_idx));
                }
                if (tensor_ready(shard->layer_graph.attn_mask)) {
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask,
                                            mfull.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask));
                }
                if (tensor_ready(shard->layer_graph.attn_mask_swa)) {
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask_swa,
                                            mswa.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask_swa));
                }
            } else {
                const int kv_len = kv_start + n;
                std::vector<float> mfull((size_t)kv_len * n, -INFINITY);
                for (int q = 0; q < n; ++q) {
                    const int abs_q = kv_start + q;
                    for (int k = 0; k <= abs_q && k < kv_len; ++k) {
                        mfull[(size_t)q * kv_len + k] = 0.0f;
                    }
                }
                if (tensor_ready(shard->layer_graph.kv_idx)) {
                    ggml_backend_tensor_set(shard->layer_graph.kv_idx,
                                            pos.data(), 0,
                                            ggml_nbytes(shard->layer_graph.kv_idx));
                }
                if (tensor_ready(shard->layer_graph.attn_mask)) {
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask,
                                            mfull.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask));
                }
                std::vector<float> mswa((size_t)kv_len * n, -INFINITY);
                for (int q = 0; q < n; ++q) {
                    const int abs_q = kv_start + q;
                    const int win_lo = std::max(0, abs_q - ref.sliding_window + 1);
                    for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                        mswa[(size_t)q * kv_len + k] = 0.0f;
                    }
                }
                if (tensor_ready(shard->layer_graph.attn_mask_swa)) {
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask_swa,
                                            mswa.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask_swa));
                }
            }

            auto st = ggml_backend_graph_compute(shard->backend,
                                                 shard->layer_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr,
                    "[laguna-target-split] mixed compute layer=%d @%d gpu=%d status=%d\n",
                    il, start, shard->gpu, (int)st);
                activation_pair_free(acts);
                return false;
            }
            start += n;
        }
        std::swap(act_in, act_out);
    }

    std::vector<float> boundary;
    if (!copy_activation_to_host(act_in, shards_.back().backend, 0,
                                 n_tokens_total, hidden, boundary)) {
        activation_pair_free(acts);
        return false;
    }
    activation_pair_free(acts);

    TargetShardForwardRequest req;
    req.base_pos = base_pos;
    req.n_tokens = n_tokens_total;
    req.boundary_activation = &boundary;
    req.ubatch = ubatch;
    req.want_argmax = false;
    req.want_logits = logits_out != nullptr;
    TargetShardForwardResponse resp;
    resp.logits_out = logits_out;
    if (!remote_target_shard_.forward(req, resp)) return false;
    last_tok = resp.last_tok;
    for (auto & shard : shards_) {
        shard.cache.cur_pos = base_pos + n_tokens_total;
        shard.cache.last_tok = last_tok;
    }
    return true;
}

bool LagunaLayerSplitAdapter::prefill(const std::vector<int32_t> & prompt,
                                      int base_pos,
                                      int & last_tok) {
    const bool ok = use_mixed_target_split()
        ? run_mixed_forward(prompt, base_pos, last_tok, &prefill_last_logits_)
        : run_forward(prompt, base_pos, last_tok, &prefill_last_logits_);
    if (ok && kvflash_active()) {
        kvflash_sync_history(prompt, base_pos);
        kvflash_pager_.zero_free_blocks();
    }
    return ok;
}

bool LagunaLayerSplitAdapter::decode_ar(
        int last_tok,
        int committed,
        int n_gen,
        const std::vector<int32_t> & history_prefix,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io) {
    if (n_gen <= 0) return true;
    if (shards_.empty()) return false;

    const auto & w = shards_.front().weights;
    const int vocab = (int)w.embedder.n_vocab;
    const bool ok = run_layer_split_ar_decode(
        last_tok, committed, n_gen, vocab, prefill_last_logits_, sampler_,
        sampler_rng_, history_prefix,
        [&](const std::vector<int32_t> & one, int pos, int & next_tok,
            std::vector<float> * logits_out) {
            return use_mixed_target_split()
                ? run_mixed_forward(one, pos - 1, next_tok, logits_out)
                : run_forward(one, pos - 1, next_tok, logits_out);
        },
        [&](int tok) { return tok == w.eos_id || tok == w.eos_chat_id; },
        out_tokens, io);
    if (ok && kvflash_active()) {
        kvflash_sync_history(out_tokens, committed);
        kvflash_maybe_reselect((int)out_tokens.size());
    }
    return ok;
}

bool LagunaLayerSplitAdapter::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || shards_.empty()) return false;
    if (snapshot_backends_.size() != shards_.size()) return false;
    auto & snap = snapshots_[(size_t)slot];
    const int snap_pos = shards_.front().cache.cur_pos;
    if (snap_pos <= 0) return false;
    if (kvflash_active() &&
        (snap_pos > kvflash_tokens_ || !kvflash_pager_.is_identity())) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                "[laguna-target-split][kvflash] snapshot skipped: pooled "
                "layout needs page-table serialization\n");
            warned = true;
        }
        return false;
    }

    snapshot_free(slot);
    if (snap.shards.size() != shards_.size()) snap.shards.resize(shards_.size());
    for (size_t i = 0; i < shards_.size(); ++i) {
        if (!laguna_snapshot_save(shards_[i].cache, snapshot_backends_[i],
                                  shards_[i].weights.n_layer,
                                  shards_[i].weights.n_head_kv,
                                  shards_[i].weights.head_dim,
                                  snap.shards[i])) {
            snapshot_free(slot);
            return false;
        }
    }
    if (use_mixed_target_split() &&
        !remote_target_shard_.snapshot_save(slot)) {
        snapshot_free(slot);
        return false;
    }
    snap.cur_pos = snap_pos;
    snap.last_tok = shards_.front().cache.last_tok;
    snap.prefill_last_logits = prefill_last_logits_;
    if (kvflash_active() &&
        kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS) {
        layer_split_kvflash_save_history_snapshot(
            kvflash_history_, snap_pos, kvflash_history_snapshots_[(size_t)slot]);
    }
    if (!use_mixed_target_split()) {
        if (!rebuild_disk_snapshot(slot)) {
            snapshot_free(slot);
            return false;
        }
    }
    return true;
}

void LagunaLayerSplitAdapter::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || snapshots_.empty()) return;
    ggml_context * disk_ctx = nullptr;
    ggml_backend_buffer_t disk_buf = nullptr;
    ggml_backend_t disk_backend = nullptr;
    if (disk_snapshot_contexts_.size() == (size_t)PREFIX_SLOTS) {
        disk_ctx = disk_snapshot_contexts_[(size_t)slot];
        disk_buf = disk_snapshot_buffers_[(size_t)slot];
        disk_snapshot_contexts_[(size_t)slot] = nullptr;
        disk_snapshot_buffers_[(size_t)slot] = nullptr;
        if (disk_snapshot_backends_.size() == (size_t)PREFIX_SLOTS) {
            disk_backend = disk_snapshot_backends_[(size_t)slot];
            disk_snapshot_backends_[(size_t)slot] = nullptr;
        }
    }
    auto & snap = snapshots_[(size_t)slot];
    for (auto & ss : snap.shards) {
        if (disk_ctx && ss.ctx == disk_ctx) {
            ss = LagunaCacheSnapshot{};
        } else {
            laguna_snapshot_free(ss);
        }
    }
    if (disk_buf) ggml_backend_buffer_free(disk_buf);
    if (disk_ctx) ggml_free(disk_ctx);
    if (disk_backend) ggml_backend_free(disk_backend);
    snap.cur_pos = 0;
    snap.last_tok = -1;
    snap.prefill_last_logits.clear();
    if (snapshot_prefill_logit_tensors_.size() == (size_t)PREFIX_SLOTS) {
        snapshot_prefill_logit_tensors_[(size_t)slot].clear();
    }
    if (kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS) {
        kvflash_history_snapshots_[(size_t)slot].clear();
    }
    if (snap.shards.size() != shards_.size()) snap.shards.resize(shards_.size());
    if (use_mixed_target_split()) {
        remote_target_shard_.snapshot_free(slot);
    }
}

bool LagunaLayerSplitAdapter::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS ||
        snapshots_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }
    const auto & snap = snapshots_[(size_t)slot];
    if (snap.cur_pos <= 0 || snap.shards.size() != shards_.size()) return false;
    if (snap.prefill_last_logits.empty()) return false;
    for (const auto & ss : snap.shards) {
        if (!ss.used) return false;
    }
    return true;
}

int LagunaLayerSplitAdapter::snapshot_cur_pos(int slot) const {
    return snapshot_used(slot) ? snapshots_[(size_t)slot].cur_pos : 0;
}

bool LagunaLayerSplitAdapter::snapshot_restore(int slot) {
    if (!snapshot_used(slot)) return false;
    auto & snap = snapshots_[(size_t)slot];
    for (size_t i = 0; i < shards_.size(); ++i) {
        if (snap.shards[i].cur_pos != snap.cur_pos) return false;
        if (!laguna_snapshot_restore(snap.shards[i], shards_[i].cache)) {
            return false;
        }
        shards_[i].cache.last_tok = snap.last_tok;
    }
    if (use_mixed_target_split() &&
        !remote_target_shard_.snapshot_restore(slot)) {
        return false;
    }
    prefill_last_logits_ = snap.prefill_last_logits;
    if (kvflash_active()) {
        if (!kvflash_sync_identity(snap.cur_pos)) return false;
        layer_split_kvflash_restore_history(
            kvflash_history_, kvflash_history_snapshots_, slot, snap.cur_pos);
    }
    return true;
}

bool LagunaLayerSplitAdapter::rebuild_disk_snapshot(int slot) {
    if (use_mixed_target_split()) return false;
    if (!snapshot_used(slot) ||
        slot < 0 || slot >= (int)snapshots_.size() ||
        disk_snapshot_contexts_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_buffers_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_backends_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }
    ggml_context * old_ctx = disk_snapshot_contexts_[(size_t)slot];
    ggml_backend_buffer_t old_buf = disk_snapshot_buffers_[(size_t)slot];
    ggml_backend_t old_backend = disk_snapshot_backends_[(size_t)slot];
    disk_snapshot_contexts_[(size_t)slot] = nullptr;
    disk_snapshot_buffers_[(size_t)slot] = nullptr;
    disk_snapshot_backends_[(size_t)slot] = nullptr;
    if (old_buf) ggml_backend_buffer_free(old_buf);
    if (old_ctx) ggml_free(old_ctx);
    if (old_backend) ggml_backend_free(old_backend);

    const auto & snap = snapshots_[(size_t)slot];
    size_t n_tensors = 1;
    for (const auto & shard_snap : snap.shards) {
        if (!shard_snap.used || !shard_snap.ctx) return false;
        for (ggml_tensor * t = ggml_get_first_tensor(shard_snap.ctx); t;
             t = ggml_get_next_tensor(shard_snap.ctx, t)) {
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

    for (size_t shard_idx = 0; shard_idx < snap.shards.size(); ++shard_idx) {
        const auto & shard_snap = snap.shards[shard_idx];
        for (ggml_tensor * src = ggml_get_first_tensor(shard_snap.ctx); src;
             src = ggml_get_next_tensor(shard_snap.ctx, src)) {
            ggml_tensor * dst = ggml_dup_tensor(ctx, src);
            if (!dst) {
                ggml_free(ctx);
                return false;
            }
            const std::string name =
                "laguna_ls" + std::to_string(shard_idx) + "_" + src->name;
            ggml_set_name(dst, name.c_str());
            copies.push_back({src, dst});
        }
    }

    if (snap.prefill_last_logits.empty()) {
        ggml_free(ctx);
        return false;
    }
    ggml_tensor * logits_t =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                           (int64_t)snap.prefill_last_logits.size());
    if (!logits_t) {
        ggml_free(ctx);
        return false;
    }
    ggml_set_name(logits_t, "laguna_snap_prefill_logits");

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
    ggml_backend_tensor_set(logits_t, snap.prefill_last_logits.data(), 0,
                            sizeof(float) * snap.prefill_last_logits.size());

    disk_snapshot_contexts_[(size_t)slot] = ctx;
    disk_snapshot_buffers_[(size_t)slot] = buf;
    disk_snapshot_backends_[(size_t)slot] = cpu;
    return true;
}

ModelBackend::SnapshotRef LagunaLayerSplitAdapter::snapshot_ref(int slot) const {
    ModelBackend::SnapshotRef ref;
    if (!snapshot_used(slot)) return ref;
    if (slot < 0 || slot >= (int)disk_snapshot_contexts_.size()) return ref;
    ref.ctx = disk_snapshot_contexts_[(size_t)slot];
    ref.buf = disk_snapshot_buffers_[(size_t)slot];
    ref.cur_pos = snapshot_cur_pos(slot);
    ref.last_tok = snapshots_[(size_t)slot].last_tok;
    return ref;
}

bool LagunaLayerSplitAdapter::snapshot_adopt(int slot,
                                             ggml_context * ctx,
                                             ggml_backend_buffer_t buf,
                                             int cur_pos,
                                             int32_t last_tok) {
    if (slot < 0 || slot >= PREFIX_SLOTS || !ctx || !buf || cur_pos <= 0 ||
        snapshots_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_contexts_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_buffers_.size() != (size_t)PREFIX_SLOTS ||
        disk_snapshot_backends_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }
    snapshot_free(slot);
    auto & snap = snapshots_[(size_t)slot];
    if (snap.shards.size() != shards_.size()) snap.shards.resize(shards_.size());
    if (snapshot_prefill_logit_tensors_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }
    snapshot_prefill_logit_tensors_[(size_t)slot].clear();

    ggml_tensor * logits_tensor = nullptr;
    for (auto & shard_snap : snap.shards) {
        shard_snap.attn_k.assign(shards_.empty() ? 0 : shards_.front().weights.n_layer, nullptr);
        shard_snap.attn_v.assign(shards_.empty() ? 0 : shards_.front().weights.n_layer, nullptr);
        shard_snap.ctx = ctx;
        shard_snap.buf = buf;
        shard_snap.cur_pos = cur_pos;
        shard_snap.used = true;
    }

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->name[0]) continue;
        if (std::strcmp(t->name, "laguna_snap_prefill_logits") == 0) {
            logits_tensor = t;
            continue;
        }
        int shard_idx = -1;
        int layer_idx = -1;
        if (std::sscanf(t->name, "laguna_ls%d_snap_k_l%d", &shard_idx, &layer_idx) == 2 &&
            shard_idx >= 0 && shard_idx < (int)snap.shards.size() &&
            layer_idx >= 0 && layer_idx < (int)snap.shards[(size_t)shard_idx].attn_k.size()) {
            snap.shards[(size_t)shard_idx].attn_k[(size_t)layer_idx] = t;
        } else if (std::sscanf(t->name, "laguna_ls%d_snap_v_l%d", &shard_idx, &layer_idx) == 2 &&
                   shard_idx >= 0 && shard_idx < (int)snap.shards.size() &&
                   layer_idx >= 0 && layer_idx < (int)snap.shards[(size_t)shard_idx].attn_v.size()) {
            snap.shards[(size_t)shard_idx].attn_v[(size_t)layer_idx] = t;
        }
    }

    if (!logits_tensor) {
        snapshot_free(slot);
        return false;
    }
    const size_t logits_n = ggml_nelements(logits_tensor);
    snap.prefill_last_logits.assign(logits_n, 0.0f);
    ggml_backend_tensor_get(logits_tensor, snap.prefill_last_logits.data(), 0,
                            sizeof(float) * logits_n);
    snapshot_prefill_logit_tensors_[(size_t)slot].push_back(logits_tensor);

    for (size_t shard_idx = 0; shard_idx < shards_.size(); ++shard_idx) {
        auto & shard_snap = snap.shards[shard_idx];
        if (shard_snap.attn_k.size() != shards_[shard_idx].cache.attn_k.size() ||
            shard_snap.attn_v.size() != shards_[shard_idx].cache.attn_v.size()) {
            snapshot_free(slot);
            return false;
        }
        for (size_t i = 0; i < shard_snap.attn_k.size(); ++i) {
            const bool cache_has_kv =
                shards_[shard_idx].cache.attn_k[i] || shards_[shard_idx].cache.attn_v[i];
            if (cache_has_kv && (!shard_snap.attn_k[i] || !shard_snap.attn_v[i])) {
                snapshot_free(slot);
                return false;
            }
        }
    }

    snap.cur_pos = cur_pos;
    snap.last_tok = last_tok;
    disk_snapshot_contexts_[(size_t)slot] = ctx;
    disk_snapshot_buffers_[(size_t)slot] = buf;
    disk_snapshot_backends_[(size_t)slot] = nullptr;
    return true;
}

int LagunaLayerSplitAdapter::current_last_token() const {
    if (shards_.empty()) return -1;
    return shards_.front().cache.last_tok;
}

void LagunaLayerSplitAdapter::shutdown() {
    kvflash_scorer_.reset();
    if (kvflash_drafter_loaded_) {
        dflash::common::free_drafter(kvflash_drafter_);
        kvflash_drafter_loaded_ = false;
    }
    for (int i = 0; i < PREFIX_SLOTS; ++i) snapshot_free(i);
    kvflash_history_snapshots_.clear();
    auto shard_metas = layer_split_shard_metas(shards_);
    free_layer_split_snapshot_backends(shard_metas, snapshot_backends_);
    remote_target_shard_.close();
    free_laguna_layer_split_shards(shards_);
}

void free_laguna_layer_split_shards(
        std::vector<LagunaLayerSplitShard> & shards) {
    for (auto & shard : shards) {
        laguna_layer_step_graph_destroy(shard.layer_graph);
        free_laguna_target_cache(shard.cache);
        free_laguna_target_weights(shard.weights);
        if (shard.backend) {
            ggml_backend_free(shard.backend);
            shard.backend = nullptr;
        }
    }
    shards.clear();
}

int run_laguna_target_shard_ipc_daemon(const char * target_path,
                                       const std::vector<int> & gpus,
                                       const std::vector<int> & layer_begins,
                                       const std::vector<int> & layer_ends,
                                       int max_ctx,
                                       int stream_fd,
                                       int payload_fd,
                                       int shared_payload_fd,
                                       size_t shared_payload_bytes,
                                       int kvflash_pool_tokens) {
#if defined(_WIN32)
    (void)target_path; (void)gpus; (void)layer_begins; (void)layer_ends;
    (void)max_ctx; (void)stream_fd; (void)payload_fd; (void)shared_payload_fd;
    (void)shared_payload_bytes; (void)kvflash_pool_tokens;
    std::fprintf(stderr, "Laguna target shard IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    if (!target_path || gpus.empty() || gpus.size() != layer_begins.size() ||
        gpus.size() != layer_ends.size() || max_ctx <= 0 || stream_fd < 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=laguna-target-shard "
            "<target.gguf> --target-gpus=N[,N...] "
            "--layer-begins=N[,N...] --layer-ends=N[,N...] "
            "--max-ctx=N --stream-fd=FD [--payload-fd=FD]\n");
        return 2;
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (gpus[i] < 0 || layer_begins[i] < 0 ||
            layer_ends[i] <= layer_begins[i] ||
            (i > 0 && layer_begins[i] != layer_ends[i - 1])) {
            std::fprintf(stderr,
                "[laguna-target-shard-daemon] bad shard config\n");
            return 2;
        }
    }

    std::vector<LagunaLayerSplitShard> shards(gpus.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        auto & shard = shards[i];
        shard.gpu = gpus[i];
        shard.layer_begin = layer_begins[i];
        shard.layer_end = layer_ends[i];
        shard.backend = ggml_backend_cuda_init(shard.gpu);
        if (!shard.backend) {
            std::fprintf(stderr,
                "[laguna-target-shard-daemon] backend init failed gpu=%d\n",
                shard.gpu);
            free_laguna_layer_split_shards(shards);
            return 1;
        }
    }
    for (size_t i = 0; i < shards.size(); ++i) {
        auto & shard = shards[i];
        const bool is_last = i + 1 == shards.size();
        const TargetLoadPlan plan =
            make_layer_split_load_plan<TargetLoadPlan>(shard, is_last);
        if (!load_target_gguf_laguna_partial(
                target_path, shard.backend, plan, shard.weights) ||
            !create_laguna_target_cache_partial(
                shard.weights, max_ctx, shard.backend,
                shard.layer_begin, shard.layer_end, shard.cache,
                kvflash_pool_tokens)) {
            std::fprintf(stderr,
                "[laguna-target-shard-daemon] load/cache failed gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
            free_laguna_layer_split_shards(shards);
            return 1;
        }
    }

    std::vector<ggml_backend_t> snapshot_backends;
    if (!init_layer_split_snapshot_backends(
            layer_split_shard_metas(shards), snapshot_backends,
            "laguna-target-shard-daemon")) {
        free_laguna_layer_split_shards(shards);
        return 1;
    }
    std::vector<LagunaLayerSplitSnapshot> snapshots(
        (size_t)ModelBackend::kMaxSlots);
    for (auto & slot : snapshots) slot.shards.resize(shards.size());
    auto free_slot = [&](int slot) {
        if (slot < 0 || slot >= ModelBackend::kMaxSlots) return;
        auto & snap = snapshots[(size_t)slot];
        for (auto & ss : snap.shards) laguna_snapshot_free(ss);
        snap.cur_pos = 0;
        snap.last_tok = -1;
        snap.prefill_last_logits.clear();
        if (snap.shards.size() != shards.size()) snap.shards.resize(shards.size());
    };

    KvFlashPager kvflash_pager;
    if (kvflash_pool_tokens > 0) {
        std::vector<ggml_tensor *> all_k;
        std::vector<ggml_tensor *> all_v;
        const int n_layer = shards.front().weights.n_layer;
        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * k = nullptr;
            ggml_tensor * v = nullptr;
            for (auto & shard : shards) {
                if (il < (int)shard.cache.attn_k.size() &&
                    shard.cache.attn_k[(size_t)il]) {
                    k = shard.cache.attn_k[(size_t)il];
                    v = shard.cache.attn_v[(size_t)il];
                    break;
                }
            }
            if (k && v) {
                all_k.push_back(k);
                all_v.push_back(v);
            }
        }
        KvFlashConfig pc;
        pc.pool_tokens = kvflash_pool_tokens;
        if (!all_k.empty()) {
            pc.tail_window_chunks =
                std::max(4, (shards.front().weights.sliding_window +
                             pc.chunk_tokens - 1) / pc.chunk_tokens + 1);
        }
        if (!kvflash_pager.attach(pc, all_k, all_v)) {
            auto metas = layer_split_shard_metas(shards);
            free_layer_split_snapshot_backends(metas, snapshot_backends);
            free_laguna_layer_split_shards(shards);
            return 1;
        }
    }

    const int hidden = shards.front().weights.n_embd;
    std::vector<float> prefill_last_logits;
    TargetShardDaemonCallbacks callbacks;
    callbacks.log_prefix = "laguna-target-shard-daemon";
    callbacks.forward = [&](const TargetShardDaemonForwardRequest & req,
                            TargetShardDaemonForwardResponse & resp) -> bool {
        if (!req.boundary_activation || req.n_tokens <= 0 ||
            (int)req.boundary_activation->size() != hidden * req.n_tokens ||
            req.base_pos < 0 || req.base_pos + req.n_tokens > max_ctx) {
            return false;
        }
        const int n_tokens = req.n_tokens;
        int ubatch = std::max(1, req.ubatch > 0 ? req.ubatch : n_tokens);
        ActivationPair acts;
        if (!activation_pair_init(acts, shards.front().backend, hidden, n_tokens)) {
            return false;
        }
        ggml_backend_tensor_set(acts.a, req.boundary_activation->data(), 0,
                                sizeof(float) * req.boundary_activation->size());
        ggml_tensor * act_in = acts.a;
        ggml_tensor * act_out = acts.b;
        LagunaLayerSplitShard * current_shard = &shards.front();
        bool ok = true;
        for (int il = shards.front().layer_begin;
             ok && il < shards.back().layer_end; ++il) {
            LagunaLayerSplitShard * shard = find_layer_split_shard(shards, il);
            if (!shard) {
                ok = false;
                break;
            }
            if (shard != current_shard) {
                ActivationPair next_acts;
                if (!activation_pair_init(next_acts, shard->backend,
                                          hidden, n_tokens)) {
                    ok = false;
                    break;
                }
                ggml_backend_synchronize(current_shard->backend);
                ggml_backend_tensor_copy(act_in, next_acts.a);
                ggml_backend_synchronize(shard->backend);
                activation_pair_free(acts);
                acts = next_acts;
                act_in = acts.a;
                act_out = acts.b;
                current_shard = shard;
            }
            for (int start = 0; ok && start < n_tokens;) {
                const int n = std::min(ubatch, n_tokens - start);
                const int kv_start = req.base_pos + start;
                if (kvflash_pool_tokens > 0 &&
                    !kvflash_pager.alloc_span(kv_start, n)) {
                    ok = false;
                    break;
                }
                if (!build_laguna_layer_step(
                        shard->layer_graph, shard->weights, shard->cache,
                        shard->backend, il, act_in, act_out, start, n,
                        kv_start,
                        kvflash_pool_tokens > 0 ? &kvflash_pager : nullptr)) {
                    ok = false;
                    break;
                }
                std::vector<int32_t> pos((size_t)n);
                for (int i = 0; i < n; ++i) pos[(size_t)i] = kv_start + i;
                if (!tensor_ready(shard->layer_graph.positions)) {
                    ok = false;
                    break;
                }
                ggml_backend_tensor_set(shard->layer_graph.positions,
                                        pos.data(), 0,
                                        sizeof(int32_t) * pos.size());
                if (kvflash_pool_tokens > 0) {
                    std::vector<int32_t> rows;
                    std::vector<float> mfull;
                    std::vector<float> mswa;
                    const ggml_tensor * mask_ref =
                        shard->layer_graph.attn_mask
                            ? shard->layer_graph.attn_mask
                            : shard->layer_graph.attn_mask_swa;
                    if (!mask_ref ||
                        !kvflash_fill_rows_and_masks(
                            kvflash_pager, kv_start, n, (int)mask_ref->ne[0],
                            shard->weights.sliding_window,
                            rows, &mfull, &mswa)) {
                        ok = false;
                        break;
                    }
                    if (tensor_ready(shard->layer_graph.kv_idx)) {
                        ggml_backend_tensor_set(
                            shard->layer_graph.kv_idx, rows.data(), 0,
                            ggml_nbytes(shard->layer_graph.kv_idx));
                    }
                    if (tensor_ready(shard->layer_graph.attn_mask)) {
                        ggml_backend_tensor_set(
                            shard->layer_graph.attn_mask, mfull.data(), 0,
                            ggml_nbytes(shard->layer_graph.attn_mask));
                    }
                    if (tensor_ready(shard->layer_graph.attn_mask_swa)) {
                        ggml_backend_tensor_set(
                            shard->layer_graph.attn_mask_swa, mswa.data(), 0,
                            ggml_nbytes(shard->layer_graph.attn_mask_swa));
                    }
                } else {
                    const int kv_len = kv_start + n;
                    std::vector<float> mfull((size_t)kv_len * n, -INFINITY);
                    for (int q = 0; q < n; ++q) {
                        const int abs_q = kv_start + q;
                        for (int k = 0; k <= abs_q && k < kv_len; ++k) {
                            mfull[(size_t)q * kv_len + k] = 0.0f;
                        }
                    }
                    if (tensor_ready(shard->layer_graph.kv_idx)) {
                        ggml_backend_tensor_set(
                            shard->layer_graph.kv_idx, pos.data(), 0,
                            ggml_nbytes(shard->layer_graph.kv_idx));
                    }
                    if (tensor_ready(shard->layer_graph.attn_mask)) {
                        ggml_backend_tensor_set(
                            shard->layer_graph.attn_mask, mfull.data(), 0,
                            ggml_nbytes(shard->layer_graph.attn_mask));
                    }
                    std::vector<float> mswa((size_t)kv_len * n, -INFINITY);
                    for (int q = 0; q < n; ++q) {
                        const int abs_q = kv_start + q;
                        const int win_lo = std::max(
                            0, abs_q - shard->weights.sliding_window + 1);
                        for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                            mswa[(size_t)q * kv_len + k] = 0.0f;
                        }
                    }
                    if (tensor_ready(shard->layer_graph.attn_mask_swa)) {
                        ggml_backend_tensor_set(
                            shard->layer_graph.attn_mask_swa, mswa.data(), 0,
                            ggml_nbytes(shard->layer_graph.attn_mask_swa));
                    }
                }
                auto st = ggml_backend_graph_compute(shard->backend,
                                                     shard->layer_graph.gf);
                ok = st == GGML_STATUS_SUCCESS;
                start += n;
            }
            std::swap(act_in, act_out);
        }
        if (ok) {
            std::vector<int32_t> argmax;
            LagunaLayerSplitShard & last = shards.back();
            ok = compute_laguna_split_projection(
                last.backend, last.weights, act_in,
                req.want_argmax ? 0 : n_tokens - 1,
                req.want_argmax ? n_tokens : 1,
                &argmax, req.want_logits ? &resp.logits : nullptr);
            if (ok && !argmax.empty()) {
                resp.last_tok = argmax.back();
                if (req.want_argmax) resp.argmax = std::move(argmax);
                if (req.want_logits) prefill_last_logits = resp.logits;
            } else {
                ok = false;
            }
        }
        activation_pair_free(acts);
        if (!ok) return false;
        for (auto & shard : shards) {
            shard.cache.cur_pos = req.base_pos + n_tokens;
            shard.cache.last_tok = resp.last_tok;
        }
        return true;
    };
    callbacks.reset_request_state = [&]() {
        for (auto & shard : shards) reset_laguna_target_cache(shard.cache);
        if (kvflash_pool_tokens > 0) kvflash_pager.reset();
        prefill_last_logits.clear();
        return true;
    };
    callbacks.kvflash_sync_identity = [&](int committed) {
        if (kvflash_pool_tokens <= 0) return true;
        return layer_split_kvflash_sync_identity(
            kvflash_pager, committed, kvflash_pool_tokens,
            "laguna-target-shard-daemon");
    };
    callbacks.snapshot_save = [&](int slot) {
        if (slot < 0 || slot >= ModelBackend::kMaxSlots ||
            snapshot_backends.size() != shards.size()) {
            return false;
        }
        free_slot(slot);
        auto & snap = snapshots[(size_t)slot];
        if (snap.shards.size() != shards.size()) snap.shards.resize(shards.size());
        for (size_t i = 0; i < shards.size(); ++i) {
            if (!laguna_snapshot_save(
                    shards[i].cache, snapshot_backends[i],
                    shards[i].weights.n_layer, shards[i].weights.n_head_kv,
                    shards[i].weights.head_dim, snap.shards[i])) {
                free_slot(slot);
                return false;
            }
        }
        snap.cur_pos = shards.front().cache.cur_pos;
        snap.last_tok = shards.front().cache.last_tok;
        snap.prefill_last_logits = prefill_last_logits;
        return true;
    };
    callbacks.snapshot_free = [&](int slot) { free_slot(slot); };
    callbacks.snapshot_restore = [&](int slot) {
        if (slot < 0 || slot >= ModelBackend::kMaxSlots) return false;
        auto & snap = snapshots[(size_t)slot];
        if (snap.cur_pos <= 0 || snap.shards.size() != shards.size()) return false;
        for (size_t i = 0; i < shards.size(); ++i) {
            if (snap.shards[i].cur_pos != snap.cur_pos ||
                !laguna_snapshot_restore(snap.shards[i], shards[i].cache)) {
                return false;
            }
            shards[i].cache.last_tok = snap.last_tok;
        }
        prefill_last_logits = snap.prefill_last_logits;
        if (kvflash_pool_tokens > 0) {
            return layer_split_kvflash_sync_identity(
                kvflash_pager, snap.cur_pos, kvflash_pool_tokens,
                "laguna-target-shard-daemon");
        }
        return true;
    };

    std::fprintf(stderr,
        "[laguna-target-shard-daemon] ready shards=%zu layers=[%d,%d)\n",
        shards.size(), shards.front().layer_begin, shards.back().layer_end);
    const int rc = run_target_shard_ipc_daemon_loop(
        hidden, (int)shards.front().weights.embedder.n_vocab,
        stream_fd, payload_fd, shared_payload_fd,
        shared_payload_bytes, std::move(callbacks));
    for (int slot = 0; slot < ModelBackend::kMaxSlots; ++slot) free_slot(slot);
    auto metas = layer_split_shard_metas(shards);
    free_layer_split_snapshot_backends(metas, snapshot_backends);
    free_laguna_layer_split_shards(shards);
    return rc;
#endif
}

}  // namespace dflash::common
