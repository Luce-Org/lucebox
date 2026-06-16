// Gemma4 layer-split adapter.

#include "gemma4_layer_split_adapter.h"

#include "common/backend_precision.h"
#include "common/dflash_layer_split_runtime.h"
#include "common/gguf_inspect.h"
#include "common/layer_split_utils.h"
#include "common/layer_split_runtime.h"
#include "dflash27b.h"
#include "qwen3/qwen3_kvflash_scorer.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace dflash::common {

namespace {

static bool tensor_ready(const ggml_tensor * t) {
    return t && t->buffer;
}

static bool gemma4_align_split_for_kv_sharing(
        const char * target_path,
        std::vector<Gemma4LayerSplitShard> & shards) {
    if (shards.size() <= 1 || !target_path) return true;

    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(target_path, gip);
    if (!gctx) return true;

    int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id < 0 || std::string(gguf_get_val_str(gctx, arch_id)) != "gemma4") {
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return true;
    }

    auto get_u32 = [&](const char * key, uint32_t def) {
        int64_t id = gguf_find_key(gctx, key);
        if (id < 0) return def;
        if (gguf_get_kv_type(gctx, id) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_n(gctx, id) == 0) return def;
            return ((const uint32_t *)gguf_get_arr_data(gctx, id))[0];
        }
        return gguf_get_val_u32(gctx, id);
    };

    const int n_layer = (int)get_u32("gemma4.block_count", 0);
    const int shared_kv = (int)get_u32("gemma4.attention.shared_kv_layers", 0);
    gguf_free(gctx);
    if (meta_ctx) ggml_free(meta_ctx);
    if (n_layer <= 0 || shared_kv <= 0 || shared_kv >= n_layer) return true;

    const int kv_sharing_start = n_layer - shared_kv;
    const int kv_source_layer = kv_sharing_start - 1;
    if (kv_source_layer <= 0) return true;
    for (size_t i = 0; i + 1 < shards.size(); ++i) {
        if (shards[i].layer_begin < kv_source_layer &&
            shards[i].layer_end > kv_source_layer) {
            shards[i].layer_end = kv_source_layer;
            shards[i + 1].layer_begin = kv_source_layer;
            break;
        }
    }
    for (const auto & shard : shards) {
        if (shard.layer_begin >= shard.layer_end) {
            std::fprintf(stderr,
                "[gemma4-target-split] KV-sharing boundary alignment produced "
                "empty shard [%d,%d); adjust --target-layer-split\n",
                shard.layer_begin, shard.layer_end);
            return false;
        }
    }
    return true;
}

}  // namespace

Gemma4LayerSplitAdapter::Gemma4LayerSplitAdapter(
        const Gemma4LayerSplitAdapterConfig & cfg)
    : cfg_(cfg) {}

Gemma4LayerSplitAdapter::~Gemma4LayerSplitAdapter() noexcept {
    try {
        shutdown();
    } catch (...) {
        // Destructors must not depend on newer libstdc++ termination helpers.
    }
}

bool Gemma4LayerSplitAdapter::init() {
    const LayerSplitRuntimeInit runtime_cfg{
        cfg_.target_path,
        &cfg_.device,
        "gemma4-target-split",
    };
    if (!init_layer_split_runtime(runtime_cfg, shards_, snapshot_backends_)) {
        return false;
    }
    if (!gemma4_align_split_for_kv_sharing(cfg_.target_path, shards_)) {
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
    std::fprintf(stderr, "[gemma4-target-split] activation=%s (%s",
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
            make_layer_split_load_plan<TargetLoadPlan>(
                shard, i + 1 == shards_.size());
        if (!load_gemma4_gguf_partial(cfg_.target_path, shard.backend,
                                      plan, shard.weights)) {
            std::fprintf(stderr,
                "[gemma4-target-split] load gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
            return false;
        }
    }

    kvflash_read_config();
    if (kvflash_active() && cfg_.fa_window > 0) {
        std::fprintf(stderr,
            "[gemma4-target-split][kvflash] fa_window is incompatible with "
            "pooled full-attention slots\n");
        return false;
    }

    for (auto & shard : shards_) {
        if (!create_gemma4_cache_partial(shard.backend, shard.weights,
                                         cfg_.device.max_ctx,
                                         shard.layer_begin, shard.layer_end,
                                         shard.cache, kvflash_tokens_)) {
            std::fprintf(stderr,
                "[gemma4-target-split] cache gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
            return false;
        }
        shard.cache.fa_window = cfg_.fa_window;
        std::fprintf(stderr, "[gemma4-target-split] gpu=%d layers=[%d,%d)\n",
                     shard.gpu, shard.layer_begin, shard.layer_end);
    }
    if (!kvflash_attach()) return false;

    snapshots_.resize(PREFIX_SLOTS);
    for (auto & slot : snapshots_) {
        slot.shards.resize(shards_.size());
    }
    kvflash_history_snapshots_.resize(PREFIX_SLOTS);

    return true;
}

void Gemma4LayerSplitAdapter::kvflash_read_config() {
    if (!std::getenv("DFLASH_KVFLASH") || shards_.empty()) return;
    kvflash_drafter_path_ = kvflash_find_drafter(cfg_.target_path);

    int64_t min_free = std::numeric_limits<int64_t>::max();
    int64_t max_bytes_per_token = 0;
    for (const auto & shard : shards_) {
        size_t gpu_free = 0, gpu_total = 0;
        if (ggml_backend_dev_t dev = ggml_backend_get_device(shard.backend)) {
            ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
        }
        min_free = std::min<int64_t>(min_free, (int64_t)gpu_free);

        int64_t bpt = 0;
        for (int il = shard.layer_begin; il < shard.layer_end; ++il) {
            if (!gemma4_has_kv(shard.weights, il) ||
                gemma4_is_swa_layer(shard.weights, il)) {
                continue;
            }
            bpt += (int64_t)gemma4_n_head_kv(shard.weights, il) * 2 *
                   (int64_t)ggml_row_size(GGML_TYPE_F16,
                                          gemma4_head_dim(shard.weights, il));
        }
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

bool Gemma4LayerSplitAdapter::kvflash_attach() {
    if (!kvflash_active()) return true;
    std::vector<ggml_tensor *> full_k;
    std::vector<ggml_tensor *> full_v;
    const int n_layer = shards_.empty() ? 0 : shards_.front().weights.n_layer;
    for (int il = 0; il < n_layer; ++il) {
        if (!gemma4_has_kv(shards_.front().weights, il) ||
            gemma4_is_swa_layer(shards_.front().weights, il)) {
            continue;
        }
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
        for (auto & shard : shards_) {
            if (il < (int)shard.cache.k.size() && shard.cache.k[(size_t)il]) {
                k = shard.cache.k[(size_t)il];
                v = shard.cache.v[(size_t)il];
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
            "[gemma4-target-split][kvflash] pager attach failed pool=%d layers=%zu\n",
            kvflash_tokens_, full_k.size());
        return false;
    }
    std::printf("[gemma4-target-split][kvflash] resident pool %d tokens over "
                "%zu full-attn layers (logical max_ctx %d), tau=%d, policy=%s\n",
                kvflash_tokens_, full_k.size(), cfg_.device.max_ctx,
                kvflash_tau_,
                !kvflash_drafter_path_.empty()
                    ? "drafter/cross-tok (attaches on first reselect)"
                    : "lru (recency-only: no Qwen3-0.6B drafter found)");
    std::fflush(stdout);
    return true;
}

bool Gemma4LayerSplitAdapter::kvflash_sync_identity(int committed) {
    if (!kvflash_active()) return true;
    if (committed > kvflash_tokens_) {
        std::fprintf(stderr,
            "[gemma4-target-split][kvflash] prefix (%d) exceeds resident pool %d\n",
            committed, kvflash_tokens_);
        return false;
    }
    kvflash_pager_.reset();
    if (!kvflash_pager_.alloc_span(0, committed)) return false;
    kvflash_pager_.zero_free_blocks();
    return true;
}

void Gemma4LayerSplitAdapter::kvflash_sync_history(
        const std::vector<int32_t> & tokens, int base_pos) {
    if (!kvflash_active()) return;
    if (base_pos == 0) {
        kvflash_history_.assign(tokens.begin(), tokens.end());
        return;
    }
    if ((int)kvflash_history_.size() > base_pos) {
        kvflash_history_.resize((size_t)base_pos);
    } else if ((int)kvflash_history_.size() < base_pos) {
        kvflash_history_.resize((size_t)base_pos, 0);
    }
    kvflash_history_.insert(kvflash_history_.end(), tokens.begin(), tokens.end());
}

void Gemma4LayerSplitAdapter::kvflash_maybe_reselect(int generated) {
    if (!kvflash_active() || kvflash_tau_ <= 0) return;
    const int tau = std::max<int>(kvflash_tau_, (int)(kvflash_history_.size() / 45));
    if (generated % tau != 0) return;
    if (!kvflash_scorer_) {
        if (kvflash_drafter_path_.empty() || kvflash_drafter_failed_) return;
        if (!kvflash_drafter_loaded_) {
            for (auto & shard : shards_) ggml_backend_synchronize(shard.backend);
            std::fprintf(stderr,
                "[gemma4-target-split][kvflash] loading residency drafter: %s\n",
                kvflash_drafter_path_.c_str());
            if (!load_drafter(kvflash_drafter_path_, /*gpu_layers=*/999,
                              shards_.front().gpu, kvflash_drafter_)) {
                std::fprintf(stderr,
                    "[gemma4-target-split][kvflash] drafter load failed (%s); "
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
            "[gemma4-target-split][kvflash] cross-tokenizer drafter scorer "
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
            "[gemma4-target-split][kvflash] reselect @gen=%d: %d page events\n",
            generated, events);
    }
}

void Gemma4LayerSplitAdapter::begin_request(const GenerateRequest & req) {
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }
}

void Gemma4LayerSplitAdapter::reset_request_state() {
    for (auto & shard : shards_) {
        shard.cache.cur_pos = 0;
        shard.cache.last_tok = -1;
    }
    if (kvflash_active()) {
        kvflash_pager_.reset();
        kvflash_history_.clear();
        kvflash_scores_.clear();
        kvflash_pager_.score_hook = nullptr;
    }
    prefill_last_logits_.clear();
}

bool Gemma4LayerSplitAdapter::run_forward(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<float> * logits_out) {
    if (shards_.empty() || tokens.empty()) return false;
    const Gemma4Weights & ref = shards_.front().weights;
    const int hidden = ref.n_embd;
    const int n_tokens_total = (int)tokens.size();
    int ubatch = cfg_.chunk > 0 ? cfg_.chunk : 512;
    if (const char * e = std::getenv("DFLASH_GEMMA4_LAYER_SPLIT_UBATCH")) {
        ubatch = std::max(1, std::atoi(e));
    }

    if (base_pos < 0 || base_pos + n_tokens_total > cfg_.device.max_ctx) {
        std::fprintf(stderr,
            "[gemma4-target-split] range [%d,%d) exceeds max_ctx=%d\n",
            base_pos, base_pos + n_tokens_total, cfg_.device.max_ctx);
        return false;
    }

    ActivationPair acts;
    if (!activation_pair_init(acts, shards_.front().backend, hidden,
                              n_tokens_total, activation_type_)) {
        std::fprintf(stderr, "[gemma4-target-split] activation alloc failed gpu=%d\n",
                     shards_.front().gpu);
        return false;
    }
    ActivationBuffer orig;
    if (!activation_buffer_init(orig, shards_.front().backend, hidden,
                                n_tokens_total, activation_type_,
                                "gemma4_target_split_orig_embed")) {
        activation_pair_free(acts);
        return false;
    }

    {
        constexpr int kEmbedBatch = 4096;
        std::vector<float> emb((size_t)hidden * std::min(kEmbedBatch, n_tokens_total));
        const float scale = std::sqrt((float)hidden);
        for (int i = 0; i < n_tokens_total; i += kEmbedBatch) {
            const int n = std::min(kEmbedBatch, n_tokens_total - i);
            if ((int)emb.size() < hidden * n) emb.resize((size_t)hidden * n);
            if (!ref.embedder.embed(tokens.data() + i, n, emb.data())) {
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }
            for (int j = 0; j < hidden * n; ++j) emb[(size_t)j] *= scale;
            const size_t off = (size_t)i * acts.a->nb[1];
            const size_t elems = (size_t)hidden * (size_t)n;
            if (!set_activation_tensor_from_f32(acts.a, emb.data(), off, elems) ||
                !set_activation_tensor_from_f32(orig.tensor, emb.data(), off, elems)) {
                std::fprintf(stderr,
                    "[gemma4-target-split] unsupported activation type: %s\n",
                    ggml_type_name(acts.a->type));
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }
        }
    }

    ggml_tensor * act_in = acts.a;
    ggml_tensor * act_out = acts.b;
    Gemma4LayerSplitShard * current_shard = &shards_.front();
    for (int il = 0; il < ref.n_layer; ++il) {
        Gemma4LayerSplitShard * shard = find_layer_split_shard(shards_, il);
        if (!shard) {
            std::fprintf(stderr,
                "[gemma4-target-split] missing owner for layer %d\n", il);
            activation_buffer_free(orig);
            activation_pair_free(acts);
            return false;
        }
        if (shard != current_shard) {
            ActivationPair next_acts;
            if (!activation_pair_init(next_acts, shard->backend, hidden,
                                      n_tokens_total, activation_type_)) {
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }
            ActivationBuffer next_orig;
            if (!activation_buffer_init(next_orig, shard->backend, hidden,
                                        n_tokens_total, activation_type_,
                                        "gemma4_target_split_orig_embed")) {
                activation_pair_free(next_acts);
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_synchronize(current_shard->backend);
            ggml_backend_tensor_copy(act_in, next_acts.a);
            ggml_backend_tensor_copy(orig.tensor, next_orig.tensor);
            ggml_backend_synchronize(shard->backend);
            activation_pair_free(acts);
            activation_buffer_free(orig);
            acts = next_acts;
            orig = next_orig;
            act_in = acts.a;
            act_out = acts.b;
            current_shard = shard;
        }

        for (int start = 0; start < n_tokens_total;) {
            int n = std::min(ubatch, n_tokens_total - start);
            const int kv_start = base_pos + start;
            if (shard->cache.swa_size > 0 &&
                shard->cache.swa_size < shard->cache.max_ctx) {
                const int swa_remaining =
                    shard->cache.swa_size - (kv_start % shard->cache.swa_size);
                n = std::min(n, swa_remaining);
            }
            const bool use_kvflash =
                kvflash_active() && !gemma4_is_swa_layer(ref, il);
            if (use_kvflash && !kvflash_pager_.alloc_span(kv_start, n)) {
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }
            if (!build_gemma4_layer_step(
                    shard->layer_graph, shard->weights, shard->cache,
                    shard->backend, il, act_in, orig.tensor, act_out,
                    start, n, kv_start,
                    use_kvflash ? &kvflash_pager_ : nullptr)) {
                std::fprintf(stderr,
                    "[gemma4-target-split] build layer=%d @%d gpu=%d\n",
                    il, start, shard->gpu);
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }

            std::vector<int32_t> pos((size_t)n);
            for (int i = 0; i < n; ++i) pos[(size_t)i] = kv_start + i;
            if (!tensor_ready(shard->layer_graph.positions)) {
                std::fprintf(stderr,
                    "[gemma4-target-split] positions input not allocated layer=%d gpu=%d\n",
                    il, shard->gpu);
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_tensor_set(shard->layer_graph.positions, pos.data(), 0,
                                    sizeof(int32_t) * pos.size());
            if (tensor_ready(shard->layer_graph.token_ids)) {
                ggml_backend_tensor_set(shard->layer_graph.token_ids,
                                        tokens.data() + start, 0,
                                        sizeof(int32_t) * (size_t)n);
            }

            if (tensor_ready(shard->layer_graph.kv_idx_full)) {
                if (use_kvflash) {
                    std::vector<int32_t> rows;
                    std::vector<float> mfull;
                    if (!kvflash_fill_rows_and_masks(
                            kvflash_pager_, kv_start, n,
                            (int)shard->layer_graph.attn_mask_full->ne[0],
                            /*swa_window=*/0, rows, &mfull, nullptr)) {
                        activation_buffer_free(orig);
                        activation_pair_free(acts);
                        return false;
                    }
                    ggml_backend_tensor_set(shard->layer_graph.kv_idx_full,
                                            rows.data(), 0,
                                            ggml_nbytes(shard->layer_graph.kv_idx_full));
                    ggml_backend_tensor_set(shard->layer_graph.attn_mask_full,
                                            mfull.data(), 0,
                                            ggml_nbytes(shard->layer_graph.attn_mask_full));
                } else {
                    ggml_backend_tensor_set(shard->layer_graph.kv_idx_full,
                                            pos.data(), 0,
                                            ggml_nbytes(shard->layer_graph.kv_idx_full));
                }
            }
            if (!use_kvflash && tensor_ready(shard->layer_graph.attn_mask_full)) {
                const int kv_len_raw = kv_start + n;
                const int kv_len_padded = (int)shard->layer_graph.attn_mask_full->ne[0];
                std::vector<float> mfull((size_t)kv_len_padded * n, -INFINITY);
                for (int q = 0; q < n; ++q) {
                    const int abs_q = kv_start + q;
                    for (int k = 0; k <= abs_q && k < kv_len_raw && k < kv_len_padded; ++k) {
                        mfull[(size_t)q * kv_len_padded + k] = 0.0f;
                    }
                }
                ggml_backend_tensor_set(shard->layer_graph.attn_mask_full,
                                        mfull.data(), 0,
                                        ggml_nbytes(shard->layer_graph.attn_mask_full));
            }

            const int swa_size = shard->cache.swa_size;
            const int swa_len_raw = std::min(kv_start + n, swa_size);
            const int swa_len_padded = tensor_ready(shard->layer_graph.attn_mask_swa)
                ? (int)shard->layer_graph.attn_mask_swa->ne[0]
                : ((swa_len_raw + 255) & ~255);
            if (tensor_ready(shard->layer_graph.kv_idx_swa)) {
                std::vector<int32_t> ring((size_t)n);
                for (int i = 0; i < n; ++i) ring[(size_t)i] = (kv_start + i) % swa_size;
                ggml_backend_tensor_set(shard->layer_graph.kv_idx_swa,
                                        ring.data(), 0,
                                        ggml_nbytes(shard->layer_graph.kv_idx_swa));
            }
            std::vector<float> mswa((size_t)swa_len_padded * n, -INFINITY);
            for (int q = 0; q < n; ++q) {
                const int abs_q = kv_start + q;
                const int win_lo = std::max(0, abs_q - ref.sliding_window + 1);
                for (int abs_k = win_lo; abs_k <= abs_q; ++abs_k) {
                    const int slot = abs_k % swa_size;
                    if (slot < swa_len_raw) {
                        mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
                    }
                }
            }
            if (tensor_ready(shard->layer_graph.attn_mask_swa)) {
                ggml_backend_tensor_set(shard->layer_graph.attn_mask_swa,
                                        mswa.data(), 0,
                                        ggml_nbytes(shard->layer_graph.attn_mask_swa));
            }

            auto st = ggml_backend_graph_compute(shard->backend,
                                                 shard->layer_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr,
                    "[gemma4-target-split] compute layer=%d @%d gpu=%d status=%d\n",
                    il, start, shard->gpu, (int)st);
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }
            start += n;
        }
        std::swap(act_in, act_out);
    }

    std::vector<int32_t> argmax;
    Gemma4LayerSplitShard & last = shards_.back();
    const bool ok = compute_gemma4_split_projection(
        last.backend, last.weights, act_in,
        n_tokens_total - 1, 1, &argmax, logits_out);
    activation_buffer_free(orig);
    activation_pair_free(acts);
    if (!ok || argmax.empty()) return false;
    last_tok = argmax.back();
    for (auto & shard : shards_) {
        shard.cache.cur_pos = base_pos + n_tokens_total;
        shard.cache.last_tok = last_tok;
    }
    return true;
}

bool Gemma4LayerSplitAdapter::prefill(const std::vector<int32_t> & prompt,
                                      int base_pos,
                                      int & last_tok) {
    const bool ok = run_forward(prompt, base_pos, last_tok, &prefill_last_logits_);
    if (ok && kvflash_active()) {
        kvflash_sync_history(prompt, base_pos);
        kvflash_pager_.zero_free_blocks();
    }
    return ok;
}

bool Gemma4LayerSplitAdapter::decode_ar(
        int last_tok,
        int committed,
        int n_gen,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io) {
    if (n_gen <= 0) return true;
    if (shards_.empty()) return false;

    const auto & w = shards_.front().weights;
    const int vocab = w.n_vocab;
    const bool ok = run_layer_split_ar_decode(
        last_tok, committed, n_gen, vocab, prefill_last_logits_, sampler_,
        sampler_rng_,
        [&](const std::vector<int32_t> & one, int pos, int & next_tok,
            std::vector<float> * logits_out) {
            return run_forward(one, pos - 1, next_tok, logits_out);
        },
        [&](int tok) { return tok == w.eos_id || tok == w.eos_chat_id; },
        out_tokens, io);
    if (ok && kvflash_active()) {
        kvflash_sync_history(out_tokens, committed);
        kvflash_maybe_reselect((int)out_tokens.size());
    }
    return ok;
}

bool Gemma4LayerSplitAdapter::snapshot_save(int slot) {
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
                "[gemma4-target-split][kvflash] snapshot skipped: pooled "
                "layout needs page-table serialization\n");
            warned = true;
        }
        return false;
    }

    snapshot_free(slot);
    if (snap.shards.size() != shards_.size()) snap.shards.resize(shards_.size());
    for (size_t i = 0; i < shards_.size(); ++i) {
        const int n_layer = shards_[i].cache.n_layer;
        auto & ss = snap.shards[i];
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 4) + 4096;
        ip.no_alloc = true;
        ss.ctx = ggml_init(ip);
        if (!ss.ctx) {
            snapshot_free(slot);
            return false;
        }
        ss.k_snap.resize((size_t)n_layer, nullptr);
        ss.v_snap.resize((size_t)n_layer, nullptr);
        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * ck = shards_[i].cache.k[il];
            if (!ck) continue;
            const int save_pos = std::min(snap_pos, (int)ck->ne[1]);
            ss.k_snap[(size_t)il] = ggml_new_tensor_3d(
                ss.ctx, ck->type, ck->ne[0], save_pos, ck->ne[2]);
            ss.v_snap[(size_t)il] = ggml_new_tensor_3d(
                ss.ctx, ck->type, ck->ne[0], save_pos, ck->ne[2]);
        }
        ss.buf = ggml_backend_alloc_ctx_tensors(ss.ctx, snapshot_backends_[i]);
        if (!ss.buf) {
            snapshot_free(slot);
            return false;
        }

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * ck = shards_[i].cache.k[il];
            if (!ck || !ss.k_snap[(size_t)il]) continue;
            const int D = (int)ck->ne[0];
            const int Hk = (int)ck->ne[2];
            const int cache_len = (int)ck->ne[1];
            const int save_pos = std::min(snap_pos, cache_len);
            const size_t elem_sz = ggml_element_size(ck);
            const size_t head_bytes_src = (size_t)D * cache_len * elem_sz;
            const size_t head_bytes_dst = (size_t)D * save_pos * elem_sz;
            const size_t copy_bytes = head_bytes_dst;
            for (int h = 0; h < Hk; ++h) {
                ggml_backend_tensor_get(shards_[i].cache.k[il],
                    (char *)ss.k_snap[(size_t)il]->data + h * head_bytes_dst,
                    h * head_bytes_src, copy_bytes);
                ggml_backend_tensor_get(shards_[i].cache.v[il],
                    (char *)ss.v_snap[(size_t)il]->data + h * head_bytes_dst,
                    h * head_bytes_src, copy_bytes);
            }
        }
        ss.cur_pos = snap_pos;
        ss.last_tok = shards_[i].cache.last_tok;
    }
    snap.cur_pos = snap_pos;
    snap.last_tok = shards_.front().cache.last_tok;
    snap.prefill_last_logits = prefill_last_logits_;
    if (kvflash_active() &&
        kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS) {
        auto & history_snap = kvflash_history_snapshots_[(size_t)slot];
        history_snap.clear();
        const size_t keep = (size_t)std::min(snap_pos, (int)kvflash_history_.size());
        history_snap.assign(kvflash_history_.begin(), kvflash_history_.begin() + keep);
        if ((int)history_snap.size() < snap_pos) {
            history_snap.resize((size_t)snap_pos, 0);
        }
    }
    return true;
}

void Gemma4LayerSplitAdapter::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || snapshots_.empty()) return;
    auto & snap = snapshots_[(size_t)slot];
    for (auto & ss : snap.shards) {
        free_gemma4_snapshot(ss);
    }
    snap.cur_pos = 0;
    snap.last_tok = -1;
    snap.prefill_last_logits.clear();
    if (kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS) {
        kvflash_history_snapshots_[(size_t)slot].clear();
    }
    if (snap.shards.size() != shards_.size()) snap.shards.resize(shards_.size());
}

bool Gemma4LayerSplitAdapter::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS ||
        snapshots_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }
    const auto & snap = snapshots_[(size_t)slot];
    if (snap.cur_pos <= 0 || snap.shards.size() != shards_.size()) return false;
    if (snap.prefill_last_logits.empty()) return false;
    for (const auto & ss : snap.shards) {
        if (!ss.ctx) return false;
    }
    return true;
}

int Gemma4LayerSplitAdapter::snapshot_cur_pos(int slot) const {
    return snapshot_used(slot) ? snapshots_[(size_t)slot].cur_pos : 0;
}

bool Gemma4LayerSplitAdapter::snapshot_restore(int slot) {
    if (!snapshot_used(slot)) return false;
    auto & snap = snapshots_[(size_t)slot];
    for (size_t i = 0; i < shards_.size(); ++i) {
        const auto & ss = snap.shards[i];
        if (ss.cur_pos != snap.cur_pos) return false;
        for (int il = 0; il < shards_[i].cache.n_layer; ++il) {
            ggml_tensor * ck = shards_[i].cache.k[il];
            if (!ck || !ss.k_snap[(size_t)il]) continue;
            const int D = (int)ck->ne[0];
            const int Hk = (int)ck->ne[2];
            const int cache_len = (int)ck->ne[1];
            const int save_pos = (int)ss.k_snap[(size_t)il]->ne[1];
            const size_t elem_sz = ggml_element_size(ck);
            const size_t head_bytes_src = (size_t)D * save_pos * elem_sz;
            const size_t head_bytes_dst = (size_t)D * cache_len * elem_sz;
            const size_t copy_bytes = head_bytes_src;
            for (int h = 0; h < Hk; ++h) {
                ggml_backend_tensor_set(shards_[i].cache.k[il],
                    (const char *)ss.k_snap[(size_t)il]->data + h * head_bytes_src,
                    h * head_bytes_dst, copy_bytes);
                ggml_backend_tensor_set(shards_[i].cache.v[il],
                    (const char *)ss.v_snap[(size_t)il]->data + h * head_bytes_src,
                    h * head_bytes_dst, copy_bytes);
            }
        }
        shards_[i].cache.cur_pos = snap.cur_pos;
        shards_[i].cache.last_tok = snap.last_tok;
    }
    prefill_last_logits_ = snap.prefill_last_logits;
    if (kvflash_active()) {
        if (!kvflash_sync_identity(snap.cur_pos)) return false;
        if (kvflash_history_snapshots_.size() == (size_t)PREFIX_SLOTS &&
            !kvflash_history_snapshots_[(size_t)slot].empty()) {
            kvflash_history_ = kvflash_history_snapshots_[(size_t)slot];
        } else {
            kvflash_history_.clear();
        }
        if ((int)kvflash_history_.size() > snap.cur_pos) {
            kvflash_history_.resize((size_t)snap.cur_pos);
        } else if ((int)kvflash_history_.size() < snap.cur_pos) {
            kvflash_history_.resize((size_t)snap.cur_pos, 0);
        }
    }
    return true;
}

int Gemma4LayerSplitAdapter::current_last_token() const {
    if (shards_.empty()) return -1;
    return shards_.front().cache.last_tok;
}

void Gemma4LayerSplitAdapter::shutdown() {
    kvflash_scorer_.reset();
    if (kvflash_drafter_loaded_) {
        dflash::common::free_drafter(kvflash_drafter_);
        kvflash_drafter_loaded_ = false;
    }
    for (int i = 0; i < PREFIX_SLOTS; ++i) snapshot_free(i);
    kvflash_history_snapshots_.clear();
    auto shard_metas = layer_split_shard_metas(shards_);
    free_layer_split_snapshot_backends(shard_metas, snapshot_backends_);
    free_gemma4_layer_split_shards(shards_);
}

void free_gemma4_layer_split_shards(
        std::vector<Gemma4LayerSplitShard> & shards) {
    for (auto & shard : shards) {
        gemma4_layer_step_graph_destroy(shard.layer_graph);
        free_gemma4_cache(shard.cache);
        free_gemma4_weights(shard.weights);
        if (shard.backend) {
            ggml_backend_free(shard.backend);
            shard.backend = nullptr;
        }
    }
    shards.clear();
}

}  // namespace dflash::common
