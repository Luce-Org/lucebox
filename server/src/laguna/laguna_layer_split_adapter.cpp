// Laguna target layer-split adapter.

#include "laguna_layer_split_adapter.h"

#include "common/dflash_layer_split_runtime.h"
#include "common/gguf_inspect.h"
#include "common/layer_split_utils.h"
#include "common/sampler.h"
#include "common/layer_split_runtime.h"
#include "dflash27b.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

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
    const LayerSplitRuntimeInit runtime_cfg{
        cfg_.target_path,
        &cfg_.device,
        "laguna-target-split",
    };
    if (!init_layer_split_runtime(runtime_cfg, shards_, snapshot_backends_)) {
        return false;
    }

    for (size_t i = 0; i < shards_.size(); ++i) {
        auto & shard = shards_[i];
        const TargetLoadPlan plan =
            make_layer_split_load_plan<TargetLoadPlan>(shard, i + 1 == shards_.size());
        if (!load_target_gguf_laguna_partial(
                cfg_.target_path, shard.backend, plan, shard.weights) ||
            !create_laguna_target_cache_partial(
                shard.weights, cfg_.device.max_ctx, shard.backend,
                shard.layer_begin, shard.layer_end, shard.cache)) {
            std::fprintf(stderr,
                "[laguna-target-split] load/cache gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
            return false;
        }
        std::fprintf(stderr, "[laguna-target-split] gpu=%d layers=[%d,%d)\n",
                     shard.gpu, shard.layer_begin, shard.layer_end);
    }

    snapshots_.resize(PREFIX_SLOTS);
    for (auto & slot : snapshots_) {
        slot.shards.resize(shards_.size());
    }
    return true;
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
                              n_tokens_total)) {
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
            const size_t bytes = sizeof(float) * (size_t)hidden * n;
            ggml_backend_tensor_set(acts.a, emb.data(), off, bytes);
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
                                      n_tokens_total)) {
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
            if (!build_laguna_layer_step(
                    shard->layer_graph, shard->weights, shard->cache,
                    shard->backend, il, act_in, act_out, start, n, kv_start)) {
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

            const int kv_len = kv_start + n;
            std::vector<float> mfull((size_t)kv_len * n, -INFINITY);
            for (int q = 0; q < n; ++q) {
                const int abs_q = kv_start + q;
                for (int k = 0; k <= abs_q && k < kv_len; ++k) {
                    mfull[(size_t)q * kv_len + k] = 0.0f;
                }
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

bool LagunaLayerSplitAdapter::prefill(const std::vector<int32_t> & prompt,
                                      int base_pos,
                                      int & last_tok) {
    return run_forward(prompt, base_pos, last_tok, &prefill_last_logits_);
}

bool LagunaLayerSplitAdapter::decode_ar(
        int last_tok,
        int committed,
        int n_gen,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io) {
    if (n_gen <= 0) return true;
    if (shards_.empty()) return false;

    const auto & w = shards_.front().weights;
    const int vocab = (int)w.embedder.n_vocab;
    return run_layer_split_ar_decode(
        last_tok, committed, n_gen, vocab, prefill_last_logits_, sampler_,
        sampler_rng_,
        [&](const std::vector<int32_t> & one, int pos, int & next_tok,
            std::vector<float> * logits_out) {
            return run_forward(one, pos - 1, next_tok, logits_out);
        },
        [&](int tok) { return tok == w.eos_id || tok == w.eos_chat_id; },
        out_tokens, io);
}

bool LagunaLayerSplitAdapter::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || shards_.empty()) return false;
    if (snapshot_backends_.size() != shards_.size()) return false;
    auto & snap = snapshots_[(size_t)slot];
    const int snap_pos = shards_.front().cache.cur_pos;
    if (snap_pos <= 0) return false;

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
    snap.cur_pos = snap_pos;
    snap.last_tok = shards_.front().cache.last_tok;
    snap.prefill_last_logits = prefill_last_logits_;
    return true;
}

void LagunaLayerSplitAdapter::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || snapshots_.empty()) return;
    auto & snap = snapshots_[(size_t)slot];
    for (auto & ss : snap.shards) {
        laguna_snapshot_free(ss);
    }
    snap.cur_pos = 0;
    snap.last_tok = -1;
    snap.prefill_last_logits.clear();
    if (snap.shards.size() != shards_.size()) snap.shards.resize(shards_.size());
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
    prefill_last_logits_ = snap.prefill_last_logits;
    return true;
}

int LagunaLayerSplitAdapter::current_last_token() const {
    if (shards_.empty()) return -1;
    return shards_.front().cache.last_tok;
}

void LagunaLayerSplitAdapter::shutdown() {
    for (int i = 0; i < PREFIX_SLOTS; ++i) snapshot_free(i);
    auto shard_metas = layer_split_shard_metas(shards_);
    free_layer_split_snapshot_backends(shard_metas, snapshot_backends_);
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

}  // namespace dflash::common
