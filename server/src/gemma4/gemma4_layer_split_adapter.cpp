// Gemma4 layer-split adapter.

#include "gemma4_layer_split_adapter.h"

#include "common/dflash_layer_split_runtime.h"
#include "common/gguf_inspect.h"
#include "common/layer_split_utils.h"
#include "dflash27b.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace dflash::common {

namespace {

struct ActivationBuffer {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * tensor = nullptr;
};

void activation_buffer_free(ActivationBuffer & b) {
    if (b.buf) {
        ggml_backend_buffer_free(b.buf);
        b.buf = nullptr;
    }
    if (b.ctx) {
        ggml_free(b.ctx);
        b.ctx = nullptr;
    }
    b.tensor = nullptr;
}

bool activation_buffer_init(ActivationBuffer & b,
                            ggml_backend_t backend,
                            int hidden,
                            int n_tokens) {
    activation_buffer_free(b);
    if (hidden <= 0 || n_tokens <= 0) return false;
    ggml_init_params ip{};
    ip.mem_size = (size_t)4 * ggml_tensor_overhead() + 16 * 1024;
    ip.no_alloc = true;
    b.ctx = ggml_init(ip);
    if (!b.ctx) return false;
    b.tensor = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, hidden, n_tokens);
    if (!b.tensor) {
        activation_buffer_free(b);
        return false;
    }
    ggml_set_name(b.tensor, "gemma4_target_split_orig_embed");
    b.buf = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    if (!b.buf) {
        activation_buffer_free(b);
        return false;
    }
    return true;
}

static bool tensor_ready(const ggml_tensor * t) {
    return t && t->buffer;
}

}  // namespace

Gemma4LayerSplitAdapter::Gemma4LayerSplitAdapter(
        const Gemma4LayerSplitAdapterConfig & cfg)
    : cfg_(cfg) {}

Gemma4LayerSplitAdapter::~Gemma4LayerSplitAdapter() { shutdown(); }

bool Gemma4LayerSplitAdapter::init() {
    if (!cfg_.target_path || cfg_.device.layer_split_gpus.size() < 2) {
        std::fprintf(stderr, "[gemma4-target-split] invalid layer-split config\n");
        return false;
    }

    const auto info = inspect_gguf_model_info(cfg_.target_path);
    const int n_layer = info.n_layer;
    if (n_layer <= 0) {
        std::fprintf(stderr, "[gemma4-target-split] failed to inspect layer count\n");
        return false;
    }

    const auto ranges = compute_layer_ranges(
        n_layer,
        (int)cfg_.device.layer_split_gpus.size(),
        cfg_.device.layer_split_weights);
    if (ranges.size() != cfg_.device.layer_split_gpus.size()) {
        std::fprintf(stderr,
            "[gemma4-target-split] bad layer split for %zu GPUs and %d layers\n",
            cfg_.device.layer_split_gpus.size(), n_layer);
        return false;
    }

    shards_.resize(cfg_.device.layer_split_gpus.size());
    auto shard_metas = layer_split_shard_metas(shards_);
    if (!init_layer_split_shard_metas(
            shard_metas, cfg_.device.layer_split_gpus, ranges,
            "gemma4-target-split")) {
        return false;
    }

    (void)enable_layer_split_peer_access(
        cfg_.device.layer_split_gpus, cfg_.device.peer_access);

    if (!init_layer_split_snapshot_backends(
            shard_metas, snapshot_backends_, "gemma4-target-split")) return false;

    for (size_t i = 0; i < shards_.size(); ++i) {
        auto & shard = shards_[i];
        const TargetLoadPlan plan =
            make_layer_split_load_plan<TargetLoadPlan>(
                shard, i + 1 == shards_.size());
        if (!load_gemma4_gguf_partial(cfg_.target_path, shard.backend,
                                      plan, shard.weights) ||
            !create_gemma4_cache_partial(shard.backend, shard.weights,
                                         cfg_.device.max_ctx,
                                         shard.layer_begin, shard.layer_end,
                                         shard.cache)) {
            std::fprintf(stderr,
                "[gemma4-target-split] load/cache gpu=%d: %s\n",
                shard.gpu, dflash27b_last_error());
            return false;
        }
        shard.cache.fa_window = cfg_.fa_window;
        std::fprintf(stderr, "[gemma4-target-split] gpu=%d layers=[%d,%d)\n",
                     shard.gpu, shard.layer_begin, shard.layer_end);
    }
    snapshots_.resize(PREFIX_SLOTS);
    for (auto & slot : snapshots_) {
        slot.shards.resize(shards_.size());
    }

    return true;
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
                              n_tokens_total)) {
        std::fprintf(stderr, "[gemma4-target-split] activation alloc failed gpu=%d\n",
                     shards_.front().gpu);
        return false;
    }
    ActivationBuffer orig;
    if (!activation_buffer_init(orig, shards_.front().backend, hidden,
                                n_tokens_total)) {
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
            const size_t bytes = sizeof(float) * (size_t)hidden * n;
            ggml_backend_tensor_set(acts.a, emb.data(), off, bytes);
            ggml_backend_tensor_set(orig.tensor, emb.data(), off, bytes);
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
                                      n_tokens_total)) {
                activation_buffer_free(orig);
                activation_pair_free(acts);
                return false;
            }
            ActivationBuffer next_orig;
            if (!activation_buffer_init(next_orig, shard->backend, hidden,
                                        n_tokens_total)) {
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
            if (!build_gemma4_layer_step(
                    shard->layer_graph, shard->weights, shard->cache,
                    shard->backend, il, act_in, orig.tensor, act_out,
                    start, n, kv_start)) {
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

            const int kv_len_raw = kv_start + n;
            const int kv_len_padded = (kv_len_raw + 255) & ~255;
            std::vector<float> mfull((size_t)kv_len_padded * n, -INFINITY);
            for (int q = 0; q < n; ++q) {
                const int abs_q = kv_start + q;
                for (int k = 0; k <= abs_q && k < kv_len_raw; ++k) {
                    mfull[(size_t)q * kv_len_padded + k] = 0.0f;
                }
            }
            if (tensor_ready(shard->layer_graph.attn_mask_full)) {
                ggml_backend_tensor_set(shard->layer_graph.attn_mask_full,
                                        mfull.data(), 0,
                                        ggml_nbytes(shard->layer_graph.attn_mask_full));
            }

            const int swa_size = shard->cache.swa_size;
            const int swa_len_raw = std::min(kv_start + n, swa_size);
            const int swa_len_padded = (swa_len_raw + 255) & ~255;
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
    return run_forward(prompt, base_pos, last_tok,
                       sampler_.needs_logit_processing() ? &prefill_last_logits_ : nullptr);
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
    std::vector<float> logits_buf;
    if (sampler_.needs_logit_processing()) {
        if ((int)prefill_last_logits_.size() != vocab) return false;
        last_tok = sample_logits(prefill_last_logits_.data(), vocab, sampler_,
                                 out_tokens, sampler_rng_);
    }
    out_tokens.push_back(last_tok);
    io.emit(last_tok);
    if (io.cancelled) {
        io.emit(-1);
        return true;
    }
    if (last_tok == w.eos_id || last_tok == w.eos_chat_id) {
        io.emit(-1);
        return true;
    }
    ++committed;

    for (int i = 1; i < n_gen; ++i) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        logits_buf.clear();
        if (!run_forward(one, committed - 1, next_tok,
                         sampler_.needs_logit_processing() ? &logits_buf : nullptr)) {
            return false;
        }
        if (sampler_.needs_logit_processing()) {
            if ((int)logits_buf.size() != vocab) return false;
            next_tok = sample_logits(logits_buf.data(), vocab, sampler_,
                                     out_tokens, sampler_rng_);
        }
        last_tok = next_tok;
        out_tokens.push_back(last_tok);
        io.emit(last_tok);
        ++committed;
        if (io.cancelled) break;
        if (last_tok == w.eos_id || last_tok == w.eos_chat_id) break;
    }
    io.emit(-1);
    return true;
}

bool Gemma4LayerSplitAdapter::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || shards_.empty()) return false;
    if (snapshot_backends_.size() != shards_.size()) return false;
    auto & snap = snapshots_[(size_t)slot];
    const int snap_pos = shards_.front().cache.cur_pos;
    if (snap_pos <= 0) return false;

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
    if (snap.shards.size() != shards_.size()) snap.shards.resize(shards_.size());
}

bool Gemma4LayerSplitAdapter::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS ||
        snapshots_.size() != (size_t)PREFIX_SLOTS) {
        return false;
    }
    const auto & snap = snapshots_[(size_t)slot];
    if (snap.cur_pos <= 0 || snap.shards.size() != shards_.size()) return false;
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
    return true;
}

int Gemma4LayerSplitAdapter::current_last_token() const {
    if (shards_.empty()) return -1;
    return shards_.front().cache.last_tok;
}

void Gemma4LayerSplitAdapter::shutdown() {
    for (int i = 0; i < PREFIX_SLOTS; ++i) snapshot_free(i);
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
