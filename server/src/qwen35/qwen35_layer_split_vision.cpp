// Multimodal prefill for Qwen35 layer-split adapter.

#include "qwen35_layer_split_adapter.h"

#include "common/dflash_layer_split_runtime.h"
#include "dflash_feature_ring.h"
#include "internal.h"
#include "qwen35/layer_split_forward.h"
#include "vision/vision_encoder.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace dflash::common {

bool Qwen35LayerSplitAdapter::supports_multimodal() const {
#ifdef DFLASH_HAVE_MMPROJ
    return vision_ && vision_->ready();
#else
    return false;
#endif
}

void Qwen35LayerSplitAdapter::sync_draft_features_range(int kv_pos, int n_tokens) {
    if (n_tokens <= 0 || !cfg_.run_dflash) return;
    if (remote_draft_.active()) {
        // Remote draft sync is handled inside spec-decode; skip here.
        return;
    }
    if (feature_ring_.target_feat && !shards_.empty()) {
        const auto & cache = shards_.front().cache;
        if (cache.target_feat) {
            draft_feature_mirror_sync_range(cache.target_feat, cache.target_feat_cap,
                                            feature_ring_, kv_pos, n_tokens);
        }
    }
}

bool Qwen35LayerSplitAdapter::prefill_activation_chunk(
        int kv_pos, int n_tokens, const float * embeds, int hidden,
        const LayerSplitAttnPrefillOpts * attn_opts, bool want_logits) {
    if (shards_.empty() || n_tokens <= 0 || !embeds) return false;
    if (use_mixed_target_split()) {
        std::fprintf(stderr, "[target-split][vision] mixed target split unsupported\n");
        return false;
    }

    int ubatch = cfg_.chunk > 0 ? cfg_.chunk : 512;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        ubatch = std::max(1, std::atoi(s));
    }

    ActivationPair acts;
    if (!activation_pair_init(acts, shards_.front().backend, hidden, n_tokens,
                              activation_type_)) {
        std::fprintf(stderr, "[target-split][vision] activation alloc failed\n");
        return false;
    }

    if (!set_activation_tensor_from_f32(
            acts.a, embeds, 0, (size_t)hidden * (size_t)n_tokens)) {
        activation_pair_free(acts);
        return false;
    }

    int last_tok = -1;
    std::vector<float> logits;
    const bool ok = run_qwen35_layer_split_forward_from_activation(
        shards_, acts, kv_pos, n_tokens, ubatch, last_tok,
        cfg_.kq_stride_pad, cfg_.fa_window,
        /*argmax_out=*/nullptr,
        want_logits ? &prefill_last_logits_ : nullptr,
        /*captures_out=*/nullptr,
        kvflash_active() ? &kvflash_pager_ : nullptr,
        /*kvflash_preallocated=*/false,
        /*tree_inputs=*/nullptr,
        /*capture_ssm_intermediates=*/false,
        attn_opts);
    activation_pair_free(acts);
    if (!ok) return false;

    if (cfg_.run_dflash && !remote_draft_.active()) {
        sync_draft_features_range(kv_pos, n_tokens);
    }
    return true;
}

int Qwen35LayerSplitAdapter::prefill_multimodal(MultimodalPrompt & mm,
                                                int & last_tok) {
#ifdef DFLASH_HAVE_MMPROJ
    if (!vision_ || !vision_->ready()) {
        std::fprintf(stderr, "[target-split][vision] encoder not initialized\n");
        return -1;
    }
    if (shards_.empty()) return -1;

    for (auto & shard : shards_) {
        reset_recurrent_state(shard.cache);
    }

    mtmd_input_chunks * chunks = vision_->tokenize(mm.marked_text, mm.images);
    if (!chunks) return -1;

    struct ChunksGuard {
        mtmd_input_chunks * p;
        ~ChunksGuard() { if (p) mtmd_input_chunks_free(p); }
    } chunks_guard{chunks};

    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        last_tok = shards_.front().cache.last_tok;
        return 0;
    }

    const int hidden = shards_.front().weights.n_embd;
    int prefill_ubatch = 512;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        prefill_ubatch = std::max(1, std::atoi(s));
    }

    std::vector<float> embed_buf;
    int committed = 0;

    for (size_t ci = 0; ci < n_chunks; ci++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, ci);
        const auto chunk_type = mtmd_input_chunk_get_type(chunk);
        const bool is_last_chunk = (ci + 1 == n_chunks);
        const int kv_pos = committed;

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens_sz = 0;
            const llama_token * toks =
                mtmd_input_chunk_get_tokens_text(chunk, &n_tokens_sz);
            const int n_tokens = (int)n_tokens_sz;
            if (n_tokens <= 0) continue;

            if (n_tokens > prefill_ubatch) {
                std::fprintf(stderr,
                    "[target-split][vision] chunking text mtmd_chunk=%zu tokens=%d ubatch=%d\n",
                    ci, n_tokens, prefill_ubatch);
            }

            std::vector<int32_t> tokens((size_t)n_tokens);
            for (int i = 0; i < n_tokens; i++) {
                tokens[(size_t)i] = (int32_t)toks[i];
            }

            for (int start = 0; start < n_tokens; ) {
                const int sub_n = std::min(prefill_ubatch, n_tokens - start);
                const bool is_last_sub = (start + sub_n >= n_tokens);
                const bool want_logits = is_last_chunk && is_last_sub;

                embed_buf.resize((size_t)hidden * (size_t)sub_n);
                if (!shards_.front().weights.embedder.embed(
                        tokens.data() + start, sub_n, embed_buf.data())) {
                    return -1;
                }

                std::vector<int32_t> pos_buf((size_t)4 * (size_t)sub_n, 0);
                for (int i = 0; i < sub_n; i++) {
                    const int p = kv_pos + start + i;
                    pos_buf[4 * i + 0] = p;
                    pos_buf[4 * i + 1] = p;
                    pos_buf[4 * i + 2] = p;
                    pos_buf[4 * i + 3] = 0;
                }
                LayerSplitAttnPrefillOpts attn;
                attn.positions = pos_buf.data();

                if (!prefill_activation_chunk(kv_pos + start, sub_n,
                                              embed_buf.data(), hidden,
                                              &attn, want_logits)) {
                    return -1;
                }
                start += sub_n;
            }
            committed = kv_pos + n_tokens;
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ||
                   chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            const int n_tokens = (int)mtmd_input_chunk_get_n_tokens(chunk);
            if (n_tokens <= 0) continue;

            if (!vision_->encode_chunk(chunk)) return -1;
            float * embd = vision_->output_embeddings();
            if (!embd) return -1;

            std::vector<int32_t> pos_buf;
            LayerSplitAttnPrefillOpts attn;
            if (vision_->uses_mrope() &&
                chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                const auto * image_tokens =
                    mtmd_input_chunk_get_tokens_image(chunk);
                if (!image_tokens) return -1;
                std::vector<mtmd_decoder_pos> rel_pos((size_t)n_tokens);
                vision_->get_image_decoder_pos(image_tokens, rel_pos.data(),
                                               (size_t)n_tokens);
                pos_buf.resize((size_t)4 * (size_t)n_tokens, 0);
                for (int i = 0; i < n_tokens; i++) {
                    pos_buf[4 * i + 0] =
                        kv_pos + (int)rel_pos[(size_t)i].t;
                    pos_buf[4 * i + 1] =
                        kv_pos + (int)rel_pos[(size_t)i].y;
                    pos_buf[4 * i + 2] =
                        kv_pos + (int)rel_pos[(size_t)i].x;
                    pos_buf[4 * i + 3] = 0;
                }
                attn.positions = pos_buf.data();
            } else {
                pos_buf.resize((size_t)4 * (size_t)n_tokens, 0);
                for (int i = 0; i < n_tokens; i++) {
                    const int p = kv_pos + i;
                    pos_buf[4 * i + 0] = p;
                    pos_buf[4 * i + 1] = p;
                    pos_buf[4 * i + 2] = p;
                    pos_buf[4 * i + 3] = 0;
                }
                attn.positions = pos_buf.data();
            }
            attn.bidirectional = vision_->uses_non_causal(chunk);

            if (!prefill_activation_chunk(kv_pos, n_tokens, embd, hidden, &attn,
                                          /*want_logits=*/is_last_chunk)) {
                return -1;
            }
            committed = kv_pos + (int)mtmd_input_chunk_get_n_pos(chunk);
        } else {
            std::fprintf(stderr, "[target-split][vision] unsupported mtmd chunk\n");
            return -1;
        }

        for (auto & shard : shards_) {
            shard.cache.cur_pos = committed;
        }
    }

    last_tok = shards_.front().cache.last_tok;
    return committed;
#else
    (void)mm;
    (void)last_tok;
    std::fprintf(stderr, "[target-split][vision] built without DFLASH_HAVE_MMPROJ\n");
    return -1;
#endif
}

}  // namespace dflash::common
