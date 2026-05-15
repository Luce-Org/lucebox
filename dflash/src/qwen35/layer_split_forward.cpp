// layer_split_forward.cpp — Multi-GPU layer-split forward pass.

#include "layer_split_forward.h"
#include "internal.h"
#include "graph_builders.h"
#include "dflash_feature_ring.h"
#include "dflash_capture.h"
#include "attn_masks.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace dflash27b {

bool compute_target_split_argmax(
        StepGraph & sg,
        const TargetWeights & w,
        ggml_backend_t backend,
        ggml_tensor * act,
        int token_offset,
        int n_tokens,
        int hidden,
        int vocab,
        std::vector<int32_t> & argmax_out) {
    step_graph_free(sg);
    ggml_init_params ip{};
    ip.mem_size = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    ggml_tensor * act_view = ggml_view_2d(
        sg.ctx, act, hidden, n_tokens, act->nb[1],
        (size_t)token_offset * act->nb[1]);
    ggml_tensor * normed = ggml_rms_norm(sg.ctx, act_view, DFLASH27B_RMS_EPS);
    normed = ggml_mul(sg.ctx, normed, w.out_norm);
    ggml_tensor * logits = ggml_mul_mat(sg.ctx, w.output, normed);
    ggml_set_name(logits, "target_split_logits");
    sg.logits = logits;
    sg.argmax_tokens = ggml_argmax(sg.ctx, logits);
    ggml_set_name(sg.argmax_tokens, "target_split_argmax");
    ggml_set_output(sg.argmax_tokens);
    sg.gf = ggml_new_graph_custom(sg.ctx, 1024, false);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);
    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) return false;
    auto st = ggml_backend_graph_compute(backend, sg.gf);
    if (st != GGML_STATUS_SUCCESS) return false;
    (void)vocab;
    argmax_out.assign((size_t)n_tokens, 0);
    ggml_backend_tensor_get(sg.argmax_tokens, argmax_out.data(), 0,
                            sizeof(int32_t) * (size_t)n_tokens);
    return true;
}

bool run_target_layer_split_forward(
        std::vector<TargetLayerSplitShard> & shards,
        const TargetWeights & embed_source,
        const std::vector<int32_t> & tokens,
        int base_pos,
        int ubatch,
        int & last_tok,
        int kq_stride_pad,
        int fa_window,
        DraftFeatureMirror * feature_ring,
        std::vector<int32_t> * argmax_out,
        std::vector<float> * logits_out,
        DFlashDraftIpcClient * remote_draft) {
    if (shards.empty() || tokens.empty()) return false;
    const int hidden = shards.front().weights.n_embd;
    const int vocab = shards.front().weights.n_vocab;
    const int n_tokens_total = (int)tokens.size();
    ubatch = std::max(1, ubatch);

    ActivationPair acts;
    if (!activation_pair_init(acts, shards.front().backend, hidden, n_tokens_total)) {
        std::fprintf(stderr, "target-split activation alloc failed on gpu %d\n", shards.front().gpu);
        return false;
    }
    ggml_tensor * act_in = acts.a;
    ggml_tensor * act_out = acts.b;

    {
        const int EMBED_BATCH = 4096;
        std::vector<float> emb_buf((size_t)hidden * std::min(EMBED_BATCH, n_tokens_total));
        for (int i = 0; i < n_tokens_total; i += EMBED_BATCH) {
            const int n = std::min(EMBED_BATCH, n_tokens_total - i);
            if ((int)emb_buf.size() < hidden * n) emb_buf.resize((size_t)hidden * n);
            if (!embed_source.embedder.embed(tokens.data() + i, n, emb_buf.data())) {
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_tensor_set(act_in, emb_buf.data(),
                                    (size_t)i * act_in->nb[1],
                                    sizeof(float) * (size_t)hidden * n);
        }
    }

    TargetLayerSplitShard * current_shard = &shards.front();
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos_buf;
    for (int il = 0; il < embed_source.n_layer; il++) {
        TargetLayerSplitShard * shard = find_target_shard(shards, il);
        if (!shard) {
            std::fprintf(stderr, "target-split missing owner for layer %d\n", il);
            activation_pair_free(acts);
            return false;
        }
        if (shard != current_shard) {
            ActivationPair next_acts;
            if (!activation_pair_init(next_acts, shard->backend, hidden, n_tokens_total)) {
                std::fprintf(stderr, "target-split activation alloc failed on gpu %d\n", shard->gpu);
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

        const bool is_attn = (((il + 1) % embed_source.full_attention_interval) == 0);
        const int capture_idx = target_capture_index(embed_source.capture_layer_ids,
                                                     embed_source.n_capture_layers, il);
        for (int start = 0; start < n_tokens_total; start += ubatch) {
            const int n = std::min(ubatch, n_tokens_total - start);
            const int kv_start = base_pos + start;
            const int kv_len = kv_start + n;
            const bool with_mask = (kq_stride_pad > KQ_MASK_PAD) || (n > 1);
            if (!build_layer_step(shard->layer_graph, shard->weights, shard->cache,
                                  shard->backend, il, act_in, act_out,
                                  start, n, kv_start, with_mask,
                                  /*capture=*/false, fa_window, kq_stride_pad)) {
                std::fprintf(stderr, "target-split build layer=%d @%d gpu=%d\n",
                             il, start, shard->gpu);
                activation_pair_free(acts);
                return false;
            }
            if (is_attn && shard->layer_graph.positions) {
                pos_buf.assign((size_t)4 * n, 0);
                for (int i = 0; i < n; i++) {
                    const int p = kv_start + i;
                    pos_buf[0 * n + i] = p;
                    pos_buf[1 * n + i] = p;
                    pos_buf[2 * n + i] = p;
                    pos_buf[3 * n + i] = 0;
                }
                ggml_backend_tensor_set(shard->layer_graph.positions, pos_buf.data(), 0,
                                        sizeof(int32_t) * pos_buf.size());
            }
            if (is_attn && with_mask && shard->layer_graph.attn_mask) {
                const int win_start_l = (fa_window > 0 && kv_start > fa_window)
                                            ? (kv_start - fa_window) : 0;
                const int win_len_l = kv_len - win_start_l;
                build_causal_mask(mask_buf, win_len_l, n, kv_start, kq_stride_pad, win_start_l);
                ggml_backend_tensor_set(shard->layer_graph.attn_mask, mask_buf.data(), 0,
                                        sizeof(uint16_t) * mask_buf.size());
            }
            auto st = ggml_backend_graph_compute(shard->backend, shard->layer_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "target-split compute layer=%d @%d gpu=%d status=%d\n",
                             il, start, shard->gpu, (int)st);
                activation_pair_free(acts);
                return false;
            }
            if ((feature_ring || remote_draft) && capture_idx >= 0) {
                if (feature_ring &&
                    !copy_capture_slice_to_draft_ring(*feature_ring, capture_idx,
                                                      act_out, shard->gpu,
                                                      start, base_pos + start, n)) {
                    std::fprintf(stderr,
                                 "target-split capture copy failed layer=%d capture=%d gpu=%d\n",
                                 il, capture_idx, shard->gpu);
                    activation_pair_free(acts);
                    return false;
                }
                if (remote_draft &&
                    !copy_capture_slice_to_remote_draft(*remote_draft, capture_idx,
                                                        act_out, shard->backend,
                                                        start, base_pos + start, n)) {
                    std::fprintf(stderr,
                                 "target-split remote capture failed layer=%d capture=%d gpu=%d\n",
                                 il, capture_idx, shard->gpu);
                    activation_pair_free(acts);
                    return false;
                }
            }
        }
        std::swap(act_in, act_out);
    }

    StepGraph final_sg;
    std::vector<int32_t> argmax_tokens;
    TargetLayerSplitShard & last_shard = shards.back();
    const bool need_all_argmax = argmax_out != nullptr;
    const int argmax_offset = need_all_argmax ? 0 : (n_tokens_total - 1);
    const int argmax_count = need_all_argmax ? n_tokens_total : 1;
    const bool ok = compute_target_split_argmax(
        final_sg, last_shard.weights, last_shard.backend, act_in,
        argmax_offset, argmax_count, hidden, vocab, argmax_tokens);
    step_graph_destroy(final_sg);
    activation_pair_free(acts);
    if (!ok) return false;
    last_tok = argmax_tokens.empty() ? -1 : argmax_tokens.back();
    if (argmax_out) *argmax_out = std::move(argmax_tokens);
    if (logits_out) logits_out->clear();
    return true;
}

void free_target_layer_split_shards(std::vector<TargetLayerSplitShard> & shards) {
    for (auto & shard : shards) {
        step_graph_destroy(shard.layer_graph);
        free_target_cache(shard.cache);
        free_target_weights(shard.weights);
        if (shard.backend) {
            ggml_backend_free(shard.backend);
            shard.backend = nullptr;
        }
    }
    shards.clear();
}

} // namespace dflash27b
