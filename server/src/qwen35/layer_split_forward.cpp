// layer_split_forward.cpp — Multi-GPU layer-split forward pass.

#include "layer_split_forward.h"
#include "common/ggml_graph_precision.h"
#include "internal.h"
#include "graph_builders.h"
#include "dflash_feature_ring.h"
#include "dflash_capture.h"
#include "attn_masks.h"
#include "common/kvflash_pager.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace dflash::common {

namespace {

bool fill_qwen35_kvflash_inputs(
        StepGraph & sg,
        const TargetWeights & w,
        const KvFlashPager & pager,
        int kv_start,
        int n_tokens) {
    if (!sg.kv_write_rows) {
        std::fprintf(stderr, "target-split kvflash requires set_rows path\n");
        return false;
    }
    std::vector<int64_t> rows((size_t)n_tokens * w.n_head_kv);
    for (int i = 0; i < n_tokens; ++i) {
        const int slot = pager.slot_of(kv_start + i);
        if (slot < 0) {
            std::fprintf(stderr,
                "target-split kvflash slot missing @%d (alloc_span not called?)\n",
                kv_start + i);
            return false;
        }
        for (int h = 0; h < w.n_head_kv; ++h) {
            rows[(size_t)h * n_tokens + i] = slot;
        }
    }
    ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                            sizeof(int64_t) * rows.size());

    if (!sg.attn_mask) return true;
    const size_t kvd = (size_t)sg.attn_mask->ne[0];
    const int q_pad = (int)sg.attn_mask->ne[1];
    std::vector<int32_t> slot_pos((size_t)pager.pool_tokens(), -1);
    pager.fill_slot_pos(slot_pos.data());
    std::vector<uint16_t> mask((size_t)kvd * q_pad, F16_NEG_INF);
    const int s_hi = std::min((int)kvd, (int)slot_pos.size());
    for (int s = 0; s < s_hi; ++s) {
        const int p = slot_pos[(size_t)s];
        if (p >= 0 && p < kv_start) {
            mask[(size_t)s] = F16_ZERO;
        }
    }
    for (int q = 1; q < n_tokens; ++q) {
        std::memcpy(mask.data() + (size_t)q * kvd, mask.data(), kvd * sizeof(uint16_t));
    }
    for (int q = 0; q < n_tokens; ++q) {
        for (int i = 0; i <= q; ++i) {
            const int slot = pager.slot_of(kv_start + i);
            if (slot >= 0 && slot < (int)kvd) {
                mask[(size_t)q * kvd + (size_t)slot] = F16_ZERO;
            }
        }
    }
    ggml_backend_tensor_set(sg.attn_mask, mask.data(), 0,
                            sizeof(uint16_t) * mask.size());
    return true;
}

}  // namespace

bool compute_target_split_projection(
        StepGraph & sg,
        const TargetWeights & w,
        ggml_backend_t backend,
        ggml_tensor * act,
        int token_offset,
        int n_tokens,
        int hidden,
        int vocab,
        std::vector<int32_t> * argmax_out,
        std::vector<float> * logits_out) {
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
    ggml_tensor * normed = ggml_rms_norm(
        sg.ctx, rms_norm_input_f32(sg.ctx, act_view), DFLASH27B_RMS_EPS);
    normed = ggml_mul(sg.ctx, normed, graph_tensor_f32(sg.ctx, w.out_norm));
    ggml_tensor * logits = ggml_mul_mat(sg.ctx, w.output, normed);
    ggml_set_name(logits, "target_split_logits");
    sg.logits = logits;
    if (argmax_out) {
        sg.argmax_tokens = ggml_argmax(sg.ctx, logits);
        ggml_set_name(sg.argmax_tokens, "target_split_argmax");
        ggml_set_output(sg.argmax_tokens);
    }
    if (logits_out) {
        ggml_set_output(sg.logits);
    }
    sg.gf = ggml_new_graph_custom(sg.ctx, 1024, false);
    if (argmax_out) ggml_build_forward_expand(sg.gf, sg.argmax_tokens);
    if (logits_out) ggml_build_forward_expand(sg.gf, sg.logits);
    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) return false;
    auto st = ggml_backend_graph_compute(backend, sg.gf);
    if (st != GGML_STATUS_SUCCESS) return false;
    if (argmax_out) {
        argmax_out->assign((size_t)n_tokens, 0);
        ggml_backend_tensor_get(sg.argmax_tokens, argmax_out->data(), 0,
                                sizeof(int32_t) * (size_t)n_tokens);
    }
    if (logits_out) {
        logits_out->assign((size_t)vocab * (size_t)n_tokens, 0.0f);
        ggml_backend_tensor_get(sg.logits, logits_out->data(), 0,
                                sizeof(float) * (size_t)vocab * (size_t)n_tokens);
    }
    return true;
}

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
    return compute_target_split_projection(
        sg, w, backend, act, token_offset, n_tokens, hidden, vocab,
        &argmax_out, nullptr);
}

bool run_qwen35_layer_split_forward(
        std::vector<Qwen35LayerSplitShard> & shards,
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
        DFlashDraftIpcClient * remote_draft,
        ggml_type activation_type,
        KvFlashPager * kvflash) {
    if (shards.empty() || tokens.empty()) return false;
    const int hidden = shards.front().weights.n_embd;
    const int vocab = shards.front().weights.n_vocab;
    const int n_tokens_total = (int)tokens.size();
    ubatch = std::max(1, ubatch);
    if (kvflash && fa_window > 0) {
        std::fprintf(stderr,
            "target-split kvflash requires full attention (fa_window=0)\n");
        return false;
    }
    if ((feature_ring || remote_draft) && activation_type != GGML_TYPE_F32) {
        std::fprintf(stderr,
            "target-split capture requires F32 activation; got %s\n",
            ggml_type_name(activation_type));
        return false;
    }

    ActivationPair acts;
    if (!activation_pair_init(acts, shards.front().backend, hidden, n_tokens_total,
                              activation_type)) {
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
            if (!set_activation_tensor_from_f32(
                    act_in, emb_buf.data(), (size_t)i * act_in->nb[1],
                    (size_t)hidden * (size_t)n)) {
                std::fprintf(stderr,
                    "target-split unsupported activation type: %s\n",
                    ggml_type_name(act_in->type));
                activation_pair_free(acts);
                return false;
            }
        }
    }

    Qwen35LayerSplitShard * current_shard = &shards.front();
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos_buf;
    for (int il = 0; il < embed_source.n_layer; il++) {
        Qwen35LayerSplitShard * shard = find_layer_split_shard(shards, il);
        if (!shard) {
            std::fprintf(stderr, "target-split missing owner for layer %d\n", il);
            activation_pair_free(acts);
            return false;
        }
        if (shard != current_shard) {
            ActivationPair next_acts;
            if (!activation_pair_init(next_acts, shard->backend, hidden, n_tokens_total,
                                      activation_type)) {
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
            const bool with_mask = kvflash ||
                (kq_stride_pad > KQ_MASK_PAD) || (n > 1);
            if (is_attn && kvflash && !kvflash->alloc_span(kv_start, n)) {
                activation_pair_free(acts);
                return false;
            }
            if (!build_layer_step(shard->layer_graph, shard->weights, shard->cache,
                                  shard->backend, il, act_in, act_out,
                                  start, n, kv_start, with_mask,
                                  /*capture=*/false, fa_window, kq_stride_pad,
                                  kvflash != nullptr)) {
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
            if (is_attn && kvflash) {
                if (!fill_qwen35_kvflash_inputs(
                        shard->layer_graph, shard->weights, *kvflash,
                        kv_start, n)) {
                    activation_pair_free(acts);
                    return false;
                }
            } else if (is_attn && with_mask && shard->layer_graph.attn_mask) {
                const int win_start_l = (fa_window > 0 && kv_start > fa_window)
                                            ? (kv_start - fa_window) : 0;
                const int win_len_l = kv_len - win_start_l;
                const int kv_pad_override = (int)shard->layer_graph.attn_mask->ne[0];
                build_causal_mask(mask_buf, win_len_l, n, kv_start, kq_stride_pad, win_start_l, kv_pad_override);
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
    Qwen35LayerSplitShard & last_shard = shards.back();
    const bool need_all_argmax = argmax_out != nullptr;
    const int argmax_offset = need_all_argmax ? 0 : (n_tokens_total - 1);
    const int argmax_count = need_all_argmax ? n_tokens_total : 1;
    const bool ok = compute_target_split_projection(
        final_sg, last_shard.weights, last_shard.backend, act_in,
        argmax_offset, argmax_count, hidden, vocab,
        &argmax_tokens, logits_out);
    step_graph_destroy(final_sg);
    activation_pair_free(acts);
    if (!ok) return false;
    last_tok = argmax_tokens.empty() ? -1 : argmax_tokens.back();
    for (auto & shard : shards) {
        shard.cache.cur_pos = base_pos + n_tokens_total;
        shard.cache.last_tok = last_tok;
    }
    if (argmax_out) *argmax_out = std::move(argmax_tokens);
    return true;
}

namespace {

bool run_qwen35_layer_split_layers_from_activation(
        std::vector<Qwen35LayerSplitShard> & shards,
        ActivationPair & acts,
        int base_pos,
        int n_tokens_total,
        int ubatch,
        int kq_stride_pad,
        int fa_window,
        std::vector<Qwen35TargetCaptureSlice> * captures_out,
        DraftFeatureMirror * feature_ring,
        DFlashDraftIpcClient * remote_draft,
        KvFlashPager * kvflash,
        bool kvflash_preallocated = false) {
    if (shards.empty() || !acts.a || !acts.b || n_tokens_total <= 0) return false;
    if (kvflash && fa_window > 0) {
        std::fprintf(stderr,
            "target-split kvflash requires full attention (fa_window=0)\n");
        return false;
    }
    const int hidden = shards.front().weights.n_embd;
    ubatch = std::max(1, ubatch);

    ggml_tensor * act_in = acts.a;
    ggml_tensor * act_out = acts.b;
    Qwen35LayerSplitShard * current_shard = &shards.front();
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos_buf;

    for (int il = shards.front().layer_begin; il < shards.back().layer_end; ++il) {
        Qwen35LayerSplitShard * shard = find_layer_split_shard(shards, il);
        if (!shard) {
            std::fprintf(stderr, "target-split missing owner for layer %d\n", il);
            return false;
        }
        if (shard != current_shard) {
            ActivationPair next_acts;
            if (!activation_pair_init(next_acts, shard->backend, hidden, n_tokens_total)) {
                std::fprintf(stderr, "target-split activation alloc failed on gpu %d\n",
                             shard->gpu);
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

        const bool is_attn = (((il + 1) % shard->weights.full_attention_interval) == 0);
        const int capture_idx = target_capture_index(shard->weights.capture_layer_ids,
                                                     shard->weights.n_capture_layers, il);
        for (int start = 0; start < n_tokens_total; start += ubatch) {
            const int n = std::min(ubatch, n_tokens_total - start);
            const int kv_start = base_pos + start;
            const int kv_len = kv_start + n;
            const bool with_mask = kvflash ||
                (kq_stride_pad > KQ_MASK_PAD) || (n > 1);
            if (is_attn && kvflash && !kvflash_preallocated &&
                !kvflash->alloc_span(kv_start, n)) {
                return false;
            }
            if (!build_layer_step(shard->layer_graph, shard->weights, shard->cache,
                                  shard->backend, il, act_in, act_out,
                                  start, n, kv_start, with_mask,
                                  /*capture=*/false, fa_window, kq_stride_pad,
                                  kvflash != nullptr)) {
                std::fprintf(stderr, "target-split build layer=%d @%d gpu=%d\n",
                             il, start, shard->gpu);
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
            if (is_attn && kvflash) {
                if (!fill_qwen35_kvflash_inputs(
                        shard->layer_graph, shard->weights, *kvflash,
                        kv_start, n)) {
                    return false;
                }
            } else if (is_attn && with_mask && shard->layer_graph.attn_mask) {
                const int win_start_l = (fa_window > 0 && kv_start > fa_window)
                                            ? (kv_start - fa_window) : 0;
                const int win_len_l = kv_len - win_start_l;
                const int kv_pad_override = (int)shard->layer_graph.attn_mask->ne[0];
                build_causal_mask(mask_buf, win_len_l, n, kv_start, kq_stride_pad,
                                  win_start_l, kv_pad_override);
                ggml_backend_tensor_set(shard->layer_graph.attn_mask, mask_buf.data(), 0,
                                        sizeof(uint16_t) * mask_buf.size());
            }
            auto st = ggml_backend_graph_compute(shard->backend, shard->layer_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "target-split compute layer=%d @%d gpu=%d status=%d\n",
                             il, start, shard->gpu, (int)st);
                return false;
            }
            if ((captures_out || feature_ring || remote_draft) && capture_idx >= 0) {
                if (captures_out) {
                    Qwen35TargetCaptureSlice capture;
                    capture.capture_idx = capture_idx;
                    capture.start_pos = base_pos + start;
                    capture.n_tokens = n;
                    if (!copy_activation_to_host(act_out, shard->backend,
                                                 start, n, hidden, capture.data)) {
                        std::fprintf(stderr,
                                     "target-split host capture failed layer=%d capture=%d gpu=%d\n",
                                     il, capture_idx, shard->gpu);
                        return false;
                    }
                    captures_out->push_back(std::move(capture));
                }
                if (feature_ring &&
                    !copy_capture_slice_to_draft_ring(*feature_ring, capture_idx,
                                                      act_out, shard->gpu,
                                                      start, base_pos + start, n)) {
                    std::fprintf(stderr,
                                 "target-split capture copy failed layer=%d capture=%d gpu=%d\n",
                                 il, capture_idx, shard->gpu);
                    return false;
                }
                if (remote_draft &&
                    !copy_capture_slice_to_remote_draft(*remote_draft, capture_idx,
                                                        act_out, shard->backend,
                                                        start, base_pos + start, n)) {
                    std::fprintf(stderr,
                                 "target-split remote capture failed layer=%d capture=%d gpu=%d\n",
                                 il, capture_idx, shard->gpu);
                    return false;
                }
            }
        }
        std::swap(act_in, act_out);
    }

    if (act_in != acts.a) {
        std::swap(acts.a, acts.b);
    }
    return true;
}

}  // namespace

bool run_qwen35_layer_split_forward_from_activation(
        std::vector<Qwen35LayerSplitShard> & shards,
        ActivationPair & acts,
        int base_pos,
        int n_tokens_total,
        int ubatch,
        int & last_tok,
        int kq_stride_pad,
        int fa_window,
        std::vector<int32_t> * argmax_out,
        std::vector<float> * logits_out,
        std::vector<Qwen35TargetCaptureSlice> * captures_out,
        KvFlashPager * kvflash,
        bool kvflash_preallocated) {
    if (!run_qwen35_layer_split_layers_from_activation(
            shards, acts, base_pos, n_tokens_total, ubatch, kq_stride_pad,
            fa_window, captures_out, nullptr, nullptr, kvflash,
            kvflash_preallocated)) {
        return false;
    }

    const int hidden = shards.front().weights.n_embd;
    const int vocab = shards.back().weights.n_vocab;
    StepGraph final_sg;
    std::vector<int32_t> argmax_tokens;
    Qwen35LayerSplitShard & last_shard = shards.back();
    const bool need_all_argmax = argmax_out != nullptr;
    const int argmax_offset = need_all_argmax ? 0 : (n_tokens_total - 1);
    const int argmax_count = need_all_argmax ? n_tokens_total : 1;
    const bool ok = compute_target_split_projection(
        final_sg, last_shard.weights, last_shard.backend, acts.a,
        argmax_offset, argmax_count, hidden, vocab,
        &argmax_tokens, logits_out);
    step_graph_destroy(final_sg);
    if (!ok) return false;
    last_tok = argmax_tokens.empty() ? -1 : argmax_tokens.back();
    for (auto & shard : shards) {
        shard.cache.cur_pos = base_pos + n_tokens_total;
        shard.cache.last_tok = last_tok;
    }
    if (argmax_out) *argmax_out = std::move(argmax_tokens);
    return true;
}

bool run_qwen35_mixed_layer_split_forward(
        std::vector<Qwen35LayerSplitShard> & local_shards,
        Qwen35TargetShardIpcClient & remote_shard,
        const TargetWeights & embed_source,
        const std::vector<int32_t> & tokens,
        int base_pos,
        int ubatch,
        int & last_tok,
        int kq_stride_pad,
        int fa_window,
        std::vector<int32_t> * argmax_out,
        std::vector<float> * logits_out,
        DraftFeatureMirror * feature_ring,
        DFlashDraftIpcClient * remote_draft,
        KvFlashPager * kvflash) {
    if (!remote_shard.active() || tokens.empty() ||
        local_shards.empty() || local_shards.front().layer_begin != 0 ||
        local_shards.back().layer_end <= 0) {
        return false;
    }
    const int hidden = local_shards.front().weights.n_embd;
    const int n_tokens_total = (int)tokens.size();
    ubatch = std::max(1, ubatch);
    if (kvflash && fa_window > 0) {
        std::fprintf(stderr,
            "mixed target-split kvflash requires full attention (fa_window=0)\n");
        return false;
    }
    if (kvflash) {
        for (int start = 0; start < n_tokens_total; start += ubatch) {
            const int n = std::min(ubatch, n_tokens_total - start);
            if (!kvflash->alloc_span(base_pos + start, n)) {
                return false;
            }
        }
    }

    ActivationPair acts;
    if (!activation_pair_init(acts, local_shards.front().backend, hidden, n_tokens_total)) {
        std::fprintf(stderr, "mixed target-split activation alloc failed gpu=%d\n",
                     local_shards.front().gpu);
        return false;
    }
    ggml_tensor * act_in = acts.a;

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

    if (!run_qwen35_layer_split_layers_from_activation(
            local_shards, acts, base_pos, n_tokens_total, ubatch,
            kq_stride_pad, fa_window, nullptr, feature_ring, remote_draft,
            kvflash, kvflash != nullptr)) {
        activation_pair_free(acts);
        return false;
    }
    act_in = acts.a;

    std::vector<float> boundary;
    if (!copy_activation_to_host(act_in, local_shards.back().backend, 0, n_tokens_total,
                                 hidden, boundary)) {
        activation_pair_free(acts);
        return false;
    }
    activation_pair_free(acts);

    std::vector<Qwen35TargetCaptureSlice> remote_captures;
    std::vector<Qwen35TargetCaptureSlice> * remote_captures_out =
        (feature_ring || remote_draft) ? &remote_captures : nullptr;
    if (!remote_shard.forward(base_pos, n_tokens_total, boundary,
                              logits_out != nullptr, last_tok,
                              argmax_out, logits_out, remote_captures_out,
                              ubatch)) {
        return false;
    }
    for (const auto & capture : remote_captures) {
        if (feature_ring &&
            !copy_host_capture_slice_to_draft_ring(
                *feature_ring, capture.capture_idx, capture.start_pos,
                capture.n_tokens, capture.data.data(), capture.data.size())) {
            std::fprintf(stderr,
                         "mixed target-split remote capture ring write failed capture=%d\n",
                         capture.capture_idx);
            return false;
        }
        if (remote_draft && remote_draft->active() &&
            !remote_draft->send_feature_slice(capture.capture_idx,
                                              capture.start_pos,
                                              capture.n_tokens,
                                              capture.data)) {
            std::fprintf(stderr,
                         "mixed target-split remote capture draft send failed capture=%d\n",
                         capture.capture_idx);
            return false;
        }
    }
    for (auto & shard : local_shards) {
        shard.cache.cur_pos = base_pos + n_tokens_total;
        shard.cache.last_tok = last_tok;
    }
    return true;
}

void free_qwen35_layer_split_shards(std::vector<Qwen35LayerSplitShard> & shards) {
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

} // namespace dflash::common
