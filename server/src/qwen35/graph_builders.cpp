#include "graph_builders.h"

#include "common/specla_mode.h"
#include "delta_net_specla.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash::common {

// ── build_layer_step ────────────────────────────────────────────

bool build_layer_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    ggml_tensor * act_in,
    ggml_tensor * act_out,
    int chunk_start,
    int n_tokens,
    int kv_start,
    bool with_mask,
    bool capture,
    int fa_window,
    int kq_stride_pad,
    bool kvflash,
    bool tree_mode) {
    if (kvflash) with_mask = true;
    step_graph_free(sg);

    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;

    sg.inp_embed = ggml_view_2d(sg.ctx, act_in,
        hidden, n_tokens,
        act_in->nb[1], (size_t)chunk_start * act_in->nb[1]);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    if (is_attn) {
        sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(sg.positions, "positions");
        ggml_set_input(sg.positions);

        if (with_mask) {
            int phys_ctx = cache.max_ctx;
            if (kvflash) {
                for (ggml_tensor * t : cache.attn_k) {
                    if (t) { phys_ctx = std::min(phys_ctx, (int)t->ne[1]); break; }
                }
            }
            // Size from the fixed physical capacity so gallocr doesn't grow
            // as kv_start advances. Under kvflash this is the resident pool.
            const int max_win_len = phys_ctx + n_tokens;
            const int kv_pad = align_up(max_win_len, kq_stride_pad);
            const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
            sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.attn_mask, "attn_mask");
            ggml_set_input(sg.attn_mask);
        }
        if (kvflash) {
            sg.kv_write_rows = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_I64,
                                                  n_tokens, w.n_head_kv);
            ggml_set_name(sg.kv_write_rows, "kv_write_rows");
            ggml_set_input(sg.kv_write_rows);
        }
    }

    if (tree_mode && !is_attn) {
        sg.parent_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(sg.parent_ids, "parent_ids");
        ggml_set_input(sg.parent_ids);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    ggml_tensor * layer_out = build_qwen35_layer(
        sg.ctx, sg.gf, w, cache, layer_idx,
        sg.inp_embed, sg.positions, sg.attn_mask,
        kv_start, n_tokens, capture, fa_window,
        /*q_tail_capture=*/nullptr, /*q_tail_start=*/0,
        sg.kv_write_rows, sg.parent_ids);
    if (!layer_out) return false;

    ggml_tensor * out_view = ggml_view_2d(sg.ctx, act_out,
        hidden, n_tokens,
        act_out->nb[1], (size_t)chunk_start * act_out->nb[1]);
    ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, layer_out, out_view));

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

bool build_layer_prefn_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    int kv_start,
    int n_tokens,
    bool with_mask,
    int fa_window,
    int kq_stride_pad,
    bool kvflash) {
    if (kvflash) with_mask = true;   // slot-space masking is mandatory on the pool
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);
    if (is_attn) {
        sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(sg.positions, "positions");
        ggml_set_input(sg.positions);
        if (with_mask) {
            // Mask width follows the PHYSICAL tensor capacity (pool-sized
            // under kvflash) so it agrees with the FA span clamp inside
            // build_full_attn_block.
            int phys_ctx = cache.max_ctx;
            for (ggml_tensor * t : cache.attn_k) {
                if (t) { phys_ctx = std::min(phys_ctx, (int)t->ne[1]); break; }
            }
            const int max_win_len = phys_ctx + n_tokens;
            const int kv_pad = align_up(max_win_len, kq_stride_pad);
            const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
            sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.attn_mask, "attn_mask");
            ggml_set_input(sg.attn_mask);
        }
        if (kvflash) {
            sg.kv_write_rows = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_I64,
                                                  n_tokens, w.n_head_kv);
            ggml_set_name(sg.kv_write_rows, "kv_write_rows");
            ggml_set_input(sg.kv_write_rows);
        }
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);
    QwenLayerPrefnOutputs go = build_qwen35_layer_prefn(
        sg.ctx, sg.gf, w, cache, layer_idx,
        sg.inp_embed, sg.positions, sg.attn_mask,
        kv_start, n_tokens, fa_window,
        sg.kv_write_rows,
        /*skip_gdn_intermediate=*/true);
    if (!go.residual || !go.post) return false;
    sg.ffn_residual = go.residual;
    sg.ffn_post = go.post;
    sg.moe_weights = go.moe_weights;
    if (go.moe_selected) {
        sg.moe_selected.assign((size_t)w.n_layer, nullptr);
        sg.moe_selected[(size_t)layer_idx] = go.moe_selected;
        ggml_set_output(go.moe_selected);
        ggml_build_forward_expand(sg.gf, go.moe_selected);
    }
    if (go.moe_weights) {
        ggml_set_output(go.moe_weights);
        ggml_build_forward_expand(sg.gf, go.moe_weights);
    }
    ggml_set_output(go.residual);
    ggml_build_forward_expand(sg.gf, go.residual);
    ggml_set_output(go.post);
    ggml_build_forward_expand(sg.gf, go.post);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// Full-layer graph for hybrid decode: pre-FFN (attention/DeltaNet + router) +
// MoE FFN (all selected experts via ggml_mul_mat_id) + shared FFN + residual.
// Outputs: sg.logits = layer_output, sg.moe_selected[layer_idx] = router picks.
// This is 1 graph compute per layer instead of 2 (pre-FFN + fused hot+shared).
bool build_hybrid_full_layer_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    int kv_start,
    int n_tokens,
    bool with_mask,
    int fa_window,
    int kq_stride_pad) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);
    if (is_attn) {
        sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(sg.positions, "positions");
        ggml_set_input(sg.positions);
        if (with_mask) {
            const int max_win_len = cache.max_ctx + n_tokens;
            const int kv_pad = align_up(max_win_len, kq_stride_pad);
            const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
            sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.attn_mask, "attn_mask");
            ggml_set_input(sg.attn_mask);
        }
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    ggml_tensor * moe_selected = nullptr;
    ggml_tensor * layer_out = build_qwen35_layer(
        sg.ctx, sg.gf, w, cache, layer_idx,
        sg.inp_embed, sg.positions, sg.attn_mask,
        kv_start, n_tokens, /*capture=*/false, fa_window,
        /*q_tail_capture=*/nullptr, /*q_tail_start=*/0,
        &moe_selected);
    if (!layer_out) return false;

    // Use hidden_input as the layer output tensor (repurpose field)
    sg.hidden_input = layer_out;
    ggml_set_output(layer_out);
    ggml_build_forward_expand(sg.gf, layer_out);

    if (moe_selected) {
        sg.moe_selected.assign((size_t)w.n_layer, nullptr);
        sg.moe_selected[(size_t)layer_idx] = moe_selected;
        ggml_set_output(moe_selected);
        ggml_build_forward_expand(sg.gf, moe_selected);
    }

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ── build_target_step ───────────────────────────────────────────

bool build_target_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    bool with_mask,
    bool capture,
    bool capture_delta_intermediate,
    int fa_window,
    int logits_tail_rows,
    int kq_stride_pad,
    bool capture_moe_router,
    bool kvflash_mask,
    bool capture_qk,
    bool paged_attention,
    int n_seqs,
    int seq_slot,
    int paged_max_kv_len,
    int n_prefill_tokens,
    const QwenPrefillSegment * prefill_segments,
    int n_prefill_segments,
    int n_logits_rows,
    bool compact_slots,
    bool stable_chain_verify) {
    // A strict Ling AR graph has no position baked into its operations: the
    // token embedding, RoPE position, causal mask, and KV destination row are
    // all inputs. Its MLA read span changes only at 256-token boundaries, so
    // retain the live graph inside that bucket instead of rebuilding the same
    // thousands of tensor descriptors on every token.
    static const bool g_bailing_ar_graph_reuse = []() {
        const char * value = std::getenv("DFLASH_BAILING_AR_GRAPH_REUSE");
        return value == nullptr || value[0] != '0';
    }();
    static const bool g_bailing_verify_graph_reuse = []() {
        const char * value = std::getenv("DFLASH_BAILING_VERIFY_GRAPH_REUSE");
        return value == nullptr || value[0] != '0';
    }();
    const bool reusable_bailing_ar =
        g_bailing_ar_graph_reuse &&
        w.is_bailingmoe3 &&
        kv_start >= 0 &&
        n_tokens == 1 &&
        with_mask &&
        !capture &&
        !capture_delta_intermediate &&
        fa_window == 0 &&
        logits_tail_rows == 0 &&
        kq_stride_pad == 256 &&
        !capture_moe_router &&
        !kvflash_mask &&
        !capture_qk &&
        !paged_attention &&
        n_seqs == 1 &&
        seq_slot == 0 &&
        paged_max_kv_len == 0 &&
        n_prefill_tokens == 0 &&
        prefill_segments == nullptr &&
        n_prefill_segments == 0 &&
        n_logits_rows == 0 &&
        !compact_slots;
    const int bailing_ar_kv_bucket = reusable_bailing_ar
        ? align_up(kv_start + n_tokens, 256)
        : 0;
    if (reusable_bailing_ar && sg.ctx && sg.gf &&
        sg.bailing_ar_kv_bucket == bailing_ar_kv_bucket) {
        return true;
    }

    // Ling's root-inclusive DSpark verify has a fixed width. KV rows, RoPE
    // positions, mask contents, embeddings, and feature-ring destinations are
    // all inputs, while MLA's read span changes only at 256-token boundaries.
    // Retain the live graph inside a bucket: this avoids rebuilding thousands
    // of descriptors and lets ggml-cuda replay the already captured graph.
    // Both rollback strategies are reusable. The checkpoint path captures
    // per-token recurrent intermediates, while optimistic exact verification
    // keeps only the durable pre-verify snapshot and restores/replays on a
    // mismatch; neither bakes kv_start into the graph inside an MLA bucket.
    // SpecLA's schedule and factor bank are included in the key below. Its
    // alternating banks rebuild host metadata, but separate salted CUDA-graph
    // keys retain both captured executions.
    const bool reusable_bailing_verify =
        g_bailing_verify_graph_reuse &&
        stable_chain_verify &&
        w.is_bailingmoe3 &&
        cache.target_feat != nullptr &&
        kv_start >= 0 &&
        n_tokens > 1 &&
        with_mask &&
        capture &&
        fa_window == 0 &&
        logits_tail_rows == 0 &&
        kq_stride_pad == 256 &&
        !capture_moe_router &&
        !kvflash_mask &&
        !capture_qk &&
        !paged_attention &&
        n_seqs == 1 &&
        seq_slot == 0 &&
        paged_max_kv_len == 0 &&
        n_prefill_tokens == 0 &&
        prefill_segments == nullptr &&
        n_prefill_segments == 0 &&
        n_logits_rows == 0 &&
        !compact_slots;
    const int bailing_verify_kv_bucket = reusable_bailing_verify
        ? align_up(kv_start + n_tokens, 256)
        : 0;
    const bool bailing_verify_specla =
        reusable_bailing_verify && specla_enabled() &&
        !cache.factor_k.empty();
    const int bailing_verify_pending_count = bailing_verify_specla
        ? cache.specla_pending_count : -1;
    const int bailing_verify_pending_bank = bailing_verify_specla
        ? cache.specla_pending_bank : -1;
    if (reusable_bailing_verify && sg.ctx && sg.gf &&
        sg.bailing_verify_kv_bucket == bailing_verify_kv_bucket &&
        sg.bailing_verify_width == n_tokens &&
        sg.bailing_verify_pending_count == bailing_verify_pending_count &&
        sg.bailing_verify_pending_bank == bailing_verify_pending_bank &&
        sg.bailing_verify_capture_delta == capture_delta_intermediate) {
        return true;
    }

    step_graph_free(sg);

    // Compact n_seqs is a decode graph bucket width, not the physical
    // slot count. active_slot_ids maps live rows to cache columns and uses -1
    // for padding, so a valid bucket may be wider than cache.n_seq_slots.
    const bool invalid_compact_width = n_seqs < 1 || n_seqs > 64;

    // Concurrent prefill rows read the pool through the ragged paged path
    // (per-row seq ids and causal positions, no mask). Fused steps put the
    // chunk rows first and the compact decode rows after; a prefill-only
    // step has no decode rows.
    if (n_prefill_tokens > n_tokens) return false;
    const bool fused = n_prefill_tokens > 0 && n_tokens > n_prefill_tokens;
    const bool prefill_only =
        n_prefill_tokens > 0 && n_tokens == n_prefill_tokens;
    if (fused && (!paged_attention ||
                  n_tokens != n_prefill_tokens + n_seqs ||
                  !compact_slots || invalid_compact_width)) {
        return false;
    }
    if (prefill_only &&
        (!paged_attention || compact_slots || n_seqs != 1 || with_mask)) {
        return false;
    }
    // Prefill rows always arrive split into per-prompt segments: dense, in
    // order, totalling n_prefill_tokens.
    if ((n_prefill_tokens > 0) !=
        (prefill_segments != nullptr && n_prefill_segments > 0)) {
        return false;
    }
    int segment_total = 0;
    for (int i = 0; i < n_prefill_segments; ++i) {
        const QwenPrefillSegment & pf = prefill_segments[i];
        if (pf.n_tokens < 1 || pf.token_offset != segment_total ||
            pf.seq_slot < 0 || pf.seq_slot >= cache.n_seq_slots) {
            return false;
        }
        segment_total += pf.n_tokens;
    }
    if (segment_total != n_prefill_tokens) return false;
    if (n_logits_rows > 0 && n_prefill_tokens == 0) return false;

    // Persistent thread_local arena: rebuilt step graphs land at identical
    // addresses, keeping the ggml-cuda CUDA-graph cache key (nodes[0]) and
    // every node property stable across AR decode steps -> captured graph
    // replays instead of re-launching every kernel. Pairs with the
    // step-invariant set_rows KV write (kv_write_rows) below.
    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    static thread_local std::vector<uint8_t> g_step_arena;
    if (g_step_arena.size() < ip.mem_size) g_step_arena.resize(ip.mem_size);
    ip.mem_buffer = g_step_arena.data();
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    // ggml-cuda keys its graph cache by nodes[0]. Salting the metadata
    // allocation shifts node addresses deterministically so each complete
    // graph shape keeps an independent captured graph while sharing this
    // single 512 MiB metadata arena.
    int decode_key = 0;
    if (compact_slots) {
        static constexpr int decode_buckets[] = {
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
        };
        bool found = false;
        for (int i = 0; i < (int)(sizeof(decode_buckets) /
                                  sizeof(decode_buckets[0])); ++i) {
            if (decode_buckets[i] == n_seqs) {
                decode_key = i + 1;
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    int graph_key_slot = decode_key;
    if (reusable_bailing_verify) {
        // Keep chain-verify captures distinct from AR/paged shapes. Hash the
        // 256-token MLA bucket plus SpecLA's alternating bank and pending
        // width into bounded metadata slots; ggml-cuda still validates node
        // properties on a collision.
        uint32_t key = (uint32_t)(bailing_verify_kv_bucket / 256);
        key = key * 33u + (uint32_t)n_tokens;
        key = key * 33u + (uint32_t)(bailing_verify_pending_count + 1);
        key = key * 3u + (uint32_t)(bailing_verify_pending_bank + 1);
        key = key * 2u + (capture_delta_intermediate ? 1u : 0u);
        graph_key_slot = 128 + (int)(key % 128u);
    }
    if (paged_attention && paged_max_kv_len > 0) {
        // Paged graphs differ by their padded launch bound. Packed recurrent
        // graphs also differ by total width and every ragged segment length.
        // Hash complete shapes into a fixed slot set: ggml-cuda still checks
        // node properties on a collision, while host and captured-graph cache
        // growth remain bounded for long-running ragged workloads.
        if (n_prefill_tokens > 0) {
            const int prefill_logit_rows = compact_slots
                ? n_logits_rows - n_seqs
                : n_logits_rows;
            if (prefill_logit_rows < 0 ||
                prefill_logit_rows > n_prefill_segments) {
                return false;
            }
        }
        std::vector<int> shape{
            decode_key, n_tokens, n_prefill_tokens, n_prefill_segments,
            n_logits_rows, compact_slots ? 1 : 0,
            (paged_max_kv_len + 255) / 256,
        };
        shape.reserve(shape.size() + (size_t)n_prefill_segments);
        for (int i = 0; i < n_prefill_segments; ++i) {
            shape.push_back(prefill_segments[i].n_tokens);
        }
        uint64_t hash = 1469598103934665603ull;
        for (int value : shape) {
            hash ^= (uint32_t)value;
            hash *= 1099511628211ull;
        }
        static constexpr int kPagedGraphSlots = 64;
        graph_key_slot = 32 + (int)(hash % kPagedGraphSlots);
    }
    for (int i = 0; i < graph_key_slot; ++i) {
        (void)ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 1);
    }

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        // Most Qwen paths retain a max-context mask so the gallocr buffer
        // never grows during generation. Ling's compressed MLA already
        // changes its K/V view at each 256-token bucket, however, so a 32K
        // mask buys no extra CUDA-graph reuse and uploads ~2 MiB per AR step.
        // Match Ling's mask to its active cache bucket instead.
        // kvflash mode: the physical span is the (smaller) pool capacity of
        // the attention tensors, so size the mask from those instead.
        int phys_ctx = cache.max_ctx;
        for (auto * t : cache.attn_k) {
            if (t) { phys_ctx = std::min(phys_ctx, (int)t->ne[1]); break; }
        }
        const int max_win_len = w.is_bailingmoe3 && !kvflash_mask
            ? kv_start + n_tokens
            : phys_ctx + n_tokens;
        const int kv_pad = align_up(max_win_len, kq_stride_pad);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
    }

    ggml_tensor * paged_block_table = nullptr;
    ggml_tensor * paged_kv_seq_lens = nullptr;
    if (paged_attention) {
        if (n_prefill_tokens == 0) {
            // Classic paged decode is one physical sequence and one token.
            // Concurrent decode always carries an explicit row-to-slot map.
            if (compact_slots) {
                if (n_tokens != n_seqs || invalid_compact_width) return false;
            } else if (n_tokens != 1 || n_seqs != 1) {
                return false;
            }
            if (with_mask) return false;
        } else if (with_mask) {
            // Causality for prefill rows comes from the kernel's per-row
            // position clamp, never from a mask.
            return false;
        }
        if (fa_window != 0) return false;
        // The paging metadata lives in the persistent target cache (next to
        // the K/V pool), not as gallocr graph inputs: contents survive graph
        // execution and rebuilds, so the backend uploads only what changed
        // between decode steps.
        if (!cache.paged_block_table || !cache.paged_kv_seq_lens) return false;
        if ((int)cache.paged_block_table->ne[1] != cache.n_seq_slots ||
            (int)cache.paged_kv_seq_lens->ne[0] != cache.n_seq_slots) {
            return false;
        }
        paged_block_table = cache.paged_block_table;
        paged_kv_seq_lens = cache.paged_kv_seq_lens;
    }
    if (paged_attention && compact_slots) {
        sg.active_slot_ids =
            ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_seqs);
        sg.state_slot_ids =
            ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_seqs);
        ggml_set_name(sg.active_slot_ids, "active_slot_ids");
        ggml_set_name(sg.state_slot_ids, "state_slot_ids");
        ggml_set_input(sg.active_slot_ids);
        ggml_set_input(sg.state_slot_ids);
    }
    if (paged_attention && n_prefill_tokens > 0) {
        // Ragged read metadata: every row of the batch (chunk rows and
        // decode rows alike) names its block-table column and its inclusive
        // logical position.
        sg.paged_query_seq_ids =
            ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
        sg.paged_query_positions =
            ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(sg.paged_query_seq_ids, "paged_query_seq_ids");
        ggml_set_name(sg.paged_query_positions, "paged_query_positions");
        ggml_set_input(sg.paged_query_seq_ids);
        ggml_set_input(sg.paged_query_positions);
    }
    if (n_logits_rows > 0) {
        sg.logits_row_indices =
            ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_logits_rows);
        ggml_set_name(sg.logits_row_indices, "logits_row_indices");
        ggml_set_input(sg.logits_row_indices);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    // Step-invariant KV write: only when topology can't vary per step.
    // DFLASH_QWEN35_NO_KVPAD=1 restores the legacy cpy append + exact-length
    // FA span (per-step node properties -> no CUDA-graph replay).
    static const bool g_no_kvpad = (std::getenv("DFLASH_QWEN35_NO_KVPAD") != nullptr);
    // kvflash_mask: kvflash mode. The mask carries pool slot validity
    // (uploaded by the caller before EVERY compute — the input's buffer
    // region is reused by graph execution) and set_rows carries per-token
    // physical slots, so the slot-mapped write stays active for masked,
    // multi-token, and feature-capturing forwards (decode AND spec verify).
    const bool bailing_verify_rows =
        w.is_bailingmoe3 && stable_chain_verify && capture && with_mask &&
        n_tokens > 1 && fa_window == 0;
    const bool use_kv_write_rows =
        paged_attention ||
        bailing_verify_rows ||
        (!g_no_kvpad && !capture_delta_intermediate &&
         (kvflash_mask
              ? (fa_window == 0)
              : (n_tokens == 1 && fa_window == 0 && !capture &&
                 (!with_mask || w.is_bailingmoe3))));
    if (use_kv_write_rows) {
        sg.kv_write_rows = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_I64,
                                              n_tokens, w.n_head_kv);
        ggml_set_name(sg.kv_write_rows, "kv_write_rows");
        ggml_set_input(sg.kv_write_rows);
    }
    if (reusable_bailing_verify) {
        sg.target_feat_rows =
            ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(sg.target_feat_rows, "target_feat_rows");
        ggml_set_input(sg.target_feat_rows);
    }

    SpecLAHLDSchedule hld_schedule;
    if (capture_delta_intermediate && specla_enabled() && !cache.factor_k.empty()) {
        std::vector<int32_t> parents((size_t)n_tokens);
        for (int t = 0; t < n_tokens; ++t) parents[(size_t)t] = t - 1;
        hld_schedule = make_specla_hld_schedule(
            parents.data(), n_tokens, cache.specla_pending_count);
        if (hld_schedule.packed.empty()) return false;
        sg.specla_hld = ggml_new_tensor_1d(
            sg.ctx, GGML_TYPE_I32, hld_schedule.packed.size());
        ggml_set_name(sg.specla_hld, "specla_hld");
        ggml_set_input(sg.specla_hld);
    }

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.capture_layers             = capture;
    gi.capture_delta_intermediate = capture_delta_intermediate;
    gi.capture_moe_router         = capture_moe_router;
    static const bool g_bailing_fused_conv_state = []() {
        const char * value =
            std::getenv("DFLASH_BAILING_FUSED_CONV_STATE");
        return value == nullptr || value[0] != '0';
    }();
    static const bool g_bailing_fused_grouped_router = []() {
        const char * value =
            std::getenv("DFLASH_BAILING_FUSED_GROUPED_ROUTER");
        return value == nullptr || value[0] != '0';
    }();
    static const bool g_bailing_fused_grouped_verify_router = []() {
        const char * value =
            std::getenv("DFLASH_BAILING_FUSED_GROUPED_VERIFY_ROUTER");
        // The multi-row kernel is mathematically equivalent on direct CUDA
        // tests, but BF16 Ling routing contains model-real ties that can alter
        // the greedy trajectory. Keep verification opt-in until those tie
        // semantics match the generic segmented sort exactly.
        return value != nullptr && value[0] != '0';
    }();
    const char * backend_name = ggml_backend_name(backend);
    const bool reusable_bailing_ar_cuda =
        reusable_bailing_ar && backend_name &&
        std::strstr(backend_name, "CUDA") != nullptr;
    const bool reusable_bailing_verify_cuda =
        reusable_bailing_verify && backend_name &&
        std::strstr(backend_name, "CUDA") != nullptr;
    gi.bailing_fuse_conv_state =
        reusable_bailing_ar_cuda && g_bailing_fused_conv_state;
    gi.bailing_fuse_grouped_router =
        g_bailing_fused_grouped_router &&
        (reusable_bailing_ar_cuda ||
         (reusable_bailing_verify_cuda &&
          g_bailing_fused_grouped_verify_router));
    gi.fa_window                  = fa_window;
    gi.logits_tail_rows           = logits_tail_rows;
    gi.kv_write_rows              = sg.kv_write_rows;
    gi.target_feat_rows           = sg.target_feat_rows;
    gi.paged_block_table          = paged_block_table;
    gi.paged_kv_seq_lens          = paged_kv_seq_lens;
    gi.active_slot_ids            = sg.active_slot_ids;
    gi.state_slot_ids             = sg.state_slot_ids;
    gi.q_capture                  = capture_qk;
    gi.n_seqs                     = n_seqs;
    gi.seq_slot                   = seq_slot;
    gi.paged_max_kv_len           = paged_max_kv_len;
    gi.n_prefill_tokens           = n_prefill_tokens;
    gi.paged_query_seq_ids        = sg.paged_query_seq_ids;
    gi.paged_query_positions      = sg.paged_query_positions;
    gi.logits_row_indices         = sg.logits_row_indices;
    gi.prefill_segments           = prefill_segments;
    gi.n_prefill_segments         = n_prefill_segments;
    gi.specla_m_strict            = sg.specla_m_strict;
    gi.specla_m_incl              = sg.specla_m_incl;
    gi.specla_m_eye               = sg.specla_m_eye;
    gi.specla_hld                 = sg.specla_hld;
    gi.specla_n_chains            = hld_schedule.n_chains;
    gi.specla_n_waves             = hld_schedule.n_waves;
    gi.specla_n_boundaries        = hld_schedule.n_boundaries;
    gi.specla_max_parallel_chains = hld_schedule.max_parallel_chains;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.hidden_states = go.hidden_states;
    sg.delta_captures = std::move(go.delta_captures);
    sg.moe_selected = std::move(go.moe_selected);
    ggml_set_output(sg.logits);
    if (w.mtp.enabled && sg.hidden_states) {
        ggml_set_name(sg.hidden_states, "target_final_hidden");
        ggml_set_output(sg.hidden_states);
        ggml_build_forward_expand(sg.gf, sg.hidden_states);
    }

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "chain_verify_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) return false;
    if (sg.specla_hld) {
        ggml_backend_tensor_set(sg.specla_hld, hld_schedule.packed.data(), 0,
            hld_schedule.packed.size()*sizeof(int32_t));
    }
    sg.bailing_ar_kv_bucket = bailing_ar_kv_bucket;
    sg.bailing_verify_kv_bucket = bailing_verify_kv_bucket;
    sg.bailing_verify_width = reusable_bailing_verify ? n_tokens : 0;
    sg.bailing_verify_pending_count = bailing_verify_pending_count;
    sg.bailing_verify_pending_bank = bailing_verify_pending_bank;
    sg.bailing_verify_capture_delta =
        reusable_bailing_verify && capture_delta_intermediate;
    return true;
}

bool build_bailingmoe3_mtp_step(
        StepGraph & sg,
        const TargetWeights & w,
        TargetCache & mtp_cache,
        ggml_backend_t backend,
        int kv_start,
        int n_tokens,
        int logits_tail_rows,
        int kq_stride_pad) {
    if (!w.is_bailingmoe3 || !w.mtp.enabled || n_tokens <= 0 ||
        kv_start < 0 || kv_start + n_tokens > mtp_cache.max_ctx) {
        return false;
    }
    step_graph_free(sg);

    // This predictor graph lives alongside the target graph, so it needs its
    // own stable metadata arena rather than build_target_step's thread-local
    // arena.  Stable tensor addresses let ggml-cuda replay the two-token graph.
    constexpr size_t kMtpMetaBytes = 64 * 1024 * 1024;
    if (sg.meta_arena.size() < kMtpMetaBytes) {
        sg.meta_arena.resize(kMtpMetaBytes);
    }
    ggml_init_params ip{};
    ip.mem_size = sg.meta_arena.size();
    ip.mem_buffer = sg.meta_arena.data();
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(
        sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.hidden_input = ggml_new_tensor_3d(
        sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.positions = ggml_new_tensor_1d(
        sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    // Match the predictor mask to the same 256-token cache bucket consumed
    // by compressed MLA. A max-context mask uploaded ~2 MiB on every decode
    // step even when only a short prefix was visible, and did not improve
    // graph reuse because the K/V view already changes at bucket boundaries.
    const int kv_pad = align_up(kv_start + n_tokens, kq_stride_pad);
    const int query_rows = logits_tail_rows > 0
        ? std::min(logits_tail_rows, n_tokens)
        : n_tokens;
    const int q_pad = align_up(query_rows, KQ_MASK_PAD);
    sg.attn_mask = ggml_new_tensor_2d(
        sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
    sg.kv_write_rows = ggml_new_tensor_2d(
        sg.ctx, GGML_TYPE_I64, n_tokens, 1);

    ggml_set_name(sg.inp_embed, "bailing_mtp_embed");
    ggml_set_name(sg.hidden_input, "bailing_mtp_target_hidden");
    ggml_set_name(sg.positions, "bailing_mtp_positions");
    ggml_set_name(sg.attn_mask, "bailing_mtp_mask");
    ggml_set_name(sg.kv_write_rows, "bailing_mtp_kv_rows");
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.hidden_input);
    ggml_set_input(sg.positions);
    ggml_set_input(sg.attn_mask);
    ggml_set_input(sg.kv_write_rows);

    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);
    QwenGraphOutputs out = build_bailingmoe3_mtp_graph(
        sg.ctx, sg.gf, w, mtp_cache, sg.inp_embed, sg.hidden_input,
        sg.positions, sg.attn_mask, sg.kv_write_rows,
        kv_start, n_tokens, logits_tail_rows);
    if (!out.logits) return false;
    sg.logits = out.logits;
    sg.hidden_states = out.hidden_states;
    ggml_set_output(sg.logits);

    // Multi-step MTP feeds the predictor's final hidden row back into the
    // same layer for the next draft token.  Preserve only that row across
    // graph execution; the full batch remains transient.
    const int hidden_rows = static_cast<int>(out.hidden_states->ne[1]);
    sg.hidden_tail = ggml_view_2d(
        sg.ctx, out.hidden_states, hidden, 1,
        out.hidden_states->nb[1],
        static_cast<size_t>(hidden_rows - 1) * out.hidden_states->nb[1]);
    ggml_set_name(sg.hidden_tail, "bailing_mtp_hidden_tail");
    ggml_set_output(sg.hidden_tail);
    ggml_build_forward_expand(sg.gf, sg.hidden_tail);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "bailing_mtp_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ── build_target_step_tree ──────────────────────────────────────

bool build_target_step_tree(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    int fa_window,
    int kq_stride_pad,
    const SpecLAHLDSchedule * hld_schedule) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    const int max_win_len = cache.max_ctx + n_tokens;
    const int kv_pad = align_up(max_win_len, kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_name(sg.attn_mask, "attn_mask");
    ggml_set_input(sg.attn_mask);

    sg.parent_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.parent_ids, "parent_ids");
    ggml_set_input(sg.parent_ids);

    // SpecLA tree verify: ancestor masks over the DFS-ordered nodes replace
    // the sequential kernel's parent_ids state fanout (parent_ids still
    // steers the tree conv). Host-filled by verify_tree from tree.parents.
    if (specla_enabled() && !cache.factor_k.empty() && hld_schedule) {
        if (hld_schedule->n_nodes != n_tokens || hld_schedule->packed.empty()) {
            return false;
        }
        sg.specla_hld = ggml_new_tensor_1d(
            sg.ctx, GGML_TYPE_I32, hld_schedule->packed.size());
        ggml_set_name(sg.specla_hld, "specla_hld");
        ggml_set_input(sg.specla_hld);
    } else if (specla_enabled() && !cache.factor_k.empty()) {
        sg.specla_m_strict = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, n_tokens, n_tokens);
        sg.specla_m_incl   = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, n_tokens, n_tokens);
        sg.specla_m_eye    = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, n_tokens, n_tokens);
        ggml_set_name(sg.specla_m_strict, "specla_m_strict");
        ggml_set_name(sg.specla_m_incl,   "specla_m_incl");
        ggml_set_name(sg.specla_m_eye,    "specla_m_eye");
        ggml_set_input(sg.specla_m_strict);
        ggml_set_input(sg.specla_m_incl);
        ggml_set_input(sg.specla_m_eye);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.fa_window                  = fa_window;
    gi.capture_layers             = true;
    gi.capture_delta_intermediate = true;
    gi.parent_ids                 = sg.parent_ids;
    gi.specla_m_strict            = sg.specla_m_strict;
    gi.specla_m_incl              = sg.specla_m_incl;
    gi.specla_m_eye               = sg.specla_m_eye;
    gi.specla_hld                 = sg.specla_hld;
    gi.specla_n_chains            = hld_schedule ? hld_schedule->n_chains : 0;
    gi.specla_n_waves             = hld_schedule ? hld_schedule->n_waves : 0;
    gi.specla_n_boundaries        = hld_schedule ? hld_schedule->n_boundaries : 0;
    gi.specla_max_parallel_chains = hld_schedule ? hld_schedule->max_parallel_chains : 0;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    ggml_set_output(sg.logits);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "tree_verify_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) return false;
    if (sg.specla_hld) {
        ggml_backend_tensor_set(sg.specla_hld, hld_schedule->packed.data(), 0,
            hld_schedule->packed.size()*sizeof(int32_t));
    }
    return true;
}


// ── build_lm_head_projection_step ───────────────────────────────

bool build_lm_head_projection_step(
    StepGraph & sg,
    const TargetWeights & w,
    ggml_backend_t backend,
    int n_tokens) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.hidden_input = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.hidden_input, "draft_hidden_for_lm_head");
    ggml_set_input(sg.hidden_input);

    sg.gf = ggml_new_graph_custom(sg.ctx, 1024, false);
    sg.logits = ggml_mul_mat(sg.ctx, w.output, sg.hidden_input);
    ggml_set_name(sg.logits, "draft_projected_logits");
    ggml_set_output(sg.logits);
    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "draft_projected_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

}  // namespace dflash::common
