// ─── Fused batched VERIFY graph (n_tokens = q in [2,4]) ─────────────────
// One whole-model graph per (q, flush, padded-comp) shape: batched attention
// (causal mask input) + batched MoE, per-token fused HC chains, in-graph
// drafter-feature capture and q-wide logits. The spec loop guarantees a
// ratio-4 boundary can only sit at the batch's LAST token (dynamic q), so the
// compressor pools at most once per step, exactly like an AR step at the last
// position. Enable with DFLASH_DS4_FUSED_VERIFY=1.

// tensor_set that tolerates inputs the graph never consumed (gallocr leaves
// them unallocated, e.g. comp-emission bundles in non-flush graphs).
static void ds4_fv_set(ggml_tensor * t, const void * data, size_t nbytes) {
    if (t && t->buffer) ggml_backend_tensor_set(t, data, 0, nbytes);
}
static bool ds4_fused_verify_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ds4_env_flag("DFLASH_DS4_FUSED_VERIFY") ? 1 : 0;
    }
    return enabled == 1;
}

struct Ds4FusedVerifyCache {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    bool disabled = false;
    uint64_t counter = 0;
    std::array<DeepSeek4FusedDecodeGraph, 12> slots;
    // verify-specific inputs/outputs per slot (parallel arrays, same index)
    struct Extra {
        ggml_tensor * pos_q = nullptr;    // i32 [q]
        ggml_tensor * neg_q = nullptr;    // i32 [q]
        ggml_tensor * rawrows = nullptr;  // i64 [1,q]
        ggml_tensor * ape4 = nullptr;     // i32 [q]
        ggml_tensor * ape128 = nullptr;   // i32 [q]
        ggml_tensor * st4 = nullptr;      // i64 [1,q]
        ggml_tensor * st128 = nullptr;    // i64 [1,q]
        ggml_tensor * capture = nullptr;  // f32 [n_embd*ncap*q], order [ci][t]
        int q = 0;
        void reset() { *this = Extra{}; }
    };
    std::array<Extra, 12> extra;
    void destroy() {
        for (auto & s : slots) s.destroy();
        for (auto & e : extra) e.reset();
    }
};

static Ds4FusedVerifyCache fused_verify_graph_cache;

static bool ds4_build_fused_verify_graph(
        DeepSeek4FusedDecodeCache & mc,          // fn mirrors (shared with AR fused)
        DeepSeek4FusedDecodeGraph & fg,
        Ds4FusedVerifyCache::Extra & ex,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights,
        const std::vector<HashRoutingTableCpu> & hash_tables,
        int kv_start,
        int q,
        bool have_token_ids,
        const std::vector<int> & capture_ids,
        std::vector<int64_t> && shape_key) {
    step_graph_free(fg.sg);
    fg.reset_nodes();
    ex.reset();
    fg.hash_ids.assign((size_t) w.n_layer, nullptr);
    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;
    const int token_pos = kv_start + q - 1;   // last batch position

    const size_t arena_size = 256u * 1024 * 1024;
    if (fg.sg.meta_arena.size() < arena_size) fg.sg.meta_arena.resize(arena_size);
    ggml_init_params params{};
    params.mem_size = fg.sg.meta_arena.size();
    params.mem_buffer = fg.sg.meta_arena.data();
    params.no_alloc = true;
    fg.sg.ctx = ggml_init(params);
    if (!fg.sg.ctx) return false;
    ggml_context * ctx = fg.sg.ctx;
    fg.sg.gf = ggml_new_graph_custom(ctx, 65536, false);
    ggml_cgraph * gf = fg.sg.gf;

    fg.inp_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, q);
    ggml_set_input(fg.inp_embed);
    ex.q = q;
    ex.pos_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);  ggml_set_input(ex.pos_q);
    ex.neg_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);  ggml_set_input(ex.neg_q);
    ex.rawrows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q); ggml_set_input(ex.rawrows);
    ex.ape4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);   ggml_set_input(ex.ape4);
    ex.ape128 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q); ggml_set_input(ex.ape128);
    ex.st4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q); ggml_set_input(ex.st4);
    ex.st128 = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q); ggml_set_input(ex.st128);
    // comp emission scalars per layer: i32 {ape_row(unused batched), comp_pos},
    // i64 {comp_row}; reuse the decode-style bundles (2 i32 + 1 i64 per layer).
    fg.i32_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * (int64_t) w.n_layer);
    ggml_set_input(fg.i32_bundle);
    fg.i64_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1 * (int64_t) w.n_layer);
    ggml_set_input(fg.i64_bundle);

    // mask bundle: per layer [n_swa + padded + q] rows × q tokens
    int64_t mask_total = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        int padded = 0;
        if (ratio > 0 && cache.layers[(size_t) il].comp_kv) {
            const int n_comp = ds4_comp_rows_used(cache.layers[(size_t) il].comp_kv,
                                                  cache.layers[(size_t) il].n_comp, ratio, token_pos);
            padded = ds4_padded_comp_rows(n_comp, (int) cache.layers[(size_t) il].comp_kv->ne[1]);
        }
        mask_total += (int64_t) (w.n_swa + padded + q) * q;
    }
    fg.mask_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mask_total);
    ggml_set_input(fg.mask_bundle);
    int64_t mask_off = 0;

    // per-token HC streams
    std::vector<ggml_tensor *> hc_cur(q);
    for (int t = 0; t < q; ++t) {
        ggml_tensor * col = ggml_view_2d(ctx, fg.inp_embed, n_embd, 1,
                                         fg.inp_embed->nb[1], (size_t) t * fg.inp_embed->nb[1]);
        hc_cur[(size_t) t] = ggml_repeat_4d(ctx, col, n_embd, n_hc, 1, 1);
    }

    std::vector<ggml_tensor *> capture_pieces(
        capture_ids.size() * (size_t) q, nullptr);   // order [ci][t]

    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];
        const int ratio = (int) w.compress_ratios[il];

        // ── HC pre (attention), per token ──
        std::vector<ggml_tensor *> hc_flat(q), split_attn(q);
        ggml_tensor * attn_in = nullptr;
        for (int t = 0; t < q; ++t) {
            hc_flat[(size_t) t] = ggml_reshape_1d(ctx, hc_cur[(size_t) t], (int64_t) n_embd * n_hc);
            ggml_tensor * working = ds4_build_fused_hc_pre(ctx, w, hc_flat[(size_t) t],
                                                           mc.fn_attn_f16[(size_t) il], L.hc_attn_base,
                                                           hlw.attn, &split_attn[(size_t) t]);
            if (!working) return false;
            ggml_tensor * w2 = ggml_reshape_2d(ctx, working, n_embd, 1);
            attn_in = attn_in ? ggml_concat(ctx, attn_in, w2, 1) : w2;
        }

        // ── Batched attention ──
        DeepSeek4AttentionGraphInputs ain{};
        ain.rope_pos = ex.pos_q;
        ain.neg_pos  = ex.neg_q;
        ain.raw_kv_rows = ex.rawrows;
        if (ratio > 0) {
            ain.attn_ape_row = (ratio == 4) ? ex.ape4 : ex.ape128;
            ain.attn_state_rows = (ratio == 4) ? ex.st4 : ex.st128;
            ain.attn_comp_pos = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 2 + 1) * sizeof(int32_t));
            ain.attn_comp_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                              (size_t) il * sizeof(int64_t));
        }
        if (ratio == 4) {
            ain.index_ape_row = ex.ape4;
            ain.index_state_rows = ex.st4;
            ain.index_comp_pos = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 2 + 1) * sizeof(int32_t));
            ain.index_comp_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                               (size_t) il * sizeof(int64_t));
        }
        int padded = 0;
        if (ratio > 0 && lc.comp_kv) {
            const int n_comp = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos);
            padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
        }
        const int64_t n_attn = (int64_t) w.n_swa + padded + q;
        ain.attn_row_mask = ggml_view_2d(ctx, fg.mask_bundle, n_attn, q,
                                         n_attn * sizeof(float), (size_t) mask_off * sizeof(float));
        ain.padded_comp = padded;
        mask_off += n_attn * q;

        std::vector<DeepSeek4I32InputBinding> i32b;
        std::vector<DeepSeek4I32ArrayBinding> i32ab;
        std::vector<DeepSeek4I64ArrayBinding> i64ab;
        ggml_tensor * normed = build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);
        ggml_tensor * attn_out = build_mla_attention(ctx, gf, normed, w, L, lc, il,
                                                     kv_start, q, &ain, i32b, i32ab, i64ab);
        if (!attn_out) return false;
        if (!i32b.empty() || !i32ab.empty() || !i64ab.empty()) {
            std::fprintf(stderr, "[ds4-fused-verify] layer %d dynamic bindings; cannot fuse\n", il);
            return false;
        }

        // ── HC post (attention) + HC pre (FFN), per token ──
        ggml_tensor * ffn_in = nullptr;
        std::vector<ggml_tensor *> split_ffn(q);
        for (int t = 0; t < q; ++t) {
            ggml_tensor * ao = ggml_view_2d(ctx, attn_out, n_embd, 1,
                                            attn_out->nb[1], (size_t) t * attn_out->nb[1]);
            ggml_tensor * ao_flat = ggml_reshape_1d(ctx, ggml_cont(ctx, ao), n_embd);
            ggml_tensor * hc_next = ggml_ds4_hc_post(ctx, hc_flat[(size_t) t], ao_flat,
                                                     split_attn[(size_t) t], n_hc);
            hc_cur[(size_t) t] = ggml_reshape_2d(ctx, hc_next, n_embd, n_hc);
            hc_flat[(size_t) t] = ggml_reshape_1d(ctx, hc_cur[(size_t) t], (int64_t) n_embd * n_hc);
            ggml_tensor * fworking = ds4_build_fused_hc_pre(ctx, w, hc_flat[(size_t) t],
                                                            mc.fn_ffn_f16[(size_t) il], L.hc_ffn_base,
                                                            hlw.ffn, &split_ffn[(size_t) t]);
            if (!fworking) return false;
            ggml_tensor * f2 = ggml_reshape_2d(ctx, fworking, n_embd, 1);
            ffn_in = ffn_in ? ggml_concat(ctx, ffn_in, f2, 1) : f2;
        }

        // ── Batched FFN ──
        ggml_tensor * ffn_normed = build_rms_norm(ctx, ffn_in, L.ffn_norm, w.rms_eps);
        ggml_tensor * ffn_out = nullptr;
        const bool hash_routed = il < w.n_hash_layer && L.ffn_gate_tid2eid &&
                                 have_token_ids && hash_tables[(size_t) il].loaded;
        if (hash_routed) {
            // ds4_build_hash_routed_ffn is single-token; run per token, concat.
            const int n_used = ds4_effective_expert_count(w);
            ggml_tensor * hids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, q);
            ggml_set_input(hids);
            fg.hash_ids[(size_t) il] = hids;
            for (int t = 0; t < q; ++t) {
                ggml_tensor * fcol = ggml_view_2d(ctx, ffn_normed, n_embd, 1,
                                                  ffn_normed->nb[1], (size_t) t * ffn_normed->nb[1]);
                ggml_tensor * hcol = ggml_view_2d(ctx, hids, n_used, 1,
                                                  hids->nb[1], (size_t) t * hids->nb[1]);
                ggml_tensor * fo = ds4_build_hash_routed_ffn(
                    ctx, w, L, ggml_cont(ctx, fcol), hcol, 1);
                if (!fo) { ffn_out = nullptr; break; }
                ffn_out = ffn_out ? ggml_concat(ctx, ffn_out, fo, 1) : fo;
            }
        } else {
            ffn_out = build_moe_ffn(ctx, ffn_normed, w, L, il, q);
        }
        if (!ffn_out) return false;

        // ── HC post (FFN), per token; capture at drafter layers ──
        for (int t = 0; t < q; ++t) {
            ggml_tensor * fo = ggml_view_2d(ctx, ffn_out, n_embd, 1,
                                            ffn_out->nb[1], (size_t) t * ffn_out->nb[1]);
            ggml_tensor * fo_flat = ggml_reshape_1d(ctx, ggml_cont(ctx, fo), n_embd);
            ggml_tensor * hc_next = ggml_ds4_hc_post(ctx, hc_flat[(size_t) t], fo_flat,
                                                     split_ffn[(size_t) t], n_hc);
            hc_cur[(size_t) t] = ggml_reshape_2d(ctx, hc_next, n_embd, n_hc);
        }
        for (size_t ci = 0; ci < capture_ids.size(); ++ci) {
            if (capture_ids[ci] != il) continue;
            for (int t = 0; t < q; ++t) {
                ggml_tensor * hs = hc_cur[(size_t) t];            // [n_embd, n_hc]
                ggml_tensor * hsT = ggml_cont(ctx, ggml_transpose(ctx, hs)); // [n_hc, n_embd]
                ggml_tensor * summed = ggml_sum_rows(ctx, hsT);   // [1, n_embd]
                ggml_tensor * mean = ggml_scale(ctx, summed, 1.0f / (float) n_hc);
                capture_pieces[ci * (size_t) q + (size_t) t] =
                    ggml_reshape_1d(ctx, ggml_cont(ctx, mean), n_embd);
            }
        }
    }

    // ── Output: per-token HC merge → batched out_norm + lm_head ──
    ggml_tensor * final_all = nullptr;
    for (int t = 0; t < q; ++t) {
        ggml_tensor * hc_flat = ggml_reshape_1d(ctx, hc_cur[(size_t) t], (int64_t) n_embd * n_hc);
        ggml_tensor * onorm = ggml_rms_norm(ctx, hc_flat, w.hc_eps);
        ggml_tensor * omix = ggml_mul_mat(ctx, mc.fn_out_f16, onorm);
        omix = ggml_reshape_1d(ctx, omix, ggml_nelements(omix));
        ggml_tensor * obase = ds4_fused_hc_base_f32(ctx, w.output_hc_base);
        if (!obase || hc_out_weights.scale_data.empty()) return false;
        ggml_tensor * fe = ggml_ds4_hc_out(ctx, omix, obase, hc_flat, n_hc,
                                           hc_out_weights.scale_data[0]);
        ggml_tensor * fe2 = ggml_reshape_2d(ctx, fe, n_embd, 1);
        final_all = final_all ? ggml_concat(ctx, final_all, fe2, 1) : fe2;
    }
    ggml_tensor * out_normed = build_rms_norm(ctx, final_all, w.out_norm, w.rms_eps);
    fg.logits = ggml_mul_mat(ctx, w.output, out_normed);   // [n_vocab, q]
    ggml_set_output(fg.logits);
    ggml_build_forward_expand(gf, fg.logits);

    if (!capture_pieces.empty()) {
        for (ggml_tensor * piece : capture_pieces) {
            if (!piece) {
                std::fprintf(stderr, "[ds4-fused-verify] capture layer id is invalid\n");
                return false;
            }
        }
        ggml_tensor * cap = capture_pieces[0];
        for (size_t i = 1; i < capture_pieces.size(); ++i) {
            cap = ggml_concat(ctx, cap, capture_pieces[i], 0);
        }
        ex.capture = cap;
        ggml_set_output(ex.capture);
        ggml_build_forward_expand(gf, ex.capture);
    }

    if (!fg.sg.alloc) {
        fg.sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!fg.sg.alloc || !ggml_gallocr_alloc_graph(fg.sg.alloc, fg.sg.gf)) {
        std::fprintf(stderr, "[ds4-fused-verify] graph alloc failed\n");
        return false;
    }
    fg.shape_key = std::move(shape_key);
    return true;
}

static int ds4_try_fused_verify_step(
        Ds4FusedVerifyCache & vc,
        DeepSeek4FusedDecodeCache & mc,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights,
        const std::vector<HashRoutingTableCpu> & hash_tables,
        std::vector<int32_t> & hash_scratch,
        const float * embed,
        int n_tokens,
        int kv_start,
        std::vector<float> & out_logits,
        const int32_t * token_ids,
        Ds4VerifyHooks * hooks,
        DeepSeek4StepTelemetry * telemetry) {
    if (vc.disabled || mc.disabled) return 0;
    if (!hooks || !hooks->capture_layer_ids) return 0;
    if (!hc_out_weights.loaded || hc_out_weights.scale_data.empty() ||
        !w.output_hc_fn || !w.output_hc_base) { vc.disabled = true; return 0; }
    for (int il = 0; il < w.n_layer; ++il) {
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        if (!hlw.attn.loaded || hlw.attn.scale_data.size() < 3 ||
            !hlw.ffn.loaded || hlw.ffn.scale_data.size() < 3 ||
            !L.hc_attn_fn || !L.hc_ffn_fn || !L.hc_attn_base || !L.hc_ffn_base) {
            vc.disabled = true;
            return 0;
        }
    }
    if (vc.owner_ctx != w.ctx || vc.backend != backend) {
        vc.destroy();
        vc.owner_ctx = w.ctx;
        vc.backend = backend;
    }
    if (mc.owner_ctx != w.ctx || mc.backend != backend) {
        mc.destroy();
        mc.owner_ctx = w.ctx;
        mc.backend = backend;
    }
    if (!ds4_fused_ensure_fn_mirrors(mc, backend, w, hc_weights, hc_out_weights)) {
        vc.disabled = true;
        return 0;
    }

    const int q = n_tokens;
    const int token_pos = kv_start + q - 1;
    std::vector<int64_t> key;
    key.reserve((size_t) w.n_layer + hooks->capture_layer_ids->size() + 4);
    key.push_back(w.n_swa);
    key.push_back(q);
    key.push_back(token_ids ? 1 : 0);
    key.push_back((int64_t) hooks->capture_layer_ids->size());
    for (int layer : *hooks->capture_layer_ids) key.push_back(layer);
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        int padded = 0;
        if (ratio > 0 && lc.comp_kv) {
            const int n_comp = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos);
            padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
        }
        int b_idx = -1;
        if (ratio > 0) {
            for (int t = 0; t < q; ++t) {
                if (((kv_start + t + 1) % ratio) == 0) { b_idx = t; break; }
            }
        }
        key.push_back(((int64_t) padded << 4) | (int64_t) (b_idx + 1));
    }

    vc.counter++;
    DeepSeek4FusedDecodeGraph * fg = nullptr;
    Ds4FusedVerifyCache::Extra * ex = nullptr;
    for (size_t i = 0; i < vc.slots.size(); ++i) {
        if (vc.slots[i].built() && vc.slots[i].shape_key == key) {
            fg = &vc.slots[i]; ex = &vc.extra[i]; break;
        }
    }
    if (!fg) {
        size_t pick = 0;
        for (size_t i = 0; i < vc.slots.size(); ++i) {
            if (!vc.slots[i].built()) { pick = i; break; }
            if (vc.slots[i].last_use < vc.slots[pick].last_use) pick = i;
        }
        fg = &vc.slots[pick]; ex = &vc.extra[pick];
        const auto build_t0 = Ds4TimingClock::now();
        if (!ds4_build_fused_verify_graph(mc, *fg, *ex, backend, w, cache,
                                          hc_weights, hc_out_weights, hash_tables,
                                          kv_start, q, token_ids != nullptr,
                                          *hooks->capture_layer_ids, std::move(key))) {
            step_graph_free(fg->sg);
            fg->reset_nodes();
            ex->reset();
            vc.disabled = true;
            return 0;
        }
        if (telemetry) telemetry->full_graph_build_us += ds4_elapsed_us(build_t0, Ds4TimingClock::now());
    }
    fg->last_use = vc.counter;

    // ── Fill inputs ──
    const auto set_t0 = Ds4TimingClock::now();
    ds4_fv_set(fg->inp_embed, embed, sizeof(float) * (size_t) w.n_embd * q);
    std::vector<int32_t> iv(q);
    std::vector<int64_t> lv(q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = kv_start + i;
    ds4_fv_set(ex->pos_q, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = -(kv_start + i);
    ds4_fv_set(ex->neg_q, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) lv[(size_t) i] = (kv_start + i) % w.n_swa;
    ds4_fv_set(ex->rawrows, lv.data(), sizeof(int64_t) * q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = (kv_start + i) % 4;
    ds4_fv_set(ex->ape4, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) lv[(size_t) i] = 4 + (kv_start + i) % 4;
    ds4_fv_set(ex->st4, lv.data(), sizeof(int64_t) * q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = (kv_start + i) % 128;
    ds4_fv_set(ex->ape128, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) lv[(size_t) i] = (kv_start + i) % 128;
    ds4_fv_set(ex->st128, lv.data(), sizeof(int64_t) * q);

    std::vector<int32_t> i32v((size_t) w.n_layer * 2, 0);
    std::vector<int64_t> i64v((size_t) w.n_layer * 1, 0);
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        if (ratio > 0) {
            int pos_b = -1;   // boundary position inside the batch (if any)
            for (int t = 0; t < q; ++t) {
                if (((kv_start + t + 1) % ratio) == 0) { pos_b = kv_start + t; break; }
            }
            i32v[(size_t) il * 2 + 0] = token_pos % ratio;
            i32v[(size_t) il * 2 + 1] = (pos_b >= 0 ? pos_b : token_pos) + 1 - ratio;
            i64v[(size_t) il] = (pos_b >= 0 ? pos_b : token_pos) / ratio;
        }
    }
    ds4_fv_set(fg->i32_bundle, i32v.data(), sizeof(int32_t) * i32v.size());
    ds4_fv_set(fg->i64_bundle, i64v.data(), sizeof(int64_t) * i64v.size());

    // causal mask values
    {
        std::vector<float> maskv((size_t) ggml_nelements(fg->mask_bundle), 0.0f);
        size_t off = 0;
        const int e = kv_start + q;
        for (int il = 0; il < w.n_layer; ++il) {
            const int ratio = (int) w.compress_ratios[il];
            DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
            int padded = 0;
            if (ratio > 0 && lc.comp_kv) {
                const int n_comp = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos);
                padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
            }
            const size_t n_attn = (size_t) w.n_swa + padded + q;
            for (int i = 0; i < q; ++i) {
                float * col = maskv.data() + off + (size_t) i * n_attn;
                const int pos_i = kv_start + i;
                for (int r = 0; r < w.n_swa; ++r) {
                    // position held by slot r AFTER this batch's ring writes
                    const int p_r = (e <= w.n_swa) ? (r < e ? r : -1)
                                  : (e - 1) - ((e - 1 - r) % w.n_swa);
                    if (p_r < 0 || p_r > pos_i) col[r] = -1.0e30f;
                }
                const int vis = (ratio > 0 && lc.comp_kv)
                              ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, pos_i) : 0;
                for (int c = 0; c < padded; ++c) {
                    if (c >= vis) col[(size_t) w.n_swa + c] = -1.0e30f;
                }
                for (int j = 0; j < q; ++j) {
                    const bool visible = (j > i) && (kv_start + j >= w.n_swa);
                    if (!visible) col[(size_t) w.n_swa + padded + j] = -1.0e30f;
                }
            }
            off += n_attn * q;
        }
        ds4_fv_set(fg->mask_bundle, maskv.data(), sizeof(float) * maskv.size());
    }

    if (token_ids) {
        for (int il = 0; il < w.n_layer; ++il) {
            ggml_tensor * hids = fg->hash_ids[(size_t) il];
            if (!hids) continue;
            const HashRoutingTableCpu & table = hash_tables[(size_t) il];
            const int n_used = ds4_effective_expert_count(w);
            hash_scratch.resize((size_t) n_used * q);
            for (int i = 0; i < q; ++i) {
                const int32_t * row = hash_routing_row(
                    table, token_ids[i], w.n_expert_used);
                if (!row) {
                    std::fprintf(stderr,
                                 "[ds4-fused-verify] token id %d outside hash table for layer %d\n",
                                 token_ids[i], il);
                    return -1;
                }
                std::memcpy(hash_scratch.data() + (size_t) i * n_used,
                            row,
                            (size_t) n_used * sizeof(int32_t));
            }
            ggml_backend_tensor_set(hids, hash_scratch.data(), 0,
                                    sizeof(int32_t) * (size_t) n_used * q);
        }
    }
    if (telemetry) {
        telemetry->full_graph_set_us += ds4_elapsed_us(set_t0, Ds4TimingClock::now());
    }

    // ── Compute + read back ──
    const auto compute_t0 = Ds4TimingClock::now();
    if (ggml_backend_graph_compute(backend, fg->sg.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[ds4-fused-verify] compute failed\n");
        return -1;
    }
    if (telemetry) {
        telemetry->full_graph_compute_us += ds4_elapsed_us(compute_t0, Ds4TimingClock::now());
    }

    const auto read_t0 = Ds4TimingClock::now();
    const int ncap = (int) hooks->capture_layer_ids->size();
    if (hooks->all_logits_out) {
        hooks->all_logits_out->resize((size_t) w.n_vocab * q);
        ggml_backend_tensor_get(fg->logits, hooks->all_logits_out->data(), 0,
                                sizeof(float) * (size_t) w.n_vocab * q);
    }
    out_logits.resize((size_t) w.n_vocab);
    ggml_backend_tensor_get(fg->logits, out_logits.data(),
                            (size_t) (q - 1) * (size_t) w.n_vocab * sizeof(float),
                            sizeof(float) * (size_t) w.n_vocab);
    if (hooks->capture_out && ex->capture && ncap > 0) {
        std::vector<float> flat((size_t) w.n_embd * ncap * q);
        ggml_backend_tensor_get(ex->capture, flat.data(), 0, sizeof(float) * flat.size());
        hooks->capture_out->assign((size_t) ncap * w.n_embd * q, 0.0f);
        for (int ci = 0; ci < ncap; ++ci) {
            for (int t = 0; t < q; ++t) {
                const float * src = flat.data() + ((size_t) ci * q + t) * w.n_embd;
                float * dst = hooks->capture_out->data() +
                              (size_t) t * ncap * w.n_embd + (size_t) ci * w.n_embd;
                std::memcpy(dst, src, sizeof(float) * (size_t) w.n_embd);
            }
        }
    }
    if (telemetry) {
        telemetry->full_graph_read_us += ds4_elapsed_us(read_t0, Ds4TimingClock::now());
    }
    return 1;
}
