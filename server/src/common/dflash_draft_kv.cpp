#include "dflash_draft_kv.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

static constexpr int MASK_KV_PAD = 32;
static inline int mask_align_up(int x, int a) { return ((x + a - 1) / a) * a; }

static constexpr uint16_t F16_ZERO    = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

bool draft_kv_init(DraftKvState & st,
                   const DraftWeights & dw,
                   ggml_backend_t backend,
                   int cap,
                   ggml_tensor * lm_head) {
    if (cap <= 0 || dw.block_size <= 0) return false;
    static const bool disable_swa =
        std::getenv("DFLASH_DISABLE_DRAFT_SWA") != nullptr;

    st.cap        = cap;
    st.q_len      = dw.block_size;
    st.a_step     = 2 * dw.block_size + 2;
    st.trash_slot = cap + dw.block_size;
    st.kv_total   = mask_align_up(cap + dw.block_size + 1, MASK_KV_PAD);
    st.fc_in      = dw.n_target_layers * dw.n_embd;
    st.any_full = st.any_swa = false;
    for (int i = 0; i < dw.n_layer; i++) {
        if (dw.layers[i].is_swa && !disable_swa) st.any_swa = true;
        else                                     st.any_full = true;
    }

    // ── persistent memory: caches + inputs (outside gallocr, stable, zeroed)
    const size_t n_mem_tensors = 2 * (size_t)dw.n_layer + 10;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * n_mem_tensors;
    ip.no_alloc = true;
    st.mem_ctx = ggml_init(ip);
    if (!st.mem_ctx) return false;

    const int64_t kv_row = (int64_t)dw.head_dim * dw.n_head_kv;
    st.cache.kv_total = st.kv_total;
    st.cache.k.resize(dw.n_layer);
    st.cache.v.resize(dw.n_layer);
    for (int il = 0; il < dw.n_layer; il++) {
        st.cache.k[il] = ggml_new_tensor_2d(st.mem_ctx, GGML_TYPE_F16, kv_row, st.kv_total);
        st.cache.v[il] = ggml_new_tensor_2d(st.mem_ctx, GGML_TYPE_F16, kv_row, st.kv_total);
    }
    st.inp_embed  = ggml_new_tensor_2d(st.mem_ctx, GGML_TYPE_F32, dw.n_embd, st.q_len);
    st.pos_q      = ggml_new_tensor_1d(st.mem_ctx, GGML_TYPE_I32, st.q_len);
    st.noise_rows = ggml_new_tensor_1d(st.mem_ctx, GGML_TYPE_I32, st.q_len);
    if (st.any_full)
        st.mask_full = ggml_new_tensor_2d(st.mem_ctx, GGML_TYPE_F16, st.kv_total, st.q_len);
    if (st.any_swa)
        st.mask_swa  = ggml_new_tensor_2d(st.mem_ctx, GGML_TYPE_F16, st.kv_total, st.q_len);
    st.ap_feat = ggml_new_tensor_2d(st.mem_ctx, GGML_TYPE_F32, st.fc_in, st.a_step);
    st.ap_pos  = ggml_new_tensor_1d(st.mem_ctx, GGML_TYPE_I32, st.a_step);
    st.ap_rows = ggml_new_tensor_1d(st.mem_ctx, GGML_TYPE_I32, st.a_step);

    st.mem_buf = ggml_backend_alloc_ctx_tensors(st.mem_ctx, backend);
    if (!st.mem_buf) {
        std::fprintf(stderr, "[draft-kv] cache alloc failed\n");
        return false;
    }
    // Zero everything: empty/pad cache slots are read by FA (masked -inf) and
    // must be finite; pad feature rows must be finite for the trash-slot rows.
    ggml_backend_buffer_clear(st.mem_buf, 0);

    // ── build the fixed-topology step graph once
    const size_t arena_sz = 16u * 1024 * 1024;
    st.meta_arena.resize(arena_sz);
    ggml_init_params gp{};
    gp.mem_size   = st.meta_arena.size();
    gp.mem_buffer = st.meta_arena.data();
    gp.no_alloc   = true;
    st.g_ctx = ggml_init(gp);
    if (!st.g_ctx) return false;
    st.gf = ggml_new_graph_custom(st.g_ctx, 4096, false);

    DraftKvAppendInputs ai{};
    ai.n_rows    = st.a_step;
    ai.feat      = st.ap_feat;
    ai.positions = st.ap_pos;
    ai.rows      = st.ap_rows;
    if (!build_draft_kv_append(st.g_ctx, st.gf, dw, st.cache, ai)) return false;

    DraftKvStepInputs si{};
    si.noise_embed = st.inp_embed;
    si.positions_q = st.pos_q;
    si.noise_rows  = st.noise_rows;
    si.mask_full   = st.mask_full;
    si.mask_swa    = st.mask_swa;
    si.lm_head     = lm_head;
    DraftGraphOutputs go = build_draft_kv_step(st.g_ctx, st.gf, dw, st.cache, si);
    if (!go.hidden_states) return false;
    st.hidden_states = go.hidden_states;
    st.logits        = go.logits;
    ggml_set_output(st.hidden_states);
    ggml_build_forward_expand(st.gf, st.hidden_states);
    if (st.logits) {
        ggml_set_output(st.logits);
        ggml_build_forward_expand(st.gf, st.logits);
    }

    st.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!st.galloc || !ggml_gallocr_alloc_graph(st.galloc, st.gf)) {
        std::fprintf(stderr, "[draft-kv] graph alloc failed\n");
        return false;
    }

    // static noise scratch slots
    std::vector<int32_t> nrows((size_t)st.q_len);
    for (int i = 0; i < st.q_len; i++) nrows[(size_t)i] = st.cap + i;
    ggml_backend_tensor_set(st.noise_rows, nrows.data(), 0,
                            sizeof(int32_t) * nrows.size());

    st.built_for = &dw;
    st.slot_pos.assign((size_t)st.cap, -1);
    st.next_pos = 0;
    std::fprintf(stderr,
        "[draft-kv] ctx-KV ring active: cap=%d kv_total=%d a_step=%d "
        "layers=%d f16 cache %.1f MiB\n",
        st.cap, st.kv_total, st.a_step, dw.n_layer,
        (double)(2ull * dw.n_layer * (size_t)kv_row * st.kv_total * 2) / (1024.0 * 1024.0));
    return true;
}

void draft_kv_reset(DraftKvState & st) {
    std::fill(st.slot_pos.begin(), st.slot_pos.end(), -1);
    st.next_pos = 0;
}

void draft_kv_free(DraftKvState & st) {
    if (st.galloc)  { ggml_gallocr_free(st.galloc); st.galloc = nullptr; }
    if (st.g_ctx)   { ggml_free(st.g_ctx); st.g_ctx = nullptr; }
    st.gf = nullptr;
    if (st.mem_buf) { ggml_backend_buffer_free(st.mem_buf); st.mem_buf = nullptr; }
    if (st.mem_ctx) { ggml_free(st.mem_ctx); st.mem_ctx = nullptr; }
    st.meta_arena.clear();
    st.meta_arena.shrink_to_fit();
    st.hidden_states = st.logits = nullptr;
    st.cache.k.clear();
    st.cache.v.clear();
    st.slot_pos.clear();
    st.next_pos = 0;
    st.built_for = nullptr;
}

// One-shot append of [start, start+n) via temporary exact-size graphs
// (used to fill the window after prefill or a rewind; runs once per request).
static bool draft_kv_bulk_append(DraftKvState & st,
                                 const DraftWeights & dw,
                                 ggml_backend_t backend,
                                 const DraftFeatureMirror & ring,
                                 int start, int n) {
    constexpr int A_BULK = 1024;
    std::vector<int32_t> pos, rows;
    while (n > 0) {
        const int c = std::min(n, A_BULK);
        ggml_init_params tp{};
        tp.mem_size = ggml_tensor_overhead() * 8;
        tp.no_alloc = true;
        ggml_context * tctx = ggml_init(tp);
        if (!tctx) return false;
        ggml_tensor * feat = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, st.fc_in, c);
        ggml_tensor * tpos = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, c);
        ggml_tensor * trow = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, c);
        ggml_backend_buffer_t tbuf = ggml_backend_alloc_ctx_tensors(tctx, backend);
        if (!tbuf) {
            std::fprintf(stderr, "[draft-kv] bulk: input alloc failed (n=%d fc_in=%d)\n", c, st.fc_in);
            ggml_free(tctx);
            return false;
        }

        bool ok = copy_feature_ring_range_to_tensor(ring, feat, start, c);
        if (!ok) std::fprintf(stderr, "[draft-kv] bulk: ring copy failed (start=%d n=%d ring_cap=%d fc_in=%d ring_type=%d)\n",
                              start, c, ring.cap, st.fc_in, (int)ring.storage_type);
        if (ok) {
            pos.resize((size_t)c);
            rows.resize((size_t)c);
            for (int i = 0; i < c; i++) {
                const int p = start + i;
                pos[(size_t)i]  = p;
                rows[(size_t)i] = p % st.cap;
                st.slot_pos[(size_t)(p % st.cap)] = p;
            }
            ggml_backend_tensor_set(tpos, pos.data(), 0, sizeof(int32_t) * pos.size());
            ggml_backend_tensor_set(trow, rows.data(), 0, sizeof(int32_t) * rows.size());

            std::vector<uint8_t> arena(16u * 1024 * 1024);
            ggml_init_params gp{};
            gp.mem_size   = arena.size();
            gp.mem_buffer = arena.data();
            gp.no_alloc   = true;
            ggml_context * gctx = ggml_init(gp);
            ok = gctx != nullptr;
            if (ok) {
                ggml_cgraph * g = ggml_new_graph_custom(gctx, 4096, false);
                DraftKvAppendInputs ai{c, feat, tpos, trow};
                ok = build_draft_kv_append(gctx, g, dw, st.cache, ai);
                if (!ok) std::fprintf(stderr, "[draft-kv] bulk: append build failed\n");
                if (ok) {
                    ggml_gallocr_t ga =
                        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    ok = ga && ggml_gallocr_alloc_graph(ga, g) &&
                         ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS;
                    if (!ok) std::fprintf(stderr, "[draft-kv] bulk: graph alloc/compute failed (n=%d)\n", c);
                    if (ga) ggml_gallocr_free(ga);
                }
                ggml_free(gctx);
            }
        }
        ggml_backend_buffer_free(tbuf);
        ggml_free(tctx);
        if (!ok) return false;
        start += c;
        n -= c;
    }
    return true;
}

bool draft_kv_begin_step(DraftKvState & st,
                         const DraftWeights & dw,
                         ggml_backend_t backend,
                         const DraftFeatureMirror & ring,
                         int committed) {
    if (!st.gf || committed <= 0) return false;
    // Rewind (prefix-cache restore / new shorter request): stale slots would
    // shadow live window positions, so rebuild from scratch.
    if (st.next_pos > committed) draft_kv_reset(st);

    const int win = std::min(committed, st.cap);
    const int lo  = committed - win;
    int64_t start = std::max<int64_t>(st.next_pos, lo);
    int n_new = (int)(committed - start);
    if (n_new > st.a_step) {
        if (!draft_kv_bulk_append(st, dw, backend, ring, (int)start, n_new)) {
            std::fprintf(stderr, "[draft-kv] bulk append failed\n");
            return false;
        }
        start = committed;
        n_new = 0;
    }

    // fold-in append inputs: real rows then trash-slot pads
    st.i32_hbuf.assign((size_t)st.a_step * 2, 0);
    int32_t * ap_pos  = st.i32_hbuf.data();
    int32_t * ap_rows = st.i32_hbuf.data() + st.a_step;
    for (int i = 0; i < st.a_step; i++) {
        if (i < n_new) {
            const int p = (int)start + i;
            ap_pos[i]  = p;
            ap_rows[i] = p % st.cap;
            st.slot_pos[(size_t)(p % st.cap)] = p;
        } else {
            ap_pos[i]  = 0;
            ap_rows[i] = st.trash_slot;
        }
    }
    if (n_new > 0 &&
        !copy_feature_ring_range_to_tensor(ring, st.ap_feat, (int)start, n_new)) {
        std::fprintf(stderr, "[draft-kv] feature copy failed\n");
        return false;
    }
    ggml_backend_tensor_set(st.ap_pos, ap_pos, 0, sizeof(int32_t) * (size_t)st.a_step);
    ggml_backend_tensor_set(st.ap_rows, ap_rows, 0, sizeof(int32_t) * (size_t)st.a_step);
    st.next_pos = committed;

    // noise positions (absolute)
    std::vector<int32_t> pq((size_t)st.q_len);
    for (int i = 0; i < st.q_len; i++) pq[(size_t)i] = committed + i;
    ggml_backend_tensor_set(st.pos_q, pq.data(), 0, sizeof(int32_t) * pq.size());

    // masks: ctx columns share one row template; noise columns differ only
    // in causality (full = block-visible, SWA = causal).
    const size_t mask_elems = (size_t)st.kv_total * st.q_len;
    if (st.mask_full) {
        st.mask_hbuf.assign(mask_elems, F16_NEG_INF);
        for (int s = 0; s < st.cap; s++) {
            const int32_t p = st.slot_pos[(size_t)s];
            if (p >= lo && p < committed) st.mask_hbuf[(size_t)s] = F16_ZERO;
        }
        for (int j = 0; j < st.q_len; j++)
            st.mask_hbuf[(size_t)(st.cap + j)] = F16_ZERO;
        for (int q = 1; q < st.q_len; q++)
            std::memcpy(st.mask_hbuf.data() + (size_t)q * st.kv_total,
                        st.mask_hbuf.data(), sizeof(uint16_t) * (size_t)st.kv_total);
        ggml_backend_tensor_set(st.mask_full, st.mask_hbuf.data(), 0,
                                sizeof(uint16_t) * mask_elems);
    }
    if (st.mask_swa) {
        // The window is anchored at `committed` for ALL noise rows on purpose:
        // this replicates the legacy one-shot drafter graph, which windows the
        // ctx via a single view [ctx_len - swa_window, ctx_len) shared by every
        // query row. A per-row lower bound would change the trained drafter's
        // attention pattern (and measured acceptance).
        const int eff_win = (dw.swa_window > 0 && win > dw.swa_window)
                                ? dw.swa_window : win;
        const int swa_lo = committed - eff_win;
        st.mask_hbuf.assign(mask_elems, F16_NEG_INF);
        for (int s = 0; s < st.cap; s++) {
            const int32_t p = st.slot_pos[(size_t)s];
            if (p >= swa_lo && p < committed) st.mask_hbuf[(size_t)s] = F16_ZERO;
        }
        for (int q = 1; q < st.q_len; q++)
            std::memcpy(st.mask_hbuf.data() + (size_t)q * st.kv_total,
                        st.mask_hbuf.data(), sizeof(uint16_t) * (size_t)st.kv_total);
        for (int q = 0; q < st.q_len; q++)
            for (int j = 0; j <= q; j++)
                st.mask_hbuf[(size_t)q * st.kv_total + (size_t)(st.cap + j)] = F16_ZERO;
        ggml_backend_tensor_set(st.mask_swa, st.mask_hbuf.data(), 0,
                                sizeof(uint16_t) * mask_elems);
    }
    return true;
}

}  // namespace dflash::common
