// LagunaDFlashTarget - DFlashTarget adapter for Poolside Laguna-XS.2.

#include "ggml-cuda.h"
#include <cmath>
#include "laguna_dflash_target.h"
#include "../common/kvflash_pager.h"
#include "../common/ddtree.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace dflash::common {

namespace {

int laguna_argmax_row(const float * row, int vocab) {
    int best = 0;
    float bv = row[0];
    for (int v = 1; v < vocab; ++v) {
        if (row[v] > bv) {
            bv = row[v];
            best = v;
        }
    }
    return best;
}

}  // namespace

LagunaDFlashTarget::LagunaDFlashTarget(
        LagunaTargetWeights & w,
        LagunaTargetCache & cache,
        ggml_backend_t backend)
    : w_(w), cache_(cache), backend_(backend) {
    capture_ids_ = cache.capture_layer_ids;
}

LagunaDFlashTarget::~LagunaDFlashTarget() {
    laguna_snapshot_free(verify_snap_);
    if (embd_gpu_buf_) { ggml_backend_buffer_free(embd_gpu_buf_); embd_gpu_buf_ = nullptr; }
    if (embd_gpu_ctx_) { ggml_free(embd_gpu_ctx_); embd_gpu_ctx_ = nullptr; }
    embd_gpu_ = nullptr;
    if (fused_backend_) { ggml_backend_free(fused_backend_); fused_backend_ = nullptr; }
}

bool LagunaDFlashTarget::verify_batch(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<int32_t> * all_argmax,
        bool capture_ssm_intermediates) {
    (void)capture_ssm_intermediates;
    const int n_tokens = (int)tokens.size();
    if (n_tokens <= 0) return false;

    const int hidden = w_.n_embd;
    std::vector<float> embed((size_t)n_tokens * (size_t)hidden);
    if (!w_.embedder.embed(tokens.data(), n_tokens, embed.data())) {
        std::fprintf(stderr, "laguna_verify_batch: embed failed\n");
        return false;
    }

    if (pager_ && !pager_->alloc_span(base_pos, n_tokens)) {
        return false;
    }

    std::vector<int32_t> argmax_buf;
    std::vector<float> * logits_out = keep_verify_logits_ ? &verify_logits_ : nullptr;
    if (!keep_verify_logits_) {
        verify_logits_.clear();
        verify_logits_n_ = 0;
    }
    if (!laguna_verify_batch(backend_, w_, cache_, embed.data(),
                             tokens.data(), n_tokens, base_pos,
                             argmax_buf, pager_, logits_out)) {
        return false;
    }
    if (keep_verify_logits_) {
        verify_logits_n_ = n_tokens;
    }

    last_tok = argmax_buf[n_tokens - 1];
    if (all_argmax) {
        *all_argmax = std::move(argmax_buf);
    }
    return true;
}

bool LagunaDFlashTarget::read_verify_logits(int n_tokens, std::vector<float> & out) {
    if (n_tokens <= 0 || verify_logits_n_ <= 0 || verify_logits_.empty()) {
        return false;
    }
    if (n_tokens > verify_logits_n_) {
        return false;
    }
    const size_t vocab = verify_logits_.size() / (size_t)verify_logits_n_;
    out.resize((size_t)n_tokens * vocab);
    std::memcpy(out.data(), verify_logits_.data(), sizeof(float) * out.size());
    return true;
}

// NOTE: the spec-decode loop (do_spec_decode) does not call snapshot_kv/
// restore_kv — it manages KV via cur_pos truncation + overwrite, exactly like
// the gemma4 path. These remain for DFlashTarget-interface completeness and any
// future caller. They use device-to-device copies (ggml_backend_tensor_copy) on
// backend_-resident dup tensors, mirroring Gemma4DFlashTarget — NOT the
// host-backed laguna_snapshot_* helpers (those assume a CPU snapshot backend).
bool LagunaDFlashTarget::snapshot_kv() {
    if (cache_.cur_pos <= 0) return false;
    if (!verify_snap_.ctx) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (size_t)(w_.n_layer * 2 + 8) + 4096;
        ip.no_alloc = true;
        verify_snap_.ctx = ggml_init(ip);
        if (!verify_snap_.ctx) return false;
        verify_snap_.attn_k.assign((size_t)w_.n_layer, nullptr);
        verify_snap_.attn_v.assign((size_t)w_.n_layer, nullptr);
        for (int il = 0; il < w_.n_layer; ++il) {
            if (cache_.attn_k[(size_t)il] && cache_.attn_v[(size_t)il]) {
                verify_snap_.attn_k[(size_t)il] =
                    ggml_dup_tensor(verify_snap_.ctx, cache_.attn_k[(size_t)il]);
                verify_snap_.attn_v[(size_t)il] =
                    ggml_dup_tensor(verify_snap_.ctx, cache_.attn_v[(size_t)il]);
            }
        }
        verify_snap_.buf = ggml_backend_alloc_ctx_tensors(verify_snap_.ctx, backend_);
        if (!verify_snap_.buf) {
            ggml_free(verify_snap_.ctx);
            verify_snap_.ctx = nullptr;
            return false;
        }
    }
    for (int il = 0; il < w_.n_layer; ++il) {
        if (cache_.attn_k[(size_t)il] && verify_snap_.attn_k[(size_t)il]) {
            ggml_backend_tensor_copy(cache_.attn_k[(size_t)il], verify_snap_.attn_k[(size_t)il]);
            ggml_backend_tensor_copy(cache_.attn_v[(size_t)il], verify_snap_.attn_v[(size_t)il]);
        }
    }
    verify_snap_.cur_pos = cache_.cur_pos;
    verify_snap_.used = true;
    return true;
}

bool LagunaDFlashTarget::restore_kv() {
    if (!verify_snap_.used) return false;
    for (int il = 0; il < w_.n_layer; ++il) {
        if (cache_.attn_k[(size_t)il] && verify_snap_.attn_k[(size_t)il]) {
            ggml_backend_tensor_copy(verify_snap_.attn_k[(size_t)il], cache_.attn_k[(size_t)il]);
            ggml_backend_tensor_copy(verify_snap_.attn_v[(size_t)il], cache_.attn_v[(size_t)il]);
        }
    }
    cache_.cur_pos = verify_snap_.cur_pos;
    return true;
}

bool LagunaDFlashTarget::supports_tree_verify() const {
    // Laguna tree verify is a logical-position, non-paged pure-attention graph
    // (physical row == logical position). It is bit-correct while the kvflash
    // pager is identity (slot == position); the call site bounds the step to the
    // resident pool and verify_tree re-checks the identity prefix defensively.
    // Past the pool, chain verify takes over.
    return pager_ == nullptr || pager_->is_identity();
}

bool LagunaDFlashTarget::verify_tree(
        int committed,
        const DDTree & tree,
        const std::vector<int32_t> & flat_tokens,
        int n_alloc,
        std::vector<int32_t> & posterior_out,
        std::vector<float> * logits_out) {
    const int N = n_alloc;
    const int N_actual = 1 + tree.n_nodes;
    if (N <= 0 || N_actual <= 0 || N_actual > N ||
        (int)flat_tokens.size() < N_actual ||
        tree.visibility.size() < (size_t)N_actual * (size_t)N_actual) {
        return false;
    }
    // kvflash: this graph is position-indexed (physical row == logical
    // position), so it is valid only while the pager is identity and the span
    // fits the pool. do_spec_decode gates on this; bail cleanly otherwise.
    if (pager_ != nullptr &&
        (committed + N > pager_->pool_tokens() ||
         !pager_->identity_prefix_covers(committed))) {
        return false;
    }

    int kv_cap = 0;
    for (ggml_tensor * k : cache_.attn_k) {
        if (k) { kv_cap = (int)k->ne[1]; break; }
    }
    if (kv_cap <= 0 || committed + N > kv_cap) {
        std::fprintf(stderr, "laguna_verify_tree: KV span exceeds cache (committed=%d N=%d cap=%d)\n",
                     committed, N, kv_cap);
        return false;
    }

    static const bool g_no_kvpad = (std::getenv("DFLASH_LAGUNA_NO_KVPAD") != nullptr);
    static const bool g_pad_cpy = (std::getenv("DFLASH_LAGUNA_PAD_CPY") != nullptr);
    const int kv_len = committed + N;
    const int kv_pad = (!g_no_kvpad && kv_cap > 0)
        ? std::min((kv_len + 255) & ~255, kv_cap) : 0;
    const int mk_w = kv_pad > 0 ? kv_pad : kv_len;
    if (kv_pad <= 0 || g_pad_cpy) {
        std::fprintf(stderr, "laguna_verify_tree: requires kv_pad set_rows path\n");
        return false;
    }

    const size_t arena_size =
        ggml_tensor_overhead() * 32768 + ggml_graph_overhead() + 32 * 1024 * 1024;
    static thread_local std::vector<uint8_t> g_tree_arena;
    if (g_tree_arena.size() < arena_size) g_tree_arena.resize(arena_size);
    ggml_init_params ip{};
    ip.mem_size = arena_size;
    ip.mem_buffer = g_tree_arena.data();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);

    ggml_tensor * ie = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w_.n_embd, N, 1);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_input(pp);
    ggml_tensor * kvi = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_input(kvi);

    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mk_w, N, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_cnv = ggml_cast(ctx, mk_full, GGML_TYPE_F16);
    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mk_w, N, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_cnv = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    ggml_tensor * feat_rows = nullptr;
    if (cache_.target_feat && cache_.target_feat_cap > 0) {
        feat_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        ggml_set_input(feat_rows);
    }

    LagunaGraphInputs gi{};
    gi.inp_embed        = ie;
    gi.positions        = pp;
    gi.attn_mask        = mk_full_cnv;
    gi.attn_mask_swa    = mk_swa_cnv;
    gi.n_tokens         = N;
    gi.kv_start         = committed;
    gi.kv_pad           = kv_pad;
    gi.kv_idx           = kvi;
    gi.output_last_only = false;
    gi.output_logits    = true;
    gi.logits_are_output = logits_out != nullptr;
    gi.capture_features = feat_rows != nullptr;
    gi.target_feat_rows = feat_rows;
    gi.hybrid           = nullptr;

    LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w_, cache_, gi);

    // Greedy tree-verify only needs the per-node argmax. Compute it on-GPU
    // (like the chain path) and read back N_actual int32 indices instead of
    // copying the full vocab*N_actual logits to host and arg-maxing on the CPU.
    // The full-logits readback path is kept only when the caller wants logits
    // (sampled tree-verify).
    ggml_tensor * argmax_t = nullptr;
    if (!logits_out) {
        argmax_t = ggml_argmax(ctx, go.logits);
        ggml_set_output(argmax_t);
        ggml_build_forward_expand(gf, argmax_t);
    }

    static ggml_gallocr_t galloc_tree = nullptr;
    if (!galloc_tree) {
        galloc_tree = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    }
    if (!ggml_gallocr_alloc_graph(galloc_tree, gf)) {
        std::fprintf(stderr, "laguna_verify_tree: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<float> embed((size_t)w_.n_embd * (size_t)N, 0.0f);
    if (!w_.embedder.embed(flat_tokens.data(), N_actual, embed.data())) {
        std::fprintf(stderr, "laguna_verify_tree: embed failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(ie, embed.data(), 0, ggml_nbytes(ie));

    std::vector<int32_t> pos((size_t)N, 0);
    pos[0] = committed;
    for (int i = 1; i < N_actual; ++i) {
        pos[(size_t)i] = committed + tree.depths[(size_t)i - 1];
    }
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));

    std::vector<int32_t> rows((size_t)N);
    for (int i = 0; i < N; ++i) rows[(size_t)i] = committed + i;
    ggml_backend_tensor_set(kvi, rows.data(), 0, ggml_nbytes(kvi));

    if (feat_rows) {
        std::vector<int32_t> feat_idx((size_t)N);
        for (int i = 0; i < N; ++i) {
            feat_idx[(size_t)i] = (committed + i) % cache_.target_feat_cap;
        }
        ggml_backend_tensor_set(feat_rows, feat_idx.data(), 0, ggml_nbytes(feat_rows));
    }

    std::vector<float> mfull((size_t)mk_w * (size_t)N, -INFINITY);
    std::vector<float> mswa((size_t)mk_w * (size_t)N, -INFINITY);
    const int W = w_.sliding_window;
    for (int q = 0; q < N_actual; ++q) {
        const int depth_q = (q == 0) ? 0 : tree.depths[(size_t)q - 1];
        const int abs_q = committed + depth_q;

        for (int k = 0; k < committed && k < mk_w; ++k) {
            mfull[(size_t)q * (size_t)mk_w + (size_t)k] = 0.0f;
        }

        const int win_lo = std::max(0, abs_q - W + 1);
        for (int k = win_lo; k < committed && k < mk_w; ++k) {
            mswa[(size_t)q * (size_t)mk_w + (size_t)k] = 0.0f;
        }

        for (int j = 0; j < N_actual; ++j) {
            if (!tree.visibility[(size_t)q * (size_t)N_actual + (size_t)j]) {
                continue;
            }
            const int slot = committed + j;
            if (slot < mk_w) {
                mfull[(size_t)q * (size_t)mk_w + (size_t)slot] = 0.0f;
                mswa[(size_t)q * (size_t)mk_w + (size_t)slot] = 0.0f;
            }
        }
    }
    ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));
    ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));

    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_verify_tree: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    posterior_out.resize((size_t)N_actual);
    if (argmax_t) {
        // On-GPU argmax: read back only N_actual indices.
        ggml_backend_tensor_get(argmax_t, posterior_out.data(), 0,
                                sizeof(int32_t) * (size_t)N_actual);
    } else {
        // Caller wants the logits (sampled tree-verify): full readback + CPU argmax.
        const int vocab = (int)w_.embedder.n_vocab;
        std::vector<float> logits((size_t)vocab * (size_t)N_actual);
        ggml_backend_tensor_get(go.logits, logits.data(), 0,
                                sizeof(float) * logits.size());
        for (int i = 0; i < N_actual; ++i) {
            posterior_out[(size_t)i] =
                laguna_argmax_row(logits.data() + (size_t)i * (size_t)vocab, vocab);
        }
        *logits_out = std::move(logits);
    }
    cache_.cur_pos = committed + N_actual;
    ggml_free(ctx);
    return true;
}

bool LagunaDFlashTarget::rollback_to_tree(
        int committed,
        const DDTree & tree,
        const std::vector<int> & accepted_dfs) {
    (void)tree;
    const int commit_n = (int)accepted_dfs.size();
    if (commit_n <= 0) return false;

    bool contiguous_prefix = true;
    for (int d = 0; d < commit_n; ++d) {
        if (accepted_dfs[(size_t)d] != d) {
            contiguous_prefix = false;
            break;
        }
    }
    if (contiguous_prefix) {
        // A failed alloc must abort: returning true with unmapped slots would
        // make the next step read stale/unmapped KV (silent corruption). The
        // call-site guard bounds the span to the resident pool so this never
        // evicts in practice.
        if (pager_ && !pager_->alloc_span(committed, commit_n)) {
            std::fprintf(stderr,
                "rollback_to_tree: kvflash alloc_span failed (committed=%d commit_n=%d)\n",
                committed, commit_n);
            return false;
        }
        cache_.cur_pos = committed + commit_n;
        cache_.last_tok = -1;
        return true;
    }

    const int n_head_kv = w_.n_head_kv;
    const size_t arena_size =
        ggml_tensor_overhead() * 1024 + ggml_graph_overhead() + 4 * 1024 * 1024;
    static thread_local std::vector<uint8_t> g_rollback_arena;
    if (g_rollback_arena.size() < arena_size) {
        g_rollback_arena.resize(arena_size);
    }

    ggml_init_params ip{};
    ip.mem_size = arena_size;
    ip.mem_buffer = g_rollback_arena.data();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);

    ggml_tensor * src_rows_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,
                                                   commit_n, n_head_kv);
    ggml_set_input(src_rows_kv);
    ggml_tensor * dst_rows_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,
                                                   commit_n, n_head_kv);
    ggml_set_input(dst_rows_kv);

    ggml_tensor * src_rows_feat = nullptr;
    ggml_tensor * dst_rows_feat = nullptr;
    if (cache_.target_feat && cache_.target_feat_cap > 0) {
        src_rows_feat = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, commit_n);
        ggml_set_input(src_rows_feat);
        dst_rows_feat = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, commit_n);
        ggml_set_input(dst_rows_feat);

        ggml_tensor * feat_gather = ggml_get_rows(ctx, cache_.target_feat, src_rows_feat);
        feat_gather = ggml_cont(ctx, feat_gather);
        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, cache_.target_feat, feat_gather, dst_rows_feat));
    }

    for (int il = 0; il < w_.n_layer; ++il) {
        ggml_tensor * ck = cache_.attn_k[(size_t)il];
        ggml_tensor * cv = cache_.attn_v[(size_t)il];
        if (!ck || !cv) continue;

        ggml_tensor * kg = ggml_get_rows(ctx, ck, src_rows_kv);
        kg = ggml_cont(ctx, kg);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, ck, kg, dst_rows_kv));

        ggml_tensor * vg = ggml_get_rows(ctx, cv, src_rows_kv);
        vg = ggml_cont(ctx, vg);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cv, vg, dst_rows_kv));
    }

    static ggml_gallocr_t galloc_rollback = nullptr;
    if (!galloc_rollback) {
        galloc_rollback =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    }
    if (!ggml_gallocr_alloc_graph(galloc_rollback, gf)) {
        std::fprintf(stderr, "laguna_rollback_to_tree: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> src_kv((size_t)commit_n * (size_t)n_head_kv);
    std::vector<int32_t> dst_kv((size_t)commit_n * (size_t)n_head_kv);
    for (int h = 0; h < n_head_kv; ++h) {
        for (int d = 0; d < commit_n; ++d) {
            src_kv[(size_t)h * (size_t)commit_n + (size_t)d] =
                committed + accepted_dfs[(size_t)d];
            dst_kv[(size_t)h * (size_t)commit_n + (size_t)d] =
                committed + d;
        }
    }
    ggml_backend_tensor_set(src_rows_kv, src_kv.data(), 0, ggml_nbytes(src_rows_kv));
    ggml_backend_tensor_set(dst_rows_kv, dst_kv.data(), 0, ggml_nbytes(dst_rows_kv));

    if (src_rows_feat && dst_rows_feat) {
        const int cap = cache_.target_feat_cap;
        std::vector<int32_t> src_feat((size_t)commit_n);
        std::vector<int32_t> dst_feat((size_t)commit_n);
        for (int d = 0; d < commit_n; ++d) {
            src_feat[(size_t)d] = (committed + accepted_dfs[(size_t)d]) % cap;
            dst_feat[(size_t)d] = (committed + d) % cap;
        }
        ggml_backend_tensor_set(src_rows_feat, src_feat.data(), 0,
                                ggml_nbytes(src_rows_feat));
        ggml_backend_tensor_set(dst_rows_feat, dst_feat.data(), 0,
                                ggml_nbytes(dst_rows_feat));
    }

    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_rollback_to_tree: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // kvflash: register the tree-committed (identity) positions so the pager
    // tracks this path's position-indexed writes (chain/AR self-register via
    // kvflash_alloc_span; this is the only path that writes KV directly). A
    // failed alloc must abort: returning true with unmapped slots would make the
    // next step read stale/unmapped KV (silent corruption).
    if (pager_ && !pager_->alloc_span(committed, commit_n)) {
        std::fprintf(stderr,
            "rollback_to_tree: kvflash alloc_span failed (committed=%d commit_n=%d)\n",
            committed, commit_n);
        ggml_free(ctx);
        return false;
    }
    cache_.cur_pos = committed + commit_n;
    cache_.last_tok = -1;
    ggml_free(ctx);
    return true;
}

bool LagunaDFlashTarget::is_eos(int token) const {
    return token == w_.eos_id || token == w_.eos_chat_id;
}

ggml_tensor * LagunaDFlashTarget::lm_head_tensor() {
    return w_.output;
}

ggml_backend_t LagunaDFlashTarget::fused_head_backend() {
    if (!fused_backend_) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
        if (dev) fused_backend_ = ggml_backend_dev_init(dev, nullptr);
    }
    return fused_backend_ ? fused_backend_ : backend_;
}

ggml_tensor * LagunaDFlashTarget::gpu_embd_table() {
    if (embd_gpu_) return embd_gpu_;
    if (embd_gpu_failed_) return nullptr;
    const int64_t nv = w_.embedder.n_vocab;
    const int64_t ne = w_.n_embd;
    if (nv <= 0 || ne <= 0) { embd_gpu_failed_ = true; return nullptr; }

    ggml_init_params ip{};
    ip.mem_size   = 2 * ggml_tensor_overhead();
    ip.no_alloc   = true;
    embd_gpu_ctx_ = ggml_init(ip);
    if (!embd_gpu_ctx_) { embd_gpu_failed_ = true; return nullptr; }
    ggml_tensor * t = ggml_new_tensor_2d(embd_gpu_ctx_, GGML_TYPE_F16, ne, nv);
    embd_gpu_buf_ = ggml_backend_alloc_ctx_tensors(embd_gpu_ctx_, backend_);
    if (!embd_gpu_buf_) {
        std::fprintf(stderr,
            "[laguna-spec] gpu_embd_table: VRAM alloc failed (%.0f MiB), fused domino disabled\n",
            (double)nv * (double)ne * 2.0 / (1024.0 * 1024.0));
        ggml_free(embd_gpu_ctx_); embd_gpu_ctx_ = nullptr;
        embd_gpu_failed_ = true;
        return nullptr;
    }

    const int64_t CH = 4096;
    std::vector<int32_t>     ids((size_t)CH);
    std::vector<float>       rows_f32((size_t)CH * (size_t)ne);
    std::vector<ggml_fp16_t> rows_f16((size_t)CH * (size_t)ne);
    for (int64_t v0 = 0; v0 < nv; v0 += CH) {
        const int n = (int)((nv - v0 < CH) ? (nv - v0) : CH);
        for (int i = 0; i < n; ++i) ids[(size_t)i] = (int32_t)(v0 + i);
        if (!w_.embedder.embed(ids.data(), n, rows_f32.data())) {
            std::fprintf(stderr, "[laguna-spec] gpu_embd_table: embed failed\n");
            ggml_backend_buffer_free(embd_gpu_buf_); embd_gpu_buf_ = nullptr;
            ggml_free(embd_gpu_ctx_); embd_gpu_ctx_ = nullptr;
            embd_gpu_failed_ = true;
            return nullptr;
        }
        ggml_fp32_to_fp16_row(rows_f32.data(), rows_f16.data(), (size_t)n * (size_t)ne);
        ggml_backend_tensor_set(t, rows_f16.data(),
                                (size_t)v0 * (size_t)ne * sizeof(ggml_fp16_t),
                                (size_t)n * (size_t)ne * sizeof(ggml_fp16_t));
    }
    std::fprintf(stderr,
        "[laguna-spec] gpu_embd_table: uploaded f16 embedding table (%lld x %lld, %.0f MiB)\n",
        (long long)nv, (long long)ne,
        (double)nv * (double)ne * 2.0 / (1024.0 * 1024.0));
    embd_gpu_ = t;
    return embd_gpu_;
}

bool LagunaDFlashTarget::embed_tokens(const int32_t * tokens, int n,
                                      float * out) const {
    return w_.embedder.embed(tokens, n, out);
}

bool LagunaDFlashTarget::project_hidden_to_tokens(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out) {
    return laguna_project_hidden(backend_, w_, hidden, n_tokens, tokens_out);
}

bool LagunaDFlashTarget::project_hidden_to_tokens_topk(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out,
        int cand_k,
        std::vector<float> * cand_probs,
        std::vector<int32_t> * cand_ids) {
    return laguna_project_hidden(backend_, w_, hidden, n_tokens, tokens_out,
                                 cand_k, cand_probs, cand_ids);
}

bool LagunaDFlashTarget::project_hidden_to_logits(
        const float * hidden,
        int n_tokens,
        std::vector<float> & logits_out) {
    if (n_tokens <= 0) return false;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w_.n_embd, n_tokens);
    ggml_set_input(inp);
    ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, inp);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    static ggml_gallocr_t galloc_logits = nullptr;
    if (!galloc_logits) {
        galloc_logits = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    }
    if (!ggml_gallocr_alloc_graph(galloc_logits, gf)) {
        std::fprintf(stderr, "laguna_project_hidden_to_logits: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, hidden, 0,
                            sizeof(float) * (size_t)n_tokens * (size_t)w_.n_embd);
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_project_hidden_to_logits: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    const int vocab = (int)w_.embedder.n_vocab;
    logits_out.resize((size_t)n_tokens * (size_t)vocab);
    ggml_backend_tensor_get(logits, logits_out.data(), 0,
                            sizeof(float) * logits_out.size());
    ggml_free(ctx);
    return true;
}

bool LagunaDFlashTarget::project_hidden_to_topk(
        const float * hidden,
        int n_tokens,
        int K,
        float temperature,
        std::vector<float> & top_log_probs,
        std::vector<int32_t> & top_token_ids) {
    if (n_tokens <= 0 || K <= 0) return false;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w_.n_embd, n_tokens);
    ggml_set_input(inp);
    ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, inp);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    static ggml_gallocr_t galloc_topk = nullptr;
    if (!galloc_topk) {
        galloc_topk = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    }
    if (!ggml_gallocr_alloc_graph(galloc_topk, gf)) {
        std::fprintf(stderr, "laguna_project_hidden_to_topk: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, hidden, 0,
                            sizeof(float) * (size_t)n_tokens * (size_t)w_.n_embd);
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_project_hidden_to_topk: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // [TAG_TOPK_ROWS] GPU top-k on the resident logits: ~KB readback instead
    // of the full vocab logits; falls through to the host path on failure or
    // non-unit temperature (the kernel bakes softmax at T=1).
    if (temperature == 1.0f && K <= 8) {
        std::vector<float>   probs_gpu((size_t)n_tokens * (size_t)K);
        std::vector<int32_t> ids_gpu((size_t)n_tokens * (size_t)K);
        if (ggml_backend_cuda_topk_rows(logits, K, probs_gpu.data(), ids_gpu.data())) {
            top_log_probs.resize((size_t)n_tokens * (size_t)K);
            top_token_ids.assign(ids_gpu.begin(), ids_gpu.end());
            for (size_t z = 0; z < top_log_probs.size(); ++z) {
                top_log_probs[z] = std::log(probs_gpu[z] > 1e-30f ? probs_gpu[z] : 1e-30f);
            }
            ggml_free(ctx);
            return true;
        }
    }

    // CPU top-K via extract_draft_topk (shared with qwen35). The GPU ggml_top_k
    // path bitonic-argsorts the full vocab, whose shared memory exceeds HIP
    // shared-mem-per-block on gfx1151 (argsort.cu GGML_ASSERT). extract_draft_topk
    // applies the temperature + log-softmax on host.
    const int vocab = (int)logits->ne[0];
    std::vector<float> logits_host((size_t)vocab * (size_t)n_tokens);
    ggml_backend_tensor_get(logits, logits_host.data(), 0,
                            sizeof(float) * logits_host.size());

    top_log_probs.assign((size_t)n_tokens * (size_t)K, 0.0f);
    top_token_ids.assign((size_t)n_tokens * (size_t)K, 0);
    extract_draft_topk(logits_host.data(), n_tokens, vocab, K,
                       top_log_probs.data(), top_token_ids.data(), temperature);

    ggml_free(ctx);
    return true;
}

const std::vector<int> & LagunaDFlashTarget::capture_layer_ids() const {
    return capture_ids_;
}

}  // namespace dflash::common
