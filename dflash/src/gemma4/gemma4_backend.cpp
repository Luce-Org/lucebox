// Gemma4Backend implementation. See gemma4_backend.h for the contract.
//
// This PR ships the AR-greedy path end-to-end and stubs the DFlash / MTP
// decode loops with TODOs pointing to where the source logic lives. The
// follow-up PR ports those loops verbatim from feature/gemma4-support's
// test_gemma4_dflash.cpp.

#include "gemma4_backend.h"
#include "gemma4_runtime_helpers.h"

#include "../common/ddtree.h"
#include "../common/sampler.h"
#include "../pflash_ggml_adapter.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash27b {

// ── End-of-context headroom (per decode phase) ───────────────────────────
//
// Each decode loop checks `committed >= max_ctx - HEADROOM` before
// advancing. The headroom is the minimum number of cache slots an
// iteration still needs to safely fit, and it differs by phase:
//
//   AR:     one new token per iteration. We stop with 2 slots free so the
//           current step has room and one slot of margin for the EOS check
//           sampling cycle.
//   DFlash: q_len is already clamped to ctx_size-committed-1 just above
//           the break, so 1 slot of margin is sufficient.
//   MTP:    target + γ=1 draft → 2 tokens per accepted step. Need 2 slots
//           for the verify+draft pair plus margin.
//
// Pre-2026-05-14 these were three different bare literals; named here so
// any future change is a deliberate per-phase decision rather than a
// drift-prone copy-paste. See dflash/docs/gemma4-pr-split/pr13-slop-audit.md
// (Additional findings — magic numbers / ctx-size headroom).
static constexpr int CTX_HEADROOM_AR     = 2;
static constexpr int CTX_HEADROOM_DFLASH = 1;
static constexpr int CTX_HEADROOM_MTP    = 2;

// ── ctor / dtor ─────────────────────────────────────────────────────────

Gemma4Backend::Gemma4Backend(const Gemma4BackendConfig & args)
    : args_(args) {}

Gemma4Backend::~Gemma4Backend() {
    shutdown();
}

// ── init ────────────────────────────────────────────────────────────────

bool Gemma4Backend::init() {
    backend_ = ggml_backend_cuda_init(0);
    if (!backend_) {
        std::fprintf(stderr, "[gemma4] ggml_backend_cuda_init failed\n");
        return false;
    }

#ifdef DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL
    if (args_.use_sparse_fa) {
        pflash_register_ggml_kernel();
    }
#endif

    if (!load_gemma4_target_gguf(args_.target_path, backend_, target_w_)) {
        std::fprintf(stderr, "[gemma4] load_gemma4_target_gguf: %s\n",
                     dflash27b_last_error());
        return false;
    }

    std::vector<int> extra_q8;
    if (args_.draft_method == Gemma4DraftMethod::kMtp && !args_.mtp_path.empty()) {
        std::vector<bool> mtp_swa;
        if (get_mtp_swa_pattern(args_.mtp_path, mtp_swa) && !mtp_swa.empty()) {
            // Donor layers needed by MTP cross-attention must avoid TQ3/FWHT.
            // resolve_mtp_donor_layers fills this once both target + mtp are loaded.
            // For now we pre-load mtp_w_ to resolve the donor set.
            if (!load_gemma4_mtp_assistant(args_.mtp_path, backend_, mtp_w_)) {
                std::fprintf(stderr, "[gemma4] load_gemma4_mtp_assistant: %s\n",
                             dflash27b_last_error());
                return false;
            }
            mtp_loaded_ = true;
            resolve_mtp_donor_layers(mtp_w_, target_w_.swa_layers);
            for (const auto & L : mtp_w_.layers) {
                if (L.donor_target_layer >= 0) extra_q8.push_back(L.donor_target_layer);
            }
        }
    }

    if (!create_gemma4_cache(target_w_, args_.max_ctx, backend_, cache_,
                             extra_q8, /*target_feat_cap_hint=*/0,
                             /*enable_dflash_capture_overrides=*/args_.draft_enable_capture_overrides)) {
        std::fprintf(stderr, "[gemma4] create_gemma4_cache: %s\n",
                     dflash27b_last_error());
        return false;
    }

    if (args_.draft_method == Gemma4DraftMethod::kMtp) {
        // Allocate mtp_h_prev + mtp_h_prev_batch tensors and enable h_prev capture.
        if (!enable_mtp_h_prev(cache_, backend_, target_w_.n_embd, args_.mtp_gamma)) {
            std::fprintf(stderr, "[gemma4] enable_mtp_h_prev failed\n");
            return false;
        }
        cache_.mtp_last_full_layer = -1;
        for (int il = (int)target_w_.swa_layers.size() - 1; il >= 0; --il) {
            if (!target_w_.swa_layers[il]) { cache_.mtp_last_full_layer = il; break; }
        }
        if (cache_.mtp_last_full_layer < 0) {
            std::fprintf(stderr, "[gemma4] error: no full-attention layer found in target\n");
            return false;
        }
    }

    if (args_.draft_method == Gemma4DraftMethod::kDFlash && !args_.draft_path.empty()) {
        if (!load_gemma4_draft_safetensors(args_.draft_path, backend_, draft_w_) &&
            !load_gemma4_draft_gguf(args_.draft_path, backend_, draft_w_)) {
            std::fprintf(stderr, "[gemma4] load_gemma4_draft_*: %s\n",
                         dflash27b_last_error());
            return false;
        }
        draft_loaded_ = true;
        // Inject target's tok_embd into draft (tied LM head — see
        // gemma4_dflash_graph.cpp comments at line 777). The draft safetensors
        // do not ship a tok_embd of their own; build_gemma4_draft_graph's
        // final ggml_mul_mat(w.tok_embd, out) would dereference NULL otherwise.
        draft_w_.tok_embd = target_w_.tok_embd;
        if (!create_draft_kv_cache(draft_w_, backend_, cache_,
                                   args_.draft_kv_cap_override)) {
            std::fprintf(stderr, "[gemma4] create_draft_kv_cache: %s\n",
                         dflash27b_last_error());
            return false;
        }
    }

    return true;
}

// ── banner ──────────────────────────────────────────────────────────────

void Gemma4Backend::print_ready_banner() const {
    const char * method = "none";
    switch (args_.draft_method) {
        case Gemma4DraftMethod::kDFlash: method = "dflash"; break;
        case Gemma4DraftMethod::kMtp:    method = "mtp";    break;
        case Gemma4DraftMethod::kNone:   method = "none";   break;
    }
    std::printf("[gemma4-daemon] ready n_layer=%d n_embd=%d vocab=%d max_ctx=%d "
                "draft=%s sparse_fa=%d\n",
                target_w_.n_layer, target_w_.n_embd, target_w_.n_vocab,
                args_.max_ctx, method, (int)args_.use_sparse_fa);
    std::fflush(stdout);
}

// ── park / unpark (target only; draft/mtp parking lands with the runtime PR) ──

bool Gemma4Backend::park(const std::string & what) {
    if (what.empty() || what == "all" || what == "target") {
        // Cache + weights remain resident; we just flag the state so the loop
        // refuses to generate. Real park frees GPU buffers — follow-up.
        target_parked_ = true;
        return true;
    }
    std::fprintf(stderr, "[gemma4] park: '%s' not yet implemented\n", what.c_str());
    return false;
}

bool Gemma4Backend::unpark(const std::string & what) {
    if (what.empty() || what == "all" || what == "target") {
        target_parked_ = false;
        return true;
    }
    return false;
}

// ── snapshots (stubbed at the table level; real save/restore in follow-up) ──

bool Gemma4Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= kMaxSlots) return false;
    slots_[slot].used    = true;
    slots_[slot].cur_pos = cache_.cur_pos;
    return true;
}

void Gemma4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= kMaxSlots) return;
    slots_[slot].used    = false;
    slots_[slot].cur_pos = -1;
}

bool Gemma4Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= kMaxSlots) return false;
    return slots_[slot].used;
}

int Gemma4Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= kMaxSlots) return -1;
    return slots_[slot].cur_pos;
}

GenerateResult Gemma4Backend::restore_and_generate(int /*slot*/,
                                                   const GenerateRequest & /*req*/,
                                                   const DaemonIO & /*io*/) {
    GenerateResult r;
    r.ok    = false;
    r.error = "restore_not_implemented";
    return r;
}

// ── handle_compress / free_drafter (PFlash compress lands with the runtime PR) ──

bool Gemma4Backend::handle_compress(const std::string & /*line*/,
                                    const DaemonIO & /*io*/) {
    std::fprintf(stderr, "[gemma4] compress not yet implemented\n");
    return false;
}

void Gemma4Backend::free_drafter() {
    if (draft_loaded_) {
        free_draft_kv_cache(cache_);
        free_gemma4_draft_weights(draft_w_);
        draft_loaded_ = false;
    }
}

// ── generate ────────────────────────────────────────────────────────────

GenerateResult Gemma4Backend::generate(const GenerateRequest & req,
                                       const DaemonIO & io) {
    GenerateResult result;
    const int N = (int)req.prompt.size();
    if (N == 0 || req.n_gen <= 0 || N + req.n_gen > args_.max_ctx) {
        result.error = "overflow";
        return result;
    }

    reset_gemma4_cache(cache_);

    std::vector<float> last_logits;
    if (!prefill(req.prompt, last_logits, result.prefill_s)) {
        result.error = "prefill";
        return result;
    }

    // Inline snapshot if requested
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= N) {
        if (snapshot_save(req.snap_slot)) {
            slots_[req.snap_slot].cur_pos = req.snap_pos;
        }
    }

    std::vector<int32_t> tokens;
    bool ok = false;
    switch (args_.draft_method) {
        case Gemma4DraftMethod::kDFlash:
            ok = decode_dflash(req.n_gen, last_logits, req, io, tokens, result.decode_s);
            break;
        case Gemma4DraftMethod::kMtp:
            ok = decode_mtp(req.n_gen, last_logits, req, io, tokens, result.decode_s);
            break;
        case Gemma4DraftMethod::kNone:
        default:
            ok = decode_autoregressive(req.n_gen, last_logits, req, io, tokens, result.decode_s);
            break;
    }

    if (!ok) {
        result.error = "decode";
        return result;
    }
    result.ok     = true;
    result.tokens = std::move(tokens);
    return result;
}

// ── shutdown ────────────────────────────────────────────────────────────

void Gemma4Backend::shutdown() {
    free_drafter();
    if (mtp_loaded_) {
        free_gemma4_mtp_assistant(mtp_w_);
        mtp_loaded_ = false;
    }
    free_mtp_h_prev(cache_);
    free_gemma4_cache(cache_);
    free_gemma4_target_weights(target_w_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

// ── internal: shared draft-KV-prefill helper ─────────────────────────────
//
// Four sites in decode_dflash (initial / warmup / sliding re-prefill /
// commit) each issued the same build-fill-compute-destroy sequence on a
// DraftKVPrefillGraph, varying only in (start_pos, n_tokens, log tag).
// Each duplication was a chance to mis-tune one of the args (e.g. the
// `start_pos % cache.target_feat_cap` ring index) and silently corrupt
// the draft KV.
//
// Helper does NOT update cache.draft_kv_pos — each phase has its own
// semantics for the post-update (initial = n, warmup = +1 capped,
// re-prefill = keep, commit = +commit_n capped), so the caller stays in
// charge of that.
//
// See dflash/docs/gemma4-pr-split/pr13-slop-audit.md (Finding S5).
static bool run_draft_kv_prefill(GemmaTargetCache        & cache,
                                 const GemmaDraftWeights & draft_w,
                                 ggml_backend_t            backend,
                                 int                       start_pos,
                                 int                       n_tokens,
                                 int                       target_feat_w,
                                 const char              * tag) {
    DraftKVPrefillGraph pkg;
    if (!draft_kv_prefill_create(pkg, draft_w, cache, backend, n_tokens)) {
        std::fprintf(stderr, "[gemma4] %s draft KV prefill build failed\n", tag);
        return false;
    }
    copy_target_feat_bf16_to_f32(backend, cache.target_feat, pkg.target_feat,
                                  start_pos % cache.target_feat_cap,
                                  n_tokens, target_feat_w);
    if (n_tokens == 1) {
        int32_t p = start_pos;
        ggml_backend_tensor_set(pkg.positions, &p, 0, sizeof(int32_t));
    } else {
        std::vector<int32_t> pos(n_tokens);
        for (int i = 0; i < n_tokens; i++) pos[i] = start_pos + i;
        ggml_backend_tensor_set(pkg.positions, pos.data(), 0,
                                sizeof(int32_t) * n_tokens);
    }
    auto st = ggml_backend_graph_compute(backend, pkg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[gemma4] %s draft KV prefill compute failed\n", tag);
        draft_kv_prefill_destroy(pkg);
        return false;
    }
    draft_kv_prefill_destroy(pkg);
    return true;
}

// ── internal: shared mask-write helper ───────────────────────────────────
//
// Five decode paths (prefill, AR, DFlash warmup, DFlash verify, MTP target
// step) emitted the same attn_mask + swa_mask host build + tensor_set
// sequence, differing only by (kv_start, n_tokens) and the swa_window
// source. Each duplication was an opportunity to silently mis-tune one
// site and break long-context decode.
//
// Helper consumes the StepGraph as-built, the cache (for the SWA view's
// effective ring length), and the (kv_start, n_tokens, swa_window) triple.
// Internally delegates to build_causal_mask / build_swa_causal_mask
// (helpers.cpp) — those are unchanged.
//
// See dflash/docs/gemma4-pr-split/pr13-slop-audit.md (Finding S4).
static int ddtree_slot_depth(const DDTree & tree, int slot) {
    return slot == 0 ? 0 : tree.depths[slot - 1];
}

static void build_tree_full_mask(std::vector<uint16_t> & out,
                                 const DDTree & tree,
                                 int kv_start) {
    const int N      = 1 + tree.n_nodes;
    const int kv_len = kv_start + N;
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(N, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);

    for (int q = 0; q < N; q++) {
        for (int k = 0; k < kv_start; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
        }
        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j]) {
                out[(size_t)q * kv_pad + kv_start + j] = F16_ZERO;
            }
        }
    }
}

static void build_tree_swa_mask(std::vector<uint16_t> & out,
                                const DDTree & tree,
                                int kv_start,
                                int swa_window,
                                int ring_size,
                                int kv_end) {
    const int N      = 1 + tree.n_nodes;
    const int kv_pad = align_up(ring_size, g_kq_stride_pad);
    const int q_pad  = align_up(N, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);

    const int latest_slot = ((kv_end - 1) % ring_size + ring_size) % ring_size;
    for (int q = 0; q < N; q++) {
        const int abs_q = kv_start + ddtree_slot_depth(tree, q);
        const int q_lo  = std::max(0, abs_q - swa_window + 1);

        for (int k_view = 0; k_view < ring_size; k_view++) {
            const int offset_back = (latest_slot - k_view + ring_size) % ring_size;
            const int abs_k       = (kv_end - 1) - offset_back;
            const bool valid_past = (abs_k >= q_lo && abs_k < kv_start && abs_k >= 0);
            if (valid_past) {
                out[(size_t)q * kv_pad + k_view] = F16_ZERO;
            }
        }

        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j]) {
                const int slot = (kv_start + j) % ring_size;
                out[(size_t)q * kv_pad + slot] = F16_ZERO;
            }
        }
    }
}

static void set_step_masks(StepGraph         & sg,
                           GemmaTargetCache  & cache,
                           int                 kv_start,
                           int                 n_tokens,
                           int                 swa_window,
                           const DDTree      * tree = nullptr) {
    if (sg.attn_mask && sg.attn_mask->buffer) {
        std::vector<uint16_t> mask_buf;
        if (tree) {
            build_tree_full_mask(mask_buf, *tree, kv_start);
        } else {
            const int kv_len = kv_start + n_tokens;
            build_causal_mask(mask_buf, kv_len, n_tokens, kv_start);
        }
        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
    }
    if (sg.swa_mask && sg.swa_mask->buffer) {
        const SwaView swa_view = compute_swa_view(kv_start, n_tokens,
                                                   swa_window, cache.swa_ctx_alloc);
        std::vector<uint16_t> swa_buf;
        if (tree) {
            build_tree_swa_mask(swa_buf,
                                *tree,
                                /*kv_start*/ kv_start,
                                /*swa_window*/ swa_window,
                                /*ring_size*/ swa_view.effective_win_len,
                                /*kv_end*/ kv_start + n_tokens);
        } else {
            build_swa_causal_mask(swa_buf,
                                  /*kv_start*/ kv_start,
                                  /*n_tokens*/ n_tokens,
                                  /*swa_window*/ swa_window,
                                  /*ring_size*/ swa_view.effective_win_len,
                                  /*kv_end*/ kv_start + n_tokens);
        }
        ggml_backend_tensor_set(sg.swa_mask, swa_buf.data(), 0,
                                sizeof(uint16_t) * swa_buf.size());
    }
}

static bool copy_tensor_slice(ggml_tensor * t,
                              size_t dst_off,
                              size_t src_off,
                              size_t nbytes,
                              const char * what) {
    if (dst_off == src_off || nbytes == 0) return true;
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)
    // Synchronous copy: subsequent ggml graph runs on non-blocking backend
    // streams that are not ordered after async copies on the null stream.
    // Use cudaMemcpy so the next graph cannot read stale/uncompacted data.
    cudaError_t ce = cudaMemcpy((char *)t->data + dst_off,
                                (const char *)t->data + src_off,
                                nbytes, cudaMemcpyDeviceToDevice);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "[gemma4] %s copy failed: %s\n",
                     what, cudaGetErrorString(ce));
        return false;
    }
#else
    std::vector<uint8_t> tmp(nbytes);
    ggml_backend_tensor_get(t, tmp.data(), src_off, nbytes);
    ggml_backend_tensor_set(t, tmp.data(), dst_off, nbytes);
#endif
    return true;
}

static bool compact_gemma4_tree_path(GemmaTargetCache & cache,
                                     const GemmaTargetWeights & w,
                                     int committed,
                                     const std::vector<int> & accepted,
                                     int commit_n) {
    for (int d = 1; d < commit_n; d++) {
        const int src_dfs = accepted[d];
        if (src_dfs == d) continue;

        if (cache.target_feat && cache.target_feat_cap > 0) {
            const int src_slot = (committed + src_dfs) % cache.target_feat_cap;
            const int dst_slot = (committed + d)       % cache.target_feat_cap;
            const size_t src_off = (size_t)src_slot * cache.target_feat->nb[1];
            const size_t dst_off = (size_t)dst_slot * cache.target_feat->nb[1];
            if (!copy_tensor_slice(cache.target_feat, dst_off, src_off,
                                   (size_t)cache.target_feat->nb[1],
                                   "target_feat compact")) {
                return false;
            }
        }

        for (int il = 0; il < w.n_layer; il++) {
            const int kv_idx = (il < (int)cache.layer_to_kv_idx.size())
                                   ? cache.layer_to_kv_idx[il] : -1;
            if (kv_idx < 0) continue;
            ggml_tensor * ck = cache.attn_k[kv_idx];
            ggml_tensor * cv = cache.attn_v[kv_idx];
            const bool is_swa = (il < (int)w.swa_layers.size()) && w.swa_layers[il];
            const int ctx_slots = (int)ck->ne[1];
            const int src_slot = is_swa ? ((committed + src_dfs) % ctx_slots)
                                        : (committed + src_dfs);
            const int dst_slot = is_swa ? ((committed + d) % ctx_slots)
                                        : (committed + d);
            const int n_kv = (int)ck->ne[2];
            for (int h = 0; h < n_kv; h++) {
                const size_t k_src_off = (size_t)src_slot * ck->nb[1] + (size_t)h * ck->nb[2];
                const size_t k_dst_off = (size_t)dst_slot * ck->nb[1] + (size_t)h * ck->nb[2];
                const size_t v_src_off = (size_t)src_slot * cv->nb[1] + (size_t)h * cv->nb[2];
                const size_t v_dst_off = (size_t)dst_slot * cv->nb[1] + (size_t)h * cv->nb[2];
                if (!copy_tensor_slice(ck, k_dst_off, k_src_off, (size_t)ck->nb[1], "KV-K compact") ||
                    !copy_tensor_slice(cv, v_dst_off, v_src_off, (size_t)cv->nb[1], "KV-V compact")) {
                    return false;
                }
            }
        }
    }
    return true;
}

// ── internal: prefill ────────────────────────────────────────────────────
//
// Runs chunked prefill via build_gemma4_graph; returns last-token logits.
// Real implementation involves ggml_init / ggml_new_graph / build_gemma4_graph
// / ggml_backend_graph_compute. Stubbed here with a clear marker; the AR
// decode loop below depends on this and will be exercised once the body
// lands in the runtime follow-up PR.

bool Gemma4Backend::prefill(const std::vector<int32_t> & prompt,
                            std::vector<float> & out_last_logits,
                            double & out_prefill_s) {
    const int n_prompt   = (int)prompt.size();
    const int swa_window = target_w_.swa_window > 0 ? target_w_.swa_window : 1024;
    const int chunk_size = std::min(n_prompt, swa_window);

    StepGraph sg{};
    bool ok = true;

    auto t0 = std::chrono::steady_clock::now();

    for (int cs = 0; cs < n_prompt; cs += chunk_size) {
        const int chunk_n  = std::min(chunk_size, n_prompt - cs);
        const bool is_last = (cs + chunk_n == n_prompt);
        const bool need_mask = (cs + chunk_n > 1);

        if (!gemma4_step(sg, target_w_, cache_, backend_,
                               cs, chunk_n, need_mask,
                               /*capture=*/true,
                               args_.use_sparse_fa, args_.sparse_fa_alpha,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/true)) {
            std::fprintf(stderr, "[gemma4] prefill build failed at %d\n", cs);
            ok = false;
            break;
        }
        if (!embed_tokens_batch(target_w_, prompt.data() + cs, chunk_n,
                                sg.inp_embed, backend_)) {
            std::fprintf(stderr, "[gemma4] embed_tokens_batch failed\n");
            ok = false;
            break;
        }

        {
            std::vector<int32_t> pos(chunk_n);
            for (int i = 0; i < chunk_n; i++) pos[i] = cs + i;
            ggml_backend_tensor_set(sg.positions, pos.data(), 0,
                                    sizeof(int32_t) * chunk_n);
        }

        set_step_masks(sg, cache_, /*kv_start*/ cs, /*n_tokens*/ chunk_n, swa_window);

        auto st = ggml_backend_graph_compute(backend_, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[gemma4] prefill compute failed at %d\n", cs);
            ok = false;
            break;
        }

        cache_.cur_pos = cs + chunk_n;

        if (is_last) {
            const int vocab = target_w_.n_vocab;
            out_last_logits.resize(vocab);
            ggml_backend_tensor_get(sg.logits, out_last_logits.data(),
                                    0, sizeof(float) * vocab);
        }

        // Note: do NOT step_graph_free(sg) here. gemma4_step calls
        // step_graph_free(sg) at its entry (helpers.cpp:97), so each
        // iteration's sg is freed by the next build. The single trailing
        // step_graph_free below owns final cleanup for both the happy path
        // (last iteration's sg) and the error-break path (partially-built sg).
    }

    auto t1 = std::chrono::steady_clock::now();
    out_prefill_s = std::chrono::duration<double>(t1 - t0).count();

    step_graph_free(sg);
    return ok && !out_last_logits.empty();
}

// ── internal: AR decode ─────────────────────────────────────────────────

static bool is_gemma4_eos(const GemmaTargetWeights & w, int32_t tok);

bool Gemma4Backend::decode_autoregressive(int n_gen,
                                          std::vector<float> & last_logits_io,
                                          const GenerateRequest & req,
                                          const DaemonIO & io,
                                          std::vector<int32_t> & out_tokens,
                                          double & out_decode_s) {
    out_tokens.clear();
    out_tokens.reserve(n_gen);

    // history for repetition penalty — start with the full prompt
    std::vector<int32_t> history(req.prompt.begin(), req.prompt.end());

    const bool do_sample = req.sampler.temp > 0.0f;

    auto argmax = [](const std::vector<float> & ll) {
        int best = 0; float bv = ll[0];
        for (size_t i = 1; i < ll.size(); ++i)
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        return best;
    };

    auto pick = [&](const std::vector<float> & ll) -> int {
        return do_sample
            ? sample_logits(ll.data(), (int)ll.size(), req.sampler, history, sampler_rng_)
            : argmax(ll);
    };

    // First token sampled from prefill logits
    int32_t cur_tok = (int32_t)pick(last_logits_io);

    StepGraph sg{};
    bool ok = true;

    auto t0 = std::chrono::steady_clock::now();

    for (int s = 0; s < n_gen; ++s) {
        // EOS check
        if (!args_.ignore_eos && is_gemma4_eos(target_w_, cur_tok)) {
            break;
        }

        out_tokens.push_back(cur_tok);
        history.push_back(cur_tok);

        if (req.stream) {
            io.emit(cur_tok);
        }

        const int committed = cache_.cur_pos;

        if (committed >= args_.max_ctx - CTX_HEADROOM_AR) {
            break;
        }

        if (!gemma4_step(sg, target_w_, cache_, backend_,
                               committed, /*n_tokens=*/1,
                               /*with_mask=*/true,
                               /*capture=*/false,
                               /*use_sparse_fa=*/false, args_.sparse_fa_alpha,
                               /*fa_window=*/0)) {
            std::fprintf(stderr, "[gemma4] AR build failed at step %d\n", s);
            ok = false;
            break;
        }

        {
            const int swa_window = target_w_.swa_window > 0 ? target_w_.swa_window : 1024;
            set_step_masks(sg, cache_, /*kv_start*/ committed, /*n_tokens*/ 1, swa_window);
        }

        if (!embed_token(target_w_, cur_tok, sg.inp_embed, backend_)) {
            std::fprintf(stderr, "[gemma4] embed_token failed for tok=%d\n", cur_tok);
            ok = false;
            break;
        }

        {
            int32_t pos_val = committed;
            ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));
        }

        {
            auto st = ggml_backend_graph_compute(backend_, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[gemma4] AR compute failed at step %d\n", s);
                ok = false;
                break;
            }
        }

        cache_.cur_pos = committed + 1;

        const int vocab = target_w_.n_vocab;
        std::vector<float> logits_cpu(vocab);
        ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                sizeof(float) * vocab);
        cur_tok = (int32_t)pick(logits_cpu);

        // Note: do NOT step_graph_free(sg) here. gemma4_step calls
        // step_graph_free(sg) at its entry (helpers.cpp:97), so the next
        // iteration's build frees this iteration's sg. Single trailing
        // step_graph_free below owns final cleanup.
    }

    auto t1 = std::chrono::steady_clock::now();
    out_decode_s = std::chrono::duration<double>(t1 - t0).count();

    step_graph_free(sg);

    if (req.stream) {
        io.emit(-1);
    }

    return ok;
}

// ── internal: DFlash speculative-decode ─────────────────────────────────
//
// Ported from feature/gemma4-support test_gemma4_dflash.cpp
// Range A (1495-1827) preamble + Range B (1828-2399) have_draft arm.
// AR fallback arm dropped (handled by decode_autoregressive).
// Stripped: IS_EOS_TOK macro (inlined), printf scaffolding, bench_mode,
// stream_emit, daemon_first_iter, getenv probes, sample_logits (DFlash
// uses argmax for both draft and target verify).

static int argmax_f32(const float * x, int n) {
    int best = 0; float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

static bool is_gemma4_eos(const GemmaTargetWeights & w, int32_t tok) {
    return (w.eos_id >= 0 && tok == w.eos_id) ||
           (w.eos_chat_id >= 0 && tok == w.eos_chat_id);
}

bool Gemma4Backend::decode_dflash(int n_gen,
                                  std::vector<float> & last_logits_io,
                                  const GenerateRequest & req,
                                  const DaemonIO & io,
                                  std::vector<int32_t> & out_tokens,
                                  double & out_decode_s) {
    out_tokens.clear();
    out_tokens.reserve(n_gen);

    // ── Initial sample from prefill logits ──────────────────────────────────
    const int vocab          = target_w_.n_vocab;
    const int ctx_size       = args_.max_ctx;
    const int target_feat_w  = draft_w_.n_target_layers * draft_w_.target_hidden;
    const int ddtree_budget  = std::max(0, args_.ddtree_budget);
    const bool ddtree_enabled = ddtree_budget > 0;
    const int dkv_cap        = (cache_.draft_kv_cap > 0)
                                   ? cache_.draft_kv_cap
                                   : (cache_.draft_k.empty() ? 0 : (int)cache_.draft_k[0]->ne[2]);

    int32_t cur_tok = (int32_t)argmax_f32(last_logits_io.data(), vocab);
    cache_.last_tok = cur_tok;

    // ── Draft KV prefill: project prompt target_feat into draft KV cache ────
    {
        const int n_prompt       = (int)req.prompt.size();
        const int draft_prefill_n    = std::min(n_prompt, dkv_cap);
        const int draft_prefill_skip = n_prompt - draft_prefill_n;

        if (!run_draft_kv_prefill(cache_, draft_w_, backend_,
                                   /*start_pos=*/ draft_prefill_skip,
                                   /*n_tokens=*/  draft_prefill_n,
                                   target_feat_w,
                                   /*tag=*/       "initial")) {
            return false;
        }
        cache_.draft_kv_pos = draft_prefill_n;
    }

    // ── Speculative decode loop ──────────────────────────────────────────────
    //
    // Each iteration proposes a block of q_len tokens via the draft model,
    // verifies with a single batched target forward, accepts the longest prefix.
    // Gemma4 is pure attention — rollback is trivially: don't advance committed
    // past the accepted prefix (stale KV is overwritten by the next verify pass).

    AdaptiveDraftMax adaptive;
    // Adaptive draft-block tuning. Heuristic: enable on MoE only.
    //   - MoE (Gemma4-26B-A4B, n_layer=30): adaptive @ floor=8 yields
    //     +1.66x over AR median on the 350W full-power bench (active
    //     params are small; small batch with high accept dominates).
    //   - Dense (Gemma4-31B, n_layer=61): adaptive regresses 30-60% on
    //     hard-to-draft prompts at any floor — chain dm=16 fixed is the
    //     stronger default because Dense weight reads amortize across the
    //     full block anyway.
    bool is_moe = (target_w_.n_layer <= 40);
    // Bench/debug override: DFLASH_NO_ADAPTIVE=1 forces adaptive off so the
    // same binary can produce both adapt-ON and adapt-OFF measurements in
    // a single session.
    if (const char * e = std::getenv("DFLASH_NO_ADAPTIVE")) {
        if (e[0] && e[0] != '0') is_moe = false;
    }
    adaptive.init(/*on=*/is_moe, args_.draft_max_block, draft_w_.block_size);
    if (is_moe) {
        adaptive.min_q = std::min(8, draft_w_.block_size);
        adaptive.current = std::max(adaptive.current, adaptive.min_q);
    }

    const int mask_tok = draft_w_.mask_token_id;
    // The DFlash noise block fills positions [1..q_len-1] with mask_tok. If the
    // draft loader ever ships a sentinel (-1) mask token, downstream behavior
    // is undefined — the embedding table get_rows would read out of bounds.
    // Catch that at decode entry rather than later when the embedder fails
    // with a less actionable error.
    GGML_ASSERT(mask_tok >= 0 && "DFlash drafter mask_token_id must be set "
                                  "(see Gemma4DraftWeights::mask_token_id; "
                                  "Finding S10)");
    const int swa_window = target_w_.swa_window > 0 ? target_w_.swa_window : 1024;

    std::vector<int32_t> noise_ids(draft_w_.block_size);
    // noise_embed_buf uses draft_w_.n_embd (= target_w_.n_embd; shared embedding table)
    std::vector<float>   noise_embed_buf((size_t)draft_w_.n_embd * draft_w_.block_size);
    std::vector<int32_t> draft_tok(draft_w_.block_size);
    const int verify_cap = draft_w_.block_size;
    std::vector<int32_t> target_tok(verify_cap);
    std::vector<float>   draft_logits_buf((size_t)vocab * draft_w_.block_size);
    std::vector<float>   verify_logits_buf((size_t)vocab * verify_cap);
    std::vector<float>   ddtree_top_log_probs;
    std::vector<int32_t> ddtree_top_token_ids;

    int committed         = cache_.cur_pos;
    int total_draft_steps = 0;
    int total_accepted    = 0;

    // Persistent tree verify graph — gallocr reuse across DDTree iterations
    StepGraph tree_sg{};

    auto t0 = std::chrono::steady_clock::now();

    while ((int)out_tokens.size() < n_gen) {
        int q_len = adaptive.enabled
                        ? adaptive.current
                        : ((args_.draft_max_block > 0 && args_.draft_max_block < draft_w_.block_size)
                               ? args_.draft_max_block : draft_w_.block_size);
        q_len = std::min(q_len, std::max(1, ctx_size - committed - 1));

        // EOS check (inlined IS_EOS_TOK)
        if (!args_.ignore_eos && is_gemma4_eos(target_w_, cur_tok)) {
            break;
        }
        if (committed >= ctx_size - CTX_HEADROOM_DFLASH) {
            break;
        }

        // Warmup: not enough context for target_feat extraction — fall back to
        // single-token target-only decode until committed >= q_len.
        if (committed < q_len) {
            StepGraph sg{};
            if (!gemma4_step(sg, target_w_, cache_, backend_,
                                   committed, /*n_tokens=*/1,
                                   /*with_mask=*/true,
                                   /*capture=*/true,
                                   /*use_sparse_fa=*/false, args_.sparse_fa_alpha,
                                   /*fa_window=*/0)) {
                std::fprintf(stderr, "[gemma4] dflash warmup build failed at step %zu\n",
                             out_tokens.size());
                return false;
            }

            set_step_masks(sg, cache_, /*kv_start*/ committed, /*n_tokens*/ 1, swa_window);

            if (!embed_token(target_w_, cur_tok, sg.inp_embed, backend_)) {
                std::fprintf(stderr, "[gemma4] dflash warmup embed failed\n");
                return false;
            }

            {
                int32_t pos_val = committed;
                ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));
            }

            auto st = ggml_backend_graph_compute(backend_, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[gemma4] dflash warmup compute failed at step %zu\n",
                             out_tokens.size());
                return false;
            }

            committed++;
            cache_.cur_pos = committed;

            // Draft KV prefill for this warmup position
            {
                const int warmup_pos = committed - 1;
                if (!run_draft_kv_prefill(cache_, draft_w_, backend_,
                                           /*start_pos=*/ warmup_pos,
                                           /*n_tokens=*/  1,
                                           target_feat_w,
                                           /*tag=*/       "warmup")) {
                    return false;
                }
                cache_.draft_kv_pos = std::min(dkv_cap, cache_.draft_kv_pos + 1);
            }

            // Sample next token from warmup logits (argmax for DFlash path)
            {
                std::vector<float> logits_cpu(vocab);
                ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                        sizeof(float) * vocab);
                const int32_t next_tok = (int32_t)argmax_f32(logits_cpu.data(), vocab);

                out_tokens.push_back(cur_tok);
                if (req.stream) io.emit(cur_tok);

                cur_tok = next_tok;
                cache_.last_tok = cur_tok;
            }

            step_graph_free(sg);
            continue;
        }

        // ── 1. Build noise block: [cur_tok, MASK, MASK, ..., MASK] ──────────
        noise_ids[0] = cur_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = mask_tok;
        if (!target_w_.embedder.embed(noise_ids.data(), q_len, noise_embed_buf.data())) {
            std::fprintf(stderr, "[gemma4] embed noise_ids failed\n");
            return false;
        }

        // ── 2. Draft graph — with sliding-window KV re-prefill if needed ────
        if (cache_.draft_kv_pos + q_len > dkv_cap) {
            const int keep = dkv_cap - q_len;
            if (keep > 0 && committed >= keep) {
                const int refill_start = committed - keep;
                cache_.draft_kv_pos    = 0;

                if (!run_draft_kv_prefill(cache_, draft_w_, backend_,
                                           /*start_pos=*/ refill_start,
                                           /*n_tokens=*/  keep,
                                           target_feat_w,
                                           /*tag=*/       "sliding re-prefill")) {
                    return false;
                }
                cache_.draft_kv_pos = keep;

                std::fprintf(stderr,
                    "[gemma4] draft KV sliding re-prefill: kept %d tokens "
                    "(positions %d..%d), dkv_cap=%d\n",
                    keep, refill_start, committed - 1, dkv_cap);
            } else {
                cache_.draft_kv_pos = 0;
            }
        }

        DraftStepGraph dsg;
        if (!draft_step_build(dsg, draft_w_, cache_, backend_, q_len, cache_.draft_kv_pos)) {
            std::fprintf(stderr, "[gemma4] draft build failed\n");
            return false;
        }

        // ── 3. Set draft inputs ──────────────────────────────────────────────
        ggml_backend_tensor_set(dsg.draft_embed, noise_embed_buf.data(), 0,
                                sizeof(float) * (size_t)draft_w_.n_embd * q_len);

        {
            std::vector<int32_t> pos(q_len);
            for (int i = 0; i < q_len; i++) pos[i] = committed + i;
            ggml_backend_tensor_set(dsg.positions, pos.data(), 0,
                                    sizeof(int32_t) * q_len);
        }

        // Draft causal mask: token i attends over [0..draft_kv_pos-1] + block [0..i]
        if (dsg.attn_mask && dsg.attn_mask->buffer) {
            const int dkv_ctx = cache_.draft_kv_pos;
            const int kv_len  = dkv_ctx + q_len;
            const int kv_pad  = align_up(kv_len, KQ_MASK_PAD);
            const int q_pad   = align_up(q_len,  KQ_MASK_PAD);
            std::vector<uint16_t> mask((size_t)kv_pad * q_pad, F16_NEG_INF);
            for (int q = 0; q < q_len; q++) {
                const int max_k = dkv_ctx + q;
                for (int k = 0; k <= max_k; k++) {
                    mask[(size_t)q * kv_pad + k] = F16_ZERO;
                }
            }
            ggml_backend_tensor_set(dsg.attn_mask, mask.data(), 0,
                                    sizeof(uint16_t) * mask.size());
        }

        // ── 4. Draft compute ─────────────────────────────────────────────────
        {
            auto st = ggml_backend_graph_compute(backend_, dsg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[gemma4] draft compute failed: %d\n", (int)st);
                return false;
            }
        }

        // ── 5. Read draft logits and argmax ──────────────────────────────────
        ggml_backend_tensor_get(dsg.logits, draft_logits_buf.data(), 0,
                                sizeof(float) * (size_t)vocab * q_len);
        for (int i = 0; i < q_len; i++) {
            draft_tok[i] = argmax_f32(draft_logits_buf.data() + (size_t)i * vocab, vocab);
        }
        // Pin draft_tok[0] to the *id stream* value (cur_tok). The pin patches
        // only the token-id sequence used by acceptance; the noise embedding
        // at noise_embed_buf row 0 was already cur_tok-embedded above (line
        // 697: noise_ids[0] = cur_tok), so the embedding is consistent.
        // The mask_tok assert at function entry guards against an
        // uninitialized sentinel breaking the rest of the row fill.
        draft_tok[0] = cur_tok;

        if (ddtree_enabled) {
            const int tree_budget = std::min(ddtree_budget,
                                             std::max(0, ctx_size - committed - 1));
            const int L = q_len - 1;
            const int ddtree_K = (tree_budget > L) ? 8 : 1;
            if (tree_budget <= 0) {
                draft_step_free(dsg);
                break;
            }
            ddtree_top_log_probs.resize((size_t)std::max(0, L) * ddtree_K);
            ddtree_top_token_ids.resize((size_t)std::max(0, L) * ddtree_K);

            if (L > 0) {
                if (ddtree_K == 1) {
                    for (int i = 0; i < L; i++) {
                        ddtree_top_log_probs[i] = 0.0f;
                        ddtree_top_token_ids[i] = draft_tok[i + 1];
                    }
                } else {
                    extract_draft_topk(draft_logits_buf.data() + (size_t)vocab,
                                       L, vocab, ddtree_K,
                                       ddtree_top_log_probs.data(),
                                       ddtree_top_token_ids.data(),
                                       args_.ddtree_temp);
                }
            }

            DDTree tree = build_ddtree(ddtree_top_log_probs.data(),
                                       ddtree_top_token_ids.data(),
                                       L, ddtree_K, tree_budget,
                                       args_.ddtree_chain_seed);
            const int N_actual = 1 + tree.n_nodes;
            if ((int)target_tok.size() < N_actual) target_tok.resize(N_actual);

            // Build (or reuse) persistent tree verify graph at the full
            // budget+1 padding shape so gallocr reservation is reused across
            // iterations (semantic equivalence of padding slots is enforced
            // by the mask + position logic in build_gemma4_step_tree).
            const int n_max = tree_budget + 1;
            if (!build_gemma4_step_tree(tree_sg, target_w_, cache_, backend_,
                                        committed, swa_window, n_max, tree,
                                        /*capture_layers=*/true)) {
                std::fprintf(stderr, "[gemma4] ddtree verify build failed\n");
                return false;
            }

            // Embed: fill N_actual tree slots; padding slots stay zero
            {
                std::vector<int32_t> flat_tokens(n_max, 0);
                flat_tokens[0] = cur_tok;
                for (int i = 0; i < tree.n_nodes; i++) {
                    flat_tokens[i + 1] = tree.token_ids[i];
                }
                if (!embed_tokens_batch(target_w_, flat_tokens.data(), n_max,
                                        tree_sg.inp_embed, backend_)) {
                    return false;
                }
            }

            // ── positions & masks are set inside build_gemma4_step_tree ───

            auto st = ggml_backend_graph_compute(backend_, tree_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[gemma4] ddtree verify compute failed: %d\n", (int)st);
                return false;
            }

            // Read in-graph argmax (only N_actual valid entries)
            {
                std::vector<int32_t> posterior(N_actual);
                ggml_backend_tensor_get(tree_sg.argmax_tokens, posterior.data(), 0,
                                        sizeof(int32_t) * N_actual);
                for (int i = 0; i < N_actual; i++) target_tok[i] = posterior[i];
            }

            int next_token = -1;
            std::vector<int> accepted =
                follow_verified_tree(tree, target_tok.data(), next_token, nullptr);

            const int accept_depth = (int)accepted.size();
            int commit_n = accept_depth;
            if (commit_n > n_gen - (int)out_tokens.size()) {
                commit_n = n_gen - (int)out_tokens.size();
            }

            bool hit_eos = false;
            int emitted_n = 0;
            for (int i = 0; i < commit_n; i++) {
                const int dfs_idx = accepted[i];
                const int32_t tok = (dfs_idx == 0) ? cur_tok : tree.token_ids[dfs_idx - 1];
                out_tokens.push_back(tok);
                emitted_n++;
                if (req.stream) io.emit(tok);
                if (!args_.ignore_eos && is_gemma4_eos(target_w_, tok)) {
                    hit_eos = true;
                    break;
                }
            }
            commit_n = emitted_n;

            if (!compact_gemma4_tree_path(cache_, target_w_, committed,
                                          accepted, commit_n)) {
                return false;
            }

            if (!run_draft_kv_prefill(cache_, draft_w_, backend_,
                                       /*start_pos=*/ committed,
                                       /*n_tokens=*/  commit_n,
                                       target_feat_w,
                                       /*tag=*/       "ddtree commit")) {
                return false;
            }
            cache_.draft_kv_pos = std::min(dkv_cap, cache_.draft_kv_pos + commit_n);

            committed      += commit_n;
            cache_.cur_pos  = committed;
            // next_token (from follow_verified_tree) is the prediction beyond
            // the FULL accepted path. If we truncated early (commit_n <
            // accept_depth from in-loop EOS or n_gen cap), use the verified
            // target argmax at the actually-committed leaf instead — same
            // pattern the chain path uses (`target_tok[commit_n - 1]`).
            if (commit_n < accept_depth && commit_n > 0) {
                cur_tok = target_tok[accepted[commit_n - 1]];
            } else {
                cur_tok = next_token;
            }
            cache_.last_tok = cur_tok;

            total_draft_steps++;
            total_accepted += commit_n;
            adaptive.observe(accept_depth, q_len, total_draft_steps);

            draft_step_free(dsg);

            if (hit_eos) break;
            continue;
        }

        // ── 6. Target verify: batched forward on draft_tok[0..q_len-1] ───────
        {
            StepGraph sg{};
            if (!gemma4_step(sg, target_w_, cache_, backend_,
                                   committed, q_len,
                                   /*with_mask=*/true, /*capture=*/true,
                                   /*use_sparse_fa=*/false, args_.sparse_fa_alpha,
                                   /*fa_window=*/0)) {
                std::fprintf(stderr, "[gemma4] verify build failed\n");
                return false;
            }

            if (!embed_tokens_batch(target_w_, draft_tok.data(), q_len,
                                    sg.inp_embed, backend_)) {
                return false;
            }

            {
                std::vector<int32_t> pos(q_len);
                for (int i = 0; i < q_len; i++) pos[i] = committed + i;
                ggml_backend_tensor_set(sg.positions, pos.data(), 0,
                                        sizeof(int32_t) * q_len);
            }

            set_step_masks(sg, cache_, /*kv_start*/ committed, /*n_tokens*/ q_len, swa_window);

            auto st = ggml_backend_graph_compute(backend_, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[gemma4] verify compute failed: %d\n", (int)st);
                return false;
            }

            // ── 7. Read target logits and argmax ─────────────────────────────
            ggml_backend_tensor_get(sg.logits, verify_logits_buf.data(), 0,
                                    sizeof(float) * (size_t)vocab * q_len);
            for (int i = 0; i < q_len; i++) {
                target_tok[i] = argmax_f32(verify_logits_buf.data() + (size_t)i * vocab, vocab);
            }

            step_graph_free(sg);
        }

        draft_step_free(dsg);

        // ── 8. Acceptance: longest prefix match ──────────────────────────────
        //   draft_tok[0] = cur_tok (accepted unconditionally)
        //   Check: draft_tok[i+1] == target_tok[i]
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int commit_n = accept_n;
        if (commit_n > n_gen - (int)out_tokens.size()) {
            commit_n = n_gen - (int)out_tokens.size();
        }

        // ── 9. Commit accepted tokens ─────────────────────────────────────────
        bool hit_eos = false;
        for (int i = 0; i < commit_n; i++) {
            out_tokens.push_back(draft_tok[i]);
            if (req.stream) io.emit(draft_tok[i]);
            if (!args_.ignore_eos && is_gemma4_eos(target_w_, draft_tok[i])) {
                hit_eos = true;
                break;
            }
        }

        // ── 10. Draft KV prefill for committed positions, then advance state ──
        {
            if (!run_draft_kv_prefill(cache_, draft_w_, backend_,
                                       /*start_pos=*/ committed,
                                       /*n_tokens=*/  commit_n,
                                       target_feat_w,
                                       /*tag=*/       "commit")) {
                return false;
            }
            cache_.draft_kv_pos = std::min(dkv_cap, cache_.draft_kv_pos + commit_n);
        }

        // Gemma4 is pure attention — no SSM/conv rollback needed.
        // Stale KV at positions [committed+commit_n .. committed+q_len-1]
        // will be overwritten by the next verify pass.
        committed      += commit_n;
        cache_.cur_pos  = committed;
        cur_tok         = target_tok[commit_n - 1];
        cache_.last_tok = cur_tok;

        total_draft_steps++;
        total_accepted += commit_n;
        adaptive.observe(accept_n, q_len, total_draft_steps);

        if (hit_eos) break;
    }

    auto t1 = std::chrono::steady_clock::now();
    out_decode_s = std::chrono::duration<double>(t1 - t0).count();

    if (req.stream) {
        io.emit(-1);
    }
    if (total_draft_steps > 0) {
        // Goes to stderr: the daemon protocol owns stdout (line-oriented
        // `ok`/`err` responses); a stray `[spec]` line on stdout corrupts
        // client parsing for `generate` / bare-prompt requests.
        const double avg_accept = (double)total_accepted / (double)total_draft_steps;
        std::fprintf(stderr, "[spec] draft_steps=%d total_accepted=%d avg_accept=%.2f\n",
                     total_draft_steps, total_accepted, avg_accept);
        std::fflush(stderr);
    }

    step_graph_free(tree_sg);

    return true;
}

// ── internal: MTP decode (γ=1) ──────────────────────────────────────────────
//
// Ported from feature/gemma4-support test_gemma4_dflash.cpp lines 2575-2780
// (the `gamma == 1` branch of the `have_mtp` decode path).
//
// Stripped: [mtp] printf scaffolding, g_ignore_eos / IS_EOS_TOK macros
// (inlined using target_w_.eos_id / eos_chat_id), now_ms() / benchmark
// counters, stream_emit (tokens returned via out_tokens / io.emit).

bool Gemma4Backend::decode_mtp(int n_gen,
                               std::vector<float> & last_logits_io,
                               const GenerateRequest & req,
                               const DaemonIO & io,
                               std::vector<int32_t> & out_tokens,
                               double & out_decode_s) {
    out_tokens.clear();
    out_tokens.reserve(n_gen);

    const int vocab    = target_w_.n_vocab;
    const int ctx_size = args_.max_ctx;

    // history for repetition penalty — start with the full prompt
    std::vector<int32_t> history(req.prompt.begin(), req.prompt.end());

    const bool do_sample = req.sampler.temp > 0.0f;

    auto argmax = [](const std::vector<float> & ll) {
        int best = 0; float bv = ll[0];
        for (size_t i = 1; i < ll.size(); ++i)
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        return best;
    };

    auto pick = [&](const std::vector<float> & ll) -> int {
        return do_sample
            ? sample_logits(ll.data(), (int)ll.size(), req.sampler, history, sampler_rng_)
            : argmax(ll);
    };

    // First token sampled from prefill logits.
    int32_t cur_tok = (int32_t)pick(last_logits_io);

    // ── Initial MTP step graph (attn_pos=0; rebuilt each step) ──────────────
    MtpStepGraph mtp_g{};
    if (!build_mtp_step_graph(mtp_w_, cache_, target_w_, mtp_g, /*attn_pos=*/0)) {
        std::fprintf(stderr, "[gemma4] build_mtp_step_graph failed: %s\n",
                     dflash27b_last_error());
        return false;
    }

    // ── Persistent MTP gallocr ───────────────────────────────────────────────
    // The MTP graph topology is identical step-to-step — only attn_pos and
    // KV view length change. ggml_gallocr supports replanning a fresh graph
    // against the same allocator via ggml_gallocr_alloc_graph; reuse it
    // across the loop instead of new/free per token. Frees once at function
    // exit. See dflash/docs/gemma4-pr-split/pr13-slop-audit.md (Finding S3).
    ggml_gallocr_t mtp_alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend_));
    if (!mtp_alloc) {
        std::fprintf(stderr, "[gemma4] mtp gallocr_new failed\n");
        free_mtp_step_graph(mtp_g);
        return false;
    }

    int committed = cache_.cur_pos;

    auto t0 = std::chrono::steady_clock::now();

    // ── MTP SPECULATIVE DECODE LOOP (γ=1) ────────────────────────────────────
    //
    // Each iteration:
    //   1. Run target forward for cur_tok at position committed,
    //      capturing mtp_h_prev from the last full-attention layer.
    //   2. Rebuild MTP step graph with current attn_pos = committed+1.
    //   3. Feed (cur_tok, mtp_h_prev) into MTP graph → draft_tok.
    //   4. Run target verify forward for draft_tok at position committed+1.
    //   5. Accept draft_tok if target agrees; otherwise accept target's
    //      token instead (standard single-draft acceptance).

    while ((int)out_tokens.size() < n_gen) {

        // EOS check (inlined IS_EOS_TOK)
        if (!args_.ignore_eos && is_gemma4_eos(target_w_, cur_tok)) {
            break;
        }
        if (committed >= ctx_size - CTX_HEADROOM_MTP) {
            break;
        }

        // ── 1. Target forward for cur_tok (captures mtp_h_prev) ─────────────
        const int swa_window = target_w_.swa_window > 0 ? target_w_.swa_window : 1024;

        StepGraph sg{};
        if (!gemma4_step(sg, target_w_, cache_, backend_,
                               committed, /*n_tokens=*/1,
                               /*with_mask=*/true,
                               /*capture=*/false,
                               /*use_sparse_fa=*/false, args_.sparse_fa_alpha,
                               /*fa_window=*/0)) {
            std::fprintf(stderr, "[gemma4] mtp target build failed at step %zu\n",
                         out_tokens.size());
            ggml_gallocr_free(mtp_alloc);
            free_mtp_step_graph(mtp_g);
            return false;
        }

        set_step_masks(sg, cache_, /*kv_start*/ committed, /*n_tokens*/ 1, swa_window);
        if (!embed_token(target_w_, cur_tok, sg.inp_embed, backend_)) {
            std::fprintf(stderr, "[gemma4] mtp target embed_token failed for tok=%d\n",
                         cur_tok);
            step_graph_free(sg);
            ggml_gallocr_free(mtp_alloc);
            free_mtp_step_graph(mtp_g);
            return false;
        }
        {
            int32_t pos_val = committed;
            ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));
        }
        {
            auto st = ggml_backend_graph_compute(backend_, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[gemma4] mtp target compute failed\n");
                step_graph_free(sg);
                ggml_gallocr_free(mtp_alloc);
                free_mtp_step_graph(mtp_g);
                return false;
            }
        }
        committed++;
        cache_.cur_pos = committed;

        // Read target logits to get target's own prediction at position committed-1.
        std::vector<float> logits_cpu(vocab);
        ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                sizeof(float) * vocab);
        const int32_t target_next = (int32_t)pick(logits_cpu);

        step_graph_free(sg);

        // ── 2. Rebuild MTP step graph with attn_pos = committed ──────────────
        free_mtp_step_graph(mtp_g);
        if (!build_mtp_step_graph(mtp_w_, cache_, target_w_, mtp_g, committed)) {
            std::fprintf(stderr, "[gemma4] build_mtp_step_graph failed: %s\n",
                         dflash27b_last_error());
            ggml_gallocr_free(mtp_alloc);
            return false;
        }

        // Replan the persistent gallocr against the freshly-rebuilt MTP
        // graph. ggml_gallocr_alloc_graph reuses the same backend buffer
        // when the topology fits, or grows it as needed.
        if (!ggml_gallocr_alloc_graph(mtp_alloc, mtp_g.gf)) {
            std::fprintf(stderr, "[gemma4] mtp gallocr_alloc_graph failed\n");
            ggml_gallocr_free(mtp_alloc);
            free_mtp_step_graph(mtp_g);
            return false;
        }

        // ── 3. Set MTP inputs and compute ────────────────────────────────────
        // in_tok_embd: pre-dequantised F32 embedding of cur_tok.
        if (!embed_token(target_w_, cur_tok, mtp_g.in_tok_embd, backend_)) {
            std::fprintf(stderr, "[gemma4] mtp embed_token failed for tok=%d\n", cur_tok);
            ggml_gallocr_free(mtp_alloc);
            free_mtp_step_graph(mtp_g);
            return false;
        }
        // in_h_prev: captured by target graph into cache_.mtp_h_prev
        ggml_backend_tensor_copy(cache_.mtp_h_prev, mtp_g.in_h_prev);
        // in_pos: position of the draft token (= committed, 0-based)
        {
            int32_t p = committed;
            ggml_backend_tensor_set(mtp_g.in_pos, &p, 0, sizeof(int32_t));
        }

        // Fill the FA mask for TQ3_0 + head_dim>=512 cross-attention layers.
        // Real positions [0..kv_seq_len-1]: 0x0000 (F16 0.0 = admit).
        // Padding positions [kv_seq_len..mask_width-1]: 0xFC00 (F16 -inf = exclude).
        if (mtp_g.fa_mask && mtp_g.fa_mask->buffer) {
            const int64_t mask_n = mtp_g.fa_mask->ne[0];
            const int64_t kv_seq = mtp_g.fa_mask_kv_seq_len;
            std::vector<uint16_t> fa_mask_buf(mask_n);
            for (int64_t i = 0; i < mask_n; i++) {
                fa_mask_buf[i] = (i < kv_seq) ? 0x0000u : 0xFC00u;
            }
            ggml_backend_tensor_set(mtp_g.fa_mask, fa_mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_n);
        }

        {
            auto st = ggml_backend_graph_compute(backend_, mtp_g.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[gemma4] mtp compute failed\n");
                ggml_gallocr_free(mtp_alloc);
                free_mtp_step_graph(mtp_g);
                return false;
            }
        }

        // Read draft token from in-graph argmax.
        int32_t draft_tok = -1;
        ggml_backend_tensor_get(mtp_g.out_argmax, &draft_tok, 0, sizeof(int32_t));

        // Emit the current token (already committed by target step above).
        out_tokens.push_back(cur_tok);
        history.push_back(cur_tok);
        if (req.stream) {
            io.emit(cur_tok);
        }

        // ── 4+5. Check if draft matches target's token ───────────────────────
        if (draft_tok == target_next) {
            // MTP was right: accept draft token as next cur_tok.
            cur_tok = draft_tok;
        } else {
            // MTP was wrong: use target's token.
            cur_tok = target_next;
        }
        cache_.last_tok = cur_tok;

        // EOS check on the new cur_tok before next iteration.
        if (!args_.ignore_eos && is_gemma4_eos(target_w_, cur_tok)) {
            break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    out_decode_s = std::chrono::duration<double>(t1 - t0).count();

    ggml_gallocr_free(mtp_alloc);
    free_mtp_step_graph(mtp_g);

    if (req.stream) {
        io.emit(-1);
    }

    return true;
}

}  // namespace dflash27b
