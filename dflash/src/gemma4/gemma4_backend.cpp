// Gemma4Backend implementation. See gemma4_backend.h for the contract.
//
// This PR ships the AR-greedy path end-to-end and stubs the DFlash / MTP
// decode loops with TODOs pointing to where the source logic lives. The
// follow-up PR ports those loops verbatim from feature/gemma4-support's
// test_gemma4_dflash.cpp.

#include "gemma4_backend.h"
#include "gemma4_runtime_helpers.h"

#include "../common/sampler.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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

Gemma4Backend::Gemma4Backend(const Gemma4BackendArgs & args)
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
                "draft=%s pflash=%d\n",
                target_w_.n_layer, target_w_.n_embd, target_w_.n_vocab,
                args_.max_ctx, method, (int)args_.use_pflash);
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

// ── handle_compress / free_drafter (pflash compress lands with the runtime PR) ──

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
static void set_step_masks(StepGraph         & sg,
                           GemmaTargetCache  & cache,
                           int                 kv_start,
                           int                 n_tokens,
                           int                 swa_window) {
    if (sg.attn_mask && sg.attn_mask->buffer) {
        const int kv_len = kv_start + n_tokens;
        std::vector<uint16_t> mask_buf;
        build_causal_mask(mask_buf, kv_len, n_tokens, kv_start);
        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
    }
    if (sg.swa_mask && sg.swa_mask->buffer) {
        const SwaView swa_view = compute_swa_view(kv_start, n_tokens,
                                                   swa_window, cache.swa_ctx_alloc);
        std::vector<uint16_t> swa_buf;
        build_swa_causal_mask(swa_buf,
                              /*kv_start*/ kv_start,
                              /*n_tokens*/ n_tokens,
                              /*swa_window*/ swa_window,
                              /*ring_size*/ swa_view.effective_win_len,
                              /*kv_end*/ kv_start + n_tokens);
        ggml_backend_tensor_set(sg.swa_mask, swa_buf.data(), 0,
                                sizeof(uint16_t) * swa_buf.size());
    }
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

        if (!build_gemma4_step(sg, target_w_, cache_, backend_,
                               cs, chunk_n, need_mask,
                               /*capture=*/true,
                               args_.use_pflash, args_.pflash_alpha,
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

        // Note: do NOT step_graph_free(sg) here. build_gemma4_step calls
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
        if ((target_w_.eos_id      >= 0 && cur_tok == target_w_.eos_id) ||
            (target_w_.eos_chat_id >= 0 && cur_tok == target_w_.eos_chat_id)) {
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

        if (!build_gemma4_step(sg, target_w_, cache_, backend_,
                               committed, /*n_tokens=*/1,
                               /*with_mask=*/true,
                               /*capture=*/false,
                               /*use_pflash=*/false, args_.pflash_alpha,
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

        // Note: do NOT step_graph_free(sg) here. build_gemma4_step calls
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
    // draft_max_adaptive knob not in args_ — default disabled (flag: no args field)
    adaptive.init(/*on=*/false, args_.draft_max_block, draft_w_.block_size);

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
    std::vector<int32_t> target_tok(draft_w_.block_size);
    std::vector<float>   draft_logits_buf((size_t)vocab * draft_w_.block_size);
    std::vector<float>   verify_logits_buf((size_t)vocab * draft_w_.block_size);

    int committed         = cache_.cur_pos;
    int total_draft_steps = 0;

    auto t0 = std::chrono::steady_clock::now();

    while ((int)out_tokens.size() < n_gen) {
        int q_len = adaptive.enabled
                        ? adaptive.current
                        : ((args_.draft_max_block > 0 && args_.draft_max_block < draft_w_.block_size)
                               ? args_.draft_max_block : draft_w_.block_size);
        q_len = std::min(q_len, std::max(1, ctx_size - committed - 1));

        // EOS check (inlined IS_EOS_TOK)
        if ((target_w_.eos_id      >= 0 && cur_tok == target_w_.eos_id) ||
            (target_w_.eos_chat_id >= 0 && cur_tok == target_w_.eos_chat_id)) {
            break;
        }
        if (committed >= ctx_size - CTX_HEADROOM_DFLASH) {
            break;
        }

        // Warmup: not enough context for target_feat extraction — fall back to
        // single-token target-only decode until committed >= q_len.
        if (committed < q_len) {
            StepGraph sg{};
            if (!build_gemma4_step(sg, target_w_, cache_, backend_,
                                   committed, /*n_tokens=*/1,
                                   /*with_mask=*/true,
                                   /*capture=*/true,
                                   /*use_pflash=*/false, args_.pflash_alpha,
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

        // ── 6. Target verify: batched forward on draft_tok[0..q_len-1] ───────
        {
            StepGraph sg{};
            if (!build_gemma4_step(sg, target_w_, cache_, backend_,
                                   committed, q_len,
                                   /*with_mask=*/true, /*capture=*/true,
                                   /*use_pflash=*/false, args_.pflash_alpha,
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
            if ((target_w_.eos_id      >= 0 && draft_tok[i] == target_w_.eos_id) ||
                (target_w_.eos_chat_id >= 0 && draft_tok[i] == target_w_.eos_chat_id)) {
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
        adaptive.observe(accept_n, q_len, total_draft_steps);

        if (hit_eos) break;
    }

    auto t1 = std::chrono::steady_clock::now();
    out_decode_s = std::chrono::duration<double>(t1 - t0).count();

    if (req.stream) {
        io.emit(-1);
    }

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
        if ((target_w_.eos_id      >= 0 && cur_tok == target_w_.eos_id) ||
            (target_w_.eos_chat_id >= 0 && cur_tok == target_w_.eos_chat_id)) {
            break;
        }
        if (committed >= ctx_size - CTX_HEADROOM_MTP) {
            break;
        }

        // ── 1. Target forward for cur_tok (captures mtp_h_prev) ─────────────
        const int swa_window = target_w_.swa_window > 0 ? target_w_.swa_window : 1024;

        StepGraph sg{};
        if (!build_gemma4_step(sg, target_w_, cache_, backend_,
                               committed, /*n_tokens=*/1,
                               /*with_mask=*/true,
                               /*capture=*/false,
                               /*use_pflash=*/false, args_.pflash_alpha,
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
        if ((target_w_.eos_id      >= 0 && cur_tok == target_w_.eos_id) ||
            (target_w_.eos_chat_id >= 0 && cur_tok == target_w_.eos_chat_id)) {
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
