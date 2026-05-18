#include "qwen35_backend.h"
#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "dflash_feature_ring.h"
#include "dflash_capture.h"
#include "common/dflash_draft_graph.h"
#include "peer_access.h"
#include "attn_masks.h"
#include "common/sampler.h"
#include "common/io_utils.h"
#include "common/restore_delta.h"
#include "common/mtp_chain_runner.h"
#include "common/mtp_orchestrator.h"
#include "qwen3/qwen3_drafter.h"
#include "qwen36/qwen36_mtp.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash27b {

#define IS_EOS_TOK(tok, w)                                         \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id)                  \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

// ── Construction / destruction ──────────────────────────────────────────

Qwen35Backend::Qwen35Backend(const Qwen35Config & cfg) : cfg_(cfg) {}

Qwen35Backend::~Qwen35Backend() { shutdown(); }

// ── init() ──────────────────────────────────────────────────────────────

bool Qwen35Backend::init() {
    split_gpus_ = (cfg_.device.gpu != cfg_.draft_gpu);

    target_backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!target_backend_) {
        std::fprintf(stderr, "target cuda init failed\n");
        return false;
    }
    draft_backend_ = target_backend_;
    if (split_gpus_) {
        draft_backend_ = ggml_backend_cuda_init(cfg_.draft_gpu);
        if (!draft_backend_) {
            std::fprintf(stderr, "draft cuda init failed\n");
            return false;
        }
    }
    if (split_gpus_ && g_peer_access_opt_in) {
        enable_peer_access_pair(cfg_.device.gpu, cfg_.draft_gpu);
    }

    // Load target
    if (!load_target_gguf(cfg_.target_path, target_backend_, w_)) {
        std::fprintf(stderr, "target load: %s\n", dflash27b_last_error());
        return false;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    // Load draft (skipped in MTP mode — mtp_gguf_path replaces the draft)
    if (cfg_.draft_path) {
        std::string dp(cfg_.draft_path);
        bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
            ? load_draft_gguf(cfg_.draft_path, draft_backend_, dw_, &w_)
            : load_draft_safetensors(cfg_.draft_path, draft_backend_, dw_, &w_);
        if (!draft_ok) {
            std::fprintf(stderr, "draft load: %s\n", dflash27b_last_error());
            return false;
        }
        std::printf("[draft]  loaded\n");

        if (cfg_.draft_swa_window > 0) {
            dw_.swa_window = cfg_.draft_swa_window;
            for (int il = 0; il < dw_.n_layer - 1; il++)
                dw_.layers[il].is_swa = true;
            std::printf("[draft]  SWA layers: %d/%d (window=%d)\n",
                        dw_.n_layer - 1, dw_.n_layer, dw_.swa_window);
        }
    }

    // Create KV cache.
    // MTP mode: size for max(gamma+1, ddtree_budget+1) verify tokens so the
    // speculative verify batch fits even without a DFlash draft block size.
    // DFlash mode: existing logic uses dw_.block_size.
    int max_verify_tokens = 0;
    if (cfg_.mtp_gguf_path) {
        const int mtp_gamma_eff = std::max(1, cfg_.mtp_gamma);
        const int budget_eff    = cfg_.ddtree_mode ? cfg_.ddtree_budget : 0;
        max_verify_tokens = std::max(mtp_gamma_eff + 1, budget_eff + 1);
        // Ensure at least DFLASH27B_DRAFT_BLOCK_SIZE so internal buffers
        // allocated by create_target_cache are sized conservatively.
        max_verify_tokens = std::max(max_verify_tokens, DFLASH27B_DRAFT_BLOCK_SIZE);
    } else {
        max_verify_tokens = cfg_.ddtree_mode
            ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
            : dw_.block_size;
    }
    if (!create_target_cache(w_, cfg_.device.max_ctx, max_verify_tokens, target_backend_, cache_,
                             /*prefill_only=*/true)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return false;
    }

    // Init feature mirror when draft model is available (needed for spec decode).
    // On single-GPU, this is an F32 conversion buffer; on split-GPU, a cross-device mirror.
    if (cfg_.draft_path) {
        const int mirror_cap = std::min({cfg_.draft_ctx_max, cfg_.device.max_ctx,
                                         cache_.target_feat_cap > 0 ? cache_.target_feat_cap : cfg_.device.max_ctx});
        if (!draft_feature_mirror_init(feature_mirror_, draft_backend_,
                                       cfg_.draft_gpu, cfg_.device.gpu, mirror_cap,
                                       DFLASH27B_DRAFT_N_TARGET_LAYERS,
                                       DFLASH27B_TARGET_HIDDEN)) {
            std::fprintf(stderr, "warning: feature mirror init failed, spec decode will use AR fallback\n");
        }
    }

    // Init MTP speculator when configured.
    if (cfg_.mtp_gguf_path) {
        if (!init_mtp_()) {
            return false;
        }
    }

    return true;
}

// ── print_ready_banner ──────────────────────────────────────────────────

void Qwen35Backend::print_ready_banner() const {
    std::printf("[daemon] ready\n");
    std::fflush(stdout);
}

// ── Park / unpark ───────────────────────────────────────────────────────

bool Qwen35Backend::park(const std::string & what) {
    bool want_draft  = (what.empty() || what == "all" || what == "draft");
    bool want_target = (what.empty() || what == "all" || what == "target");

    if (want_draft && !draft_parked_) {
        step_graph_destroy(draft_sg_);
        free_draft_weights(dw_);
        draft_parked_ = true;
        std::printf("[park] draft released\n"); std::fflush(stdout);
    }
    if (want_target && !target_parked_) {
        step_graph_destroy(proj_sg_);
        free_target_weights(w_);
        target_parked_ = true;
        std::printf("[park] target released\n"); std::fflush(stdout);
    }
    return true;
}

bool Qwen35Backend::unpark(const std::string & what) {
    bool want_target = (what.empty() || what == "all" || what == "target");
    bool want_draft  = (what.empty() || what == "all" || what == "draft");

    if (want_target && target_parked_) {
        if (!load_target_gguf(cfg_.target_path, target_backend_, w_)) {
            std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
            return false;
        }
        target_parked_ = false;
        std::printf("[unpark] target restored\n"); std::fflush(stdout);
    }
    if (want_draft && draft_parked_ && cfg_.draft_path) {
        std::string dp(cfg_.draft_path);
        bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
            ? load_draft_gguf(cfg_.draft_path, draft_backend_, dw_, &w_)
            : load_draft_safetensors(cfg_.draft_path, draft_backend_, dw_, &w_);
        if (!draft_ok) {
            std::fprintf(stderr, "[unpark] draft: %s\n", dflash27b_last_error());
            return false;
        }
        if (cfg_.draft_swa_window > 0) {
            dw_.swa_window = cfg_.draft_swa_window;
            for (int il = 0; il < dw_.n_layer - 1; il++)
                dw_.layers[il].is_swa = true;
        }
        draft_parked_ = false;
        std::printf("[unpark] draft restored\n"); std::fflush(stdout);
    }
    return true;
}

// ── Snapshots ───────────────────────────────────────────────────────────

bool Qwen35Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    snapshot_free(slot);
    PrefixSnapshot & snap = prefix_snapshots_[slot];
    return snapshot_target_cache(w_, cache_, target_backend_, snap);
}

void Qwen35Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_prefix_snapshot(prefix_snapshots_[slot]);
}

bool Qwen35Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return prefix_snapshots_[slot].ctx != nullptr;
}

int Qwen35Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return prefix_snapshots_[slot].cur_pos;
}

// ── Compress (pflash) ───────────────────────────────────────────────────

bool Qwen35Backend::handle_compress(const std::string & line, const DaemonIO & io) {
    // Check for "nopark" as the last space-delimited token (not substring)
    // to avoid false matches against file paths containing that text.
    bool skip_park = false;
    {
        auto pos = line.rfind(' ');
        if (pos != std::string::npos && line.substr(pos + 1) == "nopark")
            skip_park = true;
    }

    // Parse: "compress <path> <keep_x1000> <drafter_gguf> [nopark]"
    char ppath[1024];
    int  keep_x1000 = 0;
    char drafter_path[1024] = {0};
    const int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s",
                               ppath, &keep_x1000, drafter_path);
    if (n < 2) {
        std::fprintf(stderr, "[compress] bad args\n");
        io.emit(-1);
        return false;
    }

    const char * dpath = (n >= 3 && drafter_path[0])
        ? drafter_path
        : "/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf";

    // Park target+draft to free VRAM for the drafter (unless skip_park).
    // Also destroy the main target step graph allocator to release its CUDA buffer.
    const bool was_target_parked = target_parked_;
    const bool was_draft_parked  = draft_parked_;
    if (!skip_park) {
        step_graph_destroy(sg_);
        if (!target_parked_) park("target");
        if (!draft_parked_)  park("draft");
    }

    // Synchronize all backends to flush any outstanding async CUDA work
    // before loading the drafter. Without this, pending operations on
    // target/draft streams can corrupt the drafter's allocations.
    ggml_backend_synchronize(target_backend_);
    if (draft_backend_) ggml_backend_synchronize(draft_backend_);

    // Load drafter with its OWN backend (not target_backend_).
    // Matches test_dflash.cpp: separate backend supports multi-GPU
    // (drafter on GPU A, target on GPU B or CPU).
    // The drafter stays loaded across compress calls — only the first call
    // creates the backend + loads weights; subsequent calls reuse them.
    if (!drafter_loaded_) {
        // drafter_ctx_.backend == nullptr → load_drafter creates its own
        std::fprintf(stderr, "[compress] loading drafter from %s ...\n", dpath);
        if (!load_drafter(dpath, /*gpu_layers=*/999, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] drafter init failed: %s\n",
                         dflash27b_last_error());
            io.emit(-1);
            if (!skip_park) {
                if (!was_target_parked) unpark("target");
                if (!was_draft_parked)  unpark("draft");
            }
            return false;
        }
        drafter_loaded_ = true;
        std::fprintf(stderr, "[compress] drafter ready\n");
    }

    std::vector<int32_t> tokens = read_int32_file(ppath);
    bool ok = false;
    if (!tokens.empty()) {
        const float keep = (float)keep_x1000 / 1000.0f;
        auto compressed = drafter_score_and_compress(drafter_ctx_, tokens, keep);
        ok = !compressed.empty();
        if (ok) {
            std::fprintf(stderr, "[compress] %zu -> %zu tokens\n",
                          tokens.size(), compressed.size());
            for (int32_t t : compressed) io.emit(t);
        }
    }
    io.emit(-1);

    // Keep drafter loaded (own backend + weights persist), matching test_dflash.
    // ~1.4 GB stays resident but avoids reload cost on subsequent compresses.

    // Restore park state
    if (!skip_park) {
        if (!was_target_parked) unpark("target");
        if (!was_draft_parked)  unpark("draft");
    }

    return ok;
}

void Qwen35Backend::free_drafter() {
    if (drafter_loaded_) {
        // Drafter has its own backend — do a full free (weights + backend)
        dflash27b::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
        std::printf("[drafter] freed\n"); std::fflush(stdout);
    }
}

// ── try_handle_command (arch-specific) ──────────────────────────────────

bool Qwen35Backend::try_handle_command(const std::string & line, const DaemonIO & io) {
    // SNAPSHOT_THIN <slot> — lightweight snapshot (SSM state only, no KV copy)
    if (line.compare(0, 14, "SNAPSHOT_THIN ") == 0) {
        int slot = std::atoi(line.c_str() + 14);
        if (slot >= 0 && slot < PREFIX_SLOTS) {
            snapshot_free(slot);
            PrefixSnapshot & snap = prefix_snapshots_[slot];
            snapshot_target_cache_thin(w_, cache_, target_backend_,
                                       /*kv_start=*/0, /*kv_end=*/cache_.cur_pos, snap);
            std::printf("[snapshot_thin] slot=%d pos=%d\n", slot, snap.cur_pos);
            std::fflush(stdout);
        }
        io.emit(-1);
        return true;
    }

    return false;
}

// ── DFlash spec decode target ────────────────────────────────────────────

DFlashTarget * Qwen35Backend::dflash_target() {
    if (!dflash_target_) {
        dflash_target_ = std::make_unique<Qwen35DFlashTarget>(
            w_, cache_, target_backend_, sg_,
            cfg_.kq_stride_pad, cfg_.fa_window);
    }
    return dflash_target_.get();
}

// ── Test/bench integration hooks ────────────────────────────────────────

bool Qwen35Backend::ensure_decode_cache(int max_verify_tokens) {
    return migrate_prefill_cache(w_, cfg_.device.max_ctx,
                                 max_verify_tokens,
                                 target_backend_, cache_);
}

ggml_context * Qwen35Backend::tensor_context() const {
    return w_.ctx;
}

// ── Shutdown ────────────────────────────────────────────────────────────

void Qwen35Backend::shutdown() {
    free_drafter();
    step_graph_destroy(sg_);
    step_graph_destroy(draft_sg_);
    step_graph_destroy(proj_sg_);
    draft_feature_mirror_free(feature_mirror_);
    for (int i = 0; i < PREFIX_SLOTS; i++) {
        free_prefix_snapshot(prefix_snapshots_[i]);
    }
    if (!target_parked_) free_target_weights(w_);
    if (!draft_parked_)  free_draft_weights(dw_);
    free_target_cache(cache_);
    if (split_gpus_ && draft_backend_) {
        ggml_backend_free(draft_backend_);
        draft_backend_ = nullptr;
    }
    if (target_backend_) {
        ggml_backend_free(target_backend_);
        target_backend_ = nullptr;
    }
}

// ── Generate (speculative decode) ───────────────────────────────────────

GenerateResult Qwen35Backend::generate(const GenerateRequest & req,
                                        const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    // Zero delta-net recurrent state (SSM + conv) so a fresh prompt doesn't
    // inherit stale hidden state from the previous request. KV cache is
    // position-addressed and will be overwritten during prefill.
    reset_recurrent_state(cache_);

    // MTP path: delegate to common orchestrator. Cache was already sized in
    // init_mtp_() — no per-request migrate (idempotent no-op).
    if (supports_mtp()) {
        return common::mtp::warm_and_decode(this, req, io);
    }

    // DFlash / AR path
    auto t_prefill_start = std::chrono::steady_clock::now();
    int committed = do_prefill(req.prompt, out_io, req.snap_pos, req.snap_slot);
    if (committed < 0) { result.error = "prefill"; return result; }
    result.prefill_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_prefill_start).count();

    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();
        if (!do_spec_decode(committed, req.n_gen, result.tokens, out_io)) {
            result.error = "decode";
            return result;
        }
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }
    result.ok = true;
    return result;
}

// ── Restore + generate ──────────────────────────────────────────────────

GenerateResult Qwen35Backend::restore_and_generate(int slot,
                                                    const GenerateRequest & req,
                                                    const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (slot < 0 || slot >= PREFIX_SLOTS || !prefix_snapshots_[slot].ctx) {
        result.error = "bad slot";
        out_io.emit(-1);
        return result;
    }

    // Restore snapshot
    restore_target_cache(prefix_snapshots_[slot], cache_);

    // Now generate from restored state
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    const int snap_pos = prefix_snapshots_[slot].cur_pos;
    cache_.cur_pos = snap_pos;

    // Daemon receives the FULL prompt; slice off the cached prefix and prefill
    // only the delta at KV positions [snap_pos, snap_pos + delta.size()).
    int committed = snap_pos;
    const int prompt_len = (int)req.prompt.size();
    std::vector<float> all_prefill_hidden;

    if (prompt_len > snap_pos) {
        auto t_prefill_start = std::chrono::steady_clock::now();
        std::vector<int32_t> delta = restore_prompt_delta(req.prompt, snap_pos);

        if (supports_mtp()) {
            // MTP path: prefill delta via DFlashTarget::verify_batch so hidden
            // states are captured for warm_mtp_for_prompt_.
            committed = do_mtp_prefill_(delta, all_prefill_hidden, /*kv_offset=*/snap_pos);
            if (committed < 0) {
                result.error = "mtp_prefill";
                return result;
            }
            if (!warm_mtp_for_prompt_(req.prompt, all_prefill_hidden, cache_.last_tok)) {
                result.error = "mtp_warm";
                return result;
            }
        } else {
            committed = do_prefill(delta, out_io, req.snap_pos, req.snap_slot, /*kv_offset=*/snap_pos);
            if (committed < 0) {
                result.error = "prefill";
                return result;
            }
        }
        result.prefill_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    } else if (prompt_len > 0 && prompt_len < snap_pos) {
        // Cached more than the request — should never happen in practice.
        result.error = "snapshot_longer_than_prompt";
        out_io.emit(-1);
        return result;
    }

    // Decode
    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();
        bool ok = supports_mtp()
            ? do_mtp_decode_(committed, req.n_gen, result.tokens, io)
            : do_spec_decode(committed, req.n_gen, result.tokens, out_io);
        if (!ok) {
            result.error = "decode";
            return result;
        }
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }

    result.ok = true;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL HELPERS — will be fleshed out when the spec-decode loop is
// migrated from test_dflash.cpp. For now, these are stubs that produce an
// error so the build succeeds and the interface is validated.
// ═══════════════════════════════════════════════════════════════════════════

int Qwen35Backend::do_prefill(const std::vector<int32_t> & tokens,
                               const DaemonIO & io,
                               int snap_pos, int snap_slot,
                               int kv_offset) {
    (void)io;

    const int hidden = w_.n_embd;
    int prefill_ubatch = 512;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        prefill_ubatch = std::max(1, std::atoi(s));
    }
    const int prompt_len = (int)tokens.size();

    // Skip KV-cache migration when resuming from a snapshot — the cache was
    // already migrated when the snapshot was taken; re-running migrate would
    // clobber the restored state.
    if (kv_offset == 0) {
        migrate_prefill_cache(w_, cfg_.device.max_ctx,
                              cfg_.ddtree_mode
                                  ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
                                  : dw_.block_size,
                              target_backend_, cache_);
    }

    // Chunked prefill
    std::vector<float> embed_buf((size_t)hidden * prefill_ubatch);
    int committed = kv_offset;
    for (int start = 0; start < prompt_len;) {
        const int kv_pos = kv_offset + start;
        if (snap_pos >= 0 && snap_slot >= 0 && snap_pos == kv_pos) {
            cache_.cur_pos = kv_pos;
            if (snapshot_save(snap_slot)) {
                std::printf("[snap] inline slot=%d cur_pos=%d\n", snap_slot, kv_pos);
                std::fflush(stdout);
            }
            snap_pos = -1;
            snap_slot = -1;
        }

        int n_tokens = std::min(prefill_ubatch, prompt_len - start);
        if (snap_pos > kv_pos && snap_pos < kv_pos + n_tokens) {
            n_tokens = snap_pos - kv_pos;
        }
        const bool with_mask = (cfg_.kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/kv_pos, /*n_tokens=*/n_tokens,
                               with_mask, /*capture=*/true,
                               /*capture_delta_intermediate=*/false,
                               cfg_.fa_window,
                               /*last_token_logits_only=*/(start + n_tokens < prompt_len),
                               cfg_.kq_stride_pad)) {
            std::fprintf(stderr, "prefill build @%d\n", kv_pos);
            return -1;
        }

        // Embed
        if (!w_.embedder.embed(tokens.data() + start, n_tokens, embed_buf.data())) {
            return -1;
        }
        ggml_backend_tensor_set(sg_.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * (size_t)hidden * n_tokens);

        // Positions (M-RoPE)
        std::vector<int32_t> pos_buf((size_t)4 * n_tokens, 0);
        for (int i = 0; i < n_tokens; i++) {
            const int p = kv_pos + i;
            pos_buf[4 * i + 0] = p;
            pos_buf[4 * i + 1] = p;
            pos_buf[4 * i + 2] = p;
            pos_buf[4 * i + 3] = 0;
        }
        ggml_backend_tensor_set(sg_.positions, pos_buf.data(), 0,
                                sizeof(int32_t) * pos_buf.size());

        // Mask
        if (sg_.attn_mask) {
            const int win_start = (cfg_.fa_window > 0 && kv_pos > cfg_.fa_window)
                                      ? (kv_pos - cfg_.fa_window) : 0;
            const int kv_len = kv_pos + n_tokens - win_start;
            std::vector<uint16_t> mask_buf;
            build_causal_mask(mask_buf, kv_len, n_tokens, kv_pos, cfg_.kq_stride_pad, win_start);
            ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        // Compute
        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "prefill compute @%d failed\n", kv_pos);
            return -1;
        }

        int32_t last_tok = -1;
        const size_t argmax_off =
            (start + n_tokens < prompt_len) ? 0 : sizeof(int32_t) * (size_t)(n_tokens - 1);
        ggml_backend_tensor_get(sg_.argmax_tokens, &last_tok, argmax_off, sizeof(int32_t));
        cache_.last_tok = last_tok;

        committed = kv_pos + n_tokens;
        cache_.cur_pos = committed;

        if (snap_pos >= 0 && snap_slot >= 0 && committed == snap_pos) {
            if (snapshot_save(snap_slot)) {
                std::printf("[snap] inline slot=%d cur_pos=%d\n", snap_slot, committed);
                std::fflush(stdout);
            }
            snap_pos = -1;
            snap_slot = -1;
        }

        // Sync feature mirror if active
        if (feature_mirror_.target_feat && !draft_parked_) {
            draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, kv_pos, n_tokens);
        }

        start += n_tokens;
    }

    return committed;
}

// ── AR decode fallback (no draft model) ─────────────────────────────────

bool Qwen35Backend::do_ar_decode(int committed, int n_gen,
                                  std::vector<int32_t> & out_tokens,
                                  const DaemonIO & io) {
    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> logits_buf(vocab);
    std::vector<float> embed_buf_vec(hidden);
    float * embed_buf = embed_buf_vec.data();

    for (int i = 0; i < n_gen; i++) {
        int32_t tok = out_tokens.back();

        if (!w_.embedder.embed(&tok, 1, embed_buf)) return false;
        ggml_backend_tensor_set(sg_.inp_embed, embed_buf, 0, sizeof(float) * hidden);
        int32_t pos4[4] = {committed, committed, committed, 0};
        ggml_backend_tensor_set(sg_.positions, pos4, 0, sizeof(int32_t) * 4);

        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/committed, /*n_tokens=*/1,
                               /*with_mask=*/false, /*capture=*/true,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/false,
                               cfg_.kq_stride_pad)) {
            return false;
        }

        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) return false;

        ggml_backend_tensor_get(sg_.logits, logits_buf.data(), 0,
                                sizeof(float) * vocab);
        int32_t next_tok;
        if (sampler_.temp > 0) {
            next_tok = sample_logits(logits_buf.data(), vocab, sampler_,
                                      out_tokens, sampler_rng_);
        } else {
            next_tok = 0;
            float best = logits_buf[0];
            for (int j = 1; j < vocab; j++) {
                if (logits_buf[j] > best) { best = logits_buf[j]; next_tok = j; }
            }
        }

        out_tokens.push_back(next_tok);
        io.emit(next_tok);
        committed++;
        cache_.cur_pos = committed;
        if (io.cancelled) break;

        if (IS_EOS_TOK(next_tok, w_)) break;
    }
    return true;
}

// ── DFlash speculative decode loop ─────────────────────────────────────

bool Qwen35Backend::do_spec_decode(int committed, int n_gen,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io) {
    const int hidden = w_.n_embd;

    // Sample first token from prefill's last-position argmax.
    // The last chunk used last_token_logits_only=false, so sg_.argmax_tokens
    // holds argmax for ALL positions in that chunk. We need the LAST position.
    int32_t last_tok;
    {
        const int PREFILL_UBATCH = 512;
        int n_last_chunk = committed % PREFILL_UBATCH;
        if (n_last_chunk == 0) n_last_chunk = PREFILL_UBATCH;
        ggml_backend_tensor_get(sg_.argmax_tokens, &last_tok,
                                sizeof(int32_t) * (n_last_chunk - 1),
                                sizeof(int32_t));
    }

    // Check if we can use speculative decode:
    // - draft model loaded and not parked
    // - feature mirror initialized
    // - greedy decoding (temp == 0) — spec decode uses argmax verification
    const bool can_spec = cfg_.draft_path
        && !draft_parked_
        && feature_mirror_.target_feat
        && sampler_.temp == 0.0f;

    if (!can_spec) {
        // AR fallback: emit first token, then loop.
        out_tokens.push_back(last_tok);
        io.emit(last_tok);
        committed++;
        cache_.cur_pos = committed;
        if (io.cancelled || IS_EOS_TOK(last_tok, w_)) { io.emit(-1); return true; }
        bool ok = do_ar_decode(committed, n_gen - 1, out_tokens, io);
        io.emit(-1);
        return ok;
    }

    // ── DFlash spec-decode: draft → verify → accept → replay ──────────

    DFlashTarget * target = dflash_target();
    const int q_len = dw_.block_size;

    StepGraph draft_sg;

    std::vector<float>   noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<float>   local_hidden;

    int n_generated     = 0;
    int n_draft_steps   = 0;
    int n_accept_sum    = 0;

    auto t_dec0 = std::chrono::steady_clock::now();

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        // 1. Build noise input for draft
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = target->mask_token_id();
        if (!target->embed_tokens(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "spec-decode: noise embed failed (last_tok=%d mask=%d q_len=%d)\n",
                         last_tok, target->mask_token_id(), q_len);
            step_graph_destroy(draft_sg);
            return false;
        }

        // 2. Draft compute
        constexpr int DRAFT_CTX_MAX_DEFAULT = 2048;
        const int ring_cap = feature_mirror_.cap;
        const int draft_ctx = std::min(committed,
            std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_mirror_, committed, draft_ctx, mirror_slot0);

        if (!build_draft_step(draft_sg, dw_, /*lm_head=*/nullptr, draft_backend_,
                              draft_ctx, use_mirror_view ? &feature_mirror_ : nullptr,
                              committed)) {
            std::fprintf(stderr, "spec-decode: draft build failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_mirror_, draft_sg.target_hidden_cat,
                                               draft_start, draft_ctx)) {
            std::fprintf(stderr, "spec-decode: feature copy failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());
        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * pos_k.size());

        auto st = ggml_backend_graph_compute(draft_backend_, draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "spec-decode: draft compute failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        // Read draft hidden states to host for LM-head projection.
        local_hidden.resize((size_t)hidden * q_len);
        ggml_backend_tensor_get(draft_sg.hidden_states, local_hidden.data(), 0,
                                sizeof(float) * local_hidden.size());

        // 3. Project draft hidden → token IDs via target LM head
        if (!target->project_hidden_to_tokens(local_hidden.data(), q_len, draft_tok)) {
            std::fprintf(stderr, "spec-decode: projection failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        draft_tok[0] = last_tok;

        // 4. Verify: snapshot KV, run target forward over draft tokens
        if (!target->snapshot_kv()) {
            step_graph_destroy(draft_sg);
            return false;
        }

        int verify_last_tok = -1;
        if (!target->verify_batch(draft_tok, committed, verify_last_tok, &target_tok)) {
            std::fprintf(stderr, "spec-decode: verify failed\n");
            target->restore_kv();
            step_graph_destroy(draft_sg);
            return false;
        }

        // 5. Acceptance: longest matching prefix between draft and target argmax
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n  = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        // 6. Replay: roll back KV and re-run only accepted tokens
        if (!target->restore_kv()) {
            step_graph_destroy(draft_sg);
            return false;
        }

        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }
        int replay_last_tok = -1;
        if (!target->verify_batch(replay_tok, committed, replay_last_tok, nullptr)) {
            std::fprintf(stderr, "spec-decode: replay failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        last_tok = replay_last_tok;

        // 7. Sync features for replayed range to mirror (needed for next draft step)
        if (feature_mirror_.target_feat && cache_.target_feat) {
            draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, committed, commit_n);
        }

        // 8. Emit committed tokens (stop at EOS)
        bool hit_eos = false;
        int emitted = 0;
        for (int i = 0; i < commit_n; i++) {
            out_tokens.push_back(replay_tok[i]);
            io.emit(replay_tok[i]);
            emitted++;
            if (io.cancelled) break;
            if (IS_EOS_TOK(replay_tok[i], w_)) { hit_eos = true; break; }
        }
        committed   += emitted;
        cache_.cur_pos = committed;
        n_generated += emitted;
        n_accept_sum += std::min(accept_n, emitted);
        n_draft_steps++;
        if (io.cancelled) break;
        if (hit_eos) break;
    }

    step_graph_destroy(draft_sg);

    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_draft_steps * q_len);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    std::fprintf(stderr, "[spec-decode] tokens=%d time=%.3f s speed=%.2f tok/s "
                 "steps=%d accepted=%d/%d (%.1f%%) avg_commit=%.2f\n",
                 n_generated, decode_s,
                 n_generated > 0 ? n_generated / decode_s : 0.0,
                 n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                 n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);

    io.emit(-1);
    return true;
}

int Qwen35Backend::verify_chain(int committed, const int32_t * draft_tok, int q_len) {
    (void)committed; (void)draft_tok; (void)q_len;
    return 0;
}

int Qwen35Backend::verify_tree(int committed, const DDTree & tree) {
    (void)committed; (void)tree;
    return 0;
}

// ── MTP init helper ─────────────────────────────────────────────────────────
//
// Mirrors run_qwen36_mtp_harness lines 728-746: construct Qwen36MtpModule,
// load weights from the fused GGUF sharing the backbone tensor context,
// then attach to the DFlashTarget adapter.
//
// Called from init() when cfg_.mtp_gguf_path != nullptr.

bool Qwen35Backend::init_mtp_() {
    // Ensure the DFlashTarget adapter exists (lazy-built); the MTP module
    // needs it for attach() and for reading last_hidden() during decode.
    DFlashTarget * target = dflash_target();
    if (!target) {
        std::fprintf(stderr, "[mtp] dflash_target() unavailable\n");
        return false;
    }

    mtp_module_ = std::make_unique<mtp::Qwen36MtpModule>();
    std::string err;
    // Use the two-arg init that shares the backbone ggml_context so MTP
    // tensors are mapped from the same memory arena as the target weights.
    if (!mtp_module_->init(cfg_.mtp_gguf_path, tensor_context(), target, err)) {
        std::fprintf(stderr, "[mtp] init failed: %s\n", err.c_str());
        mtp_module_.reset();
        return false;
    }

    if (!mtp_module_->attach(target)) {
        std::fprintf(stderr, "[mtp] attach(target) failed\n");
        mtp_module_.reset();
        return false;
    }

    // Single source of truth for γ: bind to module at attach time. Orchestrator
    // + runner read module->effective_gamma() — no parallel storage. This is
    // why the regression in PR #214's earlier orchestrator (which silently used
    // module->max_gamma()) cannot happen by construction once this is in place.
    mtp_module_->set_effective_gamma(cfg_.mtp_gamma);

    // Pre-size the rollback cache once for the MTP gamma chosen at attach.
    // Per momus audit of cubic#3257248868: hoisting out of generate() makes
    // the OOM-on-first-request → nullptr ssm_intermediate → segfault path
    // unreachable (max_ctx + γ are config-time constants; check return here
    // where we can fail backend init cleanly).
    const int gamma_eff = (cfg_.mtp_gamma > 0) ? cfg_.mtp_gamma : 3;
    if (!migrate_prefill_cache(w_, cfg_.device.max_ctx,
                               std::max(gamma_eff + 1, DFLASH27B_DRAFT_BLOCK_SIZE),
                               target_backend_, cache_)) {
        std::fprintf(stderr, "[mtp] migrate_prefill_cache failed (max_ctx=%d gamma=%d)\n",
                     cfg_.device.max_ctx, gamma_eff);
        mtp_module_.reset();
        return false;
    }

    if (cfg_.mtp_draft_source && std::strcmp(cfg_.mtp_draft_source, "mtp_topk") == 0) {
        mtp_module_->set_draft_topk(std::max(1, cfg_.mtp_draft_topk));
    }

    std::printf("[mtp] loaded gamma=%d source=%s\n",
                cfg_.mtp_gamma,
                cfg_.mtp_draft_source ? cfg_.mtp_draft_source : "chain");
    std::fflush(stdout);
    return true;
}

// ── MTP warm helper ──────────────────────────────────────────────────────────
//
// Mirrors run_qwen36_mtp_harness lines 783-792: seed the MTP module with the
// backbone's final post-norm hidden after prefill, then warm the head KV cache
// over all prefill positions.
//
// prompt             : full prompt token sequence (length N)
// all_prefill_hidden : backbone post-norm hiddens [N * hidden_size], F32
// prefill_next       : backbone argmax at end of prefill (t_{N})
//
// Returns false on failure; the caller (Phase 4's do_prefill hook or generate)
// should treat this as an error.

bool Qwen35Backend::warm_mtp_for_prompt_(const std::vector<int32_t> & prompt,
                                          const std::vector<float>    & all_prefill_hidden,
                                          int32_t                       prefill_next) {
    if (!mtp_module_) return true;  // MTP not configured — no-op

    DFlashTarget * target = dflash_target();

    // Seed with the backbone's final hidden for the last prefill token.
    if (target && target->last_hidden()) {
        mtp_module_->set_initial_hidden(target->last_hidden(), target->hidden_size());
    }

    // Warm the head KV cache if we have the full-sequence hiddens.
    if (!all_prefill_hidden.empty() && prefill_next >= 0) {
        if (!mtp_module_->warm_head_kv(prompt.data(),
                                        static_cast<int>(prompt.size()),
                                        prefill_next,
                                        all_prefill_hidden.data())) {
            std::fprintf(stderr, "[mtp] warm_head_kv failed\n");
            return false;
        }
    }

    return true;
}

// ── MTP prefill helper ───────────────────────────────────────────────────────
//
// Routes chunked prefill through DFlashTarget::verify_batch instead of the
// raw build_target_step path in do_prefill.  This populates last_hidden_seq
// on the Qwen35DFlashTarget so warm_mtp_for_prompt_ has per-position hidden
// states for warm_head_kv.  kv_offset > 0 means we are resuming after a
// snapshot restore (same semantics as do_prefill's kv_offset).
//
// Returns the committed KV position (== kv_offset + tokens.size()) or -1
// on failure.

int Qwen35Backend::do_mtp_prefill_(const std::vector<int32_t> & tokens,
                                    std::vector<float>          & all_prefill_hidden_out,
                                    int                           kv_offset) {
    const int prompt_len = (int)tokens.size();
    if (prompt_len == 0) return kv_offset;

    DFlashTarget * target = dflash_target();
    if (!target) {
        std::fprintf(stderr, "[mtp_prefill] dflash_target unavailable\n");
        return -1;
    }

    // Cast to the concrete type to enable hidden-sequence capture.  Safe
    // because Qwen35Backend always constructs a Qwen35DFlashTarget via
    // dflash_target().
    auto * q35target = static_cast<Qwen35DFlashTarget *>(target);

    // Ensure the rollback cache exists (needed for verify_batch's chain-capture
    // plumbing and for do_mtp_decode_'s snapshot_kv/restore_kv calls).
    const int gamma_eff = (cfg_.mtp_gamma > 0) ? cfg_.mtp_gamma : 3;
    migrate_prefill_cache(w_, cfg_.device.max_ctx,
                          std::max(gamma_eff + 1, DFLASH27B_DRAFT_BLOCK_SIZE),
                          target_backend_, cache_);

    // Enable full-sequence hidden capture so last_hidden_seq is populated
    // after each verify_batch chunk.  Switch back to LAST_ROW_ONLY after
    // prefill so decode-side chain verifies only pay one-row download cost.
    q35target->enable_hidden_seq_capture(true);
    q35target->set_hidden_capture_mode(Qwen35DFlashTarget::VerifyCaptureMode::FULL_SEQ);

    const int hidden = target->hidden_size();
    all_prefill_hidden_out.resize((size_t)prompt_len * hidden);

    int prefill_ubatch = 512;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        prefill_ubatch = std::max(1, std::atoi(s));
    }

    int committed = kv_offset;
    int32_t last_tok = -1;

    for (int start = 0; start < prompt_len;) {
        const int n = std::min(prefill_ubatch, prompt_len - start);
        const int kv_pos = kv_offset + start;

        std::vector<int32_t> chunk(tokens.begin() + start,
                                   tokens.begin() + start + n);

        // verify_batch handles: embed, positions, mask, compute, argmax,
        // last_hidden, last_hidden_seq.  It also advances cache_.cur_pos.
        if (!target->verify_batch(chunk, kv_pos, last_tok, nullptr)) {
            std::fprintf(stderr, "[mtp_prefill] verify_batch failed at kv_pos=%d\n", kv_pos);
            q35target->enable_hidden_seq_capture(false);
            return -1;
        }

        // Collect per-chunk hidden states into all_prefill_hidden_out.
        int n_chunk = 0;
        const float * h_seq = target->last_hidden_seq(&n_chunk);
        if (h_seq && n_chunk == n) {
            std::memcpy(all_prefill_hidden_out.data() + (size_t)start * hidden,
                        h_seq,
                        sizeof(float) * (size_t)n * hidden);
        } else {
            std::fprintf(stderr,
                "[mtp_prefill] hidden seq missing: expected %d tokens, got %d\n",
                n, n_chunk);
            // Non-fatal — warm_head_kv will be skipped with an empty hidden buffer.
            all_prefill_hidden_out.clear();
        }

        committed = kv_pos + n;
        start    += n;
    }

    // Record last token for do_mtp_decode_ (mirrors do_prefill's cache_.last_tok).
    cache_.last_tok = last_tok;

    // Switch to LAST_ROW_ONLY for decode phase — subsequent chain-runner
    // verify_batch calls only need hidden_at_pos(base_pos - 1).
    q35target->set_hidden_capture_mode(Qwen35DFlashTarget::VerifyCaptureMode::LAST_ROW_ONLY);

    return committed;
}

// ── MTP decode helper ────────────────────────────────────────────────────────
//
// Drives MtpChainRunner after prefill.  Mirrors harness lines 826-841 (chain
// path only — mtp_topk is Phase 4 out-of-scope).
//
// Emits tokens via io.emit per token (req.stream=true inside the runner), and
// a terminal io.emit(-1) is issued by the runner.  The caller must NOT emit
// additional -1 tokens.
//
// Returns false on failure; caller sets result.error = "decode".

bool Qwen35Backend::do_mtp_decode_(int committed, int n_gen,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io) {
    if (!mtp_module_) return false;
    if (n_gen <= 0) {
        io.emit(-1);
        return true;
    }

    // The argmax produced at the end of prefill — this is the first generated
    // token (matches harness line 806: generated.push_back(prefill_next)).
    const int32_t prefill_next = cache_.last_tok;
    if (prefill_next < 0) {
        std::fprintf(stderr, "[mtp_decode] prefill_next invalid (%d)\n", prefill_next);
        return false;
    }

    // Emit the prefill token immediately.
    out_tokens.push_back(prefill_next);
    io.emit(prefill_next);
    if (IS_EOS_TOK(prefill_next, w_)) {
        io.emit(-1);
        return true;
    }
    if (n_gen == 1) {
        io.emit(-1);
        return true;
    }

    DFlashTarget * target = dflash_target();
    if (!target) {
        std::fprintf(stderr, "[mtp_decode] dflash_target unavailable\n");
        return false;
    }

    const int gamma = (cfg_.mtp_gamma > 0) ? cfg_.mtp_gamma : 3;

    // MtpChainRunner with stream=true: the runner calls io.emit() per accepted
    // token and issues io.emit(-1) on completion.  No double-emit needed here.
    GenerateRequest req_inner;
    req_inner.n_gen     = n_gen - 1;   // prefill_next already consumed
    req_inner.stream    = true;
    req_inner.do_sample = false;
    req_inner.sampler   = sampler_;

    mtp::MtpChainRunner runner(*mtp_module_, *target, sampler_);
    GenerateResult res = runner.run(req_inner, io, prefill_next, committed, gamma);

    if (!res.ok) {
        std::fprintf(stderr, "[mtp_decode] chain runner failed: %s\n", res.error.c_str());
        return false;
    }

    // Append tokens the runner emitted (runner already streamed them via io.emit).
    for (int32_t t : res.tokens) {
        out_tokens.push_back(t);
    }

    // Log acceptance stats (mirrors harness stderr line 1109).
    const auto & st = runner.stats();
    if (st.total_iters > 0) {
        std::fprintf(stderr,
            "[mtp_decode] iters=%d proposed=%d accepted=%d emitted=%d accept_rate=%.2f\n",
            st.total_iters, st.total_proposed, st.total_accepted, st.total_emitted,
            st.total_proposed > 0
                ? (double)st.total_accepted / (double)st.total_proposed : 0.0);
    }

    return true;
}

}  // namespace dflash27b
