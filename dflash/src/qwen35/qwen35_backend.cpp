#include "qwen35_backend.h"
#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "dflash_feature_ring.h"
#include "dflash_capture.h"
#include "peer_access.h"
#include "attn_masks.h"
#include "common/sampler.h"
#include "common/io_utils.h"
#include "qwen3/qwen3_drafter.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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

    // Load draft
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

    // Create KV cache
    const int max_verify_tokens = cfg_.ddtree_mode
        ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
        : dw_.block_size;
    if (!create_target_cache(w_, cfg_.device.max_ctx, max_verify_tokens, target_backend_, cache_,
                             /*prefill_only=*/true)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return false;
    }

    // Optionally init feature mirror
    if (cfg_.use_feature_mirror && split_gpus_) {
        const int mirror_cap = std::min(cfg_.draft_ctx_max, cfg_.device.max_ctx);
        if (!draft_feature_mirror_init(feature_mirror_, draft_backend_,
                                       cfg_.draft_gpu, cfg_.device.gpu, mirror_cap,
                                       DFLASH27B_DRAFT_N_TARGET_LAYERS,
                                       DFLASH27B_TARGET_HIDDEN)) {
            std::fprintf(stderr, "warning: feature mirror init failed, using direct copies\n");
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
    // Lazy-load drafter on first use
    if (!drafter_loaded_) {
        std::fprintf(stderr, "[compress] loading drafter...\n");
        if (!load_drafter("/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf",
                          /*gpu_layers=*/999, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] drafter init failed: %s\n",
                         dflash27b_last_error());
            io.emit(-1);
            return false;
        }
        drafter_loaded_ = true;
        std::fprintf(stderr, "[compress] drafter ready\n");
    }

    // Park target+draft to free VRAM for the drafter
    const bool was_target_parked = target_parked_;
    const bool was_draft_parked  = draft_parked_;
    if (!target_parked_) park("target");
    if (!draft_parked_)  park("draft");

    // Parse: "compress <n_draft> <prompt_path>"
    std::istringstream iss(line);
    std::string cmd;
    int n_draft = 0;
    std::string prompt_path;
    iss >> cmd >> n_draft >> prompt_path;

    bool ok = false;
    if (n_draft > 0 && !prompt_path.empty()) {
        std::vector<int32_t> tokens = read_int32_file(prompt_path);
        if (!tokens.empty()) {
            const float keep = (float)n_draft / (float)tokens.size();
            auto compressed = drafter_score_and_compress(drafter_ctx_, tokens, keep);
            ok = !compressed.empty();
            if (ok) {
                for (int32_t t : compressed) io.emit(t);
            }
        }
    }
    io.emit(-1);

    // Restore park state
    if (!was_target_parked) unpark("target");
    if (!was_draft_parked)  unpark("draft");

    return ok;
}

void Qwen35Backend::free_drafter() {
    if (drafter_loaded_) {
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
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    // Prefill
    const int committed = do_prefill(req.prompt, io, req.snap_pos, req.snap_slot);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }

    // Decode (speculative)
    if (req.n_gen > 0) {
        if (!do_spec_decode(committed, req.n_gen, result.tokens, io)) {
            result.error = "decode";
            return result;
        }
    }

    result.ok = true;
    return result;
}

// ── Restore + generate ──────────────────────────────────────────────────

GenerateResult Qwen35Backend::restore_and_generate(int slot,
                                                    const GenerateRequest & req,
                                                    const DaemonIO & io) {
    GenerateResult result;
    if (slot < 0 || slot >= PREFIX_SLOTS || !prefix_snapshots_[slot].ctx) {
        result.error = "bad slot";
        io.emit(-1);
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

    // If there are additional prompt tokens beyond the snapshot, prefill them
    int committed = snap_pos;
    if (!req.prompt.empty()) {
        // The prompt here is the diff (tokens beyond the snapshot)
        committed = do_prefill(req.prompt, io, req.snap_pos, req.snap_slot);
        if (committed < 0) {
            result.error = "prefill";
            return result;
        }
    }

    // Decode
    if (req.n_gen > 0) {
        if (!do_spec_decode(committed, req.n_gen, result.tokens, io)) {
            result.error = "decode";
            return result;
        }
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
                               int snap_pos, int snap_slot) {
    (void)io; (void)snap_pos; (void)snap_slot;

    const int hidden = w_.n_embd;
    const int PREFILL_UBATCH = 512;
    const int prompt_len = (int)tokens.size();

    // Migrate to full-mode KV cache if needed
    migrate_prefill_cache(w_, cfg_.device.max_ctx,
                          cfg_.ddtree_mode
                              ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
                              : dw_.block_size,
                          target_backend_, cache_);

    // Chunked prefill
    std::vector<float> embed_buf((size_t)hidden * PREFILL_UBATCH);
    int committed = 0;
    for (int start = 0; start < prompt_len; start += PREFILL_UBATCH) {
        const int n_tokens = std::min(PREFILL_UBATCH, prompt_len - start);
        const bool with_mask = (cfg_.kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/start, /*n_tokens=*/n_tokens,
                               with_mask, /*capture=*/true,
                               /*capture_delta_intermediate=*/false,
                               cfg_.fa_window,
                               /*last_token_logits_only=*/(start + n_tokens < prompt_len),
                               cfg_.kq_stride_pad)) {
            std::fprintf(stderr, "prefill build @%d\n", start);
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
            const int p = start + i;
            pos_buf[4 * i + 0] = p;
            pos_buf[4 * i + 1] = p;
            pos_buf[4 * i + 2] = p;
            pos_buf[4 * i + 3] = 0;
        }
        ggml_backend_tensor_set(sg_.positions, pos_buf.data(), 0,
                                sizeof(int32_t) * pos_buf.size());

        // Mask
        if (sg_.attn_mask) {
            const int win_start = (cfg_.fa_window > 0 && start > cfg_.fa_window)
                                      ? (start - cfg_.fa_window) : 0;
            const int kv_len = start + n_tokens - win_start;
            std::vector<uint16_t> mask_buf;
            build_causal_mask(mask_buf, kv_len, n_tokens, start, cfg_.kq_stride_pad, win_start);
            ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        // Compute
        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "prefill compute @%d failed\n", start);
            return -1;
        }

        // Snapshot at boundary if requested
        if (snap_pos >= 0 && snap_slot >= 0 &&
            start + n_tokens >= snap_pos && start < snap_pos) {
            // Not at boundary yet — next chunk will cross it
        } else if (snap_pos >= 0 && snap_slot >= 0 &&
                   start + n_tokens == snap_pos) {
            cache_.cur_pos = snap_pos;
            snapshot_save(snap_slot);
        }

        committed = start + n_tokens;
        cache_.cur_pos = committed;

        // Sync feature mirror if active
        if (feature_mirror_.target_feat && !draft_parked_) {
            draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, start, n_tokens);
        }
    }

    return committed;
}

bool Qwen35Backend::do_spec_decode(int committed, int n_gen,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io) {
    // TODO: Migrate the speculative decode loop from test_dflash.cpp.
    // This is the ~1200-line core that implements:
    //   1. Draft forward (DFlash model proposes q_len candidate tokens)
    //   2. Target verify (batch verify all candidates)
    //   3. Accept (determine how many candidates match)
    //   4. Rollback SSM state (via fast-rollback intermediates or replay)
    //   5. Emit accepted tokens to stream
    //   6. Repeat until EOS or n_gen reached
    //
    // For now, fall back to simple autoregressive decode (no draft).

    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> logits_buf(vocab);
    std::vector<float> embed_buf_vec(hidden);
    float * embed_buf = embed_buf_vec.data();

    for (int i = 0; i < n_gen; i++) {
        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/committed, /*n_tokens=*/1,
                               /*with_mask=*/false, /*capture=*/true,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/false,
                               cfg_.kq_stride_pad)) {
            return false;
        }

        // Get last generated token (or first prompt token for first iter)
        int32_t tok = out_tokens.empty()
            ? 0  // Should not happen — prefill emits at least one logit
            : out_tokens.back();

        if (i == 0 && out_tokens.empty()) {
            // First decode: read argmax from prefill's last logits
            int32_t argmax = 0;
            ggml_backend_tensor_get(sg_.argmax_tokens, &argmax, 0, sizeof(int32_t));
            tok = argmax;
            out_tokens.push_back(tok);
            io.emit(tok);
            if (IS_EOS_TOK(tok, w_)) { io.emit(-1); return true; }
            committed++;
            continue;
        }

        if (!w_.embedder.embed(&tok, 1, embed_buf)) return false;
        ggml_backend_tensor_set(sg_.inp_embed, embed_buf, 0, sizeof(float) * hidden);
        int32_t pos4[4] = {committed, committed, committed, 0};
        ggml_backend_tensor_set(sg_.positions, pos4, 0, sizeof(int32_t) * 4);

        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) return false;

        // Sample
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

        if (IS_EOS_TOK(next_tok, w_)) break;
    }

    io.emit(-1);
    return true;
}

int Qwen35Backend::verify_chain(int committed, const int32_t * draft_tok, int q_len) {
    (void)committed; (void)draft_tok; (void)q_len;
    // TODO: Will be implemented when the full spec-decode loop is migrated.
    return 0;
}

int Qwen35Backend::verify_tree(int committed, const DDTree & tree) {
    (void)committed; (void)tree;
    // TODO: Will be implemented when the full spec-decode loop is migrated.
    return 0;
}

}  // namespace dflash27b
