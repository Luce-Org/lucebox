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
#include "qwen3/qwen3_drafter.h"

#include "ggml-cuda.h"
#include "common/snapshot_backend.h"
#include "pflash_ggml_adapter.h"
#include "flashprefill.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

namespace dflash::common {

namespace {
static float bf16_bits_to_f32(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)bits << 16;
    return v.f;
}
}  // namespace

#define IS_EOS_TOK(tok, w)                                         \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id)                  \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

// ── Construction / destruction ──────────────────────────────────────────

Qwen35Backend::Qwen35Backend(const Qwen35Config & cfg) : cfg_(cfg) {}

Qwen35Backend::~Qwen35Backend() { shutdown(); }

// ── init() ──────────────────────────────────────────────────────────────

bool Qwen35Backend::init() {
    const bool use_remote_draft = cfg_.remote_draft.enabled();
    split_gpus_ = !use_remote_draft && (cfg_.device.gpu != cfg_.draft_gpu);

    target_backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!target_backend_) {
        std::fprintf(stderr, "target cuda init failed\n");
        return false;
    }
    draft_backend_ = use_remote_draft ? nullptr : target_backend_;
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

    // Snapshot backend: on discrete GPU uses system RAM; on unified memory
    // (Metal, iGPU) stays on compute backend.
    snap_backend_ = create_snapshot_backend(target_backend_);
    if (!snap_backend_) {
        std::fprintf(stderr, "snapshot backend init failed\n");
        return false;
    }

    // Load target
    if (!load_target_model(target_backend_, w_)) {
        std::fprintf(stderr, "target load: %s\n", dflash27b_last_error());
        return false;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    // Load draft
    if (cfg_.draft_path && use_remote_draft) {
        const int cap = cfg_.remote_draft.ring_cap > 0
            ? std::min(cfg_.remote_draft.ring_cap, cfg_.device.max_ctx)
            : std::min(cfg_.device.max_ctx, cfg_.draft_ctx_max);
        if (!remote_draft_.start(cfg_.remote_draft.ipc_bin, cfg_.draft_path,
                                 cfg_.draft_gpu, cap,
                                 cfg_.remote_draft.work_dir)) {
            std::fprintf(stderr, "remote draft start failed\n");
            return false;
        }
        dw_.n_embd = DFLASH27B_TARGET_HIDDEN;
        dw_.block_size = DFLASH27B_DRAFT_BLOCK_SIZE;
        dw_.n_target_layers = DFLASH27B_DRAFT_N_TARGET_LAYERS;
        std::printf("[draft]  remote ipc ready gpu=%d cap=%d\n",
                    cfg_.draft_gpu, cap);
    } else if (cfg_.draft_path) {
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

    // Init feature mirror when draft model is available (needed for spec decode).
    // On single-GPU, this is an F32 conversion buffer; on split-GPU, a cross-device mirror.
    if (cfg_.draft_path && !use_remote_draft) {
        const int mirror_cap = std::min({cfg_.draft_ctx_max, cfg_.device.max_ctx,
                                         cache_.target_feat_cap > 0 ? cache_.target_feat_cap : cfg_.device.max_ctx});
        if (!draft_feature_mirror_init(feature_mirror_, draft_backend_,
                                       cfg_.draft_gpu, cfg_.device.gpu, mirror_cap,
                                       w_.n_capture_layers,
                                       w_.n_embd)) {
            std::fprintf(stderr, "warning: feature mirror init failed, spec decode will use AR fallback\n");
        }
    }

    return true;
}

bool Qwen35Backend::load_target_model(ggml_backend_t backend, TargetWeights & out) {
    return load_target_gguf(cfg_.target_path, backend, out);
}

bool Qwen35Backend::run_ar_decode_path(int committed, int n_gen,
                                       std::vector<int32_t> & out_tokens,
                                       const DaemonIO & io) {
    return do_ar_decode(committed, n_gen, out_tokens, io);
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
    const bool use_remote_draft = cfg_.remote_draft.enabled();

    if (want_draft && !draft_parked_) {
        if (use_remote_draft) {
            remote_draft_.close();
        } else {
            step_graph_destroy(draft_sg_);
            free_draft_weights(dw_);
        }
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
    const bool use_remote_draft = cfg_.remote_draft.enabled();

    if (want_target && target_parked_) {
        if (!load_target_model(target_backend_, w_)) {
            std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
            return false;
        }
        target_parked_ = false;
        std::printf("[unpark] target restored\n"); std::fflush(stdout);
    }
    if (want_draft && draft_parked_ && cfg_.draft_path) {
        if (use_remote_draft) {
            const int cap = cfg_.remote_draft.ring_cap > 0
                ? std::min(cfg_.remote_draft.ring_cap, cfg_.device.max_ctx)
                : std::min(cfg_.device.max_ctx, cfg_.draft_ctx_max);
            if (!remote_draft_.start(cfg_.remote_draft.ipc_bin, cfg_.draft_path,
                                     cfg_.draft_gpu, cap,
                                     cfg_.remote_draft.work_dir)) {
                std::fprintf(stderr, "[unpark] remote draft failed\n");
                return false;
            }
        } else {
            std::string dp(cfg_.draft_path);
            bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
                ? load_draft_gguf(cfg_.draft_path, draft_backend_, dw_, &w_)
                : load_draft_safetensors(cfg_.draft_path, draft_backend_, dw_, &w_);
            if (!draft_ok) {
                std::fprintf(stderr, "[unpark] draft: %s\n", dflash27b_last_error());
                return false;
            }
            // Re-apply rope overrides after reload.
            if (dw_.rope_theta != w_.rope_theta && w_.rope_theta > 0.0f)
                dw_.rope_theta = w_.rope_theta;
            if (dw_.rope_ext_factor == 0.0f && dw_.n_layer == 8 && dw_.n_embd == 2048) {
                float yf = cfg_.draft_yarn_factor > 1.0f ? cfg_.draft_yarn_factor : 64.0f;
                dw_.rope_freq_scale = 1.0f / yf;
                dw_.rope_ext_factor = 1.0f; dw_.rope_attn_factor = 1.0f;
                dw_.rope_beta_fast = cfg_.draft_yarn_beta_fast;
                dw_.rope_beta_slow = cfg_.draft_yarn_beta_slow;
                dw_.rope_n_ctx_orig = cfg_.draft_yarn_orig_ctx;
            }
            if (cfg_.draft_swa_window > 0) {
                dw_.swa_window = cfg_.draft_swa_window;
                for (int il = 0; il < dw_.n_layer - 1; il++)
                    dw_.layers[il].is_swa = true;
            }
        }
        draft_parked_ = false;
        std::printf("[unpark] draft restored\n"); std::fflush(stdout);
    }
    return true;
}

// ── Snapshots ───────────────────────────────────────────────────────────

bool Qwen35Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    PrefixSnapshot & snap = prefix_snapshots_[slot];
    return snapshot_target_cache(w_, cache_, snap_backend_, snap);
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

ModelBackend::SnapshotRef Qwen35Backend::snapshot_ref(int slot) const {
    SnapshotRef ref;
    if (slot < 0 || slot >= PREFIX_SLOTS) return ref;
    const auto & snap = prefix_snapshots_[slot];
    if (!snap.ctx) return ref;
    ref.ctx      = snap.ctx;
    ref.buf      = snap.buf;
    ref.cur_pos  = snap.cur_pos;
    ref.last_tok = snap.last_tok;
    return ref;
}

bool Qwen35Backend::snapshot_adopt(int slot, ggml_context * ctx,
                                   ggml_backend_buffer_t buf, int cur_pos,
                                   int32_t last_tok) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    snapshot_free(slot);

    auto & snap = prefix_snapshots_[slot];

    // Count expected tensor layout from weights.
    const int n_full_attn = w_.n_layer / w_.full_attention_interval;
    const int n_delta     = w_.n_layer - n_full_attn;

    snap.attn_k_snap.assign(n_full_attn, nullptr);
    snap.attn_v_snap.assign(n_full_attn, nullptr);
    snap.ssm_state_snap.assign(n_delta, nullptr);
    snap.conv_state_snap.assign(n_delta, nullptr);
    snap.target_feat_snap = nullptr;

    // Rebind tensors by name.
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->name[0]) continue;
        int idx = -1;
        if (std::sscanf(t->name, "snap_cache_k_%d", &idx) == 1 && idx >= 0 && idx < n_full_attn) {
            snap.attn_k_snap[idx] = t;
        } else if (std::sscanf(t->name, "snap_cache_v_%d", &idx) == 1 && idx >= 0 && idx < n_full_attn) {
            snap.attn_v_snap[idx] = t;
        } else if (std::sscanf(t->name, "snap_ssm_state_%d", &idx) == 1 && idx >= 0 && idx < n_delta) {
            snap.ssm_state_snap[idx] = t;
        } else if (std::sscanf(t->name, "snap_conv_state_%d", &idx) == 1 && idx >= 0 && idx < n_delta) {
            snap.conv_state_snap[idx] = t;
        } else if (std::strcmp(t->name, "snap_target_feat") == 0) {
            snap.target_feat_snap = t;
        }
    }

    // Validate all required tensors are present.
    for (int i = 0; i < n_full_attn; ++i) {
        if (!snap.attn_k_snap[i] || !snap.attn_v_snap[i]) {
            snap.attn_k_snap.clear(); snap.attn_v_snap.clear();
            snap.ssm_state_snap.clear(); snap.conv_state_snap.clear();
            snap.target_feat_snap = nullptr;
            return false;
        }
    }
    for (int i = 0; i < n_delta; ++i) {
        if (!snap.ssm_state_snap[i] || !snap.conv_state_snap[i]) {
            snap.attn_k_snap.clear(); snap.attn_v_snap.clear();
            snap.ssm_state_snap.clear(); snap.conv_state_snap.clear();
            snap.target_feat_snap = nullptr;
            return false;
        }
    }
    if (!snap.target_feat_snap) {
        snap.attn_k_snap.clear(); snap.attn_v_snap.clear();
        snap.ssm_state_snap.clear(); snap.conv_state_snap.clear();
        return false;
    }

    snap.ctx     = ctx;
    snap.buf     = buf;
    snap.cur_pos = cur_pos;
    snap.last_tok        = last_tok;
    snap.kv_k_type       = cache_.kv_k_type;
    snap.max_ctx         = cache_.max_ctx;
    snap.target_feat_cap = cache_.target_feat_cap;
    std::fprintf(stderr, "[qwen35] snapshot adopted slot=%d pos=%d last_tok=%d\n", slot, cur_pos, last_tok);
    return true;
}

// ── Compress (pflash) ───────────────────────────────────────────────────

ModelBackend::CompressResult Qwen35Backend::compress(const CompressRequest & req) {
    CompressResult result;
    if (req.input_ids.empty() || req.drafter_path.empty()) return result;

    // Park target+draft to free VRAM for the drafter (unless skip_park).
    // Also destroy the main target step graph allocator to release its CUDA buffer.
    const bool was_target_parked = target_parked_;
    const bool was_draft_parked  = draft_parked_;
    if (!req.skip_park) {
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
        std::fprintf(stderr, "[compress] loading drafter from %s ...\n",
                     req.drafter_path.c_str());
        if (!load_drafter(req.drafter_path, /*gpu_layers=*/999,
                          req.drafter_gpu, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] drafter init failed: %s\n",
                         dflash27b_last_error());
            if (!req.skip_park) {
                if (!was_target_parked) unpark("target");
                if (!was_draft_parked)  unpark("draft");
            }
            return result;
        }
        drafter_loaded_ = true;
        std::fprintf(stderr, "[compress] drafter ready\n");
    }

    result.compressed_ids = drafter_score_and_compress(
        drafter_ctx_, req.input_ids, req.keep_ratio);
    result.ok = !result.compressed_ids.empty();
    if (result.ok) {
        std::fprintf(stderr, "[compress] %zu -> %zu tokens\n",
                     req.input_ids.size(), result.compressed_ids.size());
    }

    // Keep drafter loaded (own backend + weights persist), matching test_dflash.
    // ~1.4 GB stays resident but avoids reload cost on subsequent compresses.

    // Restore park state
    if (!req.skip_park) {
        if (!was_target_parked) unpark("target");
        if (!was_draft_parked)  unpark("draft");
    }

    return result;
}

bool Qwen35Backend::handle_compress(const std::string & line, const DaemonIO & io) {
    // Check for "nopark" suffix (must be a separate token, not part of a path)
    bool skip_park = (line.size() >= 16 &&
                      line.compare(line.size() - 7, 7, " nopark") == 0);

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

    CompressRequest req;
    req.input_ids = read_int32_file(ppath);
    req.keep_ratio = (float)keep_x1000 / 1000.0f;
    req.drafter_path = (n >= 3 && drafter_path[0])
        ? drafter_path
        : "/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf";
    req.skip_park = skip_park;

    CompressResult result = compress(req);
    for (int32_t t : result.compressed_ids) io.emit(t);
    io.emit(-1);
    return result.ok;
}

void Qwen35Backend::free_drafter() {
    if (drafter_loaded_) {
        // Drafter has its own backend — do a full free (weights + backend)
        dflash::common::free_drafter(drafter_ctx_);
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
            snapshot_target_cache_thin(w_, cache_, snap_backend_,
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
    const bool use_remote_draft = cfg_.remote_draft.enabled();
    free_drafter();
    step_graph_destroy(sg_);
    step_graph_destroy(draft_sg_);
    step_graph_destroy(proj_sg_);
    remote_draft_.close();
    draft_feature_mirror_free(feature_mirror_);
    for (int i = 0; i < PREFIX_SLOTS; i++) {
        free_prefix_snapshot(prefix_snapshots_[i]);
    }
    if (!target_parked_) free_target_weights(w_);
    if (!use_remote_draft && !draft_parked_) free_draft_weights(dw_);
    free_target_cache(cache_);
    if (split_gpus_ && draft_backend_) {
        ggml_backend_free(draft_backend_);
        draft_backend_ = nullptr;
    }
    if (target_backend_) {
        ggml_backend_free(target_backend_);
        target_backend_ = nullptr;
    }
    if (snap_backend_) {
        free_snapshot_backend(snap_backend_, target_backend_);
        snap_backend_ = nullptr;
    }
}

// ── Release scratch buffers between requests ────────────────────────────

void Qwen35Backend::release_scratch() {
    // Target graph allocator: grows during large prefill batches, not needed
    // between requests. Will be lazily recreated on next build_target_step().
    if (sg_.alloc) {
        ggml_gallocr_free(sg_.alloc);
        sg_.alloc = nullptr;
    }
    step_graph_free(sg_);

    // LM-head projection allocator (same pattern).
    if (proj_sg_.alloc) {
        ggml_gallocr_free(proj_sg_.alloc);
        proj_sg_.alloc = nullptr;
    }
    step_graph_free(proj_sg_);

    // BSA persistent CUDA buffers (blockmask, head_mask_type, softmax_lse).
#ifdef DFLASH27B_HAVE_BSA
    flashprefill::dflash_bsa_free_persistent();
#endif

    std::fprintf(stderr, "[vram] released scratch buffers\n");
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

    // Design 1: apply the per-request verify fa_window override (set by
    // http_server when pflash compresses), then restore cfg_.fa_window after
    // this generate completes so concurrent requests aren't affected. Calling
    // dflash_target() lazily constructs it on first use.
    const int eff_fa_window =
        (req.fa_window_override > 0) ? req.fa_window_override : cfg_.fa_window;
    if (auto * dt = dynamic_cast<Qwen35DFlashTarget *>(dflash_target())) {
        dt->set_fa_window(eff_fa_window);
    }

    // Zero delta-net recurrent state (SSM + conv) so a fresh prompt doesn't
    // inherit stale hidden state from the previous request. KV cache is
    // position-addressed and will be overwritten during prefill.
    reset_recurrent_state(cache_);

    // Prefill
    auto t_prefill_start = std::chrono::steady_clock::now();
    const int committed = do_prefill(req.prompt, out_io, req.snap_pos, req.snap_slot);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }
    auto t_prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();

    // C2 adaptive-mechanism gate: pflash's override always reflects the
    // FULL compressed-prompt size — we never cap visibility (would waste
    // pflash's anchor-selection work). The gate here decides whether
    // spec-decode's verify arithmetic still earns its drafter cost at
    // that window size. Threshold 2× cfg_.fa_window:
    //   override <= 4096 (32K → ~1.5K, 64K → ~3K compressed) → spec-decode
    //   override >  4096 (128K → ~6.4K compressed)            → AR fallback
    // AR uses fa_window=0 (full attention) so every kept token is visible
    // regardless of which path runs. We choose mechanism, not visibility.
    const bool fa_within_budget =
        (req.fa_window_override == 0)
     || (eff_fa_window <= 2 * cfg_.fa_window);

    // Decode (speculative or AR)
    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();
        if (!fa_within_budget) {
            // AR fallback: fa_window override too wide for spec decode.
            bool ok = do_ar_decode(committed, req.n_gen, result.tokens, out_io,
                                    req.budget_hook,
                                    &result.budget_forced_close,
                                    &result.degenerate_decode_close);
            out_io.emit(-1);
            if (!ok) { result.error = "decode"; return result; }
        } else {
            // Pass the budget hook into spec-decode. When token count nears
            // the budget edge, do_spec_decode breaks out and tails off via
            // AR with the hook still active — force-close fires correctly
            // without sacrificing spec-decode throughput for the bulk of
            // generation. Most requests never hit the tail because the
            // model closes </think> naturally well before the budget edge.
            if (!do_spec_decode(committed, req.n_gen, result.tokens, out_io,
                                 result.accept_rate, result.spec_decode_ran,
                                 req.hint_tokens, &req.budget_hook,
                                 &result.budget_forced_close,
                                 &result.degenerate_decode_close)) {
                result.error = "decode";
                return result;
            }
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
    if (prompt_len > snap_pos) {
        auto t_prefill_start = std::chrono::steady_clock::now();
        std::vector<int32_t> delta = restore_prompt_delta(req.prompt, snap_pos);
        committed = do_prefill(delta, out_io, req.snap_pos, req.snap_slot, /*kv_offset=*/snap_pos);
        if (committed < 0) {
            result.error = "prefill";
            return result;
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
        // Pass the budget hook into spec-decode. When token count nears
        // the budget edge, do_spec_decode breaks out and tails off via
        // AR with the hook still active — force-close fires correctly
        // without sacrificing spec-decode throughput for the bulk of
        // generation. Most requests never hit the tail because the
        // model closes </think> naturally well before the budget edge.
        if (!do_spec_decode(committed, req.n_gen, result.tokens, out_io,
                             result.accept_rate, result.spec_decode_ran,
                             req.hint_tokens, &req.budget_hook,
                             &result.budget_forced_close,
                             &result.degenerate_decode_close)) {
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
    const int vocab  = w_.n_vocab;
    int prefill_ubatch = 512;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        prefill_ubatch = std::max(1, std::atoi(s));
    }
    const int prompt_len = (int)tokens.size();
    prefill_last_logits_valid_ = false;

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

        // Prefill always uses full attention (fa_window=0) so that all
        // positions encode the complete context — critical for tool
        // definitions at prompt start to propagate into KV values that
        // decode-time windowed attention will later read.
        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/kv_pos, /*n_tokens=*/n_tokens,
                               with_mask, /*capture=*/true,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/(start + n_tokens < prompt_len),
                               cfg_.kq_stride_pad,
                               should_capture_moe_router())) {
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

        // Mask — full attention during prefill (no windowing)
        if (sg_.attn_mask) {
            const int win_start = 0;
            const int kv_len = kv_pos + n_tokens - win_start;
            std::vector<uint16_t> mask_buf;
            const int kv_pad_override = (int)sg_.attn_mask->ne[0];
            build_causal_mask(mask_buf, kv_len, n_tokens, kv_pos, cfg_.kq_stride_pad, win_start, kv_pad_override);
            ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        // Compute
        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "prefill compute @%d failed\n", kv_pos);
            return -1;
        }
        after_target_compute(sg_, kv_pos, n_tokens);

        int32_t last_tok = -1;
        const bool is_final_chunk = (start + n_tokens >= prompt_len);
        const size_t argmax_off =
            is_final_chunk ? sizeof(int32_t) * (size_t)(n_tokens - 1) : 0;
        ggml_backend_tensor_get(sg_.argmax_tokens, &last_tok, argmax_off, sizeof(int32_t));
        cache_.last_tok = last_tok;
        if (is_final_chunk) {
            prefill_last_logits_offset_ = (size_t)(n_tokens - 1) * (size_t)vocab * sizeof(float);
            prefill_last_logits_valid_ = true;
        }

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

        // Sync draft-side features if active.
        if (remote_draft_.active() && !draft_parked_) {
            if (!sync_remote_draft_features(kv_pos, n_tokens)) return -1;
        } else if (feature_mirror_.target_feat && !draft_parked_) {
            draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, kv_pos, n_tokens);
        }

        start += n_tokens;
    }

    return committed;
}

bool Qwen35Backend::do_ar_decode(int committed, int n_gen,
                                  std::vector<int32_t> & out_tokens,
                                  const DaemonIO & io,
                                  const BudgetHook & budget_hook,
                                  bool * forced_close_out,
                                  bool * degenerate_close_out) {
    // Budget hook state.
    //   - budget_close_started: true once we've begun injecting the close
    //     sequence. Prevents re-triggering on continued forward generation.
    //   - close_inject_pos: index into budget_hook.close_token_ids for the
    //     NEXT token to inject. While < close_token_ids.size(), each
    //     iteration overrides the sampled token with the corresponding
    //     close-sequence token (single-token close = 1 override and done;
    //     multi-token close like DeepSeek/laguna [1718,37947,32] = 3
    //     consecutive overrides). Once equal to close_token_ids.size(),
    //     normal sampling resumes (model writes visible answer).
    bool budget_close_started = false;
    int  close_inject_pos     = 0;
    // Capture entry KV position so the budget check is in the
    // "generated since entry" frame, not the absolute KV frame.
    // n_gen is the gen-only count (or the remaining-budget remap done by
    // spec-decode tail-off); subtracting committed_now (absolute KV =
    // prompt_len + tokens generated this call) directly would treat
    // prompt-length tokens as if they were generated output, firing
    // force-close prompt_len tokens early on prompted requests and
    // potentially going negative after spec-decode tail-off.
    const int committed_at_entry = committed;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;

        // Continue an already-started multi-token close sequence.
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size())
        {
            int32_t inj = budget_hook.close_token_ids[close_inject_pos];
            std::fprintf(stderr,
                "[budget-hook] close-seq continue %d/%zu: overriding "
                "sampled token %d with %d\n",
                close_inject_pos + 1,
                budget_hook.close_token_ids.size(), tok, inj);
            tok = inj;
            close_inject_pos++;
            return;
        }

        // Already injected the full sequence — no further overrides.
        if (budget_close_started) return;

        // Check if budget has tightened to the force-close trigger.
        // generated = tokens produced in THIS do_ar_decode call;
        // remaining = budget headroom, measured against n_gen (the
        // requested gen count or tail-off remap, never against the
        // absolute KV position which would mis-count the prompt).
        const int generated = committed_now - committed_at_entry;
        int remaining = n_gen - generated;
        if (remaining <= budget_hook.hard_limit_remaining) {
            // Don't trigger if the model already sampled the first close
            // token naturally — avoids a redundant override.
            int32_t first_close = budget_hook.close_token_ids.front();
            if (tok == first_close) {
                // Model self-closed at the boundary; consume that token
                // as the first of the sequence so we still inject the
                // remaining members (multi-token case) but don't double-emit.
                budget_close_started = true;
                close_inject_pos = 1;
                std::fprintf(stderr,
                    "[budget-hook] model self-emitted close[0]=%d at "
                    "committed=%d/%d (remaining=%d <= hard_limit=%d); "
                    "consuming as start of close sequence (%zu total)\n",
                    first_close, committed_now, n_gen, remaining,
                    budget_hook.hard_limit_remaining,
                    budget_hook.close_token_ids.size());
                return;
            }
            std::fprintf(stderr,
                "[budget-hook] force-close at committed=%d/%d (remaining=%d "
                "<= hard_limit=%d): overriding sampled token %d with close[0]=%d "
                "(seq len %zu)\n",
                committed_now, n_gen, remaining,
                budget_hook.hard_limit_remaining, tok, first_close,
                budget_hook.close_token_ids.size());
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            if (forced_close_out) *forced_close_out = true;
        }
    };
    if (n_gen <= 0) return true;

    auto t_dec0_ar = std::chrono::steady_clock::now();
    const size_t out_tokens_at_entry = out_tokens.size();

    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> logits_buf(vocab);
    std::vector<float> embed_buf_vec(hidden);
    float * embed_buf = embed_buf_vec.data();

    // First token: consume the final prefill position.  Do not derive this
    // offset from committed/KV position: restore paths can prefill a delta at
    // nonzero KV offsets, and committed then no longer describes chunk size.
    //
    // Continuation mode: when out_tokens is non-empty, a previous decode
    // path (e.g. spec-decode tail-off) already committed tokens and emitted
    // them. Skip the first-token block — `committed` and `cache_.last_tok`
    // are already pointing at the most recently committed token, and the
    // main loop below uses out_tokens.back() as the embed input which IS
    // that token. Without this skip we'd duplicate the last token in
    // out_tokens, double-emit it, and advance committed past the actual
    // KV state.
    const int initial_emitted = out_tokens.empty() ? 1 : 0;
    if (initial_emitted == 1) {
        int32_t first_tok;
        if (sampler_.needs_logit_processing()) {
            if (!prefill_last_logits_valid_) return false;
            ggml_backend_tensor_get(sg_.logits, logits_buf.data(), prefill_last_logits_offset_,
                                    sizeof(float) * vocab);
            first_tok = sample_logits(logits_buf.data(), vocab, sampler_,
                                      out_tokens, sampler_rng_);
        } else {
            first_tok = cache_.last_tok;
        }
        maybe_force_close(first_tok, committed);
        out_tokens.push_back(first_tok);
        io.emit(first_tok);
        if (IS_EOS_TOK(first_tok, w_)) return true;
        committed++;
        cache_.cur_pos = committed;
    }

    // AR decode loop for remaining tokens
    for (int i = initial_emitted; i < n_gen; i++) {
        int32_t tok = out_tokens.back();

        if (!w_.embedder.embed(&tok, 1, embed_buf)) return false;
        ggml_backend_tensor_set(sg_.inp_embed, embed_buf, 0, sizeof(float) * hidden);
        int32_t pos4[4] = {committed, committed, committed, 0};
        ggml_backend_tensor_set(sg_.positions, pos4, 0, sizeof(int32_t) * 4);

        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/committed, /*n_tokens=*/1,
                               /*with_mask=*/false, /*capture=*/false,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/false,
                               cfg_.kq_stride_pad,
                               should_capture_moe_router())) {
            return false;
        }

        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) return false;

        after_target_compute(sg_, committed, 1);

        ggml_backend_tensor_get(sg_.logits, logits_buf.data(), 0,
                                sizeof(float) * vocab);
        int32_t next_tok;
        if (sampler_.needs_logit_processing()) {
            next_tok = sample_logits(logits_buf.data(), vocab, sampler_,
                                      out_tokens, sampler_rng_);
        } else {
            next_tok = 0;
            float best = logits_buf[0];
            for (int j = 1; j < vocab; j++) {
                if (logits_buf[j] > best) { best = logits_buf[j]; next_tok = j; }
            }
        }

        maybe_force_close(next_tok, committed);

        out_tokens.push_back(next_tok);
        io.emit(next_tok);
        committed++;
        cache_.cur_pos = committed;
        if (io.cancelled) break;

        if (IS_EOS_TOK(next_tok, w_)) break;

        // Degenerate-decode watchdog. Once we're past the budget-hook's
        // close sequence (model in post-`</think>` content phase), watch
        // for repetition loops. The aime2025-02 case at think_max=4k
        // produces a ~50-token phrase that repeats verbatim until
        // max_tokens — pure waste.
        //
        // Sweep several common loop periods. For each period P we check
        // if the last P tokens equal the previous P tokens (one full
        // repeat). One match is enough; the model has already burned 2P
        // tokens at that point and isn't getting out. The minimum-3
        // bar would catch tighter cycles but waits ~3P tokens to fire,
        // which is wasteful for P ≥ 32. Periods are tuned to common
        // failure modes: short loops (16-24) for "we have X, X, X"
        // patterns, longer (48-64) for full-sentence restates like the
        // aime02 case.
        if (budget_close_started && close_inject_pos >= (int)budget_hook.close_token_ids.size())
        {
            // Sweep contiguous periods 12..80. Any P where the last P
            // tokens equal the previous P tokens means a loop of that
            // period. Stop early on first match. Fixed periods missed
            // the aime02 case which loops with period ~50; dense sweep
            // covers any period in this range.
            auto end = out_tokens.end();
            const int avail = (int)out_tokens.size();
            for (int P = 12; P <= 80; P++) {
                if (avail < 2 * P) break;  // larger P also won't have data
                if (std::equal(end - 2*P, end - P, end - P)) {
                    std::fprintf(stderr,
                        "[degenerate-decode] post-close period=%d repeated — "
                        "breaking AR loop at committed=%d, content_tokens=%zu\n",
                        P, committed,
                        out_tokens.size() - out_tokens_at_entry);
                    if (degenerate_close_out) *degenerate_close_out = true;
                    goto degenerate_break;
                }
            }
        }
        if (false) { degenerate_break: break; }
    }

    auto t_dec1_ar = std::chrono::steady_clock::now();
    const double ar_decode_s = std::chrono::duration<double>(t_dec1_ar - t_dec0_ar).count();
    const int ar_tokens = (int)(out_tokens.size() - out_tokens_at_entry);
    std::fprintf(stderr, "[ar-decode] tokens=%d time=%.3f s speed=%.2f tok/s\n",
                 ar_tokens, ar_decode_s,
                 ar_tokens > 0 && ar_decode_s > 0 ? ar_tokens / ar_decode_s : 0.0);
    return true;
}

bool Qwen35Backend::sync_remote_draft_features(int start_pos, int n_tokens) {
    if (!remote_draft_.active() || !cache_.target_feat || n_tokens <= 0) return true;
    if (cache_.target_feat_cap <= 0) return false;

    const int n_capture = w_.n_capture_layers;
    const int feat_hidden = w_.n_embd;
    const size_t src_stride = cache_.target_feat->nb[1];
    std::vector<float> slice((size_t)n_tokens * (size_t)feat_hidden);
    std::vector<uint16_t> bf16(feat_hidden);
    ggml_backend_synchronize(target_backend_);
    for (int cap_idx = 0; cap_idx < n_capture; ++cap_idx) {
        for (int t = 0; t < n_tokens; ++t) {
            const int slot = (start_pos + t) % cache_.target_feat_cap;
            const size_t src_offset = (size_t)slot * src_stride +
                (size_t)cap_idx * (size_t)feat_hidden * sizeof(uint16_t);
            ggml_backend_tensor_get(cache_.target_feat, bf16.data(),
                                    src_offset,
                                    sizeof(uint16_t) * (size_t)feat_hidden);
            float * dst = slice.data() + (size_t)t * feat_hidden;
            for (int h = 0; h < feat_hidden; ++h) {
                dst[h] = bf16_bits_to_f32(bf16[h]);
            }
        }
        if (!remote_draft_.send_feature_slice(cap_idx, start_pos, n_tokens, slice)) {
            std::fprintf(stderr,
                "spec-decode: remote feature sync failed capture=%d\n",
                cap_idx);
            return false;
        }
    }
    return true;
}

// ── DFlash speculative decode loop ─────────────────────────────────────

bool Qwen35Backend::do_spec_decode(int committed, int n_gen,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io,
                                    float & out_accept_rate,
                                    bool & out_spec_ran,
                                    const std::vector<int32_t> * hint_tokens,
                                    const BudgetHook * budget_hook,
                                    bool * forced_close_out,
                                    bool * degenerate_close_out) {
    out_accept_rate = 0.0f;
    out_spec_ran    = false;
    const int hidden = w_.n_embd;

    // First token: use the argmax that do_prefill already sampled and stored.
    // Reading sg_.argmax_tokens with a computed offset is fragile: when
    // restore_and_generate calls do_prefill with kv_offset != 0, committed
    // reflects total KV position but the last chunk size was
    // delta.size() % ubatch, making (committed % 512) wrong and causing an
    // out-of-bounds tensor read.  cache_.last_tok is always correct.
    int32_t last_tok = cache_.last_tok;

    // Check if we can use speculative decode:
    // - draft model loaded and not parked
    // - feature mirror initialized
    // - greedy decoding (no logit processing) — spec decode uses argmax verification
    const bool can_spec = cfg_.draft_path
        && !draft_parked_
        && (cfg_.remote_draft.enabled()
            ? remote_draft_.active()
            : feature_mirror_.target_feat != nullptr)
        && !sampler_.needs_logit_processing();

    if (!can_spec) {
        // AR fallback consumes the final prefill position itself, then advances
        // one token at a time. Pass the budget hook through so force-close
        // still fires when spec-decode is unavailable.
        bool ok = do_ar_decode(committed, n_gen, out_tokens, io,
                                budget_hook ? *budget_hook : BudgetHook{},
                                forced_close_out, degenerate_close_out);
        io.emit(-1);
        return ok;
    }

    out_spec_ran = true;

    // ── DFlash spec-decode: draft → verify → accept → replay ──────────

    DFlashTarget * target = dflash_target();
    const bool use_remote_draft = cfg_.remote_draft.enabled() && remote_draft_.active();
    const int q_len = dw_.block_size > 0 ? dw_.block_size : DFLASH27B_DRAFT_BLOCK_SIZE;

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
    int n_hint_proposed = 0;
    int n_hint_accepted = 0;

    auto t_dec0 = std::chrono::steady_clock::now();

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        // Budget tail-off: when remaining budget is within the spec-decode
        // batch size of the force-close threshold, hand off to AR for the
        // tail. AR handles the close-token override cleanly; spec-decode's
        // verify-and-accept loop can't safely inject a token mid-batch
        // without a KV-state rewrite.
        //
        // IMPORTANT: cache_.last_tok is set during do_prefill (line 701)
        // and NEVER updated by the spec-decode commit loop — local
        // `last_tok` here is the authoritative most-recently-committed
        // token. Sync it into cache_.last_tok before handing off so AR's
        // `first_tok = cache_.last_tok` seed is correct. Without this
        // sync, AR would re-seed from the prefill's last argmax (stale
        // by `n_generated` positions) and produce garbage continuation.
        if (budget_hook && !budget_hook->close_token_ids.empty()) {
            int hard = budget_hook->hard_limit_remaining;
            // Tail when remaining <= hard + one spec-decode batch worth of
            // headroom. Ensures the force-close fires within the AR tail
            // rather than after a final spec-decode batch overshoots.
            if (need_commit_budget <= hard + q_len) {
                std::fprintf(stderr,
                    "[budget-hook] spec-decode tail-off at committed=%d/%d "
                    "(remaining=%d, hard_limit=%d, batch=%d) — switching to AR\n",
                    committed, n_gen, need_commit_budget, hard, q_len);
                step_graph_destroy(draft_sg);
                cache_.last_tok = last_tok;  // sync spec-decode → AR seed
                // Build a fresh hook keyed off this call's local n_gen
                // (the remaining decode budget) so force-close fires once
                // remaining <= hard_limit relative to AR's loop counter,
                // not relative to the global decode budget. Without this
                // remap force-close would either fire on every iter
                // (negative remaining_for_hook) or never (positive but
                // misscaled).
                BudgetHook tail_hook = *budget_hook;
                int ar_n_gen = need_commit_budget;
                bool ok = do_ar_decode(committed, ar_n_gen, out_tokens, io,
                                        tail_hook, forced_close_out,
                                        degenerate_close_out);
                io.emit(-1);
                return ok;
            }
        }

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
        const int ring_cap = use_remote_draft ? remote_draft_.ring_cap() : feature_mirror_.cap;
        const int draft_ctx = std::min(committed,
            std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            !use_remote_draft &&
            draft_feature_mirror_can_view(feature_mirror_, committed, draft_ctx, mirror_slot0);

        if (use_remote_draft) {
            local_hidden.clear();
            if (!remote_draft_.propose(committed, draft_ctx, noise_embed, local_hidden)) {
                std::fprintf(stderr, "spec-decode: remote draft propose failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
        } else {
            if (!build_draft_step(draft_sg, dw_, /*lm_head=*/nullptr, draft_backend_,
                                  draft_ctx, use_mirror_view ? &feature_mirror_ : nullptr,
                                  committed,
                                  /*ctx_len_max=*/std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)))) {
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
        }

        // 3. Project draft hidden → token IDs via target LM head
        if (!target->project_hidden_to_tokens(local_hidden.data(), q_len, draft_tok)) {
            std::fprintf(stderr, "spec-decode: projection failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        draft_tok[0] = last_tok;

        // 3b. Tool call hint injection: override draft tokens with pre-known
        // structural tokens for near-100% acceptance.
        int hint_fill = 0;
        if (hint_tokens && n_generated < (int)hint_tokens->size()) {
            const int hint_avail = (int)hint_tokens->size() - n_generated;
            hint_fill = std::min(hint_avail, q_len - 1);
            for (int i = 0; i < hint_fill; i++) {
                draft_tok[1 + i] = (*hint_tokens)[n_generated + i];
            }
        }

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
        // Track hint acceptance telemetry.
        if (hint_fill > 0) {
            n_hint_proposed += hint_fill;
            n_hint_accepted += std::min(hint_fill, accept_n - 1);
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
        if (use_remote_draft && cache_.target_feat) {
            if (!sync_remote_draft_features(committed, commit_n)) {
                step_graph_destroy(draft_sg);
                return false;
            }
        } else if (feature_mirror_.target_feat && cache_.target_feat) {
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
    out_accept_rate = (float)((double)n_accept_sum / (double)total_draft_pos);
    std::fprintf(stderr, "[spec-decode] tokens=%d time=%.3f s speed=%.2f tok/s "
                 "steps=%d accepted=%d/%d (%.1f%%) avg_commit=%.2f\n",
                 n_generated, decode_s,
                 n_generated > 0 ? n_generated / decode_s : 0.0,
                 n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                 n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);
    if (n_hint_proposed > 0) {
        std::fprintf(stderr, "[spec-decode] hint tokens: %d/%d accepted (%.1f%%)\n",
                     n_hint_accepted, n_hint_proposed,
                     100.0 * (double)n_hint_accepted / (double)n_hint_proposed);
    }

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

}  // namespace dflash::common
