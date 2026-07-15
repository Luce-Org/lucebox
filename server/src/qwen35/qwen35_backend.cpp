#include "qwen35_backend.h"
#include "common/chain_rollback_policy.h"
#include "placement/skip_park_guard.h"
#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "dflash_feature_ring.h"
#include "dflash_capture.h"
#include "common/dflash_draft_graph.h"
#include "peer_access.h"
#include "attn_masks.h"
#include "common/sampler.h"
#ifdef DFLASH27B_HAVE_GPU_SAMPLER
#include "common/geometric_sampler_cuda.h"
#include <random>
#endif
#include "common/io_utils.h"
#include "common/restore_delta.h"
#include "qwen3/qwen3_drafter.h"
#include "qwen3/qwen3_kvflash_scorer.h"

#include "ggml-cuda.h"
#include "common/snapshot_backend.h"
#include "pflash_ggml_adapter.h"
#include "flashprefill.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "kv_quant.h"

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

static bool tokens_contain_recent_sequence(const std::vector<int32_t> & tokens,
                                           const std::vector<int32_t> & needle,
                                           size_t max_trailing) {
    if (needle.empty() || tokens.size() < needle.size()) return false;
    const size_t last_end = tokens.size();
    const size_t first_end = std::max(
        needle.size(),
        last_end > max_trailing ? last_end - max_trailing : needle.size());
    for (size_t end = first_end; end <= last_end; ++end) {
        const size_t start = end - needle.size();
        if (std::equal(needle.begin(), needle.end(), tokens.begin() + start)) {
            return true;
        }
    }
    return false;
}

static bool tokens_have_recent_any(const std::vector<int32_t> & tokens,
                                   const std::vector<int32_t> & candidates,
                                   size_t max_trailing) {
    if (tokens.empty() || candidates.empty()) return false;
    for (size_t trailing = 0; trailing <= max_trailing; ++trailing) {
        if (tokens.size() <= trailing) break;
        const int32_t tok = tokens[tokens.size() - 1 - trailing];
        if (std::find(candidates.begin(), candidates.end(), tok) != candidates.end()) {
            return true;
        }
    }
    return false;
}

static int env_int_or_default(const char * name, int fallback) {
    if (const char * raw = std::getenv(name)) {
        if (*raw) return std::atoi(raw);
    }
    return fallback;
}

static int dflash_min_tokens_floor() {
    static const int value = env_int_or_default("DFLASH_MIN_TOKENS", 0);
    return value;
}

static FILE * open_dflash_floor_log() {
#if defined(_WIN32)
    // Simple append-mode log on Windows (no file size check).
    return std::fopen("dflash_floor.log", "a");
#else
    static constexpr const char * kPath = "/tmp/dflash_floor.log";
    static constexpr off_t kMaxBytes = 1024 * 1024;

    int flags = O_WRONLY | O_CREAT | O_APPEND;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    int fd = ::open(kPath, flags, 0600);
    if (fd < 0) return nullptr;

    struct stat st;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        return nullptr;
    }
    if (st.st_size > kMaxBytes) {
#ifdef O_NOFOLLOW
        if (::ftruncate(fd, 0) != 0) {
            ::close(fd);
            return nullptr;
        }
#else
        ::close(fd);
        return nullptr;
#endif
    }

    FILE * out = fdopen(fd, "a");
    if (!out) ::close(fd);
    return out;
#endif
}
}  // namespace

#define IS_EOS_TOK(tok, w)                                         \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id)                  \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

static bool qwen35_empty_visible_output(const std::vector<int32_t> & tokens,
                                        const TargetWeights & w) {
    if (tokens.empty()) return false;
    for (int32_t tok : tokens) {
        if (!IS_EOS_TOK(tok, w)) return false;
    }
    return true;
}

// ── Construction / destruction ──────────────────────────────────────────

Qwen35Backend::Qwen35Backend(const Qwen35Config & cfg) : cfg_(cfg) {}

Qwen35Backend::~Qwen35Backend() { shutdown(); }

// "auto" pool budget: device-free minus a reserve (compute buffers + drafter
// when expected), converted at this model's pooled-KV density. Shared with MoE
// placement so reservation and runtime allocation size the pool identically.
KvFlashAutoBudget Qwen35Backend::make_kvflash_budget(const TargetWeights & w,
                                                     int64_t gpu_free) const {
    ggml_type kv_k = GGML_TYPE_Q8_0, kv_v = GGML_TYPE_Q8_0;
    dflash::resolve_kv_types(kv_k, kv_v);
    KvFlashAutoBudget b;
    b.free_bytes      = gpu_free;
    // Single source of truth with the qwen35moe placement path — see kv_quant.h.
    b.bytes_per_token = (int64_t)dflash::kv_reservation_bytes_per_token(
        w.n_layer, w.full_attention_interval, w.n_head_kv,
        kv_k, w.n_embd_head_k, kv_v, w.n_embd_head_v);
    b.reserve_bytes   = (int64_t)(1.5 * 1073741824.0) +
        (kvflash_drafter_path_.empty() ? 0 : (int64_t)(1.7 * 1073741824.0));
    return b;
}

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
    // kvflash (bounded residency): pool size from the env, rounded/floored/
    // clamped by the shared reader (256-stride keeps FA vec-kernel
    // eligibility; the floor keeps eviction from deadlocking).
    // Drafter-scored residency is the DEFAULT policy: explicit
    // --prefill-drafter first, then the well-known locations next to the
    // model (Spark's pattern). LRU is the fallback when nothing is found
    // (or the explicit choice via --kvflash-policy lru).
    kvflash_qk_policy_ = kvflash_policy_is_qk();
    if (std::getenv("DFLASH_KVFLASH") && !kvflash_qk_policy_) {
        kvflash_drafter_path_ = kvflash_find_drafter(cfg_.target_path);
    }
    // "auto" sizes the pool from the GPU: weights are resident at this
    // point and the cache is not yet allocated, so device-free minus a
    // reserve (compute buffers + the drafter when expected) is what the
    // pool can really use, converted at this model's pooled-KV density.
    size_t gpu_free = 0, gpu_total = 0;
    if (ggml_backend_dev_t dev = ggml_backend_get_device(target_backend_)) {
        ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
    }
    KvFlashAutoBudget kvf_budget = make_kvflash_budget(w_, (int64_t)gpu_free);
    kvflash_tokens_ = kvflash_pool_from_env(cfg_.device.max_ctx, KvFlashConfig{},
                                            kvflash_scorer_expected(),
                                            kvf_budget);
    if (kvflash_tokens_ > 0) {
        kvflash_tau_ = std::max(1, env_int_or_default("DFLASH_KVFLASH_TAU", 64));
    }
    // Subclass gate (e.g. MoE all-hot): may zero kvflash_tokens_ before the KV
    // cache is sized, so create_target_cache allocates full max_ctx KV.
    if (!post_kvflash_init_gate()) return false;
    if (!create_target_cache(w_, cfg_.device.max_ctx, max_verify_tokens, target_backend_, cache_,
                             /*prefill_only=*/true, /*ctx_alloc=*/kvflash_tokens_)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return false;
    }
    if (kvflash_active()) {
        KvFlashConfig pc;
        pc.pool_tokens = kvflash_tokens_;
        if (!kvflash_pager_.attach(pc, cache_.attn_k, cache_.attn_v)) {
            std::fprintf(stderr, "kvflash: pager attach failed (pool=%d)\n", kvflash_tokens_);
            return false;
        }
        if (kvflash_qk_policy_) {
            KvFlashQkDims qd;
            qd.n_layers   = (int)cache_.attn_k.size();
            qd.n_q_heads  = w_.n_head;
            qd.n_kv_heads = w_.n_head_kv;
            qd.head_dim   = w_.n_embd_head_k;
            kvflash_qk_pool_.reset(qd);
            auto qs = std::make_unique<KvFlashTargetQkScorer>(&kvflash_qk_pool_);
            kvflash_qk_scorer_ = qs.get();
            kvflash_scorer_ = std::move(qs);
        }
        std::printf("[kvflash] resident pool %d tokens (logical max_ctx %d), "
                    "tau=%d, policy=%s\n",
                    kvflash_tokens_, cfg_.device.max_ctx, kvflash_tau_,
                    kvflash_qk_policy_ ? "qk (target pooled-K vs decode query)"
                    : !kvflash_drafter_path_.empty()
                        ? "drafter (attaches on first reselect)"
                        : "lru (recency-only: no Qwen3-0.6B drafter found "
                          "next to the model or in --prefill-drafter)");
        std::fflush(stdout);
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
        kvflash_drafter_failed_ = false;   // fresh VRAM: allow a retry
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
    // kvflash: snapshots right-size to cur_pos, which is a LOGICAL position
    // that can exceed the physical pool once decode has paged, and they copy
    // rows assuming the identity layout, which pooled prefill / eviction
    // breaks. Snapshots of pooled state need page-table serialization
    // (follow-up); identity-mapped prefill-time snapshots remain valid.
    if (kvflash_active() &&
        (cache_.cur_pos > kvflash_tokens_ || !kvflash_pager_.is_identity())) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr, "[kvflash] snapshot skipped: cur_pos %d exceeds "
                                 "pool %d (pooled snapshots are a follow-up)\n",
                         cache_.cur_pos, kvflash_tokens_);
            warned = true;
        }
        return false;
    }
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

bool Qwen35Backend::restore_target_cache_from_snapshot(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || !prefix_snapshots_[slot].ctx) return false;
    return restore_target_cache(prefix_snapshots_[slot], cache_);
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
        // pflash + kvflash synergy: the drafter doubles as the pool's
        // Memory Indexer (tau-step reselect). Pager stays LRU without it.
        if (kvflash_active() && !kvflash_scorer_) {
            kvflash_scorer_ = std::make_unique<KvFlashDrafterScorer>(&drafter_ctx_);
            std::fprintf(stderr, "[kvflash] drafter scorer attached (tau=%d)\n",
                         kvflash_tau_);
        }
    }

    result.compressed_ids = drafter_score_and_compress(
        drafter_ctx_, req.input_ids, req.keep_ratio);
    result.ok = !result.compressed_ids.empty();
    if (result.ok) {
        std::fprintf(stderr, "[compress] %zu -> %zu tokens\n",
                     req.input_ids.size(), result.compressed_ids.size());
    }

    if (req.residency_action == DraftResidencyAction::ReleaseAfterUse) {
        free_drafter();
    }

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
    {
        size_t total_vram = 0;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
            total_vram = prop.totalGlobalMem;
        const bool allowed = dflash::common::skip_park_allowed(
            skip_park, total_vram, cfg_.device.max_ctx);
        if (skip_park && !allowed) {
            std::fprintf(stderr,
                "[server] --prefill-skip-park downgraded: <32GB GPU with max_ctx>65536"
                " (VMM VA-fragmentation guard)\n");
        }
        req.skip_park = allowed;
    }

    CompressResult result = compress(req);
    for (int32_t t : result.compressed_ids) io.emit(t);
    io.emit(-1);
    return result.ok;
}

void Qwen35Backend::free_drafter() {
    if (drafter_loaded_) {
        // The kvflash DRAFTER scorer borrows drafter_ctx_; drop it first.
        // The target-QK scorer is drafter-independent and survives.
        if (!kvflash_qk_scorer_) kvflash_scorer_.reset();
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
        auto * qt = static_cast<Qwen35DFlashTarget *>(dflash_target_.get());
        if (kvflash_active()) {
            qt->set_kvflash_pager(&kvflash_pager_);
        }
        qt->set_fast_rollback(cfg_.fast_rollback);
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
    draft_kv_free(draft_kv_);
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

GenerateResult Qwen35Backend::generate_impl(const GenerateRequest & req,
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

    // Prefill
    auto t_prefill_start = std::chrono::steady_clock::now();
    const int committed = do_prefill(req.prompt, out_io, req.snap_pos, req.snap_slot);
    if (committed < 0) {
        result.fail(GenerateErrorCode::PrefillFailed);
        return result;
    }
    auto t_prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();

    // Decode (speculative)
    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();
        // Pass the budget hook into spec-decode. When token count nears
        // the budget edge, do_spec_decode breaks out and tails off via
        // AR with the hook still active — force-close fires correctly
        // without sacrificing spec-decode throughput for the bulk of
        // generation. Most requests never hit the tail because the
        // model closes </think> naturally well before the budget edge.
        bool decode_ok = false;
        if (req.force_ar_decode) {
            decode_ok = do_ar_decode(committed, req.n_gen, result.tokens, out_io,
                                     req.budget_hook,
                                     &result.budget_forced_close,
                                     &result.degenerate_decode_close);
            out_io.emit(-1);
        } else {
            decode_ok = do_spec_decode(committed, req.n_gen, result.tokens, out_io,
                                       result.accept_rate, result.spec_decode_ran,
                                       req.hint_tokens,
                                       req.stall_tool_prefix_tokens,
                                       req.stall_action_suffix_tokens,
                                       req.stall_skip_tokens,
                                       &req.budget_hook,
                                       &result.budget_forced_close,
                                       &result.degenerate_decode_close);
            if (decode_ok) {
                result.empty_visible_output =
                    qwen35_empty_visible_output(result.tokens, w_);
            }
        }
        if (!decode_ok) {
            result.fail(GenerateErrorCode::DecodeFailed);
            return result;
        }
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }

    result.succeed();
    return result;
}

// ── Restore + generate ──────────────────────────────────────────────────

GenerateResult Qwen35Backend::restore_and_generate_impl(int slot,
                                                        const GenerateRequest & req,
                                                        const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (slot < 0 || slot >= PREFIX_SLOTS || !prefix_snapshots_[slot].ctx) {
        result.fail(GenerateErrorCode::InvalidSnapshotSlot);
        out_io.emit(-1);
        return result;
    }

    // Clear-then-restore: the step-invariant decode reads a 256-padded,
    // mask-less FA span, so rows beyond the restored prefix must be ZERO,
    // not leftovers from the previous request. cudaMemset is ~0.2ms.
    if (cache_.base_buf) ggml_backend_buffer_clear(cache_.base_buf, 0);

    // Restore snapshot
    restore_target_cache(prefix_snapshots_[slot], cache_);

    // Now generate from restored state
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    const int snap_pos = prefix_snapshots_[slot].cur_pos;
    cache_.cur_pos = snap_pos;

    // FIX(prefix-cache + spec-decode): restore_target_cache brings back KV /
    // recurrent state / target_feat, but the draft-side feature mirror is left
    // stale. Without re-syncing it the draft emits -1 -> verify_batch embed
    // fails -> empty output on every cache hit. Mirror what do_prefill does.
    if (!draft_parked_) {
        const int ring_cap = remote_draft_.active() ? remote_draft_.ring_cap()
                                                     : feature_mirror_.cap;
        const int n = std::min(cache_.cur_pos, ring_cap);
        if (n > 0) {
            const int start = cache_.cur_pos - n;
            if (remote_draft_.active()) {
                sync_remote_draft_features(start, n);
            } else if (feature_mirror_.target_feat && cache_.target_feat) {
                draft_feature_mirror_sync_range(cache_.target_feat,
                                                cache_.target_feat_cap,
                                                feature_mirror_, start, n);
            }
        }
    }

    // Daemon receives the FULL prompt; slice off the cached prefix and prefill
    // only the delta at KV positions [snap_pos, snap_pos + delta.size()).
    int committed = snap_pos;
    const int prompt_len = (int)req.prompt.size();
    if (prompt_len > snap_pos) {
        auto t_prefill_start = std::chrono::steady_clock::now();
        std::vector<int32_t> delta = restore_prompt_delta(req.prompt, snap_pos);
        committed = do_prefill(delta, out_io, req.snap_pos, req.snap_slot, /*kv_offset=*/snap_pos);
        if (committed < 0) {
            result.fail(GenerateErrorCode::PrefillFailed);
            return result;
        }
        result.prefill_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    } else if (prompt_len > 0 && prompt_len < snap_pos) {
        // The slot's snapshot covers more KV than the new prompt. This is
        // routine with agent clients (Letta, Hermes, ...) that edit or
        // summarize their history between turns. Fall back to a fresh full
        // prefill instead of failing the request with zero tokens.
        std::fprintf(stderr,
            "[pc] snapshot longer than prompt (snap=%d > prompt=%d) — "
            "fresh prefill fallback\n", snap_pos, prompt_len);
        reset_recurrent_state(cache_);
        cache_.cur_pos = 0;
        auto t_prefill_start = std::chrono::steady_clock::now();
        committed = do_prefill(req.prompt, out_io, req.snap_pos, req.snap_slot);
        if (committed < 0) {
            result.fail(GenerateErrorCode::PrefillFailed);
            return result;
        }
        result.prefill_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    } else {
        // Exact full-prompt cache hit (prompt_len == snap_pos): no prefill ran,
        // so the per-request decode step-graph sg_ was never (re)built. The
        // first decode step (do_ar_decode / do_spec_decode) writes sg_.inp_embed
        // BEFORE its own build_target_step, so a null/freed graph tensor aborts
        // in ggml_backend_tensor_set. Build a single-token decode step graph at
        // the restored position now, mirroring do_ar_decode's per-step build.
        const bool pool = kvflash_active();
        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/cache_.cur_pos, /*n_tokens=*/1,
                               /*with_mask=*/pool, /*capture=*/false,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/false,
                               cfg_.kq_stride_pad,
                               should_capture_moe_router(),
                               /*kvflash_mask=*/pool,
                               /*capture_qk=*/pool && kvflash_qk_policy_)) {
            result.fail(GenerateErrorCode::BackendSpecific,
                             "restore step-graph build");
            return result;
        }
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
        bool decode_ok = false;
        if (req.force_ar_decode) {
            decode_ok = do_ar_decode(committed, req.n_gen, result.tokens, out_io,
                                     req.budget_hook,
                                     &result.budget_forced_close,
                                     &result.degenerate_decode_close);
            out_io.emit(-1);
        } else {
            decode_ok = do_spec_decode(committed, req.n_gen, result.tokens, out_io,
                                       result.accept_rate, result.spec_decode_ran,
                                       req.hint_tokens,
                                       req.stall_tool_prefix_tokens,
                                       req.stall_action_suffix_tokens,
                                       req.stall_skip_tokens,
                                       &req.budget_hook,
                                       &result.budget_forced_close,
                                       &result.degenerate_decode_close);
            if (decode_ok) {
                result.empty_visible_output =
                    qwen35_empty_visible_output(result.tokens, w_);
            }
        }
        if (!decode_ok) {
            result.fail(GenerateErrorCode::DecodeFailed);
            return result;
        }
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }

    result.succeed();
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

    // kvflash: a prompt that fits the pool prefills contiguously (identity
    // mapping, normal chunking). A LARGER prompt switches to POOLED CHUNKED
    // PREFILL: pager-chunk-sized batches whose KV rows are slot-mapped via
    // set_rows, with a slot-space mask per chunk and live eviction as the
    // pool fills (constant VRAM, linear time). Restore offsets are not
    // supported in the pooled path (a relocated prefix cannot be restored
    // identity-style in the first place).
    const bool kvf_paged = kvflash_active() &&
        kv_offset + prompt_len > kvflash_tokens_ - kvflash_pager_.chunk_tokens();
    if (kvf_paged && kv_offset != 0) {
        std::fprintf(stderr,
            "[kvflash] restored prefix (%d) + prompt (%d) exceeds pool %d; "
            "pooled prefill requires a fresh request\n",
            kv_offset, prompt_len, kvflash_tokens_);
        set_last_error("kvflash: restore + pooled prefill unsupported");
        return -1;
    }
    if (kvf_paged) {
        prefill_ubatch = kvflash_pager_.chunk_tokens();
        kvflash_pager_.reset();
        if (kvflash_qk_policy_) {
            kvflash_qk_pool_.reset(kvflash_qk_pool_.dims());
            kvflash_qk_pooled_upto_ = 0;
        }
        std::printf("[kvflash] pooled prefill: %d tokens through a %d-token pool "
                    "(%d-token chunks, evicting)\n",
                    prompt_len, kvflash_tokens_, prefill_ubatch);
        std::fflush(stdout);
    }

    // Skip KV-cache migration when resuming from a snapshot — the cache was
    // already migrated when the snapshot was taken; re-running migrate would
    // clobber the restored state.
    if (kv_offset == 0) {
        const int max_verify_tokens = cfg_.ddtree_mode
            ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
            : dw_.block_size;
        if (!migrate_prefill_cache(w_, cfg_.device.max_ctx,
                                   max_verify_tokens,
                                   target_backend_, cache_)) {
            std::fprintf(stderr, "prefill: rollback cache migration failed: %s\n",
                         dflash27b_last_error());
            return -1;
        }
    }

    // Chunked prefill
    std::vector<float> embed_buf((size_t)hidden * prefill_ubatch);
    int committed = kv_offset;
    for (int start = 0; start < prompt_len;) {
        const int kv_pos = kv_offset + start;

        int n_tokens = std::min(prefill_ubatch, prompt_len - start);
        // FIX(bug2): do NOT shrink the prefill chunk to snap_pos. Shrinking
        // realigns every subsequent chunk, changing GPU batch sizes vs the
        // no-cache path -> FP-nondeterministic state divergence -> different
        // greedy output on cache hits. Keep uniform chunks. When snap_pos falls
        // inside this chunk, snapshot at the chunk START boundary kv_pos: the
        // largest chunk boundary <= snap_pos. That stays (a) chunk-aligned, so
        // the prefill is bit-identical to the no-cache path, and (b) strictly
        // within the requested prefix, so a later request that shares only the
        // system-prompt prefix still restores a valid cross-request hit.
        // (Rounding UP would push the snapshot to prompt end -> the full prompt
        // incl. the user message -> a different user msg restores garbage.)
        if (snap_slot >= 0 && snap_pos >= 0 &&
            kv_pos <= snap_pos && snap_pos < kv_pos + n_tokens) {
            if (kv_pos > kv_offset && !kvf_paged) {   // skip degenerate / relocated
                cache_.cur_pos = kv_pos;
                if (snapshot_save(snap_slot)) {
                    std::printf("[snap] boundary slot=%d cur_pos=%d (req snap_pos=%d)\n",
                                snap_slot, kv_pos, snap_pos);
                    std::fflush(stdout);
                }
            } else if (kvf_paged) {
                std::fprintf(stderr, "[kvflash] boundary snapshot skipped: pooled "
                                     "prefill relocates chunks\n");
            }
            snap_pos = -1;
            snap_slot = -1;
        }
        const bool with_mask = kvf_paged ||
            (cfg_.kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

        // kvflash pooled prefill: allocate this chunk's slots up front
        // (evicting the lowest-priority resident chunk once the pool fills).
        std::vector<int> kvf_slots;
        if (kvf_paged) {
            kvf_slots.resize((size_t)n_tokens);
            bool ok = true;
            for (int i = 0; i < n_tokens; i++) {
                kvf_slots[(size_t)i] = kvflash_pager_.slot_for(kv_pos + i);
                if (kvf_slots[(size_t)i] < 0) { ok = false; break; }
            }
            if (!ok) {
                std::fprintf(stderr, "[kvflash] pooled prefill: slot alloc failed @%d\n", kv_pos);
                set_last_error("kvflash: no evictable pool block");
                return -1;
            }
        }

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
                               should_capture_moe_router(),
                               /*kvflash_mask=*/kvf_paged)) {
            std::fprintf(stderr, "prefill build @%d\n", kv_pos);
            return -1;
        }
        if (kvf_paged) {
            if (!sg_.kv_write_rows) {
                std::fprintf(stderr, "[kvflash] pooled prefill requires the set_rows path\n");
                return -1;
            }
            // [n_tokens, n_head_kv] ne0-major (see verify_batch).
            std::vector<int64_t> rows((size_t)n_tokens * w_.n_head_kv);
            for (int h = 0; h < w_.n_head_kv; h++) {
                for (int i = 0; i < n_tokens; i++) {
                    rows[(size_t)h * n_tokens + i] = kvf_slots[(size_t)i];
                }
            }
            ggml_backend_tensor_set(sg_.kv_write_rows, rows.data(), 0,
                                    sizeof(int64_t) * rows.size());
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
        if (sg_.attn_mask && kvf_paged) {
            // Slot-space mask (same recipe as verify_batch): row q attends
            // (a) the slots of resident chunks holding positions < kv_pos
            // and (b) this chunk's own slots, causally.
            constexpr uint16_t F16_ZERO = 0x0000, F16_NEG_INF = 0xFC00;
            const size_t kvd = (size_t)sg_.attn_mask->ne[0];
            const int q_pad = (int)sg_.attn_mask->ne[1];
            std::vector<uint16_t> mask_buf((size_t)kvd * q_pad, F16_NEG_INF);
            const int ct = kvflash_pager_.chunk_tokens();
            for (int c = 0; c < kvflash_pager_.n_chunks(); c++) {
                const int blk = kvflash_pager_.block_of(c);
                if (blk < 0) continue;
                for (int i = 0; i < ct; i++) {
                    if ((int64_t)c * ct + i >= kv_pos) break;
                    mask_buf[(size_t)blk * ct + i] = F16_ZERO;
                }
            }
            for (int q = 1; q < n_tokens; q++) {
                std::memcpy(mask_buf.data() + (size_t)q * kvd, mask_buf.data(), kvd * 2);
            }
            for (int q = 0; q < n_tokens; q++) {
                for (int i = 0; i <= q; i++) {
                    mask_buf[(size_t)q * kvd + kvf_slots[(size_t)i]] = F16_ZERO;
                }
            }
            ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        } else if (sg_.attn_mask) {
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

        // QK policy: pool the post-RoPE keys of chunks this batch sealed
        // (they are resident — sealed inside the protected tail window).
        if (kvflash_active() && kvflash_qk_policy_) kvflash_qk_pool_to(committed);

        // Sync draft-side features if active.
        if (remote_draft_.active() && !draft_parked_) {
            if (!sync_remote_draft_features(kv_pos, n_tokens)) return -1;
        } else if (feature_mirror_.target_feat && !draft_parked_) {
            draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, kv_pos, n_tokens);
        }

        start += n_tokens;
    }

    if (kvflash_active()) {
        if (kvf_paged) {
            // The pager mapping was built live during the pooled prefill;
            // only the history / hygiene parts of the sync apply.
            kvflash_history_.assign(tokens.begin(), tokens.end());
            kvflash_pager_.zero_free_blocks();
            kvflash_mask_epoch_ = (uint64_t)-1;
        } else {
            kvflash_sync_prefill(committed, tokens, kv_offset);
        }
    }

    // End-of-prefill snapshot: scoped disk-cache saves (auto/fixed policy)
    // request snap_pos == prompt end, which never falls inside a chunk so the
    // boundary branch above cannot fire. Taking the snapshot here changes
    // nothing about the prefill computation; it only persists the final state
    // (cache_.cur_pos == committed).
    if (snap_slot >= 0 && snap_pos == committed) {
        if (snapshot_save(snap_slot)) {
            std::printf("[snap] end-of-prefill slot=%d cur_pos=%d\n",
                        snap_slot, committed);
            std::fflush(stdout);
        }
    }

    return committed;
}

// ── kvflash helpers ─────────────────────────────────────────────────

void Qwen35Backend::kvflash_sync_prefill(int committed,
                                         const std::vector<int32_t> & tokens,
                                         int kv_offset) {
    // Prefill (and snapshot restore) place rows physically contiguous at
    // [0, committed): rebuild the pager mapping identity-style and reset
    // the token history to match.
    kvflash_pager_.reset();
    for (int p = 0; p < committed; p++) {
        const int slot = kvflash_pager_.slot_for(p);
        if (slot != p) {
            // Cannot happen while prompt <= pool (blocks are handed out in
            // order from a freshly reset pager); guard against future
            // changes to the hand-out order.
            std::fprintf(stderr, "[kvflash] prefill slot mismatch %d != %d\n", slot, p);
        }
    }
    if (kv_offset == 0) {
        kvflash_history_.assign(tokens.begin(), tokens.end());
    } else {
        kvflash_history_.resize((size_t)kv_offset, 0);  // restored prefix ids unknown
        kvflash_history_.insert(kvflash_history_.end(), tokens.begin(), tokens.end());
    }
    // Slots past the prompt still hold the previous request's rows; the
    // maskless qwen35moe pipelined decode reads the whole padded pool span.
    kvflash_pager_.zero_free_blocks();
    kvflash_mask_epoch_ = (uint64_t)-1;
    if (kvflash_qk_policy_) {
        kvflash_qk_pool_.reset(kvflash_qk_pool_.dims());
        kvflash_qk_pooled_upto_ = 0;
        kvflash_qk_pool_to(committed);
    }
}

// Pool post-RoPE keys for chunks sealed before `committed` (QK policy).
// At seal time a chunk sits inside the protected tail window, so it is
// resident; a non-resident chunk here means we were called late (e.g.
// restored state) — it is skipped and scores missing (0) until repooled.
void Qwen35Backend::kvflash_qk_pool_to(int committed) {
    const int ct = kvflash_pager_.chunk_tokens();
    const int sealed = committed / ct;
    for (int c = kvflash_qk_pooled_upto_; c < sealed; c++) {
        const int blk = kvflash_pager_.block_of(c);
        if (blk < 0 || !kvflash_qk_pool_.pool_chunk(cache_.attn_k, blk, ct, c)) {
            std::fprintf(stderr, "[kvflash-qk] pool_chunk failed for chunk %d "
                                 "(block %d); chunk scores as missing\n", c, blk);
        }
    }
    kvflash_qk_pooled_upto_ = std::max(kvflash_qk_pooled_upto_, sealed);
}

void Qwen35Backend::kvflash_upload_mask() {
    if (!sg_.attn_mask) return;
    const size_t need = (size_t)sg_.attn_mask->ne[0] * sg_.attn_mask->ne[1];
    if (kvflash_mask_buf_.size() != need || kvflash_pager_.epoch() != kvflash_mask_epoch_) {
        kvflash_mask_buf_.assign(need, F16_NEG_INF);
        kvflash_pager_.fill_slot_mask(kvflash_mask_buf_.data());   // q row 0
        kvflash_mask_epoch_ = kvflash_pager_.epoch();
    }
    // Upload before EVERY compute: the input tensor's buffer region is
    // reused by graph execution, so a stale upload reads back as garbage.
    ggml_backend_tensor_set(sg_.attn_mask, kvflash_mask_buf_.data(), 0,
                            need * sizeof(uint16_t));
}

// Attach the drafter as the residency scorer outside the pflash compress
// path: with `--kvflash --prefill-drafter <gguf>` but compression off, the
// drafter would otherwise never load and the pool would silently run
// recency-only LRU. Loads lazily on the first reselect that needs it (and
// re-attaches after a draft-residency release frees the drafter).
void Qwen35Backend::kvflash_ensure_scorer() {
    if (kvflash_scorer_ || kvflash_drafter_path_.empty() || kvflash_drafter_failed_) {
        return;
    }
    if (!drafter_loaded_) {
        ggml_backend_synchronize(target_backend_);
        if (draft_backend_) ggml_backend_synchronize(draft_backend_);
        std::fprintf(stderr, "[kvflash] loading drafter for residency scoring: %s\n",
                     kvflash_drafter_path_.c_str());
        if (!load_drafter(kvflash_drafter_path_, /*gpu_layers=*/999,
                          cfg_.device.gpu, drafter_ctx_)) {
            std::fprintf(stderr, "[kvflash] drafter load failed (%s); staying on "
                                 "LRU residency\n", dflash27b_last_error());
            kvflash_drafter_failed_ = true;
            return;
        }
        drafter_loaded_ = true;
    }
    kvflash_scorer_ = std::make_unique<KvFlashDrafterScorer>(&drafter_ctx_);
    std::fprintf(stderr, "[kvflash] drafter scorer attached (tau=%d)\n", kvflash_tau_);
}

void Qwen35Backend::kvflash_maybe_reselect(int generated) {
    if (kvflash_tau_ <= 0) return;
    // Adaptive tau: a rescore costs ~0.11 ms per history token (full 0.6B
    // re-prefill; measured 0.9 s @8K, ~46 s bisected @256K), while decode
    // produces ~30 tok/s. Capping rescore overhead at ~15% of decode time
    // gives tau ~= history/45. The configured tau is the floor.
    const int tau = std::max<int>(kvflash_tau_, (int)(kvflash_history_.size() / 45));
    if (generated % tau != 0) return;
    // Lazy-load the drafter only when a rescore is actually due, so the
    // first tokens of the first request never pay the load.
    if (!kvflash_scorer_) kvflash_ensure_scorer();
    if (!kvflash_scorer_) return;
    if (kvflash_qk_scorer_) {
        // Feed the last decode step's captured query (post-RoPE/-rotation,
        // [n_fa, n_head, head_dim] f32) before scoring.
        if (!cache_.q_cap) return;
        std::vector<float> q((size_t)ggml_nelements(cache_.q_cap));
        ggml_backend_tensor_get(cache_.q_cap, q.data(), 0, q.size() * sizeof(float));
        kvflash_qk_scorer_->set_query(q.data(), q.size());
    }
    if (!kvflash_scorer_->score_chunks(kvflash_history_, kvflash_pager_.chunk_tokens(), kvflash_scores_)) {
        return;  // scorer failure -> keep LRU behavior this round
    }
    kvflash_pager_.score_hook = [this](int c) {
        return c < (int)kvflash_scores_.size() ? kvflash_scores_[c] : 1e30f;
    };
    const int events = kvflash_pager_.reselect();
    if (events > 0) {
        std::fprintf(stderr, "[kvflash] reselect @gen=%d: %d page events "
                     "(resident %d/%d blocks)\n",
                     generated, events, kvflash_pager_.resident_blocks(),
                     kvflash_tokens_ / kvflash_pager_.chunk_tokens());
    }
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
    const int _min_floor = dflash_min_tokens_floor();
    static const int _repeat_guard = []{
        const int explicit_guard =
            env_int_or_default("DFLASH_DEGENERATE_RUN_TOKENS", -1);
        if (explicit_guard >= 0) return explicit_guard;
        return dflash_min_tokens_floor() > 0 ? 32 : 0;
    }();

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
        if (kvflash_active()) kvflash_history_.push_back(first_tok);
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

        // kvflash: graph carries a slot-validity mask alongside the
        // step-invariant set_rows write; the FA span clamps to the pool.
        const bool pool = kvflash_active();
        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/committed, /*n_tokens=*/1,
                               /*with_mask=*/pool, /*capture=*/false,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/false,
                               cfg_.kq_stride_pad,
                               should_capture_moe_router(),
                               /*kvflash_mask=*/pool,
                               /*capture_qk=*/pool && kvflash_qk_policy_)) {
            return false;
        }

        // Fill kv_write_rows with this step's cache slot for set_rows:
        // the logical position directly, or its pool slot in kvflash mode.
        if (sg_.kv_write_rows) {
            const int n_head_kv = w_.n_head_kv;
            const int64_t slot = pool ? (int64_t)kvflash_pager_.slot_for(committed)
                                      : (int64_t)committed;
            if (pool && slot < 0) {
                std::fprintf(stderr, "[kvflash] no pool slot at pos %d "
                                     "(pool %d exhausted)\n",
                             committed, kvflash_tokens_);
                set_last_error("kvflash: no evictable pool block");
                return false;
            }
            std::vector<int64_t> row_vals(n_head_kv, slot);
            ggml_backend_tensor_set(sg_.kv_write_rows, row_vals.data(), 0,
                                    sizeof(int64_t) * n_head_kv);
        }
        if (pool) kvflash_upload_mask();

        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) return false;

        after_target_compute(sg_, committed, 1);

        // GPU argmax: read 4 bytes, skip the 970 KB logit D2H. Escape: DFLASH_GPU_ARGMAX=0.
        static const bool kGpuArgmaxAR = []() {
            const char * v = std::getenv("DFLASH_GPU_ARGMAX");
            return v == nullptr || v[0] != '0';
        }();
        int32_t next_tok;
        if (sampler_.needs_logit_processing()) {
#ifdef DFLASH27B_HAVE_GPU_SAMPLER
            // GPU sample straight from the device logits tensor, skipping the
            // full ~vocab-wide D2H copy (the same payoff DFLASH_GPU_ARGMAX gets
            // for greedy). Falls back to the CPU chain on -1.
            int g_tok = -1;
            if (gpu_sampler_enabled() && gpu_sampler_supports(sampler_) &&
                sg_.logits && sg_.logits->data) {
                // Draw the uniform from a copy of the RNG so the real stream is
                // not advanced yet; we only commit that single draw if the GPU
                // path succeeds (below). On fallback the stream is untouched and
                // sample_logits() draws the identical value it would with the
                // GPU disabled, so the token stream is reproducible either way.
                double r = 0.0;
                if (sampler_.temp > 0.0f) {
                    std::mt19937_64 rng_peek = sampler_rng_;
                    std::uniform_real_distribution<double> u(0.0, 1.0);
                    r = u(rng_peek);
                }
                g_tok = geometric_sample_logits_cuda(
                    static_cast<const float *>(sg_.logits->data), vocab, sampler_,
                    out_tokens, r, /*logits_on_device=*/true);
            }
            if (g_tok >= 0) {
                // Commit the single uniform the GPU draw consumed, so the RNG
                // advances by exactly one draw — matching the CPU/GPU-off path.
                if (sampler_.temp > 0.0f) {
                    std::uniform_real_distribution<double> u(0.0, 1.0);
                    (void)u(sampler_rng_);
                }
                next_tok = g_tok;
            } else
#endif
            {
                ggml_backend_tensor_get(sg_.logits, logits_buf.data(), 0,
                                        sizeof(float) * vocab);
                next_tok = sample_logits(logits_buf.data(), vocab, sampler_,
                                          out_tokens, sampler_rng_);
            }
        } else if (kGpuArgmaxAR && sg_.argmax_tokens) {
            int32_t tok_i = 0;
            ggml_backend_tensor_get(sg_.argmax_tokens, &tok_i, 0, sizeof(int32_t));
            next_tok = tok_i;
        } else {
            ggml_backend_tensor_get(sg_.logits, logits_buf.data(), 0,
                                    sizeof(float) * vocab);
            next_tok = 0;
            float best = logits_buf[0];
            for (int j = 1; j < vocab; j++) {
                if (logits_buf[j] > best) { best = logits_buf[j]; next_tok = j; }
            }
        }

        // MIN_TOKENS_BEFORE_EOS (env DFLASH_MIN_TOKENS, default 0=off): if the
        // model tries to stop before producing N tokens in this decode call,
        // suppress EOS and take the best NON-eos token instead. Targets the Q4
        // 'preamble then stop, no tool_call' agentic stall. Env-gated so the
        // default production lane is byte-for-byte unchanged.
        {
            if (_min_floor > 0 && (int)out_tokens.size() < _min_floor && IS_EOS_TOK(next_tok, w_)) {
                int alt = -1; float altbest = -1e30f;
                for (int v = 0; v < vocab; v++) {
                    if (IS_EOS_TOK(v, w_)) continue;
                    if (logits_buf[v] > altbest) { altbest = logits_buf[v]; alt = v; }
                }
                if (alt >= 0) {
                    // Debug-only diagnostic: writes happen exclusively when the
                    // operator opts into DFLASH_MIN_TOKENS, so the default
                    // production lane never touches /tmp/dflash_floor.log.
                    // Bound the local evidence file before appending.
                    FILE* _d = open_dflash_floor_log();
                    if (_d) { std::fprintf(_d, "[floor] eos@%d -> alt=%d\n", (int)out_tokens.size(), alt); std::fclose(_d); }
                    next_tok = alt;
                }
            }
        }

        maybe_force_close(next_tok, committed);

        out_tokens.push_back(next_tok);
        io.emit(next_tok);
        committed++;
        cache_.cur_pos = committed;
        if (pool) {
            kvflash_history_.push_back(next_tok);
            if (kvflash_qk_policy_) kvflash_qk_pool_to(committed);
            kvflash_maybe_reselect((int)(out_tokens.size() - out_tokens_at_entry));
        }
        if (io.cancelled) break;

        if (IS_EOS_TOK(next_tok, w_)) break;

        if (_repeat_guard > 0 && (int)out_tokens.size() >= _repeat_guard) {
            int run = 1;
            for (int j = (int)out_tokens.size() - 2; j >= 0; --j) {
                if (out_tokens[j] != next_tok) break;
                run++;
            }
            if (run >= _repeat_guard) {
                std::fprintf(stderr,
                    "[degenerate-decode] token %d repeated %d times - "
                    "breaking AR loop at committed=%d\n",
                    next_tok, run, committed);
                if (degenerate_close_out) *degenerate_close_out = true;
                break;
            }
        }

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
                                    const std::vector<int32_t> * stall_tool_prefix_tokens,
                                    const std::vector<int32_t> * stall_action_suffix_tokens,
                                    const std::vector<int32_t> * stall_skip_tokens,
                                    const BudgetHook * budget_hook,
                                    bool * forced_close_out,
                                    bool * degenerate_close_out) {
    out_accept_rate = 0.0f;
    out_spec_ran    = false;
    // [TAG_DRAFT_KV] the drafter ring persists across requests but its rows
    // belong to the previous conversation; start every request empty (the
    // first begin_step bulk-appends the live window from the feature mirror).
    if (draft_kv_.gf) draft_kv_reset(draft_kv_);
    const int hidden = w_.n_embd;

    // First token: use the argmax that do_prefill already sampled and stored.
    // Reading sg_.argmax_tokens with a computed offset is fragile: when
    // restore_and_generate calls do_prefill with kv_offset != 0, committed
    // reflects total KV position but the last chunk size was
    // delta.size() % ubatch, making (committed % 512) wrong and causing an
    // out-of-bounds tensor read.  cache_.last_tok is always correct.
    int32_t last_tok = cache_.last_tok;

    // Sampled-verify: spec decode with an active sampler. Each chain
    // position is verified against a token drawn from the target's own
    // sampler chain instead of its argmax, so every committed token is an
    // exact target sample — the output distribution is identical to AR
    // sampling. Acceptance drops vs greedy but stays far above the AR
    // floor. Opt in with DFLASH_SAMPLED_VERIFY=1; without it, sampling
    // requests fall back to AR decode (zero behavior change by default).
    static const bool kSampledVerify = []() {
        const char * e = std::getenv("DFLASH_SAMPLED_VERIFY");
        return e != nullptr && std::string(e) == "1";
    }();
    // Sampled-verify additionally requires full attention in the verify
    // path. With a finite --fa-window the verify batch applies one
    // window-start to the whole batch (unlike the AR step graph, which is
    // hardcoded to full attention): the argmax stays robust, so greedy
    // verification is unaffected, but the logit TAIL drifts at long
    // context and top-k sampling draws degenerate tokens from it
    // (reproduced at 24K: 0/12 tool calls with fa-window 2048, 4/4 with 0).
    const bool sampled_verify = kSampledVerify &&
        sampler_.needs_logit_processing() &&
        cfg_.fa_window == 0;

    // Check if we can use speculative decode:
    // - draft model loaded and not parked
    // - feature mirror initialized
    // - greedy decoding (no logit processing) — spec decode uses argmax verification
    // - sampled-verify enabled — allows spec decode with active sampling
    // - kvflash: verify_batch is slot-mapped (Qwen35DFlashTarget pooled
    //   path), and that covers --ddtree too: in the daemon, ddtree_mode
    //   configures larger verify intermediates + fast_rollback, whose
    //   snapshot_kv/restore_kv only touch DeltaNet/conv state (pool-
    //   neutral); generation runs this same chain loop either way. The
    //   tree-verify graphs exist only in the test harness (test_dflash).
    const bool can_spec = cfg_.draft_path
        && !draft_parked_
        && (cfg_.remote_draft.enabled()
            ? remote_draft_.active()
            : feature_mirror_.target_feat != nullptr)
        && (!sampler_.needs_logit_processing() || sampled_verify);

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

    // Sampled-verify: cache_.last_tok is do_prefill's argmax, and the spec
    // loop commits it verbatim as the first generated token. The first token
    // is the highest-entropy decision of the whole generation (e.g. "answer
    // with text" vs "open a tool call"), so it must be sampled like every
    // other committed token — mirror do_ar_decode's first-token sampling.
    if (sampled_verify && out_tokens.empty() && prefill_last_logits_valid_) {
        std::vector<float> first_logits(w_.n_vocab);
        ggml_backend_tensor_get(sg_.logits, first_logits.data(),
                                prefill_last_logits_offset_,
                                sizeof(float) * (size_t)w_.n_vocab);
        if (std::getenv("DFLASH_SV_DEBUG")) {
            int am = 0; float best = first_logits[0];
            for (int v = 1; v < w_.n_vocab; v++)
                if (first_logits[v] > best) { best = first_logits[v]; am = v; }
            std::fprintf(stderr,
                "[sv-debug] first-token: logits_argmax=%d cache_last_tok=%d "
                "(match=%d) top_logit=%.3f\n",
                am, cache_.last_tok, am == cache_.last_tok, best);
        }
        last_tok = sample_logits(first_logits.data(), w_.n_vocab, sampler_,
                                 out_tokens, sampler_rng_);
        cache_.last_tok = last_tok;
    }

    const int _min_floor = dflash_min_tokens_floor();

    // ── DFlash spec-decode: draft → verify → accept → replay ──────────

    DFlashTarget * target = dflash_target();
    const bool use_remote_draft = cfg_.remote_draft.enabled() && remote_draft_.active();
    const int q_len = dw_.block_size > 0 ? dw_.block_size : DFLASH27B_DRAFT_BLOCK_SIZE;
    const int max_verify_tokens = cfg_.ddtree_mode
        ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
        : dw_.block_size;
    if ((cfg_.fast_rollback || cfg_.ddtree_mode) && !cache_.rollback_ctx) {
        if (!migrate_prefill_cache(w_, cfg_.device.max_ctx,
                                   max_verify_tokens,
                                   target_backend_, cache_)) {
            std::fprintf(stderr, "spec-decode: rollback cache migration failed: %s\n",
                         dflash27b_last_error());
            return false;
        }
    }

    StepGraph draft_sg;

    std::vector<float>   noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<float>   verify_logits;   // sampled-verify: [q_len x vocab]
    std::vector<int32_t> verify_history;  // sampled-verify: penalty history
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<float>   local_hidden;

    int n_generated     = 0;
    int n_draft_steps   = 0;
    int n_accept_sum    = 0;
    int n_hint_proposed = 0;
    int n_hint_accepted = 0;
    int target_forwards = 0;
    const ChainRollbackPolicy rollback_policy = resolve_chain_rollback_policy();
    RollbackDiag rollback_diag;

    auto log_target_forward_stats = [&]() {
        std::fprintf(stderr, "[spec-decode] target_forwards=%d forwards_per_token=%.6f forwards_per_step=%.3f\n",
                     target_forwards,
                     n_generated > 0 ? (double)target_forwards / n_generated : 0.0,
                     n_draft_steps > 0 ? (double)target_forwards / n_draft_steps : 0.0);
        rollback_diag.print(rollback_policy, stderr);
    };

    // kvflash: an in-pool prompt prefills contiguously without registering
    // chunks in the pager. Map the prefix now so the pager stays consistent
    // with the tree path's direct writes and hands off cleanly to slot-mapped
    // paging once the context outgrows the pool. (A >pool prompt already paged
    // during prefill, leaving the pager non-identity, so this is skipped.)
    if (kvflash_active() && kvflash_pager_.is_identity()) {
        (void)kvflash_pager_.alloc_span(0, committed);
    }

    auto t_dec0 = std::chrono::steady_clock::now();

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        // Budget hook: no tail-off here. The close-token injection fires
        // during the emit phase (step 8) after acceptance+replay, mirroring
        // the existing floor_to_ar pattern. This keeps spec-decode active
        // through the close boundary instead of tearing down to AR early.

        if (last_tok < 0 && !out_tokens.empty()) {
            std::fprintf(stderr,
                "[spec-decode] invalid draft seed %d after %d emitted tokens; "
                "switching to AR\n",
                last_tok, (int)out_tokens.size());
            step_graph_destroy(draft_sg);
            cache_.last_tok = out_tokens.back();
            const int ar_n_gen = n_gen - n_generated;
            if (ar_n_gen <= 0) {
                log_target_forward_stats();
                io.emit(-1);
                return true;
            }
            BudgetHook tail_hook = budget_hook ? *budget_hook : BudgetHook{};
            bool ok = do_ar_decode(committed, ar_n_gen, out_tokens, io,
                                    tail_hook, forced_close_out,
                                    degenerate_close_out);
            log_target_forward_stats();
            io.emit(-1);
            return ok;
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
            // [TAG_DRAFT_KV] ring-cached drafter context KV: append newly
            // committed rows instead of re-encoding the whole feature window.
            static const bool draft_kv_on = []() {
                const char * e = std::getenv("DFLASH_DRAFT_KV");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            bool use_draft_kv = draft_kv_on && feature_mirror_.target_feat != nullptr;
            if (use_draft_kv && draft_kv_.gf &&
                draft_kv_.built_for != (const void *)&dw_) {
                draft_kv_free(draft_kv_);
            }
            if (use_draft_kv && !draft_kv_.gf) {
                const int kv_cap = std::min(ring_cap,
                    std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max));
                if (!draft_kv_init(draft_kv_, dw_, draft_backend_, kv_cap, nullptr)) {
                    draft_kv_free(draft_kv_);
                    use_draft_kv = false;
                    std::fprintf(stderr,
                        "spec-decode: draft-kv init failed; using legacy draft path\n");
                }
            }
            if (use_draft_kv) {
                if (!draft_kv_begin_step(draft_kv_, dw_, draft_backend_,
                                         feature_mirror_, committed)) {
                    std::fprintf(stderr, "spec-decode: draft-kv step prep failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                ggml_backend_tensor_set(draft_kv_.inp_embed, noise_embed.data(), 0,
                                        sizeof(float) * noise_embed.size());
                if (ggml_backend_graph_compute(draft_backend_, draft_kv_.gf) !=
                    GGML_STATUS_SUCCESS) {
                    std::fprintf(stderr, "spec-decode: draft-kv compute failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                local_hidden.resize((size_t)hidden * q_len);
                ggml_backend_tensor_get(draft_kv_.hidden_states, local_hidden.data(), 0,
                                        sizeof(float) * local_hidden.size());
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
        }

        // 3. Project draft hidden → token IDs via target LM head
        if (!target->project_hidden_to_tokens(local_hidden.data(), q_len, draft_tok)) {
            std::fprintf(stderr, "spec-decode: projection failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        draft_tok[0] = last_tok;

        // ── DDTree tree-structured verify ────────────────────────────────
        // When --ddtree is on and the target supports tree verify, build a
        // draft tree from per-position top-K, verify all nodes in one
        // ancestor-masked target forward, walk the verified path, then roll
        // recurrent/KV state forward to the accepted path. Higher acceptance
        // per step than chain verify. Local draft only. On any failure we
        // fall through is unsafe (draft graph already built for chain), so we
        // bail to false.
        // The tree path handles plain generation only. Requests using features
        // the tree branch does not implement — thinking-budget forced close,
        // tool-call hint injection, stall recovery, or the min-tokens floor
        // region — fall through to the chain verify path below, which handles
        // them. (Wiring these into the tree path is a follow-up.)
        const bool tree_special_inactive =
            !(budget_hook && !budget_hook->close_token_ids.empty()) &&
            !(hint_tokens && n_generated < (int)hint_tokens->size()) &&
            stall_tool_prefix_tokens == nullptr &&
            (int)out_tokens.size() >= _min_floor;
        // kvflash: only take the non-paged tree path while the read prefix
        // [0, committed) is identity-resident and the write span (+FA padding)
        // fits the resident pool. Full-attention only (windowed FA padding could
        // index past the pool). Otherwise the slot-mapped chain verify handles
        // it. Non-kvflash is unaffected.
        const bool kvflash_tree_ok =
            !kvflash_active() ||
            (cfg_.fa_window == 0 &&
             committed + cfg_.ddtree_budget + 1 + cfg_.kq_stride_pad <= kvflash_tokens_ &&
             kvflash_pager_.identity_prefix_covers(committed));
        if (cfg_.ddtree_mode && target->supports_tree_verify() && kvflash_tree_ok &&
            !use_remote_draft && q_len > 1 && tree_special_inactive) {
            const int L = q_len - 1;
            const int K = (cfg_.ddtree_budget > L) ? 8 : 1;
            std::vector<float>   top_lp;
            std::vector<int32_t> top_ids;
            if (!target->project_hidden_to_topk(local_hidden.data(), q_len, K,
                                                cfg_.ddtree_temp, top_lp, top_ids)) {
                std::fprintf(stderr, "spec-decode: ddtree topk projection failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            // Tree depth L draws from draft rows 1..q_len-1 (row 0 = the seed).
            // Known limitation: branch descendants beyond depth 1 still come
            // from one spine-conditioned block-draft forward, so a confident
            // draft may not beat the chain.
            DDTree tree = build_ddtree(top_lp.data() + (size_t)K, top_ids.data() + (size_t)K,
                                       L, K, cfg_.ddtree_budget, cfg_.ddtree_chain_seed);
            const int N = cfg_.ddtree_budget + 1;   // fixed alloc width

            std::vector<int32_t> flat_tokens((size_t)N, 0);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; i++) flat_tokens[1 + i] = tree.token_ids[i];

            if (!sampled_verify && !target->snapshot_kv()) {
                step_graph_destroy(draft_sg);
                return false;
            }

            std::vector<int32_t> posterior;
            std::vector<float>   node_logits;
            if (!target->verify_tree(committed, tree, flat_tokens, N, posterior,
                                     sampled_verify ? &node_logits : nullptr)) {
                std::fprintf(stderr, "spec-decode: verify_tree failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            target_forwards++;

            int next_token = -1, bonus_node = 0;
            std::vector<int> accepted;
            if (!sampled_verify) {
                accepted =
                    follow_verified_tree(tree, posterior.data(), next_token, &bonus_node);
            } else {
                accepted.reserve((size_t)tree.n_nodes + 1);
                accepted.push_back(0);
                std::vector<int32_t> hist = out_tokens;
                hist.push_back(last_tok);
                const int vocab = w_.n_vocab;
                int cur = 0;
                int stok = sample_logits(node_logits.data() + (size_t)cur * vocab,
                                         vocab, sampler_, hist, sampler_rng_);
                while (true) {
                    auto it = tree.child_maps[cur].find(stok);
                    if (it == tree.child_maps[cur].end()) break;
                    cur = it->second;
                    accepted.push_back(cur);
                    hist.push_back(stok);
                    stok = sample_logits(node_logits.data() + (size_t)cur * vocab,
                                         vocab, sampler_, hist, sampler_rng_);
                }
                next_token = stok;
            }

            int accepted_n = (int)accepted.size();        // root + accepted children
            if (accepted_n > need_commit_budget) accepted_n = need_commit_budget;
            if (accepted_n <= 0) { step_graph_destroy(draft_sg); break; }

            // Emit the accepted path: slot 0 = last_tok (pending from prev iter),
            // each subsequent accepted node = its tree token.
            bool hit_eos = false;
            int accepted_emitted = 0;
            for (int i = 0; i < accepted_n; i++) {
                const int dfs = accepted[i];
                const int32_t tok = (dfs == 0) ? last_tok : tree.token_ids[dfs - 1];
                out_tokens.push_back(tok);
                io.emit(tok);
                accepted_emitted++;
                if (io.cancelled) { hit_eos = true; break; }
                if (target->is_eos(tok)) { hit_eos = true; break; }
            }

            // Telemetry: accepted children (exclude the always-committed root).
            n_accept_sum += std::max(0, accepted_emitted - 1);

            if (accepted_emitted <= 0) { step_graph_destroy(draft_sg); break; }

            if (!sampled_verify) {
                const int root_last_tok = last_tok;
                constexpr int kFastRollbackThreshold = 5;
                const bool use_tree_fast_rollback =
                    target->supports_fast_rollback() &&
                    accepted_emitted >= kFastRollbackThreshold;

                if (use_tree_fast_rollback) {
                    // Fast greedy production path: restore to the accepted path
                    // from tree captures and defer the bonus as next step's root.
                    std::vector<int> accepted_committed(accepted.begin(),
                                                        accepted.begin() + accepted_emitted);
                    if (!target->rollback_to_tree(committed, tree, accepted_committed)) {
                        std::fprintf(stderr, "spec-decode: rollback_to_tree failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                    last_tok = next_token;

                    if (feature_mirror_.target_feat && !draft_parked_) {
                        draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                                        feature_mirror_, committed, accepted_emitted);
                    }

                    committed   += accepted_emitted;
                    cache_.cur_pos = committed;
                    n_generated += accepted_emitted;
                    n_draft_steps++;
                    if (hit_eos || io.cancelled || n_generated >= n_gen ||
                        last_tok < 0 || target->is_eos(last_tok)) {
                        break;
                    }
                    continue;
                }

                // Low-accept greedy path: mirror the chain's exact replay so the
                // next step starts from replayed F32 recurrent state. The accepted
                // path has already been emitted above; only emit the bonus here.
                int total_emitted = accepted_emitted;
                const bool can_commit_bonus =
                    !hit_eos && !io.cancelled && next_token >= 0 &&
                    total_emitted < need_commit_budget;

                std::vector<int32_t> replay_batch;
                replay_batch.reserve((size_t)accepted_emitted + (can_commit_bonus ? 1 : 0));
                for (int i = 0; i < accepted_emitted; i++) {
                    const int dfs = accepted[i];
                    replay_batch.push_back((dfs == 0) ? root_last_tok : tree.token_ids[dfs - 1]);
                }
                if (can_commit_bonus) replay_batch.push_back(next_token);

                if (!target->restore_kv()) {
                    step_graph_destroy(draft_sg);
                    return false;
                }
                int replay_last_tok = -1;
                if (!target->verify_batch(replay_batch, committed, replay_last_tok, nullptr)) {
                    std::fprintf(stderr, "spec-decode: tree replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                target_forwards++;

                if (can_commit_bonus) {
                    out_tokens.push_back(next_token);
                    io.emit(next_token);
                    total_emitted++;
                    if (io.cancelled) {
                        hit_eos = true;
                    } else if (target->is_eos(next_token)) {
                        hit_eos = true;
                    }
                }

                last_tok = replay_last_tok;
                if (feature_mirror_.target_feat && !draft_parked_) {
                    draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                                    feature_mirror_, committed, total_emitted);
                }

                committed   += total_emitted;
                cache_.cur_pos = committed;
                n_generated += total_emitted;
                n_draft_steps++;
                if (hit_eos || io.cancelled || n_generated >= n_gen ||
                    last_tok < 0 || target->is_eos(last_tok)) {
                    break;
                }
                continue;
            }

            // Sampled path keeps the distribution-preserving bonus replay but
            // still needs tree rollback for the accepted prefix first.
            std::vector<int> accepted_committed(accepted.begin(),
                                                accepted.begin() + accepted_emitted);
            if (!target->rollback_to_tree(committed, tree, accepted_committed)) {
                std::fprintf(stderr, "spec-decode: rollback_to_tree failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }

            int total_emitted = accepted_emitted;
            int bonus_last_tok = -1;
            std::vector<float> bonus_logits;
            const bool can_commit_bonus =
                !hit_eos && !io.cancelled && next_token >= 0 &&
                total_emitted < need_commit_budget;
            if (can_commit_bonus) {
                const int bonus_pos = committed + total_emitted;
                std::vector<int32_t> bonus_vec(1, next_token);
                if (!target->verify_batch(bonus_vec, bonus_pos, bonus_last_tok, nullptr)) {
                    std::fprintf(stderr, "spec-decode: tree bonus replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                target_forwards++;
                if (!target->read_verify_logits(1, bonus_logits)) {
                    std::fprintf(stderr, "spec-decode: tree bonus logits read failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                if (bonus_logits.empty()) {
                    std::fprintf(stderr, "spec-decode: tree bonus logits empty\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }

                out_tokens.push_back(next_token);
                io.emit(next_token);
                total_emitted++;
                if (io.cancelled) {
                    hit_eos = true;
                } else if (target->is_eos(next_token)) {
                    hit_eos = true;
                }

                const int vocab_v = (int)bonus_logits.size();
                last_tok = sample_logits(bonus_logits.data(), vocab_v,
                                         sampler_, out_tokens, sampler_rng_);
            } else {
                last_tok = next_token;
            }

            // Sampled path commits the bonus in-step, so sync accepted+bonus.
            if (feature_mirror_.target_feat && !draft_parked_) {
                draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                                feature_mirror_, committed, total_emitted);
            }

            committed   += total_emitted;
            cache_.cur_pos = committed;
            n_generated += total_emitted;
            n_draft_steps++;
            if (hit_eos || io.cancelled || n_generated >= n_gen || last_tok < 0) {
                break;
            }
            continue;
        }

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

        // Notify observer with draft tokens for this step.
        if (io.observer) {
            io.observer("draft", draft_tok);
        }

        // 4. Verify: snapshot KV, run target forward over draft tokens
        if (!target->snapshot_kv()) {
            step_graph_destroy(draft_sg);
            return false;
        }

        int verify_last_tok = -1;
        if (!target->verify_batch(draft_tok, committed, verify_last_tok, &target_tok,
                                   /*capture_ssm_intermediates=*/true)) {
            std::fprintf(stderr, "spec-decode: verify failed\n");
            target->restore_kv();
            step_graph_destroy(draft_sg);
            return false;
        }
        target_forwards++;

        // 5. Acceptance. Greedy: longest matching prefix between draft and
        // target argmax. Sampled-verify: walk the chain drawing each next
        // token from the target's sampler chain; accept while the draft
        // guessed the drawn token, and the first mismatch becomes the bonus
        // token (it is already a valid target sample at that position).
        int accept_n = 1;
        int bonus_tok = -1;
        if (sampled_verify) {
            if (!target->read_verify_logits(q_len, verify_logits)) {
                std::fprintf(stderr, "spec-decode: verify logits read failed\n");
                target->restore_kv();
                step_graph_destroy(draft_sg);
                return false;
            }
            const int vocab_v = (int)(verify_logits.size() / (size_t)q_len);
            static const bool kSvDebug = []() {
                const char * e = std::getenv("DFLASH_SV_DEBUG");
                return e != nullptr && std::string(e) == "1";
            }();
            if (kSvDebug) {
                // Row-alignment check: CPU argmax over each bulk-read row must
                // equal the GPU argmax (target_tok). Divergence = misaligned
                // or stale bulk read.
                for (int i = 0; i < q_len; i++) {
                    const float * row = verify_logits.data() + (size_t)i * vocab_v;
                    int am = 0; float best = row[0];
                    for (int v = 1; v < vocab_v; v++)
                        if (row[v] > best) { best = row[v]; am = v; }
                    if (am != target_tok[i]) {
                        std::fprintf(stderr,
                            "[sv-debug] ROW MISMATCH i=%d cpu_argmax=%d (%.3f) "
                            "gpu_argmax=%d (%.3f) vocab_v=%d\n",
                            i, am, best, target_tok[i],
                            target_tok[i] < vocab_v ? row[target_tok[i]] : -999.0f,
                            vocab_v);
                        break;
                    }
                }
            }
            // Penalty history must match AR exactly: when AR samples the
            // token after X, X is already in out_tokens. The seed
            // draft_tok[0] is committed by this step's replay but not yet
            // in out_tokens, so add it before the walk — without it the
            // repetition penalty never sees the seed and the sampled
            // distribution drifts from AR whenever penalties are active.
            verify_history = out_tokens;
            verify_history.push_back(draft_tok[0]);
            bool mismatched = false;
            for (int i = 0; i < q_len - 1; i++) {
                const int s = sample_logits(
                    verify_logits.data() + (size_t)i * vocab_v, vocab_v,
                    sampler_, verify_history, sampler_rng_);
                if (kSvDebug && n_draft_steps < 3 && i < 4) {
                    std::fprintf(stderr,
                        "[sv-debug] step=%d pos=%d seed/draft0=%d draft=%d "
                        "sampled=%d\n",
                        n_draft_steps, i, draft_tok[0], draft_tok[i + 1], s);
                }
                if (draft_tok[i + 1] == s) {
                    accept_n++;
                    verify_history.push_back(s);
                } else {
                    bonus_tok = s;
                    mismatched = true;
                    break;
                }
            }
            (void)mismatched;
        } else {
            for (int i = 0; i < q_len - 1; i++) {
                if (draft_tok[i + 1] == target_tok[i]) accept_n++;
                else break;
            }
            bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        }
        // Track hint acceptance telemetry.
        if (hint_fill > 0) {
            n_hint_proposed += hint_fill;
            n_hint_accepted += std::min(hint_fill, accept_n - 1);
        }
        int commit_n  = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        // 6. Fix state: adaptive fast-rollback vs legacy replay.
        //    Fast-rollback (implicit bonus, skip replay) is profitable when
        //    accept_n is large enough that skipping the replay saves more compute
        //    than the cost of deferring the bonus to the next step. The default
        //    threshold is 5; exact F32 checkpoints may opt in to a lower value.
        rollback_diag.record_accept(accept_n);
        const bool use_fast_rollback =
            target->supports_fast_rollback() &&
            (accept_n >= rollback_policy.fast_rollback_threshold);

        int replay_last_tok = -1;
        bool fast_rolled_back = false;
        if (use_fast_rollback) {
            // Fast rollback: restore SSM from captured intermediates, skip replay.
            // Implicit bonus: target_tok[commit_n-1] seeds next draft as draft_tok[0],
            // always accepted on next step.
            bonus_tok = -1;
            // Respect the generation budget: accept_n may exceed the remaining
            // budget (need_commit_budget), so committing accept_n would emit
            // more tokens than requested. commit_n was already clamped above.
            commit_n = std::min(accept_n, need_commit_budget);
            if (target->rollback_to(committed, commit_n)) {
                replay_last_tok = target_tok[commit_n - 1];
                fast_rolled_back = true;
                rollback_diag.record_fast_rollback(accept_n);
            } else {
                // Rollback failed (CUDA error / unsupported state type). The
                // pre-verify snapshot is still valid, so degrade to the legacy
                // restore+replay path below instead of aborting the request.
                std::fprintf(stderr, "spec-decode: rollback_to failed; "
                                     "falling back to restore+replay\n");
                rollback_diag.record_failed_fallback();
            }
        }
        if (!fast_rolled_back) {
            rollback_diag.record_legacy_replay();
            // Legacy replay: restore SSM snapshot, replay accepted + bonus tokens.
            // (When falling back from fast-rollback, bonus_tok is -1 and commit_n
            //  is the budget-clamped accepted count.)
            if (!target->restore_kv()) {
                step_graph_destroy(draft_sg);
                return false;
            }
            std::vector<int32_t> replay_batch((size_t)commit_n);
            for (int i = 0; i < commit_n; i++) {
                replay_batch[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
            }
            if (!target->verify_batch(replay_batch, committed, replay_last_tok, nullptr)) {
                std::fprintf(stderr, "spec-decode: replay failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            target_forwards++;
        }

        // Build replay_tok for emitting committed tokens.
        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }

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
        bool floor_to_ar = false;
        bool inject_tool_prefix = false;
        bool budget_close_fired = false;
        constexpr size_t kActionSuffixLookback = 16;
        constexpr size_t kSkipSequenceLookback = 64;
        int emitted = 0;
        for (int i = 0; i < commit_n; i++) {
            if (_min_floor > 0 && (int)out_tokens.size() < _min_floor &&
                IS_EOS_TOK(replay_tok[i], w_)) {
                // Action preambles often end as "I'll check:\n\n" before EOS.
                // Tokenization makes the colon several tokens back, so keep a
                // modest trailing window while still requiring a recent action
                // suffix token and no nearby completion phrase.
                const bool can_inject_tool =
                    stall_tool_prefix_tokens && !stall_tool_prefix_tokens->empty() &&
                    stall_action_suffix_tokens && !stall_action_suffix_tokens->empty() &&
                    tokens_have_recent_any(out_tokens, *stall_action_suffix_tokens,
                                           kActionSuffixLookback) &&
                    !(stall_skip_tokens &&
                      tokens_contain_recent_sequence(out_tokens,
                                                     *stall_skip_tokens,
                                                     kSkipSequenceLookback));
                if (can_inject_tool) {
                    // Debug-only diagnostic, same DFLASH_MIN_TOKENS gating as the
                    // AR-path floor log above; silent in the default lane.
                    FILE* _d = open_dflash_floor_log();
                    if (_d) {
                        std::fprintf(_d,
                            "[spec-tool-floor] eos@%d committed=%d emitted=%d prefix=%zu -> ar\n",
                            (int)out_tokens.size(), committed, emitted,
                            stall_tool_prefix_tokens->size());
                        std::fclose(_d);
                    }
                    floor_to_ar = true;
                    inject_tool_prefix = true;
                    break;
                }
            }
            // Budget hook: check if remaining budget has hit the force-close
            // threshold. Override this token with close_token_ids[0] and stop
            // emitting — AR handles the rest via maybe_force_close.
            if (budget_hook && !budget_hook->close_token_ids.empty() &&
                !budget_close_fired)
            {
                const int generated_now = n_generated + emitted;
                int remaining = n_gen - generated_now;
                if (remaining <= budget_hook->hard_limit_remaining) {
                    int32_t first_close = budget_hook->close_token_ids.front();
                    if (replay_tok[i] == first_close) {
                        // Model self-closed at the boundary; consume as
                        // start of close sequence (no override needed).
                        budget_close_fired = true;
                    } else {
                        // Force-close: override sampled token with close[0].
                        replay_tok[i] = first_close;
                        budget_close_fired = true;
                        if (forced_close_out) *forced_close_out = true;
                    }
                    std::fprintf(stderr,
                        "[budget-hook] spec-decode close at committed=%d/%d "
                        "(remaining=%d <= hard_limit=%d)\n",
                        committed + emitted, n_gen, remaining,
                        budget_hook->hard_limit_remaining);
                }
            }
            out_tokens.push_back(replay_tok[i]);
            io.emit(replay_tok[i]);
            emitted++;
            if (io.cancelled) break;
            if (budget_close_fired) break;
            if (IS_EOS_TOK(replay_tok[i], w_)) { hit_eos = true; break; }
        }
        int injected = 0;
        if (floor_to_ar) {
            if (!target->restore_kv()) {
                step_graph_destroy(draft_sg);
                return false;
            }
            cache_.cur_pos = committed;
            if (emitted > 0) {
                std::vector<int32_t> replay_prefix(replay_tok.begin(),
                                                   replay_tok.begin() + emitted);
                int prefix_last_tok = -1;
                if (!target->verify_batch(replay_prefix, committed,
                                          prefix_last_tok, nullptr)) {
                    std::fprintf(stderr, "spec-decode: floor prefix replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                target_forwards++;
            }
            committed += emitted;
            cache_.cur_pos = committed;
            if (inject_tool_prefix) {
                int tool_prefix_last_tok = -1;
                if (!target->verify_batch(*stall_tool_prefix_tokens, committed,
                                          tool_prefix_last_tok, nullptr)) {
                    std::fprintf(stderr, "spec-decode: tool prefix replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                target_forwards++;
                for (int32_t tok : *stall_tool_prefix_tokens) {
                    out_tokens.push_back(tok);
                    io.emit(tok);
                }
                injected = (int)stall_tool_prefix_tokens->size();
                committed += injected;
                cache_.cur_pos = committed;
            }
        } else {
            // Normal (non-floor) path: carry the replay's last token into the
            // next draft step. The floor_to_ar path never reaches the next
            // iteration — it sets cache_.last_tok directly below and returns —
            // so last_tok is intentionally left untouched when flooring.
            //
            // Sampled-verify: the seed is committed as-is by the next step
            // (draft_tok[0]), so it must itself be a sample from the target
            // distribution. replay_last_tok is the argmax — seeding with it
            // injects one greedy token per step, which biases the output and
            // locks long generations into repetition loops.
            if (sampled_verify && !replay_tok.empty() &&
                target->read_verify_logits((int)replay_tok.size(), verify_logits)) {
                const int vocab_v =
                    (int)(verify_logits.size() / replay_tok.size());
                last_tok = sample_logits(
                    verify_logits.data() +
                        (replay_tok.size() - 1) * (size_t)vocab_v,
                    vocab_v, sampler_, out_tokens, sampler_rng_);
            } else {
                last_tok = replay_last_tok;
            }
            committed += emitted;
        }
        cache_.cur_pos = committed;
        n_generated += emitted + injected;
        n_accept_sum += std::min(accept_n, emitted);
        n_draft_steps++;

        // Notify observer with accepted tokens for this step.
        if (io.observer) {
            io.observer("verify", replay_tok);
        }

        if (io.cancelled) break;
        if (floor_to_ar) {
            step_graph_destroy(draft_sg);
            cache_.last_tok = out_tokens.empty() ? last_tok : out_tokens.back();
            const int total_draft_pos = std::max(1, n_draft_steps * q_len);
            out_accept_rate =
                (float)((double)n_accept_sum / (double)total_draft_pos);
            const int ar_n_gen = n_gen - n_generated;
            if (ar_n_gen <= 0) {
                log_target_forward_stats();
                io.emit(-1);
                return true;
            }
            BudgetHook tail_hook = budget_hook ? *budget_hook : BudgetHook{};
            bool ok = do_ar_decode(committed, ar_n_gen, out_tokens, io,
                                    tail_hook, forced_close_out,
                                    degenerate_close_out);
            log_target_forward_stats();
            io.emit(-1);
            return ok;
        }
        // Budget hook close: close token was emitted during the emit phase.
        // Roll back KV to pre-replay state and replay only the overridden
        // prefix (including the close token) so KV stays consistent with
        // the emitted output before AR takes over.
        if (budget_close_fired) {
            if (!target->restore_kv()) {
                step_graph_destroy(draft_sg);
                return false;
            }
            cache_.cur_pos = committed;
            if (emitted > 0) {
                std::vector<int32_t> replay_prefix(replay_tok.begin(),
                                                   replay_tok.begin() + emitted);
                int prefix_last_tok = -1;
                if (!target->verify_batch(replay_prefix, committed,
                                          prefix_last_tok, nullptr)) {
                    std::fprintf(stderr, "spec-decode: budget-close prefix replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                target_forwards++;
            }
            committed += emitted;
            cache_.cur_pos = committed;
            step_graph_destroy(draft_sg);
            cache_.last_tok = out_tokens.empty() ? last_tok : out_tokens.back();
            const int total_draft_pos = std::max(1, n_draft_steps * q_len);
            out_accept_rate =
                (float)((double)n_accept_sum / (double)total_draft_pos);
            const int ar_n_gen = n_gen - n_generated;
            if (ar_n_gen <= 0) {
                log_target_forward_stats();
                io.emit(-1);
                return true;
            }
            BudgetHook tail_hook = budget_hook ? *budget_hook : BudgetHook{};
            tail_hook.close_token_ids.clear();
            bool ok = do_ar_decode(committed, ar_n_gen, out_tokens, io,
                                    tail_hook, forced_close_out,
                                    degenerate_close_out);
            log_target_forward_stats();
            io.emit(-1);
            return ok;
        }
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
    log_target_forward_stats();
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
