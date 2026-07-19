// LagunaBackend implementation.
//
// Translates the model-specific code from the old monolithic
// run_laguna_daemon() into ModelBackend virtual methods. The protocol
// plumbing (command parsing, sampler, stream-fd) now lives in
// daemon_loop.cpp; this file only owns laguna's forward path, cache,
// snapshots, and pflash compress lifecycle.

#include "laguna_backend.h"
#include "laguna_internal.h"
#include "qwen3/qwen3_kvflash_scorer.h"
#include "dflash27b.h"
#include "common/ddtree.h"
#include "common/domino_head.h"
#include "common/dspark_head.h"
#include "common/dflash_feature_ring.h"
#include "common/dflash_draft_graph.h"
#include "common/prof_env.h"
#include "kv_quant.h"

#include <chrono>
#include "../common/moe_hybrid_types.h"
#include "../common/moe_hybrid_types_impl.h"
#include "../common/moe_hybrid_placement.h"
#include "../common/kvflash_placement.h"
#include "../common/moe_hybrid_ffn_eval.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_swap_manager.h"
#include "../common/moe_routing_collector.h"
#include "common/step_graph.h"

#include "ggml-cuda.h"
#include "../common/adaptive_verify_width.h"
#include "ggml-alloc.h"
#include "common/snapshot_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include "common/gguf_mmap.h"

namespace dflash::common {

namespace {

// Laguna honors only the explicit per-axis --cache-type-k/v overrides.
// The DFLASH27B_KV_F16/_Q4/_TQ3 shorthands are qwen-family toggles and
// must not displace laguna's Q8_0 default (a TQ3_0/Q4_0 KV cache garbles
// laguna output).
static void resolve_laguna_kv_types(const LagunaBackendArgs & args,
                                    ggml_type & k_type,
                                    ggml_type & v_type) {
    k_type = args.kv_type;
    v_type = args.kv_type;
    if (const char * s = std::getenv("DFLASH27B_KV_K")) {
        const ggml_type parsed = dflash::parse_kv_type(s);
        if (parsed == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "[laguna] Unknown KV K type: \"%s\"\n", s);
            std::abort();
        }
        k_type = parsed;
    }
    if (const char * s = std::getenv("DFLASH27B_KV_V")) {
        const ggml_type parsed = dflash::parse_kv_type(s);
        if (parsed == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "[laguna] Unknown KV V type: \"%s\"\n", s);
            std::abort();
        }
        v_type = parsed;
    }
    if (k_type != args.kv_type || v_type != args.kv_type) {
        dflash::validate_kv_pair_or_abort(k_type, v_type, "[laguna]");
        std::fprintf(stderr, "[laguna] KV cache types overridden: K=%s V=%s\n",
                     dflash::kv_type_name(k_type), dflash::kv_type_name(v_type));
        if (k_type == GGML_TYPE_TQ3_0 || v_type == GGML_TYPE_TQ3_0 ||
            k_type == GGML_TYPE_Q4_0  || v_type == GGML_TYPE_Q4_0) {
            std::fprintf(stderr,
                "[laguna] WARNING: tq3_0/q4_0 KV caches are known to GARBLE "
                "laguna output. Use q8_0 (the default) unless you are "
                "debugging quantization itself.\n");
        }
    }
}

static bool laguna_auto_head_major_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("DFLASH_LAGUNA_AUTO_HEAD_MAJOR");
        return !(e && std::string(e) == "0");
    }();
    return enabled;
}

static bool laguna_gpu_argmax_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("DFLASH_LAGUNA_GPU_ARGMAX");
        return e == nullptr || e[0] != '0';
    }();
    return enabled;
}

}  // namespace

static bool laguna_sampled_verify_enabled(const SamplerCfg & sampler, bool do_sample) {
    static const bool kSampledVerify = []() {
        const char * e = std::getenv("DFLASH_SAMPLED_VERIFY");
        return e != nullptr && std::string(e) == "1";
    }();
    return kSampledVerify && do_sample && sampler.needs_logit_processing();
}

static bool laguna_dspark_enabled() {
    static const bool kEnabled = []() {
        const char * e = std::getenv("DFLASH_LAGUNA_DSPARK");
        return e == nullptr || std::string(e) != "0";
    }();
    return kEnabled;
}

static float laguna_dspark_confidence_threshold() {
    static const float kThreshold = []() {
        const char * e = std::getenv("DFLASH_LAGUNA_DSPARK_CONFIDENCE_THRESHOLD");
        if (!e) return 0.0f;
        float threshold = std::atof(e);
        if (threshold < 0.0f) threshold = 0.0f;
        if (threshold > 1.0f) threshold = 1.0f;
        return threshold;
    }();
    return kThreshold;
}

// ── Construction / initialisation ───────────────────────────────────────

LagunaBackend::LagunaBackend(const LagunaBackendArgs & args)
    : args_(args) {}

LagunaBackend::~LagunaBackend() { shutdown(); }

bool LagunaBackend::init() {
    backend_ = ggml_backend_cuda_init(args_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "cuda init failed gpu=%d\n", args_.device.gpu);
        return false;
    }

    snap_backend_ = create_snapshot_backend(backend_);
    if (!snap_backend_) {
        std::fprintf(stderr, "snapshot backend init failed\n");
        ggml_backend_free(backend_); backend_ = nullptr;
        return false;
    }

    if (!args_.draft_path.empty()) {
        if (!load_target_gguf_laguna(args_.target_path, backend_, w_)) {
            std::fprintf(stderr, "[laguna] full load failed: %s\n", dflash27b_last_error());
            ggml_backend_free(backend_); backend_ = nullptr;
            return false;
        }
        hybrid_mode_ = false;
    } else {
        // Dynamic placement (like qwen35moe): partial load first, compute
        // budget, then reload full if all experts fit.
        if (!init_hybrid_mode()) {
            ggml_backend_free(backend_); backend_ = nullptr;
            return false;
        }
    }

    resolve_laguna_kv_types(args_, cache_.kv_k_type, cache_.kv_v_type);
    kvflash_read_config();
    if (laguna_auto_head_major_enabled() &&
        !std::getenv("DFLASH_LAGUNA_KV_HEAD_MAJOR") &&
        kvflash_tokens_ <= 0 &&
        !args_.ddtree_mode) {
        setenv("DFLASH_LAGUNA_KV_HEAD_MAJOR", "1", 0);
        std::fprintf(stderr,
                     "[laguna] auto-enabled head-major KV layout "
                     "(disable with DFLASH_LAGUNA_AUTO_HEAD_MAJOR=0)\n");
    }
    // [TAG_SWA_RING] pooled mode: SWA layers on position rings sized for the
    // sliding window + the largest batch this backend issues. Kill switch:
    // DFLASH_LAGUNA_SWA_RING=0.
    static const bool swa_ring_env = []() {
        const char * e = std::getenv("DFLASH_LAGUNA_SWA_RING");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    const int swa_ring_rows =
        (kvflash_tokens_ > 0 && !hybrid_mode_ && swa_ring_env)
            ? std::max(2048, args_.chunk + w_.sliding_window + 64)
            : 0;
    if (!create_laguna_target_cache(w_, args_.max_ctx, backend_, cache_,
                                    kvflash_tokens_, swa_ring_rows)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w_);
        ggml_backend_free(backend_); backend_ = nullptr;
        return false;
    }
    if (!kvflash_attach()) {
        ggml_backend_free(backend_); backend_ = nullptr;
        return false;
    }

    if (!args_.draft_path.empty() && !load_decode_draft()) {
        std::fprintf(stderr, "[laguna] draft unavailable; speculative decode disabled\n");
    }

    return true;
}

// ── kvflash helpers ─────────────────────────────────────────────────────

// Laguna's pager protections: the trailing sliding_window span (+1 chunk
// for the partially filled head) must stay resident so SWA attention stays
// exact under paging. This drives both the pool floor and the attach config.
KvFlashConfig LagunaBackend::kvflash_config() const {
    KvFlashConfig pc;
    pc.tail_window_chunks =
        std::max(4, (w_.sliding_window + pc.chunk_tokens - 1) / pc.chunk_tokens + 1);
    return pc;
}

void LagunaBackend::kvflash_resolve_drafter() {
    if (std::getenv("DFLASH_KVFLASH")) {
        kvflash_drafter_path_ = kvflash_find_drafter(args_.target_path.c_str());
    }
}

KvFlashAutoBudget LagunaBackend::make_kvflash_budget(int64_t gpu_free) const {
    KvFlashAutoBudget b;
    b.free_bytes      = gpu_free;
    b.bytes_per_token = (int64_t)w_.n_layer * w_.n_head_kv * 2 *
        (int64_t)ggml_row_size(args_.kv_type, w_.head_dim);
    b.reserve_bytes   = (int64_t)(1.5 * 1073741824.0) +
        (kvflash_drafter_path_.empty() ? 0 : (int64_t)(1.7 * 1073741824.0));
    return b;
}

void LagunaBackend::kvflash_read_config() {
    kvflash_resolve_drafter();
    // "auto" sizes from the GPU (weights resident, cache not yet allocated):
    // laguna pools ALL n_layer layers at the configured KV quant.
    size_t gpu_free = 0, gpu_total = 0;
    if (ggml_backend_dev_t dev = ggml_backend_get_device(backend_)) {
        ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
    }
    KvFlashAutoBudget kvf_budget = make_kvflash_budget((int64_t)gpu_free);
    kvflash_tokens_ = kvflash_pool_from_env(args_.max_ctx, kvflash_config(),
                                            kvflash_scorer_expected(),
                                            kvf_budget);
    if (kvflash_tokens_ > 0 && placement_all_hot_full_kv_) {
        std::printf("[laguna][kvflash] disabled: placement all-hot at max_ctx %d, "
                    "pool not needed\n", args_.max_ctx);
        std::fflush(stdout);
        kvflash_tokens_ = 0;
        kvflash_tau_ = 64;
        kvflash_drafter_path_.clear();
    }
    if (kvflash_tokens_ > 0) {
        const char * tau = std::getenv("DFLASH_KVFLASH_TAU");
        kvflash_tau_ = std::max(1, tau ? std::atoi(tau) : 64);
    }
}

// Drafter rescore + repage (FlashMemory tau loop) with the cross-tokenizer
// scorer: laguna ids are detokenized and re-scored through the Qwen3-0.6B
// drafter (relevance is text-level, so the tokenizer gap is bridged by
// re-tokenization). Lazy: the drafter + tokenizers load on the first
// reselect that needs them, never on a request's first tokens.
void LagunaBackend::kvflash_maybe_reselect(const std::vector<int32_t> & history,
                                           int generated) {
    if (!kvflash_active() || kvflash_tau_ <= 0) return;
    const int tau = std::max<int>(kvflash_tau_, (int)(history.size() / 45));
    if (generated % tau != 0) return;
    if (!kvflash_scorer_) {
        if (kvflash_drafter_path_.empty() || kvflash_drafter_failed_) return;
        if (!drafter_loaded_) {
            ggml_backend_synchronize(backend_);
            std::fprintf(stderr, "[kvflash] loading drafter for residency scoring: %s\n",
                         kvflash_drafter_path_.c_str());
            if (!load_drafter(kvflash_drafter_path_, /*gpu_layers=*/999,
                              args_.device.gpu, drafter_ctx_)) {
                std::fprintf(stderr, "[kvflash] drafter load failed (%s); staying on "
                                     "LRU residency\n", dflash27b_last_error());
                kvflash_drafter_failed_ = true;
                return;
            }
            drafter_loaded_ = true;
        }
        kvflash_scorer_ = std::make_unique<KvFlashCrossTokScorer>(
            &drafter_ctx_, args_.target_path, kvflash_drafter_path_);
        std::fprintf(stderr, "[kvflash] cross-tokenizer drafter scorer attached "
                             "(tau=%d)\n", kvflash_tau_);
    }
    if (!kvflash_scorer_->score_chunks(history, kvflash_pager_.chunk_tokens(),
                                       kvflash_scores_)) {
        return;  // scorer failure -> keep LRU behavior this round
    }
    kvflash_pager_.score_hook = [this](int c) {
        return c < (int)kvflash_scores_.size() ? kvflash_scores_[c] : 1e30f;
    };
    const int events = kvflash_pager_.reselect();
    kvflash_pager_.score_hook = nullptr;
    if (events > 0) {
        std::fprintf(stderr, "[kvflash] reselect @gen=%d: %d page events\n",
                     generated, events);
    }
}

bool LagunaBackend::kvflash_attach() {
    if (!kvflash_active()) return true;
    KvFlashConfig pc = kvflash_config();
    pc.pool_tokens = kvflash_tokens_;
    // [TAG_SWA_RING] SWA layers live on position rings outside the pool, so
    // the pager only pages the full-attention layers' K/V.
    std::vector<ggml_tensor *> pool_k = cache_.attn_k;
    std::vector<ggml_tensor *> pool_v = cache_.attn_v;
    if (cache_.swa_ring_rows > 0) {
        pool_k.clear();
        pool_v.clear();
        for (int il = 0; il < w_.n_layer; ++il) {
            if (!laguna_is_full_attn_layer(w_, il)) continue;
            pool_k.push_back(cache_.attn_k[(size_t)il]);
            pool_v.push_back(cache_.attn_v[(size_t)il]);
        }
    }
    if (!kvflash_pager_.attach(pc, pool_k, pool_v)) {
        std::fprintf(stderr, "kvflash: pager attach failed (pool=%d)\n",
                     kvflash_tokens_);
        return false;
    }
    std::printf("[kvflash] resident pool %d tokens (logical max_ctx %d), "
                "policy=%s, swa_tail=%d chunks\n",
                kvflash_tokens_, args_.max_ctx,
                !kvflash_drafter_path_.empty()
                    ? "drafter/cross-tok (attaches on first reselect)"
                    : "lru (recency-only: no Qwen3-0.6B drafter found)",
                pc.tail_window_chunks);
    std::fflush(stdout);
    return true;
}

bool LagunaBackend::kvflash_alloc_span(int kv_start, int n_tok) {
    return !kvflash_active() || kvflash_pager_.alloc_span(kv_start, n_tok);
}

void LagunaBackend::print_ready_banner() const {
    std::printf("[laguna-daemon] ready vocab=%lld eos=%d eot=%d max_ctx=%d kv=%s chunk=%d\n",
                (long long)w_.embedder.n_vocab, w_.eos_id, w_.eos_chat_id,
                args_.max_ctx, ggml_type_name(args_.kv_type), args_.chunk);
}

// ── Park / unpark ───────────────────────────────────────────────────────

bool LagunaBackend::park(ParkTarget target) {
    const bool want_draft_model = park_target_includes_draft_model(target);
    const bool want_target_model = park_target_includes_target_model(target);

    if (want_draft_model && !draft_parked_ && !args_.draft_path.empty()) {
        free_decode_draft();
        draft_parked_ = true;
        std::printf("[park] draft released\n"); std::fflush(stdout);
    }

    if (want_target_model && !target_parked_) {
        if (!draft_parked_ && !args_.draft_path.empty()) {
            free_decode_draft();
            draft_parked_ = true;
        }
        free_laguna_target_cache(cache_);
        free_laguna_target_weights(w_);
        target_parked_ = true;
        std::printf("[park] target released\n"); std::fflush(stdout);
    }
    return true;
}

bool LagunaBackend::unpark(ParkTarget target) {
    const bool want_draft_model = park_target_includes_draft_model(target);
    const bool want_target_model = park_target_includes_target_model(target);
    if (want_target_model && target_parked_) {
        if (!args_.draft_path.empty()) {
            if (!load_target_gguf_laguna(args_.target_path, backend_, w_)) {
                std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
                return false;
            }
            hybrid_mode_ = false;
        } else {
            if (!init_hybrid_mode()) {
                std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
                return false;
            }
        }
        resolve_laguna_kv_types(args_, cache_.kv_k_type, cache_.kv_v_type);
        kvflash_read_config();
        if (laguna_auto_head_major_enabled() &&
            !std::getenv("DFLASH_LAGUNA_KV_HEAD_MAJOR") &&
            kvflash_tokens_ <= 0 &&
            !args_.ddtree_mode) {
            setenv("DFLASH_LAGUNA_KV_HEAD_MAJOR", "1", 0);
            std::fprintf(stderr,
                         "[laguna] auto-enabled head-major KV layout "
                         "(disable with DFLASH_LAGUNA_AUTO_HEAD_MAJOR=0)\n");
        }
        static const bool unpark_swa_ring_env = []() {
            const char * e = std::getenv("DFLASH_LAGUNA_SWA_RING");
            return !(e && e[0] == '0' && e[1] == '\0');
        }();
        const int unpark_swa_ring =
            (kvflash_tokens_ > 0 && !hybrid_mode_ && unpark_swa_ring_env)
                ? std::max(2048, args_.chunk + w_.sliding_window + 64)
                : 0;
        if (!create_laguna_target_cache(w_, args_.max_ctx, backend_, cache_,
                                        kvflash_tokens_, unpark_swa_ring)) {
            std::fprintf(stderr, "[unpark] cache: %s\n", dflash27b_last_error());
            return false;
        }
        if (!kvflash_attach()) {
            free_laguna_target_cache(cache_);
            free_laguna_target_weights(w_);
            return false;          // still parked, resources released
        }
        kvflash_drafter_failed_ = false;   // fresh VRAM: allow a retry
        target_parked_ = false;
        std::printf("[unpark] target restored\n"); std::fflush(stdout);
    }
    if (want_draft_model && draft_parked_ && !args_.draft_path.empty()) {
        if (!load_decode_draft()) return false;
    }
    return true;
}

// ── Snapshots ───────────────────────────────────────────────────────────

bool LagunaBackend::ensure_slot(int slot) {
    if (slot < 0 || slot >= kMaxSlots) {
        std::fprintf(stderr, "[snap] slot=%d out of range\n", slot);
        return false;
    }
    if (target_parked_) {
        std::fprintf(stderr, "[snap] target parked, cannot allocate snapshot\n");
        return false;
    }
    return true;
}

bool LagunaBackend::snapshot_save(int slot) {
    // kvflash: snapshots copy rows assuming identity layout, which breaks
    // after the first page-out relocates a chunk. [TAG_SWA_RING] ring-cached
    // SWA layers hold only the trailing window, so prefix snapshots are
    // impossible there too.
    if (kvflash_active() &&
        (!kvflash_pager_.is_identity() || cache_.swa_ring_rows > 0)) {
        std::fprintf(stderr, "[kvflash] snapshot skipped: pool has relocated "
                             "chunks or SWA layers are ring-cached\n");
        return false;
    }
    if (!ensure_slot(slot)) return false;
    if (!laguna_snapshot_save(cache_, snap_backend_, w_.n_layer,
                               w_.n_head_kv, w_.head_dim, snapshots_[slot])) {
        std::fprintf(stderr, "[snap] save slot=%d: %s\n",
                      slot, dflash27b_last_error());
        return false;
    }
    return true;
}

void LagunaBackend::snapshot_free(int slot) {
    if (slot >= 0 && slot < kMaxSlots) {
        laguna_snapshot_free(snapshots_[slot]);
    }
}

bool LagunaBackend::snapshot_used(int slot) const {
    return slot >= 0 && slot < kMaxSlots && snapshots_[slot].used;
}

int LagunaBackend::snapshot_cur_pos(int slot) const {
    if (slot >= 0 && slot < kMaxSlots && snapshots_[slot].used)
        return snapshots_[slot].cur_pos;
    return -1;
}

// Speculative decode

bool LagunaBackend::do_spec_decode(int committed, int n_gen,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io,
                                    const BudgetHook * budget_hook,
                                    bool * forced_close_out,
                                    float * accept_rate_out,
                                    const std::vector<int32_t> * sample_history_prefix) {
    DraftWeights * active_dw = active_dw_;
    if (!active_dw) return false;
    DraftWeights & dw = *active_dw;
    const int hidden = w_.n_embd;
    int32_t last_tok = cache_.last_tok;
    if (last_tok < 0) return false;

    DFlashTarget * target = dflash_target_;
    const int block_size = dw.block_size;
    // [TAG_LAGUNA_VERIFY_WIDTH] Speculative verify width (chain). On this MoE
    // target the batched verify forward's cost grows with the verify width: it
    // reads the union of experts the batch routes to (bandwidth-bound), and above
    // mmvq_mmid_max (8 for Q4_K/Q5_K on sm_86+) it also drops off the ggml
    // CUDA-graph path (ggml-cuda.cu [TAG_MUL_MAT_ID_CUDA_GRAPHS]). The draft
    // proposes block_size tokens but avg_commit is ~2.9, so verifying the whole
    // block is wasteful. Measured (laguna-xs2 Q4_K_M, RTX 3090): width 8 -> 110
    // tok/s, 4 -> 138, 3 -> 150 (== AR). We verify only the first q_len of the
    // drafted block; the accept rule is unchanged, so this stays lossless. AUTO
    // tracks an EWMA of the accepted length (held constant per request so the
    // verify graph stays CUDA-graph-stable); --verify-width forces a fixed width.
    const bool sampled_verify = laguna_sampled_verify_enabled(sampler_, true);
    int verify_width = args_.verify_width;
    if (const char * e = std::getenv("DFLASH_LAGUNA_VERIFY_WIDTH")) {
        const int w = std::atoi(e);
        if (w > 0) verify_width = w;
    }
    const bool adaptive_width = (verify_width <= 0);
    // AUTO width: on RTX 3090 with XS2-class drafters a marginal verify
    // position costs ~2.3ms while its marginal commit is <0.3 tokens beyond
    // width 3, so width 3 dominates for accept lengths in the 1.5-3 range
    // (HumanEval: w3 188 tok/s vs w4 173 vs old AUTO 172). Follow the accept
    // EWMA but cap at DFLASH_LAGUNA_VERIFY_WIDTH_MAX (default 3).
    static const int auto_w_max = []() {
        const char * e = std::getenv("DFLASH_LAGUNA_VERIFY_WIDTH_MAX");
        const int v = e ? std::atoi(e) : 3;
        return v > 0 ? v : 3;
    }();
    // [TAG_ADAPTIVE_WIDTH] default width policy: with the per-step
    // drafter-confidence trim active (on by default for greedy chains), run
    // from a base of 8 rows and let the trim shrink each step. The legacy
    // accept-EWMA AUTO remains the fallback when the trim is off (theta 0)
    // and for sampled verify, which has no candidate probabilities.
    const bool width_trim = adaptive_verify_width_theta() > 0.0f &&
                            !sampled_verify && !args_.ddtree_mode;
    int chain_w = adaptive_width
        ? (width_trim ? 8 : std::min((int)(spec_ewma_accept_ + 0.5) + 1, auto_w_max))
        : verify_width;
    if (chain_w < 2) chain_w = 2;
    if (chain_w > std::min(block_size, 8)) chain_w = std::min(block_size, 8);
    // DDTree sizes its batch via its budget; chain uses the width chosen above.
    const int base_q_len = args_.ddtree_mode ? block_size : chain_w;

    const bool ignore_eos = (std::getenv("DFLASH_IGNORE_EOS") != nullptr);
    if (dflash_target_) {
        dflash_target_->set_keep_verify_logits(sampled_verify);
    }

    StepGraph draft_sg;

    // The draft graph is always block_size-wide (build_draft_step uses
    // dw.block_size); chain reads/verifies only its first q_len outputs.
    std::vector<float>   noise_embed((size_t)hidden * (size_t)block_size);
    std::vector<int32_t> noise_ids((size_t)block_size);
    std::vector<int32_t> draft_tok((size_t)base_q_len);
    std::vector<int32_t> target_tok((size_t)base_q_len);
    std::vector<float>   verify_logits;
    std::vector<int32_t> verify_history;
    std::vector<int32_t> pos_q((size_t)block_size);
    std::vector<int32_t> pos_k;
    std::vector<float>   local_hidden;
    std::vector<int32_t> sample_history =
        sample_history_prefix ? *sample_history_prefix : std::vector<int32_t>{};

    int n_generated   = 0;
    int n_draft_steps = 0;
    int n_accept_sum  = 0;
    int n_draft_pos_sum = 0;

    auto argmax_logits = [](const std::vector<float> & ll) {
        int best = 0;
        float bv = ll.empty() ? 0.0f : ll[0];
        for (size_t i = 1; i < ll.size(); ++i) {
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        }
        return best;
    };

    auto run_ar_tail = [&](int ar_n_gen) -> bool {
        std::vector<float> embed_step((size_t)hidden);
        std::vector<float> logits;
        bool budget_close_started = false;
        int close_inject_pos = 0;
        int32_t tok = last_tok;

        auto maybe_force_close = [&](int32_t & t) {
            if (!budget_hook || budget_hook->close_token_ids.empty()) return;
            if (budget_close_started &&
                close_inject_pos < (int)budget_hook->close_token_ids.size()) {
                int32_t inj = budget_hook->close_token_ids[(size_t)close_inject_pos];
                std::fprintf(stderr,
                    "[budget-hook] laguna spec-tail close-seq continue %d/%zu: "
                    "overriding sampled token %d with %d\n",
                    close_inject_pos + 1,
                    budget_hook->close_token_ids.size(), t, inj);
                t = inj;
                close_inject_pos++;
                return;
            }
            if (budget_close_started) return;
            const int remaining = n_gen - n_generated;
            if (remaining <= budget_hook->hard_limit_remaining) {
                int32_t first_close = budget_hook->close_token_ids.front();
                if (t == first_close) {
                    budget_close_started = true;
                    close_inject_pos = 1;
                    return;
                }
                std::fprintf(stderr,
                    "[budget-hook] laguna spec-tail force-close at generated=%d/%d "
                    "(remaining=%d <= hard_limit=%d): overriding token %d "
                    "with close[0]=%d (seq len %zu)\n",
                    n_generated, n_gen, remaining,
                    budget_hook->hard_limit_remaining, t, first_close,
                    budget_hook->close_token_ids.size());
                t = first_close;
                budget_close_started = true;
                close_inject_pos = 1;
                if (forced_close_out) *forced_close_out = true;
            }
        };

        for (int s = 0; s < ar_n_gen; ++s) {
            maybe_force_close(tok);
            if (!ignore_eos && target->is_eos(tok)) break;

            out_tokens.push_back(tok);
            sample_history.push_back(tok);
            io.emit(tok);
            if (io.cancelled) break;

            if (!target->embed_tokens(&tok, 1, embed_step.data())) return false;
            if (!kvflash_alloc_span(committed, 1) ||
                !laguna_step(backend_, w_, cache_, embed_step.data(), 1,
                             committed, /*no_mask=*/false, logits,
                             kvflash_active() ? &kvflash_pager_ : nullptr)) {
                return false;
            }
            if (feature_mirror_.target_feat && cache_.target_feat) {
                draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                                feature_mirror_, committed, 1);
            }

            committed++;
            cache_.cur_pos = committed;
            n_generated++;
            tok = sampled_verify
                ? sample_logits(logits.data(), (int)logits.size(), sampler_,
                                sample_history, sampler_rng_)
                : argmax_logits(logits);
        }

        last_tok = tok;
        cache_.last_tok = last_tok;
        return true;
    };

    // kvflash: register the prompt prefix so the pager tracks the tree path's
    // position-indexed writes consistently (idempotent if prefill already did).
    if (kvflash_active() && kvflash_pager_.is_identity()) {
        (void)kvflash_alloc_span(0, committed);
    }

    // [TAG_DRAFT_KV] drafter context-KV ring cache: compute the ctx-side
    // K/V once per committed row instead of re-fusing the whole feature
    // window every step (draft ~10ms -> ~3ms once the window fills).
    // Kill switch: DFLASH_DRAFT_KV=0 restores the legacy one-shot graph.
    constexpr int DRAFT_CTX_MAX_DEFAULT = 2048;
    static const bool draft_kv_on = []() {
        const char * e = std::getenv("DFLASH_DRAFT_KV");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    bool use_draft_kv = draft_kv_on && feature_mirror_.target_feat != nullptr;
    if (use_draft_kv && draft_kv_.gf && draft_kv_.built_for != (const void *)&dw) {
        draft_kv_free(draft_kv_);  // drafter variant switched; graph holds old weights
    }
    if (use_draft_kv && !draft_kv_.gf) {
        const int kv_cap = std::min(feature_mirror_.cap,
                                    std::max(DRAFT_CTX_MAX_DEFAULT, args_.draft_ctx_max));
        if (!draft_kv_init(draft_kv_, dw, draft_backend_, kv_cap, nullptr)) {
            draft_kv_free(draft_kv_);
            use_draft_kv = false;
            std::fprintf(stderr,
                "[laguna-spec] draft-kv init failed; using legacy draft path\n");
        }
    }
    // The ring persists across requests but its rows belong to the previous
    // conversation; start every request from an empty ring (the first
    // begin_step bulk-appends the live window from the feature mirror).
    if (use_draft_kv) draft_kv_reset(draft_kv_);

    auto t_dec0 = std::chrono::steady_clock::now();
    static const bool step_prof = dflash_prof_enabled("step");
    double prof_draft_ms = 0.0, prof_heads_ms = 0.0, prof_verify_ms = 0.0;
    // [TAG_FUSED_LOOP] blind-spot laps: commit = verify-end -> loop-top
    // (accept/commit/emit/feature-sync), build = loop-top -> draft-input
    // upload (noise embed on host, build_draft_step, feature copy).
    double prof_commit_ms = 0.0, prof_build_ms = 0.0, prof_dwait_ms = 0.0;
    auto prof_now = std::chrono::steady_clock::now();
    auto prof_lap = [&]() {
        auto t = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t - prof_now).count();
        prof_now = t;
        return ms;
    };

    while (n_generated < n_gen) {
        if (step_prof) prof_commit_ms += prof_lap();  // [TAG_FUSED_LOOP]
        int q_len = base_q_len;
        draft_tok.resize((size_t)q_len);
        target_tok.resize((size_t)q_len);
        const int need_commit_budget = n_gen - n_generated;

        if (budget_hook && !budget_hook->close_token_ids.empty()) {
            const int hard = budget_hook->hard_limit_remaining;
            if (need_commit_budget <= hard + q_len) {
                std::fprintf(stderr,
                    "[budget-hook] laguna spec-decode tail-off at committed=%d "
                    "remaining=%d hard_limit=%d batch=%d - switching to AR\n",
                    committed, need_commit_budget, hard, q_len);
                step_graph_destroy(draft_sg);
                const bool ok = run_ar_tail(need_commit_budget);
                auto t_dec1 = std::chrono::steady_clock::now();
                const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
                const int total_draft_pos = std::max(1, n_draft_pos_sum);
                const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
                std::fprintf(stderr,
                    "[laguna-spec] tail-off-stats tokens=%d time=%.3f s speed=%.2f tok/s "
                    "steps=%d accepted=%d/%d (%.1f%%)\n",
                    n_generated, decode_s,
                    n_generated > 0 ? n_generated / decode_s : 0.0,
                    n_draft_steps, n_accept_sum, total_draft_pos, accept_pct);
                io.emit(-1);
                return ok;
            }
        }

        noise_ids[0] = last_tok;
        for (int i = 1; i < block_size; i++) noise_ids[(size_t)i] = target->mask_token_id();
        if (!target->embed_tokens(noise_ids.data(), block_size, noise_embed.data())) {
            std::fprintf(stderr, "[laguna-spec] noise embed failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        ggml_tensor * draft_hidden = nullptr;
        if (use_draft_kv) {
            // [TAG_DRAFT_KV] append newly committed rows to the ctx-KV ring
            // and refresh positions/masks; the step graph itself never
            // rebuilds (fixed topology, CUDA-graph replay from step 2).
            if (!draft_kv_begin_step(draft_kv_, dw, draft_backend_,
                                     feature_mirror_, committed)) {
                std::fprintf(stderr, "[laguna-spec] draft-kv step prep failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            ggml_backend_tensor_set(draft_kv_.inp_embed, noise_embed.data(), 0,
                                    sizeof(float) * noise_embed.size());
            if (step_prof) prof_build_ms += prof_lap();  // [TAG_FUSED_LOOP]
            if (step_prof) {
                ggml_backend_synchronize(draft_backend_);
                prof_dwait_ms += prof_lap();
            }
            if (ggml_backend_graph_compute(draft_backend_, draft_kv_.gf) !=
                GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[laguna-spec] draft-kv compute failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            draft_hidden = draft_kv_.hidden_states;
        } else {
            const int ring_cap = feature_mirror_.cap;
            const int draft_ctx = std::min(committed,
                std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, args_.draft_ctx_max)));
            const int draft_start = committed - draft_ctx;
            int mirror_slot0 = 0;
            const bool use_mirror_view =
                draft_feature_mirror_can_view(feature_mirror_, committed, draft_ctx, mirror_slot0);

            static const bool draft_pad = []() {
                const char * e = std::getenv("DFLASH_LAGUNA_DRAFT_PAD");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            // [TAG_FUSED_LOOP] persistent draft graph: force the feature-COPY
            // build (D2D peer copy, ~0.1ms) so the topology carries no per-step
            // ring-view offsets and build_draft_step can skip the rebuild while
            // ctx stays inside the same 64-aligned bucket. Kill: DFLASH_DRAFT_PERSIST=0.
            static const bool draft_persist = []() {
                const char * e = std::getenv("DFLASH_DRAFT_PERSIST");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            const bool want_view = use_mirror_view && !draft_persist;
            if (!build_draft_step(draft_sg, dw, /*lm_head=*/nullptr, draft_backend_,
                                  draft_ctx, want_view ? &feature_mirror_ : nullptr,
                                  committed,
                                  std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, args_.draft_ctx_max)),
                                  draft_pad)) {
                std::fprintf(stderr, "[laguna-spec] draft build failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            if (!want_view &&
                !copy_feature_ring_range_to_tensor(feature_mirror_, draft_sg.target_hidden_cat,
                                                   draft_start, draft_ctx)) {
                std::fprintf(stderr, "[laguna-spec] feature copy failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }

            ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                    sizeof(float) * noise_embed.size());
            const int kctx = (draft_sg.ctx_alloc > 0) ? draft_sg.ctx_alloc : draft_ctx;
            pos_k.resize((size_t)kctx + (size_t)block_size);
            for (int i = 0; i < block_size; i++) pos_q[(size_t)i] = draft_ctx + i;
            for (int i = 0; i < kctx; i++) pos_k[(size_t)i] = (i < draft_ctx) ? i : 0;
            for (int j = 0; j < block_size; j++) pos_k[(size_t)kctx + j] = draft_ctx + j;
            if (step_prof) prof_build_ms += prof_lap();  // [TAG_FUSED_LOOP]
            ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                    sizeof(int32_t) * pos_q.size());
            ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                    sizeof(int32_t) * pos_k.size());

            // [TAG_FUSED_LOOP] split pending-work wait from execution when profiling
            if (step_prof) {
                ggml_backend_synchronize(draft_backend_);
                prof_dwait_ms += prof_lap();
            }
            if (ggml_backend_graph_compute(draft_backend_, draft_sg.gf) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[laguna-spec] draft compute failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            draft_hidden = draft_sg.hidden_states;
        }

        // [TAG_FUSED_LOOP] the 64KB draft-hidden D2H readback is lazy: the
        // shipping path (fused Domino) reads the device tensor in place, so
        // only fallback heads and ddtree/dspark pay for the transfer.
        bool hidden_on_host = false;
        auto fetch_hidden = [&]() {
            if (hidden_on_host) return;
            local_hidden.resize((size_t)hidden * (size_t)q_len);
            ggml_backend_tensor_get(draft_hidden, local_hidden.data(), 0,
                                    sizeof(float) * local_hidden.size());
            hidden_on_host = true;
        };
        if (args_.ddtree_mode || sampled_verify) fetch_hidden();
        if (step_prof) prof_draft_ms += prof_lap();

        // [TAG_ADAPTIVE_WIDTH] drafter top-2 candidate probabilities per
        // slot, extracted on whichever head produces the draft tokens; used
        // below to trim the verify batch per step.
        std::vector<float>   cand_p;
        std::vector<int32_t> cand_i;
        const int cand_k = (adaptive_verify_width_theta() > 0.0f &&
                            !sampled_verify && !args_.ddtree_mode && q_len >= 3) ? 2 : 0;
        bool used_domino = false;
        if (dw.domino.enabled && q_len > 1 && !sampled_verify && !args_.ddtree_mode) {
            static std::atomic<bool> s_domino_logged{false};
            if (!s_domino_logged.exchange(true)) {
                std::fprintf(stderr,
                    "[laguna-spec] Domino GRU head active for greedy chain decode "
                    "(H=%d E=%d)\n",
                    dw.domino.gru_hidden_dim, dw.domino.emb_dim);
            }
            static const bool fused_domino = []() {
                const char * e = std::getenv("DFLASH_LAGUNA_FUSED_DOMINO");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            if (fused_domino) {
                // Run on the draft backend: same stream as the draft forward,
                // second graph key in the multi-key CUDA-graph cache.
                // [TAG_FUSED_LOOP] pass the device hidden; no D2H/H2D hop.
                if (domino_correct_greedy_chain_fused(
                        dw, draft_backend_, target->lm_head_tensor(),
                        target->gpu_embd_table(), nullptr, q_len,
                        last_tok, draft_tok,
                        cand_k, cand_k > 0 ? &cand_p : nullptr,
                        cand_k > 0 ? &cand_i : nullptr,
                        draft_hidden)) {
                    used_domino = true;
                }
            }
            if (used_domino) {
                // fused path done
            } else if (fetch_hidden(),
                       domino_correct_greedy_chain(dw, draft_backend_, *target,
                                            local_hidden.data(), q_len,
                                            last_tok, draft_tok)) {
                used_domino = true;
            } else {
                static std::atomic<bool> s_domino_warned{false};
                if (!s_domino_warned.exchange(true)) {
                    std::fprintf(stderr,
                        "[laguna-spec] Domino GRU head failed; falling back to base "
                        "DFlash projection\n");
                }
            }
        }
        bool used_dspark = false;
        if (!used_domino && laguna_dspark_enabled() && dw.dspark.enabled &&
            q_len > 1 && !sampled_verify && !args_.ddtree_mode) {
            static std::atomic<bool> s_dspark_logged{false};
            if (!s_dspark_logged.exchange(true)) {
                std::fprintf(stderr,
                    "[laguna-spec] DSpark Markov head active for greedy chain decode "
                    "(rank=%d vocab=%d confidence_dim=%d)\n",
                    dw.dspark.markov_rank, dw.dspark.vocab_size, dw.dspark.confidence_dim);
            }
            static const bool fused_dspark = []() {
                const char * e = std::getenv("DFLASH_LAGUNA_FUSED_DSPARK");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            bool ds_ok = false;
            fetch_hidden();  // [TAG_FUSED_LOOP]
            if (fused_dspark && laguna_dspark_confidence_threshold() <= 0.0f) {
                // One graph on the draft stream: lm_head + markov chain +
                // in-graph argmax; no host logits round-trip.
                ds_ok = dspark_markov_correct_greedy_chain_fused(
                    dw, draft_backend_, target->lm_head_tensor(),
                    local_hidden.data(), q_len, last_tok, draft_tok);
            }
            if (!ds_ok) {
                ds_ok = dspark_markov_correct_greedy_chain(dw, draft_backend_, *target,
                                                   local_hidden.data(), q_len,
                                                   last_tok,
                                                   laguna_dspark_confidence_threshold(),
                                                   draft_tok);
            }
            if (ds_ok) {
                used_dspark = true;
                q_len = (int)draft_tok.size();
                target_tok.resize((size_t)q_len);
            } else {
                static std::atomic<bool> s_dspark_warned{false};
                if (!s_dspark_warned.exchange(true)) {
                    std::fprintf(stderr,
                        "[laguna-spec] DSpark Markov head failed; falling back to base "
                        "DFlash projection\n");
                }
            }
        }
        if (!used_domino && !used_dspark) {
            fetch_hidden();  // [TAG_FUSED_LOOP]
            if (!target->project_hidden_to_tokens_topk(local_hidden.data(), q_len, draft_tok,
                    cand_k, cand_k > 0 ? &cand_p : nullptr,
                    cand_k > 0 ? &cand_i : nullptr)) {
                std::fprintf(stderr, "[laguna-spec] projection failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            draft_tok[0] = last_tok;
        }

        if (step_prof) prof_heads_ms += prof_lap();
        const bool tree_special_inactive =
            !(budget_hook && !budget_hook->close_token_ids.empty());
        // kvflash: the tree graph is position-indexed, so only take it while
        // the pager is identity and the step fits the resident pool; otherwise
        // the slot-mapped chain verify below handles it.
        const bool kvflash_tree_ok =
            !kvflash_active() ||
            (kvflash_pager_.is_identity() &&
             committed + args_.ddtree_budget + 1 <= kvflash_tokens_);
        if (args_.ddtree_mode && target->supports_tree_verify() && kvflash_tree_ok &&
            q_len > 1 && tree_special_inactive && !sampled_verify) {
            const int L = q_len - 1;
            const int K = (args_.ddtree_budget > L) ? 8 : 1;
            std::vector<float> top_lp;
            std::vector<int32_t> top_ids;
            static const bool dspark_tree = []() {
                const char * e = std::getenv("DFLASH_LAGUNA_DSPARK_TREE");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            bool topk_ok = false;
            if (dspark_tree && laguna_dspark_enabled() && dw.dspark.enabled) {
                static std::atomic<bool> s_dstree_logged{false};
                if (!s_dstree_logged.exchange(true)) {
                    std::fprintf(stderr,
                        "[laguna-spec] DSpark Markov head active for DDTree candidates\n");
                }
                topk_ok = dspark_markov_project_topk(dw, draft_backend_,
                                                     target->lm_head_tensor(),
                                                     local_hidden.data(), q_len, K,
                                                     args_.ddtree_temp, last_tok,
                                                     top_lp, top_ids);
            }
            if (!topk_ok &&
                !target->project_hidden_to_topk(local_hidden.data(), q_len, K,
                                                args_.ddtree_temp, top_lp, top_ids)) {
                std::fprintf(stderr, "[laguna-spec] ddtree topk projection failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }

            DDTree tree = build_ddtree(top_lp.data() + (size_t)K,
                                       top_ids.data() + (size_t)K,
                                       L, K, args_.ddtree_budget,
                                       /*chain_seed=*/true);
            const int N = args_.ddtree_budget + 1;
            std::vector<int32_t> flat_tokens((size_t)N, 0);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; ++i) {
                flat_tokens[(size_t)i + 1] = tree.token_ids[(size_t)i];
            }

            std::vector<int32_t> posterior;
            if (!target->verify_tree(committed, tree, flat_tokens, N, posterior, nullptr)) {
                std::fprintf(stderr, "[laguna-spec] verify_tree failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }

            int next_token = -1;
            int bonus_node = 0;
            std::vector<int> accepted =
                follow_verified_tree(tree, posterior.data(), next_token, &bonus_node);
            (void)bonus_node;

            int commit_n = (int)accepted.size();
            if (commit_n > need_commit_budget) commit_n = need_commit_budget;
            if (commit_n <= 0) {
                step_graph_destroy(draft_sg);
                break;
            }

            bool hit_eos = false;
            int emitted = 0;
            for (int i = 0; i < commit_n; ++i) {
                const int dfs = accepted[(size_t)i];
                const int32_t tok = (dfs == 0) ? last_tok : tree.token_ids[(size_t)dfs - 1];
                if (!ignore_eos && target->is_eos(tok)) { hit_eos = true; break; }
                out_tokens.push_back(tok);
                sample_history.push_back(tok);
                io.emit(tok);
                emitted++;
                if (io.cancelled) break;
            }

            n_accept_sum += std::max(0, emitted - 1);
            n_draft_steps++;
            n_draft_pos_sum += q_len;
            if (io.cancelled || hit_eos || emitted <= 0 || next_token < 0 ||
                (!ignore_eos && target->is_eos(next_token))) {
                committed += emitted;
                cache_.cur_pos = committed;
                n_generated += emitted;
                last_tok = next_token;
                cache_.last_tok = last_tok;
                break;
            }

            std::vector<int> accepted_committed(accepted.begin(),
                                                accepted.begin() + emitted);
            if (!target->rollback_to_tree(committed, tree, accepted_committed)) {
                std::fprintf(stderr, "[laguna-spec] rollback_to_tree failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }

            if (feature_mirror_.target_feat && cache_.target_feat) {
                draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                                feature_mirror_, committed, emitted);
            }

            last_tok = next_token;
            cache_.last_tok = last_tok;
            committed += emitted;
            cache_.cur_pos = committed;
            n_generated += emitted;
            continue;
        }

        // [TAG_ADAPTIVE_WIDTH] trim the verify batch where the drafter's own
        // reach-mass says the tail rows almost never commit: every dropped
        // row saves one MoE verify row (~1ms of expert reads for a Q4 target
        // on a 3090). Output-exactness is preserved: only the number of
        // speculated slots changes, every committed token is still verified.
        if (cand_k > 0 && !cand_p.empty() &&
            cand_p.size() >= (size_t)(q_len - 1) * (size_t)cand_k) {
            const int w_new = adaptive_verify_width(cand_p.data(), cand_k, q_len,
                                                    adaptive_verify_width_theta(),
                                                    adaptive_verify_width_min());
            if (w_new < q_len) {
                q_len = w_new;
                draft_tok.resize((size_t)q_len);
                target_tok.resize((size_t)q_len);
            }
        }

        int verify_last_tok = -1;
        if (step_prof) prof_lap();
        if (!target->verify_batch(draft_tok, committed, verify_last_tok, &target_tok)) {
            std::fprintf(stderr, "[laguna-spec] verify failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        if (step_prof) prof_verify_ms += prof_lap();
        int accept_n = 1;
        int bonus_tok = -1;
        int verify_vocab = 0;
        if (sampled_verify) {
            if (!target->read_verify_logits(q_len, verify_logits)) {
                std::fprintf(stderr, "[laguna-spec] verify logits read failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            verify_vocab = (int)(verify_logits.size() / (size_t)q_len);
            verify_history = sample_history;
            verify_history.push_back(draft_tok[0]);
            for (int i = 0; i < q_len - 1; ++i) {
                const int sampled = sample_logits(
                    verify_logits.data() + (size_t)i * (size_t)verify_vocab,
                    verify_vocab, sampler_, verify_history, sampler_rng_);
                if (draft_tok[(size_t)i + 1] == sampled) {
                    accept_n++;
                    verify_history.push_back(sampled);
                } else {
                    bonus_tok = sampled;
                    break;
                }
            }
        } else {
            for (int i = 0; i < q_len - 1; i++) {
                if (draft_tok[(size_t)i + 1] == target_tok[(size_t)i]) accept_n++;
                else break;
            }
            bonus_tok = (accept_n < q_len) ? target_tok[(size_t)accept_n - 1] : -1;
        }
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; ++i) {
            replay_tok[(size_t)i] = (i < accept_n) ? draft_tok[(size_t)i] : bonus_tok;
        }
        std::vector<int32_t> history_after_commit = sample_history;
        for (int32_t tok : replay_tok) {
            history_after_commit.push_back(tok);
        }

        cache_.cur_pos = committed + accept_n;

        const bool bonus_is_stop = bonus_tok >= 0 && !ignore_eos && target->is_eos(bonus_tok);
        bool bonus_deferred = false;
        if (bonus_tok >= 0 && !sampled_verify && !bonus_is_stop) {
            // Fold the bonus token into the next verify batch instead of
            // running a dedicated 1-token target forward for it. The bonus
            // becomes the next iteration's seed (draft_tok[0]); the next
            // verify_batch writes its KV row and feature-ring entry, exactly
            // like the DDTree path's next_token contract. This removes one of
            // the two target forwards per chain step.
            commit_n = accept_n;
            replay_tok.resize((size_t)commit_n);
            last_tok = bonus_tok;
            bonus_deferred = true;
        }
        if (bonus_tok >= 0 && bonus_is_stop && !sampled_verify) {
            // Generation stops at the bonus token: no KV row is ever needed
            // for it, so skip the bonus forward and let the emit loop hit EOS.
            last_tok = bonus_tok;
        } else if (bonus_tok >= 0 && !bonus_deferred) {
            std::vector<int32_t> bonus_vec = { bonus_tok };
            int bonus_last = -1;
            if (!target->verify_batch(bonus_vec, committed + accept_n, bonus_last, nullptr)) {
                std::fprintf(stderr, "[laguna-spec] bonus forward failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            if (sampled_verify) {
                if (!target->read_verify_logits(1, verify_logits)) {
                    std::fprintf(stderr, "[laguna-spec] bonus logits read failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                const int vocab_v = (int)verify_logits.size();
                last_tok = sample_logits(verify_logits.data(), vocab_v, sampler_,
                                         history_after_commit, sampler_rng_);
            } else {
                last_tok = bonus_last;
            }
        } else if (bonus_tok < 0) {
            if (sampled_verify && commit_n > 0) {
                if (verify_vocab <= 0) {
                    std::fprintf(stderr, "[laguna-spec] invalid sampled verify vocab\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                const int row = std::min(commit_n - 1, q_len - 1);
                last_tok = sample_logits(
                    verify_logits.data() + (size_t)row * (size_t)verify_vocab,
                    verify_vocab, sampler_, history_after_commit, sampler_rng_);
            } else {
                last_tok = verify_last_tok;
            }
        }
        cache_.last_tok = last_tok;

        if (feature_mirror_.target_feat && cache_.target_feat) {
            draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, committed, commit_n);
        }

        bool hit_eos = false;
        int emitted = 0;
        for (int i = 0; i < commit_n; i++) {
            int tok = replay_tok[(size_t)i];
            if (!ignore_eos && target->is_eos(tok)) { hit_eos = true; break; }
            out_tokens.push_back(tok);
            sample_history.push_back(tok);
            io.emit(tok);
            emitted++;
            if (io.cancelled) break;
        }

        committed += emitted;
        cache_.cur_pos = committed;
        n_generated += emitted;
        n_accept_sum += std::min(accept_n, emitted);
        n_draft_steps++;
        n_draft_pos_sum += q_len;
        if (io.cancelled) break;
        if (hit_eos) break;
    }

    step_graph_destroy(draft_sg);

    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_draft_pos_sum);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    if (step_prof && n_draft_steps > 0) {
        std::fprintf(stderr,
            "[step-prof] per-step ms: draft=%.2f heads=%.2f verify=%.2f "
            "commit=%.2f build=%.2f dwait=%.2f other=%.2f total=%.2f (steps=%d)\n",
            prof_draft_ms / n_draft_steps, prof_heads_ms / n_draft_steps,
            prof_verify_ms / n_draft_steps,
            prof_commit_ms / n_draft_steps, prof_build_ms / n_draft_steps,
            prof_dwait_ms / n_draft_steps,
            (decode_s * 1000.0 - prof_draft_ms - prof_heads_ms - prof_verify_ms -
             prof_commit_ms - prof_build_ms - prof_dwait_ms) / n_draft_steps,
            decode_s * 1000.0 / n_draft_steps, n_draft_steps);
    }
    std::fprintf(stderr, "[laguna-spec] tokens=%d time=%.3f s speed=%.2f tok/s "
                 "steps=%d accepted=%d/%d (%.1f%%) avg_commit=%.2f\n",
                 n_generated, decode_s,
                 n_generated > 0 ? n_generated / decode_s : 0.0,
                 n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                 n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);

    if (accept_rate_out) {
        *accept_rate_out = (float)(n_accept_sum / (double)total_draft_pos);
    }

    // [TAG_LAGUNA_VERIFY_WIDTH] Update the persisted accepted-length EWMA so the
    // AUTO width converges to the throughput optimum for the active draft (chain
    // only; DDTree sizes via its budget and accounts accepts differently).
    if (adaptive_width && !args_.ddtree_mode && n_draft_steps > 0) {
        const double mean_accept = (double)n_accept_sum / (double)n_draft_steps;
        spec_ewma_accept_ = 0.7 * spec_ewma_accept_ + 0.3 * mean_accept;
    }

    if (dflash_target_) {
        dflash_target_->set_keep_verify_logits(false);
    }
    io.emit(-1);
    return true;
}

// ── Generation ──────────────────────────────────────────────────────────

GenerateResult LagunaBackend::generate_impl(const GenerateRequest & req,
                                        const DaemonIO & io) {
    if (hybrid_mode_ && moe_hybrid_) {
        auto result = generate_hybrid(req, io);
        if (result.ok()) {
            // Flush routing-frequency profile if requested (independent of swap).
            if (!routing_stats_out_path_.empty() && routing_stats_) {
                std::string serr;
                if (!routing_stats_->save_csv(routing_stats_out_path_, &serr))
                    std::fprintf(stderr, "[laguna-hybrid] profile save failed: %s\n", serr.c_str());
                else
                    std::fprintf(stderr, "[laguna-hybrid] profile saved: %s\n", routing_stats_out_path_.c_str());
            }
            maybe_post_request_swap();
        }
        return result;
    }

    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    const bool should_emit = req.stream || (bool)out_io.on_token;
    const int N = (int)req.prompt.size();
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    if (N + req.n_gen > args_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow);
        return result;
    }

    // kvflash: prompts that fit the pool prefill identity-mapped. Larger
    // prompts take the pooled path: pager-chunk-sized batches allocated via
    // slot_for (evicting as the pool fills), slot-mapped KV writes and
    // slot-space masks via kvflash_fill_rows_and_masks — same recipe as the
    // qwen35 pooled prefill. Constant VRAM, linear time.
    const bool kvf_paged = kvflash_active() &&
        N > kvflash_tokens_ - kvflash_pager_.chunk_tokens();
    if (kvf_paged) {
        std::printf("[kvflash] pooled prefill: %d tokens through a %d-token "
                    "pool (%d-token chunks, evicting)\n",
                    N, kvflash_tokens_, kvflash_pager_.chunk_tokens());
        std::fflush(stdout);
    }

    reset_laguna_target_cache(cache_);
    if (kvflash_active()) kvflash_pager_.reset();
    const KvFlashPager * kvf = kvflash_active() ? &kvflash_pager_ : nullptr;

    // ── Prefill ──
    std::vector<float> embed_pf((size_t)N * w_.n_embd);
    if (!w_.embedder.embed(req.prompt.data(), N, embed_pf.data())) {
        result.fail(GenerateErrorCode::BackendSpecific, "embed_prefill");
        return result;
    }

    auto t_pf0 = std::chrono::steady_clock::now();
    std::vector<float> last_logits;
    bool ok = true;
    // Pooled batches span multiple pager chunks (graphs replay, so big
    // batches are pure win); each batch is chunk-aligned for slot_for.
    const int pf_pool_batch = std::max(kvflash_pager_.chunk_tokens(),
        (args_.chunk / std::max(1, kvflash_pager_.chunk_tokens())) *
            std::max(1, kvflash_pager_.chunk_tokens()));
    const int pf_chunk = kvf_paged ? pf_pool_batch : args_.chunk;
    const int n_chunks = (N + pf_chunk - 1) / pf_chunk;
    for (int c = 0; c < n_chunks && ok; ++c) {
        const int kv_start = c * pf_chunk;
        const int n_tok    = std::min(pf_chunk, N - c * pf_chunk);
        if (kvf_paged) {
            // Pooled path: allocate this chunk's slots up front, evicting the
            // lowest-priority resident chunk once the pool fills. laguna_step
            // then slot-maps the KV writes + masks via the shared helper.
            for (int i = 0; i < n_tok && ok; i++) {
                ok = kvflash_pager_.slot_for(kv_start + i) >= 0;
            }
            if (!ok) {
                std::fprintf(stderr,
                    "[kvflash] pooled prefill: slot alloc failed @%d\n", kv_start);
                break;
            }
        } else {
            ok = kvflash_alloc_span(kv_start, n_tok);
        }
        ok = ok &&
             laguna_step(backend_, w_, cache_,
                          embed_pf.data() + (size_t)kv_start * w_.n_embd,
                          n_tok, kv_start, no_mask, last_logits, kvf,
                          /*capture=*/true);
    }
    if (!ok) { result.fail(GenerateErrorCode::PrefillFailed); return result; }
    auto t_pf1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();

    // ── Inline snapshot (if requested) ──
    // kvflash: snapshots copy rows [0, snap_pos) assuming identity layout,
    // which holds until the first page-out relocates a chunk.
    if (kvflash_active() && req.snap_slot >= 0 &&
        (!kvflash_pager_.is_identity() || cache_.swa_ring_rows > 0)) {
        std::fprintf(stderr, "[kvflash] snapshot skipped: pool has relocated "
                             "chunks or SWA layers are ring-cached\n");
    } else
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= N) {
        if (ensure_slot(req.snap_slot) &&
            laguna_snapshot_save(cache_, snap_backend_, w_.n_layer,
                                  w_.n_head_kv, w_.head_dim, snapshots_[req.snap_slot])) {
            snapshots_[req.snap_slot].cur_pos = req.snap_pos;
            // Record the last committed token so an exact-hit restore can
            // re-embed it even when handed a zero-filled restore-only prompt.
            snapshots_[req.snap_slot].last_tok = req.prompt[req.snap_pos - 1];
            std::printf("[snap] inline slot=%d cur_pos=%d\n",
                         req.snap_slot, req.snap_pos);
            std::fflush(stdout);
        }
    }

    // ── Decode ──
    auto argmax = [](const std::vector<float> & ll) {
        int best = 0; float bv = ll[0];
        for (size_t i = 1; i < ll.size(); ++i)
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        return best;
    };

    std::vector<int32_t> history;
    history.reserve((size_t)N + (size_t)req.n_gen);
    history.insert(history.end(), req.prompt.begin(), req.prompt.end());

    auto pick = [&](const std::vector<float> & ll) -> int {
        return req.do_sample
            ? sample_logits(ll.data(), (int)ll.size(), sampler_, history, sampler_rng_)
            : argmax(ll);
    };

    cache_.last_tok = argmax(last_logits);
    result.tokens.reserve(req.n_gen);
    const bool sampled_verify = laguna_sampled_verify_enabled(sampler_, req.do_sample);
    const bool can_spec = req.n_gen > 0
        && !req.force_ar_decode
        && !args_.draft_path.empty()
        && dflash_target_
        && select_decode_draft(std::string())
        && !draft_parked_
        && feature_mirror_.target_feat
        && cache_.target_feat
        && (!sampler_.needs_logit_processing() || sampled_verify);

    if (can_spec) {
        if (sampled_verify) {
            cache_.last_tok = sample_logits(last_logits.data(), (int)last_logits.size(),
                                            sampler_, history, sampler_rng_);
        }
        auto t_g0 = std::chrono::steady_clock::now();
        if (!draft_feature_mirror_sync_tail(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, N)) {
            result.fail(GenerateErrorCode::BackendSpecific, "feature_sync");
            return result;
        }
        result.spec_decode_ran = true;
        if (!do_spec_decode(N, req.n_gen, result.tokens, out_io,
                            &req.budget_hook,
                            &result.budget_forced_close,
                            &result.accept_rate,
                            &history)) {
            result.fail(GenerateErrorCode::BackendSpecific, "spec_decode");
            return result;
        }
        auto t_g1 = std::chrono::steady_clock::now();
        result.decode_s = std::chrono::duration<double>(t_g1 - t_g0).count();
        result.succeed();
        return result;
    }

    int next_tok = pick(last_logits);

    // Budget force-close state — see model_backend.h BudgetHook docs.
    // Mirrors qwen35/do_ar_decode's maybe_force_close. Laguna has no
    // spec-decode path so this is the only override site.
    const BudgetHook & budget_hook = req.budget_hook;
    bool budget_close_started = false;
    int  close_inject_pos     = 0;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size())
        {
            int32_t inj = budget_hook.close_token_ids[close_inject_pos];
            std::fprintf(stderr,
                "[budget-hook] laguna close-seq continue %d/%zu: overriding "
                "sampled token %d with %d\n",
                close_inject_pos + 1,
                budget_hook.close_token_ids.size(), tok, inj);
            tok = inj;
            close_inject_pos++;
            return;
        }
        if (budget_close_started) return;
        int remaining = req.n_gen - committed_now;
        if (remaining <= budget_hook.hard_limit_remaining) {
            int32_t first_close = budget_hook.close_token_ids.front();
            if (tok == first_close) {
                budget_close_started = true;
                close_inject_pos = 1;
                return;
            }
            std::fprintf(stderr,
                "[budget-hook] laguna force-close at committed=%d/%d "
                "(remaining=%d <= hard_limit=%d): overriding token %d "
                "with close[0]=%d (seq len %zu)\n",
                committed_now, req.n_gen, remaining,
                budget_hook.hard_limit_remaining, tok, first_close,
                budget_hook.close_token_ids.size());
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            result.budget_forced_close = true;
        }
    };

    std::vector<float> embed_step((size_t)w_.n_embd);
    auto t_g0 = std::chrono::steady_clock::now();
    for (int s = 0; s < req.n_gen; ++s) {
        maybe_force_close(next_tok, s);
        if (!std::getenv("DFLASH_IGNORE_EOS") && (next_tok == w_.eos_id || next_tok == w_.eos_chat_id)) break;
        result.tokens.push_back(next_tok);
        history.push_back(next_tok);
        if (should_emit) {
            out_io.emit(next_tok);
            if (out_io.cancelled) break;
        }
        if (!w_.embedder.embed(&next_tok, 1, embed_step.data())) { ok = false; break; }
        std::vector<float> step_logits;
        int32_t step_argmax = -1;
        const bool use_gpu_argmax = !req.do_sample && laguna_gpu_argmax_enabled();
        if (!kvflash_alloc_span(cache_.cur_pos, 1) ||
            !laguna_step(backend_, w_, cache_, embed_step.data(), 1,
                          cache_.cur_pos, no_mask, step_logits, kvf,
                          /*capture=*/false,
                          use_gpu_argmax ? &step_argmax : nullptr,
                          /*read_logits=*/!use_gpu_argmax)) { ok = false; break; }
        kvflash_maybe_reselect(history, s + 1);
        next_tok = use_gpu_argmax ? step_argmax : pick(step_logits);
    }
    auto t_g1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_g1 - t_g0).count();

    if (should_emit) out_io.emit(-1);
    if (!ok) { result.fail(GenerateErrorCode::DecodeFailed); return result; }

    result.succeed();
    return result;
}

// ── RESTORE + generate ──────────────────────────────────────────────────

GenerateResult LagunaBackend::restore_and_generate_impl(int slot,
                                                        const GenerateRequest & req,
                                                        const DaemonIO & io) {
    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    if (!laguna_snapshot_restore(snapshots_[slot], cache_)) {
        std::fprintf(stderr, "[snap] RESTORE slot=%d: %s\n",
                      slot, dflash27b_last_error());
        result.fail(GenerateErrorCode::BackendSpecific, "restore");
        return result;
    }

    const int prefix_len = cache_.cur_pos;
    const int N = (int)req.prompt.size();
    if (N < prefix_len) {
        std::fprintf(stderr, "[snap] RESTORE prompt shorter than cached prefix (%d < %d)\n",
                      N, prefix_len);
        result.fail(GenerateErrorCode::BackendSpecific, "prefix_shorter");
        return result;
    }

    // kvflash: restore lands rows identity-mapped; the full prompt (prefix +
    // diff) must fit the pool. Rebuild the pager mapping over the prefix.
    if (kvflash_active() &&
        N > kvflash_tokens_ - kvflash_pager_.chunk_tokens()) {
        std::fprintf(stderr, "[kvflash] restore prompt (%d) exceeds pool %d; "
                             "raise --kvflash\n", N, kvflash_tokens_);
        result.fail(GenerateErrorCode::ContextOverflow);
        return result;
    }
    if (kvflash_active()) {
        kvflash_pager_.reset();
        if (!kvflash_alloc_span(0, prefix_len)) {
            result.fail(GenerateErrorCode::BackendSpecific, "kvflash_slot");
            return result;
        }
    }
    const KvFlashPager * kvf = kvflash_active() ? &kvflash_pager_ : nullptr;

    // Re-prefill diff tokens (or last cached token when diff is empty).
    bool restore_only = false;
    if (prefix_len == N) {
        if (prefix_len <= 0) {
            result.fail(GenerateErrorCode::BackendSpecific, "empty_diff");
            return result;
        }
        cache_.cur_pos = prefix_len - 1;
        restore_only = true;
    }
    const int kv_start = cache_.cur_pos;
    const int diff_n   = N - kv_start;

    // On an exact full-prompt hit (restore_only, diff_n == 1) the caller may
    // hand us a zero-filled restore-only prompt (pflash full-cache hit, where
    // effective_prompt is a dummy buffer sized to the cached/compressed length).
    // req.prompt[kv_start] would then be token 0, corrupting the final logits.
    // Re-embed the real last committed token recorded in the snapshot instead.
    std::vector<int32_t> diff_ids;
    const int32_t * diff_src = req.prompt.data() + kv_start;
    if (restore_only && snapshots_[slot].last_tok >= 0) {
        diff_ids.assign(1, snapshots_[slot].last_tok);
        diff_src = diff_ids.data();
    }

    std::vector<float> embed_diff((size_t)diff_n * w_.n_embd);
    if (!w_.embedder.embed(diff_src, diff_n, embed_diff.data())) {
        result.fail(GenerateErrorCode::BackendSpecific, "embed_prefill");
        return result;
    }

    std::vector<float> last_logits;
    bool ok = true;
    const int n_chunks = (diff_n + args_.chunk - 1) / args_.chunk;
    for (int c = 0; c < n_chunks && ok; ++c) {
        const int off   = c * args_.chunk;
        const int n_tok = std::min(args_.chunk, diff_n - off);
        const int starts = kv_start + off;
        ok = kvflash_alloc_span(starts, n_tok) &&
             laguna_step(backend_, w_, cache_,
                          embed_diff.data() + (size_t)off * w_.n_embd,
                          n_tok, starts, no_mask, last_logits, kvf,
                          /*capture=*/true);
    }
    if (!ok) { result.fail(GenerateErrorCode::PrefillFailed); return result; }

    // ── Decode ──
    auto argmax = [](const std::vector<float> & ll) {
        int best = 0; float bv = ll[0];
        for (size_t i = 1; i < ll.size(); ++i)
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        return best;
    };
    std::vector<int32_t> history(req.prompt);
    auto pick = [&](const std::vector<float> & ll) {
        return req.do_sample
            ? sample_logits(ll.data(), (int)ll.size(), sampler_, history, sampler_rng_)
            : argmax(ll);
    };

    const int committed = cache_.cur_pos;
    cache_.last_tok = argmax(last_logits);
    result.tokens.reserve(req.n_gen);
    const bool sampled_verify = laguna_sampled_verify_enabled(sampler_, req.do_sample);
    const bool can_spec = req.n_gen > 0
        && !req.force_ar_decode
        && !args_.draft_path.empty()
        && dflash_target_
        && select_decode_draft(std::string())
        && !draft_parked_
        && feature_mirror_.target_feat
        && cache_.target_feat
        && (!sampler_.needs_logit_processing() || sampled_verify);

    if (can_spec) {
        if (sampled_verify) {
            cache_.last_tok = sample_logits(last_logits.data(), (int)last_logits.size(),
                                            sampler_, history, sampler_rng_);
        }
        auto t_g0 = std::chrono::steady_clock::now();
        if (!draft_feature_mirror_sync_tail(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, committed)) {
            result.fail(GenerateErrorCode::BackendSpecific, "feature_sync");
            return result;
        }
        result.spec_decode_ran = true;
        if (!do_spec_decode(committed, req.n_gen, result.tokens, out_io,
                            &req.budget_hook,
                            &result.budget_forced_close,
                            &result.accept_rate,
                            &history)) {
            result.fail(GenerateErrorCode::BackendSpecific, "spec_decode");
            return result;
        }
        auto t_g1 = std::chrono::steady_clock::now();
        result.decode_s = std::chrono::duration<double>(t_g1 - t_g0).count();
        result.succeed();
        return result;
    }

    int next_tok = pick(last_logits);

    const BudgetHook & budget_hook = req.budget_hook;
    bool budget_close_started = false;
    int  close_inject_pos     = 0;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size())
        {
            int32_t inj = budget_hook.close_token_ids[close_inject_pos];
            std::fprintf(stderr,
                "[budget-hook] laguna(restore) close-seq continue %d/%zu: "
                "overriding sampled token %d with %d\n",
                close_inject_pos + 1,
                budget_hook.close_token_ids.size(), tok, inj);
            tok = inj;
            close_inject_pos++;
            return;
        }
        if (budget_close_started) return;
        int remaining = req.n_gen - committed_now;
        if (remaining <= budget_hook.hard_limit_remaining) {
            int32_t first_close = budget_hook.close_token_ids.front();
            if (tok == first_close) {
                budget_close_started = true;
                close_inject_pos = 1;
                return;
            }
            std::fprintf(stderr,
                "[budget-hook] laguna(restore) force-close at "
                "committed=%d/%d (remaining=%d <= hard_limit=%d): "
                "overriding token %d with close[0]=%d (seq len %zu)\n",
                committed_now, req.n_gen, remaining,
                budget_hook.hard_limit_remaining, tok, first_close,
                budget_hook.close_token_ids.size());
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            result.budget_forced_close = true;
        }
    };

    std::vector<float> embed_step((size_t)w_.n_embd);
    auto t_g0 = std::chrono::steady_clock::now();
    for (int s = 0; s < req.n_gen; ++s) {
        maybe_force_close(next_tok, s);
        if (!std::getenv("DFLASH_IGNORE_EOS") && (next_tok == w_.eos_id || next_tok == w_.eos_chat_id)) break;
        history.push_back(next_tok);
        result.tokens.push_back(next_tok);
        out_io.emit(next_tok);
        if (out_io.cancelled) break;
        if (!w_.embedder.embed(&next_tok, 1, embed_step.data())) { ok = false; break; }
        std::vector<float> step_logits;
        int32_t step_argmax = -1;
        const bool use_gpu_argmax = !req.do_sample && laguna_gpu_argmax_enabled();
        if (!kvflash_alloc_span(cache_.cur_pos, 1) ||
            !laguna_step(backend_, w_, cache_, embed_step.data(), 1,
                          cache_.cur_pos, no_mask, step_logits, kvf,
                          /*capture=*/false,
                          use_gpu_argmax ? &step_argmax : nullptr,
                          /*read_logits=*/!use_gpu_argmax)) { ok = false; break; }
        kvflash_maybe_reselect(history, s + 1);
        next_tok = use_gpu_argmax ? step_argmax : pick(step_logits);
    }
    auto t_g1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_g1 - t_g0).count();

    out_io.emit(-1);
    if (!ok) { result.fail(GenerateErrorCode::DecodeFailed); return result; }

    result.succeed();
    return result;
}

// ── Compress (pflash) ───────────────────────────────────────────────────

static std::vector<int32_t> read_uncounted_i32_local(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> ids(sz / sizeof(int32_t));
    if (!ids.empty()) {
        f.read(reinterpret_cast<char *>(ids.data()),
               (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

bool LagunaBackend::handle_compress(const std::string & line,
                                      const DaemonIO & io) {
    char ppath[1024];
    int  keep_x1000 = 0;
    char drafter_path[1024];
    const int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s",
                               ppath, &keep_x1000, drafter_path);
    if (n != 3) {
        std::fprintf(stderr,
                      "[compress] bad args, need: <bin> <keep_x1000> <drafter.gguf>\n");
        io.emit(-1);
        return true;
    }
    auto src_ids = read_uncounted_i32_local(ppath);
    if (src_ids.empty()) {
        std::fprintf(stderr, "[compress] empty input\n");
        io.emit(-1);
        return true;
    }

    const bool restore_target = !target_parked_;
    if (restore_target) park(ParkTarget::TargetModel);

    if (!drafter_loaded_) {
        if (!load_drafter(drafter_path, /*gpu_layers=*/999, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] load_drafter failed: %s\n",
                          dflash27b_last_error());
            io.emit(-1);
            return true;
        }
        drafter_loaded_ = true;
        std::printf("[drafter] loaded %s vocab=%d\n",
                     drafter_path, drafter_ctx_.weights.n_vocab);
        std::fflush(stdout);
    }

    const float keep = (float)keep_x1000 / 1000.0f;
    auto compressed = drafter_score_and_compress(drafter_ctx_, src_ids, keep);
    std::printf("[compress] %zu -> %zu tokens (keep_ratio=%.3f)\n",
                 src_ids.size(), compressed.size(), keep);
    std::fflush(stdout);

    if (restore_target) unpark(ParkTarget::TargetModel);

    for (int32_t t : compressed) io.emit(t);
    io.emit(-1);
    return true;
}

void LagunaBackend::free_drafter() {
    if (drafter_loaded_) {
        dflash::common::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
        std::printf("[drafter] freed\n"); std::fflush(stdout);
    }
}

// ── Hybrid MoE mode ─────────────────────────────────────────────────────
//
// Layer-by-layer decode: for each token, iterate through all 40 layers.
// Layer 0 (dense SwiGLU) runs as a monolithic GPU sub-graph.
// Layers 1..39 (sparse MoE) run attention+router on GPU, read back expert
// selections, then call the common hybrid FFN eval (hot on GPU, cold on CPU).

using HybridClock = std::chrono::steady_clock;
static inline uint64_t elapsed_us(HybridClock::time_point t0, HybridClock::time_point t1) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

bool LagunaBackend::init_hybrid_mode() {
    const char * hotness_path = std::getenv("DFLASH_LAGUNA_HOTNESS");
    placement_all_hot_full_kv_ = false;

    // Step 1: Load model WITHOUT expert data to GPU (partial load)
    TargetLoadPlan _hybrid_plan;
    _hybrid_plan.skip_expert_tensors = true;
    if (!load_target_gguf_laguna_partial(args_.target_path, backend_, _hybrid_plan, w_)) {
        std::fprintf(stderr, "[laguna-hybrid] partial load failed: %s\n", dflash27b_last_error());
        return false;
    }

    // Step 2: Load/build routing stats
    MoeHybridRoutingStats hotness;
    std::string err;
    std::string placement_source;
    if (hotness_path && hotness_path[0]) {
        if (!MoeHybridRoutingStats::load_csv(std::string(hotness_path), hotness, &err)) {
            std::fprintf(stderr, "[laguna-hybrid] hotness load failed: %s\n", err.c_str());
            return false;
        }
        if (hotness.n_layer != w_.n_layer || hotness.n_expert != w_.n_expert) {
            std::fprintf(stderr, "[laguna-hybrid] hotness dimensions mismatch (got %d×%d, want %d×%d)\n",
                          hotness.n_layer, hotness.n_expert, w_.n_layer, w_.n_expert);
            return false;
        }
        placement_source = "file";
    } else {
        // Uniform hotness (budget-only mode, no hotness file)
        hotness.n_layer = w_.n_layer;
        hotness.n_expert = w_.n_expert;
        hotness.n_expert_used = w_.n_expert_used;
        hotness.counts.assign((size_t)w_.n_layer * (size_t)w_.n_expert, 1);
        hotness.layer_totals.assign((size_t)w_.n_layer, (uint64_t)w_.n_expert);
        placement_source = "uniform";
    }

    // Step 3: Query GPU memory and compute expert budget
    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    if (dev) {
        ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
    }
    if (gpu_total == 0) {
        std::fprintf(stderr, "[laguna-hybrid] could not query GPU memory\n");
        return false;
    }

    // Compute per-layer expert size in bytes (laguna: separate gate/up/down, no fused)
    std::vector<uint64_t> layer_expert_bytes((size_t)w_.n_layer);
    for (int il = w_.n_layer_dense_lead; il < w_.n_layer; ++il) {
        const LagunaTargetLayer & L = w_.layers[(size_t)il];
        uint64_t bytes = 0;
        if (L.ffn_gate_exps) bytes += ggml_nbytes(L.ffn_gate_exps) / (uint64_t)w_.n_expert;
        if (L.ffn_up_exps)   bytes += ggml_nbytes(L.ffn_up_exps) / (uint64_t)w_.n_expert;
        if (L.ffn_down_exps) bytes += ggml_nbytes(L.ffn_down_exps) / (uint64_t)w_.n_expert;
        layer_expert_bytes[(size_t)il] = bytes;
    }
    // Layer 0 is dense — no experts
    for (int il = 0; il < w_.n_layer_dense_lead; ++il) {
        layer_expert_bytes[(size_t)il] = 0;
    }

    uint64_t total_expert_bytes = 0;
    for (int il = 0; il < w_.n_layer; ++il) {
        total_expert_bytes += layer_expert_bytes[(size_t)il] * (uint64_t)w_.n_expert;
    }

    // KV cache estimate
    const char * ctx_env = std::getenv("DFLASH_MAX_CONTEXT");
    int max_context = ctx_env ? std::atoi(ctx_env) : args_.max_ctx;
    if (max_context <= 0) max_context = 8192;

    const uint64_t kv_bytes_per_tok = (uint64_t)w_.n_layer * 2 *
        (uint64_t)w_.n_head_kv * (uint64_t)w_.head_dim * 2;

    const uint64_t warm_cache_bytes = 200ULL * 1024 * 1024;
    const uint64_t safety_bytes = 512ULL * 1024 * 1024;
    const uint64_t core_bytes =
        moe_hybrid_core_bytes_from_memory("laguna", gpu_free, gpu_total);

    kvflash_resolve_drafter();
    const int kvf_pool = kvflash_pool_from_env(
        max_context, kvflash_config(), kvflash_scorer_expected(),
        make_kvflash_budget((int64_t)gpu_free));
    const auto kvf_dec = dflash::common::kvflash_placement_decision(
        kv_bytes_per_tok, max_context, kvf_pool,
        gpu_total, core_bytes, total_expert_bytes,
        warm_cache_bytes, safety_bytes, /*draft_bytes=*/0);
    const uint64_t kv_total = kvf_dec.kv_total;
    const int kv_ctx_log = kvf_dec.kv_ctx;
    placement_all_hot_full_kv_ = kvf_dec.all_hot_full_kv;
    if (kvf_dec.pool_reduced) {
        std::printf("[laguna][kvflash] placement reserves pool KV (%d tokens, "
                    "not max_ctx %d) -> experts stay hot\n", kvf_pool, max_context);
        std::fflush(stdout);
    }

    uint64_t expert_budget = 0;
    if (gpu_total > core_bytes + kv_total + warm_cache_bytes + safety_bytes) {
        expert_budget = gpu_total - core_bytes - kv_total - warm_cache_bytes - safety_bytes;
    }
    if (expert_budget > total_expert_bytes) {
        expert_budget = total_expert_bytes;
    }

    // Manual budget cap (absolute MB)
    if (const char * cap_env = std::getenv("DFLASH_EXPERT_BUDGET_MB")) {
        uint64_t cap_bytes = (uint64_t)std::atoi(cap_env) * 1024ULL * 1024ULL;
        if (cap_bytes > 0 && cap_bytes < expert_budget) {
            std::printf("[laguna-hybrid] capping expert budget from %.2f GiB to %d MB\n",
                        expert_budget / 1024.0 / 1024.0 / 1024.0, std::atoi(cap_env));
            expert_budget = cap_bytes;
        }
    }

    // Percentage-based budget cap
    if (const char * pct_env = std::getenv("DFLASH_EXPERT_BUDGET_PCT")) {
        int pct = std::atoi(pct_env);
        if (pct > 0 && pct < 100) {
            uint64_t pct_bytes = total_expert_bytes * (uint64_t)pct / 100ULL;
            if (pct_bytes < expert_budget) {
                std::printf("[laguna-hybrid] capping expert budget to %d%% = %.2f GiB (of %.2f GiB)\n",
                            pct, pct_bytes / 1024.0 / 1024.0 / 1024.0,
                            total_expert_bytes / 1024.0 / 1024.0 / 1024.0);
                expert_budget = pct_bytes;
            }
        }
    }

    // Spark: clamp experts to the --spark-vram target and auto-size the cache ring.
    if (std::getenv("DFLASH_SPARK")) {
        uint64_t target = 0;
        if (const char * t = std::getenv("DFLASH_SPARK_VRAM_MB")) target = (uint64_t)std::atoll(t) << 20;
        auto sb = dflash::common::spark_budget_split(expert_budget, total_expert_bytes, w_.n_expert,
                                                     core_bytes + kv_total + safety_bytes, target);
        expert_budget = sb.hot_bytes;
        cache_slots_ = sb.cache_slots;
        std::printf("[spark] vram=%s, hot=%.2f GiB, cache=%d slots/layer\n",
                    target ? "target" : "auto(card)", expert_budget / 1073741824.0, cache_slots_);
    }

    std::printf("[laguna] dynamic placement: gpu_total=%.2f GiB, core=%.2f GiB, "
                "kv_cache=%.2f GiB (ctx=%d), warm=%.0f MB, safety=%.0f MB, "
                "expert_budget=%.2f GiB (of %.2f GiB total experts)\n",
                gpu_total / 1024.0 / 1024.0 / 1024.0,
                core_bytes / 1024.0 / 1024.0 / 1024.0,
                kv_total / 1024.0 / 1024.0 / 1024.0,
                kv_ctx_log,
                warm_cache_bytes / 1024.0 / 1024.0,
                safety_bytes / 1024.0 / 1024.0,
                expert_budget / 1024.0 / 1024.0 / 1024.0,
                total_expert_bytes / 1024.0 / 1024.0 / 1024.0);
    std::fflush(stdout);

    if (expert_budget == 0) {
        std::fprintf(stderr, "[laguna-hybrid] no VRAM budget for experts\n");
        return false;
    }

    // Stash for the Spark bootstrap rebuild (same budget + cache as this init).
    spark_expert_budget_ = expert_budget;
    layer_expert_bytes_  = layer_expert_bytes;

    // Step 4: Build placement
    MoeHybridPlacement placement;
    if (!MoeHybridPlacement::build_from_stats_with_layer_bytes(
            hotness, layer_expert_bytes, expert_budget,
            /*min_hot_per_layer=*/std::min(w_.n_expert_used, w_.n_expert),
            placement, &err)) {
        std::fprintf(stderr, "[laguna-hybrid] placement build failed: %s\n", err.c_str());
        return false;
    }

    int total_moe_experts = (w_.n_layer - w_.n_layer_dense_lead) * w_.n_expert;
    std::printf("[laguna] dynamic placement result: %d hot experts, %d cold experts\n",
                placement.total_hot, total_moe_experts - placement.total_hot);

    // If all experts fit, reload full model to GPU (non-hybrid path)
    if (placement.total_hot >= total_moe_experts) {
        std::printf("[laguna] all experts fit in VRAM, loading fully to GPU\n");
        std::fflush(stdout);
        free_laguna_target_weights(w_);
        if (!load_target_gguf_laguna(args_.target_path, backend_, w_)) {
            std::fprintf(stderr, "[laguna] full reload failed: %s\n", dflash27b_last_error());
            return false;
        }
        return true;
    }

    // Step 5: Load expert data from GGUF mmap into hot/cold split buffers
    if (!build_hybrid_storage_from_file(placement, moe_hybrid_, err)) {
        std::fprintf(stderr, "[laguna-hybrid] storage build failed: %s\n", err.c_str());
        return false;
    }

    // Print stats
    int total_cold = 0;
    uint64_t hot_bytes = 0, cold_bytes = 0;
    for (int il = w_.n_layer_dense_lead; il < w_.n_layer; ++il) {
        const auto & layer = moe_hybrid_->layers[(size_t)il];
        total_cold += (int)layer.cold_expert_ids.size();
        const uint64_t per_expert_bytes =
            (uint64_t)layer.gate_expert_bytes + (uint64_t)layer.up_expert_bytes + (uint64_t)layer.down_expert_bytes;
        hot_bytes  += per_expert_bytes * (uint64_t)layer.hot_expert_ids.size();
        cold_bytes += per_expert_bytes * (uint64_t)layer.cold_expert_ids.size();
    }
    std::printf("[laguna-hybrid] storage ready: total_hot=%d (%.2f GiB VRAM) total_cold=%d (%.2f GiB RAM) source=%s\n",
                placement.total_hot,
                hot_bytes / 1024.0 / 1024.0 / 1024.0,
                total_cold,
                cold_bytes / 1024.0 / 1024.0 / 1024.0,
                placement_source.c_str());

    if (total_cold > 0) {
        hybrid_mode_ = true;
        std::printf("[laguna-hybrid] hybrid decode path active (%d cold experts)\n", total_cold);

        // Initialize streaming engine for prefill
        if (moe_hybrid_->has_mmap()) {
            size_t max_expert_bytes = 0;
            for (int il = 0; il < w_.n_layer; ++il) {
                const auto & layer = moe_hybrid_->layers[(size_t)il];
                size_t eb = layer.fused_gate_up
                    ? (size_t)layer.gate_up_expert_bytes + (size_t)layer.down_expert_bytes
                    : (size_t)layer.gate_expert_bytes + (size_t)layer.up_expert_bytes
                      + (size_t)layer.down_expert_bytes;
                if (eb > max_expert_bytes) max_expert_bytes = eb;
            }
            std::string stream_err;
            if (stream_engine_.init(backend_, max_expert_bytes, &stream_err)) {
                std::printf("[laguna-hybrid] stream engine ready: pinned=%.1f MiB scratch=%.1f MiB\n",
                            stream_engine_.pinned_bytes() / 1024.0 / 1024.0,
                            stream_engine_.scratch_bytes() / 1024.0 / 1024.0);
            } else {
                std::fprintf(stderr, "[laguna-hybrid] stream engine init failed: %s (prefill will use CPU fallback)\n",
                            stream_err.c_str());
            }
        }
    } else {
        hybrid_mode_ = true;  // partial load: expert tensors only in hybrid storage
        std::printf("[laguna-hybrid] all experts hot — using hybrid path (all-hot)\n");
    }

    // Configure telemetry and swap policy
    if (const char * telemetry = std::getenv("DFLASH_LAGUNA_TELEMETRY")) {
        hybrid_telemetry_ = std::atoi(telemetry) != 0;
    }
    if (const char * out_path = std::getenv("DFLASH_LAGUNA_NEXT_PLACEMENT_OUT")) {
        routing_stats_out_path_ = out_path;
    }
    if (const char * swap_max = std::getenv("DFLASH_LAGUNA_SWAP_MAX")) {
        swap_policy_.max_swaps_total = std::max(0, std::atoi(swap_max));
    }
    if (const char * swap_gain = std::getenv("DFLASH_LAGUNA_SWAP_MIN_GAIN")) {
        swap_policy_.min_promote_gain = (uint64_t)std::max(1, std::atoi(swap_gain));
    }

    // Allocate routing stats collector
    // Allocate routing stats if we either dump a profile OR run online swap
    // (post-request swap needs observed frequencies to build a swap plan).
    if (!routing_stats_out_path_.empty() || swap_policy_.max_swaps_total > 0) {
        routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
        routing_stats_->n_layer = w_.n_layer;
        routing_stats_->n_expert = w_.n_expert;
        routing_stats_->n_expert_used = w_.n_expert_used;
        // Spark: seed the live accumulator from the loaded profile so calibration
        // accumulates across restarts instead of resetting to zero each boot.
        if (hotness_path && hotness_path[0] &&
            hotness.counts.size() == (size_t)w_.n_layer * (size_t)w_.n_expert) {
            routing_stats_->counts = hotness.counts;
        } else {
            routing_stats_->counts.assign((size_t)w_.n_layer * (size_t)w_.n_expert, 0);
        }
        routing_stats_->layer_totals.assign((size_t)w_.n_layer, 0);
        for (int il = 0; il < w_.n_layer; ++il)
            for (int ie = 0; ie < w_.n_expert; ++ie)
                routing_stats_->layer_totals[(size_t)il] +=
                    routing_stats_->counts[(size_t)il * (size_t)w_.n_expert + ie];
    }

    std::fflush(stdout);
    return true;
}

// ── Laguna hybrid per-layer pre-FFN graph ───────────────────────────────
//
// Builds attention + router for a single layer. For MoE layers, outputs:
//   sg.ffn_post     = post-attention normed hidden (input to FFN)
//   sg.ffn_residual = residual to add after FFN output
//   sg.moe_selected = [n_used] expert IDs
//   sg.moe_weights  = [n_used] combine weights
// For the dense layer 0, outputs the full layer result in sg.hidden_input.

static bool build_laguna_layer_prefn_step(
    StepGraph & sg,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    ggml_backend_t backend,
    int il,
    int kv_start,
    int n_tokens,
    const dflash::common::MoeHybridLayerStorage * hot_storage = nullptr)
{
    step_graph_free(sg);

    const int n_embd = w.n_embd;
    const bool is_full = laguna_is_full_attn_layer(w, il);
    const bool is_dense = (il < w.n_layer_dense_lead);
    const LagunaTargetLayer & L = w.layers[(size_t)il];
    const int kv_len = kv_start + n_tokens;
    const int n_head = w.n_head_arr[il];
    const int n_head_kv = w.n_head_kv;
    const int head_dim = w.head_dim;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead() + 8 * 1024 * 1024;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;
    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

    // Input: hidden state [n_embd, n_tokens]
    sg.inp_embed = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_name(sg.inp_embed, "inp_embed");

    // Positions
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(sg.positions);

    // Attention mask (causal)
    ggml_tensor * attn_mask = nullptr;
    if (kv_len > 0) {
        attn_mask = ggml_new_tensor_4d(sg.ctx, GGML_TYPE_F32, kv_len, n_tokens, 1, 1);
        ggml_set_input(attn_mask);
        sg.attn_mask = attn_mask;
    }

    ggml_tensor * inp = sg.inp_embed;

    // Pre-attn RMS norm
    ggml_tensor * cur = ggml_rms_norm(sg.ctx, inp, 1e-6f);
    cur = ggml_mul(sg.ctx, cur, L.attn_norm);

    // QKV projections
    const int q_dim = n_head * head_dim;
    ggml_tensor * Qcur = ggml_mul_mat(sg.ctx, L.wq, cur);  // [q_dim, n_tokens]
    ggml_tensor * Kcur = ggml_mul_mat(sg.ctx, L.wk, cur);  // [n_head_kv * head_dim, n_tokens]
    ggml_tensor * Vcur = ggml_mul_mat(sg.ctx, L.wv, cur);  // [n_head_kv * head_dim, n_tokens]

    // Per-head softplus gate
    ggml_tensor * gate = ggml_mul_mat(sg.ctx, L.wqkv_gate, cur);  // [n_head, n_tokens]
    gate = ggml_softplus(sg.ctx, gate);

    // Reshape Q to [head_dim, n_head, n_tokens]
    Qcur = ggml_reshape_3d(sg.ctx, Qcur, head_dim, n_head, n_tokens);
    Kcur = ggml_reshape_3d(sg.ctx, Kcur, head_dim, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(sg.ctx, Vcur, head_dim, n_head_kv, n_tokens);

    // Q-norm / K-norm
    Qcur = ggml_rms_norm(sg.ctx, Qcur, 1e-6f);
    Qcur = ggml_mul(sg.ctx, Qcur, L.q_norm);
    Kcur = ggml_rms_norm(sg.ctx, Kcur, 1e-6f);
    Kcur = ggml_mul(sg.ctx, Kcur, L.k_norm);

    // RoPE (YaRN on full-attention layers, plain on SWA layers)
    const float rope_th     = is_full ? w.rope_freq_base_full : w.rope_freq_base_swa;
    const int   n_rot       = is_full ? w.n_rot_full : w.n_rot_swa;
    const float ext_factor  = is_full ? 1.0f : 0.0f;
    const float attn_factor = 1.0f;
    const float beta_fast   = is_full ? w.yarn_beta_fast : 32.0f;
    const float beta_slow   = is_full ? w.yarn_beta_slow :  1.0f;
    const int   n_ctx_orig  = is_full ? w.yarn_orig_ctx  : 0;
    const float freq_scale  = is_full ? (1.0f / w.yarn_factor) : 1.0f;

    Qcur = ggml_rope_ext(sg.ctx, Qcur, sg.positions, /*freq_factors=*/nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX,
                          n_ctx_orig, rope_th, freq_scale,
                          ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(sg.ctx, Kcur, sg.positions, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX,
                          n_ctx_orig, rope_th, freq_scale,
                          ext_factor, attn_factor, beta_fast, beta_slow);

    // KV cache write — permute to [head_dim, n_tokens, n_head_kv] layout
    ggml_tensor * cache_k = cache.attn_k[(size_t)il];
    ggml_tensor * cache_v = cache.attn_v[(size_t)il];

    ggml_tensor * Kcur_T = ggml_permute(sg.ctx, Kcur, 0, 2, 1, 3);
    ggml_tensor * Vcur_T = ggml_permute(sg.ctx, Vcur, 0, 2, 1, 3);

    ggml_tensor * k_view = ggml_view_3d(sg.ctx, cache_k,
        head_dim, n_tokens, n_head_kv,
        cache_k->nb[1], cache_k->nb[2],
        cache_k->nb[1] * (size_t)kv_start);
    ggml_tensor * k_cpy = ggml_cpy(sg.ctx, Kcur_T, k_view);
    ggml_build_forward_expand(sg.gf, k_cpy);

    ggml_tensor * v_view = ggml_view_3d(sg.ctx, cache_v,
        head_dim, n_tokens, n_head_kv,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * (size_t)kv_start);
    ggml_tensor * v_cpy = ggml_cpy(sg.ctx, Vcur_T, v_view);
    ggml_build_forward_expand(sg.gf, v_cpy);

    // Flash attention
    ggml_tensor * Qfa = ggml_permute(sg.ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(sg.ctx, Qfa);

    ggml_tensor * Kfa = ggml_view_3d(sg.ctx, cache_k,
        head_dim, kv_len, n_head_kv,
        cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * Vfa = ggml_view_3d(sg.ctx, cache_v,
        head_dim, kv_len, n_head_kv,
        cache_v->nb[1], cache_v->nb[2], 0);

    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn_mask_f16 = attn_mask ? ggml_cast(sg.ctx, attn_mask, GGML_TYPE_F16) : nullptr;
    ggml_tensor * attn = ggml_flash_attn_ext(sg.ctx, Qfa, Kfa, Vfa, attn_mask_f16,
                                              kq_scale, 0.0f, 0.0f);

    // Per-head softplus gate
    ggml_tensor * gate_b = ggml_reshape_3d(sg.ctx, gate, 1, n_head, n_tokens);
    gate_b = ggml_cast(sg.ctx, gate_b, attn->type);
    attn = ggml_mul(sg.ctx, attn, gate_b);

    attn = ggml_reshape_2d(sg.ctx, attn, q_dim, n_tokens);

    // Output projection
    ggml_tensor * attn_out = ggml_mul_mat(sg.ctx, L.wo, attn);  // [n_embd, n_tokens]

    // Residual after attention
    ggml_tensor * ffn_inp = ggml_add(sg.ctx, attn_out, inp);

    if (is_dense) {
        // Dense layer 0: run full MLP in this graph
        ggml_tensor * normed = ggml_rms_norm(sg.ctx, ffn_inp, 1e-6f);
        normed = ggml_mul(sg.ctx, normed, L.ffn_norm);

        ggml_tensor * g = ggml_mul_mat(sg.ctx, L.w_gate, normed);
        ggml_tensor * u = ggml_mul_mat(sg.ctx, L.w_up, normed);
        ggml_tensor * gu = ggml_swiglu_split(sg.ctx, g, u);
        ggml_tensor * d = ggml_mul_mat(sg.ctx, L.w_down, gu);
        ggml_tensor * layer_out = ggml_add(sg.ctx, d, ffn_inp);

        sg.hidden_input = layer_out;
        ggml_set_output(layer_out);
        ggml_build_forward_expand(sg.gf, layer_out);
    } else {
        // MoE layer: output pre-FFN normed + residual + router decisions
        ggml_tensor * normed = ggml_rms_norm(sg.ctx, ffn_inp, 1e-6f);
        normed = ggml_mul(sg.ctx, normed, L.ffn_norm);
        sg.ffn_post = normed;
        ggml_set_output(normed);

        sg.ffn_residual = ffn_inp;
        ggml_set_output(ffn_inp);

        // Router: sigmoid + score-correction bias + top-k
        ggml_tensor * router_logits = ggml_mul_mat(sg.ctx, L.ffn_gate_inp, normed);
        ggml_tensor * probs = ggml_sigmoid(sg.ctx, router_logits);
        ggml_tensor * scores_sel = ggml_add(sg.ctx, probs, L.ffn_exp_probs_b);
        ggml_tensor * selected = ggml_top_k(sg.ctx, scores_sel, w.n_expert_used);
        ggml_set_output(selected);

        // Gather original probs (no bias) for combine weights
        ggml_tensor * probs_3d = ggml_reshape_3d(sg.ctx, probs, 1, w.n_expert, n_tokens);
        ggml_tensor * weights_raw = ggml_get_rows(sg.ctx, probs_3d, selected);
        weights_raw = ggml_reshape_2d(sg.ctx, weights_raw, w.n_expert_used, n_tokens);

        // Sum-normalize + scale
        ggml_tensor * w_sum = ggml_sum_rows(sg.ctx, weights_raw);
        ggml_tensor * weights_normed = ggml_div(sg.ctx, weights_raw, w_sum);
        if (w.expert_weights_scale != 1.0f) {
            weights_normed = ggml_scale(sg.ctx, weights_normed, w.expert_weights_scale);
        }
        sg.moe_weights = weights_normed;
        sg.moe_selected.resize(1);
        sg.moe_selected[0] = selected;

        static const bool g_fuse = (std::getenv("DFLASH_LAGUNA_FUSE_FFN") != nullptr);
        if (hot_storage && g_fuse && hot_storage->gate_hot) {
            // Fused routed FFN in-graph (mirrors gpu_remap), drop-on-miss via valid_lut.
            MoeLayerDesc d = make_moe_layer_desc(w.layers[(size_t)il]);
            const int nu = w.n_expert_used;
            sg.hot_local_lut = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_I32, 1, w.n_expert); ggml_set_input(sg.hot_local_lut);
            sg.valid_lut     = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, 1, w.n_expert); ggml_set_input(sg.valid_lut);
            ggml_tensor * lid = ggml_get_rows(sg.ctx, sg.hot_local_lut, selected);
            ggml_tensor * ids = ggml_cont(sg.ctx, ggml_reshape_2d(sg.ctx, lid, nu, 1));
            ggml_tensor * vm  = ggml_reshape_2d(sg.ctx, ggml_get_rows(sg.ctx, sg.valid_lut, selected), nu, 1);
            ggml_tensor * wmask = ggml_mul(sg.ctx, weights_normed, vm);
            ggml_tensor * cur3 = ggml_reshape_3d(sg.ctx, normed, w.n_embd, 1, 1);
            auto SC = [&](ggml_tensor * t, float s){ return s != 1.0f ? ggml_scale(sg.ctx, t, s) : t; };
            ggml_tensor * ge = SC(ggml_mul_mat_id(sg.ctx, hot_storage->gate_hot, cur3, ids), d.ffn_gate_exps_s);
            ggml_tensor * ue = SC(ggml_mul_mat_id(sg.ctx, hot_storage->up_hot,   cur3, ids), d.ffn_up_exps_s);
            ggml_tensor * gu = ggml_swiglu_split(sg.ctx, ge, ue);
            ggml_tensor * ex = SC(ggml_mul_mat_id(sg.ctx, hot_storage->down_hot, gu, ids), d.ffn_down_exps_s);
            ex = ggml_mul(sg.ctx, ex, ggml_reshape_3d(sg.ctx, wmask, 1, nu, 1));
            ggml_tensor * routed = nullptr;
            for (int i = 0; i < nu; ++i) {
                ggml_tensor * sl = ggml_view_2d(sg.ctx, ex, w.n_embd, 1, ex->nb[2], (size_t)i * ex->nb[1]);
                routed = (i == 0) ? sl : ggml_add(sg.ctx, routed, sl);
            }
            ggml_tensor * shared = nullptr;
            if (d.has_shared_expert()) {
                ggml_tensor * shg = SC(ggml_mul_mat(sg.ctx, d.ffn_gate_shexp, normed), d.ffn_gate_shexp_s);
                ggml_tensor * shu = SC(ggml_mul_mat(sg.ctx, d.ffn_up_shexp,   normed), d.ffn_up_shexp_s);
                ggml_tensor * shgu = ggml_swiglu_split(sg.ctx, shg, shu);
                shared = SC(ggml_mul_mat(sg.ctx, d.ffn_down_shexp, shgu), d.ffn_down_shexp_s);
                if (d.ffn_gate_inp_shexp) {
                    ggml_tensor * g2 = ggml_sigmoid(sg.ctx, SC(ggml_mul_mat(sg.ctx, d.ffn_gate_inp_shexp, normed), d.ffn_gate_inp_shexp_s));
                    shared = ggml_mul(sg.ctx, shared, g2);
                }
            }
            ggml_tensor * out = shared ? (routed ? ggml_add(sg.ctx, routed, shared) : shared) : routed;
            ggml_tensor * layer_out = ggml_cont(sg.ctx, ggml_add(sg.ctx, out, ffn_inp));
            sg.hidden_input = layer_out;
            ggml_set_output(layer_out);
            ggml_build_forward_expand(sg.gf, layer_out);
        } else {
            ggml_set_output(weights_normed);
            ggml_build_forward_expand(sg.gf, normed);
            ggml_build_forward_expand(sg.gf, ffn_inp);
            ggml_build_forward_expand(sg.gf, selected);
            ggml_build_forward_expand(sg.gf, weights_normed);
        }
    }

    // Allocate
    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) {
        return false;
    }

    return true;
}

// ── Hybrid forward: one token through all 40 layers ─────────────────────

bool LagunaBackend::hybrid_forward_one_token(int32_t tok, int kv_pos,
                                              std::vector<float> & act_cur,
                                              int32_t & argmax_out) {
    const int hidden = w_.n_embd;
    const int vocab = w_.embedder.n_vocab;
    using _pclk = std::chrono::steady_clock;
    const bool _prof = std::getenv("DFLASH_LAGUNA_PROFILE") != nullptr;
    auto _pnow = []{ return std::chrono::steady_clock::now(); };
    auto _pus = [](_pclk::time_point a, _pclk::time_point b){ return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); };
    static uint64_t g_total=0, g_ffn=0, g_logits=0, g_build=0, g_compute=0, g_calls=0;
    static uint64_t g_cold_experts=0, g_expert_layers=0;
    const auto _t_start = _prof ? _pnow() : _pclk::time_point{};

    // Embed token
    if (!w_.embedder.embed(&tok, 1, act_cur.data())) return false;

    // Single-graph hybrid decode: whole token in one graph (residency LUTs
    // set once), instead of 40 per-layer graphs. Removes the per-layer host
    // glue that caps the multi-graph path. Default ON for the hybrid-offload
    // path; set DFLASH_LAGUNA_NO_SINGLE_GRAPH=1 to fall back to per-layer decode.
    static const bool g_single_graph = (std::getenv("DFLASH_LAGUNA_NO_SINGLE_GRAPH") == nullptr);
    if (g_single_graph && moe_hybrid_) {
        static const bool _nm = (std::getenv("DFLASH_NO_MASK") != nullptr);
        static std::vector<float> _sg_logits;
        static std::vector<int32_t> _sg_sel;
        if (!kvflash_alloc_span(kv_pos, 1)) return false;
        if (!laguna_step_hybrid(backend_, w_, cache_, act_cur.data(), 1, kv_pos, _nm,
                                *moe_hybrid_, _sg_logits, &_sg_sel,
                                kvflash_active() ? &kvflash_pager_ : nullptr))
            return false;
        // Reactive cache warm + routing observe, POST-compute (off the
        // single-graph critical path): make each selected expert resident
        // for the next token; over warmup drops fall to ~0 -> exact decode.
        {
            const int _nu = w_.n_expert_used;
            static uint64_t _sg_cold = 0, _sg_calls = 0;
            static const bool _sg_pf = (std::getenv("DFLASH_LAGUNA_PROFILE") != nullptr);
            uint64_t _cold_this = 0;
            for (int il = w_.n_layer_dense_lead; il < w_.n_layer; ++il) {
                const int32_t * _sl = _sg_sel.data() + (size_t)il * _nu;
                if (routing_stats_) routing_stats_->observe(il, _sl, _nu);
                auto & _cst = moe_hybrid_->layers[(size_t)il];
                for (int k = 0; k < _nu; ++k) {
                    const int _g = _sl[k];
                    if (_g < 0) continue;
                    // Was this expert resident when the graph computed? (pre-swap residency)
                    if (_g < (int)_cst.hot_local_by_global.size() &&
                        _cst.hot_local_by_global[(size_t)_g] < 0)
                        _cold_this++;
                    if (_cst.cache_slots > 0)
                        dflash::common::moe_hybrid_cache_swap_in(_cst, _g, backend_);
                }
            }
            if (_sg_pf) {
                _sg_cold += _cold_this;
                if (++_sg_calls % 32 == 0) {
                    std::fprintf(stderr, "[sg-prof] cold_experts/tok=%.2f (over 32)\n", _sg_cold / 32.0);
                    _sg_cold = 0;
                }
            }
        }
        int _best = 0; float _bv = _sg_logits[0];
        for (size_t i = 1; i < _sg_logits.size(); ++i)
            if (_sg_logits[i] > _bv) { _bv = _sg_logits[i]; _best = (int)i; }
        argmax_out = _best;
        return true;
    }

    // GPU-resident state for MoE layers
    GpuResidentState gpu_state;
    // The per-layer fallback writes KV at literal view offsets (no set_rows),
    // which a kvflash pool cannot express once chunks relocate.
    if (kvflash_active()) {
        std::fprintf(stderr, "[kvflash] laguna per-layer hybrid decode is not "
                             "pool-aware; unset DFLASH_LAGUNA_NO_SINGLE_GRAPH\n");
        return false;
    }

    if (!init_gpu_resident_state(gpu_state, backend_, hidden)) return false;
    ggml_backend_tensor_set(gpu_state.act_cur, act_cur.data(), 0, sizeof(float) * (size_t)hidden);

    StepGraph layer_sg;
    std::vector<int32_t> selected((size_t)w_.n_expert_used);
    std::vector<float> weights_buf((size_t)w_.n_expert_used);
    ggml_backend_t cpu_be = moe_hybrid_->cpu_backend;

    for (int il = 0; il < w_.n_layer; ++il) {
        const bool is_dense = (il < w_.n_layer_dense_lead);

        const auto _t_b = _prof ? _pnow() : _pclk::time_point{};
        if (!build_laguna_layer_prefn_step(layer_sg, w_, cache_, backend_, il, kv_pos, 1, &moe_hybrid_->layers[(size_t)il])) {
            step_graph_destroy(layer_sg);
            gpu_state.destroy();
            return false;
        }
        if (_prof) g_build += _pus(_t_b, _pnow());

        // GPU→GPU: copy persistent act_cur to pre-FFN graph input
        ggml_backend_tensor_copy(gpu_state.act_cur, layer_sg.inp_embed);

        // Set positions
        int32_t pos_val = kv_pos;
        ggml_backend_tensor_set(layer_sg.positions, &pos_val, 0, sizeof(int32_t));

        // Causal mask: single token decode — all positions [0..kv_pos] visible
        if (layer_sg.attn_mask) {
            const int kv_len = kv_pos + 1;
            std::vector<float> mask_data((size_t)kv_len, 0.0f);
            ggml_backend_tensor_set(layer_sg.attn_mask, mask_data.data(), 0, sizeof(float) * (size_t)kv_len);
        }

        static const bool g_fuse_dec = (std::getenv("DFLASH_LAGUNA_FUSE_FFN") != nullptr);
        if (g_fuse_dec && !is_dense && layer_sg.hot_local_lut) {
            auto & _st = moe_hybrid_->layers[(size_t)il];
            std::vector<int32_t> _lut((size_t)w_.n_expert); std::vector<float> _vld((size_t)w_.n_expert);
            for (int g = 0; g < w_.n_expert; ++g) {
                int loc = (g < (int)_st.hot_local_by_global.size()) ? _st.hot_local_by_global[(size_t)g] : -1;
                _lut[(size_t)g] = loc >= 0 ? loc : 0;
                _vld[(size_t)g] = loc >= 0 ? 1.0f : 0.0f;
            }
            ggml_backend_tensor_set(layer_sg.hot_local_lut, _lut.data(), 0, sizeof(int32_t)*(size_t)w_.n_expert);
            ggml_backend_tensor_set(layer_sg.valid_lut, _vld.data(), 0, sizeof(float)*(size_t)w_.n_expert);
        }
        const auto _t_c = _prof ? _pnow() : _pclk::time_point{};
        auto st = ggml_backend_graph_compute(backend_, layer_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            step_graph_destroy(layer_sg);
            gpu_state.destroy();
            return false;
        }
        if (_prof) g_compute += _pus(_t_c, _pnow());

        if (is_dense) {
            // Dense layer: read full output back to GPU-resident state
            ggml_backend_tensor_copy(layer_sg.hidden_input, gpu_state.act_cur);
        } else if (g_fuse_dec && layer_sg.hidden_input) {
            // Fused FFN computed in-graph -> layer output is hidden_input
            ggml_backend_tensor_copy(layer_sg.hidden_input, gpu_state.act_cur);
            // Warm the expert cache + observe routing so coverage rises (drops -> 0
            // after warmup) and calibration still accumulates in the fused path.
            {
                ggml_backend_tensor_get(layer_sg.moe_selected[0], selected.data(), 0,
                                        sizeof(int32_t) * selected.size());
                if (routing_stats_) routing_stats_->observe(il, selected.data(), (int)selected.size());
                auto & _cst = moe_hybrid_->layers[(size_t)il];
                if (_cst.cache_slots > 0)
                    for (int _k = 0; _k < (int)selected.size(); ++_k)
                        dflash::common::moe_hybrid_cache_swap_in(_cst, selected[(size_t)_k], backend_);
            }
        } else {
            // MoE layer: read router decisions, then do hybrid FFN eval
            ggml_tensor * sel_tensor = layer_sg.moe_selected[0];
            ggml_backend_tensor_get(sel_tensor, selected.data(), 0,
                                     sizeof(int32_t) * selected.size());
            ggml_backend_tensor_get(layer_sg.moe_weights, weights_buf.data(), 0,
                                     sizeof(float) * weights_buf.size());

            if (routing_stats_) {
                routing_stats_->observe(il, selected.data(), (int)selected.size());
            }

            // Pre-gate trace capture: (block-input hidden, selected experts) per
            // MoE layer. hidden = gpu_state.act_cur BEFORE this layer's FFN, i.e.
            // exactly the signal available one step early at inference. Offline,
            // train a predictor that prefetches experts to enable graph fusion.
            {
                static FILE * g_trace = nullptr;
                static int64_t g_trace_n = 0, g_trace_max = 0, g_trace_flush = 0;
                static bool g_trace_init = false;
                if (!g_trace_init) {
                    g_trace_init = true;
                    if (const char * tp = std::getenv("DFLASH_LAGUNA_PREGATE_TRACE")) {
                        g_trace = std::fopen(tp, "wb");
                        g_trace_max = 100000;
                        if (const char * mx = std::getenv("DFLASH_LAGUNA_PREGATE_MAX"))
                            g_trace_max = std::atoll(mx);
                        if (g_trace) std::fprintf(stderr, "[lag-pregate] tracing -> %s (max %lld n_embd=%d)\n", tp, (long long)g_trace_max, hidden);
                    }
                }
                if (g_trace && g_trace_n < g_trace_max) {
                    static std::vector<float> _hbuf;
                    _hbuf.resize((size_t)hidden);
                    ggml_backend_tensor_get(gpu_state.act_cur, _hbuf.data(), 0, sizeof(float)*(size_t)hidden);
                    int16_t hdr[2] = { (int16_t)il, (int16_t)selected.size() };
                    int32_t sel8[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
                    for (int _k=0;_k<(int)selected.size() && _k<8;++_k) sel8[_k]=selected[(size_t)_k];
                    std::fwrite(hdr, sizeof(hdr), 1, g_trace);
                    std::fwrite(sel8, sizeof(sel8), 1, g_trace);
                    std::fwrite(_hbuf.data(), sizeof(float), (size_t)hidden, g_trace);
                    if (++g_trace_n - g_trace_flush >= 2000) { std::fflush(g_trace); g_trace_flush = g_trace_n; }
                    if (g_trace_n == g_trace_max) { std::fflush(g_trace); std::fprintf(stderr, "[lag-pregate] trace full (%lld)\n", (long long)g_trace_n); }
                }
            }

            // Collect routing data for predictor training
            if (routing_collector_) {
                std::vector<float> gate_input((size_t)w_.n_embd);
                ggml_backend_tensor_get(layer_sg.ffn_post, gate_input.data(), 0,
                                         sizeof(float) * (size_t)w_.n_embd);
                routing_collector_->record(il, gate_input.data(), w_.n_embd,
                                           selected.data(), (int)selected.size());
            }

            // Hybrid FFN: hot on GPU, cold on CPU, combine on GPU
            auto & storage = moe_hybrid_->layers[(size_t)il];
            { int _lc=0; for (int _k=0;_k<(int)selected.size();++_k){ int _g=selected[(size_t)_k];
                if (_g>=0 && _g<(int)storage.hot_local_by_global.size() && storage.hot_local_by_global[(size_t)_g]<0 && storage.cold_local_by_global[(size_t)_g]>=0) { _lc++; } }
              if (_prof){ g_cold_experts+=(uint64_t)_lc; if(_lc>0) g_expert_layers++; } }

            MoeHybridConfig cfg = make_moe_hybrid_config(w_);
            MoeLayerDesc desc = make_moe_layer_desc(w_.layers[(size_t)il]);
            MoeExpertCompute * expert_compute = expert_runtime_.compute_ptr();
            const MoeExpertLayer * expert_layer =
                expert_runtime_.layer_ptr((size_t)il);
            const auto _t_ffn = _prof ? _pnow() : _pclk::time_point{};
            if (!eval_moe_hybrid_ffn_gpu_resident(
                    backend_, cfg, desc, storage, cpu_be,
                    layer_sg.ffn_post, layer_sg.ffn_residual,
                    gpu_state,
                    selected.data(), weights_buf.data(),
                    (int)selected.size(), expert_compute, expert_layer)) {
                step_graph_destroy(layer_sg);
                gpu_state.destroy();
                return false;
            }
            if (_prof) g_ffn += _pus(_t_ffn, _pnow());
        }
    }

    // Read final hidden state and project logits
    ggml_backend_tensor_get(gpu_state.act_cur, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
    step_graph_destroy(layer_sg);
    gpu_state.destroy();

    const auto _t_logits = _prof ? _pnow() : _pclk::time_point{};
    // Project logits: final RMS norm + lm_head
    {
        ggml_init_params ip{};
        ip.mem_size = 64 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;

        ggml_tensor * h_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
        ggml_set_input(h_in);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);

        ggml_tensor * normed = ggml_rms_norm(ctx, h_in, 1e-6f);
        normed = ggml_mul(ctx, normed, w_.out_norm);
        ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, normed);
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(h_in, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
        if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }

        std::vector<float> logits_buf((size_t)vocab);
        ggml_backend_tensor_get(logits, logits_buf.data(), 0, sizeof(float) * (size_t)vocab);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);

        // Argmax
        argmax_out = 0;
        float best = logits_buf[0];
        for (int j = 1; j < vocab; ++j) {
            if (logits_buf[(size_t)j] > best) {
                best = logits_buf[(size_t)j];
                argmax_out = j;
            }
        }
    }
    if (_prof) {
        g_logits += _pus(_t_logits, _pnow());
        g_total  += _pus(_t_start, _pnow());
        if (++g_calls % 32 == 0) {
            const double n = 32.0;
            const double tot = g_total/n/1000.0, ffn = g_ffn/n/1000.0, lg = g_logits/n/1000.0;
            const double bld = g_build/n/1000.0, cmp = g_compute/n/1000.0;
            std::fprintf(stderr, "[lag-prof] avg/tok over 32: total=%.2f ms  prefn=%.2f (build=%.2f compute=%.2f)  ffn=%.2f  logits=%.2f  cold_experts/tok=%.1f expert_layers/tok=%.1f\n",
                         tot, tot-ffn-lg, bld, cmp, ffn, lg, g_cold_experts/n, g_expert_layers/n);
            g_total=g_ffn=g_logits=g_build=g_compute=0; g_cold_experts=g_expert_layers=0;
        }
    }
    return true;
}

bool LagunaBackend::ensure_moe_expert_compute() {
    if (!moe_hybrid_) return false;
    std::vector<MoeLayerDesc> layer_descs((size_t)w_.n_layer);
    for (int il = 0; il < w_.n_layer; ++il) {
        layer_descs[(size_t)il] = make_moe_layer_desc(w_.layers[(size_t)il]);
    }
    MoeExpertComputeRuntimeConfig runtime_cfg;
    runtime_cfg.target_path = args_.target_path;
    runtime_cfg.n_layer = w_.n_layer;
    runtime_cfg.n_expert = w_.n_expert;
    runtime_cfg.n_expert_used = w_.n_expert_used;
    runtime_cfg.n_embd = w_.n_embd;
    runtime_cfg.n_ff_exp = w_.n_ff_exp;
    runtime_cfg.enabled = true;
    runtime_cfg.log_prefix = "[laguna-hybrid]";
    std::string err;
    if (!ensure_moe_expert_compute_runtime(expert_runtime_, runtime_cfg,
                                           *moe_hybrid_, layer_descs, &err)) {
        return false;
    }
    return true;
}

// ── Hybrid generate ─────────────────────────────────────────────────────

GenerateResult LagunaBackend::generate_hybrid(const GenerateRequest & req,
                                               const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    const bool should_emit = req.stream || (bool)out_io.on_token;
    const int N = (int)req.prompt.size();

    if (N + req.n_gen > args_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow);
        return result;
    }

    // kvflash: hybrid prefill writes rows identity-mapped (legacy per-layer
    // views), so the prompt must fit the pool; the pager mapping is built up
    // front and stays identity through prefill (no eviction can trigger).
    if (kvflash_active() &&
        N > kvflash_tokens_ - kvflash_pager_.chunk_tokens()) {
        std::fprintf(stderr, "[kvflash] hybrid prompt (%d) exceeds pool %d; "
                             "raise --kvflash\n", N, kvflash_tokens_);
        result.fail(GenerateErrorCode::ContextOverflow);
        return result;
    }

    reset_laguna_target_cache(cache_);
    if (kvflash_active()) {
        kvflash_pager_.reset();
        if (!kvflash_alloc_span(0, N)) {
            result.fail(GenerateErrorCode::BackendSpecific, "kvflash_slot");
            return result;
        }
    }
    if (!ensure_moe_expert_compute()) {
        result.fail(GenerateErrorCode::BackendSpecific, "moe_expert_compute");
        return result;
    }

    // ── Hybrid Prefill: layer-by-layer pre-FFN + batched hybrid FFN ──
    const int hidden = w_.n_embd;
    const int n_expert_used = w_.n_expert_used;
    ggml_backend_t cpu_be = moe_hybrid_->cpu_backend;

    std::vector<float> embed_all((size_t)N * (size_t)hidden);
    if (!w_.embedder.embed(req.prompt.data(), N, embed_all.data())) {
        result.fail(GenerateErrorCode::BackendSpecific, "embed_prefill");
        return result;
    }

    auto t_pf0 = std::chrono::steady_clock::now();
    const int prefill_chunk = std::min(args_.chunk, N);

    StepGraph prefill_sg;  // persistent across layers to reuse GPU buffer
    ggml_gallocr_t ffn_hot_alloc = nullptr;
    ggml_gallocr_t ffn_cold_alloc = nullptr;

    for (int il = 0; il < w_.n_layer; ++il) {
        const bool is_dense = (il < w_.n_layer_dense_lead);
        const bool is_full = laguna_is_full_attn_layer(w_, il);

        for (int chunk_start = 0; chunk_start < N; chunk_start += prefill_chunk) {
            const int chunk_len = std::min(prefill_chunk, N - chunk_start);

            step_graph_free(prefill_sg);  // reset ctx/graph but keep gallocr buffer
            if (!build_laguna_layer_prefn_step(prefill_sg, w_, cache_, backend_,
                                               il, chunk_start, chunk_len)) {
                result.fail(GenerateErrorCode::BackendSpecific, "prefill_build");
                step_graph_destroy(prefill_sg);
                if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
                if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);
                return result;
            }

            // Set input embeddings
            ggml_backend_tensor_set(prefill_sg.inp_embed,
                                    embed_all.data() + (size_t)chunk_start * (size_t)hidden, 0,
                                    sizeof(float) * (size_t)chunk_len * (size_t)hidden);

            // Set positions
            std::vector<int32_t> pos_data((size_t)chunk_len);
            for (int i = 0; i < chunk_len; ++i) pos_data[i] = chunk_start + i;
            ggml_backend_tensor_set(prefill_sg.positions, pos_data.data(), 0,
                                    sizeof(int32_t) * (size_t)chunk_len);

            // Set attention mask (causal or causal+SWA depending on layer)
            if (prefill_sg.attn_mask) {
                const int kv_len = chunk_start + chunk_len;
                std::vector<float> mask((size_t)kv_len * (size_t)chunk_len, -INFINITY);
                for (int q = 0; q < chunk_len; ++q) {
                    const int abs_q = chunk_start + q;
                    const int win_lo = is_full ? 0 : std::max(0, abs_q - w_.sliding_window + 1);
                    for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                        mask[(size_t)q * (size_t)kv_len + (size_t)k] = 0.0f;
                    }
                }
                ggml_backend_tensor_set(prefill_sg.attn_mask, mask.data(), 0,
                                        sizeof(float) * mask.size());
            }

            // Compute pre-FFN graph
            auto st = ggml_backend_graph_compute(backend_, prefill_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                result.fail(GenerateErrorCode::BackendSpecific, "prefill_compute");
                step_graph_destroy(prefill_sg);
                if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
                if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);
                return result;
            }

            if (is_dense) {
                // Dense layer outputs full result directly
                std::vector<float> layer_out((size_t)chunk_len * (size_t)hidden);
                ggml_backend_tensor_get(prefill_sg.hidden_input, layer_out.data(), 0,
                                        sizeof(float) * layer_out.size());
                std::memcpy(embed_all.data() + (size_t)chunk_start * (size_t)hidden,
                            layer_out.data(),
                            sizeof(float) * layer_out.size());
            } else {
                // MoE layer: read router decisions, run hybrid FFN
                std::vector<float> chunk_residuals((size_t)chunk_len * (size_t)hidden);
                std::vector<float> chunk_post((size_t)chunk_len * (size_t)hidden);
                std::vector<int32_t> chunk_selected((size_t)chunk_len * (size_t)n_expert_used);
                std::vector<float> chunk_weights((size_t)chunk_len * (size_t)n_expert_used);

                ggml_backend_tensor_get(prefill_sg.ffn_residual, chunk_residuals.data(), 0,
                                        sizeof(float) * chunk_residuals.size());
                ggml_backend_tensor_get(prefill_sg.ffn_post, chunk_post.data(), 0,
                                        sizeof(float) * chunk_post.size());

                ggml_tensor * sel_tensor = prefill_sg.moe_selected.empty() ? nullptr : prefill_sg.moe_selected[0];
                if (!sel_tensor || !prefill_sg.moe_weights) {
                    result.fail(GenerateErrorCode::BackendSpecific,
                                     "prefill_router_outputs");
                    step_graph_destroy(prefill_sg);
                    if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
                    if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);
                    return result;
                }
                ggml_backend_tensor_get(sel_tensor, chunk_selected.data(), 0,
                                        sizeof(int32_t) * chunk_selected.size());
                ggml_backend_tensor_get(prefill_sg.moe_weights, chunk_weights.data(), 0,
                                        sizeof(float) * chunk_weights.size());

                // Observe routing stats
                if (routing_stats_) {
                    for (int i = 0; i < chunk_len; ++i) {
                        routing_stats_->observe(il, chunk_selected.data() + (size_t)i * (size_t)n_expert_used, n_expert_used);
                    }
                }

                // Collect routing data for predictor training (prefill path)
                if (routing_collector_) {
                    for (int i = 0; i < chunk_len; ++i) {
                        routing_collector_->record(
                            il,
                            chunk_post.data() + (size_t)i * (size_t)w_.n_embd,
                            w_.n_embd,
                            chunk_selected.data() + (size_t)i * (size_t)n_expert_used,
                            n_expert_used);
                    }
                }

                // Batched hybrid FFN evaluation
                auto & storage = moe_hybrid_->layers[(size_t)il];
                MoeHybridConfig chunk_cfg = make_moe_hybrid_config(w_);
                MoeLayerDesc chunk_desc = make_moe_layer_desc(w_.layers[(size_t)il]);
                MoeExpertCompute * expert_compute = expert_runtime_.compute_ptr();
                const MoeExpertLayer * expert_layer =
                    expert_runtime_.layer_ptr((size_t)il);
                std::vector<float> ffn_batch_out;
                std::string ffn_error;
                bool ffn_ok = false;

                if (storage.cold_expert_ids.empty()) {
                    // All experts hot — use batched path directly
                    ffn_ok = eval_moe_hybrid_ffn_batched(
                            backend_, cpu_be, chunk_cfg, chunk_desc, storage,
                            chunk_post.data(),
                            chunk_selected.data(),
                            chunk_weights.data(),
                            chunk_len, ffn_batch_out, &ffn_error,
                            &ffn_hot_alloc, &ffn_cold_alloc,
                            expert_compute, expert_layer);
                } else if (storage.all_routed_are_hot(chunk_selected.data(),
                                                      chunk_len * n_expert_used)) {
                    // All selected experts happen to be in VRAM — pure GPU, no CPU
                    ffn_ok = eval_moe_hot_only_batched(
                            backend_, chunk_cfg, chunk_desc, storage,
                            chunk_post.data(),
                            chunk_selected.data(),
                            chunk_weights.data(),
                            chunk_len, ffn_batch_out, &ffn_error,
                            &ffn_hot_alloc);
                } else if (moe_hybrid_->has_mmap() &&
                           !moe_hybrid_->layer_regions.empty() &&
                           stream_engine_.is_ready() && chunk_len >= 16 &&
                           !storage.cold_expert_ids.empty()) {
                    // Streaming prefill: prefetch cold data then batched eval (hot GPU + cold CPU)
                    const auto & regions = moe_hybrid_->layer_regions[(size_t)il];
                    std::vector<int32_t> cold_ids_copy(storage.cold_expert_ids.begin(),
                                                      storage.cold_expert_ids.end());
                    stream_engine_.prefetch_cold_experts(moe_hybrid_->mmap_data, moe_hybrid_->mmap_size,
                                                        regions, cold_ids_copy.data(),
                                                        (int)cold_ids_copy.size());
                    ffn_ok = eval_moe_hybrid_ffn_batched(
                            backend_, cpu_be, chunk_cfg, chunk_desc, storage,
                            chunk_post.data(),
                            chunk_selected.data(),
                            chunk_weights.data(),
                            chunk_len, ffn_batch_out, &ffn_error,
                            &ffn_hot_alloc, &ffn_cold_alloc,
                            expert_compute, expert_layer);
                } else {
                    // Fallback: batched eval handles both hot+cold (CPU for cold)
                    ffn_ok = eval_moe_hybrid_ffn_batched(
                            backend_, cpu_be, chunk_cfg, chunk_desc, storage,
                            chunk_post.data(),
                            chunk_selected.data(),
                            chunk_weights.data(),
                            chunk_len, ffn_batch_out, &ffn_error,
                            &ffn_hot_alloc, &ffn_cold_alloc,
                            expert_compute, expert_layer);
                }

                if (!ffn_ok) {
                    result.fail(GenerateErrorCode::BackendSpecific,
                                     std::move(ffn_error));
                    step_graph_destroy(prefill_sg);
                    if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
                    if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);
                    return result;
                }

                // Combine: FFN output + residual → embed_all for next layer
                for (int i = 0; i < chunk_len; ++i) {
                    const float * ffn = ffn_batch_out.data() + (size_t)i * (size_t)hidden;
                    const float * res = chunk_residuals.data() + (size_t)i * (size_t)hidden;
                    float * out_embed = embed_all.data() + (size_t)(chunk_start + i) * (size_t)hidden;
                    for (int j = 0; j < hidden; ++j) {
                        out_embed[j] = ffn[j] + res[j];
                    }
                }
            }
        }
    }
    step_graph_destroy(prefill_sg);
    if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
    if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);

    // Project logits from last token's hidden state
    cache_.cur_pos = N;
    std::vector<float> last_logits;
    {
        ggml_init_params ip{};
        ip.mem_size = 64 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);

        ggml_tensor * h_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
        ggml_set_input(h_in);
        ggml_tensor * normed = ggml_rms_norm(ctx, h_in, 1e-6f);
        normed = ggml_mul(ctx, normed, w_.out_norm);
        ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, normed);
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            result.fail(GenerateErrorCode::BackendSpecific,
                             "prefill_logits_alloc");
            return result;
        }
        // Set last token's hidden state
        ggml_backend_tensor_set(h_in,
                                embed_all.data() + (size_t)(N - 1) * (size_t)hidden, 0,
                                sizeof(float) * (size_t)hidden);
        if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            result.fail(GenerateErrorCode::BackendSpecific,
                             "prefill_logits_compute");
            return result;
        }
        last_logits.resize((size_t)w_.embedder.n_vocab);
        ggml_backend_tensor_get(logits, last_logits.data(), 0,
                                sizeof(float) * last_logits.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    auto t_pf1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();

    // ── Decode (hybrid layer-by-layer) ──
    auto argmax = [](const std::vector<float> & ll) {
        int best = 0; float bv = ll[0];
        for (size_t i = 1; i < ll.size(); ++i)
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        return best;
    };

    std::vector<int32_t> history;
    history.reserve((size_t)N + (size_t)req.n_gen);
    history.insert(history.end(), req.prompt.begin(), req.prompt.end());

    auto pick = [&](const std::vector<float> & ll) -> int {
        return req.do_sample
            ? sample_logits(ll.data(), (int)ll.size(), req.sampler, history, sampler_rng_)
            : argmax(ll);
    };

    int next_tok = pick(last_logits);
    result.tokens.reserve(req.n_gen);

    // Budget force-close (same pattern as non-hybrid path)
    const BudgetHook & budget_hook = req.budget_hook;
    bool budget_close_started = false;
    int  close_inject_pos     = 0;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size())
        {
            tok = budget_hook.close_token_ids[close_inject_pos++];
            return;
        }
        if (budget_close_started) return;
        int remaining = req.n_gen - committed_now;
        if (remaining <= budget_hook.hard_limit_remaining) {
            int32_t first_close = budget_hook.close_token_ids.front();
            if (tok == first_close) {
                budget_close_started = true;
                close_inject_pos = 1;
                return;
            }
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            result.budget_forced_close = true;
        }
    };

    std::vector<float> act_cur((size_t)w_.n_embd);
    auto t_g0 = std::chrono::steady_clock::now();
    for (int s = 0; s < req.n_gen; ++s) {
        maybe_force_close(next_tok, s);
        if (!std::getenv("DFLASH_IGNORE_EOS") && (next_tok == w_.eos_id || next_tok == w_.eos_chat_id)) break;
        result.tokens.push_back(next_tok);
        history.push_back(next_tok);
        if (should_emit) {
            out_io.emit(next_tok);
            if (out_io.cancelled) break;
        }

        // Hybrid forward: one token through all layers
        int32_t argmax_tok = 0;
        if (!hybrid_forward_one_token(next_tok, cache_.cur_pos, act_cur, argmax_tok)) {
            result.fail(GenerateErrorCode::DecodeFailed);
            break;
        }
        cache_.cur_pos++;
        kvflash_maybe_reselect(history, s + 1);

        if (req.do_sample) {
            // For sampling, we need full logits — project from act_cur
            // (hybrid_forward_one_token already computed argmax; for sampling
            // we re-project — FIXME: return logits from forward to avoid double projection)
            next_tok = argmax_tok;  // For now, use argmax even in sample mode as fallback
        } else {
            next_tok = argmax_tok;
        }
    }
    auto t_g1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_g1 - t_g0).count();

    if (should_emit) out_io.emit(-1);
    if (result.error && result.error->code == GenerateErrorCode::Incomplete) {
        result.succeed();
    }
    return result;
}

bool LagunaBackend::spark_wants_bootstrap() const {
    return moe_hybrid_ && routing_stats_ && !layer_expert_bytes_.empty() && spark_expert_budget_ > 0;
}

bool LagunaBackend::spark_bootstrap_finalize(const std::string & profile_path) {
    if (!spark_wants_bootstrap()) return false;
    std::string err;
    routing_stats_->save_csv(profile_path, &err);  // persist the observed routing
    MoeHybridPlacement placement;
    if (!MoeHybridPlacement::build_from_stats_with_layer_bytes(
            *routing_stats_, layer_expert_bytes_, spark_expert_budget_,
            std::min(w_.n_expert_used, w_.n_expert), placement, &err)) {
        std::fprintf(stderr, "[spark] bootstrap placement build failed: %s\n", err.c_str());
        return false;
    }
    if (!build_hybrid_storage_from_file(placement, moe_hybrid_, err)) {
        std::fprintf(stderr, "[spark] bootstrap storage rebuild failed: %s\n", err.c_str());
        return false;
    }
    return true;
}

bool LagunaBackend::build_hybrid_storage_from_file(
        const MoeHybridPlacement & placement,
        std::shared_ptr<MoeHybridStorage> & out_storage,
        std::string & err) {
    ggml_context * expert_meta = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &expert_meta;
    gguf_context * gctx = gguf_init_from_file(args_.target_path.c_str(), gip);
    if (!gctx) { err = "failed to re-open GGUF for expert loading"; return false; }

    GgufMmap _mf;
    std::string _mferr;
    if (!_mf.open(args_.target_path, _mferr)) {
        gguf_free(gctx);
        err = "mmap failed: " + _mferr;
        return false;
    }
    const size_t file_size = _mf.size();
    // Transfer mmap ownership out of the RAII wrapper: the hybrid storage keeps
    // the mapping alive and unmaps it in ~MoeHybridStorage. On POSIX the fd can
    // be closed now (the mapping stays valid); on Windows release() already
    // closed the mapping handle.
    GgufMmap::OwnedRegion _region = _mf.release();
    const void * mmap_addr = _region.data;
#if !defined(_WIN32)
    if (_region.fd >= 0) ::close(_region.fd);
#endif

    const size_t data_start = gguf_get_data_offset(gctx);
    const auto * file_bytes = (const uint8_t *)mmap_addr;

    std::vector<LayerExpertFileData> layer_file_data((size_t)w_.n_layer);
    for (int il = w_.n_layer_dense_lead; il < w_.n_layer; ++il) {
        char name[128];
        auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
            std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
            int64_t tid = gguf_find_tensor(gctx, name);
            if (tid < 0) return {};
            size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
            size_t sz = gguf_get_tensor_size(gctx, tid);
            if (off + sz > file_size) return {};
            return { file_bytes + off, sz };
        };
        layer_file_data[(size_t)il].gate_exps = find_tensor_data("ffn_gate_exps");
        layer_file_data[(size_t)il].up_exps   = find_tensor_data("ffn_up_exps");
        layer_file_data[(size_t)il].down_exps = find_tensor_data("ffn_down_exps");
        // laguna has no fused gate_up_exps
    }

    auto hybrid = std::make_shared<MoeHybridStorage>();
    MoeHybridConfig hybrid_cfg = make_moe_hybrid_config(w_);
    std::vector<MoeLayerDesc> layer_descs((size_t)w_.n_layer);
    for (int il = 0; il < w_.n_layer; ++il) {
        layer_descs[(size_t)il] = make_moe_layer_desc(w_.layers[(size_t)il]);
    }
    int cache_slots = 0;
    if (const char * cs = std::getenv("DFLASH_LAGUNA_CACHE_SLOTS")) cache_slots = std::max(0, std::atoi(cs));
    else if (cache_slots_ >= 0) cache_slots = cache_slots_;
    bool ok = build_moe_hybrid_storage_from_file_with_mmap(hybrid_cfg, backend_, placement,
                                                            layer_descs, layer_file_data,
                                                            mmap_addr, file_size, *hybrid, &err, cache_slots);
    gguf_free(gctx);
    if (!ok) {
#if defined(_WIN32)
        UnmapViewOfFile(const_cast<void *>(mmap_addr));
#else
        ::munmap(const_cast<void *>(mmap_addr), file_size);
#endif
        return false;
    }
    // mmap ownership transferred to the storage (unmapped in ~MoeHybridStorage)
    out_storage = std::move(hybrid);
    return true;
}

void LagunaBackend::maybe_post_request_swap() {
    if (!hybrid_mode_ || !moe_hybrid_ || swap_policy_.max_swaps_total <= 0) return;
    if (!routing_stats_) return;

    MoeHybridSwapPlan plan;
    std::string err;
    if (!build_moe_hybrid_swap_plan(moe_hybrid_->placement, *routing_stats_,
                                   swap_policy_, plan, &err)) {
        std::fprintf(stderr, "[laguna-hybrid] swap plan failed: %s\n", err.c_str());
        return;
    }
    if (plan.actions.empty()) return;

    // Rebuild storage with new placement. Partial-load mode keeps no full
    // expert tensors resident, so we must re-read from the GGUF mmap (the
    // GPU-tensor variant would read unbuffered tensors and assert).
    std::shared_ptr<MoeHybridStorage> rebuilt;
    if (!build_hybrid_storage_from_file(plan.next_placement, rebuilt, err)) {
        std::fprintf(stderr, "[laguna-hybrid] swap rebuild failed: %s\n", err.c_str());
        return;
    }
    moe_hybrid_ = std::move(rebuilt);

    // Save updated routing stats if configured
    if (!routing_stats_out_path_.empty()) {
        routing_stats_->save_csv(routing_stats_out_path_, &err);
    }

    std::printf("[laguna-hybrid] applied %zu swap actions at request boundary\n", plan.actions.size());
    std::fflush(stdout);
}

bool LagunaBackend::load_decode_draft() {
    if (args_.draft_path.empty()) return false;
    if (draft_backend_ && feature_mirror_.target_feat && !draft_variants_.empty()) {
        draft_parked_ = false;
        return true;
    }

    const int draft_gpu = (args_.draft_gpu >= 0) ? args_.draft_gpu : args_.device.gpu;
    if (args_.draft_gpu >= 0) {
        draft_backend_ = ggml_backend_cuda_init(draft_gpu);
        if (!draft_backend_) {
            std::fprintf(stderr, "[laguna] draft CUDA init failed (gpu=%d)\n", draft_gpu);
            return false;
        }
    } else {
        draft_backend_ = backend_;
    }

    draft_variants_.clear();
    draft_variants_.push_back(LagunaDraftVariant{});
    draft_variants_.back().name = "base";

    int base_fc_in = 0;
    int base_draft_hidden = 0;
    int base_n_capture = 0;

    for (LagunaDraftVariant & variant : draft_variants_) {
        if (!load_draft_gguf(args_.draft_path, draft_backend_, variant.weights,
                             nullptr)) {
            std::fprintf(stderr, "[laguna] draft load failed for variant '%s': %s\n",
                         variant.name.c_str(), dflash27b_last_error());
            free_decode_draft();
            return false;
        }

        DraftWeights & dw = variant.weights;
        dw.mask_token_id = 12;
        const int draft_hidden = (int)dw.fc->ne[1];
        const int fc_in = (int)dw.fc->ne[0];
        const int n_capture = fc_in / w_.n_embd;

        if (draft_hidden != dw.n_embd) {
            std::printf("[laguna] draft[%s]: overriding n_embd %d -> %d (from fc weight)\n",
                        variant.name.c_str(), dw.n_embd, draft_hidden);
            dw.n_embd = draft_hidden;
        }
        if (dw.n_layer > 0 && dw.layers[0].wq) {
            const int q_dim = (int)dw.layers[0].wq->ne[1];
            const int inferred_n_head = q_dim / dw.head_dim;
            if (inferred_n_head != dw.n_head) {
                std::printf("[laguna] draft[%s]: overriding n_head %d -> %d\n",
                            variant.name.c_str(), dw.n_head, inferred_n_head);
                dw.n_head = inferred_n_head;
            }
        }
        if (dw.n_layer > 0 && dw.layers[0].w_gate) {
            const int inferred_ff = (int)dw.layers[0].w_gate->ne[1];
            if (inferred_ff != dw.n_ff) {
                std::printf("[laguna] draft[%s]: overriding n_ff %d -> %d\n",
                            variant.name.c_str(), dw.n_ff, inferred_ff);
                dw.n_ff = inferred_ff;
            }
        }
        dw.n_target_layers = n_capture;
        dw.swa_window = 2048;
        for (int i = 0; i < dw.n_layer - 1 && i < (int)dw.layers.size(); i++) {
            dw.layers[(size_t)i].is_swa = true;
        }

        if (base_fc_in == 0) {
            base_fc_in = fc_in;
            base_draft_hidden = draft_hidden;
            base_n_capture = n_capture;
        } else if (fc_in != base_fc_in || draft_hidden != base_draft_hidden ||
                   n_capture != base_n_capture) {
            std::fprintf(stderr,
                "[laguna] draft variant '%s' changed draft dimensions "
                "(fc_in=%d hidden=%d capture=%d, base fc_in=%d hidden=%d capture=%d)\n",
                variant.name.c_str(), fc_in, draft_hidden, n_capture,
                base_fc_in, base_draft_hidden, base_n_capture);
            free_decode_draft();
            return false;
        }

        std::printf("[laguna] draft variant loaded: name=%s fc_in=%d "
                    "target_hidden=%d draft_hidden=%d n_capture_layers=%d swa=%d\n",
                    variant.name.c_str(), fc_in, w_.n_embd,
                    draft_hidden, n_capture, dw.swa_window);
    }

    const int n_capture = base_n_capture;

    constexpr int TARGET_FEAT_CAP = 4096;
    const int feat_cap = std::min(args_.max_ctx, TARGET_FEAT_CAP);
    if (!cache_.target_feat &&
        !create_laguna_target_feat(backend_, cache_, n_capture, w_.n_embd, feat_cap,
                                   draft_variants_[0].weights.capture_layer_ids)) {
        std::fprintf(stderr, "[laguna] target_feat alloc failed\n");
        free_decode_draft();
        return false;
    }

    const int mirror_cap = std::min(args_.draft_ctx_max, feat_cap);
    if (!draft_feature_mirror_init(feature_mirror_, draft_backend_,
                                   draft_gpu, args_.device.gpu, mirror_cap,
                                   n_capture, w_.n_embd)) {
        std::fprintf(stderr, "[laguna] feature mirror init failed\n");
        free_decode_draft();
        return false;
    }

    default_draft_variant_ = "base";
    if (!select_decode_draft(default_draft_variant_)) {
        free_decode_draft();
        return false;
    }

    delete dflash_target_;
    dflash_target_ = new LagunaDFlashTarget(w_, cache_, backend_);
    if (kvflash_active()) dflash_target_->set_kvflash_pager(&kvflash_pager_);
    draft_parked_ = false;

    std::printf("[laguna] spec-decode ready: capture_layers=%d mirror_cap=%d\n",
                n_capture, mirror_cap);
    std::printf("[laguna] capture_layer_ids:");
    for (int k = 0; k < (int)cache_.capture_layer_ids.size(); k++) {
        std::printf(" %d", cache_.capture_layer_ids[(size_t)k]);
    }
    std::printf("\n");
    std::fflush(stdout);
    return true;
}

bool LagunaBackend::select_decode_draft(const std::string & name) {
    std::string wanted = name;
    if (wanted.empty()) {
        wanted = default_draft_variant_;
    }
    for (LagunaDraftVariant & variant : draft_variants_) {
        if (variant.name == wanted) {
            if (active_dw_ != &variant.weights) {
                std::fprintf(stderr, "[laguna] selected draft variant: %s\n",
                             variant.name.c_str());
            }
            active_dw_ = &variant.weights;
            return true;
        }
    }
    std::fprintf(stderr, "[laguna] unknown draft variant '%s'\n",
                 wanted.c_str());
    return false;
}

void LagunaBackend::free_decode_draft() {
    delete dflash_target_;
    dflash_target_ = nullptr;
    draft_kv_free(draft_kv_);
    draft_feature_mirror_free(feature_mirror_);
    free_laguna_target_feat(cache_);
    for (LagunaDraftVariant & variant : draft_variants_) {
        if (variant.weights.ctx) {
            free_draft_weights(variant.weights);
        }
    }
    draft_variants_.clear();
    active_dw_ = nullptr;
    default_draft_variant_ = "base";
    if (draft_backend_ && draft_backend_ != backend_) {
        ggml_backend_free(draft_backend_);
    }
    draft_backend_ = nullptr;
}

// ── Shutdown ────────────────────────────────────────────────────────────

void LagunaBackend::shutdown() {
    for (auto & snap : snapshots_) laguna_snapshot_free(snap);
    if (drafter_loaded_) {
        dflash::common::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
    }
    free_decode_draft();
    if (!target_parked_) {
        free_laguna_target_cache(cache_);
        free_laguna_target_weights(w_);
    }
    free_snapshot_backend(snap_backend_, backend_);
    snap_backend_ = nullptr;
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace dflash::common
