// Inkling2KvFlash — 3-Tier Dual-GPU KVFlash 2.0 Engine + Mode-4 Exact D=128 Recall.

#include "inkling2_kvflash.h"
#include "inkling2_cache.h"

#include "common/kvflash_exact_fast.h"
#include "common/kvflash_gpu_mass.h"
#include "common/kvflash_pager.h"
#include "common/kvflash_prefill_reselect.h"
#include "common/kvflash_query_capture.h"

#include "ggml-cuda.h"

#if defined(LUCE_BACKEND_CUDA) || defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "common/gpu_runtime_compat.h"
#define INKLING2_KVFLASH_HAS_GPU_RT 1
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace luce::common {

namespace {

struct ScopedGpuDevice {
    int prev = -1;
    bool switched = false;
    explicit ScopedGpuDevice(int dev) {
#ifdef INKLING2_KVFLASH_HAS_GPU_RT
        if (dev >= 0 && cudaGetDevice(&prev) == cudaSuccess && prev != dev) {
            if (cudaSetDevice(dev) == cudaSuccess) {
                switched = true;
            }
        }
#else
        (void) dev;
#endif
    }
    ~ScopedGpuDevice() {
#ifdef INKLING2_KVFLASH_HAS_GPU_RT
        if (switched && prev >= 0) {
            (void) cudaSetDevice(prev);
        }
#endif
    }
};

static int env_int_or(const char * name, int def_val) {
    const char * v = std::getenv(name);
    if (!v || !*v) return def_val;
    return std::atoi(v);
}

static int backend_gpu_id(ggml_backend_t be) {
    if (!be) return -1;
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) return -1;
    return ggml_backend_cuda_get_device_id(be);
}

} // namespace

Inkling2KvFlashConfig Inkling2KvFlashConfig::from_env(
        int max_ctx,
        int rel_extent,
        int n_global_planes,
        size_t global_bytes_per_tok,
        ggml_backend_t target_backend,
        ggml_backend_t expert_backend) {
    Inkling2KvFlashConfig cfg;
    const char * env = std::getenv("LUCE_KVFLASH");
    if (!kvflash_pool_requested(env) || n_global_planes <= 0 || max_ctx <= 0) {
        return cfg;
    }

    cfg.chunk_tokens = kInkling2KvFlashChunk;
    const int min_hot_tail = std::max(rel_extent, cfg.chunk_tokens);
    int hot_tail_tok = env_int_or("LUCE_KVFLASH_HOT_TAIL", min_hot_tail);
    hot_tail_tok = std::max(cfg.chunk_tokens, hot_tail_tok);
    cfg.hot_tail_chunks = (hot_tail_tok + cfg.chunk_tokens - 1) / cfg.chunk_tokens;
    cfg.sink_chunks = std::max(1, env_int_or("LUCE_KVFLASH_SINK_CHUNKS", 2));

    KvFlashConfig pcfg;
    pcfg.chunk_tokens       = cfg.chunk_tokens;
    pcfg.sink_chunks        = cfg.sink_chunks;
    pcfg.tail_window_chunks = cfg.hot_tail_chunks;

    KvFlashAutoBudget budget;
    budget.bytes_per_token  = (int64_t) global_bytes_per_tok;
    budget.reserve_bytes    = (int64_t) 4 << 30;
    budget.speed_cap_tokens = 16384;
    const int tgpu = backend_gpu_id(target_backend);
    if (tgpu >= 0) {
        size_t fr = 0, tot = 0;
        ggml_backend_cuda_get_device_memory(tgpu, &fr, &tot);
        budget.free_bytes = (int64_t) fr;
    }

    int pool_tokens = kvflash_pool_from_env(max_ctx, pcfg, /*scorer_expected=*/true, budget);
    if (pool_tokens <= 0) return cfg;

    // Allow small pools in tests when LUCE_KVFLASH_HOT_TAIL was explicitly shrunk
    cfg.tier0_pool_tokens = (pool_tokens / cfg.chunk_tokens) * cfg.chunk_tokens;
    if (cfg.tier0_pool_tokens < KvFlashPager::min_pool_tokens(pcfg)) {
        cfg.tier0_pool_tokens = ((KvFlashPager::min_pool_tokens(pcfg) + cfg.chunk_tokens - 1) / cfg.chunk_tokens) * cfg.chunk_tokens;
    }
    if (cfg.tier0_pool_tokens > max_ctx) {
        cfg.tier0_pool_tokens = (max_ctx / cfg.chunk_tokens) * cfg.chunk_tokens;
    }
    if (cfg.tier0_pool_tokens < KvFlashPager::min_pool_tokens(pcfg)) {
        return cfg;
    }

    cfg.enabled        = true;
    cfg.mode4_enabled  = (env_int_or("LUCE_KVFLASH_PREFILL_RESELECT", 4) != 0);
    cfg.recall_blocks  = std::max(1, env_int_or("LUCE_KVFLASH_RECALL_BLOCKS", 4));
    cfg.select_unit    = std::max(1, env_int_or("LUCE_KVFLASH_SELECT_UNIT", kKvFlashRecallUnit));
    cfg.select_queries = std::clamp(env_int_or("LUCE_KVFLASH_SELECT_QUERIES", kQueryCaptureTokens), 1, kQueryCaptureTokens);

    // Clamp recall budget so sink + hot_tail + recall + 2 <= tier0_n_blocks
    const int t0_blocks = cfg.tier0_pool_tokens / cfg.chunk_tokens;
    const int avail_for_recall = std::max(0, t0_blocks - cfg.sink_chunks - cfg.hot_tail_chunks - 4);
    if (avail_for_recall > 0) {
        if (cfg.select_unit > avail_for_recall) {
            cfg.select_unit = std::max(1, avail_for_recall / 2);
        }
        const int max_rb = std::max(1, avail_for_recall / cfg.select_unit);
        cfg.recall_blocks = std::min(cfg.recall_blocks, max_rb);
    } else {
        cfg.recall_blocks = 0;
    }

    // Optional Tier 1.5 warm pool on expert_backend (e.g. Strix Halo UMA)
    cfg.tier15_pool_tokens = 0;
    if (expert_backend && max_ctx > cfg.tier0_pool_tokens) {
        const char * env_w = std::getenv("LUCE_KVFLASH_WARM_POOL");
        if (env_w && (std::strcmp(env_w, "0") == 0 || std::strcmp(env_w, "off") == 0)) {
            cfg.tier15_pool_tokens = 0;
        } else if (env_w && *env_w) {
            int req_w = std::atoi(env_w);
            if (req_w > 0) {
                req_w = ((req_w + cfg.chunk_tokens - 1) / cfg.chunk_tokens) * cfg.chunk_tokens;
                cfg.tier15_pool_tokens = std::min(req_w, max_ctx);
            }
        } else {
            // Auto-size Tier 1.5 warm pool from expert_backend free memory (leave 4 GiB headroom)
            const int egpu = backend_gpu_id(expert_backend);
            size_t efree = 0, etotal = 0;
            if (egpu >= 0) ggml_backend_cuda_get_device_memory(egpu, &efree, &etotal);
            const size_t reserve = (size_t) 4 << 30;
            if (efree > reserve && global_bytes_per_tok > 0) {
                const int need_warm = max_ctx - cfg.tier0_pool_tokens;
                const int max_fit = (int) std::min<size_t>((size_t) max_ctx, (efree - reserve) / global_bytes_per_tok);
                int wtok = std::min(need_warm, max_fit);
                wtok = (wtok / cfg.chunk_tokens) * cfg.chunk_tokens;
                if (wtok >= cfg.chunk_tokens * 2) {
                    cfg.tier15_pool_tokens = wtok;
                }
            }
        }
    }

    if (const char * sg = std::getenv("LUCE_KVFLASH_SCORER_GPU")) {
        if (std::strcmp(sg, "target") == 0) cfg.score_on_expert = false;
        else if (std::strcmp(sg, "expert") == 0) cfg.score_on_expert = true;
    }

    return cfg;
}

Inkling2KvFlash::Inkling2KvFlash() = default;

Inkling2KvFlash::~Inkling2KvFlash() {
    detach();
}

void Inkling2KvFlash::reset_scorer_() {
    if (gpu_mass_) {
        ScopedGpuDevice guard(scorer_gpu_);
        gpu_mass_.reset();
    }
}

bool Inkling2KvFlash::attach(
        const Inkling2KvFlashConfig & cfg,
        int max_ctx,
        int head_dim,
        int n_head_q,
        int n_head_kv,
        ggml_backend_t target_backend,
        ggml_backend_t expert_backend,
        const std::vector<ggml_tensor *> & tier0_k,
        const std::vector<ggml_tensor *> & tier0_v,
        const std::vector<ggml_tensor *> & tier15_k,
        const std::vector<ggml_tensor *> & tier15_v) {
    detach();
    if (!cfg.enabled || max_ctx <= 0 || tier0_k.empty() || tier0_k.size() != tier0_v.size()) {
        return false;
    }

    cfg_            = cfg;
    max_ctx_        = max_ctx;
    max_blocks_     = (max_ctx + cfg.chunk_tokens - 1) / cfg.chunk_tokens;
    n_global_       = (int) tier0_k.size();
    head_dim_       = head_dim;
    n_head_q_       = n_head_q;
    n_head_kv_      = n_head_kv;
    target_backend_ = target_backend;
    expert_backend_ = expert_backend;
    target_gpu_     = backend_gpu_id(target_backend);
    expert_gpu_     = backend_gpu_id(expert_backend);
    tier0_k_        = tier0_k;
    tier0_v_        = tier0_v;

    KvFlashConfig pcfg;
    pcfg.chunk_tokens       = cfg.chunk_tokens;
    pcfg.pool_tokens        = cfg.tier0_pool_tokens;
    pcfg.sink_chunks        = cfg.sink_chunks;
    pcfg.tail_window_chunks = cfg.hot_tail_chunks;

    {
        ScopedGpuDevice guard(target_gpu_);
        pager_ = std::make_unique<KvFlashPager>();
        if (target_gpu_ < 0) pager_->set_pinned_host_backing(false);
        if (!pager_->attach(pcfg, tier0_k_, tier0_v_)) {
            detach();
            return false;
        }
    }

    tier15_active_ = (expert_backend_ != nullptr &&
                      cfg.tier15_pool_tokens >= cfg.chunk_tokens &&
                      tier15_k.size() == tier0_k.size() &&
                      tier15_v.size() == tier0_v.size());
    if (tier15_active_) {
        tier15_k_ = tier15_k;
        tier15_v_ = tier15_v;
        tier15_n_blocks_ = cfg.tier15_pool_tokens / cfg.chunk_tokens;
    } else {
        tier15_k_.clear();
        tier15_v_.clear();
        tier15_n_blocks_ = 0;
    }

    scorer_gpu_ = (tier15_active_ && cfg.score_on_expert && expert_gpu_ >= 0) ? expert_gpu_ : target_gpu_;

    tier15_block_of_.assign((size_t) max_blocks_, -1);
    tier15_last_use_.assign((size_t) max_blocks_, 0);
    tier15_ver_.assign((size_t) max_blocks_, 0);
    tier15_free_blocks_.clear();
    for (int b = tier15_n_blocks_ - 1; b >= 0; --b) tier15_free_blocks_.push_back(b);
    tier15_clock_ = 0;

    bt_local_.assign((size_t) max_blocks_, -1);
    bt_remote_.assign((size_t) max_blocks_, -1);
    q_tail_.clear();
    last_scores_.clear();
    gpu_scorer_failed_ = false;
    snap_ = {};
    active_ = true;

    std::fprintf(stderr,
        "[inkling2-kvflash] attached: L_global=%d chunk=%d max_ctx=%d (max_blocks=%d) "
        "tier0=%d tok (%d blks) tier15=%d tok (%d blks) hot_tail=%d sink=%d mode4=%d recall=%dx%d\n",
        n_global_, cfg_.chunk_tokens, max_ctx_, max_blocks_,
        cfg_.tier0_pool_tokens, pager_->n_pool_blocks(),
        tier15_active_ ? cfg_.tier15_pool_tokens : 0, tier15_n_blocks_,
        cfg_.hot_tail_chunks * cfg_.chunk_tokens, cfg_.sink_chunks * cfg_.chunk_tokens,
        cfg_.mode4_enabled ? 1 : 0, cfg_.recall_blocks, cfg_.select_unit);
    return true;
}

void Inkling2KvFlash::detach() {
    reset_scorer_();
    if (pager_) {
        ScopedGpuDevice guard(target_gpu_);
        if (pager_->attached()) pager_->reset();
        pager_.reset();
    }
    active_ = false;
    tier15_active_ = false;
    tier0_k_.clear();
    tier0_v_.clear();
    tier15_k_.clear();
    tier15_v_.clear();
    tier15_block_of_.clear();
    tier15_last_use_.clear();
    tier15_ver_.clear();
    tier15_free_blocks_.clear();
    bt_local_.clear();
    bt_remote_.clear();
    q_tail_.clear();
    last_scores_.clear();
    active_hook_scores_.clear();
    active_hook_sealed_ = 0;
    snap_ = {};
}

void Inkling2KvFlash::reset() {
    if (!active_ || !pager_) return;
    {
        ScopedGpuDevice guard(target_gpu_);
        pager_->reset();
    }
    if (gpu_mass_) {
        ScopedGpuDevice guard(scorer_gpu_);
        gpu_mass_->drop_archive();
    }
    std::fill(tier15_block_of_.begin(), tier15_block_of_.end(), -1);
    std::fill(tier15_last_use_.begin(), tier15_last_use_.end(), 0);
    std::fill(tier15_ver_.begin(), tier15_ver_.end(), 0);
    tier15_free_blocks_.clear();
    for (int b = tier15_n_blocks_ - 1; b >= 0; --b) tier15_free_blocks_.push_back(b);
    tier15_clock_ = 0;
    std::fill(bt_local_.begin(), bt_local_.end(), -1);
    std::fill(bt_remote_.begin(), bt_remote_.end(), -1);
    ++tables_epoch_;
    q_tail_.clear();
    last_scores_.clear();
    active_hook_scores_.clear();
    active_hook_sealed_ = 0;
    snap_ = {};
}

bool Inkling2KvFlash::upload_chunk_to_tier15_(int c, int rb) {
    if (!tier15_active_ || rb < 0 || rb >= tier15_n_blocks_) return false;
    const size_t k_seg = pager_->k_seg_bytes();
    const size_t v_seg = (size_t) cfg_.chunk_tokens * tier0_v_[0]->nb[1];
    for (int l = 0; l < n_global_; ++l) {
        const uint8_t * k0 = pager_->host_k_seg(c, l, 0);
        if (!k0) return false;
        const uint8_t * v0 = k0 + (size_t) n_head_kv_ * k_seg;
        ggml_tensor * rk = tier15_k_[(size_t) l];
        ggml_tensor * rv = tier15_v_[(size_t) l];
        const size_t k_base = (size_t) rb * cfg_.chunk_tokens * rk->nb[1];
        const size_t v_base = (size_t) rb * cfg_.chunk_tokens * rv->nb[1];
        for (int h = 0; h < n_head_kv_; ++h) {
            ggml_backend_tensor_set(rk, k0 + (size_t) h * k_seg, k_base + (size_t) h * rk->nb[2], k_seg);
            ggml_backend_tensor_set(rv, v0 + (size_t) h * v_seg, v_base + (size_t) h * rv->nb[2], v_seg);
        }
    }
    return true;
}

int Inkling2KvFlash::alloc_tier15_block_(int c) {
    if (!tier15_active_ || tier15_n_blocks_ <= 0) return -1;
    if (tier15_block_of_[(size_t) c] >= 0) {
        tier15_last_use_[(size_t) c] = ++tier15_clock_;
        return tier15_block_of_[(size_t) c];
    }
    int rb = -1;
    if (!tier15_free_blocks_.empty()) {
        rb = tier15_free_blocks_.back();
        tier15_free_blocks_.pop_back();
    } else {
        // LRU eviction from Tier 1.5 -> Tier 2 Host-only
        int victim = -1;
        uint64_t best_u = UINT64_MAX;
        for (int i = 0; i < max_blocks_; ++i) {
            if (tier15_block_of_[(size_t) i] >= 0 && tier15_last_use_[(size_t) i] < best_u) {
                best_u = tier15_last_use_[(size_t) i];
                victim = i;
            }
        }
        if (victim < 0) return -1;
        rb = tier15_block_of_[(size_t) victim];
        tier15_block_of_[(size_t) victim] = -1;
        // Keep tier15_ver_[victim] so sync_tier15_and_tables_() does not re-upload
        // chunks that already overflowed Tier 1.5 to Tier 2 Host on subsequent steps.
    }
    tier15_block_of_[(size_t) c] = rb;
    tier15_last_use_[(size_t) c] = ++tier15_clock_;
    return rb;
}

void Inkling2KvFlash::free_tier15_block_(int c) {
    if (!tier15_active_ || c < 0 || c >= max_blocks_) return;
    const int rb = tier15_block_of_[(size_t) c];
    if (rb >= 0) {
        tier15_block_of_[(size_t) c] = -1;
        tier15_ver_[(size_t) c] = 0;
        tier15_free_blocks_.push_back(rb);
    }
}

void Inkling2KvFlash::rebuild_block_tables_() {
    const int nc = std::min(pager_->n_chunks(), max_blocks_);
    bool changed = (tables_epoch_ == 0);
    for (int c = 0; c < max_blocks_; ++c) {
        const int32_t nl = (c < nc) ? pager_->block_of(c) : -1;
        if (bt_local_[(size_t) c] != nl) { bt_local_[(size_t) c] = nl; changed = true; }
        if (tier15_active_) {
            const int32_t nr = (c < nc && nl < 0) ? tier15_block_of_[(size_t) c] : -1;
            if (bt_remote_[(size_t) c] != nr) { bt_remote_[(size_t) c] = nr; changed = true; }
        }
    }
    if (changed) ++tables_epoch_;
}

void Inkling2KvFlash::sync_tier15_and_tables_() {
    if (tier15_active_) {
        const int nc = std::min(pager_->n_chunks(), max_blocks_);
        // 1. Free Tier-1.5 blocks for chunks that were recalled into Tier 0 or truncated away
        for (int c = nc; c < max_blocks_; ++c) {
            if (tier15_block_of_[(size_t) c] >= 0) free_tier15_block_(c);
            tier15_ver_[(size_t) c] = 0;
        }
        for (int c = 0; c < nc; ++c) {
            if (pager_->block_of(c) >= 0) {
                if (tier15_block_of_[(size_t) c] >= 0) free_tier15_block_(c);
                tier15_ver_[(size_t) c] = 0;
            }
        }
        // 2. Promote newly evicted Tier-0 chunks into Tier 1.5 (must sync Tier-0 D2H first!)
        bool need_sync = false;
        for (int c = 0; c < nc; ++c) {
            if (pager_->block_of(c) < 0 && pager_->host_k_seg(c, 0, 0) != nullptr) {
                const uint64_t v = pager_->chunk_version(c);
                if (tier15_ver_[(size_t) c] != v) {
                    need_sync = true;
                    break;
                }
            }
        }
        if (need_sync) {
            {
                ScopedGpuDevice guard(target_gpu_);
                pager_->synchronize_paging();
            }
            for (int c = 0; c < nc; ++c) {
                if (pager_->block_of(c) < 0 && pager_->host_k_seg(c, 0, 0) != nullptr) {
                    const uint64_t v = pager_->chunk_version(c);
                    if (tier15_ver_[(size_t) c] == v) continue;
                    const int rb = alloc_tier15_block_(c);
                    if (rb >= 0 && upload_chunk_to_tier15_(c, rb)) {
                        tier15_ver_[(size_t) c] = v;
                    } else if (rb >= 0) {
                        free_tier15_block_(c);
                    }
                }
            }
        }
    }
    rebuild_block_tables_();
}

bool Inkling2KvFlash::prepare_span(int pos0, int n_tok) {
    if (!active_ || n_tok <= 0) return true;
    if (pos0 < 0 || pos0 + n_tok > max_ctx_) return false;
    bool ok = false;
    {
        ScopedGpuDevice guard(target_gpu_);
        ok = pager_->alloc_span(pos0, n_tok);
    }
    if (!ok) return false;
    sync_tier15_and_tables_();
    return true;
}

void Inkling2KvFlash::fill_tree_prefix_buf(int prefix_len, bool remote, std::vector<int32_t> & out) const {
    out.resize((size_t) (2 + max_blocks_));
    out[0] = prefix_len;
    out[1] = 0;
    const auto & bt = remote ? bt_remote_ : bt_local_;
    if (max_blocks_ > 0) {
        std::memcpy(out.data() + 2, bt.data(), (size_t) max_blocks_ * sizeof(int32_t));
    }
}

void Inkling2KvFlash::append_queries(const float * rows, int n_q) {
    if (!active_ || !rows || n_q <= 0) return;
    const size_t qsize = (size_t) n_global_ * n_head_q_ * head_dim_;
    std::vector<float> tmp(rows, rows + (size_t) n_q * qsize);
    kvflash_q_tail_append(q_tail_, tmp, qsize, cfg_.select_queries);
}

int Inkling2KvFlash::captured_queries() const {
    const size_t qsize = (size_t) n_global_ * n_head_q_ * head_dim_;
    return (qsize > 0) ? (int) (q_tail_.size() / qsize) : 0;
}

int Inkling2KvFlash::tier15_resident_blocks() const {
    if (!tier15_active_) return 0;
    int cnt = 0;
    for (int b : tier15_block_of_) if (b >= 0) ++cnt;
    return cnt;
}

int Inkling2KvFlash::tier0_resident_blocks() const {
    return (active_ && pager_) ? pager_->resident_blocks() : 0;
}

int Inkling2KvFlash::plan_mode4_rewind(const std::vector<int32_t> & prompt, int32_t user_turn_token_id) const {
    if (!active_ || !cfg_.mode4_enabled || cfg_.recall_blocks <= 0) return -1;
    const int end = (int) prompt.size();
    const int ct  = cfg_.chunk_tokens;
    const int n_pool_blocks = pager_->n_pool_blocks();
    const int recall_chunks = cfg_.recall_blocks * cfg_.select_unit;
    // Skip Mode-4 when the entire prompt already fits comfortably inside Tier 0
    if (end <= cfg_.tier0_pool_tokens) return -1;
    // Skip Mode-4 when Tier 1.5 already covers every evicted block losslessly
    if (tier15_active_ && end <= cfg_.tier0_pool_tokens + cfg_.tier15_pool_tokens) return -1;

    const int max_turn_tok = std::max(
        ct,
        std::min(512, kvflash_max_turn_tokens(ct, n_pool_blocks, 0, recall_chunks)));

    int r = -1;
    if (user_turn_token_id >= 0) {
        for (int i = end - 1; i >= 0; --i) {
            if (prompt[(size_t) i] == user_turn_token_id) {
                r = i;
                break;
            }
        }
    }
    if (r <= ct || end - r > max_turn_tok) {
        // Fallback: rewind the last `min(256, max_turn_tok)` tokens (aligned to chunk boundary when possible)
        const int tail_len = std::min(256, max_turn_tok);
        r = std::max(ct, end - tail_len);
    }
    // Align R down to a chunk boundary (clean truncate, zero partial-chunk zeroing overhead!)
    r = (r / ct) * ct;
    if (r < (cfg_.sink_chunks + 2) * ct) return -1;
    if (end - r <= 0 || end - r > max_turn_tok) {
        r = ((end - std::min(256, max_turn_tok)) / ct) * ct;
        if (r < (cfg_.sink_chunks + 2) * ct) return -1;
    }
    return r;
}

bool Inkling2KvFlash::save_rewind_snap(const Inkling2Cache & cache, const Inkling2Weights & w, int rewind_pos) {
    clear_rewind_snap();
    snap_.rewind_pos     = rewind_pos;
    snap_.dspark_ctx_pos = cache.dspark_ctx_pos;
    snap_.is_tree        = cache.tree.active;

    if (cache.tree.active) {
        const auto & tc = cache.tree;
        const int n_phys = w.n_layer_phys;
        snap_.conv_bytes.resize((size_t) n_phys);
        snap_.swa_k_bytes.resize((size_t) n_phys);
        snap_.swa_v_bytes.resize((size_t) n_phys);
        snap_.shared_tail_k_bytes.resize((size_t) n_phys);
        snap_.shared_tail_v_bytes.resize((size_t) n_phys);
        for (int ip = 0; ip < n_phys; ++ip) {
            if (tc.prefix_conv[ip]) {
                snap_.conv_bytes[ip].resize(ggml_nbytes(tc.prefix_conv[ip]));
                ggml_backend_tensor_get(tc.prefix_conv[ip], snap_.conv_bytes[ip].data(), 0, snap_.conv_bytes[ip].size());
            }
            if (w.layers[ip].is_swa) {
                snap_.swa_k_bytes[ip].resize(ggml_nbytes(tc.prefix_k[ip]));
                snap_.swa_v_bytes[ip].resize(ggml_nbytes(tc.prefix_v[ip]));
                ggml_backend_tensor_get(tc.prefix_k[ip], snap_.swa_k_bytes[ip].data(), 0, snap_.swa_k_bytes[ip].size());
                ggml_backend_tensor_get(tc.prefix_v[ip], snap_.swa_v_bytes[ip].data(), 0, snap_.swa_v_bytes[ip].size());
            } else if (tc.shared_kv_window > 0 && tc.planes[ip] > 1 && tc.shared_tail_cap > 0) {
                const int n_tail_rows = (tc.planes[ip] - 1) * tc.shared_tail_cap;
                const size_t seg_k = (size_t) n_tail_rows * tc.prefix_k[ip]->nb[1];
                const size_t seg_v = (size_t) n_tail_rows * tc.prefix_v[ip]->nb[1];
                const size_t off_k = (size_t) tc.global_pool_cap * tc.prefix_k[ip]->nb[1];
                const size_t off_v = (size_t) tc.global_pool_cap * tc.prefix_v[ip]->nb[1];
                snap_.shared_tail_k_bytes[ip].resize(seg_k * (size_t) w.n_head_kv);
                snap_.shared_tail_v_bytes[ip].resize(seg_v * (size_t) w.n_head_kv);
                for (int h = 0; h < w.n_head_kv; ++h) {
                    ggml_backend_tensor_get(tc.prefix_k[ip], snap_.shared_tail_k_bytes[ip].data() + (size_t) h * seg_k,
                                            (size_t) h * tc.prefix_k[ip]->nb[2] + off_k, seg_k);
                    ggml_backend_tensor_get(tc.prefix_v[ip], snap_.shared_tail_v_bytes[ip].data() + (size_t) h * seg_v,
                                            (size_t) h * tc.prefix_v[ip]->nb[2] + off_v, seg_v);
                }
            }
        }
    } else {
        const int n_slots = w.n_layer_eff;
        snap_.conv_bytes.resize((size_t) n_slots);
        snap_.swa_k_bytes.resize((size_t) n_slots);
        snap_.swa_v_bytes.resize((size_t) n_slots);
        for (int il = 0; il < n_slots; ++il) {
            if (cache.conv_state[il]) {
                snap_.conv_bytes[il].resize(ggml_nbytes(cache.conv_state[il]));
                ggml_backend_tensor_get(cache.conv_state[il], snap_.conv_bytes[il].data(), 0, snap_.conv_bytes[il].size());
            }
            const int ip = w.phys_layer_for_slot(il);
            if (w.layers[ip].is_swa) {
                snap_.swa_k_bytes[il].resize(ggml_nbytes(cache.k_cache[il]));
                snap_.swa_v_bytes[il].resize(ggml_nbytes(cache.v_cache[il]));
                ggml_backend_tensor_get(cache.k_cache[il], snap_.swa_k_bytes[il].data(), 0, snap_.swa_k_bytes[il].size());
                ggml_backend_tensor_get(cache.v_cache[il], snap_.swa_v_bytes[il].data(), 0, snap_.swa_v_bytes[il].size());
            }
        }
    }
    snap_.valid = true;
    return true;
}

bool Inkling2KvFlash::restore_rewind_snap(Inkling2Cache & cache, const Inkling2Weights & w) {
    if (!snap_.valid) return false;
    if (snap_.is_tree && cache.tree.active) {
        auto & tc = cache.tree;
        const int n_phys = w.n_layer_phys;
        for (int ip = 0; ip < n_phys; ++ip) {
            if (tc.prefix_conv[ip] && !snap_.conv_bytes[ip].empty()) {
                ggml_backend_tensor_set(tc.prefix_conv[ip], snap_.conv_bytes[ip].data(), 0, snap_.conv_bytes[ip].size());
            }
            if (w.layers[ip].is_swa && !snap_.swa_k_bytes[ip].empty()) {
                ggml_backend_tensor_set(tc.prefix_k[ip], snap_.swa_k_bytes[ip].data(), 0, snap_.swa_k_bytes[ip].size());
                ggml_backend_tensor_set(tc.prefix_v[ip], snap_.swa_v_bytes[ip].data(), 0, snap_.swa_v_bytes[ip].size());
            } else if (!snap_.shared_tail_k_bytes[ip].empty()) {
                const int n_tail_rows = (tc.planes[ip] - 1) * tc.shared_tail_cap;
                const size_t seg_k = (size_t) n_tail_rows * tc.prefix_k[ip]->nb[1];
                const size_t seg_v = (size_t) n_tail_rows * tc.prefix_v[ip]->nb[1];
                const size_t off_k = (size_t) tc.global_pool_cap * tc.prefix_k[ip]->nb[1];
                const size_t off_v = (size_t) tc.global_pool_cap * tc.prefix_v[ip]->nb[1];
                for (int h = 0; h < w.n_head_kv; ++h) {
                    ggml_backend_tensor_set(tc.prefix_k[ip], snap_.shared_tail_k_bytes[ip].data() + (size_t) h * seg_k,
                                            (size_t) h * tc.prefix_k[ip]->nb[2] + off_k, seg_k);
                    ggml_backend_tensor_set(tc.prefix_v[ip], snap_.shared_tail_v_bytes[ip].data() + (size_t) h * seg_v,
                                            (size_t) h * tc.prefix_v[ip]->nb[2] + off_v, seg_v);
                }
            }
        }
        tc.prefix_len = snap_.rewind_pos;
    } else if (!snap_.is_tree) {
        const int n_slots = w.n_layer_eff;
        for (int il = 0; il < n_slots; ++il) {
            if (cache.conv_state[il] && !snap_.conv_bytes[il].empty()) {
                ggml_backend_tensor_set(cache.conv_state[il], snap_.conv_bytes[il].data(), 0, snap_.conv_bytes[il].size());
            }
            const int ip = w.phys_layer_for_slot(il);
            if (w.layers[ip].is_swa && !snap_.swa_k_bytes[il].empty()) {
                ggml_backend_tensor_set(cache.k_cache[il], snap_.swa_k_bytes[il].data(), 0, snap_.swa_k_bytes[il].size());
                ggml_backend_tensor_set(cache.v_cache[il], snap_.swa_v_bytes[il].data(), 0, snap_.swa_v_bytes[il].size());
            }
        }
    }
    cache.cur_pos        = snap_.rewind_pos;
    cache.dspark_ctx_pos = snap_.dspark_ctx_pos;
    return true;
}

void Inkling2KvFlash::clear_rewind_snap() {
    snap_ = {};
}

bool Inkling2KvFlash::score_sealed_chunks(
        int sealed_chunks,
        std::vector<float> & out_chunk_scores,
        bool * out_used_gpu,
        bool * out_used_fused) {
    out_chunk_scores.clear();
    if (out_used_gpu) *out_used_gpu = false;
    if (out_used_fused) *out_used_fused = false;

    const int nq = captured_queries();
    if (!active_ || sealed_chunks <= 0 || nq <= 0) return false;

    {
        ScopedGpuDevice guard(target_gpu_);
        pager_->synchronize_paging();
    }

    const size_t qsize = (size_t) n_global_ * n_head_q_ * head_dim_;
    std::vector<const float *> qs;
    qs.reserve((size_t) nq);
    for (int i = 0; i < nq; ++i) {
        qs.push_back(q_tail_.data() + (size_t) i * qsize);
    }

    KvFlashQkDims dims;
    dims.n_layers   = n_global_;
    dims.n_q_heads  = n_head_q_;
    dims.n_kv_heads = n_head_kv_;
    dims.head_dim   = head_dim_;

    const int ct = cfg_.chunk_tokens;
    std::vector<std::vector<float>> outs;
    std::vector<KvFlashMassPlanes> planes;
    bool scored = false;

    // 1. Primary path: GPU Mode-4 scorer (KvFlashGpuMass, fused NBLK=4 CT=64 for Q8_0!)
    //    Runs on expert_backend (Strix Halo) when Tier 1.5 is active and preferred, else target_backend.
    if (!gpu_scorer_failed_) {
        const bool use_expert = (tier15_active_ && cfg_.score_on_expert && expert_backend_ != nullptr);
        ggml_backend_t s_be = use_expert ? expert_backend_ : target_backend_;
        const int s_gpu = use_expert ? expert_gpu_ : target_gpu_;
        const auto & pool_k = use_expert ? tier15_k_ : tier0_k_;

        std::vector<int> block_of((size_t) sealed_chunks, -1);
        std::vector<uint64_t> versions((size_t) sealed_chunks, 0);
        // When scoring on expert_backend (Tier 1.5), resident Tier-0 chunks need a tiny host copy
        // so host_seg can supply them to Strix's archive for exact softmax normalization Z.
        std::vector<uint8_t> tier0_host_stage;
        std::vector<int> tier0_stage_idx;
        const size_t k_seg = pager_->k_seg_bytes();
        const size_t chunk_k_bytes = (size_t) n_global_ * n_head_kv_ * k_seg;

        if (use_expert) {
            tier0_stage_idx.assign((size_t) sealed_chunks, -1);
            int n_stage = 0;
            for (int c = 0; c < sealed_chunks; ++c) {
                block_of[(size_t) c] = tier15_block_of_[(size_t) c];
                versions[(size_t) c] = pager_->chunk_version(c);
                if (block_of[(size_t) c] < 0 && pager_->block_of(c) >= 0) {
                    tier0_stage_idx[(size_t) c] = n_stage++;
                }
            }
            if (n_stage > 0) {
                tier0_host_stage.resize((size_t) n_stage * chunk_k_bytes);
                ScopedGpuDevice guard(target_gpu_);
                for (int c = 0; c < sealed_chunks; ++c) {
                    const int si = tier0_stage_idx[(size_t) c];
                    if (si < 0) continue;
                    const int b0 = pager_->block_of(c);
                    uint8_t * dst = tier0_host_stage.data() + (size_t) si * chunk_k_bytes;
                    for (int l = 0; l < n_global_; ++l) {
                        ggml_tensor * tk = tier0_k_[(size_t) l];
                        const size_t base = (size_t) b0 * ct * tk->nb[1];
                        for (int h = 0; h < n_head_kv_; ++h) {
                            ggml_backend_tensor_get(tk, dst + ((size_t) l * n_head_kv_ + h) * k_seg,
                                                    base + (size_t) h * tk->nb[2], k_seg);
                        }
                    }
                }
            }
        } else {
            for (int c = 0; c < sealed_chunks; ++c) {
                block_of[(size_t) c] = pager_->block_of(c);
                versions[(size_t) c] = pager_->chunk_version(c);
            }
        }

        auto host_seg = [&](int c, int l, int h) -> const uint8_t * {
            if (use_expert && c >= 0 && c < sealed_chunks && tier0_stage_idx[(size_t) c] >= 0) {
                const int si = tier0_stage_idx[(size_t) c];
                return tier0_host_stage.data() + (size_t) si * chunk_k_bytes + ((size_t) l * n_head_kv_ + h) * k_seg;
            }
            return pager_->host_k_seg(c, l, h);
        };

        ScopedGpuDevice guard(s_gpu);
        if (!gpu_mass_) {
            gpu_mass_ = std::make_unique<KvFlashGpuMass>(s_be);
        }
        const bool want_fused = (env_int_or("LUCE_KVFLASH_GPU_SCORE_FUSED", 1) != 0) &&
                                KvFlashGpuMass::fused_available(pool_k[0]->type, dims.head_dim, dims.n_layers);
        gpu_mass_->set_fused(want_fused);
        const bool fused = gpu_mass_->fused();

        KvFlashGpuMassTiming tm;
        std::string why;
        if (gpu_mass_->score(pool_k, block_of, host_seg, ct, dims, qs, outs, &planes,
                             &tm, &why, fused ? &versions : nullptr)) {
            scored = true;
            if (out_used_gpu) *out_used_gpu = true;
            if (out_used_fused) *out_used_fused = tm.fused;
            std::fprintf(stderr,
                "[inkling2-kvflash] gpu scorer (dev=%d): n_q=%d sealed=%d gather=%.1f h2d=%.1f "
                "graph=%.1f d2h=%.1f total=%.1f ms uploaded=%d archived=%d fused=%d\n",
                s_gpu, nq, sealed_chunks, tm.gather, tm.h2d, tm.graph, tm.d2h, tm.total,
                tm.uploaded, tm.archived, (int) tm.fused);
        } else {
            if (kvflash_gpu_failure_latches(why)) gpu_scorer_failed_ = true;
            std::fprintf(stderr, "[inkling2-kvflash] gpu scorer fallback to CPU: %s\n", why.c_str());
        }
    }

    // 2. CPU exact fallback (kvflash_exact_mass_scores_fast_multi)
    if (!scored) {
        const size_t k_seg = pager_->k_seg_bytes();
        const size_t layer_bytes = (size_t) pager_->n_pool_blocks() * n_head_kv_ * k_seg;
        std::vector<uint8_t> pool_host((size_t) n_global_ * layer_bytes);
        {
            ScopedGpuDevice guard(target_gpu_);
            for (int l = 0; l < n_global_; ++l) {
                ggml_tensor * tk = tier0_k_[(size_t) l];
                for (int b = 0; b < pager_->n_pool_blocks(); ++b) {
                    const size_t base = (size_t) b * ct * tk->nb[1];
                    for (int h = 0; h < n_head_kv_; ++h) {
                        uint8_t * dst = pool_host.data() + (size_t) l * layer_bytes +
                                        ((size_t) b * n_head_kv_ + h) * k_seg;
                        ggml_backend_tensor_get(tk, dst, base + (size_t) h * tk->nb[2], k_seg);
                    }
                }
            }
        }
        KvFlashExactKeys keys;
        keys.n_chunks     = sealed_chunks;
        keys.chunk_tokens = ct;
        keys.type         = tier0_k_[0]->type;
        keys.seg = [&](int c, int l, int h) -> const uint8_t * {
            const int b = pager_->block_of(c);
            if (b >= 0) {
                return pool_host.data() + (size_t) l * layer_bytes + ((size_t) b * n_head_kv_ + h) * k_seg;
            }
            return pager_->host_k_seg(c, l, h);
        };
        scored = kvflash_exact_mass_scores_fast_multi(keys, dims, qs, outs, 0, nullptr, &planes);
    }

    if (!scored || planes.empty()) return false;
    out_chunk_scores = kvflash_select_scores(planes);
    last_scores_ = out_chunk_scores;
    return true;
}

bool Inkling2KvFlash::mode4_score_and_recall(
        int rewind_pos,
        int prompt_end,
        Inkling2KvFlashRecallStats * out_stats) {
    Inkling2KvFlashRecallStats st;
    if (out_stats) *out_stats = st;
    if (!active_ || !cfg_.mode4_enabled || cfg_.recall_blocks <= 0) return false;

    const int ct = cfg_.chunk_tokens;
    const int sealed = rewind_pos / ct;
    if (sealed <= cfg_.sink_chunks + 1 || captured_queries() <= 0) return false;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> scores;
    bool used_gpu = false, used_fused = false;
    if (!score_sealed_chunks(sealed, scores, &used_gpu, &used_fused)) {
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();

    // Plan which cold blocks to recall into Tier 0 while leaving room for [rewind_pos, prompt_end)
    KvFlashRecallIn rin;
    rin.n_pool_blocks      = pager_->n_pool_blocks();
    rin.cur_chunk          = sealed;
    rin.recall_chunks      = cfg_.recall_blocks * cfg_.select_unit;
    rin.unit               = cfg_.select_unit;
    rin.reserve            = (prompt_end - rewind_pos + ct - 1) / ct + 1;
    rin.pin_reserve_chunks = 0;
    rin.pinned_chunks      = cfg_.sink_chunks;

    const int first_window = kvflash_recall_first_window(rin);
    const int k_units      = rin.unit > 0 ? (rin.recall_chunks / rin.unit) : 0;
    // Allow every cold chunk >= sink_chunks to be eligible for recall (even chunks 2..9 at pos 128..639)
    const int first_chunk  = std::max(cfg_.sink_chunks, rin.unit);
    const std::vector<int> picked = kvflash_recall_pick(scores, first_window, k_units, rin.unit, first_chunk);
    const int tail_chunks  = std::max(cfg_.hot_tail_chunks, std::max(0, sealed - first_window));

    // Build boosted score vector: recalled blocks rank strictly above all non-recalled cold chunks
    // and stay protected during both [rewind_pos, prompt_end) re-prefill and decode!
    active_hook_scores_ = scores;
    active_hook_sealed_ = sealed;
    for (size_t i = 0; i < picked.size(); ++i) {
        const int b = picked[i];
        const float boost = 1e6f + (float) (picked.size() - i) * 1e3f;
        for (int j = 0; j < cfg_.select_unit; ++j) {
            const int c = b * cfg_.select_unit + j;
            if (c >= 0 && c < (int) active_hook_scores_.size()) {
                active_hook_scores_[(size_t) c] = boost;
            }
        }
    }

    int events = 0;
    {
        ScopedGpuDevice guard(target_gpu_);
        // 1. Truncate Tier 0 back to rewind_pos (frees blocks written by first pass of [R, end))
        pager_->truncate(rewind_pos);
        // 2. Keep relevance score_hook installed for re-prefill & decode (cleared on reset())
        pager_->score_hook = [this](int c) -> float {
            return kvflash_hook_score(active_hook_scores_, active_hook_sealed_, c);
        };
        events = pager_->reselect(tail_chunks);
    }
    sync_tier15_and_tables_();
    const auto t2 = std::chrono::steady_clock::now();

    st.ran             = true;
    st.recalled_any    = !picked.empty();
    st.sealed_chunks   = sealed;
    st.n_queries       = captured_queries();
    st.page_events     = events;
    st.score_ms        = std::chrono::duration<double, std::milli>(t1 - t0).count();
    st.reselect_ms     = std::chrono::duration<double, std::milli>(t2 - t1).count();
    st.used_gpu_scorer = used_gpu;
    st.used_fused      = used_fused;
    st.picked_blocks   = picked;
    if (out_stats) *out_stats = st;

    std::string picked_str;
    for (size_t i = 0; i < picked.size(); ++i) {
        char buf[32];
        const int c0 = picked[i] * cfg_.select_unit;
        const int c1 = c0 + cfg_.select_unit - 1;
        std::snprintf(buf, sizeof(buf), "%s%d(c%d-%d)", i ? "," : "", picked[i], c0, c1);
        picked_str += buf;
    }
    std::fprintf(stderr,
        "[inkling2-kvflash] mode4 recall @R=%d/%d: sealed=%d n_q=%d picked=[%s] "
        "tail=%dch events=%d score=%.1fms reselect=%.1fms\n",
        rewind_pos, prompt_end, sealed, st.n_queries, picked_str.c_str(),
        tail_chunks, events, st.score_ms, st.reselect_ms);
    return true;
}

} // namespace luce::common
