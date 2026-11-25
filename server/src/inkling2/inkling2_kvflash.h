// Inkling2KvFlash — 3-Tier Dual-GPU KVFlash 2.0 Engine + Mode-4 Exact D=128 Recall
// for Inkling 2 Mini (1D 8-plane and 2DSD 10-plane per-loop / 4-plane Shared-KV W=64).
//
// Hierarchy:
//   Tier 0   (target_backend, e.g. R9700)      : Hot paged global pool (block_size=64, hot_tail >= 1024)
//   Tier 1.5 (expert_backend, e.g. Strix Halo) : Optional warm paged global pool (disjoint block table!)
//   Tier 2   (Host pinned RAM via KvFlashPager): Cold 64-token chunks, bit-exact recallable
//
// Mathematical exactness of Mode-4 on Inkling 2:
//   Inkling 2 has no RoPE and clamps relative-position bias at rel_extent = 1024
//   (rel_table[1024] = -sink, a query-only constant). Keeping hot_tail >= 1024 ensures
//   every evictable cold chunk shares the exact same query-only additive bias, which
//   cancels out of softmax chunk ranking. Thus existing KvFlashGpuMass / kvflash_fused_run
//   (NBLK=4, CT=64, including RDNA4 int8 WMMA) is mathematically exact out of the box.

#pragma once

#include "inkling2_internal.h"
#include "common/kvflash_pool_flag.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

struct Inkling2Cache;
class KvFlashPager;
class KvFlashGpuMass;

constexpr int kInkling2KvFlashChunk = 64; // Matches KvFlashPager & kvflash_fused_mass_cuda CT=64

struct Inkling2KvFlashConfig {
    bool enabled            = false;
    int  chunk_tokens       = kInkling2KvFlashChunk;
    int  tier0_pool_tokens  = 0;    // Tier-0 resident pool on target_backend (multiple of 64)
    int  tier15_pool_tokens = 0;    // Tier-1.5 warm pool on expert_backend (0 = disabled)
    int  sink_chunks        = 2;    // Leading chunks never evicted (128 tok)
    int  hot_tail_chunks    = 16;   // Trailing chunks never evicted (>= 1024 tok = rel_extent)
    int  recall_blocks      = 4;    // Mode-4 recall blocks (each = select_unit chunks)
    int  select_unit        = 10;   // Chunks per recall block (default 10 = 640 tok)
    int  select_queries     = 64;   // Trailing queries captured for Mode-4 scoring
    bool mode4_enabled      = true; // LUCE_KVFLASH_PREFILL_RESELECT != 0
    bool score_on_expert    = true; // Prefer expert_backend for Mode-4 scoring when Tier 1.5 active

    // Parse LUCE_KVFLASH* env vars and size Tier 0 & Tier 1.5 pools cleanly.
    static Inkling2KvFlashConfig from_env(
        int max_ctx,
        int rel_extent,
        int n_global_planes,
        size_t global_bytes_per_tok,
        ggml_backend_t target_backend,
        ggml_backend_t expert_backend);
};

// Lightweight host-RAM snapshot of SWA rings + ShortConv states (+ Shared-KV tail rings)
// captured at rewind point R so Mode-4 turn rewind is 100% bit-exact.
struct Inkling2RewindSnap {
    bool valid          = false;
    int  rewind_pos     = 0;
    int  dspark_ctx_pos = 0;
    bool is_tree        = false;
    std::vector<std::vector<uint8_t>> conv_bytes;
    std::vector<std::vector<uint8_t>> swa_k_bytes;
    std::vector<std::vector<uint8_t>> swa_v_bytes;
    std::vector<std::vector<uint8_t>> shared_tail_k_bytes;
    std::vector<std::vector<uint8_t>> shared_tail_v_bytes;
};

struct Inkling2KvFlashRecallStats {
    bool   ran             = false;
    bool   recalled_any    = false;
    int    sealed_chunks   = 0;
    int    n_queries       = 0;
    int    page_events     = 0;
    double score_ms        = 0.0;
    double reselect_ms     = 0.0;
    bool   used_gpu_scorer = false;
    bool   used_fused      = false;
    std::vector<int> picked_blocks;
};

class Inkling2KvFlash {
public:
    Inkling2KvFlash();
    ~Inkling2KvFlash();

    Inkling2KvFlash(const Inkling2KvFlash &) = delete;
    Inkling2KvFlash & operator=(const Inkling2KvFlash &) = delete;

    // Attach to Tier-0 global pool views (size = L_global: 8 for 1D, 10 for 2DSD per-loop,
    // 4 for 2DSD Shared-KV W>0) and optional Tier-1.5 remote global pool views (same L_global).
    bool attach(const Inkling2KvFlashConfig & cfg,
                int max_ctx,
                int head_dim,
                int n_head_q,
                int n_head_kv,
                ggml_backend_t target_backend,
                ggml_backend_t expert_backend,
                const std::vector<ggml_tensor *> & tier0_k,
                const std::vector<ggml_tensor *> & tier0_v,
                const std::vector<ggml_tensor *> & tier15_k = {},
                const std::vector<ggml_tensor *> & tier15_v = {});

    void detach();
    void reset();

    bool active() const { return active_; }
    bool tier15_active() const { return tier15_active_; }
    bool mode4_enabled() const { return active_ && cfg_.mode4_enabled; }
    const Inkling2KvFlashConfig & config() const { return cfg_; }

    int block_size() const { return cfg_.chunk_tokens; }
    int max_blocks() const { return max_blocks_; }
    int n_global_planes() const { return n_global_; }
    int tier0_pool_tokens() const { return cfg_.tier0_pool_tokens; }
    int tier15_pool_tokens() const { return tier15_active_ ? cfg_.tier15_pool_tokens : 0; }
    int select_queries() const { return cfg_.select_queries; }

    // Ensure logical positions [pos0, pos0 + n_tok) are allocated in Tier 0,
    // syncing any Tier-0 evictions to Tier 1.5 (if enabled) and updating host block tables.
    bool prepare_span(int pos0, int n_tok);

    // Host block tables of length max_blocks():
    //   bt_local()[c]  >= 0 iff chunk c is resident in Tier 0 (-1 otherwise)
    //   bt_remote()[c] >= 0 iff chunk c is resident in Tier 1.5 (-1 otherwise)
    // Invariant: !(bt_local()[c] >= 0 && bt_remote()[c] >= 0) for all c.
    const std::vector<int32_t> & bt_local()  const { return bt_local_; }
    const std::vector<int32_t> & bt_remote() const { return bt_remote_; }
    uint64_t tables_epoch() const { return tables_epoch_; }

    // Convenience: fill packed [2 + max_blocks] buffer for TREE_ATTN_REL ({prefix_len, 0, bt...}).
    void fill_tree_prefix_buf(int prefix_len, bool remote, std::vector<int32_t> & out) const;

    // Mode-4 query tail capture ([n_q][L_global * Hq * D] float32, post-clip + log-scale).
    void clear_queries() { q_tail_.clear(); }
    void append_queries(const float * rows, int n_q);
    int  captured_queries() const;

    // Compute exact Mode-4 rewind point R for a prompt of length `prompt_len`.
    // Returns -1 if Mode-4 rewind is not needed (prompt fits Tier 0 or disabled).
    int plan_mode4_rewind(const std::vector<int32_t> & prompt, int32_t user_turn_token_id = -1) const;

    // Snapshot & restore SWA + ShortConv (+ Shared-KV tail) at rewind position R.
    bool save_rewind_snap(const Inkling2Cache & cache, const Inkling2Weights & w, int rewind_pos);
    bool restore_rewind_snap(Inkling2Cache & cache, const Inkling2Weights & w);
    void clear_rewind_snap();
    bool has_rewind_snap() const { return snap_.valid; }

    // Score sealed chunks [0 .. rewind_pos / chunk_tokens) using captured queries,
    // truncate pager to rewind_pos, recall top-K cold blocks into Tier 0, and re-sync Tier 1.5.
    // Returns true if recall ran and modified Tier-0 residency (caller should re-prefill [rewind_pos, prompt_end)).
    bool mode4_score_and_recall(int rewind_pos,
                                int prompt_end,
                                Inkling2KvFlashRecallStats * out_stats = nullptr);

    // Exposed for unit testing / diagnostics.
    int tier15_resident_blocks() const;
    int tier0_resident_blocks() const;
    const std::vector<float> & last_scores() const { return last_scores_; }

    // Direct Mode-4 scoring entry point (used internally and by unit tests).
    bool score_sealed_chunks(int sealed_chunks,
                             std::vector<float> & out_chunk_scores,
                             bool * out_used_gpu = nullptr,
                             bool * out_used_fused = nullptr);

    // Host-RAM prefix checkpoint serialization/restoration (KVFLASH3 layout via KvFlashPager).
    size_t serialized_bytes_estimate(int tokens) const;
    bool   serialize_state(int tokens, std::vector<uint8_t> & out) const;
    bool   restore_state(const std::vector<uint8_t> & blob, int tokens);

private:
    void sync_tier15_and_tables_();
    bool upload_chunk_to_tier15_(int c, int rb);
    int  alloc_tier15_block_(int c);
    void free_tier15_block_(int c);
    void rebuild_block_tables_();
    void reset_scorer_();

    Inkling2KvFlashConfig cfg_;
    bool active_        = false;
    bool tier15_active_ = false;
    int  max_ctx_       = 0;
    int  max_blocks_    = 0;
    int  n_global_      = 0;
    int  head_dim_      = 0;
    int  n_head_q_      = 0;
    int  n_head_kv_     = 0;
    int  target_gpu_    = -1;
    int  expert_gpu_    = -1;
    int  scorer_gpu_    = -1;

    ggml_backend_t target_backend_ = nullptr;
    ggml_backend_t expert_backend_ = nullptr;

    std::vector<ggml_tensor *> tier0_k_;
    std::vector<ggml_tensor *> tier0_v_;
    std::vector<ggml_tensor *> tier15_k_;
    std::vector<ggml_tensor *> tier15_v_;

    std::unique_ptr<KvFlashPager> pager_;

    // Tier-1.5 warm pool state (strictly disjoint from Tier 0)
    int                   tier15_n_blocks_ = 0;
    std::vector<int>      tier15_block_of_;   // [max_blocks], -1 if not in Tier 1.5
    std::vector<uint64_t> tier15_last_use_;   // [max_blocks]
    std::vector<uint64_t> tier15_ver_;        // [max_blocks], chunk_version uploaded to Tier 1.5
    std::vector<int>      tier15_free_blocks_;
    uint64_t              tier15_clock_ = 0;

    // Host mirrors of device block tables
    std::vector<int32_t> bt_local_;
    std::vector<int32_t> bt_remote_;
    uint64_t             tables_epoch_ = 0;

    // Mode-4 query capture + GPU scorer
    std::vector<float>              q_tail_;
    std::vector<float>              last_scores_;
    std::vector<float>              active_hook_scores_;
    int                             active_hook_sealed_ = 0;
    std::unique_ptr<KvFlashGpuMass> gpu_mass_;
    bool                            gpu_scorer_failed_ = false;

    // Mode-4 exact turn rewind checkpoint
    Inkling2RewindSnap snap_;
};

} // namespace luce::common
