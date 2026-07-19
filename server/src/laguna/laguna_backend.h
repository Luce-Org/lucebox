// LagunaBackend: ModelBackend implementation for Poolside Laguna-XS.2.
//
// Wraps the existing laguna target weights, cache, snapshots, and forward
// step behind the generic ModelBackend interface so the daemon loop in
// daemon_loop.cpp can drive laguna models without laguna-specific code.

#pragma once

#include "model_backend.h"
#include "laguna_internal.h"
#include "laguna_dflash_target.h"
#include "common/dflash_feature_ring.h"
#include "common/dflash_draft_graph.h"
#include "common/dflash_draft_kv.h"
#include "placement/placement_config.h"
#include "qwen3_drafter.h"
#include "kvflash_pager.h"
#include "kvflash_scorer.h"
#include "../common/moe_hybrid_ffn_eval.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_swap_manager.h"
#include "../common/moe_hybrid_stream.h"
#include "../common/moe_routing_collector.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct LagunaBackendArgs {
    std::string target_path;
    std::string draft_path;
    int         draft_gpu = -1;
    int         draft_ctx_max = 2048;
    bool        ddtree_mode = false;
    int         ddtree_budget = 22;
    float       ddtree_temp = 1.0f;
    int         verify_width = 0;   // chain verify width; 0 = adaptive (AUTO)
    DevicePlacement device;
    int         max_ctx   = 16384;
    int         chunk     = 2048;
    ggml_type   kv_type   = GGML_TYPE_Q8_0;
};

struct LagunaDraftVariant {
    std::string                name;
    DraftWeights               weights;
};

class LagunaBackend : public ModelBackend {
public:
    explicit LagunaBackend(const LagunaBackendArgs & args);
    ~LagunaBackend() override;

    // Initialise CUDA backend, load weights, create cache.
    // Returns false on failure (prints to stderr).
    bool init();

    // ── ModelBackend interface ────────────────────────────────────────
    void print_ready_banner() const override;

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return target_parked_; }
    bool spark_wants_bootstrap() const override;
    bool spark_bootstrap_finalize(const std::string & profile_path) override;

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool handle_compress(const std::string & line,
                          const DaemonIO & io) override;
    void free_drafter() override;

    void shutdown() override;

    bool set_routing_collector(MoeRoutingCollector * c) override { routing_collector_ = c; return true; }
    const MoeHybridRoutingStats * get_routing_stats() const override { return routing_stats_.get(); }

private:
    LagunaBackendArgs                           args_;
    ggml_backend_t                              backend_   = nullptr;
    ggml_backend_t                              snap_backend_ = nullptr;
    LagunaTargetWeights                         w_;
    LagunaTargetCache                           cache_;
    std::array<LagunaCacheSnapshot, kMaxSlots>  snapshots_{};
    SamplerCfg                                  sampler_;
    std::mt19937_64                             sampler_rng_{std::random_device{}()};
    bool                                        target_parked_ = false;

    // DFlash speculative decode
    ggml_backend_t                              draft_backend_ = nullptr;
    std::vector<LagunaDraftVariant>             draft_variants_;
    DraftWeights *                              active_dw_ = nullptr;
    std::string                                 default_draft_variant_ = "base";
    DraftFeatureMirror                          feature_mirror_{};
    // [TAG_DRAFT_KV] drafter context-KV ring cache (lazy-init on first spec
    // decode; kill with DFLASH_DRAFT_KV=0). Replaces the per-step full-window
    // K/V recompute once the feature window fills.
    DraftKvState                                draft_kv_{};
    LagunaDFlashTarget *                        dflash_target_ = nullptr;
    bool                                        draft_parked_ = false;
    // [TAG_LAGUNA_VERIFY_WIDTH] EWMA of the accepted block length, persisted
    // across requests. Drives the AUTO chain verify width (seeded for width 3).
    double                                      spec_ewma_accept_ = 1.5;

    // PFlash drafter (lazy-loaded on first compress command).
    DrafterContext                              drafter_ctx_{};
    bool                                       drafter_loaded_ = false;

    // ── Hybrid MoE mode (hot/cold expert split) ──
    bool                                       hybrid_mode_ = false;
    std::shared_ptr<MoeHybridStorage>          moe_hybrid_;
    std::shared_ptr<MoeHybridRoutingStats>     routing_stats_;
    std::string                                routing_stats_out_path_;
    int                                        cache_slots_ = -1;  // Spark auto-sized (-1=unset)
    uint64_t                                   spark_expert_budget_ = 0;  // hot budget for rebuild
    std::vector<uint64_t>                      layer_expert_bytes_;       // per-layer 1-expert bytes
    MoeHybridSwapPolicy                        swap_policy_;
    bool                                       hybrid_telemetry_ = false;
    MoeHybridStreamEngine                      stream_engine_;
    MoeRoutingCollector *                       routing_collector_ = nullptr;
    MoeExpertComputeRuntime                    expert_runtime_;

    bool ensure_slot(int slot);

    // ── kvflash (bounded KV residency; see common/kvflash_pager.h) ──
    // Drafter-scored residency by default: the Qwen3-0.6B drafter scores
    // chunks through the cross-tokenizer bridge (KvFlashCrossTokScorer —
    // relevance is text-level, so the target's ids are detokenized and
    // re-tokenized for the drafter). LRU is the fallback when no drafter is
    // found or --kvflash-policy lru. The pager covers ALL 40 layers; SWA
    // exactness comes from a protected tail >= sliding_window.
    KvFlashPager                   kvflash_pager_;
    std::unique_ptr<KvFlashScorer> kvflash_scorer_;
    std::vector<float>             kvflash_scores_;
    std::string                    kvflash_drafter_path_;
    int          kvflash_tokens_ = 0;     // 0 = off
    int          kvflash_tau_    = 64;
    bool         kvflash_drafter_failed_ = false;
    bool kvflash_active() const { return kvflash_tokens_ > 0; }
    // True iff hybrid placement showed all experts fit hot with the FULL
    // max_ctx KV reservation. In that case KVFlash is redundant and can be
    // disabled before cache creation.
    bool placement_all_hot_full_kv_ = false;
    // Drafter rescore + repage every effective-tau generated tokens
    // (lazy-loads the drafter + cross-tokenizer scorer on first need).
    void kvflash_maybe_reselect(const std::vector<int32_t> & history, int generated);
    // Pager protections (SWA tail) shared by the floor and attach.
    KvFlashConfig kvflash_config() const;
    // Shared pool sizing inputs for placement and runtime allocation.
    void kvflash_resolve_drafter();
    bool kvflash_scorer_expected() const { return !kvflash_drafter_path_.empty(); }
    KvFlashAutoBudget make_kvflash_budget(int64_t gpu_free) const;
    // Read DFLASH_KVFLASH and round/clamp; call before cache creation.
    void kvflash_read_config();
    // Attach the pager to the freshly created cache (init / unpark).
    bool kvflash_attach();
    // Allocate pool slots for [kv_start, kv_start+n_tok) (evicting LRU as
    // needed) ahead of a laguna_step call. False if the pool is exhausted.
    bool kvflash_alloc_span(int kv_start, int n_tok);

    // Hybrid mode helpers
    bool init_hybrid_mode();
    // Build hot/cold expert storage for `placement` by re-reading expert weights
    // from the GGUF mmap (partial-load mode keeps no full expert tensors resident).
    // Used by both init and post-request swap so the two paths can never diverge.
    bool build_hybrid_storage_from_file(const MoeHybridPlacement & placement,
                                        std::shared_ptr<MoeHybridStorage> & out_storage,
                                        std::string & err);
    bool ensure_moe_expert_compute();
    GenerateResult generate_hybrid(const GenerateRequest & req, const DaemonIO & io);
    bool hybrid_forward_one_token(int32_t tok, int kv_pos,
                                  std::vector<float> & act_cur,
                                  int32_t & argmax_out);
    void maybe_post_request_swap();

    bool load_decode_draft();
    bool select_decode_draft(const std::string & name);
    void free_decode_draft();
    bool do_spec_decode(int committed, int n_gen,
                        std::vector<int32_t> & out_tokens,
                        const DaemonIO & io,
                        const BudgetHook * budget_hook = nullptr,
                        bool * forced_close_out = nullptr,
                        float * accept_rate_out = nullptr,
                        const std::vector<int32_t> * sample_history_prefix = nullptr);
};

}  // namespace dflash::common
