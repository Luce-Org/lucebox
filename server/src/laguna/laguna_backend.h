// LagunaBackend: ModelBackend implementation for Poolside Laguna-XS.2.
//
// Wraps the existing laguna target weights, cache, snapshots, and forward
// step behind the generic ModelBackend interface so the daemon loop in
// daemon_loop.cpp can drive laguna models without laguna-specific code.

#pragma once

#include "model_backend.h"
#include "laguna_internal.h"
#include "placement/placement_config.h"
#include "qwen3_drafter.h"
#include "../common/moe_hybrid_ffn_eval.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_swap_manager.h"
#include "../common/moe_hybrid_stream.h"

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
    DevicePlacement device;
    int         max_ctx   = 16384;
    int         chunk     = 2048;
    ggml_type   kv_type   = GGML_TYPE_Q8_0;
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

    bool park(const std::string & what) override;
    bool unpark(const std::string & what) override;
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

private:
    LagunaBackendArgs                           args_;
    ggml_backend_t                              backend_   = nullptr;
    ggml_backend_t                              snap_backend_ = nullptr;
    LagunaTargetWeights                         w_;
    LagunaTargetCache                           cache_;
    std::array<LagunaCacheSnapshot, kMaxSlots>  snapshots_{};
    std::mt19937_64                             sampler_rng_{std::random_device{}()};
    bool                                        target_parked_ = false;

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

    bool ensure_slot(int slot);

    // Hybrid mode helpers
    bool init_hybrid_mode();
    // Build hot/cold expert storage for `placement` by re-reading expert weights
    // from the GGUF mmap (partial-load mode keeps no full expert tensors resident).
    // Used by both init and post-request swap so the two paths can never diverge.
    bool build_hybrid_storage_from_file(const MoeHybridPlacement & placement,
                                        std::shared_ptr<MoeHybridStorage> & out_storage,
                                        std::string & err);
    GenerateResult generate_hybrid(const GenerateRequest & req, const DaemonIO & io);
    bool hybrid_forward_one_token(int32_t tok, int kv_pos,
                                  std::vector<float> & act_cur,
                                  int32_t & argmax_out);
    void maybe_post_request_swap();
};

}  // namespace dflash::common
