// Thin qwen35moe backend wrapper over the shared qwen35-family runtime.

#pragma once

#include "qwen35_backend.h"
#include "graph_builders.h"
#include "qwen35moe_pipelined_decode.h"
#include "../common/moe_hybrid_ffn_eval.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_stream.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_swap_manager.h"
#include "../common/moe_routing_collector.h"

#include <memory>
#include <string>

namespace dflash::common {

class Qwen35MoeBackend : public Qwen35Backend {
public:
    explicit Qwen35MoeBackend(const Qwen35Config & cfg);
    ~Qwen35MoeBackend() override = default;

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;
    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;
    bool supports_dflash_spec_decode() const override { return !target_weights().moe_hybrid; }

    bool set_routing_collector(MoeRoutingCollector * c) override { routing_collector_ = c; return true; }
    const MoeHybridRoutingStats * get_routing_stats() const override { return routing_stats_.get(); }

protected:
    bool load_target_model(ggml_backend_t backend, TargetWeights & out) override;
    bool post_kvflash_init_gate() override;
    bool run_ar_decode_path(int committed, int n_gen,
                            std::vector<int32_t> & out_tokens,
                            const DaemonIO & io) override;
    bool should_capture_moe_router() const override { return routing_stats_ != nullptr; }
    bool spark_wants_bootstrap() const override;
    bool spark_bootstrap_finalize(const std::string & profile_path) override;
    void after_target_compute(StepGraph & sg, int kv_start, int n_tokens) override;

private:
    // All-hot placement signal for post_kvflash_init_gate(): set when
    // load_target_model takes the all-hot early-return (moe_hybrid null).
    bool placement_all_hot_ = false;
    // True iff all experts fit hot with the FULL max_ctx KV reservation
    // (KVFlash redundant). When false but placement_all_hot_ is true, the pool
    // is what kept experts hot — the gate must NOT disable KVFlash.
    bool placement_all_hot_full_kv_ = false;
    std::shared_ptr<MoeHybridRoutingStats> routing_stats_;
    std::string routing_stats_out_path_;
    std::string placement_out_path_;
    int cache_slots_ = -1;  // Spark auto-sized (-1=unset)
    uint64_t spark_expert_budget_ = 0;      // hot budget, for the bootstrap rebuild
    std::vector<uint64_t> layer_expert_bytes_;
    bool rebuild_hybrid_from_placement(const MoeHybridPlacement & placement, std::string & err);
    MoeHybridSwapPolicy swap_policy_;
    bool hybrid_telemetry_ = false;
    MoeHybridStreamEngine stream_engine_;
    MoeRoutingCollector * routing_collector_ = nullptr;

    void maybe_post_request_swap();
    bool load_dynamic_placement(const char * hotness_path,
                                ggml_backend_t backend,
                                const TargetWeights & w,
                                MoeHybridPlacement & out,
                                std::string * err);

    // Hybrid speculative decode: draft tokens using DFlash draft model,
    // verify via hybrid forward (layer-by-layer with hot/cold FFN).
    bool do_hybrid_spec_decode(int committed, int n_gen,
                               std::vector<int32_t> & out_tokens,
                               const DaemonIO & io,
                               float * accept_rate_out = nullptr);

    // Run one token through hybrid forward, capturing features at capture layers.
    // Returns the logits argmax token. Advances committed by 1.
    bool hybrid_forward_one_token(int32_t tok, int kv_pos,
                                  std::vector<float> & act_cur,
                                  int32_t & argmax_out,
                                  std::vector<float> * logits_out = nullptr);

    // Batched hybrid forward: processes all tokens layer-by-layer (like prefill).
    // Returns argmax for each token. Much faster than sequential hybrid_forward_one_token.
    bool hybrid_forward_batch(const int32_t * tokens, int n_tokens, int base_pos,
                              std::vector<float> & act_cur,
                              std::vector<int32_t> & argmax_out,
                              bool capture_features);

    // Pipelined decode: uses cached DeltaNet graphs + optimized FFN loop
    bool run_pipelined_decode_path(int committed, int n_gen,
                                   std::vector<int32_t> & out_tokens,
                                   const DaemonIO & io);

    // Persistent pipelined state (initialized once, reused across requests)
    std::unique_ptr<struct PipelinedDecodeState> pipe_state_;
    bool ensure_pipe_state(int kv_start);
};

}  // namespace dflash::common
