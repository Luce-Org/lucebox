// Thin qwen35moe backend wrapper over the shared qwen35-family runtime.

#pragma once

#include "qwen35_backend.h"
#include "graph_builders.h"
#include "qwen35moe_hybrid_ffn_eval.h"
#include "qwen35moe_hybrid_storage.h"
#include "qwen35moe_routing_stats.h"
#include "qwen35moe_swap_manager.h"

#include <memory>
#include <string>

namespace dflash::common {

class Qwen35MoeBackend : public Qwen35Backend {
public:
    explicit Qwen35MoeBackend(const Qwen35Config & cfg);
    ~Qwen35MoeBackend() override = default;

    GenerateResult generate(const GenerateRequest & req,
                            const DaemonIO & io) override;
    GenerateResult restore_and_generate(int slot,
                                        const GenerateRequest & req,
                                        const DaemonIO & io) override;
    bool supports_dflash_spec_decode() const override { return !hybrid_mode_; }

protected:
    bool load_target_model(ggml_backend_t backend, TargetWeights & out) override;
    bool run_ar_decode_path(int committed, int n_gen,
                            std::vector<int32_t> & out_tokens,
                            const DaemonIO & io,
                            const BudgetHook & budget_hook = {},
                            bool * forced_close_out = nullptr,
                            bool * degenerate_close_out = nullptr) override;
    bool should_capture_moe_router() const override { return routing_stats_ != nullptr; }
    void after_target_compute(StepGraph & sg, int kv_start, int n_tokens) override;

private:
    bool hybrid_mode_ = false;
    std::shared_ptr<Qwen35MoeRoutingStats> routing_stats_;
    std::string routing_stats_out_path_;
    std::string placement_out_path_;
    Qwen35MoeSwapPolicy swap_policy_;
    uint64_t last_hot_selected_ = 0;
    uint64_t last_cold_selected_ = 0;
    bool hybrid_telemetry_ = false;

    void maybe_post_request_swap();
    bool load_dynamic_placement(const char * hotness_path,
                                ggml_backend_t backend,
                                const TargetWeights & w,
                                Qwen35MoeExpertPlacement & out,
                                std::string * err);

    // Hybrid speculative decode: draft tokens using DFlash draft model,
    // verify via hybrid forward (layer-by-layer with hot/cold FFN).
    bool do_hybrid_spec_decode(int committed, int n_gen,
                               std::vector<int32_t> & out_tokens,
                               const DaemonIO & io);

    // Run one token through hybrid forward, capturing features at capture layers.
    // Returns the logits argmax token. Advances committed by 1.
    bool hybrid_forward_one_token(int32_t tok, int kv_pos,
                                  std::vector<float> & act_cur,
                                  int32_t & argmax_out);
};

}  // namespace dflash::common
