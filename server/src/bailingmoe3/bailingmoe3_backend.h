#pragma once

#include "qwen35_backend.h"

namespace dflash::common {

// Configuration intentionally exposes only the features the first native
// Ling backend implements. Speculative decode and expert offload can be added
// after the autoregressive path has a logits-equivalent baseline.
struct BailingMoe3Config {
    const char * model_path = nullptr;
    const char * draft_path = nullptr;
    DevicePlacement device;
    int draft_gpu = 0;
    int draft_ctx_max = 4096;
    bool fast_rollback = true;
    int stream_fd = -1;
};

class BailingMoe3Backend final : public Qwen35Backend {
public:
    explicit BailingMoe3Backend(const BailingMoe3Config & cfg);
    ~BailingMoe3Backend() override;

    void print_ready_banner() const override;
    bool supports_dflash_spec_decode() const override { return true; }
    bool supports_remote_draft() const override { return false; }
    void shutdown() override;
    void release_scratch() override;

protected:
    bool load_target_model(ggml_backend_t backend, TargetWeights & out) override;
    void reset_arch_request_state() override;
    bool after_prefill_compute(
        StepGraph & sg,
        int kv_start,
        int n_tokens,
        const int32_t * next_tokens,
        bool is_final_chunk) override;
    bool has_embedded_spec_decode() const override;
    bool run_embedded_spec_decode(
        int committed,
        int n_gen,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io,
        float & out_accept_rate,
        bool & out_spec_ran,
        const BudgetHook * budget_hook,
        bool * forced_close_out,
        bool * degenerate_close_out) override;

private:
    bool ensure_mtp_runtime();
    bool run_mtp_follow(
        StepGraph & graph,
        ggml_tensor * target_hidden,
        const int32_t * next_tokens,
        int kv_start,
        int n_tokens,
        int logical_rows,
        int sample_row,
        int logits_tail_rows,
        int32_t & draft_token);
    bool extend_mtp_drafts(
        int32_t first_draft,
        ggml_tensor * previous_hidden,
        int start_pos,
        int draft_count,
        std::vector<int32_t> & drafts);
    void free_mtp_runtime();

    TargetWeights mtp_cache_weights_;  // non-owning one-layer cache descriptor
    TargetCache mtp_cache_;
    StepGraph mtp_sg_;
    StepGraph mtp_draft_sg_[2];
    bool mtp_runtime_ready_ = false;
    bool mtp_request_ready_ = false;
    int32_t mtp_draft_token_ = -1;
};

}  // namespace dflash::common
