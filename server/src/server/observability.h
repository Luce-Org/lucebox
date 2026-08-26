#pragma once

#include "common/observability/inference_profile.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dflash::common::observability {

struct ObservabilityConfig {
    bool enabled = false;
    std::string output_path = "concurrency-profile.jsonl";
    size_t max_rounds = 10000;
    size_t max_requests = 4096;
    size_t max_token_bursts = 200000;
    uint64_t checkpoint_every_rounds = 0;
    std::string git_sha = "unknown";
    std::string model_name;
    std::string model_path;
    std::string draft_path;
    std::string arch;
    std::string runtime_backend;
    int max_concurrency = 1;
    int ddtree_budget = 0;
    int draft_block_size = 0;
    std::vector<std::pair<std::string, std::string>> run_env;

    static ObservabilityConfig from_env();
};

struct LastStepSnapshot {
    uint64_t round_id = 0;
    uint64_t duration_ns = 0;
    StepPath path = StepPath::Unknown;
    uint32_t queue_depth = 0;
    uint32_t live_slots = 0;
};

struct LiveMetricsSnapshot {
    bool enabled = false;
    uint32_t schema_version = kProfileSchemaVersion;
    uint64_t rounds = 0;
    uint32_t queue_depth = 0;
    uint32_t live_slots = 0;
    uint32_t kv_blocks_total = 0;
    uint32_t kv_blocks_free = 0;
    uint64_t planned_prefill_tokens = 0;
    uint64_t executed_prefill_tokens = 0;
    uint64_t decode_lanes = 0;
    uint64_t durable_decode_tokens = 0;
    uint64_t spec_eligible_lanes = 0;
    uint64_t spec_reserved_lanes = 0;
    uint64_t spec_attempted_lanes = 0;
    uint64_t spec_proposed_draft_tokens = 0;
    uint64_t spec_verified_draft_tokens = 0;
    uint64_t spec_accepted_draft_tokens = 0;
    uint64_t spec_durable_draft_tokens = 0;
    uint64_t spec_scheduler_consumed_tokens = 0;
    uint64_t target_rows = 0;
    uint64_t target_padding_rows = 0;
    uint64_t draft_rows = 0;
    uint64_t draft_padding_rows = 0;
    uint64_t requests_completed = 0;
    uint64_t requests_failed = 0;
    uint64_t dropped_steps = 0;
    uint64_t dropped_requests = 0;
    uint64_t dropped_token_bursts = 0;
    std::array<uint64_t, kPhaseCount> phase_ns{};
    LastStepSnapshot last_step;
};

class ObservabilityState final : public ProfileSink {
public:
    explicit ObservabilityState(ObservabilityConfig config);
    ~ObservabilityState();

    bool enabled() const noexcept { return config_.enabled; }

    uint64_t job_queued() noexcept;
    void job_dequeued() noexcept;
    uint32_t queue_depth() const noexcept;
    void set_live_slots(uint32_t live_slots);

    StepProfile * begin_step(uint32_t live_slots) noexcept override;
    void commit_step(StepProfile * profile) override;

    void record_request_admitted(
        uint64_t request_id, std::string response_id,
        uint32_t prompt_tokens, uint64_t queued_ns,
        uint64_t admitted_ns);
    void record_prefill_completed(uint64_t request_id, uint64_t now_ns);
    void record_token_burst(
        uint64_t request_id, uint64_t round_id,
        uint64_t ready_ns, uint32_t token_count);
    void record_request_finished(
        uint64_t request_id, bool ok, uint32_t output_tokens,
        uint64_t completed_ns);

    LiveMetricsSnapshot snapshot() const;
    std::string snapshot_json() const;
    void flush();

private:
    struct RequestRecord {
        uint64_t request_id = 0;
        std::string response_id;
        bool ok = false;
        uint32_t prompt_tokens = 0;
        uint32_t output_tokens = 0;
        uint64_t queued_ns = 0;
        uint64_t admitted_ns = 0;
        uint64_t prefill_completed_ns = 0;
        uint64_t first_token_ns = 0;
        uint64_t completed_ns = 0;
    };

    struct TokenBurst {
        uint64_t request_id = 0;
        uint64_t round_id = 0;
        uint64_t ready_ns = 0;
        uint32_t token_count = 0;
    };

    ObservabilityConfig config_;
    std::atomic<uint32_t> queue_depth_{0};
    uint64_t next_round_id_ = 1;
    StepProfile current_step_;

    mutable std::mutex live_mu_;
    LiveMetricsSnapshot live_;

    std::vector<StepProfile> steps_;
    std::vector<RequestRecord> requests_;
    std::vector<TokenBurst> token_bursts_;
    std::unordered_map<uint64_t, size_t> active_requests_;
    uint64_t started_unix_ns_ = 0;
    uint64_t started_steady_ns_ = 0;
    uint64_t last_checkpoint_round_ = 0;
    bool flushed_ = false;

    bool write_capture(bool complete);
};

}
