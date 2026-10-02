// concurrency_capture.h — bounded lucebox.concurrency.v1 JSONL capture for
// the concurrent scheduler (LuceGraph / Lucebox Forge consumer).
//
// Off unless LUCE_PROF lists `concurrency` (or is `1`). Disabled, every entry
// point returns after one branch on a cached bool: no clock read, no
// allocation, no lock, no file. Enabled, the scheduler thread owns all state
// except the queue-depth counter, which client threads bump on enqueue.
//
// Environment:
//   LUCE_PROF=concurrency|1           enable the capture
//   LUCE_PROF_OUT=<path>              output (default concurrency-profile.jsonl)
//   LUCE_PROF_MAX_ROUNDS=<n>          retained step records     (default 10000)
//   LUCE_PROF_MAX_REQUESTS=<n>        retained request records  (default 4096)
//   LUCE_PROF_MAX_TOKEN_BURSTS=<n>    retained token bursts     (default 200000)
//   LUCE_PROF_CHECKPOINT_EVERY=<n>    rewrite a checkpoint every n rounds (0 off)
//
// Retention is keep-first: a full bound counts drops and keeps no partial
// record. Checkpoints and the final export write `<out>.tmp` and atomically
// rename it over `<out>`; checkpoints carry footer.complete=false.

#pragma once

#include "common/observability/inference_profile.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace luce::common::observability {

struct ConcurrencyCaptureConfig {
    bool enabled = false;
    std::string output_path;  // from_env: LUCE_PROF_OUT or the default name
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

    static ConcurrencyCaptureConfig from_env();
};

class ConcurrencyCapture final {
public:
    explicit ConcurrencyCapture(ConcurrencyCaptureConfig config);
    ~ConcurrencyCapture();

    ConcurrencyCapture(const ConcurrencyCapture &) = delete;
    ConcurrencyCapture & operator=(const ConcurrencyCapture &) = delete;

    bool enabled() const noexcept { return config_.enabled; }

    // Queue bookkeeping; job_queued() returns the steady-clock enqueue time
    // (0 when disabled) for the request record.
    uint64_t job_queued() noexcept;
    void job_dequeued() noexcept;

    // Null when disabled. The returned record stays valid until commit_step.
    // `started_ns` backdates the round to when planning began (0 = now).
    StepProfile * begin_step(uint64_t round_id, uint32_t live_slots,
                             uint64_t started_ns = 0) noexcept;
    void commit_step(StepProfile * profile);

    void record_request_admitted(
        uint64_t request_id, const std::string & response_id,
        uint32_t prompt_tokens, uint64_t queued_ns, uint64_t admitted_ns);
    void record_prefill_completed(uint64_t request_id, uint64_t now_ns);
    void record_token_burst(
        uint64_t request_id, uint64_t round_id,
        uint64_t ready_ns, uint32_t token_count);
    void record_request_finished(
        uint64_t request_id, bool ok, uint32_t output_tokens,
        uint64_t completed_ns);

    // Final export (footer.complete=true). Idempotent; also run on destruction.
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

    // Serialize the capture as it stands, in schema record order.
    void write_jsonl(std::ostream & out, bool complete) const;
    bool write_capture(bool complete);

    ConcurrencyCaptureConfig config_;
    std::atomic<uint32_t> queue_depth_{0};
    std::unique_ptr<StepProfile> current_step_;  // allocated only when enabled

    std::vector<StepProfile> steps_;
    std::vector<RequestRecord> requests_;
    std::vector<TokenBurst> token_bursts_;
    std::unordered_map<uint64_t, size_t> active_requests_;
    uint64_t started_unix_ns_ = 0;
    uint64_t started_steady_ns_ = 0;
    uint64_t dropped_steps_ = 0;
    uint64_t dropped_requests_ = 0;
    uint64_t dropped_token_bursts_ = 0;
    uint64_t last_checkpoint_round_ = 0;
    bool flushed_ = false;
};

}  // namespace luce::common::observability
