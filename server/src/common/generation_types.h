#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sampler.h"

namespace dflash::common {

// Called once for each token committed by transitional whole-request
// executors. Returning false requests cancellation.
using TokenCallback = std::function<bool(int32_t token)>;

// Pre-commit token substitution for thinking budgets. When the remaining
// generation budget reaches hard_limit_remaining, close_token_ids replaces
// the next sampled tokens in order so reusable KV state sees the replacement.
struct BudgetHook {
    std::vector<int32_t> close_token_ids;
    int hard_limit_remaining = 0;
};

struct GenerateRequest {
    std::vector<int32_t> prompt;
    int n_gen = 0;
    SamplerCfg sampler;
    bool do_sample = false;
    bool stream = false;
    // Existing physical snapshot coordinates during the cache migration.
    int snap_pos = -1;
    int snap_slot = -1;
    // Transitional callback used below the future Generation channel adapter.
    TokenCallback on_token;
    // Model-ready optional sequences are owned because the engine may retain
    // the request after its caller returns.
    std::vector<int32_t> hint_tokens;
    std::vector<int32_t> stall_tool_prefix_tokens;
    std::vector<int32_t> stall_action_suffix_tokens;
    std::vector<int32_t> stall_skip_tokens;
    BudgetHook budget_hook;
    // Set only for the single autoregressive retry after empty spec output.
    bool force_ar_decode = false;
};

// Backend-independent failure categories. generate_error_code() is their
// stable wire representation once a value is published.
enum class GenerateErrorCode {
    Incomplete,
    AdapterUnavailable,
    ContextOverflow,
    SamplingUnsupported,
    PrefillFailed,
    DecodeSeedMissing,
    DecodeFailed,
    InvalidSnapshotSlot,
    ModelParked,
    Cancelled,
    OutputBackpressure,
    Overloaded,
    ShuttingDown,
    BackendSpecific,
};

constexpr std::string_view generate_error_code(GenerateErrorCode error) {
    switch (error) {
    case GenerateErrorCode::Incomplete:          return "incomplete";
    case GenerateErrorCode::AdapterUnavailable:  return "adapter_unavailable";
    case GenerateErrorCode::ContextOverflow:     return "context_overflow";
    case GenerateErrorCode::SamplingUnsupported: return "sampling_unsupported";
    case GenerateErrorCode::PrefillFailed:       return "prefill_failed";
    case GenerateErrorCode::DecodeSeedMissing:   return "decode_seed_missing";
    case GenerateErrorCode::DecodeFailed:        return "decode_failed";
    case GenerateErrorCode::InvalidSnapshotSlot: return "invalid_snapshot_slot";
    case GenerateErrorCode::ModelParked:         return "model_parked";
    case GenerateErrorCode::Cancelled:           return "cancelled";
    case GenerateErrorCode::OutputBackpressure:  return "output_backpressure";
    case GenerateErrorCode::Overloaded:          return "overloaded";
    case GenerateErrorCode::ShuttingDown:        return "shutting_down";
    case GenerateErrorCode::BackendSpecific:     return "backend_specific";
    }
    return "unknown_error";
}

struct GenerateError {
    GenerateErrorCode code = GenerateErrorCode::Incomplete;
    std::string detail;
};

struct GenerateResult {
    // A producer must explicitly call succeed() before returning success.
    std::optional<GenerateError> error = GenerateError{};
    std::vector<int32_t> tokens;
    double prefill_s = 0.0;
    double decode_s = 0.0;
    // Prompt tokens confirmed by the backend's physical snapshot restore.
    int restored_prefix_tokens = 0;
    // Distinguishes engine-injected thinking closure from a natural close.
    bool budget_forced_close = false;
    // True when the post-close watchdog stopped a repetition loop.
    bool degenerate_decode_close = false;
    // accepted_draft_tokens / total_draft_positions; zero without spec decode.
    float accept_rate = 0.0f;
    // Separates a zero accept rate from an autoregressive execution.
    bool spec_decode_ran = false;
    // The attempt emitted only tokens suppressed by the response layer and is
    // eligible for the same one-time retry as an empty token vector.
    bool empty_visible_output = false;

    bool ok() const {
        return !error.has_value();
    }

    std::string_view error_code() const {
        return error ? generate_error_code(error->code) : std::string_view{};
    }

    std::string_view error_detail() const {
        return error ? std::string_view(error->detail) : std::string_view{};
    }

    void succeed() {
        error.reset();
    }

    void fail(GenerateErrorCode code, std::string detail = {}) {
        error = GenerateError{code, std::move(detail)};
    }
};

} // namespace dflash::common
