// Model-agnostic tool-call predictions shared by HTTP and native predictors.

#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace dflash::common {

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

enum class NativeToolPredictorSchedule {
    BeforeModel,
    Overlap,
};

const char * native_tool_predictor_schedule_name(
    NativeToolPredictorSchedule schedule);

bool parse_native_tool_predictor_schedule(
    const std::string & value,
    NativeToolPredictorSchedule & out);

struct SemanticToolPredictorConfig {
    std::string url;
    std::string model;
    std::string native_model_path;
    std::string native_ipc_bin;
    std::string native_work_dir;
    int native_gpu = 0;
    int native_max_ctx = 4096;
    int timeout_ms = 2000;
    int max_tokens = 96;
    NativeToolPredictorSchedule native_schedule =
        NativeToolPredictorSchedule::BeforeModel;
    // Conservative prior used by the measured tool-execution admission
    // policy. The base predictor currently emits no calibrated probability.
    double execution_confidence = 0.75;

    bool http_enabled() const { return !url.empty() && !model.empty(); }
    bool native_enabled() const {
        return !native_model_path.empty() && !native_ipc_bin.empty();
    }
    bool enabled() const { return native_enabled() || http_enabled(); }
    bool native_runs_before_model() const {
        return native_enabled() &&
            native_schedule == NativeToolPredictorSchedule::BeforeModel;
    }
};

struct SemanticToolCall {
    std::string name;
    ordered_json arguments = ordered_json::object();
};

struct SemanticToolPrediction {
    bool ok = false;
    std::string error;
    // Actual predictor used for this result. Native and HTTP fallback paths
    // share one execution gate, so response metadata must not guess.
    std::string source;
    SemanticToolCall call;
    double wall_ms = 0.0;
};

// Parse one OpenAI-compatible sidecar response and reject calls whose
// function name is absent from the request schema.  Arguments remain decoded
// JSON values; sidecar token IDs are never accepted by the target.
bool parse_semantic_tool_prediction(
    const json & response,
    const json & request_tools,
    SemanticToolCall & out,
    std::string & error);

// Materialize top-level defaults declared by the selected function before a
// prediction is executed. This turns an omitted optional default into the
// exact explicit invocation the target may emit; the normal exact-match gate
// still rejects the result if the authoritative call differs.
bool materialize_declared_tool_defaults(
    const json & request_tools,
    SemanticToolCall & call,
    std::string & error);

// Build the small OpenAI-compatible request sent to the predictor.  Only
// dialogue/tool semantics are forwarded; target-only extensions are omitted.
json build_semantic_tool_predictor_request(
    const json & target_request,
    const std::string & sidecar_model,
    int max_tokens);

// Native predictor bridge. The prompt uses the Qwen tool template and the
// decoded response is parsed semantically before any target token IDs exist.
std::string build_native_semantic_tool_predictor_prompt(
    const json & predictor_request,
    std::string & error);

bool parse_native_semantic_tool_prediction(
    const std::string & generated_text,
    const json & request_tools,
    SemanticToolCall & out,
    std::string & error);

}  // namespace dflash::common
