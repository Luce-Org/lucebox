#include "native_semantic_tool_predictor.h"

#include <chrono>
#include <utility>
#include <vector>

namespace dflash::common {

std::shared_ptr<NativeSemanticToolPredictor>
NativeSemanticToolPredictor::create(
        const SemanticToolPredictorConfig & config,
        std::string & error) {
    error.clear();
    if (!config.native_enabled()) {
        error = "native_predictor_config_incomplete";
        return nullptr;
    }
    auto predictor = std::shared_ptr<NativeSemanticToolPredictor>(
        new NativeSemanticToolPredictor(config));
    if (!predictor->tokenizer_.load_from_gguf(
            config.native_model_path.c_str())) {
        error = "native_predictor_tokenizer_load_failed";
        return nullptr;
    }
    if (!predictor->ipc_.start(
            config.native_ipc_bin, config.native_model_path,
            config.native_gpu, config.native_max_ctx,
            config.native_work_dir)) {
        error = "native_predictor_ipc_start_failed";
        return nullptr;
    }
    return predictor;
}

SemanticToolPrediction NativeSemanticToolPredictor::predict(
        const json & predictor_request,
        const json & request_tools,
        std::string * generated_text) {
    const auto started = std::chrono::steady_clock::now();
    SemanticToolPrediction prediction;
    prediction.source = "native-qwen3";
    auto finish = [&]() {
        prediction.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return prediction;
    };

    std::string prompt_error;
    const std::string prompt = build_native_semantic_tool_predictor_prompt(
        predictor_request, prompt_error);
    if (prompt.empty()) {
        prediction.error = std::move(prompt_error);
        return finish();
    }
    const std::vector<int32_t> prompt_ids = tokenizer_.encode(prompt);
    if (prompt_ids.empty()) {
        prediction.error = "native_predictor_prompt_tokenization_failed";
        return finish();
    }
    if (prompt_ids.size() + static_cast<size_t>(config_.max_tokens) >
        static_cast<size_t>(config_.native_max_ctx)) {
        prediction.error = "native_predictor_context_overflow";
        return finish();
    }

    std::vector<int32_t> output_ids;
    if (!ipc_.predict(prompt_ids, config_.max_tokens,
                      output_ids, prediction.error)) {
        return finish();
    }
    const std::string generated = tokenizer_.decode(output_ids);
    if (generated_text) *generated_text = generated;
    if (!parse_native_semantic_tool_prediction(
            generated, request_tools, prediction.call, prediction.error)) {
        return finish();
    }
    if (!materialize_declared_tool_defaults(
            request_tools, prediction.call, prediction.error)) {
        return finish();
    }
    prediction.ok = true;
    return finish();
}

}  // namespace dflash::common
