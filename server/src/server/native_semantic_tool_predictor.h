// Native Qwen semantic tool predictor built from the PFlash/Qwen runtime.

#pragma once

#include "semantic_tool_hint.h"

#include "common/qwen3_tool_predictor_ipc.h"
#include "tokenizer.h"

#include <memory>
#include <string>

namespace dflash::common {

class NativeSemanticToolPredictor {
public:
    static std::shared_ptr<NativeSemanticToolPredictor> create(
        const SemanticToolPredictorConfig & config,
        std::string & error);

    NativeSemanticToolPredictor(const NativeSemanticToolPredictor &) = delete;
    NativeSemanticToolPredictor & operator=(
        const NativeSemanticToolPredictor &) = delete;

    SemanticToolPrediction predict(
                                   const SemanticToolPredictorRequest & predictor_request,
                                   const json & request_tools,
                                   std::string * generated_text = nullptr);

    // A failed lane (timeout, transport error) is relaunched lazily, at most
    // once per cooldown, so one slow prediction does not disable the feature
    // for the server lifetime.
    bool active() const { return ipc_.active() || ipc_.try_restart(); }

private:
    explicit NativeSemanticToolPredictor(
        const SemanticToolPredictorConfig & config) : config_(config) {}

    SemanticToolPredictorConfig config_;
    Tokenizer tokenizer_;
    mutable Qwen3ToolPredictorIpcClient ipc_;
};

}  // namespace dflash::common
