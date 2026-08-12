// Optional ROCTX ranges for correlating DeepSeek4 host phases with rocprof.
// The facility is policy-neutral: it records semantic scopes only and never
// synchronizes a device or measures elapsed time.

#pragma once

#include "common/inference_phase.h"

namespace dflash::common {

struct DeepSeek4RoctxMetadata {
    InferencePhase phase = InferencePhase::Unspecified;
    int tokens = -1;
    int layer_begin = -1;
    int layer_end = -1;
    int device = -1;
};

// Callback injection keeps range lifetime and metadata independently testable
// without requiring ROCm or loading ROCTX in unit tests.
struct DeepSeek4RoctxCallbacks {
    int (*push)(const char * message) = nullptr;
    int (*pop)() = nullptr;
};

using DeepSeek4RoctxPush = int (*)(const char * message);
using DeepSeek4RoctxPop = int (*)();

// Small loader seam used by the platform adapter and focused unit tests.
struct DeepSeek4RoctxLoader {
    void * (*open)() = nullptr;
    DeepSeek4RoctxPush (*find_push)(void * handle) = nullptr;
    DeepSeek4RoctxPop (*find_pop)(void * handle) = nullptr;
    void (*close)(void * handle) = nullptr;
    void (*diagnose)(const char * message) = nullptr;
};

bool deepseek4_roctx_env_enabled(const char * value);
DeepSeek4RoctxCallbacks deepseek4_roctx_load_callbacks(
    bool enabled, DeepSeek4RoctxLoader loader);

// Propagates caller-owned semantics through tokenwise exact-prefill calls,
// where n_tokens == 1 alone cannot distinguish prefill from decode.
class DeepSeek4RoctxPhaseScope {
public:
    explicit DeepSeek4RoctxPhaseScope(InferencePhase phase);
    ~DeepSeek4RoctxPhaseScope();

    DeepSeek4RoctxPhaseScope(const DeepSeek4RoctxPhaseScope &) = delete;
    DeepSeek4RoctxPhaseScope & operator=(const DeepSeek4RoctxPhaseScope &) = delete;

private:
    InferencePhase previous_ = InferencePhase::Unspecified;
};

InferencePhase deepseek4_roctx_prefill_phase(const char * mode);
InferencePhase deepseek4_roctx_current_phase();
InferencePhase deepseek4_roctx_layer_phase(
    bool verify, int n_tokens, InferencePhase prefill_phase);
const char * deepseek4_roctx_phase_name(InferencePhase phase);

class DeepSeek4RoctxRange {
public:
    DeepSeek4RoctxRange(const char * scope,
                       const DeepSeek4RoctxMetadata & metadata = {});
    DeepSeek4RoctxRange(const char * scope,
                       const DeepSeek4RoctxMetadata & metadata,
                       bool enabled,
                       DeepSeek4RoctxCallbacks callbacks);
    ~DeepSeek4RoctxRange();

    DeepSeek4RoctxRange(const DeepSeek4RoctxRange &) = delete;
    DeepSeek4RoctxRange & operator=(const DeepSeek4RoctxRange &) = delete;

private:
    int (*pop_)() = nullptr;
    bool pushed_ = false;
};

} // namespace dflash::common
