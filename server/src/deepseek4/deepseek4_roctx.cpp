#include "deepseek4_roctx.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(DFLASH27B_BACKEND_HIP)
#  if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#  else
#    include <dlfcn.h>
#  endif
#endif

namespace dflash::common {
namespace {

thread_local InferencePhase current_phase = InferencePhase::Unspecified;

bool equals_ignore_case(const char * lhs, const char * rhs) {
    if (!lhs || !rhs) return false;
    while (*lhs && *rhs) {
        if (std::tolower(static_cast<unsigned char>(*lhs)) !=
            std::tolower(static_cast<unsigned char>(*rhs))) {
            return false;
        }
        ++lhs;
        ++rhs;
    }
    return *lhs == '\0' && *rhs == '\0';
}

#if defined(DFLASH27B_BACKEND_HIP)
#  if defined(_WIN32)
void * platform_open() {
    return reinterpret_cast<void *>(LoadLibraryA("roctx64.dll"));
}
DeepSeek4RoctxPush platform_find_push(void * handle) {
    return reinterpret_cast<DeepSeek4RoctxPush>(
        GetProcAddress(reinterpret_cast<HMODULE>(handle), "roctxRangePushA"));
}
DeepSeek4RoctxPop platform_find_pop(void * handle) {
    return reinterpret_cast<DeepSeek4RoctxPop>(
        GetProcAddress(reinterpret_cast<HMODULE>(handle), "roctxRangePop"));
}
void platform_close(void * handle) {
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
}
#  else
void * platform_open() {
    void * process = dlopen(nullptr, RTLD_LAZY | RTLD_LOCAL);
    if (process) {
        if (dlsym(process, "roctxRangePushA") &&
            dlsym(process, "roctxRangePop")) {
            return process;
        }
        dlclose(process);
    }
    return dlopen("libroctx64.so", RTLD_LAZY | RTLD_LOCAL);
}
DeepSeek4RoctxPush platform_find_push(void * handle) {
    return reinterpret_cast<DeepSeek4RoctxPush>(dlsym(handle, "roctxRangePushA"));
}
DeepSeek4RoctxPop platform_find_pop(void * handle) {
    return reinterpret_cast<DeepSeek4RoctxPop>(dlsym(handle, "roctxRangePop"));
}
void platform_close(void * handle) {
    dlclose(handle);
}
#  endif
void platform_diagnose(const char * message) {
    std::fprintf(stderr, "[deepseek4-roctx] %s\n", message);
}
#endif

DeepSeek4RoctxCallbacks runtime_callbacks() {
#if defined(DFLASH27B_BACKEND_HIP)
    static const DeepSeek4RoctxCallbacks callbacks =
        deepseek4_roctx_load_callbacks(
            true, {platform_open, platform_find_push, platform_find_pop,
                   platform_close, platform_diagnose});
#else
    static const DeepSeek4RoctxCallbacks callbacks{};
#endif
    return callbacks;
}

DeepSeek4RoctxCallbacks configured_callbacks() {
    // Serving configuration is process-scoped. Cache the disabled path too so
    // hot layer-range calls do not repeatedly query the environment.
    static const DeepSeek4RoctxCallbacks callbacks = [] {
        if (!deepseek4_roctx_env_enabled(std::getenv("DFLASH_DS4_ROCTX"))) {
            return DeepSeek4RoctxCallbacks{};
        }
        return runtime_callbacks();
    }();
    return callbacks;
}

void append_field(char * message, size_t capacity, size_t & used,
                  const char * format, const char * value) {
    if (!value || !value[0] || used >= capacity) return;
    const int written = std::snprintf(message + used, capacity - used, format, value);
    if (written > 0) used += std::min(static_cast<size_t>(written), capacity - used - 1);
}

void append_field(char * message, size_t capacity, size_t & used,
                  const char * format, int value) {
    if (value < 0 || used >= capacity) return;
    const int written = std::snprintf(message + used, capacity - used, format, value);
    if (written > 0) used += std::min(static_cast<size_t>(written), capacity - used - 1);
}

} // namespace

bool deepseek4_roctx_env_enabled(const char * value) {
    return value && (std::strcmp(value, "1") == 0 ||
                     equals_ignore_case(value, "true") ||
                     equals_ignore_case(value, "yes") ||
                     equals_ignore_case(value, "on"));
}

DeepSeek4RoctxCallbacks deepseek4_roctx_load_callbacks(
        bool enabled, DeepSeek4RoctxLoader loader) {
    if (!enabled) return {};
    if (!loader.open || !loader.find_push || !loader.find_pop || !loader.close) {
        return {};
    }

    void * handle = loader.open();
    if (!handle) {
        if (loader.diagnose) {
            loader.diagnose(
                "DFLASH_DS4_ROCTX is enabled, but the ROCTX library could not "
                "be loaded; markers are disabled");
        }
        return {};
    }

    const DeepSeek4RoctxPush push = loader.find_push(handle);
    const DeepSeek4RoctxPop pop = loader.find_pop(handle);
    if (!push || !pop) {
        loader.close(handle);
        if (loader.diagnose) {
            loader.diagnose(
                "DFLASH_DS4_ROCTX is enabled, but required ROCTX range symbols "
                "are missing; markers are disabled");
        }
        return {};
    }

    // Successful libraries remain resident for process lifetime because range
    // callbacks can run on serving threads and during static teardown.
    return {push, pop};
}

DeepSeek4RoctxPhaseScope::DeepSeek4RoctxPhaseScope(InferencePhase phase)
    : previous_(current_phase) {
    current_phase = phase;
}

DeepSeek4RoctxPhaseScope::~DeepSeek4RoctxPhaseScope() {
    current_phase = previous_;
}

InferencePhase deepseek4_roctx_prefill_phase(const char * mode) {
    if (mode && std::strcmp(mode, "exact") == 0) return InferencePhase::Exact;
    if (mode && std::strcmp(mode, "dense") == 0) return InferencePhase::Dense;
    if (mode && std::strcmp(mode, "sparse") == 0) return InferencePhase::Sparse;
    return InferencePhase::Unspecified;
}

InferencePhase deepseek4_roctx_current_phase() {
    return current_phase;
}

InferencePhase deepseek4_roctx_layer_phase(
        bool verify, int n_tokens, InferencePhase prefill_phase) {
    if (current_phase != InferencePhase::Unspecified) return current_phase;
    if (verify) return InferencePhase::Verify;
    return n_tokens == 1 ? InferencePhase::Unspecified : prefill_phase;
}

const char * deepseek4_roctx_phase_name(InferencePhase phase) {
    switch (phase) {
        case InferencePhase::Exact: return "exact";
        case InferencePhase::Dense: return "dense";
        case InferencePhase::Sparse: return "sparse";
        case InferencePhase::Decode: return "decode";
        case InferencePhase::Verify: return "verify";
        case InferencePhase::ReferenceExact: return "reference_exact";
        case InferencePhase::Sequential: return "sequential";
        case InferencePhase::Batched: return "batched";
        case InferencePhase::Unspecified: return "unspecified";
    }
    return "unspecified";
}

DeepSeek4RoctxRange::DeepSeek4RoctxRange(
        const char * scope, const DeepSeek4RoctxMetadata & metadata)
    : DeepSeek4RoctxRange(scope, metadata, true, configured_callbacks()) {}

DeepSeek4RoctxRange::DeepSeek4RoctxRange(
        const char * scope, const DeepSeek4RoctxMetadata & metadata,
        bool enabled, DeepSeek4RoctxCallbacks callbacks) {
    if (!enabled || !scope || !scope[0] || !callbacks.push || !callbacks.pop) return;

    // A stack buffer avoids allocation and per-layer string ownership. ROCTX
    // consumes the message during push, so the buffer need not outlive this call.
    char message[256];
    const int initial = std::snprintf(message, sizeof(message), "%s", scope);
    size_t used = initial > 0
        ? std::min(static_cast<size_t>(initial), sizeof(message) - 1)
        : 0;
    append_field(message, sizeof(message), used, " mode=%s",
                 deepseek4_roctx_phase_name(metadata.phase));
    append_field(message, sizeof(message), used, " tokens=%d", metadata.tokens);
    append_field(message, sizeof(message), used, " layer_begin=%d", metadata.layer_begin);
    append_field(message, sizeof(message), used, " layer_end=%d", metadata.layer_end);
    append_field(message, sizeof(message), used, " device=%d", metadata.device);

    if (callbacks.push(message) >= 0) {
        pop_ = callbacks.pop;
        pushed_ = true;
    }
}

DeepSeek4RoctxRange::~DeepSeek4RoctxRange() {
    if (pushed_ && pop_) pop_();
}

} // namespace dflash::common
