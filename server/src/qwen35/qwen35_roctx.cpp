#include "qwen35_roctx.h"

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
bool equals_ignore_case(const char * a, const char * b) {
    if (!a || !b) return false;
    while (*a && *b) {
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return false;
        ++a; ++b;
    }
    return *a == '\0' && *b == '\0';
}

#if defined(DFLASH27B_BACKEND_HIP)
using RoctxPush = int (*)(const char *);
using RoctxPop = int (*)();

#  if defined(_WIN32)
void * roctx_open() {
    return reinterpret_cast<void *>(LoadLibraryA("roctx64.dll"));
}
RoctxPush roctx_find_push(void * handle) {
    return reinterpret_cast<RoctxPush>(
        GetProcAddress(reinterpret_cast<HMODULE>(handle), "roctxRangePushA"));
}
RoctxPop roctx_find_pop(void * handle) {
    return reinterpret_cast<RoctxPop>(
        GetProcAddress(reinterpret_cast<HMODULE>(handle), "roctxRangePop"));
}
void roctx_close(void * handle) {
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
}
#  else
void * roctx_open() {
    void * process = dlopen(nullptr, RTLD_LAZY | RTLD_LOCAL);
    if (process && dlsym(process, "roctxRangePushA") &&
        dlsym(process, "roctxRangePop")) {
        return process;
    }
    if (process) dlclose(process);
    return dlopen("libroctx64.so", RTLD_LAZY | RTLD_LOCAL);
}
RoctxPush roctx_find_push(void * handle) {
    return reinterpret_cast<RoctxPush>(dlsym(handle, "roctxRangePushA"));
}
RoctxPop roctx_find_pop(void * handle) {
    return reinterpret_cast<RoctxPop>(dlsym(handle, "roctxRangePop"));
}
void roctx_close(void * handle) {
    dlclose(handle);
}
#  endif
#endif

Qwen35RoctxCallbacks configured_callbacks() {
#if defined(DFLASH27B_BACKEND_HIP)
    static const Qwen35RoctxCallbacks callbacks = [] {
        if (!qwen35_roctx_env_enabled(std::getenv("DFLASH_QWEN35_ROCTX"))) return Qwen35RoctxCallbacks{};
        void * handle = roctx_open();
        auto push = handle ? roctx_find_push(handle) : nullptr;
        auto pop = handle ? roctx_find_pop(handle) : nullptr;
        if (!push || !pop) {
            if (handle) roctx_close(handle);
            std::fprintf(stderr, "[qwen35-roctx] ROCTX symbols unavailable; markers disabled\n");
            return Qwen35RoctxCallbacks{};
        }
        return Qwen35RoctxCallbacks{push, pop};
    }();
    return callbacks;
#else
    return {};
#endif
}

void append(char * message, size_t capacity, size_t & used, const char * name, int value) {
    if (value < 0 || used >= capacity) return;
    const int n = std::snprintf(message + used, capacity - used, " %s=%d", name, value);
    if (n > 0) used += std::min((size_t)n, capacity - used - 1);
}
} // namespace

bool qwen35_roctx_env_enabled(const char * value) {
    return value && (std::strcmp(value, "1") == 0 || equals_ignore_case(value, "true") ||
                     equals_ignore_case(value, "yes") || equals_ignore_case(value, "on"));
}

Qwen35RoctxRange::Qwen35RoctxRange(const char * scope, const Qwen35RoctxMetadata & metadata)
    : Qwen35RoctxRange(scope, metadata, true, configured_callbacks()) {}

Qwen35RoctxRange::Qwen35RoctxRange(const char * scope, const Qwen35RoctxMetadata & metadata,
                                   bool enabled, Qwen35RoctxCallbacks callbacks) {
    if (!enabled || !scope || !scope[0] || !callbacks.push || !callbacks.pop) return;
    char message[256];
    const int initial = std::snprintf(message, sizeof(message), "%s", scope);
    size_t used = initial > 0 ? std::min((size_t)initial, sizeof(message) - 1) : 0;
    append(message, sizeof(message), used, "live", metadata.live);
    append(message, sizeof(message), used, "bucket", metadata.bucket);
    append(message, sizeof(message), used, "prefill_tokens", metadata.prefill_tokens);
    append(message, sizeof(message), used, "prefill_segments", metadata.prefill_segments);
    append(message, sizeof(message), used, "total_rows", metadata.total_rows);
    append(message, sizeof(message), used, "max_kv_len", metadata.max_kv_len);
    if (callbacks.push(message) >= 0) { pop_ = callbacks.pop; pushed_ = true; }
}

Qwen35RoctxRange::~Qwen35RoctxRange() { if (pushed_ && pop_) pop_(); }
} // namespace dflash::common
