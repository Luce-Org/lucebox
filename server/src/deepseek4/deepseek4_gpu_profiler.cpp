#include "deepseek4_gpu_profiler.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <new>

#if defined(GGML_USE_HIP)
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#endif

namespace dflash::common {

namespace {

constexpr size_t kPhaseCount = static_cast<size_t>(Ds4GpuPhase::Count);

const char * phase_name(Ds4GpuPhase phase) {
    switch (phase) {
        case Ds4GpuPhase::HcPre:                  return "hc_pre";
        case Ds4GpuPhase::Attention:              return "attention";
        case Ds4GpuPhase::MoeFfn:                 return "moe_ffn";
        case Ds4GpuPhase::HcPost:                 return "hc_post";
        case Ds4GpuPhase::OutputProjection:       return "output_projection";
        case Ds4GpuPhase::FusedDecodeGraph:       return "fused_decode_graph";
        case Ds4GpuPhase::ApproxFusedVerifyGraph: return "approx_fused_verify_graph";
        case Ds4GpuPhase::VerificationStep:       return "verification_step";
        case Ds4GpuPhase::Count:                  break;
    }
    return "unknown";
}

#if defined(GGML_USE_HIP)
class RoctxApi {
public:
    using PushFn = int (*)(const char *);
    using PopFn = int (*)();

    RoctxApi() {
        handle_ = dlopen("libroctx64.so", RTLD_LAZY | RTLD_LOCAL);
        if (!handle_) return;
        push_ = reinterpret_cast<PushFn>(dlsym(handle_, "roctxRangePushA"));
        pop_ = reinterpret_cast<PopFn>(dlsym(handle_, "roctxRangePop"));
        if (!push_ || !pop_) {
            dlclose(handle_);
            handle_ = nullptr;
            push_ = nullptr;
            pop_ = nullptr;
        }
    }

    ~RoctxApi() {
        if (handle_) dlclose(handle_);
    }

    void push(const char * name) const {
        if (push_) push_(name);
    }

    void pop() const {
        if (pop_) pop_();
    }

private:
    void * handle_ = nullptr;
    PushFn push_ = nullptr;
    PopFn pop_ = nullptr;
};

RoctxApi & roctx_api() {
    static RoctxApi api;
    return api;
}

void destroy_event(hipEvent_t event) {
    if (!event) return;
    const hipError_t status = hipEventDestroy(event);
    (void) status;
}
#endif

} // namespace

struct Ds4GpuProfiler::Impl {
    const char * scope = nullptr;
    const char * mode = nullptr;
    int n_tokens = 0;
    int kv_start = 0;
    int layer_begin = -1;
    int layer_end = -1;
    Ds4GpuPhase active = Ds4GpuPhase::Count;
    std::array<double, kPhaseCount> elapsed_ms{};
    std::array<uint64_t, kPhaseCount> calls{};
    bool emitted = false;
#if defined(GGML_USE_HIP)
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
#endif
};

Ds4GpuProfiler::Ds4GpuProfiler(bool enabled, const char * scope, const char * mode,
                               int n_tokens, int kv_start, int layer_begin, int layer_end) {
#if defined(GGML_USE_HIP)
    if (!enabled) return;
    Impl * impl = new (std::nothrow) Impl;
    if (!impl) return;
    impl->scope = scope;
    impl->mode = mode;
    impl->n_tokens = n_tokens;
    impl->kv_start = kv_start;
    impl->layer_begin = layer_begin;
    impl->layer_end = layer_end;
    if (hipEventCreate(&impl->start) != hipSuccess ||
        hipEventCreate(&impl->stop) != hipSuccess) {
        destroy_event(impl->start);
        destroy_event(impl->stop);
        delete impl;
        return;
    }
    impl_ = impl;
#else
    (void) enabled;
    (void) scope;
    (void) mode;
    (void) n_tokens;
    (void) kv_start;
    (void) layer_begin;
    (void) layer_end;
#endif
}

Ds4GpuProfiler::~Ds4GpuProfiler() {
#if defined(GGML_USE_HIP)
    if (!impl_) return;
    if (impl_->active != Ds4GpuPhase::Count) {
        roctx_api().pop();
    }
    destroy_event(impl_->start);
    destroy_event(impl_->stop);
    delete impl_;
#endif
}

bool Ds4GpuProfiler::enabled() const {
    return impl_ != nullptr;
}

void Ds4GpuProfiler::begin(Ds4GpuPhase phase) {
#if defined(GGML_USE_HIP)
    if (!impl_ || impl_->active != Ds4GpuPhase::Count || phase == Ds4GpuPhase::Count) return;
    if (hipEventRecord(impl_->start, nullptr) != hipSuccess ||
        hipEventSynchronize(impl_->start) != hipSuccess) {
        return;
    }
    impl_->active = phase;
    roctx_api().push(phase_name(phase));
#else
    (void) phase;
#endif
}

void Ds4GpuProfiler::end(Ds4GpuPhase phase) {
#if defined(GGML_USE_HIP)
    if (!impl_ || phase == Ds4GpuPhase::Count || impl_->active != phase) return;
    roctx_api().pop();
    impl_->active = Ds4GpuPhase::Count;
    if (hipEventRecord(impl_->stop, nullptr) != hipSuccess ||
        hipEventSynchronize(impl_->stop) != hipSuccess) {
        return;
    }
    float elapsed = 0.0f;
    if (hipEventElapsedTime(&elapsed, impl_->start, impl_->stop) != hipSuccess) return;
    const size_t index = static_cast<size_t>(phase);
    impl_->elapsed_ms[index] += elapsed;
    impl_->calls[index]++;
#else
    (void) phase;
#endif
}

void Ds4GpuProfiler::emit() {
    if (!impl_ || impl_->emitted) return;
    impl_->emitted = true;
    uint64_t total_calls = 0;
    for (uint64_t calls : impl_->calls) total_calls += calls;
    if (total_calls == 0) return;
    const bool emit_zero_core = std::strcmp(impl_->scope, "forward") == 0 &&
                                std::strcmp(impl_->mode, "exact_verify") == 0;
    char layers[24] = "";
    if (impl_->layer_begin >= 0) {
        std::snprintf(layers, sizeof(layers), " layers=%d-%d",
                      impl_->layer_begin, impl_->layer_end);
    }
    for (size_t i = 0; i < kPhaseCount; ++i) {
        const bool core_phase = i <= static_cast<size_t>(Ds4GpuPhase::OutputProjection);
        if (impl_->calls[i] == 0 && (!core_phase || !emit_zero_core)) continue;
        const auto phase = static_cast<Ds4GpuPhase>(i);
        std::fprintf(stderr,
            "[ds4-gpu-profile] clock=hip_event scope=%s mode=%s phase=%s "
            "tokens=%d kv_start=%d%s gpu_ms=%.3f calls=%llu\n",
            impl_->scope, impl_->mode, phase_name(phase), impl_->n_tokens,
            impl_->kv_start, layers, impl_->elapsed_ms[i],
            static_cast<unsigned long long>(impl_->calls[i]));
    }
}

} // namespace dflash::common
