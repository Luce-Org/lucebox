#pragma once

#include <cstdint>

namespace dflash::common {

enum class Ds4GpuPhase : uint8_t {
    HcPre,
    Attention,
    MoeFfn,
    HcPost,
    OutputProjection,
    FusedDecodeGraph,
    ApproxFusedVerifyGraph,
    VerificationStep,
    Count,
};

class Ds4GpuProfiler {
public:
    Ds4GpuProfiler(bool enabled, const char * scope, const char * mode,
                   int n_tokens, int kv_start);
    ~Ds4GpuProfiler();

    Ds4GpuProfiler(const Ds4GpuProfiler &) = delete;
    Ds4GpuProfiler & operator=(const Ds4GpuProfiler &) = delete;

    bool enabled() const;
    void begin(Ds4GpuPhase phase);
    void end(Ds4GpuPhase phase);
    void emit();

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

} // namespace dflash::common
