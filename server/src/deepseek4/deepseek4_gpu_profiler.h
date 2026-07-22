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

struct Ds4GpuProfileOptions {
    const char * scope = nullptr;
    const char * mode = nullptr;
    int n_tokens = 0;
    int kv_start = 0;
    int layer_begin = -1;
    int layer_end = -1;
    bool emit_zero_core = false;
};

class Ds4GpuProfiler {
public:
    Ds4GpuProfiler(bool enabled, const Ds4GpuProfileOptions & options);
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
