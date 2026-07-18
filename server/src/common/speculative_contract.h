// Speculative decoding contract and model-free routing policy.
//
// `exact` is deliberately conservative: until a backend has a differential
// proof for its batched verify + rollback implementation, it uses target AR.
// The existing batched chain/DDTree paths remain available only through the
// explicit `approximate` contract.

#pragma once

#include <string_view>

namespace dflash::common {

enum class SpeculativeContract {
    Exact,
    Approximate,
};

enum class EffectiveDecodeMode {
    Autoregressive,
    ApproximateChain,
    ApproximateDDTree,
};

enum class SpeculativeFallbackReason {
    None,
    ExactContractRequiresAR,
    SamplingUnsupported,
    ForcedAutoregressive,
    RuntimeUnavailable,
    EmptySpeculativeOutput,
};

constexpr std::string_view speculative_contract_name(SpeculativeContract value) {
    switch (value) {
    case SpeculativeContract::Exact:       return "exact";
    case SpeculativeContract::Approximate: return "approximate";
    }
    return "exact";
}

constexpr std::string_view effective_decode_mode_name(EffectiveDecodeMode value) {
    switch (value) {
    case EffectiveDecodeMode::Autoregressive:     return "autoregressive";
    case EffectiveDecodeMode::ApproximateChain:   return "approximate_chain";
    case EffectiveDecodeMode::ApproximateDDTree:  return "approximate_ddtree";
    }
    return "autoregressive";
}

constexpr std::string_view speculative_fallback_reason_name(
        SpeculativeFallbackReason value) {
    switch (value) {
    case SpeculativeFallbackReason::None:                    return "none";
    case SpeculativeFallbackReason::ExactContractRequiresAR: return "exact_contract_requires_ar";
    case SpeculativeFallbackReason::SamplingUnsupported:     return "sampling_unsupported";
    case SpeculativeFallbackReason::ForcedAutoregressive:    return "forced_autoregressive";
    case SpeculativeFallbackReason::RuntimeUnavailable:      return "runtime_unavailable";
    case SpeculativeFallbackReason::EmptySpeculativeOutput:  return "empty_speculative_output";
    }
    return "runtime_unavailable";
}

inline bool parse_speculative_contract(std::string_view text,
                                       SpeculativeContract & out) {
    if (text == "exact") {
        out = SpeculativeContract::Exact;
        return true;
    }
    if (text == "approximate") {
        out = SpeculativeContract::Approximate;
        return true;
    }
    return false;
}

struct SpeculativeRoute {
    bool use_speculative = false;
    EffectiveDecodeMode effective_mode = EffectiveDecodeMode::Autoregressive;
    SpeculativeFallbackReason fallback_reason = SpeculativeFallbackReason::None;
};

inline SpeculativeRoute resolve_speculative_route(
        SpeculativeContract contract,
        bool speculative_candidate,
        bool ddtree_requested,
        bool sampling_requested,
        bool force_ar_decode) {
    SpeculativeRoute route;
    if (!speculative_candidate) return route;
    if (force_ar_decode) {
        route.fallback_reason = SpeculativeFallbackReason::ForcedAutoregressive;
        return route;
    }
    if (contract == SpeculativeContract::Exact) {
        route.fallback_reason = SpeculativeFallbackReason::ExactContractRequiresAR;
        return route;
    }
    if (sampling_requested) {
        route.fallback_reason = SpeculativeFallbackReason::SamplingUnsupported;
        return route;
    }
    route.use_speculative = true;
    route.effective_mode = ddtree_requested
        ? EffectiveDecodeMode::ApproximateDDTree
        : EffectiveDecodeMode::ApproximateChain;
    return route;
}

}  // namespace dflash::common
