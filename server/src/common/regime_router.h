// Adaptive compression-regime router v2.
// No IO, no globals, no GPU, no ggml/llama deps — header-only, stdlib-only.
//
// Splits on prompt TYPE (agentic vs retrieval).
// V1 R-router (cascade expansion ratio) was refuted as a keep predictor (ρ=-0.27).
// Sparse-prompt guard and recency floor were validated zero-sum; removed.
//
// Build (standalone):
//   g++-11 -std=gnu++17 -O2 -I server/src/common
//     -o /tmp/test_regime_router server/test/test_regime_router.cpp
// CMake:  cmake --build build --target test_regime_router -j
//         ctest -R regime_router --output-on-failure
#pragma once

#include <cmath>
#include <limits>

namespace dflash::common {

// ─── V2 Router ───────────────────────────────────────────────────────────────

struct RequestFeatures {
    bool is_agentic;    // tool schemas / tool_use|tool_result blocks present
    int  prompt_tokens; // total S
};

struct RouterPolicyV2 {
    bool   enabled             = false;  // DEFAULT DISABLED → exact no-op
    int    threshold_tokens    = 32000;  // below → passthrough
    double agentic_keep_target = 0.25;   // conservative floor, agentic path
    double full_keep_target    = 1.0;    // retrieval/QA & safe fallbacks
};

struct RouterDecisionV2 {
    double      keep_target;
    bool        cascade;
    const char* reason;
};

// decide_v2 — pure, no IO, no globals.
//
// SAFE path: keep_target=full_keep_target, cascade=true.
// Returns SAFE when:
//   - p.enabled == false                    (deploy no-op, correct-by-construction)
//   - f.prompt_tokens <= 0                  (degenerate)
//   - f.prompt_tokens < p.threshold_tokens  (below threshold)
// Throttling path (only when all guards pass):
//   - is_agentic → {agentic_keep_target, cascade=false, "agentic_throttle"}
//   - else       → {full_keep_target,    cascade=true,  "retrieval_full"}
inline RouterDecisionV2 decide_v2(const RequestFeatures& f,
                                   const RouterPolicyV2&   p) {
    const RouterDecisionV2 SAFE_disabled        = { p.full_keep_target, true, "disabled_noop"   };
    const RouterDecisionV2 SAFE_degenerate      = { p.full_keep_target, true, "degenerate"      };
    const RouterDecisionV2 SAFE_below_threshold = { p.full_keep_target, true, "below_threshold" };

    if (!p.enabled)
        return SAFE_disabled;

    if (f.prompt_tokens <= 0)
        return SAFE_degenerate;

    if (f.prompt_tokens < p.threshold_tokens)
        return SAFE_below_threshold;

    if (f.is_agentic)
        return { p.agentic_keep_target, false, "agentic_throttle" };

    return { p.full_keep_target, true, "retrieval_full" };
}

// ─── PIECE 1: floor clamp ────────────────────────────────────────────────────
//
// When the router routed a request as agentic, the bandit must not compress
// harder than the router's agentic_keep_target floor.  Non-agentic sessions
// are passed through unchanged (bandit drives retrieval sessions freely).
//
// Pure, stdlib-only, no IO.
inline double clamp_keep_to_floor(double bandit_keep,
                                   double router_floor,
                                   bool   agentic) {
    if (!agentic) return bandit_keep;
    return bandit_keep >= router_floor ? bandit_keep : router_floor;
}

// ─── PIECE 2: compression failure guard ──────────────────────────────────────
//
// Returns true when a compressed agentic turn produced an empty or degenerate
// response.  Used to skip the bandit update (failure noise) and schedule a
// full-keep recovery for the next turn.
//
// Fires ONLY on the agentic+compressed path — non-compressed failures are not
// our fault and do not need recovery.
//
// Pure, stdlib-only, no IO.
inline bool compression_failed(int  response_tokens,
                                bool degenerate_close,
                                bool agentic_compressed,
                                int  min_tokens = 8) {
    if (!agentic_compressed) return false;
    return response_tokens < min_tokens || degenerate_close;
}

// ─── TYPE GATE ───────────────────────────────────────────────────────────────
//
// Coarse request-type classifier.  Pure function — no IO, no globals, no JSON.
//
// Agentic signals (any one is sufficient):
//   1. has_tools          — tools array was non-null and non-empty
//   2. has_tool_use_blocks — any message content contained a tool_use or
//                           tool_result block  (Anthropic style)
//   3. has_tool_calls     — any assistant message had a non-empty tool_calls
//                           array  (OpenAI style)
//
// The caller is responsible for extracting these bools from the wire format.
// Default: Retrieval (safe — never compresses more than intended).

enum class RequestType { Agentic, Retrieval };

// detect_request_type — pure, stdlib-only, no IO.
inline RequestType detect_request_type(bool has_tools,
                                        bool has_tool_use_blocks,
                                        bool has_tool_calls) {
    if (has_tools || has_tool_use_blocks || has_tool_calls)
        return RequestType::Agentic;
    return RequestType::Retrieval;
}

}  // namespace dflash::common
