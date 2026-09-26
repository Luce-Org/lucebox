// Skip-park guard + startup policy resolution for PFlash compression.
//
// During PFlash compression the server normally parks the resident target (and
// decode draft), loads the scoring drafter, compresses, frees the drafter and
// reloads everything. When VRAM is ample the park/unpark round-trip is pure
// overhead: the drafter can coexist with the resident models for the whole
// compression window ("skip park").
//
// This header owns the policy, not the mechanics:
//   - SkipParkMode: the requested policy (--prefill-skip-park auto|on|off)
//   - skip_park_allowed: VMM crash guard (<32GiB with ctx>64K, VMM pool only)
//   - resolve_skip_park: the single startup decision (auto estimate vs. free
//     VRAM with margin, with explicit on/off precedence), plus whether auto
//     may also keep the drafter loaded between requests
//   - SkipParkFallback / run_skip_park_window: the runtime fail-safe that
//     retries an out-of-memory no-park window with parking
//
// The per-drafter footprint inputs come from inspect_drafter_footprint()
// (common/gguf_inspect.h), which reads dims from the drafter GGUF header and
// mirrors the buffers the Qwen3.5 scorer allocates (pflash/qwen35_drafter.cpp).

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

namespace luce::common {

// Requested policy for --prefill-skip-park. Bare `--prefill-skip-park` (no
// value) keeps the historical boolean meaning = On. Auto is the default:
// the startup estimate decides; On/Off are explicit operator overrides that
// bypass the estimate (the VMM guard still applies to On).
enum class SkipParkMode { Auto, On, Off };

inline const char * skip_park_mode_name(SkipParkMode mode) {
    switch (mode) {
    case SkipParkMode::Auto: return "auto";
    case SkipParkMode::On:   return "on";
    case SkipParkMode::Off:  return "off";
    }
    return "auto";
}

inline bool parse_skip_park_mode(const std::string & value, SkipParkMode & out) {
    if (value == "auto") { out = SkipParkMode::Auto; return true; }
    if (value == "on")   { out = SkipParkMode::On;   return true; }
    if (value == "off")  { out = SkipParkMode::Off;  return true; }
    return false;
}

// VMM crash guard (1c562eb4d): on a 24 GB CUDA card at max_ctx=131072,
// dual-resident target+drafter fragmented the VMM pool's virtual address space
// and its cuMemSetAccess failed. The failure lives in ggml's VMM pool, so the
// guard only applies when the GPU backend uses it (`vmm_pool`); HIP builds
// default to GGML_HIP_NO_VMM and allocate from the legacy pool, where the
// failure mode does not exist and the auto estimate alone decides. Applies to
// Auto and On alike — a forced-on that reliably crashes is worse than
// ignoring the request.
inline bool skip_park_allowed(bool requested, size_t total_vram_bytes, int max_ctx,
                              bool vmm_pool = true) {
    return requested &&
           (!vmm_pool || total_vram_bytes >= 32ull*1024*1024*1024 ||
            max_ctx <= 65536);
}

// Worst-case incremental VRAM while the PFlash drafter scores a window, on top
// of the resident target + decode draft it skips parking for. Populated by
// inspect_drafter_footprint() from the drafter GGUF header.
struct SkipParkDrafterInfo {
    bool    recognized = false;           // false → auto resolves off
    int64_t weights_bytes = 0;            // GGUF file size (upper bound)
    int64_t runtime_bytes_per_token = 0;  // sessions + activations + score buffers
    int64_t fixed_bytes = 0;              // SSM/conv state, gallocr, chunk transients
    int     context_length = 0;           // drafter native ctx — caps the window
};

// Headroom multiplier applied to the whole footprint. Covers fragmentation,
// gallocr slack and per-arch details the estimator intentionally ignores.
constexpr int64_t kSkipParkSafetyMarginPercent = 25;

// Room the target's own prefill/decode compute buffers need beyond what is
// resident at startup — the same 1.5 GiB the KV pool budgets reserve for
// "runtime graph buffers" (qwen35/laguna/gemma4 make_kvflash_budget). Only
// the keep-loaded decision uses it: a drafter kept resident between requests
// sits beside the target's prefill, not just beside its weights.
constexpr int64_t kTargetComputeReserveBytes = 1536ll * 1024 * 1024;

inline int64_t skip_park_with_margin(int64_t raw) {
    return raw + raw * kSkipParkSafetyMarginPercent / 100;
}

inline int64_t skip_park_raw_bytes(const SkipParkDrafterInfo & info,
                                   int64_t window_tokens) {
    return info.weights_bytes + info.fixed_bytes +
           info.runtime_bytes_per_token * window_tokens;
}

inline int64_t skip_park_required_bytes(const SkipParkDrafterInfo & info,
                                        int64_t window_tokens) {
    return skip_park_with_margin(skip_park_raw_bytes(info, window_tokens));
}

// Footprint to keep the drafter loaded between requests. Everything the
// drafter allocated for a window may still be resident when the target
// prefills the next prompt (its weights, the scoring sessions it keeps for
// prefix reuse, its backend's compute pool), so the whole window estimate is
// counted beside the target's compute reserve.
inline int64_t skip_park_keep_loaded_bytes(const SkipParkDrafterInfo & info,
                                           int64_t window_tokens) {
    return skip_park_with_margin(skip_park_raw_bytes(info, window_tokens) +
                                 kTargetComputeReserveBytes);
}

struct SkipParkDecision {
    bool        enabled = false;
    // Auto only: the estimate also leaves room to keep the drafter loaded
    // between requests (draft-residency auto → KeepLoaded). Never set by an
    // explicit `on`, which runs no estimate.
    bool        keep_drafter_loaded = false;
    std::string reason;                    // short human-readable log token
    int64_t     required_bytes = 0;        // footprint incl. margin (auto only)
    int64_t     keep_loaded_bytes = 0;     // keep-loaded footprint incl. margin
    int64_t     window_tokens = 0;         // min(max_ctx, drafter ctx)
};

// The one authoritative decision, resolved once at startup:
//   off  → never skip park
//   on   → skip park unless the VMM guard blocks it; the drafter keeps the
//          residency the draft-residency policy gives it
//   auto → skip park iff the guard passes AND the estimated footprint fits
//          measured free VRAM with margin; additionally keep the drafter
//          loaded between requests iff the footprint plus the target's
//          compute reserve fits too
inline SkipParkDecision resolve_skip_park(
        SkipParkMode mode, bool drafter_configured,
        const SkipParkDrafterInfo & info,
        int64_t free_vram_bytes, int64_t total_vram_bytes, int64_t max_ctx,
        bool vmm_pool = true) {
    SkipParkDecision d;
    if (mode == SkipParkMode::Off) {
        d.reason = "off (explicit)";
        return d;
    }
    if (!drafter_configured) {
        d.reason = "off (no local pflash drafter)";
        return d;
    }
    if (!skip_park_allowed(true, size_t(std::max<int64_t>(total_vram_bytes, 0)),
                           int(std::min<int64_t>(max_ctx, INT32_MAX)),
                           vmm_pool)) {
        d.reason = "off (guard: VMM pool, <32GiB VRAM with ctx>64K)";
        return d;
    }
    if (mode == SkipParkMode::On) {
        d.enabled = true;
        d.reason = "on (explicit)";
        return d;
    }
    if (!info.recognized || info.context_length <= 0 ||
        info.runtime_bytes_per_token <= 0 || free_vram_bytes < 0) {
        d.reason = "off (auto: drafter footprint unknown)";
        return d;
    }
    d.window_tokens = std::min<int64_t>(max_ctx, info.context_length);
    d.required_bytes = skip_park_required_bytes(info, d.window_tokens);
    d.keep_loaded_bytes = skip_park_keep_loaded_bytes(info, d.window_tokens);
    d.enabled = d.required_bytes <= free_vram_bytes;
    d.keep_drafter_loaded = d.enabled && d.keep_loaded_bytes <= free_vram_bytes;
    d.reason = !d.enabled ? "off (auto: estimate exceeds free VRAM)"
             : d.keep_drafter_loaded
                 ? "on (auto: estimate fits free VRAM, drafter kept loaded)"
                 : "on (auto: estimate fits free VRAM, drafter released)";
    return d;
}

// ── Runtime fail-safe ──────────────────────────────────────────────────────
//
// The startup estimate can still be wrong (another process takes VRAM, a
// prompt shape the estimator does not model). A skip-park window whose
// drafter work fails for lack of device memory is retried once with the
// target parked. Any other failure (non-finite scores, invalid spans) is not
// a VRAM problem and is returned as is: parking would not fix it.
//
// After a parked retry recovers, the next `backoff` windows park outright,
// then skip-park is probed again. Each further out-of-memory recovery doubles
// the backoff (4 → 8 → … → 64 windows); a successful no-park window resets
// it. While any backoff is pending the drafter is released after each window
// even when the residency policy would keep it loaded.
enum class SkipParkWindowOutcome { Ok, OutOfMemory, Failed };

class SkipParkFallback {
public:
    static constexpr int kInitialBackoffWindows = 4;
    static constexpr int kMaxBackoffWindows = 64;

    // This window must park even though skip-park was requested.
    bool parking_forced() const { return forced_windows_ > 0; }
    // A recent out-of-memory window has not yet been followed by a clean
    // no-park window: VRAM is tight, so do not keep the drafter resident.
    bool memory_tight() const { return backoff_ > 0; }
    int  forced_windows() const { return forced_windows_; }
    int  backoff() const { return backoff_; }

    void on_forced_window() {
        if (forced_windows_ > 0) --forced_windows_;
    }
    void on_oom_recovered() {
        backoff_ = backoff_ == 0
            ? kInitialBackoffWindows
            : std::min(backoff_ * 2, kMaxBackoffWindows);
        forced_windows_ = backoff_;
    }
    void on_skip_park_ok() {
        backoff_ = 0;
        forced_windows_ = 0;
    }

private:
    int forced_windows_ = 0;
    int backoff_ = 0;
};

// One compression window under the skip-park policy and fail-safe.
//   run_window(park)   runs the window, parking the resident models iff park
//   classify(results)  → SkipParkWindowOutcome
//   drop_drafter()     frees whatever the failed attempt left allocated
// Returns the results of the attempt that counts.
template <class RunWindow, class Classify, class DropDrafter>
auto run_skip_park_window(bool skip_park_requested, SkipParkFallback & fallback,
                          RunWindow && run_window, Classify && classify,
                          DropDrafter && drop_drafter, const char * tag)
        -> decltype(run_window(true)) {
    if (!skip_park_requested) return run_window(true);
    if (fallback.parking_forced()) {
        fallback.on_forced_window();
        return run_window(true);
    }
    auto results = run_window(false);
    const SkipParkWindowOutcome outcome = classify(results);
    if (outcome == SkipParkWindowOutcome::Ok) {
        fallback.on_skip_park_ok();
        return results;
    }
    if (outcome != SkipParkWindowOutcome::OutOfMemory) return results;

    std::fprintf(stderr,
        "%s skip-park window ran out of device memory — parking and "
        "retrying once\n", tag);
    drop_drafter();
    auto retry = run_window(true);
    if (classify(retry) == SkipParkWindowOutcome::Ok) {
        fallback.on_oom_recovered();
        std::fprintf(stderr,
            "%s parked retry succeeded — parking the next %d windows before "
            "probing skip-park again\n", tag, fallback.forced_windows());
    }
    return retry;
}

}  // namespace luce::common
