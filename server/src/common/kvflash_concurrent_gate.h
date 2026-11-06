// Pure decisions for N KVFlash sessions on one seq engine (Track C, C1.3-02).
// No startup, no GPU: server_main / the backend call these once their inputs
// are resolved, and the tests pin the table.

#pragma once

#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

namespace luce::common {

// ── When is the seq engine built? ──────────────────────────────────────
// Paged attention only. N > 1 as before; an EXPLICIT --max-concurrency 1
// (the C0 R6 reference arm) and a KVFlash arena also build it.
struct SeqRouteInput {
    bool paged = false;
    int  n_slots = 1;
    bool max_concurrency_set = false;
    bool kvflash_arena = false;
};
inline bool seq_engine_wanted(const SeqRouteInput & in) {
    return in.paged && (in.n_slots > 1 || in.max_concurrency_set || in.kvflash_arena);
}

// ── Test-only opening of the slot path ─────────────────────────────────
// LUCE_KVFLASH_CONCURRENT_TEST=1 bypasses the public "paged attention cannot be
// combined with KVFlash" refusal so the C1.3 GPU gates can run. The public lift
// is the last C1.3-06 commit, after the WMMA and R6 gates pass.
inline bool kvflash_concurrent_test_switch() {
    const char * v = std::getenv("LUCE_KVFLASH_CONCURRENT_TEST");
    return v && std::strcmp(v, "1") == 0;
}

// ── What may run the concurrent KVFlash arena? ─────────────────────────
struct KvFlashConcurrentRequest {
    std::string arch;
    bool kvflash_requested = false, kvflash_auto = false;
    bool pool_would_be_disabled_by_placement = false;  // the MoE all-hot case; must NOT erase the refusal
    bool wmma_env = false;                             // LUCE_PAGED_WMMA
    bool wmma_null_exclusion_ready = false;            // true once the kernel excludes null blocks (-04, gate lift)
    bool draft = false, tree = false, pflash = false, images = false;
};

// First reason the combination is refused, nullopt if it may run.
// Architecture-exact (qwen35 only) and decided on the REQUESTED combination,
// before any subclass policy (Qwen35MoeBackend::post_kvflash_init_gate) runs.
inline std::optional<std::string> kvflash_concurrent_refusal(const KvFlashConcurrentRequest & r) {
    if (r.arch != "qwen35")
        return "concurrent KVFlash supports only dense arch qwen35 (got '" + r.arch + "')";
    if (r.wmma_env && !r.wmma_null_exclusion_ready)
        return "LUCE_PAGED_WMMA=1 cannot run with KVFlash until the WMMA kernel excludes null blocks";
    if (r.draft) return "concurrent KVFlash does not support a draft model or DFlash2 chains";
    if (r.tree)  return "concurrent KVFlash does not support tree verification";
    if (r.pflash) return "concurrent KVFlash does not combine with PFlash";
    if (r.images) return "concurrent KVFlash does not serve image requests";
    return std::nullopt;
}

// ── The concurrent profile is LRU only (C1.3), recall is C2 ────────────
struct KvFlashProfileRequest {
    bool explicit_qk_policy = false, explicit_recall = false, explicit_uq = false;
    bool explicit_preamble_pin = false, explicit_gpu_score = false, explicit_reselect = false;
};
struct KvFlashProfile {
    bool ok = true;
    std::string policy = "lru-only";
    bool suppress_serial_defaults = true;   // kvflash_recall_env defaults must not be applied
    std::string error, log_line;
};
inline KvFlashProfile kvflash_resolve_concurrent_profile(const KvFlashProfileRequest & r) {
    KvFlashProfile p;
    const char * what = r.explicit_qk_policy ? "QK policy" : r.explicit_recall ? "recall" :
                        r.explicit_uq ? "uq" : r.explicit_preamble_pin ? "preamble pins" :
                        r.explicit_gpu_score ? "GPU scoring" : r.explicit_reselect ? "reselect" : nullptr;
    if (what) {
        p.ok = false;
        p.error = std::string("concurrent KVFlash is LRU only; ") + what + " is C2 and not supported yet";
        return p;
    }
    p.log_line = "[kvflash] concurrent profile: lru-only (serial recall defaults suppressed)";
    return p;
}

}  // namespace luce::common
