// Pure decisions for N KVFlash sessions on one seq engine (Track C, C1.3-02).
// No startup, no GPU: server_main / the backend call these once their inputs
// are resolved, and the tests pin the table.

#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "kvflash_query_capture.h"

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

// ── The concurrent profile: per-slot recall by default on qwen35 (C1.4), LRU on request ──
struct KvFlashProfileRequest {
    std::string arch;
    bool explicit_lru_policy = false;   // LUCE_KVFLASH_POLICY=lru or --kvflash-policy lru: the opt-out
    bool explicit_qk_policy = false, explicit_recall = false, explicit_uq = false;
    bool explicit_preamble_pin = false, explicit_tau = false;   // the preamble pins stay refused until the slot engine pins
    int  reselect_mode = 0;          // LUCE_KVFLASH_PREFILL_RESELECT as parsed
    bool exact_scorer = false;       // LUCE_KVFLASH_RESELECT_SCORER=exact
    bool pflash2_scorer = false;     // LUCE_KVFLASH_RESELECT_SCORER=pflash2 (mode 5)
    int  select_queries = 0;         // queries kept per prefill, as the backend derives them (kQueryCaptureTokens in mode 4, else 0)
    bool multi_model = false;        // env defaults are process-wide: one model's defaults would leak into the next
};
struct KvFlashProfile {
    bool ok = true;
    std::string policy = "lru-only";
    bool suppress_serial_defaults = true;   // kvflash_recall_env defaults must not be applied
    std::string error, log_line;
    std::vector<std::pair<std::string, std::string>> env_defaults;   // for apply_env_defaults: a set variable keeps its value
};
inline KvFlashProfile kvflash_resolve_concurrent_profile(const KvFlashProfileRequest & r) {
    KvFlashProfile p;
    // C1.4: the recall profile (QK policy, mode-4 reselect, recall blocks) runs per slot
    // when every piece of it is asked for explicitly; uq, preamble pins and tau stay C2.
    const char * what = r.explicit_uq ? "uq" : r.explicit_preamble_pin ? "preamble pins" : r.explicit_tau ? "tau" : nullptr;
    if (what) {
        p.ok = false;
        p.error = std::string("concurrent KVFlash is LRU or per-slot recall; ") + what + " is C2 and not supported yet";
        return p;
    }
    const bool any = r.explicit_qk_policy || r.explicit_recall || r.reselect_mode != 0 || r.exact_scorer || r.pflash2_scorer;
    if (any) {
        const bool mode4 = r.reselect_mode == 4 && r.exact_scorer && r.select_queries > 0;
        const bool mode5 = r.reselect_mode == 5 && r.pflash2_scorer;   // the drafter is a scorer here: no queries, no first pass
        if (!(r.explicit_qk_policy && r.explicit_recall && (mode4 || mode5))) {
            p.ok = false;
            p.error = "concurrent KVFlash recall needs LUCE_KVFLASH_POLICY=qk, LUCE_KVFLASH_PREFILL_RESELECT=4, "
                      "LUCE_KVFLASH_RESELECT_SCORER=exact and LUCE_KVFLASH_RECALL_BLOCKS "
                      "together, or PREFILL_RESELECT=5 with RESELECT_SCORER=pflash2 and RECALL_BLOCKS "
                      "(C1.4: no defaults are installed)";
            return p;
        }
        p.policy = "recall";
        p.log_line = "[kvflash] concurrent profile: per-slot recall (C1.4, explicit env; serial defaults suppressed)";
        return p;
    }
    if (r.arch == "qwen35" && !r.explicit_lru_policy && !r.multi_model) {
        // The profile every per-slot recall run used (C1.4). The serial defaults minus uq: UQ_QUERIES=0.
        p.policy = "recall";
        p.env_defaults = {
            {"LUCE_KVFLASH_POLICY", "qk"}, {"LUCE_KVFLASH_PREFILL_RESELECT", "4"},
            {"LUCE_KVFLASH_RESELECT_SCORER", "exact"}, {"LUCE_KVFLASH_RECALL_BLOCKS", "6"},
            {"LUCE_KVFLASH_SCORE_POOL_RECENT", "1"}, {"LUCE_KVFLASH_SELECT_UNIT", "2"},
            {"LUCE_KVFLASH_UQ_QUERIES", "0"},
        };
        p.log_line = "[kvflash] concurrent profile: per-slot recall (C1.4 defaults installed; "
                     "LUCE_KVFLASH_POLICY=lru keeps LRU only)";
        return p;
    }
    p.log_line = (r.arch == "qwen35" && r.multi_model && !r.explicit_lru_policy)
        ? "[kvflash] concurrent profile: lru-only (the C1.4 recall default is single-model only; "
          "set the full explicit recall env for a multi-model server)"
        : "[kvflash] concurrent profile: lru-only (serial recall defaults suppressed)";
    return p;
}

// ── The request as the environment states it (lookup(name) = the value or nullptr) ──
using KvFlashLookup = std::function<const char *(const char *)>;
inline KvFlashProfileRequest kvflash_concurrent_request(const std::string & arch, const std::string & cli_policy,
                                                        bool cli_tau, bool multi_model, const KvFlashLookup & lookup) {
    auto is = [&](const char * name, const char * want) { const char * v = lookup(name); return v && std::strcmp(v, want) == 0; };
    auto on = [&](const char * name) { const char * v = lookup(name); return v && *v && std::strcmp(v, "0") != 0; };
    KvFlashProfileRequest r;
    r.arch = arch;
    r.explicit_qk_policy = cli_policy.empty() ? is("LUCE_KVFLASH_POLICY", "qk") : cli_policy == "qk";
    r.explicit_lru_policy = cli_policy.empty() ? is("LUCE_KVFLASH_POLICY", "lru") : cli_policy == "lru";
    r.explicit_recall = on("LUCE_KVFLASH_RECALL_BLOCKS");
    r.explicit_uq = on("LUCE_KVFLASH_UQ_QUERIES");
    r.explicit_preamble_pin = on("LUCE_KVFLASH_PREAMBLE_ROWS");
    const char * rs = lookup("LUCE_KVFLASH_PREFILL_RESELECT");
    r.reselect_mode = rs ? std::atoi(rs) : 0;
    r.exact_scorer = is("LUCE_KVFLASH_RESELECT_SCORER", "exact");
    r.pflash2_scorer = is("LUCE_KVFLASH_RESELECT_SCORER", "pflash2");
    r.select_queries = r.reselect_mode == 4 ? kQueryCaptureTokens : 0;
    r.explicit_tau = on("LUCE_KVFLASH_TAU") || cli_tau;
    r.multi_model = multi_model;
    return r;
}

// After the defaults went into the environment (a set variable kept its value): the merged request must resolve
// through the explicit recall branch (recall, nothing left to install). Empty when coherent, else the refusal.
inline std::string kvflash_merged_profile_error(const std::string & arch, const std::string & cli_policy,
                                                bool cli_tau, bool multi_model, const KvFlashLookup & lookup) {
    const KvFlashProfile p = kvflash_resolve_concurrent_profile(kvflash_concurrent_request(arch, cli_policy, cli_tau, multi_model, lookup));
    if (p.ok && p.policy == "recall" && p.env_defaults.empty()) return std::string();
    return "a LUCE_KVFLASH_* value or --kvflash-policy keeps the C1.4 recall defaults from forming a recall profile "
           "(they need POLICY=qk, PREFILL_RESELECT=4, RESELECT_SCORER=exact, RECALL_BLOCKS>0); "
           "unset it, set the full recall env, or set LUCE_KVFLASH_POLICY=lru";
}

}  // namespace luce::common
