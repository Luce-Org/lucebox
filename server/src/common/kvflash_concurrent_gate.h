// Pure decisions for N KVFlash sessions on one seq engine (Track C, C1.3-02).
// No startup, no GPU: server_main / the backend call these once their inputs
// are resolved, and the tests pin the table.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

// ── When is the seq engine built? ──────────────────────────────────────
// Paged attention only. N > 1 as before; an EXPLICIT --max-concurrency 1
// (the C0 R6 reference arm, behind the test switch until the public lift) and a KVFlash arena also build it.
struct SeqRouteInput {
    bool paged = false;
    int  n_slots = 1;
    bool max_concurrency_set = false;
    bool kvflash_arena = false;
    bool test_switch = false;   // LUCE_KVFLASH_CONCURRENT_TEST: the explicit-1 routing is not public yet
};
inline bool seq_engine_wanted(const SeqRouteInput & in) {
    return in.paged && (in.n_slots > 1 || (in.max_concurrency_set && in.test_switch) || in.kvflash_arena);
}

// ── Test-only opening of the slot path ─────────────────────────────────
// LUCE_KVFLASH_CONCURRENT_TEST=1 bypasses the public "paged attention cannot be
// combined with KVFlash" refusal so the C1.3 GPU gates can run. The public lift
// is the last C1.3-06 commit, after the WMMA and R6 gates pass.
inline bool kvflash_concurrent_test_switch() {
    const char * v = std::getenv("LUCE_KVFLASH_CONCURRENT_TEST");
    return v && std::strcmp(v, "1") == 0;
}

// ── Opt-in for DFlash2 chain verify beside the KVFlash arena (Stage C) ──
// Exactly LUCE_KVFLASH_CHAIN_SPEC=1, on top of LUCE_KVFLASH_CONCURRENT_TEST=1.
inline bool kvflash_chain_spec_switch() {
    const char * v = std::getenv("LUCE_KVFLASH_CHAIN_SPEC");
    return v && std::strcmp(v, "1") == 0;
}

// ── What may run the concurrent KVFlash arena? ─────────────────────────
struct KvFlashConcurrentRequest {
    std::string arch;
    bool draft = false, tree = false;
    // Stage C: a draft model is admitted only as a DFlash2 chain, behind the switch.
    bool chain_spec_switch = false;          // LUCE_KVFLASH_CHAIN_SPEC=1
    bool draft_local_same_device = false;    // in-process draft on the target's device, no split/TP/remote
    int  draft_block = 0;                    // the drafter's block size; a chain needs 2..16
    int  fa_window = 0;                      // the chain path needs full attention
};

// LUCE_KVFLASH_HOST_MIB: the host budget for paged-out state, 16384 MiB when unset or empty.
// nullopt when the value is not a positive integer.
inline std::optional<int64_t> kvflash_parse_host_mib(const char * env) {
    if (!env || !*env) return 16384;
    char * end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (*end != '\0' || v <= 0) return std::nullopt;
    return (int64_t)v;
}

// First reason the combination is refused, nullopt if it may run.
// Architecture-exact (qwen35 only) and decided on the REQUESTED combination,
// before any subclass policy (Qwen35MoeBackend::post_kvflash_init_gate) runs.
// PFlash compression is refused at launch (server_main) and an image request by the engine (supports_images).
inline std::optional<std::string> kvflash_concurrent_refusal(const KvFlashConcurrentRequest & r) {
    if (r.arch != "qwen35")
        return "concurrent KVFlash supports only dense arch qwen35 (got '" + r.arch + "')";
    if (r.draft) {
        if (!r.chain_spec_switch)
            return "concurrent KVFlash does not support a draft model unless LUCE_KVFLASH_CHAIN_SPEC=1 "
                   "(DFlash2 chain verify, test-only)";
        if (!r.draft_local_same_device || r.draft_block < 2 || r.draft_block > 16 || r.fa_window != 0)
            return "concurrent KVFlash admits a draft model only as a local same-device DFlash2 chain "
                   "(block size 2..16, --fa-window 0)";
    }
    if (r.tree)  return "concurrent KVFlash does not support tree verification";
    return std::nullopt;
}

// ── The concurrent profile: per-slot recall by default on qwen35, LRU on request ──
struct KvFlashProfileRequest {
    std::string arch;
    bool explicit_lru_policy = false;   // LUCE_KVFLASH_POLICY=lru or --kvflash-policy lru: the opt-out
    bool explicit_qk_policy = false, explicit_recall = false, explicit_uq = false;
    bool explicit_tau = false;       // the decode-time reselect: refused, the slot engine has none
    int  reselect_mode = 0;          // LUCE_KVFLASH_PREFILL_RESELECT as parsed: 4 scores by exact attention mass, 5 by the drafter
    bool multi_model = false;        // env defaults are process-wide: one model's defaults would leak into the next
};
struct KvFlashProfile {
    bool ok = true;
    std::string policy = "lru-only";
    bool install_recall_defaults = false;   // the server installs the serial mode-4 profile (kvflash_recall_env); a set variable keeps its value
    std::string error, log_line;
};
inline KvFlashProfile kvflash_resolve_concurrent_profile(const KvFlashProfileRequest & r) {
    KvFlashProfile p;
    // C1.4: the recall profile (QK policy, mode-4 reselect, recall blocks) runs per slot
    // when every piece of it is asked for explicitly; tau stays refused.
    if (r.explicit_tau) {
        p.ok = false;
        p.error = "concurrent KVFlash is LRU or per-slot recall; tau (the decode-time reselect) is not supported there";
        return p;
    }
    const bool any = r.explicit_qk_policy || r.explicit_recall || r.reselect_mode != 0;
    if (any) {
        if (!(r.explicit_qk_policy && r.explicit_recall && (r.reselect_mode == 4 || r.reselect_mode == 5))) {
            p.ok = false;
            p.error = "concurrent KVFlash recall needs LUCE_KVFLASH_POLICY=qk, LUCE_KVFLASH_PREFILL_RESELECT=4 (or 5) "
                      "and LUCE_KVFLASH_RECALL_BLOCKS together (C1.4: no defaults are installed)";
            return p;
        }
        p.policy = "recall";
        p.log_line = "[kvflash] concurrent profile: per-slot recall (C1.4, explicit env; serial defaults suppressed)";
        return p;
    }
    if (r.arch == "qwen35" && !r.explicit_lru_policy && !r.multi_model) {
        // One slot or many, the same recall settings: mode 4 with the question's queries and the preamble pin.
        p.policy = "recall";
        p.install_recall_defaults = true;
        p.log_line = "[kvflash] concurrent profile: per-slot recall (the mode-4 recall profile installed as defaults; "
                     "LUCE_KVFLASH_POLICY=lru keeps LRU only)";
        return p;
    }
    if (r.explicit_uq) {   // the question's queries feed the per-slot scorer: without it the setting would do nothing
        p.ok = false;
        p.error = "LUCE_KVFLASH_UQ_QUERIES needs per-slot recall; concurrent KVFlash is LRU only here";
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
    const char * rs = lookup("LUCE_KVFLASH_PREFILL_RESELECT");
    r.reselect_mode = rs ? std::atoi(rs) : 0;
    r.explicit_tau = on("LUCE_KVFLASH_TAU") || cli_tau;
    r.multi_model = multi_model;
    return r;
}

// After the defaults went into the environment (a set variable kept its value): the merged request must resolve
// through the explicit recall branch (recall, nothing left to install). Empty when coherent, else the refusal.
inline std::string kvflash_merged_profile_error(const std::string & arch, const std::string & cli_policy,
                                                bool cli_tau, bool multi_model, const KvFlashLookup & lookup) {
    const KvFlashProfile p = kvflash_resolve_concurrent_profile(kvflash_concurrent_request(arch, cli_policy, cli_tau, multi_model, lookup));
    if (p.ok && p.policy == "recall" && !p.install_recall_defaults) return std::string();
    return "a LUCE_KVFLASH_* value or --kvflash-policy keeps the C1.4 recall defaults from forming a recall profile "
           "(they need POLICY=qk, PREFILL_RESELECT=4, RECALL_BLOCKS>0); "
           "unset it, set the full recall env, or set LUCE_KVFLASH_POLICY=lru";
}

}  // namespace luce::common
