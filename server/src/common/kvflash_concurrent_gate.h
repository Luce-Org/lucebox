// Pure decisions for N KVFlash sessions on one seq engine.
// No startup, no GPU: server_main / the backend call these once their inputs
// are resolved, and the tests pin the table.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

namespace luce::common {

// ── When is the seq engine built? ──────────────────────────────────────
// Paged attention only: N > 1 or a KVFlash arena.
struct SeqRouteInput {
    bool paged = false;
    int  n_slots = 1;
    bool kvflash_arena = false;
};
inline bool seq_engine_wanted(const SeqRouteInput & in) {
    return in.paged && (in.n_slots > 1 || in.kvflash_arena);
}

// ── What may run the concurrent KVFlash arena? ─────────────────────────
struct KvFlashConcurrentRequest {
    std::string arch;
    bool draft = false, tree = false;
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
inline std::optional<std::string> kvflash_concurrent_refusal(const KvFlashConcurrentRequest & r) {
    if (r.arch != "qwen35")
        return "concurrent KVFlash supports only dense arch qwen35 (got '" + r.arch + "')";
    if (r.draft && (!r.draft_local_same_device || r.draft_block < 2 || r.draft_block > 16 || r.fa_window != 0))
        return "concurrent KVFlash admits a draft model only as a local same-device DFlash2 chain "
               "(block size 2..16, --fa-window 0)";
    if (r.tree) return "concurrent KVFlash does not support tree verification";
    return std::nullopt;
}

// Concurrent KVFlash runs per-slot mode-4 recall on single-model qwen35 unless LRU is requested.
inline bool kvflash_concurrent_use_recall(const std::string & arch, const std::string & cli_policy,
                                          const std::string & env_policy, bool multi_model) {
    const std::string & policy = cli_policy.empty() ? env_policy : cli_policy;
    return arch == "qwen35" && !multi_model && policy != "lru";
}

}  // namespace luce::common
