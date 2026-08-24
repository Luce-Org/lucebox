#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

struct ChainRollbackPolicy {
    // Exact F32 per-token checkpoints, and the rollback-from-one-accepted-token
    // they enable, are the tuned decode path: measured faster than the legacy
    // F16 replay on every target we ship, with byte-identical output. They are
    // therefore the default rather than something a launch script has to opt
    // into. It costs VRAM: the per-token SSM checkpoints move from F16 to F32,
    // measured at +1.11 GiB on Qwen3.8-27B at a 128K context (22.47 vs 21.36
    // GiB total). DFLASH_SINGLE_CHAIN_CHECKPOINT_F32=0 restores the legacy
    // F16 path and its rollback threshold of 5 when that headroom matters.
    bool checkpoint_f32 = true;
    int fast_rollback_threshold = 1;
    bool diagnostics = false;
};

inline bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

// Layer-split fast rollback needs exact F32 per-token checkpoints. Keep the
// feature, its capture work, and its additional memory behind one shared
// opt-in so allocation and runtime dispatch cannot drift apart.
inline bool split_chain_fast_rollback_enabled() {
    return env_flag_enabled("DFLASH_SPLIT_FAST_ROLLBACK");
}

inline ChainRollbackPolicy resolve_chain_rollback_policy(
        bool tensor_parallel = false,
        bool exact_fast_rollback = false) {
    ChainRollbackPolicy policy;
    if (const char * e = std::getenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32")) {
        policy.checkpoint_f32 = (e[0] != '\0' && std::strcmp(e, "0") != 0);
    }
    if (!policy.checkpoint_f32) {
        policy.fast_rollback_threshold = 5;   // legacy F16 replay behaviour
    }
    policy.diagnostics = env_flag_enabled("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG");

    // Lower thresholds are valid only with exact F32 checkpoints. This keeps
    // the established F16 behavior unchanged when no opt-in flags are set.
    if (policy.checkpoint_f32) {
        const char * value = std::getenv("DFLASH_FAST_ROLLBACK_THRESHOLD");
        if (value != nullptr) {
            const int requested = std::atoi(value);
            if (requested >= 1 && requested <= 5) {
                policy.fast_rollback_threshold = requested;
            }
        }
    }
    // Tensor-parallel rollback restores each rank's device-local recurrent
    // state directly, so it is cheaper than replay even for one accepted
    // token and does not depend on the host checkpoint precision.
    if (tensor_parallel) {
        policy.fast_rollback_threshold = 1;
    }
    // Exact device-side rollback (SpecLA factor commit, or an equivalent
    // implementation) is profitable from the first accepted token. This is
    // an actual target capability, not an environment flag: the requested
    // mode may be unavailable for the active cache/backend.
    if (exact_fast_rollback) {
        policy.fast_rollback_threshold = 1;
    }
    return policy;
}

// Rollback diagnostics shared by the two spec-decode loops
// (Qwen35Backend::do_spec_decode and run_dflash_spec_decode). Keeping the
// counters and the print format in one place prevents the loops from
// drifting when the diagnostics change.
struct RollbackDiag {
    int accept_hist[17] = {};
    int fast_low = 0;         // fast rollbacks below the F16 breakeven (accept_n < 5)
    int fast_high = 0;        // fast rollbacks at accept_n >= 5
    int legacy_replay = 0;
    int failed_fallback = 0;

    void record_accept(int accept_n) {
        accept_hist[std::min(accept_n, 16)]++;
    }
    void record_fast_rollback(int accept_n) {
        if (accept_n < 5) fast_low++;
        else fast_high++;
    }
    void record_failed_fallback() { failed_fallback++; }
    void record_legacy_replay()   { legacy_replay++; }

    void print(const ChainRollbackPolicy & policy, std::FILE * out) const {
        if (!out || !policy.diagnostics) return;
        std::fprintf(out,
            "[chain-rollback-policy] checkpoint=%s threshold=%d fast_low=%d fast_high=%d legacy_replay=%d failed_fallback=%d accept_hist=1:%d,2:%d,3:%d,4:%d,5:%d,6:%d,7:%d,8:%d,9:%d,10:%d,11:%d,12:%d,13:%d,14:%d,15:%d,16+:%d\n",
            policy.checkpoint_f32 ? "F32" : "default",
            policy.fast_rollback_threshold,
            fast_low, fast_high, legacy_replay, failed_fallback,
            accept_hist[1], accept_hist[2], accept_hist[3],
            accept_hist[4], accept_hist[5], accept_hist[6],
            accept_hist[7], accept_hist[8], accept_hist[9],
            accept_hist[10], accept_hist[11], accept_hist[12],
            accept_hist[13], accept_hist[14], accept_hist[15],
            accept_hist[16]);
    }
};

}  // namespace dflash::common
