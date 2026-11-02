#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace luce::common {

struct ChainRollbackPolicy {
    // Exact F32 per-token checkpoints, and the rollback-from-one-accepted-token
    // they enable, are the tuned decode path: measured faster than the legacy
    // F16 replay on every target we ship, with byte-identical output. They are
    // therefore the default rather than something a launch script has to opt
    // into. It costs VRAM: the per-token SSM checkpoints move from F16 to F32,
    // measured at +1.11 GiB on Qwen3.8-27B at a 128K context (22.47 vs 21.36
    // GiB total). LUCE_SINGLE_CHAIN_CHECKPOINT_F32=0 restores the legacy
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
    return env_flag_enabled("LUCE_SPLIT_FAST_ROLLBACK");
}

// Speculative fast rollback under the KVFlash pager is measured not
// token-identical to AR (25/80 outputs differ), so with the pager it needs an
// explicit --fast-rollback; the default there is exact restore + replay.
// Without the pager the requested mode stands.
inline bool spec_fast_rollback_enabled(bool pager_on, bool fast_rollback, bool explicit_optin) {
    return fast_rollback && (!pager_on || explicit_optin);
}

inline ChainRollbackPolicy resolve_chain_rollback_policy(
        bool tensor_parallel = false,
        bool exact_fast_rollback = false) {
    ChainRollbackPolicy policy;
    if (const char * e = std::getenv("LUCE_SINGLE_CHAIN_CHECKPOINT_F32")) {
        policy.checkpoint_f32 = (e[0] != '\0' && std::strcmp(e, "0") != 0);
    }
    if (!policy.checkpoint_f32) {
        policy.fast_rollback_threshold = 5;   // legacy F16 replay behaviour
    }
    policy.diagnostics = env_flag_enabled("LUCE_SINGLE_CHAIN_ROLLBACK_DIAG");

    // Lower thresholds are valid only with exact F32 checkpoints. This keeps
    // the established F16 behavior unchanged when no opt-in flags are set.
    if (policy.checkpoint_f32) {
        const char * value = std::getenv("LUCE_FAST_ROLLBACK_THRESHOLD");
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

// The recurrent-state shape of a Gated-DeltaNet target, as its GGUF header
// gives it (<arch>.ssm.* and the layer counts).
struct RollbackCacheDims {
    int n_delta_layers = 0;   // target layers that are not full attention
    int ssm_d_inner = 0;      // ssm.inner_size
    int ssm_dt_rank = 0;      // ssm.time_step_rank
    int ssm_d_state = 0;      // ssm.state_size
    int ssm_n_group = 0;      // ssm.group_count
    int ssm_d_conv = 0;       // ssm.conv_kernel
};

// Bytes migrate_prefill_cache() allocates on the target's device at the first
// speculative request, which is after the load succeeded: per delta layer the
// SSM and conv snapshots, then either the conv input window and one SSM
// checkpoint per verified token (F32 or F16), or with SpecLA the two factor
// banks. Placement counts them because nothing else reserves them: Qwen3.8-27B
// with an 8-token drafter allocates 1,386,479,616 bytes (1322.25 MiB) here.
// Keep in step with migrate_prefill_cache() (qwen35_target_graph.cpp).
inline uint64_t rollback_cache_bytes(const RollbackCacheDims & d,
                                     int max_verify_tokens,
                                     bool checkpoint_f32,
                                     bool specla) {
    if (d.n_delta_layers <= 0 || d.ssm_dt_rank <= 0 || d.ssm_d_inner <= 0 ||
        d.ssm_d_conv <= 0 || max_verify_tokens <= 0) {
        return 0;
    }
    const uint64_t f32 = sizeof(float);
    const uint64_t q = (uint64_t) max_verify_tokens;
    const uint64_t rank = (uint64_t) d.ssm_dt_rank;
    const uint64_t head_v = (uint64_t) d.ssm_d_inner / rank;
    const uint64_t head_k = (uint64_t) d.ssm_d_state;
    const uint64_t conv_ch =
        (uint64_t) d.ssm_d_inner + 2ull * (uint64_t) d.ssm_n_group * (uint64_t) d.ssm_d_state;
    const uint64_t conv_taps = (uint64_t) d.ssm_d_conv - 1;

    const uint64_t ssm_state = head_v * head_v * rank * f32;
    uint64_t per_layer = ssm_state + conv_taps * conv_ch * f32;   // the two snapshots
    if (specla) {
        // Two banks of k, v, g and conv factors per verified token.
        per_layer += 2 * q * (head_k * rank + head_v * rank + rank + conv_ch) * f32;
    } else {
        per_layer += (conv_taps + q) * conv_ch * f32;             // conv input window
        per_layer += q * (checkpoint_f32 ? ssm_state : ssm_state / 2);
    }
    return per_layer * (uint64_t) d.n_delta_layers;
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
            policy.checkpoint_f32 ? "F32" : "F16",
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

}  // namespace luce::common
