#pragma once

#include <cstdlib>

namespace dflash::common {

struct ChainRollbackPolicy {
    bool checkpoint_f32 = false;
    int fast_rollback_threshold = 5;
    bool diagnostics = false;
};

inline bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::atoi(value) != 0;
}

inline ChainRollbackPolicy resolve_chain_rollback_policy() {
    ChainRollbackPolicy policy;
    policy.checkpoint_f32 = env_flag_enabled("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32");
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
    return policy;
}

}  // namespace dflash::common
