// Pure startup refusals for KVFlash combinations. No ggml, no CUDA, so the
// feature gate and its test target can include this without the pager.
#pragma once

#include <cstdlib>
#include <string>

namespace luce::common {

// --ddtree with KVFlash fails closed. The DDTree loops commit every accepted
// token before they check termination and do not settle, so an EOS, cancel or
// budget end leaves the position one past AR's. Returns the one-line startup
// error, or "" when the combination is allowed.
inline std::string kvflash_ddtree_refusal(bool ddtree, bool kvflash_on) {
    if (!ddtree || !kvflash_on) return "";
    return "--ddtree cannot run with --kvflash (LUCE_KVFLASH): the DDTree loop does not "
           "settle terminal steps like AR; drop --ddtree";
}

// LUCE_MIN_TOKENS (the EOS floor, default off) with speculative decode and
// KVFlash is refused: the floor's tool injection replays tokens outside the
// AR-order pager and reselect accounting the spec loop guarantees. Callers pass
// kvflash_on = KVFlash requested AND a draft model configured, so AR with
// KVFlash is unchanged. `env_min_tokens` is the variable's value (null or
// unset = off). Returns the one-line startup error, or "".
inline std::string kvflash_min_tokens_refusal(bool kvflash_on, const char * env_min_tokens) {
    if (!kvflash_on || !env_min_tokens || std::atoi(env_min_tokens) <= 0) return "";
    return "LUCE_MIN_TOKENS cannot run with --kvflash (LUCE_KVFLASH) and a draft model: the EOS floor's tool "
           "injection is not AR-order equivalent under the KVFlash pager; unset LUCE_MIN_TOKENS";
}

}  // namespace luce::common
