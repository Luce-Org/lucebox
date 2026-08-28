#pragma once

#include <cstdlib>
#include <string>

// Effective graph-level FWHT K-rotation decision (TurboQuant-style outlier
// spreading applied before K-cache quantization). Single source of truth:
// the qwen35 target graph resolves it from the K-cache ggml type, and the
// disk prefix cache folds it into the identity salt, so a cache written
// under one rotation basis is never adopted by a session using the other.
//
// Unset DFLASH_KV_ROTATE resolves by type: rotation is precision-neutral for
// f16/q8_0 caches (skipped), kept for narrower types where spreading
// outliers buys accuracy. tq3_0 already rotates during quantization and
// never gets the graph-level rotation.
inline bool dflash_kv_k_rotation_enabled(const std::string & kv_k_type_name) {
    const char * e = std::getenv("DFLASH_KV_ROTATE");
    const int env_force = (!e || e[0] == '\0')
                              ? -1
                              : ((e[0] == '0' && e[1] == '\0') ? 0 : 1);
    const bool neutral = kv_k_type_name == "f16" || kv_k_type_name == "q8_0";
    const bool on = env_force < 0 ? !neutral : env_force != 0;
    return kv_k_type_name != "tq3_0" && on;
}
