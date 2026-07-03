#pragma once

#include "ggml.h"

#include <cstdlib>
#include <cstring>

namespace dflash::common {

inline bool qwen35_env_flag_is_one(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool qwen35_should_use_graph_wht_k_rotation(ggml_type kv_k_type) {
    // tq3_0 applies WHT during quantization — never rotate at graph level.
    if (kv_k_type == GGML_TYPE_TQ3_0) {
        return false;
    }
    // Opt-in: skip graph-level FWHT for a q4_0 K cache to drop two turbo-wht
    // kernels per chunk and match llama.cpp's leaner q4_0 path. Off by default
    // so warm-restore snapshots written with rotated K stay valid across upgrade.
    if (kv_k_type == GGML_TYPE_Q4_0 && qwen35_env_flag_is_one("DFLASH_SKIP_WHT")) {
        return false;
    }
    return true;
}

}  // namespace dflash::common
