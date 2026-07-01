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
    if (std::getenv("DFLASH_NO_WHT") != nullptr) {
        return false;
    }
    if (kv_k_type == GGML_TYPE_TQ3_0) {
        return false;
    }
    if (qwen35_env_flag_is_one("DFLASH_FORCE_WHT")) {
        return true;
    }
    // q4_0 on Ampere is already fast enough without graph-level FWHT; avoiding
    // it removes extra prefill/decode kernels in the parity path.
    return kv_k_type != GGML_TYPE_Q4_0;
}

}  // namespace dflash::common
