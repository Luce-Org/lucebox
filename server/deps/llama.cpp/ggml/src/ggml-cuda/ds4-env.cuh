#pragma once

#include <cstdlib>
#include <cstring>

static inline bool ds4_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static inline bool ds4_mla_split_kv_enabled(bool device_default) {
    const bool forced = ds4_env_flag_enabled("GGML_CUDA_MLA_SPLIT_KV") ||
                        ds4_env_flag_enabled("GGML_DS4_FA_SPLIT_KV");
    const bool disabled = ds4_env_flag_enabled("GGML_CUDA_MLA_NO_SPLIT_KV") ||
                          ds4_env_flag_enabled("GGML_DS4_FA_NO_SPLIT_KV");
    return !disabled && (forced || device_default);
}
