#pragma once

#include <cstdlib>
#include <cstring>

static inline bool ds4_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}
