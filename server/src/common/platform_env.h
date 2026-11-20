// Cross-platform environment variable helpers.

#pragma once

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace luce::common {

// Match POSIX setenv() semantics on every platform. In particular,
// overwrite=false must preserve an existing value; _putenv_s() does not
// provide that behavior by itself.
inline int set_environment_variable(
        const char * name, const char * value, bool overwrite) {
#if defined(_WIN32)
    if (!overwrite && std::getenv(name) != nullptr) {
        return 0;
    }
    return ::_putenv_s(name, value);
#else
    return ::setenv(name, value, overwrite ? 1 : 0);
#endif
}

inline int unset_environment_variable(const char * name) {
#if defined(_WIN32)
    return ::_putenv_s(name, "");
#else
    return ::unsetenv(name);
#endif
}

// On/off switches read from the environment. Unset, empty, "0", "false", "no"
// and "off" (in any case) are off; any other value is on.
inline bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    char lower[6] = {};
    for (size_t i = 0; value[i] != '\0'; ++i) {
        if (i == sizeof(lower) - 1) {
            return true;  // longer than every off value
        }
        lower[i] = (char) std::tolower((unsigned char) value[i]);
    }
    return std::strcmp(lower, "0") != 0 && std::strcmp(lower, "false") != 0 &&
           std::strcmp(lower, "no") != 0 && std::strcmp(lower, "off") != 0;
}

}  // namespace luce::common
