// Cross-platform environment variable helpers.

#pragma once

#include <cstdlib>
#include <cstring>

namespace dflash::common {

// Treat an unset, empty, or explicit "0" value as disabled. This lets a
// caller override a default with FOO=0 instead of mere variable presence
// accidentally enabling the feature.
inline bool environment_variable_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' &&
           std::strcmp(value, "0") != 0;
}

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

}  // namespace dflash::common
