// prof_env.h — single profiling switch for the serving hot paths.
//
// DFLASH_PROF is a comma-separated list of profiler names, replacing the
// per-profiler env-var sprawl. Example: DFLASH_PROF=step,verify,prefill
//   step     per-step decode laps (draft/verify/heads/commit/build)
//   verify   verify_batch sub-phases (prep/mask/upwait/gpu/read)
//   prefill  prefill batch sub-phases (build/fill/up/gpu/read)
// Unknown names are ignored. Profilers are debug instrumentation: zero cost
// when off, never required for correct serving.

#pragma once

#include <cstdlib>
#include <cstring>

namespace dflash::common {

inline bool dflash_prof_enabled(const char * name) {
    const char * e = std::getenv("DFLASH_PROF");
    if (!e) return false;
    const size_t n = std::strlen(name);
    for (const char * p = e; *p; ) {
        const char * q = std::strchr(p, ',');
        const size_t len = q ? (size_t)(q - p) : std::strlen(p);
        if (len == n && std::strncmp(p, name, n) == 0) return true;
        if (!q) break;
        p = q + 1;
    }
    return false;
}

}  // namespace dflash::common
