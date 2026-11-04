#pragma once

// Lucepilot switches for ggml-cuda host code: the counterpart of the server's
// src/common/pilot_switch.h, which ggml cannot include. Same semantics: a
// LUCE_PILOT_<FAMILY>_<KNOB> variable read once per call site; unset keeps the
// parent's behaviour, "0" is off, any other value is on. Same stderr lines,
// once per process and name, and only when the process was started with a
// LUCE_PILOT_* variable:
//     [pilot] read <NAME>=<value|unset>
//     [pilot] engaged <NAME> <detail>
// The two headers keep separate registries, so a name read on both sides
// prints its read line twice. Host-only.
//
//     if (GGML_LUCE_PILOT_ON("LUCE_PILOT_MMQ_TILE")) {
//         ggml_luce_pilot::engaged("LUCE_PILOT_MMQ_TILE", "x=64");
//         ...
//     }

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

#if defined(_WIN32)
#include <stdlib.h>
#else
extern "C" {
extern char ** environ;
}
#endif

namespace ggml_luce_pilot {

enum class state { unset, off, on };

inline state parse(const char * value) {
    if (value == nullptr) return state::unset;
    return std::strcmp(value, "0") == 0 ? state::off : state::on;
}

// Whether this process was started with a LUCE_PILOT_* variable.
inline bool log_active() {
    static const bool active = [] {
#if defined(_WIN32)
        char ** envp = _environ;
#else
        char ** envp = environ;
#endif
        for (char ** e = envp; e != nullptr && *e != nullptr; ++e) {
            if (std::strncmp(*e, "LUCE_PILOT_", 11) == 0) return true;
        }
        return false;
    }();
    return active;
}

// Print `line` once per (kind, name).
inline void log_once(const char * kind, const char * name, const std::string & line) {
    static std::mutex mu;
    static std::set<std::string> printed;
    if (!log_active()) return;
    std::lock_guard<std::mutex> lock(mu);
    if (!printed.insert(std::string(kind) + ' ' + name).second) return;
    std::string clean = line;
    for (char & c : clean) {
        if (static_cast<unsigned char>(c) < 0x20) c = '?';
    }
    std::fprintf(stderr, "%s\n", clean.c_str());
    std::fflush(stderr);
}

inline state read(const char * name) {
    const char * value = std::getenv(name);
    log_once("read", name, std::string("[pilot] read ") + name + '=' + (value ? value : "unset"));
    return parse(value);
}

// A tuning knob: the parent's value unless set to a whole number.
inline long long read_int(const char * name, long long parent) {
    const char * value = std::getenv(name);
    log_once("read", name, std::string("[pilot] read ") + name + '=' + (value ? value : "unset"));
    if (value == nullptr) return parent;
    if (value[0] != '-' && value[0] != '+' && (value[0] < '0' || value[0] > '9')) return parent;
    errno = 0;
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    return (errno != 0 || end == value || *end != '\0') ? parent : parsed;
}

inline void engaged(const char * name, const char * detail = nullptr) {
    std::string line = std::string("[pilot] engaged ") + name;
    if (detail != nullptr && *detail != '\0') line += std::string(" ") + detail;
    log_once("engaged", name, line);
}

}  // namespace ggml_luce_pilot

// Read once at this call site.
#define GGML_LUCE_PILOT_ON(NAME)                                                 \
    ([]() -> bool {                                                              \
        static const bool luce_pilot_on_ =                                       \
            ggml_luce_pilot::read(NAME) == ggml_luce_pilot::state::on;           \
        return luce_pilot_on_;                                                   \
    }())
