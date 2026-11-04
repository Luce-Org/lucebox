// Lucepilot switches.
//
// A Lucepilot patch puts every performance change behind an environment
// switch named LUCE_PILOT_<FAMILY>_<KNOB>, the family one of GDN, MMQ, GLUE,
// DRAFT, TREE, ATTN, NOOP or FAULT. A switch is read once, into a
// function-local static at its call site:
//
//     static const luce::pilot::Switch kRowSplit("LUCE_PILOT_GDN_ROWSPLIT");
//     if (kRowSplit.on()) {
//         kRowSplit.engaged("rows=4");
//         ... the candidate's path ...
//     }
//
// or, inline, `if (LUCE_PILOT_SWITCH("LUCE_PILOT_GDN_ROWSPLIT").on())`.
//
// Values: unset keeps the parent's behaviour, "0" turns the switch off, and
// any other value (the empty string included) turns it on. Tuning knobs read
// a value instead: int_or() and string_or() return the parent's value unless
// the variable is set (to a whole number, for int_or()).
//
// The controller proves an arm key is live from two stderr lines, each
// printed once per process and name:
//     [pilot] read <NAME>=<value|unset>   the first time the switch is read
//     [pilot] engaged <NAME> <detail>     the first time it changes behaviour
// Both are printed only when the process was started with at least one
// LUCE_PILOT_* variable, so a launch without any keeps upstream's stderr.
// LUCE_PILOT_NOOP_PROBE (read_noop_probe(), at backend init) is the
// known-good key: it is read and does nothing else.
//
// Header-only: ggml cannot include it; ggml-cuda/luce-pilot.cuh is its
// counterpart there, with the same semantics and line format.

#pragma once

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <stdlib.h>
#else
extern "C" {
extern char ** environ;
}
#endif

namespace luce::pilot {

inline constexpr const char * kPrefix = "LUCE_PILOT_";
inline constexpr const char * kNoopProbe = "LUCE_PILOT_NOOP_PROBE";

enum class SwitchState { Unset, Off, On };

// Unset (null) keeps the parent, "0" is off, every other value is on.
inline SwitchState parse_switch(const char * value) noexcept {
    if (value == nullptr) return SwitchState::Unset;
    if (std::strcmp(value, "0") == 0) return SwitchState::Off;
    return SwitchState::On;
}

// A whole decimal number (optional sign) and nothing else.
inline bool parse_whole_number(const char * value, long long & out) noexcept {
    if (value == nullptr) return false;
    const char first = value[0];
    if (first != '-' && first != '+' && (first < '0' || first > '9')) return false;
    errno = 0;
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') return false;
    out = parsed;
    return true;
}

// The process environment as a NAME=VALUE block, null-terminated.
inline char * const * process_environment() noexcept {
#if defined(_WIN32)
    return _environ;
#else
    return environ;
#endif
}

inline bool has_pilot_variable(char * const * envp) noexcept {
    if (envp == nullptr) return false;
    const size_t prefix_len = std::strlen(kPrefix);
    for (char * const * entry = envp; *entry != nullptr; ++entry) {
        if (std::strncmp(*entry, kPrefix, prefix_len) == 0) return true;
    }
    return false;
}

// "[pilot] read NAME=value" ("unset" when the variable is absent). Control
// characters in the value print as '?', so the line stays one line.
inline std::string read_line(const char * name, const char * value) {
    std::string line = "[pilot] read ";
    line += name;
    line += '=';
    if (value == nullptr) {
        line += "unset";
        return line;
    }
    for (const char * c = value; *c != '\0'; ++c) {
        line += (static_cast<unsigned char>(*c) < 0x20) ? '?' : *c;
    }
    return line;
}

// "[pilot] engaged NAME detail" (no trailing space without a detail).
inline std::string engaged_line(const char * name, const char * detail) {
    std::string line = "[pilot] engaged ";
    line += name;
    if (detail != nullptr && *detail != '\0') {
        line += ' ';
        for (const char * c = detail; *c != '\0'; ++c) {
            line += (static_cast<unsigned char>(*c) < 0x20) ? '?' : *c;
        }
    }
    return line;
}

namespace detail {

using LineSink = void (*)(const char * line);

struct LogState {
    std::mutex mu;
    std::set<std::string> read;     // names whose read line was printed
    std::set<std::string> engaged;  // names whose engaged line was printed
    LineSink sink = nullptr;        // null: stderr
    int active = -1;                // -1: decided from the environment
};

inline LogState & log_state() {
    static LogState state;
    return state;
}

// Caller holds state.mu.
inline bool logging_active(LogState & state) {
    if (state.active < 0) {
        state.active = has_pilot_variable(process_environment()) ? 1 : 0;
    }
    return state.active == 1;
}

// Caller holds state.mu.
inline void write_line(LogState & state, const std::string & line) {
    if (state.sink != nullptr) {
        state.sink(line.c_str());
        return;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
}

// Tests only: route lines to `sink` (null: stderr), force the logging gate
// (`active` 1/0, -1 to decide from the environment again) and forget which
// names were already printed.
inline void reset_for_test(LineSink sink, int active) {
    LogState & state = log_state();
    std::lock_guard<std::mutex> lock(state.mu);
    state.sink = sink;
    state.active = active;
    state.read.clear();
    state.engaged.clear();
}

}  // namespace detail

// Print the read line for `name` unless it was printed already.
inline void note_read(const char * name, const char * value) {
    detail::LogState & state = detail::log_state();
    std::lock_guard<std::mutex> lock(state.mu);
    if (!detail::logging_active(state)) return;
    if (!state.read.insert(name).second) return;
    detail::write_line(state, read_line(name, value));
}

// Print the engaged line for `name` unless it was printed already. Call it
// where the switch actually changes what the engine does.
inline void engaged(const char * name, const char * detail = nullptr) {
    detail::LogState & state = detail::log_state();
    std::lock_guard<std::mutex> lock(state.mu);
    if (!detail::logging_active(state)) return;
    if (!state.engaged.insert(name).second) return;
    detail::write_line(state, engaged_line(name, detail));
}

// One LUCE_PILOT_* variable, read from the environment when constructed.
class Switch {
public:
    explicit Switch(const char * name) : name_(name) {
        const char * value = std::getenv(name);
        state_ = parse_switch(value);
        if (value != nullptr) value_ = value;
        note_read(name, value);
    }

    Switch(const Switch &) = delete;
    Switch & operator=(const Switch &) = delete;

    const char * name() const { return name_; }
    SwitchState state() const { return state_; }
    bool is_set() const { return state_ != SwitchState::Unset; }
    bool on() const { return state_ == SwitchState::On; }
    bool off() const { return state_ == SwitchState::Off; }
    // The raw value; empty when unset.
    const std::string & value() const { return value_; }

    // A tuning knob: the parent's value unless set to a whole number.
    long long int_or(long long parent) const {
        long long parsed = 0;
        if (!is_set() || !parse_whole_number(value_.c_str(), parsed)) return parent;
        return parsed;
    }

    // A tuning knob: the parent's value unless set.
    std::string string_or(std::string parent) const {
        return is_set() ? value_ : std::move(parent);
    }

    // One-shot engaged line; a relaxed flag keeps repeat calls lock-free.
    void engaged(const char * detail = nullptr) const {
        if (engaged_.load(std::memory_order_relaxed)) return;
        engaged_.store(true, std::memory_order_relaxed);
        ::luce::pilot::engaged(name_, detail);
    }

private:
    const char * name_;
    SwitchState state_ = SwitchState::Unset;
    std::string value_;
    mutable std::atomic<bool> engaged_{false};
};

// The known-good arm key: read once, changes nothing.
inline void read_noop_probe() {
    static const Switch probe(kNoopProbe);
    (void) probe;
}

}  // namespace luce::pilot

// A switch read once at this call site: LUCE_PILOT_SWITCH("NAME").on().
#define LUCE_PILOT_SWITCH(NAME)                                       \
    ([]() -> const ::luce::pilot::Switch & {                          \
        static const ::luce::pilot::Switch luce_pilot_switch_(NAME);  \
        return luce_pilot_switch_;                                    \
    }())
