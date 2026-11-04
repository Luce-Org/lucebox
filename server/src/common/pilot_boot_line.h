// The Lucepilot boot line: one "[pilot] {json}" line on stderr per process,
// printed when the server loads its model, after argument parsing and the
// launch profile's environment defaults. It records what the controller
// needs to attribute a run: the candidate, the build and the environment.
//
//   {"schema":1,
//    "candidate":<LUCE_PILOT_CANDIDATE or null>,
//    "build_sha":<LUCE_PILOT_BUILD_SHA of luce_pilot_build_info.h, or "unknown">,
//    "pilot":{<every LUCE_PILOT_* variable>},
//    "env":{<every LUCE_*, GGML_*, HIP_*, ROCR_*, DFLASH_* variable but LUCE_PILOT_*>}}
//
// Single-line JSON; names sort, the first definition of a name wins (as
// getenv sees it), and bytes that are not UTF-8 become U+FFFD.

#pragma once

#include "pilot_switch.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>

namespace luce::pilot {

inline constexpr int kBootLineSchema = 1;
inline constexpr const char * kCandidateVariable = "LUCE_PILOT_CANDIDATE";

// Prefixes of the variables the "env" object records.
inline bool boot_line_records_env(std::string_view name) {
    static constexpr std::string_view kPrefixes[] = {
        "LUCE_", "GGML_", "HIP_", "ROCR_", "DFLASH_",
    };
    for (const std::string_view prefix : kPrefixes) {
        if (name.substr(0, prefix.size()) == prefix) return true;
    }
    return false;
}

// The boot line's JSON for an environ-style block (NAME=VALUE strings,
// null-terminated; null is an empty environment).
inline std::string boot_line_json(char * const * envp, const char * build_sha) {
    using ordered_json = nlohmann::ordered_json;
    std::map<std::string, std::string> pilot;
    std::map<std::string, std::string> env;
    const std::string_view prefix(kPrefix);
    for (char * const * entry = envp; entry != nullptr && *entry != nullptr; ++entry) {
        const char * text = *entry;
        const char * eq = std::strchr(text, '=');
        if (eq == nullptr || eq == text) continue;
        std::string name(text, (size_t)(eq - text));
        if (name.compare(0, prefix.size(), prefix) == 0) {
            pilot.emplace(std::move(name), eq + 1);
        } else if (boot_line_records_env(name)) {
            env.emplace(std::move(name), eq + 1);
        }
    }

    ordered_json line = ordered_json::object();
    line["schema"] = kBootLineSchema;
    const auto candidate = pilot.find(kCandidateVariable);
    line["candidate"] = candidate != pilot.end()
        ? ordered_json(candidate->second) : ordered_json(nullptr);
    line["build_sha"] = (build_sha != nullptr && *build_sha != '\0') ? build_sha : "unknown";
    ordered_json pilot_object = ordered_json::object();
    for (const auto & [name, value] : pilot) pilot_object[name] = value;
    line["pilot"] = std::move(pilot_object);
    ordered_json env_object = ordered_json::object();
    for (const auto & [name, value] : env) env_object[name] = value;
    line["env"] = std::move(env_object);
    return line.dump(-1, ' ', false, ordered_json::error_handler_t::replace);
}

inline std::string boot_line(char * const * envp, const char * build_sha) {
    return "[pilot] " + boot_line_json(envp, build_sha);
}

// Print the boot line for the process environment, once per process.
inline void emit_boot_line_once(const char * build_sha) {
    static std::once_flag once;
    std::call_once(once, [build_sha] {
        const std::string line = boot_line(process_environment(), build_sha);
        std::fprintf(stderr, "%s\n", line.c_str());
        std::fflush(stderr);
    });
}

}  // namespace luce::pilot
