// Launch profiles (`--profile <name>`).
//
// A profile is a qualified hardware + model configuration: the server flags
// and the tuning environment a documented recipe needs, under one name. The
// profiles live in the model cards (share/model_cards/<model>.json), beside
// the model's other Lucebox defaults. A card may name default profiles, and
// luce_server applies the first one whose qualified GPUs, with the memory it
// needs of each, are the first ones visible to a single model;
// `--profile <name>` picks another profile and `--profile none` applies none.
//
// Everything a profile sets is a default. Flags given on the command line
// replace the profile's value for that flag, `--no-<flag>` leaves a profile
// flag out, and environment variables that are already set keep their value,
// so a profile never overrides an explicit choice.

#pragma once

#include "placement/kfd_topology.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace luce::server {

struct LaunchProfileFlag {
    std::string flag;
    std::string value;  // empty for boolean flags
};

struct LaunchProfileEnv {
    std::string name;
    std::string value;
};

struct LaunchProfile {
    std::string name;
    std::vector<std::string> aliases;  // earlier names --profile still accepts
    std::string summary;
    // The GPU architectures the profile was qualified on, from hip:0 upward,
    // and the memory (GiB, as KFD reports it) each needs; empty: any.
    std::vector<std::string> gpus;
    std::vector<uint64_t> min_gpu_memory_gib;
    std::vector<LaunchProfileFlag> flags;
    std::vector<LaunchProfileEnv> env;
};

// Flags that select the same setting. A profile's --target-device yields to
// an explicit --target-devices as well as to an explicit --target-device.
inline bool launch_flags_overlap(const std::string & a, const std::string & b) {
    auto target = [](const std::string & f) {
        return f == "--target-device" || f == "--target-devices";
    };
    return a == b || (target(a) && target(b));
}

// The profiles of one model card, and the ones it applies by default.
struct CardProfiles {
    std::vector<LaunchProfile> profiles;
    std::vector<std::string> default_profiles;  // in order; the first that matches applies

    // The profile with this name or alias.
    const LaunchProfile * find(const std::string & name) const {
        for (const LaunchProfile & profile : profiles) {
            if (profile.name == name ||
                std::find(profile.aliases.begin(), profile.aliases.end(), name) != profile.aliases.end()) {
                return &profile;
            }
        }
        return nullptr;
    }

    // Whether a profile of this card sets `flag` (or a flag of the same setting).
    bool sets(const std::string & flag) const {
        for (const LaunchProfile & profile : profiles) {
            for (const LaunchProfileFlag & entry : profile.flags) {
                if (launch_flags_overlap(entry.flag, flag)) return true;
            }
        }
        return false;
    }
};

// Reads the "profiles" and "default_profile" fields of a model card. A card
// without them has no profiles. Flags keep the card's order.
inline bool parse_card_profiles(const std::string & text, CardProfiles & out, std::string & error) {
    using json = nlohmann::ordered_json;
    out = {};
    json card;
    try {
        card = json::parse(text);
    } catch (const json::exception & e) {
        error = e.what();
        return false;
    }
    if (!card.is_object()) {
        error = "not a JSON object";
        return false;
    }
    if (card.contains("profiles")) {
        if (!card["profiles"].is_object()) {
            error = "profiles is not an object";
            return false;
        }
        std::set<std::string> names;
        for (const auto & item : card["profiles"].items()) {
            LaunchProfile profile;
            profile.name = item.key();
            const std::string where = "profile " + profile.name + ": ";
            if (profile.name.empty() || profile.name == "none" || !item.value().is_object()) {
                error = where + "needs a name other than none and an object";
                return false;
            }
            if (!names.insert(profile.name).second) {
                error = where + "the name is already another profile's alias";
                return false;
            }
            for (const auto & field : item.value().items()) {
                const std::string & key = field.key();
                const json & value = field.value();
                if (key == "summary" && value.is_string()) {
                    profile.summary = value.get<std::string>();
                } else if (key == "notes" && value.is_string()) {
                    // For the reader: why the values are what they are.
                } else if (key == "aliases" && value.is_array()) {
                    for (const json & alias : value) {
                        if (!alias.is_string() || alias.get<std::string>().empty() ||
                            alias.get<std::string>() == "none" ||
                            !names.insert(alias.get<std::string>()).second) {
                            error = where + "aliases takes names no other profile or alias has, other than none";
                            return false;
                        }
                        profile.aliases.push_back(alias.get<std::string>());
                    }
                } else if (key == "gpus" && value.is_array()) {
                    for (const json & gpu : value) {
                        if (!gpu.is_string() || gpu.get<std::string>().rfind("gfx", 0) != 0) {
                            error = where + "gpus takes architecture names such as gfx1201";
                            return false;
                        }
                        profile.gpus.push_back(gpu.get<std::string>());
                    }
                } else if (key == "min_gpu_memory_gib" && value.is_array()) {
                    for (const json & gib : value) {
                        if (!gib.is_number_unsigned()) {
                            error = where + "min_gpu_memory_gib takes whole GiB, one per gpus entry";
                            return false;
                        }
                        profile.min_gpu_memory_gib.push_back(gib.get<uint64_t>());
                    }
                } else if (key == "flags" && value.is_object()) {
                    for (const auto & flag : value.items()) {
                        const bool boolean = flag.value().is_boolean() && flag.value().get<bool>();
                        if (flag.key().rfind("--", 0) != 0 || flag.key().rfind("--no-", 0) == 0 ||
                            flag.key() == "--profile" ||
                            !(boolean || (flag.value().is_string() &&
                                          !flag.value().get<std::string>().empty()))) {
                            error = where + "flag '" + flag.key() +
                                    "' needs a --name and a string value, or true";
                            return false;
                        }
                        profile.flags.push_back(
                            {flag.key(), boolean ? std::string() : flag.value().get<std::string>()});
                    }
                } else if (key == "env" && value.is_object()) {
                    for (const auto & env : value.items()) {
                        if (env.key().empty() || !env.value().is_string()) {
                            error = where + "env value of '" + env.key() + "' is not a string";
                            return false;
                        }
                        profile.env.push_back({env.key(), env.value().get<std::string>()});
                    }
                } else {
                    error = where + "unexpected field '" + key + "' or value";
                    return false;
                }
            }
            if (!profile.min_gpu_memory_gib.empty() &&
                profile.min_gpu_memory_gib.size() != profile.gpus.size()) {
                error = where + "min_gpu_memory_gib takes whole GiB, one per gpus entry";
                return false;
            }
            out.profiles.push_back(std::move(profile));
        }
    }
    if (card.contains("default_profile")) {
        // One name, or several to try in order.
        const json & names = card["default_profile"];
        const json list = names.is_array() ? names : json::array({names});
        for (const json & name : list) {
            const LaunchProfile * profile =
                name.is_string() ? out.find(name.get<std::string>()) : nullptr;
            if (!profile || profile->gpus.empty() ||
                std::find(out.default_profiles.begin(), out.default_profiles.end(), profile->name) !=
                    out.default_profiles.end()) {
                error = "default_profile must name profiles of the card that list their gpus, "
                        "each once";
                return false;
            }
            out.default_profiles.push_back(profile->name);
        }
        if (out.default_profiles.empty()) {
            error = "default_profile must name profiles of the card that list their gpus, each once";
            return false;
        }
    }
    return true;
}

// A card's profiles; a card that does not exist has none.
inline bool load_card_profiles(const std::filesystem::path & card, CardProfiles & out,
                               std::string & error) {
    out = {};
    std::error_code ec;
    const std::filesystem::file_status status = std::filesystem::status(card, ec);
    if (status.type() == std::filesystem::file_type::not_found) return true;
    if (ec || status.type() != std::filesystem::file_type::regular) {
        error = card.string() + ": " + (ec ? ec.message() : std::string("not a regular file"));
        return false;
    }
    std::ifstream in(card);
    std::stringstream text;
    text << in.rdbuf();
    if (!in || !parse_card_profiles(text.str(), out, error)) {
        error = card.string() + ": " + (error.empty() ? "cannot read" : error);
        return false;
    }
    return true;
}

// The model cards in a directory, by file name (the schema is not one).
inline std::vector<std::filesystem::path> model_card_files(const std::filesystem::path & dir) {
    std::vector<std::filesystem::path> cards;
    std::error_code ec;
    for (auto entry = std::filesystem::directory_iterator(dir, ec);
         !ec && entry != std::filesystem::directory_iterator(); entry.increment(ec)) {
        const std::filesystem::path & path = entry->path();
        if (path.extension() == ".json" && path.filename().string().rfind('_', 0) != 0) {
            cards.push_back(path);
        }
    }
    std::sort(cards.begin(), cards.end());
    return cards;
}

using VisibleGpu = luce::common::detail::KfdGpu;

// Whether a profile's GPUs are the first visible ones, in order, each with
// the memory the profile needs of it.
inline bool profile_gpus_match(const LaunchProfile & profile, const std::vector<VisibleGpu> & visible) {
    if (profile.gpus.empty() || visible.size() < profile.gpus.size()) return false;
    for (size_t i = 0; i < profile.gpus.size(); ++i) {
        if (visible[i].arch != profile.gpus[i]) return false;
        if (i < profile.min_gpu_memory_gib.size() &&
            visible[i].memory_bytes < (profile.min_gpu_memory_gib[i] << 30)) {
            return false;
        }
    }
    return true;
}

// The first of a card's default profiles that matches the visible GPUs.
inline const LaunchProfile * default_profile_for(const CardProfiles & card,
                                                 const std::vector<VisibleGpu> & visible) {
    for (const std::string & name : card.default_profiles) {
        const LaunchProfile * profile = card.find(name);
        if (profile && profile_gpus_match(*profile, visible)) return profile;
    }
    return nullptr;
}

inline std::string join_names(const std::vector<std::string> & names) {
    std::string out;
    for (const std::string & name : names) out += (out.empty() ? "" : ",") + name;
    return out;
}

// For messages: "gfx1201,gfx1151 160+ GiB", the GPUs a profile was qualified on.
inline std::string profile_gpus_text(const LaunchProfile & profile) {
    std::string out;
    for (size_t i = 0; i < profile.gpus.size(); ++i) {
        out += (i ? "," : "") + profile.gpus[i];
        if (i < profile.min_gpu_memory_gib.size() && profile.min_gpu_memory_gib[i] > 0) {
            out += " " + std::to_string(profile.min_gpu_memory_gib[i]) + "+ GiB";
        }
    }
    return out;
}

// For messages: "gfx1201 32 GiB,gfx1151 96 GiB".
inline std::string visible_gpus_text(const std::vector<VisibleGpu> & visible) {
    std::string out;
    for (const VisibleGpu & gpu : visible) {
        char gib[32];
        std::snprintf(gib, sizeof(gib), " %.0f GiB", (double) gpu.memory_bytes / (1ull << 30));
        out += (out.empty() ? "" : ",") + gpu.arch + gib;
    }
    return out;
}

// Environment defaults of expert-parallel cluster mode (--cluster-size > 1):
// the measured two-box DeepSeek V4.1 configuration. They are installed before
// a launch profile's defaults, so they win over a profile but never over a
// variable that is already set. Setting a switch to 0 turns it off;
// LUCE_CLUSTER_FAST_REDUCE_STALL_S=0 turns the stall watchdog off, and the
// cache slot counts and LUCE_DS4_ENGRAM_PRECOMPUTE take a size.
inline const std::vector<LaunchProfileEnv> & cluster_launch_env() {
    static const std::vector<LaunchProfileEnv> env = {
        // Decode exchange: the raw-RDMA fast reduce, zero-copy rows on a bf16
        // wire, no slot-reuse waits; the hybrid exchange reduces the iGPU's
        // cold partial with the R9700's hot + shared partial in one export.
        {"LUCE_CLUSTER_FAST_REDUCE", "1"},
        {"LUCE_CLUSTER_FAST_REDUCE_STALL_S", "15"},
        {"LUCE_CLUSTER_HYBRID_EXCHANGE", "1"},
        // Prefill-sized reductions stay on RCCL: 8-bit blocks, or bf16 with
        // LUCE_CLUSTER_PREFILL_Q8=0.
        {"LUCE_CLUSTER_PREFILL_Q8", "1"},
        {"LUCE_CLUSTER_PREFILL_BF16", "1"},
        // Work split: attention heads, the greedy verify head and the DSpark
        // draft head are halved across the ranks; each rank loads only its
        // slices of the dense weights.
        {"LUCE_CLUSTER_ATTENTION_PARALLEL", "1"},
        {"LUCE_CLUSTER_VOCAB_SPLIT", "1"},
        {"LUCE_CLUSTER_DRAFT_VOCAB_SPLIT", "1"},
        {"LUCE_CLUSTER_DRAFT_SPLIT", "1"},
        {"LUCE_CLUSTER_WORKER_SKIP_DRAFT", "1"},
        // DS4.1 verify step: batched verify lanes, the drafter's context K/V
        // window, GPU argmax, the drafter swapped out for long prompts (a
        // cluster rank runs without the prefix cache, so no cached prefix has
        // to match the swap's chunks), and the Engram projection run inside
        // the exchange wait.
        {"LUCE_DS4_GPU_ARGMAX_VERIFY", "1"},
        {"LUCE_DS4_LANE_BATCH", "1"},
        {"LUCE_DS4_DRAFT_CONTEXT_KV_CACHE", "1"},
        {"LUCE_DS4_DRAFT_SWAP", "1"},
        {"LUCE_DS4_SCHED_DEDICATED_COPIES", "1"},
        {"LUCE_DS4_HC_BOUNDARY_SPLIT", "1"},
        {"LUCE_DS4_ENGRAM_PRECOMPUTE", "2560"},
        {"LUCE_DS4_TP_CAPTURE_CACHE_SLOTS", "64"},
        {"LUCE_DS4_TP_FUSED_CACHE_SLOTS", "64"},
        // Expert products: q8_1 inputs handed to the expert owners, grouped
        // MUL_MAT_ID (in prefill on both owners: the R9700's hot experts too),
        // the ROCmFP2 prefetch loop and one peer-copy kernel per layer.
        {"LUCE_MOE_Q8_HANDOFF", "1"},
        {"LUCE_MMID_GROUPED", "1"},
        {"LUCE_DS4_GROUPED_HYBRID_PREFILL", "2"},
        {"LUCE_MMID_GROUPED_TYPES", "79"},
        {"LUCE_CUDA_MMVQ_MOE_FP2_PREFETCH", "1"},
        {"GGML_CUDA_PEER_COPY_KERNEL", "1"},
        // The fast reduce reads pinned host rows from the GPU.
        {"HSA_FORCE_FINE_GRAIN_PCIE", "1"},
    };
    return env;
}

// PFlash with a trained scoring head (PFLASH_SCORING_HEAD_GGUF): the release
// profile measured for the PFlash v2 post (Qwen3.8-27B target, Qwen3.5-0.8B
// drafter, R9700). With prefill compression on, the server installs these
// environment defaults and the threshold and curve below for flags the
// command line did not give; a variable that is already set keeps its value.
// Without a head the built-in defaults stay. The request-probe query needs
// PFLASH_REQUEST_PROBE_GGUF, and its candidates and pin the segment probe
// too; each yields to an experiment arm it cannot combine with.
inline constexpr int kPFlashReleaseThreshold = 16384;

inline const std::vector<std::pair<int, float>> & pflash_release_curve() {
    static const std::vector<std::pair<int, float>> curve = {
        {16384, 0.315f}, {32768, 0.270f}, {65536, 0.270f}, {120000, 0.225f}};
    return curve;
}

// Appends the TOKENS:RATIO points of one --prefill-curve argument to `curve`.
// Points may share an argument, separated by spaces or commas, so a quoted
// "16384:0.315 32768:0.270" reads the same as two arguments. A point needs
// TOKENS > 0 and 0 < RATIO <= 1 with nothing after the ratio.
inline bool parse_pflash_curve_arg(const char * arg, std::vector<std::pair<int, float>> & curve,
                                   std::string & error) {
    const auto is_sep = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == ','; };
    const char * p = arg;
    while (*p) {
        if (is_sep(*p)) { ++p; continue; }
        const char * end = p;
        while (*end && !is_sep(*end)) ++end;
        const std::string point(p, end);
        p = end;
        errno = 0;
        char * tok_end = nullptr;
        const long tok = std::strtol(point.c_str(), &tok_end, 10);
        char * ratio_end = nullptr;
        const float ratio = (*tok_end == ':') ? std::strtof(tok_end + 1, &ratio_end) : 0.0f;
        if (tok_end == point.c_str() || *tok_end != ':' || ratio_end == tok_end + 1 || *ratio_end != '\0' ||
            errno != 0 || tok <= 0 || tok > INT_MAX || !(ratio > 0.0f && ratio <= 1.0f)) {
            error = "bad point '" + point + "' (expected TOKENS:RATIO with TOKENS > 0 and 0 < RATIO <= 1)";
            return false;
        }
        curve.push_back({(int) tok, ratio});
    }
    return true;
}

inline std::vector<LaunchProfileEnv> pflash_release_env() {
    const auto set = [](const char * name) {
        const char * raw = std::getenv(name);
        return raw && *raw;
    };
    const auto is = [](const char * name, const char * value) {
        const char * raw = std::getenv(name);
        return raw && std::strcmp(raw, value) == 0;
    };
    if (!set("PFLASH_SCORING_HEAD_GGUF")) return {};
    std::vector<LaunchProfileEnv> env = {
        // Units are taken by relevance per token until the budget is full;
        // relevance spread too flat over a whole prompt gets full prefill.
        {"PFLASH_SELECT_MODE", "budget_only"},
        {"PFLASH_SELECT_SCORE", "density"},
        {"PFLASH_SELECT_CHUNK_SIZE", "256"},
        {"PFLASH_SELECT_QUERY_TOKENS", "128"},
        {"PFLASH_SELECT_QUERY_HEAD", "1"},
        {"PFLASH_SELECT_PARAGRAPH_JOIN", "0"},
        {"PFLASH_FALLBACK_SPREAD", "0.11"},
        // Admission counts only tokens no cache restores, a newly arriving
        // region is scored alone, and a drifted chat view is rebuilt only
        // once it holds 64K tokens.
        {"PFLASH_ADMISSION", "cache"},
        {"PFLASH_ARRIVE", "1"},
        {"PFLASH_CHAT_REBUILD_MIN_TOKENS", "65536"},
    };
    const bool segment_probe = set("PFLASH_SEGMENT_PROBE_GGUF");
    if (segment_probe) env.push_back({"PFLASH_SELECT_SEGMENTS", "probe"});
    // Another scorer or the identifier query replaces the request-probe query.
    if (set("PFLASH_REQUEST_PROBE_GGUF") &&
        (!set("PFLASH_SELECT_SCORER") || is("PFLASH_SELECT_SCORER", "head")) &&
        !is("PFLASH_EVAL_REF_IDENT_QUERY", "heur")) {
        env.push_back({"PFLASH_SELECT_QUERY_PROBE", "1"});
        if (segment_probe && !is("PFLASH_SELECT_QUERY_PROBE", "0") &&
            !set("PFLASH_REQUEST_STAGE2_GGUF")) {
            env.push_back({"PFLASH_QUERY_CANDIDATES", "1"});
            env.push_back({"PFLASH_QUERY_PIN", "512"});
        }
    }
    return env;
}

// The flag `--no-<flag>` leaves out of a profile ("--no-peer-access" ->
// "--peer-access"), or empty for a token of another form.
inline std::string dropped_profile_flag(const std::string & token) {
    return token.rfind("--no-", 0) == 0 && token.size() > 5 ? "--" + token.substr(5) : std::string();
}

// Profile flags the command line has not already set or dropped, flattened
// into argv tokens. `given` holds the tokens of the model block the profile
// applies to; `dropped` the flags its --no-<flag> tokens leave out.
inline std::vector<std::string> launch_profile_args(
    const LaunchProfile & profile,
    const std::vector<std::string> & given,
    const std::set<std::string> & dropped = {})
{
    // --draft-block-size reshapes a drafter loaded in process (--draft without
    // --draft-ipc-bin), and the engine refuses it otherwise; a profile sets it
    // only for such a block, so a model launched without its drafter still runs.
    const auto has = [&](const char * flag) {
        return std::find(given.begin(), given.end(), flag) != given.end();
    };
    const bool local_drafter = has("--draft") && !has("--draft-ipc-bin");
    std::vector<std::string> out;
    for (const LaunchProfileFlag & entry : profile.flags) {
        if (entry.flag == "--draft-block-size" && !local_drafter) continue;
        bool explicit_flag = std::any_of(dropped.begin(), dropped.end(),
            [&](const std::string & flag) { return launch_flags_overlap(flag, entry.flag); });
        for (const std::string & token : given) {
            if (launch_flags_overlap(token, entry.flag)) {
                explicit_flag = true;
                break;
            }
        }
        if (explicit_flag) continue;
        out.push_back(entry.flag);
        if (!entry.value.empty()) out.push_back(entry.value);
    }
    return out;
}

}  // namespace luce::server
