// Operator policy for routed-MoE expert storage.
//
// Parsing and precedence live here so model backends consume one resolved,
// typed value. They must not inspect CLI or environment strings themselves.

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace dflash::common {

inline constexpr const char * kMoeStorageEnvironment =
    "DFLASH_MOE_STORAGE";
inline constexpr const char * kLegacyMoeStorageEnvironment =
    "DFLASH_MOE_NVME_COLD_TIER";

enum class MoeStoragePolicy {
    Auto,
    Resident,
    Ssd,
};

inline const char * moe_storage_policy_name(MoeStoragePolicy policy) {
    switch (policy) {
        case MoeStoragePolicy::Auto:     return "auto";
        case MoeStoragePolicy::Resident: return "resident";
        case MoeStoragePolicy::Ssd:      return "ssd";
    }
    return "unknown";
}

inline bool parse_moe_storage_policy(std::string_view value,
                                     MoeStoragePolicy & out) {
    if (value == "auto") {
        out = MoeStoragePolicy::Auto;
        return true;
    }
    if (value == "resident") {
        out = MoeStoragePolicy::Resident;
        return true;
    }
    if (value == "ssd") {
        out = MoeStoragePolicy::Ssd;
        return true;
    }
    return false;
}

enum class MoeStoragePolicySource {
    Default,
    LegacyEnvironment,
    Environment,
    Cli,
};

inline const char * moe_storage_policy_source_name(
    MoeStoragePolicySource source) {
    switch (source) {
        case MoeStoragePolicySource::Default:           return "default";
        case MoeStoragePolicySource::LegacyEnvironment: return "legacy environment";
        case MoeStoragePolicySource::Environment:       return "environment";
        case MoeStoragePolicySource::Cli:               return "CLI";
    }
    return "unknown";
}

struct MoeStoragePolicyResolution {
    MoeStoragePolicy policy = MoeStoragePolicy::Auto;
    MoeStoragePolicySource source = MoeStoragePolicySource::Default;
    std::string error;
    std::string warning;

    bool ok() const { return error.empty(); }
};

// Resolution order is intentionally explicit and testable:
//   CLI > DFLASH_MOE_STORAGE > legacy DFLASH_MOE_NVME_COLD_TIER > auto.
// Environment values are parameters rather than read internally, keeping the
// function deterministic and avoiding hidden process-global state in tests.
inline MoeStoragePolicyResolution resolve_moe_storage_policy(
    std::optional<MoeStoragePolicy> cli,
    const char * environment,
    const char * legacy_environment) {
    MoeStoragePolicyResolution out;
    if (cli.has_value()) {
        out.policy = *cli;
        out.source = MoeStoragePolicySource::Cli;
        return out;
    }

    if (environment && environment[0] != '\0') {
        if (!parse_moe_storage_policy(environment, out.policy)) {
            out.error = "DFLASH_MOE_STORAGE expects auto, resident, or ssd; got '" +
                        std::string(environment) + "'";
            return out;
        }
        out.source = MoeStoragePolicySource::Environment;
        return out;
    }

    if (!legacy_environment || legacy_environment[0] == '\0') {
        return out;
    }

    const std::string_view legacy(legacy_environment);
    if (legacy == "auto") {
        out.policy = MoeStoragePolicy::Auto;
    } else if (legacy == "1" || legacy == "on" || legacy == "true" ||
               legacy == "ssd") {
        out.policy = MoeStoragePolicy::Ssd;
    } else if (legacy == "0" || legacy == "off" || legacy == "false" ||
               legacy == "resident") {
        out.policy = MoeStoragePolicy::Resident;
    } else {
        out.error =
            "DFLASH_MOE_NVME_COLD_TIER expects auto, on, or off; got '" +
            std::string(legacy_environment) + "'";
        return out;
    }
    out.source = MoeStoragePolicySource::LegacyEnvironment;
    out.warning =
        "DFLASH_MOE_NVME_COLD_TIER is deprecated; use --moe-storage or "
        "DFLASH_MOE_STORAGE";
    return out;
}

}  // namespace dflash::common
