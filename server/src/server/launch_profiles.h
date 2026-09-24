// Named launch profiles (`--profile <name>`).
//
// A profile is a qualified hardware + model configuration: the server flags
// and the tuning environment a documented recipe needs, under one name.
// Everything a profile sets is a default. Flags given on the command line
// replace the profile's value for that flag, and environment variables that
// are already set keep their value, so a profile never overrides an explicit
// choice.

#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace luce::server {

struct LaunchProfileFlag {
    const char * flag;
    const char * value;  // nullptr for boolean flags
};

struct LaunchProfileEnv {
    const char * name;
    const char * value;
};

struct LaunchProfile {
    const char * name;
    const char * summary;
    std::vector<LaunchProfileFlag> flags;
    std::vector<LaunchProfileEnv> env;
};

inline const std::vector<LaunchProfile> & launch_profiles() {
    static const std::vector<LaunchProfile> profiles = {
        {
            "ds4-strix",
            "DeepSeek V4 Flash on one Strix Halo (gfx1151); the gfx1151 device "
            "profile supplies the kernel defaults",
            {
                {"--max-ctx", "131072"},
                {"--chunk", "8192"},
                {"--ds4-fused-decode", nullptr},
                {"--ds4-fused-verify-f16-kv", nullptr},
                {"--ds4-expert-top-k", "6"},
                {"--ds4-prefill", "sparse"},
            },
            {},
        },
        {
            "ds4-r9700-strix",
            "DeepSeek V4 Flash with dense work and hot experts on an R9700 "
            "(gfx1201) and the remaining experts on Strix Halo (gfx1151)",
            {
                {"--target-device", "hip:0"},
                {"--expert-device", "hip:1"},
                {"--peer-access", nullptr},
                {"--max-ctx", "18432"},
                {"--chunk", "2048"},
                {"--ds4-fused-decode", nullptr},
                {"--ds4-expert-top-k", "6"},
                {"--ds4-prefill", "sparse"},
            },
            {
                {"LUCE_EXPERT_BUDGET_MB", "14350"},
                {"LUCE_DS4_SPEC_Q", "5"},
                {"LUCE_DS4_Q5_VERIFY", "1"},
                {"LUCE_DS4_FUSED_VERIFY", "1"},
                {"LUCE_DS4_FUSED_HYBRID_DECODE", "1"},
                {"LUCE_DS4_PINNED_ROLLBACK", "1"},
                {"LUCE_DS4_GPU_ARGMAX_VERIFY", "1"},
                {"LUCE_DS4_DRAFT_CONTEXT_KV_CACHE", "1"},
                {"LUCE_DS4_TP_ROUTE_PREFORK", "1"},
                {"LUCE_DS4_TP_DEVICE_JOIN", "1"},
                {"LUCE_DS4_TP_DEVICE_JOIN_SPLIT", "1"},
                {"LUCE_DS4_TP_FUSED_HC_JOIN", "1"},
                {"LUCE_DS4_TP_MAIN_ROUTE_WEIGHTS", "1"},
                {"LUCE_DS4_TP_COARSE_OWNER", "1"},
                {"LUCE_DS4_TP_NATIVE_ROUTE_WIDTH", "1"},
                {"LUCE_DS4_TP_MASKED_ROUTES", "1"},
                {"LUCE_DS4_TP_GROUPED_MMVQ", "1"},
                {"LUCE_DS4_TP_CAPTURE_CACHE_SLOTS", "4"},
                {"LUCE_MOE_TP_DYNAMIC_ROUTE_BALANCE", "1"},
                {"LUCE_MOE_TP_DYNAMIC_MAIN_SLOTS_X4", "13"},
                {"LUCE_MOE_DUPLICATE_HOT_ON_COLD", "1"},
                {"LUCE_MOE_FULL_COLD_PARALLEL", "1"},
                {"LUCE_MOE_PREFILL_PERSISTENT_OWNER_ALLOC", "1"},
                {"LUCE_DS4_HYBRID_PREFILL_GPU_HC", "1"},
                {"LUCE_DS4_HYBRID_PREFILL_EAGER", "1"},
                {"GGML_CUDA_BATCH_PEER_COPIES", "1"},
                {"LUCE_MMID_GROUPED", "1"},
                {"LUCE_MMID_GROUPED_TYPES", "8"},
                {"LUCE_MMID_GROUPED_DEVICE", "1"},
                {"LUCE_CUDA_MMVQ_MOE_ROWS_PER_BLOCK", "2"},
                {"LUCE_CUDA_MMVQ_MOE_FP3_PACKED24", "1"},
                {"LUCE_CUDA_MMVQ_FP4_X4", "1"},
                {"LUCE_DS4_DIRECT_INDEXER_TOPK", "1"},
                {"GGML_DS4_TOPK_BLOCK_RADIX", "1"},
                {"LUCE_DS4_MIX_MMQ_PREFILL", "1"},
                {"LUCE_CUDA_I32_REPEAT", "1"},
                {"ROCBLAS_USE_HIPBLASLT", "0"},
            },
        },
    };
    return profiles;
}

inline const LaunchProfile * find_launch_profile(const std::string & name) {
    for (const LaunchProfile & profile : launch_profiles()) {
        if (name == profile.name) return &profile;
    }
    return nullptr;
}

inline std::string launch_profile_names() {
    std::string names;
    for (const LaunchProfile & profile : launch_profiles()) {
        if (!names.empty()) names += ", ";
        names += profile.name;
    }
    return names;
}

// Flags that select the same setting. A profile's --target-device yields to
// an explicit --target-devices as well as to an explicit --target-device.
inline bool launch_flags_overlap(const std::string & a, const std::string & b) {
    auto target = [](const std::string & f) {
        return f == "--target-device" || f == "--target-devices";
    };
    return a == b || (target(a) && target(b));
}

// Profile flags the command line has not already set, flattened into argv
// tokens. `given` holds the tokens of the model block the profile applies to.
inline std::vector<std::string> launch_profile_args(
    const LaunchProfile & profile,
    const std::vector<std::string> & given)
{
    std::vector<std::string> out;
    for (const LaunchProfileFlag & entry : profile.flags) {
        bool explicit_flag = false;
        for (const std::string & token : given) {
            if (launch_flags_overlap(token, entry.flag)) {
                explicit_flag = true;
                break;
            }
        }
        if (explicit_flag) continue;
        out.emplace_back(entry.flag);
        if (entry.value) out.emplace_back(entry.value);
    }
    return out;
}

}  // namespace luce::server
