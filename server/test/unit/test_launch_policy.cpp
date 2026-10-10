// Unit tests for launch policy: `--target-device auto` selection and
// `--profile` expansion. Both are pure functions over resolved facts, so
// they need no model file or GPU.

#include "CppUnitTestFramework.hpp"
#include "placement/device_select.h"
#include "common/feature_gate.h"
#include "server/launch_profiles.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace CppUnitTestFramework;
using namespace luce::common;
using namespace luce::server;

namespace {

constexpr uint64_t GiB = 1ull << 30;

GpuDeviceInfo device(int index, uint64_t total_gib, bool integrated) {
    GpuDeviceInfo info;
    info.index = index;
    info.total_bytes = total_gib * GiB;
    info.integrated = integrated;
    return info;
}

bool contains_flag(const std::vector<std::string> & args, const char * flag) {
    for (const std::string & arg : args) {
        if (arg == flag) return true;
    }
    return false;
}

struct LaunchPolicyFixture : CommonFixture {
    using CommonFixture::CommonFixture;

    void test_auto_device_prefers_discrete_gpu_that_fits() {
        // R9700 (32 GiB, device 0) + Strix Halo (96 GiB, integrated, device 1).
        const std::vector<GpuDeviceInfo> devices = {
            device(0, 32, false), device(1, 96, true)};
        const AutoDeviceChoice qwen = choose_auto_target_device(devices, 15 * GiB);
        CHECK(qwen.index == 0);
        CHECK(qwen.fits);

        // The same pair enumerated the other way round still picks the dGPU.
        const std::vector<GpuDeviceInfo> reversed = {
            device(0, 96, true), device(1, 32, false)};
        CHECK(choose_auto_target_device(reversed, 15 * GiB).index == 1);
    }

    void test_auto_device_moves_large_model_to_device_that_fits() {
        const std::vector<GpuDeviceInfo> devices = {
            device(0, 32, false), device(1, 120, true)};
        const AutoDeviceChoice ds4 = choose_auto_target_device(devices, 88 * GiB);
        CHECK(ds4.index == 1);
        CHECK(ds4.fits);
    }

    void test_auto_device_falls_back_to_largest() {
        // Nothing holds 100 GiB of weights; the largest device is still the
        // right one to try.
        const std::vector<GpuDeviceInfo> devices = {
            device(0, 32, false), device(1, 96, true)};
        const AutoDeviceChoice big = choose_auto_target_device(devices, 100 * GiB);
        CHECK(big.index == 1);
        CHECK(!big.fits);

        // Issue #712: 91.5 GiB of DeepSeek V4 weights on R9700 + Strix Halo.
        const AutoDeviceChoice ds4 = choose_auto_target_device(
            devices, 91 * GiB + GiB / 2);
        CHECK(ds4.index == 1);
        CHECK(ds4.fits);

        CHECK(choose_auto_target_device({}, GiB).index == -1);
    }

    void test_auto_device_keeps_first_of_equal_discrete_gpus() {
        const std::vector<GpuDeviceInfo> devices = {
            device(0, 24, false), device(1, 24, false)};
        CHECK(choose_auto_target_device(devices, 10 * GiB).index == 0);
    }

    void test_required_bytes_margin() {
        CHECK(auto_device_required_bytes(GiB) == 3 * GiB);
        CHECK(auto_device_required_bytes(30 * GiB) == 33 * GiB);
        CHECK(auto_device_required_bytes(100 * GiB) == 104 * GiB);
    }

    void test_auto_device_counts_the_kv_cache() {
        // 26 GiB of weights fit a 32 GiB card with the flat margin alone, but
        // not with the KV cache of a long context on top.
        const std::vector<GpuDeviceInfo> devices = {
            device(0, 32, false), device(1, 96, true)};
        CHECK(auto_device_required_bytes(26 * GiB, 5 * GiB) ==
              26 * GiB + 5 * GiB + 26 * GiB / 10);
        CHECK(choose_auto_target_device(devices, 26 * GiB).index == 0);
        const AutoDeviceChoice long_ctx = choose_auto_target_device(devices, 26 * GiB, 5 * GiB);
        CHECK(long_ctx.index == 1);
        CHECK(long_ctx.fits);
    }

    // R9700 (32 GiB) + Strix Halo, as HIP reports them: the integrated GPU's
    // total is its address space, its free memory is what a model can use.
    std::vector<GpuDeviceInfo> r9700_strix(bool r9700_busy, bool strix_busy) {
        GpuDeviceInfo r9700 = device(0, 32, false);
        r9700.free_bytes = 31 * GiB;
        r9700.busy = r9700_busy;
        r9700.busy_by = r9700_busy ? "pid 42 luce_server" : "";
        GpuDeviceInfo strix = device(1, 1280, true);
        strix.free_bytes = 120 * GiB;
        strix.busy = strix_busy;
        return {r9700, strix};
    }

    void test_auto_placement_follows_gpu_availability() {
        AutoPlacementRequest ds4;
        ds4.model_bytes = 91 * GiB + GiB / 2;
        ds4.technique = arch_multi_gpu_technique("deepseek4");
        CHECK(ds4.technique == MultiGpuTechnique::ExpertParallel);

        // Both free: dense work on the R9700, the other experts on Strix Halo.
        const AutoPlacement both = choose_auto_placement(r9700_strix(false, false), ds4);
        CHECK(both.technique == MultiGpuTechnique::ExpertParallel);
        CHECK(both.gpus == std::vector<int>({0, 1}));
        CHECK(both.fits);

        // R9700 busy: Strix Halo alone, and the reason names the busy process.
        const AutoPlacement strix = choose_auto_placement(r9700_strix(true, false), ds4);
        CHECK(strix.technique == MultiGpuTechnique::None);
        CHECK(strix.gpus == std::vector<int>({1}));
        CHECK(strix.reason.find("pid 42 luce_server") != std::string::npos);

        // Strix Halo busy: the R9700 cannot hold DeepSeek4, which cannot
        // offload, so there is no placement rather than a failed load.
        CHECK(choose_auto_placement(r9700_strix(false, true), ds4).gpus.empty());
        CHECK(choose_auto_placement(r9700_strix(true, true), ds4).gpus.empty());

        // A model that cannot offload never lands on a GPU that is too small.
        AutoPlacementRequest offload = ds4;
        offload.can_offload = true;
        CHECK(choose_auto_placement(r9700_strix(false, true), offload).gpus ==
              std::vector<int>({0}));
    }

    void test_auto_placement_splits_only_what_one_discrete_gpu_cannot_hold() {
        AutoPlacementRequest large;
        large.model_bytes = 48 * GiB;
        large.technique = arch_multi_gpu_technique("qwen35");
        CHECK(large.technique == MultiGpuTechnique::LayerSplit);

        // Layers follow free memory: 31 GiB on the R9700, 120 GiB on Strix.
        const AutoPlacement split = choose_auto_placement(r9700_strix(false, false), large);
        CHECK(split.technique == MultiGpuTechnique::LayerSplit);
        CHECK(split.gpus == std::vector<int>({0, 1}));
        CHECK(split.layer_weights == std::vector<double>({31.0 * GiB, 120.0 * GiB}));

        // A model the R9700 holds alone stays there even when both are free:
        // splitting it with Strix Halo only slows it down.
        AutoPlacementRequest small = large;
        small.model_bytes = 15 * GiB;
        const AutoPlacement alone = choose_auto_placement(r9700_strix(false, false), small);
        CHECK(alone.technique == MultiGpuTechnique::None);
        CHECK(alone.gpus == std::vector<int>({0}));

        // Without a multi-GPU technique (or under a profile) one GPU must fit.
        AutoPlacementRequest single = large;
        single.technique = MultiGpuTechnique::None;
        CHECK(choose_auto_placement(r9700_strix(false, false), single).gpus ==
              std::vector<int>({1}));
    }

    void test_auto_placement_never_uses_exhausted_devices() {
        auto devices = r9700_strix(false, false);
        devices[0].free_bytes = 0;
        AutoPlacementRequest request;
        request.model_bytes = 15 * GiB;
        request.technique = MultiGpuTechnique::LayerSplit;
        const AutoPlacement placement = choose_auto_placement(devices, request);
        CHECK(placement.gpus == std::vector<int>({1}));
        CHECK(placement.fits);

        devices[1].free_bytes = 0;
        CHECK(choose_auto_placement(devices, request).gpus.empty());
        request.can_offload = true;
        CHECK(choose_auto_placement(devices, request).gpus.empty());
    }

    void test_auto_placement_preserves_requested_features() {
        const auto devices = r9700_strix(false, false);
        AutoPlacementRequest request;
        request.model_bytes = 48 * GiB;
        BackendArgs args;
        args.paged_attention = true;
        request.technique = auto_multi_gpu_technique("qwen35", args);
        const AutoPlacement paged = choose_auto_placement(devices, request);
        CHECK(paged.technique == MultiGpuTechnique::None);
        CHECK(paged.gpus == std::vector<int>({1}));
        args.device.backend = compiled_placement_backend();
        args.device.gpu = paged.gpus.front();
        CHECK(check_feature_compatibility(args, {}, "qwen35",
              compiled_placement_backend(), compiled_placement_backend()).empty());

        args.paged_attention = false;
        args.mmproj_path = "vision.gguf";
        request.technique = auto_multi_gpu_technique("qwen35", args);
        CHECK(choose_auto_placement(devices, request).gpus == std::vector<int>({1}));

        args.mmproj_path.reset();
        args.draft_path = "draft.gguf";
        request.technique = auto_multi_gpu_technique("gemma4", args);
        CHECK(choose_auto_placement(devices, request).gpus == std::vector<int>({1}));
        // Qwen's layer adapter does support its ordinary decode drafter.
        request.technique = auto_multi_gpu_technique("qwen35", args);
        CHECK(choose_auto_placement(devices, request).technique == MultiGpuTechnique::LayerSplit);
    }

    void test_auto_placement_preserves_separate_vision_encoder() {
        const auto devices = r9700_strix(false, false);
        BackendArgs args;
        args.mmproj_path = "vision.gguf";
        DevicePlacement encoder;
        encoder.backend = compiled_placement_backend();
        encoder.gpu = 0;
        args.mmproj_device = encoder;
        AutoPlacementRequest request;
        request.model_bytes = 91 * GiB;
        request.technique = auto_multi_gpu_technique("deepseek4", args);
        const AutoPlacement placement = choose_auto_placement(devices, request);
        CHECK(placement.technique == MultiGpuTechnique::None);
        CHECK(placement.gpus == std::vector<int>({1}));

        // Without a separately placed encoder, two expert owners are valid.
        args.mmproj_device.reset();
        request.technique = auto_multi_gpu_technique("deepseek4", args);
        CHECK(choose_auto_placement(devices, request).technique ==
              MultiGpuTechnique::ExpertParallel);
    }

    void test_paged_expert_parallel_requires_supported_devices() {
        auto devices = r9700_strix(false, false);
        devices[0].arch = "gfx1201";
        devices[1].arch = "gfx1151";
        AutoPlacementRequest request;
        request.model_bytes = 91 * GiB;
        request.technique = MultiGpuTechnique::ExpertParallel;
        request.paged_expert_parallel = true;
        const AutoPlacement supported = choose_auto_placement(devices, request);
        if (compiled_placement_backend() == PlacementBackend::Hip) {
            CHECK(supported.technique == MultiGpuTechnique::ExpertParallel);
            CHECK(supported.gpus == std::vector<int>({0, 1}));
        } else {
            CHECK(supported.gpus == std::vector<int>({1}));
        }
        devices[0].arch = "gfx1100";
        const AutoPlacement unsupported = choose_auto_placement(devices, request);
        CHECK(unsupported.technique == MultiGpuTechnique::None);
        CHECK(unsupported.gpus == std::vector<int>({1}));
    }

    void test_draft_placement_precedence() {
        DevicePlacement target;
        target.backend = compiled_placement_backend();
        target.gpu = 1;
        DevicePlacement unplaced;  // auto:0, the parser default

        // Explicit auto:N names GPU N of the compiled backend.
        DevicePlacement explicit_auto;
        explicit_auto.gpu = 1;
        const DevicePlacement a = resolve_draft_placement(explicit_auto, true, target, false);
        CHECK(a.backend == compiled_placement_backend());
        CHECK(a.gpu == 1);

        // An explicit placement is kept even with an auto-placed target.
        DevicePlacement explicit_hip;
        explicit_hip.backend = compiled_placement_backend();
        explicit_hip.gpu = 0;
        const DevicePlacement b = resolve_draft_placement(explicit_hip, true, target, true);
        CHECK(b.gpu == 0);

        // Unplaced drafters stay on the auto backend so backend defaults
        // (LUCE_DS4_DRAFT_GPU/_BACKEND) still apply; with an auto target the
        // index follows the target.
        const DevicePlacement c = resolve_draft_placement(unplaced, false, target, true);
        CHECK(c.backend == PlacementBackend::Auto);
        CHECK(c.gpu == 1);
        const DevicePlacement d = resolve_draft_placement(unplaced, false, target, false);
        CHECK(d.backend == PlacementBackend::Auto);
        CHECK(d.gpu == 0);
    }

    void test_profiles_are_well_formed() {
        std::set<std::string> names;
        for (const LaunchProfile & profile : launch_profiles()) {
            CHECK(names.insert(profile.name).second);
            CHECK(find_launch_profile(profile.name) == &profile);
            std::set<std::string> env;
            for (const LaunchProfileEnv & entry : profile.env) {
                CHECK(env.insert(entry.name).second);
            }
            for (const LaunchProfileFlag & flag : profile.flags) {
                CHECK(std::strncmp(flag.flag, "--", 2) == 0);
            }
        }
        CHECK(find_launch_profile("missing") == nullptr);
    }

    void test_profile_flags_yield_to_explicit_flags() {
        const LaunchProfile * profile = find_launch_profile("ds4-r9700-strix");
        CHECK(profile != nullptr);

        const std::vector<std::string> all = launch_profile_args(*profile, {});
        CHECK(contains_flag(all, "--expert-device"));
        CHECK(contains_flag(all, "--target-device"));

        const std::vector<std::string> explicit_args = {
            "--max-ctx", "65536", "--target-devices", "hip:0,hip:1"};
        const std::vector<std::string> merged =
            launch_profile_args(*profile, explicit_args);
        CHECK(!contains_flag(merged, "--max-ctx"));
        // --target-devices and --target-device select the same setting.
        CHECK(!contains_flag(merged, "--target-device"));
        CHECK(contains_flag(merged, "--chunk"));
        CHECK(!contains_flag(merged, "65536"));
    }

    void test_profile_replaces_documented_recipe() {
        // The qualified R9700 + Strix recipe in docs/DS4.md, minus the variables
        // that --expert-device and --draft now express as flags.
        const LaunchProfile * profile = find_launch_profile("ds4-r9700-strix");
        CHECK(profile != nullptr);
        for (const LaunchProfileEnv & entry : profile->env) {
            const std::string name = entry.name;
            CHECK(name != "LUCE_DS4_MOE_TP");
            CHECK(name != "LUCE_DS4_MOE_TP_INPROC");
            CHECK(name != "LUCE_DS4_MOE_TP_GPU");
            CHECK(name != "LUCE_DS4_SPEC");
            CHECK(name != "LUCE_DS4_DRAFT");
        }
        CHECK(profile->env.size() == 37);
    }

    void test_ds41_profile_is_the_lucebox_recipe() {
        // The DS41.md recommended launch: three expert owners through flags,
        // the shipped Lucebox files, no environment spelling of the owners.
        const LaunchProfile * profile = find_launch_profile("ds41-lucebox");
        CHECK(profile != nullptr);
        const std::vector<std::string> args = launch_profile_args(*profile, {});
        CHECK(contains_flag(args, "--expert-device"));
        CHECK(contains_flag(args, "--ds4-expert-placement"));
        CHECK(contains_flag(args, "--ds4-router-bias"));
        CHECK(contains_flag(args, "--ds4-protected-experts"));
        // The measured recipe (DS41.md): these values move the numbers.
        const auto value_of = [&](const char * flag) -> std::string {
            for (size_t i = 0; i + 1 < args.size(); ++i) if (args[i] == flag) return args[i + 1];
            return {};
        };
        CHECK(value_of("--max-ctx") == "131072");
        CHECK(value_of("--chunk") == "4096");
        CHECK(value_of("--ds4-prefill") == "dense");
        CHECK(value_of("--expert-device") == "hip:1");
        CHECK(value_of("--ds4-expert-placement") == "share/deepseek41/placement_lucebox.json");
        CHECK(value_of("--ds4-router-bias") == "share/deepseek41/router_bias_lucebox_40x384_f32.bin");
        CHECK(value_of("--ds4-protected-experts") == "share/deepseek41/massive_experts.json");
        bool budget = false;
        for (const LaunchProfileEnv & entry : profile->env) {
            budget = budget || (std::string(entry.name) == "LUCE_EXPERT_BUDGET_MB" &&
                                std::string(entry.value) == "10500");
        }
        CHECK(budget);
        for (const LaunchProfileEnv & entry : profile->env) {
            const std::string name = entry.name;
            CHECK(name != "LUCE_DS4_MOE_TP");
            CHECK(name != "LUCE_DS4_MOE_TP_INPROC");
            CHECK(name != "LUCE_DS4_SPEC");
            CHECK(name != "LUCE_DS4_DRAFT");
        }
    }

    void test_ds41_gorgon_profile_is_the_measured_recipe() {
        // Every expert resident (no SSD tier): the placement and router bias
        // come from the command line, the drafter shares the R9700.
        const LaunchProfile * profile = find_launch_profile("ds41-gorgon");
        CHECK(profile != nullptr);
        const std::vector<std::string> args = launch_profile_args(*profile, {});
        const auto value_of = [&](const char * flag) -> std::string {
            for (size_t i = 0; i + 1 < args.size(); ++i) if (args[i] == flag) return args[i + 1];
            return {};
        };
        CHECK(value_of("--target-device") == "hip:0");
        CHECK(value_of("--expert-device") == "hip:1");
        CHECK(value_of("--draft-device") == "hip:0");
        CHECK(value_of("--ds4-prefill") == "dense");
        CHECK(!contains_flag(args, "--ds4-expert-placement"));
        CHECK(!contains_flag(args, "--ds4-router-bias"));
        const auto env_of = [&](const char * name) -> std::string {
            for (const LaunchProfileEnv & entry : profile->env) {
                if (std::string(entry.name) == name) return entry.value;
            }
            return {};
        };
        CHECK(env_of("LUCE_EXPERT_BUDGET_MB") == "12000");
        CHECK(env_of("LUCE_DS41_DECODER_BOUNDED_REPLAY") == "1");
        CHECK(env_of("LUCE_DS4_PREFILL_PIPELINE") == "2");
        // The drafter swap stays opt-in: cache hits must match a full prefill.
        CHECK(env_of("LUCE_DS4_DRAFT_SWAP").empty());
        CHECK(env_of("GGML_CUDA_MLA_SPLIT_KV_COUNT") == "8");
        // A flag given on the command line replaces the profile's value.
        const std::vector<std::string> given = {"--draft-device", "hip:1"};
        const std::vector<std::string> merged = launch_profile_args(*profile, given);
        CHECK(!contains_flag(merged, "--draft-device"));
    }

    void test_qwen_next_split_profile_fits_its_context() {
        // The split fits the context to the R9700; a number on the command line still wins.
        const LaunchProfile * profile = find_launch_profile("qwen-next-r9700-strix");
        CHECK(profile != nullptr);
        const std::vector<std::string> args = launch_profile_args(*profile, {});
        const auto max_ctx = std::find(args.begin(), args.end(), "--max-ctx");
        CHECK(max_ctx != args.end() && max_ctx + 1 != args.end() && *(max_ctx + 1) == "auto");
        CHECK(!contains_flag(launch_profile_args(*profile, {"--max-ctx", "65536"}), "--max-ctx"));
    }
};

}  // namespace

TEST_CASE(LaunchPolicyFixture, launch_policy_suite) {
    test_auto_device_prefers_discrete_gpu_that_fits();
    test_auto_device_moves_large_model_to_device_that_fits();
    test_auto_device_falls_back_to_largest();
    test_auto_device_keeps_first_of_equal_discrete_gpus();
    test_required_bytes_margin();
    test_auto_device_counts_the_kv_cache();
    test_auto_placement_follows_gpu_availability();
    test_auto_placement_splits_only_what_one_discrete_gpu_cannot_hold();
    test_auto_placement_never_uses_exhausted_devices();
    test_auto_placement_preserves_requested_features();
    test_paged_expert_parallel_requires_supported_devices();
    test_auto_placement_preserves_separate_vision_encoder();
    test_draft_placement_precedence();
    test_profiles_are_well_formed();
    test_profile_flags_yield_to_explicit_flags();
    test_profile_replaces_documented_recipe();
    test_ds41_profile_is_the_lucebox_recipe();
    test_ds41_gorgon_profile_is_the_measured_recipe();
    test_qwen_next_split_profile_fits_its_context();
}
