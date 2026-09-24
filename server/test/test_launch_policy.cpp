// Unit tests for launch policy: `--target-device auto` selection and
// `--profile` expansion. Both are pure functions over resolved facts, so
// they need no model file or GPU.

#include "CppUnitTestFramework.hpp"
#include "placement/device_select.h"
#include "server/launch_profiles.h"

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
};

}  // namespace

TEST_CASE(LaunchPolicyFixture, launch_policy_suite) {
    test_auto_device_prefers_discrete_gpu_that_fits();
    test_auto_device_moves_large_model_to_device_that_fits();
    test_auto_device_falls_back_to_largest();
    test_auto_device_keeps_first_of_equal_discrete_gpus();
    test_required_bytes_margin();
    test_auto_device_counts_the_kv_cache();
    test_draft_placement_precedence();
    test_profiles_are_well_formed();
    test_profile_flags_yield_to_explicit_flags();
    test_profile_replaces_documented_recipe();
}
