#include "../src/common/expert_split_target_config.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void test_parse_target_list() {
    std::vector<ExpertSplitTargetSpec> specs;
    std::string err;
    expect(parse_expert_split_target_list("cuda:0,cuda:1,cpu", specs, &err), err.c_str());
    expect(specs.size() == 3, "target count");
    expect(specs[0].backend == PlacementBackend::Cuda, "target0 backend");
    expect(specs[0].device_id == 0, "target0 device");
    expect(specs[1].backend == PlacementBackend::Cuda, "target1 backend");
    expect(specs[1].device_id == 1, "target1 device");
    expect(specs[2].unlimited, "cpu target unlimited");
}

static void test_parse_capacity_overrides() {
    std::vector<uint64_t> caps;
    std::string err;
    expect(parse_expert_split_capacity_overrides("1024MB,auto,2GB", caps, &err), err.c_str());
    expect(caps.size() == 3, "cap count");
    expect(caps[0] == 1024ULL * 1024ULL * 1024ULL, "cap0 bytes");
    expect(caps[1] == 0, "cap1 auto");
    expect(caps[2] == 2ULL * 1024ULL * 1024ULL * 1024ULL, "cap2 bytes");
}

static void test_build_targets_with_primary_override() {
    std::vector<ExpertSplitTargetSpec> specs;
    std::string err;
    expect(parse_expert_split_target_list("cuda:0,cuda:1,cpu", specs, &err), err.c_str());
    specs[1].auto_capacity = false;
    specs[1].capacity_bytes = 3ULL * 1024ULL * 1024ULL * 1024ULL;

    std::vector<ExpertSplitTarget> targets;
    expect(build_expert_split_targets(specs,
                                      /*primary_capacity_bytes=*/5ULL * 1024ULL * 1024ULL * 1024ULL,
                                      targets, &err),
           err.c_str());
    expect(targets.size() == 3, "built target count");
    expect(targets[0].name == "cuda:0", "built target0 name");
    expect(targets[0].capacity_bytes == 5ULL * 1024ULL * 1024ULL * 1024ULL, "primary cap");
    expect(targets[1].name == "cuda:1", "built target1 name");
    expect(targets[1].capacity_bytes == 3ULL * 1024ULL * 1024ULL * 1024ULL, "secondary cap");
    expect(targets[2].backend == "cpu", "cpu backend");
    expect(targets[2].unlimited, "cpu unlimited");
}

static void test_reject_duplicate_targets() {
    std::vector<ExpertSplitTargetSpec> specs;
    std::string err;
    expect(!parse_expert_split_target_list("cuda:0,cuda:0", specs, &err), "duplicate rejected");
    expect(!err.empty(), "duplicate error");
}

static void test_resolve_targets_from_env() {
    std::string err;
    ::setenv("DFLASH_TEST_EXPERT_TARGETS", "cuda:0,cpu", 1);
    ::setenv("DFLASH_TEST_EXPERT_TARGET_CAPS", "4GB,auto", 1);

    std::vector<ExpertSplitTarget> targets;
    expect(resolve_expert_split_targets_from_env(
               "DFLASH_TEST_EXPERT_TARGETS",
               "DFLASH_TEST_EXPERT_TARGET_CAPS",
               /*primary_capacity_bytes=*/6ULL * 1024ULL * 1024ULL * 1024ULL,
               targets, &err),
           err.c_str());
    expect(targets.size() == 2, "resolved target count");
    expect(targets[0].name == "cuda:0", "resolved target0 name");
    expect(targets[0].capacity_bytes == 6ULL * 1024ULL * 1024ULL * 1024ULL,
           "resolved primary cap");
    expect(targets[1].backend == "cpu", "resolved cpu backend");
    expect(targets[1].unlimited, "resolved cpu unlimited");

    ::unsetenv("DFLASH_TEST_EXPERT_TARGETS");
    ::unsetenv("DFLASH_TEST_EXPERT_TARGET_CAPS");
}

static void test_validate_primary_target() {
    std::vector<ExpertSplitTarget> targets = {
        {"cuda:0", "cuda", 0, 1, 0, false},
        {"cpu", "cpu", -1, 0, 0, true},
    };
    std::string err;
    expect(validate_primary_expert_split_target(
               targets, PlacementBackend::Cuda, 0, &err),
           err.c_str());
    expect(!validate_primary_expert_split_target(
               targets, PlacementBackend::Cuda, 1, &err),
           "primary target mismatch rejected");
}

int main() {
    test_parse_target_list();
    test_parse_capacity_overrides();
    test_build_targets_with_primary_override();
    test_reject_duplicate_targets();
    test_resolve_targets_from_env();
    test_validate_primary_target();
    std::printf("OK\n");
    return 0;
}
