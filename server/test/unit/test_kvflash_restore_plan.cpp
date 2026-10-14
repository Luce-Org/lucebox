// Pure decision logic for restoring a KVFlash snapshot (patch 3/4):
// consume vs full re-prefill vs reject, and the token history a consuming
// restore hands the scorer. No GPU or filesystem state is involved.
#include "CppUnitTestFramework.hpp"
#include "../../src/qwen35/prefill_helpers.h"

#include <cstdlib>
#include <vector>

namespace {
struct KvflashRestorePlanFixture {};
}

using luce::common::KvflashRestorePlan;
using luce::common::kvflash_restore_plan;
using luce::common::kvflash_restore_history;
using luce::common::kvflash_restore_consume_enabled;

TEST_CASE(KvflashRestorePlanFixture, dense_snapshot_is_never_touched) {
    CHECK(kvflash_restore_plan(false, false, 640, 64, true) == KvflashRestorePlan::Dense);
    CHECK(kvflash_restore_plan(true,  false, 641, 64, false) == KvflashRestorePlan::Dense);
}

TEST_CASE(KvflashRestorePlanFixture, pooled_with_consume_off_reprefills_from_zero) {
    CHECK(kvflash_restore_plan(false, true, 640, 64, true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(false, true, 640, 64, false) == KvflashRestorePlan::Reject);
}

TEST_CASE(KvflashRestorePlanFixture, pooled_aligned_consume_uses_the_restored_kv) {
    CHECK(kvflash_restore_plan(true, true, 640, 64, true) == KvflashRestorePlan::Consume);
    CHECK(kvflash_restore_plan(true, true, 640, 64, false) == KvflashRestorePlan::Consume);
}

TEST_CASE(KvflashRestorePlanFixture, misaligned_offset_never_prefills_a_bare_suffix) {
    // Donor bug: kv_offset reset to 0 but only the suffix prefilled. The plan
    // must be a re-prefill of the WHOLE prompt, or a clear failure.
    CHECK(kvflash_restore_plan(true, true, 641, 64, true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(true, true, 641, 64, false) == KvflashRestorePlan::Reject);
}

TEST_CASE(KvflashRestorePlanFixture, degenerate_geometry_falls_back) {
    CHECK(kvflash_restore_plan(true, true, 0, 64, true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(true, true, 640, 0, true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(true, true, 640, 0, false) == KvflashRestorePlan::Reject);
}

TEST_CASE(KvflashRestorePlanFixture, history_cold_prefill_is_just_the_tokens) {
    const std::vector<int32_t> tok = {5, 6, 7};
    const auto h = kvflash_restore_history(tok, 0, nullptr);
    CHECK(h == tok);
}

TEST_CASE(KvflashRestorePlanFixture, history_uses_real_ids_for_the_restored_prefix) {
    const std::vector<int32_t> full = {1, 2, 3, 4, 5, 6};
    const std::vector<int32_t> suffix = {4, 5, 6};
    const auto h = kvflash_restore_history(suffix, 3, &full);
    CHECK(h == full);
}

TEST_CASE(KvflashRestorePlanFixture, history_without_full_prompt_zero_fills_prefix) {
    const std::vector<int32_t> suffix = {4, 5};
    const auto h = kvflash_restore_history(suffix, 3, nullptr);
    CHECK(h == (std::vector<int32_t>{0, 0, 0, 4, 5}));
    // A full prompt shorter than the offset cannot supply the prefix either.
    const std::vector<int32_t> short_full = {1, 2};
    CHECK(kvflash_restore_history(suffix, 3, &short_full) == h);
}

TEST_CASE(KvflashRestorePlanFixture, consume_flag_is_off_unless_exactly_one) {
    unsetenv("LUCE_KVFLASH_RESTORE_CONSUME");
    CHECK(!kvflash_restore_consume_enabled());
    setenv("LUCE_KVFLASH_RESTORE_CONSUME", "0", 1);
    CHECK(!kvflash_restore_consume_enabled());
    setenv("LUCE_KVFLASH_RESTORE_CONSUME", "", 1);
    CHECK(!kvflash_restore_consume_enabled());
    setenv("LUCE_KVFLASH_RESTORE_CONSUME", "1", 1);
    CHECK(kvflash_restore_consume_enabled());
    unsetenv("LUCE_KVFLASH_RESTORE_CONSUME");
}
