#include "CppUnitTestFramework.hpp"
#include "../../src/qwen35/spec_rollback_policy.h"

using namespace luce::qwen35;

namespace {
struct SpecRollbackPolicyFixture {};
}

// KVFlash verify: slot-mapped set_rows and per-token DeltaNet capture are
// independent (full-attention layers vs delta layers) and must coexist.
TEST_CASE(SpecRollbackPolicyFixture, kvflash_verify_has_set_rows_and_capture) {
    // 8-token verify under the pager: mask on, window 0, capture on.
    CHECK(use_kv_write_rows(/*paged*/false, /*no_kvpad*/false,
                            /*capture_delta*/true, /*kvflash_mask*/true,
                            /*fa_window*/0, /*n_tokens*/8, /*with_mask*/true,
                            /*capture*/true));
    // Without capture it is unchanged.
    CHECK(use_kv_write_rows(false, false, false, true, 0, 8, true, true));
}

TEST_CASE(SpecRollbackPolicyFixture, non_kvflash_gate_unchanged) {
    // Multi-token capture verify without the pager keeps the legacy cpy append.
    CHECK(!use_kv_write_rows(false, false, true, false, 0, 8, true, true));
    CHECK(!use_kv_write_rows(false, false, false, false, 0, 8, true, true));
    // Single-token plain decode still uses set_rows.
    CHECK(use_kv_write_rows(false, false, false, false, 0, 1, false, false));
    // LUCE_QWEN35_NO_KVPAD and a sliding window still disable it under the pager.
    CHECK(!use_kv_write_rows(false, true, true, true, 0, 8, true, true));
    CHECK(!use_kv_write_rows(false, false, true, true, 4096, 8, true, true));
    // paged attention always wins.
    CHECK(use_kv_write_rows(true, true, true, false, 4096, 8, true, true));
}
