#include "common/concurrency/chain_spec_shapes.h"
#include "host_check.h"

#include <cstdio>
#include <vector>

using namespace luce::common;

static int g_checks = 0;

int main() {
    const std::vector<int32_t> draft = {10, 11, 12, 13};
    const int32_t full_posterior[] = {11, 12, 13, 14};
    CHECK(chain_verified_prefix(
        draft, full_posterior, 4) == draft.size());

    const int32_t rejected_posterior[] = {11, 99, 13, 14};
    CHECK(chain_verified_prefix(
        draft, rejected_posterior, 4) == 2);

    const int32_t root_rejected_posterior[] = {99, 12, 13, 14};
    CHECK(chain_verified_prefix(
        draft, root_rejected_posterior, 4) == 1);

    CHECK(chain_verified_prefix(
        draft, full_posterior, 0) == 1);
    CHECK(chain_verified_prefix(
        std::vector<int32_t>{}, full_posterior, 4) == 0);

    const std::vector<int> bucket_inputs = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17,
    };
    const std::vector<int> bucket_expected = {
        1, 2, 3, 4, 6, 6, 8, 8, 12, 12, 16, 16, 24,
    };
    for (size_t i = 0; i < bucket_inputs.size(); ++i) {
        CHECK(chain_decode_bucket_width(bucket_inputs[i]) ==
              bucket_expected[i]);
    }
    CHECK(chain_draft_bucket_width(4) == 4);
    CHECK(chain_draft_bucket_width(5) == 5);
    CHECK(chain_draft_bucket_width(6) == 6);
    CHECK(chain_draft_bucket_width(7) == 8);

    const auto eos = [](int32_t token) { return token == 2; };
    const std::vector<int32_t> eos_first_child = {10, 2, 11};
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child.data(), eos_first_child.size(), 0, 3, eos) == 1);
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child.data(), eos_first_child.size(), 2, 3, eos) == 2);
    const std::vector<int32_t> eos_second_child = {10, 11, 2, 12};
    CHECK(chain_min_tokens_safe_prefix(
        eos_second_child.data(), eos_second_child.size(), 0, 3, eos) == 2);
    CHECK(chain_min_tokens_safe_prefix(
        eos_second_child.data(), eos_second_child.size(), 1, 3, eos) == 3);
    const std::vector<int32_t> eos_root = {2, 11, 12};
    CHECK(chain_min_tokens_safe_prefix(
        eos_root.data(), eos_root.size(), 0, 3, eos) == eos_root.size());
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child.data(), eos_first_child.size(), 0, 0, eos) == 2);

    // C-2: a fixed chain is ready only when every piece the verify graph reads exists.
    CHECK(chain_config_ready(true, 16, 16, 16, true));
    CHECK(chain_config_ready(true, 8, 8, 16, true));
    CHECK(!chain_config_ready(false, 16, 16, 16, true));    // an engine built without a chain config (the old KVFlash ctor)
    CHECK(!chain_config_ready(true, 1, 1, 16, true));       // width must be > 1
    CHECK(!chain_config_ready(true, 17, 17, 32, true));     // and <= 16
    CHECK(!chain_config_ready(true, 16, 8, 16, true));      // the verify width is the drafter's block
    CHECK(!chain_config_ready(true, 16, 16, 15, true));     // a slot's scratch stride must hold a whole chain
    CHECK(!chain_config_ready(true, 16, 16, 16, false));    // no target_feat ring, no chain

    // C-O2: a request that asks for AR decode (the oracle's T2) never gets a speculative burst.
    CHECK(step_allows_speculation(true, false, false));
    CHECK(!step_allows_speculation(false, false, false));   // budget not settled
    CHECK(!step_allows_speculation(true, true, false));     // a grammar masks one position at a time
    CHECK(!step_allows_speculation(true, false, true));     // force_ar_decode

    // The pool path (no KVFlash) never sizes the per-slot probe vector: reading it must give the default, not null-deref.
    CHECK(slot_value_or(std::vector<int>{}, 0, 0) == 0);
    CHECK(slot_value_or(std::vector<int>{5, 6}, 1, 0) == 6);
    CHECK(slot_value_or(std::vector<int>{5, 6}, 2, 0) == 0);
    CHECK(slot_value_or(std::vector<int>{5, 6}, -1, 7) == 7);

    // P1-1: the accepted prefix never passes the request's output budget. Prompt 100, max_tokens 8, 6 generated tokens
    // fed (committed 106), pending token 7: an 8-token accept may commit only through token 8.
    CHECK(chain_budget_cap(8, 8, 6) == 2);
    CHECK(chain_budget_cap(1, 8, 6) == 1);
    CHECK(chain_budget_cap(16, 192, 0) == 16);                // plenty left: unchanged
    CHECK(chain_budget_cap(16, 8, 7) == 1);                   // last token: the root alone
    CHECK(chain_budget_cap(16, 8, 8) == 1);                   // never below the root (the scheduler has stopped it already)
    CHECK(chain_budget_cap(5, 1 << 30, 100) == 5);            // no bound given (image / prefix-store admits)

    // P2-3: the drafter window starts at the later of its ring window and the first feature row this session owns.
    CHECK(draft_window_lo(5000, 2048, 0) == 5000 - 2048);
    CHECK(draft_window_lo(100, 2048, 0) == 0);
    CHECK(draft_window_lo(1200, 2048, 1156) == 1156);         // resumed at L=1156: rows below it are another occupant's
    CHECK(draft_window_lo(5000, 2048, 1156) == 5000 - 2048);  // the ring window is the tighter bound
    CHECK(draft_window_lo(1000, 2048, 1156) == 1000);         // a start past the committed length is clipped to it
    // rewind from 3000 to 500 in a 2048-row ring: rows 952..2999 survive, so below 500 only rows >= 952 are valid: none.
    CHECK(feature_valid_after_rewind(0, 3000, 500, 2048) == 500);
    CHECK(feature_valid_after_rewind(0, 2000, 500, 2048) == 0);          // nothing was overwritten
    CHECK(feature_valid_after_rewind(300, 2000, 500, 2048) == 300);      // a resumed session keeps its own start
    CHECK(feature_valid_after_rewind(0, 2600, 1000, 2048) == 552);       // rows 552..999 survive

    std::printf("chain spec shape tests passed: %d checks\n", g_checks);
    return 0;
}
