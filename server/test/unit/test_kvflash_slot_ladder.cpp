// C1.3-04: the deterministic prefill cuts (D1) the rung ladder sits on. The
// slab capture/restore itself is device code (box test); here: the cut rule.
#include "CppUnitTestFramework.hpp"
#include "kvflash_test_seams.h"
#include "../../src/common/kvflash_slot_ladder.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotLadderFixture {};
}  // namespace

// ── D1: cuts depend only on position and U ───────────────────────────────

TEST_CASE(KvflashSlotLadderFixture, cuts_are_multiples_of_u_inside_the_range_and_the_end_closes_it) {
    CHECK(kvflash_cut_sequence(0, 1300, 512) == (std::vector<int>{512, 1024, 1300}));
    CHECK(kvflash_cut_sequence(100, 1300, 512) == (std::vector<int>{512, 1024, 1300}));   // start offset only shortens the first batch
    CHECK(kvflash_cut_sequence(512, 1024, 512) == (std::vector<int>{1024}));
    CHECK(kvflash_cut_sequence(512, 600, 512) == (std::vector<int>{600}));
    CHECK(kvflash_cut_sequence(7, 7, 512).empty());
    CHECK(kvflash_cut_sequence(9, 7, 512).empty());
    CHECK(kvflash_next_cut(0, 100000, 512) == 512);
    CHECK(kvflash_next_cut(511, 100000, 512) == 512);
    CHECK(kvflash_next_cut(512, 100000, 512) == 1024);
    CHECK(kvflash_next_cut(1000, 1010, 512) == 1010);
}

TEST_CASE(KvflashSlotLadderFixture, resuming_at_any_cut_gives_the_same_remaining_cuts) {
    // Idle, mixed and cancelled steps all resume at a recorded boundary: the
    // continuation must be the suffix of the uninterrupted sequence.
    for (int start : {0, 1, 300, 512, 777}) {
        const auto full = kvflash_cut_sequence(start, 4000, 512);
        for (size_t i = 0; i < full.size(); i++) {
            const auto rest = kvflash_cut_sequence(full[i], 4000, 512);
            CHECK(rest == std::vector<int>(full.begin() + (long)i + 1, full.end()));
        }
        for (int c : full) if (c != 4000) CHECK(c % 512 == 0);
    }
}
