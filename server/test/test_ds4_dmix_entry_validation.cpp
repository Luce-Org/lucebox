// Malformed-sidecar coverage for the dense mix (dmix) entry rules.
//
// The two rules under test were missing while p4mix and gumix already had them:
//
//   mode > 1     the kernels branch on mode != 0, so an unrecognised future mode decoded as
//                adaptive against a codebook that means something else -- wrong output, no
//                diagnostic.
//   duplicates   the loader documents exact-once coverage per tensor, but a second entry for
//                the same (layer, class) silently replaced the first registration: the later
//                codebook won and the earlier one leaked.
//
// Both are content-dependent failures that a well-formed fixture never triggers, which is
// why they survived. The rules live in ds4_dmix_entry_reject_reason so the parser and this
// test share one definition -- the parser reads from a FILE*, and covering each case through
// it would need a hand-forged sidecar on disk per case.

#include "CppUnitTestFramework.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

// C linkage: the rules themselves sit in the loader's anonymous namespace (internal linkage,
// correct for the parser); a thin extern "C" wrapper there is what makes them reachable here
// without a second copy of the rules that could drift.
extern "C" const char * ds4_dmix_entry_reject_reason(
    uint32_t layer, uint32_t cls, uint32_t qtype, uint32_t nslices,
    uint32_t C, uint32_t K, uint8_t mode,
    uint32_t n_layers, bool already_covered);

using namespace CppUnitTestFramework;

namespace {
struct Ds4DmixEntryValidationFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};

constexpr uint32_t Q105 = 105, Q106 = 106;
constexpr uint32_t N_LAYERS = 43;

// A well-formed qtype-106 entry: C=2, K=4. 105 wants K=8.
const char * ok106(uint8_t mode = 1, bool dup = false) {
    return ds4_dmix_entry_reject_reason(0, 0, Q106, 16, 2, 4, mode, N_LAYERS, dup);
}
}  // namespace

TEST_CASE(Ds4DmixEntryValidationFixture, rejects_malformed_dmix_entries) {
    // Baseline: a valid entry is accepted, so the rejections below are attributable.
    CHECK(ok106() == nullptr);
    CHECK(ds4_dmix_entry_reject_reason(42, 4, Q105, 4096, 2, 8, 0, N_LAYERS, false) == nullptr);

    // THE TWO REGRESSIONS.
    CHECK(ok106(/*mode=*/2) != nullptr);
    CHECK(ok106(/*mode=*/255) != nullptr);
    CHECK(ok106(/*mode=*/1, /*dup=*/true) != nullptr);
    // Both valid modes remain accepted -- the bound must not be off by one.
    CHECK(ok106(/*mode=*/0) == nullptr);
    CHECK(ok106(/*mode=*/1) == nullptr);

    // Pre-existing rules, kept covered so the extraction did not drop any of them.
    CHECK(ds4_dmix_entry_reject_reason(N_LAYERS, 0, Q106, 16, 2, 4, 1, N_LAYERS, false) != nullptr);
    CHECK(ds4_dmix_entry_reject_reason(0, 5, Q106, 16, 2, 4, 1, N_LAYERS, false) != nullptr);
    CHECK(ds4_dmix_entry_reject_reason(0, 0, 107, 16, 2, 4, 1, N_LAYERS, false) != nullptr);
    CHECK(ds4_dmix_entry_reject_reason(0, 0, Q105, 16, 2, 4, 1, N_LAYERS, false) != nullptr);
    CHECK(ds4_dmix_entry_reject_reason(0, 0, Q106, 16, 2, 8, 1, N_LAYERS, false) != nullptr);
    CHECK(ds4_dmix_entry_reject_reason(0, 0, Q106, 16, 3, 4, 1, N_LAYERS, false) != nullptr);
    CHECK(ds4_dmix_entry_reject_reason(0, 0, Q106, 0, 2, 4, 1, N_LAYERS, false) != nullptr);
    CHECK(ds4_dmix_entry_reject_reason(0, 0, Q106, 4097, 2, 4, 1, N_LAYERS, false) != nullptr);

}
