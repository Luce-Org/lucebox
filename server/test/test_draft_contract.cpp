#include "draft/draft_contract.h"

using namespace dflash::common;

constexpr DraftProposalShape legacy{16, DraftProposalLayout::SeedThenProposals};
static_assert(legacy.first_proposal_row() == 1);
static_assert(legacy.proposal_count() == 15);
static_assert(legacy.verify_width() == 16);

constexpr DraftProposalShape native{4, DraftProposalLayout::ProposalsOnly};
static_assert(native.first_proposal_row() == 0);
static_assert(native.proposal_count() == 4);
static_assert(native.verify_width() == 5);

constexpr DraftProposalShape anchor_only{1, DraftProposalLayout::SeedThenProposals};
static_assert(anchor_only.proposal_count() == 0);
static_assert(anchor_only.verify_width() == 1);

constexpr DraftProposalShape empty_native{0, DraftProposalLayout::ProposalsOnly};
static_assert(empty_native.proposal_count() == 0);
static_assert(empty_native.verify_width() == 1);

constexpr DraftProposalShape default_shape{};
static_assert(default_shape.layout == DraftProposalLayout::SeedThenProposals);
static_assert(default_shape.proposal_count() == 0);
static_assert(default_shape.verify_width() == 1);

int main() { return 0; }
