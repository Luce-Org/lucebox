#pragma once

namespace dflash::common {

// Draft artifacts expose one of two row layouts. Existing DFlash and DS4
// DSpark artifacts include the accepted seed in row zero; proposal-only
// artifacts such as Bonsai rely on the runtime to prepend the accepted anchor
// before target verification.
enum class DraftProposalLayout {
    SeedThenProposals,
    ProposalsOnly,
};

struct DraftProposalShape {
    int draft_rows = 0;
    DraftProposalLayout layout = DraftProposalLayout::SeedThenProposals;

    constexpr int first_proposal_row() const noexcept {
        return layout == DraftProposalLayout::ProposalsOnly ? 0 : 1;
    }

    constexpr int proposal_count() const noexcept {
        const int count = draft_rows - first_proposal_row();
        return count > 0 ? count : 0;
    }

    // Verification always includes the already accepted anchor exactly once.
    constexpr int verify_width() const noexcept { return 1 + proposal_count(); }
};

}  // namespace dflash::common
