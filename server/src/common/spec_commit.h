#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dflash::common {

// Pure decision object for a single linear speculative-decode commit.
//
// Slot 0 is the already-selected seed. Slots 1..N are accepted only while
// they equal the target decision for the preceding slot. The first target
// decision that does not match is an optional bonus token. This class owns no
// model, KV, sampler, or rollback state; callers retain those responsibilities.
class SpecCommitDecision {
public:
    // Greedy verification from target argmax tokens. target_tokens[i] is the
    // target's decision after draft_tokens[i], and therefore approves (or
    // rejects) draft_tokens[i + 1]. The last target row is deliberately not
    // inspected: there is no subsequent draft token to match against it.
    static SpecCommitDecision greedy(
            const int32_t * draft_tokens,
            int draft_count,
            const int32_t * target_tokens,
            int target_count,
            int verify_count,
            int commit_budget) noexcept {
        if (draft_tokens == nullptr || target_tokens == nullptr ||
            verify_count <= 0 || draft_count < verify_count ||
            target_count < verify_count) {
            return {};
        }

        int accepted = 1; // seed
        while (accepted < verify_count &&
               draft_tokens[accepted] == target_tokens[accepted - 1]) {
            ++accepted;
        }

        const int32_t mismatch_token =
            accepted < verify_count ? target_tokens[accepted - 1] : 0;
        const bool has_bonus =
            accepted < verify_count && mismatch_token >= 0;
        const int32_t bonus = has_bonus ? mismatch_token : 0;
        return precomputed(accepted, verify_count, has_bonus, bonus, commit_budget);
    }

    static SpecCommitDecision greedy(
            const std::vector<int32_t> & draft_tokens,
            const std::vector<int32_t> & target_tokens,
            int verify_count,
            int commit_budget) noexcept {
        return greedy(
            draft_tokens.empty() ? nullptr : &draft_tokens[0],
            static_cast<int>(draft_tokens.size()),
            target_tokens.empty() ? nullptr : &target_tokens[0],
            static_cast<int>(target_tokens.size()),
            verify_count, commit_budget);
    }

    // Finalize a match walk performed by a model-specific verifier (for
    // example, sampled verification). The caller supplies the accepted prefix
    // and, when a mismatch occurred, the already-selected target bonus token.
    static SpecCommitDecision precomputed(
            int accepted_count,
            int verify_count,
            bool bonus_available,
            int32_t bonus_token,
            int commit_budget) noexcept {
        SpecCommitDecision result;
        // A full prefix cannot have a bonus. A strict prefix may omit one when
        // the target/sample reports the negative no-token sentinel.
        if (verify_count <= 0 || accepted_count <= 0 ||
            accepted_count > verify_count ||
            (bonus_available && accepted_count >= verify_count) ||
            (bonus_available && bonus_token < 0)) {
            return result;
        }
        result.valid_ = true;
        result.accepted_count_ = accepted_count;
        const int budget = std::max(0, commit_budget);
        result.commit_count_ = std::min(
            budget, result.accepted_count_ + (bonus_available ? 1 : 0));
        result.bonus_available_ = bonus_available;
        result.commits_bonus_ =
            bonus_available && result.commit_count_ > result.accepted_count_;
        result.bonus_token_ = bonus_available ? bonus_token : 0;
        return result;
    }

    int accepted_count() const noexcept { return accepted_count_; }
    int commit_count() const noexcept { return commit_count_; }
    bool valid() const noexcept { return valid_; }
    bool has_bonus() const noexcept { return bonus_available_; }
    bool commits_bonus() const noexcept { return commits_bonus_; }
    int32_t bonus_token() const noexcept { return bonus_token_; }

    // Select a committed token without reading beyond the draft prefix or
    // returning an unavailable/clipped bonus.
    bool token_at(
            int index,
            const int32_t * draft_tokens,
            int draft_count,
            int32_t & token) const noexcept {
        if (index < 0 || index >= commit_count_ || draft_tokens == nullptr) {
            return false;
        }
        if (!valid_) {
            return false;
        }
        if (index < accepted_count_) {
            if (index >= draft_count) {
                return false;
            }
            token = draft_tokens[index];
            if (token < 0) {
                return false;
            }
            return true;
        }
        if (index == accepted_count_ && commits_bonus_) {
            token = bonus_token_;
            return true;
        }
        return false;
    }

    bool token_at(
            int index,
            const std::vector<int32_t> & draft_tokens,
            int32_t & token) const noexcept {
        return token_at(index, draft_tokens.empty() ? nullptr : &draft_tokens[0],
                        static_cast<int>(draft_tokens.size()), token);
    }

    bool materialize(
            const std::vector<int32_t> & draft_tokens,
            std::vector<int32_t> & committed_tokens) const {
        if (!valid_) {
            committed_tokens.clear();
            return false;
        }
        committed_tokens.resize(static_cast<size_t>(commit_count_));
        for (int i = 0; i < commit_count_; ++i) {
            if (!token_at(i, draft_tokens, committed_tokens[static_cast<size_t>(i)])) {
                committed_tokens.clear();
                return false;
            }
        }
        return true;
    }

private:
    int accepted_count_ = 0;
    int commit_count_ = 0;
    bool bonus_available_ = false;
    bool commits_bonus_ = false;
    int32_t bonus_token_ = 0;
    bool valid_ = false;
};

} // namespace dflash::common
