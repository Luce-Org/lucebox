// The concurrent engine's penalty history must be what the single-lane path
// passes the sampler: the tokens generated for the request, not the prompt.
// The concurrent slot keeps prompt + generated in sample_history (replay and
// accounting need it); SeqSlot::penalty_history() is what the sampler reads.
#include "CppUnitTestFramework.hpp"
#include "common/concurrency/seq_slot_manager.h"
#include "common/sampler.h"

#include <utility>
#include <vector>

using namespace luce::common;

namespace {
struct PenaltyHistoryFixture {};

constexpr int kVocab = 16;

// Greedy pick after penalties (sampler_distribution with temp 0).
int greedy_after_penalties(const std::vector<float> & logits, const SamplerCfg & cfg,
                           const std::vector<int32_t> & history) {
    std::vector<std::pair<float, int>> dist;
    sampler_distribution(logits.data(), kVocab, cfg, history, dist);
    return dist.size() == 1 ? dist[0].second : -1;
}

SeqSlot slot_with(const std::vector<int32_t> & prompt,
                  const std::vector<int32_t> & generated, const SamplerCfg & cfg) {
    SeqSlot s;
    s.sampler = cfg;
    s.original_prompt_len = (int)prompt.size();
    s.prompt_len = (int)prompt.size();
    s.sample_history = prompt;
    s.sample_history.insert(s.sample_history.end(), generated.begin(), generated.end());
    return s;
}
}  // namespace

TEST_CASE(PenaltyHistoryFixture, prompt_only_token_is_not_presence_penalized) {
    SamplerCfg cfg;
    cfg.pres_pen = 1.5f;                 // the no-thinking card default
    std::vector<float> logits(kVocab, -1.0f);
    logits[5] = 1.2f;                    // in the prompt only
    logits[7] = 1.0f;                    // generated once
    logits[11] = 0.5f;                   // nowhere

    const std::vector<int32_t> prompt = {1, 5, 2, 5, 3};
    const std::vector<int32_t> generated = {7, 9};
    const SeqSlot s = slot_with(prompt, generated, cfg);

    CHECK(s.penalty_history() == generated);   // what single-lane passes
    CHECK(greedy_after_penalties(logits, cfg, s.penalty_history()) == 5);
    // The whole slot history would have penalized the prompt token.
    CHECK(greedy_after_penalties(logits, cfg, s.sample_history) == 11);

    // Once generated, the token is penalized like on the single-lane path.
    const SeqSlot s2 = slot_with(prompt, {7, 5}, cfg);
    std::vector<float> l2(kVocab, 0.0f);
    l2[5] = 1.2f; l2[7] = 1.0f; l2[9] = 0.9f;
    CHECK(greedy_after_penalties(l2, cfg, s2.penalty_history()) == 9);
}

TEST_CASE(PenaltyHistoryFixture, fresh_slot_has_empty_history) {
    SamplerCfg cfg;
    cfg.freq_pen = 1.0f;
    const SeqSlot s = slot_with({4, 4, 4}, {}, cfg);
    CHECK(s.penalty_history().empty());
}

TEST_CASE(PenaltyHistoryFixture, window_matches_the_full_generated_suffix) {
    SamplerCfg cfg;
    cfg.rep_pen = 1.3f;
    cfg.freq_pen = 0.4f;
    cfg.rep_window = 8;
    std::vector<int32_t> prompt(40, 3);
    std::vector<int32_t> generated;
    for (int i = 0; i < 20; i++) generated.push_back(i % kVocab);
    const SeqSlot s = slot_with(prompt, generated, cfg);

    const std::vector<int32_t> h = s.penalty_history();
    CHECK(h.size() == 8);
    CHECK(std::vector<int32_t>(generated.end() - 8, generated.end()) == h);

    std::vector<float> logits(kVocab);
    for (int t = 0; t < kVocab; t++) logits[t] = 0.1f * (float)((t * 7) % kVocab);
    std::vector<std::pair<float, int>> a, b;
    SamplerCfg sampled = cfg;
    sampled.temp = 0.8f;
    sampler_distribution(logits.data(), kVocab, sampled, h, a);
    sampler_distribution(logits.data(), kVocab, sampled, generated, b);
    CHECK(a == b);
}
