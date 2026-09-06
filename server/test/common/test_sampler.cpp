#include "common/sampler.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cmath>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace dflash::common;

// ═══════════════════════════════════════════════════════════════════════
// Sampler tests (model-independent, CPU-only)
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_sampler_cfg_defaults) {
    SamplerCfg cfg;
    TEST_ASSERT(cfg.temp == 0.0f);
    TEST_ASSERT(cfg.top_p == 1.0f);
    TEST_ASSERT(cfg.top_k == 0);
    TEST_ASSERT(cfg.rep_pen == 1.0f);
    TEST_ASSERT(cfg.rep_window == 256);
    TEST_ASSERT(cfg.seed == 0);
    TEST_ASSERT(cfg.freq_pen == 0.0f);
    TEST_ASSERT(cfg.pres_pen == 0.0f);
}

TEST_CASE(ServerUnitFixture, test_sampler_greedy_argmax) {
    // With temp=0 logic, caller uses argmax. But sample_logits with very
    // low temp should still pick the highest logit token reliably.
    float logits[] = {1.0f, 5.0f, 2.0f, 3.0f, 0.5f};
    SamplerCfg cfg;
    cfg.temp = 0.001f;  // near-zero temp → essentially greedy
    std::vector<int32_t> history;
    std::mt19937_64 rng(42);

    int tok = sample_logits(logits, 5, cfg, history, rng);
    TEST_ASSERT(tok == 1);  // token 1 has logit 5.0 (highest)
}

TEST_CASE(ServerUnitFixture, test_sampler_temperature_affects_distribution) {
    // High temperature should spread probability; verify by sampling many
    // times and checking that non-top tokens appear.
    float logits[] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    SamplerCfg cfg;
    cfg.temp = 2.0f;  // high temp → more uniform
    std::vector<int32_t> history;
    std::mt19937_64 rng(123);

    int counts[5] = {};
    for (int i = 0; i < 1000; i++) {
        int tok = sample_logits(logits, 5, cfg, history, rng);
        TEST_ASSERT(tok >= 0 && tok < 5);
        counts[tok]++;
    }
    // With high temp, non-max tokens should appear frequently
    TEST_ASSERT(counts[0] > 50);  // token 0 should appear sometimes
    TEST_ASSERT(counts[1] > 100); // token 1 still most likely
}

TEST_CASE(ServerUnitFixture, test_sampler_top_p_truncation) {
    // With very low top_p, only the top token(s) should be selected.
    float logits[] = {0.0f, 10.0f, 0.0f, 0.0f, 0.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.top_p = 0.01f;  // very restrictive → only the top token
    std::vector<int32_t> history;
    std::mt19937_64 rng(42);

    for (int i = 0; i < 100; i++) {
        int tok = sample_logits(logits, 5, cfg, history, rng);
        TEST_ASSERT(tok == 1);  // only token 1 should survive top_p
    }
}

TEST_CASE(ServerUnitFixture, test_sampler_top_k_truncation) {
    // top_k=2 should limit candidates to the top 2.
    float logits[] = {1.0f, 5.0f, 3.0f, 0.0f, 0.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.top_k = 2;
    std::vector<int32_t> history;
    std::mt19937_64 rng(42);

    int counts[5] = {};
    for (int i = 0; i < 500; i++) {
        int tok = sample_logits(logits, 5, cfg, history, rng);
        counts[tok]++;
    }
    // Only tokens 1 (logit=5) and 2 (logit=3) should appear
    TEST_ASSERT(counts[0] == 0);
    TEST_ASSERT(counts[3] == 0);
    TEST_ASSERT(counts[4] == 0);
    TEST_ASSERT(counts[1] > 0);
    TEST_ASSERT(counts[2] > 0);
}

TEST_CASE(ServerUnitFixture, test_sampler_repetition_penalty) {
    // Multiplicative rep_pen should reduce probability of repeated tokens.
    float logits[] = {3.0f, 3.0f, 3.0f, 3.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.rep_pen = 2.0f;
    std::vector<int32_t> history = {0, 1};  // tokens 0 and 1 in history
    std::mt19937_64 rng(42);

    int counts[4] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 4, cfg, history, rng);
        counts[tok]++;
    }
    // Tokens 0,1 are penalized → tokens 2,3 should appear more
    TEST_ASSERT(counts[2] + counts[3] > counts[0] + counts[1]);
}

TEST_CASE(ServerUnitFixture, test_sampler_frequency_penalty) {
    // freq_pen subtracts freq_pen * count(token) from logits.
    // Token 0 appears 5 times → logit reduced by 5*1.0 = 5.0
    float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.freq_pen = 1.0f;
    std::vector<int32_t> history = {0, 0, 0, 0, 0, 1};  // token 0 x5, token 1 x1
    std::mt19937_64 rng(42);

    int counts[4] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 4, cfg, history, rng);
        counts[tok]++;
    }
    // Token 0 penalized most (5*1.0=5), token 1 penalized some (1*1.0=1).
    // Tokens 2,3 unpenalized → should dominate.
    TEST_ASSERT(counts[2] + counts[3] > counts[0] + counts[1]);
    // Token 0 should appear less than token 1 (penalized more).
    TEST_ASSERT(counts[0] < counts[1]);
}

TEST_CASE(ServerUnitFixture, test_sampler_presence_penalty) {
    // pres_pen subtracts pres_pen * 1(appeared) from logits.
    float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.pres_pen = 3.0f;
    std::vector<int32_t> history = {0, 1};  // tokens 0,1 appeared
    std::mt19937_64 rng(42);

    int counts[4] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 4, cfg, history, rng);
        counts[tok]++;
    }
    // Tokens 0,1 penalized (logit 5-3=2), tokens 2,3 unpenalized (logit 5).
    TEST_ASSERT(counts[2] + counts[3] > counts[0] + counts[1]);
}

TEST_CASE(ServerUnitFixture, test_sampler_freq_and_pres_combined) {
    // Both penalties applied together.
    float logits[] = {5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.freq_pen = 0.5f;
    cfg.pres_pen = 1.0f;
    // Token 0 appears 4 times: penalty = 0.5*4 + 1.0 = 3.0 → logit=2.0
    // Token 1 appears 1 time:  penalty = 0.5*1 + 1.0 = 1.5 → logit=3.5
    // Token 2 never appeared:  penalty = 0                   → logit=5.0
    std::vector<int32_t> history = {0, 0, 0, 0, 1};
    std::mt19937_64 rng(42);

    int counts[3] = {};
    for (int i = 0; i < 3000; i++) {
        int tok = sample_logits(logits, 3, cfg, history, rng);
        counts[tok]++;
    }
    // Token 2 should appear most, token 0 least.
    TEST_ASSERT(counts[2] > counts[1]);
    TEST_ASSERT(counts[1] > counts[0]);
}

TEST_CASE(ServerUnitFixture, test_sampler_negative_frequency_penalty) {
    // Negative freq_pen should encourage repetition.
    float logits[] = {3.0f, 3.0f, 3.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.freq_pen = -2.0f;
    std::vector<int32_t> history = {0, 0, 0};  // token 0 appears 3x
    std::mt19937_64 rng(42);

    int counts[3] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 3, cfg, history, rng);
        counts[tok]++;
    }
    // Token 0 logit boosted by 6.0 (3*2.0) → should dominate.
    TEST_ASSERT(counts[0] > counts[1]);
    TEST_ASSERT(counts[0] > counts[2]);
}

TEST_CASE(ServerUnitFixture, test_sampler_seed_reproducibility) {
    // Same seed should produce identical sequences.
    float logits[] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    std::vector<int32_t> history;

    std::mt19937_64 rng1(12345);
    std::mt19937_64 rng2(12345);

    for (int i = 0; i < 50; i++) {
        int t1 = sample_logits(logits, 5, cfg, history, rng1);
        int t2 = sample_logits(logits, 5, cfg, history, rng2);
        TEST_ASSERT(t1 == t2);
    }
}

TEST_CASE(ServerUnitFixture, test_sampler_rep_window_limits_scope) {
    // With rep_window=2, only the last 2 history tokens should be penalized.
    float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.pres_pen = 5.0f;
    cfg.rep_window = 2;
    // History: [0, 1, 2, 3] but window=2 → only tokens 2,3 penalized.
    std::vector<int32_t> history = {0, 1, 2, 3};
    std::mt19937_64 rng(42);

    int counts[4] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 4, cfg, history, rng);
        counts[tok]++;
    }
    // Tokens 0,1 should appear much more than 2,3 (which are in-window).
    TEST_ASSERT(counts[0] + counts[1] > counts[2] + counts[3]);
}

TEST_CASE(ServerUnitFixture, test_parse_sampler_token_basic) {
    std::string line = "gen 128 samp=0.7,0.9,40,1.1,42";
    SamplerCfg cfg;
    TEST_ASSERT(parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 128");
    TEST_ASSERT(std::abs(cfg.temp - 0.7f) < 1e-5f);
    TEST_ASSERT(std::abs(cfg.top_p - 0.9f) < 1e-5f);
    TEST_ASSERT(cfg.top_k == 40);
    TEST_ASSERT(std::abs(cfg.rep_pen - 1.1f) < 1e-5f);
    TEST_ASSERT(cfg.seed == 42);
    TEST_ASSERT(cfg.freq_pen == 0.0f);  // not specified → default
    TEST_ASSERT(cfg.pres_pen == 0.0f);
}

TEST_CASE(ServerUnitFixture, test_parse_sampler_token_with_penalties) {
    std::string line = "gen 64 samp=0.5,0.95,20,1.0,0,0.8,1.2";
    SamplerCfg cfg;
    TEST_ASSERT(parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 64");
    TEST_ASSERT(std::abs(cfg.temp - 0.5f) < 1e-5f);
    TEST_ASSERT(std::abs(cfg.top_p - 0.95f) < 1e-5f);
    TEST_ASSERT(cfg.top_k == 20);
    TEST_ASSERT(std::abs(cfg.rep_pen - 1.0f) < 1e-5f);
    TEST_ASSERT(cfg.seed == 0);
    TEST_ASSERT(std::abs(cfg.freq_pen - 0.8f) < 1e-5f);
    TEST_ASSERT(std::abs(cfg.pres_pen - 1.2f) < 1e-5f);
}

TEST_CASE(ServerUnitFixture, test_parse_sampler_token_minimal) {
    // Only temp specified.
    std::string line = "gen 32 samp=0.3";
    SamplerCfg cfg;
    TEST_ASSERT(parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 32");
    TEST_ASSERT(std::abs(cfg.temp - 0.3f) < 1e-5f);
    TEST_ASSERT(cfg.top_p == 1.0f);  // default
    TEST_ASSERT(cfg.top_k == 0);
    TEST_ASSERT(cfg.freq_pen == 0.0f);
    TEST_ASSERT(cfg.pres_pen == 0.0f);
}

TEST_CASE(ServerUnitFixture, test_parse_sampler_token_no_samp) {
    std::string line = "gen 128";
    SamplerCfg cfg;
    TEST_ASSERT(!parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 128");  // unchanged
}

TEST_CASE(ServerUnitFixture, test_sampler_temp_zero_with_penalties_uses_argmax) {
    // temp=0 + penalties should apply penalties then return argmax (deterministic).
    float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 0.0f;
    cfg.pres_pen = 3.0f;
    std::vector<int32_t> history = {0, 1};  // penalize tokens 0,1
    std::mt19937_64 rng(42);

    // Tokens 0,1 have logit 5-3=2; tokens 2,3 have logit 5 (unpenalized).
    // Argmax should always return 2 or 3 (whichever sorts first = stable).
    int tok = sample_logits(logits, 4, cfg, history, rng);
    TEST_ASSERT(tok == 2 || tok == 3);

    // Must be deterministic: same result every time.
    for (int i = 0; i < 10; i++) {
        int t = sample_logits(logits, 4, cfg, history, rng);
        TEST_ASSERT(t == tok);
    }
}

TEST_CASE(ServerUnitFixture, test_sampler_needs_logit_processing) {
    SamplerCfg cfg;
    TEST_ASSERT(!cfg.needs_logit_processing());  // all defaults → no processing

    cfg.temp = 0.5f;
    TEST_ASSERT(cfg.needs_logit_processing());

    cfg.temp = 0.0f;
    cfg.freq_pen = 1.0f;
    TEST_ASSERT(cfg.needs_logit_processing());

    cfg.freq_pen = 0.0f;
    cfg.pres_pen = 0.5f;
    TEST_ASSERT(cfg.needs_logit_processing());

    cfg.pres_pen = 0.0f;
    cfg.rep_pen = 1.5f;
    TEST_ASSERT(cfg.needs_logit_processing());

    cfg.rep_pen = 1.0f;
    TEST_ASSERT(!cfg.needs_logit_processing());
}
