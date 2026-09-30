// Logprobs for /v1/chat/completions: the log-softmax/top-k the backends run
// on a logits row, request validation, and the OpenAI response shape.

#include "CppUnitTestFramework.hpp"
#include "common/token_logprobs.h"
#include "server/http_server.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using luce::common::ApiFormat;
using luce::common::TokenLogprob;
using luce::common::TokenLogprobs;
using luce::common::build_openai_logprobs;
using luce::common::compute_token_logprobs;
using luce::common::json;
using luce::common::kMaxTopLogprobs;
using luce::common::parse_request_logprobs;

namespace {
struct TokenLogprobsFixture {};

bool near(float a, double b) { return std::fabs((double)a - b) < 1e-5; }
}  // namespace

TEST_CASE(TokenLogprobsFixture, logprobs_normalize_and_rank_ties_by_id) {
    // Probabilities 1/8, 2/8, 3/8, 2/8.
    const std::vector<float> logits = {
        0.0f, std::log(2.0f), std::log(3.0f), std::log(2.0f)};
    const TokenLogprobs lp = compute_token_logprobs(logits.data(), 4, 0, 3);
    CHECK(lp.chosen.token == 0);
    CHECK(near(lp.chosen.logprob, std::log(1.0 / 8.0)));
    CHECK(lp.top.size() == 3u);
    CHECK(lp.top[0].token == 2);
    CHECK(near(lp.top[0].logprob, std::log(3.0 / 8.0)));
    // Tokens 1 and 3 tie: the lower id ranks first.
    CHECK(lp.top[1].token == 1);
    CHECK(lp.top[2].token == 3);
    CHECK(near(lp.top[2].logprob, std::log(2.0 / 8.0)));
}

TEST_CASE(TokenLogprobsFixture, logprobs_stay_finite_for_large_logits) {
    const std::vector<float> logits = {1000.0f, 1000.0f, -1000.0f};
    const TokenLogprobs lp = compute_token_logprobs(logits.data(), 3, 1, 1);
    CHECK(near(lp.chosen.logprob, -std::log(2.0)));
    CHECK(lp.top.size() == 1u);
    CHECK(lp.top[0].token == 0);
}

TEST_CASE(TokenLogprobsFixture, logprobs_top_n_is_clamped) {
    const std::vector<float> logits(64, 0.0f);
    CHECK(compute_token_logprobs(logits.data(), 64, 0, 0).top.empty());
    CHECK(compute_token_logprobs(logits.data(), 64, 0, -1).top.empty());
    CHECK(compute_token_logprobs(logits.data(), 64, 0, 100).top.size() ==
          (size_t)kMaxTopLogprobs);
    CHECK(compute_token_logprobs(logits.data(), 3, 0, 10).top.size() == 3u);
}

TEST_CASE(TokenLogprobsFixture, request_logprobs_resolve_to_top_n) {
    const ApiFormat chat = ApiFormat::OPENAI_CHAT;
    CHECK(parse_request_logprobs(json::object(), chat) == -1);
    CHECK(parse_request_logprobs({{"logprobs", false}}, chat) == -1);
    CHECK(parse_request_logprobs({{"logprobs", true}}, chat) == 0);
    CHECK(parse_request_logprobs(
              {{"logprobs", true}, {"top_logprobs", 20}}, chat) == 20);
    CHECK(parse_request_logprobs(
              {{"logprobs", true}, {"top_logprobs", nullptr}}, chat) == 0);
}

TEST_CASE(TokenLogprobsFixture, request_logprobs_rejects_invalid_values) {
    const ApiFormat chat = ApiFormat::OPENAI_CHAT;
    const std::vector<json> invalid = {
        {{"logprobs", true}, {"top_logprobs", 21}},
        {{"logprobs", true}, {"top_logprobs", -1}},
        {{"logprobs", true}, {"top_logprobs", "5"}},
        {{"logprobs", 1}},
        {{"top_logprobs", 5}},
        {{"logprobs", false}, {"top_logprobs", 5}},
    };
    for (const json & body : invalid) {
        CHECK_THROW(std::invalid_argument,
                    UNUSED_RETURN(parse_request_logprobs(body, chat)));
    }
    CHECK_THROW(std::invalid_argument,
                UNUSED_RETURN(parse_request_logprobs(
                    {{"logprobs", true}}, ApiFormat::ANTHROPIC)));
}

TEST_CASE(TokenLogprobsFixture, openai_logprobs_keep_exact_bytes_and_valid_json) {
    TokenLogprobs position;
    position.chosen = {7, -0.25f};
    position.top = {
        {7, -0.25f},
        {9, -std::numeric_limits<float>::infinity()},
    };
    // Token 9 is the first two bytes of a three-byte UTF-8 sequence.
    const json out = build_openai_logprobs(
        {position}, [](int32_t id) {
            return id == 7 ? std::string("easy") : std::string("\xE2\x82");
        });

    const json & content = out.at("content");
    CHECK(content.size() == 1u);
    const json & first = content[0];
    CHECK(first.at("token") == "easy");
    CHECK(first.at("logprob").get<float>() == -0.25f);
    CHECK(first.at("bytes") == json::array({101, 97, 115, 121}));
    const json & partial = first.at("top_logprobs")[1];
    CHECK(partial.at("bytes") == json::array({0xE2, 0x82}));
    CHECK(partial.at("token") == "\xEF\xBF\xBD\xEF\xBF\xBD");
    CHECK(partial.at("logprob").get<float>() == -9999.0f);
    CHECK_NO_THROW(UNUSED_RETURN(out.dump()));
}
