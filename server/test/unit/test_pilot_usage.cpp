// usage.luce (server/pilot_usage.h): output ids and decode step counts in the
// OpenAI chat response, streamed and not, only under LUCE_PILOT_USAGE.
// The switch is read once per process; ctest runs this fixture a second time
// with LUCE_PILOT_USAGE=1 (server_unit_pilot_usage).

#include "CppUnitTestFramework.hpp"

#include "common/model_backend.h"
#include "common/pilot_switch.h"
#include "server/api_adapters.h"
#include "server/pilot_usage.h"
#include "server/sse_emitter.h"

#include <cstdlib>
#include <string>
#include <vector>

using namespace luce::common;

namespace {

struct PilotUsageFixture {};

DecodeStepCounts spec_counts() {
    DecodeStepCounts counts;
    counts.reported = true;
    counts.steps = 3;
    counts.draft_tokens = 24;
    counts.accepted_tokens = 9;
    counts.target_forwards = 4;
    return counts;
}

GenerateResult counted_result() {
    GenerateResult result;
    result.succeed();
    result.tokens = {11, 12, 13, 14, 15, 16, 17, 18, 19};
    result.decode_counts = spec_counts();
    result.spec_decode_ran = true;
    result.accept_rate = 9.0f / 24.0f;
    return result;
}

bool usage_switch_in_environment() {
    return luce::pilot::parse_switch(std::getenv(kPilotUsageSwitch)) ==
           luce::pilot::SwitchState::On;
}

// The usage object of the stream's usage chunk.
json streamed_usage(const std::vector<std::string> & chunks) {
    for (const std::string & chunk : chunks) {
        if (chunk.rfind("data: {", 0) != 0) continue;
        const json body = json::parse(chunk.substr(6));
        if (body.contains("usage")) return body["usage"];
    }
    return json();
}

const json kExpectedLuce = {
    {"output_ids", {11, 12, 13, 14, 15, 16, 17, 18, 19}},
    {"steps", 3},
    {"draft_tokens", 24},
    {"accepted_tokens", 9},
    {"target_forwards", 4},
};

}  // namespace

TEST_CASE(PilotUsageFixture, json_carries_ids_and_counts) {
    const GenerateResult result = counted_result();
    const json luce = pilot_usage_json(result.tokens, result.decode_counts);
    CHECK_TRUE(luce == kExpectedLuce);
    // nlohmann::json sorts the keys.
    CHECK_EQUAL(luce.dump(), std::string(
        "{\"accepted_tokens\":9,\"draft_tokens\":24,\"output_ids\":[11,12,13,14,15,16,17,18,19],"
        "\"steps\":3,\"target_forwards\":4}"));
}

TEST_CASE(PilotUsageFixture, json_nulls_counts_a_backend_did_not_report) {
    const json luce = pilot_usage_json({5, 6}, DecodeStepCounts{});
    CHECK_TRUE(luce["output_ids"] == json::array({5, 6}));
    CHECK_TRUE(luce["steps"].is_null());
    CHECK_TRUE(luce["draft_tokens"].is_null());
    CHECK_TRUE(luce["accepted_tokens"].is_null());
    CHECK_TRUE(luce["target_forwards"].is_null());
}

// Plain decode: every token after the first is one step and one forward.
TEST_CASE(PilotUsageFixture, json_plain_decode_counts) {
    DecodeStepCounts counts;
    counts.reported = true;
    counts.steps = 4;
    counts.target_forwards = 4;
    const json luce = pilot_usage_json({1, 2, 3, 4, 5}, counts);
    CHECK_EQUAL(luce["steps"].get<int>(), 4);
    CHECK_EQUAL(luce["target_forwards"].get<int>(), 4);
    CHECK_EQUAL(luce["draft_tokens"].get<int>(), 0);
    CHECK_EQUAL(luce["accepted_tokens"].get<int>(), 0);
}

TEST_CASE(PilotUsageFixture, attach_only_when_enabled) {
    const GenerateResult result = counted_result();
    json off = {{"prompt_tokens", 1}};
    attach_pilot_usage(off, false, result.tokens, result.decode_counts);
    CHECK_TRUE(off == json({{"prompt_tokens", 1}}));
    json on = {{"prompt_tokens", 1}};
    attach_pilot_usage(on, true, result.tokens, result.decode_counts);
    CHECK_TRUE(on["luce"] == kExpectedLuce);
    CHECK_EQUAL(on["prompt_tokens"].get<int>(), 1);
}

// The empty-output retry keeps the retry's tokens and counts both attempts.
TEST_CASE(PilotUsageFixture, retry_merge_sums_counts) {
    GenerateResult first = counted_result();
    first.tokens.clear();
    GenerateResult retry;
    retry.succeed();
    retry.tokens = {7, 8, 9};
    retry.decode_counts.reported = true;
    retry.decode_counts.steps = 2;
    retry.decode_counts.target_forwards = 2;
    const GenerateResult merged = ModelBackend::merge_empty_spec_retry_result(first, retry);
    CHECK_TRUE(merged.tokens == std::vector<int32_t>({7, 8, 9}));
    CHECK_TRUE(merged.decode_counts.reported);
    CHECK_EQUAL(merged.decode_counts.steps, 5);
    CHECK_EQUAL(merged.decode_counts.draft_tokens, 24);
    CHECK_EQUAL(merged.decode_counts.accepted_tokens, 9);
    CHECK_EQUAL(merged.decode_counts.target_forwards, 6);
}

TEST_CASE(PilotUsageFixture, stream_usage_chunk_carries_luce_only_when_set) {
    const GenerateResult result = counted_result();
    {
        SseEmitter emitter(ApiFormat::OPENAI_CHAT, "chatcmpl-pilot", "m", 10,
                           json::array(), nullptr);
        emitter.emit_start();
        emitter.emit_token("hi");
        const json usage = streamed_usage(emitter.emit_finish(9));
        REQUIRE_TRUE(usage.is_object());
        CHECK_FALSE(usage.contains("luce"));
        CHECK_EQUAL(usage["completion_tokens"].get<int>(), 9);
    }
    {
        SseEmitter emitter(ApiFormat::OPENAI_CHAT, "chatcmpl-pilot", "m", 10,
                           json::array(), nullptr);
        emitter.set_luce_usage(pilot_usage_json(result.tokens, result.decode_counts));
        emitter.emit_start();
        emitter.emit_token("hi");
        const json usage = streamed_usage(emitter.emit_finish(9));
        REQUIRE_TRUE(usage.is_object());
        REQUIRE_TRUE(usage.contains("luce"));
        CHECK_TRUE(usage["luce"] == kExpectedLuce);
        CHECK_EQUAL(usage["completion_tokens"].get<int>(), 9);
    }
}

TEST_CASE(PilotUsageFixture, switch_matches_environment) {
    CHECK_EQUAL(pilot_usage_enabled(), usage_switch_in_environment());
}

// The non-streaming OpenAI response: usage.luce exists exactly when the
// switch is on; the rest of usage is the same either way.
TEST_CASE(PilotUsageFixture, non_streaming_response_follows_switch) {
    const GenerateResult result = counted_result();
    ParsedRequest req;
    req.format = ApiFormat::OPENAI_CHAT;
    req.response_id = "chatcmpl-pilot";
    req.model = "m";
    SseEmitter emitter(ApiFormat::OPENAI_CHAT, req.response_id, req.model, 10,
                       json::array(), nullptr);
    CompletionTokenCounts counts;
    counts.total = (int) result.tokens.size();
    counts.content = counts.total;
    const json response = build_non_streaming_response(
        req, result, /*generation_cap=*/64, GenTimings{}, counts, emitter);
    REQUIRE_TRUE(response.contains("usage"));
    const json & usage = response["usage"];
    CHECK_EQUAL(usage["completion_tokens"].get<int>(), 9);
    CHECK_TRUE(usage["spec_decode_ran"].get<bool>());
    if (usage_switch_in_environment()) {
        REQUIRE_TRUE(usage.contains("luce"));
        CHECK_TRUE(usage["luce"] == kExpectedLuce);
    } else {
        CHECK_FALSE(usage.contains("luce"));
    }
}

// Other API formats never carry the field.
TEST_CASE(PilotUsageFixture, anthropic_response_has_no_luce) {
    const GenerateResult result = counted_result();
    ParsedRequest req;
    req.format = ApiFormat::ANTHROPIC;
    req.response_id = "msg-pilot";
    req.model = "m";
    SseEmitter emitter(ApiFormat::ANTHROPIC, req.response_id, req.model, 10,
                       json::array(), nullptr);
    CompletionTokenCounts counts;
    counts.total = (int) result.tokens.size();
    const json response = build_non_streaming_response(
        req, result, /*generation_cap=*/64, GenTimings{}, counts, emitter);
    REQUIRE_TRUE(response.contains("usage"));
    CHECK_FALSE(response["usage"].contains("luce"));
}
