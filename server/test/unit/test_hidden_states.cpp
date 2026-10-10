// /v1/hidden_states: request validation, layer resolution, the "bare"
// generation prompt, chunked pooling, and the response shape.

#include "CppUnitTestFramework.hpp"
#include "common/hidden_states.h"
#include "server/chat_template.h"
#include "server/http_server.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using luce::common::ChatFormat;
using luce::common::ChatMessage;
using luce::common::HiddenStates;
using luce::common::HiddenStatesAccumulator;
using luce::common::HiddenStatesRequest;
using luce::common::HiddenStatesSpec;
using luce::common::build_hidden_states_response;
using luce::common::json;
using luce::common::parse_hidden_states_request;
using luce::common::render_chat_template;
using luce::common::render_chat_template_jinja;
using luce::common::resolve_hidden_states_spec;
using luce::common::trim_to_assistant_header;

namespace {
struct HiddenStatesFixture {};

const std::vector<ChatMessage> kConversation = {
    {"system", "Route the request.", ""},
    {"user", "What is 2+2?", ""},
    {"assistant", "4", ""},
    {"user", "And 3+3?", ""},
};

// A plain ChatML template whose generation prompt adds no think block.
const char * kPlainChatml =
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
}  // namespace

TEST_CASE(HiddenStatesFixture, request_defaults_and_options) {
    const HiddenStatesRequest plain =
        parse_hidden_states_request({{"layers", {3, -1}}});
    CHECK(plain.layers == std::vector<int>({3, -1}));
    CHECK(plain.pooling == HiddenStatesRequest::Pooling::last);
    CHECK(!plain.bare_generation_prompt);
    CHECK(!plain.cache_prefix);

    const HiddenStatesRequest both = parse_hidden_states_request(
        {{"layers", {0}}, {"pooling", "both"}, {"generation_prompt", "bare"}});
    CHECK(both.pooling == HiddenStatesRequest::Pooling::both);
    CHECK(both.wants_mean());
    CHECK(both.bare_generation_prompt);
    CHECK(parse_hidden_states_request(
              {{"layers", {0}}, {"generation_prompt", "template"}})
              .bare_generation_prompt == false);
    CHECK(parse_hidden_states_request(
              {{"layers", {0}}, {"cache_prefix", true}}).cache_prefix);
    CHECK(!parse_hidden_states_request(
               {{"layers", {0}}, {"cache_prefix", nullptr}}).cache_prefix);
}

TEST_CASE(HiddenStatesFixture, request_rejects_invalid_fields) {
    const std::vector<json> invalid = {
        json::object(),
        {{"layers", json::array()}},
        {{"layers", 3}},
        {{"layers", {1.5}}},
        {{"layers", {"3"}}},
        {{"layers", {true}}},
        {{"layers", {2, 2}}},
        {{"layers", {0}}, {"pooling", "max"}},
        {{"layers", {0}}, {"pooling", 1}},
        {{"layers", {0}}, {"generation_prompt", "none"}},
        {{"layers", {0}}, {"cache_prefix", "yes"}},
        {{"layers", {0}}, {"cache_prefix", 1}},
    };
    for (const json & body : invalid) {
        CHECK_THROW(std::invalid_argument,
                    UNUSED_RETURN(parse_hidden_states_request(body)));
    }
}

TEST_CASE(HiddenStatesFixture, layers_resolve_from_the_end_and_stay_in_range) {
    HiddenStatesRequest request;
    request.layers = {0, 23, -1, -24};
    request.pooling = HiddenStatesRequest::Pooling::mean;
    const HiddenStatesSpec spec = resolve_hidden_states_spec(request, 24);
    CHECK(spec.layers == std::vector<int>({0, 23, 23, 0}));
    CHECK(spec.mean);

    for (int layer : {24, -25}) {
        request.layers = {layer};
        CHECK_THROW(std::invalid_argument,
                    UNUSED_RETURN(resolve_hidden_states_spec(request, 24)));
    }
}

TEST_CASE(HiddenStatesFixture, bare_prompt_drops_the_template_think_block) {
    for (bool thinking : {false, true}) {
        std::string rendered = render_chat_template(
            kConversation, ChatFormat::QWEN3, /*add_generation_prompt=*/true,
            thinking);
        // The Qwen template appends a think block either way.
        CHECK(ends_with(rendered, thinking ? "<think>\n" : "</think>\n\n"));
        CHECK(trim_to_assistant_header(rendered));
        CHECK(ends_with(rendered, "<|im_start|>user\nAnd 3+3?<|im_end|>\n"
                                  "<|im_start|>assistant\n"));
        // The earlier assistant turn is history and stays intact.
        CHECK(rendered.find("<|im_start|>assistant\n4<|im_end|>") !=
              std::string::npos);
    }
}

TEST_CASE(HiddenStatesFixture, bare_prompt_keeps_a_template_without_think_block) {
    const std::string rendered = render_chat_template_jinja(
        kPlainChatml, kConversation, "", "", /*add_generation_prompt=*/true);
    CHECK(ends_with(rendered, "And 3+3?<|im_end|>\n<|im_start|>assistant\n"));
    std::string trimmed = rendered;
    CHECK(trim_to_assistant_header(trimmed));
    CHECK(trimmed == rendered);
}

TEST_CASE(HiddenStatesFixture, bare_prompt_needs_an_open_assistant_turn) {
    // Without a generation prompt the last assistant header belongs to a
    // closed turn; trimming there would cut real content.
    std::string closed = render_chat_template_jinja(
        kPlainChatml, kConversation, "", "", /*add_generation_prompt=*/false);
    const std::string before = closed;
    CHECK(!trim_to_assistant_header(closed));
    CHECK(closed == before);
    std::string other = "<|User|>hi<|Assistant|>";
    CHECK(!trim_to_assistant_header(other));
}

TEST_CASE(HiddenStatesFixture, accumulator_pools_over_every_chunk) {
    HiddenStatesSpec spec;
    spec.layers = {5, 9};
    spec.mean = true;
    HiddenStatesAccumulator accum(spec, 2);
    // Chunk 1: three rows; layer entry 0 sums to {3, 6}, entry 1 to {30, 60}.
    const float last_a0[] = {1, 2}, sum_a0[] = {3, 6};
    const float last_a1[] = {10, 20}, sum_a1[] = {30, 60};
    accum.record(0, last_a0, sum_a0);
    accum.record(1, last_a1, sum_a1);
    accum.add_rows(3);
    // Chunk 2: one row, which is the prompt's last token.
    const float last_b0[] = {7, 8}, last_b1[] = {70, 80};
    accum.record(0, last_b0, last_b0);
    accum.record(1, last_b1, last_b1);
    accum.add_rows(1);

    const HiddenStates out = accum.finish();
    CHECK(out.n_tokens == 4);
    CHECK(out.last[0] == std::vector<float>({7, 8}));
    CHECK(out.last[1] == std::vector<float>({70, 80}));
    CHECK(out.mean[0] == std::vector<float>({2.5f, 3.5f}));
    CHECK(out.mean[1] == std::vector<float>({25.0f, 35.0f}));

    spec.mean = false;
    HiddenStatesAccumulator last_only(spec, 2);
    last_only.record(0, last_a0, sum_a0);
    last_only.add_rows(1);
    CHECK(last_only.finish().mean.empty());
}

TEST_CASE(HiddenStatesFixture, response_keys_layers_as_requested) {
    HiddenStatesRequest request;
    request.layers = {-1, 3};
    HiddenStates states;
    states.last = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    states.mean = {{0.5f, 0.5f}, {0.25f, 0.25f}};
    states.n_tokens = 7;

    const json last = build_hidden_states_response(
        "qwen35-2b", request, 24, 2, 7, states, "\n", 12.5);
    CHECK(last.at("model") == "qwen35-2b");
    CHECK(last.at("n_layers") == 24);
    CHECK(last.at("n_embd") == 2);
    CHECK(last.at("prompt_tokens") == 7);
    CHECK(last.at("generation_prompt") == "template");
    CHECK(last.at("last_token") == "\n");
    CHECK(last.at("timings").at("prefill_ms").get<double>() == 12.5);
    CHECK(last.at("layers").size() == 2u);
    CHECK(last.at("layers").at("-1").at("last") == json::array({1.0f, 2.0f}));
    CHECK(last.at("layers").at("3").at("last") == json::array({3.0f, 4.0f}));
    CHECK(!last.at("layers").at("-1").contains("mean"));
    CHECK(!last.contains("cache_prefix"));

    request.pooling = HiddenStatesRequest::Pooling::mean;
    request.bare_generation_prompt = true;
    const json mean = build_hidden_states_response(
        "qwen35-2b", request, 24, 2, 7, states, "\n", 12.5);
    CHECK(mean.at("generation_prompt") == "bare");
    CHECK(mean.at("layers").at("3").at("mean") == json::array({0.25f, 0.25f}));
    CHECK(mean.at("layers").at("3").contains("last"));

    request.cache_prefix = true;
    const json cached = build_hidden_states_response(
        "qwen35-2b", request, 24, 2, 7, states, "\n", 12.5,
        /*cached_prefix_tokens=*/7);
    CHECK(cached.at("cache_prefix").at("cached_tokens") == 7);
}
