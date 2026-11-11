// Structured output: GBNF matching, JSON schema grammars, the per-request
// constraint and the sampler that honours it. Pure logic: no model or GPU.
#include "CppUnitTestFramework.hpp"

#include "common/gbnf.h"
#include "common/output_constraint.h"
#include "common/sampler.h"
#include "server/http_server.h"
#include "server/json_schema_grammar.h"
#include "server/tool_parser.h"

#include <cmath>
#include <random>
#include <string>
#include <vector>

using namespace luce::common;

namespace {

struct OutputConstraintFixture {};

std::unique_ptr<gbnf::Matcher> compile(const std::string & grammar) {
    std::string error;
    auto matcher = gbnf::Matcher::compile(grammar, "root", error);
    if (!matcher) std::fprintf(stderr, "grammar error: %s\n", error.c_str());
    return matcher;
}

bool matches(const std::string & grammar, const std::string & text) {
    auto matcher = compile(grammar);
    if (!matcher) return false;
    matcher->accept_bytes(text);
    return !matcher->dead() && matcher->can_end();
}

// A toy vocabulary: a fenced-JSON habit the grammar must break, the JSON
// pieces it should pick instead, a tool-call opener and an end token.
enum Tok : int32_t {
    FENCE, OBJ_OPEN, OBJ_CLOSE, KEY_A, COLON, ONE, NEWLINE, TOOL_CALL, THINK_END,
    PROSE, NUL_BYTE, END, N_TOK,
};

std::shared_ptr<const ConstraintVocab> toy_vocab() {
    std::vector<std::string> pieces(N_TOK);
    pieces[FENCE] = "```json";
    pieces[OBJ_OPEN] = "{";
    pieces[OBJ_CLOSE] = "}";
    pieces[KEY_A] = "\"a\"";
    pieces[COLON] = ":";
    pieces[ONE] = "1";
    pieces[NEWLINE] = "\n";
    pieces[TOOL_CALL] = "<tool_call>";
    pieces[THINK_END] = "</think>";
    pieces[PROSE] = "Sure";
    pieces[NUL_BYTE] = std::string(1, '\0');
    pieces[END] = "<|im_end|>";
    return ConstraintVocab::build(std::move(pieces), {END});
}

const char * kSchemaA = R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false})";

std::shared_ptr<OutputConstraint> schema_constraint(OutputConstraint::Options options = {},
                                                    std::vector<std::string> openers = {}) {
    const auto schema = nlohmann::ordered_json::parse(kSchemaA);
    auto matcher = compile(http_detail::output_constraint_grammar(schema, openers));
    options.release_prefixes = std::move(openers);
    return std::make_shared<OutputConstraint>(toy_vocab(), std::move(matcher), std::move(options));
}

// Logits that prefer `best`, then everything else equally.
std::vector<float> prefer(int32_t best) {
    std::vector<float> logits(N_TOK, 0.0f);
    logits[(size_t) best] = 10.0f;
    return logits;
}

}  // namespace

TEST_CASE(OutputConstraintFixture, gbnf_matches_literals_ranges_and_repetition) {
    const std::string grammar = "root ::= \"x\" [0-9]{1,3} (\",\" [a-z]+)*\n";
    CHECK(matches(grammar, "x1"));
    CHECK(matches(grammar, "x123,ab,c"));
    CHECK(!matches(grammar, "x1234"));
    CHECK(!matches(grammar, "y1"));
    CHECK(!matches(grammar, "x"));  // a prefix, not a sentence
}

TEST_CASE(OutputConstraintFixture, gbnf_reports_parse_errors) {
    std::string error;
    CHECK(gbnf::Matcher::compile("root ::= missing\n", "root", error) == nullptr);
    CHECK(!error.empty());
    CHECK(gbnf::Matcher::compile("start ::= \"a\"\n", "root", error) == nullptr);
    CHECK(gbnf::Matcher::compile("root ::= root \"a\"\n", "root", error) == nullptr);
}

TEST_CASE(OutputConstraintFixture, gbnf_follows_utf8_split_across_pieces) {
    auto matcher = compile("root ::= \"é\" \"!\"\n");
    CHECK(matcher != nullptr);
    matcher->accept_bytes("\xC3");     // first byte of é
    CHECK(!matcher->dead());
    matcher->accept_bytes("\xA9!");    // its continuation, then !
    CHECK(!matcher->dead());
    CHECK(matcher->can_end());
}

TEST_CASE(OutputConstraintFixture, schema_grammar_accepts_documents_and_rejects_others) {
    const auto schema = nlohmann::ordered_json::parse(kSchemaA);
    const std::string grammar = json_schema_to_grammar(schema);
    CHECK(matches(grammar, "{\"a\": 1}"));
    CHECK(matches(grammar, "{\n  \"a\": 42\n}"));
    CHECK(!matches(grammar, "{\"a\": \"one\"}"));          // wrong type
    CHECK(!matches(grammar, "{\"a\": 1, \"b\": 2}"));      // additionalProperties
    CHECK(!matches(grammar, "```json\n{\"a\": 1}\n```"));  // fenced
}

TEST_CASE(OutputConstraintFixture, schema_grammar_keeps_property_order_and_enums) {
    const auto schema = nlohmann::ordered_json::parse(R"({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "genre": {"type": "string", "enum": ["drama", "sci-fi"]}
        },
        "required": ["title", "genre"]
    })");
    const std::string grammar = json_schema_to_grammar(schema);
    CHECK(matches(grammar, R"({"title": "Alien", "genre": "sci-fi"})"));
    CHECK(!matches(grammar, R"({"title": "Alien", "genre": "horror"})"));
    CHECK(!matches(grammar, R"({"genre": "sci-fi", "title": "Alien"})"));
}

TEST_CASE(OutputConstraintFixture, sampler_redraws_a_rejected_token_from_the_allowed_ones) {
    auto constraint = schema_constraint();
    SamplerCfg cfg;
    cfg.constraint = constraint;
    CHECK(cfg.needs_logit_processing());
    std::mt19937_64 rng(1);
    const std::vector<int32_t> history;

    // Greedy: the model wants a code fence, the grammar wants an object.
    auto logits = prefer(FENCE);
    CHECK(sample_logits(logits.data(), N_TOK, cfg, history, rng) == OBJ_OPEN);

    // A token the grammar already allows is kept as drawn.
    logits = prefer(OBJ_OPEN);
    const int32_t first = sample_logits(logits.data(), N_TOK, cfg, history, rng);
    CHECK(first == OBJ_OPEN);
    constraint->accept(first);

    // Sampled at a temperature, the draw still lands on an allowed token.
    cfg.temp = 1.0f;
    for (int i = 0; i < 50; ++i) {
        logits = prefer(PROSE);
        const int32_t token = sample_logits(logits.data(), N_TOK, cfg, history, rng);
        CHECK(token == KEY_A || token == NEWLINE);
    }
}

TEST_CASE(OutputConstraintFixture, nul_byte_token_is_never_allowed) {
    // Text is read up to a NUL, so this token would match "nothing" and slip
    // into a string (seen live: "answer_cent\0s").
    auto constraint = schema_constraint();
    CHECK(!constraint->allows(NUL_BYTE));
    constraint->accept(OBJ_OPEN);
    constraint->accept(KEY_A);
    CHECK(!constraint->allows(NUL_BYTE));
    auto logits = prefer(NUL_BYTE);
    constraint->mask(logits.data(), N_TOK);
    CHECK(logits[NUL_BYTE] == -INFINITY);
}

TEST_CASE(OutputConstraintFixture, redraw_after_a_fence_writes_the_reply_not_a_tool_call) {
    // The model wants a code fence and would call a tool before opening the
    // object; the fence says it is writing the reply.
    auto constraint = schema_constraint({}, {"<tool_call>"});
    SamplerCfg cfg;
    cfg.constraint = constraint;
    std::mt19937_64 rng(1);
    auto logits = prefer(FENCE);
    logits[TOOL_CALL] = 9.0f;
    logits[OBJ_OPEN] = 8.0f;
    CHECK(sample_logits(logits.data(), N_TOK, cfg, {}, rng) == OBJ_OPEN);
    // A tool call the model draws itself is kept.
    logits = prefer(TOOL_CALL);
    CHECK(sample_logits(logits.data(), N_TOK, cfg, {}, rng) == TOOL_CALL);
}

TEST_CASE(OutputConstraintFixture, redraw_at_the_start_skips_blank_tokens) {
    // A newline would only postpone the choice; after it the model may well
    // drift into the tool call the fence said it did not want.
    auto constraint = schema_constraint({}, {"<tool_call>"});
    SamplerCfg cfg;
    cfg.constraint = constraint;
    std::mt19937_64 rng(1);
    auto logits = prefer(FENCE);
    logits[NEWLINE] = 9.0f;
    logits[OBJ_OPEN] = 8.0f;
    CHECK(sample_logits(logits.data(), N_TOK, cfg, {}, rng) == OBJ_OPEN);
    // Drawn by the model itself, a leading newline is kept.
    logits = prefer(NEWLINE);
    CHECK(sample_logits(logits.data(), N_TOK, cfg, {}, rng) == NEWLINE);
}

TEST_CASE(OutputConstraintFixture, forced_tool_call_redraws_onto_the_opener) {
    auto matcher = compile(http_detail::output_constraint_grammar(nullptr, {"<tool_call>"}));
    OutputConstraint::Options options;
    options.release_prefixes = {"<tool_call>"};
    options.calls_only = true;
    SamplerCfg cfg;
    cfg.constraint = std::make_shared<OutputConstraint>(toy_vocab(), std::move(matcher), options);
    std::mt19937_64 rng(1);
    auto logits = prefer(PROSE);
    CHECK(sample_logits(logits.data(), N_TOK, cfg, {}, rng) == TOOL_CALL);
}

TEST_CASE(OutputConstraintFixture, end_token_only_once_the_document_is_complete) {
    auto constraint = schema_constraint();
    for (int32_t token : {OBJ_OPEN, KEY_A, COLON, ONE}) {
        CHECK(!constraint->allows(END));
        CHECK(constraint->allows(token));
        constraint->accept(token);
    }
    CHECK(!constraint->allows(END));
    constraint->accept(OBJ_CLOSE);
    CHECK(constraint->complete());
    CHECK(constraint->allows(END));
    CHECK(!constraint->allows(OBJ_OPEN));

    auto logits = prefer(PROSE);
    CHECK(constraint->mask(logits.data(), N_TOK) >= 1);
    CHECK(logits[END] != -INFINITY);
    CHECK(logits[PROSE] == -INFINITY);
}

TEST_CASE(OutputConstraintFixture, tool_call_opener_releases_the_output) {
    auto constraint = schema_constraint({}, {"<tool_call>"});
    CHECK(constraint->allows(TOOL_CALL));
    CHECK(constraint->allows(OBJ_OPEN));
    CHECK(!constraint->allows(PROSE));
    constraint->accept(TOOL_CALL);
    CHECK(constraint->released());
    CHECK(!constraint->engaged());
    CHECK(constraint->allows(PROSE));  // the tool parser reads the call
}

TEST_CASE(OutputConstraintFixture, waits_for_the_end_of_thinking) {
    OutputConstraint::Options options;
    options.wait_for = "</think>";
    auto constraint = schema_constraint(options);
    CHECK(!constraint->engaged());
    CHECK(constraint->allows(PROSE));  // free while thinking
    constraint->accept(PROSE);
    CHECK(!constraint->engaged());
    constraint->accept(THINK_END);
    CHECK(constraint->engaged());
    constraint->accept(NEWLINE);       // leading whitespace is allowed
    CHECK(!constraint->allows(PROSE));
    CHECK(constraint->allows(OBJ_OPEN));
}

TEST_CASE(OutputConstraintFixture, off_grammar_token_breaks_instead_of_dead_ending) {
    auto constraint = schema_constraint();
    constraint->accept(PROSE);  // injected past the sampler
    CHECK(constraint->broken());
    CHECK(constraint->allows(FENCE));
    constraint->reset();
    CHECK(constraint->engaged());
    CHECK(!constraint->allows(FENCE));
}

TEST_CASE(OutputConstraintFixture, request_output_format_reads_all_three_apis) {
    using http_detail::request_output_format;
    CHECK(request_output_format(R"({"messages":[]})").is_null());
    const auto chat = request_output_format(
        R"({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"b":{},"a":{}}}}}})");
    CHECK(chat.value("type", "") == "json_schema");
    // Property order survives (nlohmann::json would sort a before b).
    const auto props = chat["json_schema"]["schema"]["properties"];
    CHECK(props.begin().key() == "b");
    CHECK(request_output_format(R"({"text":{"format":{"type":"json_object"}}})")
              .value("type", "") == "json_object");
    CHECK(request_output_format(R"({"output_format":{"type":"json_schema","schema":{}}})")
              .value("type", "") == "json_schema");
}

TEST_CASE(OutputConstraintFixture, output_format_schema_normalizes_formats) {
    using http_detail::output_format_schema;
    nlohmann::ordered_json schema;
    std::string error;
    CHECK(output_format_schema(nullptr, schema, error));
    CHECK(schema.is_null());
    CHECK(output_format_schema({{"type", "text"}}, schema, error));
    CHECK(schema.is_null());
    CHECK(output_format_schema({{"type", "json_object"}}, schema, error));
    CHECK(schema.value("type", "") == "object");
    CHECK(output_format_schema(nlohmann::ordered_json::parse(
        R"({"type":"json_schema","json_schema":{"schema":{"type":"array"}}})"), schema, error));
    CHECK(schema.value("type", "") == "array");
    CHECK(output_format_schema(nlohmann::ordered_json::parse(
        R"({"type":"json_schema","schema":{"type":"string"}})"), schema, error));
    CHECK(schema.value("type", "") == "string");
    CHECK(!output_format_schema({{"type", "xml"}}, schema, error));
    CHECK(!error.empty());
    // A non-string type is the client's mistake: an error, not an exception.
    error.clear();
    CHECK(!output_format_schema({{"type", 1}}, schema, error));
    CHECK(!error.empty());
    CHECK(!output_format_schema(nlohmann::ordered_json::parse(R"({"type":null})"), schema, error));
}

TEST_CASE(OutputConstraintFixture, tool_choice_requires_call_covers_each_api) {
    using http_detail::tool_choice_requires_call;
    CHECK(tool_choice_requires_call("required"));
    CHECK(!tool_choice_requires_call("auto"));
    CHECK(!tool_choice_requires_call(nullptr));
    CHECK(tool_choice_requires_call(json::parse(R"({"type":"function","function":{"name":"f"}})")));
    CHECK(tool_choice_requires_call(json::parse(R"({"type":"function","name":"f"})")));
    CHECK(tool_choice_requires_call(json::parse(R"({"type":"any"})")));
    CHECK(tool_choice_requires_call(json::parse(R"({"type":"tool","name":"f"})")));
    CHECK(!tool_choice_requires_call(json::parse(R"({"type":"auto"})")));
    CHECK(!tool_choice_requires_call(json::parse(R"({"type":1})")));
}

TEST_CASE(OutputConstraintFixture, forced_tool_call_grammar_admits_only_openers) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"get_weather"}}])");
    const auto openers = tool_call_openers(tools);
    CHECK(std::find(openers.begin(), openers.end(), "<tool_call>") != openers.end());
    CHECK(std::find(openers.begin(), openers.end(), "<get_weather>") != openers.end());
    const std::string grammar = http_detail::output_constraint_grammar(nullptr, openers);
    CHECK(matches(grammar, "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"));
    CHECK(matches(grammar, "\n<｜DSML｜ calls>"));
    CHECK(!matches(grammar, "I will call the tool."));
}

TEST_CASE(OutputConstraintFixture, parse_grammar_from_body_top_level_and_extra_body) {
    std::string grammar;
    std::string error;

    // Absent or empty body
    CHECK(parse_grammar_from_body(json::object(), grammar, error));
    CHECK(grammar.empty());

    // Top-level grammar string
    CHECK(parse_grammar_from_body(json{{"grammar", "root ::= [0-9]+"}}, grammar, error));
    CHECK(grammar == "root ::= [0-9]+");

    // Top-level null grammar
    CHECK(parse_grammar_from_body(json{{"grammar", nullptr}}, grammar, error));
    CHECK(grammar.empty());

    // Top-level non-string grammar
    CHECK(!parse_grammar_from_body(json{{"grammar", 123}}, grammar, error));
    CHECK(error == "grammar must be a string");

    // extra_body grammar string
    CHECK(parse_grammar_from_body(json{{"extra_body", {{"grammar", "root ::= [a-z]+"}}}}, grammar, error));
    CHECK(grammar == "root ::= [a-z]+");

    // extra_body null grammar falls back if top-level is absent
    CHECK(parse_grammar_from_body(json{{"extra_body", {{"grammar", nullptr}}}}, grammar, error));
    CHECK(grammar.empty());

    // extra_body takes precedence over top-level
    CHECK(parse_grammar_from_body(
        json{{"grammar", "root ::= \"top\""}, {"extra_body", {{"grammar", "root ::= \"extra\""}}}},
        grammar, error));
    CHECK(grammar == "root ::= \"extra\"");

    // extra_body non-string grammar reports error
    CHECK(!parse_grammar_from_body(json{{"extra_body", {{"grammar", false}}}}, grammar, error));
    CHECK(error == "grammar must be a string");
}

TEST_CASE(OutputConstraintFixture, output_format_schema_supports_grammar_type) {
    using http_detail::output_format_schema;
    nlohmann::ordered_json schema;
    std::string error;

    // Valid response_format type grammar
    CHECK(output_format_schema(nlohmann::ordered_json::parse(
        R"({"type":"grammar","grammar":"root ::= \"ok\""})"), schema, error));
    CHECK(schema.is_null());

    // Missing grammar key in type grammar
    error.clear();
    CHECK(!output_format_schema(nlohmann::ordered_json::parse(R"({"type":"grammar"})"), schema, error));
    CHECK(error == "response_format grammar must be a string");

    // Non-string grammar key
    error.clear();
    CHECK(!output_format_schema(nlohmann::ordered_json::parse(R"({"type":"grammar","grammar":42})"), schema, error));
    CHECK(error == "response_format grammar must be a string");

    // Unknown type mentions grammar in error message
    error.clear();
    CHECK(!output_format_schema(nlohmann::ordered_json::parse(R"({"type":"unknown"})"), schema, error));
    CHECK(error.find("grammar") != std::string::npos);
}

TEST_CASE(OutputConstraintFixture, custom_grammar_constrains_tokens) {
    enum Tok : int32_t { YES, NO, MAYBE, END, N_TOK };
    std::vector<std::string> pieces(N_TOK);
    pieces[YES] = "yes";
    pieces[NO] = "no";
    pieces[MAYBE] = "maybe";
    pieces[END] = "<|im_end|>";

    std::string error;
    auto matcher = gbnf::Matcher::compile("root ::= \"yes\" | \"no\"", "root", error);
    CHECK(matcher != nullptr);

    OutputConstraint constraint(ConstraintVocab::build(pieces, {END}), std::move(matcher), {});
    CHECK(constraint.allows(YES));
    CHECK(constraint.allows(NO));
    CHECK(!constraint.allows(MAYBE));
    CHECK(!constraint.allows(END));

    constraint.accept(YES);
    CHECK(constraint.complete());
    CHECK(constraint.allows(END));
    CHECK(!constraint.allows(YES));
    CHECK(!constraint.allows(NO));
}
