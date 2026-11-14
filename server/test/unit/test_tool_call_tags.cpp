// Tool calls whose arguments hold tool-call tags: a coding agent asked to test
// this parser writes commands that hold a Qwen tool call (a C++ test of the
// parser). Values are raw text in every XML-like format, so a
// value may hold the format's own tags: a close tag ends a value only where
// the rest of the reply goes on as the format writes it. Pure logic: no model
// or GPU. The helpers below build the tags.
#include "CppUnitTestFramework.hpp"

#include "server/chat_codec.h"
#include "server/sse_emitter.h"
#include "server/tool_parser.h"

#include <string>
#include <utility>
#include <vector>

using namespace luce::common;

namespace {

struct ToolCallTagsFixture {};

using Args = std::vector<std::pair<std::string, std::string>>;

std::string open_tag(const std::string & name) { return "<" + name + ">"; }
std::string close_tag(const std::string & name) { return "</" + name + ">"; }
const std::string kCall = "tool_call";

// A coding agent's tools (bash, read, write), as such agents declare them.
json agent_tools() {
    return json::parse(R"([
        {"type": "function", "function": {"name": "bash", "parameters": {"type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
            "required": ["command"]}}},
        {"type": "function", "function": {"name": "read", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "number"}},
            "required": ["path"]}}},
        {"type": "function", "function": {"name": "write", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}}])");
}

// A call as the Qwen template writes it.
std::string qwen(const std::string & name, const Args & args) {
    std::string out = open_tag(kCall) + "\n" + open_tag("function=" + name) + "\n";
    for (const auto & [key, value] : args) {
        out += open_tag("parameter=" + key) + "\n" + value + "\n" + close_tag("parameter") + "\n";
    }
    return out + close_tag("function") + "\n" + close_tag(kCall);
}

// A call as the GLM / Laguna template writes it, and as it reads once the
// detokenizer has stripped its wrapper tokens (GLM-4.7's spacing).
std::string glm(const std::string & name, const Args & args) {
    std::string out = open_tag(kCall) + name + "\n";
    for (const auto & [key, value] : args) {
        out += open_tag("arg_key") + key + close_tag("arg_key") + "\n" + open_tag("arg_value") + value +
               close_tag("arg_value") + "\n";
    }
    return out + close_tag(kCall);
}
std::string glm_stripped(const std::string & name, const Args & args) {
    std::string out = name;
    for (const auto & [key, value] : args) {
        out += open_tag("arg_key") + key + close_tag("arg_key") + open_tag("arg_value") + value + close_tag("arg_value");
    }
    return out;
}
std::string strip_wrapper(std::string text) {
    for (const std::string & tag : {open_tag(kCall), close_tag(kCall)}) {
        for (size_t at; (at = text.find(tag)) != std::string::npos;) text.erase(at, tag.size());
    }
    return text;
}

// A DSML calls block with one string argument, in a dialect's tag words.
struct Dsml {
    const char * codec, * calls, * invoke, * param;
};
std::string dsml(const Dsml & d, const std::string & name, const std::string & key, const std::string & value) {
    const std::string m = "\xEF\xBD\x9C" "DSML" "\xEF\xBD\x9C";  // the fullwidth bars around DSML
    return "<" + m + d.calls + ">\n<" + m + d.invoke + " name=\"" + name + "\">\n<" + m + d.param + " name=\"" + key +
           "\" string=\"true\">" + value + "</" + m + d.param + ">\n</" + m + d.invoke + ">\n</" + m + d.calls + ">";
}

// Calls as "NAME ARGUMENTS" lines, so a failed check prints what was read.
std::string calls_of(const std::vector<ToolCall> & calls) {
    std::string out;
    for (const auto & call : calls) out += call.name + " " + json::parse(call.arguments).dump() + "\n";
    return out;
}
std::string call(const std::string & name, const json & args) { return name + " " + args.dump() + "\n"; }

// What a codec's strict reader reads from `text` ("unread" when it rejects it),
// and what the heuristic reader reads.
std::string codec_reads(const ChatCodec & codec, const std::string & text, const json & tools) {
    ToolParseResult parsed;
    return codec.parse_tool_calls(text, tools, parsed) ? calls_of(parsed.tool_calls) : "unread\n";
}
std::string heuristic_reads(const std::string & text, const json & tools) {
    return calls_of(parse_tool_calls(text, tools).tool_calls);
}

std::string inner_qwen() { return qwen("read_file", {{"path", "/etc/hostname"}}); }

// Commands that hold a call on lines of their own, as a model writes them when
// it tests this parser, and two more.
std::vector<std::string> commands() {
    const std::string inner = inner_qwen();
    const std::string two_args = qwen("read_file", {{"path", "/etc/hostname"}, {"limit", "5"}});
    return {
        // A C++ test holding the call in a raw string.
        "mkdir -p /tmp/t && cat > /tmp/t/test_inner.cpp <<'EOF'\n#include <cassert>\n#include <string>\n"
        "int main() {\n    const std::string text = R\"(" + inner + ")\";\n"
        "    assert(text.find(\"path\") != std::string::npos);\n}\nEOF\n"
        "g++ -std=c++17 -o /tmp/t/test_inner /tmp/t/test_inner.cpp && /tmp/t/test_inner",
        // The call in a fixture file of its own, then a test that reads it.
        "cat > /tmp/t/case.txt <<'EOF'\n" + inner + "\nEOF\n./build/test_server_unit qwen_fixture /tmp/t/case.txt",
        // A test case appended to the unit tests.
        "cat >> server/test/unit/test_server_unit.cpp <<'EOF'\n\nTEST_CASE(ServerUnitFixture, test_absolute_path) {\n"
        "    const std::string text = R\"(" + inner + ")\";\n"
        "    TEST_ASSERT(parse_tool_calls(text, file_tools()).tool_calls.size() == 1);\n}\nEOF",
        // A call with two arguments, in Python.
        "python3 - <<'EOF'\ntext = \"\"\"" + two_args + "\"\"\"\nopen('/tmp/t/case.txt', 'w').write(text)\nEOF",
        // The call as the last thing in the value.
        "echo next\n" + inner,
        // Close tags in a grep, unbalanced.
        "grep -n '" + close_tag("parameter") + "\n" + close_tag("function") + "' chat_codec.cpp; grep -c '" +
            open_tag("parameter=path") + "' x",
    };
}

SseEmitter thinking_emitter(const json & tools, const ChatCodec & codec) {
    SseEmitter emitter(ApiFormat::OPENAI_CHAT, "test_id", "test-model", 10, tools, nullptr, {},
                       /*started_in_thinking=*/true);
    emitter.set_chat_codec(&codec);
    return emitter;
}

}  // namespace

TEST_CASE(ToolCallTagsFixture, qwen_codec_reads_values_holding_tool_calls) {
    const json tools = agent_tools();
    for (const char * name : {"qwen3.8", "qwen3.6", "qwen3.5"}) {
        const auto codec = make_chat_codec(name);
        for (const auto & command : commands()) {
            CHECK_EQUAL(codec_reads(*codec, qwen("bash", {{"command", command}}), tools),
                        call("bash", {{"command", command}}));
        }
    }
    const auto codec = make_chat_codec("qwen3.8");
    // Two calls stay two, the first holding a call.
    const std::string command = commands()[0];
    CHECK_EQUAL(codec_reads(*codec, qwen("bash", {{"command", command}}) + "\n" +
                                        qwen("read", {{"path", "README.md"}, {"limit", "5"}}),
                            tools),
                call("bash", {{"command", command}}) + call("read", {{"path", "README.md"}, {"limit", 5}}));
    // A file that is a call, and one whose call names this tool's own parameter
    // (path, already given, is text there).
    const std::string own = qwen("write", {{"mode", "x"}, {"path", "/etc/hostname"}});
    for (const std::string & content : {inner_qwen(), own, "before\n" + own + "\nafter"}) {
        CHECK_EQUAL(codec_reads(*codec, qwen("write", {{"path", "case.txt"}, {"content", content}}), tools),
                    call("write", {{"path", "case.txt"}, {"content", content}}));
    }
    // An argument the tool does not declare still reads, as before.
    CHECK_EQUAL(codec_reads(*codec, qwen("bash", {{"command", "ls"}, {"description", "list"}}), tools),
                call("bash", {{"command", "ls"}, {"description", "list"}}));
}

TEST_CASE(ToolCallTagsFixture, heuristic_reader_reads_values_holding_tool_calls) {
    const json tools = agent_tools();
    for (const auto & command : commands()) {
        // As the template writes it, and without the newlines around the value.
        const std::string text = qwen("bash", {{"command", command}});
        std::string drifted = text;
        drifted.erase(drifted.find(open_tag("parameter=command")) + open_tag("parameter=command").size(), 1);
        drifted.erase(drifted.rfind(close_tag("parameter")) - 1, 1);
        for (const auto & reply : {text, drifted}) {
            CHECK_EQUAL(heuristic_reads(reply, tools), call("bash", {{"command", command}}));
        }
    }
    // Two calls stay two.
    const std::string command = commands()[0];
    CHECK_EQUAL(heuristic_reads(qwen("bash", {{"command", command}}) + "\n" + qwen("read", {{"path", "a"}}), tools),
                call("bash", {{"command", command}}) + call("read", {{"path", "a"}}));
    // A missing parameter close still ends a value at the next declared argument.
    std::string unclosed = qwen("write", {{"path", "a.txt"}, {"content", "hi"}});
    unclosed.erase(unclosed.find(close_tag("parameter")), close_tag("parameter").size() + 1);
    CHECK_EQUAL(heuristic_reads(unclosed, tools), call("write", {{"path", "a.txt"}, {"content", "hi"}}));
}

TEST_CASE(ToolCallTagsFixture, deepseek_codec_reads_values_holding_tool_calls) {
    const json tools = agent_tools();
    for (const Dsml & d : {Dsml{"deepseek4.1", " calls", " invoke", " parameter"},
                           Dsml{"deepseek4", "tool_calls", "invoke", "parameter"}}) {
        const std::string inner = dsml(d, "read_file", "path", "/etc/hostname");
        const auto codec = make_chat_codec(d.codec);
        for (const std::string & command : {
                 "cat > /tmp/t/test_dsml.cpp <<'EOF'\nconst char * text = R\"(" + inner + ")\";\nEOF\n"
                 "g++ -c /tmp/t/test_dsml.cpp",
                 "cat > /tmp/t/case.txt <<'EOF'\n" + inner + "\nEOF",
                 "echo next\n" + inner}) {
            CHECK_EQUAL(codec_reads(*codec, dsml(d, "bash", "command", command), tools),
                        call("bash", {{"command", command}}));
        }
    }
}

TEST_CASE(ToolCallTagsFixture, heuristic_reader_reads_arg_values_holding_tool_calls) {
    const json tools = agent_tools();
    const std::string inner = glm("read_file", {{"path", "/etc/hostname"}});
    const std::vector<std::string> commands = {
        "cat > /tmp/t/test_glm.cpp <<'EOF'\nconst char * text = R\"(" + inner + ")\";\nEOF\ng++ -c /tmp/t/test_glm.cpp",
        "cat > /tmp/t/case.txt <<'EOF'\n" + inner + "\nEOF",
        "python3 - <<'EOF'\ntext = \"\"\"" + glm("read_file", {{"path", "/etc/hostname"}, {"limit", "5"}}) +
            "\"\"\"\nEOF",
    };
    for (const auto & command : commands) {
        CHECK_EQUAL(heuristic_reads(glm("bash", {{"command", command}}), tools), call("bash", {{"command", command}}));
    }
    // With the wrapper tokens stripped, outside and in. A quoted call of two
    // arguments is then out of reach: nothing in the text marks where it starts.
    for (size_t i = 0; i < 2; ++i) {
        const std::string command = strip_wrapper(commands[i]);
        CHECK_EQUAL(heuristic_reads(glm_stripped("bash", {{"command", command}}), tools),
                    call("bash", {{"command", command}}));
    }
    // Two calls stay two.
    CHECK_EQUAL(heuristic_reads(glm("bash", {{"command", commands[1]}}) + "\n" + glm("read", {{"path", "README.md"}}),
                                tools),
                call("bash", {{"command", commands[1]}}) + call("read", {{"path", "README.md"}}));
}

// A reply whose thinking quotes the golden empty call three times, then makes
// one real call. The quotes are thinking, not three empty calls.
TEST_CASE(ToolCallTagsFixture, emitter_reads_calls_quoted_in_thinking_as_thinking) {
    const auto codec = make_chat_codec("qwen3.8");
    const std::string quote = qwen("bash", {});
    for (const std::string & command :
         {std::string("sed -n '760,830p' server/src/server/sse_emitter.cpp"), commands()[0]}) {
        auto em = thinking_emitter(agent_tools(), *codec);
        em.emit_token("The golden case renders:\n```\n");
        em.emit_token(quote);
        em.emit_token("\n```\nand again\n```\n" + quote + "\n```\nand once more " + quote + "\nSo look at the emitter.\n");
        em.emit_token(close_tag("think") + "\n\n");
        em.emit_token(qwen("bash", {{"command", command}}));
        em.emit_finish(5);
        CHECK_EQUAL(calls_of(em.tool_calls()), call("bash", {{"command", command}}));
        CHECK(em.reasoning_text().find("and again\n```\n" + quote) != std::string::npos);
        CHECK(em.reasoning_text().find("So look at the emitter.") != std::string::npos);
        CHECK(!em.dropped_tool_call());
    }
    // A call made inside a thinking that never closes still counts, as before.
    auto open_thinking = thinking_emitter(agent_tools(), *codec);
    open_thinking.emit_token("Run it.\n");
    open_thinking.emit_token(qwen("bash", {{"command", "ls"}}));
    open_thinking.emit_finish(2);
    CHECK_EQUAL(calls_of(open_thinking.tool_calls()), call("bash", {{"command", "ls"}}));
}
