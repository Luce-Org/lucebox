#include "server/tool_parser.h"
#include "support/tool_schemas.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <string>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

// ═══════════════════════════════════════════════════════════════════════
// Tool parser tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_parse_tool_call_xml) {
    std::string text =
        "Some text\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=location>San Francisco</parameter>\n"
        "<parameter=unit>celsius</parameter>\n"
        "</function>\n"
        "</tool_call>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args.contains("location"));
        TEST_ASSERT(args["location"] == "San Francisco");
        TEST_ASSERT(args.contains("unit"));
        TEST_ASSERT(args["unit"] == "celsius");
    }
    TEST_ASSERT(result.cleaned_text.find("<tool_call>") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_bare_function_xml) {
    std::string text =
        "<function=list_files>\n"
        "<parameter=path>/home</parameter>\n"
        "</function>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "list_files");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_bare_tool_name_xml_with_function_close) {
    std::string text =
        "\n\n\nLet me find the correct line range for the tests array.\n\n\n"
        "<bash>\n"
        "<parameter=command>\n"
        "grep -n \"f5.test\" /workspace/project/tests/bootstrap.cjs\n"
        "</parameter>\n"
        "</function>\n";
    json tools = json::array({
        {{"type", "function"},
         {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] ==
                    "grep -n \"f5.test\" /workspace/project/tests/bootstrap.cjs");
    }
    TEST_ASSERT(result.cleaned_text.find("<bash>") == std::string::npos);
    TEST_ASSERT(result.cleaned_text.find("</function>") == std::string::npos);
    TEST_ASSERT(result.cleaned_text.find("Let me find the correct line range") !=
                std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_repeated_bare_edit_calls_with_trailing_close) {
    const std::string text =
        "Applying both updates.\n\n"
        "<edit>\n"
        "<parameter=edits>\n"
        "[{\"path\":\"/workspace/first.conf\",\"oldText\":\"auto\","
        "\"newText\":\"enabled\"}]\n"
        "</parameter>\n"
        "</function>\n\n"
        "<edit>\n"
        "<parameter=edits>\n"
        "[{\"path\":\"/workspace/second.conf\",\"oldText\":\"auto\","
        "\"newText\":\"enabled\"}]\n"
        "</parameter>\n"
        "</function>\n\n"
        "</edit>";

    auto result = parse_tool_calls(text, edit_tools());
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "edit");
        TEST_ASSERT(result.tool_calls[1].name == "edit");
        const auto first = json::parse(result.tool_calls[0].arguments);
        const auto second = json::parse(result.tool_calls[1].arguments);
        TEST_ASSERT(first["edits"][0]["path"] == "/workspace/first.conf");
        TEST_ASSERT(second["edits"][0]["path"] == "/workspace/second.conf");
    }
    TEST_ASSERT(result.cleaned_text == "Applying both updates.");
}

TEST_CASE(ServerUnitFixture, test_tool_syntax_scanner_declared_name_guards) {
    size_t pos = std::string::npos;
    TEST_ASSERT(find_tool_syntax_start("prefix<edit>", edit_tools(), pos));
    TEST_ASSERT(pos == 6);

    pos = std::string::npos;
    TEST_ASSERT(!find_tool_syntax_start("prefix<unknown_tool>", edit_tools(), pos));

    json invalid = edit_tools();
    invalid[0]["function"]["name"] = std::string(65, 'x');
    TEST_ASSERT(tool_syntax_holdback(invalid) == 21);
    pos = std::string::npos;
    TEST_ASSERT(!find_tool_syntax_start("<" + std::string(65, 'x') + ">",
                                        invalid, pos));
}

TEST_CASE(ServerUnitFixture, test_parse_undeclared_file_tag_stays_content) {
    const std::string text =
        "Now I understand how to add a custom model. I need to edit "
        "~/.pi/agent/models.json to add a vLLM provider with the "
        "laguna-s-2.1 model. Let me check if the file exists first.\n\n"
        "<file>\n"
        "<parameter=path>\n"
        "~/.pi/agent/models.json\n"
        "</parameter>\n"
        "</function>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(result.cleaned_text == text);
}

TEST_CASE(ServerUnitFixture, test_parse_attribute_style_tool_xml) {
    std::string text =
        "The branch already exists. Let me check the current state:\n\n"
        "<parameter name=\"bash\"><parameter name=\"command\">"
        "cd /workspace/project && "
        "git status -sb && git branch\n --show-current</parameter>\n"
        "</function>\n";
    auto result = parse_tool_calls(text, bash_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] ==
                    "cd /workspace/project && "
                    "git status -sb && git branch\n --show-current");
    }
    TEST_ASSERT(result.cleaned_text ==
                "The branch already exists. Let me check the current state:");

    const std::string wrapped =
        "Checking now.\n\n<tool_call>\n"
        "<parameter name=\"bash\"><parameter name=\"command\">"
        "git status</parameter>\n</function>\n</tool_call>";
    auto wrapped_result = parse_tool_calls(wrapped, bash_tools());
    TEST_ASSERT(wrapped_result.tool_calls.size() == 1);
    TEST_ASSERT(wrapped_result.cleaned_text == "Checking now.");
}

TEST_CASE(ServerUnitFixture, test_parse_mixed_tool_variants_preserve_source_order) {
    const std::string text =
        "<function read>\n"
        "<parameter=path>/tmp/first.md</parameter>\n"
        "</function>\n"
        "<parameter name=\"bash\"><parameter name=\"command\">"
        "cat /tmp/first.md</parameter></function>";
    auto result = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(result.tool_calls[1].name == "bash");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_attribute_style_tool_xml_rejects_malformed_body) {
    const std::string malformed =
        "<parameter name=\"bash\"><parameter name=\"command\">git status\n"
        "</function>";
    auto malformed_result = parse_tool_calls(malformed, bash_tools());
    TEST_ASSERT(malformed_result.tool_calls.empty());
    TEST_ASSERT(malformed_result.cleaned_text == malformed);

    const std::string unknown =
        "<parameter name=\"not_a_tool\"><parameter name=\"command\">"
        "git status</parameter></function>";
    auto unknown_result = parse_tool_calls(unknown, bash_tools());
    TEST_ASSERT(unknown_result.tool_calls.empty());
    TEST_ASSERT(unknown_result.cleaned_text == unknown);
}

TEST_CASE(ServerUnitFixture, test_parse_funcname_tool_xml) {
    const std::string text =
        "<tool_call>\n"
        "<funcname>read\n"
        "<parameter=limit>\n50\n</parameter>\n"
        "<parameter=offset>\n1\n</parameter>\n"
        "<parameter=path>\n"
        "/tmp/tool-input.md\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n";
    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["limit"] == 50);
        TEST_ASSERT(args["offset"] == 1);
        TEST_ASSERT(args["path"] == "/tmp/tool-input.md");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_funcname_tool_xml_rejects_malformed_or_unknown) {
    const std::string malformed =
        "<funcname>read\n"
        "<parameter=path>/tmp/task.md\n"
        "</function>";
    auto malformed_result = parse_tool_calls(malformed, read_tools());
    TEST_ASSERT(malformed_result.tool_calls.empty());
    TEST_ASSERT(malformed_result.cleaned_text == malformed);

    const std::string unknown =
        "<funcname>write\n"
        "<parameter=path>/tmp/task.md</parameter>\n"
        "</function>";
    auto unknown_result = parse_tool_calls(unknown, read_tools());
    TEST_ASSERT(unknown_result.tool_calls.empty());
    TEST_ASSERT(unknown_result.cleaned_text == unknown);
}

TEST_CASE(ServerUnitFixture, test_parse_space_function_tool_xml) {
    const std::string text =
        "Let me read the file and compute its SHA-256 hash.\n\n"
        "<function read>\n"
        "<parameter=path>\n"
        "/tmp/tool-input.md\n"
        "</parameter>\n"
        "</function>\n\n"
        "<function bash>\n"
        "<parameter=command>\n"
        "sha256sum /tmp/tool-input.md\n"
        "</parameter>\n"
        "</function>";
    auto result = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(json::parse(result.tool_calls[0].arguments)["path"] ==
                    "/tmp/tool-input.md");
        TEST_ASSERT(result.tool_calls[1].name == "bash");
        TEST_ASSERT(json::parse(result.tool_calls[1].arguments)["command"] ==
                    "sha256sum /tmp/tool-input.md");
    }
    TEST_ASSERT(result.cleaned_text ==
                "Let me read the file and compute its SHA-256 hash.");
}

TEST_CASE(ServerUnitFixture, test_parse_space_function_tool_xml_rejects_malformed_or_unknown) {
    const std::string malformed =
        "<function read>\n<parameter=path>/tmp/task.md\n</function>";
    auto malformed_result = parse_tool_calls(malformed, read_tools());
    TEST_ASSERT(malformed_result.tool_calls.empty());
    TEST_ASSERT(malformed_result.cleaned_text == malformed);

    const std::string unknown =
        "<function write>\n<parameter=path>/tmp/task.md</parameter>\n</function>";
    auto unknown_result = parse_tool_calls(unknown, read_tools());
    TEST_ASSERT(unknown_result.tool_calls.empty());
    TEST_ASSERT(unknown_result.cleaned_text == unknown);
}

TEST_CASE(ServerUnitFixture, test_parse_json_tool_call) {
    std::string text =
        "{\"name\": \"search\", \"arguments\": {\"query\": \"hello world\"}}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "search");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["query"] == "hello world");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_single_tool_bare_json_args) {
    std::string text =
        "{\n"
        "  \"command\": \"git branch --show-current\"\n"
        "}";
    auto result = parse_tool_calls(text, shell_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "shell");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "git branch --show-current");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_single_tool_bare_json_args_allows_empty_optional_object) {
    auto result = parse_tool_calls("{}", optional_shell_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "shell");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args.is_object());
        TEST_ASSERT(args.empty());
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_single_tool_bare_json_args_rejects_prose) {
    std::string text = "The command is {\"command\": \"git status\"}.";
    auto result = parse_tool_calls(text, shell_tools());
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(result.cleaned_text == text);
}

TEST_CASE(ServerUnitFixture, test_parse_single_tool_bare_json_args_rejects_ambiguous_tools) {
    std::string text = "{\"command\": \"git status\"}";
    auto result = parse_tool_calls(text, weather_tools());
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(result.cleaned_text == text);
}

TEST_CASE(ServerUnitFixture, test_parse_no_tools) {
    std::string text = "Just plain text without any tool calls.";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(!result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_tool_code_wrapper) {
    std::string text =
        "<tool_code>\n"
        "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}\n"
        "</tool_code>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_function_call_wrapper) {
    std::string text =
        "<function_call>\n"
        "{\"name\": \"bash\", \"arguments\": {\"command\": \"echo 'hello'\"}}\n"
        "</function_call>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "echo 'hello'");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_legacy_openai_function_call_json) {
    const std::string text =
        "{\"function_call\":{\"arguments\":"
        "\"{\\\"location\\\":\\\"test-city\\\"}\","
        "\"name\":\"get_weather\"}}";
    const auto result = parse_tool_calls(text, weather_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["location"] == "test-city");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_deepseek_function_parameters_json) {
    const std::string text =
        "{\"function\":\"get_weather\",\"parameters\":{"
        "\"location\":\"test-city\",\"unit\":\"celsius\"}}";
    const auto result = parse_tool_calls(text, weather_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["location"] == "test-city");
        TEST_ASSERT(args["unit"] == "celsius");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_deepseek_function_stringified_parameters_json) {
    const std::string text =
        "{\"function\":\"get_weather\",\"parameters\":"
        "\"{\\\"location\\\":\\\"test-city\\\",\\\"unit\\\":\\\"celsius\\\"}\"}";
    const auto result = parse_tool_calls(text, weather_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["location"] == "test-city");
        TEST_ASSERT(args["unit"] == "celsius");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_bare_function_json_with_parameters) {
    std::string text =
        "<function>\n"
        "{\n"
        "  \"name\": \"bash\",\n"
        "  \"parameters\": {\n"
        "    \"command\": \"ls -la \\\"/home/dpavlin/aimax project\\\"\"\n"
        "  }\n"
        "}\n"
        "</function>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "ls -la \"/home/dpavlin/aimax project\"");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_function_call_xml_invoke_name_parameters) {
    const std::string text =
        "Let me read the file.\n\n"
        "<function_call>\n"
        "<invoke_name>read</invoke_name>\n"
        "<parameters>\n"
        "<path>/home/dpavlin/koha-rfid-go/internal/rfidops/ops.go</path>\n"
        "</parameters>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home/dpavlin/koha-rfid-go/internal/rfidops/ops.go");
    }
    TEST_ASSERT(result.cleaned_text == "Let me read the file.");
}

TEST_CASE(ServerUnitFixture, test_parse_function_call_xml_multiple_parameters) {
    const std::string text =
        "<function_call>\n"
        "<invoke_name>bash</invoke_name>\n"
        "<parameters>\n"
        "<command>ls -la /tmp</command>\n"
        "<timeout>10</timeout>\n"
        "</parameters>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}},
                     {"timeout", {{"type", "integer"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "ls -la /tmp");
        TEST_ASSERT(args["timeout"] == 10);
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_tool_call_xml_invoke_name_parameters) {
    const std::string text =
        "<tool_call>\n"
        "<invoke_name>read</invoke_name>\n"
        "<parameters>\n"
        "<path>/home/dpavlin/test.go</path>\n"
        "</parameters>\n"
        "</tool_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home/dpavlin/test.go");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_legacy_tool_call_with_nested_name_tag) {
    // Regression test for Violation 1:
    // A legacy <tool_call><function=...> envelope whose parameter contains a nested <name> tag
    // must NOT be hijacked by parse_xml_tool_call_body as envelope name.
    const std::string text =
        "<tool_call>\n"
        "<function=edit_file>\n"
        "<parameter=content>\n"
        "function getTool() { return \"<name>other_tool</name>\"; }\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "edit_file"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"content", {{"type", "string"}}}
                 }}
             }}
         }}},
        {{"type", "function"}, {"function", {
             {"name", "other_tool"},
             {"parameters", {{"type", "object"}, {"properties", {}}}}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "edit_file");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["content"] == "function getTool() { return \"<name>other_tool</name>\"; }");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_xml_tool_call_malformed_parameters_rejected) {
    // Regression test for Violation 2:
    // When the parameter section is non-empty but malformed (unsupported/invalid syntax),
    // parse_xml_tool_call_body must NOT emit an empty {} tool call.
    const std::string text =
        "<function_call>\n"
        "<invoke_name>edit_file</invoke_name>\n"
        "<parameters>\n"
        "random unparseable garbage text\n"
        "</parameters>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "edit_file"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"content", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(!result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_xml_tool_call_zero_arguments_accepted) {
    // Genuinely zero-argument calls should be accepted.
    const std::string text =
        "<function_call>\n"
        "<invoke_name>get_status</invoke_name>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "get_status"},
             {"parameters", {{"type", "object"}, {"properties", {}}}}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_status");
        TEST_ASSERT(result.tool_calls[0].arguments == "{}");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_xml_tool_call_invoke_envelope_attribute_style) {
    const std::string text =
        "<function_call>\n"
        "<invoke name=\"read\">\n"
        "<parameter name=\"path\">/tmp/test.txt</parameter>\n"
        "</invoke>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/tmp/test.txt");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_tool_call_xml_function_name_tag) {
    // Regression test for Violation:
    // <tool_call> envelopes using <function_name> must be parsed cleanly by parse_xml_tool_call_body()
    const std::string text =
        "<tool_call>\n"
        "<function_name>read</function_name>\n"
        "<parameters>\n"
        "<path>/tmp/test.txt</path>\n"
        "</parameters>\n"
        "</tool_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/tmp/test.txt");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}



TEST_CASE(ServerUnitFixture, test_parse_tool_allowed_filter) {
    std::string text =
        "<function=blocked_tool>\n"
        "<parameter=x>1</parameter>\n"
        "</function>";
    json tools = json::array({
        {{"type", "function"}, {"function", {{"name", "allowed_tool"}}}}
    });
    auto result = parse_tool_calls(text, tools);
    // Tool not in allow-list should be filtered
    TEST_ASSERT(result.tool_calls.empty());
}

// ─── Pattern 5: call:<verb>{...} plain-text tool calls ─────────────────
//
// Covers the gemma plain-text emission path added in
// server/src/server/tool_parser.cpp (PR #340). The opener regex requires
// a sentinel character before `call:` (start-of-string or one of
// [\s,;:\(\[\{\}\)\]\>_]); the body is brace-balanced and string-aware;
// and the args go through coerce_relaxed_json before becoming the
// argument object.

TEST_CASE(ServerUnitFixture, test_parse_call_verb_empty_args) {
    // Bareword `call:get_weather{}` at start-of-string — sentinel
    // matches the leading `^` anchor; body is the empty object `{}`.
    auto result = parse_tool_calls("call:get_weather{}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args.is_object());
        TEST_ASSERT(args.empty());
    }
    // The matched span should be removed from cleaned_text.
    TEST_ASSERT(result.cleaned_text.find("call:get_weather") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_strict_json_args) {
    // Strict JSON args go through json::parse directly in
    // coerce_relaxed_json's fast path.
    auto result = parse_tool_calls("call:get_weather{\"city\": \"NYC\"}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["city"] == "NYC");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_namespaced_verb) {
    // `ns:foo` namespaced verbs — the colon-strip logic in pattern 5
    // strips everything up to the last `:` so the registered tool name
    // is just `foo`.
    auto result = parse_tool_calls("call:ns:foo{\"k\": 1}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "foo");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["k"] == 1);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_whitespace_before_key) {
    // Leading whitespace inside the brace body must not break parsing.
    // (Whitespace tolerance is provided by json::parse / the relaxed
    //  fallback rewriter.)
    auto result = parse_tool_calls("call:get_weather{ \"city\": \"NYC\" }");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["city"] == "NYC");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_missing_close_brace_rejected) {
    // Unbalanced opener — balanced_braces_end returns npos so pattern 5
    // bails out and produces no tool call. The text leaks through.
    auto result = parse_tool_calls("call:get_weather{\"city\": \"NYC\"");
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_narrative_without_body_rejected) {
    // Narrative usage with a non-balanced body — sentinel matches the
    // space before `call:`, but the `{` has no matching `}` so the
    // call is discarded.
    auto result = parse_tool_calls("I will call:foo{");
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_underscore_prefix) {
    // SentencePiece artifact: `_call:` (the `_` is the literal
    // underscore character; sentinel char-class includes `_` for
    // exactly this case).
    auto result = parse_tool_calls("_call:get_weather{\"city\": \"NYC\"}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["city"] == "NYC");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_nested_object_args) {
    // Nested `{}` inside the args — balanced_braces_end tracks depth so
    // the outer close isn't consumed by the inner object.
    auto result = parse_tool_calls(
        "call:get_weather{\"loc\": {\"city\": \"NYC\", \"zip\": \"10001\"}}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["loc"].is_object());
        TEST_ASSERT(args["loc"]["city"] == "NYC");
        TEST_ASSERT(args["loc"]["zip"] == "10001");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_back_to_back) {
    // Gemma frequently emits multiple invocations back-to-back. The
    // sentinel char-class includes `}` so the second `call:` is found
    // after the first closes.
    auto result = parse_tool_calls(
        "call:a{\"x\": 1}call:b{\"y\": 2}");
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "a");
        auto args0 = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args0["x"] == 1);
        TEST_ASSERT(result.tool_calls[1].name == "b");
        auto args1 = json::parse(result.tool_calls[1].arguments);
        TEST_ASSERT(args1["y"] == 2);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_relaxed_single_quotes) {
    // Relaxed-JSON fallback: single-quoted strings + bare identifier
    // keys are rewritten to strict JSON before parse.
    auto result = parse_tool_calls("call:foo{city: 'NYC'}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "foo");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["city"] == "NYC");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_glued_to_word_rejected) {
    // No sentinel char before `call:` (glued to identifier) — pattern 5
    // must NOT match. `_` is a deliberate exception covered by its
    // own test; here we use a regular letter.
    auto result = parse_tool_calls("xcall:foo{\"a\": 1}");
    // Pattern 5 should NOT fire. Pattern 6 (bare-JSON sweep) sees
    // `{"a": 1}` but it has no `name`/`function` field, so it produces
    // no tool call either.
    TEST_ASSERT(result.tool_calls.empty());
}

// ─── Pattern 5 (cont.): PR #341 imports — narrative & quoting edge cases ─
//
// These tests originated in PR #341 alongside sse_emitter Pattern-B work
// and were relocated here when #341 was split. They focus on edge cases
// that complement the core call:<verb>{} suite above.

TEST_CASE(ServerUnitFixture, test_parse_call_verb_single) {
    std::string text = "call:get_country_info{country: \"France\"}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_country_info");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["country"] == "France");
    }
    TEST_ASSERT(result.cleaned_text.find("call:") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_namespaced) {
    std::string text = "call:execute-bead:read-file{path: \"crates/foo/src/lib.rs\"}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        // Verb only — namespace stripped.
        TEST_ASSERT(result.tool_calls[0].name == "read-file");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "crates/foo/src/lib.rs");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_snake_and_hyphen) {
    std::string text =
        "call:execute-bead:list-files{path: \"src/\"}\n\n"
        "call:execute-bead:read_file{path: \"a/b.rs\"}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "list-files");
        TEST_ASSERT(result.tool_calls[1].name == "read_file");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_tool_allowed_filter) {
    std::string text = "call:disallowed_verb{x: 1}call:allowed_verb{y: 2}";
    json tools = json::array({
        {{"type", "function"}, {"function", {{"name", "allowed_verb"}}}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "allowed_verb");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_inline_prose_rejected) {
    // No sentinel char before `call:` — must NOT match.
    std::string text = "narrative.call:foo{x:1}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_inline_prose_after_space) {
    // Whitespace IS a valid sentinel — this should match.
    std::string text = "Sure, I'll call:foo{x: 1}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "foo");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_malformed_args) {
    // Unterminated brace — drop the call, don't crash.
    std::string text = "call:foo{country: \"France\"";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_inner_brace_in_string) {
    // The `{` and `}` inside the string value must not confuse the
    // balanced-brace scanner.
    std::string text = "call:foo{cmd: \"echo {not_a_brace} ok\"}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["cmd"] == "echo {not_a_brace} ok");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_unquoted_keys) {
    // Relaxed-JSON path: bare keys get quoted.
    std::string text = "call:foo{path: \"x\", count: 3}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "x");
        TEST_ASSERT(args["count"] == 3);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_cleaned_text) {
    // The matched span should be stripped from cleaned_text.
    std::string text = "Hello call:foo{x: 1} world.";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    TEST_ASSERT(result.cleaned_text.find("call:") == std::string::npos);
    TEST_ASSERT(result.cleaned_text.find("Hello") != std::string::npos);
    TEST_ASSERT(result.cleaned_text.find("world.") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_intercept_inner_json) {
    // Regression case: inner args of the form {"name": ..., "arguments": ...}
    // must NOT be picked up by pattern 6 (bare-JSON sweep) as a spurious
    // `inner` ToolCall. Exactly one ToolCall, named `outer`, with the
    // inner JSON intact in its arguments.
    std::string text = "call:outer{\"name\": \"inner\", \"arguments\": {}}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "outer");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["name"] == "inner");
        TEST_ASSERT(args["arguments"].is_object());
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_multiline_args) {
    // Snapshot rows have multi-line nested args; the balanced-brace
    // scanner is line-agnostic, so this must Just Work.
    std::string text =
        "call:default_api:analyze_data{\n"
        "  data: [{\"date\": \"2024-10-05\", \"qty\": 50}, {\"date\": \"2024-10-06\", \"qty\": 60}],\n"
        "  metric: \"qty\"\n"
        "}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "analyze_data");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["metric"] == "qty");
        TEST_ASSERT(args["data"].is_array());
        TEST_ASSERT(args["data"].size() == 2);
    }
}


TEST_CASE(ServerUnitFixture, test_parse_call_verb_singlequote_with_inner_doublequote) {
    // Cubic PR #329 review: when the relaxed-JSON rewrite converts
    // single-quoted strings to double-quoted, inner `"` chars must be
    // escaped to `\"` — otherwise `'he said "hi"'` rewrites to
    // `"he said "hi""` which is invalid JSON and the whole tool call
    // is silently dropped.
    std::string text = "call:say{quote: 'he said \"hi\" loudly'}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "say");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["quote"] == "he said \"hi\" loudly");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_backtick_with_inner_doublequote) {
    // Same escape concern as the single-quote case, but with the
    // backtick string flavor.
    std::string text = "call:say{quote: `he said \"hi\" loudly`}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["quote"] == "he said \"hi\" loudly");
    }
}


TEST_CASE(ServerUnitFixture, test_parse_function_calls_invoke_xml) {
    const std::string text =
        "Reading configuration:\n"
        "<function_calls>\n"
        "<invoke name=\"read\">\n"
        "  <param name=\"path\">server.go</param>\n"
        "  <param name=\"offset\">10</param>\n"
        "  <param name=\"limit\">50</param>\n"
        "</invoke>\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "server.go");
        TEST_ASSERT(args["offset"] == 10);
        TEST_ASSERT(args["limit"] == 50);
    }
    TEST_ASSERT(result.cleaned_text == "Reading configuration:");
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_invoke_json) {
    const std::string text =
        "<function_calls>\n"
        "<invoke name=\"read\">\n"
        "  {\"path\": \"app.py\", \"offset\": \"5\"}\n"
        "</invoke>\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "app.py");
        TEST_ASSERT(args["offset"] == 5);
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_anthropic_input_schema) {
    json anthropic_tools = json::array({
        {
            {"name", "read"},
            {"description", "Read a file"},
            {"input_schema", {
                {"type", "object"},
                {"properties", {
                    {"path", {{"type", "string"}}},
                    {"offset", {{"type", "integer"}}}
                }}
            }}
        }
    });

    const std::string text =
        "<function_calls>\n"
        "<invoke name=\"read\">\n"
        "  <param name=\"path\">main.cpp</param>\n"
        "  <param name=\"offset\">42</param>\n"
        "</invoke>\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, anthropic_tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "main.cpp");
        TEST_ASSERT(args["offset"] == 42);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_json_lines) {
    std::string text =
        "Let me read the files:\n"
        "<function_calls>\n"
        "{\"name\": \"read\", \"arguments\": {\"path\": \"file1.txt\"}}\n"
        "{\"name\": \"read\", \"arguments\": {\"path\": \"file2.txt\"}}\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(result.tool_calls[1].name == "read");
        auto a1 = json::parse(result.tool_calls[0].arguments);
        auto a2 = json::parse(result.tool_calls[1].arguments);
        TEST_ASSERT(a1["path"] == "file1.txt");
        TEST_ASSERT(a2["path"] == "file2.txt");
    }
    TEST_ASSERT(result.cleaned_text == "Let me read the files:");
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_sibling_invokes_partial_failure) {
    std::string text =
        "<function_calls>\n"
        "<invoke name=\"read\">\n"
        "  <param name=\"path\">valid.txt</param>\n"
        "</invoke>\n"
        "<invoke name=\"read\">\n"
        "  malformed parameter text\n"
        "</invoke>\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "valid.txt");
    }
    TEST_ASSERT(result.cleaned_text.find("malformed parameter text") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_rejected_invokes_do_not_expose_nested_json) {
    const std::string disallowed =
        "<function_calls>"
        "<invoke name=\"forbidden\">"
        "{\"name\":\"read\",\"arguments\":{\"path\":\"secret\"}}"
        "</invoke>"
        "</function_calls>";
    auto disallowed_result = parse_tool_calls(disallowed, read_tools());
    TEST_ASSERT(disallowed_result.tool_calls.empty());
    TEST_ASSERT(disallowed_result.cleaned_text == disallowed);

    const std::string malformed =
        "<function_calls>"
        "<invoke>"
        "{\"name\":\"read\",\"arguments\":{\"path\":\"secret\"}}"
        "</invoke>"
        "</function_calls>";
    auto malformed_result = parse_tool_calls(malformed, read_tools());
    TEST_ASSERT(malformed_result.tool_calls.empty());
    TEST_ASSERT(malformed_result.cleaned_text == malformed);
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_invoke_rejects_braced_prose) {
    const std::string text =
        "<function_calls>"
        "<invoke name=\"read\">{this is prose}</invoke>"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(result.cleaned_text == text);
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_invoke_keeps_structured_json_syntax_errors) {
    const std::string text =
        "<function_calls>"
        "<invoke name=\"read\">{\"offset\":5o1}</invoke>"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(result.tool_calls[0].arguments == "{\"offset\":5o1}");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_json_syntax_error_forwarding) {
    std::string text = "<function_call>{\"name\": \"read\", \"arguments\": {\"offset\": 5o1}}</function_call>";
    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(result.tool_calls[0].arguments == "{\"offset\": 5o1}");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_tool_call_rejects_empty_name_and_scalar_arguments) {
    // Empty tool name must be rejected
    const std::string empty_name =
        "<function_call>\n{\"name\": \"\", \"arguments\": {\"path\": \"/tmp/test\"}}\n</function_call>";
    auto res_empty = parse_tool_calls(empty_name, read_tools());
    TEST_ASSERT(res_empty.tool_calls.empty());

    // Valid scalar string argument must be rejected (not an object)
    const std::string scalar_arg =
        "<function_call>\n{\"name\": \"read\", \"arguments\": \"just a string\"}\n</function_call>";
    auto res_scalar = parse_tool_calls(scalar_arg, read_tools());
    TEST_ASSERT(res_scalar.tool_calls.empty());

    // Valid array argument must be rejected (not an object)
    const std::string array_arg =
        "<function_call>\n{\"name\": \"read\", \"arguments\": [1, 2, 3]}\n</function_call>";
    auto res_array = parse_tool_calls(array_arg, read_tools());
    TEST_ASSERT(res_array.tool_calls.empty());

    // A scalar string wrapped in braces is still prose, not object arguments.
    const std::string braced_prose =
        "<function_call>\n"
        "{\"name\": \"read\", \"arguments\": \"{this is prose}\"}\n"
        "</function_call>";
    auto res_braced_prose = parse_tool_calls(braced_prose, read_tools());
    TEST_ASSERT(res_braced_prose.tool_calls.empty());

    // Keep forwarding a structurally object-like string with a JSON syntax
    // error so the client can report the exact bad arguments to the model.
    const std::string bad_obj_string =
        "<function_call>\n"
        "{\"name\": \"read\", \"arguments\": \"{\\\"offset\\\": 5o1}\"}\n"
        "</function_call>";
    auto res_bad_obj_string = parse_tool_calls(bad_obj_string, read_tools());
    TEST_ASSERT(res_bad_obj_string.tool_calls.size() == 1);
    if (!res_bad_obj_string.tool_calls.empty()) {
        TEST_ASSERT(res_bad_obj_string.tool_calls[0].arguments == "{\"offset\": 5o1}");
    }

    // Malformed JSON object syntax (e.g. 5o1) is forwarded as raw args
    const std::string bad_obj =
        "<function_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"/tmp/test\", \"offset\": 5o1}}\n</function_call>";
    auto res_bad = parse_tool_calls(bad_obj, read_tools());
    TEST_ASSERT(res_bad.tool_calls.size() == 1);
    if (!res_bad.tool_calls.empty()) {
        TEST_ASSERT(res_bad.tool_calls[0].name == "read");
        TEST_ASSERT(res_bad.tool_calls[0].arguments.find("5o1") != std::string::npos);
    }
}

TEST_CASE(ServerUnitFixture, test_extract_raw_json_tool_fallback_with_nested_name_argument) {
    // Malformed arguments containing a "name" key before top-level "name"
    const std::string text_reversed =
        "<function_call>\n"
        "{\"arguments\": {\"name\": \"evil_command\", \"offset\": 5o1}, \"name\": \"read\"}\n"
        "</function_call>";
    auto res_rev = parse_tool_calls(text_reversed, read_tools());
    TEST_ASSERT(res_rev.tool_calls.size() == 1);
    if (!res_rev.tool_calls.empty()) {
        TEST_ASSERT(res_rev.tool_calls[0].name == "read");
        TEST_ASSERT(res_rev.tool_calls[0].arguments.find("evil_command") != std::string::npos);
    }

    // Nested function object with "name" inside arguments
    const std::string text_nested =
        "<function_call>\n"
        "{\"function\": {\"name\": \"read\", \"arguments\": {\"name\": \"evil_nested\", \"offset\": 5o1}}}\n"
        "</function_call>";
    auto res_nest = parse_tool_calls(text_nested, read_tools());
    TEST_ASSERT(res_nest.tool_calls.size() == 1);
    if (!res_nest.tool_calls.empty()) {
        TEST_ASSERT(res_nest.tool_calls[0].name == "read");
        TEST_ASSERT(res_nest.tool_calls[0].arguments.find("evil_nested") != std::string::npos);
    }
}

TEST_CASE(ServerUnitFixture, test_extract_raw_json_tool_fallback_ignores_nested_metadata_name) {
    const std::string text =
        "<function_call>"
        "{\"function\":{\"name\":\"bash\",\"metadata\":{\"name\":\"read\"},"
        "\"arguments\":{\"command\":\"pwd\",\"offset\":5o1}}}"
        "</function_call>";

    auto res = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(res.tool_calls.size() == 1);
    if (!res.tool_calls.empty()) {
        TEST_ASSERT(res.tool_calls[0].name == "bash");
        TEST_ASSERT(res.tool_calls[0].arguments.find("5o1") != std::string::npos);
    }
}

TEST_CASE(ServerUnitFixture, test_raw_json_fallback_does_not_cross_tool_envelopes) {
    const std::string text =
        "{\"metadata\":{\"arguments\":{\"offset\":5o1}},"
        "\"tool_call\":{\"name\":\"read\",\"arguments\":{\"path\":\"x\"}}}";

    auto res = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(res.tool_calls.size() == 1);
    if (!res.tool_calls.empty()) {
        TEST_ASSERT(res.tool_calls[0].name == "read");
        const json args = json::parse(res.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "x");
        TEST_ASSERT(!args.contains("offset"));
    }
}

TEST_CASE(ServerUnitFixture, test_parse_json_tool_call_checks_later_envelope_siblings) {
    const std::string text =
        "{\"function\":{\"metadata\":true},"
        "\"tool_call\":{\"name\":\"read\",\"arguments\":{\"path\":\"x\"}}}";

    auto res = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(res.tool_calls.size() == 1);
    if (!res.tool_calls.empty()) {
        TEST_ASSERT(res.tool_calls[0].name == "read");
        TEST_ASSERT(json::parse(res.tool_calls[0].arguments)["path"] == "x");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_mixed_invoke_and_json_line_siblings) {
    const std::string text =
        "Processing files:\n"
        "<function_calls>\n"
        "  <invoke name=\"read\">\n"
        "    <param name=\"path\">first.go</param>\n"
        "  </invoke>\n"
        "  {\"name\": \"read\", \"arguments\": {\"path\": \"second.go\"}}\n"
        "</function_calls>";

    auto res = parse_tool_calls(text, read_tools());
    TEST_ASSERT(res.tool_calls.size() == 2);
    if (res.tool_calls.size() == 2) {
        TEST_ASSERT(res.tool_calls[0].name == "read");
        auto args0 = json::parse(res.tool_calls[0].arguments);
        TEST_ASSERT(args0["path"] == "first.go");

        TEST_ASSERT(res.tool_calls[1].name == "read");
        auto args1 = json::parse(res.tool_calls[1].arguments);
        TEST_ASSERT(args1["path"] == "second.go");
    }
    TEST_ASSERT(res.cleaned_text == "Processing files:");
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_empty_invoke_zero_args) {
    json tools = json::array({
        {
            {"type", "function"},
            {"function", {
                {"name", "get_version"},
                {"description", "get version"},
                {"parameters", {{"type", "object"}, {"properties", json::object()}}}
            }}
        }
    });

    const std::string text =
        "<function_calls>\n"
        "  <invoke name=\"get_version\"></invoke>\n"
        "</function_calls>";

    auto res = parse_tool_calls(text, tools);
    TEST_ASSERT(res.tool_calls.size() == 1);
    if (!res.tool_calls.empty()) {
        TEST_ASSERT(res.tool_calls[0].name == "get_version");
        TEST_ASSERT(res.tool_calls[0].arguments == "{}");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_with_token) {
    const std::string text =
        "Let me read the file.\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"read\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/home/dpavlin/koha-rfid-go/internal/rfidops/ops.go</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home/dpavlin/koha-rfid-go/internal/rfidops/ops.go");
    }
    TEST_ASSERT(result.cleaned_text == "Let me read the file.");
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_multiple_parameters) {
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la /tmp</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}},
                     {"timeout", {{"type", "integer"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "ls -la /tmp");
        TEST_ASSERT(args["timeout"] == 10);
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_edit_tool) {
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"edit\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/home/dpavlin/koha-rfid-go/server.go</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"edits\" string=\"false\">[{\"oldText\": \"err1\", \"newText\": \"err2\"}]</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "edit"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}},
                     {"edits", {{"type", "array"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "edit");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home/dpavlin/koha-rfid-go/server.go");
        TEST_ASSERT(args["edits"].is_array());
        TEST_ASSERT(args["edits"][0]["oldText"] == "err1");
        TEST_ASSERT(args["edits"][0]["newText"] == "err2");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_find_tool_syntax_start_arg_key_backtracking) {
    std::string text = "Let me edit the file:\nedit<arg_key>path</arg_key><arg_val>/tmp/test.txt</arg_val>";
    json tools = json::array({
        {{"type", "function"}, {"function", {{"name", "edit"}}}}
    });
    size_t pos = std::string::npos;
    TEST_ASSERT(find_tool_syntax_start(text, tools, pos));
    TEST_ASSERT(pos == text.find("edit<arg_key>"));
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_string_attribute_semantics) {
    // string="true" should preserve leading/trailing whitespace verbatim
    // string="false" should parse JSON numbers/booleans/arrays/objects
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"custom_tool\">\n"
        "<｜DSML｜parameter name=\"verbatim_str\" string=\"true\">  hello world  \n</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"json_num\" string=\"false\">42</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"json_bool\" string=\"false\">true</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"json_arr\" string=\"false\">[1, 2, 3]</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "custom_tool"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"verbatim_str", {{"type", "string"}}},
                     {"json_num", {{"type", "integer"}}},
                     {"json_bool", {{"type", "boolean"}}},
                     {"json_arr", {{"type", "array"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "custom_tool");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["verbatim_str"] == "  hello world  \n");
        TEST_ASSERT(args["json_num"] == 42);
        TEST_ASSERT(args["json_bool"] == true);
        TEST_ASSERT(args["json_arr"].is_array() && args["json_arr"].size() == 3);
    }
}
