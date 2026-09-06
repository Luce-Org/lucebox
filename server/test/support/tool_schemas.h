#pragma once
#include <nlohmann/json.hpp>
using json = nlohmann::json;

static json weather_tools() {
    return json::array({
        {{"type", "function"},
         {"function", {
             {"name", "get_weather"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"location", {{"type", "string"}}},
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}},
        {{"type", "function"},
         {"function", {
             {"name", "terminal"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
}

static json shell_tools() {
    return json::array({
        {
            {"name", "shell"},
            {"description", "Run one read-only shell command in the repository."},
            {"input_schema", {
                {"type", "object"},
                {"properties", {
                    {"command", {
                        {"type", "string"},
                        {"description", "The shell command to run."}
                    }}
                }},
                {"required", json::array({"command"})},
                {"additionalProperties", false}
            }}
        }
    });
}

static json bash_tools() {
    json tools = shell_tools();
    tools[0]["name"] = "bash";
    return tools;
}

static json read_tools() {
    return json::array({
        {{"type", "function"},
         {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}},
                     {"offset", {{"type", "integer"}}},
                     {"limit", {{"type", "integer"}}}
                 }},
                 {"required", json::array({"path"})}
             }}
         }}}
    });
}

static json read_and_bash_tools() {
    json tools = read_tools();
    tools.push_back(bash_tools()[0]);
    return tools;
}

static json edit_tools() {
    return json::array({
        {{"type", "function"},
         {"function", {
             {"name", "edit"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"edits", {
                         {"type", "array"},
                         {"items", {{"type", "object"}}}
                     }}
                 }},
                 {"required", json::array({"edits"})}
             }}
         }}}
    });
}

static json optional_shell_tools() {
    json tools = shell_tools();
    tools[0]["input_schema"].erase("required");
    return tools;
}
