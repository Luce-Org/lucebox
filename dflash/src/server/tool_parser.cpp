// Tool call parser implementation.
//
// Six detection patterns, tried in order:
// 1. <tool_call><function=NAME>...<parameter=K>V</parameter>...</function></tool_call>
// 2. <function=NAME>...params...</function>  (bare, outside tool_call)
// 3. <function=NAME(k="v", ...)></function>  (function-signature style)
// 4. <tool_code>{JSON}</tool_code>
// 5. Bare JSON objects with name+arguments fields
// 6. Native claude-code XML tags: <bash>CMD</bash>, <read>PATH</read>, etc.

#include "tool_parser.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>

namespace dflash::common {

// ─── Helpers ────────────────────────────────────────────────────────────

static std::string generate_call_id() {
    static std::mutex rng_mu;
    static std::mt19937_64 rng(std::random_device{}());
    static const char hex[] = "0123456789abcdef";
    std::string id = "call_";
    std::lock_guard<std::mutex> lk(rng_mu);
    for (int i = 0; i < 24; i++) {
        id += hex[rng() % 16];
    }
    return id;
}

// Check if a function name is in the allowed tools list.
static bool tool_allowed(const json & tools, const std::string & name) {
    if (tools.is_null() || !tools.is_array() || tools.empty()) return true;
    for (const auto & t : tools) {
        const auto & fn = t.contains("function") ? t["function"] : t;
        if (fn.is_object() && fn.value("name", "") == name) return true;
    }
    return false;
}

// Find parameter schema properties for a function.
static json find_tool_properties(const json & tools, const std::string & name) {
    if (tools.is_null() || !tools.is_array()) return json::object();
    for (const auto & t : tools) {
        const auto & fn = t.contains("function") ? t["function"] : t;
        if (!fn.is_object() || fn.value("name", "") != name) continue;
        if (fn.contains("parameters") && fn["parameters"].is_object()) {
            const auto & params = fn["parameters"];
            if (params.contains("properties") && params["properties"].is_object()) {
                return params["properties"];
            }
        }
    }
    return json::object();
}

// Convert a string value to its JSON-schema-typed equivalent.
static json convert_param_value(const std::string & val, const std::string & key,
                                const json & props) {
    if (val == "null") return nullptr;
    if (!props.contains(key)) return val;

    const auto & cfg = props[key];
    std::string ptype = "string";
    if (cfg.is_object() && cfg.contains("type")) {
        ptype = cfg["type"].get<std::string>();
    }

    // string types
    if (ptype == "string" || ptype == "str" || ptype == "enum") return val;

    // integer types
    if (ptype.substr(0, 3) == "int" || ptype == "integer") {
        try { return std::stol(val); } catch (...) { return val; }
    }

    // number / float
    if (ptype == "number" || ptype.substr(0, 5) == "float") {
        try {
            double f = std::stod(val);
            if (f == (double)(long)f) return (long)f;
            return f;
        } catch (...) { return val; }
    }

    // boolean
    if (ptype == "boolean" || ptype == "bool") {
        std::string lower = val;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower == "true";
    }

    // object / array — try JSON parse
    if (ptype == "object" || ptype == "array") {
        try { return json::parse(val); } catch (...) { return val; }
    }

    // fallback: try JSON parse, then return as string
    try { return json::parse(val); } catch (...) { return val; }
}

// ─── Removal tracking ───────────────────────────────────────────────────

struct Span {
    size_t start, end;
};

static bool overlaps(const std::vector<Span> & spans, size_t pos) {
    for (const auto & s : spans) {
        if (s.start <= pos && pos < s.end) return true;
    }
    return false;
}

static size_t include_preceding_tool_call_open(const std::string & text, size_t pos) {
    size_t wrapper = text.rfind("<tool_call>", pos);
    if (wrapper == std::string::npos) return pos;
    for (size_t i = wrapper + std::strlen("<tool_call>"); i < pos; i++) {
        char c = text[i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return pos;
    }
    return wrapper;
}

// ─── Pattern regexes ────────────────────────────────────────────────────

// We use std::regex for portability. Compiled once (function-local static).

static const std::regex & re_tool_call_complete() {
    static std::regex r(R"(<tool_call>([\s\S]*?)</tool_call>)");
    return r;
}

static const std::regex & re_tool_call_function() {
    static std::regex r(R"(<function=([\s\S]*?)</function>|<function=([\s\S]*)$)");
    return r;
}

static const std::regex & re_tool_call_parameter() {
    static std::regex r(R"(<parameter=([\s\S]*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$))");
    return r;
}

static const std::regex & re_bare_function_xml() {
    static std::regex r(R"(<function=([A-Za-z_][\w.\-]*?)>([\s\S]*?)</function>(?:\s*</tool_call>)?)");
    return r;
}

static const std::regex & re_function_signature() {
    static std::regex r(R"(<function=([A-Za-z_][\w.\-]*?)\(([\s\S]*?)\)</function>)");
    return r;
}

static const std::regex & re_tool_code() {
    static std::regex r(R"(<tool_code>([\s\S]*?)</tool_code>)");
    return r;
}

// Pattern 6: native claude-code XML tags.
// Matches <bash>BODY</bash>, <read>BODY</read>, etc.
static const std::regex & re_native_tag() {
    static std::regex r(R"(<(bash|read|write|edit|ls|grep|glob)>([\s\S]*?)</\1>)",
                        std::regex::icase);
    return r;
}

// ─── Parameter-name alias resolution ────────────────────────────────────

// Some quantized models (e.g. Qwen3.6-Q3) emit short forms of canonical
// parameter names (cmd instead of command, path instead of file_path).
// Map the emitted key to the schema's actual key when an alias is found.
// Pure helper — returns the original `emitted` if no alias matches.
static std::string resolve_param_alias(const std::string & emitted, const json & props) {
    if (!props.is_object() || props.empty()) return emitted;

    std::string lower = emitted;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // 1. Direct case-insensitive match against schema keys.
    for (auto it = props.begin(); it != props.end(); ++it) {
        std::string name = it.key();
        std::string lname = name;
        std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
        if (lname == lower) return name;
    }

    // 2. Alias map: common shortenings <=> canonical forms.
    static const std::vector<std::pair<std::string, std::vector<std::string>>> aliases = {
        {"cmd",        {"command"}},
        {"command",    {"cmd"}},
        {"path",       {"file_path", "directory", "dir"}},
        {"file_path",  {"path", "file"}},
        {"file",       {"file_path", "path"}},
        {"filepath",   {"file_path"}},
        {"dir",        {"directory", "path"}},
        {"directory",  {"dir", "path"}},
        {"query",      {"pattern", "q"}},
        {"pattern",    {"query", "regex"}},
        {"regex",      {"pattern"}},
        {"q",          {"query", "pattern"}},
        {"expr",       {"expression"}},
        {"expression", {"expr"}},
        {"text",       {"content"}},
        {"content",    {"text"}},
        {"src",        {"source"}},
        {"source",     {"src"}},
        {"dst",        {"destination", "target"}},
        {"destination", {"dst", "target"}},
        {"target",     {"dst", "destination"}},
    };

    for (const auto & [key, candidates] : aliases) {
        if (key != lower) continue;
        for (const std::string & candidate : candidates) {
            for (auto pit = props.begin(); pit != props.end(); ++pit) {
                std::string pname = pit.key();
                std::string lpname = pname;
                std::transform(lpname.begin(), lpname.end(), lpname.begin(), ::tolower);
                if (lpname == candidate) return pname;
            }
        }
        break;
    }

    return emitted;  // no alias matched; keep as-is
}

// ─── XML parameter parser ───────────────────────────────────────────────

static json parse_xml_params(const std::string & region, const std::string & fn_name,
                             const json & tools) {
    json props = find_tool_properties(tools, fn_name);
    json args = json::object();

    auto begin = std::sregex_iterator(region.begin(), region.end(), re_tool_call_parameter());
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        std::string match_text = (*it)[1].str();
        size_t eq = match_text.find('>');
        if (eq == std::string::npos) continue;
        std::string k = match_text.substr(0, eq);
        // trim whitespace from key
        while (!k.empty() && k.back() == ' ') k.pop_back();
        while (!k.empty() && k.front() == ' ') k.erase(k.begin());

        std::string v = match_text.substr(eq + 1);
        if (!v.empty() && v.front() == '\n') v.erase(v.begin());
        if (!v.empty() && v.back() == '\n') v.pop_back();

        std::string canonical_k = resolve_param_alias(k, props);
        args[canonical_k] = convert_param_value(v, canonical_k, props);
    }
    return args;
}

// ─── JSON tool call parser ──────────────────────────────────────────────

// Parse {"name": ..., "arguments": ...} or {"function": {"name": ..., "arguments": ...}}
static bool parse_json_tool_call(const json & obj, std::string & out_name, json & out_args) {
    if (!obj.is_object()) return false;

    std::string name;
    json args;

    if (obj.contains("name") && obj["name"].is_string()) {
        name = obj["name"].get<std::string>();
        if (obj.contains("arguments")) {
            if (obj["arguments"].is_object()) {
                args = obj["arguments"];
            } else if (obj["arguments"].is_string()) {
                try { args = json::parse(obj["arguments"].get<std::string>()); }
                catch (...) { return false; }
            } else {
                return false;
            }
        }
    } else if (obj.contains("function") && obj["function"].is_object()) {
        const auto & fn = obj["function"];
        if (!fn.contains("name") || !fn["name"].is_string()) return false;
        name = fn["name"].get<std::string>();
        if (fn.contains("arguments")) {
            if (fn["arguments"].is_object()) {
                args = fn["arguments"];
            } else if (fn["arguments"].is_string()) {
                try { args = json::parse(fn["arguments"].get<std::string>()); }
                catch (...) { return false; }
            } else {
                return false;
            }
        }
    } else {
        return false;
    }

    if (name.empty() || !args.is_object()) return false;
    out_name = name;
    out_args = args;
    return true;
}

// ─── Function signature parser ──────────────────────────────────────────

// Parse key=value pairs from `<function=name(k="v", k2=123)></function>`.
// Simplified: we parse key="string" and key=number/bool/null pairs.
static bool parse_function_sig_args(const std::string & arg_text, json & out_args) {
    out_args = json::object();
    if (arg_text.empty()) return true;

    size_t pos = 0;
    while (pos < arg_text.size()) {
        // Skip whitespace and commas
        while (pos < arg_text.size() && (arg_text[pos] == ' ' || arg_text[pos] == ',' ||
               arg_text[pos] == '\n' || arg_text[pos] == '\r' || arg_text[pos] == '\t'))
            pos++;
        if (pos >= arg_text.size()) break;

        // key
        size_t eq = arg_text.find('=', pos);
        if (eq == std::string::npos) return false;
        std::string key = arg_text.substr(pos, eq - pos);
        while (!key.empty() && key.back() == ' ') key.pop_back();
        if (key.empty()) return false;
        pos = eq + 1;

        // skip whitespace after =
        while (pos < arg_text.size() && arg_text[pos] == ' ') pos++;
        if (pos >= arg_text.size()) return false;

        // value
        if (arg_text[pos] == '"' || arg_text[pos] == '\'') {
            char quote = arg_text[pos];
            pos++;
            std::string val;
            while (pos < arg_text.size() && arg_text[pos] != quote) {
                if (arg_text[pos] == '\\' && pos + 1 < arg_text.size()) {
                    val += arg_text[pos + 1];
                    pos += 2;
                } else {
                    val += arg_text[pos];
                    pos++;
                }
            }
            if (pos < arg_text.size()) pos++;  // skip closing quote
            out_args[key] = val;
        } else {
            // non-string value — read until comma or end
            size_t end = pos;
            int depth = 0;
            while (end < arg_text.size()) {
                char c = arg_text[end];
                if (c == '(' || c == '[' || c == '{') depth++;
                else if (c == ')' || c == ']' || c == '}') {
                    if (depth == 0) break;
                    depth--;
                }
                else if (c == ',' && depth == 0) break;
                end++;
            }
            std::string raw = arg_text.substr(pos, end - pos);
            while (!raw.empty() && raw.back() == ' ') raw.pop_back();
            pos = end;

            // Try to parse as JSON literal
            try {
                out_args[key] = json::parse(raw);
            } catch (...) {
                out_args[key] = raw;
            }
        }
    }
    return true;
}

// ─── Native tag helpers ─────────────────────────────────────────────────

// Case-insensitive lookup of `tag` in the tools array.
// Returns the tool's canonical name if found, otherwise returns `tag` as-is.
static std::string lookup_tool_name(const std::string & tag, const json & tools) {
    if (tools.is_null() || !tools.is_array() || tools.empty()) return tag;

    std::string lower_tag = tag;
    std::transform(lower_tag.begin(), lower_tag.end(), lower_tag.begin(), ::tolower);

    for (const auto & t : tools) {
        const auto & fn = t.contains("function") ? t["function"] : t;
        if (!fn.is_object()) continue;
        std::string name = fn.value("name", "");
        if (name.empty()) continue;
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        if (lower_name == lower_tag) return name;
    }
    return tag;  // no match → lowercase tag name
}

// Map native tag name to its default argument key + body.
static json tag_to_args(const std::string & tag, const std::string & body) {
    json args = json::object();
    std::string lower = tag;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "bash")        args["command"] = body;
    else if (lower == "read")   args["file_path"] = body;
    else if (lower == "grep")   args["pattern"] = body;
    else if (lower == "glob")   args["pattern"] = body;
    else if (lower == "write")  args["content"] = body;
    else if (lower == "edit")   args["content"] = body;
    else if (lower == "ls") {
        if (!body.empty()) args["path"] = body;
    } else {
        args["content"] = body;  // unknown tag fallback
    }
    return args;
}

// ─── Main parser ────────────────────────────────────────────────────────

ToolParseResult parse_tool_calls(const std::string & text, const json & tools) {
    ToolParseResult result;
    std::vector<Span> removals;

    auto add_call = [&](const std::string & fn_name, const json & args,
                        size_t start, size_t end) {
        if (!tool_allowed(tools, fn_name)) return;
        ToolCall tc;
        tc.id = generate_call_id();
        tc.name = fn_name;
        tc.arguments = args.dump();
        result.tool_calls.push_back(std::move(tc));
        removals.push_back({start, end});
    };

    // Pattern 1: <tool_call>...<function=NAME>...params...</function>...</tool_call>
    {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_tool_call_complete());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string body = (*it)[1].str();
            std::smatch fn_match;
            if (!std::regex_search(body, fn_match, re_tool_call_function())) continue;
            std::string fn_text = fn_match[1].matched ? fn_match[1].str() : fn_match[2].str();
            size_t gt = fn_text.find('>');
            if (gt == std::string::npos) continue;
            std::string fn_name = fn_text.substr(0, gt);
            while (!fn_name.empty() && fn_name.back() == ' ') fn_name.pop_back();
            while (!fn_name.empty() && fn_name.front() == ' ') fn_name.erase(fn_name.begin());
            std::string params_region = fn_text.substr(gt + 1);

            add_call(fn_name, parse_xml_params(params_region, fn_name, tools),
                     it->position(), it->position() + it->length());
        }
    }

    // Pattern 2: <function=NAME>...</function> (bare, not inside tool_call)
    {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_bare_function_xml());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            size_t pos = it->position();
            if (overlaps(removals, pos)) continue;
            std::string fn_name = (*it)[1].str();
            std::string params = (*it)[2].str();
            size_t removal_start = include_preceding_tool_call_open(text, pos);
            add_call(fn_name, parse_xml_params(params, fn_name, tools),
                     removal_start, pos + it->length());
        }
    }

    // Pattern 3: <function=NAME(args)></function>
    {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_function_signature());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            size_t pos = it->position();
            if (overlaps(removals, pos)) continue;
            json args;
            if (!parse_function_sig_args((*it)[2].str(), args)) continue;
            add_call((*it)[1].str(), args, pos, pos + it->length());
        }
    }

    // Pattern 4: <tool_code>{JSON}</tool_code>
    {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_tool_code());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            size_t pos = it->position();
            if (overlaps(removals, pos)) continue;
            std::string inner = (*it)[1].str();
            // trim
            size_t s = inner.find_first_not_of(" \t\n\r");
            if (s != std::string::npos) inner = inner.substr(s);
            size_t e = inner.find_last_not_of(" \t\n\r");
            if (e != std::string::npos) inner = inner.substr(0, e + 1);
            try {
                json obj = json::parse(inner);
                std::string name;
                json args;
                if (parse_json_tool_call(obj, name, args)) {
                    size_t pos = it->position();
                    add_call(name, args, pos, pos + it->length());
                }
            } catch (...) {}
        }
    }

    // Pattern 5: Bare JSON objects
    {
        size_t cursor = 0;
        while (cursor < text.size()) {
            size_t start = text.find('{', cursor);
            if (start == std::string::npos) break;
            if (overlaps(removals, start)) {
                cursor = start + 1;
                continue;
            }
            // Find balanced braces first to extract exact JSON boundaries.
            int depth = 0;
            size_t end_pos = start;
            bool in_string = false;
            for (size_t i = start; i < text.size(); i++) {
                char c = text[i];
                if (in_string) {
                    if (c == '\\') { i++; continue; }
                    if (c == '"') in_string = false;
                    continue;
                }
                if (c == '"') { in_string = true; continue; }
                if (c == '{') depth++;
                else if (c == '}') {
                    depth--;
                    if (depth == 0) { end_pos = i + 1; break; }
                }
            }
            if (end_pos <= start) {
                cursor = start + 1;
                continue;
            }

            // Parse the exact brace-balanced substring.
            std::string json_str = text.substr(start, end_pos - start);
            json obj2 = json::parse(json_str, nullptr, false);
            if (obj2.is_discarded()) {
                cursor = start + 1;
                continue;
            }

            std::string name;
            json args;
            if (parse_json_tool_call(obj2, name, args)) {
                add_call(name, args, start, end_pos);
            }
            cursor = end_pos;
        }
    }

    // Pattern 6: native claude-code XML tags (<bash>, <read>, <write>, <edit>, <ls>, <grep>, <glob>)
    {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_native_tag());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            size_t pos = it->position();
            if (overlaps(removals, pos)) continue;
            std::string tag  = (*it)[1].str();
            std::string body = (*it)[2].str();
            // Strip leading/trailing newline (consistent with parameter parser at line 172).
            if (!body.empty() && body.front() == '\n') body.erase(body.begin());
            if (!body.empty() && body.back()  == '\n') body.pop_back();

            std::string canonical = lookup_tool_name(tag, tools);
            add_call(canonical, tag_to_args(tag, body), pos, pos + it->length());
        }
    }

    // Build cleaned text by removing all matched spans
    if (removals.empty()) {
        result.cleaned_text = text;
    } else {
        // Sort and deduplicate spans
        std::sort(removals.begin(), removals.end(),
                  [](const Span & a, const Span & b) { return a.start < b.start; });

        std::string cleaned;
        size_t cursor = 0;
        for (const auto & span : removals) {
            if (span.start < cursor) continue;
            cleaned += text.substr(cursor, span.start - cursor);
            cursor = span.end;
        }
        cleaned += text.substr(cursor);

        // Trim
        size_t s = cleaned.find_first_not_of(" \t\n\r");
        size_t e = cleaned.find_last_not_of(" \t\n\r");
        if (s != std::string::npos && e != std::string::npos) {
            result.cleaned_text = cleaned.substr(s, e - s + 1);
        } else {
            result.cleaned_text.clear();
        }
    }

    return result;
}

}  // namespace dflash::common
