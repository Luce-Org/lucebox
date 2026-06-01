// Tool call parser implementation.
//
// Six detection patterns, tried in order:
// 1. <tool_call><function=NAME>...<parameter=K>V</parameter>...</function></tool_call>
// 2. <function=NAME>...params...</function>  (bare, outside tool_call)
// 3. <function=NAME(k="v", ...)></function>  (function-signature style)
// 4. <tool_code>{JSON}</tool_code>
// 5. call:<ns>?<verb>{relaxed-JSON args}    (gemma plain-text emissions)
// 6. Bare JSON objects with name+arguments fields
//
// Pattern 5 runs *before* pattern 6 so that args like
//   call:outer{"name": "inner", "arguments": {}}
// don't get hijacked by the bare-JSON sweep into a spurious `inner` tool
// call. The brace-balanced span pattern 5 records in `removals` shadows
// the inner JSON from pattern 6's view via `overlaps()`.

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

// Pattern 5: `call:<ns>?<verb>{` opener. The sentinel alternation in front
// rejects narrative usages like "I'll call:foo{x:1}" where `call:` is glued
// to a preceding word — whitespace, common punctuation, and open/close
// brackets are the realistic boundaries seen in the snapshot data. `\s`
// covers `\n` so a `call:` at the start of any line is matched without
// relying on std::regex multiline support (which is non-portable).
//
// Note that `}` is in the sentinel list — gemma frequently emits multiple
// invocations back-to-back: `call:a{x:1}call:b{y:2}`. Without `}` as a
// sentinel the second match would be missed.
//
// `_` is also in the sentinel list to handle a SentencePiece / chat-template
// artifact: post-bragi-channel-routing (commit 4b757d1) the gemma server
// occasionally emits raw tokens like `_call:get_country_info{...}` where
// the leading `_` is residual tokenizer serialization. Without `_` here
// the parser misses every such invocation — empirically confirmed against
// gemma-4-26b 2026-05-31 smoke test. Tradeoff: `my_call:foo{}` mid-
// identifier could match, but real model output doesn't emit `my_call:`
// strings (tool names come from the request's tool definitions).
static const std::regex & re_call_verb_open() {
    static std::regex r(R"((^|[\s,;:\(\[\{\}\)\]\>_])call:([A-Za-z0-9_.:\-]+)\s*\{)");
    return r;
}

// Find the index one past the `}` that matches `text[open] == '{'`.
// Respects nested {}/[] depth and skips over "..." / '...' / `...`
// string literals (with backslash escapes). Returns std::string::npos if
// no matching close is found.
static size_t balanced_braces_end(const std::string & text, size_t open) {
    int depth = 0;
    char in_str = 0;  // 0, or one of '"', '\'', '`'
    for (size_t i = open; i < text.size(); i++) {
        char c = text[i];
        if (in_str) {
            if (c == '\\' && i + 1 < text.size()) { i++; continue; }
            if (c == in_str) in_str = 0;
            continue;
        }
        if (c == '"' || c == '\'' || c == '`') { in_str = c; continue; }
        if (c == '{' || c == '[') {
            depth++;
        } else if (c == '}' || c == ']') {
            depth--;
            if (depth == 0 && c == '}') return i + 1;
            if (depth < 0) return std::string::npos;
        }
    }
    return std::string::npos;
}

// Try strict json::parse first; on failure rewrite single- and
// backtick-quoted strings to double-quoted, wrap bare identifier keys
// in double quotes, and retry. Returns true and populates `out` on
// success; returns false on irrecoverable failure (and `out` is unset).
//
// The rewrite walks the buffer char-by-char tracking string state so it
// doesn't mangle identifiers that live inside string values.
static bool coerce_relaxed_json(const std::string & payload, json & out) {
    {
        json parsed = json::parse(payload, nullptr, false);
        if (!parsed.is_discarded()) {
            out = std::move(parsed);
            return true;
        }
    }

    // Permissive pass.
    static const std::regex re_bare_key(R"(([A-Za-z_][A-Za-z0-9_]*)(\s*:))");

    std::string rewritten;
    rewritten.reserve(payload.size() + 16);
    char in_str = 0;  // 0, or the *opening* quote we saw
    for (size_t i = 0; i < payload.size(); ) {
        char c = payload[i];
        if (in_str) {
            // Inside a string we already opened. Mirror escapes verbatim.
            if (c == '\\' && i + 1 < payload.size()) {
                rewritten += c;
                rewritten += payload[i + 1];
                i += 2;
                continue;
            }
            if (c == in_str) {
                // Close — always emit a double-quote regardless of which
                // quote style opened the string. The opening side already
                // emitted a `"`.
                rewritten += '"';
                in_str = 0;
                i++;
                continue;
            }
            rewritten += c;
            i++;
            continue;
        }
        if (c == '"' || c == '\'' || c == '`') {
            rewritten += '"';
            in_str = c;
            i++;
            continue;
        }
        // Try to match a bare-key identifier here. Don't fire if the
        // previous emitted char is `"` — that would indicate we're sitting
        // right after a JSON string boundary and the "identifier" is
        // probably part of a value continuation (e.g. `"k": foo: 1` would
        // be malformed JSON anyway, but better to leave it untouched).
        std::smatch m;
        std::string tail = payload.substr(i);
        if (std::regex_search(tail, m, re_bare_key,
                              std::regex_constants::match_continuous) &&
            (rewritten.empty() || rewritten.back() != '"')) {
            rewritten += '"';
            rewritten += m[1].str();
            rewritten += '"';
            rewritten += m[2].str();
            i += m.length();
            continue;
        }
        rewritten += c;
        i++;
    }

    json parsed = json::parse(rewritten, nullptr, false);
    if (parsed.is_discarded()) return false;
    out = std::move(parsed);
    return true;
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

        args[k] = convert_param_value(v, k, props);
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

    // Pattern 5: call:<ns>?<verb>{relaxed-JSON args}
    //
    // Runs before the bare-JSON sweep so that inner JSON of the form
    //   call:outer{"name": "inner", "arguments": {}}
    // doesn't get hijacked into a spurious `inner` ToolCall.
    {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_call_verb_open());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            // Group 1: sentinel char (may be empty if matched at `^`).
            // Group 2: full verb including any embedded namespaces.
            size_t prefix_len = (*it)[1].matched ? (*it)[1].length() : 0;
            size_t call_start = it->position() + prefix_len;
            if (overlaps(removals, call_start)) continue;

            // The matched substring runs from call_start through the `{`
            // (consuming the opener and any whitespace between verb and
            // brace). Compute the brace index from the match end.
            size_t brace_open = it->position() + it->length() - 1;
            if (brace_open >= text.size() || text[brace_open] != '{') continue;

            size_t brace_close = balanced_braces_end(text, brace_open);
            if (brace_close == std::string::npos) continue;

            std::string raw_args = text.substr(brace_open, brace_close - brace_open);
            json args;
            if (!coerce_relaxed_json(raw_args, args)) continue;
            if (!args.is_object()) continue;

            std::string verb = (*it)[2].str();
            size_t colon = verb.find_last_of(':');
            if (colon != std::string::npos) verb = verb.substr(colon + 1);
            if (verb.empty()) continue;

            add_call(verb, args, call_start, brace_close);
        }
    }

    // Pattern 6: Bare JSON objects
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
