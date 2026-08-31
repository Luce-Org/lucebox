// Tool call parser implementation.
//
// Seven detection patterns, tried in order:
// 1. <tool_call><function=NAME>...<parameter=K>V</parameter>...</function></tool_call>
// 2. <function=NAME>...params...</function>  (bare, outside tool_call)
// 3. <function=NAME(k="v", ...)></function>  (function-signature style)
// 4. <tool_code>{JSON}</tool_code>
// 5. call:<ns>?<verb>{relaxed-JSON args}    (gemma plain-text emissions)
// 6. Bare JSON objects with name+arguments fields
// 7. Whole-response JSON args for exactly one declared tool
// 8. <TOOL_NAME>...<parameter=K>V</parameter>...</function>  (bare tool tag)
// 9. <parameter name="TOOL"><parameter name="K">V</parameter></function>
// 10. <funcname>TOOL<parameter=K>V</parameter>...</function>
// 11. <function TOOL><parameter=K>V</parameter>...</function>
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

static std::string trim_ws(const std::string & v) {
    const char * ws = " \t\r\n";
    const size_t a = v.find_first_not_of(ws);
    if (a == std::string::npos) return std::string();
    const size_t b = v.find_last_not_of(ws);
    return v.substr(a, b - a + 1);
}

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

static const char TOOL_OPEN[] = "<tool_call>";
static const char TOOL_CALLS_OPEN[] = "<tool_calls>";
static const char DSML_PREFIX[] = "<｜DSML｜";
static const char INVOKE_OPEN[] = "<invoke";
static const char FUNCTION_CALL_OPEN[] = "<function_call>";
static const char FUNCTION_CALLS_OPEN[] = "<function_calls>";
static const char FUNCTION_OPEN[] = "<function=";
static const char BARE_FUNCTION_OPEN[] = "<function>";
static const char FUNCTION_SPACE_OPEN[] = "<function ";
static const char FUNCNAME_OPEN[] = "<funcname>";
static const char TOOL_CODE_OPEN[] = "<tool_code>";
static const char ATTRIBUTE_PARAMETER_OPEN[] = "<parameter name=";
static const char ARG_KEY_OPEN[] = "<arg_key>";



static bool valid_tool_name(const std::string & name) {
    if (name.empty() || name.size() > 64) return false;
    auto is_alpha_or_underscore = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    };
    if (!is_alpha_or_underscore(name.front())) return false;
    for (const char c : name) {
        if (!is_alpha_or_underscore(c) && !(c >= '0' && c <= '9') &&
            c != '.' && c != '-') {
            return false;
        }
    }
    return true;
}

static std::string declared_tool_name(const json & tool) {
    const auto & fn = tool.is_object() && tool.contains("function")
        ? tool["function"] : tool;
    const std::string name = fn.is_object()
        ? fn.value("name", "") : std::string();
    return valid_tool_name(name) ? name : std::string();
}

static bool declared_tool_open_at(const std::string & text, size_t pos,
                                  const json & tools) {
    if (!tools.is_array()) return false;
    for (const auto & tool : tools) {
        const std::string name = declared_tool_name(tool);
        if (name.empty()) continue;
        const std::string opener = "<" + name + ">";
        if (text.compare(pos, opener.size(), opener) == 0) return true;
    }
    return false;
}

bool find_tool_syntax_start(const std::string & text, const json & tools,
                            size_t & pos) {
    size_t idx = text.find('<');
    while (idx != std::string::npos) {
        if (text.compare(idx, sizeof(TOOL_OPEN) - 1, TOOL_OPEN) == 0 ||
            text.compare(idx, sizeof(TOOL_CALLS_OPEN) - 1, TOOL_CALLS_OPEN) == 0 ||
            text.compare(idx, sizeof(DSML_PREFIX) - 1, DSML_PREFIX) == 0 ||
            text.compare(idx, sizeof(INVOKE_OPEN) - 1, INVOKE_OPEN) == 0 ||
            text.compare(idx, sizeof(FUNCTION_CALL_OPEN) - 1, FUNCTION_CALL_OPEN) == 0 ||
            text.compare(idx, sizeof(FUNCTION_CALLS_OPEN) - 1, FUNCTION_CALLS_OPEN) == 0 ||
            text.compare(idx, sizeof(FUNCTION_OPEN) - 1, FUNCTION_OPEN) == 0 ||
            text.compare(idx, sizeof(BARE_FUNCTION_OPEN) - 1, BARE_FUNCTION_OPEN) == 0 ||
            text.compare(idx, sizeof(FUNCTION_SPACE_OPEN) - 1,
                         FUNCTION_SPACE_OPEN) == 0 ||
            text.compare(idx, sizeof(FUNCNAME_OPEN) - 1, FUNCNAME_OPEN) == 0 ||
            text.compare(idx, sizeof(TOOL_CODE_OPEN) - 1, TOOL_CODE_OPEN) == 0 ||
            text.compare(idx, sizeof(ATTRIBUTE_PARAMETER_OPEN) - 1,
                         ATTRIBUTE_PARAMETER_OPEN) == 0 ||
            declared_tool_open_at(text, idx, tools)) {
            pos = idx;
            return true;
        }
        if (text.compare(idx, sizeof(ARG_KEY_OPEN) - 1, ARG_KEY_OPEN) == 0) {
            size_t start = idx;
            auto is_ident = [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '-';
            };
            while (start > 0 && is_ident(text[start - 1])) start--;
            if (start < idx) {
                pos = start;
                return true;
            }
        }
        // Check for bare tool tag without angle brackets if it follows a newline
        // or start of text, e.g. "tool_name\n<arg_key>..."
        if (idx == 0 || text[idx - 1] == '\n') {
            size_t start = idx;
            auto is_ident = [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '-';
            };
            while (start > 0 && is_ident(text[start - 1])) start--;
            if (start < idx) {
                pos = start;
                return true;
            }
        }
        idx = text.find('<', idx + 1);
    }
    return false;
}

size_t tool_syntax_holdback(const json & tools) {
    size_t holdback = std::max({(size_t)21,
                                sizeof(ATTRIBUTE_PARAMETER_OPEN) - 2,
                                sizeof(FUNCTION_CALLS_OPEN) - 2,
                                sizeof(FUNCTION_CALL_OPEN) - 2,
                                sizeof(TOOL_CALLS_OPEN) - 2,
                                sizeof(TOOL_OPEN) - 2,
                                sizeof(BARE_FUNCTION_OPEN) - 2});
    if (!tools.is_array()) return holdback;
    for (const auto & tool : tools) {
        const std::string name = declared_tool_name(tool);
        if (!name.empty()) {
            // Retain all but the final `>` of `<NAME>`.
            holdback = std::max(holdback, name.size() + 1);
        }
    }
    return holdback;
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
        if (fn.contains("input_schema") && fn["input_schema"].is_object()) {
            const auto & params = fn["input_schema"];
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
    if (!props.contains(key)) return val == "null" ? nullptr : json(val);

    const auto & cfg = props[key];
    std::string ptype = "string";
    if (cfg.is_object() && cfg.contains("type")) {
        const auto & t = cfg["type"];
        if (t.is_string()) {
            ptype = t.get<std::string>();
        } else if (t.is_array()) {
            // JSON Schema allows "type": ["string","null"]; take the first
            // non-null string entry instead of throwing.
            for (const auto & e : t) {
                if (e.is_string() && e.get<std::string>() != "null") {
                    ptype = e.get<std::string>();
                    break;
                }
            }
        }
    }

    // string types
    if (ptype == "string" || ptype == "str" || ptype == "enum") return val;
    if (val == "null") return nullptr;

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

static size_t include_following_tool_call_close(const std::string & text,
                                                size_t pos,
                                                const std::string & fn_name = {}) {
    size_t close_pos = pos;
    while (close_pos < text.size()) {
        const char c = text[close_pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        close_pos++;
    }
    static constexpr char TOOL_CALL_CLOSE[] = "</tool_call>";
    if (text.compare(close_pos, sizeof(TOOL_CALL_CLOSE) - 1,
                     TOOL_CALL_CLOSE) == 0) {
        return close_pos + sizeof(TOOL_CALL_CLOSE) - 1;
    }
    if (!fn_name.empty()) {
        const std::string tool_close = "</" + fn_name + ">";
        if (text.compare(close_pos, tool_close.size(), tool_close) == 0) {
            return close_pos + tool_close.size();
        }
    }
    return pos;
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

static const std::regex & re_bare_tool_name_xml() {
    static std::regex r(R"(<([A-Za-z_][\w.\-]*?)>([\s\S]*?)(?:</function>|</\1>))");
    return r;
}

static const std::regex & re_attribute_tool_xml() {
    static std::regex r(
        R"re(<parameter\s+name\s*=\s*"([A-Za-z_][\w.\-]*)"\s*>([\s\S]*?)</function>)re");
    return r;
}

static const std::regex & re_attribute_parameter_xml() {
    static std::regex r(
        R"re(<parameter\s+name\s*=\s*"([A-Za-z_][\w.\-]*)"\s*>([\s\S]*?)</parameter>)re");
    return r;
}

static const std::regex & re_funcname_tool_xml() {
    static std::regex r(
        R"(<funcname>\s*([A-Za-z_][\w.\-]*)\s*([\s\S]*?)</function>)");
    return r;
}

static const std::regex & re_space_function_xml() {
    static std::regex r(
        R"(<function\s+([A-Za-z_][\w.\-]*)\s*>([\s\S]*?)</function>(?:\s*</tool_call>)?)");
    return r;
}

static const std::regex & re_complete_parameter_xml() {
    static std::regex r(
        R"(<parameter=([A-Za-z_][\w.\-]*)>([\s\S]*?)</parameter>)");
    return r;
}

static const std::regex & re_tool_code() {
    static std::regex r(R"(<tool_code>([\s\S]*?)</tool_code>)");
    return r;
}

static const std::regex & re_function_call() {
    static std::regex r(R"(<function_call>([\s\S]*?)</function_call>)");
    return r;
}

static const std::regex & re_bare_function_json() {
    static std::regex r(R"(<function>([\s\S]*?)</function>)");
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

// Preserve syntax-error forwarding only for bodies that still have the
// structure of a JSON object. A pair of braces around prose is not enough to
// identify tool arguments.
static bool looks_like_malformed_json_object(const std::string & text) {
    if (text.size() < 2 || text.front() != '{' || text.back() != '}' ||
        balanced_braces_end(text, 0) != text.size()) {
        return false;
    }

    auto is_object_ws = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };

    size_t i = 1;
    while (i < text.size() && is_object_ws(text[i])) i++;
    if (i >= text.size() - 1) return false;

    const char first = text[i];
    if (first == '"' || first == '\'' || first == '`') {
        const char quote = first;
        bool closed = false;
        for (i++; i < text.size() - 1; i++) {
            if (text[i] == '\\' && i + 1 < text.size() - 1) {
                i++;
                continue;
            }
            if (text[i] == quote) {
                i++;
                closed = true;
                break;
            }
        }
        if (!closed) return false;
    } else {
        auto is_ident_start = [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                   c == '_';
        };
        auto is_ident_continue = [&](char c) {
            return is_ident_start(c) || (c >= '0' && c <= '9') ||
                   c == '.' || c == '-';
        };
        if (!is_ident_start(first)) return false;
        while (i < text.size() - 1 && is_ident_continue(text[i])) i++;
    }

    while (i < text.size() - 1 && is_object_ws(text[i])) i++;
    return i < text.size() - 1 && text[i] == ':';
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
            // Escape inner `"` when we opened the string with a non-`"`
            // quote (single or backtick). Without this, content like
            // `'he said "hi"'` rewrites to `"he said "hi""` which is
            // invalid JSON and silently drops the whole tool call.
            // When in_str == '"', a `"` inside should have arrived via
            // the `\\` escape branch above; a bare `"` here is malformed
            // input we pass through unchanged.
            if (in_str != '"' && c == '"') {
                rewritten += "\\\"";
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

// Parse only a body composed entirely of complete <parameter=KEY>...</parameter>
// elements. Compatibility patterns for malformed function openers use this
// stricter path so partial model output never becomes guessed arguments.
static bool parse_complete_parameter_body(const std::string & body,
                                          const std::string & fn_name,
                                          const json & tools,
                                          json & args) {
    const json props = find_tool_properties(tools, fn_name);
    args = json::object();
    bool found_param = false;
    size_t cursor = 0;

    auto begin = std::sregex_iterator(body.begin(), body.end(),
                                      re_complete_parameter_xml());
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const size_t pos = it->position();
        if (!trim_ws(body.substr(cursor, pos - cursor)).empty()) return false;

        const std::string key = (*it)[1].str();
        if (args.contains(key)) return false;
        args[key] = convert_param_value(trim_ws((*it)[2].str()), key, props);
        found_param = true;
        cursor = pos + it->length();
    }

    return found_param && trim_ws(body.substr(cursor)).empty();
}

// ─── XML tool call parser (<function_call> / <tool_call> / <｜DSML｜tool_calls> with tags) ────

static bool parse_xml_tool_call_body(const std::string & body, const json & tools,
                                     std::string & name, json & args, std::string & raw_args) {
    std::string trimmed = trim_ws(body);
    if (trimmed.empty() || trimmed.front() == '{' || trimmed.find('<') == std::string::npos) {
        return false;
    }

    // Skip legacy Qwen function bodies (<function=...>, <function>, <function >)
    if (trimmed.find("<function=") != std::string::npos ||
        trimmed.find("<function>") != std::string::npos ||
        trimmed.find("<function ") != std::string::npos) {
        return false;
    }

    // 1. Look for function name in top-level XML envelope:
    // a. <invoke name="...">...</invoke> or <invoke tool="...">...</invoke> (with optional ｜DSML｜ prefix)
    std::string param_section;
    static const std::regex re_invoke_envelope(
        R"(^\s*<(?:｜DSML｜)?invoke\s+(?:name|tool)\s*=\s*["']?([A-Za-z_][\w.\-]*)["']?\s*>([\s\S]*?)</(?:｜DSML｜)?invoke>\s*$)");
    std::smatch m_inv;
    if (std::regex_match(trimmed, m_inv, re_invoke_envelope)) {
        name = m_inv[1].str();
        param_section = m_inv[2].str();
    } else {
        // b. <invoke_name>NAME</invoke_name>, <tool_name>NAME</tool_name>, <function_name>NAME</function_name>, or leading <name>NAME</name>
        static const std::regex re_tag_name(
            R"(^\s*<(?:｜DSML｜)?(invoke_name|name|tool_name|function_name)>\s*([A-Za-z_][\w.\-]*)\s*</(?:｜DSML｜)?\1>([\s\S]*)$)");
        std::smatch m_tag;
        if (std::regex_match(trimmed, m_tag, re_tag_name)) {
            name = m_tag[2].str();
            param_section = m_tag[3].str();
        }
    }

    if (name.empty() || !tool_allowed(tools, name)) {
        return false;
    }

    const json props = find_tool_properties(tools, name);
    args = json::object();

    // 2. Look for parameters section in <parameters>, <arguments>, etc.
    static const std::regex re_section(R"(^\s*<(?:｜DSML｜)?(parameters|arguments|tool_calls|function_calls)>([\s\S]*?)</(?:｜DSML｜)?\1>\s*$)");
    std::smatch m_sec;
    std::string trimmed_params_sec = trim_ws(param_section);
    if (std::regex_match(trimmed_params_sec, m_sec, re_section)) {
        param_section = m_sec[2].str();
    }

    std::string trimmed_params = trim_ws(param_section);
    if (trimmed_params.empty()) {
        // Genuinely zero-argument call
        raw_args = "{}";
        return true;
    }

    // Check if param_section is a JSON object
    if (trimmed_params.front() == '{') {
        json j = json::parse(trimmed_params, nullptr, false);
        if (!j.is_discarded() && j.is_object()) {
            for (auto & [k, v] : j.items()) {
                if (v.is_string()) {
                    args[k] = convert_param_value(v.get<std::string>(), k, props);
                } else {
                    args[k] = v;
                }
            }
            raw_args = args.dump();
            return true;
        }
        return false;
    }

    // 3. Extract parameter key-value pairs:
    // a. Attribute style: <(param|parameter) name="key" string="true|false">value</...> or <parameter=key>value</parameter>
    static const std::regex re_attr_param(
        R"(<(?:｜DSML｜)?(?:param|parameter)\s+name\s*=\s*["']?([A-Za-z_][\w.\-]*)["']?(?:\s+string\s*=\s*["']?(true|false)["']?)?\s*>([\s\S]*?)</(?:｜DSML｜)?(?:param|parameter)>|<parameter=([A-Za-z_][\w.\-]*)>([\s\S]*?)</parameter>)");
    auto pbegin = std::sregex_iterator(trimmed_params.begin(), trimmed_params.end(), re_attr_param);
    auto pend = std::sregex_iterator();
    if (pbegin != pend) {
        size_t cursor = 0;
        bool valid = true;
        for (auto it = pbegin; it != pend; ++it) {
            size_t pos = it->position();
            if (!trim_ws(trimmed_params.substr(cursor, pos - cursor)).empty()) {
                valid = false;
                break;
            }
            std::string k = (*it)[1].matched ? (*it)[1].str() : (*it)[4].str();
            if (args.contains(k)) {
                valid = false;
                break;
            }
            if ((*it)[1].matched && (*it)[2].matched) {
                std::string is_str = (*it)[2].str();
                std::string v = (*it)[3].str();
                if (is_str == "true") {
                    args[k] = v;
                } else {
                    json j = json::parse(v, nullptr, false);
                    if (!j.is_discarded()) {
                        args[k] = std::move(j);
                    } else {
                        args[k] = convert_param_value(trim_ws(v), k, props);
                    }
                }
            } else {
                std::string v = trim_ws((*it)[1].matched ? (*it)[3].str() : (*it)[5].str());
                args[k] = convert_param_value(v, k, props);
            }
            cursor = pos + it->length();
        }
        if (valid && trim_ws(trimmed_params.substr(cursor)).empty()) {
            raw_args = args.dump();
            return true;
        }
        return false;
    }

    // b. Element tag style: <key>value</key>
    static const std::regex re_elem_param(R"(<(?:｜DSML｜)?([A-Za-z_][\w.\-]*)>([\s\S]*?)</(?:｜DSML｜)?\1>)");
    auto ebegin = std::sregex_iterator(trimmed_params.begin(), trimmed_params.end(), re_elem_param);
    auto eend = std::sregex_iterator();
    if (ebegin != eend) {
        size_t cursor = 0;
        bool valid = true;
        for (auto it = ebegin; it != eend; ++it) {
            size_t pos = it->position();
            if (!trim_ws(trimmed_params.substr(cursor, pos - cursor)).empty()) {
                valid = false;
                break;
            }
            std::string tag = (*it)[1].str();
            if (tag == "invoke_name" || tag == "name" || tag == "tool_name" ||
                tag == "function_name" || tag == "parameters" || tag == "arguments" ||
                tag == "function_call" || tag == "tool_call" || tag == "tool_calls" ||
                tag == "function_calls" || tag == "invoke") {
                valid = false;
                break;
            }
            if (args.contains(tag)) {
                valid = false;
                break;
            }
            std::string v = trim_ws((*it)[2].str());
            args[tag] = convert_param_value(v, tag, props);
            cursor = pos + it->length();
        }
        if (valid && trim_ws(trimmed_params.substr(cursor)).empty() && !args.empty()) {
            raw_args = args.dump();
            return true;
        }
        return false;
    }

    return false;
}

// ─── JSON tool call parser ──────────────────────────────────────────────

static bool parse_arg_string_or_obj(const json & val, json & out_args,
                                    std::string & out_raw_args) {
    if (val.is_object()) {
        out_args = val;
        out_raw_args = val.dump();
        return true;
    }
    if (val.is_string()) {
        out_raw_args = val.get<std::string>();
        std::string trimmed = trim_ws(out_raw_args);
        json parsed = json::parse(out_raw_args, nullptr, false);
        if (!parsed.is_discarded()) {
            if (parsed.is_object()) {
                out_args = std::move(parsed);
                return true;
            }
            return false;  // reject valid scalar, array, or boolean arguments string
        }
        // Preserve object-shaped syntax errors (for example 5o1), but do not
        // promote an arbitrary scalar string merely because it has braces.
        if (looks_like_malformed_json_object(trimmed)) {
            out_args = json::object();
            return true;
        }
        return false;
    }
    return false;
}

// Parse the named JSON tool-call envelopes emitted by supported chat models.
static bool parse_json_tool_call(const json & obj, std::string & out_name,
                                 json & out_args, std::string & out_raw_args) {
    if (!obj.is_object()) return false;

    if (obj.contains("name") && obj["name"].is_string()) {
        out_name = obj["name"].get<std::string>();
        if (out_name.empty()) return false;
        for (const char * k : {"arguments", "parameters", "args", "params", "input"}) {
            if (obj.contains(k)) return parse_arg_string_or_obj(obj[k], out_args, out_raw_args);
        }
        return false;
    }

    if (obj.contains("function") && obj["function"].is_string()) {
        out_name = obj["function"].get<std::string>();
        if (out_name.empty()) return false;
        for (const char * k : {"parameters", "arguments", "args", "params", "input"}) {
            if (obj.contains(k)) return parse_arg_string_or_obj(obj[k], out_args, out_raw_args);
        }
        return false;
    }

    for (const char * sub : {"function", "function_call", "tool_call"}) {
        if (obj.contains(sub) && obj[sub].is_object()) {
            if (parse_json_tool_call(
                    obj[sub], out_name, out_args, out_raw_args)) {
                return true;
            }
        }
    }

    return false;
}

static bool parse_json_tool_call(const json & obj, std::string & out_name, json & out_args) {
    std::string raw;
    return parse_json_tool_call(obj, out_name, out_args, raw);
}

static bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static bool span_covers_non_ws(const std::string & text, size_t start, size_t end) {
    for (size_t i = 0; i < start; i++) {
        if (!is_ws(text[i])) return false;
    }
    for (size_t i = end; i < text.size(); i++) {
        if (!is_ws(text[i])) return false;
    }
    return true;
}

static const json * single_tool_function(const json & tools) {
    if (!tools.is_array() || tools.size() != 1) return nullptr;
    const auto & t = tools[0];
    if (!t.is_object()) return nullptr;
    if (t.contains("function") && t["function"].is_object()) return &t["function"];
    return &t;
}

static const json * tool_input_schema(const json & fn) {
    if (fn.contains("parameters") && fn["parameters"].is_object()) {
        return &fn["parameters"];
    }
    if (fn.contains("input_schema") && fn["input_schema"].is_object()) {
        return &fn["input_schema"];
    }
    return nullptr;
}

static bool value_matches_type(const json & value, const std::string & type) {
    if (type == "string" || type == "str") return value.is_string();
    if (type == "integer" || type == "int") return value.is_number_integer();
    if (type == "number" || type == "float") return value.is_number();
    if (type == "boolean" || type == "bool") return value.is_boolean();
    if (type == "object") return value.is_object();
    if (type == "array") return value.is_array();
    if (type == "null") return value.is_null();
    return true;
}

static bool value_matches_type_spec(const json & value, const json & spec) {
    if (spec.is_string()) return value_matches_type(value, spec.get<std::string>());
    if (spec.is_array()) {
        for (const auto & t : spec) {
            if (t.is_string() && value_matches_type(value, t.get<std::string>())) {
                return true;
            }
        }
        return false;
    }
    return true;
}

static bool object_matches_tool_schema(const json & obj, const json * schema) {
    if (!obj.is_object()) return false;
    if (!schema) return !obj.empty();
    if (schema->contains("type") &&
        !value_matches_type_spec(obj, (*schema)["type"])) {
        return false;
    }

    const json * props = nullptr;
    if (schema->contains("properties") && (*schema)["properties"].is_object()) {
        props = &(*schema)["properties"];
    }

    bool has_required = false;
    if (schema->contains("required") && (*schema)["required"].is_array()) {
        for (const auto & key_json : (*schema)["required"]) {
            if (!key_json.is_string()) continue;
            has_required = true;
            std::string key = key_json.get<std::string>();
            if (!obj.contains(key)) return false;
        }
    }

    const bool additional_forbidden =
        schema->contains("additionalProperties") &&
        (*schema)["additionalProperties"].is_boolean() &&
        !(*schema)["additionalProperties"].get<bool>();

    bool saw_declared_key = false;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const bool declared = props && props->contains(it.key());
        if (!declared) {
            if (additional_forbidden) return false;
            continue;
        }
        saw_declared_key = true;
        const auto & prop_schema = (*props)[it.key()];
        if (prop_schema.is_object() && prop_schema.contains("type") &&
            !value_matches_type_spec(it.value(), prop_schema["type"])) {
            return false;
        }
    }

    return has_required || saw_declared_key || obj.empty() ||
           props == nullptr || props->empty();
}

static bool parse_single_tool_arg_object(const json & obj, const json & tools,
                                         std::string & out_name, json & out_args) {
    const json * fn = single_tool_function(tools);
    if (!fn || !fn->contains("name") || !(*fn)["name"].is_string()) return false;
    const json * schema = tool_input_schema(*fn);
    if (!object_matches_tool_schema(obj, schema)) return false;
    out_name = (*fn)["name"].get<std::string>();
    out_args = obj;
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

static bool extract_raw_json_tool_fallback(const std::string & text,
                                           std::string & out_name,
                                           std::string & out_raw_args) {
    static const std::regex re_args_open(
        R"re("(?:arguments|parameters|args|params|input)"\s*:\s*\{)re");
    static const char marker_key[] = "__lucebox_raw_args_marker__";
    static const std::string marker_object =
        std::string("{\"") + marker_key + "\":true}";

    // The compatibility case is malformed JSON *inside* an otherwise valid
    // arguments object (for example 5o1 instead of 501). Replace each
    // candidate with a marker object and let the normal structural parser
    // select the envelope and tool name. Requiring that same marker in the
    // selected call prevents pairing arguments from one object with the name
    // from another, while preserving the exact malformed arguments for the
    // client to report back to the model.
    auto begin = std::sregex_iterator(text.begin(), text.end(), re_args_open);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const size_t brace_open = it->position() + it->length() - 1;
        const size_t brace_close = balanced_braces_end(text, brace_open);
        if (brace_close == std::string::npos) continue;

        std::string repaired = text;
        repaired.replace(
            brace_open, brace_close - brace_open, marker_object);
        json obj = json::parse(repaired, nullptr, false);
        if (obj.is_discarded()) continue;

        std::string name;
        json args;
        if (!parse_json_tool_call(obj, name, args)) continue;
        if (!args.is_object() || args.size() != 1 ||
            !args.value(marker_key, false)) {
            continue;
        }

        out_name = std::move(name);
        out_raw_args = text.substr(
            brace_open, brace_close - brace_open);
        return true;
    }
    return false;
}

// ─── Main parser ────────────────────────────────────────────────────────

ToolParseResult parse_tool_calls(const std::string & text, const json & tools) {
    ToolParseResult result;
    std::vector<Span> removals;
    std::vector<std::pair<size_t, ToolCall>> positioned_calls;

    // JSON lines may be siblings of invoke envelopes, but JSON nested inside
    // an invoke always belongs to that envelope. Track all invoke spans,
    // including malformed or disallowed ones, so no later JSON sweep can
    // reinterpret their bodies as independent calls.
    static const std::regex re_invoke_span(
        R"(<invoke(?:\s[^>]*)?>[\s\S]*?</invoke\s*>)");
    std::vector<Span> invoke_spans;
    auto invoke_begin = std::sregex_iterator(
        text.begin(), text.end(), re_invoke_span);
    auto invoke_end = std::sregex_iterator();
    for (auto it = invoke_begin; it != invoke_end; ++it) {
        const size_t start = it->position();
        invoke_spans.push_back({start, start + it->length()});
    }

    auto add_call = [&](const std::string & fn_name, const json & args,
                        size_t start, size_t end,
                        const std::string & raw_args = "") {
        if (!tool_allowed(tools, fn_name)) return;
        ToolCall tc;
        tc.id = generate_call_id();
        tc.name = fn_name;
        tc.arguments = raw_args.empty() ? args.dump() : raw_args;
        positioned_calls.emplace_back(start, std::move(tc));
        removals.push_back({start, end});
    };

    // Pattern 8 (Laguna): <tool_call>NAME\n<arg_key>K</arg_key>\n
    // <arg_value>V</arg_value>...\n</tool_call>. Values are raw strings or
    // JSON (the template emits non-strings via tojson); coerce via the
    // declared tool schema like the other patterns. Checked before pattern 1
    // so the shared <tool_call> wrapper is not half-consumed by the Qwen
    // regexes.
    {
        size_t pos = 0;
        while ((pos = text.find("<tool_call>", pos)) != std::string::npos) {
            const size_t body_start = pos + 11;
            const size_t close = text.find("</tool_call>", body_start);
            if (close == std::string::npos) break;
            const std::string body = text.substr(body_start, close - body_start);
            const bool is_legacy_qwen =
                body.find("<function=") != std::string::npos ||
                body.find("<function>") != std::string::npos ||
                body.find("<function ") != std::string::npos;
            if (is_legacy_qwen) {
                pos = close + 12;
                continue;
            }
            std::string xml_name;
            json xml_args;
            std::string xml_raw;
            if (parse_xml_tool_call_body(body, tools, xml_name, xml_args, xml_raw)) {
                add_call(xml_name, xml_args, pos, close + 12, xml_raw);
                pos = close + 12;
                continue;
            }

            const size_t first_key = body.find("<arg_key>");
            // Only claim bodies in the Laguna shape: bare name then arg tags
            // (or a bare name alone for zero-arg calls); leave <function=...>
            // bodies to the Qwen patterns below.
            // Laguna bodies are `NAME<arg_key>...` (values may contain JSON —
            // the template serializes non-string args via tojson). Only leave
            // <function=...> and pure-JSON bodies to the Qwen patterns.
            if (!is_legacy_qwen &&
                (first_key != std::string::npos ||
                 body.find('{') == std::string::npos)) {
                std::string name = trim_ws(
                    first_key == std::string::npos ? body : body.substr(0, first_key));
                if (!name.empty() && name.find('<') == std::string::npos) {
                    const json props = find_tool_properties(tools, name);
                    json args = json::object();
                    size_t kpos = first_key;
                    while (kpos != std::string::npos) {
                        const size_t kend = body.find("</arg_key>", kpos);
                        if (kend == std::string::npos) break;
                        const std::string key =
                            trim_ws(body.substr(kpos + 9, kend - (kpos + 9)));
                        const size_t vpos = body.find("<arg_value>", kend);
                        if (vpos == std::string::npos) break;
                        const size_t vend = body.find("</arg_value>", vpos);
                        if (vend == std::string::npos) break;
                        const std::string val =
                            trim_ws(body.substr(vpos + 11, vend - (vpos + 11)));
                        if (!key.empty()) {
                            args[key] = convert_param_value(val, key, props);
                        }
                        kpos = body.find("<arg_key>", vend);
                    }
                    add_call(name, args, pos, close + 12);
                }
            }
            pos = close + 12;
        }

        // Stripped-wrapper variant: <tool_call>/</tool_call> are SPECIAL
        // tokens in the laguna vocab and detokenization removes them, so the
        // visible text is `NAME<arg_key>K</arg_key><arg_value>V</arg_value>…`.
        // Anchor on <arg_key> and walk back over identifier chars for the
        // name. No other family emits bare <arg_key>, so this cannot
        // misfire cross-family.
        size_t apos = 0;
        while ((apos = text.find("<arg_key>", apos)) != std::string::npos) {
            if (overlaps(removals, apos)) { apos += 9; continue; }
            size_t name_end = apos;
            size_t name_start = name_end;
            auto is_ident = [](char c) {
                // OpenAI-shape function names: [A-Za-z0-9_-] only. '.' must
                // stay out or prose immediately before the name gets eaten
                // ("...the weather tool.get_weather<arg_key>").
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '-';
            };
            while (name_start > 0 && is_ident(text[name_start - 1])) name_start--;
            const std::string name = text.substr(name_start, name_end - name_start);
            if (name.empty()) { apos += 9; continue; }
            const json props = find_tool_properties(tools, name);
            json args = json::object();
            size_t kpos = apos;
            size_t span_end = apos;
            while (kpos != std::string::npos && kpos == span_end) {
                const size_t kend = text.find("</arg_key>", kpos);
                if (kend == std::string::npos) break;
                const size_t vpos = text.find("<arg_value>", kend);
                if (vpos == std::string::npos) break;
                const size_t vend = text.find("</arg_value>", vpos);
                if (vend == std::string::npos) break;
                const std::string key = trim_ws(text.substr(kpos + 9, kend - (kpos + 9)));
                const std::string val = trim_ws(text.substr(vpos + 11, vend - (vpos + 11)));
                if (!key.empty()) args[key] = convert_param_value(val, key, props);
                span_end = vend + 12;
                // consume whitespace between pairs, then check for the next key
                size_t nxt = span_end;
                while (nxt < text.size() && (text[nxt] == '\n' || text[nxt] == ' ' ||
                                             text[nxt] == '\t' || text[nxt] == '\r')) nxt++;
                kpos = (text.compare(nxt, 9, "<arg_key>") == 0) ? nxt : std::string::npos;
                if (kpos != std::string::npos) span_end = nxt;
            }
            if (!args.empty()) {
                add_call(name, args, name_start, span_end);
            }
            apos = span_end > apos ? span_end : apos + 9;
        }
    }

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

    // Pattern 3b: <TOOL_NAME>...params...</function>. Some agents/models
    // emit the selected tool name as the XML tag itself, then close with the
    // Qwen </function> tag. Only accept requested tools and real parameter
    // tags so arbitrary XML-ish prose remains visible text.
    if (tools.is_array() && !tools.empty()) {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_bare_tool_name_xml());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            size_t pos = it->position();
            if (overlaps(removals, pos)) continue;
            std::string fn_name = (*it)[1].str();
            std::string params = (*it)[2].str();
            if (!tool_allowed(tools, fn_name)) continue;
            if (params.find("<parameter=") == std::string::npos) continue;
            add_call(fn_name, parse_xml_params(params, fn_name, tools), pos,
                     include_following_tool_call_close(text,
                                                       pos + it->length(),
                                                       fn_name));
        }
    }

    // Pattern 3c: <parameter name="TOOL"><parameter name="KEY">VALUE...
    // Some Qwen outputs confuse the tool wrapper with an attribute-style
    // parameter tag. Accept this only when the outer name is a requested
    // tool and the entire body consists of fully closed parameter elements.
    // This strict body check prevents partial or guessed arguments.
    if (tools.is_array() && !tools.empty()) {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_attribute_tool_xml());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            const size_t pos = it->position();
            if (overlaps(removals, pos)) continue;

            const std::string fn_name = (*it)[1].str();
            if (!tool_allowed(tools, fn_name)) continue;

            const std::string body = (*it)[2].str();
            const json props = find_tool_properties(tools, fn_name);
            json args = json::object();
            bool valid = true;
            bool found_param = false;
            size_t cursor = 0;

            auto pbegin =
                std::sregex_iterator(body.begin(), body.end(), re_attribute_parameter_xml());
            auto pend = std::sregex_iterator();
            for (auto pit = pbegin; pit != pend; ++pit) {
                const size_t ppos = pit->position();
                if (!trim_ws(body.substr(cursor, ppos - cursor)).empty()) {
                    valid = false;
                    break;
                }

                const std::string key = (*pit)[1].str();
                if (args.contains(key)) {
                    valid = false;
                    break;
                }
                args[key] =
                    convert_param_value(trim_ws((*pit)[2].str()), key, props);
                found_param = true;
                cursor = ppos + pit->length();
            }

            if (!trim_ws(body.substr(cursor)).empty()) valid = false;
            if (!valid || !found_param) continue;

            add_call(fn_name, args, include_preceding_tool_call_open(text, pos),
                     include_following_tool_call_close(text,
                                                       pos + it->length()));
        }
    }

    // Pattern 3d: <funcname>TOOL<parameter=KEY>VALUE...</function>.
    // Qwen occasionally substitutes the literal `funcname` tag for
    // `<function=TOOL>`. Accept it only for a requested tool and only when
    // the remaining body is composed entirely of complete parameter blocks.
    if (tools.is_array() && !tools.empty()) {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_funcname_tool_xml());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            const size_t pos = it->position();
            if (overlaps(removals, pos)) continue;

            const std::string fn_name = (*it)[1].str();
            if (!tool_allowed(tools, fn_name)) continue;

            json args;
            if (!parse_complete_parameter_body((*it)[2].str(), fn_name,
                                               tools, args)) {
                continue;
            }

            add_call(fn_name, args, include_preceding_tool_call_open(text, pos),
                     include_following_tool_call_close(text,
                                                       pos + it->length()));
        }
    }

    // Pattern 3e: <function TOOL><parameter=KEY>VALUE...</function>.
    // This is a malformed Qwen variant of <function=TOOL>. Accept it only
    // for a requested tool and only when every argument is fully delimited.
    if (tools.is_array() && !tools.empty()) {
        auto begin = std::sregex_iterator(text.begin(), text.end(),
                                          re_space_function_xml());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            const size_t pos = it->position();
            if (overlaps(removals, pos)) continue;

            const std::string fn_name = (*it)[1].str();
            if (!tool_allowed(tools, fn_name)) continue;

            json args;
            if (!parse_complete_parameter_body((*it)[2].str(), fn_name,
                                               tools, args)) {
                continue;
            }
            add_call(fn_name, args, include_preceding_tool_call_open(text, pos),
                     pos + it->length());
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

    // Pattern 4b: <function_call>{JSON}</function_call>
    {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_function_call());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            size_t pos = it->position();
            if (overlaps(removals, pos)) continue;
            std::string inner = (*it)[1].str();
            size_t s = inner.find_first_not_of(" \t\n\r");
            if (s != std::string::npos) inner = inner.substr(s);
            size_t e = inner.find_last_not_of(" \t\n\r");
            if (e != std::string::npos) inner = inner.substr(0, e + 1);
            json obj = json::parse(inner, nullptr, false);
            std::string name;
            json args;
            std::string raw_args;
            if (!obj.is_discarded() && parse_json_tool_call(obj, name, args, raw_args)) {
                add_call(name, args, pos, pos + it->length(), raw_args);
            } else if (parse_xml_tool_call_body(inner, tools, name, args, raw_args)) {
                add_call(name, args, pos, pos + it->length(), raw_args);
            } else if (extract_raw_json_tool_fallback(inner, name, raw_args)) {
                add_call(name, json::object(), pos, pos + it->length(), raw_args);
            }
        }
    }

    // Pattern 4c: <function>{JSON}</function>
    {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re_bare_function_json());
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            size_t pos = it->position();
            if (overlaps(removals, pos)) continue;
            std::string inner = (*it)[1].str();
            size_t s = inner.find_first_not_of(" \t\n\r");
            if (s != std::string::npos) inner = inner.substr(s);
            size_t e = inner.find_last_not_of(" \t\n\r");
            if (e != std::string::npos) inner = inner.substr(0, e + 1);
            json obj = json::parse(inner, nullptr, false);
            std::string name;
            json args;
            std::string raw_args;
            if (!obj.is_discarded() && parse_json_tool_call(obj, name, args, raw_args)) {
                add_call(name, args, pos, pos + it->length(), raw_args);
            } else if (extract_raw_json_tool_fallback(inner, name, raw_args)) {
                add_call(name, json::object(), pos, pos + it->length(), raw_args);
            }
        }
    }

    // Pattern 4d: <function_calls> or <tool_calls> containing <invoke> blocks or JSON lines
    {
        static const std::regex re_block(R"(<(?:｜DSML｜)?(?:function_calls|tool_calls)>([\s\S]*?)</(?:｜DSML｜)?(?:function_calls|tool_calls)>)");
        static const std::regex re_invoke(R"(<(?:｜DSML｜)?invoke\s+(?:name|tool)\s*=\s*["']?([A-Za-z_][\w.\-]*)["']?\s*>([\s\S]*?)</(?:｜DSML｜)?invoke>)");
        static const std::regex re_param(R"(<(?:｜DSML｜)?(param|parameter)\s+name\s*=\s*["']?([A-Za-z_][\w.\-]*)["']?(?:\s+string\s*=\s*["']?(true|false)["']?)?\s*>([\s\S]*?)</(?:｜DSML｜)?\1>)");

        auto fbegin = std::sregex_iterator(text.begin(), text.end(), re_block);
        auto fend = std::sregex_iterator();
        for (auto fit = fbegin; fit != fend; ++fit) {
            size_t bstart = fit->position();
            size_t bend = bstart + fit->length();
            if (overlaps(removals, bstart)) continue;

            std::string block_content = (*fit)[1].str();
            size_t inner_start = fit->position(1);
            struct CallMatch { std::string name; json args; std::string raw_args; size_t start, end; };
            std::vector<CallMatch> block_calls;

            // 1. Try <invoke> tags
            auto begin = std::sregex_iterator(block_content.begin(), block_content.end(), re_invoke);
            auto end = std::sregex_iterator();
            for (auto it = begin; it != end; ++it) {
                std::string fn_name = (*it)[1].str();
                if (!tool_allowed(tools, fn_name)) continue;
                std::string body = trim_ws((*it)[2].str());
                json args = json::object();
                std::string raw_args;
                if (!body.empty() && body.front() == '{') {
                    json raw_json = json::parse(body, nullptr, false);
                    if (!raw_json.is_discarded() && raw_json.is_object()) {
                        json props = find_tool_properties(tools, fn_name);
                        for (auto & [k, v] : raw_json.items()) {
                            if (v.is_string()) {
                                args[k] = convert_param_value(v.get<std::string>(), k, props);
                            } else {
                                args[k] = v;
                            }
                        }
                        raw_args = args.dump();
                    } else if (looks_like_malformed_json_object(body)) {
                        raw_args = body;
                    } else {
                        continue;
                    }
                } else {
                    size_t cursor = 0;
                    bool valid_body = true;
                    bool found_param = false;
                    auto pbegin = std::sregex_iterator(body.begin(), body.end(), re_param);
                    auto pend = std::sregex_iterator();
                    for (auto pit = pbegin; pit != pend; ++pit) {
                        size_t ppos = pit->position();
                        if (!trim_ws(body.substr(cursor, ppos - cursor)).empty()) { valid_body = false; break; }
                        std::string k = (*pit)[2].str();
                        if (args.contains(k)) { valid_body = false; break; }
                        if ((*pit)[3].matched) {
                            std::string is_str = (*pit)[3].str();
                            std::string v = (*pit)[4].str();
                            if (is_str == "true") {
                                args[k] = v;
                            } else {
                                json j = json::parse(v, nullptr, false);
                                if (!j.is_discarded()) {
                                    args[k] = std::move(j);
                                } else {
                                    args[k] = convert_param_value(trim_ws(v), k, find_tool_properties(tools, fn_name));
                                }
                            }
                        } else {
                            std::string v = trim_ws((*pit)[4].str());
                            args[k] = convert_param_value(v, k, find_tool_properties(tools, fn_name));
                        }
                        found_param = true;
                        cursor = ppos + pit->length();
                    }
                    if (!valid_body) continue;
                    if (!found_param && !trim_ws(body).empty()) continue;
                    if (!trim_ws(body.substr(cursor)).empty()) continue;
                    if (raw_args.empty()) raw_args = args.dump();
                }
                size_t istart = inner_start + it->position();
                size_t iend = istart + it->length();
                block_calls.push_back({fn_name, std::move(args), raw_args, istart, iend});
            }

            // 2. Also scan for JSON objects in spans not covered by <invoke> calls
            size_t cursor = 0;
            while (cursor < block_content.size()) {
                size_t s = block_content.find('{', cursor);
                if (s == std::string::npos) break;
                const size_t abs_s = inner_start + s;
                bool inside_invoke = false;
                for (const auto & span : invoke_spans) {
                    if (abs_s >= span.start && abs_s < span.end) {
                        inside_invoke = true;
                        cursor = span.end - inner_start;
                        break;
                    }
                }
                if (inside_invoke) continue;

                size_t e = balanced_braces_end(block_content, s);
                if (e == std::string::npos) { cursor = s + 1; continue; }
                std::string jstr = block_content.substr(s, e - s);
                json obj = json::parse(jstr, nullptr, false);
                std::string name;
                json args;
                std::string raw;
                if (!obj.is_discarded() && parse_json_tool_call(obj, name, args, raw) && tool_allowed(tools, name)) {
                    block_calls.push_back({name, std::move(args), raw, inner_start + s, inner_start + e});
                } else if (extract_raw_json_tool_fallback(jstr, name, raw) && tool_allowed(tools, name)) {
                    block_calls.push_back({name, json::object(), raw, inner_start + s, inner_start + e});
                }
                cursor = e;
            }

            if (!block_calls.empty()) {
                std::sort(block_calls.begin(), block_calls.end(),
                          [](const CallMatch & a, const CallMatch & b) { return a.start < b.start; });

                // If the block contains only whitespace around the calls, remove the whole block
                bool clean_block = true;
                size_t last_end = 0;
                for (const auto & bc : block_calls) {
                    size_t rel_start = bc.start - inner_start;
                    if (!trim_ws(block_content.substr(last_end, rel_start - last_end)).empty()) {
                        clean_block = false;
                        break;
                    }
                    last_end = bc.end - inner_start;
                }
                if (!trim_ws(block_content.substr(last_end)).empty()) clean_block = false;

                if (clean_block) {
                    for (auto & bc : block_calls) {
                        add_call(bc.name, bc.args, bstart, bend, bc.raw_args);
                    }
                } else {
                    for (auto & bc : block_calls) {
                        add_call(bc.name, bc.args, bc.start, bc.end, bc.raw_args);
                    }
                }
            }
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
            if (overlaps(removals, start) || overlaps(invoke_spans, start)) {
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
                std::string name;
                std::string raw_args;
                if (extract_raw_json_tool_fallback(json_str, name, raw_args) && tool_allowed(tools, name)) {
                    add_call(name, json::object(), start, end_pos, raw_args);
                    cursor = end_pos;
                    continue;
                }
                cursor = start + 1;
                continue;
            }

            std::string name;
            json args;
            std::string raw_args;
            if (parse_json_tool_call(obj2, name, args, raw_args)) {
                add_call(name, args, start, end_pos, raw_args);
            } else if (span_covers_non_ws(text, start, end_pos) &&
                       parse_single_tool_arg_object(obj2, tools, name, args)) {
                add_call(name, args, start, end_pos);
            }
            cursor = end_pos;
        }
    }

    // Detection runs by syntax family, so restore the calls' source order
    // before exposing them to clients. Dependent calls must execute in the
    // same order in which the model emitted them.
    std::stable_sort(positioned_calls.begin(), positioned_calls.end(),
                     [](const auto & a, const auto & b) {
                         return a.first < b.first;
                     });
    result.tool_calls.reserve(positioned_calls.size());
    for (auto & entry : positioned_calls) {
        result.tool_calls.push_back(std::move(entry.second));
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
