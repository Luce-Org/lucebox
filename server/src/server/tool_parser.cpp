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
            const size_t first_key = body.find("<arg_key>");
            // Only claim bodies in the Laguna shape: bare name then arg tags
            // (or a bare name alone for zero-arg calls); leave <function=...>
            // bodies to the Qwen patterns below.
            // Laguna bodies are `NAME<arg_key>...` (values may contain JSON —
            // the template serializes non-string args via tojson). Only leave
            // <function=...> and pure-JSON bodies to the Qwen patterns.
            if (body.find("<function") == std::string::npos &&
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
            } else if (span_covers_non_ws(text, start, end_pos) &&
                       parse_single_tool_arg_object(obj2, tools, name, args)) {
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
