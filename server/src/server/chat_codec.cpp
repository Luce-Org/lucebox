#include "chat_codec.h"

#include <map>

#include "tokenizer.h"
#include "common/sha256.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstring>
#include <string_view>

namespace luce::common {

std::string RenderedPrompt::text() const {
    std::string out;
    for (const auto & piece : pieces) out += piece.text;
    return out;
}

std::vector<int32_t> RenderedPrompt::tokens(const Tokenizer & tokenizer) const {
    std::vector<int32_t> ids;
    std::vector<int32_t> literal_ids;
    for (const auto & token : literal) {
        const int32_t id = tokenizer.token_to_id(token);
        if (id >= 0) literal_ids.push_back(id);
    }
    std::string run;
    const auto flush = [&]() {
        if (run.empty()) return;
        const auto run_ids = tokenizer.encode(run, /*match_control=*/false, literal_ids);
        ids.insert(ids.end(), run_ids.begin(), run_ids.end());
        run.clear();
    };
    for (const auto & piece : pieces) {
        if (!piece.control && piece.ids.empty()) {
            run += piece.text;
            continue;
        }
        flush();
        const auto piece_ids = piece.ids.empty() ? tokenizer.encode(piece.text) : piece.ids;
        ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
    }
    flush();
    return ids;
}

// The code points Python's str.isspace() accepts, as UTF-8.
static size_t python_space_at(const std::string & text, size_t pos) {
    const unsigned char c = (unsigned char) text[pos];
    if ((c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20)) return 1;
    static const char * const kSpaces[] = {
        "\xc2\x85", "\xc2\xa0", "\xe1\x9a\x80",
        "\xe2\x80\x80", "\xe2\x80\x81", "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85",
        "\xe2\x80\x86", "\xe2\x80\x87", "\xe2\x80\x88", "\xe2\x80\x89", "\xe2\x80\x8a",
        "\xe2\x80\xa8", "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80",
    };
    for (const char * space : kSpaces) {
        const size_t n = std::strlen(space);
        if (text.compare(pos, n, space) == 0) return n;
    }
    return 0;
}

std::string python_strip(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t n = python_space_at(text, begin);
        if (n == 0) break;
        begin += n;
    }
    size_t end = text.size();
    while (end > begin) {
        // Step back to the start of the last code point.
        size_t start = end - 1;
        while (start > begin && ((unsigned char) text[start] & 0xc0) == 0x80) --start;
        if (python_space_at(text, start) != end - start) break;
        end = start;
    }
    return text.substr(begin, end - begin);
}

std::string TurnMemory::key(const ChatMessage & turn, const std::string & context) {
    // What the template shows of the turn besides its reasoning; each field
    // length-prefixed, so no content can spell another turn's calls.
    std::string key;
    const auto field = [&key](const std::string & value) {
        key += std::to_string(value.size());
        key += ':';
        key += value;
    };
    field(context);
    field(python_strip(turn.content));
    for (const auto & call : turn.tool_calls) {
        field(call.name);
        field(call.arguments.dump());
    }
    return key;
}

void TurnMemory::remember(const ChatMessage & turn, Turn generated) {
    if (disabled()) return;
    generated.reasoning = python_strip(generated.reasoning);
    std::string k = key(turn, generated.context);
    std::lock_guard<std::mutex> lock(mu_);
    if (const auto it = entries_.find(k); it != entries_.end()) {
        bytes_ -= it->first.size() + it->second.turn.text.size() + it->second.turn.reasoning.size() +
                  it->second.turn.tokens.size() * sizeof(int32_t);
        lru_.erase(it->second.lru);
        entries_.erase(it);
    }
    bytes_ += k.size() + generated.text.size() + generated.reasoning.size() +
              generated.tokens.size() * sizeof(int32_t);
    lru_.push_back(k);
    entries_.emplace(std::move(k), Entry{std::move(generated), std::prev(lru_.end())});
    prune();
}

bool TurnMemory::lookup(const ChatMessage & turn, const std::string & context, Turn & out) {
    if (disabled()) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(key(turn, context));
    if (it == entries_.end()) return false;
    if (turn.reasoning_sent && python_strip(turn.reasoning_content) != it->second.turn.reasoning) return false;
    lru_.splice(lru_.end(), lru_, it->second.lru);
    out = it->second.turn;
    return true;
}

void TurnMemory::prune() {
    while (bytes_ > max_bytes_ && !lru_.empty()) {
        const auto it = entries_.find(lru_.front());
        bytes_ -= it->first.size() + it->second.turn.text.size() + it->second.turn.reasoning.size() +
                  it->second.turn.tokens.size() * sizeof(int32_t);
        entries_.erase(it);
        lru_.pop_front();
    }
}

namespace {

class PromptWriter {
public:
    explicit PromptWriter(RenderedPrompt & out) : out_(out) {
        for (const auto & piece : out_.pieces) {
            at_ += piece.text.size();
            hash(piece.text, piece.control);
        }
    }

    // The length of the text written so far: where the next piece begins.
    size_t at() const { return at_; }
    // The identity of the prompt written so far (TurnMemory::Turn::context).
    std::string context() const { return hash_.hex(); }

    void control(const char * token) {
        out_.pieces.push_back({token, true, {}});
        at_ += std::strlen(token);
        hash(token, true);
    }
    void text(const std::string & text) {
        if (text.empty()) return;
        at_ += text.size();
        hash(text, false);
        auto * last = out_.pieces.empty() ? nullptr : &out_.pieces.back();
        if (last && !last->control && last->ids.empty()) last->text += text;
        else out_.pieces.push_back({text, false, {}});
    }
    void tokens(const std::string & text, const std::vector<int32_t> & ids) {
        out_.pieces.push_back({text, false, ids});
        at_ += text.size();
        hash(text, false);
    }
    // Message text: each image marker the server put there becomes control tokens.
    void content(const std::string & text, const std::string & image_marker) {
        if (image_marker.empty()) return this->text(text);
        size_t pos = 0;
        for (size_t found; (found = text.find(image_marker, pos)) != std::string::npos;
             pos = found + image_marker.size()) {
            this->text(text.substr(pos, found - pos));
            out_.pieces.push_back({image_marker, true, {}});
            at_ += image_marker.size();
            hash(image_marker, true);
        }
        this->text(text.substr(pos));
    }
    // Writes message content and records where it landed.
    void content(const std::string & text, const std::string & image_marker, PromptSpan & span) {
        span.begin = at_;
        content(text, image_marker);
        span.end = at_;
    }
    // A remembered turn in place of the message's render. Its content span is
    // the client's content inside it, after the role line (`skip`) and the
    // reasoning, if the turn holds one, and empty where the model's text does
    // not hold it as sent.
    void generated(const TurnMemory::Turn & turn, const std::string & content, size_t skip, bool reasoning,
                   PromptSpan & span) {
        const size_t at = at_;
        tokens(turn.text, turn.tokens);
        const size_t think = reasoning ? turn.text.find("</think>", skip) : std::string::npos;
        const size_t from = (std::min)(think == std::string::npos ? skip : think + std::strlen("</think>"),
                                       turn.text.size());
        const std::string shown = python_strip(content);
        const size_t found = shown.empty() ? std::string::npos : turn.text.find(shown, from);
        span = found == std::string::npos ? PromptSpan{at + from, at + from}
                                          : PromptSpan{at + found, at + found + shown.size()};
    }

private:
    // Text and control tokens hash apart, so a message that spells a control
    // token is not the prompt with that token: "\0c" and "\0t" switch
    // between them, and a NUL in either is "\0\0".
    void hash(std::string_view bytes, bool control) {
        if (control != control_) {
            hash_.update(std::string_view(control ? "\0c" : "\0t", 2));
            control_ = control;
        }
        for (size_t pos = 0; pos < bytes.size();) {
            const size_t nul = bytes.find('\0', pos);
            if (nul == std::string_view::npos) {
                hash_.update(bytes.substr(pos));
                break;
            }
            hash_.update(bytes.substr(pos, nul + 1 - pos));
            hash_.update(std::string_view("\0", 1));
            pos = nul + 1;
        }
    }

    RenderedPrompt & out_;
    size_t at_ = 0;
    Sha256 hash_;
    bool control_ = false;
};

bool starts_with(const std::string & text, const char * prefix) {
    return text.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string & text, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return text.size() >= n && text.compare(text.size() - n, n, suffix) == 0;
}

// Qwen3.8 (27B and Flash-Next ship the same template): Qwen/Qwen3.8-27B
// tokenizer_config.json at 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. Qwen3.6
// (27B and 35B-A3B ship the same template): Qwen/Qwen3.6-27B at
// 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9. Qwen3.6 has no reasoning-effort
// line, shows earlier thinking only when preserve_thinking is true, and reads
// thinking out of content ("...</think>") when a turn sends no reasoning.
// Qwen3.5 (27B and 35B-A3B ship the same template): Qwen/Qwen3.5-27B at
// fc05daec18b0a78c049392ed2e771dde82bdf654. As Qwen3.6, except that it writes
// a scalar argument with Python's str() (True, False, None), and that its
// template never shows earlier thinking. The codec keeps it unless the request
// sends preserve_thinking false (the official prompt): dropping it rewrites the
// turns after the previous question at each new one, so the prefix cache and
// PFlash views could not continue (vLLM, SGLang, llama.cpp and Ollama drop it).
enum class QwenVersion { k35, k36, k38 };

class QwenCodec final : public ChatCodec {
public:
    explicit QwenCodec(QwenVersion version)
        : version_(version),
          id_(version == QwenVersion::k38 ? "qwen3.8/1" : version == QwenVersion::k36 ? "qwen3.6/1" : "qwen3.5/1") {}
    const std::string & id() const override { return id_; }
    // The template trims content and reasoning.
    std::vector<std::string> tool_call_openers() const override { return {"<tool_call>"}; }

    bool render(const ChatCodecInput & input, RenderedPrompt & out,
                std::string & error) const override;
    bool parse_tool_calls(const std::string & block, const nlohmann::json & tools,
                          ToolParseResult & out) const override;
    bool parse_turn(const std::string & turn, const nlohmann::json & tools,
                    ChatMessage & out) const override;
    bool generated_turn(const std::string & rendered, const std::vector<int32_t> & prompt_tokens,
                        const std::vector<int32_t> & generated, const Tokenizer & tokenizer,
                        const nlohmann::json & tools, ChatMessage & message,
                        TurnMemory::Turn & turn) const override;

private:
    QwenVersion version_;
    std::string id_;
};

const char kQwenToolsHeader[] = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const char kQwenToolsFooter[] =
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function "
    "call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>";
const char kQwenEffortXhigh[] =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const char kQwenEffortLow[] =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.";

bool QwenCodec::render(const ChatCodecInput & input, RenderedPrompt & out,
                         std::string & error) const {
    out = RenderedPrompt();
    if (!input.messages || input.messages->empty()) {
        error = "No messages provided.";
        return false;
    }
    const auto & messages = *input.messages;
    out.messages.assign(messages.size(), MessageLayout());
    PromptWriter w(out);
    const auto open = [&](const std::string & role) {
        w.control("<|im_start|>");
        w.text(role);
    };
    const auto close = [&]() {
        w.control("<|im_end|>");
        w.text("\n");
    };

    std::string instructions;
    if (version_ == QwenVersion::k38 && input.enable_thinking) {
        const std::string effort = input.reasoning_effort.empty()
            ? std::string("xhigh") : qwen4exp_template_effort(input.reasoning_effort);
        if (effort == "xhigh") instructions = kQwenEffortXhigh;
        else if (effort == "low") instructions = kQwenEffortLow;
        else if (effort != "medium") {
            error = "Unexpected reasoning effort " + input.reasoning_effort +
                    ". Supported types are xhigh (default), medium, and low.";
            return false;
        }
    }

    // A system (or developer) message is the system prompt only first. Later
    // ones read as user turns in place (Claude Code sends its environment
    // that way), so the prompt before them stays the same and cached.
    const auto role = [&](size_t i) -> const std::string & {
        static const std::string kSystem = "system", kUser = "user";
        const std::string & given = messages[i].role;
        if (given != "system" && given != "developer") return given;
        return i == 0 ? kSystem : kUser;
    };
    const bool has_system = role(0) == "system";
    const std::string system = has_system ? python_strip(messages[0].content) : std::string();
    const bool has_tools = input.tools && input.tools->is_array() && !input.tools->empty();
    MessageLayout * system_layout = has_system ? &out.messages[0] : nullptr;
    if (has_tools) {
        // The tools hold the system turn up to the system message's part,
        // which runs to the turn's end.
        out.tools.begin = w.at();
        open("system\n");
        if (!instructions.empty()) w.text(instructions + "\n\n");
        w.text(kQwenToolsHeader);
        for (const auto & tool : openai_tool_definitions(*input.tools)) {
            w.text("\n" + python_json_dumps(tool));
        }
        w.text(kQwenToolsFooter);
        out.tools.end = w.at();
        if (!system.empty()) {
            system_layout->turn.begin = w.at();
            w.text("\n\n");
            w.content(system, input.image_marker, system_layout->content);
        }
        close();
        if (system.empty()) out.tools.end = w.at();
        else system_layout->turn.end = w.at();
    } else if (!system.empty() || !instructions.empty()) {
        const size_t begin = w.at();
        open("system\n");
        if (!instructions.empty()) w.text(system.empty() ? instructions : instructions + "\n\n");
        if (system_layout) {
            w.content(system, input.image_marker, system_layout->content);
        } else {
            w.content(system, input.image_marker);
        }
        close();
        if (system_layout) system_layout->turn = {begin, w.at()};
    }

    // The last user message that is not only a wrapped tool response.
    size_t last_query = messages.size();
    for (size_t i = messages.size(); i-- > 0;) {
        if (role(i) != "user") continue;
        const std::string content = python_strip(messages[i].content);
        if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
            last_query = i;
            break;
        }
    }
    if (last_query == messages.size()) {
        error = "No user query found in messages.";
        return false;
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage & message = messages[i];
        MessageLayout & layout = out.messages[i];
        if (role(i) == "system") continue;  // rendered above
        layout.turn.begin = w.at();
        if (role(i) == "user") {
            open("user\n");
            w.content(python_strip(message.content), input.image_marker, layout.content);
            close();
        } else if (message.role == "assistant" && message.tool_calls_replayed) {
            // Tool memory: the turn exactly as the model generated it.
            open("assistant\n");
            w.content(message.content, input.image_marker, layout.content);
            close();
        } else if (message.role == "assistant") {
            // Qwen3.8 and Qwen3.5 keep earlier thinking unless the request turns it off; Qwen3.6
            // only when asked.
            const bool keeps_earlier = version_ == QwenVersion::k36 ? input.preserve_thinking == 1
                                                                    : input.preserve_thinking != 0;
            const bool shows_thinking = keeps_earlier || i > last_query;
            TurnMemory::Turn generated;
            if (shows_thinking && input.memory && input.memory->lookup(message, w.context(), generated)) {
                // The turn exactly as the model generated it, from the role line on.
                w.control("<|im_start|>");
                const size_t role_line = generated.text.find('\n') + 1;
                w.generated(generated, message.content, role_line,
                            generated.text.compare(role_line, std::strlen("<think>"), "<think>") == 0, layout.content);
                close();
                layout.turn.end = w.at();
                continue;
            }
            std::string content = python_strip(message.content);
            std::string reasoning = message.reasoning_content;
            const size_t think_end = content.find("</think>");
            if (version_ != QwenVersion::k38 && !message.reasoning_sent && think_end != std::string::npos) {
                // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n'), and the rest.
                std::string before = content.substr(0, think_end);
                before.erase(before.find_last_not_of('\n') + 1);
                const size_t think_open = before.rfind("<think>");
                if (think_open != std::string::npos) before = before.substr(think_open + std::strlen("<think>"));
                reasoning = before.substr(std::min(before.find_first_not_of('\n'), before.size()));
                content = content.substr(content.rfind("</think>") + std::strlen("</think>"));
                content = content.substr(std::min(content.find_first_not_of('\n'), content.size()));
            }
            open("assistant\n");
            if (shows_thinking) {
                w.text("<think>\n" + python_strip(reasoning) + "\n</think>\n\n");
            }
            w.content(content, input.image_marker, layout.content);
            for (size_t c = 0; c < message.tool_calls.size(); ++c) {
                const auto & call = message.tool_calls[c];
                w.text(c != 0 ? "\n<tool_call>\n<function=" :
                       !python_strip(content).empty() ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
                w.text(call.name + ">\n");
                if (call.arguments.is_object()) {
                    for (auto it = call.arguments.begin(); it != call.arguments.end(); ++it) {
                        w.text("<parameter=" + it.key() + ">\n");
                        const auto & value = it.value();
                        if (value.is_string()) {
                            w.text(value.get<std::string>());
                        } else if (version_ == QwenVersion::k35 && value.is_boolean()) {
                            w.text(value.get<bool>() ? "True" : "False");  // str(True)
                        } else if (version_ == QwenVersion::k35 && value.is_null()) {
                            w.text("None");
                        } else {
                            w.text(python_json_dumps(value));  // str() of a number is its JSON
                        }
                        w.text("\n</parameter>\n");
                    }
                }
                w.text("</function>\n</tool_call>");
            }
            close();
        } else if (message.role == "tool") {
            if (i != 0 && role(i - 1) != "tool") open("user");
            w.text("\n<tool_response>\n");
            w.content(python_strip(message.content), input.image_marker, layout.content);
            w.text("\n</tool_response>");
            if (i + 1 == messages.size() || role(i + 1) != "tool") close();
        } else {
            error = "Unexpected message role.";
            return false;
        }
        layout.turn.end = w.at();
    }

    if (input.add_generation_prompt) {
        // The generated turn opens at the generation prompt's role line.
        out.context = w.context();
        open("assistant\n");
        w.text(input.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
    }
    return true;
}

// The JSON types a parameter schema allows: its "type" (one or a list), the
// types of its anyOf/oneOf/allOf branches, and those of its enum or const values.
// Empty when it names none.
void qwen_schema_types(const nlohmann::json & schema, std::vector<std::string> & types, int depth = 0) {
    if (!schema.is_object() || depth > 4) return;
    const auto add = [&](std::string type) {
        // The short names some tool schemas use.
        if (type == "str" || type == "text" || type == "enum") type = "string";
        else if (type == "int") type = "integer";
        else if (type == "float") type = "number";
        else if (type == "bool") type = "boolean";
        else if (type == "dict") type = "object";
        else if (type == "list") type = "array";
        if (std::find(types.begin(), types.end(), type) == types.end()) types.push_back(std::move(type));
    };
    const auto type_of = [](const nlohmann::json & v) -> std::string {
        if (v.is_string()) return "string";
        if (v.is_boolean()) return "boolean";
        if (v.is_number_integer()) return "integer";
        if (v.is_number()) return "number";
        if (v.is_null()) return "null";
        return v.is_array() ? "array" : "object";
    };
    if (schema.contains("type")) {
        const auto & t = schema["type"];
        if (t.is_string()) add(t.get<std::string>());
        for (const auto & e : t.is_array() ? t : nlohmann::json::array()) {
            if (e.is_string()) add(e.get<std::string>());
        }
    }
    for (const char * key : {"anyOf", "oneOf", "allOf"}) {
        const auto branches = schema.find(key);
        if (branches == schema.end() || !branches->is_array()) continue;
        for (const auto & branch : *branches) qwen_schema_types(branch, types, depth + 1);
    }
    if (schema.contains("enum") && schema["enum"].is_array()) {
        for (const auto & v : schema["enum"]) add(type_of(v));
    }
    if (schema.contains("const")) add(type_of(schema["const"]));
}

// Whether `value` may meet the schema: false only when it certainly does not,
// by the schema's type, enum, const, anyOf, oneOf (read as anyOf: its
// branches' other keywords are not checked) or allOf. Other keywords and
// unknown type names accept.
bool qwen_schema_accepts(const nlohmann::json & schema, const nlohmann::json & value, int depth = 0) {
    if (!schema.is_object() || depth > 8) return true;
    if (schema.contains("type")) {
        const auto & t = schema["type"];
        const auto is = [&](const std::string & type) {
            if (type == "null") return value.is_null();
            if (type == "boolean") return value.is_boolean();
            if (type == "integer") {
                return value.is_number_integer() ||
                       (value.is_number_float() && value.get<double>() == std::trunc(value.get<double>()));
            }
            if (type == "number") return value.is_number();
            if (type == "string") return value.is_string();
            if (type == "object") return value.is_object();
            if (type == "array") return value.is_array();
            return true;
        };
        bool any = false;
        if (t.is_string()) any = is(t.get<std::string>());
        for (const auto & e : t.is_array() ? t : nlohmann::json::array()) {
            if (e.is_string() && is(e.get<std::string>())) any = true;
        }
        if ((t.is_string() || t.is_array()) && !any) return false;
    }
    if (schema.contains("enum") && schema["enum"].is_array() &&
        std::find(schema["enum"].begin(), schema["enum"].end(), value) == schema["enum"].end()) {
        return false;
    }
    if (schema.contains("const") && schema["const"] != value) return false;
    const auto count = [&](const char * key) {
        const auto branches = schema.find(key);
        if (branches == schema.end() || !branches->is_array()) return -1;
        return (int) std::count_if(branches->begin(), branches->end(), [&](const nlohmann::json & branch) {
            return qwen_schema_accepts(branch, value, depth + 1);
        });
    };
    if (count("anyOf") == 0 || count("oneOf") == 0) return false;
    if (const auto all = schema.find("allOf"); all != schema.end() && all->is_array()) {
        for (const auto & branch : *all) {
            if (!qwen_schema_accepts(branch, value, depth + 1)) return false;
        }
    }
    return true;
}

// The text read as the first of `types` it is: null, a boolean, an integer,
// a number, an object, an array, else a string. Text that is none of them
// stays text, but under a type we do not know (then the JSON when it is JSON).
nlohmann::ordered_json qwen_typed_value(const std::string & text, const std::vector<std::string> & types,
                                        bool python) {
    const auto allows = [&](const char * type) {
        return std::find(types.begin(), types.end(), type) != types.end();
    };
    if (allows("null") && (text == "null" || (python && text == "None"))) return nullptr;
    if (allows("boolean")) {
        if (text == "true" || (python && text == "True")) return true;
        if (text == "false" || (python && text == "False")) return false;
    }
    const auto parsed = nlohmann::ordered_json::parse(text, nullptr, false);
    if (!parsed.is_discarded()) {
        if (allows("integer")) {
            if (parsed.is_number_integer()) return parsed;
            if (parsed.is_number_float()) {
                // A whole number in int64_t's range (2^63 is exact as a double).
                const double d = parsed.get<double>();
                if (d >= -9223372036854775808.0 && d < 9223372036854775808.0 && d == std::trunc(d)) {
                    return (int64_t) d;
                }
            }
        }
        if (allows("number") && parsed.is_number()) return parsed;
        if (allows("object") && parsed.is_object()) return parsed;
        if (allows("array") && parsed.is_array()) return parsed;
    }
    // Text that is none of them stays text, but under a type we do not know.
    static const char * const kKnown[] = {"null", "boolean", "integer", "number", "object", "array", "string"};
    const bool unknown = std::any_of(types.begin(), types.end(), [](const std::string & type) {
        return std::none_of(std::begin(kKnown), std::end(kKnown), [&](const char * k) { return type == k; });
    });
    return unknown && !parsed.is_discarded() ? parsed : nlohmann::ordered_json(text);
}

// A parameter value as the template wrote it (strings as they are, other
// values as JSON, or as Python's str() on Qwen3.5: `python`, where None, True
// and False are null and the booleans), read back with the types
// the declared schema allows, or as the text itself when the schema certainly
// rejects that reading (by its type, enum, const or branches) and may accept
// the text. A schema with no type reads as
// a string, as vLLM's Qwen3-Coder parser reads it. Undeclared parameters stay
// strings, as in parse_tool_calls.
nlohmann::ordered_json qwen_parameter_value(const std::string & text, const nlohmann::json & properties,
                                            const std::string & key, bool python) {
    const auto schema = properties.find(key);
    if (schema == properties.end()) return text == "null" ? nlohmann::ordered_json() : nlohmann::ordered_json(text);
    std::vector<std::string> types;
    qwen_schema_types(*schema, types);
    if (types.empty()) return text;
    // The typed reading, but the text itself when the schema certainly rejects
    // that reading and may accept the text (an enum such as [1, "2"], whose
    // "2" the template wrote as 2).
    nlohmann::ordered_json value = qwen_typed_value(text, types, python);
    if (!qwen_schema_accepts(*schema, nlohmann::json::parse(value.dump())) &&
        qwen_schema_accepts(*schema, nlohmann::json(text))) {
        return text;
    }
    return value;
}

bool tool_declared(const nlohmann::json & tools, const std::string & name) {
    if (!tools.is_array()) return false;
    for (const auto & tool : tools) {
        const auto & function = tool.contains("function") ? tool["function"] : tool;
        if (function.is_object() && function.value("name", "") == name) return true;
    }
    return false;
}

// Calls in the template's format from `pos`: "<tool_call>\n<function=NAME>\n",
// then "<parameter=KEY>\nVALUE\n</parameter>\n" per argument, then
// "</function>\n</tool_call>", calls joined by "\n". Whitespace may follow.
bool parse_qwen_calls(const std::string & text, size_t pos, const nlohmann::json & tools,
                      bool python, std::vector<ChatToolCall> & calls) {
    static const std::string kOpen = "<tool_call>\n<function=", kParam = "<parameter=",
                             kParamEnd = "\n</parameter>\n", kClose = "</function>\n</tool_call>";
    const auto at = [&](const std::string & token) { return text.compare(pos, token.size(), token) == 0; };
    // A function name or argument key: the text up to ">\n", on one line.
    const auto read_name = [&](std::string & name) {
        const size_t end = text.find(">\n", pos);
        if (end == std::string::npos || end == pos) return false;
        name = text.substr(pos, end - pos);
        if (name.find('\n') != std::string::npos) return false;
        pos = end + 2;
        return true;
    };
    calls.clear();
    while (true) {
        if (!at(kOpen)) return false;
        pos += kOpen.size();
        ChatToolCall call;
        if (!read_name(call.name) || !tool_declared(tools, call.name)) return false;
        const nlohmann::json properties = tool_parameter_properties(tools, call.name);
        call.arguments = nlohmann::ordered_json::object();
        while (at(kParam)) {
            pos += kParam.size();
            std::string key;
            if (!read_name(key) || call.arguments.contains(key)) return false;
            // The value ends at the first "\n</parameter>\n" the next argument or the end follows.
            size_t end = pos;
            while (true) {
                end = text.find(kParamEnd, end);
                if (end == std::string::npos) return false;
                const size_t next = end + kParamEnd.size();
                if (text.compare(next, kParam.size(), kParam) == 0 ||
                    text.compare(next, kClose.size(), kClose) == 0) break;
                ++end;
            }
            call.arguments[key] = qwen_parameter_value(text.substr(pos, end - pos), properties, key, python);
            pos = end + kParamEnd.size();
        }
        if (!at(kClose)) return false;
        pos += kClose.size();
        calls.push_back(std::move(call));
        if (at("\n<tool_call>")) {
            ++pos;
            continue;
        }
        return python_strip(text.substr(pos)).empty();
    }
}

bool QwenCodec::parse_tool_calls(const std::string & block, const nlohmann::json & tools,
                                   ToolParseResult & out) const {
    std::vector<ChatToolCall> calls;
    if (!parse_qwen_calls(block, 0, tools, version_ == QwenVersion::k35, calls)) return false;
    out = {};
    for (auto & call : calls) {
        out.tool_calls.push_back({new_tool_call_id(), std::move(call.name), call.arguments.dump()});
    }
    return true;
}

bool QwenCodec::generated_turn(const std::string & rendered, const std::vector<int32_t> & prompt_tokens,
                                 const std::vector<int32_t> & generated, const Tokenizer & tokenizer,
                                 const nlohmann::json & tools, ChatMessage & message,
                                 TurnMemory::Turn & turn) const {
    // The generation prompt's role line and what the template put after it
    // (the opened or closed think block), then the reply.
    static const std::string kRole = "<|im_start|>assistant\n";
    const size_t role = rendered.rfind(kRole);
    const int32_t im_start = tokenizer.token_to_id("<|im_start|>");
    const auto start = std::find(prompt_tokens.rbegin(), prompt_tokens.rend(), im_start);
    if (role == std::string::npos || im_start < 0 || start == prompt_tokens.rend()) return false;
    turn.tokens.assign(start.base(), prompt_tokens.end());
    turn.tokens.insert(turn.tokens.end(), generated.begin(), generated.end());
    turn.text = rendered.substr(role + std::strlen("<|im_start|>")) + tokenizer.decode(generated);
    if (tokenizer.decode(turn.tokens) != turn.text) return false;
    if (!parse_turn(turn.text.substr(std::strlen("assistant\n")), tools, message)) return false;
    turn.reasoning = message.reasoning_content;
    return true;
}

bool QwenCodec::parse_turn(const std::string & turn, const nlohmann::json & tools,
                             ChatMessage & out) const {
    out = {};
    out.role = "assistant";
    std::string rest = turn;
    if (starts_with(turn, "<think>\n")) {
        const size_t close = turn.find("</think>");
        if (close == std::string::npos) return false;
        out.reasoning_content = python_strip(turn.substr(8, close - 8));
        rest = turn.substr(close + 8);
        if (starts_with(rest, "\n\n")) rest.erase(0, 2);
    }
    const size_t call = rest.find("<tool_call>");
    out.content = python_strip(rest.substr(0, call));
    return call == std::string::npos ||
           parse_qwen_calls(rest, call, tools, version_ == QwenVersion::k35, out.tool_calls);
}


// DeepSeek V4 and V4.1: encoding/encoding_dsv4.py in deepseek-ai/DeepSeek-V4-Flash-0731
// and encoding/encoding.py in deepseek-ai/DeepSeek-V4.1-Flash-1017 at
// 1bae0e1088993b9b82de08f59de5be5a34e42170 (the models ship no Jinja
// template), with a Chat Completions request prepared as vLLM's V4 and V4.1
// tokenizers prepare it: the tools go on the first system message (an empty
// one is added first when there is none), and a developer message is a user turn.
const char kDsBos[] = "<｜begin▁of▁sentence｜>";
const char kDsEos[] = "<｜end▁of▁sentence｜>";
const char kDsUser[] = "<｜User｜>";
const char kDsAssistant[] = "<｜Assistant｜>";
const char kDsSystem[] = "<｜System｜>";

// Google's canonical template (google/gemma-4-26B-A4B-it/chat_template.jinja)
// uses special string delimiters and bare keys recursively for calls and
// responses, not JSON or Python repr().

// What differs between DeepSeek V4's encoder (encoding/encoding_dsv4.py in
// deepseek-ai/DeepSeek-V4-Flash-0731 at 7872f01b1d1fe23eabc4c98b48bffcef5a386062)
// and V4.1's: the DSML tag names, V4.1's <｜System｜> turns and numeric effort.
struct DeepSeekDialect {
    const char * id;
    bool v41;
    // V4: vLLM's generic renderer joins text parts with "\n"; V4.1's
    // _normalize_messages and content blocks with "\n\n".
    const char * text_part_separator;
    const char * calls_open;
    const char * calls_close;
    const char * invoke_open;
    const char * invoke_close;
    const char * param_open;
    const char * param_close;
};
const DeepSeekDialect kDeepSeek4 = {
    "deepseek4/2", false, "\n",
    "<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>",
    "<｜DSML｜invoke name=\"", "</｜DSML｜invoke>",
    "<｜DSML｜parameter name=\"", "</｜DSML｜parameter>",
};
const DeepSeekDialect kDeepSeek41 = {
    "deepseek4.1/2", true, "\n\n",
    "<｜DSML｜ calls>", "</｜DSML｜ calls>",
    "<｜DSML｜ invoke name=\"", "</｜DSML｜ invoke>",
    "<｜DSML｜ parameter name=\"", "</｜DSML｜ parameter>",
};
// V4's REASONING_EFFORT_PROMPTS, at the start of the prompt in thinking mode.
const char kDs4EffortHigh[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root "
    "cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial "
    "scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered "
    "alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const char kDs4EffortMax[] =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively "
    "decompose the problem into its most fundamental components, trace every causal chain to its root, and "
    "resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are "
    "certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

class DeepSeekCodec final : public ChatCodec {
public:
    explicit DeepSeekCodec(const DeepSeekDialect & dialect) : d(dialect), id_(dialect.id) {}
    const std::string & id() const override { return id_; }
    const char * text_part_separator() const override { return d.text_part_separator; }
    // The encoder puts the calls block after the content and "\n\n", even when the content is empty.
    std::vector<std::string> tool_call_openers() const override { return {std::string("\n\n") + d.calls_open}; }

    // DeepSeek's API takes low, high and max; xhigh is high, as its
    // OpenAI-compatible APIs read it. Lucebox's medium is high, x-high max.
    std::string effort_tier(const std::string & word) const override {
        return word == "xhigh" ? "high" : ChatCodec::effort_tier(word);
    }
    std::string template_effort(const std::string & tier) const override {
        if (tier == "medium") return "high";
        if (tier == "x-high") return "max";
        return tier;
    }
    // Thinking without an effort is high, as in vLLM and the V4.1 encoder.
    std::string default_effort() const override { return "high"; }
    // V4.1's encoder also takes an integer effort in [1, 100] (vLLM and SGLang pass it through).
    bool integer_effort() const override { return d.v41; }
    bool drop_thinking_kwarg() const override { return true; }
    std::vector<std::string> effort_tiers() const override { return {"low", "high", "max"}; }

    bool render(const ChatCodecInput & input, RenderedPrompt & out,
                std::string & error) const override;
    bool parse_tool_calls(const std::string & block, const nlohmann::json & tools,
                          ToolParseResult & out) const override;
    // `turn` is the assistant message as rendered where its thinking shows:
    // "REASONING</think>CONTENT", calls after "\n\n". Without "</think>" it
    // is read as a turn without thinking.
    bool parse_turn(const std::string & turn, const nlohmann::json & tools,
                    ChatMessage & out) const override;
    bool generated_turn(const std::string & rendered, const std::vector<int32_t> & prompt_tokens,
                        const std::vector<int32_t> & generated, const Tokenizer & tokenizer,
                        const nlohmann::json & tools, ChatMessage & message,
                        TurnMemory::Turn & turn) const override;

private:
    const DeepSeekDialect & d;
    std::string id_;
};


// encoding.py's REASONING_EFFORT_MAPPINGS (default "high"), or an integer 1-100.
int deepseek41_effort_budget(const std::string & effort) {
    if (effort.empty() || effort == "high") return 75;
    if (effort == "low") return 50;
    if (effort == "max") return 100;
    if (effort.size() <= 3 && effort.find_first_not_of("0123456789") == std::string::npos) {
        const int budget = std::stoi(effort);
        if (budget >= 1 && budget <= 100) return budget;
    }
    return -1;
}

// render_tools: the instructions in the dialect's tags, then each tool's
// function object on its own line in the client's key order.
std::string deepseek_tool_block(const DeepSeekDialect & d, const nlohmann::ordered_json & tools) {
    const std::string calls = d.calls_open, calls_end = d.calls_close;
    std::string out = "## Tools\n\nYou have access to a set of tools to help answer the user's question. "
                      "You can invoke tools by writing a \"" + calls + "\" block like the following:\n\n" +
                      calls + "\n" +
                      d.invoke_open + "$TOOL_NAME\">\n" +
                      d.param_open + "$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE" + d.param_close + "\n" +
                      "...\n" + d.invoke_close + "\n" +
                      d.invoke_open + "$TOOL_NAME2\">\n...\n" + d.invoke_close + "\n" +
                      calls_end + "\n\n"
                      "String parameters should be specified as is and set `string=\"true\"`. For all other types "
                      "(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
                      "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning "
                      "inside <think>...</think> BEFORE any tool calls or final response.\n\n"
                      "Otherwise, output directly after </think> with tool calls or final response.\n\n"
                      "### Available Tool Schemas\n\n";
    bool first = true;
    for (const auto & tool : openai_tool_definitions(tools)) {
        if (!first) out += "\n";
        first = false;
        out += python_json_dumps(tool.contains("function") ? tool["function"] : tool);
    }
    out += "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
    return out;
}

// encode_arguments_to_dsml: one parameter line per argument, in the call's
// order; strings as they are, other values as JSON.
std::string deepseek_call(const DeepSeekDialect & d, const ChatToolCall & call) {
    nlohmann::ordered_json arguments = call.arguments;
    if (!arguments.is_object()) {
        // The reference tolerates JSON text, then wraps anything else.
        if (arguments.is_string()) {
            const auto parsed = nlohmann::ordered_json::parse(arguments.get<std::string>(), nullptr, false);
            if (!parsed.is_discarded() && parsed.is_object()) arguments = parsed;
        }
        if (!arguments.is_object()) arguments = {{"arguments", call.arguments}};
    }
    std::string out = std::string(d.invoke_open) + call.name + "\">\n";
    bool first = true;
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
        if (!first) out += "\n";
        first = false;
        const bool is_string = it.value().is_string();
        out += std::string(d.param_open) + it.key() + "\" string=\"" + (is_string ? "true" : "false") + "\">";
        out += is_string ? it.value().get<std::string>() : python_json_dumps(it.value());
        out += d.param_close;
    }
    return out + "\n" + d.invoke_close;
}

bool DeepSeekCodec::render(const ChatCodecInput & input, RenderedPrompt & out,
                             std::string & error) const {
    out = RenderedPrompt();
    // The turn markers are user-defined tokens in the V4 and V4.1 vocabularies: typed
    // in a message they stay text, as control tokens do.
    out.literal = {kDsUser, kDsAssistant, kDsSystem, "<｜latest_reminder｜>"};
    if (!input.messages || input.messages->empty()) {
        error = "No messages provided.";
        return false;
    }
    // The effort line in thinking mode. V4: none for low; vLLM's default is high, as V4.1's encoder's.
    std::string effort_line;
    if (d.v41) {
        const int budget = deepseek41_effort_budget(input.reasoning_effort);
        if (budget < 0) {
            error = "Invalid reasoning effort " + input.reasoning_effort +
                    " for DeepSeek V4.1: low, high, max or an integer within [1, 100].";
            return false;
        }
        effort_line = "Reasoning Effort: " + std::to_string(budget) +
                      " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
    } else if (input.reasoning_effort.empty() || input.reasoning_effort == "high") {
        effort_line = kDs4EffortHigh;
    } else if (input.reasoning_effort == "max") {
        effort_line = kDs4EffortMax;
    } else if (input.reasoning_effort != "low") {
        error = "Invalid reasoning effort " + input.reasoning_effort + " for DeepSeek V4: low, high or max.";
        return false;
    }

    // merge_tool_messages: tool results become blocks of a user message, and
    // consecutive user messages (results included) are one user turn.
    // `index`: the input message a block or turn renders (npos for the
    // empty system turn that holds the tools).
    struct Block {
        bool tool_result = false;
        std::string text;
        std::string call_id;
        size_t index = std::string::npos;
    };
    struct Turn {
        std::string role;
        const ChatMessage * message = nullptr;
        std::vector<Block> blocks;
        bool tools = false;
        size_t index = std::string::npos;
    };
    std::vector<Turn> turns;
    out.messages.assign(input.messages->size(), MessageLayout());
    for (size_t m = 0; m < input.messages->size(); ++m) {
        const ChatMessage & message = (*input.messages)[m];
        const bool tool = message.role == "tool";
        if (tool || message.role == "user" || message.role == "developer") {
            Block block{tool, message.content, message.tool_call_id, m};
            if (!turns.empty() && turns.back().role == "user") {
                turns.back().blocks.push_back(std::move(block));
            } else {
                turns.push_back({"user", &message, {std::move(block)}, false, m});
            }
        } else if (message.role == "system" || message.role == "assistant") {
            turns.push_back({message.role, &message, {}, false, m});
        } else {
            error = "Invalid role: " + message.role;
            return false;
        }
    }
    static const ChatMessage kEmptySystem{"system"};
    const bool has_tools = input.tools && input.tools->is_array() && !input.tools->empty();
    if (has_tools) {
        auto system = std::find_if(turns.begin(), turns.end(),
                                   [](const Turn & t) { return t.role == "system"; });
        if (system == turns.end()) system = turns.insert(turns.begin(), {"system", &kEmptySystem, {}, false});
        system->tools = true;
    }
    // sort_tool_results_by_call_order: a turn's results in the order of the
    // calls of the assistant turn before it.
    std::map<std::string, size_t> call_order;
    for (Turn & turn : turns) {
        if (turn.role == "assistant" && !turn.message->tool_calls.empty()) {
            call_order.clear();
            for (size_t c = 0; c < turn.message->tool_calls.size(); ++c) {
                const auto & id = turn.message->tool_calls[c].id;
                if (!id.empty()) call_order[id] = c;
            }
        } else if (turn.role == "user" && !call_order.empty()) {
            std::vector<Block> results;
            for (const Block & block : turn.blocks) {
                if (block.tool_result) results.push_back(block);
            }
            if (results.size() < 2) continue;
            const auto order = [&](const Block & b) {
                const auto it = call_order.find(b.call_id);
                return it == call_order.end() ? size_t(0) : it->second;
            };
            std::stable_sort(results.begin(), results.end(),
                             [&](const Block & a, const Block & b) { return order(a) < order(b); });
            size_t next = 0;
            for (Block & block : turn.blocks) {
                if (block.tool_result) block = results[next++];
            }
        }
    }
    // find_last_user_index: a user turn; V4.1 also counts a system message
    // after the first, and skips a user turn that holds only tool results.
    size_t last_user = std::string::npos;
    for (size_t i = turns.size(); i-- > 0;) {
        const Turn & turn = turns[i];
        const bool user = turn.role == "user" || (d.v41 && turn.role == "system" && i > 0);
        if (!user) continue;
        if (d.v41 && turn.role == "user" &&
            std::none_of(turn.blocks.begin(), turn.blocks.end(),
                         [](const Block & b) { return !b.tool_result; })) {
            continue;
        }
        last_user = i;
        break;
    }
    // With no user turn the reference's index is -1: every turn is after it.
    const auto after_last_user = [&](size_t i, bool or_at) {
        return last_user == std::string::npos || (or_at ? i >= last_user : i > last_user);
    };
    const bool thinking = input.enable_thinking;
    // Earlier thinking is dropped unless the request keeps it, and never
    // when the conversation declares tools.
    const bool drop_thinking = input.preserve_thinking != 1 && !has_tools;

    PromptWriter w(out);
    w.control(kDsBos);
    MessageLayout scratch;
    const auto layout_of = [&](size_t index) -> MessageLayout & {
        return index == std::string::npos ? scratch : out.messages[index];
    };
    for (size_t i = 0; i < turns.size(); ++i) {
        const Turn & turn = turns[i];
        const ChatMessage & message = *turn.message;
        // A first system turn holds the system marker and effort line before it.
        const size_t turn_begin = w.at();
        if (i == 0 && d.v41 && (thinking || turn.role == "system")) w.control(kDsSystem);
        if (i == 0 && thinking) w.text(effort_line);
        if (turn.role == "system") {
            MessageLayout & layout = layout_of(turn.index);
            layout.turn.begin = turn_begin;
            // V4 has no system marker: a later system message is its text in place.
            if (i > 0 && d.v41) w.control(kDsSystem);
            w.content(message.content, input.image_marker, layout.content);
            layout.turn.end = w.at();
            if (turn.tools) {
                // The tools' own system turn, when no system message has one, is theirs.
                out.tools.begin = turn.index == std::string::npos ? turn_begin : w.at();
                w.text("\n\n" + deepseek_tool_block(d, *input.tools));
                out.tools.end = w.at();
            }
        } else if (turn.role == "user") {
            // Each message is its own block of the merged user turn; the
            // first block's part holds the turn marker.
            const size_t marker = w.at();
            w.control(kDsUser);
            for (size_t b = 0; b < turn.blocks.size(); ++b) {
                if (b != 0) w.text("\n\n");
                const Block & block = turn.blocks[b];
                MessageLayout & layout = layout_of(block.index);
                layout.turn.begin = b == 0 ? marker : w.at();
                if (block.tool_result) w.text("<tool_result>");
                w.content(block.text, input.image_marker, layout.content);
                if (block.tool_result) w.text("</tool_result>");
                layout.turn.end = w.at();
            }
        } else {
            MessageLayout & layout = layout_of(turn.index);
            layout.turn.begin = w.at();
            const bool shows_thinking = thinking && (!drop_thinking || after_last_user(i, false));
            TurnMemory::Turn generated;
            if (message.tool_calls_replayed) {
                w.content(message.content, input.image_marker, layout.content);
            } else if (input.memory && input.memory->lookup(message, w.context(), generated) &&
                       generated.thinking == shows_thinking) {
                // The turn exactly as the model generated it.
                w.generated(generated, message.content, 0, generated.thinking, layout.content);
            } else {
                if (shows_thinking) w.text(message.reasoning_content + "</think>");
                w.content(message.content, input.image_marker, layout.content);
                if (!message.tool_calls.empty()) {
                    std::string calls;
                    for (const auto & call : message.tool_calls) {
                        calls += (calls.empty() ? "" : "\n") + deepseek_call(d, call);
                    }
                    w.text(std::string("\n\n") + d.calls_open + "\n" + calls + "\n" + d.calls_close);
                }
            }
            w.control(kDsEos);
            layout.turn.end = w.at();
        }
        // The assistant header follows a user turn (or, V4.1, a later system
        // message) that an assistant turn, or the generation, follows. A
        // final assistant message (prefill) is closed and the generation
        // opens a new turn, as on the other codecs: the encoder would leave
        // the generation after its end-of-sentence.
        const bool last = i + 1 == turns.size();
        if (!last && turns[i + 1].role != "assistant") continue;
        if (last && !input.add_generation_prompt) continue;
        if (turn.role == "user" || (d.v41 && turn.role == "system" && i > 0) ||
            (last && turn.role == "assistant")) {
            w.control(kDsAssistant);
            w.text(thinking && (!drop_thinking || after_last_user(i, true)) ? "<think>" : "</think>");
        }
    }
    // The generated turn opens after the generation prompt's assistant header.
    if (input.add_generation_prompt) out.context = w.context();
    return true;
}

// The calls block from `pos`: "<｜DSML｜ calls>\n", invokes joined by "\n",
// each "<｜DSML｜ invoke name=\"NAME\">\n" + parameters joined by "\n" +
// "\n</｜DSML｜ invoke>", then "\n</｜DSML｜ calls>". Whitespace may follow.
bool parse_deepseek_calls(const DeepSeekDialect & d, const std::string & text, size_t pos,
                         const nlohmann::json & tools, std::vector<ChatToolCall> & calls) {
    const auto at = [&](const std::string & token) { return text.compare(pos, token.size(), token) == 0; };
    const auto take = [&](const std::string & token) {
        if (!at(token)) return false;
        pos += token.size();
        return true;
    };
    // An attribute value: the text up to the closing quote, on one line.
    const auto read_quoted = [&](std::string & value) {
        const size_t end = text.find('"', pos);
        if (end == std::string::npos || end == pos) return false;
        value = text.substr(pos, end - pos);
        if (value.find('\n') != std::string::npos) return false;
        pos = end + 1;
        return true;
    };
    calls.clear();
    if (!take(std::string(d.calls_open) + "\n")) return false;
    while (true) {
        ChatToolCall call;
        if (!take(d.invoke_open) || !read_quoted(call.name) || !take(">\n") ||
            !tool_declared(tools, call.name)) {
            return false;
        }
        call.arguments = nlohmann::ordered_json::object();
        const std::string invoke_end = std::string("\n") + d.invoke_close;
        for (bool first = true; !at(invoke_end); first = false) {
            if (!first && !take("\n")) return false;
            std::string key, is_string;
            if (!take(d.param_open) || !read_quoted(key) || !take(" string=\"") ||
                !read_quoted(is_string) || !take(">") || call.arguments.contains(key)) {
                return false;
            }
            // The value ends at the first close the next parameter or the end of the call follows.
            size_t end = pos;
            const std::string close = d.param_close;
            while (true) {
                end = text.find(close, end);
                if (end == std::string::npos) return false;
                const size_t next = end + close.size();
                if (text.compare(next, 1 + std::strlen(d.param_open), std::string("\n") + d.param_open) == 0 ||
                    text.compare(next, 1 + std::strlen(d.invoke_close), std::string("\n") + d.invoke_close) == 0) {
                    break;
                }
                ++end;
            }
            const std::string value = text.substr(pos, end - pos);
            if (is_string == "true") {
                call.arguments[key] = value;
            } else if (is_string == "false") {
                const auto parsed = nlohmann::ordered_json::parse(value, nullptr, false);
                if (parsed.is_discarded()) return false;
                call.arguments[key] = parsed;
            } else {
                return false;
            }
            pos = end + close.size();
        }
        if (!take(std::string("\n") + d.invoke_close)) return false;
        calls.push_back(std::move(call));
        if (take(std::string("\n") + d.calls_close)) break;
        if (!take("\n")) return false;
    }
    return python_strip(text.substr(pos)).empty();
}

bool DeepSeekCodec::parse_tool_calls(const std::string & block, const nlohmann::json & tools,
                                       ToolParseResult & out) const {
    std::vector<ChatToolCall> calls;
    if (!parse_deepseek_calls(d, block, 0, tools, calls)) return false;
    out = {};
    for (auto & call : calls) {
        out.tool_calls.push_back({new_tool_call_id(), std::move(call.name), call.arguments.dump()});
    }
    return true;
}

bool DeepSeekCodec::parse_turn(const std::string & turn, const nlohmann::json & tools,
                                 ChatMessage & out) const {
    out = {};
    out.role = "assistant";
    std::string rest = turn;
    const size_t think_end = turn.find("</think>");
    if (think_end != std::string::npos) {
        out.reasoning_content = turn.substr(0, think_end);
        rest = turn.substr(think_end + std::strlen("</think>"));
    }
    const size_t calls = rest.find(std::string("\n\n") + d.calls_open);
    out.content = rest.substr(0, calls);
    return calls == std::string::npos || parse_deepseek_calls(d, rest, calls + 2, tools, out.tool_calls);
}

bool DeepSeekCodec::generated_turn(const std::string & rendered, const std::vector<int32_t> & prompt_tokens,
                                     const std::vector<int32_t> & generated, const Tokenizer & tokenizer,
                                     const nlohmann::json & tools, ChatMessage & message,
                                     TurnMemory::Turn & turn) const {
    (void) prompt_tokens;
    // The generation prompt opened the thinking ("<think>") or closed it.
    const bool thinking = ends_with(rendered, "<｜Assistant｜><think>");
    if (!thinking && !ends_with(rendered, "<｜Assistant｜></think>")) return false;
    turn.tokens = generated;
    turn.text = tokenizer.decode(generated);
    turn.thinking = thinking;
    if (thinking && turn.text.find("</think>") == std::string::npos) return false;
    if (!thinking) {
        // Read without thinking: "</think>" in the reply is content.
        message = {};
        message.role = "assistant";
        const size_t calls = turn.text.find(std::string("\n\n") + d.calls_open);
        message.content = turn.text.substr(0, calls);
        if (calls != std::string::npos &&
            !parse_deepseek_calls(d, turn.text, calls + 2, tools, message.tool_calls)) {
            return false;
        }
    } else if (!parse_turn(turn.text, tools, message)) {
        return false;
    }
    turn.reasoning = message.reasoning_content;
    return true;
}

}  // namespace

std::string lucebox_effort_tier(const std::string & word) {
    if (word == "minimal" || word == "low") return "low";
    if (word == "medium") return "medium";
    if (word == "xhigh" || word == "x-high") return "x-high";
    if (word == "max") return "max";
    return "high";
}

std::string ChatCodec::effort_tier(const std::string & word) const { return lucebox_effort_tier(word); }

std::unique_ptr<ChatCodec> make_chat_codec(const std::string & name) {
    if (name == "qwen3.8") return std::make_unique<QwenCodec>(QwenVersion::k38);
    if (name == "qwen3.6") return std::make_unique<QwenCodec>(QwenVersion::k36);
    if (name == "qwen3.5") return std::make_unique<QwenCodec>(QwenVersion::k35);
    if (name == "deepseek4") return std::make_unique<DeepSeekCodec>(kDeepSeek4);
    if (name == "deepseek4.1") return std::make_unique<DeepSeekCodec>(kDeepSeek41);
    return nullptr;
}

std::string default_chat_codec_for_arch(const std::string & arch) {
    if (arch == "deepseek41") return "deepseek4.1";
    if (arch == "deepseek4") return "deepseek4";
    if (arch == "qwen4exp") return "qwen3.8";
    // Qwen3.5 27B and 35B-A3B; the Qwen3.6 and Qwen3.8 27B cards name their own codec.
    if (arch == "qwen35" || arch == "qwen35moe") return "qwen3.5";
    return {};
}

const std::vector<std::string> & chat_codec_names() {
    static const std::vector<std::string> kNames = {"qwen3.8", "qwen3.6", "qwen3.5", "deepseek4.1", "deepseek4"};
    return kNames;
}

}  // namespace luce::common
