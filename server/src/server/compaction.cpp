#include "compaction.h"

#include "common/model_backend.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace dflash::common {
namespace {

int message_chars(const std::vector<ChatMessage> & messages) {
    int total = 0;
    for (const auto & msg : messages) {
        total += (int)msg.role.size();
        total += (int)msg.content.size();
        total += (int)msg.tool_call_id.size();
    }
    return total;
}

std::string trim(std::string text) {
    const auto is_ws = [](unsigned char ch) {
        return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t';
    };
    while (!text.empty() && is_ws((unsigned char)text.front())) {
        text.erase(text.begin());
    }
    while (!text.empty() && is_ws((unsigned char)text.back())) {
        text.pop_back();
    }
    return text;
}

bool replace_content(ChatMessage & msg, const std::string & content) {
    if (msg.content == content) return false;
    msg.content = content;
    return true;
}

std::string truncate_tool_content(const std::string & content) {
    static constexpr size_t kHeadChars = 1536;
    static constexpr size_t kTailChars = 768;
    static constexpr size_t kMinTruncate = 4096;
    if (content.size() <= kMinTruncate) return content;
    return content.substr(0, kHeadChars)
        + "\n...[tool result truncated during context compaction]...\n"
        + content.substr(content.size() - kTailChars);
}

std::string quoted_value_after(const std::string & text, const std::string & key) {
    size_t pos = text.find(key);
    if (pos == std::string::npos) return {};
    pos = text.find(':', pos + key.size());
    if (pos == std::string::npos) return {};
    pos = text.find('"', pos);
    if (pos == std::string::npos) return {};
    size_t end = text.find('"', pos + 1);
    if (end == std::string::npos || end <= pos + 1) return {};
    return text.substr(pos + 1, end - pos - 1);
}

std::string default_compaction_prompt() {
    return "You are compacting earlier chat context for an inference server. "
           "Write a concise factual summary that preserves user goals, constraints, decisions, "
           "open tasks, important tool findings, and any data the assistant must remember. "
           "Do not invent details. Output summary text only.";
}

std::string serialize_messages(const std::vector<ChatMessage> & messages) {
    std::ostringstream out;
    for (const auto & msg : messages) {
        out << "[" << msg.role;
        if (!msg.tool_call_id.empty()) {
            out << " tool_call_id=" << msg.tool_call_id;
        }
        out << "]\n" << msg.content << "\n\n";
    }
    return out.str();
}

size_t leading_system_count(const std::vector<ChatMessage> & messages) {
    size_t count = 0;
    while (count < messages.size() && messages[count].role == "system") {
        ++count;
    }
    return count;
}

}  // namespace

std::string strip_thinking_blocks(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    bool removed = false;

    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = text.find("<think>", pos);
        if (start == std::string::npos) {
            out.append(text, pos, std::string::npos);
            break;
        }
        removed = true;
        out.append(text, pos, start - pos);
        size_t end = text.find("</think>", start + 7);
        if (end == std::string::npos) {
            break;
        }
        pos = end + 8;
    }

    return removed ? trim(out) : text;
}

bool is_tool_result(const ChatMessage & msg) {
    return msg.role == "tool";
}

std::string extract_file_read_path(const ChatMessage & msg) {
    if (!is_tool_result(msg)) return {};

    for (const std::string key : {std::string("\"path\""), std::string("\"file_path\"")}) {
        std::string value = quoted_value_after(msg.content, key);
        if (!value.empty()) return value;
    }

    if (msg.content.rfind("Path ", 0) == 0) {
        size_t end = msg.content.find('\n');
        return msg.content.substr(5, end == std::string::npos ? std::string::npos : end - 5);
    }

    return {};
}

CompactionResult edit_compact(const std::vector<ChatMessage> & messages,
                              int keep_tool_uses,
                              bool strip_thinking) {
    CompactionResult result;
    result.pre_compaction_tokens = message_chars(messages);
    result.compacted_messages = messages;

    if (messages.empty()) return result;

    const int tool_limit = std::max(0, keep_tool_uses);
    int last_assistant = -1;
    for (int i = 0; i < (int)messages.size(); ++i) {
        if (messages[i].role == "assistant") last_assistant = i;
    }

    bool changed = false;
    int seen_tool_results = 0;
    std::unordered_set<std::string> kept_paths;

    for (int i = (int)result.compacted_messages.size() - 1; i >= 0; --i) {
        auto & msg = result.compacted_messages[(size_t)i];

        if (strip_thinking && msg.role == "assistant" && i != last_assistant) {
            changed |= replace_content(msg, strip_thinking_blocks(msg.content));
        }

        if (!is_tool_result(msg)) continue;

        ++seen_tool_results;
        const std::string path = extract_file_read_path(msg);
        if (!path.empty()) {
            auto inserted = kept_paths.insert(path);
            if (!inserted.second) {
                changed |= replace_content(
                    msg,
                    std::string("[Earlier tool result omitted during context compaction; a newer read of ")
                        + path + " is kept.]");
                continue;
            }
        }

        if (seen_tool_results > tool_limit) {
            changed |= replace_content(
                msg,
                path.empty()
                    ? std::string("[Tool result omitted during context compaction.]")
                    : std::string("[Tool result omitted during context compaction for ")
                        + path + ".]");
            continue;
        }

        changed |= replace_content(msg, truncate_tool_content(msg.content));
    }

    result.applied = changed;
    if (changed) {
        result.tokens_saved = std::max(0, result.pre_compaction_tokens - message_chars(result.compacted_messages));
    }
    return result;
}

CompactionResult summarize_compact(const std::vector<ChatMessage> & messages,
                                   float keep_recent_ratio,
                                   int max_summary_tokens,
                                   const std::string & compaction_prompt,
                                   void * backend_ptr,
                                   void * tokenizer_ptr,
                                   int chat_format) {
    CompactionResult result;
    result.pre_compaction_tokens = message_chars(messages);
    result.compacted_messages = messages;

    auto * backend = static_cast<ModelBackend *>(backend_ptr);
    auto * tokenizer = static_cast<Tokenizer *>(tokenizer_ptr);
    if (!backend || !tokenizer || messages.size() < 3) {
        return result;
    }

    const size_t system_prefix = leading_system_count(messages);
    const size_t non_system = messages.size() - system_prefix;
    if (non_system < 3) return result;

    const float clamped_ratio = std::max(0.05f, std::min(0.95f, keep_recent_ratio));
    const size_t keep_recent = std::max<size_t>(1, (size_t)std::ceil((double)non_system * clamped_ratio));
    if (keep_recent >= non_system) return result;

    const size_t summary_end = messages.size() - keep_recent;
    if (summary_end <= system_prefix) return result;

    std::vector<ChatMessage> older(messages.begin() + system_prefix,
                                   messages.begin() + summary_end);
    std::vector<ChatMessage> recent(messages.begin() + summary_end,
                                    messages.end());
    if (older.empty()) return result;

    const std::string prompt_text = compaction_prompt.empty()
        ? default_compaction_prompt()
        : compaction_prompt;

    std::vector<ChatMessage> summary_request = {
        ChatMessage{"system", prompt_text, ""},
        ChatMessage{"user", std::string("Summarize this earlier conversation history for future continuation:\n\n")
            + serialize_messages(older), ""}
    };

    const std::string rendered = render_chat_template(
        summary_request,
        static_cast<ChatFormat>(chat_format),
        true,
        false,
        "");

    GenerateRequest gen_req;
    gen_req.prompt = tokenizer->encode(rendered);
    gen_req.n_gen = std::max(64, max_summary_tokens);
    gen_req.sampler.temp = 0.0f;
    gen_req.do_sample = false;
    gen_req.stream = false;

    DaemonIO io;
    auto gen_result = backend->generate(gen_req, io);
    if (!gen_result.ok || gen_result.tokens.empty()) {
        return result;
    }

    std::string summary = trim(strip_thinking_blocks(tokenizer->decode(gen_result.tokens)));
    if (summary.empty()) {
        return result;
    }

    std::vector<ChatMessage> compacted;
    compacted.reserve(system_prefix + 1 + recent.size());
    compacted.insert(compacted.end(), messages.begin(), messages.begin() + system_prefix);
    compacted.push_back(ChatMessage{"system", std::string("Conversation summary:\n") + summary, ""});
    compacted.insert(compacted.end(), recent.begin(), recent.end());

    const int compacted_chars = message_chars(compacted);
    if (compacted_chars >= result.pre_compaction_tokens) {
        return result;
    }

    result.applied = true;
    result.compacted_messages = std::move(compacted);
    result.tokens_saved = result.pre_compaction_tokens - compacted_chars;
    return result;
}

CompactionResult hard_truncate(const std::vector<ChatMessage> & messages,
                               int max_messages_to_keep) {
    CompactionResult result;
    result.pre_compaction_tokens = message_chars(messages);
    result.compacted_messages = messages;

    const size_t system_prefix = leading_system_count(messages);
    const int keep = std::max(0, max_messages_to_keep);
    const size_t non_system = messages.size() - system_prefix;
    if ((size_t)keep >= non_system) {
        return result;
    }

    const size_t tail_start = messages.size() - (size_t)keep;
    std::vector<ChatMessage> compacted;
    compacted.reserve(system_prefix + (size_t)keep);
    compacted.insert(compacted.end(), messages.begin(), messages.begin() + system_prefix);
    compacted.insert(compacted.end(), messages.begin() + std::max(system_prefix, tail_start), messages.end());

    result.applied = true;
    result.compacted_messages = std::move(compacted);
    result.tokens_saved = std::max(0, result.pre_compaction_tokens - message_chars(result.compacted_messages));
    return result;
}

}  // namespace dflash::common
