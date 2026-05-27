// Reasoning parser implementation.

#include "reasoning.h"

namespace dflash::common {

static const char THINK_OPEN[]  = "<think>";
static const char THINK_CLOSE[] = "</think>";
static constexpr size_t THINK_OPEN_LEN  = 7;   // strlen("<think>")
static constexpr size_t THINK_CLOSE_LEN = 8;   // strlen("</think>")

// Strip leading </think> tags (with optional whitespace) from the start.
static std::string strip_leading_think_closers(const std::string & s) {
    size_t pos = 0;
    while (pos < s.size()) {
        // Skip whitespace
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' ||
               s[pos] == '\r' || s[pos] == '\t'))
            pos++;
        if (pos + THINK_CLOSE_LEN <= s.size() &&
            s.compare(pos, THINK_CLOSE_LEN, THINK_CLOSE) == 0) {
            pos += THINK_CLOSE_LEN;
        } else {
            break;
        }
    }
    // Trim leading/trailing whitespace from result
    size_t end = s.size();
    while (end > pos && (s[end-1] == ' ' || s[end-1] == '\n' ||
           s[end-1] == '\r' || s[end-1] == '\t'))
        end--;
    if (pos >= end) return "";
    return s.substr(pos, end - pos);
}

static std::string trim(const std::string & s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

ReasoningResult parse_reasoning(const std::string & text,
                                bool thinking_enabled,
                                bool started_in_thinking) {
    ReasoningResult result;

    // Split on first <think> tag
    size_t think_pos = text.find(THINK_OPEN);
    bool saw_open_tag = (think_pos != std::string::npos);

    std::string rest;
    if (saw_open_tag) {
        // Text before <think> is prefix (discarded from reasoning perspective)
        rest = text.substr(think_pos + THINK_OPEN_LEN);
    } else {
        rest = text;
    }

    // Look for </think> in rest
    size_t close_pos = rest.find(THINK_CLOSE);
    if (close_pos == std::string::npos) {
        // No closing tag
        if (thinking_enabled && (started_in_thinking || saw_open_tag)) {
            // All of rest is reasoning, no content
            std::string r = trim(rest);
            result.content = "";
            result.reasoning = r;
            result.has_reasoning = !r.empty();
        } else {
            result.content = strip_leading_think_closers(rest);
            result.has_reasoning = false;
        }
        return result;
    }

    // Have both open (explicit or implicit) and close
    std::string reasoning = rest.substr(0, close_pos);
    std::string content = rest.substr(close_pos + THINK_CLOSE_LEN);

    result.reasoning = trim(reasoning);
    result.has_reasoning = !result.reasoning.empty();
    result.content = strip_leading_think_closers(content);

    return result;
}

}  // namespace dflash::common
