// Prompt normalization — volatile-header stripping for stable cache keys.

#include "prompt_normalize.h"
#include <algorithm>

namespace dflash::common {

static constexpr std::string_view kBillingHeader = "x-anthropic-billing-header:";

// Returns true if `s`, after skipping leading whitespace, starts with kBillingHeader.
static bool is_billing_header_block(const std::string & s) {
    auto pos = s.find_first_not_of(" \t\r\n");
    if (pos == std::string::npos) return false;
    return s.compare(pos, kBillingHeader.size(), kBillingHeader) == 0;
}

// Strip any line whose ltrimmed text starts with kBillingHeader from a multi-line string.
static std::string strip_billing_header_lines(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    std::string::size_type start = 0;
    while (start <= s.size()) {
        auto end = s.find('\n', start);
        std::string_view line = (end == std::string::npos)
            ? std::string_view(s).substr(start)
            : std::string_view(s).substr(start, end - start);
        // ltrim check
        auto nws = line.find_first_not_of(" \t\r");
        bool is_header = (nws != std::string_view::npos) &&
                         (line.substr(nws, kBillingHeader.size()) == kBillingHeader);
        if (!is_header) {
            out.append(line);
            if (end != std::string::npos) out += '\n';
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

std::string normalize_system_for_cache(const json & system_or_messages) {
    if (system_or_messages.is_array()) {
        if (system_or_messages.empty()) return "";
        const auto & first = system_or_messages[0];
        if (first.is_object() && first.contains("role")) {
            // OpenAI messages array: strip billing-header lines from messages[0].
            if (first.value("role", "") == "system") {
                const auto & content = first["content"];
                if (content.is_string()) {
                    return strip_billing_header_lines(content.get<std::string>());
                }
                if (content.is_array()) {
                    std::string out;
                    for (const auto & block : content) {
                        if (block.is_object() && block.value("type", "") == "text") {
                            out += block.value("text", "");
                        }
                    }
                    return strip_billing_header_lines(out);
                }
            }
            return "";
        }
        // Anthropic content-block array: skip billing-header blocks entirely.
        std::string out;
        for (const auto & block : system_or_messages) {
            if (block.is_object() && block.value("type", "") == "text") {
                std::string text = block.value("text", "");
                if (!is_billing_header_block(text)) out += text;
            }
        }
        return out;
    }

    if (system_or_messages.is_string()) {
        return strip_billing_header_lines(system_or_messages.get<std::string>());
    }

    return "";
}

}  // namespace dflash::common
