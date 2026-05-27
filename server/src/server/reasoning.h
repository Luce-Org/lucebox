// Reasoning parser — extracts <think>...</think> blocks from generated text.
//
// Ported from server.py parse_reasoning().

#pragma once

#include <string>
#include <utility>

namespace dflash::common {

struct ReasoningResult {
    std::string content;    // cleaned content (think tags removed)
    std::string reasoning;  // reasoning text (empty if none)
    bool has_reasoning = false;
};

// Extract reasoning from <think>...</think> blocks.
// `started_in_thinking` accounts for prompts ending with "<think>\n"
// so the generated text begins mid-reasoning.
ReasoningResult parse_reasoning(const std::string & text,
                                bool thinking_enabled = true,
                                bool started_in_thinking = false);

}  // namespace dflash::common
