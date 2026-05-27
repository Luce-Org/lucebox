// Tool call hint generator implementation.

#include "tool_hint.h"
#include "tokenizer.h"

#include <algorithm>
#include <string>

namespace dflash::common {

// ─── HintStateMachine ───────────────────────────────────────────────────

HintStateMachine::HintStateMachine(ToolCallHint hint)
    : hint_(std::move(hint)), segment_idx_(0), offset_in_segment_(0) {
    if (hint_.segments.empty()) {
        phase_ = Phase::DONE;
    } else if (hint_.segments[0].is_forced) {
        phase_ = Phase::FORCED;
    } else {
        phase_ = Phase::GAP;
        // Pre-compute gap end pattern: look at the next forced segment's
        // first few tokens as the pattern to detect gap end.
        if (segment_idx_ + 1 < (int)hint_.segments.size() &&
            hint_.segments[segment_idx_ + 1].is_forced) {
            const auto & next = hint_.segments[segment_idx_ + 1].tokens;
            // Use up to 4 tokens as the detection pattern.
            int pat_len = std::min((int)next.size(), 4);
            gap_end_tokens_.assign(next.begin(), next.begin() + pat_len);
        }
    }
}

std::vector<int32_t> HintStateMachine::get_hint_batch(int max_tokens) const {
    if (phase_ != Phase::FORCED) return {};
    if (segment_idx_ >= (int)hint_.segments.size()) return {};

    const auto & seg = hint_.segments[segment_idx_];
    int remaining = (int)seg.tokens.size() - offset_in_segment_;
    int n = std::min(remaining, max_tokens);
    if (n <= 0) return {};

    return std::vector<int32_t>(
        seg.tokens.begin() + offset_in_segment_,
        seg.tokens.begin() + offset_in_segment_ + n);
}

void HintStateMachine::advance(int n_tokens) {
    if (phase_ == Phase::DONE) return;

    if (phase_ == Phase::FORCED) {
        offset_in_segment_ += n_tokens;
        const auto & seg = hint_.segments[segment_idx_];
        if (offset_in_segment_ >= (int)seg.tokens.size()) {
            // Finished this forced segment, advance to next.
            advance_to_next_segment();
        }
    }
    // In GAP phase, we just track progress — the gap ends when
    // end_gap() is called externally.
}

void HintStateMachine::end_gap() {
    if (phase_ != Phase::GAP) return;
    advance_to_next_segment();
}

void HintStateMachine::advance_to_next_segment() {
    segment_idx_++;
    offset_in_segment_ = 0;
    gap_end_tokens_.clear();

    if (segment_idx_ >= (int)hint_.segments.size()) {
        phase_ = Phase::DONE;
        return;
    }

    const auto & seg = hint_.segments[segment_idx_];
    if (seg.is_forced) {
        phase_ = Phase::FORCED;
    } else {
        phase_ = Phase::GAP;
        // Pre-compute gap end pattern from next forced segment.
        if (segment_idx_ + 1 < (int)hint_.segments.size() &&
            hint_.segments[segment_idx_ + 1].is_forced) {
            const auto & next = hint_.segments[segment_idx_ + 1].tokens;
            int pat_len = std::min((int)next.size(), 4);
            gap_end_tokens_.assign(next.begin(), next.begin() + pat_len);
        }
    }
}

// ─── ToolHintGenerator ──────────────────────────────────────────────────

std::vector<int32_t> ToolHintGenerator::tokenize(const std::string & text) const {
    // Use encode() which handles special tokens (e.g. <tool_call> → 248058)
    // and applies the correct pre-tokenizer + BPE pipeline to match model output.
    return tok_.encode(text);
}

ToolCallHint ToolHintGenerator::build_function_hint(const json & function_def) const {
    ToolCallHint hint;
    hint.function_name = function_def.value("name", "");

    // Build the structural prefix AFTER <tool_call> (which the model produces as
    // its first token from prefill argmax and becomes the spec-decode anchor).
    // The hint provides everything that follows: \n<function=NAME>\n<parameter=FIRST_PARAM>\n
    std::string prefix = "\n<function=" + hint.function_name + ">\n";

    // Add first parameter name if available (gives more hint tokens).
    const auto & params = function_def.value("parameters", json::object());
    std::vector<std::string> param_names;

    // Collect required params first, then all properties.
    if (params.contains("required") && params["required"].is_array()) {
        for (const auto & r : params["required"]) {
            if (r.is_string()) param_names.push_back(r.get<std::string>());
        }
    } else if (params.contains("properties") && params["properties"].is_object()) {
        for (const auto & [key, _] : params["properties"].items()) {
            param_names.push_back(key);
        }
    }

    // Build segmented structure for gap-aware hinting.
    // Segment 1: opening prefix up to first parameter value
    if (!param_names.empty()) {
        prefix += "<parameter=" + param_names[0] + ">\n";
    }

    hint.prefix_tokens = tokenize(prefix);

    // Build full segments for advanced gap-aware hinting.
    HintSegment opening;
    opening.tokens = hint.prefix_tokens;
    opening.is_forced = true;
    hint.segments.push_back(std::move(opening));

    if (!param_names.empty()) {
        // Gap for first parameter value
        HintSegment gap1;
        gap1.is_forced = false;
        hint.segments.push_back(std::move(gap1));

        // Subsequent parameters: closing + next param name
        for (size_t i = 1; i < param_names.size(); i++) {
            std::string inter = "\n</parameter>\n<parameter=" + param_names[i] + ">\n";
            HintSegment inter_seg;
            inter_seg.tokens = tokenize(inter);
            inter_seg.is_forced = true;
            hint.segments.push_back(std::move(inter_seg));

            // Gap for this parameter's value
            HintSegment gap;
            gap.is_forced = false;
            hint.segments.push_back(std::move(gap));
        }

        // Closing suffix after last parameter
        std::string suffix = "\n</parameter>\n</function>\n</tool_call>";
        HintSegment suffix_seg;
        suffix_seg.tokens = tokenize(suffix);
        suffix_seg.is_forced = true;
        hint.segments.push_back(std::move(suffix_seg));
    } else {
        // No parameters — close immediately
        std::string suffix = "</function>\n</tool_call>";
        HintSegment suffix_seg;
        suffix_seg.tokens = tokenize(suffix);
        suffix_seg.is_forced = true;
        hint.segments.push_back(std::move(suffix_seg));
    }

    return hint;
}

ToolCallHint ToolHintGenerator::build_hint(const json & tools,
                                           const json & tool_choice) const {
    ToolCallHint empty;

    // No tools → no hint
    if (!tools.is_array() || tools.empty()) return empty;

    // Parse tool_choice
    std::string choice_str;
    std::string forced_name;

    if (tool_choice.is_string()) {
        choice_str = tool_choice.get<std::string>();
    } else if (tool_choice.is_object()) {
        // {"type": "function", "function": {"name": "X"}}
        if (tool_choice.contains("function") && tool_choice["function"].is_object()) {
            forced_name = tool_choice["function"].value("name", "");
        }
    }

    // "auto" or "none" → no hint (model may not call a tool)
    if (choice_str == "auto" || choice_str == "none") return empty;

    // Find the target function definition
    const json * target_func = nullptr;

    if (!forced_name.empty()) {
        // Specific function forced
        for (const auto & tool : tools) {
            if (!tool.contains("function")) continue;
            if (tool["function"].value("name", "") == forced_name) {
                target_func = &tool["function"];
                break;
            }
        }
    } else if (choice_str == "required") {
        if (tools.size() == 1 && tools[0].contains("function")) {
            // Single tool + required → we know exactly which function
            target_func = &tools[0]["function"];
        } else {
            // Multiple tools + required → hint only the structural prefix
            // after <tool_call> (already the anchor): \n<function=
            std::string prefix = "\n<function=";
            ToolCallHint partial;
            partial.prefix_tokens = tokenize(prefix);
            HintSegment seg;
            seg.tokens = partial.prefix_tokens;
            seg.is_forced = true;
            partial.segments.push_back(std::move(seg));
            return partial;
        }
    }

    if (!target_func) return empty;

    return build_function_hint(*target_func);
}

}  // namespace dflash::common
