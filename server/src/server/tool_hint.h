// Tool call hint generator — pre-tokenizes predictable tool call structure
// for injection as speculative decode draft tokens.
//
// When tool_choice constrains the output to a known function, the structural
// tokens (XML tags, function name, parameter names) are predictable with
// ~100% confidence. These tokens can be pre-computed and fed into the spec
// decode loop as draft overrides, achieving N tokens/verify-pass instead of 1.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace dflash::common {

using json = nlohmann::json;

class Tokenizer;  // forward decl

// A segment of hint tokens: either forced (predictable) or a gap (model samples).
struct HintSegment {
    std::vector<int32_t> tokens;
    bool is_forced;  // true = hint tokens, false = gap placeholder (tokens empty)
};

// Complete hint for a single tool call.
struct ToolCallHint {
    // Flat vector of all forced prefix tokens (up to the first gap).
    // This is the primary hint used by the spec decode loop.
    std::vector<int32_t> prefix_tokens;

    // Full segmented structure (for advanced gap-aware hinting).
    std::vector<HintSegment> segments;

    // Function name that was hinted (for diagnostics).
    std::string function_name;

    bool empty() const { return prefix_tokens.empty(); }
};

// Runtime state machine for gap-aware hinting during generation.
// Tracks which segment we're in and provides hint tokens dynamically
// as the model generates through forced/gap/forced/gap patterns.
class HintStateMachine {
public:
    enum class Phase {
        FORCED,   // Currently in a forced segment (hints active)
        GAP,      // Currently in a gap (model sampling, no hints)
        DONE      // All segments exhausted
    };

    HintStateMachine() = default;
    explicit HintStateMachine(ToolCallHint hint);

    // Get up to max_tokens of hint tokens for the current generation position.
    // Returns empty if in a GAP or DONE phase.
    std::vector<int32_t> get_hint_batch(int max_tokens) const;

    // Advance the state machine by n_accepted tokens (called after spec decode
    // acceptance). For forced segments, advance through the hint. For gaps,
    // just tracks how many tokens the model generated.
    void advance(int n_tokens);

    // Notify that the gap has ended (detected by seeing the close-parameter
    // pattern in generated text). Transitions from GAP to next FORCED segment.
    void end_gap();

    Phase phase() const { return phase_; }
    bool  active() const { return phase_ == Phase::FORCED; }
    bool  done() const { return phase_ == Phase::DONE; }

    // The token pattern that signals end of a gap (e.g., "\n</parameter>").
    // Empty if not in a gap or no more segments.
    const std::vector<int32_t> & gap_end_pattern() const { return gap_end_tokens_; }

private:
    void advance_to_next_segment();

    ToolCallHint hint_;
    Phase phase_ = Phase::DONE;
    int segment_idx_ = 0;       // current segment in hint_.segments
    int offset_in_segment_ = 0; // position within current forced segment

    // Pre-tokenized pattern for detecting end of gap.
    std::vector<int32_t> gap_end_tokens_;
};

// Generates tool call hint tokens from tool definitions and tool_choice.
class ToolHintGenerator {
public:
    explicit ToolHintGenerator(const Tokenizer & tok) : tok_(tok) {}

    // Build a hint based on tool definitions and tool_choice.
    //
    // tool_choice values:
    //   "required" + single tool  → full prefix hint (tags + name + first param)
    //   "required" + multiple     → structural prefix only (tags, no name)
    //   {"type":"function","function":{"name":"X"}} → full prefix hint for X
    //   "auto" or "none"          → empty (no hint)
    //
    // Returns empty hint if no prediction is possible.
    ToolCallHint build_hint(const json & tools,
                            const json & tool_choice) const;

private:
    // Tokenize a text fragment using the server tokenizer.
    std::vector<int32_t> tokenize(const std::string & text) const;

    // Build the full segmented hint for a specific function.
    ToolCallHint build_function_hint(const json & function_def) const;

    const Tokenizer & tok_;
};

}  // namespace dflash::common
