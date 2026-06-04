// Scenario store — loads JSON scenario files and matches them against
// rendered prompts for the StubModelBackend.
//
// A scenario file is a JSON object describing one (prompt → token stream)
// pairing:
//
//   {
//     "name":        "qwen3_enable_thinking_basic",
//     "description": "Qwen3.6 enable_thinking emits reasoning before </think>",
//     "match": {
//       "prompt_suffix": "What is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"
//     },
//     "response": {
//       "finish_reason": "stop",
//       "decode_us":     0,
//       "tokens": [
//         "Let me think. ",
//         "2",
//         "+",
//         "2",
//         " = 4.",
//         { "text": "</think>", "special": true },
//         "\n\nThe answer is 4."
//       ]
//     }
//   }
//
// Match semantics: longest matching `prompt_suffix` wins. File load order
// breaks ties (with a stderr warning).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common::test {

struct ScenarioToken {
    std::string text;
    // When true, the stub backend emits this as a single special token
    // looked up via Tokenizer::token_to_id(). When false, the text is
    // BPE-encoded and emitted as the resulting sequence of IDs.
    bool        special = false;
};

struct ScenarioResponse {
    bool                       ok            = true;
    std::string                error;          // only used when ok=false
    std::string                finish_reason = "stop";  // "stop", "length", "tool_calls"
    int                        decode_us     = 0;  // optional inter-token delay
    std::vector<ScenarioToken> tokens;
};

struct Scenario {
    std::string      name;
    std::string      description;
    std::string      match_prompt_suffix;
    ScenarioResponse response;
};

class ScenarioStore {
public:
    // Load every *.json file under `dir`. Returns false if directory does
    // not exist or any file fails to parse (errors logged to stderr).
    bool load_directory(const std::string & dir);

    // Load a single scenario file. Returns false on parse error.
    bool load_file(const std::string & path);

    // Find the scenario whose `match_prompt_suffix` is the longest suffix
    // of `rendered_prompt`. Returns nullptr if none match.
    const Scenario * match(const std::string & rendered_prompt) const;

    std::size_t size() const { return scenarios_.size(); }

    // For diagnostic logging.
    const std::vector<Scenario> & scenarios() const { return scenarios_; }

private:
    std::vector<Scenario> scenarios_;
};

}  // namespace dflash::common::test
