// Chat-template boundaries in a rendered prompt: where each message ends and
// the next role starts. The prefix cache keys checkpoints at these cuts, and
// the restore points prefill is cut at come from them (kv/cuts.h: spaced on
// some engines; a prompt's own last boundary is dropped where prefill does
// not cut at it).

#pragma once

#include "server/tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

struct ChatMarkers {
    std::string family;  // "deepseek", "qwen", "gemma", or "laguna"
    // Token sequences for boundary detection
    std::vector<int32_t> sys_role_prefix;
    std::vector<std::vector<int32_t>> end_msg_seqs;
    std::vector<std::vector<int32_t>> next_role_starts;
    // The template closes only some turns with an end marker (DeepSeek: the
    // system text and user turns end where the next role starts). Every role
    // marker is then a boundary on its own.
    bool role_starts_delimit = false;
};

// Resolve chat markers from the tokenizer (detects DeepSeek, Qwen, Gemma, or Laguna family).
bool resolve_chat_markers(const Tokenizer & tok, ChatMarkers & out);

// Find all turn-boundary cut points in a token stream.
std::vector<int> find_all_boundaries(const std::vector<int32_t> & ids,
                                     const ChatMarkers & markers);


}  // namespace luce::common
