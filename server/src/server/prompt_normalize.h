// Prompt normalization — volatile-header stripping for stable cache keys.
//
// Pure functions: no IO, no globals, no CUDA deps. Tested standalone.

#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace dflash::common {

using json = nlohmann::json;

// Normalize the effective system/messages content for cache-key hashing.
//
// Accepts either:
//   - Anthropic-format: the `system` field from a /v1/messages body
//     (string or array-of-content-blocks)
//   - OpenAI-format: the full `messages` array from a /v1/chat/completions
//     body (the function inspects messages[0] when role=="system")
//
// Returns the normalized text string that represents the system content
// for the purposes of cache-key construction. Volatile claude-code headers
// (blocks or lines starting with "x-anthropic-billing-header:") are REMOVED
// so that two requests differing only in the header value hash identically.
std::string normalize_system_for_cache(const json & system_or_messages);

}  // namespace dflash::common
