// Pin-Friendly Prompt Processor (PPP)
//
// Diffs the tools/system head against recent traffic, isolates the volatile
// span, and rewrites tokens to:
//   [shared prefix][shared suffix][volatile middle][end markers]
// so PrefixCache can pin the contiguous stable blob. See docs/PIN_FRIENDLY_PROMPT.md.

#pragma once

#include "chat_template.h"
#include "server/kv/prefix_cache.h"  // ChatMarkers

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

struct PinFriendlyLayout {
    std::vector<ChatMessage> messages;
    int  pin_end_token = 0;
    bool rearranged = false;
    int  lcp_len = 0;
};

// Result of a prefix/suffix diff on one sequence against a reference.
struct TokenDiffSplit {
    int prefix_len = 0;   // shared head
    int suffix_len = 0;   // shared tail (non-overlapping with prefix)
    int middle_begin = 0; // in `current`
    int middle_end = 0;   // exclusive in `current`
};

// Full-prompt rewrite for pin-friendly layout.
struct PinFriendlyRewrite {
    std::vector<int32_t> tokens;
    int  pin_end = 0;
    bool rewritten = false;
    int  prefix_len = 0;
    int  suffix_len = 0;
    int  middle_len = 0;
};

class PinFriendlyPrompt {
public:
    static int longest_common_prefix_len(const std::vector<int32_t> & a,
                                         const std::vector<int32_t> & b);

    static int longest_common_suffix_len(const std::vector<int32_t> & a,
                                         const std::vector<int32_t> & b,
                                         int prefix_len);

    // Split `current` vs `reference` into shared prefix / volatile middle /
    // shared suffix (classic single-hunk diff).
    static TokenDiffSplit diff_split(const std::vector<int32_t> & reference,
                                     const std::vector<int32_t> & current);

    // Max chat boundary ≤ n (0 if none).
    static int safe_boundary_cut(int n, const std::vector<int> & boundaries);

    static int choose_pin_end(int lcp,
                              const std::vector<int> & boundaries,
                              int min_pin_tokens);

    // Legacy LCP-only annotate (no rewrite).
    static int annotate_pin_end(
        const std::vector<int32_t> & tokens,
        const std::vector<int> & boundaries,
        const std::vector<std::vector<int32_t>> & recent_tool_prefixes,
        int window,
        int min_pin_tokens);

    // Exclusive end of the first tools/system message (through end-of-message
    // marker). Chat boundaries from find_all_boundaries() sit *after* the next
    // role-start and must not be used as the DiffPin rewrite head.
    static int tools_system_head_end(const std::vector<int32_t> & tokens,
                                     const ChatMarkers & markers);

    // Diff the tools/system head against recent prefixes and rewrite:
    //   [tokentokentoken] with a mid clock → [tokentoken][token][time][im_end]
    // pin_end covers the contiguous stable blob. No-op when boundaries are
    // empty (custom templates), history is empty, the volatile hunk is too
    // large, or nothing moves.
    static PinFriendlyRewrite diff_make_pin_friendly(
        const std::vector<int32_t> & tokens,
        const std::vector<int> & boundaries,
        const std::vector<std::vector<int32_t>> & recent_tool_prefixes,
        const ChatMarkers & markers,
        int window,
        int min_pin_tokens,
        int max_ephemeral_tokens);

    static std::pair<std::string, std::string>
    split_ephemeral_system_tail(const std::string & system);

    static PinFriendlyLayout rearrange(const std::vector<ChatMessage> & messages,
                                       bool enable);

    static void remember_tool_prefix(
        std::vector<std::vector<int32_t>> & ring,
        const std::vector<int32_t> & prompt_ids,
        int max_prefix_tokens,
        int window);

    // Peel trailing chat end-message marker tokens from `ids`.
    static int trailing_end_marker_len(const std::vector<int32_t> & ids,
                                       const ChatMarkers & markers);
};

}  // namespace luce::common
