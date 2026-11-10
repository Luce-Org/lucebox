// KVFlash recall mode 5: PFlash segment densities as chunk scores (pure, no ggml, no GPU).

#pragma once

#include "pflash_types.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace luce::common {

// The chunk score of each chunk_tokens-sized chunk of an n_tokens prompt: the largest density of the
// segments overlapping it, 0 where none does. kvflash_recall_pick then sums
// the chunk scores of each unit and keeps the tail, window and pins out of the candidates.
inline std::vector<float> kvflash_pflash_chunk_scores(const std::vector<PFlashTokenSpan> & segments,
                                                      const std::vector<double> & densities,
                                                      int n_tokens, int chunk_tokens) {
    if (n_tokens <= 0 || chunk_tokens <= 0) return {};
    std::vector<float> out((size_t)((n_tokens + chunk_tokens - 1) / chunk_tokens), 0.0f);
    for (size_t i = 0; i < segments.size() && i < densities.size(); i++) {
        const int b = std::max(0, segments[i].begin), e = std::min(n_tokens, segments[i].end);
        if (e <= b) continue;
        for (int c = b / chunk_tokens; c <= (e - 1) / chunk_tokens; c++)
            out[(size_t)c] = std::max(out[(size_t)c], (float)densities[i]);
    }
    return out;
}

// Chat-structure token ids of the target (-1 = unknown).
struct KvFlashChatIds {
    int32_t im_start = -1, user = -1, newline = -1, im_end = -1, tool_response = -1, assistant = -1, system = -1;
};

// The drafter shares the target's tokenizer, so the prompt goes to it unchanged. False + `err` when an id
// is outside the drafter's vocabulary (no folding).
inline bool kvflash_pflash2_ids_fit(const std::vector<int32_t> & ids, int64_t n_vocab, std::string & err) {
    for (int32_t t : ids) {
        if (t < 0 || t >= n_vocab) {
            err = "prompt id " + std::to_string(t) + " is outside the drafter vocabulary (" + std::to_string(n_vocab) +
                  "); mode 5 needs the target and drafter to share a tokenizer";
            return false;
        }
    }
    return true;
}

// Every message of the rendered prompt, in order, as the request path wants it (PFlashRequestMessages.all_*):
// role plus content token span ({0,0} = no tokens), the last two human (role "user") spans latest first. Read from
// the chat structure `<|im_start|> role \n content <|im_end|>`; a user turn opening with `<tool_response>` is role
// "tool". A turn without `<|im_end|>` (the generation prompt) is no message.
// ponytail: a tool span runs to the turn's `<|im_end|>`, so it includes the closing `</tool_response>` tokens.
inline PFlashRequestMessages kvflash_message_map(const std::vector<int32_t> & ids, const KvFlashChatIds & c) {
    PFlashRequestMessages m;
    const int n = (int)ids.size();
    for (int i = 0; i + 1 < n; i++) {
        if (ids[(size_t)i] != c.im_start) continue;
        const int32_t r = ids[(size_t)i + 1];
        std::string role = r == c.system ? "system" : r == c.assistant ? "assistant" : r == c.user ? "user" : "";
        if (role.empty()) { m.all_error = "unknown role token " + std::to_string(r) + " at " + std::to_string(i + 1); break; }
        int lo = i + 2;
        if (lo < n && ids[(size_t)lo] == c.newline) lo++;
        if (role == "user" && c.tool_response >= 0 && lo < n && ids[(size_t)lo] == c.tool_response) {
            role = "tool";
            if (++lo < n && ids[(size_t)lo] == c.newline) lo++;
        }
        // The turn's `<|im_end|>` is followed by "\n" and the next turn header (or the end): a literal marker in the
        // content (tool outputs hold them) is not a boundary.
        const auto is_header = [&](int j) {
            return j + 1 < n && ids[(size_t)j] == c.im_start &&
                   (ids[(size_t)j + 1] == c.system || ids[(size_t)j + 1] == c.assistant || ids[(size_t)j + 1] == c.user);
        };
        int hi = lo;
        for (; hi < n; hi++) {
            if (ids[(size_t)hi] != c.im_end) continue;
            const int j = hi + 1 < n && ids[(size_t)hi + 1] == c.newline ? hi + 2 : hi + 1;
            if (j >= n || is_header(j)) break;
        }
        if (hi >= n) break;   // the generation prompt
        m.all_roles.push_back(role);
        m.all_spans.push_back(hi > lo ? PFlashTokenSpan{lo, hi} : PFlashTokenSpan{0, 0});
        i = hi;
    }
    if (m.all_error.empty() && m.all_roles.empty()) m.all_error = "no chat messages in the prompt";
    if (!m.all_error.empty()) { m.all_roles.clear(); m.all_spans.clear(); return m; }
    for (size_t i = m.all_roles.size(); i-- > 0 && m.human.size() < 2;) {
        if (m.all_roles[i] != "user") continue;
        m.human.push_back(m.all_spans[i]);
        m.human_index.push_back((int)i);
    }
    return m;
}

// The query end of the request: the end of the last message's content; 0 when there is none (no messages, or an
// empty final message, whose span is {0,0}) so the caller refuses instead of querying an earlier message.
inline int kvflash_message_query_end(const PFlashRequestMessages & m) {
    return m.all_spans.empty() ? 0 : m.all_spans.back().end;
}

// Mode 5 runs the PFlash release profile; "" when the environment holds it, else the first
// setting that is missing or wrong, named. Drafter sessions (default 2) must cover the slots.
inline std::string kvflash_pflash2_config_error(const std::function<std::string(const char *)> & get, int slots) {
    static const char * const kExact[][2] = {   // value "" = any non-empty
        {"PFLASH_SELECT_MODE", "budget_only"}, {"PFLASH_SELECT_SCORE", "density"}, {"PFLASH_SELECT_SEGMENTS", "probe"},
        {"PFLASH_SELECT_QUERY_HEAD", "1"}, {"PFLASH_SELECT_QUERY_PROBE", "1"},
        {"PFLASH_SCORING_HEAD_GGUF", ""}, {"PFLASH_SEGMENT_PROBE_GGUF", ""}, {"PFLASH_REQUEST_PROBE_GGUF", ""},
        {"PFLASH_QUERY_CANDIDATES", "1"}, {"PFLASH_QUERY_PIN", ""}};
    for (const auto & kv : kExact) {
        const std::string v = get(kv[0]);
        if (v.empty() || (*kv[1] && v != kv[1]))
            return std::string("mode 5 needs ") + kv[0] + (*kv[1] ? std::string("=") + kv[1] : std::string(" set")) +
                   " (got " + (v.empty() ? "unset" : v) + ")";
    }
    const std::string s = get("PFLASH_DRAFTER_SESSIONS");
    int sessions = 2;
    const auto parsed = std::from_chars(s.data(), s.data() + s.size(), sessions);
    if (!s.empty() && (parsed.ec != std::errc() || parsed.ptr != s.data() + s.size()))
        return "mode 5 needs PFLASH_DRAFTER_SESSIONS to be a whole number >= " + std::to_string(slots) + " (the slot count), got " + s;
    if (sessions < slots)
        return "mode 5 needs PFLASH_DRAFTER_SESSIONS >= " + std::to_string(slots) + " (the slot count), got " + std::to_string(sessions);
    return "";
}

}  // namespace luce::common
