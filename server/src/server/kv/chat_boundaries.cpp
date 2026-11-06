#include "server/kv/chat_boundaries.h"

#include <utility>

namespace luce::common {

// ─── Chat marker resolution ────────────────────────────────────────────

bool resolve_chat_markers(const Tokenizer & tok, ChatMarkers & out) {
    out.role_starts_delimit = false;
    // DeepSeek V4 uses full-width punctuation in its control tokens. Require
    // each marker to encode as its exact vocabulary token so an unrelated BPE
    // tokenizer cannot be misclassified merely because it can spell the text.
    const auto exact_control_token = [&tok](const char * marker) -> int32_t {
        const int32_t id = tok.token_to_id(marker);
        if (id < 0) return -1;
        const auto encoded = tok.encode(marker);
        return encoded.size() == 1 && encoded[0] == id ? id : -1;
    };
    const int32_t ds_bos = exact_control_token("<｜begin▁of▁sentence｜>");
    const int32_t ds_eos = exact_control_token("<｜end▁of▁sentence｜>");
    const int32_t ds_user = exact_control_token("<｜User｜>");
    const int32_t ds_assistant = exact_control_token("<｜Assistant｜>");
    if (ds_bos >= 0 && ds_eos >= 0 && ds_user >= 0 && ds_assistant >= 0) {
        out.family = "deepseek";
        out.sys_role_prefix = {ds_bos};
        out.end_msg_seqs = {{ds_eos}};
        out.next_role_starts = {{ds_user}, {ds_assistant}};
        out.role_starts_delimit = true;
        return true;
    }

    // Try Qwen family: <|im_end|> and <|im_start|> should be single tokens.
    auto im_end = tok.encode("<|im_end|>");
    auto im_start = tok.encode("<|im_start|>");
    if (im_end.size() == 1 && im_start.size() == 1) {
        auto sys = tok.encode("system");
        out.family = "qwen";
        out.sys_role_prefix = {im_start[0]};
        if (sys.size() == 1) out.sys_role_prefix.push_back(sys[0]);
        out.end_msg_seqs = {{im_end[0]}};
        out.next_role_starts = {{im_start[0]}};
        return true;
    }

    // Try Gemma family: <|turn> (start) and <turn|> (end) are single tokens.
    auto turn_start = tok.encode("<|turn>");
    auto turn_end   = tok.encode("<turn|>");
    if (turn_start.size() == 1 && turn_end.size() == 1) {
        out.family = "gemma";
        out.sys_role_prefix = {turn_start[0]};
        out.end_msg_seqs = {{turn_end[0]}};
        out.next_role_starts = {{turn_start[0]}};
        return true;
    }

    // Try Laguna family: XML-style markers.
    auto start_sys = tok.encode("<system>");
    auto end_sys   = tok.encode("</system>");
    auto start_usr = tok.encode("<user>");
    auto end_usr   = tok.encode("</user>");
    auto start_ast = tok.encode("<assistant>");
    auto end_ast   = tok.encode("</assistant>");
    if (!start_sys.empty() && !end_sys.empty() && !start_usr.empty() &&
        !end_usr.empty() && !start_ast.empty() && !end_ast.empty()) {
        out.family = "laguna";
        out.sys_role_prefix = start_sys;
        out.end_msg_seqs = {end_sys, end_usr, end_ast};
        out.next_role_starts = {start_usr, start_ast, start_sys};
        return true;
    }

    return false;
}

// ─── Boundary detection ─────────────────────────────────────────────────

static bool seq_at(const std::vector<int32_t> & ids, int idx,
                   const std::vector<int32_t> & seq) {
    if (idx < 0 || idx + (int)seq.size() > (int)ids.size()) return false;
    for (int k = 0; k < (int)seq.size(); k++) {
        if (ids[idx + k] != seq[k]) return false;
    }
    return true;
}

static int find_first_seq(const std::vector<int32_t> & ids,
                          const std::vector<int32_t> & seq, int start = 0) {
    if (seq.empty()) return -1;
    int n = (int)ids.size(), m = (int)seq.size();
    for (int i = start; i + m <= n; i++) {
        if (ids[i] == seq[0] && seq_at(ids, i, seq)) return i;
    }
    return -1;
}

static std::pair<int, int> find_first_seq_any(
        const std::vector<int32_t> & ids,
        const std::vector<std::vector<int32_t>> & seqs, int start = 0) {
    int best = -1, best_len = 0;
    for (const auto & s : seqs) {
        int idx = find_first_seq(ids, s, start);
        if (idx >= 0 && (best < 0 || idx < best)) {
            best = idx;
            best_len = (int)s.size();
        }
    }
    return {best, best_len};
}

std::vector<int> find_all_boundaries(const std::vector<int32_t> & ids,
                                     const ChatMarkers & markers) {
    std::vector<int> out;
    // Boundaries normally start after the system role prefix. A prompt with
    // no system message (common for API clients) still has role boundaries,
    // so start from the first role marker instead of caching nothing. Take
    // whichever comes first: a system prefix quoted inside a later message
    // (literal ChatML in content) must not displace the real leading role.
    const auto first_start = find_first_seq_any(ids, markers.next_role_starts);
    int start_idx = find_first_seq(ids, markers.sys_role_prefix);
    int start_len = (int)markers.sys_role_prefix.size();
    if (first_start.first >= 0 &&
        (start_idx < 0 || first_start.first < start_idx)) {
        start_idx = first_start.first;
        start_len = first_start.second;
    }
    if (start_idx < 0 || start_len <= 0) return out;

    int cursor = start_idx + start_len;
    if (markers.role_starts_delimit) {
        // Each role marker opens a message and closes the previous one, so
        // the cut after every marker is a reusable prefix: the system text
        // before the first user marker, each completed turn, and the
        // generation prompt itself. A marker quoted inside message content
        // adds a candidate cut too; cuts are reuse hints and exact token
        // matching decides every restore, so that costs at most a snapshot
        // taken at a less useful place.
        while (true) {
            const auto [idx, len] =
                find_first_seq_any(ids, markers.next_role_starts, cursor);
            if (idx < 0) break;
            cursor = idx + len;
            out.push_back(cursor);
        }
        return out;
    }
    int stray_skips = 0;
    while (true) {
        auto [end_idx, end_len] = find_first_seq_any(ids, markers.end_msg_seqs, cursor);
        if (end_idx < 0) break;
        int after_end = end_idx + end_len;

        int next_match = -1, next_len = 0;
        for (int skip = 0; skip < 5; skip++) {
            int probe = after_end + skip;
            for (const auto & s : markers.next_role_starts) {
                if (seq_at(ids, probe, s)) {
                    next_match = probe;
                    next_len = (int)s.size();
                    goto found;
                }
            }
        }
        found:
        if (next_match < 0) {
            // Stray end-of-message marker with no following role start — chatml
            // tokens embedded in message content (file dumps, terminal output,
            // model-echoed markers). Skip this marker and keep scanning instead
            // of truncating the whole boundary list. A lone stray previously cut
            // the walk off here, hiding every real boundary after it; that pinned
            // the inline-snapshot deepen target (second-to-last boundary) at the
            // already-restored prefix length, so no snapshot was ever deepened and
            // every turn re-prefilled the entire tail. Guard against pathological
            // input (or a marker-family mismatch) by capping consecutive strays.
            if (++stray_skips > 8192) break;
            cursor = after_end;
            continue;
        }
        int boundary = next_match + next_len;
        out.push_back(boundary);
        cursor = boundary;
        stray_skips = 0;
    }
    return out;
}

}  // namespace luce::common
