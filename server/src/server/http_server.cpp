// HTTP server implementation.
//
// Core infrastructure: socket listen/accept, client threads, HTTP parsing,
// job queue, worker thread with SSE streaming and disconnect detection.

// On Windows, winsock2.h must be included BEFORE windows.h (which comes
// transitively via internal.h → http_server.h). Classic MSVC ordering.
#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "http_server.h"
#include "image_input.h"
#include "engine/luce_engine.h"
#include "admission.h"
#include "luce.h"
#include "common/concurrency/seq_engine.h"
#include "common/model_capabilities.h"
#include "response_error.h"
#include "sse_emitter.h"
#include "prompt_normalize.h"
#include "tool_hint.h"
#include "tool_parser.h"
#include "json_schema_grammar.h"
#include "pin_friendly_prompt.h"
#include "common/kv_rotation.h"
#include "common/sha1.h"
#include "pflash/pflash_fold.h"
#include "pflash/pflash_selection.h"
#include "pflash/pflash_request_probe.h"
#include "freeze_history.h"

#ifdef LUCE_HAS_CURL
#include <curl/curl.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

using luce::common::SocketHandle;

#if defined(_WIN32)
#include <io.h>
#include <sys/stat.h>
typedef long ssize_t;
#define MSG_NOSIGNAL   0
#define MSG_DONTWAIT   0
#define SHUT_RDWR      SD_BOTH
#define socklen_t      int
#define poll(fds,nfds,timeout)  WSAPoll(fds,nfds,timeout)
// Replace fcntl(F_GETFL) / fcntl(F_SETFL, O_NONBLOCK) with ioctlsocket
static inline int sock_get_flags(SocketHandle fd) { (void)fd; return 0; /* stub */ }
static inline void sock_set_nonblock(SocketHandle fd) { u_long m = 1; ioctlsocket(fd, FIONBIO, &m); }
static inline void sock_set_block(SocketHandle fd) { u_long m = 0; ioctlsocket(fd, FIONBIO, &m); }
static inline void socket_close(SocketHandle fd) { closesocket(fd); }
#define SETSOCKOPT_CAST (const char *)
static inline const char* sock_strerror() {
    static thread_local char buf[64];
    // On Windows, use FormatMessage for WSA errors
    snprintf(buf, sizeof(buf), "WSA error %d", WSAGetLastError());
    return buf;
}
static inline int  sock_errno()          { return WSAGetLastError(); }
static inline bool sock_is_eintr (int e) { return e == WSAEINTR; }
static inline bool sock_is_eagain(int e) { return e == WSAEWOULDBLOCK; }
#else
#include <fcntl.h>
#include <sys/stat.h>
static inline int sock_get_flags(SocketHandle fd) { return fcntl(fd, F_GETFL, 0); }
static inline void sock_set_nonblock(SocketHandle fd) { int f = fcntl(fd, F_GETFL, 0); if (f >= 0) fcntl(fd, F_SETFL, f | O_NONBLOCK); }
static inline void sock_set_block(SocketHandle fd) { int f = fcntl(fd, F_GETFL, 0); if (f >= 0) fcntl(fd, F_SETFL, f & ~O_NONBLOCK); }
static inline void socket_close(SocketHandle fd) { ::close(fd); }
#define SETSOCKOPT_CAST  /* empty on POSIX */
#include <unistd.h>
static inline const char* sock_strerror() { return strerror(errno); }
static inline int  sock_errno()          { return errno; }
static inline bool sock_is_eintr (int e) { return e == EINTR; }
static inline bool sock_is_eagain(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace luce::common {

namespace {
constexpr auto kClientMonitorInterval = std::chrono::milliseconds(250);
constexpr auto kSseHeartbeatInterval = std::chrono::seconds(15);
constexpr auto kReadClosedProbeInterval = std::chrono::seconds(1);
constexpr char kSseHeartbeat[] = ": keep-alive\n\n";
}

namespace http_detail {

bool validate_pflash_environment(std::string & error) {
    luce::pflash::PFlashSelectionConfig selection;
    if (!luce::pflash::resolve_pflash_selection(1 << 16, 32, selection, error)) {
        return false;
    }
    for (const char * name : {
            "PFLASH_CHAT_VIEW", "PFLASH_CHAT_RECALL", "PFLASH_SELECT_PARAGRAPH_JOIN",
            "PFLASH_AGENT_PASTE", "PFLASH_ARRIVE", "PFLASH_FAIL_CLOSED",
            "PFLASH_DRAFTER_TRIM_POOL", "PFLASH_SESSION_OFFLOAD_PINNED",
            "PFLASH_QWEN35_LEGACY_SCORER", "PFLASH_SELECT_QUERY_PROBE_STUB"}) {
        const char * raw = std::getenv(name);
        if (raw && *raw && std::strcmp(raw, "0") != 0 && std::strcmp(raw, "1") != 0) {
            error = std::string(name) + " must be 0 or 1";
            return false;
        }
    }
    if (const char * raw = std::getenv("PFLASH_ADMISSION"); raw && *raw &&
            std::strcmp(raw, "length") != 0 && std::strcmp(raw, "cache") != 0) {
        error = "PFLASH_ADMISSION must be length or cache";
        return false;
    }
    const struct { const char * name; long minimum; } integers[] = {
        {"PFLASH_CHAT_SKELETON_TOKENS", 0},
        {"PFLASH_CHAT_COMPRESS_NEW_TOKENS", 1},
        {"PFLASH_CHAT_HISTORY_QUERIES", 0},
        {"PFLASH_CHAT_REBUILD_MIN_TOKENS", 0},
        {"PFLASH_DRAFTER_SESSIONS", 0},
    };
    for (const auto & flag : integers) {
        const char * raw = std::getenv(flag.name);
        if (!raw || !*raw) continue;
        errno = 0;
        char * end = nullptr;
        const long value = std::strtol(raw, &end, 10);
        if (errno == ERANGE || end == raw || *end != '\0' || value < flag.minimum) {
            error = std::string(flag.name) + " must be an integer >= " +
                std::to_string(flag.minimum);
            return false;
        }
    }
    if (const char * raw = std::getenv("PFLASH_CHAT_REBUILD_RATIO"); raw && *raw) {
        errno = 0;
        char * end = nullptr;
        const double value = std::strtod(raw, &end);
        if (errno == ERANGE || end == raw || *end != '\0' ||
            !std::isfinite(value) || value < 0.0) {
            error = "PFLASH_CHAT_REBUILD_RATIO must be a finite number >= 0";
            return false;
        }
    }
    const auto set = [](const char * name) {
        const char * raw = std::getenv(name);
        return raw && *raw;
    };
    const bool grid = set("PFLASH_QUERY_GRID");
    if (grid && (!selection.query_probe || !set("PFLASH_QUERY_GRID_OUT"))) {
        error = "PFLASH_QUERY_GRID needs PFLASH_SELECT_QUERY_PROBE=1 and PFLASH_QUERY_GRID_OUT";
        return false;
    }
    if ((selection.query.any() || grid) && !set("PFLASH_SEGMENT_PROBE_GGUF")) {
        error = "PFLASH_QUERY_CANDIDATES / PFLASH_QUERY_PIN / PFLASH_QUERY_GRID need PFLASH_SEGMENT_PROBE_GGUF";
        return false;
    }
    if ((selection.query.any() || grid) && set("PFLASH_REQUEST_STAGE2_GGUF")) {
        error = "PFLASH_QUERY_CANDIDATES / PFLASH_QUERY_PIN / PFLASH_QUERY_GRID do not combine with PFLASH_REQUEST_STAGE2_GGUF";
        return false;
    }
    return true;
}

PeerSocketState inspect_peer_socket(SocketHandle fd) {
    struct pollfd pfd = {fd, POLLIN, 0};
#if defined(POLLRDHUP)
    pfd.events |= POLLRDHUP;
#endif

    int ret;
    do {
        ret = poll(&pfd, 1, 0);
    } while (ret < 0 && sock_is_eintr(sock_errno()));
    if (ret == 0) return PeerSocketState::Connected;
    if (ret < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
        return PeerSocketState::Disconnected;
    }
    bool read_closed = false;
#if defined(POLLRDHUP)
    read_closed = (pfd.revents & POLLRDHUP) != 0;
#endif
    if (!(pfd.revents & POLLIN)) {
        return read_closed ? PeerSocketState::ReadClosed
                           : PeerSocketState::Connected;
    }

    char byte = 0;
    const ssize_t n = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) return PeerSocketState::ReadClosed;
    if (n > 0) return PeerSocketState::Connected;
    const int error = sock_errno();
    return (sock_is_eintr(error) || sock_is_eagain(error))
        ? PeerSocketState::Connected
        : PeerSocketState::Disconnected;
}

bool sse_chunk_has_done(
        std::string & partial_line, const char * data, size_t size) {
    static constexpr char kDoneLine[] = "data: [DONE]";
    static constexpr size_t kDoneLineSize = sizeof(kDoneLine) - 1;

    bool found = false;
    for (size_t i = 0; i < size; ++i) {
        const char ch = data[i];
        if (ch == '\r' || ch == '\n') {
            found = found || partial_line == kDoneLine;
            partial_line.clear();
        } else if (partial_line.size() <= kDoneLineSize) {
            // One byte beyond the marker length is an overflow sentinel. It
            // prevents a non-terminal SSE line from growing without bound.
            partial_line.push_back(ch);
        }
    }
    return found;
}

HeartbeatSendResult try_send_sse_heartbeat(
        SocketHandle fd, size_t & offset) {
    constexpr size_t heartbeat_size = sizeof(kSseHeartbeat) - 1;
    if (offset > heartbeat_size) {
        offset = 0;
        return HeartbeatSendResult::Disconnected;
    }
    while (offset < heartbeat_size) {
        const ssize_t n = send(
            fd, kSseHeartbeat + offset, heartbeat_size - offset,
            MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n > 0) {
            offset += (size_t)n;
            continue;
        }
        if (n < 0) {
            const int error = sock_errno();
            if (sock_is_eintr(error)) continue;
            if (sock_is_eagain(error)) return HeartbeatSendResult::Retry;
        }
        offset = 0;
        return HeartbeatSendResult::Disconnected;
    }
    offset = 0;
    return HeartbeatSendResult::Complete;
}

PflashQueryWindow find_pflash_query_window(
        const std::vector<int32_t> & prompt,
        const std::vector<int32_t> & query,
        int max_tokens,
        int search_end,
        int search_begin,
        bool anchored) {
    PflashQueryWindow result;
    if (prompt.empty() || query.empty() || max_tokens < 1) return result;

    const int limit = search_end < 0
        ? (int) prompt.size()
        : (std::min)((int) prompt.size(), search_end);
    if (limit < 1 || search_begin < 0 || search_begin >= limit) return result;
    const int widest = (std::min)(max_tokens, (int) query.size());
    // For a short query, require all available tokens. For a normal query,
    // four matching suffix tokens are enough to tolerate a BPE boundary
    // difference without accidentally selecting a lone punctuation token.
    const int narrowest = (std::min)(4, widest);
    // A configured semantic boundary is the exact content end. Shorten only
    // the suffix width there; accepting an earlier occurrence would silently
    // turn a failed latest-user mapping into preceding context.
    if (search_end >= 0 && anchored) {
        const int bounded_widest = (std::min)(widest, limit - search_begin);
        for (int width = bounded_widest; width >= narrowest; --width) {
            const auto query_begin = query.end() - width;
            if (std::equal(query_begin, query.end(),
                           prompt.begin() + limit - width)) {
                result.end = limit;
                result.tokens = width;
                return result;
            }
        }
        return result;
    }
    // Unanchored: the latest occurrence of the (suffix-trimmed) query inside
    // [search_begin, limit). An explicit query may sit before trailing
    // instructions, so it is not required to end at the content boundary.
    // Its last token may also merge with what follows it in the prompt
    // (trailing whitespace before a newline), so up to two trailing query
    // tokens may be dropped; the untrimmed query is always preferred.
    const int floor = (std::max)(0, search_begin);
    const int max_trailing = (std::min)(2, (int) query.size() - narrowest);
    for (int drop = 0; drop <= max_trailing; ++drop) {
        const auto query_end = query.end() - drop;
        const int available = (int) query.size() - drop;
        const int drop_widest = (std::min)(max_tokens, available);
        for (int width = drop_widest; width >= narrowest; --width) {
            const auto query_begin = query_end - width;
            for (int end = limit; end - width >= floor; --end) {
                if (std::equal(query_begin, query_end,
                               prompt.begin() + end - width)) {
                    result.end = end;
                    result.tokens = width;
                    result.trailing_trimmed = drop;
                    return result;
                }
            }
        }
    }
    return result;
}

PflashQueryWindow pflash_tail_query_window(
        const std::vector<int32_t> & prompt,
        int max_tokens,
        int query_end,
        int query_begin) noexcept {
    PflashQueryWindow result;
    if (prompt.empty() || max_tokens < 1) return result;
    result.end = query_end < 0
        ? static_cast<int>(prompt.size())
        : query_end;
    if (result.end < 1 || result.end > static_cast<int>(prompt.size()) ||
        query_begin < 0 || query_begin >= result.end) {
        return {};
    }
    result.tokens = (std::min)(max_tokens, result.end - query_begin);
    return result;
}

PflashQueryHeadPlan pflash_query_head_plan(
        const Tokenizer & tokenizer,
        const std::vector<int32_t> & prompt,
        int content_begin,
        int content_end,
        int head_cap,
        int tail_cap) {
    PflashQueryHeadPlan plan;
    if (content_begin < 0 || content_end <= content_begin ||
        content_end > (int) prompt.size() || head_cap < 2 || tail_cap < 2) {
        return plan;
    }
    std::string text;
    std::vector<size_t> offsets;
    offsets.reserve((size_t) (content_end - content_begin));
    for (int index = content_begin; index < content_end; ++index) {
        offsets.push_back(text.size());
        text += tokenizer.token_text(prompt[(size_t) index]);
    }
    const auto blocks = luce::pflash::pflash_query_blocks(text, {0, text.size()});
    plan.structured = blocks.structured;
    // A block this short is a stock line ("Here is the material.", "Answer
    // with the letter.") rather than the question.
    constexpr int kMinBlock = 16;
    const auto tokens_of = [&](luce::pflash::PFlashTextSpan span) {
        PFlashTokenSpan t = luce::pflash::pflash_text_to_token_span(offsets, span);
        return PFlashTokenSpan{t.begin + content_begin, t.end + content_begin};
    };
    // A block over its cap contributes its first and last cap/2 tokens.
    const auto pieces = [](PFlashTokenSpan block, int cap) {
        std::vector<PFlashTokenSpan> out;
        if (block.end - block.begin <= cap) {
            out.push_back(block);
        } else {
            out.push_back({block.begin, block.begin + cap / 2});
            out.push_back({block.end - cap / 2, block.end});
        }
        return out;
    };
    if (blocks.head.end > blocks.head.begin) {
        const PFlashTokenSpan head = tokens_of(blocks.head);
        if (head.end - head.begin >= kMinBlock) {
            plan.head_rule = blocks.head_rule;
            plan.head_tokens = head.end - head.begin;
            for (const auto & piece : pieces(head, head_cap)) plan.extra.push_back(piece);
        }
    }
    if (blocks.tail.end > blocks.tail.begin) {
        const PFlashTokenSpan tail = tokens_of(blocks.tail);
        if (tail.end - tail.begin >= kMinBlock) {
            plan.tail_tokens = tail.end - tail.begin;
            const auto parts = pieces(tail, tail_cap);
            plan.tail = parts.back();
            if (parts.size() > 1) plan.extra.push_back(parts.front());
        }
    }
    return plan;
}

PFlashTokenSpan pflash_decoded_text_span(
        const Tokenizer & tokenizer,
        const std::vector<int32_t> & prompt,
        int begin,
        int end,
        const std::string & needle) {
    if (needle.empty() || begin < 0 || end <= begin ||
        end > (int) prompt.size()) {
        return {-1, -1};
    }
    std::string decoded;
    std::vector<size_t> offsets;
    offsets.reserve((size_t)(end - begin));
    for (int index = begin; index < end; ++index) {
        offsets.push_back(decoded.size());
        decoded += tokenizer.token_text(prompt[(size_t) index]);
    }
    const size_t pos = decoded.rfind(needle);
    if (pos == std::string::npos) {
        return {-1, -1};
    }
    const size_t needle_end = pos + needle.size();
    int first = begin;
    while (first + 1 < end && offsets[(size_t)(first + 1 - begin)] <= pos) {
        ++first;
    }
    int after = first;
    while (after < end && offsets[(size_t)(after - begin)] < needle_end) {
        ++after;
    }
    return {first, after};
}

int pflash_query_search_end_from_sentinel(
        const std::vector<int32_t> & original,
        const std::vector<int32_t> & sentinel) noexcept {
    size_t common_suffix = 0;
    while (common_suffix < original.size() &&
           common_suffix < sentinel.size() &&
           original[original.size() - common_suffix - 1] ==
               sentinel[sentinel.size() - common_suffix - 1]) {
        ++common_suffix;
    }
    if (common_suffix == 0 || common_suffix >= original.size()) {
        return -1;
    }
    return static_cast<int>(original.size() - common_suffix);
}
int pflash_query_search_begin_from_sentinel(
        const std::vector<int32_t> & original,
        const std::vector<int32_t> & sentinel) noexcept {
    size_t common_prefix = 0;
    while (common_prefix < original.size() &&
           common_prefix < sentinel.size() &&
           original[common_prefix] == sentinel[common_prefix]) {
        ++common_prefix;
    }
    if (common_prefix >= original.size()) return -1;
    return static_cast<int>(common_prefix);
}

PflashInstructionMessagePlan plan_pflash_instruction_messages(
        const std::vector<ChatMessage> & messages) {
    PflashInstructionMessagePlan plan;
    for (size_t index = 0; index < messages.size(); ++index) {
        const auto & message = messages[index];
        const bool instruction_role =
            message.role == "system" || message.role == "developer";
        if (instruction_role && !message.content.empty()) {
            plan.instruction_messages.push_back(index);
        }
    }
    return plan;
}

PFlashTokenSpan pflash_changed_token_span(
        const std::vector<int32_t> & original,
        const std::vector<int32_t> & variant) noexcept {
    size_t common_prefix = 0;
    while (common_prefix < original.size() &&
           common_prefix < variant.size() &&
           original[common_prefix] == variant[common_prefix]) {
        ++common_prefix;
    }

    size_t common_suffix = 0;
    while (common_suffix < original.size() - common_prefix &&
           common_suffix < variant.size() - common_prefix &&
           original[original.size() - common_suffix - 1] ==
               variant[variant.size() - common_suffix - 1]) {
        ++common_suffix;
    }

    const int begin = static_cast<int>(common_prefix);
    const int end = static_cast<int>(original.size() - common_suffix);
    return end > begin ? PFlashTokenSpan{begin, end}
                       : PFlashTokenSpan{-1, -1};
}

namespace {

// Decoded prompt text plus the character offset each token starts at.
// Tokenizer::decode concatenates per-token text, so the offsets are exact.
struct DecodedPrompt {
    std::string text;
    std::vector<size_t> token_begin;   // size == prompt.size()
};

DecodedPrompt decode_prompt_with_offsets(
        const Tokenizer & tokenizer,
        const std::vector<int32_t> & prompt) {
    DecodedPrompt out;
    out.token_begin.reserve(prompt.size());
    std::vector<int32_t> one(1, 0);
    for (int32_t id : prompt) {
        out.token_begin.push_back(out.text.size());
        one[0] = id;
        out.text += tokenizer.decode(one);
    }
    return out;
}

// The token containing character offset `at`, clamped into the prompt.
int token_at_offset(const DecodedPrompt & decoded, size_t at) {
    const auto upper = std::upper_bound(
        decoded.token_begin.begin(), decoded.token_begin.end(), at);
    if (upper == decoded.token_begin.begin()) return 0;
    return (int) (upper - decoded.token_begin.begin() - 1);
}

} // namespace

std::vector<PFlashTokenSpan> canonicalize_pflash_token_spans(
        std::vector<PFlashTokenSpan> spans) {
    std::sort(spans.begin(), spans.end(), [] (
            const PFlashTokenSpan & left,
            const PFlashTokenSpan & right) {
        return left.begin < right.begin ||
            (left.begin == right.begin && left.end < right.end);
    });
    std::vector<PFlashTokenSpan> result;
    for (const PFlashTokenSpan & span : spans) {
        if (!result.empty() && span.begin <= result.back().end) {
            result.back().end = std::max(result.back().end, span.end);
        } else {
            result.push_back(span);
        }
    }
    return result;
}

json pflash_view_pieces_json(const std::vector<PflashViewPiece> & pieces) {
    json out = json::array();
    for (const auto & piece : pieces) {
        if (piece.recall) {
            out.push_back(json::array({piece.begin, piece.end, "recall", piece.block}));
        } else {
            out.push_back(json::array({piece.begin, piece.end, "raw"}));
        }
    }
    return out;
}

std::string pflash_token_fingerprint(
        const std::vector<int32_t> & ids) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int32_t token : ids) {
        const uint32_t value = static_cast<uint32_t>(token);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>(value >> shift);
            hash *= UINT64_C(1099511628211);
        }
    }

    char encoded[17];
    std::snprintf(encoded, sizeof(encoded), "%016llx",
                  static_cast<unsigned long long>(hash));
    return encoded;
}

PflashChatTurnSpan pflash_chat_query_turn(
        const Tokenizer & marker_tokenizer,
        const ChatMarkers & markers,
        const Tokenizer & tokenizer,
        const std::vector<int32_t> & prompt) {
    PflashChatTurnSpan chosen;
    if (prompt.empty()) return chosen;

    // Marker strings, searched in the decoded prompt text so a drafter whose
    // vocabulary spells the control tokens differently still maps.
    struct Mark {
        size_t at = 0;
        size_t len = 0;
        bool role = false;
        std::string text;
    };
    const auto seq_text = [&marker_tokenizer](
            const std::vector<int32_t> & seq) {
        std::string text;
        for (const int32_t id : seq) text += marker_tokenizer.token_text(id);
        return text;
    };
    std::vector<std::pair<std::string, bool>> needles;
    for (const auto & seq : markers.next_role_starts) {
        std::string text = seq_text(seq);
        if (!text.empty()) needles.emplace_back(std::move(text), true);
    }
    for (const auto & seq : markers.end_msg_seqs) {
        std::string text = seq_text(seq);
        if (!text.empty()) needles.emplace_back(std::move(text), false);
    }
    if (needles.empty()) return chosen;

    const DecodedPrompt decoded =
        decode_prompt_with_offsets(tokenizer, prompt);
    const std::string & text = decoded.text;
    std::vector<Mark> marks;
    for (const auto & [needle, role] : needles) {
        for (size_t at = text.find(needle); at != std::string::npos;
             at = text.find(needle, at + needle.size())) {
            marks.push_back({at, needle.size(), role, needle});
        }
    }
    if (marks.empty()) return chosen;
    std::sort(marks.begin(), marks.end(),
              [] (const Mark & a, const Mark & b) { return a.at < b.at; });

    // Families whose markers name the role (DeepSeek "<｜User｜>", Laguna
    // "<user>") start content at the marker; generic markers (Qwen
    // "<|im_start|>", Gemma "<|turn>") are followed by a "name\n" line.
    const bool marker_carries_role =
        markers.role_starts_delimit || markers.family == "laguna";
    const auto role_from_marker = [] (const std::string & marker) {
        std::string role;
        for (const char c : marker) {
            if (std::isalpha((unsigned char) c)) {
                role += (char) std::tolower((unsigned char) c);
            }
        }
        return role;
    };
    const auto is_space = [] (char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };
    const auto starts_with = [&text] (size_t at, const char * prefix) {
        return text.compare(at, std::strlen(prefix), prefix) == 0;
    };

    struct Turn {
        size_t role_at = 0;
        size_t content_at = 0;
        size_t content_end = 0;
        size_t close_end = 0;   // past the closing marker, if any
        std::string role;
        bool closed = false;
    };
    std::vector<Turn> turns;
    for (size_t index = 0; index < marks.size(); ++index) {
        const Mark & mark = marks[index];
        if (!mark.role) continue;
        Turn turn;
        turn.role_at = mark.at;
        size_t content_at = mark.at + mark.len;
        if (marker_carries_role) {
            turn.role = role_from_marker(mark.text);
        } else {
            size_t name_end = content_at;
            while (name_end < text.size() && name_end - content_at < 16 &&
                   std::isalpha((unsigned char) text[name_end])) {
                ++name_end;
            }
            turn.role = text.substr(content_at, name_end - content_at);
            if (name_end < text.size() && text[name_end] == '\n') {
                content_at = name_end + 1;
            }
        }
        // A turn ends at its end marker, or -- when role markers delimit
        // (DeepSeek user turns) -- at the next role marker. Nothing after
        // it leaves the turn open to the prompt end.
        size_t content_end = text.size();
        turn.close_end = text.size();
        if (index + 1 < marks.size()) {
            const Mark & next = marks[index + 1];
            content_end = next.at;
            turn.closed = !next.role || markers.role_starts_delimit;
            turn.close_end = next.role ? next.at : next.at + next.len;
        }
        // Content ignores the whitespace the template wraps it in.
        while (content_at < content_end && is_space(text[content_at])) {
            ++content_at;
        }
        while (content_end > content_at && is_space(text[content_end - 1])) {
            --content_end;
        }
        turn.content_at = content_at;
        turn.content_end = content_end;
        // Tool output travels in user turns on some templates.
        if (starts_with(content_at, "<tool_response>") ||
            starts_with(content_at, "<tool_result>")) {
            turn.role = "tool";
        }
        turns.push_back(std::move(turn));
    }
    if (turns.empty()) return chosen;

    // An assistant turn left open at the prompt end is the generation prompt
    // ("<|im_start|>assistant\n<think>\n"): template machinery, never query.
    size_t usable = turns.size();
    const Turn & last = turns.back();
    if (!last.closed && (last.role == "assistant" || last.role == "model")) {
        --usable;
    }
    // The query comes from the latest user turn; a conversation without one
    // falls back to its latest turn with content.
    size_t query_index = turns.size();
    for (size_t index = usable; index-- > 0;) {
        const Turn & turn = turns[index];
        if (turn.content_end <= turn.content_at) continue;
        if (turn.role == "user") { query_index = index; break; }
        if (query_index == turns.size()) query_index = index;
    }
    if (query_index == turns.size()) return chosen;
    const Turn * query = &turns[query_index];

    // Content bounds in tokens: the first token starting at-or-after each
    // character offset, so a token merged across a boundary stays with the
    // content it ends.
    const auto token_from = [&decoded] (size_t at) {
        return (int) (std::lower_bound(
            decoded.token_begin.begin(), decoded.token_begin.end(), at) -
            decoded.token_begin.begin());
    };
    chosen.role_begin = token_at_offset(decoded, query->role_at);
    chosen.content_begin = token_from(query->content_at);
    chosen.content_end = token_from(query->content_end);
    chosen.turn_end = token_from(query->close_end);
    chosen.generation_begin = usable < turns.size()
        ? token_at_offset(decoded, turns.back().role_at)
        : (int) prompt.size();
    chosen.later_turns = query_index + 1 < usable;
    if (chosen.role_begin > chosen.content_begin) {
        chosen.role_begin = chosen.content_begin;
    }
    for (size_t index = 0; index < usable; ++index) {
        const Turn & turn = turns[index];
        PflashChatTurn out;
        out.role_begin = token_at_offset(decoded, turn.role_at);
        out.content_begin = token_from(turn.content_at);
        out.content_end = token_from(turn.content_end);
        out.turn_end = token_from(turn.close_end);
        out.role_begin = (std::min)(out.role_begin, out.content_begin);
        out.role = turn.role;
        chosen.turns.push_back(std::move(out));
    }
    chosen.query_turn = (int) query_index;
    return chosen;
}

bool pflash_content_char_spans(
        const std::string & placeholder_render,
        const std::vector<std::string> & placeholders,
        const std::vector<std::string> & contents,
        const std::string & prompt_text,
        std::vector<std::pair<size_t, size_t>> & spans,
        std::string & error) {
    spans.assign(placeholders.size(), {0, 0});
    if (contents.size() != placeholders.size()) {
        error = "placeholder and content counts differ";
        return false;
    }
    std::string rebuilt;
    rebuilt.reserve(prompt_text.size());
    size_t cursor = 0;
    for (size_t i = 0; i < placeholders.size(); ++i) {
        const std::string & ph = placeholders[i];
        if (ph.empty()) continue;
        const size_t at = placeholder_render.find(ph, cursor);
        if (at == std::string::npos ||
            placeholder_render.find(ph, at + ph.size()) != std::string::npos) {
            error = "message " + std::to_string(i) + " is not rendered exactly once, in order";
            return false;
        }
        rebuilt.append(placeholder_render, cursor, at - cursor);
        spans[i] = {rebuilt.size(), rebuilt.size() + contents[i].size()};
        rebuilt += contents[i];
        cursor = at + ph.size();
    }
    rebuilt.append(placeholder_render, cursor, std::string::npos);
    if (rebuilt != prompt_text) {
        error = "the template does not render message contents verbatim";
        spans.assign(placeholders.size(), {0, 0});
        return false;
    }
    return true;
}

PFlashTokenSpan pflash_char_span_tokens(
        const std::vector<size_t> & token_begin, size_t text_size, size_t begin, size_t end) {
    if (end <= begin || begin >= text_size || token_begin.empty()) return {0, 0};
    // First token whose extent reaches past ``begin`` (the one containing it), and the first token starting at or
    // after ``end``.
    const auto upper = std::upper_bound(token_begin.begin(), token_begin.end(), begin);
    const int first = upper == token_begin.begin() ? 0 : (int) (upper - token_begin.begin() - 1);
    const int last = (int) (std::lower_bound(token_begin.begin(), token_begin.end(), end) - token_begin.begin());
    return last > first ? PFlashTokenSpan{first, last} : PFlashTokenSpan{0, 0};
}

bool pflash_human_message_spans(
        const Tokenizer & tokenizer,
        const std::vector<int32_t> & prompt,
        const std::vector<PflashChatTurn> & turns,
        const std::vector<ChatMessage> & messages,
        size_t count,
        std::vector<PFlashTokenSpan> & out) {
    out.clear();
    const auto trim = [] (const std::string & text) {
        const char * space = " \t\n\r";
        const size_t begin = text.find_first_not_of(space);
        if (begin == std::string::npos) return std::string();
        return text.substr(begin, text.find_last_not_of(space) - begin + 1);
    };
    std::vector<size_t> human;
    std::vector<std::string> human_text;
    for (size_t index = 0; index < messages.size(); ++index) {
        if (messages[index].role != "user") continue;
        human.push_back(index);
        human_text.push_back(trim(messages[index].content));
    }
    const DecodedPrompt decoded = decode_prompt_with_offsets(tokenizer, prompt);
    const auto char_at = [&](int token) {
        return token < (int) prompt.size() ? decoded.token_begin[(size_t) token] : decoded.text.size();
    };
    std::vector<std::string> turn_text(turns.size());
    std::vector<uint8_t> turn_usable(turns.size(), 0);
    for (size_t index = 0; index < turns.size(); ++index) {
        const auto & turn = turns[index];
        if (turn.role == "assistant" || turn.role == "model" ||
            turn.role == "system" || turn.role == "developer" ||
            turn.content_begin < 0 || turn.content_end < turn.content_begin ||
            turn.content_end > (int) prompt.size()) {
            continue;
        }
        turn_usable[index] = 1;
        turn_text[index] = trim(tokenizer.decode(std::vector<int32_t>(
            prompt.begin() + turn.content_begin, prompt.begin() + turn.content_end)));
    }
    const size_t wanted = (std::min)(count, human.size());
    for (size_t k = 0; k < wanted; ++k) {
        const size_t h = human.size() - 1 - k;
        const std::string & content = human_text[h];
        if (content.empty()) {
            out.push_back({0, 0});
            continue;
        }
        size_t same = 0;
        size_t rank_from_end = 0;
        for (size_t j = 0; j < human.size(); ++j) {
            if (human_text[j] != content) continue;
            ++same;
            if (j > h) ++rank_from_end;
        }
        std::vector<size_t> matches;
        for (size_t index = 0; index < turns.size(); ++index) {
            if (turn_usable[index] && turn_text[index] == content) matches.push_back(index);
        }
        if (matches.size() != same) {
            out.clear();
            return false;
        }
        const auto & turn = turns[matches[matches.size() - 1 - rank_from_end]];
        // The span of the message's exact content, its own leading and
        // trailing whitespace included (the turn bounds trim whatever
        // whitespace surrounds it): every token overlapping those characters.
        const std::string & raw = messages[human[h]].content;
        const size_t lead = raw.find_first_not_of(" \t\n\r");
        const size_t seg_begin = char_at(turn.content_begin);
        const size_t seg_end = char_at(turn.content_end);
        const std::string seg = decoded.text.substr(seg_begin, seg_end - seg_begin);
        const size_t seg_lead = seg.find_first_not_of(" \t\n\r");
        if (seg_lead == std::string::npos || seg_begin + seg_lead < lead) {
            out.clear();
            return false;
        }
        const size_t char_begin = seg_begin + seg_lead - lead;
        const size_t char_end = char_begin + raw.size();
        if (char_end > decoded.text.size() ||
            decoded.text.compare(char_begin, raw.size(), raw) != 0) {
            out.clear();
            return false;
        }
        const int begin = token_at_offset(decoded, char_begin);
        const int end = (int) (std::lower_bound(decoded.token_begin.begin(),
            decoded.token_begin.end(), char_end) - decoded.token_begin.begin());
        out.push_back({begin, end});
    }
    return true;
}

int pflash_chat_skeleton_tokens() noexcept {
    const char * raw = std::getenv("PFLASH_CHAT_SKELETON_TOKENS");
    if (!raw || !*raw) return 256;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0) return 256;
    return (int) (std::min)(value, 1L << 20);
}

bool pflash_paragraph_join() noexcept {
    const char * raw = std::getenv("PFLASH_SELECT_PARAGRAPH_JOIN");
    return !(raw && std::string(raw) == "0");
}

std::string pflash_join_kept_spans(
        const Tokenizer & tokenizer,
        const std::vector<int32_t> & ids,
        const std::vector<PFlashTokenSpan> & spans) {
    std::string out;
    int previous_end = -1;
    for (const auto & span : spans) {
        if (span.begin < 0 || span.end > (int) ids.size() || span.end <= span.begin) {
            continue;
        }
        std::string piece = tokenizer.decode(std::vector<int32_t>(
            ids.begin() + span.begin, ids.begin() + span.end));
        if (previous_end >= 0 && span.begin > previous_end && !out.empty() &&
            !piece.empty()) {
            const bool left_break = out.back() == '\n';
            const bool right_break = piece.front() == '\n';
            if (!left_break && !right_break) {
                out += "\n\n";
            } else if (left_break != right_break &&
                       !(out.size() >= 2 && out[out.size() - 2] == '\n') &&
                       !(piece.size() >= 2 && piece[1] == '\n')) {
                out += "\n";
            }
        }
        out += piece;
        previous_end = span.end;
    }
    return out;
}

bool pflash_chat_recall() noexcept {
    const char * raw = std::getenv("PFLASH_CHAT_RECALL");
    return !(raw && std::string(raw) == "0");
}

int pflash_chat_compress_new_tokens() noexcept {
    const char * raw = std::getenv("PFLASH_CHAT_COMPRESS_NEW_TOKENS");
    if (!raw || !*raw) return 16384;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 1) return 16384;
    return (int) (std::min)(value, 1L << 30);
}

int pflash_chat_history_queries() noexcept {
    const char * raw = std::getenv("PFLASH_CHAT_HISTORY_QUERIES");
    if (!raw || !*raw) return 3;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0) return 3;
    return (int) (std::min)(value, 8L);
}

std::vector<PFlashTokenSpan> pflash_subtract_token_spans(
        const std::vector<PFlashTokenSpan> & spans,
        const std::vector<PFlashTokenSpan> & minus) {
    std::vector<PFlashTokenSpan> out;
    size_t cut = 0;
    for (const auto & span : spans) {
        int begin = span.begin;
        while (cut < minus.size() && minus[cut].end <= begin) ++cut;
        for (size_t index = cut;
             index < minus.size() && minus[index].begin < span.end; ++index) {
            if (minus[index].begin > begin) {
                out.push_back({begin, minus[index].begin});
            }
            begin = (std::max)(begin, minus[index].end);
        }
        if (begin < span.end) out.push_back({begin, span.end});
    }
    return out;
}

std::string pflash_recall_excerpt(
        const std::string & text,
        const std::vector<std::string> & role_markers,
        const std::vector<std::string> & end_markers,
        bool generic_role_lines) {
    std::string out;
    out.reserve(text.size());
    size_t at = 0;
    while (at < text.size()) {
        bool matched = false;
        for (const auto & marker : role_markers) {
            if (marker.empty() || text.compare(at, marker.size(), marker) != 0) {
                continue;
            }
            at += marker.size();
            if (generic_role_lines) {
                size_t name_end = at;
                while (name_end < text.size() && name_end - at < 16 &&
                       std::isalpha((unsigned char) text[name_end])) {
                    ++name_end;
                }
                if (name_end < text.size() && text[name_end] == '\n') {
                    at = name_end + 1;
                }
            }
            out += '\n';
            matched = true;
            break;
        }
        if (matched) continue;
        for (const auto & marker : end_markers) {
            if (marker.empty() || text.compare(at, marker.size(), marker) != 0) {
                continue;
            }
            at += marker.size();
            out += '\n';
            matched = true;
            break;
        }
        if (matched) continue;
        out += text[at++];
    }
    const size_t first = out.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const size_t last = out.find_last_not_of(" \t\r\n");
    return out.substr(first, last - first + 1);
}

bool PflashChatViewStore::find(
        const std::vector<int32_t> & raw_tokens,
        const std::vector<int32_t> & drafter_ids,
        PflashChatView & out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const PflashChatView * best = nullptr;
    for (const auto & view : views_) {
        if (view.raw_gen_begin <= 0 || view.drafter_gen_begin <= 0 ||
            (size_t) view.raw_gen_begin > raw_tokens.size() ||
            (size_t) view.drafter_gen_begin > drafter_ids.size()) {
            continue;
        }
        if (!std::equal(view.raw_tokens.begin(),
                        view.raw_tokens.begin() + view.raw_gen_begin,
                        raw_tokens.begin()) ||
            !std::equal(view.drafter_ids.begin(),
                        view.drafter_ids.begin() + view.drafter_gen_begin,
                        drafter_ids.begin())) {
            continue;
        }
        if (!best || view.raw_gen_begin > best->raw_gen_begin) best = &view;
    }
    if (!best) return false;
    out = *best;
    return true;
}

void PflashChatViewStore::remember(PflashChatView view) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Drop the views this one continues: same conversation, older turn.
    views_.erase(std::remove_if(views_.begin(), views_.end(),
        [&view] (const PflashChatView & old) {
            return old.raw_gen_begin > 0 &&
                old.raw_gen_begin <= view.raw_gen_begin &&
                (size_t) old.raw_gen_begin <= view.raw_tokens.size() &&
                std::equal(old.raw_tokens.begin(),
                           old.raw_tokens.begin() + old.raw_gen_begin,
                           view.raw_tokens.begin());
        }), views_.end());
    views_.push_back(std::move(view));
    while (views_.size() > capacity_) views_.erase(views_.begin());
}

size_t PflashChatViewStore::forget(const std::vector<int32_t> & raw_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t before = views_.size();
    views_.erase(std::remove_if(views_.begin(), views_.end(),
        [&] (const PflashChatView & view) {
            return view.raw_gen_begin > 0 &&
                (size_t) view.raw_gen_begin <= raw_tokens.size() &&
                std::equal(view.raw_tokens.begin(),
                           view.raw_tokens.begin() + view.raw_gen_begin,
                           raw_tokens.begin());
        }), views_.end());
    return before - views_.size();
}

size_t PflashChatViewStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return views_.size();
}

bool pflash_full_cache_restore_allowed(
        bool selection_environment_present) noexcept {
    return !selection_environment_present;
}

int pflash_kept_tokens(
        int input_tokens,
        int chunk_size,
        int query_begin,
        int query_end,
        const std::vector<PFlashTokenSpan> & kept_spans,
        bool query_suffix_structural) noexcept {
    if (input_tokens <= 0 || chunk_size <= 0) return 0;
    int kept = 0;
    for (int begin = 0; begin < input_tokens; begin += chunk_size) {
        const int end = (std::min)(input_tokens, begin + chunk_size);
        if (luce::pflash::pflash_chunk_is_structurally_required(
                begin, end, query_begin, query_end, input_tokens,
                kept_spans, query_suffix_structural)) {
            kept += end - begin;
        }
    }
    return kept;
}

double pflash_effective_keep_ratio(
        int input_tokens, int kept_tokens, double keep_ratio) noexcept {
    if (input_tokens <= 0 || !std::isfinite(keep_ratio) || keep_ratio <= 0.0) {
        return keep_ratio;
    }
    const int kept = (std::max)(0, (std::min)(kept_tokens, input_tokens));
    // One token of slack so flooring the budget never lands below `kept`.
    const double budget =
        (double) kept + keep_ratio * (double) (input_tokens - kept) + 1.0;
    return (std::min)(1.0, budget / (double) input_tokens);
}

const char * pflash_admission_name(PflashAdmission admission) noexcept {
    switch (admission) {
    case PflashAdmission::Compress:       return "compress";
    case PflashAdmission::UnderThreshold: return "under_threshold";
    case PflashAdmission::WarmCache:      return "warm_cache";
    case PflashAdmission::NewContent:     return "new_content";
    case PflashAdmission::KeepHistory:    return "keep_history";
    }
    return "compress";
}

bool pflash_cache_admission_enabled() noexcept {
    const char * raw = std::getenv("PFLASH_ADMISSION");
    return raw && std::string(raw) == "cache";
}

PflashAdmission pflash_cache_admission(
        int uncached_droppable, int cached_droppable, int threshold,
        bool raw_fits) noexcept {
    if (!raw_fits) return PflashAdmission::Compress;
    if (uncached_droppable < threshold) return PflashAdmission::UnderThreshold;
    if (cached_droppable >= threshold) return PflashAdmission::WarmCache;
    return PflashAdmission::Compress;
}

bool pflash_has_tool_activity(const json & tools,
                              const std::vector<ChatMessage> & messages) noexcept {
    if ((tools.is_array() || tools.is_object()) && !tools.empty()) return true;
    for (const auto & message : messages) {
        if (message.role == "tool" || message.role == "function") return true;
        if (!message.tool_calls.empty()) return true;
    }
    return false;
}

bool pflash_agent_paste_enabled() noexcept {
    const char * raw = std::getenv("PFLASH_AGENT_PASTE");
    return raw && std::string(raw) == "1";
}

const char * pflash_keep_history_scope(
        const std::vector<PflashChatTurn> & turns, int & newest) noexcept {
    newest = -1;
    if (turns.empty()) return "no_turns";
    const auto & last = turns.back();
    if (last.role != "user" || last.content_begin < 0 ||
        last.content_end <= last.content_begin) {
        return "newest_not_user";
    }
    // A first turn too: whatever precedes the newest user message (the
    // system prompt and tools, earlier turns) is served as it is.
    newest = (int) turns.size() - 1;
    return nullptr;
}

PflashAdmission pflash_keep_history_admission(
        PflashAdmission admission, int new_droppable, int threshold,
        bool arrive) noexcept {
    if (admission == PflashAdmission::UnderThreshold) return admission;
    if (new_droppable >= threshold &&
        (admission == PflashAdmission::Compress ||
         admission == PflashAdmission::WarmCache)) {
        return PflashAdmission::NewContent;
    }
    if (admission == PflashAdmission::Compress ||
        (admission == PflashAdmission::WarmCache && arrive)) {
        return PflashAdmission::KeepHistory;
    }
    return admission;
}

int pflash_raw_base_target(
        const Tokenizer & target, const ChatMarkers & markers,
        const std::vector<int32_t> & prompt, const Tokenizer & drafter,
        const std::vector<int32_t> & drafter_ids,
        const std::vector<PflashChatTurn> & turns, int turn) {
    if (turn < 0 || (size_t) turn >= turns.size()) return -1;
    const int drafter_cut = turns[(size_t) turn].role_begin;
    if (drafter_cut <= 0 || drafter_cut > (int) drafter_ids.size()) return -1;
    const auto parsed = pflash_chat_query_turn(target, markers, target, prompt);
    if (parsed.turns.size() != turns.size() ||
        parsed.turns[(size_t) turn].role != turns[(size_t) turn].role) {
        return -1;
    }
    const int cut = parsed.turns[(size_t) turn].role_begin;
    if (cut <= 0 || cut > (int) prompt.size()) return -1;
    const std::string target_text = target.decode(
        std::vector<int32_t>(prompt.begin(), prompt.begin() + cut));
    const std::string drafter_text = drafter.decode(
        std::vector<int32_t>(drafter_ids.begin(), drafter_ids.begin() + drafter_cut));
    return target_text == drafter_text ? cut : -1;
}

PflashRestorablePrefix pflash_restorable_prefix(
        int ram_tokens, bool disk_enabled, int disk_tokens) noexcept {
    const int ram = (std::max)(0, ram_tokens);
    const int disk = disk_enabled ? (std::max)(0, disk_tokens) : 0;
    if (disk > ram) return {disk, "disk"};
    return {ram, ram > 0 ? "ram" : "none"};
}

int pflash_span_tokens_within(
        const std::vector<PFlashTokenSpan> & spans, int begin, int end) noexcept {
    int tokens = 0;
    for (const auto & span : spans) {
        const int lo = (std::max)(span.begin, begin);
        const int hi = (std::min)(span.end, end);
        if (hi > lo) tokens += hi - lo;
    }
    return tokens;
}

namespace {
int pflash_env_tokens(const char * name, int fallback) noexcept {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0) return fallback;
    return (int) (std::min)(value, 1L << 30);
}
}  // namespace

int pflash_chat_rebuild_min_tokens() noexcept {
    return pflash_env_tokens("PFLASH_CHAT_REBUILD_MIN_TOKENS", 0);
}

double pflash_chat_rebuild_ratio() noexcept {
    const char * raw = std::getenv("PFLASH_CHAT_REBUILD_RATIO");
    if (!raw || !*raw) return 2.0;
    char * end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || !std::isfinite(value) || value < 0.0) return 2.0;
    return value;
}

bool pflash_arrive_enabled() noexcept {
    const char * raw = std::getenv("PFLASH_ARRIVE");
    return raw && std::string(raw) == "1";
}

std::vector<PFlashTokenSpan> pflash_complement_token_spans(
        const std::vector<PFlashTokenSpan> & spans, int n) {
    std::vector<PFlashTokenSpan> out;
    int at = 0;
    for (const auto & span : spans) {
        const int begin = (std::max)(0, (std::min)(span.begin, n));
        const int end = (std::max)(0, (std::min)(span.end, n));
        if (begin > at) out.push_back({at, begin});
        at = (std::max)(at, end);
    }
    if (at < n) out.push_back({at, n});
    return out;
}

float pflash_ceiling_retry_keep_ratio(float requested_keep_ratio, int target_ceiling, int overflow,
                                      int & tightening) {
    tightening += overflow + 1;
    const int tightened = target_ceiling - tightening;
    if (tightened <= 0) return 0.0f;
    return requested_keep_ratio * (float) tightened / (float) std::max(1, target_ceiling);
}

int pflash_target_token_ceiling(
        int original_target_tokens, double keep_ratio) noexcept {
    if (original_target_tokens < 0 || !std::isfinite(keep_ratio) ||
        keep_ratio <= 0.0) {
        return -1;
    }
    const double ceiling = std::floor(
        static_cast<double>(original_target_tokens) * keep_ratio);
    if (!std::isfinite(ceiling) || ceiling < 0.0 || ceiling > INT_MAX) {
        return -1;
    }
    return static_cast<int>(ceiling);
}

bool pflash_fail_closed_from_env() noexcept {
    const char * value = std::getenv("PFLASH_FAIL_CLOSED");
    return value != nullptr && *value && std::strcmp(value, "0") != 0;
}

bool pflash_selection_gives_lifts(const luce::pflash::PFlashSelectionConfig & selection) noexcept {
    const char * legacy = std::getenv("PFLASH_QWEN35_LEGACY_SCORER");
    return selection.scorer == luce::pflash::PFlashScorer::Head &&
           selection.strategy == luce::pflash::PFlashSelectStrategy::Segments &&
           !(legacy && std::strcmp(legacy, "1") == 0);
}

std::string pflash_failure_reason(const std::string & error,
                                  const std::string & detail) {
    const auto has = [](const std::string & text, const char * needle) {
        return text.find(needle) != std::string::npos;
    };
    if (has(error, "PFlash relevance spread fallback")) return "spread";
    if (has(error, "PFlash relevance spread unavailable")) return "spread_unavailable";
    if (has(detail, "non-finite") || has(error, "non-finite")) return "nonfinite_scores";
    if (has(error, "remote PFlash drafter")) return "remote_drafter";
    if (has(error, "PFlash compression failed")) return "compress_failed";
    if (has(error, "target-token ceiling")) return "ceiling_exceeded";
    if (has(error, "mapping failed") || has(error, "boundary") ||
        has(error, "normalization failed")) return "mapping_failed";
    if (has(error, "PFlash strict selection") || has(error, "PFlash query head") ||
        has(error, "PFlash region plan")) return "selection_failed";
    return "other";
}

bool pflash_spread_fires(double threshold, bool spread_valid, double spread) noexcept {
    return threshold > 0.0 && spread_valid && std::isfinite(spread) && spread >= threshold;
}

PflashSpreadDecision pflash_spread_decision(double threshold, bool spread_valid,
                                            double spread, bool region_plan) noexcept {
    PflashSpreadDecision decision;
    if (region_plan) {
        decision.skipped = "region_plan";
        return decision;
    }
    decision.fires = pflash_spread_fires(threshold, spread_valid, spread);
    return decision;
}

bool pflash_failure_is_config_error(const std::string & error) {
    // Deterministic for every request of the run (a bad PFLASH_* setting) or
    // a request contract the raw prompt cannot honour: falling back would
    // hide the misconfiguration behind a counter.
    return error.rfind("invalid PFlash strict selection config", 0) == 0 ||
           error.rfind("PFlash pflash_required needs strict budget selection", 0) == 0;
}

PflashFallbackDecision pflash_fallback_decision(
        const std::string & error, bool fail_closed, int raw_prompt_tokens,
        int max_output, int max_ctx) {
    PflashFallbackDecision decision;
    if (fail_closed) {
        decision.refused = "fail_closed";
    } else if (pflash_failure_is_config_error(error)) {
        decision.refused = "config_error";
    } else if (effective_prompt_overflows(raw_prompt_tokens, -1, max_output, max_ctx)) {
        // The admission an uncompressed request gets (validate_request_context /
        // prepare_prompt): the raw prompt plus the reply budget within max_ctx.
        decision.refused = "raw_exceeds_max_ctx";
    } else {
        decision.fallback = true;
    }
    return decision;
}

}  // namespace http_detail

static std::string context_overflow_message(int max_ctx, int prompt_tokens, int max_output) {
    const int requested_tokens = prompt_tokens + max_output;
    return "This model's maximum context length is " + std::to_string(max_ctx) +
           " tokens. However, you requested " + std::to_string(requested_tokens) +
           " tokens (" + std::to_string(prompt_tokens) + " in the messages, " +
           std::to_string(max_output) +
           " in the completion). Please reduce the length of the messages or completion.";
}

static bool prompt_ends_in_open_think(const std::string & prompt) {
    static constexpr const char * kThinkOpen = "<think>";
    static constexpr size_t kThinkOpenLen = 7;
    size_t end = prompt.size();
    while (end > 0) {
        char c = prompt[end - 1];
        if (c != ' ' && c != '\n' && c != '\r' && c != '\t') break;
        --end;
    }
    return end >= kThinkOpenLen &&
           prompt.compare(end - kThinkOpenLen, kThinkOpenLen, kThinkOpen) == 0;
}

// ─── piecewise keep-ratio curve ─────────────────────────────────────────

static float pflash_keep_ratio(const ServerConfig & cfg, int n_tokens) {
    if (cfg.pflash_curve.empty()) return cfg.pflash_keep_ratio;
    const auto & curve = cfg.pflash_curve;
    if (n_tokens <= curve.front().first) return curve.front().second;
    if (n_tokens >= curve.back().first)  return curve.back().second;
    for (size_t i = 0; i + 1 < curve.size(); ++i) {
        if (n_tokens <= curve[i + 1].first) {
            float t = (float)(n_tokens - curve[i].first) /
                      (float)(curve[i + 1].first - curve[i].first);
            return curve[i].second + t * (curve[i + 1].second - curve[i].second);
        }
    }
    return curve.back().second;
}

namespace http_detail {

int flowkv_activation_threshold(const ServerConfig & config) {
    return config.pflash_mode == ServerConfig::PflashMode::ALWAYS
        ? kFlowKvInertMinTokens
        : std::max(kFlowKvInertMinTokens, config.pflash_threshold);
}

bool flowkv_should_activate(const ServerConfig & config,
                            int aged_token_estimate) {
    return aged_token_estimate >= flowkv_activation_threshold(config);
}

float resolve_pflash_keep_ratio(float configured_ratio,
                                const std::string & session_id,
                                const HttpServerSessions & sessions) {
    // A session adapts from the configured (curve) ratio: until its first
    // acceptance feedback it keeps that ratio, afterwards the controller's.
    return session_id.empty()
        ? configured_ratio
        : sessions.get_keep_ratio(session_id, configured_ratio);
}

bool should_clamp_flowkv_disk_cache(
        bool flowkv, const DiskPrefixCachePolicy & policy) {
    return flowkv && policy.compress;
}

bool ends_with_tool_result(const std::vector<ChatMessage> & messages) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->role == "system") continue;
        return it->role == "tool" || it->role == "function";
    }
    return false;
}

bool canonical_turn_matches_checkpoint(
        const std::vector<int32_t> & prompt,
        const std::vector<int32_t> & completed_turn,
        int checkpoint) {
    return checkpoint > 0 && checkpoint <= (int) prompt.size() &&
           checkpoint < (int) completed_turn.size() &&
           std::equal(prompt.begin(), prompt.begin() + checkpoint,
                      completed_turn.begin());
}

bool canonical_assistant_content(
        const std::string & generation_prompt,
        const std::string & sentinel_rendered,
        const std::string & sentinel,
        const std::string & generated_text,
        std::string & content) {
    const size_t pos = sentinel_rendered.find(sentinel);
    if (pos == std::string::npos ||
        sentinel_rendered.find(sentinel, pos + 1) != std::string::npos) {
        return false;
    }
    const std::string assistant_head = sentinel_rendered.substr(0, pos);
    if (generation_prompt.compare(0, assistant_head.size(), assistant_head) != 0) {
        return false;
    }
    content = generation_prompt.substr(assistant_head.size()) + generated_text;
    return true;
}

}  // namespace http_detail

// ─── curl helpers for upstream proxy ─────────────────────────────────────
#ifdef LUCE_HAS_CURL

struct CurlWriteCtx {
    bool streaming;
    bool first_chunk;
    bool chat_rewrite;   // rewrite completions → chat format
    std::string buffer;  // accumulates non-streaming response
    std::string sse_partial_line;
    std::string response_id;
    std::string model;
    std::function<bool(const void *, size_t)> send_bytes;
    std::function<void()> stop_stream;
    std::function<bool()> cancelled;
};

static size_t curl_write_passthrough(char * ptr, size_t size, size_t nmemb, void * userdata) {
    size_t total = size * nmemb;
    auto * ctx = static_cast<CurlWriteCtx *>(userdata);
    if (ctx->streaming) {
        if (http_detail::sse_chunk_has_done(
                ctx->sse_partial_line, ptr, total)) {
            ctx->stop_stream();
        }
        if (!ctx->send_bytes(ptr, total)) return 0;
    } else {
        ctx->buffer.append(ptr, total);
    }
    return total;
}

static size_t curl_write_rewrite(char * ptr, size_t size, size_t nmemb, void * userdata) {
    size_t total = size * nmemb;
    auto * ctx = static_cast<CurlWriteCtx *>(userdata);

    if (!ctx->streaming) {
        ctx->buffer.append(ptr, total);
        return total;
    }

    // Streaming: rewrite completions SSE chunks → chat completions format.
    ctx->buffer.append(ptr, total);
    std::string & buf = ctx->buffer;
    size_t pos = 0;
    while (true) {
        size_t nl = buf.find('\n', pos);
        if (nl == std::string::npos) break;
        std::string line = buf.substr(pos, nl - pos);
        pos = nl + 1;
        if (line.empty() || line == "\r") {
            std::string out = "\n";
            if (!ctx->send_bytes(out.data(), out.size())) return 0;
            continue;
        }
        if (line.size() > 0 && line.back() == '\r') line.pop_back();
        if (line.rfind("data: ", 0) != 0) {
            line += "\n";
            if (!ctx->send_bytes(line.data(), line.size())) return 0;
            continue;
        }
        std::string payload = line.substr(6);
        if (payload == "[DONE]") {
            std::string out = "data: [DONE]\n\n";
            ctx->stop_stream();
            if (!ctx->send_bytes(out.data(), out.size())) return 0;
            continue;
        }
        try {
            auto j = json::parse(payload);
            j["object"] = "chat.completion.chunk";
            if (j.contains("choices") && j["choices"].is_array()) {
                for (auto & c : j["choices"]) {
                    json delta;
                    if (ctx->first_chunk) {
                        delta["role"] = "assistant";
                        ctx->first_chunk = false;
                    }
                    if (c.contains("text")) {
                        delta["content"] = c["text"];
                        c.erase("text");
                    }
                    c["delta"] = delta;
                    c.erase("index"); // re-add below
                    if (!c.contains("index")) c["index"] = 0;
                }
            }
            std::string out = "data: " + j.dump() + "\n\n";
            if (!ctx->send_bytes(out.data(), out.size())) return 0;
        } catch (...) {
            std::string out = line + "\n";
            if (!ctx->send_bytes(out.data(), out.size())) return 0;
        }
    }
    buf.erase(0, pos);
    return total;
}

static json rewrite_completions_to_chat(const json & comp_resp) {
    json chat_resp;
    chat_resp["id"] = comp_resp.value("id", "");
    chat_resp["object"] = "chat.completion";
    chat_resp["created"] = comp_resp.value("created", 0);
    chat_resp["model"] = comp_resp.value("model", "");
    if (comp_resp.contains("usage")) chat_resp["usage"] = comp_resp["usage"];
    json choices = json::array();
    if (comp_resp.contains("choices") && comp_resp["choices"].is_array()) {
        for (const auto & c : comp_resp["choices"]) {
            json choice;
            choice["index"] = c.value("index", 0);
            choice["finish_reason"] = c.value("finish_reason", "stop");
            json msg = {{"role", "assistant"}, {"content", c.value("text", "")}};
            choice["message"] = msg;
            choices.push_back(choice);
        }
    }
    chat_resp["choices"] = choices;
    return chat_resp;
}

static int curl_progress_cancelled(
        void * userdata, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto * ctx = static_cast<CurlWriteCtx *>(userdata);
    return ctx->cancelled && ctx->cancelled() ? 1 : 0;
}

static bool curl_forward(const std::string & url,
                         const std::string & api_key, const json & body,
                         bool streaming, bool rewrite_to_chat,
                         const std::string & response_id,
                         const std::string & model,
                         std::function<bool(const void *, size_t)> send_bytes,
                         std::function<void()> stop_stream,
                         std::function<bool()> cancelled) {
    CURL * curl = curl_easy_init();
    if (!curl) return false;

    std::string body_str = body.dump();

    struct curl_slist * headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!api_key.empty()) {
        std::string auth = "Authorization: Bearer " + api_key;
        headers = curl_slist_append(headers, auth.c_str());
    }

    CurlWriteCtx ctx;
    ctx.streaming = streaming;
    ctx.first_chunk = true;
    ctx.chat_rewrite = rewrite_to_chat;
    ctx.response_id = response_id;
    ctx.model = model;
    ctx.send_bytes = std::move(send_bytes);
    ctx.stop_stream = std::move(stop_stream);
    ctx.cancelled = std::move(cancelled);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body_str.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3600L);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_progress_cancelled);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);

    if (rewrite_to_chat) {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_rewrite);
    } else {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_passthrough);
    }

    CURLcode res = curl_easy_perform(curl);
    if (streaming) ctx.stop_stream();

    bool response_sent = true;
    if (!streaming && res == CURLE_OK) {
        // Non-streaming: send accumulated response.
        if (rewrite_to_chat) {
            try {
                json resp = json::parse(ctx.buffer);
                json chat_resp = rewrite_completions_to_chat(resp);
                std::string out = chat_resp.dump();
                std::string http =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(out.size()) + "\r\n"
                    "\r\n" + out;
                response_sent = ctx.send_bytes(http.data(), http.size());
            } catch (...) {
                std::string http =
                    "HTTP/1.1 502 Bad Gateway\r\n"
                    "Content-Type: application/json\r\n"
                    "\r\n{\"error\":\"upstream response parse failed\"}";
                response_sent = ctx.send_bytes(http.data(), http.size());
            }
        } else {
            std::string http =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " + std::to_string(ctx.buffer.size()) + "\r\n"
                "\r\n" + ctx.buffer;
            response_sent = ctx.send_bytes(http.data(), http.size());
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return res == CURLE_OK && response_sent;
}
#endif // LUCE_HAS_CURL

// ─── /props constants ───────────────────────────────────────────────────
//
// SERVER_NAME / SERVER_VERSION mirror the Python server's identity strings
// so cross-server consumers (autotune, dashboards) see a stable
// `build_info` shape. Bump PROPS_SCHEMA on breaking changes only:
//   - field renamed
//   - field removed
//   - existing field's semantics change (units, nullability, type)
// Do NOT bump for additive changes (new fields, new sections).
static constexpr int  kPropsSchema  = 2;
static constexpr char kServerName[] = "luce-dflash";
#ifndef LUCE_SERVER_VERSION
#define LUCE_SERVER_VERSION "0.0.0+cpp"
#endif

// API endpoint registry served by /props. Keep in sync with the route
// handlers in handle_client() and route_request().
static const std::vector<std::string> kApiEndpoints = {
    "GET /health",
    "GET /props",
    "GET /status",
    "GET /status/events",
    "GET /status/json",
    "GET /v1/models",
    "POST /v1/chat/completions",
    "POST /v1/messages",
    "POST /v1/messages/count_tokens",
    "POST /v1/responses",
};

// ─── Utilities ──────────────────────────────────────────────────────────

static std::string generate_id(const char * prefix) {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%016llx",
                  prefix, (unsigned long long)counter.fetch_add(1));
    return buf;
}

static size_t json_array_size(const json & value) {
    return value.is_array() ? value.size() : 0;
}

int resolve_max_output_tokens(const json & body, int default_max_tokens) {
    // OpenAI-compatible clients (e.g. PocketPal's "Unlimited") send
    // max_completion_tokens: -1 to mean "no explicit limit"; 0 is also
    // invalid as a budget. Treat non-positive values as unset so they
    // fall back to the server default instead of yielding zero tokens.
    auto field_or_default = [&](const char * key) {
        const int value = body.at(key).get<int>();
        return value > 0 ? value : default_max_tokens;
    };
    if (body.contains("max_tokens")) {
        return field_or_default("max_tokens");
    }
    if (body.contains("max_output_tokens")) {
        return field_or_default("max_output_tokens");
    }
    if (body.contains("max_completion_tokens")) {
        return field_or_default("max_completion_tokens");
    }
    return default_max_tokens;
}

bool ppp_prefers_tools_boundary(bool ppp_enabled, bool has_tools) {
    return ppp_enabled && has_tools;
}

const std::vector<int32_t> & ppp_tool_prefix_source(
        bool compressed, const std::vector<int32_t> & request_tokens,
        const std::vector<int32_t> & served_tokens) {
    return compressed ? request_tokens : served_tokens;
}

// Sampler parameters. When the request omits a value, fall back to the
// model card's sampling defaults (spec §3.3); when the card doesn't
// supply one either, use the hard-coded default.
SamplerCfg parse_request_sampler(const json & body,
                                 const SamplingDefaults & defaults) {
    SamplerCfg sampler;
    sampler.temp = body.value(
        "temperature", defaults.has_temperature ? defaults.temperature : 0.0f);
    sampler.top_p = body.value(
        "top_p", defaults.has_top_p ? defaults.top_p : 1.0f);
    sampler.top_k = body.value(
        "top_k", defaults.has_top_k ? defaults.top_k : 0);
    if (body.contains("seed")) {
        sampler.seed = body["seed"].get<uint64_t>();
    }

    // OpenAI-style additive penalties.
    sampler.freq_pen = body.value("frequency_penalty", 0.0f);
    sampler.pres_pen = body.value(
        "presence_penalty",
        defaults.has_presence_penalty ? defaults.presence_penalty : 0.0f);
    // HuggingFace-style multiplicative repetition penalty (also used by
    // vLLM, llama.cpp, etc.). Accepts both "repetition_penalty" and the
    // shorter "rep_pen" for daemon compatibility.
    sampler.rep_pen = body.value(
        "repetition_penalty",
        body.value("rep_pen",
                   defaults.has_repetition_penalty
                       ? defaults.repetition_penalty
                       : 1.0f));
    if (body.contains("rep_window")) {
        sampler.rep_window = body["rep_window"].get<int>();
    }
    return sampler;
}

void apply_no_thinking_sampler_defaults(const json & body,
                                        const SamplingDefaults & no_thinking,
                                        SamplerCfg & sampler) {
    // Mirrors the body-key checks in parse_request_sampler: only backfill a
    // field the request left unset, and only when the card actually
    // supplied a no-thinking value for it.
    if (no_thinking.has_temperature && !body.contains("temperature")) {
        sampler.temp = no_thinking.temperature;
    }
    if (no_thinking.has_top_p && !body.contains("top_p")) {
        sampler.top_p = no_thinking.top_p;
    }
    if (no_thinking.has_top_k && !body.contains("top_k")) {
        sampler.top_k = no_thinking.top_k;
    }
    if (no_thinking.has_presence_penalty && !body.contains("presence_penalty")) {
        sampler.pres_pen = no_thinking.presence_penalty;
    }
    if (no_thinking.has_repetition_penalty &&
        !body.contains("repetition_penalty") && !body.contains("rep_pen")) {
        sampler.rep_pen = no_thinking.repetition_penalty;
    }
}

json require_messages_array(const json & body) {
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        throw std::invalid_argument("messages must be a non-empty array");
    }
    return body["messages"];
}

static bool env_flag_enabled(const char * name) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return false;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value != "0" && value != "false" && value != "no" &&
           value != "off";
}

static const json * find_tool_function(const json & tools,
                                       const std::string & name) {
    if (!tools.is_array() || name.empty()) return nullptr;
    for (const auto & tool : tools) {
        if (!tool.contains("function") || !tool["function"].is_object()) {
            continue;
        }
        const json & fn = tool["function"];
        if (fn.value("name", "") == name) return &fn;
    }
    return nullptr;
}

static std::string first_tool_parameter_name(const json & function_def) {
    const auto & params = function_def.value("parameters", json::object());
    if (params.contains("required") && params["required"].is_array()) {
        for (const auto & name : params["required"]) {
            if (name.is_string()) return name.get<std::string>();
        }
    }
    if (params.contains("properties") && params["properties"].is_object()) {
        for (const auto & item : params["properties"].items()) {
            return item.key();
        }
    }
    return "";
}

static const json * select_stall_recovery_function(const json & tools,
                                                   const json & tool_choice) {
    if (!tools.is_array() || tools.empty()) return nullptr;

    if (tool_choice.is_object() && tool_choice.contains("function") &&
        tool_choice["function"].is_object()) {
        const std::string forced_name =
            tool_choice["function"].value("name", "");
        // If the request forced a concrete function, recovery must honor it;
        // falling back to terminal here would synthesize invalid tool XML.
        return find_tool_function(tools, forced_name);
    }

    if (tool_choice.is_string() && tool_choice.get<std::string>() == "required" &&
        tools.size() == 1 && tools[0].contains("function") &&
        tools[0]["function"].is_object()) {
        return &tools[0]["function"];
    }

    if (const json * terminal = find_tool_function(tools, "terminal")) {
        return terminal;
    }
    if (tools.size() == 1 && tools[0].contains("function") &&
        tools[0]["function"].is_object()) {
        return &tools[0]["function"];
    }
    return nullptr;
}

static std::string build_stall_tool_prefix(const json & tools,
                                           const json & tool_choice) {
    const json * function_def =
        select_stall_recovery_function(tools, tool_choice);
    if (!function_def) return "\n<function=";

    const std::string name = function_def->value("name", "");
    if (name.empty()) return "\n<function=";

    std::string prefix = "\n<function=" + name + ">\n";
    std::string param = first_tool_parameter_name(*function_def);
    if (!param.empty()) {
        prefix += "<parameter=" + param + ">\n";
    }
    return prefix;
}

// Build the /props response body.
//
// Non-static so unit tests can call it directly (declared in http_server.h).
json build_props_body(const ServerConfig & config,
                      const PrefixCache & prefix_cache,
                      const ToolMemory & tool_memory) {
    // arch-gated capabilities (mirrors Python _capabilities()).
    const bool is_qwen = (config.arch.rfind("qwen", 0) == 0);
    const bool is_deepseek4 = (arch_is_deepseek4_family(config.arch));
    const bool reasoning_supported = is_qwen || is_deepseek4;
    const bool speculative_supported = is_qwen;
    const bool tools_supported = is_qwen || arch_is_deepseek4_family(config.arch);

    auto pcs  = prefix_cache.stats();
    auto pcfs = prefix_cache.full_stats();
    auto tms  = tool_memory.stats();

    const bool pflash_enabled =
        (config.pflash_mode != ServerConfig::PflashMode::OFF);
    // speculative_mode reports the *active* path, not arch capability. A
    // Qwen-family model started without --ddtree has the capability but no
    // active speculative decode, so it must report "off" — otherwise clients
    // see `speculative_mode == "dflash"` paired with `speculative.enabled ==
    // false` and the two contradict (codex review feedback on 8d6ff04).
    std::string speculative_mode;
    if (pflash_enabled)                    speculative_mode = "pflash";
    else if (config.speculative_enabled)   speculative_mode = "dflash";
    else                                   speculative_mode = "off";

    // Spec §4.2: the five-tier vocabulary (low | medium | high | x-high | max)
    // all activate the phase-1 envelope. Advertise the full set when the
    // arch supports reasoning so clients can negotiate the higher tiers.
    json reasoning_efforts = json::array();
    if (is_qwen) {
        reasoning_efforts.push_back("low");
        reasoning_efforts.push_back("medium");
        reasoning_efforts.push_back("high");
        reasoning_efforts.push_back("x-high");
        reasoning_efforts.push_back("max");
    } else if (is_deepseek4) {
        reasoning_efforts.push_back("low");
        reasoning_efforts.push_back("high");
        reasoning_efforts.push_back("max");
    }

    json server = {
        {"name",         kServerName},
        {"version",      LUCE_SERVER_VERSION},
        {"props_schema", kPropsSchema},
    };

    json pflash;
    if (!pflash_enabled) {
        pflash = {
            {"enabled",      false},
            {"mode",         "off"},
            {"threshold",    nullptr},
            {"keep_ratio",   nullptr},
            {"drafter_gguf", nullptr},
            {"skip_park",    nullptr},
            {"bsa_enabled",  nullptr},
            {"bsa_alpha",    nullptr},
            {"lm_head_fix",  nullptr},
            {"draft_residency", draft_residency_policy_name(config.draft_residency)},
        };
    } else {
        const char * bsa_env = std::getenv("LUCE_FP_USE_BSA");
        const char * alpha_env = std::getenv("LUCE_FP_ALPHA");
        const char * lmfix_env = std::getenv("LUCE_LM_HEAD_FIX");
        json bsa_alpha = nullptr;
        if (alpha_env && *alpha_env) {
            try { bsa_alpha = std::stod(alpha_env); }
            catch (const std::exception &) { bsa_alpha = nullptr; }
        }
        std::string mode_str =
            (config.pflash_mode == ServerConfig::PflashMode::AUTO)   ? "auto"   :
            (config.pflash_mode == ServerConfig::PflashMode::ALWAYS) ? "always" : "off";
        pflash = {
            {"enabled",      true},
            {"mode",         mode_str},
            {"threshold",    config.pflash_threshold},
            {"keep_ratio",   config.pflash_keep_ratio},
            {"drafter_gguf", config.pflash_drafter_path.empty()
                              ? json(nullptr)
                              : json(config.pflash_drafter_path)},
            {"skip_park",    config.pflash_skip_park},
            {"bsa_enabled",  (bsa_env != nullptr && *bsa_env && std::strcmp(bsa_env, "0") != 0)},
            {"bsa_alpha",    bsa_alpha},
            {"lm_head_fix",  (lmfix_env != nullptr && *lmfix_env && std::strcmp(lmfix_env, "0") != 0)},
            {"draft_residency", draft_residency_policy_name(config.draft_residency)},
        };
    }

    // Reflect actual sampler defaults the server applies when a request
    // omits the field — these come from the loaded model card's sampling
    // section (spec §3.3), not from a hard-coded greedy fallback. Clients
    // that read /props to pick their sampling shape were getting greedy
    // here regardless of what the model card said, which caused gemma4
    // benchmarks to silently run at temp=0 (degenerate-decode collapse)
    // when the model card specifies temp=1.0/top_p=0.95/top_k=64.
    const auto & smp = config.sampler_defaults;
    json body = {
        {"default_generation_settings", {
            {"n_ctx",          config.max_ctx},
            {"temperature",    smp.has_temperature        ? smp.temperature        : 0.0f},
            {"top_p",          smp.has_top_p              ? smp.top_p              : 1.0f},
            {"top_k",          smp.has_top_k              ? smp.top_k              : 0},
            {"min_p",          smp.has_min_p              ? smp.min_p              : 0.0f},
            {"repeat_penalty", smp.has_repetition_penalty ? smp.repetition_penalty : 1.0f},
        }},
        {"model_alias", config.model_name},
        {"model_path",  config.model_path},
        {"build_info",  std::string(kServerName) + " v" LUCE_SERVER_VERSION
                        " props_schema=" + std::to_string(kPropsSchema)},
        {"speculative_mode", speculative_mode},
        {"server", server},
        {"model", {
            {"arch",         config.arch},
            {"draft_path",   config.draft_path.empty() ? json(nullptr) : json(config.draft_path)},
            {"tokenizer_id", config.tokenizer_id.empty() ? json(nullptr) : json(config.tokenizer_id)},
        }},
        {"runtime", {
            {"backend",         config.runtime_backend.empty() ? "cuda" : config.runtime_backend},
            {"fa_window",       config.fa_window},
            {"kv_cache_k",      config.kv_cache_k},
            {"kv_cache_v",      config.kv_cache_v},
            {"lazy_draft",      config.lazy_draft},
            {"draft_residency", draft_residency_policy_name(config.draft_residency)},
            {"target_sharding", config.target_sharding},
            // Prefill chunk size (bargs.chunk). Surfaced so snapshot
            // tooling captures the full config — bench consumers
            // (server/scripts/bench_http_capability.py) read
            // /props.runtime wholesale into result.json.server_info.
            {"chunk",           config.chunk},
            {"continuous_batching", {
                {"admission_coalesce_ms", config.admission_coalesce_ms},
                {"decode_kv_offload_bytes", config.decode_kv_offload_bytes},
            }},
            // Device placement strings (e.g. "auto:0", "cuda:0"). Empty
            // string when no draft model is loaded.
            {"target_device",   config.target_device},
            {"draft_device",    config.draft_device.empty() ? json(nullptr) : json(config.draft_device)},
        }},
        {"reasoning", {
            {"supported",         reasoning_supported},
            {"default",           nullptr},
            {"supported_efforts", reasoning_efforts},
        }},
        // `model_card`: 1:1 with the on-disk sidecar JSON when one was
        // loaded; null when family fallback or hard fallback was used.
        // Validates against share/model_cards/_schema.json. The `source`
        // field here is the upstream model-card URL (authored in the
        // sidecar) — NOT a filepath. See spec §4.9.
        {"model_card", config.model_card_json.is_null()
                           ? json(nullptr)
                           : config.model_card_json},
        // `budget_envelope`: runtime-resolved values driving the
        // thinking-budget envelope. May differ from the authored card
        // values because of CLI overrides and max_ctx-based tier clamping
        // (spec §3.5). Always emitted regardless of model_card source.
        // See spec §4.2.
        {"budget_envelope", {
            {"model_card_source",       config.model_card_source_label},
            {"default_max_tokens",      config.default_max_tokens},
            {"hard_limit_reply_budget", config.hard_limit_reply_budget},
            {"think_max_tokens",        config.think_max_tokens},
            {"effort_tiers", {
                {"low",    config.effort_tiers.low},
                {"medium", config.effort_tiers.medium},
                {"high",   config.effort_tiers.high},
                {"x-high", config.effort_tiers.x_high},
                {"max",    config.effort_tiers.max},
            }},
        }},
        {"speculative", {
            {"enabled",       config.speculative_enabled},
            {"ddtree_budget", config.speculative_enabled
                                ? json(config.ddtree_budget) : json(nullptr)},
        }},
        {"sampling", {
            {"capabilities", {
                {"supports_temperature",        true},
                {"supports_top_p",              true},
                {"supports_top_k",              true},
                {"supports_frequency_penalty",  true},
                {"supports_seed",               true},
            }},
        }},
        {"pflash", pflash},
        {"prefix_cache", {
            {"capacity",           pcs.capacity},
            {"in_use",             pcs.in_use},
            {"lifetime_hits",      pcs.lifetime_hits},
            {"agent_turn_enabled", config.agent_turn_cache},
            {"max_resident_bytes", pcs.max_resident_bytes},
            {"resident_bytes",     pcs.resident_bytes},
            {"budget_skips",        pcs.budget_skips},
            {"capture_attempts",    pcs.capture_attempts},
            {"capture_failures",    pcs.capture_failures},
            {"capture_stall_ms_total",
                (double)pcs.capture_stall_us_total / 1000.0},
            {"capture_stall_ms_max",
                (double)pcs.capture_stall_us_max / 1000.0},
            {"restore_attempts",    pcs.restore_attempts},
            {"restore_invalidations", pcs.restore_invalidations},
            {"restore_stall_ms_total",
                (double)pcs.restore_stall_us_total / 1000.0},
            {"restore_stall_ms_max",
                (double)pcs.restore_stall_us_max / 1000.0},
        }},
        {"full_cache", {
            {"enabled",       pcfs.enabled},
            {"capacity",      pcfs.capacity},
            {"in_use",        pcfs.in_use},
            {"disk_bytes",    pcfs.disk_bytes},
            {"lifetime_hits", pcfs.lifetime_hits},
            {"disk_policy",   disk_prefix_cache_policy_name(config.disk_cache_policy)},
            {"disk_enabled",  !config.disk_cache_dir.empty()},
            {"disk_budget_mb", config.disk_cache_budget_mb},
            {"disk_ttl_hours", config.disk_cache_ttl_hours},
        }},
        {"tool_replay", {
            {"max_entries",     tms.max_entries},
            {"max_bytes",       tms.max_bytes},
            {"current_entries", tms.current_entries},
            {"current_bytes",   tms.current_bytes},
        }},
        // The C++ daemon is linked in-process; if /props is responding,
        // the daemon is alive by construction.
        {"daemon", {{"alive", true}}},
        {"api", {{"endpoints", kApiEndpoints}}},
        // Capability flags surfaced for clients that don't want to crack
        // open `reasoning` / `speculative` / etc. — matches the Python
        // server's _capabilities() helper.
        {"capabilities", {
            {"reasoning_supported",   reasoning_supported},
            {"speculative_supported", speculative_supported},
            {"tools_supported",       tools_supported},
            {"image_input_supported", config.image_input_enabled},
        }},
    };
    return body;
}

// Normalize Anthropic's `system` field (top-level on /v1/messages and
// /v1/messages/count_tokens) into a leading `{role:"system", content:...}`
// entry on `messages`. Accepts either a flat string or an array of typed
// blocks (`[{type:"text", text:"..."}]`), and strips any
// `x-anthropic-billing-header:`-prefixed block injected by Claude Code so
// it never reaches the model or the token counter.
//
// Side-effect: prepends a system message to `messages` when the body has
// a non-empty `system` field after billing-header filtering. No-op
// otherwise. Both endpoints call this with identical semantics — having
// one helper guarantees token counting and generation can't drift.
static void normalize_anthropic_system(const json & body, json & messages) {
    if (!body.contains("system")) return;
    // Delegate strip to the pure fn; insert as system message.
    std::string text = luce::common::normalize_system_for_cache(body["system"]);
    if (!text.empty()) {
        json sys_msg = {{"role", "system"}, {"content", text}};
        messages.insert(messages.begin(), sys_msg);
    }
}

static json normalize_tool_arguments(const json * arguments) {
    if (!arguments) return json::object();
    if (arguments->is_object()) return *arguments;
    if (arguments->is_string()) {
        const auto & text = arguments->get_ref<const std::string &>();
        if (text.empty()) return json::object();
        auto parsed = json::parse(text, nullptr, false);
        if (parsed.is_object()) return parsed;
    }
    // Retain malformed/non-object arguments rather than silently losing them.
    return json{{"arguments", *arguments}};
}

// Text of an Anthropic tool_result block: a string, or the text blocks of an
// array (images and documents have no text form here).
static std::string anthropic_tool_result_text(const json & block) {
    if (!block.contains("content")) return {};
    const auto & content = block["content"];
    if (content.is_string()) return content.get<std::string>();
    std::string text;
    if (content.is_array()) {
        for (const auto & part : content) {
            if (!part.is_object() || part.value("type", "") != "text") continue;
            if (!text.empty()) text += "\n";
            text += part.value("text", "");
        }
    }
    return text;
}

std::vector<ChatMessage> normalize_chat_messages(
    const json & messages,
    ApiFormat format,
    ToolMemory & tool_memory,
    bool replay_tool_memory) {
    std::vector<ChatMessage> chat_msgs;
    std::vector<std::string> system_parts;
    ChatMessage response_turn;
    std::vector<const json *> response_arguments;

    auto append_call = [](ChatMessage & message, std::vector<const json *> & arguments,
                          const json & function, std::string id, const char * argument_key) {
        message.tool_calls.push_back({std::move(id), function.value("name", ""), {}});
        const auto supplied = function.find(argument_key);
        arguments.push_back(supplied == function.end() ? nullptr : &*supplied);
    };
    auto finish_calls = [&](ChatMessage & message, const std::vector<const json *> & arguments) {
        if (message.tool_calls.empty()) return;
        std::string raw;
        if (replay_tool_memory && !tool_memory.disabled()) {
            std::vector<std::string> ids;
            ids.reserve(message.tool_calls.size());
            for (const auto & call : message.tool_calls) ids.push_back(call.id);
            // Empty IDs remain in the lookup: partial/id-less turns cannot replay.
            raw = tool_memory.lookup(ids);
        }
        if (!raw.empty()) {
            message.content = std::move(raw);
            message.tool_calls_replayed = true;
        } else {
            for (size_t i = 0; i < message.tool_calls.size(); ++i) {
                message.tool_calls[i].arguments = normalize_tool_arguments(arguments[i]);
            }
        }
    };
    auto flush_response_turn = [&]() {
        if (response_turn.role.empty()) return;
        finish_calls(response_turn, response_arguments);
        chat_msgs.push_back(std::move(response_turn));
        response_turn = {};
        response_arguments.clear();
    };

    if (messages.is_array()) {
        for (const auto & m : messages) {
            if (format == ApiFormat::RESPONSES && m.is_object()) {
                const std::string item_type = m.value("type", "message");
                if (item_type == "function_call") {
                    response_turn.role = "assistant";
                    append_call(response_turn, response_arguments, m,
                                m.value("call_id", m.value("id", "")), "arguments");
                    continue;
                }
                if (item_type != "message" || m.value("role", "user") != "assistant") {
                    flush_response_turn();
                }
                if (item_type == "function_call_output") {
                    std::string output;
                    if (m.contains("output") && m["output"].is_string()) {
                        output = m["output"].get<std::string>();
                    } else if (m.contains("output")) {
                        output = m["output"].dump();
                    }
                    chat_msgs.push_back({"tool", std::move(output),
                                         m.value("call_id", m.value("id", ""))});
                    continue;
                }
            }

            // Responses splits one assistant turn into message/function_call
            // items. Keep adjacent items together for both raw and cold replay.
            const std::string role = m.value("role", "user");
            const bool response_assistant = format == ApiFormat::RESPONSES && role == "assistant";
            ChatMessage local_message;
            std::vector<const json *> local_arguments;
            ChatMessage & cm = response_assistant ? response_turn : local_message;
            auto & arguments = response_assistant ? response_arguments : local_arguments;
            cm.role = role;

            // Carry a replayed assistant turn's prior <think> text through to
            // the Jinja renderer as message.reasoning_content (OpenAI/DeepSeek
            // dialect; `reasoning` is the OpenRouter/Anthropic-gateway flat
            // alias — same field the response side emits, see
            // format_response_message's reasoning_content/reasoning/
            // reasoning_details trio below).
            if (cm.role == "assistant") {
                if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                    cm.reasoning_content = m["reasoning_content"].get<std::string>();
                } else if (m.contains("reasoning") && m["reasoning"].is_string()) {
                    cm.reasoning_content = m["reasoning"].get<std::string>();
                }
            }

            const bool anthropic_blocks = format == ApiFormat::ANTHROPIC &&
                m.contains("content") && m["content"].is_array();

            // Anthropic tool results precede any user text in the same turn.
            if (anthropic_blocks && cm.role == "user") {
                bool has_results = false;
                std::string text;
                for (const auto & part : m["content"]) {
                    if (!part.is_object()) continue;
                    const std::string ptype = part.value("type", "");
                    if (ptype == "tool_result") {
                        chat_msgs.push_back({"tool", anthropic_tool_result_text(part),
                                             part.value("tool_use_id", "")});
                        has_results = true;
                    } else if (ptype == "text") {
                        text += part.value("text", "");
                    }
                }
                if (has_results) {
                    if (!text.empty()) chat_msgs.push_back({"user", std::move(text)});
                    continue;
                }
            }

            if (cm.role == "assistant" && m.contains("tool_calls") &&
                m["tool_calls"].is_array()) {
                cm.tool_calls.reserve(cm.tool_calls.size() + m["tool_calls"].size());
                arguments.reserve(arguments.size() + m["tool_calls"].size());
                for (const auto & tc : m["tool_calls"]) {
                    if (!tc.is_object()) continue;
                    const auto function = tc.find("function");
                    if (function == tc.end() || !function->is_object()) continue;
                    append_call(cm, arguments, *function, tc.value("id", ""), "arguments");
                }
            }
            if (anthropic_blocks && cm.role == "assistant") {
                for (const auto & part : m["content"]) {
                    if (!part.is_object() || part.value("type", "") != "tool_use") continue;
                    append_call(cm, arguments, part, part.value("id", ""), "input");
                }
            }

            if (!response_assistant) finish_calls(cm, arguments);
            if (!cm.tool_calls_replayed) {
                std::string text;
                if (m.contains("content") && m["content"].is_string()) {
                    text = m["content"].get<std::string>();
                } else if (m.contains("content") && m["content"].is_array()) {
                    for (const auto & part : m["content"]) {
                        if (!part.is_object()) continue;
                        const std::string ptype = part.value("type", "");
                        if (ptype == "text" || ptype == "input_text" || ptype == "output_text") {
                            text += part.value("text", "");
                        }
                    }
                }
                if (!text.empty()) {
                    if (cm.content.empty()) {
                        cm.content = std::move(text);
                    } else {
                        cm.content += "\n\n";
                        cm.content += text;
                    }
                }
            }
            if (cm.role == "tool" || cm.role == "function") {
                cm.tool_call_id = m.value("tool_call_id", "");
                cm.tool_name = m.value("name", "");
            }
            if (response_assistant) continue;

            if (format == ApiFormat::RESPONSES && (cm.role == "system" || cm.role == "developer")) {
                system_parts.push_back(std::move(cm.content));
            } else {
                chat_msgs.push_back(std::move(cm));
            }
        }
        flush_response_turn();
    } else if (messages.is_string()) {
        chat_msgs.push_back({"user", messages.get<std::string>()});
    }

    if (!system_parts.empty()) {
        std::string merged_system;
        for (size_t i = 0; i < system_parts.size(); i++) {
            if (i) merged_system += "\n\n";
            merged_system += system_parts[i];
        }
        chat_msgs.insert(chat_msgs.begin(), {"system", std::move(merged_system)});
    }

    // Resolve names only after the message vector is stable; views avoid
    // copying IDs/names while associating out-of-order tool results.
    std::unordered_map<std::string_view, std::string_view> tool_names;
    for (auto & message : chat_msgs) {
        for (const auto & call : message.tool_calls) {
            if (!call.id.empty()) tool_names[call.id] = call.name;
        }
        if (message.tool_name.empty() && !message.tool_call_id.empty()) {
            const auto name = tool_names.find(message.tool_call_id);
            if (name != tool_names.end()) message.tool_name = name->second;
        }
    }
    return chat_msgs;
}

// ─── Disk-cache identity salt ───────────────────────────────────────────
// Compute a 16-byte salt from inputs that affect KV cache validity:
//   model path + stat(size + mtime)  [covers rope/yarn — GGUF-derived],
//   max_ctx, sha1(chat_template_src), and the effective K-rotation basis
//   (LUCE_KV_ROTATE resolves against the K-cache type; a cache written
//   rotated must never be adopted by an un-rotated session or vice versa).
// Also covered: the KV cache types (a q8_0 snapshot must never be adopted by
// a q4_0 session), the draft model (it decides the captured target-feature
// layers), and a content fingerprint of the model file (SHA-1 of its first
// and last MiB plus its size; hashing the whole multi-GB file at every
// startup would cost tens of seconds, and a re-quantized or replaced file
// changes its header and tail).
// Returns all-zeroes if model_path is empty (back-compat / disk disabled).
static void append_file_identity(std::vector<uint8_t> & buf,
                                 const std::string & path, bool content) {
    uint32_t plen = (uint32_t)path.size();
    buf.insert(buf.end(), (uint8_t *)&plen, (uint8_t *)&plen + 4);
    buf.insert(buf.end(), path.begin(), path.end());
    if (path.empty()) return;
    struct stat st{};
    int64_t file_size = 0, file_mtime = 0;
    if (::stat(path.c_str(), &st) == 0) {
        file_size  = (int64_t)st.st_size;
        file_mtime = (int64_t)st.st_mtime;
    }
    buf.insert(buf.end(), (uint8_t *)&file_size,  (uint8_t *)&file_size  + 8);
    buf.insert(buf.end(), (uint8_t *)&file_mtime, (uint8_t *)&file_mtime + 8);
    if (!content) return;
    uint8_t digest[20] = {};
    std::ifstream in(path, std::ios::binary);
    if (in) {
        constexpr int64_t kChunk = 1 << 20;
        std::vector<uint8_t> data;
        auto read_at = [&](int64_t off, int64_t len) {
            if (len <= 0) return;
            in.clear();
            in.seekg((std::streamoff)off);
            const size_t start = data.size();
            data.resize(start + (size_t)len);
            in.read((char *)data.data() + start, (std::streamsize)len);
            data.resize(start + (size_t)std::max<std::streamsize>(0, in.gcount()));
        };
        read_at(0, std::min(kChunk, file_size));
        if (file_size > kChunk) {
            const int64_t tail = std::max(kChunk, file_size - kChunk);
            read_at(tail, file_size - tail);
        }
        sha1_hash(data.data(), data.size(), digest);
    }
    buf.insert(buf.end(), digest, digest + 20);
}

static uint64_t disk_cache_ttl_seconds(double hours) {
    if (!(hours > 0.0)) return 0;  // 0 (or invalid) = never expire
    const double secs = hours * 3600.0;
    return secs < 1.0 ? 1 : (uint64_t)std::llround(secs);
}

static std::array<uint8_t, 16> compute_disk_cache_salt(const ServerConfig & cfg) {
    std::array<uint8_t, 16> salt{};
    if (cfg.model_path.empty()) return salt;

    const std::string & path = cfg.model_path;
    struct stat st{};
    int64_t file_size  = 0;
    int64_t file_mtime = 0;
    if (::stat(path.c_str(), &st) == 0) {
        file_size  = (int64_t)st.st_size;
        file_mtime = (int64_t)st.st_mtime;
    } else {
        std::fprintf(stderr, "[disk-cache] salt: stat(%s) failed — path-only fingerprint\n",
                     path.c_str());
    }

    // Hash chat_template_src separately (can be large; fold as digest).
    uint8_t tmpl_digest[20] = {};
    sha1_hash(cfg.chat_template_src.data(), cfg.chat_template_src.size(), tmpl_digest);

    // Serialization: path_len(4) + path + file_size(8) + file_mtime(8) + max_ctx(4) + tmpl_digest(20) + k_rotated(1).
    std::vector<uint8_t> buf;
    uint32_t plen = (uint32_t)path.size();
    buf.insert(buf.end(), (uint8_t *)&plen,        (uint8_t *)&plen        + 4);
    buf.insert(buf.end(), (uint8_t *)path.data(),  (uint8_t *)path.data()  + path.size());
    buf.insert(buf.end(), (uint8_t *)&file_size,   (uint8_t *)&file_size   + 8);
    buf.insert(buf.end(), (uint8_t *)&file_mtime,  (uint8_t *)&file_mtime  + 8);
    int32_t mc = (int32_t)cfg.max_ctx;
    buf.insert(buf.end(), (uint8_t *)&mc,          (uint8_t *)&mc          + 4);
    buf.insert(buf.end(), tmpl_digest, tmpl_digest + 20);
    uint8_t k_rotated = dflash_kv_k_rotation_enabled(cfg.kv_cache_k) ? 1 : 0;
    buf.push_back(k_rotated);
    // Snapshot bytes are stored in the cache's own type.
    for (const std::string * kv : {&cfg.kv_cache_k, &cfg.kv_cache_v}) {
        uint32_t n = (uint32_t)kv->size();
        buf.insert(buf.end(), (uint8_t *)&n, (uint8_t *)&n + 4);
        buf.insert(buf.end(), kv->begin(), kv->end());
    }
    append_file_identity(buf, cfg.model_path, /*content=*/true);
    append_file_identity(buf, cfg.draft_path, /*content=*/false);

    uint8_t digest[20];
    sha1_hash(buf.data(), buf.size(), digest);
    std::memcpy(salt.data(), digest, 16);
    return salt;
}

// One source for model discovery in both single- and multi-model serving.
static json model_list(const ServerConfig & config, bool codex_schema) {
    // Codex sends ?client_version= — serve the Codex-specific schema.
    if (codex_schema) {
        json codex_models = {
            {"models", json::array({
                {{"slug", config.model_name},
                 {"display_name", config.model_name},
                 {"description", "Local DFlash speculative-decoding server"},
                 {"default_reasoning_level", "low"},
                 // Spec §4.2: every tier activates the phase-1 envelope;
                 // the difference is the budget cap selected from the
                 // model card's effort_tiers. Descriptions surface the
                 // resolved cap so clients can pick a tier purposefully.
                 {"supported_reasoning_levels", json::array({
                     {{"effort", "low"},
                      {"description", "Phase-1 budget at the model card's low tier ("
                                      + std::to_string(config.effort_tiers.low)
                                      + " tokens)"}},
                     {{"effort", "medium"},
                      {"description", "Phase-1 budget at the model card's medium tier ("
                                      + std::to_string(config.effort_tiers.medium)
                                      + " tokens)"}},
                     {{"effort", "high"},
                      {"description", "Phase-1 budget at the model card's standard recommendation ("
                                      + std::to_string(config.effort_tiers.high)
                                      + " tokens)"}},
                     {{"effort", "x-high"},
                      {"description", "Phase-1 budget between high and the complex-problem ceiling ("
                                      + std::to_string(config.effort_tiers.x_high)
                                      + " tokens)"}},
                     {{"effort", "max"},
                      {"description", "Phase-1 budget at the model card's complex-problem ceiling ("
                                      + std::to_string(config.effort_tiers.max)
                                      + " tokens)"}},
                 })},
                 {"shell_type", "shell_command"},
                 {"visibility", "list"},
                 {"supported_in_api", true},
                 {"priority", 1},
                 {"context_window", config.max_ctx},
                 {"supports_reasoning_summaries", false},
                 {"supports_parallel_tool_calls", false}}
            })}
        };
        return codex_models;
    }
    json models = {
        {"object", "list"},
        {"data", json::array({
            {{"id", config.model_name},
             {"object", "model"},
             {"owned_by", "luce"},
             {"created", 1700000000},
             {"context_length", config.max_ctx},
             {"max_context_length", config.max_ctx}}
        })}
    };
    return models;
}

// ─── HttpServer ─────────────────────────────────────────────────────────

// Memory the auto budget takes a quarter of: what is available once the
// model has loaded (a container's cgroup v2 limit included; Windows too),
// else total physical memory where the probe cannot read it.
static size_t budgetable_memory_bytes() {
    if (const auto available = available_kv_offload_memory()) return *available;
#if !defined(_WIN32)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) return (size_t)pages * (size_t)page_size;
#endif
    return 0;
}

size_t auto_concurrent_prefix_budget(size_t per_checkpoint, int slots, size_t memory) {
    // Each decode slot keeps its conversation's restore point and the capture
    // in flight, plus one shared system/tools head: 2 x slots + 1 checkpoints
    // at --max-ctx. Never less than the former fixed 4096 MiB; above that,
    // never more than a quarter of the memory available (when known).
    const size_t floor = ServerConfig::kConcurrentPrefixBudgetFloor;
    size_t bytes = std::max(floor, per_checkpoint * (2 * (size_t)std::max(1, slots) + 1));
    if (memory > 0) bytes = std::min(bytes, std::max(floor, memory / 4));
    return bytes;
}

size_t HttpServer::concurrent_prefix_budget(size_t per_checkpoint, int slots) const {
    return auto_concurrent_prefix_budget(per_checkpoint, slots, budgetable_memory_bytes());
}

PrefixCacheBudget resolve_prefix_cache_budget(const ServerConfig & config,
                                              const ModelBackend & backend) {
    PrefixCacheBudget out;
    // No prefix cache, nothing to bound (e.g. single-sequence paged serving).
    if (config.prefix_cache_cap <= 0) return out;
    if (config.concurrent_prefix_cache) {
        out.automatic = config.concurrent_prefix_cache_max_bytes ==
            ServerConfig::kPrefixCacheBudgetAuto;
        // Auto starts at the old 4 GiB default; the scheduler resizes it once
        // the batch engine, which owns paged checkpoints, can size one
        // (concurrent_prefix_budget below).
        out.bytes = out.automatic ? ServerConfig::kConcurrentPrefixBudgetFloor
                                  : config.concurrent_prefix_cache_max_bytes;
        return out;
    }
    out.automatic =
        config.prefix_cache_max_bytes == ServerConfig::kPrefixCacheBudgetAuto;
    const size_t full_context_bytes =
        backend.snapshot_bytes_estimate(std::numeric_limits<int>::max());
    out.sized = full_context_bytes > 0;
    if (!out.sized) {
        // The cache cannot check a capture it cannot size, so a requested
        // limit would never apply. Auto stays unlimited, as before.
        if (!out.automatic && config.prefix_cache_max_bytes != 0) {
            out.error = "--prefix-cache-max-mib needs a backend that can size "
                        "prefix snapshots (Qwen or DeepSeek4 single-sequence "
                        "serving); this one cannot. Use --prefix-cache-slots "
                        "to bound memory.";
        }
        return out;
    }
    if (!out.automatic) {
        out.bytes = config.prefix_cache_max_bytes;
        return out;
    }
    // Room for three snapshots at the backend's full context: the
    // system/tools head, the conversation's restore point, and the capture
    // in flight. Never more than a quarter of the memory available.
    out.bytes = full_context_bytes * 3;
    const size_t memory = budgetable_memory_bytes();
    if (memory > 0) out.bytes = std::min(out.bytes, memory / 4);
    // A backend that sized its snapshot allowance against device memory (a
    // split model's smaller target GPU) refuses captures past it; evict first.
    out.bytes = std::min(out.bytes, backend.snapshot_allowance_bytes());
    return out;
}

HttpServer::HttpServer(luce::engine::LuceEngine & engine,
                       Tokenizer & tokenizer,
                       const ServerConfig & config)
    : engine_(engine)
    , backend_(engine.backend())
    , tokenizer_(tokenizer)
    , config_(config)
    , chat_format_(ChatFormat::QWEN3)  // default, overridden by arch
    , prefix_cache_(config.prefix_cache_cap, tokenizer,
          resolve_prefix_cache_budget(config, engine.backend()).bytes)
    , disk_cache_({config.disk_cache_dir,
                   config.disk_cache_budget_mb * (size_t)(1024 * 1024),
                   config.disk_cache_min_tokens,
                   config.disk_cache_continued_interval,
                   config.disk_cache_cold_max_tokens,
                   disk_cache_ttl_seconds(config.disk_cache_ttl_hours)}, backend_)
{
    config_.image_input_enabled = backend_.supports_images() &&
        config_.pflash_upstream_base.empty() &&
        (!backend_.seq_engine() || backend_.seq_engine()->supports_images());
    if (backend_.supports_images() && !config_.image_input_enabled) {
        std::fprintf(stderr,
            "[server] WARNING: a vision projector is loaded but image input is off: it is "
            "not available with upstream forwarding or concurrent sequence scheduling\n");
    }
    #ifdef LUCE_HAS_CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    #endif
    prefix_cache_.init_full_cache(config.prefill_cache_cap);
    // Commits prune the snapshots they supersede: single-sequence serving in
    // trim_snapshots_after_commit, the concurrent scheduler as each capture lands.
    prefix_cache_.set_prunes_superseded(true);
    // Fold model+config identity into the layout fingerprint BEFORE init()
    // so compute_layout_id sees it on every learn/verify call. Prevents stale
    // KV hits when the server restarts over the same --kv-cache-dir with a
    // different model, max_ctx, or chat_template (gemma4 ↔ qwen3.6, etc.).
    if (!disk_cache_.disabled()) {
        disk_cache_.set_identity_salt(compute_disk_cache_salt(config));
    }
    disk_cache_.init();
    // Report the effective state after initialization failure or a directory
    // lock conflict; serving continues with the ordinary RAM cache.
    if (disk_cache_.disabled()) config_.disk_cache_dir.clear();
    restore_replay_blobs();
    status_html_path_ = resolve_status_html();

    // PFLASH_UNIFIED=1: read once here; a bad tool list or an unsupported variant fails startup.
    // Defaults under the flag: every tool's output may be paged (PFLASH_REGION_TOOLS unset = "*"; set it empty for
    // none) and recall is gated at theta 0.075 (PFLASH_RECALL_GATE; "off" = ungated recall).
    if (const char * unified = std::getenv("PFLASH_UNIFIED"); unified && std::strcmp(unified, "1") == 0) {
        const auto fail = [] (const std::string & message) {
            std::fprintf(stderr, "[pflash-unified] ERROR: %s\n", message.c_str());
            std::fflush(stderr);
            throw std::runtime_error(message);
        };
        pflash_unified_ = true;
        std::string tools_error;
        const char * tools = std::getenv("PFLASH_REGION_TOOLS");
        if (!pflash_unified::parse_tool_list(tools ? tools : "*", pflash_region_tools_, tools_error)) {
            fail(tools_error);
        }
        for (const char * arm : {"PFLASH_UNIFIED_INPLACE", "PFLASH_UNIFIED_DEFRAG"}) {
            if (const char * value = std::getenv(arm); value && *value) {
                fail(std::string(arm) + " is not supported");
            }
        }
        if (!luce::pflash::has_pflash_selection_environment()) {
            std::fprintf(stderr, "[pflash-unified] no strict-selection environment: every request is served by the "
                                 "chat view, as without PFLASH_UNIFIED\n");
        } else {
            luce::pflash::PFlashSelectionConfig selection;
            std::string selection_error;
            if (luce::pflash::resolve_pflash_selection(1 << 16, 32, selection, selection_error) &&
                !http_detail::pflash_selection_gives_lifts(selection)) {
                std::fprintf(stderr, "[pflash-unified] scorer %s, strategy %s give no per-candidate scores: every "
                                     "request is served by the chat view, as without PFLASH_UNIFIED\n",
                             luce::pflash::pflash_scorer_name(selection.scorer),
                             luce::pflash::pflash_select_strategy_name(selection.strategy));
            }
        }
        const char * recall = std::getenv("PFLASH_RECALL");
        pflash_unified_recall_ = !(recall && std::strcmp(recall, "0") == 0);
        // The default theta was frozen from the dev replay (largest theta whose served gold stays within 3 of
        // ungated recall); see the recall-gate notes in server/README.md.
        pflash_recall_gate_ = 0.075;
        bool gate_given = false;
        if (const char * gate = std::getenv("PFLASH_RECALL_GATE"); gate && *gate) {
            if (std::strcmp(gate, "off") == 0) {
                pflash_recall_gate_ = -1.0;
            } else {
                gate_given = true;
                char * end = nullptr;
                const double theta = std::strtod(gate, &end);
                if (end == gate || *end != '\0' || !std::isfinite(theta) || theta <= 0.0 || theta >= 1.0) {
                    fail(std::string("PFLASH_RECALL_GATE must be a number in (0, 1) or 'off', got '") + gate + "'");
                }
                pflash_recall_gate_ = theta;
            }
        }
        if (const char * source = std::getenv("PFLASH_RECALL_SOURCE"); source && *source) {
            fail("PFLASH_RECALL_SOURCE was removed: recall takes the unheld units, gated by PFLASH_RECALL_GATE");
        }
        if (const char * trace = std::getenv("PFLASH_RECALL_TRACE_UNITS"); trace && *trace) {
            if (std::strcmp(trace, "1") != 0 && std::strcmp(trace, "0") != 0) {
                fail(std::string("PFLASH_RECALL_TRACE_UNITS must be 0 or 1, got '") + trace + "'");
            }
            pflash_recall_trace_units_ = std::strcmp(trace, "1") == 0;
        }
        if (gate_given && !pflash_unified_recall_) {
            fail("PFLASH_RECALL_GATE needs recall on (PFLASH_RECALL=0 is set)");
        }
        if (!pflash_unified_recall_) pflash_recall_gate_ = -1.0;
        // Knobs of the legacy chat view and tool-session rules: they apply only to the requests the unified view
        // hands back to that path (apply_unified_view), named once.
        for (const char * legacy : {"PFLASH_AGENT_PASTE", "PFLASH_ARRIVE", "PFLASH_CHAT_RECALL",
                                    "PFLASH_CHAT_COMPRESS_NEW_TOKENS", "PFLASH_CHAT_REBUILD_MIN_TOKENS",
                                    "PFLASH_CHAT_REBUILD_RATIO", "PFLASH_SELECT_PARAGRAPH_JOIN"}) {
            if (std::getenv(legacy)) {
                std::fprintf(stderr, "[pflash-unified] %s applies only to requests handed back to the chat view\n",
                             legacy);
            }
        }
        const auto env = [] (const char * name) {
            const char * value = std::getenv(name);
            return std::string(value ? value : "");
        };
        pflash_unified_config_ = "unified-v1|tools=" + pflash_region_tools_.identity +
            "|recall=" + (pflash_unified_recall_ ? "1" : "0") +
            (pflash_recall_gate_ > 0.0 ? "|recall_gate=" + std::to_string(pflash_recall_gate_) : std::string()) +
            "|threshold=" + std::to_string(config_.pflash_threshold) +
            "|drafter=" + config_.pflash_drafter_path +
            "|head=" + env("PFLASH_SCORING_HEAD_GGUF") + "|probe=" + env("PFLASH_SEGMENT_PROBE_GGUF") +
            "|request_probe=" + env("PFLASH_REQUEST_PROBE_GGUF") +
            "|curve=";
        for (int tokens : {16384, 32768, 65536, 131072}) {
            pflash_unified_config_ += std::to_string(pflash_keep_ratio(config_, tokens)) + ",";
        }
        char gate_text[32] = "off";
        if (pflash_recall_gate_ > 0.0) std::snprintf(gate_text, sizeof gate_text, "%g", pflash_recall_gate_);
        std::fprintf(stderr, "[pflash-unified] on: region tools=%s recall=%d recall_gate=%s\n",
                     pflash_region_tools_.identity.empty() ? "(none)" : pflash_region_tools_.identity.c_str(),
                     (int) pflash_unified_recall_, gate_text);
        // With the disk session cache on, the views of shortened conversations are kept on disk too: written when
        // they leave the RAM store and at shutdown, read on a RAM miss, so neither a restart nor an eviction turns a
        // shortened conversation cold. Their identity is the disk cache's (model, template, context); the directory
        // starts with a dot, so the KV cache's scan skips it.
        if (!disk_cache_.disabled()) {
            std::string identity;
            for (uint8_t b : compute_disk_cache_salt(config_)) {
                char hex_byte[3];
                std::snprintf(hex_byte, sizeof hex_byte, "%02x", b);
                identity += hex_byte;
            }
            pflash_unified_disk_views_.init(config_.disk_cache_dir + "/.pflash-views", identity,
                                            disk_cache_ttl_seconds(config_.disk_cache_ttl_hours), 256);
            // A new conversation must not take the lineage of one on disk (forget_lineage would drop that one).
            pflash_unified_lineage_ = pflash_unified_disk_views_.max_lineage();
        }
        std::fflush(stderr);
    }

    // PPP env overrides (operator-facing; no CLI flags required).
    auto env_truthy = [](const char * v) -> bool {
        if (!v || !*v) return false;
        return !(v[0] == '0' && v[1] == '\0') &&
               !(v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
    };
    if (const char * e = std::getenv("LUCE_PPP")) {
        config_.ppp_enabled = env_truthy(e);
    }
    if (const char * e = std::getenv("LUCE_PPP_REARRANGE")) {
        config_.ppp_rearrange = env_truthy(e);
    }
    if (const char * e = std::getenv("LUCE_PPP_LCP_WINDOW")) {
        const int n = std::atoi(e);
        if (n > 0) config_.ppp_lcp_window = n;
    }
    if (const char * e = std::getenv("LUCE_PPP_MIN_PIN_TOKENS")) {
        const int n = std::atoi(e);
        if (n > 0) config_.ppp_min_pin_tokens = n;
    }
    if (const char * e = std::getenv("LUCE_PPP_MAX_EPHEMERAL")) {
        const int n = std::atoi(e);
        if (n > 0) config_.ppp_max_ephemeral_tokens = n;
    }
    std::fprintf(stderr,
        "[ppp] enabled=%d rearrange=%d lcp_window=%d min_pin=%d max_ephemeral=%d\n",
        (int)config_.ppp_enabled, (int)config_.ppp_rearrange,
        config_.ppp_lcp_window, config_.ppp_min_pin_tokens,
        config_.ppp_max_ephemeral_tokens);
}

// Resolve path to share/status.html at startup.
std::string HttpServer::resolve_status_html() {
    // 1. LUCE_SHARE_DIR env var
    if (const char * dir = std::getenv("LUCE_SHARE_DIR")) {
        std::string path = std::string(dir) + "/status.html";
        struct stat st;
        if (::stat(path.c_str(), &st) == 0) return path;
    }
    // 2. share/ relative to exe path (build dir or installed prefix)
    {
    std::string exe_dir;
#if defined(_WIN32)
    char exe_buf[MAX_PATH] = {};
    DWORD n = GetModuleFileNameA(nullptr, exe_buf, sizeof(exe_buf));
    if (n > 0 && n < sizeof(exe_buf)) {
        exe_dir = std::string(exe_buf, n);
        auto slash = exe_dir.find_last_of("/\\");
        if (slash != std::string::npos) exe_dir = exe_dir.substr(0, slash);
    }
#else
    char exe_buf[1024] = {};
    ssize_t len = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    if (len > 0) {
        exe_buf[len] = '\0';
        exe_dir = exe_buf;
        auto slash = exe_dir.rfind('/');
        if (slash != std::string::npos) exe_dir = exe_dir.substr(0, slash);
    }
#endif
    if (!exe_dir.empty()) {
            // 2a. <exe_dir>/share/status.html  (build directory layout)
            {
                std::string path = exe_dir + "/share/status.html";
                struct stat st;
                if (::stat(path.c_str(), &st) == 0) return path;
            }
            // 2b. <exe_dir>/../share/status.html  (installed prefix layout)
            {
                std::string path = exe_dir + "/../share/status.html";
                struct stat st;
                if (::stat(path.c_str(), &st) == 0) return path;
            }
        }
    }
    // 3. ./share/status.html (development)
    {
        struct stat st;
        if (::stat("share/status.html", &st) == 0) return "share/status.html";
    }
    return {};
}

// Send data to an SSE client fd with a short (1s) timeout to avoid stalling
// the inference worker. Returns false if the send fails or times out.
static bool sse_try_send(SocketHandle fd, const void * data, size_t len) {
    const char * p = static_cast<const char *>(data);
    size_t sent = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (sent < len) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) return false;

        struct pollfd pfd = {fd, POLLOUT, 0};
        int ret;
        do {
            ret = poll(&pfd, 1, (int)(remaining < 50 ? remaining : 50));
        } while (ret < 0 && sock_is_eintr(sock_errno()));
        if (ret < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
        if (ret == 0) continue;

        ssize_t n = ::send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (sock_is_eintr(sock_errno()) || sock_is_eagain(sock_errno())) continue;
            return false;
        }
        sent += n;
    }
    return true;
}

// Broadcast current status as SSE event to all connected /status/events clients.
void HttpServer::broadcast_status() {
    std::string event = status_.to_sse_event();
    std::lock_guard<std::mutex> lk(sse_mu_);
    std::vector<SocketHandle> dead;
    for (SocketHandle fd : sse_fds_) {
        if (!sse_try_send(fd, event.data(), event.size())) {
            dead.push_back(fd);
        }
    }
    for (SocketHandle fd : dead) {
        socket_close(fd);
        sse_fds_.erase(std::remove(sse_fds_.begin(), sse_fds_.end(), fd),
                       sse_fds_.end());
    }
}

// Broadcast a token text delta as an incremental SSE event.
void HttpServer::broadcast_token(const std::string & text) {
    // Token text may contain incomplete UTF-8 (tokens can split multi-byte
    // codepoints). Manually build the SSE payload with json string escaping
    // that replaces invalid UTF-8 with U+FFFD instead of throwing.
    json j;
    j["text"] = text;
    std::string event = "event: token\ndata: " +
        j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    std::lock_guard<std::mutex> lk(sse_mu_);
    std::vector<SocketHandle> dead;
    for (SocketHandle fd : sse_fds_) {
        if (!sse_try_send(fd, event.data(), event.size())) {
            dead.push_back(fd);
        }
    }
    for (SocketHandle fd : dead) {
        socket_close(fd);
        sse_fds_.erase(std::remove(sse_fds_.begin(), sse_fds_.end(), fd),
                       sse_fds_.end());
    }
}

// Send an SSE comment as a heartbeat to detect disconnected clients when idle.
// Uses non-blocking sends to avoid stalling the worker thread on slow clients.
void HttpServer::sse_heartbeat() {
    static const char ping[] = ":heartbeat\n\n";
    std::lock_guard<std::mutex> lk(sse_mu_);
    std::vector<SocketHandle> dead;
    for (SocketHandle fd : sse_fds_) {
        // Non-blocking send: if the socket buffer can't accept 12 bytes
        // immediately, the client is too far behind — treat as dead.
        ssize_t n = ::send(fd, ping, sizeof(ping) - 1, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n <= 0) {
            dead.push_back(fd);
        }
    }
    for (SocketHandle fd : dead) {
        socket_close(fd);
        sse_fds_.erase(std::remove(sse_fds_.begin(), sse_fds_.end(), fd),
                       sse_fds_.end());
    }
}

HttpServer::~HttpServer() {
    shutdown();
    #ifdef LUCE_HAS_CURL
    curl_global_cleanup();
    #endif
}

void HttpServer::shutdown() {
    // Signal worker and accept loop to stop.
    stopping_.store(true);
    queue_cv_.notify_all();
    if (socket_is_valid(listen_fd_)) {
        socket_close(listen_fd_);
        listen_fd_ = kInvalidSocket;
    }
    engine_.stop_serving();

    // Close SSE client connections.
    {
        std::lock_guard<std::mutex> lk(sse_mu_);
        for (SocketHandle fd : sse_fds_) socket_close(fd);
        sse_fds_.clear();
    }

    // Drain any pending jobs.
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        while (queue_head_) {
            ServerJob * j = queue_head_;
            queue_head_ = j->next;
            j->next = nullptr;
            std::lock_guard<std::mutex> jlk(j->mu);
            j->done = true;
            j->cv.notify_one();
        }
        queue_tail_ = nullptr;
    }

    // Graceful shutdown is one last eviction: write each conversation's
    // newest snapshot and the heads. Safe without locking — the worker is
    // joined. A crash or kill -9 loses what was only in RAM.
    flush_sessions_to_disk("shutdown");
    slot_tokens_.clear();
    slot_aux_.clear();
}

bool HttpServer::start_worker() {
    // LuceEngine owns the serving thread: a backend-provided sequence engine
    // replaces the one-request worker with the concurrent scheduler.
    // Upstream forwarding stays on the classic path even when the local
    // backend exposes an engine.
    luce::engine::LuceEngine::ServingLoops loops;
    loops.serial = [this]() { worker_loop(); };
    loops.concurrent =
        [this](SeqEngine & engine) { scheduler_loop(engine); };
    loops.request_stop = [this]() {
        request_stop();
        queue_cv_.notify_all();
    };
    if (!engine_.start_serving(
            std::move(loops), config_.pflash_upstream_base.empty())) {
        std::fprintf(stderr, "[server] failed to start LuceEngine\n");
        return false;
    }
    return true;
}

int HttpServer::run(const std::vector<HttpServer *> & models) {
    if (!models.empty()) {
        if (models.size() < 2 || models.front() != this) {
            std::fprintf(stderr, "[server] model routing requires this listener first and at least two models\n");
            return 2;
        }
        std::unordered_set<std::string> names;
        std::unordered_set<ModelBackend *> backends;
        for (HttpServer * model : models) {
            SeqEngine * engine = model ? model->backend_.seq_engine() : nullptr;
            if (!model || (engine && engine->slot_count() < 1) ||
                model->engine_.is_serving() ||
                !model->config_.pflash_upstream_base.empty() ||
                model->config_.model_name.empty() || model->config_.model_name == "auto" ||
                !names.insert(model->config_.model_name).second ||
                !backends.insert(&model->backend_).second) {
                std::fprintf(stderr, "[server] model routing requires distinct local backends and unique names other than auto\n");
                return 2;
            }
        }
        for (HttpServer * model : models) {
            SeqEngine * engine = model->backend_.seq_engine();
            models_.push_back({model, engine ? engine->slot_count() : 1});
            // CORS belongs to the one listener even when a peer formats output.
            model->config_.enable_cors = config_.enable_cors;
        }
    }

    // Every model is loaded and no worker has started: resolve one shared
    // host-memory allowance without granting each model the whole machine.
    const std::vector<HttpServer *> budget_models = models.empty()
        ? std::vector<HttpServer *>{this} : models;
    size_t explicit_bytes = 0, automatic_models = 0;
    for (auto * model : budget_models) {
        const size_t requested = model->config_.decode_kv_offload_bytes;
        auto * engine = model->backend_.seq_engine();
        if (requested == luce::common::kAutoKvOffloadBytes) {
            if (engine && engine->slot_count() > 1 && engine->kv_offload_capacity()) ++automatic_models;
        } else {
            explicit_bytes += std::min(requested,
                luce::common::kAutoKvOffloadBytes - explicit_bytes);
        }
    }
    const size_t available = automatic_models
        ? luce::common::available_kv_offload_memory().value_or(0) : 0;
    for (auto * model : budget_models) {
        auto & budget = model->config_.decode_kv_offload_bytes;
        if (budget != luce::common::kAutoKvOffloadBytes) continue;
        auto * engine = model->backend_.seq_engine();
        budget = engine && engine->slot_count() > 1
            ? luce::common::auto_kv_offload_budget(engine->kv_offload_capacity(),
                available, explicit_bytes, automatic_models) : 0;
        std::fprintf(stderr, "[server] model %s automatic decode KV offload budget: %zu bytes\n",
                     model->config_.model_name.c_str(), budget);
    }

#if !defined(_WIN32)
    // Ignore SIGPIPE so send() returns EPIPE instead of killing the process.
    signal(SIGPIPE, SIG_IGN);
#else
    // Initialise Winsock — required before any socket call on Windows.
    WSADATA wsa_data;
    const int wsa_error = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (wsa_error != 0) {
        std::fprintf(stderr, "[server] WSAStartup() failed: %d\n", wsa_error);
        return 1;
    }
#endif

    // Create listen socket.
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (!socket_is_valid(listen_fd_)) {
        std::fprintf(stderr, "[server] socket() failed: %s\n", sock_strerror());
        return 1;
    }

    int yes = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, SETSOCKOPT_CAST &yes, sizeof(yes));

    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)config_.port);
    if (inet_pton(AF_INET, config_.host.c_str(), &sa.sin_addr) != 1) {
        std::fprintf(stderr, "[server] invalid host address: %s\n", config_.host.c_str());
        socket_close(listen_fd_);
        listen_fd_ = kInvalidSocket;
        return 1;
    }

    if (bind(listen_fd_, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        std::fprintf(stderr, "[server] bind(%s:%d) failed: %s\n",
                     config_.host.c_str(), config_.port, sock_strerror());
        socket_close(listen_fd_);
        listen_fd_ = kInvalidSocket;
        return 1;
    }

    if (listen(listen_fd_, 128) < 0) {
        std::fprintf(stderr, "[server] listen() failed: %s\n", sock_strerror());
        socket_close(listen_fd_);
        listen_fd_ = kInvalidSocket;
        return 1;
    }

    // Non-blocking listen socket so the accept loop polls stopping_ on a short
    // timeout. This guarantees the loop exits on SIGTERM/SIGINT regardless of
    // which thread the signal handler runs on (it only sets the atomic flag).
    {
        int fl = sock_get_flags(listen_fd_);
        if (fl < 0) { /* Windows: ioctlsocket sets non-block directly */ }
        sock_set_nonblock(listen_fd_);
        if (false) {
            std::fprintf(stderr, "[server] fcntl(O_NONBLOCK) failed: %s\n", "n/a");
            socket_close(listen_fd_);
            listen_fd_ = kInvalidSocket;
            return 1;
        }
    }

    std::fprintf(stderr, "[server] listening on http://%s:%d\n",
                 config_.host.c_str(), config_.port);

    if (models_.empty()) {
        if (!start_worker()) return 1;
    } else {
        for (size_t i = 0; i < models_.size(); ++i) {
            if (!models_[i].server->start_worker()) {
                for (size_t j = 0; j < i; ++j)
                    models_[j].server->engine_.stop_serving();
                return 1;
            }
        }
    }

    // Accept loop.
    while (!stopping_.load()) {
        struct pollfd pfd{listen_fd_, POLLIN, 0};
        int pr = poll(&pfd, 1, 200 /* ms */);
        if (pr <= 0) {
            // 0 = timeout (re-check stopping_); <0 with EINTR = signal. Both loop.
            if (pr < 0 && !sock_is_eintr(sock_errno())) {
                std::fprintf(stderr, "[server] poll() error: %s\n", sock_strerror());
            }
            continue;
        }

        struct sockaddr_in client_sa{};
        socklen_t client_len = sizeof(client_sa);
        SocketHandle client_fd = accept(
            listen_fd_, (struct sockaddr *)&client_sa, &client_len);
        if (!socket_is_valid(client_fd)) {
            if (stopping_.load()) break;
            if (sock_is_eintr(sock_errno()) || sock_is_eagain(sock_errno())) continue;
            std::fprintf(stderr, "[server] accept() error: %s\n", sock_strerror());
            continue;
        }

        // Disable Nagle for low-latency SSE streaming.
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, SETSOCKOPT_CAST &flag, sizeof(flag));

        // Spawn client thread (detached — client_main owns the fd).
        active_clients_.fetch_add(1);
        std::thread([this, client_fd]() {
            handle_client(client_fd);
            // Publish completion under the same lock used by run()'s wait.
            // Once it observes zero, no detached handler may touch this object.
            std::lock_guard<std::mutex> lk(clients_mu_);
            if (active_clients_.fetch_sub(1) == 1) clients_cv_.notify_all();
        }).detach();
    }

    auto stop_worker = [](HttpServer * server) {
        server->request_stop();
        server->queue_cv_.notify_all();
    };
    routing_cv_.notify_all();
    stop_worker(this);
    for (auto & model : models_) stop_worker(model.server);
    engine_.stop_serving();
    for (auto & model : models_) model.server->engine_.stop_serving();

    // Every handler borrows a model context. Do not destroy those contexts
    // after an arbitrary grace period. Reads poll stopping_; workers have
    // retired all jobs and unblocked their client monitors before this wait.
    {
        std::unique_lock<std::mutex> lk(clients_mu_);
        clients_cv_.wait(lk, [this]() { return active_clients_.load() == 0; });
    }

    // Graceful shutdown is one last eviction (worker joined — no race on
    // slot_tokens_).
    flush_sessions_to_disk("shutdown");
    slot_tokens_.clear();
    slot_aux_.clear();

#if defined(_WIN32)
    // Status-event sockets remain owned until shutdown(). Keep Winsock alive
    // for their cleanup; the process reclaims the global runtime on exit.
#endif

    return 0;
}

// ─── Client thread ──────────────────────────────────────────────────────

void HttpServer::handle_client(SocketHandle fd) {
    HttpRequest hr;
    if (!read_http_request(fd, hr)) {
        send_error(fd, 400, "bad HTTP request");
        socket_close(fd);
        return;
    }

    // CORS preflight.
    if (hr.method == "OPTIONS") {
        send_response(fd, 204, "", "");
        socket_close(fd);
        return;
    }

    // Health check.
    if (hr.method == "GET" && (hr.path == "/health" || hr.path == "/")) {
        send_response(fd, 200, "application/json", "{\"status\":\"ok\"}\n");
        socket_close(fd);
        return;
    }

    if (!models_.empty() && hr.method == "GET") {
        if (hr.path == "/props" || hr.path == "/status/json") {
            json body = hr.path == "/props"
                ? build_props_body(config_, prefix_cache_, tool_memory_)
                : status_.to_json();
            if (hr.path == "/props") body["memory"] = memory_json();
            if (hr.path == "/props" && body.contains("pflash") &&
                body["pflash"].value("enabled", false)) {
                body["pflash"]["failures"] = pflash_failure_counters_json();
                if (auto keep = pflash_keep_history_json(); !keep.is_null()) {
                    body["pflash"]["keep_history"] = std::move(keep);
                }
                if (auto unified = pflash_unified_json(); !unified.is_null()) {
                    body["pflash"]["unified"] = std::move(unified);
                }
            }
            body.update(model_routing_status());
            send_response(fd, 200, "application/json", body.dump() + "\n");
            socket_close(fd);
            return;
        }
        if (hr.path == "/status" || hr.path == "/status/events") {
            send_error(fd, 404, "multi-model status is available at /status/json");
            socket_close(fd);
            return;
        }
    }

    // Introspection: server config + cache stats + arch + capabilities.
    if (hr.method == "GET" && hr.path == "/props") {
        json body = build_props_body(config_, prefix_cache_, tool_memory_);
        body["memory"] = memory_json();
        if (body.contains("pflash") && body["pflash"].value("enabled", false)) {
            body["pflash"]["failures"] = pflash_failure_counters_json();
            if (auto keep = pflash_keep_history_json(); !keep.is_null()) {
                body["pflash"]["keep_history"] = std::move(keep);
            }
            if (auto unified = pflash_unified_json(); !unified.is_null()) {
                body["pflash"]["unified"] = std::move(unified);
            }
        }
        send_response(fd, 200, "application/json", body.dump() + "\n");
        socket_close(fd);
        return;
    }

    // Status page: serve HTML file from disk.
    if (hr.method == "GET" && hr.path == "/status") {
        if (status_html_path_.empty()) {
            send_error(fd, 404,
                "status.html not found. Set LUCE_SHARE_DIR or place it in share/status.html");
            socket_close(fd);
            return;
        }
        std::ifstream ifs(status_html_path_);
        if (!ifs.is_open()) {
            send_error(fd, 500, "failed to open status.html");
            socket_close(fd);
            return;
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        send_response(fd, 200, "text/html; charset=utf-8", oss.str());
        socket_close(fd);
        return;
    }

    // Status JSON snapshot (for non-SSE clients / debugging).
    if (hr.method == "GET" && hr.path == "/status/json") {
        send_response(fd, 200, "application/json",
            status_.to_json().dump(-1, ' ', false, json::error_handler_t::replace) + "\n");
        socket_close(fd);
        return;
    }

    // Status SSE stream: hold connection open and push updates.
    if (hr.method == "GET" && hr.path == "/status/events") {
        // Send SSE headers.
        const char * headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        if (!send_all(fd, headers, std::strlen(headers))) {
            socket_close(fd);
            return;
        }
        // Send initial state immediately.
        std::string initial = status_.to_sse_event();
        send_all(fd, initial.data(), initial.size());
        // Register for future broadcasts. The fd is NOT closed here — it stays
        // open until the client disconnects (detected on next broadcast send).
        {
            std::lock_guard<std::mutex> lk(sse_mu_);
            sse_fds_.push_back(fd);
        }
        return;  // Do NOT close fd — it's now owned by the SSE broadcast loop.
    }

    // Models endpoint (including the existing Codex discovery schema).
    if (hr.method == "GET" && hr.path == "/v1/models") {
        const bool codex_schema = hr.query.find("client_version") != std::string::npos;
        json response = model_list(config_, codex_schema);
        if (!models_.empty()) {
            const char * key = codex_schema ? "models" : "data";
            for (size_t i = 0; i < models_.size(); ++i) {
                const auto & cfg = models_[i].server->config_;
                if (i != 0) response[key].push_back(model_list(cfg, codex_schema)[key][0]);
            }
        }
        send_response(fd, 200, "application/json", response.dump() + "\n");
        socket_close(fd);
        return;
    }

    // Route POST endpoints.
    if (!route_request(fd, hr)) {
        send_error(fd, 404, "unknown endpoint");
    }
    socket_close(fd);
}

// Endpoint structure is parsed once by the listener. Model defaults, templates
// and token IDs are resolved only by the selected model's handler.
bool HttpServer::route_model_request(SocketHandle fd, ParsedRequest & req,
                                     bool count_only) {
    json & body = req.raw_body;
    const std::string requested = body.value("model", std::string());
    const bool unnamed = requested.empty() || requested == "auto";
    if (count_only && unnamed) {
        send_error(fd, 400, "token counting requires an explicit model name");
        return true;
    }
    // Generation always follows operator priority. Only token counting needs
    // an explicit tokenizer, since no generating model has been selected yet.
    const bool automatic = !count_only;

    // A routing reservation owns a backend slot through retirement and output
    // draining. Engine admission, on its worker, remains the KV authority.
    struct Reservation {
        HttpServer & owner;
        RoutedModel * model;
        RoutingAdmission admission = RoutingAdmission::handled;
        ~Reservation() {
            if (!model) return;
            std::lock_guard<std::mutex> lock(owner.routing_mu_);
            --model->in_flight;
            // Busy/unfit probes released no engine capacity. Waking peers
            // here makes waiting clients repeatedly trigger each other.
            if (admission == RoutingAdmission::handled) owner.routing_cv_.notify_all();
        }
    };
    struct Waiting {
        HttpServer & owner;
        bool active = false;
        ~Waiting() {
            if (!active) return;
            std::lock_guard<std::mutex> lock(owner.routing_mu_);
            --owner.routing_waiters_;
        }
    } waiting{*this};

    std::unordered_set<HttpServer *> unfit;
    for (;;) {
        if (http_detail::inspect_peer_socket(fd) ==
                http_detail::PeerSocketState::Disconnected) return true;
        bool known = automatic;
        for (auto & model : models_) {
            if (!automatic && requested != model.server->config_.model_name) continue;
            known = true;
            if (unfit.count(model.server)) continue;
            {
                std::lock_guard<std::mutex> lock(routing_mu_);
                if (stopping_.load()) break;
                if (!count_only && model.in_flight >= model.capacity) continue;
                if (!count_only) ++model.in_flight;
                if (waiting.active) {
                    --routing_waiters_;
                    waiting.active = false;
                }
            }
            Reservation reservation{*this, count_only ? nullptr : &model};
            // Always restart from the endpoint parse, before any model-specific
            // template, tokenizer, reasoning or sampler defaults were applied.
            ParsedRequest attempt = req;
            attempt.raw_body["model"] = model.server->config_.model_name;
            model.server->handle_model_request(fd, attempt, count_only,
                                               &reservation.admission);
            if (reservation.admission == RoutingAdmission::handled) return true;
            if (reservation.admission == RoutingAdmission::unfit) {
                if (!automatic) {
                    send_error(fd, 400, "request exceeds model context or KV pool capacity");
                    return true;
                }
                unfit.insert(model.server);
            }
        }
        if (unfit.size() == models_.size()) {
            send_error(fd, 400, "request exceeds the context or KV pool capacity of every model");
            return true;
        }
        if (!known) {
            send_error(fd, 404, "unknown model '" + requested + "'");
            return true;
        }
        if (!automatic || stopping_.load()) {
            send_error(fd, 503, "no available model capacity or server stopping");
            return true;
        }
        std::unique_lock<std::mutex> lock(routing_mu_);
        if (!waiting.active) {
            if (routing_waiters_ >= config_.routing_queue_limit) {
                lock.unlock();
                send_error(fd, 503, "model routing waiting queue is full");
                return true;
            }
            ++routing_waiters_;
            waiting.active = true;
        }
        // Capacity can also change inside an engine without HTTP retirement.
        // A bounded retry probes that state on its owning worker and observes
        // cancellation/shutdown even when all model reservations stay occupied.
        routing_cv_.wait_for(lock, kClientMonitorInterval);
        lock.unlock();
        if (http_detail::inspect_peer_socket(fd) ==
                http_detail::PeerSocketState::Disconnected) return true;
    }
}

json HttpServer::model_routing_status() {
    json models = json::array();
    std::lock_guard<std::mutex> lock(routing_mu_);
    for (const auto & model : models_) {
        HttpServer & server = *model.server;
        json props = build_props_body(
            server.config_, server.prefix_cache_, server.tool_memory_);
        props["memory"] = server.memory_json();
        models.push_back({{"id", server.config_.model_name},
            {"capacity", model.capacity}, {"in_flight", model.in_flight},
            {"execution_mode", server.backend_.seq_engine() ? "batched" : "single-request"},
            {"target_device", server.config_.target_device},
            {"draft_device", server.config_.draft_device},
            {"max_context", server.config_.max_ctx},
            {"props", std::move(props)},
            {"status", server.status_.to_json()}});
    }
    return {{"routing", "primary-first"}, {"waiting", routing_waiters_},
            {"queue_limit", config_.routing_queue_limit}, {"models", models}};
}

// ─── Structured output ──────────────────────────────────────────────────

namespace http_detail {

nlohmann::ordered_json request_output_format(const std::string & body) {
    if (body.find("\"response_format\"") == std::string::npos &&
        body.find("\"output_format\"") == std::string::npos &&
        body.find("\"format\"") == std::string::npos) {
        return nullptr;
    }
    const auto ordered = nlohmann::ordered_json::parse(body, nullptr, false);
    if (!ordered.is_object()) return nullptr;
    if (ordered.contains("response_format")) return ordered["response_format"];
    if (ordered.contains("output_format")) return ordered["output_format"];
    const auto text = ordered.find("text");
    if (text != ordered.end() && text->is_object() && text->contains("format")) {
        return (*text)["format"];
    }
    return nullptr;
}

// A string member of a client object; "" when it is absent or not a string.
// (json::value would throw on a client's {"type": 1}.)
template <typename Json>
std::string string_member(const Json & object, const char * key) {
    if (!object.is_object()) return {};
    const auto it = object.find(key);
    return it != object.end() && it->is_string() ? it->template get<std::string>()
                                                 : std::string();
}

bool output_format_schema(const nlohmann::ordered_json & format,
                          nlohmann::ordered_json & schema, std::string & error) {
    schema = nullptr;
    if (format.is_null()) return true;
    if (!format.is_object()) {
        error = "response_format must be an object";
        return false;
    }
    if (format.contains("type") && !format["type"].is_string()) {
        error = "response_format type must be a string";
        return false;
    }
    const std::string type = string_member(format, "type");
    if (type.empty() || type == "text") return true;
    if (type == "json_object") {
        // A schema beside json_object narrows it, as in llama-server.
        schema = format.contains("schema") ? format["schema"]
                                           : nlohmann::ordered_json::object();
    } else if (type == "json_schema") {
        // Chat completions nest the schema under json_schema; responses and
        // messages put it on the format itself.
        const auto & holder = format.contains("json_schema") ? format["json_schema"] : format;
        schema = holder.is_object() && holder.contains("schema")
            ? holder["schema"] : nlohmann::ordered_json::object();
    } else {
        error = "response_format type must be text, json_object or json_schema, not '" +
                type + "'";
        return false;
    }
    if (!schema.is_object()) {
        error = "the response_format schema must be a JSON object";
        return false;
    }
    if (schema.empty()) schema["type"] = "object";  // an empty schema means any object
    return true;
}

bool tool_choice_requires_call(const json & tool_choice) {
    if (tool_choice.is_string()) {
        return tool_choice == "required" || tool_choice == "any";
    }
    if (!tool_choice.is_object()) return false;
    const std::string type = string_member(tool_choice, "type");
    return type == "any" || type == "tool" || type == "function" ||
           tool_choice.contains("function");
}

static std::string gbnf_literal(const std::string & text) {
    std::string out = "\"";
    for (const char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c;
        }
    }
    return out + "\"";
}

std::string output_constraint_grammar(const nlohmann::ordered_json & schema,
                                      const std::vector<std::string> & tool_openers) {
    return build_grammar([&](const GrammarBuilder & builder) {
        std::string replies;
        if (!schema.is_null()) {
            auto resolved = schema;
            builder.resolve_refs(resolved);
            replies = builder.add_schema("reply", resolved);
        }
        if (!tool_openers.empty()) {
            std::string openers;
            for (const auto & opener : tool_openers) {
                if (!openers.empty()) openers += " | ";
                openers += gbnf_literal(opener);
            }
            // The grammar only admits the opener; the tool parser reads the
            // call, and the constraint lets go of the output once it starts.
            if (!replies.empty()) replies += " | ";
            replies += builder.add_rule("tool-call", "(" + openers + ") .*");
        }
        builder.add_rule("root", "[ \\t\\n]{0,4} (" + replies + ")");
    });
}

}  // namespace http_detail

std::shared_ptr<const ConstraintVocab> HttpServer::output_constraint_vocab() {
    std::call_once(constraint_vocab_once_, [this] {
        std::vector<std::string> pieces((size_t) tokenizer_.vocab_size());
        for (int32_t id = 0; id < tokenizer_.vocab_size(); ++id) {
            pieces[(size_t) id] = tokenizer_.token_text(id);
        }
        constraint_vocab_ = ConstraintVocab::build(
            std::move(pieces), {tokenizer_.eos_id(), tokenizer_.eos_chat_id()});
    });
    return constraint_vocab_;
}

bool HttpServer::attach_output_constraint(SocketHandle fd, ParsedRequest & req) {
    nlohmann::ordered_json schema;
    std::string error;
    if (!http_detail::output_format_schema(req.output_format, schema, error)) {
        send_error(fd, 400, error);
        return false;
    }
    const bool tools_off = tool_checks::tool_choice_is_none(req.tool_choice);
    const bool has_tools = req.tools.is_array() && !req.tools.empty() && !tools_off;
    const bool must_call = has_tools && http_detail::tool_choice_requires_call(req.tool_choice);
    if (schema.is_null() && !must_call) return true;

    // With tools the reply may call one first; a forced call must.
    std::vector<std::string> openers = has_tools ? tool_call_openers(req.tools)
                                                 : std::vector<std::string>{};
    std::string grammar;
    try {
        grammar = http_detail::output_constraint_grammar(
            must_call ? nlohmann::ordered_json() : schema, openers);
    } catch (const std::exception & e) {
        send_error(fd, 400, std::string("response_format: ") + e.what());
        return false;
    }
    auto matcher = gbnf::Matcher::compile(grammar, "root", error);
    if (!matcher) {
        send_error(fd, 400, "response_format: " + error);
        return false;
    }
    OutputConstraint::Options options;
    if (req.started_in_thinking) {
        options.wait_for = config_.think_marker_token_ids.empty()
            ? std::string("</think>")
            : tokenizer_.decode(config_.think_marker_token_ids);
    }
    options.release_prefixes = std::move(openers);
    options.calls_only = must_call;
    req.sampler.constraint = std::make_shared<OutputConstraint>(
        output_constraint_vocab(), std::move(matcher), std::move(options));
    return true;
}

// ─── Request parsing ────────────────────────────────────────────────────

// Fields shared by every endpoint: stream/model/max-tokens, sampler,
// tools, prefix-cache overrides, stop sequences.
bool HttpServer::parse_common_request_fields(
        SocketHandle fd, const json & body, ParsedRequest & req) {
    req.stream = body.value("stream", false);
    req.model = config_.model_name;
    req.disk_cache_policy = config_.disk_cache_policy;

    // Accept the output-token names used by each supported API dialect.
    // Default when the client omits all three: --default-max-tokens, so
    // thinking-budget requests that omit max_tokens keep headroom for the
    // visible reply after thinking.
    req.max_output =
        resolve_max_output_tokens(body, config_.default_max_tokens);
    // Spec §4.4: clamp request max_tokens to --default-max-tokens.
    if (req.max_output > config_.default_max_tokens) {
        std::fprintf(stderr,
            "[server] max_tokens=%d clamped to default_max_tokens=%d\n",
            req.max_output, config_.default_max_tokens);
        req.max_output = config_.default_max_tokens;
    }

    req.sampler = parse_request_sampler(body, config_.sampler_defaults);
    if (body.contains("tools")) req.tools = body["tools"];
    // Tool choice constraint for hint generation.
    if (body.contains("tool_choice")) req.tool_choice = body["tool_choice"];

    if (body.contains("prefix_cache") && body["prefix_cache"].is_object()) {
        const auto & prefix_cache = body["prefix_cache"];
        if (prefix_cache.contains("scope") && prefix_cache["scope"].is_string() &&
            !apply_request_scope_override(
                req.disk_cache_policy, prefix_cache["scope"].get<std::string>())) {
            send_error(fd, 400,
                "prefix_cache.scope must be off, full, auto, auto:<window>, or a positive token count");
            return false;
        }
        if (prefix_cache.contains("window") &&
            prefix_cache["window"].is_number_integer()) {
            const int window = prefix_cache["window"].get<int>();
            if (window <= 0 || window > 1000000) {
                send_error(fd, 400,
                           "prefix_cache.window must be a positive integer");
                return false;
            }
            req.disk_cache_policy.auto_window = window;
        }
    }

    // Stop sequences — OpenAI uses "stop" (string or array), Anthropic
    // uses "stop_sequences" (array).
    if (body.contains("stop")) {
        const auto & stop = body["stop"];
        if (stop.is_string()) {
            std::string value = stop.get<std::string>();
            if (!value.empty()) req.stop_sequences.push_back(std::move(value));
        } else if (stop.is_array()) {
            for (const auto & item : stop) {
                if (!item.is_string()) continue;
                std::string value = item.get<std::string>();
                if (!value.empty()) req.stop_sequences.push_back(std::move(value));
            }
        }
    }
    if (body.contains("stop_sequences") && body["stop_sequences"].is_array()) {
        for (const auto & item : body["stop_sequences"]) {
            if (!item.is_string()) continue;
            std::string value = item.get<std::string>();
            if (!value.empty()) req.stop_sequences.push_back(std::move(value));
        }
    }
    return true;
}

// Path dispatch: sets the API format, response id, and message
// extraction for each supported dialect.
bool HttpServer::parse_endpoint_request(
        const std::string & path, const json & body, ParsedRequest & req,
        bool & count_tokens_only) {
    count_tokens_only = false;
    if (path == "/v1/chat/completions") {
        req.format = ApiFormat::OPENAI_CHAT;
        req.response_id = generate_id("chatcmpl");
        req.messages = require_messages_array(body);
        // Strip volatile billing header from messages[0] (OpenAI system).
        if (req.messages.is_array() && !req.messages.empty()) {
            auto & first_message = req.messages[0];
            if (first_message.is_object() &&
                first_message.value("role", "") == "system" &&
                first_message.contains("content") &&
                first_message["content"].is_string()) {
                first_message["content"] = normalize_system_for_cache(req.messages);
            }
        }
        return true;
    }
    if (path == "/v1/messages/count_tokens") {
        // Shares Anthropic's message parsing; the flag makes route_request
        // short-circuit before enqueueing a generation job.
        req.format = ApiFormat::ANTHROPIC;
        req.response_id = generate_id("count");
        req.messages = body.value("messages", json::array());
        normalize_anthropic_system(body, req.messages);
        count_tokens_only = true;
        return true;
    }
    if (path == "/v1/messages") {
        req.format = ApiFormat::ANTHROPIC;
        req.response_id = generate_id("msg");
        req.messages = require_messages_array(body);
        normalize_anthropic_system(body, req.messages);
        return true;
    }
    if (path == "/v1/responses") {
        req.format = ApiFormat::RESPONSES;
        req.response_id = generate_id("resp");
        // Responses API uses "input" instead of "messages".
        if (body.contains("input")) req.messages = body["input"];
        if (body.contains("instructions")) {
            // Strip billing header from codex instructions before hashing.
            std::string instructions = normalize_system_for_cache(body["instructions"]);
            json system_message = {{"role", "system"}, {"content", instructions}};
            if (req.messages.is_array()) {
                req.messages.insert(req.messages.begin(), system_message);
            } else {
                req.messages = json::array({
                    system_message,
                    {{"role", "user"},
                     {"content", body.value("input", json())}},
                });
            }
        }
        return true;
    }
    return false;
}

void apply_request_reasoning(
        const json & body, const ServerConfig & config, ParsedRequest & req) {
    // Explicit thinking budgets override reasoning-effort tiers. Template
    // kwargs can still override whether the rendered prompt enables thinking.
    // Default: thinking OFF (Qwen3.6 thinking wrecks DFlash acceptance
    // rates; clients opt in explicitly). Qwen3.8-Flash-Next's own template
    // thinks by default, so it keeps the budget envelope on unless disabled.
    bool enable_thinking = config.arch == "qwen4exp";
    int request_budget_tokens = -1;
    int request_reply_budget = -1;
    int effort_phase1_cap = -1;
    bool effort_set = false;
    std::string normalized_effort;

    req.thinking_opt_in = enable_thinking;
    req.per_req_phase1_cap = -1;
    req.per_req_reply_budget = -1;
    req.preserve_thinking = -1;

    auto apply_reasoning_effort = [&](const std::string & effort) {
        if (effort == "none") {
            enable_thinking = false;
            normalized_effort.clear();
            effort_set = true;
            return;
        }

        // Five-tier vocabulary (spec §4.2). Unknown tier → high.
        int tier_value = config.effort_tiers.high;
        if (effort == "minimal" || effort == "low") {
            tier_value = config.effort_tiers.low;
            normalized_effort = "low";
        } else if (effort == "medium") {
            tier_value = config.effort_tiers.medium;
            normalized_effort = arch_is_deepseek4_family(config.arch) ? "high" : "medium";
        } else if (effort == "xhigh") {
            // DeepSeek V4 Flash's OpenAI-compatible APIs map xhigh to high.
            // Other architectures retain Lucebox's x-high tier alias.
            if (arch_is_deepseek4_family(config.arch)) {
                tier_value = config.effort_tiers.high;
                normalized_effort = "high";
            } else {
                tier_value = config.effort_tiers.x_high;
                normalized_effort = "x-high";
            }
        } else if (effort == "x-high") {
            // Hyphenated x-high is Lucebox's explicit five-tier extension.
            tier_value = config.effort_tiers.x_high;
            normalized_effort = arch_is_deepseek4_family(config.arch) ? "max" : "x-high";
        } else if (effort == "max") {
            tier_value = config.effort_tiers.max;
            normalized_effort = "max";
        } else {
            normalized_effort = "high";
        }

        effort_phase1_cap = tier_value;
        effort_set = true;
        enable_thinking = true;
        // Spec §4.2: reasoning effort activates the budget envelope.
        req.thinking_opt_in = true;
    };

    if (body.contains("reasoning") && body["reasoning"].is_object()) {
        const auto & reasoning = body["reasoning"];
        if (reasoning.contains("effort")) {
            apply_reasoning_effort(reasoning.value("effort", "high"));
        } else {
            enable_thinking = true;
        }
    }
    if (!effort_set && body.contains("reasoning_effort") &&
        body["reasoning_effort"].is_string()) {
        apply_reasoning_effort(body["reasoning_effort"].get<std::string>());
    }
    if (body.contains("thinking") && body["thinking"].is_object()) {
        const auto & thinking = body["thinking"];
        // "adaptive" (Claude Code's default) leaves thinking to the model: the
        // same as no thinking field.
        if (thinking.contains("type") && thinking.value("type", "") != "adaptive") {
            const bool enabled = thinking.value("type", "") == "enabled";
            enable_thinking = enabled;
            req.thinking_opt_in = enabled;
        }
        if (thinking.contains("budget_tokens") &&
            thinking["budget_tokens"].is_number_integer()) {
            request_budget_tokens = thinking["budget_tokens"].get<int>();
        }
        if (thinking.contains("reply_budget") &&
            thinking["reply_budget"].is_number_integer()) {
            request_reply_budget = thinking["reply_budget"].get<int>();
        }
    }
    if (body.contains("chat_template_kwargs") &&
        body["chat_template_kwargs"].is_object()) {
        const auto & kwargs = body["chat_template_kwargs"];
        if (!effort_set && kwargs.contains("reasoning_effort") &&
            kwargs["reasoning_effort"].is_string()) {
            apply_reasoning_effort(
                kwargs["reasoning_effort"].get<std::string>());
        }
        if (kwargs.contains("thinking") && kwargs["thinking"].is_boolean()) {
            enable_thinking = kwargs["thinking"].get<bool>();
        }
        if (kwargs.contains("enable_thinking")) {
            enable_thinking = kwargs["enable_thinking"].get<bool>();
        }
        if (kwargs.contains("preserve_thinking") &&
            kwargs["preserve_thinking"].is_boolean()) {
            req.preserve_thinking = kwargs["preserve_thinking"].get<bool>() ? 1 : 0;
        }
    }
    // DeepSeek uses high whenever thinking is enabled without an explicit
    // model-facing effort. Only API-style thinking.type="enabled" also selects
    // the high budget tier; bare template toggles affect rendering alone.
    if (enable_thinking && arch_is_deepseek4_family(config.arch) &&
        normalized_effort.empty()) {
        normalized_effort = "high";
        if (req.thinking_opt_in) {
            effort_phase1_cap = config.effort_tiers.high;
            effort_set = true;
        }
    }
    if (!enable_thinking) {
        normalized_effort.clear();
        req.thinking_opt_in = false;
    }
    req.thinking_enabled = enable_thinking;
    req.reasoning_effort = normalized_effort;

    // Spec §4.3 combined precedence + §4.4 clamping: thinking.budget_tokens
    // (if set) wins over reasoning.effort for the phase-1 cap; either is
    // clamped to the server ceilings.
    if (req.thinking_opt_in && request_budget_tokens >= 0) {
        req.per_req_phase1_cap =
            (std::min)(request_budget_tokens, config.think_max_tokens);
        if (request_budget_tokens > config.think_max_tokens) {
            std::fprintf(stderr,
                "[server] thinking.budget_tokens=%d clamped to "
                "think_max_tokens=%d\n",
                request_budget_tokens, config.think_max_tokens);
        }
    } else if (req.thinking_opt_in && effort_set) {
        // Spec §4.4: effective cap is min(tier value, max_tokens - reply
        // reserve). Tier values can legitimately exceed default_max_tokens;
        // clients that want the full tier budget must pass an explicit
        // max_tokens. Otherwise we narrow silently to fit.
        const int reply_reserve = luce::common::reply_reserve_within(
            config.hard_limit_reply_budget, req.max_output);
        const int max_output_phase1_room =
            (std::max)(0, req.max_output - reply_reserve);
        req.per_req_phase1_cap =
            (std::min)(effort_phase1_cap, max_output_phase1_room);
        if (effort_phase1_cap > max_output_phase1_room) {
            std::fprintf(stderr,
                "[server] reasoning.effort tier=%d narrowed to %d "
                "(max_tokens=%d - reply reserve=%d); "
                "pass a larger max_tokens to use the full tier budget\n",
                effort_phase1_cap, req.per_req_phase1_cap,
                req.max_output, reply_reserve);
        }
    }
    if (req.thinking_opt_in && request_reply_budget >= 0) {
        req.per_req_reply_budget =
            (std::min)(request_reply_budget, config.hard_limit_reply_budget);
        if (request_reply_budget > config.hard_limit_reply_budget) {
            std::fprintf(stderr,
                "[server] thinking.reply_budget=%d clamped to "
                "hard_limit_reply_budget=%d\n",
                request_reply_budget, config.hard_limit_reply_budget);
        }
    }
    // (The effort tier doesn't influence reply_budget — spec §4.2: the
    // reply reserve falls back to --hard-limit-reply-budget.)
}

// Client strings rendered into the template carry no control tokens: see
// Tokenizer::defuse_control_tokens. Replayed turns are the server's own
// generated text and keep their exact tokens.
std::vector<ChatMessage> HttpServer::defuse_message_control_tokens(
        const std::vector<ChatMessage> & messages) const {
    const std::string keep = config_.image_input_enabled
        ? backend_.image_placeholder()
        : std::string();
    std::function<void(json &)> defuse_json = [&](json & value) {
        if (value.is_string()) {
            value = tokenizer_.defuse_control_tokens(
                value.get<std::string>());
        } else if (value.is_object()) {
            // Keys are client text too.
            json defused = json::object();
            for (auto & [key, item] : value.items()) {
                defuse_json(item);
                defused[tokenizer_.defuse_control_tokens(key)] =
                    std::move(item);
            }
            value = std::move(defused);
        } else if (value.is_array()) {
            for (auto & item : value) defuse_json(item);
        }
    };
    std::vector<ChatMessage> out = messages;
    for (ChatMessage & message : out) {
        if (message.tool_calls_replayed) continue;
        message.content =
            tokenizer_.defuse_control_tokens(message.content, keep);
        message.reasoning_content =
            tokenizer_.defuse_control_tokens(message.reasoning_content);
        message.tool_name =
            tokenizer_.defuse_control_tokens(message.tool_name);
        for (ChatToolCall & call : message.tool_calls) {
            call.name = tokenizer_.defuse_control_tokens(call.name);
            defuse_json(call.arguments);
        }
    }
    return out;
}

bool HttpServer::render_messages_to_text(
        const std::vector<ChatMessage> & chat_messages,
        const ParsedRequest & req, bool add_generation_prompt,
        std::string & rendered, std::string & error) {
    std::string tools_json;
    if (req.tools.is_array() && !req.tools.empty()) {
        tools_json = tokenizer_.defuse_control_tokens(req.tools.dump());
    }
    const std::vector<ChatMessage> defused =
        defuse_message_control_tokens(chat_messages);

    if (!config_.chat_template_src.empty()) {
        // Custom templates receive structured calls and normalized schemas;
        // unlike the built-in renderers, they cannot safely replay raw turns.
        //
        // Special tokens like <|im_start|> / <|im_end|> are stored verbatim
        // in the GGUF vocab — use raw_token() to skip the GPT-2 byte decode
        // (otherwise <0xC4><0x91> nonsense appears).
        const std::string & bos = tokenizer_.bos_id() >= 0
            ? tokenizer_.raw_token(tokenizer_.bos_id())
            : std::string();
        const std::string & eos = tokenizer_.eos_id() >= 0
            ? tokenizer_.raw_token(tokenizer_.eos_id())
            : std::string();
        try {
            rendered = render_chat_template_jinja(
                config_.chat_template_src, defused, bos, eos,
                add_generation_prompt,
                req.thinking_enabled, tools_json,
                config_.arch == "qwen4exp" ? qwen4exp_template_effort(req.reasoning_effort) : std::string(),
                req.preserve_thinking);
        } catch (const std::exception & e) {
            error = std::string("chat template (jinja) render failed: ") + e.what();
            return false;
        }
    } else {
        rendered = render_chat_template(
            defused, chat_format_, add_generation_prompt,
            req.thinking_enabled, tools_json, req.reasoning_effort);
    }

    return true;
}

// Render messages to prompt text and tokenize into req.prompt_tokens.
bool HttpServer::render_and_tokenize_request(
        SocketHandle fd, const std::vector<ChatMessage> & chat_messages,
        ParsedRequest & req) {
    std::string error;
    if (!render_messages_to_text(chat_messages, req,
                                 /*add_generation_prompt=*/true,
                                 req.rendered_prompt, error)) {
        // The template rejected these messages: a client error, which a
        // client must not retry as it would a server failure.
        send_error(fd, 400, error);
        return false;
    }
    req.started_in_thinking = prompt_ends_in_open_think(req.rendered_prompt);
    req.prompt_tokens = tokenizer_.encode(req.rendered_prompt);
    return true;
}

// Context-length gate. Oversized prompts pass through when compression
// (pflash) will run — the post-compress check is the real gate.
bool HttpServer::validate_request_context(
        SocketHandle fd, ParsedRequest & req, bool routed) {
    const int prompt_tokens = (int) req.prompt_tokens.size();
    const bool pflash_will_run =
        !req.images && config_.pflash_mode != ServerConfig::PflashMode::OFF &&
        drafter_tokenizer_ != nullptr &&
        (config_.pflash_mode == ServerConfig::PflashMode::ALWAYS ||
         prompt_tokens >= config_.pflash_threshold);
    if (!routed && !pflash_will_run) {
        const int fitted =
            fit_reply_to_context(prompt_tokens, req.max_output, config_.max_ctx);
        if (fitted > 0 && fitted < req.max_output) {
            std::fprintf(stderr,
                "[server] max_tokens=%d trimmed to %d: the prompt takes %d of "
                "the %d-token context\n",
                req.max_output, fitted, prompt_tokens, config_.max_ctx);
            req.max_output = fitted;
        }
    }
    if (!should_reject_oversized(
            prompt_tokens, req.max_output, config_.max_ctx, pflash_will_run)) {
        return true;
    }
    if (!routed) {
        if (pflash_will_run) {
            // PFlash takes raw prompts up to --max-ctx (it makes room for the reply), so only the prompt counts.
            send_error(fd, 400,
                "This model's maximum context length is " + std::to_string(config_.max_ctx) +
                " tokens. However, your messages resulted in " + std::to_string(prompt_tokens) +
                " tokens. Please reduce the length of the messages.");
        } else {
            send_error(fd, 400,
                context_overflow_message(
                    config_.max_ctx, prompt_tokens, req.max_output));
        }
    }
    return false;
}

void HttpServer::log_parsed_request(const ParsedRequest & req) const {
    std::fprintf(stderr,
        "[server] chat %s format=%s stream=%s msgs=%zu tools=%zu prompt_tokens=%zu "
        "max_tokens=%d max_ctx=%d thinking=%s reasoning_effort=%s "
        "started_in_thinking=%s stops=%zu grammar=%s model=%s\n",
        req.response_id.c_str(), api_format_name(req.format),
        req.stream ? "true" : "false",
        json_array_size(req.messages), json_array_size(req.tools),
        req.prompt_tokens.size(), req.max_output, config_.max_ctx,
        req.thinking_enabled ? "true" : "false",
        !req.thinking_enabled ? "none" :
            (req.reasoning_effort.empty() ? "default" :
                req.reasoning_effort.c_str()),
        req.started_in_thinking ? "true" : "false",
        req.stop_sequences.size(), req.sampler.constraint ? "on" : "off",
        req.model.c_str());
}

RoutingAdmission HttpServer::enqueue_request_and_wait(SocketHandle fd, ParsedRequest req,
                                           bool report_admission) {
    // Set socket non-blocking for send() stall detection during streaming.
    sock_set_nonblock(fd);

    ServerJob job;
    job.fd = fd;
    job.req = std::move(req);
    job.report_admission = report_admission;
    enqueue(&job);

    // The worker can spend minutes in prefill before its first token. Keep the
    // SSE response active and watch the read side independently so an orderly
    // client close is observed before the kernel send buffer eventually fills.
    std::unique_lock<std::mutex> lock(job.mu);
    while (!job.done) {
        if (job.cv.wait_for(lock, kClientMonitorInterval,
                            [&]() { return job.done; })) {
            break;
        }

        lock.unlock();
        const auto peer_state = http_detail::inspect_peer_socket(fd);
        if (peer_state == http_detail::PeerSocketState::Disconnected) {
            job.client_disconnected.store(true, std::memory_order_release);
        } else {
            maybe_send_job_heartbeat(
                &job,
                peer_state == http_detail::PeerSocketState::ReadClosed);
        }
        lock.lock();
    }
    return job.admission;
}

bool HttpServer::route_request(SocketHandle fd, const HttpRequest & hr) {
    if (hr.method != "POST") return false;

    std::fprintf(stderr, "[server] request path=%s body_bytes=%zu\n",
                 hr.path.c_str(), hr.body.size());

    ParsedRequest req;
    bool count_tokens_only = false;
    try {
        req.raw_body = json::parse(hr.body);
        req.output_format = http_detail::request_output_format(hr.body);
        if (!parse_endpoint_request(
                hr.path, req.raw_body, req, count_tokens_only)) return false;
        return models_.empty()
            ? handle_model_request(fd, req, count_tokens_only)
            : route_model_request(fd, req, count_tokens_only);
    } catch (const std::exception & e) {
        send_error(fd, 400, std::string("JSON parse error: ") + e.what());
        return true;
    }
}

bool HttpServer::handle_model_request(SocketHandle fd, ParsedRequest & req,
                                      bool count_tokens_only,
                                      RoutingAdmission * admission) {
#if !defined(_WIN32)
    if (!count_tokens_only && req.format == ApiFormat::OPENAI_CHAT &&
        tool_checks::requested(req.raw_body, config_.tool_checks)) {
        return serve_with_tool_checks(fd, req, admission);
    }
#endif
    try {
        const json & body = req.raw_body;
        if (!parse_common_request_fields(fd, body, req)) return true;
        // Image extraction and redaction apply only to an image-capable
        // backend; every other backend sees the request exactly as before.
        std::vector<EncodedImage> encoded_images;
        if (config_.image_input_enabled) {
            json normalized;
            std::string extraction_error;
            const ImageRequestPolicy image_policy{
                req.format == ApiFormat::OPENAI_CHAT,
                config_.image_input_enabled,
                backend_.image_placeholder(),
                backend_.image_part_separator()};
            if (!prepare_request_images(req.messages, image_policy, normalized,
                                        encoded_images, extraction_error)) {
                send_error(fd, 400, extraction_error);
                return true;
            }
            req.messages = std::move(normalized);
            redact_image_urls(req.raw_body);
            redact_image_urls(req.messages);
        }

        const std::vector<ChatMessage> chat_messages =
            normalize_chat_messages(req.messages, req.format, tool_memory_,
                                    config_.chat_template_src.empty());
        req.ends_with_tool_result =
            http_detail::ends_with_tool_result(chat_messages);
        // Reasoning must be applied BEFORE rendering: the template injects
        // the empty <think>\n\n</think>\n\n block when thinking is disabled.
        apply_request_reasoning(body, config_, req);
        // req.sampler was built by parse_common_request_fields above, before
        // the final thinking state was known — it can only have applied the
        // thinking-mode `sampler_defaults`. Now that req.thinking_enabled is
        // resolved, backfill any still-omitted fields from the card's
        // no-thinking defaults when thinking ended up OFF.
        if (!req.thinking_enabled) {
            apply_no_thinking_sampler_defaults(
                body, config_.sampler_defaults_no_thinking, req.sampler);
        }
        // Bandit: parse session_id from extra_body (opt-in adaptive keep_ratio).
        req.session_id = parse_session_id_from_body(body);
        req.pflash_query = parse_pflash_query_from_body(body);
        req.pflash_required = parse_pflash_required_from_body(body);

        // PPP rearrange (optional): peel ephemeral system banners into a
        // following system message so the first chat boundary is stable.
        std::vector<ChatMessage> render_messages = chat_messages;
        if (encoded_images.empty() && config_.ppp_enabled && config_.ppp_rearrange && !req.tools.empty()) {
            auto layout = PinFriendlyPrompt::rearrange(chat_messages, true);
            if (layout.rearranged) {
                render_messages = std::move(layout.messages);
                // Keep FlowKV / response formatting aligned with the served
                // layout (FlowKV re-renders from req.messages).
                if (req.messages.is_array() && !req.messages.empty() &&
                    req.messages[0].value("role", "") == "system" &&
                    render_messages.size() >= 2) {
                    req.messages[0]["content"] = render_messages[0].content;
                    json meta = {
                        {"role", "system"},
                        {"content", render_messages[1].content},
                    };
                    req.messages.insert(req.messages.begin() + 1, std::move(meta));
                }
                std::fprintf(stderr,
                    "[ppp] rearranged: peeled ephemeral system tail\n");
            }
        }

        if (!render_and_tokenize_request(fd, render_messages, req)) return true;

        std::string image_error;
        const ImagePrepareStatus image_status = backend_.prepare_images(
                req.prompt_tokens, std::move(encoded_images),
                uint64_t(std::max(0, config_.max_ctx)), uint64_t(std::max(0, req.max_output)),
                req.images, image_error);
        if (image_status != ImagePrepareStatus::ok) {
            // A full image gate is capacity, not a bad request: clients retry 503.
            send_error(fd, image_status == ImagePrepareStatus::busy ? 503 : 400, image_error);
            return true;
        }

        // count_tokens: short-circuit after tokenization. Skip generation
        // entirely — Anthropic's contract is just {"input_tokens": N}.
        if (count_tokens_only) {
            const json response = {
                {"input_tokens", (int) req.prompt_tokens.size()},
            };
            send_response(fd, 200, "application/json", response.dump() + "\n");
            return true;
        }
        if (!attach_output_constraint(fd, req)) return true;
    } catch (const json::parse_error & e) {
        send_error(fd, 400, std::string("JSON parse error: ") + e.what());
        return true;
    } catch (const std::exception & e) {
        send_error(fd, 400, std::string("Invalid request: ") + e.what());
        return true;
    }

    if (!validate_request_context(fd, req, admission != nullptr)) {
        if (admission) *admission = RoutingAdmission::unfit;
        return true;
    }
    log_parsed_request(req);
    const auto outcome = enqueue_request_and_wait(fd, std::move(req), admission != nullptr);
    if (admission) *admission = outcome;
    return true;
}

namespace {

// Keeps a held SSE response open with comment lines while the checks run.
class SseKeepAlive {
public:
    explicit SseKeepAlive(std::function<bool()> beat)
        : beat_(std::move(beat)), thread_([this] { run(); }) {}
    ~SseKeepAlive() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopped_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mu_);
        while (!cv_.wait_for(lock, kSseHeartbeatInterval, [this] { return stopped_; })) {
            if (!beat_()) return;
        }
    }

    std::function<bool()> beat_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stopped_ = false;
    std::thread thread_;
};

}  // namespace

bool HttpServer::inner_chat_completion(const json & request, RoutingAdmission * admission,
                                       int & status, std::string & body) {
#if defined(_WIN32)
    (void) request; (void) admission;
    status = 501;
    body = json({{"error", {{"message", "tool checks need socketpair"}}}}).dump();
    return true;
#else
    int pair[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) {
        status = 500;
        body = json({{"error", {{"message", std::string("socketpair: ") + sock_strerror()}}}}).dump();
        return true;
    }
    std::string raw;
    std::thread reader([&raw, in = pair[1]] {
        char buf[65536];
        for (;;) {
            const ssize_t n = ::recv(in, buf, sizeof(buf), 0);
            if (n > 0) raw.append(buf, (size_t) n);
            else if (n == 0 || !sock_is_eintr(sock_errno())) break;
        }
    });
    RoutingAdmission outcome = RoutingAdmission::handled;
    try {
        ParsedRequest inner;
        bool count_only = false;
        inner.raw_body = request;
        inner.output_format = http_detail::request_output_format(request.dump());
        parse_endpoint_request("/v1/chat/completions", inner.raw_body, inner, count_only);
        handle_model_request(pair[0], inner, false, admission ? &outcome : nullptr);
    } catch (const std::exception & e) {
        send_error(pair[0], 500, e.what());
    }
    ::shutdown(pair[0], SHUT_WR);
    reader.join();
    socket_close(pair[0]);
    socket_close(pair[1]);
    if (admission) {
        *admission = outcome;
        if (outcome != RoutingAdmission::handled) return false;
    }
    const size_t head_end = raw.find("\r\n\r\n");
    status = raw.size() > 12 && raw.rfind("HTTP/1.1 ", 0) == 0 ? std::atoi(raw.c_str() + 9) : 500;
    body = head_end == std::string::npos ? raw : raw.substr(head_end + 4);
    return true;
#endif
}

bool HttpServer::serve_with_tool_checks(SocketHandle fd, ParsedRequest & req,
                                        RoutingAdmission * admission) {
    namespace tc = tool_checks;
    const auto started = std::chrono::steady_clock::now();
    const bool stream = req.raw_body.value("stream", false);
    const json request = tc::agent_request(req.raw_body, tc::kEffort);
    const json & messages = request["messages"];

    // A routed request reports admission before any byte goes out, so its
    // stream opens after the first pass; otherwise it opens now.
    std::unique_ptr<SseKeepAlive> keepalive;
    auto open_stream = [&] {
        if (!stream || keepalive) return;
        std::string header = "HTTP/1.1 200 OK\r\n";
        if (config_.enable_cors) header += "Access-Control-Allow-Origin: *\r\n";
        header += "Content-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                  "Connection: keep-alive\r\n\r\n";
        send_all(fd, header.data(), header.size());
        keepalive = std::make_unique<SseKeepAlive>([this, fd] {
            return send_all(fd, kSseHeartbeat, sizeof(kSseHeartbeat) - 1);
        });
    };
    if (!admission) open_stream();

    // A pass that fails leaves the reply it would have replaced.
    auto pass = [&](const json & body, json & response) {
        if (http_detail::inspect_peer_socket(fd) == http_detail::PeerSocketState::Disconnected) {
            return false;
        }
        int status = 0;
        std::string raw;
        if (!inner_chat_completion(body, nullptr, status, raw) || status != 200) return false;
        response = json::parse(raw, nullptr, false);
        return !response.is_discarded() && tc::reply_message(response) != nullptr;
    };

    int status = 0;
    std::string raw;
    if (!inner_chat_completion(request, admission, status, raw)) return true;
    json response = json::parse(raw, nullptr, false);
    json * reply = status == 200 && !response.is_discarded() ? tc::reply_message(response) : nullptr;
    if (!reply) {
        if (keepalive) {
            keepalive->stop();
            raw.erase(raw.find_last_not_of(" \r\n") + 1);
            const std::string error = "data: " + raw + "\n\ndata: [DONE]\n\n";
            send_all(fd, error.data(), error.size());
        } else {
            send_response(fd, status ? status : 500, "application/json", raw);
        }
        return true;
    }
    open_stream();

    // If the model calls a tool when tool_choice is "none", retry
    // once with a note that tools are unavailable. If the retry still
    // calls one, drop its calls and send its text, so no tool call
    // reaches the client.
    const bool tools_off = tc::tools_off(request);
    const char * tool_free = "-";
    const char * replan = "-";
    const char * final_check = "-";
    if (tc::needs_tool_free_reply(request, *reply)) {
        json again;
        if (pass(tc::with_note(request, *reply, tc::kNoToolsNote), again)) {
            response = std::move(again);
            reply = tc::reply_message(response);
        }
        tool_free = "replaced";
        if (tc::needs_tool_free_reply(request, *reply)) {
            tc::drop_calls(response);
            tool_free = "dropped";
        }
    }
    if (!tools_off && tc::needs_replan(*reply)) {
        json again;
        json * again_reply = pass(tc::with_note(request, *reply, tc::kReplanNote), again)
            ? tc::reply_message(again) : nullptr;
        replan = "kept";
        if (again_reply && again_reply->contains("tool_calls") &&
            !(*again_reply)["tool_calls"].empty()) {
            response = std::move(again);
            reply = tc::reply_message(response);
            replan = "replanned";
        }
    }
    if (!tools_off && tc::needs_final_check(messages, *reply, checked_calls_)) {
        json again;
        json * again_reply = pass(tc::with_note(request, *reply, tc::kFinalNote), again)
            ? tc::reply_message(again) : nullptr;
        json fresh = again_reply ? tc::fresh_calls(messages, *again_reply) : json::array();
        final_check = "done";
        if (!fresh.empty()) {
            checked_calls_.add(fresh);
            (*again_reply)["tool_calls"] = std::move(fresh);
            response = std::move(again);
            final_check = "more_work";
        }
    }
    std::fprintf(stderr, "[tool-checks] %s tool_free=%s replan=%s final_check=%s ms=%lld\n",
                 response.value("id", std::string()).c_str(), tool_free, replan, final_check,
                 (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - started).count());

    if (keepalive) {
        keepalive->stop();
        const std::string events = tc::as_sse(response);
        send_all(fd, events.data(), events.size());
    } else {
        send_response(fd, 200, "application/json", response.dump() + "\n");
    }
    return true;
}

// ─── Worker thread ──────────────────────────────────────────────────────

// Non-streaming response serialization is API-specific but independent of
// prompt preparation and model execution.

namespace {

// Disk-cache staging lives above both PrefixCache pools.
constexpr int kDiskStagingSlot = ModelBackend::kMaxSlots - 1;


struct CompletionTokenCounts {
    int total = 0;
    int reasoning = 0;
    int content = 0;
};

enum class TokenDelivery {
    kSkip,      // EOS or internal control marker — never reaches the emitter
    kThinkTag,  // thinking-boundary marker mapped to its <think> text form
    kText,      // ordinary text (token_text may still be empty)
};

// Classifies one generated token for delivery to the SseEmitter, filling
// `text` with the deliverable form for kThinkTag / kText. Shared by the
// streaming on_token callback and the non-streaming replay: the two paths
// MUST classify identically, or the reasoning/content split diverges
// between streamed and non-streamed responses.
TokenDelivery classify_generated_token(
        Tokenizer & tokenizer, SseEmitter & emitter,
        int32_t token, std::string & text) {
    if (token == tokenizer.eos_id() || token == tokenizer.eos_chat_id()) {
        return TokenDelivery::kSkip;
    }

    const std::string & raw = tokenizer.raw_token(token);
    if (emitter.suppress_undeclared_tool_protocol_token(raw)) {
        return TokenDelivery::kSkip;
    }

    // Gemma4 thinking channel (<|channel> / <channel|>) and Qwen3.6
    // thinking markers share one mapped dialect. The Qwen markers
    // <think> (id 248068) and </think> (id 248069) are SINGLE special
    // tokens in the added_tokens vocab: without this mapping they would
    // hit the generic "<...>" strip below and be silently dropped, the
    // emitter would never see the reasoning→content transition, and the
    // whole answer would land in reasoning_content with empty content.
    if (raw == "<|channel>" || raw == "<think>") {
        text = "<think>";
        return TokenDelivery::kThinkTag;
    }
    if (raw == "<channel|>" || raw == "</think>") {
        text = "</think>\n";
        return TokenDelivery::kThinkTag;
    }

    // Bailing V3 stores its tool-call XML delimiters as added special
    // tokens. They are part of the model's public output protocol, not chat
    // control markers: keep them so the shared parser can reconstruct an
    // OpenAI `tool_calls` object. Stripping them leaves only three unrelated
    // text lines (function name, argument name, value).
    if (raw == "<tool_call>" || raw == "</tool_call>" ||
        raw == "<arg_key>" || raw == "</arg_key>" ||
        raw == "<arg_value>" || raw == "</arg_value>") {
        text = raw;
        return TokenDelivery::kText;
    }

    // Other special tokens are internal control markers. Byte-fallback
    // tokens such as <0xAB> are text and must still reach the emitter.
    if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') {
        return TokenDelivery::kSkip;
    }
    if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>' &&
        !(raw.size() == 6 && raw[1] == '0' && raw[2] == 'x')) {
        return TokenDelivery::kSkip;
    }

    text = tokenizer.token_text(token);
    return TokenDelivery::kText;
}

CompletionTokenCounts feed_non_streaming_tokens(
        const std::vector<int32_t> & tokens, Tokenizer & tokenizer,
        SseEmitter & emitter) {
    for (int32_t token : tokens) {
        std::string text;
        const TokenDelivery delivery =
            classify_generated_token(tokenizer, emitter, token, text);
        if (delivery == TokenDelivery::kSkip) continue;

        emitter.emit_token(text);
        // Matching the streaming path, only ordinary text is checked
        // against stop sequences — think-tag markers never count.
        if (delivery == TokenDelivery::kText && emitter.stop_hit()) break;
    }

    CompletionTokenCounts counts;
    counts.total = (int) tokens.size();
    emitter.emit_finish(counts.total);

    // Split reasoning vs content at the emitter's REASONING→CONTENT
    // transition; see first_content_token_index() in sse_emitter.h.
    // Skipped specials mean `emitted` can be below counts.total — the
    // remainder is unattributed (e.g. TOOL_BUFFER).
    const int first_content = emitter.first_content_token_index();
    const int emitted = emitter.emit_token_count();
    counts.reasoning = first_content < 0 ? emitted : first_content;
    counts.content = first_content < 0 ? 0 : emitted - first_content;
    return counts;
}

json build_openai_completion_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings,
        const CompletionTokenCounts & counts, const SseEmitter & emitter,
        const Tokenizer * tokenizer = nullptr) {
    json message = {
        {"role", "assistant"},
        {"content", emitter.accumulated_text()},
    };
    if (!emitter.reasoning_text().empty()) {
        // Multi-dialect reasoning emission — same text, three keys. See
        // docs/specs/thinking-budget.md "Response shape — multi-dialect
        // aliasing".
        //   reasoning_content : DeepSeek R1 / dflash primary
        //   reasoning         : OpenRouter / Anthropic-gateway flat
        //   reasoning_details : typed-block list; single block.
        const std::string & reasoning = emitter.reasoning_text();
        message["reasoning_content"] = reasoning;
        message["reasoning"] = reasoning;
        message["reasoning_details"] = json::array({
            {{"type", "reasoning.text"}, {"text", reasoning}},
        });
    }
    if (!emitter.tool_calls().empty()) {
        json tool_calls = json::array();
        for (const auto & tool_call : emitter.tool_calls()) {
            tool_calls.push_back({
                {"id", tool_call.id},
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", tool_call.arguments},
                }},
            });
        }
        message["tool_calls"] = tool_calls;
    }
    // The emitter only knows "stop" / "tool_calls"; it cannot see that the
    // daemon hit the n_gen cap. Derive "length" from the committed-token
    // count — OpenAI-compatible clients (open-webui, Cline) gate retry
    // logic on finish_reason == "length".
    const bool is_eos = tokenizer && !result.tokens.empty() &&
        (result.tokens.back() == tokenizer->eos_id() ||
         result.tokens.back() == tokenizer->eos_chat_id());
    std::string finish_reason = emitter.finish_reason();
    if (finish_reason == "stop" && !is_eos && counts.total >= generation_cap) {
        finish_reason = "length";
    }
    // Degenerate decode (post-close repetition-loop watchdog) also reports
    // "length": OpenAI/Anthropic/Gemini all collapse budget-class events
    // into one closed enum, with richer signal in sidecar fields below.
    if (result.degenerate_decode_close) finish_reason = "length";

    json choice = {
        {"index", 0},
        {"message", message},
        {"finish_reason", finish_reason},
    };
    if (req.thinking_opt_in) {
        // finish_details mirrors ds4_eval.c's eval_think_close_info.
        // close_kind is "natural" when the model closed its own thinking
        // block, "hard" when the BudgetHook force-closed it at the budget
        // boundary. See docs/specs/thinking-budget.md "v2 design".
        choice["finish_details"] = {
            {"close_kind", result.budget_forced_close ? "hard" : "natural"},
            {"thinking_tokens", counts.reasoning},
            {"content_tokens", counts.content},
            {"total_tokens", counts.total},
        };
        // Honest signaling: the answer after an aborted repetition loop is
        // unreliable; surface that without breaking the finish_reason enum.
        if (result.degenerate_decode_close) {
            choice["finish_details"]["degenerate_decode"] = true;
        }
    }

    // usage.completion_tokens_details.reasoning_tokens — OpenAI o1/o3
    // standard location; kept in sync with finish_details.thinking_tokens.
    // usage.timings — per-request prefill/decode wall clock, additive to
    // the OpenAI shape (ignored by clients that don't recognize it).
    // spec_decode_ran disambiguates a real zero-acceptance speculative run
    // from an autoregressive fallback. See docs/specs/thinking-budget.md §6.3.
    const int prompt_tokens = usage_prompt_tokens(req);
    const json usage = {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", counts.total},
        {"total_tokens", prompt_tokens + counts.total},
        {"completion_tokens_details", {
            {"reasoning_tokens", counts.reasoning},
        }},
        {"prompt_tokens_details", {
            {"cached_tokens", timings.cached_prefix_tokens},
        }},
        {"timings", build_timings_json(timings, counts.total)},
        {"accept_rate", result.accept_rate},
        {"spec_decode_ran", result.spec_decode_ran},
    };
    return {
        {"id", req.response_id},
        {"object", "chat.completion"},
        {"created", std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()},
        {"model", req.model},
        {"choices", json::array({choice})},
        {"usage", usage},
    };
}

json build_anthropic_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings,
        const CompletionTokenCounts & counts, const SseEmitter & emitter,
        const Tokenizer * tokenizer = nullptr) {
    json content = json::array();
    if (!emitter.reasoning_text().empty()) {
        content.push_back({
            {"type", "thinking"},
            {"thinking", emitter.reasoning_text()},
        });
    }
    // Tool-only replies intentionally omit an empty text block; Anthropic
    // clients expect the tool_use blocks to stand alone.
    if (!emitter.accumulated_text().empty()) {
        content.push_back({
            {"type", "text"},
            {"text", emitter.accumulated_text()},
        });
    }
    for (const auto & tool_call : emitter.tool_calls()) {
        json input = json::object();
        if (!tool_call.arguments.empty()) {
            json parsed = json::parse(tool_call.arguments, nullptr, false);
            if (!parsed.is_discarded() && parsed.is_object()) {
                input = std::move(parsed);
            } else {
                input = {{"_raw", tool_call.arguments}};
            }
        }
        content.push_back({
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name},
            {"input", std::move(input)},
        });
    }

    // stop_reason is Anthropic's analog of finish_reason, with the same
    // length-vs-EOS distinction — Cline / Anthropic SDK clients gate
    // retry logic on stop_reason == "max_tokens".
    const bool is_eos = tokenizer && !result.tokens.empty() &&
        (result.tokens.back() == tokenizer->eos_id() ||
         result.tokens.back() == tokenizer->eos_chat_id());
    std::string stop_reason;
    if (emitter.finish_reason() == "tool_calls") {
        stop_reason = "tool_use";
    } else if (!is_eos && counts.total >= generation_cap) {
        stop_reason = "max_tokens";
    } else {
        stop_reason = "end_turn";
    }

    const json usage = {
        {"input_tokens", usage_prompt_tokens(req)},
        {"output_tokens", counts.total},
        {"timings", build_timings_json(timings, counts.total)},
        {"accept_rate", result.accept_rate},
        {"spec_decode_ran", result.spec_decode_ran},
    };
    return {
        {"id", req.response_id},
        {"type", "message"},
        {"role", "assistant"},
        {"model", req.model},
        {"content", content},
        {"stop_reason", stop_reason},
        {"usage", usage},
    };
}

json build_responses_api_response(
        const ParsedRequest & req, const GenerateResult & result,
        const GenTimings & timings, const CompletionTokenCounts & counts,
        const SseEmitter & emitter) {
    // The same items as the stream's response.completed: the assistant
    // message first, then one function_call item per call.
    const json output = responses_output_items(
        req.response_id + "_msg", emitter.accumulated_text(), emitter.tool_calls());

    const int prompt_tokens = usage_prompt_tokens(req);
    const json usage = {
        {"input_tokens", prompt_tokens},
        {"output_tokens", counts.total},
        {"total_tokens", prompt_tokens + counts.total},
        {"input_tokens_details", {
            {"cached_tokens", timings.cached_prefix_tokens},
        }},
        {"timings", build_timings_json(timings, counts.total)},
        {"accept_rate", result.accept_rate},
        {"spec_decode_ran", result.spec_decode_ran},
    };
    return {
        {"id", req.response_id},
        {"object", "response"},
        {"status", "completed"},
        {"model", req.model},
        {"output", output},
        {"usage", usage},
    };
}

json build_non_streaming_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings,
        const CompletionTokenCounts & counts, SseEmitter & emitter,
        const Tokenizer * tokenizer = nullptr) {
    switch (req.format) {
    case ApiFormat::OPENAI_CHAT:
        return build_openai_completion_response(
            req, result, generation_cap, timings, counts, emitter, tokenizer);
    case ApiFormat::ANTHROPIC:
        return build_anthropic_response(
            req, result, generation_cap, timings, counts, emitter, tokenizer);
    case ApiFormat::RESPONSES:
        return build_responses_api_response(
            req, result, timings, counts, emitter);
    default:
        return {{"text", emitter.accumulated_text()}};
    }
}

json build_non_streaming_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings, Tokenizer & tokenizer,
        SseEmitter & emitter) {
    const CompletionTokenCounts counts = feed_non_streaming_tokens(
        result.tokens, tokenizer, emitter);
    return build_non_streaming_response(
        req, result, generation_cap, timings, counts, emitter, &tokenizer);
}

// Prompt preparation applies exactly one compression policy: FlowKV for
// continuations, a verbatim turn-one anchor, or whole-prompt PFlash.
bool is_continuation_request(const json & messages) {
    if (!messages.is_array()) return false;

    for (const auto & message : messages) {
        if (!message.is_object()) continue;
        if (message.value("role", "") == "assistant") return true;

        if (message.contains("tool_calls")) {
            const auto & tool_calls = message["tool_calls"];
            if (tool_calls.is_array() && !tool_calls.empty()) return true;
        }
        if (message.contains("content") && message["content"].is_array()) {
            for (const auto & block : message["content"]) {
                if (!block.is_object()) continue;
                const std::string type = block.value("type", "");
                if (type == "tool_result" || type == "tool_use") return true;
            }
        }

        const std::string type = message.value("type", "");
        if (type == "function_call" || type == "function_call_output") {
            return true;
        }
    }
    return false;
}

}  // namespace

void HttpServer::apply_flowkv_compression(
        const ParsedRequest & req, PreparedPrompt & prepared) {
    int hot_window = 2;
    const char * hot_window_env = std::getenv("PFLASH_FREEZE_HOT_WINDOW");
    if (hot_window_env && *hot_window_env) {
        const int configured_window = std::atoi(hot_window_env);
        if (configured_window > 0) hot_window = configured_window;
    }

    const int message_count = (int) req.messages.size();
    if (message_count < 2 + hot_window) {
        std::fprintf(stderr,
            "[flowkv] too few turns (n_msgs=%d hot_window=%d) — skip\n",
            message_count, hot_window);
        return;
    }

    const int aged_begin = 1;
    const int aged_end = message_count - hot_window;

    struct AgedMessage {
        int index = -1;
        std::vector<int32_t> drafter_ids;
        PrefixHash cache_key{};
    };

    // FlowKV's curve follows the total request length, not the size of each
    // individual message. This makes one configuration adapt consistently as
    // a conversation grows.
    const float keep_ratio = http_detail::resolve_pflash_keep_ratio(
        pflash_keep_ratio(config_, (int) req.prompt_tokens.size()),
        req.session_id, sessions_);

    auto message_text = [](const json & message) {
        std::string content;
        if (!message.is_object() || !message.contains("content")) {
            return content;
        }
        const auto & value = message["content"];
        if (value.is_string()) {
            content = value.get<std::string>();
        } else if (value.is_array()) {
            for (const auto & part : value) {
                if (!part.is_object()) continue;
                const std::string type = part.value("type", "");
                if (type == "text" || type == "input_text" ||
                    type == "output_text") {
                    content += part.value("text", "");
                }
            }
        }
        return content;
    };

    // AUTO is based on the aggregate aged history, matching the total-prompt
    // threshold users configure. Once active, avoid tiny per-message scoring
    // jobs that cost more than they remove.
    int aged_token_estimate = 0;
    std::vector<AgedMessage> aged_messages;
    for (int index = aged_begin; index < aged_end; ++index) {
        const std::string content = message_text(req.messages[index]);
        if (content.empty()) continue;

        auto ids = drafter_tokenizer_->encode(content);
        aged_token_estimate += (int) ids.size();
        if ((int) ids.size() < http_detail::kFlowKvInertMinTokens) continue;

        // The same aged message can be revisited at a different point on the
        // context-length curve. Include the selected ratio in the cache key so
        // a 10% result is not reused when the larger context asks for 2%.
        ids.push_back((int32_t) std::lround(keep_ratio * 1000000.0f));
        const PrefixHash key = frozen_block_key(
            ids.data(), 0, (int) ids.size());
        ids.pop_back();
        aged_messages.push_back({index, std::move(ids), key});
    }

    const int activation_threshold =
        http_detail::flowkv_activation_threshold(config_);
    if (!http_detail::flowkv_should_activate(
            config_, aged_token_estimate)) {
        std::fprintf(stderr,
            "[flowkv] aged band %d toks < activation threshold %d — skip\n",
            aged_token_estimate, activation_threshold);
        return;
    }
    if (aged_messages.empty()) {
        std::fprintf(stderr,
            "[flowkv] no aged messages >= %d tokens — skip\n",
            http_detail::kFlowKvInertMinTokens);
        return;
    }

    json modified_messages = req.messages;
    bool any_compressed = false;
    int cache_hits = 0;
    const auto residency_action = resolve_draft_residency_action(
        config_.draft_residency,
        DraftResidencyContext{
            DraftResidencyUse::PFlashCompress,
            config_.lazy_draft,
            !config_.draft_path.empty(),
        });

    std::vector<ModelBackend::CompressRequest> compress_requests;
    std::vector<int> compress_message_indices;
    std::vector<PrefixHash> compress_cache_keys;
    compress_requests.reserve(aged_messages.size());
    compress_message_indices.reserve(aged_messages.size());
    compress_cache_keys.reserve(aged_messages.size());

    for (auto & aged : aged_messages) {
        auto cache_it = frozen_content_cache_.find(aged.cache_key);
        if (cache_it != frozen_content_cache_.end()) {
            modified_messages[aged.index]["content"] = cache_it->second;
            any_compressed = true;
            ++cache_hits;
            std::fprintf(stderr,
                "[flowkv] msg[%d] cache hit (%zu drafter toks)\n",
                aged.index, aged.drafter_ids.size());
            continue;
        }

        ModelBackend::CompressRequest compress_request;
        compress_request.input_ids = std::move(aged.drafter_ids);
        compress_request.keep_ratio = keep_ratio;
        compress_request.drafter_path = config_.pflash_drafter_path;
        compress_request.drafter_gpu = config_.pflash_drafter_gpu;
        compress_request.skip_park = config_.pflash_skip_park;
        compress_request.residency_action = residency_action;
        compress_requests.push_back(std::move(compress_request));
        compress_message_indices.push_back(aged.index);
        compress_cache_keys.push_back(aged.cache_key);
    }

    std::vector<ModelBackend::CompressResult> compress_results;
    if (!compress_requests.empty()) {
        if (config_.pflash_remote_drafter) {
            compress_results.resize(compress_requests.size());
            if (!pflash_remote_.active() &&
                !pflash_remote_.start(config_.pflash_remote.ipc_bin,
                                      config_.pflash_drafter_path,
                                      config_.pflash_drafter_gpu,
                                      config_.pflash_remote.work_dir)) {
                std::fprintf(stderr,
                    "[flowkv] remote PFlash drafter start failed\n");
            } else {
                for (size_t index = 0; index < compress_requests.size(); ++index) {
                    auto & result = compress_results[index];
                    result.ok = pflash_remote_.compress(
                        compress_requests[index].input_ids,
                        compress_requests[index].keep_ratio,
                        result.compressed_ids);
                }
                if (residency_action == DraftResidencyAction::ReleaseAfterUse) {
                    pflash_remote_.close();
                }
            }
        } else {
            compress_results = backend_.compress_batch(compress_requests);
        }
    }

    for (size_t index = 0; index < compress_requests.size(); ++index) {
        if (index >= compress_results.size() ||
            !compress_results[index].ok ||
            compress_results[index].compressed_ids.empty()) {
            std::fprintf(stderr,
                "[flowkv] msg[%d] compress failed — kept verbatim\n",
                compress_message_indices[index]);
            continue;
        }

        const auto & result = compress_results[index];
        const std::string compressed_text =
            drafter_tokenizer_->decode(result.compressed_ids);
        modified_messages[compress_message_indices[index]]["content"] =
            compressed_text;
        any_compressed = true;
        std::fprintf(stderr,
            "[flowkv] msg[%d] %zu → %zu drafter toks (keep=%.2f)\n",
            compress_message_indices[index],
            compress_requests[index].input_ids.size(),
            result.compressed_ids.size(), compress_requests[index].keep_ratio);

        if (frozen_content_cache_.size() >= kFrozenCacheMax) {
            std::fprintf(stderr,
                "[flowkv] cache full (%zu entries) — clearing\n",
                frozen_content_cache_.size());
            frozen_content_cache_.clear();
        }
        frozen_content_cache_.emplace(
            compress_cache_keys[index], compressed_text);
    }

    if (!any_compressed) {
        std::fprintf(stderr,
            "[flowkv] no aged msgs above threshold — skip\n");
        return;
    }

    const std::vector<ChatMessage> chat_messages = normalize_chat_messages(
        modified_messages, req.format, tool_memory_, config_.chat_template_src.empty());

    std::string rendered;
    std::string render_error;
    if (!render_messages_to_text(
            chat_messages, req, /*add_generation_prompt=*/true,
            rendered, render_error)) {
        std::fprintf(stderr,
            "[flowkv] re-render failed (%s) — skipping\n",
            render_error.c_str());
        return;
    }

    const int tokens_before = (int) prepared.tokens.size();
    prepared.tokens = tokenizer_.encode(rendered);
    prepared.compressed = true;
    prepared.flowkv = true;
    std::fprintf(stderr,
        "[flowkv] %d → %d target toks "
        "(%zu eligible aged msgs, %d cache hits, keep=%.3f, hot_window=%d)\n",
        tokens_before, (int) prepared.tokens.size(),
        aged_messages.size(), cache_hits, keep_ratio, hot_window);
}

std::string HttpServer::apply_pflash_compression(
        const ParsedRequest & req, PreparedPrompt & prepared) {
    const bool selection_environment =
        luce::pflash::has_pflash_selection_environment();
    // Concurrent serving measured the cache on the scheduler thread, which
    // owns the prefix cache, before handing the request over.
    const bool cache_measured = req.pflash_cached_prefix.has_value();
    auto [full_slot, full_len] = cache_measured
        ? std::pair<int, int>{-1, 0}
        : prefix_cache_.lookup_full(req.prompt_tokens);
    if (http_detail::pflash_full_cache_restore_allowed(
            selection_environment) && full_slot >= 0) {
        std::fprintf(stderr,
            "[pflash] full-cache hit slot=%d — skipping compress\n",
            full_slot);
        prepared.compressed = true;
        prepared.full_cache_hit_slot = full_slot;
        prepared.full_cache_hit_len = full_len;
        // The restore path only needs a prompt whose length matches snap_pos.
        prepared.tokens.assign((size_t) full_len, 0);
        prepared.full_cache_served_tokens = full_len;
        return {};
    }

    const int prompt_tokens = (int) req.prompt_tokens.size();
    const std::string prompt_text = tokenizer_.decode(req.prompt_tokens);
    auto drafter_ids = drafter_tokenizer_->encode(prompt_text);

    if (drafter_ids.empty()) {
        return "PFlash drafter tokenizer produced an empty prompt";
    }

    luce::pflash::PFlashSelectionConfig experiment;
    std::string experiment_error;
    if (!luce::pflash::resolve_pflash_selection(
            (int) drafter_ids.size(), 32, experiment, experiment_error)) {
        return "invalid PFlash strict selection config: " + experiment_error;
    }
    if (!experiment.selection_active && !req.pflash_required.empty()) {
        return "PFlash pflash_required needs strict budget selection";
    }
    if (experiment.fallback_spread > 0.0 && config_.pflash_remote_drafter) {
        return "invalid PFlash strict selection config: PFLASH_FALLBACK_SPREAD needs the "
            "in-process drafter";
    }

    const bool messages_input =
        req.messages.is_array() && !req.messages.empty();
    const bool raw_text_input = req.messages.is_string();
    const char * parser_input_kind = messages_input
        ? "messages" : (raw_text_input ? "raw_text" : "unsupported");
    const bool tail_parser = experiment.query_parser ==
        luce::pflash::PFlashQueryParser::ArbitraryTail;
    std::string parser_selection_rule;
    std::string last_user_text;
    int query_content_begin = -1;
    int query_content_end = -1;
    // Complete token span of the scorer query: an explicit pflash_query
    // (benchmark override) mapped against the decoded token text, or the
    // tail of the latest user turn (chat default). The strict selector keeps
    // the whole span mandatory; the scorer window is its tail.
    PFlashTokenSpan query_span{-1, -1};
    std::string query_span_rule;
    // Assistant and tool turns after the query's turn are context the query
    // scores, not a kept suffix (strict selection only).
    bool query_suffix_candidates = false;
    // Earlier user questions of a multi-turn chat, most recent first: they
    // score the context alongside the current query at halving weights.
    std::vector<PFlashTokenSpan> history_query_spans;
    // The window a later request of this chat scores for the latest question
    // as a history query, when this request does not score it (below).
    std::vector<PFlashTokenSpan> keep_query_spans;
    // The latest user turn's tail (PFLASH_SELECT_QUERY_TOKENS), a second
    // query window next to the prompt-end query: literal strings in the
    // question (an identifier, a described function) match their passage.
    PFlashTokenSpan turn_query_span{-1, -1};
    // Header ("<|im_start|>user\n") opening the query's turn, when the chat
    // markers resolved it — pinned mandatory so a compressed prompt keeps
    // the current turn's role envelope.
    PFlashTokenSpan query_role_header{-1, -1};
    std::vector<PFlashTokenSpan> required_instruction_spans;
    // PFLASH_SELECT_QUERY_PROBE=1: the latest human user messages' content
    // spans for the request probe (nothing is computed with the flag off).
    PFlashRequestMessages request_probe_messages;
    // Kept verbatim like the required spans. The system prompt never loses
    // its pin: when it alone does not fit the context the request fails.
    // Developer messages and tool definitions that do not fit are scored
    // like any other context.
    std::vector<PFlashTokenSpan> system_spans;
    std::vector<PFlashTokenSpan> instruction_role_spans;
    // Chat-first scorer query: the latest user turn's content span, located
    // by the rendered prompt's own control markers. Feeds the strict tail
    // parser and the legacy window; unused when a benchmark parser
    // (latest_user) or a marker-less prompt needs the sentinel mapping.
    // PFLASH_SELECT_QUERY_HEAD=1 under the chat parser: the latest human
    // message's content span (the query head's source).
    PFlashTokenSpan human_query_content{-1, -1};
    http_detail::PflashChatTurnSpan chat_turn;
    // PFLASH_UNIFIED=1: the normalised messages the prompt was rendered from (unused otherwise).
    std::vector<ChatMessage> unified_messages;
    if (!experiment.configured || tail_parser) {
        ChatMarkers chat_markers;
        if (resolve_chat_markers(tokenizer_, chat_markers)) {
            chat_turn = http_detail::pflash_chat_query_turn(
                tokenizer_, chat_markers, *drafter_tokenizer_, drafter_ids);
        }
    }
    if (experiment.configured) {
        if (!messages_input && !raw_text_input) {
            return "PFlash strict selection input has no parseable text";
        }
        try {
            auto messages =
                normalize_chat_messages(req.messages, req.format, tool_memory_,
                                        config_.chat_template_src.empty());
            if (messages.empty()) {
                return "PFlash strict selection normalized messages are empty";
            }
            if (pflash_unified_) unified_messages = messages;

            int last_user_index = -1;
            for (int index = (int) messages.size() - 1; index >= 0; --index) {
                if (messages[(size_t) index].role == "user") {
                    last_user_index = index;
                    break;
                }
            }
            if (last_user_index >= 0) {
                last_user_text = messages[(size_t) last_user_index].content;
            }

            const bool semantic_parser = experiment.query_parser ==
                luce::pflash::PFlashQueryParser::SemanticUser;
            // The latest user message bounds the query; the chat parser
            // falls back to the last message when there is none.
            int boundary_index = (int) messages.size() - 1;
            if (!raw_text_input && (semantic_parser || last_user_index >= 0)) {
                boundary_index = last_user_index;
            }

            static constexpr const char * kContentBegin =
                "__LUCE_PFLASH_CONTENT_BEGIN_02C47F91__";
            static constexpr const char * kContentEnd =
                "__LUCE_PFLASH_CONTENT_END_6E6B61A8__";
            const auto map_message_content = [&] (
                    size_t message_index,
                    int & content_begin,
                    int & content_end,
                    std::string & boundary_error) -> bool {
                auto begin_messages = messages;
                begin_messages[message_index].content =
                    std::string(kContentBegin) +
                    begin_messages[message_index].content;
                auto end_messages = messages;
                end_messages[message_index].content += kContentEnd;

                std::string begin_rendered;
                std::string end_rendered;
                if (!render_messages_to_text(
                        begin_messages, req, /*add_generation_prompt=*/true,
                        begin_rendered, boundary_error)) {
                    boundary_error = "content-start render failed: " +
                        boundary_error;
                    return false;
                }
                boundary_error.clear();
                if (!render_messages_to_text(
                        end_messages, req, /*add_generation_prompt=*/true,
                        end_rendered, boundary_error)) {
                    boundary_error = "content-end render failed: " +
                        boundary_error;
                    return false;
                }

                const auto begin_ids = drafter_tokenizer_->encode(
                    tokenizer_.decode(tokenizer_.encode(begin_rendered)));
                const auto end_ids = drafter_tokenizer_->encode(
                    tokenizer_.decode(tokenizer_.encode(end_rendered)));
                content_begin =
                    http_detail::pflash_query_search_begin_from_sentinel(
                        drafter_ids, begin_ids);
                content_end =
                    http_detail::pflash_query_search_end_from_sentinel(
                        drafter_ids, end_ids);
                if (content_begin < 0 || content_end <= content_begin ||
                    content_end >= (int) drafter_ids.size()) {
                    boundary_error = "content boundary mapping failed";
                    return false;
                }
                return true;
            };
            const auto map_rendered_message = [&] (
                    size_t message_index,
                    PFlashTokenSpan & message_span,
                    std::string & boundary_error) -> bool {
                auto without_message = messages;
                without_message.erase(without_message.begin() + message_index);

                std::string without_message_rendered;
                if (!render_messages_to_text(
                        without_message, req, /*add_generation_prompt=*/true,
                        without_message_rendered, boundary_error)) {
                    boundary_error = "message-removal render failed: " +
                        boundary_error;
                    return false;
                }
                const auto without_message_ids = drafter_tokenizer_->encode(
                    tokenizer_.decode(tokenizer_.encode(
                        without_message_rendered)));
                message_span = http_detail::pflash_changed_token_span(
                    drafter_ids, without_message_ids);
                if (message_span.begin < 0) {
                    boundary_error =
                        "complete rendered message did not map to a prompt span";
                    return false;
                }
                return true;
            };

            std::string boundary_error;
            // Chat default: the latest user turn's content bounds come from
            // the rendered prompt's control markers — no sentinel re-renders.
            if (tail_parser && chat_turn.valid()) {
                query_content_begin = chat_turn.content_begin;
                query_content_end = chat_turn.content_end;
                if (chat_turn.role_begin >= 0 &&
                    chat_turn.role_begin < chat_turn.content_begin) {
                    query_role_header = {chat_turn.role_begin,
                                         chat_turn.content_begin};
                }
            }
            if (query_content_begin < 0) {
                // Marker-less prompts (and the benchmark's latest_user
                // parser) still locate the boundary through sentinel
                // renders of the boundary message.
                if (boundary_index < 0 ||
                    (semantic_parser && !raw_text_input &&
                     last_user_text.empty())) {
                    return "PFlash strict selection latest-user boundary is unavailable";
                }
                if (!map_message_content(
                        (size_t) boundary_index,
                        query_content_begin, query_content_end,
                        boundary_error)) {
                    return "PFlash strict selection " + boundary_error;
                }
            }
            if (query_content_begin < 0 ||
                query_content_end <= query_content_begin ||
                query_content_end > (int) drafter_ids.size()) {
                return "PFlash strict selection content boundary mapping failed";
            }
            // Chat default: without an explicit pflash_query the scorer
            // query is the prompt's last token -- where the model starts
            // answering, having read the whole request wherever the question
            // sits in it. Nothing is parsed out of the user's text: the
            // latest turn is scored like the rest of the conversation. A
            // prompt without chat markers falls back to its content's tail.
            if (tail_parser && req.pflash_query.empty()) {
                const int prompt_end = (int) drafter_ids.size();
                if (chat_turn.valid() &&
                    chat_turn.generation_begin > 0 &&
                    chat_turn.generation_begin < prompt_end) {
                    query_span = {prompt_end - 1, prompt_end};
                    query_span_rule = "prompt_end";
                    const auto tail = http_detail::pflash_tail_query_window(
                        drafter_ids, experiment.query_tokens,
                        chat_turn.content_end, chat_turn.content_begin);
                    if (tail.valid()) {
                        turn_query_span = {tail.end - tail.tokens, tail.end};
                    }
                } else {
                    const auto window = http_detail::pflash_tail_query_window(
                        drafter_ids, experiment.query_tokens,
                        query_content_end, query_content_begin);
                    if (window.valid()) {
                        query_span = {window.end - window.tokens, window.end};
                        query_span_rule =
                            raw_text_input ? "content_tail" : "prompt_tail";
                    }
                }
            }
            const bool prompt_end_query = query_span_rule == "prompt_end";

            if (experiment.selection_active) {
                const auto instruction_plan =
                    http_detail::plan_pflash_instruction_messages(messages);
                for (size_t instruction_index :
                        instruction_plan.instruction_messages) {
                    PFlashTokenSpan instruction_span;
                    boundary_error.clear();
                    if (!map_rendered_message(
                            instruction_index, instruction_span,
                            boundary_error)) {
                        return "PFlash strict selection instruction mapping failed: " +
                            boundary_error;
                    }
                    (messages[instruction_index].role == "system"
                        ? system_spans : instruction_role_spans)
                        .push_back(instruction_span);
                }

                if (!req.tools.is_null() && !req.tools.empty()) {
                    ParsedRequest tool_free_req = req;
                    tool_free_req.tools = json::array();
                    std::string tool_free_rendered;
                    boundary_error.clear();
                    if (!render_messages_to_text(
                            messages, tool_free_req,
                            /*add_generation_prompt=*/true,
                            tool_free_rendered, boundary_error)) {
                        return "PFlash strict selection tool mapping failed: " +
                            boundary_error;
                    }
                    const auto tool_free_ids = drafter_tokenizer_->encode(
                        tokenizer_.decode(tokenizer_.encode(tool_free_rendered)));
                    const PFlashTokenSpan tool_span =
                        http_detail::pflash_changed_token_span(
                            drafter_ids, tool_free_ids);
                    if (tool_span.begin < 0) {
                        return "PFlash strict selection tool mapping failed: "
                            "tools did not produce a retained prompt span";
                    }
                    instruction_role_spans.push_back(tool_span);
                }

                // Client-declared literal text that must survive compression
                // (e.g. an answer-format directive embedded in the user
                // message). Each string maps to its last occurrence inside
                // the boundary content; mapping failure fails the request
                // rather than silently keeping a shorter span.
                for (const auto & required : req.pflash_required) {
                    if (required.empty()) continue;
                    const PFlashTokenSpan required_span =
                        http_detail::pflash_decoded_text_span(
                            *drafter_tokenizer_, drafter_ids,
                            query_content_begin, query_content_end, required);
                    if (required_span.begin < 0) {
                        return "PFlash strict selection required-text mapping "
                            "failed: a pflash_required string does not occur "
                            "in the latest user content";
                    }
                    required_instruction_spans.push_back(required_span);
                }
                // An explicit scorer query replaces the chat-derived one and
                // pins its complete span: the whole question is mandatory
                // even though the scorer only consumes its bounded tail.
                // Mapping against the decoded content text (not a standalone
                // encoding) keeps BPE boundary merges like " What" inside the
                // span. Benchmark-only: under the chat tail parser the query
                // may sit anywhere before the user turn's closing marker;
                // latest_user still scopes it to the user message.
                if (!req.pflash_query.empty()) {
                    const int query_search_begin =
                        semantic_parser ? query_content_begin : 0;
                    query_span = http_detail::pflash_decoded_text_span(
                        *drafter_tokenizer_, drafter_ids,
                        query_search_begin, query_content_end,
                        req.pflash_query);
                    query_span_rule = "explicit_query_span";
                    if (query_span.begin < 0) {
                        return semantic_parser
                            ? "PFlash strict selection explicit query mapping "
                              "failed: pflash_query does not occur in the "
                              "latest user content"
                            : "PFlash strict selection explicit query mapping "
                              "failed: pflash_query does not occur in the "
                              "prompt";
                    }
                }
                if (query_span.begin >= 0) {
                    required_instruction_spans.push_back(query_span);
                }
                if (prompt_end_query) {
                    // The whole generation prompt stays verbatim; the query
                    // is its last token.
                    required_instruction_spans.push_back(
                        {chat_turn.generation_begin, (int) drafter_ids.size()});
                }
                if (query_role_header.begin >= 0) {
                    required_instruction_spans.push_back(query_role_header);
                }
                // An agent loop puts assistant and tool turns after the
                // user's: they compete for the budget like the context before
                // the query. What stays is the rest of the query's turn
                // through its closing marker, and the generation prompt.
                if (chat_turn.valid() && chat_turn.later_turns &&
                    query_span.begin >= chat_turn.content_begin &&
                    query_span.end <= chat_turn.content_end) {
                    query_suffix_candidates = true;
                    if (chat_turn.turn_end > query_span.end) {
                        required_instruction_spans.push_back(
                            {query_span.end, chat_turn.turn_end});
                    }
                    if (chat_turn.generation_begin < (int) drafter_ids.size()) {
                        required_instruction_spans.push_back(
                            {chat_turn.generation_begin,
                             (int) drafter_ids.size()});
                    }
                }
                // Multi-turn skeleton: every other turn keeps its role
                // header, and short user turns and assistant answers stay
                // whole -- what the conversation said rather than the
                // material it quoted. Like instructions, they are scored as
                // context when they alone would not fit.
                if (chat_turn.valid()) {
                    const int skeleton_tokens =
                        http_detail::pflash_chat_skeleton_tokens();
                    for (size_t index = 0; index < chat_turn.turns.size();
                         ++index) {
                        // With a prompt-end query the latest user turn is
                        // no longer pinned as the query: it follows the same
                        // rule as every other turn.
                        if ((int) index == chat_turn.query_turn &&
                            !prompt_end_query) {
                            continue;
                        }
                        const auto & turn = chat_turn.turns[index];
                        if (turn.role == "system") continue;
                        if (turn.content_begin > turn.role_begin) {
                            instruction_role_spans.push_back(
                                {turn.role_begin, turn.content_begin});
                        }
                        const bool conversational = turn.role == "user" ||
                            turn.role == "assistant" || turn.role == "model";
                        if (conversational && skeleton_tokens > 0 &&
                            turn.content_end - turn.content_begin <=
                                skeleton_tokens &&
                            turn.turn_end > turn.role_begin) {
                            instruction_role_spans.push_back(
                                {turn.role_begin, turn.turn_end});
                        }
                    }
                    const size_t history_queries =
                        (size_t) http_detail::pflash_chat_history_queries();
                    for (int index = chat_turn.query_turn - 1;
                         index >= 0 &&
                             history_query_spans.size() < history_queries;
                         --index) {
                        const auto & turn = chat_turn.turns[(size_t) index];
                        if (turn.role != "user") continue;
                        if (prompt_end_query) {
                            // The earlier question's counterpart of the
                            // prompt's last token: the last token of the
                            // header of the reply that followed it.
                            const size_t reply = (size_t) index + 1;
                            if (reply < chat_turn.turns.size() &&
                                chat_turn.turns[reply].role != "user" &&
                                chat_turn.turns[reply].content_begin >
                                    chat_turn.turns[reply].role_begin) {
                                const int end = chat_turn.turns[reply].content_begin;
                                history_query_spans.push_back({end - 1, end});
                            }
                            continue;
                        }
                        const auto window = http_detail::pflash_tail_query_window(
                            drafter_ids, experiment.query_tokens,
                            turn.content_end, turn.content_begin);
                        if (window.valid()) {
                            history_query_spans.push_back(
                                {window.end - window.tokens, window.end});
                        }
                    }
                    // A tool session's prompt ends after tool output, so the
                    // latest question's history window (the header of the
                    // reply that followed it) is no query here. The next
                    // question scores it; the drafter keeps its rows so that
                    // request resumes the scoring session instead of
                    // scoring the whole prompt again.
                    const size_t reply = (size_t) chat_turn.query_turn + 1;
                    if (prompt_end_query && history_queries > 0 &&
                        chat_turn.query_turn >= 0 &&
                        reply < chat_turn.turns.size() &&
                        chat_turn.turns[reply].role != "user" &&
                        chat_turn.turns[reply].content_begin >
                            chat_turn.turns[reply].role_begin) {
                        const int end = chat_turn.turns[reply].content_begin;
                        keep_query_spans.push_back({end - 1, end});
                    }
                }
                // The last non-system turns before the generation prompt
                // (latest first), from the rendered prompt's chat markers:
                // request-probe candidates (V2-A3) and the identifier query.
                http_detail::PflashChatTurnSpan marker_turns;
                bool marker_turns_ready = false;
                const auto recent_turns = [&] (int count) {
                    if (!marker_turns_ready) {
                        marker_turns = chat_turn;
                        if (marker_turns.turns.empty()) {
                            ChatMarkers recent_markers;
                            if (resolve_chat_markers(tokenizer_, recent_markers)) {
                                marker_turns = http_detail::pflash_chat_query_turn(
                                    tokenizer_, recent_markers, *drafter_tokenizer_, drafter_ids);
                            }
                        }
                        marker_turns_ready = true;
                    }
                    std::vector<PFlashTokenSpan> regions;
                    for (size_t index = marker_turns.turns.size();
                         index-- > 0 && (int) regions.size() < count;) {
                        const auto & turn = marker_turns.turns[index];
                        if (turn.role == "system") continue;
                        if (marker_turns.generation_begin > 0 &&
                            turn.role_begin >= marker_turns.generation_begin) {
                            continue;
                        }
                        if (turn.content_begin < 0 || turn.content_end <= turn.content_begin ||
                            turn.content_end > (int) drafter_ids.size()) {
                            continue;
                        }
                        regions.push_back({turn.content_begin, turn.content_end});
                    }
                    return regions;
                };
                // The content spans of the latest ``count`` human messages
                // (role "user" after normalisation: tool results are role
                // "tool" from every API format, also when the template renders
                // them inside a user turn), latest first: by the rendered
                // prompt's chat markers, else by sentinel renders.
                std::vector<size_t> mapped_human;   // indices of the last map_human_messages call
                const auto map_human_messages = [&] (
                        size_t count, std::vector<PFlashTokenSpan> & out,
                        std::string & error) -> const char * {
                    std::vector<size_t> human;
                    for (size_t index = messages.size();
                         index-- > 0 && human.size() < count;) {
                        if (messages[index].role == "user") human.push_back(index);
                    }
                    mapped_human = human;
                    recent_turns(0);   // resolves marker_turns
                    if (http_detail::pflash_human_message_spans(
                            *drafter_tokenizer_, drafter_ids, marker_turns.turns, messages,
                            human.size(), out)) {
                        return "markers";
                    }
                    out.clear();
                    for (size_t index : human) {
                        const std::string & content = messages[index].content;
                        if (content.find_first_not_of(" \t\n\r") == std::string::npos) {
                            out.push_back({0, 0});
                            continue;
                        }
                        int begin = -1;
                        int end = -1;
                        std::string mapping_error;
                        if (!map_message_content(index, begin, end, mapping_error)) {
                            out.clear();
                            error = mapping_error;
                            break;
                        }
                        out.push_back({begin, end});
                    }
                    return "sentinel";
                };
                // V2-A3 candidates: the content spans of the last ``count``
                // non-system messages (latest first), found in render order
                // from the generation prompt backwards: each message's content
                // (trimmed) must occur before the previous match.
                const auto map_recent_messages = [&] (
                        int count, std::vector<PFlashTokenSpan> & spans,
                        std::vector<int> & indices, std::string & error) {
                    recent_turns(0);
                    const http_detail::DecodedPrompt decoded =
                        http_detail::decode_prompt_with_offsets(*drafter_tokenizer_, drafter_ids);
                    size_t cursor = decoded.text.size();
                    if (marker_turns.generation_begin > 0 &&
                        marker_turns.generation_begin < (int) drafter_ids.size()) {
                        cursor = decoded.token_begin[(size_t) marker_turns.generation_begin];
                    }
                    for (size_t index = messages.size();
                         index-- > 0 && (int) spans.size() < count;) {
                        if (messages[index].role == "system") continue;
                        const std::string & raw = messages[index].content;
                        const size_t lead = raw.find_first_not_of(" \t\n\r");
                        if (lead == std::string::npos) {
                            spans.push_back({0, 0});
                            indices.push_back((int) index);
                            continue;
                        }
                        const size_t tail = raw.find_last_not_of(" \t\n\r");
                        const std::string content = raw.substr(lead, tail - lead + 1);
                        if (content.size() > cursor) {
                            error = "recent message content does not fit before the next one";
                            spans.clear();
                            indices.clear();
                            return;
                        }
                        const size_t at = decoded.text.rfind(content, cursor - content.size());
                        if (at == std::string::npos) {
                            error = "recent message content not found in render order";
                            spans.clear();
                            indices.clear();
                            return;
                        }
                        const int begin = http_detail::token_at_offset(decoded, at);
                        const int end = (int) (std::lower_bound(
                            decoded.token_begin.begin(), decoded.token_begin.end(),
                            at + content.size()) - decoded.token_begin.begin());
                        spans.push_back({begin, end});
                        indices.push_back((int) index);
                        cursor = at;
                    }
                };
                if (experiment.query_probe) {
                    const char * mapping = map_human_messages(
                        luce::pflash::kRequestMessages, request_probe_messages.human,
                        request_probe_messages.error);
                    for (size_t index : mapped_human) {
                        request_probe_messages.human_index.push_back((int) index);
                    }
                    map_recent_messages(INT_MAX,   // every non-system message (V2-A5)
                                        request_probe_messages.recent,
                                        request_probe_messages.recent_index,
                                        request_probe_messages.recent_error);
                    // cand_v5 / stage 2 (STAGE2-CONTRACT.md section 1): every message's role and the tokens
                    // overlapping its exact content, from one render with a placeholder per content (read only
                    // by a cand_v5 probe; any failure falls the request back to today's windows).
                    {
                        auto & all = request_probe_messages;
                        std::vector<ChatMessage> placeholder_messages = messages;
                        std::vector<std::string> placeholders(messages.size());
                        std::vector<std::string> contents(messages.size());
                        for (size_t index = 0; index < messages.size() && all.all_error.empty(); ++index) {
                            const std::string & role = messages[index].role;
                            if (role != "system" && role != "user" && role != "assistant" && role != "tool") {
                                all.all_error = "unsupported message role " + role;
                                break;
                            }
                            all.all_roles.push_back(role);
                            contents[index] = messages[index].content;
                            if (role == "system") continue;   // never a candidate nor in E
                            placeholders[index] = "\x1e" "PFLASH-MESSAGE-" + std::to_string(index) + "\x1f";
                            placeholder_messages[index].content = placeholders[index];
                        }
                        std::string placeholder_render;
                        if (all.all_error.empty() &&
                            !render_messages_to_text(placeholder_messages, req, /*add_generation_prompt=*/true,
                                                     placeholder_render, all.all_error)) {
                            all.all_error = "placeholder render failed: " + all.all_error;
                        }
                        if (all.all_error.empty()) {
                            const http_detail::DecodedPrompt decoded =
                                http_detail::decode_prompt_with_offsets(*drafter_tokenizer_, drafter_ids);
                            std::vector<std::pair<size_t, size_t>> chars;
                            if (http_detail::pflash_content_char_spans(placeholder_render, placeholders, contents,
                                                                       decoded.text, chars, all.all_error)) {
                                for (const auto & span : chars) {
                                    all.all_spans.push_back(http_detail::pflash_char_span_tokens(
                                        decoded.token_begin, decoded.text.size(), span.first, span.second));
                                }
                            }
                        }
                        if (!all.all_error.empty()) {
                            all.all_roles.clear();
                            all.all_spans.clear();
                        }
                    }
                    std::string spans;
                    for (const auto & span : request_probe_messages.human) {
                        spans += " [" + std::to_string(span.begin) + "," +
                            std::to_string(span.end) + ")";
                    }
                    std::string turn_spans;
                    for (const auto & span : request_probe_messages.recent) {
                        turn_spans += " [" + std::to_string(span.begin) + "," +
                            std::to_string(span.end) + ")";
                    }
                    std::fprintf(stderr,
                        "[pflash-request-probe] human messages=%zu mapped by %s:%s%s%s recent:%s\n",
                        request_probe_messages.human.size(), mapping,
                        spans.empty() ? " none" : spans.c_str(),
                        request_probe_messages.error.empty() ? "" : " error: ",
                        request_probe_messages.error.c_str(),
                        turn_spans.empty() ? " none" : turn_spans.c_str());
                    std::fflush(stderr);
                }
                // Query head under the chat parser (V2-A3): the latest human
                // message's prose blocks, wherever it sits (an agent's task
                // far back, a chat's last turn).
                if (experiment.query_head && tail_parser) {
                    std::vector<PFlashTokenSpan> latest;
                    std::string head_error;
                    map_human_messages(1, latest, head_error);
                    if (!latest.empty() && latest[0].end > latest[0].begin) {
                        human_query_content = latest[0];
                    }
                }
                if (experiment.query_ident != luce::pflash::PFlashIdentQuery::Off) {
                    // Identifier query: the content of the last turns that
                    // precede the generation prompt (system turns excluded),
                    // markup tokens blanked (pflash_ident_query.h).
                    const std::vector<PFlashTokenSpan> regions =
                        recent_turns(luce::pflash::kIdentQueryTurns);
                    if (regions.empty()) {
                        request_probe_messages.ident_error = "no turns mapped";
                    } else {
                        std::vector<std::string> texts(drafter_ids.size());
                        std::vector<unsigned char> markup(drafter_ids.size(), 0);
                        for (const auto & region : regions) {
                            for (int t = region.begin; t < region.end; ++t) {
                                const int32_t id = drafter_ids[(size_t) t];
                                texts[(size_t) t] = drafter_tokenizer_->token_text(id);
                                markup[(size_t) t] = drafter_tokenizer_->is_added_token(id) ? 1 : 0;
                            }
                        }
                        request_probe_messages.ident_turns = (int) regions.size();
                        request_probe_messages.ident_tokens =
                            luce::pflash::pflash_ident_query_tokens(texts, markup, regions);
                    }
                    std::string spans;
                    for (const auto & region : regions) {
                        spans += " [" + std::to_string(region.begin) + "," +
                            std::to_string(region.end) + ")";
                    }
                    std::fprintf(stderr,
                        "[pflash-ident-query] turns:%s identifier tokens=%zu%s%s\n",
                        spans.empty() ? " none" : spans.c_str(),
                        request_probe_messages.ident_tokens.size(),
                        request_probe_messages.ident_error.empty() ? "" : " error: ",
                        request_probe_messages.ident_error.c_str());
                    std::fflush(stderr);
                }
                required_instruction_spans =
                    http_detail::canonicalize_pflash_token_spans(
                        std::move(required_instruction_spans));
                instruction_role_spans =
                    http_detail::canonicalize_pflash_token_spans(
                        std::move(instruction_role_spans));
                system_spans =
                    http_detail::canonicalize_pflash_token_spans(
                        std::move(system_spans));
                std::string instruction_error;
                if (!luce::pflash::validate_pflash_instruction_spans(
                        required_instruction_spans,
                        (int) drafter_ids.size(), instruction_error) ||
                    !luce::pflash::validate_pflash_instruction_spans(
                        instruction_role_spans,
                        (int) drafter_ids.size(), instruction_error) ||
                    !luce::pflash::validate_pflash_instruction_spans(
                        system_spans,
                        (int) drafter_ids.size(), instruction_error)) {
                    return "PFlash strict selection instruction mapping failed: " +
                        instruction_error;
                }
            }
        } catch (const std::exception & error) {
            return std::string("PFlash retention normalization failed: ") +
                   error.what();
        }
    } else if (raw_text_input) {
        last_user_text = req.messages.get<std::string>();
    } else if (req.messages.is_array()) {
        for (int index = (int) req.messages.size() - 1; index >= 0; --index) {
            if (req.messages[index].value("role", "") != "user") continue;
            const auto & content = req.messages[index]["content"];
            if (content.is_string()) {
                last_user_text = content.get<std::string>();
            } else if (content.is_array()) {
                for (const auto & part : content) {
                    const std::string type = part.value("type", "");
                    if (type == "text" || type == "input_text" ||
                        type == "output_text") {
                        last_user_text += part.value("text", "");
                    }
                }
            }
            break;
        }
    }

    std::vector<int32_t> semantic_query_ids;
    std::vector<int32_t> expected_query_ids;
    http_detail::PflashQueryWindow query_window;
    if (!req.pflash_query.empty()) {
        // An explicit query replaces the message-tail heuristic; it must occur
        // inside the latest user content so the window maps onto real tokens.
        last_user_text = req.pflash_query;
        parser_selection_rule = "explicit_query";
    }
    if (!last_user_text.empty()) {
        semantic_query_ids = drafter_tokenizer_->encode(last_user_text);
    }
    // Unconfigured (legacy) chat mode derives the query the same way, from
    // the latest user turn's tail.
    if (!experiment.configured && req.pflash_query.empty() &&
        chat_turn.valid()) {
        const auto window = http_detail::pflash_tail_query_window(
            drafter_ids, experiment.query_tokens,
            chat_turn.content_end, chat_turn.content_begin);
        if (window.valid()) {
            query_span = {window.end - window.tokens, window.end};
            query_span_rule = "chat_user_tail";
        }
    }
    if (query_span.begin >= 0) {
        // The query span — explicit or chat-derived — was mapped onto the
        // prompt's own tokens and, under strict selection, pinned mandatory.
        // The scorer consumes the span's bounded tail window; the complete
        // span stays in the target prompt.
        parser_selection_rule = query_span_rule;
        query_window.end = query_span.end;
        query_window.tokens = (std::min)(
            experiment.query_tokens, query_span.end - query_span.begin);
        expected_query_ids.assign(
            drafter_ids.begin() + (query_window.end - query_window.tokens),
            drafter_ids.begin() + query_window.end);
    } else if (experiment.configured && (raw_text_input || tail_parser)) {
        // Content located, but no query span (a parser without a derived
        // query, or selection inactive): score the content's tail.
        parser_selection_rule = raw_text_input ? "content_tail" : "prompt_tail";
        query_window = http_detail::pflash_tail_query_window(
            drafter_ids, experiment.query_tokens,
            query_content_end, query_content_begin);
    } else if (!semantic_query_ids.empty()) {
        if (experiment.configured) parser_selection_rule = "semantic_suffix";
        query_window = http_detail::find_pflash_query_window(
            drafter_ids, semantic_query_ids,
            experiment.configured ? experiment.query_tokens : 8,
            query_content_end,
            experiment.configured ? query_content_begin : 0,
            /*anchored=*/ req.pflash_query.empty());
        if (experiment.configured && query_window.valid()) {
            const auto matched_end =
                semantic_query_ids.end() - query_window.trailing_trimmed;
            expected_query_ids.assign(
                matched_end - query_window.tokens, matched_end);
        }
    }

    // Query head (PFLASH_SELECT_QUERY_HEAD=1): the user's own prose around
    // pasted material -- a question before the paste, the one after it --
    // becomes query windows, kept verbatim. The trailing prose block replaces
    // the fixed-width tail window; without paste structure nothing changes.
    std::vector<PFlashTokenSpan> extra_query_spans;
    // Under the chat parser the query stays the prompt's end (and the latest
    // turn's tail); the head plan runs on the latest HUMAN message and its
    // blocks -- the trailing one included -- become extra query windows,
    // kept verbatim. Its trailing block replaces the turn-tail window when
    // that window lies inside the same message (a chat's last turn).
    const bool chat_query_head = tail_parser && human_query_content.begin >= 0;
    const int head_begin = chat_query_head ? human_query_content.begin : query_content_begin;
    const int head_end = chat_query_head ? human_query_content.end : query_content_end;
    if (experiment.selection_active && experiment.query_head &&
        req.pflash_query.empty() && query_window.valid() &&
        (chat_query_head || !tail_parser) &&
        head_begin >= 0 && head_end > head_begin) {
        const auto plan = http_detail::pflash_query_head_plan(
            *drafter_tokenizer_, drafter_ids, head_begin,
            head_end, experiment.query_head_tokens,
            experiment.query_tail_tokens);
        if (chat_query_head) {
            auto blocks = plan.extra;
            if (plan.tail.begin >= 0 && plan.tail.end > plan.tail.begin) {
                blocks.push_back(plan.tail);
                if (turn_query_span.begin >= 0 && turn_query_span.begin < head_end &&
                    turn_query_span.end > head_begin) {
                    turn_query_span = {-1, -1};
                }
            }
            for (auto & span : blocks) extra_query_spans.push_back(span);
            extra_query_spans = http_detail::canonicalize_pflash_token_spans(
                std::move(extra_query_spans));
        } else if (plan.tail.begin >= 0 && plan.tail.end > plan.tail.begin) {
            query_window.end = plan.tail.end;
            query_window.tokens = plan.tail.end - plan.tail.begin;
            query_window.trailing_trimmed = 0;
            expected_query_ids.assign(drafter_ids.begin() + plan.tail.begin,
                                      drafter_ids.begin() + plan.tail.end);
        }
        const int query_begin = query_window.end - query_window.tokens;
        if (chat_query_head) {
            std::vector<PFlashTokenSpan> inside;
            for (const auto & span : extra_query_spans) {
                if (span.end <= query_begin && span.end > span.begin) inside.push_back(span);
            }
            extra_query_spans = std::move(inside);
        }
        for (const auto & span : chat_query_head ? std::vector<PFlashTokenSpan>{} : plan.extra) {
            if (span.end <= query_begin && span.end > span.begin) {
                extra_query_spans.push_back(span);
            }
        }
        if (!extra_query_spans.empty()) {
            auto pinned = required_instruction_spans;
            pinned.insert(pinned.end(), extra_query_spans.begin(), extra_query_spans.end());
            pinned = http_detail::canonicalize_pflash_token_spans(std::move(pinned));
            std::string pin_error;
            if (!luce::pflash::validate_pflash_instruction_spans(
                    pinned, (int) drafter_ids.size(), pin_error)) {
                return "PFlash query head mapping failed: " + pin_error;
            }
            required_instruction_spans = std::move(pinned);
        }
        std::string windows;
        for (const auto & span : extra_query_spans) {
            windows += " [" + std::to_string(span.begin) + "," + std::to_string(span.end) + ")";
        }
        std::fprintf(stderr,
            "[pflash-query-head] structured=%d head=%s/%d tail=%d query=[%d,%d) extra=%zu%s\n",
            (int) plan.structured, plan.head_rule.c_str(), plan.head_tokens,
            plan.tail_tokens, query_begin, query_window.end,
            extra_query_spans.size(), windows.c_str());
        std::fflush(stderr);
    }

    if (pflash_unified_) {
        // The unified view takes over after the shared query and pins.
        ModelBackend::CompressRequest unified_request;
        auto pins = required_instruction_spans;
        pins.insert(pins.end(), system_spans.begin(), system_spans.end());
        pins.insert(pins.end(), instruction_role_spans.begin(), instruction_role_spans.end());
        unified_request.required_instruction_spans =
            http_detail::canonicalize_pflash_token_spans(std::move(pins));
        unified_request.query_suffix_candidates = query_suffix_candidates;
        unified_request.history_query_spans = history_query_spans;
        unified_request.keep_query_spans = keep_query_spans;
        unified_request.turn_query_span = turn_query_span;
        unified_request.extra_query_spans = extra_query_spans;
        unified_request.request_messages = request_probe_messages;
        if (query_window.valid()) {
            unified_request.score_query_end = query_window.end;
            unified_request.score_query_tokens = query_window.tokens;
        }
        unified_request.input_ids = drafter_ids;
        std::string not_ready;
        if (!messages_input) not_ready = "not_messages";
        else if (!experiment.selection_active) not_ready = "no_strict_selection";
        else if (!query_window.valid()) not_ready = "no_query";
        else if (!chat_turn.valid()) not_ready = "turns_unparsed";
        else if (config_.pflash_remote_drafter) not_ready = "remote_drafter";
        else if (!http_detail::pflash_selection_gives_lifts(experiment)) not_ready = "unscored_selection";
        else if (unified_messages.empty()) not_ready = "messages_unreadable";
        std::string handback;
        auto unified_error = apply_unified_view(req, prepared, chat_turn, unified_messages,
                                                std::move(unified_request), experiment.chunk_size, not_ready,
                                                handback);
        if (handback.empty()) return unified_error;
        // Handed back: the chat view below serves this request as it would without PFLASH_UNIFIED.
        prepared.unified_handback = handback;
    }

    // Strict selection spends the keep ratio on the droppable tokens only:
    // what it keeps anyway (instructions, tools, the query and its turn's
    // envelope, the generation prompt) is already cheap -- a stable system
    // prefix hits the prefix cache from the second turn on -- and must not
    // exhaust the budget of the history it rides with.
    int kept_tokens = 0;
    json admission_stats;   // cache-aware admission inputs (null when off)
    http_detail::PflashRegionPlan region_plan;
    // A tool session's view (PFLASH_AGENT_PASTE=1): it only continues
    // (serve_pflash_chat_view keep_history).
    bool keep_history_view = false;
    // Tool sessions under cache admission: PFlash never compresses the
    // agent's history or tool output. Tool activity (tools defined, or an
    // assistant tool call / tool result in the history) is read once and
    // only when a rule asks for it.
    const bool tool_session_rules =
        config_.pflash_mode == ServerConfig::PflashMode::AUTO &&
        http_detail::pflash_cache_admission_enabled();
    const bool raw_fits = config_.max_ctx <= 0 ||
        prompt_tokens + req.max_output <= config_.max_ctx;
    int tool_activity_state = -1;   // -1 not read, 0 none, 1 tools
    std::string tool_error;
    const auto tool_activity = [&] {
        if (tool_activity_state < 0) {
            tool_activity_state = 0;
            // Tools defined: a tool session whatever the messages.
            if (http_detail::pflash_has_tool_activity(req.tools, {})) {
                tool_activity_state = 1;
                return true;
            }
            try {
                tool_activity_state = http_detail::pflash_has_tool_activity(
                    req.tools,
                    // No tool-memory replay: only roles and tool calls are
                    // read, and the lookup must not touch the memory a
                    // second time.
                    normalize_chat_messages(
                        req.messages, req.format, tool_memory_,
                        /*replay_tool_memory=*/false)) ? 1 : 0;
            } catch (const std::exception & e) {
                // Unreadable messages: the rules do not apply.
                tool_error = e.what();
            }
        }
        return tool_activity_state == 1;
    };
    const auto count_keep_history = [this] (const char * decision) {
        std::lock_guard<std::mutex> lk(pflash_failure_mu_);
        ++pflash_keep_history_counts_[decision];
    };
    // The raw prompt, full prefill, for a tool session nothing may be
    // compressed in: counted with the raw fallbacks (reason keep_history) and
    // never refused. A raw prompt over the context then gets the
    // context-overflow error, as with full prefill. ``new_droppable`` < 0:
    // not measured.
    const auto serve_keep_raw = [&] (int new_droppable, const char * why) {
        count_keep_history("keep_history");
        {
            std::lock_guard<std::mutex> lk(pflash_failure_mu_);
            ++pflash_fallback_served_;
            ++pflash_fallback_reasons_["keep_history"];
        }
        prepared.pflash_stats = {{"admission", admission_stats}};
        prepared.pflash_stats["pflash_fallback"] = {
            {"reason", "keep_history"},
            {"prompt_tokens", prompt_tokens},
        };
        if (new_droppable >= 0) {
            prepared.pflash_stats["pflash_fallback"]["new_droppable"] = new_droppable;
        }
        std::fprintf(stderr,
            "[pflash-keep-history] serving the raw prompt (%d tokens): %s, "
            "newest user content droppable~%d, threshold %d%s\n",
            prompt_tokens, why, new_droppable, config_.pflash_threshold,
            raw_fits ? "" : "; the raw prompt does not fit the context");
        std::fflush(stderr);
    };
    // By default a tool session is served raw: nothing in it is compressed.
    // PFLASH_AGENT_PASTE=1 lets PFlash compress one thing, the newest turn
    // when it is a user message with a threshold's worth of droppable tokens
    // (a pasted document), below; that needs strict selection and a located
    // query, so without them the session is served raw as well.
    if (tool_session_rules && tool_activity() &&
        !(http_detail::pflash_agent_paste_enabled() &&
          experiment.selection_active && query_window.valid())) {
        const bool paste = http_detail::pflash_agent_paste_enabled();
        int newest = -1;
        const char * newest_kind = messages_input && chat_turn.valid()
            ? http_detail::pflash_keep_history_scope(chat_turn.turns, newest)
            : "unparsed";
        admission_stats = {{"decision", "keep_history"}, {"raw_fits", raw_fits},
                           {"keep_history", {{"in_scope", true},
                                             {"newest", newest_kind ? newest_kind : "user"},
                                             {"paste", paste}}}};
        serve_keep_raw(-1, paste ? "tool session without strict selection or a query"
                                 : "tool session (PFLASH_AGENT_PASTE off)");
        return {};
    }
    if (experiment.selection_active && query_window.valid()) {
        const int input_tokens = (int) drafter_ids.size();
        const int query_begin = query_window.end - query_window.tokens;
        const auto kept_with = [&] (
                const std::vector<PFlashTokenSpan> & spans) {
            return http_detail::pflash_kept_tokens(
                input_tokens, experiment.chunk_size, query_begin,
                query_window.end, spans, !query_suffix_candidates);
        };
        const auto target_estimate = [&] (int drafter_tokens) {
            return (int) std::ceil((double) prompt_tokens *
                (double) drafter_tokens / (double) input_tokens);
        };
        const auto merged = [&] (bool with_instructions) {
            auto spans = required_instruction_spans;
            spans.insert(spans.end(), system_spans.begin(), system_spans.end());
            if (with_instructions) {
                spans.insert(spans.end(), instruction_role_spans.begin(),
                             instruction_role_spans.end());
            }
            return http_detail::canonicalize_pflash_token_spans(
                std::move(spans));
        };
        const auto fits = [&] (int drafter_tokens) {
            return config_.max_ctx <= 0 ||
                target_estimate(drafter_tokens) + req.max_output <=
                    config_.max_ctx;
        };
        auto kept_spans = merged(/*with_instructions=*/true);
        kept_tokens = kept_with(kept_spans);
        // Developer messages and tool definitions that alone overflow the
        // context are data (a document pasted into them), not a preamble:
        // they compete for the budget against the query like any other
        // context. The system prompt keeps its pin whatever its size.
        if (!fits(kept_tokens) && !instruction_role_spans.empty()) {
            std::fprintf(stderr,
                "[pflash-select] kept instructions do not fit the context "
                "(~%d + %d > %d target tokens); scoring developer and tool "
                "spans as context\n",
                target_estimate(kept_tokens), req.max_output, config_.max_ctx);
            kept_spans = merged(/*with_instructions=*/false);
            kept_tokens = kept_with(kept_spans);
        }
        if (!fits(kept_with(system_spans))) {
            return "PFlash strict selection: the system prompt alone does not "
                "fit the context (~" +
                std::to_string(target_estimate(kept_with(system_spans))) +
                " + " + std::to_string(req.max_output) + " > " +
                std::to_string(config_.max_ctx) +
                " target tokens); PFlash does not compress system prompts";
        }
        required_instruction_spans = std::move(kept_spans);
        // Auto mode compresses when the droppable part is long enough, not
        // the whole prompt: a large system prompt plus a short chat has
        // nothing worth selecting.
        const int droppable_target =
            prompt_tokens - target_estimate(kept_tokens);
        // Cache-aware admission: a conversation PFlash has not compressed
        // yet stays raw while the reader's restorable prefix covers all but
        // a threshold's worth of its droppable tokens, or already holds a
        // threshold's worth of them. A conversation with a view keeps
        // today's rule below (the view owns its continuation).
        if (config_.pflash_mode == ServerConfig::PflashMode::AUTO &&
            http_detail::pflash_cache_admission_enabled()) {
            http_detail::PflashChatView existing;
            const bool has_view = messages_input && chat_turn.valid() &&
                pflash_views_.find(req.prompt_tokens, drafter_ids, existing);
            // A tool session here has PFLASH_AGENT_PASTE=1 (otherwise it was
            // served raw above): its history and tool output are still never
            // compressed. Only the newest turn, when it is a user message,
            // may be: its content alone, on its own budget, when its
            // droppable tokens reach the threshold. Everything else is served
            // raw, and a view is only continued. Chats without tools:
            // nothing is computed or recorded.
            json keep_history_stats;
            int keep_newest = -1;
            bool keep_scope = false;
            if (tool_activity()) {
                keep_scope = true;
                const bool parsed = messages_input && chat_turn.valid();
                const char * newest_kind = parsed
                    ? http_detail::pflash_keep_history_scope(chat_turn.turns, keep_newest)
                    : "unparsed";
                keep_history_stats = {{"in_scope", true},
                                      {"newest", newest_kind ? newest_kind : "user"},
                                      {"paste", true}};
            } else if (!tool_error.empty()) {
                std::fprintf(stderr,
                    "[pflash-keep-history] messages not readable (%s); rule skipped\n",
                    tool_error.c_str());
            }
            // Droppable target tokens of a turn's content (the pinned query
            // and instruction spans inside it excluded).
            const auto content_droppable = [&] (const http_detail::PflashChatTurn & turn) {
                const int length = turn.content_end - turn.content_begin;
                return target_estimate((std::max)(0, length -
                    http_detail::pflash_span_tokens_within(
                        required_instruction_spans, turn.content_begin,
                        turn.content_end)));
            };
            if (keep_scope && !(messages_input && chat_turn.valid())) {
                // A tool session whose turns cannot be located: nothing can
                // be told apart from the history, so nothing is compressed.
                admission_stats = {{"view", false}, {"decision", "keep_history"},
                                   {"raw_fits", raw_fits},
                                   {"keep_history", keep_history_stats}};
                serve_keep_raw(0, "turns not parsed");
                return {};
            }
            if (has_view && keep_scope) {
                // A tool session with a view (the raw history and any earlier
                // pastes as served): the view continues. A newest user
                // message over the threshold is compressed on its own after
                // the cached view, whatever PFLASH_ARRIVE; anything else --
                // a tool output of any size included -- is appended verbatim.
                admission_stats = {{"view", true}};
                int new_droppable = 0;
                bool region = false;
                if (keep_newest >= 0) {
                    const auto & newest = chat_turn.turns[(size_t) keep_newest];
                    new_droppable = content_droppable(newest);
                    region = newest.role_begin >= existing.drafter_gen_begin &&
                        new_droppable >= config_.pflash_threshold;
                    if (region) {
                        region_plan = {true, "new_content",
                                       {{newest.content_begin, newest.content_end}},
                                       existing.drafter_gen_begin,
                                       existing.view_gen_begin, input_tokens, true, 0};
                    }
                    keep_history_stats["new_begin"] = newest.content_begin;
                    keep_history_stats["new_end"] = newest.content_end;
                }
                keep_history_view = true;
                keep_history_stats["decision"] = region ? "new_content" : "view";
                keep_history_stats["new_droppable"] = new_droppable;
                keep_history_stats["base"] = "view";
                admission_stats["keep_history"] = keep_history_stats;
                count_keep_history(region ? "new_content" : "view");
                std::fprintf(stderr,
                    "[pflash-keep-history] %s on the view: newest %s droppable~%d "
                    "threshold=%d\n",
                    region ? "new_content" : "view",
                    keep_newest >= 0 ? "user content" : "turn not a user message (appended)",
                    new_droppable, config_.pflash_threshold);
            } else if (has_view) {
                admission_stats = {{"view", true}};
                if (!keep_history_stats.is_null()) {
                    admission_stats["keep_history"] = keep_history_stats;
                }
                // Arrival: a large new chunk on a compressed conversation is
                // compressed on its own; the cached view is served as it is.
                const int new_droppable = target_estimate((std::max)(0,
                    (input_tokens - existing.drafter_gen_begin) -
                    http_detail::pflash_span_tokens_within(
                        required_instruction_spans, existing.drafter_gen_begin,
                        input_tokens)));
                if (http_detail::pflash_arrive_enabled() &&
                    new_droppable >= config_.pflash_threshold) {
                    region_plan = {true, "arrive",
                                   {{existing.drafter_gen_begin, input_tokens}},
                                   existing.drafter_gen_begin,
                                   existing.view_gen_begin, input_tokens, true, 0};
                }
            } else {
                const auto restorable = cache_measured
                    ? *req.pflash_cached_prefix
                    : restorable_prefix_tokens(
                          req.prompt_tokens, req.disk_cache_policy);
                const int cached_target = restorable.tokens;
                const int cached_drafter = prompt_tokens > 0
                    ? (std::min)(input_tokens, (int) std::floor(
                          (double) input_tokens * (double) cached_target /
                          (double) prompt_tokens))
                    : 0;
                const int cached_pinned = http_detail::pflash_span_tokens_within(
                    required_instruction_spans, 0, cached_drafter);
                const int cached_droppable = cached_drafter > 0
                    ? target_estimate((std::max)(0, cached_drafter - cached_pinned))
                    : 0;
                const int uncached_droppable =
                    (std::max)(0, droppable_target - cached_droppable);
                auto admission = http_detail::pflash_cache_admission(
                    uncached_droppable, cached_droppable,
                    config_.pflash_threshold, raw_fits);
                // Agent history, no view: the history is served raw -- from
                // the prefix cache, or prefilled -- and only a newest user
                // message over the threshold is compressed (new_content, a
                // region plan on a raw base); otherwise the whole prompt is
                // served raw (keep_history; a newest tool output of any size
                // included). Under-threshold stays as it is (raw).
                int keep_new_droppable = 0;
                int raw_base = -1;
                if (keep_scope) {
                    const http_detail::PflashChatTurn * newest = keep_newest >= 0
                        ? &chat_turn.turns[(size_t) keep_newest] : nullptr;
                    keep_new_droppable = newest ? content_droppable(*newest) : 0;
                    const auto before = admission;
                    admission = http_detail::pflash_keep_history_admission(
                        admission, keep_new_droppable, config_.pflash_threshold,
                        http_detail::pflash_arrive_enabled());
                    if (admission == http_detail::PflashAdmission::NewContent) {
                        // The served history must be the raw prompt's own
                        // tokens (the prefix cache restores them, and full
                        // prefill would serve them): fail closed to raw when
                        // the two tokenizers' parses do not line up.
                        ChatMarkers markers;
                        raw_base = newest && resolve_chat_markers(tokenizer_, markers)
                            ? http_detail::pflash_raw_base_target(
                                  tokenizer_, markers, req.prompt_tokens,
                                  *drafter_tokenizer_, drafter_ids,
                                  chat_turn.turns, keep_newest)
                            : -1;
                        if (raw_base < 0) {
                            admission = http_detail::PflashAdmission::KeepHistory;
                            keep_history_stats["raw_base_failed"] = true;
                        }
                    }
                    keep_history_stats["decision_before"] =
                        http_detail::pflash_admission_name(before);
                    keep_history_stats["fired"] = admission != before;
                    if (newest) {
                        keep_history_stats["history_end"] = newest->role_begin;
                        keep_history_stats["new_begin"] = newest->content_begin;
                        keep_history_stats["new_end"] = newest->content_end;
                    }
                    keep_history_stats["new_droppable"] = keep_new_droppable;
                    keep_history_stats["base"] = raw_base >= 0 ? "raw" : "none";
                    if (raw_base >= 0) keep_history_stats["raw_base_tokens"] = raw_base;
                }
                admission_stats = {
                    {"view", false},
                    {"decision", http_detail::pflash_admission_name(admission)},
                    {"cached_tokens", cached_target},
                    {"cached_droppable", cached_droppable},
                    {"uncached_droppable", uncached_droppable},
                    {"droppable", droppable_target},
                    {"threshold", config_.pflash_threshold},
                    {"raw_fits", raw_fits},
                };
                if (!keep_history_stats.is_null()) {
                    admission_stats["keep_history"] = keep_history_stats;
                }
                // Disk cache off: no source field, the record is unchanged.
                const bool disk_cache_on = !disk_cache_.disabled();
                if (disk_cache_on) admission_stats["cached_source"] = restorable.source;
                std::fprintf(stderr,
                    "[pflash-admission] %s cached=%d cached_droppable~%d "
                    "uncached_droppable~%d droppable~%d threshold=%d%s%s\n",
                    http_detail::pflash_admission_name(admission),
                    cached_target, cached_droppable, uncached_droppable,
                    droppable_target, config_.pflash_threshold,
                    disk_cache_on ? " source=" : "",
                    disk_cache_on ? restorable.source : "");
                if (admission != http_detail::PflashAdmission::Compress) {
                    // Arrival on a raw conversation the cache holds: compress
                    // the large uncached chunk only; otherwise it stays raw.
                    if (admission == http_detail::PflashAdmission::NewContent) {
                        const auto & newest = chat_turn.turns[(size_t) keep_newest];
                        region_plan = {true, "new_content",
                                       {{newest.content_begin, newest.content_end}},
                                       newest.role_begin, raw_base, input_tokens,
                                       false, 0, /*raw_base=*/true};
                        count_keep_history("new_content");
                        std::fprintf(stderr,
                            "[pflash-keep-history] new_content: history [0,%d) served raw "
                            "(%d target tokens), new user content [%d,%d) droppable~%d "
                            ">= threshold %d\n",
                            newest.role_begin, raw_base, newest.content_begin,
                            newest.content_end, keep_new_droppable,
                            config_.pflash_threshold);
                    } else if (http_detail::pflash_arrive_enabled() &&
                        admission == http_detail::PflashAdmission::WarmCache) {
                        region_plan = {true, "arrive", {{cached_drafter, input_tokens}},
                                       0, 0, input_tokens, false, 0};
                    } else {
                        prepared.pflash_stats = {{"admission", admission_stats}};
                        if (admission == http_detail::PflashAdmission::KeepHistory) {
                            // Nothing to compress in a tool session but its
                            // history: the raw prompt, full prefill.
                            serve_keep_raw(keep_new_droppable,
                                keep_history_stats.value("raw_base_failed", false)
                                    ? "raw base mismatch"
                                    : (keep_newest >= 0 ? "newest user content under the threshold"
                                                        : "newest turn not a user message"));
                        }
                        return {};
                    }
                }
            }
            if (region_plan.active) {
                // Everything outside the regions is kept as it is served:
                // the selector spends the budget inside the regions only.
                auto mandatory = required_instruction_spans;
                const auto outside = http_detail::pflash_complement_token_spans(
                    region_plan.regions, input_tokens);
                mandatory.insert(mandatory.end(), outside.begin(), outside.end());
                mandatory = http_detail::canonicalize_pflash_token_spans(
                    std::move(mandatory));
                std::string region_error;
                if (!luce::pflash::validate_pflash_instruction_spans(
                        mandatory, input_tokens, region_error)) {
                    return "PFlash region plan mapping failed: " + region_error;
                }
                required_instruction_spans = std::move(mandatory);
                kept_tokens = kept_with(required_instruction_spans);
                region_plan.region_droppable =
                    target_estimate((std::max)(0, input_tokens - kept_tokens));
                admission_stats["region"] = {
                    {"mode", region_plan.mode},
                    {"regions", [&] {
                        json out = json::array();
                        for (const auto & span : region_plan.regions) {
                            out.push_back({span.begin, span.end});
                        }
                        return out;
                    }()},
                    {"cut_raw", region_plan.cut_raw},
                    {"cut_view", region_plan.cut_view},
                    {"tail_after", region_plan.tail_after},
                    {"region_droppable", region_plan.region_droppable},
                };
                std::fprintf(stderr,
                    "[pflash-region] %s regions=%zu cut=%d/%d tail_after=%d "
                    "region_droppable~%d view=%d\n",
                    region_plan.mode.c_str(), region_plan.regions.size(),
                    region_plan.cut_raw, region_plan.cut_view,
                    region_plan.tail_after, region_plan.region_droppable,
                    (int) region_plan.has_view);
            }
        }
        if (!region_plan.active &&
            config_.pflash_mode == ServerConfig::PflashMode::AUTO &&
            droppable_target < config_.pflash_threshold) {
            std::fprintf(stderr,
                "[pflash] skip-compress (droppable ~%d < threshold %d; "
                "kept %d of %d drafter tokens)\n",
                droppable_target, config_.pflash_threshold, kept_tokens,
                input_tokens);
            return {};
        }
    }

    ModelBackend::CompressRequest compress_request;
    compress_request.input_ids = std::move(drafter_ids);
    compress_request.required_instruction_spans =
        std::move(required_instruction_spans);
    compress_request.query_suffix_candidates = query_suffix_candidates;
    compress_request.history_query_spans = history_query_spans;
    compress_request.keep_query_spans = keep_query_spans;
    compress_request.turn_query_span = turn_query_span;
    compress_request.extra_query_spans = extra_query_spans;
    compress_request.request_messages = request_probe_messages;
    compress_request.keep_ratio = http_detail::resolve_pflash_keep_ratio(
        pflash_keep_ratio(config_, region_plan.active
            ? (std::max)(1, region_plan.region_droppable) : prompt_tokens),
        req.session_id, sessions_);
    if (experiment.selection_active && query_window.valid()) {
        const double effective = http_detail::pflash_effective_keep_ratio(
            (int) compress_request.input_ids.size(), kept_tokens,
            compress_request.keep_ratio);
        std::fprintf(stderr,
            "[pflash-select] kept=%d droppable=%d keep_ratio=%.6f "
            "effective=%.6f\n",
            kept_tokens,
            (int) compress_request.input_ids.size() - kept_tokens,
            (double) compress_request.keep_ratio, effective);
        compress_request.keep_ratio = (float) effective;
    }
    if (query_window.valid()) {
        compress_request.score_query_end = query_window.end;
        compress_request.score_query_tokens = query_window.tokens;
        if (experiment.configured) {
            const int query_begin = query_window.end - query_window.tokens;
            json instruction_spans = json::array();
            for (const auto & span :
                    compress_request.required_instruction_spans) {
                instruction_spans.push_back({span.begin, span.end});
            }
            const json provenance = {
                {"schema_version", 1},
                {"input_kind", parser_input_kind},
                {"selection_rule", parser_selection_rule},
                {"query_parser",
                 luce::pflash::pflash_query_parser_name(
                     experiment.query_parser)},
                {"input_tokens", (int) compress_request.input_ids.size()},
                {"input_fingerprint_fnv1a64",
                 http_detail::pflash_token_fingerprint(
                     compress_request.input_ids)},
                {"content_begin", query_content_begin},
                {"content_end", query_content_end},
                {"query_begin", query_begin},
                {"query_end", query_window.end},
                {"query_span_begin", query_span.begin},
                {"query_span_end", query_span.end},
                {"query_suffix_candidates", query_suffix_candidates},
                {"history_queries", history_query_spans.size()},
                {"turn_query_begin", turn_query_span.begin},
                {"turn_query_end", turn_query_span.end},
                {"extra_query_spans", [&] {
                    json spans = json::array();
                    for (const auto & span : extra_query_spans) {
                        spans.push_back({span.begin, span.end});
                    }
                    return spans;
                }()},
                {"requested_query_tokens", experiment.query_tokens},
                {"required_text_count", req.pflash_required.size()},
                {"expected_query_ids", expected_query_ids},
                {"required_instruction_spans", instruction_spans},
            };
            std::fprintf(stderr, "[pflash-parser] %s\n",
                         provenance.dump().c_str());
            std::fflush(stderr);
        }
        std::fprintf(stderr,
            "[pflash] scorer query mapped to drafter tokens [%d,%d); "
            "rendered suffix=%zu tokens\n",
            query_window.end - query_window.tokens, query_window.end,
            compress_request.input_ids.size() - (size_t) query_window.end);
    } else {
        std::fprintf(stderr,
            "[pflash] ERROR: scorer query mapping failed; refusing compression\n");
        return "PFlash scorer query mapping failed";
    }
    compress_request.drafter_path = config_.pflash_drafter_path;
    compress_request.drafter_gpu = config_.pflash_drafter_gpu;
    compress_request.skip_park = config_.pflash_skip_park;
    const auto residency = resolve_draft_residency_action(
        config_.draft_residency,
        DraftResidencyContext{
            DraftResidencyUse::PFlashCompress,
            config_.lazy_draft,
            !config_.draft_path.empty(),
        });
    compress_request.residency_action = residency;

    // The selector budget is counted in drafter tokens, but the ceiling is
    // enforced on the re-encoded target-token prompt. Vocabulary mismatches
    // between the two tokenizers can inflate the re-encode past the ceiling,
    // so retry with a tightened keep ratio; fail closed if it persists.
    int target_ceiling = experiment.selection_active
        ? http_detail::pflash_target_token_ceiling(
            prompt_tokens, compress_request.keep_ratio)
        : -1;
    // Request stage 2 widens the keep budget x1.5 (STAGE2-CONTRACT.md section 5 step 5): the ceiling follows, once.
    bool stage2_ceiling_widened = false;
    if (experiment.selection_active && target_ceiling < 0) {
        return "PFlash strict selection target-token ceiling is invalid";
    }
    const float requested_keep_ratio = compress_request.keep_ratio;

    // A turn the conversation's view serves without scoring -- the same
    // prompt again, or a small follow-up appended verbatim with recall off
    // -- skips the drafter altogether.
    if (experiment.selection_active && messages_input && chat_turn.valid() &&
        !config_.pflash_remote_drafter && !region_plan.active) {
        std::vector<int32_t> served;
        json view_stats;
        if (serve_pflash_chat_view(
                req, compress_request.input_ids, chat_turn, nullptr, nullptr,
                served, prepared.snapshot_cut, view_stats, keep_history_view)) {
            prepared.tokens = std::move(served);
            prepared.compressed = true;
            prepared.pflash_stats = {
                {"compress_ms", 0.0},
                {"drafter_input_tokens", compress_request.input_ids.size()},
                {"query_rule", parser_selection_rule},
                {"view", view_stats},
                {"drafter_fingerprint_fnv1a64",
                 http_detail::pflash_token_fingerprint(compress_request.input_ids)},
            };
            if (!admission_stats.is_null()) prepared.pflash_stats["admission"] = admission_stats;
            trace_pflash_served(req, prepared);
            return {};
        }
    }

    ModelBackend::CompressResult result;
    std::vector<int32_t> final_tokens;
    const auto compress_started = std::chrono::steady_clock::now();
    int join_overhead = 0;
    int ceiling_tightening = 0;   // tokens taken off the selector target so far (cumulative over retries)
    for (int attempt = 0; ; ++attempt) {
        result = {};
        if (config_.pflash_remote_drafter) {
            if (!pflash_remote_.active() &&
                !pflash_remote_.start(config_.pflash_remote.ipc_bin,
                                      config_.pflash_drafter_path,
                                      config_.pflash_drafter_gpu,
                                      config_.pflash_remote.work_dir)) {
                return "remote PFlash drafter start failed";
            }
            result.ok = pflash_remote_.compress(
                compress_request.input_ids, compress_request.keep_ratio,
                result.compressed_ids,
                compress_request.score_query_end,
                compress_request.score_query_tokens,
                compress_request.required_instruction_spans);
            if (residency == DraftResidencyAction::ReleaseAfterUse) {
                pflash_remote_.close();
            }
        } else {
            result = backend_.compress(compress_request);
        }

        if (!result.ok || result.compressed_ids.empty()) {
            if (!config_.pflash_remote_drafter) {
                const char * detail = luce_last_error();
                prepared.pflash_failure_detail = detail ? detail : "";
            }
            return config_.pflash_remote_drafter
                ? "remote PFlash drafter compression failed"
                : "PFlash compression failed";
        }
        // Stage 2 widened the budget past the prompt: it is served raw (below).
        if (result.raw_fits) break;
        if (result.budget_widened && !stage2_ceiling_widened && target_ceiling >= 0) {
            target_ceiling = (std::min)(prompt_tokens, target_ceiling + target_ceiling / 2);
            stage2_ceiling_widened = true;
        }

        std::string compressed_text =
            drafter_tokenizer_->decode(result.compressed_ids);
        join_overhead = 0;
        // Kept pieces that were not adjacent
        // in the prompt are joined by a paragraph break when neither side
        // already has one, so a cut does not glue two passages into one
        // run-on line ("...other bands.Document 1:").
        if (http_detail::pflash_paragraph_join() && !result.kept_spans.empty()) {
            const int plain = (int) tokenizer_.encode(compressed_text).size();
            compressed_text = http_detail::pflash_join_kept_spans(
                *drafter_tokenizer_, compress_request.input_ids,
                result.kept_spans);
            // The breaks are layout, not retained context: the ceiling
            // bounds what the selection kept.
            join_overhead = (std::max)(
                0, (int) tokenizer_.encode(compressed_text).size() - plain);
        }

        // Compression is allowed to be lossy, but the active user query must
        // survive. Re-append short queries when fewer than 80% of their tokens do.
        if (!experiment.selection_active && !last_user_text.empty()) {
            int query_kept = 0;
            if (!semantic_query_ids.empty()) {
                int query_index = (int) semantic_query_ids.size() - 1;
                for (int kept_index = (int) result.compressed_ids.size() - 1;
                     kept_index >= 0 && query_index >= 0; --kept_index) {
                    if (result.compressed_ids[kept_index] == semantic_query_ids[query_index]) {
                        ++query_kept;
                        --query_index;
                    }
                }
            }
            const float survival = (float) query_kept /
                (std::max)(1, (int) semantic_query_ids.size());
            std::fprintf(stderr,
                "[pflash] query survival: %d/%d (%.0f%%)\n",
                query_kept, (int) semantic_query_ids.size(), survival * 100.0f);
            if (survival < 0.80f && (int) semantic_query_ids.size() < 1000) {
                compressed_text += "\n" + last_user_text;
                std::fprintf(stderr,
                    "[pflash] query below 80%% — re-appended full query (%d tokens)\n",
                    (int) semantic_query_ids.size());
            } else if (survival < 0.80f) {
                std::fprintf(stderr,
                    "[pflash] query below 80%% but too large to re-append (%d tokens)\n",
                    (int) semantic_query_ids.size());
            }
        }

        final_tokens = tokenizer_.encode(compressed_text);
        if (!experiment.selection_active ||
            (int) final_tokens.size() - join_overhead <= target_ceiling) {
            break;
        }
        const int overflow =
            (int) final_tokens.size() - join_overhead - target_ceiling;
        // Each retry tightens the selector target by its own overflow ON TOP OF the earlier retries (defect
        // fix, 9 Oct: the retry used to restart from the requested ratio each time, so a smaller second
        // overflow LOOSENED the ratio again and the third attempt repeated the first: 11584 > 11567 with probe
        // pins). The first retry is unchanged.
        const float retry_ratio = http_detail::pflash_ceiling_retry_keep_ratio(
            requested_keep_ratio, target_ceiling, overflow, ceiling_tightening);
        if (attempt >= 2 || retry_ratio <= 0.0f) {
            break;
        }
        compress_request.keep_ratio = retry_ratio;
        std::fprintf(stderr,
            "[pflash-select] final prompt %zu exceeds ceiling %d "
            "(re-encode overshoot); retrying with keep_ratio %.6f\n",
            final_tokens.size(), target_ceiling,
            (double) compress_request.keep_ratio);
        std::fflush(stderr);
    }
    if (result.raw_fits) {
        std::fprintf(stderr,
            "[pflash-request-stage2] widened budget %d covers the %zu-token prompt: serving it raw\n",
            result.stage2_budget_widened, compress_request.input_ids.size());
        std::fflush(stderr);
        prepared.pflash_stats = {{"stage2", {{"widened", true}, {"raw_fits", true},
                                             {"budget", result.stage2_budget},
                                             {"budget_widened", result.stage2_budget_widened}}}};
        if (!admission_stats.is_null()) prepared.pflash_stats["admission"] = admission_stats;
        prepared.compressed = false;
        prepared.tokens = req.prompt_tokens;
        prepared.snapshot_cut = -1;
        return {};
    }
    if (experiment.selection_active) {
        std::fprintf(stderr,
            "[pflash-select] final target tokens=%zu ceiling=%d\n",
            final_tokens.size(), target_ceiling);
        std::fflush(stderr);
        if ((int) final_tokens.size() - join_overhead > target_ceiling) {
            return "PFlash strict selection final prompt exceeds target-token ceiling "
                "(" + std::to_string(final_tokens.size()) + " > " +
                std::to_string(target_ceiling) + ")";
        }
    }
    // Relevance-spread fallback (PFLASH_FALLBACK_SPREAD, opt-in): diffuse
    // relevance over the scored segments -> full prefill of the raw prompt
    // through the failure-policy (D5) path, reason "spread".
    json spread_record;
    if (experiment.fallback_spread > 0.0) {
        if (!result.spread_valid && !region_plan.active) {
            return "PFlash relevance spread unavailable";
        }
        const auto decision = http_detail::pflash_spread_decision(
            experiment.fallback_spread, result.spread_valid, result.spread,
            region_plan.active);
        const bool fires = decision.fires;
        spread_record = {{"spread", result.spread},
                         {"spread_threshold", experiment.fallback_spread},
                         {"spread_k", result.spread_k},
                         {"fired", fires}};
        if (decision.skipped) {
            spread_record["skipped"] = decision.skipped;
            spread_record["spread_valid"] = result.spread_valid;
            std::fprintf(stderr,
                "[pflash-spread] spread=%.6f (k=%d): no decision on a region plan (%s)\n",
                result.spread, result.spread_k, region_plan.mode.c_str());
            std::fflush(stderr);
        } else if (fires) {
            if (!effective_prompt_overflows(prompt_tokens, -1, req.max_output,
                                            config_.max_ctx)) {
                size_t forgotten = 0;
                if (messages_input) {
                    forgotten = pflash_views_.forget(req.prompt_tokens);
                }
                spread_record["views_forgotten"] = forgotten;
                std::fprintf(stderr,
                    "[pflash-spread] spread=%.6f >= %.6f (k=%d): full prefill of the raw "
                    "prompt (%d tokens), views forgotten=%zu\n",
                    result.spread, experiment.fallback_spread, result.spread_k,
                    prompt_tokens, forgotten);
                std::fflush(stderr);
                prepared.pflash_spread_fallback = spread_record;
                char message[160];
                std::snprintf(message, sizeof(message),
                    "PFlash relevance spread fallback (spread %.6f >= %.6f)",
                    result.spread, experiment.fallback_spread);
                return message;
            }
            spread_record["refused"] = "raw_exceeds_max_ctx";
            std::fprintf(stderr,
                "[pflash-spread] spread=%.6f >= %.6f but the raw prompt (%d tokens) does "
                "not fit max_ctx %d: serving the compressed prompt\n",
                result.spread, experiment.fallback_spread, prompt_tokens, config_.max_ctx);
            std::fflush(stderr);
        } else {
            std::fprintf(stderr, "[pflash-spread] spread=%.6f < %.6f (k=%d)\n",
                         result.spread, experiment.fallback_spread, result.spread_k);
            std::fflush(stderr);
        }
    }
    prepared.pflash_stats = {
        {"compress_ms", std::round(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - compress_started).count() * 10.0) / 10.0},
        {"drafter_input_tokens", compress_request.input_ids.size()},
        {"kept_tokens", kept_tokens},
        {"keep_ratio", compress_request.keep_ratio},
        {"compressed_tokens", final_tokens.size()},
        {"query_rule", parser_selection_rule},
        {"history_queries", history_query_spans.size()},
        {"scorer_resume", result.scorer_resume},
        {"scorer_new_tokens", result.scorer_new_tokens},
        {"scorer_forward_ms", std::round(result.scorer_forward_s * 10000.0) / 10.0},
    };
    if (!spread_record.is_null()) prepared.pflash_stats["spread"] = spread_record;
    if (result.budget_widened) {
        prepared.pflash_stats["stage2"] = {{"widened", true}, {"raw_fits", false},
                                           {"budget", result.stage2_budget},
                                           {"budget_widened", result.stage2_budget_widened},
                                           {"target_ceiling", target_ceiling}};
    }
    if (region_plan.active) {
        std::vector<int32_t> served;
        if (!result.kept_spans.empty() && serve_pflash_region_view(
                req, compress_request.input_ids, chat_turn, region_plan,
                result.kept_spans, served, prepared.snapshot_cut,
                prepared.pflash_stats["view"])) {
            final_tokens = std::move(served);
        } else {
            // Never serve a partly built region view: the raw prompt is what
            // the conversation would have served without the plan.
            std::fprintf(stderr,
                "[pflash-region] %s could not be served; serving the raw prompt\n",
                region_plan.mode.c_str());
            prepared.pflash_stats = {{"admission", admission_stats},
                                     {"region_failed", region_plan.mode}};
            prepared.compressed = false;
            prepared.tokens = req.prompt_tokens;
            prepared.snapshot_cut = -1;
            return {};
        }
    } else {
        bool viewed = false;
        if (experiment.selection_active && messages_input && chat_turn.valid() &&
            !result.kept_spans.empty()) {
            std::vector<int32_t> served;
            viewed = serve_pflash_chat_view(
                req, compress_request.input_ids, chat_turn, &final_tokens,
                &result.kept_spans, served,
                prepared.snapshot_cut, prepared.pflash_stats["view"],
                keep_history_view);
            if (viewed) final_tokens = std::move(served);
        }
        if (!viewed && keep_history_view) {
            // Agent history whose view cannot continue: the raw prompt
            // (never a fresh whole-prompt selection), counted with the raw
            // fallbacks, reason keep_history.
            std::fprintf(stderr,
                "[pflash-keep-history] the view cannot continue; serving the raw "
                "prompt (%d tokens)\n", prompt_tokens);
            std::fflush(stderr);
            {
                std::lock_guard<std::mutex> lk(pflash_failure_mu_);
                ++pflash_fallback_served_;
                ++pflash_fallback_reasons_["keep_history"];
                ++pflash_keep_history_counts_["view_raw"];
            }
            prepared.pflash_stats = {{"admission", admission_stats},
                                     {"pflash_fallback", {{"reason", "keep_history"},
                                                          {"prompt_tokens", prompt_tokens},
                                                          {"view", "not_continued"}}}};
            prepared.compressed = false;
            prepared.tokens = req.prompt_tokens;
            prepared.snapshot_cut = -1;
            return {};
        }
    }
    if (prepared.pflash_stats["view"].is_null() && !result.kept_spans.empty()) {
        std::vector<http_detail::PflashViewPiece> pieces;
        for (const auto & span : result.kept_spans) pieces.push_back({span.begin, span.end, false});
        prepared.pflash_stats["pieces"] = http_detail::pflash_view_pieces_json(pieces);
    }
    prepared.pflash_stats["drafter_fingerprint_fnv1a64"] =
        http_detail::pflash_token_fingerprint(compress_request.input_ids);
    if (!admission_stats.is_null()) prepared.pflash_stats["admission"] = admission_stats;
    prepared.tokens = std::move(final_tokens);
    prepared.compressed = true;
    trace_pflash_served(req, prepared);
    std::fprintf(stderr,
        "[pflash] %d -> %d -> %d tokens (%.1f%% kept)\n",
        prompt_tokens, (int) result.compressed_ids.size(),
        (int) prepared.tokens.size(),
        100.0 * prepared.tokens.size() / prompt_tokens);
    return {};
}

http_detail::PflashRestorablePrefix HttpServer::restorable_prefix_tokens(
        const std::vector<int32_t> & prompt,
        const DiskPrefixCachePolicy & disk_policy) {
    const int ram = ram_restorable_prefix_tokens(prompt);
    // The generation path looks the whole prompt up on disk only in Full
    // scope; its probe is index-only, so is this one.
    const bool disk_enabled = !disk_cache_.disabled() &&
        disk_policy.mode == DiskPrefixCacheMode::Full;
    int disk = 0;
    if (disk_enabled) {
        const auto probe = disk_cache_.longest_prefix(prompt);
        disk = (std::min)(probe.len, probe.cur_pos);
    }
    return http_detail::pflash_restorable_prefix(ram, disk_enabled, disk);
}

int HttpServer::ram_restorable_prefix_tokens(
        const std::vector<int32_t> & prompt) {
    if (prefix_cache_.disabled()) return 0;
    int limit = (int) prompt.size();
    while (limit > 0) {
        const auto [slot, length] =
            prefix_cache_.lookup_candidate(prompt, limit);
        if (slot < 0 || length <= 0) return 0;
        if (backend_.snapshot_used(slot)) {
            const int position = backend_.snapshot_cur_pos(slot);
            if (position > 0) return (std::min)(length, position);
        }
        // A key whose snapshot is gone: try the next shorter one; the
        // generation path invalidates it when it gets there.
        limit = length - 1;
    }
    return 0;
}

json HttpServer::pflash_failure_counters_json() const {
    std::lock_guard<std::mutex> lk(pflash_failure_mu_);
    return {
        {"fail_closed", http_detail::pflash_fail_closed_from_env()},
        {"fallback_served", pflash_fallback_served_},
        {"fallback_refused", pflash_fallback_refused_},
        {"fallback_reasons", pflash_fallback_reasons_},
        {"refused_reasons", pflash_refused_reasons_},
    };
}

json HttpServer::pflash_unified_json() const {
    if (!pflash_unified_) return nullptr;
    std::lock_guard<std::mutex> lk(pflash_failure_mu_);
    return {{"region_tools", pflash_region_tools_.identity},
            {"recall", pflash_unified_recall_},
            {"recall_gate", pflash_recall_gate_ > 0.0 ? json(pflash_recall_gate_) : json(nullptr)},
            {"views", pflash_unified_views_.size()},
            {"view_bytes", pflash_unified_views_.bytes()},
            {"disk_views", pflash_unified_disk_views_.size()},
            {"counts", pflash_unified_counts_}};
}

json HttpServer::pflash_keep_history_json() const {
    if (config_.pflash_mode != ServerConfig::PflashMode::AUTO ||
        !http_detail::pflash_cache_admission_enabled()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(pflash_failure_mu_);
    return {{"paste", http_detail::pflash_agent_paste_enabled()},
            {"decisions", pflash_keep_history_counts_}};
}

void HttpServer::trace_pflash_served(
        const ParsedRequest & req, const PreparedPrompt & prepared) {
    const char * path = std::getenv("PFLASH_VIEW_TRACE_PATH");
    if (!path || !*path) return;
    json record = {
        {"schema_version", pflash_unified_ ? 3 : 2},
        {"kind", prepared.compressed ? "compressed" : "raw"},
        {"request_id", req.response_id},
        {"prompt_tokens", req.prompt_tokens.size()},
        {"served_tokens", prepared.tokens.size()},
        {"pflash", prepared.pflash_stats},
        {"served_text", tokenizer_.decode(prepared.tokens)},
    };
    if (!prepared.unified_handback.empty()) {
        if (!record["pflash"].is_object()) record["pflash"] = json::object();
        record["pflash"]["unified"] = {{"decision", "handback"}, {"reason", prepared.unified_handback}};
    }
    // Failure policy D5: a raw record served because compression failed says so.
    if (prepared.pflash_stats.is_object() && prepared.pflash_stats.contains("pflash_fallback")) {
        record["pflash_fallback"] = prepared.pflash_stats["pflash_fallback"];
    }
    std::ofstream out(path, std::ios::app);
    if (out) out << record.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
}

bool HttpServer::serve_pflash_chat_view(
        const ParsedRequest & req,
        const std::vector<int32_t> & drafter_ids,
        const http_detail::PflashChatTurnSpan & turn,
        const std::vector<int32_t> * fresh,
        const std::vector<PFlashTokenSpan> * kept_spans,
        std::vector<int32_t> & served,
        int & snapshot_cut,
        json & stats,
        bool keep_history) {
    snapshot_cut = -1;
    stats = nullptr;
    const bool compressed = fresh != nullptr && kept_spans != nullptr;
    const char * disabled = std::getenv("PFLASH_CHAT_VIEW");
    if (disabled && std::string(disabled) == "0") return false;
    const int input = (int) drafter_ids.size();
    if (turn.generation_begin <= 0 || turn.generation_begin >= input) {
        return false;
    }
    // The generation prompt, in target tokens: the raw prompt and every
    // served prompt end with it (strict selection keeps it verbatim).
    const auto generation = tokenizer_.encode(drafter_tokenizer_->decode(
        std::vector<int32_t>(drafter_ids.begin() + turn.generation_begin,
                             drafter_ids.end())));
    const auto ends_with_generation = [&generation] (
            const std::vector<int32_t> & tokens) {
        return !generation.empty() && tokens.size() > generation.size() &&
            std::equal(generation.begin(), generation.end(),
                       tokens.end() - (long) generation.size());
    };
    if (!ends_with_generation(req.prompt_tokens) ||
        (compressed && !ends_with_generation(*fresh))) {
        return false;
    }
    const int raw_gen_begin =
        (int) (req.prompt_tokens.size() - generation.size());

    http_detail::PflashChatView next;
    next.raw_tokens = req.prompt_tokens;
    next.raw_gen_begin = raw_gen_begin;
    next.drafter_ids = drafter_ids;
    next.drafter_gen_begin = turn.generation_begin;

    http_detail::PflashChatView view;
    const bool continues =
        pflash_views_.find(req.prompt_tokens, drafter_ids, view);
    const auto serve_fresh = [&] (const char * why, int turns) {
        next.view_tokens = *fresh;
        next.view_gen_begin = (int) (fresh->size() - generation.size());
        next.spans = *kept_spans;
        next.pieces.clear();
        for (const auto & span : *kept_spans) next.pieces.push_back({span.begin, span.end, false});
        next.turns = turns;
        snapshot_cut = next.view_gen_begin;
        std::fprintf(stderr,
            "[pflash-view] %s turn=%d served=%zu\n", why, turns, fresh->size());
        std::fflush(stderr);
        stats = {{"mode", why}, {"turn", turns}, {"served_tokens", fresh->size()},
                 {"fresh_tokens", fresh->size()},
                 {"pieces", http_detail::pflash_view_pieces_json(next.pieces)}};
        served = *fresh;
        pflash_views_.remember(std::move(next));
        return true;
    };
    if (continues && view.raw_tokens == req.prompt_tokens) {
        // The same prompt again (a retry): serve what was served.
        std::fprintf(stderr, "[pflash-view] repeat turn=%d served=%zu\n",
                     view.turns, view.view_tokens.size());
        std::fflush(stderr);
        stats = {{"mode", "repeat"}, {"turn", view.turns},
                 {"served_tokens", view.view_tokens.size()},
                 {"pieces", http_detail::pflash_view_pieces_json(view.pieces)}};
        if (compressed) stats["fresh_tokens"] = fresh->size();
        snapshot_cut = view.view_gen_begin;
        served = view.view_tokens;
        return true;
    }
    const bool usable = continues &&
        view.drafter_gen_begin < turn.generation_begin &&
        view.view_gen_begin > 0 &&
        (size_t) view.view_gen_begin <= view.view_tokens.size();
    // Agent history: never replace the conversation with a fresh selection;
    // the caller serves the raw prompt.
    if (!usable) return compressed && !keep_history && serve_fresh("fresh", 1);

    // What this turn adds to the conversation, in target tokens. A small
    // follow-up is appended verbatim, the way full prefill appends it; one
    // of PFLASH_CHAT_COMPRESS_NEW_TOKENS or more (a pasted document, a large
    // tool output) goes through the compressor, and only the new material
    // is compressed, so the view it extends stays cached.
    const int new_tokens = raw_gen_begin + (int) generation.size() -
        view.raw_gen_begin;
    const bool compress_new = !keep_history &&
        new_tokens >= http_detail::pflash_chat_compress_new_tokens();
    const bool new_question = turn.role_begin >= view.drafter_gen_begin;
    const bool recall = new_question && http_detail::pflash_chat_recall();
    if (!compressed && (compress_new || recall)) return false;

    // Recall: what a fresh selection for the new question keeps that the
    // view does not hold. Only a new user turn brings a new question; an
    // agent step (assistant call plus tool output) appends without
    // recalling. What is missing is appended, whatever its size, so the
    // cached view is never thrown away for a new topic; the view starts over
    // from the fresh selection only when it outgrows it (below).
    std::vector<PFlashTokenSpan> recalled;
    if (compressed && recall) {
        auto in_view = view.spans;
        in_view.push_back({view.drafter_gen_begin, input});
        in_view = http_detail::canonicalize_pflash_token_spans(std::move(in_view));
        const auto missing =
            http_detail::pflash_subtract_token_spans(*kept_spans, in_view);
        // Whole kept pieces: a passage the view holds only part of comes
        // back in one piece, in order, not as a fragment far from the rest.
        // Agent history: the piece is clipped to the content of the message
        // that holds the missing part (a compressed paste), so the excerpt
        // block never repeats raw history or this turn's own messages.
        for (const auto & span : *kept_spans) {
            if (keep_history) {
                for (const auto & message : turn.turns) {
                    const int lo = (std::max)(span.begin, message.content_begin);
                    const int hi = (std::min)((std::min)(span.end, message.content_end),
                                              view.drafter_gen_begin);
                    if (hi <= lo) continue;
                    for (const auto & part : missing) {
                        if (part.begin < hi && part.end > lo) {
                            recalled.push_back({lo, hi});
                            break;
                        }
                    }
                }
                continue;
            }
            for (const auto & part : missing) {
                if (part.begin < span.end && part.end > span.begin) {
                    recalled.push_back(span);
                    break;
                }
            }
        }
        recalled = http_detail::canonicalize_pflash_token_spans(std::move(recalled));
    }
    std::string recall_block;
    int recalled_tokens = 0;
    std::vector<std::string> role_markers;
    std::vector<std::string> end_markers;
    bool generic_roles = false;
    if (!recalled.empty()) {
        ChatMarkers markers;
        if (resolve_chat_markers(tokenizer_, markers)) {
            const auto seq_text = [this] (const std::vector<int32_t> & seq) {
                std::string text;
                for (const int32_t id : seq) text += tokenizer_.token_text(id);
                return text;
            };
            for (const auto & seq : markers.next_role_starts) {
                role_markers.push_back(seq_text(seq));
            }
            for (const auto & seq : markers.end_msg_seqs) {
                end_markers.push_back(seq_text(seq));
            }
            generic_roles = !markers.role_starts_delimit &&
                markers.family != "laguna";
        }
        std::string excerpts;
        for (const auto & span : recalled) {
            const std::string excerpt = http_detail::pflash_recall_excerpt(
                drafter_tokenizer_->decode(std::vector<int32_t>(
                    drafter_ids.begin() + span.begin,
                    drafter_ids.begin() + span.end)),
                role_markers, end_markers, generic_roles);
            if (excerpt.empty()) continue;
            if (!excerpts.empty()) excerpts += "\n\n";
            excerpts += excerpt;
            recalled_tokens += span.end - span.begin;
        }
        if (!excerpts.empty()) {
            recall_block = "[Earlier in this conversation]\n" + excerpts +
                "\n[End of earlier excerpts]\n\n";
        }
    }

    // This turn's new tokens from where the previous generation prompt
    // started: all of them, or the parts the fresh selection keeps when they
    // are compressed. Recalled excerpts open the new user turn's content,
    // after everything the target cached.
    const auto decode_range = [&] (int begin, int end) {
        return end > begin
            ? drafter_tokenizer_->decode(std::vector<int32_t>(
                  drafter_ids.begin() + begin, drafter_ids.begin() + end))
            : std::string();
    };
    std::vector<PFlashTokenSpan> delta_spans;
    if (compress_new) {
        delta_spans = http_detail::pflash_subtract_token_spans(
            *kept_spans, {{0, view.drafter_gen_begin}});
    } else {
        delta_spans.push_back({view.drafter_gen_begin, input});
    }
    const int split = new_question && !recall_block.empty()
        ? turn.content_begin : input;
    std::vector<PFlashTokenSpan> before_split;
    std::vector<PFlashTokenSpan> after_split;
    for (const auto & span : delta_spans) {
        if (span.begin < split) {
            before_split.push_back({span.begin, (std::min)(span.end, split)});
        }
        if (span.end > split) {
            after_split.push_back({(std::max)(span.begin, split), span.end});
        }
    }
    const auto join = [&] (const std::vector<PFlashTokenSpan> & spans) {
        if (compress_new && http_detail::pflash_paragraph_join()) {
            return http_detail::pflash_join_kept_spans(
                *drafter_tokenizer_, drafter_ids, spans);
        }
        std::string text;
        for (const auto & span : spans) text += decode_range(span.begin, span.end);
        return text;
    };
    const std::string delta = join(before_split) + recall_block + join(after_split);
    served.assign(view.view_tokens.begin(),
                  view.view_tokens.begin() + view.view_gen_begin);
    const auto delta_tokens = tokenizer_.encode(delta);
    served.insert(served.end(), delta_tokens.begin(), delta_tokens.end());
    if (!ends_with_generation(served)) {
        return compressed && !keep_history && serve_fresh("fresh", 1);
    }
    // Rebuild when the view outgrew what a fresh selection keeps, or the
    // context: the fresh prompt starts a new view, prefilled from scratch.
    const bool too_long = config_.max_ctx > 0 &&
        (int) served.size() + req.max_output > config_.max_ctx;
    // PFLASH_CHAT_REBUILD_RATIO (default 2; 0 = only at the context limit).
    const double rebuild_ratio = http_detail::pflash_chat_rebuild_ratio();
    // PFLASH_CHAT_REBUILD_MIN_TOKENS (default 0): a view smaller than this is
    // never rebuilt, so short chats keep their cache; long ones stay bounded.
    // Agent history: the view keeps the whole conversation however large it
    // grows next to a fresh selection; only the context limit rebuilds it.
    const bool outgrown = !keep_history && compressed && rebuild_ratio > 0.0 &&
        (double) served.size() > rebuild_ratio * (double) fresh->size() &&
        (int) served.size() >= http_detail::pflash_chat_rebuild_min_tokens();
    // Agent history: never rebuilt, not even at the context limit (a rebuild
    // compresses the history); the caller serves the raw prompt instead.
    if (too_long || outgrown) {
        return compressed && !keep_history && serve_fresh("rebuild", view.turns + 1);
    }

    auto spans = view.spans;
    spans.insert(spans.end(), delta_spans.begin(), delta_spans.end());
    spans.insert(spans.end(), recalled.begin(), recalled.end());
    // Provenance, in served order: the cached view, this turn's tokens
    // before the recall block, the recalled excerpts, the rest.
    // The cached view is served up to its generation prompt only: cut its
    // pieces at the raw generation-prompt start.
    next.pieces.clear();
    for (auto piece : view.pieces) {
        if (piece.begin >= view.drafter_gen_begin) continue;
        piece.end = (std::min)(piece.end, view.drafter_gen_begin);
        next.pieces.push_back(piece);
    }
    for (const auto & span : before_split) next.pieces.push_back({span.begin, span.end, false});
    if (!recall_block.empty()) {
        for (const auto & span : recalled) next.pieces.push_back({span.begin, span.end, true, view.turns + 1});
    }
    for (const auto & span : after_split) next.pieces.push_back({span.begin, span.end, false});
    next.view_tokens = served;
    next.view_gen_begin = (int) (served.size() - generation.size());
    next.spans = http_detail::canonicalize_pflash_token_spans(std::move(spans));
    next.turns = view.turns + 1;
    snapshot_cut = next.view_gen_begin;
    const char * mode = compress_new ? "continue-compressed" : "continue";
    std::fprintf(stderr,
        "[pflash-view] %s turn=%d served=%zu reused=%d new=%d delta=%zu "
        "recalled=%d fresh=%d\n",
        mode, next.turns, served.size(), view.view_gen_begin, new_tokens,
        delta_tokens.size(), recalled_tokens,
        compressed ? (int) fresh->size() : -1);
    std::fflush(stderr);
    stats = {{"mode", mode}, {"turn", next.turns},
             {"served_tokens", served.size()}, {"reused_tokens", view.view_gen_begin},
             {"new_tokens", new_tokens}, {"delta_tokens", delta_tokens.size()},
             {"recalled_tokens", recalled_tokens},
             {"pieces", http_detail::pflash_view_pieces_json(next.pieces)}};
    if (compressed) stats["fresh_tokens"] = fresh->size();
    if (keep_history) stats["keep_history"] = true;
    if (const char * trace = std::getenv("PFLASH_VIEW_TRACE_PATH"); trace && *trace) {
        // Provenance of a continued view (trace only): the raw drafter ids
        // after the cached prefix, so a reader rebuilds this request's raw
        // prompt from the previous one without a selection trace.
        stats["prefix_drafter_tokens"] = view.drafter_gen_begin;
        stats["drafter_tokens"] = input;
        stats["delta_ids"] = std::vector<int32_t>(drafter_ids.begin() + view.drafter_gen_begin, drafter_ids.end());
        if (!recall_block.empty()) {
            // The recall transformation's inputs, so a reader reproduces the
            // block exactly (pflash_recall_excerpt over each recall piece).
            stats["recall_markers"] = {{"role", role_markers}, {"end", end_markers},
                                       {"generic_role_lines", generic_roles}};
        }
    }
    pflash_views_.remember(std::move(next));
    return true;
}

bool HttpServer::serve_pflash_region_view(
        const ParsedRequest & req,
        const std::vector<int32_t> & drafter_ids,
        const http_detail::PflashChatTurnSpan & turn,
        const http_detail::PflashRegionPlan & plan,
        const std::vector<PFlashTokenSpan> & kept_spans,
        std::vector<int32_t> & served,
        int & snapshot_cut,
        json & stats) {
    snapshot_cut = -1;
    stats = nullptr;
    const int input = (int) drafter_ids.size();
    if (!plan.active || plan.regions.empty() || turn.generation_begin <= 0 ||
        turn.generation_begin >= input || plan.cut_raw < 0 ||
        plan.cut_raw > turn.generation_begin) {
        return false;
    }
    const auto generation = tokenizer_.encode(drafter_tokenizer_->decode(
        std::vector<int32_t>(drafter_ids.begin() + turn.generation_begin,
                             drafter_ids.end())));
    const auto ends_with_generation = [&generation] (
            const std::vector<int32_t> & tokens) {
        return !generation.empty() && tokens.size() > generation.size() &&
            std::equal(generation.begin(), generation.end(),
                       tokens.end() - (long) generation.size());
    };
    if (!ends_with_generation(req.prompt_tokens)) return false;
    http_detail::PflashChatView view;
    if (plan.has_view) {
        if (!pflash_views_.find(req.prompt_tokens, drafter_ids, view) ||
            plan.cut_view < 0 || plan.cut_view > (int) view.view_tokens.size()) {
            return false;
        }
    } else if (plan.raw_base) {
        // The raw prompt up to the cut stands in for a view: the earlier
        // conversation, served verbatim and held as one raw span.
        if (plan.cut_raw <= 0 || plan.cut_view <= 0 ||
            plan.cut_view > (int) req.prompt_tokens.size()) {
            return false;
        }
        view.view_tokens = req.prompt_tokens;
        view.spans = {{0, plan.cut_raw}};
        view.pieces = {{0, plan.cut_raw, false}};
    }
    const bool based = plan.has_view || plan.raw_base;
    // From the cut on: the regions as the selection keeps them, every other
    // raw token verbatim (mandatory, so the selection kept it as well).
    const auto outside = http_detail::pflash_complement_token_spans(plan.regions, input);
    const auto region_kept = http_detail::pflash_subtract_token_spans(
        http_detail::canonicalize_pflash_token_spans(
            std::vector<PFlashTokenSpan>(kept_spans)), outside);
    std::vector<PFlashTokenSpan> delta = region_kept;
    for (const auto & span : http_detail::pflash_subtract_token_spans(
             {{plan.cut_raw, input}}, plan.regions)) {
        delta.push_back(span);
    }
    delta = http_detail::pflash_subtract_token_spans(
        http_detail::canonicalize_pflash_token_spans(std::move(delta)),
        {{0, plan.cut_raw}});
    const int tail_after = (std::min)((std::max)(plan.tail_after, plan.cut_raw),
                                      turn.generation_begin);
    std::vector<PFlashTokenSpan> block_part;
    std::vector<PFlashTokenSpan> rest_part;
    for (const auto & span : delta) {
        if (span.begin < tail_after) {
            block_part.push_back({span.begin, (std::min)(span.end, tail_after)});
        }
        if (span.end > tail_after) {
            rest_part.push_back({(std::max)(span.begin, tail_after), span.end});
        }
    }
    const auto join = [&] (const std::vector<PFlashTokenSpan> & spans) {
        std::string text;
        for (const auto & span : spans) {
            if (span.end > span.begin) {
                text += drafter_tokenizer_->decode(std::vector<int32_t>(
                    drafter_ids.begin() + span.begin, drafter_ids.begin() + span.end));
            }
        }
        return text;
    };
    const auto block_tokens = tokenizer_.encode(join(block_part));
    const auto rest_tokens = tokenizer_.encode(join(rest_part));
    served.clear();
    if (based) {
        served.assign(view.view_tokens.begin(), view.view_tokens.begin() + plan.cut_view);
    }
    served.insert(served.end(), block_tokens.begin(), block_tokens.end());
    const int tail_view = (int) served.size();
    served.insert(served.end(), rest_tokens.begin(), rest_tokens.end());
    if (!ends_with_generation(served)) return false;
    if (config_.max_ctx > 0 && (int) served.size() + req.max_output > config_.max_ctx) {
        return false;
    }

    http_detail::PflashChatView next;
    next.raw_tokens = req.prompt_tokens;
    next.raw_gen_begin = (int) (req.prompt_tokens.size() - generation.size());
    next.drafter_ids = drafter_ids;
    next.drafter_gen_begin = turn.generation_begin;
    next.view_tokens = served;
    next.view_gen_begin = (int) (served.size() - generation.size());
    std::vector<PFlashTokenSpan> spans;
    if (based) {
        spans = http_detail::pflash_subtract_token_spans(
            http_detail::canonicalize_pflash_token_spans(
                std::vector<PFlashTokenSpan>(view.spans)), {{plan.cut_raw, input}});
        for (auto piece : view.pieces) {
            if (piece.begin >= plan.cut_raw) continue;
            piece.end = (std::min)(piece.end, plan.cut_raw);
            next.pieces.push_back(piece);
        }
    }
    spans.insert(spans.end(), delta.begin(), delta.end());
    next.spans = http_detail::canonicalize_pflash_token_spans(std::move(spans));
    for (const auto & span : delta) next.pieces.push_back({span.begin, span.end, false});
    next.turns = (plan.has_view ? view.turns : 0) + 1;
    snapshot_cut = next.view_gen_begin;
    int region_tokens = 0;
    for (const auto & span : plan.regions) region_tokens += span.end - span.begin;
    int region_kept_tokens = 0;
    for (const auto & span : region_kept) region_kept_tokens += span.end - span.begin;
    std::fprintf(stderr,
        "[pflash-view] %s turn=%d served=%zu reused=%d%s region=%d kept=%d\n",
        plan.mode.c_str(), next.turns, served.size(), based ? plan.cut_view : 0,
        plan.raw_base ? " (raw base)" : "", region_tokens, region_kept_tokens);
    std::fflush(stderr);
    stats = {{"mode", plan.mode}, {"turn", next.turns},
             {"served_tokens", served.size()},
             {"reused_tokens", plan.has_view ? plan.cut_view : 0},
             {"region_drafter_tokens", region_tokens},
             {"region_kept_drafter_tokens", region_kept_tokens},
             {"pieces", http_detail::pflash_view_pieces_json(next.pieces)}};
    if (plan.raw_base) stats["raw_base_tokens"] = plan.cut_view;
    if (const char * trace = std::getenv("PFLASH_VIEW_TRACE_PATH"); trace && *trace) {
        stats["cut_raw"] = plan.cut_raw;
        stats["drafter_tokens"] = input;
    }
    pflash_views_.remember(std::move(next));
    return true;
}

size_t HttpServer::forget_unified_views(const std::vector<int32_t> & raw_tokens) {
    if (!pflash_unified_) return 0;
    auto lineages = pflash_unified_views_.lineages_continued_by(raw_tokens);
    const auto on_disk = pflash_unified_disk_views_.lineages_continued_by(raw_tokens);
    lineages.insert(lineages.end(), on_disk.begin(), on_disk.end());
    std::sort(lineages.begin(), lineages.end());
    lineages.erase(std::unique(lineages.begin(), lineages.end()), lineages.end());
    for (const uint64_t lineage : lineages) {
        pflash_unified_views_.forget_lineage(lineage);
        pflash_unified_disk_views_.forget_lineage(lineage);
    }
    return lineages.size();
}

std::string HttpServer::apply_unified_view(
        const ParsedRequest & req, PreparedPrompt & prepared,
        const http_detail::PflashChatTurnSpan & turn,
        const std::vector<ChatMessage> & messages,
        ModelBackend::CompressRequest request, int chunk_size,
        const std::string & not_ready, std::string & handback) {
    namespace pu = pflash_unified;
    const std::vector<int32_t> & raw = req.prompt_tokens;
    const int prompt_tokens = (int) raw.size();
    json stats = {{"version", 1}};
    const auto count = [this] (const std::string & key) {
        std::lock_guard<std::mutex> lk(pflash_failure_mu_);
        ++pflash_unified_counts_[key];
    };
    // Full prefill of the raw prompt: nothing is paged and no view continues.
    const auto serve_raw = [&] (const std::string & reason) {
        count("raw." + reason);
        stats["decision"] = "raw";
        stats["reason"] = reason;
        prepared.pflash_stats = {{"unified", stats}};
        std::fprintf(stderr, "[pflash-unified] raw prompt (%d tokens): %s\n", prompt_tokens, reason.c_str());
        std::fflush(stderr);
        return std::string();
    };
    // The chat view serves what the unified view does not take; nothing here changed yet. Once it accepts the
    // request, no later turn continues a view of this conversation (prepare_prompt).
    const auto hand_back = [&] (const std::string & reason) {
        count("handback." + reason);
        handback = reason;
        std::fprintf(stderr, "[pflash-unified] handed back to the chat view (%d tokens): %s\n", prompt_tokens,
                     reason.c_str());
        std::fflush(stderr);
        return std::string();
    };
    if (!not_ready.empty()) return hand_back(not_ready);

    const std::vector<int32_t> & drafter_ids = request.input_ids;
    const int input = (int) drafter_ids.size();
    if (turn.generation_begin <= 0 || turn.generation_begin >= input) return hand_back("no_generation_prompt");
    // The generation prompt, in target tokens: the raw prompt and every served prompt end with it.
    const auto generation = tokenizer_.encode(drafter_tokenizer_->decode(
        std::vector<int32_t>(drafter_ids.begin() + turn.generation_begin, drafter_ids.end())));
    if (generation.empty() || generation.size() >= raw.size() ||
        !std::equal(generation.begin(), generation.end(), raw.end() - (long) generation.size())) {
        return hand_back("generation_mismatch");
    }
    const int raw_gen_begin = prompt_tokens - (int) generation.size();

    // Both tokenisations decode to one text; units are cut where both start a token.
    const auto td = http_detail::decode_prompt_with_offsets(tokenizer_, raw);
    const auto dd = http_detail::decode_prompt_with_offsets(*drafter_tokenizer_, drafter_ids);
    if (td.text != dd.text) return hand_back("text_mismatch");
    const std::string & text = td.text;
    // The first token starting exactly at ``at`` (the text's end maps to the token count), else -1.
    const auto token_at = [] (const http_detail::DecodedPrompt & decoded, size_t at) {
        const auto it = std::lower_bound(decoded.token_begin.begin(), decoded.token_begin.end(), at);
        if (it == decoded.token_begin.end()) {
            return at == decoded.text.size() ? (int) decoded.token_begin.size() : -1;
        }
        return *it == at ? (int) (it - decoded.token_begin.begin()) : -1;
    };
    // The first token starting at or after ``at``.
    const auto token_from = [] (const http_detail::DecodedPrompt & decoded, size_t at) {
        return (int) (std::lower_bound(decoded.token_begin.begin(), decoded.token_begin.end(), at) -
                      decoded.token_begin.begin());
    };
    const auto target_estimate = [&] (int drafter_tokens) {
        return (int) std::ceil((double) prompt_tokens * (double) drafter_tokens / (double) input);
    };
    const auto encode_len = [&] (const std::string & piece) { return (int) tokenizer_.encode(piece).size(); };
    const auto & pins = request.required_instruction_spans;
    const int query_end = request.score_query_end;
    const int query_begin = request.score_query_end - request.score_query_tokens;
    // The selector's own rule for a structurally required candidate (pflash_chunk_is_structurally_required).
    const auto pinned = [&] (int begin, int end) {
        if (begin < query_end && end > query_begin) return true;
        if (!request.query_suffix_candidates && end > query_end) return true;
        for (const auto & span : pins) {
            if (begin < span.end && end > span.begin) return true;
        }
        return false;
    };

    // The newest turn -- every message since the last reply: the run of user messages (a document sent on
    // its own, then the question), or the group of tool messages -- holds the candidates.
    struct Candidate {
        size_t message = 0;
        pu::RegionKind kind = pu::RegionKind::UserContent;
        std::string brief;
        std::string call_id;
        std::string skip;          // why it cannot be a region; "" = it may be
        bool located = false;
        size_t cb = 0, ce = 0;
        int db = 0, de = 0, tb = 0, te = 0;
        pu::Lines lines;
        bool lines_ready = false;
    };
    std::vector<Candidate> newest;
    const size_t last = messages.size() - 1;
    if (messages[last].role == "user") {
        size_t first = last;
        while (first > 0 && messages[first - 1].role == "user") --first;
        for (size_t index = first; index <= last; ++index) {
            Candidate c;
            c.message = index;
            newest.push_back(std::move(c));
        }
    } else if (messages[last].role == "tool") {
        size_t first = last;
        while (first > 0 && messages[first - 1].role == "tool") --first;
        const ChatMessage * caller =
            first > 0 && messages[first - 1].role == "assistant" ? &messages[first - 1] : nullptr;
        for (size_t index = first; index <= last; ++index) {
            Candidate c;
            c.message = index;
            c.kind = pu::RegionKind::ToolOutput;
            const std::string & id = messages[index].tool_call_id;
            int calls = 0;
            int results = 0;
            const ChatToolCall * call = nullptr;
            if (caller && !id.empty()) {
                for (const auto & entry : caller->tool_calls) {
                    if (entry.id == id) {
                        ++calls;
                        call = &entry;
                    }
                }
                for (size_t other = first; other <= last; ++other) {
                    if (messages[other].tool_call_id == id) ++results;
                }
            }
            if (calls != 1 || results != 1) {
                c.skip = "unpaired";
            } else {
                c.call_id = id;
                c.brief = pu::call_brief(call->name, call->arguments);
                if (!pflash_region_tools_.listed(call->name)) c.skip = "unlisted";
            }
            newest.push_back(std::move(c));
        }
    }
    // Their exact content in the prompt text, from one render with a placeholder per newest message.
    if (!newest.empty()) {
        std::vector<ChatMessage> placeholder_messages = messages;
        std::vector<std::string> placeholders(messages.size());
        std::vector<std::string> contents(messages.size());
        for (const auto & c : newest) {
            placeholders[c.message] = "\x1e" "PFLASH-UNIFIED-" + std::to_string(c.message) + "\x1f";
            contents[c.message] = messages[c.message].content;
            placeholder_messages[c.message].content = placeholders[c.message];
        }
        std::string rendered;
        std::string error;
        std::vector<std::pair<size_t, size_t>> chars;
        if (render_messages_to_text(placeholder_messages, req, /*add_generation_prompt=*/true, rendered, error) &&
            http_detail::pflash_content_char_spans(rendered, placeholders, contents, text, chars, error)) {
            for (auto & c : newest) {
                c.located = true;
                c.cb = chars[c.message].first;
                c.ce = chars[c.message].second;
                c.db = http_detail::token_at_offset(dd, c.cb);
                c.de = token_from(dd, c.ce);
                c.tb = http_detail::token_at_offset(td, c.cb);
                c.te = token_from(td, c.ce);
            }
        } else {
            count("unlocated");
            std::fprintf(stderr, "[pflash-unified] newest turn not located (%s): nothing is paged\n",
                         error.c_str());
            for (auto & c : newest) c.skip = "unlocated";
        }
    }
    const auto candidate_lines = [&] (Candidate & c) -> const pu::Lines & {
        if (!c.lines_ready) {
            c.lines = pu::detect_lines(text, c.cb, c.ce);
            c.lines_ready = true;
        }
        return c.lines;
    };

    pu::View view;
    bool has_view = pflash_unified_views_.find(raw, drafter_ids, pflash_unified_config_, view);
    // A view the RAM store no longer holds (a restart, or its eviction) may be on disk: a longer one wins.
    if (pflash_unified_disk_views_.enabled()) {
        const size_t ram_len = has_view ? (size_t) (std::max)(0, view.raw_gen_begin) : 0;
        pu::View stored;
        bool loaded = pflash_unified_disk_views_.find(raw, drafter_ids, pflash_unified_config_, ram_len, stored);
        // A view read from disk is consistent in itself (view_from_json); its regions must also be this request's
        // bytes, and its unit cuts the token starts they were taken at, in both tokenisations.
        const auto matches_text = [&] (const pu::View & v) {
            for (const auto & region : v.regions) {
                if (region.ce > text.size() ||
                    pu::sha1_hex(text.substr(region.cb, region.ce - region.cb)) != region.content_sha) {
                    return false;
                }
                const auto & cuts = region.layout.cuts;
                for (size_t u = 0; u + 1 < cuts.size(); ++u) {
                    if (token_at(td, cuts[u]) != region.unit_tb[u] || token_at(td, cuts[u + 1]) != region.unit_te[u] ||
                        token_at(dd, cuts[u]) != region.unit_db[u] || token_at(dd, cuts[u + 1]) != region.unit_de[u]) {
                        return false;
                    }
                }
            }
            return true;
        };
        if (loaded && !matches_text(stored)) {
            std::fprintf(stderr, "[pflash-unified] disk view of lineage %llu does not match the prompt: dropped\n",
                         (unsigned long long) stored.lineage);
            std::fflush(stderr);
            pflash_unified_disk_views_.forget_lineage(stored.lineage);
            count("disk_view_rejected");
            loaded = false;
        }
        if (loaded) {
            count("disk_view");
            stats["disk_view"] = true;
            view = std::move(stored);
            has_view = true;
        }
    }
    // A repeat serves what was served only when the view holds every span this request pins (a pflash_required
    // text added on a retry may name lines the view left out); otherwise the request starts a new view.
    if (has_view && view.raw_tokens == raw && view.drafter_ids == drafter_ids) {
        bool served = true;
        for (const auto & span : pins) {
            for (const auto & region : view.regions) {
                for (size_t u = 0; u < region.unit_db.size(); ++u) {
                    served = served && !(span.begin < region.unit_de[u] && span.end > region.unit_db[u] &&
                                         !region.held[u] && !region.mandatory[u]);
                }
            }
        }
        if (!served) {
            count("repeat.pins_unserved");
            has_view = false;
        }
    }
    const bool found_view = has_view;
    if (has_view && view.raw_tokens == raw && view.drafter_ids == drafter_ids) {
        // The same request again: what was served.
        count("repeat");
        stats["decision"] = "repeat";
        stats["turn"] = view.turns;
        stats["served_tokens"] = view.view_tokens.size();
        prepared.tokens = view.view_tokens;
        prepared.compressed = view.view_tokens != raw;
        prepared.unified_view = true;
        prepared.snapshot_cut = view.view_gen_begin;
        prepared.pflash_stats = {{"unified", stats}, {"compress_ms", 0.0},
                                 {"drafter_input_tokens", input}};
        if (prepared.compressed) trace_pflash_served(req, prepared);
        std::fprintf(stderr, "[pflash-unified] repeat turn=%d served=%zu\n", view.turns, view.view_tokens.size());
        std::fflush(stderr);
        return {};
    }
    if (has_view && !(view.drafter_gen_begin < turn.generation_begin && view.raw_gen_begin <= raw_gen_begin &&
                      view.view_gen_begin > 0 && view.view_gen_begin <= (int) view.view_tokens.size())) {
        count("reset.unusable");
        stats["reset"] = "unusable";
        has_view = false;
    }
    // A conversation the disk session cache holds (a restarted server, or a session written out on eviction) counts
    // as cached, as in the chat view. Disk cache off: no cached record, the stats are unchanged.
    const auto restorable_prefix = restorable_prefix_tokens(raw, req.disk_cache_policy);
    const int restorable = restorable_prefix.tokens;
    if (!disk_cache_.disabled()) stats["cached"] = {{"tokens", restorable}, {"source", restorable_prefix.source}};

    for (int pass = 0; pass < 2; ++pass) {
        const bool continues = has_view && pass == 0;
        const int new_from = continues ? view.drafter_gen_begin : 0;
        std::vector<pu::Region> held_regions = continues ? view.regions : std::vector<pu::Region>{};

        // Admission: eligible, new, uncached, big, not a re-read.
        json admission = json::array();
        std::vector<size_t> region_candidates;
        std::vector<size_t> verbatim;
        for (size_t k = 0; k < newest.size(); ++k) {
            Candidate & c = newest[k];
            std::string why = c.skip;
            int droppable = 0;
            if (why.empty() && c.db < new_from) why = "not_new";
            if (why.empty() && c.tb < restorable) why = "cached";
            if (why.empty()) {
                droppable = target_estimate((std::max)(0, (c.de - c.db) -
                    http_detail::pflash_span_tokens_within(pins, c.db, c.de)));
                if (droppable < config_.pflash_threshold) why = "small";
            }
            if (why.empty() && candidate_lines(c).count() < 5) why = "no_lines";
            if (why.empty() && c.kind == pu::RegionKind::ToolOutput) {
                const auto hashes = pu::line_hashes(text, candidate_lines(c));
                for (const auto & region : held_regions) {
                    if (pu::match_reread(hashes, region.hashes).is_reread()) {
                        why = "reread";
                        break;
                    }
                }
            }
            admission.push_back({{"message", c.message},
                                 {"kind", c.kind == pu::RegionKind::ToolOutput ? "tool_output" : "user_content"},
                                 {"decision", why.empty() ? "region" : why},
                                 {"droppable", droppable}});
            if (why.empty()) {
                region_candidates.push_back(k);
            } else {
                verbatim.push_back(k);
            }
        }

        // A conversation the view would start: when the part it can never page -- the uncached, unpinned history
        // before the newest turn, and newest content it cannot cut into lines -- is a threshold's worth, the chat
        // view takes the request (its cache admission compresses such a prompt) instead of a raw prefill of it.
        // Newest content the view serves verbatim by design (unlisted tool outputs, small messages) does not count.
        if (!continues) {
            const int uncached_from = (std::min)(turn.generation_begin, (int) std::floor(
                (double) restorable * (double) input / (double) (std::max)(1, prompt_tokens)));
            const auto unpinned = [&] (int begin, int end) {
                begin = (std::max)(begin, uncached_from);
                return end > begin ? (end - begin) - http_detail::pflash_span_tokens_within(pins, begin, end) : 0;
            };
            int newest_begin = turn.generation_begin;
            for (const auto & c : newest) {
                if (c.located) newest_begin = (std::min)(newest_begin, c.db);
            }
            int cannot_page = unpinned(0, newest_begin);
            for (size_t k = 0; k < newest.size(); ++k) {
                if (admission[k].value("decision", "") == "no_lines") cannot_page += unpinned(newest[k].db, newest[k].de);
            }
            if (target_estimate(cannot_page) >= config_.pflash_threshold) {
                // Once the chat view accepts it, the view this turn would have continued goes, with its lineage.
                return hand_back(found_view ? "reset_uncached" : "uncached");
            }
        }

        // Recall trigger: a new user question on a view with regions.
        bool recall = false;
        int recall_at = -1;
        if (continues && pflash_unified_recall_ && !held_regions.empty() && !newest.empty() &&
            newest.back().kind == pu::RegionKind::UserContent && newest.back().located &&
            newest.back().db >= view.drafter_gen_begin) {
            recall_at = token_at(td, newest.back().cb);
            // Gate arms: no drafter pass for a question when every unit of every region is already held.
            bool unheld = !(pflash_recall_gate_ > 0.0);
            for (const auto & region : held_regions) {
                unheld = unheld || std::find(region.held.begin(), region.held.end(), 0) != region.held.end();
            }
            if (recall_at < view.raw_gen_begin) {
                count("recall.unplaceable");
            } else if (!unheld) {
                count("recall.complete");
                stats["recall_gate"] = {{"decision", "complete"}};
            } else {
                recall = true;
            }
        }

        // One drafter pass scores the regions to page and, at a question, the earlier regions.
        std::vector<std::pair<PFlashTokenSpan, double>> lifts;
        ModelBackend::CompressResult result;
        bool scored = false;
        double compress_ms = 0.0;
        if (!region_candidates.empty() || recall) {
            std::vector<PFlashTokenSpan> scored_spans;
            for (size_t k : region_candidates) scored_spans.push_back({newest[k].db, newest[k].de});
            if (recall) {
                for (const auto & region : held_regions) scored_spans.push_back({region.db, region.de});
            }
            scored_spans = http_detail::canonicalize_pflash_token_spans(std::move(scored_spans));
            // Everything outside the scored regions is kept: the selector spends its budget inside them.
            auto mandatory = pins;
            const auto outside = http_detail::pflash_complement_token_spans(scored_spans, input);
            mandatory.insert(mandatory.end(), outside.begin(), outside.end());
            mandatory = http_detail::canonicalize_pflash_token_spans(std::move(mandatory));
            const int kept = http_detail::pflash_kept_tokens(input, chunk_size, query_begin, query_end, mandatory,
                                                             !request.query_suffix_candidates);
            const float ratio = http_detail::resolve_pflash_keep_ratio(
                pflash_keep_ratio(config_, (std::max)(1, target_estimate((std::max)(0, input - kept)))),
                req.session_id, sessions_);
            ModelBackend::CompressRequest call = request;
            call.required_instruction_spans = std::move(mandatory);
            call.keep_ratio = (float) http_detail::pflash_effective_keep_ratio(input, kept, ratio);
            call.drafter_path = config_.pflash_drafter_path;
            call.drafter_gpu = config_.pflash_drafter_gpu;
            call.skip_park = config_.pflash_skip_park;
            call.residency_action = resolve_draft_residency_action(
                config_.draft_residency,
                DraftResidencyContext{DraftResidencyUse::PFlashCompress, config_.lazy_draft,
                                      !config_.draft_path.empty()});
            const auto started = std::chrono::steady_clock::now();
            result = backend_.compress(call);
            compress_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            if (result.ok && !result.candidate_lifts.empty()) {
                lifts = result.candidate_lifts;
                scored = true;
            } else {
                const char * detail = result.ok ? "no candidate scores" : luce_last_error();
                count(result.ok ? "drafter.no_lifts" : "drafter.failed");
                stats["drafter_error"] = detail ? detail : "";
                // PFLASH_FAIL_CLOSED=1 refuses the request, as a failed compression on the chat view; otherwise the
                // newest content is served verbatim and the view continues.
                if (http_detail::pflash_fail_closed_from_env()) {
                    prepared.pflash_failure_detail = detail ? detail : "";
                    return "PFlash unified view: the drafter pass gave no scores";
                }
                std::fprintf(stderr, "[pflash-unified] drafter pass gave no scores (%s): nothing paged or recalled\n",
                             detail ? detail : "");
                std::fflush(stderr);
            }
        }
        // The request probe's own pins (the question it found, its recency pin) stay whole, as in the chat view's
        // selection. An earlier region kept its question's units when it was paged.
        const auto probe_pinned = [&] (int begin, int end) {
            for (const auto & span : result.probe_pins) {
                if (begin < span.end && end > span.begin) return true;
            }
            return false;
        };
        // A unit's score: the token-weighted mean lift of the candidates over it.
        const auto unit_score = [&] (int begin, int end) {
            double sum = 0.0;
            double weight = 0.0;
            for (const auto & lift : lifts) {
                const int lo = (std::max)(begin, lift.first.begin);
                const int hi = (std::min)(end, lift.first.end);
                if (hi > lo) {
                    sum += lift.second * (double) (hi - lo);
                    weight += (double) (hi - lo);
                }
            }
            return weight > 0.0 ? sum / weight : std::nan("");
        };
        // A region's cap: the release curve on its droppable tokens (kept in the region); a pass whose stage 2
        // widened the budget widens it once, paging and recall alike.
        const auto base_cap = [&] (int droppable) {
            const float ratio = http_detail::resolve_pflash_keep_ratio(
                pflash_keep_ratio(config_, (std::max)(1, droppable)), req.session_id, sessions_);
            return (int) std::floor((double) ratio * (double) droppable);
        };
        const auto widened_cap = [&] (int cap, int droppable) {
            return result.budget_widened ? (std::min)(droppable, cap + cap / 2) : cap;
        };

        // Paging, per region: units between clean line cuts, admitted by score under the cap.
        struct Splice {
            int begin = 0;                   // raw target tokens [begin, end) replaced by ``tokens``
            int end = 0;
            std::vector<int32_t> tokens;
            std::vector<pu::Piece> pieces;   // relative to ``tokens``
        };
        std::vector<Splice> splices;
        std::vector<pu::Region> new_regions;
        const int region_base = (int) held_regions.size();
        for (size_t slot = 0; slot < region_candidates.size(); ++slot) {
            const size_t k = region_candidates[slot];
            Candidate & c = newest[k];
            const auto verbatim_because = [&] (const std::string & why) {
                admission[k]["decision"] = why;
                count("verbatim." + why);
                verbatim.push_back(k);
            };
            if (!scored) {
                verbatim_because("unscored");
                continue;
            }
            const pu::Lines & lines = candidate_lines(c);
            std::vector<size_t> common;
            {
                const auto t0 = std::lower_bound(td.token_begin.begin(), td.token_begin.end(), lines.begin);
                const auto t1 = std::upper_bound(td.token_begin.begin(), td.token_begin.end(), lines.end);
                const auto d0 = std::lower_bound(dd.token_begin.begin(), dd.token_begin.end(), lines.begin);
                const auto d1 = std::upper_bound(dd.token_begin.begin(), dd.token_begin.end(), lines.end);
                std::set_intersection(t0, t1, d0, d1, std::back_inserter(common));
                common.erase(std::unique(common.begin(), common.end()), common.end());
            }
            std::vector<size_t> proposed;
            for (const auto & lift : lifts) {
                if (lift.first.begin > c.db && lift.first.begin < c.de) {
                    proposed.push_back(dd.token_begin[(size_t) lift.first.begin]);
                }
            }
            pu::Region region;
            std::string why;
            if (!pu::layout_region(lines, common, proposed, region.layout, why)) {
                verbatim_because("layout");
                continue;
            }
            const int units = region.layout.units();
            std::vector<int> cut_t;
            std::vector<int> cut_d;
            for (size_t cut : region.layout.cuts) {
                cut_t.push_back(token_at(td, cut));
                cut_d.push_back(token_at(dd, cut));
            }
            if (std::count(cut_t.begin(), cut_t.end(), -1) || std::count(cut_d.begin(), cut_d.end(), -1)) {
                verbatim_because("unmapped_cut");
                continue;
            }
            std::vector<int> cost((size_t) units);
            std::vector<double> score((size_t) units);
            region.mandatory.assign((size_t) units, 0);
            int droppable = 0;
            for (int u = 0; u < units; ++u) {
                region.unit_tb.push_back(cut_t[(size_t) u]);
                region.unit_te.push_back(cut_t[(size_t) u + 1]);
                region.unit_db.push_back(cut_d[(size_t) u]);
                region.unit_de.push_back(cut_d[(size_t) u + 1]);
                cost[(size_t) u] = cut_t[(size_t) u + 1] - cut_t[(size_t) u];
                score[(size_t) u] = unit_score(cut_d[(size_t) u], cut_d[(size_t) u + 1]);
                region.mandatory[(size_t) u] =
                    pinned(cut_d[(size_t) u], cut_d[(size_t) u + 1]) ||
                    probe_pinned(cut_d[(size_t) u], cut_d[(size_t) u + 1]) ? 1 : 0;
                if (!region.mandatory[(size_t) u]) droppable += cost[(size_t) u];
            }
            if (droppable <= 0) {
                verbatim_because("all_pinned");
                continue;
            }
            const int unwidened = base_cap(droppable);
            const int cap = widened_cap(unwidened, droppable);
            std::map<std::pair<int, int>, int> marker_costs;
            const auto marker_line = [&] (int first, int last_unit) {
                return pu::inserted_line(text, lines, region.layout.cuts[(size_t) first],
                                         region.layout.cuts[(size_t) last_unit + 1],
                                         pu::marker_text(region.layout.line_count(first, last_unit)));
            };
            const auto marker_cost = [&] (int first, int last_unit) {
                const auto key = std::make_pair(first, last_unit);
                const auto it = marker_costs.find(key);
                if (it != marker_costs.end()) return it->second;
                const int tokens = encode_len(marker_line(first, last_unit));
                marker_costs.emplace(key, tokens);
                return tokens;
            };
            const int total_lines = (int) lines.count();
            const auto preamble_line = [&] (int shown) {
                return pu::inserted_line(text, lines, region.layout.head_end, region.layout.head_end,
                                         pu::preamble_text(c.kind, shown, total_lines));
            };
            int fixed = encode_len(preamble_line(total_lines));
            const auto order = pu::rank_units(score, region.mandatory);
            pu::Admission admitted;
            std::string preamble;
            for (int attempt = 0; attempt < 3; ++attempt) {
                admitted = pu::admit_paging(cost, region.mandatory, order, fixed, cap, marker_cost);
                if (!admitted.ok) break;
                int dropped_lines = 0;
                for (int u = 0; u < units; ++u) {
                    if (!admitted.admitted[(size_t) u]) dropped_lines += region.layout.line_count(u, u);
                }
                preamble = preamble_line(total_lines - dropped_lines);
                const int actual = encode_len(preamble);
                if (admitted.cost - fixed + actual <= cap) {
                    admitted.cost += actual - fixed;
                    break;
                }
                fixed = actual;
                admitted.ok = false;
            }
            if (!admitted.ok) {
                verbatim_because("cap_below_overhead");
                continue;
            }
            if (std::find(admitted.admitted.begin(), admitted.admitted.end(), 0) == admitted.admitted.end()) {
                verbatim_because("nothing_dropped");
                continue;
            }
            const int region_index = region_base + (int) new_regions.size();
            Splice splice;
            splice.begin = cut_t.front();
            splice.end = cut_t.back();
            const auto add = [&] (std::vector<int32_t> tokens, const char * kind) {
                const int at = (int) splice.tokens.size();
                splice.tokens.insert(splice.tokens.end(), tokens.begin(), tokens.end());
                splice.pieces.push_back({at, (int) splice.tokens.size(), kind, region_index, -1});
            };
            add(tokenizer_.encode(preamble), "preamble");
            int kept_tokens = 0;
            for (int u = 0; u < units;) {
                if (admitted.admitted[(size_t) u]) {
                    int b = u;
                    while (b + 1 < units && admitted.admitted[(size_t) b + 1]) ++b;
                    add(std::vector<int32_t>(raw.begin() + cut_t[(size_t) u], raw.begin() + cut_t[(size_t) b + 1]),
                        "paged_text");
                    kept_tokens += cut_t[(size_t) b + 1] - cut_t[(size_t) u];
                    u = b + 1;
                    continue;
                }
                int b = u;
                while (b + 1 < units && !admitted.admitted[(size_t) b + 1]) ++b;
                add(tokenizer_.encode(marker_line(u, b)), "marker");
                u = b + 1;
            }
            region.kind = c.kind;
            region.brief = c.brief;
            region.tool_call_id = c.call_id;
            region.content_sha = pu::sha1_hex(text.substr(c.cb, c.ce - c.cb));
            region.cb = c.cb;
            region.ce = c.ce;
            region.tb = c.tb;
            region.te = c.te;
            region.db = c.db;
            region.de = c.de;
            region.lines = lines;
            region.hashes = pu::line_hashes(text, lines);
            region.kept = admitted.admitted;
            region.held = admitted.admitted;
            region.reread.assign(lines.count(), 0);
            region.cap = unwidened;
            region.droppable = droppable;
            if (pflash_recall_trace_units_) {
                std::vector<long> cuts;
                for (size_t cut : region.layout.cuts) cuts.push_back((long) cut - (long) region.cb);
                admission[k]["trace_units"] = {{"cb", region.cb}, {"content_bytes", region.ce - region.cb},
                                               {"cuts", cuts}, {"kept", region.kept}};
            }
            admission[k]["units"] = units;
            admission[k]["kept_units"] = (int) std::count(admitted.admitted.begin(), admitted.admitted.end(), 1);
            admission[k]["mandatory_units"] =
                (int) std::count(region.mandatory.begin(), region.mandatory.end(), 1);
            admission[k]["cap"] = cap;
            admission[k]["unwidened_cap"] = unwidened;
            admission[k]["optional_served"] = admitted.cost;
            admission[k]["region_tokens"] = splice.end - splice.begin;
            admission[k]["served_tokens"] = splice.tokens.size();
            admission[k]["kept_tokens"] = kept_tokens;
            admission[k]["lines"] = total_lines;
            admission[k]["escaped_lines"] = lines.escaped;
            admission[k]["region"] = region_index;
            count(c.kind == pu::RegionKind::ToolOutput ? "region.tool_output" : "region.user_content");
            new_regions.push_back(std::move(region));
            splices.push_back(std::move(splice));
        }

        // Re-reads: a newest message served verbatim (rejected above, or fallen back while paging) that repeats lines of
        // an earlier region covers them; a unit whose informative lines are all covered is held (never recalled).
        int covered_units = 0;
        for (size_t k : verbatim) {
            Candidate & c = newest[k];
            if (!c.located || c.ce <= c.cb || held_regions.empty()) continue;
            const auto hashes = pu::line_hashes(text, candidate_lines(c));
            for (auto & region : held_regions) {
                const auto match = pu::match_reread(hashes, region.hashes);
                if (match.matched_candidate == 0) continue;
                for (size_t line = 0; line < region.reread.size(); ++line) {
                    if (match.region_covered[line]) region.reread[line] = 1;
                }
                for (int u = 0; u < region.layout.units(); ++u) {
                    if (region.held[(size_t) u]) continue;
                    if (pu::lines_covered(region.reread, match.region_informative, region.layout.first_line(u),
                                          region.layout.cut_line[(size_t) u + 1])) {
                        region.held[(size_t) u] = 1;
                        ++covered_units;
                    }
                }
            }
        }

        // Recall: each earlier region's unheld units, best first, under the region's cap, appended as
        // labelled excerpts at the start of the question; nothing served before is touched.
        json recall_stats = nullptr;
        json gate_stats = nullptr;
        bool recall_fired = false;
        if (recall && scored) {
            Splice block;
            block.begin = block.end = recall_at;
            const std::string open = pu::recall_open();
            const std::string close = pu::recall_close();
            const int wrapper = encode_len(open) + encode_len(close);
            size_t largest = 0;
            for (size_t ri = 1; ri < held_regions.size(); ++ri) {
                if (held_regions[ri].cap > held_regions[largest].cap) largest = ri;
            }
            // Each region's candidates: unheld units, their scores, costs and the region's cap.
            const size_t region_count = held_regions.size();
            std::vector<pu::RecallRegion> candidates(region_count);
            std::vector<std::vector<int>> orders(region_count);
            std::vector<int> region_caps(region_count);
            for (size_t ri = 0; ri < region_count; ++ri) {
                pu::Region & region = held_regions[ri];
                pu::RecallRegion & rc = candidates[ri];
                const int units = region.layout.units();
                rc.score.assign((size_t) units, 0.0);
                rc.unit_cost.assign((size_t) units, 0);
                rc.held = region.held;
                for (int u = 0; u < units; ++u) {
                    rc.score[(size_t) u] = unit_score(region.unit_db[(size_t) u], region.unit_de[(size_t) u]);
                    if (region.held[(size_t) u]) continue;
                    if (region.lines.escaped) {
                        const size_t from = region.layout.cuts[(size_t) u];
                        const size_t to = region.layout.cuts[(size_t) u + 1];
                        rc.unit_cost[(size_t) u] = encode_len(pu::json_unescape(text, from, to));
                    } else {
                        rc.unit_cost[(size_t) u] = region.unit_te[(size_t) u] - region.unit_tb[(size_t) u];
                    }
                }
                rc.label_cost = encode_len(pu::recall_label_text(region.kind, region.brief) + "\n") + encode_len("\n");
                region_caps[ri] = widened_cap(region.cap, region.droppable);
                rc.cap = region_caps[ri] - (ri == largest ? wrapper : 0);
                orders[ri] = pu::rank_units(rc.score, rc.held);
            }
            std::vector<pu::Admission> planned(region_count);
            const auto nothing = [&] {
                for (size_t ri = 0; ri < region_count; ++ri) {
                    planned[ri].ok = true;
                    planned[ri].admitted.assign((size_t) held_regions[ri].layout.units(), 0);
                }
            };
            // The head's mass on a span: its candidates' lift x overlap / input (the lifts are mass per token
            // relative to uniform, pflash_compress.cpp), pinned candidates excluded so that D <= S. S: every
            // candidate that is not pinned, less what this request's new regions dropped (that content is
            // neither present nor recallable here, and counting it would dilute the deficit of the older
            // regions). A non-finite lift anywhere leaves S non-finite: unscorable, never a silent zero.
            double context = 0.0;
            double new_dropped = 0.0;
            if (pflash_recall_gate_ > 0.0 || pflash_recall_trace_units_) {
                const auto mass_of = [&] (int begin, int end) {
                    double mass = 0.0;
                    for (const auto & lift : lifts) {
                        if (pinned(lift.first.begin, lift.first.end)) continue;
                        const int lo = (std::max)(begin, lift.first.begin);
                        const int hi = (std::min)(end, lift.first.end);
                        if (hi > lo) mass += lift.second * (double) (hi - lo);
                    }
                    return mass / (double) (std::max)(1, input);
                };
                for (const auto & lift : lifts) {
                    if (pinned(lift.first.begin, lift.first.end)) continue;
                    context += lift.second * (double) (lift.first.end - lift.first.begin) / (double) (std::max)(1, input);
                }
                for (const auto & region : new_regions) {
                    for (int u = 0; u < region.layout.units(); ++u) {
                        if (!region.held[(size_t) u]) {
                            new_dropped += mass_of(region.unit_db[(size_t) u], region.unit_de[(size_t) u]);
                        }
                    }
                }
                for (size_t ri = 0; ri < region_count; ++ri) {
                    const pu::Region & region = held_regions[ri];
                    auto & mass = candidates[ri].mass;
                    mass.assign((size_t) region.layout.units(), 0.0);
                    for (int u = 0; u < region.layout.units(); ++u) {
                        if (!region.held[(size_t) u] || pflash_recall_trace_units_) {
                            mass[(size_t) u] = mass_of(region.unit_db[(size_t) u], region.unit_de[(size_t) u]);
                        }
                    }
                }
            }
            if (pflash_recall_trace_units_) {
                json regions_trace = json::array();
                for (size_t ri = 0; ri < region_count; ++ri) {
                    const pu::Region & region = held_regions[ri];
                    std::vector<int> cost((size_t) region.layout.units());
                    for (int u = 0; u < region.layout.units(); ++u) {
                        cost[(size_t) u] = region.lines.escaped
                            ? encode_len(pu::json_unescape(text, region.layout.cuts[(size_t) u],
                                                           region.layout.cuts[(size_t) u + 1]))
                            : region.unit_te[(size_t) u] - region.unit_tb[(size_t) u];
                    }
                    regions_trace.push_back({{"region", ri}, {"cb", region.cb}, {"held", region.held},
                                             {"mass", candidates[ri].mass}, {"cost", cost},
                                             {"score", candidates[ri].score}, {"cap", candidates[ri].cap},
                                             {"label_cost", candidates[ri].label_cost}});
                }
                stats["recall_units"] = {{"context_mass", context}, {"new_dropped_mass", new_dropped},
                                         {"input", input}, {"regions", regions_trace}};
            }
            const double gate_context = context - new_dropped;
            if (pflash_recall_gate_ > 0.0) {
                if (!(gate_context > 0.0) || !std::isfinite(gate_context)) {
                    count("recall.unscorable");
                    gate_stats = {{"decision", "unscorable"}, {"theta", pflash_recall_gate_}};
                    nothing();
                } else {
                    auto gated = pu::admit_recall_gated(candidates, gate_context, pflash_recall_gate_);
                    planned = std::move(gated.regions);
                    count(gated.gated ? "recall.gate_present" : "recall.gate_admitted");
                    gate_stats = {{"decision", gated.gated ? "present" : "admitted"},
                                  {"theta", pflash_recall_gate_}, {"context_mass", gate_context},
                                  {"new_dropped_mass", new_dropped}, {"deficit_before", gated.deficit_before},
                                  {"deficit_planned", gated.deficit_after}};
                }
            } else {
                for (size_t ri = 0; ri < region_count; ++ri) {
                    const int label_cost = candidates[ri].label_cost;
                    planned[ri] = pu::admit_recall(candidates[ri].unit_cost, candidates[ri].held, orders[ri],
                                                   candidates[ri].cap, [&] (int, int) { return label_cost; });
                }
            }
            double final_deficit = 0.0;
            int trimmed_units = 0;
            std::vector<int32_t> excerpts;
            std::vector<pu::Piece> excerpt_pieces;
            bool excerpts_trail = true;   // the last excerpt so far ends with a newline
            recall_stats = json::array();
            for (size_t ri = 0; ri < region_count; ++ri) {
                pu::Region & region = held_regions[ri];
                const int units = region.layout.units();
                const auto run_text = [&] (int first, int last_unit) {
                    const size_t from = region.layout.cuts[(size_t) first];
                    const size_t to = region.layout.cuts[(size_t) last_unit + 1];
                    return region.lines.escaped ? pu::json_unescape(text, from, to) : text.substr(from, to - from);
                };
                const std::string label = pu::recall_label_text(region.kind, region.brief);
                const int region_cap = region_caps[ri];
                const int cap = candidates[ri].cap;
                const auto & order = orders[ri];
                auto admitted = std::move(planned[ri]);
                const int planned_taken = admitted.taken;
                // The excerpts as served; over the cap (a tokenisation join), the lowest-ranked unit goes.
                std::vector<int32_t> tokens;
                std::vector<pu::Piece> pieces;
                bool trail = excerpts_trail;
                const int region_index = (int) ri;
                // Excerpts are separated by one blank line: label line, the run's lines, blank line, next label.
                const auto build = [&] {
                    tokens.clear();
                    pieces.clear();
                    trail = excerpts_trail;
                    bool first = excerpts.empty();
                    const auto add = [&] (std::vector<int32_t> part, const char * kind) {
                        const int at = (int) tokens.size();
                        tokens.insert(tokens.end(), part.begin(), part.end());
                        pieces.push_back({at, (int) tokens.size(), kind, region_index, -1});
                    };
                    for (int u = 0; u < units;) {
                        if (!admitted.admitted[(size_t) u]) {
                            ++u;
                            continue;
                        }
                        int b = u;
                        while (b + 1 < units && admitted.admitted[(size_t) b + 1]) ++b;
                        const std::string body = run_text(u, b);
                        const bool lead = !body.empty() && body.front() == '\n';
                        if (!first) add(tokenizer_.encode(trail ? "\n" : "\n\n"), "recall_wrapper");
                        first = false;
                        add(tokenizer_.encode(label + (lead ? "" : "\n")), "recall_label");
                        if (region.lines.escaped) {
                            add(tokenizer_.encode(body), "recall_text");
                        } else {
                            add(std::vector<int32_t>(raw.begin() + region.unit_tb[(size_t) u],
                                                     raw.begin() + region.unit_te[(size_t) b]),
                                "recall_text");
                        }
                        trail = !body.empty() && body.back() == '\n';
                        u = b + 1;
                    }
                };
                build();
                for (auto it = admitted.taken > 0 ? order.rbegin() : order.rend();
                     (int) tokens.size() > cap && it != order.rend(); ++it) {
                    if (!admitted.admitted[(size_t) *it]) continue;
                    admitted.admitted[(size_t) *it] = 0;
                    --admitted.taken;
                    build();
                }
                int appended_units = 0;
                for (int u = 0; u < units; ++u) {
                    if (!admitted.admitted[(size_t) u]) {
                        // The gate's result as served: what the cap trim removed is still missing.
                        if (!candidates[ri].held[(size_t) u] && (size_t) u < candidates[ri].mass.size()) {
                            final_deficit += candidates[ri].mass[(size_t) u];
                        }
                        continue;
                    }
                    region.held[(size_t) u] = 1;
                    ++appended_units;
                }
                trimmed_units += planned_taken - admitted.taken;
                region.recalled_tokens += (int) tokens.size();
                if (!tokens.empty()) excerpts_trail = trail;
                for (auto piece : pieces) {
                    piece.begin += (int) excerpts.size();
                    piece.end += (int) excerpts.size();
                    excerpt_pieces.push_back(piece);
                }
                excerpts.insert(excerpts.end(), tokens.begin(), tokens.end());
                recall_stats.push_back({{"region", region_index}, {"units", appended_units},
                                        {"tokens", tokens.size()}, {"cap", region_cap},
                                        {"wrapper", ri == largest ? wrapper : 0},
                                        {"cumulative_tokens", region.recalled_tokens}});
            }
            if (gate_stats.is_object() && gate_stats.contains("deficit_planned")) {
                gate_stats["deficit_after"] = final_deficit / gate_context;
                gate_stats["trimmed_units"] = trimmed_units;
            }
            if (excerpts.empty()) {
                count("recall.empty");
            } else {
                recall_fired = true;
                count("recall.fired");
                const auto open_tokens = tokenizer_.encode(open);
                block.tokens = open_tokens;
                block.pieces.push_back({0, (int) open_tokens.size(), "recall_wrapper", -1, -1});
                for (auto piece : excerpt_pieces) {
                    piece.begin += (int) open_tokens.size();
                    piece.end += (int) open_tokens.size();
                    block.pieces.push_back(piece);
                }
                block.tokens.insert(block.tokens.end(), excerpts.begin(), excerpts.end());
                const int close_at = (int) block.tokens.size();
                const auto close_tokens = tokenizer_.encode(close);
                block.tokens.insert(block.tokens.end(), close_tokens.begin(), close_tokens.end());
                block.pieces.push_back({close_at, (int) block.tokens.size(), "recall_wrapper", -1, -1});
                splices.insert(splices.begin(), std::move(block));
            }
        }

        // The served prompt = the view's served prefix, then the raw prompt from where the view ended with
        // every splice applied (raw target slices between them).
        std::stable_sort(splices.begin(), splices.end(),
                         [] (const Splice & a, const Splice & b) { return a.begin < b.begin; });
        std::vector<int32_t> served;
        std::vector<pu::Piece> pieces;
        const int turn_index = (continues ? view.turns : 0) + 1;
        int position = 0;
        if (continues) {
            served.assign(view.view_tokens.begin(), view.view_tokens.begin() + view.view_gen_begin);
            for (auto piece : view.pieces) {
                if (piece.begin >= view.view_gen_begin) continue;
                piece.end = (std::min)(piece.end, view.view_gen_begin);
                pieces.push_back(piece);
            }
            position = view.raw_gen_begin;
        }
        const auto add_raw = [&] (int from, int to, const char * kind) {
            if (to <= from) return;
            const int at = (int) served.size();
            served.insert(served.end(), raw.begin() + from, raw.begin() + to);
            pieces.push_back({at, (int) served.size(), kind, -1, turn_index});
        };
        bool ordered = true;
        for (auto & splice : splices) {
            if (splice.begin < position || splice.end > raw_gen_begin) {
                ordered = false;
                break;
            }
            add_raw(position, splice.begin, "raw");
            const int at = (int) served.size();
            served.insert(served.end(), splice.tokens.begin(), splice.tokens.end());
            for (auto piece : splice.pieces) {
                piece.begin += at;
                piece.end += at;
                piece.turn = turn_index;
                pieces.push_back(piece);
            }
            position = splice.end;
        }
        if (!ordered) return serve_raw("splice_order");
        add_raw(position, raw_gen_begin, "raw");
        add_raw(raw_gen_begin, prompt_tokens, "generation");

        if (continues && config_.max_ctx > 0 && (int) served.size() + req.max_output > config_.max_ctx) {
            // A view that cannot continue is reset to admission on the raw prompt.
            count("reset.context");
            stats["reset"] = "context";
            stats.erase("recall_gate");
            stats.erase("recall_units");
            continue;
        }
        // A turn served raw instead of continuing a stored view leaves a raw view (no regions), so its
        // descendants continue what it served rather than the view it replaced. Not when the raw prompt does
        // not fit either: a retry with a smaller output allowance may still continue that view.
        const bool raw_fits = config_.max_ctx <= 0 || prompt_tokens + req.max_output <= config_.max_ctx;
        if (!continues && new_regions.empty() && !(found_view && raw_fits)) {
            stats["admission"] = admission;
            return serve_raw(region_candidates.empty() ? "no_region" : "no_region_paged");
        }

        pu::View next;
        next.lineage = continues ? view.lineage : ++pflash_unified_lineage_;
        next.version = continues ? view.version + 1 : 1;
        next.config = pflash_unified_config_;
        next.raw_tokens = raw;
        next.raw_gen_begin = raw_gen_begin;
        next.drafter_ids = drafter_ids;
        next.drafter_gen_begin = turn.generation_begin;
        next.view_tokens = served;
        next.view_gen_begin = (int) (served.size() - generation.size());
        next.regions = std::move(held_regions);
        for (auto & region : new_regions) next.regions.push_back(std::move(region));
        next.pieces = pieces;
        next.turns = turn_index;

        const char * decision = !new_regions.empty() ? (recall_fired ? "region_recall" : "region")
                              : recall_fired ? "recall" : continues ? "continue" : "raw";
        if (std::strcmp(decision, "raw") == 0) {
            count("raw.reset");
            stats["reason"] = "reset";
        } else {
            count(std::string("serve.") + decision);
        }
        stats["decision"] = decision;
        stats["turn"] = turn_index;
        stats["lineage"] = next.lineage;
        stats["continued"] = continues;
        stats["reused_tokens"] = continues ? view.view_gen_begin : 0;
        stats["served_tokens"] = served.size();
        stats["over_raw"] = (int) served.size() - prompt_tokens;
        stats["admission"] = admission;
        stats["regions"] = next.regions.size();
        stats["reread_held_units"] = covered_units;
        if (!recall_stats.is_null()) stats["recall"] = recall_stats;
        if (!gate_stats.is_null()) stats["recall_gate"] = gate_stats;
        if (recall && !scored) stats["recall"] = "unscored";
        json piece_list = json::array();
        for (const auto & piece : pieces) {
            piece_list.push_back({piece.begin, piece.end, piece.kind, piece.region, piece.turn});
        }
        stats["pieces"] = std::move(piece_list);

        prepared.tokens = served;
        prepared.compressed = served != raw;
        prepared.unified_view = true;
        prepared.snapshot_cut = next.view_gen_begin;
        prepared.pflash_stats = {
            {"unified", stats},
            {"compress_ms", std::round(compress_ms * 10.0) / 10.0},
            {"drafter_input_tokens", input},
            {"compressed_tokens", served.size()},
            {"drafter_fingerprint_fnv1a64", http_detail::pflash_token_fingerprint(drafter_ids)},
        };
        if (scored || !result.candidate_lifts.empty() || result.ok) {
            prepared.pflash_stats["scorer_resume"] = result.scorer_resume;
            prepared.pflash_stats["scorer_new_tokens"] = result.scorer_new_tokens;
            prepared.pflash_stats["scorer_forward_ms"] = std::round(result.scorer_forward_s * 10000.0) / 10.0;
            if (result.spread_valid) prepared.pflash_stats["spread"] = {{"spread", result.spread},
                                                                         {"spread_k", result.spread_k}};
            if (result.budget_widened) prepared.pflash_stats["stage2"] = {{"widened", true}};
        }
        std::fprintf(stderr,
            "[pflash-unified] %s turn=%d served=%zu raw=%d reused=%d regions=+%zu/%zu recall=%s%s%s\n",
            decision, turn_index, served.size(), prompt_tokens, continues ? view.view_gen_begin : 0,
            new_regions.size(), next.regions.size(),
            recall_fired ? "fired" : (recall_stats.is_array() ? "empty" : (recall ? "unscored" : "no")),
            disk_cache_.disabled() ? "" : " cached_source=",
            disk_cache_.disabled() ? "" : restorable_prefix.source);
        std::fflush(stderr);
        // A view the RAM store evicts goes to disk (with the disk session cache on), as an evicted KV snapshot does.
        for (const auto & evicted : pflash_unified_views_.remember(std::move(next))) {
            if (pflash_unified_disk_views_.enabled()) pflash_unified_disk_views_.save(evicted);
        }
        if (prepared.compressed) trace_pflash_served(req, prepared);
        return {};
    }
    return serve_raw("reset_failed");
}

HttpServer::PreparedPrompt HttpServer::prepare_prompt(
        const ParsedRequest & req) {
    PreparedPrompt prepared;
    prepared.tokens = req.prompt_tokens;
    if (req.images) {
        if (!config_.image_input_enabled || !req.images->matches(prepared.tokens)) {
            prepared.error_status = 400;
            prepared.error = "image request binding or serving mode is invalid";
        } else {
            prepared.images = req.images;
        }
        return prepared;
    }

    if (config_.pflash_mode != ServerConfig::PflashMode::OFF &&
        drafter_tokenizer_ != nullptr) {
        const int prompt_tokens = (int) req.prompt_tokens.size();
        bool should_compress =
            config_.pflash_mode == ServerConfig::PflashMode::ALWAYS ||
            (config_.pflash_mode == ServerConfig::PflashMode::AUTO &&
             prompt_tokens >= config_.pflash_threshold);
        const bool continuation = should_compress &&
            is_continuation_request(req.messages);
        const bool selection_environment =
            luce::pflash::has_pflash_selection_environment();
        // With a strict-selection environment PFlash owns every turn —
        // multi-turn chit-chat compresses the whole rendered history plus
        // the current user turn instead of routing to FlowKV. Only the
        // per-request FlowKV disk-compression mode still conflicts.
        const bool selection_owns_compression =
            should_compress && selection_environment;
        if (selection_owns_compression) {
            luce::pflash::PFlashSelectionConfig experiment;
            std::string experiment_error;
            if (!luce::pflash::resolve_pflash_selection(
                    0, 32, experiment, experiment_error)) {
                prepared.error_status = 500;
                prepared.error = "invalid PFlash strict selection config: " +
                    experiment_error;
                return prepared;
            }
            if (req.disk_cache_policy.compress) {
                prepared.error_status = 500;
                prepared.error =
                    "PFlash strict selection does not support FlowKV compression";
                return prepared;
            }
        }

        if (should_compress && continuation && req.messages.is_array() &&
            !selection_owns_compression) {
            // FlowKV owns continuation compression automatically. Falling
            // back to whole-prompt compression would destroy the reusable
            // system/tool prefix anchor, and requiring a separate disk-cache
            // flag made --prefill-compression auto silently do nothing.
            apply_flowkv_compression(req, prepared);
            should_compress = false;
        } else if (should_compress && continuation &&
                   !selection_owns_compression) {
            should_compress = false;
            std::fprintf(stderr,
                "[pflash] skip-compress (continuation without messages array)\n");
        }

        if (should_compress && req.disk_cache_policy.compress) {
            // Turn one stays verbatim so the next turn can reuse its KV prefix.
            should_compress = false;
            std::fprintf(stderr,
                "[flowkv] turn-1 verbatim (system kept as cache anchor)\n");
        }

        if (should_compress) {
            PreparedPrompt uncompressed = prepared;
            prepared.error = apply_pflash_compression(req, prepared);
            if (!prepared.error.empty()) {
                // Failure policy D5: serve the raw prompt when it is admissible,
                // and record the fallback; PFLASH_FAIL_CLOSED=1 keeps the error.
                // A relevance-spread fallback is a policy decision, not a
                // failure: PFLASH_FAIL_CLOSED does not refuse it.
                const bool spread_fallback = !prepared.pflash_spread_fallback.is_null();
                const bool fail_closed = http_detail::pflash_fail_closed_from_env();
                const auto decision = http_detail::pflash_fallback_decision(
                    prepared.error, fail_closed && !spread_fallback, prompt_tokens,
                    req.max_output, config_.max_ctx);
                const std::string reason = http_detail::pflash_failure_reason(
                    prepared.error, prepared.pflash_failure_detail);
                {
                    std::lock_guard<std::mutex> lk(pflash_failure_mu_);
                    if (decision.fallback) {
                        ++pflash_fallback_served_;
                        ++pflash_fallback_reasons_[reason];
                    } else {
                        ++pflash_fallback_refused_;
                        ++pflash_refused_reasons_[decision.refused];
                    }
                }
                if (!decision.fallback) {
                    if (!fail_closed) {
                        std::fprintf(stderr,
                            "[pflash-fallback] refused=%s reason=%s prompt_tokens=%d "
                            "max_ctx=%d error=%s\n",
                            decision.refused.c_str(), reason.c_str(), prompt_tokens,
                            config_.max_ctx, prepared.error.c_str());
                        std::fflush(stderr);
                    }
                    prepared.error_status = 500;
                    return prepared;
                }
                std::fprintf(stderr,
                    "[pflash-fallback] serving the raw prompt (%d tokens) reason=%s "
                    "error=%s detail=%s\n",
                    prompt_tokens, reason.c_str(), prepared.error.c_str(),
                    prepared.pflash_failure_detail.c_str());
                std::fflush(stderr);
                json record = {
                    {"reason", reason},
                    {"error", prepared.error},
                    {"prompt_tokens", prompt_tokens},
                };
                if (!prepared.pflash_failure_detail.empty()) {
                    record["detail"] = prepared.pflash_failure_detail;
                }
                if (spread_fallback) {
                    for (const auto & [key, value] : prepared.pflash_spread_fallback.items()) {
                        record[key] = value;
                    }
                }
                // A raw fallback replaces the conversation's served prefix.
                // Keeping its previous compressed view would bypass raw-cache
                // admission and revive dropped history on the next turn.
                const size_t forgotten = pflash_views_.forget(req.prompt_tokens);
                if (forgotten > 0) record["views_forgotten"] = forgotten;
                const size_t unified_forgotten = forget_unified_views(req.prompt_tokens);
                if (unified_forgotten > 0) record["unified_lineages_forgotten"] = unified_forgotten;
                const std::string unified_handback = prepared.unified_handback;
                prepared = std::move(uncompressed);
                prepared.unified_handback = unified_handback;
                prepared.pflash_stats = {{"pflash_fallback", std::move(record)}};
            }
        }
    }

    // PFLASH_VIEW_TRACE_PATH: every request leaves one record, the
    // uncompressed ones too (kind "raw"), so a reader never has to infer
    // what was served from the absence of a line.
    if (!prepared.compressed && prepared.error.empty()) trace_pflash_served(req, prepared);

    const int prompt_length = prepared.full_cache_served_tokens >= 0
        ? prepared.full_cache_served_tokens
        : (int) prepared.tokens.size();
    if (effective_prompt_overflows(
            (int) prepared.tokens.size(), prepared.full_cache_served_tokens,
            req.max_output, config_.max_ctx)) {
        prepared.error_status = 400;
        prepared.error = context_overflow_message(
            config_.max_ctx, prompt_length, req.max_output);
    }
    // A request the chat view serves after a hand-back ends the conversation's unified views; one refused keeps
    // them, so a retry can still continue its view.
    if (prepared.error.empty() && !prepared.unified_handback.empty()) forget_unified_views(req.prompt_tokens);
    return prepared;
}

bool HttpServer::forward_upstream(
        ServerJob * job, const ParsedRequest & req,
        const PreparedPrompt & prepared) {
    if (req.images) return false;
#ifdef LUCE_HAS_CURL
    if (config_.pflash_upstream_base.empty()) return false;

    const std::string & upstream = config_.pflash_upstream_base;
    const std::string & upstream_key = config_.pflash_upstream_key;
    const std::string & upstream_model = config_.pflash_upstream_model.empty()
        ? req.model : config_.pflash_upstream_model;

    if (prepared.compressed) {
        std::string compressed_text = tokenizer_.decode(prepared.tokens);
        compressed_text += "\n<|im_start|>assistant\n";

        json body;
        body["model"] = upstream_model;
        body["prompt"] = compressed_text;
        body["stream"] = req.stream;
        if (req.raw_body.contains("max_tokens")) {
            body["max_tokens"] = req.raw_body["max_tokens"];
        } else {
            body["max_tokens"] = req.max_output;
        }
        for (const char * key : {
                 "temperature", "top_p", "top_k", "min_p",
                 "frequency_penalty", "presence_penalty", "stop", "seed"}) {
            if (req.raw_body.contains(key)) body[key] = req.raw_body[key];
        }

        std::fprintf(stderr,
            "[pflash-proxy] compressed forward → %s/completions  "
            "prompt=%zu tokens  model=%s\n",
            upstream.c_str(), prepared.tokens.size(), upstream_model.c_str());
        curl_forward(
            upstream + "/completions", upstream_key, body,
            req.stream, /*rewrite_to_chat=*/true,
            req.response_id, upstream_model,
            [this, job](const void * data, size_t size) {
                return send_job_bytes(job, data, size);
            },
            [this, job]() { stop_job_stream(job); },
            [job]() {
                return job->client_disconnected.load(std::memory_order_acquire);
            });
    } else {
        json body = req.raw_body;
        body["model"] = upstream_model;

        std::fprintf(stderr,
            "[pflash-proxy] passthrough → %s/chat/completions  model=%s\n",
            upstream.c_str(), upstream_model.c_str());
        curl_forward(
            upstream + "/chat/completions", upstream_key, body,
            req.stream, /*rewrite_to_chat=*/false,
            req.response_id, upstream_model,
            [this, job](const void * data, size_t size) {
                return send_job_bytes(job, data, size);
            },
            [this, job]() { stop_job_stream(job); },
            [job]() {
                return job->client_disconnected.load(std::memory_order_acquire);
            });
    }
    return true;
#else
    (void) job;
    (void) req;
    (void) prepared;
    return false;
#endif
}

std::vector<int> HttpServer::request_restore_points(const std::vector<int32_t> & prompt,
                                                    std::initializer_list<int> cuts) const {
    return prefix_restore_points(prompt, prefix_cache_.chat_markers(), cuts,
        /*drop_last_boundary=*/!backend_.prefill_cuts_at_restore_points(),
        backend_.restore_point_spacing());
}

// Cache lookup and snapshot preparation form one lifecycle; confirmation is
// deferred until generation proves that the snapshot contains useful output.
HttpServer::GenerationCacheState HttpServer::prepare_generation_cache(
        const ParsedRequest & req, PreparedPrompt & prepared,
        GenerateRequest & generate_request) {
    request_session_slots_.clear();
    if (req.images) return {};
    auto & effective_prompt = prepared.tokens;
    // Tool-heavy requests prefer the reusable system/tool boundary under eviction.
    const bool prefer_inline_snap = !req.tools.empty();
    const bool prefer_tools_boundary =
        ppp_prefers_tools_boundary(config_.ppp_enabled, prefer_inline_snap);
    int forced_cut = req.pin_end_token;
    if (forced_cut <= 0) forced_cut = prepared.snapshot_cut;

    // PPP runs *before* lookup. Default (rearrange=0): annotate a sticky
    // pin_end only — never mutate tokens. Token-level DiffPin rewrite
    // (prefix|suffix|middle float) is opt-in via LUCE_PPP_REARRANGE=1;
    // unconstrained middle peels can scramble tool-schema JSON and yield
    // empty post-tool completions.
    bool ppp_rewrote = false;
    if (config_.ppp_enabled && prefer_tools_boundary) {
        const auto boundaries = find_all_boundaries(
            effective_prompt, prefix_cache_.chat_markers());
        if (config_.ppp_rearrange) {
            auto rewrite = PinFriendlyPrompt::diff_make_pin_friendly(
                effective_prompt, boundaries, recent_tool_prefixes_,
                prefix_cache_.chat_markers(),
                config_.ppp_lcp_window, config_.ppp_min_pin_tokens,
                config_.ppp_max_ephemeral_tokens);
            if (rewrite.rewritten) {
                effective_prompt = std::move(rewrite.tokens);
                generate_request.prompt = effective_prompt;
                ppp_rewrote = true;
                // Full-cache hits/keys from unrearranged tokens are stale.
                prepared.full_cache_hit_slot = -1;
                prepared.full_cache_hit_len = 0;
                prepared.full_cache_served_tokens = -1;
                std::fprintf(stderr,
                    "[ppp] diff-rewrite prefix=%d suffix=%d middle=%d "
                    "pin_end=%d prompt=%zu\n",
                    rewrite.prefix_len, rewrite.suffix_len, rewrite.middle_len,
                    rewrite.pin_end, effective_prompt.size());
            }
            if (forced_cut <= 0) forced_cut = rewrite.pin_end;
            if (forced_cut > 0 && !rewrite.rewritten) {
                std::fprintf(stderr,
                    "[ppp] pin_end=%d (no rewrite; prompt=%zu)\n",
                    forced_cut, effective_prompt.size());
            }
        } else if (forced_cut <= 0) {
            forced_cut = PinFriendlyPrompt::annotate_pin_end(
                effective_prompt, boundaries, recent_tool_prefixes_,
                config_.ppp_lcp_window, config_.ppp_min_pin_tokens);
            if (forced_cut > 0) {
                std::fprintf(stderr,
                    "[ppp] pin_end=%d (pin-only; prompt=%zu)\n",
                    forced_cut, effective_prompt.size());
            }
        }
        const auto & remember_ids = ppp_tool_prefix_source(
            prepared.compressed, req.prompt_tokens, effective_prompt);
        const auto remember_bounds = find_all_boundaries(
            remember_ids, prefix_cache_.chat_markers());
        const int remember_n = !remember_bounds.empty()
            ? remember_bounds.front()
            : (int)remember_ids.size();
        PinFriendlyPrompt::remember_tool_prefix(
            recent_tool_prefixes_, remember_ids, remember_n,
            config_.ppp_lcp_window);
    }

    GenerationCacheState cache;
    cache.cache_slot = prepared.full_cache_hit_slot;
    cache.prefix_len = prepared.full_cache_hit_len;
    cache.using_restore = cache.cache_slot >= 0;
    cache.disk_policy = req.disk_cache_policy;
    cache.full_snap_key_effective = ppp_rewrote;

    // Exact full-prompt snapshots. After a DiffPin rewrite, key by the
    // tokens we actually serve (effective_prompt), not the client wire form.
    if (!cache.using_restore) {
        const auto & full_key =
            ppp_rewrote ? effective_prompt : req.prompt_tokens;
        auto [full_slot, full_len] = prefix_cache_.lookup_full(full_key);
        if (full_slot >= 0) {
            cache.cache_slot = full_slot;
            cache.prefix_len = full_len;
            cache.using_restore = true;
            if (prepared.compressed) {
                effective_prompt.assign((size_t) full_len, 0);
                generate_request.prompt = effective_prompt;
            }
        }
    }
    // An entry whose backend snapshot is gone is dropped and the next
    // deepest one tried, instead of prefilling the whole prompt.
    while (!cache.using_restore) {
        const auto [inline_slot, inline_len] = prefix_cache_.lookup_candidate(
            effective_prompt, (int) effective_prompt.size());
        if (inline_slot < 0) break;
        if (!backend_.snapshot_used(inline_slot)) {
            forget_inline_slot_metadata(inline_slot);
            prefix_cache_.invalidate_inline_snap(inline_slot);
            continue;
        }
        prefix_cache_.record_inline_hit(
            inline_slot, inline_len, effective_prompt.size());
        cache.cache_slot = inline_slot;
        cache.prefix_len = inline_len;
        cache.using_restore = true;
    }

    // FlowKV may rewrite aged messages, so only its stable system prefix is
    // safe for scoped disk reuse. Whole-prompt PFlash requires Full scope.
    int system_end = 0;
    if (http_detail::should_clamp_flowkv_disk_cache(
            prepared.flowkv, req.disk_cache_policy)) {
        const auto boundaries = find_all_boundaries(
            effective_prompt, prefix_cache_.chat_markers());
        system_end = boundaries.empty() ? 0 : boundaries[0];
        if (system_end >= config_.disk_cache_min_tokens) {
            cache.disk_policy.mode = DiskPrefixCacheMode::Fixed;
            cache.disk_policy.fixed_tokens = system_end;
            std::fprintf(stderr,
                "[flowkv] disk-clamp: boundary clamped to system_end=%d\n",
                system_end);
        } else {
            cache.disk_policy.mode = DiskPrefixCacheMode::Off;
            std::fprintf(stderr,
                "[flowkv] disk-clamp: system_end=%d < min=%d — disk off\n",
                system_end, config_.disk_cache_min_tokens);
        }
    } else if (prepared.compressed && !prepared.flowkv &&
               cache.disk_policy.mode != DiskPrefixCacheMode::Full) {
        cache.disk_policy.mode = DiskPrefixCacheMode::Off;
    }

    std::vector<int> safe_boundaries;
    if (cache.disk_policy.mode == DiskPrefixCacheMode::Auto) {
        safe_boundaries = find_all_boundaries(
            effective_prompt, prefix_cache_.chat_markers());
    }

    int selected_boundary = 0;
    if (cache.disk_policy.mode == DiskPrefixCacheMode::Fixed) {
        selected_boundary = disk_prefix_cache_fixed_boundary(
            cache.disk_policy, (int) effective_prompt.size(),
            config_.disk_cache_min_tokens);
    } else if (cache.disk_policy.mode == DiskPrefixCacheMode::Auto) {
        selected_boundary = disk_prefix_cache_auto_boundary(
            effective_prompt, recent_disk_prompts_,
            cache.disk_policy.auto_window, safe_boundaries,
            config_.disk_cache_min_tokens);
        std::fprintf(stderr,
            "[disk-cache] auto scope: window=%d recent=%zu safe=%zu selected=%d\n",
            cache.disk_policy.auto_window,
            std::min(recent_disk_prompts_.size(),
                     (size_t) cache.disk_policy.auto_window),
            safe_boundaries.size(), selected_boundary);
    }

    std::vector<int> lookup_lengths;
    DiskPrefixCache::PrefixProbe disk_probe;
    if (cache.disk_policy.mode == DiskPrefixCacheMode::Full &&
        !effective_prompt.empty()) {
        // Probes are index lookups; only a hit reads a file. A restart then
        // recovers any inline snapshot the previous process persisted, not
        // just exact prompts and the short cold prefix.
        lookup_lengths = disk_prefix_cache_full_lookup_lengths(
            (int) effective_prompt.size(),
            find_all_boundaries(effective_prompt, prefix_cache_.chat_markers()),
            config_.disk_cache_min_tokens);
        // Session states (agent turns with their generated text) are keyed
        // past the prompt's chat boundaries; the index knows their lengths.
        const auto probe = disk_cache_.longest_prefix(effective_prompt);
        disk_probe = probe;
        if (probe.len > 0 &&
            std::find(lookup_lengths.begin(), lookup_lengths.end(), probe.len) ==
                lookup_lengths.end()) {
            lookup_lengths.insert(lookup_lengths.begin(), probe.len);
            std::sort(lookup_lengths.begin(), lookup_lengths.end(),
                      std::greater<int>());
        }
    } else if (selected_boundary > 0) {
        lookup_lengths.push_back(selected_boundary);
    } else if (cache.disk_policy.mode == DiskPrefixCacheMode::Auto) {
        for (auto it = safe_boundaries.rbegin();
             it != safe_boundaries.rend(); ++it) {
            if (*it >= config_.disk_cache_min_tokens &&
                *it <= (int) effective_prompt.size()) {
                lookup_lengths.push_back(*it);
            }
        }
    }

    // A RAM hit does not end the search: a session whose deep snapshot was
    // evicted to disk often still matches a shallow resident entry (the
    // shared system/tools head). The deeper disk state wins; the index
    // answers without a read, and a failed read keeps the RAM hit.
    bool disk_lookup = !cache.using_restore;
    const int ram_prefix_len = cache.using_restore ? cache.prefix_len : 0;
    if (cache.using_restore && !disk_cache_.disabled() &&
        (!prepared.compressed || prepared.unified_view) && disk_probe.len > 0 &&
        disk_probe.cur_pos > cache.prefix_len &&
        disk_probe.len < (int) effective_prompt.size()) {
        lookup_lengths.assign(1, disk_probe.len);
        disk_lookup = true;
    }
    if (disk_lookup && !disk_cache_.disabled()) {
        for (int lookup_length : lookup_lengths) {
            std::vector<int32_t> prefix_tokens(
                effective_prompt.begin(),
                effective_prompt.begin() + lookup_length);
            if (!disk_cache_.lookup(prefix_tokens, kDiskStagingSlot)) continue;

            const int hit_pos = backend_.snapshot_cur_pos(kDiskStagingSlot);
            if (hit_pos <= 0 || hit_pos > (int) effective_prompt.size()) {
                std::fprintf(stderr,
                    "[disk-cache] ignoring invalid hit pos=%d prompt=%zu\n",
                    hit_pos, effective_prompt.size());
                backend_.snapshot_free(kDiskStagingSlot);
                continue;
            }
            cache.cache_slot = kDiskStagingSlot;
            cache.prefix_len = hit_pos;
            cache.using_restore = true;
            cache.disk_hit = true;
            std::fprintf(stderr,
                "[disk-cache] hit policy=%s len=%d slot=%d pos=%d (RAM had %d)\n",
                disk_prefix_cache_policy_name(cache.disk_policy).c_str(),
                lookup_length, kDiskStagingSlot, cache.prefix_len,
                ram_prefix_len);
            break;
        }
    }

    // Scoped entries must prefill exactly to their hashed boundary before
    // the staging snapshot can be used for the remaining prompt.
    if (!cache.using_restore && !disk_cache_.disabled() &&
        selected_boundary > 0) {
        std::fprintf(stderr,
            "[disk-cache] scoped prefix: policy=%s boundary=%d\n",
            disk_prefix_cache_policy_name(cache.disk_policy).c_str(),
            selected_boundary);
        GenerateRequest scoped_request;
        scoped_request.prompt = std::vector<int32_t>(
            effective_prompt.begin(),
            effective_prompt.begin() + selected_boundary);
        scoped_request.n_gen = 0;
        scoped_request.snap_slot = kDiskStagingSlot;
        scoped_request.snap_pos = selected_boundary;
        scoped_request.restore_points = prefix_restore_points(
            scoped_request.prompt, prefix_cache_.chat_markers(), {forced_cut},
            false, backend_.restore_point_spacing());
        DaemonIO scoped_io;
        scoped_io.stream_fd = -1;
        const auto scoped_result =
            backend_.generate(scoped_request, scoped_io);
        if (scoped_result.ok() &&
            backend_.snapshot_used(kDiskStagingSlot)) {
            disk_cache_.learn_layout(kDiskStagingSlot);
            const bool saved =
                disk_cache_.save(kDiskStagingSlot, scoped_request.prompt);
            cache.cache_slot = kDiskStagingSlot;
            cache.prefix_len = selected_boundary;
            cache.using_restore = true;
            cache.disk_hit = true;
            std::fprintf(stderr,
                "[disk-cache] scoped prefix %s, restoring from %d\n",
                saved ? "saved" : "staged", selected_boundary);
        } else {
            backend_.snapshot_free(kDiskStagingSlot);
        }
    }

    // Edited or summarized histories can be shorter than the stored KV.
    // Such snapshots cannot be diff-prefilled safely.
    // The logical entry length (what the PrefixCache believes is cached, e.g.
    // the tools-head pin at 3620) can exceed the chunk-aligned KV position the
    // backend actually restored (e.g. 3584). Deepen decisions must compare
    // against the logical length; otherwise that chunk gap keeps
    // `forced_cut > restored` true on every tool-heavy turn, the pin branch
    // keeps targeting the already-cached head, and the cache never deepens
    // past the head pin (full conversation re-prefilled each turn).
    const int pre_overwrite_prefix_len = cache.prefix_len;
    int logical_prefix_len =
        cache.using_restore ? pre_overwrite_prefix_len : 0;
    if (cache.using_restore) {
        const int snapshot_length =
            backend_.snapshot_cur_pos(cache.cache_slot);
        if (snapshot_length <= 0 ||
            snapshot_length > (int) effective_prompt.size()) {
            std::fprintf(stderr,
                "[pc] slot=%d invalid snapshot pos=%d prompt=%zu — treating as miss\n",
                cache.cache_slot, snapshot_length, effective_prompt.size());
            if (cache.disk_hit) {
                backend_.snapshot_free(kDiskStagingSlot);
            } else {
                // A memory-cache key whose backend snapshot disappeared
                // must not remain discoverable on every later request.
                // Inline and full caches use disjoint slot ranges, so
                // invalidating both ownership tables is unambiguous.
                forget_inline_slot_metadata(cache.cache_slot);
                backend_.snapshot_free(cache.cache_slot);
                prefix_cache_.invalidate_inline_snap(cache.cache_slot);
                prefix_cache_.abort_full_snap(cache.cache_slot);
            }
            cache.cache_slot = -1;
            cache.prefix_len = 0;
            cache.using_restore = false;
            cache.disk_hit = false;
        } else {
            // PrefixCache keys use safe chat boundaries. Qwen may save at an
            // earlier chunk-aligned position to keep prefill numerically
            // identical, so backend state is the authoritative amount of
            // work actually reused.
            cache.prefix_len = snapshot_length;
        }
    }

    // A compressed prompt is not raw history: never persist a prefix of it.
    // With RAM prefix slots the inline snapshot holds the head and reaches
    // disk when it is evicted, so this extra prefill-and-write is only the
    // fallback without them.
    if (!cache.using_restore && !disk_cache_.disabled() &&
        prefix_cache_.disabled() && !prepared.compressed &&
        cache.disk_policy.mode == DiskPrefixCacheMode::Full) {
        const auto boundaries = find_all_boundaries(
            effective_prompt, prefix_cache_.chat_markers());
        const int cold_boundary =
            disk_cache_.cold_prefix_boundary(effective_prompt, boundaries);
        if (cold_boundary > 0) {
            std::fprintf(stderr,
                "[disk-cache] cold prefix: prefilling to boundary=%d\n",
                cold_boundary);
            GenerateRequest cold_request;
            cold_request.prompt = std::vector<int32_t>(
                effective_prompt.begin(),
                effective_prompt.begin() + cold_boundary);
            cold_request.n_gen = 0;
            cold_request.snap_slot = kDiskStagingSlot;
            cold_request.snap_pos = cold_boundary;
            cold_request.restore_points = prefix_restore_points(
                cold_request.prompt, prefix_cache_.chat_markers(), {forced_cut},
                false, backend_.restore_point_spacing());
            DaemonIO cold_io;
            cold_io.stream_fd = -1;
            const auto cold_result = backend_.generate(cold_request, cold_io);
            if (cold_result.ok() &&
                backend_.snapshot_used(kDiskStagingSlot)) {
                disk_cache_.learn_layout(kDiskStagingSlot);
                const std::vector<int32_t> prefix_tokens(
                    effective_prompt.begin(),
                    effective_prompt.begin() + cold_boundary);
                disk_cache_.save(kDiskStagingSlot, prefix_tokens);
                cache.cache_slot = kDiskStagingSlot;
                cache.prefix_len = cold_boundary;
                cache.using_restore = true;
                cache.disk_hit = true;
                // This restore was created after the pre-overwrite capture.
                logical_prefix_len = cold_boundary;
                std::fprintf(stderr,
                    "[disk-cache] cold prefix saved, restoring from %d\n",
                    cold_boundary);
            } else {
                backend_.snapshot_free(kDiskStagingSlot);
            }
        }
    }

    // A generation can save only one snapshot during prefill. Tool-heavy
    // requests prefer the reusable system/tool boundary; otherwise an
    // enabled exact full-prompt cache retains its existing priority.
    auto prepare_inline = [&]() {
        // Never let the new snapshot land in the slot this request restores
        // from: the guard below would cancel it, pinning the restore point at
        // the deepest slot on linearly-growing conversations.
        const int restore_source_slot =
            cache.using_restore ? cache.cache_slot : -1;
        // Prefill resumes at the backend's restored position, and a save
        // lands at least one snapshot step past it.
        const int reachable_from =
            (cache.using_restore ? cache.prefix_len : 0) +
            backend_.snapshot_granularity();
        cache.snap_reservation = prefix_cache_.reserve_inline_snap(
            effective_prompt,
            cache.using_restore ? logical_prefix_len : 0,
            prefer_tools_boundary,
            forced_cut,
            restore_source_slot,
            [this](int target_cut) {
                return backend_.snapshot_bytes_estimate(target_cut);
            },
            reachable_from, backend_.restore_point_spacing());
        cache.snap_slot = cache.snap_reservation.slot();
        cache.snap_cut = cache.snap_reservation.target_cut();
    };
    auto prepare_full = [&]() {
        const auto & full_key = cache.full_snap_key_effective
            ? effective_prompt : req.prompt_tokens;
        cache.full_snap_slot = prefix_cache_.prepare_full_snap(full_key);
        if (cache.full_snap_slot >= 0) {
            cache.full_snap_pos = (int) effective_prompt.size();
            generate_request.snap_slot = cache.full_snap_slot;
            generate_request.snap_pos = cache.full_snap_pos;
            cache.full_snap_prepared = true;
        }
    };

    // An agent turn that continues its generated-turn checkpoint is itself
    // continued the same way by the next request, so its own snapshot only
    // backs an interrupted or edited turn. Take one when the deepest other
    // checkpoint lies kAgentFallbackStride tokens behind; each copy costs
    // the whole prefix.
    bool skip_inline = false;
    if (cache.using_restore && !cache.disk_hit && req.ends_with_tool_result &&
        agent_turn_cache_slots_.count(cache.cache_slot)) {
        constexpr int kAgentFallbackStride = 2048;
        const auto boundaries = find_all_boundaries(
            effective_prompt, prefix_cache_.chat_markers());
        const int cut = boundaries.empty() ? 0 : boundaries.back();
        const int fallback = prefix_cache_.lookup_candidate(
            effective_prompt, logical_prefix_len - 1).second;
        skip_inline = cut - fallback < kAgentFallbackStride;
    }
    if (!skip_inline && (prefer_inline_snap || cache.using_restore)) {
        prepare_inline();
    }
    if (!cache.using_restore && cache.snap_slot < 0) {
        prepare_full();
    }

    // Full cache may be disabled or already contain this exact key. Fall
    // back to an inline snapshot when no target has been selected yet.
    if (!skip_inline && !cache.full_snap_prepared && cache.snap_slot < 0) {
        prepare_inline();
    }

    // Never destroy the snapshot needed by this request's restore. With a
    // one-slot cache there may be no independent destination for a deeper
    // checkpoint; preserving the current hit is better than invalidating
    // it before restore starts.
    if (cache.using_restore && cache.snap_slot == cache.cache_slot) {
        cache.snap_reservation.cancel();
        cache.snap_slot = -1;
        cache.snap_cut = 0;
    }
    cache.snap_prepared = cache.snap_slot >= 0;
    if (cache.snap_prepared) {
        evict_inline_slot(cache.snap_slot);
        backend_.snapshot_free(cache.snap_slot);
        generate_request.snap_slot = cache.snap_slot;
        generate_request.snap_pos = cache.snap_cut;
    }
    if (cache.full_snap_prepared) {
        backend_.snapshot_free(cache.full_snap_slot);
    }

    if (cache.using_restore && disk_cache_digest_enabled()) {
        backend_.snapshot_flush_deferred();
        int n_tensors = 0;
        uint64_t n_bytes = 0;
        const uint64_t digest = snapshot_state_digest(
            backend_.snapshot_ref(cache.cache_slot), &n_tensors, &n_bytes);
        std::fprintf(stderr,
            "[disk-cache] digest %s source=%s pos=%d tensors=%d bytes=%llu "
            "xxh64=%016llx\n",
            req.response_id.c_str(), cache.disk_hit ? "ssd" : "ram",
            cache.prefix_len, n_tensors, (unsigned long long)n_bytes,
            (unsigned long long)digest);
    }
    std::fprintf(stderr,
        "[server] chat CACHE %s restore=%s slot=%d prefix_len=%d "
        "effective_prompt=%zu pflash=%s disk_policy=%s disk_hit=%s "
        "snap_slot=%d snap_pos=%d full_snap_slot=%d full_snap_pos=%d\n",
        req.response_id.c_str(),
        cache.using_restore ? "true" : "false",
        cache.cache_slot, cache.prefix_len, effective_prompt.size(),
        prepared.compressed ? "true" : "false",
        disk_prefix_cache_policy_name(cache.disk_policy).c_str(),
        cache.disk_hit ? "true" : "false",
        cache.snap_slot, cache.snap_cut,
        cache.full_snap_slot, cache.full_snap_pos);

    if (!prefix_cache_.disabled() || !disk_cache_.disabled()) {
        generate_request.restore_points = request_restore_points(
            effective_prompt,
            {forced_cut, selected_boundary,
             cache.snap_prepared ? cache.snap_cut : 0});
    }

    status_.set_flags(
        cache.using_restore, prepared.compressed,
        !config_.draft_path.empty());
    broadcast_status();
    return cache;
}

void HttpServer::finalize_generation_cache(
        const ParsedRequest & req, const PreparedPrompt & prepared,
        GenerationCacheState & cache, const GenerateResult & result,
        int completion_tokens, bool visible_output_seen,
        bool client_disconnected) {
    if (req.images) return;
    const auto & effective_prompt = prepared.tokens;
    const bool generation_produced_output = result.ok() &&
        completion_tokens > 0 && visible_output_seen && !client_disconnected;

    // A deferred restore can be consumed, or a deeper snapshot can replace
    // its source under memory pressure. Neither entry may stay discoverable.
    if (cache.using_restore && !cache.disk_hit &&
        !backend_.snapshot_used(cache.cache_slot)) {
        forget_inline_slot_metadata(cache.cache_slot);
        prefix_cache_.invalidate_inline_snap(cache.cache_slot);
    }

    if (cache.full_snap_prepared) {
        if (generation_produced_output &&
            backend_.snapshot_used(cache.full_snap_slot)) {
            const int saved_position =
                backend_.snapshot_cur_pos(cache.full_snap_slot);
            if (saved_position > 0 &&
                saved_position <= cache.full_snap_pos) {
                const auto & full_key = cache.full_snap_key_effective
                    ? effective_prompt : req.prompt_tokens;
                prefix_cache_.confirm_full_snap(
                    cache.full_snap_slot, full_key, saved_position);
            } else {
                backend_.snapshot_free(cache.full_snap_slot);
                prefix_cache_.abort_full_snap(cache.full_snap_slot);
            }
        } else {
            backend_.snapshot_free(cache.full_snap_slot);
            prefix_cache_.abort_full_snap(cache.full_snap_slot);
        }
    }

    if (cache.snap_prepared) {
        if (generation_produced_output &&
            backend_.snapshot_used(cache.snap_slot)) {
            const int saved_position =
                backend_.snapshot_cur_pos(cache.snap_slot);
            if (saved_position > 0 && saved_position <= cache.snap_cut) {
                std::fprintf(stderr,
                    "[pc] inline snapshot requested=%d saved=%d slot=%d\n",
                    cache.snap_cut, saved_position, cache.snap_slot);
                // Key the entry at the requested cut even when Qwen saved at
                // an earlier chunk boundary. Matching the longer prefix still
                // guarantees the saved rows, the restore resumes at the
                // backend's own position, and the next request's deepen
                // decision sees the requested cut as covered. Keying at the
                // saved position re-targets the same cut on every request
                // and the restore point never moves past it.
                evict_extra_victims(cache.snap_reservation);
                cache.snap_reservation.commit(
                    effective_prompt,
                    backend_.snapshot_bytes_estimate(saved_position));
                trim_snapshots_after_commit(cache.snap_slot);
                // The disk key is exact: it must not claim rows the snapshot
                // lacks. A compressed prompt gets no key, except one served
                // from a unified view: that view is kept on disk too, so the
                // conversation's next request serves the same tokens again.
                // The write happens only if this snapshot is evicted from RAM
                // (evict_inline_slot) or at shutdown.
                if ((!prepared.compressed || prepared.unified_view) &&
                    cache.disk_policy.mode == DiskPrefixCacheMode::Full) {
                    slot_tokens_[cache.snap_slot] = std::vector<int32_t>(
                        effective_prompt.begin(),
                        effective_prompt.begin() + saved_position);
                    note_session_slot(cache.snap_slot);
                }
            } else {
                backend_.snapshot_free(cache.snap_slot);
                cache.snap_reservation.abort();
            }
        } else {
            backend_.snapshot_free(cache.snap_slot);
            cache.snap_reservation.abort();
        }
    }

    if (cache.disk_hit) backend_.snapshot_free(kDiskStagingSlot);

    // Long conversations get a continued checkpoint after crossing the
    // configured interval so a future turn can restore prompt plus output.
    // With RAM prefix slots a session's state is written when it is evicted
    // from RAM instead; this checkpoint is the fallback without them.
    if (!disk_cache_.disabled() && prefix_cache_.disabled() &&
        !prepared.compressed &&
        cache.disk_policy.mode == DiskPrefixCacheMode::Full && result.ok() &&
        generation_produced_output) {
        const int final_position =
            (int) effective_prompt.size() + (int) result.tokens.size();
        if (final_position >= disk_cache_.continued_interval()) {
            std::vector<int32_t> all_tokens(effective_prompt);
            all_tokens.insert(
                all_tokens.end(), result.tokens.begin(), result.tokens.end());
            if (backend_.snapshot_save(kDiskStagingSlot)) {
                disk_cache_.learn_layout(kDiskStagingSlot);
                disk_cache_.maybe_store_continued(
                    kDiskStagingSlot, all_tokens, final_position);
                backend_.snapshot_free(kDiskStagingSlot);
            }
        }
    }

    if (disk_cache_.disabled() || !result.ok()) return;

    if (!prepared.compressed) {
        recent_disk_prompts_.insert(
            recent_disk_prompts_.begin(), effective_prompt);
    } else if (prepared.flowkv) {
        // FlowKV history is rewritten; retain the verbatim prompt for future
        // Auto-boundary comparisons.
        recent_disk_prompts_.insert(
            recent_disk_prompts_.begin(), req.prompt_tokens);
    }
    static constexpr size_t kMaxRecentDiskPrompts = 256;
    if (recent_disk_prompts_.size() > kMaxRecentDiskPrompts) {
        recent_disk_prompts_.resize(kMaxRecentDiskPrompts);
    }
}

void HttpServer::forget_inline_slot_metadata(int slot) {
    if (slot < 0) return;
    agent_turn_cache_slots_.erase(slot);
    slot_tokens_.erase(slot);
    slot_aux_.erase(slot);
}

bool HttpServer::persist_evicted_slot(int slot, bool persist, const char * reason) {
    if (slot < 0 || !persist || disk_cache_.disabled()) return false;
    const auto key = slot_tokens_.find(slot);
    // No key: a compressed view, a non-`full` disk scope, or no snapshot.
    if (key == slot_tokens_.end() || !backend_.snapshot_used(slot)) return false;
    const auto aux = slot_aux_.find(slot);
    OwnedSnapshot snap = backend_.snapshot_detach(slot);
    if (snap.empty()) return false;
    const int pos = snap.cur_pos();
    const bool queued = disk_cache_.save_async(
        std::move(snap), key->second,
        aux == slot_aux_.end() ? std::string() : aux->second);
    const int pruned = queued
        ? disk_cache_.prune_ancestors(key->second, config_.disk_cache_cold_max_tokens)
        : 0;
    std::fprintf(stderr,
        "[disk-cache] %s slot=%d tokens=%zu pos=%d -> %s (pruned %d older)\n",
        reason, slot, key->second.size(), pos,
        queued ? "write queued" : "not written", pruned);
    return queued;
}

void HttpServer::evict_inline_slot(int slot) {
    if (slot >= 0) {
        persist_evicted_slot(slot, prefix_cache_.persist_on_evict(slot), "ram eviction");
    }
    forget_inline_slot_metadata(slot);
}

void HttpServer::evict_extra_victims(
        const PrefixCache::InlineReservation & reservation) {
    for (const int slot : reservation.extra_victim_slots()) {
        evict_inline_slot(slot);
        backend_.snapshot_free(slot);
    }
}

void HttpServer::note_session_slot(int slot) {
    if (slot >= 0) request_session_slots_.push_back(slot);
}

std::string HttpServer::session_replay_blob(const ParsedRequest & req,
                                           const SseEmitter * emitter) {
    json entries = json::array();
    if (tool_memory_.disabled()) return std::string();
    const auto messages =
        normalize_chat_messages(req.messages, req.format, tool_memory_, true);
    for (const auto & m : messages) {
        if (!m.tool_calls_replayed || m.content.empty()) continue;
        json ids = json::array();
        for (const auto & call : m.tool_calls) ids.push_back(call.id);
        entries.push_back({{"ids", ids}, {"raw", m.content}});
    }
    // This response's own calls, remembered by remember_agent_turn().
    std::vector<std::string> ids;
    if (emitter) {
        for (const auto & call : emitter->tool_calls()) ids.push_back(call.id);
    }
    if (!ids.empty()) {
        const std::string raw = tool_memory_.lookup(ids);
        if (!raw.empty()) entries.push_back({{"ids", ids}, {"raw", raw}});
    }
    if (entries.empty()) return std::string();
    return json{{"tool_memory", entries}}.dump();
}

void HttpServer::restore_replay_blobs() {
    size_t restored = 0;
    disk_cache_.for_each_aux([&](const std::string & blob) {
        const json doc = json::parse(blob, nullptr, /*allow_exceptions=*/false);
        if (!doc.is_object() || !doc.contains("tool_memory") ||
            !doc["tool_memory"].is_array()) {
            return;
        }
        for (const auto & entry : doc["tool_memory"]) {
            if (!entry.is_object() || !entry.contains("ids") ||
                !entry["ids"].is_array() || !entry.contains("raw") ||
                !entry["raw"].is_string()) {
                continue;
            }
            std::vector<std::string> ids;
            for (const auto & id : entry["ids"]) {
                if (id.is_string()) ids.push_back(id.get<std::string>());
            }
            tool_memory_.remember(ids, entry["raw"].get<std::string>());
            ++restored;
        }
    });
    if (restored > 0) {
        std::fprintf(stderr,
            "[disk-cache] restored %zu tool-call replay entries\n", restored);
    }
}

void HttpServer::record_session_aux(const ParsedRequest & req,
                                    const SseEmitter & emitter) {
    std::vector<int> slots;
    slots.swap(request_session_slots_);
    if (disk_cache_.disabled()) return;
    std::string blob;
    bool computed = false;
    for (const int slot : slots) {
        if (!slot_tokens_.count(slot)) continue;
        if (!computed) {
            blob = session_replay_blob(req, &emitter);
            computed = true;
        }
        slot_aux_[slot] = blob;
    }
}

void HttpServer::flush_sessions_to_disk(const char * reason) {
    if (disk_cache_.disabled()) return;
    const auto t0 = std::chrono::steady_clock::now();
    const double limit = config_.disk_cache_shutdown_flush_s;
    const auto elapsed = [&] {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    };
    // The views of shortened conversations first: they are small, and a shortened conversation's KV state is only
    // reused through its view.
    if (pflash_unified_disk_views_.enabled()) {
        size_t views = 0;
        for (const auto & view : pflash_unified_views_.all()) views += pflash_unified_disk_views_.save(view) ? 1 : 0;
        std::fprintf(stderr, "[pflash-unified] %s flush: %zu views written\n", reason, views);
    }
    size_t queued = 0, backups = 0, skipped = 0;
    const auto before = disk_cache_.stats();
    // Graceful shutdown evicts everything: each conversation's newest
    // snapshot and the heads, most recently used first, so the time limit
    // drops the least valuable ones. Queuing can wait for the writer
    // (bounded queued bytes), so the limit is checked between snapshots.
    for (const auto & entry : prefix_cache_.entries_for_shutdown()) {
        if (!entry.persist) {
            ++backups;
            continue;
        }
        if (limit >= 0 && elapsed() >= limit) {
            ++skipped;
            continue;
        }
        if (persist_evicted_slot(entry.slot, true, reason)) ++queued;
        forget_inline_slot_metadata(entry.slot);
    }
    const auto drained = disk_cache_.drain(
        limit < 0 ? -1.0 : (std::max)(0.0, limit - elapsed()));
    std::fprintf(stderr,
        "[disk-cache] %s flush: %zu snapshots queued (%zu edit backups not "
        "written), %llu files %.1f MB written in %.2fs (limit %.0fs; %llu "
        "writes and %zu snapshots dropped by it)\n",
        reason, queued, backups,
        (unsigned long long)(disk_cache_.stats().writes - before.writes),
        (double)(disk_cache_.stats().write_bytes - before.write_bytes) / (1024.0 * 1024.0),
        elapsed(), limit,
        (unsigned long long)drained.cancelled, skipped);
}

void HttpServer::trim_snapshots_after_commit(int slot) {
    const auto release = [this](const std::vector<int> & slots) {
        for (const int released : slots) {
            forget_inline_slot_metadata(released);
            backend_.snapshot_free(released);
        }
    };
    release(prefix_cache_.prune_superseded_ancestors(slot));
    std::vector<bool> persist;
    const auto evicted = prefix_cache_.enforce_resident_budget(slot, &persist);
    for (size_t i = 0; i < evicted.size(); ++i) {
        persist_evicted_slot(evicted[i], persist[i], "ram eviction");
        forget_inline_slot_metadata(evicted[i]);
        backend_.snapshot_free(evicted[i]);
    }
}

void HttpServer::publish_memory_report() {
    const ModelBackend::MemoryReport report = backend_.memory_report();
    json cache = nullptr;
    if (report.available) {
        cache = {
            {"location", report.cache.host ? "host" : "device"},
            {"capacity_tokens", report.cache.capacity_tokens},
            {"live_tokens", report.cache.live_tokens >= 0
                ? json(report.cache.live_tokens) : json(nullptr)},
            {"kv_bytes", report.cache.kv_bytes},
            {"recurrent_bytes", report.cache.recurrent_bytes},
            {"draft_feature_bytes", report.cache.draft_feature_bytes},
            {"other_bytes", report.cache.other_bytes},
            {"host_state_bytes", report.cache.host_state_bytes},
        };
    }
    // Slot ranges: inline prefix entries, then the exact-prompt cache, and
    // the last backend slot stages disk-cache hits.
    const int prefix_slots = prefix_cache_.stats().capacity;
    json slots = json::array();
    uint64_t host_bytes = 0, device_bytes = 0;
    for (const auto & snap : report.snapshots) {
        const char * kind = snap.slot == kDiskStagingSlot ? "disk_staging"
            : snap.slot >= prefix_slots ? "prefill_cache"
            : agent_turn_cache_slots_.count(snap.slot) ? "agent_turn"
            : "prefix";
        // Host-side copies are system RAM wherever the buffer lives.
        (snap.host ? host_bytes : device_bytes) += snap.bytes;
        host_bytes += snap.host_copy_bytes;
        slots.push_back({{"slot", snap.slot}, {"kind", kind},
                         {"tokens", snap.tokens},
                         {"bytes", snap.bytes + snap.host_copy_bytes},
                         {"location", snap.host ? "host" : "device"}});
    }
    json out = {
        {"cache", std::move(cache)},
        {"snapshots", {
            {"count", report.snapshots.size()},
            {"host_bytes", host_bytes},
            {"device_bytes", device_bytes},
            {"slots", std::move(slots)},
        }},
    };
    std::lock_guard<std::mutex> lock(memory_report_mu_);
    memory_report_ = std::move(out);
    ++memory_report_requests_;
}

// Resident and peak resident set of this process, read now.
static json process_memory_json() {
    json out = {{"rss_bytes", nullptr}, {"peak_rss_bytes", nullptr}};
#if defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string key;
    uint64_t kib = 0;
    std::string unit;
    std::string line;
    while (std::getline(status, line)) {
        std::istringstream fields(line);
        if (!(fields >> key >> kib >> unit) || unit != "kB") continue;
        if (key == "VmRSS:") out["rss_bytes"] = kib * 1024;
        if (key == "VmHWM:") out["peak_rss_bytes"] = kib * 1024;
    }
#endif
    return out;
}

json HttpServer::memory_json() const {
    json out;
    {
        std::lock_guard<std::mutex> lock(memory_report_mu_);
        // Before the first report, the same shape with nothing counted.
        out = memory_report_.is_null()
            ? json{{"cache", nullptr},
                   {"snapshots", {{"count", 0}, {"host_bytes", 0},
                                  {"device_bytes", 0},
                                  {"slots", json::array()}}}}
            : memory_report_;
        // Reports are taken at startup and after each request.
        out["reports_published"] = memory_report_requests_;
    }
    out["process"] = process_memory_json();
    return out;
}

bool HttpServer::save_generated_turn(
        const std::vector<int32_t> & prompt,
        const std::vector<int32_t> & generated,
        const std::vector<int32_t> & canonical,
        const GenerationCacheState & cache) {
    // The live state after generation holds the prompt and the generated
    // tokens (the last one sampled may not be decoded yet). It is reusable
    // up to where those tokens stop matching the turn the next request
    // renders, which is all of it when tool memory replays the text exactly.
    const int prompt_len = (int) prompt.size();
    if (canonical.size() <= prompt.size() ||
        !std::equal(prompt.begin(), prompt.end(), canonical.begin())) {
        return false;
    }
    int matched = prompt_len;
    const int limit = (std::min)(
        (int) canonical.size(), prompt_len + (int) generated.size());
    while (matched < limit &&
           generated[(size_t) (matched - prompt_len)] == canonical[(size_t) matched]) {
        ++matched;
    }
    if (matched <= prompt_len) return false;

    // Keep this request's own checkpoints: they are the fallback when the
    // next request diverges inside the generated turn.
    const int keep_slot = cache.snap_prepared ? cache.snap_slot
        : cache.using_restore && !cache.disk_hit ? cache.cache_slot : -1;
    auto reservation = prefix_cache_.reserve_inline_snap(
        canonical, prompt_len, false, matched, keep_slot,
        [this](int target_cut) {
            return backend_.snapshot_bytes_estimate(target_cut);
        });
    if (!reservation.active() || reservation.target_cut() != matched) {
        return false;
    }
    const int slot = reservation.slot();
    evict_inline_slot(slot);
    backend_.snapshot_free(slot);
    // The next request usually continues this state, so the backend may
    // keep it live instead of copying it out.
    const bool saved = backend_.snapshot_save_deferred(slot);
    const int saved_pos = saved ? backend_.snapshot_cur_pos(slot) : 0;
    if (saved_pos <= prompt_len || saved_pos > matched) {
        backend_.snapshot_free(slot);
        reservation.abort();
        return false;
    }
    evict_extra_victims(reservation);
    reservation.commit_at(
        canonical, saved_pos, backend_.snapshot_bytes_estimate(saved_pos));
    std::vector<bool> persist;
    const auto evicted = prefix_cache_.enforce_resident_budget(slot, &persist);
    for (size_t i = 0; i < evicted.size(); ++i) {
        persist_evicted_slot(evicted[i], persist[i], "ram eviction");
        forget_inline_slot_metadata(evicted[i]);
        backend_.snapshot_free(evicted[i]);
    }
    if (cache.disk_policy.mode == DiskPrefixCacheMode::Full) {
        slot_tokens_[slot] = std::vector<int32_t>(
            canonical.begin(), canonical.begin() + saved_pos);
        note_session_slot(slot);
    }
    agent_turn_cache_slots_.insert(slot);
    std::fprintf(stderr,
        "[agent-turn-cache] saved generated turn slot=%d prefix=%d "
        "(prompt=%d generated=%zu)\n",
        slot, saved_pos, prompt_len, generated.size());
    return true;
}

void HttpServer::remember_agent_turn(
        const ParsedRequest & req, const PreparedPrompt & prepared,
        const GenerationCacheState & cache, const GenerateResult & result,
        const SseEmitter & emitter, int completion_tokens,
        bool visible_output_seen, bool client_disconnected,
        bool replay_cache) {
    const bool supported_format = req.format == ApiFormat::OPENAI_CHAT ||
        req.format == ApiFormat::RESPONSES;
    const bool valid_tool_turn = result.ok() && completion_tokens > 0 &&
        visible_output_seen && !client_disconnected && !req.tools.empty() &&
        !emitter.tool_calls().empty() && !emitter.accumulated_raw().empty();
    // Arbitrary Jinja templates need structured calls to render their results;
    // raw-turn caching cannot preserve that contract.
    if (!supported_format || !valid_tool_turn || !config_.chat_template_src.empty()) return;

    std::vector<ChatMessage> messages =
        normalize_chat_messages(req.messages, req.format, tool_memory_, true);
    static constexpr const char * kSentinel =
        "__LUCE_AGENT_TURN_CONTENT_7A21D9__";
    messages.push_back({"assistant", kSentinel});

    std::string sentinel_rendered;
    std::string render_error;
    if (!render_messages_to_text(
            messages, req, /*add_generation_prompt=*/false,
            sentinel_rendered, render_error)) {
        std::fprintf(stderr, "[agent-turn-cache] render skipped: %s\n",
                     render_error.c_str());
        return;
    }
    // Preserve template-injected generation text (for example Qwen's
    // <think> prefix) in tool memory so the next stateless request renders the
    // same token stream as the checkpointed generation.
    std::string assistant_content;
    if (!http_detail::canonical_assistant_content(
            req.rendered_prompt, sentinel_rendered, kSentinel,
            emitter.accumulated_raw(), assistant_content)) {
        return;
    }
    messages.back().content = assistant_content;

    // Stateless tool APIs send structured calls back on the next request.
    // Retain the exact generated text so normalization recreates the same
    // assistant turn, even when no compatible KV checkpoint is available.
    std::vector<std::string> call_ids;
    for (const auto & call : emitter.tool_calls()) call_ids.push_back(call.id);
    tool_memory_.remember(call_ids, assistant_content);

    if (!replay_cache || req.images) return;
    if (prefix_cache_.disabled()) return;
    // Cache only stateless-equivalent prompts. Compression and token rewrites
    // need a separate replay contract.
    if (prepared.compressed || prepared.tokens != req.prompt_tokens) return;

    auto & replayed_turn = messages.back();
    replayed_turn.tool_calls_replayed = true;
    replayed_turn.tool_calls.reserve(emitter.tool_calls().size());
    for (const auto & call : emitter.tool_calls()) {
        replayed_turn.tool_calls.push_back({call.id, call.name, {}});
    }

    std::string canonical_rendered;
    if (!render_messages_to_text(
            messages, req, /*add_generation_prompt=*/false,
            canonical_rendered, render_error)) {
        return;
    }
    std::vector<int32_t> canonical_tokens = tokenizer_.encode(canonical_rendered);
    if (save_generated_turn(prepared.tokens, result.tokens, canonical_tokens,
                            cache)) {
        return;
    }
    if (!config_.agent_turn_cache || has_pending_jobs()) return;

    // Reuse the deepest checkpoint ordinary prefix caching already produced.
    // Matching at the checkpoint, instead of at the prompt end, tolerates the
    // BPE boundary change caused by appending the generated assistant turn.
    int source_slot = -1;
    int source_pos = 0;
    auto consider_source = [&](int slot) {
        if (slot < 0 || !backend_.snapshot_used(slot)) return;
        const int pos = backend_.snapshot_cur_pos(slot);
        if (pos <= source_pos ||
            !http_detail::canonical_turn_matches_checkpoint(
                prepared.tokens, canonical_tokens, pos)) {
            return;
        }
        source_slot = slot;
        source_pos = pos;
    };
    if (cache.using_restore) consider_source(cache.cache_slot);
    if (cache.snap_prepared) consider_source(cache.snap_slot);
    if (source_slot < 0) {
        std::fprintf(stderr,
            "[agent-turn-cache] no compatible prefix checkpoint; skipped\n");
        return;
    }

    const int canonical_end = (int) canonical_tokens.size();
    // The replay saves at the end of its prefill, which every backend can do
    // whatever its snapshot granularity.
    auto reservation = prefix_cache_.reserve_inline_snap(
        canonical_tokens, source_pos, false, canonical_end, source_slot,
        [this](int target_cut) {
            return backend_.snapshot_bytes_estimate(target_cut);
        });
    // No safe victim (only the restore source and/or protected pins remain)
    // or no useful boundary: nothing to replay into.
    if (!reservation.active() ||
        reservation.target_cut() != canonical_end) return;

    const int slot = reservation.slot();
    if (slot == source_slot) {
        reservation.cancel();
        return;
    }
    evict_inline_slot(slot);
    backend_.snapshot_free(slot);

    GenerateRequest replay;
    replay.prompt = canonical_tokens;
    replay.n_gen = 0;
    replay.snap_slot = slot;
    replay.snap_pos = canonical_end;
    DaemonIO replay_io;
    replay_io.should_cancel = [this]() { return has_pending_jobs(); };
    const GenerateResult replay_result = backend_.restore_and_generate(
        source_slot, replay, replay_io);
    backend_.release_scratch();
    const int saved_pos = replay_result.ok() && backend_.snapshot_used(slot)
        ? backend_.snapshot_cur_pos(slot) : 0;
    if (saved_pos > source_pos && saved_pos <= canonical_end) {
        evict_extra_victims(reservation);
        reservation.commit_at(
            canonical_tokens, saved_pos,
            backend_.snapshot_bytes_estimate(saved_pos));
        trim_snapshots_after_commit(slot);
        canonical_tokens.resize((size_t) saved_pos);
        if (cache.disk_policy.mode == DiskPrefixCacheMode::Full) {
            slot_tokens_[slot] = std::move(canonical_tokens);
            note_session_slot(slot);
        }
        agent_turn_cache_slots_.insert(slot);
        std::fprintf(stderr,
            "[agent-turn-cache] saved slot=%d prefix=%d source=%d:%d "
            "replayed=%d\n",
            slot, saved_pos, source_slot, source_pos,
            canonical_end - source_pos);
    } else {
        backend_.snapshot_free(slot);
        reservation.abort();
    }
}

// Populate model-ready input. GenerateRequest owns every retained token
// sequence, eliminating pointer lifetime coupling to GenerationInputs.
void HttpServer::prepare_generation_inputs(
        const ParsedRequest & req, const PreparedPrompt & prepared,
        GenerationInputs & inputs) {
    const bool budget_active = req.thinking_opt_in;
    const int thinking_ceiling = req.per_req_phase1_cap >= 0
        ? req.per_req_phase1_cap
        : config_.think_max_tokens;
    const int reply_budget = luce::common::reply_reserve_within(
        req.per_req_reply_budget >= 0 ? req.per_req_reply_budget
                                      : config_.hard_limit_reply_budget,
        req.max_output);
    inputs.generation_cap = budget_active
        ? (std::min)(thinking_ceiling + reply_budget, req.max_output)
        : req.max_output;

    inputs.request.prompt = prepared.tokens;
    inputs.request.images = prepared.images;
    // Image requests may speculate; each backend decides (Qwen3.5 verifies at
    // image-shifted rotary positions, DeepSeek4 drafts from the text after
    // the last image).
    inputs.request.n_gen = inputs.generation_cap;
    inputs.request.sampler = req.sampler;
    inputs.request.do_sample = req.sampler.needs_logit_processing();
    // A grammar masks one position at a time: no speculative blocks.
    inputs.request.force_ar_decode = req.sampler.constraint != nullptr;
    // Tokens are delivered through DaemonIO so all API formats share the
    // same disconnect and streaming state machine.
    inputs.request.stream = false;

    // The budget hook injects the close sequence while KV state is live,
    // leaving the remaining reserve for a visible answer.
    if (budget_active && !config_.think_close_token_ids.empty() &&
        config_.hard_limit_reply_budget > 0) {
        inputs.request.budget_hook.close_token_ids =
            config_.think_close_token_ids;
        inputs.request.budget_hook.marker_token_ids =
            config_.think_marker_token_ids;
        inputs.request.budget_hook.hard_limit_remaining = reply_budget;
    }

    if (!req.tools.empty() && !req.tool_choice.is_null()) {
        ToolHintGenerator hint_generator(tokenizer_);
        auto hint = hint_generator.build_hint(req.tools, req.tool_choice);
        if (!hint.empty()) {
            inputs.request.hint_tokens = std::move(hint.prefix_tokens);
        }
    }

    if (req.tools.empty() || !env_flag_enabled("LUCE_STALL_TOOL_PREFIX")) {
        return;
    }

    inputs.request.stall_tool_prefix_tokens = tokenizer_.encode(
        build_stall_tool_prefix(req.tools, req.tool_choice));
    inputs.request.stall_action_suffix_tokens = tokenizer_.encode(":");

    // The detector matches recent terminal tokens, not the full action
    // prefix. Collect the final token for common colon spellings.
    auto add_suffix_terminal = [&](const std::string & text) {
        const auto ids = tokenizer_.encode(text);
        if (ids.empty()) return;
        const int32_t token = ids.back();
        if (std::find(inputs.request.stall_action_suffix_tokens.begin(),
                      inputs.request.stall_action_suffix_tokens.end(), token) ==
            inputs.request.stall_action_suffix_tokens.end()) {
            inputs.request.stall_action_suffix_tokens.push_back(token);
        }
    };
    add_suffix_terminal("`:");
    add_suffix_terminal("):");
    add_suffix_terminal("\":");

    inputs.request.stall_skip_tokens = tokenizer_.encode(" done");
}

void HttpServer::configure_generation_io(
        ServerJob * job, const ParsedRequest & req, SseEmitter & emitter,
        GenerationOutputState & output, DaemonIO & io) {
    io.stream_fd = -1;
    io.should_cancel = [job]() {
        return job->client_disconnected.load(std::memory_order_acquire);
    };
    io.observer = [this](const char *, const std::vector<int32_t> & tokens) {
        std::vector<std::string> token_strings;
        token_strings.reserve(tokens.size());
        for (int32_t token : tokens) {
            token_strings.push_back(tokenizer_.token_text(token));
        }
        status_.set_draft_tokens(token_strings);
        broadcast_status();
    };

    io.on_token = [this, job, &req, &emitter, &output](
            int32_t token) -> bool {
        // The grammar moves with every token the request keeps, before the
        // backend samples the next one.
        if (req.sampler.constraint) req.sampler.constraint->accept(token);
        if (output.client_disconnected ||
            job->client_disconnected.load(std::memory_order_acquire)) {
            output.client_disconnected = true;
            return false;
        }
        ++output.completion_tokens;

        if (output.completion_tokens == 1) {
            // Prefill is over: the first generated token marks the decode phase.
            // The count goes with it, so the transition a client sees is not
            // "decode with nothing generated" — the counter otherwise only
            // moves on multiples of ten.
            status_.set_decode();
            status_.update_completion_tokens(output.completion_tokens);
            broadcast_status();
        }
        if (output.completion_tokens % 10 == 0) {
            status_.update_completion_tokens(output.completion_tokens);
            broadcast_status();
        }

        std::string text;
        const TokenDelivery delivery =
            classify_generated_token(tokenizer_, emitter, token, text);
        if (delivery == TokenDelivery::kSkip) return true;

        if (!text.empty()) {
            output.visible_output_seen = true;
            broadcast_token(text);
        }
        if (!req.stream || text.empty()) return true;

        for (const auto & chunk : emitter.emit_token(text)) {
            if (!send_job_bytes(job, chunk.data(), chunk.size())) {
                output.client_disconnected = true;
                return false;
            }
        }
        // Only ordinary text is checked against stop sequences — think-tag
        // markers never terminate generation.
        return delivery == TokenDelivery::kThinkTag || !emitter.stop_hit();
    };
}

bool HttpServer::deliver_generation_token(
        ServerJob * job, const ParsedRequest & req, SseEmitter & emitter,
        int32_t token, int & completion_tokens,
        ClientSendBuffer & send_buffer) {
    ++completion_tokens;

    std::string text;
    const TokenDelivery delivery =
        classify_generated_token(tokenizer_, emitter, token, text);
    if (delivery == TokenDelivery::kSkip) return true;

    // Non-stream replay counts every non-skipped token, including tokens
    // whose decoded text is empty. Keep concurrent usage accounting aligned;
    // streaming still has no frame to send for an empty string.
    if (text.empty() && req.stream) return true;

    const auto chunks = emitter.emit_token(text);
    if (req.stream && !chunks.empty()) {
        // Disable silent-prefill heartbeats only once there is a real frame
        // to buffer. Complete a partial comment ahead of that frame.
        stop_job_stream(job, &send_buffer);
        for (const auto & chunk : chunks) {
            send_buffer.append(chunk);
        }
    }
    return delivery == TokenDelivery::kThinkTag || !emitter.stop_hit();
}

void HttpServer::send_nonstream_response(
        const ParsedRequest & req, SocketHandle fd, SseEmitter & emitter,
        const std::vector<int32_t> & gen_tokens, int n_gen_cap,
        bool budget_forced_close, bool degenerate_decode_close,
        const GenTimings & gen_timings,
        ClientSendBuffer * send_buffer) {
    CompletionTokenCounts counts;
    counts.total = (int) gen_tokens.size();
    const bool is_eos = !gen_tokens.empty() &&
        (gen_tokens.back() == tokenizer_.eos_id() ||
         gen_tokens.back() == tokenizer_.eos_chat_id());
    emitter.emit_finish(counts.total, nullptr, n_gen_cap, is_eos);
    const int first_content = emitter.first_content_token_index();
    const int emitted = emitter.emit_token_count();
    counts.reasoning = first_content < 0 ? emitted : first_content;
    counts.content = first_content < 0 ? 0 : emitted - first_content;

    GenerateResult result;
    result.tokens = gen_tokens;
    result.budget_forced_close = budget_forced_close;
    result.degenerate_decode_close = degenerate_decode_close;

    const json response = build_non_streaming_response(
        req, result, n_gen_cap, gen_timings, counts, emitter, &tokenizer_);

    const std::string body = response.dump() + "\n";
    if (send_buffer) {
        send_buffer->append(
            format_http_response(200, "application/json", body));
    } else {
        send_response(fd, 200, "application/json", body);
    }
}

void HttpServer::worker_loop() {
    publish_memory_report();
    while (true) {
        ServerJob * job = dequeue();
        if (!job) break;  // stopping

        process_job(job);
        publish_memory_report();
    }
}

void HttpServer::process_job(ServerJob * job) {
    SocketHandle fd = job->fd;
    const auto & req = job->req;
    auto started_at = std::chrono::steady_clock::now();

    // Track live status for /status page. RAII guard ensures idle on all paths.
    std::string prompt_excerpt;
    if (req.images) {
        prompt_excerpt = req.rendered_prompt.substr(0, 200);
    } else if (!req.prompt_tokens.empty()) {
        // Decode first ~40 tokens as a prompt excerpt (cheap, bounded).
        const int excerpt_len = (std::min)((int)req.prompt_tokens.size(), 40);
        std::vector<int32_t> excerpt_toks(req.prompt_tokens.begin(),
                                           req.prompt_tokens.begin() + excerpt_len);
        prompt_excerpt = tokenizer_.decode(excerpt_toks);
        if (prompt_excerpt.size() > 200) prompt_excerpt.resize(200);
    }
    {
        ServerStatus::RequestInfo info;
        info.model = req.model;
        info.format = api_format_name(req.format);
        info.session_id = req.session_id;
        info.max_output = req.max_output;
        info.temperature = req.sampler.temp;
        info.top_p = req.sampler.top_p;
        info.top_k = req.sampler.top_k;
        info.thinking_enabled = req.thinking_enabled;
        status_.set_running(prompt_excerpt, (int)req.prompt_tokens.size(), req.stream, info);
    }
    // Store messages JSON for request inspection (truncate to avoid huge payloads).
    if (!req.messages.is_null()) {
        std::string msg_str = req.messages.dump();
        if (msg_str.size() > 4096) msg_str.resize(4096);
        status_.set_messages(msg_str);
    }
    broadcast_status();
    StatusGuard status_guard{status_};

    auto finish_job = [&]() {
        stop_job_stream(job);
        std::lock_guard<std::mutex> lk(job->mu);
        job->done = true;
        job->cv.notify_one();
    };
    std::fprintf(stderr,
        "[server] chat START %s format=%s stream=%s prompt_tokens=%zu "
        "max_tokens=%d tools=%zu\n",
        req.response_id.c_str(),
        api_format_name(req.format),
        req.stream ? "true" : "false",
        req.prompt_tokens.size(),
        req.max_output,
        json_array_size(req.tools));

    // The server owns the downstream SSE transport for local and proxied
    // generation so both paths share heartbeat and disconnect handling.
    if (req.stream) {
        if (!send_sse_headers(job)) {
            finish_job();
            return;
        }
    }

    // Create SSE emitter for streaming state machine.
    SseEmitter emitter(req.format, req.response_id, req.model,
                       (int)req.prompt_tokens.size(), req.tools,
                       &tool_memory_,
                       req.stop_sequences,
                       req.started_in_thinking);

    // Emit initial SSE events only for local generation. The upstream owns
    // the proxied event sequence.
    if (req.stream && config_.pflash_upstream_base.empty()) {
        bool start_ok = true;
        for (const auto & chunk : emitter.emit_start()) {
            if (!send_job_bytes(job, chunk.data(), chunk.size())) {
                start_ok = false;
                break;
            }
        }
        if (!start_ok) {
            finish_job();
            return;
        }
    }
    if (req.stream) start_job_stream(job);

    auto fail_request = [&](int status, const std::string & message) {
        std::fprintf(stderr, "[server] request failed: %s\n", message.c_str());
        ResponseError error;
        if (status == 400) {
            error = ResponseError::invalid_request(
                "invalid_request", message);
        } else if (status == 503) {
            error = ResponseError::unavailable("unavailable", message);
        } else {
            error = ResponseError::internal("server_error", message);
        }
        stop_job_stream(job);
        if (req.stream) {
            for (const std::string & chunk : emitter.emit_error(error)) {
                send_job_bytes(job, chunk.data(), chunk.size());
            }
        } else {
            const json body = build_error_response(
                req.format, error, req.response_id);
            send_response(fd, response_error_http_status(error),
                          "application/json", body.dump() + "\n");
        }
        finish_job();
    };

    PreparedPrompt prepared = prepare_prompt(req);
    if (prepared.error_status != 0) {
        fail_request(prepared.error_status, prepared.error);
        return;
    }
    if (forward_upstream(job, req, prepared)) {
        finish_job();
        return;
    }

    auto & effective_prompt = prepared.tokens;
    const bool pflash_compressed = prepared.compressed;

    GenerationInputs generation_inputs;
    prepare_generation_inputs(req, prepared, generation_inputs);
    GenerateRequest & gen_req = generation_inputs.request;
    const int n_gen_cap = generation_inputs.generation_cap;

    GenerationCacheState cache =
        prepare_generation_cache(req, prepared, gen_req);
    const bool using_restore = cache.using_restore;
    const int cache_slot = cache.cache_slot;
    const int prefix_len = cache.prefix_len;

    DaemonIO io;
    GenerationOutputState output;
    configure_generation_io(job, req, emitter, output, io);
    int & completion_tokens = output.completion_tokens;
    bool & visible_output_seen = output.visible_output_seen;
    bool & client_disconnected = output.client_disconnected;

    const auto dflash_residency =
        resolve_draft_residency_action(
            config_.draft_residency,
            DraftResidencyContext{
                DraftResidencyUse::DFlashDecode,
                config_.lazy_draft,
                !config_.draft_path.empty(),
            });

    // Run generation (with or without restore).
    // Request-scoped draft residency ensures decode draft is loaded only
    // around the generation window, leaving room for PFlash/target state.
    if (dflash_residency == DraftResidencyAction::ReleaseAfterUse &&
        !config_.draft_path.empty()) {
        backend_.free_drafter();    // free pflash drafter (~1.4 GB) if loaded
        backend_.unpark(ParkTarget::DraftModel);   // reload decode draft (~3.3 GB)
    }

    // The status stays in PREFILL until the first generated token (set in the
    // token callback): generate() below prefills first, so flipping here makes
    // a long prompt report decode for the whole time it is still prefilling.

    GenerateResult result;
    if (using_restore) {
        result = backend_.restore_and_generate(cache_slot, gen_req, io);
    } else {
        result = backend_.generate(gen_req, io);
    }

    if (dflash_residency == DraftResidencyAction::ReleaseAfterUse &&
        !config_.draft_path.empty()) {
        backend_.park(ParkTarget::DraftModel);
    }

    if (job->client_disconnected.load(std::memory_order_acquire)) {
        client_disconnected = true;
    }

    // Release oversized scratch buffers (gallocr, BSA cache) so VRAM
    // doesn't grow monotonically across requests with different sizes.
    backend_.release_scratch();

    // Bandit: update when spec decode actually ran — including 0-accept case,
    // which signals the current keep_ratio is too low.
    if (result.ok() && !req.session_id.empty() && result.spec_decode_ran) {
        const float configured_keep =
            pflash_keep_ratio(config_, (int) req.prompt_tokens.size());
        float old_keep = sessions_.get_keep_ratio(req.session_id, configured_keep);
        int   old_turn = sessions_.turn_count(req.session_id);
        sessions_.update(req.session_id, result.accept_rate, configured_keep);
        float new_keep = sessions_.get_keep_ratio(req.session_id);
        float ema      = sessions_.get_ema(req.session_id);
        std::fprintf(stderr,
            "[pflash-bandit] session=%s turn=%d keep=%.4f->%.4f ema=%.3f accept=%.3f\n",
            req.session_id.c_str(), old_turn + 1,
            old_keep, new_keep, ema, result.accept_rate);
    }

    // Finalizing may prune the restore source, and its agent-turn mark with
    // it, once a deeper snapshot lands.
    const bool restored_agent_turn =
        agent_turn_cache_slots_.count(cache_slot) != 0;
    finalize_generation_cache(
        req, prepared, cache, result, completion_tokens,
        visible_output_seen, client_disconnected);

    auto log_done = [&]() {
        const auto done_at = std::chrono::steady_clock::now();
        const double elapsed_s =
            std::chrono::duration<double>(done_at - started_at).count();
        const int result_tokens = (int)result.tokens.size();
        const int out_tokens = (std::max)(completion_tokens, result_tokens);
        const double tok_s = elapsed_s > 0.0 ? out_tokens / elapsed_s : 0.0;
        const double decode_tok_s =
            result.decode_s > 0.0 ? out_tokens / result.decode_s : 0.0;
        const std::string finish = client_disconnected
            ? "client_disconnect"
            : (result.ok() ? emitter.finish_reason() : "error");

        std::fprintf(stderr,
            "[server] chat DONE %s ok=%s in=%zu effective_in=%zu out=%d "
            "%.1fs %.1f tok/s finish=%s restore=%s slot=%d prefix_len=%d "
            "prefill=%.1fs decode=%.1fs(%.1ftok/s) error=%s detail=%s\n",
            req.response_id.c_str(),
            result.ok() ? "true" : "false",
            req.prompt_tokens.size(),
            effective_prompt.size(),
            out_tokens,
            elapsed_s,
            tok_s,
            finish.c_str(),
            using_restore ? "true" : "false",
            cache_slot,
            prefix_len,
            result.prefill_s,
            result.decode_s,
            decode_tok_s,
            result.ok() ? "-" : result.error_code().data(),
            result.error_detail().empty() ? "-" : result.error_detail().data());
    };

    // A backend failure terminates the request here. Everything below this
    // branch records or frames a successful generation.
    if (!result.ok()) {
        stop_job_stream(job);
        if (job->client_disconnected.load(std::memory_order_acquire)) {
            client_disconnected = true;
        }
        if (!client_disconnected) {
            const ResponseError error = to_response_error(*result.error);
            if (req.stream) {
                for (const std::string & chunk : emitter.emit_error(error)) {
                    if (!send_job_bytes(job, chunk.data(), chunk.size())) {
                        client_disconnected = true;
                        break;
                    }
                }
            } else {
                const json body = build_error_response(
                    req.format, error, req.response_id);
                sock_set_block(fd);
                send_response(fd, response_error_http_status(error),
                              "application/json", body.dump() + "\n");
            }
        }
        log_done();
        finish_job();
        return;
    }

    // Per-request wall-clock timings forwarded to the response's
    // `usage.timings` (OpenAI Chat usage chunk, Anthropic
    // message_delta usage, Responses response.completed usage).
    // See docs/specs/thinking-budget.md §6.3.
    const int effective_prompt_tokens = (int) effective_prompt.size();
    const int cached_prefix_tokens = (std::clamp)(
        result.restored_prefix_tokens, 0, effective_prompt_tokens);
    const bool cache_hit = cached_prefix_tokens > 0;
    const bool agent_turn_cache_hit = cache_hit && restored_agent_turn;
    GenTimings gen_timings{
        result.prefill_s,
        result.decode_s,
        cache_hit,
        cached_prefix_tokens,
        effective_prompt_tokens - cached_prefix_tokens,
        effective_prompt_tokens,
        agent_turn_cache_hit,
        prepared.pflash_stats,
    };

    // Record performance for /status page.
    PerfRecord perf;
    perf.prompt_tokens = effective_prompt_tokens;
    perf.completion_tokens = completion_tokens;
    // Use actual prefilled token count: on cache hit the backend only
    // prefills the delta beyond the cached prefix, so dividing the full
    // prompt size by delta time would be wrong.
    const int prefill_tokens =
        (std::max)(0, effective_prompt_tokens - cached_prefix_tokens);
    perf.prefill_tok_s = (result.prefill_s > 0.0)
        ? (double)prefill_tokens / result.prefill_s : 0.0;
    perf.decode_tok_s = (result.decode_s > 0.0)
        ? (double)completion_tokens / result.decode_s : 0.0;
    perf.accept_rate = result.accept_rate;
    perf.cache_hit = cache_hit;
    perf.pflash = pflash_compressed;
    perf.spec_decode = result.spec_decode_ran;
    perf.timestamp = std::chrono::steady_clock::now();
    status_.record_perf(perf);
    status_.update_completion_tokens(completion_tokens);
    broadcast_status();
    // Serialize final frames after disabling heartbeat comments so no comment
    // can appear after the protocol's [DONE] marker.
    stop_job_stream(job);
    if (job->client_disconnected.load(std::memory_order_acquire)) {
        client_disconnected = true;
    }
    const bool is_eos = !result.tokens.empty() &&
        (result.tokens.back() == tokenizer_.eos_id() ||
         result.tokens.back() == tokenizer_.eos_chat_id());
    if (req.stream && !client_disconnected) {
        auto final_chunks = emitter.emit_finish(
            completion_tokens, &gen_timings, n_gen_cap, is_eos);
        remember_agent_turn(
            req, prepared, cache, result, emitter, completion_tokens,
            visible_output_seen, client_disconnected,
            /*replay_cache=*/false);
        for (const auto & chunk : final_chunks) {
            if (!send_job_bytes(job, chunk.data(), chunk.size())) {
                client_disconnected = true;
                break;
            }
        }
    } else if (!req.stream && !client_disconnected) {
        const json response = build_non_streaming_response(
            req, result, n_gen_cap, gen_timings, tokenizer_, emitter);
        remember_agent_turn(
            req, prepared, cache, result, emitter, completion_tokens,
            visible_output_seen, client_disconnected,
            /*replay_cache=*/false);
        // Streaming uses non-blocking sends; restore blocking mode before
        // writing a complete JSON response on this shared socket path.
        sock_set_block(fd);
        send_response(fd, 200, "application/json",
                      response.dump() + "\n");
    }

    remember_agent_turn(
        req, prepared, cache, result, emitter, completion_tokens,
        visible_output_seen, client_disconnected,
        /*replay_cache=*/true);
    record_session_aux(req, emitter);

    if (client_disconnected) {
        std::fprintf(stderr, "[server] client disconnected — generation aborted "
                     "(prompt=%zu out=%d)\n",
                     req.prompt_tokens.size(), completion_tokens);
    }

    log_done();

    // Signal client thread that we're done.
    finish_job();
}

// ─── Job queue ──────────────────────────────────────────────────────────

void HttpServer::enqueue(ServerJob * job) {
    std::lock_guard<std::mutex> lk(queue_mu_);
    if (stopping_.load()) {
        // Return routed requests to the listener for a shutdown response.
        std::lock_guard<std::mutex> jlk(job->mu);
        job->admission = RoutingAdmission::busy;
        job->done = true;
        job->cv.notify_one();
        return;
    }
    job->next = nullptr;
    if (queue_tail_) queue_tail_->next = job;
    else queue_head_ = job;
    queue_tail_ = job;
    queue_cv_.notify_one();
}

bool HttpServer::has_pending_jobs() {
    std::lock_guard<std::mutex> lock(queue_mu_);
    return queue_head_ != nullptr;
}

ServerJob * HttpServer::dequeue() {
    std::unique_lock<std::mutex> lk(queue_mu_);
    // Use timed wait so the worker periodically wakes to send SSE heartbeats.
    while (!queue_head_ && !stopping_.load()) {
        if (queue_cv_.wait_for(lk, std::chrono::seconds(30)) == std::cv_status::timeout) {
            // Send SSE heartbeat (comment line) to detect disconnected clients.
            lk.unlock();
            sse_heartbeat();
            lk.lock();
        }
    }
    if (!queue_head_) return nullptr;
    ServerJob * j = queue_head_;
    queue_head_ = j->next;
    if (!queue_head_) queue_tail_ = nullptr;
    j->next = nullptr;
    return j;
}

ServerJob * HttpServer::try_dequeue() {
    std::lock_guard<std::mutex> lk(queue_mu_);
    if (!queue_head_) return nullptr;
    ServerJob * job = queue_head_;
    queue_head_ = job->next;
    if (!queue_head_) queue_tail_ = nullptr;
    job->next = nullptr;
    return job;
}

ServerJob * HttpServer::dequeue_for(
        std::chrono::steady_clock::duration timeout) {
    std::unique_lock<std::mutex> lk(queue_mu_);
    if (!queue_head_ && !stopping_.load() &&
        timeout > std::chrono::steady_clock::duration::zero()) {
        queue_cv_.wait_for(lk, timeout, [&] {
            return queue_head_ != nullptr || stopping_.load();
        });
    }
    if (!queue_head_) return nullptr;
    ServerJob * job = queue_head_;
    queue_head_ = job->next;
    if (!queue_head_) queue_tail_ = nullptr;
    job->next = nullptr;
    return job;
}

// ─── HTTP I/O ───────────────────────────────────────────────────────────

bool HttpServer::read_http_request(SocketHandle fd, HttpRequest & out) {
    sock_set_nonblock(fd);
    auto receive = [&](char * data, size_t size) -> ssize_t {
        while (!stopping_.load()) {
            struct pollfd pfd{fd, POLLIN, 0};
            const int ready = poll(&pfd, 1, 100);
            if (ready < 0 && sock_is_eintr(sock_errno())) continue;
            if (ready < 0 || (pfd.revents & (POLLERR | POLLNVAL))) return -1;
            if (ready == 0) continue;
            const ssize_t n = recv(fd, data, size, 0);
            if (n < 0 && (sock_is_eintr(sock_errno()) || sock_is_eagain(sock_errno()))) continue;
            return n;
        }
        return -1;
    };
    std::string buf;
    buf.reserve(8192);
    char tmp[4096];

    // Read until we find the header/body boundary (\r\n\r\n or \n\n).
    ssize_t hend = -1;
    while (hend < 0 && buf.size() < 65536) {
        ssize_t n = receive(tmp, sizeof(tmp));
        if (n <= 0) return false;
        buf.append(tmp, n);

        // Look for end of headers.
        for (size_t i = 3; i < buf.size(); i++) {
            if (buf[i-3] == '\r' && buf[i-2] == '\n' &&
                buf[i-1] == '\r' && buf[i] == '\n') {
                hend = i + 1;
                break;
            }
        }
        if (hend < 0) {
            for (size_t i = 1; i < buf.size(); i++) {
                if (buf[i-1] == '\n' && buf[i] == '\n') {
                    hend = i + 1;
                    break;
                }
            }
        }
    }
    if (hend < 0) return false;

    // Parse request line.
    size_t line_end = buf.find('\n');
    if (line_end == std::string::npos) return false;
    std::string line = buf.substr(0, line_end);
    if (!line.empty() && line.back() == '\r') line.pop_back();

    // "METHOD /path HTTP/1.1"
    size_t sp1 = line.find(' ');
    size_t sp2 = line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    out.method = line.substr(0, sp1);
    out.path = line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Separate query string from path.
    std::string query_string;
    size_t q = out.path.find('?');
    if (q != std::string::npos) {
        query_string = out.path.substr(q + 1);
        out.path = out.path.substr(0, q);
    }
    out.query = std::move(query_string);

    // Find Content-Length.
    long content_length = 0;
    {
        std::string headers = buf.substr(0, hend);
        std::string lower_headers = headers;
        std::transform(lower_headers.begin(), lower_headers.end(),
                       lower_headers.begin(), ::tolower);
        size_t cl_pos = lower_headers.find("content-length:");
        if (cl_pos != std::string::npos) {
            size_t val_start = cl_pos + 15;
            while (val_start < lower_headers.size() &&
                   lower_headers[val_start] == ' ') val_start++;
            content_length = std::strtol(headers.c_str() + val_start, nullptr, 10);
        }
    }

    if (content_length < 0 || content_length > 64 * 1024 * 1024) return false;

    // Read body.
    while ((ssize_t)buf.size() < hend + content_length) {
        ssize_t n = receive(tmp, sizeof(tmp));
        if (n <= 0) return false;
        buf.append(tmp, n);
    }

    out.body = buf.substr(hend, content_length);
    return true;
}

bool HttpServer::send_all(SocketHandle fd, const void * data, size_t len) {
    const char * p = (const char *)data;
    size_t sent = 0;
    // Stall deadline resets on each successful write (ds4 pattern).
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (sent < len) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) return false;  // stall timeout

        struct pollfd pfd = {fd, POLLOUT, 0};
        int timeout = remaining > 50 ? 50 : (int)remaining;
        int ret;
        do {
            ret = poll(&pfd, 1, timeout);
        } while (ret < 0 && sock_is_eintr(sock_errno()));
        if (ret < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
        if (ret == 0) continue;  // poll timeout, retry until deadline

        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (sock_is_eintr(sock_errno())) continue;
            if (sock_is_eagain(sock_errno())) continue;
            return false;  // EPIPE, ECONNRESET, etc.
        }
        sent += n;
        deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    }
    return true;
}

bool HttpServer::send_job_bytes(
        ServerJob * job, const void * data, size_t len) {
    std::lock_guard<std::mutex> lock(job->write_mu);
    if (job->client_disconnected.load(std::memory_order_acquire)) {
        return false;
    }
    // A heartbeat may have made a short write on the previous monitor tick.
    // Finish that comment before writing an SSE data frame so their bytes
    // cannot be interleaved on the stream.
    if (job->heartbeat_offset > 0) {
        constexpr size_t heartbeat_size = sizeof(kSseHeartbeat) - 1;
        if (!send_all(job->fd, kSseHeartbeat + job->heartbeat_offset,
                      heartbeat_size - job->heartbeat_offset)) {
            job->heartbeat_offset = 0;
            job->client_disconnected.store(true, std::memory_order_release);
            return false;
        }
        job->heartbeat_offset = 0;
    }
    if (!send_all(job->fd, data, len)) {
        job->client_disconnected.store(true, std::memory_order_release);
        return false;
    }
    job->last_stream_write = std::chrono::steady_clock::now();
    return true;
}

void HttpServer::start_job_stream(ServerJob * job) {
    std::lock_guard<std::mutex> lock(job->write_mu);
    if (job->client_disconnected.load(std::memory_order_acquire)) return;
    job->stream_ready = true;
    job->heartbeat_offset = 0;
    job->last_stream_write = std::chrono::steady_clock::now();
}

void HttpServer::stop_job_stream(
        ServerJob * job, ClientSendBuffer * pending_output) {
    std::lock_guard<std::mutex> lock(job->write_mu);
    job->stream_ready = false;
    if (pending_output && job->heartbeat_offset > 0) {
        constexpr size_t heartbeat_size = sizeof(kSseHeartbeat) - 1;
        pending_output->append(std::string_view(
            kSseHeartbeat + job->heartbeat_offset,
            heartbeat_size - job->heartbeat_offset));
        job->heartbeat_offset = 0;
    }
}

void HttpServer::maybe_send_job_heartbeat(
        ServerJob * job, bool peer_read_closed) {
    std::lock_guard<std::mutex> lock(job->write_mu);
    if (!job->stream_ready ||
        job->client_disconnected.load(std::memory_order_acquire)) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto heartbeat_interval =
        peer_read_closed ? kReadClosedProbeInterval : kSseHeartbeatInterval;
    if (now - job->last_stream_write < heartbeat_interval) {
        return;
    }

    // The scheduler also takes write_mu when it switches from prefill
    // heartbeats to buffered token output.  Never hold that shared mutex
    // across send_all()'s 30-second stall window.
    const auto result = http_detail::try_send_sse_heartbeat(
        job->fd, job->heartbeat_offset);
    if (result == http_detail::HeartbeatSendResult::Disconnected) {
        job->client_disconnected.store(true, std::memory_order_release);
        return;
    }
    if (result == http_detail::HeartbeatSendResult::Retry) return;
    job->last_stream_write = now;
}

std::string HttpServer::format_http_response(
        int status, const std::string & content_type,
        const std::string & body) {
    const char * reason = "OK";
    switch (status) {
        case 200: reason = "OK"; break;
        case 204: reason = "No Content"; break;
        case 400: reason = "Bad Request"; break;
        case 404: reason = "Not Found"; break;
        case 405: reason = "Method Not Allowed"; break;
        case 413: reason = "Payload Too Large"; break;
        case 500: reason = "Internal Server Error"; break;
        case 503: reason = "Service Unavailable"; break;
    }
    std::string header = "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\n";
    if (config_.enable_cors) {
        header += "Access-Control-Allow-Origin: *\r\n"
                  "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                  "Access-Control-Allow-Headers: *\r\n";
    }
    if (!content_type.empty()) {
        header += "Content-Type: " + content_type + "\r\n";
    }
    header += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    header += "Connection: close\r\n\r\n";
    header += body;
    return header;
}

bool HttpServer::send_response(
        SocketHandle fd, int status, const std::string & content_type,
        const std::string & body) {
    const std::string payload =
        format_http_response(status, content_type, body);
    if (status == 503 && stopping_.load()) {
        // Reject unstarted requests without delaying shutdown for a stalled
        // reader. Active native requests keep the normal send/drain path;
        // the socket closes when its handler returns.
        sock_set_nonblock(fd);
        ClientSendBuffer pending;
        pending.append(payload);
        return pending.flush(fd) && pending.empty();
    }
    return send_all(fd, payload.data(), payload.size());
}

bool HttpServer::send_error(
        SocketHandle fd, int status, const std::string & message) {
    json err = {{"error", {{"message", message}, {"type", "invalid_request_error"}}}};
    return send_response(fd, status, "application/json", err.dump() + "\n");
}

bool HttpServer::send_sse_headers(ServerJob * job) {
    std::string header = "HTTP/1.1 200 OK\r\n";
    if (config_.enable_cors) {
        header += "Access-Control-Allow-Origin: *\r\n";
    }
    header += "Content-Type: text/event-stream\r\n"
              "Cache-Control: no-cache\r\n"
              "Connection: keep-alive\r\n\r\n";
    return send_job_bytes(job, header.data(), header.size());
}

}  // namespace luce::common
