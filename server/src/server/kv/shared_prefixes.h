// Shared prefixes: what a client repeats at the start of every request (its
// tool schemas and system prompt, up to the first user turn, or up to where
// its projects differ).
//
// A request whose prefix before the first user turn is long enough captures
// that prefix (its first chat boundary) until the entry is restored, so the
// next conversation on it restores it. When requests from different
// conversations agree on a longer or shorter prefix than that boundary, the
// capture is cut where they agree instead: the cut. It is found against the
// prefix regions of the last few requests (the ring), at a chat boundary when
// one is far enough in, else mid-message, backed off to the end of the last
// line so a BPE merge at the divergence stays out of the key.
//
// --shared-prefix auto learns (the ring follows traffic), freeze keeps the
// known cuts still (benchmarks, digest tests), off is plain prefix caching.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

enum class SharedPrefixMode { off, freeze, learn };

bool parse_shared_prefix_mode(const std::string & value, SharedPrefixMode & out);
const char * shared_prefix_mode_name(SharedPrefixMode mode);

// The mode a server runs: `auto` that was not typed becomes `freeze` on the
// concurrent path, whose engines do not chunk at plan.restore_points yet, so
// a restored prefix would not match a cold run.
SharedPrefixMode effective_shared_prefix_mode(bool given, SharedPrefixMode mode, bool concurrent);

class SharedPrefixes {
public:
    // A prefix shorter than this is not worth a snapshot: on hybrid Qwen a
    // snapshot holds hundreds of MiB of recurrent state, and 512 tokens
    // prefill in well under a tenth of a second.
    static constexpr int kMinTokens = 512;
    // Prefix regions kept in the ring.
    static constexpr int kRingSize = 8;

    // `newline_ids[id]`: the token's text ends with a newline (see
    // newline_token_ids).
    SharedPrefixes(SharedPrefixMode mode, std::vector<bool> newline_ids,
                   int min_tokens = kMinTokens, int ring_size = kRingSize);

    SharedPrefixMode mode() const { return mode_; }
    bool on() const { return mode_ != SharedPrefixMode::off; }
    // The ring follows traffic (not in freeze).
    bool learns() const { return mode_ == SharedPrefixMode::learn; }
    int min_tokens() const { return min_tokens_; }
    int ring_size() const { return ring_size_; }

    // Whether a request captures its system/tools head first: a request
    // with tools (as always), or one whose prefix before the first user turn
    // (`boundaries`' first) is at least min_tokens long. Never when off.
    bool prefers_head(const std::vector<int> & boundaries, bool has_tools) const;

    // Where `prompt` agrees with the ring: the longest prefix it shares with
    // a recent region, cut at the deepest chat boundary inside it when that
    // is at least min_tokens in, else mid-message, backed off to the end of
    // its last line. 0 when it shares less than min_tokens.
    int cut(const std::vector<int32_t> & prompt, const std::vector<int> & boundaries) const;

    // The head an engine that plans no prefix store captures: the ring's cut,
    // else the first chat boundary when it is min_tokens in (the first request
    // has no ring yet). -1 when off (the engine uses its own role end), 0 when
    // no head applies.
    int head_cut(const std::vector<int32_t> & prompt, const std::vector<int> & boundaries,
                 bool has_tools) const;

    // Remember a request's prefix region: `prompt` up to its first chat
    // boundary (`boundaries`), the whole prompt without one. Only while it
    // learns. Ask cut() first: a region always agrees with itself.
    void remember(const std::vector<int32_t> & prompt, const std::vector<int> & boundaries);
    // How many ring regions share `prompt`'s first `cut` tokens: a cut inside
    // a message is pinned only once it recurs (kPinRecurrences).
    static constexpr int kPinRecurrences = 3;
    int recurrences(const std::vector<int32_t> & prompt, int cut) const;

    // The ring, oldest first (saved with the pins; DiffPin diffs against it).
    const std::vector<std::vector<int32_t>> & ring() const { return ring_; }
    void set_ring(std::vector<std::vector<int32_t>> ring);

private:
    // `n` moved back to just past the last newline token before it; `n`
    // itself when there is none at or past min_tokens.
    int back_off_to_line_end(const std::vector<int32_t> & prompt, int n) const;

    SharedPrefixMode mode_;
    std::vector<bool> newline_ids_;
    int min_tokens_;
    int ring_size_;
    std::vector<std::vector<int32_t>> ring_;
};

class Tokenizer;
// Which ids of `tokenizer`'s vocabulary decode to text ending in a newline.
std::vector<bool> newline_token_ids(const Tokenizer & tokenizer);

}  // namespace luce::common
