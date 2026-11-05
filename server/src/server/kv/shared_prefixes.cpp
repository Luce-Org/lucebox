#include "server/kv/shared_prefixes.h"

#include "tokenizer.h"

#include <algorithm>
#include <utility>

namespace luce::common {

bool parse_shared_prefix_mode(const std::string & value, SharedPrefixMode & out) {
    if (value == "auto" || value == "on") {
        out = SharedPrefixMode::learn;
    } else if (value == "freeze") {
        out = SharedPrefixMode::freeze;
    } else if (value == "off") {
        out = SharedPrefixMode::off;
    } else {
        return false;
    }
    return true;
}

SharedPrefixMode effective_shared_prefix_mode(bool given, SharedPrefixMode mode, bool concurrent) {
    return !given && concurrent && mode == SharedPrefixMode::learn ? SharedPrefixMode::freeze : mode;
}

const char * shared_prefix_mode_name(SharedPrefixMode mode) {
    switch (mode) {
        case SharedPrefixMode::off: return "off";
        case SharedPrefixMode::freeze: return "freeze";
        case SharedPrefixMode::learn: return "auto";
    }
    return "auto";
}

SharedPrefixes::SharedPrefixes(SharedPrefixMode mode, std::vector<bool> newline_ids,
                               int min_tokens, int ring_size)
    : mode_(mode), newline_ids_(std::move(newline_ids)),
      min_tokens_(std::max(1, min_tokens)), ring_size_(std::max(1, ring_size)) {}

bool SharedPrefixes::prefers_head(const std::vector<int> & boundaries,
                                  bool has_tools) const {
    if (!on()) return false;
    return has_tools || (!boundaries.empty() && boundaries.front() >= min_tokens_);
}

int SharedPrefixes::back_off_to_line_end(const std::vector<int32_t> & prompt, int n) const {
    for (int p = n; p >= min_tokens_; --p) {
        const int32_t id = prompt[(size_t)(p - 1)];
        if (id >= 0 && (size_t)id < newline_ids_.size() && newline_ids_[(size_t)id]) return p;
    }
    return n;
}

int SharedPrefixes::cut(const std::vector<int32_t> & prompt,
                        const std::vector<int> & boundaries) const {
    if (!on() || prompt.empty()) return 0;
    int common = 0;
    for (const auto & region : ring_) {
        const size_t n = std::min(region.size(), prompt.size());
        size_t i = 0;
        while (i < n && region[i] == prompt[i]) ++i;
        common = std::max(common, (int)i);
    }
    if (common < min_tokens_) return 0;
    int boundary = 0;
    for (int b : boundaries) {
        if (b > 0 && b <= common) boundary = std::max(boundary, b);
    }
    if (boundary >= min_tokens_) return boundary;
    return back_off_to_line_end(prompt, common);
}

int SharedPrefixes::head_cut(const std::vector<int32_t> & prompt,
                             const std::vector<int> & boundaries, bool has_tools) const {
    if (!on()) return -1;
    if (!prefers_head(boundaries, has_tools)) return 0;
    if (const int c = cut(prompt, boundaries)) return c;
    return !boundaries.empty() && boundaries.front() >= min_tokens_ ? boundaries.front() : 0;
}

void SharedPrefixes::remember(const std::vector<int32_t> & prompt,
                              const std::vector<int> & boundaries) {
    if (!learns() || prompt.empty()) return;
    const int n = boundaries.empty()
        ? (int)prompt.size()
        : std::min((int)prompt.size(), boundaries.front());
    if (n <= 0) return;
    ring_.emplace_back(prompt.begin(), prompt.begin() + n);
    while ((int)ring_.size() > ring_size_) ring_.erase(ring_.begin());
}

int SharedPrefixes::recurrences(const std::vector<int32_t> & prompt, int cut) const {
    if (cut <= 0 || cut > (int)prompt.size()) return 0;
    int n = 0;
    for (const auto & region : ring_) {
        if ((int)region.size() >= cut &&
            std::equal(region.begin(), region.begin() + cut, prompt.begin())) {
            ++n;
        }
    }
    return n;
}

void SharedPrefixes::set_ring(std::vector<std::vector<int32_t>> ring) {
    ring_ = std::move(ring);
    while ((int)ring_.size() > ring_size_) ring_.erase(ring_.begin());
}

std::vector<bool> newline_token_ids(const Tokenizer & tokenizer) {
    std::vector<bool> out((size_t)std::max(0, tokenizer.vocab_size()), false);
    for (int32_t id = 0; id < tokenizer.vocab_size(); ++id) {
        const std::string text = tokenizer.token_text(id);
        out[(size_t)id] = !text.empty() && text.back() == '\n';
    }
    return out;
}

}  // namespace luce::common
