#include "output_constraint.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <utility>

namespace luce::common {

std::shared_ptr<const ConstraintVocab> ConstraintVocab::build(
        std::vector<std::string> pieces, const std::vector<int32_t> & end_tokens) {
    auto vocab = std::make_shared<ConstraintVocab>();
    const size_t n = pieces.size();
    vocab->code_points.resize(n);
    vocab->partials.resize(n);
    vocab->is_end.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        // The matcher reads text up to a NUL: a piece holding one (the byte
        // token for 0x00) would match as if it were shorter, so it gets no
        // text at all and the grammar never allows it.
        if (pieces[i].find('\0') != std::string::npos) pieces[i].clear();
        auto decoded = gbnf::decode_utf8(pieces[i], {});
        vocab->code_points[i] = std::move(decoded.first);
        vocab->partials[i] = decoded.second;
    }
    for (int32_t token : end_tokens) {
        if (token >= 0 && (size_t) token < n) vocab->is_end[(size_t) token] = 1;
    }
    vocab->pieces = std::move(pieces);
    return vocab;
}

OutputConstraint::OutputConstraint(std::shared_ptr<const ConstraintVocab> vocab,
                                   std::unique_ptr<gbnf::Matcher> grammar,
                                   Options options)
    : vocab_(std::move(vocab)),
      initial_(std::move(grammar)),
      options_(std::move(options)) {
    reset();
}

OutputConstraint::OutputConstraint(const OutputConstraint & other)
    : vocab_(other.vocab_),
      initial_(other.initial_),
      grammar_(std::make_unique<gbnf::Matcher>(*other.grammar_)),
      options_(other.options_),
      state_(other.state_),
      tail_(other.tail_),
      head_(other.head_),
      release_window_(other.release_window_),
      started_(other.started_) {}

std::shared_ptr<OutputConstraint> OutputConstraint::snapshot() const {
    return std::shared_ptr<OutputConstraint>(new OutputConstraint(*this));
}

OutputConstraint::State OutputConstraint::initial_state() const {
    return options_.wait_for.empty() ? State::engaged : State::waiting;
}

void OutputConstraint::reset() {
    grammar_ = std::make_unique<gbnf::Matcher>(*initial_);
    state_ = initial_state();
    tail_.clear();
    head_.clear();
    started_ = false;
    release_window_ = 0;
    for (const auto & prefix : options_.release_prefixes) {
        release_window_ = std::max(release_window_, prefix.size());
    }
}

bool OutputConstraint::allows(int32_t token) const {
    if (!engaged()) return true;
    if (token < 0 || token >= vocab_->size()) return false;
    if (vocab_->is_end[(size_t) token]) return grammar_->can_end();
    const std::string & piece = vocab_->pieces[(size_t) token];
    if (piece.empty()) return false;
    const gbnf::PartialUtf8 partial = grammar_->partial();
    if (partial.n_remain == 0) {
        return grammar_->reject({{ 0, vocab_->code_points[(size_t) token].data(),
                                   vocab_->partials[(size_t) token] }}).empty();
    }
    const auto decoded = gbnf::decode_utf8(piece, partial);
    return grammar_->reject({{ 0, decoded.first.data(), decoded.second }}).empty();
}

int OutputConstraint::mask(float * logits, int n_vocab, bool content_only) const {
    if (!engaged()) return n_vocab;
    const bool can_end = grammar_->can_end();
    const gbnf::PartialUtf8 partial = grammar_->partial();
    const bool from_boundary = partial.n_remain == 0;
    const int n_known = std::min(n_vocab, vocab_->size());

    // Tokens past the tokenizer's vocabulary (padding rows) never match.
    for (int t = n_known; t < n_vocab; ++t) logits[t] = -INFINITY;

    thread_local std::vector<gbnf::Candidate> candidates;
    thread_local std::vector<std::pair<std::vector<uint32_t>, gbnf::PartialUtf8>> decoded;
    candidates.clear();
    decoded.clear();
    if (!from_boundary) decoded.reserve((size_t) n_known);
    for (int t = 0; t < n_known; ++t) {
        if (logits[t] == -INFINITY) continue;
        if (vocab_->is_end[(size_t) t]) {
            if (!can_end) logits[t] = -INFINITY;
            continue;
        }
        const std::string & piece = vocab_->pieces[(size_t) t];
        if (piece.empty()) {
            logits[t] = -INFINITY;
            continue;
        }
        if (from_boundary) {
            candidates.push_back({ (size_t) t, vocab_->code_points[(size_t) t].data(),
                                   vocab_->partials[(size_t) t] });
        } else {
            decoded.push_back(gbnf::decode_utf8(piece, partial));
            candidates.push_back({ (size_t) t, decoded.back().first.data(),
                                   decoded.back().second });
        }
    }
    for (const auto & reject : grammar_->reject(candidates)) {
        logits[reject.index] = -INFINITY;
    }
    if (content_only && !started_) {
        const bool keep_calls = options_.calls_only;
        for (const auto & candidate : candidates) {
            const size_t t = candidate.index;
            if (logits[t] == -INFINITY) continue;
            const std::string & piece = vocab_->pieces[t];
            if (piece.find_first_not_of(" \t\r\n") == std::string::npos ||
                (!keep_calls && starts_release(piece))) {
                logits[t] = -INFINITY;
            }
        }
    }
    int allowed = 0;
    for (int t = 0; t < n_known; ++t) {
        if (logits[t] != -INFINITY) ++allowed;
    }
    return allowed;
}

void OutputConstraint::accept(int32_t token) {
    if (state_ == State::released || state_ == State::broken) return;
    const bool known = token >= 0 && token < vocab_->size();
    // An end token ends the output; nothing follows it to constrain.
    if (known && vocab_->is_end[(size_t) token]) return;
    static const std::string no_text;
    const std::string & piece = known ? vocab_->pieces[(size_t) token] : no_text;
    if (state_ == State::waiting) {
        if (!known) return;
        tail_ += piece;
        const size_t at = tail_.find(options_.wait_for);
        if (at == std::string::npos) {
            // Keep enough to see a marker split across tokens.
            if (tail_.size() > options_.wait_for.size()) {
                tail_.erase(0, tail_.size() - options_.wait_for.size());
            }
            return;
        }
        const std::string rest = tail_.substr(at + options_.wait_for.size());
        tail_.clear();
        state_ = State::engaged;
        if (!rest.empty()) feed(rest, token);
        return;
    }
    if (!known || piece.empty()) {
        std::fprintf(stderr,
            "[constraint] token %d has no text; output is no longer constrained\n",
            token);
        state_ = State::broken;
        return;
    }
    feed(piece, token);
}

void OutputConstraint::feed(const std::string & text, int32_t token) {
    grammar_->accept_bytes(text);
    if (grammar_->dead()) {
        // Only a token injected past the sampler (a forced close, a hint)
        // gets here: let the request finish unconstrained rather than leave
        // it without a legal token.
        std::fprintf(stderr,
            "[constraint] token %d does not follow the grammar; output is no "
            "longer constrained\n", token);
        state_ = State::broken;
        return;
    }
    if (!started_ && text.find_first_not_of(" \t\r\n") != std::string::npos) started_ = true;
    check_release(text);
}

bool OutputConstraint::starts_release(const std::string & piece) const {
    const std::string text = head_ + piece;
    const size_t start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return false;
    const std::string_view output = std::string_view(text).substr(start);
    for (const auto & prefix : options_.release_prefixes) {
        const size_t n = std::min(output.size(), prefix.size());
        if (output.compare(0, n, prefix, 0, n) == 0) return true;
    }
    return false;
}

void OutputConstraint::check_release(const std::string & text) {
    if (release_window_ == 0) return;
    head_ += text;
    const size_t start = head_.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return;
    const std::string_view output = std::string_view(head_).substr(start);
    for (const auto & prefix : options_.release_prefixes) {
        if (output.size() >= prefix.size() && output.compare(0, prefix.size(), prefix) == 0) {
            state_ = State::released;
            head_.clear();
            return;
        }
    }
    if (output.size() >= release_window_) {
        // Past every opener without matching one: the output is the
        // constrained text, not a tool call.
        release_window_ = 0;
        head_.clear();
    }
}

}  // namespace luce::common
