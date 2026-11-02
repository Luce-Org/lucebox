// Constrained decoding: a request's output follows a grammar (a JSON schema
// from response_format, a forced tool call). Model-neutral.
//
// One OutputConstraint per request, shared through SamplerCfg::constraint:
//   - the sampler (sample_logits) draws the request's token as usual and, if
//     the grammar rejects it, masks every token the grammar rejects and draws
//     again from what is left (llama.cpp's grammar rejection sampling). It
//     costs one token check per step while the model already writes what the
//     grammar wants, and gives the masked distribution exactly for
//     temperature sampling (see sample_logits for top_k/top_p);
//   - the commit point advances it with every token the request keeps: the
//     scheduler (advance_slot) for paged engines, the token callback for the
//     single-request path.
// Both run on the request's worker thread. A speculative verify checks drafted
// tokens against a snapshot() it advances over them, ahead of the commit
// point; backends that do not (ModelBackend::speculates_under_constraint) and
// the scheduler decode a constrained request autoregressively.

#pragma once

#include "gbnf.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// Every token's text, decoded once per model and shared by all requests.
struct ConstraintVocab {
    std::vector<std::string> pieces;                 // decoded bytes per token id
    // pieces[i] decoded from a character boundary (0-terminated), and the
    // UTF-8 sequence it leaves open
    std::vector<std::vector<uint32_t>> code_points;
    std::vector<gbnf::PartialUtf8> partials;
    std::vector<uint8_t> is_end;                     // end-of-generation tokens

    static std::shared_ptr<const ConstraintVocab> build(
            std::vector<std::string> pieces, const std::vector<int32_t> & end_tokens);

    int size() const { return (int) pieces.size(); }
};

class OutputConstraint {
public:
    struct Options {
        // The request starts inside a thinking block: the constraint waits
        // until the output has closed it with this text (`</think>`); what
        // follows it in the same token is constrained.
        std::string wait_for;
        // Openers of the tool-call syntax. Once the output (after leading
        // whitespace) starts with one, the request is calling a tool and the
        // constraint lets go; the grammar admits them for that hand-over.
        std::vector<std::string> release_prefixes;
        // The grammar admits only a tool call (tool_choice "required"), so a
        // redraw never keeps to a reply there is none of.
        bool calls_only = false;
    };

    OutputConstraint(std::shared_ptr<const ConstraintVocab> vocab,
                     std::unique_ptr<gbnf::Matcher> grammar,
                     Options options);

    // Constraining the next token: not waiting, released, finished or broken.
    bool engaged() const { return state_ == State::engaged; }
    bool released() const { return state_ == State::released; }
    bool broken() const { return state_ == State::broken; }
    // The output so far is a complete sentence of the grammar.
    bool complete() const { return grammar_ && grammar_->can_end(); }

    bool allows(int32_t token) const;
    // Set every token the grammar rejects to -inf; returns how many remain.
    // `content_only` is for a redraw after the request drew a token the
    // grammar rejects (a code fence, a sentence): it meant to start writing.
    // Before the output has started it also drops blank tokens, which would
    // only postpone the choice, and the tokens that would start a tool call
    // when the grammar admits a reply.
    int mask(float * logits, int n_vocab, bool content_only = false) const;
    // Advance over a token the request kept. A token the grammar rejects
    // (one injected past the sampler) breaks the constraint: it stops
    // constraining instead of leaving the request with no legal token.
    void accept(int32_t token);
    // Back to the start, for a request that decodes again from its prompt.
    void reset();
    // A copy at the current position, to advance over tokens the request has
    // not kept yet (a speculative verify's drafts).
    std::shared_ptr<OutputConstraint> snapshot() const;

private:
    enum class State { waiting, engaged, released, broken };

    OutputConstraint(const OutputConstraint & other);

    State initial_state() const;
    void feed(const std::string & text, int32_t token);
    void check_release(const std::string & text);
    // The output with `piece` appended would begin a tool-call opener.
    bool starts_release(const std::string & piece) const;

    std::shared_ptr<const ConstraintVocab> vocab_;
    std::shared_ptr<const gbnf::Matcher> initial_;
    std::unique_ptr<gbnf::Matcher> grammar_;
    Options options_;
    State state_ = State::engaged;
    std::string tail_;               // output tail while waiting
    std::string head_;               // output start, for release detection
    size_t release_window_ = 0;      // longest release prefix
    bool started_ = false;           // the output has more than whitespace
};

}  // namespace luce::common
