// KvFlashDrafterScorer — pflash drafter as the KV pager's Memory Indexer.
//
// Scores 64-token chunks with the same Liu Q-hook tail attention that
// pflash compression uses (forward_qwen3_drafter_model), but returns the
// per-chunk relevance scores instead of a compressed token list. The
// DrafterContext is borrowed: the daemon shares its pflash drafter; the
// pager itself never depends on this file (see common/kvflash_scorer.h).

#pragma once

#include "kvflash_scorer.h"
#include "qwen3_drafter.h"

#include <string>

namespace dflash::common {

class KvFlashDrafterScorer : public KvFlashScorer {
public:
    // `vocab_clamp`: ids >= clamp are folded into the drafter's vocab range
    // before scoring. Needed when the target vocabulary is a superset of
    // the drafter's (e.g. Qwen3.6 target + Qwen3-0.6B drafter); prompt ids
    // tokenized for the target may be unembeddable by the drafter.
    explicit KvFlashDrafterScorer(DrafterContext * ctx, int32_t vocab_clamp = 100000)
        : ctx_(ctx), vocab_clamp_(vocab_clamp) {}

    bool score_chunks(const std::vector<int32_t> & ids, int chunk_tokens,
                      std::vector<float> & out) override;

private:
    DrafterContext * ctx_;
    int32_t vocab_clamp_;
};

// KvFlashCrossTokScorer — the same drafter scoring for targets that do NOT
// share the Qwen tokenizer (laguna, gemma4). Relevance is a property of the
// TEXT, so the bridge is re-tokenization: detokenize the target's history
// (its own tokenizer, loaded from the target GGUF), tokenize the text with
// the drafter's tokenizer (from the drafter GGUF), run the same tail-
// attention forward, then map per-drafter-token scores back onto the
// target's chunk boundaries by character spans. Tokenizers are host-only
// and lazy-loaded on first score.
class KvFlashCrossTokScorer : public KvFlashScorer {
public:
    KvFlashCrossTokScorer(DrafterContext * ctx,
                          std::string target_gguf,
                          std::string drafter_gguf)
        : ctx_(ctx), target_gguf_(std::move(target_gguf)),
          drafter_gguf_(std::move(drafter_gguf)) {}
    ~KvFlashCrossTokScorer() override;
    KvFlashCrossTokScorer(const KvFlashCrossTokScorer &) = delete;
    KvFlashCrossTokScorer & operator=(const KvFlashCrossTokScorer &) = delete;

    bool score_chunks(const std::vector<int32_t> & ids, int chunk_tokens,
                      std::vector<float> & out) override;

private:
    bool ensure_tokenizers();

    DrafterContext * ctx_;
    std::string target_gguf_, drafter_gguf_;
    // Pimpl to keep server/tokenizer.h out of backend headers.
    struct Toks;
    Toks * toks_ = nullptr;
    bool   toks_failed_ = false;
};

} // namespace dflash::common
