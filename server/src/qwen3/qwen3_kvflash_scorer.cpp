#include "qwen3_kvflash_scorer.h"

#include "qwen3_drafter_model.h"
#include "server/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace dflash::common {

namespace {

constexpr int kLookahead  = 8;
constexpr int kPoolKernel = 13;
constexpr int kMinSegment = 4096;

// Tail-attention token scores for `ids`: mean over the lookahead window of
// the drafter's running-max, then AvgPool smoothing. Same math as
// drafter_score_and_compress.
bool score_tokens_direct(DrafterContext & ctx, const std::vector<int32_t> & ids,
                         std::vector<float> & out) {
    const int S = (int)ids.size();
    std::vector<float> running_max;
    if (!forward_qwen3_drafter_model(ctx.weights, ids, kLookahead, running_max)) {
        return false;
    }
    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; j++) {
        float s = 0.0f;
        for (int t = 0; t < kLookahead; t++) s += running_max[(size_t)t * S + j];
        score[j] = s / kLookahead;
    }
    out.assign((size_t)S, 0.0f);
    const int half = kPoolKernel / 2;
    for (int j = 0; j < S; j++) {
        const int lo = std::max(0, j - half), hi = std::min(S - 1, j + half);
        float s = 0.0f;
        for (int k = lo; k <= hi; k++) s += score[k];
        out[j] = s / (hi - lo + 1);
    }
    return true;
}

void z_normalize(float * v, size_t n) {
    if (n == 0) return;
    double mean = 0;
    for (size_t i = 0; i < n; i++) mean += v[i];
    mean /= n;
    double var = 0;
    for (size_t i = 0; i < n; i++) var += (v[i] - mean) * (v[i] - mean);
    const float inv = 1.0f / ((float)std::sqrt(var / n) + 1e-6f);
    for (size_t i = 0; i < n; i++) v[i] = (float)((v[i] - mean) * inv);
}

// Score `ids` with allocation-failure resilience: try the full forward;
// on failure split into two equal halves, score each with the TRUE query
// tail (last kLookahead ids) appended so relevance stays query-aware, and
// z-normalize per segment so the merged ranking is comparable. Recursion
// floor kMinSegment. The drafter's per-call buffers (~10 KB/token) can
// fail on a fragmented CUDA heap at 32K+ even when total free VRAM is
// ample; segmented scoring trades exact cross-segment calibration for
// robustness.
bool score_tokens_resilient(DrafterContext & ctx, const std::vector<int32_t> & ids,
                            std::vector<float> & out) {
    if (score_tokens_direct(ctx, ids, out)) {
        z_normalize(out.data(), out.size());
        return true;
    }
    const int S = (int)ids.size();
    if (S <= kMinSegment) return false;

    std::fprintf(stderr, "[kvflash-scorer] forward failed at S=%d, bisecting\n", S);
    const int mid = S / 2;
    std::vector<int32_t> tail(ids.end() - kLookahead, ids.end());

    std::vector<int32_t> left(ids.begin(), ids.begin() + mid);
    left.insert(left.end(), tail.begin(), tail.end());
    std::vector<float> ls;
    if (!score_tokens_resilient(ctx, left, ls)) return false;

    std::vector<int32_t> right(ids.begin() + mid, ids.end());
    std::vector<float> rs;
    if (!score_tokens_resilient(ctx, right, rs)) return false;

    out.assign((size_t)S, 0.0f);
    std::copy(ls.begin(), ls.begin() + mid, out.begin());          // drop tail scores
    std::copy(rs.begin(), rs.begin() + (S - mid), out.begin() + mid);
    return true;
}

} // namespace

bool KvFlashDrafterScorer::score_chunks(const std::vector<int32_t> & ids,
                                   int chunk_tokens,
                                   std::vector<float> & out) {
    const int S = (int)ids.size();
    out.clear();
    if (!ctx_ || !ctx_->loaded || S < kLookahead + 1 || chunk_tokens <= 0) return false;

    std::vector<int32_t> score_ids = ids;
    if (vocab_clamp_ > 1001) {   // fold range must stay positive
        for (auto & t : score_ids) {
            if (t >= vocab_clamp_) t = 1000 + t % (vocab_clamp_ - 1000);
        }
    }

    std::vector<float> smooth;
    if (!score_tokens_resilient(*ctx_, score_ids, smooth)) return false;

    const int n_chunks = (S + chunk_tokens - 1) / chunk_tokens;
    out.assign((size_t)n_chunks, 0.0f);
    for (int c = 0; c < n_chunks; c++) {
        const int s_ = c * chunk_tokens, e_ = std::min(S, (c + 1) * chunk_tokens);
        float m = 0.0f;
        for (int j = s_; j < e_; j++) m += smooth[j];
        out[c] = m / std::max(1, e_ - s_);
    }
    return true;
}

// ── KvFlashCrossTokScorer ───────────────────────────────────────────────

struct KvFlashCrossTokScorer::Toks {
    Tokenizer target;
    Tokenizer drafter;
};

KvFlashCrossTokScorer::~KvFlashCrossTokScorer() { delete toks_; }

bool KvFlashCrossTokScorer::ensure_tokenizers() {
    if (toks_) return true;
    if (toks_failed_) return false;
    auto * t = new Toks();
    if (!t->target.load_from_gguf(target_gguf_.c_str()) ||
        !t->drafter.load_from_gguf(drafter_gguf_.c_str())) {
        std::fprintf(stderr, "[kvflash] cross-tokenizer scorer: tokenizer load "
                             "failed (%s / %s)\n",
                     target_gguf_.c_str(), drafter_gguf_.c_str());
        delete t;
        toks_failed_ = true;
        return false;
    }
    toks_ = t;
    return true;
}

bool KvFlashCrossTokScorer::score_chunks(const std::vector<int32_t> & ids,
                                         int chunk_tokens,
                                         std::vector<float> & out) {
    const int S = (int)ids.size();
    out.clear();
    if (!ctx_ || !ctx_->loaded || S < kLookahead + 1 || chunk_tokens <= 0) return false;
    if (!ensure_tokenizers()) return false;

    // 1) Target ids -> text, recording each target token's char end offset.
    //    Byte-level BPE pieces concatenate exactly, so per-id decode gives
    //    exact spans; special/template tokens may decode empty (their chunk
    //    contribution then comes from neighboring text, which is fine).
    std::string text;
    text.reserve((size_t)S * 4);
    std::vector<int32_t> tgt_end((size_t)S);
    std::vector<int32_t> one(1);
    for (int i = 0; i < S; i++) {
        one[0] = ids[(size_t)i];
        text += toks_->target.decode(one);
        tgt_end[(size_t)i] = (int32_t)text.size();
    }

    // 2) Text -> drafter ids, with each drafter token's char midpoint.
    const std::vector<int32_t> dids = toks_->drafter.encode(text);
    const int D = (int)dids.size();
    if (D < kLookahead + 1) return false;
    std::vector<float> dmid((size_t)D);
    {
        size_t pos = 0;
        for (int i = 0; i < D; i++) {
            one[0] = dids[(size_t)i];
            const size_t len = toks_->drafter.decode(one).size();
            dmid[(size_t)i] = (float)pos + (float)len * 0.5f;
            pos += len;
        }
    }

    // 3) Same tail-attention forward as the same-tokenizer scorer.
    std::vector<float> dscore;
    if (!score_tokens_resilient(*ctx_, dids, dscore)) return false;

    // 4) Map drafter-token scores onto target chunks by char span: a chunk's
    //    score is the mean of drafter tokens whose midpoint falls inside the
    //    chunk's text span. Empty spans (pure template tokens) stay at 0,
    //    i.e. z-score-neutral.
    const int n_chunks = (S + chunk_tokens - 1) / chunk_tokens;
    out.assign((size_t)n_chunks, 0.0f);
    std::vector<int> counts((size_t)n_chunks, 0);
    int d = 0;
    for (int c = 0; c < n_chunks; c++) {
        const int last_tok_idx = std::min(S, (c + 1) * chunk_tokens) - 1;
        const float span_end = (float)tgt_end[(size_t)last_tok_idx];
        while (d < D && dmid[(size_t)d] < span_end) {
            out[(size_t)c] += dscore[(size_t)d];
            counts[(size_t)c]++;
            d++;
        }
        if (counts[(size_t)c] > 0) out[(size_t)c] /= (float)counts[(size_t)c];
    }
    return true;
}

} // namespace dflash::common
