#pragma once
// [TAG_ADAPTIVE_WIDTH] Per-step speculative verify-width selection.
//
// On MoE targets every verify row routes to its own top-k experts, so each
// row costs real weight bandwidth (~1ms/row for a Q4 target on a 24GB
// Ampere). A fixed verify width pays that cost even on steps where the
// drafter is unsure and the deep slots almost never commit. This helper trims
// the verify batch per step using the drafter's own confidence: keep
// candidate slot j while the drafter's estimate of the chain surviving
// through j (the product of the top-1 probabilities of slots 1..j) stays
// above theta. Output-exactness is preserved: only the number of speculated
// slots changes, every committed token is still target-verified.
//
// On by default (theta 0.20). Debug overrides:
//   DFLASH_ADAPTIVE_WIDTH_THETA=<0..1>  0 disables (legacy fixed/EWMA width)
//   DFLASH_ADAPTIVE_WIDTH_MIN=<n>       minimum kept rows incl. seed (default 4)
//
// Model-agnostic: any family loop that has per-slot drafter top-1
// probabilities (e.g. from ggml_backend_cuda_topk_rows over the draft-head
// logits) can call this before building its verify batch.
#include <cstdio>
#include <cstdlib>

inline float adaptive_verify_width_theta() {
    static const float theta = []() {
        const char * e = std::getenv("DFLASH_ADAPTIVE_WIDTH_THETA");
        if (!e) return 0.20f;
        char * end = nullptr;
        const float v = std::strtof(e, &end);
        if (end == e || *end != '\0' || v < 0.0f || v > 1.0f) {
            std::fprintf(stderr, "[adaptive-width] ignoring "
                                 "DFLASH_ADAPTIVE_WIDTH_THETA=\"%s\" "
                                 "(want a float in [0,1]); using 0.20\n", e);
            return 0.20f;
        }
        return v;
    }();
    return theta;
}

inline int adaptive_verify_width_min() {
    static const int mn = []() {
        const char * e = std::getenv("DFLASH_ADAPTIVE_WIDTH_MIN");
        if (!e) return 4;
        const int v = std::atoi(e);
        if (v <= 0) {
            std::fprintf(stderr, "[adaptive-width] ignoring "
                                 "DFLASH_ADAPTIVE_WIDTH_MIN=\"%s\" "
                                 "(want a positive int); using 4\n", e);
            return 4;
        }
        return v;
    }();
    return mn;
}

// top1_probs[(j-1)*stride]: drafter top-1 probability of candidate slot j.
// Returns the number of rows to verify (seed row 0 included), in
// [min(min_rows, n_rows), n_rows].
inline int adaptive_verify_width(const float * top1_probs, int stride,
                                 int n_rows, float theta, int min_rows) {
    if (theta <= 0.0f || top1_probs == nullptr || n_rows <= 2) {
        return n_rows;
    }
    float reach = 1.0f;
    int w = 1;
    for (int j = 1; j < n_rows; ++j) {
        reach *= top1_probs[(size_t)(j - 1) * (size_t)stride];
        if (w >= min_rows && reach < theta) {
            break;
        }
        ++w;
    }
    return w;
}
