// Shared CPU sampler chain. See sampler.h for the protocol overview.

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
#include "geometric_sampler_cuda.h"
#endif

namespace dflash::common {

namespace {

// Binary search (quickselect via std::nth_element) for the smallest index
// `cut` such that the descending-by-value prefix cand[0,cut) has cumulative
// mass >= target, where "mass" of an element is given by `mass_of`. At each
// level, partitioning [lo,hi) at its midpoint puts exactly the top (mid-lo)
// elements of that range into [lo,mid) (in some order) — whichever half still
// contains the boundary is recursed into and the other is discarded. Mutates
// `cand` in place; only the final small base-case range ends up sorted, since
// order doesn't matter for the caller's draw, only which elements make the
// cut. Each level's cost is proportional to its (shrinking) range, so total
// work is O(cand.size()), not O(cand.size() log cand.size()) like a full sort,
// regardless of where the cutoff lands.
template <typename MassFn>
size_t nucleus_cutoff(std::vector<std::pair<float, int>> & cand, double target, MassFn mass_of) {
    constexpr size_t kBaseCase = 64;
    size_t lo = 0, hi = cand.size();
    while (hi - lo > kBaseCase) {
        const size_t mid = lo + (hi - lo) / 2;
        std::nth_element(cand.begin() + lo, cand.begin() + mid, cand.begin() + hi,
                         [](auto & a, auto & b){ return a.first > b.first; });
        double mass = 0.0;
        for (size_t i = lo; i < mid; i++) mass += mass_of(cand[i]);
        if (mass >= target) {
            hi = mid;        // cutoff lies within [lo, mid)
        } else {
            target -= mass;  // [lo, mid) fully included; keep searching [mid, hi)
            lo = mid;
        }
    }
    // Base case: small enough range left, sort it and walk the exact cumulative
    // cutoff (cand[0, lo) is already fully confirmed included from above).
    std::sort(cand.begin() + lo, cand.begin() + hi,
             [](auto & a, auto & b){ return a.first > b.first; });
    size_t cut = hi;
    double cum = 0.0;
    for (size_t i = lo; i < hi; i++) {
        cum += mass_of(cand[i]);
        if (cum >= target) { cut = i + 1; break; }
    }
    return cut;
}

// Draws a token from `cand`, whose .first fields are proportional
// probabilities (need not already sum to 1 or be sorted). `r_uniform` is a
// pre-drawn uniform in [0,1) supplied by the caller (drawn once per
// sample_logits call) so every path — GPU, GPU-assisted top_p, or CPU —
// consumes the same single RNG value.
int draw_from_weights(const std::vector<std::pair<float, int>> & cand, double r_uniform) {
    double Z = 0.0;
    for (auto & c : cand) Z += c.first;
    const double r = r_uniform * Z;
    double acc = 0.0;
    for (auto & c : cand) {
        acc += c.first;
        if (r <= acc) return c.second;
    }
    return cand.back().second;
}

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
// Given probabilities the GPU already computed (penalties + softmax(temp)
// applied, summing to ~1) for a pure top_p (no top_k) config, find the
// nucleus and draw. Skips all exp()/Z bookkeeping the raw-logit path needs,
// since the input is already normalized.
//
// Only worth calling for top_p without top_k: top_k's CPU cost is already
// cheap (partial_sort scales with k, not vocab — measured ~270-300us at
// vocab=151936), so a GPU round trip (kernel + D2H copy, ~500-800us) makes it
// slower, not faster. top_p's CPU cost without top_k is dominated by
// nucleus_cutoff's O(vocab) std::nth_element passes regardless of who
// computed the input probabilities, so skipping the CPU-side exp() pass here
// is a net win (measured ~1.4x faster end-to-end at vocab=151936).
int sample_from_gpu_probs(std::vector<float> & probs, double top_p, double r_uniform) {
    std::vector<std::pair<float, int>> cand(probs.size());
    for (size_t i = 0; i < probs.size(); i++) cand[i] = {probs[i], (int)i};

    double Z = 0.0;
    for (auto & c : cand) Z += c.first;
    const double target = top_p * Z;
    const size_t cut = nucleus_cutoff(cand, target, [](auto & c){ return (double)c.first; });
    cand.resize(cut);

    return draw_from_weights(cand, r_uniform);
}
#endif

}  // namespace

int sample_logits(const float * logits_in,
                  int vocab,
                  const SamplerCfg & cfg,
                  const std::vector<int32_t> & history,
                  std::mt19937_64 & rng) {
    // Draw the single uniform up front, exactly once, and only when we will
    // actually sample (temp>0; greedy returns before any draw). It is then
    // threaded through every path below — the full-GPU draw, the GPU-assisted
    // top_p draw, and the CPU draw all consume this same value — so the RNG
    // stream advances identically no matter which path resolves the token
    // (including a GPU→CPU fallback), keeping decode reproducible whether the
    // GPU sampler is on or off.
    double r_uniform = 0.0;
    if (cfg.temp > 0.0f) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        r_uniform = u(rng);
    }

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
    // GPU path (on by default; set DFLASH_GPU_SAMPLE=0 to disable). top_k>0,
    // top_p in (0,1) (both unsupported on the GPU, see geometric_sampler_cuda.h),
    // and any CUDA error return -1 and fall through to the CPU chain below.
    if (gpu_sampler_enabled() && gpu_sampler_supports(cfg)) {
        const int g = geometric_sample_logits_cuda(logits_in, vocab, cfg, history, r_uniform,
                                         /*logits_on_device=*/false);
        if (g >= 0) return g;
    }
#endif
    const bool need_top_k = cfg.top_k > 0 && cfg.top_k < vocab;
    const bool need_top_p = cfg.top_p > 0.0f && cfg.top_p < 1.0f;

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
    // Reaching here means the GPU either can't fully handle this config
    // (top_k/top_p above) or is disabled. For pure top_p (no top_k), the GPU
    // can still compute the shared, vocab-wide penalty+softmax prefix and
    // hand back normalized probabilities, letting the CPU skip straight to
    // nucleus_cutoff — worth it because that search's O(vocab) cost dominates
    // regardless of who computed its input. top_k (with or without top_p) is
    // deliberately excluded: its CPU cost is already cheap (partial_sort
    // scales with k, not vocab), so the GPU round trip would make it slower,
    // not faster (measured regression, not just "no win").
    if (cfg.temp > 0.0f && need_top_p && !need_top_k && gpu_sampler_enabled()) {
        std::vector<float> gpu_probs(vocab);
        if (geometric_compute_probs_cuda(logits_in, vocab, cfg, history,
                                         gpu_probs.data(), /*logits_on_device=*/false)) {
            return sample_from_gpu_probs(gpu_probs, cfg.top_p, r_uniform);
        }
        // else: fall through to the full CPU chain below.
    }
#endif

    std::vector<std::pair<float, int>> cand(vocab);
    for (int i = 0; i < vocab; i++) cand[i] = {logits_in[i], i};

    // Multiplicative repetition penalty (HuggingFace-style).
    if (cfg.rep_pen > 1.0f && !history.empty()) {
        const int win  = std::min((int)history.size(), cfg.rep_window);
        const int from = (int)history.size() - win;
        std::unordered_set<int> seen;
        for (int i = from; i < (int)history.size(); i++) seen.insert(history[i]);
        for (auto & c : cand) {
            if (seen.count(c.second)) {
                c.first = (c.first > 0.0f) ? c.first / cfg.rep_pen
                                           : c.first * cfg.rep_pen;
            }
        }
    }

    // OpenAI-style additive frequency and presence penalties.
    if ((cfg.freq_pen != 0.0f || cfg.pres_pen != 0.0f) && !history.empty()) {
        const int win  = std::min((int)history.size(), cfg.rep_window);
        const int from = (int)history.size() - win;
        std::unordered_map<int, int> counts;
        for (int i = from; i < (int)history.size(); i++) counts[history[i]]++;
        for (auto & c : cand) {
            auto it = counts.find(c.second);
            if (it != counts.end()) {
                c.first -= cfg.freq_pen * it->second;
                c.first -= cfg.pres_pen;
            }
        }
    }

    // temp=0 → deterministic argmax (after penalties have been applied above).
    // Independent of top_k/top_p (the single highest-logit token is always the
    // answer), so this skips sorting/truncation entirely: an O(vocab) max scan
    // beats the O(vocab log vocab) sort below. Ties go to the lowest token id
    // (max_element returns the first of equal maxima), matching the GPU kernel.
    if (cfg.temp <= 0.0f) {
        return std::max_element(cand.begin(), cand.end(),
                                [](auto & a, auto & b){ return a.first < b.first; })->second;
    }

    // Only sort/truncate when top_k or top_p actually need it. Softmax and the
    // inverse-CDF draw below are order-independent, so the common case (plain
    // temperature sampling, no truncation) skips sorting entirely.
    if (need_top_k) {
        std::partial_sort(cand.begin(), cand.begin() + cfg.top_k, cand.end(),
                          [](auto & a, auto & b){ return a.first > b.first; });
        cand.resize(cfg.top_k);
    }

    const float inv_t = 1.0f / std::max(1e-3f, cfg.temp);
    const float maxv_logit = need_top_k
        ? cand.front().first
        : std::max_element(cand.begin(), cand.end(),
                           [](auto & a, auto & b){ return a.first < b.first; })->first;
    const float maxv = maxv_logit * inv_t;

    if (need_top_p && !need_top_k) {
        // Nucleus cutoff over the full (untruncated) vocab. Z is the true
        // full-vocab softmax denominator (one O(vocab) exp() pass, needed
        // regardless to know the absolute mass threshold).
        double Z = 0.0;
        for (auto & c : cand) Z += std::exp((double)c.first * inv_t - maxv);
        const double target = (double)cfg.top_p * Z;
        const size_t cut = nucleus_cutoff(cand, target, [&](auto & c) {
            return std::exp((double)c.first * inv_t - maxv);
        });
        cand.resize(cut);
    }

    double Z = 0.0;
    std::vector<float> probs(cand.size());
    for (size_t i = 0; i < cand.size(); i++) {
        probs[i] = std::exp(cand[i].first * inv_t - maxv);
        Z       += probs[i];
    }
    for (auto & p : probs) p = (float)(p / Z);

    // top_k+top_p combined: cut the already top_k-truncated (and thus
    // already-sorted) subset further to the top_p nucleus within it.
    if (need_top_p && need_top_k) {
        double cum = 0.0;
        size_t cut = probs.size();
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (cum >= cfg.top_p) { cut = i + 1; break; }
        }
        probs.resize(cut); cand.resize(cut);
    }

    // Draw from the final candidate set. Same CDF walk as draw_from_weights
    // above (it renormalizes internally, which is a no-op cost here since
    // probs already sums to ~1) — reuse it instead of re-deriving the same
    // loop a second time.
    for (size_t i = 0; i < cand.size(); i++) cand[i].first = probs[i];
    return draw_from_weights(cand, r_uniform);
}

bool parse_sampler_token(std::string & line, SamplerCfg & out) {
    auto pos = line.find(" samp=");
    if (pos == std::string::npos) return false;
    auto end = line.find(' ', pos + 1);
    std::string tok = (end == std::string::npos)
                          ? line.substr(pos + 6)
                          : line.substr(pos + 6, end - (pos + 6));
    line.erase(pos, (end == std::string::npos ? std::string::npos : end - pos));
    float t = 0.0f, tp = 1.0f, rp = 1.0f, fp = 0.0f, pp = 0.0f;
    int   tk = 0;
    unsigned long long sd = 0;
    int n = std::sscanf(tok.c_str(), "%f,%f,%d,%f,%llu,%f,%f",
                        &t, &tp, &tk, &rp, &sd, &fp, &pp);
    if (n < 1) return false;
    out.temp     = t;
    out.top_p    = tp;
    out.top_k    = tk;
    out.rep_pen  = rp;
    out.seed     = sd;
    out.freq_pen = fp;
    out.pres_pen = pp;
    return true;
}

}  // namespace dflash::common
