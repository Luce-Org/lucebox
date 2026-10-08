#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <vector>

namespace luce::common {

constexpr int QWEN4EXP_MTP_MAX_DRAFT = 7;
constexpr int QWEN4EXP_MTP_MAX_VERIFY = QWEN4EXP_MTP_MAX_DRAFT + 1;

// Verify cycle cost = fixed overhead + per-row cost, fitted online. The cost
// is the device's, not the text's: the backend keeps one across requests and
// each request starts from it (Qwen4ExpMtpWidth::cost()).
struct Qwen4ExpMtpCost {
    double base = 45.0, slope = 20.0;
    // Ridge prior: 16 synthetic cycles at each end of k=1..7, forgotten
    // slowly as measurements arrive; all widths share it.
    double sw = 32.0, sx = 128.0, sxx = 800.0, sy = 32.0 * 45.0 + 128.0 * 20.0, sxy = 128.0 * 45.0 + 800.0 * 20.0;
    std::array<int, QWEN4EXP_MTP_MAX_VERIFY> samples{};   // per width; the first few are cold graph builds
    // Measured cycle time per width (warm samples): the policy uses it once a
    // width has a few, and the linear fit only for widths not measured yet. A
    // fit alone stays wrong where the policy never goes (its slope is
    // unidentifiable from one width), which kept a bad prior in charge.
    std::array<double, QWEN4EXP_MTP_MAX_VERIFY> measured{};
    std::array<int, QWEN4EXP_MTP_MAX_VERIFY> measured_n{};

    double at(int k) const { return measured_n[k] >= 4 ? measured[k] : base + slope * k; }
};

// MTP-only, per-request policy. Widths include the seed (k = width - 1).
// Learn conditional acceptance only at reached depths: a clean short draft
// says nothing about the next depth, and a rejection is not another failure
// of every deeper conditional. Products give monotone prefix survivals.
// Per-depth acceptance evidence: trials and successes of each reached depth.
struct Qwen4ExpMtpEvidence {
    std::array<double, QWEN4EXP_MTP_MAX_VERIFY> trials{}, successes{};
};

class Qwen4ExpMtpWidth {
public:
    Qwen4ExpMtpWidth(int max_draft, bool adaptive, int prompt_tokens, const Qwen4ExpMtpCost * learned = nullptr,
                     const Qwen4ExpMtpEvidence * evidence = nullptr)
        : cap_(std::clamp(max_draft, 1, QWEN4EXP_MTP_MAX_DRAFT)), adaptive_(adaptive) {
        // Optimistic but finite prior: 16 trials at 90% per reached depth.
        trials_.fill(16.0);
        successes_.fill(16.0 * 0.90);
        if (evidence) {
            carried_ = true;
            // Earlier requests' acceptance, scaled to at most 32 trials per depth so this request's text takes
            // over within a few dozen cycles. A depth no request reached keeps the prior.
            for (int d = 1; d <= QWEN4EXP_MTP_MAX_DRAFT; ++d) {
                const double t = evidence->trials[d];
                if (t < 1.0) continue;
                const double scale = std::min(1.0, 32.0 / t);
                trials_[d] = t * scale;
                successes_[d] = evidence->successes[d] * scale;
            }
        }
        if (learned) {
            cost_ = *learned;
        } else if (prompt_tokens >= 32768) {
            cost_.base = 47.0;
            cost_.sy = 32.0 * cost_.base + 128.0 * cost_.slope;
            cost_.sxy = 128.0 * cost_.base + 800.0 * cost_.slope;
        }
    }

    const Qwen4ExpMtpCost & cost() const { return cost_; }
    Qwen4ExpMtpEvidence evidence() const { return { trials_, successes_ }; }

    bool enabled() const { return adaptive_; }

    int next_width() const {
        if (!adaptive_) return cap_ + 1;
        double survival = 1.0, commits = 1.0, best = 0.0;
        int chosen = 1;
        for (int k = 1; k <= cap_; ++k) {
            survival *= successes_[k] / trials_[k];
            commits += survival;
            const double utility = commits / cost_.at(k);
            if (utility > best) { best = utility; chosen = k; }
        }
        // A wider probe every 17 cycles is unconditional on timing or a
        // clean streak. It refreshes the first censored depth even at k=1.
        if (steps_ >= 16 && (steps_ - 16) % 17 == 0) {
            chosen = std::min(cap_, chosen + 1);
            // Every eighth probe refreshes ALL depths, including after a
            // phase change where two adjacent widths both look unprofitable.
            if (((steps_ - 16) / 17) % 8 == 7) chosen = cap_;
        }
        // Require 16 cycles before narrowing below k=3 (4 when earlier requests' acceptance is carried in);
        // clean warmup cycles probe upward immediately so counting reaches k=7 quickly.
        return (steps_ < (carried_ ? 4 : 16) ? std::max(chosen, std::min(warmup_k_, cap_)) : chosen) + 1;
    }

    // A missing timing (e.g. the oracle smoke) updates acceptance only.
    void observe(int accepted_width, int offered_width, float cycle_ms = -1.0f) {
        if (!adaptive_ || offered_width <= 1) return;
        const int k = std::clamp(offered_width - 1, 1, cap_);
        const int accepted = std::clamp(accepted_width - 1, 0, k);
        ++steps_;
        warmup_k_ = accepted == k ? std::min(cap_, k + 1) : 3;
        for (int depth = 1; depth <= std::min(k, accepted + 1); ++depth) {
            // Bounded evidence adapts to changing text without the old 0.2
            // EMA's effective five-sample window. Unreached depths stay put.
            if (trials_[depth] >= 128.0) {
                trials_[depth] *= 127.0 / 128.0;
                successes_[depth] *= 127.0 / 128.0;
            }
            trials_[depth] += 1.0;
            successes_[depth] += depth <= accepted;
        }
        if (!std::isfinite(cycle_ms) || cycle_ms <= 0.0f) return;
        Qwen4ExpMtpCost & c = cost_;
        if (++c.samples[k] <= 4) return; // cold graph/shape builds
        // This width's own average: a running mean, then a 1/16 average; past a
        // few samples, a spike counts at most twice the average.
        const int n = ++c.measured_n[k];
        const double sample = n > 4 ? std::min(double(cycle_ms), 2.0 * c.measured[k]) : double(cycle_ms);
        c.measured[k] += (sample - c.measured[k]) / std::min(n, 16);
        const double predicted = c.base + c.slope * k;
        const double measured = std::clamp(double(cycle_ms), predicted * 0.75, predicted * 1.25);
        constexpr double decay = 127.0 / 128.0;
        c.sw = decay * c.sw + 1.0;
        c.sx = decay * c.sx + k;
        c.sxx = decay * c.sxx + k * k;
        c.sy = decay * c.sy + measured;
        c.sxy = decay * c.sxy + k * measured;
        // Fit total cycle cost = fixed overhead + incremental draft/verify
        // row cost. Exploration supplies width variation; priors regularize
        // sparse contexts. No individual wall-time sample chooses a width.
        const double determinant = c.sw * c.sxx - c.sx * c.sx;
        if (determinant > 1e-6) {
            c.slope = std::max(0.1, (c.sw * c.sxy - c.sx * c.sy) / determinant);
            c.base = std::max(1.0, (c.sy - c.slope * c.sx) / c.sw);
        }
    }

private:
    int cap_, steps_ = 0, warmup_k_ = 3;
    bool adaptive_, carried_ = false;
    std::array<double, QWEN4EXP_MTP_MAX_VERIFY> trials_{}, successes_{};
    Qwen4ExpMtpCost cost_;
};

// Server --verify-width: 0 = adaptive k=1..7, 1 = off, 2..8 = fixed k=1..7.
// Eight verify rows is the RDNA3 batch-invariant MMVQ/MMID ceiling.
inline Qwen4ExpMtpWidth qwen4exp_mtp_width_policy(int max_draft, bool adaptive, int prompt_tokens = 0,
                                                  const Qwen4ExpMtpCost * learned = nullptr,
                                                  const Qwen4ExpMtpEvidence * evidence = nullptr) {
    return Qwen4ExpMtpWidth(max_draft, adaptive, prompt_tokens, learned, evidence);
}

// What the backend keeps across requests. Acceptance is the drafter's on recent text. The cycle cost is the
// device's, but attention's share of a verify step grows with the context (dense below 2K, then sparse with
// per-row indexer work), so each context regime keeps its own: widths timed in a 16K request made short
// replies' narrow widths look about 10% dearer than they are, and the policy drafted too deep.
struct Qwen4ExpMtpMemory {
    static constexpr int REGIMES = 4;
    std::array<Qwen4ExpMtpCost, REGIMES> cost{};
    std::array<bool, REGIMES> cost_learned{};
    Qwen4ExpMtpEvidence evidence;
    bool evidence_learned = false;

    static int regime(int64_t context) {
        return context < 2048 ? 0 : context < 8192 ? 1 : context < 32768 ? 2 : 3;
    }

    Qwen4ExpMtpWidth start(int max_draft, bool adaptive, int64_t context) const {
        const int r = regime(context);
        return qwen4exp_mtp_width_policy(max_draft, adaptive, (int) std::min<int64_t>(context, 1 << 30),
                                         cost_learned[r] ? &cost[r] : nullptr,
                                         evidence_learned ? &evidence : nullptr);
    }

    void finish(const Qwen4ExpMtpWidth & policy, int64_t context) {
        if (!policy.enabled()) return;
        const int r = regime(context);
        cost[r] = policy.cost();
        cost_learned[r] = true;
        evidence = policy.evidence();
        evidence_learned = true;
    }
};

inline int qwen4exp_mtp_next_width(const Qwen4ExpMtpWidth & policy) {
    return policy.next_width();
}

// Fixed production choice; smoke/session controls can measure other budgets.
constexpr int QWEN4EXP_MTP_VOCAB = 64000;

// Low BPE IDs are a rank heuristic, not a corpus-frequency guarantee. Reserve
// every tokenizer control first, then fill with the lowest ordinary IDs. Sort
// so local subset IDs and their embedding rows have one deterministic mapping.
inline std::vector<int32_t> qwen4exp_mtp_vocab_ids(int n_vocab, int budget,
                                                  const std::vector<int32_t> & required) {
    if (n_vocab <= 0 || budget <= 0) return {};
    std::vector<bool> keep((size_t) n_vocab, false);
    int count = 0;
    for (int32_t id : required) if (id >= 0 && id < n_vocab && !keep[id]) {
        keep[id] = true;
        ++count;
    }
    for (int id = 0; id < n_vocab && count < budget; ++id) if (!keep[id]) {
        keep[id] = true;
        ++count;
    }
    std::vector<int32_t> ids;
    ids.reserve(count);
    for (int id = 0; id < n_vocab; ++id) if (keep[id]) ids.push_back(id);
    return ids;
}

struct Qwen4ExpMtpAcceptance {
    int n_accepted = 0;
    int n_emitted = 0;
    std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> emitted{};
};

// Samples are the trunk's own draws, in order, after any budget substitution.
// A partial sample prefix allows callers to stop at EOS without drawing future
// tokens or advancing RNG/history. Retain n_emitted verify input positions:
// the last emitted token is the next (as yet unprocessed) input, as in AR decode.
inline Qwen4ExpMtpAcceptance qwen4exp_mtp_accept(
        const int32_t * drafts, int k, const int32_t * samples, int n_samples) {
    assert(k >= 0 && k <= QWEN4EXP_MTP_MAX_DRAFT);
    assert(n_samples >= 0 && n_samples <= k + 1);
    Qwen4ExpMtpAcceptance result;
    for (int i = 0; i < n_samples; ++i) {
        result.emitted[result.n_emitted++] = samples[i];
        if (i == k || samples[i] != drafts[i]) break;
        ++result.n_accepted;
    }
    return result;
}

// A speculative forward can complete a pooled block whose suffix is rejected.
// Only wholly retained blocks remain authoritative; the next completion must
// recompute the invalidated row using replacement raw keys.
inline int qwen4exp_mtp_retained_blocks(int pooled, int retained_pos, int ratio) {
    return ratio > 1 ? std::min(pooled, retained_pos / ratio) : pooled;
}

// ne[] dimensions of the six MTP sidecar tensors that mtp_forward_batch() (qwen4exp_graph.cpp) feeds
// straight into matmuls/reshapes with no further validation: a mismatch here is a crash or silent garbage
// at inference time, not a load-time error, unless caught first.
struct Qwen4ExpMtpShapeDims {
    int64_t eh_proj_ne0 = 0, eh_proj_ne1 = 0;      // [2H, H]: mm(eh_proj, [2H, hc*T]) reshaped to [H, hc, T]
    int64_t enorm_ne0 = 0;                          // [H]: elementwise with rms_norm(inp_emb) which is [H, T]
    int64_t hnorm_ne0 = 0;                          // [H*hc]: elementwise with rms_norm(inp_h) which is [H*hc, T]
    int64_t head_norm_ne0 = 0;                      // [H*hc]: hc_mix's rms_norm gamma over the draft head
    int64_t head_down_ne0 = 0, head_down_ne1 = 0;   // [H*hc, hc_lr]: low-rank down-projection
    int64_t head_up_ne0 = 0, head_up_ne1 = 0;       // [hc_lr, H*hc]: low-rank up-projection
};

// Pure: true iff every MTP tensor shape the forward graph relies on is internally consistent with the
// trunk's embedding/hyper-connection config. Takes plain ne[] values (not ggml_tensor*) so it is
// unit-testable with synthetic good/bad shapes, without a GGUF file or a GPU.
inline bool qwen4exp_mtp_shapes_valid(const Qwen4ExpMtpShapeDims & t,
                                      int64_t n_embd, int64_t n_hc, int64_t hc_lowrank) {
    const int64_t hc_dim = n_hc * n_embd;
    return t.eh_proj_ne0 == 2 * n_embd && t.eh_proj_ne1 == n_embd &&
           t.enorm_ne0 == n_embd &&
           t.hnorm_ne0 == hc_dim &&
           t.head_norm_ne0 == hc_dim &&
           t.head_down_ne0 == hc_dim && t.head_down_ne1 == hc_lowrank &&
           t.head_up_ne0 == hc_lowrank && t.head_up_ne1 == hc_dim;
}

}  // namespace luce::common
