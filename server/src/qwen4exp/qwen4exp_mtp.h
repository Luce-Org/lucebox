#pragma once

#include "common/adaptive_spec_width.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <vector>

namespace luce::common {

constexpr int QWEN4EXP_MTP_MAX_DRAFT = 7;
constexpr int QWEN4EXP_MTP_MAX_VERIFY = QWEN4EXP_MTP_MAX_DRAFT + 1;

// The verify-width controller of qwen4exp's MTP: the shared one over widths
// 2..8 (seed included). Its cost seed is only a shape, a fixed step overhead
// plus a per-row cost, scaled to this device as widths get timed; acceptance
// is counted at reached depths, the right model for a chained MTP head whose
// acceptance can fall off a cliff at one depth. The backend keeps one
// controller across requests: SpecWidthCostMemory holds its costs per context
// range, and carry_acceptance() bounds what one request's text tells the next.
inline AdaptiveSpecWidth qwen4exp_mtp_width_controller() {
    AdaptiveSpecWidth width(QWEN4EXP_MTP_MAX_VERIFY);
    std::vector<float> shape(QWEN4EXP_MTP_MAX_VERIFY + 1, 0.0f);
    for (int w = 2; w <= QWEN4EXP_MTP_MAX_VERIFY; ++w) shape[(size_t) w] = 45.0f + 20.0f * (float) (w - 1);
    width.set_relative_costs(shape, AdaptiveSpecWidth::CostSeed::kShape);
    width.set_acceptance_model(AdaptiveSpecWidth::AcceptanceModel::kReachedDepths);
    return width;
}

// Evidence one request's text may pass to the next, per depth.
constexpr float QWEN4EXP_MTP_CARRIED_TRIALS = 32.0f;

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

// The trunk pairs not yet in the draft layer's K/V: hidden row h[i] (the trunk's final HC residual at position
// pos + i) and the token tok[i] that followed it. The next draft catches them up first.
struct Qwen4ExpMtpPending {
    std::vector<int32_t> tok;
    std::vector<float> h;
    int pos = 0;
};

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
