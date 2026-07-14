#pragma once
// [TAG_MMID_ADAPTIVE_K] per-token expert-count gating for small MUL_MAT_ID
// batches (speculative-verify sized, 2..16 tokens). Contract with the grouped
// MUL_MAT_ID path in ggml-cuda mmvq.cu (DFLASH_MMID_GROUPED=1): when a router
// ids tensor's ->extra points at an mmid_gate_extra, the grouped prep kernel
// keeps each token's leading experts until their cumulative combine weight
// reaches tau, sentinels the rest of the token's ids to -1 (skipped, exact
// zero contribution) and renormalizes the kept weights in place.
//   DFLASH_ADAPTIVE_K_TAU=<0..1>   enables (0/unset = off)
//   DFLASH_ADAPTIVE_K_DENSE=<csv>  layers kept at full top-k. DFlash capture
//                                  layers MUST stay dense so drafter
//                                  conditioning features are unchanged.
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

struct mmid_gate_extra {
    uint32_t            magic;   // MMID_GATE_MAGIC
    float               tau;
    const ggml_tensor * weights; // [n_used, n_tokens] f32 combine weights
};
#define MMID_GATE_MAGIC 0x4D474154u

// il < 0 = layer index unknown for this family: the dense-layer list cannot
// be applied, every MoE layer is gated.
inline void mmid_adaptive_k_attach(ggml_tensor * ids, const ggml_tensor * weights,
                                   int n_tokens, int il, const char * dense_default) {
    static const float tau = []() {
        const char * e = std::getenv("DFLASH_ADAPTIVE_K_TAU");
        if (!e) return 0.0f;
        char * end = nullptr;
        const float v = std::strtof(e, &end);
        if (end == e || *end != '\0' || v < 0.0f || v > 1.0f) {
            std::fprintf(stderr, "[adaptive-k] ignoring DFLASH_ADAPTIVE_K_TAU=\"%s\""
                                 " (want a float in [0,1])\n", e);
            return 0.0f;
        }
        return v;
    }();
    if (tau <= 0.0f || n_tokens < 2 || n_tokens > 16 || ids == nullptr || weights == nullptr) {
        return;
    }
    if (il < 0) {
        static const bool warned = []() {
            std::fprintf(stderr,
                "[adaptive-k] WARNING: this model family does not thread layer "
                "indices into the router yet, so DFLASH_ADAPTIVE_K_DENSE cannot "
                "be honored and ALL MoE layers are gated - including DFlash "
                "capture layers, which can degrade drafter acceptance.\n");
            return true;
        }();
        (void) warned;
    }
    if (il >= 0) {
        const char * e = std::getenv("DFLASH_ADAPTIVE_K_DENSE");
        const std::string str = e ? e : (dense_default ? dense_default : "");
        size_t pos = 0;
        while (pos < str.size()) {
            const size_t q = str.find(',', pos);
            const std::string tok = str.substr(pos, q == std::string::npos ? std::string::npos : q - pos);
            if (!tok.empty()) {
                char * end = nullptr;
                const long v = std::strtol(tok.c_str(), &end, 10);
                if (end == tok.c_str() || *end != '\0') {
                    static bool warned_bad_dense = false;
                    if (!warned_bad_dense) {
                        warned_bad_dense = true;
                        std::fprintf(stderr, "[adaptive-k] ignoring malformed "
                                             "DFLASH_ADAPTIVE_K_DENSE entry \"%s\"\n",
                                     tok.c_str());
                    }
                } else if ((int) v == il) {
                    return;
                }
            }
            if (q == std::string::npos) break;
            pos = q + 1;
        }
    }
    // The extra must outlive graph evals, and builds run repeatedly in a
    // server (ggml_new_tensor zeroes ->extra even when the persistent arena
    // hands back the same address). Keep one allocation per distinct tensor
    // address in a process-lifetime pool: with persistent-arena builders the
    // address set is small and stable, so this bounds the former
    // one-allocation-per-rebuild leak. Unsynchronized on purpose: graph
    // builds are single-threaded (same assumption as the thread_local
    // builder arenas).
    static std::unordered_map<const ggml_tensor *, mmid_gate_extra *> pool;
    mmid_gate_extra *& gx = pool[ids];
    if (gx == nullptr) gx = new mmid_gate_extra{MMID_GATE_MAGIC, tau, weights};
    gx->tau     = tau;
    gx->weights = weights;
    ids->extra  = gx;
}
