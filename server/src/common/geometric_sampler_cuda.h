// CUDA port of the shared sample_logits chain (see sampler.cpp / sampler.h).
//
// Rationale for what is / isn't on the GPU
// ----------------------------------------
// The model already produces logits on the GPU. The CPU sampler forces a full
// ~vocab-wide D2H copy of those logits every token (the existing greedy path
// already dodges this with DFLASH_GPU_ARGMAX). The vocab-wide work — penalty
// application, the softmax max/sum-exp reductions, and the multinomial
// inverse-CDF draw — is data-parallel and is what geometric_sample_logits_cuda
// moves to the GPU, entirely, for greedy and plain temperature/penalty
// sampling. A few things deliberately stay on (or partly on) the CPU, decided
// by measurement rather than what's merely possible on the GPU:
//   * building the repetition/frequency penalty index from the (<=rep_window)
//     token history — a few hundred elements, dwarfed by a kernel launch;
//   * the single scalar RNG draw (a std::mt19937_64 uniform) — kept on the host
//     so the random stream stays reproducible and identical to the CPU path;
//   * top_k selection (with or without top_p) — a CPU partial_sort scales with
//     k, not vocab, and is already cheap; a GPU round trip (kernel launch +
//     D2H copy) measured as a net *regression* here, not just a non-win, so
//     cfg.top_k>0 always returns -1 from geometric_sample_logits_cuda and the
//     caller (sample_logits) never routes it through geometric_compute_probs_cuda
//     either;
//   * top_p (nucleus) without top_k — geometric_sample_logits_cuda still
//     returns -1 for this (no on-GPU nucleus search), but geometric_compute_probs_cuda
//     lets the CPU skip straight to its O(vocab) std::nth_element-based nucleus
//     search instead of also recomputing penalties+softmax; measured as a real
//     (if modest, ~1.4x) end-to-end win, since that search's cost dominates
//     either way.

#pragma once

#include "sampler.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// GPU sample_logits. Returns the chosen token id, or -1 when the caller should
// fall back to the CPU sample_logits (unsupported config such as cfg.top_k>0
// or cfg.top_p in (0,1), or any CUDA error).
//
//   logits           : pointer to vocab contiguous floats.
//   logits_on_device : true  -> `logits` is a device pointer (e.g. a ggml CUDA
//                               tensor's ->data); the D2H copy is skipped and we
//                               read straight from device memory (the real win).
//                      false -> `logits` is host memory and is uploaded H2D.
//   r_uniform        : a pre-drawn uniform in [0,1) from the caller's RNG;
//                      ignored for greedy (cfg.temp <= 0).
int geometric_sample_logits_cuda(const float * logits,
                       int vocab,
                       const SamplerCfg & cfg,
                       const std::vector<int32_t> & history,
                       double r_uniform,
                       bool logits_on_device);

// Computes the post-penalty, post-softmax(temp) probability vector on the GPU
// and writes it to `out_probs` (vocab floats, host memory) — the same
// vocab-wide, data-parallel prefix geometric_sample_logits_cuda computes
// internally, exposed here for callers that need to do their own top_k/top_p
// truncation afterward (unsupported inside the kernel above). Lets the CPU
// skip re-deriving penalties+softmax when cfg.top_k>0 or cfg.top_p in (0,1)
// forces truncation to happen on the host.
//
// Returns false (out_probs left untouched) if cfg.temp <= 0 (softmax is
// undefined for greedy — callers should use plain argmax instead) or on any
// CUDA error, so the caller falls back to computing everything on the CPU.
bool geometric_compute_probs_cuda(const float * logits,
                        int vocab,
                        const SamplerCfg & cfg,
                        const std::vector<int32_t> & history,
                        float * out_probs,
                        bool logits_on_device);

// True unless the env var DFLASH_GPU_SAMPLE is explicitly set to "0" — the GPU
// path is enabled by default. Cached after the first call. Lets call sites
// gate the GPU path at runtime.
bool gpu_sampler_enabled();

// Whether geometric_sample_logits_cuda can (and should) handle this config
// entirely on the GPU. false for top_k>0 or top_p in (0,1) — not because
// geometric_sample_logits_cuda can't be asked to try (it always returns -1
// for those anyway), but so callers like sample_logits know to reach for
// geometric_compute_probs_cuda instead for the pure-top_p case. See the
// rationale comment at the top of this file for the per-case reasoning.
inline bool gpu_sampler_supports(const SamplerCfg & cfg) {
    if (cfg.top_k > 0) return false;                          // top_k: not implemented on GPU
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) return false;   // top_p: not implemented on GPU
    return true;
}

}  // namespace dflash::common
