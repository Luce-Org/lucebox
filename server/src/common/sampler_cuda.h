// CUDA port of the shared sample_logits chain (see sampler.cpp / sampler.h).
//
// Rationale for what is / isn't on the GPU
// ----------------------------------------
// The model already produces logits on the GPU. The CPU sampler forces a full
// ~vocab-wide D2H copy of those logits every token (the existing greedy path
// already dodges this with DFLASH_GPU_ARGMAX). The vocab-wide work — penalty
// application, the softmax max/sum-exp reductions, nucleus (top_p) thresholding
// and the multinomial inverse-CDF draw — is data-parallel and is what this file
// moves to the GPU. Three things deliberately stay on the CPU because porting
// them buys nothing:
//   * building the repetition/frequency penalty index from the (<=rep_window)
//     token history — a few hundred elements, dwarfed by a kernel launch;
//   * the single scalar RNG draw (a std::mt19937_64 uniform) — kept on the host
//     so the random stream stays reproducible and identical to the CPU path;
//   * top_k selection — a partial_sort over a single row is cheap on the CPU and
//     a full GPU sort/select per token is not worth it, so cfg.top_k>0 returns
//     -1 here and the caller falls back to the CPU chain.

#pragma once

#include "sampler.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// GPU sample_logits. Returns the chosen token id, or -1 when the caller should
// fall back to the CPU sample_logits (unsupported config such as cfg.top_k>0,
// or any CUDA error).
//
//   logits           : pointer to vocab contiguous floats.
//   logits_on_device : true  -> `logits` is a device pointer (e.g. a ggml CUDA
//                               tensor's ->data); the D2H copy is skipped and we
//                               read straight from device memory (the real win).
//                      false -> `logits` is host memory and is uploaded H2D.
//   r_uniform        : a pre-drawn uniform in [0,1) from the caller's RNG;
//                      ignored for greedy (cfg.temp <= 0).
int sample_logits_cuda(const float * logits,
                       int vocab,
                       const SamplerCfg & cfg,
                       const std::vector<int32_t> & history,
                       double r_uniform,
                       bool logits_on_device);

// True when the env opt-in DFLASH_GPU_SAMPLE is set to something other than "0".
// Cached after the first call. Lets call sites gate the GPU path at runtime.
bool gpu_sampler_enabled();

// Whether the GPU path is the *faster* choice for this config, used to gate
// production dispatch (sample_logits / the backends) — distinct from what
// sample_logits_cuda can compute correctly. Microbenched on RTX 3090 / Qwen3
// vocab: greedy ~150x, temperature & penalties ~5x faster than the CPU chain,
// but nucleus (top_p<1) is ~3x SLOWER because the single-block threshold search
// can't use the device — so top_p (and top_k, already CPU-only) stay on the CPU.
inline bool gpu_sampler_supports(const SamplerCfg & cfg) {
    if (cfg.top_k > 0) return false;                         // top_k: CPU partial_sort wins
    if (cfg.temp <= 0.0f) return true;                       // greedy: always a win
    return !(cfg.top_p > 0.0f && cfg.top_p < 1.0f);          // top_p nucleus: keep on CPU
}

}  // namespace dflash::common
