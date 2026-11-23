// Adjacent-segment mean-key cosine similarity and normalized segment mean keys
// (`k_bar`) for the PFlash bidirectional segment reranker.
//
// Given block-15 keys [D, Hk, capacity] F32 (contiguous K_dim = D * Hk floats per token)
// and contiguous segment spans [begin, end), computes:
//   out_kcos[0] = 0.0f
//   out_kcos[i] = clamp(cos(mean(keys[segs[i-1]]), mean(keys[segs[i]])), 0.0f, 1.0f)
//
// Uses a single-launch CUDA/HIP kernel on GPU backends and a chunked CPU path on
// other backends.

#pragma once

#include "common/pflash_types.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace luce::common {

// Pure CPU reference over host F32 keys [S * K_dim] (ne0 = K_dim contiguous per token).
// If `out_seg_keys` is non-null, writes normalized segment mean keys `k_bar` [N_seg * K_dim].
// If `token_mass` and `out_seg_katt` are non-null, also writes normalized mass-weighted
// keys `k_att` [N_seg * K_dim] (falling back to `k_bar` when a segment has zero mass).
void pflash_adjacent_segment_key_cosine_cpu(
    const float * keys,
    int n_tokens,
    int k_dim,
    const std::vector<PFlashTokenSpan> & segments,
    std::vector<float> & out_kcos,
    std::vector<float> * out_seg_keys = nullptr,
    const float * token_mass = nullptr,
    std::vector<float> * out_seg_katt = nullptr);

#if defined(LUCE_HAVE_PFLASH_SEG_KCOS)
// Device launcher defined in pflash_seg_kcos_cuda.cu (compiled as CUDA or HIP).
bool pflash_adjacent_segment_key_cosine_device(
    const float * keys_dev,
    const int32_t * bounds_dev,
    float * seg_keys_dev,
    float * kcos_dev,
    int n_segments,
    int k_dim,
    void * stream,
    const float * token_mass_dev = nullptr,
    float * seg_katt_dev = nullptr);
#endif

// Computes clamped adjacent-segment key cosines directly from session->keys.
// If `out_seg_keys` is non-null, also copies normalized segment mean keys `k_bar` [N_seg * K_dim] F32.
// If `token_mass` and `out_seg_katt` are non-null, also computes and copies normalized
// mass-weighted keys `k_att` [N_seg * K_dim] F32 in the same pass over `keys`.
// Returns false only on invalid inputs or backend allocation failure.
bool pflash_adjacent_segment_key_cosine(
    ggml_backend_t backend,
    ggml_tensor * keys,
    const std::vector<PFlashTokenSpan> & segments,
    std::vector<float> & out_kcos,
    std::vector<float> * out_seg_keys = nullptr,
    const float * token_mass = nullptr,
    std::vector<float> * out_seg_katt = nullptr);

} // namespace luce::common
