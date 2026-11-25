// Bidirectional segment transformer reranker for PFlash.
//
// Loaded from PFLASH_SEG_RERANKER_GGUF (schema `pflash_seg_reranker_v1`), one of the
// four components PFlash requires. Runs over segments, not tokens, with standard ggml
// ops (RMSNorm, NeoX 1D RoPE, bidirectional multi-head self-attention, SwiGLU FFN) and
// adds a residual delta to each segment's Stage-1 log density.
#pragma once

#include "common/pflash_types.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

namespace luce::common {

struct PFlashSegRerankerLayer {
    ggml_tensor * norm1_w = nullptr;
    ggml_tensor * q_proj_w = nullptr;
    ggml_tensor * k_proj_w = nullptr;
    ggml_tensor * v_proj_w = nullptr;
    ggml_tensor * o_proj_w = nullptr;
    ggml_tensor * norm2_w = nullptr;
    ggml_tensor * gate_proj_w = nullptr;
    ggml_tensor * up_proj_w = nullptr;
    ggml_tensor * down_proj_w = nullptr;
};

struct PFlashSegRerankerState {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::string feature_mode; // "kbar" (k_bar, 512-dim) or "kbar_katt" ([k_bar, k_att], 1024-dim)
    int in_dim = 0;
    int n_scalars = 12;
    int d_model = 0;
    int n_heads = 0;
    int n_layers = 0;
    int d_ff = 0;
    float rope_theta = 10000.0f;
    float norm_eps = 1e-6f;
    float delta_min = -1e9f;
    float delta_max = 1e9f;

    ggml_tensor * k_in_w = nullptr;
    ggml_tensor * k_in_b = nullptr;
    ggml_tensor * q_in_w = nullptr;
    ggml_tensor * q_in_b = nullptr;
    ggml_tensor * s_in_w = nullptr;
    ggml_tensor * s_in_b = nullptr;
    ggml_tensor * int_in_w = nullptr;
    ggml_tensor * in_norm_w = nullptr;

    std::vector<PFlashSegRerankerLayer> layers;

    ggml_tensor * out_norm_w = nullptr;
    ggml_tensor * score_head_w = nullptr;
    ggml_tensor * score_head_b = nullptr;
    ggml_tensor * pool_gate_w = nullptr;
    ggml_tensor * pool_gate_b = nullptr;
    ggml_tensor * budget_head_w = nullptr;
    ggml_tensor * budget_head_b = nullptr;

    std::string loaded_file_identity;
    bool loaded = false;
};

void free_pflash_seg_reranker(PFlashSegRerankerState & st);

// Header-only contract check for a `pflash_seg_reranker_v1` GGUF without
// reading tensor payloads or touching a backend. If `expected_k_dim > 0`, also
// verifies `in_dim` matches the feature mode and drafter key width.
bool validate_pflash_seg_reranker_gguf(
    const std::string & path,
    const std::string & expected_drafter_sha256,
    const std::string & expected_head_sha256,
    int expected_k_dim,
    std::string & error);

// Load a `pflash_seg_reranker_v1` GGUF onto `backend`.
// 2D matmul weight tensors may be F32 or F16; 1D norms and biases must be F32.
// If `expected_drafter_sha256` is non-empty, enforces exact drafter binding; if
// `expected_head_sha256` is non-empty and the GGUF names its scoring head
// (`reranker.scoring_head_sha256`), enforces exact head binding.
bool load_pflash_seg_reranker(
    const std::string & path,
    ggml_backend_t backend,
    const std::string & expected_drafter_sha256,
    const std::string & expected_head_sha256,
    PFlashSegRerankerState & out,
    std::string & error);

// Build the 12 per-segment scalar features + log_density[N_seg] defined by
// schema `pflash_seg_reranker_v1`.
void pflash_reranker_build_scalars(
    int input_tokens,
    const std::vector<float> & token_mass,
    const std::vector<PFlashTokenSpan> & segments,
    const std::vector<float> * container_probs,
    const std::vector<float> * adj_kcos,
    std::vector<float> & out_scalars,
    std::vector<float> & out_log_density);

// Run the bidirectional segment reranker on `backend`.
// Inputs:
//   seg_vec:     [n_seg * in_dim] F32 (row-major: segment i at seg_vec + i * in_dim)
//   q_vec:       [in_dim] F32
//   scalars:     [n_seg * n_scalars] F32
//   log_density: [n_seg] F32
// Outputs:
//   out_scores:  [n_seg] F32 (= log_density + delta)
//   out_delta:   optional [n_seg] F32
//   out_m_star:  optional predicted oracle keep ratio in [0.05, 0.60]
bool pflash_seg_reranker_run(
    const PFlashSegRerankerState & st,
    ggml_backend_t backend,
    int n_seg,
    const float * seg_vec,
    const float * q_vec,
    const float * scalars,
    const float * log_density,
    std::vector<float> & out_scores,
    std::vector<float> * out_delta = nullptr,
    float * out_m_star = nullptr,
    std::string * error = nullptr);

// Adaptive keep ratio computed from the reranker's optional-segment log-scores,
// Stage-1 token masses and token lengths.
// Two signals:
//   1. The sink-free token perplexity fraction `eff = exp(H_tok) / N_opt` (T = 0.85),
//      with single-segment spikes clipped at `med + 4.0`. From kPFlashAdaptiveBypassEff
//      the prompt goes whole (bypass_to_full); from kPFlashAdaptiveWidenEff the keep
//      ratio widens to eff.
//   2. When `stage1_masses` is given, `nucleus_ratio`: the share of optional tokens that
//      holds 90% of the Stage-1 attention mass, walking segments in reranked order,
//      capped at kPFlashAdaptiveBypassEff. A sharp prompt reaches 90% inside the base
//      curve and keeps it.
// Effective support share (eff below) from which the keep ratio widens to eff, and
// from which the prompt is served whole when it fits the context (else kept at
// this share). The Stage-1 mass floor is capped at the bypass share too.
inline constexpr float kPFlashAdaptiveWidenEff = 0.42f;
inline constexpr float kPFlashAdaptiveBypassEff = 0.58f;

struct PFlashAdaptiveBudget {
    bool  valid = false;
    float eff = 0.0f;
    float nucleus_ratio = 0.0f;
    float keep_ratio = 0.0f;
    bool  bypass_to_full = false;
};

PFlashAdaptiveBudget pflash_reranker_adaptive_budget(
    const std::vector<PFlashTokenSpan> & segments,
    const std::vector<float> & rerank_scores,
    const std::vector<uint8_t> & mandatory_mask,
    const std::vector<double> * stage1_masses = nullptr);

} // namespace luce::common
