// Internal interface of the Qwen3.5-0.8B drafter.
//
// The Qwen3.5-0.8B scorer runs on the Qwen3.5 target architecture
// (TargetWeights, build_qwen35_layer) rather than the Qwen3-0.6B drafter
// graph, so it lives in its own translation units: qwen35_loader.cpp loads
// the GGUF, the scoring head and the segment probe; qwen35_drafter.cpp runs
// the two scorers. qwen3_drafter.cpp dispatches here on DrafterArch.

#pragma once

#include "qwen3_drafter.h"
#include "pflash_selection.h"
#include "common/pflash_types.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

namespace dflash::common {

// Qwen3.5-0.8B scoring head. Features are the residual entering
// full-attention block 15 after the first 15 blocks (twelve GatedDeltaNet and
// three full-attention blocks). Block 15's own Q/K projections score the
// context without RoPE, exactly like the Qwen3-0.6B block-13 head; an
// optional trained head replaces those two projections.
static constexpr int kQwen35HeadBlock = 15;

struct Qwen35DrafterState {
    TargetWeights weights;
    std::string gguf_sha256;
    ggml_context *        head_ctx = nullptr;
    ggml_backend_buffer_t head_buf = nullptr;
    ggml_tensor *         head_wq  = nullptr;  // [hidden, n_head * head_dim], query rows only
    ggml_tensor *         head_wk  = nullptr;  // [hidden, n_head_kv * head_dim]
    bool                  head_loaded = false;
    // Segment probe: per-token boundary scores from the same block-14 tap.
    ggml_context *        probe_ctx = nullptr;
    ggml_backend_buffer_t probe_buf = nullptr;
    ggml_tensor *         probe_norm_w = nullptr;  // [hidden]
    ggml_tensor *         probe_norm_b = nullptr;  // [hidden]
    ggml_tensor *         probe_fc1_w  = nullptr;  // [hidden, probe_width]
    ggml_tensor *         probe_fc1_b  = nullptr;  // [probe_width]
    ggml_tensor *         probe_fc2_w  = nullptr;  // [probe_width, 1]
    ggml_tensor *         probe_fc2_b  = nullptr;  // [1]
    ggml_tensor *         probe_sub_fc2_w = nullptr; // optional sub-unit head (oversize split only)
    ggml_tensor *         probe_sub_fc2_b = nullptr;
    std::vector<float>    probe_conv_w;            // taps from the GGUF tensor, applied on the CPU
    float                 probe_conv_b = 0.0f;
    std::vector<float>    probe_sub_conv_w;
    float                 probe_sub_conv_b = 0.0f;
    float                 probe_norm_eps = 1e-5f;
    float                 probe_threshold = 0.9f;
    int                   probe_min_segment = 1;
    int                   probe_max_segment = 2048;
    int                   probe_width = 0;
    bool                  probe_loaded = false;
};

// Defined in qwen35_loader.cpp.
bool qwen35_head_block_available(const TargetWeights & w, std::string & error);
bool load_qwen35_drafter(const std::string & gguf_path, DrafterArch arch,
                         DrafterContext & out);
void free_qwen35_drafter_state(DrafterContext & ctx);

// Defined in qwen35_drafter.cpp.
//
// The legacy all-layer running-max scorer, on the Qwen3.5 architecture.
std::vector<int32_t> qwen35_score_and_compress(
    TargetWeights & w,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const dflash::qwen3::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_scores_out = nullptr);

// The block-15 scoring head under strict budget selection.
std::vector<int32_t> qwen35_strict_score_and_compress(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const dflash::qwen3::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_mass_out = nullptr,
    std::vector<PFlashTokenSpan> * segments_out = nullptr,
    bool * density_out = nullptr);

// Arch dispatch target of drafter_score_and_compress.
std::vector<int32_t> qwen35_drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const dflash::qwen3::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans);

} // namespace dflash::common
