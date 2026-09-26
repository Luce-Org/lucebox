// Internal interface of the Qwen3.5-0.8B drafter — the current pflash scorer.
//
// The scorer runs on the Qwen3.5 target architecture (TargetWeights,
// build_qwen35_layer): qwen35_loader.cpp loads the GGUF, the optional
// scoring head and the segment probe; qwen35_drafter.cpp runs the two
// scorers. pflash_drafter.cpp owns the public entry points.

#pragma once

#include "pflash_drafter.h"
#include "pflash_selection.h"
#include "common/pflash_types.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// Qwen3.5-0.8B scoring head. Features are the residual entering
// full-attention block 15 after the first 15 blocks (twelve GatedDeltaNet and
// three full-attention blocks). Block 15's own Q/K projections score the
// context without RoPE; an optional trained head replaces those two
// projections.
static constexpr int kQwen35HeadBlock = 15;

// What the scorer computed for one conversation's prompt, kept so the next
// turn only runs the new tokens. Blocks 0..14 read left to right, so the
// cache state and block-15 keys of a shared prefix never change; the keys
// carry no position (NoPE), and the probe's raw logits are per token.
struct Qwen35ScoringSession {
    std::vector<int32_t> ids;       // prompt tokens the state covers
    int checkpoint = 0;             // recurrent-state snapshot position
    int capacity = 0;               // tokens the cache and keys can hold
    TargetCache cache;              // blocks 0..14 only
    ggml_context *        key_ctx = nullptr;
    ggml_backend_buffer_t key_buf = nullptr;
    ggml_tensor *         keys = nullptr;   // [head_dim, n_head_kv, capacity] f32
    bool                  keys_trained = false;
    std::vector<float>    probe_raw;        // per token, unit logit
    std::vector<float>    subunit_raw;      // per token, when the probe has one
    // Block-14 output of recent query windows (the query and earlier
    // questions), reused while they sit in the shared prefix: an agent step
    // appends tool output after the same user turn, and a new turn's history
    // queries are earlier turns' queries. Most recent last, at most 8.
    struct QueryRows {
        int begin = -1;
        int end = -1;
        std::vector<float> rows;            // [hidden, end - begin]
    };
    std::vector<QueryRows> query_windows;
    uint64_t              last_used = 0;
};

void free_qwen35_scoring_session(Qwen35ScoringSession & session);

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
    // Strict scorer sessions, least recently used evicted
    // (PFLASH_DRAFTER_SESSIONS, default 2; 0 scores every prompt from scratch).
    std::vector<std::unique_ptr<Qwen35ScoringSession>> sessions;
    uint64_t              session_clock = 0;
};

// Defined in qwen35_loader.cpp.
bool qwen35_head_block_available(const TargetWeights & w, std::string & error);
bool load_qwen35_drafter(const std::string & gguf_path, DrafterContext & out);
void free_qwen35_drafter_state(DrafterContext & ctx);

// Defined in qwen35_drafter.cpp.
//
// Causal flash-attention mask of one ubatch of n_tokens queries at kv_start,
// computed by the graph on the device: [align32(kv_start + n_tokens),
// align32(n_tokens)] F16, row q < n_tokens zero over keys
// [0, kv_start + q + 1) and -inf after, rows past n_tokens -inf throughout.
// Its writes are expanded into gf here, ahead of the attention that reads it.
ggml_tensor * build_qwen35_causal_mask(ggml_context * ctx, ggml_cgraph * gf,
                                       int kv_start, int n_tokens);

// The legacy all-layer running-max scorer, on the Qwen3.5 architecture.
std::vector<int32_t> qwen35_score_and_compress(
    TargetWeights & w,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_scores_out = nullptr);

// The block-15 scoring head under strict budget selection.
std::vector<int32_t> qwen35_strict_score_and_compress(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
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
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans);

} // namespace luce::common
