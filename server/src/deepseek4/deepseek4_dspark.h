// DeepSeek-V4-Flash "DSpark" speculative-decode drafter.
//
// The DSpark drafter is a small (n_layer≈3) DeepSeek-V4 block stack stored under
// the checkpoint's mtp.* namespace and converted to a GGUF with arch
// "deepseek4-dflash-draft" (see scripts/convert_ds4_dspark_draft_to_gguf.py).
//
// Reference forward: deepseek-ai/DeepSeek-V4-Flash-DSpark inference/model.py
// (DSparkBlock / DSparkAttention / DSparkMarkovHead / DSparkConfidenceHead,
//  Transformer.forward_spec). Key facts encoded here:
//   - The target captures h.mean(dim=2) (mean over the hc_mult HC copies) after
//     each of dspark_target_layer_ids (=[40,41,42]) -> main_hidden [.., 3*n_embd].
//   - forward_embed (stage 0): main_x = main_norm(main_proj(main_hidden)); the
//     noise block = embed([seed]+[MASK]*(block_size-1)), HC-expanded.
//   - DSparkAttention: compress_ratio==0 (no compressor/indexer). Each layer
//     projects the shared main_x -> main_kv via its own wkv, and the block's
//     block_size query positions attend BIDIRECTIONALLY over
//     [sliding-window main-context KV] ++ [block KV].
//   - forward_head (last stage): hc_head collapse -> out_norm -> tied target
//     lm_head -> per-position Markov correction + confidence gate.
//
// The drafter's token embedding and lm_head are TIED to the target, so the
// drafter GGUF carries neither token_embd nor output; embedding + projection
// go through the DeepSeek4DFlashTarget adapter.

#pragma once

#include "deepseek4_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

namespace dflash::common {

// The drafter weights. `core` reuses DeepSeek4Weights for the n_layer decoder
// blocks + per-layer tensors + metadata + out_norm + output_hc_* tail; its
// tok_embd/output stay null (tied to target). The DSpark-specific tensors below
// live in the same ggml context / backend buffer as `core`.
struct DSparkDrafter {
    DeepSeek4Weights core;

    // Captured-feature fusion (stage 0 only in the checkpoint, but stored global).
    ggml_tensor * main_proj    = nullptr;  // dflash.fc.weight            [n_tgt*n_embd, n_embd]
    ggml_tensor * main_norm    = nullptr;  // dflash.hidden_norm.weight   [n_embd]

    // DSpark heads (last stage).
    ggml_tensor * markov_w1    = nullptr;  // dflash.dspark.markov.w1        [markov_rank, vocab]
    ggml_tensor * markov_w2    = nullptr;  // dflash.dspark.markov.w2        [markov_rank, vocab]
    ggml_tensor * confidence_w = nullptr;  // dflash.dspark.confidence.weight [confidence_dim, 1]
    ggml_tensor * confidence_b = nullptr;  // dflash.dspark.confidence.bias   [1]

    int block_size      = 5;
    int n_target_layers = 3;
    int markov_rank     = 256;
    int vocab_size      = 129280;
    int confidence_dim  = 0;
    int mask_token_id   = 128799;
    bool dspark_enabled  = false;
    bool head_hc_enabled = false;
    std::vector<int> capture_layer_ids;  // [40,41,42]
};

// Load a "deepseek4-dflash-draft" GGUF into `out`. Returns false on error;
// deepseek4_dspark_last_error() has the message.
bool load_deepseek4_dspark_drafter(const std::string & path,
                                   ggml_backend_t backend,
                                   DSparkDrafter & out);

void free_deepseek4_dspark_drafter(DSparkDrafter & d);

const char * deepseek4_dspark_last_error();

// One drafter forward. Produces block_size normed hidden states (the input to
// the tied lm_head + Markov head), conditioned on a window of captured target
// features. All host-side f32 for a simple v1 (GPU feature-ring plumbing can
// come later).
//
//   seed_tok        : the committed anchor token (block position 0's embedding)
//   noise_embed     : [n_embd * block_size] embeds of [seed]+[MASK]*(block_size-1)
//   ctx_features    : [n_target_layers*n_embd * ctx_len] captured features,
//                     ordered oldest..newest, absolute positions
//                     [committed-ctx_len .. committed-1]
//   ctx_len         : number of context feature columns (<= n_swa)
//   committed       : absolute position of the seed (block position 0)
//   out_hidden      : filled with [n_embd * block_size] = out_norm(hc_head(block))
//
// Defined in deepseek4_graph.cpp (needs the static DS4 sub-builders).
bool deepseek4_dspark_draft_forward(ggml_backend_t backend,
                                    const DSparkDrafter & d,
                                    const float * noise_embed,
                                    const float * ctx_features,
                                    int ctx_len,
                                    int committed,
                                    std::vector<float> & out_hidden);

// Batched target verify forward WITH feature capture (defined in
// deepseek4_graph.cpp so it can reuse the target sub-builders). Runs the DS4
// target over `n_tokens` embeddings at absolute position `kv_start` in one
// forward and returns:
//   argmax_out  : per-position argmax token id (size n_tokens)
//   logits_out  : if non-null, full [n_tokens * n_vocab] f32 logits
//   capture_out : [n_target_layers*n_embd * n_tokens] f32, per-position mean
//                 over the hc_mult HC copies after each capture layer
//                 (concatenated in capture_layer_ids order) — the drafter's
//                 main_hidden feed.
// Advances/updates the target cache exactly like a decode of these tokens.
bool deepseek4_dspark_verify_forward(ggml_backend_t backend,
                                     const DeepSeek4Weights & w,
                                     DeepSeek4Cache & cache,
                                     const std::vector<int> & capture_layer_ids,
                                     const float * embed,
                                     const int32_t * token_ids,
                                     int n_tokens,
                                     int kv_start,
                                     std::vector<int32_t> & argmax_out,
                                     std::vector<float> * logits_out,
                                     std::vector<float> & capture_out,
                                     DeepSeek4StepTelemetry * telemetry = nullptr,
                                     bool allow_graph_reuse = true);

// Run DSpark speculative decode: draft block_size candidates with `drafter`,
// verify against the DS4 target in one batched forward, accept the matching
// prefix, and loop. Returns generated tokens via `io.emit`. Mirrors the laguna
// DSpark loop. accept_rate_out (optional) gets mean accepted / block.
struct GenerateRequest;  // fwd (from common/…); the loop only needs n_gen + committed
bool run_deepseek4_dspark_spec_decode(
        ggml_backend_t backend,
        const DeepSeek4Weights & target_w,
        DeepSeek4Cache & target_cache,
        const DSparkDrafter & drafter,
        int committed,
        int last_tok,
        int n_gen,
        const float * prompt_feature_window,  // [n_target_layers*n_embd * win_len] captured during prefill
        int win_len,
        std::vector<int32_t> & out_tokens,
        float * accept_rate_out);

}  // namespace dflash::common
