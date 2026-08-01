// Native Kimi-K3 text-model support for Lucebox Hub.
//
// This is intentionally split into three model-neutral boundaries:
//   * GGUF loading owns tensor metadata/storage only;
//   * KimiK3Cache owns recurrent/attention state only; and
//   * kimi_k3_forward owns the architecture graph only.
//
// Routed-expert placement can therefore replace the resident expert tensors
// with the common MoE stream engine without changing KDA, MLA, AttnRes, or the
// public ModelBackend contract.

#pragma once

#include "common/moe_hybrid_storage.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeHybridRoutingStats;

class MoeHybridStreamEngine;
class MoeStreamDualOwnerExecutor;
struct MoeStreamDualOwnerPolicy;

struct KimiK3Layer {
    bool recurrent = false;

    ggml_tensor * attn_norm       = nullptr;
    ggml_tensor * ffn_norm        = nullptr;
    ggml_tensor * attn_res_score  = nullptr;
    ggml_tensor * ffn_res_score   = nullptr;

    // KDA (recurrent) attention.
    ggml_tensor * wq              = nullptr;
    ggml_tensor * wk              = nullptr;
    ggml_tensor * wv              = nullptr;
    ggml_tensor * wo              = nullptr;
    ggml_tensor * ssm_q_conv      = nullptr;
    ggml_tensor * ssm_k_conv      = nullptr;
    ggml_tensor * ssm_v_conv      = nullptr;
    ggml_tensor * ssm_f_a         = nullptr;
    ggml_tensor * ssm_f_b         = nullptr;
    ggml_tensor * ssm_beta        = nullptr;
    ggml_tensor * ssm_a           = nullptr;
    ggml_tensor * ssm_dt_b        = nullptr;
    ggml_tensor * ssm_g           = nullptr;
    ggml_tensor * ssm_o_norm      = nullptr;

    // MLA attention. Kimi-K3 uses the absorbed K-only cache when wk_b/wv_b
    // are present, which is the layout emitted by the official converter.
    ggml_tensor * wq_a            = nullptr;
    ggml_tensor * wq_a_norm       = nullptr;
    ggml_tensor * wq_b            = nullptr;
    ggml_tensor * wkv_a_mqa       = nullptr;
    ggml_tensor * wkv_a_norm      = nullptr;
    ggml_tensor * wk_b            = nullptr;
    ggml_tensor * wv_b            = nullptr;
    ggml_tensor * wkv_b           = nullptr;
    ggml_tensor * wqkv_gate       = nullptr;

    // Dense FFN (leading dense blocks).
    ggml_tensor * ffn_gate        = nullptr;
    ggml_tensor * ffn_up          = nullptr;
    ggml_tensor * ffn_down        = nullptr;

    // Latent routed MoE + full-width shared expert.
    ggml_tensor * ffn_gate_inp    = nullptr;
    ggml_tensor * ffn_exp_probs_b = nullptr;
    ggml_tensor * ffn_gate_exps   = nullptr;
    ggml_tensor * ffn_up_exps     = nullptr;
    ggml_tensor * ffn_down_exps   = nullptr;
    ggml_tensor * ffn_routed_down = nullptr;
    ggml_tensor * ffn_routed_up   = nullptr;
    ggml_tensor * ffn_routed_norm = nullptr;
    ggml_tensor * ffn_gate_shexp  = nullptr;
    ggml_tensor * ffn_up_shexp    = nullptr;
    ggml_tensor * ffn_down_shexp  = nullptr;
};

struct KimiK3Weights {
    // ctx/buf alias the first entries for compatibility with the original
    // single-file loader. Split GGUFs retain one metadata context and optional
    // resident backend buffer per shard.
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    std::vector<ggml_context *> contexts;
    std::vector<ggml_backend_buffer_t> buffers;
    std::vector<std::string> shard_paths;

    // The routed stacks may remain file-backed. Regions use MoE-layer-local
    // indices [0, n_layer-n_dense_lead), independent of model layer numbers.
    std::vector<LayerExpertRegions> streamed_layer_regions;
    size_t max_streamed_expert_bytes = 0;
    bool routed_experts_streamed = false;

    ggml_tensor * tok_embd         = nullptr;
    ggml_tensor * output_norm      = nullptr;
    ggml_tensor * output           = nullptr;
    ggml_tensor * output_res_score = nullptr;
    std::vector<KimiK3Layer> layers;

    int n_layer       = 0;
    int n_embd        = 0;
    int n_ff          = 0;
    int n_vocab       = 0;
    int n_ctx_train   = 0;
    int n_head        = 0;
    int n_expert      = 0;
    int n_expert_used = 0;
    int n_ff_exp      = 0;
    int n_expert_latent = 0;
    int n_expert_shared = 0;
    int n_dense_lead  = 0;

    int ssm_d_conv    = 0;
    int kda_head_dim  = 0;
    int q_lora_rank   = 0;
    int kv_lora_rank  = 0;
    int mla_k_head_dim = 0;
    int mla_v_head_dim = 0;
    int rope_dim       = 0;
    int attn_res_block_size = 0;

    float rms_eps              = 1.0e-5f;
    float kda_gate_lower_bound = -INFINITY;
    float expert_weights_scale = 1.0f;
    bool  expert_weights_norm  = true;
    int   expert_gating_func   = 2; // sigmoid
    float situ_beta            = 4.0f;
    float situ_linear_beta     = 25.0f;
    int32_t eos_token_id       = 2;
};

struct KimiK3LayerCache {
    ggml_tensor * conv_state = nullptr; // [d_conv-1, 3*d_inner], F32
    ggml_tensor * ssm_state  = nullptr; // [head_dim, head_dim, n_head], F32
    ggml_tensor * mla_k      = nullptr; // [kv_rank+rope_dim, 1, max_ctx], F16

    // Speculative-decode state.  ReplaySSM captures the much smaller
    // pre-KDA activation for every verify row, then re-runs only the accepted
    // recurrent transitions at commit time.  The snapshots are a failure-safe
    // for the commit graph; ordinary rejected verification leaves the live
    // recurrent tensors untouched and therefore needs no restore copy.
    ggml_tensor * conv_state_snap = nullptr;
    ggml_tensor * ssm_state_snap  = nullptr;
    ggml_tensor * replay_input    = nullptr; // [hidden, max_verify_tokens], F32
};

struct KimiK3Cache {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<KimiK3LayerCache> layers;
    int max_ctx = 0;
    int cur_pos = 0;
    int max_verify_tokens = 0;
    int snapshot_pos = -1;
    int replay_base_pos = -1;
    int replay_n_tokens = 0;
    bool snapshot_valid = false;
    bool replay_valid = false;
    bool recurrent_state_pristine = false;
};

// Model-neutral forward result shape used by the Kimi DFlash adapter.  Capture
// rows are capture-major, then token-major:
//   [capture_layer][token][hidden].
struct KimiK3ForwardOptions {
    const std::vector<int> * capture_layer_ids = nullptr;
    bool capture_replay = false;
    bool read_logits = false;
    bool read_argmax = true;
};

struct KimiK3ForwardResult {
    std::vector<float> logits;
    std::vector<int32_t> argmax;
    std::vector<float> captured_hidden;
};

bool load_kimi_k3_gguf(const std::string & path,
                       ggml_backend_t backend,
                       KimiK3Weights & out,
                       bool stream_routed_experts = false);
void free_kimi_k3_weights(KimiK3Weights & w);

bool create_kimi_k3_cache(ggml_backend_t backend,
                          const KimiK3Weights & w,
                          int max_ctx,
                          KimiK3Cache & out,
                          int max_verify_tokens = 0);
void reset_kimi_k3_cache(KimiK3Cache & cache);
void free_kimi_k3_cache(KimiK3Cache & cache);

// Batch forward used for target verification.  With capture_replay=true the
// recurrent state is read-only: KDA inputs are persisted for a later
// kimi_k3_replay_commit(), while MLA writes remain position-indexed and become
// invisible simply by restoring cur_pos.
bool kimi_k3_forward(ggml_backend_t backend,
                     const KimiK3Weights & w,
                     KimiK3Cache & cache,
                     const std::vector<int32_t> & tokens,
                     int base_pos,
                     const KimiK3ForwardOptions & options,
                     KimiK3ForwardResult & result,
                     MoeHybridStreamEngine * stream_engine = nullptr,
                     MoeStreamDualOwnerExecutor * dual_stream_executor = nullptr,
                     const MoeStreamDualOwnerPolicy * stream_owner_policy = nullptr,
                     MoeHybridRoutingStats * routing_stats = nullptr);

bool kimi_k3_replay_snapshot(ggml_backend_t backend, KimiK3Cache & cache);
bool kimi_k3_replay_restore(ggml_backend_t backend, KimiK3Cache & cache);
bool kimi_k3_replay_commit(ggml_backend_t backend,
                           const KimiK3Weights & w,
                           KimiK3Cache & cache,
                           int base_pos,
                           int commit_n);

// Compatibility wrapper for the ordinary one-token AR path. Speculative
// verification calls kimi_k3_forward directly with a bounded token batch.
bool kimi_k3_step(ggml_backend_t backend,
                  const KimiK3Weights & w,
                  KimiK3Cache & cache,
                  int32_t token,
                  int position,
                  std::vector<float> & logits,
                  MoeHybridStreamEngine * stream_engine = nullptr,
                  MoeStreamDualOwnerExecutor * dual_stream_executor = nullptr,
                  const MoeStreamDualOwnerPolicy * stream_owner_policy = nullptr,
                  MoeHybridRoutingStats * routing_stats = nullptr);

} // namespace dflash::common
