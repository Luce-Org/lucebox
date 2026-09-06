// Internal-only shared header for dflash::common library sources.
// Not installed, not exposed in the public API.

#pragma once
#define DFLASH_INTERNAL_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "dflash27b.h"
#include "common/paged_attention_config.h"

namespace dflash::common {

struct MoeHybridStorage;

// Single source of truth for error reporting.
// All loaders / graph builders push into this via set_last_error(...).
void set_last_error(std::string msg);

// ─── Target weights (Qwen3.5-27B, qwen35 hybrid, Q4_K_M in ggml context) ──
//
// Qwen3.5 uses two kinds of blocks interleaved:
//   - FULL ATTENTION block  (every `full_attention_interval`-th layer, =4):
//       attn_norm, wq, wk, wv, wo, q_norm, k_norm + FFN tensors
//       (M-RoPE applied with rope_sections [11,11,10,0] — rope dims=64 of head_dim=256)
//   - GATED DELTANET block (all other layers, ~3 out of every 4):
//       attn_norm, wqkv (fused), wqkv_gate (the "z" projection),
//       delta-net per-head parameters (beta, gate, conv), plus FFN tensors.
//
// We keep ONE struct with all possible fields and leave unused ones nullptr.
// Actual tensor names in unsloth's GGUF are read via gguf_find_tensor() in
// the loader; see task #11.

struct TargetLayer {
    // Shared
    ggml_tensor * attn_norm      = nullptr;  // [hidden]
    ggml_tensor * attn_post_norm = nullptr;  // [hidden]  (post-block norm before FFN)
    ggml_tensor * ffn_norm       = nullptr;  // [hidden]
    ggml_tensor * w_gate         = nullptr;  // [hidden, intermediate]
    ggml_tensor * w_up           = nullptr;  // [hidden, intermediate]
    ggml_tensor * w_down         = nullptr;  // [intermediate, hidden]

    // Full-attention block (non-null for layers where (il+1) % 4 == 0)
    ggml_tensor * wq             = nullptr;  // [hidden, q_dim]
    ggml_tensor * wk             = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wv             = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wo             = nullptr;  // [q_dim, hidden]
    ggml_tensor * q_norm         = nullptr;  // [head_dim]
    ggml_tensor * k_norm         = nullptr;  // [head_dim]

    // Gated DeltaNet block (non-null for the other ~3/4 of layers)
    ggml_tensor * wqkv           = nullptr;  // fused Q/K/V projection
    ggml_tensor * wqkv_gate      = nullptr;  // the "z" projection
    ggml_tensor * ssm_conv1d     = nullptr;  // [kernel, dim]  depthwise causal conv
    ggml_tensor * ssm_beta       = nullptr;  // per-token beta input projection
    ggml_tensor * ssm_alpha      = nullptr;  // per-token alpha input projection
    ggml_tensor * ssm_a          = nullptr;  // [dt_rank] per-head -A parameter
    ggml_tensor * ssm_dt_bias    = nullptr;  // [dt_rank] per-head alpha bias
    ggml_tensor * ssm_norm       = nullptr;  // [head_v_dim]
    ggml_tensor * ssm_out        = nullptr;  // output projection after delta-net
    // Zero-copy stacked projections (set by the loader when the two source
    // tensors share a type and were placed back to back in the weight buffer):
    //   wqkv_z: rows [0, n_z) = wqkv_gate (z), rows [n_z, ...) = wqkv
    //   ssm_ba: rows [0, dt_rank) = ssm_beta, rows [dt_rank, ...) = ssm_alpha
    // One GEMV each instead of two; nullptr when stacking was not possible.
    ggml_tensor * wqkv_z         = nullptr;
    ggml_tensor * ssm_ba         = nullptr;
    // Fused raw-gate GDN kernel parameters: f32 [2 * dt_rank] = [dt_bias | A],
    // one small GPU tensor per DeltaNet layer (src[9] of the GDN op).
    ggml_tensor * ssm_gate_ba    = nullptr;

    // BailingMoE3 / Ling 3 KDA. Unlike Qwen3.5's fused projection and
    // convolution, Ling projects and convolves Q, K, and V independently and
    // uses a vector-valued decay gate (KDA) per head dimension.
    ggml_tensor * ssm_conv1d_q   = nullptr;  // [conv, 1, d_inner, 1]
    ggml_tensor * ssm_conv1d_k   = nullptr;  // [conv, 1, d_inner, 1]
    ggml_tensor * ssm_conv1d_v   = nullptr;  // [conv, 1, d_inner, 1]
    ggml_tensor * ssm_f_a        = nullptr;  // [hidden, d_inner]
    ggml_tensor * ssm_g_a        = nullptr;  // [hidden, d_inner]

    // BailingMoE3 / Ling 3 MLA. The latent K/V cache stores
    // [kv_lora_rank + rope_dim] for K and [kv_lora_rank] for V.
    ggml_tensor * attn_q_a       = nullptr;
    ggml_tensor * attn_q_a_norm  = nullptr;
    ggml_tensor * attn_q_b       = nullptr;
    ggml_tensor * attn_kv_a_mqa  = nullptr;
    ggml_tensor * attn_kv_a_norm = nullptr;
    ggml_tensor * attn_k_b       = nullptr;
    ggml_tensor * attn_v_b       = nullptr;

    // MoE FFN (qwen35moe only; nullptr on dense qwen35)
    ggml_tensor * ffn_gate_inp       = nullptr;  // [hidden, n_expert] router
    ggml_tensor * ffn_gate_exps      = nullptr;  // [hidden, n_ff_exp, n_expert]
    ggml_tensor * ffn_up_exps        = nullptr;  // [hidden, n_ff_exp, n_expert]
    ggml_tensor * ffn_down_exps      = nullptr;  // [n_ff_exp, hidden, n_expert]
    ggml_tensor * ffn_exp_probs_b    = nullptr;  // [n_expert] router correction bias
    ggml_tensor * ffn_gate_up_exps   = nullptr;  // [hidden, 2*n_ff_exp, n_expert] optional fused gate/up
    ggml_tensor * ffn_gate_inp_shexp = nullptr;  // [hidden] shared-expert scalar gate
    ggml_tensor * ffn_gate_shexp     = nullptr;  // [hidden, n_ff_shexp]
    ggml_tensor * ffn_up_shexp       = nullptr;  // [hidden, n_ff_shexp]
    ggml_tensor * ffn_down_shexp     = nullptr;  // [n_ff_shexp, hidden]

    // NVFP4 per-tensor weight scales (optional; 1.0f = no scaling).
    // Each corresponds to a weight tensor above: result = mul_mat(w, x) * scale.
    // Stored as host-side floats (read from the GGUF at load time) and applied
    // via ggml_scale() — a compile-time scalar multiply with zero extra kernel
    // launches, unlike ggml_mul() with a [1]-shaped GPU tensor which adds 768
    // kernel launches per forward pass and causes catastrophic overhead in
    // batched DDTree verify mode.
    float w_gate_s       = 1.0f;
    float w_up_s         = 1.0f;
    float w_down_s       = 1.0f;
    float wq_s           = 1.0f;
    float wk_s           = 1.0f;
    float wv_s           = 1.0f;
    float wo_s           = 1.0f;
    float wqkv_s         = 1.0f;
    float wqkv_gate_s    = 1.0f;
    float ssm_beta_s     = 1.0f;
    float ssm_alpha_s    = 1.0f;
    float ssm_out_s      = 1.0f;
    float ffn_gate_inp_s       = 1.0f;
    float ffn_gate_exps_s      = 1.0f;
    float ffn_up_exps_s        = 1.0f;
    float ffn_down_exps_s      = 1.0f;
    float ffn_gate_up_exps_s   = 1.0f;
    float ffn_gate_inp_shexp_s = 1.0f;
    float ffn_gate_shexp_s     = 1.0f;
    float ffn_up_shexp_s       = 1.0f;
    float ffn_down_shexp_s     = 1.0f;

    // Optional per-layer activation limits used by late Ling 3 blocks.
    // Zero means ordinary SwiGLU.
    float ffn_swiglu_clamp_exp   = 0.0f;
    float ffn_swiglu_clamp_shexp = 0.0f;
};

// CPU-side embedder: keeps a mmap of the GGUF alive and knows how to
// dequantize individual rows of the quantized tok_embd tensor on demand.
// This matches llama.cpp's behavior of running embedding get_rows on CPU
// (because CUDA's get_rows doesn't support k-quants), so we never need to
// upload the 682 MiB token embedding to VRAM.
struct CpuEmbedder {
    void *           mmap_addr = nullptr;
    size_t           mmap_len  = 0;
#if defined(_WIN32)
    HANDLE           mmap_hfile = INVALID_HANDLE_VALUE;
    HANDLE           mmap_hmap  = nullptr;
#else
    int              mmap_fd   = -1;
#endif
    const uint8_t *  tok_embd_bytes = nullptr;  // into the mmap region
    ggml_type        tok_embd_type  = GGML_TYPE_COUNT;
    int64_t          n_embd = 0;
    int64_t          n_vocab = 0;
    size_t           row_bytes = 0;             // bytes per row in the quant format
    std::vector<uint8_t> tok_embd_owned;        // optional owned tok_embd payload

    ~CpuEmbedder();
    // Dequantize N rows specified by `ids` into `out_f32` (shape [n_embd, n]).
    // Values are written contiguously row-major (n_embd fast axis).
    bool embed(const int32_t * ids, int n, float * out_f32) const;
};

struct TargetWeights {
    ggml_context *        ctx     = nullptr;
    ggml_context *        stack_ctx = nullptr;  // owns the stacked alias tensors
    ggml_context *        gate_ctx  = nullptr;  // owns the [dt_bias | A] gate tensors
    ggml_backend_buffer_t gate_buf  = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // CPU-side embedding table (zero GPU cost).
    CpuEmbedder           embedder;

    ggml_tensor * tok_embd = nullptr;        // [hidden, vocab] (metadata only; data NOT on GPU)
    std::vector<TargetLayer> layers;         // size = 64
    ggml_tensor * out_norm = nullptr;        // [hidden]
    ggml_tensor * output   = nullptr;        // [hidden, vocab]  (lm_head)
    std::shared_ptr<MoeHybridStorage> moe_hybrid; // optional hybrid storage (hot/cold expert split)

    // Metadata from GGUF (validated at load time)
    int full_attention_interval = 4;
    int rope_sections[4]        = {11, 11, 10, 0};
    int n_embd_head_k           = 256;  // key_length
    int n_embd_head_v           = 256;  // value_length
    int n_head                  = 24;
    int n_head_kv               = 4;
    int n_layer                 = 64;
    int n_embd                  = 5120;
    int n_ff                    = 17408;
    int n_ff_exp                = 0;
    int n_ff_shexp              = 0;
    int n_expert                = 0;
    int n_expert_used           = 0;
    int n_expert_groups         = 1;
    int n_expert_groups_used    = 1;
    int n_vocab                 = DFLASH27B_TARGET_VOCAB;
    int rope_dimension_count    = 64;
    float rope_theta            = 10000000.0f;
    float rms_eps               = 1e-6f;
    float expert_weights_scale  = 1.0f;
    int expert_gating_func      = 1;    // 1=softmax, 2=sigmoid (llama.cpp enum values)
    bool is_moe                 = false;
    bool is_bailingmoe3         = false;
    bool expert_weights_norm    = true;
    int n_layer_dense_lead      = 0;

    // BailingMoE3 / Ling 3 architecture parameters.
    int kda_head_dim            = 0;
    int mla_qk_head_dim         = 0;
    int mla_v_head_dim          = 0;
    int kv_lora_rank            = 0;
    int q_lora_rank             = 0;
    float kda_gate_lower_bound  = 0.0f;
    int ssm_d_conv              = 4;
    int ssm_d_inner             = 6144;
    int ssm_d_state             = 128;
    int ssm_dt_rank             = 48;
    int ssm_n_group             = 16;

    // EOS token ids loaded from the GGUF tokenizer metadata
    // (`tokenizer.ggml.eos_token_id` and `tokenizer.ggml.eot_token_id`).
    // -1 = key absent in this GGUF; the runtime EOS check guards both
    // comparands with `>= 0` so the sentinel never matches a real token.
    int32_t eos_id      = -1;
    int32_t eos_chat_id = -1;

    // DFlash noise mask token ID (from target tokenizer, used by draft model).
    // Default: Qwen tokenizer's mask token. Overridden by GGUF metadata if available.
    int32_t mask_token_id = DFLASH27B_DRAFT_MASK_TOKEN_ID;

    // Target layer IDs captured for the DFlash draft model.
    // Computed from n_layer at load time: step = (n_layer - 2) / (N - 1),
    // ids[k] = 1 + k * step.  E.g. 27B→{1,16,31,46,61}, 9B→{1,8,15,22,29}.
    int n_capture_layers = DFLASH27B_DRAFT_N_TARGET_LAYERS;
    int capture_layer_ids[DFLASH27B_DRAFT_N_TARGET_LAYERS] = {1, 16, 31, 46, 61};
};

// Check if a token is an end-of-sequence marker for the given target weights.
inline bool is_eos_tok(int tok, const TargetWeights & w) {
    return (w.eos_chat_id >= 0 && tok == w.eos_chat_id)
        || (w.eos_id      >= 0 && tok == w.eos_id);
}

struct TargetLoadPlan {
    int  layer_begin = 0;     // inclusive
    int  layer_end   = -1;    // exclusive; <0 means all layers
    bool load_output = true;  // output_norm + lm_head
    bool skip_expert_tensors = false;  // skip ffn_*_exps from GPU (for hybrid MoE split load)
    bool metadata_only = false;        // parse tensor descriptors/scales without GPU allocation
    bool expert_metadata_only = false; // keep only routed expert tensor metadata; upload nothing
};

// Load a Q4_K_M target model from a GGUF file on disk.
// Returns false and sets last_error on failure.
bool load_target_gguf(const std::string & path,
                      ggml_backend_t backend,
                      TargetWeights & out);

bool load_target_gguf_partial(const std::string & path,
                              ggml_backend_t backend,
                              const TargetLoadPlan & plan,
                              TargetWeights & out);

// Load the autoregressive trunk of a BailingMoE3 GGUF (Ling 3.x). Embedded
// NextN/MTP blocks are intentionally ignored by this baseline backend.
bool load_bailingmoe3_gguf(const std::string & path,
                           ggml_backend_t backend,
                           TargetWeights & out);

void free_target_weights(TargetWeights & w);

// ─── Draft weights (z-lab DFlash, bf16) ───────────────────────────

// DFlash 2 grouped dynamic causal conv (two taps over the draft block, one
// instance before/after attention and one before/after the MLP):
//   dyn      = proj @ x_norm                      [2*K*groups, q_len]
//   prepare  = sum_k (base[0][k] + dyn[0][k]) * shift_k(x_norm)
//   finish   = sum_k (base[1][k] + dyn[1][k]) * shift_k(sub_block_out)
// base is per element, dyn per group of conv_group_size elements.
struct DraftConvWeights {
    ggml_tensor * base = nullptr;   // [hidden, K, 2] f32
    ggml_tensor * proj = nullptr;   // [hidden, 2*K*groups]
    bool present() const { return base != nullptr && proj != nullptr; }
};

struct DraftLayer {
    ggml_tensor * attn_norm;
    ggml_tensor * ffn_norm;
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * attn_gate = nullptr;  // optional Laguna XS 2.1 attention gate
    ggml_tensor * q_norm;
    ggml_tensor * k_norm;
    ggml_tensor * w_gate;
    ggml_tensor * w_up;
    ggml_tensor * w_down;
    DraftConvWeights attn_conv;         // optional DFlash 2 conv around attention
    DraftConvWeights mlp_conv;          // optional DFlash 2 conv around the MLP
    bool is_swa = false;  // true for SWA layers (Qwen3.6 pattern)
    bool attn_gate_per_head = false;
};

struct DraftDominoWeights {
    bool enabled = false;
    int  gru_hidden_dim = 0;
    int  emb_dim = 0;
    int  vocab_size = 0;

    ggml_tensor * start       = nullptr;  // [gru_hidden_dim] f32
    ggml_tensor * gru_w_ih    = nullptr;  // [n_embd, 3*gru_hidden_dim]
    ggml_tensor * gru_w_hh    = nullptr;  // [gru_hidden_dim, 3*gru_hidden_dim]
    ggml_tensor * gru_b_ih    = nullptr;  // [3*gru_hidden_dim] f32
    ggml_tensor * gru_b_hh    = nullptr;  // [3*gru_hidden_dim] f32
    ggml_tensor * head_w1     = nullptr;  // [n_embd+gru_hidden_dim, emb_dim]
    ggml_tensor * head_b1     = nullptr;  // [emb_dim] f32
    ggml_tensor * head_w2     = nullptr;  // [emb_dim, vocab_size]
    ggml_tensor * head_b2     = nullptr;  // [vocab_size] f32
};

struct DraftDSparkWeights {
    bool enabled = false;
    int  markov_rank = 0;
    int  vocab_size = 0;
    int  confidence_dim = 0;

    ggml_tensor * markov_w1    = nullptr;  // [markov_rank, vocab_size]
    ggml_tensor * markov_w2    = nullptr;  // [markov_rank, vocab_size]
    ggml_tensor * confidence_w = nullptr;  // [confidence_dim, 1]
    ggml_tensor * confidence_b = nullptr;  // [1] f32
};

// DFlash 2 candidate selector: top-k candidates per block position from the
// target lm_head logits, then one path through them scored by a low-rank
// bigram form  unary[c] + <pred[prev] * hproj(h), succ[c]>.
struct DraftSelectorWeights {
    bool enabled = false;
    int  rank    = 0;
    int  top_k   = 0;
    ggml_tensor * hproj   = nullptr;   // [hidden, rank]
    ggml_tensor * pred_cb = nullptr;   // [rank, vocab]  predecessor codebook
    ggml_tensor * succ_cb = nullptr;   // [rank, vocab]  successor codebook
};

struct DraftWeights {
    ggml_context *    ctx = nullptr;
    ggml_backend_t    backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor *          fc          = nullptr;   // [5*hidden, hidden]
    ggml_tensor *          hidden_norm = nullptr;   // [hidden]
    std::vector<ggml_tensor *> aux_hidden_norms;    // optional [hidden] per captured target layer
    bool context_kv_layer_norm = false;             // Laguna DFlash: per-layer input norm before context K/V
    std::vector<DraftLayer> layers;                 // size = n_layer
    ggml_tensor *          out_norm    = nullptr;   // [hidden]

    // Architecture metadata (populated by loader).
    int n_layer   = DFLASH27B_DRAFT_LAYERS;           // 5
    int n_head    = DFLASH27B_TARGET_N_HEADS;          // 32
    int n_head_kv = DFLASH27B_TARGET_N_KV_HEADS;       // 8
    int head_dim  = DFLASH27B_TARGET_HEAD_DIM;         // 128
    int n_embd    = DFLASH27B_TARGET_HIDDEN;           // 5120
    int n_ff      = DFLASH27B_TARGET_INTERMEDIATE;     // 17408
    int swa_window = 0;                 // sliding window size (0 = disabled)
    bool swa_pattern_loaded = false;    // GGUF supplied sliding_window_pattern
    float rope_theta = 0.0f;  // RoPE frequency base (must come from GGUF)

    // YaRN rope scaling (populated by loader; 0 = disabled / plain RoPE).
    float rope_freq_scale = 1.0f;   // 1/factor (e.g. 1/64 for factor=64)
    float rope_ext_factor = 0.0f;   // >0 enables YaRN interpolation
    float rope_attn_factor = 1.0f;
    float rope_beta_fast  = 0.0f;
    float rope_beta_slow  = 0.0f;
    int   rope_n_ctx_orig = 0;      // original_max_position_embeddings

    // DFlash draft-specific config (populated by loader or set by caller).
    int block_size      = DFLASH27B_DRAFT_BLOCK_SIZE;       // tokens per draft step (16 or 10)
    int n_target_layers = DFLASH27B_DRAFT_N_TARGET_LAYERS;  // captured target layers (5)
    std::vector<int> capture_layer_ids;                     // explicit captured target-layer ids (GGUF dflash.target_layer_ids); empty = derive from count
    int mask_token_id   = DFLASH27B_DRAFT_MASK_TOKEN_ID;    // noise mask token

    // Optional Domino causal correction head. When present, greedy chain
    // speculative decode corrects each draft token with a lightweight GRU
    // conditioned on the realized prefix before target verification.
    DraftDominoWeights domino;

    // Optional DSpark/DeepSpec-style Markov correction head. When present,
    // greedy chain decode adds a low-rank previous-token bias before argmax.
    DraftDSparkWeights dspark;

    // Optional DFlash 2 pieces: dynamic convs live in the layers, the
    // selector replaces argmax/markov projection for the drafted chain.
    int conv_kernel_size = 0;   // 0 = no dynamic convs
    int conv_group_size  = 0;
    DraftSelectorWeights selector;
};

bool load_draft_safetensors(const std::string & path,
                            ggml_backend_t backend,
                            DraftWeights & out,
                            const TargetWeights * target = nullptr);

// Load a Q8_0 (or F16) draft model from a GGUF file on disk.
// Alternative to load_draft_safetensors for quantized drafts.
// If `target` is non-null, draft dims (n_embd, mask_token_id, etc.) are
// cross-checked / populated from the target model.
bool load_draft_gguf(const std::string & path,
                     ggml_backend_t backend,
                     DraftWeights & out,
                     const TargetWeights * target = nullptr);

void free_draft_weights(DraftWeights & w);

// ─── Target cache (persistent state between forward calls) ────────

// Pre-allocated, backend-resident state that persists across decode steps.
// Created once via create_target_cache() and threaded through every
// build_qwen35_graph() call.
struct TargetCache {
    ggml_context *        base_ctx     = nullptr;
    ggml_backend_buffer_t base_buf     = nullptr;
    ggml_context *        rollback_ctx = nullptr;
    ggml_backend_buffer_t rollback_buf = nullptr;
    ggml_backend_t        backend  = nullptr;

    int max_ctx  = 0;         // max tokens in the KV cache
    int cur_pos  = 0;         // number of tokens already committed
    int last_tok = -1;        // post-prefill / post-decode argmax; decode seed.
                              // Used by prefix-cache RESTORE to bridge an
                              // empty-suffix prefill into the decode loop.

    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;

    // Concurrent-serving slot count (--max-concurrency). 1 = classic single-sequence
    // cache. When > 1, ssm_state/conv_state carry a trailing slot axis and the
    // paged metadata tensors widen to n_seq_slots columns.
    int n_seq_slots = 1;

    // When true, K is FWHT-rotated in the graph before writing to the
    // standard-type cache (Q4_0/Q8_0/etc), and Q is rotated at attention
    // time. This gives TurboQuant-style outlier spreading with fast FA
    // kernels that work on all GPU architectures.
    bool kv_k_rotated = false;

    // Full-attention KV cache: one K and one V per full-attention layer.
    // Layout: [head_dim, max_ctx, n_head_kv] f16, contiguous per layer.
    std::vector<ggml_tensor *> attn_k;   // size = n_full_attn_layers (16)
    std::vector<ggml_tensor *> attn_v;

    // Gated DeltaNet recurrent state: one per delta-net layer.
    // ssm_state: [S_v, S_v, H_v, n_seq_slots] f32  (head_v_dim^2 × num_v_heads)
    // conv_state: [(kernel-1), conv_channels, n_seq_slots] f32
    // where conv_channels = d_inner + 2 * n_group * d_state.
    // n_seq_slots is 1 for the classic single-sequence cache (identical
    // layout to the historical 3D/2D tensors); slot s of a multi-slot cache
    // is the contiguous slab at index s of the trailing axis.
    std::vector<ggml_tensor *> ssm_state;    // size = n_delta_layers (48)
    std::vector<ggml_tensor *> conv_state;

    // Snapshot buffers for speculative decoding rollback. Sized identically
    // to ssm_state/conv_state above. Populated by snapshot_ssm_state() and
    // restored by restore_ssm_state().
    std::vector<ggml_tensor *> ssm_state_snap;
    std::vector<ggml_tensor *> conv_state_snap;

    // Per-step SSM + conv inputs captured during a verify forward when
    // QwenGraphInputs::capture_delta_intermediate is true. Populated by
    // in-graph ggml_cpy ops in build_delta_net_block so their data lives in
    // persistent cache memory (not tracked by the per-call gallocr), matching
    // SGLang's mamba_caches.intermediate_ssm / intermediate_conv_window pattern.
    //
    //   ssm_intermediate: [S_v, S_v, H_v, max_q_len], checkpoint dtype
    //     (Q8_0 for direct caches, F16 for migrated single-target caches, or
    //     F32 for opt-in exact rollback), one per delta layer.
    //     Element t on axis 3 holds the DeltaNet recurrent state after
    //     processing verify token t. Spec decode commits t = commit_n - 1.
    //   conv_input_cache: normally [(kernel-1) + max_q_len, conv_channels]
    //     f32, one per delta layer, holding the full concat fed to
    //     ggml_ssm_conv. In SpecLA mode these are [conv_channels, max_q_len]
    //     strided views into conv_factor_all and hold raw current inputs.
    std::vector<ggml_tensor *> ssm_intermediate;    // size = n_delta (48)
    std::vector<ggml_tensor *> conv_input_cache;    // size = n_delta (48)
    std::vector<ggml_tensor *> conv_input_cache_alt;// SpecLA factor bank 1
    ggml_tensor * conv_factor_all = nullptr;
    ggml_tensor * conv_factor_all_alt = nullptr;

    // SpecLA factor buffers (allocated instead of ssm_intermediate when
    // DFLASH_SPECLA=1 on the single-target path). Two token-major banks let a
    // verify consume the preceding accepted path while writing its own raw
    // factors without aliasing:
    //   factor_k_all:     [S_k, H_v, n_delta, max_q_len] f32
    //   factor_v_new_all: [S_v, H_v, n_delta, max_q_len] f32
    //   factor_g_ps_all:  [H_v, n_delta, max_q_len] f32
    //   conv_factor_all:  [conv_channels, n_delta, max_q_len] f32
    // The per-layer vectors below are persistent VIEWS into these (created
    // after buffer allocation), shaped like independent per-layer buffers so
    // the capture path treats them exactly like other cache tensors.
    ggml_tensor * factor_k_all = nullptr;
    ggml_tensor * factor_v_new_all = nullptr;
    ggml_tensor * factor_g_ps_all = nullptr;
    ggml_tensor * factor_k_all_alt = nullptr;
    ggml_tensor * factor_v_new_all_alt = nullptr;
    ggml_tensor * factor_g_ps_all_alt = nullptr;
    std::vector<ggml_tensor *> factor_k;            // view [S_k, H_v, max_q_len]
    std::vector<ggml_tensor *> factor_v_new;        // view [S_v, H_v, max_q_len]
    std::vector<ggml_tensor *> factor_g_ps;         // view [H_v, max_q_len]
    std::vector<ggml_tensor *> factor_k_alt;
    std::vector<ggml_tensor *> factor_v_new_alt;
    std::vector<ggml_tensor *> factor_g_ps_alt;
    // Host-side bank state. pending_bank is read by the next verify; the
    // opposite bank receives that verify's factors.
    int specla_pending_bank = 0;
    int specla_pending_count = 0;
    // Device-side accepted-index scratch and uploaded pointer tables.
    ggml_tensor * specla_idx = nullptr;             // i32 [max_q_len]
    ggml_tensor * specla_state_ptrs = nullptr;      // i64 [n_delta]
    ggml_tensor * specla_conv_state_ptrs = nullptr; // i64 [n_delta]
    // i64 [8]: base pointers for bank0 {k,v,g,conv}, then bank1. HLD kernels
    // write their compact factors here directly, avoiding four cpy nodes per
    // delta layer. Consolidated layout is token-major [t, layer, head, dim].
    ggml_tensor * specla_factor_ptrs = nullptr;

    // Rolling target layer features captured during target forward passes.
    // Single-sequence shape: [5 * hidden, target_feat_cap] bf16. A concurrent
    // tree cache owns one ring per physical sequence slot and one final dead row:
    // [5 * hidden, target_feat_cap * n_seq_slots + 1]. Live row P in slot S
    // maps to S*target_feat_cap + P%target_feat_cap; bucket padding maps to the
    // dead final row because ggml_set_rows does not accept a negative index.
    // target_feat_cap remains the per-sequence ring width.
    ggml_tensor * target_feat = nullptr;
    int target_feat_cap = 0;

    // KVFlash target-QK scorer: last token's post-RoPE (and post-FWHT when
    // kv_k_rotated) query per full-attention layer, written by the graph
    // when QwenGraphInputs::q_capture is set. F32 [head_dim, n_head, n_fa].
    ggml_tensor * q_cap = nullptr;

    // Paged-attention metadata, resident next to the K/V pool (only when the
    // cache was created with paged_attention). Living here instead of as
    // gallocr-managed graph inputs lets decode steps update them append-only
    // — one table entry per new 16-token block plus a 4-byte length per step
    // — instead of re-uploading the whole live table before every compute.
    // Column s of the block table (and entry s of the lens) belongs to
    // sequence slot s; single-sequence caches have exactly one column.
    ggml_tensor * paged_block_table = nullptr;   // I32 [max_blocks_per_seq, n_seq_slots]
    ggml_tensor * paged_kv_seq_lens = nullptr;   // I32 [n_seq_slots]
};

// Snapshot the current SSM+conv state into TargetCache::*_snap tensors.
bool snapshot_ssm_state(TargetCache & c, ggml_backend_t backend);
// Restore the SSM+conv state from the snapshot.
bool restore_ssm_state(TargetCache & c, ggml_backend_t backend);
// Allocate rollback snapshot tensors mirroring live ssm/conv state (MoE path).
bool ensure_ssm_snapshot(TargetCache & c, ggml_backend_t backend);

// ─── Cross-request prefix snapshot (Phase A) ──────────────────────
//
// PrefixSnapshot captures a slim copy of TargetCache state at a
// committed-token boundary so a future request sharing the same prefix
// can restore and skip re-prefilling those tokens.
//
// Slim scope:
//   - attn_k[i], attn_v[i] for every full-attn layer (the actual KV)
//   - ssm_state[i], conv_state[i] for every delta-net layer (recurrent state)
//   - target_feat ring + cur_pos
//
// NOT captured:
//   - ssm_intermediate, conv_input_cache (within-decode rollback buffers,
//     regenerated by the first decode step after restore)
//   - rollback_ctx tensors (snapshots themselves are stateless wrt rollback)
//
// All copies are device-to-device via ggml_backend_tensor_copy. The snapshot
// owns its own ggml_context + backend buffer (allocated lazily on first
// snapshot_target_cache call to a given PrefixSnapshot).
struct PrefixSnapshot {
    int       cur_pos         = 0;
    int       last_tok        = -1;                // post-prefill argmax (decode seed)
    ggml_type kv_k_type       = GGML_TYPE_COUNT;   // for hash-key validation
    int       max_ctx         = 0;                 // for sanity check at restore
    int       target_feat_cap = 0;

    // Snap-backend-resident copies (lazy-allocated; null until first snapshot).
    // On discrete GPUs these live on the CPU backend to avoid VRAM pressure;
    // on unified-memory platforms they stay on the compute backend.
    std::vector<ggml_tensor *> attn_k_snap;     // size n_full_attn (16)
    std::vector<ggml_tensor *> attn_v_snap;
    std::vector<ggml_tensor *> ssm_state_snap;  // size n_delta (48)
    std::vector<ggml_tensor *> conv_state_snap;
    ggml_tensor *               target_feat_snap = nullptr;

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // Phase B: thin-mode snapshots cover only a KV-position range.
    bool is_thin  = false;
    int  kv_start = 0;     // inclusive (only meaningful when is_thin)
    int  kv_end   = 0;     // exclusive (only meaningful when is_thin)
    // When is_thin == true:
    //   - attn_k_snap[i] / attn_v_snap[i] are sized
    //     [HEAD_DIM, kv_end-kv_start, N_HEAD_KV] (smaller than cache).
    //   - ssm_state_snap, conv_state_snap, target_feat_snap are NOT
    //     allocated (THIN snapshots are KV-only).
};

// Snapshot the slim state of `cache` into `snap`. KV tensors are RIGHT-SIZED
// to cache.cur_pos (not max_ctx) to minimize memory. Buffers are reused when
// cur_pos matches the previous snapshot; otherwise freed and reallocated
// (right-sized allocations are tiny — KB for short prefixes). Returns false
// on allocation failure (and sets last_error).
bool snapshot_target_cache(const TargetWeights & w,
                           const TargetCache & cache,
                           ggml_backend_t backend,
                           PrefixSnapshot & snap);

// Restore `cache` from `snap`. cache must already exist (created via
// create_target_cache) and have matching shapes. Sets cache.cur_pos =
// snap.cur_pos. Does NOT touch ssm_intermediate / conv_input_cache —
// those will be repopulated by the first decode step's verify forward.
bool restore_target_cache(const PrefixSnapshot & snap, TargetCache & cache);

// Free the snapshot's GPU buffers.
void free_prefix_snapshot(PrefixSnapshot & snap);

// Thin snapshot: capture only KV slice [kv_start, kv_end).
// SSM/conv/target_feat are not preserved (caller chains thin entries
// onto a thick base via restore_target_cache_chain).
bool snapshot_target_cache_thin(const TargetWeights & w,
                                 const TargetCache & cache,
                                 ggml_backend_t backend,
                                 int kv_start,
                                 int kv_end,
                                 PrefixSnapshot & snap);

// Restore from a thick base then layer in zero or more thin entries.
// thick may be nullptr if you only want the thin layers; in that case
// cache must already hold the right base (only safe for testing).
// Each thin's [kv_start, kv_end) range is copied into cache.attn_k[i] /
// attn_v[i] at the appropriate offset. Out-of-order thins are allowed
// (later thins overwrite earlier ones in overlapping ranges); chain
// caller must walk in time order to be deterministic.
bool restore_target_cache_chain(const PrefixSnapshot * thick,
                                 const PrefixSnapshot * const * thins,
                                 int n_thins,
                                 TargetCache & cache);

// max_verify_tokens controls the per-layer ssm_intermediate and conv_input_cache
// sizes. Default is DFLASH27B_DRAFT_BLOCK_SIZE (16) for chain verify. DDTree
// mode requires max(chain, 1 + tree_budget) to hold the flat tree + root.
// Pass 0 to use the default.
// When prefill_only is true, rollback tensors (snapshots, intermediates) are
// skipped — saving ~1.4 GB on 48 DeltaNet layers. Use migrate_prefill_cache()
// to promote the cache to a full decode cache after prefill.
// `ctx_alloc` (0 = max_ctx): physical token capacity of the attention KV
// tensors. When smaller than max_ctx, a KvFlashPager maps logical positions to
// pool slots and pages cold chunks to host (bounded KV residency); the
// logical context bound stays max_ctx. Recurrent (DeltaNet) state is
// unaffected. When `paged_attention` is true, ctx_alloc may instead be
// max_ctx rounded up to PAGED_BLOCK_SIZE so the last partial page has physical
// rows. Non-paged callers retain the legacy rule that ctx_alloc cannot grow
// the allocation beyond max_ctx.
// `n_seq_slots` (concurrent serving): number of sequence slots the cache
// serves at once. > 1 requires paged_attention; it adds a trailing slot axis
// to the recurrent state, widens the paged metadata to one block-table column
// per slot, and skips the legacy rollback tensors entirely. With
// n_seq_slots > 1 the attention K/V tensors are sized by ctx_alloc (the shared
// pool capacity plus one scratch block) rather than one sequence's max_ctx.
// `concurrent_tree` declares that a paged multi-slot caller owns fixed tree
// scratch and disjoint target-feature rings for GPU promotion.
bool create_target_cache(const TargetWeights & w,
                         int max_ctx,
                         int max_verify_tokens,
                         ggml_backend_t backend,
                         TargetCache & out,
                         bool prefill_only = false,
                         int ctx_alloc = 0,
                         bool paged_attention = false,
                         int n_seq_slots = 1,
                         bool concurrent_tree = false);

// `f32_ssm_intermediates` enables exact per-token checkpoints for the opt-in
// layer-split fast rollback path. The default preserves the established Q8_0
// allocation and avoids its ~1.65 GiB incremental memory cost.
bool create_target_cache_partial(const TargetWeights & w,
                                 int max_ctx,
                                 int max_verify_tokens,
                                 ggml_backend_t backend,
                                 TargetCache & out,
                                 bool prefill_only,
                                 int layer_begin,
                                 int layer_end,
                                 bool allocate_target_feat,
                                 int ctx_alloc = 0,
                                 bool f32_ssm_intermediates = false,
                                 bool paged_attention = false,
                                 int n_seq_slots = 1,
                                 bool concurrent_tree = false);

void free_target_cache(TargetCache & c);

// Zero all state tensors (KV, SSM, conv, target_feat, rollback) in place
// without freeing/reallocating GPU buffers. Used by daemon mode between
// requests to avoid the ~5 s overhead of full cache destruction + recreation.
void reset_target_cache(TargetCache & c);

// Zero only the recurrent state (SSM + conv) without touching the KV cache.
// Much cheaper than reset_target_cache for new requests where KV will be
// overwritten during prefill anyway. Essential between HTTP requests to avoid
// stale delta-net state corrupting subsequent prefills.
void reset_recurrent_state(TargetCache & c);

// Zero one slot's recurrent state (SSM + conv slabs) in a multi-slot cache.
// Called at admission: slots are recycled without a device-side reset, so the
// slab may still hold the previous occupant's state, and the admitted
// prompt's chunked prefill advances its state in this slab directly.
void reset_recurrent_slot(TargetCache & c, int slot);

// Reallocate a prefill-only cache with full rollback tensors, copying all live
// state (KV, SSM, conv, target_feat) device-to-device. Frees the old cache.
bool migrate_prefill_cache(const TargetWeights & w,
                           int max_ctx,
                           int max_verify_tokens,
                           ggml_backend_t backend,
                           TargetCache & cache,
                           bool enable_specla = true);

// Compatibility commit for the fully factorized §4.2 fallback. The production
// HLD route instead keeps raw accepted factors pending and consumes them in
// the next state-resident verify (§5.2). Commits the bank the just-run verify
// wrote (1 - specla_pending_bank) into durable SSM and conv state, so it must
// be called before the host-side bank rotation. accepted_idx is in path order.
bool specla_commit_accepted(TargetCache & cache,
                            ggml_backend_t backend,
                            const int32_t * accepted_idx,
                            int A);


// ─── Target forward graph ─────────────────────────────────────────

// Per-delta-net-layer pointers exposed by the graph for spec-decode rollback.
// Populated when QwenGraphInputs::capture_delta_intermediate is true.
//
// Both tensors are persistent cache buffers (cache.ssm_intermediate[il] and
// cache.conv_input_cache[il]). Their ->data pointers are always valid — the
// graph just runs ggml_cpy ops to fill them during verify. Matches SGLang's
// mamba_caches.intermediate_ssm / intermediate_conv_window pattern:
// persistent memory, not managed by the per-call gallocr.
//
//   ssm_intermediate_states: [S_v, S_v, H_v, q_len] f32
//       Element t on axis 3 holds the DeltaNet state after processing verify
//       token t. Rollback reads offset (commit_n-1) * S_v*S_v*H*elt.
//   conv_input: normally [(kernel-1) + q_len, conv_channels, 1] f32 with the
//       full concat fed to ggml_ssm_conv. In SpecLA mode it is a strided
//       [conv_channels, q_len, 1] view receiving raw current inputs.
struct DeltaNetCapture {
    ggml_tensor * ssm_intermediate_states = nullptr;
    ggml_tensor * conv_input              = nullptr;
    // Concurrent tree direct-commit data. The compact replay log plus the
    // tree conv input can advance accepted recurrent prefixes without a
    // second target-model forward. These are graph-owned outputs.
    ggml_tensor * replay_log              = nullptr;

    // SpecLA factor capture (DFLASH_SPECLA=1, docs/SPECLA.md). Persistent F32
    // aliases into the bank written by this verify. In the HLD path the
    // historical field names hold raw serial-recurrence terms:
    //   factor_k:     [S_k, H_v, max_verify_tokens] — post-l2norm keys
    //   factor_v_new: [S_v, H_v, max_verify_tokens] — raw Delta-rule delta
    //   factor_g_ps:  [H_v, max_verify_tokens]      — raw log-decay g
    // The factorized compatibility route uses corrected ṽ and cumulative g⁺
    // in the same slots. ssm_intermediate_states stays null in SpecLA mode.
    ggml_tensor * factor_k     = nullptr;
    ggml_tensor * factor_v_new = nullptr;
    ggml_tensor * factor_g_ps  = nullptr;
    ggml_tensor * pending_factor_k     = nullptr;
    ggml_tensor * pending_factor_v_new = nullptr;
    ggml_tensor * pending_factor_g     = nullptr;
    ggml_tensor * pending_conv_input   = nullptr;
    ggml_tensor * factor_ptrs           = nullptr;
    int factor_n_layers = 0;
    int factor_layer = -1;
    int pending_bank = 0;
};

// One contiguous prompt chunk on the flattened token axis of a concurrent
// step. Several may be present, one per prefilling slot; segments are dense,
// in order, and precede the decode rows. Full attention needs no per-segment
// dispatch (the ragged read is row-driven); DeltaNet runs one independent
// S=1 recurrence per segment on that slot's own state slab.
struct QwenPrefillSegment {
    int token_offset = 0;
    int n_tokens = 0;
    int seq_slot = 0;
};

struct QwenGraphInputs {
    ggml_tensor * inp_embed;      // [hidden, n_tokens, 1] f32 — pre-embedded by the caller
    ggml_tensor * positions;      // [4 * n_tokens] i32 (M-RoPE needs 4 per token)
    ggml_tensor * attn_mask;      // optional [kv_len, n_tokens_padded] f32 (causal); nullptr for n_tokens==1
    int           n_tokens;       // number of new tokens in this forward
    int           kv_start;       // position where the new tokens begin
    bool          capture_layers; // if true, write captured layer features into cache.target_feat
    bool          capture_delta_intermediate = false; // if true, populate out_delta_captures
    bool          capture_tree_commit = false; // compact recurrent replay log + tree features
    bool          capture_moe_router = false; // if true, expose selected expert ids for MoE layers
    int           fa_window = 0;  // sliding window for FA layers: 0 = full attention
    int           logits_tail_rows = 0; // compute logits only for last n rows; 0 = all
    ggml_tensor * parent_ids = nullptr; // tree: [tree_width,n_tree_seqs] i32
    ggml_tensor * tree_sizes = nullptr; // tree: [n_tree_seqs] i32; 0 = padding tree
    // [n_tokens,n_head_kv] i64 physical destination rows for the
    // ggml_set_rows KV write; step-invariant.
    ggml_tensor * kv_write_rows = nullptr;
    ggml_tensor * paged_block_table = nullptr; // [max_blocks,n_seqs] i32
    // [n_seqs] i32; valid cached K/V tokens per sequence.
    ggml_tensor * paged_kv_seq_lens = nullptr;
    // [n_seqs] i32 mapping compact decode rows to physical cache/state slots.
    // A value of -1 denotes a graph-bucket padding row.
    ggml_tensor * active_slot_ids = nullptr;
    // [n_seqs] i32 gather-safe variant: padding maps to slot zero. Used only
    // for reading the much smaller conv-state slabs; writes use active ids.
    ggml_tensor * state_slot_ids = nullptr;
    // [n_tokens] i32 per-row block-table column for the ragged paged
    // attention read (prefill rows carry their slot, direct decode/tree rows
    // theirs, padding -1). A packed tree also uses this map for every row.
    ggml_tensor * paged_query_seq_ids = nullptr;
    // [n_tokens] i32 per-row inclusive logical position; the kernel clamps
    // each direct row's KV extent to position+1, which IS the causal mask.
    // Packed-tree and padding rows carry -1.
    ggml_tensor * paged_query_positions = nullptr;
    // Optional [n_rows] i32 gather of final-norm rows before the LM head:
    // multi-prompt steps sample scattered rows (each committing segment's
    // last row plus the decode rows), which a tail view cannot express.
    // Non-null overrides logits_tail_rows.
    ggml_tensor * logits_row_indices = nullptr;
    // Optional replay-stable DFlash capture destinations. When present, all
    // captured layers are concatenated once and written with ggml_set_rows.
    // In a packed-tree graph this covers only the durable prefill/AR prefix;
    // the fixed-width tree suffix is returned separately for promotion.
    ggml_tensor * target_feat_rows = nullptr; // [n_direct_tokens] i32
    // Prefill segments on the leading token axis (see QwenPrefillSegment).
    // n_prefill_tokens is their total row count. seq_slot is ignored when
    // segments are present.
    const QwenPrefillSegment * prefill_segments = nullptr;
    int n_prefill_segments = 0;
    // Concurrent-slot serving (paged only):
    //   n_seqs > 1  — batched decode: the token axis is the SEQUENCE axis
    //     (n_tokens == n_seqs, one token per slot). DeltaNet runs with
    //     n_seq_tokens=1 over the full state slabs, and the paged attention
    //     output is permuted from [D,n_seq,Hq] to the dense [D,Hq,n_tokens]
    //     layout before the reshape.
    //   seq_slot — the prefilling slot: its own conv/ssm slab carries the
    //     prompt's chunk-to-chunk recurrent state (reset at admission), and
    //     its block-table column resolves the chunk's paged K/V reads.
    //   paged_max_kv_len — max kv_seq_len over live slots INCLUDING the
    //     prefilling slot's rows written this step, used (256-padded) as the
    //     kernel launch bound instead of kv_start + n_tokens, which is
    //     meaningless across slots.
    //   n_prefill_tokens — packed prefill+decode: the first n_prefill_tokens
    //     rows are the dense concatenation of prefill_segments; optional
    //     trailing n_seqs rows are the compact batched decode. All rows
    //     read the paged pool through one ragged attention call driven by
    //     paged_query_seq_ids/positions; this step's chunk rows are visible
    //     to their own causal reads because the set_rows pool write precedes
    //     attention in the graph. Projections, FFN, norms, and the LM head
    //     run once over the whole batch; only the DeltaNet core splits.
    //     Requires n_tokens == n_prefill_tokens + n_seqs when decode rows
    //     are present. 0 = no prefill segment.
    // Packed steps use logits_row_indices for scattered committing rows and
    // compact decode rows; logits_tail_rows remains the dense-path fallback.
    int  n_seqs = 1;
    // Mixed direct-commit tree graphs place this many one-token mapped AR
    // sequences after any prefill segments and before the fixed-width
    // speculative tree suffix. Their slot IDs share active_slot_ids/
    // state_slot_ids with the tree lanes; prefills own direct slab views.
    int  mapped_ar_seqs = 0;
    int  seq_slot = 0;
    int  paged_max_kv_len = 0;
    int  n_prefill_tokens = 0;
    // Packed paged-tree metadata. Tokens are flattened sequence-major:
    // row = sequence*tree_width + node. tree_scratch_* describe the physical
    // KV scratch slab owned by each physical sequence slot.
    int  tree_width = 0;
    int  tree_scratch_base = 0;
    int  tree_scratch_stride = 0;
    // Capture the LAST token's post-RoPE/post-rotation Q per full-attention
    // layer into cache.q_cap (KVFlash target-QK scorer). Step-invariant:
    // node properties depend only on n_tokens and the layer index.
    bool q_capture = false;

    // SpecLA topology masks for the fully factorized compatibility route.
    ggml_tensor * specla_m_strict = nullptr;
    ggml_tensor * specla_m_incl   = nullptr;
    ggml_tensor * specla_m_eye    = nullptr;
    // Packed heavy-light schedule for the state-resident SpecLA kernels.
    ggml_tensor * specla_hld = nullptr;
    int specla_n_chains = 0;
    int specla_n_waves = 0;
    int specla_n_boundaries = 0;
    int specla_max_parallel_chains = 0;
};

struct QwenGraphOutputs {
    // [vocab, n_logits_rows] f32; row count is selected by the logits
    // projection controls in QwenGraphInputs.
    ggml_tensor * logits;
    // One entry per delta-net layer (48 for qwen35-27b). Only populated when
    // QwenGraphInputs::capture_delta_intermediate is true. Tensors are graph
    // views marked as ggml_set_output() so their data persists after
    // graph_compute; the spec-decode loop reads them host-side for rollback.
    std::vector<DeltaNetCapture> delta_captures;
    // BF16 [n_capture_layers*n_embd, tree_width*n_tree_seqs], packed-tree only.
    ggml_tensor * tree_features = nullptr;
    // One entry per target layer. Populated only when capture_moe_router is
    // true; qwen35 dense layers and non-MoE models leave entries null.
    std::vector<ggml_tensor *> moe_selected;
};

struct QwenLayerPrefnOutputs {
    ggml_tensor * residual = nullptr; // [hidden, n_tokens]
    ggml_tensor * post = nullptr;     // [hidden, n_tokens]
    ggml_tensor * moe_selected = nullptr; // [n_used, n_tokens] i32
    ggml_tensor * moe_weights = nullptr;  // [n_used, n_tokens] f32
};

QwenGraphOutputs build_qwen35_graph(
    ggml_context *         ctx,
    ggml_cgraph *          gf,
    const TargetWeights &  w,
    TargetCache &          cache,
    const QwenGraphInputs & in);

// Build a single-layer forward graph. Mirrors build_qwen35_graph but processes
// only one layer, taking `inp` as the input activation and returning the output.
// Used by layer-segmented prefill to iterate layers as the outer loop.
ggml_tensor * build_qwen35_layer(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,         // [hidden, n_tokens]
    ggml_tensor *         positions,   // [4 * n_tokens] i32
    ggml_tensor *         attn_mask,   // optional
    int                   kv_start,
    int                   n_tokens,
    bool                  capture,
    int                   fa_window = 0,
    ggml_tensor *         q_tail_capture = nullptr,
    int                   q_tail_start = 0,
    ggml_tensor *         kv_write_rows = nullptr,
    ggml_tensor *         parent_ids = nullptr);

// Overload that also exposes the MoE router selection tensor (if MoE layer).
ggml_tensor * build_qwen35_layer(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,
    ggml_tensor *         positions,
    ggml_tensor *         attn_mask,
    int                   kv_start,
    int                   n_tokens,
    bool                  capture,
    int                   fa_window,
    ggml_tensor *         q_tail_capture,
    int                   q_tail_start,
    ggml_tensor **        moe_selected_out,
    ggml_tensor *         kv_write_rows = nullptr,
    ggml_tensor *         parent_ids = nullptr);

QwenLayerPrefnOutputs build_qwen35_layer_prefn(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,
    ggml_tensor *         positions,
    ggml_tensor *         attn_mask,
    int                   kv_start,
    int                   n_tokens,
    int                   fa_window = 0,
    ggml_tensor *         kv_write_rows = nullptr,
    bool                  skip_gdn_intermediate = true);

} // namespace dflash::common

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)
#include <cuda_runtime.h>
// Host-staged copy between CUDA devices (no peer access required).
// Streams are device-specific: src_stream orders the D2H leg on src_dev and
// dst_stream orders the H2D leg on dst_dev. Null streams use each device's
// default stream. The helper synchronizes before returning.
bool dflash_cuda_copy_between_devices(int src_dev, const void * src,
                                      int dst_dev, void * dst, size_t nbytes,
                                      cudaStream_t src_stream = nullptr,
                                      cudaStream_t dst_stream = nullptr);
#endif
