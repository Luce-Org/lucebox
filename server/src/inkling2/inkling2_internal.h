// Inkling 2 / Inkling 2 Mini (`inkling2_loop` / `inkling2`) internal weight layout.
//
// Native Lucebox loader & graph representation: no llama_model / libllama.
// Supports:
//   - Looped middle core (24 physical layers -> 48 effective layer visits)
//   - Load-time fused attention projections (Q|K|V|Gate|R|Sink on SWA pass 0,
//     Q|Gate|R|Sink on all passes, K|k_conv_proj and V|v_conv_proj on global)
//   - Load-time merged loop controller gain|forget weights
//   - Dual-GPU R9700 + Strix Halo dynamic parallel expert split (74/26 default
//     for T>=2 verify + zero-hop R9700-only fast path for T=1),
//     79/21 LM-head split, and Split-KV Global Attention offload
//   - Native bundled DSpark speculator loaded straight from `mtp.safetensors`
//     with pre-folded `lm_head @ hidden_out` weights.

#pragma once

#include "internal.h"
#include "ggml-cuda.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

// Physical layer weights (24 physical layers in Inkling 2 Mini).
struct Inkling2Layer {
    bool is_swa = true;

    // Input norms
    ggml_tensor * attn_norm       = nullptr; // SWA: [n_embd] F32
    ggml_tensor * attn_q_inp_norm = nullptr; // Global: [n_embd] F32
    ggml_tensor * attn_k_inp_norm = nullptr; // Global: [n_embd] F32
    ggml_tensor * attn_v_inp_norm = nullptr; // Global: [n_embd] F32
    ggml_tensor * attn_qkv_norm   = nullptr; // Derived global: stacked [n_embd, 3] F32

    // Attention projections (raw from GGUF, Q8_0)
    ggml_tensor * wq              = nullptr; // [n_embd, Hq*D]
    ggml_tensor * wk              = nullptr; // [n_embd, Hk*D]
    ggml_tensor * wv              = nullptr; // [n_embd, Hk*D]
    ggml_tensor * wqkv_gate       = nullptr; // [n_embd, Hq*D]
    ggml_tensor * attn_rel        = nullptr; // [n_embd, d_rel*Hq]
    ggml_tensor * attn_sink_proj  = nullptr; // [n_embd, Hq]
    ggml_tensor * wo              = nullptr; // [Hq*D, n_embd]
    ggml_tensor * attn_k_conv_proj = nullptr; // Global: [n_embd, 4*Hk]
    ggml_tensor * attn_v_conv_proj = nullptr; // Global: [n_embd, 4*Hk]

    // Load-time fused attention projections (same Q8_0 rows concatenated contiguously!):
    //   wqgrs:  [n_embd, Hq*D + Hq*D + d_rel*Hq + Hq] (Q | Gate | R | Sink) — all layers!
    //   wqkvgrs:[n_embd, Hq*D + Hk*D + Hk*D + Hq*D + d_rel*Hq + Hq] — SWA layers (Pass 0 / single input)!
    //   wk_dyn: [n_embd, Hk*D + 4*Hk] — Global layers (K | k_conv_proj)!
    //   wv_dyn: [n_embd, Hk*D + 4*Hk] — Global layers (V | v_conv_proj)!
    ggml_tensor * wqgrs           = nullptr;
    ggml_tensor * wqkvgrs         = nullptr;
    ggml_tensor * wk_dyn          = nullptr;
    ggml_tensor * wv_dyn          = nullptr;

    // Clips, relative-position bias, short convolutions (all F32)
    ggml_tensor * attn_q_clip     = nullptr; // [D, Hq]
    ggml_tensor * attn_rel_clip   = nullptr; // [d_rel, Hq]
    ggml_tensor * attn_rel_bias   = nullptr; // [d_rel, rel_extent]
    ggml_tensor * attn_k_conv     = nullptr; // [4, Hk*D]
    ggml_tensor * attn_v_conv     = nullptr; // [4, Hk*D]
    ggml_tensor * attn_post_conv  = nullptr; // [4, n_embd]
    ggml_tensor * ffn_post_conv   = nullptr; // [4, n_embd]

    // Attention residuals (F32)
    ggml_tensor * attn_res_query  = nullptr; // nullptr for layer 0; [n_embd, 1 or 3]
    ggml_tensor * ffn_res_query   = nullptr; // [n_embd]

    // MoE FFN
    ggml_tensor * ffn_norm         = nullptr; // [n_embd] F32
    ggml_tensor * ffn_gate_inp     = nullptr; // [n_embd, n_expert] F32
    ggml_tensor * ffn_gate_inp_q8  = nullptr; // Derived [n_embd, n_expert] Q8_0 (fast RDNA4 GEMV/MMQ!)
    ggml_tensor * ffn_exp_probs_b  = nullptr; // [n_expert] F32 (optional)
    ggml_tensor * ffn_gate_up_exps = nullptr; // [n_embd, 2*n_ff_exp, n_expert] IQ4_NL/Q8_0 (on target)
    ggml_tensor * ffn_down_exps    = nullptr; // [n_ff_exp, n_embd, n_expert] IQ4_NL/Q8_0 (on target)

    // Dual-GPU split mode: tail experts [split_exp .. n_expert-1] on expert_backend (Strix Halo)
    ggml_tensor * exp_gate_up_tail = nullptr; // [n_embd, 2*n_ff_exp, n_exp_tail] on expert_backend
    ggml_tensor * exp_down_tail    = nullptr; // [n_ff_exp, n_embd, n_exp_tail] on expert_backend
};

// Cross-pass loop controller weights for core physical layers [core_start, core_end).
struct Inkling2LoopLayer {
    enum { CTRL_Q = 0, CTRL_K = 1, CTRL_V = 2, CTRL_FFN = 3, CTRL_COUNT = 4 };
    ggml_tensor * msg_query = nullptr; // [n_embd, 4] F32
    ggml_tensor * ctrl_down[CTRL_COUNT]   = {}; // [2*n_embd, gate_rank] F32
    ggml_tensor * ctrl_gain[CTRL_COUNT]   = {}; // [gate_rank, n_embd] F32
    ggml_tensor * ctrl_forget[CTRL_COUNT] = {}; // [gate_rank, n_embd] F32
    // Load-time merged & FP16-compressed loop controllers (fits 100% in R9700's 64 MB MALL!)
    ggml_tensor * ctrl_down_f16[CTRL_COUNT] = {}; // [2*n_embd, gate_rank] F16
    ggml_tensor * ctrl_gf[CTRL_COUNT]       = {}; // [gate_rank, 2*n_embd] F16 (or F32)
    // 3-way stacked Q/K/V loop controllers for single-launch 3D batched GEMV/GEMM!
    ggml_tensor * ctrl_down_qkv             = nullptr; // [2*n_embd, gate_rank, 3] F16 (or F32)
    ggml_tensor * ctrl_gf_qkv               = nullptr; // [gate_rank, 2*n_embd, 3] F16 (or F32)
};

// Bundled DSpark speculator (`mtp.safetensors`): 3-layer DFlare-mixed bidirectional
// block-8 transformer + recurrent (`rnn_hidden`) Markov head.
struct Inkling2DSparkLayer {
    bool is_local = true;
    ggml_tensor * dflare_norm = nullptr; // [n_embd] F32
    ggml_tensor * inp_norm    = nullptr; // [n_embd] F32
    ggml_tensor * post_norm   = nullptr; // [n_embd] F32
    ggml_tensor * q_norm      = nullptr; // [D] F32
    ggml_tensor * k_norm      = nullptr; // [D] F32
    ggml_tensor * k_norm_ctx  = nullptr; // [D] F32
    ggml_tensor * wqkv        = nullptr; // Fused [n_embd, Hq*D + 2*Hk*D] F16/BF16
    ggml_tensor * wkv_ctx     = nullptr; // Fused [n_embd, 2*Hk*D] F16/BF16
    ggml_tensor * wo          = nullptr; // [Hq*D, n_embd] F16/BF16
    ggml_tensor * w_gate_up   = nullptr; // Fused [n_embd, 2*inter] F16/BF16
    ggml_tensor * w_down      = nullptr; // [inter, n_embd] F16/BF16
};

struct Inkling2DSparkWeights {
    bool loaded = false;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    int n_layer      = 3;
    int n_embd       = 2304;
    int n_head       = 24;
    int n_head_kv    = 8;
    int head_dim     = 128;
    int n_ff         = 9216;
    int block_size   = 8;   // 1 anchor + 7 masks -> 7 speculative tokens
    int markov_rank  = 256;
    int swa_window   = 1024;
    float rope_theta = 1000000.0f;
    float rms_eps    = 1e-6f;
    std::vector<int> target_layer_ids = {5, 23, 41, 48};

    // fuse_alpha[row][tap] = softmax(fuse_weight[row])[tap], F32 [n_taps, n_layer + 1]
    ggml_tensor * fuse_alpha[4] = {}; // [4] F32 each: 0..2 for layers, 3 for markov_head
    ggml_tensor * mask_embed    = nullptr; // [n_embd] F32
    ggml_tensor * norm          = nullptr; // [n_embd] F32
    std::vector<Inkling2DSparkLayer> layers;

    // Recurrent (`rnn_hidden`) Markov head
    ggml_tensor * markov_dflare_norm = nullptr; // [n_embd] F32
    ggml_tensor * markov_w1          = nullptr; // [rank, n_vocab_padded] F16
    ggml_tensor * rnn_proj_ctx       = nullptr; // [2*n_embd, 3*rank] F16 (columns [2*R :])
    ggml_tensor * rnn_proj_step      = nullptr; // [2*rank, 3*rank] F32 (columns [: 2*R])
    ggml_tensor * rnn_bias           = nullptr; // [3*rank] F32
    // Pre-folded lm_head @ hidden_out: [rank, n_vocab_padded] F16 & [n_vocab_padded] F32
    ggml_tensor * fold_weight        = nullptr;
    ggml_tensor * fold_bias          = nullptr;
    // Quantized Q4_0 draft-only LM-head copy (2x faster DSpark base_logits, zero target quality impact!)
    ggml_tensor * lm_head_q4         = nullptr;
};

struct Inkling2Weights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    // Derived fused weights (wqgrs, wqkvgrs, wk_dyn, wv_dyn, ctrl_gf, attn_qkv_norm)
    ggml_context *        derived_ctx = nullptr;
    ggml_backend_buffer_t derived_buf = nullptr;

    // Optional dual-GPU expert backend (Strix Halo) for parallel verify/prefill + LM-head tail
    ggml_backend_t        expert_backend = nullptr; // not owned
    ggml_context *        expert_ctx     = nullptr;
    ggml_backend_buffer_t expert_buf     = nullptr;
    // Experts [0 .. split_exp - 1] run on target; [split_exp .. n_expert - 1] run on expert_backend
    // when split verify/prefill is active. Target still holds all 128 experts when target_has_all_experts=true
    // so T=1 decode has ZERO PCIe hops!
    int                   split_exp = 0; // 0 = no expert split
    int                   split_pct = 73; // dynamic exact-unique target share (%)
    bool                  target_has_all_experts = true;
    ggml_tensor *         split_hot_lut = nullptr;  // [n_expert] I32 on target: 0..split_exp-1 -> id, else -1
    ggml_tensor *         split_cold_lut = nullptr; // [n_expert] I32 on target: split_exp..127 -> id-split_exp, else -1

    // Optional 79/21 LM-head row split on expert_backend (last `output_tail_rows` of `output`)
    ggml_tensor *         output_tail = nullptr;
    int64_t               output_tail_rows = 0;

    // Embeddings & output head
    CpuEmbedder           embedder;
    ggml_tensor *         tok_embd         = nullptr; // [n_embd, n_vocab_padded] Q8_0 (on GPU + CPU embedder)
    ggml_tensor *         tok_norm         = nullptr; // [n_embd] F32
    ggml_tensor *         output           = nullptr; // [n_embd, n_vocab_padded] Q8_0
    ggml_tensor *         output_norm      = nullptr; // [n_embd] F32
    ggml_tensor *         output_res_query = nullptr; // [n_embd] F32

    std::vector<Inkling2Layer>     layers;      // size = n_layer_phys (24)
    std::vector<Inkling2LoopLayer> loop_layers; // size = core_end - core_start (12)

    Inkling2DSparkWeights dspark;

    // Architecture hyperparameters
    bool  is_loop          = true;
    int   n_layer_phys     = 24;
    int   core_start       = 6;
    int   core_end         = 18;
    int   n_passes         = 3;
    int   gate_rank        = 64;
    int   n_layer_eff      = 48;  // n_phys + (n_passes - 1) * (core_end - core_start)
    int   n_embd           = 2304;
    int   n_head           = 24;
    int   n_head_kv        = 8;
    int   head_dim         = 128;
    int   n_ff_exp         = 1152;
    int   n_expert         = 128;
    int   n_expert_used    = 8;
    int   n_vocab_padded   = 201024;
    int   n_vocab_unpadded = 200058;
    int   swa_window       = 1024;
    int   d_rel            = 16;
    int   rel_extent       = 1024;
    int   sconv_kernel     = 4;
    int   attn_res_block   = 6;
    int   log_scale_floor  = 8192;
    float log_scale_alpha  = 0.1f;
    float rms_eps          = 1e-6f;
    float qk_clip          = 22.62741661f;
    float rel_clip         = 8.0f;
    float situ_beta        = 4.0f;
    float situ_linear_beta = 25.0f;
    float route_scale      = 8.0f;
    float logit_softcap    = 30.0f;
    int   sconv_channels   = 0;   // 2*Hk*D + 2*n_embd = 2048 + 4608 = 6656
    int   context_length   = 262144;
    int32_t eos_id         = 200006; // <|content_model_end_sampling|>
    int32_t eos_chat_id    = 200006;

    // Per-effective-slot helpers (0..47 -> physical layer 0..23)
    int phys_layer_for_slot(int slot) const {
        if (!is_loop || slot < n_layer_phys) return slot;
        return core_start + (slot - n_layer_phys) % (core_end - core_start);
    }

    // Execution visit order: visit v in 0..n_layer_eff-1 -> (cache_slot il, phys_layer ip, pass)
    struct VisitInfo {
        int visit = 0;
        int slot  = 0; // cache slot index in 0..47
        int phys  = 0; // physical layer index in 0..23
        int pass  = 0; // 0 for prelude/coda/pass0, 1..n_passes-1 for looped passes
        bool is_core = false;
    };
    std::vector<VisitInfo> schedule;
    std::vector<int>       global_slots; // cache slot indices of global attention layers (8 slots)
    std::vector<int>       slot_to_global_idx; // -1 for SWA, 0..7 for global
};

bool load_inkling2_gguf(const std::string & path,
                        ggml_backend_t target_backend,
                        ggml_backend_t expert_backend,
                        Inkling2Weights & out,
                        std::string & error);

bool load_inkling2_dspark_safetensors(const std::string & path,
                                      ggml_backend_t backend,
                                      Inkling2Weights & weights,
                                      std::string & error);

void free_inkling2_weights(Inkling2Weights & w);

} // namespace luce::common
