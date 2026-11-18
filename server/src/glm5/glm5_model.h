// GLM-5.3-Flash (glm5-next) weights and decode state: mHC on every layer,
// KDA or NoPE MLA + DSA indexer attention, dense FFN then MoE.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

struct Glm5Layer {
    bool is_mla = false;   // MLA + DSA indexer; KDA otherwise
    bool is_moe = false;   // routed experts; dense FFN otherwise
    int  device = 0;       // index into the backend list

    ggml_tensor * attn_norm = nullptr;  // [n_embd]
    ggml_tensor * ffn_norm  = nullptr;  // [n_embd]

    // mHC; the pre/post/comb scales are read to the host.
    ggml_tensor * hc_attn_fn   = nullptr;
    ggml_tensor * hc_attn_base = nullptr;
    ggml_tensor * hc_ffn_fn    = nullptr;
    ggml_tensor * hc_ffn_base  = nullptr;
    float hc_attn_scale[3] = {0.0f, 0.0f, 0.0f};
    float hc_ffn_scale[3]  = {0.0f, 0.0f, 0.0f};

    // KDA.
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr;  // [n_embd, d_inner]
    ggml_tensor * conv_q = nullptr, * conv_k = nullptr, * conv_v = nullptr;  // [d_conv, 1, d_inner]
    ggml_tensor * ssm_f_a  = nullptr;  // [n_embd, head_dim]   low-rank decay gate
    ggml_tensor * ssm_f_b  = nullptr;  // [head_dim, d_inner]
    ggml_tensor * ssm_beta = nullptr;  // [n_embd, n_head]
    ggml_tensor * ssm_a    = nullptr;  // [n_head], holds -exp(A_log)
    ggml_tensor * ssm_dt_b = nullptr;  // [d_inner]
    ggml_tensor * ssm_g_a  = nullptr;  // [n_embd, head_dim]   low-rank output gate
    ggml_tensor * ssm_g_b  = nullptr;  // [head_dim, d_inner]
    ggml_tensor * ssm_norm = nullptr;  // [head_dim]

    // MLA (NoPE).
    ggml_tensor * q_a       = nullptr;  // [n_embd, q_lora]
    ggml_tensor * q_a_norm  = nullptr;  // [q_lora]
    ggml_tensor * q_b       = nullptr;  // [q_lora, n_head*qk_head]
    ggml_tensor * kv_a      = nullptr;  // [n_embd, kv_lora]
    ggml_tensor * kv_a_norm = nullptr;  // [kv_lora]
    ggml_tensor * k_b       = nullptr;  // [qk_head, kv_lora, n_head]
    ggml_tensor * v_b       = nullptr;  // [kv_lora, v_head, n_head]

    // DSA indexer with k-pool compression.
    ggml_tensor * idx_k_norm   = nullptr;  // [idx_dim]
    ggml_tensor * idx_k_norm_b = nullptr;  // [idx_dim]
    ggml_tensor * idx_proj     = nullptr;  // [n_embd, idx_n_head]
    ggml_tensor * idx_k        = nullptr;  // [n_embd, idx_dim]
    ggml_tensor * idx_q_b      = nullptr;  // [q_lora, idx_n_head*idx_dim]
    ggml_tensor * idx_gate     = nullptr;  // [n_embd, idx_dim]
    ggml_tensor * idx_ape      = nullptr;  // [idx_dim, kpool]

    ggml_tensor * wo = nullptr;  // attention output: [d_inner or n_head*v_head, n_embd]

    // Dense FFN.
    ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;

    // MoE.
    ggml_tensor * ffn_gate_inp   = nullptr;  // [n_embd, n_expert]
    ggml_tensor * ffn_exp_bias   = nullptr;  // [n_expert], selection only
    ggml_tensor * ffn_gate_exps  = nullptr;  // [n_embd, n_ff_exp, n_expert]
    ggml_tensor * ffn_up_exps    = nullptr;
    ggml_tensor * ffn_down_exps  = nullptr;  // [n_ff_exp, n_embd, n_expert]
    ggml_tensor * ffn_gate_shexp = nullptr;  // [n_embd, n_ff_exp*n_shared]
    ggml_tensor * ffn_up_shexp   = nullptr;
    ggml_tensor * ffn_down_shexp = nullptr;

    float swiglu_clamp_exp   = 0.0f;
    float swiglu_clamp_shexp = 0.0f;
};

struct Glm5Weights {
    // Per device.
    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    ggml_tensor * tok_embd = nullptr;  // [n_embd, n_vocab], on the first layer's device
    ggml_tensor * out_norm = nullptr;  // [n_embd], on the last layer's device
    ggml_tensor * output   = nullptr;  // [n_embd, n_vocab]
    int embd_device = 0;
    int head_device = 0;

    std::vector<Glm5Layer> layers;

    int n_layer      = 0;
    int n_embd       = 0;
    int n_vocab      = 0;
    int n_head       = 0;   // shared by KDA and MLA
    int n_ctx_train  = 0;

    float rms_eps = 1e-5f;
    float norm_eps = 1e-6f;  // the indexer key LayerNorm

    // KDA
    int kda_head_dim = 0;
    int ssm_d_conv   = 0;
    float kda_gate_lower_bound = 0.0f;

    // MLA
    int q_lora  = 0;
    int kv_lora = 0;
    int qk_head = 0;
    int v_head  = 0;

    // Indexer
    int  idx_n_head  = 0;
    int  idx_dim     = 0;
    int  idx_top_k   = 0;
    int  idx_kpool   = 0;
    bool idx_select_tail = true;

    // mHC
    int   hc_mult  = 0;
    int   hc_iters = 0;
    float hc_eps   = 0.0f;

    // FFN
    int   n_ff          = 0;
    int   n_dense_lead  = 0;
    int   n_expert      = 0;
    int   n_expert_used = 0;
    int   n_ff_exp      = 0;
    int   n_shared      = 0;
    float expert_scale  = 1.0f;
    bool  expert_norm   = true;

    // End-of-generation tokens (eos, eot, eom from the GGUF).
    std::vector<int32_t> stop_tokens;

    int d_inner() const { return kda_head_dim * n_head; }
    int n_mla_layers() const;
    int n_kda_layers() const { return n_layer - n_mla_layers(); }
};

// Loads a GGUF or the first shard of a split one; layer_device[il] picks each layer's backend.
bool load_glm5_model(const std::string & path,
                     const std::vector<ggml_backend_t> & backends,
                     const std::vector<int> & layer_device,
                     Glm5Weights & out);

void free_glm5_model(Glm5Weights & w);

// block_count of a glm5-next GGUF, 0 on failure.
int read_glm5_layer_count(const std::string & path);

// Per-layer weight bytes, for multi-device placement; empty on failure.
std::vector<size_t> read_glm5_layer_bytes(const std::string & path);

// Decode state for one sequence, on each layer's device.
struct Glm5LayerCache {
    // KDA
    ggml_tensor * conv = nullptr;  // F32 [d_conv-1, 3*d_inner]   q|k|v conv tails
    ggml_tensor * ssm  = nullptr;  // F32 [head_dim, head_dim, n_head]
    // MLA
    ggml_tensor * latent = nullptr;  // F16 [kv_lora, max_ctx]
    ggml_tensor * idx_k  = nullptr;  // F32 [idx_dim, max_ctx]   per-token indexer key
    ggml_tensor * idx_g  = nullptr;  // F32 [idx_dim, max_ctx]   per-token pool gate
    ggml_tensor * pooled = nullptr;  // F32 [idx_dim, max_ctx/kpool]
};

struct Glm5Cache {
    int max_ctx = 0;   // a multiple of 256
    int cur_pos = 0;
    std::vector<Glm5LayerCache>        layers;
    std::vector<ggml_context *>        ctxs;  // per device
    std::vector<ggml_backend_buffer_t> bufs;  // per device
};

bool create_glm5_cache(const Glm5Weights & w,
                       const std::vector<ggml_backend_t> & backends,
                       int max_ctx, Glm5Cache & out);
void free_glm5_cache(Glm5Cache & c);
// Zero every state tensor (a fresh sequence).
void clear_glm5_cache(Glm5Cache & c);

}  // namespace luce::common
