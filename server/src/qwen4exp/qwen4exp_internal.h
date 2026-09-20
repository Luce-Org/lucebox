// Qwen3.8-Flash-Next (GGUF arch `qwen4exp`) internal weight layout.
//
// The loader fills these structs straight from the GGUF tensor table and the
// backend builds its own ggml graph (not a llama.cpp model). The tensor
// inventory lives in docs/qwen4exp.md; `per_layer_token_embd` is read lazily
// from disk and never uploaded.

#pragma once

#include "internal.h"
#include "common/gguf_mmap.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpLayer {
    // Pre-attention / pre-FFN norms (the block RMSNorm inputs).
    ggml_tensor * attn_norm      = nullptr;  // [n_embd]
    ggml_tensor * attn_post_norm = nullptr;  // [n_embd]
    ggml_tensor * ffn_norm       = nullptr;  // [n_embd]

    // Hyper-connections (hc_count streams). Norm is [n_embd] reshaped to
    // [n_embd, hc]; down/up are the low-rank [hc_dim, hc_lr] / [hc_lr, hc_dim]
    // mixers; inject is [hc_dim, hc].
    ggml_tensor * hc_attn_norm   = nullptr;
    ggml_tensor * hc_attn_down   = nullptr;
    ggml_tensor * hc_attn_up     = nullptr;
    ggml_tensor * hc_attn_inject = nullptr;
    ggml_tensor * hc_ffn_norm    = nullptr;
    ggml_tensor * hc_ffn_down    = nullptr;
    ggml_tensor * hc_ffn_up      = nullptr;
    ggml_tensor * hc_ffn_inject  = nullptr;

    // Linear attention (gated delta net), 36 layers.
    ggml_tensor * attn_qkv       = nullptr;  // fused q|k|v projection
    ggml_tensor * attn_gate      = nullptr;  // the output gate ("z") projection
    ggml_tensor * ssm_conv1d     = nullptr;  // depthwise causal conv
    ggml_tensor * ssm_alpha      = nullptr;  // per-token alpha input projection
    ggml_tensor * ssm_beta       = nullptr;  // per-token beta input projection
    ggml_tensor * ssm_a          = nullptr;  // per-head -A parameter
    ggml_tensor * ssm_dt_bias    = nullptr;  // alpha bias
    ggml_tensor * ssm_norm       = nullptr;
    ggml_tensor * ssm_out        = nullptr;

    // Full attention, 12 layers (every `full_attention_interval`).
    ggml_tensor * wq             = nullptr;  // holds [q|gate] interleaved per head
    ggml_tensor * wk             = nullptr;
    ggml_tensor * wv             = nullptr;
    ggml_tensor * wo             = nullptr;
    ggml_tensor * q_norm         = nullptr;
    ggml_tensor * k_norm         = nullptr;

    // Learned sparse-attention indexer, full-attention layers only.
    ggml_tensor * indexer_q_proj = nullptr;  // [n_embd, indexer_n_head*indexer_head_size]
    ggml_tensor * indexer_k_proj = nullptr;
    ggml_tensor * indexer_q_norm = nullptr;
    ggml_tensor * indexer_k_norm = nullptr;

    // Per-layer embedding (PLE), present on `ple_layer_ids` only.
    ggml_tensor * ple_conv1d     = nullptr;  // [ple_conv_kernel, hc_dim]
    ggml_tensor * ple_key        = nullptr;  // [n_embd, hc_dim]
    ggml_tensor * ple_value      = nullptr;  // [n_embd, n_embd]
    ggml_tensor * ple_norm_conv  = nullptr;
    ggml_tensor * ple_norm_key   = nullptr;
    ggml_tensor * ple_norm_query = nullptr;

    // MoE FFN (all 48 layers).
    ggml_tensor * ffn_gate_inp        = nullptr;  // router
    ggml_tensor * ffn_exp_probs_b     = nullptr;  // router correction bias
    ggml_tensor * ffn_gate_exps       = nullptr;
    ggml_tensor * ffn_up_exps         = nullptr;
    ggml_tensor * ffn_down_exps       = nullptr;
    ggml_tensor * ffn_gate_inp_shexp  = nullptr;  // shared-expert scalar gate
    ggml_tensor * ffn_gate_shexp      = nullptr;
    ggml_tensor * ffn_up_shexp        = nullptr;
    ggml_tensor * ffn_down_shexp      = nullptr;

    bool is_full_attention = false;
    bool is_ple            = false;

    // Optional NVFP4-style per-tensor scales (1.0 = none).
    float attn_qkv_s = 1.0f;
    float wq_s       = 1.0f;
    float wk_s       = 1.0f;
    float wv_s       = 1.0f;
    float wo_s       = 1.0f;
    float ssm_out_s  = 1.0f;
};

// Lazy direct reader for shard 2's per_layer_token_embd: pread rows from a
// worker pool instead of mmap, so scattered rows do not demand-fault through
// the managed-memory fault handler.
class Qwen4ExpPleReader {
public:
    Qwen4ExpPleReader() = default;
    Qwen4ExpPleReader(const Qwen4ExpPleReader &) = delete;
    Qwen4ExpPleReader & operator=(const Qwen4ExpPleReader &) = delete;
    ~Qwen4ExpPleReader();

    // `path` is shard 2; `tensor_name` is the GGUF tensor to serve rows from.
    // `n_threads` bounds the pread worker pool.
    bool open(const std::string & path, const char * tensor_name,
              int n_threads, std::string & out_error);
    void close();

    bool available() const { return fd_ >= 0; }
    int64_t n_rows() const { return n_rows_; }
    int64_t row_bytes() const { return row_size_; }
    int64_t head_dim() const { return head_dim_; }

    // Fill dst[slot*head_dim, (slot+1)*head_dim) with row `rows[slot]`
    // dequantized to F32. Thread-safe.
    bool gather(const int32_t * rows, int64_t n, float * dst) const;

private:
    int              fd_        = -1;
    size_t           base_      = 0;   // file offset of row 0
    size_t           row_size_  = 0;   // bytes per quantized row
    int64_t          n_rows_    = 0;
    int64_t          head_dim_  = 0;
    int              n_threads_ = 4;
    ggml_type        type_      = GGML_TYPE_COUNT;
    ggml_to_float_t  to_float_  = nullptr;
};

struct Qwen4ExpWeights {
    ggml_context *        ctx     = nullptr;  // shard 1 tensor descriptors
    // Descriptor contexts of shards 2..N (split GGUFs); `ctx` covers shard 1.
    std::vector<ggml_context *> extra_meta_ctxs;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    CpuEmbedder           embedder;

    ggml_tensor * tok_embd     = nullptr;  // metadata only; data stays on CPU
    ggml_tensor * out_norm     = nullptr;
    ggml_tensor * output       = nullptr;
    ggml_tensor * output_hc_norm = nullptr;
    ggml_tensor * output_hc_down = nullptr;
    ggml_tensor * output_hc_up   = nullptr;

    std::vector<Qwen4ExpLayer> layers;

    // Config (GGUF `qwen4exp.*`).
    int n_layer               = 48;
    int n_embd                = 2560;
    int n_head                = 24;
    int n_head_kv             = 2;
    int n_embd_head_k         = 256;
    int n_embd_head_v         = 256;
    int full_attention_interval = 4;
    int n_ff_exp              = 640;
    int n_ff_shexp            = 640;
    int n_expert              = 512;
    int n_expert_used         = 10;
    int n_vocab               = 0;      // from token_embd.ne[1]
    float rms_eps             = 1e-6f;
    float rope_theta          = 1e7f;
    int rope_dimension_count  = 64;
    int rope_sections[4]      = {11, 11, 10, 0};

    // Hyper-connections.
    int n_hc                  = 4;
    int hc_lowrank            = 320;

    // Gated delta net (linear attention).
    int ssm_d_conv            = 4;
    int ssm_d_inner           = 6144;   // = n_v_heads * head_v_dim
    int ssm_d_state           = 128;    // key/value head dim
    int ssm_dt_rank           = 48;     // n_v_heads
    int ssm_n_group           = 16;     // n_k_heads
    int linear_value_heads    = 48;
    int linear_key_heads      = 16;

    // Indexer (full-attention layers).
    int indexer_n_head        = 4;
    int indexer_head_size     = 128;
    int indexer_top_k         = 2048;
    std::vector<int> compress_ratios;   // per layer: 0 = linear, >0 = full (ratio)

    // Per-layer embedding.
    int ple_ngram_size        = 3;
    int ple_heads_per_ngram   = 8;
    int ple_conv_kernel       = 4;
    int ple_head_dim          = 0;      // = embedding_length_per_layer_input
    int ple_n_heads           = 0;      // = (ngram_size - 1) * heads_per_ngram
    std::vector<int32_t> ple_layer_ids;
    std::vector<int64_t> ple_head_offsets;       // u64 in GGUF
    std::vector<int64_t> ple_head_vocab_sizes;   // u64 in GGUF
    std::vector<uint64_t> ple_layer_multipliers; // u64 in GGUF
    int32_t ple_eos_token_id  = -1;
    int32_t ple_image_token_id = -1;

    // The shard holding per_layer_token_embd plus its lazy reader.
    std::string   shard2_path;
    Qwen4ExpPleReader ple_reader;

    int32_t eos_id      = -1;
    int32_t eos_chat_id = -1;
};

// Load the autoregressive trunk of a Qwen3.8-Flash-Next (`qwen4exp`) GGUF.
// Split models ("-00001-of-00003.gguf") load every shard; single-file GGUFs
// load as one shard. The per_layer_token_embd table is discovered in whichever
// shard holds it and opened lazily. Returns false and sets last_error on
// failure.
bool load_qwen4exp_gguf(const std::string & path,
                        ggml_backend_t backend,
                        Qwen4ExpWeights & out);

void free_qwen4exp_weights(Qwen4ExpWeights & w);

}  // namespace dflash::common
