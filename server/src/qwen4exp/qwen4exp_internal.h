// Qwen3.8-Flash-Next (GGUF arch `qwen4exp`) internal weight layout.
//
// The loader fills these structs straight from the GGUF tensor table and the
// backend builds its own ggml graph (not a llama.cpp model).
// `per_layer_token_embd` is read lazily from disk and never uploaded.

#pragma once

#include "internal.h"
#include "qwen4exp_mtp.h"
#include "ggml-cuda.h"
#include "common/gguf_mmap.h"
#include "common/moe_hybrid_routing_stats.h"
#include "common/moe_hybrid_storage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// Match ggml's calling-thread overrides, restoring them even on early returns.
struct Qwen4ExpCudaScope {
    const bool optimized;
    const ggml_cuda_qwen4exp_profile previous;
    explicit Qwen4ExpCudaScope(bool gfx1151)
        : optimized(gfx1151),
          previous(ggml_backend_cuda_set_qwen4exp_profile(
              gfx1151 ? GGML_CUDA_QWEN4EXP_DEFAULT : GGML_CUDA_QWEN4EXP_OFF)) {}
    ~Qwen4ExpCudaScope() { ggml_backend_cuda_set_qwen4exp_profile(previous); }
    Qwen4ExpCudaScope(const Qwen4ExpCudaScope &) = delete;
    Qwen4ExpCudaScope & operator=(const Qwen4ExpCudaScope &) = delete;
};

struct Qwen4ExpLayer {
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
    ggml_tensor * ssm_gate_ba    = nullptr;  // [dt_bias | A] f32 [2*H_v]: steps let the recurrence apply the gates
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
    ggml_tensor * ffn_gate_exps       = nullptr;
    ggml_tensor * ffn_up_exps         = nullptr;
    ggml_tensor * ffn_down_exps       = nullptr;
    ggml_tensor * ffn_gate_inp_shexp  = nullptr;  // shared-expert scalar gate
    ggml_tensor * ffn_gate_shexp      = nullptr;
    ggml_tensor * ffn_up_shexp        = nullptr;
    ggml_tensor * ffn_down_shexp      = nullptr;

    bool is_full_attention = false;
    bool is_ple            = false;
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

// Research: the router's picks of decode and verify steps, counted per layer
// (LUCE_QWEN4EXP_ROUTING_STATS_OUT=path; the CSV is rewritten every 64 steps).
struct Qwen4ExpRouteRecorder {
    MoeHybridRoutingStats stats;
    std::string path;
    int64_t steps = 0;
};

struct Qwen4ExpWeights {
    bool gfx1151 = false;  // Cache profile support at load, before any allocation.
    bool qsa = false;      // The QSA kernels run on the target device (gfx1151, RDNA4).
    ggml_context *        ctx     = nullptr;  // shard 1 tensor descriptors
    // Descriptor contexts of shards 2..N (split GGUFs); `ctx` covers shard 1.
    std::vector<ggml_context *> extra_meta_ctxs;
    ggml_backend_buffer_t buf     = nullptr;
    // Split mode: the routed expert stacks live on a second device (set by the
    // caller before loading, not owned); everything else stays on `backend`.
    ggml_backend_t        expert_backend = nullptr;
    ggml_backend_buffer_t expert_buf     = nullptr;
    bool                  expert_gfx1151 = false;
    // Split mode, research: the most-routed experts copied to the target
    // (`hot`, from LUCE_QWEN4EXP_HOT_EXPERTS). Decode and verify steps run
    // them there while the expert device runs the rest of the picks.
    std::unique_ptr<MoeHybridStorage> hot;
    ggml_context *             lut_ctx = nullptr;
    ggml_backend_buffer_t      lut_buf = nullptr;
    std::vector<ggml_tensor *> hot_lut;    // per layer [1, n_expert] i32: hot slot, -1 if cold (null: no hot experts)
    std::vector<ggml_tensor *> cold_lut;   // per layer [1, n_expert] i32: the expert, -1 if hot
    ggml_context *             gate_ctx = nullptr;   // the layers' ssm_gate_ba
    ggml_backend_buffer_t      gate_buf = nullptr;
    std::unique_ptr<Qwen4ExpRouteRecorder> route_recorder;

    CpuEmbedder           embedder;

    ggml_tensor * tok_embd     = nullptr;  // metadata only; data stays on CPU
    ggml_tensor * output       = nullptr;
    ggml_tensor * output_hc_norm = nullptr;
    ggml_tensor * output_hc_down = nullptr;
    ggml_tensor * output_hc_up   = nullptr;

    std::vector<Qwen4ExpLayer> layers;

    // MTP draft head from an MTP sidecar: one full-attention layer (blk.<n_layer>) and the nextn
    // projections; it borrows token_embd/output. mtp_eh_proj is null when no sidecar is loaded.
    Qwen4ExpLayer mtp;
    ggml_tensor * mtp_enorm = nullptr, * mtp_hnorm = nullptr, * mtp_eh_proj = nullptr;
    ggml_tensor * mtp_head_norm = nullptr, * mtp_head_down = nullptr, * mtp_head_up = nullptr;

    // Draft-only gathered output rows and exact CPU-dequantized embeddings.
    ggml_context * mtp_vocab_ctx = nullptr;
    ggml_backend_buffer_t mtp_vocab_buf = nullptr;
    ggml_tensor * mtp_output = nullptr, * mtp_embd = nullptr;
    std::vector<int32_t> mtp_vocab_ids;

    // Config (GGUF `qwen4exp.*`).
    int n_layer               = 48;
    int n_embd                = 2560;
    int n_head                = 24;
    int n_head_kv             = 2;
    int n_embd_head_k         = 256;
    int n_embd_head_v         = 256;
    int full_attention_interval = 4;
    int n_ff_exp              = 640;
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
    int ssm_n_group           = 16;     // n_k_heads
    int linear_value_heads    = 48;

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
    int32_t ple_image_token_id = -1;   // kept in the MTP draft vocabulary

    // Lazy reader of per_layer_token_embd, in whichever shard holds it.
    Qwen4ExpPleReader ple_reader;

    int32_t eos_id      = -1;
    int32_t eos_chat_id = -1;
};

// Clears the MTP sidecar layer and its six projection/head pointers so neither can outlive the ggml_context
// that owns them. Must run on both teardown and load failure: qwen4exp_cache.cpp treats a non-null
// mtp_eh_proj as "MTP available", and a later load that skips the sidecar (no file, or override "0") never
// reassigns these fields, so a stale pointer from a prior load would otherwise look valid. Pure field
// clearing only -- no I/O, safe to unit-test without a GGUF or a GPU.
inline void reset_qwen4exp_mtp_fields(Qwen4ExpWeights & w) {
    w.mtp = Qwen4ExpLayer{};
    w.mtp_enorm = w.mtp_hnorm = w.mtp_eh_proj = nullptr;
    w.mtp_head_norm = w.mtp_head_down = w.mtp_head_up = nullptr;
    w.mtp_output = w.mtp_embd = nullptr;
    w.mtp_vocab_ids.clear();
}

// Load the autoregressive trunk of a Qwen3.8-Flash-Next (`qwen4exp`) GGUF.
// Split models ("-00001-of-00003.gguf") load every shard; single-file GGUFs
// load as one shard. The per_layer_token_embd table is discovered in whichever
// shard holds it and opened lazily. An MTP sidecar (find_qwen4exp_mtp_sidecar)
// loads as one more shard. Returns false and sets last_error on failure.
bool load_qwen4exp_gguf(const std::string & path,
                        ggml_backend_t backend,
                        Qwen4ExpWeights & out,
                        const std::string & mtp_override = "",
                        int mtp_vocab = QWEN4EXP_MTP_VOCAB); // empty = discovery, "0" = off

// Discover an MTP sidecar in the Unsloth
// layout <repo>/<quant>/<model>.gguf -> <repo>/MTP/mtp-*.gguf is searched. Empty when there is none.
std::string find_qwen4exp_mtp_sidecar(const std::string & model_path);

// The sidecar among the file names of a repo's MTP/ directory: the first mtp-*.gguf in name order, or empty.
std::string pick_qwen4exp_mtp_sidecar(std::vector<std::string> names);

void free_qwen4exp_weights(Qwen4ExpWeights & w);

}  // namespace luce::common
