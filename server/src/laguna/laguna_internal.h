// Laguna-XS.2 (Poolside) target structs for dflash daemon. Mirrors the qwen35
// abstractions in internal.h but for an iSWA + MoE arch with no SSM/delta-net
// and per-layer-varying head count.
//
// Architecture summary (from poolside/Laguna-XS.2 config):
//   - 40 layers. Pattern (FULL, SWA, SWA, SWA) x 10. Sliding window = 512.
//   - Per-layer head count: 48 on FULL layers, 64 on SWA layers. n_head_kv = 8 always.
//   - head_dim = 128. n_embd = 2048.
//   - Q-norm + K-norm (RMSNorm at head_dim level).
//   - Per-head softplus attention gate: g_proj : hidden -> n_head, applied as
//     attn_out.view(*, n_head, head_dim) * softplus(g_proj(x)).unsqueeze(-1).
//   - Per-layer-type RoPE:
//       FULL: YaRN (theta=500000, factor=32, partial_rotary=0.5, original=4096,
//             beta_slow=1, beta_fast=64). n_rot = 64.
//       SWA:  default (theta=10000, partial_rotary=1.0). n_rot = 128.
//   - Layer 0: dense SwiGLU (n_ff = 8192). Layers 1..39: sparse MoE
//     (256 experts, top-8, sigmoid router with score-correction bias,
//     sum-normalize selected weights, scale = 2.5) + always-on shared expert
//     SwiGLU (intermediate = 512).
//   - Vocab = 100352. BOS = 2. EOS = {2, 24}. Pad = 9.

#pragma once
#define DFLASH_LAGUNA_INTERNAL_H_INCLUDED

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "common/layer_split_utils.h"
#include "internal.h"  // for CpuEmbedder

namespace dflash::common {

struct LagunaTargetLayer {
    // Pre-attn + pre-ffn norms (Laguna has only these two; no post norms).
    ggml_tensor * attn_norm   = nullptr;  // [n_embd]
    ggml_tensor * ffn_norm    = nullptr;  // [n_embd]

    // Attention projections. Q/O sizing is per-layer (n_head[il] varies).
    ggml_tensor * wq          = nullptr;  // [n_embd, n_head[il] * head_dim]
    ggml_tensor * wk          = nullptr;  // [n_embd, n_head_kv * head_dim]
    ggml_tensor * wv          = nullptr;  // [n_embd, n_head_kv * head_dim]
    ggml_tensor * wo          = nullptr;  // [n_head[il] * head_dim, n_embd]

    // Q/K head-dim RMSNorm (Qwen3-style).
    ggml_tensor * q_norm      = nullptr;  // [head_dim]
    ggml_tensor * k_norm      = nullptr;  // [head_dim]

    // Per-head softplus gate. wqkv_gate shape: [n_embd, n_head[il]] (NOT n_head*head_dim).
    // Forward: gate = softplus(x @ wqkv_gate.float()), broadcast over head_dim onto attn_out.
    ggml_tensor * wqkv_gate   = nullptr;  // [n_embd, n_head[il]]

    // Dense MLP (layer 0 only — nullptr for sparse layers).
    ggml_tensor * w_gate      = nullptr;  // [n_embd, n_ff]
    ggml_tensor * w_up        = nullptr;  // [n_embd, n_ff]
    ggml_tensor * w_down      = nullptr;  // [n_ff, n_embd]

    // Sparse MoE (layers 1..n-1 only — nullptr for dense layer 0).
    ggml_tensor * ffn_gate_inp     = nullptr;  // [n_embd, n_expert]      router weights
    ggml_tensor * ffn_exp_probs_b  = nullptr;  // [n_expert]              score-correction bias
    ggml_tensor * ffn_gate_exps    = nullptr;  // [n_embd, n_ff_exp, n_expert]  stacked routed gate
    ggml_tensor * ffn_up_exps      = nullptr;  // [n_embd, n_ff_exp, n_expert]  stacked routed up
    ggml_tensor * ffn_down_exps    = nullptr;  // [n_ff_exp, n_embd, n_expert]  stacked routed down
    ggml_tensor * ffn_gate_shexp   = nullptr;  // [n_embd, n_ff_shexp]    shared expert gate
    ggml_tensor * ffn_up_shexp     = nullptr;  // [n_embd, n_ff_shexp]    shared expert up
    ggml_tensor * ffn_down_shexp   = nullptr;  // [n_ff_shexp, n_embd]    shared expert down

    // Fused views over ADJACENT weight pairs (the loader lays q|k and
    // gate_shexp|up_shexp out back-to-back and binds one tensor over both
    // regions; zero extra VRAM). Null when the pair could not be fused
    // (type mismatch / DFLASH_LAGUNA_FUSED_QK=0). Same data as wq/wk etc.
    ggml_tensor * wqk       = nullptr;  // [n_embd, (n_head+n_head_kv)*head_dim]
    ggml_tensor * shexp_gu  = nullptr;  // [n_embd, 2*n_ff_shexp]  gate rows then up rows
    ggml_tensor * qk_norm_f = nullptr;  // [head_dim, n_head+n_head_kv] f32, q_norm|k_norm per head
};

struct LagunaTargetWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // CPU-side embedding table (matches qwen35 strategy: tok_embd stays on host
    // because CUDA's get_rows doesn't support k-quants).
    CpuEmbedder           embedder;

    ggml_tensor * tok_embd = nullptr;        // metadata only; data on host
    std::vector<LagunaTargetLayer> layers;   // size = n_layer
    ggml_tensor * out_norm = nullptr;        // [n_embd]
    ggml_tensor * output   = nullptr;        // [n_embd, n_vocab]  (lm_head)

    // Metadata contexts/buffer for the fused-weight views (see LagunaTargetLayer).
    ggml_context *        fuse_ctx      = nullptr;
    ggml_context *        fuse_norm_ctx = nullptr;
    ggml_backend_buffer_t fuse_norm_buf = nullptr;

    // Architecture metadata (validated at load time).
    int  n_layer              = 40;
    int  n_head_kv            = 8;
    int  head_dim             = 128;
    int  n_embd               = 2048;
    int  n_ff                 = 8192;
    int  n_ff_exp             = 512;
    int  n_ff_shexp           = 512;
    int  n_expert             = 256;
    int  n_expert_used        = 8;
    int  n_layer_dense_lead   = 1;     // layer 0 dense, rest sparse
    int  sliding_window       = 512;
    int  swa_pattern          = 4;     // (full, sw, sw, sw) repeating

    // Per-layer head count: 48 (full) / 64 (sliding). Always n_layer entries.
    int  n_head_arr[40]       = { 48,64,64,64, 48,64,64,64, 48,64,64,64, 48,64,64,64, 48,64,64,64,
                                  48,64,64,64, 48,64,64,64, 48,64,64,64, 48,64,64,64, 48,64,64,64 };

    // RoPE per layer-type.
    float rope_freq_base_full = 500000.0f;
    float rope_freq_base_swa  = 10000.0f;
    int   n_rot_full          = 64;     // partial_rotary_factor=0.5 on full layers
    int   n_rot_swa           = 128;    // partial_rotary_factor=1.0 on sliding
    // YaRN params for full layers:
    float yarn_factor         = 32.0f;
    float yarn_beta_fast      = 64.0f;
    float yarn_beta_slow      = 1.0f;
    int   yarn_orig_ctx       = 4096;

    // MoE gating + scaling.
    bool  expert_gating_sigmoid = true;   // sigmoid (vs softmax) router
    bool  expert_weights_norm   = true;   // sum-normalize selected weights
    float expert_weights_scale  = 2.5f;   // routed combine scale

    // Tokenizer special tokens.
    int32_t bos_id      = 2;
    int32_t eos_id      = 2;
    int32_t eos_chat_id = 24;             // second eos id in HF config
    int32_t pad_id      = 9;
};

// True if layer `il` uses full attention (the FIRST layer in each group of
// `swa_pattern` is full; the remaining are sliding-window).
inline bool laguna_is_full_attn_layer(const LagunaTargetWeights & w, int il) {
    return (il % w.swa_pattern) == 0;
}

// Loader. Validates arch == "laguna", reads all hparams, mmaps GGUF, copies
// tensors to backend buffer. Returns false + sets dflash27b_last_error on failure.
bool load_target_gguf_laguna(const std::string & path,
                              ggml_backend_t       backend,
                              LagunaTargetWeights & out);

// Partial loader. With plan.skip_expert_tensors=true this performs the
// hybrid-MoE load: non-expert tensors go to GPU, expert tensors are kept
// off-GPU (metadata/shapes stay valid for size queries). Also supports
// layer-range partial loads via plan.layer_begin/layer_end/load_output.
bool load_target_gguf_laguna_partial(const std::string & path,
                                      ggml_backend_t backend,
                                      const TargetLoadPlan & plan,
                                      LagunaTargetWeights & out);

void free_laguna_target_weights(LagunaTargetWeights & w);

// ---- Forward graph (Phase 2; signatures only for now) -------------------

struct LagunaTargetCache {
    ggml_context *        base_ctx = nullptr;
    ggml_backend_buffer_t base_buf = nullptr;
    ggml_backend_t        backend  = nullptr;

    int max_ctx  = 0;
    int cur_pos  = 0;
    int last_tok = -1;

    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;

    // Per-layer KV cache. ALL 40 layers have KV (both full + swa).
    // Default layout: [head_dim, max_ctx, n_head_kv].
    // With DFLASH_LAGUNA_KV_HEAD_MAJOR=1: [head_dim*n_head_kv, max_ctx],
    // viewed as [head_dim, n_head_kv, max_ctx] for attention.
    bool kv_head_major = false;
    std::vector<ggml_tensor *> attn_k;   // size = n_layer
    std::vector<ggml_tensor *> attn_v;

    // DFlash feature capture ring buffer (BF16, allocated when draft is active).
    ggml_tensor *         target_feat = nullptr;  // [n_capture_layers*n_embd, target_feat_cap]
    int                   target_feat_cap = 0;
    int                   n_capture_layers = 0;
    std::vector<int>      capture_layer_ids;
    ggml_context *        feat_ctx = nullptr;
    ggml_backend_buffer_t feat_buf = nullptr;
};

// `ctx_alloc` (kvflash): when > 0 and < max_ctx, the per-layer K/V tensors
// are allocated at ctx_alloc rows (the resident pool) while cache.max_ctx
// keeps the logical bound. 0 = allocate at max_ctx (default).
bool create_laguna_target_cache(const LagunaTargetWeights & w,
                                 int max_ctx,
                                 ggml_backend_t backend,
                                 LagunaTargetCache & out,
                                 int ctx_alloc = 0);
bool create_laguna_target_cache_partial(const LagunaTargetWeights & w,
                                         int max_ctx,
                                         ggml_backend_t backend,
                                         int layer_begin,
                                         int layer_end,
                                         LagunaTargetCache & out,
                                         int ctx_alloc = 0);
void free_laguna_target_cache(LagunaTargetCache & c);
void reset_laguna_target_cache(LagunaTargetCache & c);

void free_laguna_target_feat(LagunaTargetCache & c);
bool create_laguna_target_feat(ggml_backend_t backend,
                                LagunaTargetCache & cache,
                                int n_capture_layers,
                                int hidden_size,
                                int cap,
                                const std::vector<int> & explicit_ids = {});

// ----------------------------------------------------------------------------
// Cache snapshots for prefix-cache slots (PrefixCache).
//
// A snapshot holds a parallel set of K/V tensors with the same shapes and
// dtypes as the live LagunaTargetCache, plus the cur_pos at which the
// snapshot was taken. SNAPSHOT performs a device-to-device copy of every
// per-layer K and V tensor from the live cache into the snapshot buffer;
// RESTORE does the reverse and resets cur_pos. The buffers are allocated
// lazily on first SNAPSHOT and freed on FREE_SNAPSHOT.
// ----------------------------------------------------------------------------
struct LagunaCacheSnapshot {
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    std::vector<ggml_tensor *> attn_k;  // size = n_layer
    std::vector<ggml_tensor *> attn_v;  // size = n_layer
    ggml_tensor *         feat_snap = nullptr;
    int                   feat_cap  = 0;
    int                   cur_pos = 0;
    // Last committed token id (the token at cur_pos-1). Recorded at save so the
    // exact-hit restore can re-embed the real last token even when the caller
    // hands us a zero-filled restore-only prompt (pflash full-cache hit). -1 =
    // not recorded (older/inline snapshots) -> restore falls back to req.prompt.
    int                   last_tok = -1;
    bool                  used    = false;
};

bool laguna_snapshot_alloc(const LagunaTargetCache & cache,
                            ggml_backend_t            backend,
                            int                       n_layer,
                            int                       snap_pos,
                            int                       n_head_kv,
                            int                       head_dim,
                            LagunaCacheSnapshot &     out);

void laguna_snapshot_free(LagunaCacheSnapshot & snap);

bool laguna_snapshot_save(const LagunaTargetCache & cache,
                           ggml_backend_t            backend,
                           int                       n_layer,
                           int                       n_head_kv,
                           int                       head_dim,
                           LagunaCacheSnapshot &     snap);

bool laguna_snapshot_restore(const LagunaCacheSnapshot & snap,
                              LagunaTargetCache &         cache);

struct MoeHybridStorage;  // defined in common/moe_hybrid_storage.h
// Descriptor that turns the shared graph builders into the hybrid-offload
// path: MoE layers serve experts from the bounded hot stack via per-layer
// residency LUTs. storage==null => full all-GPU expert stack (default).
struct LagunaHybridMoe {
    const MoeHybridStorage * storage = nullptr;
    ggml_tensor * lut_all = nullptr;  // [n_expert, n_moe] i32 global->hot-slot (caller-set)
    ggml_tensor * vld_all = nullptr;  // [n_expert, n_moe] f32 1=resident 0=drop (caller-set)
    ggml_tensor * sel_all = nullptr;  // [n_used,  n_moe] i32 selected ids (graph writes per layer)
    int dense_lead = 0;               // moe_idx = il - dense_lead
};

struct LagunaGraphInputs {
    ggml_tensor * inp_embed;      // [n_embd, n_tokens, 1] f32 (CPU-embedded by caller)
    ggml_tensor * positions;      // [n_tokens] i32 (NeoX rope; not M-RoPE)
    ggml_tensor * attn_mask;      // optional [kv_len, n_tokens] F16 (causal) for FULL layers
    ggml_tensor * attn_mask_swa;  // optional [kv_len, n_tokens] F16 (causal + sliding window) for SWA layers
    int           n_tokens;
    int           kv_start;
    // CUDA-graph-stable decode: when kv_pad > 0 the FA K/V views + masks span
    // kv_pad slots (stride-rounded; mask gates validity) and the K/V append
    // goes through ggml_set_rows with kv_idx as a graph INPUT, so node
    // properties stay identical across decode steps and the ggml-cuda
    // CUDA-graph cache replays instead of re-launching every kernel.
    int           kv_pad = 0;             // 0 = legacy exact-length views + cpy append
    ggml_tensor * kv_idx = nullptr;       // [n_tokens] I32 cache row indices (graph input)
    bool          output_logits = true;
    bool          logits_are_output = true;
    bool          output_hidden_states = false;
    bool          capture_features = false;
    ggml_tensor * target_feat_rows = nullptr;  // optional [n_tokens] I32 ring rows for replay-stable capture
    // If true, lm_head only runs on the LAST token (saves ~6 GB of logit memory
    // at n_tokens=16K, vocab=100352). Standard TTFT optimization.
    bool          output_last_only = false;
    const LagunaHybridMoe * hybrid = nullptr;  // non-null => hybrid-offload MoE path
};

struct LagunaGraphOutputs {
    ggml_tensor * logits        = nullptr;  // [vocab, n_tokens] f32
    ggml_tensor * hidden_states = nullptr;  // [n_embd, n_tokens] f32
};

// Phase 2: implement.
LagunaGraphOutputs build_laguna_graph(
    ggml_context *               ctx,
    ggml_cgraph *                gf,
    const LagunaTargetWeights &  w,
    LagunaTargetCache &          cache,
    const LagunaGraphInputs &    in);

// Build + run a single Laguna forward step on a fresh ggml context. Used by
// both the daemon (laguna_daemon.cpp) and any caller that wants a turnkey
// prefill chunk or per-token decode without managing graph allocation.
// Builds BOTH a full causal mask (FULL layers) and a sliding-window-causal
// mask (SWA layers). Updates `cache.cur_pos` to `kv_start + n_tok` on
// success. Pass `no_mask=true` for kernel ablation — semantically wrong but
// useful for perf isolation.
//
// `embed`  : [n_tok, n_embd] f32 host buffer (caller pre-embeds via
//             CpuEmbedder).
// `out_logits` : on success, resized to vocab and filled with last-token
//             logits when in.output_last_only == true (default in this
//             helper).
// `kvflash`: optional bounded-residency pager (see common/kvflash_pager.h).
// When set, the K/V append rows come from the pager's slot mapping and both
// masks are built in SLOT space (causal / sliding-window conditions evaluated
// on the position each slot holds). The caller must have allocated slots for
// [kv_start, kv_start + n_tok) via slot_for() beforehand. Requires the
// kv_pad set_rows path (refused otherwise).
bool laguna_step(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache &         cache,
    const float *               embed,
    int                         n_tok,
    int                         kv_start,
    bool                        no_mask,
    std::vector<float> &        out_logits,
    const class KvFlashPager *  kvflash = nullptr,
    bool                        capture = false,
    int32_t *                   out_argmax = nullptr,
    bool                        read_logits = true);

bool laguna_verify_batch(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache &         cache,
    const float *               embed,
    const int32_t *             token_ids,
    int                         n_tokens,
    int                         kv_start,
    std::vector<int32_t> &      out_argmax,
    const class KvFlashPager *  kvflash = nullptr,
    std::vector<float> *        out_logits = nullptr);

bool laguna_project_hidden(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    const float *               hidden,
    int                         n_tokens,
    std::vector<int32_t> &      out_tokens);

// Forward decl (full definition in common/moe_hybrid_storage.h).
struct MoeHybridStorage;

// Single-graph hybrid decode: one full-token graph, MoE layers served from
// the hot expert stack via per-layer residency LUTs set once. One compute,
// one logits readback. Does NOT mutate cache.cur_pos (caller increments).
bool laguna_step_hybrid(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache &         cache,
    const float *               embed,
    int                         n_tok,
    int                         kv_start,
    bool                        no_mask,
    const MoeHybridStorage &    hyb,
    std::vector<float> &        out_logits,
    std::vector<int32_t> *      out_selected = nullptr,
    const class KvFlashPager *  kvflash = nullptr);

struct LagunaLayerStepGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * attn_mask = nullptr;
    ggml_tensor * attn_mask_swa = nullptr;
    ggml_tensor * kv_idx = nullptr;
};

void laguna_layer_step_graph_free(LagunaLayerStepGraph & sg);
void laguna_layer_step_graph_destroy(LagunaLayerStepGraph & sg);

bool build_laguna_layer_step(
    LagunaLayerStepGraph & sg,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    ggml_tensor * act_in,
    ggml_tensor * act_out,
    int chunk_start,
    int n_tokens,
    int kv_start,
    const class KvFlashPager * kvflash = nullptr);

bool compute_laguna_split_argmax(
    ggml_backend_t backend,
    const LagunaTargetWeights & w,
    ggml_tensor * act,
    int token_offset,
    int n_tokens,
    std::vector<int32_t> & out_argmax);

bool compute_laguna_split_projection(
    ggml_backend_t backend,
    const LagunaTargetWeights & w,
    ggml_tensor * act,
    int token_offset,
    int n_tokens,
    std::vector<int32_t> * out_argmax,
    std::vector<float> * out_logits);

} // namespace dflash::common
