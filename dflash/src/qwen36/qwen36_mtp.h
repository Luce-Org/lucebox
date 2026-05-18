// qwen36_mtp.h — Native-heads MTP module for unsloth Qwen3.6 GGUFs.
//
// Qwen3.6's GGUF architecture is `qwen35` (same backbone). The MTP heads are
// stored as additional per-layer tensors on the last `qwen35.nextn_predict_layers`
// blocks of the GGUF, following DeepSeek-V3 / NextN conventions:
//
//   blk.{bid}.nextn.eh_proj          : [2*n_embd, n_embd]   — concat[embed;hidden] -> embed
//   blk.{bid}.nextn.enorm            : [n_embd]              — embed-side norm
//   blk.{bid}.nextn.hnorm            : [n_embd]              — hidden-side norm
//   blk.{bid}.nextn.embed_tokens     : [n_embd, n_vocab]     — optional; shared with backbone if absent
//   blk.{bid}.nextn.shared_head_head : [n_embd, n_vocab]     — optional; shared with backbone if absent
//   blk.{bid}.nextn.shared_head_norm : [n_embd]              — optional; shared with backbone if absent
//
// Each MTP head also carries its own transformer block tensors (attn_q/k/v/o,
// ffn_*, ssm_*) at the same layer index — these were already loaded for the
// backbone forward (qwen35 path). The MTP forward reuses them via the
// supplied Qwen35Backend / DFlashTarget.
//
// Per unsloth: recommended γ ≤ 2 (accept rate drops from ~83% at γ=1 to
// ~50% at γ=4). `max_gamma()` honors `nextn_predict_layers` from GGUF.
//
// This module implements INativeMtp from mtp_interface.h: `step_batch` emits
// up to num_heads() draft tokens per call.

#pragma once

#include "common/mtp_interface.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;

namespace dflash27b {

struct DFlashTarget;

namespace mtp {

// One Qwen3.6 MTP head's weights. There are `n_heads` such entries; head i
// corresponds to GGUF block index `(n_layer - n_heads + i)`.
//
// Shape B (DeepSeek-V3 NextN): each head owns a full transformer-block at its
// GGUF block index. These tensors are loaded from blk.{layer_idx}.* (no nextn.
// prefix). See qwen36_mtp_redesign.md for the verified GGUF tensor inventory.
struct Qwen36MtpHeadWeights {
    int layer_idx     = -1;            // absolute GGUF block index
    // NextN-specific tensors (required)
    ggml_tensor * eh_proj          = nullptr;   // [n_embd, 2*n_embd]
    ggml_tensor * enorm            = nullptr;   // [n_embd]
    ggml_tensor * hnorm            = nullptr;   // [n_embd]
    ggml_tensor * embed_tokens     = nullptr;   // optional (nullable -> use backbone)
    ggml_tensor * shared_head_head = nullptr;   // optional (nullable -> use backbone)
    ggml_tensor * shared_head_norm = nullptr;   // optional (nullable -> use backbone)
    // Head-owned transformer block (Shape B — required, not shared with backbone)
    ggml_tensor * attn_norm           = nullptr;   // [n_embd]
    ggml_tensor * attn_q              = nullptr;   // [n_embd, head_count * key_length]
    ggml_tensor * attn_q_norm         = nullptr;   // [key_length]
    ggml_tensor * attn_k              = nullptr;   // [n_embd, head_count_kv * key_length]
    ggml_tensor * attn_k_norm         = nullptr;   // [key_length]
    ggml_tensor * attn_v              = nullptr;   // [n_embd, head_count_kv * value_length]
    ggml_tensor * attn_output         = nullptr;   // [head_count * value_length, n_embd]
    ggml_tensor * post_attention_norm = nullptr;   // [n_embd]
    ggml_tensor * ffn_gate            = nullptr;   // [n_embd, ffn_length]
    ggml_tensor * ffn_up              = nullptr;   // [n_embd, ffn_length]
    ggml_tensor * ffn_down            = nullptr;   // [ffn_length, n_embd]
};

struct Qwen36MtpWeights {
    int                                   n_embd            = 0;
    int                                   n_vocab           = 0;
    int                                   n_heads           = 0;   // == nextn_predict_layers
    int                                   n_backbone_layers = 0;   // total backbone n_layer
    std::vector<Qwen36MtpHeadWeights>     heads;                   // size == n_heads
    // GGUF metadata copies for cross-checks
    std::string                           backbone_arch;           // e.g. "qwen35"
    std::string                           base_model_name;         // e.g. "Qwen3.6-27B"
    // Attention sizing — read from GGUF; needed to size per-head KV buffers.
    int                                   n_head_count    = 0;   // qwen35.attention.head_count
    int                                   n_head_kv       = 0;   // qwen35.attention.head_count_kv
    int                                   n_key_length    = 0;   // qwen35.attention.key_length
    int                                   n_value_length  = 0;   // qwen35.attention.value_length
    int                                   n_ffn_length    = 0;   // qwen35.feed_forward_length
};

// Load Qwen3.6 MTP weights from a GGUF file. Returns false if the file does
// not contain NextN tensors (i.e. it is a non-MTP GGUF). The `ctx` parameter
// owns the loaded tensors' lifetime; pass the same ctx used for backbone.
//
// `expected_n_embd` / `expected_n_vocab` are sanity-checked against GGUF
// metadata; pass values from the bound target.
bool load_qwen36_mtp_weights(const std::string & gguf_path,
                             ggml_context * ctx,
                             int expected_n_embd,
                             int expected_n_vocab,
                             Qwen36MtpWeights & out_weights,
                             std::string & out_error);

// Concrete INativeMtp impl for Qwen3.6 (unsloth -MTP-GGUF). Wraps the
// loaded weights + a bound DFlashTarget (typically Qwen35Backend's
// dflash_target()) which provides backbone embedding, KV, LM head.
class Qwen36MtpModule : public INativeMtp {
public:
    Qwen36MtpModule();
    ~Qwen36MtpModule() override;

    // One-shot construction: load weights then attach in a single call.
    // The returned module is ready for step_batch() once attach() succeeds.
    //
    // n_ctx_request: chain horizon for the head-KV cache. When > 0, the
    // module sizes its per-head K/V tensors to hold this many slots — must
    // be >= the backbone's max_ctx. When 0, falls back to the env override
    // (DFLASH27B_MTP_CTX) and finally to the legacy 8192 default; this leg
    // is retained for the test/stub paths that don't drive the daemon.
    bool init(const std::string & gguf_path,
              DFlashTarget * target,
              std::string & out_error,
              int n_ctx_request = 0);

    // Integration construction: bind NextN tensors from the backbone GGUF
    // context. The context lifetime stays with the caller; this module owns
    // only the CPU buffer used to materialize the NextN tensors it reads.
    bool init(const std::string & gguf_path,
              ggml_context * ctx,
              DFlashTarget * target,
              std::string & out_error,
              int n_ctx_request = 0);

    // ── IMtpModule ──────────────────────────────────────────────────
    int  max_gamma()       const override;
    int  effective_gamma() const override { return effective_gamma_; }
    void set_effective_gamma(int gamma) override {
        effective_gamma_ = (gamma > 0) ? std::min(gamma, max_gamma()) : max_gamma();
    }
    int  hidden_size()  const override;
    bool attach(DFlashTarget * target) override;
    void reset_chain() override;
    void shutdown() override;

    // ── INativeMtp ─────────────────────────────────────────────────
    int  num_heads() const override;
    bool step_batch(int32_t current_token,
                    int base_pos,
                    std::vector<StepOutput> & out) override;
    void set_draft_topk(int k) override;

    // Autoregressive chain draft: runs the head `chain_depth` times,
    // feeding the previous iteration's post-shared_head_norm hidden as
    // h_prev for the next.  Per-iter graph is rebuilt today (Phase A
    // honest baseline; graph caching is Phase B).  On the CPU stub path
    // (no backend) this falls back to the default `step_batch`+clamp.
    bool step_chain(int32_t current_token,
                    int base_pos,
                    int chain_depth,
                    std::vector<StepOutput> & out) override;

    // Receive the backbone's final post-norm hidden state for the last committed
    // token. Called by the chain runner before each step_batch(). The pointer
    // and dim must remain valid for the duration of step_batch(); the module
    // does NOT copy or own this buffer in PR 2c-bis (deferred to 2d-bis when
    // the forward actually uses it).
    void set_initial_hidden(const float * h_prev, int dim) override;

    // Pre-warm the head's K/V cache by running the head's K/V projections on
    // every prefill position. After this call, head_kv[0] contains valid K/V
    // entries at slots [0, n_prompt-1]; step_batch's range attention can then
    // attend to the full prompt context instead of seeing a single slot.
    //
    // Arguments:
    //   prompt_tokens : the prompt sequence, length n_prompt.
    //   n_prompt      : number of prompt tokens.
    //   prefill_next  : the backbone's argmax at the end of prefill — used as
    //                    t_{n_prompt} for slot n_prompt-1 (per DeepSeek-V3
    //                    Eq 21 the head at index i reads Emb(t_{i+k})).
    //   hiddens       : the backbone's per-position post-norm hidden states,
    //                    laid out as [token_0_hidden, ..., token_{n_prompt-1}_hidden].
    //                    Each token's hidden is hidden_size() floats, F32.
    // Returns false on dimension mismatch or tensor-dequant failure.
    bool warm_head_kv(const int32_t * prompt_tokens,
                      int             n_prompt,
                      int32_t         prefill_next,
                      const float *   hiddens) override;
    bool snapshot_head_kv(std::vector<std::vector<uint8_t>> & out_buf,
                          std::vector<int> & out_pos) const override;
    bool restore_head_kv(const std::vector<std::vector<uint8_t>> & buf,
                         const std::vector<int> & pos) override;
    bool warm_head_kv_range(const int32_t * prompt_tokens, int n_prompt,
                            int start_slot, int n_chunk,
                            int32_t prefill_next, const float * hiddens) override;

    // Read-only access for tests / introspection.
    const Qwen36MtpWeights & weights() const;

    // Inject pre-built weights for unit tests without a real GGUF file.
    // The "_for_test" / "test_" prefix is the contract: production code uses init().
    void attach_weights_for_test(const Qwen36MtpWeights & w);

    // Test-only accessors: return the last set_initial_hidden state.
    // Used by test_qwen36_mtp_step_unit; not for production paths.
    const float * test_initial_hidden_ptr() const;
    int           test_initial_hidden_dim() const;

private:
    struct State;
    std::unique_ptr<State> state_;
    int effective_gamma_ = 0;  // 0 until set_effective_gamma; orchestrator MUST set before decode

    // GPU forward path (cgraph on backbone backend); falls back to the CPU
    // path inside step_batch() when no CUDA backend is available.
    bool step_batch_gpu_(int32_t current_token,
                         int base_pos,
                         std::vector<StepOutput> & out);

    // Bug #5 fix: graphs are shape-only, keyed on (head_idx, fa_window,
    // fused_lm_head, topk_k).  Per-call slot routing (write idx, read
    // idxs, mask) is pushed as runtime tensor inputs by the caller.
    struct Qwen36MtpStepGraph * get_or_build_step_graph_(int head_idx);

    // Grow the GPU head_kv tensors to hold `required_slots`, rounded up to
    // a 1024-slot quantum and capped at the per-init n_ctx_max ceiling. A
    // no-op when current capacity already suffices or when running the CPU
    // stub. Safe to call only between requests — invalidates cached step
    // graphs and zeroes head_kv_pos.
    bool ensure_head_kv_capacity_(int required_slots);
};

}  // namespace mtp
}  // namespace dflash27b
