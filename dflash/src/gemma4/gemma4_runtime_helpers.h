// Helpers for Gemma4Backend chunked prefill, autoregressive decode,
// and DFlash speculative-decode (DraftKVPrefillGraph, DraftStepGraph,
// AdaptiveDraftMax, copy_target_feat_bf16_to_f32).
//
// Ported from feature/gemma4-support dflash/test/gemma4/test_gemma4_dflash.cpp
// (ranges 213-275, 280-450, 342-425, 525-612, 690-770 per the source digests).

#pragma once

#include "../internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dflash27b {

// ─── KQ mask padding constants ───────────────────────────────────────────────
//
// ggml_flash_attn_ext requires kv_len aligned to g_kq_stride_pad (32 for Q8/F16
// KV, 256 for TQ3_0 KV). KQ_MASK_PAD aligns the q dimension. These values match
// the runtime that build_gemma4_graph expects — the mask tensor dimensions must
// match the tensors the graph allocates for attn_mask / swa_mask.
static constexpr int      KQ_MASK_PAD    = 32;
static constexpr int      g_kq_stride_pad = 32;
static constexpr uint16_t F16_NEG_INF    = 0xFC00u;
static constexpr uint16_t F16_ZERO       = 0x0000u;

static inline int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// ─── Per-step graph state ────────────────────────────────────────────────────
//
// Rebuilt each forward pass because kv_len varies chunk to chunk.
// Freed and reallocated via build_gemma4_step / step_graph_free.

struct StepGraph {
    ggml_context   * ctx        = nullptr;
    ggml_cgraph    * gf         = nullptr;
    ggml_gallocr_t   alloc      = nullptr;
    ggml_tensor    * inp_embed  = nullptr;
    ggml_tensor    * positions  = nullptr;
    ggml_tensor    * attn_mask  = nullptr;
    ggml_tensor    * swa_mask   = nullptr;
    ggml_tensor    * logits     = nullptr;
};

void step_graph_free(StepGraph & sg);

// ─── Draft KV prefill graph state ────────────────────────────────────────────
//
// Projects n_tokens target features (bf16 ring → f32) into the draft KV cache.
// Each forward writes cache.draft_kv_pos..+n_tokens into draft_k/draft_v.
// Ported from test_gemma4_dflash.cpp Range E (lines 280-450).

struct DraftKVPrefillGraph {
    ggml_context   * ctx         = nullptr;
    ggml_cgraph    * gf          = nullptr;
    ggml_gallocr_t   alloc       = nullptr;
    ggml_tensor    * target_feat = nullptr;  // input: [n_target_layers*target_hidden, n_tokens] f32
    ggml_tensor    * positions   = nullptr;  // input: [n_tokens] i32
};

// Build and allocate a draft KV prefill graph for n_tokens positions.
// On success pkg.gf is ready for ggml_backend_graph_compute; caller must set
// pkg.target_feat and pkg.positions before computing, then call
// draft_kv_prefill_destroy when done.
bool draft_kv_prefill_create(DraftKVPrefillGraph & pkg,
                             const GemmaDraftWeights & dw,
                             GemmaTargetCache & cache,
                             ggml_backend_t backend,
                             int n_tokens);

void draft_kv_prefill_destroy(DraftKVPrefillGraph & pkg);

// ─── Draft step graph state ──────────────────────────────────────────────────
//
// One forward through the DFlash draft model. Build once per speculative block,
// compute, read logits. Wraps build_gemma4_draft_graph.
// Ported from test_gemma4_dflash.cpp Range E (lines ~350-380).

struct DraftStepGraph {
    ggml_context   * ctx         = nullptr;
    ggml_cgraph    * gf          = nullptr;
    ggml_gallocr_t   alloc       = nullptr;
    ggml_tensor    * draft_embed = nullptr;  // input: [n_embd, q_len] f32
    ggml_tensor    * positions   = nullptr;  // input: [q_len] i32
    ggml_tensor    * attn_mask   = nullptr;  // input: [kv_pad, q_pad] f16
    ggml_tensor    * logits      = nullptr;  // output: [vocab, q_len] f32
};

// Build and allocate a draft step graph for q_len tokens at KV offset kv_start.
bool draft_step_build(DraftStepGraph & dsg,
                      const GemmaDraftWeights & dw,
                      GemmaTargetCache & cache,
                      ggml_backend_t backend,
                      int q_len,
                      int kv_start);

void draft_step_free(DraftStepGraph & dsg);

// ─── Adaptive draft length controller ────────────────────────────────────────
//
// Tracks accept-rate over a sliding window; doubles q_len on high fill,
// halves on low fill. Ported from test_gemma4_dflash.cpp Range F (lines 690-770).
// printf scaffolding stripped.

struct AdaptiveDraftMax {
    bool enabled           = false;
    int  current           = 0;
    int  min_q             = 1;
    int  max_q             = 0;
    int  window_steps      = 8;
    int  window_accepted   = 0;
    int  window_capacity   = 0;
    int  window_steps_seen = 0;

    void init(bool on, int initial, int block_size) {
        enabled = on;
        max_q   = block_size;
        current = (initial > 0) ? std::min(initial, block_size) : block_size;
        current = std::max(min_q, current);
    }

    void observe(int accepted, int q_len, int /*step_no*/) {
        if (!enabled) return;
        // accepted includes the pinned cur_tok; adapt on the speculative fill.
        window_accepted   += std::max(0, accepted - 1);
        window_capacity   += std::max(1, q_len - 1);
        window_steps_seen++;
        if (window_steps_seen < window_steps || window_capacity <= 0) return;

        const double fill = (double)window_accepted / (double)window_capacity;
        if (fill < 0.35 && current > min_q) {
            current = std::max(min_q, current / 2);
        } else if (fill > 0.78 && current < max_q) {
            current = std::min(max_q, current * 2);
        }
        window_accepted   = 0;
        window_capacity   = 0;
        window_steps_seen = 0;
    }
};

// ─── Target-feature ring-buffer copy ────────────────────────────────────────
//
// Copies n_tokens rows from cache.target_feat (bf16 ring, shape
// [target_feat_w, target_feat_cap]) starting at slot start_slot into the
// pre-allocated f32 tensor dst_feat ([target_feat_w, n_tokens]).
//
// The ring may wrap; this is handled via two separate host memcpy segments.
// Uses ggml_backend_tensor_get (GPU→CPU) then ggml_backend_tensor_set (CPU→GPU).

void copy_target_feat_bf16_to_f32(ggml_backend_t backend,
                                   ggml_tensor * src_bf16,
                                   ggml_tensor * dst_f32,
                                   int start_slot,
                                   int n_tokens,
                                   int target_feat_w);

// ─── Attention mask builders ─────────────────────────────────────────────────

void build_causal_mask(std::vector<uint16_t> & out,
                       int kv_len, int n_tokens, int kv_start);

// Non-monotonic ring mask for SWA layers.
// ring_size = swa_view.effective_win_len = swa_ctx_alloc.
// kv_end    = kv_start + n_tokens.
void build_swa_causal_mask(std::vector<uint16_t> & out,
                            int kv_start,
                            int n_tokens,
                            int swa_window,
                            int ring_size,
                            int kv_end);

// ─── Graph builder ───────────────────────────────────────────────────────────

bool build_gemma4_step(StepGraph & sg,
                       const GemmaTargetWeights & w,
                       GemmaTargetCache & cache,
                       ggml_backend_t backend,
                       int kv_start,
                       int n_tokens,
                       bool with_mask,
                       bool capture,
                       bool use_pflash            = false,
                       float pflash_alpha         = 0.12f,
                       int fa_window              = 0,
                       bool last_token_logits_only = false);

// ─── Token embedding ─────────────────────────────────────────────────────────

bool embed_token(const GemmaTargetWeights & w,
                 int32_t tok,
                 ggml_tensor * inp_embed,
                 ggml_backend_t backend);

bool embed_tokens_batch(const GemmaTargetWeights & w,
                        const int32_t * ids,
                        int n,
                        ggml_tensor * inp_embed,
                        ggml_backend_t backend);

}  // namespace dflash27b
