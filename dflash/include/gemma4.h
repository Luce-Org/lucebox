// gemma4 — standalone CUDA library for DFlash speculative decoding of
// Gemma4 models (31B Dense and 26B-A4B MoE) with a DFlash draft model.

#ifndef GEMMA4_H
#define GEMMA4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Gemma4-31B Dense config ───────────────────────────────────────

#define GEMMA4_31B_HIDDEN              4096
#define GEMMA4_31B_LAYERS              60
#define GEMMA4_31B_N_HEADS             32
#define GEMMA4_31B_N_KV_HEADS          8
#define GEMMA4_31B_HEAD_DIM            128
#define GEMMA4_31B_INTERMEDIATE        16384
#define GEMMA4_31B_VOCAB               262144
#define GEMMA4_31B_SWA_WINDOW          1024

// ─── Gemma4-26B-A4B MoE config ────────────────────────────────────

#define GEMMA4_26B_HIDDEN              4096
#define GEMMA4_26B_LAYERS              30
#define GEMMA4_26B_N_HEADS             32
#define GEMMA4_26B_N_KV_HEADS          8
#define GEMMA4_26B_HEAD_DIM            128
#define GEMMA4_26B_INTERMEDIATE        16384
#define GEMMA4_26B_EXPERT_INTERMEDIATE 2048
#define GEMMA4_26B_N_EXPERTS           128
#define GEMMA4_26B_N_EXPERTS_USED      8
#define GEMMA4_26B_VOCAB               262144
#define GEMMA4_26B_SWA_WINDOW          1024

// ─── Shared constants ─────────────────────────────────────────────

#define GEMMA4_ROPE_THETA              1000000.0f
#define GEMMA4_RMS_EPS                 1e-6f
#define GEMMA4_LOGIT_SOFTCAP           30.0f
#define GEMMA4_ATTN_SCALE              1.0f

// ─── Draft model config ───────────────────────────────────────────

#define GEMMA4_DRAFT_LAYERS            5
#define GEMMA4_DRAFT_BLOCK_SIZE        16
#define GEMMA4_DRAFT_N_TARGET_LAYERS   6
#define GEMMA4_31B_DRAFT_MASK_TOKEN_ID 4
#define GEMMA4_26B_DRAFT_MASK_TOKEN_ID 4

// ─── Diagnostics ──────────────────────────────────────────────────

const char * gemma4_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // GEMMA4_H
