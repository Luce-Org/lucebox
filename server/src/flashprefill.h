// Public C++ entry point for the FlashPrefill block-sparse attention used by
// the in-process Qwen3-0.6B drafter (speculative prefill scoring).
//
// Wraps kernels 1-4 + GPU block_select into one call. Call signature mirrors
// the upstream `flash_prefill` from qhfan/FlashPrefill (arXiv:2603.06199).
//
// Tensor layout (all CUDA, bf16, contiguous, D fastest):
//   Q[B, S, n_q_heads, D]
//   K[B, S, n_k_heads, D]
//   V[B, S, n_k_heads, D]
//   O[B, S, n_q_heads, D]
//
// Backends:
//   - Default: WMMA m16n16k16 sparse forward (qualified CUDA sm_80+ and HIP).
//   - Set env DFLASH_FP_USE_BSA=1 to dispatch to the Block-Sparse-Attention
//     kernel (FA-2 derived, m16n8k16 PTX, qualified CUDA sm_80+ except GB10
//     sm_121). PFlash stays on exact prefill there because neither bundled
//     custom BF16 sparse kernel is qualified.
//     Requires building with -DDFLASH27B_ENABLE_BSA=ON. ~3x faster than WMMA
//     on RTX 3090 at S=128K.
//
// Tunables (env vars):
//   DFLASH_FP_USE_BSA      [0/1] enable BSA backend (default: 0).
//   DFLASH_FP_ALPHA        [float in (0,1)] override FlashPrefillConfig.alpha.
//                          Higher = stricter selection = fewer K-blocks per Q
//                          row = faster but riskier. Default 0.12. For long
//                          context with broad needles, 0.85-0.99 work well.
//   DFLASH_FP_PROFILE      [set] log per-stage timing (mean / score / select /
//                          forward) to stderr.
//   DFLASH_FP_DUMP_COUNTS  [set] log per-row select counts to stderr.

#pragma once

#include <cstdint>
#include "ggml-backend.h"

namespace dflash::common {
namespace flashprefill {

// Keep the low-level BF16-kernel policy separate from the product-level local
// PFlash policy. Legacy CUDA architectures have other compiled fallbacks, but
// GB10's sm_121 is the one device on which the complete local scorer path is
// currently unsafe.
inline constexpr bool custom_bf16_sparse_supports_cuda_sm(int sm) {
    return sm >= 80 && sm != 121;
}
inline constexpr bool local_pflash_supports_cuda_sm(int sm) {
    return sm != 121;
}

// Runtime helpers are implemented in flashprefill.cpp, which is present in
// CUDA and custom-kernel HIP builds. Call the PFlash helper only from CUDA
// code; ordinary HIP builds use their independent scorer implementation.
bool custom_bf16_sparse_supported_on_current_device();
bool local_pflash_supported_on_current_device();

// Algorithmic parameters for the FlashPrefill selection + sparse forward.
struct FlashPrefillConfig {
    int   block_size       = 128;   // K stride; query block size = K block size
    int   attention_sink   = 2;     // first N k-blocks always selected
    int   window           = 4;     // last `window` k-blocks before query
    int   last_n_full      = 2;     // last N q-blocks attend to all selected blocks
    float alpha            = 0.12f; // dynamic top-K threshold (score >= max_score * alpha)
};

// Runs the full FP forward (mean_K → block_score → block_select → sparse_fwd).
// Returns 0 on success, non-zero on failure (allocator OOM, bad shape, etc.).
// Output O is written in place.
//
// Scratch memory (allocated/freed per call inside): ~M*M*H*4 * 3 + M*H*4
// where M = ceil(seq_len/block_size). At S=140K, M≈1093, H=16: ~300 MB.
//
// Two implementations:
//   flash_prefill_forward_bf16 — BF16 WMMA (sm_80+, __nv_bfloat16 m16n16k16)
//   flash_prefill_forward_f16  — F16 WMMA (sm_70+, half m16n8k16, Volta/Turing)
// Both share the same scratch allocation and block_select logic.
int flash_prefill_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    const FlashPrefillConfig & cfg);

// Same as flash_prefill_forward_bf16 but operates on F16 (half) tensors.
// Uses F16 WMMA (m16n8k16) and cooperative shared-memory loads.
// Compiled when the Volta/Turing WMMA or Pascal scalar F16 path is enabled.
#if defined(DFLASH27B_HAVE_VOLTA_FLASHPREFILL) || defined(DFLASH27B_HAVE_PASCAL_FLASHPREFILL)
int flash_prefill_forward_f16(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    const FlashPrefillConfig & cfg);
#endif

// ggml flash_attn_ext-based implementation for CUDA/HIP builds supported by
// the selected ggml backend and GPU architecture.
// Same interface as flash_prefill_forward_bf16 but uses ggml's FA internally
// (chunked causal attention). Accepts BF16/F16/F32 Q/K/V tensors stored in the
// same [B, S, H, D] contiguous layout. The caller must pass the real ggml type;
// F16 and BF16 are both 2-byte values but are not bit-compatible.
//
// Builds without CUDA BF16 WMMA support use this as the available FlashPrefill
// path. CUDA builds with the custom WMMA kernels may prefer
// flash_prefill_forward_bf16 for block-sparse selection.
int flash_prefill_forward_q8(
    ggml_backend_t backend,
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    ggml_type qkv_type,
    const FlashPrefillConfig & cfg);

// ── Unified dispatch ──────────────────────────────────────────────────────────
// Picks the best available kernel at compile time + runtime buffer type:
//   BF16 buffers + qualified sm_80 build → flash_prefill_forward_bf16
//   BF16 buffers + unqualified device   → flash_prefill_forward_q8 (exact FA)
//   F16 buffers  + Volta build → flash_prefill_forward_f16
//   otherwise                  → flash_prefill_forward_q8 (ggml FA fallback)
//
// Callers no longer need to duplicate the ifdef/dispatch boilerplate.
inline int flash_prefill_forward(
    ggml_backend_t backend,
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    ggml_type qkv_type,
    const FlashPrefillConfig & cfg)
{
#if defined(DFLASH27B_HAVE_FLASHPREFILL) || defined(DFLASH27B_HAVE_SM80_FLASHPREFILL)
    if (qkv_type == GGML_TYPE_BF16) {
        if (!custom_bf16_sparse_supported_on_current_device()) {
            // Unqualified device (e.g. GB10 sm_121): run the exact ggml FA
            // path instead of failing the scorer.
            return flash_prefill_forward_q8(backend, Q, K, V, O,
                batch, seq_len, n_q_heads, n_k_heads, head_dim, scale,
                qkv_type, cfg);
        }
        return flash_prefill_forward_bf16(Q, K, V, O,
            batch, seq_len, n_q_heads, n_k_heads, head_dim, scale, cfg);
    }
#endif
#if defined(DFLASH27B_HAVE_VOLTA_FLASHPREFILL) || defined(DFLASH27B_HAVE_PASCAL_FLASHPREFILL)
    if (qkv_type == GGML_TYPE_F16) {
        return flash_prefill_forward_f16(Q, K, V, O,
            batch, seq_len, n_q_heads, n_k_heads, head_dim, scale, cfg);
    }
#endif
    return flash_prefill_forward_q8(backend, Q, K, V, O,
        batch, seq_len, n_q_heads, n_k_heads, head_dim, scale, qkv_type, cfg);
}

#ifdef DFLASH27B_HAVE_BSA
// Free BSA persistent device buffers (blockmask, head_mask_type, softmax_lse).
// Safe to call any time; idempotent. Useful before unloading the drafter to
// give the daemon's target gen path the full VRAM headroom.
extern "C" void dflash_bsa_free_persistent();
#endif

} // namespace flashprefill
} // namespace dflash::common
