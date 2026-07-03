#pragma once
#include "ggml.h"
#include <cstdint>
#include <string>

namespace dflash {

// Parses a KV-cache element type string (case-insensitive).
// Accepted: "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "tq3_0".
// Returns GGML_TYPE_COUNT on unknown input (caller should treat as error).
ggml_type parse_kv_type(const char * s);

// Returns the canonical lowercase string for a supported KV ggml_type,
// or "?" for unsupported.
const char * kv_type_name(ggml_type t);

// True iff the (K, V) ggml_type pair is supported by the CUDA flash-attention
// kernels currently compiled in (mirror of fattn.cu type-pair table when
// GGML_CUDA_FA_ALL_QUANTS=ON, which is now forced ON in dflash/CMakeLists.txt).
bool is_supported_kv_pair(ggml_type k, ggml_type v);

// Aborts with the supported-pairs listing when (k, v) is not supported by
// the compiled fattn kernels. `who` is the log prefix (e.g. "[laguna]").
void validate_kv_pair_or_abort(ggml_type k, ggml_type v, const char * who);

// Resolves K and V types from environment variables.
// Precedence (high -> low):
//   1. DFLASH27B_KV_K=<type> / DFLASH27B_KV_V=<type>  (independent override)
//   2. DFLASH27B_KV_F16 / _KV_Q4 / _KV_TQ3            (legacy shorthand, K==V)
//   3. Default: GGML_TYPE_Q4_0 for both (with FWHT K-rotation)
// On invalid input or unsupported (K,V) pair, prints an explanatory message
// and calls std::abort(). Returns the resolved pair via out params.
void resolve_kv_types(ggml_type & k_out, ggml_type & v_out);

// KV reservation bytes per token for a hybrid attention/SSM model. Single source
// of truth for both qwen35 (dense) and qwen35moe expert-placement budgeting.
// Only the full-attention layers carry a KV cache — the rest are O(1)-state
// SSM/DeltaNet — so count n_full = n_layer / full_attention_interval, and honor
// the resolved cache element type (q4_0 ≪ f16). Using n_layer or a hardcoded f16
// over-reserves KV and falsely forces experts cold (the qwen35moe:2595 bug).
inline uint64_t kv_reservation_bytes_per_token(
        int n_layer, int full_attention_interval, int n_head_kv,
        ggml_type kv_k, int n_embd_head_k,
        ggml_type kv_v, int n_embd_head_v) {
    const int n_full = (full_attention_interval > 0)
        ? (n_layer / full_attention_interval) : n_layer;
    return (uint64_t)n_full * (uint64_t)n_head_kv *
        (uint64_t)(ggml_row_size(kv_k, n_embd_head_k) +
                   ggml_row_size(kv_v, n_embd_head_v));
}

}  // namespace dflash
