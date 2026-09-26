#pragma once
#include "common.cuh"
// Quantized-weight BF16 WMMA GEMM on gfx1151, from 512 tokens up.
bool ggml_cuda_mmb_supported_mm  (const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mmb_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);
void ggml_cuda_mul_mat_mmb   (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
void ggml_cuda_mmb_begin_graph();
// producers that can emit a BF16 copy of an F32 output register it here; returns the BF16 buffer to fill (n elements)
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n);
// BF16 copy of tensor t if one is cached for the current graph (consumers may read it instead of the F32 data)
const uint16_t * ggml_cuda_mmb_cache_lookup(const ggml_tensor * t);
// As cache_lookup, but also resolves a view/reshape of a marked bf16-only tensor
// to its in-place bf16 storage (marks are keyed on the xn view, consumers often
// reshape it). Returns nullptr when t is not backed by a bf16-only tensor.
const uint16_t * ggml_cuda_mmb_bf16_src(const ggml_tensor * t);
// bf16 weight shadow as a raw pointer, or nullptr if none
const void * ggml_cuda_mmb_shadow_ptr(const ggml_tensor * w);
// producer slots (pinned until the next producer of the same kind): 0 = HC normalized stream xn, 1 = HC gate
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n);
void ggml_cuda_mmb_marks_clear();
size_t ggml_cuda_mmb_marks_count();
void ggml_cuda_mmb_mark_bf16_only(const ggml_tensor * t);
bool ggml_cuda_mmb_is_bf16_only(const ggml_tensor * t);
bool ggml_cuda_mmb_gatemix();
bool ggml_cuda_mmb_down16();
bool ggml_cuda_mmb_res16();
bool ggml_cuda_mmb_blk16();
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst, int hc, float scale, float bias);
bool ggml_cuda_mmb_supported_glu(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu);
void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu);
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
void ggml_cuda_mmb_release_all();
