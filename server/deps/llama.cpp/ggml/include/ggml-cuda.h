#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// Maximum token width handled by the registry-aware DS4 mixed-weight MMV
// kernels. The kernel maps tokens to grid.z and is validated for q=5;
// wider batches remain on the MMQ path.
#define GGML_CUDA_DS4_MIX_MMV_MAX_TOKENS 5

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// Configure streams lazily created by this backend context at the device's
// lowest scheduling priority. Must be called before the backend first submits
// work. Other backend contexts on the same device are unaffected.
GGML_BACKEND_API bool ggml_backend_cuda_set_low_priority_stream(
    ggml_backend_t backend);

// Skip the expensive per-node CUDA/HIP graph property comparison on the
// calling thread once a stable graph has already been captured.  Callers must
// bracket only immutable-topology graphs whose tensor addresses and shapes do
// not change; input contents may still be updated in place.
GGML_BACKEND_API bool ggml_backend_cuda_set_skip_props_check(bool skip);

// Retire CUDA/HIP graph-cache entries whose graph key points into a metadata
// arena that is about to be rebuilt or released. The backend is synchronized
// before native graph executables are destroyed. Returns the number of erased
// entries. Non-CUDA/HIP backends and empty ranges return zero.
GGML_BACKEND_API size_t ggml_backend_cuda_graph_invalidate_range(
        ggml_backend_t backend,
        const void *   begin,
        size_t         size);

// Disable CUDA/HIP graph capture and replay on the calling thread. Returns the
// previous value so scoped callers can restore nested overrides correctly.
GGML_BACKEND_API bool ggml_backend_cuda_set_graphs_disabled_override(bool disabled);

// Number of launches of the dense F32 dim-0 concat-transpose specialization.
// Intended for focused correctness tests of the dispatch guard.
GGML_BACKEND_API size_t ggml_backend_cuda_get_concat_transpose_f32_count(void);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// Override the plain quantized MUL_MAT MMVQ column ceiling on the calling
// thread. Pass zero to restore LUCE_MMVQ_MAX_NCOLS. This is intentionally
// thread-local so one graph builder can select a safe topology without
// changing concurrent requests or other CUDA/HIP backends.
GGML_BACKEND_API int ggml_backend_cuda_set_mmvq_max_ncols_override(int max_ncols);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// [TAG_TOPK_ROWS] top-k (k <= 8) entries + softmax probabilities per row of a
// device-resident contiguous F32 [ncols, nrows] tensor. probs_out and ids_out
// must each hold k * nrows elements (row-major: entry [r*k + j] = rank-j of
// row r). Not a graph op: call only after the SYNCHRONOUS
// ggml_backend_graph_compute() producing `logits` has returned.
GGML_BACKEND_API bool ggml_backend_cuda_topk_rows(const struct ggml_tensor * logits, int k,
                                                  float * probs_out, int32_t * ids_out);

// Attach learned per-expert decode tables to a mixed-precision tensor. The
// host variants copy the tables to the device that owns `base`. Call the
// matching unregister function before releasing the tensor's backing buffer.
// Returns false without registering when validation or device setup fails.
GGML_BACKEND_API bool ggml_cuda_rocmfp3_mix_register_host(
        const void * base, size_t expert_stride, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host);
GGML_BACKEND_API bool ggml_cuda_rocmfp2_mix_register_host(
        const void * base, size_t expert_stride, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host);
GGML_BACKEND_API void ggml_cuda_rocmfp2_mix_unregister(const void * base);
GGML_BACKEND_API void ggml_cuda_rocmfp3_mix_unregister(const void * base);

#ifdef  __cplusplus
}
#endif
