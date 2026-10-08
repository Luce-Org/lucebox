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

// Default dispatch ceiling for the registry-aware DS4 mixed-weight MMV
// kernels. Monolithic paged serving can opt into the wider, separately tested
// grid.z token range without changing the single-request or other-device policy.
#define GGML_CUDA_DS4_MIX_MMV_MAX_TOKENS 5
#define GGML_CUDA_DS4_MIX_MMV_PAGED_MAX_TOKENS 16

// HIP registry-only opt-in DS4V BF16 linear capability (not NVIDIA/CUDA).
// Lookup "ggml_backend_hip_vision_bias_bf16_workspace" as size_t (*)(ggml_backend_t):
// nonzero means the explicit op is available, and returns its retained external
// workspace reservation (76 MiB). Unsupported op shapes must fail, not fallback.
// "ggml_backend_hip_vision_bias_bf16_launches" has the same signature and returns
// actual successful Lt submissions, with or without bias. Registry names retain
// their original spelling; both modes require a matching GGML/HIP library set.

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
// not change; input contents may still be updated in place.  Bypass requires a
// known matching non-zero graph generation UID.
GGML_BACKEND_API bool ggml_backend_cuda_set_skip_props_check(bool skip);

// Retire CUDA/HIP graph-cache entries whose graph key points into a metadata
// arena that is about to be rebuilt or released. The backend is synchronized
// before native graph executables are destroyed. Returns the number of erased
// entries. Non-CUDA/HIP backends and empty ranges return zero.
GGML_BACKEND_API size_t ggml_backend_cuda_graph_invalidate_range(
        ggml_backend_t backend,
        const void *   begin,
        size_t         size);

// Returns true when the CUDA/HIP backend has instantiated a legacy device
// pool. Meta backends recursively inspect every rank-local backend. This lets
// callers and tests distinguish a trimmable cache from a VMM arena.
GGML_BACKEND_API bool ggml_backend_cuda_has_legacy_pool(ggml_backend_t backend);

// Release cached temporary allocations held by CUDA/HIP legacy device pools.
// Meta backends recursively trim every rank-local backend. Each CUDA/HIP
// backend is synchronized first, and graph executables that may reference
// released pool blocks are retired. VMM pools are already a contiguous
// reusable arena and are left intact. Returns total bytes released.
GGML_BACKEND_API size_t ggml_backend_cuda_trim_pool(ggml_backend_t backend);

// Disable CUDA/HIP graph capture and replay on the calling thread. Returns the
// previous value so scoped callers can restore nested overrides correctly.
GGML_BACKEND_API bool ggml_backend_cuda_set_graphs_disabled_override(bool disabled);

// Number of launches of the dense F32 dim-0 concat-transpose specialization.
// Intended for focused correctness tests of the dispatch guard.
GGML_BACKEND_API size_t ggml_backend_cuda_get_concat_transpose_f32_count(void);

// Calling-thread launch counters for quantized matrix-vector (MMVQ), its
// grouped-expert MMID specialization, and matrix-matrix (MMQ) kernels.
// Intended for focused tests that must prove which dispatch path executed
// rather than only checking numerical output.
GGML_BACKEND_API size_t ggml_backend_cuda_get_mmvq_launch_count(void);
GGML_BACKEND_API size_t ggml_backend_cuda_get_mmq_launch_count(void);
GGML_BACKEND_API size_t ggml_backend_cuda_get_mla_stream_topk_launch_count(void);
GGML_BACKEND_API size_t ggml_backend_cuda_get_mmvq_mmid_grouped_launch_count(void);

// Calling-thread launch counters for the scalar and grouped-column GDN
// kernels. Focused qualification tests use these to reject silent fallback.
GGML_BACKEND_API size_t ggml_backend_cuda_get_gdn_scalar_launch_count(void);
GGML_BACKEND_API size_t ggml_backend_cuda_get_gdn_grouped_cols_launch_count(void);
GGML_BACKEND_API bool ggml_backend_cuda_supports_gdn_grouped_cols(int device);

// Calling-thread launch counter for the head-size-256 MMA fattn kernel.
// Qualification tests use this to reject silent fallback to the tile kernel.
GGML_BACKEND_API size_t ggml_backend_cuda_get_fattn_mma256_launch_count(void);

// Calling-thread launch counter for the head-size-256 rocWMMA fattn kernel.
GGML_BACKEND_API size_t ggml_backend_cuda_get_fattn_wmma256_launch_count(void);

// Calling-thread launch counter for the head-size-256 WMMA paged-attention
// kernel (stage-1; gated by LUCE_PAGED_WMMA).
GGML_BACKEND_API size_t ggml_backend_cuda_get_paged_attn_wmma256_launch_count(void);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// Peer copy on the source device's side stream: the source's compute stream
// keeps running while the copy is in flight; the destination waits for it.
// Returns false when it does not apply (the caller copies the usual way).
GGML_BACKEND_API bool ggml_backend_cuda_copy_tensor_async_side(
        ggml_backend_t backend_src, ggml_backend_t backend_dst,
        const struct ggml_tensor * src, struct ggml_tensor * dst);
// Order the backend's compute stream after its side-stream copies so far.
GGML_BACKEND_API void ggml_backend_cuda_join_side_copies(ggml_backend_t backend);
// Peer copy on the source's compute stream without a destination wait:
// record an event on the source after it and wait on it where the data is
// needed. Returns false when it does not apply.
GGML_BACKEND_API bool ggml_backend_cuda_copy_tensor_async_nowait(
        ggml_backend_t backend_src, ggml_backend_t backend_dst,
        const struct ggml_tensor * src, struct ggml_tensor * dst);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// Current compute stream of a CUDA/HIP backend (cudaStream_t / hipStream_t
// returned as an opaque pointer), so work submitted outside ggml can be
// enqueued stream-ordered with the kernels of the same backend instead of
// synchronizing the host. The stream is created lazily on first use.
// Returns NULL for non-CUDA/HIP backends.
GGML_BACKEND_API void * ggml_backend_cuda_get_stream(ggml_backend_t backend);

// Device ordinal the backend was created for, or -1 for non-CUDA/HIP backends.
GGML_BACKEND_API int ggml_backend_cuda_get_device_id(ggml_backend_t backend);

// Override the plain quantized MUL_MAT MMVQ column ceiling on the calling
// thread. Pass zero to restore LUCE_MMVQ_MAX_NCOLS. This is intentionally
// thread-local so one graph builder can select a safe topology without
// changing concurrent requests or other CUDA/HIP backends.
GGML_BACKEND_API int ggml_backend_cuda_set_mmvq_max_ncols_override(int max_ncols);

// Calling-thread switch for batch-invariant quantized MUL_MAT: while set,
// products of up to eight columns stay on the matrix-vector path with the
// single-column block shape (Q8_0 reads each weight row once for all
// columns, other types run the single-column kernel per column), so each
// output column is bit-identical to the same product of that column alone
// and a speculative verifier reproduces single-token decode exactly.
// Returns the previous setting.
GGML_BACKEND_API bool ggml_backend_cuda_set_mmvq_batch_invariant(bool enabled);

// A MUL_MAT node whose src0 is a column range of wider Q8_0 rows (a view with
// the rows' stride, e.g. one rank's slice of a contraction): on the RDNA4 Q8_0
// matvec every block keeps the lane it takes in the whole rows, so the product
// is bit-identical to the whole rows times src1 zero-padded outside the range,
// at the range's reads. Other kernels compute the range's product as usual.
GGML_BACKEND_API void ggml_backend_cuda_mul_mat_whole_row_lanes(struct ggml_tensor * mul_mat);

// Calling-thread DS4 mixed-expert dispatch ceiling, scoped to a graph compute.
// Accepts 0 (the default of five) or 1..16; returns the previous ceiling.
GGML_BACKEND_API int ggml_backend_cuda_set_ds4_mix_mmv_max_tokens_override(int max_tokens);

// One stream-ordered copy per descriptor, all in one kernel launch per 48
// descriptors instead of one blit per copy. src and dst are device pointers
// (or host memory the device can address) on the backend's device; ranges of
// different descriptors must not overlap. Asynchronous on the backend stream.
struct ggml_cuda_copy_desc {
    const void * src;
    void       * dst;
    size_t       nbytes;
};
GGML_BACKEND_API void ggml_backend_cuda_copy_batch_async(ggml_backend_t backend,
                                                         const struct ggml_cuda_copy_desc * descs,
                                                         int n);
// Graph copy runs issued as one batched launch on the calling thread (see
// GGML_CUDA_DISABLE_COPY_BATCH); for tests and profiling.
GGML_BACKEND_API size_t ggml_backend_cuda_get_copy_batch_run_count(void);
// Keeps the backend's GPU out of its idle power state: on starts one wave that
// sleeps in a loop on a stream of its own (at most max_ms), off releases it.
// For a process that leaves the GPU idle for milliseconds between launches and
// pays the wake-up (about 1 ms on RDNA4) at the next one. HIP only.
GGML_BACKEND_API void ggml_backend_cuda_keepalive(ggml_backend_t backend, bool on, int max_ms);

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

// Batched concurrent-tree commit. Validation is fail-closed before any kernel
// launches; all layer replay logs and convolution windows commit on one device
// synchronization.
GGML_BACKEND_API bool ggml_backend_cuda_gdn_replay_log_commit_many(
        const struct ggml_tensor * const * replay_logs,
        struct ggml_tensor * const * states,
        const struct ggml_tensor * const * conv_inputs,
        struct ggml_tensor * const * conv_states,
        int n_layers,
        const struct ggml_tensor * accepted_prefixes,
        const struct ggml_tensor * active_slot_ids);

// Promote accepted packed-tree K/V scratch rows into pager-owned rows.
GGML_BACKEND_API bool ggml_backend_cuda_tree_cache_commit_many(
        struct ggml_tensor * const * caches, int n_caches,
        const struct ggml_tensor * commit_rows,
        const struct ggml_tensor * active_slot_ids,
        int tree_scratch_base, int tree_scratch_stride);

// Promote accepted BF16 tree feature rows into slot-local feature rings.
GGML_BACKEND_API bool ggml_backend_cuda_tree_feature_commit(
        const struct ggml_tensor * source, struct ggml_tensor * destination,
        const struct ggml_tensor * destination_rows);

// Validate every packed-tree destination before changing any cache. Once the
// first kernel launches, a device failure is fatal because fallback cannot
// recover from a partially committed state.
GGML_BACKEND_API bool ggml_backend_cuda_tree_commit_transaction(
        struct ggml_tensor * const * caches,
        int n_caches,
        const struct ggml_tensor * feature_source,
        struct ggml_tensor * feature_destination,
        const struct ggml_tensor * feature_destination_rows,
        const struct ggml_tensor * const * replay_logs,
        struct ggml_tensor * const * states,
        const struct ggml_tensor * const * conv_inputs,
        struct ggml_tensor * const * conv_states,
        int n_layers,
        const struct ggml_tensor * commit_rows,
        const struct ggml_tensor * accepted_prefixes,
        const struct ggml_tensor * active_slot_ids,
        int tree_scratch_base,
        int tree_scratch_stride);

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

// Calling-thread profile. Scoped qwen4exp callers restore the returned value;
// other models keep the generic dispatcher. DEFAULT requires a supported backend
// (checked once at load).
enum ggml_cuda_qwen4exp_profile {
    GGML_CUDA_QWEN4EXP_OFF,
    GGML_CUDA_QWEN4EXP_DEFAULT,
};
GGML_BACKEND_API enum ggml_cuda_qwen4exp_profile ggml_backend_cuda_set_qwen4exp_profile(enum ggml_cuda_qwen4exp_profile profile);
GGML_BACKEND_API bool ggml_backend_cuda_qwen4exp_supported(ggml_backend_t backend);

// True when a matmul with weight w over n_tokens rows can take an F16 activation (HIP MMB Q8_0 -> F16 route).
GGML_BACKEND_API bool ggml_backend_cuda_mmb_f16_input_ok(const struct ggml_tensor * w, int64_t n_tokens);
// True when MMB serves this qwen4exp prefill batch on gfx1151.
GGML_BACKEND_API bool ggml_backend_cuda_mmb_prefill(int64_t n_tokens);

// Integrated GPUs only. While on, a buffer allocation on `device` that would
// leave less than `carve_reserve` bytes free in the carve goes to locked host
// memory instead, up to `host_bytes` in total; the GPU reads it from the same
// DRAM. Returns the host bytes granted (capped by RLIMIT_MEMLOCK), 0 when off.
// Pass 0 to turn it off; buffers already placed stay where they are.
GGML_BACKEND_API size_t ggml_backend_cuda_set_host_spill(int device, size_t carve_reserve, size_t host_bytes);
// Bytes of live `device` buffers in locked host memory.
GGML_BACKEND_API size_t ggml_backend_cuda_host_spill_bytes(int device);

#ifdef  __cplusplus
}
#endif
