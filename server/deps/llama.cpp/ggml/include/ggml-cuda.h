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

// Tests only: turn the graph-node fusions off (process-wide), to compare the
// fused kernels with the unfused ops. Returns the previous value.
GGML_BACKEND_API bool ggml_backend_cuda_set_fusion_disabled_override(bool disabled);
// Whether the graph-node fusions run: true unless a test turned them off.
GGML_BACKEND_API bool ggml_backend_cuda_fusion_enabled(void);

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
// Calling-thread count of MMQ calls that skipped their src1 quantize because
// the Q8 memo served an earlier call's q8_1.
GGML_BACKEND_API size_t ggml_backend_cuda_get_glue_q8_memo_hit_count(void);

// Tests only: turn the Q8 memo on or off, to compare its outputs bit for bit
// with per-call quantization. On by default.
GGML_BACKEND_API void ggml_backend_cuda_set_q8_memo_override(bool on);

// Row stacking (the Qwen3.5 gate|up stack): nullptr when a MUL_MAT of `weight`
// (two weights stacked by rows, the first `rows_first` rows one member) with
// `ncols` F32 columns computes every output row bit-identically to a MUL_MAT
// of its member alone on this backend's device, else the reason it may not.
GGML_BACKEND_API const char * ggml_backend_cuda_row_stack_inexact(
        ggml_backend_t backend, const struct ggml_tensor * weight, int64_t rows_first, int64_t ncols);

// Calling-thread count of MMQ calls served by an RDNA4 tile twin with
// `tile_rows` rows (64 or 32, mmq_tile_twins.h);
// 0 for other values.
// Lets the twin parity test reject a silent fallback to the parent tile.
GGML_BACKEND_API size_t ggml_backend_cuda_get_mmq_twin_launch_count(int tile_rows);

// Tests only: turn the RDNA4 MMQ tile twins (64-row, 32-row) on or off, to
// compare them bit for bit with the parent tiles. Both are on by default.
GGML_BACKEND_API void ggml_backend_cuda_set_mmq_twins_override(bool rows64, bool rows32);

// While on (for the calling thread), the RDNA4 64-row MMQ twins split k for
// calls of 17-48 columns whose grid would leave the GPU short of workgroups,
// at 16 or more k-iterations per split on average. A split sum is not
// bit-exact against the unsplit one, so only a caller that already accepts
// batch-dependent results turns it on: the concurrent (C>1) batched verify,
// around its graph. Off by default; single-request paths never set it.
GGML_BACKEND_API void ggml_backend_cuda_set_mmq_splitk_scope(bool on);

// Calling-thread launch counters for the scalar and grouped-column GDN
// kernels. Focused qualification tests use these to reject silent fallback.
GGML_BACKEND_API size_t ggml_backend_cuda_get_gdn_scalar_launch_count(void);
GGML_BACKEND_API size_t ggml_backend_cuda_get_gdn_grouped_cols_launch_count(void);
GGML_BACKEND_API bool ggml_backend_cuda_supports_gdn_grouped_cols(int device);
// Tests only: pin the GATED_DELTA_NET route, process-wide. -1 (the default)
// lets the device pick, 0 pins the scalar kernel, 1 the grouped-cols kernel
// where it applies (S_v 128, non-KDA, a supported device); a pinned route also
// keeps the RDNA3.5 prefill recurrence off. Returns the previous value.
GGML_BACKEND_API int ggml_backend_cuda_set_gdn_route_override(int route);

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
// The runtime's figures (cudaMemGetInfo); on an integrated GPU (Linux) at most what its
// allocations can take, its own VRAM left plus the RAM available now
// (ggml-cuda/uma-budget.h), since the runtime's free need not follow what is
// allocated there. LUCE_HIP_UMA_NO_RAM_CHECK set leaves them as they are.
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
// The same, keeping out of the RAM the reserve of the managed-buffer check (1/16 of
// RAM, 1 to 8 GiB). True when that bound applied; discrete GPUs, an unknown RAM and
// LUCE_HIP_UMA_NO_RAM_CHECK set get the runtime's figures and false.
GGML_BACKEND_API bool ggml_backend_cuda_get_device_room(int device, size_t * free, size_t * total);
// What an integrated GPU's allocations can take with those that spill to (locked) host
// memory: its own VRAM left plus the RAM a managed buffer could take now, whatever its
// runtime reports, which on an APU with a large carve may be the carve alone. False,
// `room` untouched, where ggml_backend_cuda_get_device_room returns false.
GGML_BACKEND_API bool ggml_backend_cuda_get_uma_room(int device, size_t * room);

// Current compute stream of a CUDA/HIP backend (cudaStream_t / hipStream_t
// returned as an opaque pointer), so work submitted outside ggml can be
// enqueued stream-ordered with the kernels of the same backend instead of
// synchronizing the host. The stream is created lazily on first use. A lookup
// only: it settles nothing the backend still holds back (see
// ggml_backend_cuda_enter_stream). Returns NULL for non-CUDA/HIP backends.
GGML_BACKEND_API void * ggml_backend_cuda_get_stream(ggml_backend_t backend);

// The same stream, for work about to be queued on it directly: it first
// settles what the backend's own launches settle (collected peer copies, an
// open graph tail, an idle hold, pending side copies) and makes the backend's
// device current. Returns NULL for non-CUDA/HIP backends.
GGML_BACKEND_API void * ggml_backend_cuda_enter_stream(ggml_backend_t backend);

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

// Rows of one Q8_0 matrix split between an RDNA4 and an RDNA3.5 GPU. A Q8_0 MUL_MAT node flagged here runs the RDNA4
// Q8_0 matvec on RDNA3.5 too (wave32 on both: the same partials, summed in the same order), so each output row has
// the bits the RDNA4 device gives it, at one column or up to eight batch-invariant ones. RDNA4 runs that kernel for
// every such product; other devices and products ignore the flag.
GGML_BACKEND_API void ggml_backend_cuda_mul_mat_rdna4_rows(struct ggml_tensor * mul_mat);
// Whether `device` gives a flagged Q8_0 product the RDNA4 matvec's bits.
GGML_BACKEND_API bool ggml_backend_cuda_rdna4_rows_device(int device);

// A MUL_MAT node whose src0 is a column range of wider Q8_0 rows (a view with
// the rows' stride, e.g. one rank's slice of a contraction): on the RDNA4 Q8_0
// matvec every block keeps the lane it takes in the whole rows, so the product
// is bit-identical to the whole rows times src1 zero-padded outside the range,
// at the range's reads. Other kernels compute the range's product as usual.
GGML_BACKEND_API void ggml_backend_cuda_mul_mat_whole_row_lanes(struct ggml_tensor * mul_mat);

// A router's weights node (the DIV that ends the router) marked here posts wake-hold slot `slot`
// (1..GGML_MOE_FUSED_WAKE_SLOTS) once it is computed, which ends a ggml_moe_fused_wake_hold of that slot on another
// device. What the node computes does not change.
GGML_BACKEND_API void ggml_backend_cuda_div_wake_post(struct ggml_tensor * div, int slot);

// Calling-thread override of the weight rows per wave on the batch-invariant
// Q8_0 multi-column launch (mmvq.cu c_invariant_rpb): 0 = LUCE_MMVQ_INVARIANT_ROWS
// / the default policy, 1 / 2 / 4 = forced. Each row keeps its single-column
// K traversal, per-lane accumulation and warp reduction; a wave only shares
// the activation loads across its rows. Returns the previous setting.
GGML_BACKEND_API int ggml_backend_cuda_set_mmvq_invariant_rows_override(int rows);

// Calling-thread override of the weight rows per wave in the gfx1151 Q4_K MoE prefetching loop (mmvq.cu
// mul_mat_vec_q_moe_q4k_prefetch): 0 = the default (2), 1 / 2 / 4 = forced. Every count gives the same bits; tests
// compare them. Returns the previous setting.
GGML_BACKEND_API int ggml_backend_cuda_set_mmvq_moe_prefetch_rows_override(int rows);

// Calling-thread override of the LDS activation staging on the same launch
// (mmvq.cu c_invariant_lds_waves): -1 = LUCE_MMVQ_INVARIANT_LDS / the default
// policy, 0 = off (every wave streams the T activation columns itself), 4 / 8 /
// 16 = that many waves per block stage the columns' q8_1 blocks in LDS once
// and dot their own rows against them. Same per-row chain. Returns the previous.
GGML_BACKEND_API int ggml_backend_cuda_set_mmvq_invariant_lds_override(int waves);

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
// The same copies on the backend's side stream, after the work queued so far on its compute stream, on a few
// blocks so kernels queued meanwhile keep most of the GPU. The compute stream waits for them at its next use (a
// graph, an async tensor copy, an event) unless ggml_backend_cuda_set_side_overlap is on; ggml_backend_synchronize
// waits for them too. Work that overlaps them must not touch their sources or destinations.
GGML_BACKEND_API void ggml_backend_cuda_copy_batch_side(ggml_backend_t backend,
                                                        const struct ggml_cuda_copy_desc * descs,
                                                        int n);
// While on, the backend's graphs and async tensor copies run beside its pending side copy batches.
GGML_BACKEND_API void ggml_backend_cuda_set_side_overlap(ggml_backend_t backend, bool on);
// Keeps an idle device's clocks up (HIP; a no-op elsewhere): one wave spins on the backend stream until the next use
// of the backend or for at most max_us, so the work after an idle stretch starts at full clock.
GGML_BACKEND_API void ggml_backend_cuda_hold(ggml_backend_t backend, int max_us);
// While on, the peer copies of a hop this backend produces can run as the tail of its captured graph (HIP graphs; a
// no-op elsewhere): a graph evaluated right before a flush of batched kernel copies (GGML_CUDA_PEER_COPY_KERNEL with
// batched peer copies) is captured again with the copies and the hop's event record as its last nodes, and later
// replays carry each hop's copies. Off by default; a graph that already has a tail keeps it.
GGML_BACKEND_API void ggml_backend_cuda_set_copy_tail(ggml_backend_t backend, bool on);
// Graph copy runs issued as one batched launch on the calling thread (see
// GGML_CUDA_DISABLE_COPY_BATCH); for tests and profiling.
GGML_BACKEND_API size_t ggml_backend_cuda_get_copy_batch_run_count(void);
// Hops whose peer copies ran as the tail of the producer's captured graph (HIP; 0 elsewhere), process-wide; for
// tests.
GGML_BACKEND_API size_t ggml_backend_cuda_get_copy_tail_count(void);

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

// ggml_backend_cuda_tree_commit_transaction validated from the caller's host
// copies of commit_rows, accepted_prefixes, active_slot_ids and
// feature_destination_rows (the values it uploaded), with one launch per
// commit kind. Same fail-closed contract; same committed bits. The count is
// the calling thread's transactions committed this way.
GGML_BACKEND_API bool ggml_backend_cuda_tree_commit_transaction_host(
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
        int tree_scratch_stride,
        const int64_t * host_commit_rows,
        const int32_t * host_accepted,
        const int32_t * host_slots,
        const int32_t * host_feature_rows);
GGML_BACKEND_API size_t ggml_backend_cuda_get_tree_commit_host_count(void);

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

// Calling-thread counts of the Qwen3.8 split fusions taken, so their parity test (test_qn_split_fusions) can tell a
// pattern that stopped matching from one that still fuses.
enum ggml_cuda_qn_fusion {
    GGML_CUDA_QN_ROUTE_NORM,     // the router's GET_ROWS, SUM_ROWS, CLAMP and DIV in one launch
    GGML_CUDA_QN_SHARED_GATE,    // the shared expert's SIGMOID and MUL folded into DS4_MOE_COMBINE
    GGML_CUDA_QN_HC_CN_Q8,       // HC_COMBINE_NORM writing the next matmul's q8_1 rows
    GGML_CUDA_QN_HC_MIX_Q8,      // the HC mix reduce writing the next matmul's q8_1 rows
    GGML_CUDA_QN_UNARY_MUL_Q8,   // UNARY then MUL writing the next matmul's q8_1 rows
    GGML_CUDA_QN_SCALE_SILU,     // SCALE then SILU in one launch
    GGML_CUDA_QN_GDN_CONV,       // a GDN step's conv history copies, conv, SILU and q/k norm in one launch
    GGML_CUDA_QN_GATED_NORM,     // the GDN output gate: RMS_NORM, MUL, SIGMOID and MUL in one launch
    GGML_CUDA_QN_GATED_NORM_Q8,  // that launch also writing the next matmul's q8_1 rows
    GGML_CUDA_QN_RMS_NORM_SCALE, // RMS_NORM then SCALE (the GDN q/k norm) in one launch
    GGML_CUDA_QN_MMVQ_TWIN,      // two Q8_0 products of the same activations (the GDN qkv and z) in one launch
    GGML_CUDA_QN_ROUTER_TOPK,    // the router after its logits: SOFT_MAX, ARGSORT top-k, table gathers, weights
    GGML_CUDA_QN_HC_MIX_RDNA4,   // the HC down product, SCALE, SILU, up product and mix reduce in two launches
    GGML_CUDA_QN_HC_CN_ADD,      // the layer output's ADD (split hot + cold experts) folded into the next HC_COMBINE_NORM
    GGML_CUDA_QN_MMVF_TWIN,      // two F32-weight products (the GDN step's HC inject and alpha/beta gates) in one launch
    GGML_CUDA_QN_GDN_CONV_SLOTS, // the batched decode's GDN conv (history gather and write-back by slot id) in one launch
    GGML_CUDA_QN_COUNT,
};
GGML_BACKEND_API size_t ggml_backend_cuda_get_qn_fusion_count(enum ggml_cuda_qn_fusion kind);
// True when the QSA selected-attention kernels (qsa.cu) run on the backend's device: gfx1151 and RDNA4.
GGML_BACKEND_API bool ggml_backend_cuda_qsa_supported(ggml_backend_t backend);
// True when the backend's device runs GGML_CUDA_QN_MMVQ_TWIN (RDNA4).
GGML_BACKEND_API bool ggml_backend_cuda_mmvq_twin_supported(ggml_backend_t backend);

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
