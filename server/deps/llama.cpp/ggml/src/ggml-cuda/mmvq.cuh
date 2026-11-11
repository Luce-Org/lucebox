#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
// Max. batch size for MUL_MAT_ID via the dedicated multi-token MoE kernel
// (mul_mat_vec_q_moe takes ncols_dst at runtime). Staying on this path also
// keeps CUDA-graph capture enabled ([TAG_MUL_MAT_ID_CUDA_GRAPHS]); measured
// on sm_86 the fallback costs ~9ms/step for 9-16 token spec-verify batches.
#define MMVQ_MAX_MOE_BATCH_SIZE 16
// Max. columns of a batch-invariant dense product (ggml_backend_cuda_set_mmvq_invariant_max_ncols).
#define MMVQ_INVARIANT_MAX_BATCH_SIZE 16

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
// Batch-invariant mode also admits tokenwise MMID up to MMVQ_MAX_BATCH_SIZE.
// A zero ceiling (unsupported type) is never raised.
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

bool ggml_cuda_mmvq_mmid_grouped_enabled(
    ggml_type type, int cc, int64_t ncols_dst, int64_t routed_pairs);

// Set by ggml_backend_cuda_set_mmvq_batch_invariant (calling thread).
bool ggml_cuda_mmvq_batch_invariant();
// Columns up to which batch-invariant dense products stay on the matrix-vector kernels (calling thread):
// MMVQ_MAX_BATCH_SIZE unless ggml_backend_cuda_set_mmvq_invariant_max_ncols raised it.
int ggml_cuda_mmvq_invariant_max_ncols();

bool ggml_cuda_mmvq_rdna4_glu_pair(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu);
// Two Q8_0 MUL_MATs of the same activations that one RDNA4 launch computes bit for bit (fusion.twin_x).
bool ggml_cuda_mmvq_rdna4_twin(const ggml_tensor * a, const ggml_tensor * b);
// Whether the device runs that launch at all.
bool ggml_cuda_mmvq_rdna4_twin_device(int device);
// Set by ggml_backend_cuda_set_mmvq_invariant_rows_override (calling thread): 0 = env / policy.
int ggml_cuda_mmvq_invariant_rows_override();
// Set by ggml_backend_cuda_set_mmvq_moe_prefetch_rows_override (calling thread): 0 = the default (2).
int ggml_cuda_mmvq_moe_prefetch_rows_override();
int ggml_cuda_mmvq_invariant_lds_override();

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// The q8_1 rows ggml_cuda_mul_mat_vec_q reads for a contiguous F32 src1 when a producer registered them in
// ctx.luce_q8_memo (see ggml_cuda_try_hc_cn_q8), or null.
const void * ggml_cuda_mmvq_q8_memo_find(ggml_backend_cuda_context & ctx, const ggml_tensor * src1);

// Quantizes a contiguous F32 src1 to the q8_1 rows ggml_cuda_mul_mat_vec_q would make for a product with a src0 of
// that type, and registers them under its memo key, so ggml_cuda_mmvq_q8_memo_find and the later products of src1 read
// them. Null when the memo is off (then nothing is queued).
const void * ggml_cuda_mmvq_q8_memo_quantize(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_type src0_type);

// The qwen4exp hyper-connection low-rank gate and mix on RDNA4 in two launches (ggml-cuda.cu
// ggml_cuda_try_hc_mix_rdna4): the down product with SCALE -> SILU as its epilogue, then the up product whose blocks
// own whole mixed columns, with the mix reduce (and its q8_1 rows) as its epilogue. Bit for bit the four launches.
struct ggml_cuda_hc_mix_rdna4_args {
    const ggml_tensor * w_down = nullptr;  // Q8_0 [4*n_embd, lr]
    const void * xn_q8 = nullptr;          // xn's q8_1 rows, as MMVQ reads them
    float silu_scale = 1.0f;
    float silu_bias = 0.0f;
    const ggml_tensor * w_up = nullptr;    // Q8_0 [lr, 4*n_embd]
    const float * xn = nullptr;            // [4*n_embd, T]
    float * dst = nullptr;                 // the mix [n_embd, T]
    void * dst_q8 = nullptr;               // its q8_1 rows (unpadded), or null
    int64_t n_embd = 0;
    int64_t T = 0;
    float mix_scale = 1.0f;
    float mix_bias = 0.0f;
};
// Whether the device and the two Q8_0 weights take those launches at T columns (the products' own launches would be
// the RDNA4 Q8_0 kernel's shapes 2 and 0).
bool ggml_cuda_hc_mix_rdna4_ok(int device, const ggml_tensor * w_down, const ggml_tensor * w_up, int64_t n_embd,
                               int64_t T);
void ggml_cuda_hc_mix_rdna4(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_rdna4_args & args);

// Whether MMVQ memoizes q8_1 activations within an evaluation (LUCE_Q8_MEMO).
// A producer that writes an activation's q8_1 form itself may register it in
// ctx.luce_q8_memo under the key MMVQ uses (see ggml_cuda_mul_mat_vec_q).
bool ggml_cuda_luce_q8_memo_on();
