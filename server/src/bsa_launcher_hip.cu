// HIP Phase 3: BSA-compatible launcher wrapping our rocWMMA sparse FA kernel.
//
// Provides the same extern "C" symbol as bsa_launcher.cu (CUDA) so that
// flashprefill.cpp can call it when DFLASH_FP_USE_BSA=1 is set at runtime.
//
// The CUDA version converts indices/counts → BSA blockmask, then dispatches
// the CUTLASS SM80 FA-2 kernel. Here we skip the conversion entirely and
// forward directly to launch_sparse_flash_forward_bf16 (rocWMMA kernel 4).
// Q/K/V/O arrive in contiguous [B, S, H, D] layout — same layout kernel 4
// expects — so strides can be reconstructed from the shape arguments.
//
// K/V transpose pre-pass: reorders [B,S,Hk,D] → [B,Hk,S,D] so per-head
// token reads are stride-D (contiguous) rather than stride-Hk*D (interleaved).
// Persistent device buffers (kv_buf_*) grow as needed and are freed only via
// dflash_bsa_free_persistent, avoiding per-call hipMalloc overhead.

#include <hip/hip_runtime.h>
#include <cuda_runtime.h>   // hip_compat shim: cudaStream_t = hipStream_t
#include <cstdint>
#include <cstdlib>          // size_t

namespace dflash::common {
namespace flashprefill {

// Defined in flashprefill_kernels.hip.cu.
extern "C" int launch_transpose_kv_bf16(
    const void * src, void * dst,
    int B, int S, int H, int head_dim,
    hipStream_t stream);

extern "C" void launch_sparse_flash_forward_bf16(
    const void* Q, const void* K, const void* V, void* O,
    const int32_t* block_index, const int32_t* counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int q_tile, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    hipStream_t stream);

// Persistent K/V transpose buffers: allocated once, grown as needed.
// Freed only when the caller explicitly calls dflash_bsa_free_persistent().
static void   *kv_buf_K   = nullptr;
static void   *kv_buf_V   = nullptr;
static size_t  kv_buf_cap = 0;

extern "C" void dflash_bsa_free_persistent() {
    if (kv_buf_K) { hipFree(kv_buf_K); kv_buf_K = nullptr; }
    if (kv_buf_V) { hipFree(kv_buf_V); kv_buf_V = nullptr; }
    kv_buf_cap = 0;
}

extern "C" int launch_bsa_sparse_flash_forward_bf16(
    const void* Q, const void* K, const void* V, void* O,
    const int32_t* indices, const int32_t* counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream)
{
    const int B = batch, S = seq_len, H = n_q_heads, Hk = n_k_heads, D = head_dim;

    // Q strides: [B, S, H, D] layout (unchanged).
    const int s_Q_b = S * H  * D, s_Q_n = H  * D, s_Q_h = D, s_Q_d = 1;

    // Ensure persistent transpose buffers are large enough.
    const size_t kv_bytes = (size_t)B * Hk * S * D * 2;  // sizeof(bfloat16)=2
    if (kv_bytes > kv_buf_cap) {
        if (kv_buf_K) { hipFree(kv_buf_K); kv_buf_K = nullptr; }
        if (kv_buf_V) { hipFree(kv_buf_V); kv_buf_V = nullptr; }
        hipError_t err_k = hipMalloc(&kv_buf_K, kv_bytes);
        hipError_t err_v = hipMalloc(&kv_buf_V, kv_bytes);
        if (err_k != hipSuccess || err_v != hipSuccess) {
            // Roll back: free any partial allocation and reset state.
            if (kv_buf_K) { hipFree(kv_buf_K); kv_buf_K = nullptr; }
            if (kv_buf_V) { hipFree(kv_buf_V); kv_buf_V = nullptr; }
            kv_buf_cap = 0;
            return -1;
        }
        kv_buf_cap = kv_bytes;
    }

    if (launch_transpose_kv_bf16(K, kv_buf_K, B, S, Hk, D, stream) != 0) return -1;
    if (launch_transpose_kv_bf16(V, kv_buf_V, B, S, Hk, D, stream) != 0) return -1;

    // Strides for transposed [B, Hk, S, D] layout.
    const int s_Kt_b = Hk * S * D, s_Kt_h = S * D, s_Kt_n = D, s_Kt_d = 1;

    launch_sparse_flash_forward_bf16(
        Q, kv_buf_K, kv_buf_V, O, indices, counts, scale,
        B, H, Hk, S, D,
        /*q_tile=*/64, block_size,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_Kt_b, s_Kt_n, s_Kt_h, s_Kt_d,
        s_Kt_b, s_Kt_n, s_Kt_h, s_Kt_d,   // V: same transposed layout
        s_Q_b,  s_Q_n,  s_Q_h,  s_Q_d,    // O: same layout as Q
        s_idx_b, s_idx_m, s_idx_n, s_idx_h,
        s_cnt_b, s_cnt_m, s_cnt_h,
        stream);

    return 0;
}

}  // namespace flashprefill
}  // namespace dflash::common
