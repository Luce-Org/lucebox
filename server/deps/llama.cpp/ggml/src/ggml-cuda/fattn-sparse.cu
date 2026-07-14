// fattn-sparse.cu — ggml CUDA backend dispatch for GGML_OP_FLASH_ATTN_SPARSE.
//
// When a sparse kernel is registered via ggml_cuda_flash_attn_sparse_set_kernel,
// it is called with BF16 Q/K/V in flash_prefill_forward_bf16 layout:
//   Q[B, S, n_q_heads, D]   (contiguous, D fastest in C-row-major terms)
//   K[B, S, n_k_heads, D]
//   V[B, S, n_k_heads, D]
//   O[B, S, n_q_heads, D]
//
// ggml FA convention (column-major, ne[0] fastest):
//   Q src:  ne={D, S, H,  B}  =>  C-order [B][H][S][D]
//   K src:  ne={D, S, Hk, B}  =>  C-order [B][Hk][S][D]
//   V src:  ne={D, S, Hk, B}  =>  C-order [B][Hk][S][D]
//   O dst:  ne={D, S, H,  B}  =>  C-order [B][H][S][D]
//
// pFlash expects [B, S, H, D] row-major — S and H are swapped relative to ggml.
// A S<->H transpose is performed during type conversion (F32/F16 -> BF16) for
// Q/K/V inputs.  The output needs NO transpose: pFlash O[B,S,H,D] row-major
// already matches ggml dst ne={D,H,S,B} column-major, so a flat BF16->F32 copy
// suffices.
//
// When no kernel is registered, falls back to ggml_cuda_flash_attn_ext (dense FA).

#include "fattn-sparse.cuh"
#include "fattn.cuh"
#include "common.cuh"
#include "convert.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstring>

static ggml_cuda_sparse_attn_fn_t s_sparse_kernel = nullptr;

void ggml_cuda_flash_attn_sparse_set_kernel(ggml_cuda_sparse_attn_fn_t fn) {
    s_sparse_kernel = fn;
}

// Convert F32 -> BF16 with (S,H) transpose.
// src: ggml [B,H,S,D] row-major.  dst: pFlash [B,S,H,D] row-major.
__global__ void k_f32_to_bf16_transpose_sh(
    const float * __restrict__ src, __nv_bfloat16 * __restrict__ dst,
    int B, int S, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx >= total) return;
    int d = idx % D;
    int h = (idx / D) % H;
    int s = (idx / (D * H)) % S;
    int b = idx / (D * H * S);
    int src_idx = ((b * H + h) * S + s) * D + d;
    dst[idx] = __float2bfloat16(src[src_idx]);
}

// Convert F16 -> BF16 with (S,H) transpose.
// src: ggml [B,H,S,D] row-major.  dst: pFlash [B,S,H,D] row-major.
__global__ void k_f16_to_bf16_transpose_sh(
    const half * __restrict__ src, __nv_bfloat16 * __restrict__ dst,
    int B, int S, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx >= total) return;
    int d = idx % D;
    int h = (idx / D) % H;
    int s = (idx / (D * H)) % S;
    int b = idx / (D * H * S);
    int src_idx = ((b * H + h) * S + s) * D + d;
    dst[idx] = __float2bfloat16(__half2float(src[src_idx]));
}

// Flat BF16 -> F32 conversion (no transpose).
// pFlash output [B,S,H,D] row-major matches ggml dst ne={D,H,S,B} column-major —
// no transpose needed; a flat element-wise copy is correct.
__global__ void k_bf16_to_f32_flat(
    const __nv_bfloat16 * __restrict__ src, float * __restrict__ dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __bfloat162float(src[idx]);
}

void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // K/V may be quantized (Q8_0, Q4_0, etc.).  If so, we dequantize them to F16
    // into temporary buffers before the S<->H transpose into BF16 pFlash layout.

    // When no sparse kernel is registered, fall back to dense FA.
    if (!s_sparse_kernel) {
        const enum ggml_op saved_op = dst->op;
        float saved_params[GGML_MAX_OP_PARAMS / sizeof(float)];
        memcpy(saved_params, dst->op_params, GGML_MAX_OP_PARAMS);
        ggml_tensor * saved_src3 = dst->src[3];
        ggml_tensor * saved_src4 = dst->src[4];

        float op_params[2];
        memcpy(op_params, dst->op_params, sizeof(op_params));
        const float scale = op_params[0];

        float ext_params[3] = { scale, 0.0f, 0.0f };
        dst->op = GGML_OP_FLASH_ATTN_EXT;
        memset(dst->op_params, 0, GGML_MAX_OP_PARAMS);
        memcpy(dst->op_params, ext_params, sizeof(ext_params));
        dst->src[3] = nullptr;
        dst->src[4] = nullptr;

        ggml_cuda_flash_attn_ext(ctx, dst);

        dst->op = saved_op;
        memcpy(dst->op_params, saved_params, GGML_MAX_OP_PARAMS);
        dst->src[3] = saved_src3;
        dst->src[4] = saved_src4;
        return;
    }

    // Sparse path.
    // ggml src tensors (Q is F32, K/V are F16), ggml FA convention:
    //   Q: ne={D, S, H,  B}  C-order [B][H][S][D]
    //   K: ne={D, S, Hk, B}  C-order [B][Hk][S][D]
    //   V: ne={D, S, Hk, B}  C-order [B][Hk][S][D]
    // ggml dst:
    //   O: ne={D, S, H,  B}  C-order [B][H][S][D]  F32
    //
    // pFlash expects [B,S,H,D] — S and H are transposed relative to ggml for inputs.
    // Input (Q/K/V) type conversion kernels perform the S<->H transpose.
    // The output requires no transpose (flat BF16->F32 copy suffices).
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    float op_params[2];
    memcpy(op_params, dst->op_params, sizeof(op_params));
    const float scale = op_params[0];
    const float alpha = op_params[1];

    const int D  = (int)Q->ne[0];
    const int S  = (int)Q->ne[1];  // seq_len (ggml FA convention: ne[1] = tokens)
    const int H  = (int)Q->ne[2];  // n_q_heads (ggml FA convention: ne[2] = heads)
    const int Hk = (int)K->ne[2];  // n_kv_heads
    const int B  = (int)(Q->ne[3] > 0 ? Q->ne[3] : 1);

    const int Q_n = B * H  * S * D;
    const int K_n = B * Hk * S * D;
    const int O_n = Q_n;

    cudaStream_t stream = ctx.stream();
    const int block = 256;

    // Allocate pFlash-layout BF16 buffers and convert with S<->H transpose for inputs.
    // Q: F32 ggml [B,H,S,D] -> BF16 pFlash [B,S,H,D]   (S<->H transpose)
    // K: F16 ggml [B,Hk,S,D] -> BF16 pFlash [B,S,Hk,D] (S<->H transpose; dequant if needed)
    // V: F16 ggml [B,Hk,S,D] -> BF16 pFlash [B,S,Hk,D] (S<->H transpose; dequant if needed)
    // O: pFlash [B,S,H,D] BF16 -> F32 into dst->data    (flat copy, no transpose)

    __nv_bfloat16 *Q_pf;
    __nv_bfloat16 *K_pf;
    __nv_bfloat16 *V_pf;
    __nv_bfloat16 *O_pf;

    CUDA_CHECK(cudaMallocAsync(&Q_pf, Q_n * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&K_pf, K_n * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&V_pf, K_n * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&O_pf, O_n * sizeof(__nv_bfloat16), stream));

    // Q: F32 ggml [B,H,S,D] -> BF16 pFlash [B,S,H,D]  (S<->H transpose)
    k_f32_to_bf16_transpose_sh<<<(Q_n + block - 1) / block, block, 0, stream>>>(
        (const float *)Q->data, Q_pf, B, S, H, D);

    // K: dequantize to F16 if needed, then transpose S<->H into BF16 pFlash layout.
    half * K_f16_buf = nullptr;
    const half * K_f16_src = nullptr;
    if (K->type == GGML_TYPE_F16) {
        K_f16_src = (const half *)K->data;
    } else {
        to_fp16_cuda_t to_fp16_k = ggml_get_to_fp16_cuda(K->type);
        GGML_ASSERT(to_fp16_k != nullptr && "no F16 dequant for K type");
        CUDA_CHECK(cudaMallocAsync(&K_f16_buf, (size_t)K_n * sizeof(half), stream));
        to_fp16_k(K->data, K_f16_buf, K_n, stream);
        K_f16_src = K_f16_buf;
    }
    k_f16_to_bf16_transpose_sh<<<(K_n + block - 1) / block, block, 0, stream>>>(
        K_f16_src, K_pf, B, S, Hk, D);

    // V: dequantize to F16 if needed, then transpose S<->H into BF16 pFlash layout.
    half * V_f16_buf = nullptr;
    const half * V_f16_src = nullptr;
    if (V->type == GGML_TYPE_F16) {
        V_f16_src = (const half *)V->data;
    } else {
        to_fp16_cuda_t to_fp16_v = ggml_get_to_fp16_cuda(V->type);
        GGML_ASSERT(to_fp16_v != nullptr && "no F16 dequant for V type");
        CUDA_CHECK(cudaMallocAsync(&V_f16_buf, (size_t)K_n * sizeof(half), stream));
        to_fp16_v(V->data, V_f16_buf, K_n, stream);
        V_f16_src = V_f16_buf;
    }
    k_f16_to_bf16_transpose_sh<<<(K_n + block - 1) / block, block, 0, stream>>>(
        V_f16_src, V_pf, B, S, Hk, D);

    // Call the registered pFlash kernel.
    // Expects Q[B,S,H,D], K[B,S,Hk,D], V[B,S,Hk,D], O[B,S,H,D] all BF16 contiguous.
    // The registered pFlash kernel launches on the default stream, but the
    // Q/K/V conversions above ran on ctx.stream().  Order them with events so
    // the default stream waits for the conversions and ctx.stream() waits for
    // pFlash's output, without stalling the host (ggml streams are non-blocking,
    // so the legacy default stream does not auto-synchronize with them).
    cudaEvent_t ev_conv;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_conv, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev_conv, stream));
    CUDA_CHECK(cudaStreamWaitEvent(((cudaStream_t)0), ev_conv, 0));
    CUDA_CHECK(cudaEventDestroy(ev_conv));
    int err = s_sparse_kernel(Q_pf, K_pf, V_pf, O_pf,
                              B, S, H, Hk, D, scale, alpha);
    GGML_ASSERT(err == 0 && "sparse attention kernel failed");
    cudaEvent_t ev_pf;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_pf, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev_pf, ((cudaStream_t)0)));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ev_pf, 0));
    CUDA_CHECK(cudaEventDestroy(ev_pf));

    // pFlash output [B,S,H,D] row-major matches ggml dst ne={D,H,S,B} column-major —
    // no transpose needed; flat BF16->F32 copy is correct.
    k_bf16_to_f32_flat<<<(O_n + block - 1) / block, block, 0, stream>>>(
        O_pf, (float *)dst->data, O_n);

    // Free temporary F16 dequant buffers (if allocated) then pFlash BF16 buffers.
    if (K_f16_buf) CUDA_CHECK(cudaFreeAsync(K_f16_buf, stream));
    if (V_f16_buf) CUDA_CHECK(cudaFreeAsync(V_f16_buf, stream));
    CUDA_CHECK(cudaFreeAsync(Q_pf, stream));
    CUDA_CHECK(cudaFreeAsync(K_pf, stream));
    CUDA_CHECK(cudaFreeAsync(V_pf, stream));
    CUDA_CHECK(cudaFreeAsync(O_pf, stream));
}
