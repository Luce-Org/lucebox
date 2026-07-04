// Chunked-prefill FlashAttention driver.
//
// Ported from the non-TBQ path of ggml_cuda_tbq_chunked_prefill. Uses custom
// strided dequant kernels (see fattn-chunked.cuh) so K/V chunks are extracted
// per iteration at chunk_len × nh_kv × D granularity — not full-tensor up
// front. This is the critical perf fix: for a 64K prompt × 48 layers × 350
// prefill steps, we were previously dequanting 64K × nh_kv × D each call.
//
// Structure:
//   for each sequence:
//     copy Q to contiguous fp32
//     init accumulators (O_acc, l_acc, m_acc)
//     for each kv_chunk:
//       strided dequant K chunk → k_tmp (chunk_len × nh_kv × D fp32)
//       strided dequant V chunk → v_tmp (same shape)
//       for each q_batch:
//         if causal-skip: continue
//         S = scale * Q @ K^T  (cuBLAS SGEMM strided batched)
//         online softmax update (kernel updates m/l/O)
//         O_acc += P @ V        (cuBLAS SGEMM strided batched)
//     finalize (divide O by l, transpose to ggml layout)
//   skip cudaStreamSynchronize during CUDA graph capture

#include "fattn-chunked.cuh"

#include <cstdio>
#include <cstdlib>
#include <algorithm>

// Q batch: default to the prefill's own nq (DFlash prefill uses ubatch=192).
// Override via env. 1024 is way too large — inflates S buffer by 5× with no
// perf upside, which in turn collapses the adaptive chunk size.
static int chunked_q_batch_env(int64_t nq) {
    const char * e = getenv("DFLASH27B_CHUNKED_Q_BATCH");
    if (e) {
        int v = atoi(e);
        if (v >= 1) return v;
    }
    return (int)std::min(nq, (int64_t)256);
}

// Chunk sizing: prefer a fixed 4096 by default. The original VRAM-adaptive
// formula runs AFTER we've already claimed persistent scratch, so cudaMemGetInfo
// under-reports and the formula collapses to CHUNKED_PF_MIN=256 → thousands
// of chunks per fattn call → pathological. Override via env if needed.
static int chunked_chunk_env(int fallback_from_vram) {
    const char * e = getenv("DFLASH27B_CHUNKED_CHUNK");
    if (e) {
        int v = atoi(e);
        if (v >= 1) {
            int p = 1;
            while (p < v) p <<= 1;
            return std::min(std::max(p, CHUNKED_PF_MIN), CHUNKED_PF_MAX);
        }
    }
    return std::max(fallback_from_vram, 4096);
}

struct chunked_scratch {
    float * O_acc = nullptr; size_t O_bytes = 0;
    float * l_acc = nullptr; size_t l_bytes = 0;
    float * m_acc = nullptr; size_t m_bytes = 0;
    float * S     = nullptr; size_t S_bytes = 0;
    float * k_tmp = nullptr; size_t k_bytes = 0;
    float * v_tmp = nullptr; size_t v_bytes = 0;
    float * Q_f32 = nullptr; size_t Q_bytes = 0;
};
static chunked_scratch g_chunked_bufs[GGML_CUDA_MAX_DEVICES];

static float * ensure_buf(float ** p, size_t * cur_bytes, size_t need_bytes) {
    if (need_bytes <= *cur_bytes && *p != nullptr) return *p;
    if (*p != nullptr) CUDA_CHECK(cudaFree(*p));
    *p = nullptr;
    CUDA_CHECK(cudaMalloc(p, need_bytes));
    *cur_bytes = need_bytes;
    return *p;
}

// Self-test: round-trip forward→inverse rotation on 128 deterministic floats.
// Runs once per process when DFLASH_TQ3_VERIFY=1.
// Validates that k_tq3_rotate_inplace_f32 is mathematically reversible.
static void tq3_verify_roundtrip(cudaStream_t stream) {
    static bool done = false;
    if (done) return;
    done = true;

    constexpr int N = 128;
    float h_orig[N], h_result[N];
    for (int i = 0; i < N; i++) {
        h_orig[i] = sinf((float)i);
    }

    float * d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_buf, h_orig, N * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Forward rotation (direction=0): one warp, nq=1, nh=1, D=128, one group-of-128
    const dim3 grid1(1, 1, 1);
    k_tq3_rotate_inplace_f32<<<grid1, 32, 0, stream>>>(d_buf, /*nh=*/1, /*nq=*/1, N, /*direction=*/0);
    CUDA_CHECK(cudaGetLastError());

    // Inverse rotation (direction=1)
    k_tq3_rotate_inplace_f32<<<grid1, 32, 0, stream>>>(d_buf, /*nh=*/1, /*nq=*/1, N, /*direction=*/1);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_result, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_buf));

    float max_diff = 0.0f, sum_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_result[i] - h_orig[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    const float mean_diff = sum_diff / N;

    if (max_diff > 1e-3f) {
        std::fprintf(stderr, "[TQ3-VERIFY] FAIL roundtrip max_diff=%.6f mean_diff=%.6f\n",
                     max_diff, mean_diff);
    } else {
        std::fprintf(stderr, "[TQ3-VERIFY] OK roundtrip max_diff=%.6f mean_diff=%.6f\n",
                     max_diff, mean_diff);
    }
}

void ggml_cuda_flash_attn_ext_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    // TQ3 rotation roundtrip self-test (once per process).
    if (std::getenv("DFLASH_TQ3_VERIFY")) {
        tq3_verify_roundtrip(stream);
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t D     = Q->ne[0];
    const int64_t nq    = Q->ne[1];
    const int64_t nh_q  = Q->ne[2];
    const int64_t nh_kv = K->ne[2];
    const int64_t n_seq = Q->ne[3];
    const int64_t nkv   = K->ne[1];
    const int64_t gqa   = nh_q / nh_kv;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(nh_q % nh_kv == 0);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int q_batch_size = (int)std::min((int64_t)chunked_q_batch_env(nq), nq);

    // Skip the host-blocking cudaMemGetInfo when the chunk size is fixed
    // via env var (the default 4096 path). Saves ~500us * N calls per prefill;
    // on a 60-layer 4-prompt-chunk Dense prefill this dropped 48 -> 918 tok/s.
    static const bool dflash_chunk_env_set =
        std::getenv("DFLASH27B_CHUNKED_CHUNK") != nullptr;
    int tbq_chunk;
    if (!dflash_chunk_env_set) {
        // Default path: chunked_chunk_env(0) returns std::max(0, 4096) = 4096,
        // avoiding the cudaMemGetInfo hop entirely.
        tbq_chunk = chunked_chunk_env(0);
    } else {
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const int vram_chunk = chunked_pf_compute_chunk_size(
            free_bytes, nh_q, nh_kv, q_batch_size, D);
        tbq_chunk = chunked_chunk_env(vram_chunk);
    }

    const int device = ctx.device;
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    chunked_scratch & sc = g_chunked_bufs[device];

    const size_t O_bytes    = (size_t)nh_q * nq * D * sizeof(float);
    const size_t l_bytes    = (size_t)nh_q * nq     * sizeof(float);
    const size_t m_bytes    = (size_t)nh_q * nq     * sizeof(float);
    const size_t S_bytes    = (size_t)nh_q * q_batch_size * tbq_chunk * sizeof(float);
    // Per-chunk K/V dequant: [nh_kv, tbq_chunk, D] fp32. The final chunk may
    // be shorter; we still size the buffer for the max and only write chunk_len.
    const size_t kv_bytes   = (size_t)nh_kv * tbq_chunk * D * sizeof(float);
    const size_t Q_f32_bytes = (size_t)nh_q * nq * D * sizeof(float);

    float * O_acc = ensure_buf(&sc.O_acc, &sc.O_bytes, O_bytes);
    float * l_acc = ensure_buf(&sc.l_acc, &sc.l_bytes, l_bytes);
    float * m_acc = ensure_buf(&sc.m_acc, &sc.m_bytes, m_bytes);
    float * S     = ensure_buf(&sc.S,     &sc.S_bytes, S_bytes);
    float * k_tmp = ensure_buf(&sc.k_tmp, &sc.k_bytes, kv_bytes);
    float * v_tmp = ensure_buf(&sc.v_tmp, &sc.v_bytes, kv_bytes);
    float * Q_f32 = ensure_buf(&sc.Q_f32, &sc.Q_bytes, Q_f32_bytes);

    cublasHandle_t cublas_handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    const int64_t mask_stride = mask ? (int64_t)(mask->nb[1] / sizeof(half)) : 0;

    for (int64_t seq = 0; seq < n_seq; seq++) {
        const char * Q_data_seq = (const char *)Q->data + seq * Q->nb[3];
        const char * K_data_seq = (const char *)K->data + seq * K->nb[3];
        const char * V_data_seq = (const char *)V->data + seq * V->nb[3];
        const half * mask_seq   = mask ? (const half *)((const char *)mask->data + seq * mask->nb[3]) : nullptr;
        float      * dst_seq    = (float *)((char *)dst->data + seq * dst->nb[3]);

        if (Q->nb[0] == sizeof(float) &&
            Q->nb[1] == (size_t)D * sizeof(float) &&
            Q->nb[2] == (size_t)D * nq * sizeof(float)) {
            CUDA_CHECK(cudaMemcpyAsync(Q_f32, Q_data_seq,
                                       (size_t)nh_q * nq * D * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            GGML_ASSERT(Q->nb[0] == sizeof(float));
            for (int64_t h = 0; h < nh_q; h++) {
                for (int64_t q = 0; q < nq; q++) {
                    const char * src_ptr = Q_data_seq + h * Q->nb[2] + q * Q->nb[1];
                    float * dst_ptr = Q_f32 + h * nq * D + q * D;
                    CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, D * sizeof(float),
                                               cudaMemcpyDeviceToDevice, stream));
                }
            }
        }

        // Init accumulators.
        {
            const int64_t nq_heads = nh_q * nq;
            const int threads_init = (int)std::min(D, (int64_t)1024);
            k_chunked_attn_init<<<(int)nq_heads, threads_init, 0, stream>>>(
                O_acc, l_acc, m_acc, nq_heads, D);
            CUDA_CHECK(cudaGetLastError());
        }

        for (int64_t kv_start = 0; kv_start < nkv; kv_start += tbq_chunk) {
            const int64_t chunk_len = (kv_start + tbq_chunk <= nkv) ? tbq_chunk : (nkv - kv_start);

            // Strided dequant K and V chunk to contiguous [nh_kv, chunk_len, D] fp32.
            if (!chunked_dequant_launch(K->type, K_data_seq, k_tmp,
                                        D, chunk_len, nh_kv, K->nb[1], K->nb[2],
                                        kv_start, stream)) {
                GGML_ABORT("chunked prefill: unsupported K type");
            }
            if (!chunked_dequant_launch(V->type, V_data_seq, v_tmp,
                                        D, chunk_len, nh_kv, V->nb[1], V->nb[2],
                                        kv_start, stream)) {
                GGML_ABORT("chunked prefill: unsupported V type");
            }
            CUDA_CHECK(cudaGetLastError());

            for (int64_t q_start = 0; q_start < nq; q_start += q_batch_size) {
                const int64_t q_len = std::min((int64_t)q_batch_size, nq - q_start);
                if ((nkv - nq) + q_start + q_len <= kv_start) continue;

                // S = scale * Q @ K^T. k_tmp layout is [nh_kv, chunk_len, D]
                // with per-head stride chunk_len*D (compact — NOT nkv*D).
                {
                    const float alpha_v = scale;
                    const float beta_v  = 0.0f;
                    const long long stride_A = (long long)chunk_len * D;
                    const long long stride_B = (long long)nq * D;
                    const long long stride_C = (long long)q_len * tbq_chunk;

                    if (gqa == 1) {
                        CUBLAS_CHECK(cublasSgemmStridedBatched(
                            cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            (int)chunk_len, (int)q_len, (int)D,
                            &alpha_v,
                            k_tmp,                (int)D,         stride_A,
                            Q_f32 + q_start * D,  (int)D,         stride_B,
                            &beta_v,
                            S,                    (int)tbq_chunk, stride_C,
                            (int)nh_q));
                    } else {
                        for (int64_t kv_h = 0; kv_h < nh_kv; kv_h++) {
                            const float * k_head  = k_tmp + kv_h * chunk_len * D;
                            const float * q_ptr   = Q_f32 + kv_h * gqa * nq * D + q_start * D;
                            float       * s_start = S     + kv_h * gqa * (long long)q_len * tbq_chunk;
                            CUBLAS_CHECK(cublasSgemmStridedBatched(
                                cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                (int)chunk_len, (int)q_len, (int)D,
                                &alpha_v,
                                k_head,   (int)D,         0LL,
                                q_ptr,    (int)D,         (long long)nq * D,
                                &beta_v,
                                s_start,  (int)tbq_chunk, (long long)q_len * tbq_chunk,
                                (int)gqa));
                        }
                    }
                }

                // Online softmax.
                {
                    const int64_t nq_batch_heads = nh_q * q_len;
                    const int chunk_len_int = (int)chunk_len;
                    const int chunk_pad     = chunked_pf_next_pow2(chunk_len_int);
                    const int threads_sm    = (chunk_pad < 1024) ? chunk_pad : 1024;
                    const size_t smem       = ((size_t)chunk_pad + 2) * sizeof(float);
                    k_chunked_softmax_update<<<(int)nq_batch_heads, threads_sm, smem, stream>>>(
                        S, O_acc, l_acc, m_acc,
                        chunk_len_int, chunk_pad,
                        D, nq, q_len, q_start,
                        nh_q,
                        mask_seq, mask_stride, kv_start, tbq_chunk);
                    CUDA_CHECK(cudaGetLastError());
                }

                // O_acc += P @ V. v_tmp layout [nh_kv, chunk_len, D], per-head stride chunk_len*D.
                {
                    const float alpha_v = 1.0f;
                    const float beta_v  = 1.0f;
                    const long long stride_A = (long long)chunk_len * D;
                    const long long stride_B = (long long)q_len * tbq_chunk;
                    const long long stride_C = (long long)nq * D;

                    if (gqa == 1) {
                        CUBLAS_CHECK(cublasSgemmStridedBatched(
                            cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            (int)D, (int)q_len, (int)chunk_len,
                            &alpha_v,
                            v_tmp,                (int)D,         stride_A,
                            S,                    (int)tbq_chunk, stride_B,
                            &beta_v,
                            O_acc + q_start * D,  (int)D,         stride_C,
                            (int)nh_q));
                    } else {
                        for (int64_t kv_h = 0; kv_h < nh_kv; kv_h++) {
                            const float * v_head  = v_tmp  + kv_h * chunk_len * D;
                            const float * p_start = S      + kv_h * gqa * (long long)q_len * tbq_chunk;
                            float       * o_start = O_acc  + kv_h * gqa * (long long)nq * D + q_start * D;
                            CUBLAS_CHECK(cublasSgemmStridedBatched(
                                cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                (int)D, (int)q_len, (int)chunk_len,
                                &alpha_v,
                                v_head,   (int)D,         0LL,
                                p_start,  (int)tbq_chunk, (long long)q_len * tbq_chunk,
                                &beta_v,
                                o_start,  (int)D,         (long long)nq * D,
                                (int)gqa));
                        }
                    }
                }
            }
        }

        // Finalize.
        {
            const int64_t nq_heads = nh_q * nq;
            const int threads_fin = 128;
            const dim3 grid_fin((int)nq_heads, (int)((D + threads_fin - 1) / threads_fin));
            k_chunked_attn_finalize<<<grid_fin, threads_fin, 0, stream>>>(
                O_acc, l_acc, dst_seq, nq, nh_q, D);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cudaStreamCaptureStatus status;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err == cudaSuccess && status == cudaStreamCaptureStatusNone) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
