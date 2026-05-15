// dflash_feature_ring.cpp — implementation for common DFlash feature ring.

#include "dflash_feature_ring.h"
#include "peer_access.h"

#include "ggml.h"

// ggml_get_to_fp32_cuda is not in any public header — it lives in
// ggml-cuda/convert.cuh. Declare the typedef + extern here so we can link
// against it from this TU.
using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
extern "C++" to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

#include <algorithm>
#include <cstdio>

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace dflash27b {

// ── internal helpers ────────────────────────────────────────────

static bool ensure_bf16_staging(DraftFeatureMirror & mirror, size_t elems) {
    if (elems <= mirror.bf16_staging_elems) return true;
    cudaError_t err = cudaSetDevice(mirror.device);
    if (err != cudaSuccess) return false;
    if (mirror.bf16_staging) {
        cudaFree(mirror.bf16_staging);
        mirror.bf16_staging = nullptr;
        mirror.bf16_staging_elems = 0;
    }
    err = cudaMalloc(&mirror.bf16_staging, elems * sizeof(uint16_t));
    if (err != cudaSuccess) return false;
    mirror.bf16_staging_elems = elems;
    return true;
}

// ── public API ──────────────────────────────────────────────────

void draft_feature_mirror_free(DraftFeatureMirror & mirror) {
    if (mirror.bf16_staging) {
        cudaSetDevice(mirror.device);
        cudaFree(mirror.bf16_staging);
        mirror.bf16_staging = nullptr;
        mirror.bf16_staging_elems = 0;
    }
    if (mirror.buf) {
        ggml_backend_buffer_free(mirror.buf);
        mirror.buf = nullptr;
    }
    if (mirror.ctx) {
        ggml_free(mirror.ctx);
        mirror.ctx = nullptr;
    }
    mirror.target_feat = nullptr;
    mirror.device = 0;
    mirror.target_device = 0;
    mirror.cap = 0;
}

bool draft_feature_mirror_init(DraftFeatureMirror & mirror,
                               ggml_backend_t backend,
                               int device,
                               int target_device,
                               int cap,
                               int n_target_layers,
                               int hidden_size) {
    draft_feature_mirror_free(mirror);
    if (cap <= 0 || n_target_layers <= 0 || hidden_size <= 0) return false;
    mirror.device = device;
    mirror.target_device = target_device;
    mirror.n_target_layers = n_target_layers;
    mirror.hidden_size = hidden_size;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    mirror.ctx = ggml_init(ip);
    if (!mirror.ctx) return false;

    const int fc_in = n_target_layers * hidden_size;
    mirror.target_feat = ggml_new_tensor_2d(mirror.ctx, GGML_TYPE_F32, fc_in, cap);
    ggml_set_name(mirror.target_feat, "draft_target_feat_mirror");
    mirror.buf = ggml_backend_alloc_ctx_tensors(mirror.ctx, backend);
    if (!mirror.buf) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    const size_t bytes = (size_t)fc_in * (size_t)cap * sizeof(float);
    cudaSetDevice(device);
    cudaError_t err = cudaMemset(mirror.target_feat->data, 0, bytes);
    if (err != cudaSuccess) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    mirror.cap = cap;
    return true;
}

bool draft_feature_mirror_can_view(const DraftFeatureMirror & mirror,
                                   int committed,
                                   int ctx_len,
                                   int & slot0) {
    if (!mirror.target_feat || mirror.cap <= 0) return false;
    if (ctx_len <= 0 || ctx_len > mirror.cap || committed < ctx_len) return false;
    const int start = committed - ctx_len;
    slot0 = start % mirror.cap;
    return slot0 + ctx_len <= mirror.cap;
}

bool draft_feature_mirror_sync_range(const ggml_tensor * src_target_feat,
                                     int src_cap,
                                     DraftFeatureMirror & mirror,
                                     int start_pos,
                                     int n_tokens) {
    if (!src_target_feat || !mirror.target_feat || mirror.cap <= 0 || src_cap <= 0) return false;
    if (n_tokens <= 0) return true;
    if (n_tokens > mirror.cap) return false;

    const int fc_in = mirror.n_target_layers * mirror.hidden_size;
    const size_t src_stride = src_target_feat->nb[1];
    const size_t dst_stride = mirror.target_feat->nb[1];

    int done = 0;
    while (done < n_tokens) {
        const int src_slot = (start_pos + done) % src_cap;
        const int dst_slot = (start_pos + done) % mirror.cap;
        const int src_run = src_cap - src_slot;
        const int dst_run = mirror.cap - dst_slot;
        const int run = std::min(n_tokens - done, std::min(src_run, dst_run));
        const size_t elems = (size_t)run * (size_t)fc_in;
        const void * src =
            (const char *)src_target_feat->data + (size_t)src_slot * src_stride;
        void * dst =
            (char *)mirror.target_feat->data + (size_t)dst_slot * dst_stride;
        auto bf16_to_f32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        if (mirror.device == mirror.target_device) {
            cudaSetDevice(mirror.device);
            bf16_to_f32(src, (float *)dst, (int64_t)elems, nullptr);
        } else {
            if (!ensure_bf16_staging(mirror, elems)) return false;
            if (!copy_peer_async(mirror.bf16_staging, mirror.device,
                                 src, mirror.target_device,
                                 elems * sizeof(uint16_t))) {
                return false;
            }
            cudaSetDevice(mirror.device);
            bf16_to_f32(mirror.bf16_staging, (float *)dst, (int64_t)elems, nullptr);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return false;
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool draft_feature_mirror_sync_tail(const ggml_tensor * src_target_feat,
                                    int src_cap,
                                    DraftFeatureMirror & mirror,
                                    int committed) {
    if (!mirror.target_feat || committed <= 0) return true;
    const int n = std::min(committed, mirror.cap);
    return draft_feature_mirror_sync_range(src_target_feat, src_cap, mirror,
                                           committed - n, n);
}

// ── Ring ↔ tensor copy helpers ──────────────────────────────────

bool copy_capture_slice_to_draft_ring(
    DraftFeatureMirror & feature_ring,
    int capture_idx,
    const ggml_tensor * act_out,
    int src_device,
    int chunk_start,
    int start_pos,
    int n_tokens) {
    if (!feature_ring.target_feat || capture_idx < 0 || n_tokens <= 0 || start_pos < 0) return true;
    if (feature_ring.cap <= 0) return false;
    const int hidden = feature_ring.hidden_size;
    const size_t dst_stride = feature_ring.target_feat->nb[1];
    const size_t src_stride = act_out->nb[1];
    const size_t row_bytes = (size_t)hidden * sizeof(float);
    for (int i = 0; i < n_tokens; i++) {
        const int slot = (start_pos + i) % feature_ring.cap;
        const void * src = (const char *)act_out->data +
            (size_t)(chunk_start + i) * src_stride;
        void * dst = (char *)feature_ring.target_feat->data +
            (size_t)slot * dst_stride +
            (size_t)capture_idx * (size_t)hidden * sizeof(float);
        if (!copy_peer_async(dst, feature_ring.device, src, src_device, row_bytes)) {
            return false;
        }
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool copy_feature_ring_range_to_tensor(
    const DraftFeatureMirror & feature_ring,
    ggml_tensor * dst,
    int start_pos,
    int n_tokens) {
    if (!feature_ring.target_feat || !dst || feature_ring.cap <= 0) return false;
    if (n_tokens <= 0 || n_tokens > feature_ring.cap) return false;

    const int fc_in = feature_ring.n_target_layers * feature_ring.hidden_size;
    const size_t row_bytes = (size_t)fc_in * sizeof(float);
    const size_t src_stride = feature_ring.target_feat->nb[1];
    const size_t dst_stride = dst->nb[1];
    int done = 0;
    while (done < n_tokens) {
        const int slot = (start_pos + done) % feature_ring.cap;
        const int run = std::min(n_tokens - done, feature_ring.cap - slot);
        const char * src_base =
            (const char *)feature_ring.target_feat->data + (size_t)slot * src_stride;
        char * dst_base = (char *)dst->data + (size_t)done * dst_stride;
        if (src_stride == row_bytes && dst_stride == row_bytes) {
            if (!copy_peer_async(dst_base, feature_ring.device,
                                 src_base, feature_ring.device,
                                 row_bytes * (size_t)run)) {
                return false;
            }
        } else {
            for (int i = 0; i < run; i++) {
                if (!copy_peer_async(dst_base + (size_t)i * dst_stride,
                                     feature_ring.device,
                                     src_base + (size_t)i * src_stride,
                                     feature_ring.device,
                                     row_bytes)) {
                    return false;
                }
            }
        }
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace dflash27b
