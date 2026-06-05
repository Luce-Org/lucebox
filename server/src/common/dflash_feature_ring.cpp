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
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

#include "gpu_runtime_compat.h"

namespace dflash::common {

// ── internal helpers ────────────────────────────────────────────

static bool ensure_staging(DraftFeatureMirror & mirror, size_t bytes) {
    if (bytes <= mirror.staging_bytes) return true;
    cudaError_t err = cudaSetDevice(mirror.device);
    if (err != cudaSuccess) return false;
    if (mirror.staging) {
        err = cudaFree(mirror.staging);
        if (err != cudaSuccess) return false;
        mirror.staging = nullptr;
        mirror.staging_bytes = 0;
    }
    err = cudaMalloc(&mirror.staging, bytes);
    if (err != cudaSuccess) return false;
    mirror.staging_bytes = bytes;
    return true;
}

static ggml_type parse_feature_dtype() {
    const char * s = std::getenv("DFLASH_FEATURE_DTYPE");
    if (!s || !s[0] || std::strcmp(s, "f32") == 0 || std::strcmp(s, "F32") == 0) {
        return GGML_TYPE_F32;
    }
    if (std::strcmp(s, "f16") == 0 || std::strcmp(s, "F16") == 0) {
        return GGML_TYPE_F16;
    }
    if (std::strcmp(s, "bf16") == 0 || std::strcmp(s, "BF16") == 0) {
        return GGML_TYPE_BF16;
    }
    if (std::strcmp(s, "q8_0") == 0 || std::strcmp(s, "Q8_0") == 0 ||
        std::strcmp(s, "q8") == 0 || std::strcmp(s, "Q8") == 0) {
        return GGML_TYPE_Q8_0;
    }
    std::fprintf(stderr, "[dflash-feature] ignoring unsupported DFLASH_FEATURE_DTYPE=%s\n", s);
    return GGML_TYPE_F32;
}

static bool check_feature_width_compatible(ggml_type type, int width) {
    const int64_t blck = ggml_blck_size(type);
    return blck > 0 && width > 0 && width % blck == 0;
}

static bool quantize_host_f32_to_feature_type(ggml_type type,
                                              const float * src,
                                              void * dst,
                                              size_t elems) {
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->from_float_ref) return false;
    if (traits->blck_size <= 0 || elems % (size_t)traits->blck_size != 0) return false;
    traits->from_float_ref(src, dst, (int64_t)elems);
    return true;
}

static bool dequantize_feature_type_to_host_f32(ggml_type type,
                                                const void * src,
                                                float * dst,
                                                size_t elems) {
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) return false;
    if (traits->blck_size <= 0 || elems % (size_t)traits->blck_size != 0) return false;
    traits->to_float(src, dst, (int64_t)elems);
    return true;
}

static bool host_f32_to_feature_row(ggml_type type,
                                    const float * src,
                                    void * dst,
                                    size_t elems) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, elems * sizeof(float));
        return true;
    }
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *)dst, (int64_t)elems);
        return true;
    }
    if (type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(src, (ggml_bf16_t *)dst, (int64_t)elems);
        return true;
    }
    return quantize_host_f32_to_feature_type(type, src, dst, elems);
}

static bool feature_row_to_host_f32(ggml_type type,
                                    const void * src,
                                    float * dst,
                                    size_t elems) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, elems * sizeof(float));
        return true;
    }
    if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)src, dst, (int64_t)elems);
        return true;
    }
    if (type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *)src, dst, (int64_t)elems);
        return true;
    }
    return dequantize_feature_type_to_host_f32(type, src, dst, elems);
}

static bool convert_device_f32_to_feature_type(DraftFeatureMirror & mirror,
                                               const void * src,
                                               int src_device,
                                               void * dst,
                                               size_t elems) {
    if (mirror.storage_type == GGML_TYPE_F32) {
        return copy_peer_async(dst, mirror.device, src, src_device,
                               elems * sizeof(float));
    }

    std::vector<float> host(elems);
    cudaError_t err = cudaSetDevice(src_device);
    if (err != cudaSuccess) return false;
    err = cudaMemcpy(host.data(), src, elems * sizeof(float),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return false;

    const size_t row_bytes = ggml_row_size(mirror.storage_type, (int64_t)elems);
    std::vector<uint8_t> tmp(row_bytes);
    if (!host_f32_to_feature_row(mirror.storage_type, host.data(), tmp.data(), elems)) {
        return false;
    }
    ggml_backend_tensor_set(mirror.target_feat, tmp.data(),
                            (size_t)((char *)dst - (char *)mirror.target_feat->data),
                            row_bytes);
    return true;
}

static bool convert_bf16_feature_to_storage(DraftFeatureMirror & mirror,
                                            const void * src,
                                            int src_device,
                                            void * dst,
                                            size_t elems) {
    if (mirror.storage_type == GGML_TYPE_BF16) {
        return copy_peer_async(dst, mirror.device, src, src_device,
                               elems * sizeof(ggml_bf16_t));
    }

    const size_t blck = (size_t)ggml_blck_size(mirror.storage_type);
    if (blck == 0 || elems % blck != 0) return false;

    constexpr size_t max_chunk_bytes = 4u * 1024u * 1024u;
    const size_t max_chunk_elems =
        std::max(blck, (max_chunk_bytes / sizeof(float) / blck) * blck);
    const size_t dst_offset = (size_t)((char *)dst - (char *)mirror.target_feat->data);

    cudaError_t err = cudaSetDevice(src_device);
    if (err != cudaSuccess) return false;

    size_t done = 0;
    size_t dst_bytes_done = 0;
    while (done < elems) {
        size_t chunk = std::min(elems - done, max_chunk_elems);
        chunk = (chunk / blck) * blck;
        if (chunk == 0) return false;

        std::vector<ggml_bf16_t> bf16_host(chunk);
        err = cudaMemcpy(bf16_host.data(),
                         (const char *)src + done * sizeof(ggml_bf16_t),
                         chunk * sizeof(ggml_bf16_t),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return false;

        std::vector<float> host(chunk);
        ggml_bf16_to_fp32_row(bf16_host.data(), host.data(), (int64_t)chunk);

        const size_t chunk_bytes = ggml_row_size(mirror.storage_type, (int64_t)chunk);
        std::vector<uint8_t> tmp(chunk_bytes);
        if (!host_f32_to_feature_row(mirror.storage_type, host.data(), tmp.data(), chunk)) {
            return false;
        }
        ggml_backend_tensor_set(mirror.target_feat, tmp.data(),
                                dst_offset + dst_bytes_done, chunk_bytes);
        done += chunk;
        dst_bytes_done += chunk_bytes;
    }
    return true;
}

static bool copy_feature_to_f32(DraftFeatureMirror & mirror,
                                const void * src,
                                int src_device,
                                float * dst,
                                size_t elems) {
    if (mirror.storage_type == GGML_TYPE_F32) {
        return copy_peer_async(dst, mirror.device, src, src_device,
                               elems * sizeof(float));
    }
    auto to_f32 = ggml_get_to_fp32_cuda(mirror.storage_type);
    if (!to_f32) return false;
    const size_t src_bytes = ggml_row_size(mirror.storage_type, (int64_t)elems);
    if (src_device != mirror.device) {
        if (!ensure_staging(mirror, src_bytes)) return false;
        if (!copy_peer_async(mirror.staging, mirror.device, src, src_device,
                             src_bytes)) {
            return false;
        }
        src = mirror.staging;
    }
    cudaError_t err = cudaSetDevice(mirror.device);
    if (err != cudaSuccess) return false;
    to_f32(src, dst, (int64_t)elems, nullptr);
    return cudaGetLastError() == cudaSuccess;
}

// ── public API ──────────────────────────────────────────────────

void draft_feature_mirror_free(DraftFeatureMirror & mirror) {
    if (mirror.staging) {
        (void)cudaSetDevice(mirror.device);
        (void)cudaFree(mirror.staging);
        mirror.staging = nullptr;
        mirror.staging_bytes = 0;
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
    mirror.storage_type = GGML_TYPE_F32;
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
    mirror.storage_type = parse_feature_dtype();
    if (!check_feature_width_compatible(mirror.storage_type, hidden_size) ||
        !check_feature_width_compatible(mirror.storage_type, n_target_layers * hidden_size)) {
        std::fprintf(stderr,
                     "[dflash-feature] unsupported mirror dtype=%s for hidden=%d layers=%d\n",
                     ggml_type_name(mirror.storage_type), hidden_size, n_target_layers);
        mirror.storage_type = GGML_TYPE_F32;
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    mirror.ctx = ggml_init(ip);
    if (!mirror.ctx) return false;

    const int fc_in = n_target_layers * hidden_size;
    mirror.target_feat = ggml_new_tensor_2d(mirror.ctx, mirror.storage_type, fc_in, cap);
    ggml_set_name(mirror.target_feat, "draft_target_feat_mirror");
    mirror.buf = ggml_backend_alloc_ctx_tensors(mirror.ctx, backend);
    if (!mirror.buf) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    const size_t bytes = ggml_nbytes(mirror.target_feat);
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    err = cudaMemset(mirror.target_feat->data, 0, bytes);
    if (err != cudaSuccess) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    mirror.cap = cap;
    std::fprintf(stderr, "[dflash-feature] mirror dtype=%s cap=%d fc_in=%d\n",
                 ggml_type_name(mirror.storage_type), cap, fc_in);
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
    return mirror.storage_type == GGML_TYPE_F32 && slot0 + ctx_len <= mirror.cap;
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
        if (!convert_bf16_feature_to_storage(mirror, src, mirror.target_device, dst, elems)) {
            return false;
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
    const size_t row_elems = (size_t)hidden;
    for (int i = 0; i < n_tokens; i++) {
        const int slot = (start_pos + i) % feature_ring.cap;
        const void * src = (const char *)act_out->data +
            (size_t)(chunk_start + i) * src_stride;
        void * dst = (char *)feature_ring.target_feat->data +
            (size_t)slot * dst_stride +
            (size_t)capture_idx * ggml_row_size(feature_ring.storage_type, hidden);
        if (!convert_device_f32_to_feature_type(feature_ring, src, src_device, dst, row_elems)) {
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
        if (feature_ring.storage_type == GGML_TYPE_F32 &&
            src_stride == row_bytes && dst_stride == row_bytes) {
            if (!copy_peer_async(dst_base, feature_ring.device,
                                 src_base, feature_ring.device,
                                 row_bytes * (size_t)run)) {
                return false;
            }
        } else {
            for (int i = 0; i < run; i++) {
                if (!copy_feature_to_f32(
                        const_cast<DraftFeatureMirror &>(feature_ring),
                        src_base + (size_t)i * src_stride,
                        feature_ring.device,
                        (float *)(dst_base + (size_t)i * dst_stride),
                        (size_t)fc_in)) {
                    return false;
                }
            }
        }
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool copy_feature_ring_range_to_host_f32(
    const DraftFeatureMirror & feature_ring,
    int start_pos,
    int n_tokens,
    std::vector<float> & out) {
    if (!feature_ring.target_feat || feature_ring.cap <= 0) return false;
    if (n_tokens <= 0 || n_tokens > feature_ring.cap) return false;

    const int fc_in = feature_ring.n_target_layers * feature_ring.hidden_size;
    const size_t row_bytes = ggml_row_size(feature_ring.storage_type, fc_in);
    const size_t src_stride = feature_ring.target_feat->nb[1];
    std::vector<uint8_t> row(row_bytes);
    out.resize((size_t)n_tokens * (size_t)fc_in);

    for (int i = 0; i < n_tokens; ++i) {
        const int slot = (start_pos + i) % feature_ring.cap;
        ggml_backend_tensor_get(feature_ring.target_feat, row.data(),
                                (size_t)slot * src_stride, row_bytes);
        float * dst = out.data() + (size_t)i * (size_t)fc_in;
        if (!feature_row_to_host_f32(feature_ring.storage_type, row.data(), dst, fc_in)) {
            return false;
        }
    }
    return true;
}

bool copy_host_f32_to_feature_ring_range(
    DraftFeatureMirror & feature_ring,
    int start_pos,
    int n_tokens,
    const std::vector<float> & src) {
    if (!feature_ring.target_feat || feature_ring.cap <= 0) return false;
    if (n_tokens <= 0 || n_tokens > feature_ring.cap) return false;

    const int fc_in = feature_ring.n_target_layers * feature_ring.hidden_size;
    if (src.size() < (size_t)n_tokens * (size_t)fc_in) return false;
    const size_t dst_stride = feature_ring.target_feat->nb[1];
    const size_t row_bytes = ggml_row_size(feature_ring.storage_type, fc_in);
    std::vector<uint8_t> row(row_bytes);

    for (int i = 0; i < n_tokens; ++i) {
        const float * src_row = src.data() + (size_t)i * (size_t)fc_in;
        if (!host_f32_to_feature_row(feature_ring.storage_type, src_row, row.data(), fc_in)) {
            return false;
        }
        const int slot = (start_pos + i) % feature_ring.cap;
        ggml_backend_tensor_set(feature_ring.target_feat, row.data(),
                                (size_t)slot * dst_stride, row_bytes);
    }
    return true;
}

}  // namespace dflash::common
