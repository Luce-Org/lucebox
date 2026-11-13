// A pinned head as a snapshot the disk cache can hold: three named byte tensors (pager header, chunk payloads,
// slab), cur_pos = k chunks, saved under a caller-named layout (DiskPrefixCache::use_layout) and keyed by the head's
// token ids. The key is verified by the disk cache on read; the shape is verified here.
#pragma once

#include "kvflash_pinned_head.h"
#include "model_backend.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstring>

namespace luce::common {

constexpr const char * kKvFlashHeadTensors[3] = {"kvf_head_hdr", "kvf_head_chunks", "kvf_head_slab"};

// Empty (a snapshot with no context) when a part is empty or the host buffer cannot be allocated.
inline OwnedSnapshot kvflash_head_to_snapshot(const KvFlashHead & h, int ct) {
    const std::vector<uint8_t> * part[3] = {&h.hdr, &h.chunks, &h.slab};
    if (h.k <= 0 || ct <= 0 || h.hdr.empty() || h.chunks.empty() || h.slab.empty()) return {};
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return {};
    ggml_tensor * t[3];
    for (int i = 0; i < 3; i++) {
        t[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t)part[i]->size());
        ggml_set_name(t[i], kKvFlashHeadTensors[i]);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (!buf) { ggml_free(ctx); return {}; }
    for (int i = 0; i < 3; i++) ggml_backend_tensor_set(t[i], part[i]->data(), 0, part[i]->size());
    return OwnedSnapshot(ctx, buf, h.k * ct, 0);
}

// False (out untouched) unless the snapshot holds the three parts, a whole number k >= 2 of chunks that `ids` covers,
// and k x chunk_bytes of chunk payload.
inline bool kvflash_head_from_snapshot(const OwnedSnapshot & snap, const std::vector<int32_t> & ids, int ct,
                                       size_t chunk_bytes, KvFlashHead & out) {
    if (snap.empty() || ct <= 0 || snap.cur_pos() <= 0 || snap.cur_pos() % ct != 0) return false;
    const int k = snap.cur_pos() / ct;
    if (k < 2 || (int)ids.size() < snap.cur_pos()) return false;
    std::vector<uint8_t> part[3];
    for (int i = 0; i < 3; i++) {
        const ggml_tensor * t = ggml_get_tensor(snap.ctx(), kKvFlashHeadTensors[i]);
        if (!t || t->type != GGML_TYPE_I8 || ggml_nbytes(t) == 0) return false;
        part[i].resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, part[i].data(), 0, part[i].size());
    }
    if (part[1].size() != (size_t)k * chunk_bytes) return false;
    const auto path = kvflash_head_hashes(ids, k, ct);
    if ((int)path.size() < k) return false;
    out.k = k; out.hash = path.back();
    out.hdr = std::move(part[0]); out.chunks = std::move(part[1]); out.slab = std::move(part[2]);
    return true;
}

}  // namespace luce::common
