// pinned_stage.h - pinned host staging for a graph launch's small transfers.
//
// A tensor_set or tensor_get from pageable memory is a full round trip on
// HIP: the call returns only after the copy ran, and a speculative step does a
// few dozen of them (inputs, readbacks), each ~30 us with the GPU idle. Copies
// from pinned memory are truly asynchronous, so a step pays one wait for all
// of them instead of one per tensor.
//
// Usage per launch: reset(), take() slices for each transfer, issue the async
// copies on the backend that owns the tensors, synchronize that backend once,
// then read the readback slices. A slice stays valid until the next reset(),
// and must not be reused before the copies that read or write it completed.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace luce::common {

class PinnedStage {
public:
    PinnedStage() = default;
    PinnedStage(const PinnedStage &) = delete;
    PinnedStage & operator=(const PinnedStage &) = delete;
    // Moving hands the block over; the caller has no copies in flight.
    PinnedStage(PinnedStage && other) noexcept { take_from(other); }
    PinnedStage & operator=(PinnedStage && other) noexcept {
        if (this != &other) {
            release();
            take_from(other);
        }
        return *this;
    }
    ~PinnedStage() { release(); }

    // Room for `bytes` in one launch. Growing frees the old block, so the
    // caller must not have copies in flight from it.
    bool reserve(ggml_backend_t backend, size_t bytes) {
        if (buf_ && size_ >= bytes && backend_ == backend) return true;
        release();
        ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
        ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (!buft) return false;
        const size_t want = std::max(bytes, (size_t) 1 << 20);
        buf_ = ggml_backend_buft_alloc_buffer(buft, want);
        if (!buf_) return false;
        base_ = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buf_));
        if (!base_) { release(); return false; }
        size_ = want;
        backend_ = backend;
        return true;
    }

    void reset() { used_ = 0; }

    // Batched mode: set() collects its device uploads instead of queueing
    // one copy each, and the caller sends batch() as a single launch (one
    // kernel reading the pinned slices; a copy engine adds a start latency
    // of 0.03-1 ms to every step on a discrete GPU). end_batch() before the
    // next reset; the slices stay valid until then.
    struct Pending {
        const void * src;
        void *       dst;
        size_t       bytes;
    };
    void begin_batch() { batching_ = true; pending_.clear(); }
    const std::vector<Pending> & batch() const { return pending_; }
    void end_batch() { batching_ = false; pending_.clear(); }

    // A 64-byte aligned slice, or nullptr when the stage is missing or full
    // (the caller then takes its synchronous path).
    void * take(size_t bytes) {
        const size_t off = (used_ + 63) & ~(size_t) 63;
        if (!base_ || off + bytes > size_) return nullptr;
        used_ = off + bytes;
        return base_ + off;
    }

    // Asynchronous upload of `data` into `t` on `backend` through a slice.
    // Falls back to the blocking set when no slice is left or the tensor does
    // not live in `backend`'s device memory.
    void set(ggml_backend_t backend, ggml_tensor * t, const void * data,
             size_t offset, size_t bytes) {
        if (!t || !t->buffer || bytes == 0) return;
        void * slice = owned_by(backend, t) ? take(bytes) : nullptr;
        if (!slice) {
            ggml_backend_tensor_set(t, data, offset, bytes);
            return;
        }
        std::memcpy(slice, data, bytes);
        if (batching_) {
            pending_.push_back({slice, static_cast<uint8_t *>(t->data) + offset, bytes});
            return;
        }
        ggml_backend_tensor_set_async(backend, t, slice, offset, bytes);
    }

    // Asynchronous readback of `t` into a slice; the bytes are valid after
    // the backend was synchronized. nullptr: read it with the blocking get.
    const void * get(ggml_backend_t backend, const ggml_tensor * t,
                     size_t offset, size_t bytes) {
        if (!t || !t->buffer || bytes == 0 || !owned_by(backend, t)) return nullptr;
        void * slice = take(bytes);
        if (!slice) return nullptr;
        ggml_backend_tensor_get_async(backend, t, slice, offset, bytes);
        return slice;
    }

private:
    static bool owned_by(ggml_backend_t backend, const ggml_tensor * t) {
        const ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
        return backend && buf && !ggml_backend_buffer_is_host(buf) &&
               ggml_backend_buffer_get_type(buf) == ggml_backend_get_default_buffer_type(backend);
    }

    void take_from(PinnedStage & other) {
        buf_ = other.buf_;
        base_ = other.base_;
        size_ = other.size_;
        used_ = other.used_;
        backend_ = other.backend_;
        other.buf_ = nullptr;
        other.base_ = nullptr;
        other.size_ = other.used_ = 0;
        other.backend_ = nullptr;
    }

    void release() {
        if (buf_) ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
        base_ = nullptr;
        size_ = used_ = 0;
        backend_ = nullptr;
    }

    ggml_backend_buffer_t buf_ = nullptr;
    uint8_t * base_ = nullptr;
    size_t size_ = 0;
    size_t used_ = 0;
    ggml_backend_t backend_ = nullptr;
    bool batching_ = false;
    std::vector<Pending> pending_;
};

}  // namespace luce::common
