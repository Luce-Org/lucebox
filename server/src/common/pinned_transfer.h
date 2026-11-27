// pinned_transfer.h - double-buffered pinned host streaming for bulk KV copies.
//
// Large pageable host<->device transfers split across many small slices (paged
// KV offload, prefix snapshot gather/scatter) stall on ROCm/HIP when issued
// directly against pageable memory. This helper streams arbitrary spans through
// two reusable pinned host buffers (2 x 4 MiB), coalescing small copies into
// batches and overlapping host memcpy with asynchronous GPU DMA.
//
// Safe on CPU backends and backends without pinned host memory or events: falls
// back transparently to direct memcpy (host tensors) or ggml async copies.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace luce::common {

class PinnedTransfer {
public:
    static constexpr size_t kSlotBytes = size_t(4) << 20;  // 4 MiB per bounce slot

    PinnedTransfer() = default;
    PinnedTransfer(const PinnedTransfer &) = delete;
    PinnedTransfer & operator=(const PinnedTransfer &) = delete;
    PinnedTransfer(PinnedTransfer && other) noexcept { take_from(other); }
    PinnedTransfer & operator=(PinnedTransfer && other) noexcept {
        if (this != &other) {
            release();
            take_from(other);
        }
        return *this;
    }
    ~PinnedTransfer() { release(); }

    void release() {
        for (int i = 0; i < 2; ++i) {
            if (ev_[i]) {
                ggml_backend_event_free(ev_[i]);
                ev_[i] = nullptr;
            }
            if (buf_[i]) {
                ggml_backend_buffer_free(buf_[i]);
                buf_[i] = nullptr;
            }
            base_[i] = nullptr;
            used_[i] = 0;
            in_flight_[i] = false;
            pending_[i].clear();
        }
        backend_ = nullptr;
        device_buft_ = nullptr;
        staged_ = false;
        cur_ = 0;
        need_final_sync_ = false;
    }

    // RAII guard pairing begin(backend) with finish(), so early returns and
    // exceptions (e.g. std::bad_alloc) can never leave GPU DMA in flight.
    class Scope {
    public:
        Scope(PinnedTransfer & transfer, ggml_backend_t backend)
            : transfer_(&transfer) {
            transfer_->begin(backend);
        }
        Scope(const Scope &) = delete;
        Scope & operator=(const Scope &) = delete;
        ~Scope() { finish(); }

        void d2h(const ggml_tensor * tensor, size_t dev_offset,
                 void * host_dst, size_t bytes) {
            transfer_->d2h(tensor, dev_offset, host_dst, bytes);
        }
        void h2d(ggml_tensor * tensor, size_t dev_offset,
                 const void * host_src, size_t bytes) {
            transfer_->h2d(tensor, dev_offset, host_src, bytes);
        }
        void finish() {
            if (!finished_) {
                finished_ = true;
                transfer_->finish();
            }
        }

    private:
        PinnedTransfer * transfer_;
        bool finished_ = false;
    };

    Scope scope(ggml_backend_t backend) { return Scope(*this, backend); }

    // Prepare for a sequence of d2h() OR h2d() transfers on `backend`.
    void begin(ggml_backend_t backend) {
        if (!backend) {
            release();
            return;
        }
        if (backend != backend_) {
            release();
            backend_ = backend;
            device_buft_ = ggml_backend_get_default_buffer_type(backend);
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_buffer_type_t host_buft =
                dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (host_buft && device_buft_ && !ggml_backend_buft_is_host(device_buft_)) {
                bool ok = true;
                for (int i = 0; i < 2; ++i) {
                    buf_[i] = ggml_backend_buft_alloc_buffer(host_buft, kSlotBytes);
                    if (!buf_[i]) { ok = false; break; }
                    base_[i] = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buf_[i]));
                    if (!base_[i]) { ok = false; break; }
                }
                if (ok) {
                    staged_ = true;
                    ggml_backend_dev_props props{};
                    ggml_backend_dev_get_props(dev, &props);
                    if (props.caps.events) {
                        ev_[0] = ggml_backend_event_new(dev);
                        ev_[1] = ggml_backend_event_new(dev);
                        if (!ev_[0] || !ev_[1]) {
                            if (ev_[0]) { ggml_backend_event_free(ev_[0]); ev_[0] = nullptr; }
                            if (ev_[1]) { ggml_backend_event_free(ev_[1]); ev_[1] = nullptr; }
                        }
                    }
                } else {
                    for (int i = 0; i < 2; ++i) {
                        if (buf_[i]) { ggml_backend_buffer_free(buf_[i]); buf_[i] = nullptr; }
                        base_[i] = nullptr;
                    }
                    staged_ = false;
                }
            }
        }
        cur_ = 0;
        need_final_sync_ = false;
        for (int i = 0; i < 2; ++i) {
            used_[i] = 0;
            in_flight_[i] = false;
            pending_[i].clear();
        }
    }

    // Queue device-to-host copy of `bytes` from `tensor` at `dev_offset` into `host_dst`.
    void d2h(const ggml_tensor * tensor, size_t dev_offset,
             void * host_dst, size_t bytes) {
        if (!tensor || !host_dst || bytes == 0) return;
        const ggml_backend_buffer_t tb = tensor_buf(tensor);
        if (tb && ggml_backend_buffer_is_host(tb) && tensor->data) {
            std::memcpy(host_dst, static_cast<const char *>(tensor->data) + dev_offset, bytes);
            return;
        }
        if (!staged_ || !owned_by_backend(tb)) {
            if (backend_ && owned_by_backend(tb)) {
                ggml_backend_tensor_get_async(backend_, tensor, host_dst, dev_offset, bytes);
                need_final_sync_ = true;
            } else {
                ggml_backend_tensor_get(tensor, host_dst, dev_offset, bytes);
            }
            return;
        }
        auto * dst = static_cast<uint8_t *>(host_dst);
        size_t done = 0;
        while (done < bytes) {
            const size_t chunk = std::min(bytes - done, kSlotBytes);
            const size_t aligned = align64(chunk);
            if (used_[cur_] + aligned > kSlotBytes) {
                rotate();
            }
            const size_t off = used_[cur_];
            // Ensure metadata capacity before issuing GPU DMA so a bad_alloc
            // cannot abandon an in-flight copy into base_[cur_], while keeping
            // geometric growth across many small slices.
            if (pending_[cur_].size() == pending_[cur_].capacity()) {
                pending_[cur_].reserve(std::max<size_t>(64, pending_[cur_].capacity() * 2));
            }
            ggml_backend_tensor_get_async(
                backend_, tensor, base_[cur_] + off, dev_offset + done, chunk);
            pending_[cur_].push_back({dst + done, base_[cur_] + off, chunk});
            used_[cur_] = off + aligned;
            done += chunk;
        }
    }

    // Queue host-to-device copy of `bytes` from `host_src` into `tensor` at `dev_offset`.
    void h2d(ggml_tensor * tensor, size_t dev_offset,
             const void * host_src, size_t bytes) {
        if (!tensor || !host_src || bytes == 0) return;
        const ggml_backend_buffer_t tb = tensor_buf(tensor);
        if (tb && ggml_backend_buffer_is_host(tb) && tensor->data) {
            std::memcpy(static_cast<char *>(tensor->data) + dev_offset, host_src, bytes);
            return;
        }
        if (!staged_ || !owned_by_backend(tb)) {
            if (backend_ && owned_by_backend(tb)) {
                ggml_backend_tensor_set_async(backend_, tensor, host_src, dev_offset, bytes);
                need_final_sync_ = true;
            } else {
                ggml_backend_tensor_set(tensor, host_src, dev_offset, bytes);
            }
            return;
        }
        const auto * src = static_cast<const uint8_t *>(host_src);
        size_t done = 0;
        while (done < bytes) {
            const size_t chunk = std::min(bytes - done, kSlotBytes);
            const size_t aligned = align64(chunk);
            if (used_[cur_] + aligned > kSlotBytes) {
                rotate();
            }
            const size_t off = used_[cur_];
            std::memcpy(base_[cur_] + off, src + done, chunk);
            ggml_backend_tensor_set_async(
                backend_, tensor, base_[cur_] + off, dev_offset + done, chunk);
            used_[cur_] = off + aligned;
            done += chunk;
        }
    }

    // Complete all queued transfers and drain any remaining pinned D2H slices.
    void finish() {
        const bool any_d2h = !pending_[0].empty() || !pending_[1].empty();
        const bool any_h2d = used_[0] > 0 || used_[1] > 0 || in_flight_[0] || in_flight_[1];
        if (backend_ && (need_final_sync_ || any_d2h || any_h2d)) {
            ggml_backend_synchronize(backend_);
        }
        // Drain older slot first, then current slot.
        const int order[2] = {cur_ ^ 1, cur_};
        for (int idx : order) {
            for (const PendingD2H & p : pending_[idx]) {
                std::memcpy(p.dst, p.src, p.bytes);
            }
            pending_[idx].clear();
            used_[idx] = 0;
            in_flight_[idx] = false;
        }
        need_final_sync_ = false;
    }

private:
    struct PendingD2H {
        uint8_t * dst;
        const uint8_t * src;
        size_t bytes;
    };

    static size_t align64(size_t n) { return (n + 63) & ~size_t(63); }

    static ggml_backend_buffer_t tensor_buf(const ggml_tensor * t) {
        return t->view_src ? t->view_src->buffer : t->buffer;
    }

    bool owned_by_backend(ggml_backend_buffer_t tb) const {
        return tb && device_buft_ && !ggml_backend_buffer_is_host(tb) &&
               ggml_backend_buffer_get_type(tb) == device_buft_;
    }

    // Close the current slot behind an event and switch to the other one, first
    // waiting for its transfers and copying out its D2H slices.
    void rotate() {
        if (ev_[cur_]) {
            ggml_backend_event_record(ev_[cur_], backend_);
        }
        in_flight_[cur_] = true;
        cur_ ^= 1;
        if (in_flight_[cur_]) {
            if (ev_[cur_]) {
                ggml_backend_event_synchronize(ev_[cur_]);
            } else {
                ggml_backend_synchronize(backend_);
            }
            for (const PendingD2H & p : pending_[cur_]) {
                std::memcpy(p.dst, p.src, p.bytes);
            }
            pending_[cur_].clear();
            in_flight_[cur_] = false;
        }
        used_[cur_] = 0;
    }

    void take_from(PinnedTransfer & o) {
        for (int i = 0; i < 2; ++i) {
            buf_[i] = o.buf_[i]; o.buf_[i] = nullptr;
            base_[i] = o.base_[i]; o.base_[i] = nullptr;
            ev_[i] = o.ev_[i]; o.ev_[i] = nullptr;
            used_[i] = o.used_[i]; o.used_[i] = 0;
            in_flight_[i] = o.in_flight_[i]; o.in_flight_[i] = false;
            pending_[i] = std::move(o.pending_[i]);
        }
        backend_ = o.backend_; o.backend_ = nullptr;
        device_buft_ = o.device_buft_; o.device_buft_ = nullptr;
        staged_ = o.staged_; o.staged_ = false;
        cur_ = o.cur_; o.cur_ = 0;
        need_final_sync_ = o.need_final_sync_; o.need_final_sync_ = false;
    }

    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_type_t device_buft_ = nullptr;
    ggml_backend_buffer_t buf_[2] = {nullptr, nullptr};
    uint8_t * base_[2] = {nullptr, nullptr};
    ggml_backend_event_t ev_[2] = {nullptr, nullptr};
    size_t used_[2] = {0, 0};
    bool in_flight_[2] = {false, false};
    std::vector<PendingD2H> pending_[2];
    bool staged_ = false;
    int cur_ = 0;
    bool need_final_sync_ = false;
};

}  // namespace luce::common
