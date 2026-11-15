// A snapshot payload in host memory that its holder owns: what the disk
// session cache writes and reads (server/kv/disk_store.h). Backends hand one
// over from a snapshot slot (ModelBackend::snapshot_detach) and concurrent
// engines from their own checkpoints (SeqEngine::prefix_store_payloads).

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace luce::common {

// Owns its tensor context and buffer and frees both when destroyed. Move-only.
// The tensors may also live in further buffers it owns (an engine that hands
// over its checkpoint buffer without a copy adds its metadata tensors in a
// small one); only a snapshot in one buffer can be released to a new owner.
class OwnedSnapshot {
public:
    OwnedSnapshot() = default;
    OwnedSnapshot(ggml_context * ctx, ggml_backend_buffer_t buf,
                  int cur_pos, int32_t last_tok)
        : ctx_(ctx), buf_(buf), cur_pos_(cur_pos), last_tok_(last_tok) {}
    ~OwnedSnapshot() { reset(); }
    OwnedSnapshot(const OwnedSnapshot &) = delete;
    OwnedSnapshot & operator=(const OwnedSnapshot &) = delete;
    OwnedSnapshot(OwnedSnapshot && o) noexcept { *this = std::move(o); }
    OwnedSnapshot & operator=(OwnedSnapshot && o) noexcept {
        if (this != &o) {
            reset();
            ctx_ = o.ctx_; buf_ = o.buf_;
            cur_pos_ = o.cur_pos_; last_tok_ = o.last_tok_;
            extra_ = std::move(o.extra_);
            o.ctx_ = nullptr; o.buf_ = nullptr;
            o.extra_.clear();
        }
        return *this;
    }
    void reset() {
        for (ggml_backend_buffer_t b : extra_) ggml_backend_buffer_free(b);
        extra_.clear();
        if (buf_) ggml_backend_buffer_free(buf_);
        if (ctx_) ggml_free(ctx_);
        ctx_ = nullptr; buf_ = nullptr;
    }
    // A further buffer some of the tensors live in, freed with the snapshot.
    void own_buffer(ggml_backend_buffer_t buf) { if (buf) extra_.push_back(buf); }
    // One buffer holds every tensor: release() can hand it over whole.
    bool single_buffer() const { return extra_.empty(); }
    // Hand ctx and buf to a new owner (e.g. ModelBackend::snapshot_adopt).
    void release() { ctx_ = nullptr; buf_ = nullptr; }
    bool empty() const { return ctx_ == nullptr; }
    ggml_context * ctx() const { return ctx_; }
    ggml_backend_buffer_t buf() const { return buf_; }
    int cur_pos() const { return cur_pos_; }
    int32_t last_tok() const { return last_tok_; }
    uint64_t bytes() const {
        uint64_t n = 0;
        for (ggml_tensor * t = ctx_ ? ggml_get_first_tensor(ctx_) : nullptr; t;
             t = ggml_get_next_tensor(ctx_, t)) n += ggml_nbytes(t);
        return n;
    }

    // Copy every tensor of `ctx` into a fresh host (CPU) buffer.
    static OwnedSnapshot clone_to_host(ggml_context * ctx, int cur_pos,
                                       int32_t last_tok) {
        if (!ctx) return {};
        int n = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
             t = ggml_get_next_tensor(ctx, t)) ++n;
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (size_t)(n + 4) + 4096;
        ip.no_alloc = true;
        ggml_context * out = ggml_init(ip);
        if (!out) return {};
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
             t = ggml_get_next_tensor(ctx, t)) {
            ggml_tensor * c = ggml_new_tensor(out, t->type, GGML_MAX_DIMS, t->ne);
            ggml_set_name(c, t->name);
        }
        ggml_backend_buffer_t buf =
            ggml_backend_alloc_ctx_tensors_from_buft(out, ggml_backend_cpu_buffer_type());
        if (!buf) { ggml_free(out); return {}; }
        for (ggml_tensor * t = ggml_get_first_tensor(ctx),
                         * c = ggml_get_first_tensor(out);
             t && c; t = ggml_get_next_tensor(ctx, t), c = ggml_get_next_tensor(out, c)) {
            ggml_backend_tensor_get(t, c->data, 0, ggml_nbytes(t));
        }
        return OwnedSnapshot(out, buf, cur_pos, last_tok);
    }

private:
    ggml_context *        ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    int                   cur_pos_ = 0;
    int32_t               last_tok_ = -1;
    std::vector<ggml_backend_buffer_t> extra_;
};

}  // namespace luce::common
