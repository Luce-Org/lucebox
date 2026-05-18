// Gemma4Backend implementation.
//
// Uses gemma4_step() for forward pass (currently stubbed).
// Structure mirrors Qwen3Backend: prefill in chunks, autoregressive decode,
// KV cache with layer sharing, snapshot/restore.

#include "gemma4_backend.h"
#include "dflash27b.h"
#include "common/sampler.h"
#include "common/io_utils.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>

namespace dflash27b {

// ── Ctor / dtor ────────────────────────────────────────────────────────

Gemma4Backend::Gemma4Backend(const Gemma4BackendConfig & cfg)
    : cfg_(cfg) {}

Gemma4Backend::~Gemma4Backend() { shutdown(); }

bool Gemma4Backend::init() {
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[gemma4] CUDA backend init failed (gpu=%d)\n", cfg_.device.gpu);
        return false;
    }

    if (!load_gemma4_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[gemma4] GGUF load failed: %s\n",
                     dflash27b_last_error());
        return false;
    }

    if (!create_gemma4_cache(backend_, w_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[gemma4] cache alloc failed\n");
        return false;
    }

    std::printf("[gemma4] init ok: %d layers, embd=%d, vocab=%d, max_ctx=%d\n",
                w_.n_layer, w_.n_embd, w_.n_vocab, cfg_.device.max_ctx);
    std::fflush(stdout);
    return true;
}

void Gemma4Backend::print_ready_banner() const {
    std::printf("[gemma4-daemon] READY (layers=%d, embd=%d, experts=%d/%d, "
                "swa=%d, ctx=%d)\n",
                w_.n_layer, w_.n_embd, w_.n_expert_used, w_.n_expert,
                w_.sliding_window, cfg_.device.max_ctx);
    std::fflush(stdout);
}

// ── Park / Unpark ──────────────────────────────────────────────────────

bool Gemma4Backend::park(const std::string & what) {
    (void)what;
    parked_ = true;
    std::printf("[gemma4] parked\n"); std::fflush(stdout);
    return true;
}

bool Gemma4Backend::unpark(const std::string & what) {
    (void)what;
    parked_ = false;
    std::printf("[gemma4] unparked\n"); std::fflush(stdout);
    return true;
}

// ── Prefill ────────────────────────────────────────────────────────────

int Gemma4Backend::do_prefill(const std::vector<int32_t> & tokens,
                               const DaemonIO & io) {
    (void)io;
    const int n = (int)tokens.size();
    const int hidden = w_.n_embd;
    const int chunk = cfg_.chunk;

    std::vector<float> embed(chunk * hidden);
    std::vector<float> logits;

    int pos = 0;
    while (pos < n) {
        int len = std::min(chunk, n - pos);

        // Embed tokens using CPU embedder
        w_.embedder.embed(tokens.data() + pos, len, embed.data());

        // Gemma4 scales embeddings by sqrt(n_embd)
        float scale = std::sqrt((float)hidden);
        for (int i = 0; i < len * hidden; ++i) embed[i] *= scale;

        if (!gemma4_step(backend_, w_, cache_, embed.data(), len, pos, logits)) {
            std::fprintf(stderr, "[gemma4] prefill step failed at pos=%d\n", pos);
            return -1;
        }

        pos += len;
        cache_.cur_pos = pos;
    }

    return pos;
}

// ── Decode ─────────────────────────────────────────────────────────────

bool Gemma4Backend::do_decode(int committed, int n_gen,
                               std::vector<int32_t> & out_tokens,
                               const DaemonIO & io) {
    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> embed_buf(hidden);
    std::vector<float> logits;

    for (int i = 0; i < n_gen; ++i) {
        int32_t tok = out_tokens.back();

        // Embed single token
        w_.embedder.embed(&tok, 1, embed_buf.data());
        float scale = std::sqrt((float)hidden);
        for (int j = 0; j < hidden; ++j) embed_buf[j] *= scale;

        if (!gemma4_step(backend_, w_, cache_, embed_buf.data(), 1, committed, logits)) {
            return false;
        }

        // Sample
        int32_t next;
        if (sampler_.temp > 0) {
            next = sample_logits(logits.data(), vocab, sampler_,
                                 out_tokens, sampler_rng_);
        } else {
            next = 0;
            float best = logits[0];
            for (int j = 1; j < vocab; ++j) {
                if (logits[j] > best) { best = logits[j]; next = j; }
            }
        }

        out_tokens.push_back(next);
        io.emit(next);
        committed++;
        cache_.cur_pos = committed;
        if (io.cancelled) break;

        // Check EOS
        if (next == w_.eos_id || next == w_.eos_chat_id) break;
    }

    return true;
}

// ── Generate ───────────────────────────────────────────────────────────

GenerateResult Gemma4Backend::generate(const GenerateRequest & req,
                                        const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    cache_.cur_pos = 0;

    const int committed = do_prefill(req.prompt, out_io);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }

    if (req.n_gen > 0) {
        const int hidden = w_.n_embd;
        const int vocab  = w_.n_vocab;
        std::vector<float> logits;

        // Re-step last token to get logits
        int32_t last_tok = req.prompt.back();
        std::vector<float> embed_buf(hidden);
        w_.embedder.embed(&last_tok, 1, embed_buf.data());
        float scale = std::sqrt((float)hidden);
        for (int j = 0; j < hidden; ++j) embed_buf[j] *= scale;

        if (!gemma4_step(backend_, w_, cache_, embed_buf.data(), 1,
                         committed - 1, logits)) {
            result.error = "first logits";
            return result;
        }

        // Sample first token
        int32_t first;
        if (sampler_.temp > 0) {
            first = sample_logits(logits.data(), vocab, sampler_,
                                   result.tokens, sampler_rng_);
        } else {
            first = 0;
            float best = logits[0];
            for (int j = 1; j < vocab; ++j) {
                if (logits[j] > best) { best = logits[j]; first = j; }
            }
        }
        result.tokens.push_back(first);
        out_io.emit(first);
        if (out_io.cancelled) {
            out_io.emit(-1);
            result.ok = true;
            return result;
        }

        if (first == w_.eos_id || first == w_.eos_chat_id) {
            out_io.emit(-1);
            result.ok = true;
            return result;
        }

        if (req.n_gen > 1) {
            if (!do_decode(committed, req.n_gen - 1, result.tokens, out_io)) {
                result.error = "decode";
                return result;
            }
        }
    }

    out_io.emit(-1);
    result.ok = true;
    return result;
}

// ── Restore + Generate ─────────────────────────────────────────────────

GenerateResult Gemma4Backend::restore_and_generate(int slot,
                                                     const GenerateRequest & req,
                                                     const DaemonIO & io) {
    GenerateResult result;
    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].ctx) {
        result.error = "bad slot";
        io.emit(-1);
        return result;
    }

    const auto & snap = snapshots_[slot];
    // Only copy layers that have real KV
    for (int il = 0; il < cache_.n_layer; ++il) {
        if (cache_.k[il] && snap.k_snap[il]) {
            ggml_backend_tensor_copy(snap.k_snap[il], cache_.k[il]);
            ggml_backend_tensor_copy(snap.v_snap[il], cache_.v[il]);
        }
    }
    cache_.cur_pos = snap.cur_pos;

    return generate(req, io);
}

// ── Snapshots ──────────────────────────────────────────────────────────

bool Gemma4Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    snapshot_free(slot);

    auto & snap = snapshots_[slot];
    const int n_layer = cache_.n_layer;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 4) + 4096;
    ip.no_alloc = true;
    snap.ctx = ggml_init(ip);
    if (!snap.ctx) return false;

    snap.k_snap.resize(n_layer, nullptr);
    snap.v_snap.resize(n_layer, nullptr);
    for (int il = 0; il < n_layer; ++il) {
        if (cache_.k[il]) {
            snap.k_snap[il] = ggml_dup_tensor(snap.ctx, cache_.k[il]);
            snap.v_snap[il] = ggml_dup_tensor(snap.ctx, cache_.v[il]);
        }
    }

    snap.buf = ggml_backend_alloc_ctx_tensors(snap.ctx, backend_);
    if (!snap.buf) {
        ggml_free(snap.ctx); snap.ctx = nullptr;
        return false;
    }

    for (int il = 0; il < n_layer; ++il) {
        if (cache_.k[il]) {
            ggml_backend_tensor_copy(cache_.k[il], snap.k_snap[il]);
            ggml_backend_tensor_copy(cache_.v[il], snap.v_snap[il]);
        }
    }
    snap.cur_pos = cache_.cur_pos;

    std::printf("[gemma4] snapshot saved slot=%d pos=%d\n", slot, snap.cur_pos);
    std::fflush(stdout);
    return true;
}

void Gemma4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_gemma4_snapshot(snapshots_[slot]);
}

bool Gemma4Backend::snapshot_used(int slot) const {
    return slot >= 0 && slot < PREFIX_SLOTS && snapshots_[slot].ctx != nullptr;
}

int Gemma4Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].ctx) return 0;
    return snapshots_[slot].cur_pos;
}

// ── Compress / drafter ─────────────────────────────────────────────────

bool Gemma4Backend::handle_compress(const std::string & line,
                                     const DaemonIO & io) {
    (void)line; (void)io;
    // Gemma4 doesn't use pflash drafter for compression (yet).
    std::printf("[gemma4] compress: not supported\n");
    std::fflush(stdout);
    return true;
}

void Gemma4Backend::free_drafter() {
    // No drafter to free.
}

bool Gemma4Backend::try_handle_command(const std::string & line,
                                        const DaemonIO & io) {
    (void)line; (void)io;
    return false;  // no arch-specific commands
}

// ── Shutdown ───────────────────────────────────────────────────────────

void Gemma4Backend::shutdown() {
    for (int i = 0; i < PREFIX_SLOTS; ++i) snapshot_free(i);
    free_gemma4_cache(cache_);
    free_gemma4_weights(w_);
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
    std::printf("[gemma4] shutdown\n"); std::fflush(stdout);
}

}  // namespace dflash27b
