// Qwen3Backend implementation.
//
// Step forward builds a ggml graph per call covering all 28 layers.
// K/V cache is persistent (allocated once at init). Each step writes
// [kv_start, kv_start+n_tokens) into the cache and reads [0, kv_start+n_tokens).
// After all layers, out_norm + lm_head produces logits for the last token.

#include "qwen3_backend.h"
#include "qwen3_drafter.h"
#include "dflash27b.h"
#include "common/sampler.h"
#include "common/io_utils.h"

#include "ggml-cuda.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <sstream>

namespace dflash::common {

// ── Cache management ───────────────────────────────────────────────────

bool create_qwen3_cache(ggml_backend_t backend, const Qwen3DrafterWeights & w,
                          int max_ctx, Qwen3Cache & out) {
    const int n_layer = w.n_layer;
    const int D       = w.head_dim;
    const int Hk      = w.n_head_kv;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 4) + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    const ggml_type half_type = w.compute_type;

    out.k.resize(n_layer);
    out.v.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        out.k[il] = ggml_new_tensor_3d(out.ctx, half_type, D, max_ctx, Hk);
        out.v[il] = ggml_new_tensor_3d(out.ctx, half_type, D, max_ctx, Hk);
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    out.cur_pos  = 0;
    out.max_ctx  = max_ctx;
    out.n_layer  = n_layer;
    return true;
}

void free_qwen3_cache(Qwen3Cache & c) {
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.k.clear();
    c.v.clear();
    c.cur_pos = 0;
}

void free_qwen3_snapshot(Qwen3Snapshot & s) {
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    s.k_snap.clear();
    s.v_snap.clear();
    s.cur_pos = 0;
}

// ── Construction / destruction ─────────────────────────────────────────

Qwen3Backend::Qwen3Backend(const Qwen3BackendConfig & cfg) : cfg_(cfg) {}

Qwen3Backend::~Qwen3Backend() { shutdown(); }

// ── init ───────────────────────────────────────────────────────────────

bool Qwen3Backend::init() {
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[qwen3] CUDA init failed for GPU %d\n", cfg_.device.gpu);
        return false;
    }

    if (!load_qwen3_drafter_model(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[qwen3] model load failed: %s\n", dflash27b_last_error());
        return false;
    }
    std::printf("[qwen3] loaded %s (%d layers, hidden=%d, vocab=%d)\n",
                cfg_.model_path, w_.n_layer, w_.n_embd, w_.n_vocab);

    if (!create_qwen3_cache(backend_, w_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[qwen3] cache creation failed\n");
        return false;
    }
    std::printf("[qwen3] cache allocated (max_ctx=%d)\n", cfg_.device.max_ctx);

    // Init CPU embedder
    if (!w_.tok_embd) {
        std::fprintf(stderr, "[qwen3] no token embedding tensor\n");
        return false;
    }

    std::fflush(stdout);
    return true;
}

// ── Ready banner ───────────────────────────────────────────────────────

void Qwen3Backend::print_ready_banner() const {
    std::printf("[qwen3-daemon] ready (layers=%d hidden=%d vocab=%d max_ctx=%d)\n",
                w_.n_layer, w_.n_embd, w_.n_vocab, cfg_.device.max_ctx);
    std::fflush(stdout);
}

// ── Park / unpark ──────────────────────────────────────────────────────

bool Qwen3Backend::park(const std::string & what) {
    if (what == "target" || what == "all") {
        if (!parked_) {
            // Free weights buffer to reclaim VRAM (keep cache)
            if (w_.buf) {
                ggml_backend_buffer_free(w_.buf);
                w_.buf = nullptr;
            }
            parked_ = true;
            std::printf("[qwen3] target parked\n");
            std::fflush(stdout);
        }
        return true;
    }
    return false;
}

bool Qwen3Backend::unpark(const std::string & what) {
    if (what == "target" || what == "all") {
        if (parked_) {
            // Reload weights
            Qwen3DrafterWeights w_new;
            if (!load_qwen3_drafter_model(cfg_.model_path, backend_, w_new)) {
                std::fprintf(stderr, "[qwen3] unpark reload failed\n");
                return false;
            }
            w_ = w_new;
            parked_ = false;
            std::printf("[qwen3] target unparked\n");
            std::fflush(stdout);
        }
        return true;
    }
    return false;
}

// ── Step forward ───────────────────────────────────────────────────────
// Builds a ggml graph for n_tokens, appends K/V to cache at kv_start,
// returns logits for the last token.

bool Qwen3Backend::do_step(const float * embed, int n_tokens, int kv_start,
                            std::vector<float> & out_logits) {
    const int hidden = w_.n_embd;
    const int H      = w_.n_head;
    const int Hk     = w_.n_head_kv;
    const int D      = w_.head_dim;
    const int vocab  = w_.n_vocab;
    const float eps  = 1e-6f;
    const int kv_len = kv_start + n_tokens;

    const ggml_type half_type = w_.compute_type;

    // Allocate graph context
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 1536
                  + ggml_graph_overhead_custom(4096, false)
                  + 256 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    // Input: hidden state [hidden, n_tokens] f32
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_set_name(inp, "inp_embed");
    ggml_set_input(inp);
    ggml_tensor * cur = inp;

    // Positions [n_tokens] i32
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Causal attention mask [kv_len, n_tokens] f16
    // For decode (n_tokens=1), mask is nullptr (single token attends to all cached)
    ggml_tensor * attn_mask = nullptr;
    if (n_tokens > 1) {
        attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, n_tokens);
        ggml_set_name(attn_mask, "attn_mask");
        ggml_set_input(attn_mask);
    }

    // Per-layer forward
    for (int il = 0; il < w_.n_layer; ++il) {
        const auto & L = w_.layers[il];

        // Pre-attention norm
        ggml_tensor * normed = ggml_rms_norm(ctx, cur, eps);
        normed = ggml_mul(ctx, normed, L.attn_norm);

        // Q/K/V projections
        ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, normed);  // [H*D, n_tokens]
        ggml_tensor * K = ggml_mul_mat(ctx, L.wk, normed);  // [Hk*D, n_tokens]
        ggml_tensor * V = ggml_mul_mat(ctx, L.wv, normed);  // [Hk*D, n_tokens]

        // Reshape to [D, heads, n_tokens]
        Q = ggml_reshape_3d(ctx, Q, D, H, n_tokens);
        K = ggml_reshape_3d(ctx, K, D, Hk, n_tokens);
        V = ggml_reshape_3d(ctx, V, D, Hk, n_tokens);

        // Q/K norms
        Q = ggml_rms_norm(ctx, Q, eps);
        Q = ggml_mul(ctx, Q, L.q_norm);
        K = ggml_rms_norm(ctx, K, eps);
        K = ggml_mul(ctx, K, L.k_norm);

        // RoPE
        Q = ggml_rope_ext(ctx, Q, positions, nullptr,
                          D, GGML_ROPE_TYPE_NEOX, 0,
                          w_.rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, positions, nullptr,
                          D, GGML_ROPE_TYPE_NEOX, 0,
                          w_.rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);

        // Cast to half for cache
        ggml_tensor * K_half = ggml_cast(ctx, K, half_type);
        ggml_tensor * V_half = ggml_cast(ctx, V, half_type);

        // Permute K/V from [D, Hk, n_tokens] to [D, n_tokens, Hk] for cache write
        ggml_tensor * Kt = ggml_permute(ctx, K_half, 0, 2, 1, 3);
        ggml_tensor * Vt = ggml_permute(ctx, V_half, 0, 2, 1, 3);

        // Write K/V into persistent cache at [kv_start, kv_start+n_tokens)
        // Cache layout: [D, max_ctx, Hk]
        ggml_tensor * k_dst = ggml_view_3d(ctx, cache_.k[il],
                                            D, n_tokens, Hk,
                                            cache_.k[il]->nb[1], cache_.k[il]->nb[2],
                                            cache_.k[il]->nb[1] * (size_t)kv_start);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kt, k_dst));

        ggml_tensor * v_dst = ggml_view_3d(ctx, cache_.v[il],
                                            D, n_tokens, Hk,
                                            cache_.v[il]->nb[1], cache_.v[il]->nb[2],
                                            cache_.v[il]->nb[1] * (size_t)kv_start);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vt, v_dst));

        // Attention: read full K/V from cache [D, kv_len, Hk]
        ggml_tensor * K_full = ggml_view_3d(ctx, cache_.k[il],
                                             D, kv_len, Hk,
                                             cache_.k[il]->nb[1], cache_.k[il]->nb[2], 0);
        ggml_tensor * V_full = ggml_view_3d(ctx, cache_.v[il],
                                             D, kv_len, Hk,
                                             cache_.v[il]->nb[1], cache_.v[il]->nb[2], 0);

        // Permute Q from [D, H, n_tokens] to [D, n_tokens, H] for flash_attn_ext
        ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);
        Qfa = ggml_cont(ctx, Qfa);

        // flash_attn_ext: Q[D, n_tokens, H], K[D, kv_len, Hk], V[D, kv_len, Hk]
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, K_full, V_full,
                                                  attn_mask,
                                                  1.0f / std::sqrt((float)D),
                                                  0.0f, 0.0f);
        // attn output: [D, H, n_tokens, 1] f32 (permuted internally)
        // Reshape to [H*D, n_tokens] for output projection
        ggml_tensor * attn_2d = ggml_reshape_2d(ctx, attn, H * D, n_tokens);

        // Output projection + residual
        ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn_2d);
        cur = ggml_add(ctx, cur, attn_out);

        // FFN: pre-norm → gate·up → down → residual
        ggml_tensor * ffn_in = ggml_rms_norm(ctx, cur, eps);
        ffn_in = ggml_mul(ctx, ffn_in, L.ffn_norm);

        ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate, ffn_in);
        gate = ggml_silu(ctx, gate);
        ggml_tensor * up = ggml_mul_mat(ctx, L.ffn_up, ffn_in);
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, L.ffn_down, ggml_mul(ctx, gate, up));

        cur = ggml_add(ctx, cur, ffn_out);
    }

    // Output norm + lm_head (last token only)
    ggml_tensor * last_hidden;
    if (n_tokens > 1) {
        last_hidden = ggml_view_2d(ctx, cur, hidden, 1,
                                    cur->nb[1],
                                    (size_t)(n_tokens - 1) * cur->nb[1]);
    } else {
        last_hidden = cur;
    }
    ggml_tensor * normed_out = ggml_rms_norm(ctx, last_hidden, eps);
    normed_out = ggml_mul(ctx, normed_out, w_.out_norm);
    ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, normed_out);
    ggml_set_output(logits);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);

    // Allocate and compute
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend_));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "[qwen3] graph alloc failed (n_tokens=%d, kv_start=%d, kv_len=%d)\n",
                     n_tokens, kv_start, kv_len);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // Set inputs
    ggml_backend_tensor_set(inp, embed, 0,
                            sizeof(float) * (size_t)hidden * n_tokens);
    {
        std::vector<int32_t> pos(n_tokens);
        for (int i = 0; i < n_tokens; ++i) pos[i] = kv_start + i;
        ggml_backend_tensor_set(positions, pos.data(), 0,
                                sizeof(int32_t) * n_tokens);
    }

    // Build causal mask: position i can attend to [0, kv_start+i]
    if (attn_mask) {
        std::vector<ggml_fp16_t> mask_data((size_t)kv_len * n_tokens);
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int row = 0; row < n_tokens; ++row) {
            const int last_visible = kv_start + row;
            for (int col = 0; col < kv_len; ++col) {
                mask_data[(size_t)row * kv_len + col] =
                    (col <= last_visible) ? zero_h : neg_inf_h;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                                sizeof(ggml_fp16_t) * mask_data.size());
    }

    auto st = ggml_backend_graph_compute(backend_, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen3] graph compute failed (status=%d, n_tokens=%d, kv_start=%d, kv_len=%d)\n",
                     (int)st, n_tokens, kv_start, kv_len);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // Read logits
    out_logits.resize(vocab);
    ggml_backend_tensor_get(logits, out_logits.data(), 0,
                            sizeof(float) * vocab);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return true;
}

// ── Prefill ────────────────────────────────────────────────────────────

int Qwen3Backend::do_prefill(const std::vector<int32_t> & tokens,
                              const DaemonIO & io, int kv_offset) {
    const int hidden = w_.n_embd;
    const int total  = (int)tokens.size();
    const int chunk  = std::max(1, cfg_.chunk);
    int committed = 0;

    // Embed on CPU using get_rows from tok_embd
    // We need to embed chunks and feed them to do_step
    std::vector<float> embed_buf((size_t)chunk * hidden);

    for (int start = 0; start < total; start += chunk) {
        const int n = std::min(chunk, total - start);

        // CPU embedding: read rows from tok_embd (which is on GPU)
        // Build a small graph: get_rows(tok_embd, ids) → embed_buf
        {
            ggml_init_params ip{};
            ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
            ip.no_alloc = true;
            ggml_context * ectx = ggml_init(ip);
            ggml_tensor * ids = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, n);
            ggml_set_input(ids);
            ggml_tensor * emb = ggml_get_rows(ectx, w_.tok_embd, ids);
            ggml_tensor * out = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, hidden, n);
            ggml_tensor * cpy = ggml_cpy(ectx, emb, out);
            ggml_set_output(cpy);
            ggml_cgraph * gf = ggml_new_graph(ectx);
            ggml_build_forward_expand(gf, cpy);

            ggml_gallocr_t galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_));
            if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                ggml_gallocr_free(galloc);
                ggml_free(ectx);
                return -1;
            }
            ggml_backend_tensor_set(ids, tokens.data() + start, 0,
                                    sizeof(int32_t) * n);
            ggml_backend_graph_compute(backend_, gf);
            ggml_backend_tensor_get(cpy, embed_buf.data(), 0,
                                    sizeof(float) * (size_t)hidden * n);
            ggml_gallocr_free(galloc);
            ggml_free(ectx);
        }

        std::vector<float> logits;
        if (!do_step(embed_buf.data(), n, kv_offset + start, logits)) {
            return -1;
        }
        committed = kv_offset + start + n;
        cache_.cur_pos = committed;
        last_logits_ = std::move(logits);
    }

    return committed;
}

// ── Decode ─────────────────────────────────────────────────────────────

bool Qwen3Backend::do_decode(int committed, int n_gen,
                              std::vector<int32_t> & out_tokens,
                              const DaemonIO & io) {
    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> logits;
    std::vector<float> embed_buf(hidden);

    for (int i = 0; i < n_gen; ++i) {
        // Get logits: first iteration uses prefill logits, rest use step output
        if (i == 0) {
            if (last_logits_.empty()) return false;
            logits = std::move(last_logits_);
        }

        // Sample next token
        int32_t next;
        if (sampler_.needs_logit_processing()) {
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
        if (next == 151643 || next == 151645) break;

        // Last iteration — don't need logits for next step
        if (i == n_gen - 1) break;

        // Embed next token and compute logits for following iteration
        {
            ggml_init_params ip{};
            ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
            ip.no_alloc = true;
            ggml_context * ectx = ggml_init(ip);
            ggml_tensor * ids = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, 1);
            ggml_set_input(ids);
            ggml_tensor * emb = ggml_get_rows(ectx, w_.tok_embd, ids);
            ggml_tensor * out = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, hidden, 1);
            ggml_tensor * cpy = ggml_cpy(ectx, emb, out);
            ggml_set_output(cpy);
            ggml_cgraph * gf = ggml_new_graph(ectx);
            ggml_build_forward_expand(gf, cpy);
            ggml_gallocr_t galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_));
            if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                ggml_gallocr_free(galloc);
                ggml_free(ectx);
                return false;
            }
            ggml_backend_tensor_set(ids, &next, 0, sizeof(int32_t));
            ggml_backend_graph_compute(backend_, gf);
            ggml_backend_tensor_get(cpy, embed_buf.data(), 0,
                                    sizeof(float) * hidden);
            ggml_gallocr_free(galloc);
            ggml_free(ectx);
        }

        if (!do_step(embed_buf.data(), 1, committed, logits)) {
            return false;
        }
    }

    return true;
}

// ── Generate ───────────────────────────────────────────────────────────

GenerateResult Qwen3Backend::generate(const GenerateRequest & req,
                                       const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    cache_.cur_pos = 0;  // reset cache for fresh generation

    // Prefill
    const int committed = do_prefill(req.prompt, out_io);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }

    // Inline snapshot at snap_pos for prefix cache
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= committed) {
        cache_.cur_pos = req.snap_pos;
        if (snapshot_save(req.snap_slot)) {
            std::printf("[snap] inline slot=%d cur_pos=%d\n",
                        req.snap_slot, req.snap_pos);
            std::fflush(stdout);
        }
        cache_.cur_pos = committed;
    }

    // Get first token from prefill logits
    if (req.n_gen > 0) {
        // Re-run last prefill chunk to get logits (TODO: cache these)
        const int hidden = w_.n_embd;
        const int vocab  = w_.n_vocab;
        std::vector<float> logits;

        // Embed last token and step to get logits
        int32_t last_tok = req.prompt.back();
        std::vector<float> embed_buf(hidden);
        {
            ggml_init_params ip{};
            ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
            ip.no_alloc = true;
            ggml_context * ectx = ggml_init(ip);
            ggml_tensor * ids = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, 1);
            ggml_set_input(ids);
            ggml_tensor * emb = ggml_get_rows(ectx, w_.tok_embd, ids);
            ggml_tensor * out = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, hidden, 1);
            ggml_tensor * cpy = ggml_cpy(ectx, emb, out);
            ggml_set_output(cpy);
            ggml_cgraph * gf = ggml_new_graph(ectx);
            ggml_build_forward_expand(gf, cpy);
            ggml_gallocr_t galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_));
            if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                ggml_gallocr_free(galloc); ggml_free(ectx);
                result.error = "embed alloc"; return result;
            }
            ggml_backend_tensor_set(ids, &last_tok, 0, sizeof(int32_t));
            ggml_backend_graph_compute(backend_, gf);
            ggml_backend_tensor_get(cpy, embed_buf.data(), 0, sizeof(float) * hidden);
            ggml_gallocr_free(galloc);
            ggml_free(ectx);
        }

        // Re-step at committed-1 to get logits (KV already written, idempotent)
        if (!do_step(embed_buf.data(), 1, committed - 1, logits)) {
            result.error = "first logits";
            return result;
        }

        // Sample first token
        int32_t first;
        if (sampler_.needs_logit_processing()) {
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

        if (first == 151643 || first == 151645) {
            out_io.emit(-1);
            result.ok = true;
            return result;
        }

        // Continue decode — embed the first sampled token to get logits for next step
        int cur_committed = committed;
        if (req.n_gen > 1) {
            // Embed 'first' token to get logits for the next iteration
            int32_t ft = first;
            {
                ggml_init_params ip2{};
                ip2.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
                ip2.no_alloc = true;
                ggml_context * ectx2 = ggml_init(ip2);
                ggml_tensor * ids2 = ggml_new_tensor_1d(ectx2, GGML_TYPE_I32, 1);
                ggml_set_input(ids2);
                ggml_tensor * emb2 = ggml_get_rows(ectx2, w_.tok_embd, ids2);
                ggml_tensor * out2 = ggml_new_tensor_2d(ectx2, GGML_TYPE_F32, hidden, 1);
                ggml_tensor * cpy2 = ggml_cpy(ectx2, emb2, out2);
                ggml_set_output(cpy2);
                ggml_cgraph * gf2 = ggml_new_graph(ectx2);
                ggml_build_forward_expand(gf2, cpy2);
                ggml_gallocr_t galloc2 = ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(backend_));
                if (!ggml_gallocr_alloc_graph(galloc2, gf2)) {
                    ggml_gallocr_free(galloc2); ggml_free(ectx2);
                    result.error = "embed2 alloc"; return result;
                }
                ggml_backend_tensor_set(ids2, &ft, 0, sizeof(int32_t));
                ggml_backend_graph_compute(backend_, gf2);
                ggml_backend_tensor_get(cpy2, embed_buf.data(), 0, sizeof(float) * hidden);
                ggml_gallocr_free(galloc2);
                ggml_free(ectx2);
            }
            if (!do_step(embed_buf.data(), 1, cur_committed, last_logits_)) {
                result.error = "decode logits";
                return result;
            }
            cur_committed++;
            cache_.cur_pos = cur_committed;

            if (!do_decode(cur_committed, req.n_gen - 1, result.tokens, out_io)) {
                result.error = "decode";
                return result;
            }
        }
    }

    out_io.emit(-1);
    result.ok = true;
    return result;
}

// ── Restore + generate ─────────────────────────────────────────────────

GenerateResult Qwen3Backend::restore_and_generate(int slot,
                                                    const GenerateRequest & req,
                                                    const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].ctx) {
        result.error = "bad slot";
        out_io.emit(-1);
        return result;
    }

    // Restore KV cache from snapshot
    const auto & snap = snapshots_[slot];
    for (int il = 0; il < cache_.n_layer; ++il) {
        ggml_backend_tensor_copy(snap.k_snap[il], cache_.k[il]);
        ggml_backend_tensor_copy(snap.v_snap[il], cache_.v[il]);
    }
    cache_.cur_pos = snap.cur_pos;
    const int prefix_len = snap.cur_pos;

    // Set up sampler
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    // Only prefill the tokens after the restored prefix
    if (prefix_len < (int)req.prompt.size()) {
        std::vector<int32_t> remaining(req.prompt.begin() + prefix_len,
                                        req.prompt.end());
        const int committed = do_prefill(remaining, out_io, prefix_len);
        if (committed < 0) {
            result.error = "prefill after restore";
            return result;
        }
    }

    // Now generate (decode) from here
    const int total_committed = (int)req.prompt.size();
    cache_.cur_pos = total_committed;
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= total_committed) {
        cache_.cur_pos = req.snap_pos;
        if (snapshot_save(req.snap_slot)) {
            std::printf("[snap] inline slot=%d cur_pos=%d\n",
                        req.snap_slot, req.snap_pos);
            std::fflush(stdout);
        }
        cache_.cur_pos = total_committed;
    }

    if (req.n_gen > 0) {
        const int hidden = w_.n_embd;
        const int vocab  = w_.n_vocab;
        std::vector<float> logits;

        // Embed last token and step to get logits
        int32_t last_tok = req.prompt.back();
        std::vector<float> embed_buf(hidden);
        {
            ggml_init_params ip{};
            ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
            ip.no_alloc = true;
            ggml_context * ectx = ggml_init(ip);
            ggml_tensor * ids = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, 1);
            ggml_set_input(ids);
            ggml_tensor * emb = ggml_get_rows(ectx, w_.tok_embd, ids);
            ggml_tensor * out = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, hidden, 1);
            ggml_tensor * cpy = ggml_cpy(ectx, emb, out);
            ggml_set_output(cpy);
            ggml_cgraph * gf = ggml_new_graph(ectx);
            ggml_build_forward_expand(gf, cpy);
            ggml_gallocr_t galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_));
            if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                ggml_gallocr_free(galloc); ggml_free(ectx);
                result.error = "embed alloc"; return result;
            }
            ggml_backend_tensor_set(ids, &last_tok, 0, sizeof(int32_t));
            ggml_backend_graph_compute(backend_, gf);
            ggml_backend_tensor_get(cpy, embed_buf.data(), 0, sizeof(float) * hidden);
            ggml_gallocr_free(galloc);
            ggml_free(ectx);
        }

        // Re-step at last position to get logits
        if (!do_step(embed_buf.data(), 1, total_committed - 1, logits)) {
            result.error = "first logits";
            return result;
        }

        // Sample first token
        int32_t first;
        if (sampler_.needs_logit_processing()) {
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

        if (first == 151643 || first == 151645) {
            out_io.emit(-1);
            result.ok = true;
            return result;
        }

        // Continue decode
        int cur_committed = total_committed;
        if (req.n_gen > 1) {
            int32_t ft = first;
            {
                ggml_init_params ip2{};
                ip2.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
                ip2.no_alloc = true;
                ggml_context * ectx2 = ggml_init(ip2);
                ggml_tensor * ids2 = ggml_new_tensor_1d(ectx2, GGML_TYPE_I32, 1);
                ggml_set_input(ids2);
                ggml_tensor * emb2 = ggml_get_rows(ectx2, w_.tok_embd, ids2);
                ggml_tensor * out2 = ggml_new_tensor_2d(ectx2, GGML_TYPE_F32, hidden, 1);
                ggml_tensor * cpy2 = ggml_cpy(ectx2, emb2, out2);
                ggml_set_output(cpy2);
                ggml_cgraph * gf2 = ggml_new_graph(ectx2);
                ggml_build_forward_expand(gf2, cpy2);
                ggml_gallocr_t galloc2 = ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(backend_));
                if (!ggml_gallocr_alloc_graph(galloc2, gf2)) {
                    ggml_gallocr_free(galloc2); ggml_free(ectx2);
                    result.error = "embed2 alloc"; return result;
                }
                ggml_backend_tensor_set(ids2, &ft, 0, sizeof(int32_t));
                ggml_backend_graph_compute(backend_, gf2);
                ggml_backend_tensor_get(cpy2, embed_buf.data(), 0, sizeof(float) * hidden);
                ggml_gallocr_free(galloc2);
                ggml_free(ectx2);
            }
            if (!do_step(embed_buf.data(), 1, cur_committed, last_logits_)) {
                result.error = "decode logits";
                return result;
            }
            cur_committed++;
            cache_.cur_pos = cur_committed;

            if (!do_decode(cur_committed, req.n_gen - 1, result.tokens, out_io)) {
                result.error = "decode";
                return result;
            }
        }
    }

    out_io.emit(-1);
    result.ok = true;
    return result;
}

// ── Snapshots ──────────────────────────────────────────────────────────

bool Qwen3Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    snapshot_free(slot);

    auto & snap = snapshots_[slot];
    const int n_layer = cache_.n_layer;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 4) + 4096;
    ip.no_alloc = true;
    snap.ctx = ggml_init(ip);
    if (!snap.ctx) return false;

    snap.k_snap.resize(n_layer);
    snap.v_snap.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        snap.k_snap[il] = ggml_dup_tensor(snap.ctx, cache_.k[il]);
        snap.v_snap[il] = ggml_dup_tensor(snap.ctx, cache_.v[il]);
        char name[64];
        std::snprintf(name, sizeof(name), "snap_k_%d", il);
        ggml_set_name(snap.k_snap[il], name);
        std::snprintf(name, sizeof(name), "snap_v_%d", il);
        ggml_set_name(snap.v_snap[il], name);
    }

    snap.buf = ggml_backend_alloc_ctx_tensors(snap.ctx, backend_);
    if (!snap.buf) {
        ggml_free(snap.ctx);
        snap.ctx = nullptr;
        return false;
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_backend_tensor_copy(cache_.k[il], snap.k_snap[il]);
        ggml_backend_tensor_copy(cache_.v[il], snap.v_snap[il]);
    }
    snap.cur_pos = cache_.cur_pos;

    std::printf("[qwen3] snapshot saved slot=%d pos=%d\n", slot, snap.cur_pos);
    std::fflush(stdout);
    return true;
}

void Qwen3Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_qwen3_snapshot(snapshots_[slot]);
}

bool Qwen3Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return snapshots_[slot].ctx != nullptr;
}

int Qwen3Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return snapshots_[slot].cur_pos;
}

ModelBackend::SnapshotRef Qwen3Backend::snapshot_ref(int slot) const {
    SnapshotRef ref;
    if (slot < 0 || slot >= PREFIX_SLOTS) return ref;
    const auto & snap = snapshots_[slot];
    if (!snap.ctx) return ref;
    ref.ctx     = snap.ctx;
    ref.buf     = snap.buf;
    ref.cur_pos = snap.cur_pos;
    return ref;
}

bool Qwen3Backend::snapshot_adopt(int slot, ggml_context * ctx,
                                  ggml_backend_buffer_t buf, int cur_pos,
                                  int32_t /*last_tok*/) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    snapshot_free(slot);

    auto & snap = snapshots_[slot];
    const int n_layer = cache_.n_layer;

    snap.k_snap.resize(n_layer, nullptr);
    snap.v_snap.resize(n_layer, nullptr);

    // Rebind tensors by name.
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->name[0]) continue;
        int il = -1;
        if (std::sscanf(t->name, "snap_k_%d", &il) == 1 && il >= 0 && il < n_layer) {
            snap.k_snap[il] = t;
        } else if (std::sscanf(t->name, "snap_v_%d", &il) == 1 && il >= 0 && il < n_layer) {
            snap.v_snap[il] = t;
        }
    }

    // Validate all layers bound.
    for (int il = 0; il < n_layer; ++il) {
        if (!snap.k_snap[il] || !snap.v_snap[il]) {
            snap.k_snap.clear();
            snap.v_snap.clear();
            return false;
        }
    }

    snap.ctx     = ctx;
    snap.buf     = buf;
    snap.cur_pos = cur_pos;
    std::fprintf(stderr, "[qwen3] snapshot adopted slot=%d pos=%d\n", slot, cur_pos);
    return true;
}

// ── Compress ───────────────────────────────────────────────────────────

ModelBackend::CompressResult Qwen3Backend::compress(const CompressRequest & req) {
    CompressResult result;
    if (req.input_ids.empty()) return result;

    const bool was_parked = parked_;
    if (!req.skip_park && !parked_) park("target");

    if (!drafter_loaded_) {
        if (!load_drafter(req.drafter_path, 999, req.drafter_gpu, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] load failed: %s\n", dflash27b_last_error());
            if (!req.skip_park && !was_parked) unpark("target");
            return result;
        }
        drafter_loaded_ = true;
    }

    result.compressed_ids = drafter_score_and_compress(
        drafter_ctx_, req.input_ids, req.keep_ratio);
    result.ok = true;

    if (req.residency_action == DraftResidencyAction::ReleaseAfterUse) {
        free_drafter();
    }

    if (!req.skip_park && !was_parked) unpark("target");
    return result;
}

bool Qwen3Backend::handle_compress(const std::string & line, const DaemonIO & io) {
    std::istringstream iss(line.substr(9));
    std::string ppath;
    int  keep_x1000 = 0;
    std::string drafter_path;
    if (!(iss >> ppath >> keep_x1000)) {
        std::fprintf(stderr, "[compress] bad args\n");
        io.emit(-1);
        return false;
    }

    std::getline(iss >> std::ws, drafter_path);
    bool skip_park = false;
    const std::string suffix = " nopark";
    if (drafter_path.size() > suffix.size() &&
        drafter_path.compare(drafter_path.size() - suffix.size(),
                             suffix.size(), suffix) == 0) {
        skip_park = true;
        drafter_path.resize(drafter_path.size() - suffix.size());
    }
    if (drafter_path.empty()) {
        std::fprintf(stderr, "[compress] bad args\n");
        io.emit(-1);
        return false;
    }

    auto src_ids = read_int32_file(ppath);
    if (src_ids.empty()) {
        std::fprintf(stderr, "[compress] empty input\n");
        io.emit(-1);
        return false;
    }

    const bool was_parked = parked_;
    if (!skip_park && !parked_) park("target");

    if (!drafter_loaded_) {
        if (!load_drafter(drafter_path, 999, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] load failed: %s\n", dflash27b_last_error());
            if (!skip_park && !was_parked) unpark("target");
            io.emit(-1);
            return false;
        }
        drafter_loaded_ = true;
    }

    const float keep = (float)keep_x1000 / 1000.0f;
    auto compressed = drafter_score_and_compress(drafter_ctx_, src_ids, keep);
    std::printf("[compress] %zu -> %zu tokens\n", src_ids.size(), compressed.size());
    std::fflush(stdout);

    if (!skip_park && !was_parked) unpark("target");

    for (int32_t t : compressed) io.emit(t);
    io.emit(-1);
    return true;
}

void Qwen3Backend::free_drafter() {
    if (drafter_loaded_) {
        dflash::common::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
    }
}

// ── try_handle_command ──────────────────────────────────────────────────

bool Qwen3Backend::try_handle_command(const std::string & /*line*/,
                                       const DaemonIO & /*io*/) {
    return false;  // no arch-specific commands
}

// ── Shutdown ───────────────────────────────────────────────────────────

void Qwen3Backend::shutdown() {
    free_drafter();
    for (int i = 0; i < PREFIX_SLOTS; ++i) {
        free_qwen3_snapshot(snapshots_[i]);
    }
    free_qwen3_cache(cache_);
    if (!parked_) {
        free_qwen3_drafter_model(w_);
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace dflash::common
