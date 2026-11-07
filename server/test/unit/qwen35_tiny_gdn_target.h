// A two-layer synthetic qwen35 target for state-handling tests: layer 0 is a
// Gated DeltaNet layer, layer 1 a full-attention layer, both with a dense
// SwiGLU FFN; f32 weights from a fixed seed, an F16 CPU embedding table, and
// the single-sequence cache the server uses (create_target_cache_partial
// with prefill_only, then migrate_prefill_cache for the F32 checkpoints).
// No model file. Owns the backend it is given.
#pragma once

#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace luce_test {

struct TinyGdnTarget {
    // head_v_dim == head_k_dim == kState; 128 takes the grouped-cols GDN
    // kernel on RDNA, smaller sizes the scalar one.
    int kState;
    static constexpr int kVHeads   = 2;
    static constexpr int kHidden   = 64;
    static constexpr int kHeadDim  = 128;   // attention head (FA-supported)
    static constexpr int kHeads    = 2;
    static constexpr int kHeadsKv  = 1;
    static constexpr int kFf       = 128;
    static constexpr int kVocab    = 64;
    static constexpr int kMaxCtx   = 256;
    static constexpr int kMaxVerify = 16;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    luce::common::TargetWeights w;
    luce::common::TargetCache cache;
    bool ok = false;

    TinyGdnTarget(ggml_backend_t be, int state_dim, bool raw_gates)
        : kState(state_dim), backend(be) {
        if (!backend) return;
        using luce::common::TargetLayer;

        w.n_layer = 2;
        w.full_attention_interval = 2;      // layer 0 GDN, layer 1 attention
        w.n_embd = kHidden;
        w.n_embd_head_k = kHeadDim;
        w.n_embd_head_v = kHeadDim;
        w.n_head = kHeads;
        w.n_head_kv = kHeadsKv;
        w.n_ff = kFf;
        w.n_vocab = kVocab;
        w.rope_dimension_count = 64;
        w.rope_sections[0] = 11; w.rope_sections[1] = 11;
        w.rope_sections[2] = 10; w.rope_sections[3] = 0;
        w.n_capture_layers = 0;
        w.is_moe = false;
        w.ssm_d_state = kState;
        w.ssm_n_group = 1;
        w.ssm_dt_rank = kVHeads;
        w.ssm_d_inner = kState * kVHeads;
        w.ssm_d_conv = 4;
        w.eos_id = -1;
        w.eos_chat_id = -1;
        w.layers.resize(2);

        ggml_init_params ip{};
        ip.mem_size = 64 * ggml_tensor_overhead();
        ip.no_alloc = true;
        wctx = ggml_init(ip);
        if (!wctx) return;

        const int conv_ch = w.ssm_d_inner + 2 * w.ssm_n_group * w.ssm_d_state;
        const int q_dim = kHeadDim * kHeads;
        const int kv_dim = kHeadDim * kHeadsKv;
        auto f32_1d = [&](int a) { return ggml_new_tensor_1d(wctx, GGML_TYPE_F32, a); };
        auto f32_2d = [&](int a, int b) { return ggml_new_tensor_2d(wctx, GGML_TYPE_F32, a, b); };
        for (int il = 0; il < 2; ++il) {
            TargetLayer & L = w.layers[(size_t)il];
            L.attn_norm      = f32_1d(kHidden);
            L.attn_post_norm = f32_1d(kHidden);
            L.w_gate = f32_2d(kHidden, kFf);
            L.w_up   = f32_2d(kHidden, kFf);
            L.w_down = f32_2d(kFf, kHidden);
        }
        TargetLayer & G = w.layers[0];
        G.wqkv        = f32_2d(kHidden, conv_ch);
        G.wqkv_gate   = f32_2d(kHidden, w.ssm_d_inner);
        G.ssm_beta    = f32_2d(kHidden, kVHeads);
        G.ssm_alpha   = f32_2d(kHidden, kVHeads);
        G.ssm_conv1d  = f32_2d(w.ssm_d_conv, conv_ch);
        G.ssm_dt_bias = f32_1d(kVHeads);
        G.ssm_a       = f32_1d(kVHeads);
        G.ssm_norm    = f32_1d(kState);
        G.ssm_out     = f32_2d(w.ssm_d_inner, kHidden);
        if (raw_gates) G.ssm_gate_ba = f32_1d(2 * kVHeads);
        TargetLayer & A = w.layers[1];
        A.wq     = f32_2d(kHidden, 2 * q_dim);
        A.wk     = f32_2d(kHidden, kv_dim);
        A.wv     = f32_2d(kHidden, kv_dim);
        A.wo     = f32_2d(q_dim, kHidden);
        A.q_norm = f32_1d(kHeadDim);
        A.k_norm = f32_1d(kHeadDim);
        w.out_norm = f32_1d(kHidden);
        w.output   = f32_2d(kHidden, kVocab);

        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (!wbuf) return;

        std::mt19937 rng(20261104u);
        std::uniform_real_distribution<float> small(-0.08f, 0.08f);
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t;
             t = ggml_get_next_tensor(wctx, t)) {
            std::vector<float> v((size_t)ggml_nelements(t));
            const bool norm = t == A.q_norm || t == A.k_norm || t == G.ssm_norm ||
                              t == w.out_norm || t == w.layers[0].attn_norm ||
                              t == w.layers[0].attn_post_norm ||
                              t == w.layers[1].attn_norm || t == w.layers[1].attn_post_norm;
            for (float & x : v) x = norm ? 1.0f + small(rng) : small(rng);
            if (t == G.ssm_a) {
                for (float & x : v) x = -std::exp(0.5f * small(rng) * 10.0f);
            }
            if (t == G.ssm_gate_ba) continue;   // filled from dt_bias and A below
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
        if (G.ssm_gate_ba) {
            std::vector<float> ba((size_t)2 * kVHeads);
            ggml_backend_tensor_get(G.ssm_dt_bias, ba.data(), 0, sizeof(float) * kVHeads);
            ggml_backend_tensor_get(G.ssm_a, ba.data() + kVHeads, 0, sizeof(float) * kVHeads);
            ggml_backend_tensor_set(G.ssm_gate_ba, ba.data(), 0, sizeof(float) * ba.size());
        }

        // F16 embedding rows on the host, as the GGUF loader keeps them.
        w.embedder.n_embd = kHidden;
        w.embedder.n_vocab = kVocab;
        w.embedder.tok_embd_type = GGML_TYPE_F16;
        w.embedder.row_bytes = (size_t)kHidden * sizeof(ggml_fp16_t);
        w.embedder.tok_embd_owned.resize(w.embedder.row_bytes * kVocab);
        auto * rows = reinterpret_cast<ggml_fp16_t *>(w.embedder.tok_embd_owned.data());
        for (int i = 0; i < kHidden * kVocab; ++i) rows[i] = ggml_fp32_to_fp16(4.0f * small(rng));
        w.embedder.tok_embd_bytes = w.embedder.tok_embd_owned.data();

        ok = luce::common::create_target_cache_partial(
            w, kMaxCtx, kMaxVerify, backend, cache,
            /*prefill_only=*/true, /*layer_begin=*/0, /*layer_end=*/2,
            /*allocate_target_feat=*/false, /*ctx_alloc=*/0,
            /*f32_ssm_intermediates=*/false, /*paged_attention=*/false,
            /*n_seq_slots=*/1, /*concurrent_tree=*/false,
            GGML_TYPE_F16, GGML_TYPE_F16);
        if (ok) {
            ok = luce::common::migrate_prefill_cache(
                w, kMaxCtx, kMaxVerify, backend, cache, /*enable_specla=*/false);
        }
        if (!ok) std::fprintf(stderr, "[tiny-gdn] cache: %s\n", luce_last_error());
    }

    ~TinyGdnTarget() {
        luce::common::free_target_cache(cache);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (backend) ggml_backend_free(backend);
    }

    TinyGdnTarget(const TinyGdnTarget &) = delete;
    TinyGdnTarget & operator=(const TinyGdnTarget &) = delete;
};

}  // namespace luce_test
