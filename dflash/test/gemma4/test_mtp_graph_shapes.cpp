// Phase 3a shape test: MTP step graph builds without crash and output tensor
// shapes match the contract:
//   out_logits  : F32 [n_vocab, 1]
//   out_h_post  : F32 [n_embd_backbone, 1]
//   out_argmax  : I32 [1]
//
// We stub GemmaTargetCache and GemmaTargetWeights with zero-initialised tensors
// of the correct shapes. No actual inference is performed — this is a graph
// construction smoke test only.
//
// Run:
//   MTP_GGUF=/path/to/gemma-4-31B-it-assistant.Q4_K_M.gguf \
//     ./build/test_mtp_graph_shapes
//
// Requires MTP_GGUF to be set; exits 77 (autotools skip) if absent.

#include "../src/internal.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash27b;

static int fail(const char * msg) {
    std::fprintf(stderr, "[FAIL] %s\n", msg);
    return 1;
}

// Build a minimal stub GemmaTargetWeights with tok_embd of the right shape.
// The stub does NOT allocate GPU memory for embedding data; graph construction
// only needs the tensor *metadata* (ne[], type), not data.
static bool build_stub_target_weights(ggml_backend_t backend,
                                       int n_vocab,
                                       int n_embd_backbone,
                                       int n_layer,
                                       const std::vector<bool> & swa_layers,
                                       GemmaTargetWeights & out) {
    // Minimal tensor count: tok_embd + per-layer rope_freqs (optional) + out_norm + output
    const size_t n_tensors_est = (size_t)(n_layer + 8);
    ggml_init_params ip{};
    ip.mem_size   = n_tensors_est * ggml_tensor_overhead() + 4096;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    // tok_embd: [n_embd_backbone, n_vocab]  (ggml ne[0]=embedding_dim, ne[1]=n_vocab)
    out.tok_embd = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd_backbone, n_vocab);
    ggml_set_name(out.tok_embd, "stub_tok_embd");

    // Populate fields needed by build_mtp_step_graph
    // Dense 31B: head_dim=256 (from GGUF "gemma4.attention.key_length")
    out.n_embd          = n_embd_backbone;
    out.n_head          = 32;
    out.n_head_kv       = 8;
    out.head_dim        = 256;
    out.head_dim_swa    = 256;
    out.n_layer         = n_layer;
    out.rope_theta      = 1000000.0f;
    out.rope_theta_swa  = 1000000.0f;
    out.attn_scale      = 1.0f;
    out.logit_softcap   = 30.0f;
    out.swa_layers      = swa_layers;

    // Populate minimal per-layer structs (only rope_freqs is accessed by MTP graph
    // for full-attention donor layers)
    out.layers.resize((size_t)n_layer);
    // Leave rope_freqs nullptr for all layers (proportional RoPE freq_factors are
    // optional; nullptr → falls back to base rope_theta scaling).

    out.backend = backend;
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) { ggml_free(out.ctx); out.ctx = nullptr; return false; }

    // Zero-init the tok_embd (so GPU tensor is valid even though we won't run compute)
    ggml_backend_tensor_set(out.tok_embd, nullptr, 0, 0); // no-op; buffer already zeroed

    return true;
}

// Build a minimal stub GemmaTargetCache with KV tensors of the right shapes.
// attn_k[i]: [head_dim_kv, max_ctx, n_head_kv]
// attn_v[i]: [head_dim_kv, max_ctx, n_head_kv]
// head_dim_kv_swa and head_dim_kv_full allow different head_dims per attention type.
static bool build_stub_target_cache(ggml_backend_t backend,
                                     int n_layer,
                                     int n_kv_per_layer,      // n_head_kv for KV cache
                                     int head_dim_kv_swa,     // head_dim for SWA layers
                                     int head_dim_kv_full,    // head_dim for full-attn layers
                                     int max_ctx,
                                     const std::vector<bool> & swa_layers,
                                     GemmaTargetCache & out) {
    // Count KV-owning layers (non-shared). For stub, all layers own a KV slot.
    const int n_kv_slots = n_layer;  // stub: one per layer (no sharing)

    const size_t n_tensors_est = (size_t)(2 * n_kv_slots + 4);
    ggml_init_params ip{};
    ip.mem_size   = n_tensors_est * ggml_tensor_overhead() + 4096;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.base_ctx = ggml_init(ip);
    if (!out.base_ctx) return false;

    out.layer_to_kv_idx.resize((size_t)n_layer);
    out.layer_to_donor_kv.resize((size_t)n_layer, -1);
    out.attn_k.resize((size_t)n_kv_slots, nullptr);
    out.attn_v.resize((size_t)n_kv_slots, nullptr);

    for (int il = 0; il < n_layer; il++) {
        out.layer_to_kv_idx[il] = il;  // one-to-one for stub

        // Use different head_dim per attention type
        const bool is_swa = (il < (int)swa_layers.size()) && swa_layers[il];
        const int layer_head_dim = is_swa ? head_dim_kv_swa : head_dim_kv_full;
        ggml_tensor * K = ggml_new_tensor_3d(out.base_ctx, GGML_TYPE_F16,
            layer_head_dim, max_ctx, n_kv_per_layer);
        ggml_tensor * V = ggml_new_tensor_3d(out.base_ctx, GGML_TYPE_F16,
            layer_head_dim, max_ctx, n_kv_per_layer);
        char name[64];
        std::snprintf(name, sizeof(name), "stub_k_%d", il);
        ggml_set_name(K, name);
        std::snprintf(name, sizeof(name), "stub_v_%d", il);
        ggml_set_name(V, name);
        out.attn_k[il] = K;
        out.attn_v[il] = V;
    }

    out.backend       = backend;
    out.max_ctx       = max_ctx;
    out.cur_pos       = 16;   // pretend we have 16 committed tokens
    out.swa_ctx_alloc = max_ctx;
    (void)swa_layers;

    out.base_buf = ggml_backend_alloc_ctx_tensors(out.base_ctx, backend);
    if (!out.base_buf) { ggml_free(out.base_ctx); out.base_ctx = nullptr; return false; }

    // Zero-init (backend buffer is already zeroed by alloc; explicit set skipped for perf)

    return true;
}

int main() {
    const char * mtp_path = std::getenv("MTP_GGUF");
    if (!mtp_path) {
        std::fprintf(stderr, "[skip] MTP_GGUF not set; skipping test_mtp_graph_shapes\n");
        return 77;  // autotools skip code
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        return fail("ggml_backend_cuda_init(0) failed");
    }

    // ── Load MTP weights ─────────────────────────────────────────────────────
    MtpDrafterWeights mtp{};
    if (!load_gemma4_mtp_assistant(std::string(mtp_path), backend, mtp)) {
        std::fprintf(stderr, "  loader error: %s\n", gemma4_last_error());
        ggml_backend_free(backend);
        return fail("load_gemma4_mtp_assistant failed");
    }

    const int n_embd_backbone = mtp.n_embd_backbone;  // e.g. 5376
    const int n_vocab         = 262144;                 // Dense 31B vocab
    const int n_target_layers = 60;                    // Dense 31B
    const int max_ctx         = 256;                   // 256-aligned (full-attn donor view padding requires it; matches create_gemma4_cache 256-alignment in gemma4_target_graph.cpp:570)

    // Dense 31B SWA pattern: odd-indexed = SWA, even = full attention
    std::vector<bool> target_swa(n_target_layers, false);
    for (int il = 0; il < n_target_layers; il++) {
        target_swa[il] = ((il % 2) == 1);
    }

    // ── Build stub target structures ─────────────────────────────────────────
    GemmaTargetWeights stub_target{};
    if (!build_stub_target_weights(backend, n_vocab, n_embd_backbone,
                                   n_target_layers, target_swa, stub_target)) {
        ggml_backend_free(backend);
        return fail("build_stub_target_weights failed");
    }

    GemmaTargetCache stub_cache{};
    // KV head_dim: derived from MTP weight shapes (attn_q_norm->ne[0] gives per-head Q dim,
    // which must equal the target KV head_dim for flash_attn_ext to work).
    // Dense 31B: SWA layers use head_dim=256, full-attn layers use head_dim=512
    // (derived from mtp.layers[0].attn_q_norm->ne[0]=256 for SWA, [3].attn_q_norm->ne[0]=512 for full).
    const int head_dim_swa_stub = (int)mtp.layers[0].attn_q_norm->ne[0];  // SWA layers 0-2
    const int head_dim_full_stub = (int)mtp.layers[3].attn_q_norm->ne[0]; // Full-attn layer 3
    std::fprintf(stderr, "[shape_test] MTP Q head_dim: SWA=%d, full=%d\n",
        head_dim_swa_stub, head_dim_full_stub);
    if (!build_stub_target_cache(backend, n_target_layers,
                                  /*n_kv_per_layer=*/8,
                                  head_dim_swa_stub, head_dim_full_stub,
                                  max_ctx, target_swa, stub_cache)) {
        free_gemma4_target_weights(stub_target);
        ggml_backend_free(backend);
        return fail("build_stub_target_cache failed");
    }

    // ── Build MTP step graph ─────────────────────────────────────────────────
    MtpStepGraph graph{};
    const int attn_pos = stub_cache.cur_pos;  // = 16

    if (!build_mtp_step_graph(mtp, stub_cache, stub_target, graph, attn_pos)) {
        std::fprintf(stderr, "  build error: %s\n", gemma4_last_error());
        free_gemma4_target_weights(stub_target);
        // Note: stub_cache KV tensors point into base_ctx; free manually:
        if (stub_cache.base_buf) ggml_backend_buffer_free(stub_cache.base_buf);
        if (stub_cache.base_ctx) ggml_free(stub_cache.base_ctx);
        ggml_backend_free(backend);
        return fail("build_mtp_step_graph failed");
    }

    // ── Shape assertions ─────────────────────────────────────────────────────

    // 1. Input shapes
    if (!graph.in_tok || graph.in_tok->ne[0] != 1 ||
        graph.in_tok->type != GGML_TYPE_I32) {
        ggml_backend_free(backend);
        return fail("in_tok shape/type mismatch: expected I32[1]");
    }

    if (!graph.in_h_prev ||
        graph.in_h_prev->ne[0] != (int64_t)n_embd_backbone ||
        graph.in_h_prev->ne[1] != 1 ||
        graph.in_h_prev->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "  in_h_prev->ne = [%lld, %lld]\n",
            (long long)(graph.in_h_prev ? graph.in_h_prev->ne[0] : -1),
            (long long)(graph.in_h_prev ? graph.in_h_prev->ne[1] : -1));
        ggml_backend_free(backend);
        return fail("in_h_prev shape/type mismatch: expected F32[n_embd_backbone, 1]");
    }

    if (!graph.in_pos || graph.in_pos->ne[0] != 1 ||
        graph.in_pos->type != GGML_TYPE_I32) {
        ggml_backend_free(backend);
        return fail("in_pos shape/type mismatch: expected I32[1]");
    }

    // 2. out_h_post: F32 [n_embd_backbone, 1]
    if (!graph.out_h_post ||
        graph.out_h_post->ne[0] != (int64_t)n_embd_backbone ||
        graph.out_h_post->ne[1] != 1 ||
        graph.out_h_post->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "  out_h_post->ne = [%lld, %lld], type=%s\n",
            (long long)(graph.out_h_post ? graph.out_h_post->ne[0] : -1),
            (long long)(graph.out_h_post ? graph.out_h_post->ne[1] : -1),
            graph.out_h_post ? ggml_type_name(graph.out_h_post->type) : "null");
        ggml_backend_free(backend);
        return fail("out_h_post shape mismatch: expected F32[n_embd_backbone, 1]");
    }

    // 3. out_logits: F32 [n_vocab, 1]
    if (!graph.out_logits ||
        graph.out_logits->ne[0] != (int64_t)n_vocab ||
        graph.out_logits->ne[1] != 1 ||
        graph.out_logits->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "  out_logits->ne = [%lld, %lld], type=%s\n",
            (long long)(graph.out_logits ? graph.out_logits->ne[0] : -1),
            (long long)(graph.out_logits ? graph.out_logits->ne[1] : -1),
            graph.out_logits ? ggml_type_name(graph.out_logits->type) : "null");
        ggml_backend_free(backend);
        return fail("out_logits shape mismatch: expected F32[n_vocab, 1]");
    }

    // 4. out_argmax: I32 [1]
    if (!graph.out_argmax ||
        graph.out_argmax->ne[0] != 1 ||
        graph.out_argmax->type != GGML_TYPE_I32) {
        std::fprintf(stderr, "  out_argmax->ne[0]=%lld type=%s\n",
            (long long)(graph.out_argmax ? graph.out_argmax->ne[0] : -1),
            graph.out_argmax ? ggml_type_name(graph.out_argmax->type) : "null");
        ggml_backend_free(backend);
        return fail("out_argmax shape/type mismatch: expected I32[1]");
    }

    std::fprintf(stderr, "[PASS] all shape assertions passed for MTP step graph\n");
    std::fprintf(stderr, "  n_embd_backbone=%d, n_vocab=%d, n_layers=%zu, attn_pos=%d\n",
        n_embd_backbone, n_vocab, mtp.layers.size(), attn_pos);

    // Cleanup
    free_mtp_step_graph(graph);
    // Stub cache: manual teardown since we bypassed create_gemma4_cache
    if (stub_cache.base_buf) ggml_backend_buffer_free(stub_cache.base_buf);
    if (stub_cache.base_ctx) ggml_free(stub_cache.base_ctx);
    free_gemma4_target_weights(stub_target);
    free_gemma4_mtp_assistant(mtp);
    ggml_backend_free(backend);

    return 0;
}
