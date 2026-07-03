// Loads a DFlash draft model from a GGUF file on disk into a ggml context
// on the CUDA backend.
//
// This is the Q8_0-quantized counterpart of draft_safetensors_loader.cpp. The
// draft graph builder (draft_graph.cpp) doesn't care about tensor storage
// types — ggml's ggml_mul_mat handles Q8_0 × F32 dequantization transparently.
//
// GGUF arch: "qwen35-dflash-draft" (from convert_dflash_to_gguf.py /
// quantize_draft_q8.py). Tensor naming convention:
//
//   dflash.fc.weight                        [5*hidden, hidden]  Q8_0 / F16
//   dflash.hidden_norm.weight               [hidden]            F32
//   output_norm.weight                      [hidden]            F32
//   blk.<i>.attn_norm.weight                [hidden]            F32
//   blk.<i>.ffn_norm.weight                 [hidden]            F32
//   blk.<i>.attn_q.weight                   [q_dim, hidden]     Q8_0 / F16
//   blk.<i>.attn_k.weight                   [kv_dim, hidden]    Q8_0 / F16
//   blk.<i>.attn_v.weight                   [kv_dim, hidden]    Q8_0 / F16
//   blk.<i>.attn_output.weight              [hidden, q_dim]     Q8_0 / F16
//   blk.<i>.attn_q_norm.weight              [head_dim]          F32
//   blk.<i>.attn_k_norm.weight              [head_dim]          F32
//   blk.<i>.ffn_gate.weight                 [intermediate, hidden]  Q8_0 / F16
//   blk.<i>.ffn_up.weight                   [intermediate, hidden]  Q8_0 / F16
//   blk.<i>.ffn_down.weight                 [hidden, intermediate]  Q8_0 / F16

#include "internal.h"
#include "common/derived_scalars.h"
#include "common/gguf_mmap.h"
#include "common/gguf_bounds.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

namespace {

uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

int count_swa_layers(const DraftWeights & w) {
    int n_swa = 0;
    for (const DraftLayer & layer : w.layers) {
        if (layer.is_swa) n_swa++;
    }
    return n_swa;
}

bool check_shape_1d(const ggml_tensor * t, int64_t ne0, const char * name, char * buf, size_t buf_sz) {
    if (!t || t->ne[0] != ne0) {
        std::snprintf(buf, buf_sz, "draft GGUF: Domino tensor %s shape mismatch: got [%lld], expected [%lld]",
                      name, t ? (long long)t->ne[0] : -1LL, (long long)ne0);
        return false;
    }
    return true;
}

bool check_shape_2d(const ggml_tensor * t, int64_t ne0, int64_t ne1,
                    const char * name, char * buf, size_t buf_sz) {
    if (!t || t->ne[0] != ne0 || t->ne[1] != ne1) {
        std::snprintf(buf, buf_sz,
                      "draft GGUF: Domino tensor %s shape mismatch: got [%lld,%lld], expected [%lld,%lld]",
                      name,
                      t ? (long long)t->ne[0] : -1LL,
                      t ? (long long)t->ne[1] : -1LL,
                      (long long)ne0, (long long)ne1);
        return false;
    }
    return true;
}

} // namespace

bool load_draft_gguf(const std::string & path,
                     ggml_backend_t       backend,
                     DraftWeights &       out,
                     const TargetWeights * target) {

    // ── 1. Parse metadata + create ggml_context with tensor descriptors ──
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Validate arch
    std::string arch_s;
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture in draft GGUF");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        arch_s = arch;
        if (arch_s != "qwen35-dflash-draft" &&
            arch_s != "dflash-draft" &&
            arch_s != "gemma4-dflash-draft") {
            set_last_error(std::string("unexpected draft arch: ") + arch +
                           " (expected qwen35-dflash-draft, dflash-draft, or gemma4-dflash-draft)");
            gguf_free(gctx);
            return false;
        }
    }

    // Read dimensions from GGUF metadata
    const char * A = arch_s.c_str();
    char key[128];

    auto read_u32 = [&](const char * suffix, uint32_t fallback) -> uint32_t {
        std::snprintf(key, sizeof(key), "%s.%s", A, suffix);
        return get_u32_or(gctx, key, fallback);
    };
    auto read_f32 = [&](const char * suffix, float fallback) -> float {
        std::snprintf(key, sizeof(key), "%s.%s", A, suffix);
        int64_t id = gguf_find_key(gctx, key);
        if (id < 0) return fallback;
        return gguf_get_val_f32(gctx, id);
    };

    const uint32_t n_embd    = read_u32("embedding_length",        0);
    const uint32_t n_layer   = read_u32("block_count",             0);
    const uint32_t n_ff      = read_u32("feed_forward_length",     0);
    const uint32_t n_head    = read_u32("attention.head_count",    0);
    const uint32_t n_head_kv = read_u32("attention.head_count_kv", 0);
    const uint32_t head_dim  = read_u32("attention.key_length",    0);
    const uint32_t block_sz  = read_u32("dflash.block_size",       0);
    uint32_t n_tgt_lay       = read_u32("dflash.n_target_layers",  0);
    const uint32_t domino_meta_enabled = read_u32("dflash.domino.enabled", 0);
    const uint32_t domino_meta_gru     = read_u32("dflash.domino.gru_hidden_dim", 0);
    const uint32_t domino_meta_emb     = read_u32("dflash.domino.emb_dim", 0);
    const uint32_t domino_meta_vocab   = read_u32("dflash.domino.vocab_size", 0);
    const uint32_t dspark_meta_enabled = read_u32("dflash.dspark.enabled", 0);
    const uint32_t dspark_meta_rank    = read_u32("dflash.dspark.markov_rank", 0);
    const uint32_t dspark_meta_vocab   = read_u32("dflash.dspark.vocab_size", 0);
    const uint32_t dspark_meta_conf    = read_u32("dflash.dspark.confidence_dim", 0);
    // Explicit captured target-layer ids (data-driven). Lets any DFlash drafter
    // load without a hardcoded per-arch set; the array length also backstops
    // n_target_layers when the scalar KV is absent.
    {
        std::snprintf(key, sizeof(key), "%s.%s", A, "dflash.target_layer_ids");
        const int64_t target_ids_id = gguf_find_key(gctx, key);
        if (target_ids_id >= 0 &&
            gguf_get_kv_type(gctx, target_ids_id) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(gctx, target_ids_id) == GGUF_TYPE_INT32) {
            const uint32_t n = (uint32_t)gguf_get_arr_n(gctx, target_ids_id);
            const int32_t * vals =
                (const int32_t *)gguf_get_arr_data(gctx, target_ids_id);
            out.capture_layer_ids.assign(vals, vals + n);
            if (n_tgt_lay == 0) n_tgt_lay = n;
        }
    }
    if (n_tgt_lay == 0 && n_embd != 0) {
        const uint32_t n_target_features = read_u32("dflash.n_target_features", 0);
        if (n_target_features != 0 && (n_target_features % n_embd) == 0) {
            n_tgt_lay = n_target_features / n_embd;
        }
    }

    if (n_embd == 0 || n_layer == 0 || n_ff == 0 || n_head == 0 ||
        n_head_kv == 0 || head_dim == 0) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "draft GGUF: missing hparams: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // Store GGUF-declared config into DraftWeights (replaces hardcoded defaults).
    out.block_size = (int)block_sz;
    out.n_target_layers = (int)n_tgt_lay;

    // Propagate target model properties if available.
    if (target) {
        out.mask_token_id = target->mask_token_id;
    }

    // Upper bounds on hparams. Guards against malformed/hostile GGUFs that
    // would otherwise trigger huge allocations or signed-int overflow when
    // narrowed below. Limits chosen well above any plausible LLM config.
    constexpr uint32_t MAX_LAYERS  = 1024;
    constexpr uint32_t MAX_EMBD    = 1u << 17;   // 131072
    constexpr uint32_t MAX_FF      = 1u << 19;   // 524288
    constexpr uint32_t MAX_HEADS   = 1024;
    constexpr uint32_t MAX_HEADDIM = 1024;
    if (n_layer   > MAX_LAYERS  || n_embd    > MAX_EMBD  ||
        n_ff      > MAX_FF      || n_head    > MAX_HEADS ||
        n_head_kv > MAX_HEADS   || head_dim  > MAX_HEADDIM ||
        n_head_kv > n_head      || (n_head % n_head_kv) != 0) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "draft GGUF: hparams out of range: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // ── 2. Wire tensor pointers into DraftWeights ────────────────────────
    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer   = (int)n_layer;
    out.n_head    = (int)n_head;
    out.n_head_kv = (int)n_head_kv;
    out.head_dim  = (int)head_dim;
    out.n_embd    = (int)n_embd;
    out.n_ff      = (int)n_ff;
    out.rope_theta = read_f32("rope.freq_base", 0.0f);
    if (out.rope_theta == 0.0f) {
        fprintf(stderr, "[draft-gguf] WARNING: rope.freq_base not found in GGUF, draft RoPE will be wrong\n");
    }
    out.layers.assign((size_t)n_layer, DraftLayer{});

    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };
    auto g_any = [&](const char * a, const char * b) -> ggml_tensor * {
        if (ggml_tensor * t = g(a)) return t;
        return g(b);
    };

    out.fc          = g_any("dflash.fc.weight", "dflash_fc.weight");
    out.hidden_norm = g_any("dflash.hidden_norm.weight", "dflash_hidden_norm.weight");
    out.out_norm    = g("output_norm.weight");
    if (!out.fc || !out.hidden_norm || !out.out_norm) {
        set_last_error("draft GGUF: missing top-level tensors "
                       "(dflash.fc|dflash_fc / dflash.hidden_norm|dflash_hidden_norm / output_norm)");
        gguf_free(gctx);
        return false;
    }

    for (int il = 0; il < out.n_layer; il++) {
        char name[128];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        DraftLayer & L = out.layers[il];
        L.attn_norm = fnd("attn_norm.weight");
        L.ffn_norm  = fnd("ffn_norm.weight");
        if (!L.ffn_norm) L.ffn_norm = fnd("post_attention_norm.weight");
        L.wq        = fnd("attn_q.weight");
        L.wk        = fnd("attn_k.weight");
        L.wv        = fnd("attn_v.weight");
        L.wo        = fnd("attn_output.weight");
        L.q_norm    = fnd("attn_q_norm.weight");
        L.k_norm    = fnd("attn_k_norm.weight");
        L.w_gate    = fnd("ffn_gate.weight");
        L.w_up      = fnd("ffn_up.weight");
        L.w_down    = fnd("ffn_down.weight");
        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            char b[128];
            std::snprintf(b, sizeof(b), "draft GGUF: layer %d missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
    }

    out.domino = DraftDominoWeights{};
    out.domino.start    = g("dflash.domino.start");
    out.domino.gru_w_ih = g("dflash.domino.gru.weight_ih");
    out.domino.gru_w_hh = g("dflash.domino.gru.weight_hh");
    out.domino.gru_b_ih = g("dflash.domino.gru.bias_ih");
    out.domino.gru_b_hh = g("dflash.domino.gru.bias_hh");
    out.domino.head_w1  = g("dflash.domino.head.w1");
    out.domino.head_b1  = g("dflash.domino.head.b1");
    out.domino.head_w2  = g("dflash.domino.head.w2");
    out.domino.head_b2  = g("dflash.domino.head.b2");

    const bool domino_any =
        out.domino.start || out.domino.gru_w_ih || out.domino.gru_w_hh ||
        out.domino.gru_b_ih || out.domino.gru_b_hh || out.domino.head_w1 ||
        out.domino.head_b1 || out.domino.head_w2 || out.domino.head_b2 ||
        domino_meta_enabled != 0;
    if (domino_any) {
        const bool domino_all =
            out.domino.start && out.domino.gru_w_ih && out.domino.gru_w_hh &&
            out.domino.gru_b_ih && out.domino.gru_b_hh && out.domino.head_w1 &&
            out.domino.head_b1 && out.domino.head_w2 && out.domino.head_b2;
        if (!domino_all) {
            set_last_error("draft GGUF: incomplete Domino aux-head tensors");
            gguf_free(gctx);
            return false;
        }

        out.domino.gru_hidden_dim =
            domino_meta_gru != 0 ? (int)domino_meta_gru : (int)out.domino.gru_w_hh->ne[0];
        out.domino.emb_dim =
            domino_meta_emb != 0 ? (int)domino_meta_emb : (int)out.domino.head_w1->ne[1];
        out.domino.vocab_size =
            domino_meta_vocab != 0 ? (int)domino_meta_vocab : (int)out.domino.head_w2->ne[1];

        const int64_t H = out.domino.gru_hidden_dim;
        const int64_t E = out.domino.emb_dim;
        const int64_t V = out.domino.vocab_size;
        char shape_err[320];
        if (!check_shape_1d(out.domino.start, H, "start", shape_err, sizeof(shape_err)) ||
            !check_shape_2d(out.domino.gru_w_ih, out.n_embd, 3 * H, "gru.weight_ih", shape_err, sizeof(shape_err)) ||
            !check_shape_2d(out.domino.gru_w_hh, H, 3 * H, "gru.weight_hh", shape_err, sizeof(shape_err)) ||
            !check_shape_1d(out.domino.gru_b_ih, 3 * H, "gru.bias_ih", shape_err, sizeof(shape_err)) ||
            !check_shape_1d(out.domino.gru_b_hh, 3 * H, "gru.bias_hh", shape_err, sizeof(shape_err)) ||
            !check_shape_2d(out.domino.head_w1, out.n_embd + H, E, "head.w1", shape_err, sizeof(shape_err)) ||
            !check_shape_1d(out.domino.head_b1, E, "head.b1", shape_err, sizeof(shape_err)) ||
            !check_shape_2d(out.domino.head_w2, E, V, "head.w2", shape_err, sizeof(shape_err)) ||
            !check_shape_1d(out.domino.head_b2, V, "head.b2", shape_err, sizeof(shape_err))) {
            set_last_error(shape_err);
            gguf_free(gctx);
            return false;
        }

        out.domino.enabled = true;
        std::fprintf(stderr, "[draft GGUF] Domino GRU head enabled: H=%d E=%d vocab=%d\n",
                     out.domino.gru_hidden_dim, out.domino.emb_dim,
                     out.domino.vocab_size);
    }

    out.dspark = DraftDSparkWeights{};
    out.dspark.markov_w1    = g("dflash.dspark.markov.w1");
    out.dspark.markov_w2    = g("dflash.dspark.markov.w2");
    out.dspark.confidence_w = g("dflash.dspark.confidence.weight");
    out.dspark.confidence_b = g("dflash.dspark.confidence.bias");

    const bool dspark_any =
        out.dspark.markov_w1 || out.dspark.markov_w2 ||
        out.dspark.confidence_w || out.dspark.confidence_b ||
        dspark_meta_enabled != 0;
    if (dspark_any) {
        if (!out.dspark.markov_w1 || !out.dspark.markov_w2) {
            set_last_error("draft GGUF: incomplete DSpark Markov tensors");
            gguf_free(gctx);
            return false;
        }
        out.dspark.markov_rank =
            dspark_meta_rank != 0 ? (int)dspark_meta_rank : (int)out.dspark.markov_w1->ne[0];
        out.dspark.vocab_size =
            dspark_meta_vocab != 0 ? (int)dspark_meta_vocab : (int)out.dspark.markov_w1->ne[1];

        const int64_t R = out.dspark.markov_rank;
        const int64_t V = out.dspark.vocab_size;
        char shape_err[320];
        if (!check_shape_2d(out.dspark.markov_w1, R, V, "dspark.markov.w1", shape_err, sizeof(shape_err)) ||
            !check_shape_2d(out.dspark.markov_w2, R, V, "dspark.markov.w2", shape_err, sizeof(shape_err))) {
            set_last_error(shape_err);
            gguf_free(gctx);
            return false;
        }

        const bool conf_any = out.dspark.confidence_w || out.dspark.confidence_b || dspark_meta_conf != 0;
        if (conf_any) {
            if (!out.dspark.confidence_w || !out.dspark.confidence_b) {
                set_last_error("draft GGUF: incomplete DSpark confidence tensors");
                gguf_free(gctx);
                return false;
            }
            out.dspark.confidence_dim =
                dspark_meta_conf != 0 ? (int)dspark_meta_conf : (int)out.dspark.confidence_w->ne[0];
            const int64_t C = out.dspark.confidence_dim;
            if (!check_shape_2d(out.dspark.confidence_w, C, 1, "dspark.confidence.weight", shape_err, sizeof(shape_err)) ||
                !check_shape_1d(out.dspark.confidence_b, 1, "dspark.confidence.bias", shape_err, sizeof(shape_err))) {
                set_last_error(shape_err);
                gguf_free(gctx);
                return false;
            }
        }

        out.dspark.enabled = true;
        std::fprintf(stderr, "[draft GGUF] DSpark Markov head enabled: rank=%d vocab=%d confidence_dim=%d\n",
                     out.dspark.markov_rank, out.dspark.vocab_size,
                     out.dspark.confidence_dim);
    }

    // GGUF Qwen3.6 drafters carry SWA metadata emitted by the converter:
    //   dflash-draft.attention.sliding_window = 2048
    //   dflash-draft.attention.sliding_window_pattern = [true,true,true,true,false]
    out.swa_window = (int)read_u32("attention.sliding_window", 0);
    std::snprintf(key, sizeof(key), "%s.%s", A, "attention.sliding_window_pattern");
    int64_t swp_id = gguf_find_key(gctx, key);
    if (swp_id >= 0 && gguf_get_kv_type(gctx, swp_id) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(gctx, swp_id) == GGUF_TYPE_BOOL) {
        const size_t n = gguf_get_arr_n(gctx, swp_id);
        const bool * pattern = static_cast<const bool *>(gguf_get_arr_data(gctx, swp_id));
        for (size_t il = 0; il < n && il < out.layers.size(); il++) {
            out.layers[il].is_swa = pattern[il];
        }
    }
    const int n_swa = count_swa_layers(out);
    if (n_swa > 0) {
        std::fprintf(stderr, "[draft GGUF] SWA layers: %d/%d (window=%d)\n",
                     n_swa, out.n_layer, out.swa_window);
    }

    // ── 3. Allocate CUDA buffer for all tensors ──────────────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed (draft GGUF)");
        gguf_free(gctx);
        return false;
    }

    // ── 4. mmap file and copy tensor bytes to CUDA ───────────────────────
    std::string err;
    GgufMmap mm;
    if (!mm.open(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const uint8_t * mm_addr = (const uint8_t *)mm.data();
    const size_t    mm_len  = mm.size();
    const size_t data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    size_t total = 0;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t rel_off = gguf_get_tensor_offset(gctx, tid);
        const size_t sz      = gguf_get_tensor_size(gctx, tid);
        if (!gguf_tensor_in_file(data_start, rel_off, sz, mm_len)) {
            set_last_error(gguf_bounds_error("draft GGUF", tname,
                ggml_type_name(gguf_get_tensor_type(gctx, tid)),
                data_start, rel_off, sz, mm_len));
            gguf_free(gctx);
            return false;
        }
        const uint8_t * tensor_bytes = mm_addr + data_start + rel_off;
        ggml_backend_tensor_set(t, tensor_bytes, 0, sz);
        total += sz;
    }

    gguf_free(gctx);

    // Structural defense: derive head_dim / n_head / n_head_kv from weight
    // tensor shapes and assert against GGUF-declared metadata.
    // All draft layers have wq/wk (no deltanet mix), so layer 0 suffices.
    // wq: [n_embd, n_head*head_dim], ne[1]=n_head*head_dim, ne[0]=n_embd.
    // wk: [n_embd, n_head_kv*head_dim], ne[1]=n_head_kv*head_dim.
    {
        const DraftLayer & L0 = out.layers[0];
        const int64_t exp_q_dim  = (int64_t)out.n_head    * out.head_dim;
        const int64_t exp_kv_dim = (int64_t)out.n_head_kv * out.head_dim;
        const int64_t exp_n_embd = (int64_t)out.n_embd;
        std::string err;
        if (!dflash::common::verify_derived_scalars(
                L0.wq->ne[1], L0.wk->ne[1], L0.wq->ne[0],
                exp_q_dim, exp_kv_dim, exp_n_embd,
                "blk.0", err)) {
            set_last_error(err);
            return false;
        }
        // fc: [n_capture_layers*n_embd, n_embd] — ne[0] counts the CAPTURE
        // layers the fc consumes. Some draft GGUFs (gemma4) store the
        // TARGET's layer count in dflash.n_target_layers instead of the
        // capture count; per this file's own philosophy the weights are
        // ground truth, so when fc disagrees but is an exact multiple of
        // n_embd, derive the count from the tensor and warn. Fail only on
        // a genuinely inconsistent shape.
        if (out.n_target_layers > 0) {
            const int64_t derived_fc_in  = out.fc->ne[0];
            const int64_t expected_fc_in = (int64_t)out.n_target_layers * out.n_embd;
            if (derived_fc_in != expected_fc_in) {
                if (out.n_embd > 0 && derived_fc_in % out.n_embd == 0) {
                    const int derived_layers = (int)(derived_fc_in / out.n_embd);
                    std::fprintf(stderr,
                        "[draft] dflash.n_target_layers metadata (%d) != "
                        "fc-derived capture count (%d); using the weights\n",
                        out.n_target_layers, derived_layers);
                    out.n_target_layers = derived_layers;
                } else {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "GGUF shape mismatch: dflash.fc.weight->ne[0]=%lld "
                        "!= n_target_layers*n_embd=%d*%d=%lld",
                        (long long)derived_fc_in,
                        out.n_target_layers, out.n_embd, (long long)expected_fc_in);
                    set_last_error(buf);
                    return false;
                }
            }
        }
    }

    char summary[192];
    std::snprintf(summary, sizeof(summary),
        "draft GGUF loaded: %" PRId64 " tensors, %.2f GiB on GPU",
        n_tensors, total / (1024.0 * 1024.0 * 1024.0));
    set_last_error(summary);

    return true;
}

} // namespace dflash::common
