// Builds ggml compute graphs for the Gemma4 DFlash draft model
// (5-layer block-diffusion model with KV cache and logit softcapping).
//
// Architecture:
//   - 6 captured target layers  (Qwen3 used 5)
//   - FC input = 6 * target_hidden, where target_hidden = 4096 for all Gemma4
//     variants (31B dense and 26B-A4B MoE), giving FC width = 24576
//   - Logit softcapping: tanh(logits / cap) * cap, cap = 30.0
//   - Tied lm_head: uses tok_embd transposed (or a provided lm_head weight)
//   - Vocab = 262144
//   - Draft has its own lm_head + softcap — it does NOT rely on the target's
//     lm_head (unlike the Qwen3 draft which shares the target's projection)
//   - KV cache (prefix-direct): target features are projected into per-layer
//     K/V entries and stored in GemmaTargetCache::draft_k/draft_v.
//     build_draft_kv_prefill_graph materializes the context K/V;
//     build_gemma4_draft_graph writes block K/V and attends over the full cache.
//   - Layer types: 4 SWA (sliding_attention) + 1 full attention
//     The attention kernel itself is the same ggml_flash_attn_ext call in both
//     cases; the caller controls the mask to implement the sliding window.
//
// Two-step per-decode:
//   1. build_draft_kv_prefill_graph: project new committed context tokens into
//      draft KV cache (side-effect only; nullptr returned).
//   2. build_gemma4_draft_graph: attend over context+block K/V and return logits.
//
// build_gemma4_draft_graph takes:
//   - draft_embed   [draft_hidden, n_tokens] f32  (MASK token embeddings)
//   - positions     [n_tokens]               i32  (absolute token positions)
//   - attn_mask     [kv_pad, q_pad]          f16  (causal over context+block)
//   - kv_start      = cache.draft_kv_pos (context length before this block)
// and returns:
//   - logits        [n_vocab, n_tokens]      f32  (after softcapping)
//
// Safetensors tensor naming (actual file, no model. prefix):
//   fc.weight                                           → fc
//   hidden_norm.weight                                  → hidden_norm
//   norm.weight                                         → out_norm
//   layers.{i}.self_attn.q_proj.weight                  → wq
//   layers.{i}.self_attn.k_proj.weight                  → wk
//   layers.{i}.self_attn.v_proj.weight                  → wv
//   layers.{i}.self_attn.o_proj.weight                  → wo
//   layers.{i}.self_attn.q_norm.weight                  → q_norm
//   layers.{i}.self_attn.k_norm.weight                  → k_norm
//   layers.{i}.input_layernorm.weight                   → attn_norm
//   layers.{i}.post_attention_layernorm.weight          → ffn_norm
//   layers.{i}.mlp.gate_proj.weight                     → w_gate
//   layers.{i}.mlp.up_proj.weight                       → w_up
//   layers.{i}.mlp.down_proj.weight                     → w_down
//   (no embed_tokens — tok_embd is injected from the target at runtime)

#include "internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  if !defined(WIN32_LEAN_AND_MEAN)
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <cerrno>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace dflash27b {

// ─── Draft SWA truncation toggle ──────────────────────────────────────────
// Set DFLASH_DRAFT_SWA_TRUNC=1 to enable per-layer K/V truncation in the
// draft graph for SWA layers (last n-1 layers — the final layer is full).
// Mirrors PR #129 for the qwen3 drafter, ported to gemma4's cached layout.
static inline bool draft_swa_trunc_enabled() {
    static int e = -1;
    if (e < 0) {
        const char * v = std::getenv("DFLASH_DRAFT_SWA_TRUNC");
        e = (v && std::atoi(v) != 0) ? 1 : 0;
        if (e) {
            std::fprintf(stderr, "[draft-swa-trunc] enabled\n");
        }
    }
    return e == 1;
}

// ─── Draft RoPE wrapper with optional YaRN extrapolation ──────────────────
// Set DFLASH_DRAFT_YARN=1 to enable YaRN scaling for draft RoPE; assumes the
// draft was effectively trained at DFLASH_DRAFT_YARN_NCTX_ORIG (default 32768)
// despite config.json claiming a larger max_position_embeddings.
static inline ggml_tensor * draft_rope(ggml_context * ctx, ggml_tensor * x,
                                       ggml_tensor * positions, int head_dim,
                                       float rope_base) {
    static struct {
        int   nctx;
        float ext;
        float bf;
        float bs;
        bool  init;
    } p = {0, 0.0f, 0.0f, 0.0f, false};
    if (!p.init) {
        const char * en = std::getenv("DFLASH_DRAFT_YARN");
        if (en && std::atoi(en) != 0) {
            const char * nc = std::getenv("DFLASH_DRAFT_YARN_NCTX_ORIG");
            p.nctx = nc ? std::atoi(nc) : 32768;
            p.ext  = 1.0f;
            p.bf   = 32.0f;
            p.bs   = 1.0f;
            std::fprintf(stderr,
                "[draft-yarn] enabled: n_ctx_orig=%d ext_factor=%.2f beta_fast=%.1f beta_slow=%.1f\n",
                p.nctx, p.ext, p.bf, p.bs);
        }
        p.init = true;
    }
    return ggml_rope_ext(ctx, x, positions, /*freq_factors=*/nullptr,
                        head_dim, GGML_ROPE_TYPE_NEOX, p.nctx,
                        rope_base, /*freq_scale=*/1.0f,
                        p.ext, /*attn_factor=*/1.0f, p.bf, p.bs);
}

// ─── Graph builders ───────────────────────────────────────────────────────

// build_draft_kv_prefill_graph — prefix-direct KV materialisation (SGLang style).
//
// Projects n_tokens new context positions through the draft model's Wk / Wv
// (after FC → ctx_hidden) and writes the resulting K, V tensors into
// cache.draft_k[il] / cache.draft_v[il] starting at offset cache.draft_kv_pos.
//
// The function is side-effect only: it expands ggml_cpy ops into gf and
// returns nullptr.  The caller must ggml_graph_compute(gf) to materialise
// the cache entries, then increment cache.draft_kv_pos by n_tokens.
//
//   target_feat  [6*target_hidden, n_tokens] f32
//   positions    [n_tokens]                 i32   (absolute positions for RoPE)
ggml_tensor * build_draft_kv_prefill_graph(
    ggml_context *            ctx,
    ggml_cgraph *             gf,
    const GemmaDraftWeights & w,
    GemmaTargetCache &        cache,
    ggml_tensor *             target_feat,
    ggml_tensor *             positions,
    int                       n_tokens)
{
    // Guard: writing cache.draft_kv_pos..cache.draft_kv_pos+n_tokens-1 must fit.
    if (cache.draft_k.empty() ||
        cache.draft_kv_pos < 0 ||
        cache.draft_kv_pos + n_tokens > (int)cache.draft_k[0]->ne[2]) {
        const int tensor_cap = cache.draft_k.empty() ? -1 : (int)cache.draft_k[0]->ne[2];
        GGML_ABORT("draft KV prefill out of bounds: draft_kv_pos=%d n_tokens=%d cap=%d tensor_cap=%d",
                   cache.draft_kv_pos, n_tokens, cache.draft_kv_cap, tensor_cap);
    }

    const int n_kv     = w.n_head_kv;
    const int head_dim = w.head_dim;
    const float eps       = GEMMA4_RMS_EPS;
    const float rope_base = w.rope_theta;

    // ── 1. FC projection: ctx_hidden = fc @ target_feat  →  [n_embd, n_tokens]
    ggml_tensor * ctx_hidden = ggml_mul_mat(ctx, w.fc, target_feat);
    // hidden_norm: RMSNorm applied right after the fc projection
    // (matches qwen3_dflash_graph.cpp:57-59)
    ctx_hidden = ggml_rms_norm(ctx, ctx_hidden, eps);
    ctx_hidden = ggml_mul(ctx, ctx_hidden, w.hidden_norm);
    ggml_set_name(ctx_hidden, "draft_kv_prefill_ctx_hidden");

    // ── 2. Per-layer K / V projection, normalisation, RoPE, cache write
    for (int il = 0; il < w.n_layer; il++) {
        const GemmaDraftLayer & L = w.layers[il];

        // K = Wk @ ctx_hidden → [kv_dim, n_tokens] → [head_dim, n_kv, n_tokens]
        ggml_tensor * Kb = ggml_mul_mat(ctx, L.wk, ctx_hidden);
        Kb = ggml_reshape_3d(ctx, Kb, head_dim, n_kv, n_tokens);
        Kb = ggml_rms_norm(ctx, Kb, eps);
        Kb = ggml_mul(ctx, Kb, L.k_norm);
        Kb = draft_rope(ctx, Kb, positions, head_dim, rope_base);

        // V = Wv @ ctx_hidden → [kv_dim, n_tokens] → [head_dim, n_kv, n_tokens]
        ggml_tensor * Vb = ggml_mul_mat(ctx, L.wv, ctx_hidden);
        Vb = ggml_reshape_3d(ctx, Vb, head_dim, n_kv, n_tokens);

        // Write K into cache.draft_k[il] at offset cache.draft_kv_pos
        ggml_tensor * k_dst = ggml_view_3d(ctx, cache.draft_k[il],
            head_dim, n_kv, n_tokens,
            cache.draft_k[il]->nb[1], cache.draft_k[il]->nb[2],
            (size_t)cache.draft_kv_pos * cache.draft_k[il]->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kb, k_dst));

        // Write V into cache.draft_v[il] at offset cache.draft_kv_pos
        ggml_tensor * v_dst = ggml_view_3d(ctx, cache.draft_v[il],
            head_dim, n_kv, n_tokens,
            cache.draft_v[il]->nb[1], cache.draft_v[il]->nb[2],
            (size_t)cache.draft_kv_pos * cache.draft_v[il]->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vb, v_dst));
    }

    return nullptr;
}

// build_gemma4_draft_graph — KV-cached draft forward.
//
// Attends over the full draft KV cache (context K/V already materialised by
// build_draft_kv_prefill_graph, plus newly written block K/V) and returns
// logits for the n_tokens block positions.
//
//   draft_embed  [n_embd, n_tokens] f32   (MASK token embeddings)
//   positions    [n_tokens]         i32   (absolute token positions)
//   attn_mask    [kv_pad, q_pad]    f16   (causal over context+block)
//   kv_start     context length before this block (= cache.draft_kv_pos)
//
// Returns logits [n_vocab, n_tokens] f32 (softcapped).
ggml_tensor * build_gemma4_draft_graph(
    ggml_context *               ctx,
    ggml_cgraph *                gf,
    const GemmaDraftWeights &    w,
    GemmaTargetCache &           cache,
    ggml_tensor *                draft_embed,
    ggml_tensor *                positions,
    ggml_tensor *                attn_mask,
    int                          n_tokens,
    int                          kv_start)
{
    // Validate KV cache write range before any graph nodes touch it.
    if (kv_start < 0 || kv_start + n_tokens > cache.draft_kv_cap) {
        GGML_ABORT("draft KV write out of bounds: kv_start=%d n_tokens=%d cap=%d",
                   kv_start, n_tokens, cache.draft_kv_cap);
    }

    const int n_head   = w.n_head;
    const int n_kv     = w.n_head_kv;
    const int head_dim = w.head_dim;
    const float eps       = GEMMA4_RMS_EPS;
    const float rope_base = w.rope_theta;
    const int   kv_len    = kv_start + n_tokens;

    // Gemma4 scales embeddings by sqrt(hidden_size) — the draft shares the
    // target's tok_embd, so it must apply the same scaling.  Reference:
    // vLLM qwen3_dflash.py embed_normalizer = target_config.hidden_size**0.5
    ggml_tensor * hidden = ggml_scale(ctx, draft_embed, std::sqrt((float)w.n_embd));
    ggml_set_name(hidden, "gemma4_draft_scaled_embed");

    // ── 2. Transformer layers ─────────────────────────────────────────
    for (int il = 0; il < w.n_layer; il++) {
        const GemmaDraftLayer & L = w.layers[il];

        // ── 2a. Attention pre-norm
        ggml_tensor * cur = ggml_rms_norm(ctx, hidden, eps);
        cur = ggml_mul(ctx, cur, L.attn_norm);

        // ── 2b. Q / K / V projections from block hidden state
        ggml_tensor * Q  = ggml_mul_mat(ctx, L.wq, cur);  // [q_dim,  n_tokens]
        ggml_tensor * Kb = ggml_mul_mat(ctx, L.wk, cur);  // [kv_dim, n_tokens]
        ggml_tensor * Vb = ggml_mul_mat(ctx, L.wv, cur);  // [kv_dim, n_tokens]

        // ── 2c. Reshape + per-head RMSNorm for Q and block K
        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, n_tokens);
        Q = ggml_rms_norm(ctx, Q, eps);
        Q = ggml_mul(ctx, Q, L.q_norm);

        Kb = ggml_reshape_3d(ctx, Kb, head_dim, n_kv, n_tokens);
        Kb = ggml_rms_norm(ctx, Kb, eps);
        Kb = ggml_mul(ctx, Kb, L.k_norm);

        Vb = ggml_reshape_3d(ctx, Vb, head_dim, n_kv, n_tokens);

        // ── 2d. RoPE on Q and block K
        Q  = draft_rope(ctx, Q,  positions, head_dim, rope_base);
        Kb = draft_rope(ctx, Kb, positions, head_dim, rope_base);

        // ── 2e. Write block K / V into draft KV cache at [kv_start..kv_start+n_tokens)
        ggml_tensor * k_dst = ggml_view_3d(ctx, cache.draft_k[il],
            head_dim, n_kv, n_tokens,
            cache.draft_k[il]->nb[1], cache.draft_k[il]->nb[2],
            (size_t)kv_start * cache.draft_k[il]->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kb, k_dst));

        ggml_tensor * v_dst = ggml_view_3d(ctx, cache.draft_v[il],
            head_dim, n_kv, n_tokens,
            cache.draft_v[il]->nb[1], cache.draft_v[il]->nb[2],
            (size_t)kv_start * cache.draft_v[il]->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vb, v_dst));

        // ── 2f. Full K / V view (context + block) from draft KV cache
        // Optional SWA truncation: when enabled and this is an SWA layer
        // with kv_len exceeding sliding_window, restrict K/V (and the mask)
        // to the last (sliding_window + n_tokens) slots. Matches the draft
        // model's training-time SWA pattern.
        const bool layer_is_swa = (il < (int)w.layer_is_swa.size())
                                      ? w.layer_is_swa[il] : false;
        const bool use_swa_trunc = draft_swa_trunc_enabled()
                                       && layer_is_swa
                                       && w.sliding_window > 0
                                       && kv_len > (w.sliding_window + n_tokens);
        const int eff_kv_len = use_swa_trunc
                                   ? (w.sliding_window + n_tokens)
                                   : kv_len;
        const int kv_offset  = kv_len - eff_kv_len;  // 0 if no truncation

        ggml_tensor * K_full = ggml_view_3d(ctx, cache.draft_k[il],
            head_dim, n_kv, eff_kv_len,
            cache.draft_k[il]->nb[1], cache.draft_k[il]->nb[2],
            (size_t)kv_offset * cache.draft_k[il]->nb[2]);
        ggml_tensor * V_full = ggml_view_3d(ctx, cache.draft_v[il],
            head_dim, n_kv, eff_kv_len,
            cache.draft_v[il]->nb[1], cache.draft_v[il]->nb[2],
            (size_t)kv_offset * cache.draft_v[il]->nb[2]);

        // ── 2g. Permute into flash_attn_ext layout
        //   Q:      [head_dim, n_tokens,    n_head,    1]
        //   K_full: [head_dim, eff_kv_len,  n_head_kv, 1]
        //   V_full: [head_dim, eff_kv_len,  n_head_kv, 1]
        Q      = ggml_cont(ctx, ggml_permute(ctx, Q,      0, 2, 1, 3));
        K_full = ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3));
        V_full = ggml_cont(ctx, ggml_permute(ctx, V_full, 0, 2, 1, 3));

        // SWA-truncated mask view: take the last eff_kv_len rows along the
        // kv axis (axis 0). Mask shape is [kv_pad, q_pad] with kv_pad >= kv_len,
        // so the slice [kv_offset .. kv_offset+eff_kv_len) gives the same
        // causal pattern for the surviving K positions.
        ggml_tensor * eff_mask = attn_mask;
        if (use_swa_trunc && kv_offset > 0) {
            // ggml_view_2d would produce a non-contiguous tensor (row stride is
            // unchanged at kv_pad * elt). FA requires contiguous mask, so we
            // copy the slice into a fresh tensor.
            ggml_tensor * mask_view = ggml_view_2d(ctx, attn_mask,
                eff_kv_len, attn_mask->ne[1],
                attn_mask->nb[1],
                (size_t)kv_offset * ggml_element_size(attn_mask));
            eff_mask = ggml_cont(ctx, mask_view);
        }

        // ── 2h. Flash attention over full context+block KV
        //   scale = 1 / sqrt(head_dim); no logit softcap at attention level
        const float scale = 1.0f / std::sqrt((float)head_dim);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K_full, V_full, eff_mask,
                                                  scale, /*max_bias=*/0.0f,
                                                  /*logit_softcap=*/0.0f);
        // attn: [head_dim, n_head, n_tokens, 1]
        attn = ggml_reshape_2d(ctx, attn, head_dim * n_head, n_tokens);

        // ── 2i. Output projection + residual
        ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn);
        hidden = ggml_add(ctx, hidden, attn_out);

        // ── 2j. FFN pre-norm
        ggml_tensor * hf = ggml_rms_norm(ctx, hidden, eps);
        hf = ggml_mul(ctx, hf, L.ffn_norm);

        // ── 2k. SwiGLU FFN: down(silu(gate(x)) * up(x))
        ggml_tensor * g  = ggml_mul_mat(ctx, L.w_gate, hf);
        g = ggml_silu(ctx, g);
        ggml_tensor * u  = ggml_mul_mat(ctx, L.w_up, hf);
        ggml_tensor * gu = ggml_mul(ctx, g, u);
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, L.w_down, gu);

        hidden = ggml_add(ctx, hidden, ffn_out);
    }

    // ── 3. Final output norm
    ggml_tensor * out = ggml_rms_norm(ctx, hidden, eps);
    out = ggml_mul(ctx, out, w.out_norm);
    ggml_set_name(out, "gemma4_draft_hidden_out");

    // ── 4. LM head (tied: transpose of tok_embd)
    //   tok_embd: [draft_hidden, n_vocab]  ggml ne[0]=draft_hidden, ne[1]=n_vocab
    //   out:      [draft_hidden, n_tokens]
    //   logits:   [n_vocab, n_tokens]
    ggml_tensor * logits = ggml_mul_mat(ctx, w.tok_embd, out);
    ggml_set_name(logits, "gemma4_draft_logits_pre_cap");

    // ── 5. Logit softcapping: logits = cap * tanh(logits / cap)
    const float cap = w.logit_softcap;
    logits = ggml_scale(ctx, logits, 1.0f / cap);
    logits = ggml_tanh(ctx, logits);
    logits = ggml_scale(ctx, logits, cap);
    ggml_set_name(logits, "gemma4_draft_logits");

    return logits;
}

// ─── Safetensors loader ───────────────────────────────────────────────────

namespace {

struct GStEntry {
    std::string          dtype;
    std::vector<int64_t> shape;
    uint64_t             data_start;
    uint64_t             data_end;
};

using GStMap = std::unordered_map<std::string, GStEntry>;

// Minimal safetensors JSON header parser (same algorithm as safetensors_draft.cpp).
static bool parse_gst_header(const char * h, size_t hlen, GStMap & out) {
    auto skip_ws = [&](size_t & i) {
        while (i < hlen && (h[i] == ' ' || h[i] == '\t' ||
                            h[i] == '\n' || h[i] == '\r')) i++;
    };
    size_t i = 0;
    skip_ws(i);
    if (i >= hlen || h[i] != '{') return false;
    i++;
    while (i < hlen) {
        skip_ws(i);
        if (i >= hlen) return false;
        if (h[i] == '}') { i++; break; }
        if (h[i] == ',') { i++; skip_ws(i); }
        if (i >= hlen || h[i] != '"') return false;
        i++;
        size_t name_start = i;
        while (i < hlen && h[i] != '"') i++;
        if (i >= hlen) return false;
        std::string name(h + name_start, i - name_start);
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != ':') return false;
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != '{') return false;
        size_t obj_start = i;
        int depth = 0;
        size_t obj_end = i;
        for (; obj_end < hlen; obj_end++) {
            if      (h[obj_end] == '{') depth++;
            else if (h[obj_end] == '}') { if (--depth == 0) { obj_end++; break; } }
        }
        if (depth != 0) return false;
        if (name == "__metadata__") { i = obj_end; continue; }

        std::string obj(h + obj_start, obj_end - obj_start);
        GStEntry e;
        {
            auto k = obj.find("\"dtype\":\"");
            if (k == std::string::npos) return false;
            auto vs = k + 9;
            auto ve = obj.find('"', vs);
            if (ve == std::string::npos) return false;
            e.dtype = obj.substr(vs, ve - vs);
        }
        {
            auto k = obj.find("\"shape\":[");
            if (k == std::string::npos) return false;
            auto vs = k + 9;
            auto ve = obj.find(']', vs);
            if (ve == std::string::npos) return false;
            const char * p  = obj.c_str() + vs;
            const char * pe = obj.c_str() + ve;
            while (p < pe) {
                char * end = nullptr;
                long long v = std::strtoll(p, &end, 10);
                if (end == p) break;
                e.shape.push_back((int64_t)v);
                p = end;
                while (p < pe && (*p == ',' || *p == ' ')) p++;
            }
        }
        {
            auto k = obj.find("\"data_offsets\":[");
            if (k == std::string::npos) return false;
            auto vs = k + 16;
            auto ve = obj.find(']', vs);
            if (ve == std::string::npos) return false;
            unsigned long long s = 0, ed = 0;
            if (std::sscanf(obj.c_str() + vs, "%llu , %llu", &s, &ed) != 2)
                if (std::sscanf(obj.c_str() + vs, "%llu,%llu", &s, &ed) != 2) return false;
            e.data_start = s;
            e.data_end   = ed;
        }
        out.emplace(std::move(name), std::move(e));
        i = obj_end;
    }
    return true;
}

static ggml_type gst_dtype_to_ggml(const std::string & dt) {
    if (dt == "BF16") return GGML_TYPE_BF16;
    if (dt == "F16")  return GGML_TYPE_F16;
    if (dt == "F32")  return GGML_TYPE_F32;
    return GGML_TYPE_COUNT;
}

struct GMmap {
    void * addr = nullptr;
    size_t len  = 0;
#if defined(_WIN32)
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap  = nullptr;
#else
    int fd = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx failed"; return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) { err = "CreateFileMappingA failed"; return false; }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) { err = "MapViewOfFile failed"; return false; }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + ": " + std::strerror(errno); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat: " + std::string(std::strerror(errno)); return false; }
        len  = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            err  = "mmap: " + std::string(std::strerror(errno));
            addr = nullptr; return false;
        }
#endif
        return true;
    }

    ~GMmap() {
#if defined(_WIN32)
        if (addr)                             UnmapViewOfFile(addr);
        if (hMap)                             CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE)    CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

// Allocate one ggml tensor for a safetensors entry.
// HF row-major [out, in] → ggml ne[0]=in, ne[1]=out (byte layout identical).
// norm weights are kept as F32 (ggml CUDA elementwise ops require non-BF16 src1).
// Projection weights stay BF16 (Ampere+) or are converted to F16 (Turing).
static ggml_tensor * galloc_tensor(
    ggml_context *               gctx,
    const GStMap &               st,
    const std::string &          name,
    const std::vector<int64_t> & expected_shape,
    ggml_type                    gt_override = GGML_TYPE_COUNT)
{
    auto it = st.find(name);
    if (it == st.end()) {
        set_last_error("gemma4 safetensors: missing tensor '" + name + "'");
        return nullptr;
    }
    const GStEntry & e = it->second;
    if (e.dtype != "BF16") {
        set_last_error("gemma4 safetensors: '" + name + "' dtype=" + e.dtype +
                       " expected BF16");
        return nullptr;
    }
    if (e.shape.size() != expected_shape.size()) {
        set_last_error("gemma4 safetensors: '" + name + "' ndim mismatch");
        return nullptr;
    }
    for (size_t k = 0; k < expected_shape.size(); k++) {
        if (e.shape[k] != expected_shape[k]) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "gemma4 safetensors: '%s' shape[%zu]=%lld expected %lld",
                name.c_str(), k, (long long)e.shape[k], (long long)expected_shape[k]);
            set_last_error(buf);
            return nullptr;
        }
    }
    ggml_type gt = (gt_override == GGML_TYPE_COUNT) ? GGML_TYPE_BF16 : gt_override;
    ggml_tensor * t = nullptr;
    if (expected_shape.size() == 1) {
        t = ggml_new_tensor_1d(gctx, gt, expected_shape[0]);
    } else if (expected_shape.size() == 2) {
        // [out, in] → ne[0]=in, ne[1]=out
        t = ggml_new_tensor_2d(gctx, gt, expected_shape[1], expected_shape[0]);
    } else {
        set_last_error("gemma4 safetensors: unexpected ndim > 2 for '" + name + "'");
        return nullptr;
    }
    ggml_set_name(t, name.c_str());
    return t;
}

static void g_bf16_to_f32(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        std::memcpy(&dst[i], &bits, 4);
    }
}

static void g_bf16_to_f16(const uint16_t * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        uint32_t sign = (u >> 16) & 0x8000;
        int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x03FF;
        if      (exp <= 0)  dst[i] = (uint16_t)sign;
        else if (exp >= 31) dst[i] = (uint16_t)(sign | 0x7C00);
        else                dst[i] = (uint16_t)(sign | (exp << 10) | mant);
    }
}

static bool g_cuda_has_native_bf16() {
    const char * env = std::getenv("DFLASH27B_DRAFT_FP16");
    if (env && std::atoi(env) != 0) return false;
#if defined(DFLASH27B_MIN_SM) && DFLASH27B_MIN_SM < 80
    return false;
#else
    return true;
#endif
}

static uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

static float get_f32_or(const gguf_context * g, const char * key, float fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_f32(g, id);
}

} // anonymous namespace

// ─── Public loader ────────────────────────────────────────────────────────

// Load Gemma4 DFlash draft weights from a directory containing one or more
// safetensors shards.  We look for files named:
//   model.safetensors           (single-shard)
//   model-00001-of-NNNNN.safetensors  (multi-shard, first shard only for now)
//
// In practice the z-lab Gemma4 draft is small enough to fit in a single shard.
bool load_gemma4_draft_safetensors(const std::string & dir_path,
                                    ggml_backend_t       backend,
                                    GemmaDraftWeights &  out)
{
    // ── 1. Find the shard file ────────────────────────────────────────
    // Try the canonical single-shard name first.
    std::string path = dir_path + "/model.safetensors";
    {
        // Quick existence check without mmap
        int fd_check = ::open(path.c_str(), O_RDONLY);
        if (fd_check < 0) {
            // Fall back to first numbered shard
            path = dir_path + "/model-00001-of-00001.safetensors";
            fd_check = ::open(path.c_str(), O_RDONLY);
            if (fd_check < 0) {
                set_last_error("gemma4 draft: no safetensors file found in " + dir_path);
                return false;
            }
        }
        ::close(fd_check);
    }

    // ── 2. Open + mmap ───────────────────────────────────────────────
    GMmap mm;
    std::string err;
    if (!mm.open_ro(path, err)) { set_last_error(err); return false; }
    if (mm.len < 8) { set_last_error("gemma4 draft: safetensors file too small"); return false; }

    // ── 3. Parse header ──────────────────────────────────────────────
    uint64_t header_len = 0;
    std::memcpy(&header_len, mm.addr, 8);
    if (header_len == 0 || 8 + header_len > mm.len) {
        set_last_error("gemma4 draft: bad safetensors header length");
        return false;
    }
    const char * header_ptr = (const char *)mm.addr + 8;
    GStMap st;
    if (!parse_gst_header(header_ptr, (size_t)header_len, st)) {
        set_last_error("gemma4 draft: safetensors JSON parse failed");
        return false;
    }
    const uint8_t * blob      = (const uint8_t *)mm.addr + 8 + header_len;
    const size_t    blob_size = mm.len - 8 - (size_t)header_len;

    // ── 4. Infer draft dimensions from FC weight shape ───────────────
    //   fc: [n_vocab_or_target_feat_in, draft_hidden]
    //   The FC input is 6*target_hidden; FC output is draft_hidden.
    //   HF shape in safetensors: [draft_hidden, 6*target_hidden]
    {
        auto it = st.find("fc.weight");
        if (it == st.end()) {
            set_last_error("gemma4 draft: fc.weight not found");
            return false;
        }
        const GStEntry & e = it->second;
        if (e.shape.size() != 2) {
            set_last_error("gemma4 draft: model.fc.weight expected 2D");
            return false;
        }
        // HF stores as [out_features, in_features] = [draft_hidden, 6*target_hidden]
        out.n_embd        = (int)e.shape[0];
        int fc_in         = (int)e.shape[1];
        out.target_hidden = fc_in / GEMMA4_DRAFT_N_TARGET_LAYERS;
        if (fc_in % GEMMA4_DRAFT_N_TARGET_LAYERS != 0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "gemma4 draft: FC input %d not divisible by n_target_layers %d",
                fc_in, GEMMA4_DRAFT_N_TARGET_LAYERS);
            set_last_error(buf);
            return false;
        }
    }

    // Infer n_head / n_head_kv / n_ff from layer 0 weight shapes
    {
        auto iq = st.find("layers.0.self_attn.q_proj.weight");
        auto ik = st.find("layers.0.self_attn.k_proj.weight");
        auto ig = st.find("layers.0.mlp.gate_proj.weight");
        if (iq == st.end() || ik == st.end() || ig == st.end()) {
            set_last_error("gemma4 draft: missing required layer-0 weight tensors");
            return false;
        }
        // q_proj HF shape: [q_dim, n_embd] where q_dim = n_head * head_dim
        int q_dim = (int)iq->second.shape[0];
        int kv_dim = (int)ik->second.shape[0];
        out.n_head    = q_dim  / out.head_dim;
        out.n_head_kv = kv_dim / out.head_dim;
        out.n_ff      = (int)ig->second.shape[0];
        // Also set layer_is_swa: layers [0..n_layer-2] are SWA, last is full
        out.layer_is_swa.assign((size_t)out.n_layer, true);
        out.layer_is_swa[(size_t)(out.n_layer - 1)] = false;
    }

    const int64_t HIDDEN  = out.n_embd;
    const int64_t Q_DIM   = (int64_t)out.n_head    * out.head_dim;
    const int64_t KV_DIM  = (int64_t)out.n_head_kv * out.head_dim;
    const int64_t INTER   = out.n_ff;
    const int64_t HD      = out.head_dim;
    const int64_t FC_IN   = (int64_t)GEMMA4_DRAFT_N_TARGET_LAYERS * out.target_hidden;
    // VOCAB not used here; tok_embd is injected at runtime from the target model.

    // ── 5. Allocate ggml context ─────────────────────────────────────
    //   tensors: fc, hidden_norm, out_norm = 3 top-level (tok_embd injected at runtime)
    //            11 tensors × 5 layers = 55
    //   total = 58 + headroom
    const int n_tensors = 3 + 11 * out.n_layer + 8;
    ggml_init_params ip{};
    ip.mem_size   = (size_t)n_tensors * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("gemma4 draft: ggml_init failed"); return false; }
    out.backend = backend;
    out.layers.assign((size_t)out.n_layer, GemmaDraftLayer{});

    const ggml_type NORM_GT = GGML_TYPE_F32;
    const bool      nbf16   = g_cuda_has_native_bf16();
    const ggml_type PROJ_GT = nbf16 ? GGML_TYPE_COUNT : GGML_TYPE_F16;

    // ── 6. Create named tensors ──────────────────────────────────────
    out.fc          = galloc_tensor(out.ctx, st, "fc.weight",          {HIDDEN, FC_IN}, PROJ_GT);
    out.hidden_norm = galloc_tensor(out.ctx, st, "hidden_norm.weight", {HIDDEN},        NORM_GT);
    out.out_norm    = galloc_tensor(out.ctx, st, "norm.weight",        {HIDDEN},        NORM_GT);
    // tok_embd is not present in the draft safetensors; the draft shares
    // the target model's token embedding which is injected at runtime.
    out.tok_embd    = nullptr;
    if (!out.fc || !out.hidden_norm || !out.out_norm) return false;

    for (int il = 0; il < out.n_layer; il++) {
        char pfx[64];
        std::snprintf(pfx, sizeof(pfx), "layers.%d.", il);
        std::string p = pfx;
        GemmaDraftLayer & L = out.layers[(size_t)il];

        L.attn_norm = galloc_tensor(out.ctx, st, p + "input_layernorm.weight",          {HIDDEN},       NORM_GT);
        L.ffn_norm  = galloc_tensor(out.ctx, st, p + "post_attention_layernorm.weight", {HIDDEN},       NORM_GT);
        L.wq        = galloc_tensor(out.ctx, st, p + "self_attn.q_proj.weight",         {Q_DIM,  HIDDEN}, PROJ_GT);
        L.wk        = galloc_tensor(out.ctx, st, p + "self_attn.k_proj.weight",         {KV_DIM, HIDDEN}, PROJ_GT);
        L.wv        = galloc_tensor(out.ctx, st, p + "self_attn.v_proj.weight",         {KV_DIM, HIDDEN}, PROJ_GT);
        L.wo        = galloc_tensor(out.ctx, st, p + "self_attn.o_proj.weight",         {HIDDEN, Q_DIM},  PROJ_GT);
        L.q_norm    = galloc_tensor(out.ctx, st, p + "self_attn.q_norm.weight",         {HD},             NORM_GT);
        L.k_norm    = galloc_tensor(out.ctx, st, p + "self_attn.k_norm.weight",         {HD},             NORM_GT);
        L.w_gate    = galloc_tensor(out.ctx, st, p + "mlp.gate_proj.weight",            {INTER,  HIDDEN}, PROJ_GT);
        L.w_up      = galloc_tensor(out.ctx, st, p + "mlp.up_proj.weight",              {INTER,  HIDDEN}, PROJ_GT);
        L.w_down    = galloc_tensor(out.ctx, st, p + "mlp.down_proj.weight",            {HIDDEN, INTER},  PROJ_GT);

        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            return false;
        }
    }

    // ── 7. Allocate backend buffer and upload bytes ──────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("gemma4 draft: ggml_backend_alloc_ctx_tensors failed");
        return false;
    }

    std::vector<float>    scratch_f32;
    std::vector<uint16_t> scratch_f16;

    for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t != nullptr;
         t = ggml_get_next_tensor(out.ctx, t))
    {
        const char * name = ggml_get_name(t);
        auto it = st.find(name);
        if (it == st.end()) {
            set_last_error(std::string("gemma4 draft post-alloc: '") +
                           name + "' vanished from header");
            return false;
        }
        const GStEntry & e = it->second;
        if (e.data_end > (uint64_t)blob_size) {
            set_last_error(std::string("gemma4 draft: data offset out of bounds for '") +
                           name + "'");
            return false;
        }
        const size_t src_bytes = (size_t)(e.data_end - e.data_start);
        const size_t dst_bytes = ggml_nbytes(t);
        const bool same = (t->type == gst_dtype_to_ggml(e.dtype));

        if (same) {
            if (src_bytes != dst_bytes) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "gemma4 draft: byte mismatch for '%s': blob=%zu ggml=%zu",
                    name, src_bytes, dst_bytes);
                set_last_error(buf);
                return false;
            }
            ggml_backend_tensor_set(t, blob + e.data_start, 0, dst_bytes);
        } else if (e.dtype == "BF16" && t->type == GGML_TYPE_F32) {
            const size_t n = ggml_nelements(t);
            if (src_bytes != n * 2 || dst_bytes != n * 4) {
                set_last_error(std::string("gemma4 draft: BF16->F32 size mismatch for '") + name + "'");
                return false;
            }
            scratch_f32.resize(n);
            g_bf16_to_f32((const uint16_t *)(blob + e.data_start),
                          scratch_f32.data(), n);
            ggml_backend_tensor_set(t, scratch_f32.data(), 0, dst_bytes);
        } else if (e.dtype == "BF16" && t->type == GGML_TYPE_F16) {
            const size_t n = ggml_nelements(t);
            if (src_bytes != n * 2 || dst_bytes != n * 2) {
                set_last_error(std::string("gemma4 draft: BF16->F16 size mismatch for '") + name + "'");
                return false;
            }
            scratch_f16.resize(n);
            g_bf16_to_f16((const uint16_t *)(blob + e.data_start),
                          scratch_f16.data(), n);
            ggml_backend_tensor_set(t, scratch_f16.data(), 0, dst_bytes);
        } else {
            set_last_error(std::string("gemma4 draft: unsupported dtype conversion for '") +
                           name + "': " + e.dtype + " -> " + ggml_type_name(t->type));
            return false;
        }
    }

    std::fprintf(stderr,
        "[gemma4 draft] loaded: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d target_hidden=%d vocab=%d\n",
        out.n_layer, out.n_head, out.n_head_kv,
        out.n_embd, out.n_ff, out.head_dim, out.target_hidden, out.n_vocab);
    std::fflush(stderr);

    return true;
}

bool load_gemma4_draft_gguf(const std::string & path,
                            ggml_backend_t       backend,
                            GemmaDraftWeights &  out)
{
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
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("gemma4 draft GGUF: missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "gemma4-dflash-draft") {
            set_last_error(std::string("gemma4 draft GGUF: unexpected arch: ") + arch +
                           " (expected gemma4-dflash-draft)");
            gguf_free(gctx);
            return false;
        }
    }

    // Read dimensions from GGUF metadata
    int64_t arch_id2 = gguf_find_key(gctx, "general.architecture");
    const char * A   = gguf_get_val_str(gctx, arch_id2);
    char key[256];

    auto read_u32 = [&](const char * suffix, uint32_t fallback) -> uint32_t {
        std::snprintf(key, sizeof(key), "%s.%s", A, suffix);
        return get_u32_or(gctx, key, fallback);
    };
    auto read_f32 = [&](const char * suffix, float fallback) -> float {
        std::snprintf(key, sizeof(key), "%s.%s", A, suffix);
        return get_f32_or(gctx, key, fallback);
    };

    const uint32_t n_embd       = read_u32("embedding_length",        0);
    const uint32_t n_layer      = read_u32("block_count",             0);
    const uint32_t n_ff         = read_u32("feed_forward_length",     0);
    const uint32_t n_head       = read_u32("attention.head_count",    0);
    const uint32_t n_head_kv    = read_u32("attention.head_count_kv", 0);
    const uint32_t head_dim     = read_u32("attention.key_length",    0);
    const uint32_t block_sz     = read_u32("dflash.block_size",       0);
    const uint32_t n_tgt_lay    = read_u32("dflash.n_target_layers",  0);
    const uint32_t target_hid   = read_u32("dflash.target_hidden",    0);
    const uint32_t mask_tok_id  = read_u32("dflash.mask_token_id",    GEMMA4_31B_DRAFT_MASK_TOKEN_ID);
    const uint32_t sliding_win  = read_u32("dflash.sliding_window",   2048);
    const float    logit_cap    = read_f32("dflash.logit_softcap",    GEMMA4_LOGIT_SOFTCAP);
    const float    rope_theta   = read_f32("rope.freq_base",          GEMMA4_ROPE_THETA);

    if (n_embd == 0 || n_layer == 0 || n_ff == 0 || n_head == 0 ||
        n_head_kv == 0 || head_dim == 0) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "gemma4 draft GGUF: missing hparams: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // Validate block_size and n_target_layers match compiled constants
    if (block_sz != (uint32_t)GEMMA4_DRAFT_BLOCK_SIZE ||
        n_tgt_lay != (uint32_t)GEMMA4_DRAFT_N_TARGET_LAYERS) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "gemma4 draft GGUF: dflash.block_size=%u (expected %d), "
            "dflash.n_target_layers=%u (expected %d)",
            block_sz, GEMMA4_DRAFT_BLOCK_SIZE,
            n_tgt_lay, GEMMA4_DRAFT_N_TARGET_LAYERS);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // Sanity-check upper bounds
    constexpr uint32_t MAX_LAYERS  = 1024;
    constexpr uint32_t MAX_EMBD    = 1u << 17;
    constexpr uint32_t MAX_FF      = 1u << 19;
    constexpr uint32_t MAX_HEADS   = 1024;
    constexpr uint32_t MAX_HEADDIM = 1024;
    if (n_layer   > MAX_LAYERS  || n_embd    > MAX_EMBD  ||
        n_ff      > MAX_FF      || n_head    > MAX_HEADS ||
        n_head_kv > MAX_HEADS   || head_dim  > MAX_HEADDIM ||
        n_head_kv > n_head      || (n_head % n_head_kv) != 0) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "gemma4 draft GGUF: hparams out of range: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // ── 2. Populate GemmaDraftWeights scalars ────────────────────────────
    out.ctx           = meta_ctx;
    out.backend       = backend;
    out.n_layer       = (int)n_layer;
    out.n_head        = (int)n_head;
    out.n_head_kv     = (int)n_head_kv;
    out.head_dim      = (int)head_dim;
    out.n_embd        = (int)n_embd;
    out.n_ff          = (int)n_ff;
    out.block_size    = (int)block_sz;
    out.n_target_layers = (int)n_tgt_lay;
    out.target_hidden = (int)target_hid;
    out.mask_token_id = (int)mask_tok_id;
    out.sliding_window = (int)sliding_win;
    out.logit_softcap = logit_cap;
    out.rope_theta    = rope_theta;

    // layers [0..n_layer-2] are SWA, last layer is full attention
    out.layer_is_swa.assign((size_t)n_layer, true);
    out.layer_is_swa[(size_t)(n_layer - 1)] = false;

    out.layers.assign((size_t)n_layer, GemmaDraftLayer{});

    // tok_embd is injected at runtime from the target model (same as safetensors path)
    out.tok_embd = nullptr;

    // ── 3. Wire tensor pointers ──────────────────────────────────────────
    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };

    out.fc          = g("dflash.fc.weight");
    out.hidden_norm = g("dflash.hidden_norm.weight");
    out.out_norm    = g("output_norm.weight");

    if (!out.fc || !out.hidden_norm || !out.out_norm) {
        set_last_error("gemma4 draft GGUF: missing top-level tensors "
                       "(dflash.fc.weight / dflash.hidden_norm.weight / output_norm.weight)");
        gguf_free(gctx);
        return false;
    }

    for (int il = 0; il < out.n_layer; il++) {
        char name[128];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        GemmaDraftLayer & L = out.layers[il];
        L.attn_norm = fnd("attn_norm.weight");
        L.ffn_norm  = fnd("ffn_norm.weight");
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
            std::snprintf(b, sizeof(b),
                "gemma4 draft GGUF: layer %d missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
    }

    // ── 4. Allocate backend buffer for all tensors ───────────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("gemma4 draft GGUF: ggml_backend_alloc_ctx_tensors failed");
        gguf_free(gctx);
        return false;
    }

    // ── 5. mmap file and copy tensor bytes to backend ────────────────────
    std::string err;
    GMmap mm;
    if (!mm.open_ro(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const size_t  data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors  = gguf_get_n_tensors(gctx);

    size_t total = 0;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("gemma4 draft GGUF: tensor '") +
                           tname + "' overflows file");
            gguf_free(gctx);
            return false;
        }
        ggml_backend_tensor_set(t, (const uint8_t *)mm.addr + off, 0, sz);
        total += sz;
    }

    gguf_free(gctx);

    std::fprintf(stderr,
        "[gemma4 draft GGUF] loaded: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d target_hidden=%d  (%.2f GiB on GPU)\n",
        out.n_layer, out.n_head, out.n_head_kv,
        out.n_embd, out.n_ff, out.head_dim, out.target_hidden,
        total / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stderr);

    return true;
}

void free_gemma4_draft_weights(GemmaDraftWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    w.layers.clear();
    w.layer_is_swa.clear();
    w.fc          = nullptr;
    w.hidden_norm = nullptr;
    w.out_norm    = nullptr;
    w.tok_embd    = nullptr;
}

} // namespace dflash27b
