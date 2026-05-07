// Builds a ggml compute graph for one forward pass of the Gemma4 DFlash draft
// (5-layer block-diffusion model with logit softcapping).
//
// Architecture differences from the Qwen3 DFlash draft:
//   - 6 captured target layers  (Qwen3 used 5)
//   - FC input = 6 * target_hidden, where target_hidden = 4096 for all Gemma4
//     variants (31B dense and 26B-A4B MoE), giving FC width = 24576
//   - Logit softcapping: tanh(logits / cap) * cap, cap = 30.0
//   - Tied lm_head: uses tok_embd transposed (or a provided lm_head weight)
//   - Vocab = 262144
//   - Draft has its own lm_head + softcap — it does NOT rely on the target's
//     lm_head (unlike the Qwen3 draft which shares the target's projection)
//   - Attention: pure self-attention over fused hidden states
//       Q/K/V all come from the per-layer hidden state (no cross-attention concat)
//       Block-causal mask passed by the caller (shape [n_tokens, n_tokens])
//   - Layer types: 4 SWA (sliding_attention) + 1 full attention
//     The attention kernel itself is the same ggml_flash_attn_ext call in both
//     cases; the caller controls the mask to implement the sliding window.
//
// Stateless: no KV cache. Each call takes:
//   - target_feat   [6*target_hidden, n_tokens] f32   (6 captured target layers)
//   - draft_embed   [draft_hidden,    n_tokens] f32   (current draft token embeddings)
//   - positions     [n_tokens]                 i32   (absolute token positions)
//   - attn_mask     [n_tokens, n_tokens]        f32   (block-causal; nullptr ok)
// and returns:
//   - logits        [n_vocab, n_tokens]         f32   (after softcapping)
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

// ─── Graph builder ────────────────────────────────────────────────────────

// Build the Gemma4 draft model compute graph for one diffusion refinement step.
//
//   target_feat   [6*target_hidden, n_tokens] f32
//   draft_embed   [draft_hidden,    n_tokens] f32   (embeddings of current draft tokens)
//   positions     [n_tokens]                 i32
//   attn_mask     [n_tokens, n_tokens]        f32   (block-causal, nullable)
//   n_tokens      number of tokens in the block (= block_size = 16 during decode)
//
// Returns the logits tensor [n_vocab, n_tokens] f32 (softcapped).
// The returned tensor is the graph output; the caller must ggml_graph_compute().
ggml_tensor * build_gemma4_draft_graph(
    ggml_context *               ctx,
    ggml_cgraph *                gf,
    const GemmaDraftWeights &    w,
    ggml_tensor *                target_feat,
    ggml_tensor *                draft_embed,
    ggml_tensor *                positions,
    ggml_tensor *                attn_mask,
    int                          n_tokens)
{
    (void)gf;  // caller computes the graph; we just wire ops into ctx

    const int n_head   = w.n_head;
    const int n_kv     = w.n_head_kv;
    const int head_dim = w.head_dim;
    const float eps    = GEMMA4_RMS_EPS;
    const float rope_base = w.rope_theta;

    // ── 1. FC projection: hidden = fc @ target_feat  →  [draft_hidden, n_tokens]
    //    fc:          [6*target_hidden, draft_hidden]  (ggml ne[0]=6*target_hidden, ne[1]=draft_hidden)
    //    target_feat: [6*target_hidden, n_tokens]
    //    Result:      [draft_hidden, n_tokens]
    ggml_tensor * hidden = ggml_mul_mat(ctx, w.fc, target_feat);
    ggml_set_name(hidden, "gemma4_draft_fc_out");

    // ── 2. Add draft token embeddings
    hidden = ggml_add(ctx, hidden, draft_embed);

    // ── 3. Initial RMSNorm + hidden_norm scale
    hidden = ggml_rms_norm(ctx, hidden, eps);
    hidden = ggml_mul(ctx, hidden, w.hidden_norm);
    ggml_set_name(hidden, "gemma4_draft_init_hidden");

    // ── 4. Transformer layers ─────────────────────────────────────────
    for (int il = 0; il < w.n_layer; il++) {
        const GemmaDraftLayer & L = w.layers[il];

        // ── 4a. Attention pre-norm
        ggml_tensor * cur = ggml_rms_norm(ctx, hidden, eps);
        cur = ggml_mul(ctx, cur, L.attn_norm);

        // ── 4b. Q / K / V projections (all from normalised hidden)
        //   wq: [n_head*head_dim,    draft_hidden]  ggml ne[0]=draft_hidden, ne[1]=q_dim
        //   wk: [n_head_kv*head_dim, draft_hidden]
        //   wv: [n_head_kv*head_dim, draft_hidden]
        ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, cur);  // [q_dim,  n_tokens]
        ggml_tensor * K = ggml_mul_mat(ctx, L.wk, cur);  // [kv_dim, n_tokens]
        ggml_tensor * V = ggml_mul_mat(ctx, L.wv, cur);  // [kv_dim, n_tokens]

        // ── 4c. Reshape + per-head RMSNorm for Q and K
        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, n_tokens);
        Q = ggml_rms_norm(ctx, Q, eps);
        Q = ggml_mul(ctx, Q, L.q_norm);

        K = ggml_reshape_3d(ctx, K, head_dim, n_kv, n_tokens);
        K = ggml_rms_norm(ctx, K, eps);
        K = ggml_mul(ctx, K, L.k_norm);

        V = ggml_reshape_3d(ctx, V, head_dim, n_kv, n_tokens);

        // ── 4d. RoPE (NEOX style, shared positions tensor for both Q and K)
        Q = ggml_rope_ext(ctx, Q, positions, /*freq_factors=*/nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
                          rope_base, /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                          /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
        K = ggml_rope_ext(ctx, K, positions, nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // ── 4e. Permute into flash_attn_ext layout
        //   q: [head_dim, n_tokens, n_head,    1]
        //   k: [head_dim, n_tokens, n_head_kv, 1]
        //   v: [head_dim, n_tokens, n_head_kv, 1]
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        Q = ggml_cont(ctx, Q);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        K = ggml_cont(ctx, K);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);
        V = ggml_cont(ctx, V);

        // ── 4f. Flash attention (block-causal mask from caller)
        //   scale = 1 / sqrt(head_dim); no logit softcap at attention level
        const float scale = 1.0f / std::sqrt((float)head_dim);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, attn_mask,
                                                  scale, /*max_bias=*/0.0f,
                                                  /*logit_softcap=*/0.0f);
        // attn: [head_dim, n_head, n_tokens, 1]
        attn = ggml_reshape_2d(ctx, attn, head_dim * n_head, n_tokens);

        // ── 4g. Output projection + residual
        ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn);
        hidden = ggml_add(ctx, hidden, attn_out);

        // ── 4h. FFN pre-norm
        ggml_tensor * hf = ggml_rms_norm(ctx, hidden, eps);
        hf = ggml_mul(ctx, hf, L.ffn_norm);

        // ── 4i. SwiGLU FFN: down(silu(gate(x)) * up(x))
        ggml_tensor * g  = ggml_mul_mat(ctx, L.w_gate, hf);
        g = ggml_silu(ctx, g);
        ggml_tensor * u  = ggml_mul_mat(ctx, L.w_up, hf);
        ggml_tensor * gu = ggml_mul(ctx, g, u);
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, L.w_down, gu);

        hidden = ggml_add(ctx, hidden, ffn_out);
    }

    // ── 5. Final output norm
    ggml_tensor * out = ggml_rms_norm(ctx, hidden, eps);
    out = ggml_mul(ctx, out, w.out_norm);
    ggml_set_name(out, "gemma4_draft_hidden_out");

    // ── 6. LM head (tied: transpose of tok_embd)
    //   tok_embd: [draft_hidden, n_vocab]  ggml ne[0]=draft_hidden, ne[1]=n_vocab
    //   out:      [draft_hidden, n_tokens]
    //   logits:   [n_vocab, n_tokens]
    ggml_tensor * logits = ggml_mul_mat(ctx, w.tok_embd, out);
    ggml_set_name(logits, "gemma4_draft_logits_pre_cap");

    // ── 7. Logit softcapping: logits = cap * tanh(logits / cap)
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
