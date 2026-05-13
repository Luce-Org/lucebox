// Loads a Gemma4 target model (31B Dense or 26B-A4B MoE) from a GGUF file into
// a GemmaTargetWeights struct backed by the supplied ggml backend (typically
// CUDA).
//
// The expected GGUF architecture string is "gemma4". The loader supports both
// the dense variant (60 layers, pure SwiGLU FFN) and the MoE variant (30
// layers, sparse expert FFN on the "26B-A4B" config).
//
// Tensor naming follows llama.cpp's gemma4-iswa.cpp conventions:
//
//   Global:
//     token_embd.weight              [n_embd, n_vocab]
//     output_norm.weight             [n_embd]
//     output.weight                  [n_vocab, n_embd]  (optional; falls back)
//
//   Per-Layer Embedding (PLE, present when n_embd_per_layer > 0):
//     per_layer_token_embd.weight    [n_embd_per_layer * n_layer, n_vocab]
//     per_layer_model_proj.weight    [n_embd, n_embd_per_layer * n_layer]
//     per_layer_proj_norm.weight     [n_embd_per_layer]
//     blk.{i}.inp_gate.weight        [n_embd, n_embd_per_layer]
//     blk.{i}.proj.weight            [n_embd_per_layer, n_embd]
//     blk.{i}.post_norm.weight       [n_embd]
//
//   Per-Layer Attention:
//     blk.{i}.attn_norm.weight       [n_embd]
//     blk.{i}.attn_q.weight          [n_embd, n_head * head_dim]
//     blk.{i}.attn_k.weight          [n_embd, n_head_kv * head_dim]  (optional)
//     blk.{i}.attn_v.weight          [n_embd, n_head_kv * head_dim]  (optional)
//     blk.{i}.attn_output.weight     [n_head * head_dim, n_embd]
//     blk.{i}.attn_q_norm.weight     [head_dim]
//     blk.{i}.attn_k_norm.weight     [head_dim]                       (optional)
//     blk.{i}.attn_post_norm.weight  [n_embd]
//     blk.{i}.rope_freqs.weight      [head_dim/2]  (full-attn layers only)
//     blk.{i}.out_scale.weight       [1]           (optional)
//
//   Per-Layer FFN (SwiGLU):
//     blk.{i}.ffn_norm.weight        [n_embd]
//     blk.{i}.ffn_gate.weight        [n_embd, n_ff]
//     blk.{i}.ffn_up.weight          [n_embd, n_ff]
//     blk.{i}.ffn_down.weight        [n_ff, n_embd]
//     blk.{i}.ffn_post_norm.weight   [n_embd]
//
//   Per-Layer MoE (26B-A4B only, present when n_expert > 0):
//     blk.{i}.ffn_gate_inp.weight    [n_embd, n_expert]
//     blk.{i}.ffn_gate_inp.scale     [n_embd]           (optional)
//     blk.{i}.ffn_pre_norm_2.weight  [n_embd]
//     blk.{i}.ffn_gate_up_exps.weight [n_embd, n_ff_exp*2, n_expert]
//     blk.{i}.ffn_down_exps.weight   [n_ff_exp, n_embd, n_expert]
//     blk.{i}.ffn_down_exps.scale    [n_expert]         (optional)
//     blk.{i}.ffn_post_norm_1.weight [n_embd]
//     blk.{i}.ffn_post_norm_2.weight [n_embd]
//
// KV-sharing: layers with index >= (n_layer - n_kv_shared_layers) omit wk, wv,
// k_norm. Their KV is borrowed from the last non-shared layer of the same
// attention type. layer_to_kv_idx maps each layer to its KV cache slot;
// layer_to_donor_kv maps shared layers to their donor layer index.

#include "internal.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

namespace {

// ─── Thin mmap wrapper ───────────────────────────────────────────────────────
// Mirrors the Mmap struct from gguf_target_loader.cpp.  Ownership can be
// transferred to a CpuEmbedder via release().

struct Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd   = -1;
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
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingA: error " + std::to_string(GetLastError());
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + ": " + std::strerror(errno); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat: " + std::string(std::strerror(errno)); return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap: " + std::string(std::strerror(errno)); addr = nullptr; return false; }
#endif
        return true;
    }

    void release() {
        addr = nullptr;
        len  = 0;
#if defined(_WIN32)
        hFile = INVALID_HANDLE_VALUE;
        hMap  = nullptr;
#else
        fd = -1;
#endif
    }

    ~Mmap() {
#if defined(_WIN32)
        if (addr)                          UnmapViewOfFile(addr);
        if (hMap)                          CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

// ─── GGUF metadata helpers ───────────────────────────────────────────────────

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

static size_t align_up(size_t x, size_t a) {
    if (a == 0) return x;
    const size_t r = x % a;
    return r == 0 ? x : x + (a - r);
}

// ─── Tensor selection filter ─────────────────────────────────────────────────
//
// All tensors go to GPU, including token_embd.weight which doubles as the LM
// head (tied weights in Gemma4-26B-A4B).  The CPU embedder keeps its own
// read-only mmap view of tok_embd for the input embedding path, so placing
// it on GPU as well is safe and necessary for correct LM head logits.

static bool is_gemma4_gpu_tensor(const char * name) {
    (void)name;
    return true;
}

} // namespace

// ─── load_gemma4_target_gguf ─────────────────────────────────────────────────

bool load_gemma4_target_gguf(const std::string & path,
                             ggml_backend_t       backend,
                             GemmaTargetWeights & out) {

    // ── 1. Parse GGUF metadata ────────────────────────────────────────────────

    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Validate architecture string.
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "gemma4") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected gemma4)");
            gguf_free(gctx);
            return false;
        }
    }

    // Read required architecture hyperparameters.
    const uint32_t n_embd    = get_u32_or(gctx, "gemma4.embedding_length",             0);
    const uint32_t n_layer   = get_u32_or(gctx, "gemma4.block_count",                  0);
    const uint32_t n_ff      = get_u32_or(gctx, "gemma4.feed_forward_length",           0);
    const uint32_t n_head    = get_u32_or(gctx, "gemma4.attention.head_count",          0);
    // Fix A: head_count_kv may be a per-layer INT32 array, not a scalar
    std::vector<int> head_kv_per_layer;
    uint32_t n_head_kv_max = 0;
    {
        int64_t kv_id = gguf_find_key(gctx, "gemma4.attention.head_count_kv");
        if (kv_id >= 0) {
            enum gguf_type kv_type = gguf_get_kv_type(gctx, kv_id);
            if (kv_type == GGUF_TYPE_ARRAY) {
                size_t arr_n = gguf_get_arr_n(gctx, kv_id);
                const int32_t * arr = (const int32_t *)gguf_get_arr_data(gctx, kv_id);
                head_kv_per_layer.resize(arr_n);
                for (size_t i = 0; i < arr_n; i++) {
                    head_kv_per_layer[i] = (int)arr[i];
                    if ((uint32_t)arr[i] > n_head_kv_max) n_head_kv_max = (uint32_t)arr[i];
                }
            } else {
                // Scalar fallback
                n_head_kv_max = gguf_get_val_u32(gctx, kv_id);
            }
        }
    }
    const uint32_t n_head_kv = n_head_kv_max;

    // Fix D: read both full-attn and SWA head dims
    const uint32_t head_dim     = get_u32_or(gctx, "gemma4.attention.key_length",     0);
    const uint32_t head_dim_swa = get_u32_or(gctx, "gemma4.attention.key_length_swa", head_dim);

    // Fix B: vocab_size key may be absent — fall back to tokenizer array length
    uint32_t n_vocab = get_u32_or(gctx, "gemma4.vocab_size", 0);
    if (n_vocab == 0) {
        int64_t tok_id = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_id >= 0) n_vocab = (uint32_t)gguf_get_arr_n(gctx, tok_id);
    }
    const uint32_t swa_win   = get_u32_or(gctx, "gemma4.attention.sliding_window",      1024);
    const uint32_t n_kv_shared = get_u32_or(gctx, "gemma4.attention.shared_kv_layers", 0);
    const uint32_t n_embd_per_layer = get_u32_or(gctx, "gemma4.embedding_length_per_layer_input", 0);
    const uint32_t n_expert   = get_u32_or(gctx, "gemma4.expert_count",                0);
    const uint32_t n_expert_used = get_u32_or(gctx, "gemma4.expert_used_count",        0);
    const uint32_t n_ff_exp   = get_u32_or(gctx, "gemma4.expert_feed_forward_length",  0);

    const float rope_theta     = get_f32_or(gctx, "gemma4.rope.freq_base",     1000000.0f);
    const float rope_theta_swa = get_f32_or(gctx, "gemma4.rope.freq_base_swa", 1000000.0f);
    const float logit_softcap  = get_f32_or(gctx, "gemma4.final_logit_softcapping", 30.0f);

    if (n_embd == 0 || n_layer == 0 || n_ff == 0 ||
        n_head == 0 || n_head_kv == 0 || head_dim == 0 || n_vocab == 0) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "missing or zero required hparams: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u n_vocab=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim, n_vocab);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // ── 2. Build the per-layer SWA pattern ───────────────────────────────────
    //
    // swa_layers[il] != 0 → sliding-window attention; == 0 → full attention.
    // The array is stored as GGUF_TYPE_ARRAY of INT32 or BOOL.  If absent we
    // default to alternating: odd layers use SWA, even layers use full attn
    // (matches Gemma4-31B's default pattern).

    std::vector<bool> swa_layers(n_layer, false);
    {
        int64_t swa_arr_id = gguf_find_key(gctx, "gemma4.attention.sliding_window_pattern");
        // Fix C: sliding_window_pattern may be BOOL array (1-byte), not INT32
        if (swa_arr_id >= 0) {
            size_t arr_n = gguf_get_arr_n(gctx, swa_arr_id);
            enum gguf_type arr_type = gguf_get_arr_type(gctx, swa_arr_id);
            const void * arr_data = gguf_get_arr_data(gctx, swa_arr_id);
            for (size_t i = 0; i < arr_n && i < n_layer; i++) {
                if (arr_type == GGUF_TYPE_BOOL || arr_type == GGUF_TYPE_INT8 || arr_type == GGUF_TYPE_UINT8) {
                    swa_layers[i] = (((const uint8_t *)arr_data)[i] != 0);
                } else {
                    swa_layers[i] = (((const int32_t *)arr_data)[i] != 0);
                }
            }
        } else {
            // Fallback: odd-indexed layers → SWA, even → full attention.
            for (uint32_t i = 0; i < n_layer; i++) {
                swa_layers[i] = ((i % 2) == 1);
            }
        }
    }

    // ── 3. Build KV-sharing maps ──────────────────────────────────────────────
    //
    // Layers [0, n_layer - n_kv_shared_layers) own their own KV cache slot.
    // Layers [n_layer - n_kv_shared_layers, n_layer) are KV-shared: they borrow
    // KV from the last non-shared layer that has the same attention type (SWA
    // or full).  layer_to_kv_idx[il] == -1 for shared layers.

    const int n_non_shared = (int)n_layer - (int)n_kv_shared;
    if (n_non_shared < 0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "n_kv_shared_layers=%u > n_layer=%u", n_kv_shared, n_layer);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    std::vector<int> layer_to_kv_idx((size_t)n_layer, -1);
    std::vector<int> layer_to_donor_kv((size_t)n_layer, -1);
    {
        int kv_slot = 0;
        for (int il = 0; il < n_non_shared; il++) {
            layer_to_kv_idx[il] = kv_slot++;
        }
        // Shared layers find their donor: the last non-shared layer with the
        // same attention type.
        for (int il = n_non_shared; il < (int)n_layer; il++) {
            bool is_swa = swa_layers[(size_t)il];
            int donor = -1;
            for (int j = n_non_shared - 1; j >= 0; j--) {
                if (swa_layers[(size_t)j] == is_swa) { donor = j; break; }
            }
            layer_to_donor_kv[il] = donor;
            // kv_idx stays -1 (no dedicated slot).
        }
    }
    const int n_kv_slots = n_non_shared;  // total distinct KV cache entries

    // ── 4. Populate struct metadata ──────────────────────────────────────────

    out.ctx     = meta_ctx;
    out.backend = backend;

    out.n_embd              = (int)n_embd;
    out.n_head              = (int)n_head;
    out.n_head_kv           = (int)n_head_kv;
    out.head_dim            = (int)head_dim;
    out.head_dim_swa        = (int)head_dim_swa;
    out.head_kv_per_layer   = head_kv_per_layer;
    out.n_layer             = (int)n_layer;
    out.n_ff             = (int)n_ff;
    out.n_vocab          = (int)n_vocab;
    out.n_embd_per_layer = (int)n_embd_per_layer;
    out.swa_window       = (int)swa_win;
    out.swa_layers       = swa_layers;
    out.n_kv_shared_layers = (int)n_kv_shared;
    out.n_layer_kv       = n_kv_slots;
    out.rope_theta       = rope_theta;
    out.rope_theta_swa   = rope_theta_swa;
    out.n_expert         = (int)n_expert;
    out.n_expert_used    = (int)n_expert_used;
    out.n_ff_exp         = (int)n_ff_exp;
    out.logit_softcap    = logit_softcap;

    // BOS / EOS tokens (missing key → -1)
    {
        const uint32_t kMissing = 0xFFFFFFFFu;
        const uint32_t raw_bos  = get_u32_or(gctx, "tokenizer.ggml.bos_token_id",  kMissing);
        const uint32_t raw_eos  = get_u32_or(gctx, "tokenizer.ggml.eos_token_id",  kMissing);
        const uint32_t raw_eot  = get_u32_or(gctx, "tokenizer.ggml.eot_token_id",  kMissing);
        out.bos_id      = (raw_bos == kMissing) ? -1 : (int32_t)raw_bos;
        out.eos_id      = (raw_eos == kMissing) ? -1 : (int32_t)raw_eos;
        out.eos_chat_id = (raw_eot == kMissing) ? -1 : (int32_t)raw_eot;

        // Gemma4 fallback: <end_of_turn> (107) is the chat stop token.
        // Many GGUFs omit eot_token_id; default to 107 when missing.
        if (out.eos_chat_id < 0) {
            out.eos_chat_id = 107;
        }

        std::printf("[gemma4_loader] bos_id=%d eos_id=%d eos_chat_id=%d\n",
                    out.bos_id, out.eos_id, out.eos_chat_id);
    }

    // ── 5. Compute capture_layer_ids ─────────────────────────────────────────
    //
    // Use hardcoded values from the DFlash draft model config.json.
    // Fallback to evenly-spaced formula for unknown layer counts.
    {
        const int N = GEMMA4_DRAFT_N_TARGET_LAYERS;  // 6
        if ((int)n_layer == 30) {
            // Gemma4-26B-A4B — from z-lab/gemma-4-26B-A4B-it-DFlash config.json
            const int ids[6] = {1, 6, 11, 17, 22, 27};
            for (int k = 0; k < N; k++) out.capture_layer_ids[k] = ids[k];
        } else if ((int)n_layer == 60) {
            // Gemma4-31B — from z-lab/gemma-4-31B-it-DFlash config.json
            const int ids[6] = {1, 12, 23, 35, 46, 57};
            for (int k = 0; k < N; k++) out.capture_layer_ids[k] = ids[k];
        } else {
            // Fallback: evenly spaced
            const int step = ((int)n_layer - 2) / (N - 1);
            for (int k = 0; k < N; k++) out.capture_layer_ids[k] = 1 + k * step;
        }
        std::printf("[gemma4_loader] capture_layer_ids:");
        for (int k = 0; k < N; k++) std::printf(" %d", out.capture_layer_ids[k]);
        std::printf("\n");
    }

    // ── 6. Wire tensor pointers ───────────────────────────────────────────────

    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };

    out.tok_embd = g("token_embd.weight");
    out.out_norm = g("output_norm.weight");
    // output.weight is optional; fall back to token_embd for tied weights.
    out.output   = g("output.weight");
    if (!out.output) out.output = out.tok_embd;

    if (!out.tok_embd || !out.out_norm) {
        set_last_error("missing top-level tensors (token_embd.weight / output_norm.weight)");
        gguf_free(gctx);
        return false;
    }

    // Global PLE tensors (present only when n_embd_per_layer > 0)
    if (n_embd_per_layer > 0) {
        out.per_layer_tok_embd   = g("per_layer_token_embd.weight");
        out.per_layer_model_proj = g("per_layer_model_proj.weight");
        out.per_layer_proj_norm  = g("per_layer_proj_norm.weight");
        if (!out.per_layer_tok_embd || !out.per_layer_model_proj || !out.per_layer_proj_norm) {
            set_last_error("n_embd_per_layer > 0 but PLE global tensors missing");
            gguf_free(gctx);
            return false;
        }
    }

    // Load global rope_freqs tensor (full-attention layers use this for proportional RoPE).
    // Gemma4 stores one shared rope_freqs.weight (not per-layer blk.{i}.rope_freqs.weight).
    // All full-attention layers share this single tensor, matching llama.cpp's TENSOR_DUPLICATED
    // pattern (llama-model.cpp:4657-4658).
    ggml_tensor * global_rope_freqs = g("rope_freqs.weight");

    // Per-layer tensors.
    out.layers.assign((size_t)n_layer, GemmaTargetLayer{});

    for (int il = 0; il < (int)n_layer; il++) {
        char name[160];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };

        GemmaTargetLayer & L = out.layers[(size_t)il];

        // ── Attention (always present) ────────────────────────────────────────
        L.attn_norm      = fnd("attn_norm.weight");
        L.wq             = fnd("attn_q.weight");
        L.wo             = fnd("attn_output.weight");
        L.q_norm         = fnd("attn_q_norm.weight");
        // This GGUF uses "post_attention_norm.weight"; fall back to legacy name
        L.attn_post_norm = fnd("post_attention_norm.weight");
        if (!L.attn_post_norm) L.attn_post_norm = fnd("attn_post_norm.weight");

        if (!L.attn_norm || !L.wq || !L.wo || !L.q_norm || !L.attn_post_norm) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d: missing required attention tensor", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }

        // wk, wv, k_norm — absent for KV-shared layers (il >= n_non_shared).
        const bool is_kv_owner = (il < n_non_shared);
        if (is_kv_owner) {
            L.wk     = fnd("attn_k.weight");
            L.wv     = fnd("attn_v.weight");
            L.k_norm = fnd("attn_k_norm.weight");
            if (!L.wk) {
                char b[128];
                std::snprintf(b, sizeof(b), "layer %d: expected wk (non-shared), missing", il);
                set_last_error(b);
                gguf_free(gctx);
                return false;
            }
            // V may be absent on full-attention layers where V == K (shared K/V).
            if (!L.wv) {
                L.wv = L.wk;
            }
            // k_norm may be absent for SWA layers in some checkpoints; allow nullptr.
        }

        // Optional per-layer tensors
        L.rope_freqs = fnd("rope_freqs.weight");
        // Full-attention layers use proportional RoPE via rope_freqs (freq_factors).
        // Gemma4 stores a single global rope_freqs.weight (no per-layer blk.{i} variant).
        // Fall back to the global tensor for full-attention layers when the per-layer
        // variant is absent (which is always the case for this GGUF format).
        if (!L.rope_freqs && !swa_layers[(size_t)il] && global_rope_freqs) {
            L.rope_freqs = global_rope_freqs;
        }
        // This GGUF uses "layer_output_scale.weight"; fall back to legacy name
        L.out_scale  = fnd("layer_output_scale.weight");
        if (!L.out_scale) L.out_scale = fnd("out_scale.weight");

        // ── FFN (always present) ──────────────────────────────────────────────
        L.ffn_norm      = fnd("ffn_norm.weight");
        L.w_gate        = fnd("ffn_gate.weight");
        L.w_up          = fnd("ffn_up.weight");
        L.w_down        = fnd("ffn_down.weight");
        // This GGUF uses "post_ffw_norm.weight"; fall back to legacy name
        L.ffn_post_norm = fnd("post_ffw_norm.weight");
        if (!L.ffn_post_norm) L.ffn_post_norm = fnd("ffn_post_norm.weight");

        if (!L.ffn_norm || !L.w_gate || !L.w_up || !L.w_down || !L.ffn_post_norm) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d: missing required FFN tensor", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }

        // ── MoE (26B-A4B — present when n_expert > 0) ────────────────────────
        if (n_expert > 0) {
            L.ffn_gate_inp    = fnd("ffn_gate_inp.weight");
            L.ffn_gate_inp_s  = fnd("ffn_gate_inp.scale");
            // This GGUF uses "pre_ffw_norm_2.weight"; fall back to legacy name
            L.ffn_pre_norm_2  = fnd("pre_ffw_norm_2.weight");
            if (!L.ffn_pre_norm_2) L.ffn_pre_norm_2 = fnd("ffn_pre_norm_2.weight");
            L.ffn_gate_up_exps = fnd("ffn_gate_up_exps.weight");
            L.ffn_down_exps   = fnd("ffn_down_exps.weight");
            L.ffn_down_exps_s = fnd("ffn_down_exps.scale");
            // This GGUF uses "post_ffw_norm_1/2.weight"; fall back to legacy names
            L.ffn_post_norm_1 = fnd("post_ffw_norm_1.weight");
            if (!L.ffn_post_norm_1) L.ffn_post_norm_1 = fnd("ffn_post_norm_1.weight");
            L.ffn_post_norm_2 = fnd("post_ffw_norm_2.weight");
            if (!L.ffn_post_norm_2) L.ffn_post_norm_2 = fnd("ffn_post_norm_2.weight");

            if (!L.ffn_gate_inp || !L.ffn_pre_norm_2 ||
                !L.ffn_gate_up_exps || !L.ffn_down_exps ||
                !L.ffn_post_norm_1 || !L.ffn_post_norm_2) {
                char b[128];
                std::snprintf(b, sizeof(b), "layer %d: MoE model but missing expert tensor", il);
                set_last_error(b);
                gguf_free(gctx);
                return false;
            }
            // ffn_gate_inp_s, ffn_down_exps_s are optional quantization scales.
        }

        // ── Per-Layer Embedding (PLE) ─────────────────────────────────────────
        if (n_embd_per_layer > 0) {
            L.ple_inp_gate  = fnd("inp_gate.weight");
            L.ple_proj      = fnd("proj.weight");
            L.ple_post_norm = fnd("post_norm.weight");
            if (!L.ple_inp_gate || !L.ple_proj || !L.ple_post_norm) {
                char b[128];
                std::snprintf(b, sizeof(b), "layer %d: PLE model but missing per-layer embedding tensor", il);
                set_last_error(b);
                gguf_free(gctx);
                return false;
            }
        }
    }

    // ── 7. Allocate GPU buffer ────────────────────────────────────────────────
    //
    // Walk all GGUF tensors, skip token_embd.weight (stays CPU), accumulate
    // aligned sizes, allocate one contiguous backend buffer, assign each tensor.

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);

    struct TensorSlot {
        ggml_tensor * tensor      = nullptr;
        size_t        file_offset = 0;
        size_t        file_size   = 0;
        size_t        buf_offset  = 0;
    };

    std::vector<TensorSlot> slots;
    size_t total_gpu = 0;
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        if (!is_gemma4_gpu_tensor(tname)) continue;
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        total_gpu = align_up(total_gpu, alignment);
        TensorSlot s;
        s.tensor      = t;
        s.file_offset = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
        s.file_size   = gguf_get_tensor_size(gctx, tid);
        s.buf_offset  = total_gpu;
        total_gpu    += ggml_backend_buft_get_alloc_size(buft, t);
        slots.push_back(s);
    }

    if (slots.empty()) {
        set_last_error("no GPU tensors found in gemma4 GGUF");
        gguf_free(gctx);
        return false;
    }

    // Cleanup helper: release any GPU buffer and ggml context already assigned
    // to `out` before returning false.  Must be called on every failure path
    // after out.buf has been (or is about to be) allocated.
    auto cleanup_out = [&]() {
        if (out.buf) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
        }
        // out.ctx == meta_ctx; free it so the caller doesn't leak the graph.
        if (out.ctx) {
            ggml_free(out.ctx);
            out.ctx = nullptr;
        }
        out = GemmaTargetWeights{};
    };

    out.buf = ggml_backend_alloc_buffer(backend, total_gpu);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_buffer failed (gemma4 target)");
        gguf_free(gctx);
        cleanup_out();
        return false;
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    char * base = (char *)ggml_backend_buffer_get_base(out.buf);
    for (const TensorSlot & s : slots) {
        if (ggml_backend_tensor_alloc(out.buf, s.tensor, base + s.buf_offset) != GGML_STATUS_SUCCESS) {
            set_last_error("ggml_backend_tensor_alloc failed (gemma4 target)");
            gguf_free(gctx);
            cleanup_out();
            return false;
        }
    }

    // ── 8. mmap file, upload GPU tensors, keep tok_embd on CPU ───────────────

    std::string err;
    Mmap mm;
    if (!mm.open_ro(path, err)) {
        set_last_error(err);
        gguf_free(gctx);
        cleanup_out();
        return false;
    }

    const size_t data_start = gguf_get_data_offset(gctx);
    size_t gpu_bytes_uploaded = 0;
    size_t tok_embd_off  = 0;
    size_t tok_embd_sz   = 0;
    ggml_type tok_embd_type = GGML_TYPE_COUNT;

    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t    = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("tensor '") + tname + "' overflows file");
            gguf_free(gctx);
            cleanup_out();
            return false;
        }
        if (std::strcmp(tname, "token_embd.weight") == 0) {
            tok_embd_off  = off;
            tok_embd_sz   = sz;
            tok_embd_type = gguf_get_tensor_type(gctx, tid);
            // fall through: also upload to GPU for LM head (tied weights)
        }
        ggml_backend_tensor_set(t, (const uint8_t *)mm.addr + off, 0, sz);
        gpu_bytes_uploaded += sz;
    }

    gguf_free(gctx);

    if (tok_embd_off == 0 || tok_embd_type == GGML_TYPE_COUNT) {
        set_last_error("token_embd.weight not found or invalid type");
        cleanup_out();
        return false;
    }

    // Fix 2: validate tok_embd_sz divisibility before computing row stride.
    if (n_vocab == 0 || tok_embd_sz % (size_t)n_vocab != 0) {
        set_last_error("malformed GGUF: tok_embd_sz=" + std::to_string(tok_embd_sz) +
                       " not divisible by n_vocab=" + std::to_string(n_vocab));
        cleanup_out();
        return false;
    }

    // ── 9. Transfer mmap ownership to CpuEmbedder ────────────────────────────

    out.embedder.mmap_addr      = mm.addr;
    out.embedder.mmap_len       = mm.len;
#if defined(_WIN32)
    out.embedder.mmap_hfile     = mm.hFile;
    out.embedder.mmap_hmap      = mm.hMap;
#else
    out.embedder.mmap_fd        = mm.fd;
#endif
    out.embedder.tok_embd_bytes = (const uint8_t *)mm.addr + tok_embd_off;
    out.embedder.tok_embd_type  = tok_embd_type;
    out.embedder.n_embd         = (int64_t)n_embd;
    out.embedder.n_vocab        = (int64_t)n_vocab;
    out.embedder.row_bytes      = tok_embd_sz / (size_t)n_vocab;
    mm.release();

    char summary[256];
    std::snprintf(summary, sizeof(summary),
        "gemma4 target loaded: n_layer=%u n_embd=%u n_ff=%u n_expert=%u "
        "n_kv_slots=%d n_kv_shared=%u, %zu GPU tensors %.2f GiB, "
        "tok_embd %.0f MiB GPU+CPU-mmap (%s, tied LM head)",
        n_layer, n_embd, n_ff, n_expert, n_kv_slots, n_kv_shared,
        slots.size(), (double)gpu_bytes_uploaded / (1024.0 * 1024.0 * 1024.0),
        (double)tok_embd_sz / (1024.0 * 1024.0), ggml_type_name(tok_embd_type));
    set_last_error(summary);

    return true;
}

// ─── free_gemma4_target_weights ──────────────────────────────────────────────

void free_gemma4_target_weights(GemmaTargetWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    // CpuEmbedder destructor handles the mmap automatically.
    w.layers.clear();
    w.tok_embd              = nullptr;
    w.out_norm              = nullptr;
    w.output                = nullptr;
    w.per_layer_tok_embd    = nullptr;
    w.per_layer_model_proj  = nullptr;
    w.per_layer_proj_norm   = nullptr;
    w.swa_layers.clear();
}

} // namespace dflash27b
