// Loads Gemma4 from a GGUF file. Mirrors laguna_target_loader.cpp pattern
// but for the gemma4 iSWA + MoE arch.
//
// Tensor naming follows gguf-py MODEL_ARCH.GEMMA4:
//   token_embd.weight, output_norm.weight, output.weight
//   blk.<i>.attn_norm.weight, attn_q.weight, attn_k.weight, attn_v.weight,
//   attn_output.weight, attn_q_norm.weight, attn_k_norm.weight,
//   attn_post_norm.weight, ffn_norm.weight, ffn_gate.weight, ffn_up.weight,
//   ffn_down.weight, ffn_post_norm.weight
//   MoE: ffn_gate_inp.weight, ffn_gate_inp_shexp.weight,
//   ffn_gate_exps.weight, ffn_up_exps.weight, ffn_down_exps.weight, etc.

#include "gemma4_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
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

namespace dflash::common {

namespace {

struct Gemma4Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
    int     fd   = -1;

    bool open_ro(const std::string & path, std::string & err) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + " " + strerror(errno); return false; }
        struct stat st;
        if (fstat(fd, &st) < 0) { err = "fstat"; ::close(fd); fd = -1; return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap"; addr = nullptr; ::close(fd); fd = -1; return false; }
        return true;
    }
    void close_map() {
        if (addr) { ::munmap(addr, len); addr = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
};

uint32_t get_u32_or(gguf_context * g, const char * key, uint32_t def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    // Handle array type: return first element
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const uint32_t *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_u32(g, id);
}

float get_f32_or(gguf_context * g, const char * key, float def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const float *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_f32(g, id);
}

// Read a u32 array key into a vector (empty if not found or not an array).
std::vector<uint32_t> get_u32_arr(gguf_context * g, const char * key) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return {};
    const size_t n = gguf_get_arr_n(g, id);
    const uint32_t * data = (const uint32_t *)gguf_get_arr_data(g, id);
    return std::vector<uint32_t>(data, data + n);
}

ggml_tensor * find_tensor(ggml_context * ctx, const char * name) {
    return ggml_get_tensor(ctx, name);
}

}  // namespace

bool load_gemma4_gguf(const std::string & path,
                       ggml_backend_t backend,
                       Gemma4Weights & out) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) { set_last_error("gguf_init failed: " + path); return false; }

    // Validate arch
    {
        int64_t aid = gguf_find_key(gctx, "general.architecture");
        if (aid < 0) { set_last_error("missing general.architecture"); gguf_free(gctx); return false; }
        const char * arch = gguf_get_val_str(gctx, aid);
        if (std::string(arch) != "gemma4") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected gemma4)");
            gguf_free(gctx); return false;
        }
    }

    // Read hparams
    const uint32_t n_layer       = get_u32_or(gctx, "gemma4.block_count", 0);
    const uint32_t n_embd        = get_u32_or(gctx, "gemma4.embedding_length", 0);
    const uint32_t n_ff          = get_u32_or(gctx, "gemma4.feed_forward_length", 0);
    const uint32_t n_ff_exp      = get_u32_or(gctx, "gemma4.expert_feed_forward_length", 0);
    const uint32_t n_head        = get_u32_or(gctx, "gemma4.attention.head_count", 0);
    const uint32_t n_head_kv     = get_u32_or(gctx, "gemma4.attention.head_count_kv", 0);
    const uint32_t head_dim_full = get_u32_or(gctx, "gemma4.attention.key_length", 128);
    const uint32_t head_dim_swa  = get_u32_or(gctx, "gemma4.attention.key_length_swa", head_dim_full);
    const uint32_t n_expert      = get_u32_or(gctx, "gemma4.expert_count", 0);
    const uint32_t n_expert_used = get_u32_or(gctx, "gemma4.expert_used_count", 0);
    const uint32_t n_dense_lead  = get_u32_or(gctx, "gemma4.leading_dense_block_count", 0);
    const uint32_t sliding_win   = get_u32_or(gctx, "gemma4.attention.sliding_window", 0);
    const uint32_t shared_kv     = get_u32_or(gctx, "gemma4.attention.shared_kv_layers", 0);
    const uint32_t n_embd_pl     = get_u32_or(gctx, "gemma4.embedding_length_per_layer_input", 0);

    // Per-layer head_count_kv (may be array or scalar)
    std::vector<uint32_t> head_kv_arr = get_u32_arr(gctx, "gemma4.attention.head_count_kv");

    // Get vocab size from token_embd tensor shape (not always in metadata)
    uint32_t n_vocab = get_u32_or(gctx, "gemma4.vocab_size", 0);
    if (n_vocab == 0) {
        ggml_tensor * tok_embd = find_tensor(meta_ctx, "token_embd.weight");
        if (tok_embd) n_vocab = (uint32_t)tok_embd->ne[1];
    }

    const float rope_base_full   = get_f32_or(gctx, "gemma4.rope.freq_base", 1000000.0f);
    const float rope_base_swa    = get_f32_or(gctx, "gemma4.rope.freq_base_swa", 10000.0f);
    const float norm_eps         = get_f32_or(gctx, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
    const float logit_softcap    = get_f32_or(gctx, "gemma4.final_logit_softcapping", 0.0f);

    if (n_layer == 0 || n_embd == 0 || n_head == 0 || n_head_kv == 0 || n_vocab == 0) {
        set_last_error("gemma4: missing essential hparams");
        gguf_free(gctx); return false;
    }

    // Populate metadata
    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer            = (int)n_layer;
    out.n_head             = (int)n_head;
    out.n_head_kv          = (int)n_head_kv;
    out.head_dim           = (int)head_dim_swa;   // SWA head_dim (smaller)
    out.head_dim_full      = (int)head_dim_full;  // full-attn head_dim (larger)
    out.n_embd             = (int)n_embd;
    out.n_ff               = (int)n_ff;
    out.n_ff_exp           = (int)n_ff_exp;
    out.n_ff_shexp         = (int)n_ff;  // shared expert same size as dense FFN
    out.n_expert           = (int)n_expert;
    out.n_expert_used      = (int)n_expert_used;
    out.n_layer_dense_lead = (int)n_dense_lead;
    out.n_embd_per_layer   = (int)n_embd_pl;
    out.n_vocab            = (int)n_vocab;
    out.sliding_window     = (int)sliding_win;
    out.rope_freq_base_full = rope_base_full;
    out.rope_freq_base_swa  = rope_base_swa;
    out.final_logit_softcap = logit_softcap;
    out.norm_eps            = norm_eps;

    // Per-layer n_head_kv from array (or fill scalar)
    out.n_head_kv_per_layer.resize(n_layer);
    if (!head_kv_arr.empty()) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            out.n_head_kv_per_layer[il] = (int)(il < head_kv_arr.size() ? head_kv_arr[il] : head_kv_arr.back());
        }
    } else {
        for (uint32_t il = 0; il < n_layer; ++il) {
            out.n_head_kv_per_layer[il] = (int)n_head_kv;
        }
    }

    // KV sharing: last shared_kv layers reuse earlier KV
    out.kv_sharing_start = (int)(n_layer - shared_kv);
    out.has_kv.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        out.has_kv[il] = (int)il < out.kv_sharing_start;
    }

    // SWA pattern from GGUF (array of bools indicating SWA layers)
    out.swa_layers.resize(n_layer, false);
    {
        int64_t pat_id = gguf_find_key(gctx, "gemma4.attention.sliding_window_pattern");
        if (pat_id >= 0 && gguf_get_kv_type(gctx, pat_id) == GGUF_TYPE_ARRAY) {
            const size_t n = gguf_get_arr_n(gctx, pat_id);
            if (n > 0) {
                const auto arr_type = gguf_get_arr_type(gctx, pat_id);
                const void * data = gguf_get_arr_data(gctx, pat_id);
                for (uint32_t il = 0; il < n_layer; ++il) {
                    size_t idx = il % n;  // cycle if pattern shorter than n_layer
                    if (arr_type == GGUF_TYPE_BOOL || arr_type == GGUF_TYPE_UINT8) {
                        out.swa_layers[il] = ((const uint8_t *)data)[idx] != 0;
                    } else if (arr_type == GGUF_TYPE_INT32 || arr_type == GGUF_TYPE_UINT32) {
                        out.swa_layers[il] = ((const uint32_t *)data)[idx] != 0;
                    }
                }
            }
        } else {
            // Fallback: infer from per-layer head_kv (small kv = full, large kv = swa)
            // or use default pattern (alternating 5:1)
            if (!head_kv_arr.empty() && head_kv_arr.size() >= n_layer) {
                uint32_t max_kv = *std::max_element(head_kv_arr.begin(), head_kv_arr.end());
                for (uint32_t il = 0; il < n_layer; ++il) {
                    // Layers with max n_head_kv are SWA, smaller are full-attn
                    out.swa_layers[il] = (head_kv_arr[il] == max_kv);
                }
            } else {
                // Default: every 6th layer is full, rest SWA (5:1 pattern)
                for (uint32_t il = 0; il < n_layer; ++il) {
                    out.swa_layers[il] = ((il % 6) != 5);
                }
            }
        }
    }

    // Tokenizer
    const uint32_t miss = 0xFFFFFFFFu;
    out.bos_id      = (int32_t)get_u32_or(gctx, "tokenizer.ggml.bos_token_id", miss);
    out.eos_id      = (int32_t)get_u32_or(gctx, "tokenizer.ggml.eos_token_id", miss);
    out.eos_chat_id = (int32_t)get_u32_or(gctx, "tokenizer.ggml.eot_token_id", miss);
    if (out.bos_id == (int32_t)miss) out.bos_id = 2;
    if (out.eos_id == (int32_t)miss) out.eos_id = 1;
    if (out.eos_chat_id == (int32_t)miss) out.eos_chat_id = -1;

    std::printf("[gemma4-loader] n_layer=%u n_embd=%u head_dim_swa=%u head_dim_full=%u n_head=%u n_head_kv=%u\n",
                n_layer, n_embd, head_dim_swa, head_dim_full, n_head, n_head_kv);
    std::printf("[gemma4-loader] n_expert=%u used=%u dense_lead=%u sliding_window=%u\n",
                n_expert, n_expert_used, n_dense_lead, sliding_win);
    std::printf("[gemma4-loader] kv_sharing_start=%d per_layer_embd=%u logit_softcap=%g\n",
                out.kv_sharing_start, n_embd_pl, logit_softcap);
    std::fflush(stdout);

    // ── Map tensors ────────────────────────────────────────────────────
    // Mmap the GGUF; tok_embd stays on CPU (CpuEmbedder reads it directly).
    Gemma4Mmap mmap;
    {
        std::string err;
        if (!mmap.open_ro(path, err)) {
            set_last_error("gemma4 mmap: " + err);
            gguf_free(gctx); return false;
        }
    }

    // Allocate backend buffer for all tensors
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("gemma4: backend alloc failed");
        mmap.close_map();
        gguf_free(gctx); return false;
    }

    // Copy tensor data from mmap to backend; track tok_embd for CPU embedder
    const size_t data_offset = gguf_get_data_offset(gctx);
    const int n_tensors = gguf_get_n_tensors(gctx);
    size_t tok_embd_off = 0, tok_embd_sz = 0;
    ggml_type tok_embd_type = GGML_TYPE_COUNT;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        if (!t) continue;
        size_t offset = data_offset + gguf_get_tensor_offset(gctx, i);
        size_t sz     = ggml_nbytes(t);
        const void * src = (const char *)mmap.addr + offset;

        if (std::strcmp(name, "token_embd.weight") == 0) {
            tok_embd_off  = offset;
            tok_embd_sz   = sz;
            tok_embd_type = t->type;
            // Upload to GPU (needed for tied lm_head / output)
            ggml_backend_tensor_set(t, src, 0, sz);
        } else {
            ggml_backend_tensor_set(t, src, 0, sz);
        }
    }

    // Set up CPU embedder (keeps mmap alive)
    out.embedder.mmap_addr      = mmap.addr;
    out.embedder.mmap_len       = mmap.len;
    out.embedder.mmap_fd        = mmap.fd;
    out.embedder.tok_embd_bytes = (const uint8_t *)mmap.addr + tok_embd_off;
    out.embedder.tok_embd_type  = tok_embd_type;
    out.embedder.n_embd         = n_embd;
    out.embedder.n_vocab        = (int64_t)n_vocab;
    out.embedder.row_bytes      = tok_embd_sz / (size_t)n_vocab;
    // Release mmap ownership to embedder (it will munmap on destruction)
    mmap.addr = nullptr;
    mmap.fd   = -1;

    // ── Assign tensors to struct ───────────────────────────────────────
    out.tok_embd = find_tensor(meta_ctx, "token_embd.weight");
    out.out_norm = find_tensor(meta_ctx, "output_norm.weight");
    // Gemma4 uses tied embeddings: lm_head = token_embd
    out.output   = find_tensor(meta_ctx, "output.weight");
    if (!out.output) out.output = out.tok_embd;
    // Global rope_freqs (not per-layer in this model variant)
    out.rope_freqs_global = find_tensor(meta_ctx, "rope_freqs.weight");
    out.per_layer_tok_embd   = find_tensor(meta_ctx, "per_layer_tok_embd.weight");
    out.per_layer_model_proj = find_tensor(meta_ctx, "per_layer_model_proj.weight");
    out.per_layer_proj_norm  = find_tensor(meta_ctx, "per_layer_proj_norm.weight");

    out.layers.resize(n_layer);
    char buf[256];
    for (uint32_t il = 0; il < n_layer; ++il) {
        auto & L = out.layers[il];

        auto get = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(buf, sizeof(buf), "blk.%u.%s", il, suffix);
            return find_tensor(meta_ctx, buf);
        };

        L.attn_norm       = get("attn_norm.weight");
        L.wq              = get("attn_q.weight");
        L.wk              = get("attn_k.weight");
        L.wv              = get("attn_v.weight");
        L.wo              = get("attn_output.weight");
        L.q_norm          = get("attn_q_norm.weight");
        L.k_norm          = get("attn_k_norm.weight");
        L.attn_post_norm  = get("post_attention_norm.weight");
        L.rope_freqs      = get("rope_freqs.weight");

        L.ffn_norm        = get("ffn_norm.weight");
        L.ffn_gate        = get("ffn_gate.weight");
        L.ffn_up          = get("ffn_up.weight");
        L.ffn_down        = get("ffn_down.weight");
        L.ffn_post_norm   = get("post_ffw_norm.weight");

        // MoE tensors (only present for MoE models)
        L.ffn_norm_moe     = get("ffn_norm_moe.weight");
        L.ffn_gate_inp     = get("ffn_gate_inp.weight");
        L.ffn_gate_inp_s   = get("ffn_gate_inp.scale");
        L.ffn_gate_up_exps = get("ffn_gate_up_exps.weight");
        L.ffn_down_exps    = get("ffn_down_exps.weight");
        L.ffn_down_exps_s  = get("ffn_down_exps.scale");
        L.ffn_gate_shexp   = get("ffn_gate_shexp.weight");
        L.ffn_up_shexp     = get("ffn_up_shexp.weight");
        L.ffn_down_shexp   = get("ffn_down_shexp.weight");
        L.ffn_pre_norm_2   = get("pre_ffw_norm_2.weight");
        L.ffn_post_norm_1  = get("post_ffw_norm_1.weight");
        L.ffn_post_norm_2  = get("post_ffw_norm_2.weight");

        // Per-layer embedding
        L.per_layer_inp_gate  = get("per_layer_inp_gate.weight");
        L.per_layer_proj      = get("per_layer_proj.weight");
        L.per_layer_post_norm = get("per_layer_post_norm.weight");

        L.out_scale = get("layer_output_scale.weight");
    }

    std::printf("[gemma4-loader] loaded %d tensors, vocab=%d\n", n_tensors, (int)n_vocab);
    std::fflush(stdout);

    gguf_free(gctx);
    return true;
}

void free_gemma4_weights(Gemma4Weights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    w.layers.clear();
}

// ── Cache ──────────────────────────────────────────────────────────────

bool create_gemma4_cache(ggml_backend_t backend, const Gemma4Weights & w,
                          int max_ctx, Gemma4Cache & out) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(w.n_layer * 2 + 4) + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.k.resize(w.n_layer, nullptr);
    out.v.resize(w.n_layer, nullptr);
    out.kv_source.resize(w.n_layer);

    // SWA layers use a ring buffer of size min(sliding_window, max_ctx).
    const int swa_size = (w.sliding_window > 0 && w.sliding_window < max_ctx)
                             ? w.sliding_window : max_ctx;

    // Determine KV source for each layer
    int last_kv_layer = -1;
    for (int il = 0; il < w.n_layer; ++il) {
        if (w.has_kv[il]) {
            const int D  = gemma4_head_dim(w, il);
            const int Hk = gemma4_n_head_kv(w, il);
            const bool is_swa = gemma4_is_swa_layer(w, il);
            const int cache_len = is_swa ? swa_size : max_ctx;
            out.k[il] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16, D, cache_len, Hk);
            out.v[il] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16, D, cache_len, Hk);
            out.kv_source[il] = il;
            last_kv_layer = il;
        } else {
            // Reuse the most recent KV layer
            out.kv_source[il] = last_kv_layer >= 0 ? last_kv_layer : 0;
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        ggml_free(out.ctx); out.ctx = nullptr;
        return false;
    }

    out.cur_pos = 0;
    out.max_ctx = max_ctx;
    out.n_layer = w.n_layer;
    out.swa_size = swa_size;
    return true;
}

void free_gemma4_cache(Gemma4Cache & c) {
    if (c.feat_buf) { ggml_backend_buffer_free(c.feat_buf); c.feat_buf = nullptr; }
    if (c.feat_ctx) { ggml_free(c.feat_ctx); c.feat_ctx = nullptr; }
    c.target_feat = nullptr;
    c.target_feat_cap = 0;
    c.n_capture_layers = 0;
    c.capture_layer_ids.clear();
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.k.clear(); c.v.clear(); c.kv_source.clear();
    c.cur_pos = 0;
}

bool create_gemma4_target_feat(ggml_backend_t backend, Gemma4Cache & cache,
                                int n_capture_layers, int hidden_size, int cap) {
    if (n_capture_layers <= 0 || hidden_size <= 0 || cap <= 0) return false;

    // Free existing feat allocation
    if (cache.feat_buf) { ggml_backend_buffer_free(cache.feat_buf); cache.feat_buf = nullptr; }
    if (cache.feat_ctx) { ggml_free(cache.feat_ctx); cache.feat_ctx = nullptr; }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + 4096;
    ip.no_alloc = true;
    cache.feat_ctx = ggml_init(ip);
    if (!cache.feat_ctx) return false;

    const int fc_in = n_capture_layers * hidden_size;
    cache.target_feat = ggml_new_tensor_2d(cache.feat_ctx, GGML_TYPE_BF16, fc_in, cap);
    ggml_set_name(cache.target_feat, "gemma4_target_feat");

    cache.feat_buf = ggml_backend_alloc_ctx_tensors(cache.feat_ctx, backend);
    if (!cache.feat_buf) {
        ggml_free(cache.feat_ctx); cache.feat_ctx = nullptr;
        cache.target_feat = nullptr;
        return false;
    }

    cache.target_feat_cap = cap;
    cache.n_capture_layers = n_capture_layers;

    // Compute capture layer IDs using floating-point linspace with rounding.
    // This matches the training config (e.g., gemma4: [1,12,23,35,46,57]).
    cache.capture_layer_ids.resize(n_capture_layers);
    const int n_layer = cache.n_layer;
    for (int k = 0; k < n_capture_layers; k++) {
        cache.capture_layer_ids[k] = (int)std::round(
            1.0 + k * (double)(n_layer - 4) / (n_capture_layers - 1));
    }

    return true;
}

void free_gemma4_snapshot(Gemma4Snapshot & s) {
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    s.k_snap.clear(); s.v_snap.clear();
    s.feat_snap = nullptr;
    s.feat_cap  = 0;
    s.cur_pos   = 0;
    s.last_tok  = -1;
}

}  // namespace dflash::common
