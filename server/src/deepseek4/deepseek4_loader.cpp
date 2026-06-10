// Loads DeepSeek V4 Flash from a GGUF file.
//
// Tensor naming follows the ds4 GGUF conversion:
//   token_embd.weight, output_norm.weight, output.weight,
//   output_hc_base.weight, output_hc_fn.weight, output_hc_scale.weight
//   blk.<i>.attn_norm.weight, blk.<i>.attn_q_a.weight, attn_q_a_norm,
//   attn_q_b, attn_kv, attn_kv_a_norm, attn_sinks, attn_output_a, attn_output_b,
//   attn_compressor_{ape,kv,gate,norm}, indexer.{attn_q_b, proj},
//   indexer_compressor_{ape,kv,gate,norm},
//   hc_attn_fn, hc_attn_scale, hc_attn_base,
//   ffn_norm, ffn_gate_inp, exp_probs_b (bias), ffn_gate_tid2eid,
//   ffn_gate_exps, ffn_up_exps, ffn_down_exps,
//   ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp,
//   hc_ffn_fn, hc_ffn_scale, hc_ffn_base

#include "deepseek4_internal.h"
#include "internal.h"
#include "dflash27b.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_types.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

struct DS4Mmap {
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
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const uint32_t *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_u32(g, id);
}

uint64_t get_u64_or(gguf_context * g, const char * key, uint64_t def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    // Handle both u32 and u64 storage in GGUF
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32) {
        return (uint64_t)gguf_get_val_u32(g, id);
    }
    return (uint64_t)gguf_get_val_u64(g, id);
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

std::vector<uint32_t> get_u32_arr(gguf_context * g, const char * key) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return {};
    const size_t n = gguf_get_arr_n(g, id);
    // Handle both i32 and u32 element types (values are positive either way)
    const void * raw = gguf_get_arr_data(g, id);
    std::vector<uint32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = (uint32_t)((const int32_t *)raw)[i];
    }
    return out;
}

ggml_tensor * find_tensor(ggml_context * ctx, const char * name) {
    return ggml_get_tensor(ctx, name);
}

static size_t align_up_size(size_t x, size_t a) {
    if (a == 0) return x;
    const size_t r = x % a;
    return r == 0 ? x : x + (a - r);
}

static bool parse_block_tensor_name(const char * name, int & layer_id) {
    const char prefix[] = "blk.";
    const size_t prefix_len = sizeof(prefix) - 1;
    if (std::strncmp(name, prefix, prefix_len) != 0) return false;
    const char * p = name + prefix_len;
    if (*p < '0' || *p > '9') return false;
    char * end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (!end || *end != '.' || v < 0 || v > INT32_MAX) return false;
    layer_id = (int)v;
    return true;
}

static bool is_expert_tensor(const char * name) {
    return std::strstr(name, "ffn_gate_exps") != nullptr ||
           std::strstr(name, "ffn_up_exps") != nullptr ||
           std::strstr(name, "ffn_down_exps") != nullptr;
}

static bool should_keep_ds4_tensor(const char * name,
                                   const TargetLoadPlan & plan) {
    // Global tensors
    if (std::strcmp(name, "token_embd.weight") == 0 ||
        std::strcmp(name, "output_norm.weight") == 0 ||
        std::strcmp(name, "output.weight") == 0 ||
        std::strcmp(name, "output_hc_base.weight") == 0 ||
        std::strcmp(name, "output_hc_fn.weight") == 0 ||
        std::strcmp(name, "output_hc_scale.weight") == 0) {
        return plan.load_output;
    }

    int layer_id = -1;
    if (!parse_block_tensor_name(name, layer_id)) return false;
    return layer_id >= plan.layer_begin && layer_id < plan.layer_end;
}

static bool should_upload_ds4_tensor(const char * name,
                                     const TargetLoadPlan & plan) {
    if (!should_keep_ds4_tensor(name, plan)) return false;
    // token_embd stays on CPU for embedding lookup
    if (std::strcmp(name, "token_embd.weight") == 0) return false;
    return !(plan.skip_expert_tensors && is_expert_tensor(name));
}

struct DS4TensorAlloc {
    ggml_tensor * tensor = nullptr;
    size_t file_offset = 0;
    size_t file_size = 0;
    size_t buffer_offset = 0;
    bool upload_to_backend = true;
};

}  // namespace

// ─── Compute per-layer compression ratios (matches ds4.c logic) ─────────
static std::vector<uint32_t> compute_compress_ratios(int n_layer) {
    std::vector<uint32_t> ratios(n_layer, 0);
    for (int il = 0; il < n_layer; il++) {
        if (il < 2) {
            ratios[il] = 0;  // First 2 layers: no compression
        } else if ((il & 1) == 0) {
            ratios[il] = 4;  // Even layers ≥2: ratio 4
        } else {
            ratios[il] = 128;  // Odd layers ≥2: ratio 128
        }
    }
    return ratios;
}

bool load_deepseek4_gguf(const std::string & path,
                          ggml_backend_t backend,
                          DeepSeek4Weights & out) {
    TargetLoadPlan plan;
    return load_deepseek4_gguf_partial(path, backend, plan, out);
}

bool load_deepseek4_gguf_partial(const std::string & path,
                                  ggml_backend_t backend,
                                  const TargetLoadPlan & plan_in,
                                  DeepSeek4Weights & out) {
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
        if (std::string(arch) != "deepseek4") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected deepseek4)");
            gguf_free(gctx); return false;
        }
    }

    // ── Read hyperparameters ────────────────────────────────────────────
    const uint32_t n_layer        = get_u32_or(gctx, "deepseek4.block_count", 43);
    const uint32_t n_embd         = get_u32_or(gctx, "deepseek4.embedding_length", 4096);
    const uint32_t n_vocab        = get_u32_or(gctx, "deepseek4.vocab_size", 129280);
    const uint32_t n_head         = get_u32_or(gctx, "deepseek4.attention.head_count", 64);
    const uint32_t n_head_kv      = get_u32_or(gctx, "deepseek4.attention.head_count_kv", 1);
    const uint32_t head_dim       = get_u32_or(gctx, "deepseek4.attention.key_length", 512);
    const uint32_t n_rot          = get_u32_or(gctx, "deepseek4.rope.dimension_count", 64);
    const uint32_t n_lora_q       = get_u32_or(gctx, "deepseek4.attention.q_lora_rank", 1024);
    const uint32_t n_lora_o       = get_u32_or(gctx, "deepseek4.attention.output_lora_rank", 1024);
    const uint32_t n_out_group    = get_u32_or(gctx, "deepseek4.attention.output_group_count", 8);
    const uint32_t n_expert       = get_u32_or(gctx, "deepseek4.expert_count", 256);
    const uint32_t n_expert_used  = get_u32_or(gctx, "deepseek4.expert_used_count", 6);
    const uint32_t n_expert_shared = get_u32_or(gctx, "deepseek4.expert_shared_count", 1);
    const uint32_t n_ff_exp       = get_u32_or(gctx, "deepseek4.expert_feed_forward_length", 2048);
    const uint32_t n_hash_layer   = get_u32_or(gctx, "deepseek4.hash_layer_count", 3);
    const uint32_t n_swa          = get_u32_or(gctx, "deepseek4.attention.sliding_window", 128);
    const uint32_t n_indexer_head = get_u32_or(gctx, "deepseek4.attention.indexer.head_count", 64);
    const uint32_t n_indexer_head_dim = get_u32_or(gctx, "deepseek4.attention.indexer.key_length", 128);
    const uint32_t n_indexer_top_k = get_u32_or(gctx, "deepseek4.attention.indexer.top_k", 512);
    const uint32_t n_hc           = get_u32_or(gctx, "deepseek4.hyper_connection.count", 4);
    const uint32_t n_hc_sinkhorn  = get_u32_or(gctx, "deepseek4.hyper_connection.sinkhorn_iterations", 20);

    // RoPE parameters
    const float rope_freq_base    = get_f32_or(gctx, "deepseek4.rope.freq_base", 10000.0f);
    const float rope_scale_factor = get_f32_or(gctx, "deepseek4.rope.scaling.factor", 16.0f);
    const float rope_yarn_beta_fast = get_f32_or(gctx, "deepseek4.rope.scaling.yarn_beta_fast", 32.0f);
    const float rope_yarn_beta_slow = get_f32_or(gctx, "deepseek4.rope.scaling.yarn_beta_slow", 1.0f);
    const float compress_rope_freq_base = get_f32_or(gctx, "deepseek4.attention.compress_rope_freq_base", 160000.0f);
    const uint64_t rope_orig_ctx  = get_u64_or(gctx, "deepseek4.rope.scaling.original_context_length", 65536);

    // Other parameters
    const float rms_eps           = get_f32_or(gctx, "deepseek4.attention.layer_norm_rms_epsilon", 1e-6f);
    const float hc_eps            = get_f32_or(gctx, "deepseek4.hyper_connection.epsilon", 1e-6f);
    const float expert_weight_scale = get_f32_or(gctx, "deepseek4.expert_weights_scale", 1.5f);
    const float swiglu_clamp      = get_f32_or(gctx, "deepseek4.swiglu_clamp_exp", 10.0f);

    // Compression ratios from metadata (or compute default)
    std::vector<uint32_t> compress_ratios_meta = get_u32_arr(gctx, "deepseek4.attention.compress_ratios");
    std::vector<uint32_t> compress_ratios;
    if (compress_ratios_meta.size() == n_layer) {
        compress_ratios = compress_ratios_meta;
    } else {
        compress_ratios = compute_compress_ratios((int)n_layer);
    }

    std::fprintf(stderr, "[deepseek4] model: layers=%u embd=%u heads=%u head_dim=%u "
                 "lora_q=%u lora_o=%u out_groups=%u\n",
                 n_layer, n_embd, n_head, head_dim, n_lora_q, n_lora_o, n_out_group);
    std::fprintf(stderr, "[deepseek4] moe: experts=%u used=%u shared=%u ff=%u hash_layers=%u\n",
                 n_expert, n_expert_used, n_expert_shared, n_ff_exp, n_hash_layer);
    std::fprintf(stderr, "[deepseek4] attention: swa=%u rot=%u indexer_heads=%u top_k=%u hc=%u\n",
                 n_swa, n_rot, n_indexer_head, n_indexer_top_k, n_hc);

    // Fill output metadata
    out.n_layer         = (int)n_layer;
    out.n_embd          = (int)n_embd;
    out.n_vocab         = (int)n_vocab;
    out.n_head          = (int)n_head;
    out.n_head_kv       = (int)n_head_kv;
    out.head_dim        = (int)head_dim;
    out.n_rot           = (int)n_rot;
    out.n_out_group     = (int)n_out_group;
    out.n_lora_q        = (int)n_lora_q;
    out.n_lora_o        = (int)n_lora_o;
    out.n_expert        = (int)n_expert;
    out.n_expert_used   = (int)n_expert_used;
    out.n_expert_shared = (int)n_expert_shared;
    out.n_ff_exp        = (int)n_ff_exp;
    out.n_hash_layer    = (int)n_hash_layer;
    out.n_swa           = (int)n_swa;
    out.n_indexer_head  = (int)n_indexer_head;
    out.n_indexer_head_dim = (int)n_indexer_head_dim;
    out.n_indexer_top_k = (int)n_indexer_top_k;
    out.n_hc            = (int)n_hc;
    out.n_hc_sinkhorn_iter = (int)n_hc_sinkhorn;
    out.compress_ratios = compress_ratios;
    out.expert_weight_scale = expert_weight_scale;
    out.rope_freq_base  = rope_freq_base;
    out.rope_scale_factor = rope_scale_factor;
    out.rope_yarn_beta_fast = rope_yarn_beta_fast;
    out.rope_yarn_beta_slow = rope_yarn_beta_slow;
    out.compress_rope_freq_base = compress_rope_freq_base;
    out.rope_orig_ctx   = rope_orig_ctx;
    out.rms_eps         = rms_eps;
    out.hc_eps          = hc_eps;
    out.swiglu_clamp_exp = swiglu_clamp;

    out.layers.resize(n_layer);
    out.backend = backend;

    // ── Build load plan ─────────────────────────────────────────────────
    TargetLoadPlan plan = plan_in;
    if (plan.layer_end < 0) plan.layer_end = (int)n_layer;
    plan.load_output = true;

    // ── Collect tensors for allocation ──────────────────────────────────
    const int n_tensors = gguf_get_n_tensors(gctx);
    const size_t data_offset = gguf_get_data_offset(gctx);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);

    std::vector<DS4TensorAlloc> allocs;
    allocs.reserve(n_tensors);
    size_t total_buf_size = 0;

    for (int ti = 0; ti < n_tensors; ti++) {
        const char * tname = gguf_get_tensor_name(gctx, ti);
        if (!should_keep_ds4_tensor(tname, plan)) continue;

        ggml_tensor * t = find_tensor(meta_ctx, tname);
        if (!t) continue;

        const size_t offset = data_offset + gguf_get_tensor_offset(gctx, ti);
        const bool upload_to_backend = should_upload_ds4_tensor(tname, plan);

        DS4TensorAlloc a;
        a.tensor = t;
        a.file_offset = offset;
        a.file_size = gguf_get_tensor_size(gctx, ti);
        a.upload_to_backend = upload_to_backend;
        if (upload_to_backend) {
            total_buf_size = align_up_size(total_buf_size, alignment);
            a.buffer_offset = total_buf_size;
            total_buf_size += ggml_backend_buft_get_alloc_size(buft, t);
        }
        allocs.push_back(a);
    }

    // ── Allocate GPU buffer ─────────────────────────────────────────────
    ggml_backend_buffer_t buf = nullptr;
    if (total_buf_size > 0) {
        buf = ggml_backend_alloc_buffer(backend, total_buf_size);
        if (!buf) {
            set_last_error("failed to allocate GPU buffer (" + std::to_string(total_buf_size) + " bytes)");
            gguf_free(gctx);
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    out.buf = buf;

    // ── Assign tensors from meta_ctx to the backend buffer ──────────────
    // Use ggml_backend_tensor_alloc to properly set the buffer association.
    out.ctx = meta_ctx;  // Reuse the meta context (tensors already exist)
    char * buf_base = buf ? (char *)ggml_backend_buffer_get_base(buf) : nullptr;
    for (auto & a : allocs) {
        if (!a.upload_to_backend || !buf) continue;
        if (ggml_backend_tensor_alloc(buf, a.tensor, buf_base + a.buffer_offset) != GGML_STATUS_SUCCESS) {
            set_last_error("ggml_backend_tensor_alloc failed");
            ggml_backend_buffer_free(buf); out.buf = nullptr;
            gguf_free(gctx);
            return false;
        }
    }

    // ── Memory-map the file and copy tensor data ────────────────────────
    DS4Mmap mmap;
    std::string mmap_err;
    if (!mmap.open_ro(path, mmap_err)) {
        set_last_error("mmap: " + mmap_err);
        ggml_backend_buffer_free(buf); out.buf = nullptr;
        gguf_free(gctx);
        return false;
    }

    for (auto & a : allocs) {
        if (!a.upload_to_backend) continue;
        const void * src_data = (const char *)mmap.addr + a.file_offset;
        ggml_backend_tensor_set(a.tensor, src_data, 0, a.file_size);
    }
    mmap.close_map();

    // ── Bind tensors to weight struct fields ────────────────────────────
    for (auto & a : allocs) {
        const char * name = ggml_get_name(a.tensor);

        // Global tensors
        if (std::strcmp(name, "token_embd.weight") == 0) { out.tok_embd = a.tensor; continue; }
        if (std::strcmp(name, "output_norm.weight") == 0) { out.out_norm = a.tensor; continue; }
        if (std::strcmp(name, "output.weight") == 0) { out.output = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_base.weight") == 0) { out.output_hc_base = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_fn.weight") == 0) { out.output_hc_fn = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_scale.weight") == 0) { out.output_hc_scale = a.tensor; continue; }

        // Per-layer tensors
        int il = -1;
        if (!parse_block_tensor_name(name, il) || il < 0 || il >= (int)n_layer) continue;
        DeepSeek4Layer & L = out.layers[il];

        // Find the suffix after "blk.<il>."
        const char * p = name;
        while (*p && *p != '.') p++;  // skip "blk"
        if (*p == '.') p++;           // skip first '.'
        while (*p && *p != '.') p++;  // skip layer number
        if (*p == '.') p++;           // skip second '.'
        const std::string suffix(p);

        // Attention
        if (suffix == "attn_norm.weight")          { L.attn_norm = a.tensor; continue; }
        if (suffix == "attn_q_a.weight")           { L.attn_q_a = a.tensor; continue; }
        if (suffix == "attn_q_a_norm.weight")      { L.attn_q_a_norm = a.tensor; continue; }
        if (suffix == "attn_q_b.weight")           { L.attn_q_b = a.tensor; continue; }
        if (suffix == "attn_kv.weight")            { L.attn_kv = a.tensor; continue; }
        if (suffix == "attn_kv_a_norm.weight")     { L.attn_kv_a_norm = a.tensor; continue; }
        if (suffix == "attn_sinks.weight")         { L.attn_sinks = a.tensor; continue; }
        if (suffix == "attn_output_a.weight")      { L.attn_output_a = a.tensor; continue; }
        if (suffix == "attn_output_b.weight")      { L.attn_output_b = a.tensor; continue; }

        // Compressor
        if (suffix == "attn_compressor_ape.weight")  { L.attn_compressor_ape = a.tensor; continue; }
        if (suffix == "attn_compressor_kv.weight")   { L.attn_compressor_kv = a.tensor; continue; }
        if (suffix == "attn_compressor_gate.weight") { L.attn_compressor_gate = a.tensor; continue; }
        if (suffix == "attn_compressor_norm.weight") { L.attn_compressor_norm = a.tensor; continue; }

        // Indexer
        if (suffix == "indexer.attn_q_b.weight")     { L.indexer_attn_q_b = a.tensor; continue; }
        if (suffix == "indexer.proj.weight")          { L.indexer_proj = a.tensor; continue; }
        if (suffix == "indexer_compressor_ape.weight")  { L.indexer_compressor_ape = a.tensor; continue; }
        if (suffix == "indexer_compressor_kv.weight")   { L.indexer_compressor_kv = a.tensor; continue; }
        if (suffix == "indexer_compressor_gate.weight") { L.indexer_compressor_gate = a.tensor; continue; }
        if (suffix == "indexer_compressor_norm.weight") { L.indexer_compressor_norm = a.tensor; continue; }

        // HC attention
        if (suffix == "hc_attn_fn.weight")         { L.hc_attn_fn = a.tensor; continue; }
        if (suffix == "hc_attn_scale.weight")      { L.hc_attn_scale = a.tensor; continue; }
        if (suffix == "hc_attn_base.weight")       { L.hc_attn_base = a.tensor; continue; }

        // FFN
        if (suffix == "ffn_norm.weight")           { L.ffn_norm = a.tensor; continue; }
        if (suffix == "ffn_gate_inp.weight")       { L.ffn_gate_inp = a.tensor; continue; }
        if (suffix == "exp_probs_b.bias")          { L.ffn_exp_probs_b = a.tensor; continue; }
        if (suffix == "ffn_gate_tid2eid.weight")   { L.ffn_gate_tid2eid = a.tensor; continue; }
        if (suffix == "ffn_gate_exps.weight")      { L.ffn_gate_exps = a.tensor; continue; }
        if (suffix == "ffn_up_exps.weight")        { L.ffn_up_exps = a.tensor; continue; }
        if (suffix == "ffn_down_exps.weight")      { L.ffn_down_exps = a.tensor; continue; }
        if (suffix == "ffn_gate_shexp.weight")     { L.ffn_gate_shexp = a.tensor; continue; }
        if (suffix == "ffn_up_shexp.weight")       { L.ffn_up_shexp = a.tensor; continue; }
        if (suffix == "ffn_down_shexp.weight")     { L.ffn_down_shexp = a.tensor; continue; }

        // HC FFN
        if (suffix == "hc_ffn_fn.weight")          { L.hc_ffn_fn = a.tensor; continue; }
        if (suffix == "hc_ffn_scale.weight")       { L.hc_ffn_scale = a.tensor; continue; }
        if (suffix == "hc_ffn_base.weight")        { L.hc_ffn_base = a.tensor; continue; }
    }

    // ── Set up CPU embedder ─────────────────────────────────────────────
    // The embedder is set up using the mmap data directly (like gemma4).
    // For now, we use an owned copy of the token embedding table bytes.
    if (out.tok_embd) {
        // Find tok_embd in the allocs and set up embedder from its data
        for (auto & a : allocs) {
            if (std::strcmp(ggml_get_name(a.tensor), "token_embd.weight") == 0) {
                // Store embedding bytes as owned data for CPU-side embed()
                out.embedder.tok_embd_owned.resize(a.file_size);
                // Re-read from mmap (already closed). Use the GPU tensor instead:
                // Actually, we need the raw bytes for dequantization. Reopen mmap briefly.
                DS4Mmap emb_mmap;
                std::string emb_err;
                if (emb_mmap.open_ro(path, emb_err)) {
                    std::memcpy(out.embedder.tok_embd_owned.data(),
                                (const char *)emb_mmap.addr + a.file_offset, a.file_size);
                    emb_mmap.close_map();
                }
                out.embedder.tok_embd_bytes = out.embedder.tok_embd_owned.data();
                out.embedder.tok_embd_type  = a.tensor->type;
                out.embedder.n_embd         = n_embd;
                out.embedder.n_vocab        = (int64_t)n_vocab;
                out.embedder.row_bytes      = a.file_size / (size_t)n_vocab;
                break;
            }
        }
    }

    gguf_free(gctx);
    // Note: meta_ctx is now owned by out.ctx — do NOT free it here.

    std::fprintf(stderr, "[deepseek4] loaded %zu tensors, %.1f MB GPU buffer\n",
                 allocs.size(), (double)total_buf_size / (1024.0 * 1024.0));
    return true;
}

namespace {

static MoeHybridColdBackend ds4_cold_backend_from_env() {
    const char * value = std::getenv("DFLASH_MOE_COLD_BACKEND");
    if (!value || !value[0]) return MoeHybridColdBackend::Cpu;
    if (std::strcmp(value, "gpu") == 0 || std::strcmp(value, "hip") == 0 ||
        std::strcmp(value, "rocm") == 0) {
        return MoeHybridColdBackend::Gpu;
    }
    return MoeHybridColdBackend::Cpu;
}

static MoeHybridConfig make_ds4_moe_hybrid_config(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    cfg.cold_expert_backend = ds4_cold_backend_from_env();
    return cfg;
}

static MoeLayerDesc make_ds4_moe_layer_desc(const DeepSeek4Layer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps = L.ffn_gate_exps;
    desc.ffn_up_exps = L.ffn_up_exps;
    desc.ffn_down_exps = L.ffn_down_exps;
    desc.ffn_gate_up_exps = nullptr;
    desc.ffn_gate_shexp = L.ffn_gate_shexp;
    desc.ffn_up_shexp = L.ffn_up_shexp;
    desc.ffn_down_shexp = L.ffn_down_shexp;
    desc.ffn_gate_inp_shexp = nullptr;
    return desc;
}

}  // namespace

bool build_deepseek4_moe_hybrid_storage_from_file(
        const std::string & path,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const MoeHybridPlacement & placement,
        MoeHybridStorage & out,
        std::string * err) {
    ggml_context * expert_meta = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &expert_meta;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        if (err) *err = "failed to re-open GGUF for expert loading";
        return false;
    }

    DS4Mmap mmap;
    std::string mmap_err;
    if (!mmap.open_ro(path, mmap_err)) {
        gguf_free(gctx);
        if (expert_meta) ggml_free(expert_meta);
        if (err) *err = mmap_err;
        return false;
    }

    const size_t data_start = gguf_get_data_offset(gctx);
    const auto * file_bytes = static_cast<const uint8_t *>(mmap.addr);
    std::vector<LayerExpertFileData> layer_file_data((size_t)w.n_layer);

    for (int il = 0; il < w.n_layer; ++il) {
        char name[128];
        auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
            std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
            int64_t tid = gguf_find_tensor(gctx, name);
            if (tid < 0) return {};
            const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
            const size_t sz = gguf_get_tensor_size(gctx, tid);
            if (off + sz > mmap.len) return {};
            return { file_bytes + off, sz };
        };

        layer_file_data[(size_t)il].gate_exps = find_tensor_data("ffn_gate_exps");
        layer_file_data[(size_t)il].up_exps = find_tensor_data("ffn_up_exps");
        layer_file_data[(size_t)il].down_exps = find_tensor_data("ffn_down_exps");
    }

    std::vector<MoeLayerDesc> layer_descs((size_t)w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        layer_descs[(size_t)il] = make_ds4_moe_layer_desc(w.layers[(size_t)il]);
    }

    const bool ok = build_moe_hybrid_storage_from_file(
        make_ds4_moe_hybrid_config(w), backend, placement, layer_descs, layer_file_data, out, err);

    mmap.close_map();
    gguf_free(gctx);
    if (expert_meta) ggml_free(expert_meta);
    return ok;
}

void free_deepseek4_weights(DeepSeek4Weights & w) {
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    w.layers.clear();
    w.embedder.tok_embd_owned.clear();
    w.embedder.tok_embd_bytes = nullptr;
    w.moe_hybrid = false;
}

}  // namespace dflash::common
