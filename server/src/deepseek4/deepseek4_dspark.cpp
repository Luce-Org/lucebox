// DeepSeek-V4-Flash "DSpark" drafter loader.  See deepseek4_dspark.h.
//
// Self-contained (shares nothing with the target loader) so it cannot regress
// the target path.  Loads a "deepseek4-dflash-draft" GGUF into a DSparkDrafter:
// the n_layer decoder blocks reuse DeepSeek4Weights leaf-name bindings, and the
// DSpark-specific tensors (dflash.fc / hidden_norm / dspark.*) bind separately.

#include "deepseek4_dspark.h"

#include "common/gguf_bounds.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash::common {

namespace {

std::string g_dspark_err;
void set_err(const std::string & m) { g_dspark_err = m; std::fprintf(stderr, "[ds4-dspark] %s\n", m.c_str()); }

const char * ARCH = "deepseek4-dflash-draft";

uint32_t kv_u32(gguf_context * g, const std::string & key, uint32_t def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return def;
    return gguf_get_val_u32(g, id);
}
float kv_f32(gguf_context * g, const std::string & key, float def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return def;
    return gguf_get_val_f32(g, id);
}

// Read an INT32/UINT32 array KV into ints. Returns false if the key is absent.
bool kv_i32_array(gguf_context * g, const std::string & key, std::vector<int> & out) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return false;
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return false;
    const enum gguf_type at = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    out.resize(n);
    const void * raw = gguf_get_arr_data(g, id);
    if (at == GGUF_TYPE_INT32) {
        const int32_t * v = static_cast<const int32_t *>(raw);
        for (size_t i = 0; i < n; i++) out[i] = (int)v[i];
    } else if (at == GGUF_TYPE_UINT32) {
        const uint32_t * v = static_cast<const uint32_t *>(raw);
        for (size_t i = 0; i < n; i++) out[i] = (int)v[i];
    } else {
        out.clear();
        return false;
    }
    return true;
}

// Suffix after "blk.<il>." — returns "" if `name` isn't a block tensor.
bool block_suffix(const char * name, int & il, std::string & suffix) {
    if (std::strncmp(name, "blk.", 4) != 0) return false;
    const char * p = name + 4;
    if (*p < '0' || *p > '9') return false;
    char * end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (!end || *end != '.' || v < 0) return false;
    il = (int)v;
    suffix = std::string(end + 1);
    return true;
}

bool recognized_block_suffix(const std::string & suffix) {
    static const char * const names[] = {
        "attn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight",
        "attn_q_b.weight", "attn_kv.weight", "attn_kv_a_norm.weight",
        "attn_sinks.weight", "attn_output_a.weight", "attn_output_b.weight",
        "hc_attn_fn.weight", "hc_attn_scale.weight", "hc_attn_base.weight",
        "ffn_norm.weight", "ffn_gate_inp.weight", "exp_probs_b.bias",
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
        "hc_ffn_fn.weight", "hc_ffn_scale.weight", "hc_ffn_base.weight",
    };
    for (const char * name : names) {
        if (suffix == name) return true;
    }
    return false;
}

bool recognized_dspark_tensor(const char * name, int n_layer) {
    static const char * const names[] = {
        "output_norm.weight", "output_hc_base.weight", "output_hc_fn.weight",
        "output_hc_scale.weight", "dflash.fc.weight",
        "dflash.hidden_norm.weight", "dflash.dspark.markov.w1",
        "dflash.dspark.markov.w2", "dflash.dspark.confidence.weight",
        "dflash.dspark.confidence.bias",
    };
    for (const char * expected : names) {
        if (std::strcmp(name, expected) == 0) return true;
    }
    int il = -1;
    std::string suffix;
    return block_suffix(name, il, suffix) && il >= 0 && il < n_layer &&
           recognized_block_suffix(suffix);
}

bool tensor_shape_is(const ggml_tensor * tensor,
                     std::initializer_list<int64_t> dims) {
    if (!tensor || dims.size() > GGML_MAX_DIMS) return false;
    size_t i = 0;
    for (int64_t dim : dims) {
        if (tensor->ne[i++] != dim) return false;
    }
    for (; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] != 1) return false;
    }
    return true;
}

}  // namespace

const char * deepseek4_dspark_last_error() { return g_dspark_err.c_str(); }

bool load_deepseek4_dspark_drafter(const std::string & path,
                                   ggml_backend_t backend,
                                   DSparkDrafter & out) {
    g_dspark_err.clear();
    ggml_context * meta = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &meta;
    gguf_context * g = gguf_init_from_file(path.c_str(), gip);
    if (!g) { set_err("gguf_init failed: " + path); return false; }

    // ── Arch check ──────────────────────────────────────────────────────
    {
        const int64_t aid = gguf_find_key(g, "general.architecture");
        if (aid < 0) { set_err("missing general.architecture"); gguf_free(g); if (meta) ggml_free(meta); return false; }
        const char * arch = gguf_get_val_str(g, aid);
        if (std::string(arch) != ARCH) {
            set_err(std::string("unexpected arch: ") + arch + " (expected " + ARCH + ")");
            gguf_free(g); if (meta) ggml_free(meta); return false;
        }
    }

    const std::string P = std::string(ARCH) + ".";
    DeepSeek4Weights & w = out.core;

    // ── Core hparams (mirror the target loader's defaults) ──────────────
    const uint32_t n_layer  = kv_u32(g, P + "block_count", 3);
    w.n_layer          = (int)n_layer;
    w.n_embd           = (int)kv_u32(g, P + "embedding_length", 4096);
    w.n_vocab          = (int)kv_u32(g, P + "vocab_size", 129280);
    w.n_head           = (int)kv_u32(g, P + "attention.head_count", 64);
    w.n_head_kv        = (int)kv_u32(g, P + "attention.head_count_kv", 1);
    w.head_dim         = (int)kv_u32(g, P + "attention.key_length", 512);
    w.n_rot            = (int)kv_u32(g, P + "rope.dimension_count", 64);
    w.n_lora_q         = (int)kv_u32(g, P + "attention.q_lora_rank", 1024);
    w.n_lora_o         = (int)kv_u32(g, P + "attention.output_lora_rank", 1024);
    w.n_out_group      = (int)kv_u32(g, P + "attention.output_group_count", 8);
    w.n_expert         = (int)kv_u32(g, P + "expert_count", 256);
    w.n_expert_used    = (int)kv_u32(g, P + "expert_used_count", 6);
    w.n_expert_shared  = (int)kv_u32(g, P + "expert_shared_count", 1);
    w.n_ff_exp         = (int)kv_u32(g, P + "expert_feed_forward_length", 2048);
    w.n_hash_layer     = (int)kv_u32(g, P + "hash_layer_count", 3);
    w.n_swa            = (int)kv_u32(g, P + "attention.sliding_window", 128);
    w.n_indexer_head   = (int)kv_u32(g, P + "attention.indexer.head_count", 64);
    w.n_indexer_head_dim = (int)kv_u32(g, P + "attention.indexer.key_length", 128);
    w.n_indexer_top_k  = (int)kv_u32(g, P + "attention.indexer.top_k", 512);
    w.n_hc             = (int)kv_u32(g, P + "hyper_connection.count", 4);
    w.n_hc_sinkhorn_iter = (int)kv_u32(g, P + "hyper_connection.sinkhorn_iterations", 20);
    w.expert_weight_scale = kv_f32(g, P + "expert_weights_scale", 1.5f);
    w.rope_freq_base   = kv_f32(g, P + "rope.freq_base", 10000.0f);
    w.rope_scale_factor = kv_f32(g, P + "rope.scaling.factor", 16.0f);
    w.rope_yarn_beta_fast = kv_f32(g, P + "rope.scaling.yarn_beta_fast", 32.0f);
    w.rope_yarn_beta_slow = kv_f32(g, P + "rope.scaling.yarn_beta_slow", 1.0f);
    w.compress_rope_freq_base = kv_f32(g, P + "attention.compress_rope_freq_base", 160000.0f);
    w.rms_eps          = kv_f32(g, P + "attention.layer_norm_rms_epsilon", 1e-6f);
    w.hc_eps           = kv_f32(g, P + "hyper_connection.epsilon", 1e-6f);
    w.swiglu_clamp_exp = kv_f32(g, P + "swiglu_clamp_exp", 10.0f);
    w.eos_id = -1;        // drafter carries no tokenizer; EOS comes from the target
    w.eos_chat_id = -1;

    // DSpark drafter layers have NO KV compression (DSparkAttention asserts
    // compress_ratio==0). Force all-zero so no compressor/indexer tensors are
    // expected — do NOT run compute_compress_ratios (would give [0,0,4,...]).
    w.compress_ratios.assign(n_layer, 0u);

    // The DSpark blocks are NOT hash-routing layers: in the checkpoint their
    // real layer ids are n_layers+stage = 43/44/45, all >= num_hash_layers (3),
    // and the drafter GGUF carries no ffn_gate_tid2eid. Force 0 so build_moe_ffn
    // always takes the score-routed (sqrt-softplus + top-k) path.
    w.n_hash_layer = 0;

    // ── DFlash / DSpark metadata ────────────────────────────────────────
    out.n_target_layers = (int)kv_u32(g, P + "dflash.n_target_layers", 3);
    out.block_size      = (int)kv_u32(g, P + "dflash.block_size", 5);
    out.mask_token_id   = (int)kv_u32(g, P + "dflash.mask_token_id", 128799);
    out.head_hc_enabled = kv_u32(g, P + "dflash.head_hc_enabled", 0) != 0;
    if (!kv_i32_array(g, P + "dflash.capture_layer_ids", out.capture_layer_ids)) {
        out.capture_layer_ids.clear();
    }
    out.dspark_enabled  = kv_u32(g, P + "dflash.dspark.enabled", 0) != 0;
    out.markov_rank     = (int)kv_u32(g, P + "dflash.dspark.markov_rank", 256);
    out.vocab_size      = (int)kv_u32(g, P + "dflash.dspark.vocab_size", w.n_vocab);
    out.confidence_dim  = (int)kv_u32(g, P + "dflash.dspark.confidence_dim", 0);

    const bool valid_arch =
        w.n_layer > 0 && w.n_layer <= 64 &&
        w.n_embd > 0 && w.n_embd <= 32768 &&
        w.n_vocab > 0 && w.n_vocab <= 1000000 &&
        w.n_head > 0 && w.n_head <= 512 &&
        w.head_dim > 0 && w.head_dim <= 4096 &&
        w.n_lora_q > 0 && w.n_lora_q <= 32768 &&
        w.n_lora_o > 0 && w.n_lora_o <= 32768 &&
        w.n_out_group > 0 && w.n_out_group <= w.n_head &&
        w.n_head % w.n_out_group == 0 &&
        w.n_expert > 0 && w.n_expert <= 4096 &&
        w.n_expert_used > 0 && w.n_expert_used <= w.n_expert &&
        w.n_ff_exp > 0 && w.n_ff_exp <= 131072 &&
        w.n_hc > 0 && w.n_hc <= 64 &&
        out.n_target_layers > 0 && out.n_target_layers <= 64 &&
        out.block_size >= 2 && out.block_size <= 16 &&
        out.markov_rank > 0 && out.markov_rank <= 32768 &&
        out.vocab_size == w.n_vocab;
    if (!valid_arch) {
        set_err("invalid DSpark architecture metadata");
        gguf_free(g);
        if (meta) ggml_free(meta);
        out = DSparkDrafter{};
        return false;
    }

    w.layers.resize(n_layer);
    w.backend = backend;

    // Validate the complete tensor table before allocating device memory or
    // forming an absolute file offset. The drafter contract intentionally has
    // no embedding/lm-head tensors; rejecting unknown tensors prevents a stray
    // full-model tensor from consuming the entire GPU allocation.
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        set_err("open failed: " + path);
        gguf_free(g);
        ggml_free(meta);
        out = DSparkDrafter{};
        return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
        set_err("fstat failed: " + path);
        ::close(fd);
        gguf_free(g);
        ggml_free(meta);
        out = DSparkDrafter{};
        return false;
    }
    const size_t file_size = (size_t) st.st_size;
    const size_t data_off = gguf_get_data_offset(g);
    const int64_t n_tensors = gguf_get_n_tensors(g);
    bool ok = true;
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * tname = gguf_get_tensor_name(g, ti);
        ggml_tensor * tensor = ggml_get_tensor(meta, tname);
        if (!tensor) {
            set_err(std::string("meta tensor missing: ") + tname);
            ok = false;
            break;
        }
        if (!recognized_dspark_tensor(tname, w.n_layer)) {
            set_err(std::string("unexpected DSpark tensor: ") + tname);
            ok = false;
            break;
        }
        const size_t tensor_off = gguf_get_tensor_offset(g, ti);
        const size_t size = gguf_get_tensor_size(g, ti);
        if (!gguf_tensor_in_file(data_off, tensor_off, size, file_size)) {
            set_err(gguf_bounds_error("DSpark draft GGUF", tname,
                                      ggml_type_name(gguf_get_tensor_type(g, ti)),
                                      data_off, tensor_off, size, file_size));
            ok = false;
            break;
        }
    }
    if (!ok) {
        ::close(fd);
        gguf_free(g);
        ggml_free(meta);
        out = DSparkDrafter{};
        return false;
    }

    // ── Allocate validated tensors into one backend buffer ──────────────
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(meta, backend);
    if (!buf) {
        set_err("ggml_backend_alloc_ctx_tensors failed");
        ::close(fd);
        gguf_free(g);
        ggml_free(meta);
        out = DSparkDrafter{};
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // ── Stream tensor bytes from the file (pread per tensor) ────────────
    std::vector<char> staging;
    for (int64_t ti = 0; ti < n_tensors && ok; ti++) {
        const char * tname = gguf_get_tensor_name(g, ti);
        ggml_tensor * t = ggml_get_tensor(meta, tname);
        const size_t tensor_off = gguf_get_tensor_offset(g, ti);
        const size_t off  = data_off + tensor_off;  // preflight proved no overflow
        const size_t size = gguf_get_tensor_size(g, ti);
        staging.resize(size);
        size_t done = 0;
        while (done < size) {
            const ssize_t r = ::pread(fd, staging.data() + done, size - done, (off_t)(off + done));
            if (r <= 0) { set_err(std::string("pread failed for ") + tname); ok = false; break; }
            done += (size_t)r;
        }
        if (!ok) break;
        ggml_backend_tensor_set(t, staging.data(), 0, size);
    }
    ::close(fd);
    if (!ok) {
        ggml_backend_buffer_free(buf);
        gguf_free(g);
        ggml_free(meta);
        out = DSparkDrafter{};
        return false;
    }

    // ── Bind pointers by name ───────────────────────────────────────────
    for (int64_t ti = 0; ti < n_tensors; ti++) {
        const char * name = gguf_get_tensor_name(g, ti);
        ggml_tensor * t = ggml_get_tensor(meta, name);

        if (std::strcmp(name, "output_norm.weight") == 0)     { w.out_norm = t; continue; }
        if (std::strcmp(name, "output_hc_base.weight") == 0)  { w.output_hc_base = t; continue; }
        if (std::strcmp(name, "output_hc_fn.weight") == 0)    { w.output_hc_fn = t; continue; }
        if (std::strcmp(name, "output_hc_scale.weight") == 0) { w.output_hc_scale = t; continue; }
        if (std::strcmp(name, "dflash.fc.weight") == 0)          { out.main_proj = t; continue; }
        if (std::strcmp(name, "dflash.hidden_norm.weight") == 0) { out.main_norm = t; continue; }
        if (std::strcmp(name, "dflash.dspark.markov.w1") == 0)   { out.markov_w1 = t; continue; }
        if (std::strcmp(name, "dflash.dspark.markov.w2") == 0)   { out.markov_w2 = t; continue; }
        if (std::strcmp(name, "dflash.dspark.confidence.weight") == 0) { out.confidence_w = t; continue; }
        if (std::strcmp(name, "dflash.dspark.confidence.bias") == 0)   { out.confidence_b = t; continue; }

        int il = -1; std::string suffix;
        if (!block_suffix(name, il, suffix) || il < 0 || il >= (int)n_layer) continue;
        DeepSeek4Layer & L = w.layers[il];

        if      (suffix == "attn_norm.weight")       L.attn_norm       = t;
        else if (suffix == "attn_q_a.weight")        L.attn_q_a        = t;
        else if (suffix == "attn_q_a_norm.weight")   L.attn_q_a_norm   = t;
        else if (suffix == "attn_q_b.weight")        L.attn_q_b        = t;
        else if (suffix == "attn_kv.weight")         L.attn_kv         = t;
        else if (suffix == "attn_kv_a_norm.weight")  L.attn_kv_a_norm  = t;
        else if (suffix == "attn_sinks.weight")      L.attn_sinks      = t;
        else if (suffix == "attn_output_a.weight")   L.attn_output_a   = t;
        else if (suffix == "attn_output_b.weight")   L.attn_output_b   = t;
        else if (suffix == "hc_attn_fn.weight")      L.hc_attn_fn      = t;
        else if (suffix == "hc_attn_scale.weight")   L.hc_attn_scale   = t;
        else if (suffix == "hc_attn_base.weight")    L.hc_attn_base    = t;
        else if (suffix == "ffn_norm.weight")        L.ffn_norm        = t;
        else if (suffix == "ffn_gate_inp.weight")    L.ffn_gate_inp    = t;
        else if (suffix == "exp_probs_b.bias")       L.ffn_exp_probs_b = t;
        else if (suffix == "ffn_gate_exps.weight")   L.ffn_gate_exps   = t;
        else if (suffix == "ffn_up_exps.weight")     L.ffn_up_exps     = t;
        else if (suffix == "ffn_down_exps.weight")   L.ffn_down_exps   = t;
        else if (suffix == "ffn_gate_shexp.weight")  L.ffn_gate_shexp  = t;
        else if (suffix == "ffn_up_shexp.weight")    L.ffn_up_shexp    = t;
        else if (suffix == "ffn_down_shexp.weight")  L.ffn_down_shexp  = t;
        else if (suffix == "hc_ffn_fn.weight")       L.hc_ffn_fn       = t;
        else if (suffix == "hc_ffn_scale.weight")    L.hc_ffn_scale    = t;
        else if (suffix == "hc_ffn_base.weight")     L.hc_ffn_base     = t;
    }

    // Reject incomplete artifacts here, before production code can build a
    // graph that dereferences a missing tensor. The original smoke test
    // checked these bindings only after load_deepseek4_dspark_drafter had
    // already returned success.
    std::vector<std::string> missing;
    auto need = [&](const void * ptr, const std::string & name) {
        if (!ptr) missing.push_back(name);
    };
    need(out.main_proj, "dflash.fc.weight");
    need(out.main_norm, "dflash.hidden_norm.weight");
    need(out.markov_w1, "dflash.dspark.markov.w1");
    need(out.markov_w2, "dflash.dspark.markov.w2");
    need(w.out_norm, "output_norm.weight");
    need(w.output_hc_fn, "output_hc_fn.weight");
    need(w.output_hc_scale, "output_hc_scale.weight");
    need(w.output_hc_base, "output_hc_base.weight");
    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        const std::string p = "blk." + std::to_string(il) + ".";
        need(L.attn_norm, p + "attn_norm.weight");
        need(L.attn_q_a, p + "attn_q_a.weight");
        need(L.attn_q_a_norm, p + "attn_q_a_norm.weight");
        need(L.attn_q_b, p + "attn_q_b.weight");
        need(L.attn_kv, p + "attn_kv.weight");
        need(L.attn_kv_a_norm, p + "attn_kv_a_norm.weight");
        need(L.attn_output_a, p + "attn_output_a.weight");
        need(L.attn_output_b, p + "attn_output_b.weight");
        need(L.hc_attn_fn, p + "hc_attn_fn.weight");
        need(L.hc_attn_scale, p + "hc_attn_scale.weight");
        need(L.hc_attn_base, p + "hc_attn_base.weight");
        need(L.ffn_norm, p + "ffn_norm.weight");
        need(L.ffn_gate_inp, p + "ffn_gate_inp.weight");
        need(L.ffn_gate_exps, p + "ffn_gate_exps.weight");
        need(L.ffn_up_exps, p + "ffn_up_exps.weight");
        need(L.ffn_down_exps, p + "ffn_down_exps.weight");
        need(L.ffn_gate_shexp, p + "ffn_gate_shexp.weight");
        need(L.ffn_up_shexp, p + "ffn_up_shexp.weight");
        need(L.ffn_down_shexp, p + "ffn_down_shexp.weight");
        need(L.hc_ffn_fn, p + "hc_ffn_fn.weight");
        need(L.hc_ffn_scale, p + "hc_ffn_scale.weight");
        need(L.hc_ffn_base, p + "hc_ffn_base.weight");
    }

    std::string shape_error;
    auto expect_shape = [&](const ggml_tensor * tensor, const std::string & name,
                            std::initializer_list<int64_t> dims) {
        if (tensor && shape_error.empty() && !tensor_shape_is(tensor, dims)) {
            shape_error = name + " has a shape incompatible with DSpark metadata";
        }
    };
    const int64_t n_embd = w.n_embd;
    const int64_t hc_dim = n_embd * w.n_hc;
    const int64_t mix_dim = 2LL * w.n_hc + (int64_t) w.n_hc * w.n_hc;
    const int64_t q_dim = (int64_t) w.n_head * w.head_dim;
    const int64_t group_dim = (int64_t) w.head_dim * (w.n_head / w.n_out_group);
    const int64_t out_low_dim = (int64_t) w.n_lora_o * w.n_out_group;

    expect_shape(out.main_proj, "dflash.fc.weight",
                 {(int64_t) out.n_target_layers * n_embd, n_embd});
    expect_shape(out.main_norm, "dflash.hidden_norm.weight", {n_embd});
    expect_shape(out.markov_w1, "dflash.dspark.markov.w1",
                 {out.markov_rank, out.vocab_size});
    expect_shape(out.markov_w2, "dflash.dspark.markov.w2",
                 {out.markov_rank, out.vocab_size});
    expect_shape(w.out_norm, "output_norm.weight", {n_embd});
    expect_shape(w.output_hc_fn, "output_hc_fn.weight", {hc_dim, w.n_hc});
    expect_shape(w.output_hc_scale, "output_hc_scale.weight", {1});
    expect_shape(w.output_hc_base, "output_hc_base.weight", {w.n_hc});
    if (out.confidence_w && out.confidence_dim > 0) {
        expect_shape(out.confidence_w, "dflash.dspark.confidence.weight",
                     {out.confidence_dim, 1});
        expect_shape(out.confidence_b, "dflash.dspark.confidence.bias", {1});
    }
    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        const std::string p = "blk." + std::to_string(il) + ".";
        expect_shape(L.attn_norm, p + "attn_norm.weight", {n_embd});
        expect_shape(L.attn_q_a, p + "attn_q_a.weight", {n_embd, w.n_lora_q});
        expect_shape(L.attn_q_a_norm, p + "attn_q_a_norm.weight", {w.n_lora_q});
        expect_shape(L.attn_q_b, p + "attn_q_b.weight", {w.n_lora_q, q_dim});
        expect_shape(L.attn_kv, p + "attn_kv.weight", {n_embd, w.head_dim});
        expect_shape(L.attn_kv_a_norm, p + "attn_kv_a_norm.weight", {w.head_dim});
        if (L.attn_sinks) expect_shape(L.attn_sinks, p + "attn_sinks.weight", {w.n_head});
        expect_shape(L.attn_output_a, p + "attn_output_a.weight",
                     {group_dim, out_low_dim});
        expect_shape(L.attn_output_b, p + "attn_output_b.weight",
                     {out_low_dim, n_embd});
        expect_shape(L.hc_attn_fn, p + "hc_attn_fn.weight", {hc_dim, mix_dim});
        expect_shape(L.hc_attn_scale, p + "hc_attn_scale.weight", {3});
        expect_shape(L.hc_attn_base, p + "hc_attn_base.weight", {mix_dim});
        expect_shape(L.ffn_norm, p + "ffn_norm.weight", {n_embd});
        expect_shape(L.ffn_gate_inp, p + "ffn_gate_inp.weight", {n_embd, w.n_expert});
        if (L.ffn_exp_probs_b) {
            expect_shape(L.ffn_exp_probs_b, p + "exp_probs_b.bias", {w.n_expert});
        }
        expect_shape(L.ffn_gate_exps, p + "ffn_gate_exps.weight",
                     {n_embd, w.n_ff_exp, w.n_expert});
        expect_shape(L.ffn_up_exps, p + "ffn_up_exps.weight",
                     {n_embd, w.n_ff_exp, w.n_expert});
        expect_shape(L.ffn_down_exps, p + "ffn_down_exps.weight",
                     {w.n_ff_exp, n_embd, w.n_expert});
        expect_shape(L.ffn_gate_shexp, p + "ffn_gate_shexp.weight", {n_embd, w.n_ff_exp});
        expect_shape(L.ffn_up_shexp, p + "ffn_up_shexp.weight", {n_embd, w.n_ff_exp});
        expect_shape(L.ffn_down_shexp, p + "ffn_down_shexp.weight", {w.n_ff_exp, n_embd});
        expect_shape(L.hc_ffn_fn, p + "hc_ffn_fn.weight", {hc_dim, mix_dim});
        expect_shape(L.hc_ffn_scale, p + "hc_ffn_scale.weight", {3});
        expect_shape(L.hc_ffn_base, p + "hc_ffn_base.weight", {mix_dim});
    }

    std::string contract_error;
    if (!out.dspark_enabled) {
        contract_error = "dflash.dspark.enabled is false";
    } else if ((int) out.capture_layer_ids.size() != out.n_target_layers) {
        contract_error = "capture_layer_ids count does not match dflash.n_target_layers";
    } else if ((out.confidence_w == nullptr) != (out.confidence_b == nullptr)) {
        contract_error = "incomplete DSpark confidence tensor pair";
    } else if (out.confidence_w && out.confidence_dim <= 0) {
        contract_error = "DSpark confidence metadata is invalid";
    } else if (!shape_error.empty()) {
        contract_error = shape_error;
    }
    if (!missing.empty() || !contract_error.empty()) {
        std::string message = "invalid DSpark drafter GGUF";
        if (!contract_error.empty()) message += ": " + contract_error;
        if (!missing.empty()) {
            message += "; missing ";
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i != 0) message += ", ";
                message += missing[i];
            }
        }
        set_err(message);
        ggml_backend_buffer_free(buf);
        gguf_free(g);
        ggml_free(meta);
        out = DSparkDrafter{};
        return false;
    }

    w.ctx = meta;
    w.buf = buf;
    gguf_free(g);  // meta_ctx now owned by w.ctx; do not free here

    std::fprintf(stderr,
        "[ds4-dspark] loaded %s: n_layer=%d n_embd=%d vocab=%d block_size=%d "
        "n_target_layers=%d markov_rank=%d confidence_dim=%d mask_tok=%d dspark=%d head_hc=%d\n",
        path.c_str(), w.n_layer, w.n_embd, w.n_vocab, out.block_size,
        out.n_target_layers, out.markov_rank, out.confidence_dim, out.mask_token_id,
        (int)out.dspark_enabled, (int)out.head_hc_enabled);
    return true;
}

void free_deepseek4_dspark_drafter(DSparkDrafter & d) {
    reset_deepseek4_dspark_runtime_cache();
    if (d.core.buf) { ggml_backend_buffer_free(d.core.buf); d.core.buf = nullptr; }
    if (d.core.ctx) { ggml_free(d.core.ctx); d.core.ctx = nullptr; }
    d = DSparkDrafter{};
}

// Validation dump used by the load smoke test.
void deepseek4_dspark_dump(const DSparkDrafter & d) {
    const DeepSeek4Weights & w = d.core;
    auto shp = [](ggml_tensor * t) -> std::string {
        if (!t) return "NULL";
        char b[128];
        std::snprintf(b, sizeof(b), "[%lld,%lld,%lld,%lld] %s",
                      (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                      ggml_type_name(t->type));
        return b;
    };
    std::fprintf(stderr, "── DSpark drafter dump ──\n");
    std::fprintf(stderr, "  main_proj    %s\n", shp(d.main_proj).c_str());
    std::fprintf(stderr, "  main_norm    %s\n", shp(d.main_norm).c_str());
    std::fprintf(stderr, "  out_norm     %s\n", shp(w.out_norm).c_str());
    std::fprintf(stderr, "  output_hc_fn %s\n", shp(w.output_hc_fn).c_str());
    std::fprintf(stderr, "  markov_w1    %s\n", shp(d.markov_w1).c_str());
    std::fprintf(stderr, "  markov_w2    %s\n", shp(d.markov_w2).c_str());
    std::fprintf(stderr, "  conf_w       %s\n", shp(d.confidence_w).c_str());
    std::fprintf(stderr, "  conf_b       %s\n", shp(d.confidence_b).c_str());
    for (int il = 0; il < w.n_layer; il++) {
        const DeepSeek4Layer & L = w.layers[il];
        std::fprintf(stderr, "  blk.%d: attn_norm=%s q_a=%s q_b=%s kv=%s o_a=%s o_b=%s sinks=%s\n",
            il, shp(L.attn_norm).c_str(), shp(L.attn_q_a).c_str(), shp(L.attn_q_b).c_str(),
            shp(L.attn_kv).c_str(), shp(L.attn_output_a).c_str(), shp(L.attn_output_b).c_str(),
            shp(L.attn_sinks).c_str());
        std::fprintf(stderr, "         ffn_gate_exps=%s up=%s down=%s shexp(g=%s) gate_inp=%s probs_b=%s hc_attn_fn=%s hc_ffn_fn=%s\n",
            shp(L.ffn_gate_exps).c_str(), shp(L.ffn_up_exps).c_str(), shp(L.ffn_down_exps).c_str(),
            shp(L.ffn_gate_shexp).c_str(), shp(L.ffn_gate_inp).c_str(), shp(L.ffn_exp_probs_b).c_str(),
            shp(L.hc_attn_fn).c_str(), shp(L.hc_ffn_fn).c_str());
    }
}

}  // namespace dflash::common
