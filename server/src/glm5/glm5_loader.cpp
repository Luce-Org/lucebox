// GLM-5.3-Flash GGUF loader (upstream llama.cpp glm5-next tensor names).
// mmaps each shard and copies tensors unconverted to their layer's device.

#include "glm5_model.h"

#include "common/gguf_kv.h"
#include "common/gguf_mmap.h"
#include "internal.h"

#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

namespace {

constexpr const char * kArch = "glm5-next";

std::string key(const char * suffix) {
    return std::string(kArch) + "." + suffix;
}

struct Shard {
    std::string    path;
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;   // tensor shapes, no data
    GgufMmap       map;

    ~Shard() {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
    }
};

// "<prefix>-00001-of-0000N.gguf" -> all N shard paths.
std::vector<std::string> shard_paths(const std::string & first, int split_count) {
    if (split_count <= 1) return {first};
    const std::string tail = "-00001-of-";
    const size_t at = first.rfind(tail);
    if (at == std::string::npos) return {first};
    const std::string prefix = first.substr(0, at);
    std::vector<std::string> paths;
    for (int i = 1; i <= split_count; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", i, split_count);
        paths.push_back(prefix + buf);
    }
    return paths;
}

bool open_shards(const std::string & path, std::vector<std::unique_ptr<Shard>> & out,
                 bool want_mmap) {
    auto open_one = [&](const std::string & p) -> Shard * {
        auto s = std::make_unique<Shard>();
        s->path = p;
        gguf_init_params ip{};
        ip.no_alloc = true;
        ip.ctx = &s->meta;
        s->gguf = gguf_init_from_file(p.c_str(), ip);
        if (!s->gguf) {
            set_last_error("glm5: cannot read GGUF " + p);
            return nullptr;
        }
        if (want_mmap) {
            std::string err;
            if (!s->map.open(p, err)) {
                set_last_error("glm5: " + err);
                return nullptr;
            }
        }
        out.push_back(std::move(s));
        return out.back().get();
    };

    Shard * first = open_one(path);
    if (!first) return false;
    const int64_t split_id = gguf_find_key(first->gguf, "split.count");
    const int split_count = split_id >= 0 ? (int) gguf_get_val_u16(first->gguf, split_id) : 1;
    const std::vector<std::string> paths = shard_paths(path, split_count);
    if (split_count > 1 && paths.size() != (size_t) split_count) {
        set_last_error("glm5: " + path + " is a split GGUF but its name does not end in "
                       "-00001-of-NNNNN.gguf");
        return false;
    }
    for (size_t i = 1; i < paths.size(); ++i) {
        if (!open_one(paths[i])) return false;
    }
    return true;
}

// Per-layer arrays may be stored as an array or a scalar.
bool get_arr_u32(const gguf_context * g, const std::string & k, int n, std::vector<uint32_t> & out) {
    out.assign(n, 0);
    const int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) return false;
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) {
        std::fill(out.begin(), out.end(), gguf_get_val_u32(g, id));
        return true;
    }
    const size_t len = gguf_get_arr_n(g, id);
    const gguf_type t = gguf_get_arr_type(g, id);
    if ((int) len < n) return false;
    const void * data = gguf_get_arr_data(g, id);
    for (int i = 0; i < n; ++i) {
        if (t == GGUF_TYPE_UINT32) out[i] = static_cast<const uint32_t *>(data)[i];
        else if (t == GGUF_TYPE_INT32) out[i] = (uint32_t) static_cast<const int32_t *>(data)[i];
        else return false;
    }
    return true;
}

bool get_arr_f32(const gguf_context * g, const std::string & k, int n, float fallback,
                 std::vector<float> & out) {
    out.assign(n, fallback);
    const int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) return true;
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) {
        std::fill(out.begin(), out.end(), gguf_get_val_f32(g, id));
        return true;
    }
    if ((int) gguf_get_arr_n(g, id) < n || gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32) {
        return false;
    }
    const float * data = static_cast<const float *>(gguf_get_arr_data(g, id));
    std::copy(data, data + n, out.begin());
    return true;
}

bool get_bool_or(const gguf_context * g, const std::string & k, bool fallback) {
    const int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) return fallback;
    return gguf_get_val_bool(g, id);
}

bool read_hparams(const gguf_context * g, Glm5Weights & w, std::vector<uint32_t> & head_kv) {
    const int64_t arch_id = gguf_find_key(g, "general.architecture");
    const char * arch = arch_id >= 0 ? gguf_get_val_str(g, arch_id) : "";
    if (std::strcmp(arch, kArch) != 0) {
        set_last_error(std::string("glm5: general.architecture is '") + arch +
                       "', expected 'glm5-next'");
        return false;
    }

    w.n_layer     = (int) get_u32_or(g, key("block_count"), 0);
    w.n_embd      = (int) get_u32_or(g, key("embedding_length"), 0);
    w.n_head      = (int) get_u32_or(g, key("attention.head_count"), 0);
    w.n_ctx_train = (int) get_u32_or(g, key("context_length"), 0);
    w.n_ff        = (int) get_u32_or(g, key("feed_forward_length"), 0);
    w.rms_eps     = get_f32_or(g, key("attention.layer_norm_rms_epsilon"), 1e-5f);
    w.norm_eps    = get_f32_or(g, key("attention.layer_norm_epsilon"), 1e-6f);

    w.kda_head_dim = (int) get_u32_or(g, key("kda.head_dim"), 0);
    w.ssm_d_conv   = (int) get_u32_or(g, key("ssm.conv_kernel"), 0);
    w.kda_gate_lower_bound = get_f32_or(g, key("kda.gate_lower_bound"), -5.0f);

    w.q_lora  = (int) get_u32_or(g, key("attention.q_lora_rank"), 0);
    w.kv_lora = (int) get_u32_or(g, key("attention.kv_lora_rank"), 0);
    w.qk_head = (int) get_u32_or(g, key("attention.key_length_mla"), 0);
    w.v_head  = (int) get_u32_or(g, key("attention.value_length_mla"), 0);
    const int n_rot = (int) get_u32_or(g, key("rope.dimension_count"), 0);

    w.idx_n_head = (int) get_u32_or(g, key("attention.indexer.head_count"), 0);
    w.idx_dim    = (int) get_u32_or(g, key("attention.indexer.key_length"), 0);
    w.idx_top_k  = (int) get_u32_or(g, key("attention.indexer.top_k"), 0);
    w.idx_kpool  = (int) get_u32_or(g, key("attention.indexer.kpool"), 0);
    w.idx_select_tail = get_bool_or(g, key("attention.indexer.kpool_select_tail"), true);

    w.hc_mult  = (int) get_u32_or(g, key("hyper_connection.count"), 0);
    w.hc_iters = (int) get_u32_or(g, key("hyper_connection.sinkhorn_iterations"), 0);
    w.hc_eps   = get_f32_or(g, key("hyper_connection.epsilon"), 1e-6f);

    w.n_dense_lead  = (int) get_u32_or(g, key("leading_dense_block_count"), 0);
    w.n_expert      = (int) get_u32_or(g, key("expert_count"), 0);
    w.n_expert_used = (int) get_u32_or(g, key("expert_used_count"), 0);
    w.n_ff_exp      = (int) get_u32_or(g, key("expert_feed_forward_length"), 0);
    w.n_shared      = (int) get_u32_or(g, key("expert_shared_count"), 0);
    w.expert_scale  = get_f32_or(g, key("expert_weights_scale"), 1.0f);
    w.expert_norm   = get_bool_or(g, key("expert_weights_norm"), true);

    const uint32_t gating = get_u32_or(g, key("expert_gating_func"), 2);
    const uint32_t groups = get_u32_or(g, key("expert_group_count"), 1);

    // Supported configuration.
    std::string bad;
    if (w.n_layer <= 0 || w.n_embd <= 0 || w.n_head <= 0) bad = "missing core dimensions";
    else if (w.hc_mult != 4) bad = "hyper_connection.count must be 4";
    else if (n_rot != 0) bad = "MLA must be NoPE (rope.dimension_count = 0)";
    else if (w.idx_kpool < 2 || w.idx_top_k % w.idx_kpool != 0) bad = "indexer kpool/top_k";
    else if (w.kda_head_dim <= 0 || w.ssm_d_conv < 2) bad = "KDA dimensions";
    else if (gating != 2) bad = "expert_gating_func must be sigmoid (2)";
    else if (groups > 1) bad = "expert groups are not supported";
    else if (w.n_shared != 1) bad = "expected exactly one shared expert";
    else if (!w.idx_select_tail) bad = "indexer kpool_select_tail must be true";
    if (!bad.empty()) {
        set_last_error("glm5: unsupported GGUF: " + bad);
        return false;
    }

    if (!get_arr_u32(g, key("attention.head_count_kv"), w.n_layer, head_kv)) {
        set_last_error("glm5: attention.head_count_kv must list every layer");
        return false;
    }

    std::vector<float> clamp_exp, clamp_shexp;
    if (!get_arr_f32(g, key("swiglu_clamp_exp"), w.n_layer, 0.0f, clamp_exp) ||
        !get_arr_f32(g, key("swiglu_clamp_shexp"), w.n_layer, -1.0f, clamp_shexp)) {
        set_last_error("glm5: malformed swiglu clamp arrays");
        return false;
    }

    // Shared-indexer layers are not implemented (the release has none).
    {
        const int64_t id = gguf_find_key(g, key("attention.indexer.types").c_str());
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(g, id) == GGUF_TYPE_BOOL) {
            const size_t n = gguf_get_arr_n(g, id);
            const int8_t * v = static_cast<const int8_t *>(gguf_get_arr_data(g, id));
            for (size_t i = 0; i < n && (int) i < w.n_layer; ++i) {
                if (head_kv[i] != 0 && !v[i]) {
                    set_last_error("glm5: shared-indexer layers are not supported");
                    return false;
                }
            }
        }
    }

    w.layers.assign(w.n_layer, Glm5Layer{});
    for (int il = 0; il < w.n_layer; ++il) {
        Glm5Layer & L = w.layers[il];
        L.is_mla = head_kv[il] != 0;
        L.is_moe = il >= w.n_dense_lead;
        L.swiglu_clamp_exp = clamp_exp[il];
        L.swiglu_clamp_shexp = clamp_shexp[il] < 0.0f ? clamp_exp[il] : clamp_shexp[il];
    }

    w.stop_tokens.clear();
    for (const char * k : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                           "tokenizer.ggml.eom_token_id"}) {
        const int64_t id = gguf_find_key(g, k);
        if (id < 0) continue;
        const int32_t t = (int32_t) gguf_get_val_u32(g, id);
        if (std::find(w.stop_tokens.begin(), w.stop_tokens.end(), t) == w.stop_tokens.end()) {
            w.stop_tokens.push_back(t);
        }
    }
    return true;
}

// Expected shape of each tensor, given the hyperparameters.
struct TensorSpec {
    std::string    name;
    ggml_tensor ** dst;
    int            device;
    std::vector<int64_t> ne;
};

void add_layer_specs(const Glm5Weights & w, int il, Glm5Layer & L,
                     std::vector<TensorSpec> & specs) {
    const int64_t E = w.n_embd;
    const int64_t hc_mix = (2 + w.hc_mult) * w.hc_mult;
    const int64_t d_inner = w.d_inner();
    const int64_t H = w.n_head;
    const int d = L.device;
    auto add = [&](const char * suffix, ggml_tensor ** dst, std::vector<int64_t> ne) {
        specs.push_back({"blk." + std::to_string(il) + "." + suffix, dst, d, std::move(ne)});
    };

    add("attn_norm.weight", &L.attn_norm, {E});
    add("ffn_norm.weight", &L.ffn_norm, {E});
    add("hc_attn_fn.weight", &L.hc_attn_fn, {w.hc_mult * E, hc_mix});
    add("hc_attn_base.weight", &L.hc_attn_base, {hc_mix});
    add("hc_ffn_fn.weight", &L.hc_ffn_fn, {w.hc_mult * E, hc_mix});
    add("hc_ffn_base.weight", &L.hc_ffn_base, {hc_mix});

    if (!L.is_mla) {
        add("attn_q.weight", &L.wq, {E, d_inner});
        add("attn_k.weight", &L.wk, {E, d_inner});
        add("attn_v.weight", &L.wv, {E, d_inner});
        add("ssm_conv1d_q.weight", &L.conv_q, {w.ssm_d_conv, 1, d_inner});
        add("ssm_conv1d_k.weight", &L.conv_k, {w.ssm_d_conv, 1, d_inner});
        add("ssm_conv1d_v.weight", &L.conv_v, {w.ssm_d_conv, 1, d_inner});
        add("ssm_f_a.weight", &L.ssm_f_a, {E, w.kda_head_dim});
        add("ssm_f_b.weight", &L.ssm_f_b, {w.kda_head_dim, d_inner});
        add("ssm_beta.weight", &L.ssm_beta, {E, H});
        add("ssm_a", &L.ssm_a, {H});
        add("ssm_dt.bias", &L.ssm_dt_b, {d_inner});
        add("ssm_g_a.weight", &L.ssm_g_a, {E, w.kda_head_dim});
        add("ssm_g_b.weight", &L.ssm_g_b, {w.kda_head_dim, d_inner});
        add("ssm_norm.weight", &L.ssm_norm, {w.kda_head_dim});
        add("attn_output.weight", &L.wo, {d_inner, E});
    } else {
        add("attn_q_a.weight", &L.q_a, {E, w.q_lora});
        add("attn_q_a_norm.weight", &L.q_a_norm, {w.q_lora});
        add("attn_q_b.weight", &L.q_b, {w.q_lora, H * w.qk_head});
        add("attn_kv_a_mqa.weight", &L.kv_a, {E, w.kv_lora});
        add("attn_kv_a_norm.weight", &L.kv_a_norm, {w.kv_lora});
        add("attn_k_b.weight", &L.k_b, {w.qk_head, w.kv_lora, H});
        add("attn_v_b.weight", &L.v_b, {w.kv_lora, w.v_head, H});
        add("attn_output.weight", &L.wo, {H * w.v_head, E});
        add("indexer.k_norm.weight", &L.idx_k_norm, {w.idx_dim});
        add("indexer.k_norm.bias", &L.idx_k_norm_b, {w.idx_dim});
        add("indexer.proj.weight", &L.idx_proj, {E, w.idx_n_head});
        add("indexer.attn_k.weight", &L.idx_k, {E, w.idx_dim});
        add("indexer.attn_q_b.weight", &L.idx_q_b, {w.q_lora, w.idx_n_head * w.idx_dim});
        add("indexer_compressor_gate.weight", &L.idx_gate, {E, w.idx_dim});
        add("indexer_compressor_ape.weight", &L.idx_ape, {w.idx_dim, w.idx_kpool});
    }

    if (!L.is_moe) {
        add("ffn_gate.weight", &L.ffn_gate, {E, w.n_ff});
        add("ffn_up.weight", &L.ffn_up, {E, w.n_ff});
        add("ffn_down.weight", &L.ffn_down, {w.n_ff, E});
    } else {
        const int64_t n_sh = (int64_t) w.n_ff_exp * w.n_shared;
        add("ffn_gate_inp.weight", &L.ffn_gate_inp, {E, w.n_expert});
        add("exp_probs_b.bias", &L.ffn_exp_bias, {w.n_expert});
        add("ffn_gate_exps.weight", &L.ffn_gate_exps, {E, w.n_ff_exp, w.n_expert});
        add("ffn_up_exps.weight", &L.ffn_up_exps, {E, w.n_ff_exp, w.n_expert});
        add("ffn_down_exps.weight", &L.ffn_down_exps, {w.n_ff_exp, E, w.n_expert});
        add("ffn_gate_shexp.weight", &L.ffn_gate_shexp, {E, n_sh});
        add("ffn_up_shexp.weight", &L.ffn_up_shexp, {E, n_sh});
        add("ffn_down_shexp.weight", &L.ffn_down_shexp, {n_sh, E});
    }
}

// Find a tensor across the shards.
struct Located {
    Shard *       shard = nullptr;
    ggml_tensor * meta  = nullptr;
    int64_t       index = -1;
};

Located locate(const std::vector<std::unique_ptr<Shard>> & shards, const std::string & name) {
    for (const auto & s : shards) {
        const int64_t id = gguf_find_tensor(s->gguf, name.c_str());
        if (id >= 0) {
            return {s.get(), ggml_get_tensor(s->meta, name.c_str()), id};
        }
    }
    return {};
}

bool shape_matches(const ggml_tensor * t, const std::vector<int64_t> & ne) {
    // Trailing unit dims may be dropped.
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const int64_t want = i < (int) ne.size() ? ne[i] : 1;
        if (t->ne[i] != want) return false;
    }
    return true;
}

std::string shape_str(const int64_t * ne, int n) {
    std::string s = "[";
    for (int i = 0; i < n; ++i) {
        if (i) s += ", ";
        s += std::to_string(ne[i]);
    }
    return s + "]";
}

}  // namespace

int Glm5Weights::n_mla_layers() const {
    int n = 0;
    for (const Glm5Layer & L : layers) n += L.is_mla ? 1 : 0;
    return n;
}

int read_glm5_layer_count(const std::string & path) {
    std::vector<std::unique_ptr<Shard>> shards;
    if (!open_shards(path, shards, /*want_mmap=*/false)) return 0;
    return (int) get_u32_or(shards[0]->gguf, key("block_count"), 0);
}

std::vector<size_t> read_glm5_layer_bytes(const std::string & path) {
    std::vector<std::unique_ptr<Shard>> shards;
    if (!open_shards(path, shards, /*want_mmap=*/false)) return {};
    const int n_layer = (int) get_u32_or(shards[0]->gguf, key("block_count"), 0);
    std::vector<size_t> bytes(std::max(n_layer, 0), 0);
    for (const auto & s : shards) {
        const int64_t n = gguf_get_n_tensors(s->gguf);
        for (int64_t i = 0; i < n; ++i) {
            int il = -1;
            if (std::sscanf(gguf_get_tensor_name(s->gguf, i), "blk.%d.", &il) == 1 &&
                il >= 0 && il < n_layer) {
                bytes[il] += gguf_get_tensor_size(s->gguf, i);
            }
        }
    }
    return bytes;
}

bool load_glm5_model(const std::string & path,
                     const std::vector<ggml_backend_t> & backends,
                     const std::vector<int> & layer_device,
                     Glm5Weights & out) {
    std::vector<std::unique_ptr<Shard>> shards;
    if (!open_shards(path, shards, /*want_mmap=*/true)) return false;

    std::vector<uint32_t> head_kv;
    if (!read_hparams(shards[0]->gguf, out, head_kv)) return false;

    if ((int) layer_device.size() != out.n_layer) {
        set_last_error("glm5: layer placement covers " + std::to_string(layer_device.size()) +
                       " layers, the model has " + std::to_string(out.n_layer));
        return false;
    }
    for (int il = 0; il < out.n_layer; ++il) {
        if (layer_device[il] < 0 || layer_device[il] >= (int) backends.size()) {
            set_last_error("glm5: layer " + std::to_string(il) + " placed on a missing device");
            return false;
        }
        out.layers[il].device = layer_device[il];
    }
    out.embd_device = layer_device.front();
    out.head_device = layer_device.back();

    std::vector<TensorSpec> specs;
    {
        const Located emb = locate(shards, "token_embd.weight");
        if (!emb.meta) {
            set_last_error("glm5: missing token_embd.weight");
            return false;
        }
        out.n_vocab = (int) emb.meta->ne[1];
    }
    specs.push_back({"token_embd.weight", &out.tok_embd, out.embd_device, {out.n_embd, out.n_vocab}});
    specs.push_back({"output_norm.weight", &out.out_norm, out.head_device, {out.n_embd}});
    specs.push_back({"output.weight", &out.output, out.head_device, {out.n_embd, out.n_vocab}});
    for (int il = 0; il < out.n_layer; ++il) {
        add_layer_specs(out, il, out.layers[il], specs);
    }

    // Create the tensors, one context per device.
    const int n_dev = (int) backends.size();
    std::vector<size_t> per_dev(n_dev, 0);
    for (const TensorSpec & s : specs) per_dev[s.device]++;
    out.ctxs.assign(n_dev, nullptr);
    out.bufs.assign(n_dev, nullptr);
    for (int d = 0; d < n_dev; ++d) {
        if (per_dev[d] == 0) continue;
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (per_dev[d] + 8);
        ip.no_alloc = true;
        out.ctxs[d] = ggml_init(ip);
        if (!out.ctxs[d]) {
            set_last_error("glm5: ggml_init failed");
            free_glm5_model(out);
            return false;
        }
    }

    std::vector<Located> sources;
    sources.reserve(specs.size());
    for (const TensorSpec & s : specs) {
        const Located src = locate(shards, s.name);
        if (!src.meta) {
            set_last_error("glm5: missing tensor " + s.name);
            free_glm5_model(out);
            return false;
        }
        if (!shape_matches(src.meta, s.ne)) {
            set_last_error("glm5: tensor " + s.name + " has shape " +
                           shape_str(src.meta->ne, ggml_n_dims(src.meta)) + ", expected " +
                           shape_str(s.ne.data(), (int) s.ne.size()));
            free_glm5_model(out);
            return false;
        }
        ggml_tensor * t = ggml_new_tensor(out.ctxs[s.device], src.meta->type,
                                          ggml_n_dims(src.meta), src.meta->ne);
        ggml_set_name(t, s.name.c_str());
        *s.dst = t;
        sources.push_back(src);
    }

    for (int d = 0; d < n_dev; ++d) {
        if (!out.ctxs[d]) continue;
        out.bufs[d] = ggml_backend_alloc_ctx_tensors(out.ctxs[d], backends[d]);
        if (!out.bufs[d]) {
            set_last_error("glm5: out of memory allocating the weights on device " +
                           std::to_string(d) + " (" + ggml_backend_name(backends[d]) + ")");
            free_glm5_model(out);
            return false;
        }
        ggml_backend_buffer_set_usage(out.bufs[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // Copy the data out of the mapped files.
    for (size_t i = 0; i < specs.size(); ++i) {
        const Located & src = sources[i];
        ggml_tensor * dst = *specs[i].dst;
        const size_t data_off = gguf_get_data_offset(src.shard->gguf);
        const size_t off = gguf_get_tensor_offset(src.shard->gguf, src.index);
        const size_t size = gguf_get_tensor_size(src.shard->gguf, src.index);
        const size_t file_size = src.shard->map.size();
        if (size != ggml_nbytes(dst) || data_off > file_size || off > file_size - data_off ||
            size > file_size - data_off - off) {
            set_last_error("glm5: tensor " + specs[i].name + " lies outside " + src.shard->path +
                           " (truncated download?)");
            free_glm5_model(out);
            return false;
        }
        const uint8_t * base = static_cast<const uint8_t *>(src.shard->map.data());
        ggml_backend_tensor_set(dst, base + data_off + off, 0, size);
    }

    // mHC scales: three host scalars per sub-block.
    for (Glm5Layer & L : out.layers) {
        for (int k = 0; k < 2; ++k) {
            const std::string name = std::string("blk.") +
                std::to_string(&L - out.layers.data()) + (k == 0 ? ".hc_attn_scale.weight"
                                                                   : ".hc_ffn_scale.weight");
            const Located src = locate(shards, name);
            if (!src.meta || src.meta->type != GGML_TYPE_F32 || ggml_nelements(src.meta) != 3) {
                set_last_error("glm5: " + name + " must be F32 [3]");
                free_glm5_model(out);
                return false;
            }
            const uint8_t * base = static_cast<const uint8_t *>(src.shard->map.data());
            const size_t file_size = src.shard->map.size();
            const size_t data_off = gguf_get_data_offset(src.shard->gguf);
            const size_t tensor_off = gguf_get_tensor_offset(src.shard->gguf, src.index);
            if (data_off > file_size || tensor_off > file_size - data_off ||
                3 * sizeof(float) > file_size - data_off - tensor_off) {
                set_last_error("glm5: " + name + " lies outside the file");
                free_glm5_model(out);
                return false;
            }
            const size_t pos = data_off + tensor_off;
            std::memcpy(k == 0 ? L.hc_attn_scale : L.hc_ffn_scale, base + pos, 3 * sizeof(float));
        }
    }
    return true;
}

void free_glm5_model(Glm5Weights & w) {
    for (ggml_backend_buffer_t b : w.bufs) {
        if (b) ggml_backend_buffer_free(b);
    }
    for (ggml_context * c : w.ctxs) {
        if (c) ggml_free(c);
    }
    w.bufs.clear();
    w.ctxs.clear();
    w.layers.clear();
    w.tok_embd = w.out_norm = w.output = nullptr;
}

// ── Cache ────────────────────────────────────────────────────────────────

bool create_glm5_cache(const Glm5Weights & w,
                       const std::vector<ggml_backend_t> & backends,
                       int max_ctx, Glm5Cache & out) {
    free_glm5_cache(out);
    // FA reads the latent in 256-row tiles.
    max_ctx = ((std::max(max_ctx, 1) + 255) / 256) * 256;
    out.max_ctx = max_ctx;
    out.layers.assign(w.n_layer, Glm5LayerCache{});

    const int n_dev = (int) backends.size();
    out.ctxs.assign(n_dev, nullptr);
    out.bufs.assign(n_dev, nullptr);
    for (int d = 0; d < n_dev; ++d) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (size_t) (w.n_layer * 4 + 8);
        ip.no_alloc = true;
        out.ctxs[d] = ggml_init(ip);
        if (!out.ctxs[d]) return false;
    }

    std::vector<bool> used(n_dev, false);
    for (int il = 0; il < w.n_layer; ++il) {
        const Glm5Layer & L = w.layers[il];
        Glm5LayerCache & C = out.layers[il];
        ggml_context * ctx = out.ctxs[L.device];
        used[L.device] = true;
        char name[64];
        if (!L.is_mla) {
            C.conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ssm_d_conv - 1, 3 * w.d_inner());
            C.ssm  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.kda_head_dim, w.kda_head_dim, w.n_head);
            std::snprintf(name, sizeof(name), "glm5_conv_%d", il); ggml_set_name(C.conv, name);
            std::snprintf(name, sizeof(name), "glm5_ssm_%d", il);  ggml_set_name(C.ssm, name);
        } else {
            C.latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, w.kv_lora, max_ctx);
            C.idx_k  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.idx_dim, max_ctx);
            C.idx_g  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.idx_dim, max_ctx);
            C.pooled = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.idx_dim, max_ctx / w.idx_kpool);
            std::snprintf(name, sizeof(name), "glm5_latent_%d", il); ggml_set_name(C.latent, name);
            std::snprintf(name, sizeof(name), "glm5_idx_k_%d", il);  ggml_set_name(C.idx_k, name);
            std::snprintf(name, sizeof(name), "glm5_idx_g_%d", il);  ggml_set_name(C.idx_g, name);
            std::snprintf(name, sizeof(name), "glm5_pooled_%d", il); ggml_set_name(C.pooled, name);
        }
    }

    for (int d = 0; d < n_dev; ++d) {
        if (!used[d]) continue;
        out.bufs[d] = ggml_backend_alloc_ctx_tensors(out.ctxs[d], backends[d]);
        if (!out.bufs[d]) {
            set_last_error("glm5: out of memory allocating the KV/state cache (max_ctx=" +
                           std::to_string(max_ctx) + ") on device " + std::to_string(d));
            free_glm5_cache(out);
            return false;
        }
    }
    clear_glm5_cache(out);
    return true;
}

void clear_glm5_cache(Glm5Cache & c) {
    for (ggml_backend_buffer_t b : c.bufs) {
        if (b) ggml_backend_buffer_clear(b, 0);
    }
    c.cur_pos = 0;
}

void free_glm5_cache(Glm5Cache & c) {
    for (ggml_backend_buffer_t b : c.bufs) {
        if (b) ggml_backend_buffer_free(b);
    }
    for (ggml_context * x : c.ctxs) {
        if (x) ggml_free(x);
    }
    c.bufs.clear();
    c.ctxs.clear();
    c.layers.clear();
    c.cur_pos = 0;
    c.max_ctx = 0;
}

}  // namespace luce::common
