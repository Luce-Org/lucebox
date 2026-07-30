#include "kimi_k3_internal.h"

#include "common/gguf_bounds.h"
#include "common/gguf_mmap.h"
#include "internal.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace dflash::common {
namespace {

uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return fallback;
        const gguf_type type = gguf_get_arr_type(g, id);
        const void * data = gguf_get_arr_data(g, id);
        if (type == GGUF_TYPE_UINT32) return static_cast<const uint32_t *>(data)[0];
        if (type == GGUF_TYPE_INT32)  return static_cast<uint32_t>(static_cast<const int32_t *>(data)[0]);
        return fallback;
    }
    const gguf_type type = gguf_get_kv_type(g, id);
    if (type == GGUF_TYPE_UINT32) return gguf_get_val_u32(g, id);
    if (type == GGUF_TYPE_INT32)  return static_cast<uint32_t>(gguf_get_val_i32(g, id));
    return fallback;
}

float get_f32_or(const gguf_context * g, const char * key, float fallback) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(g, id);
    return fallback;
}

bool get_bool_or(const gguf_context * g, const char * key, bool fallback) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) return fallback;
    return gguf_get_val_bool(g, id);
}

std::vector<uint32_t> get_u32_array(const gguf_context * g, const char * key) {
    std::vector<uint32_t> out;
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(g, id);
    const void * data = gguf_get_arr_data(g, id);
    if (gguf_get_arr_type(g, id) == GGUF_TYPE_UINT32) {
        const auto * p = static_cast<const uint32_t *>(data);
        out.assign(p, p + n);
    } else if (gguf_get_arr_type(g, id) == GGUF_TYPE_INT32) {
        const auto * p = static_cast<const int32_t *>(data);
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.push_back(static_cast<uint32_t>(p[i]));
    }
    return out;
}

bool tensor_shape_is(const ggml_tensor * t,
                     int64_t ne0,
                     int64_t ne1 = 1,
                     int64_t ne2 = 1) {
    return t && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2;
}

} // namespace

bool load_kimi_k3_gguf(const std::string & path,
                       ggml_backend_t backend,
                       KimiK3Weights & out) {
    free_kimi_k3_weights(out);

    ggml_context * meta_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx || !meta_ctx) {
        set_last_error("Kimi-K3: failed to parse GGUF: " + path);
        if (gctx) gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    auto fail = [&](const std::string & message) {
        set_last_error("Kimi-K3: " + message);
        if (out.buf) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
        }
        gguf_free(gctx);
        ggml_free(meta_ctx);
        out = KimiK3Weights{};
        return false;
    };

    const int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id < 0 || std::strcmp(gguf_get_val_str(gctx, arch_id), "kimi-k3") != 0) {
        return fail("general.architecture must be kimi-k3");
    }

    constexpr const char * A = "kimi-k3.";
    auto key = [&](const char * suffix) { return std::string(A) + suffix; };
    auto u32 = [&](const char * suffix, uint32_t fallback = 0) {
        const std::string k = key(suffix);
        return get_u32_or(gctx, k.c_str(), fallback);
    };
    auto f32 = [&](const char * suffix, float fallback) {
        const std::string k = key(suffix);
        return get_f32_or(gctx, k.c_str(), fallback);
    };
    auto boolean = [&](const char * suffix, bool fallback) {
        const std::string k = key(suffix);
        return get_bool_or(gctx, k.c_str(), fallback);
    };

    out.ctx       = meta_ctx;
    out.backend   = backend;
    out.n_layer   = static_cast<int>(u32("block_count"));
    out.n_embd    = static_cast<int>(u32("embedding_length"));
    out.n_ff      = static_cast<int>(u32("feed_forward_length"));
    out.n_vocab   = static_cast<int>(u32("vocab_size"));
    out.n_ctx_train = static_cast<int>(u32("context_length"));
    out.n_head    = static_cast<int>(u32("attention.head_count"));
    out.n_expert  = static_cast<int>(u32("expert_count"));
    out.n_expert_used = static_cast<int>(u32("expert_used_count"));
    out.n_ff_exp  = static_cast<int>(u32("expert_feed_forward_length"));
    out.n_expert_latent = static_cast<int>(u32("expert_latent_length"));
    out.n_expert_shared = static_cast<int>(u32("expert_shared_count", 1));
    out.n_dense_lead = static_cast<int>(u32("leading_dense_block_count", 0));
    out.ssm_d_conv = static_cast<int>(u32("ssm.conv_kernel"));
    out.kda_head_dim = static_cast<int>(u32("kda.head_dim"));
    out.q_lora_rank = static_cast<int>(u32("attention.q_lora_rank"));
    out.kv_lora_rank = static_cast<int>(u32("attention.kv_lora_rank"));
    out.mla_k_head_dim = static_cast<int>(u32("attention.key_length_mla"));
    out.mla_v_head_dim = static_cast<int>(u32("attention.value_length_mla"));
    out.rope_dim = static_cast<int>(u32("rope.dimension_count"));
    out.attn_res_block_size = static_cast<int>(u32("attn_res.block_size"));
    out.rms_eps = f32("attention.layer_norm_rms_epsilon", 1.0e-5f);
    out.kda_gate_lower_bound = f32("kda.gate_lower_bound", -INFINITY);
    out.expert_weights_scale = f32("expert_weights_scale", 1.0f);
    out.expert_weights_norm = boolean("expert_weights_norm", true);
    out.expert_gating_func = static_cast<int>(u32("expert_gating_func", 2));
    out.situ_beta = f32("activation.situ_beta", 4.0f);
    out.situ_linear_beta = f32("activation.situ_linear_beta", 25.0f);
    out.eos_token_id = static_cast<int32_t>(get_u32_or(gctx, "tokenizer.ggml.eos_token_id", 2));

    auto get = [&](const char * name) { return ggml_get_tensor(meta_ctx, name); };
    out.tok_embd = get("token_embd.weight");
    out.output_norm = get("output_norm.weight");
    out.output = get("output.weight");
    out.output_res_score = get("output_res_score.weight");
    if (out.n_vocab == 0 && out.tok_embd) out.n_vocab = static_cast<int>(out.tok_embd->ne[1]);

    constexpr int MAX_LAYERS = 1024;
    constexpr int MAX_HEADS = 1024;
    constexpr int MAX_EXPERTS = 4096;
    if (out.n_layer <= 0 || out.n_layer > MAX_LAYERS ||
        out.n_embd <= 0 || out.n_head <= 0 || out.n_head > MAX_HEADS ||
        out.n_vocab <= 0 || out.n_expert <= 0 || out.n_expert > MAX_EXPERTS ||
        out.n_expert_used <= 0 || out.n_expert_used > out.n_expert ||
        out.n_expert_latent <= 0 || out.n_ff_exp <= 0 ||
        out.ssm_d_conv < 2 || out.kda_head_dim <= 0 ||
        out.attn_res_block_size <= 0 || out.kv_lora_rank <= 0 ||
        out.mla_k_head_dim <= out.rope_dim || out.mla_v_head_dim <= 0) {
        return fail("invalid or incomplete architecture metadata");
    }
    if (!tensor_shape_is(out.tok_embd, out.n_embd, out.n_vocab) ||
        !tensor_shape_is(out.output, out.n_embd, out.n_vocab) ||
        !tensor_shape_is(out.output_norm, out.n_embd) ||
        !tensor_shape_is(out.output_res_score, out.n_embd)) {
        return fail("missing or malformed top-level tensors");
    }

    std::vector<uint32_t> head_kv = get_u32_array(gctx, "kimi-k3.attention.head_count_kv");
    if (head_kv.empty()) {
        head_kv.assign(static_cast<size_t>(out.n_layer),
                       get_u32_or(gctx, "kimi-k3.attention.head_count_kv", 0));
    }
    if (head_kv.size() != static_cast<size_t>(out.n_layer)) {
        return fail("attention.head_count_kv must have one value per layer");
    }

    out.layers.assign(static_cast<size_t>(out.n_layer), KimiK3Layer{});
    for (int il = 0; il < out.n_layer; ++il) {
        char name[160];
        auto find = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return get(name);
        };
        KimiK3Layer & L = out.layers[static_cast<size_t>(il)];
        L.recurrent = head_kv[static_cast<size_t>(il)] == 0;
        L.attn_norm = find("attn_norm.weight");
        L.ffn_norm = find("ffn_norm.weight");
        L.attn_res_score = find("attn_res_score.weight");
        L.ffn_res_score = find("ffn_res_score.weight");
        L.wo = find("attn_output.weight");
        if (!L.attn_norm || !L.ffn_norm || !L.attn_res_score ||
            !L.ffn_res_score || !L.wo) {
            return fail("layer " + std::to_string(il) + " is missing common tensors");
        }

        if (L.recurrent) {
            L.wq = find("attn_q.weight");
            L.wk = find("attn_k.weight");
            L.wv = find("attn_v.weight");
            L.ssm_q_conv = find("ssm_conv1d_q.weight");
            L.ssm_k_conv = find("ssm_conv1d_k.weight");
            L.ssm_v_conv = find("ssm_conv1d_v.weight");
            L.ssm_f_a = find("ssm_f_a.weight");
            L.ssm_f_b = find("ssm_f_b.weight");
            L.ssm_beta = find("ssm_beta.weight");
            L.ssm_a = find("ssm_a");
            L.ssm_dt_b = find("ssm_dt.bias");
            L.ssm_g = find("ssm_g.weight");
            L.ssm_o_norm = find("ssm_norm.weight");
            if (!L.wq || !L.wk || !L.wv || !L.ssm_q_conv || !L.ssm_k_conv ||
                !L.ssm_v_conv || !L.ssm_f_a || !L.ssm_f_b || !L.ssm_beta ||
                !L.ssm_a || !L.ssm_dt_b || !L.ssm_g || !L.ssm_o_norm) {
                return fail("KDA layer " + std::to_string(il) + " is incomplete");
            }
        } else {
            L.wq_a = find("attn_q_a.weight");
            L.wq_a_norm = find("attn_q_a_norm.weight");
            L.wq_b = find("attn_q_b.weight");
            L.wq = find("attn_q.weight");
            L.wkv_a_mqa = find("attn_kv_a_mqa.weight");
            L.wkv_a_norm = find("attn_kv_a_norm.weight");
            L.wk_b = find("attn_k_b.weight");
            L.wv_b = find("attn_v_b.weight");
            L.wkv_b = find("attn_kv_b.weight");
            L.wqkv_gate = find("attn_gate.weight");
            const bool q_ok = L.wq || (L.wq_a && L.wq_a_norm && L.wq_b);
            if (!q_ok || !L.wkv_a_mqa || !L.wkv_a_norm || !L.wqkv_gate ||
                ((!L.wk_b || !L.wv_b) && !L.wkv_b)) {
                return fail("MLA layer " + std::to_string(il) + " is incomplete");
            }
            // The first native path intentionally requires absorbed MLA. It is
            // the official K3 layout and stores one compact K-only cache.
            if (!L.wk_b || !L.wv_b) {
                return fail("MLA layer " + std::to_string(il) +
                            " uses unabsorbed attn_kv_b; not supported by the native cache yet");
            }
        }

        if (il < out.n_dense_lead) {
            L.ffn_gate = find("ffn_gate.weight");
            L.ffn_up = find("ffn_up.weight");
            L.ffn_down = find("ffn_down.weight");
            if (!L.ffn_gate || !L.ffn_up || !L.ffn_down) {
                return fail("dense FFN layer " + std::to_string(il) + " is incomplete");
            }
        } else {
            L.ffn_gate_inp = find("ffn_gate_inp.weight");
            L.ffn_exp_probs_b = find("exp_probs_b.bias");
            L.ffn_gate_exps = find("ffn_gate_exps.weight");
            L.ffn_up_exps = find("ffn_up_exps.weight");
            L.ffn_down_exps = find("ffn_down_exps.weight");
            L.ffn_routed_down = find("ffn_routed_down.weight");
            L.ffn_routed_up = find("ffn_routed_up.weight");
            L.ffn_routed_norm = find("ffn_routed_norm.weight");
            L.ffn_gate_shexp = find("ffn_gate_shexp.weight");
            L.ffn_up_shexp = find("ffn_up_shexp.weight");
            L.ffn_down_shexp = find("ffn_down_shexp.weight");
            if (!L.ffn_gate_inp || !L.ffn_exp_probs_b || !L.ffn_gate_exps ||
                !L.ffn_up_exps || !L.ffn_down_exps || !L.ffn_routed_down ||
                !L.ffn_routed_up || !L.ffn_gate_shexp || !L.ffn_up_shexp ||
                !L.ffn_down_shexp) {
                return fail("latent MoE layer " + std::to_string(il) + " is incomplete");
            }
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) return fail("unable to allocate resident tensor buffer");

    GgufMmap mmap;
    std::string mmap_error;
    if (!mmap.open(path, mmap_error)) return fail(mmap_error);
    const auto * base = static_cast<const uint8_t *>(mmap.data());
    const size_t file_size = mmap.size();
    const size_t data_start = gguf_get_data_offset(gctx);
    size_t copied = 0;
    for (int64_t tid = 0; tid < gguf_get_n_tensors(gctx); ++tid) {
        const char * tensor_name = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * tensor = ggml_get_tensor(meta_ctx, tensor_name);
        if (!tensor) continue;
        const size_t offset = gguf_get_tensor_offset(gctx, tid);
        const size_t bytes = gguf_get_tensor_size(gctx, tid);
        if (!gguf_tensor_in_file(data_start, offset, bytes, file_size)) {
            return fail(gguf_bounds_error("Kimi-K3 GGUF", tensor_name,
                ggml_type_name(gguf_get_tensor_type(gctx, tid)), data_start,
                offset, bytes, file_size));
        }
        ggml_backend_tensor_set(tensor, base + data_start + offset, 0, bytes);
        copied += bytes;
    }

    gguf_free(gctx);
    std::fprintf(stderr,
        "[kimi-k3] loaded %.2f GiB: layers=%d (KDA=%zu MLA=%zu) hidden=%d "
        "experts=%d top=%d latent=%d vocab=%d\n",
        static_cast<double>(copied) / (1024.0 * 1024.0 * 1024.0),
        out.n_layer,
        static_cast<size_t>(std::count_if(out.layers.begin(), out.layers.end(),
                                          [](const KimiK3Layer & l) { return l.recurrent; })),
        static_cast<size_t>(std::count_if(out.layers.begin(), out.layers.end(),
                                          [](const KimiK3Layer & l) { return !l.recurrent; })),
        out.n_embd, out.n_expert, out.n_expert_used,
        out.n_expert_latent, out.n_vocab);
    std::fflush(stderr);
    return true;
}

void free_kimi_k3_weights(KimiK3Weights & w) {
    if (w.buf) ggml_backend_buffer_free(w.buf);
    if (w.ctx) ggml_free(w.ctx);
    w = KimiK3Weights{};
}

} // namespace dflash::common
