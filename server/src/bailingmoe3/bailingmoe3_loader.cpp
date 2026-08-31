// Native GGUF loader for BailingMoE3 (Ling 3.x).
//
// Ling GGUFs contain one embedded NextN/MTP block after the 42-layer
// autoregressive trunk. This loader deliberately loads only the trunk; MTP is
// a decode optimization and is not required for a correctness-first backend.

#include "internal.h"

#include "common/gguf_bounds.h"
#include "common/gguf_mmap.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace dflash::common {
namespace {

constexpr const char * kArch = "bailingmoe3";

uint32_t get_u32_or(const gguf_context * g, const std::string & key,
                    uint32_t fallback) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return fallback;
        const gguf_type type = gguf_get_arr_type(g, id);
        const void * data = gguf_get_arr_data(g, id);
        if (type == GGUF_TYPE_UINT32) return static_cast<const uint32_t *>(data)[0];
        if (type == GGUF_TYPE_INT32) {
            const int32_t value = static_cast<const int32_t *>(data)[0];
            return value < 0 ? fallback : static_cast<uint32_t>(value);
        }
        return fallback;
    }
    return gguf_get_val_u32(g, id);
}

float get_f32_or(const gguf_context * g, const std::string & key,
                 float fallback) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0 ||
            gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32) {
            return fallback;
        }
        return static_cast<const float *>(gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_f32(g, id);
}

bool get_bool_or(const gguf_context * g, const std::string & key,
                 bool fallback) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) return fallback;
    return gguf_get_val_bool(g, id);
}

std::vector<uint32_t> get_u32_array(const gguf_context * g,
                                    const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return {};
    const gguf_type type = gguf_get_arr_type(g, id);
    if (type != GGUF_TYPE_UINT32 && type != GGUF_TYPE_INT32) return {};
    const size_t n = gguf_get_arr_n(g, id);
    const void * raw = gguf_get_arr_data(g, id);
    std::vector<uint32_t> result(n);
    for (size_t i = 0; i < n; ++i) {
        if (type == GGUF_TYPE_UINT32) {
            result[i] = static_cast<const uint32_t *>(raw)[i];
        } else {
            const int32_t value = static_cast<const int32_t *>(raw)[i];
            if (value < 0) return {};
            result[i] = static_cast<uint32_t>(value);
        }
    }
    return result;
}

std::vector<float> get_f32_array(const gguf_context * g,
                                 const std::string & key,
                                 size_t count) {
    std::vector<float> result(count, 0.0f);
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return result;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32) {
        std::fill(result.begin(), result.end(), gguf_get_val_f32(g, id));
        return result;
    }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32) {
        return result;
    }
    const size_t n = std::min(count, gguf_get_arr_n(g, id));
    const float * values = static_cast<const float *>(gguf_get_arr_data(g, id));
    std::copy(values, values + n, result.begin());
    return result;
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

struct TensorAllocation {
    ggml_tensor * tensor = nullptr;
    size_t file_offset = 0;
    size_t file_size = 0;
    size_t buffer_offset = 0;
};

}  // namespace

bool load_bailingmoe3_gguf(const std::string & path,
                           ggml_backend_t backend,
                           TargetWeights & out) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) {
        set_last_error("bailingmoe3: gguf_init_from_file failed: " + path);
        return false;
    }

    auto fail = [&](const std::string & message) {
        set_last_error("bailingmoe3: " + message);
        gguf_free(gctx);
        if (meta_ctx) {
            ggml_free(meta_ctx);
            if (out.ctx == meta_ctx) out.ctx = nullptr;
            meta_ctx = nullptr;
        }
        return false;
    };

    const int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    const char * arch = arch_id >= 0 ? gguf_get_val_str(gctx, arch_id) : nullptr;
    if (!arch || std::strcmp(arch, kArch) != 0) {
        return fail(std::string("unexpected architecture '") +
                    (arch ? arch : "") + "'");
    }

    const std::string prefix = std::string(kArch) + ".";
    const uint32_t block_count = get_u32_or(gctx, prefix + "block_count", 0);
    const uint32_t nextn = get_u32_or(gctx, prefix + "nextn_predict_layers", 0);
    if (block_count == 0 || nextn >= block_count) {
        return fail("invalid block_count/nextn_predict_layers");
    }
    const uint32_t n_layer = block_count - nextn;
    const uint32_t n_embd = get_u32_or(gctx, prefix + "embedding_length", 0);
    const uint32_t n_head = get_u32_or(gctx, prefix + "attention.head_count", 0);
    const uint32_t kda_head_dim = get_u32_or(gctx, prefix + "kda.head_dim", 0);
    const uint32_t ssm_conv = get_u32_or(gctx, prefix + "ssm.conv_kernel", 0);
    const uint32_t rope_dim = get_u32_or(gctx, prefix + "rope.dimension_count", 0);
    const uint32_t qk_mla = get_u32_or(gctx, prefix + "attention.key_length_mla", 0);
    const uint32_t v_mla = get_u32_or(gctx, prefix + "attention.value_length_mla", 0);
    const uint32_t kv_lora = get_u32_or(gctx, prefix + "attention.kv_lora_rank", 0);
    const uint32_t q_lora = get_u32_or(gctx, prefix + "attention.q_lora_rank", 0);
    const uint32_t n_ff = get_u32_or(gctx, prefix + "feed_forward_length", 0);
    const uint32_t n_ff_exp = get_u32_or(gctx, prefix + "expert_feed_forward_length", 0);
    const uint32_t n_ff_shexp = get_u32_or(
        gctx, prefix + "expert_shared_feed_forward_length", 0);
    const uint32_t n_expert = get_u32_or(gctx, prefix + "expert_count", 0);
    const uint32_t n_expert_used = get_u32_or(gctx, prefix + "expert_used_count", 0);
    const uint32_t n_expert_groups = get_u32_or(
        gctx, prefix + "expert_group_count", 1);
    const uint32_t n_expert_groups_used = get_u32_or(
        gctx, prefix + "expert_group_used_count", 1);
    const uint32_t dense_lead = get_u32_or(gctx, prefix + "leading_dense_block_count", 0);
    const float kda_lower = get_f32_or(gctx, prefix + "kda.gate_lower_bound", 0.0f);

    if (n_layer == 0 || n_embd == 0 || n_head == 0 || kda_head_dim == 0 ||
        ssm_conv < 2 || rope_dim == 0 || qk_mla <= rope_dim || v_mla == 0 ||
        kv_lora == 0 || q_lora != 0 || n_ff == 0 || n_ff_exp == 0 ||
        n_ff_shexp == 0 || n_expert == 0 || n_expert_used == 0 ||
        n_expert_used > n_expert || n_expert_groups == 0 ||
        n_expert_groups_used == 0 ||
        n_expert_groups_used > n_expert_groups ||
        n_expert % n_expert_groups != 0 || dense_lead > n_layer ||
        kda_lower >= 0.0f) {
        char message[640];
        std::snprintf(message, sizeof(message),
            "invalid hparams: layers=%u(+%u MTP) embd=%u heads=%u "
            "kda{head=%u conv=%u lower=%g} mla{qk=%u v=%u rope=%u kv_lora=%u q_lora=%u} "
            "ff{dense=%u exp=%u shared=%u dense_lead=%u} "
            "experts=%u used=%u groups=%u/%u",
            n_layer, nextn, n_embd, n_head, kda_head_dim, ssm_conv,
            static_cast<double>(kda_lower), qk_mla, v_mla, rope_dim,
            kv_lora, q_lora, n_ff, n_ff_exp, n_ff_shexp, dense_lead,
            n_expert, n_expert_used, n_expert_groups_used, n_expert_groups);
        return fail(message);
    }

    // The per-layer KV-head array is the GGUF's authoritative recurrent/MLA
    // pattern: zero means KDA, non-zero means MLA. LuceBox's shared hybrid
    // cache uses a fixed interval, so verify that Ling's pattern is regular.
    const std::vector<uint32_t> kv_heads =
        get_u32_array(gctx, prefix + "attention.head_count_kv");
    if (kv_heads.size() < n_layer) {
        return fail("missing or short attention.head_count_kv array");
    }
    int full_attention_interval = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (kv_heads[il] != 0) {
            const int interval = static_cast<int>(il) + 1;
            if (full_attention_interval == 0) full_attention_interval = interval;
            if (interval % full_attention_interval != 0) {
                return fail("irregular KDA/MLA layer pattern is not supported");
            }
        }
    }
    if (full_attention_interval == 0 || n_layer % full_attention_interval != 0) {
        return fail("no regular MLA layers found");
    }
    for (uint32_t il = 0; il < n_layer; ++il) {
        const bool expected_mla = ((il + 1) % full_attention_interval) == 0;
        if ((kv_heads[il] != 0) != expected_mla) {
            return fail("attention.head_count_kv does not match the inferred interval");
        }
    }

    out.ctx = meta_ctx;
    out.backend = backend;
    out.n_layer = static_cast<int>(n_layer);
    out.n_embd = static_cast<int>(n_embd);
    out.n_head = static_cast<int>(n_head);
    out.n_head_kv = 1;  // latent MLA is MQA
    out.n_ff = static_cast<int>(n_ff);
    out.n_ff_exp = static_cast<int>(n_ff_exp);
    out.n_ff_shexp = static_cast<int>(n_ff_shexp);
    out.n_expert = static_cast<int>(n_expert);
    out.n_expert_used = static_cast<int>(n_expert_used);
    out.n_expert_groups = static_cast<int>(n_expert_groups);
    out.n_expert_groups_used = static_cast<int>(n_expert_groups_used);
    out.n_layer_dense_lead = static_cast<int>(dense_lead);
    out.full_attention_interval = full_attention_interval;
    out.rope_dimension_count = static_cast<int>(rope_dim);
    out.rope_theta = get_f32_or(gctx, prefix + "rope.freq_base", 1000000.0f);
    out.rms_eps = get_f32_or(
        gctx, prefix + "attention.layer_norm_rms_epsilon", 1.0e-6f);
    out.kda_head_dim = static_cast<int>(kda_head_dim);
    out.mla_qk_head_dim = static_cast<int>(qk_mla);
    out.mla_v_head_dim = static_cast<int>(v_mla);
    out.kv_lora_rank = static_cast<int>(kv_lora);
    out.q_lora_rank = static_cast<int>(q_lora);
    out.kda_gate_lower_bound = kda_lower;
    out.n_embd_head_k = static_cast<int>(kv_lora + rope_dim);
    out.n_embd_head_v = static_cast<int>(kv_lora);
    out.ssm_d_conv = static_cast<int>(ssm_conv);
    out.ssm_d_inner = static_cast<int>(n_head * kda_head_dim);
    out.ssm_d_state = static_cast<int>(kda_head_dim);
    out.ssm_dt_rank = static_cast<int>(n_head);
    out.ssm_n_group = static_cast<int>(n_head);
    out.expert_gating_func = static_cast<int>(
        get_u32_or(gctx, prefix + "expert_gating_func", 2));
    out.expert_weights_scale = get_f32_or(
        gctx, prefix + "expert_weights_scale", 1.0f);
    out.expert_weights_norm = get_bool_or(
        gctx, prefix + "expert_weights_norm", true);
    out.is_moe = true;
    out.is_bailingmoe3 = true;

    const uint32_t missing_token = 0xFFFFFFFFu;
    const uint32_t eos = get_u32_or(gctx, "tokenizer.ggml.eos_token_id", missing_token);
    const uint32_t eot = get_u32_or(gctx, "tokenizer.ggml.eot_token_id", missing_token);
    out.eos_id = eos == missing_token ? -1 : static_cast<int32_t>(eos);
    out.eos_chat_id = eot == missing_token ? -1 : static_cast<int32_t>(eot);

    out.layers.assign(n_layer, TargetLayer{});
    const std::vector<float> clamp_exp =
        get_f32_array(gctx, prefix + "swiglu_clamp_exp", n_layer);
    const std::vector<float> clamp_shexp =
        get_f32_array(gctx, prefix + "swiglu_clamp_shexp", n_layer);

    auto tensor = [&](const char * name) { return ggml_get_tensor(meta_ctx, name); };
    auto layer_tensor = [&](uint32_t il, const char * suffix) {
        char name[160];
        std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
        return ggml_get_tensor(meta_ctx, name);
    };

    out.tok_embd = tensor("token_embd.weight");
    out.out_norm = tensor("output_norm.weight");
    out.output = tensor("output.weight");
    if (!out.tok_embd || !out.out_norm || !out.output) {
        return fail("missing token_embd/output_norm/output tensor");
    }
    out.n_vocab = static_cast<int>(out.tok_embd->ne[1]);

    for (uint32_t il = 0; il < n_layer; ++il) {
        TargetLayer & layer = out.layers[il];
        layer.attn_norm = layer_tensor(il, "attn_norm.weight");
        layer.ffn_norm = layer_tensor(il, "ffn_norm.weight");
        layer.attn_post_norm = layer.ffn_norm;
        layer.ffn_swiglu_clamp_exp = clamp_exp[il];
        layer.ffn_swiglu_clamp_shexp = clamp_shexp[il];
        if (!layer.attn_norm || !layer.ffn_norm) {
            return fail("layer " + std::to_string(il) + " missing norm tensor");
        }

        const bool mla = ((il + 1) % full_attention_interval) == 0;
        layer.wo = layer_tensor(il, "attn_output.weight");
        if (mla) {
            layer.wq = layer_tensor(il, "attn_q.weight");
            layer.attn_q_a = layer_tensor(il, "attn_q_a.weight");
            layer.attn_q_a_norm = layer_tensor(il, "attn_q_a_norm.weight");
            layer.attn_q_b = layer_tensor(il, "attn_q_b.weight");
            layer.attn_kv_a_mqa = layer_tensor(il, "attn_kv_a_mqa.weight");
            layer.attn_kv_a_norm = layer_tensor(il, "attn_kv_a_norm.weight");
            layer.attn_k_b = layer_tensor(il, "attn_k_b.weight");
            layer.attn_v_b = layer_tensor(il, "attn_v_b.weight");
            layer.wqkv_gate = layer_tensor(il, "attn_gate.weight");
            const bool has_q = layer.wq ||
                (layer.attn_q_a && layer.attn_q_a_norm && layer.attn_q_b);
            if (!has_q || !layer.attn_kv_a_mqa || !layer.attn_kv_a_norm ||
                !layer.attn_k_b || !layer.attn_v_b || !layer.wqkv_gate || !layer.wo) {
                return fail("layer " + std::to_string(il) + " missing MLA tensor");
            }
        } else {
            layer.wq = layer_tensor(il, "attn_q.weight");
            layer.wk = layer_tensor(il, "attn_k.weight");
            layer.wv = layer_tensor(il, "attn_v.weight");
            layer.ssm_conv1d_q = layer_tensor(il, "ssm_conv1d_q.weight");
            layer.ssm_conv1d_k = layer_tensor(il, "ssm_conv1d_k.weight");
            layer.ssm_conv1d_v = layer_tensor(il, "ssm_conv1d_v.weight");
            layer.ssm_f_a = layer_tensor(il, "ssm_f_a.weight");
            layer.ssm_beta = layer_tensor(il, "ssm_beta.weight");
            layer.ssm_a = layer_tensor(il, "ssm_a");
            layer.ssm_dt_bias = layer_tensor(il, "ssm_dt.bias");
            layer.ssm_g_a = layer_tensor(il, "ssm_g_a.weight");
            layer.ssm_norm = layer_tensor(il, "ssm_norm.weight");
            if (!layer.wq || !layer.wk || !layer.wv || !layer.wo ||
                !layer.ssm_conv1d_q || !layer.ssm_conv1d_k ||
                !layer.ssm_conv1d_v || !layer.ssm_f_a || !layer.ssm_beta ||
                !layer.ssm_a || !layer.ssm_dt_bias || !layer.ssm_g_a ||
                !layer.ssm_norm) {
                return fail("layer " + std::to_string(il) + " missing KDA tensor");
            }
        }

        if (il < dense_lead) {
            layer.w_gate = layer_tensor(il, "ffn_gate.weight");
            layer.w_up = layer_tensor(il, "ffn_up.weight");
            layer.w_down = layer_tensor(il, "ffn_down.weight");
            if (!layer.w_gate || !layer.w_up || !layer.w_down) {
                return fail("layer " + std::to_string(il) + " missing dense FFN tensor");
            }
        } else {
            layer.ffn_gate_inp = layer_tensor(il, "ffn_gate_inp.weight");
            layer.ffn_exp_probs_b = layer_tensor(il, "exp_probs_b.bias");
            layer.ffn_gate_exps = layer_tensor(il, "ffn_gate_exps.weight");
            layer.ffn_up_exps = layer_tensor(il, "ffn_up_exps.weight");
            layer.ffn_down_exps = layer_tensor(il, "ffn_down_exps.weight");
            layer.ffn_gate_shexp = layer_tensor(il, "ffn_gate_shexp.weight");
            layer.ffn_up_shexp = layer_tensor(il, "ffn_up_shexp.weight");
            layer.ffn_down_shexp = layer_tensor(il, "ffn_down_shexp.weight");
            if (!layer.ffn_gate_inp || !layer.ffn_exp_probs_b ||
                !layer.ffn_gate_exps || !layer.ffn_up_exps ||
                !layer.ffn_down_exps || !layer.ffn_gate_shexp ||
                !layer.ffn_up_shexp || !layer.ffn_down_shexp) {
                return fail("layer " + std::to_string(il) + " missing MoE tensor");
            }
        }
    }

    // Allocate exactly the tensors referenced by the trunk. Prefix-based
    // selection would also upload the 124B model's embedded MTP block.
    std::unordered_set<ggml_tensor *> wanted;
    auto add = [&](ggml_tensor * value) { if (value) wanted.insert(value); };
    add(out.out_norm);
    add(out.output);
    for (TargetLayer & layer : out.layers) {
        add(layer.attn_norm); add(layer.ffn_norm);
        add(layer.w_gate); add(layer.w_up); add(layer.w_down);
        add(layer.wq); add(layer.wk); add(layer.wv); add(layer.wo);
        add(layer.ssm_conv1d_q); add(layer.ssm_conv1d_k); add(layer.ssm_conv1d_v);
        add(layer.ssm_f_a); add(layer.ssm_beta); add(layer.ssm_a);
        add(layer.ssm_dt_bias); add(layer.ssm_g_a); add(layer.ssm_norm);
        add(layer.attn_q_a); add(layer.attn_q_a_norm); add(layer.attn_q_b);
        add(layer.attn_kv_a_mqa); add(layer.attn_kv_a_norm);
        add(layer.attn_k_b); add(layer.attn_v_b); add(layer.wqkv_gate);
        add(layer.ffn_gate_inp); add(layer.ffn_exp_probs_b);
        add(layer.ffn_gate_exps); add(layer.ffn_up_exps); add(layer.ffn_down_exps);
        add(layer.ffn_gate_shexp); add(layer.ffn_up_shexp); add(layer.ffn_down_shexp);
    }

    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buffer_type);
    std::vector<TensorAllocation> allocations;
    allocations.reserve(wanted.size());
    size_t allocation_size = 0;
    for (int64_t tid = 0; tid < n_tensors; ++tid) {
        const char * name = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * value = ggml_get_tensor(meta_ctx, name);
        if (!value || wanted.find(value) == wanted.end()) continue;
        allocation_size = align_up(allocation_size, alignment);
        TensorAllocation allocation;
        allocation.tensor = value;
        allocation.file_offset =
            gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
        allocation.file_size = gguf_get_tensor_size(gctx, tid);
        allocation.buffer_offset = allocation_size;
        allocation_size += ggml_backend_buft_get_alloc_size(buffer_type, value);
        allocations.push_back(allocation);
    }
    if (allocations.size() != wanted.size()) {
        return fail("failed to resolve every trunk tensor in the GGUF table");
    }

    out.buf = ggml_backend_alloc_buffer(backend, allocation_size);
    if (!out.buf) return fail("weight buffer allocation failed");
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(out.buf));
    for (const TensorAllocation & allocation : allocations) {
        if (ggml_backend_tensor_alloc(out.buf, allocation.tensor,
                base + allocation.buffer_offset) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
            return fail("weight tensor allocation failed");
        }
    }

    GgufMmap mmap;
    std::string mmap_error;
    if (!mmap.open(path, mmap_error)) {
        ggml_backend_buffer_free(out.buf);
        out.buf = nullptr;
        return fail(mmap_error);
    }
    const uint8_t * bytes = static_cast<const uint8_t *>(mmap.data());
    const size_t file_size = mmap.size();
    for (const TensorAllocation & allocation : allocations) {
        if (allocation.file_offset + allocation.file_size < allocation.file_offset ||
            allocation.file_offset + allocation.file_size > file_size) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
            return fail("truncated tensor data for " +
                        std::string(allocation.tensor->name));
        }
        ggml_backend_tensor_set(allocation.tensor,
            bytes + allocation.file_offset, 0, allocation.file_size);
    }

    const int64_t token_tid = gguf_find_tensor(gctx, "token_embd.weight");
    if (token_tid < 0) {
        ggml_backend_buffer_free(out.buf);
        out.buf = nullptr;
        return fail("token_embd.weight missing from tensor table");
    }
    const size_t token_relative_offset = gguf_get_tensor_offset(gctx, token_tid);
    const size_t token_size = gguf_get_tensor_size(gctx, token_tid);
    const size_t data_offset = gguf_get_data_offset(gctx);
    if (!gguf_tensor_in_file(data_offset, token_relative_offset, token_size, file_size)) {
        ggml_backend_buffer_free(out.buf);
        out.buf = nullptr;
        return fail("truncated token_embd.weight");
    }
    out.embedder.tok_embd_owned.resize(token_size);
    std::memcpy(out.embedder.tok_embd_owned.data(),
        bytes + data_offset + token_relative_offset, token_size);
    out.embedder.tok_embd_bytes = out.embedder.tok_embd_owned.data();
    out.embedder.tok_embd_type = gguf_get_tensor_type(gctx, token_tid);
    out.embedder.n_embd = out.n_embd;
    out.embedder.n_vocab = out.n_vocab;
    out.embedder.row_bytes = token_size / static_cast<size_t>(out.n_vocab);

    gguf_free(gctx);
    gctx = nullptr;
    meta_ctx = nullptr;  // owned by out.ctx from here on

    char summary[384];
    std::snprintf(summary, sizeof(summary),
        "bailingmoe3 trunk loaded: %d layers (%d KDA + %d MLA), "
        "%zu tensors %.2f GiB, experts=%d/%d groups=%d/%d, "
        "MTP blocks ignored=%u, eos=%d",
        out.n_layer, out.n_layer - out.n_layer / out.full_attention_interval,
        out.n_layer / out.full_attention_interval, allocations.size(),
        allocation_size / (1024.0 * 1024.0 * 1024.0),
        out.n_expert_used, out.n_expert,
        out.n_expert_groups_used, out.n_expert_groups,
        nextn, out.eos_id);
    set_last_error(summary);
    std::fprintf(stderr, "[bailingmoe3] %s\n", summary);
    return true;
}

}  // namespace dflash::common
