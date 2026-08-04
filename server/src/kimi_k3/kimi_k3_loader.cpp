#include "kimi_k3_internal.h"

#include "common/gguf_bounds.h"
#include "common/gguf_mmap.h"
#include "internal.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
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
    if (type == GGUF_TYPE_UINT8)  return gguf_get_val_u8(g, id);
    if (type == GGUF_TYPE_UINT16) return gguf_get_val_u16(g, id);
    if (type == GGUF_TYPE_UINT32) return gguf_get_val_u32(g, id);
    if (type == GGUF_TYPE_UINT64) {
        const uint64_t value = gguf_get_val_u64(g, id);
        return value <= std::numeric_limits<uint32_t>::max()
            ? static_cast<uint32_t>(value) : fallback;
    }
    if (type == GGUF_TYPE_INT8)   return static_cast<uint32_t>(gguf_get_val_i8(g, id));
    if (type == GGUF_TYPE_INT16)  return static_cast<uint32_t>(gguf_get_val_i16(g, id));
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

bool is_routed_expert_tensor(const std::string & name) {
    return name.find(".ffn_gate_exps.weight") != std::string::npos ||
           name.find(".ffn_up_exps.weight") != std::string::npos ||
           name.find(".ffn_down_exps.weight") != std::string::npos ||
           name.find(".ffn_gate_up_exps.weight") != std::string::npos;
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

bool discover_split_paths(const std::string & supplied,
                          uint32_t split_count,
                          std::vector<std::string> & out,
                          std::string & error) {
    out.clear();
    const size_t of = supplied.rfind("-of-");
    if (split_count <= 1 && of == std::string::npos) {
        out.push_back(supplied);
        return true;
    }

    if (of == std::string::npos) {
        error = "split.count is greater than one but the GGUF filename has no -NNNNN-of-NNNNN suffix";
        return false;
    }
    const size_t index_dash = supplied.rfind('-', of - 1);
    if (index_dash == std::string::npos || index_dash + 1 >= of) {
        error = "cannot locate the split index in the GGUF filename";
        return false;
    }
    size_t total_end = of + 4;
    while (total_end < supplied.size() &&
           supplied[total_end] >= '0' && supplied[total_end] <= '9') {
        ++total_end;
    }
    const std::string index_text =
        supplied.substr(index_dash + 1, of - index_dash - 1);
    const std::string total_text =
        supplied.substr(of + 4, total_end - (of + 4));
    if (index_text.empty() || total_text.empty() ||
        index_text.find_first_not_of("0123456789") != std::string::npos ||
        total_text.find_first_not_of("0123456789") != std::string::npos) {
        error = "invalid split index/count in the GGUF filename";
        return false;
    }
    uint64_t filename_total = 0;
    try {
        filename_total = std::stoull(total_text);
    } catch (...) {
        error = "GGUF filename split count is not an integer";
        return false;
    }
    if (split_count != 0 && filename_total != split_count) {
        error = "GGUF split.count disagrees with the filename";
        return false;
    }
    if (filename_total == 0 ||
        filename_total > std::numeric_limits<uint32_t>::max()) {
        error = "GGUF filename has an invalid split count";
        return false;
    }
    split_count = static_cast<uint32_t>(filename_total);

    const std::string prefix = supplied.substr(0, index_dash + 1);
    const std::string suffix = supplied.substr(total_end);
    out.reserve(split_count);
    for (uint32_t split = 1; split <= split_count; ++split) {
        std::ostringstream path;
        path << prefix << std::setw(static_cast<int>(index_text.size()))
             << std::setfill('0') << split
             << "-of-" << total_text << suffix;
        out.push_back(path.str());
    }
    return true;
}

struct TensorSource {
    ggml_tensor * tensor = nullptr;
    uint32_t shard = 0;
    size_t file_offset = 0;
    size_t file_size = 0;
};

} // namespace

bool load_kimi_k3_gguf(const std::string & path,
                       ggml_backend_t backend,
                       KimiK3Weights & out,
                       bool stream_routed_experts) {
    free_kimi_k3_weights(out);

    ggml_context * first_meta = nullptr;
    gguf_init_params first_params{};
    first_params.no_alloc = true;
    first_params.ctx = &first_meta;
    gguf_context * first_gguf =
        gguf_init_from_file(path.c_str(), first_params);
    if (!first_gguf || !first_meta) {
        set_last_error("Kimi-K3: failed to parse GGUF: " + path);
        if (first_gguf) gguf_free(first_gguf);
        if (first_meta) ggml_free(first_meta);
        return false;
    }

    // Only the first file of a standard split GGUF is required to carry
    // global metadata. If the caller supplied another shard, infer the count
    // from its canonical filename and reopen the set in numerical order.
    const uint32_t split_count =
        get_u32_or(first_gguf, "split.count", 0);
    std::vector<std::string> shard_paths;
    std::string discovery_error;
    if (!discover_split_paths(path, split_count, shard_paths,
                              discovery_error)) {
        gguf_free(first_gguf);
        ggml_free(first_meta);
        set_last_error("Kimi-K3: " + discovery_error);
        return false;
    }

    std::vector<gguf_context *> shard_ggufs;
    shard_ggufs.reserve(shard_paths.size());
    out.contexts.reserve(shard_paths.size());
    bool used_supplied = false;
    for (size_t shard = 0; shard < shard_paths.size(); ++shard) {
        if (shard_paths[shard] == path) {
            shard_ggufs.push_back(first_gguf);
            out.contexts.push_back(first_meta);
            used_supplied = true;
            continue;
        }
        ggml_context * meta = nullptr;
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &meta;
        gguf_context * gguf =
            gguf_init_from_file(shard_paths[shard].c_str(), params);
        if (!gguf || !meta) {
            if (gguf) gguf_free(gguf);
            if (meta) ggml_free(meta);
            for (gguf_context * opened : shard_ggufs) gguf_free(opened);
            if (!used_supplied) {
                gguf_free(first_gguf);
                ggml_free(first_meta);
            }
            free_kimi_k3_weights(out);
            set_last_error("Kimi-K3: failed to parse GGUF shard: " +
                           shard_paths[shard]);
            return false;
        }
        shard_ggufs.push_back(gguf);
        out.contexts.push_back(meta);
    }
    if (!used_supplied) {
        gguf_free(first_gguf);
        ggml_free(first_meta);
    }

    auto fail = [&](const std::string & message) {
        set_last_error("Kimi-K3: " + message);
        for (gguf_context * gguf : shard_ggufs) gguf_free(gguf);
        shard_ggufs.clear();
        free_kimi_k3_weights(out);
        return false;
    };

    gguf_context * gctx = nullptr;
    for (size_t shard = 0; shard < shard_ggufs.size(); ++shard) {
        const int64_t arch_id =
            gguf_find_key(shard_ggufs[shard], "general.architecture");
        if (arch_id >= 0) {
            if (std::strcmp(
                    gguf_get_val_str(shard_ggufs[shard], arch_id),
                    "kimi-k3") != 0) {
                return fail("general.architecture must be kimi-k3");
            }
            if (!gctx) gctx = shard_ggufs[shard];
        }
        const uint32_t count =
            get_u32_or(shard_ggufs[shard], "split.count", 0);
        if (count != 0 && count != shard_paths.size()) {
            return fail("split.count is inconsistent across GGUF shards");
        }
    }
    if (!gctx) {
        return fail("no shard contains general.architecture metadata");
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

    out.ctx       = out.contexts.front();
    out.backend   = backend;
    out.shard_paths = shard_paths;
    out.routed_experts_streamed = stream_routed_experts;
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

    std::unordered_map<std::string, TensorSource> tensors;
    for (size_t shard = 0; shard < shard_ggufs.size(); ++shard) {
        gguf_context * gguf = shard_ggufs[shard];
        ggml_context * meta = out.contexts[shard];
        const size_t data_start = gguf_get_data_offset(gguf);
        const int64_t count = gguf_get_n_tensors(gguf);
        for (int64_t tid = 0; tid < count; ++tid) {
            const char * name = gguf_get_tensor_name(gguf, tid);
            if (!name || tensors.find(name) != tensors.end()) {
                return fail(std::string("duplicate or unnamed tensor across shards: ") +
                            (name ? name : "<null>"));
            }
            TensorSource source;
            source.tensor = ggml_get_tensor(meta, name);
            source.shard = static_cast<uint32_t>(shard);
            source.file_offset =
                data_start + gguf_get_tensor_offset(gguf, tid);
            source.file_size = gguf_get_tensor_size(gguf, tid);
            tensors.emplace(name, source);
        }
    }
    auto get = [&](const char * name) -> ggml_tensor * {
        const auto found = tensors.find(name);
        return found == tensors.end() ? nullptr : found->second.tensor;
    };
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

    out.streamed_layer_regions.clear();
    out.max_streamed_expert_bytes = 0;
    for (int il = out.n_dense_lead; il < out.n_layer; ++il) {
        char gate_name[160], up_name[160], down_name[160];
        std::snprintf(gate_name, sizeof(gate_name),
                      "blk.%d.ffn_gate_exps.weight", il);
        std::snprintf(up_name, sizeof(up_name),
                      "blk.%d.ffn_up_exps.weight", il);
        std::snprintf(down_name, sizeof(down_name),
                      "blk.%d.ffn_down_exps.weight", il);
        const TensorSource & gate = tensors.at(gate_name);
        const TensorSource & up = tensors.at(up_name);
        const TensorSource & down = tensors.at(down_name);
        if (gate.file_size % static_cast<size_t>(out.n_expert) != 0 ||
            up.file_size % static_cast<size_t>(out.n_expert) != 0 ||
            down.file_size % static_cast<size_t>(out.n_expert) != 0) {
            return fail("routed expert tensor size is not divisible by expert_count");
        }
        LayerExpertRegions regions;
        regions.expert_bytes_gate =
            gate.file_size / static_cast<size_t>(out.n_expert);
        regions.expert_bytes_up =
            up.file_size / static_cast<size_t>(out.n_expert);
        regions.expert_bytes_down =
            down.file_size / static_cast<size_t>(out.n_expert);
        regions.gate_exps = {
            gate.file_offset, gate.file_size, gate.shard};
        regions.up_exps = {
            up.file_offset, up.file_size, up.shard};
        regions.down_exps = {
            down.file_offset, down.file_size, down.shard};
        const size_t expert_bytes =
            regions.expert_bytes_gate + regions.expert_bytes_up +
            regions.expert_bytes_down;
        out.max_streamed_expert_bytes =
            std::max(out.max_streamed_expert_bytes, expert_bytes);
        out.streamed_layer_regions.push_back(regions);
    }

    struct ResidentAlloc {
        ggml_tensor * tensor = nullptr;
        size_t file_offset = 0;
        size_t file_size = 0;
        size_t buffer_offset = 0;
    };
    ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t copied = 0;
    size_t skipped = 0;
    for (size_t shard = 0; shard < shard_ggufs.size(); ++shard) {
        gguf_context * gguf = shard_ggufs[shard];
        ggml_context * meta = out.contexts[shard];
        const size_t data_start = gguf_get_data_offset(gguf);
        std::vector<ResidentAlloc> allocs;
        size_t allocation_bytes = 0;
        for (int64_t tid = 0; tid < gguf_get_n_tensors(gguf); ++tid) {
            const char * tensor_name = gguf_get_tensor_name(gguf, tid);
            ggml_tensor * tensor = ggml_get_tensor(meta, tensor_name);
            const size_t bytes = gguf_get_tensor_size(gguf, tid);
            if (!tensor) continue;
            if (stream_routed_experts &&
                is_routed_expert_tensor(tensor_name)) {
                skipped += bytes;
                continue;
            }
            allocation_bytes = align_up(allocation_bytes, alignment);
            ResidentAlloc allocation;
            allocation.tensor = tensor;
            allocation.file_offset =
                data_start + gguf_get_tensor_offset(gguf, tid);
            allocation.file_size = bytes;
            allocation.buffer_offset = allocation_bytes;
            allocation_bytes +=
                ggml_backend_buft_get_alloc_size(buft, tensor);
            allocs.push_back(allocation);
        }
        if (allocs.empty()) continue;

        ggml_backend_buffer_t buffer =
            ggml_backend_alloc_buffer(backend, allocation_bytes);
        if (!buffer) {
            return fail("unable to allocate resident tensor buffer for shard " +
                        std::to_string(shard + 1));
        }
        ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        out.buffers.push_back(buffer);
        char * buffer_base =
            static_cast<char *>(ggml_backend_buffer_get_base(buffer));
        for (const ResidentAlloc & allocation : allocs) {
            if (ggml_backend_tensor_alloc(
                    buffer, allocation.tensor,
                    buffer_base + allocation.buffer_offset) !=
                GGML_STATUS_SUCCESS) {
                return fail("unable to bind a resident tensor allocation");
            }
        }

        GgufMmap mmap;
        std::string mmap_error;
        if (!mmap.open(shard_paths[shard], mmap_error)) {
            return fail(mmap_error);
        }
        const auto * base =
            static_cast<const uint8_t *>(mmap.data());
        for (const ResidentAlloc & allocation : allocs) {
            if (allocation.file_offset > mmap.size() ||
                allocation.file_size >
                    mmap.size() - allocation.file_offset) {
                return fail("resident tensor range is outside GGUF shard " +
                            std::to_string(shard + 1));
            }
            ggml_backend_tensor_set(
                allocation.tensor, base + allocation.file_offset,
                0, allocation.file_size);
            copied += allocation.file_size;
        }
    }
    out.buf = out.buffers.empty() ? nullptr : out.buffers.front();

    for (gguf_context * gguf : shard_ggufs) gguf_free(gguf);
    shard_ggufs.clear();
    std::fprintf(stderr,
        "[kimi-k3] loaded resident=%.2f GiB file-backed-experts=%.2f GiB "
        "shards=%zu layers=%d (KDA=%zu MLA=%zu) hidden=%d "
        "experts=%d top=%d latent=%d vocab=%d\n",
        static_cast<double>(copied) / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(skipped) / (1024.0 * 1024.0 * 1024.0),
        out.shard_paths.size(),
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
    for (ggml_backend_buffer_t buffer : w.buffers) {
        if (buffer) ggml_backend_buffer_free(buffer);
    }
    for (ggml_context * context : w.contexts) {
        if (context) ggml_free(context);
    }
    w = KimiK3Weights{};
}

} // namespace dflash::common
