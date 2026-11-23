// Native GGUF loader for Inkling 2 / Inkling 2 Mini (`inkling2_loop`, `inkling2`)
// plus native safetensors loader for its bundled DSpark speculator (`mtp.safetensors`).

#include "inkling2_internal.h"

#include "common/gguf_kv.h"
#include "common/gguf_mmap.h"

#include "ggml-alloc.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace luce::common {
namespace {

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::vector<bool> get_bool_array(const gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return {};
    if (gguf_get_arr_type(g, id) != GGUF_TYPE_BOOL) return {};
    const size_t n = gguf_get_arr_n(g, id);
    const auto * raw = static_cast<const int8_t *>(gguf_get_arr_data(g, id));
    std::vector<bool> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = (raw[i] != 0);
    return out;
}

// Compute packed row-concatenation of contiguous 2D tensors sharing identical ne[0] and type.
struct ConcatSpec {
    ggml_tensor * dst = nullptr;
    std::vector<ggml_tensor *> srcs;
};

} // namespace

void free_inkling2_weights(Inkling2Weights & w) {
    if (w.dspark.buf) { ggml_backend_buffer_free(w.dspark.buf); w.dspark.buf = nullptr; }
    if (w.dspark.ctx) { ggml_free(w.dspark.ctx); w.dspark.ctx = nullptr; }
    w.dspark.layers.clear();
    w.dspark.loaded = false;

    if (w.expert_buf)  { ggml_backend_buffer_free(w.expert_buf); w.expert_buf = nullptr; }
    if (w.expert_ctx)  { ggml_free(w.expert_ctx); w.expert_ctx = nullptr; }
    if (w.derived_buf) { ggml_backend_buffer_free(w.derived_buf); w.derived_buf = nullptr; }
    if (w.derived_ctx) { ggml_free(w.derived_ctx); w.derived_ctx = nullptr; }
    if (w.buf)         { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx)         { ggml_free(w.ctx); w.ctx = nullptr; }

    w.layers.clear();
    w.loop_layers.clear();
    w.schedule.clear();
    w.global_slots.clear();
    w.slot_to_global_idx.clear();
    w.split_exp = 0;
    w.output_tail = nullptr;
    w.output_tail_rows = 0;
    w.expert_backend = nullptr;
}

bool load_inkling2_gguf(const std::string & path,
                        ggml_backend_t target_backend,
                        ggml_backend_t expert_backend,
                        Inkling2Weights & out,
                        std::string & error) {
    free_inkling2_weights(out);
    out.expert_backend = expert_backend;

    ggml_context * meta = nullptr;
    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g || !meta) {
        if (g) gguf_free(g);
        error = "inkling2_loader: failed to open GGUF: " + path;
        return false;
    }

    const int64_t arch_k = gguf_find_key(g, "general.architecture");
    const std::string arch = (arch_k >= 0) ? gguf_get_val_str(g, arch_k) : "";
    if (arch != "inkling2_loop" && arch != "inkling2") {
        gguf_free(g);
        ggml_free(meta);
        error = "inkling2_loader: unexpected architecture '" + arch + "'";
        return false;
    }
    out.is_loop = (arch == "inkling2_loop");
    auto kv = [&](const char * suffix) { return arch + "." + suffix; };

    out.n_layer_phys     = (int) get_u32_or(g, kv("block_count"), 24);
    out.n_embd           = (int) get_u32_or(g, kv("embedding_length"), 2304);
    out.n_head           = (int) get_u32_or(g, kv("attention.head_count"), 24);
    out.n_head_kv        = (int) get_u32_or(g, kv("attention.head_count_kv"), 8);
    out.head_dim         = (int) get_u32_or(g, kv("attention.key_length"), 128);
    out.n_ff_exp         = (int) get_u32_or(g, kv("expert_feed_forward_length"),
                                            get_u32_or(g, kv("feed_forward_length"), 1152));
    out.n_expert         = (int) get_u32_or(g, kv("expert_count"), 128);
    out.n_expert_used    = (int) get_u32_or(g, kv("expert_used_count"), 8);
    out.n_vocab_padded   = (int) get_u32_or(g, kv("vocab_size"), 201024);
    out.n_vocab_unpadded = (int) get_u32_or(g, kv("vocab_size_unpadded"), out.n_vocab_padded);
    out.swa_window       = (int) get_u32_or(g, kv("attention.sliding_window"), 1024);
    out.d_rel            = (int) get_u32_or(g, kv("attention.relative_length"), 16);
    out.rel_extent       = (int) get_u32_or(g, kv("attention.relative_extent"), 1024);
    out.sconv_kernel     = (int) get_u32_or(g, kv("shortconv.l_cache"), 4);
    out.attn_res_block   = (int) get_u32_or(g, kv("attn_res.block_size"), 6);
    out.log_scale_floor  = (int) get_u32_or(g, kv("attention.log_scale_floor"), 8192);
    out.log_scale_alpha  = get_f32_or(g, kv("attention.log_scale_alpha"), 0.1f);
    out.rms_eps          = get_f32_or(g, kv("attention.layer_norm_rms_epsilon"), 1e-6f);
    out.qk_clip          = get_f32_or(g, kv("attention.qk_clip"), 22.62741661f);
    out.rel_clip         = get_f32_or(g, kv("attention.relative_clip"), 8.0f);
    out.situ_beta        = get_f32_or(g, kv("activation.situ_beta"), 4.0f);
    out.situ_linear_beta = get_f32_or(g, kv("activation.situ_linear_beta"), 25.0f);
    out.route_scale      = get_f32_or(g, kv("expert_weights_scale"), 8.0f);
    out.logit_softcap    = get_f32_or(g, kv("final_logit_softcapping"), 30.0f);
    out.context_length   = (int) get_u32_or(g, kv("context_length"), 262144);
    out.eos_id           = (int32_t) get_u32_or(g, "tokenizer.ggml.eos_token_id", 200006);
    out.eos_chat_id      = (int32_t) get_u32_or(g, "tokenizer.ggml.eot_token_id", (uint32_t) out.eos_id);
    out.sconv_channels   = 2 * out.n_head_kv * out.head_dim + 2 * out.n_embd;

    if (out.is_loop) {
        out.core_start = (int) get_u32_or(g, kv("loop.start"), 6);
        out.core_end   = (int) get_u32_or(g, kv("loop.end"), 18);
        out.n_passes   = (int) get_u32_or(g, kv("loop.count"), 3);
        out.gate_rank  = (int) get_u32_or(g, kv("loop.gate_rank"), 64);
        out.n_layer_eff = out.n_layer_phys + (out.n_passes - 1) * (out.core_end - out.core_start);
    } else {
        out.core_start = 0;
        out.core_end   = 0;
        out.n_passes   = 1;
        out.gate_rank  = 0;
        out.n_layer_eff = out.n_layer_phys;
    }

    const std::vector<bool> swa_pat = get_bool_array(g, kv("attention.sliding_window_pattern"));
    if ((int) swa_pat.size() != out.n_layer_phys) {
        gguf_free(g);
        ggml_free(meta);
        error = "inkling2_loader: missing or invalid attention.sliding_window_pattern";
        return false;
    }

    // Build visit execution schedule & global slot table
    out.schedule.clear();
    out.schedule.reserve(out.n_layer_eff);
    int visit_idx = 0;
    if (out.is_loop) {
        for (int il = 0; il < out.core_start; ++il) {
            out.schedule.push_back({visit_idx++, il, il, 0, false});
        }
        for (int pass = 0; pass < out.n_passes; ++pass) {
            for (int ip = out.core_start; ip < out.core_end; ++ip) {
                const int il = (pass == 0) ? ip : (out.n_layer_phys + (pass - 1) * (out.core_end - out.core_start) + (ip - out.core_start));
                out.schedule.push_back({visit_idx++, il, ip, pass, true});
            }
        }
        for (int il = out.core_end; il < out.n_layer_phys; ++il) {
            out.schedule.push_back({visit_idx++, il, il, 0, false});
        }
    } else {
        for (int il = 0; il < out.n_layer_phys; ++il) {
            out.schedule.push_back({visit_idx++, il, il, 0, false});
        }
    }

    out.slot_to_global_idx.assign(out.n_layer_eff, -1);
    out.global_slots.clear();
    for (int il = 0; il < out.n_layer_eff; ++il) {
        const int ip = out.phys_layer_for_slot(il);
        if (!swa_pat[ip]) {
            out.slot_to_global_idx[il] = (int) out.global_slots.size();
            out.global_slots.push_back(il);
        }
    }

    // Allocate all raw GGUF tensors on target_backend (fits easily in 32 GB R9700: ~17.6 GB for IQ4_NL!)
    out.ctx = meta;
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, target_backend);
    if (!out.buf) {
        gguf_free(g);
        free_inkling2_weights(out);
        error = "inkling2_loader: failed to allocate model weight buffer on target device";
        return false;
    }

    // Memory-map GGUF and upload tensors cleanly
    GgufMmap mm;
    if (!mm.open(path, error)) {
        gguf_free(g);
        free_inkling2_weights(out);
        return false;
    }
    const uint8_t * mm_bytes = static_cast<const uint8_t *>(mm.data());

    const size_t data_offset = gguf_get_data_offset(g);
    const int64_t n_tensors = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * tname = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(out.ctx, tname);
        if (!t) continue;
        const size_t off = data_offset + gguf_get_tensor_offset(g, i);
        const size_t nb  = ggml_nbytes(t);
        ggml_backend_tensor_set(t, mm_bytes + off, 0, nb);
    }

    auto req = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(out.ctx, name.c_str());
        if (!t) error = "inkling2_loader: missing required tensor " + name;
        return t;
    };
    auto opt = [&](const std::string & name) -> ggml_tensor * {
        return ggml_get_tensor(out.ctx, name.c_str());
    };

    out.tok_embd         = req("token_embd.weight");
    out.tok_norm         = req("token_embd_norm.weight");
    out.output           = req("output.weight");
    out.output_norm      = req("output_norm.weight");
    out.output_res_query = req("output_res_query.weight");
    if (!error.empty()) { gguf_free(g); free_inkling2_weights(out); return false; }

    // Set up CPU embedder copy for fast host lookup if needed (optional, GPU token_embd is resident too)
    {
        const int64_t tid = gguf_find_tensor(g, "token_embd.weight");
        if (tid >= 0) {
            const size_t off = data_offset + gguf_get_tensor_offset(g, tid);
            const size_t nb  = ggml_nbytes(out.tok_embd);
            out.embedder.tok_embd_owned.resize(nb);
            std::memcpy(out.embedder.tok_embd_owned.data(), mm_bytes + off, nb);
            out.embedder.tok_embd_bytes = out.embedder.tok_embd_owned.data();
            out.embedder.tok_embd_type  = out.tok_embd->type;
            out.embedder.n_embd         = out.tok_embd->ne[0];
            out.embedder.n_vocab        = out.tok_embd->ne[1];
            out.embedder.row_bytes      = ggml_row_size(out.tok_embd->type, out.tok_embd->ne[0]);
        }
    }

    out.layers.resize(out.n_layer_phys);
    for (int i = 0; i < out.n_layer_phys; ++i) {
        auto & L = out.layers[i];
        L.is_swa = swa_pat[i];
        auto bn = [&](const char * s) { return "blk." + std::to_string(i) + "." + s; };

        L.wq             = req(bn("attn_q.weight"));
        L.wk             = req(bn("attn_k.weight"));
        L.wv             = req(bn("attn_v.weight"));
        L.wo             = req(bn("attn_output.weight"));
        L.wqkv_gate      = req(bn("attn_gate.weight"));
        L.attn_q_clip    = req(bn("attn_q_clip.weight"));
        L.attn_sink_proj = req(bn("attn_sink_proj.weight"));
        L.attn_rel       = req(bn("attn_rel.weight"));
        L.attn_rel_bias  = req(bn("attn_rel_bias.weight"));
        L.attn_rel_clip  = req(bn("attn_rel_clip.weight"));
        L.attn_k_conv    = req(bn("attn_k_conv.weight"));
        L.attn_v_conv    = req(bn("attn_v_conv.weight"));
        L.attn_post_conv = req(bn("attn_post_conv.weight"));
        L.ffn_post_conv  = req(bn("ffn_post_conv.weight"));
        if (L.is_swa) {
            L.attn_norm = req(bn("attn_norm.weight"));
        } else {
            L.attn_q_inp_norm  = req(bn("attn_q_inp_norm.weight"));
            L.attn_k_inp_norm  = req(bn("attn_k_inp_norm.weight"));
            L.attn_v_inp_norm  = req(bn("attn_v_inp_norm.weight"));
            L.attn_k_conv_proj = req(bn("attn_k_conv_proj.weight"));
            L.attn_v_conv_proj = req(bn("attn_v_conv_proj.weight"));
        }
        if (i > 0) L.attn_res_query = req(bn("attn_res_query.weight"));
        L.ffn_res_query    = req(bn("ffn_res_query.weight"));
        L.ffn_norm         = req(bn("ffn_norm.weight"));
        L.ffn_gate_inp     = req(bn("ffn_gate_inp.weight"));
        L.ffn_exp_probs_b  = opt(bn("exp_probs_b.bias"));
        L.ffn_gate_up_exps = req(bn("ffn_gate_up_exps.weight"));
        L.ffn_down_exps    = req(bn("ffn_down_exps.weight"));
        if (!error.empty()) { gguf_free(g); free_inkling2_weights(out); return false; }
    }

    if (out.is_loop) {
        const int n_core = out.core_end - out.core_start;
        out.loop_layers.resize(n_core);
        static const char * kCtrlNames[4] = {"q", "k", "v", "ffn"};
        for (int c_i = 0; c_i < n_core; ++c_i) {
            const int il = out.core_start + c_i;
            auto & LL = out.loop_layers[c_i];
            LL.msg_query = req("blk." + std::to_string(il) + ".loop_msg_query.weight");
            for (int c = 0; c < 4; ++c) {
                const std::string p = "blk." + std::to_string(il) + ".loop_" + kCtrlNames[c] + "_ctrl_";
                LL.ctrl_down[c]   = req(p + "down.weight");
                LL.ctrl_gain[c]   = req(p + "gain.weight");
                LL.ctrl_forget[c] = req(p + "forget.weight");
            }
            if (!error.empty()) { gguf_free(g); free_inkling2_weights(out); return false; }
        }
    }

    // ─── Load-time weight fusions on target_backend ─────────────────────
    // Concatenate projections sharing the exact same input vector so each layer
    // runs 1 fused Q8_0 GEMV/GEMM instead of 4-6 tiny launches!
    // Also merge ctrl_gain | ctrl_forget -> ctrl_gf [gate_rank, 2*n_embd],
    // and stack global attn_{q,k,v}_inp_norm -> [n_embd, 3].
    {
        const size_t n_derived = (size_t) out.n_layer_phys * 6 + out.loop_layers.size() * 12 + 8;
        struct ggml_init_params dgp = { n_derived * ggml_tensor_overhead() + 4096, nullptr, true };
        out.derived_ctx = ggml_init(dgp);
        std::vector<ConcatSpec> concats;
        struct Q8CopySpec {
            ggml_tensor * dst = nullptr;
            std::vector<ggml_tensor *> srcs;
        };
        std::vector<Q8CopySpec> q8_copies;
        const bool f32_ctrl = (std::getenv("LUCE_INKLING2_F32_CTRL") != nullptr &&
                               std::atoi(std::getenv("LUCE_INKLING2_F32_CTRL")) != 0);

        for (int i = 0; i < out.n_layer_phys; ++i) {
            auto & L = out.layers[i];
            // Check all Q/Gate/R/Sink share identical type (Q8_0 in official GGUFs)
            if (L.wq->type == L.wqkv_gate->type && L.wq->type == L.attn_rel->type && L.wq->type == L.attn_sink_proj->type) {
                const int64_t rows_qgrs = L.wq->ne[1] + L.wqkv_gate->ne[1] + L.attn_rel->ne[1] + L.attn_sink_proj->ne[1];
                L.wqgrs = ggml_new_tensor_2d(out.derived_ctx, L.wq->type, L.wq->ne[0], rows_qgrs);
                concats.push_back({L.wqgrs, {L.wq, L.wqkv_gate, L.attn_rel, L.attn_sink_proj}});

                if (L.is_swa && L.wk->type == L.wq->type && L.wv->type == L.wq->type) {
                    const int64_t rows_all = rows_qgrs + L.wk->ne[1] + L.wv->ne[1];
                    L.wqkvgrs = ggml_new_tensor_2d(out.derived_ctx, L.wq->type, L.wq->ne[0], rows_all);
                    concats.push_back({L.wqkvgrs, {L.wq, L.wqkv_gate, L.attn_rel, L.attn_sink_proj, L.wk, L.wv}});
                }
            }
            if (!L.is_swa) {
                L.attn_qkv_norm = ggml_new_tensor_2d(out.derived_ctx, GGML_TYPE_F32, out.n_embd, 3);
                concats.push_back({L.attn_qkv_norm, {L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm}});
                if (L.wk->type == L.attn_k_conv_proj->type) {
                    L.wk_dyn = ggml_new_tensor_2d(out.derived_ctx, L.wk->type, L.wk->ne[0], L.wk->ne[1] + L.attn_k_conv_proj->ne[1]);
                    concats.push_back({L.wk_dyn, {L.wk, L.attn_k_conv_proj}});
                }
                if (L.wv->type == L.attn_v_conv_proj->type) {
                    L.wv_dyn = ggml_new_tensor_2d(out.derived_ctx, L.wv->type, L.wv->ne[0], L.wv->ne[1] + L.attn_v_conv_proj->ne[1]);
                    concats.push_back({L.wv_dyn, {L.wv, L.attn_v_conv_proj}});
                }
            }
            if (!f32_ctrl && L.ffn_gate_inp && L.ffn_gate_inp->type == GGML_TYPE_F32 && (out.n_embd % 32) == 0) {
                L.ffn_gate_inp_q8 = ggml_new_tensor_2d(out.derived_ctx, GGML_TYPE_Q8_0, out.n_embd, out.n_expert);
                q8_copies.push_back({L.ffn_gate_inp_q8, {L.ffn_gate_inp}});
            }
        }
        for (auto & LL : out.loop_layers) {
            for (int c = 0; c < 4; ++c) {
                if (f32_ctrl) {
                    LL.ctrl_down_f16[c] = LL.ctrl_down[c];
                    LL.ctrl_gf[c] = ggml_new_tensor_2d(out.derived_ctx, GGML_TYPE_F32, out.gate_rank, 2 * out.n_embd);
                    concats.push_back({LL.ctrl_gf[c], {LL.ctrl_gain[c], LL.ctrl_forget[c]}});
                } else {
                    // Q8_0 loop controllers: 30 MiB total (fits 2x inside R9700's 64 MB MALL!) AND hits
                    // blazing luce_mmq_twin_rows32 (8-15 us) instead of slow F16 mul_mat_f (87 us) on T=8 verify!
                    LL.ctrl_down_f16[c] = ggml_new_tensor_2d(out.derived_ctx, GGML_TYPE_Q8_0, 2 * out.n_embd, out.gate_rank);
                    LL.ctrl_gf[c]       = ggml_new_tensor_2d(out.derived_ctx, GGML_TYPE_Q8_0, out.gate_rank, 2 * out.n_embd);
                    q8_copies.push_back({LL.ctrl_down_f16[c], {LL.ctrl_down[c]}});
                    q8_copies.push_back({LL.ctrl_gf[c], {LL.ctrl_gain[c], LL.ctrl_forget[c]}});
                }
            }
            // 3-way stacked Q/K/V loop controllers for single-launch 3D batched GEMV/MMQ!
            if (f32_ctrl) {
                LL.ctrl_down_qkv = ggml_new_tensor_3d(out.derived_ctx, GGML_TYPE_F32, 2 * out.n_embd, out.gate_rank, 3);
                LL.ctrl_gf_qkv   = ggml_new_tensor_3d(out.derived_ctx, GGML_TYPE_F32, out.gate_rank, 2 * out.n_embd, 3);
                concats.push_back({LL.ctrl_down_qkv, {LL.ctrl_down[0], LL.ctrl_down[1], LL.ctrl_down[2]}});
                concats.push_back({LL.ctrl_gf_qkv, {LL.ctrl_gain[0], LL.ctrl_forget[0], LL.ctrl_gain[1], LL.ctrl_forget[1], LL.ctrl_gain[2], LL.ctrl_forget[2]}});
            } else {
                LL.ctrl_down_qkv = ggml_new_tensor_3d(out.derived_ctx, GGML_TYPE_Q8_0, 2 * out.n_embd, out.gate_rank, 3);
                LL.ctrl_gf_qkv   = ggml_new_tensor_3d(out.derived_ctx, GGML_TYPE_Q8_0, out.gate_rank, 2 * out.n_embd, 3);
                q8_copies.push_back({LL.ctrl_down_qkv, {LL.ctrl_down[0], LL.ctrl_down[1], LL.ctrl_down[2]}});
                q8_copies.push_back({LL.ctrl_gf_qkv, {LL.ctrl_gain[0], LL.ctrl_forget[0], LL.ctrl_gain[1], LL.ctrl_forget[1], LL.ctrl_gain[2], LL.ctrl_forget[2]}});
            }
        }
        if (expert_backend) {
            out.split_hot_lut = ggml_new_tensor_2d(out.derived_ctx, GGML_TYPE_I32, 1, out.n_expert);
        }

        out.derived_buf = ggml_backend_alloc_ctx_tensors(out.derived_ctx, target_backend);
        if (!out.derived_buf) {
            gguf_free(g);
            free_inkling2_weights(out);
            error = "inkling2_loader: failed to allocate derived fused weights buffer";
            return false;
        }

        // Populate fused tensors straight from mmap (fast zero-extra-GPU-copy host upload)
        for (const auto & cs : concats) {
            size_t dst_off = 0;
            for (ggml_tensor * s : cs.srcs) {
                const int64_t tid = gguf_find_tensor(g, s->name);
                const size_t nb = ggml_nbytes(s);
                if (tid >= 0) {
                    const size_t off = data_offset + gguf_get_tensor_offset(g, tid);
                    ggml_backend_tensor_set(cs.dst, mm_bytes + off, dst_off, nb);
                } else {
                    std::vector<uint8_t> tmp(nb);
                    ggml_backend_tensor_get(s, tmp.data(), 0, nb);
                    ggml_backend_tensor_set(cs.dst, tmp.data(), dst_off, nb);
                }
                dst_off += nb;
            }
        }
        // Populate Q8_0-compressed loop controller & router tensors straight from mmap
        for (const auto & qc : q8_copies) {
            size_t dst_off = 0;
            for (ggml_tensor * s : qc.srcs) {
                const int64_t tid = gguf_find_tensor(g, s->name);
                const int64_t n_cols = s->ne[0];
                const int64_t n_rows = ggml_nrows(s);
                const size_t n_el = (size_t) n_cols * (size_t) n_rows;
                std::vector<float> tmp_f32(n_el);
                if (tid >= 0) {
                    const size_t off = data_offset + gguf_get_tensor_offset(g, tid);
                    std::memcpy(tmp_f32.data(), mm_bytes + off, n_el * sizeof(float));
                } else {
                    ggml_backend_tensor_get(s, tmp_f32.data(), 0, n_el * sizeof(float));
                }
                const size_t nb = (size_t) n_rows * ggml_row_size(GGML_TYPE_Q8_0, n_cols);
                std::vector<uint8_t> tmp_q8(nb);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, tmp_f32.data(), tmp_q8.data(), 0, n_rows, n_cols, nullptr);
                ggml_backend_tensor_set(qc.dst, tmp_q8.data(), dst_off, nb);
                dst_off += nb;
            }
        }
    }

    // ─── Dual-GPU R9700 + Strix Halo Split Placement ────────────────────
    // Replicate experts + LM-head tail onto expert_backend (Strix Halo, 96 GiB GTT):
    //   - T=1 decode runs 100% on R9700 (zero PCIe hops!)
    //   - T>=2 DSpark verify & prefill splits experts 74% R9700 / 26% Strix concurrently!
    //   - LM-head tail (21% of vocab rows) runs concurrently on Strix!
    if (expert_backend) {
        // Default 74% R9700 / 26% Strix Halo balances bandwidth & keeps Strix cooler under sustained load.
        int split_pct = 74;
        if (const char * env_pct = std::getenv("LUCE_INKLING2_SPLIT_PCT")) {
            const int v = std::atoi(env_pct);
            if (v >= 0 && v <= 100) split_pct = v;
        }
        out.split_pct = split_pct;
        int split_exp = (out.n_expert * split_pct) / 100;
        if (const char * env = std::getenv("LUCE_INKLING2_SPLIT_EXP")) {
            const int v = std::atoi(env);
            if (v >= 0 && v <= out.n_expert) split_exp = v;
        }
        out.split_exp = split_exp;
        out.target_has_all_experts = true;

        std::vector<int32_t> lut(out.n_expert, -1);
        for (int e = 0; e < split_exp; ++e) lut[e] = e;
        ggml_backend_tensor_set(out.split_hot_lut, lut.data(), 0, lut.size() * sizeof(int32_t));

        // 21% of LM-head rows on Strix (rounded to 64 rows) balances R9700 (580 GB/s) & Strix (210 GB/s)
        const bool both_rdna4_rows =
            out.output->type == GGML_TYPE_Q8_0 &&
            ggml_backend_cuda_rdna4_rows_device(ggml_backend_cuda_get_device_id(target_backend)) &&
            ggml_backend_cuda_rdna4_rows_device(ggml_backend_cuda_get_device_id(expert_backend));
        const int64_t tail_rows = both_rdna4_rows ? ((out.n_vocab_padded * 21) / 100 / 64) * 64 : 0;
        out.output_tail_rows = tail_rows;

        const size_t n_exp_t = (size_t) out.n_layer_phys * 2 + 4;
        struct ggml_init_params egp = { n_exp_t * ggml_tensor_overhead() + 4096, nullptr, true };
        out.expert_ctx = ggml_init(egp);

        for (int i = 0; i < out.n_layer_phys; ++i) {
            auto & L = out.layers[i];
            L.exp_gate_up_tail = ggml_new_tensor_3d(
                out.expert_ctx, L.ffn_gate_up_exps->type,
                L.ffn_gate_up_exps->ne[0], L.ffn_gate_up_exps->ne[1], L.ffn_gate_up_exps->ne[2]);
            L.exp_down_tail = ggml_new_tensor_3d(
                out.expert_ctx, L.ffn_down_exps->type,
                L.ffn_down_exps->ne[0], L.ffn_down_exps->ne[1], L.ffn_down_exps->ne[2]);
        }
        if (tail_rows > 0) {
            out.output_tail = ggml_new_tensor_2d(out.expert_ctx, out.output->type, out.output->ne[0], tail_rows);
        }

        out.expert_buf = ggml_backend_alloc_ctx_tensors(out.expert_ctx, expert_backend);
        if (!out.expert_buf) {
            std::fprintf(stderr, "[inkling2] warning: failed to allocate expert_backend replica; falling back to single-GPU\n");
            if (out.expert_ctx) { ggml_free(out.expert_ctx); out.expert_ctx = nullptr; }
            out.expert_backend = nullptr;
            out.split_exp = 0;
            out.output_tail = nullptr;
            out.output_tail_rows = 0;
        } else {
            for (int i = 0; i < out.n_layer_phys; ++i) {
                auto & L = out.layers[i];
                const int64_t tid_gu = gguf_find_tensor(g, L.ffn_gate_up_exps->name);
                const int64_t tid_dn = gguf_find_tensor(g, L.ffn_down_exps->name);
                ggml_backend_tensor_set(L.exp_gate_up_tail, mm_bytes + data_offset + gguf_get_tensor_offset(g, tid_gu), 0, ggml_nbytes(L.exp_gate_up_tail));
                ggml_backend_tensor_set(L.exp_down_tail,    mm_bytes + data_offset + gguf_get_tensor_offset(g, tid_dn), 0, ggml_nbytes(L.exp_down_tail));
            }
            if (out.output_tail) {
                const int64_t tid_out = gguf_find_tensor(g, out.output->name);
                const size_t row_bytes = ggml_row_size(out.output->type, out.output->ne[0]);
                const size_t start_off = (size_t) (out.n_vocab_padded - tail_rows) * row_bytes;
                ggml_backend_tensor_set(out.output_tail, mm_bytes + data_offset + gguf_get_tensor_offset(g, tid_out) + start_off, 0, ggml_nbytes(out.output_tail));
            }
            std::fprintf(stderr,
                "[inkling2] dual-GPU split ready: split_exp=%d/%d (target=%.0f%%, expert=%.0f%%) lm_head_tail=%lld expert_buf=%.2f GiB\n",
                out.split_exp, out.n_expert,
                100.0 * out.split_exp / out.n_expert,
                100.0 * (out.n_expert - out.split_exp) / out.n_expert,
                (long long) out.output_tail_rows,
                (double) ggml_backend_buffer_get_size(out.expert_buf) / 1073741824.0);
        }
    }

    gguf_free(g);
    std::fprintf(stderr,
        "[inkling2] loaded %s: phys=%d eff=%d (core=[%d,%d)x%d) embd=%d heads=%d/%d exp=%d(top-%d) vocab=%d/%d buf=%.2f GiB derived=%.2f GiB\n",
        arch.c_str(), out.n_layer_phys, out.n_layer_eff, out.core_start, out.core_end, out.n_passes,
        out.n_embd, out.n_head, out.n_head_kv, out.n_expert, out.n_expert_used,
        out.n_vocab_unpadded, out.n_vocab_padded,
        (double) ggml_backend_buffer_get_size(out.buf) / 1073741824.0,
        (double) ggml_backend_buffer_get_size(out.derived_buf) / 1073741824.0);
    return true;
}

// ─── Native Safetensors Loader for Inkling 2 Mini DSpark (`mtp.safetensors`) ──
bool load_inkling2_dspark_safetensors(const std::string & path,
                                      ggml_backend_t backend,
                                      Inkling2Weights & weights,
                                      std::string & error) {
    auto & ds = weights.dspark;
    if (ds.buf) { ggml_backend_buffer_free(ds.buf); ds.buf = nullptr; }
    if (ds.ctx) { ggml_free(ds.ctx); ds.ctx = nullptr; }
    ds.layers.clear();
    ds.loaded = false;

    GgufMmap mm;
    if (!mm.open(path, error) || mm.size() < 16) {
        if (error.empty()) error = "inkling2_dspark: failed to mmap " + path;
        return false;
    }
    const uint8_t * mm_bytes = static_cast<const uint8_t *>(mm.data());

    uint64_t hdr_len = 0;
    std::memcpy(&hdr_len, mm_bytes, sizeof(hdr_len));
    if (hdr_len > mm.size() - 8) {
        error = "inkling2_dspark: corrupt safetensors header in " + path;
        return false;
    }

    nlohmann::json hdr;
    try {
        const char * hdr_begin = reinterpret_cast<const char *>(mm_bytes + 8);
        hdr = nlohmann::json::parse(hdr_begin, hdr_begin + hdr_len);
    } catch (const std::exception & e) {
        error = std::string("inkling2_dspark: header JSON parse failed: ") + e.what();
        return false;
    }

    const uint8_t * base = mm_bytes + 8 + hdr_len;
    const size_t data_bytes = mm.size() - (size_t) (8 + hdr_len);
    if (!hdr.is_object()) {
        error = "inkling2_dspark: safetensors header is not a JSON object in " + path;
        return false;
    }
    struct STEntry {
        std::string dtype;
        std::vector<int64_t> shape;
        size_t off = 0;
        size_t len = 0;
    };
    std::unordered_map<std::string, STEntry> entries;
    try {
        for (auto & it : hdr.items()) {
            if (it.key() == "__metadata__") continue;
            const auto & val = it.value();
            if (!val.is_object() || !val.contains("dtype") || !val.contains("shape") || !val.contains("data_offsets")) {
                error = "inkling2_dspark: malformed tensor entry " + it.key();
                return false;
            }
            STEntry e;
            e.dtype = val.at("dtype").get<std::string>();
            e.shape = val.at("shape").get<std::vector<int64_t>>();
            auto offs = val.at("data_offsets").get<std::vector<size_t>>();
            if (offs.size() != 2 || offs[1] < offs[0] || offs[1] > data_bytes) {
                error = "inkling2_dspark: out-of-bounds data_offsets for " + it.key();
                return false;
            }
            e.off = offs[0];
            e.len = offs[1] - offs[0];
            entries[it.key()] = std::move(e);
        }
    } catch (const std::exception & ex) {
        error = std::string("inkling2_dspark: header entry parse failed: ") + ex.what();
        return false;
    }

    auto get_f32 = [&](const std::string & name, std::vector<float> & out_vec, size_t expected_elems) -> bool {
        auto it = entries.find(name);
        if (it == entries.end()) {
            error = "inkling2_dspark: missing tensor " + name;
            return false;
        }
        const auto & e = it->second;
        size_t elem_bytes = 0;
        if (e.dtype == "BF16" || e.dtype == "F16") elem_bytes = 2;
        else if (e.dtype == "F32") elem_bytes = 4;
        else {
            error = "inkling2_dspark: unsupported dtype " + e.dtype + " for " + name;
            return false;
        }
        size_t n = 1;
        for (int64_t d : e.shape) {
            if (d < 0 || (d > 0 && n > (SIZE_MAX / elem_bytes) / (size_t) d)) {
                error = "inkling2_dspark: invalid shape for " + name;
                return false;
            }
            n *= (size_t) d;
        }
        if (n != expected_elems) {
            error = "inkling2_dspark: shape mismatch for " + name +
                    " (got " + std::to_string(n) + " elements, expected " + std::to_string(expected_elems) + ")";
            return false;
        }
        const size_t need_bytes = n * elem_bytes;
        if (need_bytes > e.len || e.off > data_bytes || need_bytes > data_bytes - e.off) {
            error = "inkling2_dspark: truncated tensor data for " + name;
            return false;
        }
        out_vec.resize(n);
        if (e.dtype == "BF16") {
            const auto * src = reinterpret_cast<const uint16_t *>(base + e.off);
            for (size_t i = 0; i < n; ++i) out_vec[i] = bf16_to_f32(src[i]);
        } else if (e.dtype == "F32") {
            std::memcpy(out_vec.data(), base + e.off, need_bytes);
        } else {
            const auto * src = reinterpret_cast<const ggml_fp16_t *>(base + e.off);
            for (size_t i = 0; i < n; ++i) out_vec[i] = ggml_fp16_to_fp32(src[i]);
        }
        return true;
    };

    auto f32_to_f16_vec = [](const std::vector<float> & src) {
        std::vector<ggml_fp16_t> dst(src.size());
        for (size_t i = 0; i < src.size(); ++i) dst[i] = ggml_fp32_to_fp16(src[i]);
        return dst;
    };

    ds.n_layer     = 3;
    ds.n_embd      = weights.n_embd;
    ds.n_head      = weights.n_head;
    ds.n_head_kv   = weights.n_head_kv;
    ds.head_dim    = weights.head_dim;
    ds.n_ff        = 9216;
    ds.block_size  = 8;
    ds.markov_rank = 256;
    ds.swa_window  = weights.swa_window;
    ds.rope_theta  = 1000000.0f;
    ds.rms_eps     = weights.rms_eps;
    ds.target_layer_ids = {5, 23, 41, weights.n_layer_eff};

    const int H = ds.n_embd;
    const int R = ds.markov_rank;
    const int V = weights.n_vocab_padded;
    const int q_sz  = ds.n_head * ds.head_dim;
    const int kv_sz = ds.n_head_kv * ds.head_dim;

    const size_t n_tensors = (size_t) ds.n_layer * 11 + 16;
    struct ggml_init_params gp = { n_tensors * ggml_tensor_overhead() + 4096, nullptr, true };
    ds.ctx = ggml_init(gp);

    for (int r = 0; r < 4; ++r) {
        ds.fuse_alpha[r] = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, 4);
    }
    ds.mask_embed = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, H);
    ds.norm       = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, H);

    // Quantize DSpark 3-layer linear weights to Q8_0 (hits fast RDNA4 luce_mmq_twin_rows32
    // instead of slow F16 mul_mat_f on B=8 draft -> ~3x faster draft transformer!)
    const bool f16_dspark = (std::getenv("LUCE_INKLING2_F16_DSPARK") != nullptr &&
                             std::atoi(std::getenv("LUCE_INKLING2_F16_DSPARK")) != 0);
    const ggml_type ds_lin_type = f16_dspark ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    auto upload_lin = [&](ggml_tensor * dst, const std::vector<float> & src, int64_t n_cols, int64_t n_rows, size_t dst_off) {
        if (dst->type == GGML_TYPE_Q8_0) {
            const size_t nb = (size_t) n_rows * ggml_row_size(GGML_TYPE_Q8_0, n_cols);
            std::vector<uint8_t> q8(nb);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), q8.data(), 0, n_rows, n_cols, nullptr);
            ggml_backend_tensor_set(dst, q8.data(), dst_off, nb);
        } else {
            auto v16 = f32_to_f16_vec(src);
            ggml_backend_tensor_set(dst, v16.data(), dst_off, v16.size() * sizeof(ggml_fp16_t));
        }
    };

    ds.layers.resize(ds.n_layer);
    for (int i = 0; i < ds.n_layer; ++i) {
        auto & L = ds.layers[i];
        L.is_local    = (i == 0 || i == 2); // local_layer_ids = [0, 2]
        L.dflare_norm = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, H);
        L.inp_norm    = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, H);
        L.post_norm   = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, H);
        L.q_norm      = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, ds.head_dim);
        L.k_norm      = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, ds.head_dim);
        L.k_norm_ctx  = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, ds.head_dim);
        L.wqkv        = ggml_new_tensor_2d(ds.ctx, ds_lin_type, H, q_sz + 2 * kv_sz);
        L.wkv_ctx     = ggml_new_tensor_2d(ds.ctx, ds_lin_type, H, 2 * kv_sz);
        L.wo          = ggml_new_tensor_2d(ds.ctx, ds_lin_type, q_sz, H);
        L.w_gate_up   = ggml_new_tensor_2d(ds.ctx, ds_lin_type, H, 2 * ds.n_ff);
        L.w_down      = ggml_new_tensor_2d(ds.ctx, ds_lin_type, ds.n_ff, H);
    }

    ds.markov_dflare_norm = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, H);
    ds.markov_w1          = ggml_new_tensor_2d(ds.ctx, GGML_TYPE_F16, R, V);
    ds.rnn_proj_ctx       = ggml_new_tensor_2d(ds.ctx, ds_lin_type, 2 * H, 3 * R);
    ds.rnn_proj_step      = ggml_new_tensor_2d(ds.ctx, GGML_TYPE_F16, 2 * R, 3 * R);
    ds.rnn_bias           = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, 3 * R);
    // Q8_0 fold_weight (54 MiB -> fits 100% inside R9700's 64 MB Infinity Cache across all 7 Markov steps!)
    ds.fold_weight        = ggml_new_tensor_2d(ds.ctx, f16_dspark ? GGML_TYPE_F16 : GGML_TYPE_Q8_0, R, V);
    ds.fold_bias          = ggml_new_tensor_1d(ds.ctx, GGML_TYPE_F32, V);
    const bool use_q4_head = (std::getenv("LUCE_INKLING2_DSPARK_Q4_HEAD") == nullptr ||
                              std::atoi(std::getenv("LUCE_INKLING2_DSPARK_Q4_HEAD")) != 0);
    if (use_q4_head && weights.output && weights.output->type == GGML_TYPE_Q8_0) {
        ds.lm_head_q4 = ggml_new_tensor_2d(ds.ctx, GGML_TYPE_Q4_0, H, V);
    }

    ds.buf = ggml_backend_alloc_ctx_tensors(ds.ctx, backend);
    if (!ds.buf) {
        error = "inkling2_dspark: failed to allocate draft weight buffer";
        ggml_free(ds.ctx); ds.ctx = nullptr;
        return false;
    }

    auto cleanup_fail = [&]() -> bool {
        if (ds.buf) { ggml_backend_buffer_free(ds.buf); ds.buf = nullptr; }
        if (ds.ctx) { ggml_free(ds.ctx); ds.ctx = nullptr; }
        ds.layers.clear();
        ds.loaded = false;
        return false;
    };

    std::vector<float> tmp, tmp2, tmp3;
    // 1. fuse_weight [4, 4] -> softmax per row -> fuse_alpha[0..3]
    if (!get_f32("fuse_weight", tmp, 16)) return cleanup_fail();
    for (int r = 0; r < 4; ++r) {
        float mx = tmp[r * 4];
        for (int c = 1; c < 4; ++c) mx = std::max(mx, tmp[r * 4 + c]);
        float sm = 0.0f;
        float row[4];
        for (int c = 0; c < 4; ++c) { row[c] = std::exp(tmp[r * 4 + c] - mx); sm += row[c]; }
        for (int c = 0; c < 4; ++c) row[c] /= sm;
        ggml_backend_tensor_set(ds.fuse_alpha[r], row, 0, sizeof(row));
    }

    if (!get_f32("mask_embed", tmp, (size_t) H)) return cleanup_fail();
    ggml_backend_tensor_set(ds.mask_embed, tmp.data(), 0, tmp.size() * sizeof(float));
    if (!get_f32("norm.weight", tmp, (size_t) H)) return cleanup_fail();
    ggml_backend_tensor_set(ds.norm, tmp.data(), 0, tmp.size() * sizeof(float));

    for (int i = 0; i < ds.n_layer; ++i) {
        auto & L = ds.layers[i];
        const std::string p = "layers." + std::to_string(i) + ".";
        auto load_f32 = [&](const std::string & k, ggml_tensor * dst) {
            if (!get_f32(p + k, tmp, (size_t) ggml_nelements(dst))) return false;
            ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size() * sizeof(float));
            return true;
        };
        if (!load_f32("dflare_norm.weight", L.dflare_norm)) return cleanup_fail();
        if (!load_f32("input_layernorm.weight", L.inp_norm)) return cleanup_fail();
        if (!load_f32("post_attention_layernorm.weight", L.post_norm)) return cleanup_fail();
        if (!load_f32("self_attn.q_norm.weight", L.q_norm)) return cleanup_fail();
        if (!load_f32("self_attn.k_norm.weight", L.k_norm)) return cleanup_fail();
        if (!load_f32("self_attn.k_norm_ctx.weight", L.k_norm_ctx)) return cleanup_fail();

        // Stack q_proj | k_proj | v_proj -> wqkv
        if (!get_f32(p + "self_attn.q_proj.weight", tmp, (size_t) q_sz * H) ||
            !get_f32(p + "self_attn.k_proj.weight", tmp2, (size_t) kv_sz * H) ||
            !get_f32(p + "self_attn.v_proj.weight", tmp3, (size_t) kv_sz * H)) return cleanup_fail();
        const size_t rs_h = ggml_row_size(ds_lin_type, H);
        upload_lin(L.wqkv, tmp,  H, q_sz,  0);
        upload_lin(L.wqkv, tmp2, H, kv_sz, (size_t) q_sz * rs_h);
        upload_lin(L.wqkv, tmp3, H, kv_sz, (size_t) (q_sz + kv_sz) * rs_h);

        // Stack k_proj_ctx | v_proj_ctx -> wkv_ctx
        if (!get_f32(p + "self_attn.k_proj_ctx.weight", tmp, (size_t) kv_sz * H) ||
            !get_f32(p + "self_attn.v_proj_ctx.weight", tmp2, (size_t) kv_sz * H)) return cleanup_fail();
        upload_lin(L.wkv_ctx, tmp,  H, kv_sz, 0);
        upload_lin(L.wkv_ctx, tmp2, H, kv_sz, (size_t) kv_sz * rs_h);

        if (!get_f32(p + "self_attn.o_proj.weight", tmp, (size_t) H * q_sz)) return cleanup_fail();
        upload_lin(L.wo, tmp, q_sz, H, 0);

        // Stack gate_proj | up_proj -> w_gate_up
        if (!get_f32(p + "mlp.gate_proj.weight", tmp, (size_t) ds.n_ff * H) ||
            !get_f32(p + "mlp.up_proj.weight", tmp2, (size_t) ds.n_ff * H)) return cleanup_fail();
        upload_lin(L.w_gate_up, tmp,  H, ds.n_ff, 0);
        upload_lin(L.w_gate_up, tmp2, H, ds.n_ff, (size_t) ds.n_ff * rs_h);

        if (!get_f32(p + "mlp.down_proj.weight", tmp, (size_t) H * ds.n_ff)) return cleanup_fail();
        upload_lin(L.w_down, tmp, ds.n_ff, H, 0);
    }

    // Markov head weights
    if (!get_f32("markov_head.dflare_norm.weight", tmp, (size_t) H)) return cleanup_fail();
    ggml_backend_tensor_set(ds.markov_dflare_norm, tmp.data(), 0, tmp.size() * sizeof(float));

    if (!get_f32("markov_head.markov_w1.weight", tmp, (size_t) V * R)) return cleanup_fail();
    {
        auto w1_16 = f32_to_f16_vec(tmp);
        ggml_backend_tensor_set(ds.markov_w1, w1_16.data(), 0, w1_16.size() * sizeof(ggml_fp16_t));
    }

    if (!get_f32("markov_head.rnn_proj.bias", tmp, (size_t) 3 * R)) return cleanup_fail();
    ggml_backend_tensor_set(ds.rnn_bias, tmp.data(), 0, tmp.size() * sizeof(float));

    // Split rnn_proj.weight [3*R, 2*R + 2*H] into step [: 2*R] and ctx [2*R :]
    if (!get_f32("markov_head.rnn_proj.weight", tmp, (size_t) 3 * R * (2 * R + 2 * H))) return cleanup_fail();
    {
        const int in_total = 2 * R + 2 * H;
        std::vector<float> step_w((size_t) 3 * R * (2 * R));
        std::vector<float> ctx_w((size_t) 3 * R * (2 * H));
        for (int row = 0; row < 3 * R; ++row) {
            const float * src_row = tmp.data() + (size_t) row * in_total;
            std::memcpy(step_w.data() + (size_t) row * (2 * R), src_row, (size_t) (2 * R) * sizeof(float));
            std::memcpy(ctx_w.data()  + (size_t) row * (2 * H), src_row + 2 * R, (size_t) (2 * H) * sizeof(float));
        }
        auto step_w16 = f32_to_f16_vec(step_w);
        ggml_backend_tensor_set(ds.rnn_proj_step, step_w16.data(), 0, step_w16.size() * sizeof(ggml_fp16_t));
        upload_lin(ds.rnn_proj_ctx, ctx_w, 2 * H, 3 * R, 0);
    }

    // Pre-fold lm_head @ hidden_out.weight and lm_head @ hidden_out.bias on GPU (<5 ms!)
    std::vector<float> ho_w, ho_b;
    if (!get_f32("markov_head.hidden_out.weight", ho_w, (size_t) H * R) ||
        !get_f32("markov_head.hidden_out.bias",   ho_b, (size_t) H)) return cleanup_fail();
    {
        // PyTorch hidden_out.weight is [H, R] row-major; transpose to [R, H] so each column is length H
        std::vector<float> ho_wt((size_t) R * H);
        for (int h = 0; h < H; ++h) {
            for (int r = 0; r < R; ++r) {
                ho_wt[(size_t) r * H + h] = ho_w[(size_t) h * R + r];
            }
        }
        struct ggml_init_params fgp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
        ggml_context * fctx = ggml_init(fgp);
        ggml_tensor * t_wt = ggml_new_tensor_2d(fctx, GGML_TYPE_F32, H, R);
        ggml_tensor * t_b  = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, H);
        ggml_tensor * prod_w = ggml_mul_mat(fctx, weights.output, t_wt); // [V, R] F32
        ggml_tensor * prod_w_t = ggml_cont(fctx, ggml_transpose(fctx, prod_w)); // [R, V] F32
        ggml_tensor * cp_w = ggml_cpy(fctx, prod_w_t, ds.fold_weight);   // [R, V] Q8_0 (or F16)
        ggml_tensor * prod_b = ggml_mul_mat(fctx, weights.output, t_b);  // [V] F32
        ggml_tensor * cp_b = ggml_cpy(fctx, prod_b, ds.fold_bias);       // [V] F32

        ggml_cgraph * gf = ggml_new_graph(fctx);
        ggml_build_forward_expand(gf, cp_w);
        ggml_build_forward_expand(gf, cp_b);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            ggml_gallocr_free(ga);
            ggml_free(fctx);
            error = "inkling2_dspark: failed to allocate GPU fold graph";
            return false;
        }
        ggml_backend_tensor_set(t_wt, ho_wt.data(), 0, ho_wt.size() * sizeof(float));
        ggml_backend_tensor_set(t_b,  ho_b.data(),  0, ho_b.size()  * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        ggml_gallocr_free(ga);
        ggml_free(fctx);
    }

    // Quantize a fast Q4_0 draft-only LM-head copy on GPU/CPU (2x faster DSpark base_logits;
    // target verification still uses exact Q8_0 weights.output!)
    if (ds.lm_head_q4 && weights.output) {
        const size_t q8_row = ggml_row_size(GGML_TYPE_Q8_0, H);
        const size_t q4_row = ggml_row_size(GGML_TYPE_Q4_0, H);
        std::vector<uint8_t> src_q8((size_t) V * q8_row);
        ggml_backend_tensor_get(weights.output, src_q8.data(), 0, src_q8.size());
        std::vector<uint8_t> dst_q4((size_t) V * q4_row);
        const auto * traits_q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < V; ++v) {
            float row_f32[4096];
            traits_q8->to_float(src_q8.data() + (size_t) v * q8_row, row_f32, H);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, row_f32, dst_q4.data() + (size_t) v * q4_row, 0, 1, H, nullptr);
        }
        ggml_backend_tensor_set(ds.lm_head_q4, dst_q4.data(), 0, dst_q4.size());
    }

    ds.loaded = true;
    std::fprintf(stderr,
        "[inkling2] DSpark loaded from %s: 3 layers + rnn_hidden Markov (rank=%d, block=%d, folded lm_head=%.1f MiB, total=%.1f MiB)\n",
        path.c_str(), R, ds.block_size,
        (double) ggml_nbytes(ds.fold_weight) / 1048576.0,
        (double) ggml_backend_buffer_get_size(ds.buf) / 1048576.0);
    return true;
}

} // namespace luce::common
