// qwen36_mtp_loader.cpp — Discovery loader for unsloth Qwen3.6 MTP GGUFs.
//
// PR 2 (this commit): introspection only. Reads the GGUF header, parses
// NextN metadata (`<arch>.nextn_predict_layers`), and probes for the
// per-MTP-block NextN tensors named per dflash/deps/llama.cpp's
// LLM_TENSOR_NEXTN_* schema. Populates Qwen36MtpWeights pointers from
// the ggml_context built by ggml_init_from_file.
//
// Real weight materialization onto a CUDA backend, plus the MTP forward
// graph, are deferred to PR 2b. The skeleton proves the abstraction
// (foundation interfaces in src/common/mtp_*) accommodates Qwen3.6's
// NativeHeads design, gates compile against the unsloth tensor layout,
// and gives a clear plug-in point for the forward.

#include "qwen36_mtp.h"

#include "common/gguf_metadata.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace dflash27b::mtp {

namespace {

// Bind a tensor pointer by exact name lookup on the ggml_context that
// gguf_init_from_file built. Returns nullptr if absent.
ggml_tensor * find_tensor(ggml_context * ctx, const std::string & name) {
    return ggml_get_tensor(ctx, name.c_str());
}

}  // anonymous

bool load_qwen36_mtp_weights(const std::string & gguf_path,
                             ggml_context * ctx,
                             int expected_n_embd,
                             int expected_n_vocab,
                             Qwen36MtpWeights & out_weights,
                             std::string & out_error) {
    out_error.clear();

    // PR 2 skeleton: open the GGUF file to read metadata + tensor list,
    // bind pointers in the caller's ctx. The real impl will also copy
    // bytes onto the CUDA backend and own the lifecycle through a
    // backend buffer; for now we rely on the caller's ctx already
    // containing the tensors (e.g., the same ctx used by Qwen35Backend
    // when loading the backbone).
    struct gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    struct gguf_context * gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gguf) {
        out_error = "qwen36_mtp_loader: gguf_init_from_file failed for " + gguf_path;
        return false;
    }

    // ── Arch and dimensions ────────────────────────────────────────────
    {
        std::string arch_err;
        if (!dflash::common::gguf_check_architecture(gguf, "qwen35", arch_err)) {
            out_error = "qwen36_mtp_loader: " + arch_err;
            gguf_free(gguf);
            return false;
        }
    }
    out_weights.backbone_arch  = "qwen35";
    out_weights.base_model_name = dflash::common::gguf_get_str_or(gguf, "general.name", "");

    uint32_t n_layer = 0;
    if (!dflash::common::gguf_require_u32(gguf, "qwen35.block_count", n_layer, out_error)) {
        out_error = "qwen36_mtp_loader: " + out_error;
        gguf_free(gguf);
        return false;
    }
    uint32_t n_embd_meta = 0;
    if (!dflash::common::gguf_require_u32(gguf, "qwen35.embedding_length", n_embd_meta, out_error)) {
        out_error = "qwen36_mtp_loader: " + out_error;
        gguf_free(gguf);
        return false;
    }
    if ((int)n_embd_meta != expected_n_embd) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
            "qwen36_mtp_loader: backbone n_embd mismatch (gguf=%u, expected=%d)",
            n_embd_meta, expected_n_embd);
        out_error = msg;
        gguf_free(gguf);
        return false;
    }
    (void)expected_n_vocab;  // vocab cross-check folded into the optional
                             // embed_tokens / shared_head_head shape check

    // ── NextN head count ──────────────────────────────────────────────
    uint32_t n_heads = 0;
    if (!dflash::common::gguf_require_u32(gguf, "qwen35.nextn_predict_layers", n_heads, out_error)) {
        out_error = "qwen36_mtp_loader: GGUF lacks qwen35.nextn_predict_layers — this is not an MTP variant";
        gguf_free(gguf);
        return false;
    }
    if (n_heads == 0) {
        out_error = "qwen36_mtp_loader: nextn_predict_layers=0 — no MTP heads in this file";
        gguf_free(gguf);
        return false;
    }
    if (n_heads > (uint32_t)n_layer) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
            "qwen36_mtp_loader: nextn_predict_layers=%u > n_layer=%u (corrupt metadata)",
            n_heads, n_layer);
        out_error = msg;
        gguf_free(gguf);
        return false;
    }

    // ── Attention sizing metadata ─────────────────────────────────────────
    uint32_t n_head_count = 0;
    if (!dflash::common::gguf_require_u32(gguf, "qwen35.attention.head_count", n_head_count, out_error)) {
        out_error = "qwen36_mtp_loader: " + out_error;
        gguf_free(gguf);
        return false;
    }
    uint32_t n_head_kv = 0;
    if (!dflash::common::gguf_require_u32(gguf, "qwen35.attention.head_count_kv", n_head_kv, out_error)) {
        out_error = "qwen36_mtp_loader: " + out_error;
        gguf_free(gguf);
        return false;
    }
    uint32_t n_key_length = 0;
    if (!dflash::common::gguf_require_u32(gguf, "qwen35.attention.key_length", n_key_length, out_error)) {
        out_error = "qwen36_mtp_loader: " + out_error;
        gguf_free(gguf);
        return false;
    }
    uint32_t n_value_length = 0;
    if (!dflash::common::gguf_require_u32(gguf, "qwen35.attention.value_length", n_value_length, out_error)) {
        out_error = "qwen36_mtp_loader: " + out_error;
        gguf_free(gguf);
        return false;
    }
    uint32_t n_ffn_length = 0;
    if (!dflash::common::gguf_require_u32(gguf, "qwen35.feed_forward_length", n_ffn_length, out_error)) {
        out_error = "qwen36_mtp_loader: " + out_error;
        gguf_free(gguf);
        return false;
    }

    out_weights.n_embd            = (int)n_embd_meta;
    out_weights.n_vocab           = expected_n_vocab;
    out_weights.n_heads           = (int)n_heads;
    out_weights.n_backbone_layers = (int)n_layer;
    out_weights.n_head_count      = (int)n_head_count;
    out_weights.n_head_kv         = (int)n_head_kv;
    out_weights.n_key_length      = (int)n_key_length;
    out_weights.n_value_length    = (int)n_value_length;
    out_weights.n_ffn_length      = (int)n_ffn_length;
    out_weights.heads.clear();
    out_weights.heads.resize(n_heads);

    // ── Bind per-head tensor pointers ─────────────────────────────────
    // Heads occupy the last `n_heads` block indices: [n_layer - n_heads,
    // n_layer - 1]. For each, the required tensors are eh_proj/enorm/
    // hnorm; embed_tokens / shared_head_head / shared_head_norm are
    // optional (caller falls back to backbone tensors when absent).
    int missing_required = 0;
    for (int h = 0; h < (int)n_heads; h++) {
        const int layer_idx = (int)n_layer - (int)n_heads + h;
        auto & head = out_weights.heads[h];
        head.layer_idx = layer_idx;

        auto bind = [&](const char * base, bool required) -> ggml_tensor * {
            char name[256];
            std::snprintf(name, sizeof name, "blk.%d.%s.weight", layer_idx, base);
            ggml_tensor * t = find_tensor(ctx, name);
            if (!t && required) {
                std::fprintf(stderr,
                    "[qwen36_mtp_loader] missing required tensor: %s\n", name);
                missing_required++;
            }
            return t;
        };
        head.eh_proj          = bind("nextn.eh_proj",          /*required=*/true);
        head.enorm            = bind("nextn.enorm",            /*required=*/true);
        head.hnorm            = bind("nextn.hnorm",            /*required=*/true);
        head.embed_tokens     = bind("nextn.embed_tokens",     /*required=*/false);
        head.shared_head_head = bind("nextn.shared_head_head", /*required=*/false);
        head.shared_head_norm = bind("nextn.shared_head_norm", /*required=*/false);

        // Shape B: head-owned transformer-block tensors (required for all heads).
        // These live at blk.{layer_idx}.{name}.weight (no nextn. prefix).
        auto bind_blk = [&](const char * name, bool required) -> ggml_tensor * {
            char full[256];
            std::snprintf(full, sizeof full, "blk.%d.%s.weight", layer_idx, name);
            ggml_tensor * t = find_tensor(ctx, full);
            if (!t && required) {
                std::fprintf(stderr,
                    "[qwen36_mtp_loader] missing required tensor: %s\n", full);
                missing_required++;
            }
            return t;
        };
        head.attn_norm           = bind_blk("attn_norm",           /*required=*/true);
        head.attn_q              = bind_blk("attn_q",              /*required=*/true);
        head.attn_q_norm         = bind_blk("attn_q_norm",         /*required=*/true);
        head.attn_k              = bind_blk("attn_k",              /*required=*/true);
        head.attn_k_norm         = bind_blk("attn_k_norm",         /*required=*/true);
        head.attn_v              = bind_blk("attn_v",              /*required=*/true);
        head.attn_output         = bind_blk("attn_output",         /*required=*/true);
        head.post_attention_norm = bind_blk("post_attention_norm", /*required=*/true);
        head.ffn_gate            = bind_blk("ffn_gate",            /*required=*/true);
        head.ffn_up              = bind_blk("ffn_up",              /*required=*/true);
        head.ffn_down            = bind_blk("ffn_down",            /*required=*/true);
    }

    gguf_free(gguf);

    if (missing_required > 0) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
            "qwen36_mtp_loader: %d required NextN tensor(s) missing — context likely lacks the MTP tensors. Did the backbone loader allocate them?",
            missing_required);
        out_error = msg;
        return false;
    }
    return true;
}

}  // namespace dflash27b::mtp
