#pragma once

// Pure mapping from the upstream llama.cpp DFlash 2 GGUF schema (arch "dflash",
// PR ggml-org/llama.cpp#27342) onto the hparams our draft loader consumes.
// No gguf/ggml dependency: the loader snapshots raw KV values into
// UpstreamDflashKv and gets back either hparams or a precise error.
//
// Upstream differs from our own converter's GGUFs (arch qwen35-dflash-draft):
//   * keys are `dflash.<name>` (un-nested), not `<arch>.dflash.dflash2.<name>`
//   * `dflash.target_layers` stores HF layer id + 1 (layer-INPUT convention);
//     we capture layer OUTPUTS, so the capture id is stored value - 1
//   * the MASK id lives in `tokenizer.ggml.mask_token_id`
//   * attention is non-causal (`dflash.attention.causal` = false) with a
//     per-layer sliding window

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace luce::common {

constexpr const char * kUpstreamDflashArch = "dflash";

struct UpstreamDflashKv {
    uint32_t n_layer = 0;                          // dflash.block_count
    uint32_t n_embd = 0;                           // dflash.embedding_length
    uint32_t fc_in = 0;                            // fc.weight ne[0] (0 = unknown)
    int      target_n_layer = 0;                   // target model depth (0 = unknown)
    std::optional<uint32_t> block_size;
    std::optional<uint32_t> conv_kernel_size;
    std::optional<uint32_t> conv_group_size;
    std::optional<uint32_t> selector_rank;
    std::optional<uint32_t> selector_top_k;
    std::vector<int32_t>    target_layers;         // dflash.target_layers
    std::optional<uint32_t> mask_token_id;         // tokenizer.ggml.mask_token_id
    std::optional<bool>     causal;                // dflash.attention.causal
    std::optional<uint32_t> sliding_window;
    std::vector<bool>       sliding_window_pattern;
    bool has_logit_scale = false;                  // dflash.logit_scale
    bool has_logit_softcap = false;                // dflash.final_logit_softcapping
    bool has_hyper_connection = false;             // DSV4 DSpark marker
};

struct UpstreamDflashHParams {
    uint32_t block_size = 0;
    uint32_t conv_kernel_size = 0;
    uint32_t conv_group_size = 0;
    uint32_t selector_rank = 0;
    uint32_t selector_top_k = 0;
    std::vector<int32_t> capture_layer_ids;        // layer-output ids (gguf - 1)
    uint32_t mask_token_id = 0;
    bool     causal = true;
    uint32_t swa_window = 0;
    std::vector<bool> swa_pattern;                 // size n_layer, or empty
};

inline bool resolve_upstream_dflash_hparams(const UpstreamDflashKv & kv,
                                            UpstreamDflashHParams & out,
                                            std::string & error) {
    out = UpstreamDflashHParams{};
    error.clear();
    auto fail = [&](const std::string & m) {
        error = "upstream DFlash GGUF: " + m;
        return false;
    };
    if (kv.has_hyper_connection)
        return fail("hyper_connection present (DSV4 DSpark drafter, not DFlash 2)");
    if (kv.has_logit_scale || kv.has_logit_softcap)
        return fail("dflash.logit_scale / final_logit_softcapping not supported");
    if (!kv.block_size || *kv.block_size < 2) return fail("missing/invalid dflash.block_size");
    if (!kv.conv_kernel_size || *kv.conv_kernel_size == 0)
        return fail("missing dflash.conv_kernel_size");
    if (!kv.conv_group_size || *kv.conv_group_size == 0)
        return fail("missing dflash.conv_group_size");
    if (!kv.selector_rank || *kv.selector_rank == 0) return fail("missing dflash.selector_rank");
    if (!kv.selector_top_k || *kv.selector_top_k == 0 || *kv.selector_top_k > 256)
        return fail("missing/out-of-range dflash.selector_top_k (1..256)");
    if (!kv.mask_token_id) return fail("missing tokenizer.ggml.mask_token_id");
    if (kv.target_layers.empty()) return fail("missing dflash.target_layers");
    if (kv.n_embd && kv.fc_in &&
        kv.fc_in != kv.n_embd * (uint32_t)kv.target_layers.size())
        return fail("fc input width " + std::to_string(kv.fc_in) + " != n_embd * " +
                    std::to_string(kv.target_layers.size()) + " target layers");
    for (int32_t id : kv.target_layers) {
        if (id < 1)
            return fail("dflash.target_layers entry " + std::to_string(id) +
                        " < 1 (stored value is HF layer id + 1)");
        if (kv.target_n_layer > 0 && id - 1 >= kv.target_n_layer)
            return fail("dflash.target_layers entry " + std::to_string(id) +
                        " beyond target depth " + std::to_string(kv.target_n_layer));
        out.capture_layer_ids.push_back(id - 1);
    }
    if (!kv.sliding_window_pattern.empty() && kv.sliding_window_pattern.size() != kv.n_layer)
        return fail("sliding_window_pattern has " +
                    std::to_string(kv.sliding_window_pattern.size()) + " entries for " +
                    std::to_string(kv.n_layer) + " layers");
    bool any_swa = false;
    for (bool b : kv.sliding_window_pattern) any_swa |= b;
    if (any_swa && (!kv.sliding_window || *kv.sliding_window == 0))
        return fail("sliding_window_pattern set but dflash.attention.sliding_window missing");
    out.block_size = *kv.block_size;
    out.conv_kernel_size = *kv.conv_kernel_size;
    out.conv_group_size = *kv.conv_group_size;
    out.selector_rank = *kv.selector_rank;
    out.selector_top_k = *kv.selector_top_k;
    out.mask_token_id = *kv.mask_token_id;
    out.causal = kv.causal.value_or(true);     // absent == causal == pre-DFlash-2 behaviour
    out.swa_window = kv.sliding_window.value_or(0);
    out.swa_pattern = kv.sliding_window_pattern;
    return true;
}

// Canonical (our converter's) tensor name -> upstream name; "" if no alias.
inline std::string upstream_dflash_tensor_alias(const std::string & canonical) {
    if (canonical == "dflash.fc.weight") return "fc.weight";
    if (canonical == "dflash.hidden_norm.weight") return "enc.output_norm.weight";
    if (canonical == "dflash.selector.hproj.weight") return "selector_hidden.weight";
    if (canonical == "dflash.selector.pred_cb") return "selector_predecessor.weight";
    if (canonical == "dflash.selector.succ_cb") return "selector_successor.weight";
    static const struct { const char * from; const char * to; } blk[] = {
        {"attn_conv.base", "attn_conv_base"},
        {"attn_conv.proj.weight", "attn_conv_proj.weight"},
        {"ffn_conv.base", "ffn_conv_base"},
        {"ffn_conv.proj.weight", "ffn_conv_proj.weight"},
    };
    if (canonical.rfind("blk.", 0) == 0) {
        const size_t dot = canonical.find('.', 4);
        if (dot != std::string::npos) {
            const std::string suffix = canonical.substr(dot + 1);
            for (const auto & b : blk)
                if (suffix == b.from) return canonical.substr(0, dot + 1) + b.to;
        }
    }
    return "";
}

} // namespace luce::common
