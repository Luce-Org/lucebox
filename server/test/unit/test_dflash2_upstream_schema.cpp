#include "common/dflash2_upstream_schema.h"
#include "host_check.h"

#include <string>

using namespace luce::common;

static int g_checks = 0;

// Mirrors Qwen3.8-27B-DFlash2-Q8_0.gguf (arch "dflash").
static UpstreamDflashKv qwen38_kv() {
    UpstreamDflashKv kv;
    kv.n_layer = 5;
    kv.n_embd = 5120;
    kv.fc_in = 25600;
    kv.target_n_layer = 64;
    kv.block_size = 8;
    kv.conv_kernel_size = 2;
    kv.conv_group_size = 16;
    kv.selector_rank = 256;
    kv.selector_top_k = 16;
    kv.target_layers = {6, 20, 34, 48, 62};
    kv.mask_token_id = 248070;
    kv.causal = false;
    kv.sliding_window = 2048;
    kv.sliding_window_pattern = {true, true, true, true, true};
    return kv;
}

int main() {
    UpstreamDflashHParams hp;
    std::string err;

    // Happy path: values map 1:1, target_layers shift by -1 to capture ids.
    CHECK(resolve_upstream_dflash_hparams(qwen38_kv(), hp, err));
    CHECK(hp.block_size == 8);
    CHECK(hp.conv_kernel_size == 2 && hp.conv_group_size == 16);
    CHECK(hp.selector_rank == 256 && hp.selector_top_k == 16);
    CHECK((hp.capture_layer_ids == std::vector<int32_t>{5, 19, 33, 47, 61}));
    CHECK(hp.mask_token_id == 248070);
    CHECK(!hp.causal);
    CHECK(hp.swa_window == 2048);
    CHECK(hp.swa_pattern.size() == 5 && hp.swa_pattern[0] && hp.swa_pattern[4]);

    // Absent attention.causal keeps the causal (pre-DFlash-2) default.
    {
        auto kv = qwen38_kv();
        kv.causal.reset();
        CHECK(resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(hp.causal);
    }
    // No SWA metadata at all is fine (window 0, empty pattern).
    {
        auto kv = qwen38_kv();
        kv.sliding_window.reset();
        kv.sliding_window_pattern.clear();
        CHECK(resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(hp.swa_window == 0 && hp.swa_pattern.empty());
    }

    // Rejections.
    {
        auto kv = qwen38_kv(); kv.mask_token_id.reset();
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(err.find("mask_token_id") != std::string::npos);
    }
    {
        auto kv = qwen38_kv(); kv.target_layers = {0, 20, 34, 48, 62};
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(err.find("HF layer id + 1") != std::string::npos);
    }
    {
        auto kv = qwen38_kv(); kv.target_layers = {6, 20, 34, 48, 65};
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(err.find("beyond target depth") != std::string::npos);
    }
    {   // id 64 stored => capture layer 63 (last of 64): valid; 65 => 64: invalid
        auto kv = qwen38_kv(); kv.target_layers = {6, 20, 34, 48, 64};
        CHECK(resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(hp.capture_layer_ids.back() == 63);
    }
    {
        auto kv = qwen38_kv(); kv.fc_in = 5120 * 4;
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(err.find("fc input width") != std::string::npos);
    }
    {
        auto kv = qwen38_kv(); kv.sliding_window_pattern = {true, true, true};
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(err.find("sliding_window_pattern") != std::string::npos);
    }
    {
        auto kv = qwen38_kv(); kv.sliding_window.reset();
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
    }
    {
        auto kv = qwen38_kv(); kv.has_hyper_connection = true;
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
        CHECK(err.find("DSV4") != std::string::npos);
    }
    {
        auto kv = qwen38_kv(); kv.has_logit_scale = true;
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
    }
    {
        auto kv = qwen38_kv(); kv.block_size.reset();
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
    }
    {
        auto kv = qwen38_kv(); kv.selector_top_k = 300;
        CHECK(!resolve_upstream_dflash_hparams(kv, hp, err));
    }

    // Tensor-name aliases.
    CHECK(upstream_dflash_tensor_alias("dflash.fc.weight") == "fc.weight");
    CHECK(upstream_dflash_tensor_alias("dflash.hidden_norm.weight") == "enc.output_norm.weight");
    CHECK(upstream_dflash_tensor_alias("dflash.selector.hproj.weight") == "selector_hidden.weight");
    CHECK(upstream_dflash_tensor_alias("dflash.selector.pred_cb") == "selector_predecessor.weight");
    CHECK(upstream_dflash_tensor_alias("dflash.selector.succ_cb") == "selector_successor.weight");
    CHECK(upstream_dflash_tensor_alias("blk.0.attn_conv.base") == "blk.0.attn_conv_base");
    CHECK(upstream_dflash_tensor_alias("blk.4.attn_conv.proj.weight") == "blk.4.attn_conv_proj.weight");
    CHECK(upstream_dflash_tensor_alias("blk.12.ffn_conv.base") == "blk.12.ffn_conv_base");
    CHECK(upstream_dflash_tensor_alias("blk.3.ffn_conv.proj.weight") == "blk.3.ffn_conv_proj.weight");
    // Tensors that already share a name need no alias.
    CHECK(upstream_dflash_tensor_alias("output_norm.weight").empty());
    CHECK(upstream_dflash_tensor_alias("blk.0.attn_q.weight").empty());

    std::printf("test_dflash2_upstream_schema: %d checks passed\n", g_checks);
    return 0;
}
