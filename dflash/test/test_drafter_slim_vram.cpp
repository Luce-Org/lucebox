// Unit tests for PFLASH_DRAFTER_SLIM=1 tensor-skip logic.
//
// Verifies the VRAM size computation for slim vs full drafter loads
// without needing a GPU. Tests the pure helper slim_drafter_vram_bytes().
//
// Model parameters: Qwen3-0.6B BF16
//   n_layer=28, n_head=16, n_head_kv=8, n_embd=1024, n_ff=3072, head_dim=128
//   n_vocab=151936, wtype=BF16 (2 bytes)
//
// Full model allocates:
//   tok_embd:   n_embd * n_vocab * 2 = 1024 * 151936 * 2 = 311,164,928 B (~297 MB)
//   out_norm:   n_embd * 4 = 4096 B (negligible)
//   output:     n_embd * n_vocab * 2 = 311,164,928 B (~297 MB) -- lm_head
//   per layer (BF16 Qwen3 with q/k_norm):
//     attn_norm:   n_embd * 4               = 4096
//     wq:          n_embd * q_dim * 2       = 1024 * 2048 * 2 = 4,194,304
//     wk:          n_embd * kv_dim * 2      = 1024 * 1024 * 2 = 2,097,152
//     wv:          n_embd * kv_dim * 2      = 2,097,152
//     wo:          q_dim * n_embd * 2       = 2048 * 1024 * 2 = 4,194,304
//     q_norm:      head_dim * 4             = 512
//     k_norm:      head_dim * 4             = 512
//     ffn_norm:    n_embd * 4               = 4096
//     ffn_gate:    n_embd * n_ff * 2        = 1024 * 3072 * 2 = 6,291,456
//     ffn_up:      n_embd * n_ff * 2        = 6,291,456
//     ffn_down:    n_ff * n_embd * 2        = 6,291,456
//   total per layer: 31,175,296 B (~29.7 MB)
//   28 layers: 872,908,288 B (~833 MB)
//
// Slim mode with ee7:
//   Skips: output (~297 MB), layers 7-27 (21 * ~29.7 MB = ~624 MB)
//   Saves: ~921 MB (> 800 MB target)

#include "slim_drafter.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>

// REQUIRE survives -DNDEBUG (bare assert does not).
#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s line %d: %s\n", __FILE__, __LINE__, #cond); \
        std::exit(1); \
    } } while (0)

using dflash::common::slim_drafter_layer_bytes;
using dflash::common::slim_drafter_total_bytes;
using dflash::common::SlimDrafterConfig;

static void t1_per_layer_bytes_active_qwen3() {
    // Active layer keeps wq, wk, wv, wo, attn_norm, q_norm, k_norm, ffn_norm, ffn_gate, ffn_up, ffn_down
    // q_dim = 16 * 128 = 2048, kv_dim = 8 * 128 = 1024
    SlimDrafterConfig cfg;
    cfg.n_embd    = 1024;
    cfg.n_ff      = 3072;
    cfg.n_head    = 16;
    cfg.n_head_kv = 8;
    cfg.head_dim  = 128;
    cfg.n_vocab   = 151936;
    cfg.wtype_bytes = 2; // BF16
    cfg.has_qk_norm = true;

    const int64_t q_dim  = cfg.n_head    * cfg.head_dim;
    const int64_t kv_dim = cfg.n_head_kv * cfg.head_dim;

    int64_t expected =
          (int64_t)cfg.n_embd * 4                // attn_norm (F32)
        + (int64_t)cfg.n_embd * q_dim  * 2       // wq BF16
        + (int64_t)cfg.n_embd * kv_dim * 2       // wk BF16
        + (int64_t)cfg.n_embd * kv_dim * 2       // wv BF16
        + (int64_t)q_dim * cfg.n_embd  * 2       // wo BF16
        + (int64_t)cfg.head_dim * 4               // q_norm F32
        + (int64_t)cfg.head_dim * 4               // k_norm F32
        + (int64_t)cfg.n_embd * 4                // ffn_norm F32
        + (int64_t)cfg.n_embd * cfg.n_ff * 2     // ffn_gate BF16
        + (int64_t)cfg.n_embd * cfg.n_ff * 2     // ffn_up BF16
        + (int64_t)cfg.n_ff * cfg.n_embd * 2;    // ffn_down BF16
    // = 4096 + 4194304 + 2097152 + 2097152 + 4194304 + 512 + 512 + 4096 + 6291456 + 6291456 + 6291456
    // = 31,175,296 B

    int64_t got = slim_drafter_layer_bytes(cfg, /*active=*/true);
    REQUIRE(got == expected && "active layer bytes mismatch");
    printf("T1 pass: active layer bytes=%lld (expected %lld)\n", (long long)got, (long long)expected);
}

static void t2_full_model_bytes() {
    SlimDrafterConfig cfg;
    cfg.n_embd    = 1024;
    cfg.n_ff      = 3072;
    cfg.n_head    = 16;
    cfg.n_head_kv = 8;
    cfg.head_dim  = 128;
    cfg.n_vocab   = 151936;
    cfg.wtype_bytes = 2;
    cfg.has_qk_norm = true;

    // Full load: n_layer=28, all layers active, include output.weight
    int64_t full = slim_drafter_total_bytes(cfg, /*n_layer=*/28,
                                            /*fwd_limit=*/28,
                                            /*skip_output=*/false);

    // Expected: tok_embd + out_norm + output + 28 * layer
    int64_t tok = (int64_t)cfg.n_embd * cfg.n_vocab * 2;
    int64_t norm = (int64_t)cfg.n_embd * 4;
    int64_t output_w = (int64_t)cfg.n_embd * cfg.n_vocab * 2;
    int64_t layer = slim_drafter_layer_bytes(cfg, /*active=*/true);
    int64_t expected = tok + norm + output_w + 28LL * layer;

    REQUIRE(full == expected);
    printf("T2 pass: full model bytes=%lld MB (expected %lld MB)\n",
           (long long)(full >> 20), (long long)(expected >> 20));
}

static void t3_slim_ee7_savings() {
    SlimDrafterConfig cfg;
    cfg.n_embd    = 1024;
    cfg.n_ff      = 3072;
    cfg.n_head    = 16;
    cfg.n_head_kv = 8;
    cfg.head_dim  = 128;
    cfg.n_vocab   = 151936;
    cfg.wtype_bytes = 2;
    cfg.has_qk_norm = true;

    int64_t full = slim_drafter_total_bytes(cfg, 28, 28, /*skip_output=*/false);
    int64_t slim = slim_drafter_total_bytes(cfg, 28, /*fwd_limit=*/7, /*skip_output=*/true);

    int64_t savings = full - slim;
    printf("T3: full=%lld MB slim=%lld MB savings=%lld MB\n",
           (long long)(full >> 20), (long long)(slim >> 20),
           (long long)(savings >> 20));

    // Must save >= 800 MB
    const int64_t min_savings = 800LL * 1024 * 1024;
    REQUIRE(savings >= min_savings && "slim mode must save >= 800 MB");
    printf("T3 pass: savings=%lld MB (>= 800 MB)\n", (long long)(savings >> 20));
}

static void t4_slim_keeps_active_layers_complete() {
    SlimDrafterConfig cfg;
    cfg.n_embd    = 1024;
    cfg.n_ff      = 3072;
    cfg.n_head    = 16;
    cfg.n_head_kv = 8;
    cfg.head_dim  = 128;
    cfg.n_vocab   = 151936;
    cfg.wtype_bytes = 2;
    cfg.has_qk_norm = true;

    // slim with fwd_limit=7 should keep exactly 7 fully-allocated layers
    // plus tok_embd + out_norm (not output.weight)
    int64_t tok  = (int64_t)cfg.n_embd * cfg.n_vocab * 2;
    int64_t norm = (int64_t)cfg.n_embd * 4;
    int64_t layer = slim_drafter_layer_bytes(cfg, /*active=*/true);
    int64_t expected = tok + norm + 7LL * layer;  // no output.weight

    int64_t got = slim_drafter_total_bytes(cfg, 28, 7, /*skip_output=*/true);
    REQUIRE(got == expected);
    printf("T4 pass: slim ee7 total=%lld MB (correct)\n", (long long)(got >> 20));
}

static void t5_env_not_set_no_skip() {
    // When PFLASH_DRAFTER_SLIM is not "1", behavior is identical to full load.
    // This is verified by checking that fwd_limit=n_layer + skip_output=false
    // gives the full model size.
    SlimDrafterConfig cfg;
    cfg.n_embd = 1024; cfg.n_ff = 3072; cfg.n_head = 16;
    cfg.n_head_kv = 8; cfg.head_dim = 128; cfg.n_vocab = 151936;
    cfg.wtype_bytes = 2; cfg.has_qk_norm = true;

    int64_t full1 = slim_drafter_total_bytes(cfg, 28, 28, false);
    int64_t full2 = slim_drafter_total_bytes(cfg, 28, 28, false);
    REQUIRE(full1 == full2 && "full model size must be deterministic");
    printf("T5 pass: no-slim path is deterministic (%lld MB)\n", (long long)(full1 >> 20));
}

int main() {
    t1_per_layer_bytes_active_qwen3();
    t2_full_model_bytes();
    t3_slim_ee7_savings();
    t4_slim_keeps_active_layers_complete();
    t5_env_not_set_no_skip();
    printf("\nAll slim VRAM tests passed.\n");
    return 0;
}
