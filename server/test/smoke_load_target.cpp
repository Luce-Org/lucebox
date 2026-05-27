// Smoke test for the GGUF target loader. Loads Qwen3.5-27B from a GGUF,
// validates metadata, and prints per-layer-type counts + a spot value check.
//
// Usage: smoke_load_target <path/to/qwen35.gguf>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf failed: %s\n", dflash27b_last_error());
        return 1;
    }
    // load_target_gguf stashes a summary string in last_error on success (hack)
    std::printf("%s\n", dflash27b_last_error());

    // Count layer types
    int n_attn = 0, n_delta = 0;
    for (int il = 0; il < w.n_layer; il++) {
        const auto & L = w.layers[il];
        bool attn = L.wq && L.wk && L.wv && L.wo;
        bool ssm  = L.wqkv && L.wqkv_gate && L.ssm_conv1d;
        if (attn) n_attn++;
        if (ssm)  n_delta++;
    }
    std::printf("hparams: n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d/%d n_ff=%d fai=%d is_moe=%d\n",
        w.n_layer, w.n_embd, w.n_head, w.n_head_kv, w.n_embd_head_k, w.n_embd_head_v, w.n_ff, w.full_attention_interval, (int)w.is_moe);
    if (w.is_moe) {
        std::printf("moe:     n_expert=%d used=%d n_ff_exp=%d n_ff_shexp=%d gate_func=%d scale=%.3f\n",
            w.n_expert, w.n_expert_used, w.n_ff_exp, w.n_ff_shexp,
            w.expert_gating_func, w.expert_weights_scale);
    }
    std::printf("ssm:     conv=%d inner=%d state=%d dt_rank=%d n_group=%d\n",
        w.ssm_d_conv, w.ssm_d_inner, w.ssm_d_state, w.ssm_dt_rank, w.ssm_n_group);
    std::printf("rope sections: [%d, %d, %d, %d]\n",
        w.rope_sections[0], w.rope_sections[1], w.rope_sections[2], w.rope_sections[3]);
    std::printf("layer counts: full_attn=%d delta_net=%d\n", n_attn, n_delta);

    // Spot-check: tok_embd should be quantized (Q4_K or similar)
    std::printf("tok_embd: ne=[%" PRId64 ", %" PRId64 "] type=%s nbytes=%.2f MiB\n",
        w.tok_embd->ne[0], w.tok_embd->ne[1],
        ggml_type_name(w.tok_embd->type),
        ggml_nbytes(w.tok_embd) / (1024.0 * 1024.0));
    std::printf("output (lm_head): type=%s nbytes=%.2f MiB\n",
        ggml_type_name(w.output->type),
        ggml_nbytes(w.output) / (1024.0 * 1024.0));

    // Read output_norm back from CUDA and print a few values
    if (w.out_norm->type == GGML_TYPE_F32 && w.out_norm->ne[0] >= 8) {
        std::vector<float> buf(8);
        ggml_backend_tensor_get(w.out_norm, buf.data(), 0, sizeof(float) * 8);
        std::printf("output_norm first 8: ");
        for (int i = 0; i < 8; i++) std::printf("%.4f ", buf[i]);
        std::printf("\n");
    }

    // Peek at layer 0 (deltanet) and layer 3 (full-attn) to verify split
    auto print_layer = [&](int il) {
        const auto & L = w.layers[il];
        std::printf("layer %2d: %s ", il,
            (L.wq ? "FULL_ATTN" : "DELTANET "));
        if (L.wq) {
            std::printf("wq=%s[%" PRId64 ",%" PRId64 "] ",
                ggml_type_name(L.wq->type), L.wq->ne[0], L.wq->ne[1]);
        }
        if (L.wqkv) {
            std::printf("wqkv=%s[%" PRId64 ",%" PRId64 "] ",
                ggml_type_name(L.wqkv->type), L.wqkv->ne[0], L.wqkv->ne[1]);
        }
        if (L.w_down) {
            std::printf("ffn_down=%s\n", ggml_type_name(L.w_down->type));
        } else if (L.ffn_down_exps) {
            std::printf("ffn_down_exps=%s\n", ggml_type_name(L.ffn_down_exps->type));
        } else {
            std::printf("ffn=<missing>\n");
        }
    };
    print_layer(0);
    print_layer(3);
    if (w.n_layer > 31) print_layer(31);
    if (w.n_layer > 63) print_layer(63);

    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}
