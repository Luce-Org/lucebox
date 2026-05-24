// Smoke test for single-token qwen35moe hybrid FFN evaluation.

#include "internal.h"
#include "qwen35moe_expert_placement.h"
#include "qwen35moe_hybrid_ffn_eval.h"
#include "qwen35moe_hybrid_storage.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35moe.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "cuda init failed\n");
        return 1;
    }

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    if (!w.is_moe) {
        std::fprintf(stderr, "target is not qwen35moe\n");
        return 1;
    }

    Qwen35MoeExpertPlacement placement;
    placement.n_layer = w.n_layer;
    placement.n_expert = w.n_expert;
    placement.n_expert_used = w.n_expert_used;
    placement.total_hot = w.n_layer;
    placement.hot_counts.assign((size_t)w.n_layer, 1);
    placement.hot_expert_ids.resize((size_t)w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        placement.hot_expert_ids[(size_t)il].push_back(0);
    }

    Qwen35MoeHybridStorage hybrid;
    std::string err;
    if (!build_qwen35moe_hybrid_storage(w, backend, placement, hybrid, &err)) {
        std::fprintf(stderr, "build_qwen35moe_hybrid_storage: %s\n", err.c_str());
        return 1;
    }

    const int layer_idx = 0;
    const TargetLayer & L = w.layers[(size_t)layer_idx];
    auto & storage = hybrid.layers[(size_t)layer_idx];

    auto dump = [](const char * name, ggml_tensor * t) {
        if (!t) {
            std::printf("%s=null\n", name);
            return;
        }
        std::printf("%s ne=[%lld,%lld,%lld,%lld]\n", name,
                    (long long)t->ne[0], (long long)t->ne[1],
                    (long long)t->ne[2], (long long)t->ne[3]);
    };
    dump("gate_hot", storage.gate_hot);
    dump("up_hot", storage.up_hot);
    dump("down_hot", storage.down_hot);
    dump("gate_cold", storage.gate_cold);
    dump("up_cold", storage.up_cold);
    dump("down_cold", storage.down_cold);

    std::vector<float> cur((size_t)w.n_embd);
    for (int i = 0; i < w.n_embd; ++i) {
        cur[(size_t)i] = 0.001f * (float)((i % 17) - 8);
    }
    const int32_t selected_ids[2] = {0, 1};
    const float selected_weights[2] = {0.6f, 0.4f};

    std::vector<float> ref_out, hybrid_out;
    if (!eval_qwen35moe_reference_ffn_single(backend, w, L, cur.data(),
                                             selected_ids, selected_weights, 2,
                                             ref_out, &err)) {
        std::fprintf(stderr, "reference ffn failed: %s\n", err.c_str());
        return 1;
    }
    if (!eval_qwen35moe_hybrid_ffn_single(backend, w, L, storage, hybrid.cpu_backend,
                                          cur.data(), selected_ids, selected_weights, 2,
                                          hybrid_out, nullptr, &err)) {
        std::fprintf(stderr, "hybrid ffn failed: %s\n", err.c_str());
        return 1;
    }

    float max_diff = 0.0f;
    for (int i = 0; i < w.n_embd; ++i) {
        max_diff = std::max(max_diff, std::fabs(ref_out[(size_t)i] - hybrid_out[(size_t)i]));
    }
    if (max_diff > 1e-3f) {
        std::fprintf(stderr, "hybrid/reference mismatch: max_diff=%.6f\n", max_diff);
        return 1;
    }

    hybrid = Qwen35MoeHybridStorage{};
    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}
