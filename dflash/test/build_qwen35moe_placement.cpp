// Build a qwen35moe placement.json from routing stats and a VRAM reserve target.
//
// Usage:
//   build_qwen35moe_placement <qwen35moe.gguf> <stats.json> <out.json>
//       --gpu-vram-gib <GiB> --reserve-gib <GiB>
//       [--safety-gib <GiB>] [--min-hot-per-layer <N>]
//
// Alternative:
//   build_qwen35moe_placement ... --total-hot-budget <N>

#include "internal.h"
#include "qwen35moe_expert_placement.h"
#include "qwen35moe_routing_stats.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

static uint64_t gib_to_bytes(double gib) {
    return (uint64_t)(gib * 1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <qwen35moe.gguf> <stats.json> <out.json> "
            "[--gpu-vram-gib X --reserve-gib Y [--safety-gib Z] [--min-hot-per-layer N] | --total-hot-budget N]\n",
            argv[0]);
        return 2;
    }

    const char * model_path = argv[1];
    const char * stats_path = argv[2];
    const char * out_path   = argv[3];

    double gpu_vram_gib = 0.0;
    double reserve_gib = 0.0;
    double safety_gib = 0.5;
    int min_hot_per_layer = 1;
    int total_hot_budget = 0;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu-vram-gib" && i + 1 < argc) {
            gpu_vram_gib = std::atof(argv[++i]);
        } else if (arg == "--reserve-gib" && i + 1 < argc) {
            reserve_gib = std::atof(argv[++i]);
        } else if (arg == "--safety-gib" && i + 1 < argc) {
            safety_gib = std::atof(argv[++i]);
        } else if (arg == "--min-hot-per-layer" && i + 1 < argc) {
            min_hot_per_layer = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--total-hot-budget" && i + 1 < argc) {
            total_hot_budget = std::max(0, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return 2;
        }
    }

    std::string err;
    Qwen35MoeRoutingStats stats;
    if (!Qwen35MoeRoutingStats::load_csv(stats_path, stats, &err)) {
        std::fprintf(stderr, "load stats: %s\n", err.c_str());
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "cuda init failed\n");
        return 1;
    }

    TargetWeights w;
    if (!load_target_gguf(model_path, backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    if (!w.is_moe) {
        std::fprintf(stderr, "target is not qwen35moe\n");
        return 1;
    }
    if (!stats.matches(w)) {
        std::fprintf(stderr, "stats do not match model dimensions\n");
        return 1;
    }

    uint64_t total_weight_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(w.ctx); t != nullptr;
         t = ggml_get_next_tensor(w.ctx, t)) {
        if (t == w.tok_embd) continue; // CPU-only embedding
        total_weight_bytes += ggml_nbytes(t);
    }

    std::vector<uint64_t> layer_expert_bytes((size_t)w.n_layer, 0);
    uint64_t total_routed_bytes = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const auto & L = w.layers[(size_t)il];
        uint64_t per_expert = 0;
        if (L.ffn_gate_up_exps) {
            per_expert = (uint64_t)L.ffn_gate_up_exps->nb[2] + (uint64_t)L.ffn_down_exps->nb[2];
        } else {
            per_expert = (uint64_t)L.ffn_gate_exps->nb[2] +
                         (uint64_t)L.ffn_up_exps->nb[2] +
                         (uint64_t)L.ffn_down_exps->nb[2];
        }
        layer_expert_bytes[(size_t)il] = per_expert;
        total_routed_bytes += per_expert * (uint64_t)w.n_expert;
    }
    const uint64_t non_expert_bytes = total_weight_bytes - total_routed_bytes;

    Qwen35MoeExpertPlacement placement;
    if (total_hot_budget > 0) {
        if (!Qwen35MoeExpertPlacement::build_from_stats(stats, total_hot_budget, min_hot_per_layer,
                                                        placement, &err)) {
            std::fprintf(stderr, "build_from_stats: %s\n", err.c_str());
            return 1;
        }
    } else {
        if (gpu_vram_gib <= 0.0 || reserve_gib <= 0.0) {
            std::fprintf(stderr, "must provide either --total-hot-budget or (--gpu-vram-gib and --reserve-gib)\n");
            return 2;
        }
        const uint64_t gpu_budget_bytes = gib_to_bytes(gpu_vram_gib - reserve_gib - safety_gib);
        if (gpu_budget_bytes <= non_expert_bytes) {
            std::fprintf(stderr, "no room left for routed experts after reserve\n");
            return 1;
        }
        const uint64_t hot_budget_bytes = gpu_budget_bytes - non_expert_bytes;
        if (!Qwen35MoeExpertPlacement::build_from_stats_with_layer_bytes(
                stats, layer_expert_bytes, hot_budget_bytes, min_hot_per_layer, placement, &err)) {
            std::fprintf(stderr, "build_from_stats_with_layer_bytes: %s\n", err.c_str());
            return 1;
        }
        std::printf("[placement] total_weight=%.2f GiB non_expert=%.2f GiB routed=%.2f GiB hot_budget=%.2f GiB\n",
                    total_weight_bytes / 1024.0 / 1024.0 / 1024.0,
                    non_expert_bytes / 1024.0 / 1024.0 / 1024.0,
                    total_routed_bytes / 1024.0 / 1024.0 / 1024.0,
                    hot_budget_bytes / 1024.0 / 1024.0 / 1024.0);
    }

    if (!placement.save_json(out_path, &err)) {
        std::fprintf(stderr, "save placement: %s\n", err.c_str());
        return 1;
    }

    int total_cold = w.n_layer * w.n_expert - placement.total_hot;
    std::printf("[placement] wrote %s  total_hot=%d total_cold=%d\n",
                out_path, placement.total_hot, total_cold);

    free_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
