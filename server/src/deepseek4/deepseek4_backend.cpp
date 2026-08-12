// DeepSeek4Backend implementation — AR-only decode, chunked prefill.
#include "deepseek4_roctx.h"

#include "deepseek4_backend.h"
#include "deepseek4_budget_hook.h"
#include "deepseek4_internal.h"
#include "common/dynamic_backend.h"
#include "common/peer_access.h"
#include "common/sampler.h"

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "common/gpu_runtime_compat.h"
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <limits>
#include <new>

namespace dflash::common {

namespace {
using Clock = std::chrono::steady_clock;

static double elapsed_s(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static uint64_t elapsed_us(Clock::time_point start, Clock::time_point end) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static bool positive_env_double(const char * name, double fallback,
                                double & out, std::string * err) {
    out = fallback;
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return true;
    char * end = nullptr;
    const double parsed = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || !std::isfinite(parsed) ||
        parsed <= 0.0) {
        if (err) {
            *err = std::string(name) + " must be a finite value greater than zero";
        }
        return false;
    }
    out = parsed;
    return true;
}

static bool env_int_in_range(const char * name, int fallback,
                             int minimum, int maximum,
                             int & out, std::string * err) {
    out = fallback;
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return true;

    int parsed = 0;
    const char * end = raw + std::strlen(raw);
    const auto result = std::from_chars(raw, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end ||
        parsed < minimum || parsed > maximum) {
        if (err) {
            *err = std::string(name) + " must be an integer from " +
                   std::to_string(minimum) + " through " +
                   std::to_string(maximum);
        }
        return false;
    }
    out = parsed;
    return true;
}

static bool configure_dspark_mmvq_defaults(int gpu) {
    if (env_flag_enabled("DFLASH_DS4_Q6_VERIFY")) {
        std::fprintf(stderr,
                     "[deepseek4] q=6 verification is unsupported; use q=5\n");
        return false;
    }
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (!env_flag_enabled("DFLASH_DS4_SPEC")) {
        return true;
    }

    // q=5 verification is an explicit AMD-only path and needs the plain quantized
    // verifier matmuls to stay on MMVQ. The process-wide crossover applies to
    // both owners in the heterogeneous graph, so set it before inspecting the
    // target device (which is gfx1201 in the R9700 + gfx1151 launch).
    if (env_flag_enabled("DFLASH_DS4_Q5_VERIFY")) {
        if (std::getenv("LUCE_MMVQ_MAX_NCOLS") == nullptr &&
            ::setenv("LUCE_MMVQ_MAX_NCOLS", "5", 0) != 0) {
            std::fprintf(stderr,
                         "[deepseek4] failed to set LUCE_MMVQ_MAX_NCOLS=5\n");
            return false;
        }
        if (std::strcmp(std::getenv("LUCE_MMVQ_MAX_NCOLS"), "5") == 0) {
            std::fprintf(stderr,
                         "[deepseek4] AMD DSpark q=5: defaulting "
                         "LUCE_MMVQ_MAX_NCOLS=5\n");
        }

        const char * fp4_x4 = std::getenv("DFLASH_CUDA_MMVQ_FP4_X4");
        if (!fp4_x4 && ::setenv("DFLASH_CUDA_MMVQ_FP4_X4", "1", 0) != 0) {
            std::fprintf(stderr,
                         "[deepseek4] failed to enable ROCmFP4 x4 MMVQ\n");
            return false;
        }
        fp4_x4 = std::getenv("DFLASH_CUDA_MMVQ_FP4_X4");
        // Zero deliberately selects the generic five-column MMVQ fallback.
        if (!fp4_x4 ||
            (std::strcmp(fp4_x4, "0") != 0 &&
             std::strcmp(fp4_x4, "1") != 0)) {
            std::fprintf(stderr,
                         "[deepseek4] DFLASH_CUDA_MMVQ_FP4_X4 must be 0 or 1\n");
            return false;
        }

        cudaDeviceProp prop{};
        if (std::strcmp(fp4_x4, "1") == 0 &&
            std::getenv("DFLASH_CUDA_MMVQ_FP4_Q5_X4_PLUS1") == nullptr &&
            cudaGetDeviceProperties(&prop, gpu) == cudaSuccess &&
            std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0 &&
            ::setenv("DFLASH_CUDA_MMVQ_FP4_Q5_X4_PLUS1", "1", 0) == 0) {
            std::fprintf(stderr,
                         "[deepseek4] gfx1201 DSpark q5: defaulting "
                         "ROCmFP4 x4+1 MMVQ\n");
        }
        return true;
    }

    if (std::getenv("LUCE_MMVQ_MAX_NCOLS") != nullptr) {
        return true;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, gpu) != cudaSuccess ||
        std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0) {
        return true;
    }

    if (::setenv("LUCE_MMVQ_MAX_NCOLS", "4", 0) == 0) {
        std::fprintf(stderr,
                     "[deepseek4] gfx1151 DSpark: defaulting "
                     "LUCE_MMVQ_MAX_NCOLS=4\n");
    }
#else
    (void) gpu;
#endif
    return true;
}

static void configure_gfx1201_hybrid_sub_batch_default(int gpu) {
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (std::getenv("DFLASH_MMQ_SUB_BATCH") != nullptr) {
        return;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, gpu) != cudaSuccess ||
        std::strncmp(prop.gcnArchName, "gfx1201", 7) != 0) {
        return;
    }

    // The generic HIP fallback is q=1 because reduced-stack MMQ is unsafe on
    // older AMD parts.  ROCmFPX MMVQ on gfx1201 is qualified through q=4;
    // using that width removes 75% of hot-owner launches while retaining the
    // stable vector kernel instead of the pathological full-batch MMQ path.
    if (::setenv("DFLASH_MMQ_SUB_BATCH", "4", 0) == 0) {
        std::fprintf(stderr,
                     "[deepseek4] gfx1201 hybrid prefill: defaulting hot "
                     "expert sub-batch to 4\n");
    }
#else
    (void) gpu;
#endif
}

struct Ds4MoeTpConfig {
    bool requested = false;
    bool in_process = false;
    bool backend_valid = true;
    PlacementBackend secondary_backend = PlacementBackend::Auto;
    int secondary_gpu = 0;
    bool all_on_secondary = false;
    bool concentrate_secondary = false;
    bool profile_hot_on_secondary = false;
};

static Ds4MoeTpConfig ds4_moe_tp_config(int local_gpu) {
    Ds4MoeTpConfig result;
    result.requested = env_flag_enabled("DFLASH_DS4_MOE_TP");
    result.in_process = result.requested &&
        env_flag_enabled("DFLASH_DS4_MOE_TP_INPROC");
    result.all_on_secondary = result.requested &&
        env_flag_enabled("DFLASH_DS4_MOE_TP_ALL_COLD");
    result.concentrate_secondary = result.requested &&
        env_flag_enabled("DFLASH_DS4_MOE_TP_CONCENTRATE_COLD");
    result.profile_hot_on_secondary = result.in_process &&
        env_flag_enabled("DFLASH_DS4_MOE_TP_PEER_HOT");

    const char * raw = std::getenv("DFLASH_DS4_MOE_TP_BACKEND");
    if (!raw || !*raw) raw = std::getenv("DFLASH_MOE_TP_BACKEND");
    if (!raw || !*raw) {
#if defined(DFLASH27B_BACKEND_MIXED)
        result.secondary_backend =
            compiled_placement_backend() == PlacementBackend::Cuda
            ? PlacementBackend::Hip : PlacementBackend::Cuda;
#else
        result.secondary_backend = compiled_placement_backend();
#endif
    } else {
        result.backend_valid = parse_placement_backend(
            raw, result.secondary_backend) &&
            result.secondary_backend != PlacementBackend::Auto;
    }

    const char * gpu_raw = std::getenv("DFLASH_DS4_MOE_TP_GPU");
    if (!gpu_raw || !*gpu_raw) {
        gpu_raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU");
    }
    if (gpu_raw && *gpu_raw) {
        result.secondary_gpu = std::max(0, std::atoi(gpu_raw));
    } else if (result.backend_valid &&
               result.secondary_backend != compiled_placement_backend()) {
        // CUDA and HIP have independent device namespaces. The first device
        // in the peer runtime is therefore backend:0 even when the target is
        // also device zero in its own runtime.
        result.secondary_gpu = 0;
    } else {
        result.secondary_gpu = local_gpu == 0 ? 1 : 0;
    }
    return result;
}

static bool ds4_draft_backend(PlacementBackend & out) {
    const char * raw = std::getenv("DFLASH_DS4_DRAFT_BACKEND");
    if (!raw || !*raw) {
        out = compiled_placement_backend();
        return true;
    }
    return parse_placement_backend(raw, out) &&
           out != PlacementBackend::Auto;
}

static double gib(uint64_t bytes) {
    return (double) bytes / 1024.0 / 1024.0 / 1024.0;
}

static void add_step_tel(DeepSeek4StepTelemetry & dst, const DeepSeek4StepTelemetry & src) {
    dst.total_us += src.total_us;
    dst.embed_us += src.embed_us;
    dst.hc_pre_attn_us += src.hc_pre_attn_us;
    dst.hc_pre_build_us += src.hc_pre_build_us;
    dst.hc_pre_input_us += src.hc_pre_input_us;
    dst.hc_pre_compute_us += src.hc_pre_compute_us;
    dst.attn_build_us += src.attn_build_us;
    dst.attn_compute_us += src.attn_compute_us;
    dst.attn_read_us += src.attn_read_us;
    dst.full_graph_build_us += src.full_graph_build_us;
    dst.full_graph_set_us += src.full_graph_set_us;
    dst.full_graph_compute_us += src.full_graph_compute_us;
    dst.full_graph_read_us += src.full_graph_read_us;
    dst.hc_post_attn_us += src.hc_post_attn_us;
    dst.hc_pre_ffn_us += src.hc_pre_ffn_us;
    dst.ffn_build_us += src.ffn_build_us;
    dst.ffn_compute_us += src.ffn_compute_us;
    dst.ffn_read_us += src.ffn_read_us;
    dst.route_build_us += src.route_build_us;
    dst.route_compute_us += src.route_compute_us;
    dst.route_read_us += src.route_read_us;
    dst.route_select_us += src.route_select_us;
    dst.ffn_eval_us += src.ffn_eval_us;
    dst.ffn_hot_us += src.ffn_hot_us;
    dst.ffn_cold_us += src.ffn_cold_us;
    dst.ffn_combine_us += src.ffn_combine_us;
    dst.ffn_partition_us += src.ffn_partition_us;
    dst.ffn_hot_graph_builds += src.ffn_hot_graph_builds;
    dst.ffn_hot_graph_hits += src.ffn_hot_graph_hits;
    dst.ffn_cold_graph_builds += src.ffn_cold_graph_builds;
    dst.ffn_cold_graph_hits += src.ffn_cold_graph_hits;
    dst.hc_post_ffn_us += src.hc_post_ffn_us;
    dst.output_us += src.output_us;
    dst.sample_us += src.sample_us;
    dst.emit_us += src.emit_us;
    dst.hot_selected += src.hot_selected;
    dst.cold_selected += src.cold_selected;
}

static double ms(uint64_t us) {
    return (double)us / 1000.0;
}

static void log_step_tel(const char * phase,
                         int tokens,
                         int steps,
                         double wall_s,
                         const DeepSeek4StepTelemetry & t) {
    const double tok_s = wall_s > 0.0 ? (double)tokens / wall_s : 0.0;
    std::fprintf(stderr,
        "[deepseek4-timing] %s tokens=%d steps=%d wall=%.3fs %.2f tok/s "
        "step=%.1fms embed=%.1fms attn_build=%.1fms attn_compute=%.1fms attn_read=%.1fms "
        "full_build=%.1fms full_set=%.1fms full_compute=%.1fms full_read=%.1fms "
        "ffn_build=%.1fms ffn_compute=%.1fms ffn_read=%.1fms "
        "route_build=%.1fms route_compute=%.1fms route_read=%.1fms route_select=%.1fms "
        "ffn=%.1fms hot=%.1fms cold=%.1fms combine=%.1fms partition=%.1fms "
        "ffn_hot_graph_build=%llu ffn_hot_graph_hit=%llu ffn_cold_graph_build=%llu ffn_cold_graph_hit=%llu "
        "hc_pre=%.1fms hc_pre_build=%.1fms hc_pre_input=%.1fms hc_pre_compute=%.1fms "
        "hc_post=%.1fms output=%.1fms sample=%.1fms emit=%.1fms "
        "hot_sel=%d cold_sel=%d\n",
        phase, tokens, steps, wall_s, tok_s,
        ms(t.total_us), ms(t.embed_us), ms(t.attn_build_us), ms(t.attn_compute_us), ms(t.attn_read_us),
        ms(t.full_graph_build_us), ms(t.full_graph_set_us),
        ms(t.full_graph_compute_us), ms(t.full_graph_read_us),
        ms(t.ffn_build_us), ms(t.ffn_compute_us), ms(t.ffn_read_us),
        ms(t.route_build_us), ms(t.route_compute_us), ms(t.route_read_us), ms(t.route_select_us),
        ms(t.ffn_eval_us), ms(t.ffn_hot_us), ms(t.ffn_cold_us), ms(t.ffn_combine_us),
        ms(t.ffn_partition_us),
        (unsigned long long)t.ffn_hot_graph_builds, (unsigned long long)t.ffn_hot_graph_hits,
        (unsigned long long)t.ffn_cold_graph_builds, (unsigned long long)t.ffn_cold_graph_hits,
        ms(t.hc_pre_attn_us + t.hc_pre_ffn_us),
        ms(t.hc_pre_build_us),
        ms(t.hc_pre_input_us),
        ms(t.hc_pre_compute_us),
        ms(t.hc_post_attn_us + t.hc_post_ffn_us),
        ms(t.output_us), ms(t.sample_us), ms(t.emit_us),
        t.hot_selected, t.cold_selected);
}

static uint64_t layer_expert_bytes(const DeepSeek4Layer & layer, int n_expert) {
    if (n_expert <= 0) return 0;
    uint64_t bytes = 0;
    if (layer.ffn_gate_exps) bytes += ggml_nbytes(layer.ffn_gate_exps) / (uint64_t) n_expert;
    if (layer.ffn_up_exps) bytes += ggml_nbytes(layer.ffn_up_exps) / (uint64_t) n_expert;
    if (layer.ffn_down_exps) bytes += ggml_nbytes(layer.ffn_down_exps) / (uint64_t) n_expert;
    return bytes;
}

static uint64_t layer_shared_expert_bytes(const DeepSeek4Layer & layer) {
    uint64_t bytes = 0;
    if (layer.ffn_gate_shexp) bytes += ggml_nbytes(layer.ffn_gate_shexp);
    if (layer.ffn_up_shexp) bytes += ggml_nbytes(layer.ffn_up_shexp);
    if (layer.ffn_down_shexp) bytes += ggml_nbytes(layer.ffn_down_shexp);
    return bytes;
}

struct Ds4ExpertMemoryInfo {
    std::vector<uint64_t> layer_expert_bytes;
    uint64_t total_expert_bytes = 0;
    uint64_t bytes_per_uniform_round = 0;
    uint64_t hot_bytes = 0;
    uint64_t cold_bytes = 0;
    int total_hot = 0;
    int total_cold = 0;
};

struct Ds4HybridBudgetInfo {
    Ds4ExpertMemoryInfo mem;
    size_t gpu_free = 0;
    size_t gpu_total = 0;
    uint64_t core_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t warm_bytes = 256ULL * 1024 * 1024;
    uint64_t safety_bytes = 512ULL * 1024 * 1024;
    uint64_t expert_budget = 0;
    int max_hot_per_layer = 0;
};

static bool compute_ds4_expert_memory_info(const DeepSeek4Weights & w,
                                           const MoeHybridPlacement * placement,
                                           Ds4ExpertMemoryInfo & out,
                                           std::string * err) {
    out = {};
    out.layer_expert_bytes.assign((size_t) w.n_layer, 0);
    for (int il = 0; il < w.n_layer; ++il) {
        const uint64_t bytes = layer_expert_bytes(w.layers[(size_t) il], w.n_expert);
        out.layer_expert_bytes[(size_t) il] = bytes;
        out.total_expert_bytes += bytes * (uint64_t) w.n_expert;
        out.bytes_per_uniform_round += bytes;
    }
    if (out.bytes_per_uniform_round == 0) {
        if (err) *err = "expert tensor metadata missing after partial load";
        return false;
    }
    if (!placement) return true;
    if (!placement->matches(w.n_layer, w.n_expert, w.n_expert_used)) {
        if (err) *err = "placement does not match DS4 dimensions";
        return false;
    }
    out.total_hot = placement->total_hot;
    out.total_cold = w.n_layer * w.n_expert - placement->total_hot;
    for (int il = 0; il < w.n_layer; ++il) {
        const uint64_t layer_bytes = out.layer_expert_bytes[(size_t) il];
        const uint64_t hot_count = (uint64_t) placement->hot_counts[(size_t) il];
        out.hot_bytes += layer_bytes * hot_count;
        out.cold_bytes += layer_bytes * ((uint64_t) w.n_expert - hot_count);
    }
    return true;
}

static void log_ds4_expert_memory_info(const char * tag,
                                       const Ds4ExpertMemoryInfo & info,
                                       int n_layer) {
    (void) n_layer;
    std::fprintf(stderr,
                 "[deepseek4] %s expert_memory: total=%.2f GiB uniform_round=%.2f MiB hot=%d %.2f GiB cold=%d %.2f GiB\n",
                 tag,
                 gib(info.total_expert_bytes),
                 (double) info.bytes_per_uniform_round / 1024.0 / 1024.0,
                 info.total_hot,
                 gib(info.hot_bytes),
                 info.total_cold,
                 gib(info.cold_bytes));
}

static uint64_t estimate_ds4_cache_bytes(const DeepSeek4Weights & w, int max_ctx) {
    size_t total_bytes = 0;
    const size_t head_dim = (size_t) w.head_dim;
    const size_t swa_size = (size_t) w.n_swa;

    for (int il = 0; il < w.n_layer; ++il) {
        total_bytes += swa_size * head_dim * sizeof(uint16_t);
        const uint32_t ratio = w.compress_ratios[(size_t) il];
        if (ratio == 0) continue;

        const size_t comp_cap = (size_t) (max_ctx / (int) ratio) + 16;
        total_bytes += comp_cap * head_dim * sizeof(uint16_t);

        const size_t window = (ratio == 4) ? 8 : ratio;
        total_bytes += window * head_dim * sizeof(float) * 2;

        if (ratio == 4) {
            const size_t index_comp_width = (size_t) w.n_indexer_head * (size_t) w.n_indexer_head_dim;
            total_bytes += comp_cap * index_comp_width * sizeof(uint16_t);
            total_bytes += window * index_comp_width * sizeof(float) * 2;
        }
    }

    total_bytes += (size_t) w.n_hc * (size_t) w.n_embd * sizeof(float);
    return total_bytes;
}

static void fill_prefix_hot_placement(const DeepSeek4Weights & w,
                                      int hot_per_layer,
                                      MoeHybridPlacement & out) {
    out = {};
    out.n_layer = w.n_layer;
    out.n_expert = w.n_expert;
    out.n_expert_used = w.n_expert_used;
    out.hot_counts.assign((size_t) w.n_layer, hot_per_layer);
    out.hot_expert_ids.resize((size_t) w.n_layer);
    out.total_hot = hot_per_layer * w.n_layer;
    for (int il = 0; il < w.n_layer; ++il) {
        auto & ids = out.hot_expert_ids[(size_t) il];
        ids.reserve((size_t) hot_per_layer);
        for (int ie = 0; ie < hot_per_layer; ++ie) {
            ids.push_back((int32_t) ie);
        }
    }
}

// Cross-runtime joins are much more expensive than native peer handoffs. Keep
// approximately the same expert residency as the uniform placement, but
// concentrate the cold owner into complete layers. A partial cold layer costs
// another cross-runtime join and some CUDA prefill paths require a complete
// expert stack, so retain the small remainder on the target backend.
static int fill_concentrated_cold_placement(const DeepSeek4Weights & w,
                                            int hot_per_layer,
                                            MoeHybridPlacement & out) {
    out = {};
    out.n_layer = w.n_layer;
    out.n_expert = w.n_expert;
    out.n_expert_used = w.n_expert_used;
    out.hot_counts.assign((size_t) w.n_layer, w.n_expert);
    out.hot_expert_ids.resize((size_t) w.n_layer);

    const int requested_cold =
        w.n_layer * std::max(0, w.n_expert - hot_per_layer);
    int cold_remaining = w.n_expert > 0
        ? requested_cold / w.n_expert * w.n_expert : 0;
    const int retained_local = requested_cold - cold_remaining;
    for (int il = w.n_layer - 1; il >= 0; --il) {
        const int cold = std::min(w.n_expert, cold_remaining);
        const int hot = w.n_expert - cold;
        out.hot_counts[(size_t) il] = hot;
        auto & ids = out.hot_expert_ids[(size_t) il];
        ids.reserve((size_t) hot);
        for (int ie = 0; ie < hot; ++ie) {
            ids.push_back((int32_t) ie);
        }
        out.total_hot += hot;
        cold_remaining -= cold;
    }
    return retained_local;
}

static bool fill_profiled_hot_placement(const DeepSeek4Weights & w,
                                        int hot_per_layer,
                                        const char * profile_path,
                                        bool profile_hot_on_secondary,
                                        MoeHybridPlacement & out,
                                        std::string * err) {
    MoeHybridRoutingStats stats;
    if (!MoeHybridRoutingStats::load_csv(profile_path, stats, err)) {
        return false;
    }
    if (stats.n_layer != w.n_layer || stats.n_expert != w.n_expert) {
        if (err) {
            *err = "routing profile shape does not match DeepSeek V4 target";
        }
        return false;
    }

    out = {};
    out.n_layer = w.n_layer;
    out.n_expert = w.n_expert;
    out.n_expert_used = w.n_expert_used;
    out.hot_counts.assign((size_t)w.n_layer, hot_per_layer);
    out.hot_expert_ids.resize((size_t)w.n_layer);
    out.total_hot = hot_per_layer * w.n_layer;
    for (int il = 0; il < w.n_layer; ++il) {
        auto & ids = out.hot_expert_ids[(size_t)il];
        if (!profile_hot_on_secondary) {
            std::vector<int> ranked = stats.hot_experts(il, hot_per_layer);
            ids.assign(ranked.begin(), ranked.end());
            continue;
        }

        // `hot` is the primary-backend side of MoeHybridPlacement.  On a
        // memory-rich iGPU paired with a smaller, faster dGPU, filling that
        // primary side with the most frequently routed experts starves the
        // dGPU of useful work. Reserve the peer-sized complement for the
        // hottest experts and keep every other expert on the primary. This
        // changes ownership only; route order and reduction semantics stay
        // unchanged.
        const int peer_count = w.n_expert - hot_per_layer;
        const std::vector<int> ranked_peer =
            stats.hot_experts(il, peer_count);
        std::vector<uint8_t> on_peer((size_t)w.n_expert, 0);
        for (int expert : ranked_peer) {
            if (expert >= 0 && expert < w.n_expert) {
                on_peer[(size_t)expert] = 1;
            }
        }
        ids.reserve((size_t)hot_per_layer);
        for (int expert = 0; expert < w.n_expert; ++expert) {
            if (!on_peer[(size_t)expert]) {
                ids.push_back((int32_t)expert);
            }
        }
        if ((int)ids.size() != hot_per_layer) {
            if (err) {
                *err = "routing profile did not yield a complete expert ranking";
            }
            return false;
        }
    }
    return true;
}

// Assign the same total number of resident experts as the uniform placement,
// but distribute those slots across layers to minimize the predicted owner
// critical path.  Uniform expert counts are a poor fit for heterogeneous EP:
// routing skew varies substantially by layer, while every layer joins on the
// slower of its primary/shared and secondary expert branches.
//
// The cost model intentionally uses measured bandwidth rather than advertised
// peak bandwidth.  It is only an allocation objective; actual placement still
// uses authoritative router statistics and evaluates every selected expert.
static bool compute_ds4_hybrid_budget_info(const DeepSeek4Weights & w,
                                           ggml_backend_t backend,
                                           int max_ctx,
                                           Ds4HybridBudgetInfo & out,
                                           std::string * err) {
    out = {};
    if (!backend || !ggml_backend_get_device(backend)) {
        if (err) *err = "target backend has no device";
        return false;
    }
    ggml_backend_dev_memory(
        ggml_backend_get_device(backend), &out.gpu_free, &out.gpu_total);
    if (out.gpu_total == 0) {
        if (err) *err = "could not query GPU memory";
        return false;
    }

    if (!compute_ds4_expert_memory_info(w, nullptr, out.mem, err)) {
        return false;
    }

    out.core_bytes = moe_hybrid_core_bytes_from_memory(
        "deepseek4", out.gpu_free, out.gpu_total);
    out.kv_bytes = estimate_ds4_cache_bytes(w, max_ctx);

    if (out.gpu_total > out.core_bytes + out.kv_bytes + out.warm_bytes + out.safety_bytes) {
        out.expert_budget = out.gpu_total - out.core_bytes - out.kv_bytes - out.warm_bytes - out.safety_bytes;
    }
    if (out.expert_budget > out.mem.total_expert_bytes) {
        out.expert_budget = out.mem.total_expert_bytes;
    }
    if (const char * cap_env = std::getenv("DFLASH_EXPERT_BUDGET_MB")) {
        const uint64_t cap_bytes = (uint64_t) std::max(0, std::atoi(cap_env)) * 1024ULL * 1024ULL;
        if (cap_bytes > 0 && cap_bytes < out.expert_budget) {
            out.expert_budget = cap_bytes;
        }
    }
    if (out.expert_budget == 0) {
        if (err) *err = "no VRAM budget available for DS4 experts";
        return false;
    }

    out.max_hot_per_layer = std::min(w.n_expert, (int) (out.expert_budget / out.mem.bytes_per_uniform_round));
    if (out.max_hot_per_layer <= 0) {
        if (err) *err = "expert budget is smaller than one uniform expert round";
        return false;
    }
    return true;
}

static MoeHybridConfig make_ds4_parent_worker_cfg(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    cfg.swiglu_clamp = w.swiglu_clamp_exp;
    cfg.materialize_cold_experts = false;
    return cfg;
}

static MoeHybridConfig make_ds4_parent_cpu_tail_cfg(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg = make_ds4_parent_worker_cfg(w);
    cfg.materialize_hot_experts = false;
    cfg.materialize_cold_experts = true;
    cfg.cold_expert_backend = MoeHybridColdBackend::Cpu;
    return cfg;
}

static MoeLayerDesc make_ds4_expert_layer_desc(const DeepSeek4Layer & layer) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps = layer.ffn_gate_exps;
    desc.ffn_up_exps = layer.ffn_up_exps;
    desc.ffn_down_exps = layer.ffn_down_exps;
    desc.ffn_gate_shexp = layer.ffn_gate_shexp;
    desc.ffn_up_shexp = layer.ffn_up_shexp;
    desc.ffn_down_shexp = layer.ffn_down_shexp;
    return desc;
}

}  // namespace

DeepSeek4Backend::DeepSeek4Backend(const DeepSeek4BackendConfig & cfg)
    : cfg_(cfg) {}

DeepSeek4Backend::~DeepSeek4Backend() {
    shutdown();
}

bool DeepSeek4Backend::requires_monolithic_model() const {
    return cfg_.fused_decode ||
           prefill_attention_mode_is_approximate(cfg_.prefill_mode);
}

bool DeepSeek4Backend::validate_prefill_mode() const {
    if (cfg_.prefill_mode == PrefillAttentionMode::Exact) {
        return true;
    }
    const PlacementBackend target_backend =
        cfg_.device.backend == PlacementBackend::Auto
            ? compiled_placement_backend()
            : cfg_.device.backend;
    if (target_backend != PlacementBackend::Hip ||
        cfg_.device.is_layer_split()) {
        std::fprintf(stderr,
            "[deepseek4] %s prefill requires a single HIP target\n",
            prefill_attention_mode_name(cfg_.prefill_mode));
        return false;
    }
    if (w_.moe_hybrid || moe_hybrid_) {
        std::fprintf(stderr,
            "[deepseek4] %s prefill using heterogeneous layer-major experts\n",
            prefill_attention_mode_name(cfg_.prefill_mode));
    }
    return true;
}

bool DeepSeek4Backend::load_model() {
    const PlacementBackend target_backend =
        cfg_.device.backend == PlacementBackend::Auto
            ? compiled_placement_backend()
            : cfg_.device.backend;

    // Fused decode and layer-major prefill normally require monolithic expert
    // residency. Heterogeneous TP is the exception: its fused graph owns the
    // routed experts across two local GPU backends, so forcing a full load would
    // disable the requested split before the TP runtime can initialize.
    const bool force_full = env_flag_enabled("DFLASH_DS4_FORCE_FULL_LOAD");
    const bool heterogeneous_tp = env_flag_enabled("DFLASH_DS4_MOE_TP");
    const bool need_monolithic =
        requires_monolithic_model() && !heterogeneous_tp;
    if (target_backend == PlacementBackend::Hip &&
        (force_full || need_monolithic)) {
        std::fprintf(stderr,
                     "[deepseek4] monolithic execution requested "
                     "(forced=%s, fused_decode=%s, prefill=%s)\n",
                     force_full ? "yes" : "no",
                     cfg_.fused_decode ? "on" : "off",
                     prefill_attention_mode_name(cfg_.prefill_mode));
        if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
            if (prefill_attention_mode_is_approximate(cfg_.prefill_mode)) {
                std::fprintf(stderr,
                    "[deepseek4] monolithic HIP load required for %s prefill\n",
                    prefill_attention_mode_name(cfg_.prefill_mode));
                return false;
            }
            std::fprintf(stderr,
                         "[deepseek4] explicit HIP full-model load failed: %s\n",
                         cfg_.model_path);
            return false;
        }
    } else if (target_backend == PlacementBackend::Hip || heterogeneous_tp) {
        std::fprintf(stderr,
                     "[deepseek4] heterogeneous target detected; using hybrid expert load path\n");
        if (!init_hybrid_model()) {
            std::fprintf(stderr, "[deepseek4] hybrid mode failed: %s\n", cfg_.model_path);
            return false;
        }
    } else if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[deepseek4] full model load failed, trying hybrid mode...\n");
        if (!init_hybrid_model()) {
            std::fprintf(stderr, "[deepseek4] hybrid mode also failed: %s\n", cfg_.model_path);
            return false;
        }
    }

    if (cfg_.expert_top_k < 0 || cfg_.expert_top_k > w_.n_expert_used) {
        std::fprintf(stderr,
                     "[deepseek4] expert top-k must be in [0,%d], got %d\n",
                     w_.n_expert_used, cfg_.expert_top_k);
        return false;
    }
    w_.routed_expert_top_k = cfg_.expert_top_k;
    w_.fused_decode = cfg_.fused_decode && !moe_hybrid_;
    if (cfg_.fused_decode && moe_hybrid_) {
        std::fprintf(stderr,
                     "[deepseek4] fused decode unavailable with hybrid expert placement; "
                     "using layered decode\n");
    }
    return true;
}

bool DeepSeek4Backend::load_spec_drafter() {
    if (spec_draft_path_.empty()) return true;
    if (parked_) {
        std::fprintf(stderr,
                     "[deepseek4] cannot load DSpark drafter while target is parked\n");
        return false;
    }

    ggml_backend_t draft_backend = backend_;
    int draft_gpu = cfg_.device.gpu;
    if (const char * gpu = std::getenv("DFLASH_DS4_DRAFT_GPU")) {
        draft_gpu = std::max(0, std::atoi(gpu));
    }
    const bool separate_draft_stream =
        env_flag_enabled("DFLASH_DS4_DRAFT_SEPARATE_STREAM");
    PlacementBackend draft_kind = PlacementBackend::Auto;
    if (!ds4_draft_backend(draft_kind)) {
        std::fprintf(stderr,
                     "[deepseek4] invalid DFLASH_DS4_DRAFT_BACKEND; "
                     "expected cuda or hip\n");
        return false;
    }
    const PlacementBackend target_kind = placement_backend_of(backend_);
    if (draft_kind != target_kind || draft_gpu != cfg_.device.gpu ||
        separate_draft_stream) {
        std::string backend_error;
        spec_backend_ = init_placement_backend(
            draft_kind, draft_gpu, &backend_error);
        if (!spec_backend_) {
            std::fprintf(stderr,
                         "[deepseek4] failed to initialize DSpark %s:%d: %s\n",
                         placement_backend_name(draft_kind), draft_gpu,
                         backend_error.c_str());
            return false;
        }
        draft_backend = spec_backend_;
        const bool low_priority = separate_draft_stream &&
            env_flag_enabled("DFLASH_DS4_DRAFT_LOW_PRIORITY");
        const bool priority_configured = low_priority &&
            backend_pair_capabilities(backend_, spec_backend_).same_runtime &&
            ggml_backend_cuda_set_low_priority_stream(spec_backend_);
        std::fprintf(stderr,
                     "[deepseek4] DSpark backend=%s:%d target=%s:%d "
                     "separate_stream=%d low_priority=%d\n",
                     placement_backend_name(draft_kind), draft_gpu,
                     placement_backend_name(target_kind), cfg_.device.gpu,
                     (int) separate_draft_stream,
                     (int) priority_configured);
    }

    auto drafter = std::make_unique<DSparkDrafter>();
    if (!load_deepseek4_dspark_drafter(
            spec_draft_path_, draft_backend, *drafter)) {
        std::fprintf(stderr, "[deepseek4] DSpark drafter load FAILED: %s\n",
                     deepseek4_dspark_last_error());
        if (spec_backend_) {
            ggml_backend_free(spec_backend_);
            spec_backend_ = nullptr;
        }
        return false;
    }

    if (spec_backend_ && !clone_deepseek4_dspark_heads(*drafter, backend_)) {
        std::fprintf(stderr,
                     "[deepseek4] failed to clone DSpark sampling heads to target GPU\n");
        free_deepseek4_dspark_drafter(*drafter);
        ggml_backend_free(spec_backend_);
        spec_backend_ = nullptr;
        return false;
    }

    const DSparkDrafter & d = *drafter;
    bool compatible = d.core.n_embd == w_.n_embd &&
                      d.core.n_vocab == w_.n_vocab &&
                      d.vocab_size == w_.n_vocab &&
                      d.mask_token_id >= 0 && d.mask_token_id < w_.n_vocab &&
                      (int) d.capture_layer_ids.size() == d.n_target_layers;
    for (int layer : d.capture_layer_ids) {
        compatible = compatible && layer >= 0 && layer < w_.n_layer;
    }
    if (!compatible) {
        std::fprintf(stderr,
                     "[deepseek4] DSpark drafter is incompatible with target "
                     "(target embd/vocab/layers=%d/%d/%d, draft=%d/%d)\n",
                     w_.n_embd, w_.n_vocab, w_.n_layer,
                     d.core.n_embd, d.vocab_size);
        free_deepseek4_dspark_drafter(*drafter);
        if (spec_backend_) {
            ggml_backend_free(spec_backend_);
            spec_backend_ = nullptr;
        }
        return false;
    }

    spec_drafter_ = std::move(drafter);
    spec_enabled_ = true;
    spec_drafter_parked_ = false;
    std::fprintf(stderr, "[deepseek4] DSpark spec-decode ENABLED (drafter=%s)\n",
                 spec_draft_path_.c_str());
    return true;
}

void DeepSeek4Backend::release_spec_drafter(bool mark_parked) {
    if (spec_drafter_) {
        free_deepseek4_dspark_drafter(*spec_drafter_);
    }
    spec_drafter_.reset();
    if (spec_backend_) {
        ggml_backend_free(spec_backend_);
        spec_backend_ = nullptr;
    }
    spec_enabled_ = false;
    spec_feat_window_.clear();
    spec_drafter_parked_ = mark_parked && !spec_draft_path_.empty();
}

void DeepSeek4Backend::keep_spec_feature_tail(
        std::vector<float> & features, size_t max_rows) const {
    if (!spec_drafter_) return;
    const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
    if (feat_row <= 0 || features.size() % (size_t) feat_row != 0) {
        features.clear();
        return;
    }
    const size_t rows = features.size() / (size_t) feat_row;
    const size_t keep_rows = std::min(rows, max_rows);
    if (rows == keep_rows) return;
    const size_t keep_floats = keep_rows * (size_t) feat_row;
    const size_t drop_floats = features.size() - keep_floats;
    if (keep_floats > 0) {
        std::memmove(features.data(), features.data() + drop_floats,
                     keep_floats * sizeof(float));
    }
    features.resize(keep_floats);
}

int DeepSeek4Backend::capture_safe_prefill_tokens(
        int token_offset,
        int requested_tokens,
        int final_capture_from,
        bool batch_final_capture,
        bool snapshot_pending,
        int snapshot_capture_from,
        int snapshot_capture_to) {
    if (requested_tokens <= 0) return 0;

    int safe_tokens = requested_tokens;
    const auto split_at = [&](int boundary) {
        const int distance = boundary - token_offset;
        if (distance > 0 && distance < safe_tokens) {
            safe_tokens = distance;
        }
    };

    if (!batch_final_capture) {
        split_at(final_capture_from);
    }
    if (snapshot_pending) {
        split_at(snapshot_capture_from);
        split_at(snapshot_capture_to);
    }
    return safe_tokens;
}

bool DeepSeek4Backend::supports_batched_spec_feature_capture(
        bool hybrid,
        PrefillAttentionMode mode,
        int n_tokens) {
    if (mode == PrefillAttentionMode::Exact || n_tokens <= 4 ||
        n_tokens > DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS) {
        return false;
    }
    // The monolithic layer-major path reads only the requested token range.
    // Sparse heterogeneous prefill returns every requested capture row; the
    // caller then retains the final/snapshot window. Other hybrid modes are
    // tokenwise and must still split at capture boundaries.
    return !hybrid || mode == PrefillAttentionMode::Sparse;
}

bool DeepSeek4Backend::init() {
    // The shared MMVQ/MMQ crossover defaults to q=3 for NVIDIA. On gfx1151,
    // DSpark q=4 is faster through MMVQ. Keep AR and other devices unchanged,
    // and preserve LUCE_MMVQ_MAX_NCOLS as an explicit override.
    if (!configure_dspark_mmvq_defaults(cfg_.device.gpu)) {
        return false;
    }
    configure_gfx1201_hybrid_sub_batch_default(cfg_.device.gpu);

    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[deepseek4] failed to create CUDA backend (gpu=%d)\n",
                     cfg_.device.gpu);
        return false;
    }

    snap_backend_ = ggml_backend_init_by_name("cpu", nullptr);

    if (!load_model()) {
        return false;
    }
    if (!validate_prefill_mode()) {
        return false;
    }
    if (prefill_attention_mode_is_approximate(cfg_.prefill_mode)) {
        std::fprintf(stderr,
            "[deepseek4] warning: %s prefill is approximate and may change "
            "generated tokens; use --ds4-prefill exact for reference output\n",
            prefill_attention_mode_name(cfg_.prefill_mode));
    }

    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    if (!create_deepseek4_cache(backend_, w_, max_ctx, cache_)) {
        std::fprintf(stderr, "[deepseek4] failed to allocate KV cache (ctx=%d)\n", max_ctx);
        return false;
    }
    cache_.prefill_mode = cfg_.prefill_mode;

    if (env_flag_enabled("DFLASH_DS4_MOE_TP") && !init_moe_tensor_parallel()) {
        return false;
    }

    if (const char * stats_path = std::getenv("DFLASH_DS4_ROUTING_STATS_OUT")) {
        if (*stats_path) {
            routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
            if (!routing_stats_->init(w_.n_layer, w_.n_expert, w_.n_expert_used)) {
                std::fprintf(stderr, "[deepseek4] failed to initialize routing stats\n");
                return false;
            }
            routing_stats_out_path_ = stats_path;
            std::fprintf(stderr, "[deepseek4] routing stats enabled output=%s\n",
                         routing_stats_out_path_.c_str());
        }
    }
    if (env_flag_enabled("DFLASH_DS4_TP_ROUTE_STATS") && !routing_stats_) {
        routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
        if (!routing_stats_->init(w_.n_layer, w_.n_expert,
                                  w_.n_expert_used)) {
            std::fprintf(stderr,
                         "[deepseek4] failed to initialize TP routing stats\n");
            return false;
        }
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] in-memory routing stats enabled\n");
    }
    const int active_experts =
        w_.routed_expert_top_k > 0 ? w_.routed_expert_top_k : w_.n_expert_used;
    std::fprintf(stderr,
                 "[deepseek4] initialized: %d layers, ctx=%d, %d experts "
                 "(%d/%d routed), fused_decode=%s, prefill=%s%s\n",
                 w_.n_layer, max_ctx, w_.n_expert, active_experts, w_.n_expert_used,
                 w_.fused_decode ? "on" : "off",
                 prefill_attention_mode_name(cfg_.prefill_mode),
                 moe_hybrid_ ? " [hybrid]" : "");

    if (env_flag_enabled("DFLASH_DS4_SPEC")) {
        const char * dp = std::getenv("DFLASH_DS4_DRAFT");
        if (dp && *dp) {
            spec_draft_path_ = dp;
            if (!load_spec_drafter()) {
                std::fprintf(stderr,
                             "[deepseek4] DSpark drafter load failed; "
                             "continuing with autoregressive decode\n");
            }
        } else {
            std::fprintf(stderr, "[deepseek4] DFLASH_DS4_SPEC set but DFLASH_DS4_DRAFT gguf missing\n");
        }
    }
    return true;
}

bool DeepSeek4Backend::init_moe_tensor_parallel() {
    if (!moe_hybrid_) {
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] requires a partial local expert placement\n");
        return false;
    }

    const Ds4MoeTpConfig tp = ds4_moe_tp_config(cfg_.device.gpu);
    if (tp.in_process) {
        if (!expert_backend_ || !moe_hybrid_->materialized_cold_experts ||
            moe_hybrid_->cold_backend != expert_backend_) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] in-process expert backend is not ready\n");
            return false;
        }
        expert_runtime_.reset();
        const PlacementBackend local_kind =
            cfg_.device.backend == PlacementBackend::Auto
                ? compiled_placement_backend() : cfg_.device.backend;
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] enabled mode=in-process local=%s:%d "
                     "secondary=%s:%d primary_experts=%d "
                     "secondary_experts=%d\n",
                     placement_backend_name(local_kind), cfg_.device.gpu,
                     placement_backend_name(tp.secondary_backend),
                     tp.secondary_gpu,
                     moe_placement_.total_hot,
                     w_.n_layer * w_.n_expert - moe_placement_.total_hot);
        return true;
    }

    std::vector<MoeLayerDesc> layer_descs((size_t)w_.n_layer);
    for (int il = 0; il < w_.n_layer; ++il) {
        layer_descs[(size_t)il] = make_ds4_expert_layer_desc(w_.layers[(size_t)il]);
    }

    MoeExpertComputeRuntimeConfig runtime_cfg;
    runtime_cfg.target_path = cfg_.model_path;
    runtime_cfg.n_layer = w_.n_layer;
    runtime_cfg.n_expert = w_.n_expert;
    runtime_cfg.n_expert_used = w_.n_expert_used;
    runtime_cfg.n_embd = w_.n_embd;
    runtime_cfg.n_ff_exp = w_.n_ff_exp;
    runtime_cfg.enabled = true;
    runtime_cfg.require_remote = true;
    runtime_cfg.log_prefix = "[deepseek4-moe-tp]";

    std::string err;
    if (!ensure_moe_expert_compute_runtime(expert_runtime_, runtime_cfg,
                                           *moe_hybrid_, layer_descs, &err)) {
        std::fprintf(stderr, "[deepseek4-moe-tp] initialization failed: %s\n",
                     err.c_str());
        return false;
    }

    std::fprintf(stderr,
                 "[deepseek4-moe-tp] enabled local_experts=%d remote_experts=%d\n",
                 moe_placement_.total_hot,
                 w_.n_layer * w_.n_expert - moe_placement_.total_hot);
    return true;
}

bool DeepSeek4Backend::compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                                       int max_ctx,
                                                       MoeHybridPlacement & out,
                                                       MoeHybridPlacement * decode_out,
                                                       std::string * err) const {
    if (decode_out) *decode_out = {};
    Ds4HybridBudgetInfo budget;
    if (!compute_ds4_hybrid_budget_info(w, backend_, max_ctx, budget, err)) {
        return false;
    }

    const Ds4MoeTpConfig tp = ds4_moe_tp_config(cfg_.device.gpu);
    int hot_per_layer = tp.all_on_secondary ? 0 : budget.max_hot_per_layer;
    if (tp.all_on_secondary) {
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] all routed experts assigned to the "
                     "secondary backend\n");
    }
    const bool concentrate_requested = tp.concentrate_secondary;
    bool concentrated = false;
    int retained_local = 0;
    const char * profile_path = std::getenv("DFLASH_DS4_HOTNESS_CSV");
    const char * decode_profile_path =
        std::getenv("DFLASH_DS4_DECODE_HOTNESS_CSV");
    const bool phase_aware_placement = decode_profile_path &&
        *decode_profile_path;
    const bool critical_path_placement =
        !tp.all_on_secondary && !concentrate_requested &&
        env_flag_enabled("DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT");
    if (critical_path_placement && tp.profile_hot_on_secondary) {
        if (err) {
            *err = "critical-path placement is incompatible with "
                   "DFLASH_DS4_MOE_TP_PEER_HOT";
        }
        return false;
    }
    const int requested_cold =
        w.n_layer * std::max(0, w.n_expert - hot_per_layer);
    if (concentrate_requested && requested_cold >= w.n_expert) {
        retained_local =
            fill_concentrated_cold_placement(w, hot_per_layer, out);
        concentrated = true;
    } else if (concentrate_requested) {
        std::fprintf(stderr,
                     "[deepseek4] concentrated secondary placement needs at least "
                     "one complete layer; using uniform placement\n");
        fill_prefix_hot_placement(w, hot_per_layer, out);
    } else if (critical_path_placement) {
        if (!profile_path || !*profile_path) {
            if (err) {
                *err = "critical-path placement requires DFLASH_DS4_HOTNESS_CSV";
            }
            return false;
        }
        const char * balance_profile_path = phase_aware_placement
            ? decode_profile_path : profile_path;
        MoeHybridRoutingStats stats;
        if (!MoeHybridRoutingStats::load_csv(
                balance_profile_path, stats, err)) {
            return false;
        }
        if (stats.n_layer != w.n_layer || stats.n_expert != w.n_expert) {
            if (err) {
                *err = "routing profile shape does not match DeepSeek V4 target";
            }
            return false;
        }

        int active_experts = cfg_.expert_top_k > 0
            ? cfg_.expert_top_k : w.n_expert_used;
        if (!env_int_in_range(
                "DFLASH_DS4_TOPK", active_experts,
                1, w.n_expert_used, active_experts, err)) {
            return false;
        }

        double main_to_peer_rate = 3.4;
        if (!positive_env_double(
                "DFLASH_DS4_TP_MAIN_TO_PEER_RATE", 3.4,
                main_to_peer_rate, err)) {
            return false;
        }
        MoeHybridCriticalPathConfig balance_cfg;
        balance_cfg.active_experts = active_experts;
        balance_cfg.main_to_peer_rate = main_to_peer_rate;
        if (!env_int_in_range(
                "DFLASH_DS4_TP_BALANCE_MIN_HOT", 0,
                0, w.n_expert, balance_cfg.min_hot_per_layer, err)) {
            return false;
        }

        std::vector<uint64_t> main_fixed_bytes((size_t) w.n_layer, 0);
        for (int il = 0; il < w.n_layer; ++il) {
            main_fixed_bytes[(size_t) il] =
                layer_shared_expert_bytes(w.layers[(size_t) il]);
        }
        if (!MoeHybridPlacement::build_critical_path_balanced_from_stats(
                stats, budget.mem.layer_expert_bytes, main_fixed_bytes,
                budget.expert_budget, balance_cfg, out, err)) {
            return false;
        }

        if (phase_aware_placement) {
            if (!decode_out) {
                if (err) *err = "phase-aware placement requires decode output";
                return false;
            }
            *decode_out = out;
            MoeHybridRoutingStats residency_stats;
            if (!MoeHybridRoutingStats::load_csv(
                    profile_path, residency_stats, err)) {
                return false;
            }
            if (residency_stats.n_layer != w.n_layer ||
                residency_stats.n_expert != w.n_expert ||
                residency_stats.n_expert_used != w.n_expert_used) {
                if (err) {
                    *err = "residency routing profile shape does not match "
                           "DeepSeek V4 target";
                }
                return false;
            }
            if (!MoeHybridPlacement::expand_from_stats_with_layer_bytes(
                    residency_stats, budget.mem.layer_expert_bytes,
                    budget.expert_budget, out, err)) {
                return false;
            }
            std::fprintf(stderr,
                         "[deepseek4] hybrid phase-aware placement: "
                         "decode_profile=%s resident_profile=%s "
                         "decode=%d resident=%d\n",
                         decode_profile_path, profile_path,
                         decode_out->total_hot, out.total_hot);
        }

        const auto [min_hot, max_hot] = std::minmax_element(
            out.hot_counts.begin(), out.hot_counts.end());
        const double mean_hot = out.hot_counts.empty() ? 0.0
            : (double) out.total_hot / (double) out.hot_counts.size();
        std::fprintf(stderr,
                     "[deepseek4] hybrid critical-path placement: "
                     "profile=%s active=%d main/peer=%.3f "
                     "hot/layer=%.1f [%d,%d]\n",
                     balance_profile_path, active_experts, main_to_peer_rate,
                     mean_hot,
                     min_hot != out.hot_counts.end() ? *min_hot : 0,
                     max_hot != out.hot_counts.end() ? *max_hot : 0);
        std::fprintf(stderr,
                     "[deepseek4] hybrid critical-path hot counts:");
        for (int count : out.hot_counts) {
            std::fprintf(stderr, " %d", count);
        }
        std::fprintf(stderr, "\n");
    } else if (profile_path) {
        if (*profile_path) {
            const bool profile_hot_on_secondary =
                tp.in_process && tp.profile_hot_on_secondary;
            if (!fill_profiled_hot_placement(
                    w, hot_per_layer, profile_path,
                    profile_hot_on_secondary,
                    out, err)) {
                return false;
            }
            std::fprintf(stderr,
                         "[deepseek4] hybrid placement profile=%s%s\n",
                         profile_path,
                         profile_hot_on_secondary
                             ? " profile-hot-owner=secondary" : "");
        } else {
            fill_prefix_hot_placement(w, hot_per_layer, out);
        }
    } else {
        fill_prefix_hot_placement(w, hot_per_layer, out);
    }

    Ds4ExpertMemoryInfo placed_mem;
    if (!compute_ds4_expert_memory_info(w, &out, placed_mem, err)) {
        return false;
    }
    if (concentrated && placed_mem.hot_bytes > budget.expert_budget) {
        std::fprintf(stderr,
                     "[deepseek4] concentrated secondary placement exceeds the "
                     "primary expert budget; using uniform placement\n");
        fill_prefix_hot_placement(w, hot_per_layer, out);
        if (!compute_ds4_expert_memory_info(w, &out, placed_mem, err)) {
            return false;
        }
        concentrated = false;
    }
    if (concentrated) {
        const int cold_layers =
            w.n_expert > 0
                ? (w.n_layer * w.n_expert - out.total_hot) / w.n_expert : 0;
        std::fprintf(stderr,
                     "[deepseek4] concentrated secondary placement: "
                     "cross-owner layers=%d primary_experts=%d "
                     "secondary_experts=%d retained_primary=%d\n",
                     cold_layers, out.total_hot,
                     w.n_layer * w.n_expert - out.total_hot,
                     retained_local);
    }

    const std::string hot_label = critical_path_placement
        ? "balanced" : std::to_string(hot_per_layer);
    std::fprintf(stderr,
                 "[deepseek4] hybrid placement: gpu_total=%.2f GiB gpu_free=%.2f GiB core=%.2f GiB kv=%.2f GiB warm=%.2f GiB safety=%.2f GiB expert_budget=%.2f GiB hot/layer=%s\n",
                 gib((uint64_t) budget.gpu_total),
                 gib((uint64_t) budget.gpu_free),
                 gib(budget.core_bytes),
                 gib(budget.kv_bytes),
                 gib(budget.warm_bytes),
                 gib(budget.safety_bytes),
                 gib(budget.expert_budget),
                 hot_label.c_str());
    log_ds4_expert_memory_info("placement", placed_mem, w.n_layer);
    return true;
}

bool DeepSeek4Backend::init_hybrid_model() {
    TargetLoadPlan plan;
    plan.skip_expert_tensors = true;
    if (!load_deepseek4_gguf_partial(cfg_.model_path, backend_, plan, w_)) {
        std::fprintf(stderr, "[deepseek4] failed to partially load model for hybrid mode: %s\n",
                     cfg_.model_path);
        return false;
    }

    std::string err;
    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    if (!compute_uniform_hybrid_placement(
            w_, max_ctx, moe_placement_, &moe_decode_placement_, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to compute hybrid placement: %s\n", err.c_str());
        return false;
    }

    if (moe_placement_.total_hot >= w_.n_layer * w_.n_expert) {
        free_deepseek4_weights(w_);
        if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
            std::fprintf(stderr, "[deepseek4] failed to reload full model after placement: %s\n",
                         cfg_.model_path);
            return false;
        }
        return true;
    }

    const Ds4MoeTpConfig tp = ds4_moe_tp_config(cfg_.device.gpu);
    const bool inprocess_tp = tp.requested && tp.in_process;
    const PlacementBackend local_kind =
        cfg_.device.backend == PlacementBackend::Auto
            ? compiled_placement_backend() : cfg_.device.backend;
    if (inprocess_tp && !tp.backend_valid) {
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] invalid DFLASH_DS4_MOE_TP_BACKEND; "
                     "expected cuda or hip\n");
        return false;
    }
    const bool same_runtime_tp = inprocess_tp && tp.backend_valid &&
        tp.secondary_backend == local_kind;

    // Mix qtypes need a learned decode table for every resident tensor. The
    // same-runtime GPU path registers the matching expert subset after it
    // creates both owner tensors. CPU offload and cross-runtime peers keep the
    // existing monolithic fallback.
    bool has_mix_experts = false;
    for (const auto & L : w_.layers) {
        const struct { ggml_tensor * t; int qtype; const char * what; } mix_experts[] = {
            { L.ffn_gate_exps, GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) gate" },
            { L.ffn_up_exps,   GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) up"   },
            { L.ffn_down_exps, GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) down" },
            { L.ffn_gate_exps, GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) gate" },
            { L.ffn_up_exps,   GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) up"   },
            { L.ffn_down_exps, GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) down" },
        };
        for (const auto & m : mix_experts) {
            if (!m.t || m.t->type != m.qtype) {
                continue;
            }
            has_mix_experts = true;
            if (same_runtime_tp) continue;

            std::fprintf(stderr,
                "[deepseek4] %s experts cannot decode from hybrid/cold "
                "placement; falling back to monolithic full load\n", m.what);
            free_deepseek4_weights(w_);
            if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
                std::fprintf(stderr,
                    "[deepseek4] monolithic fallback failed (model does not "
                    "fit resident): %s\n", cfg_.model_path);
                return false;
            }
            return true;
        }
    }

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (same_runtime_tp && has_mix_experts) {
        const char * mix_mmq = std::getenv("DFLASH_DS4_MIX_MMQ_PREFILL");
        if (mix_mmq && std::strcmp(mix_mmq, "0") == 0) {
            std::fprintf(stderr,
                         "[deepseek4] heterogeneous mixed experts require "
                         "DFLASH_DS4_MIX_MMQ_PREFILL=1\n");
            return false;
        }
        if ((!mix_mmq || !mix_mmq[0]) &&
            ::setenv("DFLASH_DS4_MIX_MMQ_PREFILL", "1", 1) != 0) {
            std::fprintf(stderr,
                         "[deepseek4] failed to enable mixed-expert MMQ prefill\n");
            return false;
        }
    }
#endif

    auto hybrid = std::make_shared<MoeHybridStorage>();
    const auto fail_hybrid_init = [&]() {
        stream_engine_.destroy();
        hybrid.reset();
        if (expert_backend_) {
            ggml_backend_free(expert_backend_);
            expert_backend_ = nullptr;
        }
        return false;
    };
    MoeHybridConfig hybrid_cfg = make_ds4_parent_worker_cfg(w_);
    if (inprocess_tp) {
        const int expert_gpu = tp.secondary_gpu;
        const PlacementBackend expert_kind = tp.secondary_backend;
        if (expert_kind == local_kind && expert_gpu == cfg_.device.gpu) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] in-process secondary device must "
                         "differ from the primary device\n");
            return false;
        }
        if (expert_kind == local_kind && g_peer_access_opt_in) {
            const bool peer_ok = enable_peer_access_pair(cfg_.device.gpu, expert_gpu);
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] peer access %s:%d <-> %s:%d: %s\n",
                         placement_backend_name(local_kind), cfg_.device.gpu,
                         placement_backend_name(expert_kind), expert_gpu,
                         peer_ok ? "enabled" : "unavailable");
        } else if (expert_kind != local_kind) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] cross-vendor owner join %s:%d <-> %s:%d "
                         "uses in-process host staging\n",
                         placement_backend_name(local_kind), cfg_.device.gpu,
                         placement_backend_name(expert_kind), expert_gpu);
        }
        expert_backend_ = init_placement_backend(expert_kind, expert_gpu, &err);
        if (!expert_backend_) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] failed to initialize in-process "
                         "secondary backend %s:%d: %s\n",
                         placement_backend_name(expert_kind), expert_gpu,
                         err.c_str());
            return false;
        }
        hybrid_cfg.materialize_cold_experts = true;
        hybrid_cfg.cold_expert_backend = MoeHybridColdBackend::Gpu;
    }
    if (!build_deepseek4_moe_hybrid_storage_from_file_with_mmap(
            cfg_.model_path, backend_, w_, moe_placement_, &hybrid_cfg,
            *hybrid, &err, expert_backend_)) {
        std::fprintf(stderr, "[deepseek4] failed to build hybrid expert storage: %s\n", err.c_str());
        return fail_hybrid_init();
    }
    if (same_runtime_tp && has_mix_experts &&
        !register_deepseek4_moe_hybrid_mix_tables(
            cfg_.model_path, w_, *hybrid, &err)) {
        std::fprintf(stderr,
                     "[deepseek4] failed to register hybrid mixed experts: %s\n",
                     err.c_str());
        return fail_hybrid_init();
    }

    // The physical placement is shared by both phases. Decode may own only a
    // subset so its fast main branch does not outrun and then wait on the peer;
    // prefill continues to consume every resident expert.
    if (!moe_decode_placement_.empty()) {
        if (!moe_decode_placement_.matches(
                w_.n_layer, w_.n_expert, w_.n_expert_used)) {
            std::fprintf(stderr,
                         "[deepseek4] decode placement dimensions are invalid\n");
            return fail_hybrid_init();
        }
        for (int il = 0; il < w_.n_layer; ++il) {
            MoeHybridLayerStorage & layer = hybrid->layers[(size_t) il];
            layer.decode_hot_local_by_global.assign(
                (size_t) w_.n_expert, -1);
            for (int32_t expert :
                    moe_decode_placement_.hot_expert_ids[(size_t) il]) {
                if (expert < 0 || expert >= w_.n_expert ||
                    layer.hot_local_by_global[(size_t) expert] < 0) {
                    std::fprintf(stderr,
                                 "[deepseek4] decode owner expert %d in layer "
                                 "%d is not resident\n",
                                 (int) expert, il);
                    return fail_hybrid_init();
                }
                layer.decode_hot_local_by_global[(size_t) expert] =
                    layer.hot_local_by_global[(size_t) expert];
            }
        }
    }

    if (hybrid->has_mmap() && !hybrid->materialized_cold_experts) {
        size_t max_expert_bytes = 0;
        for (const auto & layer : hybrid->layers) {
            const size_t per_expert_bytes = layer.fused_gate_up
                ? layer.gate_up_expert_bytes + layer.down_expert_bytes
                : layer.gate_expert_bytes + layer.up_expert_bytes + layer.down_expert_bytes;
            max_expert_bytes = std::max(max_expert_bytes, per_expert_bytes);
        }
        if (max_expert_bytes == 0) {
            std::fprintf(stderr, "[deepseek4] failed to compute streaming expert size\n");
            return fail_hybrid_init();
        }
        if (!stream_engine_.init(backend_, max_expert_bytes, &err)) {
            std::fprintf(stderr, "[deepseek4] failed to init cold-expert stream engine: %s\n",
                         err.c_str());
            return fail_hybrid_init();
        }
        std::fprintf(stderr,
                     "[deepseek4] cold-expert stream engine ready: pinned=%.1f MiB scratch=%.1f MiB\n",
                     stream_engine_.pinned_bytes() / 1024.0 / 1024.0,
                     stream_engine_.scratch_bytes() / 1024.0 / 1024.0);
    }

    moe_hybrid_ = std::move(hybrid);
    w_.moe_hybrid = true;
    const int total_cold = w_.n_layer * w_.n_expert - moe_placement_.total_hot;
    const char * cold_backend =
        moe_hybrid_->cold_backend_kind == MoeHybridColdBackend::Gpu ? "gpu" : "cpu";
    std::fprintf(stderr, "[deepseek4] hybrid experts ready: hot=%d cold=%d cold_backend=%s%s\n",
                 moe_placement_.total_hot, total_cold, cold_backend, "");
    return true;
}

void DeepSeek4Backend::print_ready_banner() const {
    std::printf("[deepseek4-daemon] ready layers=%d ctx=%d experts=%d/%d\n",
                w_.n_layer, cache_.max_ctx, w_.n_expert_used, w_.n_expert);
    std::fflush(stdout);
}

bool DeepSeek4Backend::park(ParkTarget target) {
    const bool want_draft = park_target_includes_draft_model(target);
    const bool want_target_model = park_target_includes_target_model(target);

    if (want_draft && spec_drafter_) {
        release_spec_drafter(/*mark_parked=*/true);
        std::printf("[deepseek4] DSpark drafter parked (VRAM released)\n");
        std::fflush(stdout);
    }
    if (!want_target_model || parked_) return true;

    maybe_save_routing_stats();
    for (int i = 0; i < PREFIX_SLOTS; ++i) {
        snapshot_free(i);
    }
    last_logits_.clear();
    last_logits_pos_ = -1;
    free_deepseek4_cache(cache_);
    expert_runtime_.reset();
    stream_engine_.destroy();
    moe_hybrid_.reset();
    if (expert_backend_) {
        ggml_backend_free(expert_backend_);
        expert_backend_ = nullptr;
    }
    moe_placement_ = {};
    moe_decode_placement_ = {};
    free_deepseek4_weights(w_);
    parked_ = true;
    if (spec_drafter_) {
        std::printf("[deepseek4] target parked (target VRAM released; "
                    "DSpark drafter retained)\n");
    } else {
        std::printf("[deepseek4] target parked (target VRAM released)\n");
    }
    std::fflush(stdout);
    return true;
}

bool DeepSeek4Backend::unpark(ParkTarget target) {
    const bool want_draft = park_target_includes_draft_model(target);
    const bool want_target_model = park_target_includes_target_model(target);

    if (want_target_model && parked_) {
        if (!load_model()) {
            std::fprintf(stderr, "[deepseek4] unpark: failed to restore target model\n");
            free_deepseek4_weights(w_);
            stream_engine_.destroy();
            moe_hybrid_.reset();
            if (expert_backend_) {
                ggml_backend_free(expert_backend_);
                expert_backend_ = nullptr;
            }
            moe_placement_ = {};
            moe_decode_placement_ = {};
            return false;
        }

        const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
        if (!create_deepseek4_cache(backend_, w_, max_ctx, cache_)) {
            std::fprintf(stderr,
                         "[deepseek4] unpark: failed to recreate KV cache (ctx=%d)\n",
                         max_ctx);
            free_deepseek4_cache(cache_);
            free_deepseek4_weights(w_);
            stream_engine_.destroy();
            moe_hybrid_.reset();
            if (expert_backend_) {
                ggml_backend_free(expert_backend_);
                expert_backend_ = nullptr;
            }
            moe_placement_ = {};
            moe_decode_placement_ = {};
            return false;
        }

        if (env_flag_enabled("DFLASH_DS4_MOE_TP") &&
            !init_moe_tensor_parallel()) {
            free_deepseek4_cache(cache_);
            free_deepseek4_weights(w_);
            expert_runtime_.reset();
            stream_engine_.destroy();
            moe_hybrid_.reset();
            if (expert_backend_) {
                ggml_backend_free(expert_backend_);
                expert_backend_ = nullptr;
            }
            moe_placement_ = {};
            moe_decode_placement_ = {};
            return false;
        }

        parked_ = false;
        std::printf("[deepseek4] target unparked (VRAM restored)\n");
        std::fflush(stdout);
    }
    if (!validate_prefill_mode()) {
        free_deepseek4_weights(w_);
        stream_engine_.destroy();
        moe_hybrid_.reset();
        moe_placement_ = {};
        moe_decode_placement_ = {};
        return false;
    }

    if (want_draft && spec_drafter_parked_) {
        if (parked_) {
            std::fprintf(stderr,
                         "[deepseek4] unpark: restore target before DSpark drafter\n");
            return false;
        }
        if (!load_spec_drafter()) {
            std::fprintf(stderr, "[deepseek4] unpark: failed to restore DSpark drafter\n");
            return false;
        }
    }
    cache_.prefill_mode = cfg_.prefill_mode;
    return true;
}

int deepseek4_hybrid_prefill_chunk_tokens(
        int requested_chunk,
        int context_end,
        int current_cap) {
    constexpr int long_context_begin = 4096;
    constexpr int long_context_chunk = 1024;
    int bounded = std::max(1, requested_chunk);
    if (current_cap > 0) {
        bounded = std::min(bounded, current_cap);
    }
    return context_end > long_context_begin
        ? std::min(bounded, long_context_chunk)
        : bounded;
}

int DeepSeek4Backend::do_prefill(const std::vector<int32_t> & tokens,
                                  const DaemonIO & io,
                                  int kv_offset,
                                  int snap_slot,
                                  int snap_pos) {
    const InferencePhase phase = deepseek4_roctx_prefill_phase(
        prefill_attention_mode_name(cfg_.prefill_mode));
    const DeepSeek4RoctxPhaseScope roctx_phase(phase);
    const DeepSeek4RoctxRange roctx_range(
        "ds4.prefill",
        {phase, static_cast<int>(tokens.size()), 0, w_.n_layer, cfg_.device.gpu});
    // The all-hot layer-range path supports causal chunked prefill. The
    // optimized graph snapshots the previous raw SWA window, attends over
    // that snapshot plus the current ubatch, and commits only the final SWA
    // tail. Learned compressor boundaries are emitted inside the same graph.
    //
    // Mixed hot/cold hybrid execution still has single-token HC semantics, so
    // retain the reference path there.  --chunk 1 is the explicit fallback.
    const int requested_chunk = cfg_.chunk > 0 ? cfg_.chunk : w_.n_swa;
    const int n_total = (int)tokens.size();
    // Bound the layer-major graph to the topology validated by the prefill
    // kernels. Smaller tail chunks use the same scheduler or its reference
    // fallback.
    const int layer_major_cap = DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS;
    // Only sparse prefill has a qualified batched mixed-owner HC path. Dense
    // hybrid execution remains tokenwise; batching it would skip per-token HC
    // post-mixing and corrupt the hidden state.
    const bool hybrid_batch_supported =
        !moe_hybrid_ || cfg_.prefill_mode == PrefillAttentionMode::Sparse;
    const int base_chunk =
        !prefill_attention_mode_is_approximate(cfg_.prefill_mode) ||
        !hybrid_batch_supported
        ? 1
        : std::max(1, std::min(requested_chunk,
                               layer_major_cap));
    const bool bound_hybrid_scratch =
        moe_hybrid_ &&
        cfg_.prefill_mode == PrefillAttentionMode::Sparse;
    const int chunk = bound_hybrid_scratch
        ? deepseek4_hybrid_prefill_chunk_tokens(
              base_chunk, kv_offset + n_total,
              hybrid_prefill_chunk_cap_)
        : base_chunk;
    if (chunk < base_chunk) {
        hybrid_prefill_chunk_cap_ = hybrid_prefill_chunk_cap_ > 0
            ? std::min(hybrid_prefill_chunk_cap_, chunk)
            : chunk;
    }
    if (chunk < base_chunk) {
        std::fprintf(stderr,
                     "[deepseek4] hybrid prefill scratch bound: "
                     "chunk %d->%d for context_end=%d (sticky)\n",
                     base_chunk, chunk, kv_offset + n_total);
    }
    int pos = kv_offset;
    const bool save_snapshot =
        snap_slot >= 0 && snap_slot < PREFIX_SLOTS &&
        snap_pos > kv_offset && snap_pos <= kv_offset + n_total;
    // New sequence: clear the cache buffer so compressor state double-buffers
    // and compressed-KV rows start from zeros, exactly like a fresh server.
    // Without this, the first flush windows of a request pool over the
    // previous request's leftover state rows and outputs from the 2nd/3rd
    // request on can drift by a token or two.
    if (kv_offset == 0) {
        reset_deepseek4_cache(cache_);
    }
    last_logits_.clear();
    last_logits_pos_ = -1;
    int spec_final_from = n_total;
    int spec_snap_from = n_total;
    int spec_snap_to = 0;
    int spec_old_rows_for_final = 0;
    if (spec_enabled_ && spec_drafter_) {
        const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
        const int snap_tokens = save_snapshot ? snap_pos - kv_offset : n_total;
        spec_final_from = std::max(0, n_total - w_.n_swa);
        if (save_snapshot) {
            spec_snap_from = std::max(0, snap_tokens - w_.n_swa);
            spec_snap_to = snap_tokens;
        }
        if (kv_offset == 0 || feat_row <= 0 ||
            spec_feat_window_.size() % (size_t) feat_row != 0) {
            spec_feat_window_.clear();
        } else {
            // Preserve enough restored rows for both the requested checkpoint
            // and the final prompt tail. The live vector is trimmed after
            // prefill; the snapshot copy is independently trimmed at save.
            const size_t old_rows = spec_feat_window_.size() / (size_t) feat_row;
            spec_old_rows_for_final = std::max(0, w_.n_swa - n_total);
            const int old_rows_for_snap = save_snapshot
                ? std::max(0, w_.n_swa - snap_tokens) : 0;
            const size_t keep_rows = std::min(
                old_rows,
                (size_t) std::max(spec_old_rows_for_final,
                                  old_rows_for_snap));
            if (old_rows > keep_rows) {
                const size_t drop_floats = (old_rows - keep_rows) * (size_t) feat_row;
                const size_t keep_floats = keep_rows * (size_t) feat_row;
                std::memmove(spec_feat_window_.data(),
                             spec_feat_window_.data() + drop_floats,
                             keep_floats * sizeof(float));
                spec_feat_window_.resize(keep_floats);
            }
        }
    }
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;

    bool snapshot_saved = false;
    for (int i = 0; i < n_total;) {
        if (io.is_cancelled()) return pos;

        int n_tok = std::min(chunk, n_total - i);
        // A snapshot must represent an exact token boundary. Split a batched
        // prefill chunk when the requested boundary falls inside it.
        if (save_snapshot && !snapshot_saved &&
            snap_pos > pos && snap_pos < pos + n_tok) {
            n_tok = snap_pos - pos;
        }
        if (spec_enabled_ && spec_drafter_) {
            const bool batch_final_capture =
                supports_batched_spec_feature_capture(
                    w_.moe_hybrid, cache_.prefill_mode, n_tok);
            n_tok = capture_safe_prefill_tokens(
                i, n_tok, spec_final_from, batch_final_capture,
                save_snapshot && !snapshot_saved,
                spec_snap_from, spec_snap_to);
        }

        // Embed tokens
        std::vector<float> embed(w_.n_embd * n_tok);
        const auto embed_t0 = Clock::now();
        w_.embedder.embed(tokens.data() + i, n_tok, embed.data());
        DeepSeek4StepTelemetry step_tel;
        if (timing) step_tel.embed_us = elapsed_us(embed_t0, Clock::now());

        std::vector<float> logits;
        bool ok = false;
        std::vector<float> hc_state;
        Ds4VerifyHooks spec_hooks;
        std::vector<float> spec_cap;
        Ds4VerifyHooks * hp = nullptr;
        const bool capture_final = i + n_tok > spec_final_from;
        const bool capture_snapshot =
            !snapshot_saved && i < spec_snap_to &&
            i + n_tok > spec_snap_from;
        if (spec_enabled_ && spec_drafter_ &&
            (capture_final || capture_snapshot)) {
            spec_hooks.capture_layer_ids = &spec_drafter_->capture_layer_ids;
            spec_hooks.capture_out = &spec_cap;
            int capture_begin = n_tok;
            int capture_end = 0;
            if (capture_final) {
                capture_begin = std::min(
                    capture_begin, std::max(0, spec_final_from - i));
                capture_end = n_tok;
            }
            if (capture_snapshot) {
                capture_begin = std::min(
                    capture_begin, std::max(0, spec_snap_from - i));
                capture_end = std::max(
                    capture_end, std::min(n_tok, spec_snap_to - i));
            }
            spec_hooks.capture_token_begin = capture_begin;
            spec_hooks.capture_token_end = capture_end;
            hp = &spec_hooks;
        }
        if (moe_hybrid_ && (expert_runtime_.compute || expert_backend_)) {
            ok = deepseek4_step_layer_range(
                backend_, cfg_.device.gpu, w_, cache_, hc_state,
                embed.data(), n_tok, pos,
                0, w_.n_layer, &logits,
                tokens.data() + i,
                timing ? &step_tel : nullptr,
                /*allow_decode_graph_reuse=*/true, hp,
                moe_hybrid_.get(),
                expert_runtime_.compute ? &expert_runtime_ : nullptr,
                routing_stats_.get());
        } else if (moe_hybrid_) {
            ok = deepseek4_step(backend_, cfg_.device.gpu, w_, cache_, embed.data(), n_tok, pos, logits,
                                moe_hybrid_.get(), tokens.data() + i,
                                &stream_engine_,
                                timing ? &step_tel : nullptr,
                                routing_stats_.get(),
                                hp,
                                expert_runtime_.compute ? &expert_runtime_ : nullptr);
        } else {
            ok = deepseek4_step_layer_range(backend_, cfg_.device.gpu, w_, cache_, hc_state,
                                            embed.data(), n_tok, pos,
                                            0, w_.n_layer, &logits,
                                            tokens.data() + i,
                                            timing ? &step_tel : nullptr,
                                            cfg_.prefill_mode != PrefillAttentionMode::Sparse, hp);
        }
        if (ok && hp && !spec_cap.empty()) {
            const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
            for (int t = 0; t < n_tok; ++t) {
                const int token_index = i + t;
                const bool keep_for_final = token_index >= spec_final_from;
                const bool keep_for_snapshot =
                    !snapshot_saved && token_index >= spec_snap_from &&
                    token_index < spec_snap_to;
                if (!keep_for_final && !keep_for_snapshot) continue;
                spec_feat_window_.insert(spec_feat_window_.end(),
                    spec_cap.begin() + (size_t) t * feat_row,
                    spec_cap.begin() + (size_t) (t + 1) * feat_row);
            }
        }
        if (!ok) {
            std::fprintf(stderr, "[deepseek4] prefill step failed at pos=%d\n", pos);
            return -1;
        }
        if (timing) {
            add_step_tel(tel_acc, step_tel);
            steps++;
        }
        last_logits_ = std::move(logits);
        pos += n_tok;
        last_logits_pos_ = cache_.cur_pos;
        i += n_tok;
        if (save_snapshot && !snapshot_saved && pos == snap_pos) {
            snapshot_saved = snapshot_save(snap_slot);
            if (!snapshot_saved) {
                std::fprintf(stderr,
                             "[deepseek4] failed to save snapshot slot=%d pos=%d\n",
                             snap_slot, snap_pos);
            } else if (spec_enabled_ && spec_drafter_) {
                // Discard checkpoint-only rows once their snapshot is saved.
                // Retain just the already-captured prefix of the final SWA
                // window, so distant checkpoints do not bridge a huge gap in
                // host feature memory.
                const int processed = i;
                const int final_new_rows =
                    std::max(0, processed - spec_final_from);
                keep_spec_feature_tail(
                    spec_feat_window_,
                    (size_t) spec_old_rows_for_final +
                    (size_t) final_new_rows);
            }
        }
    }
    keep_spec_feature_tail(spec_feat_window_,
                           (size_t) std::max(0, w_.n_swa));
    if (timing) {
        log_step_tel("prefill", n_total, steps, elapsed_s(phase_t0), tel_acc);
    }
    return pos;
}

bool DeepSeek4Backend::do_decode(int committed, int n_gen,
                                  const std::vector<int32_t> & history_prefix,
                                  std::vector<int32_t> & out_tokens,
                                  const DaemonIO & io,
                                  const BudgetHook & budget_hook,
                                  bool * forced_close_out) {
    const DeepSeek4RoctxPhaseScope roctx_phase(InferencePhase::Decode);
    if (forced_close_out) *forced_close_out = false;
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;
    const bool process_logits = sampler_.needs_logit_processing();
    std::vector<int32_t> history;
    if (process_logits) {
        history = history_prefix;
        if (n_gen > 0) {
            history.reserve(history.size() + (size_t)n_gen);
        }
    }

    // Budget-hook state. The close sequence is injected one token per step, then decoding
    // CONTINUES so the model can spend the reserved reply budget on a visible answer. The
    // previous implementation pushed the whole close sequence and broke out of the loop, which
    // (a) never ran a forward over the injected tokens, so KV state did not reflect them, and
    // (b) ended generation, so the reply budget was reserved and never usable. Measured on
    // DeepSeek-V4-Flash: total tokens came to exactly thinking_ceiling + len(close_sequence)
    // for close sequences of 1, 3 and 23 tokens -- zero tokens of answer in every case, while
    // finish_reason still reported "stop". This mirrors qwen35_backend's override-and-continue.
    bool budget_close_started = false;
    size_t close_inject_pos = 0;

    for (int generated = 0; generated < n_gen; generated++) {
        if (io.is_cancelled()) break;

        // Get last logits and sample
        std::vector<float> logits;
        if (generated == 0 && !last_logits_.empty()) {
            logits = last_logits_;
        } else {
            std::vector<float> embed(w_.n_embd);
            int32_t tok_to_eval = out_tokens.empty() ? 0 : out_tokens.back();
            const auto embed_t0 = Clock::now();
            w_.embedder.embed(&tok_to_eval, 1, embed.data());
            DeepSeek4StepTelemetry step_tel;
            if (timing) step_tel.embed_us = elapsed_us(embed_t0, Clock::now());

            const int pos = std::max(0, committed + generated - 1);
            bool ok = false;
            if (moe_hybrid_ && (expert_runtime_.compute || expert_backend_)) {
                std::vector<float> hc_state;
                ok = deepseek4_step_layer_range(
                    backend_, cfg_.device.gpu, w_, cache_, hc_state,
                    embed.data(), 1, pos,
                    0, w_.n_layer, &logits,
                    &tok_to_eval,
                    timing ? &step_tel : nullptr,
                    /*allow_decode_graph_reuse=*/true, nullptr,
                    moe_hybrid_.get(),
                    expert_runtime_.compute ? &expert_runtime_ : nullptr,
                    routing_stats_.get());
            } else if (moe_hybrid_) {
                ok = deepseek4_step(backend_, cfg_.device.gpu, w_, cache_, embed.data(), 1,
                                    pos, logits,
                                    moe_hybrid_.get(), &tok_to_eval,
                                    &stream_engine_,
                                    timing ? &step_tel : nullptr,
                                    routing_stats_.get(),
                                    nullptr,
                                    expert_runtime_.compute ? &expert_runtime_ : nullptr);
            } else {
                std::vector<float> hc_state;
                ok = deepseek4_step_layer_range(backend_, cfg_.device.gpu, w_, cache_, hc_state,
                                                embed.data(), 1, pos,
                                                0, w_.n_layer, &logits,
                                                &tok_to_eval,
                                                timing ? &step_tel : nullptr);
            }
            if (!ok) {
                std::fprintf(stderr, "[deepseek4] decode step failed\n");
                return false;
            }
            if (timing) {
                add_step_tel(tel_acc, step_tel);
                steps++;
            }
        }

        int32_t next_token = 0;
        const auto sample_t0 = Clock::now();
        if (process_logits) {
            next_token = sample_logits(logits.data(), w_.n_vocab, sampler_,
                                       history, sampler_rng_);
        } else {
            float max_val = logits[0];
            for (int i = 1; i < w_.n_vocab; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    next_token = i;
                }
            }
        }
        if (timing) tel_acc.sample_us += elapsed_us(sample_t0, Clock::now());

        // Budget hook: steer the tail of the window into the close sequence, then let the model
        // keep going. Runs before history.push_back so penalty history records what was
        // actually emitted. The rule lives in a header-only helper so it is testable without a
        // model; see deepseek4_budget_hook.h for why this overrides rather than appends.
        {
            bool hook_forced = false;
            next_token = dflash::deepseek4::budget_hook_apply(
                budget_hook.close_token_ids, n_gen - generated,
                budget_hook.hard_limit_remaining, next_token,
                budget_close_started, close_inject_pos, hook_forced);
            if (hook_forced && forced_close_out) *forced_close_out = true;
        }

        if (process_logits) {
            history.push_back(next_token);
        }
        if (generated > 0) {
            // The forward above advanced cache_ through the previously
            // emitted token. Retain its logits so a later manual/continued
            // snapshot can resume at exactly cache_.cur_pos.
            last_logits_ = std::move(logits);
            last_logits_pos_ = cache_.cur_pos;
        }
        out_tokens.push_back(next_token);
        const auto emit_t0 = Clock::now();
        io.emit(next_token);
        if (timing) tel_acc.emit_us += elapsed_us(emit_t0, Clock::now());

        if (deepseek4_is_eos_tok(next_token, w_)) {
            break;
        }
    }
    if (timing) {
        log_step_tel("decode", (int)out_tokens.size(), steps, elapsed_s(phase_t0), tel_acc);
    }
    return true;
}

GenerateResult DeepSeek4Backend::generate_impl(const GenerateRequest & req,
                                                const DaemonIO & io) {
    return generate_from_state(req, io, 0);
}

GenerateResult DeepSeek4Backend::generate_from_state(
        const GenerateRequest & req, const DaemonIO & io, int kv_offset) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    auto t0 = Clock::now();
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    if (kv_offset < 0 || kv_offset > (int) req.prompt.size()) {
        result.fail(GenerateErrorCode::PrefillFailed,
                    "restored prefix exceeds prompt length");
        return result;
    }

    // Prefill only the suffix that is not already represented by a restored
    // snapshot. An exact full-prompt hit can decode immediately from the
    // logits and speculative feature window saved with the cache state.
    int committed = kv_offset;
    if (kv_offset == 0) {
        committed = do_prefill(req.prompt, out_io, 0,
                               req.snap_slot, req.snap_pos);
    } else if (kv_offset < (int) req.prompt.size()) {
        std::vector<int32_t> suffix(req.prompt.begin() + kv_offset,
                                    req.prompt.end());
        committed = do_prefill(suffix, out_io, kv_offset,
                               req.snap_slot, req.snap_pos);
    }
    if (committed < 0) {
        result.fail(GenerateErrorCode::PrefillFailed);
        return result;
    }
    result.prefill_s = elapsed_s(t0);

    if (out_io.is_cancelled()) {
        result.succeed();
        maybe_save_routing_stats();
        return result;
    }

    if (req.n_gen <= 0) {
        result.succeed();
        maybe_save_routing_stats();
        return result;
    }

    // Decode
    auto t1 = Clock::now();
    const bool budget_requires_ar = !req.budget_hook.close_token_ids.empty();
    // The DSpark verifier is greedy-only. Route sampling and penalties through
    // AR so the request's sampler contract is not silently ignored.
    const bool sampling_requires_ar = sampler_.needs_logit_processing();
    if (spec_enabled_ && spec_drafter_ && req.n_gen > 0 &&
        !req.force_ar_decode && !budget_requires_ar && !sampling_requires_ar) {
        if (last_logits_.empty()) {
            result.fail(GenerateErrorCode::DecodeFailed, "spec: no prefill logits");
            return result;
        }
        int seed = 0;
        { float mv = last_logits_[0];
          for (int i = 1; i < w_.n_vocab; i++) if (last_logits_[i] > mv) { mv = last_logits_[i]; seed = i; } }
        std::vector<int32_t> gen;
        gen.push_back(seed);
        out_io.emit(seed);
        float accept_rate = 0.0f;
        bool spec_ran = false;
        if (!out_io.is_cancelled() && !deepseek4_is_eos_tok(seed, w_) && req.n_gen > 1) {
            const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
            const int win_len = feat_row > 0 ? (int) (spec_feat_window_.size() / feat_row) : 0;
            std::vector<int32_t> spec_toks;
            spec_ran = true;
            // The DSpark API does not return the final target logits. Once it
            // advances the target cache, reject post-decode snapshots rather
            // than pairing that state with stale prefill logits.
            last_logits_pos_ = -1;
            if (!run_deepseek4_dspark_spec_decode(
                    backend_, cfg_.device.gpu, w_, cache_, *spec_drafter_, committed, seed,
                    req.n_gen - 1,
                    win_len > 0 ? spec_feat_window_.data() : nullptr, win_len,
                    spec_toks, &accept_rate,
                    [&out_io](int32_t tok) {
                        if (out_io.is_cancelled()) return false;
                        out_io.emit(tok);
                        return !out_io.is_cancelled();
                    },
                    (expert_runtime_.compute || expert_backend_)
                        ? moe_hybrid_.get() : nullptr,
                    expert_runtime_.compute ? &expert_runtime_ : nullptr,
                    routing_stats_.get())) {
                result.fail(GenerateErrorCode::DecodeFailed,
                            "DSpark speculative decode failed");
                return result;
            }
            gen.insert(gen.end(), spec_toks.begin(), spec_toks.end());
        }
        result.succeed();
        result.tokens = std::move(gen);
        result.decode_s = elapsed_s(t1);
        result.accept_rate = accept_rate;
        result.spec_decode_ran = spec_ran;
        std::fprintf(stderr, "[deepseek4] DSpark decode: %zu tok in %.3fs (%.1f tok/s) accept_rate=%.2f\n",
                     result.tokens.size(), result.decode_s,
                     result.decode_s > 0 ? result.tokens.size() / result.decode_s : 0.0, accept_rate);
        maybe_save_routing_stats();
        return result;
    }
    std::vector<int32_t> gen_tokens;
    gen_tokens.reserve(req.n_gen);

    bool forced_close = false;
    if (!do_decode(committed, req.n_gen, req.prompt, gen_tokens, out_io,
                   req.budget_hook, &forced_close)) {
        result.fail(GenerateErrorCode::DecodeFailed);
        return result;
    }

    result.succeed();
    result.tokens = std::move(gen_tokens);
    result.decode_s = elapsed_s(t1);
    result.budget_forced_close = forced_close;
    maybe_save_routing_stats();
    return result;
}

// ── Snapshots ───────────────────────────────────────────────────────────

bool DeepSeek4Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || !snap_backend_ ||
        cache_.cur_pos <= 0 || last_logits_pos_ != cache_.cur_pos ||
        w_.n_vocab <= 0 ||
        last_logits_.size() != (size_t) w_.n_vocab) {
        return false;
    }

    snapshot_free(slot);
    if (!deepseek4_snapshot_save(cache_, snap_backend_, snapshots_[slot])) {
        return false;
    }

    try {
        auto & aux = snapshot_aux_[slot];
        aux.last_logits = last_logits_;
        aux.spec_feat_window = spec_feat_window_;
        keep_spec_feature_tail(aux.spec_feat_window,
                               (size_t) std::max(0, w_.n_swa));
        aux.used = true;
    } catch (const std::bad_alloc &) {
        snapshot_free(slot);
        return false;
    }

    const size_t core_bytes = snapshots_[slot].buf
        ? ggml_backend_buffer_get_size(snapshots_[slot].buf) : 0;
    const size_t aux_bytes =
        (snapshot_aux_[slot].last_logits.size() +
         snapshot_aux_[slot].spec_feat_window.size()) * sizeof(float);
    std::fprintf(stderr,
                 "[deepseek4] snapshot saved slot=%d pos=%d size=%.1f MiB\n",
                 slot, snapshots_[slot].cur_pos,
                 (double) (core_bytes + aux_bytes) / (1024.0 * 1024.0));
    return true;
}

void DeepSeek4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_deepseek4_snapshot(snapshots_[slot]);
    snapshot_aux_[slot] = SnapshotAux{};
}

bool DeepSeek4Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    const auto & snap = snapshots_[slot];
    const auto & aux = snapshot_aux_[slot];
    return snap.ctx != nullptr && snap.buf != nullptr && snap.cur_pos > 0 &&
           aux.used && w_.n_vocab > 0 &&
           aux.last_logits.size() == (size_t) w_.n_vocab;
}

int DeepSeek4Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return snapshots_[slot].cur_pos;
}

bool DeepSeek4Backend::snapshot_restore(int slot) {
    if (!snapshot_used(slot)) return false;

    std::vector<float> restored_logits;
    std::vector<float> restored_features;
    try {
        restored_logits = snapshot_aux_[slot].last_logits;
        restored_features = snapshot_aux_[slot].spec_feat_window;
    } catch (const std::bad_alloc &) {
        return false;
    }

    if (!deepseek4_snapshot_restore(snapshots_[slot], cache_)) {
        return false;
    }
    last_logits_ = std::move(restored_logits);
    spec_feat_window_ = std::move(restored_features);
    last_logits_pos_ = cache_.cur_pos;
    return true;
}

GenerateResult DeepSeek4Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    GenerateResult result;
    if (!snapshot_used(slot)) {
        result.fail(GenerateErrorCode::InvalidSnapshotSlot);
        return result;
    }

    const int snap_pos = snapshot_cur_pos(slot);
    if (snap_pos > (int) req.prompt.size()) {
        std::fprintf(stderr,
                     "[pc] DeepSeek snapshot longer than prompt "
                     "(snap=%d > prompt=%zu) -- fresh prefill fallback\n",
                     snap_pos, req.prompt.size());
        return generate_impl(req, io);
    }
    if (!snapshot_restore(slot)) {
        result.fail(GenerateErrorCode::BackendSpecific, "snapshot restore");
        return result;
    }
    return generate_from_state(req, io, snap_pos);
}

bool DeepSeek4Backend::handle_compress(const std::string & line,
                                        const DaemonIO & io) {
    (void)line; (void)io;
    std::fprintf(stderr, "[deepseek4] compress not yet supported\n");
    return false;
}

void DeepSeek4Backend::free_drafter() {
    // Keep the configured path so request-scoped residency and an explicit
    // later `unpark draft` can restore the DSpark model.
    release_spec_drafter(/*mark_parked=*/true);
}

void DeepSeek4Backend::maybe_save_routing_stats() {
    if (!routing_stats_ || routing_stats_out_path_.empty()) return;
    std::string err;
    if (!routing_stats_->save_csv(routing_stats_out_path_, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to save routing stats %s: %s\n",
                     routing_stats_out_path_.c_str(), err.c_str());
    }
}

void DeepSeek4Backend::shutdown() {
    maybe_save_routing_stats();
    free_drafter();
    for (int i = 0; i < PREFIX_SLOTS; i++) {
        snapshot_free(i);
    }
    free_deepseek4_cache(cache_);
    expert_runtime_.reset();
    stream_engine_.destroy();
    moe_hybrid_.reset();
    if (expert_backend_) {
        ggml_backend_free(expert_backend_);
        expert_backend_ = nullptr;
    }
    routing_stats_.reset();
    routing_stats_out_path_.clear();
    moe_placement_ = {};
    moe_decode_placement_ = {};
    free_deepseek4_weights(w_);
    if (snap_backend_) { ggml_backend_free(snap_backend_); snap_backend_ = nullptr; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

}  // namespace dflash::common
