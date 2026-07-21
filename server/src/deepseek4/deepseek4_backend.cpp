// DeepSeek4Backend implementation — AR-only decode, chunked prefill.

#include "deepseek4_backend.h"
#include "deepseek4_internal.h"
#include "common/peer_access.h"
#include "common/sampler.h"

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "common/gpu_runtime_compat.h"
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <limits>

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

static void configure_gfx1151_dspark_mmvq_default(int gpu) {
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (!env_flag_enabled("DFLASH_DS4_SPEC") ||
        std::getenv("LUCE_MMVQ_MAX_NCOLS") != nullptr) {
        return;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, gpu) != cudaSuccess ||
        std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0) {
        return;
    }

    if (::setenv("LUCE_MMVQ_MAX_NCOLS", "4", 0) == 0) {
        std::fprintf(stderr,
                     "[deepseek4] gfx1151 DSpark: defaulting "
                     "LUCE_MMVQ_MAX_NCOLS=4\n");
    }
#else
    (void) gpu;
#endif
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

static bool ds4_inprocess_moe_tp_enabled() {
    return env_flag_enabled("DFLASH_DS4_MOE_TP_INPROC");
}

static int ds4_moe_tp_gpu(int local_gpu) {
    const char * raw = std::getenv("DFLASH_DS4_MOE_TP_GPU");
    if (!raw || !*raw) {
        raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU");
    }
    if (raw && *raw) return std::max(0, std::atoi(raw));
    return local_gpu == 0 ? 1 : 0;
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
    dst.full_graph_alloc_us += src.full_graph_alloc_us;
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
        "full_build=%.1fms full_alloc=%.1fms full_set=%.1fms full_compute=%.1fms full_read=%.1fms "
        "ffn_build=%.1fms ffn_compute=%.1fms ffn_read=%.1fms "
        "route_build=%.1fms route_compute=%.1fms route_read=%.1fms route_select=%.1fms "
        "ffn=%.1fms hot=%.1fms cold=%.1fms combine=%.1fms partition=%.1fms "
        "ffn_hot_graph_build=%llu ffn_hot_graph_hit=%llu ffn_cold_graph_build=%llu ffn_cold_graph_hit=%llu "
        "hc_pre=%.1fms hc_pre_build=%.1fms hc_pre_input=%.1fms hc_pre_compute=%.1fms "
        "hc_post=%.1fms output=%.1fms sample=%.1fms emit=%.1fms "
        "hot_sel=%d cold_sel=%d\n",
        phase, tokens, steps, wall_s, tok_s,
        ms(t.total_us), ms(t.embed_us), ms(t.attn_build_us), ms(t.attn_compute_us), ms(t.attn_read_us),
        ms(t.full_graph_build_us), ms(t.full_graph_alloc_us), ms(t.full_graph_set_us),
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

static bool fill_profiled_hot_placement(const DeepSeek4Weights & w,
                                        int hot_per_layer,
                                        const char * profile_path,
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
        std::vector<int> ranked = stats.hot_experts(il, hot_per_layer);
        auto & ids = out.hot_expert_ids[(size_t)il];
        ids.assign(ranked.begin(), ranked.end());
    }
    return true;
}

// Assign the same total number of resident experts as the uniform placement,
// but distribute those slots across layers to minimize the predicted owner
// critical path.  Uniform expert counts are a poor fit for heterogeneous EP:
// routing skew varies substantially by layer, while every layer joins on the
// slower of its R9700 hot/shared and Strix cold branches.
//
// The cost model intentionally uses measured bandwidth rather than advertised
// peak bandwidth.  It is only an allocation objective; actual placement still
// uses authoritative router statistics and evaluates every selected expert.
static bool fill_time_balanced_profiled_hot_placement(
        const DeepSeek4Weights & w,
        int active_routes,
        int hot_per_layer,
        const char * profile_path,
        MoeHybridPlacement & out,
        std::string * err) {
    MoeHybridRoutingStats stats;
    if (!MoeHybridRoutingStats::load_csv(profile_path, stats, err)) {
        return false;
    }
    if (stats.n_layer != w.n_layer || stats.n_expert != w.n_expert) {
        if (err) *err = "routing profile shape does not match DeepSeek V4 target";
        return false;
    }

    const int total_hot_budget = hot_per_layer * w.n_layer;
    if (total_hot_budget <= 0 || active_routes <= 0 ||
        active_routes > w.n_expert_used) {
        if (err) *err = "time-balanced placement requires a positive hot budget";
        return false;
    }

    const auto env_double = [](const char * name, double fallback) {
        const char * raw = std::getenv(name);
        if (!raw || !*raw) return fallback;
        const double value = std::atof(raw);
        return value > 0.0 ? value : fallback;
    };
    const auto env_int = [](const char * name, int fallback) {
        const char * raw = std::getenv(name);
        if (!raw || !*raw) return fallback;
        return std::atoi(raw);
    };

    // Sustained read probes on lucebox5, in GB/s.  Only their ratio affects
    // the optimizer, but retaining physical units makes the model auditable.
    const double main_bw = env_double(
        "DFLASH_DS4_TP_MAIN_OWNER_GBPS", 519.91);
    const double cold_bw = env_double(
        "DFLASH_DS4_TP_COLD_OWNER_GBPS", 242.36);
    const double shared_equiv = env_double(
        "DFLASH_DS4_TP_SHARED_EXPERT_EQUIV", 1.0);
    const int min_hot = std::clamp(env_int(
        "DFLASH_DS4_TP_MIN_HOT_PER_LAYER", 4), 0, w.n_expert);
    const int max_hot = std::clamp(env_int(
        "DFLASH_DS4_TP_MAX_HOT_PER_LAYER", 64), min_hot, w.n_expert);

    if ((int64_t) min_hot * w.n_layer > total_hot_budget ||
        (int64_t) max_hot * w.n_layer < total_hot_budget) {
        if (err) *err = "time-balanced placement bounds cannot satisfy hot budget";
        return false;
    }

    std::vector<std::vector<int>> ranked((size_t) w.n_layer);
    std::vector<std::vector<double>> coverage(
        (size_t) w.n_layer,
        std::vector<double>((size_t) max_hot + 1, 0.0));
    for (int il = 0; il < w.n_layer; ++il) {
        ranked[(size_t) il] = stats.ranked_experts(il);
        const uint64_t total = stats.layer_totals[(size_t) il];
        if (total == 0) {
            if (err) *err = "routing profile contains an empty layer";
            return false;
        }
        uint64_t cumulative = 0;
        for (int k = 1; k <= max_hot; ++k) {
            cumulative += stats.count(il, ranked[(size_t) il][(size_t) k - 1]);
            coverage[(size_t) il][(size_t) k] =
                (double) cumulative / (double) total;
        }
    }

    const auto layer_cost = [&](int il, int hot_count) {
        const double hot_fraction = coverage[(size_t) il][(size_t) hot_count];
        const double hot_routes = (double) active_routes * hot_fraction;
        const double cold_routes =
            (double) active_routes * (1.0 - hot_fraction);
        const double main_time = (shared_equiv + hot_routes) / main_bw;
        const double cold_time = cold_routes / cold_bw;
        return std::max(main_time, cold_time);
    };

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dp((size_t) total_hot_budget + 1, inf);
    std::vector<double> next((size_t) total_hot_budget + 1, inf);
    std::vector<std::vector<int16_t>> choice(
        (size_t) w.n_layer,
        std::vector<int16_t>((size_t) total_hot_budget + 1, -1));
    dp[0] = 0.0;

    for (int il = 0; il < w.n_layer; ++il) {
        std::fill(next.begin(), next.end(), inf);
        const int layers_left = w.n_layer - il - 1;
        for (int used = 0; used <= total_hot_budget; ++used) {
            if (!std::isfinite(dp[(size_t) used])) continue;
            for (int k = min_hot; k <= max_hot; ++k) {
                const int new_used = used + k;
                if (new_used > total_hot_budget) break;
                const int remaining = total_hot_budget - new_used;
                if (remaining < layers_left * min_hot ||
                    remaining > layers_left * max_hot) {
                    continue;
                }
                const double candidate = dp[(size_t) used] + layer_cost(il, k);
                if (candidate < next[(size_t) new_used]) {
                    next[(size_t) new_used] = candidate;
                    choice[(size_t) il][(size_t) new_used] = (int16_t) k;
                }
            }
        }
        dp.swap(next);
    }

    if (!std::isfinite(dp[(size_t) total_hot_budget])) {
        if (err) *err = "time-balanced placement optimizer found no feasible assignment";
        return false;
    }

    std::vector<int> hot_counts((size_t) w.n_layer, 0);
    int used = total_hot_budget;
    for (int il = w.n_layer - 1; il >= 0; --il) {
        const int k = choice[(size_t) il][(size_t) used];
        if (k < min_hot || k > max_hot) {
            if (err) *err = "time-balanced placement backtracking failed";
            return false;
        }
        hot_counts[(size_t) il] = k;
        used -= k;
    }

    double uniform_cost = 0.0;
    double balanced_cost = 0.0;
    for (int il = 0; il < w.n_layer; ++il) {
        uniform_cost += layer_cost(il, hot_per_layer);
        balanced_cost += layer_cost(il, hot_counts[(size_t) il]);
    }

    out = {};
    out.n_layer = w.n_layer;
    out.n_expert = w.n_expert;
    out.n_expert_used = w.n_expert_used;
    out.hot_counts = hot_counts;
    out.hot_expert_ids.resize((size_t) w.n_layer);
    out.total_hot = total_hot_budget;
    for (int il = 0; il < w.n_layer; ++il) {
        const int k = hot_counts[(size_t) il];
        auto & ids = out.hot_expert_ids[(size_t) il];
        ids.reserve((size_t) k);
        for (int rank = 0; rank < k; ++rank) {
            ids.push_back((int32_t) ranked[(size_t) il][(size_t) rank]);
        }
    }

    const auto [min_it, max_it] = std::minmax_element(
        hot_counts.begin(), hot_counts.end());
    std::fprintf(stderr,
        "[deepseek4] time-balanced placement: slots=%d layer_range=%d..%d "
        "routes=%d main/cold=%.2f/%.2f GB/s "
        "predicted_owner_reduction=%.2f%% counts=",
        total_hot_budget, *min_it, *max_it, active_routes, main_bw, cold_bw,
        uniform_cost > 0.0
            ? 100.0 * (1.0 - balanced_cost / uniform_cost)
            : 0.0);
    for (int il = 0; il < w.n_layer; ++il) {
        std::fprintf(stderr, "%s%d", il == 0 ? "[" : ",",
                     hot_counts[(size_t) il]);
    }
    std::fprintf(stderr, "]\n");
    return true;
}

static bool compute_ds4_hybrid_budget_info(const DeepSeek4Weights & w,
                                           int gpu,
                                           int max_ctx,
                                           Ds4HybridBudgetInfo & out,
                                           std::string * err) {
    out = {};
    ggml_backend_cuda_get_device_memory(gpu, &out.gpu_free, &out.gpu_total);
    if (out.gpu_total == 0) {
        if (err) *err = "could not query GPU memory";
        return false;
    }

    if (!compute_ds4_expert_memory_info(w, nullptr, out.mem, err)) {
        return false;
    }

    out.core_bytes = out.gpu_total - out.gpu_free;
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
    // Heterogeneous execution deliberately keeps routed experts split across
    // the R9700 and Strix. Only an explicit diagnostic override may request
    // the ~80-GiB monolithic HIP allocation.
    return env_flag_enabled("DFLASH_DS4_FORCE_FULL_LOAD");
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

    // HIP single-device launches should avoid the monolithic full-model load:
    // a managed ~80 GiB allocation can stall or be killed on integrated UMA
    // systems before we ever reach the existing OOM fallback path.
    if (target_backend == PlacementBackend::Hip &&
        requires_monolithic_model()) {
        std::fprintf(stderr,
                     "[deepseek4] HIP full-model load explicitly enabled\n");
        if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
            std::fprintf(stderr,
                         "[deepseek4] explicit HIP full-model load failed: %s\n",
                         cfg_.model_path);
            return false;
        }
    } else if (target_backend == PlacementBackend::Hip) {
        std::fprintf(stderr,
                     "[deepseek4] HIP target detected; using hybrid expert load path\n");
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
    if (draft_gpu != cfg_.device.gpu || separate_draft_stream) {
        spec_backend_ = ggml_backend_cuda_init(draft_gpu);
        if (!spec_backend_) {
            std::fprintf(stderr,
                         "[deepseek4] failed to initialize DSpark GPU %d\n",
                         draft_gpu);
            return false;
        }
        draft_backend = spec_backend_;
        const bool low_priority = separate_draft_stream &&
            env_flag_enabled("DFLASH_DS4_DRAFT_LOW_PRIORITY");
        const bool priority_configured = low_priority &&
            ggml_backend_cuda_set_low_priority_stream(spec_backend_);
        std::fprintf(stderr,
                     "[deepseek4] DSpark backend gpu=%d target_gpu=%d "
                     "separate_stream=%d low_priority=%d\n",
                     draft_gpu, cfg_.device.gpu,
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
    return start_spec_remote_drafter();
}

bool DeepSeek4Backend::start_spec_remote_drafter() {
    spec_remote_drafter_.reset();

    const char * ipc_bin = std::getenv("DFLASH_DS4_DRAFT_IPC_BIN");
    if (!ipc_bin || !*ipc_bin) return true;
    if (!spec_drafter_) return false;

    int ipc_gpu = 0;
    if (const char * gpu = std::getenv("DFLASH_DS4_DRAFT_IPC_GPU")) {
        ipc_gpu = std::max(0, std::atoi(gpu));
    }
    const char * ipc_work = std::getenv("DFLASH_DS4_DRAFT_IPC_WORK_DIR");
    auto remote = std::make_unique<DFlashDraftIpcClient>(
        w_.n_embd, spec_drafter_->block_size,
        spec_drafter_->n_target_layers);
    if (!remote->start(
            ipc_bin, spec_draft_path_, ipc_gpu, w_.n_swa,
            ipc_work && *ipc_work ? ipc_work : "",
            BackendIpcMode::DeepSeek4DSparkDraft)) {
        std::fprintf(stderr,
                     "[deepseek4] DSpark remote draft IPC failed; using local draft\n");
        return !env_flag_enabled("DFLASH_DS4_DRAFT_IPC_REQUIRED");
    }

    spec_remote_drafter_ = std::move(remote);
    std::fprintf(stderr,
                 "[deepseek4] DSpark remote draft IPC ENABLED gpu=%d\n",
                 ipc_gpu);
    return true;
}

void DeepSeek4Backend::release_spec_drafter(bool mark_parked) {
    // Stop the remote process before releasing the local metadata/fallback.
    spec_remote_drafter_.reset();
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

bool DeepSeek4Backend::init() {
    // The shared MMVQ/MMQ crossover defaults to q=3 for NVIDIA. On gfx1151,
    // DSpark q=4 is faster through MMVQ. Keep AR and other devices unchanged,
    // and preserve LUCE_MMVQ_MAX_NCOLS as an explicit override.
    configure_gfx1151_dspark_mmvq_default(cfg_.device.gpu);
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
    if (env_flag_enabled("DFLASH_DS4_TP_DYNAMIC_HOTSET") && !routing_stats_) {
        routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
        if (!routing_stats_->init(w_.n_layer, w_.n_expert, w_.n_expert_used)) {
            std::fprintf(stderr,
                         "[deepseek4] failed to initialize dynamic-hotset routing stats\n");
            return false;
        }
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] request-adaptive hotset enabled\n");
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
            const bool loaded = load_spec_drafter();
            if (!loaded && env_flag_enabled("DFLASH_DS4_DRAFT_IPC_REQUIRED")) {
                return false;
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

    if (ds4_inprocess_moe_tp_enabled()) {
        if (!expert_backend_ || !moe_hybrid_->materialized_cold_experts ||
            moe_hybrid_->cold_backend != expert_backend_) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] in-process expert backend is not ready\n");
            return false;
        }
        expert_runtime_.reset();
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] enabled mode=in-process local_gpu=%d "
                     "expert_gpu=%d local_experts=%d remote_experts=%d\n",
                     cfg_.device.gpu, ds4_moe_tp_gpu(cfg_.device.gpu),
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
                                                       std::string * err) const {
    Ds4HybridBudgetInfo budget;
    if (!compute_ds4_hybrid_budget_info(w, cfg_.device.gpu, max_ctx, budget, err)) {
        return false;
    }

    const bool all_cold = env_flag_enabled("DFLASH_DS4_MOE_TP_ALL_COLD");
    int active_routes = cfg_.expert_top_k;
    if (const char * raw = std::getenv("DFLASH_DS4_TOPK")) {
        const int env_routes = std::atoi(raw);
        if (env_routes > 0) active_routes = env_routes;
    }
    if (active_routes <= 0 || active_routes > w.n_expert_used) {
        active_routes = w.n_expert_used;
    }
    int hot_per_layer = all_cold ? 0 : budget.max_hot_per_layer;
    int shard_channels = 0;
    if (!all_cold && ds4_inprocess_moe_tp_enabled()) {
        if (const char * raw = std::getenv("DFLASH_DS4_TP_COLD_SHARD_CHANNELS")) {
            shard_channels = std::max(0, std::atoi(raw));
        }
    }
    if (shard_channels > 0) {
        if (shard_channels >= w.n_ff_exp || w.n_ff_exp <= 0) {
            if (err) *err = "invalid DFLASH_DS4_TP_COLD_SHARD_CHANNELS";
            return false;
        }
        // Every non-hot expert keeps a prefix shard on the main GPU.  Reserve
        // that prompt-independent footprint first, then spend the remainder
        // upgrading selected experts from a shard to their full tensors.
        const uint64_t shard_base =
            budget.mem.total_expert_bytes * (uint64_t) shard_channels /
            (uint64_t) w.n_ff_exp;
        const uint64_t upgrade_round =
            budget.mem.bytes_per_uniform_round *
            (uint64_t) (w.n_ff_exp - shard_channels) /
            (uint64_t) w.n_ff_exp;
        if (budget.expert_budget <= shard_base || upgrade_round == 0) {
            if (err) *err = "expert budget cannot fit the universal cold-expert shard";
            return false;
        }
        hot_per_layer = std::min(
            w.n_expert,
            (int) ((budget.expert_budget - shard_base) / upgrade_round));
        std::fprintf(stderr,
            "[deepseek4-tp] cold channel shard=%d/%d reserve=%.2f GiB full_hot/layer=%d\n",
            shard_channels, w.n_ff_exp, gib(shard_base), hot_per_layer);
    }
    if (all_cold) {
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] all routed experts assigned to the cold backend\n");
    }
    if (const char * profile_path = std::getenv("DFLASH_DS4_HOTNESS_CSV")) {
        if (*profile_path) {
            const bool time_balanced =
                env_flag_enabled("DFLASH_DS4_TP_TIME_BALANCED_PLACEMENT");
            const bool placed = time_balanced
                ? fill_time_balanced_profiled_hot_placement(
                    w, active_routes, hot_per_layer, profile_path, out, err)
                : fill_profiled_hot_placement(
                    w, hot_per_layer, profile_path, out, err);
            if (!placed) {
                return false;
            }
            std::fprintf(stderr,
                         "[deepseek4] hybrid placement profile=%s mode=%s\n",
                         profile_path,
                         time_balanced ? "time-balanced" : "uniform");
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

    std::fprintf(stderr,
                 "[deepseek4] hybrid placement: gpu_total=%.2f GiB gpu_free=%.2f GiB core=%.2f GiB kv=%.2f GiB warm=%.2f GiB safety=%.2f GiB expert_budget=%.2f GiB hot/layer=%d\n",
                 gib((uint64_t) budget.gpu_total),
                 gib((uint64_t) budget.gpu_free),
                 gib(budget.core_bytes),
                 gib(budget.kv_bytes),
                 gib(budget.warm_bytes),
                 gib(budget.safety_bytes),
                 gib(budget.expert_budget),
                 hot_per_layer);
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
    if (!compute_uniform_hybrid_placement(w_, max_ctx, moe_placement_, &err)) {
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

    auto hybrid = std::make_shared<MoeHybridStorage>();
    MoeHybridConfig hybrid_cfg = make_ds4_parent_worker_cfg(w_);
    const bool inprocess_tp =
        env_flag_enabled("DFLASH_DS4_MOE_TP") && ds4_inprocess_moe_tp_enabled();
    if (inprocess_tp) {
        const int expert_gpu = ds4_moe_tp_gpu(cfg_.device.gpu);
        if (expert_gpu == cfg_.device.gpu) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] in-process expert GPU must differ from local GPU\n");
            return false;
        }
        if (g_peer_access_opt_in) {
            const bool peer_ok = enable_peer_access_pair(cfg_.device.gpu, expert_gpu);
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] peer access GPU %d <-> GPU %d: %s\n",
                         cfg_.device.gpu, expert_gpu, peer_ok ? "enabled" : "unavailable");
        }
        expert_backend_ = ggml_backend_cuda_init(expert_gpu);
        if (!expert_backend_) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] failed to initialize in-process expert GPU %d\n",
                         expert_gpu);
            return false;
        }
        hybrid_cfg.materialize_cold_experts = true;
        hybrid_cfg.cold_expert_backend = MoeHybridColdBackend::Gpu;
    }
    if (!build_deepseek4_moe_hybrid_storage_from_file_with_mmap(
            cfg_.model_path, backend_, w_, moe_placement_, &hybrid_cfg,
            *hybrid, &err, expert_backend_)) {
        std::fprintf(stderr, "[deepseek4] failed to build hybrid expert storage: %s\n", err.c_str());
        if (expert_backend_) {
            ggml_backend_free(expert_backend_);
            expert_backend_ = nullptr;
        }
        return false;
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
            return false;
        }
        if (!stream_engine_.init(backend_, max_expert_bytes, &err)) {
            std::fprintf(stderr, "[deepseek4] failed to init cold-expert stream engine: %s\n",
                         err.c_str());
            return false;
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

bool DeepSeek4Backend::park(const std::string & what) {
    const bool want_draft = (what.empty() || what == "all" || what == "draft");
    const bool want_target = (what.empty() || what == "all" || what == "target");

    if (want_draft && spec_drafter_) {
        release_spec_drafter(/*mark_parked=*/true);
        std::printf("[deepseek4] DSpark drafter parked (VRAM released)\n");
        std::fflush(stdout);
    }
    if (!want_target || parked_) return true;

    maybe_save_routing_stats();
    for (int i = 0; i < PREFIX_SLOTS; ++i) {
        free_deepseek4_snapshot(snapshots_[i]);
    }
    last_logits_.clear();
    free_deepseek4_cache(cache_);
    expert_runtime_.reset();
    stream_engine_.destroy();
    moe_hybrid_.reset();
    if (expert_backend_) {
        ggml_backend_free(expert_backend_);
        expert_backend_ = nullptr;
    }
    moe_placement_ = {};
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

bool DeepSeek4Backend::unpark(const std::string & what) {
    const bool want_draft = (what.empty() || what == "all" || what == "draft");
    const bool want_target = (what.empty() || what == "all" || what == "target");

    if (want_target && parked_) {
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

int DeepSeek4Backend::do_prefill(const std::vector<int32_t> & tokens,
                                  const DaemonIO & io,
                                  int kv_offset) {
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
    const int chunk = !prefill_attention_mode_is_approximate(cfg_.prefill_mode)
        ? 1
        : std::max(1, std::min(requested_chunk,
                               layer_major_cap));
    int pos = kv_offset;
    // New sequence: clear the cache buffer so compressor state double-buffers
    // and compressed-KV rows start from zeros, exactly like a fresh server.
    // Without this, the first flush windows of a request pool over the
    // previous request's leftover state rows and outputs from the 2nd/3rd
    // request on can drift by a token or two.
    if (kv_offset == 0) {
        reset_deepseek4_cache(cache_);
    }
    last_logits_.clear();
    int spec_capture_from = n_total;
    if (spec_enabled_ && spec_drafter_) {
        const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
        if (kv_offset == 0 || n_total >= w_.n_swa || feat_row <= 0 ||
            spec_feat_window_.size() % (size_t) feat_row != 0) {
            spec_feat_window_.clear();
            spec_capture_from = std::max(0, n_total - w_.n_swa);
        } else {
            // Keep enough prior rows for the new prompt suffix, then append all
            // new rows. This bounds host capture storage at n_swa without
            // shifting a multi-megabyte feature window after every token.
            const size_t old_rows = spec_feat_window_.size() / (size_t) feat_row;
            const size_t keep_rows = (size_t) std::max(0, w_.n_swa - n_total);
            if (old_rows > keep_rows) {
                const size_t drop_floats = (old_rows - keep_rows) * (size_t) feat_row;
                const size_t keep_floats = keep_rows * (size_t) feat_row;
                std::memmove(spec_feat_window_.data(),
                             spec_feat_window_.data() + drop_floats,
                             keep_floats * sizeof(float));
                spec_feat_window_.resize(keep_floats);
            }
            spec_capture_from = 0;
        }
    }
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;

    for (int i = 0; i < n_total; i += chunk) {
        if (io.cancelled) return pos;

        const int n_tok = std::min(chunk, n_total - i);

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
        if (spec_enabled_ && spec_drafter_ && i + n_tok > spec_capture_from) {
            spec_hooks.capture_layer_ids = &spec_drafter_->capture_layer_ids;
            spec_hooks.capture_out = &spec_cap;
            hp = &spec_hooks;
        }
        if (moe_hybrid_ && (expert_runtime_.compute || expert_backend_)) {
            ok = deepseek4_step_layer_range(
                backend_, w_, cache_, hc_state,
                embed.data(), n_tok, pos,
                0, w_.n_layer, &logits,
                tokens.data() + i,
                timing ? &step_tel : nullptr,
                cfg_.prefill_mode != PrefillAttentionMode::Sparse, hp,
                moe_hybrid_.get(),
                expert_runtime_.compute ? &expert_runtime_ : nullptr,
                routing_stats_.get());
        } else if (moe_hybrid_) {
            ok = deepseek4_step(backend_, w_, cache_, embed.data(), n_tok, pos, logits,
                                moe_hybrid_.get(), tokens.data() + i,
                                &stream_engine_,
                                timing ? &step_tel : nullptr,
                                routing_stats_.get(),
                                hp,
                                expert_runtime_.compute ? &expert_runtime_ : nullptr);
        } else {
            ok = deepseek4_step_layer_range(backend_, w_, cache_, hc_state,
                                            embed.data(), n_tok, pos,
                                            0, w_.n_layer, &logits,
                                            tokens.data() + i,
                                            timing ? &step_tel : nullptr,
                                            cfg_.prefill_mode != PrefillAttentionMode::Sparse, hp);
        }
        if (ok && hp && !spec_cap.empty()) {
            const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
            const int first_capture = std::max(0, spec_capture_from - i);
            for (int t = first_capture; t < n_tok; ++t) {
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
    }
    if (timing) {
        log_step_tel("prefill", n_total, steps, elapsed_s(phase_t0), tel_acc);
    }
    return pos;
}

bool DeepSeek4Backend::do_decode(int committed, int n_gen,
                                  std::vector<int32_t> & out_tokens,
                                  const DaemonIO & io,
                                  const BudgetHook & budget_hook,
                                  bool * forced_close_out) {
    if (forced_close_out) *forced_close_out = false;
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;

    for (int generated = 0; generated < n_gen; generated++) {
        if (io.cancelled) break;

        // Budget hook: force-close if remaining budget hits threshold
        if (!budget_hook.close_token_ids.empty() &&
            (n_gen - generated) <= budget_hook.hard_limit_remaining) {
            // Inject close-tag tokens
            for (int32_t close_tok : budget_hook.close_token_ids) {
                out_tokens.push_back(close_tok);
                io.emit(close_tok);
                if (io.cancelled) break;
            }
            if (forced_close_out) *forced_close_out = true;
            break;
        }

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
                    backend_, w_, cache_, hc_state,
                    embed.data(), 1, pos,
                    0, w_.n_layer, &logits,
                    &tok_to_eval,
                    timing ? &step_tel : nullptr,
                    /*allow_decode_graph_reuse=*/true, nullptr,
                    moe_hybrid_.get(),
                    expert_runtime_.compute ? &expert_runtime_ : nullptr,
                    routing_stats_.get());
            } else if (moe_hybrid_) {
                ok = deepseek4_step(backend_, w_, cache_, embed.data(), 1,
                                    pos, logits,
                                    moe_hybrid_.get(), &tok_to_eval,
                                    &stream_engine_,
                                    timing ? &step_tel : nullptr,
                                    routing_stats_.get(),
                                    nullptr,
                                    expert_runtime_.compute ? &expert_runtime_ : nullptr);
            } else {
                std::vector<float> hc_state;
                ok = deepseek4_step_layer_range(backend_, w_, cache_, hc_state,
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

        // Sample (argmax for now)
        int32_t next_token = 0;
        {
            const auto sample_t0 = Clock::now();
            float max_val = logits[0];
            for (int i = 1; i < w_.n_vocab; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    next_token = i;
                }
            }
            if (timing) tel_acc.sample_us += elapsed_us(sample_t0, Clock::now());
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
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    auto t0 = Clock::now();

    const bool dynamic_hotset =
        env_flag_enabled("DFLASH_DS4_TP_DYNAMIC_HOTSET");
    if (dynamic_hotset) {
        if (!moe_hybrid_ || !expert_backend_ || !routing_stats_) {
            result.fail(GenerateErrorCode::BackendSpecific,
                        "dynamic hotset requires in-process heterogeneous MoE TP");
            return result;
        }
        // A previous request may have installed its own placement.  Restore
        // the calibrated general placement before observing this prompt, then
        // collect a request-local route histogram from q1 prefill only.
        std::string err;
        if (!moe_hybrid_reassign_hot_experts_from_mmap(
                *moe_hybrid_, backend_, moe_placement_.hot_expert_ids, &err)) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] dynamic-hotset restore failed: %s\n",
                         err.c_str());
            result.fail(GenerateErrorCode::BackendSpecific,
                        "dynamic hotset restore");
            return result;
        }
        if (!routing_stats_->init(w_.n_layer, w_.n_expert,
                                  w_.n_expert_used)) {
            result.fail(GenerateErrorCode::BackendSpecific,
                        "dynamic hotset stats reset");
            return result;
        }
    }

    // Prefill
    int committed = do_prefill(req.prompt, out_io);
    if (committed < 0) {
        result.fail(GenerateErrorCode::PrefillFailed);
        return result;
    }
    result.prefill_s = elapsed_s(t0);

    if (out_io.cancelled) {
        result.succeed();
        maybe_save_routing_stats();
        return result;
    }

    if (dynamic_hotset) {
        const auto adapt_t0 = Clock::now();
        std::vector<std::vector<int32_t>> request_hot(
            (size_t) w_.n_layer);
        uint64_t covered = 0;
        uint64_t total = 0;
        for (int il = 0; il < w_.n_layer; ++il) {
            const int slots = moe_hybrid_->layers[(size_t) il].hot_active;
            std::vector<int> ranked = routing_stats_->hot_experts(il, slots);
            if ((int) ranked.size() != slots) {
                result.fail(GenerateErrorCode::BackendSpecific,
                            "dynamic hotset empty prefill routes");
                return result;
            }
            auto & ids = request_hot[(size_t) il];
            ids.assign(ranked.begin(), ranked.end());
            total += routing_stats_->layer_totals[(size_t) il];
            for (int expert : ranked) {
                covered += routing_stats_->count(il, expert);
            }
        }
        std::string err;
        if (!moe_hybrid_reassign_hot_experts_from_mmap(
                *moe_hybrid_, backend_, request_hot, &err)) {
            std::fprintf(stderr,
                         "[deepseek4-moe-tp] dynamic-hotset install failed: %s\n",
                         err.c_str());
            result.fail(GenerateErrorCode::BackendSpecific,
                        "dynamic hotset install");
            return result;
        }
        const double adapt_s = elapsed_s(adapt_t0);
        result.prefill_s += adapt_s;
        std::fprintf(stderr,
                     "[deepseek4-moe-tp] dynamic-hotset prompt_routes=%" PRIu64
                     " coverage=%.3f install=%.3fs\n",
                     total, total ? (double) covered / (double) total : 0.0,
                     adapt_s);
    }

    if (req.n_gen <= 0) {
        result.succeed();
        maybe_save_routing_stats();
        return result;
    }

    // Decode
    auto t1 = Clock::now();
    const bool budget_requires_ar = !req.budget_hook.close_token_ids.empty();
    if (spec_enabled_ && spec_drafter_ && req.n_gen > 0 &&
        !req.force_ar_decode && !budget_requires_ar) {
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
        if (!out_io.cancelled && !deepseek4_is_eos_tok(seed, w_) && req.n_gen > 1) {
            const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
            const int win_len = feat_row > 0 ? (int) (spec_feat_window_.size() / feat_row) : 0;
            std::vector<int32_t> spec_toks;
            spec_ran = true;
            if (!run_deepseek4_dspark_spec_decode(
                    backend_, w_, cache_, *spec_drafter_, committed, seed,
                    req.n_gen - 1,
                    win_len > 0 ? spec_feat_window_.data() : nullptr, win_len,
                    spec_toks, &accept_rate,
                    [&out_io](int32_t tok) {
                        if (out_io.cancelled) return false;
                        out_io.emit(tok);
                        return !out_io.cancelled;
                    },
                    spec_remote_drafter_.get(),
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
    if (!do_decode(committed, req.n_gen, gen_tokens, out_io,
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
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    // TODO: Implement snapshot save (copy KV cache + HC state to CPU)
    return false;
}

void DeepSeek4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_deepseek4_snapshot(snapshots_[slot]);
}

bool DeepSeek4Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return snapshots_[slot].ctx != nullptr;
}

int DeepSeek4Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return snapshots_[slot].cur_pos;
}

GenerateResult DeepSeek4Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    // TODO: Implement snapshot restore + generate
    (void)slot;
    return generate_impl(req, io);
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
        free_deepseek4_snapshot(snapshots_[i]);
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
    free_deepseek4_weights(w_);
    if (snap_backend_) { ggml_backend_free(snap_backend_); snap_backend_ = nullptr; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

}  // namespace dflash::common
