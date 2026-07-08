// DeepSeek4Backend implementation — AR-only decode, chunked prefill.

#include "deepseek4_backend.h"
#include "deepseek4_internal.h"
#include "common/sampler.h"
#include "../common/expert_split_target_config.h"
#include "../common/moe_hybrid_types.h"

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

static double gib(uint64_t bytes) {
    return (double) bytes / 1024.0 / 1024.0 / 1024.0;
}

static int env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    return std::atoi(value);
}

static double env_double(const char * name, double fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    return std::atof(value);
}

static const char * env_str(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return (value && value[0]) ? value : fallback;
}

static int env_pct(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 100) {
        return fallback;
    }
    return (int) parsed;
}

static uint64_t apply_pct_bytes(uint64_t total_bytes, int pct) {
    if (pct >= 100) return total_bytes;
    return (total_bytes / 100ULL) * (uint64_t) pct +
           ((total_bytes % 100ULL) * (uint64_t) pct) / 100ULL;
}

static bool backend_is_cuda(ggml_backend_t backend);
static bool backend_is_hip(ggml_backend_t backend);
static bool backend_is_gpu(ggml_backend_t backend);

}  // namespace

int deepseek4_expert_split_prefill_chunk_limit_from_memory(
        int requested_chunk,
        bool expert_split_enabled,
        bool parent_is_gpu,
        const DeepSeek4ExpertSplitGpuMemoryInfo & memory) {
    int chunk = std::max(1, requested_chunk);
    if (!expert_split_enabled || !parent_is_gpu) {
        return chunk;
    }
    if (memory.total_bytes == 0) {
        return std::min(chunk, 64);
    }

    const uint64_t gib1 = 1024ULL * 1024ULL * 1024ULL;
    const uint64_t gib24 = 24ULL * gib1;
    const uint64_t gib48 = 48ULL * gib1;
    const uint64_t gib80 = 80ULL * gib1;
    const uint64_t total = memory.total_bytes;
    const uint64_t free = memory.free_bytes;
    if (total <= gib24 + 512ULL * 1024ULL * 1024ULL) {
        if (free >= 12ULL * gib1) {
            return std::min(chunk, 256);
        }
        if (free >= 6ULL * gib1) {
            return std::min(chunk, 128);
        }
        return std::min(chunk, 64);
    }
    if (total <= gib48 + 512ULL * 1024ULL * 1024ULL) {
        if (free >= 12ULL * gib1) {
            return std::min(chunk, 256);
        }
        return std::min(chunk, 128);
    }
    if (total >= gib80 && free >= 24ULL * gib1) {
        return std::min(chunk, 512);
    }
    return std::min(chunk, 256);
}

bool deepseek4_expert_split_should_disable_cached_decode_from_memory(
        bool expert_split_enabled,
        bool parent_is_gpu,
        bool parent_is_cuda,
        bool parent_is_hip,
        const DeepSeek4ExpertSplitGpuMemoryInfo & memory) {
    if (!expert_split_enabled || !parent_is_gpu) {
        return false;
    }
    if (parent_is_hip) {
        return false;
    }
    if (!parent_is_cuda) {
        return true;
    }
    if (memory.total_bytes == 0) {
        return true;
    }

    const uint64_t gib1 = 1024ULL * 1024ULL * 1024ULL;
    const uint64_t gib24 = 24ULL * gib1;
    if (memory.total_bytes > gib24 + 512ULL * 1024ULL * 1024ULL) {
        return false;
    }
    return memory.free_bytes < 3ULL * gib1;
}

bool deepseek4_expert_split_requires_parent_cuda_graph_disable(
        bool expert_split_enabled,
        bool parent_is_cuda,
        const std::vector<ExpertSplitComputeTargetRuntime> & targets) {
    if (!expert_split_enabled || !parent_is_cuda || targets.size() <= 1) {
        return false;
    }
    for (size_t i = 1; i < targets.size(); ++i) {
        const ExpertSplitComputeTargetRuntime & target = targets[i];
        if (target.placement.total_hot <= 0) {
            continue;
        }
        if (target.target.backend.empty() || target.target.backend == "cpu") {
            continue;
        }
        return true;
    }
    return false;
}

uint64_t deepseek4_expert_split_budget_from_memory(
        const DeepSeek4ExpertSplitBudgetMemoryInfo & memory) {
    const uint64_t fixed_bytes =
        memory.core_bytes +
        memory.kv_bytes +
        memory.warm_bytes +
        memory.safety_bytes;
    if (memory.total_bytes <= fixed_bytes) {
        return 0;
    }

    uint64_t budget_bytes = memory.total_bytes - fixed_bytes;
    if (memory.parent_is_igpu && memory.host_budget_cap_bytes > 0) {
        budget_bytes = std::min(budget_bytes, memory.host_budget_cap_bytes);
    }
    if (memory.total_expert_bytes > 0 &&
        budget_bytes > memory.total_expert_bytes) {
        budget_bytes = memory.total_expert_bytes;
    }
    return budget_bytes;
}

uint64_t deepseek4_expert_split_primary_capacity_for_targets(
        uint64_t expert_budget_bytes,
        uint64_t hot_budget_bytes,
        size_t configured_targets) {
    if (configured_targets <= 1 || hot_budget_bytes == 0) {
        return expert_budget_bytes;
    }
    if (expert_budget_bytes == 0) {
        return 0;
    }
    return std::min(expert_budget_bytes, hot_budget_bytes);
}

uint64_t deepseek4_expert_split_non_cpu_capacity_bytes(
        const std::vector<ExpertSplitTarget> & targets) {
    uint64_t total_bytes = 0;
    for (const ExpertSplitTarget & target : targets) {
        if (target.backend == "cpu") {
            continue;
        }
        if (target.unlimited) {
            return std::numeric_limits<uint64_t>::max();
        }
        const uint64_t usable = target.usable_bytes();
        if (usable == 0) {
            continue;
        }
        if (std::numeric_limits<uint64_t>::max() - total_bytes < usable) {
            return std::numeric_limits<uint64_t>::max();
        }
        total_bytes += usable;
    }
    return total_bytes;
}

uint64_t deepseek4_expert_split_effective_budget_for_targets(
        uint64_t requested_budget_bytes,
        uint64_t total_expert_bytes,
        const std::vector<ExpertSplitTarget> & targets) {
    if (requested_budget_bytes == 0 || total_expert_bytes == 0 || targets.empty()) {
        return requested_budget_bytes;
    }

    const bool has_cpu_target = std::any_of(
        targets.begin(), targets.end(),
        [](const ExpertSplitTarget & target) {
            return target.backend == "cpu";
        });
    if (!has_cpu_target) {
        return requested_budget_bytes;
    }

    const uint64_t non_cpu_capacity =
        deepseek4_expert_split_non_cpu_capacity_bytes(targets);
    if (non_cpu_capacity == 0) {
        return requested_budget_bytes;
    }

    const uint64_t promoted_budget =
        std::min(total_expert_bytes, non_cpu_capacity);
    return std::max(requested_budget_bytes, promoted_budget);
}

uint64_t deepseek4_expert_split_effective_hot_budget(
        uint64_t expert_budget_bytes,
        uint64_t hot_budget_bytes,
        size_t configured_targets) {
    if (expert_budget_bytes == 0) {
        return 0;
    }
    if (configured_targets > 1) {
        return expert_budget_bytes;
    }
    if (hot_budget_bytes == 0) {
        return expert_budget_bytes;
    }
    return std::min(expert_budget_bytes, hot_budget_bytes);
}

bool deepseek4_expert_split_hash_ids_to_hotness_counts(
        const int32_t * expert_ids,
        int n_token_ids,
        int n_expert_used,
        int n_expert,
        std::vector<uint64_t> & out_counts) {
    out_counts.clear();
    if (!expert_ids ||
        n_token_ids <= 0 ||
        n_expert_used <= 0 ||
        n_expert <= 0) {
        return false;
    }

    out_counts.assign((size_t) n_expert, 0);
    const size_t total_ids =
        (size_t) n_token_ids * (size_t) n_expert_used;
    for (size_t i = 0; i < total_ids; ++i) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || expert >= n_expert) {
            out_counts.clear();
            return false;
        }
        out_counts[(size_t) expert] += 1;
    }
    return true;
}

bool deepseek4_expert_split_route_bias_to_hotness_counts(
        const float * route_bias,
        int n_expert,
        uint64_t layer_total,
        std::vector<uint64_t> & out_counts) {
    out_counts.clear();
    if (!route_bias || n_expert <= 0 || layer_total == 0) {
        return false;
    }

    const uint64_t minimum_total =
        std::max<uint64_t>(layer_total, (uint64_t) n_expert);
    float max_bias = -INFINITY;
    bool saw_finite = false;
    for (int expert = 0; expert < n_expert; ++expert) {
        const float bias = route_bias[expert];
        if (!std::isfinite(bias)) {
            continue;
        }
        max_bias = saw_finite ? std::max(max_bias, bias) : bias;
        saw_finite = true;
    }
    if (!saw_finite) {
        return false;
    }

    out_counts.assign((size_t) n_expert, 1);
    uint64_t remaining = minimum_total - (uint64_t) n_expert;
    if (remaining == 0) {
        return true;
    }

    std::vector<long double> weights((size_t) n_expert, 0.0L);
    long double weight_sum = 0.0L;
    for (int expert = 0; expert < n_expert; ++expert) {
        double delta = (double) route_bias[expert] - (double) max_bias;
        if (!std::isfinite(delta)) {
            delta = -16.0;
        }
        delta = std::max(-16.0, std::min(0.0, delta));
        const long double weight = std::exp(delta);
        weights[(size_t) expert] = weight;
        weight_sum += weight;
    }
    if (!(weight_sum > 0.0L)) {
        out_counts.clear();
        return false;
    }

    struct ShareRemainder {
        int expert = 0;
        long double remainder = 0.0L;
    };
    std::vector<ShareRemainder> remainders;
    remainders.reserve((size_t) n_expert);

    uint64_t assigned = 0;
    for (int expert = 0; expert < n_expert; ++expert) {
        const long double raw =
            ((long double) remaining * weights[(size_t) expert]) / weight_sum;
        const uint64_t extra = (uint64_t) std::floor(raw);
        out_counts[(size_t) expert] += extra;
        assigned += extra;
        remainders.push_back({expert, raw - (long double) extra});
    }

    std::stable_sort(remainders.begin(), remainders.end(),
        [](const ShareRemainder & lhs, const ShareRemainder & rhs) {
            if (lhs.remainder != rhs.remainder) {
                return lhs.remainder > rhs.remainder;
            }
            return lhs.expert < rhs.expert;
        });
    for (uint64_t i = assigned; i < remaining; ++i) {
        out_counts[(size_t) remainders[(size_t) (i - assigned)].expert] += 1;
    }

    return true;
}

int deepseek4_expert_split_effective_cache_slots(
        int requested_cache_slots,
        size_t configured_targets) {
    if (configured_targets > 1) {
        return 0;
    }
    return std::max(0, requested_cache_slots);
}

static uint64_t ceil_div_u64(uint64_t value, uint64_t denom) {
    if (denom == 0) {
        return 0;
    }
    return value / denom + ((value % denom) != 0 ? 1ULL : 0ULL);
}

DeepSeek4ExpertSplitHotCachePlan deepseek4_expert_split_hot_cache_plan(
        uint64_t expert_budget_bytes,
        uint64_t total_expert_bytes,
        int n_expert,
        int n_expert_used,
        bool parent_is_igpu,
        uint64_t igpu_host_cap_bytes,
        uint64_t igpu_free_bytes) {
    DeepSeek4ExpertSplitHotCachePlan plan;
    plan.hot_bytes = expert_budget_bytes;
    if (expert_budget_bytes == 0 || total_expert_bytes == 0 || n_expert <= 0) {
        return plan;
    }

    const uint64_t bytes_per_slot =
        total_expert_bytes / (uint64_t) n_expert;
    if (bytes_per_slot == 0) {
        return plan;
    }

    const int min_hot_slots =
        std::max(0, std::min(n_expert, n_expert_used));
    const uint64_t min_hot_bytes =
        bytes_per_slot * (uint64_t) min_hot_slots;
    if (expert_budget_bytes <= min_hot_bytes) {
        return plan;
    }

    const int max_cache_slots = (int) std::min<uint64_t>(
        (expert_budget_bytes - min_hot_bytes) / bytes_per_slot,
        (uint64_t) std::max(0, n_expert - min_hot_slots));
    if (max_cache_slots <= 0) {
        return plan;
    }

    int requested_slots = 0;
    const bool explicit_slots =
        std::getenv("DFLASH_DEEPSEEK4_CACHE_SLOTS") != nullptr;
    if (const char * raw = std::getenv("DFLASH_DEEPSEEK4_CACHE_SLOTS")) {
        requested_slots = std::max(0, std::atoi(raw));
    } else {
        auto spark = spark_budget_split(
            expert_budget_bytes,
            total_expert_bytes,
            n_expert,
            /*core_kv_safety=*/0,
            /*target_bytes=*/0);
        requested_slots = std::max(0, spark.cache_slots);

        if (parent_is_igpu) {
            const int reserve_pct = env_pct(
                "DFLASH_DEEPSEEK4_CACHE_RESERVE_PCT", 20);
            uint64_t reserve_bytes =
                (expert_budget_bytes / 100ULL) * (uint64_t) reserve_pct +
                ((expert_budget_bytes % 100ULL) *
                 (uint64_t) reserve_pct) / 100ULL;
            if (const char * raw =
                    std::getenv("DFLASH_DEEPSEEK4_CACHE_RESERVE_MB")) {
                const long long reserve_mb = std::atoll(raw);
                if (reserve_mb > 0) {
                    reserve_bytes =
                        (uint64_t) reserve_mb * 1024ULL * 1024ULL;
                }
            } else {
                const int reserve_cap_mb = env_int(
                    "DFLASH_DEEPSEEK4_CACHE_RESERVE_CAP_MB", 16384);
                if (reserve_cap_mb > 0) {
                    const uint64_t reserve_cap_bytes =
                        (uint64_t) reserve_cap_mb * 1024ULL * 1024ULL;
                    reserve_bytes =
                        std::min(reserve_bytes, reserve_cap_bytes);
                }
            }
            const int min_cache_slots = std::min(
                n_expert,
                std::max(n_expert_used * 2, 12));
            if (reserve_bytes > 0) {
                requested_slots = std::max(
                    requested_slots,
                    (int) (reserve_bytes / bytes_per_slot));
            }
            requested_slots = std::max(requested_slots, min_cache_slots);
        }
    }

    if (parent_is_igpu && !explicit_slots) {
        uint64_t pinned_hot_cap = expert_budget_bytes;
        if (igpu_host_cap_bytes > 0) {
            pinned_hot_cap = std::min(pinned_hot_cap, igpu_host_cap_bytes);
        }
        if (igpu_free_bytes > 0) {
            pinned_hot_cap = std::min(pinned_hot_cap, igpu_free_bytes);
        }
        pinned_hot_cap = std::min(
            pinned_hot_cap,
            apply_pct_bytes(
                expert_budget_bytes,
                env_pct("DFLASH_DEEPSEEK4_IGPU_BOOTSTRAP_CAP_PCT", 80)));
        pinned_hot_cap = std::max(pinned_hot_cap, min_hot_bytes);
        if (expert_budget_bytes > pinned_hot_cap) {
            const uint64_t extra_slots = ceil_div_u64(
                expert_budget_bytes - pinned_hot_cap, bytes_per_slot);
            requested_slots = std::max(
                requested_slots,
                (int) std::min<uint64_t>(extra_slots, (uint64_t) max_cache_slots));
        }
    }

    plan.cache_slots = std::clamp(requested_slots, 0, max_cache_slots);
    plan.hot_bytes =
        expert_budget_bytes -
        (uint64_t) plan.cache_slots * bytes_per_slot;
    return plan;
}

namespace {

static int resolve_ds4_expert_split_prefill_chunk(
        int requested_chunk,
        bool expert_split_enabled,
        PlacementBackend device_backend,
        ggml_backend_t backend,
        int gpu_index) {
    int chunk = std::max(1, requested_chunk);
    if (!expert_split_enabled || !backend_is_gpu(backend)) {
        return chunk;
    }

    const int env_cap = env_int("DFLASH_EXPERT_SPLIT_PREFILL_CHUNK_MAX", 0);
    if (env_cap > 0) {
        return std::min(chunk, env_cap);
    }

    uint64_t gpu_free = 0;
    uint64_t gpu_total = 0;
    if (!query_expert_split_backend_memory(
            device_backend, gpu_index, gpu_free, gpu_total)) {
        return std::min(chunk, 64);
    }
    return dflash::common::deepseek4_expert_split_prefill_chunk_limit_from_memory(
        chunk,
        expert_split_enabled,
        /*parent_is_gpu=*/true,
        DeepSeek4ExpertSplitGpuMemoryInfo{gpu_free, gpu_total});
}

static bool should_disable_ds4_expert_split_cached_decode(
        bool expert_split_enabled,
        PlacementBackend device_backend,
        ggml_backend_t backend,
        int gpu_index) {
    if (!expert_split_enabled || !backend_is_gpu(backend)) {
        return false;
    }
    if (env_flag_enabled("DFLASH_DS4_FORCE_MULTI_TARGET_DECODE_CACHE")) {
        return false;
    }
    if (env_flag_enabled("DFLASH_DS4_DISABLE_MULTI_TARGET_DECODE_CACHE")) {
        return true;
    }
    if (backend_is_hip(backend)) {
        return false;
    }
    if (!backend_is_cuda(backend)) {
        return true;
    }

    uint64_t gpu_free = 0;
    uint64_t gpu_total = 0;
    if (!query_expert_split_backend_memory(
            device_backend, gpu_index, gpu_free, gpu_total)) {
        return true;
    }
    return dflash::common::deepseek4_expert_split_should_disable_cached_decode_from_memory(
        expert_split_enabled,
        /*parent_is_gpu=*/true,
        /*parent_is_cuda=*/true,
        /*parent_is_hip=*/false,
        DeepSeek4ExpertSplitGpuMemoryInfo{gpu_free, gpu_total});
}

static PlacementBackend resolved_device_backend(const DevicePlacement & device) {
    return device.backend == PlacementBackend::Auto
        ? compiled_placement_backend()
        : device.backend;
}

static bool backend_is_cuda(ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);
    return name && std::strstr(name, "CUDA") != nullptr;
}

static bool backend_is_hip(ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);
    return name &&
        (std::strstr(name, "HIP") != nullptr ||
         std::strstr(name, "ROCm") != nullptr);
}

static bool backend_is_gpu(ggml_backend_t backend) {
    return backend_is_cuda(backend) || backend_is_hip(backend);
}

static bool resolve_ds4_expert_split_targets_from_env(
        uint64_t primary_capacity_bytes,
        std::vector<ExpertSplitTarget> & out,
        std::string * err) {
    return dflash::common::resolve_expert_split_targets_from_env(
        "DFLASH_DEEPSEEK4_EXPERT_TARGETS",
        "DFLASH_DEEPSEEK4_EXPERT_TARGET_CAPS",
        primary_capacity_bytes,
        out,
        err);
}

static void add_step_tel(DeepSeek4StepTelemetry & dst, const DeepSeek4StepTelemetry & src) {
    dst.total_us += src.total_us;
    dst.embed_us += src.embed_us;
    dst.full_graph_build_us += src.full_graph_build_us;
    dst.full_graph_alloc_us += src.full_graph_alloc_us;
    dst.full_graph_set_us += src.full_graph_set_us;
    dst.full_graph_compute_us += src.full_graph_compute_us;
    dst.full_graph_read_us += src.full_graph_read_us;
    dst.hc_pre_attn_us += src.hc_pre_attn_us;
    dst.hc_pre_build_us += src.hc_pre_build_us;
    dst.hc_pre_input_us += src.hc_pre_input_us;
    dst.hc_pre_compute_us += src.hc_pre_compute_us;
    dst.attn_build_us += src.attn_build_us;
    dst.attn_compute_us += src.attn_compute_us;
    dst.attn_read_us += src.attn_read_us;
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
    dst.ffn_cache_promote_us += src.ffn_cache_promote_us;
    dst.ffn_hot_graph_builds += src.ffn_hot_graph_builds;
    dst.ffn_hot_graph_hits += src.ffn_hot_graph_hits;
    dst.ffn_cold_graph_builds += src.ffn_cold_graph_builds;
    dst.ffn_cold_graph_hits += src.ffn_cold_graph_hits;
    dst.worker_us += src.worker_us;
    dst.worker_parent_write_us += src.worker_parent_write_us;
    dst.worker_parent_wait_us += src.worker_parent_wait_us;
    dst.worker_parent_read_us += src.worker_parent_read_us;
    dst.worker_request_read_us += src.worker_request_read_us;
    dst.worker_partition_us += src.worker_partition_us;
    dst.worker_resident_eval_us += src.worker_resident_eval_us;
    dst.worker_miss_build_us += src.worker_miss_build_us;
    dst.worker_miss_eval_us += src.worker_miss_eval_us;
    dst.worker_request_bytes += src.worker_request_bytes;
    dst.worker_response_bytes += src.worker_response_bytes;
    dst.worker_hot_graph_builds += src.worker_hot_graph_builds;
    dst.worker_hot_graph_hits += src.worker_hot_graph_hits;
    dst.worker_cold_graph_builds += src.worker_cold_graph_builds;
    dst.worker_cold_graph_hits += src.worker_cold_graph_hits;
    dst.worker_hot_graph_build_us += src.worker_hot_graph_build_us;
    dst.worker_hot_input_us += src.worker_hot_input_us;
    dst.worker_hot_compute_us += src.worker_hot_compute_us;
    dst.worker_hot_read_us += src.worker_hot_read_us;
    dst.worker_cold_graph_build_us += src.worker_cold_graph_build_us;
    dst.worker_cold_input_us += src.worker_cold_input_us;
    dst.worker_cold_compute_us += src.worker_cold_compute_us;
    dst.worker_cold_read_us += src.worker_cold_read_us;
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
        "step=%.1fms embed=%.1fms full_build=%.1fms full_alloc=%.1fms full_set=%.1fms "
        "full_compute=%.1fms full_read=%.1fms "
        "attn_build=%.1fms attn_compute=%.1fms attn_read=%.1fms "
        "ffn_build=%.1fms ffn_compute=%.1fms ffn_read=%.1fms "
        "route_build=%.1fms route_compute=%.1fms route_read=%.1fms route_select=%.1fms "
        "ffn=%.1fms hot=%.1fms cold=%.1fms combine=%.1fms partition=%.1fms cache=%.1fms worker=%.1fms "
        "ffn_hot_graph_build=%llu ffn_hot_graph_hit=%llu ffn_cold_graph_build=%llu ffn_cold_graph_hit=%llu "
        "worker_write=%.1fms worker_wait=%.1fms worker_read=%.1fms worker_req_read=%.1fms "
        "worker_part=%.1fms worker_resident=%.1fms worker_miss_build=%.1fms worker_miss=%.1fms "
        "worker_hot_graph_build=%llu worker_hot_graph_hit=%llu worker_cold_graph_build=%llu worker_cold_graph_hit=%llu "
        "worker_hot_build=%.1fms worker_hot_input=%.1fms worker_hot_compute=%.1fms worker_hot_read=%.1fms "
        "worker_cold_build=%.1fms worker_cold_input=%.1fms worker_cold_compute=%.1fms worker_cold_read=%.1fms "
        "worker_req_kib=%.1f worker_resp_kib=%.1f "
        "hc_pre=%.1fms hc_pre_build=%.1fms hc_pre_input=%.1fms hc_pre_compute=%.1fms "
        "hc_post=%.1fms output=%.1fms sample=%.1fms emit=%.1fms "
        "hot_sel=%d cold_sel=%d\n",
        phase, tokens, steps, wall_s, tok_s,
        ms(t.total_us), ms(t.embed_us),
        ms(t.full_graph_build_us), ms(t.full_graph_alloc_us), ms(t.full_graph_set_us),
        ms(t.full_graph_compute_us), ms(t.full_graph_read_us),
        ms(t.attn_build_us), ms(t.attn_compute_us), ms(t.attn_read_us),
        ms(t.ffn_build_us), ms(t.ffn_compute_us), ms(t.ffn_read_us),
        ms(t.route_build_us), ms(t.route_compute_us), ms(t.route_read_us), ms(t.route_select_us),
        ms(t.ffn_eval_us), ms(t.ffn_hot_us), ms(t.ffn_cold_us), ms(t.ffn_combine_us),
        ms(t.ffn_partition_us), ms(t.ffn_cache_promote_us), ms(t.worker_us),
        (unsigned long long)t.ffn_hot_graph_builds, (unsigned long long)t.ffn_hot_graph_hits,
        (unsigned long long)t.ffn_cold_graph_builds, (unsigned long long)t.ffn_cold_graph_hits,
        ms(t.worker_parent_write_us), ms(t.worker_parent_wait_us), ms(t.worker_parent_read_us),
        ms(t.worker_request_read_us), ms(t.worker_partition_us), ms(t.worker_resident_eval_us),
        ms(t.worker_miss_build_us), ms(t.worker_miss_eval_us),
        (unsigned long long)t.worker_hot_graph_builds, (unsigned long long)t.worker_hot_graph_hits,
        (unsigned long long)t.worker_cold_graph_builds, (unsigned long long)t.worker_cold_graph_hits,
        ms(t.worker_hot_graph_build_us), ms(t.worker_hot_input_us),
        ms(t.worker_hot_compute_us), ms(t.worker_hot_read_us),
        ms(t.worker_cold_graph_build_us), ms(t.worker_cold_input_us),
        ms(t.worker_cold_compute_us), ms(t.worker_cold_read_us),
        (double)t.worker_request_bytes / 1024.0,
        (double)t.worker_response_bytes / 1024.0,
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
    bool parent_is_igpu = false;
    uint64_t host_available_bytes = 0;
    uint64_t host_budget_cap_bytes = 0;
    uint64_t core_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t warm_bytes = 256ULL * 1024 * 1024;
    uint64_t safety_bytes = 512ULL * 1024 * 1024;
    uint64_t auto_expert_budget = 0;
    uint64_t expert_budget = 0;
    uint64_t hot_budget = 0;
    int cache_slots = 0;
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

static bool compute_ds4_hybrid_budget_info(const DeepSeek4Weights & w,
                                           PlacementBackend device_backend,
                                           int gpu,
                                           int max_ctx,
                                           Ds4HybridBudgetInfo & out,
                                           std::string * err) {
    out = {};
    ExpertSplitBackendMemoryInfo memory_info;
    if (!query_expert_split_backend_memory_info(
            device_backend, gpu, memory_info) ||
        memory_info.total_bytes == 0) {
        if (err) *err = "could not query GPU memory";
        return false;
    }
    out.gpu_free = (size_t) memory_info.free_bytes;
    out.gpu_total = (size_t) memory_info.total_bytes;
    out.parent_is_igpu =
        memory_info.device_type == GGML_BACKEND_DEVICE_TYPE_IGPU;

    if (!compute_ds4_expert_memory_info(w, nullptr, out.mem, err)) {
        return false;
    }

    out.core_bytes =
        moe_hybrid_core_bytes_from_memory("deepseek4", out.gpu_free, out.gpu_total);
    out.kv_bytes = estimate_ds4_cache_bytes(w, max_ctx);
    ExpertSplitIgpuHostBudgetInfo host_budget;
    if (out.parent_is_igpu) {
        (void) query_expert_split_igpu_host_budget_info(host_budget);
        out.host_available_bytes = host_budget.available_bytes;
        out.host_budget_cap_bytes = host_budget.cap_bytes;
    }

    out.auto_expert_budget = deepseek4_expert_split_budget_from_memory(
        DeepSeek4ExpertSplitBudgetMemoryInfo{
            (uint64_t) out.gpu_total,
            (uint64_t) out.gpu_free,
            out.core_bytes,
            out.kv_bytes,
            out.warm_bytes,
            out.safety_bytes,
            out.mem.total_expert_bytes,
            out.parent_is_igpu,
            out.host_available_bytes,
            out.host_budget_cap_bytes,
        });
    out.expert_budget = out.auto_expert_budget;
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

    const auto hot_cache_plan = deepseek4_expert_split_hot_cache_plan(
        out.expert_budget,
        out.mem.total_expert_bytes,
        w.n_expert,
        w.n_expert_used,
        out.parent_is_igpu,
        out.host_budget_cap_bytes,
        (uint64_t) out.gpu_free);
    out.hot_budget = hot_cache_plan.hot_bytes;
    out.cache_slots = hot_cache_plan.cache_slots;
    out.max_hot_per_layer = std::min(
        w.n_expert,
        (int) (out.hot_budget / out.mem.bytes_per_uniform_round));
    if (out.max_hot_per_layer <= 0) {
        if (err) *err = "pinned hot budget is smaller than one uniform expert round";
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
    static const int sm = query_gpu_compute_sm();
    cfg.mmq_safe_full_batch = (sm >= 80);
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

static MoeHybridConfig make_ds4_parent_gpu_tail_cfg(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg = make_ds4_parent_worker_cfg(w);
    cfg.materialize_hot_experts = false;
    cfg.materialize_cold_experts = true;
    cfg.cold_expert_backend = MoeHybridColdBackend::Gpu;
    return cfg;
}

static MoeLayerDesc make_ds4_moe_layer_desc(const DeepSeek4Layer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps = L.ffn_gate_exps;
    desc.ffn_up_exps = L.ffn_up_exps;
    desc.ffn_down_exps = L.ffn_down_exps;
    desc.ffn_gate_up_exps = nullptr;
    desc.ffn_gate_shexp = L.ffn_gate_shexp;
    desc.ffn_up_shexp = L.ffn_up_shexp;
    desc.ffn_down_shexp = L.ffn_down_shexp;
    desc.ffn_gate_inp_shexp = nullptr;
    return desc;
}

static bool deepseek4_target_list_prefers_static_hotness_prior(
        const std::vector<ExpertSplitTarget> & targets) {
    if (targets.size() > 1) {
        return true;
    }
    return std::any_of(
        targets.begin(), targets.end(),
        [](const ExpertSplitTarget & target) {
            return target.backend == "cpu";
        });
}

static bool init_uniform_hotness(
        int n_layer,
        int n_expert,
        int n_expert_used,
        MoeHybridRoutingStats & hotness,
        std::string * err) {
    if (!hotness.init(n_layer, n_expert, n_expert_used)) {
        if (err) *err = "failed to initialize uniform hotness";
        return false;
    }
    std::fill(hotness.counts.begin(), hotness.counts.end(), 1);
    hotness.layer_totals.assign((size_t) n_layer, (uint64_t) n_expert);
    return true;
}

static bool build_deepseek4_static_hotness_prior(
        const DeepSeek4Weights & w,
        MoeHybridRoutingStats & hotness,
        std::string & source) {
    if (!hotness.init(w.n_layer, w.n_expert, w.n_expert_used)) {
        return false;
    }

    constexpr uint64_t kRouteBiasLayerTotal = 1ULL << 20;
    bool used_hash_prior = false;
    bool used_bias_prior = false;
    std::vector<uint64_t> layer_counts;
    std::vector<int32_t> hash_ids;
    std::vector<float> route_bias;

    for (int layer = 0; layer < w.n_layer; ++layer) {
        const DeepSeek4Layer & L = w.layers[(size_t) layer];
        bool seeded = false;

        if (layer < w.n_hash_layer &&
            L.ffn_gate_tid2eid &&
            L.ffn_gate_tid2eid->ne[0] == w.n_expert_used &&
            L.ffn_gate_tid2eid->ne[1] > 0) {
            const int n_token_ids = (int) L.ffn_gate_tid2eid->ne[1];
            hash_ids.resize((size_t) n_token_ids * (size_t) w.n_expert_used);
            ggml_backend_tensor_get(
                L.ffn_gate_tid2eid,
                hash_ids.data(),
                0,
                sizeof(int32_t) * hash_ids.size());
            if (deepseek4_expert_split_hash_ids_to_hotness_counts(
                    hash_ids.data(),
                    n_token_ids,
                    w.n_expert_used,
                    w.n_expert,
                    layer_counts)) {
                used_hash_prior = true;
                seeded = true;
            }
        }

        if (!seeded &&
            L.ffn_exp_probs_b &&
            L.ffn_exp_probs_b->ne[0] == w.n_expert) {
            route_bias.resize((size_t) w.n_expert);
            ggml_backend_tensor_get(
                L.ffn_exp_probs_b,
                route_bias.data(),
                0,
                sizeof(float) * route_bias.size());
            if (deepseek4_expert_split_route_bias_to_hotness_counts(
                    route_bias.data(),
                    w.n_expert,
                    kRouteBiasLayerTotal,
                    layer_counts)) {
                used_bias_prior = true;
                seeded = true;
            }
        }

        if (!seeded) {
            layer_counts.assign((size_t) w.n_expert, 1);
        }

        uint64_t layer_total = 0;
        for (int expert = 0; expert < w.n_expert; ++expert) {
            const uint64_t count = layer_counts[(size_t) expert];
            hotness.counts[(size_t) layer * (size_t) w.n_expert + (size_t) expert] =
                count;
            layer_total += count;
        }
        hotness.layer_totals[(size_t) layer] = layer_total;
    }

    if (used_hash_prior && used_bias_prior) {
        source = "static-prior(hash+bias)";
    } else if (used_hash_prior) {
        source = "static-prior(hash)";
    } else if (used_bias_prior) {
        source = "static-prior(bias)";
    } else {
        source = "uniform";
    }
    return used_hash_prior || used_bias_prior;
}

static bool build_ds4_cold_owner_placement(
    const MoeHybridPlacement & target_placement,
    MoeHybridPlacement & out,
    std::vector<std::vector<int32_t>> & cold_order_by_layer,
    std::string * err) {
    if (!target_placement.matches(target_placement.n_layer,
                                  target_placement.n_expert,
                                  target_placement.n_expert_used)) {
        if (err) *err = "invalid expert split target placement";
        return false;
    }
    out = target_placement;
    out.total_hot = 0;
    out.hot_counts.assign((size_t) target_placement.n_layer, 0);
    out.hot_expert_ids.resize((size_t) target_placement.n_layer);
    cold_order_by_layer.resize((size_t) target_placement.n_layer);
    for (int il = 0; il < target_placement.n_layer; ++il) {
        const auto & target_owned =
            target_placement.hot_expert_ids[(size_t) il];
        std::vector<uint8_t> owned((size_t) target_placement.n_expert, 0);
        auto & cold_order = cold_order_by_layer[(size_t) il];
        cold_order.clear();
        cold_order.reserve(target_owned.size());
        for (int32_t expert : target_owned) {
            if (expert < 0 || expert >= target_placement.n_expert) {
                if (err) *err = "expert split target placement expert id out of range";
                return false;
            }
            if (owned[(size_t) expert]) {
                if (err) *err = "expert split target placement contains duplicates";
                return false;
            }
            owned[(size_t) expert] = 1;
            cold_order.push_back(expert);
        }
        auto & hot_experts = out.hot_expert_ids[(size_t) il];
        hot_experts.clear();
        hot_experts.reserve((size_t) target_placement.n_expert - target_owned.size());
        for (int expert = 0; expert < target_placement.n_expert; ++expert) {
            if (!owned[(size_t) expert]) {
                hot_experts.push_back(expert);
            }
        }
        out.hot_counts[(size_t) il] = (int) hot_experts.size();
        out.total_hot += (int) hot_experts.size();
    }
    return true;
}

}  // namespace

DeepSeek4Backend::DeepSeek4Backend(const DeepSeek4BackendConfig & cfg)
    : cfg_(cfg) {}

DeepSeek4Backend::~DeepSeek4Backend() {
    shutdown();
}

bool DeepSeek4Backend::run_step_with_runtime_path(
        const float * embed,
        int n_tokens,
        int kv_start,
        std::vector<float> & out_logits,
        const int32_t * token_ids,
        bool want_logits,
        bool disable_cached_decode,
        DeepSeek4StepTelemetry * telemetry) {
    if (!moe_hybrid_) {
        return deepseek4_step_layer_range(backend_, w_, cache_, hc_state_,
                                          embed, n_tokens, kv_start,
                                          0, w_.n_layer,
                                          want_logits ? &out_logits : nullptr,
                                          token_ids, telemetry);
    }
    (void) disable_cached_decode;

    MoeExpertCompute * expert_compute =
        expert_split_runtime_.enabled ? expert_split_runtime_.compute_ptr()
                                      : expert_runtime_.compute_ptr();
    const MoeExpertLayer * expert_layers =
        expert_split_runtime_.enabled ? expert_split_runtime_.layer_ptr(0)
                                      : expert_runtime_.layer_ptr(0);

    return deepseek4_step(backend_, w_, cache_, embed, n_tokens, kv_start, out_logits,
                          moe_hybrid_.get(), token_ids, nullptr,
                          false,
                          want_logits,
                          telemetry,
                          routing_stats_.get(),
                          expert_compute,
                          expert_layers);
}

void DeepSeek4Backend::reset_request_state() {
    cache_.cur_pos = 0;
    for (auto & layer : cache_.layers) {
        layer.n_comp = 0;
        layer.n_index_comp = 0;
    }
    std::fill(hc_state_.begin(), hc_state_.end(), 0.0f);
    last_logits_.clear();
}

bool DeepSeek4Backend::init() {
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[deepseek4] failed to create CUDA backend (gpu=%d)\n",
                     cfg_.device.gpu);
        return false;
    }

    snap_backend_ = ggml_backend_init_by_name("cpu", nullptr);

    const bool force_hybrid = env_flag_enabled("DFLASH_DEEPSEEK4_FORCE_HYBRID");
    if (force_hybrid) {
        std::fprintf(stderr, "[deepseek4] force hybrid mode requested; skipping full model load\n");
    }

    // Try full load first; if GPU OOM, fall back to hybrid mode automatically.
    if (!force_hybrid && !load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[deepseek4] full model load failed, trying hybrid mode...\n");
        if (!init_hybrid_model()) {
            std::fprintf(stderr, "[deepseek4] hybrid mode also failed: %s\n", cfg_.model_path);
            return false;
        }
    } else if (force_hybrid) {
        if (!init_hybrid_model()) {
            std::fprintf(stderr, "[deepseek4] forced hybrid mode failed: %s\n", cfg_.model_path);
            return false;
        }
    }

    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    if (!create_deepseek4_cache(backend_, w_, max_ctx, cache_)) {
        std::fprintf(stderr, "[deepseek4] failed to allocate KV cache (ctx=%d)\n", max_ctx);
        return false;
    }
    hc_state_.assign((size_t) w_.n_hc * (size_t) w_.n_embd, 0.0f);

    std::fprintf(stderr, "[deepseek4] initialized: %d layers, ctx=%d, %d experts (%d used)%s\n",
                 w_.n_layer, max_ctx, w_.n_expert, w_.n_expert_used,
                 moe_hybrid_ ? " [hybrid]" : "");
    return true;
}

bool DeepSeek4Backend::compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                                       int max_ctx,
                                                       MoeHybridPlacement & out,
                                                       std::string * err) const {
    Ds4HybridBudgetInfo budget;
    if (!compute_ds4_hybrid_budget_info(
            w, resolved_device_backend(cfg_.device),
            cfg_.device.gpu, max_ctx, budget, err)) {
        return false;
    }

    const int hot_per_layer = budget.max_hot_per_layer;
    fill_prefix_hot_placement(w, hot_per_layer, out);

    Ds4ExpertMemoryInfo placed_mem;
    if (!compute_ds4_expert_memory_info(w, &out, placed_mem, err)) {
        return false;
    }

    std::fprintf(stderr,
                 "[deepseek4] hybrid placement: gpu_total=%.2f GiB gpu_free=%.2f GiB core=%.2f GiB kv=%.2f GiB warm=%.2f GiB safety=%.2f GiB host_cap=%.2f GiB expert_budget=%.2f GiB pinned_hot=%.2f GiB cache_slots=%d hot/layer=%d\n",
                 gib((uint64_t) budget.gpu_total),
                 gib((uint64_t) budget.gpu_free),
                 gib(budget.core_bytes),
                 gib(budget.kv_bytes),
                 gib(budget.warm_bytes),
                 gib(budget.safety_bytes),
                 gib(budget.host_budget_cap_bytes),
                 gib(budget.expert_budget),
                 gib(budget.hot_budget),
                 budget.cache_slots,
                 hot_per_layer);
    log_ds4_expert_memory_info("placement", placed_mem, w.n_layer);
    return true;
}

bool DeepSeek4Backend::build_expert_split_state_from_hotness(
        const MoeHybridRoutingStats & hotness,
        uint64_t expert_budget_bytes,
        uint64_t hot_budget_bytes,
        int max_hot_per_layer,
        const std::vector<ExpertSplitTarget> & configured_targets,
        const DeepSeek4Weights & w,
        std::string * err) {
    if ((int)layer_expert_bytes_.size() != w.n_layer) {
        if (err) *err = "layer expert bytes not initialized";
        return false;
    }
    if (expert_budget_bytes == 0) {
        if (err) *err = "expert budget must be > 0";
        return false;
    }
    if (!hotness.matches(w.n_layer, w.n_expert, w.n_expert_used)) {
        if (err) *err = "hotness table dimensions do not match DeepSeek4 model";
        return false;
    }

    ExpertSplitConfig cfg;
    cfg.n_layer = w.n_layer;
    cfg.n_expert = w.n_expert;
    cfg.allow_implicit_cpu_fallback = true;
    cfg.require_full_grid = true;

    std::vector<ExpertSplitTarget> targets = configured_targets;
    if (targets.size() > 1 && !targets[0].unlimited) {
        targets[0].capacity_bytes =
            deepseek4_expert_split_primary_capacity_for_targets(
                targets[0].capacity_bytes,
                hot_budget_bytes,
                targets.size());
    }
    if (targets.empty()) {
        const PlacementBackend backend = resolved_device_backend(cfg_.device);
        targets = {
            {std::string(placement_backend_name(backend)) + ":" +
                 std::to_string(cfg_.device.gpu),
             placement_backend_name(backend), cfg_.device.gpu,
             expert_budget_bytes, 0, false},
        };
    }
    if (!build_capacity_weighted_expert_split_min_per_layer(
            targets, std::min(w.n_expert_used, w.n_expert),
            cfg.min_per_layer_by_target, err)) {
        return false;
    }
    if (!targets.empty() &&
        std::none_of(cfg.min_per_layer_by_target.begin(),
                     cfg.min_per_layer_by_target.end(),
                     [](int v) { return v > 0; })) {
        cfg.min_per_layer_by_target.assign(targets.size(), 0);
        cfg.min_per_layer_by_target[0] = std::min(w.n_expert_used, w.n_expert);
    }

    std::vector<ExpertSplitUnit> units;
    units.reserve((size_t)w.n_layer * (size_t)w.n_expert);
    for (int il = 0; il < w.n_layer; ++il) {
        for (int ie = 0; ie < w.n_expert; ++ie) {
            units.push_back(ExpertSplitUnit{
                il,
                ie,
                layer_expert_bytes_[(size_t)il],
                (double)hotness.count(il, ie),
            });
        }
    }

    std::vector<int> primary_hot_limits;
    if (max_hot_per_layer > 0) {
        primary_hot_limits.assign((size_t) w.n_layer, max_hot_per_layer);
    }

    ExpertSplitStateComponents state;
    if (!build_expert_split_state(cfg, targets, units,
                                  w.n_expert_used,
                                  primary_hot_limits.empty()
                                      ? nullptr
                                      : &primary_hot_limits,
                                  state, err)) {
        return false;
    }
    if (!validate_primary_expert_split_target(state.plan.targets,
                                              resolved_device_backend(cfg_.device),
                                              cfg_.device.gpu, err)) {
        return false;
    }
    expert_split_state_ = std::move(state);
    return true;
}

bool DeepSeek4Backend::load_expert_split_placement(
        const char * hotness_path,
        const DeepSeek4Weights & w,
        int max_ctx,
        MoeHybridPlacement & out,
        std::string * err) {
    Ds4HybridBudgetInfo budget;
    if (!compute_ds4_hybrid_budget_info(
            w, resolved_device_backend(cfg_.device),
            cfg_.device.gpu, max_ctx, budget, err)) {
        return false;
    }
    layer_expert_bytes_ = budget.mem.layer_expert_bytes;

    std::vector<ExpertSplitTarget> configured_targets;
    if (!resolve_ds4_expert_split_targets_from_env(
            budget.expert_budget, configured_targets, err)) {
        return false;
    }

    uint64_t placement_budget_bytes = budget.expert_budget;
    uint64_t placement_non_cpu_capacity_bytes = 0;
    bool promoted_cpu_fallback_only = false;
    if (!configured_targets.empty() &&
        budget.auto_expert_budget > budget.expert_budget) {
        std::vector<ExpertSplitTarget> auto_capacity_targets;
        if (!resolve_ds4_expert_split_targets_from_env(
                budget.auto_expert_budget, auto_capacity_targets, err)) {
            return false;
        }
        const uint64_t promoted_budget =
            deepseek4_expert_split_effective_budget_for_targets(
                budget.expert_budget,
                budget.mem.total_expert_bytes,
                auto_capacity_targets);
        if (promoted_budget > placement_budget_bytes) {
            placement_non_cpu_capacity_bytes =
                deepseek4_expert_split_non_cpu_capacity_bytes(
                    auto_capacity_targets);
            placement_budget_bytes = promoted_budget;
            configured_targets = std::move(auto_capacity_targets);
            promoted_cpu_fallback_only = true;
        }
    }

    MoeHybridRoutingStats hotness;
    std::string hotness_source = "uniform";
    if (hotness_path && hotness_path[0]) {
        if (!MoeHybridRoutingStats::load_csv(std::string(hotness_path), hotness, err)) {
            return false;
        }
        if (!hotness.matches(w.n_layer, w.n_expert, w.n_expert_used)) {
            if (err) *err = "hotness table dimensions do not match DeepSeek4 model";
            return false;
        }
        hotness_source = hotness_path;
    } else if (deepseek4_target_list_prefers_static_hotness_prior(
                   configured_targets) &&
               build_deepseek4_static_hotness_prior(
                   w, hotness, hotness_source)) {
        // hotness_source already describes the static prior that was used
    } else if (!init_uniform_hotness(
                   w.n_layer, w.n_expert, w.n_expert_used, hotness, err)) {
        return false;
    }

    if (routing_stats_ && hotness_path && hotness_path[0] &&
        hotness.counts.size() == (size_t)w.n_layer * (size_t)w.n_expert) {
        routing_stats_->counts = hotness.counts;
        routing_stats_->layer_totals.assign((size_t)w.n_layer, 0);
        for (int il = 0; il < w.n_layer; ++il) {
            for (int ie = 0; ie < w.n_expert; ++ie) {
                routing_stats_->layer_totals[(size_t)il] +=
                    hotness.counts[(size_t)il * (size_t)w.n_expert + (size_t)ie];
            }
        }
    }

    const size_t configured_target_count =
        configured_targets.empty() ? 1 : configured_targets.size();
    const uint64_t effective_hot_budget =
        deepseek4_expert_split_effective_hot_budget(
            placement_budget_bytes,
            budget.hot_budget,
            configured_target_count);
    int effective_max_hot_per_layer = budget.max_hot_per_layer;
    if (budget.mem.bytes_per_uniform_round > 0) {
        effective_max_hot_per_layer = std::min(
            w.n_expert,
            (int) (effective_hot_budget / budget.mem.bytes_per_uniform_round));
    }
    if (effective_max_hot_per_layer <= 0) {
        if (err) {
            *err =
                "effective pinned hot budget is smaller than one uniform expert round";
        }
        return false;
    }

    expert_split_total_budget_bytes_ = budget.expert_budget;
    expert_split_hot_budget_bytes_ = effective_hot_budget;
    expert_split_cache_slots_ =
        deepseek4_expert_split_effective_cache_slots(
            budget.cache_slots,
            configured_target_count);

    if (!build_expert_split_state_from_hotness(
            hotness, placement_budget_bytes, effective_hot_budget,
            effective_max_hot_per_layer, configured_targets, w, err)) {
        return false;
    }
    expert_split_total_budget_bytes_ = placement_budget_bytes;
    if (expert_split_cache_slots_ != budget.cache_slots) {
        std::fprintf(stderr,
                     "[deepseek4] expert_split cache disabled for multi-target placement: requested_slots=%d targets=%zu\n",
                     budget.cache_slots,
                     configured_target_count);
    }
    if (promoted_cpu_fallback_only) {
        std::fprintf(stderr,
                     "[deepseek4] expert_split promoted non-CPU placement budget: requested=%.2f GiB effective=%.2f GiB non_cpu_cap=%.2f GiB total_experts=%.2f GiB\n",
                     gib(budget.expert_budget),
                     gib(placement_budget_bytes),
                     gib(placement_non_cpu_capacity_bytes),
                     gib(budget.mem.total_expert_bytes));
    }
    out = expert_split_state_.materialization.primary_placement;

    Ds4ExpertMemoryInfo placed_mem;
    if (!compute_ds4_expert_memory_info(w, &out, placed_mem, err)) {
        return false;
    }
    std::fprintf(stderr,
                 "[deepseek4] expert_split placement: gpu_total=%.2f GiB gpu_free=%.2f GiB core=%.2f GiB kv=%.2f GiB warm=%.2f GiB safety=%.2f GiB host_cap=%.2f GiB expert_budget=%.2f GiB placement_budget=%.2f GiB pinned_hot=%.2f GiB cache_slots=%d targets=%zu source=%s\n",
                 gib((uint64_t)budget.gpu_total),
                 gib((uint64_t)budget.gpu_free),
                 gib(budget.core_bytes),
                 gib(budget.kv_bytes),
                 gib(budget.warm_bytes),
                 gib(budget.safety_bytes),
                 gib(budget.host_budget_cap_bytes),
                 gib(budget.expert_budget),
                 gib(placement_budget_bytes),
                 gib(effective_hot_budget),
                 expert_split_cache_slots_,
                 configured_target_count,
                 hotness_source.c_str());
    log_ds4_expert_memory_info("expert_split", placed_mem, w.n_layer);
    return true;
}

bool DeepSeek4Backend::init_expert_split_runtime(std::string * err) {
    expert_split_runtime_.reset();
    if (!moe_hybrid_ || expert_split_state_.empty()) {
        return true;
    }
    if (!expert_split_state_.compute_runtime.matches(
            w_.n_layer, w_.n_expert, w_.n_expert_used)) {
        if (err) *err = "expert split compute runtime does not match DeepSeek4 model";
        return false;
    }

    MoeExpertComputeRuntimeConfig runtime_cfg;
    runtime_cfg.target_path = cfg_.model_path ? cfg_.model_path : "";
    runtime_cfg.n_layer = w_.n_layer;
    runtime_cfg.n_expert = w_.n_expert;
    runtime_cfg.n_expert_used = w_.n_expert_used;
    runtime_cfg.n_embd = w_.n_embd;
    runtime_cfg.n_ff_exp = w_.n_ff_exp;
    runtime_cfg.enabled = true;
    runtime_cfg.log_prefix = "[deepseek4-expert-split]";
    runtime_cfg.local_target_factory_key = "deepseek4-local-gpu-cold";
    runtime_cfg.local_target_compute_factory =
        [this](const ExpertSplitComputeTargetRuntime & split_target,
               std::string * local_err) {
            return make_expert_split_local_target_compute(split_target, local_err);
        };

    std::vector<MoeLayerDesc> layer_descs((size_t)w_.n_layer);
    for (int il = 0; il < w_.n_layer; ++il) {
        layer_descs[(size_t)il] = make_ds4_moe_layer_desc(w_.layers[(size_t)il]);
    }

    if (!ensure_multi_target_moe_expert_compute_runtime(
            expert_split_runtime_, runtime_cfg,
            expert_split_state_.compute_runtime,
            *moe_hybrid_, layer_descs, err)) {
        return false;
    }
    if (expert_split_runtime_.enabled && expert_split_runtime_.compute_ptr()) {
        std::fprintf(stderr, "[deepseek4] expert_split compute ready: %s\n",
                     expert_split_runtime_.runtime_key.c_str());
    }
    return true;
}

std::unique_ptr<MoeExpertCompute> DeepSeek4Backend::make_expert_split_local_target_compute(
    const ExpertSplitComputeTargetRuntime & split_target,
    std::string * err) const {
    if (!cfg_.model_path || !cfg_.model_path[0]) {
        if (err) *err = "DeepSeek4 local target compute requires a model path";
        return nullptr;
    }
    if (split_target.target.backend == "cpu") {
        return nullptr;
    }
    if (split_target.target.device_id < 0) {
        if (err) *err = "DeepSeek4 local target compute requires an explicit device id";
        return nullptr;
    }
    MoeHybridPlacement cold_owner_placement;
    std::vector<std::vector<int32_t>> cold_order_by_layer;
    if (!build_ds4_cold_owner_placement(split_target.placement,
                                        cold_owner_placement,
                                        cold_order_by_layer,
                                        err)) {
        return nullptr;
    }

    ggml_backend_t target_backend =
        ggml_backend_cuda_init(std::max(0, split_target.target.device_id));
    if (!target_backend) {
        if (err) *err = "failed to initialize DeepSeek4 local backend";
        return nullptr;
    }

    MoeHybridConfig local_cfg = make_ds4_parent_gpu_tail_cfg(w_);
    auto storage = std::make_unique<MoeHybridStorage>();
    if (!build_deepseek4_moe_hybrid_storage_from_file(
            cfg_.model_path,
            target_backend,
            w_,
            cold_owner_placement,
            &local_cfg,
            *storage,
            err,
            /*cache_slots=*/0,
            /*load_cold_tensors=*/true,
            &cold_order_by_layer)) {
        ggml_backend_free(target_backend);
        return nullptr;
    }

    std::vector<MoeLayerDesc> layer_descs((size_t) w_.n_layer);
    for (int il = 0; il < w_.n_layer; ++il) {
        layer_descs[(size_t) il] = make_ds4_moe_layer_desc(w_.layers[(size_t) il]);
    }
    return make_local_gpu_cold_moe_expert_compute(
        target_backend,
        std::move(storage),
        local_cfg,
        std::move(layer_descs));
}

bool DeepSeek4Backend::init_single_target_expert_runtime(std::string * err) {
    expert_runtime_.reset();
    if (!moe_hybrid_) {
        return true;
    }

    MoeExpertComputeRuntimeConfig runtime_cfg;
    runtime_cfg.target_path = cfg_.model_path ? cfg_.model_path : "";
    runtime_cfg.n_layer = w_.n_layer;
    runtime_cfg.n_expert = w_.n_expert;
    runtime_cfg.n_expert_used = w_.n_expert_used;
    runtime_cfg.n_embd = w_.n_embd;
    runtime_cfg.n_ff_exp = w_.n_ff_exp;
    runtime_cfg.enabled = true;
    runtime_cfg.log_prefix = "[deepseek4-hybrid]";

    std::vector<MoeLayerDesc> layer_descs((size_t)w_.n_layer);
    for (int il = 0; il < w_.n_layer; ++il) {
        layer_descs[(size_t)il] = make_ds4_moe_layer_desc(w_.layers[(size_t)il]);
    }

    if (!ensure_moe_expert_compute_runtime(
            expert_runtime_, runtime_cfg, *moe_hybrid_, layer_descs, err)) {
        return false;
    }
    if (expert_runtime_.compute_ptr()) {
        std::fprintf(stderr, "[deepseek4] hybrid expert compute ready: %s\n",
                     expert_runtime_.runtime_key.c_str());
    }
    return true;
}

bool DeepSeek4Backend::init_hybrid_model() {
    expert_split_total_budget_bytes_ = 0;
    expert_split_hot_budget_bytes_ = 0;
    expert_split_cache_slots_ = 0;

    TargetLoadPlan plan;
    plan.skip_expert_tensors = true;
    if (!load_deepseek4_gguf_partial(cfg_.model_path, backend_, plan, w_)) {
        std::fprintf(stderr, "[deepseek4] failed to partially load model for hybrid mode: %s\n",
                     cfg_.model_path);
        return false;
    }

    if (const char * stats_path = std::getenv("DFLASH_DEEPSEEK4_RUNTIME_STATS_OUT")) {
        routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
        MoeHybridConfig stats_cfg = make_ds4_parent_worker_cfg(w_);
        if (!routing_stats_->init(stats_cfg)) {
            std::fprintf(stderr, "[deepseek4] routing stats init failed\n");
            return false;
        }
        routing_stats_out_path_ = stats_path;
    }

    std::string err;
    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    const char * hotness_path = std::getenv("DFLASH_DEEPSEEK4_HOTNESS");
    if (!load_expert_split_placement(hotness_path, w_, max_ctx, moe_placement_, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to compute expert_split placement: %s\n", err.c_str());
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
    const std::vector<std::vector<int32_t>> * cold_order_by_layer =
        expert_split_state_.materialization.ordered_cold_union
            ? &expert_split_state_.materialization.cold_expert_ids_by_layer
            : nullptr;
    if (!build_deepseek4_moe_hybrid_storage_from_file(
            cfg_.model_path, backend_, w_, moe_placement_, nullptr,
            *hybrid, &err, expert_split_cache_slots_,
            /*load_cold_tensors=*/true,
            cold_order_by_layer)) {
        std::fprintf(stderr, "[deepseek4] failed to build hybrid expert storage: %s\n", err.c_str());
        return false;
    }

    moe_hybrid_ = std::move(hybrid);
    if (!init_single_target_expert_runtime(&err)) {
        std::fprintf(stderr, "[deepseek4] failed to initialize hybrid expert runtime: %s\n",
                     err.c_str());
        return false;
    }
    if (!init_expert_split_runtime(&err)) {
        std::fprintf(stderr, "[deepseek4] failed to initialize expert_split runtime: %s\n",
                     err.c_str());
        return false;
    }
    if (deepseek4_expert_split_requires_parent_cuda_graph_disable(
            expert_split_runtime_.enabled,
            backend_is_cuda(backend_),
            expert_split_state_.compute_runtime.targets)) {
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
        std::fprintf(stderr,
                     "[deepseek4] expert_split multi-target on CUDA: disabling CUDA graphs for parent runtime\n");
    }
    if (expert_split_runtime_.enabled) {
        const PlacementBackend device_backend = resolved_device_backend(cfg_.device);
        const bool disable_cached_decode =
            should_disable_ds4_expert_split_cached_decode(
                /*expert_split_enabled=*/true,
                device_backend,
                backend_,
                cfg_.device.gpu);
        std::fprintf(stderr,
                     "[deepseek4] expert_split decode cache: %s parent_backend=%s device=%d\n",
                     disable_cached_decode ? "eager" : "cached",
                     placement_backend_name(device_backend),
                     cfg_.device.gpu);
    }
    w_.moe_hybrid = true;
    const int total_cold = w_.n_layer * w_.n_expert - moe_placement_.total_hot;
    const char * cold_backend =
        moe_hybrid_->cold_backend_kind == MoeHybridColdBackend::Gpu ? "gpu" : "cpu";
    std::fprintf(stderr,
                 "[deepseek4] hybrid experts ready: hot=%d cold=%d cold_backend=%s cache_slots=%d hot_budget=%.2f GiB expert_budget=%.2f GiB\n",
                 moe_placement_.total_hot,
                 total_cold,
                 cold_backend,
                 expert_split_cache_slots_,
                 gib(expert_split_hot_budget_bytes_),
                 gib(expert_split_total_budget_bytes_));
    return true;
}

void DeepSeek4Backend::print_ready_banner() const {
    std::printf("[deepseek4-daemon] ready layers=%d ctx=%d experts=%d/%d\n",
                w_.n_layer, cache_.max_ctx, w_.n_expert_used, w_.n_expert);
    std::fflush(stdout);
}

bool DeepSeek4Backend::park(const std::string & what) {
    (void)what;
    // TODO: Release GPU resources
    parked_ = true;
    return true;
}

bool DeepSeek4Backend::unpark(const std::string & what) {
    (void)what;
    parked_ = false;
    return true;
}

int DeepSeek4Backend::do_prefill(const std::vector<int32_t> & tokens,
                                  const DaemonIO & io,
                                  int kv_offset) {
    const bool use_hybrid_batch_prefill =
        moe_hybrid_ &&
        backend_is_gpu(backend_) &&
        !env_flag_enabled("DFLASH_DS4_DISABLE_HYBRID_BATCH_PREFILL");
    const int requested_chunk = cfg_.chunk > 0 ? cfg_.chunk : 512;
    const PlacementBackend device_backend = resolved_device_backend(cfg_.device);
    const int chunk = use_hybrid_batch_prefill
        ? resolve_ds4_expert_split_prefill_chunk(
              requested_chunk,
              expert_split_runtime_.enabled,
              device_backend,
              backend_,
              cfg_.device.gpu)
        : (moe_hybrid_ ? 1 : requested_chunk);
    if (use_hybrid_batch_prefill &&
        expert_split_runtime_.enabled &&
        chunk < requested_chunk) {
        std::fprintf(stderr,
                     "[deepseek4] expert_split prefill chunk capped: requested=%d effective=%d\n",
                     requested_chunk, chunk);
    }
    const int n_total = (int)tokens.size();
    int pos = kv_offset;
    last_logits_.clear();
    std::vector<float> embed((size_t)w_.n_embd * (size_t)std::max(1, chunk));
    std::vector<float> logits;
    const bool force_intermediate_logits =
        env_flag_enabled("DFLASH_DS4_FORCE_INTERMEDIATE_PREFILL_LOGITS");
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;
    if (timing && moe_hybrid_) {
        const char * backend_name = backend_ ? ggml_backend_name(backend_) : "(null)";
        std::fprintf(stderr,
                     "[deepseek4] prefill plan: tokens=%d requested_chunk=%d effective_chunk=%d hybrid_batch=%s expert_split=%s intermediate_logits=%s backend=%s\n",
                     n_total,
                     requested_chunk,
                     chunk,
                     use_hybrid_batch_prefill ? "on" : "off",
                     expert_split_runtime_.enabled ? "on" : "off",
                     force_intermediate_logits ? "on" : "off",
                     backend_name ? backend_name : "(null)");
    }
    for (int i = 0; i < n_total; i += chunk) {
        if (io.cancelled) return pos;

        const int n_tok = std::min(chunk, n_total - i);
        const bool want_logits =
            force_intermediate_logits || (i + n_tok) >= n_total;

        // Embed tokens
        const auto embed_t0 = Clock::now();
        w_.embedder.embed(tokens.data() + i, n_tok, embed.data());
        DeepSeek4StepTelemetry step_tel;
        if (timing) step_tel.embed_us = elapsed_us(embed_t0, Clock::now());

        // Run forward pass
        logits.clear();
        if (!run_step_with_runtime_path(embed.data(), n_tok, pos, logits,
                                        tokens.data() + i,
                                        want_logits,
                                        false,
                                        timing ? &step_tel : nullptr)) {
            std::fprintf(stderr, "[deepseek4] prefill step failed at pos=%d\n", pos);
            return -1;
        }
        if (timing) {
            add_step_tel(tel_acc, step_tel);
            steps++;
        }
        if (want_logits) {
            last_logits_ = std::move(logits);
        }
        pos += n_tok;
    }
    if (timing) {
        log_step_tel("prefill", n_total, steps, elapsed_s(phase_t0), tel_acc);
    }
    if (env_flag_enabled("DFLASH_DS4_TRACE_DECODE")) {
        std::fprintf(stderr,
                     "[deepseek4-trace] prefill done committed=%d last_logits=%zu\n",
                     pos, last_logits_.size());
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
    std::vector<float> embed((size_t)w_.n_embd);
    std::vector<float> logits;
    const PlacementBackend device_backend = resolved_device_backend(cfg_.device);
    const bool disable_cached_decode =
        should_disable_ds4_expert_split_cached_decode(
            expert_split_runtime_.enabled,
            device_backend,
            backend_,
            cfg_.device.gpu);
    for (int generated = 0; generated < n_gen; generated++) {
        if (io.cancelled) break;

        if (env_flag_enabled("DFLASH_DS4_TRACE_DECODE")) {
            std::fprintf(stderr,
                         "[deepseek4-trace] decode iter=%d last_logits=%zu out_tokens=%zu\n",
                         generated, last_logits_.size(), out_tokens.size());
        }

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
        if (generated == 0 && !last_logits_.empty()) {
            if (env_flag_enabled("DFLASH_DS4_TRACE_DECODE")) {
                std::fprintf(stderr,
                             "[deepseek4-trace] decode iter=%d using cached prefill logits=%zu\n",
                             generated, last_logits_.size());
            }
            logits = last_logits_;
        } else {
            int32_t tok_to_eval = out_tokens.empty() ? 0 : out_tokens.back();
            if (env_flag_enabled("DFLASH_DS4_TRACE_DECODE")) {
                std::fprintf(stderr,
                             "[deepseek4-trace] decode iter=%d fallback forward tok=%d\n",
                             generated, tok_to_eval);
            }
            const auto embed_t0 = Clock::now();
            w_.embedder.embed(&tok_to_eval, 1, embed.data());
            DeepSeek4StepTelemetry step_tel;
            if (timing) step_tel.embed_us = elapsed_us(embed_t0, Clock::now());

            const int pos = std::max(0, committed + generated - 1);
            if (!run_step_with_runtime_path(embed.data(), 1, pos, logits,
                                            &tok_to_eval,
                                            true,
                                            disable_cached_decode,
                                            timing ? &step_tel : nullptr)) {
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
        if (env_flag_enabled("DFLASH_DS4_TRACE_DECODE")) {
            std::fprintf(stderr,
                         "[deepseek4-trace] decode iter=%d sampled=%d emit_begin\n",
                         generated, next_token);
        }
        const auto emit_t0 = Clock::now();
        io.emit(next_token);
        if (env_flag_enabled("DFLASH_DS4_TRACE_DECODE")) {
            std::fprintf(stderr,
                         "[deepseek4-trace] decode iter=%d sampled=%d emit_done\n",
                         generated, next_token);
        }
        if (timing) tel_acc.emit_us += elapsed_us(emit_t0, Clock::now());

        // Check EOS using the GGUF tokenizer metadata loaded with the weights.
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
    reset_request_state();

    // Prefill
    int committed = do_prefill(req.prompt, out_io);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }
    result.prefill_s = elapsed_s(t0);

    if (req.n_gen <= 0) {
        out_io.emit(-1);
        result.ok = true;
        maybe_save_routing_stats();
        return result;
    }

    // Decode
    auto t1 = Clock::now();
    std::vector<int32_t> gen_tokens;
    gen_tokens.reserve(req.n_gen);

    bool forced_close = false;
    if (!do_decode(committed, req.n_gen, gen_tokens, out_io,
                   req.budget_hook, &forced_close)) {
        result.error = "decode";
        return result;
    }

    out_io.emit(-1);
    result.ok = true;
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
    // No drafter in AR-only mode
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
    for (int i = 0; i < PREFIX_SLOTS; i++) {
        free_deepseek4_snapshot(snapshots_[i]);
    }
    free_deepseek4_cache(cache_);
    moe_hybrid_.reset();
    expert_runtime_.reset();
    expert_split_runtime_.reset();
    expert_split_state_ = {};
    layer_expert_bytes_.clear();
    hc_state_.clear();
    last_logits_.clear();
    routing_stats_.reset();
    routing_stats_out_path_.clear();
    moe_placement_ = {};
    free_deepseek4_weights(w_);
    if (snap_backend_) { ggml_backend_free(snap_backend_); snap_backend_ = nullptr; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

}  // namespace dflash::common
