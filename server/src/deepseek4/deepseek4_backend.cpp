// DeepSeek4Backend implementation — AR-only decode, chunked prefill.

#include "deepseek4_backend.h"
#include "deepseek4_internal.h"
#include "common/sampler.h"

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "common/gpu_runtime_compat.h"
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
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
    dst.full_graph_build_us += src.full_graph_build_us;
    dst.full_graph_alloc_us += src.full_graph_alloc_us;
    dst.full_graph_set_us += src.full_graph_set_us;
    dst.full_graph_compute_us += src.full_graph_compute_us;
    dst.full_graph_read_us += src.full_graph_read_us;
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
        "ffn_build=%.1fms ffn_compute=%.1fms ffn_read=%.1fms "
        "route_build=%.1fms route_compute=%.1fms route_read=%.1fms route_select=%.1fms "
        "ffn=%.1fms hot=%.1fms cold=%.1fms combine=%.1fms partition=%.1fms "
        "ffn_hot_graph_build=%llu ffn_hot_graph_hit=%llu ffn_cold_graph_build=%llu ffn_cold_graph_hit=%llu "
        "hc_pre=%.1fms hc_pre_build=%.1fms hc_pre_input=%.1fms hc_pre_compute=%.1fms "
        "hc_post=%.1fms output=%.1fms sample=%.1fms emit=%.1fms "
        "graph_build=%.1fms graph_alloc=%.1fms graph_set=%.1fms "
        "graph_compute=%.1fms graph_read=%.1fms "
        "hot_sel=%d cold_sel=%d\n",
        phase, tokens, steps, wall_s, tok_s,
        ms(t.total_us), ms(t.embed_us), ms(t.attn_build_us), ms(t.attn_compute_us), ms(t.attn_read_us),
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
        ms(t.full_graph_build_us), ms(t.full_graph_alloc_us),
        ms(t.full_graph_set_us), ms(t.full_graph_compute_us),
        ms(t.full_graph_read_us),
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
            "[deepseek4] %s prefill requires every expert to be resident; "
            "the selected placement has cold experts\n",
            prefill_attention_mode_name(cfg_.prefill_mode));
        return false;
    }
    return true;
}

bool DeepSeek4Backend::load_model() {
    const PlacementBackend target_backend =
        cfg_.device.backend == PlacementBackend::Auto
            ? compiled_placement_backend()
            : cfg_.device.backend;

    // Fused decode and layer-major prefill reference every expert directly.
    // Make their residency requirement explicit instead of silently falling
    // back to tokenwise hybrid execution.
    if (target_backend == PlacementBackend::Hip && requires_monolithic_model()) {
        std::fprintf(stderr,
                     "[deepseek4] monolithic execution requested "
                     "(fused_decode=%s, prefill=%s)\n",
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
                         "[deepseek4] monolithic HIP load failed; trying hybrid mode\n");
            if (!init_hybrid_model()) {
                std::fprintf(stderr, "[deepseek4] hybrid mode also failed: %s\n",
                             cfg_.model_path);
                return false;
            }
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
    if (parked_ || moe_hybrid_) {
        std::fprintf(stderr,
                     "[deepseek4] cannot load DSpark drafter without a resident "
                     "monolithic target\n");
        return false;
    }

    auto drafter = std::make_unique<DSparkDrafter>();
    if (!load_deepseek4_dspark_drafter(spec_draft_path_, backend_, *drafter)) {
        std::fprintf(stderr, "[deepseek4] DSpark drafter load FAILED: %s\n",
                     deepseek4_dspark_last_error());
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
    spec_enabled_ = false;
    spec_feat_window_.clear();
    spec_drafter_parked_ = mark_parked && !spec_draft_path_.empty();
}

bool DeepSeek4Backend::init() {
    // The shared MMVQ/MMQ crossover defaults to q=3 for NVIDIA. On gfx1151,
    // DSpark q=4 is faster through MMVQ. Keep AR and other devices unchanged,
    // and preserve LUCE_MMVQ_MAX_NCOLS as an explicit override.
    configure_gfx1151_dspark_mmvq_default(cfg_.device.gpu);

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

    if (moe_hybrid_) {
        // Expert IPC removed — layer split replaces expert split.
        // The DeepSeek4Backend single-GPU path now runs all experts locally.
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
            if (moe_hybrid_) {
                std::fprintf(stderr,
                             "[deepseek4] DSpark spec-decode requires monolithic model "
                             "placement; disabled for hybrid expert placement\n");
            } else {
                (void) load_spec_drafter();
            }
        } else {
            std::fprintf(stderr, "[deepseek4] DFLASH_DS4_SPEC set but DFLASH_DS4_DRAFT gguf missing\n");
        }
    }
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

    const int hot_per_layer = budget.max_hot_per_layer;
    fill_prefix_hot_placement(w, hot_per_layer, out);

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
    const MoeHybridConfig hybrid_cfg = make_ds4_parent_worker_cfg(w_);
    if (!build_deepseek4_moe_hybrid_storage_from_file_with_mmap(
            cfg_.model_path, backend_, w_, moe_placement_, &hybrid_cfg, *hybrid, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to build hybrid expert storage: %s\n", err.c_str());
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
    stream_engine_.destroy();
    moe_hybrid_.reset();
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
    const int chunk = (moe_hybrid_ ||
                       !prefill_attention_mode_is_approximate(cfg_.prefill_mode))
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
        Ds4VerifyHooks spec_hooks;
        std::vector<float> spec_cap;
        Ds4VerifyHooks * hp = nullptr;
        if (spec_enabled_ && spec_drafter_ && i + n_tok > spec_capture_from) {
            spec_hooks.capture_layer_ids = &spec_drafter_->capture_layer_ids;
            spec_hooks.capture_out = &spec_cap;
            hp = &spec_hooks;
        }
        bool ok = false;
        if (moe_hybrid_) {
            ok = deepseek4_step(backend_, w_, cache_, embed.data(), n_tok, pos,
                                logits, moe_hybrid_.get(), tokens.data() + i,
                                &stream_engine_, timing ? &step_tel : nullptr,
                                routing_stats_.get(), hp);
        } else {
            std::vector<float> hc_state;
            ok = deepseek4_step_layer_range(
                backend_, w_, cache_, hc_state, embed.data(), n_tok, pos,
                0, w_.n_layer, &logits, tokens.data() + i,
                timing ? &step_tel : nullptr,
                cfg_.prefill_mode != PrefillAttentionMode::Sparse, hp);
        }
        if (ok && hp && !spec_cap.empty()) {
            const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
            const int first_capture = std::max(0, spec_capture_from - i);
            for (int t = first_capture; t < n_tok; ++t) {
                spec_feat_window_.insert(
                    spec_feat_window_.end(),
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
            if (!deepseek4_step(backend_, w_, cache_, embed.data(), 1,
                                pos, logits,
                                moe_hybrid_.get(), &tok_to_eval,
                                moe_hybrid_ ? &stream_engine_ : nullptr,
                                timing ? &step_tel : nullptr,
                                routing_stats_.get())) {
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
                    })) {
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
    stream_engine_.destroy();
    moe_hybrid_.reset();
    routing_stats_.reset();
    routing_stats_out_path_.clear();
    moe_placement_ = {};
    free_deepseek4_weights(w_);
    if (snap_backend_) { ggml_backend_free(snap_backend_); snap_backend_ = nullptr; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

}  // namespace dflash::common
