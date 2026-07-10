// DeepSeek4Backend implementation — AR-only decode, chunked prefill.

#include "deepseek4_backend.h"
#include "deepseek4_internal.h"
#include "common/sampler.h"
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

    if (!compute_ds4_expert_memory_info(w, nullptr, out.mem, err)) {
        return false;
    }

    out.core_bytes =
        moe_hybrid_core_bytes_from_memory("deepseek4", out.gpu_free, out.gpu_total);
    out.kv_bytes = estimate_ds4_cache_bytes(w, max_ctx);
    if (out.gpu_total <= out.core_bytes + out.kv_bytes + out.warm_bytes + out.safety_bytes) {
        if (err) *err = "no VRAM budget available for DS4 experts";
        return false;
    }
    out.expert_budget =
        out.gpu_total - out.core_bytes - out.kv_bytes - out.warm_bytes - out.safety_bytes;
    out.expert_budget = std::min(out.expert_budget, out.mem.total_expert_bytes);
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

    out.max_hot_per_layer = std::min(
        w.n_expert,
        (int) (out.expert_budget / out.mem.bytes_per_uniform_round));
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
    static const int sm = query_gpu_compute_sm();
    cfg.mmq_safe_full_batch = (sm >= 80);
    cfg.materialize_cold_experts = false;
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

    return deepseek4_step(backend_, w_, cache_, embed, n_tokens, kv_start, out_logits,
                          moe_hybrid_.get(), token_ids, nullptr,
                          false,
                          want_logits,
                          telemetry,
                          routing_stats_.get(),
                          expert_runtime_.compute_ptr(),
                          expert_runtime_.layer_ptr(0));
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
            w, cfg_.device.gpu, max_ctx, budget, err)) {
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
    if (!build_deepseek4_moe_hybrid_storage_from_file(
            cfg_.model_path, backend_, w_, moe_placement_, &hybrid_cfg,
            *hybrid, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to build hybrid expert storage: %s\n", err.c_str());
        return false;
    }

    moe_hybrid_ = std::move(hybrid);
    if (!init_single_target_expert_runtime(&err)) {
        std::fprintf(stderr, "[deepseek4] failed to initialize hybrid expert runtime: %s\n",
                     err.c_str());
        return false;
    }
    w_.moe_hybrid = true;
    const int total_cold = w_.n_layer * w_.n_expert - moe_placement_.total_hot;
    const char * cold_backend =
        moe_hybrid_->cold_backend_kind == MoeHybridColdBackend::Gpu ? "gpu" : "cpu";
    std::fprintf(stderr,
                 "[deepseek4] hybrid experts ready: hot=%d cold=%d cold_backend=%s\n",
                 moe_placement_.total_hot,
                 total_cold,
                 cold_backend);
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
    const int requested_chunk = cfg_.chunk > 0 ? cfg_.chunk : 512;
    const int chunk = moe_hybrid_ ? 1 : requested_chunk;
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
                     "[deepseek4] prefill plan: tokens=%d requested_chunk=%d effective_chunk=%d hybrid=%s intermediate_logits=%s backend=%s\n",
                     n_total,
                     requested_chunk,
                     chunk,
                     moe_hybrid_ ? "on" : "off",
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
    const bool disable_cached_decode = false;
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
