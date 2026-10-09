// deepseek4_cluster.cpp - DeepSeek4 expert-parallel cluster runtime.
//
// See deepseek4_cluster.h for the model. This file has no RCCL dependency:
// collectives go through IClusterComm, the stream comes from
// ggml_backend_cuda_get_stream.

#include "deepseek4_cluster.h"
#include "deepseek4_internal.h"

#include "common/moe_hybrid_routing_stats.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace luce::common {

// LUCE_CLUSTER_TRACE_COLL=1: one stderr line per collective (rank, sequence, kind, elements, stream), so the
// two ranks' collective sequences can be diffed when a step hangs.
static void trace_coll(int rank, const char * kind, size_t n, const void * stream) {
    static const bool on = std::getenv("LUCE_CLUSTER_TRACE_COLL") != nullptr;
    static unsigned long long seq = 0;
    if (!on) return;
    std::fprintf(stderr, "[coll] r%d #%llu %s n=%zu stream=%p\n", rank, seq++, kind, n, stream);
}

namespace {

using ClusterClock = std::chrono::steady_clock;

uint64_t us_since(ClusterClock::time_point t0) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        ClusterClock::now() - t0).count();
}

// Slots cached from ClusterExpertPlacement::slot_validity. Covers decode
// (1) and every DSpark verify width (<= 5, q5 opt-in) plus headroom; larger
// prefill batches use placement.slot_owner() directly.
constexpr int kSlotValidityCacheSlots = 8;

void set_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

// Mixed ROCmFP qtypes need decode tables registered per resident tensor
// (register_deepseek4_moe_hybrid_mix_tables), which today requires a
// materialized GPU cold owner. Refuse instead of silently loading garbage.
// Supporting them needs register_deepseek4_moe_hybrid_mix_tables to accept
// hot-only storage (cold owner None).
bool ds4_cluster_has_mix_experts(const DeepSeek4Weights & w, std::string * what) {
    for (const auto & L : w.layers) {
        const struct { ggml_tensor * t; int qtype; const char * name; } mix[] = {
            { L.ffn_gate_exps, GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) gate" },
            { L.ffn_up_exps,   GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) up"   },
            { L.ffn_down_exps, GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) down" },
            { L.ffn_gate_exps, GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) gate" },
            { L.ffn_up_exps,   GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) up"   },
            { L.ffn_down_exps, GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) down" },
        };
        for (const auto & m : mix) {
            if (m.t && m.t->type == m.qtype) {
                if (what) *what = m.name;
                return true;
            }
        }
    }
    return false;
}

const char * placement_source_label(cluster::PlacementSource s) {
    return cluster::placement_source_name(s);
}

// Enqueue the collective for n floats at dev_ptr on the backend stream.
// force_wait: block the host until the reduction completed (host-resident
// consumers need this); otherwise stay stream-ordered unless rt.trace.
bool allreduce_device(Ds4ClusterRuntime & rt,
                      ggml_backend_t backend,
                      void * dev_ptr,
                      size_t n,
                      int layer,
                      bool force_wait,
                      DeepSeek4StepTelemetry * telemetry,
                      std::string * err) {
    if (!rt.cfg) {
        set_err(err, "cluster runtime has no config");
        return false;
    }
    // Before the communicator is attached (load-time forwards) a rank runs
    // its own shard alone; nothing reads those outputs.
    if (!rt.comm || rt.comm->size() <= 1 || n == 0) {
        return true;  // single rank: the partial already is the sum
    }
    if (!dev_ptr) {
        set_err(err, "cluster all-reduce: null device pointer");
        return false;
    }

    // ggml_backend_cuda_get_stream returns the stream ggml_backend_graph_compute
    // enqueues on (cuda_ctx->stream()), so the collective is ordered after the
    // producing kernels.
    cluster::DeviceStream stream = ggml_backend_cuda_get_stream(backend);
    trace_coll(rt.rank(), "host-allreduce", n, (const void *) stream);

    const uint64_t bytes_f32 = (uint64_t) n * sizeof(float);
    const cluster::AllreduceDType dtype = rt.cfg->allreduce_dtype;
    // `auto` resolves to F32 for every size. The bf16
    // compressed collective rounds each per-layer partial to 8 mantissa
    // bits; over 43 layers of a multi-token prefill (>256 KiB payload, which
    // `auto` would have sent as bf16) that visibly changes the first
    // generated token. bf16 stays available as an explicit
    // --cluster-allreduce-dtype bf16. `auto` can use the bf16 threshold once
    // a bf16 prefill is token-identical to f32 on the qualification prompts.
    const bool use_bf16 = dtype == cluster::AllreduceDType::BF16;
    if (dtype == cluster::AllreduceDType::Auto && bytes_f32 > cluster::kAutoBf16ThresholdBytes) {
        static bool logged_auto_f32 = false;
        if (!logged_auto_f32) {
            logged_auto_f32 = true;
            std::fprintf(stderr,
                         "[deepseek4-cluster] allreduce dtype auto: using f32 for %llu-byte "
                         "partials (bf16 auto-switch disabled until verified)\n",
                         (unsigned long long) bytes_f32);
        }
    }

    bool ok = false;
    uint64_t payload = 0;
    if (use_bf16) {
        if (!rt.ensure_scratch(backend, n, err)) return false;
        ok = rt.comm->allreduce_sum_bf16_compressed(dev_ptr, n, rt.scratch_bf16->data, stream, err);
        payload = (uint64_t) n * 2u;
    } else {
        ok = rt.comm->allreduce_sum_f32(dev_ptr, n, stream, err);
        payload = bytes_f32;
    }
    if (!ok) {
        if (err && err->empty()) *err = "cluster all-reduce enqueue failed";
        return false;
    }
    rt.telemetry.allreduce_calls += 1;
    rt.telemetry.allreduce_bytes += payload;
    if (telemetry) telemetry->cluster_allreduce_bytes += payload;

    if (force_wait || rt.trace) {
        const auto wait_t0 = ClusterClock::now();
        if (!rt.comm->wait_stream(stream, rt.cfg->timeout_ms, err)) {
            if (err && err->empty()) *err = "cluster all-reduce wait timed out";
            return false;
        }
        const uint64_t wait_us = us_since(wait_t0);
        rt.telemetry.allreduce_wait_us += wait_us;
        if (rt.trace) {
            std::fprintf(stderr,
                         "[deepseek4-cluster] rank %d layer %d allreduce n=%zu %s wait=%llu us\n",
                         rt.rank(), layer, n, use_bf16 ? "bf16" : "f32",
                         (unsigned long long) wait_us);
        }
    }
    return true;
}

}  // namespace

// ─── Ds4ClusterRuntime ─────────────────────────────────────────────────

Ds4ClusterRuntime::~Ds4ClusterRuntime() {
    free_scratch();
}

bool Ds4ClusterRuntime::evaluates(int layer, int expert, int slot) const {
    if (slot >= 0 && slot < slot_validity_n_slots && !slot_validity_cache.empty() &&
        layer >= 0 && layer < placement.n_layer && expert >= 0 && expert < placement.n_expert) {
        const size_t idx =
            ((size_t) layer * (size_t) slot_validity_n_slots + (size_t) slot) *
                (size_t) placement.n_expert + (size_t) expert;
        if (idx < slot_validity_cache.size()) return slot_validity_cache[idx] != 0;
    }
    return placement.slot_owner(layer, expert, slot) == rank();
}

bool Ds4ClusterRuntime::ensure_scratch(ggml_backend_t backend, size_t n_elems, std::string * err) {
    if (scratch_buf && scratch_backend == backend && scratch_elems >= n_elems) {
        return true;
    }
    free_scratch();
    ggml_init_params ip{};
    ip.mem_size = 4 * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    scratch_ctx = ggml_init(ip);
    if (!scratch_ctx) {
        set_err(err, "cluster scratch: ggml_init failed");
        return false;
    }
    scratch_f32 = ggml_new_tensor_1d(scratch_ctx, GGML_TYPE_F32, (int64_t) n_elems);
    ggml_set_name(scratch_f32, "ds4_cluster_scratch_f32");
    // bf16 payload for the compressed collective: 2 bytes per element.
    scratch_bf16 = ggml_new_tensor_1d(scratch_ctx, GGML_TYPE_I16, (int64_t) n_elems);
    ggml_set_name(scratch_bf16, "ds4_cluster_scratch_bf16");
    scratch_buf = ggml_backend_alloc_ctx_tensors(scratch_ctx, backend);
    if (!scratch_buf) {
        free_scratch();
        set_err(err, "cluster scratch: device allocation failed");
        return false;
    }
    scratch_backend = backend;
    scratch_elems = n_elems;
    return true;
}

void Ds4ClusterRuntime::free_scratch() {
    if (scratch_buf) { ggml_backend_buffer_free(scratch_buf); scratch_buf = nullptr; }
    if (scratch_ctx) { ggml_free(scratch_ctx); scratch_ctx = nullptr; }
    scratch_f32 = nullptr;
    scratch_bf16 = nullptr;
    scratch_elems = 0;
    scratch_backend = nullptr;
}

// ─── Config / placement ────────────────────────────────────────────────

MoeHybridConfig ds4_cluster_moe_config(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    cfg.swiglu_clamp = w.swiglu_clamp_exp;
    cfg.cold_expert_backend = MoeHybridColdBackend::None;
    cfg.materialize_hot_experts = true;
    cfg.materialize_cold_experts = false;
    // Same as the single-node DS4 hybrid config: reduced hot stacks stay on
    // the q<=4 MMVQ sub-batch path on gfx1151 (mmq_safe_full_batch=false).
    return cfg;
}

bool ds4_cluster_build_placement(const cluster::ClusterConfig & cfg,
                                 const DeepSeek4Weights & w,
                                 const std::string * hotness_csv_path,
                                 Ds4ClusterRuntime & rt,
                                 std::string * err) {
    if (!cfg.enabled() || cfg.rank < 0 || cfg.rank >= cfg.size) {
        set_err(err, "cluster config is not enabled or rank is out of range");
        return false;
    }
    if (cfg.shared_expert == cluster::SharedExpertMode::Rank0) {
        set_err(err, std::string("shared expert mode '") +
                     cluster::shared_expert_mode_name(cfg.shared_expert) +
                     "' is not implemented yet; use replicate or shard");
        return false;
    }

    const MoeHybridConfig mcfg = ds4_cluster_moe_config(w);
    MoeHybridRoutingStats stats;
    bool have_stats = false;
    const bool need_stats =
        cfg.placement_source == cluster::PlacementSource::Balanced ||
        (cfg.placement_source == cluster::PlacementSource::Uniform && cfg.replicate_hot > 0);
    if (need_stats) {
        if (!hotness_csv_path || hotness_csv_path->empty()) {
            set_err(err, std::string(placement_source_label(cfg.placement_source)) +
                         " placement with replicate_hot/balanced requires LUCE_DS4_HOTNESS_CSV");
            return false;
        }
        if (!MoeHybridRoutingStats::load_csv(*hotness_csv_path, stats, err)) return false;
        if (!stats.matches(mcfg)) {
            set_err(err, "routing hotness CSV shape does not match the DeepSeek V4 target "
                         "(n_layer/n_expert/n_expert_used)");
            return false;
        }
        have_stats = true;
    }

    switch (cfg.placement_source) {
        case cluster::PlacementSource::Uniform:
            if (!cluster::ClusterExpertPlacement::build_uniform(
                    cfg.size, mcfg, cfg.replicate_hot, have_stats ? &stats : nullptr,
                    rt.placement, err)) {
                return false;
            }
            break;
        case cluster::PlacementSource::Balanced:
            if (!cluster::ClusterExpertPlacement::build_balanced(
                    cfg.size, stats, cfg.replicate_hot, rt.placement, err)) {
                return false;
            }
            break;
        case cluster::PlacementSource::File:
            if (!cluster::ClusterExpertPlacement::load_json(cfg.placement_file, rt.placement, err)) {
                return false;
            }
            if (rt.placement.n_ranks != cfg.size) {
                set_err(err, "placement file was built for " + std::to_string(rt.placement.n_ranks) +
                             " ranks, cluster size is " + std::to_string(cfg.size));
                return false;
            }
            if (!rt.placement.matches(mcfg)) {
                set_err(err, "placement file dimensions do not match the DeepSeek V4 target");
                return false;
            }
            break;
    }

    if (!rt.placement.to_rank_placement(cfg.rank, rt.rank_placement, err)) return false;
    rt.placement.slot_validity(cfg.rank, kSlotValidityCacheSlots, rt.slot_validity_cache);
    rt.slot_validity_n_slots = kSlotValidityCacheSlots;
    if (&cfg != &rt.cfg_storage) rt.cfg_storage = cfg;
    rt.cfg = &rt.cfg_storage;
    rt.trace = cluster::cluster_env_trace();

    // Shared-expert sharding: this rank's slice of the intermediate axis. The
    // down projection contracts over that axis, so the slice is a partial sum
    // and has to be added to the routed partial BEFORE the all-reduce.
    rt.shexp_ff_begin = 0;
    rt.shexp_ff_count = 0;
    if (cfg.shared_expert == cluster::SharedExpertMode::Shard && cfg.size > 1) {
        if (w.n_ff_exp > 0 && w.n_ff_exp % cfg.size == 0) {
            rt.shexp_ff_count = w.n_ff_exp / cfg.size;
            rt.shexp_ff_begin = cfg.rank * rt.shexp_ff_count;
            if (w.cluster_shexp_ff_local > 0) {
                if (w.cluster_shexp_ff_local != rt.shexp_ff_count) {
                    set_err(err, "loaded shared-expert slice does not match the shard");
                    return false;
                }
                rt.shexp_ff_begin = 0;   // the loaded tensors are this slice
            }
            std::fprintf(stderr,
                         "[deepseek4-cluster] rank %d/%d shared expert intermediate %d..%d "
                         "of %d; its partial rides in the routed all-reduce\n",
                         cfg.rank, cfg.size, rt.shexp_ff_begin,
                         rt.shexp_ff_begin + rt.shexp_ff_count - 1, w.n_ff_exp);
        } else {
            std::fprintf(stderr,
                         "[deepseek4-cluster] shared expert stays replicated: %d "
                         "intermediate units do not split %d ways\n",
                         w.n_ff_exp, cfg.size);
        }
    }

    // Attention head parallelism. The split has to be in whole output groups,
    // because the grouped output projection's first stage is indexed by group;
    // a rank count that does not divide the groups (or the heads) keeps
    // attention replicated instead of splitting it unevenly.
    rt.attn_head_begin = 0;
    rt.attn_head_count = 0;
    if (ds4_cluster_attention_parallel_enabled() && cfg.size > 1 &&
        w.n_out_group > 0 && w.n_head > 0 &&
        w.n_out_group % cfg.size == 0 && w.n_head % cfg.size == 0) {
        rt.attn_head_count = w.n_head / cfg.size;
        rt.attn_head_begin = cfg.rank * rt.attn_head_count;
        if (w.cluster_heads_local > 0) {
            if (w.cluster_heads_local != rt.attn_head_count) {
                set_err(err, "loaded attention slice does not match the head split");
                return false;
            }
            rt.attn_head_begin = 0;   // the loaded tensors are this slice
        }
        std::fprintf(stderr,
                     "[deepseek4-cluster] rank %d/%d attention heads %d..%d of %d "
                     "(%d of %d output groups); the attention output is a partial sum "
                     "and is all-reduced with the routed experts\n",
                     cfg.rank, cfg.size, rt.attn_head_begin,
                     rt.attn_head_begin + rt.attn_head_count - 1, w.n_head,
                     w.n_out_group / cfg.size, w.n_out_group);
    } else if (cfg.size > 1) {
        std::fprintf(stderr,
                     "[deepseek4-cluster] rank %d/%d attention stays replicated "
                     "(%d heads / %d output groups do not split %d ways, or the "
                     "kill-switch is set)\n",
                     cfg.rank, cfg.size, w.n_head, w.n_out_group, cfg.size);
    }

    std::fprintf(stderr, "[deepseek4-cluster] rank %d/%d %s source=%s resident_experts=%d/%d\n",
                 cfg.rank, cfg.size,
                 rt.placement.describe(have_stats ? &stats : nullptr).c_str(),
                 placement_source_label(cfg.placement_source),
                 rt.rank_placement.total_hot, w.n_layer * w.n_expert);
    return true;
}

bool ds4_cluster_init_experts(const std::string & model_path,
                              ggml_backend_t backend,
                              const DeepSeek4Weights & w,
                              Ds4ClusterRuntime & rt,
                              MoeHybridStorage & storage,
                              std::string * err) {
    std::string mix_what;
    const bool has_mix = ds4_cluster_has_mix_experts(w, &mix_what);
    const MoeHybridConfig mcfg = ds4_cluster_moe_config(w);
    if (!rt.rank_placement.matches(mcfg)) {
        set_err(err, "rank placement does not match the loaded model dimensions");
        return false;
    }

    // The non-mmap builder materializes exactly the hot (resident) slices of
    // every expert tensor from the file and unmaps afterwards; with cold owner
    // None nothing else is read, so no streaming engine or retained mmap is
    // needed. (build_deepseek4_moe_hybrid_storage_from_file_with_mmap would
    // keep the whole file mapped for cold-expert streaming.)
    if (!build_deepseek4_moe_hybrid_storage_from_file(
            model_path, backend, w, rt.rank_placement, &mcfg, storage, err)) {
        return false;
    }
    if (storage.cold_backend_kind != MoeHybridColdBackend::None ||
        storage.materialized_cold_experts) {
        set_err(err, "cluster expert storage did not come back with cold owner None");
        return false;
    }

    uint64_t bytes = 0;
    int resident = 0;
    int layers_with_experts = 0;
    for (const MoeHybridLayerStorage & layer : storage.layers) {
        if (!layer.hot_buf) continue;
        bytes += (uint64_t) ggml_backend_buffer_get_size(layer.hot_buf);
        resident += (int) layer.hot_expert_ids.size();
        ++layers_with_experts;
        if (!layer.cold_expert_ids.empty() || layer.cold_buf) {
            set_err(err, "cluster expert storage unexpectedly holds cold experts");
            return false;
        }
    }
    // Adaptive (mixed ROCmFPX) experts keep their codebooks out of band and
    // need a decode table registered per resident tensor. The shard's resident
    // set is exactly its hot experts, and the registrar treats an absent cold
    // owner as "nothing to register", so hot-only storage is enough.
    if (has_mix) {
        std::string why;
        if (!register_deepseek4_moe_hybrid_mix_tables(model_path, w, storage, &why)) {
            set_err(err, "cluster shard could not register decode tables for " +
                         mix_what + " experts" + (why.empty() ? "" : ": " + why));
            return false;
        }
    }
    if (has_mix) {
        std::fprintf(stderr,
                     "[deepseek4-cluster] rank %d/%d registered %s decode tables for "
                     "its resident experts\n", rt.rank(), rt.size(), mix_what.c_str());
    }

    rt.resident_expert_bytes = bytes;
    const int total_experts = w.n_layer * w.n_expert;
    std::fprintf(stderr,
                 "[deepseek4-cluster] rank %d/%d resident routed experts: %d/%d (%.1f%%) "
                 "%.2f GiB (%.1f MiB/layer over %d layers), owned=%d replicated=%d\n",
                 rt.rank(), rt.size(), resident, total_experts,
                 total_experts > 0 ? 100.0 * (double) resident / (double) total_experts : 0.0,
                 (double) bytes / (1024.0 * 1024.0 * 1024.0),
                 layers_with_experts > 0
                     ? (double) bytes / (1024.0 * 1024.0) / (double) layers_with_experts : 0.0,
                 layers_with_experts,
                 rt.placement.total_owned(rt.rank()),
                 resident - rt.placement.total_owned(rt.rank()));
    return true;
}

bool ds4_cluster_ingraph_allreduce_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("LUCE_CLUSTER_NO_INGRAPH_ALLREDUCE");
        const bool off = v && v[0] && std::strcmp(v, "0") != 0;
        if (off) {
            std::fprintf(stderr,
                         "[deepseek4-cluster] LUCE_CLUSTER_NO_INGRAPH_ALLREDUCE=1: "
                         "per-layer host all-reduce, no fused graph\n");
        }
        return !off;
    }();
    return enabled;
}

bool ds4_cluster_prefill_pipeline_enabled(const std::vector<int> & bands) {
    if (!ds4_cluster_prefill_device_join_enabled() || bands.empty()) return false;
    static const int forced = [] {   // -1 unset, 0 off, 1 on
        const char * v = std::getenv("LUCE_CLUSTER_PREFILL_PIPELINE");
        return v && *v ? (std::strcmp(v, "0") != 0 ? 1 : 0) : -1;
    }();
    if (forced >= 0) return forced == 1;
    return std::all_of(bands.begin(), bands.end(),
                       [](int rows) { return rows >= kDs4ClusterPipelineMinBand; });
}

bool ds4_cluster_prefill_device_join_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("LUCE_CLUSTER_PREFILL_DEVICE_JOIN");
        return !v || !*v || std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

bool ds4_cluster_attention_parallel_enabled() {
    // Off unless set; cluster mode sets it (cluster_launch_env), and =0 keeps
    // attention replicated.
    static const bool enabled = [] {
        const char * v = std::getenv("LUCE_CLUSTER_ATTENTION_PARALLEL");
        return v && v[0] && std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

bool ds4_cluster_fused_graph_available(const Ds4ClusterRuntime * rt) {
    if (!rt || !rt->comm || rt->comm->size() <= 1) return false;
    if (!ds4_cluster_ingraph_allreduce_enabled()) return false;
    if (rt->cfg && rt->cfg->shared_expert == cluster::SharedExpertMode::Rank0) {
        return false;
    }
    for (int32_t owner : rt->placement.owner) {
        if (owner == cluster::kReplicated) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr,
                             "[deepseek4-cluster] fused graph unavailable: a "
                             "replicated expert is owned per token slot, which the "
                             "fused graph's per-expert owner table cannot express; "
                             "using the per-layer host all-reduce\n");
            }
            return false;
        }
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::fprintf(stderr,
                     "[deepseek4-cluster] fused whole-model graph with an "
                     "in-graph all-reduce per MoE layer\n");
    }
    return true;
}

// ─── Route masking ──────────────────────────────────────────────────────

void ds4_cluster_mask_routes(const Ds4ClusterRuntime & rt,
                             int layer,
                             int32_t * selected,
                             float * weights,
                             int route_width,
                             int n_tokens) {
    if (!selected || !weights || route_width <= 0 || n_tokens <= 0) return;
    for (int t = 0; t < n_tokens; ++t) {
        int32_t * ids = selected + (size_t) t * (size_t) route_width;
        float * wts = weights + (size_t) t * (size_t) route_width;
        for (int j = 0; j < route_width; ++j) {
            const int32_t e = ids[j];
            if (e < 0) continue;
            if (!rt.evaluates(layer, (int) e, t)) {
                ids[j] = -1;
                wts[j] = 0.0f;
            }
        }
    }
}

// ─── All-reduce ─────────────────────────────────────────────────────────

// Path 3b: invoked by GGML_MOE_FUSED_CLUSTER_ALLREDUCE while the graph runs.
// It enqueues the collective on the stream ggml handed us, which is the same
// stream the surrounding kernels use, so ordering needs no host synchronize.
static void ds4_cluster_allreduce_graph_callback(void * user, void * data,
                                                 size_t n, void * stream) {
    Ds4ClusterRuntime * rt = static_cast<Ds4ClusterRuntime *>(user);
    if (!rt || !rt->comm || rt->comm->size() <= 1 || n == 0 || !data) return;
    // LUCE_CLUSTER_PREFILL_BF16=1: prefill-sized partials cross the link as
    // bf16 (RCCL sums in bf16; every rank gets the same result). Decode-sized
    // partials never reach this branch (they stay on fast_reduce).
    static const bool prefill_bf16 = [] {
        const char * v = std::getenv("LUCE_CLUSTER_PREFILL_BF16");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    if (prefill_bf16 && n >= (size_t) 262144 && rt->scratch_bf16 && rt->scratch_elems >= n) {
        trace_coll(rt->rank(), "graph-allreduce-bf16", n, stream);
        std::string err16;
        if (!rt->comm->allreduce_sum_bf16_compressed(data, n, rt->scratch_bf16->data,
                                                     (cluster::DeviceStream) stream, &err16)) {
            if (rt->node_error.empty()) rt->node_error = err16.empty() ? "in-graph bf16 all-reduce failed" : err16;
            return;
        }
        rt->telemetry.allreduce_calls += 1;
        rt->telemetry.allreduce_bytes += (uint64_t) n * 2u;
        return;
    }
    if (rt->fast && rt->fast->ok() && rt->fast->submit((float *) data, n, stream)) {
        trace_coll(rt->rank(), "fast-allreduce", n, stream);
        rt->telemetry.allreduce_calls += 1;
        rt->telemetry.allreduce_bytes += (uint64_t) n * sizeof(float);
        return;
    }
    trace_coll(rt->rank(), "graph-allreduce", n, stream);
    std::string err;
    if (!rt->comm->allreduce_sum_f32(data, n, (cluster::DeviceStream) stream, &err)) {
        // Keep the first failure: the rest of the graph still runs, and the
        // caller turns this into a failed forward once the compute returns.
        if (rt->node_error.empty()) {
            rt->node_error = err.empty() ? "in-graph all-reduce failed" : err;
        }
        return;
    }
    rt->telemetry.allreduce_calls += 1;
    rt->telemetry.allreduce_bytes += (uint64_t) n * sizeof(float);
}

// Hybrid exchange callbacks. Each kind counts its own calls; the k-th call
// of every kind belongs to exchange k, so the host order of the dGPU and
// iGPU splits does not matter. A failure is fatal for the step: the peer is
// already waiting on this rank's row.
static void ds4_hyb_fail(Ds4ClusterRuntime * rt, const char * what) {
    if (rt->node_error.empty()) rt->node_error = std::string("hybrid exchange: ") + what + " failed";
}
static void ds4_hyb_export0_cb(void * user, void * data, size_t n, void * stream) {
    auto * rt = static_cast<Ds4ClusterRuntime *>(user);
    if (!rt || !rt->fast || !rt->fast->hyb_export((const float *) data, n, stream, 0, rt->hyb_export0++)) ds4_hyb_fail(rt, "export (hot)");
}
static void ds4_hyb_export1_cb(void * user, void * data, size_t n, void * stream) {
    auto * rt = static_cast<Ds4ClusterRuntime *>(user);
    if (!rt || !rt->fast || !rt->fast->hyb_export((const float *) data, n, stream, 1, rt->hyb_export1++)) ds4_hyb_fail(rt, "export (shared)");
}
static void ds4_hyb_combine_cb(void * user, void * data, size_t n, void * stream) {
    auto * rt = static_cast<Ds4ClusterRuntime *>(user);
    if (!rt || !rt->fast || !rt->fast->hyb_combine((const float *) data, n, stream, rt->hyb_combine++)) ds4_hyb_fail(rt, "combine");
    else { rt->telemetry.allreduce_calls += 1; rt->telemetry.allreduce_bytes += (uint64_t) n * sizeof(float); }
}
static void ds4_hyb_import_cb(void * user, void * data, size_t n, void * stream) {
    auto * rt = static_cast<Ds4ClusterRuntime *>(user);
    if (!rt || !rt->fast || !rt->fast->hyb_import((float *) data, n, stream, rt->hyb_import++)) ds4_hyb_fail(rt, "import");
}

bool ds4_cluster_hybrid_exchange_enabled(const Ds4ClusterRuntime & rt) {
    static const bool env = [] {
        const char * v = std::getenv("LUCE_CLUSTER_HYBRID_EXCHANGE");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    return env && rt.comm && rt.comm->size() == 2 && rt.fast && rt.fast->hyb_available();
}

ggml_tensor * ds4_cluster_hybrid_exchange_nodes(ggml_context * ctx,
                                                ggml_tensor * hot,
                                                ggml_tensor * shared,
                                                ggml_tensor * cold,
                                                Ds4ClusterRuntime & rt,
                                                ggml_tensor ** combine_node,
                                                ggml_tensor ** export_hot_node,
                                                ggml_tensor ** export_shared_node) {
    if (!ctx || !hot || !shared || !cold || !combine_node || !export_hot_node || !export_shared_node) return nullptr;
    for (ggml_tensor * t : {hot, shared, cold}) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) != ggml_nelements(hot)) return nullptr;
    }
    // In place throughout: each node passes its own tensor to its callback.
    // LUCE_CLUSTER_HYBRID_ONE_EXPORT=1: export hot + shared as one partial
    // (half the dGPU link bytes; the iGPU adds cold + (hot + shared)).
    static const bool one_export = [] {
        const char * v = std::getenv("LUCE_CLUSTER_HYBRID_ONE_EXPORT");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    ggml_tensor * ex0 = nullptr;
    ggml_tensor * ex1 = nullptr;
    if (one_export) {
        ggml_tensor * hs = ggml_add(ctx, hot, shared);
        ex1 = ggml_cluster_allreduce_inplace(ctx, hs, &ds4_hyb_export0_cb, &rt);
        ex0 = ex1;
    } else {
        ex0 = ggml_cluster_allreduce_inplace(ctx, hot, &ds4_hyb_export0_cb, &rt);
        ex1 = ggml_cluster_allreduce_inplace(ctx, shared, &ds4_hyb_export1_cb, &rt);
    }
    ggml_tensor * comb = ggml_cluster_allreduce_inplace(ctx, cold, &ds4_hyb_combine_cb, &rt);
    // The import overwrites the exported shared partial with the reduced sum.
    ggml_tensor * imp = ggml_cluster_allreduce_inplace(ctx, ex1, &ds4_hyb_import_cb, &rt);
    *combine_node = comb;
    *export_hot_node = ex0;
    *export_shared_node = ex1;
    return imp;
}

ggml_cluster_allreduce_fn ds4_cluster_allreduce_fn() {
    return &ds4_cluster_allreduce_graph_callback;
}

ggml_tensor * ds4_cluster_allreduce_node(ggml_context * ctx,
                                         ggml_tensor * partial,
                                         Ds4ClusterRuntime & rt,
                                         bool inplace) {
    if (!ctx || !partial) return nullptr;
    if (!rt.comm || rt.comm->size() <= 1) return partial;
    if (partial->type != GGML_TYPE_F32 || !ggml_is_contiguous(partial)) {
        std::fprintf(stderr,
                     "[deepseek4-cluster] in-graph all-reduce needs a contiguous "
                     "F32 partial\n");
        return nullptr;
    }
    return inplace
        ? ggml_cluster_allreduce_inplace(ctx, partial,
                                         &ds4_cluster_allreduce_graph_callback, &rt)
        : ggml_cluster_allreduce(ctx, partial,
                                 &ds4_cluster_allreduce_graph_callback, &rt);
}

bool ds4_cluster_allreduce_layer(Ds4ClusterRuntime & rt,
                                 ggml_backend_t backend,
                                 ggml_tensor * partial,
                                 int layer,
                                 DeepSeek4StepTelemetry * telemetry,
                                 std::string * err) {
    if (!partial) {
        set_err(err, "cluster all-reduce: null partial tensor");
        return false;
    }
    if (partial->type != GGML_TYPE_F32 || !ggml_is_contiguous(partial)) {
        set_err(err, "cluster all-reduce: partial must be contiguous F32");
        return false;
    }
    const auto t0 = ClusterClock::now();
    const size_t n = (size_t) ggml_nelements(partial);
    const bool ok = allreduce_device(rt, backend, partial->data, n, layer,
                                     /*force_wait=*/false, telemetry, err);
    if (telemetry) telemetry->cluster_allreduce_us += us_since(t0);
    return ok;
}

bool ds4_cluster_allreduce_layer_host(Ds4ClusterRuntime & rt,
                                      ggml_backend_t backend,
                                      float * partial_host,
                                      int n_embd,
                                      int n_tokens,
                                      int layer,
                                      DeepSeek4StepTelemetry * telemetry,
                                      std::string * err) {
    if (!rt.comm || rt.comm->size() <= 1) return true;
    if (!partial_host || n_embd <= 0 || n_tokens <= 0) {
        set_err(err, "cluster all-reduce: invalid host partial");
        return false;
    }
    const auto t0 = ClusterClock::now();
    const size_t n = (size_t) n_embd * (size_t) n_tokens;
    if (!rt.ensure_scratch(backend, n, err)) return false;
    // ggml_backend_tensor_set completes the H2D copy before returning
    // (ggml-cuda copies on cudaStreamPerThread and synchronizes it), so the
    // collective enqueued next on the backend stream reads final data.
    ggml_backend_tensor_set(rt.scratch_f32, partial_host, 0, n * sizeof(float));
    if (!allreduce_device(rt, backend, rt.scratch_f32->data, n, layer,
                          /*force_wait=*/true, telemetry, err)) {
        return false;
    }
    ggml_backend_tensor_get(rt.scratch_f32, partial_host, 0, n * sizeof(float));
    if (telemetry) telemetry->cluster_allreduce_us += us_since(t0);
    return true;
}

// ─── Lucebox additions ────────────────────────────────────────────────

namespace {

void broadcast_graph_callback(void * user, void * data, size_t n, void * stream) {
    Ds4ClusterRuntime * rt = static_cast<Ds4ClusterRuntime *>(user);
    if (!rt || !rt->comm || rt->comm->size() <= 1 || n == 0 || !data) return;
    if (rt->before_broadcast && !rt->before_broadcast(data, n, stream)) {
        if (rt->node_error.empty()) rt->node_error = "upload before the in-graph broadcast failed";
    }
    if (rt->fast && rt->fast->ok() &&
        rt->fast->submit_from_root((float *) data, n, stream, /*root=*/0)) {
        trace_coll(rt->rank(), "fast-broadcast", n, stream);
        return;
    }
    trace_coll(rt->rank(), "graph-broadcast", n, stream);
    std::string err;
    if (!rt->comm->broadcast_bytes(data, n * sizeof(float), /*root=*/0,
                                   (cluster::DeviceStream) stream, &err)) {
        if (rt->node_error.empty()) {
            rt->node_error = err.empty() ? "in-graph broadcast failed" : err;
        }
    }
}

// Rank 0's values bit for bit: f32 on the wire even where the fast reduce
// rounds its payloads to bf16. No upload hook (that belongs to the Engram keys).
void broadcast_exact_graph_callback(void * user, void * data, size_t n, void * stream) {
    Ds4ClusterRuntime * rt = static_cast<Ds4ClusterRuntime *>(user);
    if (!rt || !rt->comm || rt->comm->size() <= 1 || n == 0 || !data) return;
    if (rt->fast && rt->fast->ok() &&
        rt->fast->submit_from_root((float *) data, n, stream, /*root=*/0, /*exact=*/true)) {
        trace_coll(rt->rank(), "fast-broadcast-exact", n, stream);
        return;
    }
    trace_coll(rt->rank(), "graph-broadcast", n, stream);
    std::string err;
    if (!rt->comm->broadcast_bytes(data, n * sizeof(float), /*root=*/0,
                                   (cluster::DeviceStream) stream, &err)) {
        if (rt->node_error.empty()) {
            rt->node_error = err.empty() ? "in-graph broadcast failed" : err;
        }
    }
}

}  // namespace

ggml_tensor * ds4_cluster_broadcast_exact_node(ggml_context * ctx,
                                               ggml_tensor * value,
                                               Ds4ClusterRuntime & rt) {
    if (!ctx || !value) return nullptr;
    if (rt.size() <= 1) return value;
    if (value->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[deepseek4-cluster] broadcast needs an F32 tensor\n");
        return nullptr;
    }
    if (!ggml_is_contiguous(value)) value = ggml_cont(ctx, value);
    return ggml_cluster_allreduce(ctx, value, &broadcast_exact_graph_callback, &rt);
}

ggml_tensor * ds4_cluster_broadcast_node(ggml_context * ctx,
                                         ggml_tensor * value,
                                         Ds4ClusterRuntime & rt) {
    if (!ctx || !value) return nullptr;
    if (rt.size() <= 1) return value;
    if (value->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[deepseek4-cluster] broadcast needs an F32 tensor\n");
        return nullptr;
    }
    if (!ggml_is_contiguous(value)) value = ggml_cont(ctx, value);
    // The in-graph collective node runs its callback in place on a copy of
    // `value`; this callback is a broadcast from rank 0.
    return ggml_cluster_allreduce(ctx, value, &broadcast_graph_callback, &rt);
}

bool ds4_cluster_broadcast_host(Ds4ClusterRuntime & rt,
                                ggml_backend_t backend,
                                float * data,
                                size_t n,
                                std::string * err) {
    if (!rt.comm || rt.comm->size() <= 1 || n == 0) return true;
    if (!data) {
        set_err(err, "cluster broadcast: null host buffer");
        return false;
    }
    if (!rt.ensure_scratch(backend, n, err)) return false;
    cluster::DeviceStream stream = ggml_backend_cuda_get_stream(backend);
    trace_coll(rt.rank(), "host-broadcast", n, (const void *) stream);
    ggml_backend_tensor_set(rt.scratch_f32, data, 0, n * sizeof(float));
    if (!rt.comm->broadcast_bytes(rt.scratch_f32->data, n * sizeof(float), 0, stream, err)) {
        return false;
    }
    if (!rt.comm->wait_stream(stream, rt.cfg ? rt.cfg->timeout_ms : 30000u, err)) return false;
    ggml_backend_tensor_get(rt.scratch_f32, data, 0, n * sizeof(float));
    return true;
}

bool ds4_cluster_allgather_i32(Ds4ClusterRuntime & rt,
                               ggml_backend_t backend,
                               int32_t value,
                               std::vector<int32_t> & out,
                               std::string * err) {
    const int n = std::max(1, rt.size());
    std::vector<float> slots((size_t) n, 0.0f);
    slots[(size_t) std::max(0, rt.rank())] = (float) value;   // exact below 2^24
    if (rt.comm && rt.comm->size() > 1 &&
        !ds4_cluster_allreduce_layer_host(rt, backend, slots.data(), n, 1, -1, nullptr, err)) {
        return false;
    }
    out.resize((size_t) n);
    for (int i = 0; i < n; ++i) out[(size_t) i] = (int32_t) slots[(size_t) i];
    return true;
}

bool ds4_cluster_resident(const Ds4ClusterRuntime & rt, int layer, int expert) {
    return rt.rank_placement.is_hot(layer, expert);
}

}  // namespace luce::common
