// Heterogeneous MoE SSD execution tier.
//
// Exact routed experts move through a bounded NVMe -> page-locked host -> GPU
// pipeline. The model storage format remains separate: this runtime only
// consumes model-neutral LayerExpertRegions produced by a loader.

#pragma once

#include "moe_hybrid_types.h"
#include "moe_hybrid_storage.h"
#include "moe_nvme_scheduler.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeStreamConfig {
    int prefill_threshold = 8;
    int prefetch_layers = 2;
    int device_slots = 2; // double buffering is the minimum useful pipeline
    // Persistent compute graphs are keyed by tensor types, dimensions,
    // activation, scales, and batch width. A small bounded cache removes graph
    // construction from decode without assuming every layer uses one format.
    int graph_cache_entries = 8;
    // Optional adaptive GPU expert-cache budget. Zero keeps only the pipeline
    // slots. The hardware planner can safely assign otherwise-unused Strix
    // memory here while retaining its KV/graph reserve.
    size_t device_cache_bytes = 0;
    MoeNvmeConfig nvme{};

    static MoeStreamConfig from_env();
};

// Complete numerical contract for one streamed gated expert. This is the
// model-adapter boundary: storage scheduling consumes byte ranges, while the
// reusable compute path consumes this shape/type/activation description.
// input_dim and output_dim may differ, although common routed FFNs use the
// same value for both.
struct MoeStreamExpertSpec {
    int input_dim = 0;
    int intermediate_dim = 0;
    int output_dim = 0;
    ggml_type gate_type = GGML_TYPE_COUNT;
    ggml_type up_type = GGML_TYPE_COUNT;
    ggml_type down_type = GGML_TYPE_COUNT;
    ggml_type gate_up_type = GGML_TYPE_COUNT;
    bool fused_gate_up = false;
    MoeGatedActivation gated_activation = MoeGatedActivation::SwiGlu;
    float swiglu_clamp = 0.0f;
    float situ_beta = 4.0f;
    float situ_linear_beta = 25.0f;
    float gate_scale = 1.0f;
    float up_scale = 1.0f;
    float down_scale = 1.0f;
    float gate_up_scale = 1.0f;
};

// Build the common contract from an existing model's ordinary MoE metadata.
// New adapters may either use this helper or populate MoeStreamExpertSpec
// directly when their latent input/output dimensions differ.
bool make_moe_stream_expert_spec(
    const MoeHybridConfig & cfg,
    const MoeLayerDesc & desc,
    const LayerExpertRegions & regions,
    MoeStreamExpertSpec & out,
    std::string * err = nullptr);

// Reject a shape/type/layout mismatch before launching a kernel. This makes a
// bad adapter fail deterministically instead of silently interpreting the
// wrong number of weight bytes.
bool validate_moe_stream_expert_layout(
    const MoeStreamExpertSpec & spec,
    const MoeExpertIoLayout & layout,
    std::string * err = nullptr);

// Model-neutral routed batch. Inputs and outputs are token-major contiguous
// F32 matrices. resident_local_by_global is optional; when supplied, IDs with
// a non-negative entry are already owned by another tier and are skipped.
struct MoeStreamRouteBatch {
    int layer = 0;
    int n_expert = 0;
    int top_k = 0;
    int n_tokens = 0;
    const float * inputs = nullptr;
    const int32_t * selected_ids = nullptr;
    const float * selected_weights = nullptr;
    const int32_t * resident_local_by_global = nullptr;
    size_t resident_map_size = 0;
};

struct MoeStreamComputeStats {
    uint64_t graph_builds = 0;
    uint64_t graph_cache_hits = 0;
    uint64_t graph_evictions = 0;
    uint64_t graph_launches = 0;
};

// Route ownership for two concurrent SSD-backed GPU owners. An explicit
// placement takes precedence and identifies the primary GPU's hot experts.
// Without one, a stable layer/expert hash supplies a deterministic capacity
// split that preserves cache locality across tokens.
struct MoeStreamDualOwnerPolicy {
    const MoeHybridPlacement * primary_placement = nullptr;
    int primary_share_per_mille = 500;

    static MoeStreamDualOwnerPolicy from_env();
};

struct MoeStreamDualOwnerStats {
    uint64_t wall_us = 0;
    uint64_t primary_us = 0;
    uint64_t secondary_us = 0;
    int primary_routes = 0;
    int secondary_routes = 0;
    int primary_experts = 0;
    int secondary_experts = 0;
};

class MoeHybridStreamEngine {
public:
    MoeHybridStreamEngine();
    ~MoeHybridStreamEngine();

    MoeHybridStreamEngine(const MoeHybridStreamEngine &) = delete;
    MoeHybridStreamEngine & operator=(const MoeHybridStreamEngine &) = delete;
    MoeHybridStreamEngine(MoeHybridStreamEngine &&) noexcept;
    MoeHybridStreamEngine & operator=(MoeHybridStreamEngine &&) noexcept;

    // Compatibility initialization for synthetic callers. Production callers
    // should use the storage overload so actual file reads (and io_uring) are
    // available instead of relying only on mmap page faults.
    bool init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
              std::string * err = nullptr);
    bool init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
              const MoeStreamConfig & config,
              std::string * err = nullptr);
    bool init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
              const MoeHybridStorage & storage,
              std::string * err = nullptr);
    bool init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
              const MoeHybridStorage & storage,
              const MoeStreamConfig & config,
              std::string * err = nullptr);

    bool bind_storage(const MoeHybridStorage & storage, std::string * err = nullptr);
    bool bind_sources(const std::vector<MoeNvmeSource> & sources,
                      const std::vector<LayerExpertRegions> & layer_regions,
                      std::string * err = nullptr);
    bool is_ready() const;
    bool is_bound() const;
    void destroy();

    // Queue exact experts without waiting. Demand requests always outrank
    // speculative prefetches and can cancel queued speculation.
    void request_experts(int layer, const int32_t * expert_ids, int count,
                         MoeNvmePriority priority = MoeNvmePriority::Prefetch);

    // Compatibility page-cache hint. New code should call request_experts(),
    // which performs real asynchronous reads into the bounded host cache.
    void prefetch_cold_experts(const void * mmap_data, size_t mmap_size,
                               const LayerExpertRegions & regions,
                               const int32_t * cold_expert_ids,
                               int n_cold);

    // Queue one H2D transfer into a device slot, then activate it after its
    // completion event. Different slots allow transfer N+1 to overlap compute N.
    bool stage_expert_async(int layer, int expert_id, int device_slot,
                            std::string * err = nullptr);

    // Cache-aware form used by production inference. On a hit it returns the
    // existing Strix slot without host or SSD traffic. On a miss it selects an
    // unpinned LFRU victim and starts the same asynchronous upload pipeline.
    bool stage_expert_cached_async(int layer, int expert_id, int * device_slot,
                                   std::string * err = nullptr);
    bool activate_device_slot(int device_slot, std::string * err = nullptr);
    void release_device_slot(int device_slot);
    int device_slot_count() const;
    size_t device_cache_bytes() const;
    ggml_backend_t compute_backend() const;

    bool stream_expert_sync(int layer, int expert_id,
                            std::string * err = nullptr);

    // Legacy form: lazily binds a single synthetic layer.
    bool stream_expert_sync(const void * mmap_data, size_t mmap_size,
                            const LayerExpertRegions & regions,
                            int expert_id,
                            ggml_backend_t gpu_backend,
                            std::string * err = nullptr);

    const void * scratch_gate_data() const;
    const void * scratch_up_data() const;
    const void * scratch_down_data() const;
    size_t scratch_gate_bytes() const;
    size_t scratch_up_bytes() const;
    size_t scratch_down_bytes() const;

    size_t pinned_bytes() const;
    size_t scratch_bytes() const;
    const char * io_backend_name() const;
    MoeNvmeStats io_stats() const;
    MoeStreamComputeStats compute_stats() const;

private:
    friend bool eval_moe_streamed_experts(
        MoeHybridStreamEngine &,
        const MoeStreamExpertSpec &,
        const MoeStreamRouteBatch &,
        std::vector<float> &,
        std::string *);
    struct Runtime;
    std::unique_ptr<Runtime> runtime_;
};

// Persistent two-owner coordinator. The secondary compute worker is created
// once at model initialization, so decode does not pay a thread create/join at
// every routed layer. Calls are deliberately serialized; each call still
// launches the primary and secondary GPU pipelines concurrently.
class MoeStreamDualOwnerExecutor {
public:
    MoeStreamDualOwnerExecutor();
    ~MoeStreamDualOwnerExecutor();

    MoeStreamDualOwnerExecutor(const MoeStreamDualOwnerExecutor &) = delete;
    MoeStreamDualOwnerExecutor & operator=(
        const MoeStreamDualOwnerExecutor &) = delete;

    bool init(MoeHybridStreamEngine & primary,
              MoeHybridStreamEngine & secondary,
              std::string * err = nullptr);
    bool is_ready() const;
    void destroy();

    bool eval(const MoeStreamExpertSpec & spec,
              const MoeStreamRouteBatch & batch,
              const MoeStreamDualOwnerPolicy & policy,
              std::vector<float> & out,
              MoeStreamDualOwnerStats * stats = nullptr,
              std::string * err = nullptr);

private:
    struct Runtime;
    std::unique_ptr<Runtime> runtime_;
};

// Evaluate exactly the experts selected by the native router. Placement only
// decides which selected IDs arrive here; no prediction or cache policy can
// change the returned mathematical function.
bool eval_moe_streamed_experts(
    MoeHybridStreamEngine & engine,
    const MoeStreamExpertSpec & spec,
    const MoeStreamRouteBatch & batch,
    std::vector<float> & out,
    std::string * err = nullptr);

// One-shot compatibility wrapper. Long-lived model adapters should initialize
// MoeStreamDualOwnerExecutor once to avoid per-layer thread startup overhead.
bool eval_moe_streamed_experts_dual_owner(
    MoeHybridStreamEngine & primary,
    MoeHybridStreamEngine & secondary,
    const MoeStreamExpertSpec & spec,
    const MoeStreamRouteBatch & batch,
    const MoeStreamDualOwnerPolicy & policy,
    std::vector<float> & out,
    MoeStreamDualOwnerStats * stats = nullptr,
    std::string * err = nullptr);

// Exposed for deterministic, GPU-free policy tests and offline plan tooling.
bool partition_moe_stream_routes(
    const MoeStreamRouteBatch & batch,
    const MoeStreamDualOwnerPolicy & policy,
    std::vector<float> & primary_weights,
    std::vector<float> & secondary_weights,
    MoeStreamDualOwnerStats * stats = nullptr,
    std::string * err = nullptr);

// Evaluate the cold contribution for one layer. All routed SSD requests are
// admitted before compute starts, then double-buffered H2D runs concurrently
// with the preceding expert graph.
bool eval_moe_cold_experts_streaming(
    MoeHybridStreamEngine &         engine,
    ggml_backend_t                  gpu_backend,
    const void *                    mmap_data,
    size_t                          mmap_size,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    const LayerExpertRegions &      regions,
    const MoeHybridLayerStorage &   storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err = nullptr,
    int                             layer = 0);

} // namespace dflash::common
