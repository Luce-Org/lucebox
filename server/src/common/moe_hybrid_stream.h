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
    // Optional adaptive GPU expert-cache budget. Zero keeps only the pipeline
    // slots. The hardware planner can safely assign otherwise-unused Strix
    // memory here while retaining its KV/graph reserve.
    size_t device_cache_bytes = 0;
    MoeNvmeConfig nvme{};

    static MoeStreamConfig from_env();
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

private:
    struct Runtime;
    std::unique_ptr<Runtime> runtime_;
};

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
