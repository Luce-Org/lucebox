// MoE hybrid prefill streaming engine — DMA pipeline for cold expert offload.
//
// Streams cold expert weight slices from mmap (page cache) through a pinned
// host staging buffer to GPU scratch memory, pipelined with GPU compute on
// previously-transferred experts. Used during prefill when T >= threshold.

#pragma once

#include "moe_hybrid_types.h"
#include "moe_hybrid_storage.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

// Configuration for the stream engine.
struct MoeStreamConfig {
    int  prefill_threshold = 8;    // min n_tokens to activate streaming
    int  prefetch_layers   = 2;    // how many layers ahead to madvise
};

// Streaming engine: manages pinned staging buffer, GPU scratch, and DMA pipeline.
class MoeHybridStreamEngine {
public:
    MoeHybridStreamEngine() = default;
    ~MoeHybridStreamEngine();

    MoeHybridStreamEngine(const MoeHybridStreamEngine &) = delete;
    MoeHybridStreamEngine & operator=(const MoeHybridStreamEngine &) = delete;
    MoeHybridStreamEngine(MoeHybridStreamEngine &&) noexcept;
    MoeHybridStreamEngine & operator=(MoeHybridStreamEngine &&) noexcept;

    // Initialize the engine with a maximum expert size (bytes for one expert's
    // gate+up+down tensors). Allocates pinned host buffer and GPU scratch.
    bool init(ggml_backend_t gpu_backend, size_t max_expert_bytes, std::string * err = nullptr);
    bool is_ready() const;
    void destroy();

    // Issue madvise(WILLNEED) for the specified cold experts in the given layer.
    // Call this as early as possible (e.g. at start of layer or N layers ahead).
    void prefetch_cold_experts(const void * mmap_data, size_t mmap_size,
                               const LayerExpertRegions & regions,
                               const int32_t * cold_expert_ids,
                               int n_cold);

    // Stream a single cold expert from mmap to GPU scratch and return a ggml
    // tensor view over the scratch memory for each weight matrix.
    // This is a BLOCKING operation (synchronous DMA). For pipelined usage,
    // use the async variants below.
    bool stream_expert_sync(const void * mmap_data, size_t mmap_size,
                            const LayerExpertRegions & regions,
                            int expert_id,
                            ggml_backend_t gpu_backend,
                            std::string * err = nullptr);

    // Get tensor pointers into the GPU scratch buffer after a successful stream.
    // Valid until next stream call. Tensors are transient (not owned by any context).
    const void * scratch_gate_data() const { return scratch_gate_; }
    const void * scratch_up_data()   const { return scratch_up_; }
    const void * scratch_down_data() const { return scratch_down_; }
    size_t scratch_gate_bytes() const { return last_gate_bytes_; }
    size_t scratch_up_bytes()   const { return last_up_bytes_; }
    size_t scratch_down_bytes() const { return last_down_bytes_; }

    // Total pinned buffer size.
    size_t pinned_bytes() const { return pinned_size_; }
    // Total GPU scratch size.
    size_t scratch_bytes() const { return scratch_size_; }

private:
    void * pinned_buf_  = nullptr;  // cudaMallocHost'd staging buffer
    size_t pinned_size_ = 0;

    void * gpu_scratch_     = nullptr;  // GPU device memory for one expert
    size_t scratch_size_    = 0;
    ggml_backend_t backend_ = nullptr;

    // Offsets into scratch for last-streamed expert
    void * scratch_gate_ = nullptr;
    void * scratch_up_   = nullptr;
    void * scratch_down_ = nullptr;
    size_t last_gate_bytes_ = 0;
    size_t last_up_bytes_   = 0;
    size_t last_down_bytes_ = 0;
};

// Evaluate cold experts by streaming from mmap to GPU, pipelined.
// Hot experts are already computed (result in hot_partial).
// Returns combined cold expert contribution in out (sized n_embd * n_tokens).
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
    std::string *                   err = nullptr);

}  // namespace dflash::common
