// Phase 3 hybrid expert storage for qwen35moe.

#pragma once

#include "qwen35moe_expert_placement.h"

#include "ggml-alloc.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

// Cached FFN graph for a fixed number of selected experts.
// Built once, reused every token to avoid per-call graph rebuild overhead.
struct CachedFfnGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * inp = nullptr;        // [n_embd, 1] F32 input
    ggml_tensor * ids = nullptr;        // [n_hot, 1] I32 expert IDs
    ggml_tensor * weights = nullptr;    // [n_hot, 1] F32 expert weights
    ggml_tensor * output = nullptr;     // [n_embd, 1] F32 output (routed + shared)
    int n_hot = 0;                      // number of hot experts this graph supports

    bool valid() const { return ctx && gf && alloc && output; }
    void free();
};

struct Qwen35MoeHybridLayerStorage {
    ggml_context * hot_ctx = nullptr;
    ggml_backend_buffer_t hot_buf = nullptr;
    ggml_tensor * gate_hot = nullptr;
    ggml_tensor * up_hot = nullptr;
    ggml_tensor * down_hot = nullptr;
    ggml_tensor * gate_up_hot = nullptr;

    ggml_context * cold_ctx = nullptr;
    ggml_backend_buffer_t cold_buf = nullptr;
    ggml_tensor * gate_cold = nullptr;
    ggml_tensor * up_cold = nullptr;
    ggml_tensor * down_cold = nullptr;
    ggml_tensor * gate_up_cold = nullptr;

    std::vector<int32_t> hot_expert_ids;
    std::vector<int32_t> cold_expert_ids;
    std::vector<int32_t> hot_local_by_global;
    std::vector<int32_t> cold_local_by_global;

    bool fused_gate_up = false;
    size_t gate_expert_bytes = 0;
    size_t up_expert_bytes = 0;
    size_t down_expert_bytes = 0;
    size_t gate_up_expert_bytes = 0;

    std::vector<uint8_t> gate_cold_bytes;
    std::vector<uint8_t> up_cold_bytes;
    std::vector<uint8_t> down_cold_bytes;
    std::vector<uint8_t> gate_up_cold_bytes;

    // Cached FFN graphs: hot_graph for all-hot case (n_expert_used hot experts),
    // cold_graph for all-cold case (n_expert_used cold experts).
    // These cover the common case; mixed hot/cold falls back to dynamic build.
    CachedFfnGraph hot_graph;   // GPU: fused routed(n_expert_used hot) + shared
    CachedFfnGraph cold_graph;  // CPU: routed(n_expert_used cold)
};

struct Qwen35MoeHybridStorage {
    Qwen35MoeHybridStorage() = default;
    Qwen35MoeHybridStorage(const Qwen35MoeHybridStorage &) = delete;
    Qwen35MoeHybridStorage & operator=(const Qwen35MoeHybridStorage &) = delete;
    Qwen35MoeHybridStorage(Qwen35MoeHybridStorage &&) = delete;
    Qwen35MoeHybridStorage & operator=(Qwen35MoeHybridStorage &&) = delete;
    ~Qwen35MoeHybridStorage();

    ggml_backend_t cpu_backend = nullptr;
    Qwen35MoeExpertPlacement placement;
    std::vector<Qwen35MoeHybridLayerStorage> layers;

    bool matches(const TargetWeights & w) const;
    bool empty() const;
};

bool build_qwen35moe_hybrid_storage(const TargetWeights & w,
                                    ggml_backend_t backend,
                                    const Qwen35MoeExpertPlacement & placement,
                                    Qwen35MoeHybridStorage & out,
                                    std::string * err = nullptr);

// Expert tensor file data for split loading (one entry per expert tensor).
struct ExpertTensorFileData {
    const uint8_t * data = nullptr;  // pointer into mmap
    size_t size = 0;                 // total tensor size in bytes
};

// Per-layer expert tensor file data for split loading.
struct LayerExpertFileData {
    ExpertTensorFileData gate_exps;
    ExpertTensorFileData up_exps;
    ExpertTensorFileData down_exps;
    ExpertTensorFileData gate_up_exps;  // optional fused
};

// Build hybrid storage by loading expert data directly from file (mmap).
// Expert tensors in w are only used for metadata (ne/nb/type); their buffer
// may be null. Expert data is read from file_data entries.
bool build_qwen35moe_hybrid_storage_from_file(
    const TargetWeights & w,
    ggml_backend_t gpu_backend,
    const Qwen35MoeExpertPlacement & placement,
    const std::vector<LayerExpertFileData> & file_data,
    Qwen35MoeHybridStorage & out,
    std::string * err = nullptr);

}  // namespace dflash::common
