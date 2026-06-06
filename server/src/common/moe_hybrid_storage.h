// Common MoE hybrid expert storage — manages hot (GPU) and cold (CPU) expert buffers.

#pragma once

#include "moe_hybrid_types.h"
#include "moe_hybrid_placement.h"

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
    ggml_tensor * global_ids = nullptr;   // [n_hot,1] I32 global expert ids (gpu-remap)
    ggml_tensor * raw_weights = nullptr;  // [n_hot,1] F32 router weights (gpu-remap)
    ggml_tensor * hot_local_lut = nullptr;// [1,n_expert] I32 global->local hot id
    ggml_tensor * valid_lut = nullptr;    // [1,n_expert] F32 1=hot 0=cold
    ggml_tensor * residual_in = nullptr; // [n_embd,1] F32 residual (gpu-remap)
    int n_hot = 0;                      // number of hot experts this graph supports

    bool valid() const { return ctx && gf && alloc && output; }
    void free();
};

struct MoeHybridLayerStorage {
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

    // --- Bounded GPU expert cache (laguna) ---
    // Hot tensors are over-allocated by `cache_slots` spare entries appended
    // after the `hot_active` calibration-placed experts. A spare slot holds a
    // swapped-in cold expert; LRU eviction keeps the cache bounded.
    int hot_active = 0;            // # calibration-placed (pinned) hot experts
    int cache_slots = 0;          // # spare slots (cache capacity)
    std::vector<int32_t> spare_global;  // [cache_slots] global expert in each slot (-1 empty)
    std::vector<uint64_t> spare_lru;    // [cache_slots] last-use tick
    uint64_t lru_clock = 0;

    bool fused_gate_up = false;
    size_t gate_expert_bytes = 0;
    size_t up_expert_bytes = 0;
    size_t down_expert_bytes = 0;
    size_t gate_up_expert_bytes = 0;

    std::vector<uint8_t> gate_cold_bytes;
    std::vector<uint8_t> up_cold_bytes;
    std::vector<uint8_t> down_cold_bytes;
    std::vector<uint8_t> gate_up_cold_bytes;

    // Cached FFN graphs for common-case expert counts.
    CachedFfnGraph hot_graph;
    CachedFfnGraph cold_graph;
};

struct MoeHybridStorage {
    MoeHybridStorage() = default;
    MoeHybridStorage(const MoeHybridStorage &) = delete;
    MoeHybridStorage & operator=(const MoeHybridStorage &) = delete;
    MoeHybridStorage(MoeHybridStorage &&) = delete;
    MoeHybridStorage & operator=(MoeHybridStorage &&) = delete;
    ~MoeHybridStorage();

    ggml_backend_t cpu_backend = nullptr;
    MoeHybridPlacement placement;
    std::vector<MoeHybridLayerStorage> layers;

    bool matches(const MoeHybridConfig & cfg) const;
    bool empty() const;
};

// Expert tensor file data for split loading (one entry per expert tensor).
struct ExpertTensorFileData {
    const uint8_t * data = nullptr;
    size_t size = 0;
};

// Per-layer expert tensor file data for split loading.
struct LayerExpertFileData {
    ExpertTensorFileData gate_exps;
    ExpertTensorFileData up_exps;
    ExpertTensorFileData down_exps;
    ExpertTensorFileData gate_up_exps;  // optional fused
};

// Build hybrid storage from GPU-resident expert tensors.
// layer_descs: one MoeLayerDesc per MoE layer (caller constructs from model-specific types).
bool build_moe_hybrid_storage(const MoeHybridConfig & cfg,
                              ggml_backend_t gpu_backend,
                              const MoeHybridPlacement & placement,
                              const std::vector<MoeLayerDesc> & layer_descs,
                              MoeHybridStorage & out,
                              std::string * err = nullptr);

// Swap a cold expert into a spare GPU cache slot (LRU evict). Returns the new
// hot-local index, or -1 on failure. No-op (returns existing) if already hot.
int moe_hybrid_cache_swap_in(MoeHybridLayerStorage & st, int global_expert,
                             ggml_backend_t gpu_backend);

// Build hybrid storage by loading expert data directly from file (mmap).
bool build_moe_hybrid_storage_from_file(
    const MoeHybridConfig & cfg,
    ggml_backend_t gpu_backend,
    const MoeHybridPlacement & placement,
    const std::vector<MoeLayerDesc> & layer_descs,
    const std::vector<LayerExpertFileData> & file_data,
    MoeHybridStorage & out,
    std::string * err = nullptr,
    int cache_slots = 0);

// Spark: split a VRAM budget into a pinned-hot tier + an auto-sized expert
// cache ring. target_bytes==0 keeps the current budget (use the card);
// otherwise the budget is clamped to fit target_bytes minus core_kv_safety.
struct MoeSparkBudget { uint64_t hot_bytes; int cache_slots; };
MoeSparkBudget spark_budget_split(uint64_t expert_budget, uint64_t total_expert_bytes,
                                  int n_expert, uint64_t core_kv_safety,
                                  uint64_t target_bytes);

}  // namespace dflash::common
