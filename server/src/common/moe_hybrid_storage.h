// Common MoE hybrid expert storage — manages hot and cold expert buffers.

#pragma once

#include "moe_hybrid_types.h"
#include "moe_hybrid_placement.h"

#include "ggml-alloc.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeHybridRoutingStats;

// File region for one expert tensor (offset into mmap).
struct ExpertFileRegion {
    size_t offset = 0;
    size_t size   = 0;
};

// Per-layer file regions for all expert tensors (used by streaming prefill).
struct LayerExpertRegions {
    ExpertFileRegion gate_exps;
    ExpertFileRegion up_exps;
    ExpertFileRegion down_exps;
    ExpertFileRegion gate_up_exps;  // optional fused
    size_t expert_bytes_gate    = 0;
    size_t expert_bytes_up      = 0;
    size_t expert_bytes_down    = 0;
    size_t expert_bytes_gate_up = 0;
    bool   fused_gate_up        = false;
};

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
    int n_tokens = 1;                   // batched graph token count

    bool valid() const { return ctx && gf && alloc && output; }
    void free();
};

// Cached batched FFN graph for hot-only prefill (n_tokens = MMQ_SAFE_SUB_BATCH).
// Eliminates per-call graph rebuild + gallocr planning overhead.
struct CachedHotBatchedGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * inp = nullptr;        // [n_embd, n_tokens] F32 input
    ggml_tensor * sel = nullptr;        // [n_used, n_tokens] I32 hot-local IDs
    ggml_tensor * wts = nullptr;        // [n_used, n_tokens] F32 expert weights
    ggml_tensor * output = nullptr;     // [n_embd, n_tokens] F32 output
    int n_tokens = 0;

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
    ggml_backend_t cold_backend = nullptr; // Alias: either CPU backend or caller-owned GPU/HIP backend.
    MoeHybridColdBackend cold_backend_kind = MoeHybridColdBackend::Cpu;

    std::vector<int32_t> hot_expert_ids;
    std::vector<int32_t> cold_expert_ids;
    std::vector<int32_t> hot_local_by_global;
    // Optional decode ownership is independent of the physical maps used by
    // prefill. Empty maps fall back to hot_local_by_global and
    // cold_local_by_global respectively.
    std::vector<int32_t> decode_hot_local_by_global;
    std::vector<int32_t> decode_cold_local_by_global;
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

    // Bitmask: bit set = expert is in VRAM (hot). Sized from the model's
    // declared expert count so the common runtime does not impose an
    // architecture-specific expert limit.
    std::vector<uint64_t> expert_vram_mask;

    void reset_expert_vram_mask(int n_expert) {
        const size_t word_count = n_expert > 0 ? ((size_t)n_expert + 63) / 64 : 0;
        expert_vram_mask.assign(word_count, 0);
    }

    bool is_expert_hot(int global_expert) const {
        if (global_expert < 0) return false;
        const size_t word = (size_t)global_expert >> 6;
        return word < expert_vram_mask.size() &&
               ((expert_vram_mask[word] >> (global_expert & 63)) & 1ULL) != 0;
    }

    void set_expert_hot(int global_expert) {
        if (global_expert < 0) return;
        const size_t word = (size_t)global_expert >> 6;
        if (word < expert_vram_mask.size()) {
            expert_vram_mask[word] |= 1ULL << (global_expert & 63);
        }
    }

    void clear_expert_hot(int global_expert) {
        if (global_expert < 0) return;
        const size_t word = (size_t)global_expert >> 6;
        if (word < expert_vram_mask.size()) {
            expert_vram_mask[word] &= ~(1ULL << (global_expert & 63));
        }
    }

    // Fast check: are ALL routed experts in VRAM for this batch?
    // selected_ids has n_slots entries (n_tokens * n_expert_used).
    bool all_routed_are_hot(const int32_t * selected_ids, int n_slots) const {
        for (int i = 0; i < n_slots; ++i) {
            const int g = selected_ids[i];
            if (g < 0) continue;
            if (!is_expert_hot(g)) return false;
        }
        return true;
    }

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
    std::vector<CachedFfnGraph> hot_graph_by_width;
    std::vector<CachedFfnGraph> cold_graph_by_width;

    // Cached batched hot-only graph for prefill sub-batches (n_tokens=4).
    CachedHotBatchedGraph hot_batched_graph;

    // Per-n_tokens cached graphs for the MIXED (hot+cold) batched path. The
    // all-hot path already caches via hot_batched_graph, but the mixed path used
    // to rebuild+free its hot AND cold ggml graphs on every call — that churn
    // dominated the spec-decode verify cost (many cold-bearing layers x
    // sub-batches x steps). Cache per n_tokens (index 1..kMaxBatchedCache-1) so
    // steady-state verify/replay rebuilds zero graphs. Large prefill batches
    // (n_tokens >= kMaxBatchedCache) keep using the inline build.
    static constexpr int kMaxBatchedCache = 9;  // covers spec sub-batch n_tokens 1..8
    CachedHotBatchedGraph hot_batched_mixed[kMaxBatchedCache];
    CachedHotBatchedGraph cold_batched_mixed[kMaxBatchedCache];
    CachedHotBatchedGraph shared_batched_graph;
};

struct MoeHybridStorage {
    MoeHybridStorage() = default;
    MoeHybridStorage(const MoeHybridStorage &) = delete;
    MoeHybridStorage & operator=(const MoeHybridStorage &) = delete;
    MoeHybridStorage(MoeHybridStorage &&) = delete;
    MoeHybridStorage & operator=(MoeHybridStorage &&) = delete;
    ~MoeHybridStorage();

    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t cold_backend = nullptr; // Alias: either cpu_backend or caller-owned GPU/HIP backend.
    MoeHybridColdBackend cold_backend_kind = MoeHybridColdBackend::Cpu;
    bool materialized_hot_experts = true;
    bool materialized_cold_experts = true;
    MoeHybridPlacement placement;
    std::vector<MoeHybridLayerStorage> layers;

    // Long heterogeneous prefill uses one routing graph and one owner graph
    // per layer, but never executes two graphs of the same class concurrently.
    // Reusing these arenas avoids repeated HIP allocation/page-mapping churn
    // while keeping memory bounded by the largest graph rather than n_layer
    // graphs.
    ggml_gallocr_t prefill_route_alloc = nullptr;
    ggml_gallocr_t prefill_hot_alloc = nullptr;
    ggml_gallocr_t prefill_cold_alloc = nullptr;

    // Persistent mmap for streaming prefill (nullptr if not available).
    // When set, the streaming engine can DMA cold experts directly from here.
    const void * mmap_data = nullptr;
    size_t mmap_size = 0;
    int mmap_fd = -1;  // POSIX fd for madvise; -1 on Windows or if not available

    // Per-layer file region metadata for streaming (populated when mmap is active).
    std::vector<LayerExpertRegions> layer_regions;

    // Remove decode-table registrations while their owner tensors are alive.
    // Safe to call repeatedly, including during failed partial registration.
    void unregister_mix_tensors();
    bool matches(const MoeHybridConfig & cfg) const;
    bool empty() const;
    bool has_mmap() const { return mmap_data != nullptr && mmap_size > 0; }

    // Decode/verify graph arenas are shape caches, not model state. Release
    // them before a new bulk prefill needs substantially larger workspaces.
    void release_graph_caches();
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
                              std::string * err = nullptr,
                              ggml_backend_t cold_gpu_backend = nullptr);

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
    int cache_slots = 0,
    bool allocate_cold = true,
    ggml_backend_t cold_gpu_backend = nullptr);

// Spark: split a VRAM budget into a pinned-hot tier + an auto-sized expert
// cache ring. target_bytes==0 keeps the current budget (use the card);
// otherwise the budget is clamped to fit target_bytes minus core_kv_safety.
struct MoeSparkBudget { uint64_t hot_bytes; int cache_slots; };
MoeSparkBudget spark_budget_split(uint64_t expert_budget, uint64_t total_expert_bytes,
                                  int n_expert, uint64_t core_kv_safety,
                                  uint64_t target_bytes);

// Build hybrid storage from file AND retain mmap for streaming prefill.
// The caller must keep the mmap region alive for the lifetime of the storage.
// mmap_base: pointer to start of mmap'd file.
// mmap_total_size: total file size.
// This variant populates out.layer_regions for use by MoeHybridStreamEngine.
bool build_moe_hybrid_storage_from_file_with_mmap(
    const MoeHybridConfig & cfg,
    ggml_backend_t gpu_backend,
    const MoeHybridPlacement & placement,
    const std::vector<MoeLayerDesc> & layer_descs,
    const std::vector<LayerExpertFileData> & file_data,
    const void * mmap_base,
    size_t mmap_total_size,
    MoeHybridStorage & out,
    std::string * err = nullptr,
    int cache_slots = 0,
    ggml_backend_t cold_gpu_backend = nullptr);

}  // namespace dflash::common
