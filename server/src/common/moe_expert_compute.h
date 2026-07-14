// MoeExpertCompute: direct compute interface for selected MoE expert FFN work.
// Bypasses ggml graph dispatch overhead and lets MoE backends route selected
// expert work to CPU, a backend-local IPC daemon, or future residency tiers.
#pragma once

#include "moe_hybrid_storage.h"

#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

// Per-layer selected expert weight metadata; raw pointers into shared memory.
struct MoeExpertLayer {
    int layer_idx = -1;
    const void * gate_up_data = nullptr;  // fused [n_cold, n_ff*2, n_embd] quantized
    const void * gate_data = nullptr;     // separate gate [n_cold, n_ff, n_embd]
    const void * up_data = nullptr;       // separate up   [n_cold, n_ff, n_embd]
    const void * down_data = nullptr;     // [n_cold, n_embd, n_ff] quantized

    size_t gate_up_stride = 0;   // bytes between experts in gate_up tensor
    size_t gate_stride = 0;      // bytes between experts in gate tensor
    size_t up_stride = 0;        // bytes between experts in up tensor
    size_t down_stride = 0;      // bytes between experts in down tensor

    ggml_type gate_up_type = GGML_TYPE_Q4_K;  // type for fused gate_up
    ggml_type gate_type = GGML_TYPE_Q4_K;     // type for separate gate
    ggml_type up_type = GGML_TYPE_Q4_K;       // type for separate up
    ggml_type down_type = GGML_TYPE_Q4_K;     // type for down projection
    bool fused_gate_up = false;               // true if gate+up are fused

    // Scale factors (applied after matmul). 1.0 = no scaling.
    float gate_up_scale = 1.0f;
    float gate_scale = 1.0f;
    float up_scale = 1.0f;
    float down_scale = 1.0f;

    // Maps placement-local expert indices back to model-global expert ids.
    // CPU implementations consume local ids directly; remote mixed-backend
    // implementations use this to address the complementary remote placement.
    std::vector<int32_t> cold_global_by_local;
};

// Abstract compute interface. Current implementations include CPU and backend
// IPC; model backends provide their own expert metadata/adapters.
struct MoeExpertCompute {
    virtual ~MoeExpertCompute() = default;
    virtual bool healthy() const { return true; }

    // Compute selected expert FFN contributions and accumulate into output.
    // input:   [n_embd] F32 — post-norm hidden state
    // ids:     [n_cold] I32 — local cold expert indices
    // weights: [n_cold] F32 — routing weights for each cold expert
    // output:  [n_embd] F32 — accumulated weighted expert outputs (zeroed by callee)
    virtual bool compute(
        const MoeExpertLayer & layer,
        const float * input,
        const int32_t * ids,
        const float * weights,
        int n_cold,
        int n_embd,
        int n_ff,
        float * output) = 0;

    // Batched variant used by prefill. The default keeps implementations
    // simple and lets remote backends override it to amortize IPC overhead.
    virtual bool compute_batch(
        const MoeExpertLayer & layer,
        const float * input,
        const int32_t * ids,
        const float * weights,
        int n_tokens,
        int n_selected,
        int n_embd,
        int n_ff,
        float * output) {
        if (n_tokens < 0 || n_selected < 0 || n_embd <= 0 || !output) return false;
        if (n_tokens == 0 || n_selected == 0) {
            std::fill(output, output + (size_t)n_tokens * (size_t)n_embd, 0.0f);
            return true;
        }
        for (int t = 0; t < n_tokens; ++t) {
            if (!compute(layer,
                         input + (size_t)t * (size_t)n_embd,
                         ids + (size_t)t * (size_t)n_selected,
                         weights + (size_t)t * (size_t)n_selected,
                         n_selected, n_embd, n_ff,
                         output + (size_t)t * (size_t)n_embd)) {
                return false;
            }
        }
        return true;
    }
};

// Create CPU-based fused MoE expert compute.
std::unique_ptr<MoeExpertCompute> make_cpu_moe_expert_compute(int n_ff_max);

uint64_t moe_expert_placement_fingerprint(const MoeHybridStorage & hybrid,
                                         int n_layer,
                                         int n_expert,
                                         int n_expert_used);

std::vector<MoeExpertLayer> make_moe_expert_layers(
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs);

struct MoeExpertComputeIpcStartResult {
    std::unique_ptr<MoeExpertCompute> compute;
    bool started_remote = false;
};

struct MoeExpertComputeRuntime {
    std::unique_ptr<MoeExpertCompute> compute;
    std::vector<MoeExpertLayer> layers;
    std::string target_path;
    std::string runtime_key;
    uint64_t placement_fingerprint = 0;

    void reset();
    MoeExpertCompute * compute_ptr() const { return compute.get(); }
    const MoeExpertLayer * layer_ptr(size_t il) const {
        return il < layers.size() ? &layers[il] : nullptr;
    }
};

struct MoeExpertComputeRuntimeConfig {
    std::string target_path;
    int n_layer = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_embd = 0;
    int n_ff_exp = 0;
    bool enabled = true;
    const char * log_prefix = "[moe-expert-compute]";
};

bool ensure_moe_expert_compute_runtime(
    MoeExpertComputeRuntime & runtime,
    const MoeExpertComputeRuntimeConfig & cfg,
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs,
    std::string * err = nullptr);

MoeExpertComputeIpcStartResult make_moe_expert_compute_ipc(
    const std::string & bin,
    const std::string & target_path,
    int target_gpu,
    const MoeHybridPlacement & main_placement,
    int n_embd,
    int n_ff_exp,
    int n_expert_used,
    const std::string & work_dir,
    bool required);

int run_moe_expert_compute_ipc_daemon(const char * target_path,
                                      const char * placement_path,
                                      int target_gpu,
                                      int stream_fd,
                                      int payload_fd = -1,
                                      int shared_payload_fd = -1,
                                      size_t shared_payload_bytes = 0);

}  // namespace dflash::common
