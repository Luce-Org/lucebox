// MoeExpertCompute: direct compute interface for selected MoE expert FFN work.
//
// This is intentionally neutral to hot/cold placement. CPU fallback, remote
// IPC daemons, and future backend-local compute paths should share this shape
// so routing policy can evolve without changing model-specific FFN call sites.
#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "expert_split_compute_runtime.h"
#include "moe_hybrid_placement.h"
#include "moe_hybrid_storage.h"

namespace dflash::common {

// Per-layer selected expert weight metadata: raw pointers into the placement
// storage for whichever experts this compute implementation owns.
struct MoeExpertLayer {
    int layer_idx = -1;
    std::vector<int32_t> cold_global_by_local;

    const void * gate_up_data = nullptr;  // fused [n_expert, n_ff*2, n_embd]
    const void * gate_data = nullptr;     // separate gate [n_expert, n_ff, n_embd]
    const void * up_data = nullptr;       // separate up   [n_expert, n_ff, n_embd]
    const void * down_data = nullptr;     // [n_expert, n_embd, n_ff]

    size_t gate_up_stride = 0;
    size_t gate_stride = 0;
    size_t up_stride = 0;
    size_t down_stride = 0;

    ggml_type gate_up_type = GGML_TYPE_Q4_K;
    ggml_type gate_type = GGML_TYPE_Q4_K;
    ggml_type up_type = GGML_TYPE_Q4_K;
    ggml_type down_type = GGML_TYPE_Q4_K;
    bool fused_gate_up = false;

    float gate_up_scale = 1.0f;
    float gate_scale = 1.0f;
    float up_scale = 1.0f;
    float down_scale = 1.0f;
};

struct MoeExpertCompute {
    virtual ~MoeExpertCompute() = default;
    virtual bool healthy() const { return true; }
    virtual bool prefers_padded_batch() const { return false; }

    // Compute selected expert FFN contributions and accumulate into output.
    // input:   [n_embd] F32, post-norm hidden state
    // ids:     [n_selected] I32, local expert indices for this placement
    // weights: [n_selected] F32, routing weights
    // output:  [n_embd] F32, accumulated weighted expert output
    virtual bool compute(
        const MoeExpertLayer & layer,
        const float * input,
        const int32_t * ids,
        const float * weights,
        int n_selected,
        int n_embd,
        int n_ff,
        float * output) = 0;

    // Optional remote-graph preparation hook. Implementations that keep stable
    // backend-side graph state can prebuild common batch shapes before the
    // first request payload arrives. Local/CPU implementations can no-op.
    virtual bool prepare_batch(
        const MoeExpertLayer & layer,
        int n_tokens,
        int n_selected,
        int n_embd,
        int n_ff) {
        (void)layer;
        (void)n_ff;
        return n_tokens >= 0 && n_selected >= 0 && n_embd > 0;
    }

    // Optional decode-path preparation hook. This is the single-token companion
    // to prepare_batch() for remote implementations that keep stable decode
    // graphs alive on the backend.
    virtual bool prepare_single(
        const MoeExpertLayer & layer,
        int n_selected,
        int n_embd,
        int n_ff) {
        (void)layer;
        (void)n_ff;
        return n_selected >= 0 && n_embd > 0;
    }

    // Coarse-grained prefill hook. Remote implementations override this to
    // amortize command/payload overhead; CPU fallback keeps the old per-token
    // behavior through the single-token compute path.
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
        if (!input || !ids || !weights) return false;
        for (int t = 0; t < n_tokens; ++t) {
            if (!compute(layer,
                         input + (size_t)t * (size_t)n_embd,
                         ids + (size_t)t * (size_t)n_selected,
                         weights + (size_t)t * (size_t)n_selected,
                         n_selected,
                         n_embd,
                         n_ff,
                         output + (size_t)t * (size_t)n_embd)) {
                return false;
            }
        }
        return true;
    }
};

std::unique_ptr<MoeExpertCompute> make_cpu_moe_expert_compute(int n_ff_max);
int moe_expert_compute_batch_limit_from_env();
int moe_expert_compute_prepare_batch_limit_from_env();
int moe_expert_compute_prepare_selected_limit_from_env();
int moe_expert_compute_daemon_batch_limit_from_env();

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
    bool remote_started = false;

    void reset();
    MoeExpertCompute * compute_ptr() const { return compute.get(); }
    const MoeExpertLayer * layer_ptr(size_t il) const {
        return il < layers.size() ? &layers[il] : nullptr;
    }
};

struct MoeMultiTargetExpertRuntimeTarget {
    int target_index = -1;
    ExpertSplitTarget target;
    MoeHybridPlacement placement;
    MoeExpertComputeRuntime runtime;
    bool compute_active = false;
};

struct MoeMultiTargetLayerRoute {
    int target_slot = -1;
    int target_local = -1;
    int union_local = -1;
};

struct MoeMultiTargetLayerRuntime {
    std::vector<MoeMultiTargetLayerRoute> route_by_union_local;
};

struct MoeMultiTargetExpertRuntime {
    std::unique_ptr<MoeExpertCompute> compute;
    std::vector<MoeExpertLayer> layers;
    std::vector<MoeMultiTargetExpertRuntimeTarget> targets;
    std::vector<MoeMultiTargetLayerRuntime> layer_routes;
    std::string runtime_key;
    uint64_t placement_fingerprint = 0;
    bool enabled = false;

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

std::unique_ptr<MoeExpertCompute> make_multi_target_moe_expert_compute(
    MoeMultiTargetExpertRuntime * runtime);

uint64_t moe_expert_placement_fingerprint(const MoeHybridStorage & hybrid,
                                          int n_layer,
                                          int n_expert,
                                          int n_expert_used);

std::vector<MoeExpertLayer> make_moe_expert_layers(
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs);

bool ensure_moe_expert_compute_runtime(
    MoeExpertComputeRuntime & runtime,
    const MoeExpertComputeRuntimeConfig & cfg,
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs,
    std::string * err = nullptr);

bool ensure_multi_target_moe_expert_compute_runtime(
    MoeMultiTargetExpertRuntime & runtime,
    const MoeExpertComputeRuntimeConfig & cfg,
    const ExpertSplitComputeRuntime & split_runtime,
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

MoeExpertComputeIpcStartResult make_moe_expert_compute_ipc_for_placement(
    const std::string & bin,
    const std::string & target_path,
    int target_gpu,
    const MoeHybridPlacement & remote_placement,
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
