// Common MoE hybrid mode types and descriptors.
//
// Model-agnostic abstractions used by MoE backends to implement heterogeneous
// expert execution. The hot and cold owners may be two GPUs or a GPU and CPU;
// model integrations only describe architecture and tensor layout.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

namespace dflash::common {

// ─── GPU SM version query ───────────────────────────────────────────────
// Returns the compute capability as major*10+minor (e.g. 86 for sm_86).
// Returns 0 if CUDA/HIP runtime is unavailable.
int query_gpu_compute_sm();

enum class MoeHybridColdBackend {
    Cpu,
    Gpu,
};

// ─── MoE architecture config (model-agnostic) ──────────────────────────

struct MoeHybridConfig {
    int n_embd        = 0;   // hidden dimension
    int n_expert      = 0;   // total experts per layer
    int n_expert_used = 0;   // top-k selected per token
    int n_ff_exp      = 0;   // routed expert intermediate dimension
    int n_ff_shexp    = 0;   // shared expert intermediate dimension (0 = no shared)
    int n_layer       = 0;   // number of MoE layers
    int first_moe_layer = 0; // index of first MoE layer (e.g., 0 for qwen35moe, 1 for laguna)
    float swiglu_clamp = 0.0f; // 0 = regular SwiGLU; >0 clamps gate upper/up symmetric (DS4)
    MoeHybridColdBackend cold_expert_backend = MoeHybridColdBackend::Cpu;
    bool materialize_hot_experts = true;
    bool materialize_cold_experts = true;

    // Optional tensor-parallel shard of the shared FFN intermediate width.
    // 0 disables it; values in (0,1) request an explicit tensor partition;
    // 1 assigns the complete stage to peer; a negative value asks the generic
    // planner to balance from owner rates.
    // The storage builder also accepts DFLASH_MOE_TP_SHARED_FFN_PEER_FRACTION
    // (and the legacy DS4 alias), allowing every model adapter that uses the
    // common hybrid path to opt in without architecture-specific graph code.
    float shared_ffn_peer_fraction = 0.0f;
    float heterogeneous_main_rate = 1.0f;
    float heterogeneous_peer_rate = 1.0f;
    // Work already assigned to each owner when the splittable stage begins.
    // Values use the same arbitrary work unit as the stage width; adapters can
    // therefore account for routed experts or other concurrent dense stages.
    float heterogeneous_main_fixed_work = 0.0f;
    float heterogeneous_peer_fixed_work = 0.0f;

    // When true, MMQ mul_mat_id works correctly with reduced hot stacks
    // (n_hot < n_expert). Safe on sm_80+ (Ampere/Ada/Hopper/Blackwell).
    // On sm_75 (Turing) and gfx1151, the kernel has illegal memory accesses
    // with reduced stacks, requiring the <=4-token sub-batch workaround.
    bool mmq_safe_full_batch = false;
};

// ─── Per-layer expert tensor descriptor ─────────────────────────────────
//
// Provides a uniform view over model-specific layer structures. All pointers
// refer to the FULL expert tensor stacks on GPU (used for placement validation
// and metadata queries). In hybrid mode, the actual hot/cold split tensors
// live in MoeHybridLayerStorage.

struct MoeLayerDesc {
    // Routed expert weight tensors (stacked: [dim_in, dim_out, n_expert])
    ggml_tensor * ffn_gate_exps    = nullptr;
    ggml_tensor * ffn_up_exps      = nullptr;
    ggml_tensor * ffn_down_exps    = nullptr;
    ggml_tensor * ffn_gate_up_exps = nullptr;  // optional fused gate+up

    // Shared expert tensors (nullptr if no shared expert)
    ggml_tensor * ffn_gate_shexp     = nullptr;
    ggml_tensor * ffn_up_shexp       = nullptr;
    ggml_tensor * ffn_down_shexp     = nullptr;
    ggml_tensor * ffn_gate_inp_shexp = nullptr;  // optional shared-expert gating

    // Per-tensor quantization scale factors (1.0f = no scaling)
    float ffn_gate_exps_s      = 1.0f;
    float ffn_up_exps_s        = 1.0f;
    float ffn_down_exps_s      = 1.0f;
    float ffn_gate_up_exps_s   = 1.0f;
    float ffn_gate_shexp_s     = 1.0f;
    float ffn_up_shexp_s       = 1.0f;
    float ffn_down_shexp_s     = 1.0f;
    float ffn_gate_inp_shexp_s = 1.0f;

    bool has_fused_gate_up() const { return ffn_gate_up_exps != nullptr; }
    bool has_shared_expert() const { return ffn_up_shexp != nullptr; }
};

}  // namespace dflash::common
