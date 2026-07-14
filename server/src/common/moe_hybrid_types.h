// Common MoE hybrid mode types and descriptors.
//
// Model-agnostic abstractions used by both qwen35moe and laguna backends
// to implement the hybrid expert offload strategy (hot experts on GPU,
// cold experts on CPU, concurrent evaluation).

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
