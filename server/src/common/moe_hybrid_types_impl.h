// Inline implementations for MoeHybridConfig/MoeLayerDesc conversion helpers.
//
// Include this header AFTER both moe_hybrid_types.h and the relevant
// model-specific weight struct header (internal.h or laguna_internal.h).
// The preprocessor guards detect which weight structs are available and
// only define the corresponding conversion helpers.

#pragma once

namespace dflash::common {

// ─── qwen35 conversions ─────────────────────────────────────────────────

#if defined(DFLASH_INTERNAL_H_INCLUDED)

inline MoeHybridConfig make_moe_hybrid_config(const TargetWeights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd        = w.n_embd;
    cfg.n_expert      = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp      = w.n_ff_exp;
    cfg.n_ff_shexp    = w.n_ff_shexp;
    cfg.n_layer       = w.n_layer;
    cfg.first_moe_layer = 0;  // all layers are MoE in qwen35moe
    // sm_80+ (Ampere and later): MMQ mul_mat_id is safe with reduced hot stacks
    static const int sm = query_gpu_compute_sm();
    cfg.mmq_safe_full_batch = (sm >= 80);
    return cfg;
}

inline MoeLayerDesc make_moe_layer_desc(const TargetLayer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps      = L.ffn_gate_exps;
    desc.ffn_up_exps        = L.ffn_up_exps;
    desc.ffn_down_exps      = L.ffn_down_exps;
    desc.ffn_gate_up_exps   = L.ffn_gate_up_exps;
    desc.ffn_gate_shexp     = L.ffn_gate_shexp;
    desc.ffn_up_shexp       = L.ffn_up_shexp;
    desc.ffn_down_shexp     = L.ffn_down_shexp;
    desc.ffn_gate_inp_shexp = L.ffn_gate_inp_shexp;
    desc.ffn_gate_exps_s      = L.ffn_gate_exps_s;
    desc.ffn_up_exps_s        = L.ffn_up_exps_s;
    desc.ffn_down_exps_s      = L.ffn_down_exps_s;
    desc.ffn_gate_up_exps_s   = L.ffn_gate_up_exps_s;
    desc.ffn_gate_shexp_s     = L.ffn_gate_shexp_s;
    desc.ffn_up_shexp_s       = L.ffn_up_shexp_s;
    desc.ffn_down_shexp_s     = L.ffn_down_shexp_s;
    desc.ffn_gate_inp_shexp_s = L.ffn_gate_inp_shexp_s;
    return desc;
}

#endif  // DFLASH_INTERNAL_H_INCLUDED

// ─── Laguna conversions ─────────────────────────────────────────────────

#if defined(DFLASH_LAGUNA_INTERNAL_H_INCLUDED)

inline MoeHybridConfig make_moe_hybrid_config(const LagunaTargetWeights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd        = w.n_embd;
    cfg.n_expert      = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp      = w.n_ff_exp;
    cfg.n_ff_shexp    = w.n_ff_shexp;
    cfg.n_layer       = w.n_layer;
    cfg.first_moe_layer = w.n_layer_dense_lead;  // layer 0 is dense in laguna
    // sm_80+ (Ampere and later): MMQ mul_mat_id is safe with reduced hot stacks
    static const int sm = query_gpu_compute_sm();
    cfg.mmq_safe_full_batch = (sm >= 80);
    return cfg;
}

inline MoeLayerDesc make_moe_layer_desc(const LagunaTargetLayer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps      = L.ffn_gate_exps;
    desc.ffn_up_exps        = L.ffn_up_exps;
    desc.ffn_down_exps      = L.ffn_down_exps;
    desc.ffn_gate_up_exps   = nullptr;  // laguna has no fused gate_up
    desc.ffn_gate_shexp     = L.ffn_gate_shexp;
    desc.ffn_up_shexp       = L.ffn_up_shexp;
    desc.ffn_down_shexp     = L.ffn_down_shexp;
    desc.ffn_gate_inp_shexp = nullptr;  // laguna has no shared-expert gate
    // Laguna does not use per-tensor quantization scales
    desc.ffn_gate_exps_s    = 1.0f;
    desc.ffn_up_exps_s      = 1.0f;
    desc.ffn_down_exps_s    = 1.0f;
    desc.ffn_gate_up_exps_s = 1.0f;
    desc.ffn_gate_shexp_s   = 1.0f;
    desc.ffn_up_shexp_s     = 1.0f;
    desc.ffn_down_shexp_s   = 1.0f;
    desc.ffn_gate_inp_shexp_s = 1.0f;
    return desc;
}

#endif  // DFLASH_LAGUNA_INTERNAL_H_INCLUDED

}  // namespace dflash::common
