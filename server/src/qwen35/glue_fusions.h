// Glue fusions on the Qwen3.5/3.6/3.8 target graph (qwen35_target_graph.cpp):
// which nodes each one replaces on a given path. Pure decisions, so
// test_glue_fusions checks them without a GPU;
// the graph builder supplies the facts. Both are always on and
// byte-identical.
//
// Strided gates: on the dense single-sequence DeltaNet chain
// (single-request prefill and capture verify on the fused sequential
// kernels, T > 1), where the parent makes a contiguous copy (ggml_cont) of a
// strided row slice of a stacked projection only so that a consumer gets a
// contiguous tensor, that consumer reads the slice in place:
//   - the GDN chain kernel reads beta and alpha straight from the stacked
//     ssm_beta|alpha result (ggml_gated_delta_net_shared_gates): two copies
//     per DeltaNet layer;
//   - the gated output norm reads z straight from the stacked attn_gate|qkv
//     result, through the fused silu(z) * norm kernel (ggml-cuda
//     ggml_cuda_op_unary_mul, which takes a row stride): one copy per layer
//     whose z is stacked.
// Every other segment keeps both copies: tree verify, the concurrent mapped
// verify, compact multi-sequence decode, ragged prefill segments, chunked and
// SpecLA. The z rewrite would compute the same bytes there too, but no A/B
// covers those graphs, so the fusion stays inert on them.
// Bitwise: the copies are F32 -> F32 data movement, and the consumers run
// the same kernels on the same values; only their index arithmetic changes.
//
// Gate|up stack: the loader stacks ffn_gate|ffn_up as one
// zero-copy weight where the two share a type, and the FFN runs one GEMM over
// it instead of two, then splits the result with row views that the SwiGLU
// reads in place. Bitwise only where the backend computes every output row
// independently of the weight's row count (ggml-cuda
// ggml_backend_cuda_row_stack_inexact: MMQ without stream-k, every member a
// multiple of 128 rows); everywhere else the fusion is inert.

#pragma once

namespace luce::qwen35::glue {

// Which fusions are on: both. Tests turn them off to compare against the
// unfused graph (set_fusions_for_test).
struct Fusions {
    bool strided_gates = true;
    bool gate_up_stack = true;
};

namespace detail {
inline Fusions & fusions_storage() {
    static Fusions fusions;
    return fusions;
}
}  // namespace detail

inline bool strided_gates_on() { return detail::fusions_storage().strided_gates; }
inline bool gate_up_stack_on() { return detail::fusions_storage().gate_up_stack; }

// Tests only: choose the fusions before the model loads (the loader reads
// gate_up_stack_on) and before the first graph build.
inline void set_fusions_for_test(const Fusions & f) { detail::fusions_storage() = f; }

// ── Strided gates ───────────────────────────────────────────

// One DeltaNet segment of build_delta_net_block, as the parent would build it.
struct StridedGdnFacts {
    bool stacked_ba = false;       // beta/alpha are row slices of the ssm_beta|alpha stack
    bool ba_copy = false;          // the parent copies those slices (non-contiguous, T > 1)
    bool fused_chain = false;      // dense single-sequence chain on the fused sequential CUDA
                                   // kernels (not tree, mapped, compact, ragged, chunked, SpecLA)
    bool raw_gates = false;        // the GDN kernel derives the gates (ssm_gate_ba)
    bool mapped_verify = false;    // concurrent mapped verify: already reads the ba stack
    bool stacked_z = false;        // z is a row slice of the attn_gate|qkv stack
    bool z_copy = false;           // the parent copies that slice (non-contiguous, T > 1)
    bool cuda_backend = false;     // CUDA/HIP (the fused silu*mul kernel reads strides)
    bool fusion_enabled = false;   // ggml_backend_cuda_fusion_enabled(): no test turned them off
};

struct StridedGdnPlan {
    bool gates = false;                // beta/alpha: views + ggml_gated_delta_net_shared_gates
    bool z = false;                    // z: view into the fused silu*mul
};

// Called only with the fusion on (strided_gates_on()).
inline StridedGdnPlan strided_gdn_plan(const StridedGdnFacts & f) {
    StridedGdnPlan p;
    // The copies this segment's parent graph makes (the mapped verify
    // already reads beta/alpha through views, so it has no gate copies).
    const bool ba_copied = f.stacked_ba && f.ba_copy && !f.mapped_verify;
    const bool z_copied = f.stacked_z && f.z_copy;
    p.gates = ba_copied && f.fused_chain && f.raw_gates;
    // The concurrent mapped verify's GDN output is the same contiguous
    // [d_inner, T] rows as the chain's, so the fused silu*mul reads its z in
    // place too.
    // The z rewrite needs the fused CUDA/HIP silu*mul kernel: the unfused silu
    // needs a contiguous input.
    p.z = z_copied && (f.fused_chain || f.mapped_verify) && f.cuda_backend && f.fusion_enabled;
    return p;
}

// ── Gate|up stack ──────────────────────────────────────────

// One dense SwiGLU FFN of a graph build.
struct GateUpFacts {
    bool stacked = false;                    // the loader built ffn_gate|ffn_up (same type, adjacent)
    bool unit_scales = false;                // no NVFP4 per-tensor scale on gate or up (scales apply per GEMM)
    const char * backend_inexact = nullptr;  // ggml_backend_cuda_row_stack_inexact, or a reason
};

struct GateUpPlan {
    bool stack = false;                // one GEMM over the stack, split by views
};

// Called only with the fusion on (gate_up_stack_on()).
inline GateUpPlan gate_up_plan(const GateUpFacts & f) {
    GateUpPlan p;
    p.stack = f.stacked && f.unit_scales && f.backend_inexact == nullptr;
    return p;
}

}  // namespace luce::qwen35::glue
