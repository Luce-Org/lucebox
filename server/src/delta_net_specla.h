// SpecLA topology-masked delta-net verification (arXiv:2607.16673 §4.2, §5.1).
//
// Single-window UT-transform verify for speculative windows (n <= 64 tokens),
// derived from build_delta_net_chunked but:
//   - masks are host-filled ancestor-topology inputs instead of ggml_tri, so
//     one builder serves both chain drafts (lower-triangular masks) and
//     DFS-ordered tree drafts (ancestor masks);
//   - the recurrent state is NOT advanced: verification reads the committed
//     state S0 only, and besides per-node outputs it exposes the compact
//     factors (corrected values v_new and path-cumulative gates g_ps) that
//     DeltaConstruct needs to advance the state along the accepted path:
//         S_A = exp(g_ps_A) * S0 + sum_{t in path(A)} exp(g_ps_A - g_ps_t) k_t (x) v_new_t
//
// See server/docs/SPECLA.md for the derivation and the commit-side consumer.
#pragma once

#include "ggml.h"
#include <cstdint>
#include <vector>

namespace dflash::common {

struct DeltaNetSpecLAResult {
    ggml_tensor * output;  // [S_v, H_v, n_tokens, 1] — same layout as the fused op's output slice
    ggml_tensor * v_new;   // corrected values ṽ: [n_tokens, S_v, 1, H_v]
    ggml_tensor * g_ps;    // path-cumulative gate: [n_tokens, 1, 1, H_v]
};

// Heavy-light schedule consumed by the fused SpecLA recurrent kernels. Chains
// are grouped by dependency wave; every chain in a wave can execute in
// parallel because its parent boundary was produced by an earlier wave.
// The packed vector is a compact int32 ABI shared with the CUDA/HIP kernels.
struct SpecLAHLDSchedule {
    std::vector<int32_t> packed;
    int n_nodes      = 0;
    int n_chains     = 0;
    int n_waves      = 0;
    int n_boundaries = 0;
    int max_parallel_chains = 0;
};

SpecLAHLDSchedule make_specla_hld_schedule(const int32_t * parents,
                                           int n,
                                           int pending_count);

// q,k,v,g,b,s use the exact shapes build_delta_net_block passes to
// build_delta_net_chunked: q/k [S_k, H_v, n, 1] (post l2-norm, post repeat),
// v [S_v, H_v, n, 1], g/b [1, H_v, n, 1], s [S_v, S_v, H_v, 1].
//
// m_strict / m_incl / m_eye are F32 [n, n] input tensors filled host-side
// with the draft topology over DFS-ordered nodes: element (ne0=u, ne1=t) is
// 1.0f when u is a strict ancestor of t (m_strict) or an ancestor-or-self of
// t (m_incl), else 0.0f; m_eye is the identity. For a chain draft these are
// simply the strict lower triangle and the lower triangle with diagonal.
// m_eye is a host input (not ggml_fill on a view, whose output would alias
// another node's data) so no graph node aliases live intermediate storage.
DeltaNetSpecLAResult build_delta_net_specla(
        ggml_context * ctx0,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * g,
        ggml_tensor  * b,
        ggml_tensor  * s,
        ggml_tensor  * m_strict,
        ggml_tensor  * m_incl,
        ggml_tensor  * m_eye);

// Host-side helpers: fill the three topology masks from a parent-pointer
// array over DFS/topologically ordered nodes (parents[t] < t, root = -1).
// `dst` buffers hold n*n floats each. For chains pass parents[t] = t-1.
void fill_specla_masks(const int32_t * parents, int n,
                       float * m_strict, float * m_incl, float * m_eye);

}  // namespace dflash::common
