// SpecLA topology-masked delta-net verification — see delta_net_specla.h.
//
// Derived from build_delta_net_chunked (itself a port of llama.cpp's
// build_delta_net_chunking), specialized to a single speculative window:
//   - one chunk, no padding, no cross-chunk state loop;
//   - ggml_tri causal masks replaced by host-filled topology masks so tree
//     drafts verify in the same factorized form (paper §4.2:
//     A_tree = M_strict ⊙ (K K_bᵀ) ⊙ exp(g⁺_t − g⁺_u), T = (I + A)⁻¹);
//   - the recurrent state is read-only; instead of a new_state the builder
//     returns the per-token factors (ṽ, g⁺) for accepted-state
//     reconstruction (paper §5.1).
//
// Orientation conventions below follow ggml_mul_mat(a, b): with
// a = [K, M, b2, b3] and b = [K, N, b2', b3'], the result is
// C[M, N] = Σ_K a[K, M]·b[K, N], batched over dims 2..3 with broadcast.

#include "delta_net_specla.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dflash::common {

SpecLAHLDSchedule make_specla_hld_schedule(const int32_t * parents,
                                           int n,
                                           int pending_count) {
    SpecLAHLDSchedule out;
    if (!parents || n <= 0 || parents[0] >= 0 || pending_count < 0) {
        return out;
    }

    std::vector<std::vector<int>> children((size_t)n);
    for (int node = 1; node < n; ++node) {
        const int parent = parents[node];
        if (parent < 0 || parent >= node) return {};
        children[(size_t)parent].push_back(node);
    }

    // Pick the largest subtree as the heavy continuation. Ties retain the
    // topological child order, making schedules deterministic for tests and
    // CUDA graph reuse.
    std::vector<int> subtree((size_t)n, 1);
    std::vector<int> heavy((size_t)n, -1);
    for (int node = n - 1; node >= 0; --node) {
        int best_size = 0;
        for (int child : children[(size_t)node]) {
            subtree[(size_t)node] += subtree[(size_t)child];
            if (subtree[(size_t)child] > best_size) {
                best_size = subtree[(size_t)child];
                heavy[(size_t)node] = child;
            }
        }
    }

    struct Chain {
        std::vector<int> nodes;
        int parent_node = -1;
        int wave = 0;
    };
    std::vector<Chain> chains;
    std::vector<int> node_chain((size_t)n, -1);
    for (int head = 0; head < n; ++head) {
        const int parent = parents[head];
        if (head != 0 && heavy[(size_t)parent] == head) continue;
        Chain chain;
        chain.parent_node = parent;
        for (int node = head; node >= 0; node = heavy[(size_t)node]) {
            node_chain[(size_t)node] = (int)chains.size();
            chain.nodes.push_back(node);
        }
        chains.push_back(std::move(chain));
    }
    for (size_t c = 0; c < chains.size(); ++c) {
        const int parent = chains[c].parent_node;
        chains[c].wave = parent < 0 ? 0 : chains[(size_t)node_chain[(size_t)parent]].wave + 1;
    }
    std::stable_sort(chains.begin(), chains.end(),
        [](const Chain & a, const Chain & b) { return a.wave < b.wave; });

    std::vector<int32_t> order;
    std::vector<int32_t> chain_offsets;
    std::vector<int32_t> chain_parent_boundary;
    std::vector<int32_t> chain_wave;
    std::vector<int32_t> node_boundary((size_t)n, -1);
    order.reserve((size_t)n);
    chain_offsets.reserve(chains.size() + 1);
    chain_parent_boundary.reserve(chains.size());
    chain_wave.reserve(chains.size());
    chain_offsets.push_back(0);

    int n_boundaries = 0;
    int n_waves = 0;
    int max_parallel_chains = 0;
    int previous_wave = -1;
    int chains_in_wave = 0;
    for (const Chain & chain : chains) {
        int boundary = -1;
        if (chain.parent_node >= 0) {
            int32_t & slot = node_boundary[(size_t)chain.parent_node];
            if (slot < 0) slot = n_boundaries++;
            boundary = slot;
        }
        chain_parent_boundary.push_back(boundary);
        chain_wave.push_back(chain.wave);
        n_waves = std::max(n_waves, chain.wave + 1);
        if (chain.wave != previous_wave) {
            previous_wave = chain.wave;
            chains_in_wave = 0;
        }
        max_parallel_chains = std::max(max_parallel_chains, ++chains_in_wave);
        for (int node : chain.nodes) order.push_back(node);
        chain_offsets.push_back((int32_t)order.size());
    }

    // Packed metadata ABI shared with gated_delta_net.cu and ssm-conv.cu.
    // Header: magic, N, C, W, B, pending_count, five section offsets, and
    // the maximum number of chains in one dependency wave.
    constexpr int kHeader = 12;
    const int order_off   = kHeader;
    const int offsets_off = order_off + n;
    const int parent_off  = offsets_off + (int)chains.size() + 1;
    const int boundary_off = parent_off + (int)chains.size();
    const int wave_off     = boundary_off + n;
    out.packed.assign((size_t)wave_off + chains.size(), 0);
    out.packed[0]  = 0x534c4148; // "SLAH"
    out.packed[1]  = n;
    out.packed[2]  = (int32_t)chains.size();
    out.packed[3]  = n_waves;
    out.packed[4]  = n_boundaries;
    out.packed[5]  = pending_count;
    out.packed[6]  = order_off;
    out.packed[7]  = offsets_off;
    out.packed[8]  = parent_off;
    out.packed[9]  = boundary_off;
    out.packed[10] = wave_off;
    out.packed[11] = max_parallel_chains;
    std::copy(order.begin(), order.end(), out.packed.begin() + order_off);
    std::copy(chain_offsets.begin(), chain_offsets.end(), out.packed.begin() + offsets_off);
    std::copy(chain_parent_boundary.begin(), chain_parent_boundary.end(), out.packed.begin() + parent_off);
    std::copy(node_boundary.begin(), node_boundary.end(), out.packed.begin() + boundary_off);
    std::copy(chain_wave.begin(), chain_wave.end(), out.packed.begin() + wave_off);

    out.n_nodes = n;
    out.n_chains = (int)chains.size();
    out.n_waves = n_waves;
    out.n_boundaries = n_boundaries;
    out.max_parallel_chains = max_parallel_chains;
    return out;
}

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
        ggml_tensor  * m_eye) {
    const int64_t S_k = q->ne[0];
    const int64_t H_v = q->ne[1];
    const int64_t n   = q->ne[2];

    const int64_t S_v = v->ne[0];

    // Same layer family as the chunked GDA path: scalar gate per value head,
    // q/k already repeated to H_v heads by the caller.
    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_v && k->ne[2] == n);
    GGML_ASSERT(v->ne[1] == H_v && v->ne[2] == n);
    GGML_ASSERT(g->ne[0] == 1 && g->ne[1] == H_v && g->ne[2] == n);
    GGML_ASSERT(b->ne[0] == 1 && b->ne[1] == H_v && b->ne[2] == n);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v);
    GGML_ASSERT(q->ne[3] == 1 && s->ne[3] == 1);  // spec-decode verify is single-seq
    GGML_ASSERT(m_strict->ne[0] == n && m_strict->ne[1] == n);
    GGML_ASSERT(m_incl->ne[0] == n && m_incl->ne[1] == n);
    GGML_ASSERT(m_eye->ne[0] == n && m_eye->ne[1] == n);

    const float scale = 1.0f / sqrtf((float)S_k);
    q = ggml_scale(ctx0, q, scale);

    // [S, H, n, 1] → [S, n, H, 1] → [S, n, 1, H]; gates [1, H, n, 1] → [1, n, 1, H]
    q = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3)), S_k, n, 1, H_v);
    k = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3)), S_k, n, 1, H_v);
    v = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3)), S_v, n, 1, H_v);
    g = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, g, 0, 2, 1, 3)), 1,   n, 1, H_v);
    b = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, b, 0, 2, 1, 3)), 1,   n, 1, H_v);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    // Path-cumulative gate g⁺: for chains this is the plain cumsum; in
    // general g⁺_t = Σ_{u ∈ anc(t) ∪ t} g_u = m_inclᵀ · g.
    ggml_tensor * g_col = ggml_cont(ctx0, ggml_transpose(ctx0, g));      // [n, 1, 1, H]
    ggml_tensor * g_ps  = ggml_mul_mat(ctx0, m_incl, g_col);             // [n, 1, 1, H]

    // Pairwise decay D[u, t] = exp(g⁺_t − g⁺_u) on ancestor-or-self pairs.
    // Masking the exponent (not the exponential) makes out-of-window entries
    // exp(0)=1, mirroring the chunked path's tri-then-exp; they are zeroed
    // by the m_strict/m_incl products on kb/kq below.
    ggml_tensor * g_ps_row = ggml_reshape_4d(ctx0, g_ps, 1, n, 1, H_v);
    g_ps_row = ggml_repeat_4d(ctx0, g_ps_row, n, n, 1, H_v);             // [u, t, 1, H] = g⁺_t
    ggml_tensor * decay = ggml_sub(ctx0, g_ps_row, g_ps);                // g⁺_t − g⁺_u
    decay = ggml_exp(ctx0, ggml_mul(ctx0, decay, m_incl));

    // A[u, t] = M_strict[u, t] · (k_u · β_t k_t) · D[u, t]
    ggml_tensor * kb = ggml_mul_mat(ctx0, k, k_b);                       // [u, t, 1, H]
    kb = ggml_mul(ctx0, kb, decay);
    ggml_tensor * attn = ggml_mul(ctx0, kb, m_strict);

    // kq[u, t] = M_incl[u, t] · (k_u · q_t) · D[u, t] — the output-side
    // attention onto corrected values.
    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    kq = ggml_mul(ctx0, kq, decay);
    kq = ggml_mul(ctx0, kq, m_incl);

    // T = (I + A)⁻¹ via a lower-triangular solve: X = solve(I+A, −A), T = X + I.
    // Topological node order keeps A strictly lower-triangular under any tree.
    ggml_tensor * lhs = ggml_add(ctx0, attn, m_eye);
    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, ggml_neg(ctx0, attn), true, true, false);
    ggml_tensor * t_mat = ggml_add(ctx0, lin_solve, m_eye);              // [i, j, 1, H], T[j, i] entries

    // ṽ = T (β v − exp(g⁺) β k S₀): the corrected candidate updates.
    ggml_tensor * v_corr = ggml_mul_mat(ctx0,
        ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), t_mat);              // [S_v, t, 1, H] = (T v_b)_t
    ggml_tensor * g_exp = ggml_exp(ctx0, g_ps);                          // [n, 1, 1, H]
    ggml_tensor * kbg = ggml_mul(ctx0,
        ggml_cont(ctx0, ggml_transpose(ctx0, k_b)), g_exp);              // [t, S_k, 1, H] = e^{g⁺_t} β_t k_t
    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, t_mat);                 // [S_k, t, 1, H] = (T e^{g⁺} k_b)_t

    ggml_tensor * s_r = ggml_reshape_4d(ctx0, s, S_v, S_v, 1, H_v);
    ggml_tensor * v_state = ggml_mul_mat(ctx0, k_cd, s_r);               // [t, S_v, 1, H] = S₀ᵀ (T e^{g⁺} k_b)_t
    ggml_tensor * v_new = ggml_sub(ctx0,
        ggml_cont(ctx0, ggml_transpose(ctx0, v_corr)), v_state);         // [t, S_v, 1, H] = ṽ_t

    // o_t = e^{g⁺_t} S₀ᵀ q_t + Σ_u kq[u, t] ṽ_u
    ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_new, kq);                // [S_v, t, 1, H]
    ggml_tensor * g_exp_row = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp)); // [1, n, 1, H]
    ggml_tensor * q_g = ggml_mul(ctx0, q, g_exp_row);                    // [S_k, n, 1, H]
    ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s_r, q_g);             // [S_v, t, 1, H]
    ggml_tensor * o = ggml_add(ctx0, attn_inter, v_attn);                // [S_v, n, 1, H]

    DeltaNetSpecLAResult r;
    // [S_v, n, 1, H] → [S_v, H, n, 1] to match the fused op's output layout.
    r.output = ggml_cont(ctx0, ggml_permute(ctx0, o, 0, 2, 3, 1));
    r.v_new  = v_new;
    r.g_ps   = g_ps;
    return r;
}

void fill_specla_masks(const int32_t * parents, int n,
                       float * m_strict, float * m_incl, float * m_eye) {
    // Masks are [ne0 = u (ancestor), ne1 = t (node)]: element t*n + u.
    std::memset(m_strict, 0, sizeof(float) * (size_t)n * n);
    std::memset(m_incl,   0, sizeof(float) * (size_t)n * n);
    std::memset(m_eye,    0, sizeof(float) * (size_t)n * n);
    for (int t = 0; t < n; t++) {
        m_eye[(size_t)t * n + t]  = 1.0f;
        m_incl[(size_t)t * n + t] = 1.0f;
        // Ancestor closure: t inherits its parent's ancestor-or-self row.
        const int p = parents[t];
        if (p < 0) continue;
        GGML_ASSERT(p < t);  // DFS/topological order
        for (int u = 0; u <= p; u++) {
            const float a = m_incl[(size_t)p * n + u];
            m_incl[(size_t)t * n + u]   = a;
            m_strict[(size_t)t * n + u] = a;
        }
    }
}

}  // namespace dflash::common
