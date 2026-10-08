// The fused DS4 ops against the op chains the graph builds without them:
// - ggml_ds4_router_select / ggml_ds4_router_weights vs softplus, sqrt, add,
//   top_k, protected routes, get_rows, sum_rows, clamp, div, scale: the same
//   expert ids in the same order and bit-identical weights;
// - ggml_ds4_hc_collapse vs mul, permute, cont, sum_rows: bit-identical.
// - the HC sub-block boundary the CUDA backend runs as one launch
//   (rms_norm, F16 fn mix, hc_pre, hc_post, collapse, rms_norm * w) vs the
//   same chain kept unfused: bit-identical streams and normalized input.
// So LUCE_DS4_FUSE_ROUTER / LUCE_DS4_FUSE_COLLAPSE / the boundary fusion
// never change a token.
//
// Exit codes: 0 pass, 1 mismatch, 77 no GPU backend.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct RouterCase {
    int n_expert;
    int k;
    int tokens;
    bool protect;
    float scale;
};

bool run_case(ggml_backend_t backend, const RouterCase & c, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_expert, c.tokens);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.n_expert);
    ggml_tensor * native_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.n_expert);
    ggml_tensor * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_expert);
    for (ggml_tensor * t : {logits, bias, native_bias, mask}) ggml_set_input(t);

    // Reference: the unfused chain of build_moe_routing.
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    ggml_tensor * ref_ids = ggml_top_k(ctx, ggml_add(ctx, probs, bias), c.k);
    if (c.protect) {
        ggml_tensor * native = ggml_top_k(ctx, ggml_add(ctx, probs, native_bias), c.k);
        ref_ids = ggml_ds4_moe_protected_routes(ctx, ref_ids, native, mask);
    }
    ggml_tensor * ref_w = ggml_get_rows(
        ctx, ggml_reshape_3d(ctx, probs, 1, c.n_expert, c.tokens), ref_ids);
    ref_w = ggml_reshape_2d(ctx, ref_w, c.k, c.tokens);
    ggml_tensor * w_sum = ggml_clamp(ctx, ggml_sum_rows(ctx, ref_w), 6.103515625e-5f, INFINITY);
    ref_w = ggml_div(ctx, ref_w, w_sum);
    if (c.scale != 1.0f) ref_w = ggml_scale(ctx, ref_w, c.scale);

    // Fused.
    ggml_tensor * ids = ggml_ds4_router_select(
        ctx, logits, bias, c.protect ? native_bias : nullptr, c.protect ? mask : nullptr, c.k);
    ggml_tensor * wts = ggml_ds4_router_weights(ctx, logits, ids, 6.103515625e-5f, c.scale);

    for (ggml_tensor * t : {ref_ids, ref_w, ids, wts}) ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {ref_ids, ref_w, ids, wts}) ggml_build_forward_expand(gf, t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "allocation failed\n");
        ggml_free(ctx);
        return false;
    }

    // Logits on a 1/64 grid so equal router scores (ties) occur.
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> grid(-256, 256);
    std::uniform_real_distribution<float> small(-0.05f, 0.05f);
    std::vector<float> h_logits((size_t) c.n_expert * c.tokens);
    for (float & v : h_logits) v = (float) grid(rng) / 64.0f;
    std::vector<float> h_bias(c.n_expert), h_native(c.n_expert);
    std::vector<int32_t> h_mask(c.n_expert, 0);
    for (int e = 0; e < c.n_expert; ++e) {
        h_bias[e] = small(rng);
        h_native[e] = e % 3 == 0 ? h_bias[e] : small(rng);
        h_mask[e] = e % 17 == 0 ? 1 : 0;
    }
    ggml_backend_tensor_set(logits, h_logits.data(), 0, sizeof(float) * h_logits.size());
    ggml_backend_tensor_set(bias, h_bias.data(), 0, sizeof(float) * h_bias.size());
    ggml_backend_tensor_set(native_bias, h_native.data(), 0, sizeof(float) * h_native.size());
    ggml_backend_tensor_set(mask, h_mask.data(), 0, sizeof(int32_t) * h_mask.size());

    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) c.k * c.tokens;
    std::vector<int32_t> a_ids(n), b_ids(n);
    std::vector<float> a_w(n), b_w(n);
    if (ok) {
        ggml_backend_tensor_get(ref_ids, a_ids.data(), 0, sizeof(int32_t) * n);
        ggml_backend_tensor_get(ids, b_ids.data(), 0, sizeof(int32_t) * n);
        ggml_backend_tensor_get(ref_w, a_w.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(wts, b_w.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n && ok; ++i) {
            if (a_ids[i] != b_ids[i] || std::memcmp(&a_w[i], &b_w[i], sizeof(float)) != 0) {
                std::fprintf(stderr,
                             "mismatch n_expert=%d k=%d tokens=%d protect=%d scale=%g at token %zu slot %zu: "
                             "ref %d %.9g fused %d %.9g\n",
                             c.n_expert, c.k, c.tokens, c.protect ? 1 : 0, c.scale,
                             i / c.k, i % c.k, a_ids[i], a_w[i], b_ids[i], b_w[i]);
                ok = false;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

struct CollapseCase {
    int n_embd;
    int n_hc;
    int tokens;
    int pre_stride;  // floats between tokens' pre rows (>= n_hc), as an HC split view
};

bool run_collapse(ggml_backend_t backend, const CollapseCase & c, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * hc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) c.n_embd * c.n_hc, c.tokens);
    ggml_tensor * split = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.pre_stride, c.tokens);
    ggml_set_input(hc);
    ggml_set_input(split);
    ggml_tensor * pre = ggml_view_2d(ctx, split, c.n_hc, c.tokens, split->nb[1], 0);

    // Reference: ds4_build_hc_collapse without the fused op.
    ggml_tensor * hc3 = ggml_reshape_3d(ctx, hc, c.n_embd, c.n_hc, c.tokens);
    ggml_tensor * weighted = ggml_mul(ctx, hc3, ggml_reshape_3d(ctx, ggml_cont(ctx, pre), 1, c.n_hc, c.tokens));
    ggml_tensor * ref = ggml_reshape_2d(ctx,
        ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, weighted, 1, 0, 2, 3))), c.n_embd, c.tokens);
    ggml_tensor * fused = ggml_ds4_hc_collapse(ctx, hc, pre, c.n_hc);
    ggml_set_output(ref);
    ggml_set_output(fused);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, fused);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_hc((size_t) ggml_nelements(hc)), h_split((size_t) ggml_nelements(split));
    for (size_t i = 0; i < h_hc.size(); ++i) h_hc[i] = i % 29 == 0 ? 0.0f : normal(rng) * 3.0f;
    for (float & v : h_split) v = normal(rng);
    ggml_backend_tensor_set(hc, h_hc.data(), 0, sizeof(float) * h_hc.size());
    ggml_backend_tensor_set(split, h_split.data(), 0, sizeof(float) * h_split.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) c.n_embd * c.tokens;
    std::vector<float> a(n), b(n);
    if (ok) {
        ggml_backend_tensor_get(ref, a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(fused, b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n && ok; ++i) {
            // Bit for bit, signed zeros aside (an all-zero row).
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0 && !(a[i] == 0.0f && b[i] == 0.0f)) {
                std::fprintf(stderr, "collapse mismatch n_embd=%d n_hc=%d tokens=%d at %zu: ref %.9g fused %.9g\n",
                             c.n_embd, c.n_hc, c.tokens, i, a[i], b[i]);
                ok = false;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

struct BoundaryCase {
    int n_embd;
    int tokens;
    bool pre_read;  // something outside the chain reads hc_pre's output
};

// One boundary chain over the given inputs. `keep` marks the first rms_norm
// as a graph output, which keeps the backend from fusing the chain.
struct BoundaryChain {
    ggml_tensor * pre = nullptr;
    ggml_tensor * post = nullptr;
    ggml_tensor * out = nullptr;
    ggml_tensor * proj = nullptr;   // a Q8_0 projection of out (reads its q8_1 form)
};

BoundaryChain build_boundary(ggml_context * ctx, ggml_tensor * R, ggml_tensor * block_out,
                             ggml_tensor * fn, ggml_tensor * base, ggml_tensor * w, ggml_tensor * wq,
                             int n_embd, int n_hc, int tokens, bool keep, bool pre_read) {
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    BoundaryChain c;
    ggml_tensor * n1 = ggml_rms_norm(ctx, R, 1e-6f);
    if (keep) ggml_set_output(n1);
    ggml_tensor * mix = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, fn, n1), mix_dim, tokens);
    c.pre = ggml_ds4_hc_pre(ctx, mix, base, R, n_hc, 20, 0.5f, 0.75f, 1.25f);
    ggml_tensor * split = ggml_view_2d(ctx, c.pre, mix_dim, tokens,
                                       tokens > 1 ? c.pre->nb[1] : ggml_nbytes(c.pre),
                                       (size_t) n_embd * sizeof(float));
    c.post = ggml_ds4_hc_post(ctx, R, block_out, split, n_hc);   // 1-D at one token, as in the graph
    ggml_tensor * cpre = ggml_view_2d(ctx, split, n_hc, tokens, split->nb[1], 0);
    ggml_tensor * coll = ggml_ds4_hc_collapse(ctx, c.post, cpre, n_hc);
    c.out = ggml_mul(ctx, ggml_rms_norm(ctx, coll, 1e-6f), w);
    // The reference reads out through a view: no producer hands it a q8_1
    // form, so MMVQ quantizes out itself.
    c.proj = ggml_mul_mat(ctx, wq, keep ? ggml_view_2d(ctx, c.out, c.out->ne[0], c.out->ne[1], c.out->nb[1], 0)
                                         : c.out);
    if (pre_read) ggml_set_output(c.pre);
    ggml_set_output(c.post);
    ggml_set_output(c.out);
    ggml_set_output(c.proj);
    return c;
}

bool run_boundary(ggml_backend_t backend, const BoundaryCase & c, uint32_t seed) {
    const int n_hc = 4;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    const int64_t ncols = (int64_t) c.n_embd * n_hc;
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * R = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, c.tokens);
    ggml_tensor * block_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_embd, c.tokens);
    ggml_tensor * fn = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ncols, mix_dim);
    ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mix_dim);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.n_embd);
    const int n_proj = 64;
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, c.n_embd, n_proj);
    for (ggml_tensor * t : {R, block_out, fn, base, w, wq}) ggml_set_input(t);

    const BoundaryChain ref = build_boundary(ctx, R, block_out, fn, base, w, wq, c.n_embd, n_hc, c.tokens, true, c.pre_read);
    const BoundaryChain fused = build_boundary(ctx, R, block_out, fn, base, w, wq, c.n_embd, n_hc, c.tokens, false, c.pre_read);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {ref.post, ref.out, ref.proj, fused.post, fused.out, fused.proj}) ggml_build_forward_expand(gf, t);
    if (c.pre_read) {
        ggml_build_forward_expand(gf, ref.pre);
        ggml_build_forward_expand(gf, fused.pre);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_R((size_t) ncols * c.tokens), h_bo((size_t) c.n_embd * c.tokens);
    std::vector<float> h_base(mix_dim), h_w(c.n_embd);
    std::vector<ggml_fp16_t> h_fn((size_t) ncols * mix_dim);
    for (float & v : h_R) v = normal(rng) * 4.0f;
    for (float & v : h_bo) v = normal(rng) * 2.0f;
    for (float & v : h_base) v = normal(rng) * 0.5f;
    for (float & v : h_w) v = 1.0f + 0.1f * normal(rng);
    for (ggml_fp16_t & v : h_fn) v = ggml_fp32_to_fp16(normal(rng) * 0.02f);
    std::vector<float> h_wf((size_t) c.n_embd * n_proj);
    for (float & v : h_wf) v = normal(rng) * 0.05f;
    std::vector<uint8_t> h_wq(ggml_nbytes(wq));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, h_wf.data(), h_wq.data(), 0, n_proj, c.n_embd, nullptr);
    ggml_backend_tensor_set(wq, h_wq.data(), 0, h_wq.size());
    ggml_backend_tensor_set(R, h_R.data(), 0, sizeof(float) * h_R.size());
    ggml_backend_tensor_set(block_out, h_bo.data(), 0, sizeof(float) * h_bo.size());
    ggml_backend_tensor_set(base, h_base.data(), 0, sizeof(float) * h_base.size());
    ggml_backend_tensor_set(w, h_w.data(), 0, sizeof(float) * h_w.size());
    ggml_backend_tensor_set(fn, h_fn.data(), 0, sizeof(ggml_fp16_t) * h_fn.size());

    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    auto same = [&](ggml_tensor * a, ggml_tensor * b, const char * what) {
        const size_t n = (size_t) ggml_nelements(a);
        std::vector<float> va(n), vb(n);
        ggml_backend_tensor_get(a, va.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(b, vb.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&va[i], &vb[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "boundary %s mismatch n_embd=%d tokens=%d pre_read=%d at %zu: ref %.9g fused %.9g\n",
                             what, c.n_embd, c.tokens, c.pre_read ? 1 : 0, i, va[i], vb[i]);
                return false;
            }
        }
        return true;
    };
    if (ok && c.pre_read) ok = same(ref.pre, fused.pre, "pre");
    if (ok) ok = same(ref.post, fused.post, "post") && same(ref.out, fused.out, "out") &&
                 same(ref.proj, fused.proj, "proj");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// rms_norm * w read by a quantized matmul: the fused norm writes the q8_1
// form MMVQ would quantize; the reference matmul reads through a view.
bool run_norm_q8(ggml_backend_t backend, int ncols, int tokens, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    const int n_proj = 96;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, tokens);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ncols);
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, n_proj);
    for (ggml_tensor * t : {x, w, wq}) ggml_set_input(t);
    ggml_tensor * ref_n = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
    ggml_tensor * ref = ggml_mul_mat(ctx, wq, ggml_view_2d(ctx, ref_n, ncols, tokens, ref_n->nb[1], 0));
    ggml_tensor * fus_n = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
    ggml_tensor * fus = ggml_mul_mat(ctx, wq, fus_n);
    for (ggml_tensor * t : {ref_n, ref, fus_n, fus}) ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {ref_n, ref, fus_n, fus}) ggml_build_forward_expand(gf, t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_x((size_t) ncols * tokens), h_w(ncols), h_wf((size_t) ncols * n_proj);
    for (float & v : h_x) v = normal(rng) * 3.0f;
    for (float & v : h_w) v = 1.0f + 0.2f * normal(rng);
    for (float & v : h_wf) v = normal(rng) * 0.05f;
    std::vector<uint8_t> h_wq(ggml_nbytes(wq));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, h_wf.data(), h_wq.data(), 0, n_proj, ncols, nullptr);
    ggml_backend_tensor_set(x, h_x.data(), 0, sizeof(float) * h_x.size());
    ggml_backend_tensor_set(w, h_w.data(), 0, sizeof(float) * h_w.size());
    ggml_backend_tensor_set(wq, h_wq.data(), 0, h_wq.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    for (auto pr : {std::make_pair(ref_n, fus_n), std::make_pair(ref, fus)}) {
        if (!ok) break;
        const size_t n = (size_t) ggml_nelements(pr.first);
        std::vector<float> a(n), b(n);
        ggml_backend_tensor_get(pr.first, a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(pr.second, b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "norm q8 mismatch ncols=%d tokens=%d at %zu: ref %.9g fused %.9g\n",
                             ncols, tokens, i, a[i], b[i]);
                ok = false;
                break;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// ggml_moe_combine_masked vs the owner remap's mul(weights, get_rows(valid,
// ids)), mul(experts, masked) and repeat_back.
bool run_masked_combine(ggml_backend_t backend, int n_embd, int n_used, int n_expert, int tokens, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_used, tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_used, tokens);
    ggml_tensor * valid = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, n_expert, tokens, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, tokens);
    for (ggml_tensor * t : {experts, weights, valid, ids}) ggml_set_input(t);
    ggml_tensor * v = ggml_reshape_2d(ctx, ggml_get_rows(ctx, valid, ids), n_used, tokens);
    ggml_tensor * masked = ggml_mul(ctx, weights, v);
    ggml_tensor * prod = ggml_mul(ctx, experts, ggml_reshape_3d(ctx, masked, 1, n_used, tokens));
    ggml_tensor * ref = ggml_reshape_2d(ctx,
        ggml_repeat_back(ctx, prod, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, tokens)), n_embd, tokens);
    ggml_tensor * fused = ggml_moe_combine_masked(ctx, experts, weights, valid, ids);
    ggml_set_output(ref);
    ggml_set_output(fused);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, fused);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.01f, 1.0f);
    std::uniform_int_distribution<int> pick(0, n_expert - 1);
    std::vector<float> h_e((size_t) n_embd * n_used * tokens), h_w((size_t) n_used * tokens);
    std::vector<float> h_v((size_t) n_expert * tokens);
    std::vector<int32_t> h_ids((size_t) n_used * tokens);
    for (float & x : h_e) x = normal(rng);
    for (float & x : h_w) x = uni(rng);
    for (size_t i = 0; i < h_v.size(); ++i) h_v[i] = (i % 3 == 0) ? 0.0f : 1.0f;
    for (int32_t & x : h_ids) x = pick(rng);
    ggml_backend_tensor_set(experts, h_e.data(), 0, sizeof(float) * h_e.size());
    ggml_backend_tensor_set(weights, h_w.data(), 0, sizeof(float) * h_w.size());
    ggml_backend_tensor_set(valid, h_v.data(), 0, sizeof(float) * h_v.size());
    ggml_backend_tensor_set(ids, h_ids.data(), 0, sizeof(int32_t) * h_ids.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) n_embd * tokens;
    std::vector<float> a(n), b(n);
    if (ok) {
        ggml_backend_tensor_get(ref, a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(fused, b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n && ok; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0 && !(a[i] == 0.0f && b[i] == 0.0f)) {
                std::fprintf(stderr, "masked combine mismatch n_embd=%d n_used=%d tokens=%d at %zu: ref %.9g fused %.9g\n",
                             n_embd, n_used, tokens, i, a[i], b[i]);
                ok = false;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// swiglu_ds4 read by a routed (MUL_MAT_ID) Q8_0 down projection: the GLU
// launch writes the q8_1 form MMVQ would quantize; the reference reads
// through a view.
bool run_glu_q8(ggml_backend_t backend, int n_ff, int n_used, int tokens, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    const int n_out = 64, n_expert = 8;
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_used, tokens);
    ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_used, tokens);
    ggml_tensor * wq = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, n_ff, n_out, n_expert);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, tokens);
    for (ggml_tensor * t : {gate, up, wq, ids}) ggml_set_input(t);
    ggml_tensor * ref_g = ggml_swiglu_ds4_split(ctx, gate, up, 7.0f);
    ggml_tensor * ref = ggml_mul_mat_id(ctx, wq,
        ggml_view_3d(ctx, ref_g, n_ff, n_used, tokens, ref_g->nb[1], ref_g->nb[2], 0), ids);
    ggml_tensor * fus_g = ggml_swiglu_ds4_split(ctx, gate, up, 7.0f);
    ggml_tensor * fus = ggml_mul_mat_id(ctx, wq, fus_g, ids);
    for (ggml_tensor * t : {ref_g, ref, fus_g, fus}) ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {ref_g, ref, fus_g, fus}) ggml_build_forward_expand(gf, t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, n_expert - 1);
    std::vector<float> h_g((size_t) n_ff * n_used * tokens), h_u(h_g.size());
    for (float & v : h_g) v = normal(rng) * 4.0f;
    for (float & v : h_u) v = normal(rng) * 4.0f;
    std::vector<float> h_wf((size_t) n_ff * n_out * n_expert);
    for (float & v : h_wf) v = normal(rng) * 0.05f;
    std::vector<uint8_t> h_wq(ggml_nbytes(wq));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, h_wf.data(), h_wq.data(), 0, (int64_t) n_out * n_expert, n_ff, nullptr);
    std::vector<int32_t> h_ids((size_t) n_used * tokens);
    for (int32_t & v : h_ids) v = pick(rng);
    ggml_backend_tensor_set(gate, h_g.data(), 0, sizeof(float) * h_g.size());
    ggml_backend_tensor_set(up, h_u.data(), 0, sizeof(float) * h_u.size());
    ggml_backend_tensor_set(wq, h_wq.data(), 0, h_wq.size());
    ggml_backend_tensor_set(ids, h_ids.data(), 0, sizeof(int32_t) * h_ids.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    for (auto pr : {std::make_pair(ref_g, fus_g), std::make_pair(ref, fus)}) {
        if (!ok) break;
        const size_t n = (size_t) ggml_nelements(pr.first);
        std::vector<float> a(n), b(n);
        ggml_backend_tensor_get(pr.first, a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(pr.second, b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "glu q8 mismatch n_ff=%d n_used=%d tokens=%d at %zu: ref %.9g fused %.9g\n",
                             n_ff, n_used, tokens, i, a[i], b[i]);
                ok = false;
                break;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// The cluster shared expert's gate/up pair: Q8_0 row slices of wider weights
// at several columns (batch-invariant MMVQ), fused with SwiGLU-DS4 into one
// RDNA4 launch (ggml_cuda_mmvq_rdna4_glu_pair) vs the same products kept
// apart (a copy between each product and the GLU breaks the pattern).
bool run_glu_pair(ggml_backend_t backend, int n_in, int n_full, int n_ff, int ff_begin, int ncols, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 48 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * gw = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_in, n_full);
    ggml_tensor * uw = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_in, n_full);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, ncols);
    for (ggml_tensor * t : {gw, uw, x}) ggml_set_input(t);
    auto slice = [&](ggml_tensor * w) {
        return ggml_view_2d(ctx, w, n_in, n_ff, w->nb[1], (size_t) ff_begin * w->nb[1]);
    };
    ggml_tensor * fus = ggml_swiglu_ds4_split(ctx, ggml_mul_mat(ctx, slice(gw), x), ggml_mul_mat(ctx, slice(uw), x), 7.0f);
    ggml_tensor * ref = ggml_swiglu_ds4_split(ctx, ggml_cont(ctx, ggml_mul_mat(ctx, slice(gw), x)),
                                              ggml_cont(ctx, ggml_mul_mat(ctx, slice(uw), x)), 7.0f);
    for (ggml_tensor * t : {fus, ref}) ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fus);
    ggml_build_forward_expand(gf, ref);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_w((size_t) n_in * n_full), h_x((size_t) n_in * ncols);
    std::vector<uint8_t> h_q(ggml_nbytes(gw));
    for (ggml_tensor * w : {gw, uw}) {
        for (float & v : h_w) v = normal(rng) * 0.05f;
        ggml_quantize_chunk(GGML_TYPE_Q8_0, h_w.data(), h_q.data(), 0, n_full, n_in, nullptr);
        ggml_backend_tensor_set(w, h_q.data(), 0, h_q.size());
    }
    for (float & v : h_x) v = normal(rng);
    ggml_backend_tensor_set(x, h_x.data(), 0, sizeof(float) * h_x.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) n_ff * ncols;
    std::vector<float> a(n), b(n);
    if (ok) {
        ggml_backend_tensor_get(ref, a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(fus, b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "glu pair mismatch n_in=%d n_ff=%d begin=%d cols=%d at %zu: ref %.9g fused %.9g\n",
                             n_in, n_ff, ff_begin, ncols, i, a[i], b[i]);
                ok = false;
                break;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// A rank's slice of a Q8_0 contraction: a column-range view flagged with
// ggml_backend_cuda_mul_mat_whole_row_lanes vs the whole rows times the slice
// zero-padded to full width (build_shared_ffn_slice's padded form). The lanes
// are a HIP layout, so there the two are bit-identical; other backends compute
// the range's product as usual, and the check is the value to rounding.
#if defined(GGML_USE_HIP)
constexpr bool k_col_slice_bit_exact = true;
#else
constexpr bool k_col_slice_bit_exact = false;
#endif

bool run_col_slice(ggml_backend_t backend, int n_out, int n_full, int k_begin, int k_count, int ncols, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 48 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_full, n_out);
    ggml_tensor * mid = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_count, ncols);
    for (ggml_tensor * t : {w, mid}) ggml_set_input(t);
    ggml_tensor * view = ggml_view_2d(ctx, w, k_count, n_out, w->nb[1], ggml_row_size(GGML_TYPE_Q8_0, k_begin));
    ggml_tensor * fus = ggml_mul_mat(ctx, view, mid);
    ggml_backend_cuda_mul_mat_whole_row_lanes(fus);
    ggml_tensor * zeros = ggml_scale(ctx, mid, 0.0f);
    ggml_tensor * padded = nullptr;
    for (int off = 0; off < n_full; off += k_count) {
        ggml_tensor * piece = off == k_begin ? mid : zeros;
        padded = padded ? ggml_concat(ctx, padded, piece, 0) : piece;
    }
    ggml_tensor * ref = ggml_mul_mat(ctx, w, padded);
    for (ggml_tensor * t : {fus, ref}) ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fus);
    ggml_build_forward_expand(gf, ref);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_w((size_t) n_full * n_out), h_m((size_t) k_count * ncols);
    for (float & v : h_w) v = normal(rng) * 0.05f;
    for (float & v : h_m) v = normal(rng);
    std::vector<uint8_t> h_q(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, h_w.data(), h_q.data(), 0, n_out, n_full, nullptr);
    ggml_backend_tensor_set(w, h_q.data(), 0, h_q.size());
    ggml_backend_tensor_set(mid, h_m.data(), 0, sizeof(float) * h_m.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) n_out * ncols;
    std::vector<float> a(n), b(n);
    if (ok) {
        ggml_backend_tensor_get(ref, a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(fus, b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n; ++i) {
            const bool same = k_col_slice_bit_exact
                ? std::memcmp(&a[i], &b[i], sizeof(float)) == 0
                : std::fabs(a[i] - b[i]) <= 1e-5f * fmaxf(1.0f, std::fabs(a[i]));
            if (!same) {
                std::fprintf(stderr, "col slice mismatch n_out=%d n_full=%d begin=%d count=%d cols=%d at %zu: ref %.9g fused %.9g\n",
                             n_out, n_full, k_begin, k_count, ncols, i, a[i], b[i]);
                ok = false;
                break;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

#if defined(GGML_USE_HIP)
// rms_norm * w -> tail rope -> F16 copy (the K/V latent of the flash lanes)
// in one launch vs the three; the reference keeps its rms_norm as an output.
bool run_norm_rope(ggml_backend_t backend, int ncols, int tokens, int pos0, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, tokens);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ncols);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    for (ggml_tensor * t : {x, w, pos}) ggml_set_input(t);
    ggml_tensor * outs[2][2];
    for (int k = 0; k < 2; ++k) {
        ggml_tensor * nrm = ggml_rms_norm(ctx, x, 1e-6f);
        if (k == 0) ggml_set_output(nrm);
        ggml_tensor * m = ggml_mul(ctx, nrm, w);
        ggml_tensor * r = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, m, ncols, 1, tokens), pos, nullptr,
                                        64, GGML_ROPE_TYPE_NORMAL | GGML_ROPE_TYPE_TAIL, 65536,
                                        10000.0f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f);
        ggml_tensor * r2 = ggml_reshape_2d(ctx, r, ncols, tokens);
        outs[k][0] = r2;
        outs[k][1] = ggml_cast(ctx, r2, GGML_TYPE_F16);
        ggml_set_output(outs[k][0]);
        ggml_set_output(outs[k][1]);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int k = 0; k < 2; ++k) {
        ggml_build_forward_expand(gf, outs[k][1]);
        ggml_build_forward_expand(gf, outs[k][0]);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_x((size_t) ncols * tokens), h_w(ncols);
    for (float & v : h_x) v = normal(rng) * 2.0f;
    for (float & v : h_w) v = 1.0f + 0.3f * normal(rng);
    std::vector<int32_t> h_pos(tokens);
    for (int t = 0; t < tokens; ++t) h_pos[t] = pos0 + t;
    ggml_backend_tensor_set(x, h_x.data(), 0, sizeof(float) * h_x.size());
    ggml_backend_tensor_set(w, h_w.data(), 0, sizeof(float) * h_w.size());
    ggml_backend_tensor_set(pos, h_pos.data(), 0, sizeof(int32_t) * h_pos.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) ncols * tokens;
    if (ok) {
        std::vector<float> a(n), b(n);
        std::vector<ggml_fp16_t> ah(n), bh(n);
        ggml_backend_tensor_get(outs[0][0], a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(outs[1][0], b.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(outs[0][1], ah.data(), 0, sizeof(ggml_fp16_t) * n);
        ggml_backend_tensor_get(outs[1][1], bh.data(), 0, sizeof(ggml_fp16_t) * n);
        for (size_t i = 0; i < n && ok; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0 || ah[i] != bh[i]) {
                std::fprintf(stderr, "norm rope mismatch ncols=%d tokens=%d at %zu: ref %.9g fused %.9g\n",
                             ncols, tokens, i, a[i], b[i]);
                ok = false;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}
#endif

// cont(permute(x)) read only by a Q8_0 matmul: quantized straight from the
// view, vs the same graph with the cont read through a view (kept).
bool run_cont_mmvq(ggml_backend_t backend, int ne0, int ne1, int tokens, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    const int n_out = 96;
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, tokens, ne1);   // [ne0, T, groups]
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ne0 * ne1, n_out);
    for (ggml_tensor * t : {x, wq}) ggml_set_input(t);
    ggml_tensor * res[2];
    for (int k = 0; k < 2; ++k) {
        ggml_tensor * c = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));          // [ne0, groups, T]
        ggml_tensor * r = ggml_reshape_2d(ctx, c, (int64_t) ne0 * ne1, tokens);
        if (k == 0) r = ggml_view_2d(ctx, r, r->ne[0], r->ne[1], r->nb[1], 0);
        res[k] = ggml_mul_mat(ctx, wq, r);
        ggml_set_output(res[k]);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int k = 0; k < 2; ++k) ggml_build_forward_expand(gf, res[k]);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_x((size_t) ne0 * ne1 * tokens), h_wf((size_t) ne0 * ne1 * n_out);
    for (float & v : h_x) v = normal(rng);
    for (float & v : h_wf) v = normal(rng) * 0.05f;
    std::vector<uint8_t> h_wq(ggml_nbytes(wq));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, h_wf.data(), h_wq.data(), 0, n_out, (int64_t) ne0 * ne1, nullptr);
    ggml_backend_tensor_set(x, h_x.data(), 0, sizeof(float) * h_x.size());
    ggml_backend_tensor_set(wq, h_wq.data(), 0, h_wq.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) n_out * tokens;
    if (ok) {
        std::vector<float> a(n), b(n);
        ggml_backend_tensor_get(res[0], a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(res[1], b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n && ok; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "cont mmvq mismatch ne0=%d ne1=%d tokens=%d at %zu: ref %.9g fused %.9g\n",
                             ne0, ne1, tokens, i, a[i], b[i]);
                ok = false;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// Masked single-token attention softmax with a sink column: scale, add the
// mask row, concat the sinks, soft_max, view vs one sink-column launch.
bool run_sink_mask(ggml_backend_t backend, int n_attn, int n_head, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    const float kq_scale = 0.0441941738f;
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_attn, n_head);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_attn, 1);
    ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);
    for (ggml_tensor * t : {scores, mask, sinks}) ggml_set_input(t);
    ggml_tensor * s = ggml_add(ctx, ggml_scale(ctx, scores, kq_scale), mask);
    ggml_tensor * sws = ggml_concat(ctx, s, ggml_reshape_2d(ctx, sinks, 1, n_head), 0);
    ggml_tensor * pws = ggml_soft_max(ctx, sws);
    ggml_tensor * ref = ggml_cont(ctx, ggml_view_2d(ctx, pws, n_attn, n_head, pws->nb[1], 0));
    ggml_tensor * fused = ggml_reshape_2d(ctx, ggml_soft_max_ext_sink_col_mask(
        ctx, ggml_reshape_3d(ctx, scores, n_attn, 1, n_head), mask, sinks, kq_scale), n_attn, n_head);
    ggml_set_output(ref);
    ggml_set_output(fused);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, fused);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_s((size_t) n_attn * n_head), h_m(n_attn), h_k(n_head);
    for (float & v : h_s) v = normal(rng) * 20.0f;
    for (int i = 0; i < n_attn; ++i) h_m[i] = (i % 7 == 3) ? -INFINITY : 0.0f;
    for (float & v : h_k) v = normal(rng);
    ggml_backend_tensor_set(scores, h_s.data(), 0, sizeof(float) * h_s.size());
    ggml_backend_tensor_set(mask, h_m.data(), 0, sizeof(float) * h_m.size());
    ggml_backend_tensor_set(sinks, h_k.data(), 0, sizeof(float) * h_k.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) n_attn * n_head;
    if (ok) {
        std::vector<float> a(n), b(n);
        ggml_backend_tensor_get(ref, a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(fused, b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n && ok; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "sink mask mismatch n_attn=%d n_head=%d at %zu: ref %.9g fused %.9g\n",
                             n_attn, n_head, i, a[i], b[i]);
                ok = false;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// Inverse tail rope of the attention context read by the grouped output
// projection (batched Q8_0 MMVQ over a [D * hpg, T, groups] view): the rope
// writes the view's q8_1; the reference reads a copy of the rope output.
bool run_rope_q8(ggml_backend_t backend, int n_head, int hpg, int tokens, uint32_t seed) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    const int D = 512, n_out = 64;
    const int G = D * hpg, NG = n_head / hpg;
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_head, tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_tensor * wq = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, G, n_out, NG);
    for (ggml_tensor * t : {x, pos, wq}) ggml_set_input(t);
    ggml_tensor * res[2];
    for (int k = 0; k < 2; ++k) {
        ggml_tensor * r = ggml_rope_ext(ctx, x, pos, nullptr, 64, GGML_ROPE_TYPE_NORMAL | GGML_ROPE_TYPE_TAIL,
                                        65536, 10000.0f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f);
        if (k == 0) r = ggml_cont(ctx, r);
        ggml_tensor * v = ggml_permute(ctx, ggml_reshape_3d(ctx, r, G, NG, tokens), 0, 2, 1, 3);
        res[k] = ggml_mul_mat(ctx, wq, v);
        ggml_set_output(res[k]);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int k = 0; k < 2; ++k) ggml_build_forward_expand(gf, res[k]);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h_x((size_t) D * n_head * tokens), h_wf((size_t) G * n_out * NG);
    for (float & v : h_x) v = normal(rng);
    for (float & v : h_wf) v = normal(rng) * 0.05f;
    std::vector<uint8_t> h_wq(ggml_nbytes(wq));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, h_wf.data(), h_wq.data(), 0, (int64_t) n_out * NG, G, nullptr);
    std::vector<int32_t> h_pos(tokens);
    for (int t = 0; t < tokens; ++t) h_pos[t] = -(1234 + t);
    ggml_backend_tensor_set(x, h_x.data(), 0, sizeof(float) * h_x.size());
    ggml_backend_tensor_set(pos, h_pos.data(), 0, sizeof(int32_t) * h_pos.size());
    ggml_backend_tensor_set(wq, h_wq.data(), 0, h_wq.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const size_t n = (size_t) ggml_nelements(res[0]);
    if (ok) {
        std::vector<float> a(n), b(n);
        ggml_backend_tensor_get(res[0], a.data(), 0, sizeof(float) * n);
        ggml_backend_tensor_get(res[1], b.data(), 0, sizeof(float) * n);
        for (size_t i = 0; i < n && ok; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "rope q8 mismatch n_head=%d hpg=%d tokens=%d at %zu: ref %.9g fused %.9g\n",
                             n_head, hpg, tokens, i, a[i], b[i]);
                ok = false;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "no GPU backend: skipped\n");
        return 77;
    }
    const RouterCase cases[] = {
        {384, 6, 1, false, 1.5f},
        {384, 6, 1, true, 1.5f},
        {384, 6, 5, true, 1.5f},
        {384, 6, 64, true, 1.5f},
        {384, 8, 7, false, 1.0f},
        {256, 8, 16, true, 2.5f},
        {640, 6, 3, true, 1.5f},
    };
    int failures = 0;
    uint32_t seed = 1;
    for (const RouterCase & c : cases) {
        for (int rep = 0; rep < 4; ++rep) {
            if (!run_case(backend, c, seed++)) ++failures;
        }
    }
    const CollapseCase collapse_cases[] = {
        {4096, 4, 1, 24},
        {4096, 4, 3, 4},
        {4096, 4, 64, 24},
        {4096, 4, 2170, 24},
        {1024, 8, 17, 8},
        {256, 4, 70000, 4},  // past the 65535 grid.y limit
    };
    for (const CollapseCase & c : collapse_cases) {
        for (int rep = 0; rep < 2; ++rep) {
            if (!run_collapse(backend, c, seed++)) ++failures;
        }
    }
    const BoundaryCase boundary_cases[] = {
        {5120, 1, false}, {5120, 2, false}, {5120, 3, true}, {5120, 5, false},
        {5120, 5, true}, {5120, 8, false}, {4096, 4, false}, {1024, 5, true},
    };
    for (const BoundaryCase & c : boundary_cases) {
        for (int rep = 0; rep < 2; ++rep) {
            if (!run_boundary(backend, c, seed++)) ++failures;
        }
    }
    const int norm_cases[][2] = {{1280, 1}, {1280, 3}, {1280, 5}, {512, 4}, {5120, 2}, {800, 5}};
    for (const auto & nc : norm_cases) {
        if (!run_norm_q8(backend, nc[0], nc[1], seed++)) ++failures;
    }
    const int combine_cases[][4] = {{5120, 6, 384, 1}, {5120, 6, 384, 5}, {512, 8, 64, 3}, {2048, 2, 16, 7}};
    for (const auto & cc : combine_cases) {
        if (!run_masked_combine(backend, cc[0], cc[1], cc[2], cc[3], seed++)) ++failures;
    }
    const int glu_cases[][3] = {{2304, 6, 5}, {2304, 6, 1}, {2304, 3, 3}, {512, 2, 4}};
    for (const auto & gc : glu_cases) {
        if (!run_glu_q8(backend, gc[0], gc[1], gc[2], seed++)) ++failures;
    }
#if defined(GGML_USE_HIP)
    const int rope_cases[][3] = {{512, 5, 1000}, {512, 1, 77777}, {512, 3, 5}, {1024, 4, 123456}};
    for (const auto & rc : rope_cases) {
        if (!run_norm_rope(backend, rc[0], rc[1], rc[2], seed++)) ++failures;
    }
#else
    std::fprintf(stderr, "norm rope ... skipped (HIP-only candidate)\n");
#endif
    const int cont_cases[][3] = {{1024, 4, 5}, {1024, 4, 1}, {512, 8, 3}, {1024, 2, 8}};
    for (const auto & cc : cont_cases) {
        if (!run_cont_mmvq(backend, cc[0], cc[1], cc[2], seed++)) ++failures;
    }
    const int sink_cases[][2] = {{128, 32}, {129, 32}, {128, 64}, {640, 16}};
    for (const auto & sc : sink_cases) {
        if (!run_sink_mask(backend, sc[0], sc[1], seed++)) ++failures;
    }
    const int rope_q8_cases[][3] = {{32, 8, 5}, {32, 8, 2}, {64, 8, 3}, {16, 4, 5}};
    for (const auto & rc : rope_q8_cases) {
        if (!run_rope_q8(backend, rc[0], rc[1], rc[2], seed++)) ++failures;
    }
    {
        // Verify-shaped products: batch-invariant MMVQ, as the DS4 verify runs them.
        const bool prev = ggml_backend_cuda_set_mmvq_batch_invariant(true);
        const int pair_cases[][5] = {{1024, 64, 32, 0, 5}, {1024, 64, 32, 32, 4}, {5120, 96, 48, 48, 2}, {1024, 64, 32, 32, 8}};
        for (const auto & pc : pair_cases) {
            if (!run_glu_pair(backend, pc[0], pc[1], pc[2], pc[3], pc[4], seed++)) ++failures;
        }
        const int slice_cases[][5] = {{64, 2304, 0, 1152, 5}, {64, 2304, 1152, 1152, 4}, {96, 4096, 2048, 2048, 3}, {64, 2304, 1152, 1152, 1}};
        for (const auto & sc : slice_cases) {
            if (!run_col_slice(backend, sc[0], sc[1], sc[2], sc[3], sc[4], seed++)) ++failures;
        }
        ggml_backend_cuda_set_mmvq_batch_invariant(prev);
    }
    ggml_backend_free(backend);
    std::printf("ds4 fused ops: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
