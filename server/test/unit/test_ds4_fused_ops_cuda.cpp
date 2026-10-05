// The fused DS4 ops against the op chains the graph builds without them:
// - ggml_ds4_router_select / ggml_ds4_router_weights vs softplus, sqrt, add,
//   top_k, protected routes, get_rows, sum_rows, clamp, div, scale: the same
//   expert ids in the same order and bit-identical weights;
// - ggml_ds4_hc_collapse vs mul, permute, cont, sum_rows: bit-identical.
// So LUCE_DS4_FUSE_ROUTER / LUCE_DS4_FUSE_COLLAPSE never change a token.
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
    ggml_backend_free(backend);
    std::printf("ds4 fused ops: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
