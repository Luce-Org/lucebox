// Qwen3.8-Flash-Next routed experts through the real ggml CUDA/HIP MUL_MAT_ID dispatch, as a split decode or verify
// step runs them on one device: gate and up (fused with SwiGLU by the backend), then down, for T tokens of n_used
// routes each, `masked` of them -1 (the picks the other device owns). LAYERS weight copies with distinct experts, so
// the device's last-level cache does not hold a repeated expert. Prints the time per layer, the bandwidth over the
// expert bytes the step reads, and an output hash so two env settings can be compared (same hash = same arithmetic).
//
// bench_qn_moe [T=3] [masked=5] [dev=1] [iters=20] [batch_invariant=1] [part=0] [share=0] [layers=8] [n_expert=64]
// part: 0 = gate/up then down, 1 = gate/up only, 2 = down only. masked = -1 runs a dense reference instead: one
// mul_mat of a whole layer's gate experts as a [n_embd, ff*n_expert] matrix by the T columns, the bandwidth the same
// weight type reaches without expert ids. share: the chance that a token's live route picks an expert the previous
// token also picked (consecutive verify tokens overlap), which lets a grouped kernel read it once. layers x n_expert
// sets the footprint the picks spread over (the model has 48 x 512, ~70 GB).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// Random bytes with sane block scales: every quant type here starts its blocks with f16 scales, and the bench only
// reads bandwidth and compares hashes, so the values only need to be finite.
static void fill_quant(ggml_backend_t, ggml_tensor * t, std::mt19937_64 & rng) {
    const size_t bytes = ggml_nbytes(t);
    std::vector<uint8_t> host((bytes + 7) / 8 * 8);
    for (size_t i = 0; i < host.size(); i += 8) { const uint64_t r = rng(); std::memcpy(host.data() + i, &r, 8); }
    const size_t bs = ggml_type_size(t->type);
    const ggml_fp16_t hs = ggml_fp32_to_fp16(0.01f);
    for (size_t b = 0; b + bs <= bytes; b += bs) {
        std::memcpy(host.data() + b, &hs, 2);
        if (t->type != GGML_TYPE_Q8_0) std::memcpy(host.data() + b + 2, &hs, 2);
    }
    ggml_backend_tensor_set(t, host.data(), 0, bytes);
}

int main(int argc, char ** argv) {
    const int T      = argc > 1 ? std::atoi(argv[1]) : 3;
    const int masked = argc > 2 ? std::atoi(argv[2]) : 5;
    const int dev    = argc > 3 ? std::atoi(argv[3]) : 1;
    const int iters  = argc > 4 ? std::atoi(argv[4]) : 20;
    const bool invariant = argc > 5 ? std::atoi(argv[5]) != 0 : true;
    const int part   = argc > 6 ? std::atoi(argv[6]) : 0;
    const double share = argc > 7 ? std::atof(argv[7]) : 0.0;
    const bool dense = masked < 0;
    const int layers   = argc > 8 ? std::atoi(argv[8]) : 8;
    const int n_expert = argc > 9 ? std::atoi(argv[9]) : 64;
    const int n_embd = 2560, ff = 640, n_used = 10;
    const ggml_type gu_type = GGML_TYPE_Q4_K, down_type = GGML_TYPE_Q5_1;

    ggml_backend_t be = ggml_backend_cuda_init(dev);
    if (!be) { std::fprintf(stderr, "no cuda backend %d\n", dev); return 1; }
    ggml_init_params ip = { ggml_tensor_overhead() * (layers * 8 + 16), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> G(layers), U(layers), D(layers), I(layers);
    for (int l = 0; l < layers; ++l) {
        G[l] = ggml_new_tensor_3d(ctx, gu_type, n_embd, ff, n_expert);
        U[l] = ggml_new_tensor_3d(ctx, gu_type, n_embd, ff, n_expert);
        D[l] = ggml_new_tensor_3d(ctx, down_type, ff, n_embd, n_expert);
        I[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    }
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, T);
    ggml_tensor * X2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);        // dense reference columns
    ggml_tensor * GU = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ff, n_used, T);    // down-only input
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { std::fprintf(stderr, "alloc failed\n"); return 1; }

    std::mt19937_64 rng(1234);
    for (int l = 0; l < layers; ++l) { fill_quant(be, G[l], rng); fill_quant(be, U[l], rng); fill_quant(be, D[l], rng); }
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> x((size_t) n_embd * T);
        for (float & v : x) v = nd(rng);
        ggml_backend_tensor_set(X, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(X2, x.data(), 0, x.size() * sizeof(float));
        std::vector<float> g((size_t) ff * n_used * T);
        for (float & v : g) v = nd(rng);
        ggml_backend_tensor_set(GU, g.data(), 0, g.size() * sizeof(float));
    }
    // Distinct experts per layer, except that with probability `share` a live route reuses one of the previous
    // token's live experts (never twice in one token); masked routes sit at random positions. `live` counts
    // (token, route) pairs, `uniq` the distinct experts a layer reads.
    size_t live = 0, uniq = 0;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    for (int l = 0; l < layers; ++l) {
        std::vector<int> perm(n_expert);
        for (int e = 0; e < n_expert; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<int32_t> ids((size_t) n_used * T);
        int next = 0;
        std::vector<int> prev_live;
        for (int t = 0; t < T; ++t) {
            std::vector<int> pos(n_used);
            for (int r = 0; r < n_used; ++r) pos[r] = r;
            std::shuffle(pos.begin(), pos.end(), rng);
            std::vector<int> mine;
            std::vector<int> cand = prev_live;
            std::shuffle(cand.begin(), cand.end(), rng);
            for (int r = 0; r < n_used; ++r) {
                int id = -1;
                if (r >= std::max(masked, 0)) {
                    if (!cand.empty() && ud(rng) < share) { id = cand.back(); cand.pop_back(); }
                    else { id = perm[next++ % n_expert]; }
                    mine.push_back(id);
                }
                ids[(size_t) t * n_used + pos[r]] = id;
            }
            prev_live = mine;
            live += mine.size();
        }
        uniq += (size_t) std::min(next, n_expert);
        ggml_backend_tensor_set(I[l], ids.data(), 0, ids.size() * sizeof(int32_t));
    }
    const double gu_bytes = 2.0 * ggml_row_size(gu_type, n_embd) * ff, down_bytes = (double) ggml_row_size(down_type, ff) * n_embd;
    const double expert_bytes = (part == 2 ? 0.0 : gu_bytes) + (part == 1 ? 0.0 : down_bytes);

    ggml_init_params gp = { ggml_tensor_overhead() * (layers * 8 + 16) + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    std::vector<ggml_tensor *> outs(layers);
    for (int l = 0; l < layers; ++l) {
        if (dense) {
            ggml_tensor * W = ggml_reshape_2d(gctx, G[l], n_embd, (int64_t) ff * n_expert);
            outs[l] = ggml_mul_mat(gctx, W, X2);
        } else if (part == 2) {
            outs[l] = ggml_mul_mat_id(gctx, D[l], GU, I[l]);
        } else {
            ggml_tensor * g = ggml_mul_mat_id(gctx, G[l], X, I[l]);
            ggml_tensor * u = ggml_mul_mat_id(gctx, U[l], X, I[l]);
            ggml_tensor * gu = ggml_swiglu_split(gctx, g, u);
            outs[l] = part == 1 ? gu : ggml_mul_mat_id(gctx, D[l], gu, I[l]);
        }
        ggml_set_output(outs[l]);
        ggml_build_forward_expand(gf, outs[l]);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { std::fprintf(stderr, "gallocr failed\n"); return 1; }
    const bool prev = ggml_backend_cuda_set_mmvq_batch_invariant(invariant);
    for (int w = 0; w < 3; ++w) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    const auto t1 = std::chrono::steady_clock::now();
    ggml_backend_cuda_set_mmvq_batch_invariant(prev);

    uint64_t h = 1469598103934665603ull;
    for (int l = 0; l < layers; ++l) {
        std::vector<float> o(ggml_nelements(outs[l]));
        ggml_backend_tensor_get(outs[l], o.data(), 0, o.size() * sizeof(float));
        // masked routes leave their rows unwritten: hash only the live ones
        std::vector<int32_t> ids((size_t) n_used * T);
        ggml_backend_tensor_get(I[l], ids.data(), 0, ids.size() * sizeof(int32_t));
        const size_t row = (size_t) outs[l]->ne[0];
        if (dense) { h = fnv(o.data(), o.size() * sizeof(float), h); continue; }
        for (size_t r = 0; r < ids.size(); ++r)
            if (ids[r] >= 0) h = fnv(o.data() + r * row, row * sizeof(float), h);
    }
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters / layers;
    // bandwidth over the distinct experts' bytes (what a step must read at least)
    const double bytes = dense ? (double) ggml_nbytes(G[0]) : (double) uniq / layers * expert_bytes;
    const double gbs = bytes / (us * 1e3);
    std::printf("T=%d masked=%d share=%.1f part=%d picks/layer=%.1f experts/layer=%.1f dev=%d inv=%d: %.1f us/layer, %.1f GB/s, hash %016llx\n",
                T, masked, share, part, dense ? 0.0 : (double) live / layers, dense ? 0.0 : (double) uniq / layers, dev,
                (int) invariant, us, gbs, (unsigned long long) h);
    ggml_gallocr_free(ga);
    ggml_free(gctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
