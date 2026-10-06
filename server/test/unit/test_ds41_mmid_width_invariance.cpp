// DS4.1 routed-expert MUL_MAT_ID width invariance: does a T-token batch (DSpark verify, each token with its own 6
// experts) produce bit-identical outputs to the same tokens run one at a time (single-token decode)?
//
// DSpark verification is documented as bit-identical to plain decode because its products are batch-invariant
// (ggml_backend_cuda_set_mmvq_batch_invariant). The mmvq width-invariant launch excludes MUL_MAT_ID (has_ids), and
// format-specific decode kernels may only apply at one token, so this checks the routed-expert products directly,
// per expert format and DS4.1 surface, with the invariance switch off and on. Weights are random blocks with sane
// scales (every IQ2_XXS / IQ3_XXS bit pattern is a valid encoding).
//
//   test_ds41_mmid_width_invariance [tokens=3] [experts=64]
//
// Prints one JSON line per (device, type, surface, invariant mode); exits 1 when any batch differs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Case { ggml_type type; const char * surface; int k, n; };

static void fill_blocks(std::vector<uint8_t> & w, ggml_type type, size_t n_blocks, std::mt19937 & rng) {
    const size_t bs = ggml_type_size(type);
    if (type == GGML_TYPE_Q2_0_ROCMFP2) {                          // ue4m3 scales: tile quantized random blocks
        constexpr size_t pool = 4096;
        std::normal_distribution<float> gauss(0.0f, 0.02f);
        std::vector<float> f(pool * size_t(ggml_blck_size(type)));
        for (auto & v : f) v = gauss(rng);
        std::vector<uint8_t> q(pool * bs);
        ggml_get_type_traits(type)->from_float_ref(f.data(), q.data(), int64_t(f.size()));
        for (size_t i = 0; i < n_blocks; ++i) std::memcpy(w.data() + i * bs, q.data() + (rng() % pool) * bs, bs);
        return;
    }
    std::uniform_real_distribution<float> dscale(0.005f, 0.02f);
    for (auto & b : w) b = uint8_t(rng());
    for (size_t i = 0; i < n_blocks; ++i) {
        uint8_t * blk = w.data() + i * bs;
        const ggml_fp16_t d = ggml_fp32_to_fp16(dscale(rng)), m = ggml_fp32_to_fp16(dscale(rng));
        switch (type) {
            case GGML_TYPE_Q2_K:                                   // scales[16] qs[64] d dmin
                std::memcpy(blk + bs - 4, &d, 2); std::memcpy(blk + bs - 2, &m, 2); break;
            case GGML_TYPE_MXFP4:                                  // e8m0 then 16 bytes of E2M1 pairs
                blk[0] = uint8_t(118 + rng() % 10); break;
            default:                                               // IQ2_XXS, IQ3_XXS, Q8_0: fp16 d first
                std::memcpy(blk, &d, 2); break;
        }
    }
}

static bool check(ggml_backend_t backend, int device, const Case & c, int tokens, int experts, bool invariant) {
    const int used = 6;
    const size_t row = ggml_row_size(c.type, c.k);
    std::mt19937 rng(20260930u + unsigned(c.type) * 7919u + unsigned(c.k));
    std::vector<uint8_t> w(row * size_t(c.n) * experts);
    fill_blocks(w, c.type, w.size() / ggml_type_size(c.type), rng);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(size_t(c.k) * tokens);
    for (auto & v : x) v = gauss(rng);
    std::vector<int32_t> ids(size_t(used) * tokens);
    for (int t = 0; t < tokens; ++t)                               // distinct and partly shared experts across tokens
        for (int j = 0; j < used; ++j) ids[size_t(t) * used + j] = (t * 3 + j * 5) % experts;

    ggml_init_params params{};
    params.mem_size = size_t(8 * tokens + 16) * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(8 * tokens + 16, false);
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    auto a = ggml_new_tensor_3d(ctx, c.type, c.k, c.n, experts);
    // decode: one MUL_MAT_ID per token
    std::vector<ggml_tensor *> xs, is, ys;
    auto g1 = ggml_new_graph_custom(ctx, 8 * tokens + 16, false);
    for (int t = 0; t < tokens; ++t) {
        auto b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, 1, 1);
        auto id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, 1);
        auto y = ggml_mul_mat_id(ctx, a, b, id);
        xs.push_back(b); is.push_back(id); ys.push_back(y);
        ggml_build_forward_expand(g1, y);
    }
    // verify: one MUL_MAT_ID over all tokens
    auto bb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, 1, tokens);
    auto ib = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
    auto yb = ggml_mul_mat_id(ctx, a, bb, ib);
    auto g2 = ggml_new_graph_custom(ctx, 8 * tokens + 16, false);
    ggml_build_forward_expand(g2, yb);
    if (!ggml_backend_supports_op(backend, yb) || !ggml_backend_supports_op(backend, ys[0])) {
        std::printf("{\"device\":%d,\"type\":\"%s\",\"surface\":\"%s\",\"error\":\"unsupported\"}\n", device,
                    ggml_type_name(c.type), c.surface);
        ggml_free(ctx);
        return true;
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(a, w.data(), 0, w.size());
    for (int t = 0; t < tokens; ++t) {
        ggml_backend_tensor_set(xs[t], x.data() + size_t(t) * c.k, 0, size_t(c.k) * sizeof(float));
        ggml_backend_tensor_set(is[t], ids.data() + size_t(t) * used, 0, size_t(used) * sizeof(int32_t));
    }
    ggml_backend_tensor_set(bb, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(ib, ids.data(), 0, ids.size() * sizeof(int32_t));

    const bool previous = ggml_backend_cuda_set_mmvq_batch_invariant(invariant);
    bool ok = ggml_backend_graph_compute(backend, g1) == GGML_STATUS_SUCCESS &&
              ggml_backend_graph_compute(backend, g2) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    ggml_backend_cuda_set_mmvq_batch_invariant(previous);

    const size_t per_token = size_t(c.n) * used;
    std::vector<float> single(per_token), batch(per_token * tokens);
    ggml_backend_tensor_get(yb, batch.data(), 0, batch.size() * sizeof(float));
    int diff_tokens = 0;
    size_t diff_values = 0;
    double max_abs = 0.0;
    for (int t = 0; t < tokens && ok; ++t) {
        ggml_backend_tensor_get(ys[t], single.data(), 0, per_token * sizeof(float));
        const float * bt = batch.data() + size_t(t) * per_token;
        if (std::memcmp(single.data(), bt, per_token * sizeof(float)) != 0) {
            ++diff_tokens;
            for (size_t i = 0; i < per_token; ++i)
                if (std::memcmp(&single[i], &bt[i], sizeof(float)) != 0) {
                    ++diff_values;
                    const double d = std::abs(double(single[i]) - double(bt[i]));
                    if (d > max_abs) max_abs = d;
                }
        }
    }
    std::printf("{\"device\":%d,\"name\":\"%s\",\"type\":\"%s\",\"surface\":\"%s\",\"k\":%d,\"n\":%d,\"tokens\":%d,"
                "\"batch_invariant\":%s,\"identical\":%s,\"diff_tokens\":%d,\"diff_values\":%zu,\"max_abs\":%.3g,\"ok\":%s}\n",
                device, ggml_backend_name(backend), ggml_type_name(c.type), c.surface, c.k, c.n, tokens,
                invariant ? "true" : "false", diff_tokens == 0 ? "true" : "false", diff_tokens, diff_values, max_abs,
                ok ? "true" : "false");
    std::fflush(stdout);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok && diff_tokens == 0;
}

// Dense products the verify batch runs q-wide (GGML_OP_MUL_MAT): T-column batch vs T single columns.
static bool check_dense(ggml_backend_t backend, int device, ggml_type type, const char * name, int k, int n, int tokens, bool invariant) {
    std::mt19937 rng(777u + unsigned(type) * 31u + unsigned(k));
    std::vector<float> wf(size_t(k) * n), x(size_t(k) * tokens);
    std::normal_distribution<float> gauss(0.0f, 0.02f);
    for (auto & v : wf) v = gauss(rng);
    for (auto & v : x) v = gauss(rng) * 50.0f;
    ggml_init_params params{};
    params.mem_size = size_t(4 * tokens + 8) * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(4 * tokens + 8, false);
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    auto a = ggml_new_tensor_2d(ctx, type, k, n);
    std::vector<ggml_tensor *> xs, ys;
    auto g1 = ggml_new_graph_custom(ctx, 4 * tokens + 8, false);
    for (int t = 0; t < tokens; ++t) {
        auto b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
        auto y = ggml_mul_mat(ctx, a, b);
        xs.push_back(b); ys.push_back(y); ggml_build_forward_expand(g1, y);
    }
    auto bb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
    auto yb = ggml_mul_mat(ctx, a, bb);
    auto g2 = ggml_new_graph_custom(ctx, 4 * tokens + 8, false);
    ggml_build_forward_expand(g2, yb);
    if (!ggml_backend_supports_op(backend, yb)) { ggml_free(ctx); return true; }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    // weights: quantize/convert from f32 through ggml's own reference
    std::vector<uint8_t> wq(ggml_row_size(type, k) * n);
    if (type == GGML_TYPE_F32) std::memcpy(wq.data(), wf.data(), wq.size());
    else ggml_quantize_chunk(type, wf.data(), wq.data(), 0, n, k, nullptr);
    ggml_backend_tensor_set(a, wq.data(), 0, wq.size());
    for (int t = 0; t < tokens; ++t) ggml_backend_tensor_set(xs[t], x.data() + size_t(t) * k, 0, size_t(k) * sizeof(float));
    ggml_backend_tensor_set(bb, x.data(), 0, x.size() * sizeof(float));
    const bool previous = ggml_backend_cuda_set_mmvq_batch_invariant(invariant);
    bool ok = ggml_backend_graph_compute(backend, g1) == GGML_STATUS_SUCCESS &&
              ggml_backend_graph_compute(backend, g2) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    ggml_backend_cuda_set_mmvq_batch_invariant(previous);
    std::vector<float> single(n), batch(size_t(n) * tokens);
    ggml_backend_tensor_get(yb, batch.data(), 0, batch.size() * sizeof(float));
    int diff_tokens = 0; size_t diff_values = 0; double max_abs = 0.0;
    for (int t = 0; t < tokens && ok; ++t) {
        ggml_backend_tensor_get(ys[t], single.data(), 0, size_t(n) * sizeof(float));
        const float * bt = batch.data() + size_t(t) * n;
        if (std::memcmp(single.data(), bt, size_t(n) * sizeof(float)) != 0) {
            ++diff_tokens;
            for (int i = 0; i < n; ++i) if (std::memcmp(&single[i], &bt[i], 4) != 0) { ++diff_values; max_abs = std::max(max_abs, std::abs(double(single[i]) - double(bt[i]))); }
        }
    }
    std::printf("{\"device\":%d,\"op\":\"mul_mat\",\"type\":\"%s\",\"surface\":\"%s\",\"k\":%d,\"n\":%d,\"tokens\":%d,\"batch_invariant\":%s,"
                "\"identical\":%s,\"diff_tokens\":%d,\"diff_values\":%zu,\"max_abs\":%.3g,\"ok\":%s}\n",
                device, ggml_type_name(type), name, k, n, tokens, invariant ? "true" : "false", diff_tokens == 0 ? "true" : "false",
                diff_tokens, diff_values, max_abs, ok ? "true" : "false");
    std::fflush(stdout);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return ok && diff_tokens == 0;
}

int main(int argc, char ** argv) {
    const int tokens = argc > 1 ? std::atoi(argv[1]) : 3, experts = argc > 2 ? std::atoi(argv[2]) : 64;
    // The contract covers the routed-expert formats DS4.1 plans use. Q8_0 routed experts take another
    // dispatch and are known to differ on CUDA; they are run as a diagnostic, not as part of the exit status.
    struct T { ggml_type type; bool contract; } types[] = {
        {GGML_TYPE_IQ2_XXS, true}, {GGML_TYPE_IQ2_XS, true}, {GGML_TYPE_IQ3_XXS, true},
        {GGML_TYPE_Q2_K, true}, {GGML_TYPE_MXFP4, true}, {GGML_TYPE_Q2_0_ROCMFP2, true}, {GGML_TYPE_Q8_0, false},
    };
    bool all = true;
    for (int dv = 0; dv < ggml_backend_cuda_get_device_count(); ++dv) {
        auto backend = ggml_backend_cuda_init(dv);
        if (!backend) return 2;
        for (const T & t : types) {
            const Case cases[] = {{t.type, "down", 2304, 5120}, {t.type, "gate_up", 5120, 2304}};
            // inv=false is the diagnostic (the multi-token kernels may differ); only the
            // batch-invariant mode of the contract formats decides the exit status.
            for (const auto & c : cases)
                for (bool inv : {false, true}) {
                    const bool ok = check(backend, dv, c, tokens, experts, inv);
                    if (inv && t.contract) all = ok && all;
                }
        }
        // dense verify-batch products of DS4.1: BF16 output head, F16 indexer/compressor, F32 router, MXFP8 attention
        struct D { ggml_type t; const char * name; int k, n; } dense[] = {
            {GGML_TYPE_BF16, "output_head", 5120, 129280}, {GGML_TYPE_F16, "indexer_k", 5120, 2048},
            {GGML_TYPE_F16, "compressor_kv", 5120, 1024}, {GGML_TYPE_F32, "router", 5120, 384},
            {GGML_TYPE_MXFP8, "attn_q_b", 1280, 32768}, {GGML_TYPE_Q8_0, "q8_dense", 5120, 2304},
        };
        for (const auto & d : dense)
            for (bool inv : {false, true}) {
                const bool ok = check_dense(backend, dv, d.t, d.name, d.k, d.n, tokens, inv);
                if (inv) all = ok && all;
            }
        ggml_backend_free(backend);
        if (argc > 3) break;                                       // first device only
    }
    return all ? 0 : 1;
}
