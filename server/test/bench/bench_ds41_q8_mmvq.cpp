// DS4.1 dense Q8_0 projections through the real ggml CUDA/HIP mul_mat dispatch
// at 1-4 columns, as the DSpark verify step runs them (batch-invariant products
// for widths above one). Checks that every column of a multi-column product is
// bit-identical to the one-column product, prints an output hash so two builds
// or two env settings can be compared, and times one graph of LAYERS distinct
// weight copies per shape (no cache reuse across nodes).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Shape { const char * name; int K; int N; int nch; ggml_type type = GGML_TYPE_Q8_0; };

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    const char * only = argc > 1 ? argv[1] : nullptr;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 10;
    const int dev = argc > 3 ? std::atoi(argv[3]) : 0;
    std::vector<Shape> shapes = {
        {"attn_q_a",    5120,   1280, 1},
        {"attn_q_b",    1280,  32768, 1},
        {"attn_kv",     5120,    512, 1},
        {"attn_out_a",  4096,   1024, 8},
        {"attn_out_b",  8192,   5120, 1},
        {"shexp_gu",    5120,   2304, 1},
        {"shexp_down",  2304,   5120, 1},
        {"output",      5120, 129280, 1},
        // non-Q8_0 dense products of the same verify step
        {"hc_fn_f16",  20480,     24, 1, GGML_TYPE_F16},
        {"compr_f16",   5120,    512, 1, GGML_TYPE_F16},
        {"router_f32",  5120,    384, 1, GGML_TYPE_F32},
        {"engram_f16",  6144,  25600, 1, GGML_TYPE_F16},
    };
    ggml_backend_t be = ggml_backend_cuda_init(dev);
    if (!be) { std::fprintf(stderr, "no cuda backend\n"); return 1; }
    int fails = 0;
    uint64_t all_hash = 1469598103934665603ull;
    for (const Shape & s : shapes) {
        if (only && std::strcmp(only, "all") && !std::strstr(s.name, only)) continue;
        const size_t wbytes = ggml_row_size(s.type, s.K) * s.N * s.nch;
        int layers = (int) ((640ull << 20) / wbytes);
        if (layers < 1) layers = 1;
        if (layers > 40) layers = 40;
        ggml_init_params ip = { ggml_tensor_overhead() * (layers*8 + 64) + ggml_graph_overhead_custom(1024, false) * 8, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        std::vector<ggml_tensor *> W(layers);
        for (int l = 0; l < layers; ++l) W[l] = ggml_new_tensor_3d(ctx, s.type, s.K, s.N, s.nch);
        ggml_tensor * X[5];
        for (int nc = 1; nc <= 4; ++nc) X[nc] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, nc, s.nch);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) { std::fprintf(stderr, "alloc failed for %s\n", s.name); return 1; }
        std::mt19937 rng(1234);
        {
            std::vector<uint8_t> host(ggml_nbytes(W[0]));
            const size_t nb = host.size() / ggml_type_size(GGML_TYPE_Q8_0);
            if (s.type != GGML_TYPE_Q8_0) {
                // one cheap pseudo-random fill shared by every copy
                const size_t n = (size_t) s.K * s.N * s.nch;
                uint32_t st = 12345u;
                for (size_t i = 0; i < n; ++i) {
                    st = st * 1664525u + 1013904223u;
                    const float v = ((int) (st >> 16) - 32768) * (0.04f / 32768.0f);
                    if (s.type == GGML_TYPE_F16) { const ggml_fp16_t h = ggml_fp32_to_fp16(v); std::memcpy(host.data() + 2*i, &h, 2); }
                    else { std::memcpy(host.data() + 4*i, &v, 4); }
                }
                for (int l = 0; l < layers; ++l) ggml_backend_tensor_set(W[l], host.data(), 0, host.size());
            }
            for (int l = 0; l < layers && s.type == GGML_TYPE_Q8_0; ++l) {
                for (size_t b = 0; b < nb; ++b) {
                    uint8_t * blk = host.data() + b*ggml_type_size(GGML_TYPE_Q8_0);
                    const float d = (float) ((rng() & 0xffff) / 65536.0 - 0.5) * 0.02f;
                    const ggml_fp16_t h = ggml_fp32_to_fp16(d);
                    std::memcpy(blk, &h, 2);
                    for (int k = 0; k < 32; ++k) blk[2 + k] = (uint8_t) (rng() & 0xff);
                }
                ggml_backend_tensor_set(W[l], host.data(), 0, host.size());
            }
        }
        {
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> x4((size_t) s.K * 4 * s.nch);
            for (float & v : x4) v = nd(rng);
            for (int nc = 1; nc <= 4; ++nc) {
                std::vector<float> xn((size_t) s.K * nc * s.nch);
                for (int c = 0; c < s.nch; ++c)
                    for (int j = 0; j < nc; ++j)
                        std::memcpy(&xn[((size_t) c*nc + j) * s.K], &x4[((size_t) c*4 + j) * s.K], s.K * sizeof(float));
                ggml_backend_tensor_set(X[nc], xn.data(), 0, xn.size() * sizeof(float));
            }
        }
        std::vector<std::vector<float>> col1(4);
        for (int nc = 1; nc <= 4; ++nc) {
            ggml_init_params gp = { ggml_tensor_overhead() * (layers*4 + 16) + ggml_graph_overhead_custom(1024, false), nullptr, true };
            ggml_context * gctx = ggml_init(gp);
            ggml_cgraph * gf = ggml_new_graph_custom(gctx, 1024, false);
            std::vector<ggml_tensor *> outs(layers);
            for (int l = 0; l < layers; ++l) {
                outs[l] = ggml_mul_mat(gctx, W[l], X[nc]);
                ggml_build_forward_expand(gf, outs[l]);
            }
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!ggml_gallocr_alloc_graph(ga, gf)) { std::fprintf(stderr, "gallocr failed\n"); return 1; }
            const bool prev = ggml_backend_cuda_set_mmvq_batch_invariant(nc > 1);
            for (int w = 0; w < 2; ++w) ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            const auto t1 = std::chrono::steady_clock::now();
            ggml_backend_cuda_set_mmvq_batch_invariant(prev);
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters / layers;
            std::vector<float> out((size_t) s.N * nc * s.nch);
            ggml_backend_tensor_get(outs[0], out.data(), 0, out.size() * sizeof(float));
            const uint64_t h = fnv(out.data(), out.size() * sizeof(float));
            all_hash = fnv(&h, sizeof(h), all_hash);
            // column identity vs the single-column product of the same activations
            size_t ndiff = 0;
            if (nc == 1) {
                // collect nc=1 results for all four columns: rerun with each column of X[4]
                for (int j = 0; j < 4; ++j) {
                    ggml_init_params cp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(64, false), nullptr, true };
                    ggml_context * cctx = ggml_init(cp);
                    ggml_cgraph * cg = ggml_new_graph_custom(cctx, 64, false);
                    ggml_tensor * xv = ggml_view_3d(cctx, X[4], s.K, 1, s.nch, X[4]->nb[1], X[4]->nb[2], j * X[4]->nb[1]);
                    ggml_tensor * xc = ggml_cont(cctx, xv);
                    ggml_tensor * o = ggml_mul_mat(cctx, W[0], xc);
                    ggml_build_forward_expand(cg, o);
                    ggml_gallocr_t g2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                    ggml_gallocr_alloc_graph(g2, cg);
                    ggml_backend_graph_compute(be, cg);
                    col1[j].resize((size_t) s.N * s.nch);
                    ggml_backend_tensor_get(o, col1[j].data(), 0, col1[j].size() * sizeof(float));
                    ggml_gallocr_free(g2);
                    ggml_free(cctx);
                }
            } else {
                for (int c = 0; c < s.nch; ++c)
                    for (int j = 0; j < nc; ++j)
                        for (int r = 0; r < s.N; ++r) {
                            const float a = out[((size_t) c*nc + j) * s.N + r];
                            const float b = col1[j][(size_t) c * s.N + r];
                            if (std::memcmp(&a, &b, 4)) ++ndiff;
                        }
                if (ndiff) ++fails;
            }
            std::printf("%-11s K=%5d N=%6d ch=%d nc=%d layers=%2d %8.1f us/mm %6.1f GB/s hash=%016llx %s\n",
                        s.name, s.K, s.N, s.nch, nc, layers, us, wbytes / us / 1e3,
                        (unsigned long long) h,
                        nc == 1 ? "" : (ndiff ? ("COLUMN-DIFF " + std::to_string(ndiff)).c_str() : "columns==1col"));
            std::fflush(stdout);
            ggml_gallocr_free(ga);
            ggml_free(gctx);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    std::printf("ALL_HASH %016llx fails=%d\n", (unsigned long long) all_hash, fails);
    ggml_backend_free(be);
    return fails ? 1 : 0;
}
