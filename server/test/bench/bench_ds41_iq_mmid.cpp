// DS4.1 routed-expert decode matvec micro-benchmark: MUL_MAT_ID at decode width (1 token, 6 of E experts) for the IQ
// types the shipped plans use, on every GPU. Prints one JSON line per (device, type, surface) with the time per op
// and a hash of the outputs, so a kernel candidate can be checked for bit-identical results against the baseline
// build on the same inputs and timed without the model (the Lucebox full-decode measure needs the 150 GB GGUF).
//
// Weights are random bytes with a sane fp16 block scale: every bit pattern of an IQ2_XXS / IQ3_XXS block is a valid
// encoding (grid indices, sign/scale words), so all decode paths run. One graph chains R independent MUL_MAT_ID nodes
// whose expert ids rotate through E experts, so host launch overhead amortizes and weights stream from memory
// (E x matrix bytes exceeds the last-level cache) as in decode.
//
//   bench_ds41_iq_mmid [reps=50] [nodes=32] [experts=64]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

struct Case { ggml_type type; const char * surface; int k, n; };

static bool bench(ggml_backend_t backend, int device, const Case & c, int reps, int nodes, int experts) {
    const int used = 6;
    const size_t row = ggml_row_size(c.type, c.k), block = ggml_type_size(c.type), per_row = row / block;
    std::mt19937 rng(20260930u + unsigned(c.type) * 7919u + unsigned(c.k));
    std::vector<uint8_t> w(row * size_t(c.n) * experts);
    for (auto & b : w) b = uint8_t(rng());
    std::uniform_real_distribution<float> dscale(0.005f, 0.02f);
    for (size_t r = 0; r < size_t(c.n) * experts; ++r)
        for (size_t j = 0; j < per_row; ++j) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(dscale(rng));
            std::memcpy(w.data() + r * row + j * block, &d, sizeof d);  // IQ2_XXS / IQ3_XXS blocks start with fp16 d
        }
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(size_t(c.k) * nodes);
    for (auto & v : x) v = gauss(rng);
    std::vector<int32_t> ids(size_t(used) * nodes);
    for (int r = 0; r < nodes; ++r)
        for (int j = 0; j < used; ++j) ids[size_t(r) * used + j] = (r * used + j) % experts;

    ggml_init_params params{};
    params.mem_size = size_t(4 * nodes + 8) * ggml_tensor_overhead() + ggml_graph_overhead_custom(4 * nodes + 8, false);
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    auto a = ggml_new_tensor_3d(ctx, c.type, c.k, c.n, experts);
    std::vector<ggml_tensor *> xs, is, ys;
    auto graph = ggml_new_graph_custom(ctx, 4 * nodes + 8, false);
    for (int r = 0; r < nodes; ++r) {
        auto b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, 1, 1);        // one token, broadcast to its 6 experts
        auto id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, 1);
        auto y = ggml_mul_mat_id(ctx, a, b, id);
        xs.push_back(b); is.push_back(id); ys.push_back(y);
        ggml_build_forward_expand(graph, y);
    }
    if (!ggml_backend_supports_op(backend, ys[0])) {
        std::printf("{\"device\":%d,\"type\":\"%s\",\"surface\":\"%s\",\"error\":\"unsupported\"}\n", device, ggml_type_name(c.type), c.surface);
        ggml_free(ctx);
        return false;
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(a, w.data(), 0, w.size());
    for (int r = 0; r < nodes; ++r) {
        ggml_backend_tensor_set(xs[r], x.data() + size_t(r) * c.k, 0, size_t(c.k) * sizeof(float));
        ggml_backend_tensor_set(is[r], ids.data() + size_t(r) * used, 0, size_t(used) * sizeof(int32_t));
    }
    bool ok = true;
    for (int i = 0; i < 3 && ok; ++i) ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;  // warm-up
    ggml_backend_synchronize(backend);
    std::vector<double> times;
    for (int i = 0; i < reps && ok; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
        times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / nodes);
    }
    uint64_t h = 1469598103934665603ull;
    std::vector<float> out(size_t(c.n) * used);
    for (int r = 0; r < nodes && ok; ++r) {
        ggml_backend_tensor_get(ys[r], out.data(), 0, out.size() * sizeof(float));
        h = fnv1a(out.data(), out.size() * sizeof(float), h);
    }
    std::sort(times.begin(), times.end());
    const double med = times.empty() ? 0 : times[times.size() / 2], best = times.empty() ? 0 : times[0];
    std::printf("{\"device\":%d,\"name\":\"%s\",\"type\":\"%s\",\"surface\":\"%s\",\"k\":%d,\"n\":%d,\"experts\":%d,\"nodes\":%d,"
                "\"reps\":%d,\"us_median\":%.2f,\"us_best\":%.2f,\"hash\":\"%016" PRIx64 "\",\"ok\":%s}\n",
                device, ggml_backend_name(backend), ggml_type_name(c.type), c.surface, c.k, c.n, experts, nodes, reps,
                med, best, h, ok ? "true" : "false");
    std::fflush(stdout);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 50, nodes = argc > 2 ? std::atoi(argv[2]) : 32,
              experts = argc > 3 ? std::atoi(argv[3]) : 64;
    const Case cases[] = {
        {GGML_TYPE_IQ3_XXS, "down", 2304, 5120}, {GGML_TYPE_IQ3_XXS, "gate_up", 5120, 2304},
        {GGML_TYPE_IQ2_XXS, "down", 2304, 5120}, {GGML_TYPE_IQ2_XXS, "gate_up", 5120, 2304},
    };
    bool ok = true;
    for (int d = 0; d < ggml_backend_cuda_get_device_count(); ++d) {
        auto backend = ggml_backend_cuda_init(d);
        if (!backend) return 2;
        for (const auto & c : cases) ok = bench(backend, d, c, reps, nodes, experts) && ok;
        ggml_backend_free(backend);
    }
    return ok ? 0 : 1;
}
