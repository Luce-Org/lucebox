#include "CppUnitTestFramework.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
struct CudaCopyBatchFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};
}  // namespace

// ggml_backend_cuda_copy_batch_async: more descriptors than one launch holds,
// aligned and unaligned ends, sizes from 1 byte to several KB, and an empty
// descriptor. Every destination byte must hold its source byte and every
// byte between destinations must stay untouched.
TEST_CASE(CudaCopyBatchFixture, descriptor_batch) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);

    constexpr int64_t kBytes = 1 << 20;
    ggml_init_params ip{};
    ip.mem_size = 2 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);
    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, kBytes);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, kBytes);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    REQUIRE(buf != nullptr);

    std::vector<uint8_t> src_host(kBytes);
    for (int64_t i = 0; i < kBytes; ++i) {
        src_host[i] = (uint8_t) ((i * 131 + 7) & 0xff);
    }
    std::vector<uint8_t> expected(kBytes, 0xEE);
    ggml_backend_tensor_set(src, src_host.data(), 0, kBytes);
    ggml_backend_tensor_set(dst, expected.data(), 0, kBytes);

    std::vector<ggml_cuda_copy_desc> descs;
    int64_t src_off = 3;
    int64_t dst_off = 5;
    for (int i = 0; i < 101; ++i) {
        const int64_t n = i == 50 ? 0 : 1 + (int64_t) (i * 97) % 9000;
        // Every fourth copy starts 16-byte aligned on both ends.
        if (i % 4 == 0) {
            src_off = (src_off + 15) & ~int64_t(15);
            dst_off = (dst_off + 15) & ~int64_t(15);
        }
        REQUIRE(src_off + n <= kBytes);
        REQUIRE(dst_off + n <= kBytes);
        descs.push_back({(const uint8_t *) src->data + src_off,
                         (uint8_t *) dst->data + dst_off, (size_t) n});
        for (int64_t j = 0; j < n; ++j) {
            expected[dst_off + j] = src_host[src_off + j];
        }
        src_off += n + 1;
        dst_off += n + 3;   // gaps between destinations must stay 0xEE
    }

    ggml_backend_cuda_copy_batch_async(gpu, descs.data(), (int) descs.size());
    ggml_backend_synchronize(gpu);

    std::vector<uint8_t> got(kBytes);
    ggml_backend_tensor_get(dst, got.data(), 0, kBytes);
    int64_t first_bad = -1;
    for (int64_t i = 0; i < kBytes && first_bad < 0; ++i) {
        if (got[i] != expected[i]) first_bad = i;
    }
    if (first_bad >= 0) {
        std::fprintf(stderr, "[cuda-copy-batch] byte %lld: got %u want %u\n",
                     (long long) first_bad, got[first_bad], expected[first_bad]);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    REQUIRE(first_bad == -1);
}

namespace {
constexpr int64_t kGraphN = 4099;

std::vector<float> graph_pattern(float base, float step) {
    std::vector<float> v(kGraphN);
    for (int64_t i = 0; i < kGraphN; ++i) v[i] = base + step * (float) i;
    return v;
}

int64_t graph_mismatches(ggml_tensor * t, const std::vector<float> & want) {
    std::vector<float> got(want.size());
    ggml_backend_tensor_get(t, got.data(), 0, sizeof(float) * got.size());
    int64_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
    return bad;
}
}  // namespace

// Graph pass: every dependency must end a run so copies keep their order.
//   1 a -> x   2 b -> y   3 x -> z   read after write (3 reads 1's dst)
//   4 c -> w   5 d -> c              write after read (5 writes 4's src)
//   6 e -> u   7 f -> u              write after write (7 writes 6's dst)
//   8 h -> t
// The pass issues exactly four runs: [1 2] [3 4] [5 6] [7 8]. Dropping any
// one of the three independence checks merges two of them (three runs), so
// the count pins each check even when a race would not show in the data.
TEST_CASE(CudaCopyBatchFixture, graph_copy_dependencies) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);
    ggml_init_params gp{};
    gp.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead();
    gp.no_alloc = true;
    ggml_context * gctx = ggml_init(gp);
    REQUIRE(gctx != nullptr);
    auto tensor = [&] { return ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kGraphN); };
    ggml_tensor * a = tensor(); ggml_tensor * b = tensor(); ggml_tensor * c = tensor();
    ggml_tensor * d = tensor(); ggml_tensor * e = tensor(); ggml_tensor * f = tensor();
    ggml_tensor * h = tensor();
    ggml_tensor * x = tensor(); ggml_tensor * y = tensor(); ggml_tensor * z = tensor();
    ggml_tensor * w = tensor(); ggml_tensor * u = tensor(); ggml_tensor * t = tensor();
    ggml_cgraph * gf = ggml_new_graph(gctx);
    const std::pair<ggml_tensor *, ggml_tensor *> order[] = {
        {a, x}, {b, y}, {x, z}, {c, w}, {d, c}, {e, u}, {f, u}, {h, t}};
    for (const auto & [src, dst] : order) {
        ggml_build_forward_expand(gf, ggml_cpy(gctx, src, dst));
    }
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
    REQUIRE(gbuf != nullptr);
    const auto va = graph_pattern(1.0f, 1.0f), vb = graph_pattern(0.0f, -2.0f);
    const auto vc = graph_pattern(0.0f, 0.5f), vd = graph_pattern(7.0f, -1.0f);
    const auto ve = graph_pattern(2.0f, 0.25f), vf = graph_pattern(9.0f, 3.0f);
    const auto vh = graph_pattern(-4.0f, 0.75f), zero = graph_pattern(0.0f, 0.0f);
    const std::pair<ggml_tensor *, const std::vector<float> *> inputs[] = {
        {a, &va}, {b, &vb}, {c, &vc}, {d, &vd}, {e, &ve}, {f, &vf}, {h, &vh},
        {x, &zero}, {y, &zero}, {z, &zero}, {w, &zero}, {u, &zero}, {t, &zero}};
    for (const auto & [tensor_in, values] : inputs) {
        ggml_backend_tensor_set(tensor_in, values->data(), 0, sizeof(float) * kGraphN);
    }
    const size_t runs_before = ggml_backend_cuda_get_copy_batch_run_count();
    const bool computed = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    const size_t runs = ggml_backend_cuda_get_copy_batch_run_count() - runs_before;
    const std::pair<ggml_tensor *, const std::vector<float> *> checks[] = {
        {x, &va}, {y, &vb}, {z, &va}, {w, &vc}, {c, &vd}, {u, &vf}, {t, &vh}};
    int64_t bad = 0;
    for (const auto & [tensor_out, want] : checks) bad += graph_mismatches(tensor_out, *want);
    ggml_backend_buffer_free(gbuf);
    ggml_free(gctx);
    ggml_backend_free(gpu);
    const bool disabled = std::getenv("GGML_CUDA_DISABLE_COPY_BATCH") != nullptr;
    std::printf("[cuda-copy-batch] dependency graph: %lld mismatched values, %zu batched runs%s\n",
                (long long) bad, runs, disabled ? " (batching disabled)" : "");
    REQUIRE(computed);
    REQUIRE(bad == 0);
    REQUIRE(runs == (disabled ? 0u : 4u));
}

// A run of 60 independent small copies is longer than one launch.
TEST_CASE(CudaCopyBatchFixture, graph_copy_long_run) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);
    constexpr int kRun = 60;
    ggml_init_params gp{};
    gp.mem_size = (4 * kRun) * ggml_tensor_overhead() + ggml_graph_overhead();
    gp.no_alloc = true;
    ggml_context * gctx = ggml_init(gp);
    REQUIRE(gctx != nullptr);
    ggml_tensor * run_src[kRun];
    ggml_tensor * run_dst[kRun];
    ggml_cgraph * gf = ggml_new_graph(gctx);
    for (int r = 0; r < kRun; ++r) {
        run_src[r] = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 33 + r);
        run_dst[r] = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 33 + r);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, run_src[r], run_dst[r]));
    }
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
    REQUIRE(gbuf != nullptr);
    for (int r = 0; r < kRun; ++r) {
        std::vector<float> v(33 + r, 3.0f + (float) r);
        std::vector<float> zero(33 + r, 0.0f);
        ggml_backend_tensor_set(run_src[r], v.data(), 0, sizeof(float) * v.size());
        ggml_backend_tensor_set(run_dst[r], zero.data(), 0, sizeof(float) * zero.size());
    }
    const size_t runs_before = ggml_backend_cuda_get_copy_batch_run_count();
    const bool computed = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    const size_t runs = ggml_backend_cuda_get_copy_batch_run_count() - runs_before;
    int64_t bad = 0;
    for (int r = 0; r < kRun; ++r) {
        std::vector<float> v(33 + r);
        ggml_backend_tensor_get(run_dst[r], v.data(), 0, sizeof(float) * v.size());
        for (float value : v) bad += value != 3.0f + (float) r;
    }
    ggml_backend_buffer_free(gbuf);
    ggml_free(gctx);
    ggml_backend_free(gpu);
    const bool disabled = std::getenv("GGML_CUDA_DISABLE_COPY_BATCH") != nullptr;
    std::printf("[cuda-copy-batch] long run: %lld mismatched values, %zu batched runs%s\n",
                (long long) bad, runs, disabled ? " (batching disabled)" : "");
    REQUIRE(computed);
    REQUIRE(bad == 0);
    if (disabled) {
        REQUIRE(runs == 0);
    } else {
        REQUIRE(runs >= 2);
    }
}

// ggml_backend_tensor_copy between two devices, the last call having selected the destination (as the scheduler's
// copy of a graph input to another device does): the destination must hold the source.
TEST_CASE(CudaCopyBatchFixture, cross_device_tensor_copy) {
    if (ggml_backend_cuda_get_device_count() < 2) {
        SKIP("needs two CUDA/HIP devices");
    }
    ggml_backend_t gpus[2] = { ggml_backend_cuda_init(0), ggml_backend_cuda_init(1) };
    REQUIRE(gpus[0] != nullptr);
    REQUIRE(gpus[1] != nullptr);
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx[2] = { ggml_init(ip), ggml_init(ip) };
    ggml_tensor * src = ggml_new_tensor_1d(ctx[0], GGML_TYPE_I32, 4);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx[1], GGML_TYPE_I32, 4);
    ggml_backend_buffer_t buf[2] = { ggml_backend_alloc_ctx_tensors(ctx[0], gpus[0]),
                                     ggml_backend_alloc_ctx_tensors(ctx[1], gpus[1]) };
    REQUIRE(buf[0] != nullptr);
    REQUIRE(buf[1] != nullptr);
    const int32_t rows[4] = { 19, 20, 21, 22 }, zero[4] = {};
    ggml_backend_tensor_set(src, rows, 0, sizeof(rows));
    ggml_backend_tensor_set(dst, zero, 0, sizeof(zero));   // selects the destination device
    ggml_backend_tensor_copy(src, dst);
    int32_t got[4] = {};
    ggml_backend_tensor_get(dst, got, 0, sizeof(got));
    std::printf("[cuda-copy-batch] cross-device copy: %d %d %d %d\n", got[0], got[1], got[2], got[3]);
    for (int i = 0; i < 2; ++i) {
        ggml_backend_buffer_free(buf[i]);
        ggml_free(ctx[i]);
        ggml_backend_free(gpus[i]);
    }
    for (int i = 0; i < 4; ++i) REQUIRE(got[i] == rows[i]);
}

// Contiguous concatenations that are plain copies (concat.cu concat_as_copies): along dim 2 chained as a verify's
// attention rows are, along dim 1 over three samples (ne2 == 1), along dim 3, and one past the one-launch size (the
// runtime's copies). Every output element must be its operand's, bit for bit.
TEST_CASE(CudaCopyBatchFixture, concat_copies) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);
    ggml_init_params gp{};
    gp.mem_size = 32 * ggml_tensor_overhead() + ggml_graph_overhead();
    gp.no_alloc = true;
    ggml_context * gctx = ggml_init(gp);
    REQUIRE(gctx != nullptr);
    struct Concat { ggml_tensor * a, * b, * out; int dim; };
    std::vector<Concat> cases;
    auto concat = [&](ggml_tensor * a, ggml_tensor * b, int dim) {
        ggml_tensor * out = ggml_concat(gctx, a, b, dim);
        cases.push_back({ a, b, out, dim });
        return out;
    };
    ggml_tensor * rows[3];
    for (ggml_tensor *& r : rows) r = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 256, 24, 1);
    auto t4 = [&](ggml_type type, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
        return ggml_new_tensor_4d(gctx, type, n0, n1, n2, n3);
    };
    concat(concat(rows[0], rows[1], 2), rows[2], 2);
    concat(t4(GGML_TYPE_I32, 33, 5, 1, 3), t4(GGML_TYPE_I32, 33, 7, 1, 3), 1);
    concat(t4(GGML_TYPE_F32, 100, 3, 2, 1), t4(GGML_TYPE_F32, 100, 3, 2, 2), 3);
    concat(t4(GGML_TYPE_F32, 1024, 256, 1, 1), t4(GGML_TYPE_F32, 1024, 256, 1, 1), 2);   // 2 MiB: runtime copies
    ggml_cgraph * gf = ggml_new_graph(gctx);
    for (const Concat & c : cases) ggml_build_forward_expand(gf, c.out);
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
    REQUIRE(gbuf != nullptr);

    // Every operand gets its own 32-bit pattern; the chained rows' middle result is checked as an output too.
    std::vector<std::vector<uint32_t>> host(cases.size() * 2);
    uint32_t seed = 11u;
    auto fill = [&](ggml_tensor * t, std::vector<uint32_t> & v) {
        if (!v.empty()) return;
        v.resize((size_t) ggml_nelements(t));
        for (uint32_t & x : v) { seed = seed * 1664525u + 1013904223u; x = seed >> 4; }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        if (cases[i].a->op == GGML_OP_NONE) fill(cases[i].a, host[2 * i]);
        fill(cases[i].b, host[2 * i + 1]);
    }
    const bool computed = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;

    int64_t bad = 0;
    std::vector<std::vector<uint32_t>> got(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        const Concat & c = cases[i];
        got[i].resize((size_t) ggml_nelements(c.out));
        ggml_backend_tensor_get(c.out, got[i].data(), 0, ggml_nbytes(c.out));
        // The first operand's values: its pattern, or the previous case's output when chained.
        const std::vector<uint32_t> & va = c.a->op == GGML_OP_NONE ? host[2 * i] : got[i - 1];
        const std::vector<uint32_t> & vb = host[2 * i + 1];
        const int64_t * ne = c.out->ne;
        for (int64_t i3 = 0; i3 < ne[3]; ++i3) for (int64_t i2 = 0; i2 < ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < ne[1]; ++i1) for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
            int64_t idx[4] = { i0, i1, i2, i3 };
            const bool first = idx[c.dim] < c.a->ne[c.dim];
            const ggml_tensor * src = first ? c.a : c.b;
            if (!first) idx[c.dim] -= c.a->ne[c.dim];
            const int64_t at = ((idx[3] * src->ne[2] + idx[2]) * src->ne[1] + idx[1]) * src->ne[0] + idx[0];
            const int64_t out_at = ((i3 * ne[2] + i2) * ne[1] + i1) * ne[0] + i0;
            bad += got[i][(size_t) out_at] != (first ? va : vb)[(size_t) at];
        }
    }
    ggml_backend_buffer_free(gbuf);
    ggml_free(gctx);
    ggml_backend_free(gpu);
    std::printf("[cuda-copy-batch] concat copies: %lld mismatched values in %zu concatenations\n", (long long) bad,
                cases.size());
    REQUIRE(computed);
    REQUIRE(bad == 0);
}

namespace {
// GGML_CUDA_PEER_COPY_KERNEL=0 or GGML_(CUDA_)BATCH_PEER_COPIES=0: hop copies take no peer copy kernel, whatever the
// graph, so none is collected for a later flush and no graph carries a tail.
bool hop_copy_kernel_off_by_env() {
    const auto zero = [](const char * v) { return v != nullptr && std::strcmp(v, "0") == 0; };
    const char * batch = std::getenv("GGML_BATCH_PEER_COPIES");
    if (batch == nullptr || *batch == '\0') {
        batch = std::getenv("GGML_CUDA_BATCH_PEER_COPIES");
    }
    return zero(std::getenv("GGML_CUDA_PEER_COPY_KERNEL")) || zero(batch);
}

bool is_rocm(ggml_backend_t backend) {
    return std::string(ggml_backend_name(backend)).rfind("ROCm", 0) == 0;
}

struct SplitRun {
    int64_t bad      = 0;
    int     bad_step = -1;
    bool    computed = true;
    size_t  tails    = 0;
    bool    rocm     = false;
};
}  // namespace

// A graph split over two devices by the scheduler and computed again with new inputs each step, as split decode runs
// (parallel copies, captured graphs replayed). With the copy tail on for both backends
// (ggml_backend_cuda_set_copy_tail), once a producer's graph has been seen right before a flush of its hop copies, the
// copies and the hop's event record ride at the end of that graph (ggml_cuda_tail_realize); with it off, as for every
// backend that does not set it, no graph carries them. Every step must read that step's values on both devices either
// way. Hop copies take the peer copy kernel (the only path a tail can carry) under the Qwen3.8 split profile, so the
// test runs in it: on ROCm the tail must have carried copies when on, unless the environment turned the kernel or the
// batching off (=0), which must hold under that profile too.
TEST_CASE(CudaCopyBatchFixture, split_graph_tail_copies) {
    if (ggml_backend_cuda_get_device_count() < 2) {
        SKIP("needs two CUDA/HIP devices");
    }
    // Two chains cross between devices 0 and 1 at every hop (two copies per hop), with exact float arithmetic:
    // a' = 2a + b, b' = b + 1 on device hop % 2. The scheduler computes the same graph with new inputs each step.
    auto run_split_graph = [&](bool copy_tail) {
        ggml_backend_t gpus[2] = { ggml_backend_cuda_init(0), ggml_backend_cuda_init(1) };
        REQUIRE(gpus[0] != nullptr);
        REQUIRE(gpus[1] != nullptr);
        ggml_backend_cuda_set_copy_tail(gpus[0], copy_tail);
        ggml_backend_cuda_set_copy_tail(gpus[1], copy_tail);
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);   // the scheduler's last
        REQUIRE(cpu != nullptr);

        constexpr int64_t n     = 1536;
        constexpr int     kHops = 6;
        constexpr int     kSteps = 24;
        ggml_init_params ip{};
        ip.mem_size = 128 * ggml_tensor_overhead() + ggml_graph_overhead();
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        REQUIRE(ctx != nullptr);
        ggml_tensor * a0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_tensor * b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_input(a0);
        ggml_set_input(b0);
        ggml_tensor * a = a0;
        ggml_tensor * b = b0;
        std::vector<std::pair<ggml_tensor *, int>> placed;
        for (int h = 0; h < kHops; ++h) {
            ggml_tensor * a2 = ggml_scale(ctx, a, 2.0f);
            ggml_tensor * an = ggml_add(ctx, a2, b);
            ggml_tensor * bn = ggml_scale_bias(ctx, b, 1.0f, 1.0f);
            for (ggml_tensor * t : { a2, an, bn }) placed.push_back({ t, h % 2 });
            a = an;
            b = bn;
        }
        ggml_tensor * out = ggml_add(ctx, a, b);
        placed.push_back({ out, 0 });
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        ggml_backend_t backends[3] = { gpus[0], gpus[1], cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 3, 256, true, false);
        REQUIRE(sched != nullptr);
        ggml_backend_sched_set_tensor_backend(sched, a0, gpus[0]);
        ggml_backend_sched_set_tensor_backend(sched, b0, gpus[0]);
        for (const auto & [t, dev] : placed) ggml_backend_sched_set_tensor_backend(sched, t, gpus[dev]);
        REQUIRE(ggml_backend_sched_alloc_graph(sched, gf));

        SplitRun r;
        const size_t tails_before = ggml_backend_cuda_get_copy_tail_count();
        std::vector<float> va(n), vb(n), got(n);
        for (int step = 0; step < kSteps && r.computed; ++step) {
            for (int64_t i = 0; i < n; ++i) {
                va[i] = (float) ((i * 7 + step * 13) % 1000);
                vb[i] = (float) ((i * 3 + step * 5) % 500);
            }
            ggml_backend_tensor_set(a0, va.data(), 0, sizeof(float) * n);
            ggml_backend_tensor_set(b0, vb.data(), 0, sizeof(float) * n);
            r.computed = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(out, got.data(), 0, sizeof(float) * n);
            for (int64_t i = 0; i < n; ++i) {
                float x = va[i], y = vb[i];
                for (int h = 0; h < kHops; ++h) {
                    x = 2.0f * x + y;
                    y = y + 1.0f;
                }
                if (got[i] != x + y) {
                    ++r.bad;
                    if (r.bad_step < 0) r.bad_step = step;
                }
            }
        }
        r.tails = ggml_backend_cuda_get_copy_tail_count() - tails_before;
        r.rocm = is_rocm(gpus[0]);

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_backend_free(gpus[0]);
        ggml_backend_free(gpus[1]);
        return r;
    };

    const ggml_cuda_qwen4exp_profile profile_before =
        ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);
    const SplitRun off = run_split_graph(false);
    const SplitRun on  = run_split_graph(true);
    ggml_backend_cuda_set_qwen4exp_profile(profile_before);
    const bool kernel_off = hop_copy_kernel_off_by_env();
    for (const SplitRun * r : { &off, &on }) {
        std::printf("[cuda-copy-batch] split graph, copy tail %s: %lld mismatched values (first at step %d), "
                    "%zu tail launches%s%s\n", r == &on ? "on" : "off", (long long) r->bad, r->bad_step, r->tails,
                    r->rocm ? " (ROCm)" : "", kernel_off ? " (hop copy kernel off by env)" : "");
    }
    REQUIRE(off.computed);
    REQUIRE(on.computed);
    REQUIRE(off.bad == 0);
    REQUIRE(on.bad == 0);
    REQUIRE(off.tails == 0);
    if (on.rocm && !kernel_off) {
        REQUIRE(on.tails > 0);
    } else {
        REQUIRE(on.tails == 0);
    }
}

// A hop copy collected for a later flush (batched kernel copies, as under the Qwen3.8 split profile) is published by
// ggml_backend_cuda_enter_stream, which settles the stream for work queued on it directly, and not by
// ggml_backend_cuda_get_stream, which only looks the stream up. The collected copies belong to the calling thread, so
// another thread waits for both devices without publishing them: the destination then holds the source only if the
// copy was published.
TEST_CASE(CudaCopyBatchFixture, enter_stream_publishes_peer_copies) {
    if (ggml_backend_cuda_get_device_count() < 2) {
        SKIP("needs two CUDA/HIP devices");
    }
    if (hop_copy_kernel_off_by_env()) {
        SKIP("hop copies are not collected with the peer copy kernel or the batching off");
    }
    ggml_backend_t gpus[2] = { ggml_backend_cuda_init(0), ggml_backend_cuda_init(1) };
    REQUIRE(gpus[0] != nullptr);
    REQUIRE(gpus[1] != nullptr);
    if (!is_rocm(gpus[0])) {
        ggml_backend_free(gpus[0]);
        ggml_backend_free(gpus[1]);
        SKIP("hop copies are collected on ROCm only");
    }
    const ggml_cuda_qwen4exp_profile profile_before =
        ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);
    constexpr int64_t n = 4099;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx[2] = { ggml_init(ip), ggml_init(ip) };
    ggml_tensor * src = ggml_new_tensor_1d(ctx[0], GGML_TYPE_I32, n);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx[1], GGML_TYPE_I32, n);
    ggml_backend_buffer_t buf[2] = { ggml_backend_alloc_ctx_tensors(ctx[0], gpus[0]),
                                     ggml_backend_alloc_ctx_tensors(ctx[1], gpus[1]) };
    REQUIRE(buf[0] != nullptr);
    REQUIRE(buf[1] != nullptr);

    std::vector<int32_t> rows(n), zero(n, 0), got(n);
    // Fresh source rows and a zeroed destination (blocking writes), then the hop copy, collected and not launched.
    auto collect = [&](int32_t base) {
        for (int64_t i = 0; i < n; ++i) rows[i] = base + (int32_t) i;
        ggml_backend_tensor_set(src, rows.data(), 0, ggml_nbytes(src));
        ggml_backend_tensor_set(dst, zero.data(), 0, ggml_nbytes(dst));
        ggml_backend_tensor_copy_async(gpus[0], gpus[1], src, dst);
    };
    auto copied_after_other_thread_waits = [&] {
        std::thread([&] {
            ggml_backend_synchronize(gpus[0]);
            ggml_backend_synchronize(gpus[1]);
        }).join();
        ggml_backend_tensor_get(dst, got.data(), 0, ggml_nbytes(dst));
        return got == rows;
    };

    collect(100);
    const bool got_stream = ggml_backend_cuda_get_stream(gpus[1]) != nullptr;
    const bool after_get = copied_after_other_thread_waits();
    ggml_backend_synchronize(gpus[1]);   // publishes the copy and waits for it
    ggml_backend_tensor_get(dst, got.data(), 0, ggml_nbytes(dst));
    const bool after_sync = got == rows;

    collect(7000);
    const bool entered_stream = ggml_backend_cuda_enter_stream(gpus[1]) != nullptr;
    const bool after_enter = copied_after_other_thread_waits();

    ggml_backend_synchronize(gpus[0]);
    ggml_backend_synchronize(gpus[1]);
    for (int i = 0; i < 2; ++i) {
        ggml_backend_buffer_free(buf[i]);
        ggml_free(ctx[i]);
        ggml_backend_free(gpus[i]);
    }
    ggml_backend_cuda_set_qwen4exp_profile(profile_before);
    std::printf("[cuda-copy-batch] collected hop copy: published by get_stream %d, by synchronize %d, "
                "by enter_stream %d\n", after_get, after_sync, after_enter);
    REQUIRE(got_stream);
    REQUIRE(entered_stream);
    REQUIRE(!after_get);
    REQUIRE(after_sync);
    REQUIRE(after_enter);
}

// A strided hop copy (a verify's top-k expert ids are the first columns of the router's sorted rows) joins the peer
// copy batch like a contiguous one when both ends share its layout: it is collected, not run, until the destination
// is synchronized. It then moves the bytes cudaMemcpyPeerAsync would, the ggml_nbytes span from the view's first
// byte: every viewed element is its source's and no byte before or after the span is written.
TEST_CASE(CudaCopyBatchFixture, strided_hop_copy_joins_batch) {
    if (ggml_backend_cuda_get_device_count() < 2) {
        SKIP("needs two CUDA/HIP devices");
    }
    if (hop_copy_kernel_off_by_env()) {
        SKIP("hop copies are not collected with the peer copy kernel or the batching off");
    }
    ggml_backend_t gpus[2] = { ggml_backend_cuda_init(0), ggml_backend_cuda_init(1) };
    REQUIRE(gpus[0] != nullptr);
    REQUIRE(gpus[1] != nullptr);
    if (!is_rocm(gpus[0])) {
        ggml_backend_free(gpus[0]);
        ggml_backend_free(gpus[1]);
        SKIP("hop copies are collected on ROCm only");
    }
    const ggml_cuda_qwen4exp_profile profile_before =
        ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);
    // Columns 1 to 3 of 5 rows of 8: a strided view whose first byte is not 16-byte aligned.
    constexpr int64_t n_cols = 8, n_rows = 5, first = 1, k = 3, n = n_cols * n_rows;
    ggml_init_params ip{};
    ip.mem_size = 2 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx[2] = { ggml_init(ip), ggml_init(ip) };
    ggml_tensor * rows[2];
    ggml_tensor * ids[2];
    for (int d = 0; d < 2; ++d) {
        rows[d] = ggml_new_tensor_2d(ctx[d], GGML_TYPE_I32, n_cols, n_rows);
        ids[d]  = ggml_view_2d(ctx[d], rows[d], k, n_rows, rows[d]->nb[1], first * sizeof(int32_t));
    }
    ggml_backend_buffer_t buf[2] = { ggml_backend_alloc_ctx_tensors(ctx[0], gpus[0]),
                                     ggml_backend_alloc_ctx_tensors(ctx[1], gpus[1]) };
    REQUIRE(buf[0] != nullptr);
    REQUIRE(buf[1] != nullptr);

    std::vector<int32_t> src(n), zero(n, 0), before(n), after(n);
    for (int64_t i = 0; i < n; ++i) src[i] = 100 + (int32_t) i;
    ggml_backend_tensor_set(rows[0], src.data(), 0, ggml_nbytes(rows[0]));
    ggml_backend_tensor_set(rows[1], zero.data(), 0, ggml_nbytes(rows[1]));
    const bool strided = !ggml_is_contiguous(ids[0]) && !ggml_is_contiguous(ids[1]);
    const int64_t span_end = first + (ggml_nbytes(ids[1]) / sizeof(int32_t));   // one past the span's last element
    ggml_backend_tensor_copy_async(gpus[0], gpus[1], ids[0], ids[1]);
    // Another thread's waits flush only that thread's collected copies.
    std::thread([&] {
        ggml_backend_synchronize(gpus[0]);
        ggml_backend_synchronize(gpus[1]);
    }).join();
    ggml_backend_tensor_get(rows[1], before.data(), 0, ggml_nbytes(rows[1]));
    ggml_backend_synchronize(gpus[1]);   // publishes the copy and waits for it
    ggml_backend_tensor_get(rows[1], after.data(), 0, ggml_nbytes(rows[1]));

    ggml_backend_synchronize(gpus[0]);
    for (int i = 0; i < 2; ++i) {
        ggml_backend_buffer_free(buf[i]);
        ggml_free(ctx[i]);
        ggml_backend_free(gpus[i]);
    }
    ggml_backend_cuda_set_qwen4exp_profile(profile_before);

    const bool collected = before == zero;
    int64_t bad_view = 0, bad_outside = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t col = i % n_cols;
        if (col >= first && col < first + k) {
            bad_view += after[i] != src[i];
        } else if (i < first || i >= span_end) {
            bad_outside += after[i] != 0;
        }
    }
    std::printf("[cuda-copy-batch] strided hop copy: collected %d, viewed elements wrong %lld, "
                "elements outside the span written %lld\n",
                collected, (long long) bad_view, (long long) bad_outside);
    REQUIRE(strided);
    REQUIRE(collected);
    REQUIRE(bad_view == 0);
    REQUIRE(bad_outside == 0);
}

// A strided hop through the scheduler, as a verify's top-k ids cross to the expert device: device 0 computes the
// rows, device 1 reads the first k columns of each (the scheduler copies that view with its strides). With the copy
// tail on, the copy joins the hop batch and rides at the end of device 0's graph; every step must read that step's
// rows.
TEST_CASE(CudaCopyBatchFixture, split_graph_strided_hop) {
    if (ggml_backend_cuda_get_device_count() < 2) {
        SKIP("needs two CUDA/HIP devices");
    }
    if (hop_copy_kernel_off_by_env()) {
        SKIP("hop copies are not collected with the peer copy kernel or the batching off");
    }
    ggml_backend_t gpus[2] = { ggml_backend_cuda_init(0), ggml_backend_cuda_init(1) };
    REQUIRE(gpus[0] != nullptr);
    REQUIRE(gpus[1] != nullptr);
    if (!is_rocm(gpus[0])) {
        ggml_backend_free(gpus[0]);
        ggml_backend_free(gpus[1]);
        SKIP("hop copies are collected on ROCm only");
    }
    const ggml_cuda_qwen4exp_profile profile_before =
        ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);
    ggml_backend_cuda_set_copy_tail(gpus[0], true);
    ggml_backend_cuda_set_copy_tail(gpus[1], true);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);   // the scheduler's last
    REQUIRE(cpu != nullptr);

    constexpr int64_t n_cols = 64, n_rows = 4, k = 8;
    constexpr int     kSteps = 16;
    ggml_init_params ip{};
    ip.mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_rows);
    ggml_set_input(x);
    ggml_tensor * rows   = ggml_scale(ctx, x, 2.0f);
    ggml_tensor * ids    = ggml_view_2d(ctx, rows, k, n_rows, rows->nb[1], 0);
    ggml_tensor * picked = ggml_cont(ctx, ids);
    ggml_tensor * out    = ggml_scale_bias(ctx, picked, 1.0f, 1.0f);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backends[3] = { gpus[0], gpus[1], cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 3, 64, true, false);
    REQUIRE(sched != nullptr);
    ggml_backend_sched_set_tensor_backend(sched, x, gpus[0]);
    ggml_backend_sched_set_tensor_backend(sched, rows, gpus[0]);
    ggml_backend_sched_set_tensor_backend(sched, ids, gpus[0]);      // so device 1's input is the strided view
    ggml_backend_sched_set_tensor_backend(sched, picked, gpus[1]);
    ggml_backend_sched_set_tensor_backend(sched, out, gpus[1]);
    REQUIRE(ggml_backend_sched_alloc_graph(sched, gf));

    const bool strided = !ggml_is_contiguous(ids);
    const size_t tails_before = ggml_backend_cuda_get_copy_tail_count();
    std::vector<float> vx(n_cols * n_rows), got(k * n_rows);
    int64_t bad = 0;
    bool computed = true;
    for (int step = 0; step < kSteps && computed; ++step) {
        for (int64_t i = 0; i < n_cols * n_rows; ++i) vx[i] = (float) ((i * 7 + step * 13) % 1000);
        ggml_backend_tensor_set(x, vx.data(), 0, ggml_nbytes(x));
        computed = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
        for (int64_t r = 0; r < n_rows; ++r) {
            for (int64_t c = 0; c < k; ++c) {
                bad += got[r * k + c] != 2.0f * vx[r * n_cols + c] + 1.0f;
            }
        }
    }
    const size_t tails = ggml_backend_cuda_get_copy_tail_count() - tails_before;

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(gpus[0]);
    ggml_backend_free(gpus[1]);
    ggml_backend_cuda_set_qwen4exp_profile(profile_before);
    std::printf("[cuda-copy-batch] strided hop through the scheduler: %lld mismatched values, %zu tail launches\n",
                (long long) bad, tails);
    REQUIRE(strided);
    REQUIRE(computed);
    REQUIRE(bad == 0);
    REQUIRE(tails > 0);
}
