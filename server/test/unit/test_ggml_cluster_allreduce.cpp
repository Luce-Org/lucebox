// Contract test for ggml_cluster_allreduce (GGML_MOE_FUSED_CLUSTER_ALLREDUCE).
//
// The node hands a caller-registered callback the node's own buffer and the
// backend's stream. This test registers a stand-in for a two-rank group: the
// "other rank" contributes a fixed host vector, which the callback adds on the
// stream it was given. That pins what a real collective (RCCL/NCCL over the
// fabric) relies on:
//   - the callback sees the producer's value, i.e. it runs after the kernels
//     that wrote its input, and its sum is what the consumer reads;
//   - it is called with this backend's stream and the full element count;
//   - it runs on every compute: a graph holding a collective is not captured
//     by default, so the collective is never skipped by a graph replay.

#include "CppUnitTestFramework.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#define test_stream_t            hipStream_t
#define test_memcpy_async        hipMemcpyAsync
#define test_stream_sync         hipStreamSynchronize
#define test_memcpy_d2h          hipMemcpyDeviceToHost
#define test_memcpy_h2d          hipMemcpyHostToDevice
#define test_success             hipSuccess
#else
#include <cuda_runtime.h>
#define test_stream_t            cudaStream_t
#define test_memcpy_async        cudaMemcpyAsync
#define test_stream_sync         cudaStreamSynchronize
#define test_memcpy_d2h          cudaMemcpyDeviceToHost
#define test_memcpy_h2d          cudaMemcpyHostToDevice
#define test_success             cudaSuccess
#endif

#include <cmath>
#include <cstdio>
#include <vector>

using namespace CppUnitTestFramework;

namespace {

struct FakePeer {
    std::vector<float> contribution;   // the other rank's partial
    void *             expected_stream = nullptr;
    int                calls = 0;
    bool               stream_ok = true;
    bool               size_ok = true;
    bool               copy_ok = true;
};

// Eager-only stand-in: it synchronizes the stream, which a real collective
// must not do. That is fine here because the node is executed eagerly.
void fake_peer_allreduce(void * user, void * data, size_t n, void * stream) {
    auto * peer = static_cast<FakePeer *>(user);
    ++peer->calls;
    peer->stream_ok = peer->stream_ok && stream == peer->expected_stream;
    peer->size_ok = peer->size_ok && n == peer->contribution.size();
    if (n != peer->contribution.size()) return;

    auto s = static_cast<test_stream_t>(stream);
    std::vector<float> host(n);
    bool ok = test_memcpy_async(host.data(), data, n * sizeof(float),
                                test_memcpy_d2h, s) == test_success &&
              test_stream_sync(s) == test_success;
    for (size_t i = 0; ok && i < n; ++i) host[i] += peer->contribution[i];
    ok = ok &&
         test_memcpy_async(data, host.data(), n * sizeof(float),
                           test_memcpy_h2d, s) == test_success &&
         test_stream_sync(s) == test_success;
    peer->copy_ok = peer->copy_ok && ok;
}

struct GgmlClusterAllreduceFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};

}  // namespace

TEST_CASE(GgmlClusterAllreduceFixture, callback_sums_between_producer_and_consumer) {
    constexpr int n = 3 * 4096 + 17;
    constexpr int n_computes = 4;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        SKIP("CUDA/HIP backend unavailable");
    }

    FakePeer peer;
    peer.expected_stream = ggml_backend_cuda_get_stream(backend);
    peer.contribution.resize(n);
    for (int i = 0; i < n; ++i) peer.contribution[i] = 0.5f * (float)(i % 13) - 3.0f;

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    // producer -> collective -> consumer: out = (2x + peer) + x
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(x);
    ggml_tensor * partial = ggml_scale(ctx, x, 2.0f);
    ggml_tensor * summed = ggml_cluster_allreduce(ctx, partial, fake_peer_allreduce, &peer);
    ggml_tensor * out = ggml_add(ctx, summed, x);
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const bool allocated = alloc && ggml_gallocr_alloc_graph(alloc, graph);

    bool compute_ok = allocated;
    float max_err = 0.0f;
    for (int iter = 0; compute_ok && iter < n_computes; ++iter) {
        std::vector<float> x_data(n);
        for (int i = 0; i < n; ++i) x_data[i] = 0.01f * (float)((i * 7 + iter) % 101) - 0.5f;
        ggml_backend_tensor_set(x, x_data.data(), 0, n * sizeof(float));
        compute_ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (!compute_ok) break;
        std::vector<float> got(n);
        ggml_backend_tensor_get(out, got.data(), 0, n * sizeof(float));
        for (int i = 0; i < n; ++i) {
            const float want = 3.0f * x_data[i] + peer.contribution[i];
            max_err = std::fmax(max_err, std::fabs(got[i] - want));
        }
    }

    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    REQUIRE_TRUE(allocated);
    REQUIRE_TRUE(compute_ok);
    CHECK_TRUE(peer.copy_ok);
    CHECK_TRUE(peer.size_ok);
    CHECK_TRUE(peer.stream_ok);
    // Every compute reaches the collective: no replay skipped it.
    CHECK_EQUAL(peer.calls, n_computes);
    CHECK_TRUE(max_err <= 1e-5f);
}
