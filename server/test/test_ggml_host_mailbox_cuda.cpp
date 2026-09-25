// Host mailbox ops (ggml_host_mailbox_post / ggml_host_mailbox_wait): a graph
// posts a tensor to host-mapped memory, a host thread answers, and the graph
// continues with the answer, over several launches of the same graph and with
// an answer that arrives late. Every device is tested.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct Mailbox {
    uint32_t step;
    uint32_t pad0[15];
    uint32_t posted;
    uint32_t pad1[15];
    uint32_t answered;
    uint32_t pad2[15];
    int32_t ids[64];
    int32_t answer[64];
};

constexpr int kRoutes = 6;
constexpr int kTokens = 2;
constexpr int kExperts = 8;

bool run_device(int device) {
    ggml_backend_t backend = ggml_backend_cuda_init(device);
    if (!backend) return false;
    Mailbox * box = nullptr;
    if (hipHostMalloc((void **) &box, sizeof(Mailbox),
                      hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent) != hipSuccess) {
        return false;
    }
    std::memset(box, 0, sizeof(Mailbox));

    ggml_init_params params{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kRoutes, kTokens);
    ggml_set_input(ids);
    ggml_tensor * post = ggml_host_mailbox_post(ctx, ids, &box->step, &box->posted, box->ids);
    ggml_tensor * answer = ggml_host_mailbox_wait(ctx, post, GGML_TYPE_I32, 1, kExperts, kTokens,
                                                  &box->step, &box->answered, box->answer);
    ggml_set_output(answer);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, answer);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(alloc, gf);

    for (int launch = 1; ok && launch <= 4; ++launch) {
        std::vector<int32_t> in(kRoutes * kTokens);
        for (int i = 0; i < (int) in.size(); ++i) in[(size_t) i] = launch * 100 + i;
        ggml_backend_tensor_set(ids, in.data(), 0, in.size() * sizeof(int32_t));
        __atomic_store_n(&box->step, (uint32_t) launch, __ATOMIC_RELEASE);
        // The resolver: wait for the post, check it, answer (late on launch 3).
        std::atomic<bool> post_ok{false};
        std::thread host([&] {
            while (__atomic_load_n(&box->posted, __ATOMIC_ACQUIRE) != (uint32_t) launch) {
                std::this_thread::yield();
            }
            post_ok = std::memcmp(box->ids, in.data(), in.size() * sizeof(int32_t)) == 0;
            if (launch == 3) std::this_thread::sleep_for(std::chrono::milliseconds(300));
            for (int i = 0; i < kExperts * kTokens; ++i) box->answer[i] = launch * 1000 + i;
            __atomic_store_n(&box->answered, (uint32_t) launch, __ATOMIC_RELEASE);
        });
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
        host.join();
        std::vector<int32_t> out(kExperts * kTokens);
        ggml_backend_tensor_get(answer, out.data(), 0, out.size() * sizeof(int32_t));
        for (int i = 0; ok && i < (int) out.size(); ++i) ok = out[(size_t) i] == launch * 1000 + i;
        ok = ok && post_ok;
        std::fprintf(stderr, "  device %d launch %d: %s\n", device, launch, ok ? "ok" : "FAIL");
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    (void) hipHostFree(box);
    ggml_backend_free(backend);
    return ok;
}

}  // namespace

int main() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n == 0) {
        std::fprintf(stderr, "no HIP device; skipping\n");
        return 77;
    }
    bool ok = true;
    for (int d = 0; d < n; ++d) ok = run_device(d) && ok;
    std::fprintf(stderr, ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
