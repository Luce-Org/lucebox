// The stash -> slot state copy (a short park restoring the one-back state) is queued on the backend's compute stream. The next
// admission reads or writes the same slab through another stream (host reads and cudaStreamPerThread), so the copy must be complete
// when slot_state_stash_copy returns, or the new request starts from the previous session's recurrent state.
#include "CppUnitTestFramework.hpp"
#include "internal.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

using namespace luce::common;

namespace { struct SlotStateStashFenceFixture {}; }

TEST_CASE(SlotStateStashFenceFixture, the_stash_to_slot_copy_is_complete_when_it_returns) {
    if (ggml_backend_cuda_get_device_count() <= 0) SKIP("CUDA device unavailable");
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);
    constexpr int64_t kSlots = 2, kPerSlot = 64 << 20;   // 64 MiB per slot: the copy takes long enough to outlast a racing read
    ggml_init_params ip{};
    ip.mem_size = 4 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    TargetCache c;
    c.n_seq_slots = (int)kSlots;
    c.ssm_state.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kPerSlot, kSlots));
    ggml_backend_buffer_t live_buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    REQUIRE(live_buf != nullptr);
    SlotStateStash stash;
    REQUIRE(slot_state_stash_alloc(c, gpu, stash));

    std::vector<uint8_t> want((size_t)kPerSlot, 0xA5), stale((size_t)kPerSlot, 0x5A);
    uint8_t * pinned = nullptr;
    REQUIRE(cudaMallocHost((void **)&pinned, (size_t)kPerSlot) == cudaSuccess);
    int bad = 0;
    for (int rep = 0; rep < 8; ++rep) {
        ggml_backend_tensor_set(stash.ssm[0], want.data(), (size_t)kPerSlot, (size_t)kPerSlot);   // slot 1
        ggml_backend_tensor_set(c.ssm_state[0], stale.data(), (size_t)kPerSlot, (size_t)kPerSlot);
        REQUIRE(slot_state_stash_copy(gpu, c, stash, 1, false));
        // the next admission's view: another stream, no wait on the backend's
        const uint8_t * live = (const uint8_t *)c.ssm_state[0]->data + kPerSlot;
        REQUIRE(cudaMemcpyAsync(pinned, live, (size_t)kPerSlot, cudaMemcpyDeviceToHost, cudaStreamPerThread) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess);
        for (size_t i = 0; i < (size_t)kPerSlot; i += 4096) bad += pinned[i] != 0xA5;
    }
    CHECK(bad == 0);
    cudaFreeHost(pinned);
    slot_state_stash_free(stash);
    ggml_backend_buffer_free(live_buf);
    ggml_free(ctx);
    ggml_backend_free(gpu);
}
