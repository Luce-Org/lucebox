// Same-slot KV suspension must preserve bytes and host state across physical
// page reuse. Existing pool tests exercise indices only, not payload copies.
#include "CppUnitTestFramework.hpp"
#include "common/concurrency/paged_kv_offload.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace {
using namespace dflash::common;
struct PagedKvOffloadFixture {};
#define OFFLOAD_CHECK(x) do { if (!(x)) throw std::runtime_error( \
    std::string("KV offload line ") + std::to_string(__LINE__) + ": " #x); } while (false)

// Two page-indexed planes plus a slot-indexed state tensor that never moves.
// Qwen planes are heads; DeepSeek planes are compressed/indexer tensors.
struct OffloadCache {
    PagedKvPool pool{8, 3, 4};
    SeqSlotManager slots{pool, 32};
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_context * ctx = nullptr;
    ggml_tensor * kv = nullptr;
    ggml_tensor * recurrent = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::unique_ptr<PagedKvOffload> offload;

    explicit OffloadCache(ggml_type type = GGML_TYPE_F16) {
        ggml_init_params params{3 * ggml_tensor_overhead(), nullptr, true};
        ctx = ggml_init(params);
        OFFLOAD_CHECK(ctx && backend);
        kv = ggml_new_tensor_3d(ctx, type, 32, 32, 2);
        recurrent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        OFFLOAD_CHECK(buffer);
        std::vector<uint8_t> data(ggml_nbytes(kv));
        for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t)(i * 17 + i / 37);
        ggml_backend_tensor_set(kv, data.data(), 0, data.size());
        std::vector<uint8_t> state(ggml_nbytes(recurrent), 0xa5);
        ggml_backend_tensor_set(recurrent, state.data(), 0, state.size());
        offload = std::make_unique<PagedKvOffload>(slots, pool, backend,
            std::vector<PagedKvTensor>{{kv, 0, 4 * kv->nb[1]},
                                       {kv, kv->nb[2], 4 * kv->nb[1]}});
    }
    ~OffloadCache() {
        offload.reset();
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    int admit(int id, int prompt = 6, bool complete = true) {
        SamplerCfg sampler;
        sampler.temp = 0.7f;
        sampler.seed = 1234;
        const auto a = slots.admit(id, std::vector<int32_t>(prompt, 7), sampler);
        OFFLOAD_CHECK(a.status == SeqEngine::AdmitResult::Status::admitted);
        OFFLOAD_CHECK(slots.append_prefill(a.slot, complete ? prompt : 2).ok);
        if (complete) slots.commit_prefill(a.slot);
        return a.slot;
    }
    std::vector<uint8_t> payload(int slot) {
        PagedKvSequenceSnapshot seq;
        OFFLOAD_CHECK(pool.sequence(slots.slot(slot).handle, seq) == PagedKvStatus::Ok);
        const size_t page = 4 * kv->nb[1];
        std::vector<uint8_t> data(2 * seq.block_table.size() * page);
        size_t at = 0;
        for (size_t head : {0u, 1u}) {
            for (uint32_t block : seq.block_table) {
                ggml_backend_tensor_get(kv, data.data() + at,
                    head * kv->nb[2] + block * page, page);
                at += page;
            }
        }
        return data;
    }
};
}

TEST_CASE(PagedKvOffloadFixture, test_quantized_payload_and_rng_survive_page_remapping) {
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        OffloadCache c(type);
        const int a = c.admit(1), b = c.admit(2);
        const auto original = c.payload(b);
        const auto rng = c.slots.slot(b).rng;
        const auto history = c.slots.slot(b).sample_history;
        const auto handle = c.slots.slot(b).handle;
        std::string error;
        OFFLOAD_CHECK(c.offload->suspend(b, original.size(), error));
        OFFLOAD_CHECK(c.offload->state(b).suspended);
        OFFLOAD_CHECK(c.offload->state(b).bytes == original.size());
        OFFLOAD_CHECK(c.pool.active_sequence_count() == 2); // slot remains owned
        OFFLOAD_CHECK(c.slots.decoding_count() == 1);
        OFFLOAD_CHECK(c.slots.admit(3, {7}, {}).status == SeqEngine::AdmitResult::Status::busy);
        OFFLOAD_CHECK(!c.slots.append_token(b, 1).ok);
        // A reuses B's returned pages and overwrites them while B is suspended.
        std::vector<int32_t> tokens(10, 9);
        OFFLOAD_CHECK(c.slots.append_tokens(a, tokens.data(), (int)tokens.size()).ok);
        c.slots.commit_step(a);
        std::vector<uint8_t> overwritten(ggml_nbytes(c.kv), 0x7e);
        ggml_backend_tensor_set(c.kv, overwritten.data(), 0, overwritten.size());
        c.slots.retire(a);
        std::vector<int32_t> blocks;
        OFFLOAD_CHECK(c.offload->restore(b, blocks, error));
        OFFLOAD_CHECK(blocks == std::vector<int32_t>({0, 1})); // originally 3, 4
        OFFLOAD_CHECK(c.payload(b) == original);
        OFFLOAD_CHECK(c.slots.slot(b).rng == rng);
        OFFLOAD_CHECK(c.slots.slot(b).sample_history == history);
        OFFLOAD_CHECK(c.slots.slot(b).cur_pos == 6);
        OFFLOAD_CHECK(c.slots.slot(b).sampler.temp == 0.7f);
        OFFLOAD_CHECK(c.slots.slot(b).handle.slot == handle.slot);
        OFFLOAD_CHECK(c.slots.slot(b).handle.generation == handle.generation);
        OFFLOAD_CHECK(!c.offload->state(b).suspended && c.offload->state(b).bytes == 0);
        std::vector<uint8_t> state(ggml_nbytes(c.recurrent));
        ggml_backend_tensor_get(c.recurrent, state.data(), 0, state.size());
        OFFLOAD_CHECK(std::all_of(state.begin(), state.end(), [](uint8_t b) { return b == 0xa5; }));
        OFFLOAD_CHECK(c.slots.append_token(b, 11).ok);
        c.slots.commit_step(b);
        OFFLOAD_CHECK(c.slots.slot(b).sample_history.back() == 11);
    }
}

TEST_CASE(PagedKvOffloadFixture, test_budget_and_restore_failure_preserve_the_request) {
    OffloadCache c;
    const int a = c.admit(1), b = c.admit(2);
    const auto original = c.payload(b);
    const auto free_before = c.pool.free_block_count();
    std::string error;
    OFFLOAD_CHECK(!c.offload->suspend(b, original.size() - 1, error));
    OFFLOAD_CHECK(error.find("budget") != std::string::npos);
    OFFLOAD_CHECK(c.pool.free_block_count() == free_before);
    OFFLOAD_CHECK(c.payload(b) == original && !c.offload->state(b).suspended);
    OFFLOAD_CHECK(c.offload->suspend(b, original.size(), error));
    std::vector<int32_t> tokens(26, 9);
    OFFLOAD_CHECK(c.slots.append_tokens(a, tokens.data(), (int)tokens.size()).ok);
    c.slots.commit_step(a);
    OFFLOAD_CHECK(c.pool.free_block_count() == 0);
    std::vector<int32_t> blocks;
    OFFLOAD_CHECK(!c.offload->restore(b, blocks, error) && error.empty());
    OFFLOAD_CHECK(c.offload->state(b).suspended && c.offload->state(b).bytes == original.size());
    OFFLOAD_CHECK(c.pool.free_block_count() == 0);
    c.slots.retire(a);
    OFFLOAD_CHECK(c.offload->restore(b, blocks, error));
    OFFLOAD_CHECK(c.payload(b) == original);
}

TEST_CASE(PagedKvOffloadFixture, test_partial_prefill_and_cancellation_keep_slot_ownership) {
    OffloadCache c;
    const int a = c.admit(1, 6, false);
    std::string error;
    OFFLOAD_CHECK(c.offload->suspend(a, 4096, error));
    OFFLOAD_CHECK(!c.slots.is_prefilling(a) && c.pool.free_block_count() == 8);
    std::vector<int32_t> blocks;
    OFFLOAD_CHECK(c.offload->restore(a, blocks, error));
    OFFLOAD_CHECK(c.slots.is_prefilling(a) && c.slots.slot(a).cur_pos == 2);
    OFFLOAD_CHECK(c.slots.append_prefill(a, 4).ok);
    c.slots.commit_prefill(a);
    OFFLOAD_CHECK(c.offload->suspend(a, 4096, error));
    const auto old_handle = c.slots.slot(a).handle;
    c.offload->discard(a);
    c.slots.retire(a);
    OFFLOAD_CHECK(c.offload->state(a).bytes == 0 && !c.offload->state(a).suspended);
    OFFLOAD_CHECK(c.pool.free_block_count() == 8 && c.pool.active_sequence_count() == 0);
    const int replacement = c.admit(2);
    OFFLOAD_CHECK(replacement == a);
    OFFLOAD_CHECK(c.slots.slot(replacement).handle.generation != old_handle.generation);
    OFFLOAD_CHECK(c.pool.clear(old_handle) == PagedKvStatus::StaleHandle);
}

TEST_CASE(PagedKvOffloadFixture, test_invalid_layout_and_staged_decode_do_not_release_blocks) {
    OffloadCache c;
    const int slot = c.admit(1);
    PagedKvOffload invalid(c.slots, c.pool, c.backend,
                          {{c.kv, ggml_nbytes(c.kv), 1}});
    const auto free_before = c.pool.free_block_count();
    std::string error;
    OFFLOAD_CHECK(!invalid.suspend(slot, 4096, error));
    OFFLOAD_CHECK(error.find("layout") != std::string::npos);
    OFFLOAD_CHECK(c.pool.free_block_count() == free_before);
    OFFLOAD_CHECK(c.slots.append_token(slot, 42).ok);
    OFFLOAD_CHECK(!c.offload->suspend(slot, 4096, error));
    OFFLOAD_CHECK(c.slots.slot(slot).staged_tokens == std::vector<int32_t>({42}));
    c.slots.commit_step(slot);
    OFFLOAD_CHECK(c.offload->suspend(slot, 4096, error));
}

// Fixed-budget tests cannot expose granting every loaded model the full
// host headroom, overriding explicit limits, or undersizing repeated victims.
TEST_CASE(PagedKvOffloadFixture, test_auto_budget_bounds_shared_headroom_and_payload) {
    const size_t gib = size_t(1) << 30;
    OFFLOAD_CHECK(auto_kv_offload_budget(20 * gib, 16 * gib, 0, 2) == 2 * gib);
    OFFLOAD_CHECK(auto_kv_offload_budget(gib, 16 * gib, 0, 2) == gib);
    OFFLOAD_CHECK(auto_kv_offload_budget(20 * gib, 16 * gib, 2 * gib, 2) == gib);
    OFFLOAD_CHECK(auto_kv_offload_budget(20 * gib, 16 * gib, 5 * gib, 2) == 0);
    OFFLOAD_CHECK(auto_kv_offload_budget(gib, 0, 0, 1) == 0);
    OFFLOAD_CHECK(auto_kv_offload_budget(gib, 16 * gib, 0, 0) == 0);
    OFFLOAD_CHECK(auto_kv_offload_budget(gib, 16 * gib,
        (std::numeric_limits<size_t>::max)(), 2) == 0);
    OffloadCache c;
    // Repeated suspension can preserve more than one pool: after saving B,
    // A and C may grow into B's pages before C also needs to be saved.
    OFFLOAD_CHECK(c.offload->capacity() == 2 * ggml_nbytes(c.kv));
    OFFLOAD_CHECK(auto_kv_offload_budget(c.offload->capacity(), 16 * gib, 0, 1)
                  == c.offload->capacity());
}

#if defined(__linux__)
// Real-host tests cannot demonstrate a constrained container or an unlimited
// child inside a constrained parent. Exercise those allocation boundaries.
TEST_CASE(PagedKvOffloadFixture, test_memory_probe_respects_container_and_parent_limits) {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() /
        ("luce-kv-memory-" + std::to_string(getpid()));
    struct Cleanup {
        fs::path root;
        ~Cleanup() { std::error_code ec; fs::remove_all(root, ec); }
    } cleanup{root};
    fs::create_directories(root / "proc/self");
    fs::create_directories(root / "cgroup/parent/child");
    const auto proc = (root / "proc").string(), group = (root / "cgroup").string();
    auto probe = [&] { return available_kv_offload_memory(proc.c_str(), group.c_str()); };
    std::ofstream(root / "proc/meminfo") << "MemAvailable: 8192 kB\n";
    std::ofstream(root / "proc/self/cgroup") << "0::/\n";
    std::ofstream(root / "cgroup/cgroup.controllers") << "memory\n";
    OFFLOAD_CHECK(probe() == 8192 * 1024); // host hierarchy root, no memory.max
    std::ofstream(root / "cgroup/memory.max") << "4096\n";
    std::ofstream(root / "cgroup/memory.current") << "1024\n";
    OFFLOAD_CHECK(probe() == 3072); // container namespace root still constrained
    std::ofstream(root / "proc/self/cgroup") << "0::/parent/child\n";
    std::ofstream(root / "cgroup/parent/memory.max") << "2048\n";
    std::ofstream(root / "cgroup/parent/memory.current") << "512\n";
    std::ofstream(root / "cgroup/parent/child/memory.max") << "max\n";
    OFFLOAD_CHECK(probe() == 1536); // unlimited child cannot bypass parent
    std::ofstream(root / "cgroup/parent/memory.current") << "4096\n";
    OFFLOAD_CHECK(probe() == 0); // pressure must not underflow
    fs::remove(root / "cgroup/parent/memory.current");
    OFFLOAD_CHECK(!probe()); // incomplete accounting cannot enable auto
    std::ofstream(root / "proc/self/cgroup") << "5:memory:/parent\n";
    OFFLOAD_CHECK(!probe()); // unsupported controller disables automatic sizing
}
#endif

#undef OFFLOAD_CHECK
