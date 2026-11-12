// KvFlashArena: the page stream and completion shared by all slot
// pagers. CPU tensors and the no-tensor path; the stream-ordering race on a
// device is test_kvflash_pager_cuda_race.cpp.
#include "CppUnitTestFramework.hpp"
#include "kvflash_slot_world.h"
#ifdef KVFLASH_HAS_ASYNC_DMA
#include "ggml-cuda.h"
#endif

#include <memory>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashArenaFixture {};

constexpr int kN = 3, kQ = 8, kChunk = 64;

using Slot = KvFlashTestSlot;

KvFlashConfig cfg() { return kvflash_test_config(4, kChunk); }

struct World {
    KvFlashTestArena arena;
    KvFlashTestSlots s;
    int waits = 0;
    std::vector<int> pending_seen;           // sum of pending blocks seen at each host wait
    World() {
        arena.set_host_wait([this] {
            waits++;
            int p = 0;
            for (auto & x : s) p += x->pending_free_blocks();
            pending_seen.push_back(p);
        });
        s = kvflash_test_slots(kN, kQ, cfg(), &arena);
        for (auto & x : s)
            for (int pos = 0; pos < kQ * kChunk; pos++) if (x->slot_for(pos) < 0) throw std::runtime_error("fill");
    }
};
}  // namespace

TEST_CASE(KvflashArenaFixture, completion_waits_once_for_all_slots_and_only_then_reclaims) {
    World w;
    REQUIRE(w.s[0]->page_out(3));
    REQUIRE(w.s[1]->page_out(4));
    REQUIRE(w.s[1]->page_out(5));
    CHECK(w.s[1]->slot_for(kQ * kChunk) < 0);           // in flight: not allocatable before completion
    CHECK(w.arena.complete_all());
    REQUIRE(w.waits == 1);                              // one wait covers every slot
    CHECK(w.pending_seen[0] == 3);                      // the wait ran BEFORE any frees were released
    for (auto & x : w.s) CHECK(x->pending_free_blocks() == 0);
    CHECK(w.s[1]->slot_for(kQ * kChunk) >= 0);          // reclaimed
}

TEST_CASE(KvflashArenaFixture, completion_with_nothing_in_flight_does_not_wait) {
    World w;
    CHECK(!w.arena.complete_all());
    CHECK(w.waits == 0);
    REQUIRE(w.s[2]->page_out(3));
    CHECK(w.arena.complete_all());
    CHECK(w.waits == 1);
    CHECK(!w.arena.complete_all());                     // already settled
    CHECK(w.waits == 1);
}

TEST_CASE(KvflashArenaFixture, a_removed_pager_is_not_touched_and_an_attached_one_cannot_join) {
    World w;
    REQUIRE(w.s[0]->page_out(3));
    w.arena.remove(*w.s[0]);
    CHECK(!w.arena.complete_all());                     // the removed slot's frees are its own business now
    CHECK(w.s[0]->pending_free_blocks() == 1);
    Slot late;
    REQUIRE(late.attach_slice(0, kN, kQ, cfg(), {}, {}));
    CHECK(!w.arena.add(late));                          // must join before attach: the stream is chosen at attach
    CHECK(!w.arena.add(*w.s[1]));                       // already a member
}

TEST_CASE(KvflashArenaFixture, members_share_the_arena_stream_and_batch_submit) {
#ifdef KVFLASH_HAS_ASYNC_DMA
    if (ggml_backend_cuda_get_device_count() < 1) SKIP("no CUDA device: the shared stream needs one");
    World w;
    REQUIRE(w.arena.stream() != nullptr);
    for (auto & x : w.s) CHECK(x->page_stream() == w.arena.stream());
    // Destroying a member must not destroy the shared stream.
    w.s[0].reset();
    CHECK(cudaStreamQuery((cudaStream_t)w.arena.stream()) == cudaSuccess);
#endif
}
