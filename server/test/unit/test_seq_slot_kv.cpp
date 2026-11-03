// SeqSlotManager allocates through the SlotKv interface. PoolSlotKv forwards
// every call to the PagedKvPool and changes nothing: the same rows and block
// deltas as the PagedKvPool & constructor, and any other SlotKv sees every call.
#include "CppUnitTestFramework.hpp"
#include "common/concurrency/seq_slot_manager.h"
#include "common/concurrency/slot_kv.h"

#include <vector>

using namespace luce::common;

namespace {
struct SeqSlotKvFixture {};

// Forwards to a PoolSlotKv and counts what the manager asks for.
struct CountingKv : SlotKv {
    explicit CountingKv(PagedKvPool & p) : inner(p) {}
    PoolSlotKv inner;
    int acquires = 0, appends = 0, releases = 0;

    uint32_t block_size() const override { return inner.block_size(); }
    uint32_t physical_block_count() const override { return inner.physical_block_count(); }
    uint32_t max_sequences() const override { return inner.max_sequences(); }
    uint32_t free_block_count() const override { return inner.free_block_count(); }
    PagedKvStatus acquire_reserved(PagedKvRequestId id, uint32_t cap, PagedKvSequenceHandle & h) override {
        acquires++; return inner.acquire_reserved(id, cap, h);
    }
    PagedKvStatus reserve_capacity(PagedKvSequenceHandle h, uint32_t cap) override {
        return inner.reserve_capacity(h, cap);
    }
    PagedKvAppendResult append(PagedKvSequenceHandle h, uint32_t n, bool first_last) override {
        appends++; return inner.append(h, n, first_last);
    }
    PagedKvStatus rollback_append(PagedKvSequenceHandle h, uint32_t n) override {
        return inner.rollback_append(h, n);
    }
    PagedKvStatus clear(PagedKvSequenceHandle h) override { return inner.clear(h); }
    PagedKvStatus release(PagedKvSequenceHandle h) override { releases++; return inner.release(h); }
    PagedKvStatus owned_block_count(PagedKvSequenceHandle h, uint32_t & n) const override {
        return inner.owned_block_count(h, n);
    }
};

struct Trace { std::vector<int64_t> rows; std::vector<int32_t> blocks; std::vector<int64_t> step_rows; };

template <class Mgr>
Trace drive(Mgr & mgr) {
    Trace r;
    const std::vector<int32_t> prompt(40, 1);
    auto a = mgr.admit(1, prompt, SamplerCfg{});
    if (a.status != SeqEngine::AdmitResult::Status::admitted) return r;
    auto c1 = mgr.append_prefill(a.slot, 24);
    auto c2 = mgr.append_prefill(a.slot, 16);
    r.rows = c1.rows; r.rows.insert(r.rows.end(), c2.rows.begin(), c2.rows.end());
    r.blocks = c1.new_blocks; r.blocks.insert(r.blocks.end(), c2.new_blocks.begin(), c2.new_blocks.end());
    mgr.commit_prefill(a.slot);
    auto s = mgr.append_token(a.slot, 7);
    r.step_rows.push_back(s.physical_row);
    mgr.commit_step(a.slot);
    mgr.retire(a.slot);
    return r;
}
}  // namespace

TEST_CASE(SeqSlotKvFixture, the_pool_adapter_is_the_legacy_behaviour) {
    PagedKvPool p1(8, 2, 16), p2(8, 2, 16);
    SeqSlotManager legacy(p1, 64);
    PoolSlotKv adapter(p2);
    SeqSlotManager viakv(adapter, 64);
    const Trace a = drive(legacy), b = drive(viakv);
    CHECK(!a.rows.empty());
    CHECK(a.rows == b.rows);
    CHECK(a.blocks == b.blocks);
    CHECK(a.step_rows == b.step_rows);
    CHECK(p2.free_block_count() == 8);        // retire returned everything
}

TEST_CASE(SeqSlotKvFixture, the_manager_asks_the_slotkv_not_the_pool) {
    PagedKvPool pool(8, 2, 16);
    CountingKv kv(pool);
    {
        SeqSlotManager mgr(kv, 64);
        const Trace r = drive(mgr);
        CHECK(!r.rows.empty());
    }
    CHECK(kv.acquires >= 1);
    CHECK(kv.appends >= 3);                   // two prefill chunks and one decode row
    CHECK(kv.releases >= 1);
    CHECK(pool.free_block_count() == 8);
}

TEST_CASE(SeqSlotKvFixture, the_pool_adapter_forwards_every_call_to_the_pool) {
    PagedKvPool pool(8, 2, 16);
    PoolSlotKv kv(pool);
    CHECK(kv.block_size() == 16);
    CHECK(kv.physical_block_count() == 8);
    CHECK(kv.max_sequences() == 2);
    CHECK(kv.free_block_count() == 8);
    PagedKvSequenceHandle h;
    REQUIRE(kv.acquire_reserved(1, 40, h) == PagedKvStatus::Ok);
    CHECK(pool.active_sequence_count() == 1);
    uint32_t owned = 99;
    REQUIRE(kv.owned_block_count(h, owned) == PagedKvStatus::Ok);
    CHECK(owned == 3);                                   // ceil(40 / 16), reserved at admission
    CHECK(kv.free_block_count() == 5);
    CHECK(kv.append(h, 40, false).status == PagedKvStatus::Ok);
    CHECK(kv.rollback_append(h, 16) == PagedKvStatus::Ok);
    CHECK(kv.reserve_capacity(h, 64) == PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 4);                 // capacity grew to 4 blocks
    CHECK(kv.clear(h) == PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 8);
    CHECK(kv.release(h) == PagedKvStatus::Ok);
    CHECK(pool.active_sequence_count() == 0);
}
