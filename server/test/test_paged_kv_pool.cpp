// Unit tests for PagedKvPool (paged_kv_pool.h). No ggml, no GPU.

#define GENERATE_UNIT_TEST_MAIN
#include "CppUnitTestFramework.hpp"
#include "common/concurrency/paged_kv_pool.h"
#include "../src/common/paged_attention_config.h"

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace dflash::common;

namespace {
struct PagedKvPoolFixture {};
}

static PagedKvSequenceHandle acquire(PagedKvPool & pool,
                                     PagedKvRequestId request_id) {
    PagedKvSequenceHandle handle;
    if (pool.acquire(request_id, handle) != PagedKvStatus::Ok) {
        throw std::runtime_error("acquire failed");
    }
    return handle;
}

static PagedKvSequenceSnapshot sequence(PagedKvPool & pool,
                                        PagedKvSequenceHandle handle) {
    PagedKvSequenceSnapshot snapshot;
    if (pool.sequence(handle, snapshot) != PagedKvStatus::Ok) {
        throw std::runtime_error("sequence failed");
    }
    return snapshot;
}

static bool equals(const std::vector<uint32_t> & actual,
                   std::initializer_list<uint32_t> expected) {
    return actual == std::vector<uint32_t>(expected);
}

// True when `op` left the sequence's length and block table, and the pool's
// free-block count, unchanged.
template <typename Op>
static bool state_unchanged(PagedKvPool & pool,
                            PagedKvSequenceHandle handle,
                            Op && op) {
    const auto before = sequence(pool, handle);
    const uint32_t free_before = pool.free_block_count();
    op();
    const auto after = sequence(pool, handle);
    return after.kv_seq_len == before.kv_seq_len &&
           after.block_table == before.block_table &&
           after.reserved_block_count == before.reserved_block_count &&
           pool.free_block_count() == free_before;
}

// True when every handle-taking operation rejects `handle` as stale.
static bool all_ops_stale(PagedKvPool & pool, PagedKvSequenceHandle handle) {
    PagedKvSequenceSnapshot snapshot;
    uint32_t owned_blocks = 0;
    return pool.append(handle, 1).status == PagedKvStatus::StaleHandle &&
           pool.append(handle, 1, /*only_first_last_slots=*/true).status ==
               PagedKvStatus::StaleHandle &&
           pool.reserve_capacity(handle, 16) ==
               PagedKvStatus::StaleHandle &&
           pool.release(handle) == PagedKvStatus::StaleHandle &&
           pool.sequence(handle, snapshot) == PagedKvStatus::StaleHandle &&
           pool.owned_block_count(handle, owned_blocks) ==
               PagedKvStatus::StaleHandle;
}

TEST_CASE(PagedKvPoolFixture, block_boundaries) {
    const uint32_t lengths[] = {1, 15, 16, 17, 31, 32, 33};
    for (uint32_t length : lengths) {
        PagedKvPool pool(8, 2, 16);
        const auto handle = acquire(pool, 1000 + length);
        const auto append = pool.append(handle, length);
        CHECK(append.status == PagedKvStatus::Ok);
        CHECK(append.token_count == length);
        CHECK(append.write_slots.size() == length);

        const auto snapshot = sequence(pool, handle);
        const uint32_t expected_blocks = (length + 15) / 16;
        CHECK(snapshot.kv_seq_len == length);
        CHECK(snapshot.block_table.size() == expected_blocks);
        CHECK(pool.free_block_count() == 8 - expected_blocks);

        for (uint32_t i = 0; i < length; ++i) {
            const auto & slot = append.write_slots[i];
            CHECK(slot.logical_position == i);
            CHECK(slot.physical_block == i / 16);
            CHECK(slot.block_offset == i % 16);
            CHECK(slot.physical_token_index == i);
        }
    }
}

TEST_CASE(PagedKvPoolFixture, nondefault_block_size) {
    PagedKvPool pool(5, 3, 7);
    CHECK(pool.block_size() == 7);
    CHECK(pool.max_sequences() == 3);

    const auto handle = acquire(pool, 77);
    const auto append = pool.append(handle, 15);
    CHECK(append.status == PagedKvStatus::Ok);
    CHECK(equals(sequence(pool, handle).block_table, {0, 1, 2}));
    CHECK(append.write_slots[6].block_offset == 6);
    CHECK(append.write_slots[7].physical_block == 1);
    CHECK(append.write_slots[14].physical_token_index == 14);
}

TEST_CASE(PagedKvPoolFixture, reserved_acquire_feeds_chunked_append) {
    PagedKvPool pool(/*physical_block_count=*/6,
                     /*max_sequences=*/3, /*block_size=*/16);
    PagedKvSequenceHandle handle;
    CHECK(pool.acquire_reserved(77, /*token_capacity=*/40, handle) ==
          PagedKvStatus::Ok);
    CHECK(pool.active_sequence_count() == 1);
    CHECK(pool.free_block_count() == 3);

    auto snapshot = sequence(pool, handle);
    CHECK(snapshot.kv_seq_len == 0);
    CHECK(snapshot.block_table.empty());
    CHECK(snapshot.reserved_block_count == 3);
    uint32_t owned_blocks = 0;
    CHECK(pool.owned_block_count(handle, owned_blocks) == PagedKvStatus::Ok);
    CHECK(owned_blocks == 3);

    const auto first = pool.append(handle, 17);
    CHECK(first.status == PagedKvStatus::Ok);
    CHECK(equals(sequence(pool, handle).block_table, {0, 1}));
    CHECK(sequence(pool, handle).reserved_block_count == 1);
    CHECK(pool.owned_block_count(handle, owned_blocks) == PagedKvStatus::Ok);
    CHECK(owned_blocks == 3);
    // Consuming a reservation never changes globally available capacity.
    CHECK(pool.free_block_count() == 3);

    const auto tail = pool.append(handle, 23);
    CHECK(tail.status == PagedKvStatus::Ok);
    snapshot = sequence(pool, handle);
    CHECK(snapshot.kv_seq_len == 40);
    CHECK(equals(snapshot.block_table, {0, 1, 2}));
    CHECK(snapshot.reserved_block_count == 0);
    CHECK(pool.free_block_count() == 3);
    CHECK(pool.owned_block_count(handle, owned_blocks) == PagedKvStatus::Ok);
    CHECK(owned_blocks == 3);

    CHECK(pool.release(handle) == PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 6);
}

TEST_CASE(PagedKvPoolFixture, reserved_acquire_is_atomic_and_isolated) {
    PagedKvPool pool(/*physical_block_count=*/4,
                     /*max_sequences=*/3, /*block_size=*/16);
    PagedKvSequenceHandle first;
    CHECK(pool.acquire_reserved(1, /*token_capacity=*/48, first) ==
          PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 1);

    PagedKvSequenceHandle unchanged{77, 88};
    CHECK(pool.acquire_reserved(2, /*token_capacity=*/32, unchanged) ==
          PagedKvStatus::BlocksExhausted);
    CHECK(unchanged.slot == 77);
    CHECK(unchanged.generation == 88);
    CHECK(pool.active_sequence_count() == 1);
    CHECK(pool.free_block_count() == 1);

    // Unreserved work cannot steal the first sequence's promised pages.
    const auto second = acquire(pool, 2);
    CHECK(pool.append(second, 32).status == PagedKvStatus::BlocksExhausted);
    CHECK(pool.append(first, 48).status == PagedKvStatus::Ok);
    CHECK(equals(sequence(pool, first).block_table, {0, 1, 2}));

    CHECK(pool.release(first) == PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 4);
    CHECK(pool.release(second) == PagedKvStatus::Ok);

    // Retirement also returns reservations that prefill never consumed.
    PagedKvSequenceHandle unused;
    CHECK(pool.acquire_reserved(3, /*token_capacity=*/32, unused) ==
          PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 2);
    CHECK(sequence(pool, unused).reserved_block_count == 2);
    CHECK(pool.release(unused) == PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 4);
}

TEST_CASE(PagedKvPoolFixture, capacity_top_up_is_atomic_and_private) {
    PagedKvPool pool(/*physical_block_count=*/3,
                     /*max_sequences=*/2, /*block_size=*/16);
    const auto first = acquire(pool, 1);
    const auto second = acquire(pool, 2);
    CHECK(pool.append(first, 16).status == PagedKvStatus::Ok);
    CHECK(pool.append(second, 16).status == PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 1);

    // Asking for two more blocks fails atomically: neither the logical length
    // nor either ownership view changes.
    CHECK(state_unchanged(pool, first, [&] {
        CHECK(pool.reserve_capacity(first, /*token_capacity=*/48) ==
              PagedKvStatus::BlocksExhausted);
    }));

    // A one-page top-up succeeds without advancing logical state or exposing
    // a block-table entry.
    CHECK(pool.reserve_capacity(first, /*token_capacity=*/32) ==
          PagedKvStatus::Ok);
    auto first_state = sequence(pool, first);
    CHECK(first_state.kv_seq_len == 16);
    CHECK(equals(first_state.block_table, {0}));
    CHECK(first_state.reserved_block_count == 1);
    CHECK(pool.free_block_count() == 0);

    // The other sequence cannot steal the promised page; its own state stays
    // unchanged while the owner can consume the reservation.
    CHECK(state_unchanged(pool, second, [&] {
        CHECK(pool.append(second, 1).status ==
              PagedKvStatus::BlocksExhausted);
    }));
    CHECK(pool.append(first, 1).status == PagedKvStatus::Ok);
    first_state = sequence(pool, first);
    CHECK(equals(first_state.block_table, {0, 2}));
    CHECK(first_state.reserved_block_count == 0);

    // Top-ups are monotonic no-ops when the sequence already owns enough.
    CHECK(state_unchanged(pool, first, [&] {
        CHECK(pool.reserve_capacity(first, /*token_capacity=*/17) ==
              PagedKvStatus::Ok);
    }));
}

TEST_CASE(PagedKvPoolFixture, zero_token_append_is_a_no_op) {
    PagedKvPool pool(4, 2, 16);
    const auto handle = acquire(pool, 7);
    const auto append = pool.append(handle, 17);
    CHECK(append.status == PagedKvStatus::Ok);
    CHECK(equals(sequence(pool, handle).block_table, {0, 1}));

    CHECK(state_unchanged(pool, handle, [&] {
        const auto append_zero = pool.append(handle, 0);
        CHECK(append_zero.status == PagedKvStatus::Ok);
        CHECK(append_zero.write_slots.empty());
    }));
}

TEST_CASE(PagedKvPoolFixture, first_last_slots_append) {
    PagedKvPool pool(8, 1, 16);
    const auto handle = acquire(pool, 123);

    const auto prompt = pool.append(
        handle, 33, /*only_first_last_slots=*/true);
    CHECK(prompt.status == PagedKvStatus::Ok);
    CHECK(prompt.token_count == 33);
    CHECK(prompt.write_slots.empty());
    CHECK(prompt.first.logical_position == 0);
    CHECK(prompt.first.physical_token_index == 0);
    CHECK(prompt.last.logical_position == 32);
    CHECK(prompt.last.physical_block == 2);
    CHECK(prompt.last.block_offset == 0);
    CHECK(sequence(pool, handle).kv_seq_len == 33);

    const auto token = pool.append(
        handle, 1, /*only_first_last_slots=*/true);
    CHECK(token.status == PagedKvStatus::Ok);
    CHECK(token.token_count == 1);
    CHECK(token.first.logical_position == 33);
    CHECK(token.first.physical_token_index == 33);
    CHECK(token.last.logical_position == token.first.logical_position);

    const auto empty = pool.append(
        handle, 0, /*only_first_last_slots=*/true);
    CHECK(empty.status == PagedKvStatus::Ok);
    CHECK(empty.token_count == 0);
    CHECK(sequence(pool, handle).kv_seq_len == 34);
}

TEST_CASE(PagedKvPoolFixture, noncontiguous_reuse_and_isolation) {
    PagedKvPool pool(6, 3, 16);
    const auto first = acquire(pool, 101);
    const auto second = acquire(pool, 202);

    CHECK(pool.append(first, 17));
    CHECK(pool.append(second, 17));
    CHECK(equals(sequence(pool, first).block_table, {0, 1}));
    CHECK(equals(sequence(pool, second).block_table, {2, 3}));

    CHECK(pool.release(first) == PagedKvStatus::Ok);
    const auto second_more = pool.append(second, 16);
    CHECK(second_more.status == PagedKvStatus::Ok);
    CHECK(equals(sequence(pool, second).block_table, {2, 3, 0}));
    CHECK(second_more.write_slots.back().physical_block == 0);
    CHECK(second_more.write_slots.back().block_offset == 0);

    const auto third = acquire(pool, 303);
    CHECK(pool.append(third, 17));
    CHECK(equals(sequence(pool, third).block_table, {1, 4}));
    CHECK(equals(sequence(pool, second).block_table, {2, 3, 0}));

    CHECK(pool.release(third) == PagedKvStatus::Ok);
    CHECK(equals(sequence(pool, second).block_table, {2, 3, 0}));
    CHECK(pool.free_block_count() == 3);
}

TEST_CASE(PagedKvPoolFixture, exhaustion_rolls_back) {
    PagedKvPool pool(3, 2, 16);
    const auto first = acquire(pool, 1);
    const auto second = acquire(pool, 2);
    CHECK(pool.append(first, 17));
    CHECK(pool.append(second, 1));
    CHECK(pool.free_block_count() == 0);

    CHECK(state_unchanged(pool, first, [&] {
        const auto failed_append = pool.append(first, 16);
        CHECK(failed_append.status == PagedKvStatus::BlocksExhausted);
        CHECK(failed_append.write_slots.empty());
    }));

    const auto fits_existing_blocks = pool.append(first, 15);
    CHECK(fits_existing_blocks.status == PagedKvStatus::Ok);
    CHECK(sequence(pool, first).kv_seq_len == 32);

    PagedKvSequenceHandle unchanged{77, 88};
    CHECK(pool.acquire(3, unchanged) ==
          PagedKvStatus::SequenceSlotsExhausted);
    CHECK(unchanged.slot == 77);
    CHECK(unchanged.generation == 88);
    CHECK(pool.active_sequence_count() == 2);
}

TEST_CASE(PagedKvPoolFixture, request_identity_and_stale_handles) {
    PagedKvPool pool(4, 2, 16);
    const auto old_handle = acquire(pool, 9001);
    const auto other_handle = acquire(pool, 42);
    CHECK(old_handle.slot == 0);
    CHECK(other_handle.slot == 1);

    CHECK(pool.release(old_handle) == PagedKvStatus::Ok);

    const auto replacement = acquire(pool, 123456);
    CHECK(replacement.slot == old_handle.slot);
    CHECK(replacement.generation != old_handle.generation);
    CHECK(all_ops_stale(pool, old_handle));

    CHECK(pool.append(replacement, 1));
    pool.reset();
    CHECK(pool.active_sequence_count() == 0);
    CHECK(pool.free_block_count() == 4);
    CHECK(all_ops_stale(pool, replacement));

    const auto after_reset = acquire(pool, 555);
    CHECK(after_reset.slot == 0);
    CHECK(after_reset.generation != replacement.generation);
    CHECK(pool.append(after_reset, 1));
    CHECK(sequence(pool, after_reset).block_table.front() == 0);

    const PagedKvSequenceHandle out_of_range{
        std::numeric_limits<uint32_t>::max(), 1};
    CHECK(all_ops_stale(pool, out_of_range));
}

TEST_CASE(PagedKvPoolFixture, duplicate_request_is_transactional) {
    PagedKvPool pool(2, 2, 16);
    acquire(pool, 88);
    PagedKvSequenceHandle output{9, 10};
    CHECK(pool.acquire(88, output) == PagedKvStatus::DuplicateRequest);
    CHECK(output.slot == 9);
    CHECK(output.generation == 10);
    CHECK(pool.active_sequence_count() == 1);
}

static bool constructor_rejects(uint32_t physical_blocks,
                                uint32_t max_sequences,
                                uint32_t block_size) {
    try {
        PagedKvPool pool(physical_blocks, max_sequences, block_size);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

TEST_CASE(PagedKvPoolFixture, invalid_arguments) {
    CHECK(constructor_rejects(0, 1, 16));
    CHECK(constructor_rejects(
        0, std::numeric_limits<uint32_t>::max(), 16));
    CHECK(constructor_rejects(1, 0, 16));
    CHECK(constructor_rejects(1, 1, 0));
    CHECK(constructor_rejects(
        2, 1, std::numeric_limits<uint32_t>::max()));

    PagedKvPool pool(
        1, 1, std::numeric_limits<uint32_t>::max());
    const auto handle = acquire(pool, 99);
    CHECK(pool.append(handle, 1).status == PagedKvStatus::Ok);
    CHECK(state_unchanged(pool, handle, [&] {
        const auto overflow =
            pool.append(handle, std::numeric_limits<uint32_t>::max());
        CHECK(overflow.status == PagedKvStatus::InvalidArgument);
        CHECK(overflow.write_slots.empty());
    }));
}

TEST_CASE(PagedKvPoolFixture, auto_pool_sizing) {
    PagedKvAutoBudget budget;
    budget.free_bytes = 10'000;
    budget.fixed_cache_bytes = 1'000;
    budget.reserve_bytes = 1'000;
    budget.bytes_per_token = 10;
    // Memory could hold 800 tokens, but four 128-token logical contexts cap
    // the useful physical pool at 512.
    CHECK(paged_kv_auto_pool_tokens(128, 4, budget) == 512);

    budget.free_bytes = 5'000;
    // 300 raw tokens round down to 288 (18 whole blocks).
    CHECK(paged_kv_auto_pool_tokens(128, 4, budget) == 288);
    budget.free_bytes = 1'999;
    CHECK(paged_kv_auto_pool_tokens(128, 4, budget) == 0);
    budget.free_bytes = 5'000;
    budget.bytes_per_token = 0;
    CHECK(paged_kv_auto_pool_tokens(128, 4, budget) == 0);

    // A representative 24 GiB-card post-weight budget can keep one 32K
    // context plus oversubscription headroom without allocating 16 x 32K.
    budget.free_bytes = 10LL * 1024 * 1024 * 1024;
    budget.fixed_cache_bytes = 4LL * 1024 * 1024 * 1024;
    budget.reserve_bytes = 1536LL * 1024 * 1024;
    budget.bytes_per_token = 64 * 1024;
    const int64_t headline =
        paged_kv_auto_pool_tokens(32768, 16, budget);
    CHECK(headline == 73728);
    CHECK(headline >= 32768);
    CHECK(headline < 16LL * 32768);
}
