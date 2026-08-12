// Unit tests for PagedKvPool (paged_kv_pool.h). No ggml, no GPU.

#define GENERATE_UNIT_TEST_MAIN
#include "CppUnitTestFramework.hpp"
#include "../src/common/paged_kv_pool.h"

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
           pool.free_block_count() == free_before;
}

// True when every handle-taking operation rejects `handle` as stale.
static bool all_ops_stale(PagedKvPool & pool, PagedKvSequenceHandle handle) {
    PagedKvSequenceSnapshot snapshot;
    return pool.append(handle, 1).status == PagedKvStatus::StaleHandle &&
           pool.append(handle, 1, /*only_first_last_slots=*/true).status ==
               PagedKvStatus::StaleHandle &&
           pool.release(handle) == PagedKvStatus::StaleHandle &&
           pool.sequence(handle, snapshot) == PagedKvStatus::StaleHandle;
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
