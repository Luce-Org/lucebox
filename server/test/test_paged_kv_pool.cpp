#include "../src/common/paged_kv_pool.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace dflash::common;

static int failures = 0;
static int checks = 0;

#define CHECK(expr) do {                                                    \
    ++checks;                                                               \
    if (!(expr)) {                                                          \
        ++failures;                                                         \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    }                                                                       \
} while (0)

static PagedKvSequenceHandle acquire(PagedKvPool & pool,
                                     PagedKvRequestId request_id) {
    PagedKvSequenceHandle handle;
    CHECK(pool.acquire(request_id, handle) == PagedKvStatus::Ok);
    return handle;
}

static PagedKvSequenceSnapshot sequence(PagedKvPool & pool,
                                        PagedKvSequenceHandle handle) {
    PagedKvSequenceSnapshot snapshot;
    CHECK(pool.sequence(handle, snapshot) == PagedKvStatus::Ok);
    return snapshot;
}

static bool equals(const std::vector<uint32_t> & actual,
                   std::initializer_list<uint32_t> expected) {
    return actual == std::vector<uint32_t>(expected);
}

static void test_block_boundaries() {
    const uint32_t lengths[] = {1, 15, 16, 17, 31, 32, 33};
    for (uint32_t length : lengths) {
        PagedKvPool pool(8, 2);
        const auto handle = acquire(pool, 1000 + length);
        const auto append = pool.append(handle, length);
        CHECK(append.status == PagedKvStatus::Ok);
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

static void test_nondefault_block_size() {
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

static void test_reserve_then_append() {
    PagedKvPool pool(4, 2);
    const auto handle = acquire(pool, 7);

    CHECK(pool.reserve(handle, 17) == PagedKvStatus::Ok);
    auto snapshot = sequence(pool, handle);
    CHECK(snapshot.kv_seq_len == 0);
    CHECK(equals(snapshot.block_table, {0, 1}));
    CHECK(pool.free_block_count() == 2);

    const auto append = pool.append(handle, 17);
    CHECK(append.status == PagedKvStatus::Ok);
    CHECK(pool.free_block_count() == 2);
    snapshot = sequence(pool, handle);
    CHECK(snapshot.kv_seq_len == 17);
    CHECK(equals(snapshot.block_table, {0, 1}));

    CHECK(pool.reserve(handle, 16) == PagedKvStatus::Ok);
    CHECK(equals(sequence(pool, handle).block_table, {0, 1}));

    const auto before_zero = sequence(pool, handle);
    const auto append_zero = pool.append(handle, 0);
    CHECK(append_zero.status == PagedKvStatus::Ok);
    CHECK(append_zero.write_slots.empty());
    const auto after_zero = sequence(pool, handle);
    CHECK(after_zero.kv_seq_len == before_zero.kv_seq_len);
    CHECK(after_zero.block_table == before_zero.block_table);
}

static void test_noncontiguous_reuse_and_isolation() {
    PagedKvPool pool(6, 3);
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

    CHECK(pool.cancel(303) == PagedKvStatus::Ok);
    CHECK(equals(sequence(pool, second).block_table, {2, 3, 0}));
    CHECK(pool.free_block_count() == 3);
}

static void test_exhaustion_rolls_back() {
    PagedKvPool pool(3, 2);
    const auto first = acquire(pool, 1);
    const auto second = acquire(pool, 2);
    CHECK(pool.append(first, 17));
    CHECK(pool.append(second, 1));
    CHECK(pool.free_block_count() == 0);

    const auto before = sequence(pool, first);
    const auto failed_append = pool.append(first, 16);
    CHECK(failed_append.status == PagedKvStatus::BlocksExhausted);
    CHECK(failed_append.write_slots.empty());
    const auto after_append = sequence(pool, first);
    CHECK(after_append.kv_seq_len == before.kv_seq_len);
    CHECK(after_append.block_table == before.block_table);
    CHECK(pool.free_block_count() == 0);

    CHECK(pool.reserve(first, 33) == PagedKvStatus::BlocksExhausted);
    const auto after_reserve = sequence(pool, first);
    CHECK(after_reserve.kv_seq_len == before.kv_seq_len);
    CHECK(after_reserve.block_table == before.block_table);
    CHECK(pool.free_block_count() == 0);

    const auto fits_existing_blocks = pool.append(first, 15);
    CHECK(fits_existing_blocks.status == PagedKvStatus::Ok);
    CHECK(sequence(pool, first).kv_seq_len == 32);

    PagedKvSequenceHandle unchanged{77, 88};
    CHECK(pool.acquire(3, unchanged) ==
          PagedKvStatus::SequenceSlotsExhausted);
    CHECK(unchanged == (PagedKvSequenceHandle{77, 88}));
    CHECK(pool.active_sequence_count() == 2);
}

static void test_request_identity_and_stale_handles() {
    PagedKvPool pool(4, 2);
    const auto old_handle = acquire(pool, 9001);
    const auto other_handle = acquire(pool, 42);
    CHECK(old_handle.slot == 0);
    CHECK(other_handle.slot == 1);

    PagedKvSequenceHandle found;
    CHECK(pool.lookup(9001, found) == PagedKvStatus::Ok);
    CHECK(found == old_handle);
    CHECK(pool.release(old_handle) == PagedKvStatus::Ok);
    CHECK(pool.lookup(9001, found) == PagedKvStatus::RequestNotFound);

    const auto replacement = acquire(pool, 123456);
    CHECK(replacement.slot == old_handle.slot);
    CHECK(replacement.generation != old_handle.generation);
    CHECK(pool.append(old_handle, 1).status == PagedKvStatus::StaleHandle);
    CHECK(pool.reserve(old_handle, 16) == PagedKvStatus::StaleHandle);
    CHECK(pool.release(old_handle) == PagedKvStatus::StaleHandle);
    PagedKvSequenceSnapshot stale_snapshot;
    CHECK(pool.sequence(old_handle, stale_snapshot) ==
          PagedKvStatus::StaleHandle);
    CHECK(pool.cancel(9001) == PagedKvStatus::RequestNotFound);

    CHECK(pool.append(replacement, 1));
    pool.reset();
    CHECK(pool.active_sequence_count() == 0);
    CHECK(pool.free_block_count() == 4);
    CHECK(pool.append(replacement, 1).status == PagedKvStatus::StaleHandle);

    const auto after_reset = acquire(pool, 555);
    CHECK(after_reset.slot == 0);
    CHECK(after_reset.generation != replacement.generation);
    CHECK(pool.append(after_reset, 1));
    CHECK(sequence(pool, after_reset).block_table.front() == 0);

    const PagedKvSequenceHandle out_of_range{
        std::numeric_limits<uint32_t>::max(), 1};
    CHECK(pool.append(out_of_range, 1).status ==
          PagedKvStatus::StaleHandle);
    CHECK(pool.reserve(out_of_range, 1) == PagedKvStatus::StaleHandle);
    CHECK(pool.release(out_of_range) == PagedKvStatus::StaleHandle);
    PagedKvSequenceSnapshot invalid_snapshot;
    CHECK(pool.sequence(out_of_range, invalid_snapshot) ==
          PagedKvStatus::StaleHandle);
}

static void test_metadata_snapshot() {
    PagedKvPool pool(8, 4);
    const auto first = acquire(pool, 11);
    const auto second = acquire(pool, 22);
    const auto third = acquire(pool, 33);
    CHECK(pool.append(first, 17));
    CHECK(pool.append(second, 1));
    CHECK(pool.append(third, 17));
    CHECK(pool.release(second) == PagedKvStatus::Ok);
    CHECK(pool.reserve(third, 33) == PagedKvStatus::Ok);

    const auto metadata = pool.metadata_snapshot();
    CHECK(metadata.block_size == 16);
    CHECK(metadata.physical_block_count == 8);
    CHECK(metadata.sequences.size() == 2);
    CHECK(equals(metadata.block_table, {0, 1, 3, 4, 2}));

    const auto & first_meta = metadata.sequences[0];
    CHECK(first_meta.request_id == 11);
    CHECK(first_meta.generation == first.generation);
    CHECK(first_meta.sequence_slot == first.slot);
    CHECK(first_meta.kv_seq_len == 17);
    CHECK(first_meta.block_table_offset == 0);
    CHECK(first_meta.block_count == 2);

    const auto & third_meta = metadata.sequences[1];
    CHECK(third_meta.request_id == 33);
    CHECK(third_meta.generation == third.generation);
    CHECK(third_meta.sequence_slot == third.slot);
    CHECK(third_meta.kv_seq_len == 17);
    CHECK(third_meta.block_table_offset == 2);
    CHECK(third_meta.block_count == 3);
}

static void test_duplicate_request_is_transactional() {
    PagedKvPool pool(2, 2);
    const auto first = acquire(pool, 88);
    PagedKvSequenceHandle output{9, 10};
    CHECK(pool.acquire(88, output) == PagedKvStatus::DuplicateRequest);
    CHECK(output == (PagedKvSequenceHandle{9, 10}));
    CHECK(pool.active_sequence_count() == 1);
    CHECK(pool.lookup(88, output) == PagedKvStatus::Ok);
    CHECK(output == first);
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

static void test_invalid_arguments() {
    CHECK(constructor_rejects(0, 1, 16));
    CHECK(constructor_rejects(1, 0, 16));
    CHECK(constructor_rejects(1, 1, 0));
    CHECK(constructor_rejects(
        2, 1, std::numeric_limits<uint32_t>::max()));

    PagedKvPool pool(
        1, 1, std::numeric_limits<uint32_t>::max());
    const auto handle = acquire(pool, 99);
    CHECK(pool.append(handle, 1).status == PagedKvStatus::Ok);
    const auto before = sequence(pool, handle);
    const auto overflow =
        pool.append(handle, std::numeric_limits<uint32_t>::max());
    CHECK(overflow.status == PagedKvStatus::InvalidArgument);
    CHECK(overflow.write_slots.empty());
    const auto after = sequence(pool, handle);
    CHECK(after.kv_seq_len == before.kv_seq_len);
    CHECK(after.block_table == before.block_table);
}

int main() {
    test_block_boundaries();
    test_nondefault_block_size();
    test_reserve_then_append();
    test_noncontiguous_reuse_and_isolation();
    test_exhaustion_rolls_back();
    test_request_identity_and_stale_handles();
    test_metadata_snapshot();
    test_duplicate_request_is_transactional();
    test_invalid_arguments();

    if (failures != 0) {
        std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
        return 1;
    }
    std::printf("OK test_paged_kv_pool (%d checks)\n", checks);
    return 0;
}
