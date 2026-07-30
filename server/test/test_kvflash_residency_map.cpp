// Dependency-free native contract test for KVFlash ownership bookkeeping.

#include "../src/common/kvflash_residency_map.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using dflash::common::KvFlashConfig;
using dflash::common::KvFlashResidencyMap;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static KvFlashConfig small_config() {
    KvFlashConfig cfg;
    cfg.chunk_tokens = 4;
    cfg.pool_tokens = 16;
    cfg.sink_chunks = 1;
    cfg.tail_window_chunks = 1;
    cfg.max_logical_tokens = 32;
    return cfg;
}

static bool accept_page_out(void *, int, int) {
    return true;
}

static bool reject_page_out(void *, int, int) {
    return false;
}

static KvFlashResidencyMap::PreparePageOut accepting_page_out() {
    return {nullptr, &accept_page_out};
}

static void test_config_validation() {
    KvFlashConfig cfg = small_config();
    expect(KvFlashResidencyMap::valid_config(cfg), "valid config accepted");
    expect(KvFlashResidencyMap::min_pool_tokens(cfg) == 16,
           "minimum includes sink, tail, victim, and append head");

    cfg.chunk_tokens = 0;
    expect(!KvFlashResidencyMap::valid_config(cfg), "zero chunk rejected");
    cfg = small_config();
    cfg.pool_tokens = 15;
    expect(!KvFlashResidencyMap::valid_config(cfg), "partial block rejected");
    cfg = small_config();
    cfg.pool_tokens = 12;
    expect(!KvFlashResidencyMap::valid_config(cfg), "deadlocking pool rejected");
    cfg = small_config();
    cfg.sink_chunks = -1;
    expect(!KvFlashResidencyMap::valid_config(cfg), "negative sink rejected");
    cfg = small_config();
    cfg.tail_window_chunks = -1;
    expect(!KvFlashResidencyMap::valid_config(cfg), "negative tail rejected");
    cfg = small_config();
    cfg.max_logical_tokens = 15;
    expect(!KvFlashResidencyMap::valid_config(cfg),
           "logical bound below resident pool rejected");
}

static void test_identity_bijection_and_masks() {
    KvFlashResidencyMap map;
    expect(map.configure(small_config()), "configure");
    expect(map.invariant_holds(), "initial free complement");
    const uint64_t initial_epoch = map.epoch();

    for (int pos = 0; pos < 16; ++pos) {
        const auto acquired = map.acquire(pos);
        expect(acquired.slot == pos, "identity handout");
        expect(map.invariant_holds(), "bijection after acquire");
    }
    expect(map.epoch() == initial_epoch + 4, "epoch changes once per new chunk");
    expect(map.resident_blocks() == 4, "pool full");
    expect(map.is_identity(), "full identity map");
    expect(map.identity_prefix_covers(16), "identity prefix covered");
    expect(!map.identity_prefix_covers(17), "unmaterialized prefix rejected");

    std::vector<int32_t> positions(16, -2);
    std::vector<uint16_t> mask(16, 0xFFFF);
    map.fill_slot_pos(positions.data());
    map.fill_slot_mask(mask.data());
    for (int i = 0; i < 16; ++i) {
        expect(positions[(size_t)i] == i, "slot position reflects owner");
        expect(mask[(size_t)i] == 0, "resident slot unmasked");
    }

    const uint64_t epoch = map.epoch();
    expect(!map.acquire(-1), "negative logical position rejected");
    expect(map.slot_of(-1) == -1, "negative lookup rejected");
    expect(!map.acquire(32), "logical-context boundary rejected");
    expect(map.slot_of(32) == -1, "bounded lookup rejected");
    expect(!map.acquire(
        (int64_t)(std::numeric_limits<int>::max()) * 4 + 4),
        "unrepresentable logical chunk rejected");
    expect(map.epoch() == epoch, "invalid input does not change epoch");
    expect(map.invariant_holds(), "invalid input preserves invariant");
}

static void test_protection_eviction_and_recall() {
    KvFlashResidencyMap map;
    expect(map.configure(small_config()), "configure eviction map");
    for (int chunk = 0; chunk < 4; ++chunk) {
        expect(map.acquire((int64_t)chunk * 4).slot >= 0, "fill block");
    }

    struct Capture {
        int chunk = -1;
        int block = -1;
    } capture;
    const auto capture_page_out = KvFlashResidencyMap::PreparePageOut{
        &capture,
        [](void * context, int chunk, int block) {
            auto * captured = static_cast<Capture *>(context);
            captured->chunk = chunk;
            captured->block = block;
            return true;
        }};
    const auto acquired = map.acquire(
        16, {}, capture_page_out);
    expect(acquired.slot >= 0, "append acquires through eviction");
    expect(capture.chunk == 1, "LRU skips sink and protected tail");
    expect(capture.block == 1, "victim block reported");
    expect(acquired.evicted_chunk == 1 && acquired.block == 1,
           "ownership transfers atomically");
    expect(map.is_host_backed(1) && !map.is_resident(1),
           "victim becomes host-backed");
    expect(map.block_of(0) == 0 && map.block_of(3) == 3 &&
           map.block_of(4) == 1, "protected mappings retained");
    expect(!map.is_identity(), "eviction ends identity layout");
    expect(map.invariant_holds(), "post-eviction bijection");

    struct Scores {
        std::vector<float> values;
    } scores{{0.0f, std::numeric_limits<float>::infinity(), 2.0f, 0.0f, 0.0f}};
    const KvFlashResidencyMap::Score score{
        &scores,
        [](const void * context, int chunk) {
            const auto & values = static_cast<const Scores *>(context)->values;
            return values[(size_t)chunk];
        }};
    const std::vector<uint8_t> wanted = map.desired_residency(score);
    expect(wanted[0] && wanted[3] && wanted[4],
           "reselect always includes sink and protected tail");
    expect(wanted[1] && !wanted[2],
           "remaining capacity follows score after protection");

    const auto recalled = map.acquire(
        4, {}, accepting_page_out());
    expect(recalled.slot >= 0 && recalled.recalled, "host-backed chunk recalled");
    expect(map.is_resident(1), "recalled chunk resident");
    expect(map.invariant_holds(), "recall preserves bijection");
}

static void test_failed_allocation_rolls_back() {
    KvFlashResidencyMap map;
    expect(map.configure(small_config()), "configure rollback map");
    for (int chunk = 0; chunk < 4; ++chunk) {
        expect(map.acquire((int64_t)chunk * 4).slot >= 0, "fill rollback map");
    }

    const uint64_t epoch = map.epoch();
    const int chunks = map.n_chunks();
    std::vector<int> owners;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        owners.push_back(map.block_of(chunk));
    }
    struct Count {
        int calls = 0;
    } count;
    const auto reject_counting = KvFlashResidencyMap::PreparePageOut{
        &count,
        [](void * context, int, int) {
            ++static_cast<Count *>(context)->calls;
            return false;
        }};
    const auto failed = map.acquire(
        16, {}, reject_counting);
    expect(!failed, "rejected page-out fails acquire");
    expect(count.calls == 1, "victim preparation attempted once");
    expect(map.epoch() == epoch, "failed acquire preserves epoch");
    expect(map.n_chunks() == chunks, "failed acquire restores logical extent");
    for (int chunk = 0; chunk < chunks; ++chunk) {
        expect(map.block_of(chunk) == owners[(size_t)chunk],
               "failed acquire preserves ownership");
    }
    expect(map.invariant_holds(), "failed acquire preserves free complement");
}

static void test_order_page_out_and_reset() {
    KvFlashResidencyMap map;
    expect(map.configure(small_config()), "configure ordered map");
    expect(!map.set_block_order({3, 2, 1}), "partial order rejected");
    expect(!map.set_block_order({3, 2, 2, 0}), "duplicate order rejected");
    expect(map.set_block_order({3, 1, 0, 2}), "permutation accepted");
    expect(map.acquire(0).block == 3, "custom first block");
    expect(!map.set_block_order({0, 1, 2, 3}),
           "order cannot replace live ownership");

    const uint64_t before_page_out = map.epoch();
    expect(!map.page_out(0, {nullptr, &reject_page_out}),
           "failed explicit page-out rejected");
    expect(map.block_of(0) == 3 && map.epoch() == before_page_out,
           "failed page-out rolls back");
    expect(map.page_out(0, accepting_page_out()),
           "explicit page-out succeeds");
    expect(map.block_of(0) == -1 && map.is_host_backed(0),
           "explicit page-out updates state");
    expect(map.invariant_holds(), "explicit page-out preserves complement");

    const uint64_t before_reset = map.epoch();
    map.reset();
    expect(map.epoch() == before_reset + 1, "reset advances epoch");
    expect(map.n_chunks() == 0 && map.resident_blocks() == 0,
           "reset drops request state");
    expect(map.is_identity() && map.invariant_holds(),
           "reset restores empty identity invariant");
    expect(map.acquire(0).slot == 0, "reset restores identity handout order");
}

int main() {
    test_config_validation();
    test_identity_bijection_and_masks();
    test_protection_eviction_and_recall();
    test_failed_allocation_rolls_back();
    test_order_page_out_and_reset();
    std::puts("PASS: kvflash residency map");
    return 0;
}
