// Shared body for generated and legacy KVFlash residency-map harnesses.

#include "common/kvflash_residency_map.h"

#include <cassert>
#include <cstdint>

#ifndef LUCEBOX_FORMAL_CONTRACT_SYMBOL
#error "LUCEBOX_FORMAL_CONTRACT_SYMBOL must name the production map type"
#endif

#ifndef LUCEBOX_FORMAL_BLOCKS
#define LUCEBOX_FORMAL_BLOCKS 4
#endif

using ContractMap = LUCEBOX_FORMAL_CONTRACT_SYMBOL;
using dflash::common::KvFlashConfig;

namespace {

constexpr int kChunkTokens = 1;
constexpr int kPoolTokens = LUCEBOX_FORMAL_BLOCKS;
constexpr int kMaxLogicalTokens = LUCEBOX_FORMAL_BLOCKS + 2;
constexpr uint16_t kF16Zero = 0x0000;
constexpr uint16_t kF16NegInf = 0xFC00;

bool accept_page_out(void *, int, int) {
    return true;
}

bool reject_page_out(void *, int, int) {
    return false;
}

ContractMap::PreparePageOut accepting() {
    return {nullptr, &accept_page_out};
}

ContractMap::PreparePageOut rejecting() {
    return {nullptr, &reject_page_out};
}

KvFlashConfig config() {
    KvFlashConfig cfg;
    cfg.chunk_tokens = kChunkTokens;
    cfg.pool_tokens = kPoolTokens;
    cfg.sink_chunks = 1;
    cfg.tail_window_chunks = 1;
    cfg.max_logical_tokens = kMaxLogicalTokens;
    return cfg;
}

void fill_identity(ContractMap & map) {
    for (int chunk = 0; chunk < LUCEBOX_FORMAL_BLOCKS; ++chunk) {
        const auto acquired = map.acquire(chunk);
        assert(acquired);
        assert(acquired.chunk == chunk);
        assert(acquired.block == chunk);
        assert(acquired.slot == chunk);
        assert(map.slot_of(chunk) == chunk);
    }
    assert(map.resident_blocks() == LUCEBOX_FORMAL_BLOCKS);
    assert(map.free_blocks().empty());
    assert(map.is_identity());
    assert(map.identity_prefix_covers(kPoolTokens));
    assert(!map.identity_prefix_covers(kPoolTokens + 1));
    assert(map.invariant_holds());
}

}  // namespace

int main() {
    static_assert(LUCEBOX_FORMAL_BLOCKS >= 4);

    KvFlashConfig cfg = config();
    assert(ContractMap::valid_config(cfg));
    assert(ContractMap::min_pool_tokens(cfg) <= cfg.pool_tokens);
    KvFlashConfig invalid = cfg;
    invalid.chunk_tokens = 0;
    assert(!ContractMap::valid_config(invalid));
    invalid = cfg;
    invalid.max_logical_tokens = cfg.pool_tokens - 1;
    assert(!ContractMap::valid_config(invalid));

    ContractMap map;
    assert(map.configure(cfg));
    fill_identity(map);
    const uint64_t full_epoch = map.epoch();
    int owners[LUCEBOX_FORMAL_BLOCKS];
    for (int chunk = 0; chunk < LUCEBOX_FORMAL_BLOCKS; ++chunk) {
        owners[chunk] = map.block_of(chunk);
    }

    // Callback rejection must leave the complete map transition uncommitted.
    const auto rejected =
        map.acquire(LUCEBOX_FORMAL_BLOCKS, {}, rejecting());
    assert(!rejected);
    assert(map.epoch() == full_epoch);
    assert(map.n_chunks() == LUCEBOX_FORMAL_BLOCKS);
    for (int chunk = 0; chunk < LUCEBOX_FORMAL_BLOCKS; ++chunk) {
        assert(map.block_of(chunk) == owners[chunk]);
    }
    assert(map.invariant_holds());

    // With one sink and one protected tail, LRU must transfer chunk 1's block
    // to the new append head while preserving both protected mappings.
    const auto evicted =
        map.acquire(LUCEBOX_FORMAL_BLOCKS, {}, accepting());
    assert(evicted);
    assert(evicted.changed);
    assert(evicted.evicted_chunk == 1);
    assert(evicted.block == 1);
    assert(evicted.slot == 1);
    assert(map.is_resident(0));
    assert(map.block_of(0) == 0);
    assert(map.is_resident(LUCEBOX_FORMAL_BLOCKS - 1));
    assert(
        map.block_of(LUCEBOX_FORMAL_BLOCKS - 1) ==
        LUCEBOX_FORMAL_BLOCKS - 1);
    assert(!map.is_resident(1));
    assert(map.is_host_backed(1));
    assert(map.block_of(LUCEBOX_FORMAL_BLOCKS) == 1);
    assert(map.slot_of(LUCEBOX_FORMAL_BLOCKS) == 1);
    assert(map.resident_blocks() == LUCEBOX_FORMAL_BLOCKS);
    assert(map.free_blocks().empty());
    assert(map.epoch() == full_epoch + 2);
    assert(!map.is_identity());
    assert(map.invariant_holds());

    int32_t positions[kPoolTokens];
    uint16_t masks[kPoolTokens];
    map.fill_slot_pos(positions);
    map.fill_slot_mask(masks);
    for (int block = 0; block < LUCEBOX_FORMAL_BLOCKS; ++block) {
        assert(masks[block] == kF16Zero);
        if (block == 1) {
            assert(positions[block] == LUCEBOX_FORMAL_BLOCKS);
        } else {
            assert(positions[block] == block);
        }
    }

    // An out-of-range logical position is rejected before sparse growth.
    assert(!map.acquire(kMaxLogicalTokens));
    assert(map.slot_of(kMaxLogicalTokens) == -1);

    // The validity mask sentinel is also checked on an empty configured map.
    ContractMap empty;
    assert(empty.configure(cfg));
    empty.fill_slot_pos(positions);
    empty.fill_slot_mask(masks);
    for (int slot = 0; slot < kPoolTokens; ++slot) {
        assert(positions[slot] == -1);
        assert(masks[slot] == kF16NegInf);
    }
    return 0;
}
