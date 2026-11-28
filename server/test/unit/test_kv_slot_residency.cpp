#include "CppUnitTestFramework.hpp"
#include "common/concurrency/kv_slot_residency.h"
#include "common/concurrency/seq_slot_manager.h"

#include <vector>
#include <stdexcept>
#define TEST_ASSERT(x) do { if (!(x)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #x); } while (false)


using namespace luce::common;

namespace {
struct KvSlotResidencyFixture {};
}  // namespace

TEST_CASE(KvSlotResidencyFixture, test_ten_64k_agents_admit_into_160k_pool_bounded) {
    // 163,840 tokens = 10,240 blocks of 16 tokens; 10 slots x 16,384 resident cap.
    PagedKvPool pool(/*physical_block_count=*/10240, /*max_sequences=*/10, /*block_size=*/16);
    SeqSlotManager sm(pool, /*max_ctx=*/131072);
    sm.set_max_resident_tokens(16384);

    std::vector<int32_t> prompt(65536, 42);
    SamplerCfg sampler;
    for (int i = 0; i < 10; ++i) {
        auto ar = sm.admit((uint64_t)(i + 1), prompt, sampler);
        TEST_ASSERT(ar.status == SeqEngine::AdmitResult::Status::admitted);
    }
    TEST_ASSERT(pool.active_sequence_count() == 10);
}

TEST_CASE(KvSlotResidencyFixture, test_evict_recall_dense_block_table_and_shared_pin) {
    // Small pool: 160 blocks = 2,560 tokens, 2 slots, per-slot budget = 1,024 tokens (16 chunks of 64).
    PagedKvPool pool(/*physical_block_count=*/160, /*max_sequences=*/2, /*block_size=*/16);
    SeqSlotManager sm(pool, /*max_ctx=*/8192);
    sm.set_max_resident_tokens(1024);

    KvSlotResidency::Config cfg;
    cfg.per_slot_tokens = 1024;
    cfg.chunk_tokens = 64;
    cfg.preamble_reserve_tokens = 128;  // 2 pinned preamble chunks
    cfg.recall_chunks = 4;              // 2 units of 2 chunks
    cfg.tail_window_chunks = 2;
    KvSlotResidency res(pool, /*backend=*/nullptr, /*tensors=*/{}, cfg);

    // Admit slot 0 with a 1,536-token (24-chunk) prompt (> 16-chunk budget).
    std::vector<int32_t> prompt(1536, 7);
    auto ar = sm.admit(1, prompt, SamplerCfg{});
    TEST_ASSERT(ar.status == SeqEngine::AdmitResult::Status::admitted);
    const int s0 = ar.slot;
    res.init_slot(s0, sm.slot(s0).handle, /*preamble_tokens=*/128);
    TEST_ASSERT(res.pinned_chunks(s0) == 2);

    // Prefill in 256-token chunks up to r_pos = 1408 (22 sealed chunks).
    std::string err;
    for (int pos = 0; pos < 1408;) {
        const int step = std::min(256, 1408 - pos);
        TEST_ASSERT(res.ensure_capacity(s0, pos, step, err));
        auto ch = sm.append_prefill(s0, step);
        TEST_ASSERT(ch.ok);
        res.on_tokens_written(s0, pos, pos + step);
        pos += step;
    }
    TEST_ASSERT(res.resident_chunks(s0) <= 16);
    TEST_ASSERT(res.is_chunk_resident(s0, 0));
    TEST_ASSERT(res.is_chunk_resident(s0, 1));
    TEST_ASSERT(res.stats(s0).evictions >= 6);

    // Pin the first 2 chunks in PagedKvPool (#1079 style) and verify eviction skips them.
    std::vector<uint32_t> pinned;
    TEST_ASSERT(pool.pin_prefix_blocks(sm.slot(s0).handle, 8, pinned) == PagedKvStatus::Ok);

    // Evict chunk 10 & 11 explicitly (already evicted if LRU, or guarantee via extra prefill) and score unit 5 (chunks 10,11) highest.
    // Candidate floor is kvflash_recall_first_chunk(2) == 10; tail window is [16..22).
    std::vector<float> scores(22, 0.1f);
    scores[10] = 9.0f;
    scores[11] = 9.0f;
    std::vector<int> picked;
    TEST_ASSERT(res.apply_recall(s0, /*r_pos=*/1408, /*prompt_end=*/1536, scores, /*unit=*/2, &picked, err));
    TEST_ASSERT(!picked.empty());
    TEST_ASSERT(picked[0] == 5);
    TEST_ASSERT(res.is_chunk_resident(s0, 10));
    TEST_ASSERT(res.is_chunk_resident(s0, 11));

    // Dense active block table has zero evicted (-1) holes and strictly <= budget tokens.
    std::vector<int32_t> active;
    int res_toks = 0, prefix_res = 0;
    TEST_ASSERT(res.build_active_block_table(s0, 1408, active, res_toks, prefix_res, 1408));
    TEST_ASSERT(res_toks <= 1024);
    TEST_ASSERT(res_toks == prefix_res);
    TEST_ASSERT((int)active.size() * 16 == res_toks);
    for (int32_t pb : active) { TEST_ASSERT(pb >= 0 && pb < 160); }

    // Finish prefill + decode 1 token, then verify Think-Rewind back to r_pos=1408.
    TEST_ASSERT(res.ensure_capacity(s0, 1408, 128, err));
    auto tail = sm.append_prefill(s0, 128);
    TEST_ASSERT(tail.ok);
    res.on_tokens_written(s0, 1408, 1536);
    sm.commit_prefill(s0);
    TEST_ASSERT(res.truncate(s0, 1408));
    TEST_ASSERT(sm.rewind_to_prefill(s0, 1408));
    TEST_ASSERT(sm.is_prefilling(s0));
    TEST_ASSERT(sm.slot(s0).cur_pos == 1408);

    pool.unpin_prefix_blocks(pinned);
    res.clear(s0);
    sm.retire(s0);
    TEST_ASSERT(pool.free_block_count() == 160);
}

TEST_CASE(KvSlotResidencyFixture, test_cow_snapshot_capture_and_bounded_restore) {
    PagedKvPool pool(/*physical_block_count=*/160, /*max_sequences=*/2, /*block_size=*/16);
    SeqSlotManager sm(pool, /*max_ctx=*/8192);
    sm.set_max_resident_tokens(1024);

    KvSlotResidency::Config cfg;
    cfg.per_slot_tokens = 1024;
    cfg.chunk_tokens = 64;
    cfg.preamble_reserve_tokens = 128;
    cfg.recall_chunks = 4;
    cfg.tail_window_chunks = 2;
    KvSlotResidency res(pool, /*backend=*/nullptr, /*tensors=*/{}, cfg);

    // Slot 0 prefills 1,420 tokens (> 1,024 budget, unaligned tail!).
    std::vector<int32_t> p1(1420, 7);
    auto ar0 = sm.admit(1, p1, SamplerCfg{});
    TEST_ASSERT(ar0.status == SeqEngine::AdmitResult::Status::admitted);
    const int s0 = ar0.slot;
    res.init_slot(s0, sm.slot(s0).handle, /*preamble_tokens=*/128);
    std::string err;
    for (int pos = 0; pos < 1420;) {
        const int step = std::min(256, 1420 - pos);
        TEST_ASSERT(res.ensure_capacity(s0, pos, step, err));
        auto ch = sm.append_prefill(s0, step);
        TEST_ASSERT(ch.ok);
        res.on_tokens_written(s0, pos, pos + step);
        pos += step;
    }

    KvSlotResidency::Snapshot snap;
    TEST_ASSERT(res.capture_snapshot(s0, 1420, snap, err));
    TEST_ASSERT(snap.valid());
    TEST_ASSERT(snap.tokens == 1420);
    TEST_ASSERT((int)snap.chunks.size() == 23);

    // Retire slot 0 and restore into slot 1 for a 1,548-token follow-up turn!
    res.clear(s0);
    sm.retire(s0);
    std::vector<int32_t> p2(1548, 7);
    auto ar1 = sm.admit(2, p2, SamplerCfg{});
    TEST_ASSERT(ar1.status == SeqEngine::AdmitResult::Status::admitted);
    const int s1 = ar1.slot;
    std::vector<uint8_t> blk_mask, chk_mask;
    res.build_restore_resident_mask(snap, 128, blk_mask, chk_mask);
    std::vector<uint32_t> block_table;
    TEST_ASSERT(sm.seed_evicted_prefix(s1, 1420, blk_mask, block_table));
    TEST_ASSERT(res.restore_snapshot(s1, sm.slot(s1).handle, snap, 128, chk_mask, block_table, err));
    TEST_ASSERT(res.resident_chunks(s1) <= 16);
    TEST_ASSERT(sm.slot(s1).cur_pos == 1420);

    // Append remaining 128 tokens and verify dense active block table is valid & bounded.
    TEST_ASSERT(res.ensure_capacity(s1, 1420, 128, err));
    auto tail = sm.append_prefill(s1, 128);
    TEST_ASSERT(tail.ok);
    res.on_tokens_written(s1, 1420, 1548);
    std::vector<int32_t> active;
    int res_toks = 0, pre_toks = 0;
    TEST_ASSERT(res.build_active_block_table(s1, 1548, active, res_toks, pre_toks, 1420));
    TEST_ASSERT(res_toks <= 1024);
    TEST_ASSERT(res_toks - pre_toks == 128);

    res.clear(s1);
    sm.retire(s1);
    TEST_ASSERT(pool.free_block_count() == 160);
}

TEST_CASE(KvSlotResidencyFixture, test_dynamic_pin_reserve_and_low_chunk_recall) {
    // 1) Dynamic pin_reserve_chunks charges only actually pinned preamble chunks,
    //    not the full 4096-token ceiling (zero tail waste).
    {
        PagedKvPool pool(/*physical_block_count=*/2048, /*max_sequences=*/2, /*block_size=*/16);
        SeqSlotManager sm(pool, /*max_ctx=*/32768);
        sm.set_max_resident_tokens(16384);

        KvSlotResidency::Config cfg;
        cfg.per_slot_tokens = 16384;
        cfg.chunk_tokens = 64;
        cfg.preamble_reserve_tokens = 4096;  // 64-chunk ceiling
        cfg.recall_chunks = 60;
        cfg.tail_window_chunks = 4;
        KvSlotResidency res(pool, /*backend=*/nullptr, /*tensors=*/{}, cfg);

        std::vector<int32_t> p(2048, 3);
        auto a0 = sm.admit(1, p, SamplerCfg{});
        auto a1 = sm.admit(2, p, SamplerCfg{});
        TEST_ASSERT(a0.status == SeqEngine::AdmitResult::Status::admitted);
        TEST_ASSERT(a1.status == SeqEngine::AdmitResult::Status::admitted);
        res.init_slot(a0.slot, sm.slot(a0.slot).handle, /*preamble_tokens=*/1595);
        res.init_slot(a1.slot, sm.slot(a1.slot).handle, /*preamble_tokens=*/0);
        TEST_ASSERT(res.pinned_chunks(a0.slot) == 25);
        TEST_ASSERT(res.effective_pin_reserve_chunks(a0.slot) == 25);
        TEST_ASSERT(res.pinned_chunks(a1.slot) == 0);
        TEST_ASSERT(res.effective_pin_reserve_chunks(a1.slot) == 0);
        res.clear(a0.slot);
        res.clear(a1.slot);
        sm.retire(a0.slot);
        sm.retire(a1.slot);
    }

    // 2) Close the [64, 640) dead zone: with pinned_chunks < 10 and unit < 10,
    //    chunks [pinned_chunks..9] are eligible for recall instead of unrecallable.
    {
        PagedKvPool pool(/*physical_block_count=*/160, /*max_sequences=*/2, /*block_size=*/16);
        SeqSlotManager sm(pool, /*max_ctx=*/8192);
        sm.set_max_resident_tokens(1024);

        KvSlotResidency::Config cfg;
        cfg.per_slot_tokens = 1024;
        cfg.chunk_tokens = 64;
        cfg.preamble_reserve_tokens = 256;  // 4-chunk cap, only 2 chunks actually pinned
        cfg.recall_chunks = 4;
        cfg.tail_window_chunks = 2;
        KvSlotResidency res(pool, /*backend=*/nullptr, /*tensors=*/{}, cfg);

        std::vector<int32_t> prompt(1536, 7);
        auto ar = sm.admit(10, prompt, SamplerCfg{});
        TEST_ASSERT(ar.status == SeqEngine::AdmitResult::Status::admitted);
        const int s0 = ar.slot;
        res.init_slot(s0, sm.slot(s0).handle, /*preamble_tokens=*/128);
        TEST_ASSERT(res.pinned_chunks(s0) == 2);
        TEST_ASSERT(res.effective_pin_reserve_chunks(s0) == 2);

        std::string err;
        for (int pos = 0; pos < 1408;) {
            const int step = std::min(256, 1408 - pos);
            TEST_ASSERT(res.ensure_capacity(s0, pos, step, err));
            auto ch = sm.append_prefill(s0, step);
            TEST_ASSERT(ch.ok);
            res.on_tokens_written(s0, pos, pos + step);
            pos += step;
        }
        TEST_ASSERT(!res.is_chunk_resident(s0, 2));
        TEST_ASSERT(!res.is_chunk_resident(s0, 3));

        // Score unit 1 (chunks 2, 3 — inside old [64, 640) dead zone) highest.
        std::vector<float> scores(22, 0.1f);
        scores[2] = 9.0f;
        scores[3] = 9.0f;
        std::vector<int> picked;
        TEST_ASSERT(res.apply_recall(s0, /*r_pos=*/1408, /*prompt_end=*/1536, scores, /*unit=*/2, &picked, err));
        TEST_ASSERT(!picked.empty());
        TEST_ASSERT(picked[0] == 1);
        TEST_ASSERT(res.is_chunk_resident(s0, 2));
        TEST_ASSERT(res.is_chunk_resident(s0, 3));

        res.clear(s0);
        sm.retire(s0);
        TEST_ASSERT(pool.free_block_count() == 160);
    }
}
