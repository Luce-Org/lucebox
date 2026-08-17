// Host-side unit test for Qwen35SlotManager (concurrent decode slots).
//
// Mirrors test_paged_kv_pool: no model, ggml, or GPU required. Covers the
// admission checks, atomic prompt-plus-headroom reservation, rolling decode
// protection, block-table deltas, exhaustion atomicity, and pool-handle
// lifecycle across retire.

#include "qwen35/concurrency/qwen35_slot_manager.h"
#include "host_check.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

static int g_checks = 0;

static SamplerCfg greedy_sampler() {
    return SamplerCfg{};
}

static std::vector<int32_t> prompt_tokens(int count) {
    return std::vector<int32_t>((size_t)count, 1);
}

static SeqEngine::AdmitResult admit(
        Qwen35SlotManager & manager, uint64_t request_id,
        const std::vector<int32_t> & prompt, const SamplerCfg & sampler) {
    return manager.admit(
        request_id, prompt, sampler);
}

static bool is_admitted(const SeqEngine::AdmitResult & result) {
    return result.status == SeqEngine::AdmitResult::Status::admitted;
}

static bool is_busy(const SeqEngine::AdmitResult & result) {
    return result.status == SeqEngine::AdmitResult::Status::busy;
}

int main() {
    // 8 blocks x 16 tokens = 128 pool tokens, 2 slots, per-seq max_ctx 64.
    {
        PagedKvPool pool(/*physical_block_count=*/8, /*max_sequences=*/2,
                         /*block_size=*/16);
        Qwen35SlotManager mgr(pool, /*max_ctx=*/64);
        CHECK(mgr.slot_count() == 2);
        CHECK(mgr.max_context() == 64);
        CHECK(!mgr.is_active(0) && !mgr.is_active(1));

        // Invalid asks are hard errors, not busy.
        CHECK(!is_admitted(admit(mgr, 1, prompt_tokens(0), greedy_sampler())));
        CHECK(!is_admitted(admit(mgr, 1, prompt_tokens(65), greedy_sampler())));
        CHECK(!mgr.is_active(0) && !mgr.is_active(1));
        CHECK(pool.active_sequence_count() == 0);

        // Admission reserves the prompt's two blocks plus its next decode page
        // without advancing logical length; chunked append consumes only the
        // prompt portion of that private reservation.
        const std::vector<int32_t> admitted_prompt = prompt_tokens(20);
        auto a = admit(mgr, 1, admitted_prompt, greedy_sampler());
        CHECK(is_admitted(a) && !is_busy(a));
        CHECK(a.slot == 0);
        CHECK(pool.free_block_count() == 5);
        CHECK(mgr.is_active(0));
        CHECK(mgr.slot(0).sample_history == admitted_prompt);

        // Prompt allocation follows the chunks actually scheduled. The first
        // ten rows open block 0; the next ten consume its tail and open block
        // 1, returning only that block-table delta.
        auto p0 = mgr.append_prefill(a.slot, 10);
        CHECK(p0.ok);
        CHECK(p0.rows.size() == 10 && p0.rows.front() == 0 &&
              p0.rows.back() == 9);
        CHECK(p0.first_new_block == 0 && p0.new_blocks.size() == 1 &&
              p0.new_blocks[0] == 0);
        CHECK(pool.free_block_count() == 5);
        auto p1 = mgr.append_prefill(a.slot, 10);
        CHECK(p1.ok);
        CHECK(p1.rows.size() == 10 && p1.rows.front() == 10 &&
              p1.rows.back() == 19);
        CHECK(p1.first_new_block == 1 && p1.new_blocks.size() == 1 &&
              p1.new_blocks[0] == 1);
        CHECK(pool.free_block_count() == 5);
        mgr.commit_prefill(0);
        CHECK(mgr.slot(0).cur_pos == 20);

        // Decode appends: row allocation + sample_history; cur_pos advances
        // separately after the step's compute.
        auto st = mgr.append_token(0, /*fed_token=*/42);
        CHECK(st.ok);
        CHECK(st.position == 20);
        CHECK(st.physical_row == 20);     // tail of the prompt's last block
        CHECK(st.new_block < 0 && st.new_block_index < 0);
        CHECK(mgr.slot(0).cur_pos == 20);
        CHECK(mgr.slot(0).sample_history.size() == 21 &&
              mgr.slot(0).sample_history.back() == 42);
        mgr.commit_step(0);
        CHECK(mgr.slot(0).cur_pos == 21);

        // Second admission lands in slot 1 with non-identity rows.
        auto b = admit(mgr, 2, prompt_tokens(20), greedy_sampler());
        CHECK(is_admitted(b) && b.slot == 1);
        auto pb = mgr.append_prefill(b.slot, 20);
        CHECK(pb.ok && pb.rows.front() != 0);
        mgr.commit_prefill(1);

        // Third admission: no free slot -> busy.
        auto c = admit(mgr, 3, prompt_tokens(8), greedy_sampler());
        CHECK(!is_admitted(c) && is_busy(c));

        // Retire frees the slot AND the pool blocks.
        const uint32_t free_before = pool.free_block_count();
        mgr.retire(0);
        CHECK(!mgr.is_active(0));
        CHECK(pool.free_block_count() > free_before);
        CHECK(mgr.is_active(1));

        // The freed slot admits again.
        auto d = admit(mgr, 3, prompt_tokens(8), greedy_sampler());
        CHECK(is_admitted(d) && d.slot == 0);

        // Inactive-slot calls are safe no-ops.
        mgr.retire(0);
        mgr.retire(0);
        CHECK(!mgr.append_prefill(0, 1).ok);
        CHECK(!mgr.append_token(0, 1).ok);
        mgr.commit_step(0);
        mgr.retire(-1);
        mgr.retire(99);
    }

    // A physically impossible headroom page does not make an otherwise useful
    // prompt impossible: this one-block pool falls back to prompt-only.
    {
        PagedKvPool pool(/*physical_block_count=*/1,
                         /*max_sequences=*/2, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, /*max_ctx=*/128);
        auto a = admit(mgr, 1, prompt_tokens(8), greedy_sampler());
        CHECK(is_admitted(a));
        CHECK(pool.free_block_count() == 0);
        CHECK(mgr.append_prefill(a.slot, 8).ok);
        CHECK(pool.free_block_count() == 0);
    }

    // Busy-vs-never-fits classification against a small pool.
    {
        // 4 blocks x 16 = 64 pool tokens, 2 slots, max_ctx 64.
        PagedKvPool pool(4, 2, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, 64);

        // A near-limit prompt needs no additional physical page before max_ctx.
        auto exact = admit(mgr, 1, prompt_tokens(60), greedy_sampler());
        CHECK(is_admitted(exact) && mgr.max_context() == 64);
        CHECK(pool.free_block_count() == 0);
        CHECK(mgr.append_prefill(exact.slot, 60).ok);
        mgr.retire(exact.slot);

        // Occupy most of the pool, then a second request that WOULD fit an
        // empty pool reports busy (blocks held by a live sequence).
        auto big = admit(mgr, 2, prompt_tokens(48), greedy_sampler());  // 3 prompt + 1 headroom
        CHECK(is_admitted(big));
        CHECK(mgr.append_prefill(big.slot, 48).ok);
        auto blocked = admit(mgr, 3, prompt_tokens(32), greedy_sampler());  // 2 blocks
        CHECK(!is_admitted(blocked) && is_busy(blocked));
        mgr.retire(big.slot);
        auto now_fits = admit(mgr, 3, prompt_tokens(32), greedy_sampler());
        CHECK(is_admitted(now_fits));
    }

    // Never-fits: a prompt beyond the WHOLE pool is a hard error, not
    // busy — waiting for a drain could never help.
    {
        // 4 blocks x 16 = 64 pool tokens, but max_ctx allows asking for more.
        PagedKvPool pool(4, 2, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, /*max_ctx=*/128);
        auto never = admit(mgr, 1, prompt_tokens(100), greedy_sampler());
        CHECK(!is_admitted(never) && !is_busy(never));   // prompt 100 > pool 64
        CHECK(pool.active_sequence_count() == 0);

        // Impossibility wins over temporary slot pressure: do not queue an
        // oversized prompt merely because every sequence slot is occupied.
        auto live = admit(mgr, 2, prompt_tokens(16), greedy_sampler());
        CHECK(is_admitted(live));
        auto live2 = admit(mgr, 3, prompt_tokens(16), greedy_sampler());
        CHECK(is_admitted(live2));
        auto still_never = admit(mgr, 4, prompt_tokens(100), greedy_sampler());
        CHECK(!is_admitted(still_never) && !is_busy(still_never));
    }

    // Aggregate prompt reservations prevent two partial prefills from
    // consuming the same capacity and reaching a no-progress deadlock.
    {
        PagedKvPool pool(/*physical_block_count=*/4,
                         /*max_sequences=*/3, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, /*max_ctx=*/64);
        auto a = admit(mgr, 1, prompt_tokens(48), greedy_sampler());
        CHECK(is_admitted(a));
        CHECK(pool.free_block_count() == 0);

        // The second prompt fits the whole pool, but cannot reserve its prompt
        // plus headroom while the first request owns all four pages. It never
        // becomes a second partially-filled live slot.
        auto later = admit(mgr, 2, prompt_tokens(32), greedy_sampler());
        CHECK(!is_admitted(later) && is_busy(later));
        CHECK(pool.active_sequence_count() == 1);

        auto first_chunk = mgr.append_prefill(a.slot, 16);
        CHECK(first_chunk.ok);
        CHECK(pool.free_block_count() == 0);
        // Consuming a reserved page does not make it available to another
        // admission, and appending past the admitted prompt is a hard error.
        later = admit(mgr, 2, prompt_tokens(32), greedy_sampler());
        CHECK(!is_admitted(later) && is_busy(later));
        auto too_much = mgr.append_prefill(a.slot, 33);
        CHECK(!too_much.ok);
        auto final_chunk = mgr.append_prefill(a.slot, 32);
        CHECK(final_chunk.ok);
        mgr.commit_prefill(a.slot);

        mgr.retire(a.slot);
        CHECK(pool.free_block_count() == 4);
        later = admit(mgr, 2, prompt_tokens(32), greedy_sampler());
        CHECK(is_admitted(later));
        mgr.retire(later.slot);
    }

    // When the entire physical pool cannot hold one prompt page plus one decode
    // page, admission falls back to prompt-only and exhaustion stays retryable.
    {
        PagedKvPool pool(/*physical_block_count=*/1,
                         /*max_sequences=*/1, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, /*max_ctx=*/32);
        auto a = admit(mgr, 1, prompt_tokens(16), greedy_sampler());
        CHECK(is_admitted(a));
        CHECK(mgr.append_prefill(a.slot, 16).ok);
        mgr.commit_prefill(a.slot);
        auto decode_blocked = mgr.append_token(a.slot, 77);
        CHECK(!decode_blocked.ok && decode_blocked.busy);
        CHECK(mgr.slot(a.slot).sample_history == prompt_tokens(16));
        CHECK(mgr.slot(a.slot).cur_pos == 16);
    }

    // Rolling headroom has priority over younger admission. Even when a
    // decoder consumes its reserved page while the pool is full, blocks freed
    // later are topped up for that decoder before a new prompt can claim them.
    {
        PagedKvPool pool(/*physical_block_count=*/4,
                         /*max_sequences=*/2, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, /*max_ctx=*/48);
        auto older = admit(mgr, 1, prompt_tokens(16), greedy_sampler());
        auto peer = admit(mgr, 2, prompt_tokens(16), greedy_sampler());
        CHECK(is_admitted(older) && is_admitted(peer));
        CHECK(pool.free_block_count() == 0);
        CHECK(mgr.append_prefill(older.slot, 16).ok);
        CHECK(mgr.append_prefill(peer.slot, 16).ok);
        mgr.commit_prefill(older.slot);
        mgr.commit_prefill(peer.slot);

        // Enter the initially reserved next page. No block is free yet to
        // replenish a third page, but this decode token still succeeds.
        auto enter_second = mgr.append_token(older.slot, 70);
        CHECK(enter_second.ok && enter_second.new_block >= 0);
        mgr.commit_step(older.slot);
        PagedKvSequenceSnapshot snapshot;
        CHECK(pool.sequence(mgr.slot(older.slot).handle, snapshot) ==
              PagedKvStatus::Ok);
        CHECK(snapshot.block_table.size() == 2);
        CHECK(snapshot.reserved_block_count == 0);

        // Retirement frees two pages. A younger admission first restores the
        // older decoder's third-page reserve, then reports busy because only
        // one page remains for its own two-page admission.
        mgr.retire(peer.slot);
        CHECK(pool.free_block_count() == 2);
        auto younger = admit(mgr, 3, prompt_tokens(16), greedy_sampler());
        CHECK(!is_admitted(younger) && is_busy(younger));
        CHECK(pool.sequence(mgr.slot(older.slot).handle, snapshot) ==
              PagedKvStatus::Ok);
        CHECK(snapshot.reserved_block_count == 1);
        CHECK(pool.free_block_count() == 1);

        // The protected third page remains consumable at the next boundary.
        for (int i = 0; i < 15; ++i) {
            CHECK(mgr.append_token(older.slot, 71 + i).ok);
            mgr.commit_step(older.slot);
        }
        auto enter_third = mgr.append_token(older.slot, 99);
        CHECK(enter_third.ok && enter_third.new_block >= 0);
        mgr.commit_step(older.slot);

        mgr.retire(older.slot);
        younger = admit(mgr, 3, prompt_tokens(16), greedy_sampler());
        CHECK(is_admitted(younger));
    }

    // A row whose current append fits its existing page must not consume the
    // last free block as speculative future headroom before a later row that
    // needs that block for its current append. Rolling top-up is therefore an
    // admission-wide operation, never a side effect of append_token().
    {
        PagedKvPool pool(/*physical_block_count=*/5,
                         /*max_sequences=*/3, /*block_size=*/4);
        Qwen35SlotManager mgr(pool, /*max_ctx=*/16);
        auto first = admit(mgr, 1, prompt_tokens(4), greedy_sampler());
        auto boundary = admit(mgr, 2, prompt_tokens(4), greedy_sampler());
        CHECK(is_admitted(first) && is_admitted(boundary));

        PagedKvSequenceHandle blocker;
        CHECK(pool.acquire_reserved(99, /*token_capacity=*/4, blocker) ==
              PagedKvStatus::Ok);
        CHECK(pool.free_block_count() == 0);
        CHECK(mgr.append_prefill(first.slot, 4).ok);
        CHECK(mgr.append_prefill(boundary.slot, 4).ok);
        mgr.commit_prefill(first.slot);
        mgr.commit_prefill(boundary.slot);

        auto first_second_page = mgr.append_token(first.slot, 10);
        CHECK(first_second_page.ok && first_second_page.new_block_index == 1);
        mgr.commit_step(first.slot);  // position 5, inside its second page
        for (int i = 0; i < 4; ++i) {
            CHECK(mgr.append_token(boundary.slot, 20 + i).ok);
            mgr.commit_step(boundary.slot);
        }
        CHECK(mgr.slot(boundary.slot).cur_pos == 8);  // needs page 3 next

        CHECK(pool.release(blocker) == PagedKvStatus::Ok);
        CHECK(pool.free_block_count() == 1);
        CHECK(mgr.append_token(first.slot, 30).ok);
        CHECK(pool.free_block_count() == 1);
        auto boundary_append = mgr.append_token(boundary.slot, 31);
        CHECK(boundary_append.ok && boundary_append.new_block_index == 2);
        CHECK(pool.free_block_count() == 0);
    }

    // Context exhaustion: append_token refuses past max_ctx.
    {
        PagedKvPool pool(4, 1, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, /*max_ctx=*/17);
        auto a = admit(mgr, 1, prompt_tokens(16), greedy_sampler());
        CHECK(is_admitted(a));
        CHECK(mgr.append_prefill(a.slot, 16).ok);
        mgr.commit_prefill(0);
        auto s1 = mgr.append_token(0, 7);   // position 16 (== max_ctx-1)
        CHECK(s1.ok && s1.position == 16);
        CHECK(s1.new_block == 1 && s1.new_block_index == 1);
        mgr.commit_step(0);
        auto s2 = mgr.append_token(0, 8);   // cur_pos == max_ctx -> refuse
        CHECK(!s2.ok);
    }

    // Prefilling lifecycle: an admitted slot stays out of the decode batch
    // until commit_prefill() makes its first sampled token available.
    {
        PagedKvPool pool(8, 2, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, 64);
        auto a = admit(mgr, 1, prompt_tokens(20), greedy_sampler());
        CHECK(is_admitted(a));
        CHECK(mgr.is_prefilling(a.slot));
        CHECK(mgr.is_active(a.slot));
        CHECK(mgr.decoding_count() == 0);
        CHECK(!mgr.append_token(a.slot, 42).ok);

        CHECK(mgr.append_prefill(a.slot, 20).ok);
        mgr.commit_prefill(a.slot);
        CHECK(!mgr.is_prefilling(a.slot));
        CHECK(mgr.decoding_count() == 1);
        CHECK(mgr.append_token(a.slot, 42).ok);
        CHECK(!mgr.append_prefill(a.slot, 1).ok);

        // A second admission can prefill while the first slot decodes.
        auto b = admit(mgr, 2, prompt_tokens(20), greedy_sampler());
        CHECK(is_admitted(b));
        CHECK(mgr.is_active(a.slot) && mgr.is_active(b.slot));
        CHECK(mgr.decoding_count() == 1);

        // Retiring during prefill clears the state before slot reuse.
        mgr.retire(b.slot);
        CHECK(!mgr.is_prefilling(b.slot));
        auto c = admit(mgr, 3, prompt_tokens(8), greedy_sampler());
        CHECK(is_admitted(c) && c.slot == b.slot);
        CHECK(mgr.is_prefilling(c.slot));
    }

    // Seeded sampling RNG is deterministic per admission. The sampler alone
    // decides: a seed is honoured exactly when needs_logit_processing() says
    // the slot actually draws, so there is no way to ask for seeded sampling
    // and be given argmax (or the reverse).
    {
        PagedKvPool pool(4, 1, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, 64);
        SamplerCfg cfg = greedy_sampler();
        cfg.temp = 0.7f;            // needs_logit_processing() -> true
        cfg.seed = 1234;
        CHECK(cfg.needs_logit_processing());
        auto a = admit(mgr, 1, prompt_tokens(4), cfg);
        CHECK(is_admitted(a));
        const uint64_t first = mgr.slot(0).rng();
        mgr.retire(0);
        auto b = admit(mgr, 2, prompt_tokens(4), cfg);
        CHECK(is_admitted(b));
        CHECK(mgr.slot(0).rng() == first);

        // A different seed is a different stream.
        mgr.retire(0);
        cfg.seed = 5678;
        auto c = admit(mgr, 3, prompt_tokens(4), cfg);
        CHECK(is_admitted(c));
        CHECK(mgr.slot(0).rng() != first);
    }

    // A greedy sampler never draws, so its seed is irrelevant and admission
    // must not depend on one being present.
    {
        PagedKvPool pool(4, 1, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, 64);
        SamplerCfg cfg = greedy_sampler();
        cfg.seed = 1234;
        CHECK(!cfg.needs_logit_processing());
        auto a = admit(mgr, 1, prompt_tokens(4), cfg);
        CHECK(is_admitted(a));
        CHECK(!mgr.slot(0).sampler.needs_logit_processing());
    }

    // Long-prefill policy follows active request length and clears on retire.
    {
        PagedKvPool pool(128, 2, /*block_size=*/16);
        Qwen35SlotManager mgr(pool, 2048);
        CHECK(!mgr.has_prefill_prompt_at_least(768));
        auto short_req =
            admit(mgr, 201, prompt_tokens(512), greedy_sampler());
        CHECK(is_admitted(short_req));
        CHECK(!mgr.has_prefill_prompt_at_least(768));
        auto long_req =
            admit(mgr, 202, prompt_tokens(800), greedy_sampler());
        CHECK(is_admitted(long_req));
        CHECK(mgr.has_prefill_prompt_at_least(768));
        mgr.retire(long_req.slot);
        CHECK(!mgr.has_prefill_prompt_at_least(768));
    }

    std::printf("OK test_seq_slot_manager (%d checks)\n", g_checks);
    return 0;
}
