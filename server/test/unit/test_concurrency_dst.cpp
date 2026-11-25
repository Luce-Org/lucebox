// Tier 0 CPU Deterministic Simulation Test (DST) for concurrent serving.
//
// Pure host-side C++17: no model, ggml, or GPU required. Drives PagedKvPool,
// SeqSlotManager, plan_prefill_slices, and BasicPrefixCaptureTxn across
// deterministic seeded schedules combining:
//   - random request admissions (normal, busy rejection, oversized rejection)
//   - fair chunked prefills via plan_prefill_slices with round-robin rotation
//   - restored-prefix seeding (asserting disjoint physical rows)
//   - parallel prefix capture transaction commit / fail / cancel / mismatch
//   - decode headroom reservation, staged token appends, commits, and rollbacks
//   - KV detach/attach suspension and evict-for-recompute / resume cycles
//   - abrupt mid-prefill and mid-decode cancellations (retire)
//
// Asserts after every step:
//   1. Zero double-owned physical blocks across all active sequences.
//   2. Exact block conservation: sum(owned_blocks) + free_blocks == total_blocks.
//   3. Exact active sequence count parity between SeqSlotManager and PagedKvPool.
//   4. Zero leaked blocks or dangling handles after final drain.

#include "CppUnitTestFramework.hpp"
#include "common/concurrency/paged_kv_pool.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/seq_slot_manager.h"
#include "server/kv/parallel_prefix_txn.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace luce::common;

namespace {

struct ConcurrencyDstFixture {};

struct SimReservationState {
    int committed = 0;
    int cancelled = 0;
    int aborted = 0;
    int active_count = 0;
};

class SimReservation {
public:
    SimReservation() = default;
    SimReservation(SimReservationState & st, int slot, int cut)
        : state_(&st), slot_(slot), cut_(cut) {
        ++state_->active_count;
    }
    ~SimReservation() { cancel(); }

    SimReservation(const SimReservation &) = delete;
    SimReservation & operator=(const SimReservation &) = delete;
    SimReservation(SimReservation && other) noexcept { take(std::move(other)); }
    SimReservation & operator=(SimReservation && other) noexcept {
        if (this != &other) {
            cancel();
            take(std::move(other));
        }
        return *this;
    }

    bool active() const { return state_ != nullptr; }
    int slot() const { return slot_; }
    int target_cut() const { return cut_; }

    bool commit(const std::vector<int32_t> &, size_t, bool = false) {
        if (!active()) return false;
        ++state_->committed;
        release_active();
        return true;
    }

    void cancel() {
        if (!active()) return;
        ++state_->cancelled;
        release_active();
    }

    void abort() {
        if (!active()) return;
        ++state_->aborted;
        release_active();
    }

private:
    void release_active() {
        --state_->active_count;
        state_ = nullptr;
        slot_ = -1;
        cut_ = 0;
    }

    void take(SimReservation && other) {
        state_ = other.state_;
        slot_ = other.slot_;
        cut_ = other.cut_;
        other.state_ = nullptr;
        other.slot_ = -1;
        other.cut_ = 0;
    }

    SimReservationState * state_ = nullptr;
    int slot_ = -1;
    int cut_ = 0;
};

struct SimEngine {
    int discarded = 0;
    void discard_prefix_store(PrefixStoreRef) { ++discarded; }
};

using SimTxn = BasicPrefixCaptureTxn<SimReservation, SimEngine>;

std::vector<int32_t> make_prompt(int len, int32_t salt) {
    std::vector<int32_t> out((size_t)len);
    for (int i = 0; i < len; ++i) {
        out[(size_t)i] = salt * 1024 + i + 1;
    }
    return out;
}

std::string check_pool_invariants(const PagedKvPool & pool,
                                  const SeqSlotManager & mgr) {
    std::unordered_set<uint32_t> seen_blocks;
    uint32_t total_owned = 0;
    uint32_t active_slots = 0;

    for (int s = 0; s < mgr.slot_count(); ++s) {
        const SeqSlot & slot = mgr.slot(s);
        if (!slot.active()) {
            continue;
        }
        ++active_slots;

        PagedKvSequenceSnapshot snap;
        if (pool.sequence(slot.handle, snap) != PagedKvStatus::Ok) {
            return "pool.sequence failed for active slot";
        }
        uint32_t owned = 0;
        if (pool.owned_block_count(slot.handle, owned) != PagedKvStatus::Ok) {
            return "owned_block_count failed";
        }
        if (slot.parked()) {
            if (owned != 0 || snap.kv_seq_len != 0 ||
                !snap.block_table.empty() || snap.reserved_block_count != 0) {
                return "parked slot retained physical blocks or non-zero kv_seq_len";
            }
            continue;
        }
        if ((int)snap.kv_seq_len != slot.cur_pos) {
            return "kv_seq_len diverged from slot.cur_pos";
        }

        for (uint32_t blk : snap.block_table) {
            if (blk >= pool.physical_block_count()) {
                return "physical block index out of bounds";
            }
            if (!seen_blocks.insert(blk).second) {
                return "double-owned physical block detected across active lanes";
            }
        }

        if (owned != (uint32_t)snap.block_table.size() + snap.reserved_block_count) {
            return "owned_block_count != visible + reserved";
        }
        total_owned += owned;
    }

    if (active_slots != pool.active_sequence_count()) {
        return "active_slots != pool.active_sequence_count()";
    }
    if (total_owned + pool.free_block_count() != pool.physical_block_count()) {
        return "block conservation violated (leak or over-accounting)";
    }
    return {};
}

std::string run_dst_seed(uint64_t seed) {
    constexpr uint32_t kBlockSize = 16;
    constexpr uint32_t kPhysicalBlocks = 32;  // 512 pool tokens
    constexpr uint32_t kSlots = 8;
    constexpr int kMaxCtx = 160;
    constexpr int kSteps = 500;

    PagedKvPool pool(kPhysicalBlocks, kSlots, kBlockSize);
    SeqSlotManager mgr(pool, kMaxCtx);
    SimReservationState res_state;
    SimEngine sim_engine;
    std::vector<SimTxn> slot_txns(kSlots);
    std::vector<uint64_t> arrival_order(kSlots, 0);

    std::mt19937_64 rng(seed);
    uint64_t next_req_id = 1;
    uint64_t next_order = 1;
    uint64_t next_ticket = 1;
    size_t rr_cursor = 0;
    int completed_requests = 0;

    const StepPlanLimits plan_limits{
        /*max_prefill_sequences=*/4,
        /*max_prefill_tokens_per_sequence=*/32,
        /*max_prefill_tokens_total=*/64,
        /*prefill_allocation_quantum=*/16,
    };

    for (int step = 0; step < kSteps; ++step) {
        // 1. Random arrivals (including occasional oversized prompts).
        const int arrivals = (int)(rng() % 3);
        for (int a = 0; a < arrivals; ++a) {
            const bool oversized = (rng() % 20) == 0;
            const int prompt_len = oversized
                ? (kMaxCtx + 16)
                : (int)(8 + (rng() % 72));
            const auto prompt = make_prompt(prompt_len, (int32_t)(next_req_id & 0xFF));
            auto admit = mgr.admit(next_req_id++, prompt, SamplerCfg{});
            if (oversized) {
                if (admit.status == SeqEngine::AdmitResult::Status::admitted) {
                    return "oversized prompt was unexpectedly admitted";
                }
                continue;
            }
            if (admit.status == SeqEngine::AdmitResult::Status::admitted) {
                const int s = admit.slot;
                arrival_order[(size_t)s] = next_order++;

                // Optional restored-prefix seeding on fresh slot.
                if (prompt_len >= 24 && (rng() % 3) == 0) {
                    const int restored = 16;
                    auto seeded = mgr.seed_restored_prefix(s, restored);
                    if (!seeded.ok || (int)seeded.rows.size() != restored ||
                        mgr.slot(s).cur_pos != restored) {
                        return "seed_restored_prefix failed on fresh admitted slot";
                    }
                }

                // Arm an optional prefix capture transaction for this slot.
                if ((rng() % 2) == 0) {
                    const int cut = std::min(prompt_len, 16);
                    PrefixCaptureTicket tk;
                    tk.id = next_ticket++;
                    tk.checkpoint = {Checkpoint::of_cache_slot(s).id, cut};
                    slot_txns[(size_t)s] = SimTxn(
                        SimReservation(res_state, s, cut), sim_engine, tk);
                    if (!slot_txns[(size_t)s].active()) {
                        return "newly armed SimTxn was not active";
                    }
                }
            }
        }

        // 2. Advance chunked prefills via plan_prefill_slices.
        std::vector<PrefillCandidate> candidates;
        for (int s = 0; s < mgr.slot_count(); ++s) {
            if (mgr.is_prefilling(s)) {
                candidates.push_back({s, arrival_order[(size_t)s]});
            }
        }
        auto slices = plan_prefill_slices(candidates, plan_limits, rr_cursor++);
        for (const auto & slice : slices) {
            const int s = slice.slot;
            const int rem = mgr.slot(s).prompt_len - mgr.slot(s).cur_pos;
            const int take = std::min(rem, slice.max_tokens);
            if (take > 0) {
                auto chunk = mgr.append_prefill(s, take);
                if (!chunk.ok || (int)chunk.rows.size() != take) {
                    return "append_prefill failed within admitted reservation";
                }
            }
            if (slot_txns[(size_t)s].active() &&
                mgr.slot(s).cur_pos >= slot_txns[(size_t)s].target_tokens()) {
                PrefixStoreEvent ev;
                const uint64_t roll = rng() % 10;
                ev.ticket.id = next_ticket - 1;  // exercises mismatch handling too
                if (roll < 6) {
                    ev.status = PrefixStoreEvent::Status::saved;
                    ev.bytes = 2048;
                } else {
                    ev.status = PrefixStoreEvent::Status::failed;
                    ev.error = "simulated copy error";
                }
                slot_txns[(size_t)s].resolve(ev, mgr.slot(s).sample_history);
                if (slot_txns[(size_t)s].active()) {
                    return "SimTxn remained active after resolve";
                }
            }
            if (mgr.slot(s).cur_pos == mgr.slot(s).prompt_len) {
                mgr.commit_prefill(s);
                if (!mgr.slot(s).decoding()) {
                    return "slot did not enter decode after commit_prefill";
                }
            }
        }

        // 3. Batched decode step (with occasional speculative rollback).
        std::vector<int> growth((size_t)mgr.slot_count(), 0);
        for (int s = 0; s < mgr.slot_count(); ++s) {
            if (mgr.slot(s).decoding() && mgr.slot(s).cur_pos < kMaxCtx) {
                growth[(size_t)s] = 1;
            }
        }
        if (mgr.reserve_decode(growth)) {
            for (int s = 0; s < mgr.slot_count(); ++s) {
                if (growth[(size_t)s] == 0) continue;
                auto app = mgr.append_token(s, (int32_t)(100 + (step & 0x3F)));
                if (!app.ok) {
                    return "append_token failed after reserve_decode succeeded";
                }
                if ((rng() % 7) == 0) {
                    if (!mgr.rollback_step(s)) {
                        return "rollback_step failed for staged decode token";
                    }
                } else {
                    mgr.commit_step(s);
                    if (mgr.slot(s).cur_pos >= kMaxCtx ||
                        mgr.slot(s).generated_tokens() >= 24) {
                        slot_txns[(size_t)s].cancel();
                        mgr.retire(s);
                        ++completed_requests;
                    }
                }
            }
        }

        // 4. Random park/resume (suspend or evict-for-recompute) and cancellations.
        for (int s = 0; s < mgr.slot_count(); ++s) {
            const SeqSlot & slot = mgr.slot(s);
            if (!slot.active()) continue;
            const uint64_t act = rng() % 32;
            if (act == 0) {
                // Abrupt mid-prefill or mid-decode cancellation.
                slot_txns[(size_t)s].cancel();
                mgr.retire(s);
            } else if (act == 1 && slot.decoding()) {
                if (!mgr.detach_kv(s) || !mgr.slot(s).suspended()) {
                    return "detach_kv failed on decoding slot";
                }
            } else if (act == 2 && slot.decoding()) {
                if (!mgr.evict_for_recompute(s, 777)) {
                    slot_txns[(size_t)s].cancel();
                    mgr.retire(s);
                }
            } else if (slot.suspended() && mgr.kv_restore_feasible(s)) {
                (void)mgr.attach_kv(s);
            } else if (slot.recomputing() && mgr.kv_restore_feasible(s)) {
                (void)mgr.resume_recompute(s);
            }
        }

        if (std::string err = check_pool_invariants(pool, mgr); !err.empty()) {
            return "seed=" + std::to_string(seed) + " step=" + std::to_string(step) + ": " + err;
        }
    }

    // Drain every remaining slot and verify zero leaks across pool and txns.
    for (int s = 0; s < mgr.slot_count(); ++s) {
        slot_txns[(size_t)s].cancel();
        if (mgr.is_active(s)) {
            mgr.retire(s);
        }
    }
    if (pool.active_sequence_count() != 0) {
        return "dangling active sequences after drain";
    }
    if (pool.free_block_count() != pool.physical_block_count()) {
        return "physical block leak after drain";
    }
    if (res_state.active_count != 0) {
        return "leaked prefix capture reservation after drain";
    }
    if (completed_requests <= 0) {
        return "simulation completed zero requests";
    }
    return {};
}

}  // namespace

TEST_CASE(ConcurrencyDstFixture, deterministic_multiseed_simulation) {
    for (uint64_t seed = 1; seed <= 10; ++seed) {
        const std::string err = run_dst_seed(0xC0FFEE0000ULL + seed);
        REQUIRE_EQUAL(err, std::string{});
    }
}

TEST_CASE(ConcurrencyDstFixture, prefix_restore_physical_isolation) {
    PagedKvPool pool(/*physical_block_count=*/16, /*max_sequences=*/4, /*block_size=*/16);
    SeqSlotManager mgr(pool, /*max_ctx=*/96);
    const auto prompt = make_prompt(48, 7);

    auto a = mgr.admit(1, prompt, SamplerCfg{});
    REQUIRE(a.status == SeqEngine::AdmitResult::Status::admitted);
    auto a_chunk = mgr.append_prefill(a.slot, 32);
    REQUIRE(a_chunk.ok);

    auto b = mgr.admit(2, prompt, SamplerCfg{});
    REQUIRE(b.status == SeqEngine::AdmitResult::Status::admitted);
    auto b_seed = mgr.seed_restored_prefix(b.slot, 32);
    REQUIRE(b_seed.ok);

    // Restored slot must occupy strictly disjoint physical rows from source slot.
    std::unordered_set<int64_t> a_rows(a_chunk.rows.begin(), a_chunk.rows.end());
    for (int64_t row : b_seed.rows) {
        CHECK(a_rows.find(row) == a_rows.end());
    }
    REQUIRE_EQUAL(check_pool_invariants(pool, mgr), std::string{});

    mgr.retire(a.slot);
    mgr.retire(b.slot);
    CHECK(pool.active_sequence_count() == 0);
    CHECK(pool.free_block_count() == 16);
}
