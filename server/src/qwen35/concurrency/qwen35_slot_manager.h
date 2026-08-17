// Qwen35SlotManager — complete host-side state for each Qwen serving slot.
//
// Companion of PagedKvPool: the pool hands out sequence handles and physical
// blocks; this class owns everything else a slot needs between admission and
// retirement — the pool-handle lifecycle (including every error path), the
// admission arithmetic (context clamp, prompt reservation, and rolling decode
// headroom), on-demand block allocation, per-slot sampler/RNG/penalty-history
// state, and the position counters.
//
// It deliberately owns NO device state. Prefill/decode allocation returns
// physical rows and block-table deltas as plain vectors. Prompt, KV ownership,
// sampler, and progress live together here; the scheduler keeps
// only its coarse request phase.
//
// Not thread-safe; the single scheduler thread is the only caller.

#pragma once

#include "common/concurrency/paged_kv_pool.h"
#include "common/sampler.h"
#include "common/concurrency/seq_engine.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

enum class Qwen35SlotPhase {
    free,
    prefill,
    decode,
};

struct Qwen35Slot {
    Qwen35SlotPhase phase = Qwen35SlotPhase::free;
    PagedKvSequenceHandle handle;
    // Prompt tokens are the immutable prefix of sample_history. Decode tokens
    // append to the same allocation, avoiding a second full prompt copy.
    int prompt_len = 0;
    int cur_pos = 0;
    SamplerCfg sampler;
    std::mt19937_64 rng{0x9E3779B97F4A7C15ull};
    // Penalty history is recorded as fed rather than sampled: the scheduler
    // may override a sample before the model consumes it.
    std::vector<int32_t> sample_history;

    int generated_tokens() const {
        return sample_history.size() > (size_t)prompt_len
            ? (int)(sample_history.size() - (size_t)prompt_len)
            : 0;
    }

    bool active() const { return phase != Qwen35SlotPhase::free; }
    bool prefilling() const { return phase == Qwen35SlotPhase::prefill; }
    bool decoding() const { return phase == Qwen35SlotPhase::decode; }
};

class Qwen35SlotManager {
public:
    // `max_ctx` is the per-sequence logical bound; slot count comes from the
    // pool's max_sequences. The pool must outlive the manager.
    Qwen35SlotManager(PagedKvPool & pool, int max_ctx);

    // Claim a free slot and atomically reserve all K/V blocks needed by the
    // known prompt plus its next logical decode page when that page can exist
    // in both max_ctx and the physical pool. Existing decoders are topped up
    // first, so a younger admission cannot steal their next-page headroom.
    // Prompts larger than the whole pool hard-fail; temporary capacity pressure
    // reports busy. Seeds the slot RNG from sampler.seed only when the sampler
    // actually draws, else nondeterministically.
    SeqEngine::AdmitResult admit(uint64_t request_id,
                                 const std::vector<int32_t> & prompt,
                                 const SamplerCfg & sampler);

    struct PrefillChunk {
        bool ok = false;
        std::vector<int64_t> rows;
        // Delta to patch into the slot's device block-table column.
        std::vector<int32_t> new_blocks;
        int first_new_block = -1;
    };

    // Append `n_tokens` more prompt rows for a prefilling slot. Physical block
    // ids come from the slot's admission reservation, so any append within the
    // admitted prompt is guaranteed not to wait on another sequence.
    PrefillChunk append_prefill(int slot, int n_tokens);

    // Record a finished prefill and expose the slot to decode.
    void commit_prefill(int slot);

    struct StepAppend {
        bool ok = false;
        bool busy = false;    // no physical block available right now
        int64_t physical_row = -1;
        int position = -1;   // logical position the fed token is written at
        int32_t new_block = -1;
        int new_block_index = -1;
    };

    // Allocate the next decode token's cache row, report any new block-table
    // entry, and log it to sample_history. cur_pos waits for commit_step().
    StepAppend append_token(int slot, int32_t fed_token);

    // The batched step's compute succeeded: cur_pos++.
    void commit_step(int slot);

    // Release the slot's blocks and clear its state. Safe on inactive slots
    // and after a failed admission/prefill.
    void retire(int slot);

    int slot_count() const { return (int)slots_.size(); }
    int max_context() const { return max_ctx_; }
    int decoding_count() const;
    bool is_active(int slot) const;
    bool is_prefilling(int slot) const;
    bool has_prefill_prompt_at_least(int tokens) const;
    Qwen35Slot & slot(int i) { return slots_[(size_t)i]; }
    const Qwen35Slot & slot(int i) const { return slots_[(size_t)i]; }

private:
    // Logical extent whose block count includes the sequence's current pages
    // plus one future page, capped at max_ctx.
    uint32_t decode_headroom_capacity(int logical_tokens) const;
    bool capacity_fits_pool(uint32_t token_capacity) const;

    // Atomically preflight and top up every decoding slot as one cohort before
    // a younger sequence may reserve capacity.
    PagedKvStatus protect_decode_headroom();

    PagedKvPool & pool_;
    int max_ctx_ = 0;
    std::vector<Qwen35Slot> slots_;
};

}  // namespace dflash::common
