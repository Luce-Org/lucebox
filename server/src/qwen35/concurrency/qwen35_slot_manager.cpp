#include "qwen35_slot_manager.h"

#include <algorithm>
#include <cstdio>

namespace dflash::common {

Qwen35SlotManager::Qwen35SlotManager(PagedKvPool & pool, int max_ctx)
    : pool_(pool), max_ctx_(max_ctx) {
    slots_.assign(pool.max_sequences(), Qwen35Slot{});
}

int Qwen35SlotManager::decoding_count() const {
    int n = 0;
    for (const Qwen35Slot & s : slots_) {
        n += s.decoding() ? 1 : 0;
    }
    return n;
}

uint32_t Qwen35SlotManager::decode_headroom_capacity(int logical_tokens) const {
    const uint64_t extended =
        static_cast<uint64_t>(std::max(0, logical_tokens)) +
        pool_.block_size();
    return static_cast<uint32_t>(std::min<uint64_t>(
        static_cast<uint64_t>(max_ctx_), extended));
}

bool Qwen35SlotManager::capacity_fits_pool(uint32_t token_capacity) const {
    const uint64_t blocks = token_capacity == 0 ? 0 :
        1 + (static_cast<uint64_t>(token_capacity) - 1) /
                pool_.block_size();
    return blocks <= pool_.physical_block_count();
}

PagedKvStatus Qwen35SlotManager::protect_decode_headroom() {
    struct TopUp {
        PagedKvSequenceHandle handle;
        uint32_t token_capacity = 0;
    };

    std::vector<TopUp> topups;
    topups.reserve(slots_.size());
    uint64_t total_additional = 0;
    const uint64_t block_size = pool_.block_size();
    for (const Qwen35Slot & slot : slots_) {
        if (!slot.decoding()) continue;
        const uint32_t capacity = decode_headroom_capacity(slot.cur_pos);
        if (!capacity_fits_pool(capacity)) continue;

        uint32_t owned_blocks = 0;
        const PagedKvStatus status =
            pool_.owned_block_count(slot.handle, owned_blocks);
        if (status != PagedKvStatus::Ok) return status;
        const uint64_t target_blocks = capacity == 0 ? 0 :
            1 + (static_cast<uint64_t>(capacity) - 1) / block_size;
        if (target_blocks <= owned_blocks) continue;
        const uint32_t additional =
            static_cast<uint32_t>(target_blocks - owned_blocks);
        total_additional += additional;
        topups.push_back({slot.handle, capacity});
    }

    // Preflight the whole cohort before moving a block, so a failed admission
    // attempt cannot protect only whichever decoder happened to be visited
    // first.
    if (total_additional > pool_.free_block_count()) {
        return PagedKvStatus::BlocksExhausted;
    }
    for (const TopUp & topup : topups) {
        const PagedKvStatus status =
            pool_.reserve_capacity(topup.handle, topup.token_capacity);
        if (status != PagedKvStatus::Ok) return status;
    }
    return PagedKvStatus::Ok;
}

bool Qwen35SlotManager::is_active(int slot) const {
    return slot >= 0 && slot < (int)slots_.size() &&
           slots_[(size_t)slot].active();
}

bool Qwen35SlotManager::is_prefilling(int slot) const {
    return is_active(slot) && slots_[(size_t)slot].prefilling();
}

bool Qwen35SlotManager::has_prefill_prompt_at_least(int tokens) const {
    if (tokens <= 0) return true;
    return std::any_of(slots_.begin(), slots_.end(),
        [tokens](const Qwen35Slot & slot) {
            return slot.prefilling() && slot.prompt_len >= tokens;
        });
}

SeqEngine::AdmitResult Qwen35SlotManager::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    using AdmitStatus = SeqEngine::AdmitResult::Status;
    SeqEngine::AdmitResult r;
    if (prompt.empty()) {
        r.error = "empty prompt";
        return r;
    }
    if (prompt.size() > static_cast<size_t>(max_ctx_)) {
        r.error = "prompt exceeds max_ctx";
        return r;
    }
    const int prompt_len = static_cast<int>(prompt.size());

    // A prompt larger than the whole pool can NEVER be admitted; waiting
    // for other sequences to drain would stall the queue forever and then
    // fail anyway. Hard-fail it up front instead of reporting busy.
    const uint64_t pool_capacity =
        (uint64_t)pool_.physical_block_count() * pool_.block_size();
    if ((uint64_t)prompt_len > pool_capacity) {
        r.error = "prompt needs " + std::to_string(prompt_len) +
                  " KV tokens but the pool holds " +
                  std::to_string(pool_capacity) +
                  "; raise --kv-pool-tokens or shorten the prompt";
        return r;
    }

    int slot = -1;
    for (int i = 0; i < (int)slots_.size(); i++) {
        if (!slots_[(size_t)i].active()) { slot = i; break; }
    }
    if (slot < 0) {
        r.status = AdmitStatus::busy;
        r.error = "all decode slots are busy";
        return r;
    }

    // A newly freed block belongs to any older decoder missing its rolling
    // next-page reserve before it can belong to this admission.
    const PagedKvStatus headroom_status = protect_decode_headroom();
    if (headroom_status != PagedKvStatus::Ok) {
        r.status = headroom_status == PagedKvStatus::BlocksExhausted
            ? AdmitStatus::busy : AdmitStatus::failed;
        r.error = r.status == AdmitStatus::busy
            ? "existing decoders need the available KV headroom"
            : paged_kv_status_string(headroom_status);
        return r;
    }

    PagedKvSequenceHandle handle;
    uint32_t reservation_capacity =
        decode_headroom_capacity(prompt_len);
    if (!capacity_fits_pool(reservation_capacity)) {
        // The prompt itself fits, but this physical pool can never hold its
        // following page. Preserve useful prompt-only behavior and report
        // decode exhaustion later if the sequence reaches that boundary.
        reservation_capacity = static_cast<uint32_t>(prompt_len);
    }
    const PagedKvStatus status = pool_.acquire_reserved(
        request_id, reservation_capacity, handle);
    if (status != PagedKvStatus::Ok) {
        r.status = status == PagedKvStatus::SequenceSlotsExhausted ||
                           status == PagedKvStatus::BlocksExhausted
            ? AdmitStatus::busy : AdmitStatus::failed;
        r.error = status == PagedKvStatus::BlocksExhausted
            ? "not enough unreserved KV blocks for the prompt and decode headroom"
            : paged_kv_status_string(status);
        return r;
    }

    Qwen35Slot & s = slots_[(size_t)slot];
    s.phase = Qwen35SlotPhase::prefill;
    s.handle = handle;
    s.cur_pos = 0;
    s.prompt_len = prompt_len;
    s.sampler = sampler;
    s.sample_history = prompt;
    // Same predicate the engine uses to pick CPU sampling over GPU argmax:
    // a seed only means anything when the sampler actually draws.
    if (sampler.needs_logit_processing() && sampler.seed != 0) {
        s.rng.seed(sampler.seed);
    } else {
        s.rng.seed(std::random_device{}());
    }

    r.status = AdmitStatus::admitted;
    r.slot = slot;
    return r;
}

Qwen35SlotManager::PrefillChunk Qwen35SlotManager::append_prefill(
        int slot, int n_tokens) {
    PrefillChunk out;
    if (!is_prefilling(slot) || n_tokens < 1) return out;

    Qwen35Slot & s = slots_[(size_t)slot];
    if (s.cur_pos > s.prompt_len ||
        n_tokens > s.prompt_len - s.cur_pos) {
        return out;
    }

    PagedKvAppendResult app = pool_.append(s.handle, (uint32_t)n_tokens);
    if (!app) {
        // Admission reserved the whole prompt. Treat exhaustion here as a
        // broken invariant, not a retryable condition: retrying a batch of
        // all-prefill slots without any decoder able to retire would livelock.
        if (app.status == PagedKvStatus::BlocksExhausted) {
            std::fprintf(stderr,
                "[parallel] reserved prefill capacity missing for slot %d\n",
                slot);
        }
        return out;
    }

    out.rows.reserve(app.write_slots.size());
    for (const PagedKvWriteSlot & write : app.write_slots) {
        out.rows.push_back((int64_t)write.physical_token_index);
        if (write.block_offset == 0) {
            if (out.first_new_block < 0) {
                out.first_new_block =
                    (int)(write.logical_position / pool_.block_size());
            }
            out.new_blocks.push_back((int32_t)write.physical_block);
        }
    }
    s.cur_pos += n_tokens;
    out.ok = true;
    return out;
}

void Qwen35SlotManager::commit_prefill(int slot) {
    if (!is_prefilling(slot)) return;
    Qwen35Slot & s = slots_[(size_t)slot];
    if (s.cur_pos != s.prompt_len) return;
    s.phase = Qwen35SlotPhase::decode;
}

Qwen35SlotManager::StepAppend Qwen35SlotManager::append_token(int slot,
                                                        int32_t fed_token) {
    StepAppend out;
    if (!is_active(slot) || !slots_[(size_t)slot].decoding()) return out;
    Qwen35Slot & s = slots_[(size_t)slot];
    if (s.cur_pos >= max_ctx_) {
        // No context left; the scheduler should have stopped this slot.
        return out;
    }
    PagedKvAppendResult app = pool_.append(
        s.handle, 1, /*only_first_last_slots=*/true);
    if (!app || app.token_count != 1 ||
        app.last.logical_position != (uint32_t)s.cur_pos) {
        out.busy = app.status == PagedKvStatus::BlocksExhausted;
        return out;
    }
    s.sample_history.push_back(fed_token);

    out.ok = true;
    out.physical_row = (int64_t)app.last.physical_token_index;
    out.position = s.cur_pos;
    if ((uint32_t)s.cur_pos % pool_.block_size() == 0) {
        out.new_block = (int32_t)app.last.physical_block;
        out.new_block_index = s.cur_pos / (int)pool_.block_size();
    }
    return out;
}

void Qwen35SlotManager::commit_step(int slot) {
    if (!is_active(slot)) return;
    slots_[(size_t)slot].cur_pos += 1;
}

void Qwen35SlotManager::retire(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    Qwen35Slot & s = slots_[(size_t)slot];
    if (!s.active()) return;
    const PagedKvStatus status = pool_.release(s.handle);
    if (status != PagedKvStatus::Ok && status != PagedKvStatus::StaleHandle) {
        std::fprintf(stderr, "[parallel] slot %d release failed: %s\n",
                     slot, paged_kv_status_string(status));
    }
    s = Qwen35Slot{};
}

}  // namespace dflash::common
