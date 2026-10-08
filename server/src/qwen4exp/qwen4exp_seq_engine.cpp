#include "qwen4exp_seq_engine.h"

#include "common/sampler.h"
#include "qwen4exp_graph.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <limits>
#include <utility>

namespace luce::common {

namespace {
uint32_t pool_blocks(int max_ctx, size_t slots) {
    if (max_ctx <= 0 || slots == 0 ||
        slots > std::numeric_limits<uint32_t>::max()) return 0;
    const uint64_t blocks_per_slot =
        (uint64_t(max_ctx) + 255) / 256;
    const uint64_t blocks = blocks_per_slot * slots;
    return blocks <= std::numeric_limits<uint32_t>::max()
        ? (uint32_t)blocks : 0;
}
} // namespace

Qwen4ExpSeqEngine::Qwen4ExpSeqEngine(
        ggml_backend_t backend, const Qwen4ExpWeights & weights,
        std::vector<Qwen4ExpCache *> caches, int max_ctx, int prefill_granule,
        size_t prefix_allowance, int verify_width, AdaptiveSpecWidth * mtp_width,
        SpecWidthCostMemory * mtp_costs)
    : backend_(backend), weights_(weights), caches_(std::move(caches)),
      pool_(pool_blocks(max_ctx, caches_.size()),
            (uint32_t)caches_.size(), 256),
      slots_(pool_, max_ctx),
      prefill_granule_(std::max(prefill_granule, 1)),
      prefix_allowance_(prefix_allowance),
      checkpoints_((size_t)kPrefixCheckpoints),
      prefill_cuts_(caches_.size()),
      mtp_slot_(verify_width != 1 && !caches_.empty() && caches_[0] &&
                qwen4exp_verify_supported(*caches_[0]) ? 0 : -1),
      verify_width_(verify_width),
      mtp_width_(mtp_width ? mtp_width : &own_mtp_width_),
      mtp_costs_(mtp_costs ? mtp_costs : &own_mtp_costs_) {}

Qwen4ExpSeqEngine::~Qwen4ExpSeqEngine() {
    ggml_backend_synchronize(backend_);
    for (Qwen4ExpSnapshot & checkpoint : checkpoints_) free_qwen4exp_snapshot(checkpoint);
    clear_qwen4exp_batched_decode_workspace(decode_workspace_);
}

bool Qwen4ExpSeqEngine::token_is_eos(int32_t token) const {
    return token == weights_.eos_id || token == weights_.eos_chat_id;
}

SeqEngine::AdmitResult Qwen4ExpSeqEngine::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    return admit_cold(request_id, prompt, sampler);
}

SeqEngine::AdmitResult Qwen4ExpSeqEngine::admit_cold(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    for (int32_t token : prompt) {
        if (token < 0 || token >= weights_.n_vocab) {
            AdmitResult invalid;
            invalid.status = AdmitResult::Status::failed;
            invalid.error = "qwen4exp prompt contains an invalid token id";
            return invalid;
        }
    }
    AdmitResult result = slots_.admit(request_id, prompt, sampler);
    if (result.status != AdmitResult::Status::admitted) return result;
    if (!backend_ || result.slot < 0 ||
        result.slot >= (int)caches_.size() || !caches_[(size_t)result.slot] ||
        caches_[(size_t)result.slot]->max_ctx < max_context()) {
        slots_.retire(result.slot);
        result.status = AdmitResult::Status::failed;
        result.error = "invalid qwen4exp full-cache slot";
        return result;
    }
    if (pool_.reserve_capacity(slots_.slot(result.slot).handle,
                               (uint32_t)max_context()) != PagedKvStatus::Ok) {
        slots_.retire(result.slot);
        result.status = AdmitResult::Status::busy;
        result.error = "qwen4exp full-context reservation unavailable";
        return result;
    }
    reset_qwen4exp_state(backend_, *caches_[(size_t)result.slot]);
    ggml_backend_synchronize(backend_);
    prefill_cuts_[(size_t)result.slot].clear();
    if (result.slot == mtp_slot_) {
        mtp_ = MtpState{};
        mtp_.live = true;
        mtp_.context = (int)prompt.size();
        if (verify_width_ == 0) {   // as the single-slot loop starts a request
            mtp_costs_->load(*mtp_width_, mtp_.context);
            mtp_width_->carry_acceptance(QWEN4EXP_MTP_CARRIED_TRIALS);
        }
    }
    return result;
}

size_t Qwen4ExpSeqEngine::estimate_prefix_store_bytes(int tokens) const {
    if (caches_.empty() || !caches_[0] || tokens <= 0) return 0;
    size_t host = 0;
    const size_t device = qwen4exp_snapshot_bytes(backend_, *caches_[0], tokens, &host);
    if (!device) return 0;
    return device + host + (size_t(tokens) + std::max(0, weights_.ple_ngram_size - 1)) * sizeof(int32_t);
}

int Qwen4ExpSeqEngine::checkpoint_index(PrefixStoreRef checkpoint) const {
    if (!checkpoint.valid() || checkpoint.id > (uint64_t)kPrefixCheckpoints) return -1;
    return (int)checkpoint.id - 1;
}

void Qwen4ExpSeqEngine::discard_prefix_store(PrefixStoreRef checkpoint) {
    const int index = checkpoint_index(checkpoint);
    if (index >= 0 && checkpoints_[(size_t)index].buf &&
        checkpoints_[(size_t)index].cur_pos == checkpoint.tokens)
        free_qwen4exp_snapshot(checkpoints_[(size_t)index]);
}

// A checkpoint restores only into a cold slot and only when its tokens are the
// prompt's own prefix; any failure leaves the slot cold.
bool Qwen4ExpSeqEngine::restore_prefix(int slot, const std::vector<int32_t> & prompt,
                                       PrefixStoreRef checkpoint) {
    const int index = checkpoint_index(checkpoint);
    if (index < 0 || checkpoint.tokens >= (int)prompt.size()) return false;
    const Qwen4ExpSnapshot & s = checkpoints_[(size_t)index];
    if (!s.buf || s.cur_pos != checkpoint.tokens ||
        s.tokens.size() != (size_t)checkpoint.tokens ||
        !std::equal(s.tokens.begin(), s.tokens.end(), prompt.begin())) return false;
    Qwen4ExpCache & cache = *caches_[(size_t)slot];
    if (!restore_qwen4exp_snapshot(backend_, s, cache)) return false;
    if (!slots_.seed_restored_prefix(slot, checkpoint.tokens).ok) {
        reset_qwen4exp_state(backend_, cache);
        ggml_backend_synchronize(backend_);
        return false;
    }
    return true;
}

SeqEngine::AdmitResult Qwen4ExpSeqEngine::admit_with_prefix(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler, const PrefixStorePlan & plan) {
    AdmitResult result = admit_cold(request_id, prompt, sampler);
    if (result.status != AdmitResult::Status::admitted) return result;
    const int slot = result.slot;
    slots_.slot(slot).pending_capture = {};
    prefill_cuts_[(size_t)slot] = plan.restore_points;
    std::sort(prefill_cuts_[(size_t)slot].begin(), prefill_cuts_[(size_t)slot].end());
    if (plan.restore.valid()) {
        const auto started = std::chrono::steady_clock::now();
        const bool restored = restore_prefix(slot, prompt, plan.restore);
        const uint64_t elapsed_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (restored) {
            result.prefix_store.restored = plan.restore;
            std::fprintf(stderr, "[qwen4exp-seq] restored checkpoint=%llu slot=%d tokens=%d time_ms=%.1f\n",
                (unsigned long long)plan.restore.id, slot, plan.restore.tokens, (double)elapsed_us / 1000.0);
        } else {
            discard_prefix_store(plan.restore);
            result.prefix_store.invalidated = plan.restore;
        }
        result.prefix_store.restore_attempted = true;
        result.prefix_store.restore_elapsed_us = elapsed_us;
    }
    const SeqSlot & seq = slots_.slot(slot);
    if (!result.prefix_store.invalidated.valid() && plan.capture.valid() &&
        checkpoint_index(plan.capture.checkpoint) >= 0 &&
        plan.capture.checkpoint.tokens > seq.cur_pos &&
        plan.capture.checkpoint.tokens <= seq.prompt_len) {
        slots_.slot(slot).pending_capture = plan.capture;
        result.prefix_store.capture = plan.capture;
    }
    return result;
}

// A prompt chunk of the MTP slot, as the single-slot loop runs it: chunks
// fill the draft layer's K/V inline and keep the last trunk row pending; a
// one-row chunk catches its pair up through the draft layer instead. The
// pending row before the chunk is the cache's (mtp_prev_hidden at
// mtp_prev_pos), so a cache restored from a prefix snapshot keeps drafting; a
// cache without that row prefills plainly and its request decodes undrafted.
bool Qwen4ExpSeqEngine::mtp_prefill(Qwen4ExpCache & cache, const int32_t * tokens, int n, int pos0,
                                    std::vector<float> & logits) {
    const size_t hd = (size_t)weights_.n_embd * weights_.n_hc;
    Qwen4ExpMtpPending & pending = mtp_.pending;
    pending.tok.clear();
    if (pos0 > 0 && cache.mtp_prev_pos != pos0 - 1) {
        mtp_.live = false;
        pending.h.clear();
        return qwen4exp_forward(backend_, weights_, cache, tokens, n, pos0, logits).ok;
    }
    pending.h.resize(pos0 > 0 ? hd : 0);
    if (pos0 > 0) ggml_backend_tensor_get(cache.mtp_prev_hidden, pending.h.data(), 0, hd * sizeof(float));
    pending.pos = std::max(0, pos0 - 1);
    std::vector<float> hidden;
    if (n > 1) {
        if (!qwen4exp_forward(backend_, weights_, cache, tokens, n, pos0, logits, &hidden, false, true).ok)
            return false;
        pending.h.swap(hidden);
        pending.pos = pos0 + n - 1;
        return true;
    }
    if (!qwen4exp_forward(backend_, weights_, cache, tokens, 1, pos0, logits, &hidden).ok) return false;
    if (pos0 > 0) pending.tok.assign(tokens, tokens + 1);
    pending.h.insert(pending.h.end(), hidden.begin(), hidden.end());
    if (!qwen4exp_mtp_catch_up(backend_, weights_, cache, pending, 0) || pending.h.size() != hd) return false;
    ggml_backend_tensor_set(cache.mtp_prev_hidden, pending.h.data(), 0, hd * sizeof(float));
    cache.mtp_prev_pos = pos0;
    return true;
}

bool Qwen4ExpSeqEngine::mtp_eligible(const StepPlan & plan) const {
    return mtp_slot_ >= 0 && mtp_.live && plan.prefills.empty() &&
        plan.decode.size() == 1 && plan.decode[0].slot == mtp_slot_ &&
        plan.decode[0].allow_speculation;
}

// The MTP slot alone takes the single-slot loop's step (qwen4exp_mtp_step). The
// accepted drafts are durable children of the output; the last emitted token
// stays pending for the scheduler, as in one-token decode.
SeqEngine::StepResult Qwen4ExpSeqEngine::mtp_step(const StepInput & input) {
    StepResult result;
    const int id = input.slot;
    Qwen4ExpCache & cache = *caches_[(size_t)id];
    SeqSlot & slot = slots_.slot(id);
    const int pos = slot.cur_pos;
    const auto t0 = std::chrono::steady_clock::now();
    auto verify_builds = [&cache] {
        uint64_t n = 0;
        for (const auto & ws : cache.verify_workspace) n += ws.builds;
        return n;
    };
    const uint64_t builds0 = verify_builds();

    // Rows are sampled as one-token decode would, each after its own fed token, up to EOS.
    const size_t history0 = slot.sample_history.size();
    auto sample_row = [&](int32_t fed, const float * row, int32_t & token) {
        slot.sample_history.push_back(fed);
        token = slot.sampler.needs_logit_processing()
            ? sample_logits(row, weights_.n_vocab, slot.sampler, slot.sample_history, slot.rng)
            : (int32_t)(std::max_element(row, row + weights_.n_vocab) - row);
        return !token_is_eos(token);
    };
    mtp_.pending.tok.push_back(input.token);
    std::vector<float> logits;
    Qwen4ExpMtpStep step;
    const bool ok = qwen4exp_mtp_step(backend_, weights_, cache, pos, input.token, max_context() - pos - 1,
                                      verify_width_ == 0 ? mtp_width_ : nullptr, &mtp_.pending, sample_row,
                                      logits, step);
    slot.sample_history.resize(history0);
    if (!ok) {
        result.error = step.error;
        return result;
    }
    mtp_.pending.tok.pop_back();   // the scheduler feeds it back as the next step's token
    ggml_backend_synchronize(backend_);
    const Qwen4ExpMtpAcceptance & decision = step.decision;
    const int retained = decision.n_emitted;
    std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> fed{};
    fed[0] = input.token;
    std::copy(decision.emitted.begin(), decision.emitted.begin() + retained - 1, fed.begin() + 1);
    cache.cur_pos = pos + retained;
    const auto appended = slots_.append_tokens(id, fed.data(), retained);
    if (!appended.ok || appended.position != pos) {
        result.error = "qwen4exp MTP reservation failed";
        return result;
    }
    slots_.commit_step(id);

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    mtp_.draft_s += step.draft_s;
    mtp_.verify_s += step.verify_s;
    mtp_.verify_builds += verify_builds() - builds0;
    ++mtp_.steps;
    mtp_.drafts += step.k;
    mtp_.accepted += decision.n_accepted;
    mtp_.tokens += retained;
    mtp_.decode_s += ms / 1e3;

    DecodeOutput output;
    output.slot = id;
    output.token = decision.emitted[(size_t)retained - 1];
    output.committed_tokens.assign(decision.emitted.begin(), decision.emitted.begin() + retained - 1);
    result.decode.push_back(std::move(output));
    return result;
}

PrefixStoreEvent Qwen4ExpSeqEngine::capture_prefix(int slot, PrefixCaptureTicket ticket) {
    PrefixStoreEvent event;
    event.ticket = ticket;
    event.status = PrefixStoreEvent::Status::failed;
    const int index = checkpoint_index(ticket.checkpoint);
    if (!ticket.valid() || index < 0 || !slots_.is_prefilling(slot) ||
        slots_.slot(slot).cur_pos != ticket.checkpoint.tokens ||
        caches_[(size_t)slot]->cur_pos != ticket.checkpoint.tokens) {
        event.error = "invalid prefix capture boundary";
        return event;
    }
    const auto started = std::chrono::steady_clock::now();
    auto elapsed_us = [&started] {
        return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    };
    // The capture replaces checkpoint `index`; every other resident one counts against the allowance.
    size_t resident = 0;
    for (size_t i = 0; i < checkpoints_.size(); ++i)
        if (i != (size_t)index && checkpoints_[i].buf)
            resident += ggml_backend_buffer_get_size(checkpoints_[i].buf);
    const size_t need = estimate_prefix_store_bytes(ticket.checkpoint.tokens);
    if (!need || resident > prefix_allowance_ || need > prefix_allowance_ - resident) {
        event.elapsed_us = elapsed_us();
        event.error = "qwen4exp prefix checkpoint allowance exhausted";
        return event;
    }
    Qwen4ExpSnapshot & s = checkpoints_[(size_t)index];
    if (!save_qwen4exp_snapshot(backend_, *caches_[(size_t)slot], s)) {
        free_qwen4exp_snapshot(s);
        event.elapsed_us = elapsed_us();
        event.error = "qwen4exp prefix capture failed";
        return event;
    }
    const std::vector<int32_t> & history = slots_.slot(slot).sample_history;
    s.tokens.assign(history.begin(), history.begin() + ticket.checkpoint.tokens);
    event.status = PrefixStoreEvent::Status::saved;
    event.bytes = ggml_backend_buffer_get_size(s.buf);
    event.elapsed_us = elapsed_us();
    std::fprintf(stderr, "[qwen4exp-seq] saved checkpoint=%llu slot=%d tokens=%d bytes=%zu time_ms=%.1f\n",
        (unsigned long long)ticket.checkpoint.id, slot, ticket.checkpoint.tokens, event.bytes,
        (double)event.elapsed_us / 1000.0);
    return event;
}

int Qwen4ExpSeqEngine::prefill_segment(int slot, int max_tokens) const {
    const SeqSlot & s = slots_.slot(slot);
    int end = std::min({s.prompt_len, s.cur_pos + max_tokens, s.cur_pos + prefill_granule_});
    for (int cut : prefill_cuts_[(size_t)slot]) {
        if (cut > s.cur_pos) {
            end = std::min(end, cut);
            break;
        }
    }
    const PrefixCaptureTicket & capture = s.pending_capture;
    if (capture.valid() && s.cur_pos < capture.checkpoint.tokens)
        end = std::min(end, capture.checkpoint.tokens);
    return end - s.cur_pos;
}

// One granule per prompt forward, whatever the step carries. While slots
// decode, one granule in total per step bounds the hold on the live streams.
StepPlanLimits Qwen4ExpSeqEngine::step_plan_limits(int decode_rows) const {
    const int prefill_slots = slot_count() - std::clamp(decode_rows, 0, slot_count());
    if (decode_rows > 0) {
        const int sequences = std::min(prefill_slots, 1);
        return {sequences, prefill_granule_, sequences * prefill_granule_, prefill_granule_};
    }
    return {prefill_slots, prefill_granule_, prefill_slots * prefill_granule_, prefill_granule_};
}

bool Qwen4ExpSeqEngine::reserve_decode(const StepPlan & plan) {
    if ((int)plan.decode.size() != slots_.decoding_count()) return false;
    std::vector<int> growth((size_t)slot_count(), 0);
    std::vector<uint8_t> assigned((size_t)slot_count(), 0);
    for (const StepInput & input : plan.decode) {
        if (input.slot < 0 || input.slot >= slot_count() ||
            input.token < 0 || input.token >= weights_.n_vocab ||
            growth[(size_t)input.slot] != 0 ||
            !slots_.slot(input.slot).decoding()) return false;
        growth[(size_t)input.slot] = 1;
        assigned[(size_t)input.slot] = 1;
    }
    if (mtp_eligible(plan)) {
        const int slot = plan.decode[0].slot;
        growth[(size_t)slot] = std::max(1, std::min(caches_[(size_t)slot]->mtp_draft + 1,
                                                    max_context() - slots_.slot(slot).cur_pos));
    }
    const StepPlanLimits limits = step_plan_limits((int)plan.decode.size());
    if (plan.prefills.size() > (size_t)limits.max_prefill_sequences) return false;
    int prefill_total = 0;
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= slot_count() ||
            slice.max_tokens < 1 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            (prefill_total += slice.max_tokens) > limits.max_prefill_tokens_total ||
            assigned[(size_t)slice.slot] ||
            !slots_.is_prefilling(slice.slot)) return false;
        assigned[(size_t)slice.slot] = 1;
    }
    return slots_.reserve_decode(growth);
}

SeqEngine::StepResult Qwen4ExpSeqEngine::step(const StepPlan & plan) {
    StepResult result;
    const int n = slot_count();
    auto fail = [&result](const char * message) -> StepResult {
        result.decode.clear();
        result.prefills.clear();
        result.error = message;
        return std::move(result);
    };

    if (!backend_ ||
        (int)plan.decode.size() != slots_.decoding_count())
        return fail("qwen4exp decode plan does not cover live slots");

    std::vector<uint8_t> seen((size_t)n, 0);
    for (const StepInput & input : plan.decode) {
        if (input.slot < 0 || input.slot >= n || input.token < 0 ||
            input.token >= weights_.n_vocab || seen[(size_t)input.slot] ||
            !slots_.slot(input.slot).decoding() ||
            !caches_[(size_t)input.slot] ||
            caches_[(size_t)input.slot]->cur_pos !=
                slots_.slot(input.slot).cur_pos ||
            slots_.slot(input.slot).cur_pos >= max_context())
            return fail("invalid or duplicate qwen4exp decode row");
        seen[(size_t)input.slot] = 1;
    }
    const StepPlanLimits limits = step_plan_limits((int)plan.decode.size());
    if (plan.prefills.size() > (size_t)limits.max_prefill_sequences)
        return fail("qwen4exp prefill plan exceeds available slots");
    for (const PrefillSlice & slice : plan.prefills) {
        const int remaining = slice.slot >= 0 && slice.slot < n &&
            slots_.is_prefilling(slice.slot)
            ? slots_.slot(slice.slot).prompt_len - slots_.slot(slice.slot).cur_pos
            : 0;
        if (slice.slot < 0 || slice.slot >= n || slice.max_tokens < 1 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            remaining <= 0 ||
            seen[(size_t)slice.slot] || !slots_.is_prefilling(slice.slot) ||
            !caches_[(size_t)slice.slot] ||
            caches_[(size_t)slice.slot]->cur_pos !=
                slots_.slot(slice.slot).cur_pos)
            return fail("invalid qwen4exp prefill slice");
        seen[(size_t)slice.slot] = 1;
    }
    if (plan.decode.empty() && plan.prefills.empty()) return result;
    if (mtp_eligible(plan)) return mtp_step(plan.decode[0]);

    struct PendingPrefill {
        int slot;
        bool complete;
        size_t segment;
    };
    std::vector<PendingPrefill> pending_prefills;
    std::vector<std::vector<int32_t>> segment_tokens;
    std::vector<Qwen4ExpForwardSegment> forward_segments;
    std::vector<size_t> decode_rows;
    std::vector<int> decode_positions;
    segment_tokens.reserve(plan.decode.size() + plan.prefills.size());
    forward_segments.reserve(plan.decode.size() + plan.prefills.size());
    decode_rows.reserve(plan.decode.size());
    decode_positions.reserve(plan.decode.size());

    for (const StepInput & input : plan.decode) {
        const auto appended = slots_.append_token(input.slot, input.token);
        if (!appended.ok) return fail("qwen4exp decode reservation failed");
        segment_tokens.emplace_back(1, input.token);
        decode_positions.push_back(appended.position);
        decode_rows.push_back(forward_segments.size());
        forward_segments.push_back({
            caches_[(size_t)input.slot], segment_tokens.back().data(),
            1, appended.position});
    }
    for (const PrefillSlice & slice : plan.prefills) {
        const SeqSlot & before = slots_.slot(slice.slot);
        const int count = prefill_segment(slice.slot, slice.max_tokens);
        if (count <= 0) return fail("qwen4exp prefill made no progress");
        const int pos = before.cur_pos;
        segment_tokens.emplace_back(
            before.sample_history.begin() + pos,
            before.sample_history.begin() + pos + count);
        const SeqSlotManager::PrefillChunk appended =
            slots_.append_prefill(slice.slot, count);
        if (!appended.ok || appended.rows.size() != (size_t) count)
            return fail("qwen4exp prefill reservation failed");
        const size_t segment_index = forward_segments.size();
        forward_segments.push_back({
            caches_[(size_t) slice.slot], segment_tokens.back().data(),
            count, pos});
        const bool complete = slots_.slot(slice.slot).cur_pos ==
                              slots_.slot(slice.slot).prompt_len;
        pending_prefills.push_back({slice.slot, complete, segment_index});
    }

    std::vector<std::vector<float>> logits;
    if (!plan.decode.empty()) {
        std::vector<int32_t> tokens;
        std::vector<int32_t> positions;
        std::vector<Qwen4ExpCache *> caches;
        tokens.reserve(plan.decode.size());
        positions.reserve(plan.decode.size());
        caches.reserve(plan.decode.size());
        int hidden_slot = -1;
        for (size_t i = 0; i < plan.decode.size(); ++i) {
            tokens.push_back(plan.decode[i].token);
            positions.push_back(decode_positions[i]);
            caches.push_back(caches_[(size_t) plan.decode[i].slot]);
            if (plan.decode[i].slot == mtp_slot_ && mtp_.live) hidden_slot = (int)i;
        }
        // The MTP slot's trunk row keeps its draft layer caught up for when it decodes alone again.
        std::vector<float> hidden;
        const auto forward = qwen4exp_forward_batched(
            backend_, weights_, caches.data(), tokens.data(), positions.data(),
            (int)tokens.size(), decode_workspace_, logits, hidden_slot,
            hidden_slot >= 0 ? &hidden : nullptr);
        if (!forward.ok || logits.size() != tokens.size())
            return fail("qwen4exp batched decode forward failed");
        if (hidden_slot >= 0) {
            const size_t hd = (size_t)weights_.n_embd * weights_.n_hc;
            Qwen4ExpMtpPending & pending = mtp_.pending;
            pending.tok.push_back(tokens[(size_t)hidden_slot]);
            if (hidden.size() != hd) {
                mtp_.live = false;   // no trunk row: this request decodes without drafts from here on
            } else {
                pending.h.insert(pending.h.end(), hidden.begin(), hidden.end());
                if (pending.tok.size() >= 64 &&
                    !qwen4exp_mtp_catch_up(backend_, weights_, *caches_[(size_t)mtp_slot_], pending, 0))
                    return fail("qwen4exp MTP catch-up failed");
            }
        }
    }
    logits.resize(forward_segments.size());
    for (size_t i = plan.decode.size(); i < forward_segments.size(); ++i) {
        const auto & segment = forward_segments[i];
        const bool mtp = mtp_slot_ >= 0 && mtp_.live && segment.cache == caches_[(size_t)mtp_slot_];
        if (mtp ? !mtp_prefill(*segment.cache, segment.tokens, segment.n_tokens, segment.pos0, logits[i])
                : !qwen4exp_forward(backend_, weights_, *segment.cache, segment.tokens,
                                    segment.n_tokens, segment.pos0, logits[i]).ok)
            return fail("qwen4exp solo prefill forward failed");
    }
    if (std::any_of(logits.begin(), logits.end(), [this](const auto & row) {
            return row.size() != (size_t) weights_.n_vocab;
        })) return fail("qwen4exp forward returned malformed logits");
    if (!forward_segments.empty()) {
        ggml_backend_synchronize(backend_);
        for (const Qwen4ExpForwardSegment & segment : forward_segments)
            segment.cache->cur_pos = segment.pos0 + segment.n_tokens;
    }

    result.decode.reserve(plan.decode.size());
    for (size_t i = 0; i < plan.decode.size(); ++i) {
        const int slot_id = plan.decode[i].slot;
        SeqSlot & slot = slots_.slot(slot_id);
        slots_.commit_step(slot_id);
        DecodeOutput output;
        output.slot = slot_id;
        const std::vector<float> & row = logits[decode_rows[i]];
        output.token = slot.sampler.needs_logit_processing()
            ? sample_logits(row.data(), weights_.n_vocab, slot.sampler,
                            slot.sample_history, slot.rng)
            : (int32_t)(std::max_element(row.begin(), row.end()) - row.begin());
        result.decode.push_back(std::move(output));
    }
    for (const PendingPrefill & pending : pending_prefills) {
        PrefillOutput output;
        output.slot = pending.slot;
        const PrefixCaptureTicket capture = slots_.slot(pending.slot).pending_capture;
        if (capture.valid() &&
            slots_.slot(pending.slot).cur_pos == capture.checkpoint.tokens) {
            output.prefix_store = capture_prefix(pending.slot, capture);
            slots_.slot(pending.slot).pending_capture = {};
        }
        if (pending.complete) {
            SeqSlot & slot = slots_.slot(pending.slot);
            output.status = PrefillOutput::Status::completed;
            const std::vector<float> & row = logits[pending.segment];
            output.token = slot.sampler.needs_logit_processing()
                ? sample_logits(row.data(), weights_.n_vocab,
                                slot.sampler, slot.sample_history, slot.rng)
                : (int32_t)(std::max_element(row.begin(), row.end()) - row.begin());
            slots_.commit_prefill(pending.slot);
        }
        result.prefills.push_back(std::move(output));
    }
    return result;
}

void Qwen4ExpSeqEngine::retire(int slot) {
    if (slot < 0 || slot >= slot_count()) return;
    if (slot == mtp_slot_ && mtp_.context > 0) {
        if (mtp_.drafts > 0) {
            std::fprintf(stderr,
                "[qwen4exp-mtp] slot=%d drafts=%lld accepted=%lld rate=%.3f tokens_per_step=%.3f "
                "decode=%.2f tok/s mtp_steps=%lld live=%d step_ms=%.1f draft_ms=%.1f verify_ms=%.1f "
                "verify_builds=%llu batched_builds=%llu batched_replays=%llu\n",
                slot, mtp_.drafts, mtp_.accepted, (double)mtp_.accepted / (double)mtp_.drafts,
                mtp_.steps > 0 ? (double)mtp_.tokens / (double)mtp_.steps : 0.0,
                mtp_.decode_s > 0.0 ? (double)mtp_.tokens / mtp_.decode_s : 0.0, mtp_.steps, (int)mtp_.live,
                1e3 * mtp_.decode_s / (double)mtp_.steps, 1e3 * mtp_.draft_s / (double)mtp_.steps,
                1e3 * mtp_.verify_s / (double)mtp_.steps, (unsigned long long)mtp_.verify_builds,
                (unsigned long long)decode_workspace_.builds, (unsigned long long)decode_workspace_.replays);
        }
        if (verify_width_ == 0) mtp_costs_->store(*mtp_width_, mtp_.context);
        mtp_ = MtpState{};
    }
    ggml_backend_synchronize(backend_);
    if (slots_.is_active(slot)) slots_.retire(slot);
    prefill_cuts_[(size_t)slot].clear();
    reset_qwen4exp_state(backend_, *caches_[(size_t)slot]);
    ggml_backend_synchronize(backend_);
}

} // namespace luce::common
