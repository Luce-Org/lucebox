#include "qwen4exp_seq_engine.h"

#include "common/sampler.h"
#include "qwen4exp_graph.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

namespace luce::common {

namespace {
double mono_now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
        std::vector<Qwen4ExpCache *> caches, int max_ctx, int prefill_chunk)
    : backend_(backend), weights_(weights), caches_(std::move(caches)),
      pool_(pool_blocks(max_ctx, caches_.size()),
            (uint32_t)caches_.size(), 256),
      slots_(pool_, max_ctx),
      prefill_chunk_(std::clamp(prefill_chunk, 1, 512)) {}

Qwen4ExpSeqEngine::~Qwen4ExpSeqEngine() {
    ggml_backend_synchronize(backend_);
    clear_qwen4exp_batched_decode_workspace(decode_workspace_);
}

bool Qwen4ExpSeqEngine::token_is_eos(int32_t token) const {
    return token == weights_.eos_id || token == weights_.eos_chat_id;
}

SeqEngine::AdmitResult Qwen4ExpSeqEngine::admit(
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
    return result;
}

StepPlanLimits Qwen4ExpSeqEngine::step_plan_limits(int decode_rows) const {
    // The packed graph can hold 2048 total rows. Give every selected prefill
    // slot its normal chunk, rather than dividing one chunk across the whole
    // cohort; subtract live decode rows so the graph's fixed row ceiling holds.
    constexpr int packed_graph_max_rows = 2048;
    const int decode_capacity = std::clamp(decode_rows, 0, slot_count());
    const int prefill_slots = std::max(0, slot_count() - decode_capacity);
    const int total_prefill_rows = std::min(
        prefill_slots * prefill_chunk_,
        packed_graph_max_rows - decode_capacity);
    return {prefill_slots, prefill_chunk_, total_prefill_rows, 1};
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

    const char * batched_env = std::getenv("QWEN4EXP_BATCHED_DECODE");
    const char * upstream_env = std::getenv("QWEN4EXP_UPSTREAM");
    if (!backend_ || !batched_env || std::atoi(batched_env) == 0 ||
        (upstream_env && std::atoi(upstream_env) != 0) ||
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
        return fail("qwen4exp prefill plan exceeds packed graph capacity");
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

    static const bool telemetry_on = std::getenv("QWEN4EXP_PROF") != nullptr;
    uint64_t telemetry_step = 0;
    const double step_begin_s = telemetry_on ? mono_now_s() : 0.0;
    if (telemetry_on) {
        static uint64_t step_id = 0;
        telemetry_step = ++step_id;
        int live = 0;
        for (int slot = 0; slot < n; ++slot) live += slots_.is_active(slot) ? 1 : 0;
        const uint64_t blocks_per_slot =
            ((uint64_t) max_context() + 255) / 256;
        std::fprintf(stderr,
            "[qwen4exp-seq-step] step=%llu live=%d decode_rows=%zu "
            "prefill_slices=%zu kv_mode=full-cache bookkeeping_blocks=%llu slots=",
            (unsigned long long) telemetry_step, live,
            plan.decode.size(), plan.prefills.size(),
            (unsigned long long) (blocks_per_slot * live));
        bool first_slot = true;
        for (int slot = 0; slot < n; ++slot) {
            if (!slots_.is_active(slot)) continue;
            const SeqSlot & state = slots_.slot(slot);
            std::fprintf(stderr, "%s%d:%s:%d/%d",
                first_slot ? "" : ",", slot,
                state.decoding() ? "decode" : "prefill",
                state.cur_pos, state.prompt_len);
            first_slot = false;
        }
        std::fprintf(stderr, " decode_plan=");
        for (size_t i = 0; i < plan.decode.size(); ++i)
            std::fprintf(stderr, "%s%d@%d", i ? "," : "",
                plan.decode[i].slot,
                slots_.slot(plan.decode[i].slot).cur_pos);
        std::fprintf(stderr, " prefill_plan=");
        for (size_t i = 0; i < plan.prefills.size(); ++i)
            std::fprintf(stderr, "%s%d+%d", i ? "," : "",
                plan.prefills[i].slot, plan.prefills[i].max_tokens);
        std::fprintf(stderr, "\n");
    }

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
        const int count = std::min(slice.max_tokens,
                                   before.prompt_len - before.cur_pos);
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
    if (telemetry_on) {
        int packed_rows = 0;
        for (const Qwen4ExpForwardSegment & segment : forward_segments)
            packed_rows += segment.n_tokens;
        std::fprintf(stderr,
            "[qwen4exp-step-work] step=%llu graph=%s segments=%zu rows=%d decode_rows=%zu prefill_sequences=%zu prefill_tokens=%d\n",
            (unsigned long long) telemetry_step,
            plan.prefills.empty() ? "decode" : "packed",
            forward_segments.size(), packed_rows, plan.decode.size(),
            plan.prefills.size(), packed_rows - (int) plan.decode.size());
    }

    std::vector<std::vector<float>> logits;
    Qwen4ExpForwardResult forward;
    if (!plan.prefills.empty()) {
        forward = qwen4exp_forward_packed(
            backend_, weights_, forward_segments.data(),
            (int) forward_segments.size(), decode_workspace_, logits);
        if (!forward.ok || logits.size() != forward_segments.size())
            return fail("qwen4exp packed prefill/decode forward failed");
    } else if (!plan.decode.empty()) {
        std::vector<int32_t> tokens;
        std::vector<int32_t> positions;
        std::vector<Qwen4ExpCache *> caches;
        tokens.reserve(plan.decode.size());
        positions.reserve(plan.decode.size());
        caches.reserve(plan.decode.size());
        for (size_t i = 0; i < plan.decode.size(); ++i) {
            tokens.push_back(plan.decode[i].token);
            positions.push_back(decode_positions[i]);
            caches.push_back(caches_[(size_t) plan.decode[i].slot]);
        }
        forward = qwen4exp_forward_batched(
            backend_, weights_, caches.data(), tokens.data(), positions.data(),
            (int)tokens.size(), decode_workspace_, logits);
        if (!forward.ok || logits.size() != tokens.size())
            return fail("qwen4exp batched decode forward failed");
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
    if (telemetry_on) {
        for (size_t i = 0; i < result.decode.size(); ++i) {
            const DecodeOutput & output = result.decode[i];
            std::fprintf(stderr,
                "[qwen4exp-output] step=%llu kind=decode slot=%d input_pos=%d token=%d done_abs=%.9f\n",
                (unsigned long long) telemetry_step, output.slot,
                decode_positions[i], output.token, mono_now_s());
        }
    }
    for (const PendingPrefill & pending : pending_prefills) {
        PrefillOutput output;
        output.slot = pending.slot;
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
    if (telemetry_on) {
        for (const PrefillOutput & output : result.prefills) {
            std::fprintf(stderr,
                "[qwen4exp-output] step=%llu kind=prefill slot=%d token=%d status=%d done_abs=%.9f\n",
                (unsigned long long) telemetry_step, output.slot, output.token,
                (int) output.status, mono_now_s());
        }
        std::fprintf(stderr,
            "[qwen4exp-step-done] step=%llu begin_abs=%.9f done_abs=%.9f decode_rows=%zu prefill_slices=%zu\n",
            (unsigned long long) telemetry_step, step_begin_s, mono_now_s(),
            result.decode.size(), result.prefills.size());
    }
    return result;
}

void Qwen4ExpSeqEngine::retire(int slot) {
    if (slot < 0 || slot >= slot_count()) return;
    ggml_backend_synchronize(backend_);
    if (slots_.is_active(slot)) slots_.retire(slot);
    reset_qwen4exp_state(backend_, *caches_[(size_t)slot]);
    ggml_backend_synchronize(backend_);
}

} // namespace luce::common
