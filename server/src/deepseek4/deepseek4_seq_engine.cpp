#include "deepseek4_seq_engine.h"

#include "deepseek4_backend.h"
#include "common/sampler.h"

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

DeepSeek4SeqEngine::DeepSeek4SeqEngine(
        DeepSeek4Backend & backend, PagedKvPool & pool, int max_ctx,
        uint32_t table_stride)
    : b_(backend), slots_(pool, max_ctx), stride_(table_stride),
      host_tables_((size_t)pool.max_sequences() * table_stride, -1) {}

bool DeepSeek4SeqEngine::token_is_eos(int32_t token) const {
    return deepseek4_is_eos_tok(token, b_.w_);
}

StepPlanLimits DeepSeek4SeqEngine::step_plan_limits(
        int decode_rows) const {
    // The gathered graph accepts at most six independent sequences. A live
    // decoder owns one row. On the monolithic exact path a prompt may add up
    // to four chronological rows per step, sixteen rows in total; each row is
    // its own gathered lane and reads the rows before it in graph order.
    // Heterogeneous execution keeps one prompt token per lane.
    const int available = std::max(
        0, DEEPSEEK4_MAX_PAGED_SEQUENCES - decode_rows);
    if (b_.moe_hybrid_) return {available, 1, available, 1};
    const int rows = std::max(0, DEEPSEEK4_MAX_GATHERED_ROWS - decode_rows);
    const int sequences = std::min(available, rows);
    return {sequences, DEEPSEEK4_MAX_PROMPT_ROWS_PER_STEP, rows,
            DEEPSEEK4_MAX_PROMPT_ROWS_PER_STEP};
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    using AdmitStatus = AdmitResult::Status;
    AdmitResult result = slots_.admit(
        request_id, prompt, sampler);
    if (result.status != AdmitStatus::admitted) return result;
    if (result.slot < 0 || result.slot >= slots_.slot_count()) {
        result.status = AdmitStatus::failed;
        result.error = "invalid DeepSeek4 serving slot";
        return result;
    }
    std::fill_n(host_tables_.data() + (size_t)result.slot * stride_,
                stride_, -1);
    reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)result.slot);
    return result;
}

bool DeepSeek4SeqEngine::set_block(int slot, int logical, int32_t physical) {
    if (slot < 0 || slot >= slots_.slot_count() || logical < 0 ||
        (uint32_t) logical >= stride_) return false;
    host_tables_[(size_t)slot * stride_ + (size_t)logical] = physical;
    return true;
}

void DeepSeek4SeqEngine::fail_prefill(
        int slot, std::vector<PrefillOutput> & outputs,
        const std::string & error) {
    std::fprintf(stderr, "[deepseek4-parallel] prefill slot %d: %s\n",
                 slot, error.c_str());
    PrefillOutput out;
    out.slot = slot;
    out.status = PrefillOutput::Status::failed;
    out.error = error;
    outputs.push_back(std::move(out));
}

SeqEngine::StepResult DeepSeek4SeqEngine::step(const StepPlan & plan) {
    using Clock = std::chrono::steady_clock;
    const auto phase_t0 = Clock::now();
    const char * timing_env = std::getenv("DFLASH_DS4_TIMING");
    const bool timing = timing_env && *timing_env && std::strcmp(timing_env, "0") != 0;
    DeepSeek4StepTelemetry telemetry;
    StepResult result;
    const std::vector<StepInput> & inputs = plan.decode;
    const int n_slots = slots_.slot_count();

    auto fail_step = [&](const std::string & error) {
        result.decode.clear();
        result.prefills.clear();
        result.error = error;
        return std::move(result);
    };

    if ((int)inputs.size() != slots_.decoding_count()) {
        return fail_step("decode plan does not cover every live DeepSeek4 slot");
    }
    std::vector<uint8_t> decode_seen((size_t)n_slots, 0);
    for (const StepInput & input : inputs) {
        if (input.slot < 0 || input.slot >= n_slots || input.token < 0 ||
            decode_seen[(size_t)input.slot] ||
            !slots_.is_active(input.slot) ||
            slots_.is_prefilling(input.slot)) {
            return fail_step("invalid or duplicate DeepSeek4 decode row");
        }
        decode_seen[(size_t)input.slot] = 1;
    }

    const StepPlanLimits limits = step_plan_limits((int)inputs.size());
    if ((int)plan.prefills.size() > limits.max_prefill_sequences) {
        return fail_step("DeepSeek4 step exceeds the six-lane graph");
    }
    std::vector<uint8_t> prefill_seen((size_t)n_slots, 0);
    int prefill_rows = 0;
    for (const PrefillSlice & slice : plan.prefills) {
        prefill_rows += slice.max_tokens;
        if (slice.slot < 0 || slice.slot >= n_slots ||
            slice.max_tokens < 1 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            prefill_rows > limits.max_prefill_tokens_total ||
            prefill_seen[(size_t)slice.slot] ||
            decode_seen[(size_t)slice.slot] ||
            !slots_.is_prefilling(slice.slot)) {
            return fail_step("invalid or duplicate DeepSeek4 prefill row");
        }
        prefill_seen[(size_t)slice.slot] = 1;
    }
    if (inputs.empty() && plan.prefills.empty()) return result;

    std::vector<int32_t> lane_tokens;
    std::vector<int64_t> lane_positions;
    std::vector<int32_t> lane_slots;
    std::vector<int> decode_lanes;
    lane_tokens.reserve(inputs.size() + plan.prefills.size());
    lane_positions.reserve(inputs.size() + plan.prefills.size());
    lane_slots.reserve(inputs.size() + plan.prefills.size());
    decode_lanes.reserve(inputs.size());
    result.decode.reserve(inputs.size());
    result.prefills.reserve(plan.prefills.size());

    for (const StepInput & input : inputs) {
        DecodeOutput out;
        out.slot = input.slot;
        const SeqSlotManager::StepAppend append =
            slots_.append_token(input.slot, input.token);
        if (!append.ok) {
            out.failed = true;
            out.error = append.busy
                ? "paged KV pool exhausted during DeepSeek4 decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "DeepSeek4 decode K/V append failed";
            decode_lanes.push_back(-1);
            result.decode.push_back(std::move(out));
            continue;
        }
        if (append.new_block >= 0 &&
            !set_block(input.slot, append.new_block_index,
                       append.new_block)) {
            out.failed = true;
            out.error = "DeepSeek4 decode block-table update failed";
            decode_lanes.push_back(-1);
            result.decode.push_back(std::move(out));
            continue;
        }
        decode_lanes.push_back((int)lane_tokens.size());
        lane_tokens.push_back(input.token);
        lane_positions.push_back(append.position);
        lane_slots.push_back(input.slot);
        result.decode.push_back(std::move(out));
    }

    struct PrefillLane {
        int slot = -1;
        int lane = -1;
        bool commit = false;
    };
    std::vector<PrefillLane> prefill_lanes;
    prefill_lanes.reserve(plan.prefills.size());
    for (const PrefillSlice & slice : plan.prefills) {
        const SeqSlot & before = slots_.slot(slice.slot);
        const int remaining = before.prompt_len - before.cur_pos;
        const int n_rows = std::min(slice.max_tokens, remaining);
        SeqSlotManager::PrefillChunk chunk =
            slots_.append_prefill(slice.slot, n_rows);
        if (n_rows < 1 || !chunk.ok || chunk.rows.size() != (size_t) n_rows) {
            fail_prefill(slice.slot, result.prefills,
                         "DeepSeek4 prefill K/V append failed");
            continue;
        }
        bool blocks_ok = true;
        for (size_t i = 0; i < chunk.new_blocks.size(); ++i) {
            blocks_ok = set_block(slice.slot, chunk.first_new_block + (int) i,
                                  chunk.new_blocks[i]) && blocks_ok;
        }
        if (!blocks_ok) {
            fail_prefill(slice.slot, result.prefills,
                         "DeepSeek4 prefill block-table update failed");
            continue;
        }
        const SeqSlot & slot = slots_.slot(slice.slot);
        const bool commit = slot.cur_pos == slot.prompt_len;
        // One gathered lane per prompt row, in chronological order. Only the
        // chunk's last row can need logits.
        const int first = slot.cur_pos - n_rows;
        for (int pos = first; pos < slot.cur_pos; ++pos) {
            lane_tokens.push_back(slot.sample_history[(size_t) pos]);
            lane_positions.push_back(pos);
            lane_slots.push_back(slice.slot);
        }
        prefill_lanes.push_back(
            {slice.slot, (int) lane_tokens.size() - 1, commit});
    }

    if (lane_tokens.empty()) return result;
    if (lane_tokens.size() > (size_t) DEEPSEEK4_MAX_GATHERED_ROWS) {
        return fail_step("DeepSeek4 gathered step exceeds sixteen rows");
    }

    const auto embed_t0 = Clock::now();
    std::vector<float> embeddings(
        (size_t)b_.w_.n_embd * lane_tokens.size());
    if (!b_.w_.embedder.embed(lane_tokens.data(), (int)lane_tokens.size(),
                              embeddings.data())) {
        return fail_step("DeepSeek4 token embedding failed");
    }

    if (timing) telemetry.embed_us += std::chrono::duration_cast<
        std::chrono::microseconds>(Clock::now() - embed_t0).count();
    std::vector<int32_t> compact_tables(
        lane_tokens.size() * stride_, -1);
    for (size_t lane = 0; lane < lane_slots.size(); ++lane) {
        std::copy_n(
            host_tables_.data() + (size_t)lane_slots[lane] * stride_,
            stride_, compact_tables.data() + lane * stride_);
    }

    std::vector<uint8_t> logit_lanes(lane_tokens.size(), 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
        const int lane = decode_lanes[i];
        if (lane >= 0 &&
            slots_.slot(inputs[i].slot).sampler.needs_logit_processing()) {
            logit_lanes[(size_t) lane] = 1;
        }
    }
    for (const PrefillLane & prefill : prefill_lanes) {
        if (prefill.commit &&
            slots_.slot(prefill.slot).sampler.needs_logit_processing()) {
            logit_lanes[(size_t) prefill.lane] = 1;
        }
    }

    std::vector<float> logits;
    std::vector<int32_t> argmax;
    const bool bucket_history = plan.prefills.empty() && b_.moe_hybrid_;
    if (!deepseek4_paged_gathered_step(
            b_.backend_, b_.cfg_.device.gpu, b_.w_, b_.paged_cache_,
            embeddings.data(), lane_tokens.data(), lane_positions.data(),
            lane_slots.data(), (uint32_t)lane_tokens.size(),
            compact_tables.data(), stride_, bucket_history,
            logit_lanes.data(), logits, argmax,
            b_.moe_hybrid_.get(), b_.routing_stats_.get(),
            timing ? &telemetry : nullptr)) {
        return fail_step("DeepSeek4 gathered paged graph failed");
    }

    const auto sample_t0 = Clock::now();
    auto sample_lane = [&](int slot_id, int lane) {
        SeqSlot & slot = slots_.slot(slot_id);
        if (!slot.sampler.needs_logit_processing()) {
            return argmax[(size_t)lane];
        }
        return sample_logits(
            logits.data() + (size_t)lane * b_.w_.n_vocab,
            b_.w_.n_vocab, slot.sampler, slot.sample_history, slot.rng);
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        const int lane = decode_lanes[i];
        if (lane < 0) continue;
        DecodeOutput & out = result.decode[i];
        slots_.commit_step(out.slot);
        out.token = sample_lane(out.slot, lane);
    }
    for (const PrefillLane & prefill : prefill_lanes) {
        PrefillOutput out;
        out.slot = prefill.slot;
        if (prefill.commit) {
            out.status = PrefillOutput::Status::completed;
            out.token = sample_lane(prefill.slot, prefill.lane);
            slots_.commit_prefill(prefill.slot);
        }
        result.prefills.push_back(std::move(out));
    }
    if (timing) {
        telemetry.sample_us += std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - sample_t0).count();
        const char * phase = plan.prefills.empty() ? "paged-decode"
            : inputs.empty() ? "paged-prefill" : "paged-mixed";
        log_deepseek4_step_telemetry(phase, (int)lane_tokens.size(), 1,
            std::chrono::duration<double>(Clock::now() - phase_t0).count(), telemetry);
    }
    return result;
}

void DeepSeek4SeqEngine::retire(int slot) {
    if (!slots_.is_active(slot)) return;
    slots_.retire(slot);
    reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)slot);
    std::fill_n(host_tables_.data() + (size_t)slot * stride_, stride_, -1);
}

} // namespace dflash::common
