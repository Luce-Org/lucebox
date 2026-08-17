// SeqEngine — concurrent serving over decode slots (iteration-level
// scheduling), the boundary between the HTTP scheduler and a backend that can
// hold several live sequences at once.
//
// A backend qualifies when it can keep several independent sequences in a
// paged KV cache and execute a batched decode step. Any additional
// per-sequence model state is owned by the concrete engine, not by this
// interface. admit() claims a slot and queues its prompt without compute.
// Each step() then advances a scheduler-selected cohort of prompt slices
// alongside the complete live decode batch. Once a prefill completes, the
// scheduler advances that slot one token per step(), feeding each sampled
// token back as the next step's input — which is what lets it override a token
// (thinking-budget force-close) before it is committed to the cache.
//
// The split of duties is deliberate and is the reason this interface exists
// apart from ModelBackend:
//   scheduler   policy — who gets admitted, when a slot stops, what reaches
//               the client, how a slow reader is handled
//   engine      mechanism — KV blocks, backend-owned per-sequence state, the
//               batched forward, sampling
// Nothing above this interface knows about block tables, and nothing below it
// knows about sockets.
//
// Threading contract: every call comes from ONE thread — the same one that
// calls ModelBackend::generate() — so implementations need no locking.
//
// ── Adding a backend ────────────────────────────────────────────────────
// Implement this interface next to the model;
// qwen35/concurrency/qwen35_seq_engine.* is the worked example. Return it
// from ModelBackend::seq_engine(). Nothing else in the server changes:
// scheduler_loop() takes over the worker thread
// as soon as seq_engine() returns non-null.
//
// Reuse only genuinely model-neutral pieces such as PagedKvPool. Prompt and
// slot lifecycle state belongs beside the backend that interprets it; a new
// engine supplies that host record together with its device-side prefill,
// batched forward, and metadata uploads.
//
// What must never reach this interface, or anything above it: block tables,
// graph shapes, and per-sequence model state (recurrent/SSM/conv tensors,
// cache layout). A model that seems to need SeqEngine widened to serve
// concurrently is the signal that the split has slipped — the model-specific
// part belongs inside the engine, not in the contract.
//
// test/seq_engine_contract.h drives an engine through exactly the call
// sequence the scheduler makes and returns the violations it finds; run a new
// engine through it before wiring it up.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/sampler.h"

namespace dflash::common {

// Model-neutral prefill planning for continuous batching. The scheduler owns
// arrival order and fairness; the engine advertises the useful work envelope
// and lowers the selected slices into its model-specific graph.
struct PrefillCandidate {
    int slot = -1;
    uint64_t order = 0;
};

struct PrefillSlice {
    int slot = -1;
    int max_tokens = 0;
};

struct StepPlanLimits {
    int max_prefill_sequences = 1;
    int max_prefill_tokens_per_sequence = 512;
    int max_prefill_tokens_total = 512;
    int prefill_allocation_quantum = 512;
};

// Select the oldest eligible sequences, then distribute the step's token
// capacity in engine-owned quanta. Strict FIFO determines cohort membership;
// a rotating cursor prevents the oldest member from always winning a partial
// final round.
inline std::vector<PrefillSlice> plan_prefill_slices(
        const std::vector<PrefillCandidate> & candidates,
        const StepPlanLimits & limits,
        size_t round_robin_start = 0) {
    std::vector<PrefillSlice> slices;
    if (candidates.empty() || limits.max_prefill_sequences <= 0 ||
        limits.max_prefill_tokens_per_sequence <= 0 ||
        limits.max_prefill_tokens_total <= 0 ||
        limits.prefill_allocation_quantum <= 0) {
        return slices;
    }

    std::vector<PrefillCandidate> ordered = candidates;
    std::stable_sort(ordered.begin(), ordered.end(),
        [](const PrefillCandidate & a, const PrefillCandidate & b) {
            if (a.order != b.order) return a.order < b.order;
            return a.slot < b.slot;
        });

    const int selected = std::min(
        (int)ordered.size(), limits.max_prefill_sequences);
    const int64_t selected_capacity =
        (int64_t)selected * limits.max_prefill_tokens_per_sequence;
    int budget = (int)std::min<int64_t>(
        limits.max_prefill_tokens_total, selected_capacity);
    slices.reserve((size_t)selected);
    for (int i = 0; i < selected; ++i) {
        slices.push_back({ordered[(size_t)i].slot, 0});
    }

    while (budget > 0) {
        bool granted = false;
        for (size_t offset = 0; offset < slices.size(); ++offset) {
            const size_t idx =
                (round_robin_start + offset) % slices.size();
            PrefillSlice & slice = slices[idx];
            const int room =
                limits.max_prefill_tokens_per_sequence - slice.max_tokens;
            if (room <= 0) continue;
            const int grant = std::min({
                limits.prefill_allocation_quantum, room, budget});
            if (grant <= 0) continue;
            slice.max_tokens += grant;
            budget -= grant;
            granted = true;
            if (budget == 0) break;
        }
        if (!granted) break;
    }

    slices.erase(
        std::remove_if(slices.begin(), slices.end(),
            [](const PrefillSlice & slice) { return slice.max_tokens <= 0; }),
        slices.end());
    return slices;
}

class SeqEngine {
public:
    virtual ~SeqEngine() = default;

    // Number of decode slots served concurrently. Fixed for the engine's
    // lifetime: the scheduler sizes its own per-slot array from this once and
    // then indexes it by the slot id admit() returns.
    virtual int slot_count() const = 0;
    // Per-sequence logical context bound. The scheduler owns generation
    // policy and clamps its output cap against this value after admission.
    virtual int max_context() const = 0;

    struct AdmitResult {
        enum class Status {
            admitted,
            busy,
            failed,
        };

        Status status = Status::failed;
        int slot = -1;
        std::string error;
    };

    // Admit one request into a free slot and queue its prompt for chunked
    // prefill. No model compute is performed here. Implementations may reserve
    // persistent capacity atomically so that every admitted prompt can finish;
    // subsequent step() calls advance scheduler-selected prompt slices.
    //
    // `sampler` is the only source of truth for how the slot samples:
    // sampler.needs_logit_processing() selects CPU sampling over GPU argmax
    // AND decides whether sampler.seed is honoured. There is deliberately no
    // separate do_sample flag — a second copy of that one fact is something
    // an engine can disagree with, and the failure mode is silent (a seeded
    // request sampling nondeterministically, with no error anywhere).
    virtual AdmitResult admit(uint64_t request_id,
                              const std::vector<int32_t> & prompt,
                              const SamplerCfg & sampler) = 0;

    struct StepInput {
        int     slot  = -1;
        int32_t token = -1;   // token to commit at this slot's next position
    };
    struct DecodeOutput {
        int     slot   = -1;
        int32_t token  = -1;  // newly sampled token (pending until next step)
        bool    failed = false;
        // Present when failed=true so the scheduler can report an honest
        // per-request error instead of silently truncating generation.
        std::string error;
    };

    struct PrefillOutput {
        enum class Status {
            advanced,
            completed,
            failed,
        };

        int slot = -1;
        Status status = Status::advanced;
        // Present only for completed: the request's first sampled token,
        // pending until the scheduler feeds it into the next decode step.
        int32_t token = -1;
        // Present only for failed.
        std::string error;
    };

    // One scheduler iteration owns both kinds of logical work. `decode` must
    // contain every currently decoding slot exactly once; `prefills` is the
    // bounded subset of pending prompt work selected by scheduler policy.
    // Device graph shapes, staging indices, cache blocks, and model state stay
    // behind the engine boundary.
    struct StepPlan {
        std::vector<StepInput>    decode;
        std::vector<PrefillSlice> prefills;
    };

    struct StepResult {
        std::vector<DecodeOutput>  decode;
        std::vector<PrefillOutput> prefills;
        // A non-empty error is fatal for the whole live cohort and carries
        // no usable row output. Validation
        // failures occur before mutation; a device/build/compute failure may
        // leave backend state partially advanced, so the caller must retire
        // every live sequence before invoking step() again.
        std::string error;

        bool ok() const { return error.empty(); }
    };

    // Useful per-step work envelope at the requested live decode width. The
    // scheduler fills this capacity and distributes it fairly; an engine may
    // advertise different sequence, per-sequence, and total-token limits for
    // idle, mixed, or larger decode buckets.
    virtual StepPlanLimits step_plan_limits(int decode_rows) const = 0;

    // A successful result returns one decode output for every decode input and
    // one explicit advanced/completed/failed result for every selected
    // prefill. Invalid plans return a fatal error without advancing state.
    // Runtime failures are terminal for the live cohort and may follow partial
    // backend mutation, but expose no consumable payload.
    virtual StepResult step(const StepPlan & plan) = 0;

    // Release a slot's KV blocks and mark it free. Safe on failed slots.
    virtual void retire(int slot) = 0;

    // EOS check for scheduler-side stop decisions.
    virtual bool token_is_eos(int32_t token) const = 0;
};

// Validate the model-neutral step protocol before the scheduler consumes any
// output. Malformed row ownership is fatal because re-feeding a token after an
// omitted output would silently corrupt that sequence.
inline std::string validate_step_result(
        const SeqEngine::StepPlan & plan,
        const SeqEngine::StepResult & result,
        int slot_count) {
    if (slot_count < 1) return "slot count must be positive";
    const bool payload_empty = result.decode.empty() && result.prefills.empty();
    if (!result.error.empty()) {
        return payload_empty ? std::string{}
                             : "fatal result exposes partial payload";
    }

    std::vector<uint8_t> decode_planned((size_t)slot_count, 0);
    std::vector<uint8_t> prefill_planned((size_t)slot_count, 0);
    for (const SeqEngine::StepInput & input : plan.decode) {
        if (input.slot < 0 || input.slot >= slot_count || input.token < 0)
            return "decode plan contains an invalid row";
        if (decode_planned[(size_t)input.slot])
            return "decode plan contains a duplicate slot";
        decode_planned[(size_t)input.slot] = 1;
    }
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= slot_count ||
            slice.max_tokens <= 0)
            return "prefill plan contains an invalid slice";
        if (decode_planned[(size_t)slice.slot] ||
            prefill_planned[(size_t)slice.slot])
            return "step plan assigns a slot more than once";
        prefill_planned[(size_t)slice.slot] = 1;
    }

    std::vector<uint8_t> decode_seen((size_t)slot_count, 0);
    for (const SeqEngine::DecodeOutput & output : result.decode) {
        if (output.slot < 0 || output.slot >= slot_count ||
            !decode_planned[(size_t)output.slot])
            return "decode output names an unplanned slot";
        if (decode_seen[(size_t)output.slot])
            return "step returned duplicate decode outputs";
        if (output.failed && (output.token >= 0 || output.error.empty()))
            return "failed decode has invalid payload";
        if (!output.failed && (output.token < 0 || !output.error.empty()))
            return "successful decode has invalid payload";
        decode_seen[(size_t)output.slot] = 1;
    }

    using PrefillStatus = SeqEngine::PrefillOutput::Status;
    std::vector<uint8_t> prefill_seen((size_t)slot_count, 0);
    for (const SeqEngine::PrefillOutput & output : result.prefills) {
        if (output.slot < 0 || output.slot >= slot_count ||
            !prefill_planned[(size_t)output.slot])
            return "prefill output names an unselected slot";
        if (prefill_seen[(size_t)output.slot])
            return "step returned duplicate prefill outputs";
        if (output.status != PrefillStatus::advanced &&
            output.status != PrefillStatus::completed &&
            output.status != PrefillStatus::failed)
            return "prefill output has an unknown status";
        if (output.status == PrefillStatus::advanced &&
            (output.token >= 0 || !output.error.empty()))
            return "advanced prefill carries completion payload";
        if (output.status == PrefillStatus::completed &&
            (output.token < 0 || !output.error.empty()))
            return "completed prefill has invalid payload";
        if (output.status == PrefillStatus::failed &&
            (output.token >= 0 || output.error.empty()))
            return "failed prefill has invalid payload";
        prefill_seen[(size_t)output.slot] = 1;
    }

    for (const SeqEngine::StepInput & input : plan.decode)
        if (!decode_seen[(size_t)input.slot])
            return "step omitted an output for a decode slot";
    for (const PrefillSlice & slice : plan.prefills)
        if (!prefill_seen[(size_t)slice.slot])
            return "step omitted an output for a selected prefill";
    if (payload_empty && (!plan.decode.empty() || !plan.prefills.empty()))
        return "valid planned work returned no output";
    return {};
}

}  // namespace dflash::common
