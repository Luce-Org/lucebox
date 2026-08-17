// Host-side executable contract for common/concurrency/seq_engine.h.
//
// The checker deliberately knows only logical slots, decode inputs, and
// scheduler-selected prefill slices. It exercises the largest useful cohort
// up to two sequences allowed by the advertised limits, mixed
// decode/prefill, validation failures, retryable blocking, and slot reuse
// without importing a model graph or cache representation.

#pragma once

#include "common/concurrency/seq_engine.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

inline std::vector<std::string> check_seq_engine_contract(SeqEngine & engine) {
    std::vector<std::string> violations;
    auto require = [&violations](bool ok, const char * message) {
        if (!ok) violations.emplace_back(message);
    };

    const int n_slots = engine.slot_count();
    require(n_slots >= 2,
            "slot_count() must be at least 2 for the concurrency contract");
    if (n_slots < 2) return violations;

    const StepPlanLimits idle_limits = engine.step_plan_limits(0);
    const StepPlanLimits mixed_limits = engine.step_plan_limits(1);
    const auto supports_one_prefill = [](const StepPlanLimits & limits) {
        return limits.max_prefill_sequences >= 1 &&
               limits.max_prefill_tokens_per_sequence >= 1 &&
               limits.max_prefill_tokens_total >= 1 &&
               limits.prefill_allocation_quantum >= 1;
    };
    require(supports_one_prefill(idle_limits),
            "idle step_plan_limits() must permit one prefill token");
    require(supports_one_prefill(mixed_limits),
            "mixed step_plan_limits() must permit one prefill token");
    require(engine.max_context() >= 3,
            "max_context() must fit the contract-check prompts");
    if (!supports_one_prefill(idle_limits) ||
        !supports_one_prefill(mixed_limits) || engine.max_context() < 3) {
        return violations;
    }

    // Exercise at most two concurrent prefills, but never infer K=2 support
    // from slot_count(): sequence count and the total-token cap are separate
    // engine capabilities.
    const int idle_cohort_size = std::min({
        2,
        n_slots,
        idle_limits.max_prefill_sequences,
        idle_limits.max_prefill_tokens_total,
    });

    const SamplerCfg greedy{};
    SamplerCfg seeded{};
    seeded.temp = 0.7f;
    seeded.seed = 20260809;

    std::vector<bool> active((size_t)n_slots, false);
    std::vector<bool> decoding((size_t)n_slots, false);
    std::vector<int> remaining((size_t)n_slots, 0);
    std::vector<int32_t> next_token((size_t)n_slots, -1);

    auto retire_all = [&]() {
        for (int slot = 0; slot < n_slots; ++slot) {
            if (active[(size_t)slot]) engine.retire(slot);
            active[(size_t)slot] = false;
            decoding[(size_t)slot] = false;
            remaining[(size_t)slot] = 0;
        }
    };

    auto record_admit = [&](uint64_t request_id,
                            const std::vector<int32_t> & prompt,
                            const SamplerCfg & sampler) {
        const SeqEngine::AdmitResult result =
            engine.admit(request_id, prompt, sampler);
        const bool admitted =
            result.status == SeqEngine::AdmitResult::Status::admitted;
        require(admitted, "admit() must succeed while a slot is free");
        if (!admitted) return -1;
        require(result.slot >= 0 && result.slot < n_slots,
                "admit() returned an unknown slot");
        if (result.slot < 0 || result.slot >= n_slots) return -1;
        require(!active[(size_t)result.slot],
                "admit() reused a live slot");
        if (active[(size_t)result.slot]) return -1;
        active[(size_t)result.slot] = true;
        decoding[(size_t)result.slot] = false;
        remaining[(size_t)result.slot] = (int)prompt.size();
        return result.slot;
    };

    auto slice_limit = [&](const SeqEngine::StepPlan & plan) {
        return engine.step_plan_limits((int)plan.decode.size())
            .max_prefill_tokens_per_sequence;
    };

    // Validate and apply one successful logical step to the checker mirror.
    // Contract prefills below request one token at a time, so advanced means
    // exactly one prompt token without reintroducing a progress counter.
    auto apply_progress = [&](const SeqEngine::StepPlan & plan,
                              const SeqEngine::StepResult & result) {
        const std::string protocol_error =
            validate_step_result(plan, result, n_slots);
        if (!protocol_error.empty()) {
            violations.emplace_back(protocol_error);
            return false;
        }
        require(result.ok(), "valid planned work must succeed");
        if (!result.ok()) return false;

        std::vector<bool> decode_answered((size_t)n_slots, false);
        std::vector<bool> prefill_answered((size_t)n_slots, false);

        for (const SeqEngine::DecodeOutput & output : result.decode) {
            require(!output.failed,
                    "contract-check decode work must not fail");
            if (output.slot < 0 || output.slot >= n_slots ||
                output.failed) {
                continue;
            }
            decode_answered[(size_t)output.slot] = true;
            next_token[(size_t)output.slot] = output.token;
        }

        using PrefillStatus = SeqEngine::PrefillOutput::Status;
        for (const SeqEngine::PrefillOutput & output : result.prefills) {
            require(output.status != PrefillStatus::failed,
                    "contract-check prefill work must not fail");
            if (output.slot < 0 || output.slot >= n_slots ||
                output.status == PrefillStatus::failed) {
                continue;
            }
            auto selected = std::find_if(
                plan.prefills.begin(), plan.prefills.end(),
                [&](const PrefillSlice & slice) {
                    return slice.slot == output.slot;
                });
            if (selected == plan.prefills.end()) continue;
            require(selected->max_tokens == 1,
                    "contract checker must use unit prefill slices");
            prefill_answered[(size_t)output.slot] = true;
            if (output.status == PrefillStatus::advanced) {
                require(remaining[(size_t)output.slot] > 1,
                        "prefill reported advanced for its final token");
                --remaining[(size_t)output.slot];
            } else {
                require(remaining[(size_t)output.slot] == 1,
                        "prefill reported completion before its final token");
                remaining[(size_t)output.slot] = 0;
                decoding[(size_t)output.slot] = true;
                next_token[(size_t)output.slot] = output.token;
            }
        }

        for (const SeqEngine::StepInput & input : plan.decode) {
            if (input.slot >= 0 && input.slot < n_slots) {
                require(decode_answered[(size_t)input.slot],
                        "step() left a decoding slot without an output");
            }
        }
        for (const PrefillSlice & slice : plan.prefills) {
            require(slice.max_tokens > 0 &&
                        slice.max_tokens <= slice_limit(plan),
                    "StepPlan prefill slice exceeded engine limits");
            if (slice.slot >= 0 && slice.slot < n_slots) {
                require(prefill_answered[(size_t)slice.slot],
                        "step() left a selected prefill without an output");
            }
        }
        return true;
    };

    auto execute = [&](const SeqEngine::StepPlan & plan) {
        return apply_progress(plan, engine.step(plan));
    };

    auto decode_inputs = [&]() {
        std::vector<SeqEngine::StepInput> inputs;
        for (int slot = 0; slot < n_slots; ++slot) {
            if (active[(size_t)slot] && decoding[(size_t)slot]) {
                inputs.push_back({slot, next_token[(size_t)slot]});
            }
        }
        return inputs;
    };

    // Admit the long prompt in the same idle cohort only when the engine can
    // actually execute two slices. K=1 engines admit it after the short
    // prompt releases its staging resource, then exercise the same mixed path.
    const int short_slot = record_admit(1, {11}, greedy);
    int long_slot = -1;
    if (idle_cohort_size >= 2) {
        long_slot = record_admit(2, {21, 22, 23}, seeded);
    }
    if (short_slot < 0 || (idle_cohort_size >= 2 && long_slot < 0)) {
        retire_all();
        return violations;
    }
    if (long_slot >= 0) {
        require(short_slot != long_slot,
                "two pending admissions must own distinct slots");
    }

    SeqEngine::StepPlan first_plan;
    first_plan.prefills.push_back({short_slot, 1});
    if (long_slot >= 0) {
        first_plan.prefills.push_back({long_slot, 1});
    }
    if (!execute(first_plan)) {
        retire_all();
        return violations;
    }
    require(decoding[(size_t)short_slot],
            "the short prefill must complete in its selected slice");

    if (long_slot < 0) {
        long_slot = record_admit(2, {21, 22, 23}, seeded);
        if (long_slot < 0) {
            retire_all();
            return violations;
        }
    }
    require(!decoding[(size_t)long_slot] &&
                remaining[(size_t)long_slot] > 0,
            "long member must remain pending after the short member completes");

    for (int iteration = 0;
         remaining[(size_t)long_slot] > 0 && iteration < 8;
         ++iteration) {
        SeqEngine::StepPlan mixed;
        mixed.decode = decode_inputs();
        const StepPlanLimits limits =
            engine.step_plan_limits((int)mixed.decode.size());
        require(supports_one_prefill(limits),
                "mixed step_plan_limits() must permit continued prefill");
        if (!supports_one_prefill(limits)) break;
        mixed.prefills.push_back({long_slot, 1});
        if (!execute(mixed)) break;
    }
    require(remaining[(size_t)long_slot] == 0,
            "mixed prefill did not complete within bounded progress steps");

    // Full decode coverage, including the token handoff back into the engine.
    SeqEngine::StepPlan decode_plan;
    decode_plan.decode = decode_inputs();
    require(decode_plan.decode.size() == 2,
            "both completed admissions must enter decode");
    if (!execute(decode_plan)) {
        retire_all();
        return violations;
    }

    // A full engine is retryable admission pressure, not a request error.
    if (n_slots == 2) {
        const SeqEngine::AdmitResult full = engine.admit(3, {31}, greedy);
        require(full.status == SeqEngine::AdmitResult::Status::busy &&
                    full.slot < 0,
                "a full engine must report busy without claiming a slot");
    }

    // Omitting a decoder or attaching prefill work to a decoding slot is a
    // terminal plan-validation failure and must not partially advance state.
    auto require_failed = [&](const SeqEngine::StepPlan & invalid,
                              const char * message) {
        const SeqEngine::StepResult result = engine.step(invalid);
        require(!result.ok(), message);
        require(!result.error.empty(),
                "failed step must explain the validation error");
        require(result.decode.empty() && result.prefills.empty(),
                "failed step must not report partial progress");
    };

    SeqEngine::StepPlan omitted;
    omitted.decode.push_back(decode_plan.decode.front());
    require_failed(omitted,
                   "step() must reject a plan that omits a decoding slot");

    SeqEngine::StepPlan invalid_prefill = decode_plan;
    invalid_prefill.prefills.push_back({short_slot, 1});
    require_failed(invalid_prefill,
                   "step() must reject prefill work for a decoding slot");

    // Start fresh and complete the advertised idle cohort in one step. When
    // K=2 is available, this still catches engines that accidentally retain a
    // scalar completion/output path.
    retire_all();
    const int simultaneous_a = record_admit(10, {41}, greedy);
    int simultaneous_b = -1;
    if (idle_cohort_size >= 2) {
        simultaneous_b = record_admit(11, {51}, seeded);
    }
    if (simultaneous_a >= 0 &&
        (idle_cohort_size < 2 || simultaneous_b >= 0)) {
        SeqEngine::StepPlan simultaneous;
        simultaneous.prefills.push_back({simultaneous_a, 1});
        if (simultaneous_b >= 0) {
            simultaneous.prefills.push_back({simultaneous_b, 1});
        }
        execute(simultaneous);
        require(decoding[(size_t)simultaneous_a],
                "selected prefill must report completion");
        if (simultaneous_b >= 0) {
            require(decoding[(size_t)simultaneous_b],
                    "one K=2 step must report simultaneous completions");
        }
    }

    // Retire is idempotent, frees capacity, and cancels unfinished prefill.
    const int freed = simultaneous_a;
    if (freed >= 0) {
        engine.retire(freed);
        engine.retire(freed);
        active[(size_t)freed] = false;
        decoding[(size_t)freed] = false;
        const int replacement = record_admit(12, {61, 62}, greedy);
        require(replacement >= 0,
                "admit() must reuse capacity after retire()");
        if (replacement >= 0) {
            engine.retire(replacement);
            engine.retire(replacement);
            active[(size_t)replacement] = false;
            decoding[(size_t)replacement] = false;
            remaining[(size_t)replacement] = 0;
        }
    }

    retire_all();
    const SeqEngine::StepResult idle = engine.step({});
    require(idle.ok(), "step() with no work must succeed");
    require(idle.decode.empty() && idle.prefills.empty() &&
                idle.error.empty(),
            "idle result must not retain work or errors");

    const int reused = record_admit(13, {71}, greedy);
    require(reused >= 0,
            "an engine must admit again after every slot retires");
    retire_all();
    return violations;
}

}  // namespace dflash::common
