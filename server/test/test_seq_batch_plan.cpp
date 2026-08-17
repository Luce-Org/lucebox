#include "common/concurrency/seq_engine.h"
#include "host_check.h"

#include <cstdio>
#include <limits>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

int main() {
    const std::vector<PrefillCandidate> pending{
        {7, 30}, {2, 10}, {5, 20}, {1, 40},
    };

    StepPlanLimits idle_limits{/*max sequences=*/2,
                               /*per sequence=*/2048,
                               /*total=*/4096,
                               /*allocation quantum=*/512};
    StepPlanLimits mixed_limits{/*max sequences=*/2,
                                /*per sequence=*/512,
                                /*total=*/1024,
                                /*allocation quantum=*/512};

    auto idle = plan_prefill_slices(
        pending, idle_limits);
    CHECK(idle.size() == 2);
    CHECK(idle[0].slot == 2);
    CHECK(idle[1].slot == 5);
    CHECK(idle[0].max_tokens == 2048);
    CHECK(idle[1].max_tokens == 2048);

    auto mixed = plan_prefill_slices(
        pending, mixed_limits);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].max_tokens == 512);
    CHECK(mixed[1].max_tokens == 512);

    mixed_limits.max_prefill_tokens_per_sequence = 256;
    mixed_limits.max_prefill_tokens_total = 512;
    mixed = plan_prefill_slices(pending, mixed_limits);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].max_tokens == 256);
    CHECK(mixed[1].max_tokens == 256);

    mixed_limits.max_prefill_tokens_total = 301;
    mixed = plan_prefill_slices(pending, mixed_limits);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].slot == 2);
    CHECK(mixed[0].max_tokens == 256);
    CHECK(mixed[1].slot == 5);
    CHECK(mixed[1].max_tokens == 45);

    auto rotated = plan_prefill_slices(
        pending, mixed_limits, /*round_robin_start=*/1);
    CHECK(rotated.size() == 2);
    CHECK(rotated[0].slot == 2 && rotated[0].max_tokens == 45);
    CHECK(rotated[1].slot == 5 && rotated[1].max_tokens == 256);

    // A budget smaller than one quantum advances one lane; rotation prevents
    // the oldest lane from winning every step.
    mixed_limits.max_prefill_tokens_total = 200;
    auto clamped0 = plan_prefill_slices(
        pending, mixed_limits, 0);
    auto clamped1 = plan_prefill_slices(
        pending, mixed_limits, 1);
    CHECK(clamped0.size() == 1 && clamped0[0].slot == 2 &&
          clamped0[0].max_tokens == 200);
    CHECK(clamped1.size() == 1 && clamped1[0].slot == 5 &&
          clamped1[0].max_tokens == 200);

    // The packed Qwen policy fills all eight idle lanes with one 512-token
    // segment, while a mixed step rotates four such segments fairly.
    const std::vector<PrefillCandidate> packed{
        {0, 0}, {1, 1}, {2, 2}, {3, 3},
        {4, 4}, {5, 5}, {6, 6}, {7, 7},
    };
    StepPlanLimits packed_mixed{/*max sequences=*/8,
                                /*per sequence=*/512,
                                /*total=*/2048,
                                /*allocation quantum=*/512};
    auto packed0 = plan_prefill_slices(packed, packed_mixed, 0);
    CHECK(packed0.size() == 4);
    for (int i = 0; i < 4; ++i) {
        CHECK(packed0[(size_t)i].slot == i);
        CHECK(packed0[(size_t)i].max_tokens == 512);
    }
    auto packed4 = plan_prefill_slices(packed, packed_mixed, 4);
    CHECK(packed4.size() == 4);
    for (int i = 0; i < 4; ++i) {
        CHECK(packed4[(size_t)i].slot == i + 4);
        CHECK(packed4[(size_t)i].max_tokens == 512);
    }

    packed_mixed.prefill_allocation_quantum = 0;
    CHECK(plan_prefill_slices(packed, packed_mixed).empty());

    mixed_limits.max_prefill_tokens_per_sequence = 0;
    CHECK(plan_prefill_slices(
        pending, mixed_limits).empty());

    idle_limits.max_prefill_sequences = 0;
    CHECK(plan_prefill_slices(
        pending, idle_limits).empty());
    CHECK(plan_prefill_slices({}, idle_limits).empty());

    StepPlanLimits large_limits{/*max sequences=*/2,
                                /*per sequence=*/std::numeric_limits<int>::max(),
                                /*total=*/std::numeric_limits<int>::max(),
                                /*allocation quantum=*/std::numeric_limits<int>::max()};
    auto large = plan_prefill_slices(pending, large_limits);
    CHECK(large.size() == 1);
    CHECK(large[0].slot == 2);
    CHECK(large[0].max_tokens == std::numeric_limits<int>::max());

    // The same model-neutral layer validates engine row ownership before the
    // scheduler mutates socket/request state.
    SeqEngine::StepPlan work;
    work.decode = {{0, 7}};
    work.prefills = {{1, 4}};

    SeqEngine::StepResult good;
    good.decode.push_back({0, 11, false, {}});
    good.prefills.push_back({
        1, SeqEngine::PrefillOutput::Status::advanced, -1, {}});
    CHECK(validate_step_result(work, good, 2).empty());

    SeqEngine::StepResult complete = good;
    complete.prefills[0] = {
        1, SeqEngine::PrefillOutput::Status::completed, 12, {}};
    CHECK(validate_step_result(work, complete, 2).empty());

    SeqEngine::StepResult missing_decode = good;
    missing_decode.decode.clear();
    CHECK(!validate_step_result(work, missing_decode, 2).empty());

    SeqEngine::StepResult duplicate_decode = good;
    duplicate_decode.decode.push_back({0, 12, false, {}});
    CHECK(!validate_step_result(work, duplicate_decode, 2).empty());

    SeqEngine::StepResult missing_prefill = good;
    missing_prefill.prefills.clear();
    CHECK(!validate_step_result(work, missing_prefill, 2).empty());

    // A selected prefill may terminate with a per-request error; the
    // scheduler retires only that slot.
    SeqEngine::StepResult prefill_failure = good;
    prefill_failure.prefills[0] = {
        1, SeqEngine::PrefillOutput::Status::failed, -1, "prefill failed"};
    CHECK(validate_step_result(work, prefill_failure, 2).empty());

    SeqEngine::StepResult bad_row_failure = prefill_failure;
    bad_row_failure.prefills.back().error.clear();
    CHECK(!validate_step_result(work, bad_row_failure, 2).empty());

    SeqEngine::StepResult unknown_prefill = good;
    unknown_prefill.prefills[0].status =
        static_cast<SeqEngine::PrefillOutput::Status>(-1);
    CHECK(!validate_step_result(work, unknown_prefill, 2).empty());

    SeqEngine::StepResult bad_decode = good;
    bad_decode.decode[0] = {0, -1, true, {}};
    CHECK(!validate_step_result(work, bad_decode, 2).empty());

    SeqEngine::StepResult failed_with_token = good;
    failed_with_token.decode[0] = {0, 12, true, "decode failed"};
    CHECK(!validate_step_result(work, failed_with_token, 2).empty());

    SeqEngine::StepResult success_with_error = good;
    success_with_error.decode[0].error = "contradictory diagnostic";
    CHECK(!validate_step_result(work, success_with_error, 2).empty());

    SeqEngine::StepResult failed;
    failed.error = "device compute failed";
    CHECK(validate_step_result(work, failed, 2).empty());
    failed.decode.push_back({0, -1, true, "partial"});
    CHECK(!validate_step_result(work, failed, 2).empty());

    SeqEngine::StepResult idle_result;
    CHECK(validate_step_result({}, idle_result, 2).empty());
    CHECK(!validate_step_result(work, idle_result, 2).empty());

    SeqEngine::StepPlan duplicate_plan = work;
    duplicate_plan.decode.push_back({0, 8});
    CHECK(!validate_step_result(duplicate_plan, good, 2).empty());

    std::printf("test_seq_batch_plan: %d checks passed\n", g_checks);
    return 0;
}
