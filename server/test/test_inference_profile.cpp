#include "common/observability/inference_profile.h"

#include <cstdint>
#include <cstdio>
#include <string_view>

using namespace dflash::common::observability;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                     __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (false)

namespace {

uint64_t fake_now = 0;
int fake_clock_calls = 0;

uint64_t fake_clock() noexcept {
    ++fake_clock_calls;
    fake_now += 10;
    return fake_now;
}

class FakeSink final : public ProfileSink {
public:
    StepProfile * begin_step(uint32_t live_slots) noexcept override {
        profile.live_slots = live_slots;
        return &profile;
    }
    void commit_step(StepProfile * value) override {
        committed = value == &profile;
    }

    StepProfile profile;
    bool committed = false;
};

}

int main() {
    CHECK(active_profile_sink() == nullptr);
    FakeSink sink;
    {
        ActiveProfileSinkScope active(&sink);
        CHECK(active_profile_sink() == &sink);
        StepProfile * direct = active_profile_sink()->begin_step(1);
        active_profile_sink()->commit_step(direct);
    }
    CHECK(active_profile_sink() == nullptr);
    CHECK(sink.profile.live_slots == 1);
    CHECK(sink.committed);

    fake_now = 0;
    fake_clock_calls = 0;
    {
        PhaseScope scope(nullptr, Phase::TargetCompute, fake_clock);
    }
    CHECK(fake_clock_calls == 0);

    StepProfile profile;
    profile.started_ns = 5;
    {
        PhaseScope scope(&profile, Phase::TargetCompute, fake_clock);
    }
    CHECK(fake_clock_calls == 2);
    CHECK(profile.phase_count == 1);
    CHECK(profile.phases[0].phase == Phase::TargetCompute);
    CHECK(profile.phases[0].start_offset_ns == 5);
    CHECK(profile.phases[0].duration_ns == 10);

    LaneProfile lane;
    lane.request_id = 42;
    lane.slot = 3;
    lane.kind = LaneKind::Decode;
    CHECK(profile.add_lane(lane) != nullptr);
    CHECK(profile.find_lane(3, LaneKind::Decode)->request_id == 42);
    CHECK(profile.find_lane(3, LaneKind::Prefill) == nullptr);

    StepProfile full;
    for (size_t i = 0; i < kMaxProfileLanes; ++i) {
        lane.slot = static_cast<int32_t>(i);
        CHECK(full.add_lane(lane) != nullptr);
    }
    CHECK(full.add_lane(lane) == nullptr);
    CHECK(full.dropped_lanes == 1);

    CHECK(std::string_view(step_path_name(StepPath::Speculative)) ==
          "speculative");
    CHECK(std::string_view(spec_decision_name(
              SpecDecision::PromptWorkPresent)) ==
          "prompt_work_present");
    CHECK(std::string_view(spec_decision_name(SpecDecision::InvalidSlot)) ==
          "invalid_slot");
    CHECK(std::string_view(phase_name(Phase::ClientFlush)) ==
          "client_flush");

    std::printf("test_inference_profile: passed\n");
    return 0;
}
