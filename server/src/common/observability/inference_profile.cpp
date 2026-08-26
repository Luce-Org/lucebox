#include "common/observability/inference_profile.h"

#include <chrono>

namespace dflash::common::observability {
namespace {

thread_local ProfileSink * g_active_profile_sink = nullptr;

}

LaneProfile * StepProfile::add_lane(const LaneProfile & lane) noexcept {
    if (lane_count >= lanes.size()) {
        ++dropped_lanes;
        return nullptr;
    }
    lanes[lane_count] = lane;
    return &lanes[lane_count++];
}

LaneProfile * StepProfile::find_lane(int32_t slot, LaneKind kind) noexcept {
    for (uint32_t i = 0; i < lane_count; ++i) {
        if (lanes[i].slot == slot && lanes[i].kind == kind) {
            return &lanes[i];
        }
    }
    return nullptr;
}

void StepProfile::add_phase(PhaseSpan span) noexcept {
    if (phase_count >= phases.size()) {
        ++dropped_phases;
        return;
    }
    phases[phase_count++] = span;
}

const char * step_path_name(StepPath path) noexcept {
    switch (path) {
    case StepPath::Unknown: return "unknown";
    case StepPath::Packed: return "packed";
    case StepPath::Speculative: return "speculative";
    }
    return "unknown";
}

const char * lane_kind_name(LaneKind kind) noexcept {
    switch (kind) {
    case LaneKind::Decode: return "decode";
    case LaneKind::Prefill: return "prefill";
    }
    return "unknown";
}

const char * spec_decision_name(SpecDecision decision) noexcept {
    switch (decision) {
    case SpecDecision::None: return "none";
    case SpecDecision::Selected: return "selected";
    case SpecDecision::InvalidSlot: return "invalid_slot";
    case SpecDecision::PromptWorkPresent: return "prompt_work_present";
    case SpecDecision::CallerDisallowed: return "caller_disallowed";
    case SpecDecision::FeatureUnavailable: return "feature_unavailable";
    case SpecDecision::SamplingUnsupported: return "sampling_unsupported";
    case SpecDecision::InsufficientContext: return "insufficient_context";
    case SpecDecision::DraftPrepareFailed: return "draft_prepare_failed";
    }
    return "unknown";
}

const char * phase_name(Phase phase) noexcept {
    switch (phase) {
    case Phase::SchedulerPlan: return "scheduler_plan";
    case Phase::InputStaging: return "input_staging";
    case Phase::DraftPrepare: return "draft_prepare";
    case Phase::DraftCompute: return "draft_compute";
    case Phase::ProposalSelect: return "proposal_select";
    case Phase::TargetGraphBuild: return "target_graph_build";
    case Phase::MetadataUpload: return "metadata_upload";
    case Phase::TargetCompute: return "target_compute";
    case Phase::ReadbackSync: return "readback_sync";
    case Phase::Acceptance: return "acceptance";
    case Phase::StatePromotion: return "state_promotion";
    case Phase::SamplingCommit: return "sampling_commit";
    case Phase::OutputProcessing: return "output_processing";
    case Phase::ClientFlush: return "client_flush";
    }
    return "unknown";
}

uint64_t steady_time_ns() noexcept {
    const auto elapsed = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

PhaseScope::PhaseScope(
        StepProfile * profile, Phase phase, ProfileClock clock) noexcept
    : profile_(profile), phase_(phase), clock_(profile ? clock : nullptr) {
    if (clock_) started_ns_ = clock_();
}

PhaseScope::~PhaseScope() {
    if (!profile_ || !clock_) return;
    const uint64_t finished_ns = clock_();
    profile_->add_phase({
        phase_,
        started_ns_ >= profile_->started_ns
            ? started_ns_ - profile_->started_ns
            : 0,
        finished_ns >= started_ns_ ? finished_ns - started_ns_ : 0,
    });
}

ProfileSink * active_profile_sink() noexcept {
    return g_active_profile_sink;
}

ActiveProfileSinkScope::ActiveProfileSinkScope(ProfileSink * sink) noexcept
    : previous_(g_active_profile_sink) {
    g_active_profile_sink = sink;
}

ActiveProfileSinkScope::~ActiveProfileSinkScope() {
    g_active_profile_sink = previous_;
}

}
