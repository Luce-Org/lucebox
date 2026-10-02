// inference_profile.h — fixed-capacity, model-neutral record of one
// scheduler service round for the lucebox.concurrency.v1 capture.
//
// The scheduler owns a StepProfile only while LUCE_PROF enables the capture
// and passes it to the engine through SeqEngine::StepPlan::profile. A null
// pointer is the whole disabled path: PhaseScope then reads no clock, and
// engines guard every other profile write with `if (profile)`. Phase spans
// are host steady-clock intervals around existing operations; they never
// synchronize a device and are not GPU execution times.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace luce::common::observability {

inline constexpr uint32_t kProfileSchemaVersion = 1;
inline constexpr size_t kMaxProfileLanes = 64;
inline constexpr size_t kMaxProfilePhases = 32;
inline constexpr size_t kMaxSpecPositions = 64;

// `Chain` is the Qwen fixed-width DFlash2 chain round; v1 consumers that
// predate it know only unknown/packed/speculative.
enum class StepPath : uint8_t {
    Unknown,
    Packed,
    Speculative,
    Chain,
};

enum class LaneKind : uint8_t {
    Decode,
    Prefill,
};

enum class SpecDecision : uint8_t {
    None,
    Selected,
    InvalidSlot,
    PromptWorkPresent,
    CallerDisallowed,
    FeatureUnavailable,
    SamplingUnsupported,
    InsufficientContext,
    DraftPrepareFailed,
};

enum class Phase : uint8_t {
    SchedulerPlan,
    InputStaging,
    DraftPrepare,
    DraftCompute,
    ProposalSelect,
    TargetGraphBuild,
    MetadataUpload,
    TargetCompute,
    ReadbackSync,
    Acceptance,
    StatePromotion,
    SamplingCommit,
    OutputProcessing,
    ClientFlush,
};

struct PhaseSpan {
    Phase phase = Phase::SchedulerPlan;
    uint64_t start_offset_ns = 0;
    uint64_t duration_ns = 0;
};

struct LaneProfile {
    uint64_t request_id = 0;
    int32_t slot = -1;
    LaneKind kind = LaneKind::Decode;
    SpecDecision spec = SpecDecision::None;
    uint32_t context_tokens = 0;
    uint32_t requested_prefill_tokens = 0;
    uint32_t executed_prefill_tokens = 0;
    uint32_t proposed_draft_tokens = 0;
    uint32_t verified_draft_tokens = 0;
    uint32_t accepted_draft_tokens = 0;
    uint32_t durable_draft_tokens = 0;
    uint32_t scheduler_consumed_tokens = 0;
    bool pending_token_sampled = false;
    bool pending_token_consumed = false;
};

struct StepProfile {
    uint32_t schema_version = kProfileSchemaVersion;
    uint64_t round_id = 0;
    uint64_t started_ns = 0;
    uint64_t duration_ns = 0;
    StepPath path = StepPath::Unknown;
    bool ok = true;

    uint32_t queue_depth = 0;
    uint32_t live_slots = 0;
    uint32_t planned_decode_lanes = 0;
    uint32_t planned_prefill_lanes = 0;
    uint32_t planned_prefill_tokens = 0;
    uint32_t executed_decode_lanes = 0;
    uint32_t executed_prefill_lanes = 0;
    uint32_t executed_prefill_tokens = 0;

    uint32_t spec_eligible_lanes = 0;
    uint32_t spec_reserved_lanes = 0;
    uint32_t spec_attempted_lanes = 0;
    uint32_t spec_proposed_draft_tokens = 0;
    uint32_t spec_verified_draft_tokens = 0;
    uint32_t spec_accepted_draft_tokens = 0;
    uint32_t spec_pending_tokens = 0;
    uint32_t spec_durable_draft_tokens = 0;
    uint32_t spec_scheduler_consumed_tokens = 0;

    uint32_t target_rows = 0;
    uint32_t target_padding_rows = 0;
    uint32_t draft_rows = 0;
    uint32_t draft_padding_rows = 0;
    uint32_t decode_bucket = 0;
    uint32_t draft_bucket = 0;
    uint32_t spec_tree_width = 0;
    uint32_t max_kv_len = 0;
    uint32_t kv_blocks_total = 0;
    uint32_t kv_blocks_free_before = 0;
    uint32_t kv_blocks_free_after = 0;
    uint32_t active_sequences = 0;
    uint32_t target_forwards = 0;
    uint32_t draft_forwards = 0;

    // Index 0 is the speculative root.
    std::array<uint32_t, kMaxSpecPositions> proposed_by_position{};
    std::array<uint32_t, kMaxSpecPositions> accepted_by_position{};
    std::array<LaneProfile, kMaxProfileLanes> lanes{};
    std::array<PhaseSpan, kMaxProfilePhases> phases{};
    uint32_t lane_count = 0;
    uint32_t phase_count = 0;
    uint32_t dropped_lanes = 0;
    uint32_t dropped_phases = 0;

    // Keep-first: a full array counts the drop and keeps no partial record.
    LaneProfile * add_lane(const LaneProfile & lane) noexcept;
    LaneProfile * find_lane(int32_t slot, LaneKind kind) noexcept;
    void add_phase(PhaseSpan span) noexcept;
    // Record a phase that started at `started_ns` and ends now.
    void end_phase(Phase phase, uint64_t started_ns) noexcept;
};

const char * step_path_name(StepPath path) noexcept;
const char * lane_kind_name(LaneKind kind) noexcept;
const char * spec_decision_name(SpecDecision decision) noexcept;
const char * phase_name(Phase phase) noexcept;

uint64_t steady_time_ns() noexcept;
using ProfileClock = uint64_t (*)() noexcept;

// Host-side span around an existing operation. With a null profile it does
// nothing, not even a clock read.
class PhaseScope final {
public:
    PhaseScope(StepProfile * profile, Phase phase,
               ProfileClock clock = steady_time_ns) noexcept;
    ~PhaseScope();

    PhaseScope(const PhaseScope &) = delete;
    PhaseScope & operator=(const PhaseScope &) = delete;

private:
    StepProfile * profile_ = nullptr;
    Phase phase_ = Phase::SchedulerPlan;
    ProfileClock clock_ = nullptr;
    uint64_t started_ns_ = 0;
};

// Scheduler-owned service round id on the serving thread, so ROCTX markers
// deep inside an adapter carry the same round_id as the capture step. Zero
// outside a round (for example the classic single-request worker).
uint64_t current_service_round() noexcept;

class ServiceRoundScope final {
public:
    explicit ServiceRoundScope(uint64_t round_id) noexcept;
    ~ServiceRoundScope();

    ServiceRoundScope(const ServiceRoundScope &) = delete;
    ServiceRoundScope & operator=(const ServiceRoundScope &) = delete;

private:
    uint64_t previous_ = 0;
};

}  // namespace luce::common::observability
