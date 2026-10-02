#include "common/observability/concurrency_capture.h"

#include "common/prof_env.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <string_view>
#include <system_error>

namespace luce::common::observability {
namespace {

uint64_t env_u64(const char * name, uint64_t fallback) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    const std::string_view text(raw);
    uint64_t value = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} &&
            parsed.ptr == text.data() + text.size()
        ? value : fallback;
}

size_t env_size(const char * name, size_t fallback) {
    return static_cast<size_t>(std::min<uint64_t>(
        env_u64(name, fallback), std::numeric_limits<size_t>::max()));
}

void write_json_string(std::ostream & out, std::string_view value) {
    static const char kHex[] = "0123456789abcdef";
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u00" << kHex[c >> 4] << kHex[c & 0xf];
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    out << '"';
}

const char * json_bool(bool value) { return value ? "true" : "false"; }

void write_positions(std::ostream & out,
                     const std::array<uint32_t, kMaxSpecPositions> & values,
                     size_t count) {
    out << '[';
    for (size_t i = 0; i < count; ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

void write_step(std::ostream & out, const StepProfile & step) {
    // Trim trailing positions where both histograms are zero.
    size_t positions = 0;
    for (size_t i = kMaxSpecPositions; i > 0; --i) {
        if (step.proposed_by_position[i - 1] ||
            step.accepted_by_position[i - 1]) {
            positions = i;
            break;
        }
    }
    out << "{\"type\":\"step\",\"schema_version\":" << step.schema_version
        << ",\"round_id\":" << step.round_id
        << ",\"started_ns\":" << step.started_ns
        << ",\"duration_ns\":" << step.duration_ns
        << ",\"path\":\"" << step_path_name(step.path) << '"'
        << ",\"ok\":" << json_bool(step.ok)
        << ",\"queue_depth\":" << step.queue_depth
        << ",\"live_slots\":" << step.live_slots
        << ",\"planned_decode_lanes\":" << step.planned_decode_lanes
        << ",\"planned_prefill_lanes\":" << step.planned_prefill_lanes
        << ",\"planned_prefill_tokens\":" << step.planned_prefill_tokens
        << ",\"executed_decode_lanes\":" << step.executed_decode_lanes
        << ",\"executed_prefill_lanes\":" << step.executed_prefill_lanes
        << ",\"executed_prefill_tokens\":" << step.executed_prefill_tokens
        << ",\"spec_eligible_lanes\":" << step.spec_eligible_lanes
        << ",\"spec_reserved_lanes\":" << step.spec_reserved_lanes
        << ",\"spec_attempted_lanes\":" << step.spec_attempted_lanes
        << ",\"spec_proposed_draft_tokens\":" << step.spec_proposed_draft_tokens
        << ",\"spec_verified_draft_tokens\":" << step.spec_verified_draft_tokens
        << ",\"spec_accepted_draft_tokens\":" << step.spec_accepted_draft_tokens
        << ",\"spec_pending_tokens\":" << step.spec_pending_tokens
        << ",\"spec_durable_draft_tokens\":" << step.spec_durable_draft_tokens
        << ",\"spec_scheduler_consumed_tokens\":"
        << step.spec_scheduler_consumed_tokens
        << ",\"target_rows\":" << step.target_rows
        << ",\"target_padding_rows\":" << step.target_padding_rows
        << ",\"draft_rows\":" << step.draft_rows
        << ",\"draft_padding_rows\":" << step.draft_padding_rows
        << ",\"decode_bucket\":" << step.decode_bucket
        << ",\"draft_bucket\":" << step.draft_bucket
        << ",\"spec_tree_width\":" << step.spec_tree_width
        << ",\"max_kv_len\":" << step.max_kv_len
        << ",\"kv_blocks_total\":" << step.kv_blocks_total
        << ",\"kv_blocks_free_before\":" << step.kv_blocks_free_before
        << ",\"kv_blocks_free_after\":" << step.kv_blocks_free_after
        << ",\"active_sequences\":" << step.active_sequences
        << ",\"target_forwards\":" << step.target_forwards
        << ",\"draft_forwards\":" << step.draft_forwards
        << ",\"dropped_lanes\":" << step.dropped_lanes
        << ",\"dropped_phases\":" << step.dropped_phases
        << ",\"proposed_by_position\":";
    write_positions(out, step.proposed_by_position, positions);
    out << ",\"accepted_by_position\":";
    write_positions(out, step.accepted_by_position, positions);
    out << ",\"lanes\":[";
    for (uint32_t i = 0; i < step.lane_count; ++i) {
        const LaneProfile & lane = step.lanes[i];
        out << (i ? "," : "")
            << "{\"request_id\":" << lane.request_id
            << ",\"slot\":" << lane.slot
            << ",\"kind\":\"" << lane_kind_name(lane.kind) << '"'
            << ",\"spec\":\"" << spec_decision_name(lane.spec) << '"'
            << ",\"context_tokens\":" << lane.context_tokens
            << ",\"requested_prefill_tokens\":" << lane.requested_prefill_tokens
            << ",\"executed_prefill_tokens\":" << lane.executed_prefill_tokens
            << ",\"proposed_draft_tokens\":" << lane.proposed_draft_tokens
            << ",\"verified_draft_tokens\":" << lane.verified_draft_tokens
            << ",\"accepted_draft_tokens\":" << lane.accepted_draft_tokens
            << ",\"durable_draft_tokens\":" << lane.durable_draft_tokens
            << ",\"scheduler_consumed_tokens\":"
            << lane.scheduler_consumed_tokens
            << ",\"pending_token_sampled\":"
            << json_bool(lane.pending_token_sampled)
            << ",\"pending_token_consumed\":"
            << json_bool(lane.pending_token_consumed) << '}';
    }
    out << "],\"phases\":[";
    for (uint32_t i = 0; i < step.phase_count; ++i) {
        const PhaseSpan & span = step.phases[i];
        out << (i ? "," : "")
            << "{\"phase\":\"" << phase_name(span.phase) << '"'
            << ",\"start_offset_ns\":" << span.start_offset_ns
            << ",\"duration_ns\":" << span.duration_ns << '}';
    }
    out << "]}\n";
}

}  // namespace

ConcurrencyCaptureConfig ConcurrencyCaptureConfig::from_env() {
    ConcurrencyCaptureConfig config;
    config.enabled = dflash_prof_enabled("concurrency") ||
                     dflash_prof_enabled("1");
    if (!config.enabled) return config;
    const char * path = std::getenv("LUCE_PROF_OUT");
    config.output_path = path && *path ? path : "concurrency-profile.jsonl";
    config.max_rounds = env_size("LUCE_PROF_MAX_ROUNDS", config.max_rounds);
    config.max_requests =
        env_size("LUCE_PROF_MAX_REQUESTS", config.max_requests);
    config.max_token_bursts =
        env_size("LUCE_PROF_MAX_TOKEN_BURSTS", config.max_token_bursts);
    config.checkpoint_every_rounds = env_u64("LUCE_PROF_CHECKPOINT_EVERY", 0);
    for (const char * name : {
             "LUCE_PROF",
             "LUCE_QWEN35_ROCTX",
             "LUCE_DS4_ROCTX",
             "LUCE_QWEN35_DFLASH2_TREE",
             "LUCE_QWEN35_DSPARK_TREE",
             "LUCE_QWEN35_SPEC_STEP_RATIO",
             "LUCE_ADAPTIVE_SPEC_WIDTH",
             "LUCE_DRAFT_KV",
             "LUCE_DISABLE_DRAFT_SWA",
             "LUCE_QWEN35_NO_KVPAD",
             "LUCE_PREFILL_UBATCH",
             "LUCE_KVFLASH",
             "LUCE_MIN_TOKENS",
         }) {
        if (const char * value = std::getenv(name)) {
            config.run_env.emplace_back(name, value);
        }
    }
    return config;
}

ConcurrencyCapture::ConcurrencyCapture(ConcurrencyCaptureConfig config)
    : config_(std::move(config)) {
    if (!config_.enabled) return;
    started_unix_ns_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    started_steady_ns_ = steady_time_ns();
    current_step_ = std::make_unique<StepProfile>();
    steps_.reserve(config_.max_rounds);
    requests_.reserve(config_.max_requests);
    token_bursts_.reserve(config_.max_token_bursts);
    std::fprintf(stderr,
        "[prof] lucebox.concurrency.v1 capture -> %s "
        "(max_rounds=%zu checkpoint_every=%llu)\n",
        config_.output_path.c_str(), config_.max_rounds,
        (unsigned long long)config_.checkpoint_every_rounds);
}

ConcurrencyCapture::~ConcurrencyCapture() { flush(); }

uint64_t ConcurrencyCapture::job_queued() noexcept {
    if (!config_.enabled) return 0;
    queue_depth_.fetch_add(1, std::memory_order_relaxed);
    return steady_time_ns();
}

void ConcurrencyCapture::job_dequeued() noexcept {
    if (!config_.enabled) return;
    uint32_t depth = queue_depth_.load(std::memory_order_relaxed);
    while (depth != 0 && !queue_depth_.compare_exchange_weak(
               depth, depth - 1, std::memory_order_relaxed)) {}
}

StepProfile * ConcurrencyCapture::begin_step(
        uint64_t round_id, uint32_t live_slots, uint64_t started_ns) noexcept {
    if (!config_.enabled) return nullptr;
    StepProfile & step = *current_step_;
    step = StepProfile{};
    step.round_id = round_id;
    step.started_ns = started_ns ? started_ns : steady_time_ns();
    step.queue_depth = queue_depth_.load(std::memory_order_relaxed);
    step.live_slots = live_slots;
    return &step;
}

void ConcurrencyCapture::commit_step(StepProfile * profile) {
    if (!profile) return;
    if (profile->duration_ns == 0) {
        const uint64_t now = steady_time_ns();
        profile->duration_ns =
            now >= profile->started_ns ? now - profile->started_ns : 0;
    }
    if (steps_.size() < config_.max_rounds) {
        steps_.push_back(*profile);
    } else {
        ++dropped_steps_;
    }
    if (config_.checkpoint_every_rounds != 0 &&
        profile->round_id - last_checkpoint_round_ >=
            config_.checkpoint_every_rounds &&
        write_capture(false)) {
        last_checkpoint_round_ = profile->round_id;
    }
}

void ConcurrencyCapture::record_request_admitted(
        uint64_t request_id, const std::string & response_id,
        uint32_t prompt_tokens, uint64_t queued_ns, uint64_t admitted_ns) {
    if (!config_.enabled) return;
    if (requests_.size() >= config_.max_requests) {
        ++dropped_requests_;
        return;
    }
    active_requests_[request_id] = requests_.size();
    RequestRecord record;
    record.request_id = request_id;
    record.response_id = response_id;
    record.prompt_tokens = prompt_tokens;
    record.queued_ns = queued_ns;
    record.admitted_ns = admitted_ns;
    requests_.push_back(std::move(record));
}

void ConcurrencyCapture::record_prefill_completed(
        uint64_t request_id, uint64_t now_ns) {
    if (!config_.enabled) return;
    const auto it = active_requests_.find(request_id);
    if (it != active_requests_.end()) {
        requests_[it->second].prefill_completed_ns = now_ns;
    }
}

void ConcurrencyCapture::record_token_burst(
        uint64_t request_id, uint64_t round_id,
        uint64_t ready_ns, uint32_t token_count) {
    if (!config_.enabled || token_count == 0) return;
    const auto it = active_requests_.find(request_id);
    if (it != active_requests_.end() &&
        requests_[it->second].first_token_ns == 0) {
        requests_[it->second].first_token_ns = ready_ns;
    }
    if (token_bursts_.size() < config_.max_token_bursts) {
        token_bursts_.push_back({request_id, round_id, ready_ns, token_count});
    } else {
        ++dropped_token_bursts_;
    }
}

void ConcurrencyCapture::record_request_finished(
        uint64_t request_id, bool ok, uint32_t output_tokens,
        uint64_t completed_ns) {
    if (!config_.enabled) return;
    const auto it = active_requests_.find(request_id);
    if (it == active_requests_.end()) return;
    RequestRecord & request = requests_[it->second];
    request.ok = ok;
    request.output_tokens = output_tokens;
    request.completed_ns = completed_ns;
    active_requests_.erase(it);
}

void ConcurrencyCapture::write_jsonl(std::ostream & out, bool complete) const {
    out << "{\"type\":\"metadata\",\"schema\":\"lucebox.concurrency.v1\""
        << ",\"schema_version\":" << kProfileSchemaVersion << ",\"git_sha\":";
    write_json_string(out, config_.git_sha);
    out << ",\"model_name\":";
    write_json_string(out, config_.model_name);
    out << ",\"model_path\":";
    write_json_string(out, config_.model_path);
    out << ",\"draft_path\":";
    write_json_string(out, config_.draft_path);
    out << ",\"arch\":";
    write_json_string(out, config_.arch);
    out << ",\"runtime_backend\":";
    write_json_string(out, config_.runtime_backend);
    out << ",\"max_concurrency\":" << config_.max_concurrency
        << ",\"ddtree_budget\":" << config_.ddtree_budget
        << ",\"draft_block_size\":" << config_.draft_block_size
        << ",\"started_unix_ns\":" << started_unix_ns_
        << ",\"started_steady_ns\":" << started_steady_ns_
        << ",\"round_retention\":\"keep_first\""
        << ",\"max_rounds\":" << config_.max_rounds
        << ",\"step_record_bytes\":" << sizeof(StepProfile)
        << ",\"checkpoint_every_rounds\":" << config_.checkpoint_every_rounds
        << ",\"env\":{";
    for (size_t i = 0; i < config_.run_env.size(); ++i) {
        if (i) out << ',';
        write_json_string(out, config_.run_env[i].first);
        out << ':';
        write_json_string(out, config_.run_env[i].second);
    }
    out << "}}\n";
    for (const StepProfile & step : steps_) write_step(out, step);
    for (const RequestRecord & request : requests_) {
        // A request still active at a checkpoint has not finished: ok=null.
        const bool active = active_requests_.count(request.request_id) != 0;
        out << "{\"type\":\"request\",\"request_id\":" << request.request_id
            << ",\"response_id\":";
        write_json_string(out, request.response_id);
        out << ",\"ok\":" << (active ? "null" : json_bool(request.ok))
            << ",\"prompt_tokens\":" << request.prompt_tokens
            << ",\"output_tokens\":" << request.output_tokens
            << ",\"queued_ns\":" << request.queued_ns
            << ",\"admitted_ns\":" << request.admitted_ns
            << ",\"prefill_completed_ns\":" << request.prefill_completed_ns
            << ",\"first_token_ns\":" << request.first_token_ns
            << ",\"completed_ns\":" << (active ? 0 : request.completed_ns)
            << "}\n";
    }
    for (const TokenBurst & burst : token_bursts_) {
        out << "{\"type\":\"token_burst\",\"request_id\":" << burst.request_id
            << ",\"round_id\":" << burst.round_id
            << ",\"ready_ns\":" << burst.ready_ns
            << ",\"token_count\":" << burst.token_count << "}\n";
    }
    out << "{\"type\":\"footer\",\"dropped_steps\":" << dropped_steps_
        << ",\"dropped_requests\":" << dropped_requests_
        << ",\"dropped_token_bursts\":" << dropped_token_bursts_
        << ",\"complete\":" << json_bool(complete) << "}\n";
}

bool ConcurrencyCapture::write_capture(bool complete) {
    if (!config_.enabled || config_.output_path.empty()) return false;
    const std::string temporary = config_.output_path + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (out) write_jsonl(out, complete);
        out.close();
        if (!out) {
            std::fprintf(stderr, "[prof] failed to write %s\n",
                         temporary.c_str());
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, config_.output_path, error);
#if defined(_WIN32)
    if (error) {
        std::filesystem::remove(config_.output_path, error);
        error.clear();
        std::filesystem::rename(temporary, config_.output_path, error);
    }
#endif
    if (error) {
        std::fprintf(stderr, "[prof] failed to publish %s: %s\n",
                     config_.output_path.c_str(), error.message().c_str());
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    return true;
}

void ConcurrencyCapture::flush() {
    if (!config_.enabled || flushed_) return;
    if (write_capture(true)) flushed_ = true;
}

}  // namespace luce::common::observability
