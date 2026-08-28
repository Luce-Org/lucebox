#include "server/observability.h"

#include "common/prof_env.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace dflash::common::observability {
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

size_t bounded_size_env(const char * name, size_t fallback) {
    return static_cast<size_t>(std::min<uint64_t>(
        env_u64(name, fallback), std::numeric_limits<size_t>::max()));
}

void capture_env(
        ObservabilityConfig & config,
        std::initializer_list<const char *> names) {
    for (const char * name : names) {
        if (const char * value = std::getenv(name)) {
            config.run_env.emplace_back(name, value);
        }
    }
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
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
                out << "\\u" << std::hex << std::setw(4)
                    << std::setfill('0') << static_cast<int>(c)
                    << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

void write_u32_array(
        std::ostream & out,
        const std::array<uint32_t, kMaxSpecPositions> & values,
        size_t count) {
    out << '[';
    for (size_t i = 0; i < count; ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_step_json(std::ostream & out, const StepProfile & step) {
    size_t position_count = 0;
    for (size_t i = kMaxSpecPositions; i > 0; --i) {
        if (step.proposed_by_position[i - 1] != 0 ||
            step.accepted_by_position[i - 1] != 0) {
            position_count = i;
            break;
        }
    }
    out << "{\"type\":\"step\",\"schema_version\":"
        << step.schema_version
        << ",\"round_id\":" << step.round_id
        << ",\"started_ns\":" << step.started_ns
        << ",\"duration_ns\":" << step.duration_ns
        << ",\"path\":\"" << step_path_name(step.path) << "\""
        << ",\"ok\":" << (step.ok ? "true" : "false")
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
        << ",\"spec_proposed_draft_tokens\":"
        << step.spec_proposed_draft_tokens
        << ",\"spec_verified_draft_tokens\":"
        << step.spec_verified_draft_tokens
        << ",\"spec_accepted_draft_tokens\":"
        << step.spec_accepted_draft_tokens
        << ",\"spec_pending_tokens\":" << step.spec_pending_tokens
        << ",\"spec_durable_draft_tokens\":"
        << step.spec_durable_draft_tokens
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
    write_u32_array(out, step.proposed_by_position, position_count);
    out << ",\"accepted_by_position\":";
    write_u32_array(out, step.accepted_by_position, position_count);
    out << ",\"lanes\":[";
    for (uint32_t i = 0; i < step.lane_count; ++i) {
        if (i) out << ',';
        const LaneProfile & lane = step.lanes[i];
        out << "{\"request_id\":" << lane.request_id
            << ",\"slot\":" << lane.slot
            << ",\"kind\":\"" << lane_kind_name(lane.kind) << "\""
            << ",\"spec\":\"" << spec_decision_name(lane.spec) << "\""
            << ",\"context_tokens\":" << lane.context_tokens
            << ",\"requested_prefill_tokens\":"
            << lane.requested_prefill_tokens
            << ",\"executed_prefill_tokens\":"
            << lane.executed_prefill_tokens
            << ",\"proposed_draft_tokens\":"
            << lane.proposed_draft_tokens
            << ",\"verified_draft_tokens\":"
            << lane.verified_draft_tokens
            << ",\"accepted_draft_tokens\":"
            << lane.accepted_draft_tokens
            << ",\"durable_draft_tokens\":"
            << lane.durable_draft_tokens
            << ",\"scheduler_consumed_tokens\":"
            << lane.scheduler_consumed_tokens
            << ",\"pending_token_sampled\":"
            << (lane.pending_token_sampled ? "true" : "false")
            << ",\"pending_token_consumed\":"
            << (lane.pending_token_consumed ? "true" : "false") << '}';
    }
    out << "],\"phases\":[";
    for (uint32_t i = 0; i < step.phase_count; ++i) {
        if (i) out << ',';
        const PhaseSpan & span = step.phases[i];
        out << "{\"phase\":\"" << phase_name(span.phase) << "\""
            << ",\"start_offset_ns\":" << span.start_offset_ns
            << ",\"duration_ns\":" << span.duration_ns << '}';
    }
    out << "]}\n";
}

}

ObservabilityConfig ObservabilityConfig::from_env() {
    ObservabilityConfig config;
    config.enabled = dflash_prof_enabled("concurrency");
    if (const char * path = std::getenv("DFLASH_PROF_OUT")) {
        if (*path) config.output_path = path;
    }
    config.max_rounds = bounded_size_env("DFLASH_PROF_MAX_ROUNDS", 10000);
    config.max_requests = bounded_size_env("DFLASH_PROF_MAX_REQUESTS", 4096);
    config.max_token_bursts = bounded_size_env(
        "DFLASH_PROF_MAX_TOKEN_BURSTS", 200000);
    config.checkpoint_every_rounds = env_u64(
        "DFLASH_PROF_CHECKPOINT_EVERY", 0);
    capture_env(config, {
        "DFLASH_QWEN35_DFLASH2_TREE",
        "DFLASH_QWEN35_DSPARK_TREE",
        "DFLASH_QWEN35_SPEC_STEP_RATIO",
        "DFLASH_DRAFT_KV",
        "DFLASH_DISABLE_DRAFT_SWA",
        "DFLASH_QWEN35_NO_KVPAD",
        "DFLASH_BAILING_K_AS_V",
        "DFLASH_GDN_DIRECT_FAST",
        "DFLASH27B_PREFILL_UBATCH",
        "DFLASH_KVFLASH",
        "DFLASH_ADAPTIVE_K_TAU",
        "DFLASH_ADAPTIVE_K_DENSE",
        "DFLASH_ADAPTIVE_SPEC_WIDTH",
        "DFLASH_DSPARK_ADAPTIVE_MAX_WIDTH",
        "DFLASH_DSPARK_VERIFY_WIDTH",
        "DFLASH_CUDA_MMVQ_GROUPED_TPG",
        "DFLASH_CUDA_MMVQ_GROUPED_GLU",
    });
    return config;
}

ObservabilityState::ObservabilityState(ObservabilityConfig config)
    : config_(std::move(config)) {
    live_.enabled = config_.enabled;
    if (!config_.enabled) return;
    started_unix_ns_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    started_steady_ns_ = steady_time_ns();
    steps_.reserve(config_.max_rounds);
    requests_.reserve(config_.max_requests);
    token_bursts_.reserve(config_.max_token_bursts);
    active_requests_.reserve(config_.max_requests);
}

ObservabilityState::~ObservabilityState() {
    flush();
}

uint64_t ObservabilityState::job_queued() noexcept {
    if (!config_.enabled) return 0;
    queue_depth_.fetch_add(1, std::memory_order_relaxed);
    return steady_time_ns();
}

void ObservabilityState::job_dequeued() noexcept {
    if (!config_.enabled) return;
    uint32_t depth = queue_depth_.load(std::memory_order_relaxed);
    while (depth != 0 && !queue_depth_.compare_exchange_weak(
               depth, depth - 1, std::memory_order_relaxed)) {}
}

uint32_t ObservabilityState::queue_depth() const noexcept {
    return config_.enabled
        ? queue_depth_.load(std::memory_order_relaxed)
        : 0;
}

void ObservabilityState::set_live_slots(uint32_t live_slots) {
    if (!config_.enabled) return;
    std::lock_guard<std::mutex> lock(live_mu_);
    live_.live_slots = live_slots;
}

StepProfile * ObservabilityState::begin_step(uint32_t live_slots) noexcept {
    if (!config_.enabled) return nullptr;
    current_step_ = {};
    current_step_.round_id = next_round_id_++;
    current_step_.started_ns = steady_time_ns();
    current_step_.queue_depth = queue_depth();
    current_step_.live_slots = live_slots;
    return &current_step_;
}

void ObservabilityState::commit_step(StepProfile * profile) {
    if (!profile) return;
    if (profile->duration_ns == 0) {
        const uint64_t now = steady_time_ns();
        profile->duration_ns = now >= profile->started_ns
            ? now - profile->started_ns : 0;
    }

    uint64_t consumed = 0;
    for (uint32_t i = 0; i < profile->lane_count; ++i) {
        if (profile->lanes[i].kind == LaneKind::Decode) {
            consumed += profile->lanes[i].scheduler_consumed_tokens;
        }
    }

    {
        std::lock_guard<std::mutex> lock(live_mu_);
        ++live_.rounds;
        live_.queue_depth = queue_depth();
        live_.live_slots = profile->live_slots;
        live_.kv_blocks_total = profile->kv_blocks_total;
        live_.kv_blocks_free = profile->kv_blocks_free_after;
        live_.planned_prefill_tokens += profile->planned_prefill_tokens;
        live_.executed_prefill_tokens += profile->executed_prefill_tokens;
        live_.decode_lanes += profile->executed_decode_lanes;
        live_.durable_decode_tokens += consumed;
        live_.spec_eligible_lanes += profile->spec_eligible_lanes;
        live_.spec_reserved_lanes += profile->spec_reserved_lanes;
        live_.spec_attempted_lanes += profile->spec_attempted_lanes;
        live_.spec_proposed_draft_tokens +=
            profile->spec_proposed_draft_tokens;
        live_.spec_verified_draft_tokens +=
            profile->spec_verified_draft_tokens;
        live_.spec_accepted_draft_tokens +=
            profile->spec_accepted_draft_tokens;
        live_.spec_durable_draft_tokens +=
            profile->spec_durable_draft_tokens;
        live_.spec_scheduler_consumed_tokens +=
            profile->spec_scheduler_consumed_tokens;
        live_.target_rows += profile->target_rows;
        live_.target_padding_rows += profile->target_padding_rows;
        live_.draft_rows += profile->draft_rows;
        live_.draft_padding_rows += profile->draft_padding_rows;
        for (uint32_t i = 0; i < profile->phase_count; ++i) {
            const size_t phase = static_cast<size_t>(profile->phases[i].phase);
            if (phase < live_.phase_ns.size()) {
                live_.phase_ns[phase] += profile->phases[i].duration_ns;
            }
        }
        live_.last_step = {
            profile->round_id, profile->duration_ns, profile->path,
            profile->queue_depth, profile->live_slots,
        };
    }

    if (steps_.size() < config_.max_rounds) {
        steps_.push_back(*profile);
    } else {
        std::lock_guard<std::mutex> lock(live_mu_);
        ++live_.dropped_steps;
    }
    if (config_.checkpoint_every_rounds != 0 &&
        profile->round_id - last_checkpoint_round_ >=
            config_.checkpoint_every_rounds &&
        write_capture(false)) {
        last_checkpoint_round_ = profile->round_id;
    }
}

void ObservabilityState::record_request_admitted(
        uint64_t request_id, std::string response_id,
        uint32_t prompt_tokens, uint64_t queued_ns,
        uint64_t admitted_ns) {
    if (!config_.enabled) return;
    if (requests_.size() >= config_.max_requests) {
        std::lock_guard<std::mutex> lock(live_mu_);
        ++live_.dropped_requests;
        return;
    }
    active_requests_[request_id] = requests_.size();
    requests_.push_back({request_id, std::move(response_id), false,
                         prompt_tokens, 0, queued_ns, admitted_ns});
}

void ObservabilityState::record_prefill_completed(
        uint64_t request_id, uint64_t now_ns) {
    if (!config_.enabled) return;
    const auto it = active_requests_.find(request_id);
    if (it != active_requests_.end()) {
        requests_[it->second].prefill_completed_ns = now_ns;
    }
}

void ObservabilityState::record_token_burst(
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
        std::lock_guard<std::mutex> lock(live_mu_);
        ++live_.dropped_token_bursts;
    }
}

void ObservabilityState::record_request_finished(
        uint64_t request_id, bool ok, uint32_t output_tokens,
        uint64_t completed_ns) {
    if (!config_.enabled) return;
    const auto it = active_requests_.find(request_id);
    if (it != active_requests_.end()) {
        RequestRecord & request = requests_[it->second];
        request.ok = ok;
        request.output_tokens = output_tokens;
        request.completed_ns = completed_ns;
        active_requests_.erase(it);
    }
    std::lock_guard<std::mutex> lock(live_mu_);
    ++live_.requests_completed;
    if (!ok) ++live_.requests_failed;
}

LiveMetricsSnapshot ObservabilityState::snapshot() const {
    if (!config_.enabled) return live_;
    std::lock_guard<std::mutex> lock(live_mu_);
    LiveMetricsSnapshot result = live_;
    result.queue_depth = queue_depth();
    return result;
}

std::string ObservabilityState::snapshot_json() const {
    const LiveMetricsSnapshot s = snapshot();
    std::ostringstream out;
    out << "{\"enabled\":" << (s.enabled ? "true" : "false")
        << ",\"schema_version\":" << s.schema_version
        << ",\"rounds\":" << s.rounds
        << ",\"queue_depth\":" << s.queue_depth
        << ",\"live_slots\":" << s.live_slots
        << ",\"kv_blocks_total\":" << s.kv_blocks_total
        << ",\"kv_blocks_free\":" << s.kv_blocks_free
        << ",\"planned_prefill_tokens\":" << s.planned_prefill_tokens
        << ",\"executed_prefill_tokens\":" << s.executed_prefill_tokens
        << ",\"decode_lanes\":" << s.decode_lanes
        << ",\"durable_decode_tokens\":" << s.durable_decode_tokens
        << ",\"spec_eligible_lanes\":" << s.spec_eligible_lanes
        << ",\"spec_reserved_lanes\":" << s.spec_reserved_lanes
        << ",\"spec_attempted_lanes\":" << s.spec_attempted_lanes
        << ",\"spec_proposed_draft_tokens\":"
        << s.spec_proposed_draft_tokens
        << ",\"spec_verified_draft_tokens\":"
        << s.spec_verified_draft_tokens
        << ",\"spec_accepted_draft_tokens\":"
        << s.spec_accepted_draft_tokens
        << ",\"spec_durable_draft_tokens\":"
        << s.spec_durable_draft_tokens
        << ",\"spec_scheduler_consumed_tokens\":"
        << s.spec_scheduler_consumed_tokens
        << ",\"target_rows\":" << s.target_rows
        << ",\"target_padding_rows\":" << s.target_padding_rows
        << ",\"draft_rows\":" << s.draft_rows
        << ",\"draft_padding_rows\":" << s.draft_padding_rows
        << ",\"requests_completed\":" << s.requests_completed
        << ",\"requests_failed\":" << s.requests_failed
        << ",\"dropped_steps\":" << s.dropped_steps
        << ",\"dropped_requests\":" << s.dropped_requests
        << ",\"dropped_token_bursts\":" << s.dropped_token_bursts
        << ",\"phases\":{";
    for (size_t i = 0; i < s.phase_ns.size(); ++i) {
        if (i) out << ',';
        out << '"' << phase_name(static_cast<Phase>(i)) << "\":"
            << s.phase_ns[i];
    }
    out << "},\"last_step\":{\"round_id\":" << s.last_step.round_id
        << ",\"duration_ns\":" << s.last_step.duration_ns
        << ",\"path\":\"" << step_path_name(s.last_step.path) << "\""
        << ",\"queue_depth\":" << s.last_step.queue_depth
        << ",\"live_slots\":" << s.last_step.live_slots << "}}\n";
    return out.str();
}

bool ObservabilityState::write_capture(bool complete) {
    if (!config_.enabled || config_.output_path.empty()) return false;
    const std::string temporary_path = config_.output_path + ".tmp";
    std::ofstream out(temporary_path);
    if (!out) {
        std::cerr << "[observability] failed to write "
                  << temporary_path << '\n';
        return false;
    }
    out << "{\"type\":\"metadata\",\"schema\":\"lucebox.concurrency.v1\""
        << ",\"schema_version\":" << kProfileSchemaVersion
        << ",\"git_sha\":\"" << json_escape(config_.git_sha) << "\""
        << ",\"model_name\":\"" << json_escape(config_.model_name) << "\""
        << ",\"model_path\":\"" << json_escape(config_.model_path) << "\""
        << ",\"draft_path\":\"" << json_escape(config_.draft_path) << "\""
        << ",\"arch\":\"" << json_escape(config_.arch) << "\""
        << ",\"runtime_backend\":\""
        << json_escape(config_.runtime_backend) << "\""
        << ",\"max_concurrency\":" << config_.max_concurrency
        << ",\"ddtree_budget\":" << config_.ddtree_budget
        << ",\"draft_block_size\":" << config_.draft_block_size
        << ",\"started_unix_ns\":" << started_unix_ns_
        << ",\"started_steady_ns\":" << started_steady_ns_
        << ",\"round_retention\":\"keep_first\""
        << ",\"max_rounds\":" << config_.max_rounds
        << ",\"step_record_bytes\":" << sizeof(StepProfile)
        << ",\"checkpoint_every_rounds\":"
        << config_.checkpoint_every_rounds
        << ",\"env\":{";
    for (size_t i = 0; i < config_.run_env.size(); ++i) {
        if (i) out << ',';
        out << '\"' << json_escape(config_.run_env[i].first) << "\":\""
            << json_escape(config_.run_env[i].second) << '\"';
    }
    out << "}}\n";
    for (const StepProfile & step : steps_) write_step_json(out, step);
    for (const RequestRecord & request : requests_) {
        out << "{\"type\":\"request\",\"request_id\":"
            << request.request_id
            << ",\"response_id\":\"" << json_escape(request.response_id)
            << "\",\"ok\":";
        if (request.completed_ns == 0) out << "null";
        else out << (request.ok ? "true" : "false");
        out << ",\"prompt_tokens\":" << request.prompt_tokens
            << ",\"output_tokens\":" << request.output_tokens
            << ",\"queued_ns\":" << request.queued_ns
            << ",\"admitted_ns\":" << request.admitted_ns
            << ",\"prefill_completed_ns\":"
            << request.prefill_completed_ns
            << ",\"first_token_ns\":" << request.first_token_ns
            << ",\"completed_ns\":" << request.completed_ns << "}\n";
    }
    for (const TokenBurst & burst : token_bursts_) {
        out << "{\"type\":\"token_burst\",\"request_id\":"
            << burst.request_id << ",\"round_id\":" << burst.round_id
            << ",\"ready_ns\":" << burst.ready_ns
            << ",\"token_count\":" << burst.token_count << "}\n";
    }
    const LiveMetricsSnapshot s = snapshot();
    out << "{\"type\":\"footer\",\"dropped_steps\":"
        << s.dropped_steps << ",\"dropped_requests\":"
        << s.dropped_requests << ",\"dropped_token_bursts\":"
        << s.dropped_token_bursts << ",\"complete\":"
        << (complete ? "true" : "false") << "}\n";
    out.close();
    if (!out) {
        std::cerr << "[observability] failed to finish "
                  << temporary_path << '\n';
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        return false;
    }

    std::error_code error;
    std::filesystem::rename(
        temporary_path, config_.output_path, error);
#if defined(_WIN32)
    if (error) {
        std::filesystem::remove(config_.output_path, error);
        error.clear();
        std::filesystem::rename(
            temporary_path, config_.output_path, error);
    }
#endif
    if (error) {
        std::cerr << "[observability] failed to publish "
                  << config_.output_path << ": " << error.message() << '\n';
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        return false;
    }
    return true;
}

void ObservabilityState::flush() {
    if (!config_.enabled || config_.output_path.empty() || flushed_) return;
    if (write_capture(true)) flushed_ = true;
}

}
