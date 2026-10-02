// Producer-side contract test for the LUCE_PROF lucebox.concurrency.v1
// capture: drives the emitter API the scheduler and engines use, writes a
// real capture, and checks it against the consumer schema (record order,
// required fields and types, enum values, round identity, retention bounds,
// checkpoint semantics). Host only; no GPU.
//
// Usage: test_concurrency_capture [kept-capture.jsonl]

#include "common/observability/concurrency_capture.h"
#include "common/observability/inference_profile.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace luce::common::observability;
using nlohmann::json;

namespace {

int failures = 0;

#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
} } while (0)

uint64_t fake_now = 0;
int clock_reads = 0;
uint64_t fake_clock() noexcept {
    ++clock_reads;
    return fake_now;
}

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

std::vector<json> read_capture(const std::string & path) {
    std::vector<json> records;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        records.push_back(json::parse(line));  // throws on invalid JSON
    }
    return records;
}

bool is_uint(const json & record, const char * key) {
    return record.contains(key) && record[key].is_number_unsigned();
}

bool is_int(const json & record, const char * key) {
    return record.contains(key) && record[key].is_number_integer();
}

bool is_str(const json & record, const char * key) {
    return record.contains(key) && record[key].is_string();
}

bool is_bool(const json & record, const char * key) {
    return record.contains(key) && record[key].is_boolean();
}

bool in_set(const json & value, const std::set<std::string> & allowed) {
    return value.is_string() && allowed.count(value.get<std::string>()) != 0;
}

// Schema check shared by the checkpoint and the final capture. Returns the
// records so callers can make capture-specific assertions.
std::vector<json> check_schema(const std::string & path, bool complete) {
    const std::vector<json> records = read_capture(path);
    CHECK(records.size() >= 2);
    if (records.size() < 2) return records;

    // Record order: metadata, step*, request*, token_burst*, footer.
    const std::vector<std::string> order = {
        "metadata", "step", "request", "token_burst", "footer"};
    size_t rank = 0;
    for (size_t i = 0; i < records.size(); ++i) {
        CHECK(is_str(records[i], "type"));
        const std::string type = records[i].value("type", "");
        size_t r = 0;
        while (r < order.size() && order[r] != type) ++r;
        CHECK(r < order.size());
        CHECK(r >= rank);
        rank = r;
        if (i == 0) CHECK(type == "metadata");
        if (i + 1 == records.size()) CHECK(type == "footer");
        if (type == "metadata" || type == "footer") {
            CHECK((i == 0) == (type == "metadata"));
            CHECK((i + 1 == records.size()) == (type == "footer"));
        }
    }

    const json & meta = records.front();
    CHECK(meta.value("schema", "") == "lucebox.concurrency.v1");
    CHECK(meta.value("schema_version", 0) == 1);
    for (const char * key : {"git_sha", "model_name", "model_path",
                             "draft_path", "arch", "runtime_backend"}) {
        CHECK(is_str(meta, key));
    }
    for (const char * key : {"max_concurrency", "ddtree_budget",
                             "draft_block_size"}) {
        CHECK(is_int(meta, key));
    }
    for (const char * key : {"started_unix_ns", "started_steady_ns",
                             "max_rounds", "step_record_bytes",
                             "checkpoint_every_rounds"}) {
        CHECK(is_uint(meta, key));
    }
    CHECK(meta.value("round_retention", "") == "keep_first");
    CHECK(meta.contains("env") && meta["env"].is_object());

    const std::set<std::string> paths = {
        "unknown", "packed", "speculative", "chain"};
    const std::set<std::string> kinds = {"decode", "prefill"};
    const std::set<std::string> specs = {
        "none", "selected", "invalid_slot", "prompt_work_present",
        "caller_disallowed", "feature_unavailable", "sampling_unsupported",
        "insufficient_context", "draft_prepare_failed"};
    const std::set<std::string> phases = {
        "scheduler_plan", "input_staging", "draft_prepare", "draft_compute",
        "proposal_select", "target_graph_build", "metadata_upload",
        "target_compute", "readback_sync", "acceptance", "state_promotion",
        "sampling_commit", "output_processing", "client_flush"};

    std::set<uint64_t> round_ids;
    for (const json & record : records) {
        const std::string type = record.value("type", "");
        if (type == "step") {
            CHECK(record.value("schema_version", 0) == 1);
            for (const char * key : {
                     "round_id", "started_ns", "duration_ns", "queue_depth",
                     "live_slots", "planned_decode_lanes",
                     "planned_prefill_lanes", "planned_prefill_tokens",
                     "executed_decode_lanes", "executed_prefill_lanes",
                     "executed_prefill_tokens", "spec_eligible_lanes",
                     "spec_reserved_lanes", "spec_attempted_lanes",
                     "spec_proposed_draft_tokens",
                     "spec_verified_draft_tokens",
                     "spec_accepted_draft_tokens", "spec_pending_tokens",
                     "spec_durable_draft_tokens",
                     "spec_scheduler_consumed_tokens", "target_rows",
                     "target_padding_rows", "draft_rows",
                     "draft_padding_rows", "decode_bucket", "draft_bucket",
                     "spec_tree_width", "max_kv_len", "kv_blocks_total",
                     "kv_blocks_free_before", "kv_blocks_free_after",
                     "active_sequences", "target_forwards", "draft_forwards",
                     "dropped_lanes", "dropped_phases"}) {
                CHECK(is_uint(record, key));
            }
            CHECK(in_set(record["path"], paths));
            CHECK(is_bool(record, "ok"));
            CHECK(record["proposed_by_position"].is_array());
            CHECK(record["accepted_by_position"].is_array());
            CHECK(record["proposed_by_position"].size() ==
                  record["accepted_by_position"].size());
            const json & proposed = record["proposed_by_position"];
            if (!proposed.empty()) {
                // Trailing all-zero positions are trimmed.
                CHECK(proposed.back().get<uint64_t>() != 0 ||
                      record["accepted_by_position"].back().get<uint64_t>() != 0);
            }
            CHECK(round_ids.insert(record["round_id"].get<uint64_t>()).second);
            for (const json & lane : record["lanes"]) {
                for (const char * key : {
                         "request_id", "context_tokens",
                         "requested_prefill_tokens", "executed_prefill_tokens",
                         "proposed_draft_tokens", "verified_draft_tokens",
                         "accepted_draft_tokens", "durable_draft_tokens",
                         "scheduler_consumed_tokens"}) {
                    CHECK(is_uint(lane, key));
                }
                CHECK(is_int(lane, "slot"));
                CHECK(in_set(lane["kind"], kinds));
                CHECK(in_set(lane["spec"], specs));
                CHECK(is_bool(lane, "pending_token_sampled"));
                CHECK(is_bool(lane, "pending_token_consumed"));
            }
            for (const json & span : record["phases"]) {
                CHECK(in_set(span["phase"], phases));
                CHECK(is_uint(span, "start_offset_ns"));
                CHECK(is_uint(span, "duration_ns"));
            }
        } else if (type == "request") {
            for (const char * key : {
                     "request_id", "prompt_tokens", "output_tokens",
                     "queued_ns", "admitted_ns", "prefill_completed_ns",
                     "first_token_ns", "completed_ns"}) {
                CHECK(is_uint(record, key));
            }
            CHECK(is_str(record, "response_id"));
            CHECK(record.contains("ok") &&
                  (record["ok"].is_boolean() || record["ok"].is_null()));
            if (record["ok"].is_null()) {
                CHECK(record["completed_ns"].get<uint64_t>() == 0);
            }
            if (complete) CHECK(!record["ok"].is_null() ||
                                record["completed_ns"].get<uint64_t>() == 0);
        } else if (type == "token_burst") {
            for (const char * key : {"request_id", "round_id", "ready_ns",
                                     "token_count"}) {
                CHECK(is_uint(record, key));
            }
        } else if (type == "footer") {
            for (const char * key : {"dropped_steps", "dropped_requests",
                                     "dropped_token_bursts"}) {
                CHECK(is_uint(record, key));
            }
            CHECK(record.value("complete", !complete) == complete);
        }
    }
    return records;
}

void test_disabled_is_inert() {
    set_env("LUCE_PROF", nullptr);
    const ConcurrencyCaptureConfig config = ConcurrencyCaptureConfig::from_env();
    CHECK(!config.enabled);
    set_env("LUCE_PROF", "step,verify");
    CHECK(!ConcurrencyCaptureConfig::from_env().enabled);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "luce_prof_disabled.jsonl";
    std::filesystem::remove(path);
    ConcurrencyCaptureConfig off;
    off.output_path = path.string();
    {
        ConcurrencyCapture capture(off);
        CHECK(!capture.enabled());
        CHECK(capture.job_queued() == 0);
        CHECK(capture.begin_step(1, 1) == nullptr);
        capture.commit_step(nullptr);
        capture.record_request_admitted(1, "r", 1, 0, 0);
        capture.record_token_burst(1, 1, 1, 1);
        capture.record_request_finished(1, true, 1, 1);
        capture.flush();
    }
    CHECK(!std::filesystem::exists(path));

    // A null profile makes a phase scope free: not even a clock read.
    clock_reads = 0;
    { PhaseScope scope(nullptr, Phase::TargetCompute, fake_clock); }
    CHECK(clock_reads == 0);
}

void test_env_config() {
    set_env("LUCE_PROF", "step,concurrency");
    set_env("LUCE_PROF_OUT", "/tmp/x.jsonl");
    set_env("LUCE_PROF_MAX_ROUNDS", "7");
    set_env("LUCE_PROF_MAX_REQUESTS", "8");
    set_env("LUCE_PROF_MAX_TOKEN_BURSTS", "9");
    set_env("LUCE_PROF_CHECKPOINT_EVERY", "bogus");
    ConcurrencyCaptureConfig config = ConcurrencyCaptureConfig::from_env();
    CHECK(config.enabled);
    CHECK(config.output_path == "/tmp/x.jsonl");
    CHECK(config.max_rounds == 7);
    CHECK(config.max_requests == 8);
    CHECK(config.max_token_bursts == 9);
    CHECK(config.checkpoint_every_rounds == 0);  // unparsable -> default
    bool saw_prof = false;
    for (const auto & entry : config.run_env) {
        saw_prof = saw_prof || entry.first == "LUCE_PROF";
    }
    CHECK(saw_prof);

    set_env("LUCE_PROF", "1");
    set_env("LUCE_PROF_CHECKPOINT_EVERY", "3");
    config = ConcurrencyCaptureConfig::from_env();
    CHECK(config.enabled);
    CHECK(config.checkpoint_every_rounds == 3);
    for (const char * name : {
             "LUCE_PROF", "LUCE_PROF_OUT", "LUCE_PROF_MAX_ROUNDS",
             "LUCE_PROF_MAX_REQUESTS", "LUCE_PROF_MAX_TOKEN_BURSTS",
             "LUCE_PROF_CHECKPOINT_EVERY"}) {
        set_env(name, nullptr);
    }
}

void test_service_round_scope() {
    CHECK(current_service_round() == 0);
    {
        ServiceRoundScope outer(5);
        CHECK(current_service_round() == 5);
        {
            ServiceRoundScope inner(6);
            CHECK(current_service_round() == 6);
        }
        CHECK(current_service_round() == 5);
    }
    CHECK(current_service_round() == 0);
}

void test_bounded_arrays() {
    StepProfile step;
    for (size_t i = 0; i < kMaxProfileLanes + 3; ++i) {
        LaneProfile lane;
        lane.slot = static_cast<int32_t>(i);
        step.add_lane(lane);
    }
    CHECK(step.lane_count == kMaxProfileLanes);
    CHECK(step.dropped_lanes == 3);
    CHECK(step.find_lane(1, LaneKind::Decode) == &step.lanes[1]);
    CHECK(step.find_lane(1, LaneKind::Prefill) == nullptr);
    for (size_t i = 0; i < kMaxProfilePhases + 2; ++i) {
        step.add_phase({Phase::TargetCompute, 0, 1});
    }
    CHECK(step.phase_count == kMaxProfilePhases);
    CHECK(step.dropped_phases == 2);

    // PhaseScope measures relative to the step start with the given clock.
    StepProfile timed;
    timed.started_ns = 1000;
    fake_now = 1250;
    {
        PhaseScope scope(&timed, Phase::DraftCompute, fake_clock);
        fake_now = 1900;
    }
    CHECK(timed.phase_count == 1);
    CHECK(timed.phases[0].phase == Phase::DraftCompute);
    CHECK(timed.phases[0].start_offset_ns == 250);
    CHECK(timed.phases[0].duration_ns == 650);
}

// Drive the capture the way scheduler_loop and Qwen35SeqEngine do.
void run_round(ConcurrencyCapture & capture, uint64_t round_id,
               StepPath path, uint64_t request_id, uint32_t accepted) {
    const ServiceRoundScope round(round_id);
    StepProfile * profile = capture.begin_step(round_id, 2);
    CHECK(profile != nullptr);
    if (!profile) return;
    CHECK(current_service_round() == profile->round_id);
    profile->end_phase(Phase::SchedulerPlan, profile->started_ns);
    profile->planned_decode_lanes = 1;
    LaneProfile lane;
    lane.request_id = request_id;
    lane.slot = 0;
    profile->add_lane(lane);
    profile->path = path;
    profile->executed_decode_lanes = 1;
    if (path == StepPath::Chain) {
        { PhaseScope draft(profile, Phase::DraftCompute); }
        profile->spec_tree_width = 4;
        profile->spec_attempted_lanes = 1;
        profile->spec_proposed_draft_tokens = 3;
        profile->spec_verified_draft_tokens = 3;
        profile->spec_accepted_draft_tokens = accepted;
        profile->spec_durable_draft_tokens = accepted;
        for (size_t p = 1; p < 4; ++p) ++profile->proposed_by_position[p];
        for (size_t p = 1; p <= accepted; ++p) {
            ++profile->accepted_by_position[p];
        }
        LaneProfile * decode = profile->find_lane(0, LaneKind::Decode);
        CHECK(decode != nullptr);
        if (decode) {
            decode->spec = SpecDecision::Selected;
            decode->durable_draft_tokens = accepted;
        }
    }
    { PhaseScope compute(profile, Phase::TargetCompute); }
    const uint32_t consumed = accepted + 1;
    if (LaneProfile * decode = profile->find_lane(0, LaneKind::Decode)) {
        decode->pending_token_sampled = true;
        decode->scheduler_consumed_tokens = consumed;
        decode->pending_token_consumed = true;
    }
    capture.record_token_burst(request_id, round_id, steady_time_ns(), consumed);
    capture.commit_step(profile);
}

void test_capture_roundtrip(const std::string & kept_path) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "luce_prof_capture_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = kept_path.empty()
        ? (dir / "capture.jsonl").string() : kept_path;

    ConcurrencyCaptureConfig config;
    config.enabled = true;
    config.output_path = path;
    config.max_rounds = 3;
    config.max_requests = 2;
    config.max_token_bursts = 4;
    config.checkpoint_every_rounds = 2;
    config.model_name = "model \"q\"\n\x01";  // exercises JSON escaping
    config.arch = "qwen35";
    config.runtime_backend = "hip";
    config.max_concurrency = 2;
    config.run_env = {{"LUCE_PROF", "1"}};

    {
        ConcurrencyCapture capture(config);
        const uint64_t queued = capture.job_queued();
        capture.job_queued();
        capture.job_dequeued();
        capture.record_request_admitted(1, "chatcmpl-\"1\"", 32, queued,
                                        steady_time_ns());
        capture.record_request_admitted(2, "chatcmpl-2", 16, queued,
                                        steady_time_ns());
        capture.record_request_admitted(3, "dropped", 8, queued,
                                        steady_time_ns());
        capture.record_prefill_completed(1, steady_time_ns());

        run_round(capture, 1, StepPath::Packed, 1, 0);
        run_round(capture, 2, StepPath::Chain, 1, 2);
        // Round 2 wrote a checkpoint: requests still active report ok=null.
        const std::vector<json> checkpoint = check_schema(path, false);
        size_t active = 0;
        for (const json & record : checkpoint) {
            if (record.value("type", "") == "request" && record["ok"].is_null()) {
                ++active;
            }
        }
        CHECK(active == 2);
        CHECK(!std::filesystem::exists(path + ".tmp"));

        capture.record_request_finished(1, true, 4, steady_time_ns());
        run_round(capture, 3, StepPath::Chain, 2, 1);
        run_round(capture, 4, StepPath::Unknown, 2, 0);  // exceeds max_rounds
        capture.record_request_finished(2, false, 3, steady_time_ns());
        capture.flush();
    }

    const std::vector<json> records = check_schema(path, true);
    size_t steps = 0, requests = 0, bursts = 0;
    std::set<uint64_t> step_rounds;
    std::set<std::string> step_paths;
    for (const json & record : records) {
        const std::string type = record.value("type", "");
        if (type == "metadata") {
            CHECK(record["model_name"] == "model \"q\"\n\x01");
            CHECK(record["arch"] == "qwen35");
            CHECK(record["max_concurrency"] == 2);
            CHECK(record["max_rounds"] == 3);
            CHECK(record["checkpoint_every_rounds"] == 2);
            CHECK(record["env"]["LUCE_PROF"] == "1");
        } else if (type == "step") {
            ++steps;
            step_rounds.insert(record["round_id"].get<uint64_t>());
            step_paths.insert(record["path"].get<std::string>());
            CHECK(!record["phases"].empty());
            CHECK(record["lanes"].size() == 1);
            if (record["round_id"] == 1) CHECK(record["queue_depth"] == 1);
            if (record["path"] == "chain") {
                CHECK(record["proposed_by_position"].size() == 4);
                CHECK(record["proposed_by_position"][0] == 0);
                CHECK(record["lanes"][0]["spec"] == "selected");
            } else {
                CHECK(record["proposed_by_position"].empty());
            }
        } else if (type == "request") {
            ++requests;
            CHECK(!record["ok"].is_null());
            if (record["request_id"] == 1) {
                CHECK(record["ok"] == true);
                CHECK(record["response_id"] == "chatcmpl-\"1\"");
                CHECK(record["prefill_completed_ns"].get<uint64_t>() > 0);
                CHECK(record["first_token_ns"].get<uint64_t>() > 0);
            } else {
                CHECK(record["ok"] == false);
            }
        } else if (type == "token_burst") {
            ++bursts;
            // Bursts name scheduler rounds; round 4 only lost its step
            // record to the max_rounds bound.
            const uint64_t round = record["round_id"].get<uint64_t>();
            CHECK(round == 4 || step_rounds.count(round) == 1);
        } else if (type == "footer") {
            CHECK(record["dropped_steps"] == 1);
            CHECK(record["dropped_requests"] == 1);
            CHECK(record["dropped_token_bursts"] == 0);
        }
    }
    CHECK(steps == 3);
    CHECK(requests == 2);
    CHECK(bursts == 4);
    CHECK(step_paths == std::set<std::string>({"packed", "chain"}));
    if (kept_path.empty()) std::filesystem::remove_all(dir);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        test_disabled_is_inert();
        test_env_config();
        test_service_round_scope();
        test_bounded_arrays();
        test_capture_roundtrip(argc > 1 ? argv[1] : "");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL exception: %s\n", error.what());
        return 1;
    }
    if (failures == 0) std::printf("test_concurrency_capture: OK\n");
    return failures == 0 ? 0 : 1;
}
