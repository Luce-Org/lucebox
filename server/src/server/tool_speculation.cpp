#if defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE
#endif

#include "tool_speculation.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <utility>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <poll.h>
#  include <signal.h>
#  include <spawn.h>
#  include <sys/socket.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <features.h>
#    include <sched.h>
#  endif
#endif

#if !defined(_WIN32) && defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#  if __GLIBC_PREREQ(2, 34)
#    define DFLASH_TOOL_SPEC_HAS_CLOSEFROM 1
#  endif
#endif
#ifndef DFLASH_TOOL_SPEC_HAS_CLOSEFROM
#  define DFLASH_TOOL_SPEC_HAS_CLOSEFROM 0
#endif

namespace dflash::common {
namespace {

constexpr size_t kMaxExecutorRequestBytes = 64 * 1024;

bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) return values[middle];
    return (values[middle - 1] + values[middle]) / 2.0;
}

bool request_declares_tool(const json & tools, const std::string & name) {
    if (!tools.is_array() || name.empty()) return false;
    for (const auto & tool : tools) {
        if (!tool.is_object()) continue;
        if (tool.value("name", "") == name) return true;
        if (tool.contains("function") && tool["function"].is_object() &&
            tool["function"].value("name", "") == name) {
            return true;
        }
    }
    return false;
}

std::string format_cpu_affinity(const std::vector<int> & cpus) {
    std::string value;
    for (const int cpu : cpus) {
        if (!value.empty()) value.push_back(',');
        value += std::to_string(cpu);
    }
    return value;
}

#if !defined(_WIN32)
bool send_all_socket(int fd, const void * data, size_t bytes) {
    const char * cursor = static_cast<const char *>(data);
    while (bytes > 0) {
        int flags = 0;
#  if defined(MSG_NOSIGNAL)
        flags = MSG_NOSIGNAL;
#  endif
        const ssize_t written = ::send(fd, cursor, bytes, flags);
        if (written < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (written == 0) return false;
        cursor += written;
        bytes -= static_cast<size_t>(written);
    }
    return true;
}

void signal_executor_process_group(
        pid_t leader, int signal, bool leader_fallback = true) {
    if (leader <= 0) return;
    if (::kill(-leader, signal) != 0 && errno == ESRCH && leader_fallback) {
        // Defensive fallback for older platforms that ignored the requested
        // spawn process group.
        (void)::kill(leader, signal);
    }
}

std::vector<std::string> executor_environment(
        int resource_percentage,
        const std::string & accelerator_relation,
        const std::vector<int> & cpu_affinity) {
    std::vector<std::string> values;
    // Do not copy the long-running server's environment into a tool process:
    // it commonly contains model-provider keys and upstream credentials. Keep
    // only the small runtime surface needed by executable/script adapters.
    static constexpr const char * kInherited[] = {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "LD_LIBRARY_PATH",
        "DFLASH_TRACE_TRAINING_REPORT",
        "DFLASH_TRACE_WORKFLOW_REGISTRY",
    };
    for (const char * name : kInherited) {
        if (const char * value = std::getenv(name); value && *value) {
            values.push_back(std::string(name) + "=" + value);
        }
    }
    values.push_back("DFLASH_TOOL_SPECULATION=1");
    values.push_back(
        "DFLASH_TOOL_SPECULATION_RESOURCE_PERCENTAGE=" +
        std::to_string(resource_percentage));
    values.push_back(
        "DFLASH_TOOL_SPECULATION_ACCELERATOR_RELATION=" +
        accelerator_relation);
    if (!cpu_affinity.empty()) {
        values.push_back(
            "DFLASH_TOOL_SPECULATION_CPU_AFFINITY=" +
            format_cpu_affinity(cpu_affinity));
    }
    return values;
}

#  if defined(__linux__)
bool wait_for_child_cpu_affinity(
        pid_t child,
        const std::vector<int> & cpus,
        std::string & error) {
    if (cpus.empty()) {
        error.clear();
        return true;
    }
    cpu_set_t requested;
    CPU_ZERO(&requested);
    for (const int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            error = "executor CPU is outside CPU_SETSIZE: " +
                    std::to_string(cpu);
            return false;
        }
        CPU_SET(cpu, &requested);
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(250);
    do {
        cpu_set_t observed;
        CPU_ZERO(&observed);
        if (::sched_getaffinity(child, sizeof(observed), &observed) == 0) {
            bool matches = true;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (CPU_ISSET(cpu, &requested) != CPU_ISSET(cpu, &observed)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                error.clear();
                return true;
            }
        } else if (errno != EINTR) {
            error = std::string("executor sched_getaffinity failed: ") +
                    std::strerror(errno);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    error = "executor CPU affinity verification mismatch";
    return false;
}
#  endif
#endif

}  // namespace

bool tool_speculation_executor_isolation_supported() {
    return DFLASH_TOOL_SPEC_HAS_CLOSEFROM != 0;
}

bool CanonicalToolInvocation::from_parts(
        const std::string & name,
        const json & arguments,
        CanonicalToolInvocation & out,
        std::string & error) {
    if (name.empty()) {
        error = "tool name must not be empty";
        return false;
    }
    if (!arguments.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    out.name = name;
    out.arguments = arguments;
    // nlohmann::json's default object type is key ordered, so dump() is a
    // stable canonical identity independent of input object insertion order.
    out.arguments_json = arguments.dump();
    error.clear();
    return true;
}

bool CanonicalToolInvocation::from_tool_call(
        const ToolCall & call,
        CanonicalToolInvocation & out,
        std::string & error) {
    try {
        const json arguments = call.arguments.empty()
            ? json::object()
            : json::parse(call.arguments);
        return from_parts(call.name, arguments, out, error);
    } catch (const std::exception & exception) {
        error = std::string("authoritative tool arguments are invalid JSON: ") +
                exception.what();
        return false;
    }
}

bool build_tool_speculation_prediction(
        const std::string & name,
        const json & arguments,
        double confidence,
        ToolSpeculationPrediction & out,
        std::string & error) {
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0) {
        error = "tool prediction confidence must be between 0 and 1";
        return false;
    }
    CanonicalToolInvocation invocation;
    if (!CanonicalToolInvocation::from_parts(
            name, arguments, invocation, error)) {
        return false;
    }
    out.call = std::move(invocation);
    out.confidence = confidence;
    error.clear();
    return true;
}

bool parse_tool_speculation_prediction(
        const json & value,
        const json & tools,
        ToolSpeculationPrediction & out,
        std::string & error) {
    if (!value.is_object()) {
        error = "tool_speculation must be an object";
        return false;
    }
    if (!value.contains("call") || !value["call"].is_object()) {
        error = "tool_speculation.call must be an object";
        return false;
    }
    if (!value.contains("confidence") || !value["confidence"].is_number()) {
        error = "tool_speculation.confidence must be a number";
        return false;
    }
    const double confidence = value["confidence"].get<double>();
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0) {
        error = "tool_speculation.confidence must be between 0 and 1";
        return false;
    }

    const json & call = value["call"];
    if (!call.contains("name") || !call["name"].is_string()) {
        error = "tool_speculation.call.name must be a string";
        return false;
    }
    if (!call.contains("arguments")) {
        error = "tool_speculation.call.arguments is required";
        return false;
    }
    ToolSpeculationPrediction prediction;
    if (!build_tool_speculation_prediction(
            call["name"].get<std::string>(), call["arguments"], confidence,
            prediction, error)) {
        return false;
    }
    if (!request_declares_tool(tools, prediction.call.name)) {
        error = "tool_speculation.call.name is not declared in tools";
        return false;
    }
    out = std::move(prediction);
    error.clear();
    return true;
}

bool parse_tool_speculation_cpu_affinity(
        const std::string & value,
        std::vector<int> & out,
        std::string & error) {
    out.clear();
    if (value.empty()) {
        error = "tool CPU affinity must not be empty";
        return false;
    }
    size_t cursor = 0;
    while (cursor < value.size()) {
        const size_t comma = value.find(',', cursor);
        const size_t end = comma == std::string::npos ? value.size() : comma;
        const std::string token = value.substr(cursor, end - cursor);
        if (token.empty()) {
            error = "tool CPU affinity contains an empty item";
            out.clear();
            return false;
        }
        const size_t dash = token.find('-');
        auto parse_cpu = [&](const std::string & item, int & cpu) {
            if (item.empty() || !std::all_of(
                    item.begin(), item.end(), [](unsigned char character) {
                        return character >= '0' && character <= '9';
                    })) {
                return false;
            }
            char * parsed_end = nullptr;
            errno = 0;
            const long parsed = std::strtol(item.c_str(), &parsed_end, 10);
            if (errno != 0 || !parsed_end || *parsed_end != '\0' ||
                parsed < 0 || parsed > std::numeric_limits<int>::max()) {
                return false;
            }
            cpu = static_cast<int>(parsed);
            return true;
        };
        int first = -1;
        int last = -1;
        if (dash == std::string::npos) {
            if (!parse_cpu(token, first)) {
                error = "invalid tool CPU affinity item: " + token;
                out.clear();
                return false;
            }
            last = first;
        } else if (token.find('-', dash + 1) != std::string::npos ||
                   !parse_cpu(token.substr(0, dash), first) ||
                   !parse_cpu(token.substr(dash + 1), last) ||
                   first > last) {
            error = "invalid tool CPU affinity range: " + token;
            out.clear();
            return false;
        }
        if (static_cast<unsigned long long>(last) -
                static_cast<unsigned long long>(first) > 65535ULL) {
            error = "tool CPU affinity range is too large: " + token;
            out.clear();
            return false;
        }
        for (int cpu = first; cpu <= last; ++cpu) {
            out.push_back(cpu);
            if (cpu == std::numeric_limits<int>::max()) break;
        }
        if (comma == std::string::npos) break;
        cursor = comma + 1;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    error.clear();
    return true;
}

bool qualify_tool_speculation_cpu_affinity(
        ToolSpeculationConfig & config,
        std::string & error) {
    config.model_cpu_affinity.clear();
    config.cpu_affinity_isolated = false;
    if (config.cpu_affinity.empty()) {
        error.clear();
        return true;
    }
#if defined(__linux__)
    if (::access("/usr/bin/taskset", X_OK) != 0) {
        error = "tool CPU affinity requires executable /usr/bin/taskset";
        return false;
    }
    const long configured_cpus = ::sysconf(_SC_NPROCESSORS_CONF);
    if (configured_cpus <= 0) {
        error = "cannot determine configured CPU count";
        return false;
    }
    cpu_set_t model_set;
    CPU_ZERO(&model_set);
    if (::sched_getaffinity(0, sizeof(model_set), &model_set) != 0) {
        error = std::string("model sched_getaffinity failed: ") +
                std::strerror(errno);
        return false;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &model_set)) {
            config.model_cpu_affinity.push_back(cpu);
        }
    }
    if (config.model_cpu_affinity.empty()) {
        error = "model CPU affinity is empty";
        return false;
    }
    for (const int cpu : config.cpu_affinity) {
        if (cpu < 0 || cpu >= CPU_SETSIZE || cpu >= configured_cpus) {
            error = "tool CPU is not configured on this host: " +
                    std::to_string(cpu);
            return false;
        }
        if (CPU_ISSET(cpu, &model_set)) {
            error = "tool CPU affinity overlaps model CPU " +
                    std::to_string(cpu);
            return false;
        }
    }
    config.cpu_affinity_isolated = true;
    error.clear();
    return true;
#else
    error = "tool CPU affinity isolation is supported only on Linux";
    return false;
#endif
}

bool ToolSpeculationPolicy::load_file(
        const std::string & path, std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open tool-speculation profile: " + path;
        lanes_.clear();
        baseline_task_ms_ = 0.0;
        profile_status_ = "qualified";
        executor_contract_.clear();
        return false;
    }
    try {
        json report;
        input >> report;
        return load_json(report, error);
    } catch (const std::exception & exception) {
        error = std::string("invalid tool-speculation profile JSON: ") +
                exception.what();
        lanes_.clear();
        baseline_task_ms_ = 0.0;
        profile_status_ = "qualified";
        executor_contract_.clear();
        return false;
    }
}

bool ToolSpeculationPolicy::load_json(
        const json & report, std::string & error) {
    lanes_.clear();
    baseline_task_ms_ = 0.0;
    profile_status_ = "qualified";
    executor_contract_.clear();
    if (!report.is_object() || !report.contains("path_summary") ||
        !report["path_summary"].is_object() ||
        report["path_summary"].empty()) {
        error = "tool-speculation profile needs a non-empty path_summary";
        return false;
    }

    if (!report.contains("profile_status") ||
        !report["profile_status"].is_string()) {
        error = "tool-speculation profile needs profile_status=qualified";
        return false;
    }
    profile_status_ = report["profile_status"].get<std::string>();
    if (profile_status_ != "qualified") {
        error = "tool-speculation profile_status must be qualified";
        return false;
    }
    if (!report.contains("executor") || !report["executor"].is_string()) {
        error = "tool-speculation profile needs an executor contract";
        return false;
    }
    executor_contract_ = report["executor"].get<std::string>();
    if (executor_contract_.empty()) {
        error = "tool-speculation executor contract cannot be empty";
        return false;
    }

    std::vector<double> controls;
    try {
        for (auto item = report["path_summary"].begin();
             item != report["path_summary"].end(); ++item) {
            size_t parsed = 0;
            const int resource_percentage = std::stoi(item.key(), &parsed);
            if (parsed != item.key().size() ||
                resource_percentage < 1 || resource_percentage > 100) {
                throw std::runtime_error(
                    "invalid resource percentage " + item.key());
            }
            const json & paths = item.value();
            const json & hit = paths.at("hit");
            const json & miss = paths.at("miss");
            const double hit_control = hit.at("control_task_mean_ms").get<double>();
            const double miss_control = miss.at("control_task_mean_ms").get<double>();
            const double hit_task = hit.at("speculative_task_mean_ms").get<double>();
            const double miss_task = miss.at("speculative_task_mean_ms").get<double>();
            const double slowdown_percent = std::max(
                hit.at("model_slowdown_percent").get<double>(),
                miss.at("model_slowdown_percent").get<double>());
            bool decode_interference_qualified = false;
            if (paths.contains("decode_interference_qualified")) {
                if (!paths["decode_interference_qualified"].is_boolean()) {
                    throw std::runtime_error(
                        "decode_interference_qualified must be boolean");
                }
                decode_interference_qualified =
                    paths["decode_interference_qualified"].get<bool>();
            }
            if (!paths.contains("accelerator_relation") ||
                !paths["accelerator_relation"].is_string()) {
                throw std::runtime_error(
                    "accelerator_relation must be explicit");
            }
            const std::string accelerator_relation =
                paths["accelerator_relation"].get<std::string>();
            if (accelerator_relation != "non_accelerator" &&
                accelerator_relation != "separate_physical_gpu") {
                throw std::runtime_error(
                    "accelerator_relation must be non_accelerator or "
                    "separate_physical_gpu");
            }
            if (!finite_positive(hit_control) ||
                !finite_positive(miss_control) ||
                !finite_positive(hit_task) ||
                !finite_positive(miss_task) ||
                !std::isfinite(slowdown_percent) || slowdown_percent < -100.0) {
                throw std::runtime_error("non-positive or non-finite profile latency");
            }
            const double control = (hit_control + miss_control) / 2.0;
            lanes_.push_back({
                resource_percentage,
                control,
                hit_task,
                miss_task,
                1.0 + slowdown_percent / 100.0,
                decode_interference_qualified,
                accelerator_relation,
            });
            controls.push_back(control);
        }
    } catch (const std::exception & exception) {
        error = std::string("invalid tool-speculation path_summary: ") +
                exception.what();
        lanes_.clear();
        profile_status_ = "qualified";
        executor_contract_.clear();
        return false;
    }

    std::sort(lanes_.begin(), lanes_.end(),
              [](const auto & left, const auto & right) {
                  return left.resource_percentage <
                         right.resource_percentage;
              });
    for (size_t index = 1; index < lanes_.size(); ++index) {
        if (lanes_[index - 1].resource_percentage ==
            lanes_[index].resource_percentage) {
            error =
                "tool-speculation profile has duplicate resource percentages";
            lanes_.clear();
            return false;
        }
    }
    baseline_task_ms_ = median(std::move(controls));
    error.clear();
    return true;
}

ToolSpeculationAdmission ToolSpeculationPolicy::choose(
        double confidence,
        double max_model_slowdown_ratio) const {
    ToolSpeculationAdmission decision;
    decision.expected_task_ms = baseline_task_ms_;
    if (lanes_.empty() || !finite_positive(baseline_task_ms_)) {
        decision.reason = "profile_unavailable";
        return decision;
    }
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0) {
        decision.reason = "invalid_confidence";
        return decision;
    }
    if (!std::isfinite(max_model_slowdown_ratio) ||
        max_model_slowdown_ratio < 1.0) {
        decision.reason = "invalid_slowdown_guardrail";
        return decision;
    }

    bool qualified_lane_available = false;
    bool lane_passed_guardrail = false;
    double best = baseline_task_ms_;
    for (const ToolSpeculationLane & lane : lanes_) {
        // Token speculation is an invariant, not a fallback choice. A lane
        // may overlap DS4/DSpark only after its exact executor and placement
        // passed the output-identity interference gate.
        if (!lane.decode_interference_qualified) continue;
        qualified_lane_available = true;
        if (lane.model_slowdown_ratio > max_model_slowdown_ratio) continue;
        lane_passed_guardrail = true;
        const double expected =
            confidence * lane.hit_task_ms +
            (1.0 - confidence) * lane.miss_task_ms;
        if (expected < best) {
            best = expected;
            decision.admitted = true;
            decision.resource_percentage = lane.resource_percentage;
            decision.expected_task_ms = expected;
            decision.decode_interference_qualified =
                lane.decode_interference_qualified;
            decision.accelerator_relation = lane.accelerator_relation;
        }
    }
    if (!decision.admitted) {
        decision.reason = !qualified_lane_available
            ? "decode_interference_unqualified"
            : lane_passed_guardrail
                ? "below_profile_break_even"
                : "model_slowdown_guardrail";
        return decision;
    }
    decision.expected_speedup = baseline_task_ms_ / best;
    decision.reason = "expected_latency_gain";
    return decision;
}

bool ToolSpeculationConfig::allows(const std::string & name) const {
    return std::find(allowed_tools.begin(), allowed_tools.end(), name) !=
           allowed_tools.end();
}

ToolSpeculationAttempt::ToolSpeculationAttempt(
        const ToolSpeculationConfig & config,
        const ToolSpeculationPrediction & prediction,
        const std::string & request_id)
    : config_(config)
    , prediction_(prediction)
    , request_id_(request_id) {
    if (!config_.enabled()) {
        admission_.reason = "engine_disabled";
    } else if (!config_.allows(prediction_.call.name)) {
        admission_.reason = "tool_not_allowlisted";
    } else {
        admission_ = config_.policy.choose(
            prediction_.confidence, config_.max_model_slowdown_ratio);
    }
}

ToolSpeculationAttempt::~ToolSpeculationAttempt() {
    if (!resolved_) terminate_executor();
}

std::unique_ptr<ToolSpeculationAttempt> ToolSpeculationAttempt::create(
        const ToolSpeculationConfig & config,
        const ToolSpeculationPrediction & prediction,
        const std::string & request_id) {
    return std::unique_ptr<ToolSpeculationAttempt>(
        new ToolSpeculationAttempt(config, prediction, request_id));
}

void ToolSpeculationAttempt::start() {
    if (started_) return;
    started_ = true;
    if (!admission_.admitted) return;
    const json request = {
        {"protocol", "dflash.tool-speculation.v1"},
        {"request_id", request_id_},
        {"mode", "speculative"},
        {"resource_percentage", admission_.resource_percentage},
        {"accelerator_relation", admission_.accelerator_relation},
        {"cpu_affinity", config_.cpu_affinity},
        {"cpu_affinity_isolated", config_.cpu_affinity_isolated},
        {"call", {
            {"name", prediction_.call.name},
            {"arguments", prediction_.call.arguments},
        }},
    };
    started_at_ = std::chrono::steady_clock::now();
#if defined(_WIN32)
    launch_error_ = "tool speculation child executors are not implemented on Windows";
    return;
#else
    const std::string payload = request.dump() + "\n";
    if (payload.size() > kMaxExecutorRequestBytes) {
        launch_error_ = "executor request exceeds 64 KiB";
        return;
    }

    int input_socket[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, input_socket) != 0) {
        launch_error_ = std::string("executor stdin socket failed: ") +
                        std::strerror(errno);
        return;
    }
#  if defined(SO_NOSIGPIPE)
    int no_sigpipe = 1;
    ::setsockopt(input_socket[0], SOL_SOCKET, SO_NOSIGPIPE,
                 &no_sigpipe, sizeof(no_sigpipe));
#  endif
    int output_pipe[2] = {-1, -1};
    if (::pipe(output_pipe) != 0) {
        launch_error_ = std::string("executor stdout pipe failed: ") +
                        std::strerror(errno);
        ::close(input_socket[0]);
        ::close(input_socket[1]);
        return;
    }

    posix_spawn_file_actions_t actions;
    int spawn_status = posix_spawn_file_actions_init(&actions);
    const bool actions_initialized = spawn_status == 0;
    posix_spawnattr_t attributes;
    const int attributes_status = posix_spawnattr_init(&attributes);
    const bool attributes_initialized = attributes_status == 0;
    if (spawn_status == 0 && attributes_status != 0) {
        spawn_status = attributes_status;
    }
    if (spawn_status == 0) {
        spawn_status = posix_spawnattr_setpgroup(&attributes, 0);
    }
    if (spawn_status == 0) {
        spawn_status = posix_spawnattr_setflags(
            &attributes, POSIX_SPAWN_SETPGROUP);
    }
    if (spawn_status == 0) {
        spawn_status = posix_spawn_file_actions_adddup2(
            &actions, input_socket[1], STDIN_FILENO);
    }
    if (spawn_status == 0) {
        spawn_status = posix_spawn_file_actions_adddup2(
            &actions, output_pipe[1], STDOUT_FILENO);
    }
    if (spawn_status == 0) {
        spawn_status = posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
    }
    if (spawn_status == 0) {
        spawn_status = posix_spawn_file_actions_addclose(&actions, input_socket[0]);
    }
    if (spawn_status == 0 && input_socket[1] != STDIN_FILENO) {
        spawn_status = posix_spawn_file_actions_addclose(&actions, input_socket[1]);
    }
    if (spawn_status == 0 && output_pipe[1] != STDOUT_FILENO) {
        spawn_status = posix_spawn_file_actions_addclose(&actions, output_pipe[1]);
    }
#  if DFLASH_TOOL_SPEC_HAS_CLOSEFROM
    // The executor receives only stdin/stdout/stderr. In particular, it must
    // not inherit listening sockets, live client connections, model IPC pipes,
    // or accelerator descriptors from the long-running server.
    if (spawn_status == 0) {
        spawn_status = posix_spawn_file_actions_addclosefrom_np(
            &actions, STDERR_FILENO + 1);
    }
#  endif

    std::vector<std::string> env_storage = executor_environment(
        admission_.resource_percentage, admission_.accelerator_relation,
        config_.cpu_affinity);
    std::vector<char *> env;
    env.reserve(env_storage.size() + 1);
    for (std::string & value : env_storage) env.push_back(value.data());
    env.push_back(nullptr);

    std::string executable = config_.cpu_affinity.empty()
        ? config_.executor_path
        : "/usr/bin/taskset";
    std::string protocol_arg = "--dflash-tool-spec-v1";
    std::string affinity_arg = format_cpu_affinity(config_.cpu_affinity);
    std::string taskset_cpu_arg = "-c";
    std::vector<char *> argv;
    argv.push_back(executable.data());
    if (!config_.cpu_affinity.empty()) {
        // taskset applies the mask before execve(), so no executor startup
        // code can run on the model CPUs. The request payload remains withheld
        // until the parent verifies the resulting mask below.
        argv.push_back(taskset_cpu_arg.data());
        argv.push_back(affinity_arg.data());
        argv.push_back(config_.executor_path.data());
    }
    argv.push_back(protocol_arg.data());
    argv.push_back(nullptr);
    pid_t child = -1;
    if (spawn_status == 0) {
        spawn_status = ::posix_spawn(
            &child, executable.c_str(), &actions, &attributes, argv.data(),
            env.data());
    }
    if (actions_initialized) {
        posix_spawn_file_actions_destroy(&actions);
    }
    if (attributes_initialized) {
        posix_spawnattr_destroy(&attributes);
    }
    ::close(input_socket[1]);
    ::close(output_pipe[1]);
    if (spawn_status != 0) {
        launch_error_ = std::string("executor spawn failed: ") +
                        std::strerror(spawn_status);
        ::close(input_socket[0]);
        ::close(output_pipe[0]);
        return;
    }

#  if defined(__linux__)
    if (!config_.cpu_affinity.empty()) {
        std::string affinity_error;
        if (!wait_for_child_cpu_affinity(
                child, config_.cpu_affinity, affinity_error)) {
            signal_executor_process_group(child, SIGKILL);
            int child_status = 0;
            while (::waitpid(child, &child_status, 0) < 0 && errno == EINTR) {}
            ::close(input_socket[0]);
            ::close(output_pipe[0]);
            launch_error_ = std::move(affinity_error);
            return;
        }
    }
#  else
    if (!config_.cpu_affinity.empty()) {
        signal_executor_process_group(child, SIGKILL);
        int child_status = 0;
        while (::waitpid(child, &child_status, 0) < 0 && errno == EINTR) {}
        ::close(input_socket[0]);
        ::close(output_pipe[0]);
        launch_error_ = "tool CPU affinity isolation is supported only on Linux";
        return;
    }
#  endif

    child_pid_ = static_cast<int>(child);
    child_stdin_fd_ = input_socket[0];
    child_stdout_fd_ = output_pipe[0];
    const int flags = ::fcntl(child_stdout_fd_, F_GETFL, 0);
    if (flags >= 0) {
        ::fcntl(child_stdout_fd_, F_SETFL, flags | O_NONBLOCK);
    }
    running_ = true;
    if (!send_all_socket(child_stdin_fd_, payload.data(), payload.size())) {
        launch_error_ = std::string("executor request write failed: ") +
                        std::strerror(errno);
        terminate_executor();
        return;
    }
    std::fprintf(stderr,
        "[tool-spec] launched request=%s tool=%s confidence=%.3f "
        "resource=%d%%\n",
        request_id_.c_str(), prediction_.call.name.c_str(),
        prediction_.confidence, admission_.resource_percentage);
#endif
}

json ToolSpeculationAttempt::base_metadata() const {
    json metadata = {
        {"protocol", "dflash.tool-speculation.v1"},
        {"confidence", prediction_.confidence},
        {"prediction", {
            {"name", prediction_.call.name},
            {"arguments", prediction_.call.arguments},
        }},
        {"resource_percentage",
         admission_.admitted
            ? json(admission_.resource_percentage)
            : json(nullptr)},
        {"expected_speedup",
         admission_.admitted ? json(admission_.expected_speedup) : json(nullptr)},
        {"decode_interference_qualified",
         admission_.admitted
            ? json(admission_.decode_interference_qualified)
            : json(nullptr)},
        {"accelerator_relation",
         admission_.admitted
            ? json(admission_.accelerator_relation)
            : json(nullptr)},
        {"cpu_affinity", config_.cpu_affinity},
        {"cpu_affinity_isolated", config_.cpu_affinity_isolated},
    };
    return metadata;
}

bool ToolSpeculationAttempt::send_control(const char * operation) {
#if defined(_WIN32)
    (void)operation;
    return false;
#else
    if (child_stdin_fd_ < 0 || !operation || !*operation) return false;
    const std::string command = json({
        {"protocol", "dflash.tool-speculation.v1"},
        {"request_id", request_id_},
        {"op", operation},
        {"authoritative_resource_percentage", 100},
    }).dump() + "\n";
    return send_all_socket(
        child_stdin_fd_, command.data(), command.size());
#endif
}

void ToolSpeculationAttempt::terminate_executor(bool allow_control_grace) {
#if !defined(_WIN32)
    if (child_stdin_fd_ >= 0) {
        ::close(child_stdin_fd_);
        child_stdin_fd_ = -1;
    }
    if (child_stdout_fd_ >= 0) {
        ::close(child_stdout_fd_);
        child_stdout_fd_ = -1;
    }
    if (child_pid_ > 0) {
        const pid_t pid = static_cast<pid_t>(child_pid_);
        auto wait_until = [&](std::chrono::steady_clock::time_point deadline) {
            int status = 0;
            while (std::chrono::steady_clock::now() < deadline) {
                const pid_t waited = ::waitpid(pid, &status, WNOHANG);
                if (waited == pid || (waited < 0 && errno == ECHILD)) {
                    child_pid_ = -1;
                    running_ = false;
                    return true;
                }
                if (waited < 0 && errno != EINTR) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            return false;
        };
        const int grace_ms = std::max(0, config_.cancel_grace_ms);
        if (allow_control_grace && wait_until(
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(grace_ms))) {
            // The executor contract does not permit detached descendants.
            // Clean up any process that outlived its group leader.
            signal_executor_process_group(pid, SIGKILL, false);
            return;
        }
        signal_executor_process_group(pid, SIGTERM);
        const int term_grace_ms = allow_control_grace
            ? std::min(20, grace_ms) : grace_ms;
        if (wait_until(std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(term_grace_ms))) {
            signal_executor_process_group(pid, SIGKILL, false);
            return;
        }
        int status = 0;
        signal_executor_process_group(pid, SIGKILL);
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        child_pid_ = -1;
    }
#endif
    running_ = false;
}

bool ToolSpeculationAttempt::collect_executor_result(
        json & result,
        double & wait_ms,
        std::string & error) {
#if defined(_WIN32)
    (void)result;
    wait_ms = 0.0;
    error = "tool speculation executors are not implemented on Windows";
    return false;
#else
    const auto wait_started = std::chrono::steady_clock::now();
    const auto deadline = started_at_ +
        std::chrono::milliseconds(std::max(1, config_.timeout_ms));
    if (wait_started >= deadline) {
        error = "executor_timeout";
        terminate_executor();
        wait_ms = 0.0;
        return false;
    }
    std::string output;
    bool eof = false;
    while (!eof) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            error = "executor_timeout";
            terminate_executor();
            wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wait_started).count();
            return false;
        }
        const int remaining_ms = std::max(1, static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now).count()));
        pollfd descriptor{child_stdout_fd_, POLLIN | POLLHUP, 0};
        const int polled = ::poll(&descriptor, 1, remaining_ms);
        if (polled < 0) {
            if (errno == EINTR) continue;
            error = std::string("executor_poll_failed: ") + std::strerror(errno);
            terminate_executor();
            return false;
        }
        if (polled == 0) continue;
        if (descriptor.revents & (POLLERR | POLLNVAL)) {
            error = "executor_stdout_failed";
            terminate_executor();
            return false;
        }
        if (descriptor.revents & (POLLIN | POLLHUP)) {
            char buffer[8192];
            while (true) {
                const ssize_t count = ::read(
                    child_stdout_fd_, buffer, sizeof(buffer));
                if (count > 0) {
                    if (output.size() + static_cast<size_t>(count) >
                        config_.max_result_bytes) {
                        error = "executor_result_too_large";
                        terminate_executor();
                        return false;
                    }
                    output.append(buffer, static_cast<size_t>(count));
                    continue;
                }
                if (count == 0) {
                    eof = true;
                    break;
                }
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                error = std::string("executor_read_failed: ") +
                        std::strerror(errno);
                terminate_executor();
                return false;
            }
        }
    }
    ::close(child_stdout_fd_);
    child_stdout_fd_ = -1;

    int child_status = 0;
    while (true) {
        const pid_t waited = ::waitpid(
            static_cast<pid_t>(child_pid_), &child_status, WNOHANG);
        if (waited == static_cast<pid_t>(child_pid_)) break;
        if (waited < 0) {
            if (errno == EINTR) continue;
            error = std::string("executor_wait_failed: ") +
                    std::strerror(errno);
            child_pid_ = -1;
            running_ = false;
            return false;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            error = "executor_exit_timeout";
            terminate_executor();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // A successful adapter must not leave detached subprocesses behind.
    signal_executor_process_group(
        static_cast<pid_t>(child_pid_), SIGKILL, false);
    child_pid_ = -1;
    running_ = false;
    wait_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wait_started).count();
    if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
        error = WIFEXITED(child_status)
            ? "executor_exit_" + std::to_string(WEXITSTATUS(child_status))
            : "executor_terminated";
        return false;
    }

    try {
        const json envelope = json::parse(output);
        if (!envelope.is_object() || !envelope.value("ok", false) ||
            !envelope.contains("result")) {
            error = "executor_rejected_or_invalid_envelope";
            return false;
        }
        result = envelope["result"];
        error.clear();
        return true;
    } catch (const std::exception & exception) {
        error = std::string("executor_invalid_json: ") + exception.what();
        return false;
    }
#endif
}

json ToolSpeculationAttempt::resolve(
        const std::vector<ToolCall> & authoritative_calls) {
    if (resolved_) {
        json metadata = base_metadata();
        metadata["status"] = "failed";
        metadata["reason"] = "already_resolved";
        return metadata;
    }
    resolved_ = true;
    json metadata = base_metadata();
    if (!admission_.admitted) {
        metadata["status"] = "deferred";
        metadata["reason"] = admission_.reason;
        return metadata;
    }
    if (!launch_error_.empty() || !running_) {
        metadata["status"] = "failed";
        metadata["reason"] = "executor_launch_failed";
        metadata["detail"] = launch_error_.empty()
            ? "executor did not start" : launch_error_;
        terminate_executor();
        return metadata;
    }
    if (authoritative_calls.size() != 1) {
        send_control("cancel");
        terminate_executor(true);
        metadata["status"] = "miss";
        metadata["reason"] = "authoritative_call_count";
        return metadata;
    }

    CanonicalToolInvocation authoritative;
    std::string canonical_error;
    if (!CanonicalToolInvocation::from_tool_call(
            authoritative_calls[0], authoritative, canonical_error)) {
        send_control("cancel");
        terminate_executor(true);
        metadata["status"] = "miss";
        metadata["reason"] = "invalid_authoritative_call";
        return metadata;
    }
    if (!(authoritative == prediction_.call)) {
        send_control("cancel");
        terminate_executor(true);
        metadata["status"] = "miss";
        metadata["reason"] = "invocation_mismatch";
        return metadata;
    }

    json result;
    double wait_ms = 0.0;
    std::string executor_error;
    const bool commit_signal_sent = send_control("commit");
#if !defined(_WIN32)
    if (child_stdin_fd_ >= 0) {
        ::close(child_stdin_fd_);
        child_stdin_fd_ = -1;
    }
#endif
    if (!collect_executor_result(result, wait_ms, executor_error)) {
        metadata["status"] = "failed";
        metadata["reason"] = "speculative_executor_failure";
        metadata["detail"] = executor_error;
        metadata["commit_signal_sent"] = commit_signal_sent;
        metadata["commit_wait_ms"] = wait_ms;
        return metadata;
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started_at_).count();
    metadata["status"] = "hit";
    metadata["call_id"] = authoritative_calls[0].id;
    metadata["result"] = std::move(result);
    metadata["commit_signal_sent"] = commit_signal_sent;
    metadata["executor_wall_ms"] = wall_ms;
    metadata["commit_wait_ms"] = wait_ms;
    std::fprintf(stderr,
        "[tool-spec] hit request=%s tool=%s resource=%d%% "
        "wall_ms=%.1f wait_ms=%.1f\n",
        request_id_.c_str(), prediction_.call.name.c_str(),
        admission_.resource_percentage, wall_ms, wait_ms);
    return metadata;
}

json ToolSpeculationAttempt::cancel(const std::string & reason) {
    if (!resolved_) {
        resolved_ = true;
        send_control("cancel");
        terminate_executor(true);
    }
    json metadata = base_metadata();
    metadata["status"] = "cancelled";
    metadata["reason"] = reason;
    return metadata;
}

std::string render_tool_speculation_sse(
        ApiFormat api_format,
        const std::string & request_id,
        const std::string & model,
        const json & metadata) {
    switch (api_format) {
    case ApiFormat::OPENAI_CHAT: {
        const json event = {
            {"id", request_id},
            {"object", "chat.completion.chunk"},
            {"model", model},
            {"choices", json::array()},
            {"dflash_tool_speculation", metadata},
        };
        return "data: " + event.dump() + "\n\n";
    }
    case ApiFormat::ANTHROPIC: {
        const json event = {
            {"type", "dflash_tool_speculation"},
            {"dflash_tool_speculation", metadata},
        };
        return "event: dflash_tool_speculation\ndata: " +
               event.dump() + "\n\n";
    }
    case ApiFormat::RESPONSES: {
        const json event = {
            {"type", "response.dflash_tool_speculation"},
            {"response_id", request_id},
            {"dflash_tool_speculation", metadata},
        };
        return "event: response.dflash_tool_speculation\ndata: " +
               event.dump() + "\n\n";
    }
    default:
        return "data: " + json({{"dflash_tool_speculation", metadata}}).dump() +
               "\n\n";
    }
}

}  // namespace dflash::common
