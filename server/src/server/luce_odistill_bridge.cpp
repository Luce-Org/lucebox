#include "luce_odistill_bridge.h"

#include "common/gguf_inspect.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <unordered_set>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace dflash::common {

namespace {

constexpr size_t kMaxManifestBytes = 1024 * 1024;
constexpr size_t kMaxEventBytes = 32 * 1024 * 1024;
constexpr size_t kMaxQueuedBytes = 64 * 1024 * 1024;
constexpr size_t kMaxQueuedEvents = 64;

bool is_sha256(const nlohmann::json & value) {
    if (!value.is_string()) return false;
    const std::string text = value.get<std::string>();
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

bool has_exact_keys(const nlohmann::json & value,
                    std::initializer_list<const char *> keys) {
    if (!value.is_object() || value.size() != keys.size()) return false;
    for (const char * key : keys) {
        if (!value.contains(key)) return false;
    }
    return true;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return value;
}

std::string trim_ascii(std::string value) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && is_space((unsigned char)value.front())) {
        value.erase(value.begin());
    }
    while (!value.empty() && is_space((unsigned char)value.back())) {
        value.pop_back();
    }
    return value;
}

bool truthy(const std::string & value) {
    const std::string normalized = lower_ascii(trim_ascii(value));
    return normalized == "1" || normalized == "true" ||
           normalized == "yes" || normalized == "on";
}

}  // namespace

bool parse_luce_odistill_selection(const nlohmann::json & value,
                                   LuceODistillSelection & out,
                                   std::string & error) {
    out = LuceODistillSelection{};
    if (!has_exact_keys(value, {
            "schema_version", "kind", "experimental", "selection_state",
            "release_id", "source_artifact_sha256", "runtime_artifact",
            "target_profile_sha256", "drafter_profile_sha256",
            "evaluation_report_sha256"})) {
        error = "selection must contain the exact versioned fields";
        return false;
    }
    if (value["schema_version"] != 1 ||
        value["kind"] != "luce_odistill_lucebox_selection" ||
        value["experimental"] != true) {
        error = "selection schema is not the experimental Lucebox contract";
        return false;
    }
    if (!value["selection_state"].is_string()) {
        error = "selection_state must be a string";
        return false;
    }
    const std::string state = value["selection_state"].get<std::string>();
    if (state == "baseline") {
        if (!value["release_id"].is_null() ||
            !value["evaluation_report_sha256"].is_null()) {
            error = "baseline selection cannot name a release or report";
            return false;
        }
    } else if (state == "promoted") {
        if (!value["release_id"].is_string() ||
            value["release_id"].get<std::string>().rfind("release-", 0) != 0 ||
            !is_sha256(value["evaluation_report_sha256"])) {
            error = "promoted selection requires a release and report digest";
            return false;
        }
    } else {
        error = "selection_state must be baseline or promoted";
        return false;
    }
    for (const char * key : {
             "source_artifact_sha256", "target_profile_sha256",
             "drafter_profile_sha256"}) {
        if (!is_sha256(value[key])) {
            error = std::string(key) + " must be a lowercase SHA-256 digest";
            return false;
        }
    }
    const auto & runtime = value["runtime_artifact"];
    if (!has_exact_keys(runtime, {"format", "path", "sha256"}) ||
        runtime["format"] != "gguf" || !runtime["path"].is_string() ||
        !is_sha256(runtime["sha256"])) {
        error = "runtime_artifact must bind one absolute GGUF path and digest";
        return false;
    }
    const std::filesystem::path runtime_path(runtime["path"].get<std::string>());
    if (!runtime_path.is_absolute()) {
        error = "runtime_artifact.path must be absolute";
        return false;
    }

    out.configured = true;
    out.selection_state = state;
    if (value["release_id"].is_string()) {
        out.release_id = value["release_id"].get<std::string>();
    }
    out.source_artifact_sha256 = value["source_artifact_sha256"].get<std::string>();
    out.runtime_artifact_path = runtime_path.string();
    out.runtime_artifact_sha256 = runtime["sha256"].get<std::string>();
    out.target_profile_sha256 = value["target_profile_sha256"].get<std::string>();
    out.drafter_profile_sha256 = value["drafter_profile_sha256"].get<std::string>();
    if (value["evaluation_report_sha256"].is_string()) {
        out.evaluation_report_sha256 =
            value["evaluation_report_sha256"].get<std::string>();
    }
    return true;
}

bool load_luce_odistill_selection(const std::string & manifest_path,
                                  LuceODistillSelection & out,
                                  std::string & error) {
#if !defined(_WIN32)
    struct stat metadata{};
    if (lstat(manifest_path.c_str(), &metadata) != 0 ||
        !S_ISREG(metadata.st_mode) || S_ISLNK(metadata.st_mode) ||
        (metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        error = "selection manifest must be a real non-shared-writable regular file";
        return false;
    }
#endif
    std::ifstream stream(manifest_path, std::ios::binary);
    if (!stream) {
        error = "cannot open selection manifest";
        return false;
    }
    std::string raw((std::istreambuf_iterator<char>(stream)),
                    std::istreambuf_iterator<char>());
    if (raw.empty() || raw.size() > kMaxManifestBytes) {
        error = "selection manifest size is invalid";
        return false;
    }
    nlohmann::json value;
    try {
        value = nlohmann::json::parse(raw);
    } catch (const std::exception &) {
        error = "selection manifest is not valid JSON";
        return false;
    }
    LuceODistillSelection parsed;
    if (!parse_luce_odistill_selection(value, parsed, error)) return false;

#if !defined(_WIN32)
    struct stat runtime_metadata{};
    if (lstat(parsed.runtime_artifact_path.c_str(), &runtime_metadata) != 0 ||
        !S_ISREG(runtime_metadata.st_mode) || S_ISLNK(runtime_metadata.st_mode) ||
        (runtime_metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        error = "selected runtime artifact must be a real non-shared-writable regular file";
        return false;
    }
#endif
    const GgufMetadata runtime =
        read_gguf_metadata(parsed.runtime_artifact_path, /*compute_sha256=*/true);
    if (!runtime.ok || runtime.sha256.empty()) {
        error = "selected runtime artifact is not a readable GGUF";
        return false;
    }
    if (runtime.sha256 != parsed.runtime_artifact_sha256) {
        error = "selected runtime artifact SHA-256 does not match the manifest";
        return false;
    }
    out = std::move(parsed);
    return true;
}

nlohmann::json luce_odistill_runtime_drafter_json(
        const LuceODistillSelection & selection) {
    return {
        {"selection_state", selection.selection_state},
        {"release_id", selection.release_id.empty()
            ? nlohmann::json(nullptr) : nlohmann::json(selection.release_id)},
        {"source_artifact_sha256", selection.source_artifact_sha256},
        {"runtime_artifact_sha256", selection.runtime_artifact_sha256},
        {"evaluation_report_sha256", selection.evaluation_report_sha256.empty()
            ? nlohmann::json(nullptr)
            : nlohmann::json(selection.evaluation_report_sha256)},
        {"target_profile_sha256", selection.target_profile_sha256},
        {"drafter_profile_sha256", selection.drafter_profile_sha256},
    };
}

LuceODistillConsent parse_luce_odistill_capture_headers(
        const std::vector<std::pair<std::string, std::string>> & headers) {
    LuceODistillConsent result;
    std::unordered_map<std::string, std::string> selected;
    std::unordered_set<std::string> duplicates;
    for (const auto & [raw_name, raw_value] : headers) {
        const std::string name = lower_ascii(trim_ascii(raw_name));
        if (name != "x-luce-odistill-consent" &&
            name != "x-luce-odistill-subject" &&
            name != "x-luce-odistill-conversation" &&
            name != "x-luce-odistill-profile") {
            continue;
        }
        if (!selected.emplace(name, trim_ascii(raw_value)).second) {
            duplicates.insert(name);
        }
    }
    const auto consent = selected.find("x-luce-odistill-consent");
    result.requested = consent != selected.end() && truthy(consent->second);
    if (!result.requested || !duplicates.empty()) return result;
    const auto subject = selected.find("x-luce-odistill-subject");
    if (subject == selected.end() || subject->second.empty() ||
        subject->second.size() > 512) {
        return result;
    }
    result.subject = subject->second;
    const auto conversation = selected.find("x-luce-odistill-conversation");
    if (conversation != selected.end()) {
        if (conversation->second.empty() || conversation->second.size() > 512) {
            return result;
        }
        result.conversation = conversation->second;
    }
    const auto profile = selected.find("x-luce-odistill-profile");
    if (profile != selected.end()) {
        if (profile->second.empty() || profile->second.size() > 128) return result;
        result.profile = profile->second;
    }
    result.granted = true;
    return result;
}

nlohmann::json build_luce_odistill_native_event(
        const std::string & served_model,
        const nlohmann::json & request,
        const nlohmann::json & response,
        const LuceODistillConsent & consent,
        const LuceODistillSelection & selection,
        double latency_seconds) {
    return {
        {"schema_version", 1},
        {"kind", "luce_odistill_native_capture"},
        {"served_model", served_model},
        {"request", request},
        {"response", response},
        {"consent", {
            {"granted", consent.granted},
            {"subject", consent.subject},
            {"conversation", consent.conversation.empty()
                ? nlohmann::json(nullptr) : nlohmann::json(consent.conversation)},
            {"profile", consent.profile},
            {"source", "x-luce-odistill-consent"},
        }},
        {"runtime_drafter", luce_odistill_runtime_drafter_json(selection)},
        {"observation", {{"latency_seconds", latency_seconds}}},
    };
}

LuceODistillTraceSink::LuceODistillTraceSink(std::string socket_path)
    : socket_path_(std::move(socket_path)) {
    if (!socket_path_.empty()) {
        worker_ = std::thread(&LuceODistillTraceSink::worker_loop, this);
    }
}

LuceODistillTraceSink::~LuceODistillTraceSink() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        queue_.clear();
        queued_bytes_ = 0;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

bool LuceODistillTraceSink::enqueue(const nlohmann::json & event) {
    if (socket_path_.empty()) return false;
    std::string payload;
    try {
        payload = event.dump();
    } catch (const std::exception &) {
        return false;
    }
    if (payload.empty() || payload.size() > kMaxEventBytes) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || queue_.size() >= kMaxQueuedEvents ||
            queued_bytes_ + payload.size() > kMaxQueuedBytes) {
            return false;
        }
        queued_bytes_ += payload.size();
        queue_.push_back(std::move(payload));
    }
    cv_.notify_one();
    return true;
}

void LuceODistillTraceSink::worker_loop() {
    while (true) {
        std::string payload;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (stopping_) return;
            payload = std::move(queue_.front());
            queue_.pop_front();
            queued_bytes_ -= payload.size();
        }
        (void)send_payload(payload);
    }
}

bool LuceODistillTraceSink::send_payload(const std::string & payload) const {
#if defined(_WIN32)
    (void)payload;
    return false;
#else
    sockaddr_un address{};
    if (socket_path_.size() >= sizeof(address.sun_path) ||
        payload.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    timeval timeout{};
    timeout.tv_sec = 1;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path_.c_str(), socket_path_.size() + 1);
    if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        close(fd);
        return false;
    }
    const uint32_t size = (uint32_t)payload.size();
    const unsigned char prefix[4] = {
        (unsigned char)((size >> 24) & 0xff),
        (unsigned char)((size >> 16) & 0xff),
        (unsigned char)((size >> 8) & 0xff),
        (unsigned char)(size & 0xff),
    };
    auto send_all = [&](const void * raw, size_t length) {
        const char * data = static_cast<const char *>(raw);
        size_t sent = 0;
        while (sent < length) {
            const ssize_t count = send(fd, data + sent, length - sent, MSG_NOSIGNAL);
            if (count <= 0) return false;
            sent += (size_t)count;
        }
        return true;
    };
    const bool ok = send_all(prefix, sizeof(prefix)) &&
                    send_all(payload.data(), payload.size());
    close(fd);
    return ok;
#endif
}

}  // namespace dflash::common
