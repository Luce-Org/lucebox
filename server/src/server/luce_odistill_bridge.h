// Experimental local bridge between the Lucebox data plane and the
// luce_odistill controller. Training and promotion policy stay outside the
// engine; this module only verifies one startup selection and emits consented
// trace events over a local Unix socket.

#pragma once

#include <nlohmann/json.hpp>

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dflash::common {

struct LuceODistillSelection {
    bool        configured = false;
    std::string selection_state;
    std::string release_id;
    std::string source_artifact_sha256;
    std::string runtime_artifact_path;
    std::string runtime_artifact_sha256;
    std::string target_profile_sha256;
    std::string drafter_profile_sha256;
    std::string evaluation_report_sha256;
};

// Pure schema parser used by startup and model-free tests. It does not touch
// the runtime artifact; load_luce_odistill_selection additionally opens the
// GGUF and verifies its exact SHA-256 before returning.
bool parse_luce_odistill_selection(const nlohmann::json & value,
                                   LuceODistillSelection & out,
                                   std::string & error);
bool load_luce_odistill_selection(const std::string & manifest_path,
                                  LuceODistillSelection & out,
                                  std::string & error);

nlohmann::json luce_odistill_runtime_drafter_json(
    const LuceODistillSelection & selection);

struct LuceODistillConsent {
    bool        requested = false;
    bool        granted = false;
    std::string subject;
    std::string conversation;
    std::string profile = "default";
};

// Header names are case-insensitive. Duplicate internal headers, an invalid
// truth value, an absent subject, or overlong identity values fail closed for
// capture while leaving the served request untouched.
LuceODistillConsent parse_luce_odistill_capture_headers(
    const std::vector<std::pair<std::string, std::string>> & headers);

nlohmann::json build_luce_odistill_native_event(
    const std::string & served_model,
    const nlohmann::json & request,
    const nlohmann::json & response,
    const LuceODistillConsent & consent,
    const LuceODistillSelection & selection,
    double latency_seconds);

class LuceODistillTraceSink {
public:
    explicit LuceODistillTraceSink(std::string socket_path);
    ~LuceODistillTraceSink();

    LuceODistillTraceSink(const LuceODistillTraceSink &) = delete;
    LuceODistillTraceSink & operator=(const LuceODistillTraceSink &) = delete;

    // Best effort and bounded: a missing/stalled collector drops the event and
    // never delays the inference worker or its client response.
    bool enqueue(const nlohmann::json & event);

private:
    void worker_loop();
    bool send_payload(const std::string & payload) const;

    std::string socket_path_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    size_t queued_bytes_ = 0;
    bool stopping_ = false;
    std::thread worker_;
};

}  // namespace dflash::common
