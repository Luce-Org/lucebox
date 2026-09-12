// Real HTTP + existing batch schedulers, with deterministic host-only engines.
// Protects cross-model admission, tokenization, retirement and worker ownership.
#include "CppUnitTestFramework.hpp"
#include "server/http_server.h"
#include "common/concurrency/seq_engine.h"
#include "gguf.h"

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using namespace dflash::common;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
struct ModelRoutingFixture {};
#define ROUTING_CHECK(expression) do { \
    if (!(expression)) throw std::runtime_error( \
        std::string("routing check at line ") + std::to_string(__LINE__) + ": " #expression); \
} while (false)

class Socket {
public:
    explicit Socket(int fd) : fd_(fd) { ROUTING_CHECK(fd >= 0); }
    ~Socket() { close(); }
    Socket(Socket && other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    Socket(const Socket &) = delete;
    int get() const { return fd_; }
    void close() { if (fd_ >= 0) ::close(fd_); fd_ = -1; }
    void send(const std::string & data) {
        size_t offset = 0;
        while (offset < data.size()) {
            const auto n = ::send(fd_, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
            ROUTING_CHECK(n > 0);
            offset += (size_t)n;
        }
    }
    std::string read() {
        std::string response;
        const auto deadline = Clock::now() + 5s;
        for (;;) {
            ROUTING_CHECK(Clock::now() < deadline);
            pollfd pfd{fd_, POLLIN, 0};
            const int ready = poll(&pfd, 1, 100);
            ROUTING_CHECK(ready >= 0);
            if (!ready) continue;
            char bytes[4096];
            const auto n = recv(fd_, bytes, sizeof(bytes), 0);
            ROUTING_CHECK(n >= 0);
            if (n == 0) return response;
            response.append(bytes, (size_t)n);
        }
    }
private:
    int fd_;
};

json response_body(const std::string & response, int status = 200) {
    ROUTING_CHECK(response.rfind("HTTP/1.1 " + std::to_string(status) + " ", 0) == 0);
    const auto boundary = response.find("\r\n\r\n");
    ROUTING_CHECK(boundary != std::string::npos);
    return json::parse(response.substr(boundary + 4));
}

class HeldEngine final : public SeqEngine {
public:
    int slot_count() const override { return 2; }
    int max_context() const override { return 64; }
    bool token_is_eos(int32_t token) const override { return token == 2; }
    StepPlanLimits step_plan_limits(int) const override { return {2, 64, 128}; }
    AdmitResult admit(uint64_t, const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override {
        std::lock_guard<std::mutex> lock(mu);
        ++attempts;
        cv.notify_all();
        if (prompt.size() > prompt_capacity) return {AdmitResult::Status::capacity_exceeded, -1, "prompt exceeds pool"};
        if (capacity_busy || suspended[0] || suspended[1])
            return {AdmitResult::Status::busy, -1, "KV headroom occupied"};
        if (fail_next) {
            fail_next = false;
            return {AdmitResult::Status::failed, -1, "injected admission failure"};
        }
        for (int i = 0; i < 2; ++i) {
            if (active[i]) continue;
            active[i] = true;
            prompts.push_back(prompt);
            temperatures.push_back(sampler.temp);
            ++admissions;
            cv.notify_all();
            return {AdmitResult::Status::admitted, i, {}};
        }
        return {AdmitResult::Status::busy, -1, {}};
    }
    StepResult step(const StepPlan & plan) override {
        std::unique_lock<std::mutex> lock(mu);
        if (block_step) {
            inside_block = true;
            cv.notify_all();
            cv.wait(lock, [&] { return !block_step; });
        }
        StepResult result;
        for (const auto & slice : plan.prefills) {
            result.prefills.push_back({slice.slot,
                release ? PrefillOutput::Status::completed : PrefillOutput::Status::advanced,
                release ? 0 : -1, {}});
        }
        for (const auto & input : plan.decode) {
            ROUTING_CHECK(!suspended[(size_t)input.slot]);
            auto & steps = decode_steps[(size_t)input.slot];
            const int token = non_eos[(size_t)input.slot] ? 1 :
                decode_pressure && !suspensions && steps < 2 ? 1 : 2;
            ++steps;
            result.decode.push_back({input.slot, token, false, {}});
        }
        // An unfinished fake prefill yields control to the real scheduler,
        // allowing admission and cancellation without timing-based completion.
        lock.unlock();
        std::this_thread::yield();
        return result;
    }
    void retire(int slot) override {
        std::lock_guard<std::mutex> lock(mu);
        active.at((size_t)slot) = false;
        suspended.at((size_t)slot) = false;
        recompute.at((size_t)slot) = false;
        ++retirements;
        cv.notify_all();
    }
    bool reserve_decode(const StepPlan & plan) override {
        std::lock_guard<std::mutex> lock(mu);
        return !decode_pressure || plan.decode.size() < 2 ||
               decode_steps[0] == 0 || decode_steps[1] == 0;
    }
    size_t kv_offload_capacity() const override { return 64; }
    KvOffloadState kv_offload_state(int slot) const override {
        std::lock_guard<std::mutex> lock(mu);
        const bool parked = suspended.at((size_t)slot);
        const bool recomputing = recompute.at((size_t)slot);
        return {parked, recomputing, parked && !recomputing ? 64u : 0u};
    }
    bool offload_kv(int slot, size_t available, std::string & error) override {
        std::lock_guard<std::mutex> lock(mu);
        if (available < 64) { error = "RAM budget exhausted"; return false; }
        ROUTING_CHECK(active.at((size_t)slot) && !suspended.at((size_t)slot));
        suspended[(size_t)slot] = true;
        ++suspensions;
        block_step = hold_on_suspend;
        cv.notify_all();
        return true;
    }
    bool restore_kv(int slot, std::string & error) override {
        std::lock_guard<std::mutex> lock(mu);
        // Only a checkpoint copy can fail; a recompute resume is pure
        // host-side reservation and always succeeds.
        if (fail_restore && !recompute.at((size_t)slot)) {
            error = "injected restore failure";
            return false;
        }
        ROUTING_CHECK(suspended.at((size_t)slot));
        for (size_t other = 0; other < active.size(); ++other) {
            resumed_with_resident |= (int)other != slot &&
                active[other] && !suspended[other];
        }
        suspended[(size_t)slot] = false;
        recompute[(size_t)slot] = false;
        ++resumptions;
        cv.notify_all();
        return true;
    }
    bool evict_kv(int slot, int32_t, std::string & error) override {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_evict) { error = "injected evict failure"; return false; }
        ROUTING_CHECK(active.at((size_t)slot));
        suspended[(size_t)slot] = true;
        recompute[(size_t)slot] = true;
        ++evictions;
        cv.notify_all();
        return true;
    }
    bool kv_restore_feasible(int slot) const override {
        std::lock_guard<std::mutex> lock(mu);
        return feasible_restore && suspended.at((size_t)slot);
    }
    void start_decode_pressure(bool hold = true) {
        std::lock_guard<std::mutex> lock(mu);
        decode_pressure = true;
        hold_on_suspend = hold;
        release = true;
    }
    void wait_suspension() {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return suspensions == 1 && inside_block; }));
    }
    void wait_suspensions(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return suspensions >= count; }));
    }
    void wait_admissions(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return admissions >= count; }));
    }
    void wait_retirements(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return retirements >= count; }));
    }
    void wait_resumptions(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return resumptions >= count; }));
    }
    void wait_evictions(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return evictions >= count; }));
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mu);
        release = true;
        block_step = false;
        cv.notify_all();
    }
    mutable std::mutex mu;
    std::condition_variable cv;
    std::array<bool, 2> active{}, suspended{}, recompute{}, non_eos{};
    std::array<int, 2> decode_steps{};
    bool decode_pressure = false, hold_on_suspend = false, fail_restore = false;
    bool fail_evict = false, feasible_restore = false;
    bool resumed_with_resident = false;
    int suspensions = 0, resumptions = 0, evictions = 0;
    size_t prompt_capacity = 64;
    bool capacity_busy = false;
    bool release = false, fail_next = false, block_step = false, inside_block = false;
    int attempts = 0;
    int admissions = 0, retirements = 0;
    std::vector<std::vector<int32_t>> prompts;
    std::vector<float> temperatures;
};

struct RoutedBackend : ModelBackend {
    HeldEngine engine;
    SeqEngine * seq_engine() override { return &engine; }
    void print_ready_banner() const override {}
    bool park(ParkTarget) override { return true; }
    bool unpark(ParkTarget) override { return true; }
    bool is_target_parked() const override { return false; }
    GenerateResult generate_impl(const GenerateRequest &, const DaemonIO &) override { return {}; }
    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int snapshot_cur_pos(int) const override { return 0; }
    GenerateResult restore_and_generate_impl(int, const GenerateRequest &, const DaemonIO &) override { return {}; }
    bool handle_compress(const std::string &, const DaemonIO &) override { return false; }
    void free_drafter() override {}
    void shutdown() override {}
};

// Exercises the existing ModelBackend::generate path without a SeqEngine.
// The gate models work boundaries where real backends poll DaemonIO cancellation.
struct HeldSingleBackend final : RoutedBackend {
    SeqEngine * seq_engine() override { return nullptr; }
    GenerateResult generate_impl(const GenerateRequest & req, const DaemonIO & io) override {
        std::unique_lock<std::mutex> lock(mu);
        ++calls;
        prompts.push_back(req.prompt);
        temperatures.push_back(req.sampler.temp);
        cv.notify_all();
        const auto deadline = Clock::now() + 5s;
        while (!released && !io.is_cancelled() && Clock::now() < deadline) {
            cv.wait_for(lock, 10ms);
        }
        GenerateResult result;
        if (io.is_cancelled()) {
            ++cancellations;
            cv.notify_all();
        } else if (!released) {
            return result; // Bounded failure if the server never cancels or releases us.
        } else {
            result.tokens = {0, 2};
            for (int32_t token : result.tokens) io.emit(token);
        }
        result.succeed();
        return result;
    }
    void wait_calls(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return calls >= count; }));
    }
    void wait_cancellations(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return cancellations >= count; }));
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mu);
        released = true;
        cv.notify_all();
    }
    std::mutex mu;
    std::condition_variable cv;
    bool released = false;
    int calls = 0, cancellations = 0;
    std::vector<std::vector<int32_t>> prompts;
    std::vector<float> temperatures;
};

void load_tokenizer(Tokenizer & tokenizer, bool second) {
    gguf_context * ctx = gguf_init_empty();
    const char * first[] = {"q", "x", "<eos>", "y", "s"};
    const char * other[] = {"s", "y", "<eos>", "x", "q"};
    const uint32_t types[] = {1, 1, 3, 1, 1};
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", second ? other : first, 5);
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_UINT32, types, 5);
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 2);
    const auto path = std::filesystem::temp_directory_path() /
        ("luce-routing-" + std::to_string(getpid()) + (second ? "-b.gguf" : "-a.gguf"));
    gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    const bool loaded = tokenizer.load_from_gguf(path.c_str());
    std::filesystem::remove(path);
    ROUTING_CHECK(loaded);
}

class RunningModels {
public:
    RoutedBackend first, second;
    HeldSingleBackend single;
    HeldSingleBackend first_single;
    Tokenizer first_tok, second_tok;
    std::unique_ptr<HttpServer> listener, peer;
    std::thread runner;
    int port = 0;
    std::atomic<int> result{-1};

    explicit RunningModels(bool single_peer = false, bool single_listener = false,
                           bool load_balancing = true, int queue_limit = 0,
                           bool reverse_priority = false, size_t offload_bytes = 0) {
        load_tokenizer(first_tok, false);
        load_tokenizer(second_tok, true);
        // Obtain a loopback test port from the OS, then hand it to HttpServer.
        Socket reservation(socket(AF_INET, SOCK_STREAM, 0));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        ROUTING_CHECK(bind(reservation.get(), (sockaddr *)&address, sizeof(address)) == 0);
        socklen_t length = sizeof(address);
        ROUTING_CHECK(getsockname(reservation.get(), (sockaddr *)&address, &length) == 0);
        port = ntohs(address.sin_port);
        ServerConfig config;
        config.host = "127.0.0.1";
        config.port = port;
        config.model_name = "qwen";
        config.max_ctx = 64;
        config.routing_queue_limit = queue_limit;
        config.decode_kv_offload_bytes = offload_bytes;
        config.default_max_tokens = 4;
        config.prefix_cache_cap = 0;
        config.ppp_enabled = false;
        config.admission_coalesce_ms = 0;
        config.chat_template_src = "{{ messages[0]['content'] }}";
        config.sampler_defaults.has_temperature = true;
        config.sampler_defaults.temperature = 0.2f;
        ModelBackend & first_backend = single_listener ? static_cast<ModelBackend &>(first_single) : first;
        listener = std::make_unique<HttpServer>(first_backend, first_tok, config);
        config.model_name = "ds4";
        config.chat_template_src = "y{{ messages[0]['content'] }}";
        config.sampler_defaults.temperature = 0.7f;
        ModelBackend & peer_backend = single_peer ? static_cast<ModelBackend &>(single) : second;
        peer = std::make_unique<HttpServer>(peer_backend, second_tok, config);
        if (reverse_priority) std::swap(listener, peer);
        reservation.close();
        runner = std::thread([this, load_balancing] {
            result = load_balancing ? listener->run({listener.get(), peer.get()})
                                    : listener->run();
        });
        try {
            // Wait on actual listener readiness, not a fixed startup sleep.
            const auto deadline = Clock::now() + 5s;
            for (;;) {
                ROUTING_CHECK(result.load() == -1);
                Socket probe(socket(AF_INET, SOCK_STREAM, 0));
                address.sin_port = htons(port);
                if (connect(probe.get(), (sockaddr *)&address, sizeof(address)) == 0) break;
                ROUTING_CHECK(Clock::now() < deadline);
                std::this_thread::yield();
            }
        } catch (...) { stop(); throw; }
    }
    ~RunningModels() { stop(); }
    void stop() {
        listener->request_stop();
        first.engine.finish();
        second.engine.finish();
        single.finish();
        first_single.finish();
        if (runner.joinable()) runner.join();
    }
    Socket connect_client() {
        Socket client(socket(AF_INET, SOCK_STREAM, 0));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        ROUTING_CHECK(connect(client.get(), (sockaddr *)&address, sizeof(address)) == 0);
        return client;
    }
    Socket post(json body, const std::string & path = "/v1/chat/completions") {
        auto client = connect_client();
        const auto text = body.dump();
        client.send("POST " + path + " HTTP/1.1\r\nHost: localhost\r\nContent-Length: " +
                    std::to_string(text.size()) + "\r\n\r\n" + text);
        return client;
    }
    json get(const std::string & path) {
        auto client = connect_client();
        client.send("GET " + path + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
        return response_body(client.read());
    }
    void wait_load(int first_count, int second_count) {
        const auto deadline = Clock::now() + 5s;
        for (;;) {
            const auto status = get("/status/json")["models"];
            if (status[0]["in_flight"] == first_count && status[1]["in_flight"] == second_count) return;
            ROUTING_CHECK(Clock::now() < deadline);
            std::this_thread::yield();
        }
    }
};

json chat(const char * model = "qwen", bool stream = false) {
    return {{"model", model}, {"stream", stream}, {"max_tokens", 4},
            {"messages", {{{"role", "user"}, {"content", "x"}}}}};
}
} // namespace

TEST_CASE(ModelRoutingFixture, test_two_plus_two_preserves_batching_and_independent_progress) {
    RunningModels models;
    std::vector<Socket> clients;
    for (int i = 0; i < 4; ++i) {
        clients.push_back(models.post(chat(i % 2 ? "ds4" : "qwen")));
        (i < 2 ? models.first : models.second).engine.wait_admissions(i % 2 + 1);
    }
    models.wait_load(2, 2);
    response_body(models.post(chat("ds4")).read(), 503);
    models.first.engine.finish();
    for (int i : {0, 1}) {
        const auto body = response_body(clients[i].read());
        ROUTING_CHECK(body["model"] == "qwen");
        ROUTING_CHECK(body["choices"][0]["message"]["content"] == "q");
    }
    models.wait_load(0, 2); // DS4 cannot stall Qwen's worker or response path.
    models.second.engine.finish();
    for (int i : {2, 3}) {
        const auto body = response_body(clients[i].read());
        ROUTING_CHECK(body["model"] == "ds4");
        ROUTING_CHECK(body["choices"][0]["message"]["content"] == "s");
    }
    models.wait_load(0, 0);
    ROUTING_CHECK(models.first.engine.prompts[0] == std::vector<int32_t>({1}));
    ROUTING_CHECK(models.second.engine.prompts[0] == std::vector<int32_t>({1, 3}));
    ROUTING_CHECK(models.first.engine.temperatures[0] == 0.2f);
    ROUTING_CHECK(models.second.engine.temperatures[0] == 0.7f);
    const auto list = models.get("/v1/models")["data"];
    ROUTING_CHECK(list.size() == 2);
    ROUTING_CHECK(list[0]["id"] == "qwen" && list[1]["id"] == "ds4");
    const auto codex = models.get("/v1/models?client_version=test")["models"];
    ROUTING_CHECK(codex.size() == 2 && codex[1]["slug"] == "ds4");
    const auto props = models.get("/props");
    ROUTING_CHECK(props["server"]["props_schema"] == 2);
    ROUTING_CHECK(props["models"][1]["props"]["model_alias"] == "ds4");
}

TEST_CASE(ModelRoutingFixture, test_rejected_requests_and_engine_failure_release_capacity) {
    RunningModels models;
    response_body(models.post(chat("missing"), "/v1/messages/count_tokens").read(), 404);
    auto invalid = chat(); invalid.erase("messages");
    response_body(models.post(invalid).read(), 400);
    invalid = chat(); invalid["model"] = 42;
    response_body(models.post(invalid).read(), 400);
    invalid = chat(); invalid["messages"][0]["content"] = std::string(100, 'x');
    response_body(models.post(invalid).read(), 400);
    response_body(models.post(chat("auto"), "/v1/messages/count_tokens").read(), 400);
    ROUTING_CHECK(response_body(models.post(chat("ds4"), "/v1/messages/count_tokens").read())["input_tokens"] == 2);
    ROUTING_CHECK(response_body(models.post(chat("qwen"), "/v1/messages/count_tokens").read())["input_tokens"] == 1);
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.fail_next = true;
    }
    response_body(models.post(chat("qwen")).read(), 500);
    models.wait_load(0, 0);
    models.first.engine.finish();
    ROUTING_CHECK(response_body(models.post(chat("qwen")).read())["model"] == "qwen");
}

TEST_CASE(ModelRoutingFixture, test_disconnect_holds_capacity_until_engine_retirement) {
    RunningModels models;
    auto a = models.post(chat("qwen"));
    auto b = models.post(chat("qwen"));
    models.first.engine.wait_admissions(2);
    {
        std::unique_lock<std::mutex> lock(models.first.engine.mu);
        models.first.engine.block_step = true;
        ROUTING_CHECK(models.first.engine.cv.wait_for(lock, 5s, [&] { return models.first.engine.inside_block; }));
    }
    // A TCP reset is an unambiguous disconnect. FIN alone also represents a
    // legitimate HTTP client that half-closes its write side and keeps reading.
    linger reset{1, 0};
    ROUTING_CHECK(setsockopt(a.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)) == 0);
    a.close(); // A running step still owns this request's state.
    models.second.engine.finish();
    ROUTING_CHECK(response_body(models.post(chat("qwen")).read())["model"] == "ds4");
    models.wait_load(2, 0); // Fallback did not reclaim a primary slot mid-step.
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.block_step = false;
        models.first.engine.cv.notify_all();
    }
    models.first.engine.wait_retirements(1);
    models.wait_load(1, 0);
    auto replacement = models.post(chat("qwen"));
    models.first.engine.wait_admissions(3);
    models.first.engine.finish();
    ROUTING_CHECK(response_body(b.read())["model"] == "qwen");
    ROUTING_CHECK(response_body(replacement.read())["model"] == "qwen");
    models.wait_load(0, 0);
}

TEST_CASE(ModelRoutingFixture, test_response_protocols_use_selected_model_and_terminal_event) {
    RunningModels models;
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.capacity_busy = true;
    }
    models.second.engine.finish();
    const auto stream = models.post(chat("ds4", true)).read();
    ROUTING_CHECK(stream.rfind("HTTP/1.1 200 ", 0) == 0);
    ROUTING_CHECK(stream.find("\"model\":\"ds4\"") != std::string::npos);
    ROUTING_CHECK(stream.find("\"content\":\"s\"") != std::string::npos);
    ROUTING_CHECK(stream.find("data: [DONE]") != std::string::npos);
    const auto message = response_body(models.post(chat("ds4"), "/v1/messages").read());
    ROUTING_CHECK(message["model"] == "ds4");
    ROUTING_CHECK(message["content"][0]["text"] == "s");
    json request = {{"model", "ds4"}, {"input", {{{"role", "user"}, {"content", "x"}}}},
                    {"max_output_tokens", 4}};
    const auto response = response_body(models.post(request, "/v1/responses").read());
    ROUTING_CHECK(response["model"] == "ds4");
    ROUTING_CHECK(response["output"][0]["content"][0]["text"] == "s");
    models.wait_load(0, 0);
}

TEST_CASE(ModelRoutingFixture, test_shutdown_drains_both_models_and_incomplete_upload) {
    RunningModels models;
    auto a = models.post(chat("qwen"));
    models.first.engine.wait_admissions(1);
    auto extra = models.post(chat("auto"));
    models.first.engine.wait_admissions(2);
    auto b = models.post(chat("ds4"));
    auto incomplete = models.connect_client();
    incomplete.send("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\n{");
    models.first.engine.wait_admissions(1);
    models.second.engine.wait_admissions(1);
    models.listener->request_stop();
    models.runner.join();
    ROUTING_CHECK(models.result == 0);
    ROUTING_CHECK(models.first.engine.retirements == 2);
    ROUTING_CHECK(models.second.engine.retirements == 1);
}

TEST_CASE(ModelRoutingFixture, test_hybrid_admission_uses_single_capacity_and_independent_workers) {
    RunningModels models(true);
    const auto props = models.get("/props")["models"];
    ROUTING_CHECK(props[0]["capacity"] == 2 && props[0]["execution_mode"] == "batched");
    ROUTING_CHECK(props[1]["capacity"] == 1 && props[1]["execution_mode"] == "single-request");
    auto first = models.post(chat());
    models.first.engine.wait_admissions(1);
    auto third = models.post(chat());
    models.first.engine.wait_admissions(2);
    auto second = models.post(chat("ds4"));
    models.single.wait_calls(1);
    models.wait_load(2, 1);
    response_body(models.post(chat("ds4")).read(), 503);
    models.first.engine.finish();
    ROUTING_CHECK(response_body(first.read())["model"] == "qwen");
    ROUTING_CHECK(response_body(third.read())["model"] == "qwen");
    models.wait_load(0, 1);
    models.single.finish();
    const auto body = response_body(second.read());
    ROUTING_CHECK(body["model"] == "ds4");
    ROUTING_CHECK(body["choices"][0]["message"]["content"] == "s");
    models.wait_load(0, 0);
    ROUTING_CHECK(models.single.prompts[0] == std::vector<int32_t>({1, 3}));
    ROUTING_CHECK(models.single.temperatures[0] == 0.7f);
}

TEST_CASE(ModelRoutingFixture, test_hybrid_disconnect_cancels_single_worker_and_reuses_capacity) {
    RunningModels models(true);
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.capacity_busy = true;
    }
    auto client = models.post(chat("ds4", true));
    models.single.wait_calls(1);
    linger reset{1, 0};
    ROUTING_CHECK(setsockopt(client.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)) == 0);
    client.close();
    models.single.wait_cancellations(1);
    models.wait_load(0, 0);
    models.single.finish();
    const auto stream = models.post(chat("ds4", true)).read();
    ROUTING_CHECK(stream.find("\"model\":\"ds4\"") != std::string::npos);
    ROUTING_CHECK(stream.find("\"content\":\"s\"") != std::string::npos);
    ROUTING_CHECK(stream.find("data: [DONE]") != std::string::npos);
}

TEST_CASE(ModelRoutingFixture, test_hybrid_shutdown_drains_single_worker_before_destroying_contexts) {
    RunningModels models(true);
    auto first = models.post(chat("qwen"));
    models.first.engine.wait_admissions(1);
    auto extra = models.post(chat("auto"));
    models.first.engine.wait_admissions(2);
    auto second = models.post(chat("ds4", true));
    models.single.wait_calls(1);
    models.listener->request_stop();
    models.single.finish();
    models.runner.join();
    ROUTING_CHECK(models.result == 0);
    ROUTING_CHECK(models.single.cancellations == 0);
    const auto stream = second.read();
    ROUTING_CHECK(stream.find("data: [DONE]") != std::string::npos);
    ROUTING_CHECK(stream.find("\"content\":\"s\"") != std::string::npos);
    ROUTING_CHECK(models.first.engine.retirements == 2);
}

// The prior fixtures cover batched/batched and batched/native. These protect
// the pilot's native/native ownership, including the listener's own worker.
TEST_CASE(ModelRoutingFixture, test_two_native_workers_preserve_priority_and_capacity) {
    RunningModels models(true, true);
    const auto props = models.get("/props");
    ROUTING_CHECK(props["routing"] == "primary-first");
    for (const auto & model : props["models"]) {
        ROUTING_CHECK(model["capacity"] == 1);
        ROUTING_CHECK(model["execution_mode"] == "single-request");
    }
    auto first = models.post(chat("qwen"));
    models.first_single.wait_calls(1);
    auto second = models.post(chat("ds4"));
    models.single.wait_calls(1);
    models.wait_load(1, 1);
    response_body(models.post(chat("ds4")).read(), 503);
    models.first_single.finish();
    const auto qwen = response_body(first.read());
    ROUTING_CHECK(qwen["model"] == "qwen");
    ROUTING_CHECK(qwen["choices"][0]["message"]["content"] == "q");
    models.wait_load(0, 1);
    models.single.finish();
    const auto ds4 = response_body(second.read());
    ROUTING_CHECK(ds4["model"] == "ds4");
    ROUTING_CHECK(ds4["choices"][0]["message"]["content"] == "s");
    models.wait_load(0, 0);
    ROUTING_CHECK(models.first_single.prompts[0] == std::vector<int32_t>({1}));
    ROUTING_CHECK(models.single.prompts[0] == std::vector<int32_t>({1, 3}));
    ROUTING_CHECK(response_body(models.post(chat("auto")).read())["model"] == "qwen");
}

TEST_CASE(ModelRoutingFixture, test_two_native_workers_finish_before_shutdown_returns) {
    RunningModels models(true, true);
    auto first = models.post(chat("qwen", true));
    models.first_single.wait_calls(1);
    auto second = models.post(chat("ds4"));
    models.single.wait_calls(1);
    models.listener->request_stop();
    models.first_single.finish();
    models.single.finish();
    models.runner.join();
    ROUTING_CHECK(models.result == 0);
    ROUTING_CHECK(models.first_single.cancellations == 0);
    ROUTING_CHECK(models.single.cancellations == 0);
    const auto stream = first.read();
    ROUTING_CHECK(stream.find("data: [DONE]") != std::string::npos);
    ROUTING_CHECK(stream.find("\"content\":\"q\"") != std::string::npos);
    ROUTING_CHECK(response_body(second.read())["choices"][0]["message"]["content"] == "s");
}

TEST_CASE(ModelRoutingFixture, test_default_native_worker_finishes_active_response_on_shutdown) {
    for (bool stream : {false, true}) {
        RunningModels models(false, true, false);
        auto client = models.post(chat("alias", stream));
        models.first_single.wait_calls(1);
        models.listener->request_stop();
        models.first_single.finish();
        models.runner.join();
        ROUTING_CHECK(models.result == 0);
        ROUTING_CHECK(models.first_single.cancellations == 0);
        const auto response = client.read();
        if (stream) {
            ROUTING_CHECK(response.find("data: [DONE]") != std::string::npos);
            ROUTING_CHECK(response.find("\"content\":\"q\"") != std::string::npos);
        } else {
            ROUTING_CHECK(response_body(response)["choices"][0]["message"]["content"] == "q");
        }
    }
}

// Exercise primary preference, engine KV pressure below slot capacity, and
// listener-owned waiting/cancellation independently of request model names.
TEST_CASE(ModelRoutingFixture, test_model_names_and_auto_fill_primary_then_secondary_and_drain_waiter) {
    RunningModels models(false, false, true, 1);
    auto omitted = chat(); omitted.erase("model");
    auto a = models.post(omitted);
    models.first.engine.wait_admissions(1);
    auto b = models.post(chat("ds4"));
    models.first.engine.wait_admissions(2);
    auto c = models.post(chat("auto"));
    models.second.engine.wait_admissions(1);
    auto d = models.post(chat("qwen"));
    models.second.engine.wait_admissions(2);
    auto waiting = models.post(chat("unrecognized-client-alias"));
    const auto deadline = Clock::now() + 5s;
    while (models.get("/status/json")["waiting"] != 1) {
        ROUTING_CHECK(Clock::now() < deadline);
        std::this_thread::yield();
    }
    response_body(models.post(chat("qwen")).read(), 503);
    models.first.engine.finish();
    ROUTING_CHECK(response_body(a.read())["model"] == "qwen");
    ROUTING_CHECK(response_body(b.read())["model"] == "qwen");
    ROUTING_CHECK(response_body(waiting.read())["model"] == "qwen");
    models.wait_load(0, 2);
    models.second.engine.finish();
    ROUTING_CHECK(response_body(c.read())["model"] == "ds4");
    ROUTING_CHECK(response_body(d.read())["model"] == "ds4");
}

TEST_CASE(ModelRoutingFixture, test_kv_busy_falls_back_before_stream_headers_and_retokenizes) {
    RunningModels models;
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.capacity_busy = true;
    }
    models.second.engine.finish();
    ROUTING_CHECK(response_body(models.post(chat("qwen")).read())["model"] == "ds4");
    const auto stream = models.post(chat("auto", true)).read();
    ROUTING_CHECK(stream.find("HTTP/1.1 200 ") == 0);
    ROUTING_CHECK(stream.find("HTTP/1.1", 1) == std::string::npos);
    ROUTING_CHECK(stream.find("\"model\":\"qwen\"") == std::string::npos);
    ROUTING_CHECK(stream.find("\"model\":\"ds4\"") != std::string::npos);
    ROUTING_CHECK(stream.find("data: [DONE]") != std::string::npos);
    models.wait_load(0, 0);
    ROUTING_CHECK(models.first.engine.admissions == 0);
    ROUTING_CHECK(models.second.engine.prompts[0] == std::vector<int32_t>({1, 3}));
    ROUTING_CHECK(models.second.engine.temperatures[0] == 0.7f);
}

TEST_CASE(ModelRoutingFixture, test_permanent_capacity_falls_back_or_rejects_without_waiting) {
    RunningModels models;
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.prompt_capacity = 0;
    }
    models.second.engine.finish();
    ROUTING_CHECK(response_body(models.post(chat("auto")).read())["model"] == "ds4");
    models.wait_load(0, 0);
    {
        std::lock_guard<std::mutex> lock(models.second.engine.mu);
        models.second.engine.prompt_capacity = 0;
    }
    response_body(models.post(chat("auto")).read(), 400);
    ROUTING_CHECK(models.get("/status/json")["waiting"] == 0);
    auto oversized = chat("auto");
    oversized["messages"][0]["content"] = std::string(100, 'x');
    response_body(models.post(oversized).read(), 400);
    ROUTING_CHECK(models.get("/status/json")["waiting"] == 0);
}

TEST_CASE(ModelRoutingFixture, test_busy_engines_retry_without_retirement_notification) {
    RunningModels models(false, false, true, 1);
    {
        std::scoped_lock lock(models.first.engine.mu, models.second.engine.mu);
        models.first.engine.capacity_busy = true;
        models.second.engine.capacity_busy = true;
    }
    auto waiting = models.post(chat("auto"));
    const auto deadline = Clock::now() + 5s;
    while (models.get("/status/json")["waiting"] != 1) {
        ROUTING_CHECK(Clock::now() < deadline);
        std::this_thread::yield();
    }
    {
        std::lock_guard<std::mutex> lock(models.second.engine.mu);
        models.second.engine.capacity_busy = false;
    }
    models.second.engine.wait_admissions(1);
    ROUTING_CHECK(models.get("/status/json")["waiting"] == 0);
    models.second.engine.finish();
    ROUTING_CHECK(response_body(waiting.read())["model"] == "ds4");
}

TEST_CASE(ModelRoutingFixture, test_waiting_disconnect_and_shutdown_release_requests) {
    RunningModels models(true, true, true, 1);
    auto a = models.post(chat("auto"));
    models.first_single.wait_calls(1);
    auto b = models.post(chat("auto"));
    models.single.wait_calls(1);
    auto waiting = models.post(chat("auto"));
    auto wait_count = [&](int count) {
        const auto deadline = Clock::now() + 5s;
        while (models.get("/status/json")["waiting"] != count) {
            ROUTING_CHECK(Clock::now() < deadline);
            std::this_thread::yield();
        }
    };
    wait_count(1);
    linger reset{1, 0};
    ROUTING_CHECK(setsockopt(waiting.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)) == 0);
    waiting.close();
    wait_count(0);
    auto replacement = models.post(chat("auto"));
    wait_count(1);
    models.listener->request_stop();
    models.first_single.finish();
    models.single.finish();
    models.runner.join();
    ROUTING_CHECK(models.result == 0);
    response_body(replacement.read(), 503);
}

// Busy probes must not wake each other into an unbounded admission loop.
// Existing retry coverage has only one waiter, so it cannot expose that loop.
TEST_CASE(ModelRoutingFixture, test_busy_waiters_do_not_trigger_each_others_retries) {
    RunningModels models(false, false, true, 2);
    {
        std::scoped_lock lock(models.first.engine.mu, models.second.engine.mu);
        models.first.engine.capacity_busy = true;
        models.second.engine.capacity_busy = true;
    }
    auto a = models.post(chat());
    auto b = models.post(chat());
    {
        std::unique_lock<std::mutex> lock(models.first.engine.mu);
        // Over one 250 ms retry interval, two clients should make only a
        // handful of attempts. Allow ample scheduling/spurious-wakeup slack.
        ROUTING_CHECK(!models.first.engine.cv.wait_for(lock, 250ms, [&] {
            return models.first.engine.attempts >= 32;
        }));
    }
    models.listener->request_stop();
    models.runner.join();
    response_body(a.read(), 503);
    response_body(b.read(), 503);
}

// The routed test covers this error only with admission feedback enabled.
TEST_CASE(ModelRoutingFixture, test_single_model_permanent_capacity_is_client_error) {
    RunningModels models(false, false, false);
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.prompt_capacity = 0;
    }
    for (bool stream : {false, true}) {
        response_body(models.post(chat("qwen", stream)).read(), 400);
    }
    ROUTING_CHECK(models.first.engine.admissions == 0);
}

// Disabled balancing must not start a peer or let a client alias change identity.
TEST_CASE(ModelRoutingFixture, test_single_model_run_does_not_enable_balancing) {
    RunningModels models(false, false, false);
    ROUTING_CHECK(!models.get("/props").contains("routing"));
    ROUTING_CHECK(models.get("/v1/models")["data"].size() == 1);
    models.first.engine.finish();
    auto request = chat(); request.erase("model");
    ROUTING_CHECK(response_body(models.post(request).read())["model"] == "qwen");
    ROUTING_CHECK(response_body(models.post(chat("ds4")).read())["model"] == "qwen");
    ROUTING_CHECK(models.second.engine.admissions == 0);
}

// Protect priority reversal and removal of secondary-name pinning. The old
// suite only selected the first registered model when both had capacity.
TEST_CASE(ModelRoutingFixture, test_reversed_priority_ignores_generation_model_names) {
    RunningModels models(false, false, true, 0, true);
    models.first.engine.finish();
    models.second.engine.finish();
    for (const char * name : {"qwen", "ds4", "auto", "", "client-alias"}) {
        ROUTING_CHECK(response_body(models.post(chat(name)).read())["model"] == "ds4");
    }
    ROUTING_CHECK(models.first.engine.admissions == 0);
    ROUTING_CHECK(response_body(models.post(chat("qwen"), "/v1/messages/count_tokens").read())["input_tokens"] == 1);
    {
        std::lock_guard<std::mutex> lock(models.second.engine.mu);
        models.second.engine.capacity_busy = true;
    }
    ROUTING_CHECK(response_body(models.post(chat("ds4")).read())["model"] == "qwen");
}

// These tests exercise live HTTP ownership during decode growth. Earlier
// routing tests stop at admission; they cannot expose lost suspended streams.
TEST_CASE(ModelRoutingFixture, test_decode_pressure_preserves_stream_and_routes_new_work) {
    // Exercise startup auto resolution through the actual HTTP scheduler,
    // then verify the resulting capacity is sufficient to preserve a stream.
    if (dflash::common::available_kv_offload_memory().value_or(0) < 512) {
        throw CppUnitTestFramework::TestSkippedException("automatic RAM sizing unavailable");
    }
    RunningModels models(false, false, true, 1, false, dflash::common::kAutoKvOffloadBytes);
    auto a = models.post(chat("qwen", true));
    models.first.engine.wait_admissions(1);
    auto b = models.post(chat("qwen", true));
    models.first.engine.wait_admissions(2);
    models.first.engine.start_decode_pressure();
    models.first.engine.wait_suspension();
    const auto status = models.get("/status/json")["models"][0];
    ROUTING_CHECK(status["in_flight"] == 2);
    ROUTING_CHECK(status["status"]["suspended_requests"] == 1);
    ROUTING_CHECK(status["status"]["offloaded_kv_bytes"] == 64);
    ROUTING_CHECK(models.get("/props")["models"][0]["props"]["runtime"]["continuous_batching"]["decode_kv_offload_bytes"] == 64);
    models.second.engine.finish();
    ROUTING_CHECK(response_body(models.post(chat("qwen")).read())["model"] == "ds4");
    models.first.engine.finish();
    for (Socket * client : {&a, &b}) {
        const auto response = client->read();
        ROUTING_CHECK(response.find("HTTP/1.1 200 ") == 0);
        ROUTING_CHECK(response.find("HTTP/1.1", 1) == std::string::npos);
        ROUTING_CHECK(response.find("data: [DONE]") != std::string::npos);
        std::string content;
        size_t start = 0;
        while ((start = response.find("data: {", start)) != std::string::npos) {
            start += 6;
            const auto end = response.find('\n', start);
            const auto event = json::parse(response.substr(start, end - start));
            if (event.contains("choices") && !event["choices"].empty()) {
                content += event["choices"][0]["delta"].value("content", std::string());
            }
            start = end;
        }
        ROUTING_CHECK(content == "qx"); // pending token neither lost nor replayed
    }
    models.wait_load(0, 0);
    ROUTING_CHECK(models.first.engine.admissions == 2);
    ROUTING_CHECK(models.first.engine.suspensions == 1 && models.first.engine.resumptions == 1);
    ROUTING_CHECK(models.get("/status/json")["models"][0]["status"]["offloaded_kv_bytes"] == 0);
}

TEST_CASE(ModelRoutingFixture, test_suspended_disconnect_and_shutdown_release_all_state) {
    for (bool shutdown : {false, true}) {
        RunningModels models(false, false, true, 1, false, 64);
        auto a = models.post(chat());
        models.first.engine.wait_admissions(1);
        auto b = models.post(chat());
        models.first.engine.wait_admissions(2);
        models.first.engine.start_decode_pressure();
        models.first.engine.wait_suspension();
        if (shutdown) {
            models.listener->request_stop();
        } else {
            linger reset{1, 0};
            ROUTING_CHECK(setsockopt(b.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)) == 0);
            b.close();
        }
        models.first.engine.finish();
        if (shutdown) {
            models.runner.join();
            ROUTING_CHECK(models.result == 0);
        } else {
            ROUTING_CHECK(response_body(a.read())["model"] == "qwen");
            models.wait_load(0, 0);
        }
        ROUTING_CHECK(models.first.engine.retirements == 2);
        ROUTING_CHECK(!models.first.engine.kv_offload_state(0).suspended);
        ROUTING_CHECK(!models.first.engine.kv_offload_state(1).suspended);
    }
}

// When a checkpoint cannot be taken (RAM budget) or cannot be restored
// (copy failure), the request must still survive: it parks without a
// payload and re-prefills its retained history once capacity returns.
TEST_CASE(ModelRoutingFixture, test_evict_for_recompute_preserves_request) {
    for (bool restore_error : {false, true}) {
        RunningModels models(false, false, true, 1, false, restore_error ? 64 : 1);
        auto a = models.post(chat());
        models.first.engine.wait_admissions(1);
        auto b = models.post(chat());
        models.first.engine.wait_admissions(2);
        if (restore_error) {
            std::lock_guard<std::mutex> lock(models.first.engine.mu);
            models.first.engine.fail_restore = true;
        }
        models.first.engine.start_decode_pressure(false);
        ROUTING_CHECK(response_body(a.read())["model"] == "qwen");
        ROUTING_CHECK(response_body(b.read())["model"] == "qwen");
        models.wait_load(0, 0);
        ROUTING_CHECK(models.first.engine.evictions == 1);
        ROUTING_CHECK(models.first.engine.retirements == 2);
        ROUTING_CHECK(models.first.engine.kv_offload_state(1).bytes == 0);
    }
}

// A suspended request need not wait for a full drain when the engine can
// prove its resume reservation fits the free pool with growth headroom.
TEST_CASE(ModelRoutingFixture, test_suspension_resumes_before_drain_when_feasible) {
    RunningModels models(false, false, true, 1, false, 64);
    auto a = models.post(chat("qwen", true));
    models.first.engine.wait_admissions(1);
    auto b = models.post(chat("qwen", true));
    models.first.engine.wait_admissions(2);
    {
        // A cannot finish between the suspension and the next pass's resume
        // check: its decode emits non-EOS tokens, and feasibility is armed
        // before pressure so the restore cannot wait for a drain.
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.non_eos[0] = true;
        models.first.engine.feasible_restore = true;
    }
    models.first.engine.start_decode_pressure(false);
    models.first.engine.wait_resumptions(1);
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        ROUTING_CHECK(models.first.engine.resumed_with_resident);
        models.first.engine.non_eos[0] = false;
        models.first.engine.decode_pressure = false;
    }
    models.first.engine.finish();
    for (Socket * client : {&a, &b}) {
        const auto response = client->read();
        ROUTING_CHECK(response.find("HTTP/1.1 200 ") == 0);
        ROUTING_CHECK(response.find("data: [DONE]") != std::string::npos);
    }
    models.wait_load(0, 0);
    ROUTING_CHECK(models.first.engine.suspensions >= 1);
    ROUTING_CHECK(models.first.engine.retirements == 2);
}

// Termination is now reserved for requests whose context cannot fit the
// pool even alone — exercised here by refusing eviction as well.
TEST_CASE(ModelRoutingFixture, test_offload_budget_and_restore_errors_fail_one_request_cleanly) {
    for (bool restore_error : {false, true}) {
        RunningModels models(false, false, true, 1, false, restore_error ? 64 : 1);
        auto a = models.post(chat());
        models.first.engine.wait_admissions(1);
        auto b = models.post(chat());
        models.first.engine.wait_admissions(2);
        {
            std::lock_guard<std::mutex> lock(models.first.engine.mu);
            models.first.engine.fail_evict = true;
            models.first.engine.fail_restore = restore_error;
        }
        models.first.engine.start_decode_pressure(false);
        ROUTING_CHECK(response_body(a.read())["model"] == "qwen");
        const auto error = response_body(b.read(), 500)["error"]["message"].get<std::string>();
        ROUTING_CHECK(error.find("evict failure") != std::string::npos);
        models.wait_load(0, 0);
        ROUTING_CHECK(models.first.engine.retirements == 2);
        ROUTING_CHECK(models.first.engine.kv_offload_state(1).bytes == 0);
    }
}

#undef ROUTING_CHECK
#endif
