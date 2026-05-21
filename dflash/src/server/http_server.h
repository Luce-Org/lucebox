// HTTP server infrastructure for dflash27b native server.
//
// Ported from ds4_server.c's socket/threading/HTTP layer, converted to C++.
// Architecture:
//   - Main thread: listen + accept
//   - Per-client thread: parse HTTP request, enqueue job, wait for completion
//   - Single worker thread: dequeue jobs, call ModelBackend::generate()
//
// Client disconnect detection: the worker writes SSE chunks via send().
// If send() fails (EPIPE/ECONNRESET), generation aborts immediately.

#pragma once

#include "common/model_backend.h"
#include "tokenizer.h"
#include "chat_template.h"
#include "tool_memory.h"
#include "prefix_cache.h"
#include "disk_prefix_cache.h"
#include "api_types.h"
#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace dflash27b {

using json = nlohmann::json;

// ─── Forward declarations ───────────────────────────────────────────────
struct ServerJob;

// ─── Server configuration ───────────────────────────────────────────────
struct ServerConfig {
    std::string host        = "0.0.0.0";
    int         port        = 8080;
    int         max_tokens  = 4096;     // default max output tokens
    int         max_ctx     = 0;        // 0 = use backend's DevicePlacement default (8192)
    bool        enable_cors = true;
    std::string model_name  = "dflash";
    int         prefix_cache_cap = 32;  // prefix cache slots (0 disables)

    // PFlash (speculative prefill compression)
    enum class PflashMode { OFF, AUTO, ALWAYS };
    PflashMode  pflash_mode      = PflashMode::OFF;
    int         pflash_threshold = 32000;   // token count threshold for AUTO mode
    float       pflash_keep_ratio = 0.05f;  // fraction of tokens to keep
    std::string pflash_drafter_path;        // path to drafter GGUF (Qwen3-0.6B)
    bool        pflash_skip_park = false;   // skip park/unpark for ≥32GB GPUs

    // Disk prefix cache
    std::string disk_cache_dir;             // empty = disabled
    size_t      disk_cache_budget_mb = 4096; // max disk usage in MB
    int         disk_cache_min_tokens = 512; // only persist >= this many tokens
    int         disk_cache_continued_interval = 10240; // continued checkpoint every N tokens
    int         disk_cache_cold_max_tokens = 10240;    // cold prefix for prompts longer than this
};

// ─── Parsed request ─────────────────────────────────────────────────────

struct ParsedRequest {
    ApiFormat                  format;
    std::vector<int32_t>      prompt_tokens;  // tokenized prompt
    int                       max_output   = 4096;
    bool                      stream       = true;
    SamplerCfg                sampler;
    std::string               model;
    // Tool definitions (stored as JSON for response formatting)
    json                      tools;
    // Original messages (for response formatting)
    json                      messages;
    // Response ID
    std::string               response_id;
    // Thinking/reasoning state
    bool                      thinking_enabled = true;
    bool                      started_in_thinking = false;
};

// ─── HTTP server ────────────────────────────────────────────────────────
class HttpServer {
public:
    HttpServer(ModelBackend & backend,
               Tokenizer & tokenizer,
               const ServerConfig & config);
    ~HttpServer();

    HttpServer(const HttpServer &) = delete;
    HttpServer & operator=(const HttpServer &) = delete;

    // Set the optional pflash drafter tokenizer.
    void set_drafter_tokenizer(Tokenizer * tok) { drafter_tokenizer_ = tok; }

    // Start listening. Blocks until shutdown() is called.
    int run();

    // Signal the server to stop accepting new connections and drain.
    void shutdown();

    // Async-signal-safe: only sets stopping flag and closes listen socket.
    void request_stop() {
        stopping_.store(true, std::memory_order_relaxed);
        if (listen_fd_ >= 0) {
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
    }

private:
    // Client thread: read HTTP request, parse, enqueue job, wait.
    void handle_client(int fd);

    // Worker thread: process jobs sequentially.
    void worker_loop();

    // Parse HTTP request from socket.
    struct HttpRequest {
        std::string method;
        std::string path;
        std::string query;  // raw query string (after '?')
        std::string body;
    };
    bool read_http_request(int fd, HttpRequest & out);

    // Route request to appropriate parser.
    bool route_request(int fd, const HttpRequest & hr);

    // Send HTTP response helpers.
    bool send_response(int fd, int status, const std::string & content_type,
                       const std::string & body);
    bool send_error(int fd, int status, const std::string & message);
    bool send_sse_headers(int fd);

    // Send raw bytes with stall detection.
    bool send_all(int fd, const void * data, size_t len);

    // Job queue.
    void enqueue(ServerJob * job);
    ServerJob * dequeue();

    // Members.
    ModelBackend &   backend_;
    Tokenizer &      tokenizer_;
    Tokenizer *      drafter_tokenizer_ = nullptr;  // pflash drafter (optional)
    ServerConfig     config_;
    ChatFormat       chat_format_;
    ToolMemory       tool_memory_;
    PrefixCache      prefix_cache_;
    DiskPrefixCache  disk_cache_;

    // Track prompt tokens for each snapshot slot (for shutdown save).
    std::unordered_map<int, std::vector<int32_t>> slot_tokens_;

    // Worker thread.
    std::thread                     worker_thread_;
    std::mutex                      queue_mu_;
    std::condition_variable         queue_cv_;
    ServerJob *                     queue_head_ = nullptr;
    ServerJob *                     queue_tail_ = nullptr;
    std::atomic<bool>               stopping_{false};

    // Active client thread tracking.
    std::atomic<int>                active_clients_{0};
    std::mutex                      clients_mu_;
    std::condition_variable         clients_cv_;

    // Listen socket.
    int listen_fd_ = -1;
};

// ─── Job (stack-owned by client thread) ─────────────────────────────────
struct ServerJob {
    int           fd = -1;
    ParsedRequest req;
    bool          done = false;
    std::mutex    mu;
    std::condition_variable cv;
    ServerJob *   next = nullptr;
};

}  // namespace dflash27b
