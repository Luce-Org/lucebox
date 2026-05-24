// HTTP server infrastructure for dflash::common native server.
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

namespace dflash::common {

using json = nlohmann::json;

// ─── Forward declarations ───────────────────────────────────────────────
struct ServerJob;

// ─── Server configuration ───────────────────────────────────────────────
struct ServerConfig {
    std::string host        = "0.0.0.0";
    int         port        = 8080;
    int         max_tokens  = 4096;     // default max output tokens (legacy alias for default_max_tokens)
    int         max_ctx     = 0;        // 0 = use backend's DevicePlacement default (8192)
    bool        enable_cors = true;
    std::string model_name  = "dflash";
    int         prefix_cache_cap = 32;  // prefix cache slots (0 disables)

    // Thinking-budget v2 (mirrors antirez/ds4 ds4_eval.c knob names).
    // Applied when a request opts in via `thinking: {type: "enabled"}`.
    // think_max_tokens caps the phase-1 reasoning generation; the combined
    // (reasoning + content) cap is the request's max_tokens, defaulting to
    // default_max_tokens when omitted. The phase-1/phase-2 reprompt logic
    // itself is a follow-up PR — this PR ships only the wire surface
    // (CLI flag + finish_details emission) so consumers can probe + plan.
    int         think_max_tokens    = 10000;
    int         default_max_tokens  = 16000;
    // Level 2 force-close (in-process, KV-continuous). When > 0 AND the
    // request opted into thinking, the backend's AR decode overrides
    // the next sampled token with `</think>` once (n_gen - committed)
    // <= hard_limit_reply_budget. Matches ds4_eval.c's hard_limit_reply
    // _budget default of 512. 0 disables the hook (falls back to Level
    // 1 phase-2 reprompt only).
    int         hard_limit_reply_budget = 512;

    // Token IDs resolved at server startup for the model's </think>
    // close-tag sequence. Single special token for Qwen3.6 (id 248069);
    // multiple tokens for DeepSeek/laguna ([1718, 37947, 32]). When
    // non-empty, used as BudgetHook.close_token_ids. server_main
    // populates this from the tokenizer after loading; HttpServer just
    // forwards into GenerateRequest.budget_hook when thinking is opted in.
    std::vector<int32_t> think_close_token_ids;

    // /props introspection inputs — captured at startup by server_main so
    // the /props handler doesn't need to crack open BackendArgs or env.
    // Matches dflash/scripts/server.py:1221-1312 field-for-field.
    std::string arch;                  // detected model arch (qwen35/36, laguna, gemma4, ...)
    std::string model_path;            // bargs.model_path
    std::string draft_path;            // bargs.draft_path (empty if no draft)
    std::string tokenizer_id;          // tokenizer name from GGUF metadata (best-effort)
    std::string kv_cache_k;            // effective KV K type ("q4_0", "tq3_0", "f16", ...)
    std::string kv_cache_v;            // effective KV V type
    std::string runtime_backend;       // "cuda" | "hip" | "cpu"
    int         fa_window           = 0;
    int         ddtree_budget       = 0;
    bool        speculative_enabled = false;
    bool        target_sharding     = false;

    // PFlash (speculative prefill compression)
    enum class PflashMode { OFF, AUTO, ALWAYS };
    PflashMode  pflash_mode      = PflashMode::OFF;
    int         pflash_threshold = 32000;   // token count threshold for AUTO mode
    float       pflash_keep_ratio = 0.05f;  // fraction of tokens to keep
    std::string pflash_drafter_path;        // path to drafter GGUF (Qwen3-0.6B)
    bool        pflash_skip_park = false;   // skip park/unpark for ≥32GB GPUs
    bool        lazy_draft      = false;   // park decode draft when idle to save VRAM

    // Disk prefix cache
    std::string disk_cache_dir;             // empty = disabled
    size_t      disk_cache_budget_mb = 4096; // max disk usage in MB
    int         disk_cache_min_tokens = 512; // only persist >= this many tokens
    int         disk_cache_continued_interval = 10240; // continued checkpoint every N tokens
    int         disk_cache_cold_max_tokens = 10240;    // cold prefix for prompts longer than this

    // Optional Jinja chat template (overrides the hardcoded ChatFormat::QWEN3
    // / LAGUNA renderer when non-empty). Used for tool-using agents that need
    // the Anthropic tool_use envelope, e.g. froggeric Qwen3.6 template.
    std::string chat_template_src;          // literal Jinja source (loaded from file)
    std::string chat_template_path;         // path it was loaded from (logged at startup)
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
    // Tool choice constraint (stored for hint generation)
    json                      tool_choice;
    // Original messages (for response formatting)
    json                      messages;
    // Response ID
    std::string               response_id;
    // Thinking/reasoning state
    bool                      thinking_enabled = true;
    bool                      started_in_thinking = false;
    // True when the request opted in to the thinking-budget envelope via
    // `thinking: {type: "enabled"}`. Distinct from thinking_enabled (which
    // can be set via the chat template kwarg alone). When true, the response
    // includes a `finish_details` block. Mirrors server.py:2271 conditional.
    bool                      thinking_opt_in = false;
    // Stop sequences (OpenAI "stop" + Anthropic "stop_sequences")
    std::vector<std::string>  stop_sequences;
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

    // Set the chat template format (detected from model arch).
    void set_chat_format(ChatFormat fmt) { chat_format_ = fmt; }

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

}  // namespace dflash::common
