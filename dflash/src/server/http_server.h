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
#include "placement/remote_draft_config.h"
#include "common/pflash_drafter_ipc.h"
#include "model_card.h"
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

    // Thinking-budget v2. Applied when a request opts in via
    // `thinking: {type: "enabled"}` or `reasoning: {effort: ...}`.
    // think_max_tokens caps phase-1 reasoning generation; the combined
    // (reasoning + content) cap is the request's max_tokens, defaulting
    // to default_max_tokens when omitted. The defaults below are the
    // hard fallback (antirez/ds4 ds4_eval.c reference values); at startup
    // server_main may raise them by loading share/model_cards/<name>.json
    // when a sidecar matches the loaded model. CLI flags override both.
    // See docs/specs/thinking-budget.md §3 for resolution order.
    int         think_max_tokens    = 15488;  // = default_max_tokens - hard_limit_reply_budget
    int         default_max_tokens  = 16000;
    // Level 2 force-close (in-process, KV-continuous). When > 0 AND the
    // request opted into thinking, the backend's AR decode overrides
    // the next sampled token with `</think>` once (n_gen - committed)
    // <= hard_limit_reply_budget. 0 disables the hook.
    //
    // Default 4096. The original 512 came from ds4_eval.c, which sized
    // for DeepSeek-V4-flash's terse style. For most models that's far
    // too small — Qwen3.6 restates work after `</think>` (needs ~4k);
    // Gemma 4 after the channel-thought force-close + transition cue
    // writes a clean coordinate-geometry proof for AIME (~2-4k tokens).
    // Without priors on a specific model, 4096 is the safer default
    // — bench results from gemma4-26b-thinking-control-2026-05-25
    // showed every force-closed thinking probe getting truncated
    // mid-answer at 512 reply tokens.
    int         hard_limit_reply_budget = 4096;

    // Token IDs resolved at server startup for the model's </think>
    // close-tag sequence. Single special token for Qwen3.6 (id 248069);
    // multiple tokens for DeepSeek/laguna ([1718, 37947, 32]). When
    // non-empty, used as BudgetHook.close_token_ids. server_main
    // populates this from the tokenizer after loading; HttpServer just
    // forwards into GenerateRequest.budget_hook when thinking is opted in.
    std::vector<int32_t> think_close_token_ids;

    // Phase-1 budgets per `reasoning.effort` tier (spec §4.2). Selected
    // by the request parser when `reasoning.effort` is present. Each
    // value is itself capped at `think_max_tokens` at startup.
    // Populated by server_main from the resolved model card; CLI flags
    // (--reasoning-effort-<tier>) override individual tiers.
    EffortTiers effort_tiers;

    // Sampler defaults from the model card (spec §3.3). Used to fill
    // values the request body did not specify. has_* fields distinguish
    // "card supplied a value" from "C++ default". HttpServer reads these
    // in the request parser; CLI does not currently override.
    SamplingDefaults sampler_defaults;

    // Operator-facing tag for the startup banner: e.g.
    // "share/model_cards/qwen3.6-27b.json", "family:qwen35", "hard-fallback".
    // Surfaced at /props.budget_envelope.model_card_source per
    // docs/specs/props-endpoint.md §4.2.
    std::string model_card_source_label;

    // Cached on startup by server_main after resolve_model_card. Null
    // (`.is_null()` returns true) when family or hard fallback was used.
    // Exposed verbatim under /props.model_card; validates against
    // share/model_cards/_schema.json. See docs/specs/props-endpoint.md
    // §4.9 and docs/specs/model-cards.md.
    nlohmann::json model_card_json = nullptr;

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
    // Prefill chunk size (bargs.chunk). Exposed at /props.runtime.chunk so
    // bench/snapshot tooling can capture the full server config — needed
    // because pre-c35a8a4 snapshots had no /props capture and post-hoc
    // forensics on which chunk was used are otherwise impossible. See
    // dflash/docs/specs/props-endpoint.md §4.5.
    int         chunk               = 0;
    // Resolved device placement strings (e.g. "auto:0", "cuda:0"). Sourced
    // from placement_device_name(bargs.device / bargs.draft_device) in
    // server_main after CLI parse.
    std::string target_device;
    std::string draft_device;

    // PFlash (speculative prefill compression)
    enum class PflashMode { OFF, AUTO, ALWAYS };
    PflashMode  pflash_mode      = PflashMode::OFF;
    int         pflash_threshold = 32000;   // token count threshold for AUTO mode
    float       pflash_keep_ratio = 0.05f;  // fraction of tokens to keep
    std::string pflash_drafter_path;        // path to drafter GGUF (Qwen3-0.6B)
    int         pflash_drafter_gpu = 0;     // backend-local GPU for PFlash drafter
    bool        pflash_remote_drafter = false; // use IPC drafter for mixed backends
    RemoteDraftConfig pflash_remote;        // IPC binary/work-dir for remote PFlash drafter
    bool        pflash_skip_park = false;   // skip park/unpark for >=32GB GPUs
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
    // True when the request opted in to the thinking-budget envelope via
    // `thinking: {type: "enabled"}`. Distinct from thinking_enabled (which
    // can be set via the chat template kwarg alone). When true, the response
    // includes a `finish_details` block. Mirrors server.py:2271 conditional.
    bool                      thinking_opt_in = false;
    // Per-request thinking-budget envelope (spec §4). Populated from
    // `thinking.budget_tokens` and `thinking.reply_budget`, or selected
    // from server-configured effort tiers when `reasoning.effort` is set.
    // -1 = not set; the server falls back to its global think_max_tokens /
    // hard_limit_reply_budget. Values are already clamped to those ceilings.
    int                       per_req_phase1_cap   = -1;
    int                       per_req_reply_budget = -1;
    // Stop sequences (OpenAI "stop" + Anthropic "stop_sequences")
    std::vector<std::string>  stop_sequences;
};

// Build the /props response body. Exposed (non-static) so unit tests
// can assert on its shape without spinning up a real socket. See
// docs/specs/props-endpoint.md for the wire contract.
json build_props_body(const ServerConfig & config,
                      const PrefixCache & prefix_cache,
                      const ToolMemory & tool_memory);

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
    PFlashDrafterIpcClient pflash_remote_;
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
