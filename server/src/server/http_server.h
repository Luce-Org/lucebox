// HTTP server infrastructure for dflash::common native server.
//
// Ported from ds4_server.c's socket/threading/HTTP layer, converted to C++.
// Architecture:
//   - Main thread: listen + accept
//   - Per-client thread: parse HTTP request, enqueue job, wait for completion
//   - Single worker thread: dequeue jobs, call ModelBackend::generate()
//
// Client disconnect detection: the client thread watches the socket while the
// worker generates, and streaming writes provide a second failure signal.
// Heartbeat comments keep long prefill phases alive through HTTP clients with
// body-idle timeouts.

#pragma once

#include "socket_handle.h"
#include "client_send_buffer.h"
#include "common/model_backend.h"
#include "tokenizer.h"
#include "chat_template.h"
#include "tool_memory.h"
#include "prefix_cache.h"
#include "disk_prefix_cache.h"
#include "freeze_history.h"
#include "api_types.h"
#include "placement/draft_residency.h"
#include "placement/remote_draft_config.h"
#include "common/pflash_drafter_ipc.h"
#include "model_card.h"
#include "adaptive_keep_ratio.h"
#include "server_status.h"
#include "sse_emitter.h"
#include <nlohmann/json.hpp>

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <unordered_map>
#include <vector>

namespace dflash::common {

using json = nlohmann::json;

// ─── Forward declarations ───────────────────────────────────────────────
struct ServerJob;

namespace http_detail {
// Non-consuming peer-state probe used by the client-thread job monitor.
// A read half-close is not a disconnect: HTTP clients may finish writing a
// request and continue reading its response. Public for model-free tests.
enum class PeerSocketState {
    Connected,
    ReadClosed,
    Disconnected,
};
PeerSocketState inspect_peer_socket(SocketHandle fd);

// Incrementally inspect complete SSE lines for the terminal data event.
// `partial_line` carries an unterminated line across transport chunks.
bool sse_chunk_has_done(std::string & partial_line,
                        const char * data, size_t size);

// Advance one heartbeat on an already-nonblocking job socket without waiting
// for writability. `offset` preserves a partial write across monitor ticks.
// Public for the model-free socket regression test.
enum class HeartbeatSendResult {
    Complete,
    Retry,
    Disconnected,
};
HeartbeatSendResult try_send_sse_heartbeat(SocketHandle fd, size_t & offset);
}

// ─── Server configuration ───────────────────────────────────────────────
struct ServerConfig {
    std::string host        = "0.0.0.0";
    int         port        = 8080;
    int         max_tokens  = 4096;     // default max output tokens (legacy alias for default_max_tokens)
    int         max_ctx     = 0;        // 0 = use backend's DevicePlacement default (8192)
    bool        enable_cors = true;
    std::string model_name  = "dflash";
    int         prefix_cache_cap = 32;  // prefix cache slots (0 disables)
    int         prefill_cache_cap = 0;  // full-prompt/prefill cache slots (0 disables)

    // Pin-Friendly Prompt Processor (PPP): LCP pin_end + optional rearrange.
    // See docs/PIN_FRIENDLY_PROMPT.md. Env: DFLASH_PPP=0|1,
    // DFLASH_PPP_REARRANGE=0|1, DFLASH_PPP_LCP_WINDOW=N,
    // DFLASH_PPP_MIN_PIN_TOKENS=N, DFLASH_PPP_MAX_EPHEMERAL=N.
    bool        ppp_enabled = true;
    bool        ppp_rearrange = false;
    int         ppp_lcp_window = 8;
    int         ppp_min_pin_tokens = 512;
    int         ppp_max_ephemeral_tokens = 256;  // diff hunk relocate cap

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
    // Idle-to-busy batching window. It is ignored by single-slot engines and
    // never delays an already decoding request.
    int admission_coalesce_ms = 20;

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
    // Passthrough proxy — forward to upstream OpenAI-compatible server
    std::string pflash_upstream_base;      // e.g. "http://localhost:8080/v1"
    std::string pflash_upstream_key;       // Bearer token for upstream
    std::string pflash_upstream_model;     // model name in forwarded requests
    // Piecewise keep-ratio curve: (token_threshold, keep_ratio) sorted ascending.
    // If empty, uses pflash_keep_ratio as flat value.
    std::vector<std::pair<int, float>> pflash_curve;
    bool        lazy_draft      = false;   // legacy alias for request-scoped draft residency
    DraftResidencyPolicy draft_residency = DraftResidencyPolicy::Auto;

    // Disk prefix cache
    std::string disk_cache_dir;             // empty = disabled
    size_t      disk_cache_budget_mb = 4096; // max disk usage in MB
    int         disk_cache_min_tokens = 512; // only persist >= this many tokens
    int         disk_cache_continued_interval = 10240; // continued checkpoint every N tokens
    int         disk_cache_cold_max_tokens = 10240;    // cold prefix for prompts longer than this
    DiskPrefixCachePolicy disk_cache_policy;

    // Optional Jinja chat template (overrides the hardcoded ChatFormat::QWEN3
    // / LAGUNA renderer when non-empty). Used for tool-using agents that need
    // the Anthropic tool_use envelope, e.g. froggeric Qwen3.6 template.
    std::string chat_template_src;          // literal Jinja source (loaded from file)
    std::string chat_template_path;         // path it was loaded from (logged at startup)

    // Expert frequency tracking (--freq): print frequency analysis at shutdown.
    bool        freq_tracking = false;

    // Routing data collection (--collect-routing <path>): write binary per-token
    // routing data (hidden states + expert selections) for predictor training.
    std::string collect_routing_path;
};

namespace http_detail {

inline constexpr int kFlowKvInertMinTokens = 512;

// Small policy helpers kept outside HttpServer so model-free unit tests use
// the same decisions as the request path.
int flowkv_activation_threshold(const ServerConfig & config);
bool flowkv_should_activate(const ServerConfig & config,
                            int aged_token_estimate);
float resolve_pflash_keep_ratio(float configured_ratio,
                                const std::string & session_id,
                                const HttpServerSessions & sessions);
bool should_clamp_flowkv_disk_cache(
    bool flowkv, const DiskPrefixCachePolicy & policy);

}  // namespace http_detail

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
    // Original request body (for upstream proxy forwarding)
    json                      raw_body;
    // Response ID
    std::string               response_id;
    // Thinking/reasoning state
    bool                      thinking_enabled = true;
    bool                      started_in_thinking = false;
    // True when the request opted in to the thinking-budget envelope via
    // `thinking: {type: "enabled"}`. Distinct from thinking_enabled (which
    // can be set via the chat template kwarg alone). When true, the response
    // includes a `finish_details` block when thinking was opted in.
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
    // Bandit: per-session adaptive keep_ratio opt-in
    std::string               session_id;
    DiskPrefixCachePolicy     disk_cache_policy;
    // PPP: stable pin cut for tool-heavy requests (0 = use default boundary).
    int                       pin_end_token = 0;
};

// Parse request sampler fields, applying model-card defaults where present.
SamplerCfg parse_request_sampler(const json & body,
                                 const SamplingDefaults & defaults);

// Read the required `messages` field. Throws std::invalid_argument when
// it is missing or not a non-empty array; route_request's catch turns
// that into a 400.
json require_messages_array(const json & body);

// Resolve the supported output-token aliases in precedence order. Only the
// selected field is parsed, so malformed lower-priority aliases are ignored.
int resolve_max_output_tokens(const json & body, int default_max_tokens);

// Sticky tools-boundary pinning is part of PPP and must follow its master
// toggle. Kept as a small policy helper so the disabled path is testable.
bool ppp_prefers_tools_boundary(bool ppp_enabled, bool has_tools);

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

    // Async-signal-safe: only sets the stopping flag. The accept loop polls
    // this flag on a short timeout, so it wakes regardless of which thread the
    // signal is delivered to. (Closing listen_fd_ here is unsafe: on Linux a
    // close() from another thread does not wake a blocked accept(), and it
    // races the accept loop's own close on exit.)
    void request_stop() {
        stopping_.store(true, std::memory_order_relaxed);
    }

private:
    // Client thread: read HTTP request, parse, enqueue job, wait.
    void handle_client(SocketHandle fd);

    // Worker thread: process jobs sequentially. process_job owns the
    // lifecycle of one dequeued request, including signaling completion.
    void worker_loop();
    void process_job(ServerJob * job);

    struct PreparedPrompt {
        std::vector<int32_t> tokens;
        bool compressed = false;
        bool flowkv = false;
        int full_cache_served_tokens = -1;
        int full_cache_hit_slot = -1;
        int full_cache_hit_len = 0;
        int error_status = 0;
        std::string error;
    };

    // Prompt preparation keeps the FlowKV and whole-prompt PFlash policies
    // out of the decode path while preserving their shared precedence rules.
    PreparedPrompt prepare_prompt(const ParsedRequest & req);
    void apply_flowkv_compression(const ParsedRequest & req,
                                  PreparedPrompt & prepared);
    std::string apply_pflash_compression(const ParsedRequest & req,
                                         PreparedPrompt & prepared);
    bool forward_upstream(ServerJob * job, const ParsedRequest & req,
                          const PreparedPrompt & prepared);

    struct GenerationCacheState {
        DiskPrefixCachePolicy disk_policy;
        int cache_slot = -1;
        int prefix_len = 0;
        bool using_restore = false;
        bool disk_hit = false;
        int full_snap_slot = -1;
        int full_snap_pos = 0;
        bool full_snap_prepared = false;
        // When DiffPin rewrote tokens, full-cache keys must use
        // prepared.tokens (effective), not req.prompt_tokens.
        bool full_snap_key_effective = false;
        int snap_slot = -1;
        int snap_cut = 0;
        bool snap_prepared = false;
    };

    GenerationCacheState prepare_generation_cache(
        const ParsedRequest & req, PreparedPrompt & prepared,
        GenerateRequest & generate_request);
    void finalize_generation_cache(
        const ParsedRequest & req, const PreparedPrompt & prepared,
        const GenerationCacheState & cache, const GenerateResult & result,
        int completion_tokens, bool visible_output_seen,
        bool client_disconnected);

    struct GenerationInputs {
        GenerateRequest request;
        int generation_cap = 0;
        std::vector<int32_t> hint_tokens;
        std::vector<int32_t> stall_tool_prefix_tokens;
        std::vector<int32_t> stall_action_suffix_tokens;
        std::vector<int32_t> stall_skip_tokens;
    };

    struct GenerationOutputState {
        int completion_tokens = 0;
        bool visible_output_seen = false;
        bool client_disconnected = false;
    };

    void prepare_generation_inputs(
        const ParsedRequest & req, const PreparedPrompt & prepared,
        GenerationInputs & inputs);
    void configure_generation_io(
        ServerJob * job, const ParsedRequest & req, SseEmitter & emitter,
        GenerationOutputState & output, DaemonIO & io);

    // Worker thread, concurrent mode (the backend exposes a SeqEngine):
    // iteration-level scheduler. Admission is claim-only; this baseline
    // drains its pending prefill between decode iterations, then advances
    // active slots together in one batched step.
    void scheduler_loop(SeqEngine & engine);

    // Non-blocking dequeue used for admission polling between decode steps.
    ServerJob * try_dequeue();
    // Bounded wait used only during an idle-to-busy admission window.
    ServerJob * dequeue_for(
        std::chrono::steady_clock::duration timeout);

    // Concurrent-scheduler token delivery and shared response construction.
    // A send buffer keeps slow clients off the shared decode loop.
    bool deliver_generation_token(
        ServerJob * job, const ParsedRequest & req, SseEmitter & emitter,
        int32_t token, int & completion_tokens,
        ClientSendBuffer & send_buffer);
    void send_nonstream_response(
        const ParsedRequest & req, SocketHandle fd, SseEmitter & emitter,
        const std::vector<int32_t> & gen_tokens, int n_gen_cap,
        bool budget_forced_close, bool degenerate_decode_close,
        const GenTimings & gen_timings,
        ClientSendBuffer * send_buffer = nullptr);
    std::string format_http_response(
        int status, const std::string & content_type,
        const std::string & body);
    static std::array<std::string, 2> sse_error_close_chunks(
        const std::string & message);

    // Parse HTTP request from socket.
    struct HttpRequest {
        std::string method;
        std::string path;
        std::string query;  // raw query string (after '?')
        std::string body;
    };
    bool read_http_request(SocketHandle fd, HttpRequest & out);

    // Route request to appropriate parser.
    bool route_request(SocketHandle fd, const HttpRequest & hr);
    // parse_common_request_fields, render_and_tokenize_request, and
    // validate_request_context return false after sending an error
    // response. parse_endpoint_request returns false only when the path
    // is unsupported (the caller sends the 404).
    bool parse_common_request_fields(SocketHandle fd, const json & body,
                                     ParsedRequest & req);
    bool parse_endpoint_request(const std::string & path, const json & body,
                                ParsedRequest & req, bool & count_tokens_only);
    void apply_request_reasoning(const json & body, ParsedRequest & req);
    bool render_and_tokenize_request(
        SocketHandle fd, const std::vector<ChatMessage> & chat_messages,
        ParsedRequest & req);
    bool validate_request_context(SocketHandle fd, const ParsedRequest & req);
    void log_parsed_request(const ParsedRequest & req) const;
    void enqueue_request_and_wait(SocketHandle fd, ParsedRequest req);

    // Send HTTP response helpers.
    bool send_response(SocketHandle fd, int status, const std::string & content_type,
                       const std::string & body);
    bool send_error(SocketHandle fd, int status, const std::string & message);
    bool send_sse_headers(ServerJob * job);

    // Send raw bytes with stall detection.
    bool send_all(SocketHandle fd, const void * data, size_t len);
    bool send_job_bytes(ServerJob * job, const void * data, size_t len);
    void start_job_stream(ServerJob * job);
    void stop_job_stream(ServerJob * job,
                         ClientSendBuffer * pending_output = nullptr);
    void maybe_send_job_heartbeat(ServerJob * job, bool peer_read_closed);

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

    // Per-session adaptive keep_ratio bandit state.
    HttpServerSessions sessions_;

    // Live status tracker (read by /status/json, written by worker thread).
    ServerStatus status_;

    // SSE client connections for /status/events push.
    std::mutex             sse_mu_;
    std::vector<SocketHandle> sse_fds_;

    // Broadcast current status to all SSE clients. Removes dead fds.
    void broadcast_status();

    // Broadcast incremental token text to SSE clients.
    void broadcast_token(const std::string & text);

    // Send SSE heartbeat comment to prune disconnected clients.
    void sse_heartbeat();

    // Resolve and cache path to share/status.html.
    std::string status_html_path_;
    std::string resolve_status_html();

    // Track prompt tokens for each snapshot slot (for shutdown save).
    std::unordered_map<int, std::vector<int32_t>> slot_tokens_;
    std::vector<std::vector<int32_t>> recent_disk_prompts_;
    // Recent tool-bearing prompt prefixes for PPP LCP annotate.
    std::vector<std::vector<int32_t>> recent_tool_prefixes_;

    // FlowKV freeze-history: per-message compression cache.
    // Key: SHA-1 hash of the drafter-token slice and selected keep ratio.
    // Value: compressed content text (output of drafter_tokenizer_->decode).
    // Bounded to kFrozenCacheMax entries; cleared on overflow (simple eviction).
    static constexpr size_t kFrozenCacheMax = 256;
    struct PrefixHashEqual {
        bool operator()(const PrefixHash & a, const PrefixHash & b) const { return a == b; }
    };
    struct PrefixHashHasher {
        size_t operator()(const PrefixHash & h) const {
            size_t v = 0;
            for (size_t i = 0; i < h.size(); ++i)
                v ^= (size_t)h[i] << ((i % sizeof(size_t)) * 8);
            return v;
        }
    };
    std::unordered_map<PrefixHash, std::string,
                       PrefixHashHasher, PrefixHashEqual> frozen_content_cache_;

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
    SocketHandle listen_fd_ = kInvalidSocket;
};

// ─── Job (stack-owned by client thread) ─────────────────────────────────
struct ServerJob {
    SocketHandle  fd = kInvalidSocket;
    ParsedRequest req;
    bool          done = false;
    std::mutex    mu;
    std::condition_variable cv;
    // Streaming output is written by both the worker (SSE data) and the
    // client-thread monitor (heartbeat comments). Serialize complete frames
    // so their bytes can never interleave.
    std::mutex    write_mu;
    bool          stream_ready = false;
    size_t        heartbeat_offset = 0;
    std::chrono::steady_clock::time_point last_stream_write{};
    std::atomic<bool> client_disconnected{false};
    ServerJob *   next = nullptr;

    // Concurrent-scheduler state that survives a pool-full admission retry.
    // The classic worker leaves these fields untouched.
    bool          announced = false;
    bool          sse_started = false;
    // First concurrent-scheduler attempt; retained across busy deferrals so
    // server-side prefill/elapsed telemetry does not erase queueing delay.
    std::chrono::steady_clock::time_point parallel_started_at{};
    std::unique_ptr<SseEmitter> emitter;
};

// ─── Parse session_id from a chat-completion JSON body ──────────────────
// Returns empty string when session_id is absent or not a string (int/null/array).
// Checks extra_body.session_id first, then top-level session_id.
inline std::string parse_session_id_from_body(const json & body) {
    if (body.contains("extra_body")) {
        const auto & eb = body["extra_body"];
        if (eb.is_object() && eb.contains("session_id") && eb["session_id"].is_string()) {
            return eb["session_id"].get<std::string>();
        }
    }
    if (body.contains("session_id") && body["session_id"].is_string()) {
        return body["session_id"].get<std::string>();
    }
    return {};
}

}  // namespace dflash::common
