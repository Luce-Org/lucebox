// HTTP server infrastructure for luce::common native server.
//
// Ported from ds4_server.c's socket/threading/HTTP layer, converted to C++.
// Architecture:
//   - Main thread: listen + accept
//   - Per-client thread: parse HTTP request, enqueue job, wait for completion
//   - LuceEngine execution thread: run the selected backend serving loop
//
// Client disconnect detection: the client thread watches the socket while the
// worker generates, and streaming writes provide a second failure signal.
// Heartbeat comments keep long prefill phases alive through HTTP clients with
// body-idle timeouts.

#pragma once

#include "socket_handle.h"
#include "client_send_buffer.h"
#include "common/model_backend.h"
#include "common/concurrency/paged_kv_offload.h"
#include "common/output_constraint.h"
#include "tokenizer.h"
#include "chat_template.h"
#include "tool_memory.h"
#include "tool_checks.h"
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
#include "pflash_unified.h"
#include "pflash/pflash_selection.h"
#include <nlohmann/json.hpp>

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace luce::engine {
class LuceEngine;
}

namespace luce::common {

using json = nlohmann::json;

// ─── Forward declarations ───────────────────────────────────────────────
struct ServerJob;

// Admission feedback is returned before any response bytes or model compute.
enum class RoutingAdmission { handled, busy, unfit };

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
    int         routing_queue_limit = 32; // waiting auto requests across the listener
    int         max_ctx     = 0;        // 0 = use backend's DevicePlacement default (8192)
    bool        enable_cors = true;
    std::string model_name  = "luce";
    int         prefix_cache_cap = 32;  // prefix cache slots (0 disables)
    // Resident system-memory budget for single-sequence prefix snapshots.
    // kPrefixCacheBudgetAuto sizes it from the backend's snapshot size at
    // its full context; zero means unlimited.
    static constexpr size_t kPrefixCacheBudgetAuto = (size_t)-1;
    static constexpr size_t kConcurrentPrefixBudgetFloor = (size_t)4 * 1024 * 1024 * 1024;
    size_t      prefix_cache_max_bytes = kPrefixCacheBudgetAuto;
    // Resident system-memory budget for copied paged checkpoints. The
    // scheduler enforces it only when concurrent prefix storage is
    // active. kPrefixCacheBudgetAuto sizes it when the scheduler starts, from
    // the slot count and the batch engine's checkpoint size at --max-ctx
    // (at least kConcurrentPrefixBudgetFloor); zero means unlimited.
    size_t      concurrent_prefix_cache_max_bytes = kPrefixCacheBudgetAuto;
    bool        concurrent_prefix_cache = false;
    int         prefill_cache_cap = 0;  // full-prompt/prefill cache slots (0 disables)
    // Extend the existing prefix cache through generated tool-call turns.
    bool        agent_turn_cache = false;
    // Tool-use checks on OpenAI chat requests with tools (tool_checks.h).
    // A request turns them off or on with "tool_checks": false / true.
    bool        tool_checks = true;

    // Pin-Friendly Prompt Processor (PPP): LCP pin_end + optional rearrange.
    // See docs/PIN_FRIENDLY_PROMPT.md. Env: LUCE_PPP=0|1,
    // LUCE_PPP_REARRANGE=0|1, LUCE_PPP_LCP_WINDOW=N,
    // LUCE_PPP_MIN_PIN_TOKENS=N, LUCE_PPP_MAX_EPHEMERAL=N.
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
    // The bare end-of-thinking marker, used as BudgetHook.marker_token_ids
    // so the hook stops once the model closes thinking on its own.
    std::vector<int32_t> think_marker_token_ids;

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

    // Sampler defaults for requests whose final thinking state is OFF
    // (model card's `sampling_no_thinking`, spec §3.3). has_* fields are
    // all false when the card doesn't define the block; omitted request
    // fields then keep falling back to `sampler_defaults` above, matching
    // pre-existing behaviour. Only applied once the request's final
    // thinking state is known — see apply_no_thinking_sampler_defaults.
    SamplingDefaults sampler_defaults_no_thinking;

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
    bool        image_input_enabled = false;
    bool        target_sharding     = false;
    // Prefill chunk size (bargs.chunk). Exposed at /props.runtime.chunk so
    // bench/snapshot tooling can capture the full server config — needed
    // because pre-c35a8a4 snapshots had no /props capture and post-hoc
    // forensics on which chunk was used are otherwise impossible. See
    // docs/specs/props-endpoint.md §4.5.
    int         chunk               = 0;
    // Resolved device placement strings (e.g. "auto:0", "cuda:0"). Sourced
    // from placement_device_name(bargs.device / bargs.draft_device) in
    // server_main after CLI parse.
    std::string target_device;
    std::string draft_device;
    // Idle-to-busy batching window. It is ignored by single-slot engines and
    // never delays an already decoding request.
    int admission_coalesce_ms = 20;
    // Auto resolves after all models load, before workers start. Zero disables.
    size_t decode_kv_offload_bytes = luce::common::kAutoKvOffloadBytes;

    // PFlash (speculative prefill compression)
    bool        pflash           = false;   // --pflash on: compress prompts from pflash_threshold tokens
    int         pflash_threshold = 32000;   // token count threshold for AUTO mode
    float       pflash_keep_ratio = 0.05f;  // fraction of tokens to keep
    std::string pflash_drafter_path;        // path to drafter GGUF (Qwen3.5-0.8B)
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
    size_t      disk_cache_budget_mb = 131072; // max disk usage in MB (128 GB)
    int         disk_cache_min_tokens = 512; // only persist >= this many tokens
    int         disk_cache_continued_interval = 10240; // continued checkpoint every N tokens
    int         disk_cache_cold_max_tokens = 10240;    // cold prefix for prompts longer than this
    double      disk_cache_ttl_hours = 72.0;  // delete files unused this long (0 = never)
    // Graceful shutdown writes the RAM snapshots to disk (one last
    // eviction) for at most this long; < 0 = no limit.
    double      disk_cache_shutdown_flush_s = 120.0;
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
// Validate configured selection, query and serving flags before loading models.
// Keeps the runtime defaults and clamping of valid integer overrides.
bool validate_pflash_environment(std::string & error);
bool should_clamp_flowkv_disk_cache(
    bool flowkv, const DiskPrefixCachePolicy & policy);
// True when the last message other than a system note is a tool result.
// Agent clients append system notes (a context budget, reminders) after
// the tool results; those notes are append-only history too.
bool ends_with_tool_result(const std::vector<ChatMessage> & messages);
// Structured output. The format object a request asks its output to follow:
// response_format (chat completions), text.format (responses) or
// output_format (messages); null when it asks for none. Read from the body
// text as ordered JSON, so the grammar keeps the schema's property order.
nlohmann::ordered_json request_output_format(const std::string & body);
// The JSON schema a format object asks for. False with `error` set for an
// unsupported format type; true with a null schema for plain text.
bool output_format_schema(const nlohmann::ordered_json & format,
                          nlohmann::ordered_json & schema, std::string & error);
// Whether tool_choice obliges the reply to call a tool ("required", or a
// named function in any of the three APIs).
bool tool_choice_requires_call(const json & tool_choice);
// The grammar for a constrained reply: optional leading whitespace, then the
// schema's JSON (when a schema is given) or a tool call that starts with one
// of `tool_openers` (when given). Throws on a schema it cannot convert.
std::string output_constraint_grammar(const nlohmann::ordered_json & schema,
                                      const std::vector<std::string> & tool_openers);
bool canonical_turn_matches_checkpoint(
    const std::vector<int32_t> & prompt,
    const std::vector<int32_t> & completed_turn,
    int checkpoint);
bool canonical_assistant_content(
    const std::string & generation_prompt,
    const std::string & sentinel_rendered,
    const std::string & sentinel,
    const std::string & generated_text,
    std::string & content);

struct PflashQueryWindow {
    int end = -1;       // exclusive token offset in the rendered prompt
    int tokens = 0;     // width of the matching query suffix
    int trailing_trimmed = 0;  // query tokens dropped from its end to match

    bool valid() const { return end >= tokens && tokens > 0; }
};

struct PflashInstructionMessagePlan {
    std::vector<size_t> instruction_messages;
};

PflashInstructionMessagePlan plan_pflash_instruction_messages(
    const std::vector<ChatMessage> & messages);

// Return the conservative token interval in `original` changed by rendering
// a request variant. Used to retain tool definitions independently of where
// an arbitrary chat template places them.
PFlashTokenSpan pflash_changed_token_span(
    const std::vector<int32_t> & original,
    const std::vector<int32_t> & variant) noexcept;

// Sort and merge overlapping/adjacent mapped spans before selector validation.
std::vector<PFlashTokenSpan> canonicalize_pflash_token_spans(
    std::vector<PFlashTokenSpan> spans);

// Content length (drafter tokens) up to which a user turn or assistant
// answer of a multi-turn chat is kept whole: PFLASH_CHAT_SKELETON_TOKENS,
// default 256; 0 keeps only role headers.
int pflash_chat_skeleton_tokens() noexcept;

// Earlier user questions of a multi-turn chat that score alongside the
// current one, most recent first at weights 1/2, 1/4, ...:
// PFLASH_CHAT_HISTORY_QUERIES, default 3, at most 8; 0 scores the current
// question alone.
int pflash_chat_history_queries() noexcept;

// Multi-turn follow-ups: PFLASH_CHAT_RECALL=0 turns recall off (a small
// follow-up is then appended exactly as full prefill appends it, without
// scoring); PFLASH_CHAT_COMPRESS_NEW_TOKENS (default 16384) is the size of
// new material in one turn from which it is compressed instead of appended.
bool pflash_chat_recall() noexcept;

// The compressed text is rebuilt from the kept spans with a paragraph break
// between pieces that were not adjacent in the prompt, unless one side
// already ends or starts a paragraph (PFLASH_SELECT_PARAGRAPH_JOIN=0 turns
// it off).
bool pflash_paragraph_join() noexcept;
std::string pflash_join_kept_spans(
    const Tokenizer & tokenizer,
    const std::vector<int32_t> & ids,
    const std::vector<PFlashTokenSpan> & spans);
int pflash_chat_compress_new_tokens() noexcept;


// The parts of ``spans`` that ``minus`` does not cover. Both canonical.
std::vector<PFlashTokenSpan> pflash_subtract_token_spans(
    const std::vector<PFlashTokenSpan> & spans,
    const std::vector<PFlashTokenSpan> & minus);

// Text of a recalled segment made safe to quote inside a user turn: chat
// control markers are removed, with the role-name line that follows a
// generic role marker ("<|im_start|>assistant\n"), and the result is trimmed.
std::string pflash_recall_excerpt(
    const std::string & text,
    const std::vector<std::string> & role_markers,
    const std::vector<std::string> & end_markers,
    bool generic_role_lines);

// A multi-turn PFlash view: what was served for one turn of a conversation,
// kept so the next turn can append to it instead of recompressing. The next
// request continues the view when its raw prompt starts with this one's up
// to the generation prompt, in target and drafter tokens alike.
// One piece of a served view, in order: raw drafter tokens [begin, end) of
// the request it was taken from (the view's prompt prefix, so the current
// request's coordinates), served verbatim, or (recall) as a recalled excerpt
// of that range under the [Earlier in this conversation] block.
struct PflashViewPiece {
    int begin = 0;
    int end = 0;
    bool recall = false;
    int block = -1;       // recall pieces: the view turn whose recall block holds them
};

struct PflashChatView {
    std::vector<int32_t> raw_tokens;       // target tokens, raw prompt
    int raw_gen_begin = -1;                 // its generation prompt
    std::vector<int32_t> drafter_ids;      // drafter tokens, raw prompt
    int drafter_gen_begin = -1;
    std::vector<int32_t> view_tokens;      // target tokens served
    int view_gen_begin = -1;
    std::vector<PFlashTokenSpan> spans;    // raw drafter spans in the view
    std::vector<PflashViewPiece> pieces;   // the served view, in order (provenance)
    int turns = 0;
};

// One scored turn that compresses only `regions` of the raw prompt (drafter
// tokens, canonical) and serves everything else as the view holds it or raw:
// "arrive" = a large uncached chunk; "new_content" = the newest user turn's
// content of a tool session (PFLASH_AGENT_PASTE=1). The served
// prompt keeps the cached view up to cut_view (raw position cut_raw) and
// serves the rest anew. ``raw_base`` (no view): the raw prompt's own target
// tokens [0, cut_view) stand in for the view, so the earlier conversation is
// served token for token as full prefill would serve it.
struct PflashRegionPlan {
    bool active = false;
    std::string mode;
    std::vector<PFlashTokenSpan> regions;
    int cut_raw = 0;
    int cut_view = 0;
    int tail_after = 0;
    bool has_view = false;
    int region_droppable = 0;
    bool raw_base = false;
};

class PflashChatViewStore {
public:
    explicit PflashChatViewStore(size_t capacity = 8) : capacity_(capacity) {}
    // The stored view the prompt continues (longest match), if any.
    bool find(const std::vector<int32_t> & raw_tokens,
              const std::vector<int32_t> & drafter_ids,
              PflashChatView & out) const;
    // Store a view, replacing the one it continues.
    void remember(PflashChatView view);
    // Drop every view whose raw prefix the prompt continues; return the
    // number dropped. Raw fallback must also work when drafter parsing failed.
    size_t forget(const std::vector<int32_t> & raw_tokens);
    size_t size() const;

private:
    mutable std::mutex mutex_;
    size_t capacity_;
    std::vector<PflashChatView> views_;   // most recent last
};

// Find the last sufficiently-specific suffix of the user query inside the
// rendered drafter-tokenized prompt. Public for model-free regression tests.
PflashQueryWindow find_pflash_query_window(
    const std::vector<int32_t> & prompt,
    const std::vector<int32_t> & query,
    int max_tokens = 8,
    int search_end = -1,
    int search_begin = 0,
    bool anchored = true);

PflashQueryWindow pflash_tail_query_window(
    const std::vector<int32_t> & prompt,
    int max_tokens,
    int query_end = -1,
    int query_begin = 0) noexcept;

// Map the last occurrence of `needle` inside the decoded token text of
// `prompt[begin, end)` to the token span covering it. Searching the joined
// per-token text (not a standalone encoding of `needle`) keeps the mapping
// correct at BPE boundary merges: a token that spans the needle's first or
// last character (e.g. " What" after "Question:") is included in the span.
// Returns {-1, -1} when the needle is absent or the range is invalid.
PFlashTokenSpan pflash_decoded_text_span(
    const Tokenizer & tokenizer,
    const std::vector<int32_t> & prompt,
    int begin,
    int end,
    const std::string & needle);

// PFLASH_SELECT_QUERY_HEAD=1: query windows from the structure of the
// latest user content ``prompt[content_begin, content_end)`` (see
// pflash_query_blocks). ``extra`` holds the leading prose block (or its
// first and last head_cap/2 tokens when longer) and, for a trailing block
// longer than tail_cap, its first tail_cap/2 tokens; ``tail`` is the main
// query window -- the trailing prose block, or its last tail_cap/2 tokens --
// or {-1, -1} to keep today's window. A block under 16 tokens is ignored (a
// stock line such as "Here is the material."), so a short trailing block
// keeps today's window.
struct PflashQueryHeadPlan {
    std::vector<PFlashTokenSpan> extra;
    PFlashTokenSpan tail{-1, -1};
    std::string head_rule = "none";
    bool structured = false;
    int head_tokens = 0;
    int tail_tokens = 0;
};
PflashQueryHeadPlan pflash_query_head_plan(
    const Tokenizer & tokenizer,
    const std::vector<int32_t> & prompt,
    int content_begin,
    int content_end,
    int head_cap,
    int tail_cap);

// The chat turn the scorer query comes from, located by the model's own
// chat control markers in the rendered prompt rather than message
// bookkeeping: the latest user turn (tool output wrapped in a user turn does
// not count), else the latest turn with content. An assistant turn left open
// at the prompt end is the generation prompt -- whatever think or channel
// prefix the template adds to it -- and never a candidate.
// ``role_begin`` is the marker opening the turn (the header to pin);
// ``content_begin`` skips the role-name line ("<|im_start|>user\n") when the
// family uses generic role markers; ``content_end`` sits before the turn's
// closing marker. Both trim the whitespace the template wraps content in.
// ``turn_end`` sits past that closing marker; ``generation_begin`` is the
// generation prompt's marker (the prompt end when there is none);
// ``later_turns`` says assistant or tool turns sit between the two.
// Offsets are token indices in ``prompt``'s own vocabulary. ``markers`` were
// resolved on ``marker_tokenizer`` (the target model's); its marker strings
// are searched in the decoded prompt text, so a drafter whose vocabulary
// lacks the control tokens still maps correctly. Invalid when the prompt
// carries no chat markers.
struct PflashChatTurn {
    int role_begin = -1;
    int content_begin = -1;
    int content_end = -1;
    int turn_end = -1;
    std::string role;       // "user", "assistant", "system", "tool", ...
};

struct PflashChatTurnSpan {
    int role_begin = -1;
    int content_begin = -1;
    int content_end = -1;
    int turn_end = -1;
    int generation_begin = -1;
    bool later_turns = false;
    // Every turn before the generation prompt, in order; ``query_turn``
    // indexes the one above. Tool output wrapped in a user turn has role
    // "tool".
    std::vector<PflashChatTurn> turns;
    int query_turn = -1;

    bool valid() const {
        return content_begin >= 0 && content_end > content_begin;
    }
};

PflashChatTurnSpan pflash_chat_query_turn(
    const Tokenizer & marker_tokenizer,
    const ChatMarkers & markers,
    const Tokenizer & tokenizer,
    const std::vector<int32_t> & prompt);

// PFLASH_SELECT_QUERY_PROBE=1: the content spans of the last ``count``
// messages whose ROLE is "user" (latest first), found among the rendered
// ``turns`` by exact content. A message maps when its trimmed content
// equals the trimmed decoded content of exactly as many turns (other than
// assistant/model/system/developer turns) as there are user messages with
// that content; equal contents map in order. The span covers every token
// overlapping the message's exact content, its own leading/trailing
// whitespace included. A message with empty content maps to the empty span
// {0, 0}. False (``out`` cleared) when any message
// does not map that way -- the caller then maps by sentinel renders.
// cand_v4 / stage 2 (STAGE2-CONTRACT.md section 1): the character span of every message's exact content in
// ``prompt_text``, from ``placeholder_render`` -- a render of the same messages with content i replaced by
// ``placeholders[i]`` ("" = left as is). Each non-empty placeholder must occur exactly once, in message order, and
// putting the contents back must give ``prompt_text`` byte for byte (a template that rewrites content fails).
// ``spans[i]`` = {begin, end} characters ({0, 0} for a message left as is). False + ``error`` otherwise.
bool pflash_content_char_spans(
    const std::string & placeholder_render,
    const std::vector<std::string> & placeholders,
    const std::vector<std::string> & contents,
    const std::string & prompt_text,
    std::vector<std::pair<size_t, size_t>> & spans,
    std::string & error);
// The tokens whose character extent [token_begin[t], token_begin[t+1]) overlaps [begin, end) (the last token ends
// at ``text_size``); {0, 0} when the range is empty.
PFlashTokenSpan pflash_char_span_tokens(
    const std::vector<size_t> & token_begin, size_t text_size, size_t begin, size_t end);

bool pflash_human_message_spans(
    const Tokenizer & tokenizer,
    const std::vector<int32_t> & prompt,
    const std::vector<PflashChatTurn> & turns,
    const std::vector<ChatMessage> & messages,
    size_t count,
    std::vector<PFlashTokenSpan> & out);

// Return the original prompt offset immediately before the stable trailing
// suffix shared with a version whose latest user message carries a sentinel.
// Invalid when no such bounded suffix can establish the semantic boundary.
int pflash_query_search_end_from_sentinel(
    const std::vector<int32_t> & original,
    const std::vector<int32_t> & sentinel) noexcept;

// Return the first token offset affected by a version whose selected message
// content carries a leading sentinel. This is a conservative lower bound for
// the selected content in the original rendered prompt.
int pflash_query_search_begin_from_sentinel(
    const std::vector<int32_t> & original,
    const std::vector<int32_t> & sentinel) noexcept;

// The served view's provenance pieces as [[begin, end, "raw"|"recall"], ...].
json pflash_view_pieces_json(const std::vector<PflashViewPiece> & pieces);

std::string pflash_token_fingerprint(
    const std::vector<int32_t> & ids);

bool pflash_full_cache_restore_allowed(
    bool selection_environment_present) noexcept;
// Tokens the strict selector keeps whatever their score, counted the way it
// charges them: every fixed chunk overlapping the query window or a kept span
// and, with ``query_suffix_structural``, every chunk after the query. Probe
// segments cut exactly at those edges, so this is an upper bound for them.
int pflash_kept_tokens(
    int input_tokens,
    int chunk_size,
    int query_begin,
    int query_end,
    const std::vector<PFlashTokenSpan> & kept_spans,
    bool query_suffix_structural) noexcept;
// The keep ratio that spends ``keep_ratio`` on the droppable tokens only:
// (kept + keep_ratio * (input - kept) + 1) / input, capped at 1.
double pflash_effective_keep_ratio(
    int input_tokens, int kept_tokens, double keep_ratio) noexcept;
// The selector keep ratio for the next re-encode-overshoot retry: ``tightening`` (cumulative over the retries of
// one request, starts at 0) grows by overflow + 1 and the ratio is requested * (ceiling - tightening) / ceiling,
// so the ratios only tighten (defect fix 9 Oct 2026; the first retry is what it always was). <= 0: no retry.
float pflash_ceiling_retry_keep_ratio(float requested_keep_ratio, int target_ceiling, int overflow,
                                      int & tightening);
int pflash_target_token_ceiling(
    int original_target_tokens, double keep_ratio) noexcept;

// PFlash failure policy (user decision D5, 9 Oct 2026). When compression of a
// prompt fails at runtime (non-finite scores, scorer / probe / selection or
// mapping errors), the request is served uncompressed -- if the raw prompt
// passes the same admission an uncompressed request gets (max_ctx with the
// reply budget) -- and the fallback is recorded (log line, view-trace record,
// usage.timings.pflash.pflash_fallback, /props counters). PFLASH_FAIL_CLOSED=1
// restores the previous behaviour: the request fails with the PFlash error.
// Configuration and request-contract errors never fall back.
struct PflashFallbackDecision {
    bool fallback = false;
    // Why the fallback was not taken: "fail_closed", "config_error" or
    // "raw_exceeds_max_ctx"; empty when ``fallback``.
    std::string refused;
};
bool pflash_fail_closed_from_env() noexcept;
// Whether a strict selection scores every candidate (the block-15 head over segments), which the unified view pages
// by; the legacy and split scorers (or PFLASH_QWEN35_LEGACY_SCORER=1) and the fold strategy do not.
bool pflash_selection_gives_lifts(const luce::pflash::PFlashSelectionConfig & selection) noexcept;
// Short machine-readable reason for a PFlash error string (and the backend's
// last error, when the drafter reported one): "nonfinite_scores",
// "compress_failed", "ceiling_exceeded", "mapping_failed", "remote_drafter",
// "selection_failed", "spread" (the relevance-spread fallback, not a failure),
// "spread_unavailable" or "other".
std::string pflash_failure_reason(const std::string & error,
                                  const std::string & detail);
// PFLASH_FALLBACK_SPREAD decision: fire when the threshold is on (> 0), the
// spread is valid and spread >= threshold.
bool pflash_spread_fires(double threshold, bool spread_valid, double spread) noexcept;
// The spread fallback's decision for one selection. A request compressed by a
// region plan (arrive: a mid-chat paste on a cached history) takes no
// decision: the threshold was calibrated on whole-prompt selections, where
// the prompt-start candidate holds most of the score mass; a region has no
// such candidate, so the spread there has no calibrated meaning. It is
// still computed and recorded (fired false, skipped "region_plan").
// Without a region plan: exactly pflash_spread_fires.
struct PflashSpreadDecision {
    bool fires = false;
    const char * skipped = nullptr;   // "region_plan" when no decision was taken
};
PflashSpreadDecision pflash_spread_decision(double threshold, bool spread_valid,
                                            double spread, bool region_plan) noexcept;
bool pflash_failure_is_config_error(const std::string & error);
PflashFallbackDecision pflash_fallback_decision(
    const std::string & error, bool fail_closed, int raw_prompt_tokens,
    int max_output, int max_ctx);

// Cache-aware admission (PFLASH_ADMISSION=cache; default "length" = today's
// rule, droppable tokens >= threshold). A conversation without a PFlash view
// is compressed only when the reader would have to prefill at least
// ``threshold`` droppable tokens that its restorable prefix cache does not
// hold, and that cache holds less than ``threshold`` droppable tokens:
// compressing never throws away a threshold's worth of reusable cache. A raw
// prompt that does not fit the context is always compressed.
enum class PflashAdmission {
    Compress, UnderThreshold, WarmCache, NewContent, KeepHistory };
const char * pflash_admission_name(PflashAdmission admission) noexcept;
bool pflash_cache_admission_enabled() noexcept;
PflashAdmission pflash_cache_admission(
    int uncached_droppable, int cached_droppable, int threshold,
    bool raw_fits = true) noexcept;
// Tool activity: the request defines tools, or its history carries an
// assistant tool call or a tool result (normalized messages, any API format).
bool pflash_has_tool_activity(const json & tools,
                              const std::vector<ChatMessage> & messages) noexcept;
// Tool sessions (cache admission, --pflash on). In a
// conversation with tool activity (a "tool session") the agent's history and
// tool output are never compressed. By default nothing in a tool session is:
// it is served raw, full prefill (``KeepHistory``). PFLASH_AGENT_PASTE=1
// (exactly "1"; anything else = off) lets PFlash compress one thing, the
// newest turn when it is a user message whose droppable tokens reach the
// threshold (decision ``NewContent``: a region plan over that content alone,
// served on a raw base -- the raw prompt's own tokens before it, so a first
// turn's system prompt and tools as well -- or on the conversation's view).
// Everything else is served raw (``KeepHistory``, or the raw decision that
// already applies), and a view is only continued: new turns appended
// verbatim (a tool output of any size included), recall at the newest user
// message, never rebuilt and never compressed again as a whole. A raw prompt
// over the context is not compressed either: unless a big newest paste
// brings it under (switch on), the request gets the context-overflow error
// full prefill gets.
bool pflash_agent_paste_enabled() noexcept;
// The newest turn of ``turns`` when it is a user turn with content: returns
// nullptr and sets ``newest``; otherwise what the newest turn is
// ("no_turns", "newest_not_user": a tool output, an assistant prefill or an
// empty user turn) and newest = -1.
const char * pflash_keep_history_scope(
    const std::vector<PflashChatTurn> & turns, int & newest) noexcept;
// The decision for a tool session without a view: UnderThreshold stays;
// newest user content of at least ``threshold`` droppable tokens turns
// Compress and WarmCache into NewContent; otherwise (``new_droppable`` is 0
// when the newest turn is not a user message) Compress, and WarmCache when
// ``arrive`` would compress the uncached tail, become KeepHistory; WarmCache
// without arrive stays (raw).
PflashAdmission pflash_keep_history_admission(
    PflashAdmission admission, int new_droppable, int threshold,
    bool arrive) noexcept;
// Target tokens of ``prompt`` that hold exactly the drafter tokens
// ``drafter_ids``[0, drafter_cut), where drafter_cut is the role_begin of
// ``turns``[turn] (the drafter parse): the same turn in a parse of the
// target tokens, verified by decoding both prefixes. -1 when the parses or
// the decoded prefixes differ (fail closed: the caller serves raw).
int pflash_raw_base_target(
    const Tokenizer & target, const ChatMarkers & markers,
    const std::vector<int32_t> & prompt, const Tokenizer & drafter,
    const std::vector<int32_t> & drafter_ids,
    const std::vector<PflashChatTurn> & turns, int turn);
// The restorable prefix PFlash admission counts: the RAM prefix snapshot, or
// the deepest entry of the disk session cache when that cache is on and holds
// more (a restarted server, or a session written out on eviction). Ties go to
// RAM. With the disk cache off it is the RAM prefix, and ``source`` is "ram"
// (or "none" when nothing is cached).
struct PflashRestorablePrefix {
    int tokens = 0;
    const char * source = "none";
};
PflashRestorablePrefix pflash_restorable_prefix(
    int ram_tokens, bool disk_enabled, int disk_tokens) noexcept;
// Tokens of the canonical ``spans`` inside [begin, end).
int pflash_span_tokens_within(
    const std::vector<PFlashTokenSpan> & spans, int begin, int end) noexcept;
// Large-chunk arrival (PFLASH_ARRIVE=1, cache admission only) and the chat
// view's rebuild ratio (PFLASH_CHAT_REBUILD_RATIO, default 2; 0 = rebuild only
// at the context limit).
double pflash_chat_rebuild_ratio() noexcept;
int pflash_chat_rebuild_min_tokens() noexcept;
bool pflash_arrive_enabled() noexcept;
// [0, n) minus the canonical ``spans``.
std::vector<PFlashTokenSpan> pflash_complement_token_spans(
    const std::vector<PFlashTokenSpan> & spans, int n);
}  // namespace http_detail

// ─── Parsed request ─────────────────────────────────────────────────────

struct ParsedRequest {
    ApiFormat                  format;
    std::vector<int32_t>      prompt_tokens;  // tokenized prompt
    ImagePromptHandle         images;
    std::string               rendered_prompt;
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
    // Structured output format (http_detail::request_output_format), null
    // when the request asks for none.
    nlohmann::ordered_json    output_format;
    // Response ID
    std::string               response_id;
    // Thinking/reasoning state
    bool                      thinking_enabled = true;
    bool                      started_in_thinking = false;
    // Normalized model-facing effort. DeepSeek V4 officially defines low,
    // high, and max; high and max select distinct prompt prefixes.
    std::string               reasoning_effort;
    // True when the request opted in to the thinking-budget envelope via
    // thinking.type="enabled" or an explicit reasoning effort. Distinct from
    // thinking_enabled, which is the final template-rendering state after
    // overrides. Bare chat-template toggles remain renderer-only. When true,
    // the response includes a `finish_details` block.
    bool                      thinking_opt_in = false;
    // Per-request thinking-budget envelope (spec §4). Populated from
    // `thinking.budget_tokens` and `thinking.reply_budget`, or selected
    // from server-configured effort tiers when `reasoning.effort` is set.
    // -1 = not set; the server falls back to its global think_max_tokens /
    // hard_limit_reply_budget. Values are already clamped to those ceilings.
    int                       per_req_phase1_cap   = -1;
    int                       per_req_reply_budget = -1;
    // Jinja-path only: `preserve_thinking` from `chat_template_kwargs`.
    // Tri-state: -1 = not set (template default applies, typically true),
    // 0 = false, 1 = true. Controls whether official templates (e.g.
    // qwen4exp) replay earlier assistant turns' recorded <think> block
    // (message.reasoning_content) or strip it.
    int                       preserve_thinking = -1;
    // Stop sequences (OpenAI "stop" + Anthropic "stop_sequences")
    std::vector<std::string>  stop_sequences;
    // Bandit: per-session adaptive keep_ratio opt-in
    std::string               session_id;
    std::string               pflash_query;   // explicit scorer query text (optional request field)
    // Literal strings inside the boundary message that must survive
    // compression (e.g. an answer-format directive embedded in a user
    // message). Each occurrence is mapped and retained as a mandatory span.
    std::vector<std::string>  pflash_required;
    DiskPrefixCachePolicy     disk_cache_policy;
    // PPP: stable pin cut for tool-heavy requests (0 = use default boundary).
    int                       pin_end_token = 0;
    // The prompt ends with tool results: an agent's step, which clients only
    // append to.
    bool                      ends_with_tool_result = false;
    // Concurrent serving: the restorable prefix of prompt_tokens (target
    // tokens) the scheduler measured before it handed the request to the
    // PFlash thread, which never reads the prefix cache. Empty = not measured.
    std::optional<http_detail::PflashRestorablePrefix> pflash_cached_prefix;
    // The client prompt's size when prompt_tokens holds the compressed
    // prompt the concurrent scheduler serves; -1: prompt_tokens is the
    // client's. usage.prompt_tokens reports it (usage_prompt_tokens).
    int                       client_prompt_tokens = -1;
};

inline int usage_prompt_tokens(const ParsedRequest & req) {
    return req.client_prompt_tokens >= 0 ? req.client_prompt_tokens
                                         : (int) req.prompt_tokens.size();
}

// Resident budget the PrefixCache enforces: the concurrent limit in paged
// mode, else --prefix-cache-max-mib, with auto sized from the backend's
// snapshot at full context.
struct PrefixCacheBudget {
    size_t bytes = 0;         // 0 = unlimited
    bool automatic = false;   // sized by auto
    bool sized = true;        // false: the backend cannot size snapshots
    std::string error;        // set when an explicit limit cannot apply;
                              // startup rejects that configuration
};
// Automatic concurrent prefix budget: 2 x slots + 1 checkpoints of
// per_checkpoint bytes, at least ServerConfig::kConcurrentPrefixBudgetFloor
// and, above that, at most memory / 4 when memory is known (non-zero).
size_t auto_concurrent_prefix_budget(size_t per_checkpoint, int slots, size_t memory);

PrefixCacheBudget resolve_prefix_cache_budget(const ServerConfig & config,
                                              const ModelBackend & backend);

// Parse request sampler fields, applying model-card defaults where present.
SamplerCfg parse_request_sampler(const json & body,
                                 const SamplingDefaults & defaults);

// Overlay the model card's no-thinking sampler defaults onto an
// already-parsed `sampler`, for any field the request body did not
// explicitly set. parse_request_sampler runs before the request's final
// thinking state is known (apply_request_reasoning resolves it afterward,
// since the effort-tier phase-1 cap depends on ParsedRequest::max_output),
// so it can only apply the thinking-mode `sampling` defaults; this patches
// the no-thinking fields in once the final state is available. Call only
// when the resolved thinking state is OFF. A no-op when `no_thinking`
// carries no has_* fields (card without `sampling_no_thinking`), which
// preserves pre-existing behaviour: omitted fields keep the thinking-mode
// defaults regardless of thinking state.
void apply_no_thinking_sampler_defaults(const json & body,
                                        const SamplingDefaults & no_thinking,
                                        SamplerCfg & sampler);

// Read the required `messages` field. Throws std::invalid_argument when
// it is missing or not a non-empty array; route_request's catch turns
// that into a 400.
json require_messages_array(const json & body);

// Resolve the supported output-token aliases in precedence order. Only the
// selected field is parsed, so malformed lower-priority aliases are ignored.
int resolve_max_output_tokens(const json & body, int default_max_tokens);

// Apply request-level thinking controls and resolve the model-facing effort
// plus the server's phase-1 budget. Kept independent of HttpServer so the
// wire-format precedence and compatibility aliases can be unit-tested.
void apply_request_reasoning(const json & body,
                             const ServerConfig & config,
                             ParsedRequest & req);

// Sticky tools-boundary pinning is part of PPP and must follow its master
// toggle. Kept as a small policy helper so the disabled path is testable.
bool ppp_prefers_tools_boundary(bool ppp_enabled, bool has_tools);

// The tokens PPP remembers as a recent tool head: what the client sent, never
// a PFlash view. A view is the server's own rewrite; one that dropped a token
// of the system/tools head (e.g. the system turn's "<|im_end|>\n") would make
// the next uncompressed request pin -- and every later one restore -- at the
// view's divergence point instead of the head boundary, so an uncompressed
// prompt would prefill split differently than with PFlash off. The remembered
// head is unchanged when nothing was compressed, or when the view kept the
// head verbatim.
const std::vector<int32_t> & ppp_tool_prefix_source(
    bool compressed, const std::vector<int32_t> & request_tokens,
    const std::vector<int32_t> & served_tokens);

// Build the /props response body. Exposed (non-static) so unit tests
// can assert on its shape without spinning up a real socket. See
// docs/specs/props-endpoint.md for the wire contract.
json build_props_body(const ServerConfig & config,
                      const PrefixCache & prefix_cache,
                      const ToolMemory & tool_memory);

// ─── HTTP server ────────────────────────────────────────────────────────
class HttpServer {
public:
    HttpServer(luce::engine::LuceEngine & engine,
               Tokenizer & tokenizer,
               const ServerConfig & config);
    ~HttpServer();

    HttpServer(const HttpServer &) = delete;
    HttpServer & operator=(const HttpServer &) = delete;

    // Set the optional pflash drafter tokenizer.
    void set_drafter_tokenizer(Tokenizer * tok) { drafter_tokenizer_ = tok; }

    // Set the chat template format (detected from model arch).
    void set_chat_format(ChatFormat fmt) { chat_format_ = fmt; }

    // Start one listener. Optional model contexts are borrowed until run()
    // returns and must include this first. Model names must be explicit and unique.
    // Each context gets its existing scheduler or single-request worker, never another socket.
    int run(const std::vector<HttpServer *> & models = {});

    // Finalize after run() returns; also called by the destructor.
    // Use request_stop() to stop a running listener from another thread.
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
    friend struct SchedulerTestHarness;
    friend struct HttpServerTestAccess;

    // Client thread: read HTTP request, parse, enqueue job, wait.
    void handle_client(SocketHandle fd);

    struct HttpRequest;
    bool start_worker();
    bool route_model_request(SocketHandle fd, ParsedRequest & req, bool count_only);
    bool handle_model_request(SocketHandle fd, ParsedRequest & req, bool count_only,
                              RoutingAdmission * admission = nullptr);
    // Tool-use checks: the request runs as inner chat completions through
    // handle_model_request, and the checked reply goes to the client.
    bool serve_with_tool_checks(SocketHandle fd, ParsedRequest & req,
                                RoutingAdmission * admission);
    // One inner non-streaming chat completion. False when the router's
    // admission turned it away (nothing was sent); otherwise `status` and
    // `body` hold the HTTP response.
    bool inner_chat_completion(const json & request, RoutingAdmission * admission,
                               int & status, std::string & body);
    json model_routing_status();

    // Worker thread: process jobs sequentially. process_job owns the
    // lifecycle of one dequeued request, including signaling completion.
    void worker_loop();
    void process_job(ServerJob * job);

    struct PreparedPrompt {
        std::vector<int32_t> tokens;
        ImagePromptHandle images;
        bool compressed = false;
        bool flowkv = false;
        int full_cache_served_tokens = -1;
        int full_cache_hit_slot = -1;
        int full_cache_hit_len = 0;
        // Where to take this request's prefix-cache snapshot when nothing
        // else asks for one: a multi-turn PFlash view sets the start of its
        // generation prompt, where the next turn's prompt branches off.
        int snapshot_cut = -1;
        // PFlash details for usage.timings.pflash (see build_timings_json).
        nlohmann::json pflash_stats;
        // Why the unified view handed this request to the chat view ("" = it did not).
        std::string unified_handback;
        // A hand-back the unified view continues from once the chat view has served it (a cold history, or newest
        // content it cannot page): this request's drafter ids and where its generation prompt starts in them.
        std::vector<int32_t> unified_adopt_drafter_ids;
        int unified_adopt_drafter_gen_begin = -1;
        // Served from a unified view (or adopted by one). A compressed one may still get a disk key (its served tokens):
        // the view is kept on disk too, so a resumed conversation serves the same tokens again.
        bool unified_view = false;
        int error_status = 0;
        std::string error;
        // The drafter's own error when compression failed in the backend
        // (luce_last_error at that moment), for the fallback record.
        std::string pflash_failure_detail;
        // PFLASH_FALLBACK_SPREAD fired: the spread record for the fallback
        // (null otherwise). Not a failure: PFLASH_FAIL_CLOSED does not apply.
        nlohmann::json pflash_spread_fallback;
    };

    // Prompt preparation keeps the FlowKV and whole-prompt PFlash policies
    // out of the decode path while preserving their shared precedence rules.
    PreparedPrompt prepare_prompt(const ParsedRequest & req);
    // Concurrent dense Qwen3.5 serving compresses on the scheduler's PFlash
    // thread (scheduler.cpp), for prompts up to --max-ctx; other engines keep
    // their paths.
    bool pflash_concurrent_serving() {
        return backend_.seq_engine() != nullptr && config_.arch == "qwen35" &&
               config_.pflash &&
               drafter_tokenizer_ != nullptr;
    }
    void apply_flowkv_compression(const ParsedRequest & req,
                                  PreparedPrompt & prepared);
    std::string apply_pflash_compression(const ParsedRequest & req,
                                         PreparedPrompt & prepared);
    // After the shared query and pins, when the selection gives per-candidate scores: pages the newest turn's large uncached content
    // (a user message, or listed tool outputs), continues the conversation's view, and recalls unheld units of
    // earlier regions at a new user question. ``request`` carries the drafter ids, the pinned spans and the
    // learned query; ``messages`` the normalised messages the prompt was rendered from. ``handback`` set (the
    // reason): the view did not take the request -- ``not_ready`` non-empty, a prompt it cannot cut, or a conversation it would
    // start while a threshold's worth of uncached, unpinned prompt lies outside its regions -- and the chat view
    // (the release path) serves it; nothing of the unified view's state changed.
    std::string apply_unified_view(const ParsedRequest & req, PreparedPrompt & prepared,
                                   const http_detail::PflashChatTurnSpan & turn,
                                   const std::vector<ChatMessage> & messages,
                                   ModelBackend::CompressRequest request, int chunk_size,
                                   const std::string & not_ready, std::string & handback);
    // A conversation that left the unified view (a raw fallback, or a hand-back to the chat view) is not continued
    // later from one of its views: drops the lineages these raw tokens would continue, in RAM and on disk. Returns
    // how many.
    size_t forget_unified_views(const std::vector<int32_t> & raw_tokens);
    // The chat view served a request the unified view handed back with an adoption (unified_adopt_*): what it served
    // becomes the conversation's unified view, with no regions, so the next turn continues it append-only.
    void adopt_unified_view(const ParsedRequest & req, PreparedPrompt & prepared);
    // The generation prompt in target tokens: the drafter ids from ``generation_begin`` re-encoded.
    std::vector<int32_t> pflash_generation_tokens(const std::vector<int32_t> & drafter_ids,
                                                  int generation_begin) const;
    // Multi-turn: serve the conversation's previous view plus this turn's
    // new material -- verbatim below PFLASH_CHAT_COMPRESS_NEW_TOKENS, else
    // the parts the fresh selection keeps -- with the segments the fresh
    // selection wants that the view lacks recalled at the new user turn.
    // Without a fresh compression (``fresh`` null) it serves only what needs
    // no scoring: a repeated prompt, or a small follow-up with recall off;
    // it returns false for the caller to compress. With one it always
    // serves: a continued view, or the fresh prompt as a new view.
    // ``keep_history`` (a tool session, PFLASH_AGENT_PASTE=1): the view
    // only continues -- new tokens are appended verbatim (no compress-new),
    // recalled passages are clipped to the message content that holds them,
    // the view is never rebuilt (neither when it outgrows a fresh selection
    // nor at the context limit), and when it cannot continue it returns false
    // for the caller to serve the raw prompt.
    bool serve_pflash_chat_view(
        const ParsedRequest & req,
        const std::vector<int32_t> & drafter_ids,
        const http_detail::PflashChatTurnSpan & turn,
        const std::vector<int32_t> * fresh,
        const std::vector<PFlashTokenSpan> * kept_spans,
        const std::vector<PFlashTokenSpan> & pins,
        std::vector<int32_t> & served,
        int & snapshot_cut,
        nlohmann::json & stats,
        bool keep_history = false);
    // PFLASH_VIEW_TRACE_PATH: one JSONL record per compressed request with
    // the served prompt's text, for evidence checks in evaluations.
    // Target tokens of ``prompt`` the reader can restore right now: its
    // in-memory prefix cache (a live backend snapshot, its authoritative
    // position) or, when the disk session cache is on and the request's disk
    // policy is Full (the scope the generation path looks up), the deepest
    // disk entry -- an index lookup, no file read. No cache state changes.
    http_detail::PflashRestorablePrefix restorable_prefix_tokens(
        const std::vector<int32_t> & prompt,
        const DiskPrefixCachePolicy & disk_policy);
    int ram_restorable_prefix_tokens(const std::vector<int32_t> & prompt);
    // Serve a PflashRegionPlan turn from the selection's kept spans; false
    // when it cannot be served (the caller serves the raw prompt).
    bool serve_pflash_region_view(
        const ParsedRequest & req,
        const std::vector<int32_t> & drafter_ids,
        const http_detail::PflashChatTurnSpan & turn,
        const http_detail::PflashRegionPlan & plan,
        const std::vector<PFlashTokenSpan> & kept_spans,
        std::vector<int32_t> & served,
        int & snapshot_cut,
        json & stats);
    void trace_pflash_served(const ParsedRequest & req,
                             const PreparedPrompt & prepared);
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
        PrefixCache::InlineReservation snap_reservation;
        int snap_slot = -1;
        int snap_cut = 0;
        bool snap_prepared = false;
    };

    // A request's restore points (prefix_restore_points) on both serving
    // paths: its chat boundaries spaced for the backend, plus `cuts`. The
    // generation prompt's own boundary, where the request's snapshot goes,
    // stays only on a backend that cuts prefill at every restore point.
    std::vector<int> request_restore_points(const std::vector<int32_t> & prompt,
                                            std::initializer_list<int> cuts) const;
    GenerationCacheState prepare_generation_cache(
        const ParsedRequest & req, PreparedPrompt & prepared,
        GenerateRequest & generate_request);
    void finalize_generation_cache(
        const ParsedRequest & req, const PreparedPrompt & prepared,
        GenerationCacheState & cache, const GenerateResult & result,
        int completion_tokens, bool visible_output_seen,
        bool client_disconnected);
    // Save the live post-generation state as an inline checkpoint of the
    // prompt plus those generated tokens that agree with `canonical` (the
    // conversation with this turn appended, as the next request renders it).
    bool save_generated_turn(
        const std::vector<int32_t> & prompt,
        const std::vector<int32_t> & generated,
        const std::vector<int32_t> & canonical,
        const GenerationCacheState & cache);
    // Remember a tool-call turn in tool memory as the text of its generated
    // `tokens`, with the text the generation prompt opened it with, so the
    // next stateless request renders the turn as it was generated. Returns the
    // conversation with the turn appended, or nothing when there is no turn
    // to remember.
    std::vector<ChatMessage> remember_tool_turn(
        const ParsedRequest & req, const std::vector<int32_t> & tokens,
        const SseEmitter & emitter);
    void remember_agent_turn(
        const ParsedRequest & req, const PreparedPrompt & prepared,
        const GenerationCacheState & cache, const GenerateResult & result,
        const SseEmitter & emitter, int completion_tokens,
        bool visible_output_seen, bool client_disconnected,
        bool replay_cache);
    void forget_inline_slot_metadata(int slot);
    // Free a RAM snapshot slot for reuse, writing it to disk first when
    // PrefixCache::persist_on_evict() says so (see persist_evicted_slot).
    void evict_inline_slot(int slot);
    // Just before `reservation` commits: write (when persist_on_evict says
    // so) and free the further entries its resident-budget plan evicts
    // (PrefixCache::InlineReservation::extra_victim_slots). The commit then
    // drops their metadata. A capture that fails never gets here.
    void evict_extra_victims(const PrefixCache::InlineReservation & reservation);
    // Disk persistence (docs/PREFIX_CACHE.md): a snapshot is written only
    // when it leaves RAM, never after every turn. Hands the payload of
    // `slot` to the disk writer when `persist` is set and the slot has a disk
    // key (slot_tokens_); the slot is left empty. Returns true when queued.
    bool persist_evicted_slot(int slot, bool persist, const char * reason);
    // Slots committed during the current request get this request's
    // tool-call replay text (slot_aux_) once its response is known.
    void note_session_slot(int slot);
    void record_session_aux(const ParsedRequest & req, const SseEmitter & emitter);
    // The tool-call replay text a session's prompt was rendered with
    // (ToolMemory), stored with its disk snapshot so a restart renders the
    // same tokens; restored into ToolMemory at startup.
    std::string session_replay_blob(const ParsedRequest & req,
                                    const SseEmitter * emitter);
    void restore_replay_blobs();
    // Graceful shutdown: evict every RAM snapshot to disk (newest of each
    // conversation and the heads; not edit backups), bounded by
    // disk_cache_shutdown_flush_s.
    void flush_sessions_to_disk(const char * reason);
    // After a commit in `slot`: free the snapshots it supersedes, then evict
    // until committed snapshots fit the resident budget again.
    void trim_snapshots_after_commit(int slot);

    struct GenerationInputs {
        GenerateRequest request;
        int generation_cap = 0;
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
    // Automatic concurrent prefix budget for `slots` checkpoints of `per_checkpoint` bytes.
    size_t concurrent_prefix_budget(size_t per_checkpoint, int slots) const;

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
    bool render_and_tokenize_request(
        SocketHandle fd, const std::vector<ChatMessage> & chat_messages,
        ParsedRequest & req);
    bool render_messages_to_text(
        const std::vector<ChatMessage> & chat_messages,
        const ParsedRequest & req, bool add_generation_prompt,
        std::string & rendered, std::string & error);
    // A direct request's max_tokens is trimmed to the context its prompt
    // leaves; a routed one (another model may hold it whole) is refused
    // without a response, so the router can try the next model.
    bool validate_request_context(SocketHandle fd, ParsedRequest & req,
                                  bool routed = false);
    // Constrain the reply to the request's structured output format and
    // tool_choice (common/output_constraint.h). False after sending a 400
    // for a format it cannot serve.
    bool attach_output_constraint(SocketHandle fd, ParsedRequest & req);
    // Every token's text, built on first use and shared by all requests.
    std::shared_ptr<const ConstraintVocab> output_constraint_vocab();
    void log_parsed_request(const ParsedRequest & req) const;
    RoutingAdmission enqueue_request_and_wait(SocketHandle fd, ParsedRequest req,
                                  bool report_admission = false);

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
    bool has_pending_jobs();

    // Members.
    luce::engine::LuceEngine & engine_;
    ModelBackend &   backend_;
    Tokenizer &      tokenizer_;
    Tokenizer *      drafter_tokenizer_ = nullptr;  // pflash drafter (optional)
    ServerConfig     config_;
    ChatFormat       chat_format_;
    std::once_flag   constraint_vocab_once_;
    std::shared_ptr<const ConstraintVocab> constraint_vocab_;
    PFlashDrafterIpcClient pflash_remote_;
    ToolMemory       tool_memory_;
    PrefixCache      prefix_cache_;
    DiskPrefixCache  disk_cache_;

    // Per-session adaptive keep_ratio bandit state.
    HttpServerSessions sessions_;

    // Multi-turn PFlash views, matched by raw prompt prefix.
    http_detail::PflashChatViewStore pflash_views_;

    // Read once at construction: one view and recall mechanism for chat and agent sessions, on when the PFlash
    // selection gives per-candidate scores. Off, none of it runs and nothing below is read.
    bool pflash_unified_ = false;
    bool pflash_unified_recall_ = true;
    // PFLASH_RECALL_GATE=<theta> (default 0.075): recall only
    // while the head mass on unheld units is at least theta of the context mass; also skips the drafter pass when
    // every unit of every region is held. < 0 = "off" (fill each region's cap).
    double pflash_recall_gate_ = -1.0;
    // PFLASH_RECALL_TRACE_UNITS=1: the view record carries each region's units (byte cuts, kept at paging; at a
    // question: held, head mass, cost, score) for an offline replay of recall policies. Stats only.
    bool pflash_recall_trace_units_ = false;
    pflash_unified::ToolList pflash_region_tools_;
    std::string pflash_unified_config_;   // config identity a stored view must match
    pflash_unified::ViewStore pflash_unified_views_;
    pflash_unified::ViewDiskStore pflash_unified_disk_views_;   // with the disk session cache on
    std::atomic<uint64_t> pflash_unified_lineage_{0};

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

    // Disk key of each RAM snapshot slot (written on eviction/shutdown).
    std::unordered_map<int, std::vector<int32_t>> slot_tokens_;
    std::unordered_set<int> agent_turn_cache_slots_;
    // Calls a tool-use final check added (tool_checks.h).
    tool_checks::CheckedCalls checked_calls_;
    // Replay blob stored with each slot's disk snapshot.
    std::unordered_map<int, std::string> slot_aux_;
    std::vector<int> request_session_slots_;

    // Memory held by the live cache and saved snapshots, rebuilt on the
    // generation thread after each request and read by /props (see
    // publish_memory_report and memory_json).
    mutable std::mutex memory_report_mu_;
    json memory_report_;
    uint64_t memory_report_requests_ = 0;
    void publish_memory_report();
    json memory_json() const;
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

    // Immutable model table after run() starts; only reservations mutate under
    // routing_mu_. A reservation spans parsing through job retirement/output
    // draining, so disconnects never make still-running engine work invisible.
    struct RoutedModel {
        HttpServer * server;
        int capacity;
        int in_flight = 0;
    };
    std::vector<RoutedModel> models_;
    std::mutex routing_mu_;
    std::condition_variable routing_cv_;
    int routing_waiters_ = 0;

    // Request queue consumed by the serving loop owned by LuceEngine.
    std::mutex                      queue_mu_;
    std::condition_variable         queue_cv_;
    ServerJob *                     queue_head_ = nullptr;
    ServerJob *                     queue_tail_ = nullptr;
    std::atomic<bool>               stopping_{false};
    // PFlash failure policy counters (D5), exposed under /props pflash.failures.
    mutable std::mutex              pflash_failure_mu_;
    uint64_t                        pflash_fallback_served_ = 0;
    uint64_t                        pflash_fallback_refused_ = 0;
    std::map<std::string, uint64_t> pflash_fallback_reasons_;   // served, by reason
    std::map<std::string, uint64_t> pflash_refused_reasons_;    // refused, by why
    // Tool-session decisions, by name (/props pflash.keep_history).
    std::map<std::string, uint64_t> pflash_keep_history_counts_;
    // Unified decisions and events, by name (/props pflash.unified).
    std::map<std::string, uint64_t> pflash_unified_counts_;
    nlohmann::json pflash_failure_counters_json() const;
    nlohmann::json pflash_keep_history_json() const;
    nlohmann::json pflash_unified_json() const;

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
    bool          report_admission = false; // return busy before committing a response
    RoutingAdmission admission = RoutingAdmission::handled; // published with done
    bool          announced = false;
    // First concurrent-scheduler attempt; retained across busy deferrals so
    // server-side prefill/elapsed telemetry does not erase queueing delay.
    std::chrono::steady_clock::time_point parallel_started_at{};
    // Concurrent PFlash (scheduler.cpp): the job came back from the
    // compression thread. When it was compressed, req.prompt_tokens holds the
    // served prompt and req.client_prompt_tokens the client prompt's size.
    bool          pflash_prepared = false;
    nlohmann::json pflash_stats;          // usage.timings.pflash
    int           pflash_error_status = 0; // prompt preparation failed
    std::string   pflash_error;
};

// ─── Parse session_id from a chat-completion JSON body ──────────────────
// Returns empty string when session_id is absent or not a string (int/null/array).
// Checks extra_body.session_id first, then top-level session_id.
// PFlash: an explicit scorer query. The compressor scores context against
// this text instead of the last user message's tail, so a caller that knows
// its question (a benchmark, a RAG layer) can hand it over. Accepted at the top
// level or under extra_body, like session_id.
inline std::string parse_pflash_query_from_body(const json & body) {
    if (body.contains("extra_body")) {
        const auto & eb = body["extra_body"];
        if (eb.is_object() && eb.contains("pflash_query") && eb["pflash_query"].is_string()) {
            return eb["pflash_query"].get<std::string>();
        }
    }
    if (body.contains("pflash_query") && body["pflash_query"].is_string()) {
        return body["pflash_query"].get<std::string>();
    }
    return {};
}

// PFlash: literal strings that must survive compression. Each string must
// occur inside the boundary message's content; the compressor marks its last
// occurrence there as a mandatory retention span. Accepted at the top level
// or under extra_body, like pflash_query.
inline std::vector<std::string> parse_pflash_required_from_body(const json & body) {
    const json * field = nullptr;
    if (body.contains("extra_body")) {
        const auto & eb = body["extra_body"];
        if (eb.is_object() && eb.contains("pflash_required") && eb["pflash_required"].is_array()) {
            field = &eb["pflash_required"];
        }
    }
    if (!field && body.contains("pflash_required") && body["pflash_required"].is_array()) {
        field = &body["pflash_required"];
    }
    std::vector<std::string> result;
    if (!field) return result;
    for (const auto & entry : *field) {
        if (entry.is_string()) result.push_back(entry.get<std::string>());
    }
    return result;
}

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

}  // namespace luce::common
