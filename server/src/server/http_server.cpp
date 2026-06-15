// HTTP server implementation.
//
// Core infrastructure: socket listen/accept, client threads, HTTP parsing,
// job queue, worker thread with SSE streaming and disconnect detection.

#include "http_server.h"
#include "admission.h"
#include "sse_emitter.h"
#include "prompt_normalize.h"
#include "tool_hint.h"
#include "common/sha1.h"
#include "freeze_history.h"

#ifdef DFLASH_HAS_CURL
#include <curl/curl.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash::common {

// ─── piecewise keep-ratio curve ─────────────────────────────────────────

static float pflash_keep_ratio(const ServerConfig & cfg, int n_tokens) {
    if (cfg.pflash_curve.empty()) return cfg.pflash_keep_ratio;
    const auto & curve = cfg.pflash_curve;
    if (n_tokens <= curve.front().first) return curve.front().second;
    if (n_tokens >= curve.back().first)  return curve.back().second;
    for (size_t i = 0; i + 1 < curve.size(); ++i) {
        if (n_tokens <= curve[i + 1].first) {
            float t = (float)(n_tokens - curve[i].first) /
                      (float)(curve[i + 1].first - curve[i].first);
            return curve[i].second + t * (curve[i + 1].second - curve[i].second);
        }
    }
    return curve.back().second;
}

// ─── curl helpers for upstream proxy ─────────────────────────────────────
#ifdef DFLASH_HAS_CURL

struct CurlWriteCtx {
    int client_fd;
    bool streaming;
    bool first_chunk;
    bool chat_rewrite;   // rewrite completions → chat format
    std::string buffer;  // accumulates non-streaming response
    std::string response_id;
    std::string model;
};

static size_t curl_write_passthrough(char * ptr, size_t size, size_t nmemb, void * userdata) {
    size_t total = size * nmemb;
    auto * ctx = static_cast<CurlWriteCtx *>(userdata);
    if (ctx->streaming) {
        ::send(ctx->client_fd, ptr, total, MSG_NOSIGNAL);
    } else {
        ctx->buffer.append(ptr, total);
    }
    return total;
}

static size_t curl_write_rewrite(char * ptr, size_t size, size_t nmemb, void * userdata) {
    size_t total = size * nmemb;
    auto * ctx = static_cast<CurlWriteCtx *>(userdata);

    if (!ctx->streaming) {
        ctx->buffer.append(ptr, total);
        return total;
    }

    // Streaming: rewrite completions SSE chunks → chat completions format.
    ctx->buffer.append(ptr, total);
    std::string & buf = ctx->buffer;
    size_t pos = 0;
    while (true) {
        size_t nl = buf.find('\n', pos);
        if (nl == std::string::npos) break;
        std::string line = buf.substr(pos, nl - pos);
        pos = nl + 1;
        if (line.empty() || line == "\r") {
            std::string out = "\n";
            ::send(ctx->client_fd, out.data(), out.size(), MSG_NOSIGNAL);
            continue;
        }
        if (line.size() > 0 && line.back() == '\r') line.pop_back();
        if (line.rfind("data: ", 0) != 0) {
            line += "\n";
            ::send(ctx->client_fd, line.data(), line.size(), MSG_NOSIGNAL);
            continue;
        }
        std::string payload = line.substr(6);
        if (payload == "[DONE]") {
            std::string out = "data: [DONE]\n\n";
            ::send(ctx->client_fd, out.data(), out.size(), MSG_NOSIGNAL);
            continue;
        }
        try {
            auto j = json::parse(payload);
            j["object"] = "chat.completion.chunk";
            if (j.contains("choices") && j["choices"].is_array()) {
                for (auto & c : j["choices"]) {
                    json delta;
                    if (ctx->first_chunk) {
                        delta["role"] = "assistant";
                        ctx->first_chunk = false;
                    }
                    if (c.contains("text")) {
                        delta["content"] = c["text"];
                        c.erase("text");
                    }
                    c["delta"] = delta;
                    c.erase("index"); // re-add below
                    if (!c.contains("index")) c["index"] = 0;
                }
            }
            std::string out = "data: " + j.dump() + "\n\n";
            ::send(ctx->client_fd, out.data(), out.size(), MSG_NOSIGNAL);
        } catch (...) {
            std::string out = line + "\n";
            ::send(ctx->client_fd, out.data(), out.size(), MSG_NOSIGNAL);
        }
    }
    buf.erase(0, pos);
    return total;
}

static json rewrite_completions_to_chat(const json & comp_resp) {
    json chat_resp;
    chat_resp["id"] = comp_resp.value("id", "");
    chat_resp["object"] = "chat.completion";
    chat_resp["created"] = comp_resp.value("created", 0);
    chat_resp["model"] = comp_resp.value("model", "");
    if (comp_resp.contains("usage")) chat_resp["usage"] = comp_resp["usage"];
    json choices = json::array();
    if (comp_resp.contains("choices") && comp_resp["choices"].is_array()) {
        for (const auto & c : comp_resp["choices"]) {
            json choice;
            choice["index"] = c.value("index", 0);
            choice["finish_reason"] = c.value("finish_reason", "stop");
            json msg = {{"role", "assistant"}, {"content", c.value("text", "")}};
            choice["message"] = msg;
            choices.push_back(choice);
        }
    }
    chat_resp["choices"] = choices;
    return chat_resp;
}

static bool curl_forward(int client_fd, const std::string & url,
                         const std::string & api_key, const json & body,
                         bool streaming, bool rewrite_to_chat,
                         const std::string & response_id,
                         const std::string & model) {
    CURL * curl = curl_easy_init();
    if (!curl) return false;

    std::string body_str = body.dump();

    struct curl_slist * headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!api_key.empty()) {
        std::string auth = "Authorization: Bearer " + api_key;
        headers = curl_slist_append(headers, auth.c_str());
    }

    CurlWriteCtx ctx;
    ctx.client_fd = client_fd;
    ctx.streaming = streaming;
    ctx.first_chunk = true;
    ctx.chat_rewrite = rewrite_to_chat;
    ctx.response_id = response_id;
    ctx.model = model;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body_str.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);

    if (rewrite_to_chat) {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_rewrite);
    } else {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_passthrough);
    }

    if (streaming) {
        std::string sse_header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "\r\n";
        ::send(client_fd, sse_header.data(), sse_header.size(), MSG_NOSIGNAL);
    }

    CURLcode res = curl_easy_perform(curl);

    if (!streaming && res == CURLE_OK) {
        // Non-streaming: send accumulated response.
        if (rewrite_to_chat) {
            try {
                json resp = json::parse(ctx.buffer);
                json chat_resp = rewrite_completions_to_chat(resp);
                std::string out = chat_resp.dump();
                std::string http =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(out.size()) + "\r\n"
                    "\r\n" + out;
                ::send(client_fd, http.data(), http.size(), MSG_NOSIGNAL);
            } catch (...) {
                std::string http =
                    "HTTP/1.1 502 Bad Gateway\r\n"
                    "Content-Type: application/json\r\n"
                    "\r\n{\"error\":\"upstream response parse failed\"}";
                ::send(client_fd, http.data(), http.size(), MSG_NOSIGNAL);
            }
        } else {
            std::string http =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " + std::to_string(ctx.buffer.size()) + "\r\n"
                "\r\n" + ctx.buffer;
            ::send(client_fd, http.data(), http.size(), MSG_NOSIGNAL);
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return res == CURLE_OK;
}
#endif // DFLASH_HAS_CURL

// ─── /props constants ───────────────────────────────────────────────────
//
// SERVER_NAME / SERVER_VERSION mirror the Python server's identity strings
// so cross-server consumers (autotune, dashboards) see a stable
// `build_info` shape. Bump PROPS_SCHEMA on breaking changes only:
//   - field renamed
//   - field removed
//   - existing field's semantics change (units, nullability, type)
// Do NOT bump for additive changes (new fields, new sections).
static constexpr int  kPropsSchema  = 2;
static constexpr char kServerName[] = "luce-dflash";
#ifndef DFLASH_SERVER_VERSION
#define DFLASH_SERVER_VERSION "0.0.0+cpp"
#endif

// API endpoint registry served by /props. Keep in sync with the route
// handlers in handle_client() and route_request().
static const std::vector<std::string> kApiEndpoints = {
    "GET /health",
    "GET /props",
    "GET /status",
    "GET /status/events",
    "GET /status/json",
    "GET /v1/models",
    "POST /v1/chat/completions",
    "POST /v1/messages",
    "POST /v1/messages/count_tokens",
    "POST /v1/responses",
};

// ─── Utilities ──────────────────────────────────────────────────────────

static std::string generate_id(const char * prefix) {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%016llx",
                  prefix, (unsigned long long)counter.fetch_add(1));
    return buf;
}

// Logging helpers shared by route_request() / worker_loop(). Kept static
// (file-scope) so they don't leak into the public ABI; the chat lifecycle
// logs that use them are part of #270's request-tracing instrumentation.
static const char * api_format_name(ApiFormat format) {
    switch (format) {
    case ApiFormat::OPENAI_CHAT: return "chat";
    case ApiFormat::ANTHROPIC:   return "anthropic";
    case ApiFormat::RESPONSES:   return "responses";
    default:                     return "unknown";
    }
}

static size_t json_array_size(const json & value) {
    return value.is_array() ? value.size() : 0;
}

static bool env_flag_enabled(const char * name) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return false;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value != "0" && value != "false" && value != "no" &&
           value != "off";
}

static const json * find_tool_function(const json & tools,
                                       const std::string & name) {
    if (!tools.is_array() || name.empty()) return nullptr;
    for (const auto & tool : tools) {
        if (!tool.contains("function") || !tool["function"].is_object()) {
            continue;
        }
        const json & fn = tool["function"];
        if (fn.value("name", "") == name) return &fn;
    }
    return nullptr;
}

static std::string first_tool_parameter_name(const json & function_def) {
    const auto & params = function_def.value("parameters", json::object());
    if (params.contains("required") && params["required"].is_array()) {
        for (const auto & name : params["required"]) {
            if (name.is_string()) return name.get<std::string>();
        }
    }
    if (params.contains("properties") && params["properties"].is_object()) {
        for (const auto & item : params["properties"].items()) {
            return item.key();
        }
    }
    return "";
}

static const json * select_stall_recovery_function(const json & tools,
                                                   const json & tool_choice) {
    if (!tools.is_array() || tools.empty()) return nullptr;

    if (tool_choice.is_object() && tool_choice.contains("function") &&
        tool_choice["function"].is_object()) {
        const std::string forced_name =
            tool_choice["function"].value("name", "");
        // If the request forced a concrete function, recovery must honor it;
        // falling back to terminal here would synthesize invalid tool XML.
        return find_tool_function(tools, forced_name);
    }

    if (tool_choice.is_string() && tool_choice.get<std::string>() == "required" &&
        tools.size() == 1 && tools[0].contains("function") &&
        tools[0]["function"].is_object()) {
        return &tools[0]["function"];
    }

    if (const json * terminal = find_tool_function(tools, "terminal")) {
        return terminal;
    }
    if (tools.size() == 1 && tools[0].contains("function") &&
        tools[0]["function"].is_object()) {
        return &tools[0]["function"];
    }
    return nullptr;
}

static std::string build_stall_tool_prefix(const json & tools,
                                           const json & tool_choice) {
    const json * function_def =
        select_stall_recovery_function(tools, tool_choice);
    if (!function_def) return "\n<function=";

    const std::string name = function_def->value("name", "");
    if (name.empty()) return "\n<function=";

    std::string prefix = "\n<function=" + name + ">\n";
    std::string param = first_tool_parameter_name(*function_def);
    if (!param.empty()) {
        prefix += "<parameter=" + param + ">\n";
    }
    return prefix;
}

// Build the /props response body.
//
// Non-static so unit tests can call it directly (declared in http_server.h).
json build_props_body(const ServerConfig & config,
                      const PrefixCache & prefix_cache,
                      const ToolMemory & tool_memory) {
    // arch-gated capabilities (mirrors Python _capabilities()).
    const bool is_qwen = (config.arch.rfind("qwen", 0) == 0);
    const bool reasoning_supported = is_qwen;
    const bool speculative_supported = is_qwen;
    const bool tools_supported = is_qwen;

    auto pcs  = prefix_cache.stats();
    auto pcfs = prefix_cache.full_stats();
    auto tms  = tool_memory.stats();

    const bool pflash_enabled =
        (config.pflash_mode != ServerConfig::PflashMode::OFF);
    // speculative_mode reports the *active* path, not arch capability. A
    // Qwen-family model started without --ddtree has the capability but no
    // active speculative decode, so it must report "off" — otherwise clients
    // see `speculative_mode == "dflash"` paired with `speculative.enabled ==
    // false` and the two contradict (codex review feedback on 8d6ff04).
    std::string speculative_mode;
    if (pflash_enabled)                    speculative_mode = "pflash";
    else if (config.speculative_enabled)   speculative_mode = "dflash";
    else                                   speculative_mode = "off";

    // Spec §4.2: the five-tier vocabulary (low | medium | high | x-high | max)
    // all activate the phase-1 envelope. Advertise the full set when the
    // arch supports reasoning so clients can negotiate the higher tiers.
    json reasoning_efforts = json::array();
    if (reasoning_supported) {
        reasoning_efforts.push_back("low");
        reasoning_efforts.push_back("medium");
        reasoning_efforts.push_back("high");
        reasoning_efforts.push_back("x-high");
        reasoning_efforts.push_back("max");
    }

    json server = {
        {"name",         kServerName},
        {"version",      DFLASH_SERVER_VERSION},
        {"props_schema", kPropsSchema},
    };

    json pflash;
    if (!pflash_enabled) {
        pflash = {
            {"enabled",      false},
            {"mode",         "off"},
            {"threshold",    nullptr},
            {"keep_ratio",   nullptr},
            {"drafter_gguf", nullptr},
            {"skip_park",    nullptr},
            {"bsa_enabled",  nullptr},
            {"bsa_alpha",    nullptr},
            {"lm_head_fix",  nullptr},
            {"draft_residency", draft_residency_policy_name(config.draft_residency)},
        };
    } else {
        const char * bsa_env = std::getenv("DFLASH_FP_USE_BSA");
        const char * alpha_env = std::getenv("DFLASH_FP_ALPHA");
        const char * lmfix_env = std::getenv("DFLASH27B_LM_HEAD_FIX");
        json bsa_alpha = nullptr;
        if (alpha_env && *alpha_env) {
            try { bsa_alpha = std::stod(alpha_env); }
            catch (const std::exception &) { bsa_alpha = nullptr; }
        }
        std::string mode_str =
            (config.pflash_mode == ServerConfig::PflashMode::AUTO)   ? "auto"   :
            (config.pflash_mode == ServerConfig::PflashMode::ALWAYS) ? "always" : "off";
        pflash = {
            {"enabled",      true},
            {"mode",         mode_str},
            {"threshold",    config.pflash_threshold},
            {"keep_ratio",   config.pflash_keep_ratio},
            {"drafter_gguf", config.pflash_drafter_path.empty()
                              ? json(nullptr)
                              : json(config.pflash_drafter_path)},
            {"skip_park",    config.pflash_skip_park},
            {"bsa_enabled",  (bsa_env != nullptr && *bsa_env && std::strcmp(bsa_env, "0") != 0)},
            {"bsa_alpha",    bsa_alpha},
            {"lm_head_fix",  (lmfix_env != nullptr && *lmfix_env && std::strcmp(lmfix_env, "0") != 0)},
            {"draft_residency", draft_residency_policy_name(config.draft_residency)},
        };
    }

    // Reflect actual sampler defaults the server applies when a request
    // omits the field — these come from the loaded model card's sampling
    // section (spec §3.3), not from a hard-coded greedy fallback. Clients
    // that read /props to pick their sampling shape were getting greedy
    // here regardless of what the model card said, which caused gemma4
    // benchmarks to silently run at temp=0 (degenerate-decode collapse)
    // when the model card specifies temp=1.0/top_p=0.95/top_k=64.
    const auto & smp = config.sampler_defaults;
    json body = {
        {"default_generation_settings", {
            {"n_ctx",          config.max_ctx},
            {"temperature",    smp.has_temperature        ? smp.temperature        : 0.0f},
            {"top_p",          smp.has_top_p              ? smp.top_p              : 1.0f},
            {"top_k",          smp.has_top_k              ? smp.top_k              : 0},
            {"min_p",          smp.has_min_p              ? smp.min_p              : 0.0f},
            {"repeat_penalty", smp.has_repetition_penalty ? smp.repetition_penalty : 1.0f},
        }},
        {"model_alias", config.model_name},
        {"model_path",  config.model_path},
        {"build_info",  std::string(kServerName) + " v" DFLASH_SERVER_VERSION
                        " props_schema=" + std::to_string(kPropsSchema)},
        {"speculative_mode", speculative_mode},
        {"server", server},
        {"model", {
            {"arch",         config.arch},
            {"draft_path",   config.draft_path.empty() ? json(nullptr) : json(config.draft_path)},
            {"tokenizer_id", config.tokenizer_id.empty() ? json(nullptr) : json(config.tokenizer_id)},
        }},
        {"runtime", {
            {"backend",         config.runtime_backend.empty() ? "cuda" : config.runtime_backend},
            {"fa_window",       config.fa_window},
            {"kv_cache_k",      config.kv_cache_k},
            {"kv_cache_v",      config.kv_cache_v},
            {"lazy_draft",      config.lazy_draft},
            {"draft_residency", draft_residency_policy_name(config.draft_residency)},
            {"target_sharding", config.target_sharding},
            // Prefill chunk size (bargs.chunk). Surfaced so snapshot
            // tooling captures the full config — bench consumers
            // (dflash/scripts/bench_http_capability.py) read
            // /props.runtime wholesale into result.json.server_info.
            {"chunk",           config.chunk},
            // Device placement strings (e.g. "auto:0", "cuda:0"). Empty
            // string when no draft model is loaded.
            {"target_device",   config.target_device},
            {"draft_device",    config.draft_device.empty() ? json(nullptr) : json(config.draft_device)},
        }},
        {"reasoning", {
            {"supported",         reasoning_supported},
            {"default",           nullptr},
            {"supported_efforts", reasoning_efforts},
        }},
        // `model_card`: 1:1 with the on-disk sidecar JSON when one was
        // loaded; null when family fallback or hard fallback was used.
        // Validates against share/model_cards/_schema.json. The `source`
        // field here is the upstream model-card URL (authored in the
        // sidecar) — NOT a filepath. See spec §4.9.
        {"model_card", config.model_card_json.is_null()
                           ? json(nullptr)
                           : config.model_card_json},
        // `budget_envelope`: runtime-resolved values driving the
        // thinking-budget envelope. May differ from the authored card
        // values because of CLI overrides and max_ctx-based tier clamping
        // (spec §3.5). Always emitted regardless of model_card source.
        // See spec §4.2.
        {"budget_envelope", {
            {"model_card_source",       config.model_card_source_label},
            {"default_max_tokens",      config.default_max_tokens},
            {"hard_limit_reply_budget", config.hard_limit_reply_budget},
            {"think_max_tokens",        config.think_max_tokens},
            {"effort_tiers", {
                {"low",    config.effort_tiers.low},
                {"medium", config.effort_tiers.medium},
                {"high",   config.effort_tiers.high},
                {"x-high", config.effort_tiers.x_high},
                {"max",    config.effort_tiers.max},
            }},
        }},
        {"speculative", {
            {"enabled",       config.speculative_enabled},
            {"ddtree_budget", config.speculative_enabled
                                ? json(config.ddtree_budget) : json(nullptr)},
        }},
        {"sampling", {
            {"capabilities", {
                {"supports_temperature",        true},
                {"supports_top_p",              true},
                {"supports_top_k",              true},
                {"supports_frequency_penalty",  true},
                {"supports_seed",               true},
            }},
        }},
        {"pflash", pflash},
        {"prefix_cache", {
            {"capacity",      pcs.capacity},
            {"in_use",        pcs.in_use},
            {"lifetime_hits", pcs.lifetime_hits},
        }},
        {"full_cache", {
            {"enabled",       pcfs.enabled},
            {"capacity",      pcfs.capacity},
            {"in_use",        pcfs.in_use},
            {"disk_bytes",    pcfs.disk_bytes},
            {"lifetime_hits", pcfs.lifetime_hits},
            {"disk_policy",   disk_prefix_cache_policy_name(config.disk_cache_policy)},
        }},
        {"tool_replay", {
            {"max_entries",     tms.max_entries},
            {"max_bytes",       tms.max_bytes},
            {"current_entries", tms.current_entries},
            {"current_bytes",   tms.current_bytes},
        }},
        // The C++ daemon is linked in-process; if /props is responding,
        // the daemon is alive by construction.
        {"daemon", {{"alive", true}}},
        {"api", {{"endpoints", kApiEndpoints}}},
        // Capability flags surfaced for clients that don't want to crack
        // open `reasoning` / `speculative` / etc. — matches the Python
        // server's _capabilities() helper.
        {"capabilities", {
            {"reasoning_supported",   reasoning_supported},
            {"speculative_supported", speculative_supported},
            {"tools_supported",       tools_supported},
        }},
    };
    return body;
}

// Normalize Anthropic's `system` field (top-level on /v1/messages and
// /v1/messages/count_tokens) into a leading `{role:"system", content:...}`
// entry on `messages`. Accepts either a flat string or an array of typed
// blocks (`[{type:"text", text:"..."}]`), and strips any
// `x-anthropic-billing-header:`-prefixed block injected by Claude Code so
// it never reaches the model or the token counter.
//
// Side-effect: prepends a system message to `messages` when the body has
// a non-empty `system` field after billing-header filtering. No-op
// otherwise. Both endpoints call this with identical semantics — having
// one helper guarantees token counting and generation can't drift.
static void normalize_anthropic_system(const json & body, json & messages) {
    if (!body.contains("system")) return;
    // Delegate strip to the pure fn; insert as system message.
    std::string text = dflash::common::normalize_system_for_cache(body["system"]);
    if (!text.empty()) {
        json sys_msg = {{"role", "system"}, {"content", text}};
        messages.insert(messages.begin(), sys_msg);
    }
}

json parse_responses_arguments(const json & item) {
    if (!item.contains("arguments")) return json::object();
    const auto & arguments = item["arguments"];
    if (arguments.is_object()) return arguments;
    if (arguments.is_string()) {
        try {
            return json::parse(arguments.get<std::string>());
        } catch (const std::exception &) {
            return json::object();
        }
    }
    return json::object();
}

std::string render_tool_call_xml(const std::string & name, const json & arguments) {
    std::string out = "<function=" + name + ">\n";
    if (arguments.is_object()) {
        for (const auto & [key, value] : arguments.items()) {
            out += "<parameter=" + key + ">\n";
            out += value.is_string() ? value.get<std::string>() : value.dump();
            out += "\n</parameter>\n";
        }
    }
    out += "</function>\n";
    return out;
}

std::vector<ChatMessage> normalize_chat_messages(
    const json & messages,
    ApiFormat format,
    ToolMemory & tool_memory) {
    std::vector<ChatMessage> chat_msgs;
    std::vector<std::string> system_parts;

    if (messages.is_array()) {
        for (const auto & m : messages) {
            if (format == ApiFormat::RESPONSES && m.is_object()) {
                std::string item_type = m.value("type", "message");
                if (item_type == "function_call") {
                    std::string call_id = m.value("call_id", m.value("id", ""));
                    std::string raw;
                    if (!call_id.empty()) {
                        raw = tool_memory.lookup({call_id});
                    }
                    if (raw.empty()) {
                        raw = render_tool_call_xml(m.value("name", ""),
                                                   parse_responses_arguments(m));
                    }
                    chat_msgs.push_back({"assistant", raw});
                    continue;
                }
                if (item_type == "function_call_output") {
                    std::string output;
                    if (m.contains("output") && m["output"].is_string()) {
                        output = m["output"].get<std::string>();
                    } else if (m.contains("output")) {
                        output = m["output"].dump();
                    }
                    chat_msgs.push_back({"tool", output,
                                         m.value("call_id", m.value("id", ""))});
                    continue;
                }
            }

            ChatMessage cm;
            cm.role = m.value("role", "user");

            bool replayed = false;
            if (cm.role == "assistant" && m.contains("tool_calls") &&
                m["tool_calls"].is_array() && !m["tool_calls"].empty()) {
                std::vector<std::string> call_ids;
                for (const auto & tc : m["tool_calls"]) {
                    std::string id = tc.value("id", "");
                    if (!id.empty()) call_ids.push_back(id);
                }
                std::string raw = tool_memory.lookup(call_ids);
                if (!raw.empty()) {
                    cm.content = raw;
                    replayed = true;
                }
            }

            if (!replayed) {
                if (m.contains("content") && m["content"].is_string()) {
                    cm.content = m["content"].get<std::string>();
                } else if (m.contains("content") && m["content"].is_array()) {
                    for (const auto & part : m["content"]) {
                        std::string ptype = part.value("type", "");
                        if (ptype == "text" || ptype == "input_text" ||
                            ptype == "output_text") {
                            cm.content += part.value("text", "");
                        }
                    }
                }
            }

            if (format == ApiFormat::RESPONSES &&
                (cm.role == "system" || cm.role == "developer")) {
                system_parts.push_back(cm.content);
            } else {
                chat_msgs.push_back(std::move(cm));
            }
        }
    } else if (messages.is_string()) {
        chat_msgs.push_back({"user", messages.get<std::string>()});
    }

    if (!system_parts.empty()) {
        std::string merged_system;
        for (size_t i = 0; i < system_parts.size(); i++) {
            if (i) merged_system += "\n\n";
            merged_system += system_parts[i];
        }
        chat_msgs.insert(chat_msgs.begin(), {"system", merged_system});
    }

    return chat_msgs;
}

// ─── Disk-cache identity salt ───────────────────────────────────────────
// Compute a 16-byte salt from inputs that affect KV cache validity:
//   model path + stat(size + mtime)  [covers rope/yarn — GGUF-derived],
//   max_ctx, and sha1(chat_template_src).
// Returns all-zeroes if model_path is empty (back-compat / disk disabled).
static std::array<uint8_t, 16> compute_disk_cache_salt(const ServerConfig & cfg) {
    std::array<uint8_t, 16> salt{};
    if (cfg.model_path.empty()) return salt;

    const std::string & path = cfg.model_path;
    struct stat st{};
    int64_t file_size  = 0;
    int64_t file_mtime = 0;
    if (::stat(path.c_str(), &st) == 0) {
        file_size  = (int64_t)st.st_size;
        file_mtime = (int64_t)st.st_mtime;
    } else {
        std::fprintf(stderr, "[disk-cache] salt: stat(%s) failed — path-only fingerprint\n",
                     path.c_str());
    }

    // Hash chat_template_src separately (can be large; fold as digest).
    uint8_t tmpl_digest[20] = {};
    sha1_hash(cfg.chat_template_src.data(), cfg.chat_template_src.size(), tmpl_digest);

    // Serialization: path_len(4) + path + file_size(8) + file_mtime(8) + max_ctx(4) + tmpl_digest(20).
    std::vector<uint8_t> buf;
    uint32_t plen = (uint32_t)path.size();
    buf.insert(buf.end(), (uint8_t *)&plen,        (uint8_t *)&plen        + 4);
    buf.insert(buf.end(), (uint8_t *)path.data(),  (uint8_t *)path.data()  + path.size());
    buf.insert(buf.end(), (uint8_t *)&file_size,   (uint8_t *)&file_size   + 8);
    buf.insert(buf.end(), (uint8_t *)&file_mtime,  (uint8_t *)&file_mtime  + 8);
    int32_t mc = (int32_t)cfg.max_ctx;
    buf.insert(buf.end(), (uint8_t *)&mc,          (uint8_t *)&mc          + 4);
    buf.insert(buf.end(), tmpl_digest, tmpl_digest + 20);

    uint8_t digest[20];
    sha1_hash(buf.data(), buf.size(), digest);
    std::memcpy(salt.data(), digest, 16);
    return salt;
}

// ─── HttpServer ─────────────────────────────────────────────────────────

HttpServer::HttpServer(ModelBackend & backend,
                       Tokenizer & tokenizer,
                       const ServerConfig & config)
    : backend_(backend)
    , tokenizer_(tokenizer)
    , config_(config)
    , chat_format_(ChatFormat::QWEN3)  // default, overridden by arch
    , prefix_cache_(config.prefix_cache_cap, tokenizer)
    , disk_cache_({config.disk_cache_dir,
                   config.disk_cache_budget_mb * (size_t)(1024 * 1024),
                   config.disk_cache_min_tokens,
                   config.disk_cache_continued_interval,
                   config.disk_cache_cold_max_tokens}, backend)
{
    #ifdef DFLASH_HAS_CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    #endif
    // Fold model+config identity into the layout fingerprint BEFORE init()
    // so compute_layout_id sees it on every learn/verify call. Prevents stale
    // KV hits when the server restarts over the same --kv-cache-dir with a
    // different model, max_ctx, or chat_template (gemma4 ↔ qwen3.6, etc.).
    if (!disk_cache_.disabled()) {
        disk_cache_.set_identity_salt(compute_disk_cache_salt(config));
    }
    disk_cache_.init();
    status_html_path_ = resolve_status_html();
}

// Resolve path to share/status.html at startup.
std::string HttpServer::resolve_status_html() {
    // 1. DFLASH_SHARE_DIR env var
    if (const char * dir = std::getenv("DFLASH_SHARE_DIR")) {
        std::string path = std::string(dir) + "/status.html";
        struct stat st;
        if (::stat(path.c_str(), &st) == 0) return path;
    }
    // 2. share/ relative to /proc/self/exe (build dir or installed prefix)
    char exe_buf[1024] = {};
    ssize_t len = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    if (len > 0) {
        exe_buf[len] = '\0';
        std::string exe_dir(exe_buf);
        auto slash = exe_dir.rfind('/');
        if (slash != std::string::npos) {
            exe_dir = exe_dir.substr(0, slash);
            // 2a. <exe_dir>/share/status.html  (build directory layout)
            {
                std::string path = exe_dir + "/share/status.html";
                struct stat st;
                if (::stat(path.c_str(), &st) == 0) return path;
            }
            // 2b. <exe_dir>/../share/status.html  (installed prefix layout)
            {
                std::string path = exe_dir + "/../share/status.html";
                struct stat st;
                if (::stat(path.c_str(), &st) == 0) return path;
            }
        }
    }
    // 3. ./share/status.html (development)
    {
        struct stat st;
        if (::stat("share/status.html", &st) == 0) return "share/status.html";
    }
    return {};
}

// Send data to an SSE client fd with a short (1s) timeout to avoid stalling
// the inference worker. Returns false if the send fails or times out.
static bool sse_try_send(int fd, const void * data, size_t len) {
    const char * p = static_cast<const char *>(data);
    size_t sent = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (sent < len) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) return false;

        struct pollfd pfd = {fd, POLLOUT, 0};
        int ret;
        do {
            ret = poll(&pfd, 1, static_cast<int>(std::min(remaining, (long)50)));
        } while (ret < 0 && errno == EINTR);
        if (ret < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
        if (ret == 0) continue;

        ssize_t n = ::send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return false;
        }
        sent += n;
    }
    return true;
}

// Broadcast current status as SSE event to all connected /status/events clients.
void HttpServer::broadcast_status() {
    std::string event = status_.to_sse_event();
    std::lock_guard<std::mutex> lk(sse_mu_);
    std::vector<int> dead;
    for (int fd : sse_fds_) {
        if (!sse_try_send(fd, event.data(), event.size())) {
            dead.push_back(fd);
        }
    }
    for (int fd : dead) {
        ::close(fd);
        sse_fds_.erase(std::remove(sse_fds_.begin(), sse_fds_.end(), fd),
                       sse_fds_.end());
    }
}

// Broadcast a token text delta as an incremental SSE event.
void HttpServer::broadcast_token(const std::string & text) {
    // Token text may contain incomplete UTF-8 (tokens can split multi-byte
    // codepoints). Manually build the SSE payload with json string escaping
    // that replaces invalid UTF-8 with U+FFFD instead of throwing.
    json j;
    j["text"] = text;
    std::string event = "event: token\ndata: " +
        j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    std::lock_guard<std::mutex> lk(sse_mu_);
    std::vector<int> dead;
    for (int fd : sse_fds_) {
        if (!sse_try_send(fd, event.data(), event.size())) {
            dead.push_back(fd);
        }
    }
    for (int fd : dead) {
        ::close(fd);
        sse_fds_.erase(std::remove(sse_fds_.begin(), sse_fds_.end(), fd),
                       sse_fds_.end());
    }
}

// Send an SSE comment as a heartbeat to detect disconnected clients when idle.
// Uses non-blocking sends to avoid stalling the worker thread on slow clients.
void HttpServer::sse_heartbeat() {
    static const char ping[] = ":heartbeat\n\n";
    std::lock_guard<std::mutex> lk(sse_mu_);
    std::vector<int> dead;
    for (int fd : sse_fds_) {
        // Non-blocking send: if the socket buffer can't accept 12 bytes
        // immediately, the client is too far behind — treat as dead.
        ssize_t n = ::send(fd, ping, sizeof(ping) - 1, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n <= 0) {
            dead.push_back(fd);
        }
    }
    for (int fd : dead) {
        ::close(fd);
        sse_fds_.erase(std::remove(sse_fds_.begin(), sse_fds_.end(), fd),
                       sse_fds_.end());
    }
}

HttpServer::~HttpServer() {
    shutdown();
    #ifdef DFLASH_HAS_CURL
    curl_global_cleanup();
    #endif
}

void HttpServer::shutdown() {
    // Signal worker and accept loop to stop.
    stopping_.store(true);
    queue_cv_.notify_all();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    // Close SSE client connections.
    {
        std::lock_guard<std::mutex> lk(sse_mu_);
        for (int fd : sse_fds_) ::close(fd);
        sse_fds_.clear();
    }

    // Drain any pending jobs.
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        while (queue_head_) {
            ServerJob * j = queue_head_;
            queue_head_ = j->next;
            j->next = nullptr;
            std::lock_guard<std::mutex> jlk(j->mu);
            j->done = true;
            j->cv.notify_one();
        }
        queue_tail_ = nullptr;
    }

    // Shutdown save: persist all tracked snapshot slots to disk.
    // Safe to access slot_tokens_ without locking — worker is joined.
    if (!disk_cache_.disabled() && !slot_tokens_.empty()) {
        std::fprintf(stderr, "[disk-cache] shutdown: saving %zu tracked slots\n",
                     slot_tokens_.size());
        for (auto & [slot, tokens] : slot_tokens_) {
            if (backend_.snapshot_used(slot)) {
                disk_cache_.learn_layout(slot);
                disk_cache_.save(slot, tokens);
            }
        }
        slot_tokens_.clear();
    }
}

int HttpServer::run() {
    // Ignore SIGPIPE so send() returns EPIPE instead of killing the process.
    signal(SIGPIPE, SIG_IGN);

    // Create listen socket.
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        std::fprintf(stderr, "[server] socket() failed: %s\n", strerror(errno));
        return 1;
    }

    int yes = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)config_.port);
    if (inet_pton(AF_INET, config_.host.c_str(), &sa.sin_addr) != 1) {
        std::fprintf(stderr, "[server] invalid host address: %s\n", config_.host.c_str());
        ::close(listen_fd_);
        listen_fd_ = -1;
        return 1;
    }

    if (bind(listen_fd_, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        std::fprintf(stderr, "[server] bind(%s:%d) failed: %s\n",
                     config_.host.c_str(), config_.port, strerror(errno));
        ::close(listen_fd_);
        listen_fd_ = -1;
        return 1;
    }

    if (listen(listen_fd_, 128) < 0) {
        std::fprintf(stderr, "[server] listen() failed: %s\n", strerror(errno));
        ::close(listen_fd_);
        listen_fd_ = -1;
        return 1;
    }

    // Non-blocking listen socket so the accept loop polls stopping_ on a short
    // timeout. This guarantees the loop exits on SIGTERM/SIGINT regardless of
    // which thread the signal handler runs on (it only sets the atomic flag).
    {
        int fl = fcntl(listen_fd_, F_GETFL, 0);
        if (fl < 0 || fcntl(listen_fd_, F_SETFL, fl | O_NONBLOCK) < 0) {
            std::fprintf(stderr, "[server] fcntl(O_NONBLOCK) failed: %s\n", strerror(errno));
            ::close(listen_fd_);
            listen_fd_ = -1;
            return 1;
        }
    }

    std::fprintf(stderr, "[server] listening on http://%s:%d\n",
                 config_.host.c_str(), config_.port);

    // Start worker thread.
    worker_thread_ = std::thread([this]() { worker_loop(); });

    // Accept loop.
    while (!stopping_.load()) {
        struct pollfd pfd{listen_fd_, POLLIN, 0};
        int pr = poll(&pfd, 1, 200 /* ms */);
        if (pr <= 0) {
            // 0 = timeout (re-check stopping_); <0 with EINTR = signal. Both loop.
            if (pr < 0 && errno != EINTR) {
                std::fprintf(stderr, "[server] poll() error: %s\n", strerror(errno));
            }
            continue;
        }

        struct sockaddr_in client_sa{};
        socklen_t client_len = sizeof(client_sa);
        int client_fd = accept(listen_fd_, (struct sockaddr *)&client_sa, &client_len);
        if (client_fd < 0) {
            if (stopping_.load()) break;
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            std::fprintf(stderr, "[server] accept() error: %s\n", strerror(errno));
            continue;
        }

        // Disable Nagle for low-latency SSE streaming.
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        // Spawn client thread (detached — client_main owns the fd).
        active_clients_.fetch_add(1);
        std::thread([this, client_fd]() {
            handle_client(client_fd);
            if (active_clients_.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(clients_mu_);
                clients_cv_.notify_all();
            }
        }).detach();
    }

    // Wake the worker thread so it can observe stopping_ and exit.
    queue_cv_.notify_all();

    // Wait for client threads to drain, but bound it: a client mid-stream (long
    // SSE generation) must not hold the process resident on shutdown. After the
    // grace period we proceed — detached client threads are torn down on exit.
    {
        std::unique_lock<std::mutex> lk(clients_mu_);
        bool drained = clients_cv_.wait_for(
            lk, std::chrono::seconds(5),
            [this]() { return active_clients_.load() == 0; });
        if (!drained) {
            std::fprintf(stderr,
                         "[server] shutdown: %d client thread(s) still active "
                         "after grace period, exiting anyway\n",
                         active_clients_.load());
        }
    }

    // Wait for worker to finish.
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    // Persist disk cache (worker joined — no race on slot_tokens_).
    if (!disk_cache_.disabled() && !slot_tokens_.empty()) {
        std::fprintf(stderr, "[disk-cache] shutdown: saving %zu tracked slots\n",
                     slot_tokens_.size());
        for (auto & [slot, tokens] : slot_tokens_) {
            if (backend_.snapshot_used(slot)) {
                disk_cache_.learn_layout(slot);
                disk_cache_.save(slot, tokens);
            }
        }
        slot_tokens_.clear();
    }

    return 0;
}

// ─── Client thread ──────────────────────────────────────────────────────

void HttpServer::handle_client(int fd) {
    HttpRequest hr;
    if (!read_http_request(fd, hr)) {
        send_error(fd, 400, "bad HTTP request");
        ::close(fd);
        return;
    }

    // CORS preflight.
    if (hr.method == "OPTIONS") {
        send_response(fd, 204, "", "");
        ::close(fd);
        return;
    }

    // Health check.
    if (hr.method == "GET" && (hr.path == "/health" || hr.path == "/")) {
        send_response(fd, 200, "application/json", "{\"status\":\"ok\"}\n");
        ::close(fd);
        return;
    }

    // Introspection: server config + cache stats + arch + capabilities.
    if (hr.method == "GET" && hr.path == "/props") {
        json body = build_props_body(config_, prefix_cache_, tool_memory_);
        send_response(fd, 200, "application/json", body.dump() + "\n");
        ::close(fd);
        return;
    }

    // Status page: serve HTML file from disk.
    if (hr.method == "GET" && hr.path == "/status") {
        if (status_html_path_.empty()) {
            send_error(fd, 404,
                "status.html not found. Set DFLASH_SHARE_DIR or place it in share/status.html");
            ::close(fd);
            return;
        }
        std::ifstream ifs(status_html_path_);
        if (!ifs.is_open()) {
            send_error(fd, 500, "failed to open status.html");
            ::close(fd);
            return;
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        send_response(fd, 200, "text/html; charset=utf-8", oss.str());
        ::close(fd);
        return;
    }

    // Status JSON snapshot (for non-SSE clients / debugging).
    if (hr.method == "GET" && hr.path == "/status/json") {
        send_response(fd, 200, "application/json",
            status_.to_json().dump(-1, ' ', false, json::error_handler_t::replace) + "\n");
        ::close(fd);
        return;
    }

    // Status SSE stream: hold connection open and push updates.
    if (hr.method == "GET" && hr.path == "/status/events") {
        // Send SSE headers.
        const char * headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        if (!send_all(fd, headers, std::strlen(headers))) {
            ::close(fd);
            return;
        }
        // Send initial state immediately.
        std::string initial = status_.to_sse_event();
        send_all(fd, initial.data(), initial.size());
        // Register for future broadcasts. The fd is NOT closed here — it stays
        // open until the client disconnects (detected on next broadcast send).
        {
            std::lock_guard<std::mutex> lk(sse_mu_);
            sse_fds_.push_back(fd);
        }
        return;  // Do NOT close fd — it's now owned by the SSE broadcast loop.
    }

    // Models endpoint.
    if (hr.method == "GET" && hr.path == "/v1/models") {
        // Codex sends ?client_version= — serve the Codex-specific schema.
        if (hr.query.find("client_version") != std::string::npos) {
            json codex_models = {
                {"models", json::array({
                    {{"slug", config_.model_name},
                     {"display_name", config_.model_name},
                     {"description", "Local DFlash speculative-decoding server"},
                     {"default_reasoning_level", "low"},
                     // Spec §4.2: every tier activates the phase-1 envelope;
                     // the difference is the budget cap selected from the
                     // model card's effort_tiers. Descriptions surface the
                     // resolved cap so clients can pick a tier purposefully.
                     {"supported_reasoning_levels", json::array({
                         {{"effort", "low"},
                          {"description", "Phase-1 budget at the model card's low tier ("
                                          + std::to_string(config_.effort_tiers.low)
                                          + " tokens)"}},
                         {{"effort", "medium"},
                          {"description", "Phase-1 budget at the model card's medium tier ("
                                          + std::to_string(config_.effort_tiers.medium)
                                          + " tokens)"}},
                         {{"effort", "high"},
                          {"description", "Phase-1 budget at the model card's standard recommendation ("
                                          + std::to_string(config_.effort_tiers.high)
                                          + " tokens)"}},
                         {{"effort", "x-high"},
                          {"description", "Phase-1 budget between high and the complex-problem ceiling ("
                                          + std::to_string(config_.effort_tiers.x_high)
                                          + " tokens)"}},
                         {{"effort", "max"},
                          {"description", "Phase-1 budget at the model card's complex-problem ceiling ("
                                          + std::to_string(config_.effort_tiers.max)
                                          + " tokens)"}},
                     })},
                     {"shell_type", "shell_command"},
                     {"visibility", "list"},
                     {"supported_in_api", true},
                     {"priority", 1},
                     {"context_window", config_.max_ctx},
                     {"supports_reasoning_summaries", false},
                     {"supports_parallel_tool_calls", false}}
                })}
            };
            send_response(fd, 200, "application/json", codex_models.dump() + "\n");
            ::close(fd);
            return;
        }
        json models = {
            {"object", "list"},
            {"data", json::array({
                {{"id", config_.model_name},
                 {"object", "model"},
                 {"owned_by", "dflash"},
                 {"created", 1700000000},
                 {"context_length", config_.max_ctx},
                 {"max_context_length", config_.max_ctx}}
            })}
        };
        send_response(fd, 200, "application/json", models.dump() + "\n");
        ::close(fd);
        return;
    }

    // Route POST endpoints.
    if (!route_request(fd, hr)) {
        send_error(fd, 404, "unknown endpoint");
    }
    ::close(fd);
}

bool HttpServer::route_request(int fd, const HttpRequest & hr) {
    if (hr.method != "POST") return false;

    std::fprintf(stderr, "[server] request path=%s body_bytes=%zu\n",
                 hr.path.c_str(), hr.body.size());

    ParsedRequest req;
    std::string err;

    try {
        json body = json::parse(hr.body);
        req.raw_body = body;

        // Common fields.
        req.stream = body.value("stream", false);
        req.model = body.value("model", config_.model_name);
        req.disk_cache_policy = config_.disk_cache_policy;
        // Default when client omits all three: use --default-max-tokens
        // (16000, matches ds4_eval.c). Codex review flagged that
        // --default-max-tokens was previously a dead flag because the
        // parser read config_.max_tokens (legacy 4096) instead. The new
        // default protects thinking-budget requests that omit max_tokens
        // from being capped at 4096 — thinking alone can consume that,
        // leaving no headroom for the visible reply.
        req.max_output = body.value("max_tokens",
                         body.value("max_output_tokens",
                         body.value("max_completion_tokens", config_.default_max_tokens)));
        // Spec §4.4: clamp request max_tokens to --default-max-tokens.
        if (req.max_output > config_.default_max_tokens) {
            std::fprintf(stderr,
                "[server] max_tokens=%d clamped to default_max_tokens=%d\n",
                req.max_output, config_.default_max_tokens);
            req.max_output = config_.default_max_tokens;
        }

        // Sampler parameters. When the request omits a value, fall back to
        // the model card's sampling defaults (spec §3.3); when the card
        // doesn't supply one either, use the hard-coded default.
        const auto & sd = config_.sampler_defaults;
        req.sampler.temp = body.value("temperature",
                                      sd.has_temperature ? sd.temperature : 0.0f);
        req.sampler.top_p = body.value("top_p",
                                       sd.has_top_p ? sd.top_p : 1.0f);
        req.sampler.top_k = body.value("top_k",
                                       sd.has_top_k ? sd.top_k : 0);
        if (body.contains("seed")) {
            req.sampler.seed = body["seed"].get<uint64_t>();
        }

        // OpenAI-style additive penalties.
        req.sampler.freq_pen = body.value("frequency_penalty", 0.0f);
        req.sampler.pres_pen = body.value("presence_penalty",
                                          sd.has_presence_penalty ? sd.presence_penalty : 0.0f);

        // HuggingFace-style multiplicative repetition penalty (also used by
        // vLLM, llama.cpp, etc.). Accepts both "repetition_penalty" and
        // the shorter "rep_pen" for daemon compatibility.
        req.sampler.rep_pen = body.value("repetition_penalty",
                              body.value("rep_pen",
                                  sd.has_repetition_penalty ? sd.repetition_penalty : 1.0f));
        if (body.contains("rep_window")) {
            req.sampler.rep_window = body["rep_window"].get<int>();
        }

        // Tools.
        if (body.contains("tools")) {
            req.tools = body["tools"];
        }
        // Tool choice constraint for hint generation.
        if (body.contains("tool_choice")) {
            req.tool_choice = body["tool_choice"];
        }

        if (body.contains("prefix_cache") && body["prefix_cache"].is_object()) {
            const auto & pc = body["prefix_cache"];
            if (pc.contains("scope") && pc["scope"].is_string()) {
                if (!apply_request_scope_override(req.disk_cache_policy,
                                                  pc["scope"].get<std::string>())) {
                    send_error(fd, 400,
                        "prefix_cache.scope must be off, full, auto, auto:<window>, or a positive token count");
                    return true;
                }
            }
            if (pc.contains("window") && pc["window"].is_number_integer()) {
                const int window = pc["window"].get<int>();
                if (window <= 0 || window > 1000000) {
                    send_error(fd, 400, "prefix_cache.window must be a positive integer");
                    return true;
                }
                req.disk_cache_policy.auto_window = window;
            }
        }

        // Stop sequences — OpenAI uses "stop" (string or array), Anthropic uses "stop_sequences" (array).
        if (body.contains("stop")) {
            auto & stop = body["stop"];
            if (stop.is_string()) {
                std::string s = stop.get<std::string>();
                if (!s.empty()) req.stop_sequences.push_back(s);
            } else if (stop.is_array()) {
                for (const auto & item : stop) {
                    if (item.is_string()) {
                        std::string s = item.get<std::string>();
                        if (!s.empty()) req.stop_sequences.push_back(s);
                    }
                }
            }
        }
        if (body.contains("stop_sequences") && body["stop_sequences"].is_array()) {
            for (const auto & item : body["stop_sequences"]) {
                if (item.is_string()) {
                    std::string s = item.get<std::string>();
                    if (!s.empty()) req.stop_sequences.push_back(s);
                }
            }
        }

        // count_tokens shares Anthropic's message parsing; flag so we
        // short-circuit before enqueueing the generation job.
        bool count_tokens_only = false;

        if (hr.path == "/v1/chat/completions") {
            req.format = ApiFormat::OPENAI_CHAT;
            req.response_id = generate_id("chatcmpl");
            req.messages = body["messages"];
            // Strip volatile billing header from messages[0] (OpenAI system).
            if (req.messages.is_array() && !req.messages.empty()) {
                auto & m0 = req.messages[0];
                if (m0.is_object() && m0.value("role", "") == "system" &&
                    m0.contains("content") && m0["content"].is_string()) {
                    m0["content"] = dflash::common::normalize_system_for_cache(req.messages);
                }
            }
        } else if (hr.path == "/v1/messages/count_tokens") {
            req.format = ApiFormat::ANTHROPIC;
            req.response_id = generate_id("count");
            req.messages = body.value("messages", json::array());
            normalize_anthropic_system(body, req.messages);
            count_tokens_only = true;
        } else if (hr.path == "/v1/messages") {
            req.format = ApiFormat::ANTHROPIC;
            req.response_id = generate_id("msg");
            req.messages = body["messages"];
            normalize_anthropic_system(body, req.messages);
        } else if (hr.path == "/v1/responses") {
            req.format = ApiFormat::RESPONSES;
            req.response_id = generate_id("resp");
            // Responses API uses "input" instead of "messages".
            if (body.contains("input")) {
                req.messages = body["input"];
            }
            if (body.contains("instructions")) {
                // Strip billing header from codex instructions before hashing.
                std::string inst = dflash::common::normalize_system_for_cache(body["instructions"]);
                json sys_msg = {{"role", "system"}, {"content", inst}};
                if (req.messages.is_array()) {
                    req.messages.insert(req.messages.begin(), sys_msg);
                } else {
                    req.messages = json::array({sys_msg, {{"role", "user"}, {"content", body["input"]}}});
                }
            }
        } else {
            return false;
        }

        // Render messages to text and tokenize.
        std::vector<ChatMessage> chat_msgs =
            normalize_chat_messages(req.messages, req.format, tool_memory_);

        // Determine thinking mode BEFORE rendering so the template can inject
        // the <think>\n\n</think>\n\n block when thinking is disabled.
        // Default: thinking OFF (Qwen3.6 thinking wrecks
        // DFlash acceptance rates; clients opt in explicitly).
        bool enable_thinking = false;

        // Track which fields the request explicitly set, so we can apply
        // §4.3 combined precedence: thinking.budget_tokens beats
        // reasoning.effort for the phase-1 cap, but the effort tier still
        // selects defaults for any unspecified thinking.* field.
        int  request_budget_tokens   = -1;  // from thinking.budget_tokens
        int  request_reply_budget    = -1;  // from thinking.reply_budget
        int  effort_phase1_cap       = -1;  // from reasoning.effort lookup
        bool effort_set              = false;

        // OpenAI Responses API: "reasoning" field. Spec §4.2.
        if (body.contains("reasoning")) {
            auto & r = body["reasoning"];
            if (r.contains("effort")) {
                std::string effort = r.value("effort", "high");
                // Five-tier vocabulary (spec §4.2). Unknown → high.
                int tier_value = config_.effort_tiers.high;
                if      (effort == "low")    tier_value = config_.effort_tiers.low;
                else if (effort == "medium") tier_value = config_.effort_tiers.medium;
                else if (effort == "high")   tier_value = config_.effort_tiers.high;
                else if (effort == "x-high") tier_value = config_.effort_tiers.x_high;
                else if (effort == "max")    tier_value = config_.effort_tiers.max;
                // else: unknown tier → fall back to high (no error).

                effort_phase1_cap = tier_value;
                effort_set = true;
                enable_thinking = true;
                // Spec §4.2: reasoning.effort activates the budget envelope.
                req.thinking_opt_in = true;
            } else {
                enable_thinking = true;
            }
        }
        // Anthropic-style: "thinking" field. Presence-as-opt-in: any
        // request that sends this field has opted in to the thinking-budget
        // envelope (and will see a `finish_details` block on the response).
        if (body.contains("thinking")) {
            auto & th = body["thinking"];
            if (th.contains("type")) {
                std::string type = th.value("type", "");
                enable_thinking = (type == "enabled");
                req.thinking_opt_in = (type == "enabled");
            }
            // Spec §4.1 fields. Clamp to server ceilings (§4.4).
            if (th.contains("budget_tokens") && th["budget_tokens"].is_number_integer()) {
                request_budget_tokens = th["budget_tokens"].get<int>();
            }
            if (th.contains("reply_budget") && th["reply_budget"].is_number_integer()) {
                request_reply_budget = th["reply_budget"].get<int>();
            }
        }
        // Direct: chat_template_kwargs.enable_thinking
        if (body.contains("chat_template_kwargs")) {
            auto & kwargs = body["chat_template_kwargs"];
            if (kwargs.contains("enable_thinking")) {
                enable_thinking = kwargs["enable_thinking"].get<bool>();
            }
        }

        req.thinking_enabled = enable_thinking;

        // Spec §4.3 combined precedence + §4.4 clamping.
        // Phase-1 cap:
        //   thinking.budget_tokens (if set) wins over reasoning.effort.
        //   Either is clamped to think_max_tokens.
        if (request_budget_tokens >= 0) {
            int eff = std::min(request_budget_tokens, config_.think_max_tokens);
            if (request_budget_tokens > config_.think_max_tokens) {
                std::fprintf(stderr,
                    "[server] thinking.budget_tokens=%d clamped to "
                    "think_max_tokens=%d\n",
                    request_budget_tokens, config_.think_max_tokens);
            }
            req.per_req_phase1_cap = eff;
        } else if (effort_set) {
            // Spec §4.4: when reasoning.effort is set, the effective phase-1
            // cap is min(effort_tier_value, request.max_tokens -
            // hard_limit_reply_budget). The effort tier value can legitimately
            // exceed default_max_tokens (e.g. Qwen3.6 max=81408 with
            // default=32768) — clients that want that full budget must pass
            // an explicit max_tokens. Otherwise we narrow silently to fit.
            const int max_output_phase1_room = std::max(0,
                req.max_output - config_.hard_limit_reply_budget);
            int eff = std::min(effort_phase1_cap, max_output_phase1_room);
            if (effort_phase1_cap > max_output_phase1_room) {
                // Info-level: this is normal when clients use a tier name but
                // don't pass an explicit max_tokens. Not a warning.
                std::fprintf(stderr,
                    "[server] reasoning.effort tier=%d narrowed to %d "
                    "(max_tokens=%d - hard_limit_reply_budget=%d); "
                    "pass a larger max_tokens to use the full tier budget\n",
                    effort_phase1_cap, eff,
                    req.max_output, config_.hard_limit_reply_budget);
            }
            req.per_req_phase1_cap = eff;
        }
        // Reply budget:
        if (request_reply_budget >= 0) {
            int eff = std::min(request_reply_budget, config_.hard_limit_reply_budget);
            if (request_reply_budget > config_.hard_limit_reply_budget) {
                std::fprintf(stderr,
                    "[server] thinking.reply_budget=%d clamped to "
                    "hard_limit_reply_budget=%d\n",
                    request_reply_budget, config_.hard_limit_reply_budget);
            }
            req.per_req_reply_budget = eff;
        }
        // (effort tier doesn't influence reply_budget — spec §4.2: "the reply
        // reserve falls back to --hard-limit-reply-budget".)

        // Bandit: parse session_id from extra_body (opt-in adaptive keep_ratio)
        req.session_id = parse_session_id_from_body(body);

        // Serialize tools JSON for template injection.
        std::string tools_json;
        if (req.tools.is_array() && !req.tools.empty()) {
            tools_json = req.tools.dump();
        }

        std::string rendered;
        if (!config_.chat_template_src.empty()) {
            // Jinja path: caller supplied a chat template file via
            // --chat-template-file. Override the hardcoded QWEN3/LAGUNA
            // renderer. Used for tool-using agents that need the Anthropic
            // tool_use envelope (e.g. froggeric Qwen3.6 template).
            //
            // Special tokens like <|im_start|> / <|im_end|> are stored
            // verbatim in the GGUF vocab — use raw_token() to skip the
            // GPT-2 byte decode (otherwise <0xC4><0x91> nonsense appears).
            const std::string & bos_str = (tokenizer_.bos_id() >= 0)
                ? tokenizer_.raw_token(tokenizer_.bos_id())
                : std::string();
            const std::string & eos_str = (tokenizer_.eos_id() >= 0)
                ? tokenizer_.raw_token(tokenizer_.eos_id())
                : std::string();
            try {
                rendered = render_chat_template_jinja(
                    config_.chat_template_src,
                    chat_msgs,
                    bos_str,
                    eos_str,
                    /*add_generation_prompt=*/true,
                    enable_thinking,
                    tools_json);
            } catch (const std::exception & e) {
                send_error(fd, 500,
                    std::string("chat template (jinja) render failed: ") + e.what());
                return true;
            }
        } else {
            rendered = render_chat_template(chat_msgs, chat_format_,
                                            true, enable_thinking,
                                            tools_json);
        }
        req.prompt_tokens = tokenizer_.encode(rendered);

        // count_tokens: short-circuit after tokenization. Skip generation
        // entirely — Anthropic's contract is just `{"input_tokens": N}`.
        if (count_tokens_only) {
            json resp = {{"input_tokens", (int)req.prompt_tokens.size()}};
            send_response(fd, 200, "application/json", resp.dump() + "\n");
            return true;
        }

    } catch (const std::exception & e) {
        send_error(fd, 400, std::string("JSON parse error: ") + e.what());
        return true;  // handled (with error)
    }

    // Check context length; oversized + compression_enabled passes through — post-compress check is the real gate.
    {
        const int n_prompt = (int)req.prompt_tokens.size();
        const bool pflash_will_run =
            (config_.pflash_mode != ServerConfig::PflashMode::OFF) &&
            (drafter_tokenizer_ != nullptr) &&
            (config_.pflash_mode == ServerConfig::PflashMode::ALWAYS ||
             n_prompt >= config_.pflash_threshold);
        if (should_reject_oversized(n_prompt, req.max_output,
                                    config_.max_ctx, pflash_will_run)) {
            send_error(fd, 400, "prompt + max_tokens exceeds context window");
            return true;
        }
    }

    std::fprintf(stderr,
        "[server] chat %s format=%s stream=%s msgs=%zu tools=%zu prompt_tokens=%zu "
        "max_tokens=%d max_ctx=%d thinking=%s stops=%zu model=%s\n",
        req.response_id.c_str(),
        api_format_name(req.format),
        req.stream ? "true" : "false",
        json_array_size(req.messages),
        json_array_size(req.tools),
        req.prompt_tokens.size(),
        req.max_output,
        config_.max_ctx,
        req.thinking_enabled ? "true" : "false",
        req.stop_sequences.size(),
        req.model.c_str());

    // Set socket non-blocking for send() stall detection during streaming.
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    // Enqueue job and wait for worker.
    ServerJob job;
    job.fd = fd;
    job.req = std::move(req);

    enqueue(&job);

    // Wait for the worker to signal completion.
    {
        std::unique_lock<std::mutex> lk(job.mu);
        job.cv.wait(lk, [&]() { return job.done; });
    }

    return true;
}

// ─── Worker thread ──────────────────────────────────────────────────────

void HttpServer::worker_loop() {
    while (true) {
        ServerJob * job = dequeue();
        if (!job) break;  // stopping

        int fd = job->fd;
        const auto & req = job->req;
        auto started_at = std::chrono::steady_clock::now();

        // Track live status for /status page. RAII guard ensures idle on all paths.
        std::string prompt_excerpt;
        if (!req.prompt_tokens.empty()) {
            // Decode first ~40 tokens as a prompt excerpt (cheap, bounded).
            const int excerpt_len = std::min((int)req.prompt_tokens.size(), 40);
            std::vector<int32_t> excerpt_toks(req.prompt_tokens.begin(),
                                               req.prompt_tokens.begin() + excerpt_len);
            prompt_excerpt = tokenizer_.decode(excerpt_toks);
            if (prompt_excerpt.size() > 200) prompt_excerpt.resize(200);
        }
        {
            ServerStatus::RequestInfo info;
            info.model = req.model;
            info.format = api_format_name(req.format);
            info.session_id = req.session_id;
            info.max_output = req.max_output;
            info.temperature = req.sampler.temp;
            info.top_p = req.sampler.top_p;
            info.top_k = req.sampler.top_k;
            info.thinking_enabled = req.thinking_enabled;
            status_.set_running(prompt_excerpt, (int)req.prompt_tokens.size(), req.stream, info);
        }
        // Store messages JSON for request inspection (truncate to avoid huge payloads).
        if (!req.messages.is_null()) {
            std::string msg_str = req.messages.dump();
            if (msg_str.size() > 4096) msg_str.resize(4096);
            status_.set_messages(msg_str);
        }
        broadcast_status();
        StatusGuard status_guard{status_};

        auto finish_job = [&]() {
            std::lock_guard<std::mutex> lk(job->mu);
            job->done = true;
            job->cv.notify_one();
        };
        auto fail_request = [&](int status, const std::string & message) {
            std::fprintf(stderr, "[server] request failed: %s\n", message.c_str());
            if (req.stream) {
                json err = {{"error", {{"message", message}, {"type", "server_error"}}}};
                const std::string chunk = "data: " + err.dump() + "\n\n";
                send_all(fd, chunk.data(), chunk.size());
                const char done[] = "data: [DONE]\n\n";
                send_all(fd, done, sizeof(done) - 1);
            } else {
                send_error(fd, status, message);
            }
            finish_job();
        };

        std::fprintf(stderr,
            "[server] chat START %s format=%s stream=%s prompt_tokens=%zu "
            "max_tokens=%d tools=%zu\n",
            req.response_id.c_str(),
            api_format_name(req.format),
            req.stream ? "true" : "false",
            req.prompt_tokens.size(),
            req.max_output,
            json_array_size(req.tools));

        // Send SSE headers (skip when proxying — curl_forward handles its own headers).
        if (req.stream && config_.pflash_upstream_base.empty()) {
            if (!send_sse_headers(fd)) {
                finish_job();
                continue;
            }
        }

        // Create SSE emitter for streaming state machine.
        SseEmitter emitter(req.format, req.response_id, req.model,
                           (int)req.prompt_tokens.size(), req.tools,
                           &tool_memory_,
                           req.stop_sequences);

        // Emit initial SSE events (skip when proxying).
        if (req.stream && config_.pflash_upstream_base.empty()) {
            bool start_ok = true;
            for (const auto & chunk : emitter.emit_start()) {
                if (!send_all(fd, chunk.data(), chunk.size())) {
                    start_ok = false;
                    break;
                }
            }
            if (!start_ok) {
                finish_job();
                continue;
            }
        }

        // ── PFlash / FlowKV unified gate: FlowKV > WS1 skip > whole-prompt pFlash ──
        std::vector<int32_t> effective_prompt = req.prompt_tokens;
        bool pflash_compressed = false;
        // Compressed token count served from pFlash full-cache (-1 = not a full-cache hit).
        int pflash_full_cache_served_tokens = -1;

        if (config_.pflash_mode != ServerConfig::PflashMode::OFF &&
            drafter_tokenizer_ != nullptr)
        {
            const int n_prompt = (int)req.prompt_tokens.size();
            bool should_compress = false;
            if (config_.pflash_mode == ServerConfig::PflashMode::ALWAYS) {
                should_compress = true;
            } else if (config_.pflash_mode == ServerConfig::PflashMode::AUTO) {
                should_compress = (n_prompt >= config_.pflash_threshold);
            }

            // Detect whether this is a multi-turn continuation.
            bool is_continuation = false;
            if (should_compress && req.messages.is_array()) {
                for (const auto & _m : req.messages) {
                    if (!_m.is_object()) continue;
                    const std::string _role = _m.value("role", "");
                    if (_role == "assistant") { is_continuation = true; break; }
                    if (_m.contains("tool_calls")) {
                        const auto & _tc = _m["tool_calls"];
                        if (_tc.is_array() && !_tc.empty()) { is_continuation = true; break; }
                    }
                    if (_m.contains("content") && _m["content"].is_array()) {
                        for (const auto & _b : _m["content"]) {
                            if (_b.is_object() &&
                                (_b.value("type", "") == "tool_result" ||
                                 _b.value("type", "") == "tool_use")) {
                                is_continuation = true; break;
                            }
                        }
                    }
                    const std::string _itype = _m.value("type", "");
                    if (_itype == "function_call" || _itype == "function_call_output") {
                        is_continuation = true; break;
                    }
                    if (is_continuation) break;
                }
            }

            // FlowKV: compress aged msgs[1..n-hot_window) once per session; system + hot tail verbatim.
            if (should_compress && is_continuation && req.disk_cache_policy.compress &&
                req.messages.is_array())
            {
                int hot_window = 2;
                {
                    const char * hwe = std::getenv("PFLASH_FREEZE_HOT_WINDOW");
                    if (hwe && *hwe) {
                        int v = std::atoi(hwe);
                        if (v > 0) hot_window = v;
                    }
                }
                const int n_msgs = (int)req.messages.size();
                if (n_msgs >= 2 + hot_window) {
                    const int aged_begin = 1;
                    const int aged_end   = n_msgs - hot_window;  // exclusive

                    // Inert-guard: skip if aged band < 512 drafter tokens.
                    int aged_token_estimate = 0;
                    for (int mi = aged_begin; mi < aged_end; ++mi) {
                        const auto & msg = req.messages[mi];
                        if (!msg.is_object()) continue;
                        std::string mc;
                        if (msg.contains("content")) {
                            const auto & c = msg["content"];
                            if (c.is_string()) mc = c.get<std::string>();
                            else if (c.is_array()) {
                                for (const auto & part : c) {
                                    if (!part.is_object()) continue;
                                    const std::string pt = part.value("type", "");
                                    if (pt == "text" || pt == "input_text" ||
                                        pt == "output_text")
                                        mc += part.value("text", "");
                                }
                            }
                        }
                        if (!mc.empty())
                            aged_token_estimate += (int)drafter_tokenizer_->encode(mc).size();
                    }
                    static constexpr int kFkvInertMinTokens = 512;
                    if (aged_token_estimate < kFkvInertMinTokens) {
                        std::fprintf(stderr,
                            "[flowkv] inert-guard: aged band %d toks < %d — skip\n",
                            aged_token_estimate, kFkvInertMinTokens);
                        should_compress = false;
                    } else {
                        json modified_messages = req.messages;
                        bool any_compressed = false;
                        int n_cache_hits = 0;

                        for (int mi = aged_begin; mi < aged_end; ++mi) {
                            auto & msg = modified_messages[mi];
                            if (!msg.is_object()) continue;

                            std::string msg_content;
                            if (msg.contains("content")) {
                                const auto & c = msg["content"];
                                if (c.is_string()) {
                                    msg_content = c.get<std::string>();
                                } else if (c.is_array()) {
                                    for (const auto & part : c) {
                                        if (!part.is_object()) continue;
                                        const std::string ptype = part.value("type", "");
                                        if (ptype == "text" || ptype == "input_text" ||
                                            ptype == "output_text")
                                            msg_content += part.value("text", "");
                                    }
                                }
                            }
                            if (msg_content.empty()) continue;

                            auto msg_drafter_ids = drafter_tokenizer_->encode(msg_content);
                            if ((int)msg_drafter_ids.size() < config_.pflash_threshold) continue;

                            const PrefixHash msg_key = frozen_block_key(
                                msg_drafter_ids.data(), 0, (int)msg_drafter_ids.size());

                            std::string compressed_text;
                            auto cache_it = frozen_content_cache_.find(msg_key);
                            if (cache_it != frozen_content_cache_.end()) {
                                compressed_text = cache_it->second;
                                ++n_cache_hits;
                                std::fprintf(stderr,
                                    "[flowkv] msg[%d] cache hit (%zu drafter toks)\n",
                                    mi, msg_drafter_ids.size());
                            } else {
                                ModelBackend::CompressRequest creq;
                                creq.input_ids    = std::move(msg_drafter_ids);
                                creq.keep_ratio   = pflash_keep_ratio(config_, (int)creq.input_ids.size());
                                creq.drafter_path = config_.pflash_drafter_path;
                                creq.drafter_gpu  = config_.pflash_drafter_gpu;
                                creq.skip_park    = config_.pflash_skip_park;
                                creq.residency_action = resolve_draft_residency_action(
                                    config_.draft_residency,
                                    DraftResidencyContext{
                                        DraftResidencyUse::PFlashCompress,
                                        config_.lazy_draft,
                                        !config_.draft_path.empty(),
                                    });

                                auto cresult = backend_.compress(creq);
                                if (!cresult.ok || cresult.compressed_ids.empty()) {
                                    std::fprintf(stderr,
                                        "[flowkv] msg[%d] compress failed — kept verbatim\n", mi);
                                    continue;
                                }
                                compressed_text = drafter_tokenizer_->decode(cresult.compressed_ids);
                                std::fprintf(stderr,
                                    "[flowkv] msg[%d] %zu → %zu drafter toks (keep=%.2f)\n",
                                    mi, creq.input_ids.size(),
                                    cresult.compressed_ids.size(), creq.keep_ratio);

                                if (frozen_content_cache_.size() >= kFrozenCacheMax) {
                                    std::fprintf(stderr,
                                        "[flowkv] cache full (%zu entries) — clearing\n",
                                        frozen_content_cache_.size());
                                    frozen_content_cache_.clear();
                                }
                                frozen_content_cache_.emplace(msg_key, compressed_text);
                            }

                            msg["content"] = compressed_text;
                            any_compressed = true;
                        }

                        if (any_compressed) {
                            const bool   fkv_enable_thinking = req.thinking_enabled;
                            std::string  fkv_tools_json;
                            if (req.tools.is_array() && !req.tools.empty()) {
                                fkv_tools_json = req.tools.dump();
                            }
                            std::vector<ChatMessage> fkv_chat_msgs =
                                normalize_chat_messages(modified_messages, req.format,
                                                        tool_memory_);
                            std::string fkv_rendered;
                            bool fkv_render_ok = true;
                            if (!config_.chat_template_src.empty()) {
                                const std::string & bos_str = (tokenizer_.bos_id() >= 0)
                                    ? tokenizer_.raw_token(tokenizer_.bos_id())
                                    : std::string();
                                const std::string & eos_str = (tokenizer_.eos_id() >= 0)
                                    ? tokenizer_.raw_token(tokenizer_.eos_id())
                                    : std::string();
                                try {
                                    fkv_rendered = render_chat_template_jinja(
                                        config_.chat_template_src,
                                        fkv_chat_msgs,
                                        bos_str, eos_str,
                                        /*add_generation_prompt=*/true,
                                        fkv_enable_thinking,
                                        fkv_tools_json);
                                } catch (const std::exception & e) {
                                    std::fprintf(stderr,
                                        "[flowkv] jinja re-render failed (%s) — skipping\n",
                                        e.what());
                                    fkv_render_ok = false;
                                }
                            } else {
                                fkv_rendered = render_chat_template(
                                    fkv_chat_msgs, chat_format_,
                                    true, fkv_enable_thinking, fkv_tools_json);
                            }
                            if (fkv_render_ok) {
                                const int n_before = (int)effective_prompt.size();
                                effective_prompt  = tokenizer_.encode(fkv_rendered);
                                pflash_compressed = true;
                                std::fprintf(stderr,
                                    "[flowkv] %d → %d target toks "
                                    "(%d aged msgs, %d cache hits, hot_window=%d)\n",
                                    n_before, (int)effective_prompt.size(),
                                    aged_end - aged_begin, n_cache_hits, hot_window);
                            }
                            should_compress = false;
                        } else {
                            should_compress = false;
                            std::fprintf(stderr,
                                "[flowkv] no aged msgs above threshold — skip\n");
                        }
                    }
                } else {
                    should_compress = false;
                    std::fprintf(stderr,
                        "[flowkv] too few turns (n_msgs=%d hot_window=%d) — skip\n",
                        n_msgs, hot_window);
                }
            } else if (should_compress && is_continuation) {
                // Continuation without FlowKV: skip compression to preserve prefix KV cache (~22x).
                should_compress = false;
                std::fprintf(stderr,
                    "[pflash] skip-compress (continuation: prior assistant/tool history)\n");
            }

            // WS1: turn-1 verbatim anchor — compressing would cold-poison turn-2 cache key.
            if (should_compress && !is_continuation && req.disk_cache_policy.compress) {
                should_compress = false;
                std::fprintf(stderr,
                    "[flowkv] turn-1 verbatim (system kept as cache anchor)\n");
            }

            if (should_compress) {
                // Check full-compress cache FIRST — if we've seen this exact
                // raw prompt before, skip the expensive compress cycle entirely.
                auto [full_slot, full_len] = prefix_cache_.lookup_full(req.prompt_tokens);
                if (full_slot >= 0) {
                    std::fprintf(stderr, "[pflash] full-cache hit slot=%d — skipping compress\n", full_slot);
                    pflash_compressed = true;
                    // effective_prompt stays as req.prompt_tokens — the cached KV
                    // state will be restored via cache_slot below.
                    // Record the compressed size for the post-compress budget gate.
                    pflash_full_cache_served_tokens = full_len;
                } else {
                    std::string compression_error;
                    // 1. Decode prompt to text using target tokenizer
                    std::string prompt_text = tokenizer_.decode(req.prompt_tokens);

                    // 2. Re-encode with drafter tokenizer
                    auto drafter_ids = drafter_tokenizer_->encode(prompt_text);

                    if (drafter_ids.empty()) {
                        compression_error = "PFlash drafter tokenizer produced an empty prompt";
                    } else {
                        // 3. Compress via typed API
                        ModelBackend::CompressRequest creq;
                        creq.input_ids = std::move(drafter_ids);
                        // Bandit overrides curve when session_id is present.
                        creq.keep_ratio = req.session_id.empty()
                            ? pflash_keep_ratio(config_, n_prompt)
                            : sessions_.get_keep_ratio(req.session_id);
                        creq.drafter_path = config_.pflash_drafter_path;
                        creq.drafter_gpu = config_.pflash_drafter_gpu;
                        creq.skip_park = config_.pflash_skip_park;
                        const auto pflash_residency =
                            resolve_draft_residency_action(
                                config_.draft_residency,
                                DraftResidencyContext{
                                    DraftResidencyUse::PFlashCompress,
                                    config_.lazy_draft,
                                    !config_.draft_path.empty(),
                                });
                        creq.residency_action = pflash_residency;

                        ModelBackend::CompressResult cresult;
                        if (config_.pflash_remote_drafter) {
                            if (!pflash_remote_.active() &&
                                !pflash_remote_.start(config_.pflash_remote.ipc_bin,
                                                       config_.pflash_drafter_path,
                                                       config_.pflash_drafter_gpu,
                                                       config_.pflash_remote.work_dir)) {
                                compression_error = "remote PFlash drafter start failed";
                            } else {
                                cresult.ok = pflash_remote_.compress(
                                    creq.input_ids, creq.keep_ratio,
                                    cresult.compressed_ids);
                                if (pflash_residency == DraftResidencyAction::ReleaseAfterUse) {
                                    pflash_remote_.close();
                                }
                            }
                        } else {
                            cresult = backend_.compress(creq);
                        }

                        // 4. Decode compressed IDs with drafter tokenizer
                        if (cresult.ok && !cresult.compressed_ids.empty()) {
                            std::string compressed_text =
                                drafter_tokenizer_->decode(cresult.compressed_ids);

                            // 5. Query survival check: verify the last user
                            //    message survived compression. If < 80% of its
                            //    tokens are present, re-append the full query.
                            std::string last_user_text;
                            if (req.messages.is_array()) {
                                for (int mi = (int)req.messages.size() - 1; mi >= 0; --mi) {
                                    if (req.messages[mi].value("role", "") == "user") {
                                        auto & c = req.messages[mi]["content"];
                                        if (c.is_string()) {
                                            last_user_text = c.get<std::string>();
                                        } else if (c.is_array()) {
                                            for (const auto & part : c) {
                                                std::string ptype = part.value("type", "");
                                                if (ptype == "text" || ptype == "input_text" ||
                                                    ptype == "output_text") {
                                                    last_user_text += part.value("text", "");
                                                }
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                            if (!last_user_text.empty() && drafter_tokenizer_) {
                                auto query_ids = drafter_tokenizer_->encode(last_user_text);
                                int query_kept = 0;
                                if (!query_ids.empty()) {
                                    int qi = (int)query_ids.size() - 1;
                                    for (int ki = (int)cresult.compressed_ids.size() - 1; ki >= 0 && qi >= 0; --ki) {
                                        if (cresult.compressed_ids[ki] == query_ids[qi]) {
                                            ++query_kept;
                                            --qi;
                                        }
                                    }
                                }
                                float survival = (float)query_kept / std::max(1, (int)query_ids.size());
                                std::fprintf(stderr, "[pflash] query survival: %d/%d (%.0f%%)\n",
                                             query_kept, (int)query_ids.size(), survival * 100.0f);
                                if (survival < 0.80f && (int)query_ids.size() < 1000) {
                                    compressed_text += "\n" + last_user_text;
                                    std::fprintf(stderr, "[pflash] query below 80%% — re-appended full query (%d tokens)\n",
                                                 (int)query_ids.size());
                                } else if (survival < 0.80f) {
                                    std::fprintf(stderr, "[pflash] query below 80%% but too large to re-append (%d tokens)\n",
                                                 (int)query_ids.size());
                                }
                            }

                            // 6. Re-tokenize with target tokenizer
                            effective_prompt = tokenizer_.encode(compressed_text);
                            pflash_compressed = true;

                            std::fprintf(stderr,
                                "[pflash] %d -> %d -> %d tokens (%.1f%% kept)\n",
                                n_prompt, (int)cresult.compressed_ids.size(),
                                (int)effective_prompt.size(),
                                100.0 * effective_prompt.size() / n_prompt);
                        } else if (compression_error.empty()) {
                            compression_error = config_.pflash_remote_drafter
                                ? "remote PFlash drafter compression failed"
                                : "PFlash compression failed";
                        }
                    }
                    if (!pflash_compressed && !compression_error.empty()) {
                        fail_request(500, compression_error);
                        continue;
                    }
                }
            }
        }

        // Post-compress gate: reject if still oversized after FlowKV/pFlash.
        // On a pFlash full-cache hit, use the cached compressed size (not the raw
        // effective_prompt) — the KV was built from the compressed form.
        if (effective_prompt_overflows((int)effective_prompt.size(),
                                       pflash_full_cache_served_tokens,
                                       req.max_output, config_.max_ctx)) {
            fail_request(400, "effective prompt + max_tokens exceeds context window after compression");
            continue;
        }

        // ── Upstream proxy: forward to remote server if configured ────
#ifdef DFLASH_HAS_CURL
        if (!config_.pflash_upstream_base.empty()) {
            const std::string & upstream = config_.pflash_upstream_base;
            const std::string & upstream_key = config_.pflash_upstream_key;
            const std::string & upstream_model = config_.pflash_upstream_model.empty()
                ? req.model : config_.pflash_upstream_model;

            if (pflash_compressed) {
                std::string compressed_text = tokenizer_.decode(effective_prompt);
                compressed_text += "\n<|im_start|>assistant\n";

                json comp_body;
                comp_body["model"] = upstream_model;
                comp_body["prompt"] = compressed_text;
                comp_body["stream"] = req.stream;
                if (req.raw_body.contains("max_tokens"))
                    comp_body["max_tokens"] = req.raw_body["max_tokens"];
                else
                    comp_body["max_tokens"] = req.max_output;
                for (const char * key : {"temperature", "top_p", "top_k", "min_p",
                                         "frequency_penalty", "presence_penalty",
                                         "stop", "seed"}) {
                    if (req.raw_body.contains(key)) comp_body[key] = req.raw_body[key];
                }

                std::fprintf(stderr,
                    "[pflash-proxy] compressed forward → %s/completions  prompt=%zu tokens  model=%s\n",
                    upstream.c_str(), effective_prompt.size(), upstream_model.c_str());

                curl_forward(fd, upstream + "/completions",
                             upstream_key, comp_body,
                             req.stream, /*rewrite_to_chat=*/true,
                             req.response_id, upstream_model);
            } else {
                json fwd_body = req.raw_body;
                fwd_body["model"] = upstream_model;

                std::fprintf(stderr,
                    "[pflash-proxy] passthrough → %s/chat/completions  model=%s\n",
                    upstream.c_str(), upstream_model.c_str());

                curl_forward(fd, upstream + "/chat/completions",
                             upstream_key, fwd_body,
                             req.stream, /*rewrite_to_chat=*/false,
                             req.response_id, upstream_model);
            }
            finish_job();
            continue;
        }
#endif // DFLASH_HAS_CURL

        // Build generate request.
        //
        // Thinking-budget v2 (Level 2): when caller opts in via
        // `thinking:{type:enabled}`, cap n_gen at think_max + reply_budget
        // so the BudgetHook fires at the boundary, mid-stream, with KV
        // state intact. Applies uniformly to streaming and non-streaming
        // requests — the BudgetHook lives inside do_ar_decode /
        // do_spec_decode and injects close tokens at the budget edge
        // regardless of how the server is delivering the result.
        const bool budget_active = req.thinking_opt_in;
        // Effective think cap: per-request value (already clamped to
        // config_.think_max_tokens above) wins over the server-wide
        // think_max_tokens. Then both must fit inside the combined
        // max_output. Spec §4.4 + §5.3.
        const int effective_think_ceiling = (req.per_req_phase1_cap >= 0)
            ? req.per_req_phase1_cap
            : config_.think_max_tokens;
        // The effective per-request reply budget is the operator's choice
        // (CLI / sidecar / per-request override). The AR loop force-closes
        // when `n_gen - generated <= eff_reply`, which means n_gen must
        // include BOTH the think budget AND the reply reserve. Without the
        // `+ eff_reply` term, force-close fires immediately when
        // `eff_reply == effective_think_ceiling` (e.g. think_max=4096,
        // hard_limit=4096 → remaining starts at 4096, condition fires
        // before the model emits a single thinking token). Spec §4.4.
        const int eff_reply_for_n_gen = (req.per_req_reply_budget >= 0)
            ? req.per_req_reply_budget
            : config_.hard_limit_reply_budget;
        const int n_gen_cap = budget_active
            ? std::min(effective_think_ceiling + eff_reply_for_n_gen, req.max_output)
            : req.max_output;

        GenerateRequest gen_req;
        gen_req.prompt = effective_prompt;
        gen_req.n_gen = n_gen_cap;
        gen_req.sampler = req.sampler;
        gen_req.do_sample = req.sampler.needs_logit_processing();
        gen_req.stream = false;  // we handle streaming via on_token callback

        // Level 2 force-close: when thinking is opted in, the server is
        // configured with a hard-limit reply budget, and we resolved the
        // close-tag sequence at startup, wire the BudgetHook so the
        // backend's AR decode injects `</think>` at the budget boundary.
        // The model gets to write the visible answer in-stream rather than
        // running unbounded.
        //
        // hard_limit_remaining is the per-request reply_budget when set
        // (already clamped to config_.hard_limit_reply_budget above), else
        // the server default. Spec §4.4 + §5.3.
        if (budget_active && !config_.think_close_token_ids.empty() &&
            config_.hard_limit_reply_budget > 0)
        {
            int eff_reply_budget = (req.per_req_reply_budget >= 0)
                ? req.per_req_reply_budget
                : config_.hard_limit_reply_budget;
            gen_req.budget_hook.close_token_ids = config_.think_close_token_ids;
            gen_req.budget_hook.hard_limit_remaining = eff_reply_budget;
        }

        // Tool call hint generation: pre-tokenize predictable structural tokens
        // to accelerate spec decode when tool_choice constrains the output.
        std::vector<int32_t> hint_tokens_storage;
        if (!req.tools.empty() && !req.tool_choice.is_null()) {
            ToolHintGenerator hint_gen(tokenizer_);
            auto hint = hint_gen.build_hint(req.tools, req.tool_choice);
            if (!hint.empty()) {
                hint_tokens_storage = std::move(hint.prefix_tokens);
                gen_req.hint_tokens = &hint_tokens_storage;
            }
        }
        std::vector<int32_t> stall_tool_prefix_tokens_storage;
        std::vector<int32_t> stall_action_suffix_tokens_storage;
        std::vector<int32_t> stall_skip_tokens_storage;
        if (!req.tools.empty() && env_flag_enabled("DFLASH_STALL_TOOL_PREFIX")) {
            stall_tool_prefix_tokens_storage =
                tokenizer_.encode(build_stall_tool_prefix(req.tools,
                                                          req.tool_choice));
            stall_action_suffix_tokens_storage = tokenizer_.encode(":");
            // The stall detector only asks "did the model just end on an action
            // suffix?" by matching individual recent tokens (tokens_have_recent_any),
            // so we collect the trailing token of each colon variant rather than the
            // full encoded sequence. The final ":" piece is what actually precedes the
            // premature EOS regardless of how the preamble before it tokenizes, so a
            // flat set of these terminal tokens is the right granularity here.
            auto add_suffix_terminal = [&](const std::string & text) {
                auto ids = tokenizer_.encode(text);
                if (ids.empty()) return;
                int32_t tok = ids.back();
                if (std::find(stall_action_suffix_tokens_storage.begin(),
                              stall_action_suffix_tokens_storage.end(), tok) ==
                    stall_action_suffix_tokens_storage.end()) {
                    stall_action_suffix_tokens_storage.push_back(tok);
                }
            };
            add_suffix_terminal("`:");
            add_suffix_terminal("):");
            add_suffix_terminal("\":");
            stall_skip_tokens_storage = tokenizer_.encode(" done");
            gen_req.stall_tool_prefix_tokens = &stall_tool_prefix_tokens_storage;
            gen_req.stall_action_suffix_tokens = &stall_action_suffix_tokens_storage;
            gen_req.stall_skip_tokens = &stall_skip_tokens_storage;
        }

        // Prefix cache: check for cached KV state.
        auto [cache_slot, prefix_len] = prefix_cache_.lookup(effective_prompt);
        bool using_restore = (cache_slot >= 0);

        // Full-compress cache: if we compressed, check for cached KV.
        if (pflash_compressed) {
            auto [full_slot, full_len] = prefix_cache_.lookup_full(req.prompt_tokens);
            if (full_slot >= 0) {
                // Exact-match hit on the raw (uncompressed) prompt — skip compression.
                cache_slot = full_slot;
                prefix_len = full_len;
                using_restore = true;
                std::fprintf(stderr, "[pflash] full-cache hit slot=%d\n", full_slot);
            }
        }

        // Disk prefix cache: try disk if memory missed.
        // Staging slot is the last ModelBackend slot, reserved for disk loads.
        // PrefixCache inline uses 0..cap-1 and full uses cap..cap+full_cap-1,
        // so slot 63 is safe as long as total cache slots < 63.
        static constexpr int DISK_STAGING_SLOT = ModelBackend::kMaxSlots - 1;
        bool disk_hit = false;
        DiskPrefixCachePolicy disk_policy = req.disk_cache_policy;
        // system_end: first chat-marker boundary; FlowKV clamps disk cache to verbatim system prefix.
        int system_end = 0;
        if (pflash_compressed && req.disk_cache_policy.compress) {
            // FlowKV: cache only [0, system_end) — stable cross-session key, never compressed.
            auto fkv_boundaries =
                find_all_boundaries(effective_prompt, prefix_cache_.chat_markers());
            system_end = fkv_boundaries.empty() ? 0 : fkv_boundaries[0];
            if (system_end >= config_.disk_cache_min_tokens) {
                disk_policy.mode = DiskPrefixCacheMode::Fixed;
                disk_policy.fixed_tokens = system_end;
                std::fprintf(stderr,
                    "[flowkv] disk-clamp: boundary clamped to system_end=%d\n", system_end);
            } else {
                disk_policy.mode = DiskPrefixCacheMode::Off;
                std::fprintf(stderr,
                    "[flowkv] disk-clamp: system_end=%d < min=%d — disk off\n",
                    system_end, config_.disk_cache_min_tokens);
            }
        } else if (pflash_compressed) {
            // Standard PFlash (compress=false): effective_prompt is rewritten; only Full cache is safe.
            if (disk_policy.mode != DiskPrefixCacheMode::Full) {
                disk_policy.mode = DiskPrefixCacheMode::Off;
            }
        }
        std::vector<int> safe_boundaries;
        if (disk_policy.mode == DiskPrefixCacheMode::Auto) {
            safe_boundaries =
                find_all_boundaries(effective_prompt, prefix_cache_.chat_markers());
        }
        int selected_prefix_boundary = 0;
        if (disk_policy.mode == DiskPrefixCacheMode::Fixed) {
            selected_prefix_boundary =
                disk_prefix_cache_fixed_boundary(
                    disk_policy, (int)effective_prompt.size(),
                    config_.disk_cache_min_tokens);
        } else if (disk_policy.mode == DiskPrefixCacheMode::Auto) {
            selected_prefix_boundary =
                disk_prefix_cache_auto_boundary(
                    effective_prompt, recent_disk_prompts_, disk_policy.auto_window,
                    safe_boundaries, config_.disk_cache_min_tokens);
            std::fprintf(stderr,
                "[disk-cache] auto scope: window=%d recent=%zu safe=%zu selected=%d\n",
                disk_policy.auto_window,
                std::min(recent_disk_prompts_.size(), (size_t)disk_policy.auto_window),
                safe_boundaries.size(), selected_prefix_boundary);
        }
        std::vector<int> disk_lookup_lengths;
        if (disk_policy.mode == DiskPrefixCacheMode::Full &&
            !effective_prompt.empty()) {
            disk_lookup_lengths.push_back((int)effective_prompt.size());
            if ((int)effective_prompt.size() > config_.disk_cache_cold_max_tokens) {
                auto boundaries =
                    find_all_boundaries(effective_prompt, prefix_cache_.chat_markers());
                int cold_boundary = 0;
                for (int b : boundaries) {
                    if (b <= config_.disk_cache_cold_max_tokens &&
                        b >= config_.disk_cache_min_tokens) {
                        cold_boundary = b;
                    }
                }
                if (cold_boundary > 0 &&
                    cold_boundary != (int)effective_prompt.size()) {
                    disk_lookup_lengths.push_back(cold_boundary);
                }
            }
        } else if (selected_prefix_boundary > 0) {
            disk_lookup_lengths.push_back(selected_prefix_boundary);
        } else if (disk_policy.mode == DiskPrefixCacheMode::Auto) {
            for (auto it = safe_boundaries.rbegin(); it != safe_boundaries.rend(); ++it) {
                const int b = *it;
                if (b >= config_.disk_cache_min_tokens &&
                    b <= (int)effective_prompt.size()) {
                    disk_lookup_lengths.push_back(b);
                }
            }
        }
        if (!using_restore && !disk_cache_.disabled()) {
            for (int lookup_len : disk_lookup_lengths) {
                std::vector<int32_t> prefix_tokens(
                    effective_prompt.begin(), effective_prompt.begin() + lookup_len);
                if (disk_cache_.lookup(prefix_tokens, DISK_STAGING_SLOT)) {
                    cache_slot = DISK_STAGING_SLOT;
                    prefix_len = backend_.snapshot_cur_pos(DISK_STAGING_SLOT);
                    if (prefix_len <= 0 || prefix_len > (int)effective_prompt.size()) {
                        std::fprintf(stderr,
                            "[disk-cache] ignoring invalid hit pos=%d prompt=%zu\n",
                            prefix_len, effective_prompt.size());
                        backend_.snapshot_free(DISK_STAGING_SLOT);
                        continue;
                    }
                    using_restore = true;
                    disk_hit = true;
                    std::fprintf(stderr,
                        "[disk-cache] hit policy=%s len=%d slot=%d pos=%d\n",
                        disk_prefix_cache_policy_name(disk_policy).c_str(), lookup_len,
                        DISK_STAGING_SLOT, prefix_len);
                    break;
                }
            }
        }

        // Scoped prefix save: auto/fixed modes prefill exactly to the
        // selected token boundary and save that snapshot.
        // This keeps the disk key and snapshot position aligned; unlike the
        // legacy full-prompt key path, scoped entries must not point at a
        // longer snapshot than their token hash covers.
        if (!using_restore && !disk_cache_.disabled() &&
            selected_prefix_boundary > 0) {
            const int scoped_boundary = selected_prefix_boundary;
            if (scoped_boundary > 0) {
                std::fprintf(stderr,
                    "[disk-cache] scoped prefix: policy=%s boundary=%d\n",
                    disk_prefix_cache_policy_name(disk_policy).c_str(), scoped_boundary);
                GenerateRequest scoped_req;
                scoped_req.prompt = std::vector<int32_t>(
                    effective_prompt.begin(), effective_prompt.begin() + scoped_boundary);
                scoped_req.n_gen = 0;
                scoped_req.snap_slot = DISK_STAGING_SLOT;
                scoped_req.snap_pos = scoped_boundary;
                DaemonIO scoped_io;
                scoped_io.stream_fd = -1;
                auto scoped_result = backend_.generate(scoped_req, scoped_io);
                if (scoped_result.ok && backend_.snapshot_used(DISK_STAGING_SLOT)) {
                    disk_cache_.learn_layout(DISK_STAGING_SLOT);
                    const bool saved =
                        disk_cache_.save(DISK_STAGING_SLOT, scoped_req.prompt);
                    cache_slot = DISK_STAGING_SLOT;
                    prefix_len = scoped_boundary;
                    using_restore = true;
                    disk_hit = true;
                    std::fprintf(stderr,
                        "[disk-cache] scoped prefix %s, restoring from %d\n",
                        saved ? "saved" : "staged", scoped_boundary);
                } else {
                    backend_.snapshot_free(DISK_STAGING_SLOT);
                }
            }
        }

        // A slot whose snapshot covers more KV than this prompt cannot be
        // diff-prefilled (the client edited/summarized its history since the
        // snapshot was saved). Treat as a cache miss; the backend-side
        // fallback also guards this, but skipping restore here avoids a
        // pointless snapshot copy-in.
        if (using_restore) {
            const int snap_len = backend_.snapshot_cur_pos(cache_slot);
            if (snap_len > (int)effective_prompt.size()) {
                std::fprintf(stderr,
                    "[pc] slot=%d snapshot pos=%d > prompt=%zu — treating as miss\n",
                    cache_slot, snap_len, effective_prompt.size());
                cache_slot = -1;
                prefix_len = 0;
                using_restore = false;
                disk_hit = false;
            }
        }

        // Cold prefix save: for long prompts with no cache hit, prefill to a
        // turn boundary and save a cold checkpoint before the full generation.
        // This makes subsequent requests to similar (but not identical) prompts
        // much faster by reusing the cold prefix.
        if (!using_restore && !disk_cache_.disabled() &&
            disk_policy.mode == DiskPrefixCacheMode::Full) {
            auto boundaries = find_all_boundaries(effective_prompt, prefix_cache_.chat_markers());
            int cold_boundary = disk_cache_.cold_prefix_boundary(effective_prompt, boundaries);
            if (cold_boundary > 0) {
                std::fprintf(stderr, "[disk-cache] cold prefix: prefilling to boundary=%d\n",
                             cold_boundary);
                // Phase 1: prefill to cold_boundary with snapshot save.
                GenerateRequest cold_req;
                cold_req.prompt = std::vector<int32_t>(effective_prompt.begin(),
                                                       effective_prompt.begin() + cold_boundary);
                cold_req.n_gen = 0;  // no decode, just prefill
                cold_req.snap_slot = DISK_STAGING_SLOT;
                cold_req.snap_pos = cold_boundary;  // save at end of prefix
                DaemonIO cold_io;
                cold_io.stream_fd = -1;
                auto cold_result = backend_.generate(cold_req, cold_io);
                if (cold_result.ok && backend_.snapshot_used(DISK_STAGING_SLOT)) {
                    disk_cache_.learn_layout(DISK_STAGING_SLOT);
                    std::vector<int32_t> prefix_tokens(effective_prompt.begin(),
                                                       effective_prompt.begin() + cold_boundary);
                    disk_cache_.save(DISK_STAGING_SLOT, prefix_tokens);
                    // Use this cold snapshot as restore point for full generation.
                    cache_slot = DISK_STAGING_SLOT;
                    prefix_len = cold_boundary;
                    using_restore = true;
                    disk_hit = true;  // ensure staging slot is freed after use
                    std::fprintf(stderr, "[disk-cache] cold prefix saved, restoring from %d\n",
                                 cold_boundary);
                } else {
                    backend_.snapshot_free(DISK_STAGING_SLOT);
                }
            }
        }

        // Prepare inline snapshot for future cache hits.
        auto [snap_slot, snap_cut] = prefix_cache_.prepare_inline_snap(effective_prompt);
        bool snap_prepared = (snap_slot >= 0);
        if (snap_prepared) {
            gen_req.snap_slot = snap_slot;
            gen_req.snap_pos = snap_cut;
        }

        std::fprintf(stderr,
            "[server] chat CACHE %s restore=%s slot=%d prefix_len=%d "
            "effective_prompt=%zu pflash=%s disk_policy=%s disk_hit=%s "
            "snap_slot=%d snap_pos=%d\n",
            req.response_id.c_str(),
            using_restore ? "true" : "false",
            cache_slot,
            prefix_len,
            effective_prompt.size(),
            pflash_compressed ? "true" : "false",
            disk_prefix_cache_policy_name(disk_policy).c_str(),
            disk_hit ? "true" : "false",
            snap_slot,
            snap_cut);

        // Update status page with cache/pflash/spec-decode flags.
        status_.set_flags(using_restore, pflash_compressed, !config_.draft_path.empty());
        broadcast_status();

        // Set up DaemonIO with on_token callback for streaming + disconnect.
        DaemonIO io;
        io.stream_fd = -1;  // no pipe — we write SSE directly

        // Inference observer: updates status page with draft tokens per step.
        io.observer = [&](const char * phase, const std::vector<int32_t> & tokens) {
            std::vector<std::string> token_strs;
            token_strs.reserve(tokens.size());
            for (int32_t t : tokens) {
                token_strs.push_back(tokenizer_.token_text(t));
            }
            status_.set_draft_tokens(token_strs);
            broadcast_status();
        };

        int completion_tokens = 0;
        bool visible_output_seen = false;
        bool client_disconnected = false;

        io.on_token = [&](int32_t token) -> bool {
            if (client_disconnected) return false;
            completion_tokens++;

            // Update status page every 10 tokens (low overhead).
            if (completion_tokens % 10 == 0) {
                status_.update_completion_tokens(completion_tokens);
                broadcast_status();
            }

            // Skip EOS/EOT/special tokens — don't forward to SSE.
            int32_t eos = tokenizer_.eos_id();
            int32_t eot = tokenizer_.eos_chat_id();
            if (token == eos || token == eot) return true;

            const std::string & raw = tokenizer_.raw_token(token);

            // Gemma4 thinking channel: map <|channel> → <think>, <channel|> → </think>\n
            if (raw == "<|channel>") {
                visible_output_seen = true;
                broadcast_token("<think>");
                if (req.stream) {
                    auto chunks = emitter.emit_token("<think>");
                    for (const auto & chunk : chunks)
                        if (!send_all(fd, chunk.data(), chunk.size())) { client_disconnected = true; return false; }
                }
                return true;
            }
            if (raw == "<channel|>") {
                visible_output_seen = true;
                broadcast_token("</think>\n");
                if (req.stream) {
                    auto chunks = emitter.emit_token("</think>\n");
                    for (const auto & chunk : chunks)
                        if (!send_all(fd, chunk.data(), chunk.size())) { client_disconnected = true; return false; }
                }
                return true;
            }

            // Qwen3.6 thinking tokens: <think> (id 248068) and </think> (id 248069)
            // are SINGLE special tokens in the added_tokens vocab. Without this
            // mapping they hit the generic "skip <...>" filter below and get
            // silently dropped — which means the emitter never sees the
            // reasoning→content transition and stuffs everything into
            // reasoning_content with empty visible content. Forward the text
            // form into the emitter so parse_reasoning() can split correctly.
            if (raw == "<think>" || raw == "</think>") {
                visible_output_seen = true;
                broadcast_token(raw == "</think>" ? "</think>\n" : "<think>");
                if (req.stream) {
                    auto chunks = emitter.emit_token(
                        raw == "</think>" ? "</think>\n" : "<think>");
                    for (const auto & chunk : chunks)
                        if (!send_all(fd, chunk.data(), chunk.size())) { client_disconnected = true; return false; }
                }
                return true;
            }

            // Skip other special tokens (starting with <|, or any <...> except byte-fallback)
            if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') return true;
            if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>') {
                if (!(raw.size() == 6 && raw[1] == '0' && raw[2] == 'x'))
                    return true;
            }

            std::string text = tokenizer_.token_text(token);

            // Send token text to status page clients (browser accumulates).
            if (!text.empty()) {
                visible_output_seen = true;
                broadcast_token(text);
            }

            if (req.stream && !text.empty()) {
                auto chunks = emitter.emit_token(text);
                for (const auto & chunk : chunks) {
                    if (!send_all(fd, chunk.data(), chunk.size())) {
                        client_disconnected = true;
                        return false;
                    }
                }
                // Stop generation if a stop sequence was hit.
                if (emitter.stop_hit()) return false;
            }
            return true;
        };

        const auto dflash_residency =
            resolve_draft_residency_action(
                config_.draft_residency,
                DraftResidencyContext{
                    DraftResidencyUse::DFlashDecode,
                    config_.lazy_draft,
                    !config_.draft_path.empty(),
                });

        // Run generation (with or without restore).
        // Request-scoped draft residency ensures decode draft is loaded only
        // around the generation window, leaving room for PFlash/target state.
        if (dflash_residency == DraftResidencyAction::ReleaseAfterUse &&
            !config_.draft_path.empty()) {
            backend_.free_drafter();    // free pflash drafter (~1.4 GB) if loaded
            backend_.unpark("draft");   // reload decode draft (~3.3 GB)
        }

        // Transition status to decode phase.
        status_.set_decode();
        broadcast_status();

        GenerateResult result;
        if (using_restore) {
            result = backend_.restore_and_generate(cache_slot, gen_req, io);
        } else {
            result = backend_.generate(gen_req, io);
        }

        if (dflash_residency == DraftResidencyAction::ReleaseAfterUse &&
            !config_.draft_path.empty()) {
            backend_.park("draft");
        }

        // Release oversized scratch buffers (gallocr, BSA cache) so VRAM
        // doesn't grow monotonically across requests with different sizes.
        backend_.release_scratch();

        // Bandit: update when spec decode actually ran — including 0-accept case,
        // which signals the current keep_ratio is too low.
        if (!req.session_id.empty() && result.spec_decode_ran) {
            float old_keep = sessions_.get_keep_ratio(req.session_id);
            int   old_turn = sessions_.turn_count(req.session_id);
            sessions_.update(req.session_id, result.accept_rate);
            float new_keep = sessions_.get_keep_ratio(req.session_id);
            float ema      = sessions_.get_ema(req.session_id);
            std::fprintf(stderr,
                "[pflash-bandit] session=%s turn=%d keep=%.4f->%.4f ema=%.3f accept=%.3f\n",
                req.session_id.c_str(), old_turn + 1,
                old_keep, new_keep, ema, result.accept_rate);
        }


        // Confirm or abort the inline snapshot.
        if (snap_prepared) {
            if (completion_tokens > 0 && visible_output_seen && !client_disconnected &&
                backend_.snapshot_used(snap_slot)) {
                prefix_cache_.confirm_inline_snap(snap_slot, snap_cut, effective_prompt);
                // Track for shutdown save.
                slot_tokens_[snap_slot] = std::vector<int32_t>(
                    effective_prompt.begin(), effective_prompt.begin() + snap_cut);
                // Save to disk cache if threshold met.
                if (!disk_cache_.disabled()) {
                    disk_cache_.learn_layout(snap_slot);
                    if (disk_policy.mode == DiskPrefixCacheMode::Full) {
                        disk_cache_.save(snap_slot, effective_prompt);
                    }
                }
            } else {
                prefix_cache_.abort_inline_snap(snap_slot);
            }
        }

        // Free the disk staging slot after use.
        if (disk_hit) {
            backend_.snapshot_free(DISK_STAGING_SLOT);
        }

        // Continued checkpoint: save if total tokens crossed an interval boundary.
        // This captures prompt + all generated tokens for long conversation reuse.
        if (!disk_cache_.disabled() && disk_policy.mode == DiskPrefixCacheMode::Full &&
            result.ok && completion_tokens > 0 &&
            visible_output_seen && !client_disconnected) {
            int final_pos = (int)effective_prompt.size() + (int)result.tokens.size();
            if (final_pos >= disk_cache_.continued_interval()) {
                // Build all_tokens = effective_prompt + result.tokens
                std::vector<int32_t> all_tokens(effective_prompt);
                all_tokens.insert(all_tokens.end(), result.tokens.begin(), result.tokens.end());
                // Save a snapshot of the live KV at end-of-generation.
                if (backend_.snapshot_save(DISK_STAGING_SLOT)) {
                    disk_cache_.learn_layout(DISK_STAGING_SLOT);
                    disk_cache_.maybe_store_continued(DISK_STAGING_SLOT, all_tokens, final_pos);
                    backend_.snapshot_free(DISK_STAGING_SLOT);
                }
            }
        }

        if (!disk_cache_.disabled()) {
            if (!pflash_compressed) {
                recent_disk_prompts_.insert(recent_disk_prompts_.begin(), effective_prompt);
            } else if (req.disk_cache_policy.compress) {
                // FlowKV: record verbatim prompt so Auto boundary lookups see stable content.
                recent_disk_prompts_.insert(recent_disk_prompts_.begin(), req.prompt_tokens);
            }
            // pflash_compressed && !compress: skip (effective_prompt is rewritten).
            static constexpr size_t kMaxRecentDiskPrompts = 256;
            if (recent_disk_prompts_.size() > kMaxRecentDiskPrompts) {
                recent_disk_prompts_.resize(kMaxRecentDiskPrompts);
            }
        }

        // Full-compress cache: reserve + confirm after successful generation.
        if (pflash_compressed && completion_tokens > 0 &&
            visible_output_seen && !client_disconnected) {
            int full_slot = prefix_cache_.prepare_full_snap(req.prompt_tokens);
            if (full_slot >= 0) {
                prefix_cache_.confirm_full_snap(full_slot, req.prompt_tokens,
                                                (int)effective_prompt.size());
            }
        }

        // close_kind reflects the Level 2 BudgetHook outcome: "hard" when
        // the backend's AR/spec decode injected the close-token sequence
        // at the budget boundary, "natural" when the model self-closed
        // (or the request never opted in). Emitted as part of
        // finish_details for thinking-budget callers.
        std::string close_kind =
            (req.thinking_opt_in && result.budget_forced_close)
                ? "hard"
                : "natural";

        // Finalize.
        // Per-request wall-clock timings forwarded to the response's
        // `usage.timings` (OpenAI Chat usage chunk, Anthropic
        // message_delta usage, Responses response.completed usage).
        // See docs/specs/thinking-budget.md §6.3.
        GenTimings gen_timings{ result.prefill_s, result.decode_s };

        // Record performance for /status page.
        if (result.ok) {
            PerfRecord perf;
            perf.prompt_tokens = (int)req.prompt_tokens.size();
            perf.completion_tokens = completion_tokens;
            // Use actual prefilled token count: on cache hit the backend only
            // prefills the delta beyond the cached prefix, so dividing the full
            // prompt size by delta time would be wrong.
            const int prefill_tokens = using_restore
                ? std::max(0, (int)effective_prompt.size() - prefix_len)
                : (int)effective_prompt.size();
            perf.prefill_tok_s = (result.prefill_s > 0.0)
                ? (double)prefill_tokens / result.prefill_s : 0.0;
            perf.decode_tok_s = (result.decode_s > 0.0)
                ? (double)completion_tokens / result.decode_s : 0.0;
            perf.accept_rate = result.accept_rate;
            perf.cache_hit = using_restore;
            perf.pflash = pflash_compressed;
            perf.spec_decode = result.spec_decode_ran;
            perf.timestamp = std::chrono::steady_clock::now();
            status_.record_perf(perf);
            status_.update_completion_tokens(completion_tokens);
            broadcast_status();
        }
        if (req.stream && !client_disconnected) {
            auto final_chunks = emitter.emit_finish(completion_tokens, &gen_timings);
            for (const auto & chunk : final_chunks) {
                if (!send_all(fd, chunk.data(), chunk.size())) {
                    client_disconnected = true;
                    break;
                }
            }
        } else if (!req.stream && !client_disconnected) {
            // Non-streaming: build complete response using emitter state.
            // Feed all tokens through emitter (skip specials like streaming path).
            auto feed_tokens = [&](const std::vector<int32_t> & toks) -> bool {
                for (int32_t tok : toks) {
                    const std::string & raw = tokenizer_.raw_token(tok);
                    if (tok == tokenizer_.eos_id()) continue;
                    if (tok == tokenizer_.eos_chat_id()) continue;
                    // Gemma4 channel → think mapping
                    if (raw == "<|channel>") { emitter.emit_token("<think>"); continue; }
                    if (raw == "<channel|>") { emitter.emit_token("</think>\n"); continue; }
                    // Qwen3.6 thinking tokens (id 248068 / 248069) — must
                    // forward as text so the emitter transitions
                    // reasoning→content. Without this the generic <...>
                    // strip below drops them silently, leaving content
                    // empty and the model's whole answer wedged in
                    // reasoning_content. Mirrors the streaming-path fix
                    // above.
                    if (raw == "<think>") { emitter.emit_token("<think>"); continue; }
                    if (raw == "</think>") { emitter.emit_token("</think>\n"); continue; }
                    if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') continue;
                    if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>') {
                        if (!(raw.size() == 6 && raw[1] == '0' && raw[2] == 'x'))
                            continue;
                    }
                    std::string text = tokenizer_.token_text(tok);
                    emitter.emit_token(text);
                    if (emitter.stop_hit()) return false;
                }
                return true;
            };

            feed_tokens(result.tokens);
            const int total_completion_tokens = (int)result.tokens.size();
            emitter.emit_finish(total_completion_tokens);

            // Derive per-mode token counts from the emitter's REASONING
            // → CONTENT transition. first_content_token_index() returns
            // the emit_token index that first ran with mode == CONTENT;
            // tokens before that index were emitted while the emitter
            // was in REASONING (the `</think>`-carrying token itself
            // lands in REASONING and the NEXT token is the first
            // CONTENT). EOS/special tokens are skipped by feed_tokens
            // above, so emit_token_count() may be smaller than
            // result.tokens.size(); the remainder counts as
            // unattributed (e.g., TOOL_BUFFER).
            const int fci = emitter.first_content_token_index();
            const int emitted = emitter.emit_token_count();
            const int reasoning_tokens_emitted =
                fci < 0 ? emitted : fci;
            const int content_tokens_emitted =
                fci < 0 ? 0 : emitted - fci;

            json resp;
            switch (req.format) {
            case ApiFormat::OPENAI_CHAT: {
                json msg = {{"role", "assistant"}, {"content", emitter.accumulated_text()}};
                if (!emitter.reasoning_text().empty()) {
                    // Multi-dialect reasoning emission — same text, three keys.
                    // See docs/specs/thinking-budget.md "Response shape —
                    // multi-dialect aliasing".
                    //   reasoning_content : DeepSeek R1 / dflash primary
                    //   reasoning         : OpenRouter / Anthropic-gateway flat
                    //   reasoning_details : typed-block list; single block.
                    const std::string & rt = emitter.reasoning_text();
                    msg["reasoning_content"] = rt;
                    msg["reasoning"]         = rt;
                    msg["reasoning_details"] = json::array({
                        {{"type", "reasoning.text"}, {"text", rt}}
                    });
                }
                if (!emitter.tool_calls().empty()) {
                    json tcs = json::array();
                    for (const auto & tc : emitter.tool_calls()) {
                        tcs.push_back({{"id", tc.id}, {"type", "function"},
                                       {"function", {{"name", tc.name},
                                                     {"arguments", tc.arguments}}}});
                    }
                    msg["tool_calls"] = tcs;
                }
                // finish_reason: emitter only knows about "stop" / "tool_calls"
                // (EOS / tool-call detection). It can't see that the daemon
                // hit the n_gen cap. Compute "length" here from the
                // committed-token count vs the n_gen cap.
                // OpenAI/Anthropic clients (open-webui, Cline) gate retry
                // logic on finish_reason="length".
                std::string effective_finish_reason = emitter.finish_reason();
                if (effective_finish_reason == "stop") {
                    bool at_cap = (int)result.tokens.size() >= n_gen_cap;
                    if (at_cap) {
                        effective_finish_reason = "length";
                    }
                }
                if (result.degenerate_decode_close) {
                    effective_finish_reason = "length";
                }
                json choice = {
                    {"index", 0}, {"message", msg},
                    {"finish_reason", effective_finish_reason}
                };
                // finish_details — mirrors ds4_eval.c's eval_think_close_info.
                // Emitted when the caller opted in to the thinking-budget
                // envelope via `thinking:{type:enabled}`. close_kind reflects
                // whether the model self-closed the thinking block ("natural")
                // or the BudgetHook force-closed it at the budget boundary
                // ("hard"). See docs/specs/thinking-budget.md "v2 design".
                if (req.thinking_opt_in) {
                    // thinking_tokens / content_tokens come from the
                    // emitter's REASONING→CONTENT transition tracking;
                    // total_tokens is the raw committed-token count.
                    choice["finish_details"] = {
                        {"close_kind",      close_kind},
                        {"thinking_tokens", reasoning_tokens_emitted},
                        {"content_tokens",  content_tokens_emitted},
                        {"total_tokens",    total_completion_tokens},
                    };
                    // Honest signaling: when the post-close watchdog
                    // detected an n-gram repetition loop and aborted
                    // generation, surface a sibling flag so callers know
                    // the answer is unreliable. finish_reason stays
                    // "length" (SDK-safe per the truncation-signaling
                    // convention: OpenAI/Anthropic/Gemini all collapse
                    // budget-class events to one closed enum and put
                    // richer signal in sidecar fields).
                    if (result.degenerate_decode_close) {
                        choice["finish_details"]["degenerate_decode"] = true;
                    }
                }
                // usage.completion_tokens_details.reasoning_tokens — OpenAI
                // o1/o3 standard location, also OR's normalized shape. Mirrors
                // finish_details.thinking_tokens; kept in sync.
                // usage.timings — per-request prefill / decode wall clock
                // (always emitted; additive to OpenAI shape, ignored by
                // clients that don't recognize it). See spec §6.3.
                json chat_usage = {
                    {"prompt_tokens", (int)req.prompt_tokens.size()},
                    {"completion_tokens", total_completion_tokens},
                    {"total_tokens", (int)req.prompt_tokens.size() + total_completion_tokens},
                    {"completion_tokens_details", {
                        // Match finish_details.thinking_tokens
                        // (emitter-tracked split).
                        {"reasoning_tokens", reasoning_tokens_emitted}
                    }},
                    {"timings", build_timings_json(gen_timings, total_completion_tokens)},
                    {"accept_rate", result.accept_rate}
                };
                resp = {
                    {"id", req.response_id},
                    {"object", "chat.completion"},
                    {"created", std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()},
                    {"model", req.model},
                    {"choices", json::array({choice})},
                    {"usage", chat_usage}
                };
                break;
            }
            case ApiFormat::ANTHROPIC: {
                json content = json::array();
                if (!emitter.reasoning_text().empty()) {
                    content.push_back({{"type", "thinking"}, {"thinking", emitter.reasoning_text()}});
                }
                // Only emit a text block when there is actual text. When the
                // model emitted ONLY a tool_call (Qwen3 XML), accumulated_text
                // is empty — pushing an empty text block confuses Anthropic
                // SDK clients (they expect tool_use blocks alone).
                if (!emitter.accumulated_text().empty()) {
                    content.push_back({{"type", "text"}, {"text", emitter.accumulated_text()}});
                }
                // Tool calls — the OPENAI_CHAT branch above does this; the
                // ANTHROPIC branch was missing the tool_use serialisation,
                // so stop_reason="tool_use" was returned with empty content.
                // tc.arguments is a JSON-encoded string; parse to object for
                // Anthropic's `input` field (Anthropic expects object, not
                // string). Fall back to empty object on parse failure.
                if (!emitter.tool_calls().empty()) {
                    for (const auto & tc : emitter.tool_calls()) {
                        json input_obj;
                        try {
                            input_obj = tc.arguments.empty()
                                ? json::object()
                                : json::parse(tc.arguments);
                        } catch (const std::exception &) {
                            input_obj = json::object();
                        }
                        content.push_back({
                            {"type",  "tool_use"},
                            {"id",    tc.id},
                            {"name",  tc.name},
                            {"input", input_obj}
                        });
                    }
                }
                // stop_reason: Anthropic's analog of finish_reason. Same
                // length-vs-EOS distinction as OpenAI — Cline / Anthropic
                // SDK gate retry on stop_reason=="max_tokens".
                std::string anthropic_stop_reason;
                {
                    std::string er = emitter.finish_reason();
                    bool at_cap = (int)result.tokens.size() >= n_gen_cap;
                    if (er == "tool_calls") anthropic_stop_reason = "tool_use";
                    else if (at_cap)        anthropic_stop_reason = "max_tokens";
                    else                    anthropic_stop_reason = "end_turn";
                }
                json anth_usage = {
                    {"input_tokens", (int)req.prompt_tokens.size()},
                    {"output_tokens", total_completion_tokens},
                    {"timings", build_timings_json(gen_timings, total_completion_tokens)},
                    {"accept_rate", result.accept_rate}
                };
                resp = {
                    {"id", req.response_id}, {"type", "message"},
                    {"role", "assistant"}, {"model", req.model},
                    {"content", content},
                    {"stop_reason", anthropic_stop_reason},
                    {"usage", anth_usage}
                };
                break;
            }
            case ApiFormat::RESPONSES: {
                json output = json::array();
                if (!emitter.tool_calls().empty()) {
                    for (const auto & tc : emitter.tool_calls()) {
                        output.push_back({
                            {"type", "function_call"}, {"id", tc.id},
                            {"status", "completed"}, {"call_id", tc.id},
                            {"name", tc.name}, {"arguments", tc.arguments}
                        });
                    }
                } else {
                    output.push_back({
                        {"type", "message"}, {"id", req.response_id + "_msg"},
                        {"status", "completed"}, {"role", "assistant"},
                        {"content", json::array({{
                            {"type", "output_text"}, {"text", emitter.accumulated_text()},
                            {"annotations", json::array()}
                        }})}
                    });
                }
                json resp_usage = {
                    {"input_tokens", (int)req.prompt_tokens.size()},
                    {"output_tokens", total_completion_tokens},
                    {"total_tokens", (int)req.prompt_tokens.size() + total_completion_tokens},
                    {"timings", build_timings_json(gen_timings, total_completion_tokens)},
                    {"accept_rate", result.accept_rate}
                };
                resp = {
                    {"id", req.response_id}, {"object", "response"},
                    {"status", "completed"}, {"model", req.model},
                    {"output", output},
                    {"usage", resp_usage}
                };
                break;
            }
            default:
                resp = {{"text", emitter.accumulated_text()}};
            }
            // Set socket back to blocking for the final send.
            int flags = fcntl(fd, F_GETFL, 0);
            if (flags >= 0) fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
            send_response(fd, 200, "application/json", resp.dump() + "\n");
        }

        if (client_disconnected) {
            std::fprintf(stderr, "[server] client disconnected — generation aborted "
                         "(prompt=%zu out=%d)\n",
                         req.prompt_tokens.size(), completion_tokens);
        }

        const auto done_at = std::chrono::steady_clock::now();
        const double elapsed_s =
            std::chrono::duration<double>(done_at - started_at).count();
        const int result_tokens = (int)result.tokens.size();
        const int out_tokens = std::max(completion_tokens, result_tokens);
        const double tok_s = elapsed_s > 0.0 ? out_tokens / elapsed_s : 0.0;
        const double decode_tok_s =
            result.decode_s > 0.0 ? out_tokens / result.decode_s : 0.0;
        const std::string finish = client_disconnected
            ? "client_disconnect"
            : (result.ok ? emitter.finish_reason() : "error");

        std::fprintf(stderr,
            "[server] chat DONE %s ok=%s in=%zu effective_in=%zu out=%d "
            "%.1fs %.1f tok/s finish=%s restore=%s slot=%d prefix_len=%d "
            "prefill=%.1fs decode=%.1fs(%.1ftok/s) error=%s\n",
            req.response_id.c_str(),
            result.ok ? "true" : "false",
            req.prompt_tokens.size(),
            effective_prompt.size(),
            out_tokens,
            elapsed_s,
            tok_s,
            finish.c_str(),
            using_restore ? "true" : "false",
            cache_slot,
            prefix_len,
            result.prefill_s,
            result.decode_s,
            decode_tok_s,
            result.error.empty() ? "-" : result.error.c_str());

        // Signal client thread that we're done.
        finish_job();
    }
}

// ─── Job queue ──────────────────────────────────────────────────────────

void HttpServer::enqueue(ServerJob * job) {
    std::lock_guard<std::mutex> lk(queue_mu_);
    if (stopping_.load()) {
        // Server is shutting down — immediately signal job as done.
        std::lock_guard<std::mutex> jlk(job->mu);
        job->done = true;
        job->cv.notify_one();
        return;
    }
    job->next = nullptr;
    if (queue_tail_) queue_tail_->next = job;
    else queue_head_ = job;
    queue_tail_ = job;
    queue_cv_.notify_one();
}

ServerJob * HttpServer::dequeue() {
    std::unique_lock<std::mutex> lk(queue_mu_);
    // Use timed wait so the worker periodically wakes to send SSE heartbeats.
    while (!queue_head_ && !stopping_.load()) {
        if (queue_cv_.wait_for(lk, std::chrono::seconds(30)) == std::cv_status::timeout) {
            // Send SSE heartbeat (comment line) to detect disconnected clients.
            lk.unlock();
            sse_heartbeat();
            lk.lock();
        }
    }
    if (!queue_head_) return nullptr;
    ServerJob * j = queue_head_;
    queue_head_ = j->next;
    if (!queue_head_) queue_tail_ = nullptr;
    j->next = nullptr;
    return j;
}

// ─── HTTP I/O ───────────────────────────────────────────────────────────

bool HttpServer::read_http_request(int fd, HttpRequest & out) {
    std::string buf;
    buf.reserve(8192);
    char tmp[4096];

    // Read until we find the header/body boundary (\r\n\r\n or \n\n).
    ssize_t hend = -1;
    while (hend < 0 && buf.size() < 65536) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        buf.append(tmp, n);

        // Look for end of headers.
        for (size_t i = 3; i < buf.size(); i++) {
            if (buf[i-3] == '\r' && buf[i-2] == '\n' &&
                buf[i-1] == '\r' && buf[i] == '\n') {
                hend = i + 1;
                break;
            }
        }
        if (hend < 0) {
            for (size_t i = 1; i < buf.size(); i++) {
                if (buf[i-1] == '\n' && buf[i] == '\n') {
                    hend = i + 1;
                    break;
                }
            }
        }
    }
    if (hend < 0) return false;

    // Parse request line.
    size_t line_end = buf.find('\n');
    if (line_end == std::string::npos) return false;
    std::string line = buf.substr(0, line_end);
    if (!line.empty() && line.back() == '\r') line.pop_back();

    // "METHOD /path HTTP/1.1"
    size_t sp1 = line.find(' ');
    size_t sp2 = line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    out.method = line.substr(0, sp1);
    out.path = line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Separate query string from path.
    std::string query_string;
    size_t q = out.path.find('?');
    if (q != std::string::npos) {
        query_string = out.path.substr(q + 1);
        out.path = out.path.substr(0, q);
    }
    out.query = std::move(query_string);

    // Find Content-Length.
    long content_length = 0;
    {
        std::string headers = buf.substr(0, hend);
        std::string lower_headers = headers;
        std::transform(lower_headers.begin(), lower_headers.end(),
                       lower_headers.begin(), ::tolower);
        size_t cl_pos = lower_headers.find("content-length:");
        if (cl_pos != std::string::npos) {
            size_t val_start = cl_pos + 15;
            while (val_start < lower_headers.size() &&
                   lower_headers[val_start] == ' ') val_start++;
            content_length = std::strtol(headers.c_str() + val_start, nullptr, 10);
        }
    }

    if (content_length < 0 || content_length > 64 * 1024 * 1024) return false;

    // Read body.
    while ((ssize_t)buf.size() < hend + content_length) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        buf.append(tmp, n);
    }

    out.body = buf.substr(hend, content_length);
    return true;
}

bool HttpServer::send_all(int fd, const void * data, size_t len) {
    const char * p = (const char *)data;
    size_t sent = 0;
    // Stall deadline resets on each successful write (ds4 pattern).
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (sent < len) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) return false;  // stall timeout

        struct pollfd pfd = {fd, POLLOUT, 0};
        int timeout = remaining > 50 ? 50 : (int)remaining;
        int ret;
        do {
            ret = poll(&pfd, 1, timeout);
        } while (ret < 0 && errno == EINTR);
        if (ret < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
        if (ret == 0) continue;  // poll timeout, retry until deadline

        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return false;  // EPIPE, ECONNRESET, etc.
        }
        sent += n;
        deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    }
    return true;
}

bool HttpServer::send_response(int fd, int status, const std::string & content_type,
                               const std::string & body) {
    const char * reason = "OK";
    switch (status) {
        case 200: reason = "OK"; break;
        case 204: reason = "No Content"; break;
        case 400: reason = "Bad Request"; break;
        case 404: reason = "Not Found"; break;
        case 405: reason = "Method Not Allowed"; break;
        case 413: reason = "Payload Too Large"; break;
        case 500: reason = "Internal Server Error"; break;
        case 503: reason = "Service Unavailable"; break;
    }
    std::string header = "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\n";
    if (config_.enable_cors) {
        header += "Access-Control-Allow-Origin: *\r\n"
                  "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                  "Access-Control-Allow-Headers: *\r\n";
    }
    if (!content_type.empty()) {
        header += "Content-Type: " + content_type + "\r\n";
    }
    header += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    header += "Connection: close\r\n\r\n";
    header += body;
    return send_all(fd, header.data(), header.size());
}

bool HttpServer::send_error(int fd, int status, const std::string & message) {
    json err = {{"error", {{"message", message}, {"type", "invalid_request_error"}}}};
    return send_response(fd, status, "application/json", err.dump() + "\n");
}

bool HttpServer::send_sse_headers(int fd) {
    std::string header = "HTTP/1.1 200 OK\r\n";
    if (config_.enable_cors) {
        header += "Access-Control-Allow-Origin: *\r\n";
    }
    header += "Content-Type: text/event-stream\r\n"
              "Cache-Control: no-cache\r\n"
              "Connection: keep-alive\r\n\r\n";
    return send_all(fd, header.data(), header.size());
}

}  // namespace dflash::common
