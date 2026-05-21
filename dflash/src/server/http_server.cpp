// HTTP server implementation.
//
// Core infrastructure: socket listen/accept, client threads, HTTP parsing,
// job queue, worker thread with SSE streaming and disconnect detection.

#include "http_server.h"
#include "sse_emitter.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

namespace dflash27b {

// ─── Utilities ──────────────────────────────────────────────────────────

static std::string generate_id(const char * prefix) {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%016llx",
                  prefix, (unsigned long long)counter.fetch_add(1));
    return buf;
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
    disk_cache_.init();
}

HttpServer::~HttpServer() {
    shutdown();
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

    std::fprintf(stderr, "[server] listening on http://%s:%d\n",
                 config_.host.c_str(), config_.port);

    // Start worker thread.
    worker_thread_ = std::thread([this]() { worker_loop(); });

    // Accept loop.
    while (!stopping_.load()) {
        struct sockaddr_in client_sa{};
        socklen_t client_len = sizeof(client_sa);
        int client_fd = accept(listen_fd_, (struct sockaddr *)&client_sa, &client_len);
        if (client_fd < 0) {
            if (stopping_.load()) break;
            if (errno == EINTR) continue;
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

    // Wait for all client threads to finish.
    {
        std::unique_lock<std::mutex> lk(clients_mu_);
        clients_cv_.wait(lk, [this]() { return active_clients_.load() == 0; });
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
                     {"supported_reasoning_levels", json::array({
                         {{"effort", "low"}, {"description", "No thinking"}},
                         {{"effort", "medium"}, {"description", "Thinking enabled"}},
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

    ParsedRequest req;
    std::string err;

    try {
        json body = json::parse(hr.body);

        // Common fields.
        req.stream = body.value("stream", false);
        req.model = body.value("model", config_.model_name);
        req.max_output = body.value("max_tokens",
                         body.value("max_output_tokens",
                         body.value("max_completion_tokens", config_.max_tokens)));

        // Sampler parameters.
        req.sampler.temp = body.value("temperature", 0.0f);
        req.sampler.top_p = body.value("top_p", 1.0f);
        req.sampler.top_k = body.value("top_k", 0);
        if (body.contains("seed")) {
            req.sampler.seed = body["seed"].get<uint64_t>();
        }

        // Tools.
        if (body.contains("tools")) {
            req.tools = body["tools"];
        }

        if (hr.path == "/v1/chat/completions") {
            req.format = ApiFormat::OPENAI_CHAT;
            req.response_id = generate_id("chatcmpl");
            req.messages = body["messages"];
        } else if (hr.path == "/v1/messages") {
            req.format = ApiFormat::ANTHROPIC;
            req.response_id = generate_id("msg");
            req.messages = body["messages"];
            if (body.contains("system")) {
                // Anthropic puts system as a top-level field.
                // Strip billing header blocks injected by Claude Code.
                json sys_content = body["system"];
                if (sys_content.is_array()) {
                    json filtered = json::array();
                    for (const auto & block : sys_content) {
                        if (block.is_object() && block.value("type", "") == "text") {
                            std::string text = block.value("text", "");
                            if (text.rfind("x-anthropic-billing-header:", 0) == 0) {
                                continue;  // skip billing header block
                            }
                        }
                        filtered.push_back(block);
                    }
                    sys_content = std::move(filtered);
                } else if (sys_content.is_string()) {
                    std::string s = sys_content.get<std::string>();
                    if (s.rfind("x-anthropic-billing-header:", 0) == 0) {
                        sys_content = "";
                    }
                }
                if (!sys_content.empty()) {
                    json sys_msg = {{"role", "system"}, {"content", sys_content}};
                    req.messages.insert(req.messages.begin(), sys_msg);
                }
            }
        } else if (hr.path == "/v1/responses") {
            req.format = ApiFormat::RESPONSES;
            req.response_id = generate_id("resp");
            // Responses API uses "input" instead of "messages".
            if (body.contains("input")) {
                req.messages = body["input"];
            }
            if (body.contains("instructions")) {
                json sys_msg = {{"role", "system"}, {"content", body["instructions"]}};
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
        std::vector<ChatMessage> chat_msgs;
        if (req.messages.is_array()) {
            for (const auto & m : req.messages) {
                ChatMessage cm;
                cm.role = m.value("role", "user");

                // Check for tool memory replay on assistant messages with tool_calls.
                bool replayed = false;
                if (cm.role == "assistant" && m.contains("tool_calls") &&
                    m["tool_calls"].is_array() && !m["tool_calls"].empty()) {
                    // Extract call IDs for tool memory lookup.
                    std::vector<std::string> call_ids;
                    for (const auto & tc : m["tool_calls"]) {
                        std::string id = tc.value("id", "");
                        if (!id.empty()) call_ids.push_back(id);
                    }
                    std::string raw = tool_memory_.lookup(call_ids);
                    if (!raw.empty()) {
                        cm.content = raw;
                        replayed = true;
                    }
                }

                if (!replayed) {
                    if (m.contains("content") && m["content"].is_string()) {
                        cm.content = m["content"].get<std::string>();
                    } else if (m.contains("content") && m["content"].is_array()) {
                        // Multi-part content (text parts only for now).
                        for (const auto & part : m["content"]) {
                            std::string ptype = part.value("type", "");
                            if (ptype == "text" || ptype == "input_text") {
                                cm.content += part.value("text", "");
                            }
                        }
                    }
                }
                chat_msgs.push_back(std::move(cm));
            }
        } else if (req.messages.is_string()) {
            // Simple string input (Responses API shorthand).
            chat_msgs.push_back({"user", req.messages.get<std::string>()});
        }

        // Determine thinking mode BEFORE rendering so the template can inject
        // the <think>\n\n</think>\n\n block when thinking is disabled.
        // Default: thinking OFF (matches server.py — Qwen3.6 thinking wrecks
        // DFlash acceptance rates; clients opt in explicitly).
        bool enable_thinking = false;

        // OpenAI Responses API: "reasoning" field
        if (body.contains("reasoning")) {
            auto & r = body["reasoning"];
            if (r.contains("effort")) {
                std::string effort = r.value("effort", "low");
                enable_thinking = (effort != "low");
            } else {
                enable_thinking = true;
            }
        }
        // Anthropic: "thinking" field
        if (body.contains("thinking")) {
            auto & th = body["thinking"];
            if (th.contains("type")) {
                enable_thinking = (th.value("type", "") == "enabled");
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

        // Serialize tools JSON for template injection.
        std::string tools_json;
        if (req.tools.is_array() && !req.tools.empty()) {
            tools_json = req.tools.dump();
        }

        std::string rendered = render_chat_template(chat_msgs, chat_format_,
                                                    true, enable_thinking,
                                                    tools_json);
        req.prompt_tokens = tokenizer_.encode(rendered);

        // Detect if prompt ends with <think> (model will start in reasoning mode).
        if (enable_thinking) {
            size_t end = rendered.size();
            while (end > 0 && (rendered[end-1] == ' ' || rendered[end-1] == '\n' ||
                   rendered[end-1] == '\r' || rendered[end-1] == '\t'))
                end--;
            if (end >= 7 && rendered.compare(end - 7, 7, "<think>") == 0) {
                req.started_in_thinking = true;
            }
        }

    } catch (const std::exception & e) {
        send_error(fd, 400, std::string("JSON parse error: ") + e.what());
        return true;  // handled (with error)
    }

    // Check context length.
    if ((int)req.prompt_tokens.size() + req.max_output > config_.max_ctx) {
        send_error(fd, 400, "prompt + max_tokens exceeds context window");
        return true;
    }

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

        // Send SSE headers.
        if (req.stream) {
            if (!send_sse_headers(fd)) {
                // Client already disconnected before we started.
                std::lock_guard<std::mutex> lk(job->mu);
                job->done = true;
                job->cv.notify_one();
                continue;
            }
        }

        // Create SSE emitter for streaming state machine.
        SseEmitter emitter(req.format, req.response_id, req.model,
                           (int)req.prompt_tokens.size(), req.tools,
                           &tool_memory_, req.started_in_thinking);

        // Emit initial SSE events.
        if (req.stream) {
            bool start_ok = true;
            for (const auto & chunk : emitter.emit_start()) {
                if (!send_all(fd, chunk.data(), chunk.size())) {
                    start_ok = false;
                    break;
                }
            }
            if (!start_ok) {
                std::lock_guard<std::mutex> lk(job->mu);
                job->done = true;
                job->cv.notify_one();
                continue;
            }
        }

        // ── PFlash speculative prefill compression ────────────────────
        // If pflash is enabled and prompt exceeds threshold, compress.
        std::vector<int32_t> effective_prompt = req.prompt_tokens;
        bool pflash_compressed = false;

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

            if (should_compress) {
                // Check full-compress cache FIRST — if we've seen this exact
                // raw prompt before, skip the expensive compress cycle entirely.
                auto [full_slot, full_len] = prefix_cache_.lookup_full(req.prompt_tokens);
                if (full_slot >= 0) {
                    std::fprintf(stderr, "[pflash] full-cache hit slot=%d — skipping compress\n", full_slot);
                    pflash_compressed = true;
                    // effective_prompt stays as req.prompt_tokens — the cached KV
                    // state will be restored via cache_slot below.
                } else {
                    // 1. Decode prompt to text using target tokenizer
                    std::string prompt_text = tokenizer_.decode(req.prompt_tokens);

                    // 2. Re-encode with drafter tokenizer
                    auto drafter_ids = drafter_tokenizer_->encode(prompt_text);

                    if (!drafter_ids.empty()) {
                        // 3. Compress via typed API
                        ModelBackend::CompressRequest creq;
                        creq.input_ids = std::move(drafter_ids);
                        creq.keep_ratio = config_.pflash_keep_ratio;
                        creq.drafter_path = config_.pflash_drafter_path;
                        creq.skip_park = config_.pflash_skip_park;

                        auto cresult = backend_.compress(creq);

                        // 4. Decode compressed IDs with drafter tokenizer
                        if (cresult.ok && !cresult.compressed_ids.empty()) {
                            std::string compressed_text =
                                drafter_tokenizer_->decode(cresult.compressed_ids);

                            // 5. Re-tokenize with target tokenizer
                            effective_prompt = tokenizer_.encode(compressed_text);
                            pflash_compressed = true;

                            std::fprintf(stderr,
                                "[pflash] %d -> %d -> %d tokens (%.1f%% kept)\n",
                                n_prompt, (int)cresult.compressed_ids.size(),
                                (int)effective_prompt.size(),
                                100.0 * effective_prompt.size() / n_prompt);
                        }
                    }
                }
            }
        }

        // Build generate request.
        GenerateRequest gen_req;
        gen_req.prompt = effective_prompt;
        gen_req.n_gen = req.max_output;
        gen_req.sampler = req.sampler;
        gen_req.do_sample = req.sampler.temp > 0.0f;
        gen_req.stream = false;  // we handle streaming via on_token callback

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
        if (!using_restore && !disk_cache_.disabled()) {
            if (disk_cache_.lookup(effective_prompt, DISK_STAGING_SLOT)) {
                cache_slot = DISK_STAGING_SLOT;
                prefix_len = backend_.snapshot_cur_pos(DISK_STAGING_SLOT);
                using_restore = true;
                disk_hit = true;
                std::fprintf(stderr, "[disk-cache] hit, loaded to slot=%d pos=%d\n",
                             DISK_STAGING_SLOT, prefix_len);
            }
        }

        // Cold prefix save: for long prompts with no cache hit, prefill to a
        // turn boundary and save a cold checkpoint before the full generation.
        // This makes subsequent requests to similar (but not identical) prompts
        // much faster by reusing the cold prefix.
        if (!using_restore && !disk_cache_.disabled()) {
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

        // Set up DaemonIO with on_token callback for streaming + disconnect.
        DaemonIO io;
        io.stream_fd = -1;  // no pipe — we write SSE directly

        int completion_tokens = 0;
        bool client_disconnected = false;

        io.on_token = [&](int32_t token) -> bool {
            if (client_disconnected) return false;
            completion_tokens++;

            // Skip EOS/EOT/special tokens — don't forward to SSE.
            int32_t eos = tokenizer_.eos_id();
            if (token == eos) return true;
            // Also skip common Qwen3 special tokens by checking if the raw
            // token text starts with '<|' (e.g. <|im_end|>, <|im_start|>).
            const std::string & raw = tokenizer_.raw_token(token);
            if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') return true;

            std::string text = tokenizer_.token_text(token);

            if (req.stream && !text.empty()) {
                auto chunks = emitter.emit_token(text);
                for (const auto & chunk : chunks) {
                    if (!send_all(fd, chunk.data(), chunk.size())) {
                        client_disconnected = true;
                        return false;
                    }
                }
            }
            return true;
        };

        // Run generation (with or without restore).
        GenerateResult result;
        if (using_restore) {
            result = backend_.restore_and_generate(cache_slot, gen_req, io);
        } else {
            result = backend_.generate(gen_req, io);
        }

        // Confirm or abort the inline snapshot.
        if (snap_prepared) {
            if (completion_tokens > 0 && !client_disconnected) {
                prefix_cache_.confirm_inline_snap(snap_slot, snap_cut, effective_prompt);
                // Track for shutdown save.
                slot_tokens_[snap_slot] = std::vector<int32_t>(
                    effective_prompt.begin(), effective_prompt.begin() + snap_cut);
                // Save to disk cache if threshold met.
                if (!disk_cache_.disabled()) {
                    disk_cache_.learn_layout(snap_slot);
                    disk_cache_.save(snap_slot, effective_prompt);
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
        if (!disk_cache_.disabled() && result.ok && completion_tokens > 0 && !client_disconnected) {
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

        // Full-compress cache: reserve + confirm after successful generation.
        if (pflash_compressed && completion_tokens > 0 && !client_disconnected) {
            int full_slot = prefix_cache_.prepare_full_snap(req.prompt_tokens);
            if (full_slot >= 0) {
                prefix_cache_.confirm_full_snap(full_slot, req.prompt_tokens,
                                                (int)effective_prompt.size());
            }
        }

        // Finalize.
        if (req.stream && !client_disconnected) {
            auto final_chunks = emitter.emit_finish(completion_tokens);
            for (const auto & chunk : final_chunks) {
                if (!send_all(fd, chunk.data(), chunk.size())) {
                    client_disconnected = true;
                    break;
                }
            }
        } else if (!req.stream && !client_disconnected) {
            // Non-streaming: build complete response using emitter state.
            // Feed all tokens through emitter (skip specials like streaming path).
            for (int32_t tok : result.tokens) {
                const std::string & raw = tokenizer_.raw_token(tok);
                if (tok == tokenizer_.eos_id()) continue;
                if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') continue;
                std::string text = tokenizer_.token_text(tok);
                emitter.emit_token(text);
            }
            emitter.emit_finish((int)result.tokens.size());

            json resp;
            switch (req.format) {
            case ApiFormat::OPENAI_CHAT: {
                json msg = {{"role", "assistant"}, {"content", emitter.accumulated_text()}};
                if (!emitter.reasoning_text().empty()) {
                    msg["reasoning_content"] = emitter.reasoning_text();
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
                resp = {
                    {"id", req.response_id},
                    {"object", "chat.completion"},
                    {"created", std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()},
                    {"model", req.model},
                    {"choices", json::array({{
                        {"index", 0}, {"message", msg},
                        {"finish_reason", emitter.finish_reason()}
                    }})},
                    {"usage", {
                        {"prompt_tokens", (int)req.prompt_tokens.size()},
                        {"completion_tokens", (int)result.tokens.size()},
                        {"total_tokens", (int)(req.prompt_tokens.size() + result.tokens.size())}
                    }}
                };
                break;
            }
            case ApiFormat::ANTHROPIC: {
                json content = json::array();
                if (!emitter.reasoning_text().empty()) {
                    content.push_back({{"type", "thinking"}, {"thinking", emitter.reasoning_text()}});
                }
                content.push_back({{"type", "text"}, {"text", emitter.accumulated_text()}});
                resp = {
                    {"id", req.response_id}, {"type", "message"},
                    {"role", "assistant"}, {"model", req.model},
                    {"content", content},
                    {"stop_reason", emitter.finish_reason() == "stop" ? "end_turn" : "tool_use"},
                    {"usage", {
                        {"input_tokens", (int)req.prompt_tokens.size()},
                        {"output_tokens", (int)result.tokens.size()}
                    }}
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
                resp = {
                    {"id", req.response_id}, {"object", "response"},
                    {"status", "completed"}, {"model", req.model},
                    {"output", output},
                    {"usage", {
                        {"input_tokens", (int)req.prompt_tokens.size()},
                        {"output_tokens", (int)result.tokens.size()},
                        {"total_tokens", (int)(req.prompt_tokens.size() + result.tokens.size())}
                    }}
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

        // Signal client thread that we're done.
        {
            std::lock_guard<std::mutex> lk(job->mu);
            job->done = true;
            job->cv.notify_one();
        }
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
    queue_cv_.wait(lk, [this]() { return queue_head_ != nullptr || stopping_.load(); });
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

}  // namespace dflash27b
