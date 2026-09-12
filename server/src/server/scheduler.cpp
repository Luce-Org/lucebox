// Concurrent scheduler for --max-concurrency serving: the worker thread's
// iteration-level loop over a backend's SeqEngine decode slots.
//
// Split from http_server.cpp: this TU owns non-blocking admission (one
// prefill chunk per engine step, fused with the live decode batch), FIFO
// pool-full deferrals, per-slot streaming through ClientSendBuffer, and
// retirement. SSE emission, error-close chunks, and HTTP response
// formatting are shared with the classic worker so both paths emit
// matching wire formats.

#include "http_server.h"
#include "common/concurrency/seq_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace dflash::common {

namespace {

// Per-slot request state for the iteration-level scheduler. Indexed by the
// engine slot id returned from admit(), so scheduler and engine agree on
// which engine-owned state record a request owns. This remains the one
// external phase: sockets stay here, prompt/KV/sampler/progress stay in Qwen.
struct SchedSlot {
    ServerJob * job = nullptr;
    SocketHandle fd = kInvalidSocket;
    std::unique_ptr<SseEmitter> emitter;
    bool prefilling = false;
    uint64_t admission_order = 0;
    std::chrono::steady_clock::time_point started_at{};
    std::chrono::steady_clock::time_point decode_started_at{};
    double prefill_s = 0.0;
    int n_gen_cap = 0;
    int completion_tokens = 0;
    bool client_disconnected = false;
    bool failed = false;
    std::string error;
    bool finished = false;
    std::vector<int32_t> gen_tokens;   // committed + pending, in order
    int32_t pending_tok = -1;          // sampled, fed back next step
    // Buffered client output (see client_send_buffer.h): chunks append here and
    // a non-blocking flush runs every scheduler iteration, so one slow
    // reader can never head-of-line-block the shared decode loop.
    dflash::common::ClientSendBuffer send_buffer;
    // Thinking-budget force-close, applied scheduler-side before the token
    // is fed back (mirrors do_ar_decode's maybe_force_close).
    dflash::common::BudgetHook hook;
    bool hook_started = false;
    int  hook_pos = 0;
    bool budget_forced_close = false;
    bool degenerate_close = false;
};

// Outcome of one admission attempt. The three cases differ in who owns the
// job afterwards: Admitted hands it to a slot, Deferred hands it back to the
// caller to retry at the head of the line, Retired means the job is already
// answered and holds nothing.
enum class AdmissionDisposition {
    Admitted,  // job owns an engine slot and its first token is on the wire
    Deferred,  // engine had no room; caller keeps the job and retries it first
    Retired,   // job finished or failed during admission; no slot was taken
};

}  // namespace

void HttpServer::scheduler_loop(SeqEngine & engine) {
    const int n_slots = engine.slot_count();
    std::vector<SchedSlot> slots((size_t)n_slots);
    uint64_t next_request_id = 1;
    uint64_t next_admission_order = 0;
    // Admission-deferred job (pool blocks/slots exhausted). Kept at the head
    // of the line so FIFO order survives the deferral.
    ServerJob * deferred = nullptr;
    std::chrono::steady_clock::time_point deferred_retry_at{};

    // Degenerate-run guard shared with do_ar_decode: explicit env override,
    // else 32 when the min-tokens floor is active, else off.
    static const int repeat_guard = [] {
        if (const char * s = std::getenv("DFLASH_DEGENERATE_RUN_TOKENS")) {
            const int v = std::atoi(s);
            if (v >= 0) return v;
        }
        const char * f = std::getenv("DFLASH_MIN_TOKENS");
        return (f && std::atoi(f) > 0) ? 32 : 0;
    }();

    // Cached live-slot count — incremented on admit, decremented on retire.
    // Replaces the O(n_slots) scan that was called 2-3× per iteration.
    int live_slots = 0;

    int published_live_count = -1;
    int published_prefill_count = -1, published_suspended_count = -1;
    size_t published_offloaded_bytes = 0;
    const size_t offload_budget = config_.decode_kv_offload_bytes;
    auto suspended = [&](int slot) {
        return offload_budget && engine.kv_offload_state(slot).suspended;
    };
    auto publish_live_count = [&]() {
        int prefilling = 0, paused = 0;
        size_t bytes = 0;
        for (int i = 0; i < n_slots; ++i) {
            const SchedSlot & s = slots[(size_t)i];
            if (!s.job) continue;
            const auto state = offload_budget ? engine.kv_offload_state(i)
                                             : SeqEngine::KvOffloadState{};
            paused += state.suspended;
            bytes += state.bytes;
            if (s.prefilling && !state.suspended) ++prefilling;
        }
        if (live_slots == published_live_count &&
            prefilling == published_prefill_count &&
            paused == published_suspended_count && bytes == published_offloaded_bytes) return;
        published_live_count = live_slots;
        published_prefill_count = prefilling;
        published_suspended_count = paused;
        published_offloaded_bytes = bytes;
        if (live_slots > 0) {
            status_.set_concurrent_requests(live_slots, prefilling, paused, bytes);
        } else status_.set_idle();
        broadcast_status();
    };

    auto finish_job = [this](ServerJob * job) {
        stop_job_stream(job);
        std::lock_guard<std::mutex> lk(job->mu);
        job->done = true;
        job->cv.notify_one();
    };

    // A stalled reader may buffer at most this much before being dropped.
    constexpr size_t kMaxSlotSendBuffer = 1u << 20;
    constexpr auto kClientStallTimeout = std::chrono::seconds(30);

    // Final payloads of retired slots still draining to their sockets. The
    // job is signalled done only once its bytes are out (or the deadline /
    // hard error gives up) because the parked client thread closes the fd
    // the moment it wakes.
    struct DrainJob {
        ServerJob * job = nullptr;
        SocketHandle fd = kInvalidSocket;
        ClientSendBuffer send_buffer;
        std::chrono::steady_clock::time_point deadline{};
    };
    std::vector<DrainJob> drains;

    auto service_drains = [&]() {
        if (drains.empty()) return;
        const auto now = std::chrono::steady_clock::now();
        for (size_t i = 0; i < drains.size();) {
            DrainJob & d = drains[i];
            const size_t pending_before = d.send_buffer.pending();
            bool ok = false;
            if (d.job) {
                std::lock_guard<std::mutex> lock(d.job->write_mu);
                ok = d.send_buffer.flush(d.fd);
            } else {
                ok = d.send_buffer.flush(d.fd);
            }
            if (d.send_buffer.pending() < pending_before) {
                d.deadline = std::chrono::steady_clock::now() +
                             kClientStallTimeout;
            }
            if (!ok || d.send_buffer.empty() || now > d.deadline) {
                finish_job(d.job);
                drains[i] = std::move(drains.back());
                drains.pop_back();
                continue;
            }
            ++i;
        }
    };

    auto maybe_force_close = [](SchedSlot & s, int32_t & tok) {
        if (s.hook.close_token_ids.empty()) return;
        if (s.hook_started) {
            if (s.hook_pos < (int)s.hook.close_token_ids.size()) {
                tok = s.hook.close_token_ids[(size_t)s.hook_pos++];
            }
            return;
        }
        const int generated = (int)s.gen_tokens.size();
        const int remaining = s.n_gen_cap - generated;
        if (remaining <= s.hook.hard_limit_remaining) {
            const int32_t first_close = s.hook.close_token_ids.front();
            s.hook_started = true;
            s.hook_pos = 1;
            if (tok != first_close) {
                tok = first_close;
                s.budget_forced_close = true;
            }
        }
    };

    // Advances one slot by a single sampled token — the post-sample path
    // shared by the first (prefill-logits) token and every decode-step token.
    // Note `tok` is by value but not passthrough: maybe_force_close may
    // *substitute* a close token for it, and that substitute is what gets
    // recorded, emitted, and fed back. Appends to gen_tokens, streams the
    // delta into send_buffer, and parks the token in pending_tok as the next
    // step's input for this slot. Sets s.finished — but never retires the
    // slot — on EOS, gen cap, stop-sequence hit, or degenerate repetition.
    auto advance_slot = [&](SchedSlot & s, int32_t tok) {
        maybe_force_close(s, tok);
        s.gen_tokens.push_back(tok);
        const bool cont = deliver_generation_token(
            s.job, s.job->req, *s.emitter, tok, s.completion_tokens,
            s.send_buffer);
        s.pending_tok = tok;
        if (!cont || engine.token_is_eos(tok) ||
            (int)s.gen_tokens.size() >= s.n_gen_cap) {
            s.finished = true;
            return;
        }
        // Single-token run guard (matches do_ar_decode's repeat break).
        if (repeat_guard > 0 && (int)s.gen_tokens.size() >= repeat_guard) {
            int run = 1;
            for (int j = (int)s.gen_tokens.size() - 2; j >= 0; --j) {
                if (s.gen_tokens[(size_t)j] != tok) break;
                run++;
            }
            if (run >= repeat_guard) {
                std::fprintf(stderr,
                    "[parallel] token %d repeated %d times — stopping slot\n",
                    tok, run);
                s.degenerate_close = true;
                s.finished = true;
                return;
            }
        }
        // Post-close repetition watchdog (periods 12..80), mirrors
        // do_ar_decode's sweep once the close sequence has fully injected.
        if (s.hook_started &&
            s.hook_pos >= (int)s.hook.close_token_ids.size()) {
            const auto end = s.gen_tokens.end();
            const int avail = (int)s.gen_tokens.size();
            for (int P = 12; P <= 80; P++) {
                if (avail < 2 * P) break;
                if (std::equal(end - 2 * P, end - P, end - P)) {
                    std::fprintf(stderr,
                        "[parallel] post-close period=%d repeated — "
                        "stopping slot\n", P);
                    s.degenerate_close = true;
                    s.finished = true;
                    return;
                }
            }
        }
    };

    auto retire_slot = [&](int idx, bool backend_ok) {
        SchedSlot & s = slots[(size_t)idx];
        if (!s.job) return;
        const ParsedRequest & req = s.job->req;
        // Stop monitor-thread heartbeats before queuing terminal frames.
        stop_job_stream(s.job, &s.send_buffer);
        const double decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - s.decode_started_at).count();
        const int prompt_tokens = (int)req.prompt_tokens.size();
        GenTimings gen_timings{
            s.prefill_s,
            decode_s,
            /*cache_hit=*/false,
            /*cached_prefix_tokens=*/0,
            /*prefilled_tokens=*/prompt_tokens,
            /*effective_prompt_tokens=*/prompt_tokens,
        };

        if (backend_ok && !s.failed) {
            PerfRecord perf;
            perf.prompt_tokens = (int)req.prompt_tokens.size();
            perf.completion_tokens = s.completion_tokens;
            perf.prefill_tok_s = s.prefill_s > 0.0
                ? (double)req.prompt_tokens.size() / s.prefill_s : 0.0;
            perf.decode_tok_s = decode_s > 0.0
                ? (double)s.completion_tokens / decode_s : 0.0;
            status_.record_perf(perf);
        }

        if (s.failed || !backend_ok) {
            const std::string message =
                s.error.empty() ? "generation failed" : s.error;
            if (!s.client_disconnected) {
                if (req.stream) {
                    for (const std::string & chunk :
                         sse_error_close_chunks(message)) {
                        s.send_buffer.append(chunk);
                    }
                } else {
                    json err = {{"error", {{"message", message},
                                           {"type", "invalid_request_error"}}}};
                    s.send_buffer.append(format_http_response(
                        500, "application/json", err.dump() + "\n"));
                }
            }
        } else if (req.stream && !s.client_disconnected) {
            const bool is_eos = !s.gen_tokens.empty() &&
                engine.token_is_eos(s.gen_tokens.back());
            auto final_chunks =
                s.emitter->emit_finish(
                    s.completion_tokens, &gen_timings,
                    s.n_gen_cap, is_eos);
            for (const auto & chunk : final_chunks) {
                s.send_buffer.append(chunk);
            }
        } else if (!req.stream && !s.client_disconnected) {
            send_nonstream_response(req, s.fd, *s.emitter, s.gen_tokens,
                                    s.n_gen_cap, s.budget_forced_close,
                                    s.degenerate_close, gen_timings,
                                    &s.send_buffer);
        }

        const double elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - s.started_at).count();
        const int out_tokens = (int)s.gen_tokens.size();
        std::fprintf(stderr,
            "[server] chat DONE %s ok=%s in=%zu out=%d %.1fs %.1f tok/s "
            "finish=%s slot=%d prefill=%.1fs decode=%.1fs(%.1ftok/s) parallel\n",
            req.response_id.c_str(),
            (!s.failed && backend_ok) ? "true" : "false",
            req.prompt_tokens.size(), out_tokens, elapsed_s,
            elapsed_s > 0.0 ? out_tokens / elapsed_s : 0.0,
            s.client_disconnected ? "client_disconnect"
                                  : s.emitter->finish_reason().c_str(),
            idx, s.prefill_s, decode_s,
            decode_s > 0.0 ? out_tokens / decode_s : 0.0);

        engine.retire(idx);
        // A retirement may have released the blocks the head job needs.
        deferred_retry_at = {};

        // Hand any undrained bytes to the drain list; the job stays parked
        // until they are out (or the drain gives up).
        bool drained = s.client_disconnected;
        if (!drained) {
            if (s.job) {
                std::lock_guard<std::mutex> lock(s.job->write_mu);
                drained = s.send_buffer.flush(s.fd) ? s.send_buffer.empty() : true;
            } else {
                drained = s.send_buffer.flush(s.fd) ? s.send_buffer.empty() : true;
            }
        }
        if (drained) {
            finish_job(s.job);
        } else if (s.send_buffer.pending() > kMaxSlotSendBuffer) {
            // Non-streaming output is materialized only at retirement, after
            // the live-slot cap check.  Do not park an oversized final
            // response in drains when the client cannot accept it now.
            std::fprintf(stderr,
                "[parallel] slot %d final response exceeds client buffer "
                "cap -- dropping response\n", idx);
            finish_job(s.job);
        } else {
            DrainJob d;
            d.job = s.job;
            d.fd = s.fd;
            d.send_buffer = std::move(s.send_buffer);
            d.deadline = std::chrono::steady_clock::now() +
                         kClientStallTimeout;
            drains.push_back(std::move(d));
        }
        s = SchedSlot{};
        live_slots--;
        publish_live_count();
    };

    auto admit_job = [&](ServerJob * job) -> AdmissionDisposition {
        const ParsedRequest & req = job->req;
        if (job->parallel_started_at ==
                std::chrono::steady_clock::time_point{}) {
            job->parallel_started_at = std::chrono::steady_clock::now();
        }
        const auto started_at = job->parallel_started_at;

        // Same thinking-budget n_gen math as the classic worker loop.
        const bool budget_active = req.thinking_opt_in;
        const int effective_think_ceiling = (req.per_req_phase1_cap >= 0)
            ? req.per_req_phase1_cap
            : config_.think_max_tokens;
        const int eff_reply_for_n_gen = (req.per_req_reply_budget >= 0)
            ? req.per_req_reply_budget
            : config_.hard_limit_reply_budget;
        const int n_gen_cap = budget_active
            ? (std::min)(effective_think_ceiling + eff_reply_for_n_gen,
                         req.max_output)
            : req.max_output;

        if (n_gen_cap < 1) {
            // Degenerate ask: reply with an empty completion, no slot needed.
            SseEmitter emitter(req.format, req.response_id, req.model,
                               (int)req.prompt_tokens.size(), req.tools,
                               &tool_memory_, req.stop_sequences,
                               req.started_in_thinking);
            GenTimings t{
                0.0,
                0.0,
                /*cache_hit=*/false,
                /*cached_prefix_tokens=*/0,
                /*prefilled_tokens=*/0,
                /*effective_prompt_tokens=*/(int)req.prompt_tokens.size(),
            };
            if (req.stream) {
                if (send_sse_headers(job)) {
                    bool ok = true;
                    for (const auto & c : emitter.emit_start()) {
                        if (!send_job_bytes(job, c.data(), c.size())) { ok = false; break; }
                    }
                    if (ok) {
                        for (const auto & c : emitter.emit_finish(0, &t, n_gen_cap)) {
                            if (!send_job_bytes(job, c.data(), c.size())) break;
                        }
                    }
                }
            } else {
                send_nonstream_response(req, job->fd, emitter, {}, n_gen_cap,
                                        false, false, t);
            }
            finish_job(job);
            return AdmissionDisposition::Retired;
        }

        if (!job->announced) {
            job->announced = true;
            std::fprintf(stderr,
                "[server] chat START %s format=%s stream=%s prompt_tokens=%zu "
                "max_tokens=%d live=%d parallel\n",
                req.response_id.c_str(), api_format_name(req.format),
                req.stream ? "true" : "false", req.prompt_tokens.size(),
                req.max_output, live_slots);
        }

        // Admission only claims the slot and queues the prompt. Prefill
        // advances one chunk per engine step alongside live decode.
        auto ar = engine.admit(next_request_id, req.prompt_tokens,
                               req.sampler);
        if (ar.status == SeqEngine::AdmitResult::Status::capacity_exceeded) {
            if (job->report_admission) {
                job->admission = RoutingAdmission::unfit;
            } else {
                send_error(job->fd, 400, "admission failed: " + ar.error);
            }
            finish_job(job);
            return AdmissionDisposition::Retired;
        }
        if (ar.status == SeqEngine::AdmitResult::Status::busy) {
            if (job->report_admission) {
                job->admission = RoutingAdmission::busy;
                finish_job(job);
                return AdmissionDisposition::Retired;
            }
            return AdmissionDisposition::Deferred;
        }
        if (ar.status != SeqEngine::AdmitResult::Status::admitted) {
            std::fprintf(stderr, "[server] admit failed: %s\n",
                         ar.error.c_str());
            send_error(job->fd, 500, "admission failed: " + ar.error);
            finish_job(job);
            return AdmissionDisposition::Retired;
        }
        next_request_id++;

        // Commit response bytes only after admission. A busy routed request
        // can still try another model; no tokenizer/model identity is on wire.
        auto emitter = std::make_unique<SseEmitter>(
            req.format, req.response_id, req.model,
            (int)req.prompt_tokens.size(), req.tools, &tool_memory_,
            req.stop_sequences, req.started_in_thinking);
        if (req.stream) {
            bool ok = send_sse_headers(job);
            if (ok) {
                for (const auto & c : emitter->emit_start()) {
                    if (!send_job_bytes(job, c.data(), c.size())) {
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) {
                engine.retire(ar.slot);
                finish_job(job);
                return AdmissionDisposition::Retired;
            }
            start_job_stream(job);
        }


        SchedSlot & s = slots[(size_t)ar.slot];
        s = SchedSlot{};
        s.job = job;
        s.fd = job->fd;
        s.prefilling = true;
        s.admission_order = next_admission_order++;
        s.started_at = started_at;
        s.decode_started_at = started_at;  // sane on prefill failure
        s.n_gen_cap = std::min(
            n_gen_cap,
            engine.max_context() - (int)req.prompt_tokens.size() + 1);
        s.emitter = std::move(emitter);
        s.send_buffer.mark_progress(std::chrono::steady_clock::now());
        if (budget_active && !config_.think_close_token_ids.empty() &&
            config_.hard_limit_reply_budget > 0) {
            s.hook.close_token_ids = config_.think_close_token_ids;
            s.hook.hard_limit_remaining = eff_reply_for_n_gen;
        }
        live_slots++;
        publish_live_count();
        return AdmissionDisposition::Admitted;
    };

    // Scheduler loop. Every iteration walks the same five phases:
    //
    //   1. Admit    — fill every available slot (deferred first, then the
    //                 queue). Their prefills advance inside later steps.
    //   2. Idle     — nothing live: service drains and loop back, where the
    //                 admission phase parks in the blocking dequeue.
    //   3. Step     — advance one pending prefill chunk alongside every
    //                 decoding slot in one engine pass.
    //   4. Flush    — non-blocking write of the buffered chunks; readers that
    //                 stall or overflow their buffer are dropped.
    //   5. Reap     — service drains, then retire whatever finished this
    //                 iteration so its blocks are free for the next admit.
    //
    // Exits on stopping_ (checked after admission), leaving the teardown
    // below to answer every client still parked in a slot, drain, or queue.
    // Hoisted per-iteration buffers — capacity persists across iterations.
    SeqEngine::StepPlan step_plan;
    step_plan.decode.reserve((size_t)n_slots);
    step_plan.prefills.reserve((size_t)n_slots);
    std::vector<PrefillCandidate> prefill_candidates;
    prefill_candidates.reserve((size_t)n_slots);
    size_t prefill_round_robin_start = 0;

    while (true) {
        // Phase 1 — Admission: deferred job first (FIFO), then the queue.
        // Blocking dequeue only when idle; between decode steps only a poll.
        // Engines reject atomically when their current slot, staging, or
        // reserved-capacity limit is reached, so policy stays model-neutral.
        auto idle_admission_deadline =
            std::chrono::steady_clock::time_point{};
        while (live_slots < n_slots && !stopping_.load()) {
            // A deferred job owns the front of the line, so nothing else may
            // be admitted while its retry backoff is still running.
            if (deferred &&
                std::chrono::steady_clock::now() < deferred_retry_at) {
                break;
            }
            // Retry the deferred job first; it was already queued ahead of
            // everything still in the queue.
            ServerJob * job = deferred;
            deferred = nullptr;
            bool woke_from_idle = false;
            if (!job) {
                if (live_slots == 0 && drains.empty()) {
                    // Fully idle: no stream is waiting on us, so give the
                    // scratch memory back and park in a blocking dequeue.
                    publish_live_count();
                    backend_.release_scratch();
                    job = dequeue();
                    woke_from_idle = job != nullptr;
                } else {
                    // A fresh idle transition may spend its bounded batching
                    // window here; all ongoing decode paths only poll.
                    const auto now = std::chrono::steady_clock::now();
                    if (idle_admission_deadline > now) {
                        job = dequeue_for(idle_admission_deadline - now);
                        if (!job) idle_admission_deadline = {};
                    } else {
                        job = try_dequeue();
                    }
                }
            }
            if (!job) break;  // queue empty: go decode what is already live
            if (job->client_disconnected.load(std::memory_order_acquire)) {
                finish_job(job);
                continue;
            }
            const AdmissionDisposition outcome = admit_job(job);
            if (outcome == AdmissionDisposition::Deferred) {
                deferred = job;
                deferred_retry_at = std::chrono::steady_clock::now() +
                                    std::chrono::seconds(1);
                break;  // wait for a retire to free blocks
            }
            if (outcome == AdmissionDisposition::Admitted) {
                if (woke_from_idle && n_slots > 1 &&
                    config_.admission_coalesce_ms > 0) {
                    idle_admission_deadline =
                        std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(
                            config_.admission_coalesce_ms);
                }
                continue;  // fill the remaining slots before step()
            }
        }
        if (stopping_.load()) break;

        // Phase 2 — Idle: no slot to step, so only the drains need service.
        if (live_slots == 0) {
            service_drains();
            if (deferred) {
                // A defensive busy response with no live sequence must not
                // turn the worker into a tight retry loop. Real capacity
                // releases clear deferred_retry_at in retire_slot().
                const auto now = std::chrono::steady_clock::now();
                if (deferred_retry_at > now) {
                    std::this_thread::sleep_until(std::min(
                        deferred_retry_at, now + std::chrono::milliseconds(5)));
                }
                continue;
            }
            if (!drains.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;             // keep draining; don't block in dequeue
            }
            continue;                 // dequeue() blocks in admission
        }

        // Retire cancellations before spending another model step.
        for (int i = 0; i < n_slots; ++i) {
            SchedSlot & s = slots[(size_t)i];
            if (s.job && s.job->client_disconnected.load(
                    std::memory_order_acquire)) {
                s.client_disconnected = true;
                s.finished = true;
                retire_slot(i, true);
            }
        }
        if (live_slots == 0) continue;

        if (offload_budget) {
            int resident = 0, oldest = -1;
            for (int i = 0; i < n_slots; ++i) {
                if (!slots[(size_t)i].job) continue;
                if (!suspended(i)) { ++resident; continue; }
                if (oldest < 0 || slots[(size_t)i].admission_order <
                                  slots[(size_t)oldest].admission_order) oldest = i;
            }
            // Drain resident requests before restoring the oldest suspension,
            // unless the engine reports comfortable headroom — restoring then
            // cannot re-trigger eviction on the next step, and a parked
            // request behind one long resident would otherwise wait the whole
            // generation. One resume per pass keeps the cadence bounded.
            if (oldest >= 0 &&
                (resident == 0 || engine.kv_restore_feasible(oldest))) {
                const bool drained = resident == 0;
                const auto oldest_state = engine.kv_offload_state(oldest);
                std::string error;
                if (engine.restore_kv(oldest, error)) {
                    auto & s = slots[(size_t)oldest];
                    if (oldest_state.recompute) {
                        s.prefilling = true;
                        s.pending_tok = -1;
                        std::fprintf(stderr,
                            "[parallel] slot %d resumed via re-prefill\n",
                            oldest);
                    } else {
                        std::fprintf(stderr,
                            "[parallel] slot %d resumed from RAM\n", oldest);
                    }
                    publish_live_count();
                } else if (!error.empty() && !oldest_state.recompute &&
                           engine.evict_kv(oldest,
                               slots[(size_t)oldest].pending_tok, error)) {
                    // A checkpoint copy that fails for a real reason does not
                    // self-heal: drop the payload and park for recompute,
                    // keeping the request alive through its retained history.
                    std::fprintf(stderr,
                        "[parallel] slot %d checkpoint lost; parked for recompute\n",
                        oldest);
                    publish_live_count();
                } else if (drained) {
                    // Capacity failure (empty error) with an empty cohort
                    // means the request cannot fit even alone; recompute
                    // cannot help either, so this remains legitimately fatal.
                    auto & s = slots[(size_t)oldest];
                    s.failed = true;
                    s.error = error.empty()
                        ? "suspended request cannot fit its KV state and next token in the pool"
                        : error;
                    retire_slot(oldest, false);
                }
                continue;
            }
        }

        // Phase 3 — Build the resident cohort. Suspended requests retain their
        // socket/emitter and pending token, but never enter a device graph.
        auto build_plan = [&]() {
            step_plan.decode.clear();
            prefill_candidates.clear();
            for (int i = 0; i < n_slots; ++i) {
                const auto & s = slots[(size_t)i];
                if (!s.job || suspended(i)) continue;
                if (!s.prefilling) {
                    step_plan.decode.push_back(
                        {i, s.pending_tok, s.hook.close_token_ids.empty()});
                } else {
                    prefill_candidates.push_back({i, s.admission_order});
                }
            }
            step_plan.prefills = plan_prefill_slices(prefill_candidates,
                engine.step_plan_limits((int)step_plan.decode.size()),
                prefill_round_robin_start);
        };
        build_plan();
        while (offload_budget && !engine.reserve_decode(step_plan)) {
            // A speculative burst is optional. Try a one-token round before
            // suspending a request to make room for a wider accepted chain.
            for (auto & input : step_plan.decode) input.allow_speculation = false;
            if (engine.reserve_decode(step_plan)) break;

            std::vector<int> residents;
            size_t used_bytes = 0;
            for (int i = 0; i < n_slots; ++i) {
                if (!slots[(size_t)i].job) continue;
                const auto state = engine.kv_offload_state(i);
                used_bytes += state.bytes;
                if (!state.suspended) residents.push_back(i);
            }
            std::sort(residents.begin(), residents.end(), [&](int a, int b) {
                return slots[(size_t)a].admission_order > slots[(size_t)b].admission_order;
            });
            bool saved = false;
            std::string error;
            if (residents.size() > 1) {
                const size_t available = offload_budget - std::min(offload_budget, used_bytes);
                for (int victim : residents) {
                    if (engine.offload_kv(victim, available, error)) {
                        std::fprintf(stderr, "[parallel] slot %d suspended to RAM (%zu bytes)\n",
                                     victim, engine.kv_offload_state(victim).bytes);
                        saved = true;
                        publish_live_count();
                        break;
                    }
                }
            }
            if (!saved) {
                // No checkpoint fit the RAM cap: park the newest decoder for
                // recompute. Its blocks free immediately at zero RAM cost and
                // chunked prefill rebuilds its state on resume — termination
                // remains only for a request that cannot fit the pool alone.
                int victim = -1;
                for (int candidate : residents) {
                    if (!slots[(size_t)candidate].prefilling) { victim = candidate; break; }
                }
                if (victim < 0) break; // engines reserve prefills at admission
                auto & s = slots[(size_t)victim];
                if (engine.evict_kv(victim, s.pending_tok, error)) {
                    std::fprintf(stderr,
                        "[parallel] slot %d parked for KV recompute\n", victim);
                    publish_live_count();
                } else {
                    s.failed = true;
                    s.error = error.empty()
                        ? "paged KV pool cannot fit the request's next decode token"
                        : "paged KV growth could not be preserved: " + error;
                    retire_slot(victim, false);
                }
            }
            build_plan();
        }
        if (!prefill_candidates.empty()) ++prefill_round_robin_start;
        if (step_plan.decode.empty() && step_plan.prefills.empty()) continue;

        SeqEngine::StepResult step_result = engine.step(step_plan);
        const std::string protocol_error =
            validate_step_result(step_plan, step_result, n_slots);
        if (!protocol_error.empty()) {
            step_result.decode.clear();
            step_result.prefills.clear();
            step_result.error =
                "engine step protocol violation: " + protocol_error;
        }

        if (!step_result.ok()) {
            const std::string & error = step_result.error;
            std::fprintf(stderr,
                "[parallel] engine step failed: %s — "
                "failing all live requests\n", error.c_str());
            for (int i = 0; i < n_slots; i++) {
                if (slots[(size_t)i].job) {
                    slots[(size_t)i].failed = true;
                    slots[(size_t)i].error = error;
                    retire_slot(i, false);
                }
            }
            continue;
        }
        for (const auto & out : step_result.decode) {
            if (out.slot < 0 || out.slot >= n_slots) continue;
            SchedSlot & s = slots[(size_t)out.slot];
            if (!s.job) continue;
            if (out.failed) {
                s.failed = true;
                s.error = out.error;
                s.finished = true;
                continue;
            }
            consume_decode_output_tokens(out, [&](int32_t token) {
                if (s.finished) return false;
                advance_slot(s, token);
                return !s.finished;
            });
        }
        using PrefillStatus = SeqEngine::PrefillOutput::Status;
        for (const auto & out : step_result.prefills) {
            if (out.slot < 0 || out.slot >= n_slots) continue;
            SchedSlot & s = slots[(size_t)out.slot];
            if (!s.job) continue;
            if (out.status == PrefillStatus::failed) {
                s.failed = true;
                s.error = out.error;
                s.finished = true;
                continue;
            }
            if (out.status == PrefillStatus::completed) {
                s.prefilling = false;
                publish_live_count();
                // A prefill lane just became reusable, so the FIFO head may
                // be admissible even though no KV blocks were retired.
                deferred_retry_at = {};
                s.decode_started_at = std::chrono::steady_clock::now();
                s.prefill_s = std::chrono::duration<double>(
                    s.decode_started_at - s.started_at).count();
                advance_slot(s, out.token);
                continue;
            }
        }
        // Phase 4 — Non-blocking flush of every live slot's chunks. Progress
        // resets the stall clock; a reader that makes no progress for 30 s
        // or lets the buffer hit the cap is dropped (its slot retires).
        {
            const auto now = std::chrono::steady_clock::now();
            for (int i = 0; i < n_slots; i++) {
                SchedSlot & s = slots[(size_t)i];
                if (!s.job || s.client_disconnected) continue;
                bool flush_ok = false;
                {
                    std::lock_guard<std::mutex> lock(s.job->write_mu);
                    flush_ok = s.send_buffer.flush(s.fd);
                }
                if (!flush_ok) {
                    s.client_disconnected = true;
                    s.finished = true;
                    continue;
                }
                if (s.send_buffer.should_drop(now, kClientStallTimeout,
                                              kMaxSlotSendBuffer)) {
                    std::fprintf(stderr,
                        "[parallel] slot %d client stalled — dropping stream\n", i);
                    s.client_disconnected = true;
                    s.finished = true;
                }
            }
        }
        // Phase 5 — Reap: finish the drains, then hand back the blocks of
        // every slot that ended this iteration so the next admit can use them.
        service_drains();
        for (int i = 0; i < n_slots; i++) {
            if (slots[(size_t)i].job && slots[(size_t)i].finished) {
                retire_slot(i, true);
            }
        }
    }

    // Shutdown: unblock every parked client thread.
    for (int i = 0; i < n_slots; i++) {
        if (slots[(size_t)i].job) {
            slots[(size_t)i].failed = true;
            retire_slot(i, false);
        }
    }
    service_drains();
    for (DrainJob & d : drains) finish_job(d.job);
    drains.clear();
    if (deferred) {
        // No response has been committed for a deferred admission.
        send_error(deferred->fd, 503, "server shutting down");
        finish_job(deferred);
    }
    // Jobs that never reached admission are still parked in their client
    // threads too. Drain the raw queue before returning so run() does not hit
    // its client-shutdown timeout and the destructor never has to wake threads
    // after the server/backend teardown has already started.
    while (ServerJob * queued = try_dequeue()) {
        send_error(queued->fd, 503, "server shutting down");
        finish_job(queued);
    }
}


}  // namespace dflash::common
