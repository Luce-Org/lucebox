// Concurrent scheduler for --max-concurrency serving: the worker thread's
// iteration-level loop over a backend's SeqEngine decode slots.
//
// Split from http_server.cpp: this TU owns non-blocking admission (one
// prefill chunk per engine step, fused with the live decode batch), FIFO
// pool-full deferrals, per-slot streaming through ClientSendBuffer, and
// retirement. SSE emission, terminal errors, and HTTP response
// formatting are shared with the classic worker so both paths emit
// matching wire formats.

#include "http_server.h"
#include "common/concurrency/seq_engine.h"
#include "common/output_constraint.h"
#include "common/spec_acceptance.h"
#include "parallel_prefix_txn.h"
#include "response_error.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

namespace luce::common {

namespace {

// Per-slot request state for the iteration-level scheduler. Indexed by the
// engine slot id returned from admit(), so scheduler and engine agree on
// which engine-owned state record a request owns. This remains the one
// external phase: sockets stay here, prompt/KV/sampler/progress stay in Qwen.
using PrefixCaptureTxn = BasicPrefixCaptureTxn<
    PrefixCache::InlineReservation, SeqEngine>;

struct SchedSlot {
    ServerJob * job = nullptr;
    SocketHandle fd = kInvalidSocket;
    std::unique_ptr<SseEmitter> emitter;
    bool prefilling = false;
    // Prompt tokens still to prefill, less each step's grant (a step can end
    // its chunk early, so it may run low), and the new tokens the prompt
    // brought (StepPlanLimits::time_share_prompt_tokens). A recomputed
    // request counts as long.
    int prompt_left = 0;
    int prompt_new = 0;
    uint64_t admission_order = 0;
    std::chrono::steady_clock::time_point started_at{};
    std::chrono::steady_clock::time_point decode_started_at{};
    double prefill_s = 0.0;
    int cached_prefix_tokens = 0;
    PrefixCaptureTxn cache_capture;
    int n_gen_cap = 0;
    int completion_tokens = 0;
    bool client_disconnected = false;
    std::optional<ResponseError> error;
    bool finished = false;
    std::vector<int32_t> gen_tokens;   // committed + pending, in order
    int32_t pending_tok = -1;          // sampled, fed back next step
    // Buffered client output (see client_send_buffer.h): chunks append here and
    // a non-blocking flush runs every scheduler iteration, so one slow
    // reader can never head-of-line-block the shared decode loop.
    luce::common::ClientSendBuffer send_buffer;
    // Thinking-budget force-close, applied scheduler-side before the token
    // is fed back (the same rule as do_ar_decode, common/thinking_budget.h).
    luce::common::BudgetHook hook;
    luce::common::ThinkingBudget budget;
    bool degenerate_close = false;
    bool loop_close_armed = false;
    // Speculative verifier positions over the request's rounds (see
    // SeqEngine::DecodeOutput), for its /status perf record.
    luce::common::SpecAcceptanceStats acceptance;
};

// Outcome of one admission attempt. The three cases differ in who owns the
// job afterwards: Admitted hands it to a slot, Deferred hands it back to the
// caller to retry at the head of the line, Retired means the job is already
// answered and holds nothing.
enum class AdmissionDisposition {
    Admitted,  // job owns an engine slot and its first token is on the wire
    Deferred,  // engine had no room; caller keeps the job and retries it first
    Loading,   // its disk snapshot is being read; retried first, without backoff
    Retired,   // job finished or failed during admission; no slot was taken
};

// Long prompts handed to the PFlash compression thread: the scheduler
// pushes, the compression thread pops.
struct PflashStageQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<ServerJob *> jobs;
    bool stop = false;
};

}  // namespace

void HttpServer::scheduler_loop(SeqEngine & engine) {
    const int n_slots = engine.slot_count();
    if (config_.concurrent_prefix_cache &&
        config_.concurrent_prefix_cache_max_bytes == ServerConfig::kPrefixCacheBudgetAuto) {
        const size_t per = engine.estimate_prefix_store_bytes(engine.max_context());
        if (per > 0) {
            const size_t bytes = concurrent_prefix_budget(per, n_slots);
            prefix_cache_.set_max_resident_bytes(bytes);
            std::fprintf(stderr,
                "[pc] concurrent prefix budget (auto): %zu MiB = %d slots, %zu MiB per "
                "checkpoint at %d tokens\n",
                bytes / (1024 * 1024), n_slots, per / (1024 * 1024), engine.max_context());
        }
    }
    std::vector<SchedSlot> slots((size_t)n_slots);
    uint64_t next_request_id = 1;
    uint64_t next_admission_order = 0;
    // Admission-deferred job (pool blocks/slots exhausted). Kept at the head
    // of the line so FIFO order survives the deferral.
    ServerJob * deferred = nullptr;
    std::chrono::steady_clock::time_point deferred_retry_at{};
    // Requests waiting for the prefix another slot is prefilling to be
    // saved (see find_prefix_leader), oldest first. They hold no slot, so
    // the line behind them keeps moving.
    struct PrefixWaiter {
        ServerJob * job = nullptr;
        int leader = -1;
        uint64_t leader_order = 0;
        int cut = 0;
        int leader_left = 0;
        std::chrono::steady_clock::time_point since{};
        std::chrono::steady_clock::time_point progressed_at{};
    };
    std::vector<PrefixWaiter> prefix_waiters;
    // A leader that prefills nothing for this long (parked, or starved by
    // longer prompts) releases its waiters to prefill on their own.
    constexpr auto kPrefixLeaderStall = std::chrono::seconds(30);

    // Disk session cache. Checkpoints that leave RAM (a capture replacing
    // its reservation's victim) are written by the disk writer thread; a
    // prompt whose deepest state is on disk waits at the head of the line
    // while a reader thread loads it, so live streams keep decoding. Only
    // engines whose checkpoints live in the backend's snapshot slots
    // (Qwen paged) take part.
    const bool disk_enabled = !disk_cache_.disabled() &&
        engine.supports_prefix_store() && !prefix_cache_.disabled() &&
        engine.prefix_store_in_backend_snapshots();
    if (!disk_cache_.disabled() && !disk_enabled) {
        std::fprintf(stderr,
            "[disk-cache] off for concurrent serving: this engine keeps no "
            "checkpoints in backend snapshot slots\n");
    }
    // Recorded with the disk off too: a capture whose resident-budget plan
    // evicts further checkpoints (PrefixCache::reserve_inline_snap) frees
    // their payloads here.
    prefix_cache_.set_record_evictions(true);
    struct DiskLoad {
        ServerJob * job = nullptr;
        std::vector<int32_t> key;
        struct Result {
            std::atomic<bool> done{false};
            OwnedSnapshot snap;
        };
        std::shared_ptr<Result> result;
        std::thread reader;
    };
    std::optional<DiskLoad> disk_load;
    std::vector<DiskLoad> finished_readers;
    // Staging checkpoint for a disk restore: PrefixCache never uses it.
    constexpr int kDiskStagingIndex = PrefixCache::MAX_CACHE_SLOTS;
    auto drop_disk_load = [&]() {
        if (!disk_load) return;
        if (disk_load->reader.joinable()) {
            finished_readers.push_back(std::move(*disk_load));
        }
        disk_load.reset();
    };
    auto reap_readers = [&]() {
        for (auto it = finished_readers.begin(); it != finished_readers.end();) {
            if (it->result->done.load(std::memory_order_acquire)) {
                it->reader.join();
                it = finished_readers.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Degenerate-run guard shared with do_ar_decode: explicit env override,
    // else 32 when the min-tokens floor is active, else off.
    static const int repeat_guard = [] {
        if (const char * s = std::getenv("LUCE_DEGENERATE_RUN_TOKENS")) {
            const int v = std::atoi(s);
            if (v >= 0) return v;
        }
        const char * f = std::getenv("LUCE_MIN_TOKENS");
        return (f && std::atoi(f) > 0) ? 32 : 0;
    }();

    // Cached live-slot count — incremented on admit, decremented on retire.
    // Replaces the O(n_slots) scan that was called 2-3× per iteration.
    int live_slots = 0;

    // Capture tickets are never reused during this scheduler run.
    uint64_t next_prefix_capture_id = 1;

    int published_live_count = -1;
    int published_prefill_count = -1, published_parked_count = -1;
    size_t published_offloaded_bytes = 0;
    const size_t offload_budget = config_.decode_kv_offload_bytes;
    auto parked = [&](int slot) {
        return offload_budget && engine.kv_offload_state(slot).parked;
    };
    auto publish_live_count = [&]() {
        int prefilling = 0, parked_count = 0;
        size_t bytes = 0;
        for (int i = 0; i < n_slots; ++i) {
            const SchedSlot & s = slots[(size_t)i];
            if (!s.job) continue;
            const auto state = offload_budget ? engine.kv_offload_state(i)
                                             : SeqEngine::KvOffloadState{};
            parked_count += state.parked;
            bytes += state.bytes;
            if (s.prefilling && !state.parked) ++prefilling;
        }
        if (live_slots == published_live_count &&
            prefilling == published_prefill_count &&
            parked_count == published_parked_count && bytes == published_offloaded_bytes) return;
        published_live_count = live_slots;
        published_prefill_count = prefilling;
        published_parked_count = parked_count;
        published_offloaded_bytes = bytes;
        if (live_slots > 0) {
            status_.set_concurrent_requests(live_slots, prefilling, parked_count, bytes);
        } else status_.set_idle();
        broadcast_status();
    };

    auto finish_job = [this](ServerJob * job) {
        stop_job_stream(job);
        {
            std::lock_guard<std::mutex> lk(job->mu);
            job->done = true;
            job->cv.notify_one();
        }
        publish_memory_report();
    };
    publish_memory_report();

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

    // Advances one slot by a single sampled token — the post-sample path
    // shared by the first (prefill-logits) token and every decode-step token.
    // Note `tok` is by value but not passthrough: the shared budget may
    // substitute a close token, and that substitute is what gets
    // recorded, emitted, and fed back. Appends to gen_tokens, streams the
    // delta into send_buffer, and parks the token in pending_tok as the next
    // step's input for this slot. Sets s.finished — but never retires the
    // slot — on EOS, gen cap, stop-sequence hit, or degenerate repetition.
    auto advance_slot = [&](SchedSlot & s, int32_t tok) {
        tok = s.budget.apply(tok).token;
        // The grammar moves with the token fed back, the budget's included.
        if (OutputConstraint * constraint = s.job->req.sampler.constraint.get()) {
            constraint->accept(tok);
        }
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
        if (s.budget.forced_close() && !s.budget.injecting()) {
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

    auto retire_slot = [&](int idx) {
        SchedSlot & s = slots[(size_t)idx];
        if (!s.job) return;
        const ParsedRequest & req = s.job->req;
        s.cache_capture.cancel();
        // Stop monitor-thread heartbeats before queuing terminal frames.
        stop_job_stream(s.job, &s.send_buffer);
        const double decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - s.decode_started_at).count();
        const int prompt_tokens = (int)req.prompt_tokens.size();
        GenTimings gen_timings{
            s.prefill_s,
            decode_s,
            /*cache_hit=*/s.cached_prefix_tokens > 0,
            /*cached_prefix_tokens=*/s.cached_prefix_tokens,
            /*prefilled_tokens=*/prompt_tokens - s.cached_prefix_tokens,
            /*effective_prompt_tokens=*/prompt_tokens,
        };
        gen_timings.pflash = s.job->pflash_stats;

        if (!s.error) {
            PerfRecord perf;
            perf.prompt_tokens = (int)req.prompt_tokens.size();
            perf.pflash = req.client_prompt_tokens >= 0;
            perf.completion_tokens = s.completion_tokens;
            const int computed_prompt_tokens =
                prompt_tokens - s.cached_prefix_tokens;
            perf.prefill_tok_s = s.prefill_s > 0.0
                ? (double)computed_prompt_tokens / s.prefill_s : 0.0;
            perf.decode_tok_s = decode_s > 0.0
                ? (double)s.completion_tokens / decode_s : 0.0;
            perf.spec_decode = s.acceptance.offered() > 0;
            perf.accept_rate = s.acceptance.rate();
            status_.record_perf(perf);
        }

        if (s.error) {
            if (!s.client_disconnected) {
                if (req.stream) {
                    for (const std::string & chunk :
                         s.emitter->emit_error(*s.error)) {
                        s.send_buffer.append(chunk);
                    }
                } else {
                    const json body = build_error_response(
                        req.format, *s.error, req.response_id);
                    s.send_buffer.append(format_http_response(
                        response_error_http_status(*s.error),
                        "application/json", body.dump() + "\n"));
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
                                    s.n_gen_cap, s.budget.forced_close(),
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
            s.error ? "false" : "true",
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
        const int eff_reply_for_n_gen = luce::common::reply_reserve_within(
            req.per_req_reply_budget >= 0 ? req.per_req_reply_budget
                                          : config_.hard_limit_reply_budget,
            req.max_output);
        const int n_gen_cap = budget_active
            ? (std::min)(effective_think_ceiling + eff_reply_for_n_gen,
                         req.max_output)
            : req.max_output;

        if (n_gen_cap < 1) {
            // Degenerate ask: reply with an empty completion, no slot needed.
            SseEmitter emitter(req.format, req.response_id, req.model,
                               usage_prompt_tokens(req), req.tools,
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

        // PrefixCache owns token policy; SeqEngine owns checkpoint payloads.
        // Unsupported engines never receive a plan, so cold fallback cannot
        // invalidate a valid entry.
        PrefixStorePlan prefix_plan;
        PrefixCaptureTxn prepared_capture;
        PrefixCache::InlineReservation capture_reservation;
        int restore_policy_slot = -1;
        bool disk_restore = false;
        // Tokens alone do not identify an image: image requests never touch
        // the prefix cache.
        const bool prefix_supported = !req.images &&
            engine.supports_prefix_store() && !prefix_cache_.disabled();
        if (prefix_supported) {
            const auto hit = prefix_cache_.lookup_candidate(
                req.prompt_tokens,
                (int)req.prompt_tokens.size() - 1);
            if (hit.first >= 0 && hit.second > 0 &&
                hit.second < (int)req.prompt_tokens.size()) {
                restore_policy_slot = hit.first;
                prefix_plan.restore = {
                    (uint64_t)hit.first + 1, hit.second};
            }
            // A deeper state on disk (the index answers without a read):
            // load it off this thread, then restore from the staging slot.
            std::vector<int> scoped_lengths;
            const auto mode = req.disk_cache_policy.mode;
            if (mode == DiskPrefixCacheMode::Fixed) {
                scoped_lengths.push_back(disk_prefix_cache_fixed_boundary(
                    req.disk_cache_policy, (int)req.prompt_tokens.size(),
                    config_.disk_cache_min_tokens));
            } else if (mode == DiskPrefixCacheMode::Auto) {
                scoped_lengths = find_all_boundaries(
                    req.prompt_tokens, prefix_cache_.chat_markers());
            }
            const auto probe = disk_enabled && mode != DiskPrefixCacheMode::Off
                ? disk_cache_.longest_prefix(req.prompt_tokens,
                    mode == DiskPrefixCacheMode::Full ? nullptr : &scoped_lengths)
                : DiskPrefixCache::PrefixProbe{};
            if (disk_load && disk_load->job != job) drop_disk_load();
            if (probe.cur_pos > prefix_plan.restore.tokens &&
                probe.cur_pos == probe.len &&
                probe.cur_pos < (int)req.prompt_tokens.size()) {
                if (!disk_load) {
                    disk_load.emplace();
                    disk_load->job = job;
                    disk_load->key.assign(req.prompt_tokens.begin(),
                                          req.prompt_tokens.begin() + probe.len);
                    disk_load->result = std::make_shared<DiskLoad::Result>();
                    disk_load->reader = std::thread(
                        [this, key = disk_load->key, out = disk_load->result]() {
                            out->snap = disk_cache_.load(key);
                            out->done.store(true, std::memory_order_release);
                        });
                    std::fprintf(stderr,
                        "[disk-cache] %s: reading %d-token state from disk "
                        "(RAM has %d)\n", req.response_id.c_str(), probe.len,
                        prefix_plan.restore.tokens);
                }
                if (!disk_load->result->done.load(std::memory_order_acquire)) {
                    return AdmissionDisposition::Loading;
                }
                disk_load->reader.join();
                OwnedSnapshot snap = std::move(disk_load->result->snap);
                disk_load.reset();
                const int pos = snap.cur_pos();
                if (!snap.empty() &&
                    disk_cache_.adopt(std::move(snap), kDiskStagingIndex)) {
                    disk_restore = true;
                    restore_policy_slot = -1;
                    prefix_plan.restore = {(uint64_t)kDiskStagingIndex + 1, pos};
                } else {
                    std::fprintf(stderr,
                        "[disk-cache] %s: disk state unusable, %s\n",
                        req.response_id.c_str(),
                        prefix_plan.restore.valid() ? "restoring from RAM"
                                                    : "prefilling");
                }
            } else if (disk_load && disk_load->job == job) {
                drop_disk_load();
            }

            if (prefix_plan.restore.valid() && disk_cache_digest_enabled()) {
                int n_tensors = 0;
                uint64_t n_bytes = 0;
                const uint64_t digest = snapshot_state_digest(
                    backend_.snapshot_ref((int)prefix_plan.restore.id - 1),
                    &n_tensors, &n_bytes);
                std::fprintf(stderr,
                    "[disk-cache] digest %s source=%s pos=%d tensors=%d "
                    "bytes=%llu xxh64=%016llx\n",
                    req.response_id.c_str(), disk_restore ? "ssd" : "ram",
                    prefix_plan.restore.tokens, n_tensors,
                    (unsigned long long)n_bytes, (unsigned long long)digest);
            }

            // The restore source is never the capture destination: the new
            // checkpoint lands in a different slot so the restore point can
            // slide forward past the deepest slot (same rule as the classic
            // worker's restore_source_slot).
            const int restored_len =
                prefix_plan.restore.valid() ? prefix_plan.restore.tokens : 0;
            const auto reserve_capture = [&](int forced_cut) {
                return prefix_cache_.reserve_inline_snap(
                    req.prompt_tokens, restored_len,
                    /*prefer_tools_boundary=*/!req.tools.empty(),
                    forced_cut,
                    restore_policy_slot,
                    [&engine](int target_cut) {
                        return engine.estimate_prefix_store_bytes(target_cut);
                    },
                    /*reachable_from=*/0, backend_.restore_point_spacing());
            };
            capture_reservation = reserve_capture(req.pin_end_token);
            // Subagents launched together: when the requests still queued
            // share more of this prompt than its own capture keeps whole
            // (system messages that differ only after the tools), it
            // captures the shared prefix instead, for them to restore.
            const int wait_min = config_.shared_prefix_wait_min_tokens;
            if (capture_reservation.active() && wait_min > 0 &&
                req.pin_end_token <= restored_len) {
                const int own_cut = capture_reservation.target_cut();
                const int shared = queued_shared_prefix(req.prompt_tokens);
                if (shared < own_cut && shared - restored_len >= wait_min) {
                    capture_reservation.cancel();
                    capture_reservation = reserve_capture(shared);
                    if (capture_reservation.active() &&
                        capture_reservation.target_cut() <= shared &&
                        capture_reservation.target_cut() - restored_len >= wait_min) {
                        std::fprintf(stderr,
                            "[pc] %s captures the %d-token prefix it shares with "
                            "queued requests, not %d\n", req.response_id.c_str(),
                            capture_reservation.target_cut(), own_cut);
                    } else {
                        capture_reservation.cancel();
                        capture_reservation = reserve_capture(req.pin_end_token);
                    }
                }
            }
            if (capture_reservation.active()) {
                const uint64_t capture_id = next_prefix_capture_id++;
                if (next_prefix_capture_id == 0)
                    next_prefix_capture_id = 1;
                prefix_plan.capture.id = capture_id;
                prefix_plan.capture.checkpoint = {
                    (uint64_t)capture_reservation.slot() + 1,
                    capture_reservation.target_cut()};
                if (prefix_plan.restore.valid() &&
                    prefix_plan.capture.checkpoint == prefix_plan.restore) {
                    capture_reservation.cancel();
                    prefix_plan.capture = {};
                }
            }
        }
        if (prefix_supported) {
            // The same cuts on hits and misses, whatever the cache holds (as the
            // single-sequence path passes GenerateRequest::restore_points).
            prefix_plan.restore_points = request_restore_points(
                req.prompt_tokens,
                {prefix_plan.restore.tokens, prefix_plan.capture.checkpoint.tokens});
        }
        if (prefix_plan.capture.valid()) {
            prepared_capture = PrefixCaptureTxn(
                std::move(capture_reservation), engine,
                prefix_plan.capture);
        }

        // Admission only claims the slot and queues the prompt. Prefill
        // advances one chunk per engine step alongside live decode.
        const PrefixStorePlan requested_plan = prefix_plan;
        auto ar = req.images
            ? engine.admit_images(
                  next_request_id, req.prompt_tokens, req.sampler, req.images)
            : prefix_supported
            ? engine.admit_with_prefix(
                  next_request_id, req.prompt_tokens, req.sampler,
                  requested_plan)
            : engine.admit(
                  next_request_id, req.prompt_tokens, req.sampler);

        std::string prefix_protocol_error;
        const PrefixStoreAdmission & prefix = ar.prefix_store;
        if (prefix.restore_attempted) {
            prefix_cache_.record_restore_attempt(
                prefix.restore_elapsed_us, prefix.restored.valid());
        }
        if (prefix.malformed_restore_state()) {
            prefix_protocol_error =
                "engine returned malformed prefix restore state";
            // A malformed outcome does not tell us whether the engine-owned
            // payload is still usable. Drop both sides of the requested
            // checkpoint so a later lookup cannot retry stale metadata.
            prepared_capture.cancel();
            if (requested_plan.restore.valid() && restore_policy_slot >= 0) {
                engine.discard_prefix_store(requested_plan.restore);
                prefix_cache_.invalidate_inline_snap(restore_policy_slot);
            }
        }
        if (disk_restore) {
            // The engine copied the staging checkpoint into the sequence's
            // pages during admission (or rejected it); it is not kept.
            backend_.snapshot_free(kDiskStagingIndex);
            if (prefix.restored.valid() &&
                prefix.restored == requested_plan.restore) {
                std::fprintf(stderr,
                    "[disk-cache] hit %s: restored %d tokens from disk\n",
                    req.response_id.c_str(), prefix.restored.tokens);
            }
        }
        if (disk_restore && prefix.invalidated.valid() &&
            prefix.invalidated == requested_plan.restore) {
            // A disk state that does not fit this cache layout: the engine
            // already admitted the prompt cold.
            std::fprintf(stderr,
                "[disk-cache] %s: disk state rejected by the engine, prefilling\n",
                req.response_id.c_str());
        } else if (prefix.invalidated.valid()) {
            if (prefix.invalidated != requested_plan.restore ||
                restore_policy_slot < 0) {
                prefix_protocol_error =
                    "engine invalidated an unrequested prefix checkpoint";
            } else {
                // Cancel this admission's reservation before removing the
                // stale restore metadata. The invalidation itself must not
                // clear a capture reservation owned by another live request.
                // Qwen already discarded the engine-owned payload.
                prepared_capture.cancel();
                prefix_cache_.invalidate_inline_snap(restore_policy_slot);
            }
        }
        if (prefix.restored.valid() &&
            prefix.restored != requested_plan.restore) {
            engine.discard_prefix_store(prefix.restored);
            prefix_protocol_error =
                "engine restored an unrequested prefix checkpoint";
        }
        if (prefix.restored.valid() && prefix.invalidated.valid()) {
            prefix_protocol_error =
                "engine both restored and invalidated one checkpoint";
        }
        if (prefix.capture.valid() &&
            prefix.capture != requested_plan.capture) {
            prefix_protocol_error =
                "engine accepted an unrequested prefix capture";
        }
        if (ar.status != SeqEngine::AdmitResult::Status::admitted &&
            (prefix.restored.valid() || prefix.capture.valid())) {
            prefix_protocol_error =
                "failed admission returned accepted prefix state";
        }
        if (!prefix_protocol_error.empty()) {
            prepared_capture.cancel();
            if (ar.status == SeqEngine::AdmitResult::Status::admitted)
                engine.retire(ar.slot);
            ar.status = SeqEngine::AdmitResult::Status::failed;
            ar.error = "prefix admission protocol violation: " +
                prefix_protocol_error;
        } else if (prefix.capture != requested_plan.capture) {
            // Rejected capture: no payload was touched, so preserve any
            // incumbent selected by the cache's transactional eviction.
            prepared_capture.cancel();
        }


        // Invalid restore cleanup deliberately precedes the status handling
        // below. Every non-admitted outcome releases the untouched capture
        // reservation so the next request can plan its own.
        if (ar.status == SeqEngine::AdmitResult::Status::capacity_exceeded) {
            prepared_capture.cancel();
            if (job->report_admission) {
                job->admission = RoutingAdmission::unfit;
            } else {
                const ResponseError error = ResponseError::invalid_request(
                    "admission_failed", "admission failed: " + ar.error);
                const json body = build_error_response(
                    req.format, error, req.response_id);
                send_response(job->fd, response_error_http_status(error),
                              "application/json", body.dump() + "\n");
            }
            finish_job(job);
            return AdmissionDisposition::Retired;
        }
        if (ar.status == SeqEngine::AdmitResult::Status::busy) {
            prepared_capture.cancel();
            // A routed prompt the drafter already scored waits for this
            // model rather than being scored again for another.
            if (job->report_admission && !job->pflash_prepared) {
                job->admission = RoutingAdmission::busy;
                finish_job(job);
                return AdmissionDisposition::Retired;
            }
            return AdmissionDisposition::Deferred;
        }
        if (ar.status != SeqEngine::AdmitResult::Status::admitted) {
            prepared_capture.cancel();
            std::fprintf(stderr, "[server] admit failed: %s\n",
                         ar.error.c_str());
            const ResponseError error = ResponseError::internal(
                "admission_failed", "admission failed: " + ar.error);
            const json body = build_error_response(
                req.format, error, req.response_id);
            send_response(job->fd, response_error_http_status(error),
                          "application/json", body.dump() + "\n");
            finish_job(job);
            return AdmissionDisposition::Retired;
        }
        next_request_id++;
        if (prefix.restored.valid() && restore_policy_slot >= 0) {
            prefix_cache_.record_inline_hit(
                restore_policy_slot, prefix.restored.tokens,
                req.prompt_tokens.size());
        }

        // Commit response bytes only after admission. A busy routed request
        // can still try another model; no tokenizer/model identity is on wire.
        auto emitter = std::make_unique<SseEmitter>(
            req.format, req.response_id, req.model,
            usage_prompt_tokens(req), req.tools, &tool_memory_,
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
                prepared_capture.cancel();
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
        s.cached_prefix_tokens = prefix.restored.tokens;
        s.prompt_left = (int)req.prompt_tokens.size() - prefix.restored.tokens;
        s.prompt_new = s.prompt_left;
        s.cache_capture = std::move(prepared_capture);
        s.n_gen_cap = std::min(
            n_gen_cap,
            engine.max_context() - (int)req.prompt_tokens.size() + 1);
        s.emitter = std::move(emitter);
        s.send_buffer.mark_progress(std::chrono::steady_clock::now());
        if (budget_active && !config_.think_close_token_ids.empty() &&
            config_.hard_limit_reply_budget > 0) {
            s.hook.close_token_ids = config_.think_close_token_ids;
            s.hook.marker_token_ids = config_.think_marker_token_ids;
            s.hook.hard_limit_remaining = eff_reply_for_n_gen;
            s.hook.loop_repeats = config_.thinking_loop_repeats;
        }
        s.budget = luce::common::ThinkingBudget(s.hook, s.n_gen_cap);
        live_slots++;
        publish_live_count();
        return AdmissionDisposition::Admitted;
    };

    // The slot whose prefill will save the deepest prefix of this prompt, at
    // least shared_prefix_wait_min_tokens past what the prompt restores now
    // and short of the whole prompt; -1 when no live capture is worth the
    // wait. Subagents launched together share their system prompt and tools:
    // the first one admitted saves them, the others restore them instead of
    // prefilling the same tokens beside it.
    auto find_prefix_leader = [&](const ServerJob * job) {
        const int min_tokens = config_.shared_prefix_wait_min_tokens;
        const std::vector<int32_t> & prompt = job->req.prompt_tokens;
        if (min_tokens <= 0 || job->req.images ||
            !engine.supports_prefix_store() || prefix_cache_.disabled()) {
            return -1;
        }
        int leader = -1, best_cut = 0;
        for (int i = 0; i < n_slots; ++i) {
            const SchedSlot & s = slots[(size_t)i];
            if (!s.job || !s.prefilling || parked(i)) continue;
            const int cut = s.cache_capture.target_tokens();
            const std::vector<int32_t> & other = s.job->req.prompt_tokens;
            if (cut < min_tokens || cut <= best_cut ||
                cut >= (int)prompt.size() || cut > (int)other.size() ||
                !std::equal(prompt.begin(), prompt.begin() + cut,
                            other.begin())) {
                continue;
            }
            leader = i;
            best_cut = cut;
        }
        if (leader < 0) return -1;
        const int restorable = prefix_cache_.lookup_candidate(
            prompt, (int)prompt.size() - 1).second;
        return best_cut - restorable >= min_tokens ? leader : -1;
    };
    // The oldest waiter whose wait is over: its leader saved the prefix (or
    // gave up on it), stopped prefilling, stalled, or its client left.
    auto take_ready_waiter = [&]() -> ServerJob * {
        const auto now = std::chrono::steady_clock::now();
        for (auto it = prefix_waiters.begin(); it != prefix_waiters.end(); ++it) {
            PrefixWaiter & w = *it;
            const SchedSlot & leader = slots[(size_t)w.leader];
            const char * reason = nullptr;
            if (w.job->client_disconnected.load(std::memory_order_acquire)) {
                reason = "client left";
            } else if (!leader.job || leader.admission_order != w.leader_order ||
                       !leader.prefilling ||
                       leader.cache_capture.target_tokens() != w.cut) {
                reason = "prefix capture resolved";
            } else if (leader.prompt_left != w.leader_left) {
                w.leader_left = leader.prompt_left;
                w.progressed_at = now;
            } else if (now - w.progressed_at > kPrefixLeaderStall) {
                reason = "leader stalled";
            }
            if (!reason) continue;
            ServerJob * job = w.job;
            std::fprintf(stderr,
                "[pc] %s waited %.1fs for a %d-token shared prefix: %s\n",
                job->req.response_id.c_str(),
                std::chrono::duration<double>(now - w.since).count(),
                w.cut, reason);
            prefix_waiters.erase(it);
            return job;
        }
        return nullptr;
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
    // The decoding requests' turn beside a long prompt
    // (StepPlanLimits::prefill_time_share): no prompt chunk joins a step that
    // has them until it ends.
    std::chrono::steady_clock::time_point decode_only_until{};
    // Longest burst a decode output carries (0: unknown, no bursts while a
    // thinking budget can still replace a token).
    const int max_decode_tokens = engine.max_decode_tokens();
    // Steps in a row that ran short prompts while longer ones waited. After
    // kShortFirstSteps, the longer ones join the next step: short prompts
    // arriving one after another (agents' next turns) cannot stall them.
    constexpr int kShortFirstSteps = 4;
    int short_first_steps = 0;
    bool held_long_prompts = false, long_prompt_waiting = false, long_prompt_in_step = false;

    // PFlash. A prompt over the compression threshold leaves the line for
    // the compression thread: live slots keep decoding while the drafter
    // scores it, and it comes back to the head of the queue with the prompt
    // to serve. That thread owns what prompt preparation touches (the
    // drafter, the chat-view store); the prefix cache stays with this
    // thread, which measures the prompt's restorable prefix before the
    // hand-off.
    const bool pflash_concurrent = pflash_concurrent_serving();
    PflashStageQueue pflash_stage;
    auto pflash_wants = [&](const ServerJob * job) {
        if (!pflash_concurrent || job->pflash_prepared || job->req.images) {
            return false;
        }
        return config_.pflash_mode == ServerConfig::PflashMode::ALWAYS ||
            (int)job->req.prompt_tokens.size() >= config_.pflash_threshold;
    };
    auto pflash_prepare = [this](ServerJob * job) {
        if (job->client_disconnected.load(std::memory_order_acquire)) return;
        const auto started = std::chrono::steady_clock::now();
        const int raw_tokens = (int)job->req.prompt_tokens.size();
        PreparedPrompt prepared = prepare_prompt(job->req);
        if (prepared.error_status != 0) {
            job->pflash_error_status = prepared.error_status;
            job->pflash_error = std::move(prepared.error);
            return;
        }
        job->pflash_stats = std::move(prepared.pflash_stats);
        if (prepared.compressed) {
            job->req.client_prompt_tokens = raw_tokens;
            job->req.prompt_tokens = std::move(prepared.tokens);
            // A view's generation prompt is where its next turn branches off.
            if (job->req.pin_end_token <= 0 && prepared.snapshot_cut > 0) {
                job->req.pin_end_token = prepared.snapshot_cut;
            }
            // A compressed prompt is not raw history: never persist it.
            job->req.disk_cache_policy.mode = DiskPrefixCacheMode::Off;
        }
        std::fprintf(stderr,
            "[pflash-concurrent] %s: %d -> %zu tokens in %.2fs\n",
            job->req.response_id.c_str(), raw_tokens,
            job->req.prompt_tokens.size(),
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count());
    };
    // Prepared jobs return to the head of the queue, behind the prepared
    // ones already waiting there, so they keep their order.
    auto requeue_prepared = [this](ServerJob * job) {
        std::lock_guard<std::mutex> lk(queue_mu_);
        ServerJob ** link = &queue_head_;
        while (*link && (*link)->pflash_prepared) link = &(*link)->next;
        job->next = *link;
        *link = job;
        if (!job->next) queue_tail_ = job;
        queue_cv_.notify_one();
    };
    std::thread pflash_thread;
    if (pflash_concurrent) {
        pflash_thread = std::thread([&pflash_stage, pflash_prepare,
                                     requeue_prepared] {
            for (;;) {
                ServerJob * job = nullptr;
                {
                    std::unique_lock<std::mutex> lk(pflash_stage.mu);
                    pflash_stage.cv.wait(lk, [&] {
                        return pflash_stage.stop || !pflash_stage.jobs.empty();
                    });
                    if (pflash_stage.stop) return;
                    job = pflash_stage.jobs.front();
                    pflash_stage.jobs.pop_front();
                }
                pflash_prepare(job);
                job->pflash_prepared = true;
                requeue_prepared(job);
            }
        });
    }
    auto stop_pflash_thread = [&]() {
        if (!pflash_thread.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(pflash_stage.mu);
            pflash_stage.stop = true;
        }
        pflash_stage.cv.notify_all();
        pflash_thread.join();
    };
    struct StopOnExit {
        decltype(stop_pflash_thread) & stop;
        ~StopOnExit() { stop(); }
    } pflash_thread_guard{stop_pflash_thread};
    // No response bytes are committed before admission, so a failed
    // preparation answers like a failed admission.
    auto fail_preparation = [&](ServerJob * job) {
        const ParsedRequest & req = job->req;
        std::fprintf(stderr, "[server] request failed: %s\n",
                     job->pflash_error.c_str());
        const ResponseError error = job->pflash_error_status == 400
            ? ResponseError::invalid_request("invalid_request", job->pflash_error)
            : job->pflash_error_status == 503
            ? ResponseError::unavailable("unavailable", job->pflash_error)
            : ResponseError::internal("server_error", job->pflash_error);
        const json body = build_error_response(
            req.format, error, req.response_id);
        send_response(job->fd, response_error_http_status(error),
                      "application/json", body.dump() + "\n");
        finish_job(job);
    };

    while (true) {
        reap_readers();
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
            // Then the waiters whose shared prefix is ready (they were
            // dequeued before anything still in the queue).
            if (!job) job = take_ready_waiter();
            const bool fresh = job == nullptr;
            if (!job) {
                if (live_slots == 0 && drains.empty()) {
                    // Fully idle: no stream is waiting on us, so give the
                    // scratch memory back and park in a blocking dequeue.
                    publish_live_count();
                    backend_.release_scratch();
                    job = finished_readers.empty() ? dequeue()
                        : dequeue_for(std::chrono::milliseconds(5));
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
                if (disk_load && disk_load->job == job) drop_disk_load();
                finish_job(job);
                continue;
            }
            if (pflash_wants(job)) {
                if (job->parallel_started_at ==
                        std::chrono::steady_clock::time_point{}) {
                    job->parallel_started_at = std::chrono::steady_clock::now();
                }
                job->req.pflash_cached_prefix = restorable_prefix_tokens(
                    job->req.prompt_tokens, job->req.disk_cache_policy);
                {
                    std::lock_guard<std::mutex> lk(pflash_stage.mu);
                    pflash_stage.jobs.push_back(job);
                }
                pflash_stage.cv.notify_one();
                continue;
            }
            if (job->pflash_error_status != 0) {
                fail_preparation(job);
                continue;
            }
            if (fresh) {
                const int leader = find_prefix_leader(job);
                if (leader >= 0) {
                    const SchedSlot & l = slots[(size_t)leader];
                    const auto now = std::chrono::steady_clock::now();
                    if (job->parallel_started_at ==
                            std::chrono::steady_clock::time_point{}) {
                        job->parallel_started_at = now;
                    }
                    PrefixWaiter w;
                    w.job = job;
                    w.leader = leader;
                    w.leader_order = l.admission_order;
                    w.cut = l.cache_capture.target_tokens();
                    w.leader_left = l.prompt_left;
                    w.since = now;
                    w.progressed_at = now;
                    std::fprintf(stderr,
                        "[pc] %s waits for the %d-token prefix %s is "
                        "prefilling (prompt %zu tokens)\n",
                        job->req.response_id.c_str(), w.cut,
                        l.job->req.response_id.c_str(),
                        job->req.prompt_tokens.size());
                    prefix_waiters.push_back(w);
                    continue;
                }
            }
            const AdmissionDisposition outcome = admit_job(job);
            if (outcome == AdmissionDisposition::Loading) {
                deferred = job;
                deferred_retry_at = std::chrono::steady_clock::now() +
                                    std::chrono::milliseconds(2);
                break;  // keep decoding while the disk read runs
            }
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
                retire_slot(i);
            }
        }
        if (live_slots == 0) continue;

        if (offload_budget) {
            int resident = 0, oldest = -1;
            for (int i = 0; i < n_slots; ++i) {
                if (!slots[(size_t)i].job) continue;
                if (!parked(i)) { ++resident; continue; }
                if (oldest < 0 || slots[(size_t)i].admission_order <
                                  slots[(size_t)oldest].admission_order) oldest = i;
            }
            // Drain resident requests before restoring the oldest parked slot,
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
                        s.prompt_left = std::numeric_limits<int>::max();
                        s.prompt_new = std::numeric_limits<int>::max();
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
                } else if (drained || !error.empty()) {
                    // Capacity failure (empty error) with an empty cohort
                    // means the request cannot fit even alone; recompute
                    // cannot help either, so this remains legitimately fatal.
                    // A real restore error that evict_kv could not convert to
                    // recompute leaves no recovery path — also fatal, rather
                    // than retrying a broken checkpoint every pass.
                    auto & s = slots[(size_t)oldest];
                    s.error = to_response_error({GenerateErrorCode::DecodeFailed,
                        error.empty()
                            ? "parked request cannot fit its KV state and next token in the pool"
                            : error});
                    retire_slot(oldest);
                }
                // A feasibility hint is not a reservation. Even when restore
                // cannot proceed, let residents advance and release capacity.
            }
        }

        // Phase 3 — Build the resident cohort. Parked requests retain their
        // socket/emitter and pending token, but never enter a device graph.
        // A slot whose thinking loops closes it at its next single-token
        // step: the close replaces that token, never one inside a burst.
        const auto arm_loop_close = [&](int i) {
            SchedSlot & s = slots[(size_t)i];
            if (!s.budget.looping() || s.loop_close_armed) return;
            s.budget.arm_loop_close();
            s.loop_close_armed = true;
            std::fprintf(stderr,
                "[parallel] %s thinking repeats itself after %zu tokens: "
                "closing it\n", s.job->req.response_id.c_str(),
                s.gen_tokens.size());
        };
        auto build_plan = [&]() {
            step_plan.decode.clear();
            prefill_candidates.clear();
            for (int i = 0; i < n_slots; ++i) {
                const auto & s = slots[(size_t)i];
                if (!s.job || parked(i)) continue;
                if (!s.prefilling) {
                    // A grammar masks one position at a time: no bursts.
                    // While thinking, a burst must not reach the token the
                    // budget replaces: none closes inside it.
                    const bool budget_clear = s.budget.settled() ||
                        (max_decode_tokens > 0 &&
                         !s.budget.needs_ar_tail(max_decode_tokens));
                    if (!budget_clear) arm_loop_close(i);
                    step_plan.decode.push_back(
                        {i, s.pending_tok,
                         budget_clear && !s.job->req.sampler.constraint});
                } else {
                    prefill_candidates.push_back({i, s.admission_order});
                }
            }
            const bool decoders_turn = !step_plan.decode.empty() &&
                std::chrono::steady_clock::now() < decode_only_until;
            const auto limits = engine.step_plan_limits((int)step_plan.decode.size());
            // Prompts that finish in one step go first, without the longer
            // ones: a question asked beside a long document answers after one
            // short forward, not after the document's next chunk.
            const auto fits_one_step = [&](const PrefillCandidate & c) {
                return slots[(size_t)c.slot].prompt_left <= limits.max_prefill_tokens_per_sequence;
            };
            long_prompt_waiting =
                !std::all_of(prefill_candidates.begin(), prefill_candidates.end(), fits_one_step);
            held_long_prompts = !decoders_turn && long_prompt_waiting &&
                short_first_steps < kShortFirstSteps &&
                std::any_of(prefill_candidates.begin(), prefill_candidates.end(), fits_one_step);
            if (held_long_prompts) {
                prefill_candidates.erase(std::remove_if(prefill_candidates.begin(), prefill_candidates.end(),
                    [&](const PrefillCandidate & c) { return !fits_one_step(c); }),
                    prefill_candidates.end());
            }
            step_plan.prefills = decoders_turn
                ? std::vector<PrefillSlice>{}
                : plan_prefill_slices(prefill_candidates, limits, prefill_round_robin_start);
            long_prompt_in_step = std::any_of(step_plan.prefills.begin(), step_plan.prefills.end(),
                [&](const PrefillSlice & slice) {
                    return slots[(size_t)slice.slot].prompt_left > limits.max_prefill_tokens_per_sequence;
                });
        };
        build_plan();
        while (!engine.reserve_decode(step_plan)) {
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
                if (!state.parked) residents.push_back(i);
            }
            std::sort(residents.begin(), residents.end(), [&](int a, int b) {
                return slots[(size_t)a].admission_order > slots[(size_t)b].admission_order;
            });
            bool saved = false;
            std::string error;
            if (offload_budget && residents.size() > 1) {
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
                // recompute. With recovery disabled, fail only that request
                // before compute, then retry the remaining cohort.
                // Prefer the newest decoder whose KV its token history can
                // rebuild; if none can be, the newest decoder is failed below.
                int victim = -1, fallback = -1;
                for (int candidate : residents) {
                    if (slots[(size_t)candidate].prefilling) continue;
                    if (fallback < 0) fallback = candidate;
                    if (engine.kv_recomputable(candidate)) { victim = candidate; break; }
                }
                if (victim < 0) victim = fallback;
                if (victim < 0) break; // engines reserve prefills at admission
                auto & s = slots[(size_t)victim];
                if (offload_budget && engine.evict_kv(victim, s.pending_tok, error)) {
                    std::fprintf(stderr,
                        "[parallel] slot %d parked for KV recompute\n", victim);
                    publish_live_count();
                } else {
                    s.error = to_response_error({GenerateErrorCode::DecodeFailed,
                        error.empty()
                            ? "paged KV pool cannot fit the request's next decode token"
                            : "paged KV growth could not be preserved: " + error});
                    retire_slot(victim);
                }
            }
            build_plan();
        }
        if (!prefill_candidates.empty()) ++prefill_round_robin_start;
        if (step_plan.decode.empty() && step_plan.prefills.empty()) continue;
        if (held_long_prompts) ++short_first_steps;
        else if (long_prompt_in_step || !long_prompt_waiting) short_first_steps = 0;

        const auto step_started = std::chrono::steady_clock::now();
        SeqEngine::StepResult step_result = engine.step(step_plan);
        // Long prompt work shares the time with the decoding requests, from
        // its first chunk to its last; a short prompt (an agent's next turn,
        // even when a prefix capture splits it in two forwards) does not hold
        // them up. The work is every prompt still prefilling, not one: an
        // agent's subagents arrive together, each prompt short, and the
        // decoding requests beside them would otherwise wait for all of them.
        const auto step_limits = engine.step_plan_limits((int)step_plan.decode.size());
        const int long_prompt_tokens = step_limits.time_share_prompt_tokens > 0
            ? step_limits.time_share_prompt_tokens
            : step_limits.max_prefill_tokens_per_sequence;
        long long prompt_work = 0;
        for (int i = 0; i < n_slots; ++i) {
            const SchedSlot & s = slots[(size_t)i];
            if (s.job && s.prefilling) prompt_work += s.prompt_new;
        }
        const bool long_prompt = !step_plan.prefills.empty() &&
            prompt_work > long_prompt_tokens;
        for (const PrefillSlice & slice : step_plan.prefills) {
            SchedSlot & s = slots[(size_t)slice.slot];
            if (s.prompt_left != std::numeric_limits<int>::max())
                s.prompt_left = std::max(0, s.prompt_left - slice.max_tokens);
        }
        if (long_prompt && !step_plan.decode.empty()) {
            const double share = step_limits.prefill_time_share;
            if (share > 0.0 && share < 1.0) {
                const auto now = std::chrono::steady_clock::now();
                decode_only_until = now + std::chrono::duration_cast<
                    std::chrono::steady_clock::duration>(
                        (now - step_started) * ((1.0 - share) / share));
            }
        }
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
                    slots[(size_t)i].error = ResponseError::internal(
                        "engine_step_failed", error);
                    retire_slot(i);
                }
            }
            continue;
        }
        for (const auto & out : step_result.decode) {
            if (out.slot < 0 || out.slot >= n_slots) continue;
            SchedSlot & s = slots[(size_t)out.slot];
            if (!s.job) continue;
            if (out.failed) {
                s.error = to_response_error(
                    {GenerateErrorCode::DecodeFailed, out.error});
                s.finished = true;
                continue;
            }
            const int room = s.n_gen_cap - (int)s.gen_tokens.size();
            const size_t generated = s.gen_tokens.size();
            consume_decode_output_tokens(out, [&](int32_t token) {
                if (s.finished) return false;
                advance_slot(s, token);
                return !s.finished;
            });
            // Capped as the single-lane decode caps it: positions past the
            // output budget, or past an EOS, could not be used.
            if (out.spec_offered > 0) {
                s.acceptance.record_chain(
                    out.spec_accepted, out.spec_offered,
                    (int)(s.gen_tokens.size() - generated), room);
            }
        }
        using PrefillStatus = SeqEngine::PrefillOutput::Status;
        for (const auto & out : step_result.prefills) {
            if (out.slot < 0 || out.slot >= n_slots) continue;
            SchedSlot & s = slots[(size_t)out.slot];
            if (!s.job) continue;
            if (out.prefix_store.attempted()) {
                prefix_cache_.record_capture_attempt(
                    out.prefix_store.elapsed_us,
                    out.prefix_store.status ==
                        PrefixStoreEvent::Status::saved);
                using Resolution = PrefixCaptureTxn::Resolution;
                const Resolution resolution = s.cache_capture.resolve(
                    out.prefix_store, s.job->req.prompt_tokens);
                const int captured =
                    (int)out.prefix_store.ticket.checkpoint.id - 1;
                // The checkpoint this capture replaced in its slot (the
                // reservation's victim) left RAM: write it to disk unless it
                // was only an edit backup. Claimed on every outcome, and
                // freed here when there is no disk to write it to.
                OwnedSnapshot displaced = backend_.snapshot_take_displaced(captured);
                for (auto & evicted : prefix_cache_.take_evicted()) {
                    if (evicted.slot != captured) {
                        // A further checkpoint the capture's resident-budget
                        // plan evicted as it committed: its payload is still
                        // in its own slot. Written like the displaced one,
                        // then freed.
                        bool queued = false;
                        if (disk_enabled && evicted.persist && slot_tokens_.count(evicted.slot)) {
                            const auto aux = slot_aux_.find(evicted.slot);
                            const std::string blob =
                                aux == slot_aux_.end() ? std::string() : aux->second;
                            OwnedSnapshot snap = backend_.snapshot_detach(evicted.slot);
                            const int pos = snap.empty() ? 0 : snap.cur_pos();
                            queued = !snap.empty() &&
                                disk_cache_.save_async(std::move(snap), evicted.ids, blob);
                            if (queued) {
                                disk_cache_.prune_ancestors(
                                    evicted.ids, config_.disk_cache_cold_max_tokens);
                            }
                            std::fprintf(stderr,
                                "[disk-cache] ram eviction checkpoint=%d tokens=%zu "
                                "pos=%d -> %s\n", evicted.slot + 1, evicted.ids.size(),
                                pos, queued ? "write queued" : "not written");
                        }
                        engine.discard_prefix_store(
                            {(uint64_t)evicted.slot + 1, (int)evicted.ids.size()});
                        forget_inline_slot_metadata(evicted.slot);
                        continue;
                    }
                    if (displaced.empty()) continue;
                    if (evicted.persist && slot_tokens_.count(captured)) {
                        const auto aux = slot_aux_.find(captured);
                        const int pos = displaced.cur_pos();
                        const bool queued = disk_cache_.save_async(
                            std::move(displaced), evicted.ids,
                            aux == slot_aux_.end() ? std::string() : aux->second);
                        if (queued) {
                            disk_cache_.prune_ancestors(
                                evicted.ids, config_.disk_cache_cold_max_tokens);
                        }
                        std::fprintf(stderr,
                            "[disk-cache] ram eviction checkpoint=%d tokens=%zu "
                            "pos=%d -> %s\n", captured + 1, evicted.ids.size(),
                            pos, queued ? "write queued" : "not written");
                    }
                    displaced.reset();
                }
                if (resolution == Resolution::saved) {
                    // The slot holds the new checkpoint now.
                    forget_inline_slot_metadata(captured);
                    if (disk_enabled && s.job->req.disk_cache_policy.mode == DiskPrefixCacheMode::Full) {
                        const int tokens = out.prefix_store.ticket.checkpoint.tokens;
                        const auto & prompt = s.job->req.prompt_tokens;
                        slot_tokens_[captured] = std::vector<int32_t>(
                            prompt.begin(), prompt.begin() + tokens);
                        slot_aux_[captured] =
                            session_replay_blob(s.job->req, nullptr);
                    }
                    // A conversation's checkpoints are prefixes of each other;
                    // restores use only the deepest, so the ones it supersedes
                    // would hold memory the other conversations need.
                    std::vector<int> tokens;
                    const auto pruned = prefix_cache_.prune_superseded_ancestors(
                        captured, &tokens);
                    for (size_t i = 0; i < pruned.size(); ++i) {
                        forget_inline_slot_metadata(pruned[i]);
                        engine.discard_prefix_store({(uint64_t)pruned[i] + 1, tokens[i]});
                    }
                } else if (resolution == Resolution::failed) {
                    std::fprintf(stderr,
                        "[parallel-pc] capture failed checkpoint=%llu: %s\n",
                        (unsigned long long)
                            out.prefix_store.ticket.checkpoint.id,
                        out.prefix_store.error.c_str());
                } else if (resolution == Resolution::mismatched ||
                           resolution == Resolution::inactive) {
                    // The transaction has already aborted only its own
                    // destination. Never act on an event-supplied checkpoint.
                    std::fprintf(stderr,
                        "[parallel-pc] capture ticket mismatch id=%llu "
                        "checkpoint=%llu\n",
                        (unsigned long long)out.prefix_store.ticket.id,
                        (unsigned long long)
                            out.prefix_store.ticket.checkpoint.id);
                }
            }
            if (out.status == PrefillStatus::failed) {
                s.error = to_response_error(
                    {GenerateErrorCode::PrefillFailed, out.error});
                s.finished = true;
                continue;
            }
            if (out.status == PrefillStatus::completed) {
                s.prefilling = false;
                publish_live_count();
                // A prefill lane just became reusable, so the FIFO head may
                // be admissible even though no KV blocks were retired.
                deferred_retry_at = {};
                // Recompute continues the same generation. Keep its original
                // timing boundary so decode wall time includes parking and
                // replay, matching the full response's completion-token count.
                if (s.gen_tokens.empty()) {
                    s.decode_started_at = std::chrono::steady_clock::now();
                    s.prefill_s = std::chrono::duration<double>(
                        s.decode_started_at - s.started_at).count();
                }
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
                retire_slot(i);
            }
        }
    }

    // A prompt still being compressed returns to the queue below; the ones
    // not started yet are answered here.
    stop_pflash_thread();
    for (ServerJob * waiting : pflash_stage.jobs) {
        const ParsedRequest & req = waiting->req;
        const ResponseError error = ResponseError::unavailable(
            "server_shutting_down", "server shutting down");
        const json body = build_error_response(
            req.format, error, req.response_id);
        send_response(waiting->fd, response_error_http_status(error),
                      "application/json", body.dump() + "\n");
        finish_job(waiting);
    }
    pflash_stage.jobs.clear();

    drop_disk_load();
    for (auto & reader : finished_readers) reader.reader.join();
    finished_readers.clear();

    // Shutdown: unblock every parked client thread.
    for (int i = 0; i < n_slots; i++) {
        if (slots[(size_t)i].job) {
            slots[(size_t)i].error = ResponseError::unavailable(
                "server_shutting_down", "server shutting down");
            retire_slot(i);
        }
    }
    service_drains();
    for (DrainJob & d : drains) finish_job(d.job);
    drains.clear();
    // Neither has a response committed: both are answered like the queue.
    for (const PrefixWaiter & w : prefix_waiters) {
        const ParsedRequest & req = w.job->req;
        const ResponseError error = ResponseError::unavailable(
            "server_shutting_down", "server shutting down");
        const json body = build_error_response(
            req.format, error, req.response_id);
        send_response(w.job->fd, response_error_http_status(error),
                      "application/json", body.dump() + "\n");
        finish_job(w.job);
    }
    prefix_waiters.clear();
    if (deferred) {
        // No response has been committed for a deferred admission.
        const ParsedRequest & req = deferred->req;
        const ResponseError error = ResponseError::unavailable(
            "server_shutting_down", "server shutting down");
        const json body = build_error_response(
            req.format, error, req.response_id);
        send_response(deferred->fd, response_error_http_status(error),
                      "application/json", body.dump() + "\n");
        finish_job(deferred);
    }
    // Jobs that never reached admission are still parked in their client
    // threads too. Drain the raw queue before returning so run() does not hit
    // its client-shutdown timeout and the destructor never has to wake threads
    // after the server/backend teardown has already started.
    while (ServerJob * queued = try_dequeue()) {
        const ParsedRequest & req = queued->req;
        const ResponseError error = ResponseError::unavailable(
            "server_shutting_down", "server shutting down");
        const json body = build_error_response(
            req.format, error, req.response_id);
        send_response(queued->fd, response_error_http_status(error),
                      "application/json", body.dump() + "\n");
        finish_job(queued);
    }
}


}  // namespace luce::common
