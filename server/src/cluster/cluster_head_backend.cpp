// cluster_head_backend.cpp - rank-0 ModelBackend decorator.

#include "cluster/cluster_head_backend.h"

#include "cluster/cluster_identity.h"
#include "deepseek4/deepseek4_cluster.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace luce::cluster {

using common::DaemonIO;
using common::GenerateErrorCode;
using common::GenerateRequest;
using common::GenerateResult;
using common::ParkTarget;

namespace {

constexpr int kAbortCodeControl   = 1;   // control channel failure
constexpr int kAbortCodeCollective = 2;  // RCCL failure
// Watchdog poll period. Heartbeats are 1 s with a 10 s timeout, so this only
// bounds how quickly the abort follows the detection, not the detection.
constexpr int kWatchdogPeriodMs   = 250;
// Time between aborting the cluster and exiting, so the failed request's HTTP
// error response is written before the supervisor takes the process.
constexpr int kFatalExitGraceMs   = 2000;
constexpr int kAbortCodeRequest   = 3;   // a rank failed the request

// Self-test sizes: 1000 x 64 KiB, 43 x 16 MiB.
constexpr int    kSelftestIters   = 1000;
constexpr size_t kSelftestSmallN  = 64u * 1024u / sizeof(float);
constexpr size_t kSelftestLargeN  = 16u * 1024u * 1024u / sizeof(float);

std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return buf;
}

uint64_t now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Bootstrap deadline: model loading on the workers can take minutes, so the
// accept phase waits much longer than a per-collective timeout. Workers retry
// their connect for ControlTimeouts::connect_retries * connect_ms (5 min by
// default), and this matches that window.
uint32_t bootstrap_deadline_ms(const ClusterConfig & cfg) {
    return std::max<uint32_t>(cfg.timeout_ms, 300000u);
}

}  // namespace

// ─── Identity ───────────────────────────────────────────────────────────

uint64_t backend_placement_hash(const common::ModelBackend & backend) {
    const auto * p = dynamic_cast<const common::IClusterParticipant *>(&backend);
    return p ? p->cluster_placement_hash() : 0;
}

// ─── Request mapping ────────────────────────────────────────────────────

// GenerateRequest -> RequestMsg. Fields the wire format does not carry (each
// would need a kProtocolVersion bump):
//   SamplerCfg.rep_window, freq_pen, pres_pen  -> affect the AR-vs-spec
//       routing (needs_logit_processing) on workers; today the worker derives
//       do_sample from temperature > 0 and never sees the penalties. Harmless
//       because the head computes decode_mode from its own full sampler and
//       the worker only ever narrows spec to AR, never the other way round.
//   (snap_slot / snap_pos and the restore slot travel since protocol 2.)
//   BudgetHook.hard_limit_remaining            -> only close_token_ids travel
//       (stop_token_ids); the worker gets hard_limit_remaining = 0, which is
//       fine because the head's override arrives inside the token itself.
//   hint_tokens / stall_* pointers, observer   -> spec-only, not replicated.
RequestMsg ClusterHeadBackend::build_request_msg(uint64_t request_id, const GenerateRequest & req,
                                                 int restore_slot, int restore_kv_offset,
                                                 bool spec_available) {
    RequestMsg m;
    m.request_id = request_id;
    m.prompt_tokens = req.prompt;
    m.n_gen = req.n_gen;
    m.max_ctx = 0;  // not part of GenerateRequest; workers use their own --max-ctx
    m.temperature = req.sampler.temp;
    m.top_p = req.sampler.top_p;
    m.top_k = req.sampler.top_k;
    m.min_p = 0.0f;  // SamplerCfg has no min_p
    m.repeat_penalty = req.sampler.rep_pen;
    m.seed = req.sampler.seed;
    const bool budget_requires_ar = !req.budget_hook.close_token_ids.empty();
    const bool sampling_requires_ar = req.sampler.needs_logit_processing();
    m.decode_mode = (spec_available && !req.force_ar_decode && !budget_requires_ar &&
                     !sampling_requires_ar)
        ? DecodeMode::Speculative : DecodeMode::Autoregressive;
    m.force_ar = req.force_ar_decode;
    // Restore and inline save are independent: the HTTP layer may resume
    // from one slot and check-point this request's longer prefix into
    // another. Both have to travel, or the workers' slots drift away from the
    // head's and a later restore silently resumes different KV.
    m.restore_slot   = restore_slot;
    m.kv_offset      = restore_slot >= 0 ? restore_kv_offset : 0;
    m.snapshot_slot  = req.snap_slot;
    m.snapshot_pos   = req.snap_pos;
    m.stop_token_ids = req.budget_hook.close_token_ids;
    return m;
}

// ─── Lifecycle ──────────────────────────────────────────────────────────

ClusterHeadBackend::ClusterHeadBackend(std::unique_ptr<common::ModelBackend> inner,
                                       const ClusterConfig & cfg,
                                       std::string model_path,
                                       int device)
    : inner_(std::move(inner)), cfg_(cfg), model_path_(std::move(model_path)), device_(device) {}

ClusterHeadBackend::~ClusterHeadBackend() {
    if (!shut_down_) shutdown();
}

bool ClusterHeadBackend::init() {
    if (!inner_) {
        std::fprintf(stderr, "[cluster] head: no inner backend\n");
        return false;
    }
    const std::string cfg_err = cfg_.validate();
    if (!cfg_err.empty() || !cfg_.is_head()) {
        std::fprintf(stderr, "[cluster] head: invalid cluster config: %s\n",
                     cfg_err.empty() ? "rank is not 0" : cfg_err.c_str());
        return false;
    }
    // server_main already exported the NCCL_* environment before the
    // factory ran; nothing to repeat here.

    std::string err;
    ControlTimeouts timeouts;
    timeouts.recv_ms = cfg_.timeout_ms;
    control_.set_timeouts(timeouts);
    if (!control_.listen(cfg_.head_host, cfg_.head_port, &err)) {
        std::fprintf(stderr, "[cluster] head: listen %s:%d failed: %s\n",
                     cfg_.head_host.c_str(), cfg_.head_port, err.c_str());
        return false;
    }

    // The placement hash comes from the inner backend's built placement
    // (backend_factory called set_cluster(&cfg, nullptr) before init()).
    const uint64_t placement_hash = backend_placement_hash(*inner_);
    if (placement_hash == 0) {
        std::fprintf(stderr, "[cluster] head: inner backend has no cluster placement; "
                             "set_cluster() must run before init()\n");
        control_.close();
        return false;
    }
    identity_ = cluster_identity(0, cfg_.size, model_path_, placement_hash);
    std::fprintf(stderr, "[cluster] head rank 0/%d identity build=%s model=%s placement=%s\n",
                 cfg_.size, identity_.build_sha.c_str(), identity_.model_sha.c_str(),
                 hex64(identity_.placement_hash).c_str());
    // Launcher contract (scripts/cluster/launch_cluster.sh waits for this
    // exact stdout prefix before it starts the workers).
    std::printf("cluster: listening on %s:%d, waiting for %d worker(s)\n",
                cfg_.head_host.c_str(), cfg_.head_port, cfg_.size - 1);
    std::fflush(stdout);

    const uint64_t t0 = now_us();
    if (!control_.accept_workers(cfg_.size - 1, identity_, bootstrap_deadline_ms(cfg_), hellos_, &err)) {
        std::fprintf(stderr, "[cluster] head: worker handshake failed: %s\n", err.c_str());
        control_.close();
        return false;
    }
    for (const HelloMsg & h : hellos_) {
        std::fprintf(stderr, "[cluster] head: rank %d joined from %s\n", h.rank, h.hostname.c_str());
    }
    std::printf("cluster: all %d ranks connected\n", cfg_.size);
    std::fflush(stdout);

    RcclUniqueId uid{};
    if (!rccl_get_unique_id(uid, &err)) {
        std::fprintf(stderr, "[cluster] head: rccl_get_unique_id failed: %s\n", err.c_str());
        fail_cluster("rccl_get_unique_id: " + err, kAbortCodeCollective, 0);
        return false;
    }

    WelcomeMsg welcome;
    welcome.size = cfg_.size;
    welcome.rccl_unique_id = uid;
    welcome.placement_hash = identity_.placement_hash;
    welcome.timeout_ms = cfg_.timeout_ms;
    welcome.allreduce_dtype = (uint8_t) cfg_.allreduce_dtype;
    welcome.shared_expert = (uint8_t) cfg_.shared_expert;
    welcome.verify_hash_every = cfg_.verify_hash_every;
    welcome.rank_hostnames.resize((size_t) cfg_.size);
    welcome.rank_hostnames[0] = identity_.hostname;
    for (const HelloMsg & h : hellos_) {
        if (h.rank > 0 && h.rank < cfg_.size) welcome.rank_hostnames[(size_t) h.rank] = h.hostname;
    }
    if (!control_.broadcast(make_frame(welcome), &err)) {
        std::fprintf(stderr, "[cluster] head: Welcome broadcast failed: %s\n", err.c_str());
        fail_cluster("welcome: " + err, kAbortCodeControl, 0);
        return false;
    }

    RcclCommInit init;
    init.rank = 0;
    init.size = cfg_.size;
    init.unique_id = uid;
    init.device = device_;   // local HIP ordinal the inner backend runs on
    init.timeout_ms = cfg_.timeout_ms;
    init.blocking = false;
    comm_ = create_rccl_cluster_comm(init, &err);
    if (!comm_) {
        std::fprintf(stderr, "[cluster] head: RCCL communicator failed: %s\n", err.c_str());
        fail_cluster("rccl init: " + err, kAbortCodeCollective, 0);
        return false;
    }
    std::fprintf(stderr, "[cluster] head: %s communicator up, %d ranks, bootstrap %.2f s\n",
                 comm_->backend_name(), comm_->size(), (now_us() - t0) / 1e6);
    std::printf("cluster: rccl communicator ready\n");
    std::fflush(stdout);

    if (cfg_.selftest) {
        const bool ok = run_cluster_selftest(*comm_, init.device, kSelftestIters, kSelftestSmallN,
                                             kSelftestLargeN, &err);
        std::fprintf(stderr, "[cluster] head: selftest %s%s%s\n", ok ? "PASSED" : "FAILED",
                     err.empty() ? "" : ": ", err.c_str());
        ShutdownMsg bye;
        bye.reason = ok ? 0 : 1;
        control_.broadcast(make_frame(bye), nullptr);
        control_.close();
        comm_.reset();
        std::fflush(nullptr);
        std::exit(ok ? 0 : 1);
    }

    HeadHooksConfig hcfg;
    hcfg.timeout_ms = cfg_.timeout_ms;
    hcfg.verify_hash_every = cfg_.verify_hash_every;
    hcfg.strict_hash = false;
    hcfg.trace = cluster_env_trace();
    hooks_ = std::make_unique<HeadHooks>(control_, hcfg);

    // Second set_cluster call: attaches the communicator to the
    // runtime the factory created before init(); placement-defining fields
    // must be unchanged, which they are (same ClusterConfig).
    common::IClusterParticipant * part = cluster_participant();
    if (!part || !part->cluster_attach(&cfg_, comm_.get())) {
        fail_cluster("cluster_attach(comm) refused by the backend", kAbortCodeCollective, 0);
        return false;
    }
    part->cluster_set_hooks(hooks_.get());

    if (!comm_->barrier(&err)) {
        std::fprintf(stderr, "[cluster] head: post-init barrier failed: %s\n", err.c_str());
        fail_cluster("barrier: " + err, kAbortCodeCollective, 0);
        return false;
    }
    initialized_ = true;
    start_watchdog();
    return true;
}

void ClusterHeadBackend::print_ready_banner() const {
    inner_->print_ready_banner();
    std::printf("[cluster] head ready: size=%d workers=%d rccl=%s verify_hash_every=%d\n",
                cfg_.size, control_.n_workers(), rccl_version_string().c_str(),
                cfg_.verify_hash_every);
    std::fflush(stdout);
}

// ─── Failure ────────────────────────────────────────────────────────────

void ClusterHeadBackend::fail_cluster(const std::string & reason, int code, uint64_t request_id) {
    if (fatal_) return;
    fatal_ = true;
    {
        std::lock_guard<std::mutex> lk(fault_mu_);
        if (fault_reason_.empty()) fault_reason_ = reason;
    }
    std::fprintf(stderr, "[cluster] head: FATAL (%d): %s\n", code, reason.c_str());
    AbortMsg abort;
    abort.rank = 0;
    abort.code = code;
    abort.request_id = request_id;
    abort.reason = reason;
    control_.broadcast(make_frame(abort), nullptr);
    if (comm_) comm_->abort();
    control_.close();
    if (exit_on_abort_) {
        std::fprintf(stderr, "[cluster] head: exiting so the supervisor restarts all ranks\n");
        std::fflush(nullptr);
        std::_Exit(3);
    }
}

// ─── M4: peer-failure watchdog ──────────────────────────────────────────

void ClusterHeadBackend::request_fatal_exit(const std::string & reason, int code,
                                            uint64_t request_id) {
    if (exit_scheduled_.exchange(true)) return;
    std::fprintf(stderr, "[cluster] head: FATAL (%d): %s\n", code, reason.c_str());
    fatal_ = true;
    {
        std::lock_guard<std::mutex> lk(fault_mu_);
        if (fault_reason_.empty()) fault_reason_ = reason;
    }
    // Abort the communicator FIRST: that is what makes a collective that is
    // already enqueued on the GPU return an error instead of waiting for a
    // rank that will never arrive. Without it the forward never returns.
    if (comm_) comm_->abort();
    AbortMsg abort;
    abort.rank = 0;
    abort.code = code;
    abort.request_id = request_id;
    abort.reason = reason;
    control_.broadcast(make_frame(abort), nullptr);
    if (!exit_on_abort_) return;
    // Give the request thread time to unwind and the HTTP layer time to write
    // the error response, then hand over to the supervisor. Detached because
    // this may run from the watchdog itself.
    std::thread([reason]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(kFatalExitGraceMs));
        std::fprintf(stderr,
                     "[cluster] head: exiting so the supervisor restarts all ranks (%s)\n",
                     reason.c_str());
        std::fflush(nullptr);
        std::_Exit(3);
    }).detach();
}

void ClusterHeadBackend::watchdog_loop() {
    while (!watchdog_stop_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWatchdogPeriodMs));
        if (watchdog_stop_.load()) break;
        // Only meaningful while a forward can be blocked in a collective.
        if (!request_in_flight_.load() || fatal_) continue;

        std::string err;
        const std::vector<int> dead = control_.dead_ranks();
        const bool async_failed = comm_ && comm_->async_error(&err);
        if (dead.empty() && !async_failed) continue;

        std::string reason;
        if (!dead.empty()) {
            reason = "rank";
            for (int r : dead) reason += " " + std::to_string(r);
            reason += " stopped responding during a request";
        } else {
            reason = "communicator error during a request: " + err;
        }
        request_fatal_exit(reason, dead.empty() ? kAbortCodeCollective : kAbortCodeControl,
                           last_report_.request_id);
    }
}

void ClusterHeadBackend::start_watchdog() {
    if (watchdog_.joinable()) return;
    watchdog_stop_.store(false);
    watchdog_ = std::thread([this] { watchdog_loop(); });
}

void ClusterHeadBackend::stop_watchdog() {
    watchdog_stop_.store(true);
    if (watchdog_.joinable()) watchdog_.join();
}

// ─── Generation ─────────────────────────────────────────────────────────

bool ClusterHeadBackend::spec_available() const {
    // Rank 0 decides for the whole cluster whether a request decodes
    // speculatively. A worker that happens to lack a drafter must not quietly
    // fall back to AR while the head speculates, so it rejects the request
    // instead (cluster_worker_main.cpp).
    const common::IClusterParticipant * part = cluster_participant();
    return part && part->cluster_spec_decode_ready();
}

GenerateResult ClusterHeadBackend::generate_impl(const GenerateRequest & req, const DaemonIO & io) {
    return run_request(req, io, -1);
}

GenerateResult ClusterHeadBackend::restore_and_generate_impl(int slot, const GenerateRequest & req,
                                                             const DaemonIO & io) {
    return run_request(req, io, slot);
}

GenerateResult ClusterHeadBackend::run_request(const GenerateRequest & req, const DaemonIO & io,
                                               int restore_slot) {
    GenerateResult result;
    if (!initialized_ || fatal_) {
        result.fail(GenerateErrorCode::DecodeFailed,
                    fatal_ ? "cluster aborted" : "cluster not initialized");
        return result;
    }
    const uint64_t request_id = next_request_id_++;
    const int restore_kv_offset = restore_slot >= 0 ? inner_->snapshot_cur_pos(restore_slot) : 0;
    const RequestMsg msg = build_request_msg(request_id, req, restore_slot, restore_kv_offset,
                                            spec_available());

    hooks_->set_current_request(request_id);
    hooks_->reset_counters();
    if (comm_) comm_->reset_stats();
    last_report_ = ClusterRequestReport{};
    last_report_.request_id = request_id;
    last_report_.size = cfg_.size;

    std::string err;
    if (!control_.broadcast(make_frame(msg), &err)) {
        fail_cluster("request broadcast: " + err, kAbortCodeControl, request_id);
        result.fail(GenerateErrorCode::DecodeFailed, "cluster: request broadcast failed: " + err);
        return result;
    }

    const auto compute_t0 = std::chrono::steady_clock::now();
    request_in_flight_.store(true);
    result = restore_slot >= 0 ? inner_->restore_and_generate_impl(restore_slot, req, io)
                               : inner_->generate_impl(req, io);
    request_in_flight_.store(false);
    // Rank 0's own prefill+decode wall time, the counterpart of the workers'
    // compute_us. Comparing them across ranks is how an imbalanced expert
    // placement or a slow node shows up.
    head_compute_us_ = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - compute_t0).count();

    // The decode loop reports channel failures as DecodeFailed with the
    // message already logged; a dead worker also shows up here.
    if (!result.ok() && (!control_.all_alive() || (comm_ && comm_->async_error(&err)))) {
        fail_cluster("request " + std::to_string(request_id) + " failed: " +
                     std::string(result.error_detail()) + (err.empty() ? "" : " / " + err),
                     kAbortCodeRequest, request_id);
        return result;
    }

    RequestEndMsg end;
    end.request_id = request_id;
    end.steps = (uint32_t) result.tokens.size();
    end.status = result.ok() ? 0 : (int32_t) result.error->code;
    if (!control_.broadcast(make_frame(end), &err)) {
        fail_cluster("RequestEnd broadcast: " + err, kAbortCodeControl, request_id);
        if (result.ok()) result.fail(GenerateErrorCode::DecodeFailed, "cluster: RequestEnd failed");
        return result;
    }
    if (!gather_reports(request_id, end.steps)) {
        fail_cluster("RequestReport gather: " + last_report_.error, kAbortCodeControl, request_id);
        if (result.ok()) result.fail(GenerateErrorCode::DecodeFailed, "cluster: report gather failed");
        return result;
    }
    return result;
}

bool ClusterHeadBackend::gather_reports(uint64_t request_id, uint32_t steps) {
    last_report_.steps = steps;
    last_report_.ranks.assign((size_t) cfg_.size, RequestReportMsg{});
    RequestReportMsg & head = last_report_.ranks[0];
    head.request_id = request_id;
    head.rank = 0;
    head.steps = steps;
    head.compute_us = head_compute_us_;
    if (comm_) {
        const ClusterCommStats s = comm_->stats();
        head.allreduce_calls = s.allreduce_calls;
        head.allreduce_bytes = s.allreduce_bytes;
        head.allreduce_wait_us = s.allreduce_wait_us;
    }
    head.ctrl_wait_us = hooks_->counters().ctrl_wait_us;
    const uint64_t head_device_bytes = cluster_device_bytes_in_use(device_);
    if (head_device_bytes > peak_device_bytes_) peak_device_bytes_ = head_device_bytes;
    head.peak_device_bytes = peak_device_bytes_;
    last_report_.head_ctrl_wait_us = head.ctrl_wait_us;
    last_report_.hash_probes = hooks_->counters().hash_probes;
    last_report_.hash_mismatches = hooks_->counters().hash_mismatches;
    last_report_.first_mismatch_rank = hooks_->first_mismatch_rank();
    last_report_.first_mismatch_step = hooks_->first_mismatch_step();

    std::vector<Frame> frames;
    std::string err;
    if (!control_.gather(MsgType::RequestReport, cfg_.timeout_ms, frames, &err)) {
        last_report_.error = err;
        return false;
    }
    for (size_t i = 0; i < frames.size(); i++) {
        RequestReportMsg r;
        if (!parse_frame(frames[i], r)) {
            last_report_.error = "malformed RequestReport from rank " + std::to_string(i + 1);
            return false;
        }
        if (r.request_id != request_id) {
            last_report_.error = "RequestReport from rank " + std::to_string(r.rank) +
                                 " is for request " + std::to_string(r.request_id);
            return false;
        }
        if (r.rank <= 0 || r.rank >= cfg_.size) {
            last_report_.error = "RequestReport with bad rank " + std::to_string(r.rank);
            return false;
        }
        last_report_.ranks[(size_t) r.rank] = r;
    }
    last_report_.complete = true;
    if (cluster_env_trace()) {
        for (const RequestReportMsg & r : last_report_.ranks) {
            std::fprintf(stderr,
                "[cluster] req=%llu rank=%d steps=%u compute_us=%llu allreduce=%llu calls/%llu B/%llu us "
                "ctrl_wait_us=%llu\n",
                (unsigned long long) request_id, r.rank, r.steps, (unsigned long long) r.compute_us,
                (unsigned long long) r.allreduce_calls, (unsigned long long) r.allreduce_bytes,
                (unsigned long long) r.allreduce_wait_us, (unsigned long long) r.ctrl_wait_us);
        }
    }
    return true;
}

// ─── Replicated mutations ───────────────────────────────────────────────

bool ClusterHeadBackend::broadcast_op(BackendOpKind kind, std::vector<int64_t> args) {
    if (!initialized_ || fatal_) return false;
    BackendOpMsg op;
    op.request_id = 0;  // process-level; ops never run while a request is in flight (single stream)
    op.kind = kind;
    op.args = std::move(args);
    std::string err;
    if (!hooks_->backend_op(op, &err)) {
        fail_cluster("BackendOp broadcast: " + err, kAbortCodeControl, 0);
        return false;
    }
    return true;
}

bool ClusterHeadBackend::park(ParkTarget target) {
    if (!broadcast_op(BackendOpKind::Park, {(int64_t) target})) return false;
    return inner_->park(target);
}

bool ClusterHeadBackend::unpark(ParkTarget target) {
    if (!broadcast_op(BackendOpKind::Unpark, {(int64_t) target})) return false;
    return inner_->unpark(target);
}

bool ClusterHeadBackend::is_target_parked() const { return inner_->is_target_parked(); }

bool ClusterHeadBackend::snapshot_save(int slot) {
    if (!broadcast_op(BackendOpKind::SnapshotSave, {slot})) return false;
    if (inner_->snapshot_save(slot)) return true;
    // The workers have taken the snapshot and this rank has not. Leaving it
    // there would make a later restore resume from a slot only some ranks
    // hold, which diverges silently. Drop it everywhere instead.
    std::fprintf(stderr,
                 "[cluster] head: snapshot_save(%d) failed here; freeing the slot on "
                 "every rank so no rank restores from it\n", slot);
    broadcast_op(BackendOpKind::SnapshotFree, {slot});
    return false;
}

void ClusterHeadBackend::snapshot_free(int slot) {
    broadcast_op(BackendOpKind::SnapshotFree, {slot});
    inner_->snapshot_free(slot);
}

bool ClusterHeadBackend::snapshot_used(int slot) const { return inner_->snapshot_used(slot); }
int  ClusterHeadBackend::snapshot_cur_pos(int slot) const { return inner_->snapshot_cur_pos(slot); }

common::ModelBackend::SnapshotRef ClusterHeadBackend::snapshot_ref(int slot) const {
    return inner_->snapshot_ref(slot);
}

bool ClusterHeadBackend::snapshot_adopt(int slot, ggml_context * ctx, ggml_backend_buffer_t buf,
                                        int cur_pos, int32_t last_tok) {
    // Adopting a deserialized snapshot on the head only would leave the
    // workers without it; refusing keeps every rank identical (M4: ship it).
    (void) slot; (void) ctx; (void) buf; (void) cur_pos; (void) last_tok;
    std::fprintf(stderr, "[cluster] head: snapshot_adopt is not replicated; refusing\n");
    return false;
}

common::ModelBackend::CompressResult ClusterHeadBackend::compress(const CompressRequest & req) {
    return inner_->compress(req);
}

std::vector<common::ModelBackend::CompressResult> ClusterHeadBackend::compress_batch(
        const std::vector<CompressRequest> & requests) {
    return inner_->compress_batch(requests);
}

bool ClusterHeadBackend::handle_compress(const std::string & line, const DaemonIO & io) {
    // DeepSeek4Backend::handle_compress is unsupported and mutates nothing,
    // so no BackendOp is sent; revisit if PFlash lands for DS4.
    return inner_->handle_compress(line, io);
}

void ClusterHeadBackend::free_drafter() {
    broadcast_op(BackendOpKind::FreeDrafter, {});
    inner_->free_drafter();
}

bool ClusterHeadBackend::try_handle_command(const std::string & line, const DaemonIO & io) {
    return inner_->try_handle_command(line, io);
}

bool ClusterHeadBackend::supports_dflash_spec_decode() const {
    return inner_->supports_dflash_spec_decode();
}
common::DFlashTarget * ClusterHeadBackend::dflash_target() { return inner_->dflash_target(); }
void ClusterHeadBackend::release_scratch() { inner_->release_scratch(); }
bool ClusterHeadBackend::supports_remote_draft() const { return inner_->supports_remote_draft(); }
bool ClusterHeadBackend::supports_kvflash() const { return inner_->supports_kvflash(); }
bool ClusterHeadBackend::supports_mixed_backend_layer_split() const {
    return inner_->supports_mixed_backend_layer_split();
}
bool ClusterHeadBackend::set_routing_collector(common::MoeRoutingCollector * collector) {
    return inner_->set_routing_collector(collector);
}
const common::MoeHybridRoutingStats * ClusterHeadBackend::get_routing_stats() const {
    return inner_->get_routing_stats();
}
bool ClusterHeadBackend::spark_wants_bootstrap() const { return inner_->spark_wants_bootstrap(); }
bool ClusterHeadBackend::spark_bootstrap_finalize(const std::string & profile_path) {
    return inner_->spark_bootstrap_finalize(profile_path);
}

void ClusterHeadBackend::shutdown() {
    if (shut_down_) return;
    shut_down_ = true;
    stop_watchdog();
    if (initialized_ && !fatal_) {
        ShutdownMsg bye;
        bye.reason = 0;
        control_.broadcast(make_frame(bye), nullptr);
    }
    if (inner_) {
        if (common::IClusterParticipant * part = cluster_participant()) {
            part->cluster_set_hooks(nullptr);
        }
        inner_->shutdown();
    }
    control_.close();
    comm_.reset();
    hooks_.reset();
}

}  // namespace luce::cluster
