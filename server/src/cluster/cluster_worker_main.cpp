// cluster_worker_main.cpp - lockstep worker loop for ranks 1..N-1.

#include "cluster/cluster_worker_main.h"

#include "cluster/cluster_comm.h"
#include "cluster/cluster_config.h"
#include "cluster/cluster_control.h"
#include "cluster/cluster_decision_hooks.h"
#include "cluster/cluster_head_backend.h"   // backend_placement_hash
#include "cluster/cluster_identity.h"
#include "cluster/cluster_protocol.h"
#include "common/backend_factory.h"
#include "common/cluster_participant.h"
#include "common/model_backend.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace luce::cluster {

using common::DaemonIO;
using common::GenerateRequest;
using common::GenerateResult;

namespace {

constexpr int kExitOk          = 0;
constexpr int kExitConfig      = 2;
constexpr int kExitAborted     = 3;
constexpr int kExitBackend     = 4;
constexpr int kExitProtocol    = 5;

constexpr int    kSelftestIters  = 1000;
constexpr size_t kSelftestSmallN = 64u * 1024u / sizeof(float);
constexpr size_t kSelftestLargeN = 16u * 1024u * 1024u / sizeof(float);

uint64_t now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// RequestMsg -> GenerateRequest. Inverse of ClusterHeadBackend::
// build_request_msg; see the list there of the fields that do not travel.
GenerateRequest request_from_msg(const RequestMsg & m) {
    GenerateRequest req;
    req.prompt = m.prompt_tokens;
    req.n_gen = m.n_gen;
    req.sampler.temp = m.temperature;
    req.sampler.top_p = m.top_p;
    req.sampler.top_k = m.top_k;
    req.sampler.rep_pen = m.repeat_penalty;
    req.sampler.seed = m.seed;
    req.do_sample = m.temperature > 0.0f;
    req.stream = false;
    req.force_ar_decode = m.force_ar || m.decode_mode == DecodeMode::Autoregressive;
    // Since protocol 2 both halves travel, so this rank writes the same
    // inline snapshot at the same position as the head.
    req.snap_slot = m.snapshot_slot;
    req.snap_pos = m.snapshot_pos;
    req.budget_hook.close_token_ids = m.stop_token_ids;
    req.budget_hook.hard_limit_remaining = 0;  // the head's override arrives inside the token
    return req;
}

struct WorkerState {
    ClusterConfig                              cfg;
    std::unique_ptr<common::ModelBackend>      backend_owner;
    // The backend as the worker replays requests into it, and the same object
    // seen as a cluster participant. Both are borrowed from backend_owner.
    common::ModelBackend *        model = nullptr;
    common::IClusterParticipant * part  = nullptr;
    ClusterWorkerControl                       control;
    std::unique_ptr<IClusterComm>              comm;
    std::unique_ptr<WorkerHooks>               hooks;
    uint64_t                                   requests_served = 0;
    int                                        device = -1;   // local HIP ordinal
    // Watchdog: with the in-graph all-reduce nothing on the
    // host waits for a collective, so a dead head would leave this rank
    // blocked inside HIP forever. The thread aborts the communicator, which
    // makes the pending collective return an error.
    std::thread                                watchdog;
    std::atomic<bool>                          watchdog_stop{false};
    std::atomic<bool>                          request_in_flight{false};
    // High-water mark of device memory in use, sampled at request ends.
    uint64_t                                   peak_device_bytes = 0;
};

void send_abort(WorkerState & st, int code, uint64_t request_id, const std::string & reason) {
    std::fprintf(stderr, "[cluster] worker %d: ABORT (%d): %s\n", st.cfg.rank, code, reason.c_str());
    AbortMsg abort;
    abort.rank = st.cfg.rank;
    abort.code = code;
    abort.request_id = request_id;
    abort.reason = reason;
    st.control.send(make_frame(abort), nullptr);
    if (st.comm) st.comm->abort();
}

void teardown(WorkerState & st) {
    st.watchdog_stop.store(true);
    if (st.watchdog.joinable()) st.watchdog.join();
    if (st.part) st.part->cluster_set_hooks(nullptr);
    if (st.backend_owner) st.backend_owner->shutdown();
    st.control.close();
    st.comm.reset();
    st.hooks.reset();
}

// Executes one BackendOpMsg on the local backend. Returns false on an
// argument error (the head never sends those; treat as protocol failure).
bool apply_backend_op(WorkerState & st, const BackendOpMsg & op) {
    auto arg = [&](size_t i, int64_t def) { return i < op.args.size() ? op.args[i] : def; };
    switch (op.kind) {
    case BackendOpKind::SnapshotSave:
        if (!st.model->snapshot_save((int) arg(0, -1))) {
            // A rank without the snapshot would silently resume different KV
            // on the next restore. Fail the run instead.
            std::fprintf(stderr,
                         "[cluster] worker %d: snapshot_save(%lld) failed; the slot would "
                         "differ from the head's\n",
                         st.cfg.rank, (long long) arg(0, -1));
            return false;
        }
        return true;
    case BackendOpKind::SnapshotFree:
        st.model->snapshot_free((int) arg(0, -1));
        return true;
    case BackendOpKind::SnapshotRestore:
        // Restore is carried by RequestMsg (snapshot_slot + kv_offset), not
        // as a standalone op; accept and ignore for forward compatibility.
        return true;
    case BackendOpKind::Park:
        if (!st.model->park((common::ParkTarget) arg(0, (int64_t) common::ParkTarget::Empty))) {
            std::fprintf(stderr, "[cluster] worker %d: park failed\n", st.cfg.rank);
        }
        return true;
    case BackendOpKind::Unpark:
        if (!st.model->unpark((common::ParkTarget) arg(0, (int64_t) common::ParkTarget::Empty))) {
            std::fprintf(stderr, "[cluster] worker %d: unpark failed\n", st.cfg.rank);
            return false;
        }
        return true;
    case BackendOpKind::FreeDrafter:
        st.model->free_drafter();
        return true;
    case BackendOpKind::ResetRequestState:
        st.model->release_scratch();
        return true;
    case BackendOpKind::HandleCompress:
        // DeepSeek4Backend::handle_compress is unsupported upstream; nothing to mirror.
        return true;
    }
    return false;
}

void worker_watchdog_loop(WorkerState & st) {
    constexpr int kPeriodMs = 250;
    while (!st.watchdog_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kPeriodMs));
        if (st.watchdog_stop.load()) break;
        if (!st.request_in_flight.load()) continue;
        std::string err;
        const bool head_gone = !st.control.alive();
        const bool async_failed = st.comm && st.comm->async_error(&err);
        if (!head_gone && !async_failed) continue;
        std::fprintf(stderr,
                     "[cluster] worker %d: %s during a request; aborting the communicator\n",
                     st.cfg.rank,
                     head_gone ? "the head stopped responding"
                               : ("communicator error: " + err).c_str());
        if (st.comm) st.comm->abort();
        // Leave the exit to the request path: the aborted collective makes the
        // forward fail, handle_request returns, and the loop tears down.
        break;
    }
}

// Returns the exit code to use, or -1 to keep serving.
int handle_request(WorkerState & st, const RequestMsg & msg) {
    const uint64_t t0 = now_us();
    GenerateRequest req = request_from_msg(msg);
    DaemonIO io;            // discarding sink: no fd, no callback, never cancels
    io.stream_fd = -1;

    st.hooks->set_current_request(msg.request_id);
    st.hooks->reset_counters();
    if (st.comm) st.comm->reset_stats();
    st.request_in_flight.store(true);
    struct InFlightGuard {
        WorkerState & st;
        ~InFlightGuard() { st.request_in_flight.store(false); }
    } in_flight_guard{st};

    // The head speculates only when it broadcast DecodeMode::Speculative.
    // A worker without a drafter would silently decode AR instead and then
    // wait for a Decision frame that never comes, so fail the run loudly.
    if (msg.decode_mode == DecodeMode::Speculative && !st.part->cluster_spec_decode_ready()) {
        send_abort(st, kExitProtocol, msg.request_id,
                   "head requested speculative decode but this rank has no DSpark drafter "
                   "(pass the same --draft / LUCE_DS4_SPEC settings to every rank)");
        return kExitProtocol;
    }

    GenerateResult result;
    if (msg.restore_slot >= 0) {
        result = st.model->restore_and_generate_impl(msg.restore_slot, req, io);
    } else {
        result = st.model->generate_impl(req, io);
    }
    const uint64_t compute_us = now_us() - t0;

    // The head always follows the request with RequestEnd; if our decode
    // loop already consumed it (head failed the request early) it is parked.
    Frame frame;
    std::string err;
    bool have_end = false;
    if (st.hooks->take_pending_frame(frame)) {
        have_end = frame.type == MsgType::RequestEnd;
    } else if (st.hooks->aborted()) {
        std::fprintf(stderr, "[cluster] worker %d: head aborted request %llu: %s\n", st.cfg.rank,
                     (unsigned long long) msg.request_id, st.hooks->abort_reason().c_str());
        return kExitAborted;
    } else if (st.control.recv(frame, st.cfg.timeout_ms, &err)) {
        have_end = frame.type == MsgType::RequestEnd;
    } else {
        send_abort(st, kExitProtocol, msg.request_id, "no RequestEnd: " + err);
        return kExitProtocol;
    }

    if (!have_end) {
        if (frame.type == MsgType::Abort) {
            AbortMsg abort;
            parse_frame(frame, abort);
            std::fprintf(stderr, "[cluster] worker %d: head aborted: %s\n", st.cfg.rank,
                         abort.reason.c_str());
            return kExitAborted;
        }
        if (frame.type == MsgType::Shutdown) return kExitOk;
        send_abort(st, kExitProtocol, msg.request_id,
                   std::string("expected RequestEnd, got ") + msg_type_name(frame.type));
        return kExitProtocol;
    }
    RequestEndMsg end;
    if (!parse_frame(frame, end) || end.request_id != msg.request_id) {
        send_abort(st, kExitProtocol, msg.request_id, "RequestEnd for a different request");
        return kExitProtocol;
    }

    if (!result.ok() && end.status == 0) {
        // Head succeeded, we did not: replicated state has diverged.
        send_abort(st, kExitBackend, msg.request_id,
                   "worker generate failed while head succeeded: " + std::string(result.error_detail()));
        return kExitBackend;
    }
    if (result.ok() && (uint32_t) result.tokens.size() != end.steps) {
        send_abort(st, kExitBackend, msg.request_id,
                   "token count " + std::to_string(result.tokens.size()) + " != head " +
                   std::to_string(end.steps));
        return kExitBackend;
    }

    RequestReportMsg report;
    report.request_id = msg.request_id;
    report.rank = st.cfg.rank;
    report.steps = (uint32_t) result.tokens.size();
    report.compute_us = compute_us;
    if (st.comm) {
        const ClusterCommStats s = st.comm->stats();
        report.allreduce_calls = s.allreduce_calls;
        report.allreduce_bytes = s.allreduce_bytes;
        report.allreduce_wait_us = s.allreduce_wait_us;
    }
    report.ctrl_wait_us = st.hooks->counters().ctrl_wait_us;
    const uint64_t device_bytes = cluster_device_bytes_in_use(st.device);
    if (device_bytes > st.peak_device_bytes) st.peak_device_bytes = device_bytes;
    report.peak_device_bytes = st.peak_device_bytes;
    if (!st.control.send(make_frame(report), &err)) {
        std::fprintf(stderr, "[cluster] worker %d: RequestReport send failed: %s\n", st.cfg.rank,
                     err.c_str());
        return kExitProtocol;
    }
    st.requests_served++;
    if (cluster_env_trace()) {
        std::fprintf(stderr, "[cluster] worker %d: req=%llu %zu tokens in %.3f s (ctrl wait %.1f ms)\n",
                     st.cfg.rank, (unsigned long long) msg.request_id, result.tokens.size(),
                     compute_us / 1e6, report.ctrl_wait_us / 1e3);
    }
    return -1;
}

}  // namespace

int run_cluster_worker(const ClusterConfig & cfg,
                       std::unique_ptr<common::ModelBackend> backend,
                       const std::string & model_path,
                       int device) {
    WorkerState st;
    st.cfg = cfg;
    const std::string cfg_err = st.cfg.validate();
    if (!cfg_err.empty() || !st.cfg.is_worker()) {
        std::fprintf(stderr, "[cluster] worker: invalid cluster config: %s\n",
                     cfg_err.empty() ? "rank must be >= 1" : cfg_err.c_str());
        return kExitConfig;
    }
    // NCCL_* environment: server_main exported it before the backend loaded.
    st.backend_owner = std::move(backend);
    if (!st.backend_owner) {
        std::fprintf(stderr, "[cluster] worker %d: no backend\n", st.cfg.rank);
        return kExitBackend;
    }
    st.model = st.backend_owner.get();
    st.part  = dynamic_cast<common::IClusterParticipant *>(st.model);
    if (!st.part) {
        std::fprintf(stderr, "[cluster] worker %d: this architecture does not "
                             "implement IClusterParticipant, so it cannot shard\n",
                     st.cfg.rank);
        teardown(st);
        return kExitConfig;
    }

    // Join the cluster. Identity fields are computed by cluster_identity.h on
    // every rank; the placement hash is this rank's built placement.
    const uint64_t placement_hash = backend_placement_hash(*st.model);
    if (placement_hash == 0) {
        std::fprintf(stderr, "[cluster] worker %d: backend has no cluster placement\n", st.cfg.rank);
        teardown(st);
        return kExitBackend;
    }
    const HelloMsg hello = cluster_identity(st.cfg.rank, st.cfg.size, model_path, placement_hash);
    ControlTimeouts timeouts;
    timeouts.recv_ms = st.cfg.timeout_ms;
    st.control.set_timeouts(timeouts);
    WelcomeMsg welcome;
    std::string err;
    std::fprintf(stderr, "[cluster] worker %d/%d connecting to %s:%d (build=%s model=%s)\n",
                 st.cfg.rank, st.cfg.size, st.cfg.head_host.c_str(), st.cfg.head_port,
                 hello.build_sha.c_str(), hello.model_sha.c_str());
    if (!st.control.connect_and_handshake(st.cfg.head_host, st.cfg.head_port, hello, welcome, &err)) {
        std::fprintf(stderr, "[cluster] worker %d: handshake failed: %s\n", st.cfg.rank, err.c_str());
        teardown(st);
        return kExitProtocol;
    }
    if (welcome.size != st.cfg.size) {
        std::fprintf(stderr, "[cluster] worker %d: head reports size %d, we were started with %d\n",
                     st.cfg.rank, welcome.size, st.cfg.size);
        teardown(st);
        return kExitConfig;
    }
    // The head's values win for knobs that must agree cluster-wide.
    // shared_expert is placement-defining (set_cluster refuses a change after
    // init), so a mismatch is an operator error rather than a silent override.
    st.cfg.timeout_ms = welcome.timeout_ms ? welcome.timeout_ms : st.cfg.timeout_ms;
    st.cfg.verify_hash_every = welcome.verify_hash_every;
    st.cfg.allreduce_dtype = (AllreduceDType) welcome.allreduce_dtype;
    if ((SharedExpertMode) welcome.shared_expert != st.cfg.shared_expert) {
        std::fprintf(stderr, "[cluster] worker %d: --cluster-shared-expert differs from the head "
                             "(%s vs %s)\n", st.cfg.rank,
                     shared_expert_mode_name(st.cfg.shared_expert),
                     shared_expert_mode_name((SharedExpertMode) welcome.shared_expert));
        teardown(st);
        return kExitConfig;
    }

    RcclCommInit init;
    init.rank = st.cfg.rank;
    init.size = st.cfg.size;
    init.unique_id = welcome.rccl_unique_id;
    init.device = device;  // local HIP ordinal of the primary GPU
    st.device = init.device;
    init.timeout_ms = st.cfg.timeout_ms;
    init.blocking = false;
    st.comm = create_rccl_cluster_comm(init, &err);
    if (!st.comm) {
        std::fprintf(stderr, "[cluster] worker %d: RCCL communicator failed: %s\n", st.cfg.rank, err.c_str());
        send_abort(st, kExitBackend, 0, "rccl init: " + err);
        teardown(st);
        return kExitBackend;
    }
    std::fprintf(stderr, "[cluster] worker %d: %s communicator up (%d ranks)\n", st.cfg.rank,
                 st.comm->backend_name(), st.comm->size());
    st.watchdog = std::thread([&st] { worker_watchdog_loop(st); });

    if (st.cfg.selftest) {
        const bool ok = run_cluster_selftest(*st.comm, init.device, kSelftestIters, kSelftestSmallN,
                                             kSelftestLargeN, &err);
        std::fprintf(stderr, "[cluster] worker %d: selftest %s%s%s\n", st.cfg.rank,
                     ok ? "PASSED" : "FAILED", err.empty() ? "" : ": ", err.c_str());
        Frame f;
        st.control.recv(f, st.cfg.timeout_ms, nullptr);  // head's Shutdown
        teardown(st);
        return ok ? kExitOk : 1;
    }

    WorkerHooksConfig hcfg;
    hcfg.rank = st.cfg.rank;
    hcfg.timeout_ms = st.cfg.timeout_ms;
    hcfg.verify_hash_every = st.cfg.verify_hash_every;
    st.hooks = std::make_unique<WorkerHooks>(st.control, hcfg);
    // Second set_cluster call: attach the communicator to the
    // runtime the factory created before init().
    if (!st.part->cluster_attach(&st.cfg, st.comm.get())) {
        send_abort(st, kExitBackend, 0, "set_cluster(comm) refused by the backend");
        teardown(st);
        return kExitBackend;
    }
    st.part->cluster_set_hooks(st.hooks.get());

    if (!st.comm->barrier(&err)) {
        std::fprintf(stderr, "[cluster] worker %d: post-init barrier failed: %s\n", st.cfg.rank, err.c_str());
        send_abort(st, kExitBackend, 0, "barrier: " + err);
        teardown(st);
        return kExitBackend;
    }
    std::fprintf(stderr, "[cluster] worker %d ready\n", st.cfg.rank);

    // ── Main loop ─────────────────────────────────────────────────────
    int exit_code = kExitOk;
    for (;;) {
        Frame frame;
        if (!st.hooks->take_pending_frame(frame)) {
            if (!st.control.recv(frame, st.cfg.timeout_ms, &err)) {
                // Idle periods between requests are unbounded; a recv deadline
                // is only fatal when the heartbeat says the head is gone.
                if (st.control.alive()) continue;
                std::fprintf(stderr, "[cluster] worker %d: head connection lost: %s\n", st.cfg.rank,
                             err.c_str());
                if (st.comm) st.comm->abort();
                exit_code = kExitAborted;
                break;
            }
        }
        if (frame.type == MsgType::Request) {
            RequestMsg msg;
            if (!parse_frame(frame, msg)) {
                send_abort(st, kExitProtocol, 0, "malformed Request frame");
                exit_code = kExitProtocol;
                break;
            }
            const int rc = handle_request(st, msg);
            if (rc >= 0) { exit_code = rc; break; }
        } else if (frame.type == MsgType::BackendOp) {
            BackendOpMsg op;
            if (!parse_frame(frame, op) || !apply_backend_op(st, op)) {
                send_abort(st, kExitProtocol, op.request_id, "BackendOp failed");
                exit_code = kExitProtocol;
                break;
            }
        } else if (frame.type == MsgType::RequestEnd) {
            // Stale RequestEnd (e.g. after a head-side prefill failure that
            // our decode never reached); nothing to do.
            continue;
        } else if (frame.type == MsgType::Shutdown) {
            ShutdownMsg bye;
            parse_frame(frame, bye);
            std::fprintf(stderr, "[cluster] worker %d: shutdown (reason %d) after %llu request(s)\n",
                         st.cfg.rank, bye.reason, (unsigned long long) st.requests_served);
            exit_code = kExitOk;
            break;
        } else if (frame.type == MsgType::Abort) {
            AbortMsg abort;
            parse_frame(frame, abort);
            std::fprintf(stderr, "[cluster] worker %d: head abort (%d): %s\n", st.cfg.rank, abort.code,
                         abort.reason.c_str());
            if (st.comm) st.comm->abort();
            exit_code = kExitAborted;
            break;
        } else {
            send_abort(st, kExitProtocol, 0, std::string("unexpected frame ") + msg_type_name(frame.type));
            exit_code = kExitProtocol;
            break;
        }
    }
    teardown(st);
    return exit_code;
}

}  // namespace luce::cluster
