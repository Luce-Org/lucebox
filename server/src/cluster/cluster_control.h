// cluster_control.h - TCP control channel between the head and its workers.
//
// One long-lived TCP connection per worker to the head (rank 0). Frames from
// cluster_protocol.h are written with a 12-byte header and read back with
// exact-length reads; every blocking receive takes a deadline so a dead peer
// turns into an error, never a hang. Heartbeats are sent by an internal
// thread while the channel is idle and their absence for
// `heartbeat_timeout_ms` marks the peer dead.
//
// POSIX sockets only (the whole multi-node path is Linux-only, like the
// existing backend IPC). The header itself is portable so the protocol unit
// test builds on the Windows CI leg without a control channel.
//
// Thread safety: send() may be called from one thread while recv() blocks on
// another; both are internally serialized per direction.

#pragma once

#include "cluster/cluster_protocol.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace luce::cluster {

struct ControlTimeouts {
    uint32_t connect_ms          = 5000;
    uint32_t connect_retries     = 60;      // workers retry until head is up (5 min)
    uint32_t recv_ms             = 30000;   // default deadline for a blocking recv
    uint32_t heartbeat_period_ms = 1000;
    uint32_t heartbeat_timeout_ms = 10000;
};

// A single connected peer (one worker as seen by the head, or the head as
// seen by a worker).
class ControlPeer {
public:
    virtual ~ControlPeer() = default;
    virtual int  peer_rank() const = 0;
    virtual bool send(const Frame & frame, std::string * err) = 0;
    // Blocks until a non-heartbeat frame arrives or deadline_ms elapses.
    // Heartbeats are consumed internally and refresh liveness.
    virtual bool recv(Frame & frame, uint32_t deadline_ms, std::string * err) = 0;
    virtual bool alive() const = 0;
    virtual uint64_t last_heartbeat_us() const = 0;
    virtual void close() = 0;
};

// Head side: accept exactly size-1 workers, each identifying itself with a
// HelloMsg, then fan out frames.
class ClusterHeadControl {
public:
    ClusterHeadControl();
    ~ClusterHeadControl();
    ClusterHeadControl(const ClusterHeadControl &) = delete;
    ClusterHeadControl & operator=(const ClusterHeadControl &) = delete;

    bool listen(const std::string & host, int port, std::string * err);

    // Waits for `n_workers` connections and their Hello frames. Validates
    // rank uniqueness, range [1, n_workers], protocol version, and that
    // build_sha / model_sha / placement_hash equal the head's values (any
    // mismatch -> false with a rank-attributed error). On success
    // hellos[rank-1] is filled and workers are addressable by rank.
    bool accept_workers(int n_workers, const HelloMsg & head_identity, uint32_t deadline_ms,
                        std::vector<HelloMsg> & hellos, std::string * err);

    bool send_to(int rank, const Frame & frame, std::string * err);
    // Sends to every worker; returns false if any send failed (err names the
    // first failing rank).
    bool broadcast(const Frame & frame, std::string * err);
    bool recv_from(int rank, Frame & frame, uint32_t deadline_ms, std::string * err);
    // Collects one frame of the given type from every worker (e.g. all
    // HashProbe or RequestReport messages); out[rank-1] is filled.
    bool gather(MsgType type, uint32_t deadline_ms, std::vector<Frame> & out, std::string * err);

    int  n_workers() const;
    bool all_alive() const;
    // Ranks whose heartbeat is stale; empty when healthy.
    std::vector<int> dead_ranks() const;

    void set_timeouts(const ControlTimeouts & t);
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Worker side: connect to the head, send Hello, receive Welcome.
class ClusterWorkerControl {
public:
    ClusterWorkerControl();
    ~ClusterWorkerControl();
    ClusterWorkerControl(const ClusterWorkerControl &) = delete;
    ClusterWorkerControl & operator=(const ClusterWorkerControl &) = delete;

    // Connects (with retries), sends `hello`, waits for Welcome. Returns
    // false on refusal (the head sends Abort with the mismatch reason).
    bool connect_and_handshake(const std::string & head_host, int head_port, const HelloMsg & hello,
                               WelcomeMsg & welcome, std::string * err);

    bool send(const Frame & frame, std::string * err);
    bool recv(Frame & frame, uint32_t deadline_ms, std::string * err);
    bool alive() const;

    void set_timeouts(const ControlTimeouts & t);
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Utility used by both sides and by tests: exact-length socket I/O with a
// deadline. fd is a connected stream socket. Returns false on EOF, error or
// timeout (err distinguishes them).
bool write_exact_deadline(int fd, const void * data, size_t n, uint32_t deadline_ms, std::string * err);
bool read_exact_deadline(int fd, void * data, size_t n, uint32_t deadline_ms, std::string * err);
bool write_frame(int fd, const Frame & frame, uint32_t deadline_ms, std::string * err);
bool read_frame(int fd, Frame & frame, uint32_t deadline_ms, std::string * err);

// Monotonic microseconds, shared clock for heartbeats and telemetry.
uint64_t monotonic_us();

// Resolve the local hostname for HelloMsg::hostname (never fails; falls back
// to "rank<N>").
std::string local_hostname(int rank);

}  // namespace luce::cluster
