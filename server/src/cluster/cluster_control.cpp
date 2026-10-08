// cluster_control.cpp - POSIX TCP control channel between head and workers
// (cluster_control.h).
//
// Design: one socket per worker. After the Hello/Welcome handshake every peer
// owns a service thread that (a) polls the socket and moves complete frames
// into a queue, filtering heartbeats and refreshing liveness on any inbound
// frame, and (b) sends a HeartbeatMsg every heartbeat_period_ms. recv() pops
// from the queue under a deadline, so a dead peer becomes an error, never a
// hang, even when nobody is currently receiving from it.
//
// Windows: the whole implementation is compiled out and every entry point
// returns false with "cluster control channel is POSIX-only" so the Windows
// CI leg still links the protocol tests.

#include "cluster/cluster_control.h"

#include <chrono>
#include <cstring>

#if !defined(_WIN32)

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

namespace luce::cluster {

namespace {

void set_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

std::string errno_string(const std::string & what) {
    return what + ": " + std::strerror(errno);
}

uint64_t deadline_from_now_us(uint32_t ms) {
    return monotonic_us() + (uint64_t)ms * 1000ull;
}

// Remaining milliseconds until an absolute monotonic deadline (>= 0).
int remaining_ms(uint64_t deadline_us) {
    const uint64_t now = monotonic_us();
    if (now >= deadline_us) return 0;
    const uint64_t rem = (deadline_us - now + 999) / 1000;
    return rem > 0x7fffffffull ? 0x7fffffff : (int)rem;
}

// Poll one fd for `events` until the deadline. Returns 1 ready, 0 timeout,
// -1 error (errno set; EINTR retried).
int poll_until(int fd, short events, uint64_t deadline_us) {
    for (;;) {
        struct pollfd p;
        p.fd = fd;
        p.events = events;
        p.revents = 0;
        const int r = ::poll(&p, 1, remaining_ms(deadline_us));
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) return 0;
        if (p.revents & (POLLERR | POLLNVAL)) {
            errno = ECONNRESET;
            return -1;
        }
        // POLLHUP with readable data still delivers the tail; a read returning
        // 0 reports EOF afterwards.
        return 1;
    }
}

bool set_nonblocking(int fd, bool on) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0) return false;
    const int next = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return ::fcntl(fd, F_SETFL, next) == 0;
}

void apply_stream_options(int fd) {
    int one = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    (void)::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
#ifdef TCP_KEEPIDLE
    int idle = 5;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
#endif
#ifdef TCP_KEEPINTVL
    int intvl = 2;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
#endif
#ifdef TCP_KEEPCNT
    int cnt = 5;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
#endif
}

bool write_exact_deadline_impl(int fd, const void * data, size_t n, uint64_t deadline_us,
                               std::string * err) {
    const uint8_t * p = (const uint8_t *)data;
    size_t done = 0;
    while (done < n) {
        const ssize_t w = ::send(fd, p + done, n - done, MSG_NOSIGNAL);
        if (w > 0) {
            done += (size_t)w;
            continue;
        }
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            const int r = poll_until(fd, POLLOUT, deadline_us);
            if (r == 0) {
                set_err(err, "control channel write timed out");
                return false;
            }
            if (r < 0) {
                set_err(err, errno_string("control channel poll(POLLOUT) failed"));
                return false;
            }
            continue;
        }
        set_err(err, w == 0 ? std::string("control channel write returned 0")
                            : errno_string("control channel write failed"));
        return false;
    }
    return true;
}

bool read_exact_deadline_impl(int fd, void * data, size_t n, uint64_t deadline_us,
                              std::string * err) {
    uint8_t * p = (uint8_t *)data;
    size_t done = 0;
    while (done < n) {
        const ssize_t r = ::recv(fd, p + done, n - done, 0);
        if (r > 0) {
            done += (size_t)r;
            continue;
        }
        if (r == 0) {
            set_err(err, "control channel closed by peer (EOF)");
            return false;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            const int pr = poll_until(fd, POLLIN, deadline_us);
            if (pr == 0) {
                set_err(err, "control channel read timed out");
                return false;
            }
            if (pr < 0) {
                set_err(err, errno_string("control channel poll(POLLIN) failed"));
                return false;
            }
            continue;
        }
        set_err(err, errno_string("control channel read failed"));
        return false;
    }
    return true;
}

bool write_frame_impl(int fd, const Frame & frame, uint64_t deadline_us, std::string * err) {
    if (frame.payload.size() > kMaxPayloadBytes) {
        set_err(err, "frame payload too large to send");
        return false;
    }
    FrameHeader h;
    h.type = (uint16_t)frame.type;
    h.payload_len = (uint32_t)frame.payload.size();
    uint8_t hdr[kFrameHeaderBytes];
    encode_header(h, hdr);
    // One buffer so header and payload share a segment for small frames.
    std::vector<uint8_t> buf;
    buf.reserve(kFrameHeaderBytes + frame.payload.size());
    buf.insert(buf.end(), hdr, hdr + kFrameHeaderBytes);
    buf.insert(buf.end(), frame.payload.begin(), frame.payload.end());
    return write_exact_deadline_impl(fd, buf.data(), buf.size(), deadline_us, err);
}

bool read_frame_impl(int fd, Frame & frame, uint64_t deadline_us, std::string * err) {
    uint8_t hdr[kFrameHeaderBytes];
    if (!read_exact_deadline_impl(fd, hdr, kFrameHeaderBytes, deadline_us, err)) return false;
    FrameHeader h;
    if (!decode_header(hdr, h, err)) return false;
    frame.type = (MsgType)h.type;
    frame.payload.resize(h.payload_len);
    if (h.payload_len > 0 &&
        !read_exact_deadline_impl(fd, frame.payload.data(), h.payload_len, deadline_us, err)) {
        return false;
    }
    return true;
}

// Resolve host:port to a list of addresses. `passive` selects AI_PASSIVE for
// bind on an empty host.
bool resolve(const std::string & host, int port, bool passive, struct addrinfo ** out,
             std::string * err) {
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = passive ? AI_PASSIVE : 0;
    if (!passive) hints.ai_flags |= AI_ADDRCONFIG;
    const std::string port_str = std::to_string(port);
    const char * node = host.empty() ? nullptr : host.c_str();
    const int rc = ::getaddrinfo(node, port_str.c_str(), &hints, out);
    if (rc != 0) {
        set_err(err, "getaddrinfo(" + (host.empty() ? std::string("*") : host) + ":" +
                     port_str + ") failed: " + ::gai_strerror(rc));
        return false;
    }
    return true;
}

// Connect with a deadline. Returns the connected fd or -1.
int connect_with_deadline(const std::string & host, int port, uint32_t connect_ms,
                          std::string * err) {
    struct addrinfo * ai = nullptr;
    if (!resolve(host, port, /*passive=*/false, &ai, err)) return -1;

    int fd = -1;
    std::string last_err = "no addresses resolved";
    for (struct addrinfo * a = ai; a != nullptr; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) {
            last_err = errno_string("socket() failed");
            continue;
        }
        if (!set_nonblocking(fd, true)) {
            last_err = errno_string("fcntl(O_NONBLOCK) failed");
            ::close(fd);
            fd = -1;
            continue;
        }
        const uint64_t deadline_us = deadline_from_now_us(connect_ms);
        int rc = ::connect(fd, a->ai_addr, a->ai_addrlen);
        if (rc < 0 && errno == EINPROGRESS) {
            const int pr = poll_until(fd, POLLOUT, deadline_us);
            if (pr == 0) {
                last_err = "connect timed out";
                ::close(fd);
                fd = -1;
                continue;
            }
            if (pr < 0) {
                last_err = errno_string("connect poll failed");
                ::close(fd);
                fd = -1;
                continue;
            }
            int so_err = 0;
            socklen_t len = sizeof(so_err);
            if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len) < 0 || so_err != 0) {
                errno = so_err != 0 ? so_err : errno;
                last_err = errno_string("connect failed");
                ::close(fd);
                fd = -1;
                continue;
            }
            rc = 0;
        }
        if (rc < 0) {
            last_err = errno_string("connect failed");
            ::close(fd);
            fd = -1;
            continue;
        }
        break;
    }
    ::freeaddrinfo(ai);
    if (fd < 0) {
        set_err(err, last_err);
        return -1;
    }
    apply_stream_options(fd);
    return fd;
}

// ─── Peer ───────────────────────────────────────────────────────────────
// One connected socket plus its service thread. `self_rank` is stamped into
// outgoing heartbeats; `rank_` is the remote rank.

class SocketPeer final : public ControlPeer {
public:
    SocketPeer(int fd, int remote_rank, int self_rank, const ControlTimeouts & t)
        : fd_(fd), rank_(remote_rank), self_rank_(self_rank), timeouts_(t),
          last_hb_us_(monotonic_us()) {
        set_nonblocking(fd_, true);
    }

    ~SocketPeer() override { close(); }

    // Starts the service thread. Called once after the handshake.
    void start() {
        std::lock_guard<std::mutex> lk(state_mu_);
        if (running_.load() || closed_.load()) return;
        running_.store(true);
        thread_ = std::thread([this] { service_loop(); });
    }

    int peer_rank() const override { return rank_; }

    bool send(const Frame & frame, std::string * err) override {
        std::lock_guard<std::mutex> lk(send_mu_);
        if (closed_.load()) {
            set_err(err, "control peer rank " + std::to_string(rank_) + " is closed: " + fail_reason());
            return false;
        }
        std::string e;
        if (!write_frame_impl(fd_, frame, deadline_from_now_us(timeouts_.recv_ms), &e)) {
            mark_failed("send " + std::string(msg_type_name(frame.type)) + " to rank " +
                        std::to_string(rank_) + " failed: " + e);
            set_err(err, fail_reason());
            return false;
        }
        return true;
    }

    bool recv(Frame & frame, uint32_t deadline_ms, std::string * err) override {
        const uint64_t deadline_us = deadline_from_now_us(deadline_ms);
        std::unique_lock<std::mutex> lk(queue_mu_);
        for (;;) {
            if (!queue_.empty()) {
                frame = std::move(queue_.front());
                queue_.pop_front();
                return true;
            }
            if (closed_.load()) {
                set_err(err, "control peer rank " + std::to_string(rank_) + " is closed: " + fail_reason());
                return false;
            }
            if (!running_.load()) {
                // Handshake phase: no service thread yet, read synchronously.
                lk.unlock();
                return recv_direct(frame, deadline_us, err);
            }
            const int rem = remaining_ms(deadline_us);
            if (rem <= 0) {
                set_err(err, "timed out waiting for a frame from rank " + std::to_string(rank_));
                return false;
            }
            queue_cv_.wait_for(lk, std::chrono::milliseconds(rem));
        }
    }

    bool alive() const override {
        if (closed_.load()) return false;
        const uint64_t now = monotonic_us();
        const uint64_t last = last_hb_us_.load();
        return now < last + (uint64_t)timeouts_.heartbeat_timeout_ms * 1000ull;
    }

    uint64_t last_heartbeat_us() const override { return last_hb_us_.load(); }

    void close() override {
        bool join = false;
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            stop_ = true;
            if (fd_ >= 0) ::shutdown(fd_, SHUT_RDWR);
            join = running_.load() && thread_.joinable() &&
                   thread_.get_id() != std::this_thread::get_id();
        }
        if (join) thread_.join();
        {
            // send_mu_ so no writer is mid-send on the descriptor we close.
            std::lock_guard<std::mutex> slk(send_mu_);
            std::lock_guard<std::mutex> lk(state_mu_);
            running_.store(false);
            if (fd_ >= 0) {
                ::close(fd_);
                fd_ = -1;
            }
            closed_.store(true);
        }
        queue_cv_.notify_all();
    }

    std::string fail_reason() const {
        std::lock_guard<std::mutex> lk(state_mu_);
        return fail_reason_;
    }

private:
    // Synchronous frame read used before the service thread exists.
    bool recv_direct(Frame & frame, uint64_t deadline_us, std::string * err) {
        for (;;) {
            std::string e;
            if (!read_frame_impl(fd_, frame, deadline_us, &e)) {
                set_err(err, e);
                return false;
            }
            last_hb_us_.store(monotonic_us());
            if (frame.type == MsgType::Heartbeat) continue;
            return true;
        }
    }

    void mark_failed(const std::string & reason) {
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            if (fail_reason_.empty()) fail_reason_ = reason;
            stop_ = true;
        }
        closed_.store(true);
        queue_cv_.notify_all();
    }

    bool send_heartbeat() {
        HeartbeatMsg hb;
        hb.rank = self_rank_;
        hb.mono_us = monotonic_us();
        const Frame f = make_frame(hb);
        std::lock_guard<std::mutex> lk(send_mu_);
        if (closed_.load()) return false;
        std::string e;
        if (!write_frame_impl(fd_, f, deadline_from_now_us(timeouts_.heartbeat_period_ms), &e)) {
            mark_failed("heartbeat to rank " + std::to_string(rank_) + " failed: " + e);
            return false;
        }
        return true;
    }

    void service_loop() {
        uint64_t next_hb_us = monotonic_us();
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(state_mu_);
                if (stop_) break;
            }
            const uint64_t now = monotonic_us();
            if (now >= next_hb_us) {
                if (!send_heartbeat()) break;
                next_hb_us = now + (uint64_t)timeouts_.heartbeat_period_ms * 1000ull;
            }
            const int pr = poll_until(fd_, POLLIN, next_hb_us);
            if (pr < 0) {
                if (errno == EBADF) break;  // closed underneath us
                mark_failed("control socket to rank " + std::to_string(rank_) + " failed: " +
                            std::strerror(errno));
                break;
            }
            if (pr == 0) continue;  // heartbeat due
            Frame f;
            std::string e;
            // A frame is at least partially readable; finish it under the
            // recv deadline so a stalled sender cannot pin this thread.
            if (!read_frame_impl(fd_, f, deadline_from_now_us(timeouts_.recv_ms), &e)) {
                mark_failed("control socket to rank " + std::to_string(rank_) + ": " + e);
                break;
            }
            last_hb_us_.store(monotonic_us());
            if (f.type == MsgType::Heartbeat) continue;
            {
                std::lock_guard<std::mutex> lk(queue_mu_);
                queue_.push_back(std::move(f));
            }
            queue_cv_.notify_all();
        }
        // Detect a peer that stopped heartbeating without closing: the poll
        // above returns on every period, so the liveness check is implicit in
        // alive(); nothing more to do here.
    }

    int fd_;
    const int rank_;
    const int self_rank_;
    ControlTimeouts timeouts_;

    mutable std::mutex state_mu_;
    std::atomic<bool> running_{false};
    bool stop_ = false;
    std::string fail_reason_;
    std::thread thread_;

    std::mutex send_mu_;

    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::deque<Frame> queue_;

    std::atomic<bool> closed_{false};
    std::atomic<uint64_t> last_hb_us_;
};

}  // namespace

// ─── Free helpers ───────────────────────────────────────────────────────

bool write_exact_deadline(int fd, const void * data, size_t n, uint32_t deadline_ms, std::string * err) {
    return write_exact_deadline_impl(fd, data, n, deadline_from_now_us(deadline_ms), err);
}

bool read_exact_deadline(int fd, void * data, size_t n, uint32_t deadline_ms, std::string * err) {
    return read_exact_deadline_impl(fd, data, n, deadline_from_now_us(deadline_ms), err);
}

bool write_frame(int fd, const Frame & frame, uint32_t deadline_ms, std::string * err) {
    return write_frame_impl(fd, frame, deadline_from_now_us(deadline_ms), err);
}

bool read_frame(int fd, Frame & frame, uint32_t deadline_ms, std::string * err) {
    return read_frame_impl(fd, frame, deadline_from_now_us(deadline_ms), err);
}

uint64_t monotonic_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string local_hostname(int rank) {
    char buf[256];
    std::memset(buf, 0, sizeof(buf));
    if (::gethostname(buf, sizeof(buf) - 1) == 0 && buf[0] != '\0') {
        return std::string(buf);
    }
    return "rank" + std::to_string(rank);
}

// ─── ClusterHeadControl ─────────────────────────────────────────────────

struct ClusterHeadControl::Impl {
    int listen_fd = -1;
    ControlTimeouts timeouts;
    // rank -> peer (ranks 1..n_workers)
    std::map<int, std::unique_ptr<SocketPeer>> peers;
    std::mutex mu;

    ~Impl() { close_all(); }

    void close_all() {
        std::map<int, std::unique_ptr<SocketPeer>> dead;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (listen_fd >= 0) {
                ::close(listen_fd);
                listen_fd = -1;
            }
            dead.swap(peers);
        }
        for (auto & kv : dead) kv.second->close();
    }

    SocketPeer * peer(int rank) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = peers.find(rank);
        return it == peers.end() ? nullptr : it->second.get();
    }
};

ClusterHeadControl::ClusterHeadControl() : impl_(new Impl) {}
ClusterHeadControl::~ClusterHeadControl() { close(); }

bool ClusterHeadControl::listen(const std::string & host, int port, std::string * err) {
    if (impl_->listen_fd >= 0) {
        set_err(err, "control channel already listening");
        return false;
    }
    const bool any = host.empty() || host == "0.0.0.0" || host == "*" || host == "::";
    struct addrinfo * ai = nullptr;
    if (!resolve(any ? std::string() : host, port, /*passive=*/true, &ai, err)) return false;

    int fd = -1;
    std::string last_err = "no addresses resolved";
    for (struct addrinfo * a = ai; a != nullptr; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) {
            last_err = errno_string("socket() failed");
            continue;
        }
        int one = 1;
        (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (a->ai_family == AF_INET6) {
            int zero = 0;
            (void)::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof(zero));
        }
        if (::bind(fd, a->ai_addr, a->ai_addrlen) < 0) {
            last_err = errno_string("bind(" + host + ":" + std::to_string(port) + ") failed");
            ::close(fd);
            fd = -1;
            continue;
        }
        if (::listen(fd, 16) < 0) {
            last_err = errno_string("listen() failed");
            ::close(fd);
            fd = -1;
            continue;
        }
        break;
    }
    ::freeaddrinfo(ai);
    if (fd < 0) {
        set_err(err, last_err);
        return false;
    }
    if (!set_nonblocking(fd, true)) {
        set_err(err, errno_string("fcntl(O_NONBLOCK) on listen socket failed"));
        ::close(fd);
        return false;
    }
    impl_->listen_fd = fd;
    return true;
}

bool ClusterHeadControl::accept_workers(int n_workers, const HelloMsg & head_identity,
                                        uint32_t deadline_ms, std::vector<HelloMsg> & hellos,
                                        std::string * err) {
    if (impl_->listen_fd < 0) {
        set_err(err, "accept_workers called before listen()");
        return false;
    }
    if (n_workers < 1) {
        set_err(err, "accept_workers needs at least one worker");
        return false;
    }
    const uint64_t deadline_us = deadline_from_now_us(deadline_ms);
    hellos.assign((size_t)n_workers, HelloMsg{});
    std::vector<bool> seen((size_t)n_workers + 1, false);
    std::map<int, std::unique_ptr<SocketPeer>> accepted;

    // Refuse a worker: send Abort with the reason, close, fail the whole
    // bootstrap so the operator sees one attributable error.
    auto refuse = [&](int fd, int rank, const std::string & reason) -> bool {
        AbortMsg ab;
        ab.rank = 0;
        ab.code = 1;
        ab.reason = reason;
        std::string ignored;
        (void)write_frame_impl(fd, make_frame(ab), deadline_from_now_us(2000), &ignored);
        ::close(fd);
        set_err(err, "worker rank " + std::to_string(rank) + " refused: " + reason);
        for (auto & kv : accepted) kv.second->close();
        return false;
    };

    int n_accepted = 0;
    while (n_accepted < n_workers) {
        const int pr = poll_until(impl_->listen_fd, POLLIN, deadline_us);
        if (pr == 0) {
            set_err(err, "timed out waiting for workers (" + std::to_string(n_accepted) + "/" +
                         std::to_string(n_workers) + " connected)");
            for (auto & kv : accepted) kv.second->close();
            return false;
        }
        if (pr < 0) {
            set_err(err, errno_string("poll on listen socket failed"));
            for (auto & kv : accepted) kv.second->close();
            return false;
        }
        struct sockaddr_storage ss;
        socklen_t sl = sizeof(ss);
        const int fd = ::accept(impl_->listen_fd, (struct sockaddr *)&ss, &sl);
        if (fd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            set_err(err, errno_string("accept() failed"));
            for (auto & kv : accepted) kv.second->close();
            return false;
        }
        apply_stream_options(fd);
        set_nonblocking(fd, true);

        Frame f;
        std::string e;
        if (!read_frame_impl(fd, f, deadline_us, &e)) {
            ::close(fd);
            set_err(err, "reading Hello failed: " + e);
            for (auto & kv : accepted) kv.second->close();
            return false;
        }
        HelloMsg hello;
        if (f.type != MsgType::Hello || !parse_frame(f, hello)) {
            return refuse(fd, -1, "expected a well-formed Hello frame, got " +
                                  std::string(msg_type_name(f.type)));
        }
        if (hello.protocol_version != kProtocolVersion) {
            return refuse(fd, hello.rank, "protocol version " + std::to_string(hello.protocol_version) +
                                          " != head " + std::to_string(kProtocolVersion));
        }
        if (hello.rank < 1 || hello.rank > n_workers) {
            return refuse(fd, hello.rank, "rank out of range [1, " + std::to_string(n_workers) + "]");
        }
        if (seen[(size_t)hello.rank]) {
            return refuse(fd, hello.rank, "duplicate rank");
        }
        if (hello.size != head_identity.size) {
            return refuse(fd, hello.rank, "--cluster-size " + std::to_string(hello.size) +
                                          " != head " + std::to_string(head_identity.size));
        }
        if (hello.build_sha != head_identity.build_sha) {
            return refuse(fd, hello.rank, "build_sha '" + hello.build_sha + "' != head '" +
                                          head_identity.build_sha + "'");
        }
        if (hello.model_sha != head_identity.model_sha) {
            return refuse(fd, hello.rank, "model_sha '" + hello.model_sha + "' != head '" +
                                          head_identity.model_sha + "'");
        }
        if (hello.placement_hash != head_identity.placement_hash) {
            return refuse(fd, hello.rank, "placement_hash mismatch (worker " +
                                          std::to_string(hello.placement_hash) + ", head " +
                                          std::to_string(head_identity.placement_hash) + ")");
        }
        seen[(size_t)hello.rank] = true;
        hellos[(size_t)hello.rank - 1] = hello;
        accepted[hello.rank].reset(new SocketPeer(fd, hello.rank, /*self_rank=*/0, impl_->timeouts));
        ++n_accepted;
    }

    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        for (auto & kv : accepted) {
            kv.second->start();
            impl_->peers[kv.first] = std::move(kv.second);
        }
    }
    return true;
}

bool ClusterHeadControl::send_to(int rank, const Frame & frame, std::string * err) {
    SocketPeer * p = impl_->peer(rank);
    if (!p) {
        set_err(err, "no worker with rank " + std::to_string(rank));
        return false;
    }
    return p->send(frame, err);
}

bool ClusterHeadControl::broadcast(const Frame & frame, std::string * err) {
    std::vector<SocketPeer *> targets;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        for (auto & kv : impl_->peers) targets.push_back(kv.second.get());
    }
    bool ok = true;
    for (SocketPeer * p : targets) {
        std::string e;
        if (!p->send(frame, &e)) {
            if (ok) set_err(err, "broadcast to rank " + std::to_string(p->peer_rank()) + " failed: " + e);
            ok = false;
        }
    }
    return ok;
}

bool ClusterHeadControl::recv_from(int rank, Frame & frame, uint32_t deadline_ms, std::string * err) {
    SocketPeer * p = impl_->peer(rank);
    if (!p) {
        set_err(err, "no worker with rank " + std::to_string(rank));
        return false;
    }
    return p->recv(frame, deadline_ms, err);
}

bool ClusterHeadControl::gather(MsgType type, uint32_t deadline_ms, std::vector<Frame> & out,
                                std::string * err) {
    const int n = n_workers();
    out.assign((size_t)n, Frame{});
    const uint64_t deadline_us = deadline_from_now_us(deadline_ms);
    for (int rank = 1; rank <= n; ++rank) {
        Frame f;
        std::string e;
        const int rem = remaining_ms(deadline_us);
        if (!recv_from(rank, f, (uint32_t)rem, &e)) {
            set_err(err, "gather(" + std::string(msg_type_name(type)) + ") from rank " +
                         std::to_string(rank) + " failed: " + e);
            return false;
        }
        if (f.type != type) {
            if (f.type == MsgType::Abort) {
                AbortMsg ab;
                if (parse_frame(f, ab)) {
                    set_err(err, "rank " + std::to_string(rank) + " aborted: " + ab.reason);
                    return false;
                }
            }
            set_err(err, "gather(" + std::string(msg_type_name(type)) + ") from rank " +
                         std::to_string(rank) + " got " + msg_type_name(f.type));
            return false;
        }
        out[(size_t)rank - 1] = std::move(f);
    }
    return true;
}

int ClusterHeadControl::n_workers() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return (int)impl_->peers.size();
}

bool ClusterHeadControl::all_alive() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (const auto & kv : impl_->peers) {
        if (!kv.second->alive()) return false;
    }
    return true;
}

std::vector<int> ClusterHeadControl::dead_ranks() const {
    std::vector<int> out;
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (const auto & kv : impl_->peers) {
        if (!kv.second->alive()) out.push_back(kv.first);
    }
    return out;
}

void ClusterHeadControl::set_timeouts(const ControlTimeouts & t) {
    impl_->timeouts = t;
}

void ClusterHeadControl::close() {
    if (impl_) impl_->close_all();
}

// ─── ClusterWorkerControl ───────────────────────────────────────────────

struct ClusterWorkerControl::Impl {
    ControlTimeouts timeouts;
    std::unique_ptr<SocketPeer> head;
    std::mutex mu;

    ~Impl() { close_head(); }

    void close_head() {
        std::unique_ptr<SocketPeer> dead;
        {
            std::lock_guard<std::mutex> lk(mu);
            dead = std::move(head);
        }
        if (dead) dead->close();
    }
};

ClusterWorkerControl::ClusterWorkerControl() : impl_(new Impl) {}
ClusterWorkerControl::~ClusterWorkerControl() { close(); }

bool ClusterWorkerControl::connect_and_handshake(const std::string & head_host, int head_port,
                                                 const HelloMsg & hello, WelcomeMsg & welcome,
                                                 std::string * err) {
    if (impl_->head) {
        set_err(err, "worker control channel already connected");
        return false;
    }
    const ControlTimeouts & t = impl_->timeouts;
    int fd = -1;
    std::string last_err;
    const uint32_t attempts = t.connect_retries == 0 ? 1u : t.connect_retries;
    for (uint32_t attempt = 0; attempt < attempts && fd < 0; ++attempt) {
        fd = connect_with_deadline(head_host, head_port, t.connect_ms, &last_err);
        if (fd < 0 && attempt + 1 < attempts) {
            const uint32_t sleep_ms = t.connect_ms < 1000 ? t.connect_ms : 1000;
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }
    if (fd < 0) {
        set_err(err, "connect to head " + head_host + ":" + std::to_string(head_port) +
                     " failed after " + std::to_string(attempts) + " attempts: " + last_err);
        return false;
    }
    set_nonblocking(fd, true);

    std::string e;
    if (!write_frame_impl(fd, make_frame(hello), deadline_from_now_us(t.recv_ms), &e)) {
        ::close(fd);
        set_err(err, "sending Hello failed: " + e);
        return false;
    }
    Frame f;
    for (;;) {
        if (!read_frame_impl(fd, f, deadline_from_now_us(t.recv_ms), &e)) {
            ::close(fd);
            set_err(err, "waiting for Welcome failed: " + e);
            return false;
        }
        if (f.type == MsgType::Heartbeat) continue;
        break;
    }
    if (f.type == MsgType::Abort) {
        AbortMsg ab;
        ::close(fd);
        if (parse_frame(f, ab)) {
            set_err(err, "head refused this worker: " + ab.reason);
        } else {
            set_err(err, "head refused this worker (unparseable Abort)");
        }
        return false;
    }
    if (f.type != MsgType::Welcome || !parse_frame(f, welcome)) {
        ::close(fd);
        set_err(err, "expected Welcome, got " + std::string(msg_type_name(f.type)));
        return false;
    }

    std::unique_ptr<SocketPeer> peer(new SocketPeer(fd, /*remote_rank=*/0, hello.rank, t));
    peer->start();
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->head = std::move(peer);
    return true;
}

bool ClusterWorkerControl::send(const Frame & frame, std::string * err) {
    SocketPeer * p = nullptr;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        p = impl_->head.get();
    }
    if (!p) {
        set_err(err, "worker control channel not connected");
        return false;
    }
    return p->send(frame, err);
}

bool ClusterWorkerControl::recv(Frame & frame, uint32_t deadline_ms, std::string * err) {
    SocketPeer * p = nullptr;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        p = impl_->head.get();
    }
    if (!p) {
        set_err(err, "worker control channel not connected");
        return false;
    }
    return p->recv(frame, deadline_ms, err);
}

bool ClusterWorkerControl::alive() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->head && impl_->head->alive();
}

void ClusterWorkerControl::set_timeouts(const ControlTimeouts & t) {
    impl_->timeouts = t;
}

void ClusterWorkerControl::close() {
    if (impl_) impl_->close_head();
}

}  // namespace luce::cluster

#else  // _WIN32 ─────────────────────────────────────────────────────────────

namespace luce::cluster {

namespace {
const char * kPosixOnly = "cluster control channel is POSIX-only";
void set_posix_only(std::string * err) {
    if (err) *err = kPosixOnly;
}
}  // namespace

bool write_exact_deadline(int, const void *, size_t, uint32_t, std::string * err) {
    set_posix_only(err);
    return false;
}

bool read_exact_deadline(int, void *, size_t, uint32_t, std::string * err) {
    set_posix_only(err);
    return false;
}

bool write_frame(int, const Frame &, uint32_t, std::string * err) {
    set_posix_only(err);
    return false;
}

bool read_frame(int, Frame &, uint32_t, std::string * err) {
    set_posix_only(err);
    return false;
}

uint64_t monotonic_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string local_hostname(int rank) {
    return "rank" + std::to_string(rank);
}

struct ClusterHeadControl::Impl {
    ControlTimeouts timeouts;
};

ClusterHeadControl::ClusterHeadControl() : impl_(new Impl) {}
ClusterHeadControl::~ClusterHeadControl() = default;

bool ClusterHeadControl::listen(const std::string &, int, std::string * err) {
    set_posix_only(err);
    return false;
}

bool ClusterHeadControl::accept_workers(int, const HelloMsg &, uint32_t, std::vector<HelloMsg> & hellos,
                                        std::string * err) {
    hellos.clear();
    set_posix_only(err);
    return false;
}

bool ClusterHeadControl::send_to(int, const Frame &, std::string * err) {
    set_posix_only(err);
    return false;
}

bool ClusterHeadControl::broadcast(const Frame &, std::string * err) {
    set_posix_only(err);
    return false;
}

bool ClusterHeadControl::recv_from(int, Frame &, uint32_t, std::string * err) {
    set_posix_only(err);
    return false;
}

bool ClusterHeadControl::gather(MsgType, uint32_t, std::vector<Frame> & out, std::string * err) {
    out.clear();
    set_posix_only(err);
    return false;
}

int ClusterHeadControl::n_workers() const { return 0; }
bool ClusterHeadControl::all_alive() const { return false; }
std::vector<int> ClusterHeadControl::dead_ranks() const { return {}; }
void ClusterHeadControl::set_timeouts(const ControlTimeouts & t) { impl_->timeouts = t; }
void ClusterHeadControl::close() {}

struct ClusterWorkerControl::Impl {
    ControlTimeouts timeouts;
};

ClusterWorkerControl::ClusterWorkerControl() : impl_(new Impl) {}
ClusterWorkerControl::~ClusterWorkerControl() = default;

bool ClusterWorkerControl::connect_and_handshake(const std::string &, int, const HelloMsg &,
                                                 WelcomeMsg &, std::string * err) {
    set_posix_only(err);
    return false;
}

bool ClusterWorkerControl::send(const Frame &, std::string * err) {
    set_posix_only(err);
    return false;
}

bool ClusterWorkerControl::recv(Frame &, uint32_t, std::string * err) {
    set_posix_only(err);
    return false;
}

bool ClusterWorkerControl::alive() const { return false; }
void ClusterWorkerControl::set_timeouts(const ControlTimeouts & t) { impl_->timeouts = t; }
void ClusterWorkerControl::close() {}

}  // namespace luce::cluster

#endif  // _WIN32
