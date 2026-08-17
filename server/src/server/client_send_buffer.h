// ClientSendBuffer — buffered non-blocking writer for one client socket.
//
// The concurrent scheduler must never block on a client: a stalled SSE
// reader would head-of-line-block every co-scheduled request's decode.
// Token chunks and final responses append here; the scheduler drains the
// buffer with non-blocking sends once per decode iteration and consults
// the stall policy (no drain progress for a deadline, or a byte cap) to
// decide when a dead reader should be dropped.
//
// The fd is owned by the caller and must already be non-blocking (the HTTP
// server sets client sockets non-blocking at enqueue time).

#pragma once

#include "socket_handle.h"

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#else
#include <cerrno>
#include <sys/socket.h>
#include <sys/types.h>
#endif

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <string_view>

namespace dflash::common {

class ClientSendBuffer {
public:
    void append(std::string_view bytes) { buf_.append(bytes); }

    bool empty() const { return off_ == buf_.size(); }
    size_t pending() const { return buf_.size() - off_; }

    // Drain as much as the socket accepts right now. Never blocks. Returns
    // false on a hard socket error (peer gone). Records drain progress for
    // the stall policy and compacts the drained prefix.
    bool flush(SocketHandle fd) {
        const size_t off_before = off_;
        while (off_ < buf_.size()) {
            const auto n = send_some(fd, buf_.data() + off_,
                                     buf_.size() - off_);
            if (n > 0) { off_ += (size_t)n; continue; }
            if (n < 0) {
                const int error = send_error();
                if (send_would_block(error)) break;
                if (send_was_interrupted(error)) continue;
            }
            return false;
        }
        const bool made_progress = off_ != off_before;
        if (off_ == buf_.size()) {
            buf_.clear();
            off_ = 0;
        } else if (off_ > kCompactAt) {
            buf_.erase(0, off_);
            off_ = 0;
        }
        if (made_progress) {
            last_progress_ = std::chrono::steady_clock::now();
        }
        return true;
    }

    // True when the reader should be dropped: bytes are pending and either
    // the buffer exceeds `cap` or nothing drained since `stall` ago. The
    // deadline does NOT reset while the reader makes no progress — a
    // trickling reader is bounded by the cap instead.
    bool should_drop(std::chrono::steady_clock::time_point now,
                     std::chrono::seconds stall, size_t cap) const {
        if (empty()) return false;
        return pending() > cap || now - last_progress_ > stall;
    }

    // Start (or restart) the stall clock, e.g. at admission.
    void mark_progress(std::chrono::steady_clock::time_point now) {
        last_progress_ = now;
    }

private:
    static constexpr size_t kCompactAt = 64u << 10;

#if defined(_WIN32)
    static int send_some(SocketHandle fd, const char * data, size_t len) {
        const int chunk = static_cast<int>((std::min)(
            len, static_cast<size_t>((std::numeric_limits<int>::max)())));
        return ::send(fd, data, chunk, 0);
    }

    static int send_error() {
        return WSAGetLastError();
    }

    static bool send_would_block(int error) {
        return error == WSAEWOULDBLOCK;
    }

    static bool send_was_interrupted(int error) {
        return error == WSAEINTR;
    }
#else
    static ssize_t send_some(SocketHandle fd, const char * data, size_t len) {
#if defined(MSG_NOSIGNAL)
        return ::send(fd, data, len, MSG_NOSIGNAL);
#else
        return ::send(fd, data, len, 0);
#endif
    }

    static int send_error() {
        return errno;
    }

    static bool send_would_block(int error) {
        return error == EAGAIN || error == EWOULDBLOCK;
    }

    static bool send_was_interrupted(int error) {
        return error == EINTR;
    }
#endif

    std::string buf_;
    size_t off_ = 0;
    std::chrono::steady_clock::time_point last_progress_{};
};

}  // namespace dflash::common
