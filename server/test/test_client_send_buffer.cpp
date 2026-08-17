// Host-side unit test for ClientSendBuffer (concurrent scheduler writer).
//
// Uses a socketpair with a shrunken send buffer to exercise partial drains,
// backpressure, the stall policy, and hard-error detection.

#include "server/client_send_buffer.h"
#include "host_check.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

using namespace dflash::common;
using clock_t_ = std::chrono::steady_clock;
using std::chrono::seconds;

static int g_checks = 0;

static void make_pair(int fds[2]) {
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    // Non-blocking writer (matches the HTTP server's client sockets) and a
    // small send buffer so a few KB reliably back-pressures.
    CHECK(fcntl(fds[0], F_SETFL, O_NONBLOCK) == 0);
    const int sndbuf = 4096;
    CHECK(setsockopt(fds[0], SOL_SOCKET, SO_SNDBUF,
                     &sndbuf, sizeof(sndbuf)) == 0);
}

static std::string drain_peer(int fd) {
    std::string got;
    char tmp[4096];
    for (;;) {
        const ssize_t n = recv(fd, tmp, sizeof(tmp), MSG_DONTWAIT);
        if (n <= 0) break;
        got.append(tmp, (size_t)n);
    }
    return got;
}

int main() {
    // Happy path: everything fits the socket buffer in one flush.
    {
        int fds[2];
        make_pair(fds);
        ClientSendBuffer buffer;
        CHECK(buffer.empty());
        buffer.append("hello ");
        buffer.append("world");
        CHECK(buffer.pending() == 11);
        CHECK(buffer.flush(fds[0]));
        CHECK(buffer.empty());
        CHECK(drain_peer(fds[1]) == "hello world");
        close(fds[0]);
        close(fds[1]);
    }

    // Backpressure: a payload larger than the socket buffer drains across
    // flushes as the peer reads, and never blocks.
    {
        int fds[2];
        make_pair(fds);
        ClientSendBuffer buffer;
        std::string payload(256 * 1024, '\0');
        for (size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<char>((i * 131 + 17) & 0xff);
        }
        buffer.append(payload);
        CHECK(buffer.flush(fds[0]));
        CHECK(!buffer.empty());          // socket buffer full, remainder pending
        std::string got;
        for (int i = 0; i < 1000 && !buffer.empty(); i++) {
            got += drain_peer(fds[1]);
            CHECK(buffer.flush(fds[0]));
        }
        got += drain_peer(fds[1]);
        CHECK(buffer.empty());
        CHECK(got == payload);
        close(fds[0]);
        close(fds[1]);
    }

    // A single flush that sends past the compaction threshold but leaves a
    // tail must still count as progress. Compaction normalizes off_ back to
    // zero, so progress cannot be inferred from the post-compaction offset.
    {
        int fds[2];
        CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
        CHECK(fcntl(fds[0], F_SETFL, O_NONBLOCK) == 0);
        const int sndbuf = 256 * 1024;
        CHECK(setsockopt(fds[0], SOL_SOCKET, SO_SNDBUF,
                         &sndbuf, sizeof(sndbuf)) == 0);

        ClientSendBuffer buffer;
        const std::string payload(2 * 1024 * 1024, 'p');
        buffer.append(payload);
        buffer.mark_progress(clock_t_::now() - seconds(3600));
        CHECK(buffer.flush(fds[0]));
        CHECK(!buffer.empty());
        CHECK(payload.size() - buffer.pending() > (64u << 10));
        CHECK(!buffer.should_drop(clock_t_::now(), seconds(30),
                                  payload.size()));
        close(fds[0]);
        close(fds[1]);
    }

    // Stall policy: pending bytes with no progress trip the deadline and the
    // cap; an empty send buffer never drops; progress resets the clock.
    {
        int fds[2];
        make_pair(fds);
        ClientSendBuffer buffer;
        const auto t0 = clock_t_::now();
        buffer.mark_progress(t0);
        CHECK(!buffer.should_drop(t0 + seconds(3600), seconds(30), 1u << 20));

        buffer.append(std::string(64 * 1024, 'y'));
        CHECK(buffer.flush(fds[0]));     // fills the 4 KB socket buffer, stalls
        CHECK(!buffer.empty());
        CHECK(!buffer.should_drop(clock_t_::now(), seconds(30), 1u << 20));
        CHECK(buffer.should_drop(clock_t_::now() + seconds(31), seconds(30),
                                 1u << 20));
        CHECK(buffer.should_drop(clock_t_::now(), seconds(30), /*cap=*/1024));

        // Reader progress resets the deadline.
        drain_peer(fds[1]);
        CHECK(buffer.flush(fds[0]));
        CHECK(!buffer.should_drop(clock_t_::now() + seconds(29), seconds(30),
                                  1u << 20));
        close(fds[0]);
        close(fds[1]);
    }

    // Hard error: peer closed -> flush returns false.
    {
        int fds[2];
        make_pair(fds);
        close(fds[1]);
        ClientSendBuffer buffer;
        buffer.append("doomed");
        CHECK(!buffer.flush(fds[0]));
        close(fds[0]);
    }

    std::printf("OK test_client_send_buffer (%d checks)\n", g_checks);
    return 0;
}
