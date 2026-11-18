// Swap groups (server/swap_group.h): a member's file round-trips to its peers,
// the files of dead processes are dropped, the group's lock excludes a second
// holder until released, and a peer's release request is a plain local POST.

#include "CppUnitTestFramework.hpp"
#include "../../src/server/swap_group.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace luce::server;
namespace fs = std::filesystem;

namespace {
struct SwapGroupFixture {};

// A fresh directory under the system temporary directory, removed with the object.
struct TempDir {
    std::string path;
    TempDir() {
        path = (fs::temp_directory_path() / ("luce-swap-test-" + std::to_string(getpid()) + "-" +
                                             std::to_string(std::rand()))).string();
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

SwapMember member(int port, std::vector<int> gpus, std::vector<int> released, std::vector<int> keepable = {}) {
    SwapMember m;
    m.pid = (int) getpid();
    m.port = port;
    m.model = "m" + std::to_string(port);
    m.gpus = std::move(gpus);
    m.released = std::move(released);
    m.keepable = std::move(keepable);
    m.bytes.assign(m.gpus.size(), 1);
    return m;
}
}  // namespace

TEST_CASE(SwapGroupFixture, held_is_gpus_minus_released) {
    CHECK((member(1, {0, 1}, {}).held() == std::vector<int>{0, 1}));
    CHECK((member(1, {0, 1}, {0}).held() == std::vector<int>{1}));
    CHECK(member(1, {0, 1}, {0, 1}).held().empty());
    CHECK((gpu_overlap({0, 1}, {1, 2}) == std::vector<int>{1}));
    CHECK(gpu_overlap({0}, {1}).empty());
}

TEST_CASE(SwapGroupFixture, members_see_each_other) {
    TempDir dir;
    std::string err;
    auto a = SwapGroup::open(dir.path, 9001, err);
    auto b = SwapGroup::open(dir.path, 9002, err);
    CHECK(a && b);
    CHECK(a->publish(member(9001, {0}, {})));
    CHECK(b->publish(member(9002, {0, 1}, {0}, {1})));
    const auto from_a = a->peers();
    CHECK(from_a.size() == 1);
    CHECK(from_a[0].port == 9002);
    CHECK((from_a[0].held() == std::vector<int>{1}));
    CHECK((from_a[0].keepable == std::vector<int>{1}));
    const auto from_b = b->peers();
    CHECK(from_b.size() == 1);
    CHECK(from_b[0].port == 9001);
    b.reset();   // leaving the group removes its file
    CHECK(a->peers().empty());
}

TEST_CASE(SwapGroupFixture, dead_members_are_dropped) {
    TempDir dir;
    std::string err;
    auto a = SwapGroup::open(dir.path, 9001, err);
    CHECK(a != nullptr);
    {
        std::ofstream out(fs::path(dir.path) / "9003.json");
        // A pid far above any pid_max: no such process.
        out << R"({"pid": 2147483000, "port": 9003, "model": "gone", "gpus": [0], "released": []})" << "\n";
    }
    CHECK(a->peers().empty());
    CHECK(!fs::exists(fs::path(dir.path) / "9003.json"));
}

TEST_CASE(SwapGroupFixture, lock_excludes_a_second_holder) {
    TempDir dir;
    std::string err;
    auto a = SwapGroup::open(dir.path, 9001, err);
    auto b = SwapGroup::open(dir.path, 9002, err);
    CHECK(a->lock(nullptr, 5, err));
    int waits = 0;
    // b gives up after a few waits while a holds the lock.
    CHECK(!b->lock([&waits] { return ++waits < 3; }, 5, err));
    CHECK(waits == 3);
    a->unlock();
    CHECK(b->lock(nullptr, 5, err));
    b->unlock();
}

TEST_CASE(SwapGroupFixture, release_request_is_a_local_post) {
    // A one-shot HTTP server on a free loopback port.
    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    CHECK(::bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
    CHECK(::listen(listener, 1) == 0);
    socklen_t len = sizeof(addr);
    ::getsockname(listener, reinterpret_cast<sockaddr *>(&addr), &len);
    const int port = ntohs(addr.sin_port);
    std::string seen;
    std::thread server([&] {
        const int fd = ::accept(listener, nullptr, nullptr);
        char buf[4096];
        while (seen.find("}") == std::string::npos) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) break;
            seen.append(buf, (size_t) n);
        }
        const std::string reply = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 12\r\n"
                                  "Connection: close\r\n\r\n{\"ok\":true}\n";
        ::send(fd, reply.data(), reply.size(), 0);
        ::close(fd);
    });
    SwapMember peer = member(port, {0}, {});
    std::string err;
    CHECK(SwapGroup::ask_release(peer, {0}, 10, err));
    server.join();
    ::close(listener);
    CHECK(seen.rfind("POST /v1/residency HTTP/1.1", 0) == 0);
    CHECK(seen.find(R"("state":"released")") != std::string::npos);
    CHECK(seen.find(R"("gpus":[0])") != std::string::npos);
}

TEST_CASE(SwapGroupFixture, unreachable_peer_is_an_error) {
    // Port 1 on loopback: nothing listens there.
    std::string err;
    CHECK(!SwapGroup::ask_release(member(1, {0}, {}), {0}, 2, err));
    CHECK(err.find("no answer") != std::string::npos);
}
