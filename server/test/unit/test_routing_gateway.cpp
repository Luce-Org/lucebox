#include "CppUnitTestFramework.hpp"
#include "server/routing_gateway.h"

#ifndef _WIN32
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

namespace {
void require_gateway(bool condition) {
    if (!condition) throw std::runtime_error("routing gateway lifecycle assertion failed");
}
struct RoutingGatewayFixture {
    std::string dir;
    RoutingGatewayFixture() {
        char pattern[] = "/tmp/luce-gateway-XXXXXX";
        const char * path = mkdtemp(pattern);
        require_gateway(path != nullptr);
        dir = path;
    }
    ~RoutingGatewayFixture() { std::filesystem::remove_all(dir); }
    std::string script(const std::string & body) {
        const std::string path = dir + "/gateway";
        std::ofstream(path) << "#!/bin/sh\n" << body;
        require_gateway(chmod(path.c_str(), 0700) == 0);
        return path;
    }
};
template<class Predicate> void wait_for(Predicate ready) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!ready() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    require_gateway(ready());
}
}

TEST_CASE(RoutingGatewayFixture, missing_executable_fails_without_child) {
    luce::server::RoutingGateway gateway;
    require_gateway(!gateway.start(dir + "/missing", "127.0.0.1", 8080, 8491, [] {}));
}

TEST_CASE(RoutingGatewayFixture, unexpected_exit_stops_owner) {
    std::atomic<bool> stopped{false};
    luce::server::RoutingGateway gateway;
    require_gateway(gateway.start(script("exit 17\n"), "127.0.0.1", 8080, 8491,
                                  [&] { stopped.store(true); }));
    wait_for([&] { return stopped.load(); });
    gateway.stop();
    require_gateway(gateway.failed());
}

TEST_CASE(RoutingGatewayFixture, forwards_listener_and_reaps_on_shutdown) {
    std::atomic<bool> stopped{false};
    const std::string path = script(
        "trap 'exit 0' TERM\n"
        "printf '%s\\n' \"$@\" > \"$0.args\"\n"
        "echo $$ > \"$0.pid\"\n"
        "while :; do sleep 1; done\n");
    luce::server::RoutingGateway gateway;
    require_gateway(gateway.start(path, "0.0.0.0", 8087, 8497, [&] { stopped.store(true); }));
    int pid = 0;
    wait_for([&] { std::ifstream(path + ".pid") >> pid; return pid > 0; });
    std::ifstream args(path + ".args");
    const std::string text((std::istreambuf_iterator<char>(args)), {});
    require_gateway(text == "--engine-url\nhttp://127.0.0.1:8497\n--host\n0.0.0.0\n--port\n8087\n");
    gateway.stop();
    require_gateway(!gateway.failed() && !stopped.load());
    require_gateway(kill(pid, 0) == -1 && errno == ESRCH);
    int status;
    require_gateway(waitpid(pid, &status, WNOHANG) == -1 && errno == ECHILD);
}
#endif
