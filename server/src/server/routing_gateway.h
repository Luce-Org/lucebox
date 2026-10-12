#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace luce::server {

// One optional local gateway. Only the monitor thread reaps or signals the child.
// Keep policy/configuration in the executable; no shell interpretation here.
class RoutingGateway {
public:
    ~RoutingGateway() { stop(); }
    RoutingGateway() = default;
    RoutingGateway(const RoutingGateway &) = delete;
    RoutingGateway & operator=(const RoutingGateway &) = delete;

    bool start(const std::string & executable, const std::string & host, int port,
               int engine_port, std::function<void()> on_exit) {
#ifdef _WIN32
        (void)executable; (void)host; (void)port; (void)engine_port; (void)on_exit;
        std::fprintf(stderr, "[server] --routing-gateway requires a POSIX host\n");
        return false;
#else
        std::vector<std::string> args = {executable, "--engine-url",
            "http://127.0.0.1:" + std::to_string(engine_port),
            "--host", host, "--port", std::to_string(port)};
        std::vector<char *> argv;
        for (auto & arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        posix_spawnattr_t attr;
        posix_spawnattr_init(&attr);
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attr, 0);
        posix_spawn_file_actions_t files;
        posix_spawn_file_actions_init(&files);
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 34)
        // Do not inherit model files, device handles, or server sockets.
        posix_spawn_file_actions_addclosefrom_np(&files, 3);
#endif
#endif
        pid_t pid;
        const int error = posix_spawn(&pid, executable.c_str(), &files, &attr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&files);
        posix_spawnattr_destroy(&attr);
        if (error) {
            std::fprintf(stderr, "[server] cannot start routing gateway %s: %s\n",
                         executable.c_str(), std::strerror(error));
            return false;
        }
        monitor_ = std::thread([this, pid, on_exit = std::move(on_exit)] {
            int status = 0;
            while (!stopping_.load()) {
                const pid_t result = waitpid(pid, &status, WNOHANG);
                if (result == pid || (result < 0 && errno != EINTR)) {
                    // A wrapper must exec its gateway, but also stop descendants
                    // if a broken wrapper leaves them behind when it exits.
                    kill(-pid, SIGTERM);
                    failed_.store(true);
                    std::fprintf(stderr, "[server] routing gateway exited unexpectedly (wait status %d)\n", status);
                    on_exit();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            kill(-pid, SIGTERM);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                const pid_t result = waitpid(pid, &status, WNOHANG);
                if (result == pid || (result < 0 && errno != EINTR)) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            kill(-pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        });
        return true;
#endif
    }

    void stop() {
        stopping_.store(true);
        if (monitor_.joinable()) monitor_.join();
    }
    bool failed() const { return failed_.load(); }

private:
    std::atomic<bool> stopping_{false};
    std::atomic<bool> failed_{false};
    std::thread monitor_;
};

} // namespace luce::server
