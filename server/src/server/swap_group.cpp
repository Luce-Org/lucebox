#include "swap_group.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace luce::server {

using json = nlohmann::json;
namespace fs = std::filesystem;

std::vector<int> SwapMember::held() const {
    std::vector<int> out;
    for (int g : gpus) {
        if (std::find(released.begin(), released.end(), g) == released.end()) out.push_back(g);
    }
    return out;
}

std::vector<int> gpu_overlap(const std::vector<int> & a, const std::vector<int> & b) {
    std::vector<int> out;
    for (int g : a) {
        if (std::find(b.begin(), b.end(), g) != b.end() &&
            std::find(out.begin(), out.end(), g) == out.end()) out.push_back(g);
    }
    return out;
}

#if defined(_WIN32)

// Swap groups rely on flock(), kill(pid, 0) and POSIX sockets: on Windows a
// server refuses --swap-group at start-up.
std::unique_ptr<SwapGroup> SwapGroup::open(const std::string &, int, std::string & error) {
    error = "swap groups are not supported on Windows";
    return nullptr;
}
SwapGroup::~SwapGroup() = default;
std::string SwapGroup::member_path(int port) const { return dir_ + "/" + std::to_string(port) + ".json"; }
bool SwapGroup::publish(const SwapMember &) { return false; }
std::vector<SwapMember> SwapGroup::peers() const { return {}; }
bool SwapGroup::lock(const std::function<bool()> &, int, std::string & error) {
    error = "swap groups are not supported on Windows";
    return false;
}
void SwapGroup::unlock() {}
bool SwapGroup::ask_release(const SwapMember &, const std::vector<int> &, int, std::string & error) {
    error = "swap groups are not supported on Windows";
    return false;
}
int local_http_post(int, const std::string &, const std::string &, std::string & response, int) {
    response.clear();
    return 0;
}

#else

std::unique_ptr<SwapGroup> SwapGroup::open(const std::string & dir, int port, std::string & error) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir)) {
        error = "cannot create swap group directory " + dir + (ec ? ": " + ec.message() : "");
        return nullptr;
    }
    return std::unique_ptr<SwapGroup>(new SwapGroup(dir, port));
}

SwapGroup::~SwapGroup() {
    unlock();
    std::error_code ec;
    fs::remove(member_path(port_), ec);
}

std::string SwapGroup::member_path(int port) const {
    return (fs::path(dir_) / (std::to_string(port) + ".json")).string();
}

bool SwapGroup::publish(const SwapMember & self) {
    json j = {{"pid", self.pid}, {"port", self.port}, {"model", self.model}, {"gpus", self.gpus},
              {"released", self.released}, {"keepable", self.keepable}, {"bytes", self.bytes}};
    const std::string path = member_path(port_);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) return false;
        out << j.dump() << "\n";
        if (!out) return false;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    return !ec;
}

std::vector<SwapMember> SwapGroup::peers() const {
    std::vector<SwapMember> out;
    std::error_code ec;
    for (const auto & entry : fs::directory_iterator(dir_, ec)) {
        if (entry.path().extension() != ".json") continue;
        std::ifstream in(entry.path());
        json j = json::parse(in, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object()) continue;
        SwapMember m;
        try {
            m.pid = j.value("pid", 0);
            m.port = j.value("port", 0);
            m.model = j.value("model", std::string());
            m.gpus = j.value("gpus", std::vector<int>{});
            m.released = j.value("released", std::vector<int>{});
            m.keepable = j.value("keepable", std::vector<int>{});
            m.bytes = j.value("bytes", std::vector<size_t>{});
        } catch (const json::exception &) {
            continue;   // a field of the wrong type: not a member file
        }
        if (m.port == port_ || m.port <= 0) continue;
        if (m.pid <= 0 || (kill(m.pid, 0) != 0 && errno == ESRCH)) {
            std::error_code rm;
            fs::remove(entry.path(), rm);   // its process is gone
            continue;
        }
        out.push_back(std::move(m));
    }
    std::sort(out.begin(), out.end(), [](const SwapMember & a, const SwapMember & b) { return a.port < b.port; });
    return out;
}

bool SwapGroup::lock(const std::function<bool()> & on_wait, int timeout_s, std::string & error) {
    if (lock_fd_ >= 0) return true;
    const std::string path = (fs::path(dir_) / "swap.lock").string();
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        error = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno != EWOULDBLOCK && errno != EINTR) {
            error = std::string("swap group lock: ") + std::strerror(errno);
            ::close(fd);
            return false;
        }
        if (on_wait && !on_wait()) {
            error = "stopped waiting for the swap group lock";
            ::close(fd);
            return false;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            error = "timed out waiting for the swap group lock";
            ::close(fd);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    lock_fd_ = fd;
    return true;
}

void SwapGroup::unlock() {
    if (lock_fd_ < 0) return;
    flock(lock_fd_, LOCK_UN);
    ::close(lock_fd_);
    lock_fd_ = -1;
}

bool SwapGroup::ask_release(const SwapMember & peer, const std::vector<int> & gpus, int timeout_s,
                            std::string & error) {
    const json body = {{"state", "released"}, {"gpus", gpus}};
    std::string response;
    const int status = local_http_post(peer.port, "/v1/residency", body.dump(), response, timeout_s);
    if (status == 200) return true;
    std::ostringstream os;
    os << "peer " << peer.model << " on port " << peer.port << " did not release GPU";
    for (int g : gpus) os << " " << g;
    os << ": " << (status == 0 ? std::string("no answer") : "HTTP " + std::to_string(status) + " " + response);
    error = os.str();
    return false;
}

int local_http_post(int port, const std::string & path, const std::string & body,
                    std::string & response, int timeout_s) {
    response.clear();
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return 0;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return 0;
    }
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\nHost: 127.0.0.1:" << port
        << "\r\nContent-Type: application/json\r\nContent-Length: " << body.size()
        << "\r\nConnection: close\r\n\r\n" << body;
    const std::string bytes = req.str();
    for (size_t sent = 0; sent < bytes.size();) {
        const ssize_t n = ::send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            ::close(fd);
            return 0;
        }
        sent += (size_t) n;
    }
    std::string in;
    char buf[4096];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (true) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            ::close(fd);
            return 0;
        }
        pollfd p{fd, POLLIN, 0};
        const int ready = ::poll(&p, 1, (int) std::min<long long>(left, 1000));
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) break;
        if (ready == 0) continue;
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        in.append(buf, (size_t) n);
    }
    ::close(fd);
    int status = 0;
    if (in.compare(0, 5, "HTTP/") == 0) {
        const size_t sp = in.find(' ');
        if (sp != std::string::npos) status = std::atoi(in.c_str() + sp + 1);
    }
    const size_t head_end = in.find("\r\n\r\n");
    if (head_end != std::string::npos) response = in.substr(head_end + 4);
    return status;
}

#endif  // !defined(_WIN32)

}  // namespace luce::server
