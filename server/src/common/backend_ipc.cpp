// backend_ipc.cpp - generic backend IPC process launcher.

#include "backend_ipc.h"
#include "io_utils.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <array>
#include <algorithm>
#include <climits>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#  include <cerrno>
#  include <fcntl.h>
#  include <poll.h>
#  include <signal.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  if defined(__linux__)
#    include <sys/syscall.h>
#  endif
#  include <unistd.h>
#endif

namespace dflash::common {

namespace {

#if !defined(_WIN32)
bool read_exact_with_timeout(int fd, void * data, size_t bytes,
                             int timeout_ms, bool & timed_out) {
    timed_out = false;
    auto * cursor = static_cast<char *>(data);
    size_t received = 0;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    while (received < bytes) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            timed_out = true;
            return false;
        }
        const int64_t remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now).count();
        pollfd descriptor{fd, POLLIN | POLLHUP, 0};
        const int wait_ms = static_cast<int>((std::min)(
            int64_t{INT_MAX}, (std::max)(int64_t{1}, remaining)));
        const int polled = ::poll(&descriptor, 1, wait_ms);
        if (polled == 0) {
            timed_out = true;
            return false;
        }
        if (polled < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (descriptor.revents & (POLLERR | POLLNVAL)) return false;
        const ssize_t count = ::read(fd, cursor + received, bytes - received);
        if (count == 0) return false;
        if (count < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        received += static_cast<size_t>(count);
    }
    return true;
}

bool close_descriptor_range(unsigned int first, unsigned int last) {
    if (first > last) return true;
#if defined(__linux__) && defined(SYS_close_range)
    int rc = -1;
    do {
        rc = static_cast<int>(::syscall(SYS_close_range, first, last, 0));
    } while (rc != 0 && errno == EINTR);
    return rc == 0;
#else
    (void)first;
    (void)last;
    errno = ENOSYS;
    return false;
#endif
}

bool isolate_child_descriptors(int payload_fd, int stream_fd, int shared_fd) {
    std::array<int, 3> keep{payload_fd, stream_fd, shared_fd};
    std::sort(keep.begin(), keep.end());
    unsigned int first = STDERR_FILENO + 1;
    int previous = -1;
    for (const int fd : keep) {
        if (fd < static_cast<int>(first) || fd == previous) continue;
        if (!close_descriptor_range(first, static_cast<unsigned int>(fd - 1))) {
            return false;
        }
        first = static_cast<unsigned int>(fd) + 1U;
        previous = fd;
    }
    return close_descriptor_range(
        first, (std::numeric_limits<unsigned int>::max)());
}
#endif

}  // namespace

const char * backend_ipc_mode_name(BackendIpcMode mode) {
    switch (mode) {
        case BackendIpcMode::Invalid: return "invalid";
        case BackendIpcMode::DFlashDraft: return "dflash-draft";
        case BackendIpcMode::PFlashCompress: return "pflash-compress";
        case BackendIpcMode::Qwen3ToolPredict: return "qwen3-tool-predict";
        case BackendIpcMode::Qwen35TargetShard: return "qwen35-target-shard";
        case BackendIpcMode::Gemma4TargetShard: return "gemma4-target-shard";
        case BackendIpcMode::LagunaTargetShard: return "laguna-target-shard";
        case BackendIpcMode::MoeExpertCompute: return "moe-expert-compute";
        case BackendIpcMode::DeepSeek4TargetShard: return "deepseek4-target-shard";
    }
    return "unknown";
}

bool parse_backend_ipc_mode(const std::string & value, BackendIpcMode & out) {
    if (value == "dflash-draft") {
        out = BackendIpcMode::DFlashDraft;
        return true;
    }
    if (value == "pflash-compress") {
        out = BackendIpcMode::PFlashCompress;
        return true;
    }
    if (value == "qwen3-tool-predict") {
        out = BackendIpcMode::Qwen3ToolPredict;
        return true;
    }
    if (value == "qwen35-target-shard") {
        out = BackendIpcMode::Qwen35TargetShard;
        return true;
    }
    if (value == "gemma4-target-shard") {
        out = BackendIpcMode::Gemma4TargetShard;
        return true;
    }
    if (value == "laguna-target-shard") {
        out = BackendIpcMode::LagunaTargetShard;
        return true;
    }
    if (value == "moe-expert-compute") {
        out = BackendIpcMode::MoeExpertCompute;
        return true;
    }
    if (value == "deepseek4-target-shard") {
        out = BackendIpcMode::DeepSeek4TargetShard;
        return true;
    }
    return false;
}

const char * backend_ipc_payload_transport_name(BackendIpcPayloadTransport transport) {
    switch (transport) {
        case BackendIpcPayloadTransport::Stream: return "stream";
        case BackendIpcPayloadTransport::Shared: return "shared";
        case BackendIpcPayloadTransport::Auto: return "auto";
    }
    return "unknown";
}

bool parse_backend_ipc_payload_transport(const std::string & value,
                                         BackendIpcPayloadTransport & out) {
    if (value == "stream") {
        out = BackendIpcPayloadTransport::Stream;
        return true;
    }
    if (value == "shared") {
        out = BackendIpcPayloadTransport::Shared;
        return true;
    }
    if (value == "auto") {
        out = BackendIpcPayloadTransport::Auto;
        return true;
    }
    return false;
}

bool BackendIpcProcess::start(const BackendIpcLaunchConfig & cfg) {
#if defined(_WIN32)
    (void)cfg;
    std::fprintf(stderr, "Backend IPC is only implemented on POSIX hosts\n");
    return false;
#else
    close();
    if (cfg.bin.empty() || cfg.payload_path.empty()) return false;
    if (!init_work_dir(cfg.work_dir, cfg.require_private_work_dir)) return false;

    int cmd_pipe[2] = {-1, -1};
    int payload_pipe[2] = {-1, -1};
    int stream_pipe[2] = {-1, -1};
    if (::pipe(cmd_pipe) != 0 || ::pipe(payload_pipe) != 0 || ::pipe(stream_pipe) != 0) {
        std::fprintf(stderr, "backend-ipc pipe failed: %s\n", std::strerror(errno));
        if (cmd_pipe[0] >= 0) ::close(cmd_pipe[0]);
        if (cmd_pipe[1] >= 0) ::close(cmd_pipe[1]);
        if (payload_pipe[0] >= 0) ::close(payload_pipe[0]);
        if (payload_pipe[1] >= 0) ::close(payload_pipe[1]);
        if (stream_pipe[0] >= 0) ::close(stream_pipe[0]);
        if (stream_pipe[1] >= 0) ::close(stream_pipe[1]);
        return false;
    }
    const bool shared_required =
        cfg.payload_transport == BackendIpcPayloadTransport::Shared;
    const bool shared_requested =
        shared_required || cfg.payload_transport == BackendIpcPayloadTransport::Auto;
    if (cfg.payload_transport == BackendIpcPayloadTransport::Shared &&
        cfg.shared_payload_bytes == 0) {
        std::fprintf(stderr, "backend-ipc shared payload requested with zero capacity\n");
        ::close(cmd_pipe[0]); ::close(cmd_pipe[1]);
        ::close(payload_pipe[0]); ::close(payload_pipe[1]);
        ::close(stream_pipe[0]); ::close(stream_pipe[1]);
        return false;
    }
    if (shared_requested && cfg.shared_payload_bytes > 0) {
        if (!init_shared_payload(cfg.shared_payload_bytes)) {
            if (shared_required) {
                close();
                ::close(cmd_pipe[0]); ::close(cmd_pipe[1]);
                ::close(payload_pipe[0]); ::close(payload_pipe[1]);
                ::close(stream_pipe[0]); ::close(stream_pipe[1]);
                return false;
            }
            std::fprintf(stderr,
                         "backend-ipc auto shared payload unavailable; using stream\n");
        }
    }
    resolved_payload_transport_ = has_shared_payload()
        ? BackendIpcPayloadTransport::Shared
        : BackendIpcPayloadTransport::Stream;

    pid_ = ::fork();
    if (pid_ < 0) {
        std::fprintf(stderr, "backend-ipc fork failed: %s\n", std::strerror(errno));
        ::close(cmd_pipe[0]); ::close(cmd_pipe[1]);
        ::close(payload_pipe[0]); ::close(payload_pipe[1]);
        ::close(stream_pipe[0]); ::close(stream_pipe[1]);
        close();
        pid_ = -1;
        return false;
    }
    if (pid_ == 0) {
        if (cmd_pipe[0] != STDIN_FILENO && ::dup2(cmd_pipe[0], STDIN_FILENO) < 0) {
            std::fprintf(stderr, "backend-ipc dup2 failed: %s\n", std::strerror(errno));
            _exit(127);
        }
        if (cmd_pipe[0] != STDIN_FILENO) ::close(cmd_pipe[0]);
        ::close(cmd_pipe[1]);
        ::close(payload_pipe[1]);
        ::close(stream_pipe[0]);

        std::vector<std::string> argv_storage;
        argv_storage.reserve(cfg.args.size() + 7);
        const std::string & exec_bin = cfg.bin;
        argv_storage.emplace_back(cfg.bin);
        argv_storage.emplace_back(
            std::string("--backend-ipc-mode=") + backend_ipc_mode_name(cfg.mode));
        argv_storage.emplace_back(cfg.payload_path);
        for (const std::string & arg : cfg.args) argv_storage.emplace_back(arg);
        argv_storage.emplace_back("--payload-fd=" + std::to_string(payload_pipe[0]));
        if (has_shared_payload()) {
            argv_storage.emplace_back("--shared-payload-fd=" +
                                      std::to_string(shared_payload_fd_));
            argv_storage.emplace_back("--shared-payload-bytes=" +
                                      std::to_string(shared_payload_capacity_));
        }
        argv_storage.emplace_back("--stream-fd=" +
                                  std::to_string(stream_pipe[1]));

        std::vector<char *> argv;
        argv.reserve(argv_storage.size() + 1);
        for (std::string & arg : argv_storage) argv.push_back(arg.data());
        argv.push_back(nullptr);
        if (cfg.isolate_inherited_fds &&
            !isolate_child_descriptors(
                payload_pipe[0], stream_pipe[1], shared_payload_fd_)) {
            std::fprintf(stderr,
                         "backend-ipc descriptor isolation failed: %s\n",
                         std::strerror(errno));
            _exit(127);
        }
        ::execv(exec_bin.c_str(), argv.data());
        std::fprintf(stderr, "backend-ipc exec failed: %s: %s\n",
                     exec_bin.c_str(), std::strerror(errno));
        _exit(127);
    }

    ::close(cmd_pipe[0]);
    ::close(payload_pipe[0]);
    ::close(stream_pipe[1]);
    payload_fd_ = payload_pipe[1];
    stream_fd_ = stream_pipe[0];
    cmd_ = ::fdopen(cmd_pipe[1], "w");
    if (!cmd_) {
        std::fprintf(stderr, "backend-ipc fdopen failed: %s\n", std::strerror(errno));
        ::close(cmd_pipe[1]);
        close();
        return false;
    }
    int32_t status = -1;
    bool readiness_timed_out = false;
    const bool status_read = cfg.readiness_timeout_ms > 0
        ? read_exact_with_timeout(stream_fd_, &status, sizeof(status),
                                  cfg.readiness_timeout_ms,
                                  readiness_timed_out)
        : read_exact_fd(stream_fd_, &status, sizeof(status));
    if (!status_read || status != 0) {
        int child_status = 0;
        const pid_t exited = ::waitpid(pid_, &child_status, WNOHANG);
        if (exited == pid_) {
            if (WIFEXITED(child_status)) {
                std::fprintf(stderr,
                             "backend-ipc daemon did not become ready (status=%d, exit=%d)\n",
                             status, WEXITSTATUS(child_status));
            } else if (WIFSIGNALED(child_status)) {
                std::fprintf(stderr,
                             "backend-ipc daemon did not become ready (status=%d, signal=%d)\n",
                             status, WTERMSIG(child_status));
            } else {
                std::fprintf(stderr,
                             "backend-ipc daemon did not become ready (status=%d, child-status=%d)\n",
                             status, child_status);
            }
            pid_ = -1;
        } else {
            std::fprintf(stderr,
                         readiness_timed_out
                             ? "backend-ipc daemon readiness timed out after %d ms\n"
                             : "backend-ipc daemon did not become ready (status=%d)\n",
                         readiness_timed_out ? cfg.readiness_timeout_ms : status);
        }
        if (pid_ > 0 && cfg.readiness_timeout_ms > 0) {
            terminate();
        } else {
            close();
        }
        return false;
    }
    active_ = true;
    std::printf("[backend-ipc] ready mode=%s payload_transport=%s bin=%s work_dir=%s\n",
                backend_ipc_mode_name(cfg.mode),
                backend_ipc_payload_transport_name(resolved_payload_transport_),
                cfg.bin.c_str(), work_dir_.c_str());
    return true;
#endif
}

void BackendIpcProcess::close() {
    close_impl(false);
}

void BackendIpcProcess::terminate() {
    close_impl(true);
}

void BackendIpcProcess::close_impl(bool force_terminate) {
#if !defined(_WIN32)
    const pid_t child = pid_;
    if (force_terminate && child > 0) {
        (void)::kill(child, SIGTERM);
    }
    if (cmd_) {
        std::fclose(cmd_);
        cmd_ = nullptr;
    }
    if (stream_fd_ >= 0) {
        ::close(stream_fd_);
        stream_fd_ = -1;
    }
    if (payload_fd_ >= 0) {
        ::close(payload_fd_);
        payload_fd_ = -1;
    }
    if (shared_payload_map_) {
        ::munmap(shared_payload_map_, shared_payload_bytes_);
        shared_payload_map_ = nullptr;
    }
    if (shared_payload_fd_ >= 0) {
        ::close(shared_payload_fd_);
        shared_payload_fd_ = -1;
    }
    if (child > 0) {
        int status = 0;
        if (force_terminate) {
            bool reaped = false;
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(100);
            while (std::chrono::steady_clock::now() < deadline) {
                const pid_t waited = ::waitpid(child, &status, WNOHANG);
                if (waited == child ||
                    (waited < 0 && errno == ECHILD)) {
                    reaped = true;
                    break;
                }
                if (waited < 0 && errno != EINTR) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            if (!reaped) {
                (void)::kill(child, SIGKILL);
                while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            }
        } else {
            while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
        }
        pid_ = -1;
    }
    if (owns_work_dir_ && !work_dir_.empty()) {
        ::rmdir(work_dir_.c_str());
    }
#else
    (void)force_terminate;
#endif
    active_ = false;
    owns_work_dir_ = false;
    shared_payload_bytes_ = 0;
    shared_payload_capacity_ = 0;
    shared_payload_seq_ = 0;
    resolved_payload_transport_ = BackendIpcPayloadTransport::Stream;
    work_dir_.clear();
    seq_ = 0;
}

std::string BackendIpcProcess::next_path(const char * prefix) {
    return work_dir_ + "/" + prefix + "_" + std::to_string(seq_++) + ".bin";
}

bool BackendIpcProcess::write_shared_payload(const void * data, size_t bytes, uint64_t & seq) {
    BackendIpcPayloadSegment segment{data, bytes};
    return write_shared_payload_segments(&segment, 1, seq);
}

bool BackendIpcProcess::write_shared_payload_segments(
        const BackendIpcPayloadSegment * segments,
        size_t n_segments,
        uint64_t & seq) {
    if (!shared_payload_map_ || (!segments && n_segments > 0)) return false;
    size_t bytes = 0;
    for (size_t i = 0; i < n_segments; ++i) {
        if (segments[i].bytes > 0 && !segments[i].data) return false;
        if (!backend_ipc_checked_add_size(bytes, segments[i].bytes, bytes)) {
            return false;
        }
    }
    if (bytes > shared_payload_capacity_) return false;
    auto * header = static_cast<BackendIpcSharedPayloadHeader *>(shared_payload_map_);
    auto * payload = static_cast<char *>(
        static_cast<char *>(shared_payload_map_) + backend_ipc_shared_payload_header_bytes());
    size_t off = 0;
    for (size_t i = 0; i < n_segments; ++i) {
        if (segments[i].bytes > 0) {
            std::memcpy(payload + off, segments[i].data, segments[i].bytes);
            off += segments[i].bytes;
        }
    }
    seq = ++shared_payload_seq_;
    // The command pipe publishes this request to another process. Publish the
    // mapped payload header explicitly so the daemon cannot observe the prior
    // response header after receiving the new command.
    backend_ipc_publish_shared_payload_header(
        header, seq, static_cast<uint64_t>(bytes));
    return true;
}

bool BackendIpcProcess::read_shared_payload(void * data, size_t bytes, uint64_t seq) const {
    if (!shared_payload_map_ || bytes > shared_payload_capacity_) return false;
    if (bytes > 0 && !data) return false;
    const auto * header =
        static_cast<const BackendIpcSharedPayloadHeader *>(shared_payload_map_);
    if (!backend_ipc_shared_payload_header_matches(
            header, seq, static_cast<uint64_t>(bytes))) {
        return false;
    }
    const void * payload = static_cast<const void *>(
        static_cast<const char *>(shared_payload_map_) +
        backend_ipc_shared_payload_header_bytes());
    if (bytes > 0) {
        std::memcpy(data, payload, bytes);
    }
    return true;
}

#if !defined(_WIN32)
bool BackendIpcProcess::init_shared_payload(size_t bytes) {
    if (bytes == 0) return false;
    size_t map_bytes = 0;
    if (!backend_ipc_shared_payload_map_bytes(bytes, map_bytes)) return false;
    int fd = -1;
#  if defined(__linux__) && defined(SYS_memfd_create)
    fd = (int)::syscall(SYS_memfd_create, "backend-ipc-payload", 0);
#  endif
    if (fd < 0) {
        const char * tmp = std::getenv("TMPDIR");
        std::string templ = std::string(tmp && *tmp ? tmp : "/tmp") +
                            "/backend-ipc-payload-XXXXXX";
        std::vector<char> templ_buf(templ.begin(), templ.end());
        templ_buf.push_back('\0');
        fd = ::mkstemp(templ_buf.data());
        if (fd >= 0) {
            ::unlink(templ_buf.data());
        }
    }
    if (fd < 0) {
        std::fprintf(stderr, "backend-ipc shared payload fd failed: %s\n",
                     std::strerror(errno));
        return false;
    }
    if (::ftruncate(fd, (off_t)map_bytes) != 0) {
        std::fprintf(stderr, "backend-ipc shared payload truncate failed: %s\n",
                     std::strerror(errno));
        ::close(fd);
        return false;
    }
    void * mapped = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        std::fprintf(stderr, "backend-ipc shared payload mmap failed: %s\n",
                     std::strerror(errno));
        ::close(fd);
        return false;
    }
    std::memset(mapped, 0, map_bytes);
    shared_payload_fd_ = fd;
    shared_payload_map_ = mapped;
    shared_payload_bytes_ = map_bytes;
    shared_payload_capacity_ = bytes;
    return true;
}

bool BackendIpcProcess::init_work_dir(const std::string & requested,
                                      bool require_private) {
    if (!requested.empty()) {
        work_dir_ = requested;
        owns_work_dir_ = false;
        if (::mkdir(work_dir_.c_str(), 0700) != 0) {
            if (errno != EEXIST) {
                std::fprintf(stderr, "backend-ipc mkdir failed: %s: %s\n",
                             work_dir_.c_str(), std::strerror(errno));
                return false;
            }
        }
        struct stat st;
        const int stat_result = require_private
            ? ::lstat(work_dir_.c_str(), &st)
            : ::stat(work_dir_.c_str(), &st);
        if (stat_result != 0 ||
            !S_ISDIR(st.st_mode)) {
            std::fprintf(stderr,
                "backend-ipc work_dir is not a directory: %s\n",
                work_dir_.c_str());
            return false;
        }
        if (require_private &&
            (st.st_uid != ::geteuid() || (st.st_mode & 0777) != 0700)) {
            std::fprintf(stderr,
                "backend-ipc work_dir must be owned by the server and mode 0700: %s\n",
                work_dir_.c_str());
            return false;
        }
        return true;
    }
    const char * tmp = std::getenv("TMPDIR");
    std::string templ = std::string(tmp && *tmp ? tmp : "/tmp") +
                        "/backend-ipc-XXXXXX";
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    char * dir = ::mkdtemp(buf.data());
    if (!dir) {
        std::fprintf(stderr, "backend-ipc mkdtemp failed: %s\n", std::strerror(errno));
        return false;
    }
    work_dir_ = dir;
    owns_work_dir_ = true;
    return true;
}
#endif

}  // namespace dflash::common
