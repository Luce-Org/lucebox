// backend_ipc.cpp - generic backend IPC process launcher.

#include "backend_ipc.h"
#include "io_utils.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <cerrno>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  if defined(__linux__)
#    include <sys/syscall.h>
#  endif
#  include <unistd.h>
#endif

namespace dflash::common {

const char * backend_ipc_mode_name(BackendIpcMode mode) {
    switch (mode) {
        case BackendIpcMode::DFlashDraft: return "dflash-draft";
        case BackendIpcMode::PFlashCompress: return "pflash-compress";
        case BackendIpcMode::Qwen35TargetShard: return "qwen35-target-shard";
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
    if (value == "qwen35-target-shard") {
        out = BackendIpcMode::Qwen35TargetShard;
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
    if (!init_work_dir(cfg.work_dir)) return false;

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
        argv_storage.reserve(cfg.args.size() + 6);
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
        argv_storage.emplace_back("--stream-fd=" + std::to_string(stream_pipe[1]));

        std::vector<char *> argv;
        argv.reserve(argv_storage.size() + 1);
        for (std::string & arg : argv_storage) argv.push_back(arg.data());
        argv.push_back(nullptr);
        ::execv(cfg.bin.c_str(), argv.data());
        std::fprintf(stderr, "backend-ipc exec failed: %s: %s\n",
                     cfg.bin.c_str(), std::strerror(errno));
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
    if (!read_exact_fd(stream_fd_, &status, sizeof(status)) || status != 0) {
        std::fprintf(stderr, "backend-ipc daemon did not become ready (status=%d)\n", status);
        close();
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
#if !defined(_WIN32)
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
    if (pid_ > 0) {
        int status = 0;
        ::waitpid(pid_, &status, 0);
        pid_ = -1;
    }
    if (owns_work_dir_ && !work_dir_.empty()) {
        ::rmdir(work_dir_.c_str());
    }
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
    if (!shared_payload_map_ || bytes > shared_payload_capacity_) return false;
    if (bytes > 0 && !data) return false;
    auto * header = static_cast<BackendIpcSharedPayloadHeader *>(shared_payload_map_);
    void * payload = static_cast<void *>(
        static_cast<char *>(shared_payload_map_) + backend_ipc_shared_payload_header_bytes());
    if (bytes > 0) {
        std::memcpy(payload, data, bytes);
    }
    seq = ++shared_payload_seq_;
    header->bytes = (uint64_t)bytes;
    header->sequence = seq;
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

bool BackendIpcProcess::init_work_dir(const std::string & requested) {
    if (!requested.empty()) {
        work_dir_ = requested;
        owns_work_dir_ = false;
        if (::mkdir(work_dir_.c_str(), 0700) != 0) {
            if (errno != EEXIST) {
                std::fprintf(stderr, "backend-ipc mkdir failed: %s: %s\n",
                             work_dir_.c_str(), std::strerror(errno));
                return false;
            }
            struct stat st;
            if (::stat(work_dir_.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
                std::fprintf(stderr, "backend-ipc work_dir is not a directory: %s\n",
                             work_dir_.c_str());
                return false;
            }
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
