// backend_ipc.h - generic backend IPC process launcher.
//
// Owns the out-of-process backend daemon lifecycle: fork/exec, command pipe,
// binary status stream, and scratch work directory. Individual IPC modes keep
// their own payload protocol on top of this process wrapper.

#pragma once

#include <cstdio>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <sys/types.h>
#endif

namespace dflash::common {

enum class BackendIpcMode {
    DFlashDraft,
    PFlashCompress,
};

const char * backend_ipc_mode_name(BackendIpcMode mode);
bool parse_backend_ipc_mode(const std::string & value, BackendIpcMode & out);

struct BackendIpcLaunchConfig {
    std::string bin;
    BackendIpcMode mode = BackendIpcMode::DFlashDraft;
    std::string payload_path;
    std::vector<std::string> args;
    std::string work_dir;
};

class BackendIpcProcess {
public:
    BackendIpcProcess() = default;
    BackendIpcProcess(const BackendIpcProcess &) = delete;
    BackendIpcProcess & operator=(const BackendIpcProcess &) = delete;
    ~BackendIpcProcess() { close(); }

    bool start(const BackendIpcLaunchConfig & cfg);
    void close();

    bool active() const { return active_; }
    FILE * command_stream() const { return cmd_; }
    int stream_fd() const { return stream_fd_; }
    const std::string & work_dir() const { return work_dir_; }

    std::string next_path(const char * prefix);

private:
#if !defined(_WIN32)
    bool init_work_dir(const std::string & requested);

    pid_t pid_ = -1;
#endif
    FILE * cmd_ = nullptr;
    int stream_fd_ = -1;
    std::string work_dir_;
    int seq_ = 0;
    bool owns_work_dir_ = false;
    bool active_ = false;
};

}  // namespace dflash::common
