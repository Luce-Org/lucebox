// backend_ipc.h - generic backend IPC process launcher.
//
// Owns the out-of-process backend daemon lifecycle: fork/exec, command pipe,
// parent->daemon stream payload, optional shared payload buffer, binary status
// stream, and scratch work directory. Individual IPC modes keep their own
// payload protocol on top of this process wrapper.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <sys/types.h>
#endif

namespace dflash::common {

enum class BackendIpcMode {
    Invalid,
    DFlashDraft,
    PFlashCompress,
    Qwen35TargetShard,
    Gemma4TargetShard,
    LagunaTargetShard,
    MoeExpertCompute,
    DeepSeek4TargetShard,
    DeepSeek4DSparkDraft,
};

const char * backend_ipc_mode_name(BackendIpcMode mode);
bool parse_backend_ipc_mode(const std::string & value, BackendIpcMode & out);

enum class BackendIpcPayloadTransport {
    Stream,
    Shared,
    Auto,
};

const char * backend_ipc_payload_transport_name(BackendIpcPayloadTransport transport);
bool parse_backend_ipc_payload_transport(const std::string & value,
                                         BackendIpcPayloadTransport & out);

inline bool backend_ipc_checked_add_size(size_t a, size_t b, size_t & out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

inline bool backend_ipc_payload_in_bounds(size_t offset,
                                          size_t bytes,
                                          size_t capacity) {
    size_t end = 0;
    return backend_ipc_checked_add_size(offset, bytes, end) && end <= capacity;
}

struct BackendIpcSharedPayloadHeader {
    uint64_t sequence = 0;
    uint64_t bytes = 0;
};

inline size_t backend_ipc_shared_payload_header_bytes() {
    return sizeof(BackendIpcSharedPayloadHeader);
}

inline bool backend_ipc_shared_payload_map_bytes(size_t payload_bytes,
                                                 size_t & map_bytes) {
    return backend_ipc_checked_add_size(
        backend_ipc_shared_payload_header_bytes(), payload_bytes, map_bytes);
}

struct BackendIpcLaunchConfig {
    std::string bin;
    BackendIpcMode mode = BackendIpcMode::DFlashDraft;
    std::string payload_path;
    std::vector<std::string> args;
    std::string work_dir;
    BackendIpcPayloadTransport payload_transport = BackendIpcPayloadTransport::Auto;
    size_t shared_payload_bytes = 0;
};

struct BackendIpcPayloadSegment {
    const void * data = nullptr;
    size_t bytes = 0;
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
    int payload_fd() const { return payload_fd_; }
    int stream_fd() const { return stream_fd_; }
    BackendIpcPayloadTransport resolved_payload_transport() const {
        return resolved_payload_transport_;
    }
    bool has_shared_payload() const { return shared_payload_map_ != nullptr; }
    size_t shared_payload_capacity() const { return shared_payload_capacity_; }
    const std::string & work_dir() const { return work_dir_; }

    std::string next_path(const char * prefix);
    bool write_shared_payload(const void * data, size_t bytes, uint64_t & seq);
    bool write_shared_payload_segments(const BackendIpcPayloadSegment * segments,
                                       size_t n_segments,
                                       uint64_t & seq);
    bool read_shared_payload(void * data, size_t bytes, uint64_t seq) const;

private:
#if !defined(_WIN32)
    bool init_work_dir(const std::string & requested);
    bool init_shared_payload(size_t bytes);

    pid_t pid_ = -1;
#endif
    FILE * cmd_ = nullptr;
    int payload_fd_ = -1;
    int stream_fd_ = -1;
    int shared_payload_fd_ = -1;
    void * shared_payload_map_ = nullptr;
    size_t shared_payload_bytes_ = 0;
    size_t shared_payload_capacity_ = 0;
    uint64_t shared_payload_seq_ = 0;
    BackendIpcPayloadTransport resolved_payload_transport_ =
        BackendIpcPayloadTransport::Stream;
    std::string work_dir_;
    int seq_ = 0;
    bool owns_work_dir_ = false;
    bool active_ = false;
};

}  // namespace dflash::common
