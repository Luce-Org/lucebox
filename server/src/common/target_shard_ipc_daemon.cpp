// Generic target-shard IPC daemon loop for mixed-backend layer split.

#include "target_shard_ipc_daemon.h"

#include "backend_ipc.h"
#include "io_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <sys/mman.h>
#endif

namespace dflash::common {

namespace {

const char * daemon_prefix(const TargetShardDaemonCallbacks & callbacks) {
    return callbacks.log_prefix ? callbacks.log_prefix : "target-shard-daemon";
}

bool stream_daemon_status(const TargetShardDaemonCallbacks & callbacks,
                          int stream_fd,
                          int status) {
    const int32_t status_i32 = (int32_t)status;
    if (!write_exact_fd(stream_fd, &status_i32, sizeof(status_i32))) {
        std::fprintf(stderr, "[%s] failed to write status=%d\n",
                     daemon_prefix(callbacks), status);
        return false;
    }
    return true;
}

}  // namespace

int run_target_shard_ipc_daemon_loop(
        int hidden,
        int vocab,
        int stream_fd,
        int payload_fd,
        int shared_payload_fd,
        size_t shared_payload_bytes,
        TargetShardDaemonCallbacks callbacks) {
#if defined(_WIN32)
    (void)hidden;
    (void)vocab;
    (void)stream_fd;
    (void)payload_fd;
    (void)shared_payload_fd;
    (void)shared_payload_bytes;
    (void)callbacks;
    std::fprintf(stderr, "target shard IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    const char * prefix = daemon_prefix(callbacks);
    if (hidden <= 0 || vocab <= 0 || stream_fd < 0 || !callbacks.forward) {
        std::fprintf(stderr, "[%s] bad daemon configuration\n", prefix);
        if (stream_fd >= 0) stream_daemon_status(callbacks, stream_fd, -1);
        return 2;
    }

    void * shared_payload = nullptr;
    void * shared_payload_data = nullptr;
    size_t shared_payload_capacity = 0;
    size_t shared_payload_map_bytes = 0;
    if (shared_payload_fd >= 0 || shared_payload_bytes > 0) {
        if (shared_payload_fd < 0 || shared_payload_bytes == 0 ||
            !backend_ipc_shared_payload_map_bytes(shared_payload_bytes,
                                                  shared_payload_map_bytes)) {
            std::fprintf(stderr, "[%s] bad shared payload fd/size\n", prefix);
            stream_daemon_status(callbacks, stream_fd, -1);
            return 1;
        }
        shared_payload = ::mmap(nullptr, shared_payload_map_bytes,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                shared_payload_fd, 0);
        if (shared_payload == MAP_FAILED) {
            std::fprintf(stderr, "[%s] shared payload mmap failed\n", prefix);
            stream_daemon_status(callbacks, stream_fd, -1);
            return 1;
        }
        shared_payload_data =
            static_cast<char *>(shared_payload) + backend_ipc_shared_payload_header_bytes();
        shared_payload_capacity = shared_payload_bytes;
    }

    if (!stream_daemon_status(callbacks, stream_fd, 0)) {
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }

    std::vector<float> host_act;
    std::vector<int32_t> token_ids;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") {
            break;
        }

        int base_pos = -1;
        int n_tokens = 0;
        int want_argmax = 0;
        int want_logits = 0;
        int has_token_ids = 0;
        int forward_ubatch = 0;
        int token_count = 0;
        size_t bytes = 0;
        bool payload_ok = false;

        if (cmd == "forward_pipe") {
            iss >> base_pos >> n_tokens >> want_argmax >> want_logits >> bytes >>
                has_token_ids >> forward_ubatch >> token_count;
            const size_t expected_bytes =
                (size_t)std::max(0, n_tokens) * (size_t)hidden * sizeof(float);
            if (payload_fd >= 0 && n_tokens > 0 && bytes == expected_bytes) {
                host_act.assign(bytes / sizeof(float), 0.0f);
                payload_ok = read_exact_fd(payload_fd, host_act.data(), bytes);
            }
        } else if (cmd == "forward_shared") {
            uint64_t seq = 0;
            iss >> base_pos >> n_tokens >> want_argmax >> want_logits >> bytes >> seq >>
                has_token_ids >> forward_ubatch >> token_count;
            const size_t expected_bytes =
                (size_t)std::max(0, n_tokens) * (size_t)hidden * sizeof(float);
            const auto * header =
                static_cast<const BackendIpcSharedPayloadHeader *>(shared_payload);
            if (shared_payload && shared_payload != MAP_FAILED && shared_payload_data &&
                seq != 0 && n_tokens > 0 && bytes == expected_bytes &&
                backend_ipc_payload_in_bounds(0, bytes, shared_payload_capacity) &&
                header->sequence == seq && header->bytes == (uint64_t)bytes) {
                host_act.assign(bytes / sizeof(float), 0.0f);
                std::memcpy(host_act.data(), shared_payload_data, bytes);
                payload_ok = true;
            }
        } else {
            if (cmd == "reset_request_state") {
                const bool ok = callbacks.reset_request_state
                    ? callbacks.reset_request_state()
                    : false;
                stream_daemon_status(callbacks, stream_fd, ok ? 0 : -1);
                continue;
            }
            if (cmd == "kvflash_sync_identity") {
                int committed = -1;
                iss >> committed;
                const bool ok = callbacks.kvflash_sync_identity
                    ? callbacks.kvflash_sync_identity(committed)
                    : false;
                stream_daemon_status(callbacks, stream_fd, ok ? 0 : -1);
                continue;
            }
            if (cmd == "prefix_snapshot_save") {
                int slot = -1;
                iss >> slot;
                const bool ok = callbacks.snapshot_save
                    ? callbacks.snapshot_save(slot)
                    : false;
                stream_daemon_status(callbacks, stream_fd, ok ? 0 : -1);
                continue;
            }
            if (cmd == "prefix_snapshot_free") {
                int slot = -1;
                iss >> slot;
                if (callbacks.snapshot_free) callbacks.snapshot_free(slot);
                stream_daemon_status(callbacks, stream_fd, 0);
                continue;
            }
            if (cmd == "prefix_snapshot_restore") {
                int slot = -1;
                iss >> slot;
                const bool ok = callbacks.snapshot_restore
                    ? callbacks.snapshot_restore(slot)
                    : false;
                stream_daemon_status(callbacks, stream_fd, ok ? 0 : -1);
                continue;
            }
            std::fprintf(stderr, "[%s] unknown command: %s\n",
                         prefix, line.c_str());
            stream_daemon_status(callbacks, stream_fd, -1);
            continue;
        }

        bool ok = payload_ok && base_pos >= 0 && n_tokens > 0;
        if (ok && has_token_ids) {
            ok = payload_fd >= 0 && token_count == n_tokens;
            if (ok) {
                token_ids.assign((size_t)n_tokens, 0);
                ok = read_exact_fd(payload_fd, token_ids.data(),
                                   sizeof(int32_t) * token_ids.size());
            }
        } else {
            token_ids.clear();
            ok = ok && token_count == 0;
        }

        TargetShardDaemonForwardResponse resp;
        if (ok) {
            TargetShardDaemonForwardRequest req;
            req.base_pos = base_pos;
            req.n_tokens = n_tokens;
            req.ubatch = forward_ubatch > 0 ? forward_ubatch : n_tokens;
            req.want_argmax = want_argmax != 0;
            req.want_logits = want_logits != 0;
            req.boundary_activation = &host_act;
            req.token_ids = has_token_ids ? &token_ids : nullptr;
            ok = callbacks.forward(req, resp);
        }

        const int32_t status = ok ? 0 : -1;
        if (!write_exact_fd(stream_fd, &status, sizeof(status))) break;
        if (!ok) continue;

        if (!write_exact_fd(stream_fd, &resp.last_tok, sizeof(resp.last_tok))) break;
        if (want_argmax) {
            if ((int)resp.argmax.size() != n_tokens ||
                !write_exact_fd(stream_fd, resp.argmax.data(),
                                sizeof(int32_t) * resp.argmax.size())) {
                break;
            }
        }
        if (want_logits) {
            const int logits_tokens = want_argmax ? n_tokens : 1;
            const size_t expected_logits =
                (size_t)logits_tokens * (size_t)vocab;
            if (resp.logits.size() != expected_logits ||
                !write_exact_fd(stream_fd, resp.logits.data(),
                                sizeof(float) * resp.logits.size())) {
                break;
            }
        }
    }

    if (shared_payload && shared_payload != MAP_FAILED) {
        ::munmap(shared_payload, shared_payload_map_bytes);
    }
    return 0;
#endif
}

}  // namespace dflash::common
