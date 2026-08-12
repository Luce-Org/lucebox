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
#include <limits>
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

bool forward_payload_bytes(int hidden, int n_tokens, size_t & bytes) {
    if (hidden <= 0 || n_tokens <= 0 ||
        static_cast<size_t>(n_tokens) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(hidden)) {
        return false;
    }
    const size_t elements = static_cast<size_t>(n_tokens) *
        static_cast<size_t>(hidden);
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    bytes = elements * sizeof(float);
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
        int32_t semantic_phase_value = 0;
        size_t bytes = 0;
        bool payload_ok = false;
        bool framing_fields_present = false;
        bool command_fields_ok = false;
        bool pipe_forward = false;
        bool shared_forward = false;
        uint64_t shared_seq = 0;

        if (cmd == "forward_pipe") {
            pipe_forward = true;
            iss >> base_pos >> n_tokens >> want_argmax >> want_logits >> bytes >>
                has_token_ids >> forward_ubatch >> token_count;
            framing_fields_present = static_cast<bool>(iss);
            if (framing_fields_present) {
                iss >> semantic_phase_value;
                const bool phase_present = static_cast<bool>(iss);
                if (phase_present) {
                    iss >> std::ws;
                    command_fields_ok = iss.eof();
                }
            }
        } else if (cmd == "forward_shared") {
            shared_forward = true;
            iss >> base_pos >> n_tokens >> want_argmax >> want_logits >> bytes >>
                shared_seq >> has_token_ids >> forward_ubatch >> token_count;
            framing_fields_present = static_cast<bool>(iss);
            if (framing_fields_present) {
                iss >> semantic_phase_value;
                const bool phase_present = static_cast<bool>(iss);
                if (phase_present) {
                    iss >> std::ws;
                    command_fields_ok = iss.eof();
                }
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

        size_t expected_bytes = 0;
        const bool framing_ok = framing_fields_present && base_pos >= 0 &&
            forward_ubatch >= 0 && (want_argmax == 0 || want_argmax == 1) &&
            (want_logits == 0 || want_logits == 1) &&
            (has_token_ids == 0 || has_token_ids == 1) &&
            forward_payload_bytes(hidden, n_tokens, expected_bytes) &&
            bytes == expected_bytes &&
            token_count == (has_token_ids ? n_tokens : 0);

        if (framing_ok && pipe_forward && payload_fd >= 0) {
            host_act.assign(bytes / sizeof(float), 0.0f);
            payload_ok = read_exact_fd(payload_fd, host_act.data(), bytes);
        } else if (framing_ok && shared_forward) {
            const auto * header =
                static_cast<const BackendIpcSharedPayloadHeader *>(shared_payload);
            if (shared_payload && shared_payload != MAP_FAILED &&
                shared_payload_data && shared_seq != 0 &&
                backend_ipc_payload_in_bounds(0, bytes, shared_payload_capacity) &&
                backend_ipc_shared_payload_header_matches(
                    header, shared_seq, static_cast<uint64_t>(bytes))) {
                host_act.assign(bytes / sizeof(float), 0.0f);
                std::memcpy(host_act.data(), shared_payload_data, bytes);
                payload_ok = true;
            }
        }

        const int token_fd = payload_fd >= 0 ? payload_fd : stream_fd;
        bool token_ids_ok = !has_token_ids;
        token_ids.clear();
        if (framing_ok && has_token_ids) {
            token_ids.assign(static_cast<size_t>(n_tokens), 0);
            token_ids_ok = read_exact_fd(
                token_fd, token_ids.data(), sizeof(int32_t) * token_ids.size());
        }

        InferencePhase semantic_phase = InferencePhase::Unspecified;
        const bool semantic_phase_ok = command_fields_ok &&
            inference_phase_from_wire_value(semantic_phase_value, semantic_phase);
        bool ok = framing_ok && payload_ok && token_ids_ok && semantic_phase_ok;

        TargetShardDaemonForwardResponse resp;
        if (ok) {
            TargetShardDaemonForwardRequest req;
            req.base_pos = base_pos;
            req.n_tokens = n_tokens;
            req.ubatch = forward_ubatch > 0 ? forward_ubatch : n_tokens;
            req.want_argmax = want_argmax != 0;
            req.want_logits = want_logits != 0;
            req.semantic_phase = semantic_phase;
            req.boundary_activation = &host_act;
            req.token_ids = has_token_ids ? &token_ids : nullptr;
            ok = callbacks.forward(req, resp);
        }

        const int32_t status = ok ? 0 : -1;
        if (!write_exact_fd(stream_fd, &status, sizeof(status))) break;
        if (!framing_ok) break;
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
