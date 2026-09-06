#include "common/io_utils.h"
#include "common/backend_ipc.h"
#include "support/environment.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cstring>
#include <string>
#include <vector>
#include <limits>
#include <fcntl.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

TEST_CASE(ServerUnitFixture, test_backend_ipc_rejects_file_work_dir) {
    const std::string file_path = "/tmp/dflash_test_backend_ipc_work_dir_file";
    unlink(file_path.c_str());
    int fd = open(file_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    TEST_ASSERT(fd >= 0);
    if (fd >= 0) {
        const char payload[] = "not a dir";
        (void)write(fd, payload, sizeof(payload) - 1);
        close(fd);
    }

    BackendIpcLaunchConfig cfg;
    cfg.bin = "/bin/true";
    cfg.payload_path = "/tmp/dflash_test_backend_ipc_payload";
    cfg.work_dir = file_path;

    BackendIpcProcess proc;
    TEST_ASSERT(!proc.start(cfg));
    TEST_ASSERT(!proc.active());
    unlink(file_path.c_str());
}

TEST_CASE(ServerUnitFixture, test_backend_ipc_payload_pipe_round_trip) {
    int payload_pipe[2] = {-1, -1};
    int status_pipe[2] = {-1, -1};
    TEST_ASSERT(pipe(payload_pipe) == 0);
    TEST_ASSERT(pipe(status_pipe) == 0);
    const std::vector<float> payload = {1.0f, 2.5f, -3.0f, 4.25f};
    TEST_ASSERT(write_exact_fd(payload_pipe[1],
                               payload.data(),
                               payload.size() * sizeof(float)));
    close(payload_pipe[1]);
    payload_pipe[1] = -1;

    std::vector<float> received(payload.size(), 0.0f);
    TEST_ASSERT(read_exact_fd(payload_pipe[0],
                              received.data(),
                              received.size() * sizeof(float)));
    close(payload_pipe[0]);
    payload_pipe[0] = -1;
    TEST_ASSERT(received == payload);

    const int32_t ready = 0;
    TEST_ASSERT(write_exact_fd(status_pipe[1], &ready, sizeof(ready)));
    close(status_pipe[1]);
    status_pipe[1] = -1;
    int32_t status = -1;
    TEST_ASSERT(read_exact_fd(status_pipe[0], &status, sizeof(status)));
    TEST_ASSERT(status == 0);
    close(status_pipe[0]);
}

TEST_CASE(ServerUnitFixture, test_backend_ipc_payload_transport_parse) {
    BackendIpcMode mode = BackendIpcMode::DFlashDraft;
    TEST_ASSERT(parse_backend_ipc_mode("dflash-draft", mode));
    TEST_ASSERT(mode == BackendIpcMode::DFlashDraft);
    TEST_ASSERT(parse_backend_ipc_mode("pflash-compress", mode));
    TEST_ASSERT(mode == BackendIpcMode::PFlashCompress);
    TEST_ASSERT(parse_backend_ipc_mode("qwen35-target-shard", mode));
    TEST_ASSERT(mode == BackendIpcMode::Qwen35TargetShard);
    TEST_ASSERT(parse_backend_ipc_mode("moe-expert-compute", mode));
    TEST_ASSERT(mode == BackendIpcMode::MoeExpertCompute);
    TEST_ASSERT(!parse_backend_ipc_mode("moe-ffn", mode));

    BackendIpcPayloadTransport transport = BackendIpcPayloadTransport::Auto;
    TEST_ASSERT(parse_backend_ipc_payload_transport("stream", transport));
    TEST_ASSERT(transport == BackendIpcPayloadTransport::Stream);
    TEST_ASSERT(parse_backend_ipc_payload_transport("shared", transport));
    TEST_ASSERT(transport == BackendIpcPayloadTransport::Shared);
    TEST_ASSERT(parse_backend_ipc_payload_transport("auto", transport));
    TEST_ASSERT(transport == BackendIpcPayloadTransport::Auto);
    TEST_ASSERT(!parse_backend_ipc_payload_transport("pipe", transport));
    TEST_ASSERT(std::strcmp(
        backend_ipc_payload_transport_name(BackendIpcPayloadTransport::Stream),
        "stream") == 0);
}

TEST_CASE(ServerUnitFixture, test_backend_ipc_payload_bounds) {
    size_t out = 0;
    TEST_ASSERT(backend_ipc_checked_add_size(4, 8, out));
    TEST_ASSERT(out == 12);
    TEST_ASSERT(!backend_ipc_checked_add_size(
        std::numeric_limits<size_t>::max(), 1, out));
    TEST_ASSERT(backend_ipc_payload_in_bounds(0, 16, 16));
    TEST_ASSERT(backend_ipc_payload_in_bounds(4, 8, 16));
    TEST_ASSERT(!backend_ipc_payload_in_bounds(9, 8, 16));
    TEST_ASSERT(!backend_ipc_payload_in_bounds(
        std::numeric_limits<size_t>::max(), 1, 16));
}

TEST_CASE(ServerUnitFixture, test_backend_ipc_shared_payload_map_sizing) {
    size_t map_bytes = 0;
    TEST_ASSERT(backend_ipc_shared_payload_map_bytes(1024, map_bytes));
    TEST_ASSERT(map_bytes == 1024 + backend_ipc_shared_payload_header_bytes());

    BackendIpcSharedPayloadHeader header;
    backend_ipc_publish_shared_payload_header(&header, 7, 1024);
    TEST_ASSERT(backend_ipc_shared_payload_header_matches(&header, 7, 1024));
    TEST_ASSERT(!backend_ipc_shared_payload_header_matches(&header, 0, 1024));
    TEST_ASSERT(!backend_ipc_shared_payload_header_matches(&header, 8, 1024));
    TEST_ASSERT(!backend_ipc_shared_payload_header_matches(&header, 7, 512));
    uint64_t sequence = 0;
    uint64_t bytes = 0;
    backend_ipc_load_shared_payload_header(&header, sequence, bytes);
    TEST_ASSERT(sequence == 7);
    TEST_ASSERT(bytes == 1024);

    TEST_ASSERT(!backend_ipc_shared_payload_map_bytes(
        std::numeric_limits<size_t>::max(), map_bytes));
}

TEST_CASE(ServerUnitFixture, test_backend_ipc_shared_payload_segment_contract) {
    const BackendIpcPayloadSegment a{reinterpret_cast<const void *>(1), 16};
    const BackendIpcPayloadSegment b{reinterpret_cast<const void *>(2), 32};
    const BackendIpcPayloadSegment segments[] = {a, b};
    size_t total = 0;
    for (const BackendIpcPayloadSegment & segment : segments) {
        TEST_ASSERT(backend_ipc_checked_add_size(total, segment.bytes, total));
    }
    TEST_ASSERT(total == 48);
    TEST_ASSERT(backend_ipc_payload_in_bounds(0, total, 48));
    TEST_ASSERT(!backend_ipc_payload_in_bounds(0, total + 1, 48));
}
