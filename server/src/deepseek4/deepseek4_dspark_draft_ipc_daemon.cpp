// Out-of-process DeepSeek4 DSpark proposal worker.

#include "deepseek4_dspark.h"

#include "common/dflash_draft_ipc.h"
#include "common/io_utils.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

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
#  include <unistd.h>
#endif

namespace dflash::common {

int run_deepseek4_dspark_draft_ipc_daemon(
        const char * draft_path,
        int ring_cap,
        int draft_gpu,
        int stream_fd,
        int payload_fd,
        int shared_payload_fd,
        size_t shared_payload_bytes) {
#if defined(_WIN32)
    (void) draft_path;
    (void) ring_cap;
    (void) draft_gpu;
    (void) stream_fd;
    (void) payload_fd;
    (void) shared_payload_fd;
    (void) shared_payload_bytes;
    return 2;
#else
    const bool shared_requested =
        shared_payload_fd >= 0 || shared_payload_bytes > 0;
    if (!draft_path || !*draft_path || ring_cap <= 0 ||
        stream_fd < 0 || (payload_fd < 0 && !shared_requested) ||
        (shared_requested &&
         (shared_payload_fd < 0 || shared_payload_bytes == 0))) {
        std::fprintf(stderr, "[ds4-dspark-ipc] bad daemon configuration\n");
        if (stream_fd >= 0) stream_status(stream_fd, -1);
        return 2;
    }

    void * shared_payload = nullptr;
    void * shared_payload_data = nullptr;
    size_t shared_payload_map_bytes = 0;
    if (shared_requested) {
        if (!backend_ipc_shared_payload_map_bytes(
                shared_payload_bytes, shared_payload_map_bytes)) {
            std::fprintf(stderr, "[ds4-dspark-ipc] invalid shared payload size\n");
            stream_status(stream_fd, -1);
            return 1;
        }
        shared_payload = ::mmap(nullptr, shared_payload_map_bytes,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                shared_payload_fd, 0);
        if (shared_payload == MAP_FAILED) {
            std::fprintf(stderr, "[ds4-dspark-ipc] shared payload mmap failed\n");
            stream_status(stream_fd, -1);
            return 1;
        }
        shared_payload_data = static_cast<char *>(shared_payload) +
                              backend_ipc_shared_payload_header_bytes();
    }
    auto unmap_shared = [&]() {
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
            shared_payload = nullptr;
            shared_payload_data = nullptr;
        }
    };
    auto shared_request_valid = [&](size_t bytes, uint64_t sequence) {
        const auto * header =
            static_cast<const BackendIpcSharedPayloadHeader *>(shared_payload);
        return sequence != 0 && shared_payload && shared_payload_data &&
               backend_ipc_payload_in_bounds(
                   0, bytes, shared_payload_bytes) &&
               header->sequence == sequence && header->bytes == bytes;
    };

    ggml_backend_t backend = ggml_backend_cuda_init(std::max(0, draft_gpu));
    if (!backend) {
        std::fprintf(stderr, "[ds4-dspark-ipc] backend init failed gpu=%d\n",
                     draft_gpu);
        stream_status(stream_fd, -1);
        unmap_shared();
        return 1;
    }

    DSparkDrafter drafter;
    if (!load_deepseek4_dspark_drafter(draft_path, backend, drafter)) {
        std::fprintf(stderr, "[ds4-dspark-ipc] draft load failed: %s\n",
                     deepseek4_dspark_last_error());
        stream_status(stream_fd, -1);
        ggml_backend_free(backend);
        unmap_shared();
        return 1;
    }

    const int hidden = drafter.core.n_embd;
    const int block = drafter.block_size;
    const int n_target_layers = drafter.n_target_layers;
    const int feature_row = hidden * n_target_layers;
    if (hidden <= 0 || block <= 0 || n_target_layers <= 0 || feature_row <= 0) {
        std::fprintf(stderr, "[ds4-dspark-ipc] invalid draft dimensions\n");
        stream_status(stream_fd, -1);
        free_deepseek4_dspark_drafter(drafter);
        ggml_backend_free(backend);
        unmap_shared();
        return 1;
    }

    std::vector<float> feature_ring((size_t) ring_cap * feature_row, 0.0f);
    std::vector<float> noise_embed((size_t) hidden * block);
    std::vector<float> context;
    std::vector<float> hidden_out;

    auto ring_slot = [ring_cap](int position) {
        int slot = position % ring_cap;
        return slot < 0 ? slot + ring_cap : slot;
    };

    auto store_feature_slice = [&](int capture_idx, int start_pos, int n_tokens,
                                   const std::vector<float> & slice) {
        if (capture_idx < 0 || capture_idx >= n_target_layers ||
            start_pos < 0 || n_tokens <= 0 ||
            slice.size() != (size_t) n_tokens * hidden) {
            return false;
        }
        for (int i = 0; i < n_tokens; ++i) {
            float * dst = feature_ring.data() +
                         (size_t) ring_slot(start_pos + i) * feature_row +
                         (size_t) capture_idx * hidden;
            std::memcpy(dst, slice.data() + (size_t) i * hidden,
                        sizeof(float) * (size_t) hidden);
        }
        return true;
    };

    auto store_feature_block = [&](int start_pos, int n_tokens,
                                   const float * features,
                                   size_t feature_count) {
        const size_t expected = (size_t) n_tokens * (size_t) feature_row;
        if (start_pos < 0 || n_tokens <= 0 || n_tokens > ring_cap ||
            !features || feature_count != expected) {
            return false;
        }
        for (int i = 0; i < n_tokens; ++i) {
            float * dst = feature_ring.data() +
                         (size_t) ring_slot(start_pos + i) * feature_row;
            std::memcpy(dst, features + (size_t) i * feature_row,
                        sizeof(float) * (size_t) feature_row);
        }
        return true;
    };

    auto run_proposal = [&](int committed, int ctx_len,
                            const float * noise) {
        if (committed < 0 || ctx_len <= 0 || ctx_len > ring_cap || !noise) {
            return false;
        }
        context.resize((size_t) ctx_len * feature_row);
        const int start_pos = committed - ctx_len;
        for (int i = 0; i < ctx_len; ++i) {
            const float * src = feature_ring.data() +
                               (size_t) ring_slot(start_pos + i) * feature_row;
            std::memcpy(context.data() + (size_t) i * feature_row, src,
                        sizeof(float) * (size_t) feature_row);
        }
        return deepseek4_dspark_draft_forward(
            backend, drafter, noise, context.data(), ctx_len,
            committed, hidden_out);
    };

    std::fprintf(stderr,
                 "[ds4-dspark-ipc] ready gpu=%d ring_cap=%d hidden=%d block=%d captures=%d transport=%s\n",
                 draft_gpu, ring_cap, hidden, block, n_target_layers,
                 shared_payload ? "shared" : "stream");
    if (!stream_status(stream_fd, 0)) {
        free_deepseek4_dspark_drafter(drafter);
        ggml_backend_free(backend);
        unmap_shared();
        return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") break;

        if (cmd == "feature_slice_pipe") {
            int capture_idx = -1;
            int start_pos = -1;
            int n_tokens = 0;
            size_t bytes = 0;
            iss >> capture_idx >> start_pos >> n_tokens >> bytes;
            const size_t expected = (size_t) std::max(0, n_tokens) * hidden *
                                    sizeof(float);
            bool ok = iss && bytes == expected && n_tokens > 0;
            std::vector<float> slice(ok ? bytes / sizeof(float) : 0);
            if (ok) ok = read_exact_fd(payload_fd, slice.data(), bytes);
            if (ok) ok = store_feature_slice(capture_idx, start_pos, n_tokens, slice);
            if (!stream_status(stream_fd, ok ? 0 : -1)) break;
            continue;
        }

        if (cmd == "feature_slice_shared") {
            int capture_idx = -1;
            int start_pos = -1;
            int n_tokens = 0;
            size_t bytes = 0;
            uint64_t sequence = 0;
            iss >> capture_idx >> start_pos >> n_tokens >> bytes >> sequence;
            const size_t expected = (size_t) std::max(0, n_tokens) * hidden *
                                    sizeof(float);
            bool ok = iss && bytes == expected && n_tokens > 0 &&
                      capture_idx >= 0 && capture_idx < n_target_layers &&
                      shared_request_valid(bytes, sequence);
            std::vector<float> slice(ok ? bytes / sizeof(float) : 0);
            if (ok) std::memcpy(slice.data(), shared_payload_data, bytes);
            if (ok) {
                ok = store_feature_slice(
                    capture_idx, start_pos, n_tokens, slice);
            }
            if (!stream_status(stream_fd, ok ? 0 : -1)) break;
            continue;
        }

        if (cmd == "feature_block_pipe") {
            int start_pos = -1;
            int n_tokens = 0;
            size_t bytes = 0;
            iss >> start_pos >> n_tokens >> bytes;
            const size_t expected = (size_t) std::max(0, n_tokens) *
                                    (size_t) feature_row * sizeof(float);
            bool ok = iss && payload_fd >= 0 && bytes == expected &&
                      n_tokens > 0 && n_tokens <= ring_cap;
            std::vector<float> features(ok ? bytes / sizeof(float) : 0);
            if (ok) ok = read_exact_fd(payload_fd, features.data(), bytes);
            if (ok) {
                ok = store_feature_block(
                    start_pos, n_tokens, features.data(), features.size());
            }
            if (!stream_status(stream_fd, ok ? 0 : -1)) break;
            continue;
        }

        if (cmd == "feature_block_shared") {
            int start_pos = -1;
            int n_tokens = 0;
            size_t bytes = 0;
            uint64_t sequence = 0;
            iss >> start_pos >> n_tokens >> bytes >> sequence;
            const size_t expected = (size_t) std::max(0, n_tokens) *
                                    (size_t) feature_row * sizeof(float);
            bool ok = iss && bytes == expected && n_tokens > 0 &&
                      n_tokens <= ring_cap &&
                      shared_request_valid(bytes, sequence);
            if (ok) {
                ok = store_feature_block(
                    start_pos, n_tokens,
                    static_cast<const float *>(shared_payload_data),
                    bytes / sizeof(float));
            }
            if (!stream_status(stream_fd, ok ? 0 : -1)) break;
            continue;
        }

        if (cmd == "propose_pipe") {
            int committed = -1;
            int ctx_len = 0;
            size_t bytes = 0;
            iss >> committed >> ctx_len >> bytes;
            const size_t expected = noise_embed.size() * sizeof(float);
            bool ok = iss && bytes == expected;
            if (!ok) {
                std::fprintf(stderr,
                             "[ds4-dspark-ipc] bad propose header committed=%d ctx=%d bytes=%zu expected=%zu\n",
                             committed, ctx_len, bytes, expected);
            }
            if (ok && !read_exact_fd(payload_fd, noise_embed.data(), bytes)) {
                std::fprintf(stderr, "[ds4-dspark-ipc] propose payload read failed\n");
                ok = false;
            }
            if (ok && !run_proposal(
                    committed, ctx_len, noise_embed.data())) {
                std::fprintf(stderr,
                             "[ds4-dspark-ipc] proposal graph failed committed=%d ctx=%d\n",
                             committed, ctx_len);
                ok = false;
            }
            if (!stream_status(stream_fd, ok ? 0 : -1)) break;
            if (ok && !write_exact_fd(stream_fd, hidden_out.data(),
                                      hidden_out.size() * sizeof(float))) {
                break;
            }
            continue;
        }

        if (cmd == "propose_shared_bidir") {
            int committed = -1;
            int ctx_len = 0;
            size_t bytes = 0;
            uint64_t sequence = 0;
            iss >> committed >> ctx_len >> bytes >> sequence;
            const size_t expected = noise_embed.size() * sizeof(float);
            bool ok = iss && bytes == expected &&
                      shared_request_valid(bytes, sequence);
            if (ok) {
                ok = run_proposal(
                    committed, ctx_len,
                    static_cast<const float *>(shared_payload_data));
            }
            if (ok && hidden_out.size() * sizeof(float) == bytes) {
                std::memcpy(shared_payload_data, hidden_out.data(), bytes);
                auto * header =
                    static_cast<BackendIpcSharedPayloadHeader *>(shared_payload);
                header->bytes = bytes;
                header->sequence = sequence;
            } else {
                ok = false;
            }
            if (!stream_status(stream_fd, ok ? 0 : -1)) break;
            continue;
        }

        std::fprintf(stderr, "[ds4-dspark-ipc] unknown command: %s\n",
                     line.c_str());
        if (!stream_status(stream_fd, -1)) break;
    }

    free_deepseek4_dspark_drafter(drafter);
    ggml_backend_free(backend);
    unmap_shared();
    return 0;
#endif
}

}  // namespace dflash::common
