// dflash_draft_ipc_daemon.cpp — generic DFlash draft IPC daemon entry point.
//
// The DFlash draft model is a single universal Qwen3-style network shared
// across every target architecture, so the daemon body is target-agnostic
// and lives in the common/ layer. Only the CLI flag wiring (each target's
// _daemon.cpp) decides whether to launch this daemon.

#include "dflash_draft_ipc.h"
#include "internal.h"
#include "dflash_feature_ring.h"
#include "dflash_draft_graph.h"
#include "step_graph.h"
#include "io_utils.h"

#include "ggml.h"
#include "ggml-alloc.h"
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

namespace {

enum class ProposalResult {
    Ok,
    CommandFailed,
    StreamFailed,
};

}  // namespace

int run_dflash_draft_ipc_daemon(const char * draft_path,
                                int ring_cap,
                                int draft_gpu,
                                int stream_fd,
                                int payload_fd,
                                int shared_payload_fd,
                                size_t shared_payload_bytes) {
#if defined(_WIN32)
    (void)draft_path; (void)ring_cap; (void)draft_gpu; (void)stream_fd;
    (void)payload_fd; (void)shared_payload_fd; (void)shared_payload_bytes;
    std::fprintf(stderr, "DFlash draft IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    if (!draft_path || ring_cap <= 0 || stream_fd < 0) {
        std::fprintf(stderr,
                     "usage: backend_ipc_daemon --backend-ipc-mode=dflash-draft "
                     "<draft> --ring-cap=N --stream-fd=FD [--draft-gpu=N]\n");
        return 2;
    }

    void * shared_payload = nullptr;
    void * shared_payload_data = nullptr;
    size_t shared_payload_capacity = 0;
    size_t shared_payload_map_bytes = 0;
    if (shared_payload_fd >= 0 || shared_payload_bytes > 0) {
        if (shared_payload_fd < 0 || shared_payload_bytes == 0) {
            std::fprintf(stderr, "[draft-ipc-daemon] bad shared payload fd/size\n");
            stream_status(stream_fd, -1);
            return 1;
        }
        if (!backend_ipc_shared_payload_map_bytes(shared_payload_bytes,
                                                  shared_payload_map_bytes)) {
            std::fprintf(stderr, "[draft-ipc-daemon] bad shared payload size\n");
            stream_status(stream_fd, -1);
            return 1;
        }
        shared_payload = ::mmap(nullptr, shared_payload_map_bytes,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED, shared_payload_fd, 0);
        if (shared_payload == MAP_FAILED) {
            std::fprintf(stderr, "[draft-ipc-daemon] shared payload mmap failed\n");
            stream_status(stream_fd, -1);
            return 1;
        }
        shared_payload_data =
            static_cast<char *>(shared_payload) + backend_ipc_shared_payload_header_bytes();
        shared_payload_capacity = shared_payload_bytes;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(std::max(0, draft_gpu));
    if (!backend) {
        std::fprintf(stderr, "[draft-ipc-daemon] backend init failed gpu=%d\n", draft_gpu);
        stream_status(stream_fd, -1);
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }

    DraftWeights draft_weights;
    std::string dp(draft_path);
    bool draft_ok = false;
    if (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf") {
        draft_ok = load_draft_gguf(draft_path, backend, draft_weights);
    } else {
        draft_ok = load_draft_safetensors(draft_path, backend, draft_weights);
    }
    if (!draft_ok) {
        std::fprintf(stderr, "[draft-ipc-daemon] draft load failed: %s\n",
                     dflash27b_last_error());
        stream_status(stream_fd, -1);
        ggml_backend_free(backend);
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }

    DraftFeatureMirror feature_ring;
    if (!draft_feature_mirror_init(feature_ring, backend, draft_gpu, draft_gpu, ring_cap,
                                   draft_weights.n_target_layers, draft_weights.n_embd)) {
        std::fprintf(stderr, "[draft-ipc-daemon] feature ring init failed cap=%d gpu=%d\n",
                     ring_cap, draft_gpu);
        stream_status(stream_fd, -1);
        free_draft_weights(draft_weights);
        ggml_backend_free(backend);
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }

    std::fprintf(stderr, "[draft-ipc-daemon] ready gpu=%d ring_cap=%d\n",
                 draft_gpu, ring_cap);
    stream_status(stream_fd, 0);

    const int hidden = draft_weights.n_embd;
    const int q_len = draft_weights.block_size;
    const int n_tgt_layers = draft_weights.n_target_layers;
    StepGraph draft_sg;
    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<float> hidden_out((size_t)hidden * q_len);

    auto store_feature_slice = [&](int capture_idx,
                                   int start_pos,
                                   int n_tokens,
                                   const std::vector<float> & slice) -> bool {
        const size_t expected = (size_t)n_tokens * (size_t)hidden;
        if (slice.size() != expected) return false;
        const size_t dst_stride = feature_ring.target_feat->nb[1];
        const size_t slice_offset =
            (size_t)capture_idx * (size_t)hidden * sizeof(float);
        for (int i = 0; i < n_tokens; i++) {
            const int slot = (start_pos + i) % feature_ring.cap;
            const size_t dst_off = (size_t)slot * dst_stride + slice_offset;
            ggml_backend_tensor_set(feature_ring.target_feat,
                                    slice.data() + (size_t)i * hidden,
                                    dst_off,
                                    (size_t)hidden * sizeof(float));
        }
        ggml_backend_synchronize(backend);
        return true;
    };

    auto run_proposal = [&](int committed, int ctx_len) -> ProposalResult {
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_ring, committed, ctx_len, mirror_slot0);
        if (!build_draft_step(draft_sg, draft_weights, /*lm_head=*/nullptr, backend,
                              ctx_len, use_mirror_view ? &feature_ring : nullptr,
                              committed,
                              /*ctx_len_max=*/feature_ring.cap)) {
            std::fprintf(stderr, "[draft-ipc-daemon] draft build failed\n");
            stream_status(stream_fd, -1);
            return ProposalResult::CommandFailed;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_ring,
                                               draft_sg.target_hidden_cat,
                                               committed - ctx_len,
                                               ctx_len)) {
            std::fprintf(stderr, "[draft-ipc-daemon] feature copy failed\n");
            stream_status(stream_fd, -1);
            return ProposalResult::CommandFailed;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                noise_embed.size() * sizeof(float));
        pos_k.resize((size_t)ctx_len + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = ctx_len + i;
        for (int i = 0; i < ctx_len + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                pos_q.size() * sizeof(int32_t));
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                pos_k.size() * sizeof(int32_t));
        auto st = ggml_backend_graph_compute(backend, draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[draft-ipc-daemon] draft compute failed status=%d\n",
                         (int)st);
            stream_status(stream_fd, -1);
            return ProposalResult::CommandFailed;
        }
        ggml_backend_tensor_get(draft_sg.hidden_states, hidden_out.data(), 0,
                                hidden_out.size() * sizeof(float));
        if (!stream_status(stream_fd, 0) ||
            !write_exact_fd(stream_fd, hidden_out.data(),
                            hidden_out.size() * sizeof(float))) {
            std::fprintf(stderr, "[draft-ipc-daemon] stream write failed\n");
            return ProposalResult::StreamFailed;
        }
        return ProposalResult::Ok;
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") {
            break;
        }
        if (cmd == "feature_slice") {
            int capture_idx = -1;
            int start_pos = -1;
            int n_tokens = 0;
            iss >> capture_idx >> start_pos >> n_tokens;
            std::string path = read_line_tail(iss);
            if (capture_idx < 0 || capture_idx >= n_tgt_layers ||
                start_pos < 0 || n_tokens <= 0 || path.empty()) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad feature_slice: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            std::vector<float> slice((size_t)n_tokens * hidden);
            if (!read_binary_file_exact(path, slice.data(), slice.size() * sizeof(float))) {
                std::fprintf(stderr, "[draft-ipc-daemon] read feature_slice failed: %s\n",
                             path.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            if (!store_feature_slice(capture_idx, start_pos, n_tokens, slice)) {
                std::fprintf(stderr, "[draft-ipc-daemon] store feature_slice failed\n");
                stream_status(stream_fd, -1);
                continue;
            }
            stream_status(stream_fd, 0);
            continue;
        }
        if (cmd == "feature_slice_pipe") {
            int capture_idx = -1;
            int start_pos = -1;
            int n_tokens = 0;
            size_t bytes = 0;
            iss >> capture_idx >> start_pos >> n_tokens >> bytes;
            const size_t expected_bytes = (size_t)n_tokens * (size_t)hidden * sizeof(float);
            if (payload_fd < 0 || capture_idx < 0 || capture_idx >= n_tgt_layers ||
                start_pos < 0 || n_tokens <= 0 || bytes != expected_bytes) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad feature_slice_pipe: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            std::vector<float> slice(bytes / sizeof(float));
            if (!read_exact_fd(payload_fd, slice.data(), bytes)) {
                std::fprintf(stderr, "[draft-ipc-daemon] read feature_slice pipe failed\n");
                stream_status(stream_fd, -1);
                break;
            }
            if (!store_feature_slice(capture_idx, start_pos, n_tokens, slice)) {
                std::fprintf(stderr, "[draft-ipc-daemon] store feature_slice_pipe failed\n");
                stream_status(stream_fd, -1);
                continue;
            }
            stream_status(stream_fd, 0);
            continue;
        }
        if (cmd == "get_feature_range") {
            int start_pos = -1;
            int n_tokens = 0;
            iss >> start_pos >> n_tokens;
            if (start_pos < 0 || n_tokens <= 0 || n_tokens > feature_ring.cap) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad get_feature_range: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            const int fc_in = feature_ring.n_target_layers * feature_ring.hidden_size;
            const size_t row_bytes = (size_t)fc_in * sizeof(float);
            const size_t src_stride = feature_ring.target_feat->nb[1];
            // Stream in bounded chunks so daemon host memory stays O(chunk)
            // instead of O(n_tokens) — avoids large transient allocs / OOM
            // on long-context snapshot ranges. Total bytes on the wire are
            // unchanged, so the client read is identical.
            constexpr int kFeatRangeChunkTok = 256;
            const int chunk_tok = std::min(kFeatRangeChunkTok, n_tokens);
            std::vector<float> data((size_t)chunk_tok * (size_t)fc_in);
            if (!stream_status(stream_fd, 0)) {
                std::fprintf(stderr, "[draft-ipc-daemon] feature range stream failed\n");
                break;
            }
            bool stream_ok = true;
            for (int base = 0; base < n_tokens && stream_ok; base += chunk_tok) {
                const int nc = std::min(chunk_tok, n_tokens - base);
                for (int i = 0; i < nc; ++i) {
                    const int slot = (start_pos + base + i) % feature_ring.cap;
                    ggml_backend_tensor_get(feature_ring.target_feat,
                                            data.data() + (size_t)i * (size_t)fc_in,
                                            (size_t)slot * src_stride,
                                            row_bytes);
                }
                stream_ok = write_exact_fd(stream_fd, data.data(),
                                           (size_t)nc * row_bytes);
            }
            if (!stream_ok) {
                std::fprintf(stderr, "[draft-ipc-daemon] feature range stream failed\n");
                break;
            }
            continue;
        }
        if (cmd == "set_feature_range") {
            int start_pos = -1;
            int n_tokens = 0;
            iss >> start_pos >> n_tokens;
            std::string path = read_line_tail(iss);
            if (start_pos < 0 || n_tokens <= 0 || n_tokens > feature_ring.cap ||
                path.empty()) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad set_feature_range: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            const int fc_in = feature_ring.n_target_layers * feature_ring.hidden_size;
            const size_t row_bytes = (size_t)fc_in * sizeof(float);
            const size_t dst_stride = feature_ring.target_feat->nb[1];
            // Read + apply in bounded chunks so daemon host memory stays
            // O(chunk) instead of O(n_tokens) — avoids large transient allocs
            // / OOM on long-context snapshot restore.
            constexpr int kFeatRangeChunkTok = 256;
            const int chunk_tok = std::min(kFeatRangeChunkTok, n_tokens);
            std::ifstream fin(path, std::ios::binary);
            if (!fin) {
                std::fprintf(stderr, "[draft-ipc-daemon] read feature range failed: %s\n",
                             path.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            std::vector<float> data((size_t)chunk_tok * (size_t)fc_in);
            bool read_ok = true;
            for (int base = 0; base < n_tokens && read_ok; base += chunk_tok) {
                const int nc = std::min(chunk_tok, n_tokens - base);
                const std::streamsize want = (std::streamsize)((size_t)nc * row_bytes);
                fin.read(reinterpret_cast<char *>(data.data()), want);
                if (fin.gcount() != want) {
                    read_ok = false;
                    break;
                }
                for (int i = 0; i < nc; ++i) {
                    const int slot = (start_pos + base + i) % feature_ring.cap;
                    ggml_backend_tensor_set(feature_ring.target_feat,
                                            data.data() + (size_t)i * (size_t)fc_in,
                                            (size_t)slot * dst_stride,
                                            row_bytes);
                }
            }
            if (!read_ok) {
                std::fprintf(stderr, "[draft-ipc-daemon] read feature range failed: %s\n",
                             path.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            ggml_backend_synchronize(backend);
            stream_status(stream_fd, 0);
            continue;
        }
        if (cmd == "feature_slice_shared") {
            int capture_idx = -1;
            int start_pos = -1;
            int n_tokens = 0;
            size_t bytes = 0;
            uint64_t sequence = 0;
            iss >> capture_idx >> start_pos >> n_tokens >> bytes >> sequence;
            const size_t expected_bytes = (size_t)n_tokens * (size_t)hidden * sizeof(float);
            const auto * header =
                static_cast<const BackendIpcSharedPayloadHeader *>(shared_payload);
            if (!iss || sequence == 0 || !shared_payload || !shared_payload_data ||
                capture_idx < 0 || capture_idx >= n_tgt_layers ||
                start_pos < 0 || n_tokens <= 0 || bytes != expected_bytes ||
                !backend_ipc_payload_in_bounds(0, bytes, shared_payload_capacity) ||
                header->sequence != sequence || header->bytes != (uint64_t)bytes) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad feature_slice_shared: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            std::vector<float> slice(bytes / sizeof(float));
            std::memcpy(slice.data(), shared_payload_data, bytes);
            if (!store_feature_slice(capture_idx, start_pos, n_tokens, slice)) {
                std::fprintf(stderr, "[draft-ipc-daemon] store feature_slice_shared failed\n");
                stream_status(stream_fd, -1);
                continue;
            }
            stream_status(stream_fd, 0);
            continue;
        }
        if (cmd == "propose") {
            int committed = -1;
            int ctx_len = 0;
            iss >> committed >> ctx_len;
            std::string path = read_line_tail(iss);
            if (committed < 0 || ctx_len <= 0 || ctx_len > feature_ring.cap || path.empty()) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad propose: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            if (!read_binary_file_exact(path, noise_embed.data(),
                                        noise_embed.size() * sizeof(float))) {
                std::fprintf(stderr, "[draft-ipc-daemon] read noise failed: %s\n",
                             path.c_str());
                stream_status(stream_fd, -1);
                continue;
            }

            if (run_proposal(committed, ctx_len) == ProposalResult::StreamFailed) {
                break;
            }
            continue;
        }
        if (cmd == "propose_shared") {
            int committed = -1;
            int ctx_len = 0;
            size_t bytes = 0;
            uint64_t sequence = 0;
            iss >> committed >> ctx_len >> bytes >> sequence;
            const size_t expected_bytes = noise_embed.size() * sizeof(float);
            const auto * header =
                static_cast<const BackendIpcSharedPayloadHeader *>(shared_payload);
            if (!iss || sequence == 0 || !shared_payload || !shared_payload_data ||
                committed < 0 || ctx_len <= 0 || ctx_len > feature_ring.cap ||
                bytes != expected_bytes ||
                !backend_ipc_payload_in_bounds(0, bytes, shared_payload_capacity) ||
                header->sequence != sequence || header->bytes != (uint64_t)bytes) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad propose_shared: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            std::memcpy(noise_embed.data(), shared_payload_data, bytes);
            if (run_proposal(committed, ctx_len) == ProposalResult::StreamFailed) {
                break;
            }
            continue;
        }
        if (cmd == "propose_pipe") {
            int committed = -1;
            int ctx_len = 0;
            size_t bytes = 0;
            iss >> committed >> ctx_len >> bytes;
            const size_t expected_bytes = noise_embed.size() * sizeof(float);
            if (payload_fd < 0 || committed < 0 || ctx_len <= 0 ||
                ctx_len > feature_ring.cap || bytes != expected_bytes) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad propose_pipe: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            if (!read_exact_fd(payload_fd, noise_embed.data(), bytes)) {
                std::fprintf(stderr, "[draft-ipc-daemon] read noise pipe failed\n");
                stream_status(stream_fd, -1);
                break;
            }
            if (run_proposal(committed, ctx_len) == ProposalResult::StreamFailed) {
                break;
            }
            continue;
        }
        std::fprintf(stderr, "[draft-ipc-daemon] unknown command: %s\n", line.c_str());
        stream_status(stream_fd, -1);
    }

    step_graph_destroy(draft_sg);
    draft_feature_mirror_free(feature_ring);
    free_draft_weights(draft_weights);
    ggml_backend_free(backend);
    if (shared_payload && shared_payload != MAP_FAILED) {
        ::munmap(shared_payload, shared_payload_map_bytes);
    }
    std::fprintf(stderr, "[draft-ipc-daemon] stopped\n");
    return 0;
#endif
}

} // namespace dflash::common
