// dflash_draft_ipc.cpp - DFlashDraftIpcClient + remote feature copy.
//
// Target-agnostic portion of the DFlash draft IPC: parent-side client that
// spawns the daemon, sends commands, and the row-extraction helper that
// ships feature slices to it.

#include "dflash_draft_ipc.h"
#include "io_utils.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace dflash::common {

// ── DFlashDraftIpcClient ────────────────────────────────────────────

bool DFlashDraftIpcClient::start(
        const std::string & bin,
        const std::string & draft_path,
        int draft_gpu,
        int ring_cap,
        const std::string & work_dir) {
#if defined(_WIN32)
    (void)bin; (void)draft_path; (void)draft_gpu; (void)ring_cap; (void)work_dir;
    std::fprintf(stderr, "DFlash draft IPC is only implemented on POSIX hosts\n");
    return false;
#else
    close();
    if (bin.empty() || draft_path.empty() || ring_cap <= 0) return false;
    BackendIpcLaunchConfig launch;
    launch.bin = bin;
    launch.mode = BackendIpcMode::DFlashDraft;
    launch.payload_path = draft_path;
    launch.work_dir = work_dir;
    launch.args.push_back("--ring-cap=" + std::to_string(ring_cap));
    launch.args.push_back("--draft-gpu=" + std::to_string(std::max(0, draft_gpu)));
    if (!process_.start(launch)) {
        std::fprintf(stderr, "draft-ipc backend process start failed\n");
        return false;
    }
    ring_cap_ = ring_cap;
    active_ = true;
    std::printf("[draft-ipc] ready bin=%s gpu=%d ring_cap=%d work_dir=%s\n",
                bin.c_str(), draft_gpu, ring_cap, process_.work_dir().c_str());
    return true;
#endif
}

bool DFlashDraftIpcClient::send_feature_slice(
        int capture_idx,
        int start_pos,
        int n_tokens,
        const std::vector<float> & slice) {
#if defined(_WIN32)
    (void)capture_idx; (void)start_pos; (void)n_tokens; (void)slice;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    const int payload_fd = process_.payload_fd();
    if (!active_ || !cmd || stream_fd < 0 || capture_idx < 0 ||
        capture_idx >= n_target_layers_ || start_pos < 0 || n_tokens <= 0) {
        return false;
    }
    const size_t expected = (size_t)n_tokens * hidden_size_;
    if (slice.size() != expected) return false;
    if (payload_fd >= 0) {
        const size_t bytes = slice.size() * sizeof(float);
        std::fprintf(cmd, "feature_slice_pipe %d %d %d %zu\n",
                     capture_idx, start_pos, n_tokens, bytes);
        std::fflush(cmd);
        if (!write_exact_fd(payload_fd, slice.data(), bytes)) {
            std::fprintf(stderr, "draft-ipc feature_slice payload write failed\n");
            return false;
        }
        int32_t status = -1;
        const bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
        if (!ok) {
            std::fprintf(stderr, "draft-ipc feature_slice failed status=%d\n", status);
        }
        return ok;
    }
    const std::string path = process_.next_path("feature");
    if (!write_binary_file(path, slice.data(), slice.size() * sizeof(float))) {
        std::fprintf(stderr, "draft-ipc write feature failed: %s\n", path.c_str());
        return false;
    }
    std::fprintf(cmd, "feature_slice %d %d %d %s\n",
                 capture_idx, start_pos, n_tokens, path.c_str());
    std::fflush(cmd);
    int32_t status = -1;
    const bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
    std::remove(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "draft-ipc feature_slice failed status=%d\n", status);
    }
    return ok;
#endif
}

bool DFlashDraftIpcClient::propose(
        int committed,
        int ctx_len,
        const std::vector<float> & noise_embed,
        std::vector<float> & hidden_out) {
#if defined(_WIN32)
    (void)committed; (void)ctx_len; (void)noise_embed; (void)hidden_out;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    const int payload_fd = process_.payload_fd();
    if (!active_ || !cmd || stream_fd < 0 || committed < 0 ||
        ctx_len <= 0 || ctx_len > ring_cap_) {
        return false;
    }
    const size_t noise_expected =
        (size_t)hidden_size_ * block_size_;
    if (noise_embed.size() != noise_expected) return false;
    if (payload_fd >= 0) {
        const size_t bytes = noise_embed.size() * sizeof(float);
        std::fprintf(cmd, "propose_pipe %d %d %zu\n", committed, ctx_len, bytes);
        std::fflush(cmd);
        if (!write_exact_fd(payload_fd, noise_embed.data(), bytes)) {
            std::fprintf(stderr, "draft-ipc propose payload write failed\n");
            return false;
        }
        int32_t status = -1;
        bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
        if (ok) {
            hidden_out.assign(noise_expected, 0.0f);
            ok = read_exact_fd(stream_fd, hidden_out.data(),
                               hidden_out.size() * sizeof(float));
        }
        if (!ok) {
            std::fprintf(stderr, "draft-ipc propose failed status=%d\n", status);
        }
        return ok;
    }
    const std::string path = process_.next_path("noise");
    if (!write_binary_file(path, noise_embed.data(), noise_embed.size() * sizeof(float))) {
        std::fprintf(stderr, "draft-ipc write noise failed: %s\n", path.c_str());
        return false;
    }
    std::fprintf(cmd, "propose %d %d %s\n", committed, ctx_len, path.c_str());
    std::fflush(cmd);
    int32_t status = -1;
    bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
    if (ok) {
        hidden_out.assign(noise_expected, 0.0f);
        ok = read_exact_fd(stream_fd, hidden_out.data(),
                           hidden_out.size() * sizeof(float));
    }
    std::remove(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "draft-ipc propose failed status=%d\n", status);
    }
    return ok;
#endif
}

bool DFlashDraftIpcClient::get_feature_range(int start_pos, int n_tokens,
                                             std::vector<float> & out) {
#if defined(_WIN32)
    (void)start_pos; (void)n_tokens; (void)out;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || ring_cap_ <= 0 ||
        start_pos < 0 || n_tokens <= 0 || n_tokens > ring_cap_) return false;
    const size_t count =
        (size_t)n_tokens * (size_t)n_target_layers_ * (size_t)hidden_size_;
    std::fprintf(cmd, "get_feature_range %d %d\n", start_pos, n_tokens);
    std::fflush(cmd);
    int32_t status = -1;
    bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
    if (ok) {
        out.assign(count, 0.0f);
        ok = read_exact_fd(stream_fd, out.data(), out.size() * sizeof(float));
    }
    if (!ok) {
        std::fprintf(stderr, "draft-ipc get_feature_range failed status=%d\n", status);
    }
    return ok;
#endif
}

bool DFlashDraftIpcClient::set_feature_range(int start_pos, int n_tokens,
                                             const std::vector<float> & data) {
#if defined(_WIN32)
    (void)start_pos; (void)n_tokens; (void)data;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || ring_cap_ <= 0 ||
        start_pos < 0 || n_tokens <= 0 || n_tokens > ring_cap_) return false;
    const size_t expected =
        (size_t)n_tokens * (size_t)n_target_layers_ * (size_t)hidden_size_;
    if (data.size() != expected) return false;
    const std::string path = process_.next_path("feature_range");
    if (!write_binary_file(path, data.data(), data.size() * sizeof(float))) {
        std::fprintf(stderr, "draft-ipc write feature range failed: %s\n", path.c_str());
        return false;
    }
    std::fprintf(cmd, "set_feature_range %d %d %s\n",
                 start_pos, n_tokens, path.c_str());
    std::fflush(cmd);
    int32_t status = -1;
    const bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
    std::remove(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "draft-ipc set_feature_range failed status=%d\n", status);
    }
    return ok;
#endif
}

void DFlashDraftIpcClient::close() {
    process_.close();
    active_ = false;
    ring_cap_ = 0;
}

// ── Remote draft feature copy helper ────────────────────────────────

bool copy_capture_slice_to_remote_draft(
        DFlashDraftIpcClient & remote,
        int capture_idx,
        const ggml_tensor * act_out,
        ggml_backend_t src_backend,
        int chunk_start,
        int start_pos,
        int n_tokens) {
    if (!remote.active() || !act_out || capture_idx < 0 || n_tokens <= 0) return true;
    const int hidden = remote.hidden_size();
    const size_t row_bytes = (size_t)hidden * sizeof(float);
    const size_t src_stride = act_out->nb[1];
    std::vector<float> host((size_t)n_tokens * hidden);
    ggml_backend_synchronize(src_backend);
    if (src_stride == row_bytes) {
        ggml_backend_tensor_get(act_out, host.data(),
                                (size_t)chunk_start * src_stride,
                                row_bytes * (size_t)n_tokens);
    } else {
        for (int i = 0; i < n_tokens; i++) {
            ggml_backend_tensor_get(act_out,
                                    host.data() + (size_t)i * hidden,
                                    (size_t)(chunk_start + i) * src_stride,
                                    row_bytes);
        }
    }
    return remote.send_feature_slice(capture_idx, start_pos, n_tokens, host);
}

} // namespace dflash::common
