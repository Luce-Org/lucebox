// dflash_draft_ipc.cpp — DFlashDraftIpcClient + remote feature copy.
//
// Target-agnostic portion of the DFlash draft IPC: parent-side client that
// spawns the daemon, sends commands, and the row-extraction helper that
// ships feature slices to it. The daemon implementation lives next to the
// owning target architecture (e.g. qwen35/draft_ipc_daemon.cpp) because it
// drives a target-specific draft graph builder.

#include "dflash_draft_ipc.h"
#include "io_utils.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace dflash27b {

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
    if (!init_work_dir(work_dir)) return false;

    int cmd_pipe[2] = {-1, -1};
    int stream_pipe[2] = {-1, -1};
    if (::pipe(cmd_pipe) != 0 || ::pipe(stream_pipe) != 0) {
        std::fprintf(stderr, "draft-ipc pipe failed: %s\n", std::strerror(errno));
        if (cmd_pipe[0] >= 0) ::close(cmd_pipe[0]);
        if (cmd_pipe[1] >= 0) ::close(cmd_pipe[1]);
        if (stream_pipe[0] >= 0) ::close(stream_pipe[0]);
        if (stream_pipe[1] >= 0) ::close(stream_pipe[1]);
        return false;
    }

    pid_ = ::fork();
    if (pid_ < 0) {
        std::fprintf(stderr, "draft-ipc fork failed: %s\n", std::strerror(errno));
        ::close(cmd_pipe[0]); ::close(cmd_pipe[1]);
        ::close(stream_pipe[0]); ::close(stream_pipe[1]);
        pid_ = -1;
        return false;
    }
    if (pid_ == 0) {
        ::dup2(cmd_pipe[0], STDIN_FILENO);
        ::close(cmd_pipe[0]);
        ::close(cmd_pipe[1]);
        ::close(stream_pipe[0]);

        const std::string cap_arg = "--ring-cap=" + std::to_string(ring_cap);
        const std::string gpu_arg = "--draft-gpu=" + std::to_string(std::max(0, draft_gpu));
        const std::string fd_arg = "--stream-fd=" + std::to_string(stream_pipe[1]);
        ::execl(bin.c_str(), bin.c_str(),
                "--draft-ipc-daemon", draft_path.c_str(),
                cap_arg.c_str(), gpu_arg.c_str(), fd_arg.c_str(),
                (char *)nullptr);
        std::fprintf(stderr, "draft-ipc exec failed: %s: %s\n",
                     bin.c_str(), std::strerror(errno));
        _exit(127);
    }

    ::close(cmd_pipe[0]);
    ::close(stream_pipe[1]);
    stream_fd_ = stream_pipe[0];
    cmd_ = ::fdopen(cmd_pipe[1], "w");
    if (!cmd_) {
        std::fprintf(stderr, "draft-ipc fdopen failed: %s\n", std::strerror(errno));
        ::close(cmd_pipe[1]);
        close();
        return false;
    }
    int32_t status = -1;
    if (!read_exact_fd(stream_fd_, &status, sizeof(status)) || status != 0) {
        std::fprintf(stderr, "draft-ipc daemon did not become ready (status=%d)\n", status);
        close();
        return false;
    }
    ring_cap_ = ring_cap;
    active_ = true;
    std::printf("[draft-ipc] ready bin=%s gpu=%d ring_cap=%d work_dir=%s\n",
                bin.c_str(), draft_gpu, ring_cap, work_dir_.c_str());
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
    if (!active_ || !cmd_ || n_tokens <= 0) return false;
    const size_t expected = (size_t)n_tokens * hidden_size_;
    if (slice.size() != expected) return false;
    const std::string path = next_path("feature");
    if (!write_binary_file(path, slice.data(), slice.size() * sizeof(float))) {
        std::fprintf(stderr, "draft-ipc write feature failed: %s\n", path.c_str());
        return false;
    }
    std::fprintf(cmd_, "feature_slice %d %d %d %s\n",
                 capture_idx, start_pos, n_tokens, path.c_str());
    std::fflush(cmd_);
    int32_t status = -1;
    const bool ok = read_exact_fd(stream_fd_, &status, sizeof(status)) && status == 0;
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
    if (!active_ || !cmd_ || ctx_len <= 0) return false;
    const size_t noise_expected =
        (size_t)hidden_size_ * block_size_;
    if (noise_embed.size() != noise_expected) return false;
    const std::string path = next_path("noise");
    if (!write_binary_file(path, noise_embed.data(), noise_embed.size() * sizeof(float))) {
        std::fprintf(stderr, "draft-ipc write noise failed: %s\n", path.c_str());
        return false;
    }
    std::fprintf(cmd_, "propose %d %d %s\n", committed, ctx_len, path.c_str());
    std::fflush(cmd_);
    int32_t status = -1;
    bool ok = read_exact_fd(stream_fd_, &status, sizeof(status)) && status == 0;
    if (ok) {
        hidden_out.assign(noise_expected, 0.0f);
        ok = read_exact_fd(stream_fd_, hidden_out.data(),
                           hidden_out.size() * sizeof(float));
    }
    std::remove(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "draft-ipc propose failed status=%d\n", status);
    }
    return ok;
#endif
}

void DFlashDraftIpcClient::close() {
#if !defined(_WIN32)
    if (cmd_) {
        std::fclose(cmd_);
        cmd_ = nullptr;
    }
    if (stream_fd_ >= 0) {
        ::close(stream_fd_);
        stream_fd_ = -1;
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
    ring_cap_ = 0;
}

#if !defined(_WIN32)
bool DFlashDraftIpcClient::init_work_dir(const std::string & requested) {
    if (!requested.empty()) {
        work_dir_ = requested;
        owns_work_dir_ = false;
        if (::mkdir(work_dir_.c_str(), 0700) != 0 && errno != EEXIST) {
            std::fprintf(stderr, "draft-ipc mkdir failed: %s: %s\n",
                         work_dir_.c_str(), std::strerror(errno));
            return false;
        }
        return true;
    }
    const char * tmp = std::getenv("TMPDIR");
    std::string templ = std::string(tmp && *tmp ? tmp : "/tmp") +
                        "/dflash-draft-ipc-XXXXXX";
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    char * dir = ::mkdtemp(buf.data());
    if (!dir) {
        std::fprintf(stderr, "draft-ipc mkdtemp failed: %s\n", std::strerror(errno));
        return false;
    }
    work_dir_ = dir;
    owns_work_dir_ = true;
    return true;
}

std::string DFlashDraftIpcClient::next_path(const char * prefix) {
    return work_dir_ + "/" + prefix + "_" + std::to_string(seq_++) + ".bin";
}
#endif

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

} // namespace dflash27b
