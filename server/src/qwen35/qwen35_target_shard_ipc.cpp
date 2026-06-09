// Qwen35 target-shard IPC client.

#include "qwen35_target_shard_ipc.h"

#include "common/io_utils.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <inttypes.h>
#include <limits>
#include <string>
#include <vector>

namespace dflash::common {

namespace {

bool checked_mul_size(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

BackendIpcPayloadTransport target_shard_ipc_transport_from_env() {
    const char * raw = std::getenv("DFLASH_TARGET_SHARD_IPC_TRANSPORT");
    if (!raw || !*raw) {
        return BackendIpcPayloadTransport::Stream;
    }
    BackendIpcPayloadTransport transport = BackendIpcPayloadTransport::Stream;
    if (!parse_backend_ipc_payload_transport(raw, transport)) {
        return BackendIpcPayloadTransport::Stream;
    }
    return transport;
}

bool write_snapshot_tensor_header_fd(int fd,
                                     const Qwen35TargetShardSnapshotTensor & t) {
    if (fd < 0 || t.shard < 0 || t.name.empty() ||
        t.name.size() >= (size_t)GGML_MAX_NAME ||
        t.data.size() > (size_t)std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    const int32_t shard = (int32_t)t.shard;
    const int32_t name_len = (int32_t)t.name.size();
    const uint32_t type = t.type;
    const uint64_t nbytes = (uint64_t)t.data.size();
    return write_exact_fd(fd, &shard, sizeof(shard)) &&
           write_exact_fd(fd, &name_len, sizeof(name_len)) &&
           write_exact_fd(fd, t.name.data(), (size_t)name_len) &&
           write_exact_fd(fd, &type, sizeof(type)) &&
           write_exact_fd(fd, t.ne, sizeof(t.ne)) &&
           write_exact_fd(fd, &nbytes, sizeof(nbytes));
}

bool read_snapshot_tensor_header_fd(int fd,
                                    Qwen35TargetShardSnapshotTensor & t) {
    int32_t shard = -1;
    int32_t name_len = 0;
    uint32_t type = 0;
    uint64_t nbytes = 0;
    if (!read_exact_fd(fd, &shard, sizeof(shard)) ||
        !read_exact_fd(fd, &name_len, sizeof(name_len)) ||
        shard < 0 || name_len <= 0 || name_len >= GGML_MAX_NAME) {
        return false;
    }
    std::string name((size_t)name_len, '\0');
    if (!read_exact_fd(fd, name.data(), (size_t)name_len) ||
        !read_exact_fd(fd, &type, sizeof(type)) ||
        !read_exact_fd(fd, t.ne, sizeof(t.ne)) ||
        !read_exact_fd(fd, &nbytes, sizeof(nbytes)) ||
        nbytes > (uint64_t)std::numeric_limits<size_t>::max()) {
        return false;
    }
    t.shard = shard;
    t.name = std::move(name);
    t.type = type;
    t.data.clear();
    t.data.resize((size_t)nbytes);
    return true;
}

size_t target_shard_required_shared_bytes(int hidden, int max_tokens) {
    if (hidden <= 0 || max_tokens <= 0) return 0;
    size_t elements = 0;
    size_t bytes = 0;
    if (!checked_mul_size((size_t)hidden, (size_t)max_tokens, elements) ||
        !checked_mul_size(elements, sizeof(float), bytes)) {
        return 0;
    }
    return bytes;
}

size_t target_shard_shared_bytes_from_env(size_t required_bytes) {
    const char * raw = std::getenv("DFLASH_TARGET_SHARD_IPC_SHARED_BYTES");
    if (!raw || !*raw) {
        return required_bytes;
    }
    if (raw[0] == '-') {
        return required_bytes;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' ||
        parsed > (unsigned long long)std::numeric_limits<size_t>::max()) {
        return required_bytes;
    }
    return std::max((size_t)parsed, required_bytes);
}

}  // namespace

bool copy_activation_to_host(const ggml_tensor * act,
                             ggml_backend_t src_backend,
                             int token_offset,
                             int n_tokens,
                             int hidden,
                             std::vector<float> & out) {
    if (!act || !src_backend || token_offset < 0 || n_tokens <= 0 || hidden <= 0) {
        return false;
    }
    const size_t row_bytes = (size_t)hidden * sizeof(float);
    const size_t src_stride = act->nb[1];
    out.assign((size_t)n_tokens * (size_t)hidden, 0.0f);
    ggml_backend_synchronize(src_backend);
    if (src_stride == row_bytes) {
        ggml_backend_tensor_get(act, out.data(),
                                (size_t)token_offset * src_stride,
                                row_bytes * (size_t)n_tokens);
    } else {
        for (int i = 0; i < n_tokens; i++) {
            ggml_backend_tensor_get(act,
                                    out.data() + (size_t)i * (size_t)hidden,
                                    (size_t)(token_offset + i) * src_stride,
                                    row_bytes);
        }
    }
    return true;
}

bool Qwen35TargetShardIpcClient::start(
        const std::string & bin,
        const std::string & target_path,
        const std::vector<int> & gpus,
        const std::vector<int> & layer_begins,
        const std::vector<int> & layer_ends,
        int max_ctx,
        int max_verify_tokens,
        int kq_stride_pad,
        int fa_window,
        int hidden,
        int vocab,
        int max_tokens,
        const std::string & work_dir,
        bool enable_dflash) {
#if defined(_WIN32)
    (void)bin; (void)target_path; (void)gpus; (void)layer_begins; (void)layer_ends;
    (void)max_ctx; (void)max_verify_tokens; (void)kq_stride_pad; (void)fa_window;
    (void)hidden; (void)vocab; (void)max_tokens; (void)work_dir; (void)enable_dflash;
    std::fprintf(stderr, "Qwen35 target shard IPC is only implemented on POSIX hosts\n");
    return false;
#else
    close();
    if (bin.empty() || target_path.empty() || gpus.empty() ||
        gpus.size() != layer_begins.size() || gpus.size() != layer_ends.size() ||
        max_ctx <= 0 || hidden <= 0 ||
        vocab <= 0 || max_tokens <= 0) {
        return false;
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (gpus[i] < 0 || layer_begins[i] < 0 ||
            layer_ends[i] <= layer_begins[i]) {
            return false;
        }
        if (i > 0 && layer_begins[i] != layer_ends[i - 1]) {
            return false;
        }
    }

    BackendIpcLaunchConfig launch;
    launch.bin = bin;
    launch.mode = BackendIpcMode::Qwen35TargetShard;
    launch.payload_path = target_path;
    launch.work_dir = work_dir;
    launch.payload_transport = target_shard_ipc_transport_from_env();
    launch.shared_payload_bytes = target_shard_shared_bytes_from_env(
        target_shard_required_shared_bytes(hidden, max_tokens));
    std::string gpu_list;
    std::string layer_begin_list;
    std::string layer_end_list;
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (i > 0) {
            gpu_list += ",";
            layer_begin_list += ",";
            layer_end_list += ",";
        }
        gpu_list += std::to_string(gpus[i]);
        layer_begin_list += std::to_string(layer_begins[i]);
        layer_end_list += std::to_string(layer_ends[i]);
    }
    launch.args.push_back("--target-gpus=" + gpu_list);
    launch.args.push_back("--layer-begins=" + layer_begin_list);
    launch.args.push_back("--layer-ends=" + layer_end_list);
    launch.args.push_back("--max-ctx=" + std::to_string(max_ctx));
    launch.args.push_back("--max-verify-tokens=" + std::to_string(max_verify_tokens));
    launch.args.push_back("--kq-stride-pad=" + std::to_string(kq_stride_pad));
    launch.args.push_back("--fa-window=" + std::to_string(fa_window));
    if (enable_dflash) {
        launch.args.push_back("--enable-dflash");
    }
    if (!process_.start(launch)) {
        std::fprintf(stderr, "qwen35-target-shard backend process start failed\n");
        return false;
    }

    hidden_ = hidden;
    vocab_ = vocab;
    active_ = true;
    std::printf("[qwen35-target-shard-ipc] ready bin=%s shards=%zu layers=[%d,%d) work_dir=%s\n",
                bin.c_str(), gpus.size(), layer_begins.front(), layer_ends.back(),
                process_.work_dir().c_str());
    return true;
#endif
}

bool Qwen35TargetShardIpcClient::forward(
        int base_pos,
        int n_tokens,
        const std::vector<float> & boundary_activation,
        bool need_logits,
        int & last_tok,
        std::vector<int32_t> * argmax_out,
        std::vector<float> * logits_out,
        std::vector<Qwen35TargetCaptureSlice> * captures_out) {
#if defined(_WIN32)
    (void)base_pos; (void)n_tokens; (void)boundary_activation; (void)need_logits;
    (void)last_tok; (void)argmax_out; (void)logits_out; (void)captures_out;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    const int payload_fd = process_.payload_fd();
    if (!active_ || !cmd || stream_fd < 0 || base_pos < 0 ||
        n_tokens <= 0 || hidden_ <= 0 || vocab_ <= 0) {
        return false;
    }
    const size_t expected = (size_t)n_tokens * (size_t)hidden_;
    if (boundary_activation.size() != expected) return false;
    const size_t bytes = boundary_activation.size() * sizeof(float);
    const int want_argmax = argmax_out ? 1 : 0;
    const int want_logits = need_logits ? 1 : 0;
    const int want_captures = captures_out ? 1 : 0;
    if (captures_out) captures_out->clear();

    if (process_.resolved_payload_transport() == BackendIpcPayloadTransport::Shared) {
        uint64_t seq = 0;
        if (!process_.write_shared_payload(boundary_activation.data(), bytes, seq)) {
            std::fprintf(stderr,
                         "qwen35-target-shard shared payload too large bytes=%zu capacity=%zu\n",
                         bytes, process_.shared_payload_capacity());
            return false;
        }
        std::fprintf(cmd, "forward_shared %d %d %d %d %zu %" PRIu64 " %d\n",
                     base_pos, n_tokens, want_argmax, want_logits, bytes, seq,
                     want_captures);
        std::fflush(cmd);
    } else if (payload_fd >= 0) {
        std::fprintf(cmd, "forward_pipe %d %d %d %d %zu %d\n",
                     base_pos, n_tokens, want_argmax, want_logits, bytes,
                     want_captures);
        std::fflush(cmd);
        if (!write_exact_fd(payload_fd, boundary_activation.data(), bytes)) {
            std::fprintf(stderr, "qwen35-target-shard payload write failed\n");
            return false;
        }
    } else {
        return false;
    }

    int32_t status = -1;
    int32_t remote_last_tok = -1;
    if (!read_exact_fd(stream_fd, &status, sizeof(status)) || status != 0 ||
        !read_exact_fd(stream_fd, &remote_last_tok, sizeof(remote_last_tok))) {
        std::fprintf(stderr, "qwen35-target-shard forward failed status=%d\n", status);
        return false;
    }
    last_tok = remote_last_tok;

    if (argmax_out) {
        argmax_out->assign((size_t)n_tokens, 0);
        if (!read_exact_fd(stream_fd, argmax_out->data(),
                           sizeof(int32_t) * (size_t)n_tokens)) {
            return false;
        }
    }
    if (need_logits) {
        if (!logits_out) return false;
        const int logits_tokens = argmax_out ? n_tokens : 1;
        logits_out->assign((size_t)vocab_ * (size_t)logits_tokens, 0.0f);
        if (!read_exact_fd(stream_fd, logits_out->data(),
                           sizeof(float) * logits_out->size())) {
            return false;
        }
    }
    if (captures_out) {
        int32_t n_captures = 0;
        if (!read_exact_fd(stream_fd, &n_captures, sizeof(n_captures)) ||
            n_captures < 0) {
            return false;
        }
        captures_out->resize((size_t)n_captures);
        for (int32_t i = 0; i < n_captures; ++i) {
            int32_t capture_idx = -1;
            int32_t capture_start_pos = 0;
            int32_t capture_n_tokens = 0;
            int32_t capture_elems = 0;
            if (!read_exact_fd(stream_fd, &capture_idx, sizeof(capture_idx)) ||
                !read_exact_fd(stream_fd, &capture_start_pos, sizeof(capture_start_pos)) ||
                !read_exact_fd(stream_fd, &capture_n_tokens, sizeof(capture_n_tokens)) ||
                !read_exact_fd(stream_fd, &capture_elems, sizeof(capture_elems)) ||
                capture_idx < 0 || capture_start_pos < 0 || capture_n_tokens <= 0 ||
                capture_elems != capture_n_tokens * hidden_) {
                return false;
            }
            auto & capture = (*captures_out)[(size_t)i];
            capture.capture_idx = capture_idx;
            capture.start_pos = capture_start_pos;
            capture.n_tokens = capture_n_tokens;
            capture.data.assign((size_t)capture_elems, 0.0f);
            if (!read_exact_fd(stream_fd, capture.data.data(),
                               sizeof(float) * capture.data.size())) {
                return false;
            }
        }
    }
    return true;
#endif
}

bool Qwen35TargetShardIpcClient::project_hidden_to_tokens(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out) {
#if defined(_WIN32)
    (void)hidden; (void)n_tokens; (void)tokens_out;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    const int payload_fd = process_.payload_fd();
    if (!active_ || !cmd || !hidden || stream_fd < 0 || payload_fd < 0 ||
        n_tokens <= 0 || hidden_ <= 0) {
        return false;
    }
    const size_t elems = (size_t)n_tokens * (size_t)hidden_;
    const size_t bytes = elems * sizeof(float);
    std::fprintf(cmd, "project_pipe %d %zu\n", n_tokens, bytes);
    std::fflush(cmd);
    if (!write_exact_fd(payload_fd, hidden, bytes)) {
        return false;
    }
    int32_t status = -1;
    if (!read_exact_fd(stream_fd, &status, sizeof(status)) || status != 0) {
        return false;
    }
    tokens_out.assign((size_t)n_tokens, 0);
    return read_exact_fd(stream_fd, tokens_out.data(),
                         sizeof(int32_t) * (size_t)n_tokens);
#endif
}

bool Qwen35TargetShardIpcClient::snapshot_kv() {
#if defined(_WIN32)
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0) return false;
    std::fprintf(cmd, "snapshot\n");
    std::fflush(cmd);
    int32_t status = -1;
    return read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
#endif
}

bool Qwen35TargetShardIpcClient::restore_kv() {
#if defined(_WIN32)
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0) return false;
    std::fprintf(cmd, "restore\n");
    std::fflush(cmd);
    int32_t status = -1;
    return read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
#endif
}

bool Qwen35TargetShardIpcClient::reset_request_state() {
#if defined(_WIN32)
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0) return false;
    std::fprintf(cmd, "reset_request_state\n");
    std::fflush(cmd);
    int32_t status = -1;
    return read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
#endif
}

bool Qwen35TargetShardIpcClient::snapshot_save(int slot) {
#if defined(_WIN32)
    (void)slot;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || slot < 0) return false;
    std::fprintf(cmd, "prefix_snapshot_save %d\n", slot);
    std::fflush(cmd);
    int32_t status = -1;
    return read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
#endif
}

void Qwen35TargetShardIpcClient::snapshot_free(int slot) {
#if defined(_WIN32)
    (void)slot;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || slot < 0) return;
    std::fprintf(cmd, "prefix_snapshot_free %d\n", slot);
    std::fflush(cmd);
    int32_t status = -1;
    (void)read_exact_fd(stream_fd, &status, sizeof(status));
#endif
}

bool Qwen35TargetShardIpcClient::snapshot_restore(int slot) {
#if defined(_WIN32)
    (void)slot;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || slot < 0) return false;
    std::fprintf(cmd, "prefix_snapshot_restore %d\n", slot);
    std::fflush(cmd);
    int32_t status = -1;
    return read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
#endif
}

bool Qwen35TargetShardIpcClient::snapshot_export(
        int slot,
        Qwen35TargetShardSnapshotData & out) {
#if defined(_WIN32)
    (void)slot; (void)out;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || slot < 0) return false;
    out = Qwen35TargetShardSnapshotData{};
    std::fprintf(cmd, "prefix_snapshot_export %d\n", slot);
    std::fflush(cmd);

    int32_t status = -1;
    int32_t cur_pos = 0;
    int32_t last_tok = -1;
    int32_t shard_count = 0;
    int32_t n_tensors = 0;
    int32_t logits_count = 0;
    if (!read_exact_fd(stream_fd, &status, sizeof(status)) || status != 0 ||
        !read_exact_fd(stream_fd, &cur_pos, sizeof(cur_pos)) ||
        !read_exact_fd(stream_fd, &last_tok, sizeof(last_tok)) ||
        !read_exact_fd(stream_fd, &shard_count, sizeof(shard_count)) ||
        !read_exact_fd(stream_fd, &n_tensors, sizeof(n_tensors)) ||
        !read_exact_fd(stream_fd, &logits_count, sizeof(logits_count)) ||
        cur_pos <= 0 || shard_count <= 0 || n_tensors <= 0 ||
        logits_count <= 0) {
        return false;
    }

    out.shard_count = shard_count;
    out.cur_pos = cur_pos;
    out.last_tok = last_tok;
    out.tensors.resize((size_t)n_tensors);
    for (int32_t i = 0; i < n_tensors; ++i) {
        if (!read_snapshot_tensor_header_fd(stream_fd, out.tensors[(size_t)i]) ||
            out.tensors[(size_t)i].shard >= shard_count) {
            return false;
        }
    }
    for (auto & t : out.tensors) {
        if (!t.data.empty() &&
            !read_exact_fd(stream_fd, t.data.data(), t.data.size())) {
            return false;
        }
    }
    out.logits.assign((size_t)logits_count, 0.0f);
    return read_exact_fd(stream_fd, out.logits.data(),
                         sizeof(float) * out.logits.size());
#endif
}

bool Qwen35TargetShardIpcClient::snapshot_import(
        int slot,
        const Qwen35TargetShardSnapshotData & data) {
#if defined(_WIN32)
    (void)slot; (void)data;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    const int payload_fd = process_.payload_fd();
    if (!active_ || !cmd || stream_fd < 0 || payload_fd < 0 || slot < 0 ||
        data.shard_count <= 0 || data.cur_pos <= 0 ||
        data.tensors.empty() || data.logits.empty() ||
        data.tensors.size() > (size_t)std::numeric_limits<int32_t>::max() ||
        data.logits.size() > (size_t)std::numeric_limits<int32_t>::max()) {
        return false;
    }
    const int32_t n_tensors = (int32_t)data.tensors.size();
    const int32_t logits_count = (int32_t)data.logits.size();
    for (const auto & t : data.tensors) {
        if (t.shard < 0 || t.shard >= data.shard_count ||
            t.name.empty() || t.name.size() >= (size_t)GGML_MAX_NAME ||
            t.data.size() > (size_t)std::numeric_limits<uint64_t>::max()) {
            return false;
        }
    }
    std::fprintf(cmd, "prefix_snapshot_import %d %d %d %d %d %d\n",
                 slot, data.cur_pos, data.last_tok, data.shard_count,
                 n_tensors, logits_count);
    std::fflush(cmd);

    int32_t status = -1;
    if (!read_exact_fd(stream_fd, &status, sizeof(status)) || status != 0) {
        return false;
    }

    for (const auto & t : data.tensors) {
        if (!write_snapshot_tensor_header_fd(payload_fd, t)) {
            return false;
        }
    }
    if (!read_exact_fd(stream_fd, &status, sizeof(status)) || status != 0) {
        return false;
    }

    for (const auto & t : data.tensors) {
        if (!t.data.empty() &&
            !write_exact_fd(payload_fd, t.data.data(), t.data.size())) {
            return false;
        }
    }
    if (!write_exact_fd(payload_fd, data.logits.data(),
                        sizeof(float) * data.logits.size())) {
        return false;
    }
    return read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
#endif
}

void Qwen35TargetShardIpcClient::close() {
    process_.close();
    active_ = false;
    hidden_ = 0;
    vocab_ = 0;
}

}  // namespace dflash::common
