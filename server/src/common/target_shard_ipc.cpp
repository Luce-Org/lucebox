// Generic target-shard IPC session for mixed-backend layer split.

#include "target_shard_ipc.h"

#include "common/io_utils.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace dflash::common {

namespace {

bool checked_mul_size(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}

size_t required_shared_bytes(int hidden, int max_tokens) {
    if (hidden <= 0 || max_tokens <= 0) return 0;
    size_t elems = 0;
    size_t bytes = 0;
    if (!checked_mul_size((size_t)hidden, (size_t)max_tokens, elems) ||
        !checked_mul_size(elems, sizeof(float), bytes)) {
        return 0;
    }
    return bytes;
}

size_t shared_bytes_from_env(size_t required_bytes) {
    const char * raw = std::getenv("DFLASH_TARGET_SHARD_IPC_SHARED_BYTES");
    if (!raw || !*raw) return required_bytes;
    if (raw[0] == '-') return required_bytes;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' ||
        parsed > (unsigned long long)std::numeric_limits<size_t>::max()) {
        return required_bytes;
    }
    return std::max((size_t)parsed, required_bytes);
}

BackendIpcPayloadTransport target_shard_ipc_transport_from_env() {
    const char * raw = std::getenv("DFLASH_TARGET_SHARD_IPC_TRANSPORT");
    if (!raw || !*raw) return BackendIpcPayloadTransport::Stream;
    BackendIpcPayloadTransport transport = BackendIpcPayloadTransport::Stream;
    if (!parse_backend_ipc_payload_transport(raw, transport)) {
        return BackendIpcPayloadTransport::Stream;
    }
    return transport;
}

bool write_int32_vec_fd(int fd, const std::vector<int32_t> & values) {
    return values.empty() || write_exact_fd(fd, values.data(),
                                           sizeof(int32_t) * values.size());
}

bool copy_activation_row_to_host(const ggml_tensor * act,
                                 size_t offset,
                                 int hidden,
                                 float * dst) {
    if (!act || !dst || hidden <= 0) return false;
    if (act->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(act, dst, offset, (size_t)hidden * sizeof(float));
        return true;
    }
    if (act->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t)hidden);
        ggml_backend_tensor_get(act, tmp.data(), offset,
                                tmp.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), dst, hidden);
        return true;
    }
    if (act->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> tmp((size_t)hidden);
        ggml_backend_tensor_get(act, tmp.data(), offset,
                                tmp.size() * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row(tmp.data(), dst, hidden);
        return true;
    }
    return false;
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
    const size_t row_bytes = ggml_row_size(act->type, hidden);
    if (row_bytes == 0) return false;
    const size_t src_stride = act->nb[1];
    out.assign((size_t)n_tokens * (size_t)hidden, 0.0f);
    ggml_backend_synchronize(src_backend);
    for (int i = 0; i < n_tokens; ++i) {
        if (!copy_activation_row_to_host(
                act, (size_t)(token_offset + i) * src_stride,
                hidden, out.data() + (size_t)i * (size_t)hidden)) {
            return false;
        }
    }
    return true;
}

bool TargetShardIpcSession::start(const TargetShardIpcLaunchConfig & cfg) {
#if defined(_WIN32)
    (void)cfg;
    std::fprintf(stderr, "target shard IPC is only implemented on POSIX hosts\n");
    return false;
#else
    close();
    if (cfg.mode != BackendIpcMode::Qwen35TargetShard &&
        cfg.mode != BackendIpcMode::Gemma4TargetShard &&
        cfg.mode != BackendIpcMode::LagunaTargetShard &&
        cfg.mode != BackendIpcMode::DeepSeek4TargetShard) {
        std::fprintf(stderr, "target shard IPC requires an explicit target-shard mode\n");
        return false;
    }
    if (cfg.bin.empty() || cfg.target_path.empty() || cfg.gpus.empty() ||
        cfg.gpus.size() != cfg.layer_begins.size() ||
        cfg.gpus.size() != cfg.layer_ends.size() ||
        cfg.max_ctx <= 0 || cfg.hidden <= 0 || cfg.vocab <= 0 ||
        cfg.max_tokens <= 0) {
        return false;
    }
    for (size_t i = 0; i < cfg.gpus.size(); ++i) {
        if (cfg.gpus[i] < 0 || cfg.layer_begins[i] < 0 ||
            cfg.layer_ends[i] <= cfg.layer_begins[i]) {
            return false;
        }
        if (i > 0 && cfg.layer_begins[i] != cfg.layer_ends[i - 1]) {
            return false;
        }
    }

    BackendIpcLaunchConfig launch;
    launch.bin = cfg.bin;
    launch.mode = cfg.mode;
    launch.payload_path = cfg.target_path;
    launch.work_dir = cfg.work_dir;
    launch.payload_transport = target_shard_ipc_transport_from_env();
    launch.shared_payload_bytes = shared_bytes_from_env(
        required_shared_bytes(cfg.hidden, cfg.max_tokens));
    std::string gpu_list;
    std::string layer_begin_list;
    std::string layer_end_list;
    for (size_t i = 0; i < cfg.gpus.size(); ++i) {
        if (i > 0) {
            gpu_list += ",";
            layer_begin_list += ",";
            layer_end_list += ",";
        }
        gpu_list += std::to_string(cfg.gpus[i]);
        layer_begin_list += std::to_string(cfg.layer_begins[i]);
        layer_end_list += std::to_string(cfg.layer_ends[i]);
    }
    launch.args.push_back("--target-gpus=" + gpu_list);
    launch.args.push_back("--layer-begins=" + layer_begin_list);
    launch.args.push_back("--layer-ends=" + layer_end_list);
    launch.args.push_back("--max-ctx=" + std::to_string(cfg.max_ctx));
    launch.args.push_back("--hidden=" + std::to_string(cfg.hidden));
    launch.args.push_back("--vocab=" + std::to_string(cfg.vocab));
    launch.args.push_back("--max-tokens=" + std::to_string(cfg.max_tokens));
    launch.args.push_back("--fa-window=" + std::to_string(cfg.fa_window));
    if (cfg.kvflash_pool_tokens > 0) {
        launch.args.push_back("--kvflash-pool=" +
                              std::to_string(cfg.kvflash_pool_tokens));
    }
    for (const std::string & arg : cfg.model_args) {
        if (!arg.empty()) launch.args.push_back(arg);
    }
    if (!process_.start(launch)) {
        std::fprintf(stderr, "target shard backend process start failed\n");
        return false;
    }

    mode_ = cfg.mode;
    hidden_ = cfg.hidden;
    vocab_ = cfg.vocab;
    active_ = true;
    std::printf("[target-shard-ipc] ready mode=%s bin=%s shards=%zu layers=[%d,%d) work_dir=%s\n",
                backend_ipc_mode_name(cfg.mode), cfg.bin.c_str(), cfg.gpus.size(),
                cfg.layer_begins.front(), cfg.layer_ends.back(),
                process_.work_dir().c_str());
    return true;
#endif
}

bool TargetShardIpcSession::forward(
        const TargetShardForwardRequest & req,
        TargetShardForwardResponse & resp) {
#if defined(_WIN32)
    (void)req; (void)resp;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    const int payload_fd = process_.payload_fd();
    const int base_pos = req.base_pos;
    const int n_tokens = req.n_tokens;
    const std::vector<int32_t> * token_ids = req.token_ids;
    const int ubatch = req.ubatch;
    if (!active_ || !cmd || stream_fd < 0 || base_pos < 0 || n_tokens <= 0 ||
        hidden_ <= 0 || vocab_ <= 0 || !req.boundary_activation) {
        return false;
    }
    const std::vector<float> & boundary_activation = *req.boundary_activation;
    const size_t expected = (size_t)n_tokens * (size_t)hidden_;
    if (boundary_activation.size() != expected) return false;
    if (token_ids && (int)token_ids->size() != n_tokens) return false;
    if (req.want_argmax != (resp.argmax_out != nullptr)) return false;
    if (req.want_logits != (resp.logits_out != nullptr)) return false;
    const size_t bytes = boundary_activation.size() * sizeof(float);
    const int want_argmax = req.want_argmax ? 1 : 0;
    const int want_logits = req.want_logits ? 1 : 0;
    const int forward_ubatch = ubatch > 0 ? ubatch : n_tokens;
    std::vector<int32_t> empty_ids;
    const std::vector<int32_t> * ids = token_ids ? token_ids : &empty_ids;

    if (process_.resolved_payload_transport() == BackendIpcPayloadTransport::Shared) {
        uint64_t seq = 0;
        if (!process_.write_shared_payload(boundary_activation.data(), bytes, seq)) {
            std::fprintf(stderr,
                         "target-shard shared payload too large bytes=%zu capacity=%zu\n",
                         bytes, process_.shared_payload_capacity());
            return false;
        }
        std::fprintf(cmd, "forward_shared %d %d %d %d %zu %" PRIu64 " %d %d %d\n",
                     base_pos, n_tokens, want_argmax, want_logits, bytes, seq,
                     token_ids ? 1 : 0, forward_ubatch, (int)ids->size());
        std::fflush(cmd);
    } else if (payload_fd >= 0) {
        std::fprintf(cmd, "forward_pipe %d %d %d %d %zu %d %d %d\n",
                     base_pos, n_tokens, want_argmax, want_logits, bytes,
                     token_ids ? 1 : 0, forward_ubatch, (int)ids->size());
        std::fflush(cmd);
        if (!write_exact_fd(payload_fd, boundary_activation.data(), bytes)) {
            std::fprintf(stderr, "target-shard payload write failed\n");
            return false;
        }
    } else {
        return false;
    }
    if (token_ids && !write_int32_vec_fd(payload_fd >= 0 ? payload_fd : stream_fd, *token_ids)) {
        return false;
    }

    int32_t status = -1;
    int32_t remote_last_tok = -1;
    if (!read_exact_fd(stream_fd, &status, sizeof(status)) || status != 0 ||
        !read_exact_fd(stream_fd, &remote_last_tok, sizeof(remote_last_tok))) {
        std::fprintf(stderr, "target-shard forward failed status=%d\n", status);
        return false;
    }
    resp.last_tok = remote_last_tok;

    if (req.want_argmax) {
        resp.argmax_out->assign((size_t)n_tokens, 0);
        if (!read_exact_fd(stream_fd, resp.argmax_out->data(),
                           sizeof(int32_t) * (size_t)n_tokens)) {
            return false;
        }
    }
    if (req.want_logits) {
        const int logits_tokens = req.want_argmax ? n_tokens : 1;
        resp.logits_out->assign((size_t)vocab_ * (size_t)logits_tokens, 0.0f);
        if (!read_exact_fd(stream_fd, resp.logits_out->data(),
                           sizeof(float) * resp.logits_out->size())) {
            return false;
        }
    }
    return true;
#endif
}

bool TargetShardIpcSession::reset_request_state() {
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

bool TargetShardIpcSession::kvflash_sync_identity(int committed) {
#if defined(_WIN32)
    (void)committed;
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || committed < 0) return false;
    std::fprintf(cmd, "kvflash_sync_identity %d\n", committed);
    std::fflush(cmd);
    int32_t status = -1;
    return read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
#endif
}

bool TargetShardIpcSession::snapshot_save(int slot) {
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

void TargetShardIpcSession::snapshot_free(int slot) {
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

bool TargetShardIpcSession::snapshot_restore(int slot) {
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

void TargetShardIpcSession::close() {
    process_.close();
    active_ = false;
    hidden_ = 0;
    vocab_ = 0;
    mode_ = BackendIpcMode::Invalid;
}

}  // namespace dflash::common
