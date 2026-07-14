// Standalone backend IPC daemon entry point.

#include "backend_ipc.h"
#include "moe_expert_compute.h"

#include "dflash_draft_ipc.h"
#include "deepseek4/deepseek4_layer_split_adapter.h"
#include "gemma4/gemma4_layer_split_adapter.h"
#include "laguna/laguna_layer_split_adapter.h"
#include "pflash_drafter_ipc.h"
#include "qwen35_target_shard_ipc.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

bool parse_int_list(const char * raw, std::vector<int> & out) {
    out.clear();
    if (!raw || !*raw) return false;
    std::string text(raw);
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t end = text.find(',', begin);
        const std::string item = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty()) return false;
        char * parse_end = nullptr;
        const long value = std::strtol(item.c_str(), &parse_end, 10);
        if (parse_end == item.c_str() || *parse_end != '\0') return false;
        out.push_back((int)value);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return !out.empty();
}

bool parse_nonnegative_int(const char * text, int & out) {
    if (!text || !*text) return false;
    const char * p = text;
    while (std::isspace((unsigned char)*p)) ++p;
    if (*p == '-') return false;
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (errno == ERANGE || end == p || *end != '\0' ||
        value < 0 || value > std::numeric_limits<int>::max()) {
        return false;
    }
    out = (int)value;
    return true;
}

bool parse_size_arg(const char * text, size_t & out) {
    if (!text || !*text) return false;
    const char * p = text;
    while (std::isspace((unsigned char)*p)) ++p;
    if (*p == '-') return false;
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = std::strtoull(p, &end, 10);
    if (errno == ERANGE || end == p || *end != '\0' ||
        value > (unsigned long long)std::numeric_limits<size_t>::max()) {
        return false;
    }
    out = (size_t)value;
    return true;
}

bool require_value(int & i, int argc, char ** argv, const char * opt, const char *& out) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "[backend-ipc-daemon] missing value for %s\n", opt);
        return false;
    }
    out = argv[++i];
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    BackendIpcMode mode = BackendIpcMode::DFlashDraft;
    const char * payload_path = nullptr;
    int arg_begin = 0;

    if (argc >= 3 && std::strncmp(argv[1], "--backend-ipc-mode=", 19) == 0) {
        std::string mode_name(argv[1] + 19);
        if (!parse_backend_ipc_mode(mode_name, mode)) {
            std::fprintf(stderr, "[backend-ipc-daemon] unknown mode: %s\n",
                         mode_name.c_str());
            return 2;
        }
        payload_path = argv[2];
        arg_begin = 3;
    } else if (argc >= 3 && std::strcmp(argv[1], "--backend-ipc-mode") == 0) {
        if (!parse_backend_ipc_mode(argv[2], mode) || argc < 4) {
            std::fprintf(stderr, "[backend-ipc-daemon] bad --backend-ipc-mode\n");
            return 2;
        }
        payload_path = argv[3];
        arg_begin = 4;
    } else {
        std::fprintf(stderr,
            "usage: %s --backend-ipc-mode=dflash-draft <draft.safetensors|draft.gguf> "
            "--ring-cap=N --stream-fd=FD [--payload-fd=FD] "
            "[--shared-payload-fd=FD --shared-payload-bytes=N] [--draft-gpu=N]\n"
            "   or: %s --backend-ipc-mode=pflash-compress <drafter.gguf> "
            "--stream-fd=FD [--draft-gpu=N]\n"
            "   or: %s --backend-ipc-mode=qwen35-target-shard <target.gguf> "
            "--stream-fd=FD --target-gpu=N --layer-begin=N --layer-end=N "
            "--max-ctx=N [--hidden=N --vocab=N --max-tokens=N]\n"
            "   or: %s --backend-ipc-mode=gemma4-target-shard <target.gguf> "
            "--stream-fd=FD --target-gpus=N[,N...] --layer-begins=N[,N...] "
            "--layer-ends=N[,N...] --max-ctx=N "
            "[--hidden=N --vocab=N --max-tokens=N]\n"
            "   or: %s --backend-ipc-mode=laguna-target-shard <target.gguf> "
            "--stream-fd=FD --target-gpus=N[,N...] --layer-begins=N[,N...] "
            "--layer-ends=N[,N...] --max-ctx=N "
            "[--hidden=N --vocab=N --max-tokens=N]\n"
            "   or: %s --backend-ipc-mode=deepseek4-target-shard <target.gguf> "
            "--stream-fd=FD --target-gpus=N[,N...] --layer-begins=N[,N...] "
            "--layer-ends=N[,N...] --max-ctx=N\n"
            "   or: %s --backend-ipc-mode=moe-expert-compute <target.gguf> "
            "--stream-fd=FD --target-gpu=N --placement=PATH\n",
            argv[0],
            argv[0],
            argv[0],
            argv[0],
            argv[0],
            argv[0]);
        return 2;
    }

    int ring_cap = 4096;
    int draft_gpu = 0;
    int target_gpu = 0;
    std::vector<int> target_gpus;
    std::vector<int> layer_begins;
    std::vector<int> layer_ends;
    int layer_begin = -1;
    int layer_end = -1;
    int max_ctx = 8192;
    int max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    int target_hidden = 0;
    int target_vocab = 0;
    int target_max_tokens = 0;
    int kq_stride_pad = 32;
    int fa_window = 0;
    int payload_fd = -1;
    int stream_fd = -1;
    int shared_payload_fd = -1;
    size_t shared_payload_bytes = 0;
    bool enable_dflash = false;
    int kvflash_pool_tokens = 0;
    const char * placement_path = nullptr;
    bool ds4_fixed_slot_graphs = false;
    for (int i = arg_begin; i < argc; i++) {
        if (std::strncmp(argv[i], "--ring-cap=", 11) == 0) {
            if (!parse_nonnegative_int(argv[i] + 11, ring_cap)) return 2;
        } else if (std::strcmp(argv[i], "--ring-cap") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--ring-cap", value) ||
                !parse_nonnegative_int(value, ring_cap)) return 2;
        } else if (std::strncmp(argv[i], "--draft-gpu=", 12) == 0) {
            if (!parse_nonnegative_int(argv[i] + 12, draft_gpu)) return 2;
        } else if (std::strcmp(argv[i], "--draft-gpu") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--draft-gpu", value) ||
                !parse_nonnegative_int(value, draft_gpu)) return 2;
        } else if (std::strncmp(argv[i], "--target-gpu=", 13) == 0) {
            target_gpu = std::max(0, std::atoi(argv[i] + 13));
        } else if (std::strcmp(argv[i], "--target-gpu") == 0) {
            if (i + 1 < argc) target_gpu = std::max(0, std::atoi(argv[++i]));
        } else if (std::strncmp(argv[i], "--target-gpus=", 14) == 0) {
            if (!parse_int_list(argv[i] + 14, target_gpus)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --target-gpus\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--target-gpus") == 0) {
            if (i + 1 >= argc || !parse_int_list(argv[++i], target_gpus)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --target-gpus\n");
                return 2;
            }
        } else if (std::strncmp(argv[i], "--layer-begin=", 14) == 0) {
            layer_begin = std::atoi(argv[i] + 14);
        } else if (std::strcmp(argv[i], "--layer-begin") == 0) {
            if (i + 1 < argc) layer_begin = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--layer-begins=", 15) == 0) {
            if (!parse_int_list(argv[i] + 15, layer_begins)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --layer-begins\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--layer-begins") == 0) {
            if (i + 1 >= argc || !parse_int_list(argv[++i], layer_begins)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --layer-begins\n");
                return 2;
            }
        } else if (std::strncmp(argv[i], "--layer-end=", 12) == 0) {
            layer_end = std::atoi(argv[i] + 12);
        } else if (std::strcmp(argv[i], "--layer-end") == 0) {
            if (i + 1 < argc) layer_end = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--layer-ends=", 13) == 0) {
            if (!parse_int_list(argv[i] + 13, layer_ends)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --layer-ends\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--layer-ends") == 0) {
            if (i + 1 >= argc || !parse_int_list(argv[++i], layer_ends)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --layer-ends\n");
                return 2;
            }
        } else if (std::strncmp(argv[i], "--max-ctx=", 10) == 0) {
            max_ctx = std::atoi(argv[i] + 10);
        } else if (std::strcmp(argv[i], "--max-ctx") == 0) {
            if (i + 1 < argc) max_ctx = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--max-verify-tokens=", 20) == 0) {
            max_verify_tokens = std::atoi(argv[i] + 20);
        } else if (std::strcmp(argv[i], "--max-verify-tokens") == 0) {
            if (i + 1 < argc) max_verify_tokens = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--hidden=", 9) == 0) {
            if (!parse_nonnegative_int(argv[i] + 9, target_hidden)) return 2;
        } else if (std::strcmp(argv[i], "--hidden") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--hidden", value) ||
                !parse_nonnegative_int(value, target_hidden)) return 2;
        } else if (std::strncmp(argv[i], "--vocab=", 8) == 0) {
            if (!parse_nonnegative_int(argv[i] + 8, target_vocab)) return 2;
        } else if (std::strcmp(argv[i], "--vocab") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--vocab", value) ||
                !parse_nonnegative_int(value, target_vocab)) return 2;
        } else if (std::strncmp(argv[i], "--max-tokens=", 13) == 0) {
            if (!parse_nonnegative_int(argv[i] + 13, target_max_tokens)) return 2;
        } else if (std::strcmp(argv[i], "--max-tokens") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--max-tokens", value) ||
                !parse_nonnegative_int(value, target_max_tokens)) return 2;
        } else if (std::strncmp(argv[i], "--kq-stride-pad=", 16) == 0) {
            kq_stride_pad = std::atoi(argv[i] + 16);
        } else if (std::strcmp(argv[i], "--kq-stride-pad") == 0) {
            if (i + 1 < argc) kq_stride_pad = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--fa-window=", 12) == 0) {
            fa_window = std::atoi(argv[i] + 12);
        } else if (std::strcmp(argv[i], "--fa-window") == 0) {
            if (i + 1 < argc) fa_window = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            if (!parse_nonnegative_int(argv[i] + 12, stream_fd)) return 2;
        } else if (std::strcmp(argv[i], "--stream-fd") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--stream-fd", value) ||
                !parse_nonnegative_int(value, stream_fd)) return 2;
        } else if (std::strncmp(argv[i], "--payload-fd=", 13) == 0) {
            if (!parse_nonnegative_int(argv[i] + 13, payload_fd)) return 2;
        } else if (std::strcmp(argv[i], "--payload-fd") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--payload-fd", value) ||
                !parse_nonnegative_int(value, payload_fd)) return 2;
        } else if (std::strncmp(argv[i], "--shared-payload-fd=", 20) == 0) {
            if (!parse_nonnegative_int(argv[i] + 20, shared_payload_fd)) return 2;
        } else if (std::strcmp(argv[i], "--shared-payload-fd") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--shared-payload-fd", value) ||
                !parse_nonnegative_int(value, shared_payload_fd)) return 2;
        } else if (std::strncmp(argv[i], "--shared-payload-bytes=", 23) == 0) {
            if (!parse_size_arg(argv[i] + 23, shared_payload_bytes)) return 2;
        } else if (std::strcmp(argv[i], "--shared-payload-bytes") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--shared-payload-bytes", value) ||
                !parse_size_arg(value, shared_payload_bytes)) return 2;
        } else if (std::strcmp(argv[i], "--enable-dflash") == 0) {
            enable_dflash = true;
        } else if (std::strncmp(argv[i], "--kvflash-pool=", 15) == 0) {
            if (!parse_nonnegative_int(argv[i] + 15, kvflash_pool_tokens)) return 2;
        } else if (std::strcmp(argv[i], "--kvflash-pool") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--kvflash-pool", value)) return 2;
            if (!parse_nonnegative_int(value, kvflash_pool_tokens)) return 2;
        } else if (std::strncmp(argv[i], "--placement=", 12) == 0) {
            placement_path = argv[i] + 12;
        } else if (std::strcmp(argv[i], "--placement") == 0) {
            if (!require_value(i, argc, argv, "--placement", placement_path)) return 2;
        } else if (std::strcmp(argv[i], "--fixed-slot-graphs") == 0) {
            ds4_fixed_slot_graphs = true;
        } else {
            std::fprintf(stderr, "[backend-ipc-daemon] unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    (void)target_hidden;
    (void)target_vocab;
    (void)target_max_tokens;

    if (ds4_fixed_slot_graphs) {
        setenv("DFLASH_MOE_FIXED_SLOT_GRAPHS", "1", 1);
    }

    switch (mode) {
        case BackendIpcMode::Invalid:
            break;
        case BackendIpcMode::DFlashDraft:
            return run_dflash_draft_ipc_daemon(payload_path, ring_cap, draft_gpu,
                                               stream_fd, payload_fd,
                                               shared_payload_fd,
                                               shared_payload_bytes);
        case BackendIpcMode::PFlashCompress:
            return run_pflash_drafter_ipc_daemon(payload_path, draft_gpu, stream_fd);
        case BackendIpcMode::Qwen35TargetShard:
            if (target_gpus.empty()) target_gpus.push_back(target_gpu);
            if (layer_begins.empty()) layer_begins.push_back(layer_begin);
            if (layer_ends.empty()) layer_ends.push_back(layer_end);
            return run_qwen35_target_shard_ipc_daemon(
                payload_path, target_gpus, layer_begins, layer_ends, max_ctx,
                max_verify_tokens, kq_stride_pad, fa_window, stream_fd,
                payload_fd, shared_payload_fd, shared_payload_bytes,
                enable_dflash, kvflash_pool_tokens);
        case BackendIpcMode::Gemma4TargetShard:
            if (target_gpus.empty()) target_gpus.push_back(target_gpu);
            if (layer_begins.empty()) layer_begins.push_back(layer_begin);
            if (layer_ends.empty()) layer_ends.push_back(layer_end);
            return run_gemma4_target_shard_ipc_daemon(
                payload_path, target_gpus, layer_begins, layer_ends, max_ctx,
                fa_window, stream_fd, payload_fd, shared_payload_fd,
                shared_payload_bytes, kvflash_pool_tokens);
        case BackendIpcMode::LagunaTargetShard:
            if (target_gpus.empty()) target_gpus.push_back(target_gpu);
            if (layer_begins.empty()) layer_begins.push_back(layer_begin);
            if (layer_ends.empty()) layer_ends.push_back(layer_end);
            return run_laguna_target_shard_ipc_daemon(
                payload_path, target_gpus, layer_begins, layer_ends, max_ctx,
                stream_fd, payload_fd, shared_payload_fd, shared_payload_bytes,
                kvflash_pool_tokens);
        case BackendIpcMode::MoeExpertCompute:
            return run_moe_expert_compute_ipc_daemon(payload_path, placement_path,
                                                     target_gpu, stream_fd,
                                                     payload_fd, shared_payload_fd,
                                                     shared_payload_bytes);
        case BackendIpcMode::DeepSeek4TargetShard:
            if (target_gpus.empty()) target_gpus.push_back(target_gpu);
            if (layer_begins.empty()) layer_begins.push_back(layer_begin);
            if (layer_ends.empty()) layer_ends.push_back(layer_end);
            return run_deepseek4_target_shard_ipc_daemon(
                payload_path, target_gpus, layer_begins, layer_ends, max_ctx,
                stream_fd, payload_fd);
    }
    std::fprintf(stderr, "[backend-ipc-daemon] unsupported mode\n");
    return 2;
}
