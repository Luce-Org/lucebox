// Qwen35 target-shard IPC daemon.

#include "qwen35_target_shard_ipc.h"

#include "common/backend_ipc.h"
#include "common/dflash_draft_ipc.h"
#include "common/dflash_layer_split_runtime.h"
#include "common/io_utils.h"
#include "common/layer_split_utils.h"
#include "common/model_backend.h"
#include "common/snapshot_backend.h"
#include "graph_builders.h"
#include "internal.h"
#include "layer_split_forward.h"
#include "layer_split_types.h"

#include "ggml.h"
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

int run_qwen35_target_shard_ipc_daemon(const char * target_path,
                                       const std::vector<int> & gpus,
                                       const std::vector<int> & layer_begins,
                                       const std::vector<int> & layer_ends,
                                       int max_ctx,
                                       int max_verify_tokens,
                                       int kq_stride_pad,
                                       int fa_window,
                                       int stream_fd,
                                       int payload_fd,
                                       int shared_payload_fd,
                                       size_t shared_payload_bytes,
                                       bool enable_dflash) {
#if defined(_WIN32)
    (void)target_path; (void)gpus; (void)layer_begins; (void)layer_ends;
    (void)max_ctx; (void)max_verify_tokens; (void)kq_stride_pad; (void)fa_window;
    (void)stream_fd; (void)payload_fd; (void)shared_payload_fd;
    (void)shared_payload_bytes; (void)enable_dflash;
    std::fprintf(stderr, "Qwen35 target shard IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    if (!target_path || gpus.empty() || gpus.size() != layer_begins.size() ||
        gpus.size() != layer_ends.size() || max_ctx <= 0 || stream_fd < 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=qwen35-target-shard "
            "<target.gguf> --target-gpus=N[,N...] "
            "--layer-begins=N[,N...] --layer-ends=N[,N...] "
            "--max-ctx=N --stream-fd=FD [--payload-fd=FD]\n");
        return 2;
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (gpus[i] < 0 || layer_begins[i] < 0 ||
            layer_ends[i] <= layer_begins[i]) {
            std::fprintf(stderr, "[qwen35-target-shard-daemon] bad shard config\n");
            return 2;
        }
        if (i > 0 && layer_begins[i] != layer_ends[i - 1]) {
            std::fprintf(stderr, "[qwen35-target-shard-daemon] remote layers must be contiguous\n");
            return 2;
        }
    }

    void * shared_payload = nullptr;
    void * shared_payload_data = nullptr;
    size_t shared_payload_capacity = 0;
    size_t shared_payload_map_bytes = 0;
    if (shared_payload_fd >= 0 || shared_payload_bytes > 0) {
        if (shared_payload_fd < 0 || shared_payload_bytes == 0) {
            std::fprintf(stderr, "[qwen35-target-shard-daemon] bad shared payload fd/size\n");
            stream_status(stream_fd, -1);
            return 1;
        }
        if (!backend_ipc_shared_payload_map_bytes(shared_payload_bytes,
                                                  shared_payload_map_bytes)) {
            std::fprintf(stderr, "[qwen35-target-shard-daemon] bad shared payload size\n");
            stream_status(stream_fd, -1);
            return 1;
        }
        shared_payload = ::mmap(nullptr, shared_payload_map_bytes, PROT_READ | PROT_WRITE,
                                MAP_SHARED, shared_payload_fd, 0);
        if (shared_payload == MAP_FAILED) {
            std::fprintf(stderr, "[qwen35-target-shard-daemon] shared payload mmap failed\n");
            stream_status(stream_fd, -1);
            return 1;
        }
        shared_payload_data =
            static_cast<char *>(shared_payload) + backend_ipc_shared_payload_header_bytes();
        shared_payload_capacity = shared_payload_bytes;
    }

    std::vector<Qwen35LayerSplitShard> shards(gpus.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        auto & shard = shards[i];
        shard.gpu = gpus[i];
        shard.layer_begin = layer_begins[i];
        shard.layer_end = layer_ends[i];
        shard.backend = ggml_backend_cuda_init(shard.gpu);
        if (!shard.backend) {
            std::fprintf(stderr,
                         "[qwen35-target-shard-daemon] backend init failed gpu=%d\n",
                         shard.gpu);
            stream_status(stream_fd, -1);
            free_qwen35_layer_split_shards(shards);
            if (shared_payload && shared_payload != MAP_FAILED) {
                ::munmap(shared_payload, shared_payload_map_bytes);
            }
            return 1;
        }
    }

    for (auto & shard : shards) {
        const bool is_last = (&shard == &shards.back());
        const TargetLoadPlan plan =
            make_layer_split_load_plan<TargetLoadPlan>(shard, is_last);
        if (!load_target_gguf_partial(target_path, shard.backend, plan, shard.weights) ||
            !create_target_cache_partial(shard.weights, max_ctx, max_verify_tokens,
                                         shard.backend, shard.cache,
                                         /*prefill_only=*/!enable_dflash,
                                         shard.layer_begin, shard.layer_end,
                                         /*allocate_target_feat=*/false)) {
            std::fprintf(stderr,
                         "[qwen35-target-shard-daemon] load/cache failed gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            stream_status(stream_fd, -1);
            free_qwen35_layer_split_shards(shards);
            if (shared_payload && shared_payload != MAP_FAILED) {
                ::munmap(shared_payload, shared_payload_map_bytes);
            }
            return 1;
        }
    }

    std::vector<ggml_backend_t> snapshot_backends(shards.size(), nullptr);
    auto free_snapshot_backends = [&]() {
        for (size_t i = 0; i < snapshot_backends.size(); ++i) {
            if (i < shards.size() && snapshot_backends[i]) {
                free_snapshot_backend(snapshot_backends[i], shards[i].backend);
            }
            snapshot_backends[i] = nullptr;
        }
    };
    for (size_t i = 0; i < shards.size(); ++i) {
        snapshot_backends[i] = create_snapshot_backend(shards[i].backend);
        if (!snapshot_backends[i]) {
            std::fprintf(stderr,
                         "[qwen35-target-shard-daemon] snapshot backend init failed gpu=%d\n",
                         shards[i].gpu);
            stream_status(stream_fd, -1);
            free_snapshot_backends();
            free_qwen35_layer_split_shards(shards);
            if (shared_payload && shared_payload != MAP_FAILED) {
                ::munmap(shared_payload, shared_payload_map_bytes);
            }
            return 1;
        }
    }
    std::vector<std::vector<PrefixSnapshot>> prefix_snapshots(
        (size_t)ModelBackend::kMaxSlots);
    for (auto & slot : prefix_snapshots) {
        slot.resize(shards.size());
    }
    std::vector<std::vector<float>> snapshot_logits(
        (size_t)ModelBackend::kMaxSlots);
    auto free_prefix_slot = [&](int slot) {
        if (slot < 0 || slot >= ModelBackend::kMaxSlots) return;
        for (auto & snap : prefix_snapshots[(size_t)slot]) {
            free_prefix_snapshot(snap);
        }
        snapshot_logits[(size_t)slot].clear();
    };
    auto prefix_slot_used = [&](int slot) -> bool {
        if (slot < 0 || slot >= ModelBackend::kMaxSlots) return false;
        const auto & snaps = prefix_snapshots[(size_t)slot];
        if (snaps.size() != shards.size()) return false;
        for (const auto & snap : snaps) {
            if (!snap.ctx) return false;
        }
        if (snapshot_logits[(size_t)slot].empty()) return false;
        return true;
    };

    if (shards.empty()) {
        stream_status(stream_fd, -1);
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }

    const int hidden = shards.front().weights.n_embd;
    std::vector<float> host_act;
    std::vector<int32_t> argmax_tokens;
    std::vector<float> prefill_last_logits;

    std::fprintf(stderr,
                 "[qwen35-target-shard-daemon] ready shards=%zu layers=[%d,%d)\n",
                 shards.size(), shards.front().layer_begin, shards.back().layer_end);
    for (const auto & shard : shards) {
        std::fprintf(stderr,
                     "[qwen35-target-shard-daemon] shard gpu=%d layers=[%d,%d)\n",
                     shard.gpu, shard.layer_begin, shard.layer_end);
    }
    stream_status(stream_fd, 0);

    auto run_forward = [&](int base_pos,
                           int n_tokens,
                           const std::vector<float> & boundary,
                           bool want_argmax,
                           bool want_logits,
                           bool want_captures) -> bool {
        if (n_tokens <= 0 || (int)boundary.size() != hidden * n_tokens) {
            return false;
        }
        if (base_pos < 0 || base_pos + n_tokens > max_ctx) {
            return false;
        }
        ActivationPair acts;
        if (!activation_pair_init(acts, shards.front().backend, hidden, n_tokens)) {
            std::fprintf(stderr, "[qwen35-target-shard-daemon] activation alloc failed\n");
            return false;
        }
        ggml_backend_tensor_set(acts.a, boundary.data(), 0,
                                sizeof(float) * boundary.size());
        int ignored_last_tok = -1;
        argmax_tokens.clear();
        std::vector<float> logits;
        std::vector<Qwen35TargetCaptureSlice> captures;
        const bool forward_ok = run_qwen35_layer_split_forward_from_activation(
            shards, acts, base_pos, n_tokens, std::max(1, n_tokens), ignored_last_tok,
            kq_stride_pad, fa_window,
            want_argmax ? &argmax_tokens : nullptr,
            want_logits ? &logits : nullptr,
            want_captures ? &captures : nullptr);
        activation_pair_free(acts);
        if (!forward_ok) return false;

        if (!want_argmax && argmax_tokens.empty()) {
            argmax_tokens.push_back(ignored_last_tok);
        }

        const int32_t status = 0;
        const int32_t last_tok = argmax_tokens.empty() ? -1 : argmax_tokens.back();
        if (!write_exact_fd(stream_fd, &status, sizeof(status)) ||
            !write_exact_fd(stream_fd, &last_tok, sizeof(last_tok))) {
            return false;
        }
        if (want_argmax &&
            !write_exact_fd(stream_fd, argmax_tokens.data(),
                            sizeof(int32_t) * argmax_tokens.size())) {
            return false;
        }
        if (want_logits &&
            !write_exact_fd(stream_fd, logits.data(),
                            sizeof(float) * logits.size())) {
            return false;
        }
        if (want_logits) {
            prefill_last_logits = logits;
        }
        if (want_captures) {
            const int32_t n_captures = (int32_t)captures.size();
            if (!write_exact_fd(stream_fd, &n_captures, sizeof(n_captures))) {
                return false;
            }
            for (const auto & capture : captures) {
                const int32_t capture_idx = capture.capture_idx;
                const int32_t capture_start_pos = capture.start_pos;
                const int32_t capture_n_tokens = capture.n_tokens;
                const int32_t capture_elems = (int32_t)capture.data.size();
                if (!write_exact_fd(stream_fd, &capture_idx, sizeof(capture_idx)) ||
                    !write_exact_fd(stream_fd, &capture_start_pos, sizeof(capture_start_pos)) ||
                    !write_exact_fd(stream_fd, &capture_n_tokens, sizeof(capture_n_tokens)) ||
                    !write_exact_fd(stream_fd, &capture_elems, sizeof(capture_elems)) ||
                    !write_exact_fd(stream_fd, capture.data.data(),
                                    sizeof(float) * capture.data.size())) {
                    return false;
                }
            }
        }
        for (auto & shard : shards) {
            shard.cache.cur_pos = base_pos + n_tokens;
            shard.cache.last_tok = last_tok;
        }
        return true;
    };

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
        int want_captures = 0;
        size_t bytes = 0;
        bool payload_ok = false;

        if (cmd == "forward_pipe") {
            iss >> base_pos >> n_tokens >> want_argmax >> want_logits >> bytes;
            if (!(iss >> want_captures)) want_captures = 0;
            const size_t expected_bytes =
                (size_t)n_tokens * (size_t)hidden * sizeof(float);
            if (payload_fd >= 0 && bytes == expected_bytes) {
                host_act.assign(bytes / sizeof(float), 0.0f);
                payload_ok = read_exact_fd(payload_fd, host_act.data(), bytes);
            }
        } else if (cmd == "forward_shared") {
            uint64_t seq = 0;
            iss >> base_pos >> n_tokens >> want_argmax >> want_logits >> bytes >> seq;
            if (!(iss >> want_captures)) want_captures = 0;
            const size_t expected_bytes =
                (size_t)n_tokens * (size_t)hidden * sizeof(float);
            const auto * header =
                static_cast<const BackendIpcSharedPayloadHeader *>(shared_payload);
            if (shared_payload && shared_payload != MAP_FAILED && shared_payload_data &&
                seq != 0 && bytes == expected_bytes &&
                backend_ipc_payload_in_bounds(0, bytes, shared_payload_capacity) &&
                header->sequence == seq && header->bytes == (uint64_t)bytes) {
                host_act.assign(bytes / sizeof(float), 0.0f);
                std::memcpy(host_act.data(), shared_payload_data, bytes);
                payload_ok = true;
            }
        } else {
            if (cmd == "snapshot") {
                if (!enable_dflash) {
                    stream_status(stream_fd, -1);
                } else {
                    for (auto & shard : shards) snapshot_ssm_state(shard.cache);
                    stream_status(stream_fd, 0);
                }
                continue;
            }
            if (cmd == "restore") {
                if (!enable_dflash) {
                    stream_status(stream_fd, -1);
                } else {
                    for (auto & shard : shards) restore_ssm_state(shard.cache);
                    stream_status(stream_fd, 0);
                }
                continue;
            }
            if (cmd == "reset_request_state") {
                for (auto & shard : shards) reset_target_cache(shard.cache);
                prefill_last_logits.clear();
                stream_status(stream_fd, 0);
                continue;
            }
            if (cmd == "project_pipe") {
                int project_tokens = 0;
                size_t project_bytes = 0;
                iss >> project_tokens >> project_bytes;
                const size_t expected_bytes =
                    (size_t)project_tokens * (size_t)hidden * sizeof(float);
                std::vector<float> hidden_in;
                hidden_in.assign(project_bytes / sizeof(float), 0.0f);
                std::vector<int32_t> project_tokens_out;
                bool project_ok =
                    payload_fd >= 0 && project_tokens > 0 &&
                    project_bytes == expected_bytes &&
                    read_exact_fd(payload_fd, hidden_in.data(), project_bytes) &&
                    [&]() {
                        Qwen35LayerSplitShard & back = shards.back();
                        StepGraph project_sg;
                        const bool built = build_lm_head_projection_step(
                            project_sg, back.weights, back.backend, project_tokens);
                        bool ok = built;
                        if (ok) {
                            ggml_backend_tensor_set(
                                project_sg.hidden_input, hidden_in.data(), 0,
                                sizeof(float) * hidden_in.size());
                            auto st = ggml_backend_graph_compute(back.backend, project_sg.gf);
                            ok = st == GGML_STATUS_SUCCESS;
                        }
                        if (ok) {
                            project_tokens_out.resize((size_t)project_tokens);
                            ggml_backend_tensor_get(
                                project_sg.argmax_tokens, project_tokens_out.data(), 0,
                                sizeof(int32_t) * (size_t)project_tokens);
                        }
                        step_graph_destroy(project_sg);
                        return ok;
                    }();
                if (!project_ok) {
                    stream_status(stream_fd, -1);
                } else {
                    stream_status(stream_fd, 0);
                    write_exact_fd(stream_fd, project_tokens_out.data(),
                                   sizeof(int32_t) * project_tokens_out.size());
                }
                continue;
            }
            if (cmd == "prefix_snapshot_save") {
                int slot = -1;
                iss >> slot;
                bool ok = slot >= 0 && slot < ModelBackend::kMaxSlots &&
                          snapshot_backends.size() == shards.size();
                if (ok) {
                    free_prefix_slot(slot);
                    for (size_t i = 0; i < shards.size(); ++i) {
                        if (!snapshot_target_cache(shards[i].weights, shards[i].cache,
                                                   snapshot_backends[i],
                                                   prefix_snapshots[(size_t)slot][i])) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (ok) {
                    snapshot_logits[(size_t)slot] = prefill_last_logits;
                } else {
                    free_prefix_slot(slot);
                }
                stream_status(stream_fd, ok ? 0 : -1);
                continue;
            }
            if (cmd == "prefix_snapshot_free") {
                int slot = -1;
                iss >> slot;
                free_prefix_slot(slot);
                stream_status(stream_fd, 0);
                continue;
            }
            if (cmd == "prefix_snapshot_restore") {
                int slot = -1;
                iss >> slot;
                bool ok = prefix_slot_used(slot);
                int cur_pos = 0;
                if (ok) {
                    cur_pos = prefix_snapshots[(size_t)slot].front().cur_pos;
                    for (size_t i = 0; i < shards.size(); ++i) {
                        const auto & snap = prefix_snapshots[(size_t)slot][i];
                        if (snap.cur_pos != cur_pos ||
                            !restore_target_cache(snap, shards[i].cache)) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (ok) {
                    prefill_last_logits = snapshot_logits[(size_t)slot];
                }
                stream_status(stream_fd, ok ? 0 : -1);
                continue;
            }
            std::fprintf(stderr, "[qwen35-target-shard-daemon] unknown command: %s\n",
                         line.c_str());
        }

        if (!payload_ok ||
            !run_forward(base_pos, n_tokens, host_act,
                         want_argmax != 0, want_logits != 0,
                         want_captures != 0)) {
            const int32_t status = -1;
            write_exact_fd(stream_fd, &status, sizeof(status));
        }
    }

    for (int slot = 0; slot < ModelBackend::kMaxSlots; ++slot) {
        free_prefix_slot(slot);
    }
    free_snapshot_backends();
    free_qwen35_layer_split_shards(shards);
    if (shared_payload && shared_payload != MAP_FAILED) {
        ::munmap(shared_payload, shared_payload_map_bytes);
    }
    return 0;
#endif
}

}  // namespace dflash::common
