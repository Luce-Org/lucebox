// Qwen35 target-shard IPC daemon.

#include "qwen35_target_shard_ipc.h"

#include "common/backend_ipc.h"
#include "common/dflash_draft_ipc.h"
#include "common/dflash_layer_split_runtime.h"
#include "common/io_utils.h"
#include "common/layer_split_utils.h"
#include "common/model_backend.h"
#include "common/kvflash_pager.h"
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
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace dflash::common {

namespace {

struct IpcSnapshotTensor {
    int shard = -1;
    std::string name;
    uint32_t type = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    uint64_t nbytes = 0;
    ggml_tensor * tensor = nullptr;
};

bool write_snapshot_tensor_header_fd(int fd,
                                     int shard,
                                     const ggml_tensor * t) {
    if (fd < 0 || shard < 0 || !t || !t->name[0]) return false;
    const size_t name_len_size = std::strlen(t->name);
    if (name_len_size == 0 || name_len_size >= GGML_MAX_NAME) return false;
    const int32_t shard_i = (int32_t)shard;
    const int32_t name_len = (int32_t)name_len_size;
    const uint32_t type = (uint32_t)t->type;
    const uint64_t nbytes = (uint64_t)ggml_nbytes(t);
    return write_exact_fd(fd, &shard_i, sizeof(shard_i)) &&
           write_exact_fd(fd, &name_len, sizeof(name_len)) &&
           write_exact_fd(fd, t->name, (size_t)name_len) &&
           write_exact_fd(fd, &type, sizeof(type)) &&
           write_exact_fd(fd, t->ne, sizeof(t->ne)) &&
           write_exact_fd(fd, &nbytes, sizeof(nbytes));
}

bool read_snapshot_tensor_header_fd(int fd, IpcSnapshotTensor & t) {
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
        !read_exact_fd(fd, &nbytes, sizeof(nbytes))) {
        return false;
    }
    t.shard = shard;
    t.name = std::move(name);
    t.type = type;
    t.nbytes = nbytes;
    return true;
}

bool drain_exact_fd(int fd, uint64_t bytes) {
    if (fd < 0) return false;
    std::vector<uint8_t> tmp(4 * 1024 * 1024);
    uint64_t done = 0;
    while (done < bytes) {
        const size_t chunk = (size_t)std::min<uint64_t>(
            (uint64_t)tmp.size(), bytes - done);
        if (!read_exact_fd(fd, tmp.data(), chunk)) {
            return false;
        }
        done += chunk;
    }
    return true;
}

bool bind_snapshot_tensor(PrefixSnapshot & snap,
                          const TargetCache & cache,
                          ggml_tensor * tensor) {
    if (!tensor || !tensor->name[0]) return false;
    int idx = -1;
    if (std::sscanf(tensor->name, "snap_cache_k_%d", &idx) == 1 &&
        idx >= 0 && idx < (int)snap.attn_k_snap.size()) {
        snap.attn_k_snap[(size_t)idx] = tensor;
        return true;
    }
    if (std::sscanf(tensor->name, "snap_cache_v_%d", &idx) == 1 &&
        idx >= 0 && idx < (int)snap.attn_v_snap.size()) {
        snap.attn_v_snap[(size_t)idx] = tensor;
        return true;
    }
    if (std::sscanf(tensor->name, "snap_ssm_state_%d", &idx) == 1 &&
        idx >= 0 && idx < (int)snap.ssm_state_snap.size()) {
        snap.ssm_state_snap[(size_t)idx] = tensor;
        return true;
    }
    if (std::sscanf(tensor->name, "snap_conv_state_%d", &idx) == 1 &&
        idx >= 0 && idx < (int)snap.conv_state_snap.size()) {
        snap.conv_state_snap[(size_t)idx] = tensor;
        return true;
    }
    if (std::strcmp(tensor->name, "snap_target_feat") == 0) {
        snap.target_feat_snap = tensor;
        return cache.target_feat != nullptr;
    }
    return false;
}

bool validate_snapshot_tensors(const PrefixSnapshot & snap,
                               const TargetCache & cache) {
    for (size_t i = 0; i < snap.attn_k_snap.size(); ++i) {
        const bool cache_has_kv =
            (i < cache.attn_k.size() && cache.attn_k[i]) ||
            (i < cache.attn_v.size() && cache.attn_v[i]);
        if (cache_has_kv) {
            if (i >= snap.attn_v_snap.size() ||
                !snap.attn_k_snap[i] || !snap.attn_v_snap[i]) {
                return false;
            }
        }
    }
    for (size_t i = 0; i < snap.ssm_state_snap.size(); ++i) {
        const bool cache_has_state =
            (i < cache.ssm_state.size() && cache.ssm_state[i]) ||
            (i < cache.conv_state.size() && cache.conv_state[i]);
        if (cache_has_state) {
            if (i >= snap.conv_state_snap.size() ||
                !snap.ssm_state_snap[i] || !snap.conv_state_snap[i]) {
                return false;
            }
        }
    }
    return !cache.target_feat || snap.target_feat_snap;
}

}  // namespace

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
                                       bool enable_dflash,
                                       int kvflash_pool_tokens) {
#if defined(_WIN32)
    (void)target_path; (void)gpus; (void)layer_begins; (void)layer_ends;
    (void)max_ctx; (void)max_verify_tokens; (void)kq_stride_pad; (void)fa_window;
    (void)stream_fd; (void)payload_fd; (void)shared_payload_fd;
    (void)shared_payload_bytes; (void)enable_dflash; (void)kvflash_pool_tokens;
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
                                         /*allocate_target_feat=*/false,
                                         kvflash_pool_tokens)) {
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

    KvFlashPager kvflash_pager;
    if (kvflash_pool_tokens > 0) {
        if (fa_window > 0) {
            std::fprintf(stderr,
                "[qwen35-target-shard-daemon][kvflash] fa_window must be 0\n");
            stream_status(stream_fd, -1);
            free_snapshot_backends();
            free_qwen35_layer_split_shards(shards);
            if (shared_payload && shared_payload != MAP_FAILED) {
                ::munmap(shared_payload, shared_payload_map_bytes);
            }
            return 1;
        }
        std::vector<ggml_tensor *> full_k;
        std::vector<ggml_tensor *> full_v;
        const int n_full = shards.front().weights.n_layer /
            shards.front().weights.full_attention_interval;
        for (int i = 0; i < n_full; ++i) {
            ggml_tensor * k = nullptr;
            ggml_tensor * v = nullptr;
            for (auto & shard : shards) {
                if (i < (int)shard.cache.attn_k.size() &&
                    shard.cache.attn_k[(size_t)i]) {
                    k = shard.cache.attn_k[(size_t)i];
                    v = shard.cache.attn_v[(size_t)i];
                    break;
                }
            }
            if (k && v) {
                full_k.push_back(k);
                full_v.push_back(v);
            }
        }
        KvFlashConfig pc;
        pc.pool_tokens = kvflash_pool_tokens;
        if (!kvflash_pager.attach(pc, full_k, full_v)) {
            std::fprintf(stderr,
                "[qwen35-target-shard-daemon][kvflash] pager attach failed "
                "pool=%d layers=%zu\n",
                kvflash_pool_tokens, full_k.size());
            stream_status(stream_fd, -1);
            free_snapshot_backends();
            free_qwen35_layer_split_shards(shards);
            if (shared_payload && shared_payload != MAP_FAILED) {
                ::munmap(shared_payload, shared_payload_map_bytes);
            }
            return 1;
        }
        std::fprintf(stderr,
            "[qwen35-target-shard-daemon][kvflash] resident pool %d tokens "
            "over %zu full-attn layers (logical max_ctx %d, policy=lru)\n",
            kvflash_pool_tokens, full_k.size(), max_ctx);
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
                           bool want_captures,
                           int forward_ubatch) -> bool {
        if (n_tokens <= 0 || (int)boundary.size() != hidden * n_tokens) {
            return false;
        }
        if (base_pos < 0 || base_pos + n_tokens > max_ctx) {
            return false;
        }
        forward_ubatch = std::max(1, forward_ubatch > 0 ? forward_ubatch : n_tokens);
        if (kvflash_pool_tokens > 0) {
            for (int start = 0; start < n_tokens; start += forward_ubatch) {
                const int n = std::min(forward_ubatch, n_tokens - start);
                if (!kvflash_pager.alloc_span(base_pos + start, n)) {
                    return false;
                }
            }
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
            shards, acts, base_pos, n_tokens, forward_ubatch, ignored_last_tok,
            kq_stride_pad, fa_window,
            want_argmax ? &argmax_tokens : nullptr,
            want_logits ? &logits : nullptr,
            want_captures ? &captures : nullptr,
            kvflash_pool_tokens > 0 ? &kvflash_pager : nullptr,
            kvflash_pool_tokens > 0);
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
        int forward_ubatch = 0;
        size_t bytes = 0;
        bool payload_ok = false;

        if (cmd == "forward_pipe") {
            iss >> base_pos >> n_tokens >> want_argmax >> want_logits >> bytes;
            if (!(iss >> want_captures)) want_captures = 0;
            if (!(iss >> forward_ubatch)) forward_ubatch = n_tokens;
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
            if (!(iss >> forward_ubatch)) forward_ubatch = n_tokens;
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
                if (kvflash_pool_tokens > 0) kvflash_pager.reset();
                prefill_last_logits.clear();
                stream_status(stream_fd, 0);
                continue;
            }
            if (cmd == "kvflash_sync_identity") {
                int committed = -1;
                iss >> committed;
                bool ok = kvflash_pool_tokens > 0 && committed >= 0 &&
                          committed <= kvflash_pool_tokens;
                if (ok) {
                    int min_cur_pos = std::numeric_limits<int>::max();
                    for (const auto & shard : shards) {
                        min_cur_pos = std::min(min_cur_pos, shard.cache.cur_pos);
                    }
                    ok = min_cur_pos != std::numeric_limits<int>::max() &&
                         committed <= min_cur_pos;
                }
                if (ok) {
                    kvflash_pager.reset();
                    for (int p = 0; p < committed; ++p) {
                        const int slot = kvflash_pager.slot_for(p);
                        if (slot != p) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) kvflash_pager.zero_free_blocks();
                }
                stream_status(stream_fd, ok ? 0 : -1);
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
            if (cmd == "prefix_snapshot_export") {
                int slot = -1;
                iss >> slot;
                bool ok = prefix_slot_used(slot);
                std::vector<std::pair<int, ggml_tensor *>> tensors;
                int32_t cur_pos = 0;
                int32_t last_tok = -1;
                if (ok) {
                    const auto & snaps = prefix_snapshots[(size_t)slot];
                    cur_pos = snaps.front().cur_pos;
                    last_tok = snaps.front().last_tok;
                    for (size_t shard_idx = 0; shard_idx < snaps.size(); ++shard_idx) {
                        const auto & snap = snaps[shard_idx];
                        for (ggml_tensor * t = ggml_get_first_tensor(snap.ctx); t;
                             t = ggml_get_next_tensor(snap.ctx, t)) {
                            tensors.push_back({(int)shard_idx, t});
                        }
                    }
                    ok = !tensors.empty() &&
                         !snapshot_logits[(size_t)slot].empty() &&
                         tensors.size() <= (size_t)std::numeric_limits<int32_t>::max() &&
                         snapshot_logits[(size_t)slot].size() <=
                             (size_t)std::numeric_limits<int32_t>::max();
                }
                if (!ok) {
                    stream_status(stream_fd, -1);
                    continue;
                }

                const int32_t status = 0;
                const int32_t shard_count = (int32_t)shards.size();
                const int32_t n_tensors = (int32_t)tensors.size();
                const int32_t logits_count =
                    (int32_t)snapshot_logits[(size_t)slot].size();
                ok = write_exact_fd(stream_fd, &status, sizeof(status)) &&
                     write_exact_fd(stream_fd, &cur_pos, sizeof(cur_pos)) &&
                     write_exact_fd(stream_fd, &last_tok, sizeof(last_tok)) &&
                     write_exact_fd(stream_fd, &shard_count, sizeof(shard_count)) &&
                     write_exact_fd(stream_fd, &n_tensors, sizeof(n_tensors)) &&
                     write_exact_fd(stream_fd, &logits_count, sizeof(logits_count));
                for (const auto & entry : tensors) {
                    ok = ok && write_snapshot_tensor_header_fd(
                        stream_fd, entry.first, entry.second);
                }
                std::vector<uint8_t> tmp(4 * 1024 * 1024);
                for (const auto & entry : tensors) {
                    ggml_tensor * t = entry.second;
                    const size_t nbytes = ggml_nbytes(t);
                    size_t offset = 0;
                    while (ok && offset < nbytes) {
                        const size_t chunk = std::min(tmp.size(), nbytes - offset);
                        ggml_backend_tensor_get(t, tmp.data(), offset, chunk);
                        ok = write_exact_fd(stream_fd, tmp.data(), chunk);
                        offset += chunk;
                    }
                }
                const auto & logits = snapshot_logits[(size_t)slot];
                ok = ok && write_exact_fd(stream_fd, logits.data(),
                                          sizeof(float) * logits.size());
                if (!ok) break;
                continue;
            }
            if (cmd == "prefix_snapshot_import") {
                int slot = -1;
                int cur_pos = 0;
                int last_tok = -1;
                int shard_count = 0;
                int n_tensors = 0;
                int logits_count = 0;
                iss >> slot >> cur_pos >> last_tok >> shard_count >>
                    n_tensors >> logits_count;
                bool ok = payload_fd >= 0 &&
                          slot >= 0 && slot < ModelBackend::kMaxSlots &&
                          cur_pos > 0 &&
                          shard_count == (int)shards.size() &&
                          n_tensors > 0 &&
                          logits_count > 0;
                if (!stream_status(stream_fd, ok ? 0 : -1)) break;
                if (!ok) {
                    continue;
                }

                std::vector<IpcSnapshotTensor> table;
                table.resize((size_t)n_tensors);
                std::vector<int> tensor_counts(shards.size(), 0);
                uint64_t payload_bytes = 0;
                bool headers_read = true;
                for (int i = 0; i < n_tensors; ++i) {
                    auto & entry = table[(size_t)i];
                    if (!read_snapshot_tensor_header_fd(payload_fd, entry)) {
                        headers_read = false;
                        ok = false;
                        break;
                    }
                    const bool valid =
                        entry.shard >= 0 &&
                        entry.shard < shard_count &&
                        entry.nbytes <=
                            (uint64_t)std::numeric_limits<size_t>::max() &&
                        payload_bytes <=
                            std::numeric_limits<uint64_t>::max() -
                                entry.nbytes;
                    if (!valid) {
                        ok = false;
                        continue;
                    }
                    tensor_counts[(size_t)entry.shard]++;
                    payload_bytes += entry.nbytes;
                }
                if (!headers_read) break;
                ok = ok && payload_bytes <=
                               std::numeric_limits<uint64_t>::max() -
                                   (uint64_t)logits_count * sizeof(float);
                payload_bytes += (uint64_t)logits_count * sizeof(float);
                for (int count : tensor_counts) {
                    if (count <= 0) ok = false;
                }
                if (!stream_status(stream_fd, ok ? 0 : -1)) break;
                if (!ok) {
                    continue;
                }

                std::vector<PrefixSnapshot> imported;
                if (ok) {
                    imported.resize(shards.size());
                    for (size_t shard_idx = 0; ok && shard_idx < shards.size(); ++shard_idx) {
                        auto & snap = imported[shard_idx];
                        snap.attn_k_snap.assign(shards[shard_idx].cache.attn_k.size(), nullptr);
                        snap.attn_v_snap.assign(shards[shard_idx].cache.attn_v.size(), nullptr);
                        snap.ssm_state_snap.assign(shards[shard_idx].cache.ssm_state.size(), nullptr);
                        snap.conv_state_snap.assign(shards[shard_idx].cache.conv_state.size(), nullptr);
                        snap.target_feat_snap = nullptr;
                        snap.cur_pos = cur_pos;
                        snap.last_tok = last_tok;
                        snap.kv_k_type = shards[shard_idx].cache.kv_k_type;
                        snap.max_ctx = shards[shard_idx].cache.max_ctx;
                        snap.target_feat_cap = shards[shard_idx].cache.target_feat_cap;

                        ggml_init_params ip{};
                        ip.mem_size =
                            ggml_tensor_overhead() *
                            (size_t)(tensor_counts[shard_idx] + 16);
                        ip.no_alloc = true;
                        snap.ctx = ggml_init(ip);
                        ok = snap.ctx != nullptr;
                    }
                    for (auto & entry : table) {
                        if (!ok) break;
                        PrefixSnapshot & snap = imported[(size_t)entry.shard];
                        ggml_tensor * t = ggml_new_tensor(
                            snap.ctx, (ggml_type)entry.type, 4, entry.ne);
                        if (!t || ggml_nbytes(t) != (size_t)entry.nbytes) {
                            ok = false;
                            break;
                        }
                        ggml_set_name(t, entry.name.c_str());
                        entry.tensor = t;
                        ok = bind_snapshot_tensor(
                            snap, shards[(size_t)entry.shard].cache, t);
                    }
                    for (size_t shard_idx = 0; ok && shard_idx < shards.size(); ++shard_idx) {
                        auto & snap = imported[shard_idx];
                        snap.buf = ggml_backend_alloc_ctx_tensors(
                            snap.ctx, snapshot_backends[shard_idx]);
                        ok = snap.buf != nullptr;
                    }
                }

                std::vector<uint8_t> tmp(4 * 1024 * 1024);
                uint64_t consumed_payload_bytes = 0;
                bool payload_read = true;
                if (ok) {
                    for (auto & entry : table) {
                        size_t offset = 0;
                        const size_t nbytes = (size_t)entry.nbytes;
                        while (ok && offset < nbytes) {
                            const size_t chunk = std::min(tmp.size(), nbytes - offset);
                            if (!read_exact_fd(payload_fd, tmp.data(), chunk)) {
                                payload_read = false;
                                ok = false;
                                break;
                            }
                            ggml_backend_tensor_set(
                                entry.tensor, tmp.data(), offset, chunk);
                            offset += chunk;
                            consumed_payload_bytes += chunk;
                        }
                    }
                }
                if (!payload_read) break;

                std::vector<float> imported_logits;
                if (ok) {
                    imported_logits.assign((size_t)logits_count, 0.0f);
                    const size_t logits_bytes =
                        sizeof(float) * imported_logits.size();
                    if (!read_exact_fd(payload_fd, imported_logits.data(),
                                       logits_bytes)) {
                        payload_read = false;
                        ok = false;
                    } else {
                        consumed_payload_bytes += (uint64_t)logits_bytes;
                    }
                }
                if (!payload_read) break;
                for (size_t shard_idx = 0; ok && shard_idx < shards.size(); ++shard_idx) {
                    ok = validate_snapshot_tensors(
                        imported[shard_idx], shards[shard_idx].cache);
                }
                if (ok) {
                    free_prefix_slot(slot);
                    prefix_snapshots[(size_t)slot] = std::move(imported);
                    snapshot_logits[(size_t)slot] = std::move(imported_logits);
                } else {
                    for (auto & snap : imported) {
                        free_prefix_snapshot(snap);
                    }
                    if (consumed_payload_bytes < payload_bytes) {
                        const bool drained = drain_exact_fd(
                            payload_fd, payload_bytes - consumed_payload_bytes);
                        if (!drained) break;
                    }
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
                         want_captures != 0, forward_ubatch)) {
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
