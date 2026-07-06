// DeepSeek4 target shard IPC daemon.
//
// Runs on the remote Halo (AMD HIP) GPU, receives boundary activations from the
// CUDA parent shard, runs layers [begin, end) + optionally the output head, and
// returns either the boundary activation or logits.

#include "deepseek4_layer_split_adapter.h"
#include "deepseek4_internal.h"
#include "common/target_shard_ipc.h"
#include "common/target_shard_ipc_daemon.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace dflash::common {

namespace {
using TargetShardClock = std::chrono::steady_clock;

static void free_deepseek4_target_shards(
        std::vector<DeepSeek4LayerSplitShard> & shards) {
    for (auto & shard : shards) {
        free_deepseek4_cache(shard.cache);
        free_deepseek4_weights(shard.weights);
        if (shard.backend) {
            ggml_backend_free(shard.backend);
            shard.backend = nullptr;
        }
    }
}

static bool validate_target_shard_config(
        const char * target_path,
        const std::vector<int> & gpus,
        const std::vector<int> & layer_begins,
        const std::vector<int> & layer_ends,
        int max_ctx,
        int stream_fd) {
    if (!target_path || gpus.empty() || gpus.size() != layer_begins.size() ||
        gpus.size() != layer_ends.size() || max_ctx <= 0 || stream_fd < 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=deepseek4-target-shard "
            "<target.gguf> --target-gpus=N[,N...] "
            "--layer-begins=N[,N...] --layer-ends=N[,N...] "
            "--max-ctx=N --stream-fd=FD [--payload-fd=FD]\n");
        return false;
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (gpus[i] < 0 || layer_begins[i] < 0 ||
            layer_ends[i] <= layer_begins[i]) {
            std::fprintf(stderr, "[deepseek4-target-shard] bad shard config\n");
            return false;
        }
        if (i > 0 && layer_begins[i] != layer_ends[i - 1]) {
            std::fprintf(stderr,
                         "[deepseek4-target-shard] remote layers must be contiguous\n");
            return false;
        }
    }
    return true;
}

static uint64_t target_shard_elapsed_us(TargetShardClock::time_point start,
                                        TargetShardClock::time_point end) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static bool target_shard_timing_enabled() {
    const char * value = std::getenv("DFLASH_DS4_TIMING");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static double target_shard_ms(uint64_t us) {
    return (double)us / 1000.0;
}

static void log_target_shard_timing(int n_tokens,
                                    int layer_begin,
                                    int layer_end,
                                    const DeepSeek4StepTelemetry & tel) {
    std::fprintf(stderr,
        "[deepseek4-target-timing] tokens=%d layers=[%d,%d) total=%.1fms "
        "hc_pre=%.1fms hc_pre_build=%.1fms hc_pre_input=%.1fms hc_pre_compute=%.1fms "
        "attn_build=%.1fms attn_compute=%.1fms attn_read=%.1fms "
        "ffn_build=%.1fms ffn_compute=%.1fms ffn_read=%.1fms hc_post=%.1fms output=%.1fms\n",
        n_tokens, layer_begin, layer_end,
        target_shard_ms(tel.total_us),
        target_shard_ms(tel.hc_pre_attn_us + tel.hc_pre_ffn_us),
        target_shard_ms(tel.hc_pre_build_us),
        target_shard_ms(tel.hc_pre_input_us),
        target_shard_ms(tel.hc_pre_compute_us),
        target_shard_ms(tel.attn_build_us),
        target_shard_ms(tel.attn_compute_us),
        target_shard_ms(tel.attn_read_us),
        target_shard_ms(tel.ffn_build_us),
        target_shard_ms(tel.ffn_compute_us),
        target_shard_ms(tel.ffn_read_us),
        target_shard_ms(tel.hc_post_attn_us + tel.hc_post_ffn_us),
        target_shard_ms(tel.output_us));
}
} // namespace

int run_deepseek4_target_shard_ipc_daemon(
        const char * target_path,
        const std::vector<int> & gpus,
        const std::vector<int> & layer_begins,
        const std::vector<int> & layer_ends,
        int max_ctx,
        int stream_fd,
        int payload_fd) {
    if (!validate_target_shard_config(
            target_path, gpus, layer_begins, layer_ends, max_ctx, stream_fd)) {
        return 2;
    }

    std::fprintf(stderr,
                 "[deepseek4-target-shard] starting: %zu shard(s) layers=[%d,%d) max_ctx=%d\n",
                 gpus.size(), layer_begins.front(), layer_ends.back(), max_ctx);

    std::vector<DeepSeek4LayerSplitShard> shards(gpus.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        auto & shard = shards[i];
        shard.gpu = gpus[i];
        shard.layer_begin = layer_begins[i];
        shard.layer_end = layer_ends[i];
        shard.backend = ggml_backend_cuda_init(shard.gpu);
        if (!shard.backend) {
            std::fprintf(stderr,
                         "[deepseek4-target-shard] failed to init GPU %d\n",
                         shard.gpu);
            free_deepseek4_target_shards(shards);
            return 1;
        }
    }

    for (size_t i = 0; i < shards.size(); ++i) {
        auto & shard = shards[i];
        TargetLoadPlan plan;
        plan.layer_begin = shard.layer_begin;
        plan.layer_end = shard.layer_end;
        plan.load_output = (i + 1 == shards.size());

        if (!load_deepseek4_gguf_partial(target_path, shard.backend, plan, shard.weights)) {
            std::fprintf(stderr,
                         "[deepseek4-target-shard] failed to load weights for shard %zu\n",
                         i);
            free_deepseek4_target_shards(shards);
            return 1;
        }
        if (!create_deepseek4_cache(shard.backend, shard.weights, max_ctx, shard.cache)) {
            std::fprintf(stderr,
                         "[deepseek4-target-shard] failed to create cache for shard %zu\n",
                         i);
            free_deepseek4_target_shards(shards);
            return 1;
        }
        if (i > 0) {
            const auto & first = shards.front().weights;
            if (shard.weights.n_embd != first.n_embd ||
                shard.weights.n_hc != first.n_hc ||
                shard.weights.n_vocab != first.n_vocab) {
                std::fprintf(stderr,
                             "[deepseek4-target-shard] inconsistent shard dimensions\n");
                free_deepseek4_target_shards(shards);
                return 1;
            }
        }
        std::fprintf(stderr,
                     "[deepseek4-target-shard] loaded shard %zu: gpu=%d layers=[%d,%d)%s\n",
                     i, shard.gpu, shard.layer_begin, shard.layer_end,
                     i + 1 == shards.size() ? " (+output)" : "");
    }

    const int hidden = shards.front().weights.n_embd * shards.front().weights.n_hc;
    std::vector<float> hc_state;
    auto shard_metas = layer_split_shard_metas(shards);
    std::vector<ggml_backend_t> snapshot_backends;
    if (!init_layer_split_snapshot_backends(shard_metas, snapshot_backends,
                                            "deepseek4-target-shard")) {
        free_deepseek4_target_shards(shards);
        return 1;
    }
    struct Snapshot {
        int cur_pos = 0;
        std::vector<float> hc_state;
        std::vector<DeepSeek4Snapshot> shards;
        bool used = false;
    };
    std::vector<Snapshot> snapshots((size_t)ModelBackend::kMaxSlots);
    auto free_slot = [&](int slot) {
        if (slot < 0 || slot >= ModelBackend::kMaxSlots) return;
        auto & snap = snapshots[(size_t)slot];
        for (auto & shard_snap : snap.shards) {
            free_deepseek4_snapshot(shard_snap);
        }
        snap.cur_pos = 0;
        snap.hc_state.clear();
        snap.used = false;
        if (snap.shards.size() != shards.size()) snap.shards.resize(shards.size());
    };

    // Set up daemon callbacks
    TargetShardDaemonCallbacks callbacks;
    callbacks.log_prefix = "deepseek4-target-shard";

    callbacks.forward = [&](const TargetShardDaemonForwardRequest & req,
                            TargetShardDaemonForwardResponse & resp) -> bool {
        const int n_tokens = req.n_tokens;
        if (!req.boundary_activation || n_tokens <= 0 || req.base_pos < 0 ||
            req.base_pos + n_tokens > max_ctx ||
            req.boundary_activation->size() != (size_t)hidden * (size_t)n_tokens) {
            return false;
        }
        if (req.token_ids && (int)req.token_ids->size() != n_tokens) {
            return false;
        }
        const bool timing = target_shard_timing_enabled();

        std::vector<float> logits;
        const float * shard_input = req.boundary_activation->data();
        for (size_t i = 0; i < shards.size(); ++i) {
            auto & shard = shards[i];
            const bool is_last = (i + 1 == shards.size());
            DeepSeek4StepTelemetry tel;
            const auto forward_t0 = TargetShardClock::now();
            std::fprintf(stderr,
                         "[deepseek4-target-shard] forward shard %zu: n_tokens=%d base_pos=%d layers=[%d,%d)\n",
                         i, n_tokens, req.base_pos, shard.layer_begin, shard.layer_end);
            const bool ok = deepseek4_step_layer_range(
                shard.backend, shard.weights, shard.cache, hc_state,
                shard_input, n_tokens, req.base_pos,
                shard.layer_begin, shard.layer_end,
                is_last ? &logits : nullptr,
                req.token_ids ? req.token_ids->data() : nullptr,
                timing ? &tel : nullptr);
            if (!ok) {
                std::fprintf(stderr,
                             "[deepseek4-target-shard] forward failed on shard %zu\n",
                             i);
                return false;
            }
            if (timing) {
                const uint64_t wall_us =
                    target_shard_elapsed_us(forward_t0, TargetShardClock::now());
                if (wall_us > tel.total_us) tel.total_us = wall_us;
                log_target_shard_timing(n_tokens, shard.layer_begin, shard.layer_end, tel);
            }
            shard.cache.cur_pos = req.base_pos + n_tokens;
            shard_input = hc_state.data();
        }

        const int vocab = shards.back().weights.n_vocab;
        if (vocab <= 0 || logits.size() != (size_t)vocab) {
            std::fprintf(stderr,
                         "[deepseek4-target-shard] invalid logits payload: expected=%d got=%zu\n",
                         vocab, logits.size());
            return false;
        }

        const float * last_logits = logits.data();
        int32_t best = 0;
        float best_val = last_logits[0];
        for (int i = 1; i < vocab; ++i) {
            if (last_logits[i] > best_val) {
                best_val = last_logits[i];
                best = i;
            }
        }
        resp.last_tok = best;
        if (req.want_logits) {
            resp.logits = std::move(logits);
        }

        return true;
    };

    callbacks.reset_request_state = [&]() -> bool {
        for (auto & shard : shards) {
            shard.cache.cur_pos = 0;
            for (auto & lc : shard.cache.layers) {
                lc.n_comp = 0;
                lc.n_index_comp = 0;
            }
        }
        std::fill(hc_state.begin(), hc_state.end(), 0.0f);
        return true;
    };

    callbacks.snapshot_save = [&](int slot) -> bool {
        if (slot < 0 || slot >= ModelBackend::kMaxSlots ||
            snapshot_backends.size() != shards.size()) {
            return false;
        }
        const int snap_pos = shards.empty() ? 0 : shards.front().cache.cur_pos;
        if (snap_pos <= 0) return false;
        free_slot(slot);
        auto & snap = snapshots[(size_t)slot];
        if (snap.shards.size() != shards.size()) snap.shards.resize(shards.size());
        for (size_t i = 0; i < shards.size(); ++i) {
            if (!deepseek4_snapshot_save(shards[i].cache, snapshot_backends[i],
                                         snap.shards[i])) {
                free_slot(slot);
                return false;
            }
        }
        snap.cur_pos = snap_pos;
        snap.hc_state = hc_state;
        snap.used = true;
        return true;
    };

    callbacks.snapshot_free = [&](int slot) {
        free_slot(slot);
    };

    callbacks.snapshot_restore = [&](int slot) -> bool {
        if (slot < 0 || slot >= ModelBackend::kMaxSlots) return false;
        auto & snap = snapshots[(size_t)slot];
        if (!snap.used || snap.cur_pos <= 0 || snap.shards.size() != shards.size()) {
            return false;
        }
        for (size_t i = 0; i < shards.size(); ++i) {
            if (snap.shards[i].cur_pos != snap.cur_pos ||
                !deepseek4_snapshot_restore(snap.shards[i], shards[i].cache)) {
                return false;
            }
        }
        hc_state = snap.hc_state;
        return true;
    };

    std::fprintf(stderr,
                 "[deepseek4-target-shard] ready: %zu shard(s) layers=[%d,%d) hidden=%d vocab=%d\n",
                 shards.size(), layer_begins.front(), layer_ends.back(),
                 hidden, shards.back().weights.n_vocab);

    const int rc = run_target_shard_ipc_daemon_loop(
        hidden, shards.back().weights.n_vocab,
        stream_fd, payload_fd,
        /*shared_payload_fd=*/-1, /*shared_payload_bytes=*/0,
        std::move(callbacks));

    for (int slot = 0; slot < ModelBackend::kMaxSlots; ++slot) free_slot(slot);
    free_layer_split_snapshot_backends(shard_metas, snapshot_backends);
    free_deepseek4_target_shards(shards);
    return rc;
}

}  // namespace dflash::common
