#include "layer_split_utils.h"

#include "common/peer_access.h"
#include "common/snapshot_backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace dflash::common {

std::vector<LayerSplitRange> compute_layer_ranges(
    int n_layer,
    int n_gpus,
    const std::vector<double> & weights)
{
    std::vector<LayerSplitRange> ranges;
    if (n_layer <= 0 || n_gpus <= 0 || n_gpus > n_layer) return ranges;

    std::vector<double> w = weights;
    if (w.empty()) w.assign((size_t)n_gpus, 1.0);
    if ((int)w.size() != n_gpus) return ranges;

    double sum = 0.0;
    for (double v : w) sum += v;
    if (sum <= 0.0) return ranges;

    ranges.reserve((size_t)n_gpus);
    int begin = 0;
    double accum = 0.0;
    for (int i = 0; i < n_gpus; i++) {
        accum += w[i];
        int end = (i == n_gpus - 1)
            ? n_layer
            : (int)std::llround((accum / sum) * n_layer);
        const int min_end = begin + 1;
        const int max_end = n_layer - (n_gpus - i - 1);
        end = std::max(min_end, std::min(max_end, end));
        ranges.push_back({begin, end});
        begin = end;
    }
    return ranges;
}

bool compute_mixed_layer_split_plan(
        const DevicePlacement & device,
        PlacementBackend local_backend,
        MixedLayerSplitPlan & out,
        const char * log_prefix) {
    const char * prefix = log_prefix ? log_prefix : "target-split";
    out = MixedLayerSplitPlan{};
    if (!device.is_mixed_layer_split() || device.layer_split_gpus.size() < 2) {
        std::fprintf(stderr, "[%s] mixed layer split requires at least two shards\n",
                     prefix);
        return false;
    }
    if (device.layer_split_backend(0) != local_backend) {
        std::fprintf(stderr,
            "[%s] first mixed shard must match compiled backend (%s)\n",
            prefix, placement_backend_name(local_backend));
        return false;
    }
    size_t remote_begin = 0;
    while (remote_begin < device.layer_split_gpus.size() &&
           device.layer_split_backend(remote_begin) == local_backend) {
        ++remote_begin;
    }
    if (remote_begin == 0 || remote_begin >= device.layer_split_gpus.size()) {
        std::fprintf(stderr,
            "[%s] mixed layer split requires one local backend group followed by "
            "one remote backend group\n", prefix);
        return false;
    }
    const PlacementBackend remote_backend =
        device.layer_split_backend(remote_begin);
    for (size_t i = remote_begin; i < device.layer_split_gpus.size(); ++i) {
        if (device.layer_split_backend(i) != remote_backend) {
            std::fprintf(stderr,
                "[%s] mixed layer split supports only one backend boundary\n",
                prefix);
            return false;
        }
    }
    out.remote_begin = remote_begin;
    out.remote_backend = remote_backend;
    return true;
}

bool compute_target_shard_layer_split_plan(
        const DevicePlacement & device,
        PlacementBackend local_backend,
        MixedLayerSplitPlan & out,
        const char * log_prefix) {
    const char * prefix = log_prefix ? log_prefix : "target-split";
    out = MixedLayerSplitPlan{};
    if (!device.is_layer_split() || device.layer_split_gpus.size() < 2) {
        std::fprintf(stderr,
            "[%s] target-shard layer split requires at least two shards\n",
            prefix);
        return false;
    }
    if (device.layer_split_backend(0) != local_backend) {
        std::fprintf(stderr,
            "[%s] first target-shard split shard must match compiled backend (%s)\n",
            prefix, placement_backend_name(local_backend));
        return false;
    }
    if (device.is_mixed_layer_split()) {
        return compute_mixed_layer_split_plan(device, local_backend, out, prefix);
    }
    out.remote_begin = 1;
    out.remote_backend = local_backend;
    return true;
}

bool init_layer_split_shard_metas(
        std::vector<LayerSplitShardMeta *> shards,
        const std::vector<int> & gpus,
        const std::vector<LayerSplitRange> & ranges,
        const char * log_prefix) {
    if (shards.size() != gpus.size() || shards.size() != ranges.size()) return false;
    const char * prefix = log_prefix ? log_prefix : "target-split";
    for (size_t i = 0; i < shards.size(); ++i) {
        auto * shard = shards[i];
        if (!shard) return false;
        shard->placement_backend = PlacementBackend::Auto;
        shard->gpu = gpus[i];
        shard->layer_begin = ranges[i].begin;
        shard->layer_end = ranges[i].end;
        shard->backend = ggml_backend_cuda_init(shard->gpu);
        if (!shard->backend) {
            std::fprintf(stderr, "[%s] backend init failed gpu=%d\n",
                         prefix, shard->gpu);
            return false;
        }
    }
    return true;
}

bool enable_layer_split_peer_access(
        const std::vector<int> & gpus,
        bool peer_access) {
    if (!peer_access) return true;
    for (size_t i = 0; i < gpus.size(); ++i) {
        for (size_t j = i + 1; j < gpus.size(); ++j) {
            (void)enable_peer_access_pair(gpus[i], gpus[j]);
        }
    }
    return true;
}

bool init_layer_split_snapshot_backends(
        const std::vector<LayerSplitShardMeta *> & shards,
        std::vector<ggml_backend_t> & snapshot_backends,
        const char * log_prefix) {
    const char * prefix = log_prefix ? log_prefix : "target-split";
    snapshot_backends.assign(shards.size(), nullptr);
    auto rollback = [&]() {
        for (size_t j = 0; j < snapshot_backends.size(); ++j) {
            if (shards[j]) {
                free_snapshot_backend(snapshot_backends[j], shards[j]->backend);
            }
            snapshot_backends[j] = nullptr;
        }
        snapshot_backends.clear();
    };
    for (size_t i = 0; i < shards.size(); ++i) {
        const auto * shard = shards[i];
        if (!shard || !shard->backend) {
            rollback();
            return false;
        }
        snapshot_backends[i] = create_snapshot_backend(shard->backend);
        if (!snapshot_backends[i]) {
            std::fprintf(stderr,
                "[%s] snapshot backend init failed gpu=%d\n",
                prefix, shard->gpu);
            rollback();
            return false;
        }
    }
    return true;
}

void free_layer_split_snapshot_backends(
        const std::vector<LayerSplitShardMeta *> & shards,
        std::vector<ggml_backend_t> & snapshot_backends) {
    const size_t n = std::min(shards.size(), snapshot_backends.size());
    for (size_t i = 0; i < n; ++i) {
        if (!shards[i]) continue;
        free_snapshot_backend(snapshot_backends[i], shards[i]->backend);
    }
    snapshot_backends.clear();
}

}  // namespace dflash::common
