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
