#include "layer_split_utils.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace dflash::common {

std::vector<std::pair<int,int>> compute_layer_ranges(
    int n_layer,
    int n_gpus,
    const std::vector<double> & weights)
{
    std::vector<std::pair<int,int>> ranges;
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

std::string validate_device_placement(
    const DevicePlacement & dp,
    int cuda_device_count)
{
    if (cuda_device_count <= 0) return "no CUDA devices available";

    if (dp.gpu < 0 || dp.gpu >= cuda_device_count) {
        return "primary gpu " + std::to_string(dp.gpu) +
               " out of range [0, " + std::to_string(cuda_device_count) + ")";
    }

    if (!dp.layer_split_gpus.empty()) {
        if (dp.layer_split_gpus.size() < 2) {
            return "layer_split_gpus must have at least 2 entries";
        }

        std::set<int> seen;
        for (int g : dp.layer_split_gpus) {
            if (g < 0 || g >= cuda_device_count) {
                return "layer_split gpu " + std::to_string(g) +
                       " out of range [0, " + std::to_string(cuda_device_count) + ")";
            }
            if (!seen.insert(g).second) {
                return "duplicate gpu " + std::to_string(g) + " in layer_split_gpus";
            }
        }

        if (!dp.layer_split_weights.empty() &&
            dp.layer_split_weights.size() != dp.layer_split_gpus.size()) {
            return "layer_split_weights size (" +
                   std::to_string(dp.layer_split_weights.size()) +
                   ") != gpu count (" +
                   std::to_string(dp.layer_split_gpus.size()) + ")";
        }

        for (double w : dp.layer_split_weights) {
            if (w <= 0.0 || !std::isfinite(w)) {
                return "layer_split_weights must be positive finite values";
            }
        }
    }

    if (dp.max_ctx <= 0) return "max_ctx must be positive";

    return {};  // ok
}

}  // namespace dflash::common
