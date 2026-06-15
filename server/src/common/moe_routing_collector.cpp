#include "moe_routing_collector.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace dflash::common {

bool MoeRoutingCollector::open(const std::string & path) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_) {
        std::fclose(fd_);
        fd_ = nullptr;
    }
    fd_ = std::fopen(path.c_str(), "wb");
    if (!fd_) {
        std::fprintf(stderr, "[routing-collector] failed to open '%s': %s\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }
    samples_ = 0;
    std::fprintf(stderr, "[routing-collector] collecting routing data to '%s'\n", path.c_str());
    return true;
}

void MoeRoutingCollector::record(int layer_idx, const float * hidden, int n_embd,
                                 const int32_t * expert_ids, int K) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!fd_) return;

    int32_t li = layer_idx;
    int32_t ki = K;
    // A short write desyncs the fixed-layout binary stream: every subsequent
    // record would be misaligned and the file silently unparseable. Treat any
    // partial write as fatal for this file — stop writing and don't count the
    // sample, rather than emitting corrupt training data with an inflated count.
    if (std::fwrite(&li, sizeof(int32_t), 1, fd_) != 1 ||
        std::fwrite(&ki, sizeof(int32_t), 1, fd_) != 1 ||
        std::fwrite(hidden, sizeof(float), (size_t)n_embd, fd_) != (size_t)n_embd ||
        std::fwrite(expert_ids, sizeof(int32_t), (size_t)K, fd_) != (size_t)K) {
        std::fprintf(stderr, "[routing-collector] write error after %lld samples: %s\n",
                     (long long)samples_, std::strerror(errno));
        std::fclose(fd_);
        fd_ = nullptr;
        return;
    }
    samples_++;
}

void MoeRoutingCollector::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!fd_) return;
    std::fclose(fd_);
    fd_ = nullptr;
    std::fprintf(stderr, "[routing-collector] closed, %lld samples written\n",
                 (long long)samples_);
}

}  // namespace dflash::common
