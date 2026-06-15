#include "moe_routing_collector.h"

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
    std::fwrite(&li, sizeof(int32_t), 1, fd_);
    std::fwrite(&ki, sizeof(int32_t), 1, fd_);
    std::fwrite(hidden, sizeof(float), (size_t)n_embd, fd_);
    std::fwrite(expert_ids, sizeof(int32_t), (size_t)K, fd_);
    samples_++;
}

void MoeRoutingCollector::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!fd_) return;
    std::fclose(fd_);
    fd_ = nullptr;
    std::fprintf(stderr, "[routing-collector] closed, %d samples written\n", samples_);
}

}  // namespace dflash::common
