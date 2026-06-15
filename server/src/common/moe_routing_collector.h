// MoE routing data collector — writes per-token binary routing data for
// predictor training (gate input hidden states + expert selections).
//
// Binary format per sample (matches StreamMoE train_predictor.py format):
//   int32  layer_idx
//   int32  K  (number of selected experts)
//   float32[n_embd]  hidden_state (gate input / post-norm hidden for this layer)
//   int32[K]  expert_indices (actual routing result)
//
// Enable via --collect-routing <path> on the server CLI or via env var
// DFLASH_COLLECT_ROUTING=<path>.

#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace dflash::common {

class MoeRoutingCollector {
public:
    MoeRoutingCollector() = default;
    ~MoeRoutingCollector() { close(); }

    MoeRoutingCollector(const MoeRoutingCollector &) = delete;
    MoeRoutingCollector & operator=(const MoeRoutingCollector &) = delete;

    // Open the output file. Returns false on failure.
    bool open(const std::string & path);

    // Record one routing sample: gate input + expert selection for one token at one layer.
    // hidden must point to n_embd floats; expert_ids to K int32s.
    void record(int layer_idx, const float * hidden, int n_embd,
                const int32_t * expert_ids, int K);

    // Close and flush the output file. Prints summary to stderr.
    void close();

    bool    is_open() const { return fd_ != nullptr; }
    int64_t sample_count() const { return samples_; }

private:
    std::FILE * fd_ = nullptr;
    int64_t samples_ = 0;  // wide: collection runs can exceed 2^31 samples
    std::mutex mu_;
};

}  // namespace dflash::common
