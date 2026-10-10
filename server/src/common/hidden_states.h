// Residual-stream activations read out of a prefill (/v1/hidden_states).
//
// A hidden-states request runs the prompt through prefill only — no decode,
// no sampling — and reports, for each requested block L, the residual stream
// after block L (before the final norm) at the last prompt token and,
// optionally, averaged over every prompt token. Prefill is chunked, so the
// backend hands each chunk's rows to HiddenStatesAccumulator, which keeps the
// running per-layer sum and the latest last row.

#pragma once

#include <cstddef>
#include <vector>

namespace luce::common {

struct HiddenStatesSpec {
    // Block indices in [0, n_layers), in response order. Entries may repeat
    // when two requested spellings (e.g. 23 and -1) name the same block.
    std::vector<int> layers;
    // Also report the mean over all prompt tokens.
    bool mean = false;
};

struct HiddenStates {
    // One n_embd vector per HiddenStatesSpec::layers entry, in the same order.
    std::vector<std::vector<float>> last;
    // Parallel to `last`; empty unless HiddenStatesSpec::mean.
    std::vector<std::vector<float>> mean;
    // Prompt tokens the activations were read over.
    int n_tokens = 0;
};

class HiddenStatesAccumulator {
public:
    HiddenStatesAccumulator() = default;
    HiddenStatesAccumulator(HiddenStatesSpec spec, int n_embd)
        : spec_(std::move(spec)), n_embd_(n_embd),
          last_(spec_.layers.size(), std::vector<float>((size_t)n_embd, 0.0f)),
          sum_(spec_.mean ? spec_.layers.size() : 0,
               std::vector<double>((size_t)n_embd, 0.0)) {}

    const HiddenStatesSpec & spec() const { return spec_; }
    int n_embd() const { return n_embd_; }

    // Record layer entry `k` of one prefill chunk: its last row and, when the
    // spec asks for the mean, the sum of its rows. Call add_rows() once per
    // chunk after every entry is recorded.
    void record(size_t k, const float * last_row, const float * row_sum) {
        last_[k].assign(last_row, last_row + n_embd_);
        if (!spec_.mean || !row_sum) return;
        std::vector<double> & sum = sum_[k];
        for (int i = 0; i < n_embd_; ++i) sum[(size_t)i] += row_sum[i];
    }
    void add_rows(int n_rows) { rows_ += n_rows; }
    int rows() const { return rows_; }

    HiddenStates finish() const {
        HiddenStates out;
        out.n_tokens = rows_;
        out.last = last_;
        if (spec_.mean) {
            out.mean.reserve(sum_.size());
            for (const std::vector<double> & sum : sum_) {
                std::vector<float> mean(sum.size(), 0.0f);
                if (rows_ > 0) {
                    for (size_t i = 0; i < sum.size(); ++i) {
                        mean[i] = (float)(sum[i] / (double)rows_);
                    }
                }
                out.mean.push_back(std::move(mean));
            }
        }
        return out;
    }

private:
    HiddenStatesSpec spec_;
    int n_embd_ = 0;
    int rows_ = 0;
    std::vector<std::vector<float>> last_;
    std::vector<std::vector<double>> sum_;
};

}  // namespace luce::common
