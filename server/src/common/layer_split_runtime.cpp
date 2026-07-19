#include "layer_split_runtime.h"

namespace dflash::common {

bool run_layer_split_ar_decode(
        int last_tok,
        int committed,
        int n_gen,
        int vocab,
        const std::vector<float> & prefill_last_logits,
        const SamplerCfg & sampler,
        std::mt19937_64 & rng,
        const std::vector<int32_t> & history_prefix,
        const LayerSplitForwardStep & forward_one,
        const std::function<bool(int)> & is_eos,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io) {
    if (n_gen <= 0) return true;

    std::vector<int32_t> history;
    if (sampler.needs_logit_processing()) {
        history.reserve(history_prefix.size() + out_tokens.size() + (size_t)n_gen);
        history.insert(history.end(), history_prefix.begin(), history_prefix.end());
        history.insert(history.end(), out_tokens.begin(), out_tokens.end());
        if ((int)prefill_last_logits.size() != vocab) return false;
        last_tok = sample_logits(prefill_last_logits.data(), vocab, sampler,
                                 history, rng);
    }

    out_tokens.push_back(last_tok);
    if (sampler.needs_logit_processing()) history.push_back(last_tok);
    io.emit(last_tok);
    if (io.cancelled) {
        io.emit(-1);
        return true;
    }
    if (is_eos(last_tok)) {
        io.emit(-1);
        return true;
    }
    ++committed;

    std::vector<float> logits_buf;
    for (int i = 1; i < n_gen; ++i) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        logits_buf.clear();
        if (!forward_one(one, committed, next_tok,
                         sampler.needs_logit_processing() ? &logits_buf : nullptr)) {
            return false;
        }
        if (sampler.needs_logit_processing()) {
            if ((int)logits_buf.size() != vocab) return false;
            next_tok = sample_logits(logits_buf.data(), vocab, sampler,
                                     history, rng);
        }

        last_tok = next_tok;
        out_tokens.push_back(last_tok);
        if (sampler.needs_logit_processing()) history.push_back(last_tok);
        io.emit(last_tok);
        ++committed;
        if (io.cancelled) break;
        if (is_eos(last_tok)) break;
    }

    io.emit(-1);
    return true;
}

}  // namespace dflash::common
