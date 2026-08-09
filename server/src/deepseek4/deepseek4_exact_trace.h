#pragma once

#include "deepseek4_internal.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

struct GenerateRequest;

class DeepSeek4ExactTraceWriter {
public:
    static std::unique_ptr<DeepSeek4ExactTraceWriter> create_from_env(
        const DeepSeek4Weights & weights);

    DeepSeek4ExactTraceWriter(const DeepSeek4ExactTraceWriter &) = delete;
    DeepSeek4ExactTraceWriter & operator=(const DeepSeek4ExactTraceWriter &) = delete;

    bool begin_request(const GenerateRequest & request, bool restored, int cache_position);
    void end_request(bool ok, int cache_position, const std::vector<int32_t> & tokens);

    bool wants_step(int position_begin, int n_tokens) const;

    void record_reset(int cache_position);
    void record_step(int position_begin, int n_tokens, int cache_position, bool logits_present);
    void record_layer(
        ggml_backend_t backend,
        const DeepSeek4Cache & cache,
        const std::vector<float> & hc_state,
        int layer,
        int position_begin,
        int n_tokens,
        bool hash_routed,
        const std::vector<int32_t> & routing_ids,
        const std::vector<float> & routing_weights);
    void record_capture(
        int position_begin,
        int n_tokens,
        int capture_begin,
        const std::vector<int> & layer_ids,
        const std::vector<float> & rows);
    void record_logits(int position, const std::vector<float> & logits);
    void record_snapshot(const char * kind, int slot, const DeepSeek4Cache & cache);

    static std::string hash_bytes(const void * data, size_t bytes);
    static std::string hash_token_ids(const std::vector<int32_t> & tokens);

private:
    DeepSeek4ExactTraceWriter(
        const DeepSeek4Weights & weights,
        std::ofstream output,
        int width);

    const DeepSeek4Weights & weights_;
    std::ofstream output_;
    int width_ = 0;
    int request_index_ = -1;
    int prompt_tokens_ = 0;
    int snapshot_position_ = -1;
    std::vector<int> compressor_boundaries_;

    void write_relations(int position_begin, int position_end);
    std::string cache_state_hash(const DeepSeek4Cache & cache) const;
};

}  // namespace dflash::common
