#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::common::qwen4exp_bridge {

// Semantic state only. Runtime input rings and graph/workspace allocations are
// deliberately recreated, not checkpointed.
struct Tensor {
    std::string name;
    std::string type;
    std::array<uint32_t, 4> ne{}; // ggml order: ne[0] is fastest
    std::vector<uint8_t> data;
};

struct Checkpoint {
    uint32_t cut = 0;        // explicit forward position; never inferred from cache.cur_pos
    uint32_t prompt_len = 0;
    int32_t indexer_blocks = 0;
    std::vector<int32_t> full_layer_ids;
    std::vector<int32_t> linear_layer_ids;
    std::vector<int32_t> indexer_layer_ids;
    std::vector<int32_t> ple_layer_ids;
    std::vector<Tensor> tensors;
};

bool validate(const Checkpoint & checkpoint, std::string * error = nullptr);
std::vector<uint8_t> encode(const Checkpoint & checkpoint,
                           std::string * error = nullptr);
bool decode(const std::vector<uint8_t> & bytes, Checkpoint & checkpoint,
            std::string * error = nullptr);

// Clone a native C896 checkpoint and replace only its 12 full-attention K/V
// pairs. Donor non-K/V state is deliberately ignored.
bool compose_kv(const Checkpoint & native, const Checkpoint & donor,
                Checkpoint & output, size_t * replacements = nullptr,
                std::string * error = nullptr);

} // namespace luce::common::qwen4exp_bridge
