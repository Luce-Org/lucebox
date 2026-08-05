// Single-process DeepSeek V4 attention-group parallelism.
//
// The primary GPU owns shared MLA attention and its authoritative cache state.
// A second GPU owns a contiguous tail of the grouped output-A projection.
// Both branches run in one ggml scheduler graph and meet at one deferred
// device-side copy per layer.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

namespace dflash::common {

struct DeepSeek4Weights;

struct DeepSeek4AttentionParallelPlan {
    int total_groups = 0;
    int main_groups = 0;
    int peer_groups = 0;
    int heads_per_group = 0;

    bool valid() const {
        return total_groups > 1 && main_groups > 0 && peer_groups > 0 &&
               main_groups + peer_groups == total_groups &&
               heads_per_group > 0;
    }
};

// Returns an empty plan when the model dimensions or requested split are not
// legal.  Groups are the indivisible unit: no attention head is split between
// devices.
DeepSeek4AttentionParallelPlan plan_deepseek4_attention_groups(
    int n_head,
    int n_out_group,
    int requested_peer_groups);

struct DeepSeek4AttentionParallelLayer {
    // Peer-owned slice containing the final `peer_groups * n_lora_o` rows of
    // the grouped output-A projection.
    ggml_tensor * output_a = nullptr;
};

struct DeepSeek4AttentionParallelState {
    DeepSeek4AttentionParallelState() = default;
    ~DeepSeek4AttentionParallelState();
    DeepSeek4AttentionParallelState(
        const DeepSeek4AttentionParallelState &) = delete;
    DeepSeek4AttentionParallelState & operator=(
        const DeepSeek4AttentionParallelState &) = delete;

    ggml_backend_t main_backend = nullptr;
    ggml_backend_t peer_backend = nullptr;
    DeepSeek4AttentionParallelPlan plan;
    int min_context = 0;

    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;
    std::vector<DeepSeek4AttentionParallelLayer> layers;

    bool enabled_for(int n_tokens, int kv_start, bool stable_cache_graph) const {
        return plan.valid() && stable_cache_graph &&
               n_tokens > 1 && n_tokens <= 6 && kv_start >= min_context;
    }
};

// Per-layer graph metadata used when the fused verifier pins work to the two
// local HIP backends and registers the single peer-to-main handoff.
struct DeepSeek4AttentionParallelGraphInputs {
    std::vector<ggml_tensor *> peer_nodes;
    std::vector<ggml_tensor *> deferred_peer_copy_nodes;
    ggml_tensor * main_output = nullptr;
    ggml_tensor * peer_output = nullptr;
    ggml_tensor * output = nullptr;
};

bool init_deepseek4_attention_parallel(
    ggml_backend_t main_backend,
    ggml_backend_t peer_backend,
    const DeepSeek4Weights & weights,
    int peer_groups,
    int min_context,
    DeepSeek4AttentionParallelState & out,
    std::string * error = nullptr);

void free_deepseek4_attention_parallel(DeepSeek4AttentionParallelState & state);

}  // namespace dflash::common
