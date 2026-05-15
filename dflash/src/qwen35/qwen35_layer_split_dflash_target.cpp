// Qwen35LayerSplitDFlashTarget — DFlashTarget adapter for qwen35 layer-split.

#include "qwen35_layer_split_dflash_target.h"

#include "internal.h"
#include "graph_builders.h"
#include "step_graph.h"

namespace dflash27b {

Qwen35LayerSplitDFlashTarget::~Qwen35LayerSplitDFlashTarget() {
    step_graph_destroy(proj_sg_);
}

Qwen35LayerSplitDFlashTarget::Qwen35LayerSplitDFlashTarget(
        std::vector<TargetLayerSplitShard> & shards,
        DraftFeatureMirror * feature_ring,
        int kq_stride_pad,
        int fa_window,
        DFlashDraftIpcClient * remote_draft)
    : shards_(shards),
      feature_ring_(feature_ring),
      kq_stride_pad_(kq_stride_pad),
      fa_window_(fa_window),
      remote_draft_(remote_draft) {
    if (!shards_.empty()) {
        const TargetWeights & w = shards_.front().weights;
        capture_ids_.assign(w.capture_layer_ids,
                            w.capture_layer_ids + w.n_capture_layers);
    }
}

bool Qwen35LayerSplitDFlashTarget::verify_batch(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<int32_t> * all_argmax) {
    if (shards_.empty()) return false;
    return run_target_layer_split_forward(
        shards_, shards_.front().weights, tokens, base_pos, (int)tokens.size(),
        last_tok, kq_stride_pad_, fa_window_,
        feature_ring_,
        all_argmax, /*logits_out=*/nullptr, remote_draft_);
}

bool Qwen35LayerSplitDFlashTarget::snapshot_kv() {
    for (auto & shard : shards_) snapshot_ssm_state(shard.cache);
    return true;
}

bool Qwen35LayerSplitDFlashTarget::restore_kv() {
    for (auto & shard : shards_) restore_ssm_state(shard.cache);
    return true;
}

bool Qwen35LayerSplitDFlashTarget::is_eos(int token) const {
    if (shards_.empty()) return false;
    return is_eos_tok(token, shards_.front().weights);
}

bool Qwen35LayerSplitDFlashTarget::embed_tokens(
        const int32_t * tokens, int n, float * out) const {
    if (shards_.empty()) return false;
    return shards_.front().weights.embedder.embed(tokens, n, out);
}

bool Qwen35LayerSplitDFlashTarget::project_hidden_to_tokens(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out) {
    if (shards_.empty() || n_tokens <= 0) return false;

    auto & back = shards_.back();
    if (!proj_sg_.gf || !proj_sg_.hidden_input ||
        proj_sg_.hidden_input->ne[1] != n_tokens) {
        if (!build_lm_head_projection_step(proj_sg_, back.weights,
                                           back.backend, n_tokens)) {
            return false;
        }
    }

    ggml_backend_tensor_set(proj_sg_.hidden_input, hidden, 0,
                            sizeof(float) * (size_t)n_tokens *
                                back.weights.n_embd);

    auto st = ggml_backend_graph_compute(back.backend, proj_sg_.gf);
    if (st != GGML_STATUS_SUCCESS) return false;

    tokens_out.resize(n_tokens);
    ggml_backend_tensor_get(proj_sg_.argmax_tokens, tokens_out.data(), 0,
                            sizeof(int32_t) * n_tokens);
    return true;
}

int Qwen35LayerSplitDFlashTarget::hidden_size() const {
    return shards_.empty() ? 0 : shards_.front().weights.n_embd;
}

int Qwen35LayerSplitDFlashTarget::mask_token_id() const {
    return shards_.empty() ? 0 : shards_.front().weights.mask_token_id;
}

const std::vector<int> & Qwen35LayerSplitDFlashTarget::capture_layer_ids() const {
    return capture_ids_;
}

}  // namespace dflash27b
