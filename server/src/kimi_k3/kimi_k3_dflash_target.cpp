#include "kimi_k3_dflash_target.h"

#include "common/dflash_feature_ring.h"

#include "ggml-alloc.h"

#include <cstdio>
#include <utility>

namespace dflash::common {

KimiK3DFlashTarget::KimiK3DFlashTarget(
        KimiK3Weights & weights,
        KimiK3Cache & cache,
        ggml_backend_t backend,
        DraftFeatureMirror & feature_ring,
        std::vector<int> capture_layer_ids,
        int mask_token_id,
        bool fast_rollback,
        MoeHybridStreamEngine * stream_engine,
        MoeStreamDualOwnerExecutor * dual_stream_executor,
        const MoeStreamDualOwnerPolicy * stream_owner_policy,
        MoeHybridRoutingStats * routing_stats)
    : weights_(weights),
      cache_(cache),
      backend_(backend),
      feature_ring_(feature_ring),
      capture_layer_ids_(std::move(capture_layer_ids)),
      mask_token_id_(mask_token_id),
      fast_rollback_(fast_rollback),
      stream_engine_(stream_engine),
      dual_stream_executor_(dual_stream_executor),
      stream_owner_policy_(stream_owner_policy),
      routing_stats_(routing_stats) {}

KimiK3DFlashTarget::~KimiK3DFlashTarget() {
    step_graph_destroy(embedding_graph_);
    step_graph_destroy(projection_graph_);
}

bool KimiK3DFlashTarget::sync_captures(
        const KimiK3ForwardResult & result,
        int base_pos,
        int n_tokens) {
    const size_t capture_values =
        static_cast<size_t>(weights_.n_embd) * n_tokens;
    if (result.captured_hidden.size() !=
        capture_values * capture_layer_ids_.size()) {
        std::fprintf(stderr,
            "[kimi-k3-dspark] target capture shape mismatch: got=%zu expected=%zu\n",
            result.captured_hidden.size(),
            capture_values * capture_layer_ids_.size());
        return false;
    }
    for (size_t i = 0; i < capture_layer_ids_.size(); ++i) {
        if (!copy_host_capture_slice_to_draft_ring(
                feature_ring_, static_cast<int>(i), base_pos, n_tokens,
                result.captured_hidden.data() + i * capture_values,
                capture_values)) {
            std::fprintf(stderr,
                "[kimi-k3-dspark] feature-ring copy failed at capture %zu\n", i);
            return false;
        }
    }
    return true;
}

bool KimiK3DFlashTarget::forward_token(
        int32_t token, int position, std::vector<float> & logits) {
    KimiK3ForwardOptions options;
    options.capture_layer_ids = &capture_layer_ids_;
    options.read_logits = true;
    options.read_argmax = false;
    KimiK3ForwardResult result;
    if (!kimi_k3_forward(
            backend_, weights_, cache_, std::vector<int32_t>{token}, position,
            options, result, stream_engine_, dual_stream_executor_,
            stream_owner_policy_, routing_stats_) ||
        !sync_captures(result, position, 1)) {
        return false;
    }
    logits = std::move(result.logits);
    return true;
}

bool KimiK3DFlashTarget::verify_batch(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<int32_t> * all_argmax,
        bool capture_ssm_intermediates) {
    KimiK3ForwardOptions options;
    options.capture_layer_ids = &capture_layer_ids_;
    options.capture_replay = fast_rollback_ && capture_ssm_intermediates;
    options.read_logits = false;
    options.read_argmax = true;
    KimiK3ForwardResult result;
    if (!kimi_k3_forward(
            backend_, weights_, cache_, tokens, base_pos,
            options, result, stream_engine_, dual_stream_executor_,
            stream_owner_policy_, routing_stats_) ||
        result.argmax.size() != tokens.size() ||
        !sync_captures(result, base_pos, static_cast<int>(tokens.size()))) {
        return false;
    }
    last_tok = result.argmax.back();
    if (all_argmax) *all_argmax = std::move(result.argmax);
    return true;
}

bool KimiK3DFlashTarget::snapshot_kv() {
    return kimi_k3_replay_snapshot(backend_, cache_);
}

bool KimiK3DFlashTarget::restore_kv() {
    return kimi_k3_replay_restore(backend_, cache_);
}

bool KimiK3DFlashTarget::supports_fast_rollback() const {
    return fast_rollback_ && cache_.max_verify_tokens > 0 &&
           cache_.snapshot_valid && cache_.replay_valid;
}

bool KimiK3DFlashTarget::prefer_fast_rollback_over_replay() const {
    // ReplaySSM recomputes only the recurrent KDA transitions; replaying a
    // token would also reread every routed MoE layer from external storage.
    return stream_engine_ != nullptr;
}

bool KimiK3DFlashTarget::rollback_to(int base_pos, int commit_n) {
    return kimi_k3_replay_commit(
        backend_, weights_, cache_, base_pos, commit_n);
}

bool KimiK3DFlashTarget::is_eos(int token) const {
    return token == weights_.eos_token_id;
}

bool KimiK3DFlashTarget::build_embedding_graph(int n_tokens) const {
    StepGraph & graph = embedding_graph_;
    step_graph_free(graph);
    ggml_init_params params{};
    params.mem_size = 4ull * 1024ull * 1024ull;
    params.no_alloc = true;
    graph.ctx = ggml_init(params);
    if (!graph.ctx) return false;
    graph.gf = ggml_new_graph_custom(graph.ctx, 256, false);
    graph.token_ids = ggml_new_tensor_1d(
        graph.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(graph.token_ids);
    graph.hidden_states = ggml_get_rows(
        graph.ctx, weights_.tok_embd, graph.token_ids);
    if (graph.hidden_states->type != GGML_TYPE_F32) {
        graph.hidden_states = ggml_cast(
            graph.ctx, graph.hidden_states, GGML_TYPE_F32);
    }
    ggml_set_output(graph.hidden_states);
    ggml_build_forward_expand(graph.gf, graph.hidden_states);
    if (!graph.alloc) {
        graph.alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
    }
    return graph.alloc && ggml_gallocr_alloc_graph(graph.alloc, graph.gf);
}

bool KimiK3DFlashTarget::embed_tokens(
        const int32_t * tokens, int n, float * out) const {
    if (!tokens || !out || n <= 0 || !build_embedding_graph(n)) return false;
    ggml_backend_tensor_set(
        embedding_graph_.token_ids, tokens, 0, sizeof(int32_t) * n);
    if (ggml_backend_graph_compute(backend_, embedding_graph_.gf) !=
        GGML_STATUS_SUCCESS) {
        return false;
    }
    ggml_backend_tensor_get(
        embedding_graph_.hidden_states, out, 0,
        sizeof(float) * static_cast<size_t>(weights_.n_embd) * n);
    return true;
}

bool KimiK3DFlashTarget::build_projection_graph(int n_tokens) {
    StepGraph & graph = projection_graph_;
    step_graph_free(graph);
    ggml_init_params params{};
    params.mem_size = 4ull * 1024ull * 1024ull;
    params.no_alloc = true;
    graph.ctx = ggml_init(params);
    if (!graph.ctx) return false;
    graph.gf = ggml_new_graph_custom(graph.ctx, 256, false);
    graph.hidden_input = ggml_new_tensor_2d(
        graph.ctx, GGML_TYPE_F32, weights_.n_embd, n_tokens);
    ggml_set_input(graph.hidden_input);
    graph.logits = ggml_mul_mat(
        graph.ctx, weights_.output, graph.hidden_input);
    graph.argmax_tokens = ggml_argmax(graph.ctx, graph.logits);
    ggml_set_output(graph.logits);
    ggml_set_output(graph.argmax_tokens);
    ggml_build_forward_expand(graph.gf, graph.logits);
    ggml_build_forward_expand(graph.gf, graph.argmax_tokens);
    if (!graph.alloc) {
        graph.alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
    }
    return graph.alloc && ggml_gallocr_alloc_graph(graph.alloc, graph.gf);
}

bool KimiK3DFlashTarget::project_hidden_to_tokens(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out) {
    if (!hidden || n_tokens <= 0 || !build_projection_graph(n_tokens)) return false;
    ggml_backend_tensor_set(
        projection_graph_.hidden_input, hidden, 0,
        sizeof(float) * static_cast<size_t>(weights_.n_embd) * n_tokens);
    if (ggml_backend_graph_compute(backend_, projection_graph_.gf) !=
        GGML_STATUS_SUCCESS) {
        return false;
    }
    tokens_out.resize(static_cast<size_t>(n_tokens));
    ggml_backend_tensor_get(
        projection_graph_.argmax_tokens, tokens_out.data(), 0,
        sizeof(int32_t) * tokens_out.size());
    return true;
}

bool KimiK3DFlashTarget::project_hidden_to_logits(
        const float * hidden,
        int n_tokens,
        std::vector<float> & logits_out) {
    if (!hidden || n_tokens <= 0 || !build_projection_graph(n_tokens)) return false;
    ggml_backend_tensor_set(
        projection_graph_.hidden_input, hidden, 0,
        sizeof(float) * static_cast<size_t>(weights_.n_embd) * n_tokens);
    if (ggml_backend_graph_compute(backend_, projection_graph_.gf) !=
        GGML_STATUS_SUCCESS) {
        return false;
    }
    logits_out.resize(
        static_cast<size_t>(weights_.n_vocab) * n_tokens);
    ggml_backend_tensor_get(
        projection_graph_.logits, logits_out.data(), 0,
        sizeof(float) * logits_out.size());
    return true;
}

ggml_tensor * KimiK3DFlashTarget::lm_head_tensor() {
    return weights_.output;
}

ggml_tensor * KimiK3DFlashTarget::gpu_embd_table() {
    return weights_.tok_embd;
}

ggml_backend_t KimiK3DFlashTarget::fused_head_backend() {
    return backend_;
}

int KimiK3DFlashTarget::hidden_size() const {
    return weights_.n_embd;
}

int KimiK3DFlashTarget::mask_token_id() const {
    return mask_token_id_;
}

const std::vector<int> & KimiK3DFlashTarget::capture_layer_ids() const {
    return capture_layer_ids_;
}

int KimiK3DFlashTarget::default_adaptive_verify_min_rows() const {
    // Each additional row can route another set of file-backed experts.  The
    // DSpark confidence head should therefore be allowed to stop after the
    // seed row; resident targets retain the shared runtime's default floor.
    return stream_engine_ ? 1 : 0;
}

} // namespace dflash::common
