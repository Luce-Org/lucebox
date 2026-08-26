#include "bailingmoe3_backend.h"

#include "common/dflash_target.h"
#include "qwen35/graph_builders.h"
#include "qwen35/prefill_helpers.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash::common {
namespace {

bool bailing_mtp_enabled() {
    const char * value = std::getenv("DFLASH_BAILING_MTP");
    return !value || std::strcmp(value, "0") != 0;
}

Qwen35Config make_qwen_runtime_config(const BailingMoe3Config & cfg) {
    Qwen35Config runtime;
    runtime.target_path = cfg.model_path;
    runtime.draft_path = cfg.draft_path;
    runtime.device = cfg.device;
    runtime.draft_gpu = cfg.draft_gpu;
    runtime.draft_ctx_max = cfg.draft_ctx_max;
    runtime.fast_rollback = cfg.fast_rollback;
    runtime.stream_fd = cfg.stream_fd;
    // Ling uses the ordinary contiguous F16/Q4 KV cache. Its compressed MLA
    // head is 576-wide, whose CUDA kernel contract uses a 256-row K/V span and
    // an explicit visibility mask even for decode. External DSpark remains a
    // single-device path and reuses Qwen35Backend's verified chain machinery.
    runtime.kq_stride_pad = 256;
    runtime.paged_attention = false;
    runtime.max_concurrency = 1;
    return runtime;
}

}  // namespace

BailingMoe3Backend::BailingMoe3Backend(const BailingMoe3Config & cfg)
    : Qwen35Backend(make_qwen_runtime_config(cfg)) {}

BailingMoe3Backend::~BailingMoe3Backend() {
    free_mtp_runtime();
}

bool BailingMoe3Backend::load_target_model(ggml_backend_t backend,
                                           TargetWeights & out) {
    return load_bailingmoe3_gguf(cfg_.target_path, backend, out);
}

void BailingMoe3Backend::print_ready_banner() const {
    const TargetWeights & weights = target_weights();
    std::printf(
        "[bailingmoe3-daemon] ready layers=%d kda=%d mla=%d "
        "experts=%d/%d groups=%d/%d mtp=%s dspark=%s ctx=%d\n",
        weights.n_layer,
        weights.n_layer - weights.n_layer / weights.full_attention_interval,
        weights.n_layer / weights.full_attention_interval,
        weights.n_expert_used, weights.n_expert,
        weights.n_expert_groups_used, weights.n_expert_groups,
        weights.mtp.enabled && !cfg_.draft_path && bailing_mtp_enabled()
            ? "on" : "off",
        cfg_.draft_path ? "on" : "off",
        cfg_.device.max_ctx);
    std::fflush(stdout);
}

void BailingMoe3Backend::shutdown() {
    free_mtp_runtime();
    Qwen35Backend::shutdown();
}

void BailingMoe3Backend::release_scratch() {
    if (mtp_sg_.alloc) {
        ggml_gallocr_free(mtp_sg_.alloc);
        mtp_sg_.alloc = nullptr;
    }
    step_graph_free(mtp_sg_);
    for (StepGraph & graph : mtp_draft_sg_) {
        if (graph.alloc) {
            ggml_gallocr_free(graph.alloc);
            graph.alloc = nullptr;
        }
        step_graph_free(graph);
    }
    Qwen35Backend::release_scratch();
}

void BailingMoe3Backend::reset_arch_request_state() {
    mtp_request_ready_ = false;
    mtp_draft_token_ = -1;
}

bool BailingMoe3Backend::has_embedded_spec_decode() const {
    return target_weights().mtp.enabled && bailing_mtp_enabled();
}

bool BailingMoe3Backend::ensure_mtp_runtime() {
    if (mtp_runtime_ready_) return true;

    const TargetWeights & source = target_weights();
    if (!source.mtp.enabled || !target_backend()) return false;

    // create_target_cache_partial() derives its cache layout from the model
    // layer pattern.  Describe the embedded predictor as a synthetic
    // one-layer, all-MLA model while borrowing every tensor from the main
    // model allocation.  No tensor or mmap ownership is copied here.
    TargetWeights & mtp = mtp_cache_weights_;
    mtp.n_layer = 1;
    mtp.full_attention_interval = 1;
    mtp.n_embd = source.n_embd;
    mtp.n_head = source.n_head;
    mtp.n_head_kv = source.n_head_kv;
    mtp.n_embd_head_k = source.n_embd_head_k;
    mtp.n_embd_head_v = source.n_embd_head_v;
    mtp.n_ff = source.n_ff;
    mtp.n_ff_exp = source.n_ff_exp;
    mtp.n_ff_shexp = source.n_ff_shexp;
    mtp.n_expert = source.n_expert;
    mtp.n_expert_used = source.n_expert_used;
    mtp.n_expert_groups = source.n_expert_groups;
    mtp.n_expert_groups_used = source.n_expert_groups_used;
    mtp.n_layer_dense_lead = 0;
    mtp.rope_dimension_count = source.rope_dimension_count;
    mtp.rope_theta = source.rope_theta;
    mtp.rms_eps = source.rms_eps;
    mtp.expert_weights_scale = source.expert_weights_scale;
    mtp.expert_gating_func = source.expert_gating_func;
    mtp.expert_weights_norm = source.expert_weights_norm;
    mtp.kda_head_dim = source.kda_head_dim;
    mtp.mla_qk_head_dim = source.mla_qk_head_dim;
    mtp.mla_v_head_dim = source.mla_v_head_dim;
    mtp.kv_lora_rank = source.kv_lora_rank;
    mtp.q_lora_rank = source.q_lora_rank;
    mtp.ssm_d_conv = source.ssm_d_conv;
    mtp.ssm_d_inner = source.ssm_d_inner;
    mtp.ssm_d_state = source.ssm_d_state;
    mtp.ssm_dt_rank = source.ssm_dt_rank;
    mtp.ssm_n_group = source.ssm_n_group;
    mtp.n_vocab = source.n_vocab;
    mtp.is_moe = true;
    mtp.is_bailingmoe3 = true;
    mtp.layers.assign(1, source.mtp.layer);
    mtp.output = source.output;
    mtp.mtp = source.mtp;

    if (!create_target_cache_partial(
            mtp, cfg_.device.max_ctx, /*max_verify_tokens=*/2,
            target_backend(), mtp_cache_, /*prefill_only=*/true,
            /*layer_begin=*/0, /*layer_end=*/1,
            /*allocate_target_feat=*/false)) {
        std::fprintf(stderr, "[bailing-mtp] cache allocation failed: %s\n",
                     dflash27b_last_error());
        mtp_cache_weights_.layers.clear();
        mtp_cache_weights_.mtp = {};
        mtp_cache_weights_.output = nullptr;
        return false;
    }

    mtp_runtime_ready_ = true;
    std::fprintf(stderr,
        "[bailing-mtp] embedded predictor ready: layer=%d ctx=%d\n",
        source.mtp.layer_index, cfg_.device.max_ctx);
    return true;
}

void BailingMoe3Backend::free_mtp_runtime() {
    step_graph_destroy(mtp_sg_);
    for (StepGraph & graph : mtp_draft_sg_) step_graph_destroy(graph);
    free_target_cache(mtp_cache_);
    mtp_cache_weights_.layers.clear();
    mtp_cache_weights_.output = nullptr;
    mtp_cache_weights_.mtp = {};
    mtp_runtime_ready_ = false;
    mtp_request_ready_ = false;
    mtp_draft_token_ = -1;
}

bool BailingMoe3Backend::run_mtp_follow(
        StepGraph & graph,
        ggml_tensor * target_hidden,
        const int32_t * next_tokens,
        int kv_start,
        int n_tokens,
        int logical_rows,
        int sample_row,
        int logits_tail_rows,
        int32_t & draft_token) {
    if (!ensure_mtp_runtime() || !target_hidden || !next_tokens ||
        n_tokens <= 0 || logical_rows <= 0 || logical_rows > n_tokens ||
        sample_row < 0 || sample_row >= logical_rows ||
        mtp_cache_.cur_pos != kv_start) {
        return false;
    }

    const TargetWeights & weights = target_weights();
    if (target_hidden->ne[0] != weights.n_embd ||
        target_hidden->ne[1] < n_tokens) {
        return false;
    }
    if (!build_bailingmoe3_mtp_step(
            graph, weights, mtp_cache_, target_backend(),
            kv_start, n_tokens, logits_tail_rows, cfg_.kq_stride_pad)) {
        std::fprintf(stderr,
            "[bailing-mtp] graph build failed at pos=%d rows=%d\n",
            kv_start, n_tokens);
        return false;
    }

    std::vector<float> embeddings(
        static_cast<size_t>(weights.n_embd) * n_tokens);
    if (!weights.embedder.embed(next_tokens, n_tokens, embeddings.data())) {
        return false;
    }
    // The reference Bailing MTP implementation masks the token embedding at
    // absolute position zero.  Preserve that training-time boundary rule.
    if (kv_start == 0) {
        std::fill_n(embeddings.data(), weights.n_embd, 0.0f);
    }
    ggml_backend_tensor_set(
        graph.inp_embed, embeddings.data(), 0,
        sizeof(float) * embeddings.size());
    bool same_layout = target_hidden->type == graph.hidden_input->type;
    for (int dim = 0; dim < GGML_MAX_DIMS && same_layout; ++dim) {
        same_layout = target_hidden->ne[dim] == graph.hidden_input->ne[dim] &&
                      target_hidden->nb[dim] == graph.hidden_input->nb[dim];
    }
    if (same_layout) {
        ggml_backend_tensor_copy_async(
            target_backend(), target_backend(),
            target_hidden, graph.hidden_input);
    } else {
        // Rejection correction uses only the valid prefix of a larger target
        // verify batch.  It is rare, so a small host bounce is preferable to
        // carrying padded invalid rows through every normal MTP step.
        std::vector<float> hidden_prefix(
            static_cast<size_t>(weights.n_embd) * n_tokens);
        ggml_backend_tensor_get(
            target_hidden, hidden_prefix.data(), 0,
            sizeof(float) * hidden_prefix.size());
        ggml_backend_tensor_set(
            graph.hidden_input, hidden_prefix.data(), 0,
            sizeof(float) * hidden_prefix.size());
    }

    std::vector<int32_t> positions(static_cast<size_t>(4 * n_tokens));
    fill_qwen35_mrope_positions(positions.data(), kv_start, n_tokens);
    ggml_backend_tensor_set(
        graph.positions, positions.data(), 0,
        sizeof(int32_t) * positions.size());

    std::vector<int64_t> rows(static_cast<size_t>(n_tokens));
    for (int i = 0; i < n_tokens; ++i) rows[(size_t)i] = kv_start + i;
    ggml_backend_tensor_set(
        graph.kv_write_rows, rows.data(), 0,
        sizeof(int64_t) * rows.size());
    const int query_rows = logits_tail_rows > 0
        ? std::min(logits_tail_rows, n_tokens)
        : n_tokens;
    upload_qwen35_causal_mask(
        graph.attn_mask, kv_start + n_tokens - query_rows,
        query_rows, cfg_.kq_stride_pad);

    const ggml_status status =
        ggml_backend_graph_compute(target_backend(), graph.gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "[bailing-mtp] compute failed at pos=%d rows=%d status=%d\n",
            kv_start, n_tokens, static_cast<int>(status));
        return false;
    }

    const int projected_rows = static_cast<int>(graph.argmax_tokens->ne[0]);
    const int projected_begin = n_tokens - projected_rows;
    const int projected_index = sample_row - projected_begin;
    if (projected_index < 0 || projected_index >= projected_rows) {
        std::fprintf(stderr,
            "[bailing-mtp] requested row %d is outside projected tail [%d,%d)\n",
            sample_row, projected_begin, n_tokens);
        return false;
    }
    ggml_backend_tensor_get(
        graph.argmax_tokens, &draft_token,
        sizeof(int32_t) * static_cast<size_t>(projected_index),
        sizeof(int32_t));

    // A rejected target proposal computes a padded second MTP row only to
    // retain the two-token CUDA-graph topology.  Its cache row is stale and
    // deliberately excluded from the logical position; the next step
    // overwrites it through set_rows before it can become visible.
    mtp_cache_.cur_pos = kv_start + logical_rows;
    return draft_token >= 0;
}

bool BailingMoe3Backend::extend_mtp_drafts(
        int32_t first_draft,
        ggml_tensor * previous_hidden,
        int start_pos,
        int draft_count,
        std::vector<int32_t> & drafts) {
    drafts.clear();
    if (draft_count <= 0 || first_draft < 0 || !previous_hidden ||
        previous_hidden->ne[0] != target_weights().n_embd ||
        previous_hidden->ne[1] != 1 || mtp_cache_.cur_pos != start_pos) {
        return false;
    }
    drafts.reserve(static_cast<size_t>(draft_count));
    drafts.push_back(first_draft);

    ggml_tensor * feedback = previous_hidden;
    for (int i = 1; i < draft_count; ++i) {
        StepGraph & graph = mtp_draft_sg_[(i - 1) % 2];
        const int32_t input_token = drafts.back();
        int32_t next_draft = -1;
        if (!run_mtp_follow(
                graph, feedback, &input_token,
                start_pos + i - 1, /*n_tokens=*/1,
                /*logical_rows=*/1, /*sample_row=*/0,
                /*logits_tail_rows=*/1, next_draft) ||
            !graph.hidden_tail) {
            return false;
        }
        drafts.push_back(next_draft);
        feedback = graph.hidden_tail;
    }
    return true;
}

bool BailingMoe3Backend::after_prefill_compute(
        StepGraph & sg,
        int kv_start,
        int n_tokens,
        const int32_t * next_tokens,
        bool is_final_chunk) {
    if (cfg_.draft_path || !target_weights().mtp.enabled ||
        !bailing_mtp_enabled()) return true;
    if (!ensure_mtp_runtime()) {
        std::fprintf(stderr,
            "[bailing-mtp] unavailable; request will use AR decode\n");
        return true;
    }
    if (kv_start == 0) reset_target_cache(mtp_cache_);
    if (!sg.hidden_states || mtp_cache_.cur_pos != kv_start) {
        std::fprintf(stderr,
            "[bailing-mtp] prefill state mismatch target_pos=%d mtp_pos=%d\n",
            kv_start, mtp_cache_.cur_pos);
        return true;
    }

    int32_t candidate = -1;
    if (!run_mtp_follow(
            mtp_sg_, sg.hidden_states, next_tokens, kv_start, n_tokens,
            /*logical_rows=*/n_tokens,
            /*sample_row=*/n_tokens - 1,
            /*logits_tail_rows=*/1, candidate)) {
        std::fprintf(stderr,
            "[bailing-mtp] prefill follower failed at pos=%d; using AR\n",
            kv_start);
        mtp_request_ready_ = false;
        return true;
    }
    if (is_final_chunk) {
        mtp_draft_token_ = candidate;
        mtp_request_ready_ = true;
    }
    return true;
}

bool BailingMoe3Backend::run_embedded_spec_decode(
        int committed,
        int n_gen,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io,
        float & out_accept_rate,
        bool & out_spec_ran,
        const BudgetHook * budget_hook,
        bool * forced_close_out,
        bool * degenerate_close_out) {
    out_accept_rate = 0.0f;
    out_spec_ran = false;
    if (n_gen <= 0) {
        io.emit(-1);
        return true;
    }

    const auto env_enabled = [](const char * name) {
        const char * value = std::getenv(name);
        return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
    };
    const bool has_budget_hook =
        budget_hook && !budget_hook->close_token_ids.empty();
    const bool needs_ar_policy =
        sampler_config().needs_logit_processing() || has_budget_hook ||
        env_enabled("DFLASH_MIN_TOKENS") ||
        env_enabled("DFLASH_DEGENERATE_RUN_TOKENS");

    DFlashTarget * target = dflash_target();
    if (needs_ar_policy || !mtp_request_ready_ || mtp_draft_token_ < 0 ||
        !target || !target->supports_fast_rollback() || !out_tokens.empty()) {
        const bool ok = run_standard_ar_decode(
            committed, n_gen, out_tokens, io,
            budget_hook ? *budget_hook : BudgetHook{},
            forced_close_out, degenerate_close_out);
        io.emit(-1);
        return ok;
    }

    const int max_drafts = std::clamp([] {
        const char * value = std::getenv("DFLASH_BAILING_MTP_TOKENS");
        return value && value[0] ? std::atoi(value) : 1;
    }(), 1, 4);

    out_spec_ran = true;
    const auto started = std::chrono::steady_clock::now();
    const size_t output_begin = out_tokens.size();
    int generated = 0;
    int proposals = 0;
    int accepted = 0;
    int target_steps = 0;
    bool hit_eos = false;
    double verify_seconds = 0.0;
    double rollback_seconds = 0.0;
    double follow_seconds = 0.0;

    // The first MTP forward ran alongside prefill and produced one proposal.
    // Feed its hidden state back through the same trained predictor to extend
    // that proposal into a short autoregressive draft chain.
    std::vector<int32_t> drafts;
    const auto initial_draft_started = std::chrono::steady_clock::now();
    if (!extend_mtp_drafts(
            mtp_draft_token_, mtp_sg_.hidden_tail, mtp_cache_.cur_pos,
            max_drafts, drafts)) {
        const bool ok = run_standard_ar_decode(
            committed, n_gen, out_tokens, io,
            budget_hook ? *budget_hook : BudgetHook{},
            forced_close_out, degenerate_close_out);
        io.emit(-1);
        return ok;
    }
    follow_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initial_draft_started).count();

    int32_t pending = target_cache().last_tok;
    out_tokens.push_back(pending);
    io.emit(pending);
    ++generated;
    hit_eos = target->is_eos(pending) || io.is_cancelled();

    auto continue_in_ar = [&]() {
        target_cache().last_tok = pending;
        if (!target->finish_speculative_state()) return false;
        const int remaining = n_gen - generated;
        const bool ok = run_standard_ar_decode(
            committed, remaining, out_tokens, io,
            budget_hook ? *budget_hook : BudgetHook{},
            forced_close_out, degenerate_close_out);
        io.emit(-1);
        return ok;
    };

    while (!hit_eos && generated < n_gen) {
        const int base_pos = committed;
        const int draft_count = std::min(max_drafts, n_gen - generated);
        if (static_cast<int>(drafts.size()) < draft_count) {
            std::fprintf(stderr,
                "[bailing-mtp] short draft chain; continuing in AR\n");
            return continue_in_ar();
        }

        std::vector<int32_t> verify_tokens;
        verify_tokens.reserve(static_cast<size_t>(draft_count + 1));
        verify_tokens.push_back(pending);
        verify_tokens.insert(
            verify_tokens.end(), drafts.begin(), drafts.begin() + draft_count);

        std::vector<int32_t> posterior;
        int32_t target_last = -1;
        const auto verify_started = std::chrono::steady_clock::now();
        if (!target->verify_batch(
                verify_tokens, base_pos, target_last, &posterior,
                /*capture_ssm_intermediates=*/true) ||
            posterior.size() != verify_tokens.size()) {
            std::fprintf(stderr,
                "[bailing-mtp] target verify failed at pos=%d\n", base_pos);
            return false;
        }
        verify_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - verify_started).count();
        ++target_steps;
        proposals += draft_count;

        int accepted_prefix = 0;
        while (accepted_prefix < draft_count &&
               posterior[accepted_prefix] == drafts[accepted_prefix]) {
            ++accepted_prefix;
        }
        accepted += accepted_prefix;
        const bool all_accepted = accepted_prefix == draft_count;

        // If EOS occurs inside the accepted prefix, do not commit or emit any
        // later verified rows.  Otherwise retain the root plus every accepted
        // draft input; the replacement/bonus remains pending for next round.
        int accepted_to_emit = accepted_prefix;
        bool eos_in_accepted = false;
        for (int i = 0; i < accepted_prefix; ++i) {
            if (target->is_eos(drafts[i])) {
                accepted_to_emit = i + 1;
                eos_in_accepted = true;
                break;
            }
        }
        const int commit_rows = 1 +
            (eos_in_accepted ? accepted_to_emit : accepted_prefix);
        const auto rollback_started = std::chrono::steady_clock::now();
        if (!target->rollback_to(base_pos, commit_rows)) {
            std::fprintf(stderr,
                "[bailing-mtp] target rollback failed at pos=%d keep=%d\n",
                base_pos, commit_rows);
            return false;
        }
        rollback_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - rollback_started).count();
        committed += commit_rows;

        for (int i = 0; i < accepted_to_emit; ++i) {
            pending = drafts[i];
            out_tokens.push_back(pending);
            io.emit(pending);
            ++generated;
            if (target->is_eos(pending) || io.is_cancelled()) {
                hit_eos = true;
                break;
            }
        }
        if (hit_eos || generated >= n_gen) break;

        std::vector<int32_t> correction_tokens;
        correction_tokens.reserve(static_cast<size_t>(commit_rows));
        correction_tokens.insert(
            correction_tokens.end(), drafts.begin(),
            drafts.begin() + accepted_prefix);

        if (all_accepted) {
            // The final target row supplies the exact bonus token.  It is
            // emitted pending: the next target verify writes its K/V row.
            pending = posterior[draft_count];
        } else {
            // First rejected position is replaced by the target's own argmax.
            pending = posterior[accepted_prefix];
        }
        correction_tokens.push_back(pending);
        out_tokens.push_back(pending);
        io.emit(pending);
        ++generated;
        hit_eos = target->is_eos(pending) || io.is_cancelled();
        if (hit_eos || generated >= n_gen) break;

        // Draft generation may have advanced the predictor several rows past
        // the target.  Rewind its logical cursor, overwrite the accepted path
        // with target-conditioned hidden states, then extend a fresh chain.
        mtp_cache_.cur_pos = base_pos;
        int32_t first_draft = -1;
        const auto follow_started = std::chrono::steady_clock::now();
        if (!run_mtp_follow(
                mtp_sg_, target_step_graph().hidden_states,
                correction_tokens.data(), base_pos, commit_rows,
                /*logical_rows=*/commit_rows,
                /*sample_row=*/commit_rows - 1,
                /*logits_tail_rows=*/1, first_draft) ||
            !extend_mtp_drafts(
                first_draft, mtp_sg_.hidden_tail, mtp_cache_.cur_pos,
                max_drafts, drafts)) {
            std::fprintf(stderr,
                "[bailing-mtp] follower failed; continuing remaining tokens in AR\n");
            return continue_in_ar();
        }
        follow_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - follow_started).count();
    }

    target_cache().last_tok = pending;
    if (!target->finish_speculative_state()) {
        std::fprintf(stderr,
            "[bailing-mtp] final target state flush failed\n");
        return false;
    }

    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const int emitted = static_cast<int>(out_tokens.size() - output_begin);
    out_accept_rate = proposals > 0
        ? static_cast<float>(accepted) / proposals
        : 0.0f;
    std::fprintf(stderr,
        "[bailing-mtp] tokens=%d time=%.3f s speed=%.2f tok/s "
        "accepted=%d/%d (%.1f%%) avg_commit=%.2f\n",
        emitted, seconds,
        seconds > 0.0 ? emitted / seconds : 0.0,
        accepted, proposals, 100.0 * out_accept_rate,
        target_steps > 0 ? static_cast<double>(emitted) / target_steps : 0.0);
    std::fprintf(stderr,
        "[bailing-mtp-profile] verify=%.3f s (%.2f ms/step) "
        "rollback=%.3f s (%.2f ms/step) follow=%.3f s (%.2f ms/step)\n",
        verify_seconds,
        target_steps > 0 ? 1000.0 * verify_seconds / target_steps : 0.0,
        rollback_seconds,
        target_steps > 0 ? 1000.0 * rollback_seconds / target_steps : 0.0,
        follow_seconds,
        target_steps > 0 ? 1000.0 * follow_seconds / target_steps : 0.0);
    io.emit(-1);
    return true;
}

}  // namespace dflash::common
