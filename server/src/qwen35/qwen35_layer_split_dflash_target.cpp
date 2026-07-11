// Qwen35LayerSplitDFlashTarget — DFlashTarget adapter for qwen35 layer-split.

#include "qwen35_layer_split_dflash_target.h"
#include "qwen35_layer_split_tree_guard.h"

#include "internal.h"
#include "graph_builders.h"
#include "step_graph.h"
#include "common/kvflash_pager.h"
#include "common/gpu_runtime_compat.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>

using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
extern "C++" to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

namespace dflash::common {

namespace {

static bool split_rollback_context_fatal(cudaError_t err) {
    return err == cudaErrorIllegalAddress ||
           err == cudaErrorAssert ||
           err == cudaErrorLaunchFailure;
}

}  // namespace

Qwen35LayerSplitDFlashTarget::~Qwen35LayerSplitDFlashTarget() {
    step_graph_destroy(proj_sg_);
}

Qwen35LayerSplitDFlashTarget::Qwen35LayerSplitDFlashTarget(
        std::vector<Qwen35LayerSplitShard> & shards,
        DraftFeatureMirror * feature_ring,
        int kq_stride_pad,
        int fa_window,
        DFlashDraftIpcClient * remote_draft,
        Qwen35TargetShardIpcClient * remote_target_shard,
        KvFlashPager * kvflash)
    : shards_(shards),
      feature_ring_(feature_ring),
      kq_stride_pad_(kq_stride_pad),
      fa_window_(fa_window),
      remote_draft_(remote_draft),
      remote_target_shard_(remote_target_shard),
      kvflash_(kvflash) {
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
        std::vector<int32_t> * all_argmax,
        bool capture_ssm_intermediates) {
    if (shards_.empty()) return false;
    if (rollback_poisoned_) {
        if (std::getenv("DFLASH_SPLIT_CHAIN_ROLLBACK_DIAG") != nullptr) {
            std::fprintf(stderr,
                "[target-split][rollback-poison] verify_batch_refused_while_poisoned=1 base_pos=%d n_tokens=%zu\n",
                base_pos, tokens.size());
        }
        return false;
    }
    Qwen35SplitCaptureStats capture_stats;
    const bool capture_selftest = std::getenv("DFLASH_SPLIT_CAPTURE_SELFTEST") != nullptr;
    capture_stats.reset(shards_.size());
    if (capture_ssm_intermediates) capture_stats.requested++;

    bool ok = false;
    if (remote_target_shard_ && remote_target_shard_->active()) {
        ok = run_qwen35_mixed_layer_split_forward(
            shards_, *remote_target_shard_, shards_.front().weights, tokens,
            base_pos, (int)tokens.size(), last_tok, kq_stride_pad_, fa_window_,
            all_argmax, /*logits_out=*/nullptr, feature_ring_, remote_draft_,
            kvflash_, capture_ssm_intermediates, &capture_stats);
    } else {
        ok = run_qwen35_layer_split_forward(
            shards_, shards_.front().weights, tokens, base_pos, (int)tokens.size(),
            last_tok, kq_stride_pad_, fa_window_,
            feature_ring_,
            all_argmax, /*logits_out=*/nullptr, remote_draft_,
            /*activation_type=*/GGML_TYPE_F32, kvflash_,
            capture_ssm_intermediates, &capture_stats);
    }

    if (capture_selftest || capture_ssm_intermediates) {
        std::fprintf(stderr,
            "[target-split][capture] split_capture_requested=%llu split_capture_enabled=%llu split_capture_missing_owner_count=%llu ok=%d\n",
            (unsigned long long)capture_stats.requested,
            (unsigned long long)capture_stats.enabled,
            (unsigned long long)capture_stats.missing_owner_count, ok ? 1 : 0);
        for (size_t i = 0; i < capture_stats.layers_owned_per_shard.size(); ++i) {
            const int gpu = i < shards_.size() ? shards_[i].gpu : -1;
            std::fprintf(stderr,
                "[target-split][capture] shard=%zu gpu=%d split_capture_layers_owned_per_shard=%llu split_capture_slots_written_per_shard=%llu\n",
                i, gpu,
                (unsigned long long)capture_stats.layers_owned_per_shard[i],
                (unsigned long long)capture_stats.slots_written_per_shard[i]);
            const unsigned long long feature_owned =
                i < capture_stats.feature_taps_owned_per_shard.size()
                    ? (unsigned long long)capture_stats.feature_taps_owned_per_shard[i] : 0ULL;
            const unsigned long long feature_written =
                i < capture_stats.feature_slots_written_per_shard.size()
                    ? (unsigned long long)capture_stats.feature_slots_written_per_shard[i] : 0ULL;
            std::fprintf(stderr,
                "[target-split][capture] shard=%zu gpu=%d split_feature_taps_owned_per_shard=%llu split_feature_slots_written_per_shard=%llu target_feat_absent_by_config=true\n",
                i, gpu, feature_owned, feature_written);
        }
    }
    bool capture_gate_ok = false;
    bool feature_gate_ok = true;
    bool shard_gate_ok = !capture_stats.layers_owned_per_shard.empty();
    if (capture_ssm_intermediates) {
        capture_gate_ok = ok && capture_stats.enabled > 0 &&
            capture_stats.missing_owner_count == 0;
        for (size_t i = 0; i < capture_stats.layers_owned_per_shard.size(); ++i) {
            if (capture_stats.layers_owned_per_shard[i] > 0 &&
                capture_stats.slots_written_per_shard[i] == 0) {
                shard_gate_ok = false;
            }
            if (i < capture_stats.feature_taps_owned_per_shard.size() &&
                capture_stats.feature_taps_owned_per_shard[i] > 0 &&
                capture_stats.feature_slots_written_per_shard[i] == 0) {
                feature_gate_ok = false;
            }
        }
        // Stage 3 tree verification depends on capture storage/config validation,
        // not on whether the most recent forward pass requested a fresh capture.
        // Once the current target instance proves its capture configuration, keep
        // that gate sticky across no-capture verification/replay passes. A future
        // storage/config rebuild creates a new target instance; do not clear this
        // on ordinary restore_kv() or no-capture passes.
        split_capture_validated_ = split_capture_validated_ ||
            (capture_gate_ok && shard_gate_ok && feature_gate_ok);
    }
    if (capture_selftest && capture_ssm_intermediates) {
        if (!split_capture_validated_) {
            std::fprintf(stderr, "[target-split][capture] self-test failed closed\n");
            return false;
        }
    }
    return ok;
}

bool Qwen35LayerSplitDFlashTarget::snapshot_kv() {
    for (auto & shard : shards_) snapshot_ssm_state(shard.cache);
    if (remote_target_shard_ && remote_target_shard_->active()) {
        return remote_target_shard_->snapshot_kv();
    }
    return true;
}

bool Qwen35LayerSplitDFlashTarget::restore_kv() {
    if (remote_target_shard_ && remote_target_shard_->active()) {
        if (!remote_target_shard_->restore_kv()) {
            return false;
        }
    }
    for (auto & shard : shards_) restore_ssm_state(shard.cache);
    rollback_poisoned_ = false;
    return true;
}

bool Qwen35LayerSplitDFlashTarget::supports_fast_rollback() const {
    const char * e = std::getenv("DFLASH_SPLIT_FAST_ROLLBACK");
    const bool env_enabled = e != nullptr && std::strcmp(e, "0") != 0;
    const bool local_only = !(remote_target_shard_ && remote_target_shard_->active());
    const bool supported = env_enabled && local_only && split_capture_validated_;
    if (std::getenv("DFLASH_SPLIT_CHAIN_ROLLBACK_DIAG") != nullptr) {
        std::fprintf(stderr,
            "[target-split][chain-rollback] split_chain_fast_rollback_supported=%d env_enabled=%d capture_validated=%d local_only=%d split_tree_verify_supported=0 split_rollback_to_tree_supported=0\n",
            supported ? 1 : 0, env_enabled ? 1 : 0,
            split_capture_validated_ ? 1 : 0, local_only ? 1 : 0);
    }
    return supported;
}

bool Qwen35LayerSplitDFlashTarget::rollback_to(int base_pos, int commit_n) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool diag = std::getenv("DFLASH_SPLIT_CHAIN_ROLLBACK_DIAG") != nullptr;
    last_rollback_context_fatal_ = false;
    auto fail = [&](const char * why, cudaError_t err = cudaSuccess) {
        const bool has_cuda_error = err != cudaSuccess;
        const bool context_fatal = has_cuda_error && split_rollback_context_fatal(err);
        last_rollback_context_fatal_ = context_fatal;
        if (diag) {
            std::fprintf(stderr,
                "[target-split][chain-rollback] split_chain_fast_rollback_fail=1 reason=%s split_chain_fast_rollback_fallback_restore_replay=%d split_chain_rollback_cuda_error_name=%s split_chain_rollback_failure_class=%s split_chain_rollback_context_fatal=%d\n",
                why,
                context_fatal ? 0 : 1,
                has_cuda_error ? cudaGetErrorName(err) : "cudaSuccess",
                context_fatal ? "context-fatal" : "recoverable-return",
                context_fatal ? 1 : 0);
        }
        return false;
    };

    if (!supports_fast_rollback()) return fail("unsupported_or_not_validated");
    if (remote_target_shard_ && remote_target_shard_->active()) return fail("remote_target_shard_active");
    if (shards_.empty()) return fail("no_shards");
    if (commit_n <= 0) return fail("commit_n_nonpositive");

    int q_len = -1;
    for (const auto & shard : shards_) {
        const int shard_q = shard.cache.cur_pos - base_pos;
        if (shard_q < 0) return fail("negative_q_len");
        if (q_len < 0) q_len = shard_q;
        if (q_len != shard_q) return fail("cross_shard_cur_pos_mismatch");
    }

    if (commit_n >= q_len) {
        for (auto & shard : shards_) shard.cache.cur_pos = base_pos + commit_n;
        if (diag) {
            std::fprintf(stderr,
                "[target-split][chain-rollback] split_chain_fast_rollback_attempts=1 split_chain_fast_rollback_success=1 split_chain_rollback_commit_n=%d split_chain_rollback_slot_idx=%d split_chain_rollback_context_device_per_shard=none_no_restore split_chain_rollback_stream_source_per_shard=none_no_restore split_chain_feature_ring_action=none_required_position_aligned split_tree_verify_supported=0 split_rollback_to_tree_supported=0\n",
                commit_n, commit_n - 1);
        }
        return true;
    }

    const int rollback_idx = commit_n - 1;
    std::vector<int> restored_per_shard(shards_.size(), 0);
    int prior_device = -1;
    (void)cudaGetDevice(&prior_device);

    for (size_t si = 0; si < shards_.size(); ++si) {
        auto & shard = shards_[si];
        auto & cache = shard.cache;
        const auto & w = shard.weights;
        cudaError_t ce = cudaSetDevice(shard.gpu);
        if (ce != cudaSuccess) return fail("cuda_set_device_failed", ce);
        const cudaStream_t shard_stream = cudaStreamPerThread;
        if (diag) {
            int current_device = -1;
            cudaGetDevice(&current_device);
            std::fprintf(stderr,
                "[target-split][chain-rollback] shard=%zu gpu=%d split_chain_rollback_context_device_per_shard=%d split_chain_rollback_stream_source_per_shard=cudaStreamPerThread\n",
                si, shard.gpu, current_device);
        }
        int dn_idx = 0;
        for (int il = 0; il < w.n_layer; ++il) {
            const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
            if (is_attn) continue;
            const bool owns_layer = il >= shard.layer_begin && il < shard.layer_end;
            if (!owns_layer) { dn_idx++; continue; }
            if (dn_idx >= (int)cache.ssm_state.size() || dn_idx >= (int)cache.conv_state.size() ||
                dn_idx >= (int)cache.ssm_intermediate.size() || dn_idx >= (int)cache.conv_input_cache.size()) {
                return fail("capture_index_oob");
            }
            ggml_tensor * ssm_state = cache.ssm_state[dn_idx];
            ggml_tensor * conv_state = cache.conv_state[dn_idx];
            ggml_tensor * ssm_inter = cache.ssm_intermediate[dn_idx];
            ggml_tensor * conv_input = cache.conv_input_cache[dn_idx];
            if (!ssm_state || !conv_state || !ssm_inter || !conv_input) return fail("missing_capture_storage");
            if (rollback_idx >= (int)ssm_inter->ne[3]) return fail("rollback_idx_oob");

            const size_t ssm_elems = (size_t)ssm_state->ne[0] *
                (size_t)ssm_state->ne[1] * (size_t)ssm_state->ne[2];
            const void * ssm_src = (const char *)ssm_inter->data +
                (size_t)rollback_idx * ssm_inter->nb[3];
            if (ssm_inter->type == GGML_TYPE_F32) {
                ce = cudaMemcpyAsync(ssm_state->data, ssm_src,
                                     ssm_elems * sizeof(float),
                                     cudaMemcpyDeviceToDevice, shard_stream);
                if (ce != cudaSuccess) return fail("ssm_f32_copy_failed", ce);
            } else {
                const auto to_fp32 = ggml_get_to_fp32_cuda(ssm_inter->type);
                if (!to_fp32) return fail("missing_ssm_converter");
                to_fp32(ssm_src, (float *)ssm_state->data, (int64_t)ssm_elems, shard_stream);
                ce = cudaPeekAtLastError();
                if (ce != cudaSuccess) return fail("ssm_convert_launch_failed", ce);
            }

            const int K_conv = w.ssm_d_conv;
            if (commit_n + K_conv - 1 > (int)conv_input->ne[0]) return fail("conv_input_oob");
            const int row_cnt = (int)conv_input->ne[1];
            const size_t elt = ggml_element_size(conv_input);
            const size_t dpitch = (size_t)(K_conv - 1) * elt;
            const size_t spitch = conv_input->nb[1];
            const size_t width = (size_t)(K_conv - 1) * elt;
            const void * conv_src = (const char *)conv_input->data + (size_t)commit_n * elt;
            ce = cudaMemcpy2DAsync(conv_state->data, dpitch,
                                   conv_src, spitch,
                                   width, row_cnt,
                                   cudaMemcpyDeviceToDevice, shard_stream);
            if (ce != cudaSuccess) return fail("conv_copy_failed", ce);
            restored_per_shard[si]++;
            dn_idx++;
        }
        cudaError_t sync = cudaStreamSynchronize(shard_stream);
        if (sync != cudaSuccess) return fail("stream_sync_failed", sync);
    }
    if (prior_device >= 0) (void)cudaSetDevice(prior_device);

    for (auto & shard : shards_) shard.cache.cur_pos = base_pos + commit_n;
    const auto t1 = std::chrono::steady_clock::now();
    const auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    if (diag) {
        std::fprintf(stderr,
            "[target-split][chain-rollback] split_chain_fast_rollback_attempts=1 split_chain_fast_rollback_success=1 split_chain_fast_rollback_fail=0 split_chain_fast_rollback_fallback_restore_replay=0 split_chain_rollback_commit_n=%d split_chain_rollback_slot_idx=%d split_chain_rollback_latency_us=%lld split_chain_rollback_cuda_error_name=cudaSuccess split_chain_rollback_failure_class=recoverable-return split_chain_rollback_context_fatal=0 split_chain_feature_ring_action=none_required_position_aligned split_tree_verify_supported=0 split_rollback_to_tree_supported=0\n",
            commit_n, rollback_idx, (long long)latency_us);
        for (size_t si = 0; si < restored_per_shard.size(); ++si) {
            std::fprintf(stderr,
                "[target-split][chain-rollback] shard=%zu gpu=%d split_chain_rollback_layers_restored_per_shard=%d\n",
                si, shards_[si].gpu, restored_per_shard[si]);
        }
    }
    return true;
}

bool Qwen35LayerSplitDFlashTarget::supports_tree_verify() const {
    // Production fail-closed boundary: sibling visibility and depth positions
    // are not wired for layer-split execution. Evidence overlays may expose a
    // separate test-only seam, but ordinary runtime must never advertise this.
    return false;
}

bool Qwen35LayerSplitDFlashTarget::verify_tree(
        int committed,
        const DDTree & tree,
        const std::vector<int32_t> & flat_tokens,
        int n_alloc,
        std::vector<int32_t> & posterior_out,
        std::vector<float> * logits_out) {
    auto fail_precheck = [&](const char * reason) {
        std::fprintf(stderr,
            "[target-split][pure-chain-guard] verify_precheck_fail=%s committed=%d n_alloc=%d n_actual=%d n_nodes=%d parents_size=%zu flat_tokens_size=%zu\n",
            reason, committed, n_alloc, 1 + tree.n_nodes, tree.n_nodes,
            tree.parents.size(), flat_tokens.size());
        return false;
    };
    const int n_actual = 1 + tree.n_nodes;
    if (n_actual <= 0) return fail_precheck("n_actual_nonpositive");
    if ((int)tree.parents.size() < n_actual) return fail_precheck("parents_size_lt_n_actual");
    if (!qwen35_split_run_if_root_inclusive_pure_chain(
            tree.parents.data(), tree.parents.size(), (size_t)n_actual)) {
        return fail_precheck("not_root_inclusive_pure_chain");
    }
    if (rollback_poisoned_) return fail_precheck("rollback_poisoned");
    if (!supports_tree_verify()) return fail_precheck("production_capability_disabled");
    if (n_alloc < n_actual) return fail_precheck("n_alloc_lt_n_actual");
    if ((int)flat_tokens.size() < n_actual) return fail_precheck("flat_tokens_size_lt_n_actual");
    if ((int)tree.token_ids.size() < tree.n_nodes) return fail_precheck("token_ids_size_lt_n_nodes");
    if ((int)tree.depths.size() < tree.n_nodes) return fail_precheck("depths_size_lt_n_nodes");
    if (tree.visibility.size() < (size_t)n_actual * (size_t)n_actual) {
        return fail_precheck("visibility_size_lt_n_actual_sq");
    }
    if (committed < 0) return fail_precheck("negative_committed");
    required_tree_slots_ = n_alloc;

    // Padded graph reuse is retained only behind the disabled production seam.
    // Padding is embedded as zeros and cannot enter posterior decisions.
    std::vector<int32_t> padded_tokens((size_t)n_alloc, 0);
    std::copy(flat_tokens.begin(), flat_tokens.begin() + n_actual, padded_tokens.begin());

    std::vector<int32_t> parent_ids((size_t)n_alloc, 0);
    parent_ids[0] = -1;
    for (int s = 1; s < n_actual; ++s) {
        const int pflat = tree.parents[(size_t)s];
        if (pflat < 0 || pflat >= s) return fail_precheck("parent_value_oob");
        parent_ids[(size_t)s] = (int32_t)pflat;
    }
    std::vector<float> host_act((size_t)n_alloc * (size_t)hidden_size(), 0.0f);
    if (!embed_tokens(padded_tokens.data(), n_actual, host_act.data())) return fail_precheck("embed_tokens_failed");

    ActivationPair acts;
    if (!activation_pair_init(acts, shards_.front().backend, hidden_size(), n_alloc)) {
        return fail_precheck("activation_pair_init_failed");
    }
    ggml_backend_tensor_set(acts.a, host_act.data(), 0,
                            sizeof(float) * host_act.size());

    Qwen35SplitTreeInputs tree_inputs;
    tree_inputs.parent_ids = parent_ids.data();
    tree_inputs.visibility = nullptr;
    tree_inputs.n_actual = n_alloc;
    tree_inputs.committed = committed;

    int last_tok = -1;
    std::vector<int32_t> all_argmax;
    std::vector<Qwen35TargetCaptureSlice> tree_captures;
    const bool ok = run_qwen35_layer_split_tree_verify_from_activation(
        shards_, acts, committed, n_alloc, n_alloc, last_tok,
        kq_stride_pad_, fa_window_, tree_inputs, &all_argmax, logits_out,
        &tree_captures, kvflash_, /*kvflash_preallocated=*/false);
    activation_pair_free(acts);
    if (!ok) return fail_precheck("tree_verify_from_activation_failed");
    if ((int)all_argmax.size() < n_actual) return fail_precheck("all_argmax_size_lt_n_actual");

    posterior_out.assign(all_argmax.begin(), all_argmax.begin() + n_actual);
    if (logits_out && !logits_out->empty()) {
        const int vocab = shards_.back().weights.n_vocab;
        if ((int)logits_out->size() >= n_alloc * vocab) {
            logits_out->resize((size_t)n_actual * (size_t)vocab);
        }
    }
    return true;
}

bool Qwen35LayerSplitDFlashTarget::rollback_to_tree(
        int committed,
        const DDTree & tree,
        const std::vector<int> & accepted_dfs) {
    const int n_actual_ri = 1 + tree.n_nodes;
    if (n_actual_ri <= 0 ||
        !qwen35_split_run_if_root_inclusive_pure_chain(
            tree.parents.data(), tree.parents.size(), (size_t)n_actual_ri)) {
        std::fprintf(stderr,
            "[target-split][pure-chain-guard] rollback_precheck_fail=not_root_inclusive_pure_chain n_actual=%d parents_size=%zu\n",
            n_actual_ri, tree.parents.size());
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool diag = std::getenv("DFLASH_SPLIT_CHAIN_ROLLBACK_DIAG") != nullptr;
    last_rollback_context_fatal_ = false;
    enum class Phase { AValidate, BRestore, CBarrier1, DCompact, EBarrier2, FCommit };
    auto phase_name = [](Phase p) -> const char * {
        switch (p) {
            case Phase::AValidate: return "A_validate_all";
            case Phase::BRestore:  return "B_restore_recurrent_all";
            case Phase::CBarrier1: return "C_barrier_1";
            case Phase::DCompact:  return "D_compact_kv_and_feature_all";
            case Phase::EBarrier2: return "E_barrier_2";
            case Phase::FCommit:   return "F_commit_advance";
        }
        return "unknown";
    };
    Phase phase = Phase::AValidate;
    auto fail = [&](const char * why, cudaError_t err = cudaSuccess,
                    int shard_idx = -1, int layer_id = -1, int dn_idx = -1) {
        const bool has_cuda_error = err != cudaSuccess;
        const bool context_fatal = has_cuda_error && split_rollback_context_fatal(err);
        last_rollback_context_fatal_ = context_fatal || phase != Phase::AValidate;
        if (phase != Phase::AValidate) rollback_poisoned_ = true;
        if (diag) {
            std::fprintf(stderr,
                "[target-split][pure-chain-rollback] split_rollback_to_tree_fail=1 phase=%s reason=%s shard=%d layer=%d dn_idx=%d cuda_error_name=%s split_rollback_to_tree_context_fatal=%d split_rollback_to_tree_poisoned=%d\n",
                phase_name(phase), why, shard_idx, layer_id, dn_idx,
                has_cuda_error ? cudaGetErrorName(err) : "cudaSuccess",
                context_fatal ? 1 : 0, last_rollback_context_fatal_ ? 1 : 0);
        }
        return false;
    };

    if (rollback_poisoned_) return fail("rollback_poisoned");
    if (!supports_tree_verify()) return fail("unsupported_or_not_validated");
    if (remote_target_shard_ && remote_target_shard_->active()) return fail("remote_target_shard_active");
    if (shards_.empty()) return fail("no_shards");
    if (committed < 0) return fail("negative_committed");
    const int commit_n = (int)accepted_dfs.size();
    if (commit_n <= 0) return fail("commit_n_nonpositive");
    if ((int)tree.parents.size() < n_actual_ri) return fail("parents_size_lt_n_actual");
    if (tree.parents[0] != -1) return fail("parents_root_not_minus1");
    // Parents use the root-inclusive flat-slot contract: slot 0 is the
    // synthetic root and each later value names its parent flat slot.
    auto parent_flat_slot = [&](int flat_slot) -> int {
        if (flat_slot == 0) return -1;
        if (flat_slot < 0 || flat_slot > tree.n_nodes) return -2;
        const int pflat = tree.parents[(size_t)flat_slot];
        if (pflat < 0 || pflat >= flat_slot) return -2;
        return pflat;
    };
    if (accepted_dfs[0] != 0) return fail("accepted_dfs_not_starting_at_root");
    for (int d = 0; d < commit_n; ++d) {
        const int dfs_idx = accepted_dfs[(size_t)d];
        if (dfs_idx < 0 || dfs_idx > tree.n_nodes) return fail("accepted_dfs_oob");
        if (d > 0) {
            if (accepted_dfs[(size_t)d] <= accepted_dfs[(size_t)d - 1]) {
                return fail("accepted_dfs_not_strictly_increasing");
            }
            int cur = dfs_idx;
            const int prev_flat = accepted_dfs[(size_t)d - 1];
            bool parent_found = false;
            for (int guard = 0; guard <= tree.n_nodes && cur >= 0; ++guard) {
                if (cur == prev_flat) { parent_found = true; break; }
                cur = parent_flat_slot(cur);
            }
            if (!parent_found) return fail("accepted_path_not_parent_chain");
        }
    }
    // accepted_dfs is root-inclusive, so the deepest accepted flat slot is
    // used directly with no index conversion.
    const int rollback_dfs = accepted_dfs.back();
    const bool walked_sibling = [&]() {
        for (int i = 0; i < commit_n; ++i) if (accepted_dfs[(size_t)i] != i) return true;
        return false;
    }();


    struct OwnedDelta { size_t si; int layer_id; int dn_idx; };
    std::vector<OwnedDelta> owned;
    owned.reserve(64);
    for (size_t si = 0; si < shards_.size(); ++si) {
        const auto & shard = shards_[si];
        const auto & cache = shard.cache;
        const auto & w = shard.weights;
        int dn_idx = 0;
        for (int il = 0; il < w.n_layer; ++il) {
            const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
            if (is_attn) continue;
            const bool owns_layer = il >= shard.layer_begin && il < shard.layer_end;
            if (!owns_layer) { dn_idx++; continue; }
            if (dn_idx >= (int)cache.ssm_state.size() || dn_idx >= (int)cache.conv_state.size() ||
                dn_idx >= (int)cache.ssm_intermediate.size() || dn_idx >= (int)cache.conv_input_cache.size()) {
                return fail("capture_index_oob", cudaSuccess, (int)si, il, dn_idx);
            }
            ggml_tensor * ssm_state = cache.ssm_state[(size_t)dn_idx];
            ggml_tensor * conv_state = cache.conv_state[(size_t)dn_idx];
            ggml_tensor * ssm_inter = cache.ssm_intermediate[(size_t)dn_idx];
            ggml_tensor * conv_input = cache.conv_input_cache[(size_t)dn_idx];
            if (!ssm_state || !conv_state || !ssm_inter || !conv_input) {
                return fail("missing_capture_storage", cudaSuccess, (int)si, il, dn_idx);
            }
            if (ssm_inter->type != GGML_TYPE_F32 || conv_input->type != GGML_TYPE_F32 ||
                ssm_state->type != GGML_TYPE_F32 || conv_state->type != GGML_TYPE_F32) {
                return fail("non_f32_recurrent_storage", cudaSuccess, (int)si, il, dn_idx);
            }
            // Both operands are root-inclusive flat-slot domain: ne[3] was
            // filled by tree token t directly into persistent slot t.
            if (rollback_dfs >= (int)ssm_inter->ne[3]) {
                return fail("rollback_dfs_oob", cudaSuccess, (int)si, il, dn_idx);
            }
            const int K_conv = w.ssm_d_conv;
            if (K_conv <= 1 || conv_input->ne[0] < K_conv ||
                conv_state->ne[0] < K_conv - 1 ||
                conv_input->ne[1] != conv_state->ne[1]) {
                return fail("conv_shape_invalid", cudaSuccess, (int)si, il, dn_idx);
            }
            if (!walked_sibling) {
                if (rollback_dfs + K_conv > (int)conv_input->ne[0]) {
                    return fail("conv_contiguous_oob", cudaSuccess, (int)si, il, dn_idx);
                }
            } else {
                int virt = rollback_dfs;
                for (int k = K_conv - 2; k >= 0; --k) {
                    const int sx_slot = (K_conv - 1) + virt;
                    if (sx_slot < 0 || sx_slot >= (int)conv_input->ne[0]) {
                        return fail("conv_ancestry_oob", cudaSuccess, (int)si, il, dn_idx);
                    }
                    virt = (virt >= 0) ? parent_flat_slot(virt) : (virt - 1);
                }
            }
            owned.push_back({si, il, dn_idx});
            dn_idx++;
        }
    }
    if (owned.empty()) return fail("no_owned_delta_layers");

    if (feature_ring_) {
        if (!feature_ring_->target_feat || feature_ring_->cap <= 0 ||
            commit_n > feature_ring_->cap || committed < 0) {
            return fail("feature_ring_precondition_failed");
        }
    }

    int prior_device = -1;
    (void)cudaGetDevice(&prior_device);
    std::vector<int> restored_per_shard(shards_.size(), 0);

    phase = Phase::BRestore;
    for (const auto & od : owned) {
        auto & shard = shards_[od.si];
        auto & cache = shard.cache;
        const auto & w = shard.weights;
        cudaError_t ce = cudaSetDevice(shard.gpu);
        if (ce != cudaSuccess) return fail("cuda_set_device_failed", ce, (int)od.si, od.layer_id, od.dn_idx);
        const cudaStream_t stream = cudaStreamPerThread;
        ggml_tensor * ssm_state = cache.ssm_state[(size_t)od.dn_idx];
        ggml_tensor * conv_state = cache.conv_state[(size_t)od.dn_idx];
        ggml_tensor * ssm_inter = cache.ssm_intermediate[(size_t)od.dn_idx];
        ggml_tensor * conv_input = cache.conv_input_cache[(size_t)od.dn_idx];
        const size_t ssm_elems = (size_t)ssm_state->ne[0] *
            (size_t)ssm_state->ne[1] * (size_t)ssm_state->ne[2];
        // Exact-domain restore: root-inclusive flat slot -> root-inclusive
        // ssm_intermediate ne[3] slot. I2 probes neighbors empirically.
        const void * ssm_src = (const char *)ssm_inter->data +
            (size_t)rollback_dfs * ssm_inter->nb[3];
        ce = cudaMemcpyAsync(ssm_state->data, ssm_src, ssm_elems * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) return fail("ssm_copy_failed", ce, (int)od.si, od.layer_id, od.dn_idx);

        const int K_conv = w.ssm_d_conv;
        const int row_cnt = (int)conv_input->ne[1];
        const size_t elt = ggml_element_size(conv_input);
        const size_t dpitch = (size_t)(K_conv - 1) * elt;
        const size_t spitch = conv_input->nb[1];
        if (!walked_sibling) {
            // conv_input row domain is [K_conv-1 prefix | root-inclusive
            // verify rows]: flat slot t is physical row (K_conv-1)+t, so the
            // K_conv-1 history rows ending at accepted flat slot rollback_dfs
            // are physical rows rollback_dfs+1 .. rollback_dfs+K_conv-1
            // (directive 168 §1B; not an isolated conv-row change — it is
            // re-derived together with the corrected rollback_dfs domain).
            const void * conv_src = (const char *)conv_input->data + (size_t)(rollback_dfs + 1) * elt;
            ce = cudaMemcpy2DAsync(conv_state->data, dpitch, conv_src, spitch,
                                   (size_t)(K_conv - 1) * elt, row_cnt,
                                   cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) return fail("conv_contiguous_copy_failed", ce, (int)od.si, od.layer_id, od.dn_idx);
        } else {
            std::vector<int> virt((size_t)(K_conv - 1));
            int cur = rollback_dfs;
            for (int k = K_conv - 2; k >= 0; --k) {
                virt[(size_t)k] = cur;
                cur = (cur >= 0) ? parent_flat_slot(cur) : (cur - 1);
            }
            for (int k = 0; k < K_conv - 1; ++k) {
                // virt is root-inclusive flat-slot domain; K_conv-1 converts
                // it to the corresponding row in conv_input's concatenation.
                const int sx_slot = (K_conv - 1) + virt[(size_t)k];
                const void * src_col = (const char *)conv_input->data + (size_t)sx_slot * elt;
                char * dst_col = (char *)conv_state->data + (size_t)k * elt;
                ce = cudaMemcpy2DAsync(dst_col, dpitch, src_col, spitch, elt, row_cnt,
                                       cudaMemcpyDeviceToDevice, stream);
                if (ce != cudaSuccess) return fail("conv_ancestry_copy_failed", ce, (int)od.si, od.layer_id, od.dn_idx);
            }
        }
        restored_per_shard[od.si]++;
    }

    phase = Phase::CBarrier1;
    for (size_t si = 0; si < shards_.size(); ++si) {
        cudaError_t ce = cudaSetDevice(shards_[si].gpu);
        if (ce != cudaSuccess) return fail("cuda_set_device_failed", ce, (int)si);
        ce = cudaStreamSynchronize(cudaStreamPerThread);
        if (ce != cudaSuccess) return fail("stream_sync_failed", ce, (int)si);
    }

    phase = Phase::DCompact;
    for (size_t si = 0; si < shards_.size(); ++si) {
        auto & shard = shards_[si];
        auto & cache = shard.cache;
        cudaError_t ce = cudaSetDevice(shard.gpu);
        if (ce != cudaSuccess) return fail("cuda_set_device_failed", ce, (int)si);
        const cudaStream_t stream = cudaStreamPerThread;
        for (int d = 0; d < commit_n; ++d) {
            // Root-inclusive accepted flat slot s was written at committed+s
            // during the tree pass (root row at committed+0); no +1
            // conversion (directive 168 §1B).
            const int src_dfs = accepted_dfs[(size_t)d];
            for (size_t l = 0; l < cache.attn_k.size(); ++l) {
                ggml_tensor * ck = cache.attn_k[l];
                ggml_tensor * cv = l < cache.attn_v.size() ? cache.attn_v[l] : nullptr;
                if (!ck || !cv) continue;
                const int src_pos = committed + src_dfs;
                const int dst_pos = committed + d;
                if (src_pos < 0 || dst_pos < 0 ||
                    src_pos >= (int)ck->ne[1] || dst_pos >= (int)ck->ne[1] ||
                    src_pos >= (int)cv->ne[1] || dst_pos >= (int)cv->ne[1]) {
                    return fail("kv_compact_oob", cudaSuccess, (int)si, -1, (int)l);
                }
                const int n_kv = (int)ck->ne[2];
                for (int h = 0; h < n_kv; ++h) {
                    const size_t k_bytes = ck->nb[1];
                    const size_t v_bytes = cv->nb[1];
                    const size_t k_src = (size_t)src_pos * ck->nb[1] + (size_t)h * ck->nb[2];
                    const size_t k_dst = (size_t)dst_pos * ck->nb[1] + (size_t)h * ck->nb[2];
                    const size_t v_src = (size_t)src_pos * cv->nb[1] + (size_t)h * cv->nb[2];
                    const size_t v_dst = (size_t)dst_pos * cv->nb[1] + (size_t)h * cv->nb[2];
                    ce = cudaMemcpyAsync((char *)ck->data + k_dst,
                                         (const char *)ck->data + k_src,
                                         k_bytes, cudaMemcpyDeviceToDevice, stream);
                    if (ce != cudaSuccess) return fail("kv_k_copy_failed", ce, (int)si, -1, (int)l);
                    ce = cudaMemcpyAsync((char *)cv->data + v_dst,
                                         (const char *)cv->data + v_src,
                                         v_bytes, cudaMemcpyDeviceToDevice, stream);
                    if (ce != cudaSuccess) return fail("kv_v_copy_failed", ce, (int)si, -1, (int)l);
                }
            }
        }
    }

    long long feature_realign_latency_us = 0;
    if (feature_ring_) {
        const auto feature_t0 = std::chrono::steady_clock::now();
        // One N+1 staging buffer: accepted-DFS rows plus one guard row, then
        // read back in committed-spine order.
        // The helper reads and writes by logical position, matching
        // draft_feature_mirror_sync_range(). This prewindow path uses host F32
        // staging for inspectable exactness; a device-staging fast path is an
        // explicit follow-up optimization, not claimed here.
        const int fc_in = feature_ring_->n_target_layers * feature_ring_->hidden_size;
        if (fc_in <= 0) return fail("feature_ring_width_invalid");
        const int stage_rows = commit_n + 1;
        if (stage_rows > feature_ring_->cap) return fail("feature_stage_rows_exceed_cap");
        std::vector<float> source_stage((size_t)stage_rows * (size_t)fc_in, 0.0f);
        std::vector<float> one_row;
        for (int d = 0; d < commit_n; ++d) {
            // Root-inclusive accepted flat slot: feature row for accepted
            // element d lives at committed+accepted_dfs[d]; no +1 (168 §1B).
            if (!copy_feature_ring_range_to_host_f32(*feature_ring_, committed + accepted_dfs[(size_t)d], 1, one_row)) {
                return fail("feature_source_stage_copy_failed");
            }
            if ((int)one_row.size() != fc_in) return fail("feature_source_stage_width_mismatch");
            std::copy(one_row.begin(), one_row.end(),
                      source_stage.begin() + (size_t)d * (size_t)fc_in);
        }
        if (!copy_host_f32_to_feature_ring_range(*feature_ring_, committed, commit_n, source_stage)) {
            return fail("feature_destination_backfill_failed");
        }
        const auto feature_t1 = std::chrono::steady_clock::now();
        feature_realign_latency_us = (long long)std::chrono::duration_cast<std::chrono::microseconds>(
            feature_t1 - feature_t0).count();
    }

    phase = Phase::EBarrier2;
    for (size_t si = 0; si < shards_.size(); ++si) {
        cudaError_t ce = cudaSetDevice(shards_[si].gpu);
        if (ce != cudaSuccess) return fail("cuda_set_device_failed", ce, (int)si);
        ce = cudaStreamSynchronize(cudaStreamPerThread);
        if (ce != cudaSuccess) return fail("stream_sync_failed", ce, (int)si);
    }
    if (feature_ring_) {
        cudaError_t ce = cudaSetDevice(feature_ring_->device);
        if (ce != cudaSuccess) return fail("feature_cuda_set_device_failed", ce);
        ce = cudaDeviceSynchronize();
        if (ce != cudaSuccess) return fail("feature_device_sync_failed", ce);
    }

    phase = Phase::FCommit;
    if (kvflash_ && !kvflash_->alloc_span(committed, commit_n)) {
        return fail("kvflash_alloc_span_failed");
    }
    for (auto & shard : shards_) {
        shard.cache.cur_pos = committed + commit_n;
        shard.cache.last_tok = -1;
    }
    if (prior_device >= 0) (void)cudaSetDevice(prior_device);
    const auto t1 = std::chrono::steady_clock::now();
    const auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    if (diag) {
        std::fprintf(stderr,
            "[target-split][pure-chain-rollback] split_rollback_to_tree_success=1 split_rollback_to_tree_commit_n=%d split_rollback_to_tree_rollback_dfs=%d split_rollback_to_tree_walked_sibling=%d split_rollback_to_tree_latency_us=%lld split_rollback_to_tree_feature_realign=%d split_rollback_to_tree_post_verify_required=1\n",
            commit_n, rollback_dfs, walked_sibling ? 1 : 0,
            (long long)latency_us, feature_ring_ ? 1 : 0);
        std::fprintf(stderr,
            "[target-split][pure-chain-rollback] split_rollback_to_tree_feature_realign_latency_us=%lld split_rollback_to_tree_feature_staging=device_followup_host_f32_prewindow split_rollback_to_tree_feature_stage_rows=%d\n",
            feature_realign_latency_us, feature_ring_ ? (commit_n + 1) : 0);
        for (size_t si = 0; si < restored_per_shard.size(); ++si) {
            std::fprintf(stderr,
                "[target-split][pure-chain-rollback] shard=%zu gpu=%d split_rollback_to_tree_layers_restored_per_shard=%d\n",
                si, shards_[si].gpu, restored_per_shard[si]);
        }
    }
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
    if (remote_target_shard_ && remote_target_shard_->active()) {
        return remote_target_shard_->project_hidden_to_tokens(hidden, n_tokens,
                                                              tokens_out);
    }

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

}  // namespace dflash::common
