// dflash_spec_decode.cpp — Generic DFlash speculative-decode loop.

#include "dflash_spec_decode.h"

#include "internal.h"        // DraftWeights
#include "io_utils.h"
#include "dflash_draft_graph.h"  // build_draft_step
#include "dspark_head.h"
#include "adaptive_verify_width.h"
#include "step_graph.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "chain_rollback_policy.h"

namespace dflash::common {

namespace {
// RAII guard so any early `return false` path frees the per-call draft graph.
struct StepGraphGuard {
    StepGraph & sg;
    ~StepGraphGuard() { step_graph_destroy(sg); }
};
}  // namespace

bool run_dflash_spec_decode(
        DFlashTarget & target,
        DraftWeights & draft_weights,
        ggml_backend_t draft_backend,
        DraftFeatureMirror & feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int last_tok,
        const char * out_path,
        int draft_ctx_max,
        int stream_fd,
        DFlashDraftIpcClient * remote_draft,
        const std::vector<int32_t> * hint_tokens,
        int base_pos) {
    DaemonIO io;
    io.stream_fd = stream_fd;
    return run_dflash_spec_decode(target, draft_weights, draft_backend,
                                  feature_ring, prompt, n_gen, last_tok,
                                  out_path, draft_ctx_max, io,
                                  remote_draft, hint_tokens, base_pos);
}

bool run_dflash_spec_decode(
        DFlashTarget & target,
        DraftWeights & draft_weights,
        ggml_backend_t draft_backend,
        DraftFeatureMirror & feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int last_tok,
        const char * out_path,
        int draft_ctx_max,
        const DaemonIO & io,
        DFlashDraftIpcClient * remote_draft,
        const std::vector<int32_t> * hint_tokens,
        int base_pos,
        double * accept_rate_out) {
    const bool use_remote_draft = remote_draft && remote_draft->active();
    if (!use_remote_draft && !feature_ring.target_feat) return false;

    const int hidden = draft_weights.n_embd;
    const int draft_block_size = draft_weights.block_size;
    const bool dspark_block = draft_weights.dspark.enabled;
    const int max_verify_width = draft_weights.max_chain_verify_tokens();
    if (hidden <= 0 || draft_block_size <= 0 || max_verify_width <= 0) return false;
    const float width_theta = adaptive_verify_width_theta();
    const int target_width_min = target.default_adaptive_verify_min_rows();
    const int width_min = adaptive_verify_width_min(target_width_min);

    StepGraph draft_sg;
    StepGraphGuard draft_sg_guard{draft_sg};

    std::vector<float>   noise_embed((size_t)hidden * draft_block_size);
    std::vector<int32_t> noise_ids(draft_block_size);
    std::vector<int32_t> draft_tok(max_verify_width);
    std::vector<int32_t> target_tok(max_verify_width);
    std::vector<int32_t> pos_q(draft_block_size);
    std::vector<int32_t> pos_k;
    std::vector<float>   local_hidden;       // host buffer for local draft hidden states
    std::vector<float>   remote_hidden;      // host buffer for remote-draft hidden states

    std::vector<int32_t> out_all = prompt;
    int committed       = base_pos + (int)prompt.size();
    int n_generated     = 0;
    int n_draft_steps   = 0;
    int n_accept_sum    = 0;
    int n_verify_rows   = 0;
    int n_hint_proposed = 0;
    int n_hint_accepted = 0;
    const ChainRollbackPolicy rollback_policy = resolve_chain_rollback_policy();
    RollbackDiag rollback_diag;

    auto t_dec0 = std::chrono::steady_clock::now();
    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;
        int q_len = max_verify_width;

        // ── Build noise input for draft ────────────────────────────────────
        noise_ids[0] = last_tok;
        for (int i = 1; i < draft_block_size; i++) {
            noise_ids[i] = target.mask_token_id();
        }
        if (!target.embed_tokens(
                noise_ids.data(), draft_block_size, noise_embed.data())) {
            std::fprintf(stderr, "dflash-spec noise embed failed\n");
            return false;
        }

        constexpr int DRAFT_CTX_MAX_DEFAULT = 2048;
        const int ring_cap = use_remote_draft ? remote_draft->ring_cap() : feature_ring.cap;
        const int draft_ctx = std::min(committed, std::min(ring_cap,
            std::max(DRAFT_CTX_MAX_DEFAULT, draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            !use_remote_draft &&
            draft_feature_mirror_can_view(feature_ring, committed, draft_ctx, mirror_slot0);

        // ── Draft compute (local or remote) ───────────────────────────────
        const float * draft_hidden_host = nullptr;
        if (use_remote_draft) {
            if (!remote_draft->propose(committed, draft_ctx, noise_embed, remote_hidden)) {
                std::fprintf(stderr, "dflash-spec remote draft propose failed\n");
                return false;
            }
            draft_hidden_host = remote_hidden.data();
        } else {
            if (!build_draft_step(draft_sg, draft_weights, /*lm_head=*/nullptr, draft_backend,
                                  draft_ctx, use_mirror_view ? &feature_ring : nullptr,
                                  committed,
                                  /*ctx_len_max=*/std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, draft_ctx_max)))) {
                std::fprintf(stderr, "dflash-spec draft build failed\n");
                return false;
            }
            if (!use_mirror_view &&
                !copy_feature_ring_range_to_tensor(feature_ring, draft_sg.target_hidden_cat,
                                                   draft_start, draft_ctx)) {
                std::fprintf(stderr, "dflash-spec draft feature copy failed\n");
                return false;
            }
            ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                    sizeof(float) * noise_embed.size());
            pos_k.resize((size_t)draft_ctx + draft_block_size);
            for (int i = 0; i < draft_block_size; i++) pos_q[i] = draft_ctx + i;
            for (int i = 0; i < draft_ctx + draft_block_size; i++) pos_k[i] = i;
            ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                    sizeof(int32_t) * pos_q.size());
            ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                    sizeof(int32_t) * pos_k.size());
            auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "dflash-spec draft compute %d\n", (int)st);
                return false;
            }
            // Read draft hidden states out to host so the target adapter can
            // project them through its own LM head (target-internal layout).
            local_hidden.resize((size_t)hidden * draft_block_size);
            ggml_backend_tensor_get(draft_sg.hidden_states, local_hidden.data(), 0,
                                    sizeof(float) * local_hidden.size());
            draft_hidden_host = local_hidden.data();
        }

        // ── Project draft hidden → token IDs via the shared draft head ────
        // DSpark is an auxiliary head on the universal DFlash checkpoint, not
        // a target-model implementation.  Keep its Markov correction and
        // confidence policy here so Kimi, Qwen, Laguna, and future adapters
        // share exactly one implementation.
        std::vector<float> confidence;
        std::vector<float> candidate_probs;
        std::vector<int32_t> candidate_ids;
        std::vector<int32_t> proposals;
        bool projected = false;
        if (draft_backend && dspark_block) {
            ggml_backend_t head_backend = target.fused_head_backend();
            const bool same_device_head =
                head_backend && target.lm_head_tensor() &&
                ggml_backend_get_device(head_backend) ==
                    ggml_backend_get_device(draft_backend);
            if (same_device_head) {
                projected = dspark_markov_propose_greedy_block_fused(
                    draft_weights, draft_backend, target.lm_head_tensor(),
                    draft_hidden_host, draft_block_size, last_tok, proposals,
                    &confidence);
            }
            if (!projected) {
                projected = dspark_markov_propose_greedy_block(
                    draft_weights, draft_backend, target,
                    draft_hidden_host, draft_block_size, last_tok, proposals);
            }
        }
        if (dspark_block && !projected) {
            // Preserve the DSpark block contract even if the auxiliary head
            // cannot execute: every hidden row still predicts a proposal.
            projected = target.project_hidden_to_tokens(
                draft_hidden_host, draft_block_size, proposals);
        }
        if (dspark_block && projected) {
            if (proposals.size() < static_cast<size_t>(draft_block_size)) {
                projected = false;
            } else {
                draft_tok.clear();
                draft_tok.reserve(static_cast<size_t>(max_verify_width));
                draft_tok.push_back(last_tok);
                draft_tok.insert(
                    draft_tok.end(), proposals.begin(),
                    proposals.begin() + draft_block_size);
            }
        } else if (!dspark_block) {
            const int candidate_k =
                width_theta > 0.0f && draft_block_size > 2 ? 2 : 0;
            projected = target.project_hidden_to_tokens_topk(
                draft_hidden_host, draft_block_size, draft_tok, candidate_k,
                candidate_k > 0 ? &candidate_probs : nullptr,
                candidate_k > 0 ? &candidate_ids : nullptr);
            if (projected &&
                draft_tok.size() >= static_cast<size_t>(draft_block_size)) {
                draft_tok[0] = last_tok;
            }
        }
        if (!projected ||
            draft_tok.size() < static_cast<size_t>(max_verify_width)) {
            std::fprintf(stderr, "dflash-spec projection failed\n");
            return false;
        }

        if (dspark_block &&
            confidence.size() >= static_cast<size_t>(draft_block_size)) {
            q_len = adaptive_verify_width(
                confidence.data(), 1, max_verify_width,
                width_theta, width_min);
        } else if (candidate_probs.size() >=
                   static_cast<size_t>((max_verify_width - 1) * 2)) {
            q_len = adaptive_verify_width(
                candidate_probs.data(), 2, max_verify_width,
                width_theta, width_min);
        }

        // ── Tool call hint injection ──────────────────────────────────────
        // Override draft tokens with pre-known hint tokens for near-100%
        // acceptance on predictable structural positions.
        int hint_filled = 0;
        if (hint_tokens && n_generated < (int)hint_tokens->size()) {
            const int hint_avail = (int)hint_tokens->size() - n_generated;
            q_len = max_verify_width;
            hint_filled = std::min(hint_avail, max_verify_width - 1);
            for (int i = 0; i < hint_filled; i++) {
                draft_tok[1 + i] = (*hint_tokens)[n_generated + i];
            }
        }

        // Never verify rows that cannot fit in the remaining generation
        // budget.  The draft graph remains fixed-width for graph reuse; this
        // only trims expensive target/MoE work.
        q_len = std::max(1, std::min(q_len, need_commit_budget));
        draft_tok.resize(static_cast<size_t>(q_len));
        target_tok.resize(static_cast<size_t>(q_len));

        // Notify observer with draft tokens for this step.
        if (io.observer) {
            io.observer("draft", draft_tok);
        }

        // ── Verify pass: speculative target forward over q_len tokens ────
        if (!target.snapshot_kv()) {
            std::fprintf(stderr, "dflash-spec snapshot_kv failed\n");
            return false;
        }

        int verify_last_tok = -1;
        if (!target.verify_batch(draft_tok, committed, verify_last_tok, &target_tok,
                                  /*capture_ssm_intermediates=*/true)) {
            std::fprintf(stderr, "dflash-spec verify failed\n");
            // Roll the snapshot back so we don't leak the speculative KV
            // mutations into the caller's target cache.
            if (!target.restore_kv()) {
                std::fprintf(stderr, "dflash-spec restore_kv after verify failure failed\n");
            }
            return false;
        }

        // Acceptance: longest matching prefix between draft and target argmax.
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        // Track hint acceptance telemetry.
        if (hint_filled > 0) {
            n_hint_proposed += hint_filled;
            n_hint_accepted += std::min(hint_filled, accept_n - 1);
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n  = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        // ── Commit accepted tokens to KV state ──────────────────────────
        // Adaptive: use fast-rollback when acceptance is high enough to benefit.
        rollback_diag.record_accept(accept_n);
        const bool use_fast_rollback =
            target.supports_fast_rollback() &&
            (target.prefer_fast_rollback_over_replay() ||
             accept_n >= rollback_policy.fast_rollback_threshold);

        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }
        // Never commit hidden state past EOS when a verify batch happens to
        // accept additional candidates after it.
        for (int i = 0; i < commit_n; ++i) {
            if (target.is_eos(replay_tok[(size_t)i])) {
                commit_n = i + 1;
                accept_n = std::min(accept_n, commit_n);
                bonus_tok = -1;
                replay_tok.resize((size_t)commit_n);
                break;
            }
        }

        bool fast_rolled_back = false;
        if (use_fast_rollback) {
            // Fast rollback: restore SSM from intermediates, skip replay.
            // Implicit bonus: deferred to next step as draft_tok[0].
            // Respect the generation budget: accept_n can exceed the remaining
            // budget (need_commit_budget). Committing accept_n would both
            // overrun the budget and grow replay_tok with zero-initialised
            // tokens (it was sized to the clamped commit_n above).
            bonus_tok = -1;
            commit_n = std::min(accept_n, need_commit_budget);
            replay_tok.resize(commit_n);
            if (target.rollback_to(committed, commit_n)) {
                last_tok = target_tok[commit_n - 1];
                fast_rolled_back = true;
                rollback_diag.record_fast_rollback(accept_n);
            } else {
                // Rollback failed (e.g. CUDA error / unsupported state type).
                // The pre-verify snapshot is still valid, so degrade to the
                // legacy restore+replay path below instead of aborting.
                std::fprintf(stderr, "dflash-spec rollback_to failed; "
                                     "falling back to restore+replay\n");
                rollback_diag.record_failed_fallback();
            }
        }
        if (!fast_rolled_back) {
            rollback_diag.record_legacy_replay();
            // Legacy path: restore SSM snapshot and replay accepted + bonus tokens.
            // (When falling back from fast-rollback, bonus_tok is already -1 and
            //  replay_tok/commit_n reflect the budget-clamped accepted set.)
            if (!target.restore_kv()) {
                std::fprintf(stderr, "dflash-spec restore_kv failed\n");
                return false;
            }
            int replay_last_tok = -1;
            if (!target.verify_batch(replay_tok, committed, replay_last_tok, nullptr)) {
                std::fprintf(stderr, "dflash-spec replay failed\n");
                return false;
            }
            last_tok = replay_last_tok;
        }

        bool hit_eos = false;
        int emitted = 0;
        for (int i = 0; i < commit_n; i++) {
            out_all.push_back(replay_tok[i]);
            io.emit(replay_tok[i]);
            if (io.cancelled) break;
            ++emitted;
            if (target.is_eos(replay_tok[i])) {
                hit_eos = true;
                break;
            }
        }
        committed   += emitted;
        n_generated += emitted;
        n_accept_sum += std::min(accept_n, emitted);
        n_verify_rows += q_len;
        n_draft_steps++;

        // Notify observer with accepted tokens for this step.
        if (io.observer) {
            io.observer("verify", replay_tok);
        }

        if (io.cancelled) break;
        if (hit_eos) break;
    }
    if (!use_remote_draft && draft_backend) ggml_backend_synchronize(draft_backend);
    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_verify_rows);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    if (accept_rate_out) {
        *accept_rate_out = total_draft_pos > 0
            ? (double)n_accept_sum / (double)total_draft_pos : 0.0;
    }
    std::printf("[target-split-dflash] decode tokens=%d time=%.3f s speed=%.2f tok/s\n",
                n_generated, decode_s, n_generated > 0 ? n_generated / decode_s : 0.0);
    std::printf("[target-split-dflash] %d draft steps, accepted=%d/%d (%.1f%%), avg commit/step=%.2f\n",
                n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);
    rollback_diag.print(rollback_policy, stdout);
    if (n_hint_proposed > 0) {
        std::printf("[target-split-dflash] hint tokens: %d/%d accepted (%.1f%%)\n",
                    n_hint_accepted, n_hint_proposed,
                    100.0 * (double)n_hint_accepted / (double)n_hint_proposed);
    }
    if (out_path) write_int32_file(out_path, out_all);

    return true;
}

} // namespace dflash::common
