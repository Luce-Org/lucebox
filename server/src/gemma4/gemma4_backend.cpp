// Gemma4Backend implementation.
//
// Uses gemma4_step() for forward pass with per-layer embedding support.
// Structure mirrors Qwen3Backend: prefill in chunks, autoregressive decode,
// KV cache with layer sharing, snapshot/restore.

#include "gemma4_backend.h"
#include "dflash27b.h"
#include "common/sampler.h"
#include "common/io_utils.h"
#include "common/dflash_feature_ring.h"
#include "common/dflash_draft_graph.h"
#include "common/step_graph.h"

#include "ggml-cuda.h"
#include "common/snapshot_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>

namespace dflash::common {

// ── Ctor / dtor ────────────────────────────────────────────────────────

Gemma4Backend::Gemma4Backend(const Gemma4BackendConfig & cfg)
    : cfg_(cfg) {}

Gemma4Backend::~Gemma4Backend() { shutdown(); }

bool Gemma4Backend::init() {
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[gemma4] CUDA backend init failed (gpu=%d)\n", cfg_.device.gpu);
        return false;
    }

    snap_backend_ = create_snapshot_backend(backend_);
    if (!snap_backend_) {
        std::fprintf(stderr, "[gemma4] snapshot backend init failed\n");
        ggml_backend_free(backend_); backend_ = nullptr;
        return false;
    }

    if (!load_gemma4_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[gemma4] GGUF load failed: %s\n",
                     dflash27b_last_error());
        return false;
    }

    if (!create_gemma4_cache(backend_, w_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[gemma4] cache alloc failed\n");
        return false;
    }
    cache_.fa_window = cfg_.fa_window;

    // Load draft model for speculative decode
    if (cfg_.draft_path) {
        const int draft_gpu = (cfg_.draft_gpu >= 0) ? cfg_.draft_gpu : cfg_.device.gpu;
        draft_backend_ = ggml_backend_cuda_init(draft_gpu);
        if (!draft_backend_) {
            std::fprintf(stderr, "[gemma4] draft CUDA init failed (gpu=%d)\n", draft_gpu);
        } else {
            // Load draft GGUF — pass nullptr for target (Gemma4 != TargetWeights)
            if (!load_draft_gguf(cfg_.draft_path, draft_backend_, dw_, nullptr)) {
                std::fprintf(stderr, "[gemma4] draft load failed: %s\n", dflash27b_last_error());
                ggml_backend_free(draft_backend_); draft_backend_ = nullptr;
            } else {
                // Override mask_token_id for Gemma4 (token 4 per model card)
                dw_.mask_token_id = 4;

                // Fix draft dimensions from actual tensor shapes (GGUF metadata is wrong)
                // fc.weight: [fc_in, draft_hidden]
                const int draft_hidden = (int)dw_.fc->ne[1];
                const int fc_in = (int)dw_.fc->ne[0];
                const int n_capture = fc_in / w_.n_embd;

                if (draft_hidden != dw_.n_embd) {
                    std::printf("[gemma4] draft: overriding n_embd %d -> %d (from fc weight)\n",
                                dw_.n_embd, draft_hidden);
                    dw_.n_embd = draft_hidden;
                }
                // Infer n_head from wq shape: wq.ne[1] = n_head * head_dim
                if (dw_.n_layer > 0 && dw_.layers[0].wq) {
                    const int q_dim = (int)dw_.layers[0].wq->ne[1];
                    const int inferred_n_head = q_dim / dw_.head_dim;
                    if (inferred_n_head != dw_.n_head) {
                        std::printf("[gemma4] draft: overriding n_head %d -> %d\n",
                                    dw_.n_head, inferred_n_head);
                        dw_.n_head = inferred_n_head;
                    }
                }
                // Infer n_ff from ffn_gate shape
                if (dw_.n_layer > 0 && dw_.layers[0].w_gate) {
                    const int inferred_ff = (int)dw_.layers[0].w_gate->ne[1];
                    if (inferred_ff != dw_.n_ff) {
                        std::printf("[gemma4] draft: overriding n_ff %d -> %d\n",
                                    dw_.n_ff, inferred_ff);
                        dw_.n_ff = inferred_ff;
                    }
                }
                // Override n_target_layers from fc shape
                dw_.n_target_layers = n_capture;

                // Gemma4 DFlash draft: layers 0-3 are SWA (causal), layer 4 is full (non-causal)
                // (from model card: layer_types = [sliding*4, full_attention])
                dw_.swa_window = 2048;
                for (int i = 0; i < dw_.n_layer - 1 && i < (int)dw_.layers.size(); i++)
                    dw_.layers[i].is_swa = true;

                std::printf("[gemma4] draft loaded: fc_in=%d target_hidden=%d "
                            "draft_hidden=%d n_capture_layers=%d swa=%d\n",
                            fc_in, w_.n_embd, draft_hidden, n_capture, dw_.swa_window);

                // Allocate target_feat ring buffer
                constexpr int TARGET_FEAT_CAP = 4096;
                const int feat_cap = std::min(cfg_.device.max_ctx, TARGET_FEAT_CAP);
                if (!create_gemma4_target_feat(backend_, cache_, n_capture, w_.n_embd, feat_cap)) {
                    std::fprintf(stderr, "[gemma4] target_feat alloc failed\n");
                } else {
                    // Init feature mirror on draft GPU
                    const int mirror_cap = std::min(cfg_.draft_ctx_max, feat_cap);
                    if (!draft_feature_mirror_init(feature_mirror_, draft_backend_,
                                                   draft_gpu, cfg_.device.gpu, mirror_cap,
                                                   n_capture, w_.n_embd)) {
                        std::fprintf(stderr, "[gemma4] feature mirror init failed\n");
                    } else {
                        // Create DFlash target adapter
                        dflash_target_ = new Gemma4DFlashTarget(w_, cache_, backend_);
                        std::printf("[gemma4] spec-decode ready: capture_layers=%d mirror_cap=%d\n",
                                    n_capture, mirror_cap);
                        std::printf("[gemma4] capture_layer_ids:");
                        for (int k = 0; k < (int)cache_.capture_layer_ids.size(); k++)
                            std::printf(" %d", cache_.capture_layer_ids[k]);
                        std::printf("\n");
                    }
                }
            }
        }
    }

    std::printf("[gemma4] init ok: %d layers, embd=%d, vocab=%d, max_ctx=%d\n",
                w_.n_layer, w_.n_embd, w_.n_vocab, cfg_.device.max_ctx);
    std::fflush(stdout);
    return true;
}

void Gemma4Backend::print_ready_banner() const {
    std::printf("[gemma4-daemon] READY (layers=%d, embd=%d, experts=%d/%d, "
                "swa=%d, ctx=%d)\n",
                w_.n_layer, w_.n_embd, w_.n_expert_used, w_.n_expert,
                w_.sliding_window, cfg_.device.max_ctx);
    std::fflush(stdout);
}

// ── Park / Unpark ──────────────────────────────────────────────────────

bool Gemma4Backend::park(const std::string & what) {
    (void)what;
    if (parked_) return true;

    // Free snapshots first (they reference the snap_backend buffer)
    for (int i = 0; i < PREFIX_SLOTS; ++i) {
        free_gemma4_snapshot(snapshots_[i]);
    }

    // Free KV cache (GPU memory)
    free_gemma4_cache(cache_);

    // Free model weights (GPU memory)
    free_gemma4_weights(w_);

    parked_ = true;
    std::printf("[gemma4] parked (VRAM released)\n"); std::fflush(stdout);
    return true;
}

bool Gemma4Backend::unpark(const std::string & what) {
    (void)what;
    if (!parked_) return true;

    // Reload weights from disk
    if (!load_gemma4_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[gemma4] unpark: failed to reload weights\n");
        return false;
    }

    // Recreate KV cache
    if (!create_gemma4_cache(backend_, w_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[gemma4] unpark: failed to recreate cache\n");
        free_gemma4_weights(w_);
        return false;
    }

    parked_ = false;
    std::printf("[gemma4] unparked (VRAM restored)\n"); std::fflush(stdout);
    return true;
}

// ── Prefill ────────────────────────────────────────────────────────────

int Gemma4Backend::do_prefill(const std::vector<int32_t> & tokens,
                               const DaemonIO & io, int kv_offset) {
    (void)io;
    const int n = (int)tokens.size();
    const int hidden = w_.n_embd;
    const int chunk = cfg_.chunk;

    std::vector<float> embed(chunk * hidden);
    std::vector<float> logits;

    int pos = 0;
    while (pos < n) {
        int len = std::min(chunk, n - pos);

        // Limit chunk to avoid ring-buffer wrap for SWA layers
        if (cache_.swa_size > 0 && cache_.swa_size < cache_.max_ctx) {
            const int swa_remaining = cache_.swa_size - ((kv_offset + pos) % cache_.swa_size);
            len = std::min(len, swa_remaining);
        }

        // Embed tokens using CPU embedder
        w_.embedder.embed(tokens.data() + pos, len, embed.data());

        // Gemma4 scales embeddings by sqrt(n_embd)
        float scale = std::sqrt((float)hidden);
        for (int i = 0; i < len * hidden; ++i) embed[i] *= scale;

        const int kv_pos = kv_offset + pos;
        if (!gemma4_step(backend_, w_, cache_, embed.data(),
                         tokens.data() + pos, len, kv_pos, logits)) {
            std::fprintf(stderr, "[gemma4] prefill step failed at pos=%d\n", kv_pos);
            return -1;
        }

        pos += len;
        cache_.cur_pos = kv_offset + pos;

        // Store last_tok from final chunk's logits (argmax of last position)
        if (pos >= n && !logits.empty()) {
            int32_t best_tok = 0;
            float best_val = logits[0];
            for (int j = 1; j < (int)logits.size(); ++j) {
                if (logits[j] > best_val) { best_val = logits[j]; best_tok = j; }
            }
            cache_.last_tok = best_tok;
        }

        // Sync captured features to draft mirror
        if (feature_mirror_.target_feat && cache_.target_feat && !draft_parked_) {
            draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, kv_pos, len);
        }
    }

    return kv_offset + pos;
}

// ── Decode ─────────────────────────────────────────────────────────────

bool Gemma4Backend::do_decode(int committed, int n_gen,
                               std::vector<int32_t> & out_tokens,
                               const DaemonIO & io,
                               const BudgetHook & budget_hook,
                               bool * forced_close_out) {
    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> embed_buf(hidden);
    std::vector<float> logits;

    // Budget force-close state — same shape as qwen35's maybe_force_close.
    // See dflash/src/common/model_backend.h BudgetHook docs for the
    // single- vs multi-token close-tag semantics.
    bool budget_close_started = false;
    int  close_inject_pos     = 0;
    // Capture entry KV position so the budget check is in the
    // "generated since entry" frame, not the absolute KV frame.
    // committed_now (KV position) = prompt_len + tokens_generated; n_gen
    // is the gen-only count (or remaining-budget remap from the
    // spec-decode tail-off). Without this entry capture, force-close
    // fires prompt_len tokens early on prompted requests and goes
    // negative immediately after a tail-off. (Mirrors qwen35 fix 5c785f0
    // — same bug since this lambda was ported verbatim.)
    const int committed_at_entry = committed;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size())
        {
            int32_t inj = budget_hook.close_token_ids[close_inject_pos];
            std::fprintf(stderr,
                "[budget-hook] gemma4 close-seq continue %d/%zu: overriding "
                "sampled token %d with %d\n",
                close_inject_pos + 1,
                budget_hook.close_token_ids.size(), tok, inj);
            tok = inj;
            close_inject_pos++;
            return;
        }
        if (budget_close_started) return;
        const int generated = committed_now - committed_at_entry;
        int remaining = n_gen - generated;
        if (remaining <= budget_hook.hard_limit_remaining) {
            int32_t first_close = budget_hook.close_token_ids.front();
            if (tok == first_close) {
                budget_close_started = true;
                close_inject_pos = 1;
                return;
            }
            std::fprintf(stderr,
                "[budget-hook] gemma4 force-close at committed=%d/%d "
                "(remaining=%d <= hard_limit=%d): overriding token %d "
                "with close[0]=%d (seq len %zu)\n",
                committed_now, n_gen, remaining,
                budget_hook.hard_limit_remaining, tok, first_close,
                budget_hook.close_token_ids.size());
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            if (forced_close_out) *forced_close_out = true;
        }
    };

    for (int i = 0; i < n_gen; ++i) {
        // Seed for this iteration's embed step:
        //  - Normal case: previous iteration just pushed a sampled
        //    token onto out_tokens; we re-embed it to advance KV +
        //    produce next-token logits.
        //  - Empty case (spec-decode tail-off at iter 0): no prior
        //    iteration ran, so use cache_.last_tok — that's the
        //    prefill argmax that spec-decode would have consumed as
        //    its initial seed. Mirrors qwen35's initial_emitted=1
        //    pattern; without this, out_tokens.back() on an empty
        //    vector is UB. (Codex r2 P2 follow-up: the previous fix
        //    pushed last_tok onto out_tokens here in the caller, but
        //    that grew out_tokens by an uncounted extra token and the
        //    caller's `result.tokens.size()` over-counted against the
        //    budget. Reading from cache instead keeps the budget
        //    honest.)
        int32_t tok = out_tokens.empty() ? cache_.last_tok : out_tokens.back();

        // Embed single token
        w_.embedder.embed(&tok, 1, embed_buf.data());
        float scale = std::sqrt((float)hidden);
        for (int j = 0; j < hidden; ++j) embed_buf[j] *= scale;

        if (!gemma4_step(backend_, w_, cache_, embed_buf.data(),
                         &tok, 1, committed, logits)) {
            return false;
        }

        // Sample
        int32_t next;
        if (sampler_.needs_logit_processing()) {
            next = sample_logits(logits.data(), vocab, sampler_,
                                 out_tokens, sampler_rng_);
        } else {
            next = 0;
            float best = logits[0];
            for (int j = 1; j < vocab; ++j) {
                if (logits[j] > best) { best = logits[j]; next = j; }
            }
        }
        maybe_force_close(next, committed);

        out_tokens.push_back(next);
        io.emit(next);
        committed++;
        cache_.cur_pos = committed;
        if (io.cancelled) break;

        // Check EOS
        if (next == w_.eos_id || next == w_.eos_chat_id) break;
    }

    return true;
}

// ── Speculative Decode ─────────────────────────────────────────────────

bool Gemma4Backend::do_spec_decode(int committed, int n_gen,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io,
                                    const BudgetHook * budget_hook,
                                    bool * forced_close_out) {
    const int hidden = w_.n_embd;
    int32_t last_tok = cache_.last_tok;

    DFlashTarget * target = dflash_target_;
    const int q_len = dw_.block_size;

    StepGraph draft_sg;

    std::vector<float>   noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<float>   local_hidden;

    int n_generated     = 0;
    int n_draft_steps   = 0;
    int n_accept_sum    = 0;

    auto t_dec0 = std::chrono::steady_clock::now();

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        // Budget tail-off: when remaining budget is within one spec-decode
        // batch of the force-close threshold, hand off to do_decode for the
        // tail. AR handles the close-token override cleanly; spec-decode's
        // verify-and-accept loop can't safely inject a token mid-batch
        // without rewriting KV.
        //
        // Gemma4's do_decode reads `out_tokens.back()` as the seed each
        // iter. After the first spec-decode iteration the most-recently-
        // committed token is on out_tokens, but on a small-budget request
        // (budget_tokens <= reply_budget + q_len) tail-off can fire on
        // iter 0 before out_tokens has been seeded. Codex review flagged
        // the resulting UB on out_tokens.back(); we set cache_.last_tok
        // and let do_decode pick it up when out_tokens is empty.
        //
        // Budget accounting (codex r2 P2): in the previous patch we
        // also push_back'd last_tok before calling do_decode, which
        // grew out_tokens by an extra token outside the budget — the
        // caller (http_server) then saw `result.tokens.size() ==
        // need_commit_budget + 1` and double-counted that seed against
        // the budget. Mirror qwen35 instead: cache the seed via
        // cache_.last_tok, leave out_tokens untouched, and have
        // do_decode read the seed from cache when out_tokens is empty
        // (initial_emitted=1 path below). That keeps the budget honest
        // and matches the symmetry between qwen35 and gemma4 backends.
        if (budget_hook && !budget_hook->close_token_ids.empty()) {
            int hard = budget_hook->hard_limit_remaining;
            if (need_commit_budget <= hard + q_len) {
                std::fprintf(stderr,
                    "[budget-hook] gemma4 spec-decode tail-off at "
                    "committed=%d/%d (remaining=%d, hard_limit=%d, "
                    "batch=%d) — switching to AR\n",
                    committed, n_gen, need_commit_budget, hard, q_len);
                step_graph_destroy(draft_sg);
                cache_.last_tok = last_tok;  // do_decode reads this when out_tokens empty
                BudgetHook tail_hook = *budget_hook;
                int ar_n_gen = need_commit_budget;
                bool ok = do_decode(committed, ar_n_gen, out_tokens, io,
                                    tail_hook, forced_close_out);
                auto t_dec1 = std::chrono::steady_clock::now();
                const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
                const int total_draft_pos = std::max(1, n_draft_steps * q_len);
                const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
                std::fprintf(stderr,
                    "[gemma4-spec] tail-off-stats tokens=%d time=%.3f s "
                    "speed=%.2f tok/s steps=%d accepted=%d/%d (%.1f%%)\n",
                    n_generated, decode_s,
                    n_generated > 0 ? n_generated / decode_s : 0.0,
                    n_draft_steps, n_accept_sum, total_draft_pos,
                    accept_pct);
                io.emit(-1);
                return ok;
            }
        }

        // 1. Build noise input: [last_tok, MASK, MASK, ..., MASK]
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = target->mask_token_id();
        if (!target->embed_tokens(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "[gemma4-spec] noise embed failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        // 2. Draft compute
        constexpr int DRAFT_CTX_MAX_DEFAULT = 2048;
        const int ring_cap = feature_mirror_.cap;
        const int draft_ctx = std::min(committed,
            std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_mirror_, committed, draft_ctx, mirror_slot0);

        if (!build_draft_step(draft_sg, dw_, /*lm_head=*/nullptr, draft_backend_,
                              draft_ctx, use_mirror_view ? &feature_mirror_ : nullptr,
                              committed,
                              std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)))) {
            std::fprintf(stderr, "[gemma4-spec] draft build failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_mirror_, draft_sg.target_hidden_cat,
                                               draft_start, draft_ctx)) {
            std::fprintf(stderr, "[gemma4-spec] feature copy failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());
        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * pos_k.size());

        auto st = ggml_backend_graph_compute(draft_backend_, draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[gemma4-spec] draft compute failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        // Read draft hidden states
        local_hidden.resize((size_t)hidden * q_len);
        ggml_backend_tensor_get(draft_sg.hidden_states, local_hidden.data(), 0,
                                sizeof(float) * local_hidden.size());

        // 3. Project draft hidden → token IDs via target LM head
        if (!target->project_hidden_to_tokens(local_hidden.data(), q_len, draft_tok)) {
            std::fprintf(stderr, "[gemma4-spec] projection failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        draft_tok[0] = last_tok;

        // 4. Verify: run target forward over all draft tokens.
        // Gemma4 is a pure transformer — after verify, KV entries at accepted
        // positions are already correct (causal masking guarantees independence
        // from rejected tokens at later positions). We use KV truncation instead
        // of the expensive snapshot/restore/replay approach.
        int verify_last_tok = -1;
        if (!target->verify_batch(draft_tok, committed, verify_last_tok, &target_tok)) {
            std::fprintf(stderr, "[gemma4-spec] verify failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        // 5. Acceptance: longest matching prefix
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n  = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        // 6. KV truncation: discard rejected positions, keep accepted.
        // Accepted positions 0..accept_n-1 already have correct KV from verify.
        cache_.cur_pos = committed + accept_n;

        // If there's a bonus token, run a 1-token forward to get its KV + features.
        if (bonus_tok >= 0) {
            std::vector<int32_t> bonus_vec = {bonus_tok};
            int bonus_last = -1;
            if (!target->verify_batch(bonus_vec, committed + accept_n, bonus_last, nullptr)) {
                std::fprintf(stderr, "[gemma4-spec] bonus forward failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            last_tok = bonus_last;
        } else {
            last_tok = verify_last_tok;
        }

        // 7. Sync features from verify (positions 0..accept_n-1 are correct)
        // and from bonus forward (position accept_n, if present).
        if (feature_mirror_.target_feat && cache_.target_feat) {
            draft_feature_mirror_sync_range(cache_.target_feat, cache_.target_feat_cap,
                                            feature_mirror_, committed, commit_n);
        }

        // 8. Emit committed tokens
        bool hit_eos = false;
        int emitted = 0;
        for (int i = 0; i < commit_n; i++) {
            int tok = (i < accept_n) ? draft_tok[i] : bonus_tok;
            out_tokens.push_back(tok);
            io.emit(tok);
            emitted++;
            if (io.cancelled) break;
            if (tok == w_.eos_id || tok == w_.eos_chat_id) {
                hit_eos = true; break;
            }
        }
        committed   += emitted;
        cache_.cur_pos = committed;
        n_generated += emitted;
        n_accept_sum += std::min(accept_n, emitted);
        n_draft_steps++;
        if (io.cancelled) break;
        if (hit_eos) break;
    }

    step_graph_destroy(draft_sg);

    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_draft_steps * q_len);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    std::fprintf(stderr, "[gemma4-spec] tokens=%d time=%.3f s speed=%.2f tok/s "
                 "steps=%d accepted=%d/%d (%.1f%%) avg_commit=%.2f\n",
                 n_generated, decode_s,
                 n_generated > 0 ? n_generated / decode_s : 0.0,
                 n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                 n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);

    io.emit(-1);
    return true;
}

// ── Generate ───────────────────────────────────────────────────────────

GenerateResult Gemma4Backend::generate(const GenerateRequest & req,
                                        const DaemonIO & io) {
    GenerateResult result;
    if (parked_) { result.error = "model is parked"; return result; }

    DaemonIO out_io = io.with_token_callback(req.on_token);
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    cache_.cur_pos = 0;

    const int committed = do_prefill(req.prompt, out_io, /*kv_offset=*/0);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }

    // Inline snapshot at snap_pos for prefix cache
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= committed) {
        cache_.cur_pos = req.snap_pos;
        if (snapshot_save(req.snap_slot)) {
            std::fprintf(stderr, "[gemma4] inline-snap slot=%d cur_pos=%d\n",
                         req.snap_slot, req.snap_pos);
        }
        cache_.cur_pos = committed;
    }

    if (req.n_gen > 0) {
        // Try speculative decode if draft is available and temp==0
        const bool can_spec = dflash_target_
            && !draft_parked_
            && feature_mirror_.target_feat
            && !sampler_.needs_logit_processing();

        if (can_spec) {
            if (!do_spec_decode(committed, req.n_gen, result.tokens, out_io,
                                &req.budget_hook,
                                &result.budget_forced_close)) {
                result.error = "spec_decode";
                return result;
            }
        } else {
            const int hidden = w_.n_embd;
            const int vocab  = w_.n_vocab;
            std::vector<float> logits;

            // Re-step last token to get logits
            int32_t last_tok = req.prompt.back();
            std::vector<float> embed_buf(hidden);
            w_.embedder.embed(&last_tok, 1, embed_buf.data());
            float scale = std::sqrt((float)hidden);
            for (int j = 0; j < hidden; ++j) embed_buf[j] *= scale;

            if (!gemma4_step(backend_, w_, cache_, embed_buf.data(),
                             &last_tok, 1, committed - 1, logits)) {
                result.error = "first logits";
                return result;
            }

            // Sample first token
            int32_t first;
            if (sampler_.needs_logit_processing()) {
                first = sample_logits(logits.data(), vocab, sampler_,
                                       result.tokens, sampler_rng_);
            } else {
                first = 0;
                float best = logits[0];
                for (int j = 1; j < vocab; ++j) {
                    if (logits[j] > best) { best = logits[j]; first = j; }
                }
            }
            result.tokens.push_back(first);
            out_io.emit(first);
            if (out_io.cancelled) {
                out_io.emit(-1);
                result.ok = true;
                return result;
            }

            if (first == w_.eos_id || first == w_.eos_chat_id) {
                out_io.emit(-1);
                result.ok = true;
                return result;
            }

            if (req.n_gen > 1) {
                if (!do_decode(committed, req.n_gen - 1, result.tokens, out_io,
                               req.budget_hook,
                               &result.budget_forced_close)) {
                    result.error = "decode";
                    return result;
                }
            }
            out_io.emit(-1);
        }
    } else {
        out_io.emit(-1);
    }
    result.ok = true;
    return result;
}

// ── Restore + Generate ─────────────────────────────────────────────────

GenerateResult Gemma4Backend::restore_and_generate(int slot,
                                                     const GenerateRequest & req,
                                                     const DaemonIO & io) {
    GenerateResult result;
    if (parked_) { result.error = "model is parked"; return result; }

    DaemonIO out_io = io.with_token_callback(req.on_token);

    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].ctx) {
        result.error = "bad slot";
        out_io.emit(-1);
        return result;
    }

    const auto & snap = snapshots_[slot];
    // Restore snapshot into cache per-head (cache: [D, cache_len, Hk]).
    for (int il = 0; il < cache_.n_layer; ++il) {
        if (cache_.k[il] && snap.k_snap[il]) {
            ggml_tensor * ck = cache_.k[il];
            const int D   = (int)ck->ne[0];
            const int Hk  = (int)ck->ne[2];
            const int cache_len = (int)ck->ne[1];
            const int save_pos = (int)snap.k_snap[il]->ne[1];  // min(snap.cur_pos, cache_len)
            const size_t elem_sz = ggml_element_size(ck);
            const size_t head_bytes_src = (size_t)D * save_pos * elem_sz;
            const size_t head_bytes_dst = (size_t)D * cache_len * elem_sz;
            const size_t copy_bytes     = head_bytes_src;

            for (int h = 0; h < Hk; ++h) {
                ggml_backend_tensor_set(cache_.k[il],
                    (const char *)snap.k_snap[il]->data + h * head_bytes_src,
                    h * head_bytes_dst, copy_bytes);
                ggml_backend_tensor_set(cache_.v[il],
                    (const char *)snap.v_snap[il]->data + h * head_bytes_src,
                    h * head_bytes_dst, copy_bytes);
            }
        }
    }

    // Restore target_feat from snapshot
    if (snap.feat_snap && cache_.target_feat) {
        const size_t feat_nbytes = ggml_nbytes(snap.feat_snap);
        ggml_backend_tensor_set(cache_.target_feat, snap.feat_snap->data, 0, feat_nbytes);
    }

    const int snap_pos = snap.cur_pos;
    cache_.cur_pos = snap_pos;
    cache_.last_tok = snap.last_tok;

    // Set up sampler
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    // Diff-prefill: only prefill tokens beyond the cached prefix
    const int prompt_len = (int)req.prompt.size();
    int committed = snap_pos;

    if (prompt_len > snap_pos) {
        // Compute delta (tokens after the snapshot)
        std::vector<int32_t> delta(req.prompt.begin() + snap_pos, req.prompt.end());
        committed = do_prefill(delta, out_io, /*kv_offset=*/snap_pos);
        if (committed < 0) {
            result.error = "prefill";
            return result;
        }
    } else if (prompt_len > 0 && prompt_len < snap_pos) {
        result.error = "snapshot_longer_than_prompt";
        out_io.emit(-1);
        return result;
    }
    // else: prompt_len == snap_pos → no delta, committed stays at snap_pos

    // Inline snapshot at snap_pos for prefix cache (new snap from this request)
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= committed) {
        cache_.cur_pos = req.snap_pos;
        if (snapshot_save(req.snap_slot)) {
            std::fprintf(stderr, "[gemma4] inline-snap slot=%d cur_pos=%d\n",
                         req.snap_slot, req.snap_pos);
        }
        cache_.cur_pos = committed;
    }

    // Full feature mirror resync after restore: do_prefill only synced the
    // delta [snap_pos..committed). Re-sync the entire [0..committed) range so
    // the draft model sees correct features for the full context.
    if (feature_mirror_.target_feat && cache_.target_feat && !draft_parked_ && committed > 0) {
        draft_feature_mirror_sync_tail(cache_.target_feat, cache_.target_feat_cap,
                                       feature_mirror_, committed);
    }

    // Generate
    if (req.n_gen > 0) {
        const bool can_spec = dflash_target_
            && !draft_parked_
            && feature_mirror_.target_feat
            && sampler_.temp == 0.0f;

        if (can_spec) {
            if (!do_spec_decode(committed, req.n_gen, result.tokens, out_io,
                                &req.budget_hook,
                                &result.budget_forced_close)) {
                result.error = "spec_decode";
                return result;
            }
        } else {
            const int hidden = w_.n_embd;
            const int vocab  = w_.n_vocab;
            std::vector<float> logits;

            // Re-step last token to get logits for first generated token
            int32_t last_tok = req.prompt.back();
            std::vector<float> embed_buf(hidden);
            w_.embedder.embed(&last_tok, 1, embed_buf.data());
            float scale = std::sqrt((float)hidden);
            for (int j = 0; j < hidden; ++j) embed_buf[j] *= scale;

            if (!gemma4_step(backend_, w_, cache_, embed_buf.data(),
                             &last_tok, 1, committed - 1, logits)) {
                result.error = "first logits";
                return result;
            }

            // Sample first token
            int32_t first;
            if (sampler_.temp > 0) {
                first = sample_logits(logits.data(), vocab, sampler_,
                                       result.tokens, sampler_rng_);
            } else {
                first = 0;
                float best = logits[0];
                for (int j = 1; j < vocab; ++j) {
                    if (logits[j] > best) { best = logits[j]; first = j; }
                }
            }
            result.tokens.push_back(first);
            out_io.emit(first);
            if (out_io.cancelled) {
                out_io.emit(-1);
                result.ok = true;
                return result;
            }

            if (first == w_.eos_id || first == w_.eos_chat_id) {
                out_io.emit(-1);
                result.ok = true;
                return result;
            }

            if (req.n_gen > 1) {
                if (!do_decode(committed, req.n_gen - 1, result.tokens, out_io,
                               req.budget_hook,
                               &result.budget_forced_close)) {
                    result.error = "decode";
                    return result;
                }
            }
            out_io.emit(-1);
        }
    } else {
        out_io.emit(-1);
    }
    result.ok = true;
    return result;
}

// ── Snapshots ──────────────────────────────────────────────────────────

bool Gemma4Backend::snapshot_save(int slot) {
    if (parked_) return false;
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;

    auto & snap = snapshots_[slot];
    const int n_layer = cache_.n_layer;
    const int snap_pos = cache_.cur_pos;
    if (snap_pos <= 0) return false;

    // Reuse buffer if shapes match (same cur_pos); otherwise reallocate.
    const bool needs_alloc = (snap.ctx == nullptr) || (snap.cur_pos != snap_pos);
    if (needs_alloc) {
        free_gemma4_snapshot(snap);

        const int n_feat_tensors = (cache_.target_feat && cache_.target_feat_cap > 0) ? 1 : 0;
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + n_feat_tensors + 4) + 4096;
        ip.no_alloc = true;
        snap.ctx = ggml_init(ip);
        if (!snap.ctx) return false;

        snap.k_snap.resize(n_layer, nullptr);
        snap.v_snap.resize(n_layer, nullptr);
        for (int il = 0; il < n_layer; ++il) {
            if (cache_.k[il]) {
                ggml_tensor * ck = cache_.k[il];
                const int cache_len = (int)ck->ne[1];
                // Save min(snap_pos, cache_len) positions
                const int save_pos = std::min(snap_pos, cache_len);
                snap.k_snap[il] = ggml_new_tensor_3d(snap.ctx, ck->type,
                                                      ck->ne[0], save_pos, ck->ne[2]);
                snap.v_snap[il] = ggml_new_tensor_3d(snap.ctx, ck->type,
                                                      ck->ne[0], save_pos, ck->ne[2]);
            }
        }

        // target_feat: save min(snap_pos, target_feat_cap) positions
        snap.feat_snap = nullptr;
        snap.feat_cap  = 0;
        if (cache_.target_feat && cache_.target_feat_cap > 0) {
            const int feat_len = std::min(snap_pos, cache_.target_feat_cap);
            snap.feat_snap = ggml_new_tensor_2d(snap.ctx, cache_.target_feat->type,
                                                 cache_.target_feat->ne[0], feat_len);
            snap.feat_cap = cache_.target_feat_cap;
        }

        snap.buf = ggml_backend_alloc_ctx_tensors(snap.ctx, snap_backend_);
        if (!snap.buf) {
            ggml_free(snap.ctx); snap.ctx = nullptr;
            snap.k_snap.clear(); snap.v_snap.clear();
            snap.feat_snap = nullptr;
            return false;
        }
    }

    // Copy valid positions per head.
    // Cache: [D, cache_len, Hk], Snap: [D, save_pos, Hk]
    for (int il = 0; il < n_layer; ++il) {
        if (cache_.k[il] && snap.k_snap[il]) {
            ggml_tensor * ck = cache_.k[il];
            const int D   = (int)ck->ne[0];
            const int Hk  = (int)ck->ne[2];
            const int cache_len = (int)ck->ne[1];
            const int save_pos = std::min(snap_pos, cache_len);
            const size_t elem_sz = ggml_element_size(ck);
            const size_t head_bytes_src = (size_t)D * cache_len * elem_sz;
            const size_t head_bytes_dst = (size_t)D * save_pos * elem_sz;
            const size_t copy_bytes     = head_bytes_dst;

            for (int h = 0; h < Hk; ++h) {
                ggml_backend_tensor_get(cache_.k[il],
                    (char *)snap.k_snap[il]->data + h * head_bytes_dst,
                    h * head_bytes_src, copy_bytes);
                ggml_backend_tensor_get(cache_.v[il],
                    (char *)snap.v_snap[il]->data + h * head_bytes_dst,
                    h * head_bytes_src, copy_bytes);
            }
        }
    }
    snap.cur_pos = snap_pos;
    snap.last_tok = cache_.last_tok;

    // target_feat: copy min(snap_pos, cap) positions from GPU to snapshot
    if (snap.feat_snap && cache_.target_feat) {
        const size_t feat_nbytes = ggml_nbytes(snap.feat_snap);
        ggml_backend_tensor_get(cache_.target_feat, snap.feat_snap->data, 0, feat_nbytes);
    }

    std::fprintf(stderr, "[gemma4] snapshot saved slot=%d pos=%d\n", slot, snap.cur_pos);
    return true;
}

void Gemma4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_gemma4_snapshot(snapshots_[slot]);
}

bool Gemma4Backend::snapshot_used(int slot) const {
    return slot >= 0 && slot < PREFIX_SLOTS && snapshots_[slot].ctx != nullptr;
}

int Gemma4Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].ctx) return 0;
    return snapshots_[slot].cur_pos;
}

// ── Compress / drafter ─────────────────────────────────────────────────

bool Gemma4Backend::handle_compress(const std::string & line,
                                     const DaemonIO & io) {
    // Check for "nopark" suffix
    bool skip_park = (line.size() >= 16 &&
                      line.compare(line.size() - 7, 7, " nopark") == 0);

    // Parse: "compress <path> <keep_x1000> <drafter_gguf> [nopark]"
    char ppath[1024];
    int  keep_x1000 = 0;
    char drafter_path[1024] = {0};
    const int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s",
                               ppath, &keep_x1000, drafter_path);
    if (n < 2) {
        std::fprintf(stderr, "[compress] bad args\n");
        io.emit(-1);
        return false;
    }

    const char * dpath = (n >= 3 && drafter_path[0])
        ? drafter_path
        : "/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf";

    // Park target to free VRAM for the drafter (unless skip_park).
    const bool was_parked = parked_;
    if (!skip_park && !parked_) {
        park("target");
    }

    // Synchronize backend
    ggml_backend_synchronize(backend_);

    // Load drafter (lazy — stays resident for subsequent calls)
    if (!drafter_loaded_) {
        std::fprintf(stderr, "[compress] loading drafter from %s ...\n", dpath);
        if (!load_drafter(dpath, /*gpu_layers=*/999, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] drafter init failed: %s\n",
                         dflash27b_last_error());
            io.emit(-1);
            if (!skip_park && !was_parked) unpark("target");
            return false;
        }
        drafter_loaded_ = true;
        std::fprintf(stderr, "[compress] drafter ready\n");
    }

    std::vector<int32_t> tokens = read_int32_file(ppath);
    bool ok = false;
    if (!tokens.empty()) {
        const float keep = (float)keep_x1000 / 1000.0f;
        auto compressed = drafter_score_and_compress(drafter_ctx_, tokens, keep);
        ok = !compressed.empty();
        if (ok) {
            std::fprintf(stderr, "[compress] %zu -> %zu tokens\n",
                          tokens.size(), compressed.size());
            for (int32_t t : compressed) io.emit(t);
        }
    }
    io.emit(-1);

    // Restore park state
    if (!skip_park && !was_parked) {
        unpark("target");
    }

    return ok;
}

void Gemma4Backend::free_drafter() {
    if (drafter_loaded_) {
        ::dflash::common::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
    }
}

bool Gemma4Backend::try_handle_command(const std::string & line,
                                        const DaemonIO & io) {
    (void)line; (void)io;
    return false;  // no arch-specific commands
}

// ── Shutdown ───────────────────────────────────────────────────────────

void Gemma4Backend::shutdown() {
    for (int i = 0; i < PREFIX_SLOTS; ++i) snapshot_free(i);
    free_drafter();
    // Clean up DFlash spec-decode resources
    delete dflash_target_; dflash_target_ = nullptr;
    draft_feature_mirror_free(feature_mirror_);
    if (dw_.ctx) { free_draft_weights(dw_); }
    if (draft_backend_) { ggml_backend_free(draft_backend_); draft_backend_ = nullptr; }
    free_gemma4_cache(cache_);
    free_gemma4_weights(w_);
    free_snapshot_backend(snap_backend_, backend_);
    snap_backend_ = nullptr;
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
    std::printf("[gemma4] shutdown\n"); std::fflush(stdout);
}

}  // namespace dflash::common
