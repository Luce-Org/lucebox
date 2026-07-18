// DeepSeek-V4-Flash DSpark speculative decode: DFlashTarget adapter + spec loop.
//
// The drafter (DSparkDrafter, deepseek4_dspark.{h,cpp}) proposes block_size
// candidates conditioned on captured target features; the DS4 target verifies
// them in one batched forward. Feature capture + verify live in
// deepseek4_dspark_verify_forward (deepseek4_graph.cpp). The DSpark Markov head
// (common/dspark_head.cpp) is target-agnostic and reused verbatim.
//
// Fast spec loop (default): ONE batched verify per step, verify width capped
// at DS4_SPEC_Q=4 tokens (seed + 3 candidates). With q <= ratio(4) the verify
// crosses at most one compression boundary and never aliases rolling-state
// rows, so rejection rollback needs no full KV snapshot:
//   - at-risk raw ring rows are saved before the verify and rejected rows are
//     restored after wrap (accepted rows remain committed),
//   - comp rows are index-addressed (pos / ratio)        -> idempotent,
//   - n_comp / n_index_comp are pure functions of commit position,
//   - the other non-idempotent state is the ratio-4 compressor prev-half
//     (4 rows/state, flushed cur->prev at chunk boundaries), a few KB/layer,
//     saved host-side before the verify and restored only when the flush
//     happened at-or-past the rollback point.
// The legacy full-snapshot + double-verify path is kept behind
// DFLASH_DS4_FULL_SNAP=1 for A/B validation.

#include "deepseek4_dspark.h"
#include "deepseek4_internal.h"
#include "internal.h"
#include "common/dspark_head.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace dflash::common {

// ── DFlashTarget adapter over the DS4 target ────────────────────────────
class DeepSeek4DFlashTarget : public DFlashTarget {
public:
    DeepSeek4DFlashTarget(const DeepSeek4Weights & w, DeepSeek4Cache & cache,
                          ggml_backend_t backend, ggml_backend_t snap_backend,
                          std::vector<int> capture_ids, int mask_tok)
        : w_(w), cache_(cache), backend_(backend), snap_backend_(snap_backend),
          capture_ids_(std::move(capture_ids)), mask_tok_(mask_tok) {}

    ~DeepSeek4DFlashTarget() override { clear_snapshot(); }

    bool verify_batch(const std::vector<int32_t> & tokens, int base_pos, int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override {
        (void) capture_ssm_intermediates;
        const int n = (int) tokens.size();
        embed_buf_.resize((size_t) n * w_.n_embd);
        if (!w_.embedder.embed(tokens.data(), n, embed_buf_.data())) {
            std::fprintf(stderr, "[ds4-verify] embed FAILED n=%d tok0=%d tok1=%d vocab=%d\n",
                         n, n > 0 ? tokens[0] : -1, n > 1 ? tokens[1] : -1, w_.n_vocab);
            return false;
        }
        // Sequential verify (measurement mode): q single-token forwards through
        // the legacy AR decode path. Causal by construction, compressor fed every
        // token. Slow; used to measure the drafter's token-at-a-time accept rate.
        // It is not a bit-exact oracle: graph shape can change floating-point
        // reduction order around near-tied logits. Enable: DFLASH_DS4_SEQ_VERIFY=1
        // (pair with DFLASH_DS4_FULL_SNAP=1 so rollback/replay stay exact).
        static const bool seq_verify = [] {
            const char * v = std::getenv("DFLASH_DS4_SEQ_VERIFY");
            return v && *v && *v != '0';
        }();
        if (seq_verify) {
            std::vector<int32_t> am_all;
            std::vector<float> feat_all;
            std::vector<float> logits_all;
            am_all.reserve(n);
            for (int t = 0; t < n; t++) {
                std::vector<int32_t> am1;
                std::vector<float> feat1;
                std::vector<float> logits1;
                if (!deepseek4_dspark_verify_forward(backend_, w_, cache_, capture_ids_,
                                                     embed_buf_.data() + (size_t) t * w_.n_embd,
                                                     tokens.data() + t, 1, base_pos + t, am1,
                                                     keep_logits_ ? &logits1 : nullptr,
                                                     feat1, telemetry_,
                                                     /*allow_graph_reuse=*/false)) {
                    return false;
                }
                if (am1.empty()) return false;
                am_all.push_back(am1[0]);
                feat_all.insert(feat_all.end(), feat1.begin(), feat1.end());
                if (keep_logits_) logits_all.insert(logits_all.end(), logits1.begin(), logits1.end());
            }
            verify_features_ = std::move(feat_all);
            if (keep_logits_) verify_logits_ = std::move(logits_all);
            last_tok = am_all.back();
            verify_n_ = n;
            if (all_argmax) *all_argmax = std::move(am_all);
            return true;
        }
        std::vector<int32_t> am;
        // n==1 must take the dynamic (non-reuse) path: the reused decode graph
        // skips the capture/all-logits hooks (backend HC), which this needs.
        if (!deepseek4_dspark_verify_forward(backend_, w_, cache_, capture_ids_,
                                             embed_buf_.data(), tokens.data(), n, base_pos, am,
                                             keep_logits_ ? &verify_logits_ : nullptr,
                                             verify_features_, telemetry_,
                                             /*allow_graph_reuse=*/n > 1)) {
            return false;
        }
        if (am.empty()) return false;
        last_tok = am.back();
        verify_n_ = n;
        if (all_argmax) *all_argmax = std::move(am);
        return true;
    }

    bool read_verify_logits(int n_tokens, std::vector<float> & out) override {
        if (!keep_logits_ || verify_logits_.empty()) return false;
        const size_t need = (size_t) n_tokens * w_.n_vocab;
        if (verify_logits_.size() < need) return false;
        out.assign(verify_logits_.begin(), verify_logits_.begin() + need);
        return true;
    }

    bool snapshot_kv() override { return deepseek4_snapshot_save(cache_, snap_backend_, snap_); }
    bool restore_kv() override { return deepseek4_snapshot_restore(snap_, cache_); }

    bool is_eos(int token) const override { return deepseek4_is_eos_tok(token, w_); }

    bool embed_tokens(const int32_t * tokens, int n, float * out) const override {
        return w_.embedder.embed(tokens, n, out);
    }

    bool project_hidden_to_tokens(const float * hidden, int n_tokens,
                                  std::vector<int32_t> & tokens_out) override {
        std::vector<float> logits;
        if (!project_hidden_to_logits(hidden, n_tokens, logits)) return false;
        tokens_out.resize(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            const float * row = logits.data() + (size_t) t * w_.n_vocab;
            int best = 0; float bv = row[0];
            for (int i = 1; i < w_.n_vocab; i++) if (row[i] > bv) { bv = row[i]; best = i; }
            tokens_out[t] = best;
        }
        return true;
    }

    // The drafter hidden is already out_norm'd (drafter tail); project with the
    // tied target lm_head only (mul_mat, no norm), matching the reference head.
    bool project_hidden_to_logits(const float * hidden, int n_tokens,
                                  std::vector<float> & logits_out) override {
        if (n_tokens <= 0) return false;
        ggml_init_params ip{};
        ip.mem_size = 32u * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w_.n_embd, n_tokens);
        ggml_set_input(h);
        ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, h);   // [n_vocab, n_tokens]
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(h, hidden, 0, sizeof(float) * (size_t) w_.n_embd * n_tokens);
        const ggml_status st = ggml_backend_graph_compute(backend_, gf);
        if (st != GGML_STATUS_SUCCESS) { ggml_gallocr_free(alloc); ggml_free(ctx); return false; }
        logits_out.resize((size_t) n_tokens * w_.n_vocab);
        ggml_backend_tensor_get(logits, logits_out.data(), 0, sizeof(float) * logits_out.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return true;
    }

    ggml_tensor * lm_head_tensor() override { return w_.output; }
    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override { return mask_tok_; }
    const std::vector<int> & capture_layer_ids() const override { return capture_ids_; }

    void set_keep_logits(bool b) { keep_logits_ = b; }
    void set_telemetry(DeepSeek4StepTelemetry * t) { telemetry_ = t; }
    const std::vector<float> & last_features() const { return verify_features_; }
    int last_verify_n() const { return verify_n_; }
    void clear_snapshot() { free_deepseek4_snapshot(snap_); }

private:
    const DeepSeek4Weights & w_;
    DeepSeek4Cache & cache_;
    ggml_backend_t backend_;
    ggml_backend_t snap_backend_;
    std::vector<int> capture_ids_;
    int mask_tok_;
    DeepSeek4Snapshot snap_{};
    DeepSeek4StepTelemetry * telemetry_ = nullptr;
    bool keep_logits_ = false;
    int verify_n_ = 0;
    std::vector<float> embed_buf_;
    std::vector<float> verify_logits_;
    std::vector<float> verify_features_;
};

namespace {

// Build the DraftWeights shim the target-agnostic dspark head expects (only the
// DSpark head fields + n_embd are read).
DraftWeights make_dspark_shim(const DSparkDrafter & d) {
    DraftWeights dw{};
    dw.n_embd = d.core.n_embd;
    dw.dspark.enabled = d.dspark_enabled;
    dw.dspark.markov_rank = d.markov_rank;
    dw.dspark.vocab_size = d.vocab_size;
    dw.dspark.confidence_dim = d.confidence_dim;
    dw.dspark.markov_w1 = d.markov_w1;
    dw.dspark.markov_w2 = d.markov_w2;
    dw.dspark.confidence_w = d.confidence_w;
    dw.dspark.confidence_b = d.confidence_b;
    return dw;
}

bool spec_env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v && *v && *v != '0';
}

// Calibrated cumulative-confidence thresholds for widening the DS4 verify.
// They are part of the policy, not deployment knobs: artifacts without a
// compatible confidence head transparently retain the existing EWMA policy.
constexpr float kConfidenceQ3Threshold = 0.40f;
constexpr float kConfidenceQ4Threshold = 0.30f;

// ── Light rollback state ────────────────────────────────────────────────
// prev-half = first 4 rows of a [comp_width, 8] ratio-4 rolling state.
// ratio-128 states ([comp_width, 128]) are pure position rings -> skip.
void save_prev_half(ggml_tensor * t, std::vector<uint8_t> & buf) {
    if (!t || t->ne[1] != 8) { buf.clear(); return; }
    const size_t bytes = (size_t) t->nb[1] * 4;
    if (buf.size() != bytes) buf.resize(bytes);
    ggml_backend_tensor_get(t, buf.data(), 0, bytes);
}

void restore_prev_half(ggml_tensor * t, const std::vector<uint8_t> & buf) {
    if (!t || buf.empty()) return;
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

void spec_rollback_save_impl(const DeepSeek4Cache & cache,
                             DeepSeek4SpecRollback & rb,
                             int raw_pos, int raw_count) {
    rb.raw_pos = raw_pos;
    rb.raw_count = std::max(0, raw_count);
    rb.layers.resize(cache.layers.size());
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const DeepSeek4LayerCache & lc = cache.layers[il];
        DeepSeek4SpecRollback::Layer & s = rb.layers[il];
        save_prev_half(lc.attn_compressor.state_kv,       s.attn_kv);
        save_prev_half(lc.attn_compressor.state_score,    s.attn_sc);
        save_prev_half(lc.indexer_compressor.state_kv,    s.idx_kv);
        save_prev_half(lc.indexer_compressor.state_score, s.idx_sc);
        s.raw_row_bytes = 0;
        s.raw_rows.clear();
        if (lc.raw_kv && lc.raw_kv->ne[1] > 0 && rb.raw_count > 0) {
            s.raw_row_bytes = ggml_row_size(lc.raw_kv->type, lc.raw_kv->ne[0]);
            s.raw_rows.resize(s.raw_row_bytes * (size_t) rb.raw_count);
            for (int t = 0; t < rb.raw_count; ++t) {
                int row = (rb.raw_pos + t) % (int) lc.raw_kv->ne[1];
                if (row < 0) row += (int) lc.raw_kv->ne[1];
                ggml_backend_tensor_get(
                    lc.raw_kv,
                    s.raw_rows.data() + (size_t) t * s.raw_row_bytes,
                    (size_t) row * lc.raw_kv->nb[1], s.raw_row_bytes);
            }
        }
    }
    if (cache.hc_state) {
        const size_t bytes = ggml_nbytes(cache.hc_state);
        if (rb.hc_state.size() != bytes) rb.hc_state.resize(bytes);
        ggml_backend_tensor_get(cache.hc_state, rb.hc_state.data(), 0, bytes);
    }
}

// Truncate the cache to commit_pos. restore_prev is set when the verify
// crossed a ratio-4 boundary at-or-past commit_pos: that flush filled the
// prev-half rows with a chunk containing rejected tokens, so put the
// pre-verify rows back. (A boundary strictly inside the committed range is a
// legitimate flush and must be kept.)
void spec_rollback_apply_impl(const DeepSeek4SpecRollback & rb,
                              const DeepSeek4Weights & w,
                              DeepSeek4Cache & cache,
                              int commit_pos,
                              bool restore_prev) {
    cache.cur_pos = commit_pos;
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        DeepSeek4LayerCache & lc = cache.layers[il];
        const uint32_t ratio = il < w.compress_ratios.size() ? w.compress_ratios[il] : 0;
        if (ratio > 0) lc.n_comp = commit_pos / (int) ratio;
        if (ratio == 4) lc.n_index_comp = commit_pos / 4;
        if (restore_prev && il < rb.layers.size()) {
            const DeepSeek4SpecRollback::Layer & s = rb.layers[il];
            restore_prev_half(lc.attn_compressor.state_kv,       s.attn_kv);
            restore_prev_half(lc.attn_compressor.state_score,    s.attn_sc);
            restore_prev_half(lc.indexer_compressor.state_kv,    s.idx_kv);
            restore_prev_half(lc.indexer_compressor.state_score, s.idx_sc);
        }
        if (il < rb.layers.size() && lc.raw_kv && lc.raw_kv->ne[1] > 0) {
            const DeepSeek4SpecRollback::Layer & s = rb.layers[il];
            const int first_rejected = std::max(0, commit_pos - rb.raw_pos);
            for (int t = first_rejected;
                 t < rb.raw_count &&
                 s.raw_row_bytes > 0 &&
                 s.raw_rows.size() >= (size_t) (t + 1) * s.raw_row_bytes;
                 ++t) {
                int row = (rb.raw_pos + t) % (int) lc.raw_kv->ne[1];
                if (row < 0) row += (int) lc.raw_kv->ne[1];
                ggml_backend_tensor_set(
                    lc.raw_kv,
                    s.raw_rows.data() + (size_t) t * s.raw_row_bytes,
                    (size_t) row * lc.raw_kv->nb[1], s.raw_row_bytes);
            }
        }
    }
    if (restore_prev && cache.hc_state && !rb.hc_state.empty()) {
        ggml_backend_tensor_set(cache.hc_state, rb.hc_state.data(), 0, rb.hc_state.size());
    }
}

using SpecClock = std::chrono::steady_clock;

double spec_ms_since(SpecClock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(SpecClock::now() - t0).count() / 1000.0;
}

}  // namespace

void deepseek4_spec_rollback_save(const DeepSeek4Cache & cache,
                                  DeepSeek4SpecRollback & rollback,
                                  int raw_pos,
                                  int raw_count) {
    spec_rollback_save_impl(cache, rollback, raw_pos, raw_count);
}

void deepseek4_spec_rollback_apply(const DeepSeek4SpecRollback & rollback,
                                   const DeepSeek4Weights & weights,
                                   DeepSeek4Cache & cache,
                                   int commit_pos,
                                   bool restore_prev) {
    spec_rollback_apply_impl(rollback, weights, cache, commit_pos, restore_prev);
}

// Batched target verify + capture: wraps the existing multi-token
// deepseek4_step_layer_range (dynamic attention + batched HC), which never
// touches the fused single-token 23 tok/s path, with the Ds4VerifyHooks that
// add per-layer mean-over-HC capture and full per-position logits.
bool deepseek4_dspark_verify_forward(ggml_backend_t backend,
                                     const DeepSeek4Weights & w,
                                     DeepSeek4Cache & cache,
                                     const std::vector<int> & capture_layer_ids,
                                     const float * embed,
                                     const int32_t * token_ids,
                                     int n_tokens,
                                     int kv_start,
                                     std::vector<int32_t> & argmax_out,
                                     std::vector<float> * logits_out,
                                     std::vector<float> & capture_out,
                                     DeepSeek4StepTelemetry * telemetry,
                                     bool allow_graph_reuse) {
    std::vector<float> hc_state;
    std::vector<float> all_logits;
    std::vector<float> last_logits;
    Ds4VerifyHooks hooks;
    hooks.capture_layer_ids = &capture_layer_ids;
    hooks.capture_out = &capture_out;
    hooks.all_logits_out = &all_logits;
    if (!deepseek4_step_layer_range(backend, w, cache, hc_state, embed, n_tokens, kv_start,
                                    0, w.n_layer, &last_logits, token_ids,
                                    telemetry, allow_graph_reuse,
                                    &hooks)) {
        std::fprintf(stderr, "[ds4-verify] step_layer_range returned false (n_tokens=%d kv_start=%d)\n",
                     n_tokens, kv_start);
        return false;
    }
    if ((int) all_logits.size() < w.n_vocab * n_tokens) {
        std::fprintf(stderr, "[ds4-verify] all_logits too small: got=%zu need=%d (cap=%zu)\n",
                     all_logits.size(), w.n_vocab * n_tokens, capture_out.size());
        return false;
    }
    argmax_out.resize(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        const float * row = all_logits.data() + (size_t) t * w.n_vocab;
        int best = 0; float bv = row[0];
        for (int i = 1; i < w.n_vocab; i++) if (row[i] > bv) { bv = row[i]; best = i; }
        argmax_out[t] = best;
    }
    if (logits_out) *logits_out = std::move(all_logits);
    return true;
}

bool run_deepseek4_dspark_spec_decode(
        ggml_backend_t backend,
        const DeepSeek4Weights & target_w,
        DeepSeek4Cache & target_cache,
        const DSparkDrafter & drafter,
        int committed,
        int last_tok,
        int n_gen,
        const float * prompt_feature_window,
        int win_len,
        std::vector<int32_t> & out_tokens,
        float * accept_rate_out,
        const std::function<bool(int32_t)> & on_token) {
    const int n_embd = target_w.n_embd;
    const int n_tgt = drafter.n_target_layers;
    const int block = drafter.block_size;
    const int n_swa = target_w.n_swa;
    const int feat_row = n_tgt * n_embd;

    const bool debug = spec_env_flag("DFLASH_DS4_DSPARK_DEBUG");
    const bool timing = spec_env_flag("DFLASH_DS4_TIMING");
    const bool full_snap = spec_env_flag("DFLASH_DS4_FULL_SNAP");
    const bool seq_verify_mode = spec_env_flag("DFLASH_DS4_SEQ_VERIFY");
    // Laguna-style adaptive verify width: EWMA of accepted candidates, width =
    // ewma + 2 (avg_commit << block means the wide tail is usually wasted).
    // /tmp/ds4_awidth: 1 = on, 0 = off (default on).
    bool adaptive_width = true;
    if (std::FILE * f = std::fopen("/tmp/ds4_awidth", "r")) {
        int v = 1;
        if (std::fscanf(f, "%d", &v) == 1) adaptive_width = (v != 0);
        std::fclose(f);
    }
    const bool use_confidence_width = adaptive_width && !seq_verify_mode &&
        drafter.confidence_w != nullptr && drafter.confidence_b != nullptr &&
        (drafter.confidence_dim == n_embd ||
         drafter.confidence_dim == n_embd + drafter.markov_rank);
    if (timing && use_confidence_width) {
        std::fprintf(stderr, "[ds4-spec] adaptive width policy=confidence\n");
    }
    double ewma_accept = 1.5;

    // Fast path caps the verify at the compression ratio (4): one boundary max,
    // no rolling-state row aliasing -> snapshot-free rollback stays exact.
    // Full snapshots change rollback strategy but not the compressor-window
    // limit below. The legacy sequential measurement path is validated only
    // through q=4.
    int q_cap = full_snap ? block + 1 : 4;
    if (const char * qs = std::getenv("DFLASH_DS4_SPEC_Q")) {
        const int v = std::atoi(qs);
        if (v >= 2 && v <= block + 1) q_cap = full_snap ? v : std::min(v, 4);
    }
    if (std::FILE * qf = std::fopen("/tmp/ds4_spec_q", "r")) {
        // Per-request override for perf experiments (no server restart needed).
        // q=1 disables drafting: pure AR pushed through the batched-verify path
        // (diagnoses batched-vs-sequential target divergence).
        int v = 0;
        if (std::fscanf(qf, "%d", &v) == 1 && v >= 1 && v <= block + 1) {
            q_cap = full_snap ? v : std::min(v, 4);
        }
        std::fclose(qf);
    }
    if (seq_verify_mode && q_cap > 4) {
        std::fprintf(stderr,
                     "[ds4-spec] sequential verify supports q<=4; "
                     "capping requested q=%d to 4\n",
                     q_cap);
        q_cap = 4;
    }

    // Snapshot backend for the legacy full-snapshot rollback path.
    ggml_backend_t snap_backend = ggml_backend_cpu_init();
    if (!snap_backend) { std::fprintf(stderr, "[ds4-spec] no CPU snapshot backend\n"); return false; }

    DeepSeek4DFlashTarget target(target_w, target_cache, backend, snap_backend,
                                 drafter.capture_layer_ids, drafter.mask_token_id);
    DraftWeights dw = make_dspark_shim(drafter);
    DeepSeek4SpecRollback rollback;
    DeepSeek4StepTelemetry tel{};
    if (timing) target.set_telemetry(&tel);

    // Host feature window ring [feat_row, n_swa] of absolute positions
    // [committed-N .. committed-1]. Seed from the prefill window.
    std::vector<float> feat_win((size_t) feat_row * n_swa, 0.0f);
    int win_have = win_len > n_swa ? n_swa : win_len;
    if (prompt_feature_window && win_have > 0) {
        // copy the last win_have columns of the prefill window
        const int src_off = (win_len - win_have);
        std::memcpy(feat_win.data(),
                    prompt_feature_window + (size_t) src_off * feat_row,
                    sizeof(float) * (size_t) feat_row * win_have);
    }
    int feat_count = win_have;   // number of valid feature columns ending at committed-1

    auto push_feature = [&](const float * col) {
        // Shift-append one feature column (keep last n_swa).
        if (feat_count >= n_swa) {
            std::memmove(feat_win.data(), feat_win.data() + feat_row,
                         sizeof(float) * (size_t) feat_row * (n_swa - 1));
            std::memcpy(feat_win.data() + (size_t) feat_row * (n_swa - 1), col,
                        sizeof(float) * feat_row);
        } else {
            std::memcpy(feat_win.data() + (size_t) feat_row * feat_count, col,
                        sizeof(float) * feat_row);
            feat_count++;
        }
    };

    int lt = last_tok;
    int pos = committed;      // absolute position of the seed (block slot 0)
    int n_generated = 0;
    long accept_sum = 0, offered_sum = 0, steps = 0;
    bool ok = true;
    bool stop_requested = false;

    std::vector<float> noise_embed((size_t) n_embd * block);
    std::vector<int32_t> noise_ids(block);
    std::vector<float> local_hidden, confidence_hidden;
    std::vector<float> padded_hidden((size_t) n_embd * (block + 1), 0.0f);
    std::vector<float> padded_confidence_hidden((size_t) n_embd * (block + 1), 0.0f);
    std::vector<int32_t> draft_tok, tgt_am;
    std::vector<float> draft_confidence;

    // Cumulative phase timings (ms).
    double tm_draft = 0, tm_head = 0, tm_save = 0, tm_verify = 0, tm_apply = 0, tm_feat = 0;
    const SpecClock::time_point run_t0 = SpecClock::now();

    while (n_generated < n_gen) {
        const int ctx_len = feat_count < n_swa ? feat_count : n_swa;

        // Noise block = [seed] + [MASK]*(block-1).
        SpecClock::time_point t0 = SpecClock::now();
        if (q_cap >= 2) {
            noise_ids[0] = lt;
            for (int i = 1; i < block; i++) noise_ids[i] = drafter.mask_token_id;
            if (!target.embed_tokens(noise_ids.data(), block, noise_embed.data())) {
                ok = false;
                break;
            }

            // Drafter forward -> normalized states for token logits plus the
            // pre-output-norm states expected by the confidence head.
            if (!deepseek4_dspark_draft_forward(backend, drafter, noise_embed.data(),
                                                ctx_len > 0 ? feat_win.data() : nullptr,
                                                ctx_len, pos, local_hidden,
                                                use_confidence_width ? &confidence_hidden : nullptr)) {
                std::fprintf(stderr, "[ds4-spec] drafter forward failed\n");
                ok = false;
                break;
            }
        }
        tm_draft += spec_ms_since(t0);

        if (debug) {
            size_t lh_nan = 0; double lh_ss = 0;
            for (float v : local_hidden) { if (!std::isfinite(v)) lh_nan++; else lh_ss += (double) v * v; }
            std::fprintf(stderr, "[ds4-spec] hidden nnan=%zu/%zu rms=%.4f ctx_len=%d\n",
                         lh_nan, local_hidden.size(),
                         lh_nan < local_hidden.size() ? std::sqrt(lh_ss / (double) local_hidden.size()) : 0.0,
                         ctx_len);
        }

        // DSpark Markov chain over the first q_cap-1 candidates. Reference
        // predicts token i+1 from block slot i, so prepend a dummy row 0 and
        // let the (row-0-skipping) chain use slots 1..q-1.
        t0 = SpecClock::now();
        draft_tok.clear();
        draft_confidence.clear();
        bool ds_ok = false;
        // Batched-verify exactness: the batch must not cross a ratio-4
        // boundary except at its last token (state rows stay distinct and the
        // comp emission matches AR). Boundaries sit at p % 4 == 3. The legacy
        // sequential measurement path is independently capped to q=4 above.
        int q_step_cap = seq_verify_mode
                       ? std::min(q_cap, 4)
                       : std::min(q_cap, 4 - (pos & 3));
        if (adaptive_width && !use_confidence_width && !seq_verify_mode) {
            const int w_cap = (int) ewma_accept + 2;
            if (w_cap < q_step_cap) q_step_cap = w_cap;
        }
        if (q_step_cap >= 2) {
            std::memcpy(padded_hidden.data() + n_embd, local_hidden.data(),
                        sizeof(float) * (size_t) n_embd * block);
            if (use_confidence_width) {
                std::memcpy(padded_confidence_hidden.data() + n_embd,
                            confidence_hidden.data(),
                            sizeof(float) * (size_t) n_embd * block);
            }
            ds_ok = dspark_markov_correct_greedy_chain_fused(
                            dw, backend, target.lm_head_tensor(), padded_hidden.data(),
                            q_step_cap, lt, draft_tok,
                            use_confidence_width ? &draft_confidence : nullptr,
                            use_confidence_width ? padded_confidence_hidden.data() : nullptr);
            if (!ds_ok) {
                ds_ok = dspark_markov_correct_greedy_chain(dw, backend, target,
                            padded_hidden.data(), q_step_cap, lt, 0.0f, draft_tok);
            }
            if (!ds_ok || (int) draft_tok.size() < 2) {
                // Fallback: plain projection of the block hiddens.
                std::vector<int32_t> pj;
                if (!target.project_hidden_to_tokens(local_hidden.data(), q_step_cap - 1, pj)) {
                    ok = false;
                    break;
                }
                draft_tok.clear();
                draft_tok.push_back(lt);
                for (int i = 0; i < q_step_cap - 1; i++) draft_tok.push_back(pj[i]);
            }
        } else {
            draft_tok.push_back(lt);   // q=1: seed only, no speculation
        }
        // Confidence estimates are per candidate. Their cumulative product is
        // the estimated probability that the target accepts the whole prefix
        // unlocked by a wider verify. The defaults were calibrated on q=4
        // traces and keep q=4 for high-confidence prefixes while avoiding its
        // extra verify cost on low-acceptance prompts.
        if (use_confidence_width && draft_confidence.size() >= 2 && draft_tok.size() >= 3) {
            const float confidence_p2 = draft_confidence[0] * draft_confidence[1];
            int selected_q = confidence_p2 >= kConfidenceQ3Threshold ? 3 : 2;
            if (selected_q == 3 && draft_confidence.size() >= 3 && draft_tok.size() >= 4) {
                const float confidence_p3 = confidence_p2 * draft_confidence[2];
                if (confidence_p3 >= kConfidenceQ4Threshold) selected_q = 4;
            }
            if ((int) draft_tok.size() > selected_q) draft_tok.resize((size_t) selected_q);
        } else if (use_confidence_width && !seq_verify_mode) {
            // The fused head should always return confidence for a compatible
            // artifact. Preserve the old policy if a backend cannot do so.
            const int selected_q = (int) ewma_accept + 2;
            if ((int) draft_tok.size() > selected_q) draft_tok.resize((size_t) selected_q);
        }
        if ((int) draft_tok.size() > q_step_cap) draft_tok.resize(q_step_cap);
        const int q = (int) draft_tok.size();   // seed + candidates
        tm_head += spec_ms_since(t0);

        if (debug) {
            std::fprintf(stderr, "[ds4-spec] dbg ds_ok=%d q=%d lt=%d draft=[%d %d %d %d]\n",
                         (int) ds_ok, q, lt,
                         q > 0 ? draft_tok[0] : -1, q > 1 ? draft_tok[1] : -1,
                         q > 2 ? draft_tok[2] : -1, q > 3 ? draft_tok[3] : -1);
        }

        // ── Rollback state save (cheap) or legacy full snapshot ──
        t0 = SpecClock::now();
        if (full_snap) {
            if (!target.snapshot_kv()) {
                std::fprintf(stderr, "[ds4-spec] snapshot failed\n");
                ok = false;
                break;
            }
        } else {
            deepseek4_spec_rollback_save(target_cache, rollback, pos, q);
        }
        tm_save += spec_ms_since(t0);

        // First ratio-4 boundary position touched by this verify (p % 4 == 3).
        const int first_boundary = pos + (3 - (pos & 3));
        const bool boundary_crossed = first_boundary <= pos + q - 1;

        // ── ONE batched verify (writes cache + captures features for all q) ──
        t0 = SpecClock::now();
        int verify_last = -1;
        if (!target.verify_batch(draft_tok, pos, verify_last, &tgt_am)) {
            if (full_snap) {
                if (!target.restore_kv()) {
                    std::fprintf(stderr, "[ds4-spec] restore after verify failure failed\n");
                }
            } else {
                deepseek4_spec_rollback_apply(
                    rollback, target_w, target_cache, pos, boundary_crossed);
            }
            std::fprintf(stderr, "[ds4-spec] verify failed\n");
            ok = false;
            break;
        }
        tm_verify += spec_ms_since(t0);

        // Accept the longest matching prefix. accept counts the seed (slot 0)
        // plus each candidate the target agrees with.
        int accept = 1;
        for (int i = 0; i < q - 1; i++) {
            if (draft_tok[i + 1] == tgt_am[i]) accept++;
            else break;
        }
        const int matched = accept - 1;                       // accepted candidates
        const int bonus = tgt_am[accept - 1];                 // target's token at the accept point
        const int commit_pos = pos + accept;                  // seed + accepted candidates in KV

        if (timing && steps < 8 && q >= 2) {
            // Alignment probe: draft candidate i should match tgt_am[i-1]. A
            // consistent draft[i]==tgt_am[i] pattern instead = off-by-one.
            std::fprintf(stderr, "[ds4-spec-cmp] step=%ld pos=%d draft=[%d %d %d] tgt=[%d %d %d %d] acc=%d\n",
                         steps, pos,
                         q > 1 ? draft_tok[1] : -1, q > 2 ? draft_tok[2] : -1, q > 3 ? draft_tok[3] : -1,
                         tgt_am[0], q > 1 ? tgt_am[1] : -1, q > 2 ? tgt_am[2] : -1, q > 3 ? tgt_am[3] : -1,
                         accept);
        }

        // ── Rollback: truncate to the committed prefix ──
        // The bonus token is DEFERRED: it becomes the next step's seed, whose
        // KV is written then.
        t0 = SpecClock::now();
        if (full_snap) {
            // Legacy: full restore + replay the committed tokens through the
            // target so ring/compressor/n_comp advance exactly.
            std::vector<int32_t> kv_toks;
            kv_toks.push_back(lt);
            for (int i = 1; i < accept; i++) kv_toks.push_back(draft_tok[i]);
            if (!target.restore_kv()) {
                std::fprintf(stderr, "[ds4-spec] snapshot restore failed\n");
                ok = false;
                break;
            }
            int replay_last = -1;
            std::vector<int32_t> replay_am;
            if (!target.verify_batch(kv_toks, pos, replay_last, &replay_am)) {
                std::fprintf(stderr, "[ds4-spec] replay verify failed\n");
                ok = false;
                break;
            }
        } else if (accept < q) {
            // The prev-half flush is bad only if the boundary sits at-or-past
            // the commit point (its chunk then contains rejected tokens).
            const bool restore_prev = boundary_crossed && first_boundary >= commit_pos;
            deepseek4_spec_rollback_apply(
                rollback, target_w, target_cache, commit_pos, restore_prev);
        }
        // accept == q on the fast path: cur_pos/n_comp already exact, keep.
        tm_apply += spec_ms_since(t0);

        // Push the committed positions' features (slots 0..accept-1 = positions
        // pos..pos+accept-1) into the drafter's context window.
        t0 = SpecClock::now();
        const std::vector<float> & feats = target.last_features();
        const int fN = full_snap ? target.last_verify_n() : accept;
        for (int i = 0; i < fN; i++) push_feature(feats.data() + (size_t) i * feat_row);
        tm_feat += spec_ms_since(t0);

        // Output tokens this step = accepted candidates + bonus.
        bool hit_eos = false;
        for (int i = 1; i <= accept; i++) {
            const int t = (i < accept) ? draft_tok[i] : bonus;
            out_tokens.push_back(t);
            n_generated++;
            if (on_token && !on_token(t)) {
                stop_requested = true;
                break;
            }
            if (target.is_eos(t)) { hit_eos = true; break; }
            if (n_generated >= n_gen) break;
        }
        pos = commit_pos;              // seed + accepted candidates now in KV
        lt = bonus;                    // deferred bonus becomes next seed
        accept_sum += matched;
        offered_sum += q - 1;
        ewma_accept = 0.7 * ewma_accept + 0.3 * (double) matched;
        steps++;
        if (timing && (steps <= 4 || (steps & 31) == 0)) {
            std::fprintf(stderr,
                "[ds4-spec-t] step=%ld q=%d acc=%d | draft=%.1f head=%.1f save=%.1f "
                "verify=%.1f apply=%.1f feat=%.1f ms (cum means)\n",
                steps, q, accept,
                tm_draft / steps, tm_head / steps, tm_save / steps,
                tm_verify / steps, tm_apply / steps, tm_feat / steps);
        }
        if (hit_eos || stop_requested) break;
    }

    const double total_ms = spec_ms_since(run_t0);
    if (accept_rate_out) {
        *accept_rate_out = offered_sum > 0
            ? (float) accept_sum / (float) offered_sum
            : 0.0f;
    }
    std::fprintf(stderr,
                 "[ds4-spec] gen=%d steps=%ld mean_accept=%.2f/%.2f "
                 "q_cap=%d full_snap=%d\n",
                 n_generated, steps,
                 steps ? (double) accept_sum / steps : 0.0,
                 steps ? (double) offered_sum / steps : 0.0, q_cap,
                 (int) full_snap);
    if (steps > 0) {
        std::fprintf(stderr,
            "[ds4-spec-t] TOTAL %.1f ms, %ld steps (%.1f ms/step), %d tok (%.1f tok/s) | "
            "means: draft=%.1f head=%.1f save=%.1f verify=%.1f apply=%.1f feat=%.1f ms\n",
            total_ms, steps, total_ms / steps, n_generated,
            total_ms > 0 ? n_generated * 1000.0 / total_ms : 0.0,
            tm_draft / steps, tm_head / steps, tm_save / steps,
            tm_verify / steps, tm_apply / steps, tm_feat / steps);
    }
    if (timing && steps > 0) {
        const double s = 1000.0 * steps;   // us -> ms per-step means
        std::fprintf(stderr,
            "[ds4-spec-t] verify tel/step: hc_pre_a=%.1f attn_b=%.1f attn_c=%.1f attn_r=%.1f "
            "hc_post_a=%.1f hc_pre_f=%.1f route(b/c/r/s)=%.1f/%.1f/%.1f/%.1f "
            "ffn(b/c/r)=%.1f/%.1f/%.1f eval=%.1f hot=%.1f cold=%.1f comb=%.1f part=%.1f "
            "full(b/s/c/r)=%.1f/%.1f/%.1f/%.1f ghits=%llu gbuilds=%llu ms\n",
            tel.hc_pre_attn_us / s, tel.attn_build_us / s, tel.attn_compute_us / s,
            tel.attn_read_us / s, tel.hc_post_attn_us / s, tel.hc_pre_ffn_us / s,
            tel.route_build_us / s, tel.route_compute_us / s, tel.route_read_us / s,
            tel.route_select_us / s,
            tel.ffn_build_us / s, tel.ffn_compute_us / s, tel.ffn_read_us / s,
            tel.ffn_eval_us / s, tel.ffn_hot_us / s, tel.ffn_cold_us / s,
            tel.ffn_combine_us / s, tel.ffn_partition_us / s,
            tel.full_graph_build_us / s, tel.full_graph_set_us / s,
            tel.full_graph_compute_us / s, tel.full_graph_read_us / s,
            (unsigned long long) tel.ffn_hot_graph_hits,
            (unsigned long long) tel.ffn_hot_graph_builds);
    }
    // Snapshot buffers must be released while their backend is still alive.
    // clear_snapshot() is idempotent, so the target destructor remains a
    // safety net for future exits that are added above this point.
    target.clear_snapshot();
    ggml_backend_free(snap_backend);
    return ok;
}

}  // namespace dflash::common
