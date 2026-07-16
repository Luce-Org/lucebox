// DeepSeek4Backend implementation — full-weight HIP/CUDA backend.

#include "deepseek4_backend.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

namespace {
using Clock = std::chrono::steady_clock;

static double elapsed_s(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static uint64_t elapsed_us(Clock::time_point start, Clock::time_point end) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static void reset_full_request_state(DeepSeek4Cache & cache,
                                     std::vector<float> & hc_state,
                                     std::vector<float> & last_logits) {
    cache.cur_pos = 0;
    for (auto & lc : cache.layers) {
        lc.n_comp = 0;
        lc.n_index_comp = 0;
    }
    std::fill(hc_state.begin(), hc_state.end(), 0.0f);
    last_logits.clear();
}

static bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static double ms(uint64_t us) {
    return (double)us / 1000.0;
}

static void add_step_tel(DeepSeek4StepTelemetry & dst, const DeepSeek4StepTelemetry & src) {
    dst.total_us += src.total_us;
    dst.embed_us += src.embed_us;
    dst.hc_pre_attn_us += src.hc_pre_attn_us;
    dst.hc_pre_build_us += src.hc_pre_build_us;
    dst.hc_pre_input_us += src.hc_pre_input_us;
    dst.hc_pre_compute_us += src.hc_pre_compute_us;
    dst.attn_build_us += src.attn_build_us;
    dst.attn_compute_us += src.attn_compute_us;
    dst.attn_read_us += src.attn_read_us;
    dst.hc_post_attn_us += src.hc_post_attn_us;
    dst.hc_pre_ffn_us += src.hc_pre_ffn_us;
    dst.ffn_build_us += src.ffn_build_us;
    dst.ffn_compute_us += src.ffn_compute_us;
    dst.ffn_read_us += src.ffn_read_us;
    dst.route_build_us += src.route_build_us;
    dst.route_compute_us += src.route_compute_us;
    dst.route_read_us += src.route_read_us;
    dst.route_select_us += src.route_select_us;
    dst.ffn_eval_us += src.ffn_eval_us;
    dst.hc_post_ffn_us += src.hc_post_ffn_us;
    dst.output_us += src.output_us;
    dst.sample_us += src.sample_us;
    dst.emit_us += src.emit_us;
}

static void log_step_tel(const char * phase,
                         int tokens,
                         int steps,
                         double wall_s,
                         const DeepSeek4StepTelemetry & t) {
    const double tok_s = wall_s > 0.0 ? (double)tokens / wall_s : 0.0;
    std::fprintf(stderr,
        "[deepseek4-timing] %s tokens=%d steps=%d wall=%.3fs %.2f tok/s "
        "step=%.1fms embed=%.1fms attn_build=%.1fms attn_compute=%.1fms attn_read=%.1fms "
        "ffn_build=%.1fms ffn_compute=%.1fms ffn_read=%.1fms "
        "route_build=%.1fms route_compute=%.1fms route_read=%.1fms route_select=%.1fms "
        "ffn=%.1fms hc_pre=%.1fms hc_pre_build=%.1fms hc_pre_input=%.1fms hc_pre_compute=%.1fms "
        "hc_post=%.1fms output=%.1fms sample=%.1fms emit=%.1fms\n",
        phase, tokens, steps, wall_s, tok_s,
        ms(t.total_us), ms(t.embed_us), ms(t.attn_build_us), ms(t.attn_compute_us), ms(t.attn_read_us),
        ms(t.ffn_build_us), ms(t.ffn_compute_us), ms(t.ffn_read_us),
        ms(t.route_build_us), ms(t.route_compute_us), ms(t.route_read_us), ms(t.route_select_us),
        ms(t.ffn_eval_us), ms(t.hc_pre_attn_us + t.hc_pre_ffn_us),
        ms(t.hc_pre_build_us), ms(t.hc_pre_input_us), ms(t.hc_pre_compute_us),
        ms(t.hc_post_attn_us + t.hc_post_ffn_us),
        ms(t.output_us), ms(t.sample_us), ms(t.emit_us));
}

}  // namespace

DeepSeek4Backend::DeepSeek4Backend(const DeepSeek4BackendConfig & cfg)
    : cfg_(cfg) {}

DeepSeek4Backend::~DeepSeek4Backend() {
    shutdown();
}

bool DeepSeek4Backend::load_full_model() {
    if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[deepseek4] failed to load full model weights\n");
        return false;
    }
    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    if (!create_deepseek4_cache(backend_, w_, max_ctx, cache_)) {
        std::fprintf(stderr, "[deepseek4] failed to create KV cache (ctx=%d)\n", max_ctx);
        free_deepseek4_weights(w_);
        return false;
    }
    hc_state_.clear();
    return true;
}

bool DeepSeek4Backend::init() {
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[deepseek4] failed to init backend gpu=%d\n", cfg_.device.gpu);
        return false;
    }
    snap_backend_ = ggml_backend_cpu_init();
    if (!snap_backend_) {
        std::fprintf(stderr, "[deepseek4] failed to init snapshot CPU backend\n");
        return false;
    }
    if (!load_full_model()) {
        return false;
    }
    return true;
}

void DeepSeek4Backend::print_ready_banner() const {
    std::printf("[deepseek4] ready (layers=%d, embd=%d, vocab=%d, full-weight backend)\n",
                w_.n_layer, w_.n_embd, w_.n_vocab);
    std::fflush(stdout);
}

bool DeepSeek4Backend::park(const std::string & what) {
    if (what != "target") return false;
    free_deepseek4_cache(cache_);
    free_deepseek4_weights(w_);
    hc_state_.clear();
    last_logits_.clear();
    parked_ = true;
    return true;
}

bool DeepSeek4Backend::unpark(const std::string & what) {
    if (what != "target") return false;
    if (!parked_) return true;
    if (!load_full_model()) return false;
    parked_ = false;
    return true;
}

int DeepSeek4Backend::do_prefill(const std::vector<int32_t> & tokens,
                                 const DaemonIO & io,
                                 int kv_offset,
                                 int snap_pos,
                                 int snap_slot) {
    const int chunk = cfg_.chunk > 0 ? cfg_.chunk : 512;
    const int n_total = (int)tokens.size();
    int pos = kv_offset;
    last_logits_.clear();
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;

    for (int i = 0; i < n_total; ) {
        if (io.cancelled) return pos;
        int n_tok = std::min(chunk, n_total - i);
        if (snap_slot >= 0 && snap_pos > pos && snap_pos < pos + n_tok) {
            n_tok = snap_pos - pos;
        }

        std::vector<float> embed((size_t)w_.n_embd * (size_t)n_tok);
        const auto embed_t0 = Clock::now();
        if (!w_.embedder.embed(tokens.data() + i, n_tok, embed.data())) {
            std::fprintf(stderr, "[deepseek4] embedding failed at pos=%d\n", pos);
            return -1;
        }
        DeepSeek4StepTelemetry step_tel;
        if (timing) step_tel.embed_us = elapsed_us(embed_t0, Clock::now());

        std::vector<float> logits;
        if (!deepseek4_step_layer_range(
                backend_, w_, cache_, hc_state_,
                embed.data(), n_tok, pos, 0, w_.n_layer,
                &logits, tokens.data() + i, timing ? &step_tel : nullptr)) {
            std::fprintf(stderr, "[deepseek4] prefill step failed at pos=%d n_tokens=%d\n", pos, n_tok);
            return -1;
        }
        last_logits_ = std::move(logits);
        pos += n_tok;
        if (snap_slot >= 0 && snap_pos >= 0 && pos == snap_pos) {
            if (snapshot_save(snap_slot)) {
                std::printf("[snap] inline slot=%d cur_pos=%d\n", snap_slot, snap_pos);
                std::fflush(stdout);
            }
        }
        if (timing) {
            add_step_tel(tel_acc, step_tel);
            steps++;
        }
        i += n_tok;
    }
    if (timing) {
        log_step_tel("prefill", n_total, steps, elapsed_s(phase_t0), tel_acc);
    }
    return pos;
}

bool DeepSeek4Backend::do_decode(int committed,
                                 int n_gen,
                                 std::vector<int32_t> & out_tokens,
                                 const DaemonIO & io,
                                 const BudgetHook & budget_hook,
                                 bool * forced_close_out) {
    if (forced_close_out) *forced_close_out = false;
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;
    int last_tok = -1;
    bool budget_close_started = false;
    int close_inject_pos = 0;
    const int committed_at_entry = committed;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size()) {
            tok = budget_hook.close_token_ids[(size_t)close_inject_pos++];
            return;
        }
        if (budget_close_started) return;
        const int generated = committed_now - committed_at_entry;
        const int remaining = n_gen - generated;
        if (remaining <= budget_hook.hard_limit_remaining) {
            const int32_t first_close = budget_hook.close_token_ids.front();
            if (tok == first_close) {
                budget_close_started = true;
                close_inject_pos = 1;
                return;
            }
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            if (forced_close_out) *forced_close_out = true;
        }
    };

    for (int step = 0; step < n_gen; ++step) {
        if (io.cancelled) return true;

        int32_t tok_to_eval;
        if (step == 0) {
            if (last_logits_.empty()) return false;
            if (sampler_.needs_logit_processing()) {
                tok_to_eval = sample_logits(last_logits_.data(), w_.n_vocab, sampler_, out_tokens, sampler_rng_);
            } else {
                tok_to_eval = (int)std::distance(last_logits_.begin(),
                    std::max_element(last_logits_.begin(), last_logits_.end()));
            }
        } else {
            tok_to_eval = last_tok;
        }
        maybe_force_close(tok_to_eval, committed + step);

        out_tokens.push_back(tok_to_eval);
        io.emit(tok_to_eval);
        if (deepseek4_is_eos_tok(tok_to_eval, w_)) {
            io.emit(-1);
            return true;
        }

        std::vector<float> embed((size_t)w_.n_embd);
        if (!w_.embedder.embed(&tok_to_eval, 1, embed.data())) return false;
        DeepSeek4StepTelemetry step_tel;
        std::vector<float> logits;
        if (!deepseek4_step_layer_range(
                backend_, w_, cache_, hc_state_,
                embed.data(), 1, committed + step, 0, w_.n_layer,
                &logits, &tok_to_eval, timing ? &step_tel : nullptr)) {
            return false;
        }
        last_logits_ = std::move(logits);
        if (timing) {
            add_step_tel(tel_acc, step_tel);
            steps++;
        }

        if (sampler_.needs_logit_processing()) {
            last_tok = sample_logits(last_logits_.data(), w_.n_vocab, sampler_, out_tokens, sampler_rng_);
        } else {
            last_tok = (int)std::distance(last_logits_.begin(),
                std::max_element(last_logits_.begin(), last_logits_.end()));
        }
    }

    if (timing) {
        log_step_tel("decode", n_gen, steps, elapsed_s(phase_t0), tel_acc);
    }
    io.emit(-1);
    return true;
}

GenerateResult DeepSeek4Backend::generate_impl(const GenerateRequest & req,
                                               const DaemonIO & io) {
    GenerateResult result;
    sampler_ = req.sampler;
    if (sampler_.seed != 0) sampler_rng_.seed(sampler_.seed);

    if (parked_ && !unpark("target")) {
        result.fail(GenerateErrorCode::ModelParked, "target_parked");
        return result;
    }
    if ((int)req.prompt.size() + req.n_gen + 1 > cache_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow);
        return result;
    }
    reset_full_request_state(cache_, hc_state_, last_logits_);

    const auto t0 = Clock::now();
    const int committed = do_prefill(req.prompt, io, 0, req.snap_pos, req.snap_slot);
    if (committed < 0) {
        result.fail(GenerateErrorCode::PrefillFailed);
        return result;
    }
    result.prefill_s = elapsed_s(t0);

    bool forced_close = false;
    if (!do_decode(committed, req.n_gen, result.tokens, io, req.budget_hook, &forced_close)) {
        result.fail(GenerateErrorCode::DecodeFailed);
        return result;
    }
    result.decode_s = elapsed_s(t0) - result.prefill_s;
    result.succeed();
    if (forced_close) result.budget_forced_close = true;
    return result;
}

bool DeepSeek4Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    if (!deepseek4_snapshot_save(cache_, snap_backend_, snapshots_[slot])) {
        return false;
    }
    snapshots_[slot].last_logits = last_logits_;
    return true;
}

void DeepSeek4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_deepseek4_snapshot(snapshots_[slot]);
}

bool DeepSeek4Backend::snapshot_used(int slot) const {
    return slot >= 0 && slot < PREFIX_SLOTS && snapshots_[slot].ctx != nullptr;
}

int DeepSeek4Backend::snapshot_cur_pos(int slot) const {
    return snapshot_used(slot) ? snapshots_[slot].cur_pos : 0;
}

GenerateResult DeepSeek4Backend::restore_and_generate_impl(int slot,
                                                           const GenerateRequest & req,
                                                           const DaemonIO & io) {
    GenerateResult result;
    if (parked_ && !unpark("target")) {
        result.fail(GenerateErrorCode::ModelParked, "target_parked");
        return result;
    }
    if (!snapshot_used(slot) || !deepseek4_snapshot_restore(snapshots_[slot], cache_)) {
        result.fail(GenerateErrorCode::InvalidSnapshotSlot, "restore");
        return result;
    }
    last_logits_ = snapshots_[slot].last_logits;

    sampler_ = req.sampler;
    if (sampler_.seed != 0) sampler_rng_.seed(sampler_.seed);
    const int snap_pos = cache_.cur_pos;
    if (snap_pos < 0) {
        result.fail(GenerateErrorCode::InvalidSnapshotSlot, "restore_pos");
        return result;
    }
    if (snap_pos > (int)req.prompt.size()) {
        std::fprintf(stderr,
            "[pc] deepseek4 snapshot longer than prompt (snap=%d > prompt=%zu) — "
            "fresh prefill fallback\n", snap_pos, req.prompt.size());
        reset_full_request_state(cache_, hc_state_, last_logits_);
        return generate_impl(req, io);
    }
    if ((int)req.prompt.size() + req.n_gen + 1 > cache_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow);
        return result;
    }
    std::vector<int32_t> remaining(req.prompt.begin() + snap_pos, req.prompt.end());
    const auto t0 = Clock::now();
    const int committed = remaining.empty() ? snap_pos : do_prefill(remaining, io, snap_pos);
    if (committed < 0) {
        result.fail(GenerateErrorCode::PrefillFailed);
        return result;
    }
    result.prefill_s = elapsed_s(t0);
    bool forced_close = false;
    if (!do_decode(committed, req.n_gen, result.tokens, io, req.budget_hook, &forced_close)) {
        result.fail(GenerateErrorCode::DecodeFailed);
        return result;
    }
    result.decode_s = elapsed_s(t0) - result.prefill_s;
    result.succeed();
    if (forced_close) result.budget_forced_close = true;
    return result;
}

bool DeepSeek4Backend::handle_compress(const std::string & line,
                                       const DaemonIO & io) {
    (void)line;
    (void)io;
    return false;
}

void DeepSeek4Backend::shutdown() {
    for (auto & snap : snapshots_) {
        free_deepseek4_snapshot(snap);
    }
    free_deepseek4_cache(cache_);
    free_deepseek4_weights(w_);
    if (snap_backend_) {
        ggml_backend_free(snap_backend_);
        snap_backend_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    hc_state_.clear();
    last_logits_.clear();
}

}  // namespace dflash::common
