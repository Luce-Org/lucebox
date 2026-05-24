// LagunaBackend implementation.
//
// Translates the model-specific code from the old monolithic
// run_laguna_daemon() into ModelBackend virtual methods. The protocol
// plumbing (command parsing, sampler, stream-fd) now lives in
// daemon_loop.cpp; this file only owns laguna's forward path, cache,
// snapshots, and pflash compress lifecycle.

#include "laguna_backend.h"
#include "laguna_internal.h"
#include "dflash27b.h"

#include "ggml-cuda.h"
#include "common/snapshot_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace dflash::common {

// ── Construction / initialisation ───────────────────────────────────────

LagunaBackend::LagunaBackend(const LagunaBackendArgs & args)
    : args_(args) {}

LagunaBackend::~LagunaBackend() { shutdown(); }

bool LagunaBackend::init() {
    backend_ = ggml_backend_cuda_init(0);
    if (!backend_) {
        std::fprintf(stderr, "cuda init failed\n");
        return false;
    }

    snap_backend_ = create_snapshot_backend(backend_);
    if (!snap_backend_) {
        std::fprintf(stderr, "snapshot backend init failed\n");
        ggml_backend_free(backend_); backend_ = nullptr;
        return false;
    }

    if (!load_target_gguf_laguna(args_.target_path, backend_, w_)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend_); backend_ = nullptr;
        return false;
    }

    cache_.kv_k_type = args_.kv_type;
    cache_.kv_v_type = args_.kv_type;
    if (!create_laguna_target_cache(w_, args_.max_ctx, backend_, cache_)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w_);
        ggml_backend_free(backend_); backend_ = nullptr;
        return false;
    }

    return true;
}

void LagunaBackend::print_ready_banner() const {
    std::printf("[laguna-daemon] ready vocab=%lld eos=%d eot=%d max_ctx=%d kv=%s chunk=%d\n",
                (long long)w_.embedder.n_vocab, w_.eos_id, w_.eos_chat_id,
                args_.max_ctx, ggml_type_name(args_.kv_type), args_.chunk);
}

// ── Park / unpark ───────────────────────────────────────────────────────

bool LagunaBackend::park(const std::string & what) {
    const bool want_target = (what.empty() || what == "all" || what == "target");
    // "park draft" is a no-op on the laguna path (no DFlash draft).
    if (want_target && !target_parked_) {
        free_laguna_target_cache(cache_);
        free_laguna_target_weights(w_);
        target_parked_ = true;
        std::printf("[park] target released\n"); std::fflush(stdout);
    }
    return true;
}

bool LagunaBackend::unpark(const std::string & what) {
    const bool want_target = (what.empty() || what == "all" || what == "target");
    if (want_target && target_parked_) {
        if (!load_target_gguf_laguna(args_.target_path, backend_, w_)) {
            std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
            return false;
        }
        cache_.kv_k_type = args_.kv_type;
        cache_.kv_v_type = args_.kv_type;
        if (!create_laguna_target_cache(w_, args_.max_ctx, backend_, cache_)) {
            std::fprintf(stderr, "[unpark] cache: %s\n", dflash27b_last_error());
            return false;
        }
        target_parked_ = false;
        std::printf("[unpark] target restored\n"); std::fflush(stdout);
    }
    return true;
}

// ── Snapshots ───────────────────────────────────────────────────────────

bool LagunaBackend::ensure_slot(int slot) {
    if (slot < 0 || slot >= kMaxSlots) {
        std::fprintf(stderr, "[snap] slot=%d out of range\n", slot);
        return false;
    }
    if (target_parked_) {
        std::fprintf(stderr, "[snap] target parked, cannot allocate snapshot\n");
        return false;
    }
    return true;
}

bool LagunaBackend::snapshot_save(int slot) {
    if (!ensure_slot(slot)) return false;
    if (!laguna_snapshot_save(cache_, snap_backend_, w_.n_layer,
                               w_.n_head_kv, w_.head_dim, snapshots_[slot])) {
        std::fprintf(stderr, "[snap] save slot=%d: %s\n",
                      slot, dflash27b_last_error());
        return false;
    }
    return true;
}

void LagunaBackend::snapshot_free(int slot) {
    if (slot >= 0 && slot < kMaxSlots) {
        laguna_snapshot_free(snapshots_[slot]);
    }
}

bool LagunaBackend::snapshot_used(int slot) const {
    return slot >= 0 && slot < kMaxSlots && snapshots_[slot].used;
}

int LagunaBackend::snapshot_cur_pos(int slot) const {
    if (slot >= 0 && slot < kMaxSlots && snapshots_[slot].used)
        return snapshots_[slot].cur_pos;
    return -1;
}

// ── Generation ──────────────────────────────────────────────────────────

GenerateResult LagunaBackend::generate(const GenerateRequest & req,
                                        const DaemonIO & io) {
    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    const bool should_emit = req.stream || (bool)out_io.on_token;
    const int N = (int)req.prompt.size();

    if (N + req.n_gen > args_.max_ctx) {
        result.error = "overflow";
        return result;
    }

    reset_laguna_target_cache(cache_);

    // ── Prefill ──
    std::vector<float> embed_pf((size_t)N * w_.n_embd);
    if (!w_.embedder.embed(req.prompt.data(), N, embed_pf.data())) {
        result.error = "embed_prefill";
        return result;
    }

    auto t_pf0 = std::chrono::steady_clock::now();
    std::vector<float> last_logits;
    bool ok = true;
    const int n_chunks = (N + args_.chunk - 1) / args_.chunk;
    for (int c = 0; c < n_chunks && ok; ++c) {
        const int kv_start = c * args_.chunk;
        const int n_tok    = std::min(args_.chunk, N - c * args_.chunk);
        ok = laguna_step(backend_, w_, cache_,
                          embed_pf.data() + (size_t)kv_start * w_.n_embd,
                          n_tok, kv_start, no_mask, last_logits);
    }
    if (!ok) { result.error = "prefill"; return result; }
    auto t_pf1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();

    // ── Inline snapshot (if requested) ──
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= N) {
        if (ensure_slot(req.snap_slot) &&
            laguna_snapshot_save(cache_, snap_backend_, w_.n_layer,
                                  w_.n_head_kv, w_.head_dim, snapshots_[req.snap_slot])) {
            snapshots_[req.snap_slot].cur_pos = req.snap_pos;
            std::printf("[snap] inline slot=%d cur_pos=%d\n",
                         req.snap_slot, req.snap_pos);
            std::fflush(stdout);
        }
    }

    // ── Decode ──
    auto argmax = [](const std::vector<float> & ll) {
        int best = 0; float bv = ll[0];
        for (size_t i = 1; i < ll.size(); ++i)
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        return best;
    };

    std::vector<int32_t> history;
    history.reserve((size_t)N + (size_t)req.n_gen);
    history.insert(history.end(), req.prompt.begin(), req.prompt.end());

    auto pick = [&](const std::vector<float> & ll) -> int {
        return req.do_sample
            ? sample_logits(ll.data(), (int)ll.size(), req.sampler, history, sampler_rng_)
            : argmax(ll);
    };

    int next_tok = pick(last_logits);
    result.tokens.reserve(req.n_gen);

    // Budget force-close state — see model_backend.h BudgetHook docs.
    // Mirrors qwen35/do_ar_decode's maybe_force_close. Laguna has no
    // spec-decode path so this is the only override site.
    const BudgetHook & budget_hook = req.budget_hook;
    bool budget_close_started = false;
    int  close_inject_pos     = 0;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size())
        {
            int32_t inj = budget_hook.close_token_ids[close_inject_pos];
            std::fprintf(stderr,
                "[budget-hook] laguna close-seq continue %d/%zu: overriding "
                "sampled token %d with %d\n",
                close_inject_pos + 1,
                budget_hook.close_token_ids.size(), tok, inj);
            tok = inj;
            close_inject_pos++;
            return;
        }
        if (budget_close_started) return;
        int remaining = req.n_gen - committed_now;
        if (remaining <= budget_hook.hard_limit_remaining) {
            int32_t first_close = budget_hook.close_token_ids.front();
            if (tok == first_close) {
                budget_close_started = true;
                close_inject_pos = 1;
                return;
            }
            std::fprintf(stderr,
                "[budget-hook] laguna force-close at committed=%d/%d "
                "(remaining=%d <= hard_limit=%d): overriding token %d "
                "with close[0]=%d (seq len %zu)\n",
                committed_now, req.n_gen, remaining,
                budget_hook.hard_limit_remaining, tok, first_close,
                budget_hook.close_token_ids.size());
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            result.budget_forced_close = true;
        }
    };

    std::vector<float> embed_step((size_t)w_.n_embd);
    auto t_g0 = std::chrono::steady_clock::now();
    for (int s = 0; s < req.n_gen; ++s) {
        maybe_force_close(next_tok, s);
        if (next_tok == w_.eos_id || next_tok == w_.eos_chat_id) break;
        result.tokens.push_back(next_tok);
        history.push_back(next_tok);
        if (should_emit) {
            out_io.emit(next_tok);
            if (out_io.cancelled) break;
        }
        if (!w_.embedder.embed(&next_tok, 1, embed_step.data())) { ok = false; break; }
        std::vector<float> step_logits;
        if (!laguna_step(backend_, w_, cache_, embed_step.data(), 1,
                          cache_.cur_pos, no_mask, step_logits)) { ok = false; break; }
        next_tok = pick(step_logits);
    }
    auto t_g1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_g1 - t_g0).count();

    if (should_emit) out_io.emit(-1);
    if (!ok) { result.error = "decode"; return result; }

    result.ok = true;
    return result;
}

// ── RESTORE + generate ──────────────────────────────────────────────────

GenerateResult LagunaBackend::restore_and_generate(int slot,
                                                     const GenerateRequest & req,
                                                     const DaemonIO & io) {
    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);

    if (!laguna_snapshot_restore(snapshots_[slot], cache_)) {
        std::fprintf(stderr, "[snap] RESTORE slot=%d: %s\n",
                      slot, dflash27b_last_error());
        result.error = "restore";
        return result;
    }

    const int prefix_len = cache_.cur_pos;
    const int N = (int)req.prompt.size();
    if (N < prefix_len) {
        std::fprintf(stderr, "[snap] RESTORE prompt shorter than cached prefix (%d < %d)\n",
                      N, prefix_len);
        result.error = "prefix_shorter";
        return result;
    }

    // Re-prefill diff tokens (or last cached token when diff is empty).
    if (prefix_len == N) {
        if (prefix_len <= 0) { result.error = "empty_diff"; return result; }
        cache_.cur_pos = prefix_len - 1;
    }
    const int kv_start = cache_.cur_pos;
    const int diff_n   = N - kv_start;

    std::vector<float> embed_diff((size_t)diff_n * w_.n_embd);
    if (!w_.embedder.embed(req.prompt.data() + kv_start, diff_n, embed_diff.data())) {
        result.error = "embed_prefill";
        return result;
    }

    std::vector<float> last_logits;
    bool ok = true;
    const int n_chunks = (diff_n + args_.chunk - 1) / args_.chunk;
    for (int c = 0; c < n_chunks && ok; ++c) {
        const int off   = c * args_.chunk;
        const int n_tok = std::min(args_.chunk, diff_n - off);
        const int starts = kv_start + off;
        ok = laguna_step(backend_, w_, cache_,
                          embed_diff.data() + (size_t)off * w_.n_embd,
                          n_tok, starts, no_mask, last_logits);
    }
    if (!ok) { result.error = "prefill"; return result; }

    // ── Decode ──
    auto argmax = [](const std::vector<float> & ll) {
        int best = 0; float bv = ll[0];
        for (size_t i = 1; i < ll.size(); ++i)
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        return best;
    };
    std::vector<int32_t> history(req.prompt);
    auto pick = [&](const std::vector<float> & ll) {
        return req.do_sample
            ? sample_logits(ll.data(), (int)ll.size(), req.sampler, history, sampler_rng_)
            : argmax(ll);
    };

    int next_tok = pick(last_logits);

    const BudgetHook & budget_hook = req.budget_hook;
    bool budget_close_started = false;
    int  close_inject_pos     = 0;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size())
        {
            int32_t inj = budget_hook.close_token_ids[close_inject_pos];
            std::fprintf(stderr,
                "[budget-hook] laguna(restore) close-seq continue %d/%zu: "
                "overriding sampled token %d with %d\n",
                close_inject_pos + 1,
                budget_hook.close_token_ids.size(), tok, inj);
            tok = inj;
            close_inject_pos++;
            return;
        }
        if (budget_close_started) return;
        int remaining = req.n_gen - committed_now;
        if (remaining <= budget_hook.hard_limit_remaining) {
            int32_t first_close = budget_hook.close_token_ids.front();
            if (tok == first_close) {
                budget_close_started = true;
                close_inject_pos = 1;
                return;
            }
            std::fprintf(stderr,
                "[budget-hook] laguna(restore) force-close at "
                "committed=%d/%d (remaining=%d <= hard_limit=%d): "
                "overriding token %d with close[0]=%d (seq len %zu)\n",
                committed_now, req.n_gen, remaining,
                budget_hook.hard_limit_remaining, tok, first_close,
                budget_hook.close_token_ids.size());
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            result.budget_forced_close = true;
        }
    };

    std::vector<float> embed_step((size_t)w_.n_embd);
    auto t_g0 = std::chrono::steady_clock::now();
    for (int s = 0; s < req.n_gen; ++s) {
        maybe_force_close(next_tok, s);
        if (next_tok == w_.eos_id || next_tok == w_.eos_chat_id) break;
        history.push_back(next_tok);
        result.tokens.push_back(next_tok);
        out_io.emit(next_tok);
        if (out_io.cancelled) break;
        if (!w_.embedder.embed(&next_tok, 1, embed_step.data())) { ok = false; break; }
        std::vector<float> step_logits;
        if (!laguna_step(backend_, w_, cache_, embed_step.data(), 1,
                          cache_.cur_pos, no_mask, step_logits)) { ok = false; break; }
        next_tok = pick(step_logits);
    }
    auto t_g1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_g1 - t_g0).count();

    out_io.emit(-1);
    if (!ok) { result.error = "decode"; return result; }

    result.ok = true;
    return result;
}

// ── Compress (pflash) ───────────────────────────────────────────────────

static std::vector<int32_t> read_uncounted_i32_local(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> ids(sz / sizeof(int32_t));
    if (!ids.empty()) {
        f.read(reinterpret_cast<char *>(ids.data()),
               (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

bool LagunaBackend::handle_compress(const std::string & line,
                                      const DaemonIO & io) {
    char ppath[1024];
    int  keep_x1000 = 0;
    char drafter_path[1024];
    const int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s",
                               ppath, &keep_x1000, drafter_path);
    if (n != 3) {
        std::fprintf(stderr,
                      "[compress] bad args, need: <bin> <keep_x1000> <drafter.gguf>\n");
        io.emit(-1);
        return true;
    }
    auto src_ids = read_uncounted_i32_local(ppath);
    if (src_ids.empty()) {
        std::fprintf(stderr, "[compress] empty input\n");
        io.emit(-1);
        return true;
    }

    const bool restore_target = !target_parked_;
    if (restore_target) park("target");

    if (!drafter_loaded_) {
        if (!load_drafter(drafter_path, /*gpu_layers=*/999, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] load_drafter failed: %s\n",
                          dflash27b_last_error());
            io.emit(-1);
            return true;
        }
        drafter_loaded_ = true;
        std::printf("[drafter] loaded %s vocab=%d\n",
                     drafter_path, drafter_ctx_.weights.n_vocab);
        std::fflush(stdout);
    }

    const float keep = (float)keep_x1000 / 1000.0f;
    auto compressed = drafter_score_and_compress(drafter_ctx_, src_ids, keep);
    std::printf("[compress] %zu -> %zu tokens (keep_ratio=%.3f)\n",
                 src_ids.size(), compressed.size(), keep);
    std::fflush(stdout);

    if (restore_target) unpark("target");

    for (int32_t t : compressed) io.emit(t);
    io.emit(-1);
    return true;
}

void LagunaBackend::free_drafter() {
    if (drafter_loaded_) {
        dflash::common::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
        std::printf("[drafter] freed\n"); std::fflush(stdout);
    }
}

// ── Shutdown ────────────────────────────────────────────────────────────

void LagunaBackend::shutdown() {
    for (auto & snap : snapshots_) laguna_snapshot_free(snap);
    if (drafter_loaded_) {
        dflash::common::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
    }
    if (!target_parked_) {
        free_laguna_target_cache(cache_);
        free_laguna_target_weights(w_);
    }
    free_snapshot_backend(snap_backend_, backend_);
    snap_backend_ = nullptr;
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace dflash::common
