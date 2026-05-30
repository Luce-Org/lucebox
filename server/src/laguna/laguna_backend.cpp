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

#include "../common/moe_hybrid_types.h"
#include "../common/moe_hybrid_types_impl.h"
#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_ffn_eval.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_swap_manager.h"
#include "common/step_graph.h"

#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "common/snapshot_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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

    // Check if hybrid mode is requested BEFORE full load
    const char * hotness_path = std::getenv("DFLASH_LAGUNA_HOTNESS");
    const char * budget_pct_env = std::getenv("DFLASH_EXPERT_BUDGET_PCT");
    if ((hotness_path && hotness_path[0]) || (budget_pct_env && budget_pct_env[0])) {
        // Hybrid mode: partial load (skip expert tensors)
        if (!init_hybrid_mode()) {
            ggml_backend_free(backend_); backend_ = nullptr;
            return false;
        }
    } else {
        // Full GPU mode: load everything
        if (!load_target_gguf_laguna(args_.target_path, backend_, w_)) {
            std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
            ggml_backend_free(backend_); backend_ = nullptr;
            return false;
        }
        // Print VRAM allocation summary
        size_t gpu_free = 0, gpu_total = 0;
        ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
        if (dev) ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
        const int n_moe_layers = w_.n_layer - w_.n_layer_dense_lead;
        const int total_experts = n_moe_layers * w_.n_expert;
        std::printf("[laguna] VRAM: %.2f GiB total, %.2f GiB used, %.2f GiB free\n",
                    gpu_total / 1024.0 / 1024.0 / 1024.0,
                    (gpu_total - gpu_free) / 1024.0 / 1024.0 / 1024.0,
                    gpu_free / 1024.0 / 1024.0 / 1024.0);
        std::printf("[laguna] all %d experts on GPU (%d MoE layers × %d experts)\n",
                    total_experts, n_moe_layers, w_.n_expert);
        std::fflush(stdout);
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
    if (hybrid_mode_ && moe_hybrid_) {
        auto result = generate_hybrid(req, io);
        if (result.ok) maybe_post_request_swap();
        return result;
    }

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

// ── Hybrid MoE mode ─────────────────────────────────────────────────────
//
// Layer-by-layer decode: for each token, iterate through all 40 layers.
// Layer 0 (dense SwiGLU) runs as a monolithic GPU sub-graph.
// Layers 1..39 (sparse MoE) run attention+router on GPU, read back expert
// selections, then call the common hybrid FFN eval (hot on GPU, cold on CPU).

using HybridClock = std::chrono::steady_clock;
static inline uint64_t elapsed_us(HybridClock::time_point t0, HybridClock::time_point t1) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

bool LagunaBackend::init_hybrid_mode() {
    const char * hotness_path = std::getenv("DFLASH_LAGUNA_HOTNESS");

    // Step 1: Load model WITHOUT expert data to GPU (partial load)
    if (!load_target_gguf_laguna_partial(args_.target_path, backend_, w_)) {
        std::fprintf(stderr, "[laguna-hybrid] partial load failed: %s\n", dflash27b_last_error());
        return false;
    }

    // Step 2: Load/build routing stats
    MoeHybridRoutingStats hotness;
    std::string err;
    if (hotness_path && hotness_path[0]) {
        if (!MoeHybridRoutingStats::load_csv(std::string(hotness_path), hotness, &err)) {
            std::fprintf(stderr, "[laguna-hybrid] hotness load failed: %s\n", err.c_str());
            return false;
        }
        if (hotness.n_layer != w_.n_layer || hotness.n_expert != w_.n_expert) {
            std::fprintf(stderr, "[laguna-hybrid] hotness dimensions mismatch (got %d×%d, want %d×%d)\n",
                          hotness.n_layer, hotness.n_expert, w_.n_layer, w_.n_expert);
            return false;
        }
    } else {
        // Uniform hotness (budget-only mode, no hotness file)
        hotness.n_layer = w_.n_layer;
        hotness.n_expert = w_.n_expert;
        hotness.n_expert_used = w_.n_expert_used;
        hotness.counts.assign((size_t)w_.n_layer * (size_t)w_.n_expert, 1);
        hotness.layer_totals.assign((size_t)w_.n_layer, (uint64_t)w_.n_expert);
    }

    // Step 3: Query GPU memory and compute expert budget
    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    if (dev) {
        ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
    }
    if (gpu_total == 0) {
        std::fprintf(stderr, "[laguna-hybrid] could not query GPU memory\n");
        return false;
    }

    // Compute per-layer expert size in bytes (laguna: separate gate/up/down, no fused)
    std::vector<uint64_t> layer_expert_bytes((size_t)w_.n_layer);
    for (int il = w_.n_layer_dense_lead; il < w_.n_layer; ++il) {
        const LagunaTargetLayer & L = w_.layers[(size_t)il];
        uint64_t bytes = 0;
        if (L.ffn_gate_exps) bytes += ggml_nbytes(L.ffn_gate_exps) / (uint64_t)w_.n_expert;
        if (L.ffn_up_exps)   bytes += ggml_nbytes(L.ffn_up_exps) / (uint64_t)w_.n_expert;
        if (L.ffn_down_exps) bytes += ggml_nbytes(L.ffn_down_exps) / (uint64_t)w_.n_expert;
        layer_expert_bytes[(size_t)il] = bytes;
    }
    // Layer 0 is dense — no experts
    for (int il = 0; il < w_.n_layer_dense_lead; ++il) {
        layer_expert_bytes[(size_t)il] = 0;
    }

    uint64_t total_expert_bytes = 0;
    for (int il = 0; il < w_.n_layer; ++il) {
        total_expert_bytes += layer_expert_bytes[(size_t)il] * (uint64_t)w_.n_expert;
    }

    // KV cache estimate
    const char * ctx_env = std::getenv("DFLASH_MAX_CONTEXT");
    int max_context = ctx_env ? std::atoi(ctx_env) : args_.max_ctx;
    if (max_context <= 0) max_context = 8192;

    const uint64_t kv_bytes_per_tok = (uint64_t)w_.n_layer * 2 *
        (uint64_t)w_.n_head_kv * (uint64_t)w_.head_dim * 2;
    const uint64_t kv_total = kv_bytes_per_tok * (uint64_t)max_context;

    const uint64_t warm_cache_bytes = 200ULL * 1024 * 1024;
    const uint64_t safety_bytes = 512ULL * 1024 * 1024;
    const uint64_t core_bytes = gpu_total - gpu_free;

    uint64_t expert_budget = 0;
    if (gpu_total > core_bytes + kv_total + warm_cache_bytes + safety_bytes) {
        expert_budget = gpu_total - core_bytes - kv_total - warm_cache_bytes - safety_bytes;
    }
    if (expert_budget > total_expert_bytes) {
        expert_budget = total_expert_bytes;
    }

    // Manual budget cap (absolute MB)
    if (const char * cap_env = std::getenv("DFLASH_EXPERT_BUDGET_MB")) {
        uint64_t cap_bytes = (uint64_t)std::atoi(cap_env) * 1024ULL * 1024ULL;
        if (cap_bytes > 0 && cap_bytes < expert_budget) {
            std::printf("[laguna-hybrid] capping expert budget from %.2f GiB to %d MB\n",
                        expert_budget / 1024.0 / 1024.0 / 1024.0, std::atoi(cap_env));
            expert_budget = cap_bytes;
        }
    }

    // Percentage-based budget cap
    if (const char * pct_env = std::getenv("DFLASH_EXPERT_BUDGET_PCT")) {
        int pct = std::atoi(pct_env);
        if (pct > 0 && pct < 100) {
            uint64_t pct_bytes = total_expert_bytes * (uint64_t)pct / 100ULL;
            if (pct_bytes < expert_budget) {
                std::printf("[laguna-hybrid] capping expert budget to %d%% = %.2f GiB (of %.2f GiB)\n",
                            pct, pct_bytes / 1024.0 / 1024.0 / 1024.0,
                            total_expert_bytes / 1024.0 / 1024.0 / 1024.0);
                expert_budget = pct_bytes;
            }
        }
    }

    std::printf("[laguna-hybrid] dynamic placement: gpu_total=%.2f GiB, core=%.2f GiB, "
                "kv_cache=%.2f GiB (ctx=%d), expert_budget=%.2f GiB (of %.2f GiB total experts)\n",
                gpu_total / 1024.0 / 1024.0 / 1024.0,
                core_bytes / 1024.0 / 1024.0 / 1024.0,
                kv_total / 1024.0 / 1024.0 / 1024.0,
                max_context,
                expert_budget / 1024.0 / 1024.0 / 1024.0,
                total_expert_bytes / 1024.0 / 1024.0 / 1024.0);
    std::fflush(stdout);

    if (expert_budget == 0) {
        std::fprintf(stderr, "[laguna-hybrid] no VRAM budget for experts\n");
        return false;
    }

    // Step 4: Build placement
    MoeHybridPlacement placement;
    if (!MoeHybridPlacement::build_from_stats_with_layer_bytes(
            hotness, layer_expert_bytes, expert_budget,
            /*min_hot_per_layer=*/std::min(w_.n_expert_used, w_.n_expert),
            placement, &err)) {
        std::fprintf(stderr, "[laguna-hybrid] placement build failed: %s\n", err.c_str());
        return false;
    }

    int total_moe_experts = (w_.n_layer - w_.n_layer_dense_lead) * w_.n_expert;
    std::printf("[laguna-hybrid] placement result: %d hot experts, %d cold experts\n",
                placement.total_hot, total_moe_experts - placement.total_hot);

    // If all experts fit, still use hybrid storage (partial load skipped expert buffers)
    if (placement.total_hot >= total_moe_experts) {
        std::printf("[laguna-hybrid] all experts fit in VRAM — all-hot hybrid path\n");
        std::fflush(stdout);
    }

    // Step 5: Load expert data from GGUF mmap into hot/cold split buffers
    {
        ggml_context * expert_meta = nullptr;
        gguf_init_params gip{};
        gip.no_alloc = true;
        gip.ctx = &expert_meta;
        gguf_context * gctx = gguf_init_from_file(args_.target_path.c_str(), gip);
        if (!gctx) {
            std::fprintf(stderr, "[laguna-hybrid] failed to re-open GGUF for expert loading\n");
            return false;
        }

        int fd = ::open(args_.target_path.c_str(), O_RDONLY);
        if (fd < 0) {
            gguf_free(gctx);
            std::fprintf(stderr, "[laguna-hybrid] open failed for mmap\n");
            return false;
        }
        struct stat st;
        if (::fstat(fd, &st) < 0) {
            ::close(fd);
            gguf_free(gctx);
            std::fprintf(stderr, "[laguna-hybrid] fstat failed\n");
            return false;
        }
        const size_t file_size = (size_t)st.st_size;
        void * mmap_addr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (mmap_addr == MAP_FAILED) {
            gguf_free(gctx);
            std::fprintf(stderr, "[laguna-hybrid] mmap failed\n");
            return false;
        }

        const size_t data_start = gguf_get_data_offset(gctx);
        const auto * file_bytes = (const uint8_t *)mmap_addr;

        // Build per-layer expert file data
        std::vector<LayerExpertFileData> layer_file_data((size_t)w_.n_layer);
        for (int il = w_.n_layer_dense_lead; il < w_.n_layer; ++il) {
            char name[128];
            auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
                std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
                int64_t tid = gguf_find_tensor(gctx, name);
                if (tid < 0) return {};
                size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
                size_t sz = gguf_get_tensor_size(gctx, tid);
                if (off + sz > file_size) return {};
                return { file_bytes + off, sz };
            };

            layer_file_data[(size_t)il].gate_exps    = find_tensor_data("ffn_gate_exps");
            layer_file_data[(size_t)il].up_exps      = find_tensor_data("ffn_up_exps");
            layer_file_data[(size_t)il].down_exps    = find_tensor_data("ffn_down_exps");
            // laguna has no fused gate_up_exps
        }

        auto hybrid = std::make_shared<MoeHybridStorage>();
        MoeHybridConfig hybrid_cfg = make_moe_hybrid_config(w_);
        std::vector<MoeLayerDesc> layer_descs((size_t)w_.n_layer);
        for (int il = 0; il < w_.n_layer; ++il) {
            layer_descs[(size_t)il] = make_moe_layer_desc(w_.layers[(size_t)il]);
        }
        if (!build_moe_hybrid_storage_from_file(hybrid_cfg, backend_, placement, layer_descs, layer_file_data, *hybrid, &err)) {
            ::munmap(mmap_addr, file_size);
            gguf_free(gctx);
            std::fprintf(stderr, "[laguna-hybrid] storage build failed: %s\n", err.c_str());
            return false;
        }

        ::munmap(mmap_addr, file_size);
        gguf_free(gctx);

        moe_hybrid_ = std::move(hybrid);
    }

    // Print stats
    int total_cold = 0;
    uint64_t hot_bytes = 0, cold_bytes = 0;
    for (int il = w_.n_layer_dense_lead; il < w_.n_layer; ++il) {
        const auto & layer = moe_hybrid_->layers[(size_t)il];
        total_cold += (int)layer.cold_expert_ids.size();
        const uint64_t per_expert_bytes =
            (uint64_t)layer.gate_expert_bytes + (uint64_t)layer.up_expert_bytes + (uint64_t)layer.down_expert_bytes;
        hot_bytes  += per_expert_bytes * (uint64_t)layer.hot_expert_ids.size();
        cold_bytes += per_expert_bytes * (uint64_t)layer.cold_expert_ids.size();
    }
    std::printf("[laguna-hybrid] storage ready: total_hot=%d (%.2f GiB VRAM) total_cold=%d (%.2f GiB RAM)\n",
                placement.total_hot,
                hot_bytes / 1024.0 / 1024.0 / 1024.0,
                total_cold,
                cold_bytes / 1024.0 / 1024.0 / 1024.0);

    if (total_cold > 0) {
        hybrid_mode_ = true;
        std::printf("[laguna-hybrid] hybrid decode path active (%d cold experts)\n", total_cold);
    } else {
        hybrid_mode_ = true;  // partial load: expert tensors only in hybrid storage
        std::printf("[laguna-hybrid] all experts hot — using hybrid path (all-hot)\n");
    }

    // Configure telemetry and swap policy
    if (const char * telemetry = std::getenv("DFLASH_LAGUNA_TELEMETRY")) {
        hybrid_telemetry_ = std::atoi(telemetry) != 0;
    }
    if (const char * out_path = std::getenv("DFLASH_LAGUNA_NEXT_PLACEMENT_OUT")) {
        routing_stats_out_path_ = out_path;
    }
    if (const char * swap_max = std::getenv("DFLASH_LAGUNA_SWAP_MAX")) {
        swap_policy_.max_swaps_total = std::max(0, std::atoi(swap_max));
    }
    if (const char * swap_gain = std::getenv("DFLASH_LAGUNA_SWAP_MIN_GAIN")) {
        swap_policy_.min_promote_gain = (uint64_t)std::max(1, std::atoi(swap_gain));
    }

    // Allocate routing stats collector
    if (!routing_stats_out_path_.empty()) {
        routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
        routing_stats_->n_layer = w_.n_layer;
        routing_stats_->n_expert = w_.n_expert;
        routing_stats_->n_expert_used = w_.n_expert_used;
        routing_stats_->counts.assign((size_t)w_.n_layer * (size_t)w_.n_expert, 0);
        routing_stats_->layer_totals.assign((size_t)w_.n_layer, 0);
    }

    std::fflush(stdout);
    return true;
}

// ── Laguna hybrid per-layer pre-FFN graph ───────────────────────────────
//
// Builds attention + router for a single layer. For MoE layers, outputs:
//   sg.ffn_post     = post-attention normed hidden (input to FFN)
//   sg.ffn_residual = residual to add after FFN output
//   sg.moe_selected = [n_used] expert IDs
//   sg.moe_weights  = [n_used] combine weights
// For the dense layer 0, outputs the full layer result in sg.hidden_input.

static bool build_laguna_layer_prefn_step(
    StepGraph & sg,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    ggml_backend_t backend,
    int il,
    int kv_start,
    int n_tokens)
{
    step_graph_free(sg);

    const int n_embd = w.n_embd;
    const bool is_full = laguna_is_full_attn_layer(w, il);
    const bool is_dense = (il < w.n_layer_dense_lead);
    const LagunaTargetLayer & L = w.layers[(size_t)il];
    const int kv_len = kv_start + n_tokens;
    const int n_head = w.n_head_arr[il];
    const int n_head_kv = w.n_head_kv;
    const int head_dim = w.head_dim;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead() + 8 * 1024 * 1024;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;
    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

    // Input: hidden state [n_embd, n_tokens]
    sg.inp_embed = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_name(sg.inp_embed, "inp_embed");

    // Positions
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(sg.positions);

    // Attention mask (causal)
    ggml_tensor * attn_mask = nullptr;
    if (kv_len > 0) {
        attn_mask = ggml_new_tensor_4d(sg.ctx, GGML_TYPE_F32, kv_len, n_tokens, 1, 1);
        ggml_set_input(attn_mask);
        sg.attn_mask = attn_mask;
    }

    ggml_tensor * inp = sg.inp_embed;

    // Pre-attn RMS norm
    ggml_tensor * cur = ggml_rms_norm(sg.ctx, inp, 1e-6f);
    cur = ggml_mul(sg.ctx, cur, L.attn_norm);

    // QKV projections
    const int q_dim = n_head * head_dim;
    ggml_tensor * Qcur = ggml_mul_mat(sg.ctx, L.wq, cur);  // [q_dim, n_tokens]
    ggml_tensor * Kcur = ggml_mul_mat(sg.ctx, L.wk, cur);  // [n_head_kv * head_dim, n_tokens]
    ggml_tensor * Vcur = ggml_mul_mat(sg.ctx, L.wv, cur);  // [n_head_kv * head_dim, n_tokens]

    // Per-head softplus gate
    ggml_tensor * gate = ggml_mul_mat(sg.ctx, L.wqkv_gate, cur);  // [n_head, n_tokens]
    gate = ggml_softplus(sg.ctx, gate);

    // Reshape Q to [head_dim, n_head, n_tokens]
    Qcur = ggml_reshape_3d(sg.ctx, Qcur, head_dim, n_head, n_tokens);
    Kcur = ggml_reshape_3d(sg.ctx, Kcur, head_dim, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(sg.ctx, Vcur, head_dim, n_head_kv, n_tokens);

    // Q-norm / K-norm
    Qcur = ggml_rms_norm(sg.ctx, Qcur, 1e-6f);
    Qcur = ggml_mul(sg.ctx, Qcur, L.q_norm);
    Kcur = ggml_rms_norm(sg.ctx, Kcur, 1e-6f);
    Kcur = ggml_mul(sg.ctx, Kcur, L.k_norm);

    // RoPE (YaRN on full-attention layers, plain on SWA layers)
    const float rope_th     = is_full ? w.rope_freq_base_full : w.rope_freq_base_swa;
    const int   n_rot       = is_full ? w.n_rot_full : w.n_rot_swa;
    const float ext_factor  = is_full ? 1.0f : 0.0f;
    const float attn_factor = 1.0f;
    const float beta_fast   = is_full ? w.yarn_beta_fast : 32.0f;
    const float beta_slow   = is_full ? w.yarn_beta_slow :  1.0f;
    const int   n_ctx_orig  = is_full ? w.yarn_orig_ctx  : 0;
    const float freq_scale  = is_full ? (1.0f / w.yarn_factor) : 1.0f;

    Qcur = ggml_rope_ext(sg.ctx, Qcur, sg.positions, /*freq_factors=*/nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX,
                          n_ctx_orig, rope_th, freq_scale,
                          ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(sg.ctx, Kcur, sg.positions, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX,
                          n_ctx_orig, rope_th, freq_scale,
                          ext_factor, attn_factor, beta_fast, beta_slow);

    // KV cache write — permute to [head_dim, n_tokens, n_head_kv] layout
    ggml_tensor * cache_k = cache.attn_k[(size_t)il];
    ggml_tensor * cache_v = cache.attn_v[(size_t)il];

    ggml_tensor * Kcur_T = ggml_permute(sg.ctx, Kcur, 0, 2, 1, 3);
    ggml_tensor * Vcur_T = ggml_permute(sg.ctx, Vcur, 0, 2, 1, 3);

    ggml_tensor * k_view = ggml_view_3d(sg.ctx, cache_k,
        head_dim, n_tokens, n_head_kv,
        cache_k->nb[1], cache_k->nb[2],
        cache_k->nb[1] * (size_t)kv_start);
    ggml_tensor * k_cpy = ggml_cpy(sg.ctx, Kcur_T, k_view);
    ggml_build_forward_expand(sg.gf, k_cpy);

    ggml_tensor * v_view = ggml_view_3d(sg.ctx, cache_v,
        head_dim, n_tokens, n_head_kv,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * (size_t)kv_start);
    ggml_tensor * v_cpy = ggml_cpy(sg.ctx, Vcur_T, v_view);
    ggml_build_forward_expand(sg.gf, v_cpy);

    // Flash attention
    ggml_tensor * Qfa = ggml_permute(sg.ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(sg.ctx, Qfa);

    ggml_tensor * Kfa = ggml_view_3d(sg.ctx, cache_k,
        head_dim, kv_len, n_head_kv,
        cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * Vfa = ggml_view_3d(sg.ctx, cache_v,
        head_dim, kv_len, n_head_kv,
        cache_v->nb[1], cache_v->nb[2], 0);

    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn_mask_f16 = attn_mask ? ggml_cast(sg.ctx, attn_mask, GGML_TYPE_F16) : nullptr;
    ggml_tensor * attn = ggml_flash_attn_ext(sg.ctx, Qfa, Kfa, Vfa, attn_mask_f16,
                                              kq_scale, 0.0f, 0.0f);

    // Per-head softplus gate
    ggml_tensor * gate_b = ggml_reshape_3d(sg.ctx, gate, 1, n_head, n_tokens);
    gate_b = ggml_cast(sg.ctx, gate_b, attn->type);
    attn = ggml_mul(sg.ctx, attn, gate_b);

    attn = ggml_reshape_2d(sg.ctx, attn, q_dim, n_tokens);

    // Output projection
    ggml_tensor * attn_out = ggml_mul_mat(sg.ctx, L.wo, attn);  // [n_embd, n_tokens]

    // Residual after attention
    ggml_tensor * ffn_inp = ggml_add(sg.ctx, attn_out, inp);

    if (is_dense) {
        // Dense layer 0: run full MLP in this graph
        ggml_tensor * normed = ggml_rms_norm(sg.ctx, ffn_inp, 1e-6f);
        normed = ggml_mul(sg.ctx, normed, L.ffn_norm);

        ggml_tensor * g = ggml_mul_mat(sg.ctx, L.w_gate, normed);
        ggml_tensor * u = ggml_mul_mat(sg.ctx, L.w_up, normed);
        ggml_tensor * gu = ggml_swiglu_split(sg.ctx, g, u);
        ggml_tensor * d = ggml_mul_mat(sg.ctx, L.w_down, gu);
        ggml_tensor * layer_out = ggml_add(sg.ctx, d, ffn_inp);

        sg.hidden_input = layer_out;
        ggml_set_output(layer_out);
        ggml_build_forward_expand(sg.gf, layer_out);
    } else {
        // MoE layer: output pre-FFN normed + residual + router decisions
        ggml_tensor * normed = ggml_rms_norm(sg.ctx, ffn_inp, 1e-6f);
        normed = ggml_mul(sg.ctx, normed, L.ffn_norm);
        sg.ffn_post = normed;
        ggml_set_output(normed);

        sg.ffn_residual = ffn_inp;
        ggml_set_output(ffn_inp);

        // Router: sigmoid + score-correction bias + top-k
        ggml_tensor * router_logits = ggml_mul_mat(sg.ctx, L.ffn_gate_inp, normed);
        ggml_tensor * probs = ggml_sigmoid(sg.ctx, router_logits);
        ggml_tensor * scores_sel = ggml_add(sg.ctx, probs, L.ffn_exp_probs_b);
        ggml_tensor * selected = ggml_top_k(sg.ctx, scores_sel, w.n_expert_used);
        ggml_set_output(selected);

        // Gather original probs (no bias) for combine weights
        ggml_tensor * probs_3d = ggml_reshape_3d(sg.ctx, probs, 1, w.n_expert, n_tokens);
        ggml_tensor * weights_raw = ggml_get_rows(sg.ctx, probs_3d, selected);
        weights_raw = ggml_reshape_2d(sg.ctx, weights_raw, w.n_expert_used, n_tokens);

        // Sum-normalize + scale
        ggml_tensor * w_sum = ggml_sum_rows(sg.ctx, weights_raw);
        ggml_tensor * weights_normed = ggml_div(sg.ctx, weights_raw, w_sum);
        if (w.expert_weights_scale != 1.0f) {
            weights_normed = ggml_scale(sg.ctx, weights_normed, w.expert_weights_scale);
        }
        sg.moe_weights = weights_normed;
        ggml_set_output(weights_normed);

        sg.moe_selected.resize(1);
        sg.moe_selected[0] = selected;

        ggml_build_forward_expand(sg.gf, normed);
        ggml_build_forward_expand(sg.gf, ffn_inp);
        ggml_build_forward_expand(sg.gf, selected);
        ggml_build_forward_expand(sg.gf, weights_normed);
    }

    // Allocate
    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) {
        return false;
    }

    return true;
}

// ── Hybrid forward: one token through all 40 layers ─────────────────────

bool LagunaBackend::hybrid_forward_one_token(int32_t tok, int kv_pos,
                                              std::vector<float> & act_cur,
                                              int32_t & argmax_out) {
    const int hidden = w_.n_embd;
    const int vocab = w_.embedder.n_vocab;

    // Embed token
    if (!w_.embedder.embed(&tok, 1, act_cur.data())) return false;

    // GPU-resident state for MoE layers
    GpuResidentState gpu_state;
    if (!init_gpu_resident_state(gpu_state, backend_, hidden)) return false;
    ggml_backend_tensor_set(gpu_state.act_cur, act_cur.data(), 0, sizeof(float) * (size_t)hidden);

    StepGraph layer_sg;
    std::vector<int32_t> selected((size_t)w_.n_expert_used);
    std::vector<float> weights_buf((size_t)w_.n_expert_used);
    ggml_backend_t cpu_be = moe_hybrid_->cpu_backend;

    for (int il = 0; il < w_.n_layer; ++il) {
        const bool is_dense = (il < w_.n_layer_dense_lead);

        if (!build_laguna_layer_prefn_step(layer_sg, w_, cache_, backend_, il, kv_pos, 1)) {
            step_graph_destroy(layer_sg);
            gpu_state.destroy();
            return false;
        }

        // GPU→GPU: copy persistent act_cur to pre-FFN graph input
        ggml_backend_tensor_copy(gpu_state.act_cur, layer_sg.inp_embed);

        // Set positions
        int32_t pos_val = kv_pos;
        ggml_backend_tensor_set(layer_sg.positions, &pos_val, 0, sizeof(int32_t));

        // Causal mask: single token decode — all positions [0..kv_pos] visible
        if (layer_sg.attn_mask) {
            const int kv_len = kv_pos + 1;
            std::vector<float> mask_data((size_t)kv_len, 0.0f);
            ggml_backend_tensor_set(layer_sg.attn_mask, mask_data.data(), 0, sizeof(float) * (size_t)kv_len);
        }

        auto st = ggml_backend_graph_compute(backend_, layer_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            step_graph_destroy(layer_sg);
            gpu_state.destroy();
            return false;
        }

        if (is_dense) {
            // Dense layer: read full output back to GPU-resident state
            ggml_backend_tensor_copy(layer_sg.hidden_input, gpu_state.act_cur);
        } else {
            // MoE layer: read router decisions, then do hybrid FFN eval
            ggml_tensor * sel_tensor = layer_sg.moe_selected[0];
            ggml_backend_tensor_get(sel_tensor, selected.data(), 0,
                                     sizeof(int32_t) * selected.size());
            ggml_backend_tensor_get(layer_sg.moe_weights, weights_buf.data(), 0,
                                     sizeof(float) * weights_buf.size());

            if (routing_stats_) {
                routing_stats_->observe(il, selected.data(), (int)selected.size());
            }

            // Hybrid FFN: hot on GPU, cold on CPU, combine on GPU
            auto & storage = moe_hybrid_->layers[(size_t)il];
            MoeHybridConfig cfg = make_moe_hybrid_config(w_);
            MoeLayerDesc desc = make_moe_layer_desc(w_.layers[(size_t)il]);
            if (!eval_moe_hybrid_ffn_gpu_resident(
                    backend_, cfg, desc, storage, cpu_be,
                    layer_sg.ffn_post, layer_sg.ffn_residual,
                    gpu_state,
                    selected.data(), weights_buf.data(),
                    (int)selected.size())) {
                step_graph_destroy(layer_sg);
                gpu_state.destroy();
                return false;
            }
        }
    }

    // Read final hidden state and project logits
    ggml_backend_tensor_get(gpu_state.act_cur, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
    step_graph_destroy(layer_sg);
    gpu_state.destroy();

    // Project logits: final RMS norm + lm_head
    {
        ggml_init_params ip{};
        ip.mem_size = 64 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;

        ggml_tensor * h_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
        ggml_set_input(h_in);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);

        ggml_tensor * normed = ggml_rms_norm(ctx, h_in, 1e-6f);
        normed = ggml_mul(ctx, normed, w_.out_norm);
        ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, normed);
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(h_in, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
        if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }

        std::vector<float> logits_buf((size_t)vocab);
        ggml_backend_tensor_get(logits, logits_buf.data(), 0, sizeof(float) * (size_t)vocab);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);

        // Argmax
        argmax_out = 0;
        float best = logits_buf[0];
        for (int j = 1; j < vocab; ++j) {
            if (logits_buf[(size_t)j] > best) {
                best = logits_buf[(size_t)j];
                argmax_out = j;
            }
        }
    }
    return true;
}

// ── Hybrid generate ─────────────────────────────────────────────────────

GenerateResult LagunaBackend::generate_hybrid(const GenerateRequest & req,
                                               const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    const bool should_emit = req.stream || (bool)out_io.on_token;
    const int N = (int)req.prompt.size();

    if (N + req.n_gen > args_.max_ctx) {
        result.error = "overflow";
        return result;
    }

    reset_laguna_target_cache(cache_);

    // ── Hybrid Prefill: layer-by-layer pre-FFN + batched hybrid FFN ──
    const int hidden = w_.n_embd;
    const int n_expert_used = w_.n_expert_used;
    ggml_backend_t cpu_be = moe_hybrid_->cpu_backend;

    std::vector<float> embed_all((size_t)N * (size_t)hidden);
    if (!w_.embedder.embed(req.prompt.data(), N, embed_all.data())) {
        result.error = "embed_prefill";
        return result;
    }

    auto t_pf0 = std::chrono::steady_clock::now();
    const int prefill_chunk = std::min(args_.chunk, N);

    for (int il = 0; il < w_.n_layer; ++il) {
        const bool is_dense = (il < w_.n_layer_dense_lead);
        const bool is_full = laguna_is_full_attn_layer(w_, il);

        for (int chunk_start = 0; chunk_start < N; chunk_start += prefill_chunk) {
            const int chunk_len = std::min(prefill_chunk, N - chunk_start);

            StepGraph prefill_sg;
            if (!build_laguna_layer_prefn_step(prefill_sg, w_, cache_, backend_,
                                               il, chunk_start, chunk_len)) {
                result.error = "prefill_build";
                step_graph_destroy(prefill_sg);
                return result;
            }

            // Set input embeddings
            ggml_backend_tensor_set(prefill_sg.inp_embed,
                                    embed_all.data() + (size_t)chunk_start * (size_t)hidden, 0,
                                    sizeof(float) * (size_t)chunk_len * (size_t)hidden);

            // Set positions
            std::vector<int32_t> pos_data((size_t)chunk_len);
            for (int i = 0; i < chunk_len; ++i) pos_data[i] = chunk_start + i;
            ggml_backend_tensor_set(prefill_sg.positions, pos_data.data(), 0,
                                    sizeof(int32_t) * (size_t)chunk_len);

            // Set attention mask (causal or causal+SWA depending on layer)
            if (prefill_sg.attn_mask) {
                const int kv_len = chunk_start + chunk_len;
                std::vector<float> mask((size_t)kv_len * (size_t)chunk_len, -INFINITY);
                for (int q = 0; q < chunk_len; ++q) {
                    const int abs_q = chunk_start + q;
                    const int win_lo = is_full ? 0 : std::max(0, abs_q - w_.sliding_window + 1);
                    for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                        mask[(size_t)q * (size_t)kv_len + (size_t)k] = 0.0f;
                    }
                }
                ggml_backend_tensor_set(prefill_sg.attn_mask, mask.data(), 0,
                                        sizeof(float) * mask.size());
            }

            // Compute pre-FFN graph
            auto st = ggml_backend_graph_compute(backend_, prefill_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                result.error = "prefill_compute";
                step_graph_destroy(prefill_sg);
                return result;
            }

            if (is_dense) {
                // Dense layer outputs full result directly
                std::vector<float> layer_out((size_t)chunk_len * (size_t)hidden);
                ggml_backend_tensor_get(prefill_sg.hidden_input, layer_out.data(), 0,
                                        sizeof(float) * layer_out.size());
                std::memcpy(embed_all.data() + (size_t)chunk_start * (size_t)hidden,
                            layer_out.data(),
                            sizeof(float) * layer_out.size());
            } else {
                // MoE layer: read router decisions, run hybrid FFN
                std::vector<float> chunk_residuals((size_t)chunk_len * (size_t)hidden);
                std::vector<float> chunk_post((size_t)chunk_len * (size_t)hidden);
                std::vector<int32_t> chunk_selected((size_t)chunk_len * (size_t)n_expert_used);
                std::vector<float> chunk_weights((size_t)chunk_len * (size_t)n_expert_used);

                ggml_backend_tensor_get(prefill_sg.ffn_residual, chunk_residuals.data(), 0,
                                        sizeof(float) * chunk_residuals.size());
                ggml_backend_tensor_get(prefill_sg.ffn_post, chunk_post.data(), 0,
                                        sizeof(float) * chunk_post.size());

                ggml_tensor * sel_tensor = prefill_sg.moe_selected.empty() ? nullptr : prefill_sg.moe_selected[0];
                if (!sel_tensor || !prefill_sg.moe_weights) {
                    result.error = "prefill_router_outputs";
                    step_graph_destroy(prefill_sg);
                    return result;
                }
                ggml_backend_tensor_get(sel_tensor, chunk_selected.data(), 0,
                                        sizeof(int32_t) * chunk_selected.size());
                ggml_backend_tensor_get(prefill_sg.moe_weights, chunk_weights.data(), 0,
                                        sizeof(float) * chunk_weights.size());

                // Observe routing stats
                if (routing_stats_) {
                    for (int i = 0; i < chunk_len; ++i) {
                        routing_stats_->observe(il, chunk_selected.data() + (size_t)i * (size_t)n_expert_used, n_expert_used);
                    }
                }

                // Batched hybrid FFN (sub-batched to avoid MMQ OOB)
                auto & storage = moe_hybrid_->layers[(size_t)il];
                MoeHybridConfig chunk_cfg = make_moe_hybrid_config(w_);
                MoeLayerDesc chunk_desc = make_moe_layer_desc(w_.layers[(size_t)il]);
                std::vector<float> ffn_batch_out((size_t)chunk_len * (size_t)hidden);
                constexpr int kFfnSafeBatch = 8;
                for (int fb = 0; fb < chunk_len; fb += kFfnSafeBatch) {
                    const int fl = std::min(kFfnSafeBatch, chunk_len - fb);
                    std::vector<float> sub_out;
                    if (!eval_moe_hybrid_ffn_batched(
                            backend_, cpu_be, chunk_cfg, chunk_desc, storage,
                            chunk_post.data()     + (size_t)fb * (size_t)hidden,
                            chunk_selected.data() + (size_t)fb * (size_t)n_expert_used,
                            chunk_weights.data()  + (size_t)fb * (size_t)n_expert_used,
                            fl, sub_out, &result.error)) {
                        step_graph_destroy(prefill_sg);
                        return result;
                    }
                    std::memcpy(ffn_batch_out.data() + (size_t)fb * (size_t)hidden,
                                sub_out.data(),
                                (size_t)fl * (size_t)hidden * sizeof(float));
                }

                // Combine: FFN output + residual → embed_all for next layer
                for (int i = 0; i < chunk_len; ++i) {
                    const float * ffn = ffn_batch_out.data() + (size_t)i * (size_t)hidden;
                    const float * res = chunk_residuals.data() + (size_t)i * (size_t)hidden;
                    float * out_embed = embed_all.data() + (size_t)(chunk_start + i) * (size_t)hidden;
                    for (int j = 0; j < hidden; ++j) {
                        out_embed[j] = ffn[j] + res[j];
                    }
                }
            }

            step_graph_destroy(prefill_sg);
        }
    }

    // Project logits from last token's hidden state
    cache_.cur_pos = N;
    std::vector<float> last_logits;
    {
        ggml_init_params ip{};
        ip.mem_size = 64 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);

        ggml_tensor * h_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
        ggml_set_input(h_in);
        ggml_tensor * normed = ggml_rms_norm(ctx, h_in, 1e-6f);
        normed = ggml_mul(ctx, normed, w_.out_norm);
        ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, normed);
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            result.error = "prefill_logits_alloc";
            return result;
        }
        // Set last token's hidden state
        ggml_backend_tensor_set(h_in,
                                embed_all.data() + (size_t)(N - 1) * (size_t)hidden, 0,
                                sizeof(float) * (size_t)hidden);
        if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            result.error = "prefill_logits_compute";
            return result;
        }
        last_logits.resize((size_t)w_.embedder.n_vocab);
        ggml_backend_tensor_get(logits, last_logits.data(), 0,
                                sizeof(float) * last_logits.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    auto t_pf1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();

    // ── Decode (hybrid layer-by-layer) ──
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

    // Budget force-close (same pattern as non-hybrid path)
    const BudgetHook & budget_hook = req.budget_hook;
    bool budget_close_started = false;
    int  close_inject_pos     = 0;
    auto maybe_force_close = [&](int32_t & tok, int committed_now) {
        if (budget_hook.close_token_ids.empty()) return;
        if (budget_close_started &&
            close_inject_pos < (int)budget_hook.close_token_ids.size())
        {
            tok = budget_hook.close_token_ids[close_inject_pos++];
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
            tok = first_close;
            budget_close_started = true;
            close_inject_pos = 1;
            result.budget_forced_close = true;
        }
    };

    std::vector<float> act_cur((size_t)w_.n_embd);
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

        // Hybrid forward: one token through all layers
        int32_t argmax_tok = 0;
        if (!hybrid_forward_one_token(next_tok, cache_.cur_pos, act_cur, argmax_tok)) {
            result.error = "decode";
            break;
        }
        cache_.cur_pos++;

        if (req.do_sample) {
            // For sampling, we need full logits — project from act_cur
            // (hybrid_forward_one_token already computed argmax; for sampling
            // we re-project — FIXME: return logits from forward to avoid double projection)
            next_tok = argmax_tok;  // For now, use argmax even in sample mode as fallback
        } else {
            next_tok = argmax_tok;
        }
    }
    auto t_g1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_g1 - t_g0).count();

    if (should_emit) out_io.emit(-1);
    result.ok = (result.error.empty());
    return result;
}

void LagunaBackend::maybe_post_request_swap() {
    if (!hybrid_mode_ || !moe_hybrid_ || swap_policy_.max_swaps_total <= 0) return;
    if (!routing_stats_) return;

    MoeHybridSwapPlan plan;
    std::string err;
    if (!build_moe_hybrid_swap_plan(moe_hybrid_->placement, *routing_stats_,
                                   swap_policy_, plan, &err)) {
        std::fprintf(stderr, "[laguna-hybrid] swap plan failed: %s\n", err.c_str());
        return;
    }
    if (plan.actions.empty()) return;

    // Rebuild storage with new placement
    auto rebuilt = std::make_shared<MoeHybridStorage>();
    MoeHybridConfig swap_cfg = make_moe_hybrid_config(w_);
    std::vector<MoeLayerDesc> swap_descs((size_t)w_.n_layer);
    for (int il = 0; il < w_.n_layer; ++il) {
        swap_descs[(size_t)il] = make_moe_layer_desc(w_.layers[(size_t)il]);
    }
    if (!build_moe_hybrid_storage(swap_cfg, backend_,
                                        plan.next_placement, swap_descs, *rebuilt, &err)) {
        std::fprintf(stderr, "[laguna-hybrid] swap rebuild failed: %s\n", err.c_str());
        return;
    }
    moe_hybrid_ = std::move(rebuilt);

    // Save updated routing stats if configured
    if (!routing_stats_out_path_.empty()) {
        routing_stats_->save_csv(routing_stats_out_path_, &err);
    }

    std::printf("[laguna-hybrid] applied %zu swap actions at request boundary\n", plan.actions.size());
    std::fflush(stdout);
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
