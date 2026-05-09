// Gemma4 DFlash speculative decoding end-to-end test / benchmark driver.
//
// Pipeline:
//   1. Load target (Gemma4-31B or 26B-A4B GGUF) + draft (z-lab Gemma4-DFlash
//      safetensors directory).
//   2. Prefill: chunked batched forward over prompt tokens (up to swa_window
//      tokens per chunk), capture_layers=true so target_feat gets populated.
//   3. Decode loop (until n_predict):
//      a. [target-only path, always active]
//         Run target forward for last committed token → logits → sample next.
//      b. [speculative path, active when draft is loaded]
//         i.  Get target_feat from cache.
//         ii. Run draft model to propose a block of tokens.
//         iii. Verify proposals against target in one batched forward.
//         iv. Accept longest verified prefix + bonus token, advance cache.
//   4. Print generated text and timing stats.
//
// Usage:
//   test_gemma4_dflash --model <gemma4.gguf> [--draft <dir>]
//                      [--prompt <text>] [--n-predict <N>]
//                      [--ctx-size <N>] [--kv-k <type>] [--kv-v <type>]
//                      [--seed <N>] [--temp <F>] [--top-k <N>] [--top-p <F>]
//                      [--budget <N>] [--gpu <N>] [--bench]

#include "internal.h"
#include "dflash27b.h"
#include "gemma4.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include "../src/pflash_ggml_adapter.h"

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <random>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace dflash27b;

// bf16→f32 CUDA conversion kernel (defined in f16_convert.cu)
extern "C" void dflash27b_launch_bf16_to_f32(const void * src,
                                             void * dst,
                                             size_t n_elems,
                                             cudaStream_t stream);

// ─── Utilities ────────────────────────────────────────────────────────────

static constexpr int    KQ_MASK_PAD  = 32;
static constexpr uint16_t F16_ZERO   = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int g_kq_stride_pad = KQ_MASK_PAD;

static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

static int argmax_f32(const float * x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > bv) { bv = x[i]; best = i; }
    }
    return best;
}

// ─── Sampler ──────────────────────────────────────────────────────────────

struct SamplerCfg {
    float    temp       = 0.0f;
    float    top_p      = 1.0f;
    int      top_k      = 0;
    float    rep_pen    = 1.0f;
    int      rep_window = 256;
    uint64_t seed       = 0;
};

static int sample_logits(const float * logits_in,
                         int vocab,
                         const SamplerCfg & cfg,
                         const std::vector<int32_t> & history,
                         std::mt19937_64 & rng) {
    if (cfg.temp <= 0.0f) {
        return argmax_f32(logits_in, vocab);
    }

    std::vector<std::pair<float, int>> cand(vocab);
    for (int i = 0; i < vocab; i++) cand[i] = {logits_in[i], i};

    if (cfg.rep_pen > 1.0f && !history.empty()) {
        const int win = std::min((int)history.size(), cfg.rep_window);
        const int from = (int)history.size() - win;
        std::unordered_set<int> seen;
        for (int i = from; i < (int)history.size(); i++) seen.insert(history[i]);
        for (auto & c : cand) {
            if (seen.count(c.second)) {
                c.first = (c.first > 0.0f) ? c.first / cfg.rep_pen
                                           : c.first * cfg.rep_pen;
            }
        }
    }

    if (cfg.top_k > 0 && cfg.top_k < vocab) {
        std::partial_sort(cand.begin(), cand.begin() + cfg.top_k, cand.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
        cand.resize(cfg.top_k);
    } else {
        std::sort(cand.begin(), cand.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
    }

    const float inv_t = 1.0f / std::max(1e-3f, cfg.temp);
    float maxv = cand.front().first * inv_t;
    double Z = 0.0;
    std::vector<float> probs(cand.size());
    for (size_t i = 0; i < cand.size(); i++) {
        probs[i] = std::exp(cand[i].first * inv_t - maxv);
        Z += probs[i];
    }
    for (auto & p : probs) p = (float)(p / Z);

    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        double cum = 0.0;
        size_t cut = probs.size();
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (cum >= cfg.top_p) { cut = i + 1; break; }
        }
        probs.resize(cut);
        cand.resize(cut);
        double zz = 0.0;
        for (auto p : probs) zz += p;
        for (auto & p : probs) p = (float)(p / zz);
    }

    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng);
    double acc = 0.0;
    for (size_t i = 0; i < probs.size(); i++) {
        acc += probs[i];
        if (r <= acc) return cand[i].second;
    }
    return cand.back().second;
}

// ─── Causal mask builder ──────────────────────────────────────────────────

static void build_causal_mask(std::vector<uint16_t> & out,
                              int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        for (int k = 0; k <= abs_q && k < kv_len; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
        }
    }
}

// ─── SWA causal mask builder (for chunked batched prefill) ───────────────────
//
// Non-monotonic ring mask.  The K view is always the full ring (ring_size slots,
// ring_win_start==0).  Slot k_view maps to absolute position via:
//   latest_slot = (kv_end - 1) % ring_size
//   offset_back = (latest_slot - k_view + ring_size) % ring_size
//   abs_k       = (kv_end - 1) - offset_back
//
// mask[q_idx][k_view_idx] = 0 (attend) iff:
//   abs_k >= (abs_q - swa_window + 1) AND abs_k <= abs_q AND abs_k >= 0
// else -inf.
static void build_swa_causal_mask(std::vector<uint16_t> & out,
                                   int kv_start,
                                   int n_tokens,
                                   int swa_window,
                                   int ring_size,    // = swa_view.effective_win_len = swa_ctx_alloc
                                   int kv_end) {     // = kv_start + n_tokens
    const int kv_pad = align_up(ring_size, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int latest_slot = ((kv_end - 1) % ring_size + ring_size) % ring_size;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int q_lo  = std::max(0, abs_q - swa_window + 1);
        for (int k_view = 0; k_view < ring_size; k_view++) {
            const int offset_back = (latest_slot - k_view + ring_size) % ring_size;
            const int abs_k       = (kv_end - 1) - offset_back;
            const bool valid = (abs_k >= q_lo && abs_k <= abs_q && abs_k >= 0);
            if (valid) {
                out[(size_t)q * kv_pad + k_view] = F16_ZERO;
            }
        }
    }
}

// ─── Per-step graph state (rebuilt each forward pass since kv_len varies) ─

struct StepGraph {
    ggml_context   * ctx        = nullptr;
    ggml_cgraph    * gf         = nullptr;
    ggml_gallocr_t   alloc      = nullptr;
    ggml_tensor    * inp_embed  = nullptr;
    ggml_tensor    * positions  = nullptr;
    ggml_tensor    * attn_mask  = nullptr;
    ggml_tensor    * swa_mask   = nullptr;
    ggml_tensor    * logits     = nullptr;
};

static void step_graph_free(StepGraph & sg) {
    if (sg.ctx) { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.gf        = nullptr;
    sg.inp_embed = nullptr;
    sg.positions = nullptr;
    sg.attn_mask = nullptr;
    sg.swa_mask  = nullptr;
    sg.logits    = nullptr;
}

static void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    step_graph_free(sg);
}

// ─── Draft step graph state ───────────────────────────────────────────────

struct DraftStepGraph {
    ggml_context   * ctx         = nullptr;
    ggml_cgraph    * gf          = nullptr;
    ggml_gallocr_t   alloc       = nullptr;
    ggml_tensor    * draft_embed = nullptr;
    ggml_tensor    * positions   = nullptr;
    ggml_tensor    * attn_mask   = nullptr;
    ggml_tensor    * logits      = nullptr;
};

static void draft_step_free(DraftStepGraph & dsg) {
    if (dsg.ctx) { ggml_free(dsg.ctx); dsg.ctx = nullptr; }
    dsg.gf          = nullptr;
    dsg.draft_embed = nullptr;
    dsg.positions   = nullptr;
    dsg.attn_mask   = nullptr;
    dsg.logits      = nullptr;
}

static void draft_step_destroy(DraftStepGraph & dsg) {
    if (dsg.alloc) { ggml_gallocr_free(dsg.alloc); dsg.alloc = nullptr; }
    draft_step_free(dsg);
}

// ─── Draft KV prefill graph state ────────────────────────────────────────────

struct DraftKVPrefillGraph {
    ggml_context   * ctx         = nullptr;
    ggml_cgraph    * gf          = nullptr;
    ggml_gallocr_t   alloc       = nullptr;
    ggml_tensor    * target_feat = nullptr;  // input: [6*target_hidden, n_tokens]
    ggml_tensor    * positions   = nullptr;  // input: [n_tokens] i32
};

static void draft_kv_prefill_free(DraftKVPrefillGraph & pkg) {
    if (pkg.ctx) { ggml_free(pkg.ctx); pkg.ctx = nullptr; }
    pkg.gf          = nullptr;
    pkg.target_feat = nullptr;
    pkg.positions   = nullptr;
}

static void draft_kv_prefill_destroy(DraftKVPrefillGraph & pkg) {
    if (pkg.alloc) { ggml_gallocr_free(pkg.alloc); pkg.alloc = nullptr; }
    draft_kv_prefill_free(pkg);
}

// Build a single-step target forward graph.
//   n_tokens  - number of tokens in this forward (1 for decode, >1 for prefill)
//   kv_start  - index of the first new token in the KV cache
//   with_mask - whether to allocate an attention-mask input (required for n_tokens > 1)
//   capture   - whether to write captured layer features to cache.target_feat
static bool build_gemma4_step(StepGraph & sg,
                              const GemmaTargetWeights & w,
                              GemmaTargetCache & cache,
                              ggml_backend_t backend,
                              int kv_start,
                              int n_tokens,
                              bool with_mask,
                              bool capture,
                              bool use_pflash   = false,
                              float pflash_alpha = 0.12f,
                              bool last_token_logits_only = false) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, w.n_embd, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        const int kv_len = kv_start + n_tokens;
        const int kv_pad = align_up(kv_len, g_kq_stride_pad);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);

        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
        ggml_set_output(sg.attn_mask);  // force gallocr to allocate even if no op references it

        if (n_tokens > 1) {
            // SWA mask needed for sliding-window attention layers in batched prefill.
            // Must be sized by the SWA window view, not the full kv_len, so that
            // its column count matches the K view that build_gemma4_graph passes to FA.
            const SwaView swa_view = compute_swa_view(kv_start, n_tokens,
                                                       w.swa_window, cache.swa_ctx_alloc);
            const int swa_kv_pad = align_up(swa_view.effective_win_len, g_kq_stride_pad);
            sg.swa_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, swa_kv_pad, q_pad);
            ggml_set_name(sg.swa_mask, "swa_mask");
            ggml_set_input(sg.swa_mask);
            ggml_set_output(sg.swa_mask);  // force gallocr to allocate even if no op references it
        }
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    GemmaGraphInputs gi{};
    gi.inp_embed      = sg.inp_embed;
    gi.positions      = sg.positions;
    gi.attn_mask      = sg.attn_mask;
    gi.swa_mask       = sg.swa_mask;
    gi.n_tokens       = n_tokens;
    gi.kv_start       = kv_start;
    gi.capture_layers           = capture;
    gi.use_pflash               = use_pflash;
    gi.pflash_alpha             = pflash_alpha;
    gi.last_token_logits_only   = last_token_logits_only;

    GemmaGraphOutputs go = build_gemma4_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// Build a draft KV prefill graph: project target features → draft KV cache.
static bool build_draft_kv_prefill(DraftKVPrefillGraph & pkg,
                                   const GemmaDraftWeights & dw,
                                   GemmaTargetCache & cache,
                                   ggml_backend_t backend,
                                   int n_tokens) {
    // Free previous graph state
    if (pkg.ctx) { ggml_free(pkg.ctx); pkg.ctx = nullptr; }
    pkg.gf          = nullptr;
    pkg.target_feat = nullptr;
    pkg.positions   = nullptr;

    const int target_feat_w = dw.n_target_layers * dw.target_hidden;

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    pkg.ctx = ggml_init(ip);
    if (!pkg.ctx) return false;

    pkg.target_feat = ggml_new_tensor_2d(pkg.ctx, GGML_TYPE_F32, target_feat_w, n_tokens);
    ggml_set_name(pkg.target_feat, "prefill_target_feat");
    ggml_set_input(pkg.target_feat);

    pkg.positions = ggml_new_tensor_1d(pkg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pkg.positions, "prefill_positions");
    ggml_set_input(pkg.positions);

    pkg.gf = ggml_new_graph_custom(pkg.ctx, 4096, false);

    build_draft_kv_prefill_graph(pkg.ctx, pkg.gf, dw, cache,
                                 pkg.target_feat, pkg.positions, n_tokens);

    if (!pkg.alloc) {
        pkg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(pkg.alloc, pkg.gf);
}

// Build a draft model forward graph for one diffusion step.
static bool build_draft_step(DraftStepGraph & dsg,
                             const GemmaDraftWeights & dw,
                             GemmaTargetCache & cache,
                             ggml_backend_t backend,
                             int n_tokens,
                             int kv_start) {
    draft_step_free(dsg);

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    dsg.ctx = ggml_init(ip);
    if (!dsg.ctx) return false;

    dsg.draft_embed = ggml_new_tensor_2d(dsg.ctx, GGML_TYPE_F32, dw.n_embd, n_tokens);
    ggml_set_name(dsg.draft_embed, "draft_embed");
    ggml_set_input(dsg.draft_embed);

    dsg.positions = ggml_new_tensor_1d(dsg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(dsg.positions, "positions");
    ggml_set_input(dsg.positions);

    // Attention mask: block tokens attend to context + block (causal).
    const int kv_len = kv_start + n_tokens;
    const int kv_pad = align_up(kv_len, KQ_MASK_PAD);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    dsg.attn_mask = ggml_new_tensor_2d(dsg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_name(dsg.attn_mask, "draft_attn_mask");
    ggml_set_input(dsg.attn_mask);

    dsg.gf = ggml_new_graph_custom(dsg.ctx, 8192, false);
    dsg.logits = build_gemma4_draft_graph(
        dsg.ctx, dsg.gf, dw, cache,
        dsg.draft_embed, dsg.positions, dsg.attn_mask,
        n_tokens, kv_start);
    if (!dsg.logits) return false;
    ggml_set_output(dsg.logits);
    ggml_build_forward_expand(dsg.gf, dsg.logits);

    if (!dsg.alloc) {
        dsg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(dsg.alloc, dsg.gf);
}

// ─── Embed one token into the inp_embed input tensor ─────────────────────

static bool embed_token(const GemmaTargetWeights & w,
                        int32_t token_id,
                        ggml_tensor * inp_embed,
                        ggml_backend_t backend) {
    const int hidden = w.n_embd;
    std::vector<float> emb(hidden);
    if (!w.embedder.embed(&token_id, 1, emb.data())) {
        std::fprintf(stderr, "[embed] failed for token %d\n", token_id);
        return false;
    }
    // inp_embed shape: [hidden, 1, 1]
    ggml_backend_tensor_set(inp_embed, emb.data(), 0, sizeof(float) * hidden);
    (void)backend;
    return true;
}

// Embed a batch of tokens (for chunked prefill).
static bool embed_tokens_batch(const GemmaTargetWeights & w,
                               const int32_t * ids,
                               int n,
                               ggml_tensor * inp_embed,
                               ggml_backend_t backend) {
    const int hidden = w.n_embd;
    std::vector<float> emb((size_t)hidden * n);
    if (!w.embedder.embed(ids, n, emb.data())) {
        std::fprintf(stderr, "[embed_batch] failed for %d tokens\n", n);
        return false;
    }
    ggml_backend_tensor_set(inp_embed, emb.data(), 0, sizeof(float) * hidden * n);
    (void)backend;
    return true;
}

// ─── EOS check ───────────────────────────────────────────────────────────

static bool g_ignore_eos = false;

#define IS_EOS_TOK(tok, w) \
    (!g_ignore_eos && \
     (((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id) || \
      ((w).eos_id      >= 0 && (tok) == (w).eos_id)))

// ─── KV type resolution helper ───────────────────────────────────────────

static ggml_type kv_type_from_string(const std::string & s) {
    if (s == "f16")   return GGML_TYPE_F16;
    if (s == "q8_0")  return GGML_TYPE_Q8_0;
    if (s == "q4_0")  return GGML_TYPE_Q4_0;
    if (s == "tq3_0") return GGML_TYPE_TQ3_0;
    return GGML_TYPE_Q8_0;  // default
}

// ─── Nanosecond wall clock ────────────────────────────────────────────────

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ─── Minimal tokenizer stub ──────────────────────────────────────────────
//
// A proper tokenizer (SentencePiece / tiktoken) requires linking to an
// external library. For benchmarking purposes we provide two modes:
//
//   1. Pre-tokenised input via --tokens <id0,id1,...>
//      Pass comma-separated integer token IDs directly. This is the
//      recommended path for reproducible benchmarks.
//
//   2. Byte-fallback: each byte of the --prompt string becomes one token.
//      This is NOT linguistically valid but lets the driver run without any
//      tokenizer library. Override with --tokens for real evaluation.

static std::vector<int32_t> tokenize_byte_fallback(const std::string & text) {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
        ids.push_back((int32_t)c);
    }
    return ids;
}

static std::vector<int32_t> parse_token_ids(const std::string & s) {
    std::vector<int32_t> ids;
    const char * p = s.c_str();
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        ids.push_back((int32_t)v);
        if (*end == '\0') break;
        if (*end == ',') { p = end + 1; continue; }
        break;
    }
    return ids;
}

// ─── Binary token file helper (daemon mode) ──────────────────────────────

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), (std::streamsize)sz);
    return out;
}

// Parse optional " samp=temp,top_p,top_k,rep_pen[,seed]" suffix from line.
// Erases the matched suffix from line. Returns true if parsed.
static bool parse_sampler_token(std::string & line, SamplerCfg & out) {
    auto pos = line.find(" samp=");
    if (pos == std::string::npos) return false;
    auto end = line.find(' ', pos + 1);
    std::string tok = (end == std::string::npos)
                          ? line.substr(pos + 6)
                          : line.substr(pos + 6, end - (pos + 6));
    line.erase(pos, (end == std::string::npos ? std::string::npos : end - pos));
    float t = 0.0f, tp = 1.0f, rp = 1.0f;
    int   tk = 0;
    unsigned long long sd = 0;
    int n = std::sscanf(tok.c_str(), "%f,%f,%d,%f,%llu",
                        &t, &tp, &tk, &rp, &sd);
    if (n < 1) return false;
    out.temp    = t;
    out.top_p   = tp;
    out.top_k   = tk;
    out.rep_pen = rp;
    out.seed    = sd;
    return true;
}

// ─── Main ─────────────────────────────────────────────────────────────────

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --model <gemma4.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --model  <path>   path to Gemma4 GGUF (target, required)\n"
        "  --draft  <dir>    path to z-lab DFlash safetensors directory (optional)\n"
        "  --prompt <text>   input prompt text (default: \"Hello, world!\")\n"
        "  --tokens <ids>    comma-separated prompt token IDs (overrides --prompt)\n"
        "  --tokens-file <path> read comma-separated token IDs from a file (for long prompts)\n"
        "  --n-predict <N>   max tokens to generate (default: 128)\n"
        "  --ctx-size  <N>   max context size (default: 4096)\n"
        "  --kv-k <type>     KV cache K type: f16/q8_0/q4_0/tq3_0 (default: q8_0)\n"
        "  --kv-v <type>     KV cache V type: f16/q8_0/q4_0/tq3_0 (default: q8_0)\n"
        "  --seed <N>        RNG seed (default: 0)\n"
        "  --temp <F>        temperature, 0 = greedy (default: 0.0)\n"
        "  --top-k <N>       top-k sampling, 0 = disabled (default: 0)\n"
        "  --top-p <F>       nucleus sampling (default: 1.0)\n"
        "  --budget <N>      DDTree budget for speculative decoding (default: 22)\n"
        "  --gpu <N>         CUDA device index (default: 0)\n"
        "  --bench           benchmark mode: repeat generation, report statistics\n"
        "  --fa-window <N>   sliding attention window for full layers (0 = full, default: 0)\n"
        "  --pflash          use pFlash prefill for prompts >= 4096 tokens\n"
        "  --pflash-alpha <F> pFlash block selection threshold (default: 0.12)\n"
        "\n",
        prog);
}

// Draft method selection
enum class DraftMethod { Auto, None, Dflash, Mtp };

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    // ── Parse CLI arguments ───────────────────────────────────────────────
    std::string  model_path;
    std::string  draft_path;
    std::string  mtp_path;
    std::string  prompt_text  = "Hello, world!";
    std::string  token_ids_str;
    std::string  tokens_file;
    int          n_predict    = 128;
    int          ctx_size     = 4096;
    std::string  kv_k_str     = "q8_0";
    std::string  kv_v_str     = "q8_0";
    int          gpu           = 0;
    int          ddtree_budget = 22;
    bool         bench_mode   = false;
    int          fa_window    = 0;
    bool         use_pflash   = false;
    float        pflash_alpha = 0.12f;
    SamplerCfg   sampler;
    bool         daemon_mode  = false;
    int          stream_fd    = -1;
    int          draft_max    = 0;   // 0 = use model's block_size (default 16)
    DraftMethod  draft_method = DraftMethod::Auto;

    for (int i = 1; i < argc; i++) {
        auto require_next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };

        if      (std::strcmp(argv[i], "--model")     == 0) model_path    = require_next("--model");
        else if (std::strcmp(argv[i], "--draft")     == 0) draft_path    = require_next("--draft");
        else if (std::strcmp(argv[i], "--prompt")    == 0) prompt_text   = require_next("--prompt");
        else if (std::strcmp(argv[i], "--tokens")      == 0) token_ids_str = require_next("--tokens");
        else if (std::strcmp(argv[i], "--tokens-file") == 0) tokens_file   = require_next("--tokens-file");
        else if (std::strcmp(argv[i], "--n-predict") == 0) n_predict     = std::atoi(require_next("--n-predict"));
        else if (std::strcmp(argv[i], "--ctx-size")  == 0) ctx_size      = std::atoi(require_next("--ctx-size"));
        else if (std::strncmp(argv[i], "--ctx-size=", 11) == 0) ctx_size = std::atoi(argv[i] + 11);
        else if (std::strcmp(argv[i], "--max-ctx")   == 0) ctx_size      = std::atoi(require_next("--max-ctx"));
        else if (std::strncmp(argv[i], "--max-ctx=", 10) == 0) ctx_size  = std::atoi(argv[i] + 10);
        else if (std::strcmp(argv[i], "--kv-k")      == 0) kv_k_str      = require_next("--kv-k");
        else if (std::strcmp(argv[i], "--kv-v")      == 0) kv_v_str      = require_next("--kv-v");
        else if (std::strcmp(argv[i], "--seed")      == 0) sampler.seed  = (uint64_t)std::atoll(require_next("--seed"));
        else if (std::strcmp(argv[i], "--temp")      == 0) sampler.temp  = (float)std::atof(require_next("--temp"));
        else if (std::strcmp(argv[i], "--ignore-eos")== 0) g_ignore_eos  = true;
        else if (std::strcmp(argv[i], "--top-k")     == 0) sampler.top_k = std::atoi(require_next("--top-k"));
        else if (std::strcmp(argv[i], "--top-p")     == 0) sampler.top_p = (float)std::atof(require_next("--top-p"));
        else if (std::strcmp(argv[i], "--budget")    == 0) ddtree_budget = std::atoi(require_next("--budget"));
        else if (std::strcmp(argv[i], "--gpu")       == 0) gpu           = std::atoi(require_next("--gpu"));
        else if (std::strcmp(argv[i], "--fa-window")    == 0) fa_window     = std::atoi(require_next("--fa-window"));
        else if (std::strcmp(argv[i], "--bench")        == 0) bench_mode    = true;
        else if (std::strcmp(argv[i], "--daemon")       == 0) daemon_mode   = true;
        else if (std::strcmp(argv[i], "--pflash")       == 0) use_pflash    = true;
        else if (std::strcmp(argv[i], "--pflash-alpha") == 0) pflash_alpha  = (float)std::atof(require_next("--pflash-alpha"));
        else if (std::strcmp(argv[i], "--draft-max")    == 0) draft_max     = std::atoi(require_next("--draft-max"));
        else if (std::strcmp(argv[i], "--mtp") == 0) mtp_path = require_next("--mtp");
        else if (std::strcmp(argv[i], "--draft-method") == 0) {
            const char * m = require_next("--draft-method");
            if      (std::strcmp(m, "none")   == 0) draft_method = DraftMethod::None;
            else if (std::strcmp(m, "dflash") == 0) draft_method = DraftMethod::Dflash;
            else if (std::strcmp(m, "mtp")    == 0) draft_method = DraftMethod::Mtp;
            else { std::fprintf(stderr, "error: unknown --draft-method %s\n", m); return 1; }
        }
        else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
        // No-op flags forwarded by server.py for Qwen3 compatibility:
        else if (std::strcmp(argv[i], "--fast-rollback")  == 0) { /* no-op */ }
        else if (std::strcmp(argv[i], "--ddtree")         == 0) { /* no-op */ }
        else if (std::strncmp(argv[i], "--ddtree-budget=", 16) == 0) { /* no-op */ }
        else if (std::strncmp(argv[i], "--ddtree-temp=",   14) == 0) { /* no-op */ }
        else if (std::strcmp(argv[i], "--ddtree-no-chain-seed") == 0) { /* no-op */ }
        else if (std::strcmp(argv[i], "--help")      == 0 ||
                 std::strcmp(argv[i], "-h")          == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "warning: unknown argument: %s\n", argv[i]);
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return 2;
    }

    // ── Resolve Auto draft method ─────────────────────────────────────────
    if (draft_method == DraftMethod::Auto) {
        if (!draft_path.empty() && !mtp_path.empty()) {
            std::fprintf(stderr, "error: both --draft and --mtp provided; use --draft-method to disambiguate\n");
            return 1;
        } else if (!mtp_path.empty()) {
            draft_method = DraftMethod::Mtp;
        } else if (!draft_path.empty()) {
            draft_method = DraftMethod::Dflash;
        } else {
            draft_method = DraftMethod::None;
        }
    }
    if (draft_method == DraftMethod::Mtp && mtp_path.empty()) {
        std::fprintf(stderr, "error: --draft-method mtp requires --mtp <path>\n");
        return 1;
    }
    if (draft_method == DraftMethod::Dflash && draft_path.empty()) {
        std::fprintf(stderr, "error: --draft-method dflash requires --draft <path>\n");
        return 1;
    }

    const bool have_draft = (draft_method == DraftMethod::Dflash);
    const bool have_mtp   = (draft_method == DraftMethod::Mtp);

    // ── Load token IDs from file if --tokens-file was specified ──────────
    if (!tokens_file.empty()) {
        FILE * f = fopen(tokens_file.c_str(), "r");
        if (!f) {
            std::fprintf(stderr, "error: cannot open tokens file: %s\n", tokens_file.c_str());
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        rewind(f);
        std::string content(sz, '\0');
        fread(&content[0], 1, sz, f);
        fclose(f);
        token_ids_str = content;
    }

    // ── KV type env vars (consumed by create_gemma4_cache → resolve_kv_types) ─
    setenv("DFLASH27B_KV_K", kv_k_str.c_str(), 1);
    setenv("DFLASH27B_KV_V", kv_v_str.c_str(), 1);

    // After argv parsing, the KV type may have been chosen via --kv-k tq3_0 / --kv-v tq3_0,
    // which sets DFLASH27B_KV_K / DFLASH27B_KV_V env vars. Re-check for TQ3 here so
    // g_kq_stride_pad matches the chunked-FA driver's align_up(kv_len, 256); otherwise the
    // host-built mask is short and the kernel reads past its end.
    auto kv_env_is_tq3 = [](const char * name) {
        const char * s = std::getenv(name);
        if (!s) return false;
        std::string lc;
        for (const char * p = s; *p; ++p) lc += (char)std::tolower((unsigned char)*p);
        return lc.rfind("tq3", 0) == 0;
    };
    if (kv_env_is_tq3("DFLASH27B_KV_K") || kv_env_is_tq3("DFLASH27B_KV_V")) {
        g_kq_stride_pad = 256;
    }

    // ── CUDA device validation ────────────────────────────────────────────
    int cuda_device_count = 0;
    cudaGetDeviceCount(&cuda_device_count);
    if (gpu >= cuda_device_count) {
        std::fprintf(stderr, "error: --gpu %d out of range (device_count=%d)\n",
                     gpu, cuda_device_count);
        return 2;
    }
    cudaSetDevice(gpu);

    std::printf("[cfg] model=%s draft=%s gpu=%d ctx=%d n_predict=%d kv_k=%s kv_v=%s "
                "temp=%.2f top_k=%d top_p=%.2f budget=%d bench=%d fa_window=%d\n",
                model_path.c_str(),
                draft_path.empty() ? "(none)" : draft_path.c_str(),
                gpu, ctx_size, n_predict,
                kv_k_str.c_str(), kv_v_str.c_str(),
                sampler.temp, sampler.top_k, sampler.top_p,
                ddtree_budget, (int)bench_mode, fa_window);

    // ── Backend init ──────────────────────────────────────────────────────
    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        std::fprintf(stderr, "error: ggml_backend_cuda_init(%d) failed\n", gpu);
        return 1;
    }

    // Register the pFlash GGML custom kernel so ggml_flash_attn_sparse ops
    // dispatched from build_gemma4_graph (full-attention layers, use_pflash=true)
    // have a backend implementation available.
    if (use_pflash) {
        pflash_register_ggml_kernel();
    }

    // ── Load target weights ───────────────────────────────────────────────
    GemmaTargetWeights w;
    {
        double t0 = now_ms();
        if (!load_gemma4_target_gguf(model_path, backend, w)) {
            std::fprintf(stderr, "load_gemma4_target_gguf: %s\n", dflash27b_last_error());
            return 1;
        }
        double t1 = now_ms();
        std::printf("[target] loaded %d layers, n_embd=%d, vocab=%d  (%.1f ms)\n",
                    w.n_layer, w.n_embd, w.n_vocab, t1 - t0);
    }

    // ── Load draft weights (optional) ────────────────────────────────────
    // Draft state: declared in main scope so they persist across bench iterations
    // and are accessible in cleanup.
    GemmaDraftWeights    dw;
    ggml_context       * tok_embd_ctx = nullptr;
    ggml_backend_buffer_t tok_embd_buf = nullptr;

    if (have_draft) {
        double t0 = now_ms();
        // Auto-detect:
        //   1. If path ends with .gguf, use GGUF loader directly
        //   2. If path is a directory containing draft-q8_0.gguf, prefer it
        //      (Q8 GGUF is ~2x smaller than the BF16 safetensors and avoids
        //      a memory-pressure perf trap on Dense + TQ3 KV that drops
        //      target prefill 20x; see commit notes for details)
        //   3. Otherwise fall back to safetensors loader
        std::string resolved_draft_path = draft_path;
        bool is_gguf = (draft_path.size() >= 5 &&
                        draft_path.compare(draft_path.size() - 5, 5, ".gguf") == 0);
        if (!is_gguf) {
            // Check if path is a directory with a draft-q8_0.gguf inside
            const std::string candidate = draft_path + "/draft-q8_0.gguf";
            std::ifstream probe(candidate.c_str());
            if (probe.good()) {
                resolved_draft_path = candidate;
                is_gguf = true;
                std::fprintf(stderr,
                    "[draft] auto-selected Q8 GGUF: %s\n"
                    "        (%s also present; Q8 is ~2x smaller and ~20x faster on Dense+TQ3)\n",
                    candidate.c_str(),
                    (draft_path + "/model.safetensors").c_str());
            }
        }
        bool ok = false;
        if (is_gguf) {
            ok = load_gemma4_draft_gguf(resolved_draft_path, backend, dw);
            if (!ok) std::fprintf(stderr, "load_gemma4_draft_gguf: %s\n", dflash27b_last_error());
        } else {
            ok = load_gemma4_draft_safetensors(resolved_draft_path, backend, dw);
            if (!ok) std::fprintf(stderr, "load_gemma4_draft_safetensors: %s\n", dflash27b_last_error());
        }
        if (!ok) return 1;
        double t1 = now_ms();

        // Upload tok_embd from target embedder to GPU (tied lm_head for draft).
        // tw.embedder keeps the bytes CPU-side; we upload once and inject a pointer.
        {
            ggml_init_params ep{};
            ep.mem_size   = ggml_tensor_overhead() * 2;
            ep.mem_buffer = nullptr;
            ep.no_alloc   = true;
            tok_embd_ctx = ggml_init(ep);
            if (!tok_embd_ctx) {
                std::fprintf(stderr, "[draft] ggml_init for tok_embd failed\n");
                return 1;
            }

            const ggml_type emb_type  = w.embedder.tok_embd_type;
            const int64_t   n_embd_t  = w.embedder.n_embd;
            const int64_t   n_vocab_t = w.embedder.n_vocab;

            // ggml convention: ne[0] = n_embd (fast axis), ne[1] = n_vocab
            ggml_tensor * te = ggml_new_tensor_2d(tok_embd_ctx, emb_type, n_embd_t, n_vocab_t);
            ggml_set_name(te, "tok_embd_gpu");

            tok_embd_buf = ggml_backend_alloc_ctx_tensors(tok_embd_ctx, backend);
            if (!tok_embd_buf) {
                std::fprintf(stderr, "[draft] ggml_backend_alloc_ctx_tensors for tok_embd failed\n");
                ggml_free(tok_embd_ctx);
                tok_embd_ctx = nullptr;
                return 1;
            }

            const size_t emb_bytes = (size_t)w.embedder.row_bytes * (size_t)n_vocab_t;
            ggml_backend_tensor_set(te, w.embedder.tok_embd_bytes, 0, emb_bytes);
            std::printf("[tok_embd] uploaded %.1f MiB to GPU (%s [%" PRId64 ", %" PRId64 "])\n",
                        (double)emb_bytes / (1024.0 * 1024.0),
                        ggml_type_name(emb_type), n_embd_t, n_vocab_t);

            dw.tok_embd = te;
            dw.n_vocab  = (int)n_vocab_t;
        }

        std::printf("[draft] loaded n_layer=%d n_head=%d n_embd=%d n_vocab=%d "
                    "target_hidden=%d block_size=%d  (%.1f ms)\n",
                    dw.n_layer, dw.n_head, dw.n_embd, dw.n_vocab,
                    dw.target_hidden, dw.block_size, t1 - t0);
    }

    // ── Create KV cache ───────────────────────────────────────────────────
    GemmaTargetCache cache;
    {
        double t0 = now_ms();
        if (!create_gemma4_cache(w, ctx_size, backend, cache)) {
            std::fprintf(stderr, "create_gemma4_cache: %s\n", dflash27b_last_error());
            return 1;
        }
        double t1 = now_ms();
        std::printf("[cache] created max_ctx=%d, kv_layers=%zu  (%.1f ms)\n",
                    cache.max_ctx, cache.attn_k.size(), t1 - t0);
    }

    // ── Allocate draft KV cache (requires cache to already exist) ─────────
    if (have_draft) {
        if (!create_draft_kv_cache(dw, backend, cache)) {
            std::fprintf(stderr, "create_draft_kv_cache failed\n");
            return 1;
        }
        std::printf("[draft] KV cache allocated: %d slots\n", cache.draft_kv_cap);
    }

    // ── MTP weights + step graph (optional) ──────────────────────────────
    MtpDrafterWeights mtp_w;
    MtpStepGraph      mtp_g;
    // mtp_h_prev context/buffer: separate small allocation so base_ctx stays
    // unmodified and free_gemma4_cache() doesn't double-free it.
    ggml_context        * mtp_h_prev_ctx = nullptr;
    ggml_backend_buffer_t mtp_h_prev_buf = nullptr;

    if (have_mtp) {
        double t0 = now_ms();
        if (!load_gemma4_mtp_assistant(mtp_path, backend, mtp_w)) {
            std::fprintf(stderr, "load_gemma4_mtp_assistant: %s\n", dflash27b_last_error());
            return 1;
        }
        double t1 = now_ms();
        std::printf("[mtp] loaded n_layers=%d n_embd=%d n_embd_backbone=%d  (%.1f ms)\n",
                    (int)mtp_w.layers.size(), mtp_w.n_embd, mtp_w.n_embd_backbone, t1 - t0);

        // Re-resolve donor target layers using the actual target SWA pattern.
        // The loader uses a hardcoded alternating assumption; the real pattern
        // from the GGUF may differ (e.g., layer 59 may be full-attention, not SWA).
        resolve_mtp_donor_layers(mtp_w, w.swa_layers);

        // Allocate mtp_h_prev tensor: [n_embd_backbone, 1] f32, GPU-resident,
        // persistent across decode steps. Separate context so free_gemma4_cache
        // doesn't free it.
        {
            ggml_init_params ep{};
            ep.mem_size   = ggml_tensor_overhead() + 256;
            ep.mem_buffer = nullptr;
            ep.no_alloc   = true;
            mtp_h_prev_ctx = ggml_init(ep);
            if (!mtp_h_prev_ctx) {
                std::fprintf(stderr, "[mtp] ggml_init for mtp_h_prev failed\n");
                return 1;
            }
            cache.mtp_h_prev = ggml_new_tensor_2d(mtp_h_prev_ctx,
                                                    GGML_TYPE_F32,
                                                    mtp_w.n_embd_backbone, 1);
            ggml_set_name(cache.mtp_h_prev, "mtp_h_prev");
            mtp_h_prev_buf = ggml_backend_alloc_ctx_tensors(mtp_h_prev_ctx, backend);
            if (!mtp_h_prev_buf) {
                std::fprintf(stderr, "[mtp] alloc mtp_h_prev failed\n");
                ggml_free(mtp_h_prev_ctx); mtp_h_prev_ctx = nullptr;
                return 1;
            }
            // Zero-initialize
            std::vector<float> zeros_f(mtp_w.n_embd_backbone, 0.0f);
            ggml_backend_tensor_set(cache.mtp_h_prev, zeros_f.data(), 0,
                                    sizeof(float) * mtp_w.n_embd_backbone);
        }

        // Determine last full-attention layer index from swa_layers
        cache.mtp_last_full_layer = -1;
        for (int il = w.n_layer - 1; il >= 0; il--) {
            const bool is_swa = (il < (int)w.swa_layers.size()) && w.swa_layers[il];
            if (!is_swa) {
                cache.mtp_last_full_layer = il;
                break;
            }
        }
        if (cache.mtp_last_full_layer < 0) {
            std::fprintf(stderr, "[mtp] error: no full-attention layer found in target\n");
            return 1;
        }
        std::printf("[mtp] mtp_last_full_layer=%d\n", cache.mtp_last_full_layer);

        cache.mtp_h_prev_enabled = true;

        // Build the MTP step graph (attn_pos=0 initially; will be rebuilt per step)
        if (!build_mtp_step_graph(mtp_w, cache, w, mtp_g, /*attn_pos=*/0)) {
            std::fprintf(stderr, "build_mtp_step_graph: %s\n", dflash27b_last_error());
            return 1;
        }
        std::printf("[mtp] step graph built ok\n");
    }

    // ── RNG ───────────────────────────────────────────────────────────────
    std::mt19937_64 rng(sampler.seed);

    // ── Daemon mode: stream token fd write helper ─────────────────────────
    auto stream_emit = [&](int32_t tok) {
        if (stream_fd < 0) return;
        int32_t v = tok;
#ifdef _WIN32
        DWORD written;
        WriteFile((HANDLE)(intptr_t)stream_fd, &v, sizeof(v), &written, nullptr);
#else
        ssize_t n = ::write(stream_fd, &v, sizeof(v));
        (void)n;
#endif
    };

    // ── Daemon mode ───────────────────────────────────────────────────────
    if (daemon_mode) {
        std::printf("[daemon] ready\n");
        std::fflush(stdout);

        StepGraph      sg;
        DraftStepGraph dsg;
        bool daemon_first_iter = true;
        std::string line;

        while (std::getline(std::cin, line)) {
            // Per-request sampler (reset to CLI defaults each request).
            SamplerCfg req_sampler = sampler;
            parse_sampler_token(line, req_sampler);
            // Always reseed per request so requests are independent.
            // seed==0 means "random": use std::random_device for a fresh seed.
            uint64_t actual_seed = req_sampler.seed;
            if (actual_seed == 0) {
                actual_seed = std::random_device{}();
            }
            rng.seed(actual_seed);

            // ── Unsupported commands: emit -1 sentinel and continue ────────
            auto starts_with = [](const std::string & s, const char * pre) {
                size_t n = std::strlen(pre);
                return s.size() >= n && s.compare(0, n, pre) == 0;
            };
            bool unsupported = (starts_with(line, "RESTORE")       ||
                                starts_with(line, "SNAPSHOT")      ||
                                starts_with(line, "FREE_SNAPSHOT")  ||
                                starts_with(line, "LIST_SLOTS")     ||
                                starts_with(line, "compress ")      ||
                                starts_with(line, "park")           ||
                                starts_with(line, "unpark")         ||
                                line == "free drafter"              ||
                                line == "drafter free");
            if (unsupported) {
                std::fprintf(stderr,
                    "[daemon] command not supported in gemma4 daemon: %s\n",
                    line.c_str());
                std::fflush(stderr);
                stream_emit(-1);
                continue;
            }

            // ── Parse: <prompt_bin_path> <n_gen> ──────────────────────────
            char ppath[1024] = {0};
            int  n_gen = 0;
            if (std::sscanf(line.c_str(), "%1023s %d", ppath, &n_gen) != 2 || n_gen <= 0) {
                std::fprintf(stderr, "[daemon] bad command line: %s\n", line.c_str());
                std::fflush(stderr);
                stream_emit(-1);
                continue;
            }

            // Read binary prompt file (int32 LE token IDs).
            std::vector<int32_t> prompt_ids = read_int32_file(ppath);
            if (prompt_ids.empty()) {
                std::fprintf(stderr, "[daemon] empty or unreadable prompt file: %s\n", ppath);
                std::fflush(stderr);
                stream_emit(-1);
                continue;
            }
            std::printf("[daemon] prompt=%zu tokens n_gen=%d\n",
                        prompt_ids.size(), n_gen);
            std::fflush(stdout);

            // Reset KV cache between requests.
            if (!daemon_first_iter) {
                step_graph_free(sg);
                reset_gemma4_cache(cache);  // also resets draft_kv_pos
                if (have_draft) {
                    draft_step_free(dsg);
                }
            }
            daemon_first_iter = false;

            if ((int)prompt_ids.size() + n_gen > ctx_size) {
                std::fprintf(stderr,
                    "[daemon] prompt (%zu) + n_gen (%d) > ctx_size (%d)\n",
                    prompt_ids.size(), n_gen, ctx_size);
                std::fflush(stderr);
                stream_emit(-1);
                continue;
            }

            // ── Prefill ───────────────────────────────────────────────────
            int last_logit_tok = -1;
            {
                const int n_prompt   = (int)prompt_ids.size();
                const int swa_window = w.swa_window > 0 ? w.swa_window : 1024;
                const int chunk_size = std::min(n_prompt, swa_window);

                for (int cs = 0; cs < n_prompt; cs += chunk_size) {
                    const int chunk_n   = std::min(chunk_size, n_prompt - cs);
                    const bool is_last  = (cs + chunk_n == n_prompt);
                    const bool need_mask = (cs + chunk_n > 1);

                    if (!build_gemma4_step(sg, w, cache, backend,
                                           cs, chunk_n, need_mask,
                                           /*capture=*/true,
                                           use_pflash, pflash_alpha,
                                           /*last_token_logits_only=*/true)) {
                        std::fprintf(stderr, "[daemon] prefill build failed at %d\n", cs);
                        std::fflush(stderr);
                        break;
                    }

                    if (!embed_tokens_batch(w, prompt_ids.data() + cs, chunk_n,
                                            sg.inp_embed, backend)) {
                        std::fprintf(stderr, "[daemon] embed_tokens_batch failed\n");
                        std::fflush(stderr);
                        break;
                    }

                    {
                        std::vector<int32_t> pos(chunk_n);
                        for (int i = 0; i < chunk_n; i++) pos[i] = cs + i;
                        ggml_backend_tensor_set(sg.positions, pos.data(), 0,
                                                sizeof(int32_t) * chunk_n);
                    }

                    if (sg.attn_mask && sg.attn_mask->buffer) {
                        const int kv_len = cs + chunk_n;
                        std::vector<uint16_t> mask_buf;
                        build_causal_mask(mask_buf, kv_len, chunk_n, cs);
                        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                                sizeof(uint16_t) * mask_buf.size());
                    }

                    if (sg.swa_mask && sg.swa_mask->buffer) {
                        const SwaView swa_view = compute_swa_view(cs, chunk_n,
                                                                    swa_window, cache.swa_ctx_alloc);
                        std::vector<uint16_t> swa_buf;
                        build_swa_causal_mask(swa_buf,
                                              /*kv_start*/ cs,
                                              /*n_tokens*/ chunk_n,
                                              /*swa_window*/ swa_window,
                                              /*ring_size*/ swa_view.effective_win_len,
                                              /*kv_end*/ cs + chunk_n);
                        ggml_backend_tensor_set(sg.swa_mask, swa_buf.data(), 0,
                                                sizeof(uint16_t) * swa_buf.size());
                    }

                    auto st = ggml_backend_graph_compute(backend, sg.gf);
                    if (st != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "[daemon] prefill compute failed at %d\n", cs);
                        std::fflush(stderr);
                        break;
                    }

                    // ── TQ3_0 K-cache write probe ─────────────────────────────────────
                    if (getenv("DFLASH_TQ3_PROBE_CACHE_WRITE") &&
                        (cs == 0 || cs == chunk_size) &&
                        !cache.attn_k.empty()) {
                        ggml_tensor * cache_k_layer0 = cache.attn_k[0];
                        if (cache_k_layer0 && cache_k_layer0->type == GGML_TYPE_TQ3_0) {
                            // nb[1] is the stride in bytes between successive token slots
                            const size_t off = (size_t)cache_k_layer0->nb[1] * (size_t)cs;
                            uint8_t blk[14] = {};
                            ggml_backend_tensor_get(cache_k_layer0, blk, off, 14);
                            std::fprintf(stderr, "[CACHE-WRITE-PROBE] cs=%d off=%zu bytes=", cs, off);
                            for (int _i = 0; _i < 14; _i++)
                                std::fprintf(stderr, "%02x ", blk[_i]);
                            std::fprintf(stderr, "\n");
                            std::fflush(stderr);
                        }
                    }
                    // ─────────────────────────────────────────────────────────────────

                    cache.cur_pos = cs + chunk_n;

                    if (is_last) {
                        const int vocab = w.n_vocab;
                        std::vector<float> logits_cpu(vocab);
                        ggml_backend_tensor_get(sg.logits, logits_cpu.data(),
                                                0, sizeof(float) * vocab);
                        last_logit_tok = sample_logits(logits_cpu.data(), vocab,
                                                       req_sampler, prompt_ids, rng);
                        cache.last_tok = last_logit_tok;
                    }

                    step_graph_free(sg);
                }

                // Draft KV prefill after target prefill.
                if (have_draft && last_logit_tok >= 0) {
                    const int target_feat_w  = dw.n_target_layers * dw.target_hidden;
                    const int draft_kv_cap   = cache.draft_kv_cap > 0
                                                   ? cache.draft_kv_cap
                                                   : (int)cache.draft_k[0]->ne[2];
                    const int draft_prefill_n    = std::min(n_prompt, draft_kv_cap);
                    const int draft_prefill_skip = n_prompt - draft_prefill_n;

                    DraftKVPrefillGraph pkg;
                    if (build_draft_kv_prefill(pkg, dw, cache, backend, draft_prefill_n)) {
                        // Ring-buffer aware bf16→f32 conversion (same as non-daemon path).
                        const int    cap      = cache.target_feat_cap;
                        const size_t feat_elt = ggml_element_size(cache.target_feat);
                        const int    slot0    = draft_prefill_skip % cap;
                        const int    pre_n    = std::min(draft_prefill_n, cap - slot0);
                        const int    post_n   = draft_prefill_n - pre_n;

                        dflash27b_launch_bf16_to_f32(
                            (const char *)cache.target_feat->data + (size_t)slot0 * feat_elt * target_feat_w,
                            (float *)pkg.target_feat->data,
                            (size_t)pre_n * target_feat_w, nullptr);
                        if (post_n > 0) {
                            dflash27b_launch_bf16_to_f32(
                                (const char *)cache.target_feat->data,
                                (float *)pkg.target_feat->data + (size_t)pre_n * target_feat_w,
                                (size_t)post_n * target_feat_w, nullptr);
                        }
                        cudaDeviceSynchronize();

                        std::vector<int32_t> pos(draft_prefill_n);
                        for (int pi = 0; pi < draft_prefill_n; pi++) pos[pi] = draft_prefill_skip + pi;
                        ggml_backend_tensor_set(pkg.positions, pos.data(), 0,
                                                sizeof(int32_t) * draft_prefill_n);

                        auto dst = ggml_backend_graph_compute(backend, pkg.gf);
                        if (dst != GGML_STATUS_SUCCESS) {
                            std::fprintf(stderr, "[daemon] draft KV prefill compute failed\n");
                            std::fflush(stderr);
                        }
                        cache.draft_kv_pos = draft_prefill_n % draft_kv_cap;
                    }
                    draft_kv_prefill_destroy(pkg);
                }
            }

            if (last_logit_tok < 0) {
                std::fprintf(stderr, "[daemon] prefill produced no logit token\n");
                std::fflush(stderr);
                stream_emit(-1);
                continue;
            }

            // ── Decode loop ───────────────────────────────────────────────
            std::vector<int32_t> history(prompt_ids);
            int committed = cache.cur_pos;
            int32_t cur_tok = last_logit_tok;
            int n_generated = 0;

            while (n_generated < n_gen) {
                if (IS_EOS_TOK(cur_tok, w)) {
                    std::printf("[daemon] EOS at step %d\n", n_generated);
                    std::fflush(stdout);
                    break;
                }
                if (committed >= ctx_size - 1) {
                    std::printf("[daemon] context full\n");
                    std::fflush(stdout);
                    break;
                }

                if (!build_gemma4_step(sg, w, cache, backend,
                                       committed, 1,
                                       /*with_mask=*/true,
                                       /*capture=*/false)) {
                    std::fprintf(stderr, "[daemon] decode build failed at step %d\n", n_generated);
                    std::fflush(stderr);
                    break;
                }

                if (sg.attn_mask && sg.attn_mask->buffer) {
                    const int kv_len = committed + 1;
                    std::vector<uint16_t> mask_buf;
                    build_causal_mask(mask_buf, kv_len, 1, committed);
                    ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                            sizeof(uint16_t) * mask_buf.size());
                }

                if (!embed_token(w, cur_tok, sg.inp_embed, backend)) {
                    std::fprintf(stderr, "[daemon] embed_token failed\n");
                    std::fflush(stderr);
                    break;
                }

                int32_t pos_val = committed;
                ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));

                auto st = ggml_backend_graph_compute(backend, sg.gf);
                if (st != GGML_STATUS_SUCCESS) {
                    std::fprintf(stderr, "[daemon] decode compute failed at step %d\n", n_generated);
                    std::fflush(stderr);
                    break;
                }

                committed++;
                cache.cur_pos = committed;

                const int vocab = w.n_vocab;
                std::vector<float> logits_cpu(vocab);
                ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                        sizeof(float) * vocab);

                const int32_t next_tok = (int32_t)sample_logits(
                    logits_cpu.data(), vocab, req_sampler, history, rng);

                // Emit current token to stream fd before advancing.
                stream_emit(cur_tok);

                history.push_back(cur_tok);
                n_generated++;

                cur_tok = next_tok;
                cache.last_tok = cur_tok;

                step_graph_free(sg);
            }

            // Sentinel: end of stream.
            stream_emit(-1);
            std::printf("[daemon] generated %d tokens\n", n_generated);
            std::fflush(stdout);
        }

        // ── Daemon exit: clean up ─────────────────────────────────────────
        step_graph_destroy(sg);
        draft_step_destroy(dsg);
        if (have_draft) {
            free_draft_kv_cache(cache);
            dw.tok_embd = nullptr;
            free_gemma4_draft_weights(dw);
            if (tok_embd_buf) ggml_backend_buffer_free(tok_embd_buf);
            if (tok_embd_ctx) ggml_free(tok_embd_ctx);
        }
        free_gemma4_cache(cache);
        free_gemma4_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    // ── Non-daemon: tokenize prompt ───────────────────────────────────────
    std::vector<int32_t> prompt_ids;
    if (!token_ids_str.empty()) {
        prompt_ids = parse_token_ids(token_ids_str);
        if (prompt_ids.empty()) {
            std::fprintf(stderr, "error: --tokens produced no valid token IDs\n");
            return 2;
        }
        std::printf("[tokens] using %zu pre-tokenised IDs from --tokens\n",
                    prompt_ids.size());
    } else {
        prompt_ids = tokenize_byte_fallback(prompt_text);
        std::printf("[tokens] byte-fallback tokenisation: %zu tokens "
                    "(pass --tokens <ids> for real tokenisation)\n",
                    prompt_ids.size());
    }

    // ── Ensure BOS is prepended (Gemma4 requires BOS at position 0) ──
    if (w.bos_id >= 0 && (prompt_ids.empty() || prompt_ids[0] != w.bos_id)) {
        prompt_ids.insert(prompt_ids.begin(), w.bos_id);
        std::printf("[tokens] prepended BOS token %d\n", w.bos_id);
    }

    if ((int)prompt_ids.size() >= ctx_size) {
        std::fprintf(stderr, "error: prompt (%zu tokens) >= ctx_size (%d)\n",
                     prompt_ids.size(), ctx_size);
        return 2;
    }

    // ── Benchmark loop outer container ────────────────────────────────────
    const int bench_runs = bench_mode ? 3 : 1;
    std::vector<double> bench_tok_per_sec;

    // Declared here (main scope) so step_graph_destroy(sg)/draft_step_destroy(dsg)
    // in cleanup is valid.
    StepGraph      sg;
    DraftStepGraph dsg;

    // Speculative decode stats (accumulated across bench iterations when bench_mode)
    int total_draft_steps = 0;
    int total_accepted    = 0;

    for (int bench_iter = 0; bench_iter < bench_runs; bench_iter++) {

        if (bench_runs > 1) {
            reset_gemma4_cache(cache);
            // Reset draft step state for the new bench iteration
            draft_step_free(dsg);
            total_draft_steps = 0;
            total_accepted    = 0;
            std::printf("[bench] run %d/%d\n", bench_iter + 1, bench_runs);
        }

        // ── Prefill ───────────────────────────────────────────────────────
        //
        // Chunked batched prefill: process up to swa_window tokens per chunk.
        // Each chunk dispatches a single GPU graph covering all tokens in the
        // chunk, which is far cheaper than one dispatch per token.
        //
        // For a chunk [cs, cs+chunk_n):
        //   1. Embed chunk tokens → inp_embed
        //   2. Set positions[i] = cs + i
        //   3. Build causal mask covering [0, cs+chunk_n) for the chunk rows
        //   4. Build SWA mask for sliding-window layers (when cs > 0)
        //   5. Compute graph → KV + target_feat (logits discarded except last)

        std::printf("[prefill] %zu tokens ...\n", prompt_ids.size());
        double prefill_t0 = now_ms();
        int last_logit_tok = -1;

        {
            const int n_prompt   = (int)prompt_ids.size();

            {
                const int swa_window = w.swa_window > 0 ? w.swa_window : 1024;
                const int chunk_size = std::min(n_prompt, swa_window);

                for (int cs = 0; cs < n_prompt; cs += chunk_size) {
                    const int chunk_n   = std::min(chunk_size, n_prompt - cs);
                    const bool is_last  = (cs + chunk_n == n_prompt);
                    const bool need_mask = (cs + chunk_n > 1);

                    if (!build_gemma4_step(sg, w, cache, backend,
                                           /*kv_start=*/cs, chunk_n,
                                           need_mask, /*capture=*/true,
                                           use_pflash, pflash_alpha,
                                           /*last_token_logits_only=*/true)) {
                        std::fprintf(stderr, "prefill chunk build failed at offset %d\n", cs);
                        return 1;
                    }

                    if (!embed_tokens_batch(w, prompt_ids.data() + cs, chunk_n,
                                            sg.inp_embed, backend)) {
                        return 1;
                    }

                    {
                        std::vector<int32_t> pos(chunk_n);
                        for (int i = 0; i < chunk_n; i++) pos[i] = cs + i;
                        ggml_backend_tensor_set(sg.positions, pos.data(), 0,
                                                sizeof(int32_t) * chunk_n);
                    }

                    if (sg.attn_mask && sg.attn_mask->buffer) {
                        const int kv_len = cs + chunk_n;
                        std::vector<uint16_t> mask_buf;
                        build_causal_mask(mask_buf, kv_len, chunk_n, cs);
                        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                                sizeof(uint16_t) * mask_buf.size());
                    }

                    if (sg.swa_mask && sg.swa_mask->buffer) {
                        const SwaView swa_view = compute_swa_view(cs, chunk_n,
                                                                    swa_window, cache.swa_ctx_alloc);
                        std::vector<uint16_t> swa_buf;
                        build_swa_causal_mask(swa_buf,
                                              /*kv_start*/ cs,
                                              /*n_tokens*/ chunk_n,
                                              /*swa_window*/ swa_window,
                                              /*ring_size*/ swa_view.effective_win_len,
                                              /*kv_end*/ cs + chunk_n);
                        ggml_backend_tensor_set(sg.swa_mask, swa_buf.data(), 0,
                                                sizeof(uint16_t) * swa_buf.size());
                    }

                    auto st = ggml_backend_graph_compute(backend, sg.gf);
                    if (st != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "prefill compute failed at chunk offset %d\n", cs);
                        return 1;
                    }

                    cache.cur_pos = cs + chunk_n;

                    if (is_last) {
                        const int vocab = w.n_vocab;
                        std::vector<float> logits_cpu(vocab);
                        // last_token_logits_only=true → logits has shape [vocab, 1];
                        // read from offset 0 instead of skipping (chunk_n-1)*vocab floats.
                        ggml_backend_tensor_get(sg.logits, logits_cpu.data(),
                                                0,
                                                sizeof(float) * vocab);
                        last_logit_tok = sample_logits(logits_cpu.data(), vocab,
                                                       sampler, prompt_ids, rng);
                        cache.last_tok = last_logit_tok;
                    }

                    step_graph_free(sg);
                }
            }
        }

        double prefill_t1 = now_ms();
        {
            const int    n_prompt    = (int)prompt_ids.size();
            const double prefill_ms  = prefill_t1 - prefill_t0;
            {
                const int swa_window = w.swa_window > 0 ? w.swa_window : 1024;
                const int chunk_size = std::min(n_prompt, swa_window);
                std::printf("[prefill] %d tokens in %.1f ms (%.1f tok/s) "
                            "[chunked%s, chunk_size=%d]  (last sampled token: %d)\n",
                            n_prompt, prefill_ms,
                            prefill_ms > 0.0 ? (double)n_prompt / (prefill_ms / 1000.0) : 0.0,
                            use_pflash ? "+pflash" : "", chunk_size, last_logit_tok);
            }
        }

        // ── Draft KV prefill: materialize draft KV for all prompt positions ─
        if (have_draft) {
            const int n_prompt = (int)prompt_ids.size();
            const int target_feat_w = dw.n_target_layers * dw.target_hidden;

            // Clamp to draft KV cache capacity. When the prompt is longer than the
            // draft cache, we prefill only the LAST draft_prefill_n tokens so that
            // the context that matters most (closest to the first decode step) is
            // represented in the draft KV cache.
            const int draft_kv_cap      = cache.draft_kv_cap > 0
                                              ? cache.draft_kv_cap
                                              : (int)cache.draft_k[0]->ne[2];
            const int draft_prefill_n    = std::min(n_prompt, draft_kv_cap);
            const int draft_prefill_skip = n_prompt - draft_prefill_n;

            DraftKVPrefillGraph pkg;
            if (!build_draft_kv_prefill(pkg, dw, cache, backend, draft_prefill_n)) {
                std::fprintf(stderr, "[draft] KV prefill build failed\n");
                return 1;
            }

            // Extract target_feat from ring buffer (bf16 → f32) directly into GPU tensor.
            // The ring buffer stores tokens at slot (pos % cap).
            // We want the LAST draft_prefill_n hidden states (positions draft_prefill_skip
            // through n_prompt-1). Their slots in the ring buffer start at
            // draft_prefill_skip % target_feat_cap and wrap as normal.
            {
                const int    cap      = cache.target_feat_cap;
                const size_t feat_elt = ggml_element_size(cache.target_feat);
                const int    slot0    = draft_prefill_skip % cap;
                const int    pre_n    = std::min(draft_prefill_n, cap - slot0);
                const int    post_n   = draft_prefill_n - pre_n;

                dflash27b_launch_bf16_to_f32(
                    (const char *)cache.target_feat->data + (size_t)slot0 * feat_elt * target_feat_w,
                    (float *)pkg.target_feat->data,
                    (size_t)pre_n * target_feat_w, nullptr);
                if (post_n > 0) {
                    dflash27b_launch_bf16_to_f32(
                        (const char *)cache.target_feat->data,
                        (float *)pkg.target_feat->data + (size_t)pre_n * target_feat_w,
                        (size_t)post_n * target_feat_w, nullptr);
                }
                cudaDeviceSynchronize();
            }

            // Positions: [draft_prefill_skip, ..., n_prompt-1]
            {
                std::vector<int32_t> pos(draft_prefill_n);
                for (int i = 0; i < draft_prefill_n; i++) pos[i] = draft_prefill_skip + i;
                ggml_backend_tensor_set(pkg.positions, pos.data(), 0, sizeof(int32_t) * draft_prefill_n);
            }

            auto st = ggml_backend_graph_compute(backend, pkg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[draft] KV prefill compute failed\n");
                draft_kv_prefill_destroy(pkg);
                return 1;
            }
            // draft_kv_pos tracks entries written, bounded by draft_kv_cap.
            cache.draft_kv_pos = draft_prefill_n % draft_kv_cap;

            draft_kv_prefill_destroy(pkg);
            std::printf("[draft] KV prefill done: %d positions materialized "
                        "(skipped %d early tokens, cap=%d)\n",
                        draft_prefill_n, draft_prefill_skip, draft_kv_cap);
        }

        // ── Decode loop ───────────────────────────────────────────────────

        std::vector<int32_t> generated;
        generated.reserve(n_predict);
        std::vector<int32_t> history(prompt_ids);

        int committed = cache.cur_pos;
        int32_t cur_tok = last_logit_tok;

        double decode_t0 = now_ms();
        double first_token_ms = -1.0;

        if (have_draft) {
            // ── SPECULATIVE DECODE LOOP ───────────────────────────────────
            //
            // Each iteration proposes a block of q_len tokens via the draft
            // model, then verifies with a single batched target forward.
            // Accepted prefix tokens are committed; the loop advances by
            // accept_n tokens per target call instead of 1.
            //
            // Gemma4 is pure attention (no SSM/conv state), so rollback is
            // trivially: just don't advance committed past accepted tokens.
            // Stale KV at positions [committed+commit_n..committed+q_len-1]
            // will be overwritten by the next verify pass.

            const int q_len        = (draft_max > 0 && draft_max < dw.block_size)
                                         ? draft_max : dw.block_size;
            const int mask_tok     = dw.mask_token_id; // 4
            const int target_feat_w = dw.n_target_layers * dw.target_hidden;
            const int vocab         = w.n_vocab;
            const int dkv_cap       = cache.draft_kv_cap > 0
                                          ? cache.draft_kv_cap
                                          : (int)cache.draft_k[0]->ne[2];

            std::vector<int32_t> noise_ids(q_len);
            std::vector<float>   noise_embed_buf((size_t)dw.n_embd * q_len);
            std::vector<int32_t> draft_tok(q_len);
            std::vector<int32_t> target_tok(q_len);
            std::vector<float>   draft_logits_buf((size_t)vocab * q_len);
            std::vector<float>   verify_logits_buf((size_t)vocab * q_len);

            while ((int)generated.size() < n_predict) {

                if (IS_EOS_TOK(cur_tok, w)) {
                    std::printf("\n[decode] EOS token %d\n", cur_tok);
                    break;
                }
                if (committed >= ctx_size - q_len) {
                    std::printf("\n[decode] context full\n");
                    break;
                }

                // Not enough context for target_feat extraction yet:
                // fall back to single-token target-only decode.
                if (committed < q_len) {
                    if (!build_gemma4_step(sg, w, cache, backend,
                                           committed, /*n_tokens=*/1,
                                           /*with_mask=*/true,
                                           /*capture=*/true)) {
                        std::fprintf(stderr, "[decode] warmup build failed at step %zu\n",
                                     generated.size());
                        return 1;
                    }

                    if (sg.attn_mask && sg.attn_mask->buffer) {
                        const int kv_len = committed + 1;
                        std::vector<uint16_t> mask_buf;
                        build_causal_mask(mask_buf, kv_len, 1, committed);
                        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                                sizeof(uint16_t) * mask_buf.size());
                    }

                    if (!embed_token(w, cur_tok, sg.inp_embed, backend)) return 1;

                    int32_t pos_val = committed;
                    ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));

                    double step_t0 = now_ms();
                    auto st = ggml_backend_graph_compute(backend, sg.gf);
                    double step_t1 = now_ms();

                    if (st != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "[decode] warmup compute failed at step %zu\n",
                                     generated.size());
                        return 1;
                    }

                    committed++;
                    cache.cur_pos = committed;

                    // Draft KV prefill for this warmup token (position committed-1).
                    {
                        const int warmup_pos     = committed - 1;
                        const int target_feat_w_w = dw.n_target_layers * dw.target_hidden;
                        DraftKVPrefillGraph wpkg;
                        if (!build_draft_kv_prefill(wpkg, dw, cache, backend, 1)) {
                            std::fprintf(stderr, "[decode] warmup draft KV prefill build failed\n");
                            return 1;
                        }
                        {
                            const int    cap      = cache.target_feat_cap;
                            const size_t feat_elt = ggml_element_size(cache.target_feat);
                            const int    slot     = warmup_pos % cap;
                            dflash27b_launch_bf16_to_f32(
                                (const char *)cache.target_feat->data + (size_t)slot * feat_elt * target_feat_w_w,
                                (float *)wpkg.target_feat->data,
                                (size_t)target_feat_w_w, nullptr);
                            cudaDeviceSynchronize();
                        }
                        {
                            int32_t p = warmup_pos;
                            ggml_backend_tensor_set(wpkg.positions, &p, 0, sizeof(int32_t));
                        }
                        auto wst = ggml_backend_graph_compute(backend, wpkg.gf);
                        if (wst != GGML_STATUS_SUCCESS) {
                            std::fprintf(stderr, "[decode] warmup draft KV prefill compute failed\n");
                            draft_kv_prefill_destroy(wpkg);
                            return 1;
                        }
                        cache.draft_kv_pos = (cache.draft_kv_pos + 1) % dkv_cap;
                        draft_kv_prefill_destroy(wpkg);
                    }

                    const int vocab_inner = w.n_vocab;
                    std::vector<float> logits_cpu(vocab_inner);
                    ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                            sizeof(float) * vocab_inner);

                    const int32_t next_tok = (int32_t)sample_logits(
                        logits_cpu.data(), vocab_inner, sampler, history, rng);

                    generated.push_back(cur_tok);
                    history.push_back(cur_tok);

                    if (first_token_ms < 0.0) {
                        first_token_ms = step_t1 - step_t0;
                    }

                    std::printf("%d ", cur_tok);
                    std::fflush(stdout);

                    cur_tok = next_tok;
                    cache.last_tok = cur_tok;

                    step_graph_free(sg);
                    continue;
                }

                // ── 1. Build noise block: [cur_tok, MASK, MASK, ..., MASK]
                noise_ids[0] = cur_tok;
                for (int i = 1; i < q_len; i++) noise_ids[i] = mask_tok;
                if (!w.embedder.embed(noise_ids.data(), q_len, noise_embed_buf.data())) {
                    std::fprintf(stderr, "[spec] embed noise_ids failed\n");
                    return 1;
                }

                // ── 2. Build draft graph (KV-cached, no target_feat input)
                // The draft model operates in its own KV address space bounded by
                // draft_kv_cap. Use cache.draft_kv_pos (number of entries written into
                // the draft KV cache) as kv_start, NOT the absolute committed position.
                if (cache.draft_kv_pos + q_len > dkv_cap) {
                    // Sliding-window re-prefill: instead of wiping all draft KV context,
                    // keep the most recent (dkv_cap - q_len) committed tokens by
                    // re-projecting their target_feat into the beginning of the draft
                    // KV cache.  This preserves the drafter's context continuity across
                    // ring-buffer wrap points, which is the root cause of acceptance
                    // collapsing from ~10/16 at 32K to ~1/16 at 64K.
                    const int keep = dkv_cap - q_len;
                    if (keep > 0 && committed >= keep) {
                        // Absolute positions of the (keep) tokens we want to retain:
                        // [committed - keep, committed).
                        const int refill_start = committed - keep;

                        // Reset draft_kv_pos to 0 so build_draft_kv_prefill_graph writes
                        // to slot [0, keep) — the ASSERT inside the graph builder requires
                        // draft_kv_pos + n_tokens <= ne[2].
                        cache.draft_kv_pos = 0;

                        DraftKVPrefillGraph rpkg;
                        if (!build_draft_kv_prefill(rpkg, dw, cache, backend, keep)) {
                            std::fprintf(stderr, "[spec] draft KV re-prefill build failed\n");
                            return 1;
                        }

                        // Copy target_feat for [refill_start, refill_start+keep) from the
                        // ring buffer (bf16) into rpkg.target_feat (f32).
                        {
                            const int    cap      = cache.target_feat_cap;
                            const size_t feat_elt = ggml_element_size(cache.target_feat);
                            const int    slot0    = refill_start % cap;
                            const int    pre_n    = std::min(keep, cap - slot0);
                            const int    post_n   = keep - pre_n;

                            dflash27b_launch_bf16_to_f32(
                                (const char *)cache.target_feat->data + (size_t)slot0 * feat_elt * target_feat_w,
                                (float *)rpkg.target_feat->data,
                                (size_t)pre_n * target_feat_w, nullptr);
                            if (post_n > 0) {
                                dflash27b_launch_bf16_to_f32(
                                    (const char *)cache.target_feat->data,
                                    (float *)rpkg.target_feat->data + (size_t)pre_n * target_feat_w,
                                    (size_t)post_n * target_feat_w, nullptr);
                            }
                            cudaDeviceSynchronize();
                        }

                        // Absolute positions for RoPE — must match training.
                        {
                            std::vector<int32_t> rpos(keep);
                            for (int i = 0; i < keep; i++) rpos[i] = refill_start + i;
                            ggml_backend_tensor_set(rpkg.positions, rpos.data(), 0,
                                                    sizeof(int32_t) * keep);
                        }

                        auto rst = ggml_backend_graph_compute(backend, rpkg.gf);
                        if (rst != GGML_STATUS_SUCCESS) {
                            std::fprintf(stderr, "[spec] draft KV re-prefill compute failed\n");
                            draft_kv_prefill_destroy(rpkg);
                            return 1;
                        }
                        cache.draft_kv_pos = keep;
                        draft_kv_prefill_destroy(rpkg);

                        std::fprintf(stderr,
                            "[spec] draft KV sliding re-prefill: kept %d tokens "
                            "(positions %d..%d), dkv_cap=%d\n",
                            keep, refill_start, committed - 1, dkv_cap);
                    } else {
                        // Not enough committed history to re-prefill — hard reset.
                        // This only happens at the very beginning of decode (committed < keep).
                        cache.draft_kv_pos = 0;
                    }
                }
                if (!build_draft_step(dsg, dw, cache, backend, q_len, cache.draft_kv_pos)) {
                    std::fprintf(stderr, "[spec] draft build failed\n");
                    return 1;
                }

                // ── 3. Set draft inputs

                // draft_embed: noise embeddings [n_embd, q_len] f32
                ggml_backend_tensor_set(dsg.draft_embed, noise_embed_buf.data(), 0,
                                        sizeof(float) * noise_embed_buf.size());

                // positions: absolute [committed, committed+1, ..., committed+q_len-1]
                // (absolute positions are used for RoPE — they must match training)
                {
                    std::vector<int32_t> pos(q_len);
                    for (int i = 0; i < q_len; i++) pos[i] = committed + i;
                    ggml_backend_tensor_set(dsg.positions, pos.data(), 0, sizeof(int32_t) * q_len);
                }

                // Causal mask: block token i attends to all draft KV context
                // [0..draft_kv_pos-1] plus block tokens [0..i].
                // Use draft_kv_pos (draft KV address space), not committed.
                if (dsg.attn_mask && dsg.attn_mask->buffer) {
                    const int dkv_ctx = cache.draft_kv_pos;
                    const int kv_len  = dkv_ctx + q_len;
                    const int kv_pad  = align_up(kv_len, KQ_MASK_PAD);
                    const int q_pad   = align_up(q_len, KQ_MASK_PAD);
                    std::vector<uint16_t> mask((size_t)kv_pad * q_pad, F16_NEG_INF);
                    for (int q = 0; q < q_len; q++) {
                        const int max_k = dkv_ctx + q;
                        for (int k = 0; k <= max_k; k++) {
                            mask[(size_t)q * kv_pad + k] = F16_ZERO;
                        }
                    }
                    ggml_backend_tensor_set(dsg.attn_mask, mask.data(), 0,
                                            sizeof(uint16_t) * mask.size());
                }

                // ── 4. Draft compute
                {
                    auto st = ggml_backend_graph_compute(backend, dsg.gf);
                    if (st != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "[spec] draft compute failed: %d\n", (int)st);
                        return 1;
                    }
                }

                // ── 5. Read draft logits and argmax
                ggml_backend_tensor_get(dsg.logits, draft_logits_buf.data(), 0,
                                        sizeof(float) * draft_logits_buf.size());
                for (int i = 0; i < q_len; i++) {
                    draft_tok[i] = argmax_f32(draft_logits_buf.data() + (size_t)i * vocab, vocab);
                }
                draft_tok[0] = cur_tok;  // pin first token (it was cur_tok, not a prediction)

                // ── 6. Target verify: batched forward on draft_tok[0..q_len-1]
                if (!build_gemma4_step(sg, w, cache, backend,
                                       committed, q_len,
                                       /*with_mask=*/true, /*capture=*/true)) {
                    std::fprintf(stderr, "[spec] verify build failed\n");
                    return 1;
                }

                if (!embed_tokens_batch(w, draft_tok.data(), q_len, sg.inp_embed, backend)) {
                    return 1;
                }

                // Target positions: [committed, committed+1, ..., committed+q_len-1]
                {
                    std::vector<int32_t> pos(q_len);
                    for (int i = 0; i < q_len; i++) pos[i] = committed + i;
                    ggml_backend_tensor_set(sg.positions, pos.data(), 0, sizeof(int32_t) * q_len);
                }

                // Causal mask for target verify
                if (sg.attn_mask && sg.attn_mask->buffer) {
                    const int kv_len = committed + q_len;
                    std::vector<uint16_t> mask_buf;
                    build_causal_mask(mask_buf, kv_len, q_len, committed);
                    ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                            sizeof(uint16_t) * mask_buf.size());
                }

                // SWA mask for target verify (required when n_tokens > 1)
                if (sg.swa_mask && sg.swa_mask->buffer) {
                    const SwaView swa_view = compute_swa_view(committed, q_len,
                                                               w.swa_window, cache.swa_ctx_alloc);
                    std::vector<uint16_t> swa_buf;
                    build_swa_causal_mask(swa_buf,
                                          /*kv_start*/ committed,
                                          /*n_tokens*/ q_len,
                                          /*swa_window*/ w.swa_window,
                                          /*ring_size*/ swa_view.effective_win_len,
                                          /*kv_end*/ committed + q_len);
                    ggml_backend_tensor_set(sg.swa_mask, swa_buf.data(), 0,
                                            sizeof(uint16_t) * swa_buf.size());
                }

                {
                    auto st = ggml_backend_graph_compute(backend, sg.gf);
                    if (st != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "[spec] verify compute failed: %d\n", (int)st);
                        return 1;
                    }
                }

                // ── 7. Read target logits and argmax
                ggml_backend_tensor_get(sg.logits, verify_logits_buf.data(), 0,
                                        sizeof(float) * verify_logits_buf.size());
                for (int i = 0; i < q_len; i++) {
                    target_tok[i] = argmax_f32(verify_logits_buf.data() + (size_t)i * vocab, vocab);
                }

                // ── 8. Acceptance: longest prefix match
                //   draft_tok[0] = cur_tok (accepted unconditionally as the current token)
                //   target_tok[i] = target's prediction for position committed+i+1
                //   Check: draft_tok[i+1] == target_tok[i]  (draft proposed the right next token)
                int accept_n = 1;
                for (int i = 0; i < q_len - 1; i++) {
                    if (draft_tok[i + 1] == target_tok[i]) accept_n++;
                    else break;
                }
                int commit_n = accept_n;
                if (commit_n > n_predict - (int)generated.size()) {
                    commit_n = n_predict - (int)generated.size();
                }

                // ── 9. Commit accepted tokens
                bool hit_eos = false;
                for (int i = 0; i < commit_n; i++) {
                    generated.push_back(draft_tok[i]);
                    history.push_back(draft_tok[i]);
                    std::printf("%d ", draft_tok[i]);
                    std::fflush(stdout);
                    if (IS_EOS_TOK(draft_tok[i], w)) { hit_eos = true; break; }
                }

                // ── 10. Draft KV prefill for the committed positions, then advance state.
                //   The target verify pass (step 6) captured target_feat for positions
                //   [committed..committed+q_len-1]. We prefill draft KV for the accepted
                //   prefix [committed..committed+commit_n-1] before advancing committed.
                {
                    DraftKVPrefillGraph cpkg;
                    if (!build_draft_kv_prefill(cpkg, dw, cache, backend, commit_n)) {
                        std::fprintf(stderr, "[spec] draft KV prefill build failed\n");
                        return 1;
                    }

                    // Extract target_feat for positions [committed..committed+commit_n-1]
                    // from the ring buffer (bf16 → f32).
                    {
                        const int    cap      = cache.target_feat_cap;
                        const size_t feat_elt = ggml_element_size(cache.target_feat);
                        const int    slot0    = committed % cap;
                        const int    pre_n    = std::min(commit_n, cap - slot0);
                        const int    post_n   = commit_n - pre_n;

                        dflash27b_launch_bf16_to_f32(
                            (const char *)cache.target_feat->data + (size_t)slot0 * feat_elt * target_feat_w,
                            (float *)cpkg.target_feat->data,
                            (size_t)pre_n * target_feat_w, nullptr);
                        if (post_n > 0) {
                            dflash27b_launch_bf16_to_f32(
                                (const char *)cache.target_feat->data,
                                (float *)cpkg.target_feat->data + (size_t)pre_n * target_feat_w,
                                (size_t)post_n * target_feat_w, nullptr);
                        }
                        cudaDeviceSynchronize();
                    }

                    {
                        std::vector<int32_t> pos(commit_n);
                        for (int i = 0; i < commit_n; i++) pos[i] = committed + i;
                        ggml_backend_tensor_set(cpkg.positions, pos.data(), 0,
                                                sizeof(int32_t) * commit_n);
                    }

                    auto cst = ggml_backend_graph_compute(backend, cpkg.gf);
                    if (cst != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "[spec] draft KV prefill compute failed\n");
                        draft_kv_prefill_destroy(cpkg);
                        return 1;
                    }
                    cache.draft_kv_pos = (cache.draft_kv_pos + commit_n) % dkv_cap;
                    draft_kv_prefill_destroy(cpkg);
                }

                //   Gemma4 is pure attention — no SSM/conv rollback needed.
                //   Stale KV at positions [committed+commit_n..committed+q_len-1]
                //   will be overwritten by the next verify pass.
                committed += commit_n;
                cache.cur_pos = committed;
                cur_tok = target_tok[commit_n - 1];
                cache.last_tok = cur_tok;

                total_draft_steps++;
                total_accepted += commit_n;

                if (first_token_ms < 0.0) {
                    first_token_ms = now_ms() - decode_t0;
                }

                double avg_accept = (total_draft_steps > 0)
                    ? (double)total_accepted / total_draft_steps : 0.0;
                std::printf("[step %d] accept=%d/%d avg=%.1f\n",
                            total_draft_steps, accept_n, q_len, avg_accept);

                if (hit_eos) break;

                step_graph_free(sg);
                draft_step_free(dsg);
            }

        } else if (have_mtp) {
            // ── MTP SPECULATIVE DECODE LOOP (γ=1 v1) ─────────────────────
            //
            // Each iteration:
            //   1. Run target forward for cur_tok at position `committed`,
            //      capturing mtp_h_prev from the last full-attention layer.
            //   2. Rebuild MTP step graph with current attn_pos = committed+1.
            //   3. Feed (cur_tok, mtp_h_prev) into MTP graph → draft_tok.
            //   4. Run target verify forward for draft_tok at position committed+1.
            //   5. Accept draft_tok if target agrees; otherwise accept target's
            //      token instead (standard single-draft acceptance).
            //   γ=1: one MTP draft per step. Correctness gate before γ>1.

            int mtp_steps    = 0;
            int mtp_accepted = 0;

            while ((int)generated.size() < n_predict) {

                if (IS_EOS_TOK(cur_tok, w)) {
                    std::printf("\n[mtp] EOS token %d at step %zu\n",
                                cur_tok, generated.size());
                    break;
                }
                if (committed >= ctx_size - 2) {
                    std::printf("\n[mtp] context full at step %zu\n",
                                generated.size());
                    break;
                }

                // ── 1. Target forward for cur_tok (captures mtp_h_prev) ──
                if (!build_gemma4_step(sg, w, cache, backend,
                                       committed, /*n_tokens=*/1,
                                       /*with_mask=*/true,
                                       /*capture=*/false)) {
                    std::fprintf(stderr, "[mtp] target build failed at step %zu\n",
                                 generated.size());
                    return 1;
                }

                if (sg.attn_mask && sg.attn_mask->buffer) {
                    const int kv_len = committed + 1;
                    std::vector<uint16_t> mask_buf;
                    build_causal_mask(mask_buf, kv_len, 1, committed);
                    ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                            sizeof(uint16_t) * mask_buf.size());
                }
                if (!embed_token(w, cur_tok, sg.inp_embed, backend)) return 1;
                {
                    int32_t pos_val = committed;
                    ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));
                }
                {
                    auto st = ggml_backend_graph_compute(backend, sg.gf);
                    if (st != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "[mtp] target compute failed\n");
                        return 1;
                    }
                }
                committed++;
                cache.cur_pos = committed;

                // Read target logits to get target's own prediction at position committed-1
                const int vocab = w.n_vocab;
                std::vector<float> logits_cpu(vocab);
                ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                        sizeof(float) * vocab);
                const int32_t target_next = (int32_t)sample_logits(
                    logits_cpu.data(), vocab, sampler, history, rng);

                step_graph_free(sg);

                // ── 2. Rebuild MTP step graph with attn_pos = committed ──
                free_mtp_step_graph(mtp_g);
                if (!build_mtp_step_graph(mtp_w, cache, w, mtp_g, committed)) {
                    std::fprintf(stderr, "[mtp] build_mtp_step_graph failed: %s\n",
                                 dflash27b_last_error());
                    return 1;
                }

                // Allocate MTP graph (needs gallocr; build_mtp_step_graph creates
                // the ggml context but not the backend buffers)
                ggml_gallocr_t mtp_alloc = ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(backend));
                bool mtp_alloc_ok = ggml_gallocr_alloc_graph(mtp_alloc, mtp_g.gf);
                if (!mtp_alloc_ok) {
                    std::fprintf(stderr, "[mtp] gallocr_alloc_graph failed\n");
                    ggml_gallocr_free(mtp_alloc);
                    return 1;
                }

                // ── 3. Set MTP inputs and compute ────────────────────────
                // in_tok_embd: pre-dequantised F32 embedding of cur_tok.
                // embed_token dequantises via w.embedder.embed() on CPU, avoiding
                // ggml_get_rows on a Q4_K source (unsupported in CUDA get_rows).
                if (!embed_token(w, cur_tok, mtp_g.in_tok_embd, backend)) {
                    std::fprintf(stderr, "[mtp] embed_token failed for tok=%d\n", cur_tok);
                    ggml_gallocr_free(mtp_alloc);
                    return 1;
                }
                // in_h_prev: captured by target graph into cache.mtp_h_prev
                ggml_backend_tensor_copy(cache.mtp_h_prev, mtp_g.in_h_prev);
                // in_pos: position of the draft token (= committed, 0-based)
                {
                    int32_t p = committed;
                    ggml_backend_tensor_set(mtp_g.in_pos, &p, 0, sizeof(int32_t));
                }

                {
                    auto st = ggml_backend_graph_compute(backend, mtp_g.gf);
                    if (st != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "[mtp] MTP compute failed\n");
                        ggml_gallocr_free(mtp_alloc);
                        return 1;
                    }
                }

                // Read draft token from in-graph argmax
                int32_t draft_tok = -1;
                ggml_backend_tensor_get(mtp_g.out_argmax, &draft_tok, 0, sizeof(int32_t));

                ggml_gallocr_free(mtp_alloc);

                // Emit the current token (already committed by target step above)
                generated.push_back(cur_tok);
                history.push_back(cur_tok);
                std::printf("%d ", cur_tok);
                std::fflush(stdout);

                if (first_token_ms < 0.0) {
                    first_token_ms = now_ms() - decode_t0;
                }

                mtp_steps++;

                // ── 4+5. Check if draft matches target's greedy token ───
                if (draft_tok == target_next) {
                    // MTP was right: accept draft token as next cur_tok
                    mtp_accepted++;
                    cur_tok = draft_tok;
                } else {
                    // MTP was wrong: use target's token
                    cur_tok = target_next;
                }
                cache.last_tok = cur_tok;

                if ((int)generated.size() % 8 == 0) {
                    std::printf("[mtp-step %d] accept_rate=%.2f\n",
                                mtp_steps,
                                mtp_steps > 0 ? (float)mtp_accepted / mtp_steps : 0.0f);
                }

                if (IS_EOS_TOK(cur_tok, w)) {
                    std::printf("\n[mtp] EOS token %d\n", cur_tok);
                    break;
                }
            }

            if (mtp_steps > 0) {
                std::printf("\n[mtp] steps=%d accepted=%d accept_rate=%.2f\n",
                            mtp_steps, mtp_accepted,
                            (float)mtp_accepted / mtp_steps);
            }

        } else {
            // ── TARGET-ONLY DECODE LOOP ───────────────────────────────────
            //
            // Single-token autoregressive path.
            // Each iteration:
            //   1. Feed `cur_tok` through the target at position `committed`.
            //   2. Sample the next token from logits.
            //   3. Append to generated sequence.
            //   4. Stop if EOS or n_predict reached.

            while ((int)generated.size() < n_predict) {

                if (IS_EOS_TOK(cur_tok, w)) {
                    std::printf("\n[decode] EOS token %d at step %zu\n",
                                cur_tok, generated.size());
                    break;
                }

                if (committed >= ctx_size - 1) {
                    std::printf("\n[decode] context full at step %zu\n",
                                generated.size());
                    break;
                }

                // Build single-token decode graph
                if (!build_gemma4_step(sg, w, cache, backend,
                                       committed, /*n_tokens=*/1,
                                       /*with_mask=*/true,
                                       /*capture=*/false)) {
                    std::fprintf(stderr, "[decode] build failed at step %zu\n",
                                 generated.size());
                    return 1;
                }

                if (sg.attn_mask && sg.attn_mask->buffer) {
                    const int kv_len = committed + 1;
                    std::vector<uint16_t> mask_buf;
                    build_causal_mask(mask_buf, kv_len, 1, committed);
                    ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                            sizeof(uint16_t) * mask_buf.size());
                }

                if (!embed_token(w, cur_tok, sg.inp_embed, backend)) return 1;

                int32_t pos_val = committed;
                ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));

                double step_t0 = now_ms();
                auto st = ggml_backend_graph_compute(backend, sg.gf);
                double step_t1 = now_ms();

                if (st != GGML_STATUS_SUCCESS) {
                    std::fprintf(stderr, "[decode] compute failed at step %zu\n",
                                 generated.size());
                    return 1;
                }

                committed++;
                cache.cur_pos = committed;

                // Fetch logits and sample
                const int vocab = w.n_vocab;
                std::vector<float> logits_cpu(vocab);
                ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                        sizeof(float) * vocab);

                const int32_t next_tok = (int32_t)sample_logits(
                    logits_cpu.data(), vocab, sampler, history, rng);

                generated.push_back(cur_tok);
                history.push_back(cur_tok);

                if (first_token_ms < 0.0 && !generated.empty()) {
                    first_token_ms = step_t1 - step_t0;
                }

                // Print token id (a proper decoder would map id -> string here)
                std::printf("%d ", cur_tok);
                std::fflush(stdout);

                cur_tok = next_tok;
                cache.last_tok = cur_tok;

                step_graph_free(sg);
            }
        }

        double decode_t1 = now_ms();
        const double decode_ms = decode_t1 - decode_t0;
        const int    n_gen     = (int)generated.size();
        const double tps       = (decode_ms > 0.0 && n_gen > 0)
                                     ? n_gen / (decode_ms / 1000.0)
                                     : 0.0;

        bench_tok_per_sec.push_back(tps);

        std::printf("\n");
        std::printf("[stats] generated=%d  decode_ms=%.1f  tok/s=%.2f  "
                    "first_tok_ms=%.2f\n",
                    n_gen, decode_ms, tps, first_token_ms);
        std::printf("[stats] prefill=%zu tokens  context_used=%d/%d\n",
                    prompt_ids.size(), committed, ctx_size);

        if (have_draft && total_draft_steps > 0) {
            std::printf("[spec] draft_steps=%d total_accepted=%d avg_accept=%.2f\n",
                        total_draft_steps, total_accepted,
                        (double)total_accepted / total_draft_steps);
        }

        // ── Memory stats ──────────────────────────────────────────────────
        {
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            const double used_gb  = (total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0);
            const double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
            std::printf("[mem]  VRAM used=%.2f GB  total=%.2f GB\n",
                        used_gb, total_gb);
        }

    } // bench loop

    // ── Benchmark summary ─────────────────────────────────────────────────
    if (bench_mode && bench_tok_per_sec.size() > 1) {
        std::sort(bench_tok_per_sec.begin(), bench_tok_per_sec.end());
        const double median = bench_tok_per_sec[bench_tok_per_sec.size() / 2];
        const double best   = bench_tok_per_sec.back();
        std::printf("\n[bench] median=%.2f tok/s  best=%.2f tok/s  runs=%zu\n",
                    median, best, bench_tok_per_sec.size());
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    step_graph_destroy(sg);
    draft_step_destroy(dsg);
    if (have_draft) {
        free_draft_kv_cache(cache);
        dw.tok_embd = nullptr;  // prevent double-free (tok_embd lives in tok_embd_buf)
        free_gemma4_draft_weights(dw);
        if (tok_embd_buf) ggml_backend_buffer_free(tok_embd_buf);
        if (tok_embd_ctx) ggml_free(tok_embd_ctx);
    }
    if (have_mtp) {
        free_mtp_step_graph(mtp_g);
        free_gemma4_mtp_assistant(mtp_w);
        // mtp_h_prev lives in mtp_h_prev_buf/ctx (not base_ctx).
        // Null out the pointer in cache before free_gemma4_cache to avoid
        // dangling reference (cache struct is stack-allocated; the pointer
        // would otherwise reference freed memory).
        cache.mtp_h_prev         = nullptr;
        cache.mtp_h_prev_enabled = false;
        if (mtp_h_prev_buf) { ggml_backend_buffer_free(mtp_h_prev_buf); mtp_h_prev_buf = nullptr; }
        if (mtp_h_prev_ctx) { ggml_free(mtp_h_prev_ctx); mtp_h_prev_ctx = nullptr; }
    }
    free_gemma4_cache(cache);
    free_gemma4_target_weights(w);
    ggml_backend_free(backend);

    return 0;
}
