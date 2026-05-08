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
#include <string>
#include <unordered_set>
#include <vector>
#include <random>

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

static void build_swa_causal_mask(std::vector<uint16_t> & out,
                                   int kv_len, int n_tokens, int kv_start,
                                   int swa_window) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int lo = std::max(0, abs_q - swa_window + 1);
        for (int k = lo; k <= abs_q && k < kv_len; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
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

        if (!use_pflash) {
            sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.attn_mask, "attn_mask");
            ggml_set_input(sg.attn_mask);
        }

        if (n_tokens > 1) {
            // SWA mask needed for sliding-window attention layers in batched prefill
            sg.swa_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.swa_mask, "swa_mask");
            ggml_set_input(sg.swa_mask);
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

#define IS_EOS_TOK(tok, w) \
    (((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id) || \
     ((w).eos_id      >= 0 && (tok) == (w).eos_id))

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

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    // ── Parse CLI arguments ───────────────────────────────────────────────
    std::string  model_path;
    std::string  draft_path;
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
        else if (std::strcmp(argv[i], "--kv-k")      == 0) kv_k_str      = require_next("--kv-k");
        else if (std::strcmp(argv[i], "--kv-v")      == 0) kv_v_str      = require_next("--kv-v");
        else if (std::strcmp(argv[i], "--seed")      == 0) sampler.seed  = (uint64_t)std::atoll(require_next("--seed"));
        else if (std::strcmp(argv[i], "--temp")      == 0) sampler.temp  = (float)std::atof(require_next("--temp"));
        else if (std::strcmp(argv[i], "--top-k")     == 0) sampler.top_k = std::atoi(require_next("--top-k"));
        else if (std::strcmp(argv[i], "--top-p")     == 0) sampler.top_p = (float)std::atof(require_next("--top-p"));
        else if (std::strcmp(argv[i], "--budget")    == 0) ddtree_budget = std::atoi(require_next("--budget"));
        else if (std::strcmp(argv[i], "--gpu")       == 0) gpu           = std::atoi(require_next("--gpu"));
        else if (std::strcmp(argv[i], "--fa-window")    == 0) fa_window     = std::atoi(require_next("--fa-window"));
        else if (std::strcmp(argv[i], "--bench")        == 0) bench_mode    = true;
        else if (std::strcmp(argv[i], "--pflash")       == 0) use_pflash    = true;
        else if (std::strcmp(argv[i], "--pflash-alpha") == 0) pflash_alpha  = (float)std::atof(require_next("--pflash-alpha"));
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
    const bool have_draft = !draft_path.empty();

    // Draft state: declared in main scope so they persist across bench iterations
    // and are accessible in cleanup.
    GemmaDraftWeights    dw;
    ggml_context       * tok_embd_ctx = nullptr;
    ggml_backend_buffer_t tok_embd_buf = nullptr;

    if (have_draft) {
        double t0 = now_ms();
        // Auto-detect: if path ends with .gguf, use GGUF loader; else safetensors dir
        bool ok = false;
        const bool is_gguf = (draft_path.size() >= 5 &&
                              draft_path.compare(draft_path.size() - 5, 5, ".gguf") == 0);
        if (is_gguf) {
            ok = load_gemma4_draft_gguf(draft_path, backend, dw);
            if (!ok) std::fprintf(stderr, "load_gemma4_draft_gguf: %s\n", dflash27b_last_error());
        } else {
            ok = load_gemma4_draft_safetensors(draft_path, backend, dw);
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

    // ── Tokenize prompt ───────────────────────────────────────────────────
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

    // ── RNG ───────────────────────────────────────────────────────────────
    std::mt19937_64 rng(sampler.seed);

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

                    if (sg.attn_mask) {
                        const int kv_len = cs + chunk_n;
                        std::vector<uint16_t> mask_buf;
                        build_causal_mask(mask_buf, kv_len, chunk_n, cs);
                        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                                sizeof(uint16_t) * mask_buf.size());
                    }

                    if (sg.swa_mask) {
                        const int kv_len = cs + chunk_n;
                        std::vector<uint16_t> swa_buf;
                        build_swa_causal_mask(swa_buf, kv_len, chunk_n, cs, swa_window);
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

            const int q_len        = dw.block_size;   // 16
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

                    if (sg.attn_mask) {
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
                    cache.draft_kv_pos = 0;
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
                {
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
                if (sg.attn_mask) {
                    const int kv_len = committed + q_len;
                    std::vector<uint16_t> mask_buf;
                    build_causal_mask(mask_buf, kv_len, q_len, committed);
                    ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                            sizeof(uint16_t) * mask_buf.size());
                }

                // SWA mask for target verify (required when n_tokens > 1)
                if (sg.swa_mask) {
                    const int kv_len = committed + q_len;
                    std::vector<uint16_t> swa_buf;
                    build_swa_causal_mask(swa_buf, kv_len, q_len, committed,
                                          w.swa_window);
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

                if (sg.attn_mask) {
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
    free_gemma4_cache(cache);
    free_gemma4_target_weights(w);
    ggml_backend_free(backend);

    return 0;
}
