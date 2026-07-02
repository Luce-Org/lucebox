// Loads Poolside Laguna-XS.2 from a GGUF file on disk into a ggml context
// on the CUDA backend. Mirrors gguf_target_loader.cpp's qwen35 path but for
// the laguna arch (iSWA + sigmoid-routed MoE + per-head softplus attn gate +
// per-layer-varying head count + per-layer-type partial RoPE with YaRN).
//
// Tensor naming (matches gguf-py MODEL_ARCH.LAGUNA list, see
// rnd_experiments/gguf_converter_quantizer/patches/laguna/arch_patch_summary.md):
//
//   Top-level:
//     token_embd.weight              [n_embd, n_vocab]                Q4_K_M (kept on CPU)
//     output_norm.weight             [n_embd]                          F32
//     output.weight                  [n_embd, n_vocab]                 Q6_K
//
//   Per layer blk.<i> (shared, all layers):
//     attn_norm.weight               [n_embd]                          F32
//     ffn_norm.weight                [n_embd]                          F32
//     attn_q.weight                  [n_embd, n_head[il] * head_dim]   Q4_K
//     attn_k.weight                  [n_embd, n_head_kv * head_dim]    Q8_0
//     attn_v.weight                  [n_embd, n_head_kv * head_dim]    Q8_0
//     attn_output.weight             [n_head[il] * head_dim, n_embd]   Q5_K
//     attn_q_norm.weight             [head_dim]                        F32
//     attn_k_norm.weight             [head_dim]                        F32
//     attn_gate.weight               [n_embd, n_head[il]]              Q4_K   (per-head softplus gate)
//
//   Layer 0 (dense MLP only):
//     ffn_gate.weight                [n_embd, n_ff]                    Q4_K
//     ffn_up.weight                  [n_embd, n_ff]                    Q4_K
//     ffn_down.weight                [n_ff, n_embd]                    Q5_K
//
//   Layers 1..n-1 (sparse MoE only):
//     ffn_gate_inp.weight            [n_embd, n_expert]                F32  (sigmoid router)
//     exp_probs_b.bias               [n_expert]                        F32  (DeepSeek-style score-correction bias)
//     ffn_gate_exps.weight           [n_embd, n_ff_exp, n_expert]      Q4_K (stacked)
//     ffn_up_exps.weight             [n_embd, n_ff_exp, n_expert]      Q4_K
//     ffn_down_exps.weight           [n_ff_exp, n_embd, n_expert]      Q5_K (or Q4_K)
//     ffn_gate_shexp.weight          [n_embd, n_ff_shexp]              Q4_K (always-on shared expert)
//     ffn_up_shexp.weight            [n_embd, n_ff_shexp]              Q4_K
//     ffn_down_shexp.weight          [n_ff_shexp, n_embd]              Q5_K

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"
#include "common/gguf_mmap.h"
#include "common/gguf_bounds.h"

#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

// fwd-decl: defined below at file scope, used by should_load_laguna_tensor
static bool is_laguna_expert_tensor(const char * name);

namespace {

int32_t  get_i32_or(const gguf_context * g, const char * key, int32_t fallback) {
    int64_t id = gguf_find_key(g, key); return (id < 0) ? fallback : gguf_get_val_i32(g, id);
}
uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(g, key); return (id < 0) ? fallback : gguf_get_val_u32(g, id);
}
float    get_f32_or(const gguf_context * g, const char * key, float fallback) {
    int64_t id = gguf_find_key(g, key); return (id < 0) ? fallback : gguf_get_val_f32(g, id);
}
bool     get_bool_or(const gguf_context * g, const char * key, bool fallback) {
    int64_t id = gguf_find_key(g, key); return (id < 0) ? fallback : (bool)gguf_get_val_bool(g, id);
}

size_t align_up_size(size_t x, size_t a) {
    if (a == 0) return x;
    const size_t r = x % a;
    return r == 0 ? x : x + (a - r);
}

bool parse_block_tensor_name(const char * name, int & layer_id) {
    const char prefix[] = "blk.";
    const size_t prefix_len = sizeof(prefix) - 1;
    if (std::strncmp(name, prefix, prefix_len) != 0) return false;
    const char * p = name + prefix_len;
    if (*p < '0' || *p > '9') return false;
    char * end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (!end || *end != '.' || v < 0 || v > INT_MAX) return false;
    layer_id = (int)v;
    return true;
}

bool should_load_laguna_tensor(const char * name, const TargetLoadPlan & plan) {
    if (std::strcmp(name, "token_embd.weight") == 0) return false;
    if (std::strcmp(name, "output_norm.weight") == 0 ||
        std::strcmp(name, "output.weight") == 0) {
        return plan.load_output;
    }
    if (plan.skip_expert_tensors && is_laguna_expert_tensor(name)) return false;
    int layer_id = -1;
    if (parse_block_tensor_name(name, layer_id)) {
        return layer_id >= plan.layer_begin && layer_id < plan.layer_end;
    }
    return false;
}

struct LagunaTensorAlloc {
    ggml_tensor * tensor = nullptr;
    size_t file_offset = 0;
    size_t file_size = 0;
    size_t buffer_offset = 0;
};

} // namespace

bool load_target_gguf_laguna(const std::string & path,
                              ggml_backend_t       backend,
                              LagunaTargetWeights & out) {
    TargetLoadPlan plan;
    return load_target_gguf_laguna_partial(path, backend, plan, out);
}

bool load_target_gguf_laguna_partial(const std::string & path,
                                      ggml_backend_t backend,
                                      const TargetLoadPlan & plan_in,
                                      LagunaTargetWeights & out) {

    // ── 1. Parse metadata ────────────────────────────────────────────────
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) { set_last_error("gguf_init_from_file failed: " + path); return false; }

    // Validate arch.
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) { set_last_error("missing general.architecture"); gguf_free(gctx); return false; }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "laguna") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected laguna)");
            gguf_free(gctx);
            return false;
        }
    }

    // Read scalar hparams.
    const uint32_t n_layer       = get_u32_or(gctx, "laguna.block_count",                     0);
    const uint32_t n_embd        = get_u32_or(gctx, "laguna.embedding_length",                0);
    const uint32_t n_ff          = get_u32_or(gctx, "laguna.feed_forward_length",             0);
    const uint32_t n_ff_exp      = get_u32_or(gctx, "laguna.expert_feed_forward_length",      0);
    const uint32_t n_ff_shexp    = get_u32_or(gctx, "laguna.expert_shared_feed_forward_length",0);
    const uint32_t n_head_kv     = get_u32_or(gctx, "laguna.attention.head_count_kv",         0);
    const uint32_t key_length    = get_u32_or(gctx, "laguna.attention.key_length",            0);
    const uint32_t value_length  = get_u32_or(gctx, "laguna.attention.value_length",          0);
    const uint32_t n_expert      = get_u32_or(gctx, "laguna.expert_count",                    0);
    const uint32_t n_expert_used = get_u32_or(gctx, "laguna.expert_used_count",               0);
    const uint32_t n_dense_lead  = get_u32_or(gctx, "laguna.leading_dense_block_count",       1);
    const uint32_t sliding_win   = get_u32_or(gctx, "laguna.attention.sliding_window",        0);
    const uint32_t n_rot_full    = get_u32_or(gctx, "laguna.rope.dimension_count",            0);
    const uint32_t n_rot_swa     = get_u32_or(gctx, "laguna.rope.dimension_count_swa",        0);
    const uint32_t n_vocab       = get_u32_or(gctx, "laguna.vocab_size",                      0);

    const float  rope_base_full   = get_f32_or(gctx, "laguna.rope.freq_base",     0.0f);
    const float  rope_base_swa    = get_f32_or(gctx, "laguna.rope.freq_base_swa", 0.0f);
    const float  yarn_factor      = get_f32_or(gctx, "laguna.rope.scaling.factor",          0.0f);
    const float  yarn_attn_factor = get_f32_or(gctx, "laguna.rope.scaling.yarn_attn_factor", 1.0f);
    const float  yarn_beta_fast   = get_f32_or(gctx, "laguna.rope.scaling.yarn_beta_fast",  64.0f);
    const float  yarn_beta_slow   = get_f32_or(gctx, "laguna.rope.scaling.yarn_beta_slow",   1.0f);
    const uint32_t yarn_orig_ctx  = get_u32_or(gctx, "laguna.rope.scaling.original_context_length", 4096);

    const float  exp_w_scale      = get_f32_or(gctx, "laguna.expert_weights_scale", 1.0f);
    const bool   exp_w_norm       = get_bool_or(gctx, "laguna.expert_weights_norm", true);
    // expert_gating_func: 0=NONE, 1=SOFTMAX, 2=SIGMOID (LLAMA_EXPERT_GATING_FUNC_TYPE_*)
    const uint32_t exp_gate_fn    = get_u32_or(gctx, "laguna.expert_gating_func", 2);
    (void)yarn_attn_factor; // currently unused at load time; consumed by the graph builder

    if (n_layer == 0 || n_embd == 0 || n_head_kv == 0 || key_length == 0 || value_length == 0 ||
        n_ff == 0 || n_ff_exp == 0 || n_ff_shexp == 0 || n_expert == 0 || n_expert_used == 0 ||
        sliding_win == 0 || n_rot_full == 0 || n_rot_swa == 0 || n_vocab == 0) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "missing or zero hparams: n_layer=%u n_embd=%u n_head_kv=%u key=%u val=%u "
            "n_ff=%u n_ff_exp=%u n_ff_shexp=%u n_expert=%u used=%u sw=%u n_rot{full=%u swa=%u} vocab=%u",
            n_layer, n_embd, n_head_kv, key_length, value_length,
            n_ff, n_ff_exp, n_ff_shexp, n_expert, n_expert_used,
            sliding_win, n_rot_full, n_rot_swa, n_vocab);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }
    if (key_length != value_length) {
        set_last_error("laguna: key_length != value_length not supported");
        gguf_free(gctx); return false;
    }
    if (n_expert_used > n_expert) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "laguna: expert_used_count (%u) exceeds expert_count (%u)",
            n_expert_used, n_expert);
        set_last_error(buf);
        gguf_free(gctx); return false;
    }

    // Per-layer head count (ARRAY of length n_layer).
    std::vector<uint32_t> heads_per_layer((size_t)n_layer, 0);
    {
        int64_t aid = gguf_find_key(gctx, "laguna.attention.head_count");
        if (aid < 0) { set_last_error("missing laguna.attention.head_count"); gguf_free(gctx); return false; }
        const enum gguf_type kt = gguf_get_kv_type(gctx, aid);
        if (kt == GGUF_TYPE_ARRAY) {
            const size_t n = gguf_get_arr_n(gctx, aid);
            if (n != (size_t)n_layer) {
                char b[160];
                std::snprintf(b, sizeof(b),
                    "laguna.attention.head_count array len %zu != n_layer %u", n, n_layer);
                set_last_error(b);
                gguf_free(gctx); return false;
            }
            const enum gguf_type elt = gguf_get_arr_type(gctx, aid);
            if (elt != GGUF_TYPE_INT32 && elt != GGUF_TYPE_UINT32) {
                set_last_error("laguna.attention.head_count array element type must be i32 or u32");
                gguf_free(gctx); return false;
            }
            const void * p = gguf_get_arr_data(gctx, aid);
            for (uint32_t i = 0; i < n_layer; ++i) {
                heads_per_layer[i] = (elt == GGUF_TYPE_INT32)
                    ? (uint32_t)((const int32_t *)p)[i]
                    : ((const uint32_t *)p)[i];
            }
        } else {
            // Some GGUF writers may emit a scalar even when override exists; fall back.
            const uint32_t scalar = gguf_get_val_u32(gctx, aid);
            for (uint32_t i = 0; i < n_layer; ++i) heads_per_layer[i] = scalar;
        }
    }

    // Tokenizer special tokens.
    const uint32_t kEosKeyMissing = 0xFFFFFFFFu;
    const uint32_t raw_bos      = get_u32_or(gctx, "tokenizer.ggml.bos_token_id",     kEosKeyMissing);
    const uint32_t raw_eos      = get_u32_or(gctx, "tokenizer.ggml.eos_token_id",     kEosKeyMissing);
    const uint32_t raw_eot      = get_u32_or(gctx, "tokenizer.ggml.eot_token_id",     kEosKeyMissing);
    const uint32_t raw_pad      = get_u32_or(gctx, "tokenizer.ggml.padding_token_id", kEosKeyMissing);

    // Populate metadata
    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer            = (int)n_layer;
    out.n_embd             = (int)n_embd;
    out.n_ff               = (int)n_ff;
    out.n_ff_exp           = (int)n_ff_exp;
    out.n_ff_shexp         = (int)n_ff_shexp;
    out.n_head_kv          = (int)n_head_kv;
    out.head_dim           = (int)key_length;
    out.n_expert           = (int)n_expert;
    out.n_expert_used      = (int)n_expert_used;
    out.n_layer_dense_lead = (int)n_dense_lead;
    out.sliding_window     = (int)sliding_win;
    out.swa_pattern        = 4;  // (full, sw, sw, sw); fixed by Laguna design
    out.n_rot_full         = (int)n_rot_full;
    out.n_rot_swa          = (int)n_rot_swa;
    out.rope_freq_base_full= rope_base_full > 0 ? rope_base_full : 500000.0f;
    out.rope_freq_base_swa = rope_base_swa  > 0 ? rope_base_swa  :  10000.0f;
    out.yarn_factor        = yarn_factor    > 0 ? yarn_factor    :     32.0f;
    out.yarn_beta_fast     = yarn_beta_fast;
    out.yarn_beta_slow     = yarn_beta_slow;
    out.yarn_orig_ctx      = (int)yarn_orig_ctx;
    out.expert_weights_scale  = exp_w_scale > 0 ? exp_w_scale : 2.5f;
    out.expert_weights_norm   = exp_w_norm;
    out.expert_gating_sigmoid = (exp_gate_fn == 2);
    out.bos_id      = (raw_bos == kEosKeyMissing) ? -1 : (int32_t)raw_bos;
    out.eos_id      = (raw_eos == kEosKeyMissing) ? -1 : (int32_t)raw_eos;
    // Laguna GGUF only ships tokenizer.ggml.eos_token_id (id 2 =
    // 〈|EOS|〉); the chat-template end-of-turn marker is </assistant>
    // (id 24). Without this fallback the decoder check
    // `next == eos_chat_id` matches -1 (impossible) and the model
    // never stops mid-stream — it just emits </assistant> and keeps
    // going, then re-greets the user and answers again until
    // max_tokens. See chat_template.cpp ChatFormat::LAGUNA and
    // laguna_internal.h for the matching constant.
    out.eos_chat_id = (raw_eot == kEosKeyMissing) ? 24 : (int32_t)raw_eot;
    out.pad_id      = (raw_pad == kEosKeyMissing) ? -1 : (int32_t)raw_pad;
    if ((int)n_layer <= (int)(sizeof(out.n_head_arr)/sizeof(out.n_head_arr[0]))) {
        for (uint32_t i = 0; i < n_layer; ++i) out.n_head_arr[i] = (int)heads_per_layer[i];
    } else {
        set_last_error("laguna: n_layer exceeds compiled-in n_head_arr capacity (40)");
        gguf_free(gctx); return false;
    }

    // Diagnostic.
    std::printf("[laguna-loader] n_layer=%u n_embd=%u head_dim=%u n_head_kv=%u\n",
                n_layer, n_embd, key_length, n_head_kv);
    std::printf("[laguna-loader] dense_lead=%u sliding_window=%u (pattern fwd,swa,swa,swa)\n",
                n_dense_lead, sliding_win);
    std::printf("[laguna-loader] rope full=%g swa=%g  n_rot full=%u swa=%u  yarn factor=%g orig_ctx=%u\n",
                rope_base_full, rope_base_swa, n_rot_full, n_rot_swa, yarn_factor, yarn_orig_ctx);
    std::printf("[laguna-loader] MoE n_expert=%u used=%u  ff_exp=%u  ff_shexp=%u  scale=%g  sigmoid_router=%d\n",
                n_expert, n_expert_used, n_ff_exp, n_ff_shexp, exp_w_scale, (int)(exp_gate_fn == 2));
    std::printf("[laguna-loader] specials bos=%d eos=%d eot=%d pad=%d  vocab=%u\n",
                out.bos_id, out.eos_id, out.eos_chat_id, out.pad_id, n_vocab);

    out.layers.assign((size_t)n_layer, LagunaTargetLayer{});

    // ── 2. Resolve tensor pointers ───────────────────────────────────────
    auto g = [&](const char * name) -> ggml_tensor * { return ggml_get_tensor(meta_ctx, name); };
    out.tok_embd = g("token_embd.weight");
    out.out_norm = g("output_norm.weight");
    out.output   = g("output.weight");
    if (!out.tok_embd || !out.out_norm) {
        set_last_error("missing top-level tensors (token_embd / output_norm)");
        gguf_free(gctx); return false;
    }
    // output (lm_head) may be tied to token_embd in some quants; we always need the data
    // to be uploadable. Laguna's converter tells parent NOT to tie, so this must exist.
    if (!out.output) {
        set_last_error("missing output.weight (Laguna does not tie embeddings)");
        gguf_free(gctx); return false;
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        char name[160];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        LagunaTargetLayer & L = out.layers[il];

        // Always present:
        L.attn_norm = fnd("attn_norm.weight");
        L.ffn_norm  = fnd("ffn_norm.weight");
        L.wq        = fnd("attn_q.weight");
        L.wk        = fnd("attn_k.weight");
        L.wv        = fnd("attn_v.weight");
        L.wo        = fnd("attn_output.weight");
        L.q_norm    = fnd("attn_q_norm.weight");
        L.k_norm    = fnd("attn_k_norm.weight");
        L.wqkv_gate = fnd("attn_gate.weight");

        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.wqkv_gate) {
            char b[160];
            std::snprintf(b, sizeof(b), "layer %u missing required attention tensor(s)", il);
            set_last_error(b);
            gguf_free(gctx); return false;
        }

        const bool is_dense = (il < n_dense_lead);
        if (is_dense) {
            L.w_gate = fnd("ffn_gate.weight");
            L.w_up   = fnd("ffn_up.weight");
            L.w_down = fnd("ffn_down.weight");
            if (!L.w_gate || !L.w_up || !L.w_down) {
                char b[160];
                std::snprintf(b, sizeof(b), "dense layer %u missing ffn_gate/up/down", il);
                set_last_error(b);
                gguf_free(gctx); return false;
            }
        } else {
            // Sparse MoE
            L.ffn_gate_inp     = fnd("ffn_gate_inp.weight");
            L.ffn_exp_probs_b  = fnd("exp_probs_b.bias");
            L.ffn_gate_exps    = fnd("ffn_gate_exps.weight");
            L.ffn_up_exps      = fnd("ffn_up_exps.weight");
            L.ffn_down_exps    = fnd("ffn_down_exps.weight");
            L.ffn_gate_shexp   = fnd("ffn_gate_shexp.weight");
            L.ffn_up_shexp     = fnd("ffn_up_shexp.weight");
            L.ffn_down_shexp   = fnd("ffn_down_shexp.weight");
            if (!L.ffn_gate_inp || !L.ffn_exp_probs_b || !L.ffn_gate_exps ||
                !L.ffn_up_exps  || !L.ffn_down_exps  || !L.ffn_gate_shexp ||
                !L.ffn_up_shexp || !L.ffn_down_shexp) {
                char b[200];
                std::snprintf(b, sizeof(b),
                    "sparse layer %u missing MoE tensor(s) "
                    "(gate_inp=%p probs_b=%p gate_exps=%p up_exps=%p down_exps=%p gate_sh=%p up_sh=%p down_sh=%p)",
                    il,
                    (void*)L.ffn_gate_inp, (void*)L.ffn_exp_probs_b,
                    (void*)L.ffn_gate_exps, (void*)L.ffn_up_exps, (void*)L.ffn_down_exps,
                    (void*)L.ffn_gate_shexp, (void*)L.ffn_up_shexp, (void*)L.ffn_down_shexp);
                set_last_error(b);
                gguf_free(gctx); return false;
            }
        }
    }

    TargetLoadPlan plan = plan_in;
    if (plan.layer_begin < 0) plan.layer_begin = 0;
    if (plan.layer_end < 0) plan.layer_end = (int)n_layer;
    if (plan.layer_begin > plan.layer_end || plan.layer_end > (int)n_layer) {
        char e[160];
        std::snprintf(e, sizeof(e),
            "laguna: invalid layer range [%d,%d) for n_layer=%u",
            plan.layer_begin, plan.layer_end, n_layer);
        set_last_error(e);
        gguf_free(gctx);
        return false;
    }

    // ── 3. Allocate backend buffer only for selected tensors. Token embedding
    //       stays CPU-only and is owned by the CpuEmbedder mmap.
    GgufMmap mm;
    std::string err;
    if (!mm.open(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const uint8_t * mm_addr = (const uint8_t *)mm.data();
    const size_t    mm_len  = mm.size();
    const size_t data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors  = gguf_get_n_tensors(gctx);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    std::vector<LagunaTensorAlloc> allocs;
    size_t alloc_total = 0;
    for (int64_t tid = 0; tid < n_tensors; ++tid) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t || !should_load_laguna_tensor(tname, plan)) continue;
        alloc_total = align_up_size(alloc_total, alignment);
        LagunaTensorAlloc a;
        a.tensor = t;
        a.file_offset = data_start + gguf_get_tensor_offset(gctx, tid);
        a.file_size = gguf_get_tensor_size(gctx, tid);
        a.buffer_offset = alloc_total;
        alloc_total += ggml_backend_buft_get_alloc_size(buft, t);
        allocs.push_back(a);
    }

    // Lay attn_q|attn_k and ffn_gate_shexp|ffn_up_shexp adjacently in the
    // weight buffer so one fused tensor can view both regions (saves one
    // matmul launch + one activation quantization per pair per forward).
    static const bool fuse_qk_env = []() {
        const char * e = getenv("DFLASH_LAGUNA_FUSED_QK");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    if (fuse_qk_env) {
        // For each fusable pair, `first` must land before `second` in the
        // buffer regardless of GGUF order (attn_k typically precedes attn_q).
        auto rename_sub = [](const std::string & s, const char * from, const char * to) {
            std::string p = s;
            const size_t pos = p.find(from);
            if (pos == std::string::npos) return std::string();
            p.replace(pos, std::string(from).size(), to);
            return p;
        };
        std::vector<LagunaTensorAlloc> arranged;
        arranged.reserve(allocs.size());
        std::vector<char> used(allocs.size(), 0);
        auto idx_of = [&](const std::string & nm) -> int {
            if (nm.empty()) return -1;
            for (size_t j = 0; j < allocs.size(); ++j) {
                if (!used[j] && nm == ggml_get_name(allocs[j].tensor)) return (int)j;
            }
            return -1;
        };
        for (size_t i = 0; i < allocs.size(); ++i) {
            if (used[i]) continue;
            const std::string nm = ggml_get_name(allocs[i].tensor);
            std::string first, second;
            if (nm.find("attn_q.weight") != std::string::npos) {
                first = nm; second = rename_sub(nm, "attn_q.weight", "attn_k.weight");
            } else if (nm.find("attn_k.weight") != std::string::npos) {
                first = rename_sub(nm, "attn_k.weight", "attn_q.weight"); second = nm;
            } else if (nm.find("ffn_gate_shexp.weight") != std::string::npos) {
                first = nm; second = rename_sub(nm, "ffn_gate_shexp.weight", "ffn_up_shexp.weight");
            } else if (nm.find("ffn_up_shexp.weight") != std::string::npos) {
                first = rename_sub(nm, "ffn_up_shexp.weight", "ffn_gate_shexp.weight"); second = nm;
            } else {
                used[i] = 1;
                arranged.push_back(allocs[i]);
                continue;
            }
            const int fi = idx_of(first);
            const int si = idx_of(second);
            if (fi >= 0) { used[fi] = 1; arranged.push_back(allocs[fi]); }
            if (si >= 0) { used[si] = 1; arranged.push_back(allocs[si]); }
        }
        allocs.swap(arranged);
        alloc_total = 0;
        for (LagunaTensorAlloc & a : allocs) {
            alloc_total = align_up_size(alloc_total, alignment);
            a.buffer_offset = alloc_total;
            alloc_total += ggml_backend_buft_get_alloc_size(buft, a.tensor);
        }
    }

    if (allocs.empty()) {
        set_last_error("laguna: load plan selected no GPU tensors");
        gguf_free(gctx);
        return false;
    }

    out.buf = ggml_backend_alloc_buffer(backend, alloc_total);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_buffer failed (laguna target)");
        gguf_free(gctx); return false;
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    char * base = (char *)ggml_backend_buffer_get_base(out.buf);
    for (const LagunaTensorAlloc & a : allocs) {
        if (ggml_backend_tensor_alloc(out.buf, a.tensor,
                                      base + a.buffer_offset) != GGML_STATUS_SUCCESS) {
            set_last_error("ggml_backend_tensor_alloc failed (laguna target)");
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
            gguf_free(gctx);
            return false;
        }
    }

    // Bind fused-view tensors over adjacent pairs. The fused tensor shares
    // the pair's bytes (no copy); wq/wk stay valid for all other paths.
    if (fuse_qk_env) {
        ggml_init_params fip{};
        fip.mem_size = ggml_tensor_overhead() * (size_t)(2 * n_layer + 8);
        fip.no_alloc = true;
        out.fuse_ctx = ggml_init(fip);
    }
    int n_fused_qk = 0, n_fused_gu = 0;
    if (out.fuse_ctx) {
        auto adjacent = [&](ggml_tensor * a, ggml_tensor * b) {
            return a && b && a->type == b->type && a->ne[0] == b->ne[0] &&
                   a->data && b->data &&
                   (char *) b->data == (char *) a->data + ggml_nbytes(a) &&
                   ggml_backend_buft_get_alloc_size(buft, a) == ggml_nbytes(a);
        };
        for (uint32_t il = 0; il < n_layer; ++il) {
            LagunaTargetLayer & L = out.layers[il];
            if (adjacent(L.wq, L.wk)) {
                ggml_tensor * t = ggml_new_tensor_2d(out.fuse_ctx, L.wq->type,
                                                     L.wq->ne[0], L.wq->ne[1] + L.wk->ne[1]);
                ggml_format_name(t, "blk.%u.attn_qk_fused", il);
                if (ggml_backend_tensor_alloc(out.buf, t, L.wq->data) == GGML_STATUS_SUCCESS) {
                    L.wqk = t;
                    n_fused_qk++;
                }
            }
            if (adjacent(L.ffn_gate_shexp, L.ffn_up_shexp)) {
                ggml_tensor * t = ggml_new_tensor_2d(out.fuse_ctx, L.ffn_gate_shexp->type,
                                                     L.ffn_gate_shexp->ne[0],
                                                     L.ffn_gate_shexp->ne[1] + L.ffn_up_shexp->ne[1]);
                ggml_format_name(t, "blk.%u.ffn_shexp_gu_fused", il);
                if (ggml_backend_tensor_alloc(out.buf, t, L.ffn_gate_shexp->data) == GGML_STATUS_SUCCESS) {
                    L.shexp_gu = t;
                    n_fused_gu++;
                }
            }
        }
        std::printf("[laguna-loader] fused adjacent weights: qk=%d shexp_gu=%d layers\n",
                    n_fused_qk, n_fused_gu);
    }

    // ── 4. Copy selected tensor bytes to GPU; remember tok_embd for embedder ─
    size_t total = 0;
    size_t tok_embd_off = 0, tok_embd_sz = 0;
    ggml_type tok_embd_type = GGML_TYPE_COUNT;
    for (int64_t tid = 0; tid < n_tensors; ++tid) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t rel_off = gguf_get_tensor_offset(gctx, tid);
        const size_t off = data_start + rel_off;
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (!gguf_tensor_in_file(data_start, rel_off, sz, mm_len)) {
            set_last_error(gguf_bounds_error("laguna target GGUF", tname,
                ggml_type_name(gguf_get_tensor_type(gctx, tid)),
                data_start, rel_off, sz, mm_len));
            gguf_free(gctx); return false;
        }
        if (std::string(tname) == "token_embd.weight") {
            tok_embd_off  = off;
            tok_embd_sz   = sz;
            tok_embd_type = gguf_get_tensor_type(gctx, tid);
            continue;
        }
        if (!should_load_laguna_tensor(tname, plan)) continue;
        ggml_backend_tensor_set(t, mm_addr + off, 0, sz);
        total += sz;
    }

    // Fused per-head q/k norm weights: [head_dim, n_head+n_head_kv] f32 with
    // q_norm replicated over the query heads and k_norm over the kv heads.
    // Lets the fused-QK path run ONE rms_norm+mul+rope over all heads
    // (bit-identical: every op is per-row/per-head independent).
    if (n_fused_qk > 0) {
        ggml_init_params nip{};
        nip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer + 4);
        nip.no_alloc = true;
        out.fuse_norm_ctx = ggml_init(nip);
        const int hd  = out.head_dim;
        const int nkv = out.n_head_kv;
        if (out.fuse_norm_ctx) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                LagunaTargetLayer & L = out.layers[il];
                if (!L.wqk || !L.q_norm || !L.k_norm ||
                    L.q_norm->type != GGML_TYPE_F32 || L.k_norm->type != GGML_TYPE_F32 ||
                    L.q_norm->ne[0] != hd || L.k_norm->ne[0] != hd) {
                    L.wqk = nullptr;  // fused path needs the fused norm too
                    continue;
                }
                ggml_tensor * t = ggml_new_tensor_2d(out.fuse_norm_ctx, GGML_TYPE_F32,
                                                     hd, out.n_head_arr[il] + nkv);
                ggml_format_name(t, "blk.%u.qk_norm_fused", il);
                L.qk_norm_f = t;
            }
            out.fuse_norm_buf = ggml_backend_alloc_ctx_tensors(out.fuse_norm_ctx, backend);
            if (out.fuse_norm_buf) {
                std::vector<float> qn((size_t)hd), kn((size_t)hd), fused;
                for (uint32_t il = 0; il < n_layer; ++il) {
                    LagunaTargetLayer & L = out.layers[il];
                    if (!L.wqk || !L.qk_norm_f) continue;
                    const int nh = out.n_head_arr[il];
                    ggml_backend_tensor_get(L.q_norm, qn.data(), 0, sizeof(float) * hd);
                    ggml_backend_tensor_get(L.k_norm, kn.data(), 0, sizeof(float) * hd);
                    fused.resize((size_t)hd * (size_t)(nh + nkv));
                    for (int h = 0; h < nh + nkv; ++h) {
                        const float * src = h < nh ? qn.data() : kn.data();
                        std::memcpy(fused.data() + (size_t)h * hd, src, sizeof(float) * hd);
                    }
                    ggml_backend_tensor_set(L.qk_norm_f, fused.data(), 0,
                                            sizeof(float) * fused.size());
                }
            } else {
                for (uint32_t il = 0; il < n_layer; ++il) {
                    out.layers[il].wqk = nullptr;
                    out.layers[il].qk_norm_f = nullptr;
                }
            }
        }
    }

    gguf_free(gctx);

    if (tok_embd_off == 0 || tok_embd_type == GGML_TYPE_COUNT) {
        set_last_error("token_embd.weight not found or invalid type");
        return false;
    }

    // ── 5. Hand mmap to CpuEmbedder ──────────────────────────────────────
    // Transfer ownership of the mapping out of the RAII wrapper; the embedder's
    // destructor unmaps it. On Windows GgufMmap::release() has already closed the
    // mapping handle (the view stays valid until UnmapViewOfFile), and the file
    // handle was closed at open() time, so the embedder only needs the address.
    GgufMmap::OwnedRegion region = mm.release();
    out.embedder.mmap_addr      = const_cast<void *>(region.data);
    out.embedder.mmap_len       = region.size;
#if defined(_WIN32)
    out.embedder.mmap_hfile     = INVALID_HANDLE_VALUE;
    out.embedder.mmap_hmap      = nullptr;
#else
    out.embedder.mmap_fd        = region.fd;
#endif
    out.embedder.tok_embd_bytes = (const uint8_t *)region.data + tok_embd_off;
    out.embedder.tok_embd_type  = tok_embd_type;
    out.embedder.n_embd         = out.n_embd;
    out.embedder.n_vocab        = (int64_t)n_vocab;
    out.embedder.row_bytes      = tok_embd_sz / (size_t)n_vocab;

    char summary[224];
    std::snprintf(summary, sizeof(summary),
        "laguna target loaded: layers [%d,%d) output=%d tensors=%zu GPU %.2f GiB, tok_embd %.0f MiB CPU-only (%s)",
        plan.layer_begin, plan.layer_end, plan.load_output ? 1 : 0,
        allocs.size(), total / (1024.0 * 1024.0 * 1024.0),
        tok_embd_sz / (1024.0 * 1024.0), ggml_type_name(tok_embd_type));
    set_last_error(summary);
    std::printf("[laguna-loader] %s\n", summary);
    return true;
}

void free_laguna_target_weights(LagunaTargetWeights & w) {
    if (w.fuse_norm_buf) { ggml_backend_buffer_free(w.fuse_norm_buf); w.fuse_norm_buf = nullptr; }
    if (w.fuse_norm_ctx) { ggml_free(w.fuse_norm_ctx); w.fuse_norm_ctx = nullptr; }
    if (w.fuse_ctx)      { ggml_free(w.fuse_ctx);      w.fuse_ctx = nullptr; }
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    // CpuEmbedder destructor handles mmap.
    w.layers.clear();
    w.tok_embd = nullptr;
    w.out_norm = nullptr;
    w.output   = nullptr;
}

void free_laguna_target_feat(LagunaTargetCache & c) {
    if (c.feat_buf) { ggml_backend_buffer_free(c.feat_buf); c.feat_buf = nullptr; }
    if (c.feat_ctx) { ggml_free(c.feat_ctx); c.feat_ctx = nullptr; }
    c.target_feat = nullptr;
    c.target_feat_cap = 0;
    c.n_capture_layers = 0;
    c.capture_layer_ids.clear();
}

bool create_laguna_target_feat(ggml_backend_t backend,
                                LagunaTargetCache & cache,
                                int n_capture_layers,
                                int hidden_size,
                                int cap,
                                const std::vector<int> & explicit_ids) {
    if (n_capture_layers <= 0 || hidden_size <= 0 || cap <= 0) return false;

    free_laguna_target_feat(cache);

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + 4096;
    ip.no_alloc = true;
    cache.feat_ctx = ggml_init(ip);
    if (!cache.feat_ctx) return false;

    const int fc_in = n_capture_layers * hidden_size;
    cache.target_feat = ggml_new_tensor_2d(cache.feat_ctx, GGML_TYPE_BF16, fc_in, cap);
    ggml_set_name(cache.target_feat, "laguna_target_feat");

    cache.feat_buf = ggml_backend_alloc_ctx_tensors(cache.feat_ctx, backend);
    if (!cache.feat_buf) {
        ggml_free(cache.feat_ctx);
        cache.feat_ctx = nullptr;
        cache.target_feat = nullptr;
        return false;
    }

    cache.target_feat_cap = cap;
    cache.n_capture_layers = n_capture_layers;

    if ((int)explicit_ids.size() == n_capture_layers) {
        // Data-driven: ids shipped in the draft GGUF (dflash.target_layer_ids),
        // copied by the converter from the drafter's config.json. Works for any
        // drafter without a hardcoded per-arch set.
        cache.capture_layer_ids = explicit_ids;
    } else if (n_capture_layers == 5) {
        // Legacy fallback for GGUFs converted before target_layer_ids was
        // emitted (poolside Laguna-XS.2 speculator: {1,9,17,36,39}).
        cache.capture_layer_ids = {1, 9, 17, 36, 39};
    } else {
        std::fprintf(stderr,
            "[laguna] warning: DFlash draft has %d capture layers and no "
            "dflash.target_layer_ids in the GGUF; falling back to linspace ids "
            "(reconvert the drafter to embed its trained capture ids)\n",
            n_capture_layers);
        cache.capture_layer_ids.resize((size_t)n_capture_layers);
        const int n_layer = !cache.attn_k.empty() ? (int)cache.attn_k.size() : 40;
        if (n_capture_layers == 1) {
            cache.capture_layer_ids[0] = 1;
        } else {
            for (int k = 0; k < n_capture_layers; k++) {
                cache.capture_layer_ids[(size_t)k] = (int)std::round(
                    1.0 + k * (double)(n_layer - 4) / (n_capture_layers - 1));
            }
        }
    }

    return true;
}

// ── Partial loader (hybrid mode) ────────────────────────────────────────
//
// Loads laguna GGUF but skips uploading expert tensor DATA to GPU.
// Tensor metadata (shapes, offsets) is still parsed so that the hybrid
// storage builder can use ggml_nbytes() to compute per-expert sizes.
// Expert data will be loaded via mmap into the hot/cold split buffers.

static bool is_laguna_expert_tensor(const char * name) {
    // Expert tensors are: ffn_gate_exps, ffn_up_exps, ffn_down_exps
    // (per-layer, named blk.<N>.ffn_{gate,up,down}_exps.weight)
    return std::strstr(name, "ffn_gate_exps") != nullptr ||
           std::strstr(name, "ffn_up_exps") != nullptr ||
           std::strstr(name, "ffn_down_exps") != nullptr;
}
} // namespace dflash::common
