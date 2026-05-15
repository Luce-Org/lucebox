// DFlash speculative decoding end-to-end test.
//
// Pipeline:
//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).
//   2. Prefill: single-token decode over the prompt, capture_layers=true so
//      target_feat gets populated for every prompt pos.
//   3. Decode loop (until max_new):
//      a. Build noise block [last_tok, MASK*15] on CPU via target.tok_embd.
//      b. Draft forward (uses target_feat[0..committed] + noise) → 16 candidates.
//      c. snapshot SSM state. Batched target verify on the 16 draft tokens with
//         causal mask, capture_layers=true.
//      d. Greedy longest-prefix accept + 1 bonus token from target's argmax.
//      e. Restore SSM state. Replay the accepted tokens through target (batched
//         with causal mask, capture_layers=true) so state + target_feat are
//         cleanly advanced only by what was committed.
//      f. Update committed, last_tok.
//
// Usage: test_dflash <target.gguf> <draft.safetensors> <prompt_ids.bin>
//                    <n_gen> <out_ids.bin>

#include "dflash27b.h"
#include "internal.h"
#include "draft_graph.h"
#include "qwen3_drafter.h"
#include "gpu_runtime_compat.h"
#include "laguna_daemon.h"  // arch dispatch - laguna targets are served by
                            // dflash27b::run_laguna_daemon() instead of the
                            // qwen35 + DFlash + DDTree pipeline below.
#include "qwen35_daemon.h"  // arch dispatch - single-GPU qwen35 daemon mode
#include "qwen35_layer_split.h" // multi-GPU layer-split daemon args
#include "layer_split_daemon_loop.h" // extracted layer-split daemon loop
#include "qwen3_daemon.h"   // arch dispatch - qwen3 (0.6B standalone)
#include "gemma4_daemon.h"  // arch dispatch - gemma4 (iSWA + MoE)
#include "sampler.h"        // shared CPU sampler chain (SamplerCfg /
                            // sample_logits / parse_sampler_token) used by
                            // both arches; behaviour stays identical.

#include "device_runtime.h"

#include "ggml.h"
#include "gguf.h"  // gguf_init_from_file / gguf_find_key for arch detect
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

// ggml-cuda dequantize: Q8_0/F16/BF16 → F32. Replaces the custom
// f16_convert.cu kernels with ggml's built-in converter dispatch.
// On HIP, device_runtime.h aliases cudaStream_t → hipStream_t, and ggml-hip
// (built from the same convert.cu source) exports the same symbol name.
using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <io.h>
#ifdef _WIN64
#define ssize_t __int64
#else
#define ssize_t long
#endif
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#endif

#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <unordered_set>

using namespace dflash27b;

static SamplerCfg      g_sampler;
static std::mt19937_64 g_sampler_rng{std::random_device{}()};

// True iff `tok` matches one of the model's declared end-of-output ids
// (loaded into TargetWeights from GGUF tokenizer metadata). Replaces the
// previous hardcoded `tok == 248045` check at the spec-decode commit
// sites: token 248045 is `<|im_start|>` (chat-START), not EOS, and the
// old check truncated chat-formatted output at the natural turn boundary
// `<|im_end|>\n<|im_start|>`. The `>= 0` guards make a missing GGUF key
// (-1 sentinel) a never-match.
#define IS_EOS_TOK(tok, w)                                         \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id)                  \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

// ─── Small utilities — extracted to src/common/io_utils.h ──────────
#include "io_utils.h"
using dflash27b::read_int32_file;
using dflash27b::write_int32_file;
using dflash27b::stream_emit_fd;
using dflash27b::argmax_f32;
using dflash27b::write_binary_file;
using dflash27b::read_binary_file_exact;
using dflash27b::read_line_tail;
#if !defined(_WIN32)
using dflash27b::read_exact_fd;
using dflash27b::write_exact_fd;
#endif

// CPU sampler chain (SamplerCfg / sample_logits / parse_sampler_token) lives
// in src/sampler.{h,cpp} and is shared with src/laguna_daemon.cpp. Behaviour
// is unchanged: greedy when cfg.temp <= 0, otherwise rep_penalty -> top_k ->
// softmax(temp) -> top_p -> draw. The DDTree skeleton itself stays argmax to
// keep the accept rate intact; sample_logits only runs at committed-token
// sites when ` samp=` was on the request line.

// ggml_flash_attn_ext expects kv_len aligned to KQ_MASK_PAD (32) on the
// f16/Q* paths, and to FATTN_KQ_STRIDE (256) on the TurboQuant FA paths.
// The global `g_kq_stride_pad` below is set at init time and forwarded to
// build_causal_mask / build_tree_mask (now in src/qwen35/attn_masks.h).
#include "attn_masks.h"
using dflash27b::KQ_MASK_PAD;
using dflash27b::F16_ZERO;
using dflash27b::F16_NEG_INF;
using dflash27b::align_up;
using dflash27b::build_causal_mask;
using dflash27b::build_tree_mask;
static int g_kq_stride_pad = KQ_MASK_PAD;   // overridden to 256 when TBQ KV is active
static int g_max_ctx_override = 0;           // overridden by --max-ctx=N (default 4096)
static int g_fa_window       = 2048;         // overridden by DFLASH27B_FA_WINDOW=N
static int g_draft_swa_window = 0;           // draft SWA window (0 = disabled); --draft-swa=N
static int g_draft_ctx_max   = 4096;        // draft context cap; --draft-ctx-max=N

// ─── DDTree support ───────────────────────────────────────────────────
// Extracted to src/qwen35/ddtree.{h,cpp}. Provides DDTree struct,
// extract_draft_topk(), build_ddtree(), follow_verified_tree().
#include "ddtree.h"
using dflash27b::DDTree;
using dflash27b::extract_draft_topk;
using dflash27b::build_ddtree;
using dflash27b::follow_verified_tree;

// ─── StepGraph — extracted to src/qwen35/step_graph.h ──
#include "step_graph.h"
using dflash27b::StepGraph;
using dflash27b::step_graph_free;
using dflash27b::step_graph_destroy;

// ─── Peer access + DraftFeatureMirror — extracted to src/qwen35/ ──
#include "peer_access.h"
#include "dflash_feature_ring.h"
using dflash27b::g_peer_access_opt_in;
using dflash27b::g_peer_pair_ok_cache;
using dflash27b::enable_peer_access_one_way;
using dflash27b::enable_peer_access_pair;
using dflash27b::cross_device_peer_memcpy_ok;
using dflash27b::copy_peer_async;
using dflash27b::DraftFeatureMirror;
using dflash27b::draft_feature_mirror_free;
using dflash27b::draft_feature_mirror_init;
using dflash27b::draft_feature_mirror_can_view;
using dflash27b::draft_feature_mirror_sync_range;
using dflash27b::draft_feature_mirror_sync_tail;

// ─── Graph builders — extracted to src/qwen35/graph_builders.{h,cpp} ──
#include "graph_builders.h"
#include "dflash_draft_graph.h"
using dflash27b::build_layer_step;
using dflash27b::build_target_step;
using dflash27b::build_target_step_tree;
using dflash27b::build_draft_step;
using dflash27b::build_lm_head_projection_step;

// ─── Layer split types — extracted to src/qwen35/layer_split_types.h ──
#include "layer_split_types.h"
using dflash27b::LayerSplitRuntimeConfig;
using dflash27b::TargetLayerSplitShard;
using dflash27b::ActivationPair;
using dflash27b::activation_pair_free;
using dflash27b::activation_pair_init;
using dflash27b::find_target_shard;

static bool parse_int_list(const char * text, std::vector<int> & out) {
    out.clear();
    if (!text || !*text) return false;
    const char * p = text;
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return false;
        out.push_back((int)v);
        if (*end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return !out.empty();
}

static bool parse_float_list(const char * text, std::vector<double> & out) {
    out.clear();
    if (!text || !*text) return false;
    const char * p = text;
    while (*p) {
        char * end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p || v <= 0.0) return false;
        out.push_back(v);
        if (*end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return !out.empty();
}

// ─── Draft IPC — extracted to src/qwen35/draft_ipc.{h,cpp} ──
#include "dflash_draft_ipc.h"
using dflash27b::DFlashDraftIpcClient;
using dflash27b::copy_capture_slice_to_remote_draft;
using dflash27b::stream_status;
using dflash27b::run_dflash_draft_ipc_daemon;

// ─── GGUF inspection — extracted to src/common/gguf_inspect.{h,cpp} ──
#include "gguf_inspect.h"

// ─── Layer ranges — extracted to src/common/layer_split_utils.{h,cpp} ──
#include "layer_split_utils.h"
using dflash27b::compute_layer_ranges;

// ─── Feature copy helpers — extracted to src/qwen35/feature_copy.{h,cpp} ──
#include "dflash_capture.h"
using dflash27b::target_capture_index;
using dflash27b::copy_capture_slice_to_draft_ring;
using dflash27b::copy_feature_ring_range_to_tensor;

// ─── Layer-split forward — extracted to src/qwen35/layer_split_forward.{h,cpp} ──
#include "layer_split_forward.h"
using dflash27b::compute_target_split_argmax;
using dflash27b::run_target_layer_split_forward;
using dflash27b::free_target_layer_split_shards;


// ─── Speculative decode — generic loop in common/, qwen35 layer-split adapter.
#include "qwen35_layer_split_dflash_target.h"
#include "common/dflash_spec_decode.h"
using dflash27b::is_eos_tok;

// ─── Layer-split daemon — extracted to src/qwen35/layer_split_daemon.{h,cpp} ─
#include "layer_split_daemon.h"
using dflash27b::run_target_layer_split_request;

static int run_target_layer_split_daemon(
        const char * target_path,
        const char * draft_path,
        const std::vector<int> & target_gpus,
        const std::vector<double> & split_weights,
        int draft_gpu,
        bool load_draft,
        bool run_dflash,
        int max_ctx,
        int max_verify_tokens,
        bool peer_access,
        int stream_fd) {
    LayerSplitDaemonConfig cfg;
    cfg.target_path = target_path;
    cfg.draft_path = draft_path;
    cfg.target_gpus = target_gpus;
    cfg.split_weights = split_weights;
    cfg.draft_gpu = draft_gpu;
    cfg.load_draft = load_draft;
    cfg.run_dflash = run_dflash;
    cfg.max_ctx = max_ctx;
    cfg.max_verify_tokens = max_verify_tokens;
    cfg.peer_access = peer_access;
    cfg.stream_fd = stream_fd;
    cfg.kq_stride_pad = g_kq_stride_pad;
    cfg.fa_window = g_fa_window;
    cfg.draft_ctx_max = g_draft_ctx_max;
    return run_layer_split_daemon(cfg);
}

static int run_target_layer_split_harness(
        const char * target_path,
        const char * draft_path,
        const char * prompt_path,
        int n_gen,
        const char * out_path,
        const std::vector<int> & target_gpus,
        const std::vector<double> & split_weights,
        int draft_gpu,
        bool load_draft,
        bool run_draft_smoke,
        bool run_dflash,
        int max_ctx,
        int max_verify_tokens,
        bool peer_access,
        const char * draft_ipc_bin = nullptr,
        int draft_ipc_gpu = 0,
        const char * draft_ipc_work_dir = nullptr,
        int draft_ipc_ring_cap = 0) {
    g_peer_access_opt_in = peer_access;
    g_peer_pair_ok_cache.clear();
    if (!prompt_path || !out_path) {
        std::fprintf(stderr, "target layer split requires prompt/n_gen/out positional args\n");
        return 2;
    }
    const int n_layer = dflash27b::inspect_gguf_model_info(target_path).n_layer;
    if (n_layer <= 0) {
        std::fprintf(stderr, "target-split could not read qwen35.block_count\n");
        return 1;
    }
    const auto ranges = compute_layer_ranges(n_layer, (int)target_gpus.size(), split_weights);
    if ((int)ranges.size() != (int)target_gpus.size()) {
        std::fprintf(stderr, "bad --target-layer-split for %zu target GPUs and %d layers\n",
                     target_gpus.size(), n_layer);
        return 2;
    }
    std::vector<TargetLayerSplitShard> shards;
    shards.resize(target_gpus.size());
    for (size_t i = 0; i < target_gpus.size(); i++) {
        shards[i].gpu = target_gpus[i];
        shards[i].layer_begin = ranges[i].first;
        shards[i].layer_end = ranges[i].second;
    }
    for (auto & shard : shards) {
        shard.backend = ggml_backend_cuda_init(shard.gpu);
        if (!shard.backend) {
            std::fprintf(stderr, "target-split cuda init failed for gpu %d\n", shard.gpu);
            free_target_layer_split_shards(shards);
            return 1;
        }
    }
    for (size_t i = 0; i < target_gpus.size(); i++) {
        for (size_t j = i + 1; j < target_gpus.size(); j++) {
            if (!enable_peer_access_pair(target_gpus[i], target_gpus[j])) {
                std::fprintf(stderr,
                             "warning: CUDA peer access not fully enabled for target gpus %d,%d\n",
                             target_gpus[i], target_gpus[j]);
            }
        }
    }
    for (auto & shard : shards) {
        TargetLoadPlan plan;
        plan.layer_begin = shard.layer_begin;
        plan.layer_end = shard.layer_end;
        plan.load_output = (&shard == &shards.back());
        if (!load_target_gguf_partial(target_path, shard.backend, plan, shard.weights)) {
            std::fprintf(stderr, "target-split load gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] gpu=%d layers=[%d,%d) %s\n",
                    shard.gpu, shard.layer_begin, shard.layer_end,
                    dflash27b_last_error());
        const bool allocate_target_feat = false;
        if (!create_target_cache_partial(shard.weights, max_ctx, max_verify_tokens,
                                         shard.backend, shard.cache,
                                         /*prefill_only=*/!run_dflash,
                                         shard.layer_begin, shard.layer_end,
                                         allocate_target_feat)) {
            std::fprintf(stderr, "target-split cache gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            free_target_layer_split_shards(shards);
            return 1;
        }
    }

    ggml_backend_t draft_backend = nullptr;
    DraftWeights draft_weights;
    DraftFeatureMirror feature_ring;
    DFlashDraftIpcClient remote_draft;
    bool draft_backend_owned = false;
    const bool use_remote_draft = draft_ipc_bin && *draft_ipc_bin;
    if (load_draft) {
        const int cap = draft_ipc_ring_cap > 0
            ? std::min(draft_ipc_ring_cap, max_ctx)
            : std::min(max_ctx, 4096);
        if (use_remote_draft) {
            if (!remote_draft.start(draft_ipc_bin, draft_path, draft_ipc_gpu,
                                    cap, draft_ipc_work_dir ? draft_ipc_work_dir : "")) {
                std::fprintf(stderr, "target-split remote draft start failed\n");
                free_target_layer_split_shards(shards);
                return 1;
            }
        } else {
            for (auto & shard : shards) {
                if (shard.gpu == draft_gpu) {
                    draft_backend = shard.backend;
                    break;
                }
            }
            if (!draft_backend) {
                draft_backend = ggml_backend_cuda_init(draft_gpu);
                if (!draft_backend) {
                    std::fprintf(stderr, "target-split draft cuda init failed for gpu %d\n", draft_gpu);
                    free_target_layer_split_shards(shards);
                    return 1;
                }
                draft_backend_owned = true;
            }
            std::string dp(draft_path);
            bool draft_ok = false;
            if (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf") {
                draft_ok = load_draft_gguf(draft_path, draft_backend, draft_weights);
            } else {
                draft_ok = load_draft_safetensors(draft_path, draft_backend, draft_weights);
            }
            if (!draft_ok) {
                std::fprintf(stderr, "target-split draft load gpu=%d: %s\n",
                             draft_gpu, dflash27b_last_error());
                free_draft_weights(draft_weights);
                if (draft_backend_owned) ggml_backend_free(draft_backend);
                free_target_layer_split_shards(shards);
                return 1;
            }
            std::printf("[target-split] draft loaded on gpu=%d format=%s\n",
                        draft_gpu,
                        (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
                            ? "gguf" : "safetensors");
            if (g_draft_swa_window > 0) {
                draft_weights.swa_window = g_draft_swa_window;
                for (int il = 0; il < draft_weights.n_layer - 1; il++) {
                    draft_weights.layers[il].is_swa = true;
                }
                std::printf("[target-split] draft SWA layers: %d/%d (window=%d)\n",
                            draft_weights.n_layer - 1, draft_weights.n_layer,
                            draft_weights.swa_window);
            }
            if (!draft_feature_mirror_init(feature_ring, draft_backend,
                                           draft_gpu, draft_gpu, cap,
                                           draft_weights.n_target_layers,
                                           draft_weights.n_embd)) {
                std::fprintf(stderr, "target-split feature ring init failed on gpu=%d\n", draft_gpu);
                draft_feature_mirror_free(feature_ring);
                free_draft_weights(draft_weights);
                if (draft_backend_owned) ggml_backend_free(draft_backend);
                free_target_layer_split_shards(shards);
                return 1;
            }
            std::printf("[target-split] draft feature ring cap=%d gpu=%d\n", cap, draft_gpu);
        }
    }

    auto prompt = read_int32_file(prompt_path);
    if (prompt.empty()) {
        std::fprintf(stderr, "target-split empty prompt\n");
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }
    if ((int)prompt.size() + n_gen + 1 > max_ctx) {
        std::fprintf(stderr, "target-split prompt (%zu) + gen (%d) exceeds max_ctx (%d)\n",
                     prompt.size(), n_gen, max_ctx);
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }

    int ubatch = (prompt.size() > 2048) ? 384 : 16;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        ubatch = std::max(1, std::atoi(s));
    }
    std::printf("[target-split] n_gpus=%zu n_layer=%d ubatch=%d max_ctx=%d\n",
                target_gpus.size(), n_layer, ubatch, max_ctx);

    int last_tok = -1;
    auto t_pf0 = std::chrono::steady_clock::now();
    if (!run_target_layer_split_forward(shards, shards.front().weights,
                                        prompt, 0, ubatch, last_tok,
                                        g_kq_stride_pad, g_fa_window,
                                        (load_draft && !use_remote_draft) ? &feature_ring : nullptr,
                                        nullptr, nullptr,
                                        (load_draft && use_remote_draft) ? &remote_draft : nullptr)) {
        std::fprintf(stderr, "target-split prefill failed\n");
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();
    std::printf("[target-split] prefill tokens=%zu time=%.3f s speed=%.2f tok/s last_tok=%d\n",
                prompt.size(), prefill_s, prompt.size() / prefill_s, last_tok);

    if (run_draft_smoke) {
        const int hidden = DFLASH27B_TARGET_HIDDEN;
        const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
        const int ring_cap = use_remote_draft ? remote_draft.ring_cap() : feature_ring.cap;
        const int draft_ctx = std::min((int)prompt.size(), ring_cap);
        const int draft_start = (int)prompt.size() - draft_ctx;
        std::vector<int32_t> noise_ids(q_len, DFLASH27B_DRAFT_MASK_TOKEN_ID);
        noise_ids[0] = last_tok;
        std::vector<float> noise_embed((size_t)hidden * q_len);
        if (!shards.front().weights.embedder.embed(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "target-split draft smoke embed failed\n");
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        if (use_remote_draft) {
            std::vector<float> hidden_out;
            auto t_ds0 = std::chrono::steady_clock::now();
            const bool ok = remote_draft.propose((int)prompt.size(), draft_ctx,
                                                 noise_embed, hidden_out);
            auto t_ds1 = std::chrono::steady_clock::now();
            if (!ok) {
                std::fprintf(stderr, "target-split remote draft smoke failed\n");
                draft_feature_mirror_free(feature_ring);
                free_draft_weights(draft_weights);
                if (draft_backend_owned) ggml_backend_free(draft_backend);
                free_target_layer_split_shards(shards);
                return 1;
            }
            std::printf("[target-split] remote draft smoke ctx=%d q=%d time=%.3f ms\n",
                        draft_ctx, q_len,
                        std::chrono::duration<double, std::milli>(t_ds1 - t_ds0).count());
        } else {
            StepGraph draft_sg;
            int mirror_slot0 = 0;
            const bool use_mirror_view =
                draft_feature_mirror_can_view(feature_ring, (int)prompt.size(),
                                              draft_ctx, mirror_slot0);
            if (!build_draft_step(draft_sg, draft_weights, /*lm_head=*/nullptr, draft_backend,
                                  draft_ctx, use_mirror_view ? &feature_ring : nullptr,
                                  (int)prompt.size())) {
                std::fprintf(stderr, "target-split draft smoke build failed\n");
                step_graph_destroy(draft_sg);
                draft_feature_mirror_free(feature_ring);
                free_draft_weights(draft_weights);
                if (draft_backend_owned) ggml_backend_free(draft_backend);
                free_target_layer_split_shards(shards);
                return 1;
            }
            if (!use_mirror_view &&
                !copy_feature_ring_range_to_tensor(feature_ring,
                                                   draft_sg.target_hidden_cat,
                                                   draft_start, draft_ctx)) {
                std::fprintf(stderr, "target-split draft smoke feature copy failed\n");
                step_graph_destroy(draft_sg);
                draft_feature_mirror_free(feature_ring);
                free_draft_weights(draft_weights);
                if (draft_backend_owned) ggml_backend_free(draft_backend);
                free_target_layer_split_shards(shards);
                return 1;
            }
            ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                    sizeof(float) * noise_embed.size());
            std::vector<int32_t> pos_q(q_len), pos_k(draft_ctx + q_len);
            for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
            for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
            ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                    sizeof(int32_t) * pos_q.size());
            ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                    sizeof(int32_t) * pos_k.size());
            auto t_ds0 = std::chrono::steady_clock::now();
            auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
            auto t_ds1 = std::chrono::steady_clock::now();
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "target-split draft smoke compute failed status=%d\n", (int)st);
                step_graph_destroy(draft_sg);
                draft_feature_mirror_free(feature_ring);
                free_draft_weights(draft_weights);
                if (draft_backend_owned) ggml_backend_free(draft_backend);
                free_target_layer_split_shards(shards);
                return 1;
            }
            std::printf("[target-split] draft smoke ctx=%d q=%d time=%.3f ms\n",
                        draft_ctx, q_len,
                        std::chrono::duration<double, std::milli>(t_ds1 - t_ds0).count());
            step_graph_destroy(draft_sg);
        }
    }

    if (run_dflash) {
        Qwen35LayerSplitDFlashTarget target(
            shards, &feature_ring, g_kq_stride_pad, g_fa_window,
            use_remote_draft ? &remote_draft : nullptr);
        const bool ok = run_dflash_spec_decode(
            target, draft_weights, draft_backend, feature_ring,
            prompt, n_gen, last_tok, out_path,
            g_draft_ctx_max, /*stream_fd=*/-1,
            use_remote_draft ? &remote_draft : nullptr);
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return ok ? 0 : 1;
    }

    std::vector<int32_t> out_all = prompt;
    auto t_dec0 = std::chrono::steady_clock::now();
    int generated = 0;
    for (; generated < n_gen; generated++) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            one, (int)out_all.size(), 1, next_tok,
                                            g_kq_stride_pad, g_fa_window,
                                            (load_draft && !use_remote_draft) ? &feature_ring : nullptr,
                                            nullptr, nullptr,
                                            (load_draft && use_remote_draft) ? &remote_draft : nullptr)) {
            std::fprintf(stderr, "target-split decode failed at %d\n", generated);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        out_all.push_back(last_tok);
        if (IS_EOS_TOK(last_tok, shards.front().weights)) {
            generated++;
            break;
        }
        last_tok = next_tok;
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    std::printf("[target-split] decode tokens=%d time=%.3f s speed=%.2f tok/s\n",
                generated, decode_s, generated > 0 ? generated / decode_s : 0.0);
    if (out_path) write_int32_file(out_path, out_all);
    draft_feature_mirror_free(feature_ring);
    free_draft_weights(draft_weights);
    if (draft_backend_owned) ggml_backend_free(draft_backend);
    free_target_layer_split_shards(shards);
    return 0;
}

// ─── Main ─────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--draft-ipc-daemon") == 0) {
        if (argc < 3) {
            std::fprintf(stderr,
                "usage: %s --draft-ipc-daemon <draft.safetensors|draft.gguf> --ring-cap=N --stream-fd=FD [--draft-gpu=N]\n",
                argv[0]);
            return 2;
        }
        const char * ipc_draft_path = argv[2];
        int ipc_ring_cap = 4096;
        int ipc_draft_gpu = 0;
        int ipc_stream_fd = -1;
        for (int i = 3; i < argc; i++) {
            if (std::strncmp(argv[i], "--ring-cap=", 11) == 0) {
                ipc_ring_cap = std::atoi(argv[i] + 11);
            } else if (std::strcmp(argv[i], "--ring-cap") == 0) {
                if (i + 1 < argc) ipc_ring_cap = std::atoi(argv[++i]);
            } else if (std::strncmp(argv[i], "--draft-gpu=", 12) == 0) {
                ipc_draft_gpu = std::max(0, std::atoi(argv[i] + 12));
            } else if (std::strcmp(argv[i], "--draft-gpu") == 0) {
                if (i + 1 < argc) ipc_draft_gpu = std::max(0, std::atoi(argv[++i]));
            } else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
                ipc_stream_fd = std::atoi(argv[i] + 12);
            } else if (std::strcmp(argv[i], "--stream-fd") == 0) {
                if (i + 1 < argc) ipc_stream_fd = std::atoi(argv[++i]);
            }
        }
        return run_dflash_draft_ipc_daemon(ipc_draft_path,
                                           ipc_ring_cap,
                                           ipc_draft_gpu,
                                           ipc_stream_fd);
    }
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <target.gguf> <draft.safetensors> [<prompt_ids.bin> <n_gen> <out_ids.bin>] [--daemon] [-ctk <type>] [-ctv <type>] ...\n"
            "       %s --draft-ipc-daemon <draft.safetensors|draft.gguf> --ring-cap=N --stream-fd=FD [--draft-gpu=N]\n",
            argv[0], argv[0]);
        return 2;
    }
    // TurboQuant FA kernel requires kv_len aligned to FATTN_KQ_STRIDE=256.
    // Bump the mask stride accordingly so the mask dim matches the kv view.
    if (const char * s = std::getenv("DFLASH27B_KV_TBQ")) {
        if (std::atoi(s) != 0) g_kq_stride_pad = 256;
    }
    if (const char * s = std::getenv("DFLASH27B_KV_TQ3")) {
        if (std::atoi(s) != 0) g_kq_stride_pad = 256;
    }
    if (const char * s = std::getenv("DFLASH27B_FA_WINDOW")) {
        g_fa_window = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH27B_DRAFT_SWA")) {
        g_draft_swa_window = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH27B_DRAFT_CTX_MAX")) {
        g_draft_ctx_max = std::max(0, std::atoi(s));
    }
    const char * target_path = argv[1];

    // ---- Architecture detection ------------------------------------------
    // Read general.architecture from the target GGUF before parsing argv
    // shape so we can route laguna requests to run_laguna_daemon() and
    // accept the no-draft argv layout server.py uses for that arch.
    #include "gguf_inspect.h"
    const auto model_info   = dflash27b::inspect_gguf_model_info(target_path);
    const std::string detected_arch = model_info.arch;
    const bool is_laguna = (detected_arch == "laguna");
    const bool is_qwen3  = (detected_arch == "qwen3");
    const bool is_gemma4 = (detected_arch == "gemma4");

    // When arch == laguna there is no DFlash draft model (Poolside hasn't
    // released one); server.py omits --draft from the spawn cmd. Accept the
    // shorter argv layout: argv[1] = target, argv[2..] = flags. Same fall-
    // back applies if the user manually drops the draft (argv[2] starts with
    // a dash) on any arch — keeps the binary friendly to ad-hoc invocation.
    const bool no_draft_layout = is_laguna || is_qwen3 || is_gemma4 || (argc >= 3 && argv[2][0] == '-');
    const char * draft_path  = no_draft_layout ? nullptr : argv[2];
    const int    flags_start = no_draft_layout ? 2 : 3;
    const bool   has_positional_args =
        (!no_draft_layout) && (argc >= 6 && argv[3][0] != '-');
    const char * prompt_path = has_positional_args ? argv[3] : nullptr;
    int          n_gen       = has_positional_args ? std::atoi(argv[4]) : 0;
    const char * out_path    = has_positional_args ? argv[5] : nullptr;
    // --seq-verify: run the target verify as q_len independent single-token
    // decodes instead of one batched forward with a causal mask. Isolates
    // the correctness-of-batched-verify hypothesis from z-lab issue #57.
    //
    // --fast-rollback: use per-step SSM intermediate-state capture (kernel mod
    // in ggml_gated_delta_net) to roll back state after verify without the
    // replay forward pass. Implicit-bonus variant: commit only accept_n tokens
    // per step, let next iter's draft pick up the "bonus" via last_tok.
    //
    // --ddtree [--ddtree-budget=B]: use DDTree-style tree-structured verify on
    // top of the fast-rollback path. Ported from liranringel/ddtree.py with
    // our tree-aware gated_delta_net kernel handling the DeltaNet/SSM tree
    // recurrence via parent_ids (Qwen3.5 hybrid support beyond the original
    // paper's pure-attention Qwen3 experiments). Default budget = 64.
    bool  seq_verify    = false;
    bool  fast_rollback = false;
    bool  ddtree_mode   = false;
    int   ddtree_budget = 64;
    float ddtree_temp   = 1.0f;   // softmax temperature for top-K extract
    bool  ddtree_chain_seed = true;  // pre-seed full chain (vs paper's pure best-first)
    bool  profile_scaling = false;  // microbench: time target forward at varying N
    bool  test_window_mode = false;
    bool  draft_feature_mirror = false;
    bool  target_split_load_draft = false;
    bool  target_split_dflash = false;
    int   target_gpu = 0;
    int   draft_gpu = 0;
    const char * draft_ipc_bin = nullptr;
    const char * draft_ipc_work_dir = nullptr;
    int   draft_ipc_gpu = 0;
    int   draft_ipc_ring_cap = 0;
    std::vector<int> target_gpus;
    std::vector<double> target_split_weights;
    if (const char * s = std::getenv("DFLASH_TARGET_GPU")) {
        target_gpu = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH_DRAFT_GPU")) {
        draft_gpu = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH_DRAFT_IPC_BIN")) {
        draft_ipc_bin = s;
    }
    if (const char * s = std::getenv("DFLASH_DRAFT_IPC_GPU")) {
        draft_ipc_gpu = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH_DRAFT_IPC_WORK_DIR")) {
        draft_ipc_work_dir = s;
    }
    if (const char * s = std::getenv("DFLASH_DRAFT_IPC_RING_CAP")) {
        draft_ipc_ring_cap = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH_TARGET_GPUS")) {
        if (!parse_int_list(s, target_gpus)) {
            std::fprintf(stderr, "bad DFLASH_TARGET_GPUS=%s\n", s);
            return 2;
        }
    }
    if (const char * s = std::getenv("DFLASH_TARGET_LAYER_SPLIT")) {
        if (!parse_float_list(s, target_split_weights)) {
            std::fprintf(stderr, "bad DFLASH_TARGET_LAYER_SPLIT=%s\n", s);
            return 2;
        }
    }
    int   stream_fd     = -1;     // write each committed token to this fd (int32 LE) as they land
    bool  daemon_mode   = false;
    for (int i = flags_start; i < argc; i++) {
        if      (std::strcmp(argv[i], "--daemon") == 0)        daemon_mode = true;
        else if (std::strcmp(argv[i], "--seq-verify") == 0)    seq_verify = true;
        else if (std::strcmp(argv[i], "--fast-rollback") == 0) fast_rollback = true;
        else if (std::strcmp(argv[i], "--ddtree") == 0)        { ddtree_mode = true; fast_rollback = true; }
        else if (std::strncmp(argv[i], "--ddtree-budget=", 16) == 0) {
            ddtree_budget = std::atoi(argv[i] + 16);
            if (ddtree_budget <= 0) ddtree_budget = 64;
        }
        else if (std::strncmp(argv[i], "--ddtree-temp=", 14) == 0) {
            ddtree_temp = (float)std::atof(argv[i] + 14);
            if (ddtree_temp <= 0.0f) ddtree_temp = 1.0f;
        }
        else if (std::strcmp(argv[i], "--ddtree-no-chain-seed") == 0) {
            ddtree_chain_seed = false;
        }
        else if (std::strcmp(argv[i], "--test-window") == 0)      { test_window_mode = true; }
        else if (std::strcmp(argv[i], "--draft-feature-mirror") == 0) {
            draft_feature_mirror = true;
        }
        else if (std::strcmp(argv[i], "--peer-access") == 0) {
            g_peer_access_opt_in = true;
        }
        else if (std::strcmp(argv[i], "--target-split-load-draft") == 0) {
            target_split_load_draft = true;
        }
        else if (std::strcmp(argv[i], "--target-split-dflash") == 0) {
            target_split_dflash = true;
            target_split_load_draft = true;
        }
        else if (std::strncmp(argv[i], "--target-gpu=", 13) == 0) {
            target_gpu = std::max(0, std::atoi(argv[i] + 13));
        }
        else if (std::strcmp(argv[i], "--target-gpu") == 0) {
            if (i + 1 < argc) target_gpu = std::max(0, std::atoi(argv[++i]));
        }
        else if (std::strncmp(argv[i], "--target-gpus=", 14) == 0) {
            if (!parse_int_list(argv[i] + 14, target_gpus)) {
                std::fprintf(stderr, "bad --target-gpus value\n");
                return 2;
            }
        }
        else if (std::strcmp(argv[i], "--target-gpus") == 0) {
            if (i + 1 >= argc || !parse_int_list(argv[++i], target_gpus)) {
                std::fprintf(stderr, "bad --target-gpus value\n");
                return 2;
            }
        }
        else if (std::strncmp(argv[i], "--target-layer-split=", 21) == 0) {
            if (!parse_float_list(argv[i] + 21, target_split_weights)) {
                std::fprintf(stderr, "bad --target-layer-split value\n");
                return 2;
            }
        }
        else if (std::strcmp(argv[i], "--target-layer-split") == 0) {
            if (i + 1 >= argc || !parse_float_list(argv[++i], target_split_weights)) {
                std::fprintf(stderr, "bad --target-layer-split value\n");
                return 2;
            }
        }
        else if (std::strncmp(argv[i], "--draft-gpu=", 12) == 0) {
            draft_gpu = std::max(0, std::atoi(argv[i] + 12));
        }
        else if (std::strcmp(argv[i], "--draft-gpu") == 0) {
            if (i + 1 < argc) draft_gpu = std::max(0, std::atoi(argv[++i]));
        }
        else if (std::strncmp(argv[i], "--draft-ipc-bin=", 16) == 0) {
            draft_ipc_bin = argv[i] + 16;
        }
        else if (std::strcmp(argv[i], "--draft-ipc-bin") == 0) {
            if (i + 1 < argc) draft_ipc_bin = argv[++i];
        }
        else if (std::strncmp(argv[i], "--draft-ipc-gpu=", 16) == 0) {
            draft_ipc_gpu = std::max(0, std::atoi(argv[i] + 16));
        }
        else if (std::strcmp(argv[i], "--draft-ipc-gpu") == 0) {
            if (i + 1 < argc) draft_ipc_gpu = std::max(0, std::atoi(argv[++i]));
        }
        else if (std::strncmp(argv[i], "--draft-ipc-work-dir=", 21) == 0) {
            draft_ipc_work_dir = argv[i] + 21;
        }
        else if (std::strcmp(argv[i], "--draft-ipc-work-dir") == 0) {
            if (i + 1 < argc) draft_ipc_work_dir = argv[++i];
        }
        else if (std::strncmp(argv[i], "--draft-ipc-ring-cap=", 21) == 0) {
            draft_ipc_ring_cap = std::max(0, std::atoi(argv[i] + 21));
        }
        else if (std::strcmp(argv[i], "--draft-ipc-ring-cap") == 0) {
            if (i + 1 < argc) draft_ipc_ring_cap = std::max(0, std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--profile-scaling") == 0) {
            profile_scaling = true;
        }
        else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
        else if (std::strncmp(argv[i], "--max-ctx=", 10) == 0) {
            g_max_ctx_override = std::atoi(argv[i] + 10);
        }
        // KV cache type flags (mirror llama-cli -ctk / -ctv).
        // Set the env var before resolve_kv_types() reads it inside create_target_cache.
        else if (std::strcmp(argv[i], "--cache-type-k") == 0 || std::strcmp(argv[i], "-ctk") == 0) {
            if (i + 1 < argc) setenv("DFLASH27B_KV_K", argv[++i], 1);
        }
        else if (std::strncmp(argv[i], "--cache-type-k=", 15) == 0) {
            setenv("DFLASH27B_KV_K", argv[i] + 15, 1);
        }
        else if (std::strncmp(argv[i], "-ctk=", 5) == 0) {
            setenv("DFLASH27B_KV_K", argv[i] + 5, 1);
        }
        else if (std::strcmp(argv[i], "--cache-type-v") == 0 || std::strcmp(argv[i], "-ctv") == 0) {
            if (i + 1 < argc) setenv("DFLASH27B_KV_V", argv[++i], 1);
        }
        else if (std::strncmp(argv[i], "--cache-type-v=", 15) == 0) {
            setenv("DFLASH27B_KV_V", argv[i] + 15, 1);
        }
        else if (std::strncmp(argv[i], "-ctv=", 5) == 0) {
            setenv("DFLASH27B_KV_V", argv[i] + 5, 1);
        }
        else if (std::strncmp(argv[i], "--draft-swa=", 12) == 0) {
            g_draft_swa_window = std::max(0, std::atoi(argv[i] + 12));
        }
        else if (std::strncmp(argv[i], "--draft-ctx-max=", 16) == 0) {
            g_draft_ctx_max = std::max(0, std::atoi(argv[i] + 16));
        }
    }

    // The KV type may also have been chosen via -ctk/-ctv, which sets
    // DFLASH27B_KV_K / DFLASH27B_KV_V during the argv loop above. Re-check
    // for TQ3 here so g_kq_stride_pad matches the chunked-FA driver's
    // align_up(kv_len, 256); otherwise the host-built mask is short and the
    // kernel reads past its end.
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

    if (!is_laguna && !daemon_mode && !test_window_mode && (!prompt_path || !out_path)) {
        std::fprintf(stderr, "Missing positional arguments for non-daemon mode.\n");
        return 2;
    }

    // ---- Arch dispatch: hand laguna targets to the dedicated daemon -----
    // The qwen35 + DFlash + DDTree code path below assumes the target is a
    // qwen35-shaped hybrid (attention + DeltaNet/SSM) and that a draft model
    // exists. Laguna is a pure-attention MoE arch with no published draft,
    // so dispatch to run_laguna_daemon() before any qwen35-specific init.
    // The daemon protocol it speaks (bare prompt, samp= tail, generate cmd)
    // matches what scripts/server.py emits, so the OpenAI HTTP path is
    // byte-identical for the two arches — only the binary'́s internal
    // forward kernels differ.
    if (is_laguna) {
        ggml_type kv = GGML_TYPE_Q8_0;
        if (const char * kvs = std::getenv("DFLASH27B_KV_K")) {
            std::string s = kvs;
            if      (s == "q4_0") kv = GGML_TYPE_Q4_0;
            else if (s == "q5_0") kv = GGML_TYPE_Q5_0;
            else if (s == "q8_0") kv = GGML_TYPE_Q8_0;
            else if (s == "f16")  kv = GGML_TYPE_F16;
        }
        const int max_ctx_eff = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;
        int chunk = 2048;
        if (const char * ck = std::getenv("DFLASH27B_LAGUNA_CHUNK")) {
            const int v = std::atoi(ck);
            if (v > 0) chunk = v;
        }
        std::fprintf(stderr,
            "[test_dflash] arch=laguna -> dispatching to run_laguna_daemon "
            "(max_ctx=%d kv=%s chunk=%d stream_fd=%d). DFlash + DDTree disabled.\n",
            max_ctx_eff, ggml_type_name(kv), chunk, stream_fd);
        dflash27b::LagunaDaemonArgs largs;
        largs.target_path     = target_path;
        largs.device.max_ctx  = max_ctx_eff;
        largs.chunk           = chunk;
        largs.kv_type         = kv;
        largs.stream_fd       = stream_fd;
        return dflash27b::run_laguna_daemon(largs);
    }

    // ---- Arch dispatch: qwen3 targets to the dedicated daemon -----
    if (is_qwen3 && daemon_mode) {
        const int max_ctx_eff = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;
        std::fprintf(stderr,
            "[test_dflash] arch=qwen3 -> dispatching to run_qwen3_daemon "
            "(max_ctx=%d stream_fd=%d)\n", max_ctx_eff, stream_fd);
        dflash27b::Qwen3DaemonArgs q3args;
        q3args.model_path     = target_path;
        q3args.device.gpu     = target_gpu;
        q3args.device.max_ctx = max_ctx_eff;
        q3args.stream_fd      = stream_fd;
        q3args.chunk          = 512;
        return dflash27b::run_qwen3_daemon(q3args);
    }

    // ---- Arch dispatch: gemma4 targets to the dedicated daemon -----
    if (is_gemma4 && daemon_mode) {
        const int max_ctx_eff = g_max_ctx_override > 0 ? g_max_ctx_override : 8192;
        std::fprintf(stderr,
            "[test_dflash] arch=gemma4 -> dispatching to run_gemma4_daemon "
            "(max_ctx=%d stream_fd=%d)\n", max_ctx_eff, stream_fd);
        dflash27b::Gemma4DaemonArgs g4args;
        g4args.model_path     = target_path;
        g4args.device.gpu     = target_gpu;
        g4args.device.max_ctx = max_ctx_eff;
        g4args.stream_fd      = stream_fd;
        g4args.chunk          = 512;
        return dflash27b::run_gemma4_daemon(g4args);
    }

    // Helper: write a committed token to the stream fd immediately (int32 LE).
    // Caller invokes after every out_all.push_back(tok) when stream_fd >= 0.
    // On Windows stream_fd holds a Win32 HANDLE value (passed via msvcrt.get_osfhandle).
    auto stream_emit = [&](int32_t tok) {
        if (stream_fd < 0) return;
        int32_t v = tok;
#if defined(_WIN32)
        DWORD written;
        WriteFile((HANDLE)(intptr_t)stream_fd, &v, sizeof(v), &written, nullptr);
#else
        ssize_t n = ::write(stream_fd, &v, sizeof(v));
        (void)n;
#endif
    };
    if (fast_rollback && seq_verify && !ddtree_mode) {
        std::fprintf(stderr, "--fast-rollback and --seq-verify are mutually exclusive\n");
        return 2;
    }
    if (target_split_dflash) target_split_load_draft = true;
    if (target_gpus.empty()) target_gpus.push_back(target_gpu);
    if (target_gpus.size() == 1) target_gpu = target_gpus[0];
    if (draft_ipc_bin && target_gpus.size() <= 1) {
        std::fprintf(stderr,
                     "--draft-ipc-bin currently applies to --target-gpus layer-split DFlash only\n");
        return 2;
    }
    if (draft_ipc_bin && !target_split_load_draft) {
        std::fprintf(stderr,
                     "--draft-ipc-bin requires --target-split-dflash or --target-split-load-draft\n");
        return 2;
    }
    std::printf("[cfg] seq_verify=%d fast_rollback=%d ddtree=%d budget=%d temp=%.2f chain_seed=%d fa_window=%d draft_swa=%d draft_ctx_max=%d draft_feature_mirror=%d peer_access=%d target_gpu=%d draft_gpu=%d\n",
                (int)seq_verify, (int)fast_rollback, (int)ddtree_mode,
                ddtree_budget, ddtree_temp, (int)ddtree_chain_seed, g_fa_window,
                g_draft_swa_window, g_draft_ctx_max, (int)draft_feature_mirror,
                (int)g_peer_access_opt_in, target_gpu, draft_gpu);
    if (draft_ipc_bin) {
        std::printf("[cfg] draft_ipc_bin=%s draft_ipc_gpu=%d draft_ipc_ring_cap=%d\n",
                    draft_ipc_bin, draft_ipc_gpu, draft_ipc_ring_cap);
    }

    int cuda_device_count = 0;
    cudaGetDeviceCount(&cuda_device_count);
    for (int gpu : target_gpus) {
        if (gpu >= cuda_device_count) {
            std::fprintf(stderr, "bad target gpu id %d device_count=%d\n",
                         gpu, cuda_device_count);
            return 2;
        }
    }
    if (target_gpu >= cuda_device_count ||
        (!draft_ipc_bin && draft_gpu >= cuda_device_count)) {
        std::fprintf(stderr, "bad gpu ids target=%d draft=%d device_count=%d\n",
                     target_gpu, draft_gpu, cuda_device_count);
        return 2;
    }
    if (target_gpus.size() > 1) {
        if (test_window_mode || profile_scaling) {
            std::fprintf(stderr, "--target-gpus path does not support test-window/profile-scaling modes\n");
            return 2;
        }
        if (daemon_mode) {
            dflash27b::Qwen35LayerSplitDaemonArgs lsargs;
            lsargs.target_path = target_path;
            lsargs.draft_path  = draft_path;
            lsargs.device.layer_split_gpus    = target_gpus;
            lsargs.device.layer_split_weights = target_split_weights;
            lsargs.device.peer_access         = g_peer_access_opt_in;
            lsargs.device.max_ctx = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;
            lsargs.draft_gpu   = draft_gpu;
            lsargs.load_draft  = target_split_load_draft;
            lsargs.run_dflash  = target_split_dflash;
            lsargs.max_verify_tokens = ddtree_mode
                ? std::max<int>(DFLASH27B_DRAFT_BLOCK_SIZE, ddtree_budget + 1)
                : DFLASH27B_DRAFT_BLOCK_SIZE;
            lsargs.stream_fd   = stream_fd;
            // TODO: migrate to run_qwen35_layer_split_daemon() once helpers
            // are extracted to src/qwen35/. For now, call the local function.
            return run_target_layer_split_daemon(
                lsargs.target_path, lsargs.draft_path,
                lsargs.device.layer_split_gpus,
                lsargs.device.layer_split_weights,
                lsargs.draft_gpu,
                lsargs.load_draft,
                lsargs.run_dflash,
                lsargs.device.max_ctx,
                lsargs.max_verify_tokens,
                lsargs.device.peer_access,
                lsargs.stream_fd);
        }
        if (target_split_dflash && fast_rollback) {
            std::fprintf(stderr,
                         "warning: --fast-rollback is not implemented for target layer-split DFlash; using replay rollback\n");
        }
        return run_target_layer_split_harness(target_path, draft_path, prompt_path, n_gen, out_path,
                                             target_gpus, target_split_weights,
                                             draft_gpu,
                                             target_split_load_draft,
                                             target_split_load_draft && !target_split_dflash,
                                             target_split_dflash,
                                             g_max_ctx_override > 0 ? g_max_ctx_override : 4096,
                                             ddtree_mode
                                                 ? std::max<int>(DFLASH27B_DRAFT_BLOCK_SIZE, ddtree_budget + 1)
                                                 : DFLASH27B_DRAFT_BLOCK_SIZE,
                                             g_peer_access_opt_in,
                                             draft_ipc_bin,
                                             draft_ipc_gpu,
                                             draft_ipc_work_dir,
                                             draft_ipc_ring_cap);
    }

    // ---- Single-GPU qwen35 daemon: dispatch to run_qwen35_daemon() -----
    // This avoids the duplicated 1800-line inline loop below. The inline
    // loop remains for one-shot, test-window, and profile-scaling modes.
    if (daemon_mode && target_gpus.size() <= 1) {
        const int max_ctx_eff = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;
        dflash27b::Qwen35DaemonArgs qargs;
        qargs.target_path       = target_path;
        qargs.draft_path        = draft_path;
        qargs.device.gpu        = target_gpu;
        qargs.device.max_ctx    = max_ctx_eff;
        qargs.draft_gpu         = draft_gpu;
        qargs.stream_fd         = stream_fd;
        qargs.chunk             = 512;
        qargs.fa_window         = g_fa_window;
        qargs.kq_stride_pad     = g_kq_stride_pad;
        qargs.draft_swa_window  = g_draft_swa_window;
        qargs.draft_ctx_max     = g_draft_ctx_max;
        qargs.fast_rollback     = fast_rollback;
        qargs.seq_verify        = seq_verify;
        qargs.ddtree_mode       = ddtree_mode;
        qargs.ddtree_budget     = ddtree_budget;
        qargs.ddtree_temp       = ddtree_temp;
        qargs.ddtree_chain_seed = ddtree_chain_seed;
        qargs.use_feature_mirror = draft_feature_mirror;
        std::fprintf(stderr,
            "[test_dflash] arch=qwen35 daemon -> dispatching to run_qwen35_daemon "
            "(max_ctx=%d stream_fd=%d)\n", max_ctx_eff, stream_fd);
        return dflash27b::run_qwen35_daemon(qargs);
    }

    const bool split_gpus = target_gpu != draft_gpu;
    ggml_backend_t target_backend = ggml_backend_cuda_init(target_gpu);
    if (!target_backend) { std::fprintf(stderr, "target cuda init failed\n"); return 1; }
    ggml_backend_t draft_backend = target_backend;
    if (split_gpus) {
        draft_backend = ggml_backend_cuda_init(draft_gpu);
        if (!draft_backend) { std::fprintf(stderr, "draft cuda init failed\n"); return 1; }
    }
    if (split_gpus && g_peer_access_opt_in) {
        if (!enable_peer_access_pair(target_gpu, draft_gpu)) {
            std::fprintf(stderr,
                         "warning: --peer-access requested but CUDA peer access could not be enabled "
                         "for target=%d draft=%d; using staged host copies.\n",
                         target_gpu, draft_gpu);
        }
    }
    ggml_backend_t backend = target_backend; // legacy target-side alias

    TargetWeights w;
    if (!load_target_gguf(target_path, target_backend, w)) {
        std::fprintf(stderr, "target load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    DraftWeights dw;
    {
        // Auto-detect draft format: .gguf → GGUF loader, else safetensors.
        std::string dp(draft_path);
        bool draft_ok = false;
        if (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf") {
            draft_ok = load_draft_gguf(draft_path, draft_backend, dw);
        } else {
            draft_ok = load_draft_safetensors(draft_path, draft_backend, dw);
        }
        if (!draft_ok) {
            std::fprintf(stderr, "draft load: %s\n", dflash27b_last_error());
            return 1;
        }
    }
    std::printf("[draft]  loaded\n");

    // Apply --draft-swa=N: mark layers 0..n-2 as SWA, last layer stays full.
    if (g_draft_swa_window > 0) {
        dw.swa_window = g_draft_swa_window;
        for (int il = 0; il < dw.n_layer - 1; il++) {
            dw.layers[il].is_swa = true;
        }
        std::printf("[draft]  SWA layers: %d/%d (window=%d)\n",
                    dw.n_layer - 1, dw.n_layer, dw.swa_window);
    }

    const int max_ctx = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;
    // Size the ssm_intermediate / conv_input_cache buffers to cover whichever
    // verify mode we'll use. DDTree needs room for 1 + ddtree_budget tree nodes.
    // Profile mode intentionally keeps the intermediate cache tiny (no capture)
    // so we can go up to n_tokens=128 without OOM.
    const int max_verify_tokens = profile_scaling
        ? DFLASH27B_DRAFT_BLOCK_SIZE
        : (ddtree_mode
            ? std::max<int>(DFLASH27B_DRAFT_BLOCK_SIZE, ddtree_budget + 1)
            : DFLASH27B_DRAFT_BLOCK_SIZE);
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend, cache,
                             /*prefill_only=*/true)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return 1;
    }

    // ── Profile mode: microbench target forward at varying N ───────────
    if (profile_scaling) {
        const int hidden_p = DFLASH27B_TARGET_HIDDEN;
        StepGraph psg;
        const int n_values[] = { 1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128 };
        std::printf("[profile] target forward ms at varying N (kv_start=0, no capture)\n");
        std::printf("%6s %10s %10s\n", "N", "total_ms", "ms_per_N");
        for (int n : n_values) {
            if (!build_target_step(psg, w, cache, backend,
                                   /*kv_start=*/0, /*n_tokens=*/n,
                                   /*with_mask=*/true,
                                   /*capture=*/false,
                                   /*capture_delta_intermediate=*/false,
                                   /*fa_window=*/0,
                                   /*last_token_logits_only=*/false,
                                   g_kq_stride_pad)) {
                std::fprintf(stderr, "profile build N=%d failed\n", n); return 1;
            }
            // Fake embed input (zeros) + fake positions + fake causal mask.
            std::vector<float> emb(hidden_p * n, 0.0f);
            ggml_backend_tensor_set(psg.inp_embed, emb.data(), 0, sizeof(float) * emb.size());
            std::vector<int32_t> pos4(4 * n);
            for (int i = 0; i < n; i++) {
                pos4[0 * n + i] = i;
                pos4[1 * n + i] = i;
                pos4[2 * n + i] = i;
                pos4[3 * n + i] = 0;
            }
            ggml_backend_tensor_set(psg.positions, pos4.data(), 0, sizeof(int32_t) * 4 * n);
            if (psg.attn_mask) {
                const int kv_pad = (int)psg.attn_mask->ne[0];
                const int q_pad  = (int)psg.attn_mask->ne[1];
                std::vector<uint16_t> mask_buf_p((size_t)kv_pad * q_pad, F16_NEG_INF);
                for (int q = 0; q < n; q++) {
                    for (int k = 0; k <= q; k++) {
                        mask_buf_p[(size_t)q * kv_pad + k] = F16_ZERO;
                    }
                }
                ggml_backend_tensor_set(psg.attn_mask, mask_buf_p.data(), 0,
                                        sizeof(uint16_t) * mask_buf_p.size());
            }
            // Warmup
            ggml_backend_graph_compute(backend, psg.gf);
            // Time 5 runs, take median
            std::vector<double> times;
            for (int rep = 0; rep < 5; rep++) {
                auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(backend, psg.gf);
                auto t1 = std::chrono::steady_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(times.begin(), times.end());
            double median = times[times.size() / 2];
            std::printf("%6d %10.2f %10.3f\n", n, median, median / n);
        }
        free_target_cache(cache);
        free_target_weights(w);
        if (split_gpus) ggml_backend_free(draft_backend);
        ggml_backend_free(target_backend);
        return 0;
    }

    // ── Sliding-window regression tests ──────────────────────────────────
    if (test_window_mode) {
        int n_pass = 0, n_fail = 0;
        auto check = [&](bool cond, const char * msg) {
            if (cond) { n_pass++; std::printf("  PASS  %s\n", msg); }
            else      { n_fail++; std::fprintf(stderr, "  FAIL  %s\n", msg); }
        };

        // ── Test 1: build_causal_mask unit tests (CPU, no GPU needed) ──
        std::printf("[test-window] === Test 1: build_causal_mask ===\n");
        {
            std::vector<uint16_t> buf;
            // 1a: standard causal mask, no window: 4 queries at positions 2-5, 6 KV
            build_causal_mask(buf, /*kv_len=*/6, /*n_tokens=*/4, /*kv_start=*/2, g_kq_stride_pad);
            const int pad = align_up(6, g_kq_stride_pad);
            // q=0 at pos 2: attend k=[0..2], 3 zeros
            // q=3 at pos 5: attend k=[0..5], 6 zeros
            check(buf[0 * pad + 0] == F16_ZERO, "1a: q=0,k=0 attendable");
            check(buf[0 * pad + 2] == F16_ZERO, "1a: q=0,k=2 attendable");
            check(buf[0 * pad + 3] == F16_NEG_INF, "1a: q=0,k=3 masked");
            check(buf[3 * pad + 5] == F16_ZERO, "1a: q=3,k=5 attendable");
            check(buf[3 * pad + 5] == F16_ZERO, "1a: q=3,k=5 attendable (diagonal)");
        }
        {
            std::vector<uint16_t> buf;
            // 1b: windowed mask: kv_start=10, n_tokens=3, win_start=8, win_len=5
            // Queries at positions 10,11,12. KV entries [8,9,10,11,12].
            build_causal_mask(buf, /*kv_len=*/5, /*n_tokens=*/3, /*kv_start=*/10, g_kq_stride_pad, /*win_start=*/8);
            const int pad = align_up(5, g_kq_stride_pad);
            // q=0 (pos 10): attend k=[8..10] → indices [0..2]
            check(buf[0 * pad + 0] == F16_ZERO, "1b: q=0,k_abs=8 attendable");
            check(buf[0 * pad + 2] == F16_ZERO, "1b: q=0,k_abs=10 attendable");
            check(buf[0 * pad + 3] == F16_NEG_INF, "1b: q=0,k_abs=11 masked (future)");
            // q=2 (pos 12): attend k=[8..12] → indices [0..4]
            check(buf[2 * pad + 4] == F16_ZERO, "1b: q=2,k_abs=12 attendable");
        }
        {
            std::vector<uint16_t> buf;
            // 1c: large window > kv_start (window inactive): kv_start=100, win_start=0
            build_causal_mask(buf, /*kv_len=*/106, /*n_tokens=*/6, /*kv_start=*/100, g_kq_stride_pad, /*win_start=*/0);
            const int pad = align_up(106, g_kq_stride_pad);
            // q=0 (pos 100): attend k=[0..100]
            check(buf[0 * pad + 0] == F16_ZERO, "1c: q=0,k=0 attendable (no window)");
            check(buf[0 * pad + 100] == F16_ZERO, "1c: q=0,k=100 attendable");
            check(buf[0 * pad + 101] == F16_NEG_INF, "1c: q=0,k=101 masked");
            // q=5 (pos 105): attend k=[0..105]
            check(buf[5 * pad + 105] == F16_ZERO, "1c: q=5,k=105 attendable");
        }

        // ── Tests 2 & 3: GPU regression tests ───────────────────────────
        const int hidden_t = DFLASH27B_TARGET_HIDDEN;
        const int vocab_t  = DFLASH27B_TARGET_VOCAB;
        auto do_prefill = [&](StepGraph & psg, int n_tokens) -> int32_t {
            const int pf_ub = 384;
            int32_t lt = -1;
            int committed_p = 0;
            std::vector<float> pf_emb;
            std::vector<int32_t> pf_pos;
            std::vector<uint16_t> pf_mask;
            std::vector<float> pf_logits;
            for (int start = 0; start < n_tokens; start += pf_ub) {
                const int nt = std::min(pf_ub, n_tokens - start);
                const int kv_len_p = start + nt;
                const bool with_m = (g_kq_stride_pad > KQ_MASK_PAD) || (nt > 1);
                if (!build_target_step(psg, w, cache, backend,
                                        start, nt, with_m, true,
                                        /*capture_delta_intermediate=*/false,
                                        /*fa_window=*/0,
                                        /*last_token_logits_only=*/false,
                                        g_kq_stride_pad)) {
                    std::fprintf(stderr, "prefill build @%d\n", start); return -1;
                }
                pf_emb.assign((size_t)hidden_t * nt, 0.0f);
                std::vector<int32_t> tokens(nt, 220);
                if (!w.embedder.embed(tokens.data(), nt, pf_emb.data())) return -1;
                ggml_backend_tensor_set(psg.inp_embed, pf_emb.data(), 0,
                                        sizeof(float) * pf_emb.size());
                pf_pos.assign((size_t)4 * nt, 0);
                for (int i = 0; i < nt; i++) {
                    const int p = start + i;
                    pf_pos[0 * nt + i] = p;
                    pf_pos[1 * nt + i] = p;
                    pf_pos[2 * nt + i] = p;
                    pf_pos[3 * nt + i] = 0;
                }
                ggml_backend_tensor_set(psg.positions, pf_pos.data(), 0,
                                        sizeof(int32_t) * pf_pos.size());
                if (with_m) {
                    build_causal_mask(pf_mask, kv_len_p, nt, start, g_kq_stride_pad);
                    ggml_backend_tensor_set(psg.attn_mask, pf_mask.data(), 0,
                                            sizeof(uint16_t) * pf_mask.size());
                }
                auto st = ggml_backend_graph_compute(backend, psg.gf);
                if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "prefill fail @%d\n", start); return -1; }
                pf_logits.assign(vocab_t, 0.0f);
                ggml_backend_tensor_get(psg.logits, pf_logits.data(),
                                        (size_t)(nt - 1) * vocab_t * sizeof(float),
                                        sizeof(float) * vocab_t);
                lt = argmax_f32(pf_logits.data(), vocab_t);
                committed_p = start + nt;
            }
            return lt;
        };

        auto decode_one = [&](StepGraph & dsg, int kv_start, int32_t tok,
                               int32_t pos, int fa_w, float * logits_out) -> bool {
            if (!build_target_step(dsg, w, cache, backend,
                                    kv_start, 1, false, true, false, fa_w,
                                    /*last_token_logits_only=*/false,
                                    g_kq_stride_pad)) {
                std::fprintf(stderr, "decode build failed\n"); return false;
            }
            float emb_buf[5120];
            if (!w.embedder.embed(&tok, 1, emb_buf)) return false;
            ggml_backend_tensor_set(dsg.inp_embed, emb_buf, 0, sizeof(float) * hidden_t);
            int32_t pos4[4] = {pos, pos, pos, 0};
            ggml_backend_tensor_set(dsg.positions, pos4, 0, sizeof(int32_t) * 4);
            auto st = ggml_backend_graph_compute(backend, dsg.gf);
            if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "decode compute failed\n"); return false; }
            ggml_backend_tensor_get(dsg.logits, logits_out, 0, sizeof(float) * vocab_t);
            return true;
        };

        auto cosine_sim = [](const float * a, const float * b, int n) -> double {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
            return (na < 1e-30 || nb < 1e-30) ? 0.0 : dot / (std::sqrt(na) * std::sqrt(nb));
        };

        // ── Test 2: Short context identity (window inactive) ────────────
        std::printf("[test-window] === Test 2: short-ctx identity (512 tokens) ===\n");
        {
            StepGraph psg2;
            int32_t lt2 = do_prefill(psg2, 512);
            check(lt2 >= 0, "prefill 512 tokens succeeded");

            // Need rollback tensors for snapshot/restore
            step_graph_free(psg2);
            psg2 = StepGraph{};
            migrate_prefill_cache(w, max_ctx, max_verify_tokens, target_backend, cache);

            snapshot_ssm_state(cache);
            std::vector<float> logits_full(vocab_t), logits_win(vocab_t);
            bool ok = decode_one(psg2, 512, lt2, 512, 0, logits_full.data());
            check(ok, "decode full-attention succeeded");
            restore_ssm_state(cache);
            ok = decode_one(psg2, 512, lt2, 512, 2048, logits_win.data());
            check(ok, "decode window=2048 succeeded");

            int tok_full = argmax_f32(logits_full.data(), vocab_t);
            int tok_win  = argmax_f32(logits_win.data(), vocab_t);
            check(tok_full == tok_win, "short-ctx: argmax matches");
            double cs = cosine_sim(logits_full.data(), logits_win.data(), vocab_t);
            char msg[128];
            std::snprintf(msg, sizeof(msg), "short-ctx: cosine_sim=%.8f (expect >0.9999)", cs);
            check(cs > 0.9999, msg);

            step_graph_free(psg2);
        }

        // Reset cache for next test
        free_target_cache(cache);
        if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend, cache, true)) {
            std::fprintf(stderr, "cache realloc failed\n"); return 1;
        }

        // ── Test 3: Long context argmax match ───────────────────────────
        std::printf("[test-window] === Test 3: long-ctx quality (4096 tokens) ===\n");
        {
            StepGraph psg3;
            int32_t lt3 = do_prefill(psg3, 4096);
            check(lt3 >= 0, "prefill 4096 tokens succeeded");

            step_graph_free(psg3);
            psg3 = StepGraph{};
            migrate_prefill_cache(w, max_ctx, max_verify_tokens, target_backend, cache);

            snapshot_ssm_state(cache);
            std::vector<float> logits_full(vocab_t), logits_win(vocab_t);
            bool ok = decode_one(psg3, 4096, lt3, 4096, 0, logits_full.data());
            check(ok, "decode full-attention succeeded");
            restore_ssm_state(cache);
            ok = decode_one(psg3, 4096, lt3, 4096, 1024, logits_win.data());
            check(ok, "decode window=1024 succeeded");

            int tok_full = argmax_f32(logits_full.data(), vocab_t);
            int tok_win  = argmax_f32(logits_win.data(), vocab_t);
            char msg[128];
            std::snprintf(msg, sizeof(msg), "long-ctx: argmax full=%d win=%d match", tok_full, tok_win);
            check(tok_full == tok_win, msg);
            double cs = cosine_sim(logits_full.data(), logits_win.data(), vocab_t);
            std::snprintf(msg, sizeof(msg), "long-ctx: cosine_sim=%.6f (expect >0.90)", cs);
            check(cs > 0.90, msg);

            step_graph_free(psg3);
        }

        std::printf("[test-window] === Results: %d passed, %d failed ===\n", n_pass, n_fail);
        free_target_cache(cache);
        free_target_weights(w);
        if (split_gpus) ggml_backend_free(draft_backend);
        ggml_backend_free(target_backend);
        return n_fail > 0 ? 1 : 0;
    }

    const int q_len  = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab  = DFLASH27B_TARGET_VOCAB;
    const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;

    if (daemon_mode) {
        std::printf("[daemon] ready\n");
        std::fflush(stdout);
    }

    constexpr int PREFIX_CACHE_SLOTS = 8;
    PrefixSnapshot prefix_snapshots[PREFIX_CACHE_SLOTS];   // default-constructed, ctx==nullptr

    StepGraph sg;
    StepGraph draft_sg;
    StepGraph proj_sg;
    DraftFeatureMirror feature_mirror;
    bool daemon_first_iter = true;
    bool target_parked = false;
    bool draft_parked  = false;
    // pflash drafter (lazy-loaded on first `compress` command)
    dflash27b::DrafterContext drafter_ctx;
    bool drafter_loaded = false;

    while (true) {
        std::string prompt_file_str;
        bool restore_from_slot        = false;
        int  restore_slot_id          = -1;
        bool chain_restore_requested  = false;
        int  chain_thick_slot         = -1;
        std::vector<int> chain_thin_ids;
        // Inline-snap: snapshot at boundary during prefill (single snap only;
        // multi-snap "snap=A:1,B:2" is not implemented — use separate SNAPSHOT).
        int  snap_pos  = -1;
        int  snap_slot = -1;

        if (daemon_mode) {
            std::string line;
            if (!std::getline(std::cin, line)) break;
            g_sampler = SamplerCfg{};
            if (parse_sampler_token(line, g_sampler) && g_sampler.seed != 0) {
                g_sampler_rng.seed(g_sampler.seed);
            }

            // ── Park/unpark commands (additive on top of latest daemon) ─────
            // "park draft" frees ~3.3GB, "park target" frees ~15GB,
            // "park all" or "park" frees both. ACK via stream_emit(-1).
            auto starts_with = [](const std::string& s, const char* pre) {
                size_t n = std::strlen(pre);
                return s.size() >= n && s.compare(0, n, pre) == 0;
            };
            if (starts_with(line, "park")) {
                bool want_draft  = (line == "park" || line == "park all" || line == "park draft");
                bool want_target = (line == "park" || line == "park all" || line == "park target");
                if (want_draft && !draft_parked) {
                    free_draft_weights(dw);
                    draft_parked = true;
                    std::printf("[park] draft released\n"); std::fflush(stdout);
                }
                if (want_target && !target_parked) {
                    step_graph_destroy(proj_sg);
                    free_target_weights(w);
                    target_parked = true;
                    std::printf("[park] target released\n"); std::fflush(stdout);
                }
                stream_emit(-1);
                continue;
            }
            if (line == "free drafter" || line == "drafter free") {
                if (drafter_loaded) {
                    dflash27b::free_drafter(drafter_ctx);
                    drafter_loaded = false;
                    std::printf("[drafter] freed\n"); std::fflush(stdout);
                }
                stream_emit(-1);
                continue;
            }
            if (starts_with(line, "unpark")) {
                bool want_draft  = (line == "unpark" || line == "unpark all" || line == "unpark draft");
                bool want_target = (line == "unpark" || line == "unpark all" || line == "unpark target");
                if (want_target && target_parked) {
                    if (!load_target_gguf(target_path, target_backend, w)) {
                        std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    target_parked = false;
                    std::printf("[unpark] target restored\n"); std::fflush(stdout);
                }
                if (want_draft && draft_parked) {
                    std::string dp(draft_path);
                    bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
                        ? load_draft_gguf(draft_path, draft_backend, dw)
                        : load_draft_safetensors(draft_path, draft_backend, dw);
                    if (!draft_ok) {
                        std::fprintf(stderr, "[unpark] draft: %s\n", dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    if (g_draft_swa_window > 0) {
                        dw.swa_window = g_draft_swa_window;
                        for (int il = 0; il < dw.n_layer - 1; il++)
                            dw.layers[il].is_swa = true;
                    }
                    draft_parked = false;
                    std::printf("[unpark] draft restored\n"); std::fflush(stdout);
                }
                stream_emit(-1);
                continue;
            }

            // ── Compress command (pflash speculative prefill) ───────────────
            // Format: "compress <src_bin_path> <keep_ratio_x1000> <drafter_gguf> [drafter_arch]"
            //   src_bin_path:   int32 token IDs file (drafter vocab)
            //   keep_ratio_x1000: integer keep ratio × 1000 (e.g. 20 → 0.020)
            //   drafter_gguf:   path to drafter GGUF (loaded lazily once)
            //   drafter_arch:   qwen3-0.6b (default) or qwen35-0.8b
            // Output: stream of int32 compressed token IDs, terminated by -1.
            // Drafter coexists with target+draft via libllama in the same
            // ggml allocator — no park/unpark needed for compression itself.
            if (starts_with(line, "compress ")) {
                char ppath[1024];
                int  keep_x1000 = 0;
                char drafter_path[1024];
                char arch_name[64] = "qwen3-0.6b";
                int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s %63s",
                                    ppath, &keep_x1000, drafter_path, arch_name);
                if (n < 3) {
                    std::fprintf(stderr,
                                 "[compress] bad args, need: <bin> <keep_x1000> <drafter_gguf> [drafter_arch]\n");
                    stream_emit(-1); continue;
                }
                dflash27b::DrafterArch drafter_arch;
                if (!dflash27b::parse_drafter_arch(arch_name, drafter_arch)) {
                    std::fprintf(stderr, "[compress] bad drafter_arch: %s\n", arch_name);
                    stream_emit(-1); continue;
                }
                auto src_ids = read_int32_file(ppath);
                if (src_ids.empty()) {
                    std::fprintf(stderr, "[compress] empty input\n");
                    stream_emit(-1); continue;
                }

                // Park target + draft before allocating drafter context so
                // the drafter's KV (~1.3 GB Q4_0) + scratch (~600 MB) have
                // headroom on a 24 GB card. Restore after scoring.
                // On >=32 GB GPUs, DFLASH_COMPRESS_NO_PARK=1 skips parking
                // so the scorer stays co-resident with target+draft.
                const bool no_park = (std::getenv("DFLASH_COMPRESS_NO_PARK") &&
                                      std::atoi(std::getenv("DFLASH_COMPRESS_NO_PARK")) != 0);
                bool restore_target = !target_parked && !no_park;
                bool restore_draft  = !draft_parked && !no_park;
                if (restore_target) {
                    step_graph_destroy(proj_sg);
                    free_target_weights(w);
                    target_parked = true;
                    std::printf("[compress] target parked\n"); std::fflush(stdout);
                }
                if (restore_draft) {
                    free_draft_weights(dw);
                    draft_parked = true;
                    std::printf("[compress] draft parked\n"); std::fflush(stdout);
                }

                if (!drafter_loaded) {
                    if (!dflash27b::load_drafter(drafter_path, /*gpu_layers=*/999, drafter_arch, drafter_ctx)) {
                        std::fprintf(stderr, "[compress] load_drafter failed: %s\n",
                                     dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    drafter_loaded = true;
                    if (drafter_arch == dflash27b::DrafterArch::Qwen3_0p6b) {
                        std::printf("[drafter] loaded %s arch=%s (n_layer=%d n_head=%d n_head_kv=%d)\n",
                                    drafter_path, dflash27b::drafter_arch_name(drafter_arch), drafter_ctx.weights.n_layer,
                                    drafter_ctx.weights.n_head, drafter_ctx.weights.n_head_kv);
                    } else {
                        std::printf("[drafter] loaded %s arch=%s\n",
                                    drafter_path, dflash27b::drafter_arch_name(drafter_arch));
                    }
                    std::fflush(stdout);
                } else if (drafter_ctx.arch != drafter_arch) {
                    std::fprintf(stderr, "[compress] requested arch=%s but loaded arch=%s\n",
                                 dflash27b::drafter_arch_name(drafter_arch),
                                 dflash27b::drafter_arch_name(drafter_ctx.arch));
                    stream_emit(-1); continue;
                }

                float keep = (float)keep_x1000 / 1000.0f;
                auto compressed = dflash27b::drafter_score_and_compress(
                    drafter_ctx, src_ids, keep);
                std::printf("[compress] %zu -> %zu tokens (keep_ratio=%.3f)\n",
                            src_ids.size(), compressed.size(), keep);
                std::fflush(stdout);

                // Restore daemon state for the (almost certainly) following
                // generate command.
                if (restore_target) {
                    if (!load_target_gguf(target_path, target_backend, w)) {
                        std::fprintf(stderr, "[compress] target restore: %s\n",
                                     dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    target_parked = false;
                    std::printf("[compress] target restored\n"); std::fflush(stdout);
                }
                if (restore_draft) {
                    if (!load_draft_safetensors(draft_path, draft_backend, dw)) {
                        std::fprintf(stderr, "[compress] draft restore: %s\n",
                                     dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    if (g_draft_swa_window > 0) {
                        dw.swa_window = g_draft_swa_window;
                        for (int il = 0; il < dw.n_layer - 1; il++)
                            dw.layers[il].is_swa = true;
                    }
                    draft_parked = false;
                    std::printf("[compress] draft restored\n"); std::fflush(stdout);
                }

                for (int32_t t : compressed) stream_emit(t);
                stream_emit(-1);
                continue;
            }

            // ── Prefix-cache snapshot commands (#59) ──────────────────────
            // Check longer prefixes before shorter ones to avoid mis-dispatch
            // (SNAPSHOT_THIN must come before SNAPSHOT, RESTORE_CHAIN before RESTORE).
            if (line.rfind("SNAPSHOT_THIN ", 0) == 0) {
                int slot = -1, kv_start = -1, kv_end = -1;
                if (std::sscanf(line.c_str() + 14, "%d %d %d", &slot, &kv_start, &kv_end) != 3
                    || slot < 0 || slot >= PREFIX_CACHE_SLOTS) {
                    std::fprintf(stderr, "[snap] SNAPSHOT_THIN bad args\n");
                    continue;
                }
                if (!snapshot_target_cache_thin(w, cache, backend, kv_start, kv_end,
                                                 prefix_snapshots[slot])) {
                    std::fprintf(stderr, "[snap] thin failed slot=%d: %s\n", slot,
                                 dflash27b_last_error());
                    continue;
                }
                std::printf("[snap] thin slot=%d kv=%d,%d\n", slot, kv_start, kv_end);
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("SNAPSHOT ", 0) == 0) {
                int slot = -1;
                if (std::sscanf(line.c_str() + 9, "%d", &slot) != 1
                    || slot < 0 || slot >= PREFIX_CACHE_SLOTS) {
                    std::fprintf(stderr, "[snap] invalid slot %d\n", slot);
                    continue;
                }
                if (!snapshot_target_cache(w, cache, backend, prefix_snapshots[slot])) {
                    std::fprintf(stderr, "[snap] failed slot=%d: %s\n", slot, dflash27b_last_error());
                    continue;
                }
                std::printf("[snap] slot=%d cur_pos=%d\n", slot, prefix_snapshots[slot].cur_pos);
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("FREE_SNAPSHOT ", 0) == 0) {
                int slot = -1;
                if (std::sscanf(line.c_str() + 14, "%d", &slot) != 1
                    || slot < 0 || slot >= PREFIX_CACHE_SLOTS) continue;
                free_prefix_snapshot(prefix_snapshots[slot]);
                std::printf("[snap] freed slot=%d\n", slot);
                std::fflush(stdout);
                continue;
            }
            if (line == "LIST_SLOTS") {
                std::printf("[snap] slots=");
                bool first = true;
                for (int i = 0; i < PREFIX_CACHE_SLOTS; i++) {
                    if (prefix_snapshots[i].ctx != nullptr) {
                        std::printf("%s%d", first ? "" : ",", i);
                        first = false;
                    }
                }
                std::printf("\n");
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("RESTORE_CHAIN ", 0) == 0) {
                // Format: RESTORE_CHAIN <thick_slot> <thin_slot_list> <prompt_file> <n_gen>
                // <thin_slot_list> is "0,1,2" or "-" for empty.
                int  thick_slot_local = -2;
                char thin_str[256]    = {0};
                char ppath[1024]      = {0};
                int  n_gen_local      = 0;
                if (std::sscanf(line.c_str() + 14, "%d %255s %1023s %d",
                                &thick_slot_local, thin_str, ppath, &n_gen_local) != 4) {
                    std::fprintf(stderr, "[snap] RESTORE_CHAIN bad args\n");
                    stream_emit(-1);
                    continue;
                }
                // Validate thick_slot (-1 = none).
                if (thick_slot_local != -1
                    && (thick_slot_local < 0 || thick_slot_local >= PREFIX_CACHE_SLOTS
                        || prefix_snapshots[thick_slot_local].ctx == nullptr
                        || prefix_snapshots[thick_slot_local].is_thin)) {
                    std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thick slot=%d\n", thick_slot_local);
                    stream_emit(-1);
                    continue;
                }
                // Parse thin slot list. Strict: every comma-separated token
                // must be a valid non-negative integer (rejects "1,foo,3",
                // empty entries "1,,3", trailing junk). Codex review fix.
                std::vector<int> thin_ids_local;
                bool thin_parse_ok = true;
                if (std::strcmp(thin_str, "-") != 0 && thin_str[0] != '\0') {
                    const char * p = thin_str;
                    while (*p && thin_parse_ok) {
                        char * end = nullptr;
                        long id_l = std::strtol(p, &end, 10);
                        if (end == p) {
                            std::fprintf(stderr,
                                "[snap] RESTORE_CHAIN malformed thin list near '%s'\n", p);
                            thin_parse_ok = false; break;
                        }
                        int id = (int)id_l;
                        if (id < 0 || id >= PREFIX_CACHE_SLOTS
                            || prefix_snapshots[id].ctx == nullptr
                            || !prefix_snapshots[id].is_thin) {
                            std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thin slot=%d\n", id);
                            thin_parse_ok = false; break;
                        }
                        thin_ids_local.push_back(id);
                        if (*end == '\0') break;
                        if (*end != ',') {
                            std::fprintf(stderr,
                                "[snap] RESTORE_CHAIN expected ',' after slot %d, got '%c'\n",
                                id, *end);
                            thin_parse_ok = false; break;
                        }
                        p = end + 1;
                        if (*p == '\0' || *p == ',') {
                            std::fprintf(stderr,
                                "[snap] RESTORE_CHAIN empty thin slot entry\n");
                            thin_parse_ok = false; break;
                        }
                    }
                }
                if (!thin_parse_ok) {
                    stream_emit(-1);
                    continue;
                }
                n_gen                    = n_gen_local;
                prompt_file_str          = ppath;
                prompt_path              = prompt_file_str.c_str();
                chain_restore_requested  = true;
                chain_thick_slot         = thick_slot_local;
                chain_thin_ids           = std::move(thin_ids_local);
                // Fall through into the existing cache-rebuild + prefill path.
            } else if (line.rfind("RESTORE ", 0) == 0) {
                int slot = -1;
                char ppath[1024];
                if (std::sscanf(line.c_str() + 8, "%d %1023s %d", &slot, ppath, &n_gen) != 3
                    || slot < 0 || slot >= PREFIX_CACHE_SLOTS
                    || prefix_snapshots[slot].ctx == nullptr) {
                    std::fprintf(stderr, "[snap] RESTORE bad args or empty slot %d\n", slot);
                    stream_emit(-1);
                    continue;
                }
                prompt_file_str = ppath;
                prompt_path = prompt_file_str.c_str();
                restore_from_slot = true;
                restore_slot_id   = slot;
                // Parse optional inline-snap suffix: snap=<pos>:<slot_id>
                if (const char * sp = std::strstr(line.c_str(), "snap=")) {
                    if (std::sscanf(sp, "snap=%d:%d", &snap_pos, &snap_slot) != 2
                        || snap_slot < 0 || snap_slot >= PREFIX_CACHE_SLOTS) {
                        std::fprintf(stderr, "[snap] bad inline-snap arg\n");
                        snap_pos = -1; snap_slot = -1;
                    }
                }
                // Fall through into the existing prefill path; the cache reset
                // and restore happen after the cache rebuild block below.
            } else {
                // Legacy: bare `<prompt_file> <n_gen>` line — full reset path.
                char ppath[1024];
                if (std::sscanf(line.c_str(), "%1023s %d", ppath, &n_gen) != 2) continue;
                prompt_file_str = ppath;
                prompt_path = prompt_file_str.c_str();
                // Parse optional inline-snap suffix: snap=<pos>:<slot_id>
                if (const char * sp = std::strstr(line.c_str(), "snap=")) {
                    if (std::sscanf(sp, "snap=%d:%d", &snap_pos, &snap_slot) != 2
                        || snap_slot < 0 || snap_slot >= PREFIX_CACHE_SLOTS) {
                        std::fprintf(stderr, "[snap] bad inline-snap arg\n");
                        snap_pos = -1; snap_slot = -1;
                    }
                }
            }

            // Reset cache state between requests. On the first request the
            // cache was promoted from prefill-only to full (with rollback
            // tensors) by migrate_prefill_cache. On subsequent requests we
            // just zero all state tensors in place — no GPU buffer free/alloc.
            if (!daemon_first_iter) {
                step_graph_free(sg);
                reset_target_cache(cache);
            }
            daemon_first_iter = false;

            // After cache is fresh, optionally restore from snapshot.
            if (restore_from_slot) {
                if (!restore_target_cache(prefix_snapshots[restore_slot_id], cache)) {
                    std::fprintf(stderr, "[snap] restore failed: %s\n", dflash27b_last_error());
                    stream_emit(-1);
                    continue;
                }
                std::printf("[snap] restored slot=%d cur_pos=%d\n",
                            restore_slot_id, cache.cur_pos);
                std::fflush(stdout);
            }

            // After cache is fresh, optionally apply chain restore.
            if (chain_restore_requested) {
                const PrefixSnapshot * thick_ptr =
                    (chain_thick_slot == -1) ? nullptr : &prefix_snapshots[chain_thick_slot];
                std::vector<const PrefixSnapshot *> thin_ptrs;
                for (int id : chain_thin_ids) thin_ptrs.push_back(&prefix_snapshots[id]);
                if (!restore_target_cache_chain(thick_ptr,
                                                 thin_ptrs.empty() ? nullptr : thin_ptrs.data(),
                                                 (int)thin_ptrs.size(),
                                                 cache)) {
                    std::fprintf(stderr, "[snap] RESTORE_CHAIN failed: %s\n", dflash27b_last_error());
                    stream_emit(-1);
                    continue;
                }
                std::printf("[snap] chain restored thick=%d thins=%zu cur_pos=%d\n",
                            chain_thick_slot, thin_ptrs.size(), cache.cur_pos);
                std::fflush(stdout);
            }
        }

        auto prompt = read_int32_file(prompt_path);
        if (prompt.empty()) {
            std::fprintf(stderr, "empty prompt\n");
            if (daemon_mode) { stream_emit(-1); continue; } else return 1;
        }
        std::printf("[prompt] %zu tokens\n", prompt.size());

        if ((int)prompt.size() + n_gen + q_len > max_ctx) {
            std::fprintf(stderr, "prompt (%zu) + gen (%d) + block (%d) = %d exceeds max_ctx (%d)\n",
                         prompt.size(), n_gen, q_len, (int)prompt.size() + n_gen + q_len, max_ctx);
            if (daemon_mode) { stream_emit(-1); continue; } else return 1;
        }

        std::vector<float>   embed_buf(hidden);
        std::vector<int32_t> out_all = prompt;
        int committed = 0;
        int32_t last_tok = -1;

    // ── Prefill: two modes available ────────────────────────────────────
    // Layer-segmented: iterate layers (outer) × token chunks (inner).
    //   Reads each layer's weights once per chunk instead of once per full
    //   forward. Better L2 cache warmth on weights across token chunks.
    // Token-segmented (legacy): iterate token chunks (outer) × layers (inner).
    //   Matches llama.cpp's n_ubatch behavior.
    // Controlled by DFLASH27B_LAYER_PREFILL=1 env var (default: off).
    // Currently faster only at short contexts (<8K); at longer contexts the
    // graph rebuild overhead per layer dominates.
    const int prompt_len_auto = (int)prompt.size();
    bool layer_prefill = false;
    if (const char * s = std::getenv("DFLASH27B_LAYER_PREFILL")) {
        layer_prefill = (std::atoi(s) != 0);
    }

    // ── Layer-segmented prefill ─────────────────────────────────────────
    if (layer_prefill) {
        int layer_ubatch_env = 384;
        if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
            layer_ubatch_env = std::max(1, std::atoi(s));
        }
        const int LAYER_UBATCH = layer_ubatch_env;
        std::printf("[prefill] layer-segmented ubatch=%d\n", LAYER_UBATCH);
        const int prompt_len = (int)prompt.size();

        // Allocate ping-pong activation buffers [hidden, prompt_len]
        ggml_init_params act_ip{};
        act_ip.mem_size   = (size_t)4 * ggml_tensor_overhead();
        act_ip.mem_buffer = nullptr;
        act_ip.no_alloc   = true;
        ggml_context * act_ctx = ggml_init(act_ip);
        ggml_tensor * act_in  = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, prompt_len);
        ggml_set_name(act_in, "act_in");
        ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, prompt_len);
        ggml_set_name(act_out, "act_out");
        ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, backend);
        if (!act_buf) {
            std::fprintf(stderr, "activation buffer alloc failed\n"); return 1;
        }

        // Embed all prompt tokens into act_in (batched)
        {
            const int EMBED_BATCH = 4096;
            std::vector<float> emb_buf((size_t)hidden * EMBED_BATCH);
            for (int i = 0; i < prompt_len; i += EMBED_BATCH) {
                const int n = std::min(EMBED_BATCH, prompt_len - i);
                if (!w.embedder.embed(prompt.data() + i, n, emb_buf.data())) return 1;
                ggml_backend_tensor_set(act_in, emb_buf.data(),
                                        (size_t)i * act_in->nb[1],
                                        sizeof(float) * hidden * n);
            }
        }

        auto t_pf0 = std::chrono::steady_clock::now();
        StepGraph lsg;

        for (int il = 0; il < w.n_layer; il++) {
            const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);

            for (int start = 0; start < prompt_len; start += LAYER_UBATCH) {
                const int n_tokens = std::min(LAYER_UBATCH, prompt_len - start);
                const int kv_len   = start + n_tokens;
                const bool with_mask = (g_kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

                if (!build_layer_step(lsg, w, cache, backend, il,
                                      act_in, act_out, start, n_tokens,
                                      start, with_mask, true,
                                      /*fa_window=*/0, g_kq_stride_pad)) {
                    std::fprintf(stderr, "layer-seg build layer=%d @%d\n", il, start);
                    return 1;
                }

                // M-RoPE positions for this chunk (FA layers only)
                if (is_attn && lsg.positions) {
                    std::vector<int32_t> pos_buf((size_t)4 * n_tokens, 0);
                    for (int i = 0; i < n_tokens; i++) {
                        const int p = start + i;
                        pos_buf[0 * n_tokens + i] = p;
                        pos_buf[1 * n_tokens + i] = p;
                        pos_buf[2 * n_tokens + i] = p;
                        pos_buf[3 * n_tokens + i] = 0;
                    }
                    ggml_backend_tensor_set(lsg.positions, pos_buf.data(), 0,
                                            sizeof(int32_t) * pos_buf.size());
                }

                if (is_attn && with_mask && lsg.attn_mask) {
                    std::vector<uint16_t> mask_buf;
                    build_causal_mask(mask_buf, kv_len, n_tokens, /*kv_start=*/start, g_kq_stride_pad);
                    ggml_backend_tensor_set(lsg.attn_mask, mask_buf.data(), 0,
                                            sizeof(uint16_t) * mask_buf.size());
                }

                auto st = ggml_backend_graph_compute(backend, lsg.gf);
                if (st != GGML_STATUS_SUCCESS) {
                    std::fprintf(stderr, "layer-seg compute layer=%d @%d\n", il, start);
                    return 1;
                }
            }

            // Swap activation buffers after each layer
            std::swap(act_in, act_out);
        }

        // Final norm + LM head on last token only
        {
            step_graph_free(lsg);
            ggml_init_params fip{};
            fip.mem_size   = 512 * 1024 * 1024;
            fip.mem_buffer = nullptr;
            fip.no_alloc   = true;
            lsg.ctx = ggml_init(fip);

            ggml_tensor * last_row = ggml_view_1d(lsg.ctx, act_in,
                hidden, (size_t)(prompt_len - 1) * act_in->nb[1]);
            ggml_tensor * normed   = ggml_rms_norm(lsg.ctx, last_row, DFLASH27B_RMS_EPS);
            normed = ggml_mul(lsg.ctx, normed, w.out_norm);
            ggml_tensor * logits   = ggml_mul_mat(lsg.ctx, w.output, normed);
            ggml_set_name(logits, "logits");
            ggml_set_output(logits);
            lsg.logits = logits;
            lsg.gf = ggml_new_graph_custom(lsg.ctx, 1024, false);
            ggml_build_forward_expand(lsg.gf, logits);

            if (!lsg.alloc) {
                lsg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            }
            if (!ggml_gallocr_alloc_graph(lsg.alloc, lsg.gf)) {
                std::fprintf(stderr, "final norm alloc failed\n"); return 1;
            }

            auto st = ggml_backend_graph_compute(backend, lsg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "final norm compute failed\n"); return 1;
            }

            std::vector<float> logits_buf(vocab, 0.0f);
            ggml_backend_tensor_get(lsg.logits, logits_buf.data(), 0,
                                    sizeof(float) * vocab);
            last_tok = (g_sampler.temp > 0.0f)
                ? sample_logits(logits_buf.data(), vocab, g_sampler, out_all, g_sampler_rng)
                : argmax_f32(logits_buf.data(), vocab);
            step_graph_destroy(lsg);
        }

        committed = prompt_len;
        ggml_backend_buffer_free(act_buf);
        ggml_free(act_ctx);

        auto t_pf1 = std::chrono::steady_clock::now();
        std::printf("[prefill] layer-seg %d tokens in %.2f s, last_tok=%d\n",
                    committed,
                    std::chrono::duration<double>(t_pf1 - t_pf0).count(),
                    last_tok);

        // Promote prefill-only cache to full decode cache
        auto t_mig0 = std::chrono::steady_clock::now();
        step_graph_destroy(sg);
        if (!migrate_prefill_cache(w, max_ctx, max_verify_tokens, target_backend, cache)) {
            std::fprintf(stderr, "cache migration: %s\n", dflash27b_last_error());
            return 1;
        }
        auto t_mig1 = std::chrono::steady_clock::now();
        std::printf("[migrate] %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t_mig1 - t_mig0).count());
    }
    // ── Token-segmented prefill (legacy) ────────────────────────────────
    if (!layer_prefill) {
    // Prefill only needs last-token logits to seed decode. Skip computing
    // the full [vocab, ubatch] lm_head matmul — saves ~233MB scratch at
    // ubatch=384 and eliminates a large matmul per prefill step.
    // Daemon target_prefill batch. The previous (>2048 ? 384 : 16) default was
    // tuned for raw long prompts; the PFlash compress path hands us 200-2000
    // already-compressed tokens where ubatch=16 leaves a 2-3x wall-clock win on
    // the table (measured 12.4s -> 5.2s at 1205 tokens on gfx1151). Bumping
    // both branches: large prompts already amortise launch overhead, small
    // prompts (compressed) need a meaningful tile to keep the GPU busy.
    int prefill_ubatch_env = (prompt_len_auto > 2048) ? 512 : 256;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        prefill_ubatch_env = std::max(1, std::atoi(s));
    }
    const int PREFILL_UBATCH = prefill_ubatch_env;
    std::printf("[prefill] token-seg ubatch=%d\n", PREFILL_UBATCH);
    auto t_pf0 = std::chrono::steady_clock::now();
    std::vector<uint16_t> pf_mask_buf;
    std::vector<float>    pf_embed_buf;
    std::vector<int32_t>  pf_pos_buf;
    std::vector<float>    pf_logits_buf;
    const int prompt_len     = (int)prompt.size();
    const int prefill_start  = cache.cur_pos;   // 0 for fresh cache; >0 after snapshot restore

    // Pre-reserve gallocr: build a max-size graph so gallocr allocates its
    // buffer upfront, preventing reallocations as the mask grows during prefill.
    // With fa_window, the mask is capped at ~fa_window+ubatch regardless of
    // prompt length, so the reserve is always small.
    if (prompt_len > PREFILL_UBATCH) {
        // Use kv_start near the end so the mask reaches its maximum windowed size.
        const int reserve_kv = std::max(prompt_len - PREFILL_UBATCH, PREFILL_UBATCH);
        if (!build_target_step(sg, w, cache, backend,
                                /*kv_start=*/reserve_kv,
                                /*n_tokens=*/PREFILL_UBATCH,
                                /*with_mask=*/true, /*capture=*/true,
                                /*capture_delta_intermediate=*/false,
                                /*fa_window=*/g_fa_window,
                                /*last_token_logits_only=*/true,
                                g_kq_stride_pad)) {
            // Issue #114: gallocr OOM. Free all prefix snapshots so the next
            // request has VRAM headroom; abort this request cleanly in daemon
            // mode instead of killing the process.
            std::fprintf(stderr, "prefill gallocr pre-reserve failed (OOM)\n");
            for (int _i = 0; _i < PREFIX_CACHE_SLOTS; _i++) free_prefix_snapshot(prefix_snapshots[_i]);
            std::printf("[snap] all-cleared\n"); std::fflush(stdout);
            if (daemon_mode) { stream_emit(-1); continue; } else return 1;
        }
        // gallocr is now reserved at peak size; subsequent builds will reuse it.
    }

    for (int start = prefill_start; start < prompt_len; start += PREFILL_UBATCH) {
        int n_tokens = std::min(PREFILL_UBATCH, prompt_len - start);

        // Inline-snap: if snap_pos == start exactly, fire snapshot before any
        // prefill work this iteration, then continue with the full ubatch.
        if (snap_pos >= 0 && snap_pos == start) {
            cache.cur_pos = start;
            if (snap_slot >= 0) {
                if (snapshot_target_cache(w, cache, backend, prefix_snapshots[snap_slot])) {
                    std::printf("[snap] inline slot=%d cur_pos=%d\n", snap_slot, start);
                    std::fflush(stdout);
                } else {
                    std::fprintf(stderr, "[snap] inline snap failed slot=%d: %s\n",
                                 snap_slot, dflash27b_last_error());
                }
            }
            snap_pos = -1; snap_slot = -1;   // consume
            // n_tokens is unchanged; continue prefilling this ubatch.
        }

        // Inline-snap: if snap_pos falls inside this ubatch, clip n_tokens to
        // land exactly at snap_pos so the snapshot captures the right boundary.
        bool fire_snap_after = false;
        if (snap_pos > start && snap_pos <= start + n_tokens) {
            n_tokens = snap_pos - start;   // land exactly at snap_pos
            fire_snap_after = (n_tokens > 0);
            if (n_tokens == 0) {
                // snap_pos == start already handled above; shouldn't reach here.
                snap_pos = -1; snap_slot = -1;
            }
        }

        const int kv_len   = start + n_tokens;
        const bool pf_with_mask = (g_kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);
        if (!build_target_step(sg, w, cache, backend,
                                /*kv_start=*/start, /*n_tokens=*/n_tokens,
                                /*with_mask=*/pf_with_mask, /*capture=*/true,
                                /*capture_delta_intermediate=*/false,
                                /*fa_window=*/g_fa_window,
                                /*last_token_logits_only=*/true,
                                g_kq_stride_pad)) {
            std::fprintf(stderr, "prefill build @%d failed (OOM)\n", start);
            for (int _i = 0; _i < PREFIX_CACHE_SLOTS; _i++) free_prefix_snapshot(prefix_snapshots[_i]);
            std::printf("[snap] all-cleared\n"); std::fflush(stdout);
            if (daemon_mode) { stream_emit(-1); goto _req_aborted_oom; } else return 1;
        }

        pf_embed_buf.assign((size_t)hidden * n_tokens, 0.0f);
        if (!w.embedder.embed(prompt.data() + start, n_tokens, pf_embed_buf.data())) {
            std::fprintf(stderr, "prefill embed @%d failed\n", start);
            if (daemon_mode) { stream_emit(-1); goto _req_aborted_oom; } else return 1;
        }
        ggml_backend_tensor_set(sg.inp_embed, pf_embed_buf.data(), 0,
                                sizeof(float) * pf_embed_buf.size());

        // M-RoPE 4D text layout: [axis0 × n_tokens, axis1 × n_tokens,
        // axis2 × n_tokens, axis3 × n_tokens]. First 3 axes hold absolute
        // positions, axis 3 is 0 for plain text.
        pf_pos_buf.assign((size_t)4 * n_tokens, 0);
        for (int i = 0; i < n_tokens; i++) {
            const int p = start + i;
            pf_pos_buf[0 * n_tokens + i] = p;
            pf_pos_buf[1 * n_tokens + i] = p;
            pf_pos_buf[2 * n_tokens + i] = p;
            pf_pos_buf[3 * n_tokens + i] = 0;
        }
        ggml_backend_tensor_set(sg.positions, pf_pos_buf.data(), 0,
                                sizeof(int32_t) * pf_pos_buf.size());

        // Causal mask required when n_tokens > 1 OR when the TBQ FA kernel
        // is active (which pads kv_len to 256 and needs -inf on the padding
        // positions even for a single query).
        if (pf_with_mask) {
            const int pf_win_start = (g_fa_window > 0 && start > g_fa_window)
                                         ? (start - g_fa_window) : 0;
            const int pf_win_len = kv_len - pf_win_start;
            build_causal_mask(pf_mask_buf, pf_win_len, n_tokens,
                              /*kv_start=*/start, g_kq_stride_pad, /*win_start=*/pf_win_start);
            ggml_backend_tensor_set(sg.attn_mask, pf_mask_buf.data(), 0,
                                    sizeof(uint16_t) * pf_mask_buf.size());
        }

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "prefill compute @%d failed (OOM)\n", start);
            for (int _i = 0; _i < PREFIX_CACHE_SLOTS; _i++) free_prefix_snapshot(prefix_snapshots[_i]);
            std::printf("[snap] all-cleared\n"); std::fflush(stdout);
            if (daemon_mode) { stream_emit(-1); goto _req_aborted_oom; } else return 1;
        }

        // Logits are [vocab, 1] (last_token_logits_only), read from offset 0.
        pf_logits_buf.assign(vocab, 0.0f);
        ggml_backend_tensor_get(sg.logits, pf_logits_buf.data(), 0,
                                sizeof(float) * vocab);
        last_tok = (g_sampler.temp > 0.0f)
            ? sample_logits(pf_logits_buf.data(), vocab, g_sampler, out_all, g_sampler_rng)
            : argmax_f32(pf_logits_buf.data(), vocab);
        committed = start + n_tokens;

        // Fire inline snapshot after compute, so cache boundary is exact.
        if (fire_snap_after) {
            cache.cur_pos  = committed;
            cache.last_tok = last_tok;
            if (snap_slot >= 0) {
                if (snapshot_target_cache(w, cache, backend, prefix_snapshots[snap_slot])) {
                    std::printf("[snap] inline slot=%d cur_pos=%d\n", snap_slot, committed);
                    std::fflush(stdout);
                } else {
                    std::fprintf(stderr, "[snap] inline snap failed slot=%d: %s\n",
                                 snap_slot, dflash27b_last_error());
                }
            }
            snap_pos = -1; snap_slot = -1;   // consume
            // Adjust loop increment: next iteration must start at committed,
            // not at (start + PREFILL_UBATCH). Override via start arithmetic:
            // the for-loop does start += PREFILL_UBATCH, so back-adjust.
            start = committed - PREFILL_UBATCH;
        }
    }
    // Issue #114: in-loop prefill OOM lands here, then re-enters the daemon
    // read loop without killing the process. No-op when daemon_mode=false.
    if (false) {
    _req_aborted_oom:
        continue;
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    // If prefill was a no-op due to a snapshot RESTORE (cache.cur_pos already
    // covers the prompt), seed last_tok from the restored cache so the decode
    // loop has a valid starting token. Detected by prefill_start == prompt_len:
    // the for loop ran zero iterations and `committed` stayed at 0.
    if (last_tok == -1 && cache.last_tok != -1 && prefill_start == prompt_len) {
        last_tok  = cache.last_tok;
        committed = prompt_len;
    }
    std::printf("[prefill] %d tokens in %.2f s, last_tok=%d\n",
                committed,
                std::chrono::duration<double>(t_pf1 - t_pf0).count(),
                last_tok);

    // Promote prefill-only cache to full decode cache with rollback tensors.
    // Copies KV, SSM/conv state, and target_feat device→device (~1 ms).
    auto t_mig0 = std::chrono::steady_clock::now();
    step_graph_destroy(sg);
    if (!migrate_prefill_cache(w, max_ctx, max_verify_tokens, target_backend, cache)) {
        std::fprintf(stderr, "cache migration: %s\n", dflash27b_last_error());
        return 1;
    }
    auto t_mig1 = std::chrono::steady_clock::now();
    std::printf("[migrate] %.2f ms\n",
                std::chrono::duration<double, std::milli>(t_mig1 - t_mig0).count());
    } // end if (!layer_prefill)

    if (draft_feature_mirror) {
        if (!feature_mirror.target_feat || feature_mirror.cap != cache.target_feat_cap) {
            if (!draft_feature_mirror_init(feature_mirror, draft_backend,
                                           draft_gpu, target_gpu,
                                           cache.target_feat_cap,
                                           DFLASH27B_DRAFT_N_TARGET_LAYERS,
                                           DFLASH27B_TARGET_HIDDEN)) {
                std::fprintf(stderr, "draft feature mirror init failed\n");
                return 1;
            }
            std::printf("[draft-mirror] init cap=%d type=f32 device=%d target_device=%d\n",
                        feature_mirror.cap, draft_gpu, target_gpu);
        }
        if (!draft_feature_mirror_sync_tail(cache.target_feat, cache.target_feat_cap,
                                            feature_mirror, committed)) {
            std::fprintf(stderr, "draft feature mirror initial sync failed\n");
            return 1;
        }
        std::printf("[draft-mirror] synced tail committed=%d cap=%d\n",
                    committed, feature_mirror.cap);
    }

    // ── DFlash decode loop
    int n_draft_steps = 0, n_accept_sum = 0, n_generated = 0;
    std::vector<float>   noise_embed_buf(hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len), target_tok(q_len);
    std::vector<float>   draft_logits_buf((size_t)vocab * q_len);
    // Sized for the max of chain q_len and DDTree flat tree size (budget+1).
    const int verify_max_tokens = std::max(q_len, ddtree_mode ? ddtree_budget + 1 : q_len);
    std::vector<float>   verify_logits_buf((size_t)vocab * verify_max_tokens);
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos_q_buf(q_len), pos_k_buf(max_ctx + q_len);
    std::vector<int32_t> pos4_buf(4 * q_len);

    auto t_gen0 = std::chrono::steady_clock::now();

    // Per-phase timing accumulators (microseconds)
    double tt_draft_build = 0, tt_draft_copy_feat = 0, tt_draft_set = 0,
           tt_draft_compute = 0, tt_draft_bridge = 0, tt_draft_logits = 0,
           tt_snap = 0, tt_verify_build = 0, tt_verify_set = 0,
           tt_verify_compute = 0, tt_verify_logits = 0,
           tt_accept = 0, tt_restore = 0,
           tt_replay_build = 0, tt_replay_set = 0, tt_replay_compute = 0,
           tt_replay_logits = 0, tt_mirror_sync = 0;
    auto sync_us = [&](){
        ggml_backend_synchronize(target_backend);
        if (split_gpus) ggml_backend_synchronize(draft_backend);
        return std::chrono::steady_clock::now();
    };
    auto sync_draft_feature_mirror = [&](int start_pos, int n_tokens) -> bool {
        if (!draft_feature_mirror || !feature_mirror.target_feat || n_tokens <= 0) {
            return true;
        }
        auto t0 = sync_us();
        const bool ok = draft_feature_mirror_sync_range(cache.target_feat, cache.target_feat_cap,
                                                        feature_mirror,
                                                        start_pos, n_tokens);
        auto t1 = sync_us();
        tt_mirror_sync += std::chrono::duration<double, std::micro>(t1 - t0).count();
        return ok;
    };

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        auto T0 = sync_us();

        // 1) Noise block [last_tok, MASK*15]
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = mask_tok;
        if (!w.embedder.embed(noise_ids.data(), q_len, noise_embed_buf.data())) return 1;

        // Draft target-attention window. The draft transformer attends over
        // a slice of the history captured in cache.target_feat; this caps the
        // slice so the draft's [5*hidden × ctx_len] target_hidden_cat tensor
        // stays bounded even at 16K+ target context. Tokens older than the
        // window are invisible to the draft but still in the target's KV
        // cache (the target verify uses the full history).
        constexpr int DRAFT_CTX_MAX = 2048;
        const int draft_ctx   = std::min(committed,
            std::max(DRAFT_CTX_MAX, g_draft_ctx_max));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_mirror, committed, draft_ctx, mirror_slot0);
        const bool draft_hidden_bridge = split_gpus;

        // 2) Draft forward
        if (!build_draft_step(draft_sg, dw,
                              draft_hidden_bridge ? nullptr : w.output,
                              draft_backend, /*ctx_len=*/draft_ctx,
                              use_mirror_view ? &feature_mirror : nullptr,
                              committed)) {
            std::fprintf(stderr, "draft build failed\n"); return 1;
        }
        auto T_draft_build = sync_us();
        tt_draft_build += std::chrono::duration<double, std::micro>(T_draft_build - T0).count();

        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed_buf.data(), 0,
                                sizeof(float) * noise_embed_buf.size());

        if (!use_mirror_view) {
            if (draft_feature_mirror) {
                // Mirror ring is on the draft device; never read cache.target_feat (target VRAM)
                // from the draft device when the ggml view is not contiguous.
                if (!copy_feature_ring_range_to_tensor(feature_mirror, draft_sg.target_hidden_cat,
                                                       draft_start, draft_ctx)) {
                    std::fprintf(stderr, "draft mirror ring copy to target_hidden_cat failed\n");
                    return 1;
                }
            } else {
                // target_hidden_cat: widen BF16 cache.target_feat into draft-side F32.
                // Same GPU: widen in place on the draft CUDA device.
                // Split GPU: read BF16 rows from target via backend_get, convert on CPU,
                // upload F32 to draft (no P2P required).
                const size_t fc_in    = (size_t)5 * hidden;
                const int    cap      = cache.target_feat_cap;
                const size_t elt_feat = ggml_element_size(cache.target_feat);
                const size_t row_bf16 = fc_in * elt_feat;
                const int    slot0    = draft_start % cap;
                const int    pre_n    = std::min(draft_ctx, cap - slot0);
                const int    post_n   = draft_ctx - pre_n;

                if (!split_gpus) {
                    cudaSetDevice(draft_gpu);
                    auto bf16_to_f32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
                    bf16_to_f32(
                        (const char *)cache.target_feat->data + (size_t)slot0 * row_bf16,
                        (float *)draft_sg.target_hidden_cat->data,
                        (int64_t)pre_n * fc_in,
                        nullptr);
                    if (post_n > 0) {
                        bf16_to_f32(
                            (const char *)cache.target_feat->data,
                            (float *)((char *)draft_sg.target_hidden_cat->data +
                                      (size_t)pre_n * fc_in * sizeof(float)),
                            (int64_t)post_n * fc_in,
                            nullptr);
                    }
                } else {
                    std::vector<uint16_t> bf16_lin((size_t)draft_ctx * fc_in);
                    for (int i = 0; i < pre_n; i++) {
                        const int    slot = slot0 + i;
                        const size_t off  = (size_t)slot * row_bf16;
                        ggml_backend_tensor_get(cache.target_feat,
                                                bf16_lin.data() + (size_t)i * fc_in,
                                                off, row_bf16);
                    }
                    for (int j = 0; j < post_n; j++) {
                        const size_t off = (size_t)j * row_bf16;
                        ggml_backend_tensor_get(cache.target_feat,
                                                bf16_lin.data() + (size_t)(pre_n + j) * fc_in,
                                                off, row_bf16);
                    }
                    std::vector<float> f32_lin((size_t)draft_ctx * fc_in);
                    for (size_t k = 0; k < bf16_lin.size(); k++) {
                        uint32_t bits = (uint32_t)bf16_lin[k] << 16;
                        float    f;
                        std::memcpy(&f, &bits, sizeof(f));
                        f32_lin[k] = f;
                    }
                    ggml_backend_tensor_set(draft_sg.target_hidden_cat, f32_lin.data(), 0,
                                            f32_lin.size() * sizeof(float));
                }
            }
        }
        auto T_draft_copy = sync_us();
        tt_draft_copy_feat += std::chrono::duration<double, std::micro>(T_draft_copy - T_draft_build).count();

        for (int i = 0; i < q_len; i++) pos_q_buf[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k_buf[i] = i;
        ggml_backend_tensor_set(draft_sg.positions,   pos_q_buf.data(), 0, sizeof(int32_t) * q_len);
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k_buf.data(), 0, sizeof(int32_t) * (draft_ctx + q_len));
        auto T_draft_set = sync_us();
        tt_draft_set += std::chrono::duration<double, std::micro>(T_draft_set - T_draft_copy).count();

        auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "draft compute %d\n", (int)st); return 1; }
        auto T_draft_compute = sync_us();
        tt_draft_compute += std::chrono::duration<double, std::micro>(T_draft_compute - T_draft_set).count();

        if (draft_hidden_bridge) {
            if (!proj_sg.gf || !proj_sg.hidden_input ||
                proj_sg.hidden_input->ne[1] != q_len) {
                if (!build_lm_head_projection_step(proj_sg, w, target_backend, q_len)) {
                    std::fprintf(stderr, "draft lm-head projection build failed\n");
                    return 1;
                }
            }
            if (!proj_sg.hidden_input || !proj_sg.logits) {
                std::fprintf(stderr, "draft lm-head projection build failed\n");
                return 1;
            }
            const size_t hidden_bytes = ggml_nbytes(draft_sg.hidden_states);
            if (!copy_peer_async(proj_sg.hidden_input->data, target_gpu,
                                 draft_sg.hidden_states->data, draft_gpu,
                                 hidden_bytes)) {
                std::fprintf(stderr, "draft hidden peer copy failed\n");
                return 1;
            }
            cudaSetDevice(target_gpu);
            cudaDeviceSynchronize();
            auto st_proj = ggml_backend_graph_compute(target_backend, proj_sg.gf);
            if (st_proj != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "draft lm-head projection compute %d\n", (int)st_proj);
                return 1;
            }
            ggml_backend_tensor_get(proj_sg.logits, draft_logits_buf.data(), 0,
                                    sizeof(float) * vocab * q_len);
        }
        auto T_draft_bridge = sync_us();
        tt_draft_bridge += std::chrono::duration<double, std::micro>(T_draft_bridge - T_draft_compute).count();

        // DDTree top-K: use GPU argmax for draft_tok; full logits transfer
        // only when DDTree needs top-K (K>1) for sibling expansion.
        const int ddtree_K = (ddtree_budget > q_len - 1) ? 8 : 1;

        if (draft_hidden_bridge) {
            for (int i = 0; i < q_len; i++) {
                draft_tok[i] = argmax_f32(draft_logits_buf.data() + (size_t)i * vocab, vocab);
            }
        } else {
            std::vector<int32_t> gpu_argmax(q_len);
            ggml_backend_tensor_get(draft_sg.argmax_tokens, gpu_argmax.data(), 0,
                                    sizeof(int32_t) * q_len);
            for (int i = 0; i < q_len; i++) draft_tok[i] = gpu_argmax[i];
        }
        // The block-diffusion draft is free to "denoise" position 0 even though
        // the input there is the unmasked last_tok. Pin it back so verify and
        // replay see the correct prefix.
        draft_tok[0] = last_tok;

        // DDTree top-K extraction. Positions 1..q_len-1 of the draft output
        // are the per-position distributions for the block-diffusion tree
        // (position 0 is the root/bonus slot and is fixed to last_tok). Only
        // computed in ddtree_mode to keep the argmax fast path untouched.
        // ddtree_K controls how many candidates per position are available as
        // tree siblings. Budget <= L means pure chain → no siblings needed, so
        // we can skip the O(L*vocab) top-K extract entirely and just fill rank 0
        // from draft_tok. For larger budgets we need real top-K.
        static std::vector<float>   ddtree_top_log_probs; // [L × K]
        static std::vector<int32_t> ddtree_top_token_ids; // [L × K]
        if (ddtree_mode) {
            const int L = q_len - 1;
            if ((int)ddtree_top_log_probs.size() < L * ddtree_K) {
                ddtree_top_log_probs.assign((size_t)L * ddtree_K, 0.0f);
                ddtree_top_token_ids.assign((size_t)L * ddtree_K, 0);
            }
            if (ddtree_K == 1) {
                // Fast path: draft_tok already holds the top-1 per position.
                // Skip extract; log-probs are irrelevant for pure chain build.
                for (int i = 0; i < L; i++) {
                    ddtree_top_log_probs[i] = 0.0f;
                    ddtree_top_token_ids[i] = draft_tok[i + 1];  // +1 to skip slot 0
                }
            } else {
                // DDTree K>1: need real log-probs for best-first tree scoring.
                // Transfer full logits for positions 1..q_len-1.
                if (!draft_hidden_bridge) {
                    ggml_backend_tensor_get(draft_sg.logits, draft_logits_buf.data(), 0,
                                            sizeof(float) * vocab * q_len);
                }
                extract_draft_topk(draft_logits_buf.data() + (size_t)vocab,
                                   L, vocab, ddtree_K,
                                   ddtree_top_log_probs.data(),
                                   ddtree_top_token_ids.data(),
                                   ddtree_temp);
            }
        }
        auto T_draft_logits = sync_us();
        tt_draft_logits += std::chrono::duration<double, std::micro>(T_draft_logits - T_draft_bridge).count();

        // 3) Snapshot SSM state (skipped in fast_rollback mode: the patched
        //    gated_delta_net kernel captures per-step intermediate states, so
        //    we don't need a pre-verify snapshot to restore from).
        if (!fast_rollback) {
            snapshot_ssm_state(cache);
        }
        auto T_snap = sync_us();
        tt_snap += std::chrono::duration<double, std::micro>(T_snap - T_draft_logits).count();

        // 4) Target verify on draft tokens.
        //
        // Two paths, toggled by --seq-verify:
        //   - Batched (default): one target forward over q_len tokens with a causal
        //     mask. Fast but suspected of numerical divergence vs stepwise decode
        //     per z-lab issue #57 ("batched greedy verification diverges from
        //     stepwise baseline").
        //   - Sequential: q_len independent single-token decodes. Slow but
        //     bit-equivalent to what the target would produce during plain
        //     autoregressive decode — the ground truth for greedy spec decoding.
        //
        // In both paths we set capture=true so if commit_n == q_len we can skip
        // the replay entirely. For commit_n < q_len, replay overwrites target_feat
        // at [committed..committed+commit_n-1]; positions past that are stale but
        // never read by the next iteration's draft.

        auto T_verify_build = T_snap;
        auto T_verify_set = T_snap;
        auto T_verify_compute = T_snap;

        // ── DDTree path: tree-structured verify + walk + rollback ─────────
        //
        // Structure of one DDTree round (ported from liranringel/ddtree.py):
        //   1. Build tree from draft top-K via best-first heap (Algorithm 1)
        //   2. Flatten tree in DFS order: slot 0 = root (= last_tok), slots
        //      1..n_nodes = tree nodes. Positions = committed + depth.
        //   3. Build an ancestor-only attention mask + parent_ids array.
        //   4. Run target forward via build_target_step_tree (our kernel mod
        //      handles DeltaNet/SSM tree recurrence via parent_ids).
        //   5. Walk the tree from root following target.argmax; the matched
        //      path is the accepted prefix. First unmatched target token
        //      becomes the next round's bonus (implicit via last_tok).
        //   6. Rollback: SSM state ← cache.ssm_intermediate[last_accepted_dfs_idx]
        //      KV cache ← cudaMemcpy the accepted DFS-order slots to slots 0..k-1
        //      conv_state ← use the (depth-1)-th slot of cache.conv_input_cache
        if (ddtree_mode) {
            const int L = q_len - 1;
            DDTree tree = build_ddtree(
                ddtree_top_log_probs.data(),
                ddtree_top_token_ids.data(),
                L, ddtree_K, ddtree_budget,
                ddtree_chain_seed);

            const int N_actual = 1 + tree.n_nodes;  // actual tree size
            const int N = ddtree_budget + 1;         // fixed allocation size for gallocr reuse

            if (!build_target_step_tree(sg, w, cache, backend,
                                        /*kv_start=*/committed, /*n_tokens=*/N,
                                        g_fa_window, g_kq_stride_pad)) {
                std::fprintf(stderr, "ddtree verify build failed\n"); return 1;
            }
            T_verify_build = sync_us();
            tt_verify_build += std::chrono::duration<double, std::micro>(T_verify_build - T_snap).count();

            // Embeddings: [last_tok, tree.token_ids[0..n_nodes-1], padding...]
            std::vector<int32_t> flat_tokens(N, 0);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; i++) flat_tokens[1 + i] = tree.token_ids[i];
            // Pad remaining slots with token 0 — their outputs are masked

            std::vector<float> tree_embed((size_t)hidden * N, 0.0f);
            if (!w.embedder.embed(flat_tokens.data(), N_actual, tree_embed.data())) return 1;
            // Leave padding slots as zero
            ggml_backend_tensor_set(sg.inp_embed, tree_embed.data(), 0,
                                    sizeof(float) * hidden * N);

            // M-RoPE axis-major positions
            std::vector<int32_t> pos4(4 * N, 0);
            for (int i = 0; i < N_actual; i++) {
                int p = committed + (i == 0 ? 0 : tree.depths[i - 1]);
                pos4[0 * N + i] = p;
                pos4[1 * N + i] = p;
                pos4[2 * N + i] = p;
                pos4[3 * N + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, pos4.data(), 0, sizeof(int32_t) * 4 * N);

            // Ancestor-only attention mask (f16). Build for the full N slots
            // but only the first N_actual have valid visibility; padding slots
            // get -inf everywhere (default from assign).
            const int tree_win_start = (g_fa_window > 0 && committed > g_fa_window)
                                           ? (committed - g_fa_window) : 0;
            {
                // Use the same kv_pad as the tensor allocation (max_ctx + N)
                const int max_win_len = cache.max_ctx + N;
                const int kv_pad_m = align_up(max_win_len, g_kq_stride_pad);
                const int q_pad_m  = align_up(N, KQ_MASK_PAD);
                mask_buf.assign((size_t)kv_pad_m * q_pad_m, F16_NEG_INF);
                // Fill rows 0..N_actual-1 using the tree visibility
                for (int q = 0; q < N_actual; q++) {
                    // Past KV positions are visible to all tree nodes
                    for (int k = std::max(0, tree_win_start); k < committed; k++) {
                        mask_buf[(size_t)q * kv_pad_m + (k - tree_win_start)] = F16_ZERO;
                    }
                    // Tree self-visibility
                    for (int j = 0; j < N_actual; j++) {
                        if (tree.visibility[(size_t)q * N_actual + j]) {
                            mask_buf[(size_t)q * kv_pad_m + (committed + j - tree_win_start)] = F16_ZERO;
                        }
                    }
                }
                // Rows N_actual..N-1 remain all -inf (padding slots see nothing)
            }
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());

            // parent_ids: actual tree nodes, then padding → point to root (slot 0)
            std::vector<int32_t> parent_ids(N, 0);
            parent_ids[0] = -1;
            for (int i = 1; i < N_actual; i++) parent_ids[i] = (int32_t)tree.parents[i];
            // Padding slots: parent=0 (root). DeltaNet kernel processes them
            // but their outputs are never used (masked out in attention).
            ggml_backend_tensor_set(sg.parent_ids, parent_ids.data(), 0,
                                    sizeof(int32_t) * N);

            T_verify_set = sync_us();
            tt_verify_set += std::chrono::duration<double, std::micro>(T_verify_set - T_verify_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "ddtree verify compute %d\n", (int)st); return 1;
            }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_verify_set).count();

            // Read only the actual tree slots (not padding)
            std::vector<int32_t> posterior(N_actual);
            ggml_backend_tensor_get(sg.argmax_tokens, posterior.data(), 0,
                                    sizeof(int32_t) * N_actual);

            // Walk tree: accepted DFS indices and next bonus token.
            int next_token = -1;
            int bonus_node_idx = 0;
            std::vector<int> accepted = follow_verified_tree(tree, posterior.data(), next_token, &bonus_node_idx);
            if (g_sampler.temp > 0.0f) {
                std::vector<float> bonus_logits(vocab);
                ggml_backend_tensor_get(sg.logits, bonus_logits.data(),
                                        (size_t)bonus_node_idx * sg.logits->nb[1],
                                        (size_t)vocab * sizeof(float));
                next_token = sample_logits(bonus_logits.data(), vocab, g_sampler, out_all, g_sampler_rng);
            }
            const int accept_depth = (int)accepted.size();  // includes root

            // Detect when the walk takes a sibling branch (accepted node
            // whose DFS index is OUTSIDE the chain spine [0..L]).
            bool walked_sibling = false;
            for (int x : accepted) {
                if (x > L) { walked_sibling = true; break; }
            }
            if (walked_sibling || n_draft_steps < 2) {
                std::printf("[dbg sib step %d] N=%d accept=%d walked_sib=%d\n",
                            n_draft_steps, N_actual, accept_depth, walked_sibling ? 1 : 0);
                std::printf("  walk:");
                for (int x : accepted) std::printf(" %d", x);
                if (walked_sibling) {
                    std::printf("\n  sibling info:");
                    for (int i = L; i < tree.n_nodes; i++) {
                        std::printf(" [%d:d%d:p%d:%d]",
                            i + 1, tree.depths[i], tree.parents[i + 1], tree.token_ids[i]);
                    }
                }
                std::printf("\n");
            }


            std::printf("[step %d] committed=%d last_tok=%d tree_N=%d accept=%d next=%d\n",
                        n_draft_steps, committed, last_tok, N_actual, accept_depth, next_token);

            // Commit count: matches chain mode's accept_n semantics. The root
            // (= previous iter's last_tok) is "pending" — not yet in out_all —
            // and gets committed here along with each accepted child token.
            // next_token (target's correction at the deepest accepted node)
            // becomes the new last_tok, pending for the next iter.
            int commit_n = accept_depth;  // root + accepted children
            if (commit_n > need_commit_budget) commit_n = need_commit_budget;

            // Push the accepted path's tokens to out_all. The root token is
            // last_tok (the pending token from the previous iter). Each
            // subsequent accepted node contributes its own tree.token_ids
            // entry (dfs_idx - 1 because flat slot 0 = root, slot 1..N-1 =
            // tree.token_ids[0..n_nodes-1]).
            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                const int dfs_idx = accepted[i];
                const int32_t tok = (dfs_idx == 0)
                    ? last_tok
                    : tree.token_ids[dfs_idx - 1];
                out_all.push_back(tok); stream_emit(tok);
                if (IS_EOS_TOK(tok, w)) hit_eos = true;
            }
            last_tok = next_token;

            auto T_accept = sync_us();
            tt_accept += std::chrono::duration<double, std::micro>(T_accept - T_verify_compute).count();

            if (hit_eos) break;

            // Rollback: per-layer DeltaNet SSM and conv state + KV compaction
            // for full-attention layers.
            //
            // SSM: the kernel wrote intermediate[i] for each flat-tree token i
            // (i = DFS index, 0 = root). We want the state AFTER processing
            // all `commit_n` tokens we just committed, i.e. the state after
            // the last committed DFS node = accepted[commit_n - 1]. For the
            // common case commit_n == accept_depth, this is the deepest
            // accepted node.
            const int rollback_dfs = (commit_n > 0)
                ? accepted[commit_n - 1]
                : 0;
            // Fast path detection: pure-chain walk has accepted[i] == i for
            // every i. Used by rollback to skip the parent-chain gather.
            bool walked_sibling_for_rollback = false;
            for (int i = 0; i < commit_n; i++) {
                if (accepted[i] != i) { walked_sibling_for_rollback = true; break; }
            }

            {
                const int n_delta = (int)sg.delta_captures.size();
                cudaStream_t stream = nullptr;
                for (int il = 0; il < n_delta; il++) {
                    const DeltaNetCapture & cap = sg.delta_captures[il];
                    if (!cap.ssm_intermediate_states || !cap.conv_input) {
                        std::fprintf(stderr, "ddtree rollback: missing capture layer %d\n", il);
                        return 1;
                    }
                    // SSM state rollback: source is cache.ssm_intermediate_states
                    // ([S_v, S_v, H_v, max_verify_tokens]) at slot rollback_dfs.
                    // Destination is cache.ssm_state[il] (f32). Use ggml's
                    // built-in dequantize to widen Q8_0/F16 → f32.
                    const size_t ssm_elems =
                        (size_t)cache.ssm_state[il]->ne[0] *
                        (size_t)cache.ssm_state[il]->ne[1] *
                        (size_t)cache.ssm_state[il]->ne[2];
                    const size_t ssm_src_offset =
                        (size_t)rollback_dfs * cap.ssm_intermediate_states->nb[3];
                    const void * ssm_src =
                        (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
                    ggml_get_to_fp32_cuda(cap.ssm_intermediate_states->type)(
                        ssm_src, (float *)cache.ssm_state[il]->data,
                        (int64_t)ssm_elems, stream);
                    cudaError_t ce = cudaSuccess;  // launch error checked in the conv block below

                    // Conv rollback: copy the K-1 most recent inputs along
                    // the rolled-back token's ANCESTRY (not DFS order). Two
                    // paths:
                    //   - Pure chain accept (walked_sibling == false): the
                    //     conv window is 3 contiguous slots in conv_input, so
                    //     a single cudaMemcpy2DAsync handles it. Hot path.
                    //   - Sibling accept: scattered slots, fall back to K-1
                    //     individual column copies via parent-chain walk.
                    const int K_conv = 4;
                    const int row_cnt = (int)cap.conv_input->ne[1];
                    const size_t elt = ggml_element_size(cap.conv_input);
                    const size_t dpitch = (K_conv - 1) * elt;
                    const size_t spitch = cap.conv_input->nb[1];
                    if (!walked_sibling_for_rollback) {
                        // Fast path: 3 contiguous slots ending at rollback_dfs.
                        const int conv_off = rollback_dfs + 1;
                        const void * conv_src =
                            (const char *)cap.conv_input->data + (size_t)conv_off * elt;
                        ce = cudaMemcpy2DAsync(cache.conv_state[il]->data, dpitch,
                                               conv_src, spitch,
                                               (K_conv - 1) * elt, row_cnt,
                                               cudaMemcpyDeviceToDevice, stream);
                        if (ce != cudaSuccess) {
                            std::fprintf(stderr, "ddtree conv fast il=%d: %s\n",
                                         il, cudaGetErrorString(ce));
                            return 1;
                        }
                    } else {
                        int virt[K_conv - 1];
                        virt[K_conv - 2] = rollback_dfs;
                        for (int k = K_conv - 3; k >= 0; k--) {
                            const int prev = virt[k + 1];
                            virt[k] = (prev >= 0) ? (int)tree.parents[prev] : (prev - 1);
                        }
                        for (int k = 0; k < K_conv - 1; k++) {
                            const int sx_slot = (K_conv - 1) + virt[k];
                            const void * src_col =
                                (const char *)cap.conv_input->data + (size_t)sx_slot * elt;
                            char * dst_col =
                                (char *)cache.conv_state[il]->data + (size_t)k * elt;
                            ce = cudaMemcpy2DAsync(dst_col, dpitch,
                                                   src_col, spitch,
                                                   elt, row_cnt,
                                                   cudaMemcpyDeviceToDevice, stream);
                            if (ce != cudaSuccess) {
                                std::fprintf(stderr, "ddtree conv col il=%d k=%d: %s\n",
                                             il, k, cudaGetErrorString(ce));
                                return 1;
                            }
                        }
                    }
                }

                // target_feat compaction: written in DFS order during verify
                // (column kv_start+i = dfs slot i's features). Same logic as
                // KV cache: when accepted[d] != d, copy the accepted DFS slot's
                // features to the spine slot at d so next iter's draft reads
                // the right history. Position→slot uses `% target_feat_cap`
                // to account for the ring buffer.
                if (cache.target_feat) {
                    const size_t elt = ggml_element_size(cache.target_feat);
                    const int    fc_in = (int)cache.target_feat->ne[0];  // 5*hidden
                    const size_t col_stride = cache.target_feat->nb[1];
                    const int    tcap = cache.target_feat_cap;
                    for (int d = 1; d < commit_n; d++) {
                        const int src_dfs = accepted[d];
                        if (src_dfs == d) continue;
                        const int    src_slot = (committed + src_dfs) % tcap;
                        const int    dst_slot = (committed + d)       % tcap;
                        const size_t src_off  = (size_t)src_slot * col_stride;
                        const size_t dst_off  = (size_t)dst_slot * col_stride;
                        cudaMemcpyAsync((char *)cache.target_feat->data + dst_off,
                                        (const char *)cache.target_feat->data + src_off,
                                        (size_t)fc_in * elt,
                                        cudaMemcpyDeviceToDevice, stream);
                    }
                }

                // Full-attention KV compaction: the verify wrote K/V at slots
                // [committed..committed+N-1] in DFS tree order (slot 0 = root).
                // For the next iter's verify to see the correct committed
                // prefix, slots [committed..committed+commit_n-1] must hold
                // the K/V of the accepted path's committed tokens. For each
                // committed position d in 0..commit_n-1, the source K/V is at
                // DFS slot accepted[d]. d==0 is always the root (DFS slot 0),
                // trivially aligned. For d>=1, copy if accepted[d] != d.
                const int n_full_attn = (int)cache.attn_k.size();
                for (int d = 0; d < commit_n; d++) {
                    const int src_dfs = accepted[d];
                    const int dst_slot = d;
                    if (src_dfs == dst_slot) continue;  // already aligned
                    for (int l = 0; l < n_full_attn; l++) {
                        // Each slot: head_dim * n_kv floats in f16 per tensor.
                        ggml_tensor * ck = cache.attn_k[l];
                        ggml_tensor * cv = cache.attn_v[l];
                        const size_t slot_bytes = ck->nb[1];  // stride between slots
                        const size_t src_off = (size_t)(committed + src_dfs) * slot_bytes;
                        const size_t dst_off = (size_t)(committed + dst_slot) * slot_bytes;
                        // Per-head-kv layout: shape [head_dim, max_ctx, n_head_kv].
                        // nb[2] is distance between heads; we copy one slot's
                        // slice per head. For simplicity, do a 2D copy across
                        // the head dimension.
                        const int n_kv = (int)ck->ne[2];
                        for (int h = 0; h < n_kv; h++) {
                            const size_t head_src = src_off + (size_t)h * ck->nb[2];
                            const size_t head_dst = dst_off + (size_t)h * ck->nb[2];
                            cudaMemcpyAsync((char *)ck->data + head_dst,
                                            (const char *)ck->data + head_src,
                                            slot_bytes, cudaMemcpyDeviceToDevice, stream);
                            cudaMemcpyAsync((char *)cv->data + head_dst,
                                            (const char *)cv->data + head_src,
                                            slot_bytes, cudaMemcpyDeviceToDevice, stream);
                        }
                    }
                }
                // No explicit sync: stream==nullptr (default stream) serializes
                // these copies before the next iter's draft/verify kernels.
                // The CPU returns immediately and the next iter's CPU work
                // (graph build, embed) can overlap with the GPU compaction.
            }

            if (!sync_draft_feature_mirror(committed, commit_n)) {
                std::fprintf(stderr, "draft feature mirror sync failed after ddtree commit\n");
                return 1;
            }

            committed    += commit_n;
            n_generated  += commit_n;
            n_accept_sum += commit_n;  // for stats
            n_draft_steps++;
            continue;  // skip the rest of the verify/commit logic for this iter
        }

        if (!seq_verify) {
            const int verify_fa_window = g_fa_window;
            if (!build_target_step(sg, w, cache, backend,
                                    /*kv_start=*/committed, /*n_tokens=*/q_len,
                                    /*with_mask=*/true, /*capture=*/true,
                                    /*capture_delta_intermediate=*/fast_rollback,
                                    verify_fa_window,
                                    /*last_token_logits_only=*/false,
                                    g_kq_stride_pad)) {
                std::fprintf(stderr, "verify build failed\n"); return 1;
            }
            T_verify_build = sync_us();
            tt_verify_build += std::chrono::duration<double, std::micro>(T_verify_build - T_snap).count();

            std::vector<float> verify_embed(hidden * q_len);
            if (!w.embedder.embed(draft_tok.data(), q_len, verify_embed.data())) return 1;
            ggml_backend_tensor_set(sg.inp_embed, verify_embed.data(), 0,
                                    sizeof(float) * verify_embed.size());

            // M-RoPE axis-major layout: [axis0_tok0..axis0_tokN-1, axis1_..., axis2_..., axis3_...].
            // First 3 axes hold the token position; axis 3 is always 0 for text.
            for (int i = 0; i < q_len; i++) {
                int p = committed + i;
                pos4_buf[0 * q_len + i] = p;
                pos4_buf[1 * q_len + i] = p;
                pos4_buf[2 * q_len + i] = p;
                pos4_buf[3 * q_len + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, pos4_buf.data(), 0, sizeof(int32_t) * 4 * q_len);

            {
                const int win_start_v = (verify_fa_window > 0 && committed > verify_fa_window)
                                            ? (committed - verify_fa_window) : 0;
                const int win_len_v = committed + q_len - win_start_v;
                build_causal_mask(mask_buf, win_len_v, q_len, committed, g_kq_stride_pad, win_start_v);
            }
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0, sizeof(uint16_t) * mask_buf.size());
            T_verify_set = sync_us();
            tt_verify_set += std::chrono::duration<double, std::micro>(T_verify_set - T_verify_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "verify compute %d\n", (int)st); return 1; }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_verify_set).count();

            ggml_backend_tensor_get(sg.argmax_tokens, target_tok.data(), 0,
                                    sizeof(int32_t) * q_len);
        } else {
            // Sequential verify: q_len independent single-token decodes.
            // Each call writes K/V at slot committed+i and advances SSM by 1.
            // After the loop, target cache state is identical to the batched
            // path's state (both end at committed+q_len). Restore/replay below
            // still apply correctly.
            std::vector<float> single_embed(hidden);
            int32_t p4_single[4];
            for (int i = 0; i < q_len; i++) {
                if (!build_target_step(sg, w, cache, backend,
                                        /*kv_start=*/committed + i, /*n_tokens=*/1,
                                        /*with_mask=*/false, /*capture=*/true,
                                        /*capture_delta_intermediate=*/false,
                                        /*fa_window=*/0,
                                        /*last_token_logits_only=*/false,
                                        g_kq_stride_pad)) {
                    std::fprintf(stderr, "seq verify build %d failed\n", i); return 1;
                }
                int32_t t = draft_tok[i];
                if (!w.embedder.embed(&t, 1, single_embed.data())) return 1;
                ggml_backend_tensor_set(sg.inp_embed, single_embed.data(), 0,
                                        sizeof(float) * hidden);
                int p = committed + i;
                p4_single[0] = p; p4_single[1] = p; p4_single[2] = p; p4_single[3] = 0;
                ggml_backend_tensor_set(sg.positions, p4_single, 0, sizeof(int32_t) * 4);

                st = ggml_backend_graph_compute(backend, sg.gf);
                if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "seq verify compute %d at %d\n", (int)st, i); return 1; }

                ggml_backend_tensor_get(sg.logits,
                                        verify_logits_buf.data() + (size_t)i * vocab,
                                        0, sizeof(float) * vocab);
                target_tok[i] = argmax_f32(verify_logits_buf.data() + (size_t)i * vocab, vocab);
            }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_snap).count();
        }
        auto T_verify_logits = sync_us();
        tt_verify_logits += std::chrono::duration<double, std::micro>(T_verify_logits - T_verify_compute).count();

        std::printf("[step %d] committed=%d last_tok=%d\n", n_draft_steps, committed, last_tok);

        // 5) Greedy longest-prefix accept with standard spec-decoding comparison.
        //
        //   - draft_tok[0] should equal last_tok (the correct first token from the
        //     previous forward). Accept it unconditionally.
        //   - target_tok[i] = argmax(logit at position committed+i) = target's
        //     prediction for the token AT position committed+i+1 (given draft_tok[0..i]).
        //   - So the check is: draft_tok[i+1] == target_tok[i], for i=0..q_len-2.
        //   - First mismatch at i=k → accept draft_tok[0..k] (k+1 tokens),
        //     bonus = target_tok[k] (the correct replacement for draft_tok[k+1]).

        int accept_n = 1;  // draft_tok[0] assumed = last_tok
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        // Two commit strategies:
        //   - Legacy (replay path): commit_n = accept_n + 1, the extra is the
        //     "bonus" token (target's correction or target_tok[q_len-1] when
        //     all accepted). Requires a replay forward pass to advance state.
        //   - Fast-rollback path: commit_n = accept_n, no explicit bonus. Use
        //     verify_logits[accept_n-1] as next last_tok; the "bonus" becomes
        //     draft_tok[0] of the next iter and is accepted unconditionally.
        //     Identical output stream, one fewer commit per iter tallied, but
        //     no extra forward pass needed.
        int bonus_tok = -1;
        int commit_n;
        if (fast_rollback) {
            commit_n = accept_n;
        } else {
            if (accept_n < q_len) {
                bonus_tok = target_tok[accept_n - 1];
            }
            commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        }
        std::printf("[step %d] accept_n=%d bonus=%d commit_n=%d\n",
                    n_draft_steps, accept_n, bonus_tok, commit_n);

        // Don't overshoot n_gen
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            // If we were going to add the bonus but budget is tight, drop it.
            if (commit_n <= accept_n) bonus_tok = -1;
        }
        auto T_accept = sync_us();
        tt_accept += std::chrono::duration<double, std::micro>(T_accept - T_verify_logits).count();

        // 6) Rollback and commit.
        //
        // Fast-rollback path: no replay. Use the per-step SSM intermediate states
        // captured during verify to roll back DeltaNet state, and slice the conv
        // input tensor for the conv state. Next last_tok comes from verify's
        // logits at position (accept_n - 1) — the target's prediction at position
        // committed+accept_n given the accepted prefix. The implicit bonus
        // becomes the next iter's draft_tok[0].
        //
        // Legacy path: restore SSM state from snapshot and run the replay.
        double t_rollback_us = 0, t_replay_build_us = 0, t_replay_set_us = 0;
        double t_replay_compute_us = 0, t_replay_logits_us = 0;

        if (fast_rollback) {
            auto T_rb0 = sync_us();

            // Rollback SSM + conv state unless we fully accepted (in which case
            // state after processing all q_len tokens is exactly what we want).
            if (commit_n < q_len) {
                const int rollback_idx = commit_n - 1;  // index into per-step intermediates
                // Temporary ctx for view tensors (no data alloc — views inherit
                // data pointers from their already-live sources).
                ggml_init_params tp{};
                tp.mem_size   = 1024 * 1024;
                tp.mem_buffer = nullptr;
                tp.no_alloc   = true;
                ggml_context * tmp_ctx = ggml_init(tp);
                if (!tmp_ctx) { std::fprintf(stderr, "rollback ctx init failed\n"); return 1; }

                const int n_delta = (int)sg.delta_captures.size();
                cudaStream_t stream = nullptr;  // use default stream
                for (int il = 0; il < n_delta; il++) {
                    const DeltaNetCapture & cap = sg.delta_captures[il];
                    if (!cap.ssm_intermediate_states || !cap.conv_input) {
                        std::fprintf(stderr, "rollback: missing capture at layer %d\n", il);
                        return 1;
                    }

                    // ── SSM rollback: copy intermediate[rollback_idx] → cache.ssm_state[il]
                    //
                    // cap.ssm_intermediate_states is the persistent cache buffer
                    // cache.ssm_intermediate[il], shape [S_v, S_v, H_v, q_len].
                    // Stored in Q8_0 (or F16 legacy) to reduce memory;
                    // cache.ssm_state[il] is f32. Use ggml's built-in dequantize
                    // to convert on copy, same as the DDtree rollback path.
                    const size_t ssm_elems =
                        (size_t)cache.ssm_state[il]->ne[0] *
                        (size_t)cache.ssm_state[il]->ne[1] *
                        (size_t)cache.ssm_state[il]->ne[2];
                    const size_t ssm_src_offset =
                        (size_t)rollback_idx * cap.ssm_intermediate_states->nb[3];
                    const void * ssm_src =
                        (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
                    ggml_get_to_fp32_cuda(cap.ssm_intermediate_states->type)(
                        ssm_src, (float *)cache.ssm_state[il]->data,
                        (int64_t)ssm_elems, stream);
                    cudaError_t ce = cudaSuccess;

                    // ── Conv rollback: copy conv_input[commit_n..commit_n+K-2, :, :]
                    //    into cache.conv_state[il].
                    //
                    // conv_input shape: [kernel-1 + n_tokens, conv_channels, 1]
                    //   nb[0] = elt, nb[1] = (kernel-1+n_tokens)*elt
                    // conv_state shape: [kernel-1, conv_channels, 1]
                    //   nb[0] = elt, nb[1] = (kernel-1)*elt
                    //
                    // Need cudaMemcpy2D because the source has a larger row stride
                    // (spans kernel-1+n_tokens values along dim 0) than the dest.
                    const int K_conv = 4;                            // qwen3.5 DeltaNet conv kernel
                    const int row_cnt = (int)cap.conv_input->ne[1];  // conv_channels (10240)
                    const size_t elt = ggml_element_size(cap.conv_input);
                    const size_t dpitch = (K_conv - 1) * elt;        // 12 bytes
                    const size_t spitch = cap.conv_input->nb[1];     // (K-1+n_tokens)*elt
                    const size_t width  = (K_conv - 1) * elt;        // copy 3 floats per row
                    const void * conv_src =
                        (const char *)cap.conv_input->data + commit_n * elt;
                    ce = cudaMemcpy2DAsync(cache.conv_state[il]->data, dpitch,
                                           conv_src, spitch,
                                           width, row_cnt,
                                           cudaMemcpyDeviceToDevice, stream);
                    if (ce != cudaSuccess) {
                        std::fprintf(stderr, "cudaMemcpy2D conv rollback il=%d: %s\n",
                                     il, cudaGetErrorString(ce));
                        return 1;
                    }
                }
                cudaStreamSynchronize(stream);

                ggml_free(tmp_ctx);
            }

            // Next last_tok: target's prediction at position committed+accept_n
            // given the accepted prefix.
            //   - commit_n < q_len: verify_logits[accept_n-1] (target_tok[accept_n-1]).
            //   - commit_n == q_len: verify_logits[q_len-1]  (target_tok[q_len-1]).
            // Both already computed as `target_tok[commit_n-1]` during accept.
            last_tok = target_tok[commit_n - 1];

            auto T_rb1 = sync_us();
            t_rollback_us = std::chrono::duration<double, std::micro>(T_rb1 - T_rb0).count();
            tt_restore += t_rollback_us;

            // Commit: push accepted draft tokens to out_all. No bonus — next iter
            // picks it up as last_tok.
            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                out_all.push_back(draft_tok[i]); stream_emit(draft_tok[i]);
                if (IS_EOS_TOK(draft_tok[i], w)) hit_eos = true;
            }
            if (hit_eos) break;
        } else {
            // ── Legacy replay path ──
            restore_ssm_state(cache);
            auto T_restore = sync_us();
            tt_restore += std::chrono::duration<double, std::micro>(T_restore - T_accept).count();
            std::vector<int32_t> replay_tok(commit_n);
            for (int i = 0; i < commit_n; i++) {
                if (i < accept_n && i < (int)draft_tok.size()) {
                    replay_tok[i] = draft_tok[i];
                } else {
                    replay_tok[i] = bonus_tok;
                }
            }

            bool replay_with_mask = (commit_n > 1);
            const int replay_fa_window = g_fa_window;
            if (!build_target_step(sg, w, cache, backend,
                                    committed, commit_n,
                                    replay_with_mask, /*capture=*/true,
                                    false, replay_fa_window,
                                    /*last_token_logits_only=*/false,
                                    g_kq_stride_pad)) {
                std::fprintf(stderr, "replay build failed\n"); return 1;
            }
            auto T_replay_build = sync_us();
            tt_replay_build += std::chrono::duration<double, std::micro>(T_replay_build - T_restore).count();

            std::vector<float> replay_embed(hidden * commit_n);
            if (!w.embedder.embed(replay_tok.data(), commit_n, replay_embed.data())) return 1;
            ggml_backend_tensor_set(sg.inp_embed, replay_embed.data(), 0, sizeof(float) * replay_embed.size());
            std::vector<int32_t> replay_pos(4 * commit_n);
            for (int i = 0; i < commit_n; i++) {
                int p = committed + i;
                replay_pos[0 * commit_n + i] = p;
                replay_pos[1 * commit_n + i] = p;
                replay_pos[2 * commit_n + i] = p;
                replay_pos[3 * commit_n + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, replay_pos.data(), 0, sizeof(int32_t) * 4 * commit_n);
            if (replay_with_mask) {
                const int win_start_r = (replay_fa_window > 0 && committed > replay_fa_window)
                                            ? (committed - replay_fa_window) : 0;
                const int win_len_r = committed + commit_n - win_start_r;
                build_causal_mask(mask_buf, win_len_r, commit_n, committed, g_kq_stride_pad, win_start_r);
                ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0, sizeof(uint16_t) * mask_buf.size());
            }
            auto T_replay_set = sync_us();
            tt_replay_set += std::chrono::duration<double, std::micro>(T_replay_set - T_replay_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "replay compute %d\n", (int)st); return 1; }
            auto T_replay_compute = sync_us();
            tt_replay_compute += std::chrono::duration<double, std::micro>(T_replay_compute - T_replay_set).count();

            std::vector<float> last_logits(vocab);
            ggml_backend_tensor_get(sg.logits, last_logits.data(),
                                    sizeof(float) * vocab * (commit_n - 1),
                                    sizeof(float) * vocab);
            last_tok = argmax_f32(last_logits.data(), vocab);
            auto T_replay_logits = sync_us();
            tt_replay_logits += std::chrono::duration<double, std::micro>(T_replay_logits - T_replay_compute).count();

            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                out_all.push_back(replay_tok[i]); stream_emit(replay_tok[i]);
                if (IS_EOS_TOK(replay_tok[i], w)) hit_eos = true;
            }
            if (hit_eos) break;
        }

        if (!sync_draft_feature_mirror(committed, commit_n)) {
            std::fprintf(stderr, "draft feature mirror sync failed after commit\n");
            return 1;
        }

        committed    += commit_n;
        n_generated  += commit_n;
        n_accept_sum += accept_n;
        n_draft_steps++;
    }

    auto t_gen1 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(t_gen1 - t_gen0).count();
    double tps = n_generated / std::max(1e-9, gen_s);

    auto avg_ms = [&](double us){ return us / std::max(1, n_draft_steps) / 1000.0; };
    std::printf("\n[timing] per-step averages over %d steps (ms):\n", n_draft_steps);
    std::printf("  draft_build    %.2f\n", avg_ms(tt_draft_build));
    std::printf("  draft_copyfeat %.2f\n", avg_ms(tt_draft_copy_feat));
    std::printf("  draft_set      %.2f\n", avg_ms(tt_draft_set));
    std::printf("  draft_compute  %.2f\n", avg_ms(tt_draft_compute));
    std::printf("  draft_bridge   %.2f\n", avg_ms(tt_draft_bridge));
    std::printf("  draft_logits   %.2f\n", avg_ms(tt_draft_logits));
    std::printf("  snapshot_ssm   %.2f\n", avg_ms(tt_snap));
    std::printf("  verify_build   %.2f\n", avg_ms(tt_verify_build));
    std::printf("  verify_set     %.2f\n", avg_ms(tt_verify_set));
    std::printf("  verify_compute %.2f\n", avg_ms(tt_verify_compute));
    std::printf("  verify_logits  %.2f\n", avg_ms(tt_verify_logits));
    std::printf("  accept         %.2f\n", avg_ms(tt_accept));
    std::printf("  restore_ssm    %.2f\n", avg_ms(tt_restore));
    std::printf("  replay_build   %.2f\n", avg_ms(tt_replay_build));
    std::printf("  replay_set     %.2f\n", avg_ms(tt_replay_set));
    std::printf("  replay_compute %.2f\n", avg_ms(tt_replay_compute));
    std::printf("  replay_logits  %.2f\n", avg_ms(tt_replay_logits));
    std::printf("  mirror_sync    %.2f\n", avg_ms(tt_mirror_sync));
    double sum_ms = avg_ms(tt_draft_build + tt_draft_copy_feat + tt_draft_set + tt_draft_compute + tt_draft_logits
                           + tt_draft_bridge
                           + tt_snap + tt_verify_build + tt_verify_set + tt_verify_compute + tt_verify_logits
                           + tt_accept + tt_restore + tt_replay_build + tt_replay_set + tt_replay_compute + tt_replay_logits
                           + tt_mirror_sync);
    std::printf("  ----- sum     %.2f\n", sum_ms);

    std::printf("\n[dflash] generated %d tokens in %.3f s  ->  %.2f tok/s\n",
                n_generated, gen_s, tps);
    std::printf("[dflash] %d draft steps, accepted=%d/%d (%.1f%% per step), "
                "avg commit/step=%.2f\n",
                n_draft_steps, n_accept_sum, n_draft_steps * q_len,
                (n_draft_steps > 0 ? 100.0 * n_accept_sum / (n_draft_steps * q_len) : 0.0),
                (n_draft_steps > 0 ? (double)n_generated / n_draft_steps : 0.0));
    std::printf("[dflash] output tail: ");
    int tail_start = std::max(0, (int)out_all.size() - 20);
    for (int i = tail_start; i < (int)out_all.size(); i++) std::printf("%d ", out_all[i]);
    std::printf("\n");

    if (daemon_mode) {
        // Update cache.cur_pos / cache.last_tok to reflect end-of-generation
        // state so a subsequent SNAPSHOT command captures the correct boundary.
        // Both fields are otherwise unused by the prefill/decode hot path
        // (kv_start is tracked separately, last_tok is a local) — they exist
        // for cross-request snapshot accounting.
        cache.cur_pos  = (int)out_all.size();
        cache.last_tok = last_tok;
        stream_emit(-1);
    } else {
        if (out_path) write_int32_file(out_path, out_all);
        break;
    }

    } // end while(true)

    draft_feature_mirror_free(feature_mirror);
    step_graph_destroy(proj_sg);
    step_graph_destroy(draft_sg);
    if (daemon_mode) {
        for (int i = 0; i < PREFIX_CACHE_SLOTS; i++) free_prefix_snapshot(prefix_snapshots[i]);
    }
    step_graph_destroy(sg);
    free_target_cache(cache);
    free_draft_weights(dw);
    free_target_weights(w);
    if (split_gpus) ggml_backend_free(draft_backend);
    ggml_backend_free(target_backend);
    return 0;
}
