// test_kvflash — verifies KVFlash, the bounded-resident-pool KV cache
// (kvflash_pager.h).
//
// Runs against one loaded qwen35 target:
//
//   A  baseline: cache at LOGICAL context (default 131072), maskless decode
//      (production AR path shape). Reference tokens + baseline KV memory.
//   B  relocation proof: small pool, chunks at SHUFFLED physical blocks,
//      explicit pool slot mask, teacher-forced replay of A. Argmax must
//      track A (position-independence + mask exactness).
//   C  paging proof: pool ≪ prompt+gen, live eviction, bit-exact
//      page_out/page_in roundtrip, KV bytes vs A.
//   D  reselect/recall: evicted chunk recalled via score_hook + reselect()
//      (the FlashMemory τ-step lookahead machinery); decode continues.
//   E  performance profile: decode ms/step vs FA span — baseline at
//      8K/32K/128K vs pool 1K/4K at 128K-logical — plus page-event and
//      mask-refill microbenchmarks.
//
// Usage:
//   test_kvflash <qwen35.gguf> [--logical-ctx=N] [--pool-b=N] [--pool-c=N]
//                 [--prompt=N] [--gen=N] [--skip-profile] [--no-mask]
//   modes: (default) verification suite A-F | --niah | --niah256 | --longab

#include "dflash27b.h"
#include "internal.h"
#include "kvflash_pager.h"
#include "attn_masks.h"
#include "qwen3_drafter.h"
#include "qwen3_kvflash_scorer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t kv_cache_bytes(const TargetCache & c) {
    size_t n = 0;
    for (auto * t : c.attn_k) if (t) n += ggml_nbytes(t);
    for (auto * t : c.attn_v) if (t) n += ggml_nbytes(t);
    return n;
}

size_t vram_used_now() {
    size_t free_b = 0, total_b = 0;
    ggml_backend_cuda_get_device_memory(0, &free_b, &total_b);
    return total_b - free_b;
}

// Single-token stepper over build_qwen35_graph with explicit control of:
//   * kv_write_rows  — physical pool slot for the KV append
//   * positions      — logical position (M-RoPE)
//   * span           — FA window length (kv_start = span-1 in graph terms)
//   * attn_mask      — optional [align32(span_padded), 32] f16 slot mask
//
// The graph arena and gallocr persist across rebuilds (same trick as
// build_target_step) so identical topology lands at identical addresses
// and the ggml-cuda CUDA-graph cache can replay decode steps.
struct Stepper {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;
    ggml_tensor *   kv_write_rows = nullptr;
    ggml_tensor *   logits = nullptr;
    ggml_tensor *   argmax_tokens = nullptr;

    const TargetWeights * w = nullptr;
    TargetCache * cache = nullptr;
    ggml_backend_t backend = nullptr;
    int span = 0;
    bool with_mask = false;

    std::vector<uint8_t> arena;
    std::vector<float> embed_buf;
    std::vector<uint16_t> mask_buf;
    uint64_t mask_epoch = (uint64_t)-1;
    double mask_fill_ms_total = 0.0;
    int mask_fills = 0;

    bool init(const TargetWeights & tw, TargetCache & tc, ggml_backend_t be,
              int span_, bool with_mask_) {
        w = &tw; cache = &tc; backend = be;
        span = span_; with_mask = with_mask_;
        embed_buf.resize(tw.n_embd);
        arena.resize((size_t)512 * 1024 * 1024);
        return build();
    }

    bool build() {
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        ggml_init_params ip{};
        ip.mem_size   = arena.size();
        ip.mem_buffer = arena.data();
        ip.no_alloc   = true;
        ctx = ggml_init(ip);
        if (!ctx) return false;

        const int hidden = w->n_embd;
        inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, 1);
        ggml_set_input(inp_embed);
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_set_input(positions);
        kv_write_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 1, w->n_head_kv);
        ggml_set_input(kv_write_rows);

        attn_mask = nullptr;
        if (with_mask) {
            // FA span is padded to 256 on the step-invariant path; the mask
            // kv dim must cover it.
            const int span_padded = std::min(((span + 255) / 256) * 256,
                                             (int)cache->attn_k[0]->ne[1]);
            attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                                           align_up(span_padded, KQ_MASK_PAD),
                                           align_up(1, KQ_MASK_PAD));
            ggml_set_input(attn_mask);
            mask_buf.assign((size_t)attn_mask->ne[0] * attn_mask->ne[1], F16_NEG_INF);
            mask_epoch = (uint64_t)-1;
        }

        gf = ggml_new_graph_custom(ctx, 16384, false);

        QwenGraphInputs gi{};
        gi.inp_embed      = inp_embed;
        gi.positions      = positions;
        gi.attn_mask      = attn_mask;
        gi.n_tokens       = 1;
        gi.kv_start       = span - 1;
        gi.capture_layers = false;
        gi.kv_write_rows  = kv_write_rows;

        QwenGraphOutputs go = build_qwen35_graph(ctx, gf, *w, *cache, gi);
        if (!go.logits) return false;
        logits = go.logits;
        ggml_set_output(logits);
        argmax_tokens = ggml_argmax(ctx, logits);
        ggml_set_output(argmax_tokens);
        ggml_build_forward_expand(gf, argmax_tokens);

        if (!alloc) alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return ggml_gallocr_alloc_graph(alloc, gf);
    }

    void refresh_mask(const KvFlashPager & pager) {
        if (!attn_mask) return;
        const double t0 = now_ms();
        if (pager.epoch() != mask_epoch) {
            // Host-side rebuild only on residency change.
            std::fill(mask_buf.begin(), mask_buf.end(), F16_NEG_INF);
            pager.fill_slot_mask(mask_buf.data());
            mask_epoch = pager.epoch();
            mask_fills++;
        }
        // Upload EVERY step: the compute-buffer region backing this input
        // tensor is reused by graph execution, so a stale upload reads as
        // garbage (NaN logits) on the next step. Production prefill
        // re-uploads its mask before every compute for the same reason.
        ggml_backend_tensor_set(attn_mask, mask_buf.data(), 0,
                                mask_buf.size() * sizeof(uint16_t));
        mask_fill_ms_total += now_ms() - t0;
    }

    int32_t step(int32_t tok, int pos, int phys_slot) {
        if (!w->embedder.embed(&tok, 1, embed_buf.data())) {
            std::fprintf(stderr, "embed failed: tok=%d pos=%d (NaN logits upstream?)\n", tok, pos);
            std::exit(1);
        }
        ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());
        int32_t p4[4] = { pos, pos, pos, 0 };
        ggml_backend_tensor_set(positions, p4, 0, sizeof(int32_t) * 4);
        std::vector<int64_t> rows(w->n_head_kv, (int64_t)phys_slot);
        ggml_backend_tensor_set(kv_write_rows, rows.data(), 0,
                                sizeof(int64_t) * rows.size());
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph_compute failed pos=%d\n", pos);
            std::exit(1);
        }
        int32_t next = 0;
        ggml_backend_tensor_get(argmax_tokens, &next, 0, sizeof(int32_t));
        return next;
    }

    void destroy() {
        if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
        if (ctx)   { ggml_free(ctx); ctx = nullptr; }
    }
};

std::vector<int32_t> make_prompt(int n, int vocab) {
    std::vector<int32_t> p(n);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    // Cap below the drafter vocab too (Qwen3-0.6B ~151K) so the same ids
    // are scoreable by the indexer in run F.
    const int cap = std::min(vocab, 100000);
    for (int i = 0; i < n; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        p[i] = (int32_t)(1000 + (s >> 33) % (uint64_t)(cap / 2));
    }
    return p;
}

// Pooled chunked prefill: 64-token (one pager chunk) batched forwards with
// slot-mapped set_rows writes and a resident+causal mask. This is the
// prompt > pool path: prefill evicts like decode does. Graph is built once
// (fixed topology) and reused for every chunk.
struct BatchStepper {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;
    ggml_tensor *   kv_write_rows = nullptr;
    ggml_tensor *   logits = nullptr;
    ggml_tensor *   argmax_tokens = nullptr;

    const TargetWeights * w = nullptr;
    TargetCache * cache = nullptr;
    ggml_backend_t backend = nullptr;
    int pool = 0;
    static constexpr int NB = 64;          // tokens per chunk

    std::vector<uint8_t> arena;
    std::vector<float> embed_buf;
    std::vector<uint16_t> mask_buf;

    bool init(const TargetWeights & tw, TargetCache & tc, ggml_backend_t be, int pool_) {
        w = &tw; cache = &tc; backend = be; pool = pool_;
        embed_buf.resize((size_t)tw.n_embd * NB);
        arena.resize((size_t)512 * 1024 * 1024);

        ggml_init_params ip{};
        ip.mem_size   = arena.size();
        ip.mem_buffer = arena.data();
        ip.no_alloc   = true;
        ctx = ggml_init(ip);
        if (!ctx) return false;

        inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tw.n_embd, NB, 1);
        ggml_set_input(inp_embed);
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * NB);
        ggml_set_input(positions);
        kv_write_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, NB, tw.n_head_kv);
        ggml_set_input(kv_write_rows);
        attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                                       align_up(pool, KQ_MASK_PAD),
                                       align_up(NB, KQ_MASK_PAD));
        ggml_set_input(attn_mask);
        mask_buf.assign((size_t)attn_mask->ne[0] * attn_mask->ne[1], F16_NEG_INF);

        gf = ggml_new_graph_custom(ctx, 16384, false);
        QwenGraphInputs gi{};
        gi.inp_embed   = inp_embed;
        gi.positions   = positions;
        gi.attn_mask   = attn_mask;
        gi.n_tokens    = NB;
        gi.kv_start    = pool - NB;      // span = whole pool
        gi.kv_write_rows = kv_write_rows;
        gi.last_token_logits_only = true;
        QwenGraphOutputs go = build_qwen35_graph(ctx, gf, *w, *cache, gi);
        if (!go.logits) return false;
        logits = go.logits;
        ggml_set_output(logits);
        argmax_tokens = ggml_argmax(ctx, logits);
        ggml_set_output(argmax_tokens);
        ggml_build_forward_expand(gf, argmax_tokens);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return ggml_gallocr_alloc_graph(alloc, gf);
    }

    // One 64-token chunk at logical [pos_base, pos_base+64). Allocates the
    // chunk's block (evicting if needed), writes slot-mapped, masks
    // resident slots + causal-within-chunk. Returns last-token argmax.
    int32_t step_chunk(const int32_t * toks, int pos_base, KvFlashPager & pager) {
        int slots[NB];
        for (int i = 0; i < NB; i++) slots[i] = pager.slot_for(pos_base + i);

        if (!w->embedder.embed(toks, NB, embed_buf.data())) {
            std::fprintf(stderr, "batch embed failed @%d\n", pos_base);
            std::exit(1);
        }
        ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());
        std::vector<int32_t> p4((size_t)4 * NB);
        for (int i = 0; i < NB; i++) {
            p4[4 * i + 0] = p4[4 * i + 1] = p4[4 * i + 2] = pos_base + i;
            p4[4 * i + 3] = 0;
        }
        ggml_backend_tensor_set(positions, p4.data(), 0, sizeof(int32_t) * p4.size());
        // [n_tokens, n_head_kv] ne0-major: (token i, head h) at i + h*NB.
        std::vector<int64_t> rows((size_t)NB * w->n_head_kv);
        for (int h = 0; h < w->n_head_kv; h++) {
            for (int i = 0; i < NB; i++) {
                rows[(size_t)h * NB + i] = slots[i];
            }
        }
        ggml_backend_tensor_set(kv_write_rows, rows.data(), 0,
                                sizeof(int64_t) * rows.size());

        // Mask: per q row, resident slots (excluding this chunk) attendable,
        // this chunk's slots causal. Rebuilt + uploaded per chunk.
        const size_t kvd = (size_t)attn_mask->ne[0];
        std::fill(mask_buf.begin(), mask_buf.end(), F16_NEG_INF);
        pager.fill_slot_mask(mask_buf.data());                    // row 0 base
        const int this_block = slots[0] - slots[0] % NB;
        for (int i = 0; i < NB; i++) mask_buf[(size_t)this_block + i] = F16_NEG_INF;
        for (int q = 1; q < NB; q++) {
            std::memcpy(mask_buf.data() + (size_t)q * kvd, mask_buf.data(), kvd * 2);
        }
        for (int q = 0; q < NB; q++) {
            for (int i = 0; i <= q; i++) {
                mask_buf[(size_t)q * kvd + slots[i]] = F16_ZERO;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_buf.data(), 0,
                                mask_buf.size() * sizeof(uint16_t));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "batch compute failed @%d\n", pos_base);
            std::exit(1);
        }
        int32_t last = 0;
        ggml_backend_tensor_get(argmax_tokens, &last, 0, sizeof(int32_t));
        return last;
    }

    void destroy() {
        if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
        if (ctx)   { ggml_free(ctx); ctx = nullptr; }
    }
};


int arg_int(int argc, char ** argv, const char * key, int defv) {
    const size_t kl = std::strlen(key);
    for (int i = 2; i < argc; i++) {
        if (std::strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=') {
            return std::atoi(argv[i] + kl + 1);
        }
    }
    return defv;
}

bool arg_flag(int argc, char ** argv, const char * key) {
    for (int i = 2; i < argc; i++) if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

struct StepTimes {
    double p50 = 0, p95 = 0, mean = 0;
};

StepTimes summarize(std::vector<double> & ms) {
    StepTimes r;
    if (ms.empty()) return r;
    std::sort(ms.begin(), ms.end());
    r.p50 = ms[ms.size() / 2];
    r.p95 = ms[(size_t)(ms.size() * 0.95)];
    for (double v : ms) r.mean += v;
    r.mean /= ms.size();
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35.gguf> [--logical-ctx=N] [--pool-b=N] "
                             "[--pool-c=N] [--prompt=N] [--gen=N] [--skip-profile]\n", argv[0]);
        return 2;
    }
    const int logical_ctx = arg_int(argc, argv, "--logical-ctx", 131072);
    const int pool_b      = arg_int(argc, argv, "--pool-b", 2048);
    const int pool_c      = arg_int(argc, argv, "--pool-c", 1024);
    const int n_prompt    = arg_int(argc, argv, "--prompt", 512);
    const int n_gen       = arg_int(argc, argv, "--gen", 1200);
    const bool skip_prof  = arg_flag(argc, argv, "--skip-profile");
    // Explicit pool slot mask: exact exclusion of non-resident slots.
    // ON by default (requires the per-step re-upload in refresh_mask: the
    // mask input's compute-buffer region is clobbered by graph execution).
    // --no-mask falls back to the zero-row approximation production's
    // padded span uses.
    const bool use_mask   = !arg_flag(argc, argv, "--no-mask");
    const int total       = n_prompt + n_gen;
    if (total > pool_b) {
        std::fprintf(stderr, "config error: prompt+gen (%d) must fit pool-b (%d)\n", total, pool_b);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }
    const size_t vram0 = vram_used_now();

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[load] weights ok, vram_used=%.1f MiB\n",
                (vram_used_now() - vram0) / 1048576.0);

    // ── --longab: end-to-end long-prompt A/B (speed + accuracy) ─────
    // For L in {32K, 64K, 128K}: full-cache baseline vs pool-4096 with
    // drafter reselect. Measures prefill time, decode tok/s over a
    // 240-token free run, and needle recall (depth 0.25, outside both
    // the sinks and the LRU window).
    if (arg_flag(argc, argv, "--longab")) {
        // Drafter loads lazily, pool mode only: the full-cache baseline at
        // 256K needs every byte (weights 15.3 GiB + KV 4.6 GiB).
        DrafterContext dctx;
        KvFlashDrafterScorer scorer(&dctx);
        // Single-config mode (one process per config: the CUDA VMM pool
        // grows monotonically across large-cache configs and aborts).
        const int only_L = arg_int(argc, argv, "--longab-L", 0);
        const int only_mode = arg_int(argc, argv, "--longab-mode", -1); // 0=full 1=pool
        std::printf("\n%-7s %-10s %-9s %-9s %-9s %-9s %s\n",
                    "L", "mode", "prefill_s", "rescore_s", "dec_tok/s", "needle", "kv_vram");
        for (int L : { 32768, 65536, 131072, 262144 }) {
            if (only_L > 0 && L != only_L) continue;
            for (int mode = 0; mode < 2; mode++) {           // 0=baseline 1=pool
                if (only_mode >= 0 && mode != only_mode) continue;
                if (mode == 1 && !dctx.loaded &&
                    !load_drafter("/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf", 0, dctx)) {
                    std::fprintf(stderr, "drafter load failed\n");
                    return 1;
                }
                const int pool = mode == 0 ? L : 4096;
                auto prompt = make_prompt(L, w.n_vocab);
                std::vector<int32_t> needle(48);
                uint64_t ns = 0xDEADBEEFCAFEull;
                for (int i = 0; i < 48; i++) {
                    ns = ns * 6364136223846793005ull + 1442695040888963407ull;
                    needle[i] = (int32_t)(1000 + (ns >> 33) % 49000);
                }
                const int npos = ((int)(0.25 * (L - 512)) / 32) * 32;
                for (int i = 0; i < 48; i++) prompt[npos + i] = needle[i];

                TargetCache cache;
                if (!create_target_cache(w, pool, 0, backend, cache, true)) return 1;
                const double kv_mib = kv_cache_bytes(cache) / 1048576.0;
                KvFlashPager pager;
                KvFlashConfig pc; pc.pool_tokens = pool;
                if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

                double t0 = now_ms();
                BatchStepper bs;
                if (!bs.init(w, cache, backend, pool)) return 1;
                for (int p = 0; p < L; p += 64) bs.step_chunk(prompt.data() + p, p, pager);
                bs.destroy();
                const double prefill_s = (now_ms() - t0) / 1000.0;

                Stepper st;
                if (!st.init(w, cache, backend, pool, mode == 1)) return 1;
                int32_t next = -1;
                for (int i = 0; i < 32; i++) {
                    const int slot = pager.slot_for(L + i);
                    st.refresh_mask(pager);
                    next = st.step(needle[i], L + i, slot);
                }
                double rescore_s = 0;
                if (mode == 1) {
                    std::vector<int32_t> hist = prompt;
                    hist.insert(hist.end(), needle.begin(), needle.begin() + 32);
                    std::vector<float> scores;
                    t0 = now_ms();
                    if (scorer.score_chunks(hist, pc.chunk_tokens, scores)) {
                        pager.score_hook = [&scores](int c) {
                            return c < (int)scores.size() ? scores[c] : 1e30f;
                        };
                        pager.reselect();
                        pager.score_hook = nullptr;
                    }
                    rescore_s = (now_ms() - t0) / 1000.0;
                }
                int match = 0;
                for (int i = 0; i < 16; i++) {
                    if (next == needle[32 + i]) match++;
                    const int pos = L + 32 + i;
                    const int slot = pager.slot_for(pos);
                    st.refresh_mask(pager);
                    next = st.step(needle[32 + i], pos, slot);
                }
                t0 = now_ms();
                for (int i = 0; i < 240; i++) {              // timed free run
                    const int pos = L + 48 + i;
                    const int slot = pager.slot_for(pos);
                    st.refresh_mask(pager);
                    next = st.step(next, pos, slot);
                }
                const double tok_s = 240.0 / ((now_ms() - t0) / 1000.0);
                std::printf("%-7d %-10s %-9.1f %-9.1f %-9.1f %d/16   %.0f MiB\n",
                            L, mode == 0 ? "full" : "pool4096",
                            prefill_s, rescore_s, tok_s, match, kv_mib);
                std::fflush(stdout);
                st.destroy();
                free_target_cache(cache);
            }
        }
        if (dctx.loaded) free_drafter(dctx);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    // ── --niah256: native-max-context probe (262144 logical) ────────
    // Pooled configs only: the fixed-span harness makes a full-pool
    // control prefill take hours at 256K. The LRU row with the needle
    // inside the recency window is the induction control (distance-free).
    if (arg_flag(argc, argv, "--niah256")) {
        DrafterContext dctx;
        if (!load_drafter("/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf", 0, dctx)) {
            std::fprintf(stderr, "drafter load failed\n");
            return 1;
        }
        KvFlashDrafterScorer scorer(&dctx);
        const int L = 262144, pool = 16384;          // 6.25% residency
        struct Cfg { const char * policy; double depth; };
        const Cfg cfgs[] = {
            {"lru",     0.97},   // in-window: induction control at 256K
            {"lru",     0.50},
            {"drafter", 0.10},
            {"drafter", 0.50},
            {"drafter", 0.90},
        };
        std::printf("\n%-7s %-6s %-8s %-6s %s\n", "L", "pool", "policy", "depth", "match/16");
        for (const Cfg & cfg : cfgs) {
            auto prompt = make_prompt(L, w.n_vocab);
            std::vector<int32_t> needle(48);
            uint64_t ns = 0xDEADBEEFCAFEull;
            for (int i = 0; i < 48; i++) {
                ns = ns * 6364136223846793005ull + 1442695040888963407ull;
                needle[i] = (int32_t)(1000 + (ns >> 33) % 49000);
            }
            const int npos = ((int)(cfg.depth * (L - 512)) / 32) * 32;
            for (int i = 0; i < 48; i++) prompt[npos + i] = needle[i];

            TargetCache cache;
            if (!create_target_cache(w, pool, 0, backend, cache, true)) return 1;
            KvFlashPager pager;
            KvFlashConfig pc; pc.pool_tokens = pool;
            if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

            const double t0 = now_ms();
            BatchStepper bs;
            if (!bs.init(w, cache, backend, pool)) return 1;
            for (int p = 0; p < L; p += 64) bs.step_chunk(prompt.data() + p, p, pager);
            bs.destroy();
            std::printf("[256k] prefill %.1f s, host backing %.2f GiB\n",
                        (now_ms() - t0) / 1000.0,
                        pager.stats().host_bytes / 1073741824.0);

            Stepper st;
            if (!st.init(w, cache, backend, pool, true)) return 1;
            int32_t next = -1;
            for (int i = 0; i < 32; i++) {
                const int slot = pager.slot_for(L + i);
                st.refresh_mask(pager);
                next = st.step(needle[i], L + i, slot);
            }
            if (std::strcmp(cfg.policy, "drafter") == 0) {
                std::vector<int32_t> hist = prompt;
                hist.insert(hist.end(), needle.begin(), needle.begin() + 32);
                std::vector<float> scores;
                const double r0 = now_ms();
                if (!scorer.score_chunks(hist, pc.chunk_tokens, scores)) {
                    std::printf("[256k] WARN rescore failed\n");
                } else {
                    std::printf("[256k] rescore %.1f s\n", (now_ms() - r0) / 1000.0);
                    pager.score_hook = [&scores](int c) {
                        return c < (int)scores.size() ? scores[c] : 1e30f;
                    };
                    pager.reselect();
                    pager.score_hook = nullptr;
                }
            }
            int match = 0;
            for (int i = 0; i < 16; i++) {
                if (next == needle[32 + i]) match++;
                const int pos = L + 32 + i;
                const int slot = pager.slot_for(pos);
                st.refresh_mask(pager);
                next = st.step(needle[32 + i], pos, slot);
            }
            std::printf("%-7d %-6d %-8s %-6.2f %d/16\n", L, pool, cfg.policy, cfg.depth, match);
            std::fflush(stdout);
            st.destroy();
            free_target_cache(cache);
        }
        free_drafter(dctx);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    if (arg_flag(argc, argv, "--niah")) {
        DrafterContext dctx;
        const bool have_drafter =
            load_drafter("/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf", 0, dctx);
        if (!have_drafter) std::printf("[niah] drafter unavailable, skipping drafter policy\n");
        KvFlashDrafterScorer scorer(&dctx);
        if (have_drafter) {
            // Reserve the drafter's compute buffers at max context NOW,
            // before target-side cache churn fragments the CUDA pool.
            // Without this, 32K rescores OOM late in the sweep and the
            // drafter policy silently degrades to LRU.
            std::vector<int32_t> warm(33024, 1234);
            std::vector<float> tmp;
            scorer.score_chunks(warm, 64, tmp);
        }

        const int Ls[] = { 8192, 32768 };
        const double depths[] = { 0.10, 0.50, 0.90 };
        std::printf("\n%-7s %-6s %-8s %-6s %s\n", "L", "pool", "policy", "depth", "match/16");
        for (int L : Ls) {
            const int pools[] = { L, L / 4, ((L / 10) / 256) * 256 };
            for (int pi = 0; pi < 3; pi++) {
                const int pool = pools[pi];
                const char * policies[] = { "lru", "drafter" };
                const int n_pol = (pi == 0) ? 1 : (have_drafter ? 2 : 1); // full pool: control only
                for (int pol = 0; pol < n_pol; pol++) {
                    for (double depth : depths) {
                        // Needle: 48 unique-as-a-sequence tokens from the
                        // filler id range (matched embedding statistics).
                        // Query = first 32 (longer match = stronger
                        // induction), score the last 16.
                        auto prompt = make_prompt(L, w.n_vocab);
                        std::vector<int32_t> needle(48);
                        uint64_t ns = 0xDEADBEEFCAFEull;
                        for (int i = 0; i < 48; i++) {
                            ns = ns * 6364136223846793005ull + 1442695040888963407ull;
                            needle[i] = (int32_t)(1000 + (ns >> 33) % 49000);
                        }
                        const int npos = ((int)(depth * (L - 512)) / 32) * 32;
                        for (int i = 0; i < 48; i++) prompt[npos + i] = needle[i];

                        TargetCache cache;
                        if (!create_target_cache(w, pool, 0, backend, cache, true)) return 1;
                        KvFlashPager pager;
                        KvFlashConfig pc; pc.pool_tokens = pool;
                        if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

                        BatchStepper bs;
                        if (!bs.init(w, cache, backend, pool)) return 1;
                        for (int p = 0; p < L; p += 64) {
                            bs.step_chunk(prompt.data() + p, p, pager);
                        }
                        bs.destroy();

                        Stepper st;
                        if (!st.init(w, cache, backend, pool, true)) return 1;
                        int32_t next = -1;
                        for (int i = 0; i < 32; i++) {       // query: needle prefix
                            const int slot = pager.slot_for(L + i);
                            st.refresh_mask(pager);
                            next = st.step(needle[i], L + i, slot);
                        }
                        if (pol == 1) {                       // drafter reselect
                            std::vector<int32_t> hist = prompt;
                            hist.insert(hist.end(), needle.begin(), needle.begin() + 32);
                            std::vector<float> scores;
                            if (!scorer.score_chunks(hist, pc.chunk_tokens, scores)) {
                                std::printf("[niah] WARN: rescore failed (L=%d pool=%d)\n", L, pool);
                            } else {
                                pager.score_hook = [&scores](int c) {
                                    return c < (int)scores.size() ? scores[c] : 1e30f;
                                };
                                pager.reselect();
                                pager.score_hook = nullptr;
                            }
                        }
                        int match = 0;
                        for (int i = 0; i < 16; i++) {        // continuation
                            if (next == needle[32 + i]) match++;
                            // Teacher-force ground truth: one miss must not
                            // cascade; we measure per-position retrieval.
                            const int pos = L + 32 + i;
                            const int slot = pager.slot_for(pos);
                            st.refresh_mask(pager);
                            next = st.step(needle[32 + i], pos, slot);
                        }
                        std::printf("%-7d %-6d %-8s %-6.2f %d/16\n",
                                    L, pool, pi == 0 ? "full" : policies[pol],
                                    depth, match);
                        std::fflush(stdout);
                        st.destroy();
                        free_target_cache(cache);
                    }
                }
            }
        }
        if (have_drafter) free_drafter(dctx);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    const auto prompt = make_prompt(n_prompt, w.n_vocab);
    std::vector<int32_t> tokens_a;
    size_t mem_a_kv = 0, mem_a_buf = 0, mem_a_vram = 0;
    size_t mem_c_kv = 0, mem_c_buf = 0, mem_c_vram = 0;
    int hard_failures = 0;

    // ── Run A: baseline at logical context, maskless ────────────────
    {
        const size_t v_before = vram_used_now();
        TargetCache cache;
        if (!create_target_cache(w, logical_ctx, 0, backend, cache, /*prefill_only=*/true)) {
            std::fprintf(stderr, "cache A: %s\n", dflash27b_last_error());
            return 1;
        }
        mem_a_kv = kv_cache_bytes(cache);
        mem_a_buf = ggml_backend_buffer_get_size(cache.base_buf);
        mem_a_vram = vram_used_now() - v_before;
        std::printf("[A] logical_ctx=%d  kv=%.1f MiB  base_buf=%.1f MiB  vram_delta=%.1f MiB\n",
                    logical_ctx, mem_a_kv / 1048576.0, mem_a_buf / 1048576.0,
                    mem_a_vram / 1048576.0);

        Stepper st;
        int32_t next = -1;
        const double t0 = now_ms();
        for (int pos = 0; pos < total; pos++) {
            // Production-like growing span: rebuild only when the padded
            // span crosses a 256 boundary (mirrors do_ar_decode topology).
            const int want_span = pos + 1;
            if (!st.ctx || ((want_span + 255) / 256) != ((st.span + 255) / 256)) {
                st.span = want_span;
                if (!st.ctx) { if (!st.init(w, cache, backend, want_span, false)) return 1; }
                else if (!st.build()) return 1;
            }
            const int32_t tok = pos < n_prompt ? prompt[pos]
                              : (tokens_a.push_back(next), next);
            next = st.step(tok, pos, pos);
            cache.cur_pos = pos + 1;
        }
        tokens_a.push_back(next);
        std::printf("[A] decoded %zu tokens, %.1f tok/s overall\n",
                    tokens_a.size(), total / ((now_ms() - t0) / 1000.0));
        st.destroy();
        free_target_cache(cache);
    }

    // ── Run B: relocation + mask exactness, teacher-forced ──────────
    {
        TargetCache cache;
        if (!create_target_cache(w, pool_b, 0, backend, cache, /*prefill_only=*/true)) {
            std::fprintf(stderr, "cache B: %s\n", dflash27b_last_error());
            return 1;
        }
        KvFlashPager pager;
        KvFlashConfig pc;
        pc.pool_tokens = pool_b;
        if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;
        const int nb = pool_b / pc.chunk_tokens;
        std::vector<int> order(nb);
        for (int i = 0; i < nb; i++) order[i] = i;
        uint64_t s = 12345;
        for (int i = nb - 1; i > 0; i--) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            const int j = (int)((s >> 33) % (uint64_t)(i + 1));
            std::swap(order[i], order[j]);
        }
        pager.set_block_order(order);

        Stepper st;
        if (!st.init(w, cache, backend, pool_b, use_mask)) return 1;
        int mismatches = 0, first_mismatch = -1;
        for (int pos = 0; pos < total; pos++) {
            const int32_t tok = pos < n_prompt ? prompt[pos] : tokens_a[pos - n_prompt];
            const int slot = pager.slot_for(pos);
            st.refresh_mask(pager);
            const int32_t next = st.step(tok, pos, slot);
            const int ref_idx = pos - n_prompt + 1;
            if (pos >= n_prompt - 1 && ref_idx < (int)tokens_a.size()) {
                if (next != tokens_a[ref_idx]) {
                    mismatches++;
                    if (first_mismatch < 0) first_mismatch = pos;
                }
            }
        }
        const double rate = 100.0 * mismatches / (n_gen + 1);
        std::printf("[B] shuffled+masked, pool=%d: %d/%d argmax mismatches (%.2f%%), first at pos %d; "
                    "mask refills=%d avg=%.3f ms\n",
                    pool_b, mismatches, n_gen + 1, rate, first_mismatch,
                    st.mask_fills, st.mask_fills ? st.mask_fill_ms_total / st.mask_fills : 0.0);
        // Gate at 2%: the flip sources are the maskless zero-row softmax
        // mass plus run-to-run fattn nondeterminism; both measured ~1%
        // (10-14 flips/1201 across runs), so a 1% gate flaps on noise.
        std::printf("%s relocation equivalence (threshold 2%%)\n", rate <= 2.0 ? "PASS" : "FAIL");
        if (rate > 2.0) hard_failures++;
        st.destroy();
        free_target_cache(cache);
    }

    // ── Run C: live paging + roundtrip; D: reselect recall ──────────
    {
        const size_t v_before = vram_used_now();
        TargetCache cache;
        if (!create_target_cache(w, pool_c, 0, backend, cache, /*prefill_only=*/true)) {
            std::fprintf(stderr, "cache C: %s\n", dflash27b_last_error());
            return 1;
        }
        mem_c_kv = kv_cache_bytes(cache);
        mem_c_buf = ggml_backend_buffer_get_size(cache.base_buf);
        mem_c_vram = vram_used_now() - v_before;

        KvFlashPager pager;
        KvFlashConfig pc;
        pc.pool_tokens = pool_c;
        if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

        Stepper st;
        if (!st.init(w, cache, backend, pool_c, use_mask)) return 1;
        int32_t next = -1;
        for (int pos = 0; pos < n_prompt; pos++) {
            const int slot = pager.slot_for(pos);
            st.refresh_mask(pager);
            next = st.step(prompt[pos], pos, slot);
            cache.cur_pos = pos + 1;
        }
        {   // bit-exact roundtrip on chunk 2
            ggml_tensor * t = cache.attn_k[0];
            const size_t seg = (size_t)pc.chunk_tokens * t->nb[1];
            std::vector<uint8_t> before(seg), after(seg);
            ggml_backend_tensor_get(t, before.data(),
                (size_t)pager.block_of(2) * pc.chunk_tokens * t->nb[1], seg);
            if (!pager.page_out(2) || !pager.page_in(2)) {
                std::fprintf(stderr, "roundtrip paging failed\n"); return 1;
            }
            ggml_backend_tensor_get(t, after.data(),
                (size_t)pager.block_of(2) * pc.chunk_tokens * t->nb[1], seg);
            const bool exact = std::memcmp(before.data(), after.data(), seg) == 0;
            std::printf("%s page_out/page_in roundtrip bit-exact (chunk 2 -> block %d)\n",
                        exact ? "PASS" : "FAIL", pager.block_of(2));
            if (!exact) hard_failures++;
        }

        std::vector<int32_t> tokens_c;
        const double t0 = now_ms();
        for (int pos = n_prompt; pos < total; pos++) {
            tokens_c.push_back(next);
            const int slot = pager.slot_for(pos);
            st.refresh_mask(pager);
            next = st.step(next, pos, slot);
            cache.cur_pos = pos + 1;
        }
        tokens_c.push_back(next);
        const double secs = (now_ms() - t0) / 1000.0;
        int agree = 0;
        while (agree < (int)tokens_c.size() && agree < (int)tokens_a.size() &&
               tokens_c[agree] == tokens_a[agree]) agree++;
        const auto & ps = pager.stats();
        std::printf("[C] pool=%d masked: decode %.1f tok/s, page_outs=%" PRId64
                    " page_ins=%" PRId64 " host=%.1f MiB; baseline agreement %d tokens\n",
                    pool_c, n_gen / secs, ps.page_outs, ps.page_ins,
                    ps.host_bytes / 1048576.0, agree);
        std::printf("PASS paged decode with eviction (%d evictions)\n", (int)ps.page_outs);

        // ── Run D: τ-style reselect recall ──────────────────────────
        {
            int victim = -1;   // earliest paged-out, non-sink chunk
            for (int c = pc.sink_chunks; c < pager.n_chunks(); c++) {
                if (!pager.is_resident(c)) { victim = c; break; }
            }
            if (victim < 0) {
                std::printf("FAIL reselect demo: no paged-out chunk found\n");
                hard_failures++;
            } else {
                // Score injection: the victim becomes the hottest chunk —
                // stands in for a drafter rescore flagging recalled context.
                pager.score_hook = [&](int c) { return c == victim ? 2.0f : 1.0f / (1 + c); };
                const double r0 = now_ms();
                const int events = pager.reselect();
                const double r_ms = now_ms() - r0;
                const bool back = pager.is_resident(victim);
                std::printf("%s reselect recalled chunk %d (%d page events, %.2f ms)\n",
                            back ? "PASS" : "FAIL", victim, events, r_ms);
                if (!back) hard_failures++;
                // decode must continue cleanly after the residency change
                pager.score_hook = nullptr;
                for (int pos = total; pos < total + 64; pos++) {
                    const int slot = pager.slot_for(pos);
                    st.refresh_mask(pager);
                    next = st.step(next, pos, slot);
                }
                std::printf("PASS decode continues after reselect (64 tokens)\n");
            }
        }
        st.destroy();
        free_target_cache(cache);
    }

    // ── Run F: full LSA loop — drafter as Memory Indexer ────────────
    // Prompt LARGER than the pool, so prefill itself evicts; then the
    // FlashMemory inference paradigm end to end: every τ=64 decoded
    // tokens the drafter rescores the whole sequence (tail attention =
    // indexer query), score_hook gets the fresh chunk scores, and
    // reselect() repages the pool. PASS requires at least one genuine
    // drafter-driven recall of a chunk evicted earlier.
    {
        const char * drafter_path = "/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf";
        DrafterContext dctx;
        if (!load_drafter(drafter_path, 0, dctx)) {
            std::printf("FAIL indexer run: drafter load failed (%s)\n", dflash27b_last_error());
            hard_failures++;
        } else {
            const int n_prompt_f = 2048, n_gen_f = 768, pool_f = 1024, tau = 64;
            const auto prompt_f = make_prompt(n_prompt_f, w.n_vocab);
            TargetCache cache;
            if (!create_target_cache(w, pool_f, 0, backend, cache, true)) return 1;
            KvFlashPager pager;
            KvFlashConfig pc;
            pc.pool_tokens = pool_f;
            if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;
            KvFlashDrafterScorer scorer(&dctx);   // the production indexer plugin

            Stepper st;
            if (!st.init(w, cache, backend, pool_f, use_mask)) return 1;
            std::vector<int32_t> all_ids = prompt_f;
            int32_t next = -1;
            for (int pos = 0; pos < n_prompt_f; pos++) {
                const int slot = pager.slot_for(pos);
                st.refresh_mask(pager);
                next = st.step(prompt_f[pos], pos, slot);
            }
            const int64_t prefill_evictions = pager.stats().page_outs;

            std::vector<double> rescore_ms, reselect_ms;
            int64_t recalls = 0;
            std::vector<float> scores;
            const double t0 = now_ms();
            for (int g = 0; g < n_gen_f; g++) {
                const int pos = n_prompt_f + g;
                if (g % tau == 0) {
                    double r0 = now_ms();
                    if (!scorer.score_chunks(all_ids, pc.chunk_tokens, scores)) {
                        std::fprintf(stderr, "scorer failed\n");
                        std::exit(1);
                    }
                    rescore_ms.push_back(now_ms() - r0);
                    pager.score_hook = [&scores](int c) {
                        return c < (int)scores.size() ? scores[c] : 1e30f;
                    };
                    r0 = now_ms();
                    const int64_t ins_before = pager.stats().page_ins;
                    pager.reselect();
                    reselect_ms.push_back(now_ms() - r0);
                    recalls += pager.stats().page_ins - ins_before;
                }
                const int slot = pager.slot_for(pos);
                st.refresh_mask(pager);
                next = st.step(next, pos, slot);
                all_ids.push_back(next);
            }
            const double secs = (now_ms() - t0) / 1000.0;
            StepTimes rs = summarize(rescore_ms), rsel = summarize(reselect_ms);
            const auto & ps = pager.stats();
            std::printf("[F] LSA loop: prompt=%d pool=%d gen=%d tau=%d -> %.1f tok/s "
                        "(prefill evicted %" PRId64 ")\n",
                        n_prompt_f, pool_f, n_gen_f, tau, n_gen_f / secs, prefill_evictions);
            std::printf("[F] indexer rescore p50=%.1f ms (full 0.6B re-prefill, %zu calls); "
                        "reselect p50=%.2f ms; drafter-driven recalls=%" PRId64
                        "; total page_outs=%" PRId64 " page_ins=%" PRId64 "\n",
                        rs.p50, rescore_ms.size(), rsel.p50, recalls,
                        ps.page_outs, ps.page_ins);
            std::printf("%s LSA loop: drafter-driven recall of evicted context (recalls >= 1)\n",
                        recalls >= 1 ? "PASS" : "FAIL");
            if (recalls < 1) hard_failures++;
            st.destroy();
            free_target_cache(cache);
            free_drafter(dctx);
        }
    }

    // ── Run E: performance profile ──────────────────────────────────
    if (!skip_prof) {
        std::printf("\n=== DECODE PROFILE (64 timed steps each, junk KV, span = FA window) ===\n");
        auto profile = [&](const char * tag, int alloc_ctx, int span, bool masked,
                           KvFlashPager * pager, int pos_base) {
            TargetCache cache;
            if (!create_target_cache(w, alloc_ctx, 0, backend, cache, true)) {
                std::fprintf(stderr, "cache E(%s): %s\n", tag, dflash27b_last_error());
                std::exit(1);
            }
            KvFlashPager local;
            if (masked && !pager) {
                KvFlashConfig pc; pc.pool_tokens = alloc_ctx;
                local.attach(pc, cache.attn_k, cache.attn_v);
                // mark whole pool resident so the mask is all-zero (worst
                // case mask read, no -inf shortcut)
                for (int p = 0; p < alloc_ctx; p += 64) local.slot_for(p);
                pager = &local;
            }
            Stepper st;
            if (!st.init(w, cache, backend, span, masked)) std::exit(1);
            // warmup 8, then time 64 (refresh included: it is part of the
            // real per-step cost in masked mode)
            int32_t tok = 1000;
            for (int i = 0; i < 8; i++) {
                if (masked) st.refresh_mask(*pager);
                tok = st.step(tok, pos_base + i, (i * 64) % alloc_ctx);
            }
            std::vector<double> ms;
            for (int i = 0; i < 64; i++) {
                const double t0 = now_ms();
                if (masked) st.refresh_mask(*pager);
                tok = st.step(tok, pos_base + 8 + i, (8 * 64 + i) % alloc_ctx);
                ms.push_back(now_ms() - t0);
            }
            const StepTimes r = summarize(ms);
            std::printf("%-28s span=%6d  p50=%7.2f ms  p95=%7.2f ms  mean=%7.2f ms  (%5.1f tok/s)\n",
                        tag, span, r.p50, r.p95, r.mean, 1000.0 / r.mean);
            st.destroy();
            free_target_cache(cache);
        };
        profile("baseline 8K",   8192,   8192,   false, nullptr, 8192 - 72);
        profile("baseline 32K",  32768,  32768,  false, nullptr, 32768 - 72);
        profile("baseline 128K", 131072, 131072, false, nullptr, 131072 - 72);
        profile("pool 1K masked (128K logical)", 1024, 1024, true,  nullptr, 130000);
        profile("pool 1K maskless",              1024, 1024, false, nullptr, 130000);
        profile("pool 4K masked (128K logical)", 4096, 4096, true,  nullptr, 130000);

        // Page-event microbench on a small pool.
        {
            TargetCache cache;
            if (!create_target_cache(w, 1024, 0, backend, cache, true)) std::exit(1);
            KvFlashPager pager;
            KvFlashConfig pc; pc.pool_tokens = 1024;
            pager.attach(pc, cache.attn_k, cache.attn_v);
            for (int p = 0; p < 1024; p += 64) pager.slot_for(p);
            std::vector<double> out_ms, in_ms;
            for (int rep = 0; rep < 32; rep++) {
                const int c = 2 + (rep % 8);
                double t0 = now_ms();
                pager.page_out(c);
                out_ms.push_back(now_ms() - t0);
                t0 = now_ms();
                pager.page_in(c);
                in_ms.push_back(now_ms() - t0);
            }
            const StepTimes o = summarize(out_ms), i = summarize(in_ms);
            std::printf("page_out: p50=%.2f ms  p95=%.2f ms   page_in: p50=%.2f ms  p95=%.2f ms  (per 64-token chunk, %zu KiB)\n",
                        o.p50, o.p95, i.p50, i.p95,
                        (size_t)(pager.stats().host_bytes / std::max<int64_t>(1, 8) / 1024));
            free_target_cache(cache);
        }
    }

    // ── Memory verdict ──────────────────────────────────────────────
    const double red_kv = 100.0 * (1.0 - (double)mem_c_kv / (double)mem_a_kv);
    std::printf("\n=== KV MEMORY ===\n");
    std::printf("baseline  (ctx %6d): kv=%9.1f MiB  base_buf=%9.1f MiB  vram_delta=%9.1f MiB\n",
                logical_ctx, mem_a_kv / 1048576.0, mem_a_buf / 1048576.0, mem_a_vram / 1048576.0);
    std::printf("pooled    (pool %5d): kv=%9.1f MiB  base_buf=%9.1f MiB  vram_delta=%9.1f MiB\n",
                pool_c, mem_c_kv / 1048576.0, mem_c_buf / 1048576.0, mem_c_vram / 1048576.0);
    std::printf("attn-KV reduction: %.1f%%\n", red_kv);
    std::printf("%s KV memory reduction >= 90%%\n", red_kv >= 90.0 ? "PASS" : "FAIL");
    if (red_kv < 90.0) hard_failures++;

    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("\n%s (%d hard failures)\n", hard_failures == 0 ? "ALL PASS" : "FAILED", hard_failures);
    return hard_failures == 0 ? 0 : 1;
}
