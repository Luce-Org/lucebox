// GPU unit tests for Inkling2KvFlash (3-Tier Dual-GPU KVFlash 2.0 + Mode-4 Exact D=128 Recall).
//
// Tests:
//   1. 3-Tier disjoint block tables (Tier 0 + Tier 1.5 + Tier 2 Host), bit-exact Tier 1.5 migration,
//      overflow to Tier 2 Host, and bit-exact recall back into Tier 0.
//   2. Mode-4 exact D=128 needle scoring & recall on Q8_0 (fused NBLK=4 CT=64 GPU kernel).
//   3. Mode-4 exact D=128 needle scoring & recall on F16 (stock ggml logits graph).
//   4. Shared-KV (W=64) strided nb[2] layout: verifies pager DMA & fused Mode-4 scorer respect
//      tail-ring stride without corrupting the packed [pool_cap .. pool_cap + tail) ring bytes.

#include "inkling2/inkling2_kvflash.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace luce::common;

namespace {

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const char * name, const char * detail = "") {
    if (ok) {
        std::printf("[PASS] %s %s\n", name, detail);
        ++g_pass;
    } else {
        std::printf("[FAIL] %s %s\n", name, detail);
        ++g_fail;
    }
}

static void encode_row(ggml_type type, const float * src, int D, uint8_t * dst) {
    if (type == GGML_TYPE_Q8_0) {
        ggml_quantize_chunk(GGML_TYPE_Q8_0, src, dst, 0, 1, D, nullptr);
    } else {
        ggml_fp32_to_fp16_row(src, reinterpret_cast<ggml_fp16_t *>(dst), D);
    }
}

// Write one 64-token chunk `c` into its Tier-0 physical block `b` across all L layers & Hk heads.
// Also records the exact encoded bytes per (l, h) into `expected_k[c]` / `expected_v[c]` for bit-exact checks.
static void write_chunk_to_tier0(
        const std::vector<ggml_tensor *> & t0_k,
        const std::vector<ggml_tensor *> & t0_v,
        int b,
        int ct,
        int D,
        int Hk,
        const std::vector<float> & k_vec, // [D] template key for this chunk
        const std::vector<float> & v_vec, // [D] template value for this chunk
        int salt,
        std::vector<uint8_t> & out_k_bytes,
        std::vector<uint8_t> & out_v_bytes) {
    const int L = (int) t0_k.size();
    const ggml_type kt = t0_k[0]->type;
    const size_t row_b = ggml_row_size(kt, D);
    const size_t seg_b = (size_t) ct * row_b;
    out_k_bytes.resize((size_t) L * Hk * seg_b);
    out_v_bytes.resize((size_t) L * Hk * seg_b);

    std::vector<float> row_f(D);
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < Hk; ++h) {
            uint8_t * kp = out_k_bytes.data() + ((size_t) l * Hk + h) * seg_b;
            uint8_t * vp = out_v_bytes.data() + ((size_t) l * Hk + h) * seg_b;
            for (int t = 0; t < ct; ++t) {
                const float tweak = 1.0f + 0.001f * (float) ((t + h + l + salt) % 17);
                for (int d = 0; d < D; ++d) {
                    row_f[d] = k_vec[d] * tweak;
                }
                encode_row(kt, row_f.data(), D, kp + (size_t) t * row_b);
                for (int d = 0; d < D; ++d) {
                    row_f[d] = v_vec[d] * tweak;
                }
                encode_row(kt, row_f.data(), D, vp + (size_t) t * row_b);
            }
            const size_t off_k = (size_t) b * ct * t0_k[l]->nb[1] + (size_t) h * t0_k[l]->nb[2];
            const size_t off_v = (size_t) b * ct * t0_v[l]->nb[1] + (size_t) h * t0_v[l]->nb[2];
            ggml_backend_tensor_set(t0_k[l], kp, off_k, seg_b);
            ggml_backend_tensor_set(t0_v[l], vp, off_v, seg_b);
        }
    }
}

static bool verify_chunk_bytes(
        const std::vector<ggml_tensor *> & pool_k,
        const std::vector<ggml_tensor *> & pool_v,
        int b,
        int ct,
        int D,
        int Hk,
        const std::vector<uint8_t> & exp_k,
        const std::vector<uint8_t> & exp_v) {
    const int L = (int) pool_k.size();
    const size_t seg_b = (size_t) ct * ggml_row_size(pool_k[0]->type, D);
    std::vector<uint8_t> got(seg_b);
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < Hk; ++h) {
            const size_t off_k = (size_t) b * ct * pool_k[l]->nb[1] + (size_t) h * pool_k[l]->nb[2];
            const size_t off_v = (size_t) b * ct * pool_v[l]->nb[1] + (size_t) h * pool_v[l]->nb[2];
            ggml_backend_tensor_get(pool_k[l], got.data(), off_k, seg_b);
            if (std::memcmp(got.data(), exp_k.data() + ((size_t) l * Hk + h) * seg_b, seg_b) != 0) return false;
            ggml_backend_tensor_get(pool_v[l], got.data(), off_v, seg_b);
            if (std::memcmp(got.data(), exp_v.data() + ((size_t) l * Hk + h) * seg_b, seg_b) != 0) return false;
        }
    }
    return true;
}

static void test_3tier_disjoint_and_eviction(ggml_backend_t be) {
    constexpr int L = 4, D = 128, Hq = 24, Hk = 8, CT = 64;
    constexpr int T0_TOK = 512;   // 8 blocks
    constexpr int T15_TOK = 384;  // 6 blocks (simulated on same backend for single-GPU CI!)
    constexpr int MAX_CTX = 1280; // 20 chunks

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    std::vector<ggml_tensor *> t0_k(L), t0_v(L), t15_k(L), t15_v(L);
    for (int l = 0; l < L; ++l) {
        t0_k[l]  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, T0_TOK, Hk);
        t0_v[l]  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, T0_TOK, Hk);
        t15_k[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, T15_TOK, Hk);
        t15_v[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, T15_TOK, Hk);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    Inkling2KvFlashConfig cfg;
    cfg.enabled            = true;
    cfg.chunk_tokens       = CT;
    cfg.tier0_pool_tokens  = T0_TOK;
    cfg.tier15_pool_tokens = T15_TOK;
    cfg.sink_chunks        = 1;
    cfg.hot_tail_chunks    = 2;
    cfg.recall_blocks      = 2;
    cfg.select_unit        = 1;
    cfg.mode4_enabled      = true;

    Inkling2KvFlash kf;
    const bool attached = kf.attach(cfg, MAX_CTX, D, Hq, Hk, be, be, t0_k, t0_v, t15_k, t15_v);

    std::vector<std::vector<uint8_t>> exp_k(20), exp_v(20);
    bool span_ok = true;
    bool disjoint_ok = true;
    for (int c = 0; c < 20; ++c) {
        if (!kf.prepare_span(c * CT, CT)) { span_ok = false; break; }
        const int b0 = kf.bt_local()[c];
        if (b0 < 0) { span_ok = false; break; }
        std::vector<float> kv(D), vv(D);
        for (int d = 0; d < D; ++d) {
            kv[d] = std::sin((float) (c * 13 + d) * 0.17f);
            vv[d] = std::cos((float) (c * 7 + d) * 0.23f);
        }
        write_chunk_to_tier0(t0_k, t0_v, b0, CT, D, Hk, kv, vv, c, exp_k[c], exp_v[c]);

        // Verify strict disjointness after every chunk write!
        for (int i = 0; i < kf.max_blocks(); ++i) {
            if (kf.bt_local()[i] >= 0 && kf.bt_remote()[i] >= 0) disjoint_ok = false;
        }
    }

    // Verify resident Tier 0 (8 blocks), Tier 1.5 (6 blocks), and Tier 2 Host-only (6 blocks)
    int n_t0 = 0, n_t15 = 0, n_t2 = 0;
    bool t0_bytes_ok = true, t15_bytes_ok = true;
    for (int c = 0; c < 20; ++c) {
        const int b0 = kf.bt_local()[c];
        const int b1 = kf.bt_remote()[c];
        if (b0 >= 0) {
            ++n_t0;
            if (!verify_chunk_bytes(t0_k, t0_v, b0, CT, D, Hk, exp_k[c], exp_v[c])) t0_bytes_ok = false;
        } else if (b1 >= 0) {
            ++n_t15;
            if (!verify_chunk_bytes(t15_k, t15_v, b1, CT, D, Hk, exp_k[c], exp_v[c])) t15_bytes_ok = false;
        } else {
            ++n_t2;
        }
    }

    char detail[160];
    std::snprintf(detail, sizeof(detail), "(t0=%d t15=%d t2=%d disjoint=%d t0_exact=%d t15_exact=%d)",
                  n_t0, n_t15, n_t2, disjoint_ok ? 1 : 0, t0_bytes_ok ? 1 : 0, t15_bytes_ok ? 1 : 0);
    check(attached && span_ok && disjoint_ok && n_t0 == 8 && n_t15 == 6 && n_t2 == 6 && t0_bytes_ok && t15_bytes_ok,
          "3tier_disjoint_and_eviction", detail);

    kf.detach();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void test_mode4_needle(ggml_backend_t be, ggml_type kv_type, bool strided_shared_kv, const char * label) {
    constexpr int L = 4, D = 128, Hq = 24, Hk = 8, CT = 64;
    constexpr int T0_TOK = 512;   // 8 chunks resident in Tier 0
    constexpr int TAIL_PAD = 256; // 2 * shared_tail_cap when testing Shared-KV W=64 strided layout!
    constexpr int N_CHUNKS = 16;  // 1024 tokens total -> 8 chunks evicted to Tier 2 Host!
    constexpr int MAX_CTX = N_CHUNKS * CT;
    constexpr int C_NEEDLE = 4;   // Early cold chunk that gets evicted by LRU!

    const int alloc_rows = strided_shared_kv ? (T0_TOK + TAIL_PAD) : T0_TOK;
    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    std::vector<ggml_tensor *> raw_k(L), raw_v(L), view_k(L), view_v(L);
    for (int l = 0; l < L; ++l) {
        raw_k[l] = ggml_new_tensor_3d(ctx, kv_type, D, alloc_rows, Hk);
        raw_v[l] = ggml_new_tensor_3d(ctx, kv_type, D, alloc_rows, Hk);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    // Fill the Shared-KV tail ring [T0_TOK .. T0_TOK + TAIL_PAD) with a canary pattern to verify zero corruption!
    const size_t row_b = ggml_row_size(kv_type, D);
    std::vector<uint8_t> canary((size_t) TAIL_PAD * row_b, 0xA5);
    if (strided_shared_kv) {
        for (int l = 0; l < L; ++l) {
            for (int h = 0; h < Hk; ++h) {
                const size_t off = (size_t) T0_TOK * raw_k[l]->nb[1] + (size_t) h * raw_k[l]->nb[2];
                ggml_backend_tensor_set(raw_k[l], canary.data(), off, canary.size());
                ggml_backend_tensor_set(raw_v[l], canary.data(), off, canary.size());
            }
        }
    }

    for (int l = 0; l < L; ++l) {
        view_k[l] = ggml_new_tensor_3d(ctx, kv_type, D, T0_TOK, Hk);
        view_k[l]->nb[0] = raw_k[l]->nb[0]; view_k[l]->nb[1] = raw_k[l]->nb[1]; view_k[l]->nb[2] = raw_k[l]->nb[2];
        view_k[l]->data = raw_k[l]->data; view_k[l]->buffer = raw_k[l]->buffer;
        view_v[l] = ggml_new_tensor_3d(ctx, kv_type, D, T0_TOK, Hk);
        view_v[l]->nb[0] = raw_v[l]->nb[0]; view_v[l]->nb[1] = raw_v[l]->nb[1]; view_v[l]->nb[2] = raw_v[l]->nb[2];
        view_v[l]->data = raw_v[l]->data; view_v[l]->buffer = raw_v[l]->buffer;
    }

    Inkling2KvFlashConfig cfg;
    cfg.enabled            = true;
    cfg.chunk_tokens       = CT;
    cfg.tier0_pool_tokens  = T0_TOK;
    cfg.tier15_pool_tokens = 0;
    cfg.sink_chunks        = 1;
    cfg.hot_tail_chunks    = 2;
    cfg.recall_blocks      = 2;
    cfg.select_unit        = 1;
    cfg.select_queries     = 4;
    cfg.mode4_enabled      = true;

    Inkling2KvFlash kf;
    const bool attached = kf.attach(cfg, MAX_CTX, D, Hq, Hk, be, nullptr, view_k, view_v);

    // Plant a strong unit-aligned needle key in chunk C_NEEDLE=4; orthogonal small noise elsewhere
    std::vector<float> k_needle(D, 0.0f), k_noise(D, 0.0f), v_dummy(D, 0.25f);
    for (int d = 0; d < 32; ++d) k_needle[d] = 3.0f;
    for (int d = 64; d < 96; ++d) k_noise[d] = 0.5f;

    std::vector<std::vector<uint8_t>> exp_k(N_CHUNKS), exp_v(N_CHUNKS);
    for (int c = 0; c < N_CHUNKS; ++c) {
        kf.prepare_span(c * CT, CT);
        const int b0 = kf.bt_local()[c];
        const auto & kv = (c == C_NEEDLE) ? k_needle : k_noise;
        write_chunk_to_tier0(view_k, view_v, b0, CT, D, Hk, kv, v_dummy, c, exp_k[c], exp_v[c]);
    }

    // Confirm needle chunk was evicted to Tier 2 Host by LRU!
    const bool was_evicted = (kf.bt_local()[C_NEEDLE] < 0);

    // Capture 4 queries strongly aligned with k_needle across all L * Hq planes
    const size_t qsize = (size_t) L * Hq * D;
    std::vector<float> q_rows(4 * qsize, 0.0f);
    for (int q = 0; q < 4; ++q) {
        for (int p = 0; p < L * Hq; ++p) {
            float * qp = q_rows.data() + ((size_t) q * L * Hq + p) * D;
            for (int d = 0; d < 32; ++d) qp[d] = 2.0f;
        }
    }
    kf.append_queries(q_rows.data(), 4);

    // Run Mode-4 exact scoring + recall at rewind_pos = 15 * 64 = 960, prompt_end = 1024
    Inkling2KvFlashRecallStats st;
    const bool recalled = kf.mode4_score_and_recall(15 * CT, MAX_CTX, &st);

    // Check needle ranked #1 among sealed chunks [0..15)
    int best_c = -1;
    float best_s = -1.0f;
    for (int c = 0; c < (int) kf.last_scores().size(); ++c) {
        if (kf.last_scores()[c] > best_s) {
            best_s = kf.last_scores()[c];
            best_c = c;
        }
    }
    const int b_after = kf.bt_local()[C_NEEDLE];
    const bool needle_resident = (b_after >= 0);
    const bool needle_exact = needle_resident &&
        verify_chunk_bytes(view_k, view_v, b_after, CT, D, Hk, exp_k[C_NEEDLE], exp_v[C_NEEDLE]);

    // Also verify canary tail ring bytes are 100% untouched!
    bool canary_ok = true;
    if (strided_shared_kv) {
        std::vector<uint8_t> got_c(canary.size());
        for (int l = 0; l < L; ++l) {
            for (int h = 0; h < Hk; ++h) {
                const size_t off = (size_t) T0_TOK * raw_k[l]->nb[1] + (size_t) h * raw_k[l]->nb[2];
                ggml_backend_tensor_get(raw_k[l], got_c.data(), off, got_c.size());
                if (std::memcmp(got_c.data(), canary.data(), canary.size()) != 0) canary_ok = false;
                ggml_backend_tensor_get(raw_v[l], got_c.data(), off, got_c.size());
                if (std::memcmp(got_c.data(), canary.data(), canary.size()) != 0) canary_ok = false;
            }
        }
    }

    const bool want_fused = (kv_type == GGML_TYPE_Q8_0);
    const bool fused_ok = !want_fused || st.used_fused;
    const bool picked_needle = !st.picked_blocks.empty() && st.picked_blocks[0] == C_NEEDLE;

    char detail[192];
    std::snprintf(detail, sizeof(detail),
                  "(evicted=%d gpu=%d fused=%d picked0=%d top1=c%d(%.3f) recalled_blk=%d exact=%d canary=%d %.1fms)",
                  was_evicted ? 1 : 0, st.used_gpu_scorer ? 1 : 0, st.used_fused ? 1 : 0,
                  st.picked_blocks.empty() ? -1 : st.picked_blocks[0],
                  best_c, best_s, b_after, needle_exact ? 1 : 0, canary_ok ? 1 : 0, st.score_ms);
    check(attached && was_evicted && recalled && st.used_gpu_scorer && fused_ok && picked_needle &&
          best_c == C_NEEDLE && best_s > 0.90f && needle_resident && needle_exact && canary_ok,
          label, detail);

    kf.detach();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

} // namespace

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) {
        std::fprintf(stderr, "No GPU backend available; skipping test_inkling2_kvflash\n");
        return 77;
    }

    test_3tier_disjoint_and_eviction(be);
    test_mode4_needle(be, GGML_TYPE_Q8_0, /*strided_shared_kv=*/false, "mode4_exact_needle_q8_0_fused");
    test_mode4_needle(be, GGML_TYPE_F16,  /*strided_shared_kv=*/false, "mode4_exact_needle_f16_graph");
    test_mode4_needle(be, GGML_TYPE_Q8_0, /*strided_shared_kv=*/true,  "mode4_shared_kv_strided_nb2");

    ggml_backend_free(be);
    std::printf("\n=== test_inkling2_kvflash: %d/%d PASSED ===\n", g_pass, g_pass + g_fail);
    return (g_fail == 0) ? 0 : 1;
}
