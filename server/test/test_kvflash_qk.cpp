// Unit test for the KVFlash target-QK chunk scorer (pure math; no GPU).
// Verifies the Phase-0-validated semantics on synthetic pooled keys:
//   1. ranking: the chunk aligned with the query outranks orthogonal chunks
//   2. max over group q-heads: alignment with ANY q-head in a group counts
//   3. layer-mean: a chunk aligned in 1 of 2 layers scores half of one
//      aligned in both
//   4. cosine: query/key magnitudes do not change scores
//   5. missing pooled keys keep the missing_score sentinel
//
// Phase 2 (snapshot×ledger): seeded fallback scores
//   6. after restore, chunks with no pooled keys but a ledger score use that
//      score (not missing_score=-2.0); chunks WITH pooled keys still use the
//      cosine path; sentinel-valued seeded entries keep missing_score.

#define KVFLASH_QK_PURE_ONLY
#include "kvflash_qk.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash::common;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) std::printf("PASS %s\n", msg); \
    else { std::printf("FAIL %s\n", msg); g_fail++; } \
} while (0)

static bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    KvFlashQkDims d;
    d.n_layers = 2; d.n_q_heads = 6; d.n_kv_heads = 2; d.head_dim = 8;
    const int group = d.n_q_heads / d.n_kv_heads;   // 3
    const size_t kstride = (size_t)d.n_layers * d.n_kv_heads * d.head_dim;
    const size_t qsize   = (size_t)d.n_layers * d.n_q_heads * d.head_dim;

    auto e = [&](int i) {                 // unit basis vector
        std::vector<float> v((size_t)d.head_dim, 0.0f);
        v[(size_t)i] = 1.0f;
        return v;
    };
    auto put_k = [&](std::vector<float> & ks, int l, int h, const std::vector<float> & v) {
        std::copy(v.begin(), v.end(),
                  ks.begin() + ((size_t)l * d.n_kv_heads + h) * d.head_dim);
    };
    auto put_q = [&](std::vector<float> & q, int l, int h, const std::vector<float> & v) {
        std::copy(v.begin(), v.end(),
                  q.begin() + ((size_t)l * d.n_q_heads + h) * d.head_dim);
    };

    // Query: every (layer, q-head) points at basis e0 except layer0/q-head 5
    // (last head of kv-group 1) which points at e1.
    std::vector<float> query(qsize, 0.0f);
    for (int l = 0; l < d.n_layers; l++)
        for (int h = 0; h < d.n_q_heads; h++) put_q(query, l, h, e(0));
    put_q(query, 0, 5, e(1));

    // Chunks (all keys L2-normalized by construction):
    //  c0: keys orthogonal to everything (e2)            -> score ~0... but max
    //      over heads of dot(e0,e2)=0 -> 0 per layer -> 0
    //  c1: kv-head0 = e0 in BOTH layers                  -> 1.0
    //  c2: kv-head0 = e0 in layer0 only, e2 in layer1    -> 0.5
    //  c3: kv-head1 = e1 in layer0 only (matches q-head5 via group max) -> 0.5
    //  c4: missing
    //  c5: same direction as c1 (ranking tie with c1)
    std::vector<std::vector<float>> chunks(6, std::vector<float>(kstride, 0.0f));
    for (int l = 0; l < d.n_layers; l++)
        for (int h = 0; h < d.n_kv_heads; h++) {
            put_k(chunks[0], l, h, e(2));
            put_k(chunks[2], l, h, e(2));
            put_k(chunks[3], l, h, e(2));
        }
    for (int l = 0; l < d.n_layers; l++)
        for (int h = 0; h < d.n_kv_heads; h++) put_k(chunks[1], l, h, e(0));
    put_k(chunks[2], 0, 0, e(0));
    put_k(chunks[3], 0, 1, e(1));
    chunks[5] = chunks[1];

    std::vector<const float *> pk(6, nullptr);
    for (int c = 0; c < 6; c++) if (c != 4) pk[(size_t)c] = chunks[(size_t)c].data();

    std::vector<float> out;
    kvflash_qk_chunk_scores(pk, query.data(), d, out, /*missing=*/-9.0f);

    CHECK(out.size() == 6, "output size matches chunk count");
    CHECK(near(out[0], 0.0f), "orthogonal chunk scores ~0");
    CHECK(near(out[1], 1.0f), "fully aligned chunk scores ~1 (cosine, layer-mean)");
    CHECK(near(out[2], 0.5f), "1-of-2-layer alignment scores ~0.5 (layer-MEAN aggregation)");
    CHECK(near(out[3], 0.5f), "alignment with one group q-head counts (max over group heads)");
    CHECK(near(out[4], -9.0f), "missing pooled keys keep missing_score");
    CHECK(out[1] > out[2] && out[2] > out[0], "ranking: aligned > partial > orthogonal");
    CHECK(near(out[5], out[1]), "identical chunks tie");

    // Default missing_score is worst-case (< -1, below the cosine-mean floor)
    // so a missing chunk never outranks a real chunk with negative correlation.
    std::vector<float> out_def;
    kvflash_qk_chunk_scores(pk, query.data(), d, out_def);   // default missing
    CHECK(out_def[4] < -1.0f, "default missing_score ranks below any real cosine");

    // Cosine invariance: scaling the query must not change anything.
    std::vector<float> query_scaled(query);
    for (float & v : query_scaled) v *= 37.5f;
    std::vector<float> out2;
    kvflash_qk_chunk_scores(pk, query_scaled.data(), d, out2, -9.0f);
    bool same = out2.size() == out.size();
    for (size_t i = 0; same && i < out.size(); i++) same = near(out[i], out2[i]);
    CHECK(same, "query magnitude invariance (cosine)");

    // Mixed-direction key: pooled key = normalize(e0+e1): cos with e0-query
    // = 1/sqrt(2) in that layer.
    {
        std::vector<float> mixed(kstride, 0.0f);
        std::vector<float> k01((size_t)d.head_dim, 0.0f);
        k01[0] = k01[1] = (float)(1.0 / std::sqrt(2.0));
        for (int l = 0; l < d.n_layers; l++) {
            put_k(mixed, l, 0, k01);
            put_k(mixed, l, 1, e(2));
        }
        std::vector<const float *> pk1{ mixed.data() };
        std::vector<float> o1;
        kvflash_qk_chunk_scores(pk1, query.data(), d, o1);
        CHECK(near(o1[0], (float)(1.0 / std::sqrt(2.0))),
              "fractional cosine propagates exactly");
    }

    // ── Phase 2: seeded fallback scores (restore×ledger) ──────────────────
    // After a snapshot restore the QK pool has no entries: pk[c] == nullptr
    // for every chunk.  The seeded[] fallback array carries the per-chunk scores
    // that were live at serialize time.  For chunks without pooled keys, the
    // scorer must use seeded[c] (when it is not the sentinel) instead of
    // missing_score=-2.0.  Chunks WITH pooled keys still go through the cosine
    // path and seeded[] is ignored for them.
    {
        // 4 chunks, 2 have pooled keys (c0, c2), 2 are restore-only (c1, c3).
        // Seeded scores: c1=0.7, c3=-0.3, c0/c2 sentinel (cosine path wins).
        const float kSentinel = -std::numeric_limits<float>::infinity();
        std::vector<float> seeded(4, kSentinel);
        seeded[1] = 0.7f;
        seeded[3] = -0.3f;

        // c0: fully aligned key (cosine=1.0 with the query below)
        // c2: half-aligned (cosine ~0.5)
        KvFlashQkDims d2;
        d2.n_layers = 2; d2.n_q_heads = 2; d2.n_kv_heads = 2; d2.head_dim = 4;
        const size_t kstride2 = (size_t)d2.n_layers * d2.n_kv_heads * d2.head_dim;
        const size_t qsize2   = (size_t)d2.n_layers * d2.n_q_heads * d2.head_dim;

        // Query: all heads / layers point at basis e0
        std::vector<float> q2(qsize2, 0.0f);
        for (size_t i = 0; i < (size_t)d2.n_layers * d2.n_q_heads; i++) {
            q2[i * d2.head_dim + 0] = 1.0f;
        }

        // c0: L2-normalized key e0 in both layers/heads -> cosine=1.0
        std::vector<float> k0(kstride2, 0.0f);
        for (int l = 0; l < d2.n_layers; l++)
            for (int h = 0; h < d2.n_kv_heads; h++)
                k0[((size_t)l * d2.n_kv_heads + h) * d2.head_dim + 0] = 1.0f;

        // c2: L2-normalized key e0 layer0 only, e2 layer1 -> cosine=0.5
        std::vector<float> k2(kstride2, 0.0f);
        k2[((size_t)0 * d2.n_kv_heads + 0) * d2.head_dim + 0] = 1.0f;
        k2[((size_t)0 * d2.n_kv_heads + 1) * d2.head_dim + 0] = 1.0f;
        k2[((size_t)1 * d2.n_kv_heads + 0) * d2.head_dim + 2] = 1.0f;
        k2[((size_t)1 * d2.n_kv_heads + 1) * d2.head_dim + 2] = 1.0f;

        std::vector<const float *> pk2(4, nullptr);
        pk2[0] = k0.data();  // c0: pooled key present
        // c1: no pooled key → seeded fallback 0.7
        pk2[2] = k2.data();  // c2: pooled key present
        // c3: no pooled key → seeded fallback -0.3

        std::vector<float> out2;
        kvflash_qk_chunk_scores(pk2, q2.data(), d2, out2,
                                /*missing_score=*/-2.0f,
                                seeded.data(), kSentinel,
                                /*seeded_n=*/(int)seeded.size());

        CHECK(out2.size() == 4, "phase2: output size == 4");
        CHECK(near(out2[0], 1.0f),  "phase2: c0 pooled key -> cosine 1.0 (seeded ignored)");
        CHECK(near(out2[1], 0.7f),  "phase2: c1 no pooled key -> seeded 0.7 (not -2.0)");
        CHECK(near(out2[2], 0.5f),  "phase2: c2 pooled key -> cosine 0.5 (seeded ignored)");
        CHECK(near(out2[3], -0.3f), "phase2: c3 no pooled key -> seeded -0.3 (not -2.0)");

        // A seeded sentinel must fall through to missing_score.
        std::vector<float> all_sentinel(4, kSentinel);
        std::vector<float> out3;
        kvflash_qk_chunk_scores(pk2, q2.data(), d2, out3,
                                /*missing_score=*/-2.0f,
                                all_sentinel.data(), kSentinel,
                                /*seeded_n=*/(int)all_sentinel.size());
        CHECK(near(out3[1], -2.0f), "phase2: seeded=sentinel -> missing_score");
        CHECK(near(out3[3], -2.0f), "phase2: seeded=sentinel -> missing_score (c3)");

        // Null seeded ptr must behave like the original (missing_score).
        std::vector<float> out4;
        kvflash_qk_chunk_scores(pk2, q2.data(), d2, out4,
                                /*missing_score=*/-2.0f,
                                /*seeded=*/nullptr, kSentinel,
                                /*seeded_n=*/0);
        CHECK(near(out4[1], -2.0f), "phase2: null seeded ptr -> missing_score");

        // ── Phase 2b (P1 fix): seeded_n bounds the seeded read ──────────────
        // A restored ledger can be SHORTER than the current n_chunks (pool grew
        // after restore). seeded_n must clamp the read so chunks at/above
        // seeded_n fall back to missing_score instead of reading past the buffer.
        // RED (pre-fix): no seeded_n → reads seeded[3..5] out of bounds → UB.
        // GREEN (post-fix): out[3..5] == missing_score.
        {
            // 6 chunks, none pooled; ledger only covered the first 3.
            std::vector<float> seeded_short = { 0.5f, 0.6f, 0.7f }; // length 3
            std::vector<const float *> pk6(6, nullptr);
            std::vector<float> out6;
            kvflash_qk_chunk_scores(pk6, q2.data(), d2, out6,
                                    /*missing_score=*/-2.0f,
                                    seeded_short.data(), kSentinel,
                                    /*seeded_n=*/(int)seeded_short.size());
            CHECK(out6.size() == 6, "phase2b: output size == 6");
            CHECK(near(out6[0], 0.5f), "phase2b: c0 seeded 0.5");
            CHECK(near(out6[1], 0.6f), "phase2b: c1 seeded 0.6");
            CHECK(near(out6[2], 0.7f), "phase2b: c2 seeded 0.7");
            CHECK(near(out6[3], -2.0f), "phase2b: c3 beyond seeded_n -> missing_score (no OOB)");
            CHECK(near(out6[4], -2.0f), "phase2b: c4 beyond seeded_n -> missing_score (no OOB)");
            CHECK(near(out6[5], -2.0f), "phase2b: c5 beyond seeded_n -> missing_score (no OOB)");
        }
    }

    std::printf("%s (%d failures)\n", g_fail == 0 ? "ALL PASS" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
