// Unit test for the KVFlash target-QK chunk scorer (pure math; no GPU).
// Verifies the Phase-0-validated semantics on synthetic pooled keys:
//   1. ranking: the chunk aligned with the query outranks orthogonal chunks
//   2. max over group q-heads: alignment with ANY q-head in a group counts
//   3. layer-mean: a chunk aligned in 1 of 2 layers scores half of one
//      aligned in both
//   4. cosine: query/key magnitudes do not change scores
//   5. missing pooled keys keep the missing_score sentinel

#include "CppUnitTestFramework.hpp"

#define KVFLASH_QK_PURE_ONLY
#include "kvflash_qk.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash::common;

static bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

namespace {
struct KvflashQkFixture {};
}

TEST_CASE(KvflashQkFixture, kvflash_qk_suite) {
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

    CHECK(out.size() == 6);
    CHECK(near(out[0], 0.0f));
    CHECK(near(out[1], 1.0f));
    CHECK(near(out[2], 0.5f));
    CHECK(near(out[3], 0.5f));
    CHECK(near(out[4], -9.0f));
    CHECK(out[1] > out[2] && out[2] > out[0]);
    CHECK(near(out[5], out[1]));

    // Default missing_score is worst-case (< -1, below the cosine-mean floor)
    // so a missing chunk never outranks a real chunk with negative correlation.
    std::vector<float> out_def;
    kvflash_qk_chunk_scores(pk, query.data(), d, out_def);   // default missing
    CHECK(out_def[4] < -1.0f);

    // Cosine invariance: scaling the query must not change anything.
    std::vector<float> query_scaled(query);
    for (float & v : query_scaled) v *= 37.5f;
    std::vector<float> out2;
    kvflash_qk_chunk_scores(pk, query_scaled.data(), d, out2, -9.0f);
    bool same = out2.size() == out.size();
    for (size_t i = 0; same && i < out.size(); i++) same = near(out[i], out2[i]);
    CHECK(same);

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
        CHECK(near(o1[0], (float)(1.0 / std::sqrt(2.0))));
    }

    std::printf("ALL PASS\n");
}
