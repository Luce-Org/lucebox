// Selector-v2 GPU scorer (kvflash_gpu_mass.h) against the AVX2 CPU scorer, on the ggml CPU backend so ctest needs no
// GPU. Same graph, same pool / staging / bias logic; only the backend differs. Pass: identical top-scoring chunks and
// select-score relative error < 1e-2 against kvflash_exact_mass_scores_fast_multi.

#include "CppUnitTestFramework.hpp"

#include "kvflash_exact_fast.h"
#include "kvflash_gpu_mass.h"
#include "kvflash_mass_synth.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashGpuMassFixture {};

struct Outcome {
    bool ok = false;
    std::string why;
    double sel_rel = 0, mean_rel = 0, top_rel = 0, max_rel = 0;
    bool picks_same = false;
    size_t n_picks = 0;
};

// Small Qwen-like geometry: 2 layers, 2 kv heads x 3 q heads, D 64, 16-token chunks, a 24-block pool.
SynthCfg small_cfg(ggml_type kt, int sealed, int nq, int keep_newest = -1) {
    SynthCfg c;
    c.L = 2; c.HK = 2; c.G = 3; c.D = 64; c.CT = 16; c.NB = 24;
    c.KT = kt; c.sealed = sealed; c.N = nq; c.keep_newest = keep_newest; c.tail = 16;
    return c;
}

// `scored` chunks are scored out of a synth built with cfg.sealed >= scored: the chunks beyond stay in their pool
// blocks (the server's chunks >= sealed) and must be masked. host_ok=false makes host_seg return null.
Outcome run(const SynthCfg & cfg, int scored) {
    Outcome o;
    Synth s;
    s.build(cfg);
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { o.why = "no cpu backend"; return o; }
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(cfg.L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < cfg.L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, cfg.KT, cfg.D, (int64_t)cfg.NB * cfg.CT, cfg.HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    for (int l = 0; l < cfg.L; l++) ggml_backend_tensor_set(pool_k[(size_t)l], s.pool_img[(size_t)l].data(), 0, ggml_nbytes(pool_k[(size_t)l]));

    // CPU oracle over the first `scored` chunks.
    KvFlashExactKeys keys;
    keys.n_chunks = scored; keys.chunk_tokens = cfg.CT; keys.type = cfg.KT;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * { return s.blk[(size_t)c] >= 0 ? s.pool_seg(s.blk[(size_t)c], l, h) : s.host_seg(c, l, h); };
    const auto qp = s.qptrs();
    std::vector<std::vector<float>> outs_cpu, outs_gpu;
    std::vector<KvFlashMassPlanes> planes_cpu, planes_gpu;
    kvflash_exact_mass_scores_fast_multi(keys, s.dims(), qp, outs_cpu, 0, nullptr, &planes_cpu);

    KvFlashGpuMass gm(be);
    std::vector<int> block_of(s.blk.begin(), s.blk.begin() + scored);
    KvFlashGpuMassTiming tm;
    o.ok = gm.score(pool_k, block_of, [&](int c, int l, int h) { return s.host_seg(c, l, h); }, cfg.CT, s.dims(), qp, outs_gpu, &planes_gpu, &tm, &o.why);
    if (o.ok) {
        const auto sel_cpu = kvflash_select_scores(planes_cpu), sel_gpu = kvflash_select_scores(planes_gpu);
        auto rel = [](const std::vector<float> & a, const std::vector<float> & b) {
            double m = 0;
            for (size_t c = 0; c < a.size(); c++) if (a[c] > 1e-9) m = std::max(m, (double)std::fabs(a[c] - b[c]) / a[c]);
            return m;
        };
        o.sel_rel = rel(sel_cpu, sel_gpu);
        for (int n = 0; n < cfg.N; n++) {
            o.mean_rel = std::max(o.mean_rel, rel(outs_cpu[(size_t)n], outs_gpu[(size_t)n]));
            o.top_rel = std::max(o.top_rel, rel(planes_cpu[(size_t)n].top, planes_gpu[(size_t)n].top));
            o.max_rel = std::max(o.max_rel, rel(planes_cpu[(size_t)n].max, planes_gpu[(size_t)n].max));
        }
        // Take as many top chunks as there are planted chunks the queries cover below the window: the rest is noise.
        const int fwc = scored - cfg.tail;
        int k = 0;
        for (int j = 0; j < s.n_hot && j < cfg.N; j++) k += 10 + 7 * j + 1 < fwc;
        k = std::max(1, k);
        auto top = [&](const std::vector<float> & sc) {
            std::vector<int> ids((size_t)std::max(fwc, 0));
            for (int c = 0; c < fwc; c++) ids[(size_t)c] = c;
            std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return sc[(size_t)a] > sc[(size_t)b]; });
            ids.resize((size_t)std::min<int>(k, (int)ids.size()));
            std::sort(ids.begin(), ids.end());
            return ids;
        };
        const auto pc = top(sel_cpu), pg = top(sel_gpu);
        o.picks_same = pc == pg;
        o.n_picks = pc.size();
    }
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
    return o;
}

}  // namespace

// The framework's asserts need the test object, so this is a macro, not a helper.
#define expect_parity(expr) do { \
    const Outcome o_ = (expr); \
    if (!o_.ok) std::fprintf(stderr, "gpu mass: %s\n", o_.why.c_str()); \
    REQUIRE(o_.ok); \
    CHECK(o_.picks_same); \
    CHECK(o_.n_picks > 0); \
    CHECK(o_.sel_rel < 1e-2); \
    CHECK(o_.mean_rel < 1e-2); \
    CHECK(o_.top_rel < 1e-2); \
} while (0)

TEST_CASE(KvflashGpuMassFixture, q4_0_mixed_residency_matches_cpu) { expect_parity(run(small_cfg(GGML_TYPE_Q4_0, 60, 8), 60)); }
TEST_CASE(KvflashGpuMassFixture, q8_0_mixed_residency_matches_cpu) { expect_parity(run(small_cfg(GGML_TYPE_Q8_0, 60, 8), 60)); }
TEST_CASE(KvflashGpuMassFixture, all_resident_matches_cpu) {
    SynthCfg c = small_cfg(GGML_TYPE_Q4_0, 36, 8);   // 36 chunks in a 40-block pool: nothing evicted, 4 blocks free
    c.NB = 40; c.tail = 8;
    expect_parity(run(c, 36));
}
TEST_CASE(KvflashGpuMassFixture, all_evicted_but_sinks_matches_cpu) { expect_parity(run(small_cfg(GGML_TYPE_Q4_0, 60, 8, 0), 60)); }
TEST_CASE(KvflashGpuMassFixture, fewer_queries_than_64_matches_cpu) {
    expect_parity(run(small_cfg(GGML_TYPE_Q4_0, 60, 3), 60));
    expect_parity(run(small_cfg(GGML_TYPE_Q8_0, 60, 1), 60));
}

// Free blocks hold loud junk keys; they and chunks >= the scored count must get -inf.
TEST_CASE(KvflashGpuMassFixture, free_pool_blocks_are_masked) {
    SynthCfg c = small_cfg(GGML_TYPE_Q4_0, 30, 8);   // 30 chunks, 24 blocks: sinks + 22 newest resident, 0 free
    c.NB = 40; c.junk_chunks = 1; c.tail = 8;        // 40 blocks: 10 free ones with junk
    expect_parity(run(c, 30));
}
TEST_CASE(KvflashGpuMassFixture, pool_blocks_holding_chunks_beyond_sealed_are_masked) {
    // 60 chunks laid out, only the first 52 sealed: the 8 newest are resident in the pool and must not score.
    expect_parity(run(small_cfg(GGML_TYPE_Q4_0, 60, 8), 52));
    expect_parity(run(small_cfg(GGML_TYPE_Q8_0, 60, 8), 44));
}

TEST_CASE(KvflashGpuMassFixture, reuse_across_calls_and_growth) {
    // One module, a small call then a larger one then the small again: buffers grow, results stay right.
    Synth s;
    SynthCfg c = small_cfg(GGML_TYPE_Q4_0, 60, 8);
    s.build(c);
    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(c.L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < c.L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, c.KT, c.D, (int64_t)c.NB * c.CT, c.HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    for (int l = 0; l < c.L; l++) ggml_backend_tensor_set(pool_k[(size_t)l], s.pool_img[(size_t)l].data(), 0, ggml_nbytes(pool_k[(size_t)l]));
    KvFlashGpuMass gm(be);
    std::vector<std::vector<float>> first, again, mid;
    std::vector<KvFlashMassPlanes> pl;
    auto host = [&](int ch, int l, int h) { return s.host_seg(ch, l, h); };
    const auto qp = s.qptrs();
    std::string why;
    std::vector<int> b40(s.blk.begin(), s.blk.begin() + 40), b60 = s.blk;
    REQUIRE(gm.score(pool_k, b40, host, c.CT, s.dims(), qp, first, &pl, nullptr, &why));
    REQUIRE(gm.score(pool_k, b60, host, c.CT, s.dims(), qp, mid, &pl, nullptr, &why));
    REQUIRE(gm.score(pool_k, b40, host, c.CT, s.dims(), qp, again, &pl, nullptr, &why));
    CHECK(first == again);
    CHECK(mid[0].size() == 60u);
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
}

TEST_CASE(KvflashGpuMassFixture, bad_inputs_fail_without_crashing) {
    Synth s;
    SynthCfg c = small_cfg(GGML_TYPE_Q4_0, 30, 2);
    s.build(c);
    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(c.L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < c.L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, c.KT, c.D, (int64_t)c.NB * c.CT, c.HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    KvFlashGpuMass gm(be);
    std::vector<std::vector<float>> o;
    std::string why;
    auto host = [&](int ch, int l, int h) { return s.host_seg(ch, l, h); };
    auto missing = [&](int, int, int) -> const uint8_t * { return nullptr; };
    std::vector<int> bo = s.blk;
    CHECK(!gm.score(pool_k, {}, host, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));            // nothing sealed
    bo[3] = c.NB;                                                                                          // outside the pool
    CHECK(!gm.score(pool_k, bo, host, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));
    bo = s.blk; bo[12] = bo[13];                                                                           // two chunks, one block
    CHECK(!gm.score(pool_k, bo, host, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));
    std::vector<int> ev(30, -1);                                                                           // all evicted, host copy absent
    CHECK(!gm.score(pool_k, ev, missing, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));
    CHECK(why.find("host K") != std::string::npos);
    CHECK(gm.score(pool_k, s.blk, host, c.CT, s.dims(), {}, o, nullptr, nullptr, &why));                  // no queries
    CHECK(o.empty());
    // A working set far over free memory is refused before anything is allocated. Batching bounds the graph, so only the
    // queries (12 GB at 4M) count: run the call only where fits_now says it does not fit (a big-RAM host does fit it).
    std::vector<const float *> many((size_t)4000000, s.qs.data());
    int n_ev = 0;
    for (int b : s.blk) n_ev += b < 0;
    if (!gm.fits_now(s.dims(), c.KT, c.CT, c.NB, n_ev, (int)many.size())) {
        CHECK(!gm.score(pool_k, s.blk, host, c.CT, s.dims(), many, o, nullptr, nullptr, &why));
        CHECK(why.find("free") != std::string::npos);
    }
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
}

TEST_CASE(KvflashGpuMassFixture, chunk_of_maps_blocks_and_marks_the_rest_masked) {
    std::vector<int> ev;
    // pool of 5 blocks; chunks 0..5: 0->3, 1 evicted, 2->0, 3 evicted, 4->4, 5 evicted. Blocks 1, 2 hold nothing sealed.
    const auto m = kvflash_gpu_mass_chunk_of({3, -1, 0, -1, 4, -1}, 5, ev);
    REQUIRE(m.size() == 8u);
    CHECK(m[0] == 2); CHECK(m[1] == -1); CHECK(m[2] == -1); CHECK(m[3] == 0); CHECK(m[4] == 4);
    CHECK(m[5] == 1); CHECK(m[6] == 3); CHECK(m[7] == 5);
    CHECK(ev == std::vector<int>({1, 3, 5}));
    CHECK(kvflash_gpu_mass_chunk_of({5}, 5, ev).empty());
}

// Bytes one query keeps live in a batched graph (the formula behind kvflash_gpu_mass_batch).
static size_t per_query_bytes(const KvFlashQkDims & d, int ct, int nbt) {
    return (3 * (size_t)nbt * ct * d.n_q_heads + (size_t)d.n_layers * d.n_q_heads * nbt) * 4;
}

TEST_CASE(KvflashGpuMassFixture, batch_plan_is_balanced_and_within_budget) {
    KvFlashQkDims d; d.n_layers = 16; d.n_q_heads = 24; d.n_kv_heads = 4; d.head_dim = 256;
    const size_t budget = kKvFlashGpuMassBatchBytes;
    CHECK(kvflash_gpu_mass_batch(d, 64, 2062, 128, budget) == 13);   // 10 batches
    CHECK(kvflash_gpu_mass_batch(d, 64, 530, 128, budget) == 43);    // 3 batches
    CHECK(kvflash_gpu_mass_batch(d, 64, 1, 128, budget) == 128);     // tiny T: one batch
    CHECK(kvflash_gpu_mass_batch(d, 64, 2062, 128, 1) == 1);         // budget below one query
    CHECK(kvflash_gpu_mass_batch(d, 64, 2062, 0, budget) == 0);
    for (int nq : {1, 7, 64, 128, 129, 500})
        for (size_t mib : {1, 16, 100, 512, 4096}) {
            const int nb = kvflash_gpu_mass_batch(d, 64, 700, nq, (size_t)mib << 20);
            const size_t pq = per_query_bytes(d, 64, 700);
            REQUIRE(nb >= 1);
            CHECK(nb <= nq);
            if (nb > 1) CHECK((size_t)nb * pq <= ((size_t)mib << 20));
            const int nbatch = (nq + nb - 1) / nb;
            CHECK((nq + nbatch - 1) / nbatch == nb);                  // balanced: no smaller split gives fewer queries
            CHECK(nq - (nbatch - 1) * nb >= 1);                       // no empty trailing batch
        }
}

TEST_CASE(KvflashGpuMassFixture, batched_scoring_matches_one_batch) {
    const SynthCfg cfg = small_cfg(GGML_TYPE_Q4_0, 60, 8);
    Synth s;
    s.build(cfg);
    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(cfg.L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < cfg.L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, cfg.KT, cfg.D, (int64_t)cfg.NB * cfg.CT, cfg.HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    for (int l = 0; l < cfg.L; l++) ggml_backend_tensor_set(pool_k[(size_t)l], s.pool_img[(size_t)l].data(), 0, ggml_nbytes(pool_k[(size_t)l]));

    int E = 0;
    for (int b : s.blk) E += b < 0;
    const int NBT = cfg.NB + E;
    const size_t budget3 = 3 * per_query_bytes(s.dims(), cfg.CT, NBT);   // exactly 3 queries
    const auto qp = s.qptrs();
    auto host = [&](int c, int l, int h) { return s.host_seg(c, l, h); };

    struct R { std::vector<std::vector<float>> outs; std::vector<KvFlashMassPlanes> planes; KvFlashGpuMassTiming tm; bool ok = false; };
    auto go = [&](size_t bud) {
        R r;
        KvFlashGpuMass gm(be, bud);
        r.ok = gm.score(pool_k, s.blk, host, cfg.CT, s.dims(), qp, r.outs, &r.planes, &r.tm, nullptr);
        return r;
    };
    const R one = go(kKvFlashGpuMassBatchBytes), many = go(budget3);
    REQUIRE(one.ok);
    REQUIRE(many.ok);
    CHECK(one.tm.batches == 1);
    CHECK(many.tm.batches == 3);
    CHECK(many.outs == one.outs);
    REQUIRE(many.planes.size() == one.planes.size());
    bool same = true;
    for (size_t n = 0; n < one.planes.size(); n++)
        same = same && many.planes[n].top == one.planes[n].top && many.planes[n].max == one.planes[n].max;
    CHECK(same);
    CHECK(many.tm.peak_bytes < one.tm.peak_bytes);
    CHECK(one.tm.peak_bytes <= kvflash_gpu_mass_bytes(s.dims(), cfg.KT, cfg.CT, cfg.NB, E, cfg.N, kKvFlashGpuMassBatchBytes));
    CHECK(many.tm.peak_bytes <= kvflash_gpu_mass_bytes(s.dims(), cfg.KT, cfg.CT, cfg.NB, E, cfg.N, budget3));
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
}
