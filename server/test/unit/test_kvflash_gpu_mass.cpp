// Selector-v2 GPU scorer (kvflash_gpu_mass.h) against the AVX2 CPU scorer, on the ggml CPU backend so ctest needs no
// GPU. Same graph, same pool / staging / bias logic; only the backend differs. Pass: identical top-scoring chunks and
// select-score relative error < 1e-2 against kvflash_exact_mass_scores_fast_multi.

#include "CppUnitTestFramework.hpp"

#include "kvflash_exact_fast.h"
#include "kvflash_gpu_mass.h"
#include "kvflash_mass_synth.h"

#include "ggml-cpu.h"
#ifdef LUCE_HAVE_KVFLASH_FUSED_MASS
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
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
    CHECK(o_.max_rel < 2e-2);   /* the largest plane is the noisiest statistic: 1.3e-2 measured on the mixed-residency cases */ \
} while (0)

// Quantised keys: ggml_mul_mat quantises the queries to 8 bits, the AVX2 scorer keeps them F32, so the two scorers differ by
// a rounding error that grows with the number of sealed chunks (measured on the ggml CPU backend: 2e-3 at 60, 1.7e-2 at 200,
// 4e-2 at 400). Same quantity, not the same numbers: the bound below is the documented one (kvflash_gpu_mass.h).
static constexpr double kQuantQueryBound = 5e-2;

TEST_CASE(KvflashGpuMassFixture, q4_0_400_chunks_within_the_quantised_query_bound) {
    const Outcome o = run(small_cfg(GGML_TYPE_Q4_0, 400, 8), 400);
    REQUIRE(o.ok);
    CHECK(o.sel_rel < kQuantQueryBound);
    CHECK(o.mean_rel < kQuantQueryBound);
    CHECK(o.top_rel < kQuantQueryBound);
}
TEST_CASE(KvflashGpuMassFixture, f32_keys_400_chunks_match_cpu_to_float_rounding) {
    const Outcome o = run(small_cfg(GGML_TYPE_F32, 400, 8), 400);   // no query quantisation: the graph itself is exact
    REQUIRE(o.ok);
    CHECK(o.sel_rel < 1e-4);
    CHECK(o.mean_rel < 1e-4);
    CHECK(o.top_rel < 1e-4);
    CHECK(o.picks_same);
}

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
    why.clear();
    CHECK(!gm.score(pool_k, ev, KvFlashKSegFn{}, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));   // evicted chunks, no host_seg
    CHECK(why.find("host") != std::string::npos);
    // One NaN key (a Q4_0 block scale of fp16 NaN) in an evicted chunk fails the call instead of returning NaN scores.
    std::vector<uint8_t> poisoned((size_t)c.CT * ggml_row_size(c.KT, c.D));
    auto nan_host = [&](int ch, int l, int h) -> const uint8_t * {
        std::memcpy(poisoned.data(), s.host_seg(ch, l, h), poisoned.size());
        poisoned[0] = 0x00; poisoned[1] = 0x7e;
        return poisoned.data();
    };
    why.clear();
    CHECK(!gm.score(pool_k, ev, nan_host, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));
    CHECK(why.find("non-finite") != std::string::npos);
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

// ---- Device archive of evicted keys ----------------------------------------------------------------------------
// With chunk versions the scorer keeps the evicted K it uploaded on the device, so a later call uploads only the
// chunks evicted since. Every call is checked against a fresh scorer that uploads everything (the old behaviour).

namespace {
using Slot = KvFlashGpuArchiveSlot;

// Every sealed chunk's K in the pager's host layout, and pool tensors rebuilt per residency: one scorer sees the
// same chunks move between the pool and the host across calls.
struct World {
    SynthCfg cfg;
    Synth s;
    std::vector<uint8_t> all;          // [chunk][layer][kv head][CT rows]
    std::vector<uint64_t> ver;         // per chunk, as the pager would hand out
    std::vector<int> blk;
    std::atomic<int> segs{0};          // host segments the scorer asked for
    ggml_backend_t be = nullptr;
    ggml_context * pctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr;
    std::vector<ggml_tensor *> pool_k;

    explicit World(const SynthCfg & c, ggml_backend_t dev = nullptr) : cfg(c) {
        s.build(cfg);
        all.resize((size_t)cfg.sealed * s.CHUNK);
        for (int ch = 0; ch < cfg.sealed; ch++)
            for (int l = 0; l < cfg.L; l++)
                for (int h = 0; h < cfg.HK; h++)
                    std::memcpy(seg_rw(ch, l, h), s.blk[(size_t)ch] >= 0 ? s.pool_seg(s.blk[(size_t)ch], l, h) : s.host_seg(ch, l, h), s.SEG);
        for (int ch = 0; ch < cfg.sealed; ch++) ver.push_back(100 + (uint64_t)ch);
        be = dev ? dev : ggml_backend_cpu_init();
        ggml_init_params ip{ggml_tensor_overhead() * (size_t)(cfg.L + 4), nullptr, true};
        pctx = ggml_init(ip);
        for (int l = 0; l < cfg.L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, cfg.KT, cfg.D, (int64_t)cfg.NB * cfg.CT, cfg.HK));
        pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    }
    ~World() { ggml_backend_buffer_free(pbuf); ggml_free(pctx); ggml_backend_free(be); }
    uint8_t * seg_rw(int ch, int l, int h) { return all.data() + (size_t)ch * s.CHUNK + ((size_t)l * cfg.HK + h) * s.SEG; }

    // The first `sealed` chunks with `resident` in the pool (block = position in the list), the rest on the host.
    void place(int sealed, const std::vector<int> & resident) {
        blk.assign((size_t)sealed, -1);
        for (size_t b = 0; b < resident.size(); b++) blk[(size_t)resident[b]] = (int)b;
        for (int l = 0; l < cfg.L; l++) {
            std::vector<uint8_t> img((size_t)cfg.HK * cfg.NB * cfg.CT * s.RB, 0);
            for (size_t b = 0; b < resident.size(); b++)
                for (int h = 0; h < cfg.HK; h++)
                    std::memcpy(img.data() + (size_t)h * cfg.NB * cfg.CT * s.RB + b * s.SEG, seg_rw(resident[b], l, h), s.SEG);
            ggml_backend_tensor_set(pool_k[(size_t)l], img.data(), 0, img.size());
        }
    }
    int evicted() const { return (int)std::count(blk.begin(), blk.end(), -1); }
    // Scores with `gm` (versions when `keyed`); returns the chunks whose host K the call read.
    int score(KvFlashGpuMass & gm, bool keyed, std::vector<std::vector<float>> & outs, std::vector<KvFlashMassPlanes> & planes,
              KvFlashGpuMassTiming * tm = nullptr) {
        segs = 0;
        const std::vector<uint64_t> v(ver.begin(), ver.begin() + (long)blk.size());
        std::string why;
        const bool ok = gm.score(pool_k, blk, [&](int c, int l, int h) { segs++; return (const uint8_t *)seg_rw(c, l, h); }, cfg.CT, s.dims(),
                                 s.qptrs(), outs, &planes, tm, &why, keyed ? &v : nullptr);
        if (!ok) { std::fprintf(stderr, "gpu mass: %s\n", why.c_str()); return -1; }
        return segs / (cfg.L * cfg.HK);
    }
};

std::vector<int> newest(int sealed, int keep) {   // the two sinks and the `keep` newest chunks
    std::vector<int> r = {0, 1};
    for (int c = sealed - keep; c < sealed; c++) r.push_back(c);
    return r;
}

// Same masses (1e-4 relative where they are not noise) and the same top chunks left of the window.
bool same_scores(const std::vector<std::vector<float>> & a, const std::vector<KvFlashMassPlanes> & pa,
                 const std::vector<std::vector<float>> & b, const std::vector<KvFlashMassPlanes> & pb, int tail) {
    if (a.size() != b.size() || a.empty()) return false;
    for (size_t n = 0; n < a.size(); n++)
        for (size_t c = 0; c < a[n].size(); c++)
            if (std::fabs(a[n][c] - b[n][c]) > 1e-4f * std::max(a[n][c], 1e-9f)) return false;
    const auto sa = kvflash_select_scores(pa), sb = kvflash_select_scores(pb);
    auto top = [&](const std::vector<float> & sc) {
        std::vector<int> ids((size_t)std::max<int>((int)sc.size() - tail, 0));
        for (size_t c = 0; c < ids.size(); c++) ids[c] = (int)c;
        std::stable_sort(ids.begin(), ids.end(), [&](int x, int y) { return sc[(size_t)x] > sc[(size_t)y]; });
        ids.resize(std::min<size_t>(4, ids.size()));
        return ids;
    };
    return top(sa) == top(sb);
}
}  // namespace

// What the archive does for a call, without a device: which evicted chunks need an upload and what each slot scores.
TEST_CASE(KvflashGpuMassFixture, archive_plan_uploads_only_what_it_does_not_hold) {
    const std::vector<uint64_t> ver = {10, 11, 12, 13, 14, 15};
    auto p = kvflash_gpu_archive_plan({}, {1, 3, 4}, ver);
    CHECK(!p.reset);
    CHECK(p.upload == (std::vector<int>{1, 3, 4}));
    CHECK(p.chunk_of == (std::vector<int>{1, 3, 4}));
    const std::vector<Slot> held = {{1, 11}, {3, 13}, {4, 14}};
    p = kvflash_gpu_archive_plan(held, {1, 3, 4}, ver);          // nothing moved
    CHECK(p.upload.empty());
    CHECK(p.chunk_of == (std::vector<int>{1, 3, 4}));
    p = kvflash_gpu_archive_plan(held, {1, 4, 5}, ver);          // 3 recalled into the pool (scored there), 5 evicted
    CHECK(p.upload == (std::vector<int>{5}));
    CHECK(p.chunk_of == (std::vector<int>{1, -1, 4, 5}));
    p = kvflash_gpu_archive_plan(held, {1}, {10, 11});           // fewer chunks sealed: slots of later chunks wait, masked
    CHECK(!p.reset);
    CHECK(p.upload.empty());
    CHECK(p.chunk_of == (std::vector<int>{1, -1, -1}));
}
TEST_CASE(KvflashGpuMassFixture, archive_plan_never_scores_a_rewritten_chunk_from_its_old_slot) {
    const std::vector<Slot> held = {{1, 11}, {3, 13}, {4, 14}, {5, 15}};
    auto p = kvflash_gpu_archive_plan(held, {1, 3, 4, 5}, {10, 11, 12, 99, 14, 15});   // chunk 3 was rewritten
    CHECK(!p.reset);
    CHECK(p.upload == (std::vector<int>{3}));
    CHECK(p.chunk_of == (std::vector<int>{1, -1, 4, 5, 3}));
    p = kvflash_gpu_archive_plan(held, {1, 3, 4, 5}, {10, 91, 12, 93, 94, 15});        // most slots stale: start over
    CHECK(p.reset);
    CHECK(p.upload == (std::vector<int>{1, 3, 4, 5}));
    CHECK(p.chunk_of == (std::vector<int>{1, 3, 4, 5}));
}

TEST_CASE(KvflashGpuMassFixture, archive_uploads_each_evicted_chunk_once_and_scores_like_a_fresh_scorer) {
    World w(small_cfg(GGML_TYPE_Q8_0, 60, 8));
    KvFlashGpuMass gm(w.be);
    std::vector<std::vector<float>> o, ref;
    std::vector<KvFlashMassPlanes> pl, pref;
    KvFlashGpuMassTiming tm;
    struct Step { int sealed; std::vector<int> resident; int uploads; };
    std::vector<int> recalled = newest(60, 18);
    recalled.push_back(5); recalled.push_back(6);                // 5 and 6 recalled, 40 and 41 evicted for them
    const std::vector<Step> steps = {
        {40, newest(40, 20), 18},     // first call: every evicted chunk
        {60, newest(60, 20), 20},     // 20 chunks later: only the 20 evicted since (growth keeps the 18)
        {60, recalled, 2},            // 40 and 41 are new on the host; 5 and 6 are scored from the pool, not twice
        {60, newest(60, 20), 0},      // 5 and 6 evicted again: still held
        {48, newest(48, 20), 0},      // a rewind scores fewer chunks: nothing to upload
        {60, newest(60, 20), 0},
    };
    for (const Step & st : steps) {
        w.place(st.sealed, st.resident);
        CHECK(w.score(gm, true, o, pl, &tm) == st.uploads);
        CHECK(tm.uploaded == st.uploads);
        KvFlashGpuMass fresh(w.be);
        CHECK(w.score(fresh, false, ref, pref) == w.evicted());
        CHECK(same_scores(o, pl, ref, pref, 16));
    }
    CHECK(tm.archived == 40);
}

TEST_CASE(KvflashGpuMassFixture, archive_reuploads_a_chunk_whose_version_changed) {
    World w(small_cfg(GGML_TYPE_Q8_0, 60, 8));
    KvFlashGpuMass gm(w.be), fresh(w.be);
    std::vector<std::vector<float>> o, ref;
    std::vector<KvFlashMassPlanes> pl, pref;
    w.place(60, newest(60, 20));
    REQUIRE(w.score(gm, true, o, pl) == 38);
    // chunk 7 is rewritten with the keys of the hottest planted chunk and gets a new version
    std::memcpy(w.seg_rw(7, 0, 0), w.seg_rw(10 + 7 * (w.s.n_hot - 1), 0, 0), w.s.CHUNK);
    w.ver[7] = 1000;
    CHECK(w.score(gm, true, o, pl) == 1);
    REQUIRE(w.score(fresh, false, ref, pref) == 38);
    CHECK(same_scores(o, pl, ref, pref, 16));
    // a new conversation: every version differs, the archive starts over
    for (uint64_t & v : w.ver) v += 5000;
    KvFlashGpuMassTiming tm;
    CHECK(w.score(gm, true, o, pl, &tm) == 38);
    CHECK(tm.archived == 38);
    CHECK(same_scores(o, pl, ref, pref, 16));
}

TEST_CASE(KvflashGpuMassFixture, without_versions_every_call_uploads_every_evicted_chunk) {
    World w(small_cfg(GGML_TYPE_Q4_0, 60, 8));
    KvFlashGpuMass gm(w.be);
    std::vector<std::vector<float>> o;
    std::vector<KvFlashMassPlanes> pl;
    KvFlashGpuMassTiming tm;
    w.place(60, newest(60, 20));
    CHECK(w.score(gm, false, o, pl, &tm) == 38);
    CHECK(w.score(gm, false, o, pl, &tm) == 38);
    CHECK(tm.archived == 0);
}

// The fused kernel reads the pool K and the archive where they are: the same residency steps, against the logits graph of a fresh scorer.
TEST_CASE(KvflashGpuMassFixture, fused_kernel_over_the_archive_matches_the_logits_graph) {
#ifdef LUCE_HAVE_KVFLASH_FUSED_MASS
    ggml_backend_t dev = ggml_backend_cuda_init(0);
    if (!dev) SKIP("no CUDA/HIP device");
    SynthCfg c = small_cfg(GGML_TYPE_Q8_0, 60, 8);
    c.L = 8;
    World w(c, dev);
    KvFlashGpuMass gm(w.be);
    gm.set_fused(true);
    std::vector<std::vector<float>> o, ref;
    std::vector<KvFlashMassPlanes> pl, pref;
    KvFlashGpuMassTiming tm, tr;
    std::vector<int> recalled = newest(60, 18);
    recalled.push_back(5); recalled.push_back(6);
    struct Step { int sealed; std::vector<int> resident; int uploads; };
    const std::vector<Step> steps = {{40, newest(40, 20), 18}, {60, newest(60, 20), 20}, {60, recalled, 2}, {60, newest(60, 20), 0}, {48, newest(48, 20), 0}};
    for (const Step & st : steps) {
        w.place(st.sealed, st.resident);
        CHECK(w.score(gm, true, o, pl, &tm) == st.uploads);
        CHECK(tm.fused);
        KvFlashGpuMass fresh(w.be);
        REQUIRE(w.score(fresh, false, ref, pref, &tr) == w.evicted());
        CHECK(!tr.fused);
        double worst = 0;
        for (size_t n = 0; n < ref.size(); n++)
            for (size_t k = 0; k < ref[n].size(); k++) {
                const double a = ref[n][k], b = o[n][k], ta = pref[n].top[k], tb = pl[n].top[k], xa = pref[n].max[k], xb = pl[n].max[k];
                if (a > 1e-6) worst = std::max(worst, std::fabs(a - b) / std::max(a, b));
                if (ta > 1e-6) worst = std::max(worst, std::fabs(ta - tb) / std::max(ta, tb));
                if (xa > 1e-6) worst = std::max(worst, std::fabs(xa - xb) / std::max(xa, xb));
            }
        std::fprintf(stderr, "fused over archive vs graph: sealed=%d evicted=%d worst rel diff %.3g\n", st.sealed, w.evicted(), worst);
        CHECK(worst <= 1e-3);
    }
#else
    SKIP("needs a CUDA/HIP backend and the fused kernel");
#endif
}

// Which K the fused kernel takes. Everything else is scored by the logits graph, and the backend passes no versions then (no archive).
TEST_CASE(KvflashGpuMassFixture, fused_kernel_is_selected_only_for_the_k_it_can_score) {
#ifdef LUCE_HAVE_KVFLASH_FUSED_MASS
    CHECK(KvFlashGpuMass::fused_available(GGML_TYPE_Q8_0, 128, 16));
    CHECK(KvFlashGpuMass::fused_available(GGML_TYPE_Q8_0, 256, 64));
#else
    CHECK(!KvFlashGpuMass::fused_available(GGML_TYPE_Q8_0, 128, 16));
#endif
    for (ggml_type kt : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_1})
        CHECK(!KvFlashGpuMass::fused_available(kt, 128, 16));
    CHECK(!KvFlashGpuMass::fused_available(GGML_TYPE_Q8_0, 48, 16));    // not whole 32-blocks
    CHECK(!KvFlashGpuMass::fused_available(GGML_TYPE_Q8_0, 288, 16));   // over the kernel's head dim
    CHECK(!KvFlashGpuMass::fused_available(GGML_TYPE_Q8_0, 128, 65));   // over the kernel's layer table
}

// A scorer asked for the fused kernel on a K it cannot score runs the logits graph and still matches the CPU scorer.
TEST_CASE(KvflashGpuMassFixture, fused_request_on_other_k_types_scores_with_the_logits_graph) {
    World w(small_cfg(GGML_TYPE_Q4_0, 60, 8));
    w.place(60, newest(60, 20));
    KvFlashGpuMass gm(w.be), plain(w.be);
    gm.set_fused(true);
    std::vector<std::vector<float>> o, ref;
    std::vector<KvFlashMassPlanes> pl, pref;
    KvFlashGpuMassTiming tm;
    REQUIRE(w.score(gm, false, o, pl, &tm) == 38);
    CHECK(!tm.fused);
    CHECK(tm.archived == 0);
    REQUIRE(w.score(plain, false, ref, pref) == 38);
    CHECK(o == ref);
}
