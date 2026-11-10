// Selector-v2 exact chunk mass on the target backend. The same quantity as kvflash_exact_mass_scores_fast_multi
// (kvflash_exact_fast.h): per (query, chunk) the mean over the (layer, q-head) planes, plus KvFlashMassPlanes{top,max}.
// NOT the same numbers: with Q4_0/Q8_0 keys ggml_mul_mat quantises the queries to 8 bits (the CPU backend and, by reading, the
// CUDA/HIP MMQ path), while the AVX2 scorer keeps them in F32. Measured against it on the ggml CPU backend (CUDA/HIP size not measured), the largest
// relative deviation of the select score is ~2e-3 at 60 sealed chunks, 1.7e-2 at 200 and 4e-2 at 400, and the k-highest
// chunk sets can differ for some k. F16 keys deviate by <1e-3 and F32 keys by <1e-5. The two scorers are therefore not
// interchangeable inside one session: a caller that falls back from one to the other changes which chunk sits at a
// residency boundary. The CPU scorer needs ~1.7 s for 64 queries at 514 sealed chunks.
//
// Side graphs per call, stock ggml ops only (ggml / ggml-backend / ggml-alloc; no ggml-cuda.h), so it runs on
// CUDA, HIP, Vulkan and, for the unit test, the ggml CPU backend:
//   keys     resident chunks are read in place (the pool tensors [D, pool tokens, Hkv] are graph leaves); evicted
//            chunks are gathered from their host copies into a pinned staging buffer, one 3D tensor [D, E*ct, Hkv]
//            per layer (a plain tensor, so the staging mul_mat can take MMQ like the pool one), uploaded once.
//   per layer  mul_mat(pool K, q) and mul_mat(staging K, q), q as [D, G*m, Hkv] (each kv head serves its G q heads);
//            concat; add a [T] bias (0 on blocks holding a sealed chunk, -inf on every other block: free, or holding
//            a chunk >= sealed as in the first pass of a turn); soft_max_ext at 1/sqrt(D); sum_rows over [ct, blocks].
//   host     per batch one D2H of [planes, blocks, m]; per (query, chunk) mean, top-8 mean (nth_element) and max over planes.
// Device memory is call-scoped: nothing is held between calls. The evicted K, all queries and the bias are uploaded
// once per call; the N queries are scored in balanced batches of m under a byte budget (kvflash_gpu_mass_batch), each
// with its own graph and gallocr buffer, freed before the call returns. Only the pinned HOST staging / output
// buffers persist across calls and grow on demand.
//
// WSL silently spills over-committed VRAM into system RAM (slow, no error), so the working set is estimated first
// and the call refuses (returns false) when it exceeds the device's free memory minus a margin.

#pragma once

#include "kvflash_exact_mass.h"
#include "parallel_for.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace luce::common {

struct KvFlashGpuMassTiming {   // ms
    double gather = 0, build = 0, h2d = 0, graph = 0, d2h = 0, agg = 0, total = 0;
    int batches = 0;
    size_t peak_bytes = 0;   // device bytes held at the call's peak: call buffer + the largest gallocr buffer
};

inline constexpr size_t kKvFlashGpuMassBatchBytes = (size_t)512 << 20;

// Pure: the chunk each graph block holds, -1 = masked. Blocks [0, n_pool) are the pool's; block n_pool + e is the
// e-th evicted chunk (in `evicted`, ascending). block_of[c] = pool block of sealed chunk c, -1 = evicted; pool blocks
// no sealed chunk maps to (free, or chunk >= sealed) stay -1. Empty result when block_of points outside the pool.
inline std::vector<int> kvflash_gpu_mass_chunk_of(const std::vector<int> & block_of, int n_pool, std::vector<int> & evicted) {
    evicted.clear();
    std::vector<int> chunk_of((size_t)std::max(n_pool, 0), -1);
    for (size_t c = 0; c < block_of.size(); c++) {
        const int b = block_of[c];
        if (b >= n_pool) return {};
        if (b >= 0) { if (chunk_of[(size_t)b] >= 0) return {}; chunk_of[(size_t)b] = (int)c; }
        else evicted.push_back((int)c);
    }
    for (int c : evicted) chunk_of.push_back(c);
    return chunk_of;
}

// Pure: bytes one query's graph holds: up to 3 [T, G, Hkv] f32 logits rows live (conservative; measured 2) plus its
// [planes, blocks] output.
inline size_t kvflash_gpu_mass_per_query(const KvFlashQkDims & d, int ct, int n_blocks_total) {
    const size_t blocks = (size_t)std::max(n_blocks_total, 1);
    return (3 * blocks * ct * d.n_q_heads + (size_t)d.n_layers * d.n_q_heads * blocks) * sizeof(float);
}

// Pure: queries per graph; the cap is what fits in `budget` (at least 1), and the batches are balanced.
inline int kvflash_gpu_mass_batch(const KvFlashQkDims & d, int ct, int n_blocks_total, int nq, size_t budget) {
    if (nq <= 0) return 0;
    const size_t per_q = kvflash_gpu_mass_per_query(d, ct, n_blocks_total);
    const size_t cap = std::min<size_t>((size_t)nq, std::max<size_t>(budget / std::max<size_t>(per_q, 1), 1));
    const size_t n_batches = ((size_t)nq + cap - 1) / cap;
    return (int)(((size_t)nq + n_batches - 1) / n_batches);
}

// Pure: peak device bytes of a batched call: the evicted K staging (all layers), the queries (all nq), the bias, and
// the largest batch's graph (nb queries x per-query logits and output, see kvflash_gpu_mass_batch).
inline size_t kvflash_gpu_mass_bytes(const KvFlashQkDims & d, ggml_type kt, int ct, int n_blocks, int n_evicted, int nq,
                                     size_t budget = kKvFlashGpuMassBatchBytes) {
    const size_t T = (size_t)(n_blocks + n_evicted) * ct;
    const int nb = kvflash_gpu_mass_batch(d, ct, n_blocks + n_evicted, nq, budget);
    const size_t staging = (size_t)d.n_layers * d.n_kv_heads * n_evicted * ct * ggml_row_size(kt, d.head_dim);
    const size_t q = (size_t)nq * d.n_layers * d.n_q_heads * d.head_dim * sizeof(float);
    return staging + q + T * sizeof(float) + (size_t)nb * kvflash_gpu_mass_per_query(d, ct, n_blocks + n_evicted);
}

// Pure: whether a call needing `need` bytes fits in the device's free memory minus a margin.
inline bool kvflash_gpu_mass_fits(size_t need, size_t free_bytes, size_t total_bytes, size_t margin) {
    return !(total_bytes > 0 && need + margin > free_bytes);
}

class KvFlashGpuMass {
public:
    static constexpr size_t kMarginBytes = (size_t)512 << 20;   // VRAM kept free beside the scorer

    explicit KvFlashGpuMass(ggml_backend_t be, size_t batch_bytes = kKvFlashGpuMassBatchBytes) : be_(be), batch_bytes_(batch_bytes) {}
    KvFlashGpuMass(const KvFlashGpuMass &) = delete;
    KvFlashGpuMass & operator=(const KvFlashGpuMass &) = delete;
    ~KvFlashGpuMass() {
        if (stage_) ggml_backend_buffer_free(stage_);
        if (out_) ggml_backend_buffer_free(out_);
    }

    // Would a call with `nq` queries pass the free-memory guard right now (same test as score())?
    // `need_out`, when set, gets the bytes the call needs, `free_out` the device's free bytes.
    bool fits_now(const KvFlashQkDims & d, ggml_type kt, int ct, int n_blocks, int n_evicted, int nq, size_t * need_out = nullptr,
                  size_t * free_out = nullptr) const {
        const size_t need = kvflash_gpu_mass_bytes(d, kt, ct, n_blocks, n_evicted, nq, batch_bytes_);
        if (need_out) *need_out = need;
        ggml_backend_dev_t dev = be_ ? ggml_backend_get_device(be_) : nullptr;
        if (!dev) return true;
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(dev, &fr, &tot);
        if (free_out) *free_out = fr;
        return kvflash_gpu_mass_fits(need, fr, tot, kMarginBytes);
    }

    // pool_k[l]: [D, n_blocks * ct, Hkv] of the K type (cache_.attn_k). block_of[c] for c < sealed = block_of.size().
    // host_seg(c, l, h): the evicted chunk's ct rows for (layer, kv head), as KvFlashKSegFn. qs[i]: L*Hq*D floats.
    // Fills outs / planes like kvflash_exact_mass_scores_fast_multi. false (+ why): outs / planes are not valid.
    bool score(const std::vector<ggml_tensor *> & pool_k, const std::vector<int> & block_of, const KvFlashKSegFn & host_seg,
               int ct, const KvFlashQkDims & d, const std::vector<const float *> & qs,
               std::vector<std::vector<float>> & outs, std::vector<KvFlashMassPlanes> * planes,
               KvFlashGpuMassTiming * timing = nullptr, std::string * why = nullptr) {
        auto fail = [&](const std::string & m) { if (why) *why = m; return false; };
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        KvFlashGpuMassTiming tm;
        const auto t0 = clk::now();
        const int L = d.n_layers, HK = d.n_kv_heads, HQ = d.n_q_heads, D = d.head_dim, N = (int)qs.size(), sealed = (int)block_of.size();
        if (!be_) return fail("no backend");
        if (L <= 0 || HK <= 0 || HQ <= 0 || D <= 0 || HQ % HK || ct <= 0) return fail("bad dims");
        const int G = HQ / HK, PL = L * HQ;
        if (sealed <= 0) return fail("no sealed chunks");
        if ((int)pool_k.size() != L) return fail("pool tensor count != layers");
        for (ggml_tensor * t : pool_k)
            if (!t || t->ne[0] != D || t->ne[2] != HK || t->ne[1] % ct || t->type != pool_k[0]->type || t->ne[1] != pool_k[0]->ne[1])
                return fail("unexpected pool K tensor shape or type");
        for (const float * q : qs) if (!q) return fail("null query");
        const ggml_type kt = pool_k[0]->type;
        const int NB = (int)(pool_k[0]->ne[1] / ct);
        if (N == 0) { outs.clear(); if (planes) planes->clear(); return true; }
        std::vector<int> ev;
        const std::vector<int> chunk_of = kvflash_gpu_mass_chunk_of(block_of, NB, ev);
        if (chunk_of.empty()) return fail("block_of points outside the pool or maps two chunks to one block");
        const int E = (int)ev.size(), NBT = NB + E, T = NBT * ct;
        if (E > 0 && !host_seg) return fail("missing host K segment callback");
        const size_t RB = ggml_row_size(kt, D), SEG = (size_t)ct * RB, QS = (size_t)L * HQ * D;
        const size_t stage_k = (size_t)L * HK * E * SEG, stage_total = stage_k + (size_t)N * QS * sizeof(float);
        const size_t out_bytes = (size_t)PL * NBT * N * sizeof(float);

        // Free-memory guard.
        ggml_backend_dev_t dev = ggml_backend_get_device(be_);
        size_t need = 0, fr = 0;
        if (!fits_now(d, kt, ct, NB, E, N, &need, &fr))
            return fail("needs " + std::to_string(need >> 20) + " MiB (+" + std::to_string(kMarginBytes >> 20) + " MiB margin) but " +
                        std::to_string(fr >> 20) + " MiB are free");
        if (!host_buft_) {
            host_buft_ = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (!host_buft_) host_buft_ = ggml_backend_get_default_buffer_type(be_);
            if (!host_buft_ || !ggml_backend_buft_is_host(host_buft_)) { host_buft_ = nullptr; return fail("no host-accessible buffer type"); }
        }
        if (!grow(stage_, stage_total)) return fail("pinned staging alloc failed");
        if (!grow(out_, out_bytes)) return fail("pinned output alloc failed");
        outs.assign((size_t)N, std::vector<float>((size_t)sealed, 0.0f));
        if (planes) planes->assign((size_t)N, KvFlashMassPlanes{std::vector<float>((size_t)sealed, 0.0f), std::vector<float>((size_t)sealed, 0.0f)});
        uint8_t * stage = (uint8_t *)ggml_backend_buffer_get_base(stage_);
        float * qbuf = (float *)(stage + stage_k);
        float * out = (float *)ggml_backend_buffer_get_base(out_);

        // Gather: evicted K into the staging layout [layer][kv head][e][row], and the queries behind it.
        std::atomic<bool> absent{false};
        parallel_for(L, [&](int l) {
            for (int h = 0; h < HK; h++)
                for (int e = 0; e < E; e++) {
                    const uint8_t * s = host_seg(ev[(size_t)e], l, h);
                    if (!s) { absent = true; continue; }
                    std::memcpy(stage + (((size_t)l * HK + h) * E + e) * SEG, s, SEG);
                }
        });
        if (absent) return fail("missing host K segment");
        for (int n = 0; n < N; n++) std::memcpy(qbuf + (size_t)n * QS, qs[(size_t)n], QS * sizeof(float));
        std::vector<float> bias((size_t)T);
        for (int t = 0; t < T; t++) bias[(size_t)t] = chunk_of[(size_t)(t / ct)] >= 0 ? 0.0f : -INFINITY;
        const auto t1 = clk::now();

        // Call-scoped device memory: evicted K, all queries, bias; uploaded once.
        struct Scope {
            ggml_context * cctx = nullptr; ggml_backend_buffer_t cbuf = nullptr; ggml_gallocr_t ga = nullptr;
            ~Scope() { if (ga) ggml_gallocr_free(ga); if (cbuf) ggml_backend_buffer_free(cbuf); if (cctx) ggml_free(cctx); }
        } sc;
        ggml_init_params cp{ggml_tensor_overhead() * (size_t)(L + 4), nullptr, true};
        sc.cctx = ggml_init(cp);
        if (!sc.cctx) return fail("ggml_init failed");
        std::vector<ggml_tensor *> kst((size_t)L, nullptr);
        for (int l = 0; l < L && E; l++) kst[(size_t)l] = ggml_new_tensor_3d(sc.cctx, kt, D, (int64_t)E * ct, HK);
        ggml_tensor * q_all = ggml_new_tensor_4d(sc.cctx, GGML_TYPE_F32, D, HQ, L, N);
        ggml_tensor * b_t = ggml_new_tensor_1d(sc.cctx, GGML_TYPE_F32, T);
        sc.cbuf = ggml_backend_alloc_ctx_tensors(sc.cctx, be_);
        if (!sc.cbuf) return fail("call buffer allocation failed");
        for (int l = 0; l < L && E; l++)
            ggml_backend_tensor_set_async(be_, kst[(size_t)l], stage + (size_t)l * HK * E * SEG, 0, ggml_nbytes(kst[(size_t)l]));
        ggml_backend_tensor_set_async(be_, q_all, qbuf, 0, ggml_nbytes(q_all));
        ggml_backend_tensor_set_async(be_, b_t, bias.data(), 0, ggml_nbytes(b_t));
        ggml_backend_synchronize(be_);
        tm.h2d = ms(t1, clk::now());

        const int nb = kvflash_gpu_mass_batch(d, ct, NBT, N, batch_bytes_);
        sc.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be_));
        if (!sc.ga) return fail("graph allocator creation failed");
        const int max_nodes = 32 * L + 64;
        size_t galloc_peak = 0;
        for (int n0 = 0; n0 < N; n0 += nb) {
            const int m = std::min(nb, N - n0);
            const auto tb = clk::now();
            ggml_init_params ip{ggml_tensor_overhead() * (size_t)(40 * L + 64) + ggml_graph_overhead_custom(max_nodes, false), nullptr, true};
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) return fail("ggml_init failed");
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
            ggml_tensor * o_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, PL, NBT, m);
            ggml_set_output(o_t);
            for (int l = 0; l < L; l++) {
                ggml_tensor * qv = ggml_view_4d(ctx, q_all, D, G, HK, m, q_all->nb[1], G * q_all->nb[1], q_all->nb[3],
                                                (size_t)l * q_all->nb[2] + (size_t)n0 * q_all->nb[3]);
                ggml_tensor * qr = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, qv, 0, 1, 3, 2)), D, G * m, HK);
                ggml_tensor * lg = ggml_mul_mat(ctx, pool_k[(size_t)l], qr);                       // [NB*ct, G*m, HK]
                if (E) lg = ggml_concat(ctx, lg, ggml_mul_mat(ctx, kst[(size_t)l], qr), 0);        // [T, G*m, HK]
                lg = ggml_add(ctx, lg, b_t);
                ggml_tensor * sm = ggml_soft_max_ext(ctx, lg, nullptr, 1.0f / std::sqrt((float)D), 0.0f);
                ggml_tensor * mm = ggml_sum_rows(ctx, ggml_reshape_4d(ctx, sm, ct, NBT, G * m, HK));   // [1, NBT, G*m, HK]
                ggml_tensor * mp = ggml_permute(ctx, ggml_reshape_4d(ctx, mm, NBT, G, m, HK), 2, 0, 3, 1);   // [G, HK, NBT, m]
                ggml_tensor * dst = ggml_view_4d(ctx, o_t, G, HK, NBT, m, G * sizeof(float), o_t->nb[1], o_t->nb[2], (size_t)l * HQ * sizeof(float));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, mp, dst));
            }
            if (!ggml_gallocr_alloc_graph(sc.ga, gf)) { ggml_free(ctx); return fail("graph allocation failed"); }
            galloc_peak = std::max(galloc_peak, ggml_gallocr_get_buffer_size(sc.ga, 0));
            const auto tg = clk::now();
            const bool ok = ggml_backend_graph_compute(be_, gf) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(be_);
            const auto td = clk::now();
            if (ok) { ggml_backend_tensor_get_async(be_, o_t, out + (size_t)n0 * PL * NBT, 0, ggml_nbytes(o_t)); ggml_backend_synchronize(be_); }
            const auto te = clk::now();
            ggml_free(ctx);
            if (!ok) return fail("graph compute failed");
            tm.build += ms(tb, tg); tm.graph += ms(tg, td); tm.d2h += ms(td, te);
            tm.batches++;
        }
        tm.peak_bytes = ggml_backend_buffer_get_size(sc.cbuf) + galloc_peak;
        const auto t5 = clk::now();

        // Aggregate over the planes, per query in parallel.
        constexpr int K = kExactTopPlanes;
        const int Kp = std::min(K, PL);
        std::atomic<bool> non_finite{false};
        parallel_for(N, [&](int n) {
            std::vector<float> tmp((size_t)PL);
            for (int gb = 0; gb < NBT; gb++) {
                const int c = chunk_of[(size_t)gb];
                if (c < 0) continue;
                const float * x = out + ((size_t)n * NBT + gb) * PL;
                double s = 0;
                for (int i = 0; i < PL; i++) s += x[i];
                if (!std::isfinite(s)) { non_finite = true; return; }   // NaN would also break the nth_element below
                outs[(size_t)n][(size_t)c] = (float)(s / PL);
                if (!planes) continue;
                std::copy(x, x + PL, tmp.begin());
                std::nth_element(tmp.begin(), tmp.begin() + (Kp - 1), tmp.end(), std::greater<float>());
                double top = 0;
                for (int i = 0; i < Kp; i++) top += tmp[(size_t)i];
                (*planes)[(size_t)n].top[(size_t)c] = (float)(top / Kp);
                (*planes)[(size_t)n].max[(size_t)c] = *std::max_element(tmp.begin(), tmp.begin() + Kp);
            }
        });
        if (non_finite) return fail("non-finite mass");
        const auto t6 = clk::now();
        tm.gather = ms(t0, t1);
        tm.agg = ms(t5, t6); tm.total = ms(t0, t6);
        if (timing) *timing = tm;
        return true;
    }

private:
    bool grow(ggml_backend_buffer_t & buf, size_t bytes) {
        if (buf && ggml_backend_buffer_get_size(buf) >= bytes) return true;
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        buf = ggml_backend_buft_alloc_buffer(host_buft_, std::max<size_t>(bytes, 1));
        return buf && ggml_backend_buffer_get_base(buf);
    }

    ggml_backend_t            be_;
    ggml_backend_buffer_type_t host_buft_ = nullptr;
    size_t                    batch_bytes_;
    ggml_backend_buffer_t     stage_ = nullptr;   // pinned: evicted K staging, then the queries
    ggml_backend_buffer_t     out_ = nullptr;     // pinned: [planes, blocks, N] masses
};

}  // namespace luce::common
