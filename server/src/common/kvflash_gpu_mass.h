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
//            chunks are gathered from their host copies into a pinned staging buffer and uploaded into the archive,
//            one 3D tensor [D, slots*ct, Hkv] per layer (a plain tensor, so its mul_mat can take MMQ like the pool one).
//   per layer  mul_mat(pool K, q) and mul_mat(staging K, q), q as [D, G*m, Hkv] (each kv head serves its G q heads);
//            concat; add a [T] bias (0 on blocks holding a sealed chunk, -inf on every other block: free, or holding
//            a chunk >= sealed as in the first pass of a turn); soft_max_ext at 1/sqrt(D); sum_rows over [ct, blocks].
//   host     per batch one D2H of [planes, blocks, m]; per (query, chunk) mean, top-8 mean (nth_element) and max over planes.
// The queries and the bias are uploaded once per call; the N queries are scored in balanced batches of m under a byte
// budget (kvflash_gpu_mass_batch), each with its own graph and gallocr buffer, freed before the call returns. The
// pinned HOST staging / output buffers persist across calls and grow on demand.
//
// Archive: uploading every evicted chunk's K was the call's host-to-device cost (1.06 MiB per chunk for the 27B at
// q8_0: 1.4 GiB at 1340 evicted chunks, every call). A caller that passes chunk versions (KvFlashPager::
// chunk_version) lets the archive stay on the device between calls: a slot is scored only while its chunk still has
// the version it was uploaded with and is evicted now, so a call uploads just the chunks evicted since the last one.
// It holds what one call allocated before, for longer. Without versions the archive is call-scoped, as before.
//
// WSL silently spills over-committed VRAM into system RAM (slow, no error), so the working set is estimated first
// and the call refuses (returns false) when it exceeds the device's free memory minus a margin.

#pragma once

#include "kvflash_exact_mass.h"
#include "kvflash_fused_mass.h"
#include "kvflash_fused_mass_cuda.h"
#include "parallel_for.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
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
    bool fused = false;               // the fused kernel scored this call: graph = its upload + kernel + reduce, d2h = its copy back
    int uploaded = 0, archived = 0;   // evicted chunks uploaded by this call; archive slots held after it
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

// What an archive slot holds: chunk `chunk` as it was at version `ver`.
struct KvFlashGpuArchiveSlot { int chunk = -1; uint64_t ver = 0; };

struct KvFlashGpuArchivePlan {
    bool             reset = false;   // most slots are stale: drop them all first
    std::vector<int> upload;          // evicted chunks to upload, ascending; they take the slots after the kept ones
    std::vector<int> chunk_of;        // per slot after the upload: the chunk it is scored as, -1 = masked
};

// Pure: `slots` = what the archive holds, `evicted` = this call's evicted chunks (ascending), ver[c] = version of
// sealed chunk c. A slot scores only when its chunk is evicted now and still has the slot's version: a resident chunk
// is scored from the pool, a rewritten one from a new slot. Slots of chunks >= ver.size() (a call that scores fewer
// chunks) are kept and masked.
inline KvFlashGpuArchivePlan kvflash_gpu_archive_plan(const std::vector<KvFlashGpuArchiveSlot> & slots, const std::vector<int> & evicted,
                                                      const std::vector<uint64_t> & ver) {
    KvFlashGpuArchivePlan p;
    auto stale = [&](const KvFlashGpuArchiveSlot & s) { return s.chunk >= 0 && (size_t)s.chunk < ver.size() && ver[(size_t)s.chunk] != s.ver; };
    p.reset = 2 * (size_t)std::count_if(slots.begin(), slots.end(), stale) > slots.size();
    std::vector<int> slot_of(ver.size(), -1);
    if (!p.reset) {
        p.chunk_of.assign(slots.size(), -1);
        for (size_t i = 0; i < slots.size(); i++)
            if (slots[i].chunk >= 0 && (size_t)slots[i].chunk < ver.size() && !stale(slots[i])) slot_of[(size_t)slots[i].chunk] = (int)i;
    }
    for (int c : evicted) {
        const int s = (size_t)c < slot_of.size() ? slot_of[(size_t)c] : -1;
        if (s >= 0) p.chunk_of[(size_t)s] = c;
        else { p.upload.push_back(c); p.chunk_of.push_back(c); }
    }
    return p;
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

// Pure: peak device bytes of a batched call: the evicted K (all layers; n_evicted = archive slots), the queries (all
// nq), the bias, and the largest batch's graph (nb queries x per-query logits and output, see kvflash_gpu_mass_batch).
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
        arch_free();
#ifdef LUCE_HAVE_KVFLASH_FUSED_MASS
        if (fws_) luce::common::kvflash_fused_ws_free(fws_);
#endif
        if (stage_) ggml_backend_buffer_free(stage_);
        if (out_) ggml_backend_buffer_free(out_);
    }

    // The fused chunk-mass kernel (kvflash_fused_mass_cuda.h) instead of the logits graph, when the K is q8_0 and the build has it.
    // Its scores follow the same MMQ numerics but are not bit-equal to the graph's.
    void set_fused(bool on) { fused_ = on; }
    // Can the fused kernel score this K? The logits graph spans every archive slot, held for this call's chunks or not, so callers
    // keep the archive (pass versions) only when this holds.
    static bool fused_available(ggml_type kt, int D, int L) {
#ifdef LUCE_HAVE_KVFLASH_FUSED_MASS
        return kt == GGML_TYPE_Q8_0 && D % 32 == 0 && D <= 256 && L <= luce::common::kFusedMaxLayers;
#else
        (void)kt; (void)D; (void)L;
        return false;
#endif
    }

    // Would a call with `nq` queries pass the free-memory guard right now (same test as score())? The archive counts
    // as free: a call that does not fit beside it starts it over. `need_out`, when set, gets the bytes the call
    // needs with every evicted chunk uploaded, `free_out` the device's free bytes plus the archive's.
    bool fits_now(const KvFlashQkDims & d, ggml_type kt, int ct, int n_blocks, int n_evicted, int nq, size_t * need_out = nullptr,
                  size_t * free_out = nullptr) const {
        const size_t need = kvflash_gpu_mass_bytes(d, kt, ct, n_blocks, n_evicted, nq, batch_bytes_);
        if (need_out) *need_out = need;
        ggml_backend_dev_t dev = be_ ? ggml_backend_get_device(be_) : nullptr;
        if (!dev) return true;
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(dev, &fr, &tot);
        fr += arch_bytes();
        if (free_out) *free_out = fr;
        return kvflash_gpu_mass_fits(need, fr, tot, kMarginBytes);
    }

    // pool_k[l]: [D, n_blocks * ct, Hkv] of the K type (cache_.attn_k). block_of[c] for c < sealed = block_of.size().
    // host_seg(c, l, h): the evicted chunk's ct rows for (layer, kv head), as KvFlashKSegFn. qs[i]: L*Hq*D floats.
    // Fills outs / planes like kvflash_exact_mass_scores_fast_multi. false (+ why): outs / planes are not valid.
    // versions: (*versions)[c] = KvFlashPager::chunk_version(c) for every sealed chunk keeps the uploaded K on the
    // device for the next call; null = upload every evicted chunk and free it on return.
    bool score(const std::vector<ggml_tensor *> & pool_k, const std::vector<int> & block_of, const KvFlashKSegFn & host_seg,
               int ct, const KvFlashQkDims & d, const std::vector<const float *> & qs,
               std::vector<std::vector<float>> & outs, std::vector<KvFlashMassPlanes> * planes,
               KvFlashGpuMassTiming * timing = nullptr, std::string * why = nullptr,
               const std::vector<uint64_t> * versions = nullptr) {
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
        std::vector<int> chunk_of = kvflash_gpu_mass_chunk_of(block_of, NB, ev);
        if (chunk_of.empty()) return fail("block_of points outside the pool or maps two chunks to one block");
        const int E = (int)ev.size();
        if (E > 0 && !host_seg) return fail("missing host K segment callback");
        if (versions && (int)versions->size() != sealed) return fail("chunk versions do not cover the sealed chunks");
        const bool keep = versions != nullptr;
        const size_t RB = ggml_row_size(kt, D), SEG = (size_t)ct * RB, QS = (size_t)L * HQ * D;
        const size_t slot_bytes = (size_t)L * HK * SEG;   // device bytes of one archived chunk
        const std::array<int64_t, 5> geom = {(int64_t)kt, D, ct, HK, L};
        if (!keep || geom != ageom_) arch_free();
        struct Drop { KvFlashGpuMass * g; bool on; ~Drop() { if (on) g->arch_free(); } } drop{this, !keep};
        ageom_ = geom;

        // The archive for this call: as it is, grown (kept slots carried over on the device), or started over. Each
        // is taken only if it passes the free-memory guard; starting over with exactly the evicted chunks is the
        // guard the call-scoped scorer had.
        static const std::vector<uint64_t> kNoVersions;
        KvFlashGpuArchivePlan plan = kvflash_gpu_archive_plan(aslots_, ev, keep ? *versions : kNoVersions);
        ggml_backend_dev_t dev = ggml_backend_get_device(be_);
        size_t fr = 0, tot = 0;
        if (dev) ggml_backend_dev_memory(dev, &fr, &tot);
        auto fits = [&](int cap, bool beside) {   // a call with `cap` slots; beside: the archive held now stays (as it is, or while it is copied)
            const size_t need = kvflash_gpu_mass_bytes(d, kt, ct, NB, cap, N, batch_bytes_) - (beside && cap == acap_ ? (size_t)cap * slot_bytes : 0);
            return kvflash_gpu_mass_fits(need, fr + (beside ? 0 : arch_bytes()), tot, kMarginBytes);
        };
        auto room = [&](int n) { return keep && n > 0 ? n + std::max(n / 8, 16) : n; };   // slots to allocate for n in use
        const int n_slots = (int)plan.chunk_of.size();
        bool over = plan.reset || aslots_.empty();
        if (!over && !(n_slots <= acap_ && fits(acap_, true)))
            over = !(fits(room(n_slots), true) && arch_alloc(room(n_slots), (int)aslots_.size(), true));
        if (over) {
            if (!aslots_.empty() && !plan.reset) plan = kvflash_gpu_archive_plan({}, ev, kNoVersions);
            aslots_.clear();
            const int cap = fits(room(E), false) ? room(E) : E;
            if (!fits(cap, false)) {
                const size_t need = kvflash_gpu_mass_bytes(d, kt, ct, NB, E, N, batch_bytes_);
                return fail("needs " + std::to_string(need >> 20) + " MiB (+" + std::to_string(kMarginBytes >> 20) + " MiB margin) but " +
                            std::to_string((fr + arch_bytes()) >> 20) + " MiB are free");
            }
            if (cap != acap_) {
                arch_free();
                if (cap > 0 && !arch_alloc(cap, 0, cap > E)) return fail("needs " + std::to_string(((size_t)cap * slot_bytes) >> 20) + " MiB for the evicted keys: allocation failed");
            }
        }
        const int U = (int)plan.upload.size(), n_kept = (int)plan.chunk_of.size() - U;
        const int A = E > 0 ? acap_ : 0;                  // archive slots in the graph
        const int NBT = NB + A, T = NBT * ct;
        chunk_of.resize((size_t)NB);
        chunk_of.insert(chunk_of.end(), plan.chunk_of.begin(), plan.chunk_of.end());
        chunk_of.resize((size_t)NBT, -1);
        const size_t stage_k = (size_t)U * slot_bytes, stage_total = stage_k + (size_t)N * QS * sizeof(float);
        const size_t out_bytes = (size_t)PL * NBT * N * sizeof(float);

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

        // Gather: the K to upload into the staging layout [layer][kv head][u][row], and the queries behind it.
        std::atomic<bool> absent{false};
        parallel_for(L, [&](int l) {
            for (int h = 0; h < HK; h++)
                for (int u = 0; u < U; u++) {
                    const uint8_t * s = host_seg(plan.upload[(size_t)u], l, h);
                    if (!s) { absent = true; continue; }
                    std::memcpy(stage + (((size_t)l * HK + h) * U + u) * SEG, s, SEG);
                }
        });
        if (absent) return fail("missing host K segment");
        for (int n = 0; n < N; n++) std::memcpy(qbuf + (size_t)n * QS, qs[(size_t)n], QS * sizeof(float));
        std::vector<float> bias((size_t)T);
        for (int t = 0; t < T; t++) bias[(size_t)t] = chunk_of[(size_t)(t / ct)] >= 0 ? 0.0f : -INFINITY;
        const auto t1 = clk::now();

        // Call-scoped device memory: all queries, bias; uploaded once, with the new archive slots [n_kept, n_kept + U).
        struct Scope {
            ggml_context * cctx = nullptr; ggml_backend_buffer_t cbuf = nullptr; ggml_gallocr_t ga = nullptr;
            ~Scope() { if (ga) ggml_gallocr_free(ga); if (cbuf) ggml_backend_buffer_free(cbuf); if (cctx) ggml_free(cctx); }
        } sc;
        ggml_init_params cp{ggml_tensor_overhead() * 4, nullptr, true};
        sc.cctx = ggml_init(cp);
        if (!sc.cctx) return fail("ggml_init failed");
        ggml_tensor * q_all = ggml_new_tensor_4d(sc.cctx, GGML_TYPE_F32, D, HQ, L, N);
        ggml_tensor * b_t = ggml_new_tensor_1d(sc.cctx, GGML_TYPE_F32, T);
        sc.cbuf = ggml_backend_alloc_ctx_tensors(sc.cctx, be_);
        if (!sc.cbuf) return fail("call buffer allocation failed");
        for (int l = 0; l < L && U; l++) {
            ggml_tensor * a = ak_[(size_t)l];
            if (U == acap_) ggml_backend_tensor_set_async(be_, a, stage + (size_t)l * HK * U * SEG, 0, ggml_nbytes(a));
            else for (int h = 0; h < HK; h++)
                ggml_backend_tensor_set_async(be_, a, stage + ((size_t)l * HK + h) * U * SEG, (size_t)h * a->nb[2] + (size_t)n_kept * SEG, (size_t)U * SEG);
        }
        ggml_backend_synchronize(be_);
        for (int c : plan.upload) aslots_.push_back({c, keep ? (*versions)[(size_t)c] : 0});
        tm.uploaded = U; tm.archived = keep ? (int)aslots_.size() : 0;
#ifdef LUCE_HAVE_KVFLASH_FUSED_MASS
        // The fused kernel: one pass over the pool K and the archive where they are, no logits. Any failure: the graph below.
        if (fused_ && fused_available(kt, D, L)) {
            tm.h2d = ms(t1, clk::now());
            const auto t2 = clk::now();
            luce::common::KvFlashFusedGeom g;
            g.L = L; g.HK = HK; g.Hq = HQ; g.D = D; g.ct = ct; g.C = sealed; g.nq = N;
            luce::common::KvFlashFusedK k;
            for (int l = 0; l < L; l++) {
                k.pool.push_back((const uint8_t *)pool_k[(size_t)l]->data);
                if (E) k.copy.push_back((const uint8_t *)ak_[(size_t)l]->data);
            }
            k.pool_nb2 = pool_k[0]->nb[2];
            if (E) k.copy_nb2 = ak_[0]->nb[2];
            k.src.assign((size_t)sealed, 0);
            for (int c = 0; c < sealed; c++) k.src[(size_t)c] = block_of[(size_t)c];
            for (int s = 0; s < (int)plan.chunk_of.size(); s++) {
                const int c = plan.chunk_of[(size_t)s];
                if (c >= 0 && c < sealed && block_of[(size_t)c] < 0) k.src[(size_t)c] = -1 - s;
            }
            const size_t QB = QS / 32;
            std::vector<int8_t> q8((size_t)N * QS);
            std::vector<float> qd((size_t)N * QB), om((size_t)N * sealed), ot((size_t)N * sealed), ox((size_t)N * sealed);
            parallel_for(N, [&](int n) { luce::common::kvflash_quantize_q8_1(qs[(size_t)n], (int)QS, &q8[(size_t)n * QS], &qd[(size_t)n * QB]); });
            if (!fws_) fws_ = luce::common::kvflash_fused_ws_new();
            luce::common::KvFlashFusedTiming ft;
            std::string e;
            if (luce::common::kvflash_fused_run(fws_, g, k, q8.data(), qd.data(), fused_part_bytes_, om.data(), ot.data(), ox.data(), &ft, &e)) {
                for (int n = 0; n < N; n++) {
                    outs[(size_t)n].assign(om.begin() + (size_t)n * sealed, om.begin() + (size_t)(n + 1) * sealed);
                    if (!planes) continue;
                    (*planes)[(size_t)n].top.assign(ot.begin() + (size_t)n * sealed, ot.begin() + (size_t)(n + 1) * sealed);
                    (*planes)[(size_t)n].max.assign(ox.begin() + (size_t)n * sealed, ox.begin() + (size_t)(n + 1) * sealed);
                }
                tm.fused = true; tm.gather = ms(t0, t1); tm.graph = ft.up_ms + ft.kernel_ms + ft.reduce_ms; tm.d2h = ft.down_ms;
                tm.total = ms(t0, clk::now()); tm.agg = ms(t2, clk::now()) - tm.graph - tm.d2h;   // agg: query quantization and the copies out
                tm.batches = ft.tiles; tm.peak_bytes = ft.partial_bytes + arch_bytes();
                if (timing) *timing = tm;
                return true;
            }
            if (!fused_warned_) { fused_warned_ = true; std::fprintf(stderr, "[kvflash] fused scorer unavailable: %s; using the logits graph\n", e.c_str()); }
        }
#endif
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
                if (A) lg = ggml_concat(ctx, lg, ggml_mul_mat(ctx, ak_[(size_t)l], qr), 0);        // [T, G*m, HK]
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
        tm.peak_bytes = arch_bytes() + ggml_backend_buffer_get_size(sc.cbuf) + galloc_peak;
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

    size_t arch_bytes() const { return abuf_ ? ggml_backend_buffer_get_size(abuf_) : 0; }
    void arch_free() {
        if (abuf_) ggml_backend_buffer_free(abuf_);
        if (actx_) ggml_free(actx_);
        abuf_ = nullptr; actx_ = nullptr; ak_.clear(); acap_ = 0; aslots_.clear();
    }
    // A new archive of `cap` slots in the ageom_ geometry; the first n_keep slots' bytes are carried over on the
    // device. zero: slots will stay unfilled; they are masked with -inf, which needs finite logits under it.
    // false: nothing changed.
    bool arch_alloc(int cap, int n_keep, bool zero) {
        const ggml_type kt = (ggml_type)ageom_[0];
        const int D = (int)ageom_[1], ct = (int)ageom_[2], HK = (int)ageom_[3], L = (int)ageom_[4];
        ggml_init_params ip{ggml_tensor_overhead() * (size_t)L, nullptr, true};
        ggml_context * nctx = ggml_init(ip);
        if (!nctx) return false;
        std::vector<ggml_tensor *> nk;
        for (int l = 0; l < L; l++) nk.push_back(ggml_new_tensor_3d(nctx, kt, D, (int64_t)cap * ct, HK));
        ggml_backend_buffer_t nbuf = ggml_backend_alloc_ctx_tensors(nctx, be_);
        if (!nbuf) { ggml_free(nctx); return false; }
        if (zero) ggml_backend_buffer_clear(nbuf, 0);
        if (n_keep > 0) {
            // ggml has no ranged device copy: a plain 2D tensor over each (layer, kv head) run of the old and the new buffer
            ggml_init_params tp{ggml_tensor_overhead() * (size_t)(2 * L * HK), nullptr, true};
            ggml_context * t = ggml_init(tp);
            if (!t) { ggml_backend_buffer_free(nbuf); ggml_free(nctx); return false; }
            for (int l = 0; l < L; l++)
                for (int h = 0; h < HK; h++) {
                    ggml_tensor * a = ggml_new_tensor_2d(t, kt, D, (int64_t)n_keep * ct), * b = ggml_new_tensor_2d(t, kt, D, (int64_t)n_keep * ct);
                    ggml_backend_tensor_alloc(ak_[(size_t)l]->buffer, a, (char *)ak_[(size_t)l]->data + (size_t)h * ak_[(size_t)l]->nb[2]);
                    ggml_backend_tensor_alloc(nk[(size_t)l]->buffer, b, (char *)nk[(size_t)l]->data + (size_t)h * nk[(size_t)l]->nb[2]);
                    ggml_backend_tensor_copy(a, b);
                }
            ggml_backend_synchronize(be_);
            ggml_free(t);
        }
        std::vector<KvFlashGpuArchiveSlot> kept(aslots_.begin(), aslots_.begin() + n_keep);
        arch_free();
        abuf_ = nbuf; actx_ = nctx; ak_ = std::move(nk); acap_ = cap; aslots_ = std::move(kept);
        return true;
    }

    ggml_backend_t            be_;
    ggml_backend_buffer_type_t host_buft_ = nullptr;
    size_t                    batch_bytes_;
    bool                      fused_ = false, fused_warned_ = false;
    size_t                    fused_part_bytes_ = (size_t)256 << 20;   // device bytes of one partials tile
    luce::common::KvFlashFusedWs * fws_ = nullptr;
    ggml_backend_buffer_t     stage_ = nullptr;   // pinned: evicted K staging, then the queries
    ggml_backend_buffer_t     out_ = nullptr;     // pinned: [planes, blocks, N] masses
    // Device archive of evicted K: ak_[l] = [D, acap_ * ct, Hkv]; aslots_[i] = what slot i holds (slots fill in order).
    ggml_context *            actx_ = nullptr;
    ggml_backend_buffer_t     abuf_ = nullptr;
    std::vector<ggml_tensor *> ak_;
    int                       acap_ = 0;
    std::vector<KvFlashGpuArchiveSlot> aslots_;
    std::array<int64_t, 5>    ageom_{};           // K type, D, ct, Hkv, layers the archive was built for
};

}  // namespace luce::common
