// Sparse-mask flash attention: skipping the KV tiles every query masks out is bit-identical (HIP).
//
// ggml_flash_attn_ext_set_sparse_mask marks an attention whose mask leaves whole spans of the KV cache at -inf for
// every query: the DFlash drafter's ring, which spans kv_total slots while only the committed context and the noise
// columns are visible. The CUDA/HIP backend then scans the mask into a live-granule bitmap and the tile kernel skips
// the dead KV tiles. Each case here runs the same attention twice in one graph, without and with the mark, and
// compares the outputs byte for byte, over the drafter's shapes (head 128, 16 queries) and its mask layouts: a ring
// before it wraps, a sliding window with causal noise columns, a wrapped ring, scattered live spans and a ring holding
// only noise. A second run puts NaN in K and V wherever a whole aligned 128-slot block is dead (so every KV tile over it
// is skippable, whatever the kernel's tile width): the unmarked attention reads them and turns non-finite, the marked
// one must skip them and stay equal to its finite run, which proves the skip engaged.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace {

constexpr int D  = 128;    // head dim of the DFlash2 drafter
constexpr int Hq = 32;     // query heads
constexpr int Hk = 8;      // KV heads

uint32_t g_rng = 0x2545f491u;
float next_float() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((int32_t)(g_rng >> 8) - 8388608) / 8388608.0f; // [-1, 1)
}

// visible(q, kv): whether query row q may attend to KV slot kv.
using Visible = std::function<bool(int, int)>;

struct Case {
    const char * name;
    int S;      // KV slots (the ring's kv_total)
    int nq;     // queries
    Visible visible;
};

struct Run {
    std::vector<float> plain, sparse;
    size_t nan_slots = 0;     // slots that hold NaN (nan_dead runs)
    bool ok = false;
};

// Both runs of a case (finite, then NaN in the dead blocks) draw the same values from the same seed.
Run run_case(ggml_backend_t gpu, const Case & c, bool nan_dead, uint32_t seed) {
    Run r;
    g_rng = seed;
    ggml_init_params params = {ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * Q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, c.nq, Hq);
    ggml_tensor * K    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, c.S, Hk);
    ggml_tensor * V    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, c.S, Hk);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.S, c.nq);
    for (ggml_tensor * t : {Q, K, V, mask}) ggml_set_input(t);
    const float scale = 1.0f / std::sqrt((float) D);
    ggml_tensor * plain  = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
    ggml_tensor * sparse = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_sparse_mask(sparse, true);
    if (!ggml_flash_attn_ext_get_sparse_mask(sparse) || ggml_flash_attn_ext_get_sparse_mask(plain)) {
        std::printf("FAIL %s: the sparse-mask op param does not read back\n", c.name);
        ggml_free(ctx);
        return r;
    }
    ggml_set_output(plain);
    ggml_set_output(sparse);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, plain);
    ggml_build_forward_expand(graph, sparse);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, graph)) {
        std::printf("FAIL %s: allocation\n", c.name);
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return r;
    }

    std::vector<float> qv(ggml_nelements(Q));
    for (float & x : qv) x = next_float();
    ggml_backend_tensor_set(Q, qv.data(), 0, ggml_nbytes(Q));

    // A slot is dead when no query sees it.
    std::vector<char> dead((size_t) c.S, 1);
    std::vector<ggml_fp16_t> mv((size_t) c.S * c.nq);
    for (int q = 0; q < c.nq; ++q) {
        for (int kv = 0; kv < c.S; ++kv) {
            const bool vis = c.visible(q, kv);
            mv[(size_t) q * c.S + kv] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
            if (vis) dead[(size_t) kv] = 0;
        }
    }
    ggml_backend_tensor_set(mask, mv.data(), 0, mv.size() * sizeof(ggml_fp16_t));

    // NaN goes only where the whole aligned 128-slot block is dead: every KV tile over it is then skippable.
    std::vector<char> poison((size_t) c.S, 0);
    for (int b0 = 0; nan_dead && b0 + 128 <= c.S; b0 += 128) {
        bool all_dead = true;
        for (int s = b0; s < b0 + 128; ++s) all_dead = all_dead && dead[(size_t) s];
        for (int s = b0; all_dead && s < b0 + 128; ++s) poison[(size_t) s] = 1;
    }
    for (char x : poison) r.nan_slots += x;

    for (ggml_tensor * t : {K, V}) {
        std::vector<ggml_fp16_t> kv((size_t) D * c.S * Hk);
        for (int h = 0; h < Hk; ++h) {
            for (int s = 0; s < c.S; ++s) {
                for (int d = 0; d < D; ++d) {
                    const float drawn = next_float();      // drawn either way: both runs see the same live data
                    const float x = poison[(size_t) s] ? std::numeric_limits<float>::quiet_NaN() : drawn;
                    kv[((size_t) h * c.S + s) * D + d] = ggml_fp32_to_fp16(x);
                }
            }
        }
        ggml_backend_tensor_set(t, kv.data(), 0, kv.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS) {
        std::printf("FAIL %s: compute\n", c.name);
    } else {
        r.plain.resize(ggml_nelements(plain));
        r.sparse.resize(ggml_nelements(sparse));
        ggml_backend_tensor_get(plain, r.plain.data(), 0, ggml_nbytes(plain));
        ggml_backend_tensor_get(sparse, r.sparse.data(), 0, ggml_nbytes(sparse));
        r.ok = true;
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return r;
}

bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// The ring of the DFlash drafter: cap context slots, then the noise columns (16), padded to kv_total.
constexpr int kCap = 4096, kNoise = 16, kTotal = 4128;

Visible ring(int lo, int hi, bool causal_noise) {
    return [=](int q, int kv) {
        if (kv >= kCap) return kv < kCap + kNoise && (!causal_noise || kv - kCap <= q);
        return kv >= lo && kv < hi;
    };
}

} // namespace

int main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) {
        std::printf("SKIP: no CUDA/HIP device\n");
        return 0;
    }

    const std::vector<Case> cases = {
        {"ring ctx 1",            kTotal, 16, ring(0, 1, false)},
        {"ring ctx 31",           kTotal, 16, ring(0, 31, false)},
        {"ring ctx 32",           kTotal, 16, ring(0, 32, false)},
        {"ring ctx 33",           kTotal, 16, ring(0, 33, false)},
        {"ring ctx 817",          kTotal, 16, ring(0, 817, false)},
        {"ring ctx 4095",         kTotal, 16, ring(0, 4095, false)},
        {"ring wrapped",          kTotal, 16, ring(0, kCap, false)},
        {"ring noise only",       kTotal, 16, ring(0, 0, false)},
        {"swa window 600..1500",  kTotal, 16, ring(600, 1500, true)},
        {"swa window 3000..4096", kTotal, 16, ring(3000, kCap, true)},
        {"scattered spans",       kTotal, 16, [](int, int kv) {
             return kv < 100 || (kv >= 1000 && kv < 1100) || (kv >= 2050 && kv < 2051) ||
                    (kv >= kCap && kv < kCap + kNoise);
         }},
        {"per-row causal tail",   kTotal, 16, [](int q, int kv) { return kv < 500 + 37 * q; }},
        {"proxy ring 1056 ctx 700", 1056, 16, [](int, int kv) { return kv < 700 || (kv >= 1024 && kv < 1040); }},
        {"one query ctx 817",     kTotal, 1,  ring(0, 817, false)},
        {"64 queries ctx 817",    kTotal, 64, ring(0, 817, false)},
    };

    int failed = 0;
    uint32_t seed = 0x2545f491u;
    for (const Case & c : cases) {
        seed = seed * 747796405u + 2891336453u;
        const Run fin = run_case(gpu, c, false, seed);
        const Run nan = run_case(gpu, c, true, seed);
        if (!fin.ok || !nan.ok) {
            ++failed;
            continue;
        }
        const bool equal = std::memcmp(fin.plain.data(), fin.sparse.data(), fin.plain.size() * sizeof(float)) == 0;
        const bool finite = all_finite(fin.plain);
        // With NaN in the dead slots the marked run must still match the finite one exactly (it never read them).
        // Whether the unmarked run turns non-finite tells whether any slot was dead in a skippable tile.
        const bool skipped_equal =
            std::memcmp(nan.sparse.data(), fin.sparse.data(), fin.sparse.size() * sizeof(float)) == 0;
        const bool plain_poisoned = !all_finite(nan.plain);
        bool ok = equal && finite && skipped_equal;
        if (nan.nan_slots > 0) {
            ok = ok && plain_poisoned;      // a dead 128-slot block exists: the unmarked run must have read it
        }
        std::printf("%s %-26s S=%4d nq=%2d: marked == unmarked %s, finite %s; NaN in %zu dead slots: marked "
                    "unchanged %s, unmarked poisoned %s\n",
                    ok ? "PASS" : "FAIL", c.name, c.S, c.nq, equal ? "yes" : "NO", finite ? "yes" : "NO",
                    nan.nan_slots, skipped_equal ? "yes" : "NO", plain_poisoned ? "yes" : "no");
        failed += !ok;
    }
    ggml_backend_free(gpu);
    std::printf("%s: %d of %zu cases failed\n", failed ? "FAIL" : "PASS", failed, cases.size());
    return failed ? 1 : 0;
}
