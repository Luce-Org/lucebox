// Fused exact chunk-mass kernel (Amendment 10): one pass over K, per (query, plane, chunk) a local max and local sum-exp, no
// stored logits; then a small reduce per plane and a per-chunk aggregate (mean, top-8 mean, max over planes).
// Numerics follow ggml's MMQ path for q8_0 K x f32 q (quantize_mmq_q8_1, layout D4): per 32-block d_inv = 127 / amax, q = roundf(x * d_inv),
// dQ = 1 / d_inv as a float; the dot is an int8 dp4a per block, logit = scale * sum_b (dK_b * dQ_b) * isum_b in ascending block order.
// Compiled on CUDA and HIP (LANGUAGE HIP through the hip_compat shim, as geometric_draft_topk_cuda.cu); LUCE_HAVE_KVFLASH_FUSED_MASS.

#pragma once

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

// The int8 WMMA kernel body is compiled only where the device pass defines __gfx1200__ / __gfx1201__ (see LUCE_FUSED_WMMA in the .cu).
// The host cannot see those macros, so the runtime gate must name exactly the same archs; "gfx12-generic" and other gfx12xx parts run dp4a.
// gcnArchName may carry a feature suffix ("gfx1201:sramecc-:xnack-"), so match the exact name up to ':'.
inline bool kvflash_fused_arch_has_wmma(const char * gcn_arch_name) {
    if (!gcn_arch_name) return false;
    const size_t n = std::strcspn(gcn_arch_name, ":");
    return n == 7 && (std::strncmp(gcn_arch_name, "gfx1200", 7) == 0 || std::strncmp(gcn_arch_name, "gfx1201", 7) == 0);
}

constexpr int kFusedMaxLayers = 64;

struct KvFlashFusedGeom {
    int L = 0, HK = 0, Hq = 0, D = 0, ct = 0;   // Hq % HK == 0, D % 32 == 0
    int C = 0;                                  // sealed chunks (the sealed bound: only these are read)
    int nq = 0;
    int impl = 0;   // 0 auto (int8 WMMA on gfx12, dp4a elsewhere), 1 dp4a, 2 WMMA (fails off gfx12)
};

// Where the K of each chunk lives: layer l's q8_0 tensor [D, tokens, HK] in device memory, rows of 34-byte blocks.
// src[c] >= 0: block src[c] of the pool tensors; src[c] < 0: chunk (-1 - src[c]) of the device copy.
struct KvFlashFusedK {
    std::vector<const uint8_t *> pool, copy;   // per layer, device pointers (copy may be empty when no chunk uses it)
    size_t pool_nb2 = 0, copy_nb2 = 0;         // bytes between kv heads
    std::vector<int> src;                      // [C]
};

struct KvFlashFusedTiming {
    double up_ms = 0, kernel_ms = 0, reduce_ms = 0, down_ms = 0;
    size_t partial_bytes = 0;   // peak device bytes of the partials tile
    int tiles = 0;
};

class KvFlashFusedWs;
KvFlashFusedWs * kvflash_fused_ws_new();
void kvflash_fused_ws_free(KvFlashFusedWs *);

// q8 (host): [nq][L*Hq][D] int8 and qd: [nq][L*Hq][D/32] float scales. Outputs (host) [nq][C] each.
bool kvflash_fused_run(KvFlashFusedWs * ws, const KvFlashFusedGeom & g, const KvFlashFusedK & k, const int8_t * q8, const float * qd,
                       size_t max_partial_bytes, float * out_mean, float * out_top, float * out_max, KvFlashFusedTiming * tm,
                       std::string * err);

}  // namespace luce::common
