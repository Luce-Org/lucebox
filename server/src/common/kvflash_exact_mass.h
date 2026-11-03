// Exact attention-mass chunk scorer: types shared by the AVX2 (kvflash_exact_fast.h) and GPU (kvflash_gpu_mass.h)
// scorers. The plain reference implementation is the test oracle (test/unit/kvflash_mass_synth.h).
//
// For one query, the softmax mass each sealed chunk receives over ALL sealed chunks, averaged over the
// full-attention layers and query heads. Replaces the cosine-on-pooled-keys
// router with the quantity attention actually computes.
//
//   s_t(l,h)    = (q[l][h] . k_t[l][h / group]) * scale     scale = 1/sqrt(head_dim)
//   lse_c(l,h)  = logsumexp of s_t over the tokens of chunk c
//   Z(l,h)      = logsumexp of lse_c over chunks
//   score[c]    = mean over (l,h) of exp(lse_c - Z)
//
// scale matches kq_scale in qwen35_target_graph.cpp. q and K live in the same
// basis (post-RoPE, and the same FWHT rotation when the cache is rotated), so
// the dot products equal the attention graph's. K is read through ggml's type
// traits (to_float): f32/f16/q8_0/q4_0/...

#pragma once

#include "kvflash_qk.h"
#include "ggml.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace luce::common {

// seg(c, layer, kv_head) -> the chunk's K rows for that (layer, head):
// chunk_tokens contiguous rows of head_dim elements of `type`; nullptr = absent
// (the chunk then contributes nothing and scores 0).
using KvFlashKSegFn = std::function<const uint8_t *(int c, int l, int h)>;

struct KvFlashExactKeys {
    int           n_chunks     = 0;
    int           chunk_tokens = 0;
    ggml_type     type         = GGML_TYPE_F16;
    KvFlashKSegFn seg;
};

constexpr int kExactTopPlanes = 8;   // planes averaged by KvFlashMassPlanes::top

// Per-chunk aggregates over the (layer, head) planes besides the mean.
struct KvFlashMassPlanes {
    std::vector<float> top;   // mean of the kExactTopPlanes largest plane masses
    std::vector<float> max;   // the largest plane mass
};

// Thread-summed CPU time per phase (ms), filled when a non-null pointer is passed.
struct KvFlashExactTiming {
    std::atomic<int64_t> dequant_ns{0}, dot_ns{0}, exp_ns{0};
};

}  // namespace luce::common
