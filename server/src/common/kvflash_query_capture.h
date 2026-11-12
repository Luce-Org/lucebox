// Per-token query capture for selector v2 and the user-question queries.
//
// The prefill graph keeps the last kQueryCaptureTokens tokens' queries of every batch
// (TargetCache::q_cap_all), so the graph, the memory estimate and the backend agree on one number.

#pragma once

#include "kvflash_pool_flag.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace luce::common {

// Per-token queries the graph captures per batch.
constexpr int kQueryCaptureTokens = 64;

// Only a KVFlash process (LUCE_KVFLASH names a pool) keeps the per-token queries: a pooled prefill
// scores with them (selector v2, the user-question queries). Without a pool nothing is allocated.
// The gate reads the env only, not the resolved recall plan: every process that builds a cache
// (create_target_cache_partial, including layer-split shards) reserves the rows. Under tp_degree > 1
// kvflash_recall_features turns recall off, so that reservation is unused: a deliberate per-process
// cost of 64 x head_dim x n_head x n_full_attn x 4 B (about 25 MB at 256 x 24 x 16) in exchange for
// not threading the plan through every cache builder and the VRAM estimate.
inline bool kvflash_pool_process() {
    return kvflash_pool_requested(std::getenv("LUCE_KVFLASH"));
}
inline int kvflash_q_cap_all_rows() { return kvflash_pool_process() ? kQueryCaptureTokens : 0; }

// Strided read of the first n_q rows of every plane of q_cap_all ([fa][head][T][hd] f32, so rows of one plane are
// contiguous and planes are T*hd floats apart): arguments of ggml_backend_tensor_get_2d(offset 0), reading
// planes * n_q * hd floats instead of the whole tensor. The destination is packed [plane][n_q][hd].
struct KvflashQCapRead { size_t size, n_copies, stride_tensor, stride_data; };
inline KvflashQCapRead kvflash_q_cap_all_read(size_t T, size_t head_dim, size_t planes, size_t n_q) {
    return {n_q * head_dim * sizeof(float), planes, T * head_dim * sizeof(float), n_q * head_dim * sizeof(float)};
}
// [plane][n_q][hd] -> the scorer's per-token [t][plane][hd].
inline void kvflash_regroup_queries(const std::vector<float> & packed, size_t planes, size_t n_q, size_t head_dim,
                                    std::vector<float> & out) {
    out.resize(n_q * planes * head_dim);
    for (size_t t = 0; t < n_q; t++)
        for (size_t p = 0; p < planes; p++)
            std::copy_n(packed.data() + (p * n_q + t) * head_dim, head_dim, out.data() + (t * planes + p) * head_dim);
}

// Whether a graph writes the per-token query capture q_cap_all. Only the prefill windows read it (recall mode 4
// and the user-question queries); decode captures q_cap alone. q_cap_all has no tensor-parallel split, so TP never writes it.
inline bool kvflash_capture_q_all(bool recall_on, bool uq_on, int tp_degree, bool capture_qk) {
    return capture_qk && tp_degree <= 1 && (recall_on || uq_on);
}

// LUCE_KVFLASH_UQ_QUERIES=N: the latest human question's last N queries join the selector's.
// 0 when unset or invalid, at most the capture window.
inline int kvflash_uq_queries() {
    const char * e = std::getenv("LUCE_KVFLASH_UQ_QUERIES");
    if (!e || !*e) return 0;
    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    return (*end || v <= 0) ? 0 : (int)std::min<long>(v, kQueryCaptureTokens);
}

// Rows of the verify-query stage (TargetCache::q_stage): the spec verify pass
// stages the query of each of its rows so q_cap can take the last accepted one.
// Only a KVFlash process (LUCE_KVFLASH names a pool) allocates it.
inline int kvflash_q_stage_rows(int max_verify_tokens) {
    return kvflash_pool_process() ? std::max(max_verify_tokens, 0) : 0;
}

// Selector-v2 query tail (kvflash_q_tail_): rows are query vectors of `qsize` floats, oldest first, the last
// row being the prompt's final token (position end-1). Positions are implicit.
inline void kvflash_q_tail_append(std::vector<float> & tail, const std::vector<float> & rows, size_t qsize, int keep_rows) {
    tail.insert(tail.end(), rows.begin(), rows.end());
    const size_t keep = (size_t)std::max(keep_rows, 0) * qsize;
    if (tail.size() > keep) tail.erase(tail.begin(), tail.end() - (long)keep);
}

// A rewind to R recomputes [R, end) under a new pool: drop the tail rows at positions >= R so the second
// pass captures them again. A tail that is not whole rows cannot be addressed, so it is cleared.
inline void kvflash_q_tail_trim(std::vector<float> & tail, size_t qsize, int end, int r) {
    if (qsize == 0 || tail.size() % qsize != 0) { tail.clear(); return; }
    const size_t n = tail.size() / qsize;
    const size_t drop = std::min(n, (size_t)std::max(end - r, 0));
    tail.resize((n - drop) * qsize);
}

} // namespace luce::common
