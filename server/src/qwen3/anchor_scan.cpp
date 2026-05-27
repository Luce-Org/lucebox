#include "anchor_scan.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace dflash::qwen3 {

// Force chunk and its radius-neighborhood into `forced`.
static void force_neighborhood(std::vector<uint8_t>& forced, int n_chunks,
                                int chunk, int radius) {
    int lo = std::max(0, chunk - radius);
    int hi = std::min(n_chunks - 1, chunk + radius);
    for (int c = lo; c <= hi; ++c) forced[(size_t)c] = 1;
}

void scan_and_force(
    const std::vector<int32_t>& ids,
    int body_end,
    const std::vector<int32_t>& query_pool,
    const AnchorScanCfg& cfg,
    std::vector<uint8_t>& forced)
{
    const int n_chunks = (int)forced.size();
    const int ngram    = cfg.ngram;
    const int search_end = std::max(0, body_end - ngram);

    for (int qi = 0; qi + ngram <= (int)query_pool.size(); ++qi) {
        int hits = 0;
        int hit_pos[8];
        for (int p = 0; p <= search_end && hits <= cfg.max_anchor_hits; ++p) {
            bool same = true;
            for (int k = 0; k < ngram; ++k) {
                if (ids[(size_t)p + k] != query_pool[(size_t)qi + k]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                if (hits < 8) hit_pos[hits] = p;
                ++hits;
            }
        }
        if (hits > 0 && hits <= cfg.max_anchor_hits) {
            for (int i = 0; i < hits && i < 8; ++i) {
                force_neighborhood(forced, n_chunks,
                                   hit_pos[i] / cfg.chunk_size,
                                   cfg.anchor_radius);
            }
        }
    }
}

// Helper: count set entries in forced.
static int count_set(const std::vector<uint8_t>& forced) {
    int n = 0;
    for (uint8_t v : forced) n += (v != 0);
    return n;
}

void scan_and_force_transitive(
    const std::vector<int32_t>& ids,
    int body_end,
    const std::vector<int32_t>& initial_query_pool,
    const AnchorScanCfg& cfg,
    int max_iters,
    std::vector<uint8_t>& forced)
{
    auto pool = initial_query_pool;
    const int n_chunks = (int)forced.size();

    // Precompute token frequencies in body once.
    std::unordered_map<int32_t, int> body_freq;
    body_freq.reserve((size_t)body_end);
    for (int j = 0; j < body_end; ++j) ++body_freq[ids[(size_t)j]];

    // Build inverted index: token -> list of body positions (for rare tokens only).
    std::unordered_map<int32_t, std::vector<int>> rare_positions;
    if (cfg.rare_token_max_freq > 0) {
        for (auto& kv : body_freq) {
            if (kv.second <= cfg.rare_token_max_freq) {
                rare_positions[kv.first] = {};
            }
        }
        for (int p = 0; p < body_end; ++p) {
            auto it = rare_positions.find(ids[(size_t)p]);
            if (it != rare_positions.end()) it->second.push_back(p);
        }
    }

    // Pass-1: run the initial scan.
    const int count_before_pass1 = count_set(forced);
    scan_and_force(ids, body_end, pool, cfg, forced);
    const int gained_pass1 = count_set(forced) - count_before_pass1;

    // Gating: if pass-1 already found many anchors, skip the cascade entirely.
    if (cfg.cascade_min_anchor_count > 0 && gained_pass1 >= cfg.cascade_min_anchor_count) {
        return;
    }

    // Cascade loop: expand pool with newly-forced tokens and re-scan.
    std::vector<uint8_t> prev_forced;
    for (int it = 0; it < max_iters; ++it) {
        prev_forced = forced;

        // Rare-token single-match: worklist-driven so cascades within a pass are
        // caught (e.g. hop3 forces hop2 which forces hop1 in one outer iteration).
        if (cfg.rare_token_max_freq > 0) {
            std::vector<int> worklist;
            for (int c = 0; c < n_chunks; ++c) {
                if (forced[c] && !prev_forced[c]) worklist.push_back(c);
            }
            // On first iteration, seed from everything forced so far (pass-1 results).
            if (it == 0) {
                worklist.clear();
                for (int c = 0; c < n_chunks; ++c) {
                    if (forced[c]) worklist.push_back(c);
                }
            }
            for (int wi = 0; wi < (int)worklist.size(); ++wi) {
                int c = worklist[wi];
                int s = c * cfg.chunk_size;
                int e = std::min(body_end, (c + 1) * cfg.chunk_size);
                for (int j = s; j < e; ++j) {
                    auto it2 = rare_positions.find(ids[(size_t)j]);
                    if (it2 == rare_positions.end()) continue;
                    for (int p : it2->second) {
                        int target_c = p / cfg.chunk_size;
                        if (!forced[(size_t)target_c]) {
                            force_neighborhood(forced, n_chunks,
                                               target_c, cfg.anchor_radius);
                            worklist.push_back(target_c);
                        }
                    }
                }
            }
        }

        // Hard cap: if we exceeded max_forced_count, revert this iteration and stop.
        if (count_set(forced) > cfg.max_forced_count) {
            forced = prev_forced;
            break;
        }

        if (forced == prev_forced) break;

        // Expand pool with tokens from newly-forced chunks (feeds next 4-gram pass).
        for (int c = 0; c < n_chunks; ++c) {
            if (forced[c] && !prev_forced[c]) {
                int s = c * cfg.chunk_size;
                int e = std::min((int)ids.size(), (c + 1) * cfg.chunk_size);
                for (int j = s; j < e; ++j) pool.push_back(ids[j]);
            }
        }

        // 4-gram scan with expanded pool for next iteration.
        prev_forced = forced;
        scan_and_force(ids, body_end, pool, cfg, forced);

        // Hard cap check after 4-gram expansion too.
        if (count_set(forced) > cfg.max_forced_count) {
            forced = prev_forced;
            break;
        }
    }
}

} // namespace dflash::qwen3
