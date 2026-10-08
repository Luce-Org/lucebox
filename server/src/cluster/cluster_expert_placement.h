// cluster_expert_placement.h - which rank owns which routed expert.
//
// Generalizes the two-owner hot/cold split of moe_hybrid_placement.h to N
// ranks. Every (layer, expert) has exactly one owner rank, or is marked
// kReplicated: replicated experts are resident on every rank and the rank
// that computes a replicated expert for token slot t is (expert + t) % N, so
// each route is still evaluated exactly once cluster-wide. Ranks evaluate
// their local experts as owner partial sums (the existing masked-route path
// of build_moe_hybrid_ffn_graph with cold owner None) and the cluster
// all-reduce sums the partials.
//
// The placement is built once at startup (uniform, balanced from routing
// stats, or loaded from JSON), hashed, and the hash is exchanged in Hello so
// ranks with different placements refuse to form a cluster.

#pragma once

#include "common/moe_hybrid_placement.h"
#include "common/moe_hybrid_routing_stats.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::cluster {

inline constexpr int32_t kReplicated = -1;

struct ClusterExpertPlacement {
    int n_ranks  = 0;
    int n_layer  = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int replicate_hot = 0;   // replicated experts per layer (informational)
    // Diagnostics only: set by load_json so a hand-written file may leave a
    // rank with no owned expert in a layer (e.g. rank 0 owns everything and
    // rank 1 nothing, which must reproduce the single-node output exactly if
    // the cluster path is correct). Builders never set it.
    bool allow_empty_shards = false;

    // Flattened [layer * n_expert + expert] -> owner rank, or kReplicated.
    std::vector<int32_t> owner;

    // ── Queries ──────────────────────────────────────────────────────────
    int32_t owner_of(int layer, int expert) const;
    bool    is_replicated(int layer, int expert) const;
    // Rank that evaluates (layer, expert) for token slot `slot` (0-based
    // position inside the current batch). Owned experts: the owner;
    // replicated experts: (expert + slot) % n_ranks.
    int32_t slot_owner(int layer, int expert, int slot) const;
    // Experts resident on `rank` for `layer`: owned + replicated, ascending.
    std::vector<int32_t> resident_experts(int layer, int rank) const;
    // Number of owned (non-replicated) experts of `rank` in `layer`.
    int owned_count(int layer, int rank) const;
    // Sum over layers of owned_count.
    int total_owned(int rank) const;

    bool valid(std::string * err = nullptr) const;
    bool matches(const common::MoeHybridConfig & cfg) const;
    uint64_t hash() const;

    // Rank-local view for the existing hybrid machinery: hot set = resident
    // experts of `rank`, cold set = everything else (never materialized,
    // cold owner None). hot_expert_ids are sorted ascending so the LUT order
    // is deterministic across ranks.
    bool to_rank_placement(int rank, common::MoeHybridPlacement & out,
                           std::string * err = nullptr) const;

    // Per-(layer, slot) LUT rows for replicated experts: for every expert e
    // resident on `rank`, valid[layer][slot][e] says whether this rank
    // evaluates e at that slot. Owned experts are always valid; replicated
    // ones only when slot_owner == rank. n_slots is the verify width / batch
    // size. Output is [n_layer][n_slots][n_expert] flattened, 0/1.
    void slot_validity(int rank, int n_slots, std::vector<uint8_t> & out) const;

    // ── Builders ─────────────────────────────────────────────────────────
    // expert e of every layer -> rank e % n_ranks. When replicate_hot > 0 and
    // stats is non-null, the replicate_hot hottest experts of each layer
    // (by stats) become kReplicated instead.
    static bool build_uniform(int n_ranks,
                              const common::MoeHybridConfig & cfg,
                              int replicate_hot,
                              const common::MoeHybridRoutingStats * stats,
                              ClusterExpertPlacement & out,
                              std::string * err = nullptr);

    // Longest-processing-time greedy: per layer, experts sorted by hotness
    // descending, each assigned to the rank with the smallest accumulated
    // hotness, subject to |owned_count(rank) - n_expert/n_ranks| <= 1 so no
    // rank's shard grows beyond one expert of the mean (memory balance). The
    // replicate_hot hottest experts are marked kReplicated first.
    static bool build_balanced(int n_ranks,
                               const common::MoeHybridRoutingStats & stats,
                               int replicate_hot,
                               ClusterExpertPlacement & out,
                               std::string * err = nullptr);

    // JSON: {"n_ranks":N,"n_layer":L,"n_expert":E,"n_expert_used":K,
    //        "replicate_hot":R,"owner":[[...E ints...] x L]}
    bool save_json(const std::string & path, std::string * err = nullptr) const;
    static bool load_json(const std::string & path, ClusterExpertPlacement & out,
                          std::string * err = nullptr);

    // Human-readable summary for the startup log: per-rank owned counts,
    // replicated count, expected imbalance from stats if given.
    std::string describe(const common::MoeHybridRoutingStats * stats = nullptr) const;
};

}  // namespace luce::cluster
