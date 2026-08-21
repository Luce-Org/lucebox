// DDTree — Dynamic Draft Tree for speculative decoding.
//
// Port of build_ddtree_tree() from liranringel/ddtree/ddtree.py.
// Builds a best-first tree from per-position top-K log-probability
// distributions (from the draft model) for tree-structured verify.
//
// Self-contained: depends only on standard library.

#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace dflash::common {

// A flat DFS-ordered tree built from the draft's top-K softmax distributions.
// Slot 0 is the tree root (the bonus token from the previous spec round);
// slots 1..n_nodes are the DFS-ordered tree nodes.
struct DDTree {
    int                         n_nodes = 0;          // excludes root
    std::vector<int32_t>        token_ids;            // size n_nodes
    std::vector<int>            depths;               // size n_nodes (1..L)
    std::vector<int>            parents;              // size n_nodes + 1
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // size n_nodes + 1
    std::vector<uint8_t>        visibility;           // (1 + n_nodes)^2 row-major
};

// Per-position top-K softmax extraction. Computes log-probabilities via a
// single pass over the vocab that maintains top-K in a heap and computes
// logsumexp online. Runs on CPU. Parallelized across positions via OpenMP.
//
// Input:  logits [n_positions × vocab] f32
// Output: out_log_probs [n_positions × K] f32, out_token_ids [n_positions × K] i32
//         both sorted by log-probability DESCENDING (rank 0 = argmax).
void extract_draft_topk(const float * logits,
                        int n_positions, int vocab, int K,
                        float * out_log_probs,
                        int32_t * out_token_ids,
                        float temperature = 1.0f);

// Build a DDTree from per-position top-K distributions.
//
// top_log_probs: [L × K]  the drafter's per-position top-K log-probabilities
// top_token_ids: [L × K]  matching token ids, rank 0 = argmax per position
// L:             max tree depth
// K:             top-K per position
// budget:        maximum number of non-root tree nodes
// chain_seed:    pre-seed full top-1 chain (defensive) vs pure best-first
// tau_tree:      SpecLA confidence margin (arXiv:2607.16673 §6.1): only
//                candidates with cumulative path log-probability
//                q(v) >= q* - tau_tree are expanded, where q* is the best
//                candidate score. Because best-first pops in descending q,
//                one comparison prunes width and depth jointly, and the
//                retained set is ancestor-closed by construction. The node
//                budget still applies on top. Non-finite (default) = off.
//                The chain seed is pruned at the first out-of-window depth,
//                preserving ancestor closure.
DDTree build_ddtree(const float * top_log_probs,
                    const int32_t * top_token_ids,
                    int L, int K, int budget,
                    bool chain_seed = true,
                    float tau_tree = std::numeric_limits<float>::infinity());

// Branch-conditioned variant used by SpecLA. The callback receives the token
// prefix of a tree node and the 1-based depth to predict next, and returns K
// sorted log-probabilities/token ids for that exact prefix.
using DDTreeConditionalTopK = std::function<bool(
    const std::vector<int32_t> & prefix,
    int next_depth,
    std::vector<float> & log_probs,
    std::vector<int32_t> & token_ids)>;

DDTree build_ddtree_conditional(const DDTreeConditionalTopK & next_topk,
                                int L, int K, int budget,
                                bool chain_seed = true,
                                float tau_tree =
                                    std::numeric_limits<float>::infinity());

// Walk the verified tree following the target's argmax (posterior) at each
// node. Returns the list of flat-tree indices that make up the accepted path
// (starting at root), plus the next "bonus" token.
std::vector<int> follow_verified_tree(const DDTree & tree,
                                      const int32_t * posterior,
                                      int & out_next_token,
                                      int * out_node_idx = nullptr);

}  // namespace dflash::common
