// DDTree implementation.
// See ddtree.h for public interface.

#include "ddtree.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dflash::common {

void extract_draft_topk(const float * logits,
                        int n_positions, int vocab, int K,
                        float * out_log_probs,
                        int32_t * out_token_ids,
                        float temperature) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);

        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            if (l > running_max) {
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);

        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

namespace {

// Fill the ancestor-only visibility mask: slot v can see every slot on the
// root-to-v path (including itself), nothing else. visibility is row-major
// (1 + n_nodes)^2, iteration order exploits parents[i] < i (DFS/level order).
void build_visibility(DDTree & tree) {
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t)N * N, 0);
    tree.visibility[0] = 1;  // root sees itself
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[(size_t)i];
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] =
                tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }
}

// Shared best-first DDTree construction.
//
// `topk(prefix, depth, log_probs, token_ids)` returns the depth-th (1-based)
// position's sorted top-K distribution conditioned on `prefix` (the token ids
// chosen so far). It fills both out vectors with exactly K entries on success
// and returns false on failure (e.g. an exact prefix that cannot be scored).
// `depth` is 1-based and `prefix` contains the previously chosen token ids.
//
// Candidates are kept in a max-heap keyed by cumulative path log-probability
// q(v); popping in descending q makes the SpecLA confidence window
// (keep q(v) >= q* - tau_tree) a single early-stop comparison.
//
// `eager_siblings` selects between the two historical expansion schedules:
// the conditional builder pushes every sibling/child rank eagerly, while the
// precomputed builder pushes only the next sibling and rank-0 child lazily.
// The schedules produce the same candidate set, but `std::priority_queue`
// tie-breaks differently on equal cumulative scores, so each public builder
// keeps its original schedule.
DDTree build_ddtree_impl(const DDTreeConditionalTopK & topk,
                         int L, int K, int budget,
                         bool chain_seed,
                         bool eager_siblings,
                         float tau_tree,
                         const float * precomputed_log_probs,
                         const int32_t * precomputed_token_ids) {
    DDTree tree;
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();

    // A negative tau would reject even q* itself and leave a degenerate
    // proposal (the conditional Qwen path treats that as a failure). Clamp it
    // to zero: q* still passes, and positive margins prune normally.
    if (tau_tree < 0.0f) tau_tree = 0.0f;

    if (budget <= 0 || L <= 0 || K <= 0) {
        tree.visibility.assign(1, 1);
        return tree;
    }

    // Precomputed rows are prefix-independent, so the lazy builder can index
    // them directly without allocating/copying two K-element vectors for
    // every sibling and child expansion.
    const bool precomputed = !eager_siblings && precomputed_log_probs &&
                             precomputed_token_ids;

    // Fetch the depth-1 distribution. Its rank-0 score is q*, the best
    // candidate score in the whole tree.
    std::vector<float> lp;
    std::vector<int32_t> ids;
    if (!precomputed &&
        (!topk({}, 1, lp, ids) ||
         (int)lp.size() < K || (int)ids.size() < K)) {
        tree.visibility.assign(1, 1);
        return tree;
    }
    const float q_star = precomputed ? precomputed_log_probs[0] : lp[0];

    struct Candidate {
        float                logw;
        int                  parent;
        int                  depth;
        int                  rank;
        int32_t              token;
        std::vector<int32_t> path;  // token ids root->this node
    };
    struct Worse {
        bool operator()(const Candidate & a, const Candidate & b) const {
            return a.logw < b.logw;
        }
    };
    std::priority_queue<Candidate, std::vector<Candidate>, Worse> heap;

    auto push_scored_candidate = [&](int parent, int depth, float logw,
                                     const std::vector<int32_t> & prefix,
                                     int rank, int32_t token) {
        Candidate c;
        c.logw   = logw;
        c.parent = parent;
        c.depth  = depth;
        c.rank   = rank;
        c.token  = token;
        if (!precomputed) {
            c.path = prefix;
            c.path.push_back(c.token);
        }
        heap.push(std::move(c));
    };

    // Push one rank of one depth's distribution. The candidate's path is
    // `prefix` + the candidate's own token.
    auto push_candidate = [&](int parent, int depth, float parent_logw,
                              const std::vector<int32_t> & prefix,
                              int rank,
                              const std::vector<float> & probs,
                              const std::vector<int32_t> & toks) {
        push_scored_candidate(parent, depth,
                              parent_logw + probs[(size_t)rank],
                              prefix, rank, toks[(size_t)rank]);
    };

    // Push ranks [first_rank, K) of one depth's distribution as siblings of
    // `parent`.
    auto push_children = [&](int parent, int depth, float parent_logw,
                             const std::vector<int32_t> & prefix,
                             const std::vector<float> & probs,
                             const std::vector<int32_t> & toks,
                             int first_rank) {
        for (int rank = first_rank; rank < K; ++rank) {
            push_candidate(parent, depth, parent_logw, prefix, rank,
                           probs, toks);
        }
    };

    if (chain_seed) {
        // Defensively pre-seed the top-1 chain (up to the budget), preserving
        // ancestor closure: stop at the first depth whose cumulative top-1
        // score leaves the confidence window (every deeper descendant would
        // be out of window too).
        std::vector<int32_t> prefix;
        float cumulative = 0.0f;
        int parent = 0;
        std::vector<float> cur_lp = lp;
        std::vector<int32_t> cur_ids = ids;
        const int chain_depth = std::min(L, budget);
        for (int depth = 1; depth <= chain_depth; ++depth) {
            const int row = (depth - 1)*K;
            const float rank0_logp = precomputed
                ? precomputed_log_probs[row] : cur_lp[0];
            const int32_t rank0_token = precomputed
                ? precomputed_token_ids[row] : cur_ids[0];
            const float next_logw = cumulative + rank0_logp;
            if (q_star - next_logw > tau_tree) break;

            // Sibling candidates below the top-1 at this depth.
            if (eager_siblings) {
                push_children(parent, depth, cumulative, prefix,
                              cur_lp, cur_ids, 1);
            } else if (K > 1) {
                if (precomputed) {
                    push_scored_candidate(
                        parent, depth,
                        cumulative + precomputed_log_probs[row + 1],
                        prefix, 1, precomputed_token_ids[row + 1]);
                } else {
                    push_candidate(parent, depth, cumulative, prefix, 1,
                                   cur_lp, cur_ids);
                }
            }

            const int node = tree.n_nodes + 1;
            const int32_t token = rank0_token;
            tree.token_ids.push_back(token);
            tree.depths.push_back(depth);
            tree.parents.push_back(parent);
            tree.child_maps.emplace_back();
            tree.child_maps[(size_t)parent][token] = node;
            tree.n_nodes++;

            if (!precomputed) prefix.push_back(token);
            cumulative = next_logw;
            parent = node;

            if (depth == L) break;
            if (!precomputed &&
                (!topk(prefix, depth + 1, cur_lp, cur_ids) ||
                 (int)cur_lp.size() < K || (int)cur_ids.size() < K)) {
                break;
            }
        }
    } else {
        // Pure best-first: every depth-1 candidate is a root child. The lazy
        // schedule starts with rank 0 alone and discovers its siblings as it
        // pops, matching the precomputed builder's original heap layout.
        if (eager_siblings) {
            push_children(0, 1, 0.0f, {}, lp, ids, 0);
        } else if (precomputed) {
            push_scored_candidate(0, 1, precomputed_log_probs[0], {}, 0,
                                  precomputed_token_ids[0]);
        } else {
            push_candidate(0, 1, 0.0f, {}, 0, lp, ids);
        }
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        Candidate c = heap.top();
        heap.pop();
        // Best-first pops in descending q(v): the first out-of-window
        // candidate proves every remaining one is out of the window too.
        if (q_star - c.logw > tau_tree) break;

        const int node = tree.n_nodes + 1;
        tree.token_ids.push_back(c.token);
        tree.depths.push_back(c.depth);
        tree.parents.push_back(c.parent);
        tree.child_maps.emplace_back();
        tree.child_maps[(size_t)c.parent][c.token] = node;
        tree.n_nodes++;

        if (eager_siblings) {
            if (c.depth < L &&
                topk(c.path, c.depth + 1, lp, ids) &&
                (int)lp.size() >= K && (int)ids.size() >= K) {
                push_children(node, c.depth + 1, c.logw, c.path, lp, ids, 0);
            }
        } else {
            // Lazy schedule: expose the next sibling and the rank-0 child one
            // at a time (the original precomputed-array expansion order).
            if (precomputed) {
                const int row = (c.depth - 1)*K;
                if (c.rank + 1 < K) {
                    const float parent_logw =
                        c.logw - precomputed_log_probs[row + c.rank];
                    push_scored_candidate(
                        c.parent, c.depth,
                        parent_logw + precomputed_log_probs[row + c.rank + 1],
                        {}, c.rank + 1,
                        precomputed_token_ids[row + c.rank + 1]);
                }
                if (c.depth < L) {
                    const int child_row = c.depth*K;
                    push_scored_candidate(
                        node, c.depth + 1,
                        c.logw + precomputed_log_probs[child_row],
                        {}, 0, precomputed_token_ids[child_row]);
                }
            } else if (c.rank + 1 < K) {
                std::vector<float> parent_lp;
                std::vector<int32_t> parent_ids;
                std::vector<int32_t> parent_prefix = c.path;
                parent_prefix.pop_back();
                if (topk(parent_prefix, c.depth, parent_lp, parent_ids) &&
                    (int)parent_lp.size() >= K &&
                    (int)parent_ids.size() >= K) {
                    push_candidate(c.parent, c.depth,
                                   c.logw - parent_lp[(size_t)c.rank],
                                   parent_prefix, c.rank + 1,
                                   parent_lp, parent_ids);
                }
            }
            if (!precomputed && c.depth < L &&
                topk(c.path, c.depth + 1, lp, ids) &&
                (int)lp.size() >= K && (int)ids.size() >= K) {
                push_candidate(node, c.depth + 1, c.logw, c.path, 0, lp, ids);
            }
        }
    }

    build_visibility(tree);
    return tree;
}

}  // namespace

DDTree build_ddtree(const float * top_log_probs,
                    const int32_t * top_token_ids,
                    int L, int K, int budget,
                    bool chain_seed,
                    float tau_tree) {
    return build_ddtree_impl(DDTreeConditionalTopK{}, L, K, budget,
                             chain_seed, /*eager_siblings=*/false, tau_tree,
                             top_log_probs, top_token_ids);
}

DDTree build_ddtree_conditional(const DDTreeConditionalTopK & next_topk,
                                int L, int K, int budget,
                                bool chain_seed,
                                float tau_tree) {
    return build_ddtree_impl(next_topk, L, K, budget, chain_seed,
                             /*eager_siblings=*/true, tau_tree,
                             /*precomputed_log_probs=*/nullptr,
                             /*precomputed_token_ids=*/nullptr);
}

std::vector<int> follow_verified_tree(const DDTree & tree,
                                      const int32_t * posterior,
                                      int & out_next_token,
                                      int * out_node_idx) {
    std::vector<int> accepted;
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);

    int current_index = 0;
    int next_token    = posterior[current_index];
    while (true) {
        const auto & children = tree.child_maps[current_index];
        auto it = children.find(next_token);
        if (it == children.end()) break;
        current_index = it->second;
        accepted.push_back(current_index);
        next_token = posterior[current_index];
    }
    out_next_token = next_token;
    if (out_node_idx) *out_node_idx = current_index;
    return accepted;
}

}  // namespace dflash::common
