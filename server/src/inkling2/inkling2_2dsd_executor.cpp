// 2DSD (Two-Dimensional Speculative Decoding) Greedy Tree Executor with Draft Confidence Pruning.

#include "inkling2_2dsd_executor.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace luce::common {

Inkling2TwoDSDConfig inkling2_2dsd_config_from_env(int R, int block_size) {
    Inkling2TwoDSDConfig cfg;
    cfg.R = R;
    cfg.k = std::clamp(block_size, 1, 8);
    cfg.d1 = 1;
    cfg.d2 = std::min(2, std::max(1, R - 1));
    cfg.stage = (R >= 3) ? Inkling2TwoDSDStage::Stage2b : Inkling2TwoDSDStage::Stage1;
    cfg.draft_confidence = 0.0f;

    if (const char * es = std::getenv("LUCE_INKLING2_2DSD_STAGE")) {
        if (std::strcmp(es, "stage1") == 0) cfg.stage = Inkling2TwoDSDStage::Stage1;
        else if (std::strcmp(es, "stage2a") == 0) cfg.stage = Inkling2TwoDSDStage::Stage2a;
        else if (std::strcmp(es, "stage2b") == 0) cfg.stage = Inkling2TwoDSDStage::Stage2b;
    }
    if (const char * ed1 = std::getenv("LUCE_INKLING2_2DSD_D1")) {
        const int v = std::atoi(ed1);
        if (v >= 1 && v <= R) cfg.d1 = v;
    }
    if (const char * ed2 = std::getenv("LUCE_INKLING2_2DSD_D2")) {
        const int v = std::atoi(ed2);
        if (v > cfg.d1 && v < R) cfg.d2 = v;
    }
    if (const char * ek = std::getenv("LUCE_INKLING2_2DSD_K")) {
        const int v = std::atoi(ek);
        if (v >= 1 && v <= block_size) cfg.k = v;
    }
    if (const char * ec = std::getenv("LUCE_INKLING2_2DSD_CONF")) {
        const float v = std::strtof(ec, nullptr);
        if (v >= 0.0f && v <= 1.0f) cfg.draft_confidence = v;
    }
    if (cfg.stage != Inkling2TwoDSDStage::Stage1 && !(cfg.d1 < cfg.d2 && cfg.d2 < cfg.R)) {
        cfg.stage = Inkling2TwoDSDStage::Stage1;
    }
    return cfg;
}

namespace {

enum class NodeRole { Main = 0, Frozen = 1, Fork = 2 };

struct TreeNode {
    int      id             = -1;
    int      parent_id      = -1; // -1 => parent is last committed prefix token
    int      slot           = -1; // tree scratch slot in [0..max_nodes-1]
    int32_t  token          = 0;
    int      position       = 0;
    int      depth          = 0;  // completed loops [0..R]
    NodeRole role           = NodeRole::Main;
    bool     growing        = true;
    bool     ever_frozen    = false;
    bool     promoted       = false;
    bool     from_draft     = false;
    int      proposal_depth = 0;

    int32_t  readouts[5]    = { -1, -1, -1, -1, -1 }; // indexed by depth 1..R
    bool     has_readout[5] = {};
    std::unordered_map<int32_t, int> children; // token -> child node id
    std::vector<int> frozen_suffix;            // node ids starting at this frozen head
};

struct DraftEvent {
    int head_id   = -1;
    int parent_pos = 0;
    int depth     = 0;
};

struct TrieState {
    Inkling2TwoDSDRunner & runner;
    const Inkling2TwoDSDConfig & cfg;
    Inkling2TwoDSDStats & stats;
    int output_limit = 0;

    int next_id = 0;
    int cursor_id = -1;
    std::vector<int> free_slots;
    std::unordered_map<int, TreeNode> nodes;
    std::unordered_map<int, std::vector<int>> blocks; // head_id -> [head_id, s1, ...]
    std::vector<DraftEvent> events;

    TrieState(Inkling2TwoDSDRunner & r, const Inkling2TwoDSDConfig & c, Inkling2TwoDSDStats & st, int out_lim)
        : runner(r), cfg(c), stats(st), output_limit(out_lim) {
        const int cap = runner.max_nodes();
        free_slots.reserve(cap);
        for (int s = cap - 1; s >= 0; --s) free_slots.push_back(s);
    }

    int alloc_slot() {
        if (free_slots.empty()) return -1;
        const int s = free_slots.back();
        free_slots.pop_back();
        return s;
    }

    void release_slot(int s) {
        if (s >= 0) free_slots.push_back(s);
    }

    int add_child(int parent_id, int32_t token, NodeRole role, int proposal_depth) {
        if (parent_id >= 0) {
            auto & p = nodes.at(parent_id);
            auto it = p.children.find(token);
            if (it != p.children.end()) return it->second;
        }
        const int slot = alloc_slot();
        if (slot < 0) return -1;
        const int pos = (parent_id < 0) ? runner.prefix_length() : (nodes.at(parent_id).position + 1);
        TreeNode nd;
        nd.id             = next_id++;
        nd.parent_id      = parent_id;
        nd.slot           = slot;
        nd.token          = token;
        nd.position       = pos;
        nd.depth          = 0;
        nd.role           = role;
        nd.growing        = true;
        nd.proposal_depth = proposal_depth;
        const int nid = nd.id;
        nodes.emplace(nid, std::move(nd));
        if (parent_id >= 0) {
            nodes.at(parent_id).children[token] = nid;
        }
        stats.max_live_nodes = std::max(stats.max_live_nodes, (int) nodes.size());
        return nid;
    }

    void collect_descendants(int nid, std::vector<int> & out) const {
        std::vector<int> stack = { nid };
        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            out.push_back(cur);
            auto it = nodes.find(cur);
            if (it != nodes.end()) {
                for (const auto & kv : it->second.children) {
                    stack.push_back(kv.second);
                }
            }
        }
    }

    void drop_subtrees(const std::vector<int> & drop_ids) {
        for (int nid : drop_ids) {
            auto it = nodes.find(nid);
            if (it == nodes.end()) continue;
            const int pid = it->second.parent_id;
            if (pid >= 0) {
                auto pit = nodes.find(pid);
                if (pit != nodes.end()) {
                    pit->second.children.erase(it->second.token);
                }
            }
        }
        for (int nid : drop_ids) {
            auto it = nodes.find(nid);
            if (it == nodes.end()) continue;
            release_slot(it->second.slot);
            blocks.erase(nid);
            nodes.erase(it);
        }
    }

    std::vector<int> ancestor_slots(int nid) const {
        std::vector<int> slots;
        int p = nodes.at(nid).parent_id;
        while (p >= 0) {
            const auto & pn = nodes.at(p);
            slots.push_back(pn.slot);
            p = pn.parent_id;
        }
        std::reverse(slots.begin(), slots.end());
        return slots;
    }

    void queue_anchor(int head_id, int parent_pos, int depth) {
        if (nodes.at(head_id).position >= output_limit) return;
        events.push_back({ head_id, parent_pos, depth });
    }

    int new_anchor(int parent_id, int depth, NodeRole role = NodeRole::Main) {
        const auto & p = nodes.at(parent_id);
        const int32_t tok = p.readouts[depth];
        if (p.children.count(tok)) return -1;
        const int ppos = p.position;
        const int head_id = add_child(parent_id, tok, role, depth);
        if (head_id < 0) return -1;
        queue_anchor(head_id, ppos, depth);
        return head_id;
    }

    void freeze_suffix(const std::vector<int> & suffix, size_t from_idx) {
        const int head_id = suffix[from_idx];
        auto & h = nodes.at(head_id);
        h.role = NodeRole::Frozen;
        h.frozen_suffix.assign(suffix.begin() + (ptrdiff_t) from_idx, suffix.end());
        for (size_t i = from_idx; i < suffix.size(); ++i) {
            auto & n = nodes.at(suffix[i]);
            n.growing = false;
            n.ever_frozen = true;
        }
    }

    void scan_suffix(int head_id, const std::vector<int> & suffix, int depth) {
        nodes.at(head_id).growing = true;
        int parent_id = head_id;
        for (size_t idx = 0; idx < suffix.size(); ++idx) {
            const int cid = suffix[idx];
            const auto & parent = nodes.at(parent_id);
            const auto & child  = nodes.at(cid);
            if (parent.readouts[depth] != child.token) {
                if (cfg.frozen_suffix) {
                    freeze_suffix(suffix, idx);
                } else {
                    std::vector<int> rem;
                    collect_descendants(cid, rem);
                    drop_subtrees(rem);
                }
                new_anchor(parent_id, depth, NodeRole::Main);
                return;
            }
            auto & cmut = nodes.at(cid);
            cmut.growing = true;
            cmut.proposal_depth = depth;
            parent_id = cid;
        }
        if (nodes.at(parent_id).position + 1 < output_limit) {
            new_anchor(parent_id, depth, NodeRole::Main);
        }
    }

    void run_stage1() {
        if (cfg.d1 == cfg.R) return;
        std::vector<int> keys;
        keys.reserve(blocks.size());
        for (const auto & kv : blocks) keys.push_back(kv.first);

        for (int head_id : keys) {
            auto bit = blocks.find(head_id);
            if (bit == blocks.end()) continue;
            auto nit = nodes.find(head_id);
            if (nit == nodes.end()) {
                blocks.erase(bit);
                continue;
            }
            if (nit->second.depth < cfg.d1) continue;
            std::vector<int> ids = std::move(bit->second);
            blocks.erase(bit);
            if (!nit->second.growing) continue;
            std::vector<int> suffix(ids.begin() + 1, ids.end());
            scan_suffix(head_id, suffix, cfg.d1);
        }
    }

    void run_stage2(const std::vector<int> & advanced_ids) {
        if (cfg.stage == Inkling2TwoDSDStage::Stage1) return;
        std::unordered_set<int> handled;
        for (int pid : advanced_ids) {
            auto pit = nodes.find(pid);
            if (pit == nodes.end()) continue;
            const auto & parent = pit->second;
            if (parent.depth != cfg.d2 || !parent.growing || handled.count(pid)) continue;

            int main_id = -1;
            int frozen_id = -1;
            for (const auto & kv : parent.children) {
                const auto & ch = nodes.at(kv.second);
                if (ch.role == NodeRole::Main) main_id = kv.second;
                else if (ch.role == NodeRole::Frozen) frozen_id = kv.second;
            }
            if (main_id < 0) continue;

            const int32_t pred = parent.readouts[cfg.d2];
            if (pred == nodes.at(main_id).token) continue;

            if (frozen_id >= 0 && pred == nodes.at(frozen_id).token) {
                auto & fnode = nodes.at(frozen_id);
                std::vector<int> ids = fnode.frozen_suffix;
                bool all_at_d2 = !ids.empty();
                for (int id : ids) {
                    auto it = nodes.find(id);
                    if (it == nodes.end() || it->second.depth != cfg.d2) { all_at_d2 = false; break; }
                }
                if (all_at_d2) {
                    for (int id : ids) handled.insert(id);
                    fnode.promoted = true;
                    fnode.frozen_suffix.clear();
                    ++stats.promotions;
                    std::vector<int> suffix(ids.begin() + 1, ids.end());
                    scan_suffix(frozen_id, suffix, cfg.d2);
                }
            } else if (cfg.stage == Inkling2TwoDSDStage::Stage2b &&
                       parent.position + 1 < output_limit &&
                       !parent.children.count(pred) &&
                       (int) free_slots.size() > cfg.k) {
                if (new_anchor(pid, cfg.d2, NodeRole::Fork) >= 0) {
                    ++stats.forks;
                }
            }
        }
    }

    bool run_verify(const std::function<bool(int32_t &)> & on_commit, bool & stopped, std::string & error) {
        std::vector<int> accepted_ids;
        std::vector<int> removed_ids;
        stopped = false;

        while (nodes.at(cursor_id).depth == cfg.R) {
            const int cur_id = cursor_id;
            const auto & cur = nodes.at(cur_id);
            int32_t emit_tok = cur.readouts[cfg.R];
            const bool more = on_commit(emit_tok);

            int chosen_id = -1;
            auto cit = cur.children.find(emit_tok);
            bool is_bonus = false;
            if (cit != cur.children.end()) {
                chosen_id = cit->second;
                const auto & ch = nodes.at(chosen_id);
                if (ch.from_draft) ++stats.commit_draft;
                if (ch.role == NodeRole::Fork) ++stats.commit_fork;
                else if (ch.role == NodeRole::Frozen || (ch.ever_frozen && ch.proposal_depth == 0)) ++stats.commit_frozen;
                else ++stats.commit_main;
            } else {
                is_bonus = true;
                ++stats.commit_bonus;
                if (more) {
                    // Prune all rejected children first so free_slots is guaranteed non-empty for bonus node!
                    for (const auto & kv : cur.children) {
                        collect_descendants(kv.second, removed_ids);
                    }
                    drop_subtrees(removed_ids);
                    removed_ids.clear();
                    chosen_id = add_child(cur_id, emit_tok, NodeRole::Main, cfg.R);
                    if (chosen_id < 0) {
                        error = "inkling2_2dsd: tree slot pool exhausted on bonus token";
                        return false;
                    }
                }
            }

            if (!is_bonus) {
                for (const auto & kv : cur.children) {
                    if (kv.second != chosen_id) {
                        collect_descendants(kv.second, removed_ids);
                    }
                }
            }

            accepted_ids.push_back(cur_id);
            const int cur_pos = cur.position;
            cursor_id = chosen_id;

            if (!more) {
                stopped = true;
                break;
            }
            if (is_bonus) {
                queue_anchor(chosen_id, cur_pos, cfg.R);
            }
        }

        if (!accepted_ids.empty()) {
            if (!removed_ids.empty()) drop_subtrees(removed_ids);

            std::vector<int> commit_slots;
            commit_slots.reserve(accepted_ids.size());
            for (int nid : accepted_ids) {
                commit_slots.push_back(nodes.at(nid).slot);
            }
            if (!runner.commit_nodes(commit_slots.data(), (int) commit_slots.size())) {
                error = "inkling2_2dsd: commit_nodes failed";
                return false;
            }
            if (cursor_id >= 0 && nodes.count(cursor_id)) {
                nodes.at(cursor_id).parent_id = -1;
            }
            for (int nid : accepted_ids) {
                auto it = nodes.find(nid);
                if (it != nodes.end()) {
                    release_slot(it->second.slot);
                    blocks.erase(nid);
                    nodes.erase(it);
                }
            }
        }
        return true;
    }

    bool run_draft_events(std::string & error) {
        std::vector<DraftEvent> evs;
        evs.swap(events);
        if (evs.empty()) return true;

        std::unordered_set<int> seen;
        std::vector<int> plan_heads;
        std::vector<int> plan_widths;
        std::vector<Inkling2TwoDSDDraftReq> reqs;

        for (const auto & ev : evs) {
            auto it = nodes.find(ev.head_id);
            if (it == nodes.end() || !it->second.growing) continue;
            if (seen.count(ev.head_id) || !it->second.children.empty()) continue;
            seen.insert(ev.head_id);

            const auto & head = it->second;
            const int max_by_limit = output_limit - head.position;
            const int max_by_slots = (int) free_slots.size() + 1;
            const int width = std::min({ cfg.k, max_by_limit, max_by_slots });
            blocks[head.id] = { head.id };
            if (width <= 1) continue;

            // Build in-flight ancestor span [cursor .. parent] for loop-context drafting
            std::vector<int> span;
            if (ev.depth < cfg.R) {
                span = ancestor_slots(head.id);
                if ((int) span.size() > 16) continue; // safety bound for static drafter span bucket
            }

            Inkling2TwoDSDDraftReq dr;
            dr.head_token = head.token;
            dr.anchor_pos = head.position;
            dr.depth      = ev.depth;
            dr.span_slots = std::move(span);
            reqs.push_back(std::move(dr));
            plan_heads.push_back(head.id);
            plan_widths.push_back(width);
        }

        if (reqs.empty()) return true;

        const auto td0 = std::chrono::steady_clock::now();
        std::vector<Inkling2TwoDSDDraftOut> outs;
        if (!runner.draft_batch(reqs, outs) || outs.size() != reqs.size()) {
            error = "inkling2_2dsd: draft_batch failed";
            return false;
        }
        const auto td1 = std::chrono::steady_clock::now();
        stats.draft_s += std::chrono::duration<double>(td1 - td0).count();
        ++stats.drafter_batches;
        stats.drafter_calls += (long long) reqs.size();

        for (size_t i = 0; i < reqs.size(); ++i) {
            const int head_id = plan_heads[i];
            const int n_want  = std::min(plan_widths[i] - 1, (int) outs[i].tokens.size());
            int n_keep = n_want;
            // Static draft_confidence pruning: truncate before first slot below threshold!
            if (cfg.draft_confidence > 0.0f && (int) outs[i].confidence.size() >= n_want) {
                for (int s = 0; s < n_want; ++s) {
                    if (outs[i].confidence[s] < cfg.draft_confidence) {
                        n_keep = s;
                        break;
                    }
                }
            }
            stats.draft_slots     += n_keep;
            stats.draft_slots_cut += (n_want - n_keep);

            int parent_id = head_id;
            for (int s = 0; s < n_keep; ++s) {
                const int cid = add_child(parent_id, outs[i].tokens[s], NodeRole::Main, 0);
                if (cid < 0) break;
                nodes.at(cid).from_draft = true;
                blocks[head_id].push_back(cid);
                parent_id = cid;
            }
        }
        return true;
    }
};

} // namespace

bool inkling2_2dsd_run_greedy(
        Inkling2TwoDSDRunner & runner,
        const Inkling2TwoDSDConfig & cfg,
        int32_t first_token,
        int max_new,
        const std::function<bool(int32_t &)> & on_commit,
        Inkling2TwoDSDStats & stats,
        std::string & error) {
    stats = {};
    if (max_new <= 0) return true;

    int32_t first = first_token;
    const bool more0 = on_commit(first);
    if (!more0) return true;

    const int pref0 = runner.prefix_length();
    const int output_limit = std::min(pref0 + max_new, runner.max_ctx());
    if (pref0 >= output_limit) return true;

    uint32_t readout_mask = (1u << cfg.d1) | (1u << cfg.R);
    if (cfg.stage != Inkling2TwoDSDStage::Stage1) {
        readout_mask |= (1u << cfg.d2);
    }

    TrieState trie(runner, cfg, stats, output_limit);
    trie.cursor_id = trie.add_child(-1, first, NodeRole::Main, cfg.R);
    if (trie.cursor_id < 0) {
        error = "inkling2_2dsd: failed to allocate initial cursor node";
        return false;
    }
    trie.queue_anchor(trie.cursor_id, pref0 - 1, cfg.R);
    if (!trie.run_draft_events(error)) return false;

    bool stopped = false;
    while (!stopped) {
        // Gather all active nodes (depth < R) sorted by (position, node_id)
        std::vector<int> active_ids;
        active_ids.reserve(trie.nodes.size());
        for (const auto & kv : trie.nodes) {
            if (kv.second.depth < cfg.R) active_ids.push_back(kv.first);
        }
        if (active_ids.empty()) {
            error = "inkling2_2dsd: finalized trie has no active nodes";
            return false;
        }
        std::sort(active_ids.begin(), active_ids.end(), [&](int a, int b) {
            const auto & na = trie.nodes.at(a);
            const auto & nb = trie.nodes.at(b);
            return na.position < nb.position || (na.position == nb.position && na.id < nb.id);
        });

        std::vector<Inkling2TwoDSDNodeDesc> descs;
        descs.reserve(active_ids.size());
        for (int nid : active_ids) {
            const auto & nd = trie.nodes.at(nid);
            Inkling2TwoDSDNodeDesc d;
            d.token     = nd.token;
            d.slot      = nd.slot;
            d.pos       = nd.position;
            d.depth     = nd.depth;
            d.ancestors = trie.ancestor_slots(nid);
            descs.push_back(std::move(d));
        }

        std::vector<int> readout_rows;
        std::vector<int32_t> readout_argmax;
        if (!runner.advance_nodes(descs, readout_mask, readout_rows, readout_argmax)) {
            error = "inkling2_2dsd: advance_nodes failed";
            return false;
        }
        ++stats.target_calls;
        stats.target_cells += (long long) active_ids.size();

        for (int nid : active_ids) {
            trie.nodes.at(nid).depth += 1;
        }
        for (size_t i = 0; i < readout_rows.size(); ++i) {
            const int nid = active_ids[readout_rows[i]];
            auto & nd = trie.nodes.at(nid);
            nd.readouts[nd.depth]    = readout_argmax[i];
            nd.has_readout[nd.depth] = true;
        }

        if (!trie.run_verify(on_commit, stopped, error)) return false;
        if (!stopped) {
            trie.run_stage2(active_ids);
            trie.run_stage1();
            if (!trie.run_draft_events(error)) return false;
        }
    }
    return true;
}

} // namespace luce::common
