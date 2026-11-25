// 2DSD (Two-Dimensional Speculative Decoding) Greedy Tree Executor with Draft Confidence Pruning.

#include "inkling2_2dsd_executor.h"

#include <algorithm>
#include <array>
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
    if (const char * eb = std::getenv("LUCE_INKLING2_3DSD_TREE_BRANCH")) {
        cfg.tree_branch = (std::atoi(eb) != 0);
    }
    if (const char * el = std::getenv("LUCE_INKLING2_3DSD_LOOKUP")) {
        cfg.prompt_lookup = (std::atoi(el) != 0);
    }
    if (const char * elk = std::getenv("LUCE_INKLING2_3DSD_LOOKUP_K")) {
        const int v = std::atoi(elk);
        if (v >= 2 && v <= 32) cfg.lookup_max_k = v;
    }
    if (const char * bi = std::getenv("LUCE_BATCH_INVARIANT")) {
        if (std::atoi(bi) != 0) cfg.lookup_max_k = std::min(cfg.lookup_max_k, 7);
    }
    if (const char * ee = std::getenv("LUCE_INKLING2_2DSD_EARLY_EXIT_P")) {
        const float v = std::strtof(ee, nullptr);
        if (v >= 0.0f && v <= 1.0f) cfg.early_exit_p = v;
    }
    if (cfg.stage != Inkling2TwoDSDStage::Stage1 && !(cfg.d1 < cfg.d2 && cfg.d2 < cfg.R)) {
        cfg.stage = Inkling2TwoDSDStage::Stage1;
    }
    return cfg;
}

Inkling2TwoDSDConfig inkling2_2dsd_config_for_concurrency(int R, int block_size, int active_slots) {
    Inkling2TwoDSDConfig cfg = inkling2_2dsd_config_from_env(R, block_size);
    const bool explicit_conf  = (std::getenv("LUCE_INKLING2_2DSD_CONF") != nullptr);
    const bool explicit_k     = (std::getenv("LUCE_INKLING2_2DSD_K") != nullptr);
    const bool explicit_stage = (std::getenv("LUCE_INKLING2_2DSD_STAGE") != nullptr);
    if (active_slots <= 1) {
        return cfg;
    }
    if (active_slots >= 8) {
        if (!explicit_stage) cfg.stage = Inkling2TwoDSDStage::Stage1;
        if (!explicit_k)     cfg.k = 1; // depth-only pipelining: ~3 cells/slot -> 24+ cells across 8 slots, 0 draft cost!
        if (!explicit_conf)  cfg.draft_confidence = 0.0f;
    } else if (active_slots >= 4) {
        if (!explicit_k)    cfg.k = std::min(cfg.k, 6);
        if (!explicit_conf) cfg.draft_confidence = std::max(cfg.draft_confidence, 0.75f);
    } else { // 2..3 active slots
        if (!explicit_conf) cfg.draft_confidence = std::max(cfg.draft_confidence, 0.65f);
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

    int32_t  readouts[5]     = { -1, -1, -1, -1, -1 }; // indexed by depth 1..R
    bool     has_readout[5]  = {};
    float    readout_pmax[5] = {};
    bool     early_exited_d2 = false;
    bool     has_sample_r    = false;
    Inkling2TwoDSDDepthRSample sample_r;
    std::unordered_map<int32_t, int> children; // token -> child node id
    std::vector<int> frozen_suffix;            // node ids starting at this frozen head
};

// Lightweight O(1) trigram prompt-lookup matcher supporting in-flight ancestor suffixes.
struct TwoDSDSuffixLookup {
    static constexpr int kWays = 4;
    static constexpr int kMinMatch = 4;
    static constexpr int kMaxMatch = 32;

    std::vector<int32_t> hist;
    std::unordered_map<uint64_t, std::array<int32_t, kWays>> index;

    static uint64_t key3(int32_t a, int32_t b, int32_t c) {
        uint64_t h = 0xcbf29ce484222325ull;
        h = (h ^ (uint32_t) a) * 0x100000001b3ull;
        h = (h ^ (uint32_t) b) * 0x100000001b3ull;
        h = (h ^ (uint32_t) c) * 0x100000001b3ull;
        return h;
    }

    void reset(const std::vector<int32_t> & tokens) {
        hist.clear();
        index.clear();
        hist.reserve(tokens.size() + 256);
        append(tokens.data(), tokens.size());
    }

    void append(const int32_t * tokens, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            hist.push_back(tokens[i]);
            const size_t end = hist.size() - 1;
            if (end < 2) continue;
            auto it = index.find(key3(hist[end - 2], hist[end - 1], hist[end]));
            if (it == index.end()) {
                std::array<int32_t, kWays> ways;
                ways.fill(-1);
                ways[0] = (int32_t) end;
                index.emplace(key3(hist[end - 2], hist[end - 1], hist[end]), ways);
            } else {
                auto & ways = it->second;
                for (int j = kWays - 1; j > 0; --j) ways[j] = ways[j - 1];
                ways[0] = (int32_t) end;
            }
        }
    }

    // Match `suffix` (representing committed_tail + in-flight ancestors + head.token) against `hist`.
    // `live_len` is the total logical sequence length (hist.size() + in_flight_len); excludes matching self.
    int propose(const std::vector<int32_t> & suffix, int max_k, std::vector<int32_t> & out) const {
        out.clear();
        const int slen = (int) suffix.size();
        const int hlen = (int) hist.size();
        if (max_k <= 0 || slen < kMinMatch || hlen < kMinMatch + 2) return 0;
        const auto it = index.find(key3(suffix[(size_t) slen - 3], suffix[(size_t) slen - 2], suffix[(size_t) slen - 1]));
        if (it == index.end()) return 0;
        int best_len = 0;
        int best_end = -1;
        for (int e : it->second) {
            if (e < kMinMatch - 1 || e + 1 >= hlen) continue;
            // Avoid matching the exact same tail when suffix came solely from hist
            if (e == hlen - 1) continue;
            int len = 0;
            const int lim = std::min({ kMaxMatch, e + 1, slen });
            while (len < lim && hist[(size_t) (e - len)] == suffix[(size_t) (slen - 1 - len)]) ++len;
            if (len > best_len) {
                best_len = len;
                best_end = e;
            }
        }
        if (best_len < kMinMatch) return 0;
        out.reserve((size_t) max_k);
        for (int p = best_end + 1; p < hlen && (int) out.size() < max_k; ++p) {
            out.push_back(hist[(size_t) p]);
        }
        return (int) out.size();
    }
};

struct DraftEvent {
    int head_id   = -1;
    int parent_pos = 0;
    int depth     = 0;
};

struct TrieState {
    Inkling2TwoDSDRunner & runner;
    Inkling2TwoDSDConfig cfg;
    Inkling2TwoDSDStats & stats;
    int output_limit = 0;
    const Inkling2TwoDSDSamplerHooks * sampler = nullptr;

    int next_id = 0;
    int cursor_id = -1;
    std::vector<int> free_slots;
    std::unordered_map<int, TreeNode> nodes;
    std::unordered_map<int, std::vector<int>> blocks; // head_id -> [head_id, s1, ...]
    std::vector<DraftEvent> events;
    TwoDSDSuffixLookup lookup;
    std::vector<float> last_pmax;

    TrieState(Inkling2TwoDSDRunner & r, const Inkling2TwoDSDConfig & c, Inkling2TwoDSDStats & st,
              int out_lim, const Inkling2TwoDSDSamplerHooks * samp = nullptr)
        : runner(r), cfg(c), stats(st), output_limit(out_lim), sampler(samp) {
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
                const int32_t pred_tok = parent.readouts[depth];
                const int ppos = parent.position;
                if (cfg.frozen_suffix) {
                    freeze_suffix(suffix, idx);
                } else {
                    std::vector<int> rem;
                    collect_descendants(cid, rem);
                    drop_subtrees(rem);
                }
                auto pit = nodes.at(parent_id).children.find(pred_tok);
                if (pit != nodes.at(parent_id).children.end()) {
                    // Pre-branched top-2 sibling HIT at d1/d2! Promote to Main (already +1 depth ahead!).
                    // If it already has a readout at `depth`, immediately anchor its predicted child so
                    // the entire drafted block stays 100% depth-aligned (0 freeze/promote desync!).
                    const int alt_id = pit->second;
                    auto & alt = nodes.at(alt_id);
                    alt.role           = NodeRole::Main;
                    alt.growing        = true;
                    alt.proposal_depth = depth;
                    if (alt.children.empty() && !blocks.count(alt_id) && alt.position + 1 < output_limit) {
                        if (alt.depth >= depth && alt.has_readout[depth]) {
                            new_anchor(alt_id, depth, NodeRole::Main);
                        } else {
                            queue_anchor(alt_id, ppos, depth);
                        }
                    }
                    ++stats.promotions;
                } else {
                    new_anchor(parent_id, depth, NodeRole::Main);
                }
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
                       (int) free_slots.size() > cfg.k) {
                auto cit = parent.children.find(pred);
                if (cit == parent.children.end()) {
                    if (new_anchor(pid, cfg.d2, NodeRole::Fork) >= 0) {
                        ++stats.forks;
                    }
                } else {
                    const int cid = cit->second;
                    auto & ch = nodes.at(cid);
                    if (ch.role == NodeRole::Fork && ch.children.empty() && !blocks.count(cid)) {
                        ch.growing = true;
                        if (ch.depth >= cfg.d2 && ch.has_readout[cfg.d2] && ch.position + 1 < output_limit) {
                            new_anchor(cid, cfg.d2, NodeRole::Main);
                        } else if (ch.depth >= cfg.d1 && ch.has_readout[cfg.d1] && ch.position + 1 < output_limit) {
                            new_anchor(cid, cfg.d1, NodeRole::Main);
                        } else {
                            queue_anchor(cid, parent.position, cfg.d2);
                        }
                        ++stats.promotions;
                    }
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
            int32_t emit_tok = (sampler && sampler->sample && cur.has_sample_r)
                ? sampler->sample(cur.sample_r, cur.readouts[cfg.R])
                : cur.readouts[cfg.R];
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
            } else if (nodes.at(chosen_id).children.empty() && !blocks.count(chosen_id) &&
                       nodes.at(chosen_id).position + 1 < output_limit) {
                queue_anchor(chosen_id, cur_pos, cfg.R);
            }
        }

        if (!accepted_ids.empty()) {
            if (!removed_ids.empty()) drop_subtrees(removed_ids);

            std::vector<int> commit_slots;
            commit_slots.reserve(accepted_ids.size());
            for (int nid : accepted_ids) {
                const auto & nd = nodes.at(nid);
                commit_slots.push_back(nd.early_exited_d2 ? -(nd.slot + 1) : nd.slot);
                if (cfg.prompt_lookup) {
                    lookup.append(&nd.token, 1);
                }
            }
            if (!runner.commit_nodes(commit_slots.data(), (int) commit_slots.size())) {
                error = "inkling2_2dsd: commit_nodes failed";
                return false;
            }
            if (cursor_id >= 0 && nodes.count(cursor_id)) {
                nodes.at(cursor_id).parent_id = -1;
            }
            for (int nid : accepted_ids) {
                auto bit = blocks.find(nid);
                if (bit != blocks.end()) {
                    // If a pre-branched head reached R and committed before its newly drafted
                    // suffix was scanned at d1, re-root the surviving suffix at cursor_id.
                    if (cursor_id >= 0 && bit->second.size() >= 2 && bit->second[1] == cursor_id &&
                        nodes.count(cursor_id) && !blocks.count(cursor_id)) {
                        blocks[cursor_id].assign(bit->second.begin() + 1, bit->second.end());
                    }
                    blocks.erase(bit);
                }
                auto it = nodes.find(nid);
                if (it != nodes.end()) {
                    release_slot(it->second.slot);
                    nodes.erase(it);
                }
            }
        }
        return true;
    }

    void collect_draft_reqs(
            std::vector<Inkling2TwoDSDDraftReq> & reqs,
            std::vector<int> & plan_heads,
            std::vector<int> & plan_widths) {
        reqs.clear();
        plan_heads.clear();
        plan_widths.clear();
        std::vector<DraftEvent> evs;
        evs.swap(events);
        if (evs.empty()) return;

        std::unordered_set<int> seen;
        for (const auto & ev : evs) {
            auto it = nodes.find(ev.head_id);
            if (it == nodes.end() || !it->second.growing) continue;
            if (seen.count(ev.head_id) || !it->second.children.empty()) continue;
            seen.insert(ev.head_id);

            const auto & head = it->second;
            const int max_by_limit = output_limit - head.position;
            const int max_by_slots = (int) free_slots.size() + 1;

            // 3DSD Lever 3: 0.01 ms CPU Prompt-Lookup Trie Injection (skips GPU drafter on match!)
            if (cfg.prompt_lookup && lookup.hist.size() >= 8 && max_by_limit > 2 && max_by_slots > 2) {
                std::vector<int32_t> suffix;
                suffix.reserve(40);
                // Reconstruct in-flight token chain [cursor .. head]
                std::vector<int32_t> inflight;
                for (int cur_id = head.id; cur_id >= 0;) {
                    const auto & nd = nodes.at(cur_id);
                    inflight.push_back(nd.token);
                    cur_id = nd.parent_id;
                }
                std::reverse(inflight.begin(), inflight.end());
                const int need_hist = std::max(0, 32 - (int) inflight.size());
                const int h0 = std::max(0, (int) lookup.hist.size() - need_hist);
                for (int hi = h0; hi < (int) lookup.hist.size(); ++hi) {
                    suffix.push_back(lookup.hist[(size_t) hi]);
                }
                for (int32_t tk : inflight) suffix.push_back(tk);

                const int look_cap = std::min({ cfg.lookup_max_k, max_by_limit - 1, max_by_slots - 1 });
                std::vector<int32_t> proposed;
                if (look_cap >= 2 && lookup.propose(suffix, look_cap, proposed) >= 2) {
                    blocks[head.id] = { head.id };
                    int parent_id = head.id;
                    for (int32_t tk : proposed) {
                        const int cid = add_child(parent_id, tk, NodeRole::Main, 0);
                        if (cid < 0) break;
                        nodes.at(cid).from_draft = true;
                        blocks[head.id].push_back(cid);
                        parent_id = cid;
                    }
                    ++stats.lookup_hits;
                    stats.lookup_slots += (long long) proposed.size();
                    stats.draft_slots  += (long long) proposed.size();
                    continue;
                }
            }

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
    }

    void apply_draft_outs(
            const std::vector<int> & plan_heads,
            const std::vector<int> & plan_widths,
            const std::vector<Inkling2TwoDSDDraftOut> & outs) {
        for (size_t i = 0; i < plan_heads.size() && i < outs.size(); ++i) {
            const int head_id = plan_heads[i];
            if (!nodes.count(head_id)) continue;
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
            bool branched = false;
            const bool can_branch = cfg.tree_branch &&
                                    outs[i].alt_tokens.size() >= (size_t) n_keep &&
                                    outs[i].confidence.size() >= (size_t) n_keep;
            for (int s = 0; s < n_keep; ++s) {
                const int cid = add_child(parent_id, outs[i].tokens[s], NodeRole::Main, 0);
                if (cid < 0) break;
                nodes.at(cid).from_draft = true;
                blocks[head_id].push_back(cid);
                // 3DSD Lever 2: Confidence-guided top-2 tree branch at the first uncertain slot!
                if (can_branch && !branched && free_slots.size() > 8 &&
                    outs[i].confidence[s] >= cfg.branch_conf_low &&
                    outs[i].confidence[s] <= cfg.branch_conf_high &&
                    outs[i].alt_tokens[s] != outs[i].tokens[s]) {
                    const int alt_id = add_child(parent_id, outs[i].alt_tokens[s], NodeRole::Fork, 0);
                    if (alt_id >= 0) {
                        nodes.at(alt_id).from_draft = true;
                        ++stats.draft_branches;
                        branched = true;
                    }
                }
                parent_id = cid;
            }
        }
    }

    bool run_draft_events(std::string & error) {
        std::vector<Inkling2TwoDSDDraftReq> reqs;
        std::vector<int> plan_heads;
        std::vector<int> plan_widths;
        collect_draft_reqs(reqs, plan_heads, plan_widths);
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

        apply_draft_outs(plan_heads, plan_widths, outs);
        return true;
    }
};

} // namespace

struct Inkling2TwoDSDSession::Impl {
    TrieState trie;
    Impl(Inkling2TwoDSDRunner & r, const Inkling2TwoDSDConfig & c, Inkling2TwoDSDStats & st,
         int out_lim, const Inkling2TwoDSDSamplerHooks * samp)
        : trie(r, c, st, out_lim, samp) {}
};

Inkling2TwoDSDSession::Inkling2TwoDSDSession(
        Inkling2TwoDSDRunner & runner,
        const Inkling2TwoDSDConfig & cfg,
        Inkling2TwoDSDStats & stats,
        int output_limit,
        const Inkling2TwoDSDSamplerHooks * sampler)
    : impl_(std::make_unique<Impl>(runner, cfg, stats, output_limit, sampler)) {}

Inkling2TwoDSDSession::~Inkling2TwoDSDSession() = default;

uint32_t Inkling2TwoDSDSession::readout_mask() const {
    uint32_t mask = (1u << impl_->trie.cfg.d1) | (1u << impl_->trie.cfg.R);
    if (impl_->trie.cfg.stage != Inkling2TwoDSDStage::Stage1) {
        mask |= (1u << impl_->trie.cfg.d2);
    }
    return mask;
}

bool Inkling2TwoDSDSession::seed_cursor(int32_t first_token, int prefix_len_before_first, std::string & error) {
    auto & trie = impl_->trie;
    trie.cursor_id = trie.add_child(-1, first_token, NodeRole::Main, trie.cfg.R);
    if (trie.cursor_id < 0) {
        error = "inkling2_2dsd: failed to allocate initial cursor node";
        return false;
    }
    trie.queue_anchor(trie.cursor_id, prefix_len_before_first, trie.cfg.R);
    return true;
}

void Inkling2TwoDSDSession::set_prompt_history(const std::vector<int32_t> & prompt_tokens) {
    impl_->trie.lookup.reset(prompt_tokens);
}

void Inkling2TwoDSDSession::set_last_readout_pmax(const std::vector<float> & pmax) {
    impl_->trie.last_pmax = pmax;
}

void Inkling2TwoDSDSession::collect_draft_reqs(
        std::vector<Inkling2TwoDSDDraftReq> & out_reqs,
        std::vector<int> & out_plan_heads,
        std::vector<int> & out_plan_widths) {
    impl_->trie.collect_draft_reqs(out_reqs, out_plan_heads, out_plan_widths);
}

void Inkling2TwoDSDSession::apply_draft_outs(
        const std::vector<int> & plan_heads,
        const std::vector<int> & plan_widths,
        const std::vector<Inkling2TwoDSDDraftOut> & outs) {
    impl_->trie.apply_draft_outs(plan_heads, plan_widths, outs);
}

bool Inkling2TwoDSDSession::collect_active(
        std::vector<int> & out_active_ids,
        std::vector<Inkling2TwoDSDNodeDesc> & out_descs,
        std::string & error) const {
    const auto & trie = impl_->trie;
    out_active_ids.clear();
    out_descs.clear();
    out_active_ids.reserve(trie.nodes.size());
    for (const auto & kv : trie.nodes) {
        if (kv.second.depth < trie.cfg.R) out_active_ids.push_back(kv.first);
    }
    if (out_active_ids.empty()) {
        error = "inkling2_2dsd: finalized trie has no active nodes";
        return false;
    }
    std::sort(out_active_ids.begin(), out_active_ids.end(), [&](int a, int b) {
        const auto & na = trie.nodes.at(a);
        const auto & nb = trie.nodes.at(b);
        return na.position < nb.position || (na.position == nb.position && na.id < nb.id);
    });
    out_descs.reserve(out_active_ids.size());
    for (int nid : out_active_ids) {
        const auto & nd = trie.nodes.at(nid);
        Inkling2TwoDSDNodeDesc d;
        d.token     = nd.token;
        d.slot      = nd.slot;
        d.pos       = nd.position;
        d.depth     = nd.depth;
        d.ancestors = trie.ancestor_slots(nid);
        out_descs.push_back(std::move(d));
    }
    return true;
}

bool Inkling2TwoDSDSession::apply_step_and_verify(
        const std::vector<int> & active_ids,
        const std::vector<int> & readout_rows,
        const std::vector<int32_t> & readout_argmax,
        const std::function<bool(int32_t &)> & on_commit,
        bool & stopped,
        std::string & error) {
    auto & trie = impl_->trie;
    for (int nid : active_ids) {
        trie.nodes.at(nid).depth += 1;
    }
    for (size_t i = 0; i < readout_rows.size(); ++i) {
        const int nid = active_ids[readout_rows[i]];
        auto & nd = trie.nodes.at(nid);
        nd.readouts[nd.depth]    = readout_argmax[i];
        nd.has_readout[nd.depth] = true;
        if (i < trie.last_pmax.size()) {
            nd.readout_pmax[nd.depth] = trie.last_pmax[i];
        }
        if (nd.depth == trie.cfg.R && trie.sampler && trie.sampler->capture) {
            trie.sampler->capture((int) i, nd.sample_r);
            nd.has_sample_r = true;
        }
    }
    trie.last_pmax.clear();
    // 3DSD Lever 4 (Opt-In): Native varloop d2 early exit when argmax(d1)==argmax(d2) and p_max >= threshold.
    // Walk strictly from cursor_id along matching depth-R parents so every promoted node is
    // guaranteed to be committed immediately by run_verify() (copying plane 1 -> plane 2).
    if (trie.cfg.early_exit_p > 0.0f && trie.cfg.R == 3 && !trie.sampler) {
        int cur = trie.cursor_id;
        while (cur >= 0) {
            auto it = trie.nodes.find(cur);
            if (it == trie.nodes.end()) break;
            auto & nd = it->second;
            if (nd.depth == 2 && nd.position < trie.output_limit &&
                nd.has_readout[1] && nd.has_readout[2] &&
                nd.readouts[2] == nd.readouts[1] &&
                nd.readout_pmax[2] >= trie.cfg.early_exit_p) {
                nd.depth = trie.cfg.R;
                nd.readouts[trie.cfg.R]    = nd.readouts[2];
                nd.has_readout[trie.cfg.R] = true;
                nd.early_exited_d2 = true;
                ++trie.stats.early_exits;
            }
            if (nd.depth < trie.cfg.R || !nd.has_readout[trie.cfg.R]) break;
            auto cit = nd.children.find(nd.readouts[trie.cfg.R]);
            if (cit == nd.children.end()) break;
            cur = cit->second;
        }
    }
    if (!trie.run_verify(on_commit, stopped, error)) return false;
    if (!stopped) {
        trie.run_stage2(active_ids);
        trie.run_stage1();
    }
    return true;
}

void Inkling2TwoDSDSession::update_config(const Inkling2TwoDSDConfig & cfg) {
    impl_->trie.cfg = cfg;
}

int Inkling2TwoDSDSession::live_node_count() const {
    return (int) impl_->trie.nodes.size();
}

int32_t Inkling2TwoDSDSession::cursor_token() const {
    const auto & trie = impl_->trie;
    auto it = trie.nodes.find(trie.cursor_id);
    return (it != trie.nodes.end()) ? it->second.token : -1;
}

int Inkling2TwoDSDSession::cursor_pos() const {
    const auto & trie = impl_->trie;
    auto it = trie.nodes.find(trie.cursor_id);
    return (it != trie.nodes.end()) ? it->second.position : -1;
}

int Inkling2TwoDSDSession::cursor_depth() const {
    const auto & trie = impl_->trie;
    auto it = trie.nodes.find(trie.cursor_id);
    return (it != trie.nodes.end()) ? it->second.depth : -1;
}

void Inkling2TwoDSDSession::clear_cursor_children() {
    auto & trie = impl_->trie;
    auto it = trie.nodes.find(trie.cursor_id);
    if (it == trie.nodes.end()) return;
    std::vector<int> drop;
    for (const auto & kv : it->second.children) {
        trie.collect_descendants(kv.second, drop);
    }
    if (!drop.empty()) {
        trie.drop_subtrees(drop);
    }
    it->second.children.clear();
    trie.events.clear();
    trie.blocks.clear();
    trie.blocks[trie.cursor_id] = { trie.cursor_id };
}

bool inkling2_2dsd_run(
        Inkling2TwoDSDRunner & runner,
        const Inkling2TwoDSDConfig & cfg,
        int32_t first_token,
        int max_new,
        const std::function<bool(int32_t &)> & on_commit,
        const Inkling2TwoDSDSamplerHooks * sampler,
        Inkling2TwoDSDStats & stats,
        std::string & error,
        const std::vector<int32_t> * prompt_history,
        const std::vector<float> * last_pmax) {
    stats = {};
    if (max_new <= 0) return true;

    int32_t first = first_token;
    const bool more0 = on_commit(first);
    if (!more0) return true;

    const int pref0 = runner.prefix_length();
    const int output_limit = std::min(pref0 + max_new, runner.max_ctx());
    if (pref0 >= output_limit) return true;

    Inkling2TwoDSDSession session(runner, cfg, stats, output_limit, sampler);
    if (prompt_history && !prompt_history->empty()) {
        session.set_prompt_history(*prompt_history);
    }
    if (!session.seed_cursor(first, pref0 - 1, error)) return false;

    auto run_pending_drafts = [&]() -> bool {
        std::vector<Inkling2TwoDSDDraftReq> reqs;
        std::vector<int> heads, widths;
        session.collect_draft_reqs(reqs, heads, widths);
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
        session.apply_draft_outs(heads, widths, outs);
        return true;
    };

    if (!run_pending_drafts()) return false;

    const uint32_t rmask = session.readout_mask();
    bool stopped = false;
    while (!stopped) {
        std::vector<int> active_ids;
        std::vector<Inkling2TwoDSDNodeDesc> descs;
        if (!session.collect_active(active_ids, descs, error)) return false;

        std::vector<int> readout_rows;
        std::vector<int32_t> readout_argmax;
        if (!runner.advance_nodes(descs, rmask, readout_rows, readout_argmax)) {
            error = "inkling2_2dsd: advance_nodes failed";
            return false;
        }
        ++stats.target_calls;
        stats.target_cells += (long long) active_ids.size();

        if (last_pmax && !last_pmax->empty()) {
            session.set_last_readout_pmax(*last_pmax);
        }
        if (!session.apply_step_and_verify(active_ids, readout_rows, readout_argmax, on_commit, stopped, error)) {
            return false;
        }
        if (!stopped && !run_pending_drafts()) return false;
    }
    return true;
}

bool inkling2_2dsd_run_greedy(
        Inkling2TwoDSDRunner & runner,
        const Inkling2TwoDSDConfig & cfg,
        int32_t first_token,
        int max_new,
        const std::function<bool(int32_t &)> & on_commit,
        Inkling2TwoDSDStats & stats,
        std::string & error) {
    return inkling2_2dsd_run(runner, cfg, first_token, max_new, on_commit, nullptr, stats, error);
}

} // namespace luce::common
