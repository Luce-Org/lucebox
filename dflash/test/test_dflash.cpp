// DFlash speculative decoding end-to-end test.
//
// Pipeline:
//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).
//   2. Prefill: single-token decode over the prompt, capture_layers=true so
//      target_feat gets populated for every prompt pos.
//   3. Decode loop (until max_new):
//      a. Build noise block [last_tok, MASK*15] on CPU via target.tok_embd.
//      b. Draft forward (uses target_feat[0..committed] + noise) → 16 candidates.
//      c. snapshot SSM state. Batched target verify on the 16 draft tokens with
//         causal mask, capture_layers=true.
//      d. Greedy longest-prefix accept + 1 bonus token from target's argmax.
//      e. Restore SSM state. Replay the accepted tokens through target (batched
//         with causal mask, capture_layers=true) so state + target_feat are
//         cleanly advanced only by what was committed.
//      f. Update committed, last_tok.
//
// Usage: test_dflash <target.gguf> <draft.safetensors> <prompt_ids.bin>
//                    <n_gen> <out_ids.bin>

#include "dflash27b.h"
#include "internal.h"
#include "dflash_graph.h"
#include "qwen3_drafter.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

// Half-precision → f32 widen kernel launchers (src/f16_convert.cu). Used by
// the DDtree rollback (ssm_intermediate slot → cache.ssm_state) and the
// drafter prep path (target_feat → sg.target_hidden_cat). We store the
// per-token intermediate cache in f16 and the target_feat buffer in bf16 to
// halve their memory footprint.
extern "C" void dflash27b_launch_f16_to_f32(const void * src,
                                            void * dst,
                                            size_t n_elems,
                                            cudaStream_t stream);
extern "C" void dflash27b_launch_bf16_to_f32(const void * src,
                                             void * dst,
                                             size_t n_elems,
                                             cudaStream_t stream);

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cmath>
#include <cstdint>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <windows.h>
#ifdef _WIN64
#define ssize_t __int64
#else
#define ssize_t long
#endif
#else
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace dflash27b;

// True iff `tok` matches one of the model's declared end-of-output ids
// (loaded into TargetWeights from GGUF tokenizer metadata). Replaces the
// previous hardcoded `tok == 248045` check at the spec-decode commit
// sites: token 248045 is `<|im_start|>` (chat-START), not EOS, and the
// old check truncated chat-formatted output at the natural turn boundary
// `<|im_end|>\n<|im_start|>`. The `>= 0` guards make a missing GGUF key
// (-1 sentinel) a never-match.
#define IS_EOS_TOK(tok, w)                                         \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id)                  \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

static constexpr int PREFIX_CACHE_SLOTS = 8;

// ─── Small utilities ──────────────────────────────────────────────

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

static bool write_int32_file(const std::string & path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

#if defined(_WIN32)
using StreamFd = uintptr_t;
static constexpr StreamFd kInvalidStreamFd = static_cast<StreamFd>(-1);
static StreamFd parse_stream_fd(const char * text) {
    return static_cast<StreamFd>(_strtoui64(text, nullptr, 10));
}
#else
using StreamFd = int;
static constexpr StreamFd kInvalidStreamFd = -1;
static StreamFd parse_stream_fd(const char * text) {
    return std::atoi(text);
}
#endif

static bool parse_int(const std::string & text, int & out) {
    char * end = nullptr;
    long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') return false;
    out = static_cast<int>(value);
    return true;
}

static bool parse_daemon_generate_command(const std::string & line,
                                          std::string & prompt_path,
                                          int & n_gen) {
    const size_t tab = line.find('\t');
    if (tab != std::string::npos) {
        prompt_path = line.substr(0, tab);
        return parse_int(line.substr(tab + 1), n_gen);
    }

    char ppath[1024];
    int parsed_n_gen = 0;
    if (std::sscanf(line.c_str(), "%1023s %d", ppath, &parsed_n_gen) != 2) {
        return false;
    }
    prompt_path = ppath;
    n_gen = parsed_n_gen;
    return true;
}

static bool parse_daemon_compress_command(const std::string & line,
                                          std::string & src_path,
                                          int & keep_x1000,
                                          std::string & drafter_path) {
    constexpr const char * tab_prefix = "compress\t";
    constexpr size_t tab_prefix_len = 9;
    if (line.compare(0, tab_prefix_len, tab_prefix) == 0) {
        const size_t first = line.find('\t', tab_prefix_len);
        if (first == std::string::npos) return false;
        const size_t second = line.find('\t', first + 1);
        if (second == std::string::npos) return false;
        src_path = line.substr(tab_prefix_len, first - tab_prefix_len);
        if (!parse_int(line.substr(first + 1, second - first - 1), keep_x1000)) {
            return false;
        }
        drafter_path = line.substr(second + 1);
        return !src_path.empty() && !drafter_path.empty();
    }

    char ppath[1024];
    char dpath[1024];
    int parsed_keep = 0;
    const int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s",
                              ppath, &parsed_keep, dpath);
    if (n != 3) return false;
    src_path = ppath;
    keep_x1000 = parsed_keep;
    drafter_path = dpath;
    return true;
}

static bool parse_daemon_continue_command(const std::string & line, int & n_gen) {
    constexpr const char * cont_long = "CONTINUE";
    constexpr const char * cont_short = "CONT";
    size_t pos = std::string::npos;
    if (line.rfind(cont_long, 0) == 0) {
        pos = std::strlen(cont_long);
    } else if (line.rfind(cont_short, 0) == 0) {
        pos = std::strlen(cont_short);
    } else {
        return false;
    }
    if (pos >= line.size() || (line[pos] != ' ' && line[pos] != '\t' && line[pos] != '=')) {
        return false;
    }
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t' || line[pos] == '=')) pos++;
    if (pos >= line.size()) return false;
    return parse_int(line.substr(pos), n_gen);
}

static bool parse_daemon_start_command(const std::string & line,
                                       std::string & prompt_path,
                                       int & total_gen,
                                       int & quantum) {
    constexpr const char * start_kw = "START";
    if (line.rfind(start_kw, 0) != 0) return false;
    if (line.size() <= 5 || (line[5] != ' ' && line[5] != '\t')) return false;
    char ppath[1024];
    int parsed_total = 0;
    int parsed_quantum = 0;
    const int n = std::sscanf(line.c_str() + 5, "%1023s %d %d",
                              ppath, &parsed_total, &parsed_quantum);
    if (n != 3) return false;
    prompt_path = ppath;
    total_gen = parsed_total;
    quantum = parsed_quantum;
    return true;
}

static bool parse_daemon_slot_prefix(std::string & line, int & slot_id) {
    slot_id = 0;
    bool has_prefix = false;
    size_t pos = 0;
    if (line.rfind("SLOT", 0) == 0 || line.rfind("slot", 0) == 0) {
        if (line.size() > 4 && (line[4] == ' ' || line[4] == '\t')) {
            has_prefix = true;
            pos = 5;
        } else if (line.size() > 5 && line[4] == '=') {
            has_prefix = true;
            pos = 5;
        }
    }
    if (!has_prefix) return true;

    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size()) return false;

    char * end = nullptr;
    long parsed = std::strtol(line.c_str() + pos, &end, 10);
    if (end == line.c_str() + pos || parsed < 0 || parsed > 1024) return false;
    size_t end_pos = (size_t)(end - line.c_str());
    if (end_pos >= line.size()) return false;
    if (line[end_pos] != ' ' && line[end_pos] != '\t') return false;
    while (end_pos < line.size() && (line[end_pos] == ' ' || line[end_pos] == '\t')) end_pos++;
    if (end_pos >= line.size()) return false;
    slot_id = (int)parsed;
    line.erase(0, end_pos);
    return true;
}

static bool parse_daemon_request_prefix(std::string & line, int & request_id) {
    request_id = 0;
    size_t prefix_len = 0;
    if (line.rfind("REQ", 0) == 0) {
        prefix_len = 3;
    } else if (line.rfind("REQUEST", 0) == 0) {
        prefix_len = 7;
    } else {
        return true;
    }

    size_t pos = prefix_len;
    if (line.size() <= pos) return false;
    if (line[pos] == '=') {
        pos++;
    } else if (line[pos] == ' ' || line[pos] == '\t') {
        while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    } else {
        return true;
    }
    if (pos >= line.size()) return false;

    char * end = nullptr;
    long parsed = std::strtol(line.c_str() + pos, &end, 10);
    if (end == line.c_str() + pos || parsed < 0 || parsed > 0x3fffffff) return false;

    size_t end_pos = (size_t)(end - line.c_str());
    while (end_pos < line.size() && (line[end_pos] == ' ' || line[end_pos] == '\t')) end_pos++;
    if (end_pos >= line.size()) return false;
    request_id = (int)parsed;
    line.erase(0, end_pos);
    return true;
}

static bool load_draft_auto(const std::string & path,
                            ggml_backend_t backend,
                            DraftWeights & out) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".gguf") {
        return load_draft_gguf(path, backend, out);
    }
    return load_draft_safetensors(path, backend, out);
}

static int argmax_f32(const float * x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

// ggml_flash_attn_ext expects kv_len aligned to KQ_MASK_PAD (32) on the
// f16/Q* paths, and to FATTN_KQ_STRIDE (256) on the TurboQuant FA paths.
// The global `g_kq_stride_pad` below is set at init time and applied by
// both build_causal_mask and build_tree_mask so the mask dim matches the
// K/V view length used in build_attn_block.
static constexpr int KQ_MASK_PAD = 32;
static int g_kq_stride_pad = KQ_MASK_PAD;   // overridden to 256 when TBQ KV is active
static int g_max_ctx_override = 0;           // overridden by --max-ctx=N (default 4096)
static int g_fa_window       = 2048;         // overridden by DFLASH27B_FA_WINDOW=N
static int g_draft_swa_window = 0;           // draft SWA window (0 = disabled); --draft-swa=N
static int g_draft_ctx_max   = 4096;        // draft context cap; --draft-ctx-max=N
static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// F16 encoding for the two values we use: 0 and -inf.
// 0 in F16 is 0x0000. -inf is 0xFC00.
static constexpr uint16_t F16_ZERO = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static void build_causal_mask(std::vector<uint16_t> & out,
                              int kv_len, int n_tokens, int kv_start,
                              int win_start = 0) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int abs_end = win_start + kv_len;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int min_k = std::max(0, win_start);
        const int max_k = abs_q;
        for (int k = min_k; k <= max_k && k < abs_end; k++) {
            out[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
    }
}

// ─── DDTree support (ported from liranringel/ddtree/ddtree.py) ────────

// Per-position top-K softmax extraction. Computes log-probabilities (needed
// so that cross-depth prefix comparisons in the best-first heap are valid)
// via a single pass over the vocab that also maintains top-K in a heap and
// computes logsumexp online. Runs on CPU since draft logits are already on
// host after ggml_backend_tensor_get.
//
// Input:  logits [n_positions × vocab] f32
// Output: out_log_probs [n_positions × K] f32, out_token_ids [n_positions × K] i32
//         both sorted by log-probability DESCENDING (rank 0 = argmax).
static void extract_draft_topk(const float * logits,
                               int n_positions, int vocab, int K,
                               float * out_log_probs,
                               int32_t * out_token_ids,
                               float temperature = 1.0f) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    // Temperature scaling: dividing logits by T<1 sharpens the softmax,
    // widening the gap between top-1 and lower ranks. This compensates for
    // Q4_K_M quantization that flattens the draft's softmax — without it,
    // pure best-first picks shallow bushy trees instead of going deep.
    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    // Parallelize across positions when OpenMP is enabled — each i is independent.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);

        // Online log-sum-exp with running max. Single pass over the vocab,
        // simultaneously maintaining top-K.
        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            // Online logsumexp
            if (l > running_max) {
                // rescale previous sum to the new max
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            // Top-K maintenance
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

        // Sort the K entries descending (largest logit first) and emit.
        // sort_heap with cmp_greater on a min-heap already produces descending
        // order (cppreference: "sort_heap leaves the range sorted in the same
        // order as sort would with the same comparator" — greater→descending).
        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

// A flat DFS-ordered tree built from the draft's top-K softmax distributions.
// Slot 0 is the tree root (the bonus token from the previous spec round);
// slots 1..n_nodes are the DFS-ordered tree nodes. `parents[i]` gives each
// node's parent index in the same flat array (parents[0] = -1). `depth[i]`
// is the absolute depth within the block-diffusion prediction window, with
// the root at depth 0 and its children at depth 1. `child_maps[i]` maps a
// token_id to the child's flat index, used for the tree walk post-verify.
// `visibility[i][j]` (ancestor-only mask) is true iff j is an ancestor of i
// in the tree (including j == i); used to build the attention mask.
struct DDTree {
    int                         n_nodes = 0;          // excludes root
    std::vector<int32_t>        token_ids;            // size n_nodes
    std::vector<int>            depths;               // size n_nodes (1..L)
    std::vector<int>            parents;              // size n_nodes + 1
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // size n_nodes + 1
    std::vector<uint8_t>        visibility;           // (1 + n_nodes)^2 row-major
};

// Port of build_ddtree_tree() from ddtree.py. Runs a best-first heap over
// prefixes of the per-position top-K distributions, pops until `budget`
// nodes are accumulated. Populates the flat DFS-ordered tree structure.
//
// top_log_probs: [L × K]  the drafter's per-position top-K log-probabilities
// top_token_ids: [L × K]  matching token ids, rank 0 = argmax per position
// L:             max tree depth (e.g. q_len - 1 for a block diffusion block)
// K:             top-K per position (same as used in extract_draft_topk)
// budget:        maximum number of non-root tree nodes
static DDTree build_ddtree(const float * top_log_probs,
                           const int32_t * top_token_ids,
                           int L, int K, int budget,
                           bool chain_seed = true) {
    DDTree tree;
    if (budget <= 0 || L <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    // Heap entry:
    //   neg_logw, ranks (encoded as a small vector), parent_index, depth, rank, logw
    // We sort by neg_logw ASCENDING, which is equivalent to logw DESCENDING.
    struct HeapEntry {
        float                neg_logw;
        std::vector<int>     ranks;        // rank tuple used only to prevent duplicate state; not strictly needed
        int                  parent_index; // index in the flat tree of this candidate's parent
        int                  depth;        // 1..L
        int                  rank;         // rank within top-K at depth-1 (0-indexed)
        float                logw;         // actual log-prob sum so far
    };
    struct HeapCmp {
        bool operator()(const HeapEntry & a, const HeapEntry & b) const {
            // std::priority_queue is a max-heap; we want SMALLEST neg_logw at the top
            // so that we pop the highest-probability prefix first.
            return a.neg_logw > b.neg_logw;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);
    tree.parents.push_back(-1);                 // root
    tree.child_maps.emplace_back();             // root's children

    // Two seeding strategies:
    //   - chain_seed=true: pre-seed full top-1 chain (defensive, guarantees
    //     AL >= chain mode even with flat-softmax draft like Q4_K_M). Compensates
    //     for quantization that shrinks top-1/top-2 logp gap.
    //   - chain_seed=false: paper's pure best-first — heap starts with just
    //     the depth-1 top-1 child of the root. Tree shape emerges from log-prob
    //     ordering. Works only when the draft top-1 is dominant enough.
    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({
                    /*neg_logw*/ -sibling_logw,
                    /*ranks   */ {1},
                    /*parent  */ prev_idx,
                    /*depth   */ d,
                    /*rank    */ 1,
                    /*logw    */ sibling_logw,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        // Paper-style pure best-first: seed heap with depth-1 top-1 only.
        const float root_logw = top_log_probs[0 * K + 0];
        heap.push({
            /*neg_logw*/ -root_logw,
            /*ranks   */ {0},
            /*parent  */ 0,  // root flat index
            /*depth   */ 1,
            /*rank    */ 0,
            /*logw    */ root_logw,
        });
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int    depth_minus_1 = top.depth - 1;
        const int    rank          = top.rank;
        const int32_t token_id     = top_token_ids[(size_t)depth_minus_1 * K + rank];

        const int current_index = tree.n_nodes + 1;  // slot in flat tree
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        // Push next sibling (same depth, next-best rank at this depth).
        if (rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)depth_minus_1 * K + rank]
                + top_log_probs[(size_t)depth_minus_1 * K + rank + 1];
            std::vector<int> sibling_ranks = top.ranks;
            sibling_ranks.back() = rank + 1;
            heap.push({
                /*neg_logw*/ -sibling_logw,
                /*ranks   */ std::move(sibling_ranks),
                /*parent  */ top.parent_index,
                /*depth   */ top.depth,
                /*rank    */ rank + 1,
                /*logw    */ sibling_logw,
            });
        }

        // Push first child (next depth, top-1 rank under this node).
        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth /*new depth_minus_1*/ * K + 0];
            std::vector<int> child_ranks = top.ranks;
            child_ranks.push_back(0);
            heap.push({
                /*neg_logw*/ -child_logw,
                /*ranks   */ std::move(child_ranks),
                /*parent  */ current_index,
                /*depth   */ top.depth + 1,
                /*rank    */ 0,
                /*logw    */ child_logw,
            });
        }
    }

    // Build ancestor-only visibility mask (flat row-major, (1+n)^2).
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t)N * N, 0);
    tree.visibility[0 * N + 0] = 1;  // root sees itself
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[i];  // immediate parent
        // Inherit the parent's visibility row up to column i-1,
        // then mark self at column i.
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] = tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }

    return tree;
}

// Walk the verified tree following the target's argmax (posterior) at each
// node. Returns the list of flat-tree indices that make up the accepted path
// (starting at root), plus the next "bonus" token (target's argmax at the
// deepest accepted node, which didn't match any of that node's children).
static std::vector<int> follow_verified_tree(const DDTree & tree,
                                             const int32_t * posterior,
                                             int & out_next_token) {
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
    return accepted;
}

// Build an f16 ancestor-only attention mask for tree verify:
//   mask[q=i][k<past_length]          = 0    (past KV cache, attend freely)
//   mask[q=i][k=past_length+j]        = 0 iff j is an ancestor of i in the tree
//                                              (including j == i)
//                                     = -inf otherwise
// Shape matches the ggml flash_attn_ext expectation: [kv_pad, q_pad] f16.
static void build_tree_mask(const DDTree & tree, int past_length,
                            std::vector<uint16_t> & out_mask,
                            int win_start = 0) {
    const int N      = 1 + tree.n_nodes;
    const int win_len = past_length + N - win_start;
    const int kv_pad = align_up(win_len, g_kq_stride_pad);
    const int q_pad  = align_up(N,      KQ_MASK_PAD);
    out_mask.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < N; q++) {
        for (int k = std::max(0, win_start); k < past_length; k++) {
            out_mask[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j]) {
                out_mask[(size_t)q * kv_pad + (past_length + j - win_start)] = F16_ZERO;
            }
        }
    }
}

// ─── StepGraph — rebuilt per call since kv_len varies ──

struct StepGraph {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;

    // Named inputs (look up via ggml_get_tensor by name)
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;     // may be null
    ggml_tensor *   parent_ids = nullptr;    // DDTree tree-mode; null for chain mode
    ggml_tensor *   target_hidden_cat = nullptr;  // draft only
    ggml_tensor *   positions_k = nullptr;        // draft only
    ggml_tensor *   hidden_input = nullptr;        // lm-head projection only

    // Output
    ggml_tensor *   logits = nullptr;
    ggml_tensor *   hidden_states = nullptr;       // draft hidden-only output
    ggml_tensor *   argmax_tokens = nullptr; // [n_tokens] i32, GPU-side argmax of logits
    ggml_tensor *   topk_indices = nullptr;  // [K, n_tokens] i32, GPU-side top-K indices

    // Per-delta-net-layer captures (verify only). One entry per delta-net layer.
    // Each entry's tensors are graph views on the gated_delta_net result:
    //   ssm_intermediate_states: [S_v, S_v, H_v, n_tokens]  (f32, ~50 MB/layer for n_tokens=16)
    //   conv_input:              [kernel-1+n_tokens, conv_channels, 1]
    // Marked as graph outputs so their data is valid after ggml_backend_graph_compute.
    std::vector<DeltaNetCapture> delta_captures;
};

struct DraftFeatureMirror {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * target_feat = nullptr; // F32 [5*hidden, cap]
    void * bf16_staging = nullptr;
    size_t bf16_staging_elems = 0;
    int device = 0;
    int target_device = 0;
    int cap = 0;
};

struct DaemonSlotState {
    TargetCache cache;
    PrefixSnapshot prefix_snapshots[PREFIX_CACHE_SLOTS];
    StepGraph sg;
    StepGraph draft_sg;
    StepGraph proj_sg;
    DraftFeatureMirror feature_mirror;
    bool first_iter = true;
};

struct DaemonRequestState {
    bool active = false;
    int request_id = 0;
    int slot_id = 0;
    int remaining = 0;
    int quantum = 0;
    int emitted = 0;
    int epoch = 0;
};

struct DaemonPendingQuantum {
    int request_id = 0;
    int slot_id = 0;
    int epoch = 0;
    int n_gen = 0;
};

struct DaemonSlotRuntimeRef {
    int slot_id = 0;
    TargetCache * cache = nullptr;
    StepGraph * sg = nullptr;
    StepGraph * draft_sg = nullptr;
    StepGraph * proj_sg = nullptr;
    DraftFeatureMirror * feature_mirror = nullptr;
    bool * first_iter = nullptr;
};

struct DaemonBatchCandidate {
    int request_id = 0;
    int slot_id = 0;
    int epoch = 0;
    int n_gen = 0;
    int remaining = 0;
    DaemonSlotRuntimeRef runtime;
};

struct DaemonAlignedBucketSelection {
    std::vector<DaemonBatchCandidate> batch;
    int kv_start = -1;
    int considered = 0;
    int eligible = 0;
    int ineligible = 0;
    int buckets = 0;
    int singleton_buckets = 0;
    int front_kv_start = -1;
    int front_bucket_size = 0;
    int blocked_by_front_singleton = 0;
    int last_selected_slot_id = -1;
    bool tail_only = false;
};

static DaemonAlignedBucketSelection select_aligned_scheduler_bucket(
    const std::vector<DaemonBatchCandidate> & candidates,
    int max_batch,
    bool tail_only) {
    DaemonAlignedBucketSelection out{};
    out.tail_only = tail_only;
    out.considered = (int)candidates.size();
    if (max_batch < 2 || candidates.empty()) return out;

    std::unordered_map<int, std::vector<size_t>> by_pos;
    std::vector<int> pos_order;
    for (size_t i = 0; i < candidates.size(); i++) {
        const DaemonBatchCandidate & candidate = candidates[i];
        TargetCache * c = candidate.runtime.cache;
        if (!c || c->cur_pos <= 0 || c->last_tok < 0
            || candidate.remaining <= 0
            || (tail_only && candidate.remaining != 1)) {
            out.ineligible++;
            continue;
        }
        out.eligible++;
        const int pos = c->cur_pos;
        auto it = by_pos.find(pos);
        if (it == by_pos.end()) {
            pos_order.push_back(pos);
            it = by_pos.emplace(pos, std::vector<size_t>{}).first;
        }
        if (out.front_kv_start < 0) out.front_kv_start = pos;
        it->second.push_back(i);
    }

    out.buckets = (int)pos_order.size();
    for (int pos : pos_order) {
        const int bucket_size = (int)by_pos[pos].size();
        if (bucket_size == 1) out.singleton_buckets++;
        if (pos == out.front_kv_start) out.front_bucket_size = bucket_size;
    }

    if (out.front_kv_start < 0) return out;
    if (out.front_bucket_size < 2) {
        out.blocked_by_front_singleton = out.front_bucket_size == 1 ? 1 : 0;
        return out;
    }

    const std::vector<size_t> & selected = by_pos[out.front_kv_start];
    const int n = std::min(max_batch, (int)selected.size());
    if (n < 2) return out;
    out.kv_start = out.front_kv_start;
    out.batch.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        const DaemonBatchCandidate & candidate = candidates[selected[(size_t)i]];
        out.batch.push_back(candidate);
        out.last_selected_slot_id = candidate.slot_id;
    }
    return out;
}

static void print_aligned_bucket_log(const DaemonAlignedBucketSelection & selection,
                                     int max_batch) {
    const bool ready = (int)selection.batch.size() >= 2;
    std::printf("[scheduler] aligned_bucket_%s count=%d kv_start=%d buckets=%d considered=%d eligible=%d ineligible=%d singleton_buckets=%d front_kv_start=%d front_bucket_size=%d blocked_by_front_singleton=%d max_batch=%d tail_only=%d",
                ready ? "ready" : "miss",
                (int)selection.batch.size(),
                selection.kv_start,
                selection.buckets,
                selection.considered,
                selection.eligible,
                selection.ineligible,
                selection.singleton_buckets,
                selection.front_kv_start,
                selection.front_bucket_size,
                selection.blocked_by_front_singleton,
                max_batch,
                (int)selection.tail_only);
    for (const DaemonBatchCandidate & candidate : selection.batch) {
        std::printf(" req=%d:slot=%d:epoch=%d:n=%d:remaining=%d",
                    candidate.request_id,
                    candidate.slot_id,
                    candidate.epoch,
                    candidate.n_gen,
                    candidate.remaining);
    }
    std::printf("\n");
    std::fflush(stdout);
}

static size_t next_cursor_after_aligned_batch(
    const DaemonAlignedBucketSelection & selection,
    size_t n_slots,
    size_t current_cursor) {
    if (n_slots == 0 || selection.last_selected_slot_id < 0) {
        return current_cursor;
    }
    return ((size_t)selection.last_selected_slot_id + 1) % n_slots;
}

static int run_scheduler_bucket_selftest() {
    auto make_candidates = [](
        const std::vector<int> & positions,
        const std::vector<int> & last_tokens,
        const std::vector<int> & remaining) {
        std::vector<TargetCache> caches(positions.size());
        std::vector<DaemonBatchCandidate> candidates;
        candidates.reserve(positions.size());
        for (size_t i = 0; i < positions.size(); i++) {
            caches[i].cur_pos = positions[i];
            caches[i].last_tok = last_tokens[i];
            DaemonSlotRuntimeRef runtime{};
            runtime.slot_id = (int)i;
            runtime.cache = &caches[i];
            candidates.push_back(DaemonBatchCandidate{
                (int)i + 1,
                (int)i,
                1,
                1,
                remaining[i],
                runtime});
        }
        return std::pair<std::vector<TargetCache>,
                         std::vector<DaemonBatchCandidate>>(
            std::move(caches), std::move(candidates));
    };
    auto require_true = [](bool ok, const char * label) {
        if (!ok) std::fprintf(stderr, "[scheduler-test] failed: %s\n", label);
        return ok;
    };

    {
        auto data = make_candidates({17, 17, 10}, {1, 2, 3}, {4, 4, 4});
        DaemonAlignedBucketSelection sel =
            select_aligned_scheduler_bucket(data.second, 2, false);
        if (!require_true(sel.batch.size() == 2
                          && sel.kv_start == 17
                          && sel.buckets == 2
                          && sel.singleton_buckets == 1,
                          "front aligned bucket selected")) return 1;
    }
    {
        auto data = make_candidates({10, 17}, {1, 2}, {4, 4});
        DaemonAlignedBucketSelection sel =
            select_aligned_scheduler_bucket(data.second, 2, false);
        if (!require_true(sel.batch.empty()
                          && sel.blocked_by_front_singleton == 1,
                          "singleton front falls back")) return 1;
    }
    {
        auto data = make_candidates({20, 20, 20}, {1, 2, 3}, {2, 1, 1});
        DaemonAlignedBucketSelection sel =
            select_aligned_scheduler_bucket(data.second, 2, true);
        if (!require_true(sel.batch.size() == 2
                          && sel.eligible == 2
                          && sel.ineligible == 1
                          && sel.kv_start == 20,
                          "tail_only filters remaining != 1")) return 1;
    }
    {
        auto data = make_candidates({0, 18, 18}, {-1, 7, 8}, {1, 1, 1});
        DaemonAlignedBucketSelection sel =
            select_aligned_scheduler_bucket(data.second, 2, false);
        if (!require_true(sel.batch.size() == 2
                          && sel.eligible == 2
                          && sel.ineligible == 1,
                          "invalid cur_pos/last_tok filtered")) return 1;
    }
    {
        auto data = make_candidates({10, 17, 17}, {1, 2, 3}, {4, 4, 4});
        DaemonAlignedBucketSelection sel =
            select_aligned_scheduler_bucket(data.second, 2, false);
        if (!require_true(sel.batch.empty()
                          && sel.front_kv_start == 10
                          && sel.front_bucket_size == 1
                          && sel.blocked_by_front_singleton == 1,
                          "fairness blocks tail bucket behind singleton")) return 1;
    }
    {
        auto data = make_candidates({17, 17, 17}, {1, 2, 3}, {4, 4, 4});
        DaemonAlignedBucketSelection sel =
            select_aligned_scheduler_bucket(data.second, 2, false);
        size_t next = next_cursor_after_aligned_batch(sel, 3, 0);
        if (!require_true(sel.batch.size() == 2
                          && sel.last_selected_slot_id == 1
                          && next == 2,
                          "cursor advances after last selected slot")) return 1;
    }

    std::printf("[scheduler-test] aligned bucket selftest ok\n");
    std::fflush(stdout);
    return 0;
}

static bool cuda_copy_bytes_d2d(const void * src, void * dst, size_t bytes) {
    if (bytes == 0) return true;
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy D2D failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

static bool copy_target_cache_to_batch_slot(const TargetCache & src,
                                            TargetCache & dst,
                                            int seq_idx) {
    if (seq_idx < 0) return false;
    if (src.attn_k.size() != dst.attn_k.size()
        || src.attn_v.size() != dst.attn_v.size()
        || src.ssm_state.size() != dst.ssm_state.size()
        || src.conv_state.size() != dst.conv_state.size()) {
        std::fprintf(stderr, "[scheduler] batch probe cache shape mismatch\n");
        return false;
    }

    for (size_t i = 0; i < src.attn_k.size(); i++) {
        ggml_tensor * sk = src.attn_k[i];
        ggml_tensor * sv = src.attn_v[i];
        ggml_tensor * dk = dst.attn_k[i];
        ggml_tensor * dv = dst.attn_v[i];
        if (!sk || !sv || !dk || !dv || dk->ne[3] <= seq_idx || dv->ne[3] <= seq_idx) {
            std::fprintf(stderr, "[scheduler] batch probe invalid KV tensor\n");
            return false;
        }
        const int n_kv = (int)sk->ne[2];
        if (dk->ne[0] != sk->ne[0] || dk->ne[1] != sk->ne[1]
            || dk->ne[2] != sk->ne[2] || dv->ne[0] != sv->ne[0]
            || dv->ne[1] != sv->ne[1] || dv->ne[2] != sv->ne[2]) {
            std::fprintf(stderr, "[scheduler] batch probe KV dims mismatch\n");
            return false;
        }
        for (int h = 0; h < n_kv; h++) {
            const size_t k_src_off = (size_t)h * sk->nb[2];
            const size_t k_dst_off = (size_t)seq_idx * dk->nb[3] + (size_t)h * dk->nb[2];
            const size_t v_src_off = (size_t)h * sv->nb[2];
            const size_t v_dst_off = (size_t)seq_idx * dv->nb[3] + (size_t)h * dv->nb[2];
            if (!cuda_copy_bytes_d2d((const char *)sk->data + k_src_off,
                                     (char *)dk->data + k_dst_off,
                                     (size_t)sk->nb[2])) return false;
            if (!cuda_copy_bytes_d2d((const char *)sv->data + v_src_off,
                                     (char *)dv->data + v_dst_off,
                                     (size_t)sv->nb[2])) return false;
        }
    }

    for (size_t i = 0; i < src.ssm_state.size(); i++) {
        ggml_tensor * ss = src.ssm_state[i];
        ggml_tensor * ds = dst.ssm_state[i];
        ggml_tensor * sc = src.conv_state[i];
        ggml_tensor * dc = dst.conv_state[i];
        if (!ss || !ds || !sc || !dc || ds->ne[3] <= seq_idx || dc->ne[2] <= seq_idx) {
            std::fprintf(stderr, "[scheduler] batch probe invalid DeltaNet tensor\n");
            return false;
        }
        if (!cuda_copy_bytes_d2d(ss->data,
                                 (char *)ds->data + (size_t)seq_idx * ds->nb[3],
                                 ggml_nbytes(ss))) return false;
        if (!cuda_copy_bytes_d2d(sc->data,
                                 (char *)dc->data + (size_t)seq_idx * dc->nb[2],
                                 ggml_nbytes(sc))) return false;
    }
    if (src.target_feat && dst.target_feat) {
        if (dst.target_feat->ne[2] <= seq_idx
            || src.target_feat->ne[0] != dst.target_feat->ne[0]
            || src.target_feat->ne[1] != dst.target_feat->ne[1]
            || ggml_nbytes(src.target_feat) > (size_t)dst.target_feat->nb[2]) {
            std::fprintf(stderr, "[scheduler] batch probe target_feat dims mismatch\n");
            return false;
        }
        if (!cuda_copy_bytes_d2d(
                src.target_feat->data,
                (char *)dst.target_feat->data + (size_t)seq_idx * dst.target_feat->nb[2],
                ggml_nbytes(src.target_feat))) return false;
    }
    return true;
}

static bool copy_target_cache_to_single_cache(const TargetCache & src,
                                              TargetCache & dst) {
    if (src.max_ctx != dst.max_ctx
        || src.kv_k_type != dst.kv_k_type
        || src.kv_v_type != dst.kv_v_type
        || src.attn_k.size() != dst.attn_k.size()
        || src.attn_v.size() != dst.attn_v.size()
        || src.ssm_state.size() != dst.ssm_state.size()
        || src.conv_state.size() != dst.conv_state.size()) {
        std::fprintf(stderr, "[scheduler] batch reference cache shape mismatch\n");
        return false;
    }

    for (size_t i = 0; i < src.attn_k.size(); i++) {
        ggml_tensor * sk = src.attn_k[i];
        ggml_tensor * sv = src.attn_v[i];
        ggml_tensor * dk = dst.attn_k[i];
        ggml_tensor * dv = dst.attn_v[i];
        if (!sk || !sv || !dk || !dv
            || ggml_nbytes(sk) != ggml_nbytes(dk)
            || ggml_nbytes(sv) != ggml_nbytes(dv)) {
            std::fprintf(stderr, "[scheduler] batch reference invalid KV tensor\n");
            return false;
        }
        if (!cuda_copy_bytes_d2d(sk->data, dk->data, ggml_nbytes(sk))) return false;
        if (!cuda_copy_bytes_d2d(sv->data, dv->data, ggml_nbytes(sv))) return false;
    }

    for (size_t i = 0; i < src.ssm_state.size(); i++) {
        ggml_tensor * ss = src.ssm_state[i];
        ggml_tensor * sc = src.conv_state[i];
        ggml_tensor * ds = dst.ssm_state[i];
        ggml_tensor * dc = dst.conv_state[i];
        if (!ss || !sc || !ds || !dc
            || ggml_nbytes(ss) != ggml_nbytes(ds)
            || ggml_nbytes(sc) != ggml_nbytes(dc)) {
            std::fprintf(stderr, "[scheduler] batch reference invalid DeltaNet tensor\n");
            return false;
        }
        if (!cuda_copy_bytes_d2d(ss->data, ds->data, ggml_nbytes(ss))) return false;
        if (!cuda_copy_bytes_d2d(sc->data, dc->data, ggml_nbytes(sc))) return false;
    }

    dst.cur_pos  = src.cur_pos;
    dst.last_tok = src.last_tok;
    if (src.target_feat && dst.target_feat) {
        if (ggml_nbytes(src.target_feat) != ggml_nbytes(dst.target_feat)) {
            std::fprintf(stderr, "[scheduler] batch reference target_feat dims mismatch\n");
            return false;
        }
        if (!cuda_copy_bytes_d2d(src.target_feat->data, dst.target_feat->data,
                                 ggml_nbytes(src.target_feat))) return false;
    }
    return true;
}

static bool copy_batch_slot_to_target_cache(const TargetCache & src,
                                            int seq_idx,
                                            TargetCache & dst) {
    if (seq_idx < 0
        || src.max_ctx != dst.max_ctx
        || src.kv_k_type != dst.kv_k_type
        || src.kv_v_type != dst.kv_v_type
        || src.attn_k.size() != dst.attn_k.size()
        || src.attn_v.size() != dst.attn_v.size()
        || src.ssm_state.size() != dst.ssm_state.size()
        || src.conv_state.size() != dst.conv_state.size()) {
        std::fprintf(stderr, "[scheduler] batch commit cache shape mismatch\n");
        return false;
    }

    for (size_t i = 0; i < dst.attn_k.size(); i++) {
        ggml_tensor * sk = src.attn_k[i];
        ggml_tensor * sv = src.attn_v[i];
        ggml_tensor * dk = dst.attn_k[i];
        ggml_tensor * dv = dst.attn_v[i];
        if (!sk || !sv || !dk || !dv
            || sk->ne[3] <= seq_idx
            || sk->ne[0] != dk->ne[0] || sk->ne[1] != dk->ne[1]
            || sk->ne[2] != dk->ne[2]
            || sv->ne[3] <= seq_idx
            || sv->ne[0] != dv->ne[0] || sv->ne[1] != dv->ne[1]
            || sv->ne[2] != dv->ne[2]) {
            std::fprintf(stderr, "[scheduler] batch commit invalid KV tensor\n");
            return false;
        }
        const int n_kv = (int)dk->ne[2];
        for (int h = 0; h < n_kv; h++) {
            const size_t k_src_off = (size_t)seq_idx * sk->nb[3] + (size_t)h * sk->nb[2];
            const size_t k_dst_off = (size_t)h * dk->nb[2];
            const size_t v_src_off = (size_t)seq_idx * sv->nb[3] + (size_t)h * sv->nb[2];
            const size_t v_dst_off = (size_t)h * dv->nb[2];
            if (!cuda_copy_bytes_d2d((const char *)sk->data + k_src_off,
                                     (char *)dk->data + k_dst_off,
                                     (size_t)dk->nb[2])) return false;
            if (!cuda_copy_bytes_d2d((const char *)sv->data + v_src_off,
                                     (char *)dv->data + v_dst_off,
                                     (size_t)dv->nb[2])) return false;
        }
    }

    for (size_t i = 0; i < dst.ssm_state.size(); i++) {
        ggml_tensor * ss = src.ssm_state[i];
        ggml_tensor * sc = src.conv_state[i];
        ggml_tensor * ds = dst.ssm_state[i];
        ggml_tensor * dc = dst.conv_state[i];
        if (!ss || !sc || !ds || !dc
            || ss->ne[3] <= seq_idx
            || sc->ne[2] <= seq_idx
            || ggml_nbytes(ds) > (size_t)ss->nb[3]
            || ggml_nbytes(dc) > (size_t)sc->nb[2]) {
            std::fprintf(stderr, "[scheduler] batch commit invalid DeltaNet tensor\n");
            return false;
        }
        if (!cuda_copy_bytes_d2d((const char *)ss->data + (size_t)seq_idx * ss->nb[3],
                                 ds->data,
                                 ggml_nbytes(ds))) return false;
        if (!cuda_copy_bytes_d2d((const char *)sc->data + (size_t)seq_idx * sc->nb[2],
                                 dc->data,
                                 ggml_nbytes(dc))) return false;
    }
    if (src.target_feat && dst.target_feat) {
        if (src.target_feat->ne[2] <= seq_idx
            || src.target_feat->ne[0] != dst.target_feat->ne[0]
            || src.target_feat->ne[1] != dst.target_feat->ne[1]
            || ggml_nbytes(dst.target_feat) > (size_t)src.target_feat->nb[2]) {
            std::fprintf(stderr, "[scheduler] batch commit target_feat dims mismatch\n");
            return false;
        }
        if (!cuda_copy_bytes_d2d(
                (const char *)src.target_feat->data + (size_t)seq_idx * src.target_feat->nb[2],
                dst.target_feat->data,
                ggml_nbytes(dst.target_feat))) return false;
    }
    return true;
}

static void swap_daemon_slot_state(TargetCache & cache,
                                   PrefixSnapshot (&prefix_snapshots)[PREFIX_CACHE_SLOTS],
                                   StepGraph & sg,
                                   StepGraph & draft_sg,
                                   StepGraph & proj_sg,
                                   DraftFeatureMirror & feature_mirror,
                                   bool & first_iter,
                                   DaemonSlotState & slot) {
    using std::swap;
    swap(cache, slot.cache);
    for (int i = 0; i < PREFIX_CACHE_SLOTS; i++) {
        swap(prefix_snapshots[i], slot.prefix_snapshots[i]);
    }
    swap(sg, slot.sg);
    swap(draft_sg, slot.draft_sg);
    swap(proj_sg, slot.proj_sg);
    swap(feature_mirror, slot.feature_mirror);
    swap(first_iter, slot.first_iter);
}

struct ActiveDaemonSlot {
    TargetCache & cache;
    PrefixSnapshot (&prefix_snapshots)[PREFIX_CACHE_SLOTS];
    StepGraph & sg;
    StepGraph & draft_sg;
    StepGraph & proj_sg;
    DraftFeatureMirror & feature_mirror;
    bool & first_iter;
    DaemonSlotState * slot = nullptr;

    ActiveDaemonSlot(TargetCache & cache_,
                     PrefixSnapshot (&prefix_snapshots_)[PREFIX_CACHE_SLOTS],
                     StepGraph & sg_,
                     StepGraph & draft_sg_,
                     StepGraph & proj_sg_,
                     DraftFeatureMirror & feature_mirror_,
                     bool & first_iter_,
                     DaemonSlotState * slot_)
        : cache(cache_),
          prefix_snapshots(prefix_snapshots_),
          sg(sg_),
          draft_sg(draft_sg_),
          proj_sg(proj_sg_),
          feature_mirror(feature_mirror_),
          first_iter(first_iter_),
          slot(slot_) {
        if (slot) {
            swap_daemon_slot_state(cache, prefix_snapshots, sg, draft_sg,
                                   proj_sg, feature_mirror, first_iter, *slot);
        }
    }

    ActiveDaemonSlot(const ActiveDaemonSlot &) = delete;
    ActiveDaemonSlot & operator=(const ActiveDaemonSlot &) = delete;

    ~ActiveDaemonSlot() {
        if (slot) {
            swap_daemon_slot_state(cache, prefix_snapshots, sg, draft_sg,
                                   proj_sg, feature_mirror, first_iter, *slot);
        }
    }
};

static bool enable_peer_access_one_way(int device, int peer) {
    if (device == peer) return true;
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, device, peer);
    if (err != cudaSuccess || !can_access) return false;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) return false;
    err = cudaDeviceEnablePeerAccess(peer, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return true;
    }
    return err == cudaSuccess;
}

static bool enable_peer_access_pair(int a, int b) {
    if (a == b) return true;
    const bool ab = enable_peer_access_one_way(a, b);
    const bool ba = enable_peer_access_one_way(b, a);
    return ab && ba;
}

static bool copy_peer_async(void * dst, int dst_device,
                            const void * src, int src_device,
                            size_t bytes,
                            cudaStream_t stream = nullptr) {
    if (bytes == 0) return true;
    cudaError_t err = cudaSuccess;
    if (dst_device == src_device) {
        err = cudaSetDevice(dst_device);
        if (err != cudaSuccess) return false;
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    } else {
        err = cudaSetDevice(dst_device);
        if (err != cudaSuccess) return false;
        err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream);
    }
    return err == cudaSuccess;
}

static bool ensure_bf16_staging(DraftFeatureMirror & mirror, size_t elems) {
    if (elems <= mirror.bf16_staging_elems) return true;
    cudaError_t err = cudaSetDevice(mirror.device);
    if (err != cudaSuccess) return false;
    if (mirror.bf16_staging) {
        cudaFree(mirror.bf16_staging);
        mirror.bf16_staging = nullptr;
        mirror.bf16_staging_elems = 0;
    }
    err = cudaMalloc(&mirror.bf16_staging, elems * sizeof(uint16_t));
    if (err != cudaSuccess) return false;
    mirror.bf16_staging_elems = elems;
    return true;
}

static void draft_feature_mirror_free(DraftFeatureMirror & mirror) {
    if (mirror.bf16_staging) {
        cudaSetDevice(mirror.device);
        cudaFree(mirror.bf16_staging);
        mirror.bf16_staging = nullptr;
        mirror.bf16_staging_elems = 0;
    }
    if (mirror.buf) {
        ggml_backend_buffer_free(mirror.buf);
        mirror.buf = nullptr;
    }
    if (mirror.ctx) {
        ggml_free(mirror.ctx);
        mirror.ctx = nullptr;
    }
    mirror.target_feat = nullptr;
    mirror.device = 0;
    mirror.target_device = 0;
    mirror.cap = 0;
}

static bool draft_feature_mirror_init(DraftFeatureMirror & mirror,
                                      ggml_backend_t backend,
                                      int device,
                                      int target_device,
                                      int cap) {
    draft_feature_mirror_free(mirror);
    if (cap <= 0) return false;
    mirror.device = device;
    mirror.target_device = target_device;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    mirror.ctx = ggml_init(ip);
    if (!mirror.ctx) return false;

    const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * DFLASH27B_TARGET_HIDDEN;
    mirror.target_feat = ggml_new_tensor_2d(mirror.ctx, GGML_TYPE_F32, fc_in, cap);
    ggml_set_name(mirror.target_feat, "draft_target_feat_mirror");
    mirror.buf = ggml_backend_alloc_ctx_tensors(mirror.ctx, backend);
    if (!mirror.buf) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    const size_t bytes = (size_t)fc_in * (size_t)cap * sizeof(float);
    cudaSetDevice(device);
    cudaError_t err = cudaMemset(mirror.target_feat->data, 0, bytes);
    if (err != cudaSuccess) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    mirror.cap = cap;
    return true;
}

static bool draft_feature_mirror_can_view(const DraftFeatureMirror & mirror,
                                          int committed,
                                          int ctx_len,
                                          int & slot0) {
    if (!mirror.target_feat || mirror.cap <= 0) return false;
    if (ctx_len <= 0 || ctx_len > mirror.cap || committed < ctx_len) return false;
    const int start = committed - ctx_len;
    slot0 = start % mirror.cap;
    return slot0 + ctx_len <= mirror.cap;
}

static bool draft_feature_mirror_sync_range(const TargetCache & cache,
                                            const DraftFeatureMirror & mirror,
                                            int start_pos,
                                            int n_tokens) {
    if (!cache.target_feat || !mirror.target_feat || mirror.cap <= 0) return false;
    if (n_tokens <= 0) return true;
    if (n_tokens > mirror.cap) return false;

    const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * DFLASH27B_TARGET_HIDDEN;
    const int src_cap = cache.target_feat_cap;
    const size_t src_stride = cache.target_feat->nb[1];
    const size_t dst_stride = mirror.target_feat->nb[1];

    int done = 0;
    while (done < n_tokens) {
        const int src_slot = (start_pos + done) % src_cap;
        const int dst_slot = (start_pos + done) % mirror.cap;
        const int src_run = src_cap - src_slot;
        const int dst_run = mirror.cap - dst_slot;
        const int run = std::min(n_tokens - done, std::min(src_run, dst_run));
        const size_t elems = (size_t)run * (size_t)fc_in;
        const void * src =
            (const char *)cache.target_feat->data + (size_t)src_slot * src_stride;
        void * dst =
            (char *)mirror.target_feat->data + (size_t)dst_slot * dst_stride;
        if (mirror.device == mirror.target_device) {
            cudaSetDevice(mirror.device);
            dflash27b_launch_bf16_to_f32(src, dst, elems, nullptr);
        } else {
            DraftFeatureMirror & mutable_mirror =
                const_cast<DraftFeatureMirror &>(mirror);
            if (!ensure_bf16_staging(mutable_mirror, elems)) return false;
            if (!copy_peer_async(mirror.bf16_staging, mirror.device,
                                 src, mirror.target_device,
                                 elems * sizeof(uint16_t))) {
                return false;
            }
            cudaSetDevice(mirror.device);
            dflash27b_launch_bf16_to_f32(mirror.bf16_staging, dst, elems, nullptr);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return false;
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

static bool draft_feature_mirror_sync_tail(const TargetCache & cache,
                                           const DraftFeatureMirror & mirror,
                                           int committed) {
    if (!mirror.target_feat || committed <= 0) return true;
    const int n = std::min(committed, mirror.cap);
    return draft_feature_mirror_sync_range(cache, mirror, committed - n, n);
}

// Reset the per-call graph state (ctx + graph + tensor handles) but KEEP the
// persistent CUDA buffer in `sg.alloc` alive across steps. When the next
// build_*_step re-walks ggml_gallocr_alloc_graph on the same (or smaller)
// peak shape, gallocr reuses the existing CUDA buffer instead of doing a
// fresh cudaMalloc/cudaFree of multi-GB at every step. That's the single
// biggest per-step cost at long context — see the [timing] breakdown:
// draft_compute 21ms and verify_compute 61ms both include the alloc cycle
// inside ggml_backend_graph_compute when the buffer is fresh.
static void step_graph_free(StepGraph & sg) {
    if (sg.ctx)   { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.gf = nullptr;
    sg.inp_embed = sg.positions = sg.attn_mask = nullptr;
    sg.target_hidden_cat = sg.positions_k = nullptr;
    sg.hidden_input = nullptr;
    sg.parent_ids = nullptr;
    sg.logits = nullptr;
    sg.hidden_states = nullptr;
    sg.argmax_tokens = nullptr;
    sg.topk_indices = nullptr;
    sg.delta_captures.clear();
}

// Called at shutdown only. Releases the persistent gallocr + its CUDA
// backing buffer in addition to what step_graph_free already does.
static void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    step_graph_free(sg);
}

static void scrub_daemon_slot_state(
    TargetCache & cache,
    StepGraph & sg,
    StepGraph & draft_sg,
    StepGraph & proj_sg,
    DraftFeatureMirror & feature_mirror,
    bool & first_iter) {
    step_graph_free(sg);
    step_graph_free(draft_sg);
    step_graph_free(proj_sg);
    draft_feature_mirror_free(feature_mirror);
    reset_target_cache(cache);
    first_iter = true;
}

// Build a single-layer forward graph for layer-segmented prefill.
// Processes n_tokens tokens through one layer, reading from act_in and
// writing to act_out. Returns false on failure.
static bool build_layer_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    ggml_tensor * act_in,      // [hidden, prompt_len] full activation buffer
    ggml_tensor * act_out,     // [hidden, prompt_len] full activation buffer
    int chunk_start,           // token offset into activation buffers
    int n_tokens,
    int kv_start,
    bool with_mask,
    bool capture,
    int fa_window = 0)
{
    step_graph_free(sg);

    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;

    sg.inp_embed = ggml_view_2d(sg.ctx, act_in,
        hidden, n_tokens,
        act_in->nb[1], (size_t)chunk_start * act_in->nb[1]);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    if (is_attn) {
        sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(sg.positions, "positions");
        ggml_set_input(sg.positions);

        if (with_mask) {
            const int win_start_l = (fa_window > 0 && kv_start > fa_window)
                                        ? (kv_start - fa_window) : 0;
            const int win_len_l = kv_start + n_tokens - win_start_l;
            const int kv_pad = align_up(win_len_l, g_kq_stride_pad);
            const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
            sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.attn_mask, "attn_mask");
            ggml_set_input(sg.attn_mask);
        }
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    ggml_tensor * layer_out = dflash27b::build_qwen35_layer(
        sg.ctx, sg.gf, w, cache, layer_idx,
        sg.inp_embed, sg.positions, sg.attn_mask,
        kv_start, n_tokens, capture, fa_window);
    if (!layer_out) return false;

    ggml_tensor * out_view = ggml_view_2d(sg.ctx, act_out,
        hidden, n_tokens,
        act_out->nb[1], (size_t)chunk_start * act_out->nb[1]);
    ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, layer_out, out_view));

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// Forward declaration used only by the tree-mode build_target_step_tree.
struct DDTree;  // full def above

// Build a target verify graph configured for a DDTree-flattened tree block.
// Key differences vs build_target_step:
//   - n_tokens = 1 + tree.n_nodes (root + flat tree nodes)
//   - with_mask is always true; the caller fills a custom ancestor-only mask
//     rather than the usual causal strip
//   - A parent_ids[n_tokens] i32 input is added and wired into the delta-net
//     kernel so recurrent state is reloaded at DFS branch transitions
//   - capture_layers and capture_delta_intermediate are always on (the spec
//     loop uses per-step SSM states for rollback and target_feat for the
//     next iter's draft)
static bool build_target_step_tree(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    int fa_window = 0);   // implemented below after the regular build_target_step

static bool build_target_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    bool with_mask,
    bool capture,
    bool capture_delta_intermediate = false,
    int fa_window = 0) {
    step_graph_free(sg);

    ggml_init_params ip{};
    // ctx arena holds tensor *descriptors* only (no_alloc = true), so size
    // just needs to cover the struct count. 512 MB is plenty for the target
    // graph even with capture_delta_intermediate enabled (the 48 extra delta
    // captures add ~48 descriptors, nothing).
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        const int win_start = (fa_window > 0 && kv_start > fa_window) ? (kv_start - fa_window) : 0;
        const int win_len = kv_start + n_tokens - win_start;
        const int kv_pad = align_up(win_len, g_kq_stride_pad);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.capture_layers             = capture;
    gi.capture_delta_intermediate = capture_delta_intermediate;
    gi.fa_window                  = fa_window;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    ggml_set_output(sg.logits);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "chain_verify_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

static bool build_target_batch_probe_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_seqs,
    bool capture_layers = false) {
    step_graph_free(sg);
    if (n_seqs <= 1) return false;

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32,
                                      hidden, /*n_tokens=*/1, n_seqs);
    ggml_set_name(sg.inp_embed, "batch_probe_inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4);
    ggml_set_name(sg.positions, "batch_probe_positions");
    ggml_set_input(sg.positions);

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = nullptr;
    gi.n_tokens                   = 1;
    gi.n_seqs                     = n_seqs;
    gi.kv_start                   = kv_start;
    gi.capture_layers             = capture_layers;
    gi.capture_delta_intermediate = false;
    gi.fa_window                  = 0;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    ggml_set_output(sg.logits);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "batch_probe_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// DDTree tree-verify graph builder. Same shape as build_target_step except:
//   - n_tokens is the flat tree size (1 + tree.n_nodes)
//   - attn_mask is caller-filled (ancestor-only); we build the tensor here
//     but the values come from build_tree_mask() before compute
//   - A fresh parent_ids[n_tokens] i32 input tensor is added and wired into
//     QwenGraphInputs so build_delta_net_block can call ggml_gated_delta_net_tree
//   - capture_layers=true, capture_delta_intermediate=true (spec loop relies
//     on per-step intermediates for rollback)
static bool build_target_step_tree(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    int fa_window) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    // Use max possible mask size so gallocr shape stays fixed across steps.
    // Actual valid region is filled before compute; unused area is -inf.
    const int max_win_len = cache.max_ctx + n_tokens;
    const int kv_pad = align_up(max_win_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_name(sg.attn_mask, "attn_mask");
    ggml_set_input(sg.attn_mask);

    sg.parent_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.parent_ids, "parent_ids");
    ggml_set_input(sg.parent_ids);

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.fa_window                  = fa_window;
    gi.capture_layers             = true;
    gi.capture_delta_intermediate = true;
    gi.parent_ids                 = sg.parent_ids;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    ggml_set_output(sg.logits);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "tree_verify_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

static bool build_draft_step(
    StepGraph & sg,
    const DraftWeights & dw,
    const TargetWeights * tw,   // optional target lm_head
    ggml_backend_t backend,
    int ctx_len,
    const DraftFeatureMirror * mirror = nullptr,
    int committed = 0) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int q_len  = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int fc_in  = DFLASH27B_DRAFT_N_TARGET_LAYERS * hidden;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    int mirror_slot0 = 0;
    if (mirror && draft_feature_mirror_can_view(*mirror, committed, ctx_len, mirror_slot0)) {
        const size_t stride = mirror->target_feat->nb[1];
        sg.target_hidden_cat = ggml_view_3d(
            sg.ctx,
            mirror->target_feat,
            fc_in, ctx_len, 1,
            stride,
            stride * (size_t)ctx_len,
            (size_t)mirror_slot0 * stride);
    } else {
        sg.target_hidden_cat = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, fc_in, ctx_len, 1);
        ggml_set_input(sg.target_hidden_cat);
    }
    ggml_set_name(sg.target_hidden_cat, "target_hidden_cat");

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, q_len);
    ggml_set_name(sg.positions, "positions_q");
    ggml_set_input(sg.positions);

    sg.positions_k = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, ctx_len + q_len);
    ggml_set_name(sg.positions_k, "positions_k");
    ggml_set_input(sg.positions_k);

    if (draft_graph_needs_swa_mask(dw, ctx_len)) {
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, ctx_len + q_len, q_len);
        ggml_set_name(sg.attn_mask, "draft_swa_mask");
        ggml_set_input(sg.attn_mask);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

    DraftGraphInputs gi{};
    gi.ctx_len           = ctx_len;
    gi.noise_embed       = sg.inp_embed;
    gi.target_hidden_cat = sg.target_hidden_cat;
    gi.positions_q       = sg.positions;
    gi.positions_k       = sg.positions_k;
    gi.attn_mask         = sg.attn_mask;
    gi.lm_head           = tw ? tw->output : nullptr; // project through target.output when local
    DraftGraphOutputs go = build_draft_graph(sg.ctx, dw, gi);
    sg.hidden_states = go.hidden_states;
    sg.logits = go.logits;
    if (!sg.hidden_states) {
        std::fprintf(stderr, "draft graph missing hidden_states\n");
        return false;
    }
    if (sg.logits) {
        // GPU-side argmax: avoids 16 CPU argmaxes over 248K vocab.
        sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
        ggml_set_name(sg.argmax_tokens, "argmax_tokens");
        ggml_set_output(sg.argmax_tokens);
        ggml_build_forward_expand(sg.gf, sg.argmax_tokens);
    } else {
        ggml_set_output(sg.hidden_states);
        ggml_build_forward_expand(sg.gf, sg.hidden_states);
    }

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

static bool build_lm_head_projection_step(
    StepGraph & sg,
    const TargetWeights & w,
    ggml_backend_t backend,
    int n_tokens) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.hidden_input = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.hidden_input, "draft_hidden_for_lm_head");
    ggml_set_input(sg.hidden_input);

    sg.gf = ggml_new_graph_custom(sg.ctx, 1024, false);
    sg.logits = ggml_mul_mat(sg.ctx, w.output, sg.hidden_input);
    ggml_set_name(sg.logits, "draft_projected_logits");
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

struct TargetLayerSplitShard {
    int gpu = 0;
    int layer_begin = 0;
    int layer_end = 0;
    ggml_backend_t backend = nullptr;
    TargetWeights weights;
    TargetCache cache;
    StepGraph layer_graph;
};

struct ActivationPair {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_backend_t backend = nullptr;
    int n_tokens = 0;
};

static void activation_pair_free(ActivationPair & p) {
    if (p.buf) { ggml_backend_buffer_free(p.buf); p.buf = nullptr; }
    if (p.ctx) { ggml_free(p.ctx); p.ctx = nullptr; }
    p.a = nullptr;
    p.b = nullptr;
    p.backend = nullptr;
    p.n_tokens = 0;
}

static bool activation_pair_init(ActivationPair & p,
                                 ggml_backend_t backend,
                                 int hidden,
                                 int n_tokens) {
    activation_pair_free(p);
    if (n_tokens <= 0) return false;
    p.backend = backend;
    p.n_tokens = n_tokens;
    ggml_init_params ip{};
    ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    p.ctx = ggml_init(ip);
    if (!p.ctx) return false;
    p.a = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, hidden, n_tokens);
    p.b = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_set_name(p.a, "target_split_act_a");
    ggml_set_name(p.b, "target_split_act_b");
    p.buf = ggml_backend_alloc_ctx_tensors(p.ctx, backend);
    if (!p.buf) {
        activation_pair_free(p);
        return false;
    }
    return true;
}

static bool parse_int_list(const char * text, std::vector<int> & out) {
    out.clear();
    if (!text || !*text) return false;
    const char * p = text;
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return false;
        out.push_back((int)v);
        if (*end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return !out.empty();
}

static bool parse_float_list(const char * text, std::vector<double> & out) {
    out.clear();
    if (!text || !*text) return false;
    const char * p = text;
    while (*p) {
        char * end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p || v <= 0.0) return false;
        out.push_back(v);
        if (*end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return !out.empty();
}

static int inspect_target_layer_count(const char * target_path) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(target_path, gip);
    if (!gctx) return -1;
    int64_t id = gguf_find_key(gctx, "qwen35.block_count");
    int n_layer = id >= 0 ? (int)gguf_get_val_u32(gctx, id) : -1;
    gguf_free(gctx);
    if (meta_ctx) ggml_free(meta_ctx);
    return n_layer;
}

static std::vector<std::pair<int, int>> compute_layer_ranges(
        int n_layer,
        int n_shards,
        const std::vector<double> & weights) {
    std::vector<std::pair<int, int>> ranges;
    if (n_layer <= 0 || n_shards <= 0 || n_shards > n_layer) return ranges;
    std::vector<double> w = weights;
    if (w.empty()) w.assign((size_t)n_shards, 1.0);
    if ((int)w.size() != n_shards) return ranges;
    double sum = 0.0;
    for (double v : w) sum += v;
    if (sum <= 0.0) return ranges;
    ranges.reserve((size_t)n_shards);
    int begin = 0;
    double accum = 0.0;
    for (int i = 0; i < n_shards; i++) {
        accum += w[i];
        int end = (i == n_shards - 1)
            ? n_layer
            : (int)std::llround((accum / sum) * n_layer);
        const int min_end = begin + 1;
        const int max_end = n_layer - (n_shards - i - 1);
        end = std::max(min_end, std::min(max_end, end));
        ranges.push_back({begin, end});
        begin = end;
    }
    return ranges;
}

static TargetLayerSplitShard * find_target_shard(
        std::vector<TargetLayerSplitShard> & shards,
        int layer_idx) {
    for (auto & shard : shards) {
        if (layer_idx >= shard.layer_begin && layer_idx < shard.layer_end) {
            return &shard;
        }
    }
    return nullptr;
}

static int target_capture_index(const TargetWeights & w, int layer_idx) {
    for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
        if (w.capture_layer_ids[k] == layer_idx) return k;
    }
    return -1;
}

static bool copy_capture_slice_to_draft_ring(
        DraftFeatureMirror & feature_ring,
        int capture_idx,
        const ggml_tensor * act_out,
        int src_device,
        int chunk_start,
        int start_pos,
        int n_tokens) {
    if (!feature_ring.target_feat || capture_idx < 0 || n_tokens <= 0) return true;
    if (feature_ring.cap <= 0) return false;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const size_t dst_stride = feature_ring.target_feat->nb[1];
    const size_t src_stride = act_out->nb[1];
    const size_t row_bytes = (size_t)hidden * sizeof(float);
    for (int i = 0; i < n_tokens; i++) {
        const int slot = (start_pos + i) % feature_ring.cap;
        const void * src = (const char *)act_out->data +
            (size_t)(chunk_start + i) * src_stride;
        void * dst = (char *)feature_ring.target_feat->data +
            (size_t)slot * dst_stride +
            (size_t)capture_idx * (size_t)hidden * sizeof(float);
        if (!copy_peer_async(dst, feature_ring.device, src, src_device, row_bytes)) {
            return false;
        }
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

static bool copy_feature_ring_range_to_tensor(
        const DraftFeatureMirror & feature_ring,
        ggml_tensor * dst,
        int start_pos,
        int n_tokens) {
    if (!feature_ring.target_feat || !dst || feature_ring.cap <= 0) return false;
    if (n_tokens <= 0 || n_tokens > feature_ring.cap) return false;

    const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * DFLASH27B_TARGET_HIDDEN;
    const size_t row_bytes = (size_t)fc_in * sizeof(float);
    const size_t src_stride = feature_ring.target_feat->nb[1];
    const size_t dst_stride = dst->nb[1];
    int done = 0;
    while (done < n_tokens) {
        const int slot = (start_pos + done) % feature_ring.cap;
        const int run = std::min(n_tokens - done, feature_ring.cap - slot);
        const char * src_base =
            (const char *)feature_ring.target_feat->data + (size_t)slot * src_stride;
        char * dst_base = (char *)dst->data + (size_t)done * dst_stride;
        if (src_stride == row_bytes && dst_stride == row_bytes) {
            if (!copy_peer_async(dst_base, feature_ring.device,
                                 src_base, feature_ring.device,
                                 row_bytes * (size_t)run)) {
                return false;
            }
        } else {
            for (int i = 0; i < run; i++) {
                if (!copy_peer_async(dst_base + (size_t)i * dst_stride,
                                     feature_ring.device,
                                     src_base + (size_t)i * src_stride,
                                     feature_ring.device,
                                     row_bytes)) {
                    return false;
                }
            }
        }
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

static bool compute_target_split_argmax(
        StepGraph & sg,
        const TargetWeights & w,
        ggml_backend_t backend,
        ggml_tensor * act,
        int token_offset,
        int n_tokens,
        int hidden,
        int vocab,
        std::vector<int32_t> & argmax_out) {
    step_graph_free(sg);
    ggml_init_params ip{};
    ip.mem_size = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    ggml_tensor * act_view = ggml_view_2d(
        sg.ctx, act, hidden, n_tokens, act->nb[1],
        (size_t)token_offset * act->nb[1]);
    ggml_tensor * normed = ggml_rms_norm(sg.ctx, act_view, DFLASH27B_RMS_EPS);
    normed = ggml_mul(sg.ctx, normed, w.out_norm);
    ggml_tensor * logits = ggml_mul_mat(sg.ctx, w.output, normed);
    ggml_set_name(logits, "target_split_logits");
    sg.logits = logits;
    sg.argmax_tokens = ggml_argmax(sg.ctx, logits);
    ggml_set_name(sg.argmax_tokens, "target_split_argmax");
    ggml_set_output(sg.argmax_tokens);
    sg.gf = ggml_new_graph_custom(sg.ctx, 1024, false);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);
    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) return false;
    auto st = ggml_backend_graph_compute(backend, sg.gf);
    if (st != GGML_STATUS_SUCCESS) return false;
    (void)vocab;
    argmax_out.assign((size_t)n_tokens, 0);
    ggml_backend_tensor_get(sg.argmax_tokens, argmax_out.data(), 0,
                            sizeof(int32_t) * (size_t)n_tokens);
    return true;
}

static bool run_target_layer_split_forward(
        std::vector<TargetLayerSplitShard> & shards,
        const TargetWeights & embed_source,
        const std::vector<int32_t> & tokens,
        int base_pos,
        int ubatch,
        int & last_tok,
        DraftFeatureMirror * feature_ring = nullptr,
        std::vector<int32_t> * argmax_out = nullptr,
        std::vector<float> * logits_out = nullptr) {
    if (shards.empty() || tokens.empty()) return false;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab = DFLASH27B_TARGET_VOCAB;
    const int n_tokens_total = (int)tokens.size();
    ubatch = std::max(1, ubatch);

    ActivationPair acts;
    if (!activation_pair_init(acts, shards.front().backend, hidden, n_tokens_total)) {
        std::fprintf(stderr, "target-split activation alloc failed on gpu %d\n", shards.front().gpu);
        return false;
    }
    ggml_tensor * act_in = acts.a;
    ggml_tensor * act_out = acts.b;

    {
        const int EMBED_BATCH = 4096;
        std::vector<float> emb_buf((size_t)hidden * std::min(EMBED_BATCH, n_tokens_total));
        for (int i = 0; i < n_tokens_total; i += EMBED_BATCH) {
            const int n = std::min(EMBED_BATCH, n_tokens_total - i);
            if ((int)emb_buf.size() < hidden * n) emb_buf.resize((size_t)hidden * n);
            if (!embed_source.embedder.embed(tokens.data() + i, n, emb_buf.data())) {
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_tensor_set(act_in, emb_buf.data(),
                                    (size_t)i * act_in->nb[1],
                                    sizeof(float) * (size_t)hidden * n);
        }
    }

    TargetLayerSplitShard * current_shard = &shards.front();
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos_buf;
    for (int il = 0; il < embed_source.n_layer; il++) {
        TargetLayerSplitShard * shard = find_target_shard(shards, il);
        if (!shard) {
            std::fprintf(stderr, "target-split missing owner for layer %d\n", il);
            activation_pair_free(acts);
            return false;
        }
        if (shard != current_shard) {
            ActivationPair next_acts;
            if (!activation_pair_init(next_acts, shard->backend, hidden, n_tokens_total)) {
                std::fprintf(stderr, "target-split activation alloc failed on gpu %d\n", shard->gpu);
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_synchronize(current_shard->backend);
            ggml_backend_tensor_copy(act_in, next_acts.a);
            ggml_backend_synchronize(shard->backend);
            activation_pair_free(acts);
            acts = next_acts;
            act_in = acts.a;
            act_out = acts.b;
            current_shard = shard;
        }

        const bool is_attn = (((il + 1) % embed_source.full_attention_interval) == 0);
        const int capture_idx = target_capture_index(embed_source, il);
        for (int start = 0; start < n_tokens_total; start += ubatch) {
            const int n = std::min(ubatch, n_tokens_total - start);
            const int kv_start = base_pos + start;
            const int kv_len = kv_start + n;
            const bool with_mask = (g_kq_stride_pad > KQ_MASK_PAD) || (n > 1);
            if (!build_layer_step(shard->layer_graph, shard->weights, shard->cache,
                                  shard->backend, il, act_in, act_out,
                                  start, n, kv_start, with_mask,
                                  /*capture=*/false, g_fa_window)) {
                std::fprintf(stderr, "target-split build layer=%d @%d gpu=%d\n",
                             il, start, shard->gpu);
                activation_pair_free(acts);
                return false;
            }
            if (is_attn && shard->layer_graph.positions) {
                pos_buf.assign((size_t)4 * n, 0);
                for (int i = 0; i < n; i++) {
                    const int p = kv_start + i;
                    pos_buf[0 * n + i] = p;
                    pos_buf[1 * n + i] = p;
                    pos_buf[2 * n + i] = p;
                    pos_buf[3 * n + i] = 0;
                }
                ggml_backend_tensor_set(shard->layer_graph.positions, pos_buf.data(), 0,
                                        sizeof(int32_t) * pos_buf.size());
            }
            if (is_attn && with_mask && shard->layer_graph.attn_mask) {
                const int win_start_l = (g_fa_window > 0 && kv_start > g_fa_window)
                                            ? (kv_start - g_fa_window) : 0;
                const int win_len_l = kv_len - win_start_l;
                build_causal_mask(mask_buf, win_len_l, n, kv_start, win_start_l);
                ggml_backend_tensor_set(shard->layer_graph.attn_mask, mask_buf.data(), 0,
                                        sizeof(uint16_t) * mask_buf.size());
            }
            auto st = ggml_backend_graph_compute(shard->backend, shard->layer_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "target-split compute layer=%d @%d gpu=%d status=%d\n",
                             il, start, shard->gpu, (int)st);
                activation_pair_free(acts);
                return false;
            }
            if (feature_ring && capture_idx >= 0) {
                if (!copy_capture_slice_to_draft_ring(*feature_ring, capture_idx,
                                                      act_out, shard->gpu,
                                                      start, base_pos + start, n)) {
                    std::fprintf(stderr,
                                 "target-split capture copy failed layer=%d capture=%d gpu=%d\n",
                                 il, capture_idx, shard->gpu);
                    activation_pair_free(acts);
                    return false;
                }
            }
        }
        std::swap(act_in, act_out);
    }

    StepGraph final_sg;
    std::vector<int32_t> argmax_tokens;
    TargetLayerSplitShard & last_shard = shards.back();
    const bool need_all_argmax = argmax_out != nullptr;
    const int argmax_offset = need_all_argmax ? 0 : (n_tokens_total - 1);
    const int argmax_count = need_all_argmax ? n_tokens_total : 1;
    const bool ok = compute_target_split_argmax(
        final_sg, last_shard.weights, last_shard.backend, act_in,
        argmax_offset, argmax_count, hidden, vocab, argmax_tokens);
    step_graph_destroy(final_sg);
    activation_pair_free(acts);
    if (!ok) return false;
    last_tok = argmax_tokens.empty() ? -1 : argmax_tokens.back();
    if (argmax_out) *argmax_out = std::move(argmax_tokens);
    if (logits_out) logits_out->clear();
    return true;
}

static void free_target_layer_split_shards(std::vector<TargetLayerSplitShard> & shards) {
    for (auto & shard : shards) {
        step_graph_destroy(shard.layer_graph);
        free_target_cache(shard.cache);
        free_target_weights(shard.weights);
        if (shard.backend) {
            ggml_backend_free(shard.backend);
            shard.backend = nullptr;
        }
    }
    shards.clear();
}

static bool run_target_layer_split_dflash_decode(
        std::vector<TargetLayerSplitShard> & shards,
        DraftWeights & draft_weights,
        ggml_backend_t draft_backend,
        int draft_gpu,
        DraftFeatureMirror & feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int last_tok,
        const char * out_path) {
    if (shards.empty() || !feature_ring.target_feat) return false;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab = DFLASH27B_TARGET_VOCAB;
    const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int output_gpu = shards.back().gpu;
    ggml_backend_t output_backend = shards.back().backend;

    StepGraph draft_sg;
    StepGraph proj_sg;
    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<int32_t> out_all = prompt;
    int committed = (int)prompt.size();
    int n_generated = 0;
    int n_draft_steps = 0;
    int n_accept_sum = 0;

    auto sync_all = [&]() {
        for (auto & shard : shards) ggml_backend_synchronize(shard.backend);
        ggml_backend_synchronize(draft_backend);
    };

    auto t_dec0 = std::chrono::steady_clock::now();
    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = DFLASH27B_DRAFT_MASK_TOKEN_ID;
        if (!shards.front().weights.embedder.embed(noise_ids.data(), q_len,
                                                   noise_embed.data())) {
            std::fprintf(stderr, "target-split-dflash noise embed failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }

        constexpr int DRAFT_CTX_MAX = 2048;
        const int draft_ctx = std::min(committed, std::min(feature_ring.cap,
            std::max(DRAFT_CTX_MAX, g_draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_ring, committed, draft_ctx, mirror_slot0);
        if (!build_draft_step(draft_sg, draft_weights, nullptr, draft_backend,
                              draft_ctx, use_mirror_view ? &feature_ring : nullptr,
                              committed)) {
            std::fprintf(stderr, "target-split-dflash draft build failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_ring, draft_sg.target_hidden_cat,
                                               draft_start, draft_ctx)) {
            std::fprintf(stderr, "target-split-dflash draft feature copy failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());
        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * pos_k.size());
        auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "target-split-dflash draft compute %d\n", (int)st);
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }

        if (!proj_sg.gf || !proj_sg.hidden_input || proj_sg.hidden_input->ne[1] != q_len) {
            if (!build_lm_head_projection_step(proj_sg, shards.back().weights,
                                               output_backend, q_len)) {
                std::fprintf(stderr, "target-split-dflash projection build failed\n");
                step_graph_destroy(draft_sg);
                step_graph_destroy(proj_sg);
                return false;
            }
        }
        const size_t hidden_bytes = ggml_nbytes(draft_sg.hidden_states);
        if (!copy_peer_async(proj_sg.hidden_input->data, output_gpu,
                             draft_sg.hidden_states->data, draft_gpu,
                             hidden_bytes)) {
            std::fprintf(stderr, "target-split-dflash hidden peer copy failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        cudaSetDevice(output_gpu);
        cudaDeviceSynchronize();
        st = ggml_backend_graph_compute(output_backend, proj_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "target-split-dflash projection compute %d\n", (int)st);
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        ggml_backend_tensor_get(proj_sg.argmax_tokens, draft_tok.data(), 0,
                                sizeof(int32_t) * q_len);
        draft_tok[0] = last_tok;

        for (auto & shard : shards) snapshot_ssm_state(shard.cache);

        int verify_last_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            draft_tok, committed, q_len,
                                            verify_last_tok, &feature_ring,
                                            &target_tok)) {
            std::fprintf(stderr, "target-split-dflash verify failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }

        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        for (auto & shard : shards) restore_ssm_state(shard.cache);

        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }
        int replay_last_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            replay_tok, committed, commit_n,
                                            replay_last_tok, &feature_ring)) {
            std::fprintf(stderr, "target-split-dflash replay failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        last_tok = replay_last_tok;

        bool hit_eos = false;
        for (int i = 0; i < commit_n; i++) {
            out_all.push_back(replay_tok[i]);
            if (IS_EOS_TOK(replay_tok[i], shards.front().weights)) hit_eos = true;
        }
        committed += commit_n;
        n_generated += commit_n;
        n_accept_sum += std::min(accept_n, commit_n);
        n_draft_steps++;
        if (hit_eos) break;
    }
    sync_all();
    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_draft_steps * q_len);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    std::printf("[target-split-dflash] decode tokens=%d time=%.3f s speed=%.2f tok/s\n",
                n_generated, decode_s, n_generated > 0 ? n_generated / decode_s : 0.0);
    std::printf("[target-split-dflash] %d draft steps, accepted=%d/%d (%.1f%%), avg commit/step=%.2f\n",
                n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);
    if (out_path) write_int32_file(out_path, out_all);

    step_graph_destroy(draft_sg);
    step_graph_destroy(proj_sg);
    return true;
}

static int run_target_layer_split_harness(
        const char * target_path,
        const char * draft_path,
        const char * prompt_path,
        int n_gen,
        const char * out_path,
        const std::vector<int> & target_gpus,
        const std::vector<double> & split_weights,
        int draft_gpu,
        bool load_draft,
        bool run_draft_smoke,
        bool run_dflash,
        int max_ctx,
        int max_verify_tokens) {
    if (!prompt_path || !out_path) {
        std::fprintf(stderr, "target layer split requires prompt/n_gen/out positional args\n");
        return 2;
    }
    const int n_layer = inspect_target_layer_count(target_path);
    if (n_layer <= 0) {
        std::fprintf(stderr, "target-split could not read qwen35.block_count\n");
        return 1;
    }
    const auto ranges = compute_layer_ranges(n_layer, (int)target_gpus.size(), split_weights);
    if ((int)ranges.size() != (int)target_gpus.size()) {
        std::fprintf(stderr, "bad --target-layer-split for %zu target GPUs and %d layers\n",
                     target_gpus.size(), n_layer);
        return 2;
    }
    std::vector<TargetLayerSplitShard> shards;
    shards.resize(target_gpus.size());
    for (size_t i = 0; i < target_gpus.size(); i++) {
        shards[i].gpu = target_gpus[i];
        shards[i].layer_begin = ranges[i].first;
        shards[i].layer_end = ranges[i].second;
    }
    for (auto & shard : shards) {
        shard.backend = ggml_backend_cuda_init(shard.gpu);
        if (!shard.backend) {
            std::fprintf(stderr, "target-split cuda init failed for gpu %d\n", shard.gpu);
            free_target_layer_split_shards(shards);
            return 1;
        }
    }
    for (size_t i = 0; i < target_gpus.size(); i++) {
        for (size_t j = i + 1; j < target_gpus.size(); j++) {
            if (!enable_peer_access_pair(target_gpus[i], target_gpus[j])) {
                std::fprintf(stderr,
                             "warning: CUDA peer access not fully enabled for target gpus %d,%d\n",
                             target_gpus[i], target_gpus[j]);
            }
        }
    }
    for (auto & shard : shards) {
        TargetLoadPlan plan;
        plan.layer_begin = shard.layer_begin;
        plan.layer_end = shard.layer_end;
        plan.load_output = (&shard == &shards.back());
        if (!load_target_gguf_partial(target_path, shard.backend, plan, shard.weights)) {
            std::fprintf(stderr, "target-split load gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] gpu=%d layers=[%d,%d) %s\n",
                    shard.gpu, shard.layer_begin, shard.layer_end,
                    dflash27b_last_error());
        const bool allocate_target_feat = false;
        if (!create_target_cache_partial(shard.weights, max_ctx, max_verify_tokens,
                                         shard.backend, shard.cache,
                                         /*prefill_only=*/!run_dflash,
                                         shard.layer_begin, shard.layer_end,
                                         allocate_target_feat)) {
            std::fprintf(stderr, "target-split cache gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            free_target_layer_split_shards(shards);
            return 1;
        }
    }

    ggml_backend_t draft_backend = nullptr;
    DraftWeights draft_weights;
    DraftFeatureMirror feature_ring;
    bool draft_backend_owned = false;
    if (load_draft) {
        for (auto & shard : shards) {
            if (shard.gpu == draft_gpu) {
                draft_backend = shard.backend;
                break;
            }
        }
        if (!draft_backend) {
            draft_backend = ggml_backend_cuda_init(draft_gpu);
            if (!draft_backend) {
                std::fprintf(stderr, "target-split draft cuda init failed for gpu %d\n", draft_gpu);
                free_target_layer_split_shards(shards);
                return 1;
            }
            draft_backend_owned = true;
        }
        std::string dp(draft_path);
        bool draft_ok = false;
        if (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf") {
            draft_ok = load_draft_gguf(draft_path, draft_backend, draft_weights);
        } else {
            draft_ok = load_draft_safetensors(draft_path, draft_backend, draft_weights);
        }
        if (!draft_ok) {
            std::fprintf(stderr, "target-split draft load gpu=%d: %s\n",
                         draft_gpu, dflash27b_last_error());
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] draft loaded on gpu=%d format=%s\n",
                    draft_gpu,
                    (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
                        ? "gguf" : "safetensors");
        if (g_draft_swa_window > 0) {
            draft_weights.swa_window = g_draft_swa_window;
            for (int il = 0; il < draft_weights.n_layer - 1; il++) {
                draft_weights.layers[il].is_swa = true;
            }
            std::printf("[target-split] draft SWA layers: %d/%d (window=%d)\n",
                        draft_weights.n_layer - 1, draft_weights.n_layer,
                        draft_weights.swa_window);
        }
        const int cap = std::min(max_ctx, 4096);
        if (!draft_feature_mirror_init(feature_ring, draft_backend,
                                       draft_gpu, draft_gpu, cap)) {
            std::fprintf(stderr, "target-split feature ring init failed on gpu=%d\n", draft_gpu);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] draft feature ring cap=%d gpu=%d\n", cap, draft_gpu);
    }

    auto prompt = read_int32_file(prompt_path);
    if (prompt.empty()) {
        std::fprintf(stderr, "target-split empty prompt\n");
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }
    if ((int)prompt.size() + n_gen + 1 > max_ctx) {
        std::fprintf(stderr, "target-split prompt (%zu) + gen (%d) exceeds max_ctx (%d)\n",
                     prompt.size(), n_gen, max_ctx);
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }

    int ubatch = (prompt.size() > 2048) ? 384 : 16;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        ubatch = std::max(1, std::atoi(s));
    }
    std::printf("[target-split] n_gpus=%zu n_layer=%d ubatch=%d max_ctx=%d\n",
                target_gpus.size(), n_layer, ubatch, max_ctx);

    int last_tok = -1;
    auto t_pf0 = std::chrono::steady_clock::now();
    if (!run_target_layer_split_forward(shards, shards.front().weights,
                                        prompt, 0, ubatch, last_tok,
                                        load_draft ? &feature_ring : nullptr)) {
        std::fprintf(stderr, "target-split prefill failed\n");
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();
    std::printf("[target-split] prefill tokens=%zu time=%.3f s speed=%.2f tok/s last_tok=%d\n",
                prompt.size(), prefill_s, prompt.size() / prefill_s, last_tok);

    if (run_draft_smoke) {
        const int hidden = DFLASH27B_TARGET_HIDDEN;
        const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
        const int draft_ctx = std::min((int)prompt.size(), feature_ring.cap);
        const int draft_start = (int)prompt.size() - draft_ctx;
        StepGraph draft_sg;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_ring, (int)prompt.size(),
                                          draft_ctx, mirror_slot0);
        if (!build_draft_step(draft_sg, draft_weights, nullptr, draft_backend,
                              draft_ctx, use_mirror_view ? &feature_ring : nullptr,
                              (int)prompt.size())) {
            std::fprintf(stderr, "target-split draft smoke build failed\n");
            step_graph_destroy(draft_sg);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_ring,
                                               draft_sg.target_hidden_cat,
                                               draft_start, draft_ctx)) {
            std::fprintf(stderr, "target-split draft smoke feature copy failed\n");
            step_graph_destroy(draft_sg);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::vector<int32_t> noise_ids(q_len, DFLASH27B_DRAFT_MASK_TOKEN_ID);
        noise_ids[0] = last_tok;
        std::vector<float> noise_embed((size_t)hidden * q_len);
        if (!shards.front().weights.embedder.embed(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "target-split draft smoke embed failed\n");
            step_graph_destroy(draft_sg);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());
        std::vector<int32_t> pos_q(q_len), pos_k(draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * pos_k.size());
        auto t_ds0 = std::chrono::steady_clock::now();
        auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
        auto t_ds1 = std::chrono::steady_clock::now();
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "target-split draft smoke compute failed status=%d\n", (int)st);
            step_graph_destroy(draft_sg);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] draft smoke ctx=%d q=%d time=%.3f ms\n",
                    draft_ctx, q_len,
                    std::chrono::duration<double, std::milli>(t_ds1 - t_ds0).count());
        step_graph_destroy(draft_sg);
    }

    if (run_dflash) {
        const bool ok = run_target_layer_split_dflash_decode(
            shards, draft_weights, draft_backend, draft_gpu, feature_ring,
            prompt, n_gen, last_tok, out_path);
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return ok ? 0 : 1;
    }

    std::vector<int32_t> out_all = prompt;
    auto t_dec0 = std::chrono::steady_clock::now();
    int generated = 0;
    for (; generated < n_gen; generated++) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            one, (int)out_all.size(), 1, next_tok,
                                            load_draft ? &feature_ring : nullptr)) {
            std::fprintf(stderr, "target-split decode failed at %d\n", generated);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        out_all.push_back(last_tok);
        if (IS_EOS_TOK(last_tok, shards.front().weights)) {
            generated++;
            break;
        }
        last_tok = next_tok;
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    std::printf("[target-split] decode tokens=%d time=%.3f s speed=%.2f tok/s\n",
                generated, decode_s, generated > 0 ? generated / decode_s : 0.0);
    if (out_path) write_int32_file(out_path, out_all);
    draft_feature_mirror_free(feature_ring);
    free_draft_weights(draft_weights);
    if (draft_backend_owned) ggml_backend_free(draft_backend);
    free_target_layer_split_shards(shards);
    return 0;
}

// ─── Main ─────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--test-scheduler-buckets") == 0) {
        return run_scheduler_bucket_selftest();
    }
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <target.gguf> <draft.safetensors|draft.gguf> [<prompt_ids.bin> <n_gen> <out_ids.bin>] [--daemon] [--synthetic-prompt-tokens N --n-gen N --out out.bin] [-ctk <type>] [-ctv <type>] [--target-gpu N] [--draft-gpu N] [--target-cache-slots N] [--stream-tagged] ...\n       %s --test-scheduler-buckets\n", argv[0], argv[0]);
        return 2;
    }
    // TurboQuant FA kernel requires kv_len aligned to FATTN_KQ_STRIDE=256.
    // Bump the mask stride accordingly so the mask dim matches the kv view.
    if (const char * s = std::getenv("DFLASH27B_KV_TBQ")) {
        if (std::atoi(s) != 0) g_kq_stride_pad = 256;
    }
    if (const char * s = std::getenv("DFLASH27B_KV_TQ3")) {
        if (std::atoi(s) != 0) g_kq_stride_pad = 256;
    }
    if (const char * s = std::getenv("DFLASH27B_FA_WINDOW")) {
        g_fa_window = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH27B_DRAFT_SWA")) {
        g_draft_swa_window = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH27B_DRAFT_CTX_MAX")) {
        g_draft_ctx_max = std::max(0, std::atoi(s));
    }
    const char * target_path = argv[1];
    const char * draft_path  = argv[2];
    const char * prompt_path = (argc >= 6 && argv[3][0] != '-') ? argv[3] : nullptr;
    int          n_gen       = (argc >= 6 && argv[3][0] != '-') ? std::atoi(argv[4]) : 0;
    const char * out_path    = (argc >= 6 && argv[3][0] != '-') ? argv[5] : nullptr;
    std::string out_path_storage;
    // --seq-verify: run the target verify as q_len independent single-token
    // decodes instead of one batched forward with a causal mask. Isolates
    // the correctness-of-batched-verify hypothesis from z-lab issue #57.
    //
    // --fast-rollback: use per-step SSM intermediate-state capture (kernel mod
    // in ggml_gated_delta_net) to roll back state after verify without the
    // replay forward pass. Implicit-bonus variant: commit only accept_n tokens
    // per step, let next iter's draft pick up the "bonus" via last_tok.
    //
    // --ddtree [--ddtree-budget=B]: use DDTree-style tree-structured verify on
    // top of the fast-rollback path. Ported from liranringel/ddtree.py with
    // our tree-aware gated_delta_net kernel handling the DeltaNet/SSM tree
    // recurrence via parent_ids (Qwen3.5 hybrid support beyond the original
    // paper's pure-attention Qwen3 experiments). Default budget = 64.
    bool  seq_verify    = false;
    bool  fast_rollback = false;
    bool  ddtree_mode   = false;
    int   ddtree_budget = 64;
    float ddtree_temp   = 1.0f;   // softmax temperature for top-K extract
    bool  ddtree_chain_seed = true;  // pre-seed full chain (vs paper's pure best-first)
    bool  profile_scaling = false;  // microbench: time target forward at varying N
    bool  test_window_mode = false;
    bool  draft_feature_mirror = false;
    int   target_cache_slots = 1;  // daemon-only: independent TargetCache states sharing weights
    int   target_gpu = 0;
    int   draft_gpu = 0;
    bool  stream_tagged = false;
    int   synthetic_prompt_tokens = 0;
    int   synthetic_prompt_token = 220;
    if (const char * s = std::getenv("DFLASH_TARGET_GPU")) {
        target_gpu = std::max(0, std::atoi(s));
    }
    if (const char * s = std::getenv("DFLASH_DRAFT_GPU")) {
        draft_gpu = std::max(0, std::atoi(s));
    }
    StreamFd stream_fd  = kInvalidStreamFd;  // write committed tokens as int32 LE
    bool  daemon_mode   = false;
    for (int i = 3; i < argc; i++) {
        if      (std::strcmp(argv[i], "--daemon") == 0)        daemon_mode = true;
        else if (std::strcmp(argv[i], "--seq-verify") == 0)    seq_verify = true;
        else if (std::strcmp(argv[i], "--fast-rollback") == 0) fast_rollback = true;
        else if (std::strcmp(argv[i], "--ddtree") == 0)        { ddtree_mode = true; fast_rollback = true; }
        else if (std::strncmp(argv[i], "--ddtree-budget=", 16) == 0) {
            ddtree_budget = std::atoi(argv[i] + 16);
            if (ddtree_budget <= 0) ddtree_budget = 64;
        }
        else if (std::strncmp(argv[i], "--ddtree-temp=", 14) == 0) {
            ddtree_temp = (float)std::atof(argv[i] + 14);
            if (ddtree_temp <= 0.0f) ddtree_temp = 1.0f;
        }
        else if (std::strcmp(argv[i], "--ddtree-no-chain-seed") == 0) {
            ddtree_chain_seed = false;
        }
        else if (std::strcmp(argv[i], "--test-window") == 0)      { test_window_mode = true; }
        else if (std::strcmp(argv[i], "--draft-feature-mirror") == 0) {
            draft_feature_mirror = true;
        }
        else if (std::strncmp(argv[i], "--target-cache-slots=", 21) == 0) {
            target_cache_slots = std::atoi(argv[i] + 21);
        }
        else if (std::strncmp(argv[i], "--cache-slots=", 14) == 0) {
            target_cache_slots = std::atoi(argv[i] + 14);
        }
        else if (std::strcmp(argv[i], "--target-cache-slots") == 0 ||
                 std::strcmp(argv[i], "--cache-slots") == 0) {
            if (i + 1 < argc) target_cache_slots = std::atoi(argv[++i]);
        }
        else if (std::strncmp(argv[i], "--target-gpu=", 13) == 0) {
            target_gpu = std::max(0, std::atoi(argv[i] + 13));
        }
        else if (std::strcmp(argv[i], "--target-gpu") == 0) {
            if (i + 1 < argc) target_gpu = std::max(0, std::atoi(argv[++i]));
        }
        else if (std::strncmp(argv[i], "--draft-gpu=", 12) == 0) {
            draft_gpu = std::max(0, std::atoi(argv[i] + 12));
        }
        else if (std::strcmp(argv[i], "--draft-gpu") == 0) {
            if (i + 1 < argc) draft_gpu = std::max(0, std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--profile-scaling") == 0) {
            profile_scaling = true;
        }
        else if (std::strncmp(argv[i], "--synthetic-prompt-tokens=", 26) == 0) {
            synthetic_prompt_tokens = std::max(0, std::atoi(argv[i] + 26));
        }
        else if (std::strcmp(argv[i], "--synthetic-prompt-tokens") == 0) {
            if (i + 1 < argc) synthetic_prompt_tokens = std::max(0, std::atoi(argv[++i]));
        }
        else if (std::strncmp(argv[i], "--synthetic-token=", 18) == 0) {
            synthetic_prompt_token = std::max(0, std::atoi(argv[i] + 18));
        }
        else if (std::strcmp(argv[i], "--synthetic-token") == 0) {
            if (i + 1 < argc) synthetic_prompt_token = std::max(0, std::atoi(argv[++i]));
        }
        else if (std::strncmp(argv[i], "--n-gen=", 8) == 0) {
            n_gen = std::atoi(argv[i] + 8);
        }
        else if (std::strcmp(argv[i], "--n-gen") == 0) {
            if (i + 1 < argc) n_gen = std::atoi(argv[++i]);
        }
        else if (std::strncmp(argv[i], "--out=", 6) == 0) {
            out_path_storage = argv[i] + 6;
            out_path = out_path_storage.c_str();
        }
        else if (std::strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out_path_storage = argv[++i];
                out_path = out_path_storage.c_str();
            }
        }
        else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = parse_stream_fd(argv[i] + 12);
        }
        else if (std::strcmp(argv[i], "--stream-tagged") == 0 ||
                 std::strcmp(argv[i], "--tagged-stream") == 0) {
            stream_tagged = true;
        }
        else if (std::strncmp(argv[i], "--max-ctx=", 10) == 0) {
            g_max_ctx_override = std::atoi(argv[i] + 10);
        }
        // KV cache type flags (mirror llama-cli -ctk / -ctv).
        // Set the env var before resolve_kv_types() reads it inside create_target_cache.
        else if (std::strcmp(argv[i], "--cache-type-k") == 0 || std::strcmp(argv[i], "-ctk") == 0) {
            if (i + 1 < argc) setenv("DFLASH27B_KV_K", argv[++i], 1);
        }
        else if (std::strncmp(argv[i], "--cache-type-k=", 15) == 0) {
            setenv("DFLASH27B_KV_K", argv[i] + 15, 1);
        }
        else if (std::strncmp(argv[i], "-ctk=", 5) == 0) {
            setenv("DFLASH27B_KV_K", argv[i] + 5, 1);
        }
        else if (std::strcmp(argv[i], "--cache-type-v") == 0 || std::strcmp(argv[i], "-ctv") == 0) {
            if (i + 1 < argc) setenv("DFLASH27B_KV_V", argv[++i], 1);
        }
        else if (std::strncmp(argv[i], "--cache-type-v=", 15) == 0) {
            setenv("DFLASH27B_KV_V", argv[i] + 15, 1);
        }
        else if (std::strncmp(argv[i], "-ctv=", 5) == 0) {
            setenv("DFLASH27B_KV_V", argv[i] + 5, 1);
        }
        else if (std::strncmp(argv[i], "--draft-swa=", 12) == 0) {
            g_draft_swa_window = std::max(0, std::atoi(argv[i] + 12));
        }
        else if (std::strncmp(argv[i], "--draft-ctx-max=", 16) == 0) {
            g_draft_ctx_max = std::max(0, std::atoi(argv[i] + 16));
        }
    }
    target_cache_slots = std::max(1, std::min(target_cache_slots, 16));

    if (!daemon_mode && !test_window_mode &&
        (((!prompt_path && synthetic_prompt_tokens <= 0) || !out_path || n_gen <= 0))) {
        std::fprintf(stderr, "Missing positional arguments for non-daemon mode.\n");
        return 2;
    }

    int current_stream_request_id = 0;
    auto stream_write_i32 = [&](const int32_t * data, size_t count) {
        if (stream_fd == kInvalidStreamFd) return;
#if defined(_WIN32)
        DWORD written;
        WriteFile(reinterpret_cast<HANDLE>(stream_fd), data,
                  (DWORD)(sizeof(int32_t) * count), &written, nullptr);
#else
        ssize_t n = ::write(stream_fd, data, sizeof(int32_t) * count);
        (void)n;
#endif
    };

    // Helper: write a committed token to the stream fd immediately.
    // Default protocol is legacy int32 tokens. With --stream-tagged each token
    // is framed as [-2, request_id, token] so future interleaving can demux.
    // Caller invokes after every out_all.push_back(tok) when stream_fd >= 0.
    // On Windows stream_fd holds a Win32 HANDLE value (passed via msvcrt.get_osfhandle).
    auto stream_emit = [&](int32_t tok) {
        if (stream_fd == kInvalidStreamFd) return;
        if (stream_tagged) {
            int32_t frame[3] = {-2, current_stream_request_id, tok};
            stream_write_i32(frame, 3);
        } else {
            int32_t v = tok;
            stream_write_i32(&v, 1);
        }
    };
    auto stream_emit_for = [&](int request_id, int32_t tok) {
        if (stream_fd == kInvalidStreamFd) return;
        if (stream_tagged) {
            int32_t frame[3] = {-2, request_id, tok};
            stream_write_i32(frame, 3);
        } else {
            int32_t v = tok;
            stream_write_i32(&v, 1);
        }
    };
    if (fast_rollback && seq_verify && !ddtree_mode) {
        std::fprintf(stderr, "--fast-rollback and --seq-verify are mutually exclusive\n");
        return 2;
    }
    if (target_split_dflash) target_split_load_draft = true;
    if (target_gpus.empty()) target_gpus.push_back(target_gpu);
    if (target_gpus.size() == 1) target_gpu = target_gpus[0];
    std::printf("[cfg] seq_verify=%d fast_rollback=%d ddtree=%d budget=%d temp=%.2f chain_seed=%d fa_window=%d draft_swa=%d draft_ctx_max=%d draft_feature_mirror=%d target_cache_slots=%d stream_tagged=%d target_gpu=%d draft_gpu=%d\n",
                (int)seq_verify, (int)fast_rollback, (int)ddtree_mode,
                ddtree_budget, ddtree_temp, (int)ddtree_chain_seed, g_fa_window,
                g_draft_swa_window, g_draft_ctx_max, (int)draft_feature_mirror,
                daemon_mode ? target_cache_slots : 1,
                (int)stream_tagged, target_gpu, draft_gpu);
    auto env_enabled = [](const char * name) {
        const char * value = std::getenv(name);
        return value && std::atoi(value) != 0;
    };
    const bool step_debug = env_enabled("DFLASH27B_STEP_DEBUG");
    const bool scheduler_copyback_validate =
        env_enabled("DFLASH27B_SCHED_COPYBACK_VALIDATE");
    const bool timing_debug = env_enabled("DFLASH27B_TIMING_DEBUG");
    const bool output_tail = env_enabled("DFLASH27B_OUTPUT_TAIL");

    int cuda_device_count = 0;
    cudaGetDeviceCount(&cuda_device_count);
    if (target_gpu >= cuda_device_count || draft_gpu >= cuda_device_count) {
        std::fprintf(stderr, "bad gpu ids target=%d draft=%d device_count=%d\n",
                     target_gpu, draft_gpu, cuda_device_count);
        return 2;
    }

    const bool split_gpus = target_gpu != draft_gpu;
    ggml_backend_t target_backend = ggml_backend_cuda_init(target_gpu);
    if (!target_backend) { std::fprintf(stderr, "target cuda init failed\n"); return 1; }
    ggml_backend_t draft_backend = target_backend;
    if (split_gpus) {
        draft_backend = ggml_backend_cuda_init(draft_gpu);
        if (!draft_backend) { std::fprintf(stderr, "draft cuda init failed\n"); return 1; }
    }
    if (split_gpus && !enable_peer_access_pair(target_gpu, draft_gpu)) {
        std::fprintf(stderr,
                     "warning: CUDA peer access is not fully enabled for target=%d draft=%d; split transfers may fail\n",
                     target_gpu, draft_gpu);
    }
    ggml_backend_t backend = target_backend; // legacy target-side alias

    TargetWeights w;
    if (!load_target_gguf(target_path, target_backend, w)) {
        std::fprintf(stderr, "target load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    DraftWeights dw;
    {
        if (!load_draft_auto(draft_path, draft_backend, dw)) {
            std::fprintf(stderr, "draft load: %s\n", dflash27b_last_error());
            return 1;
        }
    }
    std::printf("[draft]  loaded\n");

    // Apply --draft-swa=N: mark layers 0..n-2 as SWA, last layer stays full.
    if (g_draft_swa_window > 0) {
        dw.swa_window = g_draft_swa_window;
        for (int il = 0; il < dw.n_layer - 1; il++) {
            dw.layers[il].is_swa = true;
        }
        std::printf("[draft]  SWA layers: %d/%d (window=%d)\n",
                    dw.n_layer - 1, dw.n_layer, dw.swa_window);
    }

    const int max_ctx = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;
    // Size the ssm_intermediate / conv_input_cache buffers to cover whichever
    // verify mode we'll use. DDTree needs room for 1 + ddtree_budget tree nodes.
    // Profile mode intentionally keeps the intermediate cache tiny (no capture)
    // so we can go up to n_tokens=128 without OOM.
    const int max_verify_tokens = profile_scaling
        ? DFLASH27B_DRAFT_BLOCK_SIZE
        : (ddtree_mode
            ? std::max<int>(DFLASH27B_DRAFT_BLOCK_SIZE, ddtree_budget + 1)
            : DFLASH27B_DRAFT_BLOCK_SIZE);
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend, cache,
                             /*prefill_only=*/true)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return 1;
    }

    // ── Profile mode: microbench target forward at varying N ───────────
    if (profile_scaling) {
        const int hidden_p = DFLASH27B_TARGET_HIDDEN;
        StepGraph psg;
        const int n_values[] = { 1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128 };
        std::printf("[profile] target forward ms at varying N (kv_start=0, no capture)\n");
        std::printf("%6s %10s %10s\n", "N", "total_ms", "ms_per_N");
        for (int n : n_values) {
            if (!build_target_step(psg, w, cache, backend,
                                   /*kv_start=*/0, /*n_tokens=*/n,
                                   /*with_mask=*/true,
                                   /*capture=*/false,
                                   /*capture_delta_intermediate=*/false)) {
                std::fprintf(stderr, "profile build N=%d failed\n", n); return 1;
            }
            // Fake embed input (zeros) + fake positions + fake causal mask.
            std::vector<float> emb(hidden_p * n, 0.0f);
            ggml_backend_tensor_set(psg.inp_embed, emb.data(), 0, sizeof(float) * emb.size());
            std::vector<int32_t> pos4(4 * n);
            for (int i = 0; i < n; i++) {
                pos4[0 * n + i] = i;
                pos4[1 * n + i] = i;
                pos4[2 * n + i] = i;
                pos4[3 * n + i] = 0;
            }
            ggml_backend_tensor_set(psg.positions, pos4.data(), 0, sizeof(int32_t) * 4 * n);
            if (psg.attn_mask) {
                const int kv_pad = (int)psg.attn_mask->ne[0];
                const int q_pad  = (int)psg.attn_mask->ne[1];
                std::vector<uint16_t> mask_buf_p((size_t)kv_pad * q_pad, F16_NEG_INF);
                for (int q = 0; q < n; q++) {
                    for (int k = 0; k <= q; k++) {
                        mask_buf_p[(size_t)q * kv_pad + k] = F16_ZERO;
                    }
                }
                ggml_backend_tensor_set(psg.attn_mask, mask_buf_p.data(), 0,
                                        sizeof(uint16_t) * mask_buf_p.size());
            }
            // Warmup
            ggml_backend_graph_compute(backend, psg.gf);
            // Time 5 runs, take median
            std::vector<double> times;
            for (int rep = 0; rep < 5; rep++) {
                auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(backend, psg.gf);
                auto t1 = std::chrono::steady_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(times.begin(), times.end());
            double median = times[times.size() / 2];
            std::printf("%6d %10.2f %10.3f\n", n, median, median / n);
        }
        free_target_cache(cache);
        free_target_weights(w);
        if (split_gpus) ggml_backend_free(draft_backend);
        ggml_backend_free(target_backend);
        return 0;
    }

    // ── Sliding-window regression tests ──────────────────────────────────
    if (test_window_mode) {
        int n_pass = 0, n_fail = 0;
        auto check = [&](bool cond, const char * msg) {
            if (cond) { n_pass++; std::printf("  PASS  %s\n", msg); }
            else      { n_fail++; std::fprintf(stderr, "  FAIL  %s\n", msg); }
        };

        // ── Test 1: build_causal_mask unit tests (CPU, no GPU needed) ──
        std::printf("[test-window] === Test 1: build_causal_mask ===\n");
        {
            std::vector<uint16_t> buf;
            // 1a: standard causal mask, no window: 4 queries at positions 2-5, 6 KV
            build_causal_mask(buf, /*kv_len=*/6, /*n_tokens=*/4, /*kv_start=*/2);
            const int pad = align_up(6, g_kq_stride_pad);
            // q=0 at pos 2: attend k=[0..2], 3 zeros
            // q=3 at pos 5: attend k=[0..5], 6 zeros
            check(buf[0 * pad + 0] == F16_ZERO, "1a: q=0,k=0 attendable");
            check(buf[0 * pad + 2] == F16_ZERO, "1a: q=0,k=2 attendable");
            check(buf[0 * pad + 3] == F16_NEG_INF, "1a: q=0,k=3 masked");
            check(buf[3 * pad + 5] == F16_ZERO, "1a: q=3,k=5 attendable");
            check(buf[3 * pad + 5] == F16_ZERO, "1a: q=3,k=5 attendable (diagonal)");
        }
        {
            std::vector<uint16_t> buf;
            // 1b: windowed mask: kv_start=10, n_tokens=3, win_start=8, win_len=5
            // Queries at positions 10,11,12. KV entries [8,9,10,11,12].
            build_causal_mask(buf, /*kv_len=*/5, /*n_tokens=*/3, /*kv_start=*/10, /*win_start=*/8);
            const int pad = align_up(5, g_kq_stride_pad);
            // q=0 (pos 10): attend k=[8..10] → indices [0..2]
            check(buf[0 * pad + 0] == F16_ZERO, "1b: q=0,k_abs=8 attendable");
            check(buf[0 * pad + 2] == F16_ZERO, "1b: q=0,k_abs=10 attendable");
            check(buf[0 * pad + 3] == F16_NEG_INF, "1b: q=0,k_abs=11 masked (future)");
            // q=2 (pos 12): attend k=[8..12] → indices [0..4]
            check(buf[2 * pad + 4] == F16_ZERO, "1b: q=2,k_abs=12 attendable");
        }
        {
            std::vector<uint16_t> buf;
            // 1c: large window > kv_start (window inactive): kv_start=100, win_start=0
            build_causal_mask(buf, /*kv_len=*/106, /*n_tokens=*/6, /*kv_start=*/100, /*win_start=*/0);
            const int pad = align_up(106, g_kq_stride_pad);
            // q=0 (pos 100): attend k=[0..100]
            check(buf[0 * pad + 0] == F16_ZERO, "1c: q=0,k=0 attendable (no window)");
            check(buf[0 * pad + 100] == F16_ZERO, "1c: q=0,k=100 attendable");
            check(buf[0 * pad + 101] == F16_NEG_INF, "1c: q=0,k=101 masked");
            // q=5 (pos 105): attend k=[0..105]
            check(buf[5 * pad + 105] == F16_ZERO, "1c: q=5,k=105 attendable");
        }

        // ── Tests 2 & 3: GPU regression tests ───────────────────────────
        const int hidden_t = DFLASH27B_TARGET_HIDDEN;
        const int vocab_t  = DFLASH27B_TARGET_VOCAB;
        auto do_prefill = [&](StepGraph & psg, int n_tokens) -> int32_t {
            const int pf_ub = 384;
            int32_t lt = -1;
            int committed_p = 0;
            std::vector<float> pf_emb;
            std::vector<int32_t> pf_pos;
            std::vector<uint16_t> pf_mask;
            std::vector<float> pf_logits;
            for (int start = 0; start < n_tokens; start += pf_ub) {
                const int nt = std::min(pf_ub, n_tokens - start);
                const int kv_len_p = start + nt;
                const bool with_m = (g_kq_stride_pad > KQ_MASK_PAD) || (nt > 1);
                if (!build_target_step(psg, w, cache, backend,
                                        start, nt, with_m, true)) {
                    std::fprintf(stderr, "prefill build @%d\n", start); return -1;
                }
                pf_emb.assign((size_t)hidden_t * nt, 0.0f);
                std::vector<int32_t> tokens(nt, 220);
                if (!w.embedder.embed(tokens.data(), nt, pf_emb.data())) return -1;
                ggml_backend_tensor_set(psg.inp_embed, pf_emb.data(), 0,
                                        sizeof(float) * pf_emb.size());
                pf_pos.assign((size_t)4 * nt, 0);
                for (int i = 0; i < nt; i++) {
                    const int p = start + i;
                    pf_pos[0 * nt + i] = p;
                    pf_pos[1 * nt + i] = p;
                    pf_pos[2 * nt + i] = p;
                    pf_pos[3 * nt + i] = 0;
                }
                ggml_backend_tensor_set(psg.positions, pf_pos.data(), 0,
                                        sizeof(int32_t) * pf_pos.size());
                if (with_m) {
                    build_causal_mask(pf_mask, kv_len_p, nt, start);
                    ggml_backend_tensor_set(psg.attn_mask, pf_mask.data(), 0,
                                            sizeof(uint16_t) * pf_mask.size());
                }
                auto st = ggml_backend_graph_compute(backend, psg.gf);
                if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "prefill fail @%d\n", start); return -1; }
                pf_logits.assign(vocab_t, 0.0f);
                ggml_backend_tensor_get(psg.logits, pf_logits.data(),
                                        (size_t)(nt - 1) * vocab_t * sizeof(float),
                                        sizeof(float) * vocab_t);
                lt = argmax_f32(pf_logits.data(), vocab_t);
                committed_p = start + nt;
            }
            return lt;
        };

        auto decode_one = [&](StepGraph & dsg, int kv_start, int32_t tok,
                               int32_t pos, int fa_w, float * logits_out) -> bool {
            if (!build_target_step(dsg, w, cache, backend,
                                    kv_start, 1, false, true, false, fa_w)) {
                std::fprintf(stderr, "decode build failed\n"); return false;
            }
            float emb_buf[5120];
            if (!w.embedder.embed(&tok, 1, emb_buf)) return false;
            ggml_backend_tensor_set(dsg.inp_embed, emb_buf, 0, sizeof(float) * hidden_t);
            int32_t pos4[4] = {pos, pos, pos, 0};
            ggml_backend_tensor_set(dsg.positions, pos4, 0, sizeof(int32_t) * 4);
            auto st = ggml_backend_graph_compute(backend, dsg.gf);
            if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "decode compute failed\n"); return false; }
            ggml_backend_tensor_get(dsg.logits, logits_out, 0, sizeof(float) * vocab_t);
            return true;
        };

        auto cosine_sim = [](const float * a, const float * b, int n) -> double {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
            return (na < 1e-30 || nb < 1e-30) ? 0.0 : dot / (std::sqrt(na) * std::sqrt(nb));
        };

        // ── Test 2: Short context identity (window inactive) ────────────
        std::printf("[test-window] === Test 2: short-ctx identity (512 tokens) ===\n");
        {
            StepGraph psg2;
            int32_t lt2 = do_prefill(psg2, 512);
            check(lt2 >= 0, "prefill 512 tokens succeeded");

            // Need rollback tensors for snapshot/restore
            step_graph_free(psg2);
            psg2 = StepGraph{};
            migrate_prefill_cache(w, max_ctx, max_verify_tokens, target_backend, cache);

            snapshot_ssm_state(cache);
            std::vector<float> logits_full(vocab_t), logits_win(vocab_t);
            bool ok = decode_one(psg2, 512, lt2, 512, 0, logits_full.data());
            check(ok, "decode full-attention succeeded");
            restore_ssm_state(cache);
            ok = decode_one(psg2, 512, lt2, 512, 2048, logits_win.data());
            check(ok, "decode window=2048 succeeded");

            int tok_full = argmax_f32(logits_full.data(), vocab_t);
            int tok_win  = argmax_f32(logits_win.data(), vocab_t);
            check(tok_full == tok_win, "short-ctx: argmax matches");
            double cs = cosine_sim(logits_full.data(), logits_win.data(), vocab_t);
            char msg[128];
            std::snprintf(msg, sizeof(msg), "short-ctx: cosine_sim=%.8f (expect >0.9999)", cs);
            check(cs > 0.9999, msg);

            step_graph_free(psg2);
        }

        // Reset cache for next test
        free_target_cache(cache);
        if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend, cache, true)) {
            std::fprintf(stderr, "cache realloc failed\n"); return 1;
        }

        // ── Test 3: Long context argmax match ───────────────────────────
        std::printf("[test-window] === Test 3: long-ctx quality (4096 tokens) ===\n");
        {
            StepGraph psg3;
            int32_t lt3 = do_prefill(psg3, 4096);
            check(lt3 >= 0, "prefill 4096 tokens succeeded");

            step_graph_free(psg3);
            psg3 = StepGraph{};
            migrate_prefill_cache(w, max_ctx, max_verify_tokens, target_backend, cache);

            snapshot_ssm_state(cache);
            std::vector<float> logits_full(vocab_t), logits_win(vocab_t);
            bool ok = decode_one(psg3, 4096, lt3, 4096, 0, logits_full.data());
            check(ok, "decode full-attention succeeded");
            restore_ssm_state(cache);
            ok = decode_one(psg3, 4096, lt3, 4096, 1024, logits_win.data());
            check(ok, "decode window=1024 succeeded");

            int tok_full = argmax_f32(logits_full.data(), vocab_t);
            int tok_win  = argmax_f32(logits_win.data(), vocab_t);
            char msg[128];
            std::snprintf(msg, sizeof(msg), "long-ctx: argmax full=%d win=%d match", tok_full, tok_win);
            check(tok_full == tok_win, msg);
            double cs = cosine_sim(logits_full.data(), logits_win.data(), vocab_t);
            std::snprintf(msg, sizeof(msg), "long-ctx: cosine_sim=%.6f (expect >0.90)", cs);
            check(cs > 0.90, msg);

            step_graph_free(psg3);
        }

        std::printf("[test-window] === Results: %d passed, %d failed ===\n", n_pass, n_fail);
        free_target_cache(cache);
        free_target_weights(w);
        if (split_gpus) ggml_backend_free(draft_backend);
        ggml_backend_free(target_backend);
        return n_fail > 0 ? 1 : 0;
    }

    const int q_len  = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab  = DFLASH27B_TARGET_VOCAB;
    const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;

    std::vector<std::unique_ptr<DaemonSlotState>> daemon_extra_slots;
    if (daemon_mode && target_cache_slots > 1) {
        daemon_extra_slots.reserve((size_t)target_cache_slots - 1);
        for (int sid = 1; sid < target_cache_slots; sid++) {
            auto slot = std::make_unique<DaemonSlotState>();
            if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend,
                                     slot->cache, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache slot %d: %s\n", sid, dflash27b_last_error());
                for (auto & allocated : daemon_extra_slots) {
                    free_target_cache(allocated->cache);
                }
                free_target_cache(cache);
                free_draft_weights(dw);
                free_target_weights(w);
                if (split_gpus) ggml_backend_free(draft_backend);
                ggml_backend_free(target_backend);
                return 1;
            }
            daemon_extra_slots.push_back(std::move(slot));
        }
        std::printf("[daemon] target_cache_slots=%d (shared weights, serialized protocol)\n",
                    target_cache_slots);
    }

    if (daemon_mode) {
        std::printf("[daemon] ready\n");
        std::fflush(stdout);
    }

    PrefixSnapshot prefix_snapshots[PREFIX_CACHE_SLOTS];   // default-constructed, ctx==nullptr

    StepGraph sg;
    StepGraph draft_sg;
    StepGraph proj_sg;
    StepGraph daemon_batch_probe_sg;
    TargetCache daemon_batch_probe_cache;
    int daemon_batch_probe_capacity = 0;
    StepGraph daemon_batch_ref_sg;
    TargetCache daemon_batch_ref_cache;
    bool daemon_batch_ref_cache_ready = false;
    DraftFeatureMirror feature_mirror;
    bool daemon_first_iter = true;
    bool target_parked = false;
    bool draft_parked  = false;
    // pflash drafter (lazy-loaded on first `compress` command)
    dflash27b::DrafterContext drafter_ctx;
    bool drafter_loaded = false;

    auto destroy_target_graphs_all_slots = [&]() {
        step_graph_destroy(proj_sg);
        step_graph_destroy(sg);
        step_graph_destroy(daemon_batch_probe_sg);
        step_graph_destroy(daemon_batch_ref_sg);
        free_target_cache(daemon_batch_probe_cache);
        daemon_batch_probe_capacity = 0;
        free_target_cache(daemon_batch_ref_cache);
        daemon_batch_ref_cache_ready = false;
        for (auto & slot : daemon_extra_slots) {
            step_graph_destroy(slot->proj_sg);
            step_graph_destroy(slot->sg);
        }
    };
    auto destroy_draft_graphs_all_slots = [&]() {
        step_graph_destroy(draft_sg);
        for (auto & slot : daemon_extra_slots) {
            step_graph_destroy(slot->draft_sg);
        }
    };

    std::vector<DaemonRequestState> daemon_requests((size_t)target_cache_slots);
    std::deque<DaemonPendingQuantum> daemon_pending_quanta;
    std::deque<std::string> daemon_input_lines;
    std::mutex daemon_input_mu;
    std::condition_variable daemon_input_cv;
    bool daemon_input_eof = !daemon_mode;
    size_t daemon_scheduler_cursor = 0;
    bool daemon_scheduler_drain = false;
    auto get_daemon_slot_runtime_ref = [&](int slot_id) -> DaemonSlotRuntimeRef {
        DaemonSlotRuntimeRef ref{};
        ref.slot_id = slot_id;
        if (slot_id == 0) {
            ref.cache = &cache;
            ref.sg = &sg;
            ref.draft_sg = &draft_sg;
            ref.proj_sg = &proj_sg;
            ref.feature_mirror = &feature_mirror;
            ref.first_iter = &daemon_first_iter;
            return ref;
        }
        if (slot_id > 0 && slot_id < target_cache_slots) {
            DaemonSlotState * slot = daemon_extra_slots[(size_t)slot_id - 1].get();
            ref.cache = &slot->cache;
            ref.sg = &slot->sg;
            ref.draft_sg = &slot->draft_sg;
            ref.proj_sg = &slot->proj_sg;
            ref.feature_mirror = &slot->feature_mirror;
            ref.first_iter = &slot->first_iter;
        }
        return ref;
    };
    auto collect_scheduler_candidates = [&]() -> std::vector<DaemonBatchCandidate> {
        std::vector<DaemonBatchCandidate> candidates;
        if (daemon_requests.empty()) return candidates;
        const size_t n_slots = daemon_requests.size();
        candidates.reserve(n_slots);
        for (size_t i = 0; i < n_slots; i++) {
            const size_t idx = (daemon_scheduler_cursor + i) % n_slots;
            const DaemonRequestState & req = daemon_requests[idx];
            if (!req.active || req.remaining <= 0) continue;
            DaemonSlotRuntimeRef runtime = get_daemon_slot_runtime_ref(req.slot_id);
            if (!runtime.cache || !runtime.sg || !runtime.draft_sg) continue;
            const int q = std::max(1, std::min(req.quantum, req.remaining));
            candidates.push_back(DaemonBatchCandidate{
                req.request_id, req.slot_id, req.epoch, q, req.remaining, runtime});
        }
        return candidates;
    };
    auto collect_scheduler_aligned_bucket = [&](int max_batch,
                                                bool tail_only,
                                                bool emit_log)
        -> DaemonAlignedBucketSelection {
        DaemonAlignedBucketSelection selection =
            select_aligned_scheduler_bucket(collect_scheduler_candidates(),
                                            max_batch,
                                            tail_only);
        if (emit_log) print_aligned_bucket_log(selection, max_batch);
        return selection;
    };
    auto run_scheduler_batch_probe = [&](int max_batch) -> bool {
        DaemonAlignedBucketSelection selection =
            collect_scheduler_aligned_bucket(max_batch, /*tail_only=*/false,
                                             /*emit_log=*/true);
        std::vector<DaemonBatchCandidate> batch = selection.batch;
        if ((int)batch.size() < 2) {
            std::fprintf(stderr, "[scheduler] batch probe needs an aligned cur_pos bucket with at least 2 active requests\n");
            return false;
        }

        const int batch_size = (int)batch.size();
        const int kv_start = selection.kv_start;
        if (kv_start <= 0) {
            std::fprintf(stderr, "[scheduler] batch probe requires decoded slots\n");
            return false;
        }
        for (const DaemonBatchCandidate & candidate : batch) {
            TargetCache * c = candidate.runtime.cache;
            if (!c || c->cur_pos != kv_start || c->last_tok < 0) {
                std::fprintf(stderr,
                             "[scheduler] batch probe requires same cur_pos and valid last_tok\n");
                return false;
            }
        }

        if (daemon_batch_probe_capacity < batch_size) {
            step_graph_free(daemon_batch_probe_sg);
            free_target_cache(daemon_batch_probe_cache);
            if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend,
                                     daemon_batch_probe_cache,
                                     /*prefill_only=*/true, batch_size)) {
                std::fprintf(stderr, "[scheduler] batch probe cache: %s\n",
                             dflash27b_last_error());
                daemon_batch_probe_capacity = 0;
                return false;
            }
            daemon_batch_probe_capacity = batch_size;
        }

        reset_target_cache(daemon_batch_probe_cache);
        for (int i = 0; i < batch_size; i++) {
            if (!copy_target_cache_to_batch_slot(*batch[(size_t)i].runtime.cache,
                                                 daemon_batch_probe_cache, i)) {
                return false;
            }
        }

        if (!build_target_batch_probe_step(daemon_batch_probe_sg, w,
                                           daemon_batch_probe_cache,
                                           target_backend, kv_start,
                                           batch_size)) {
            std::fprintf(stderr, "[scheduler] batch probe graph build failed: %s\n",
                         dflash27b_last_error());
            return false;
        }

        std::vector<int32_t> seed_tokens((size_t)batch_size);
        for (int i = 0; i < batch_size; i++) {
            seed_tokens[(size_t)i] = batch[(size_t)i].runtime.cache->last_tok;
        }
        std::vector<float> embed((size_t)hidden * batch_size);
        if (!w.embedder.embed(seed_tokens.data(), batch_size, embed.data())) {
            std::fprintf(stderr, "[scheduler] batch probe embed failed\n");
            return false;
        }
        ggml_backend_tensor_set(daemon_batch_probe_sg.inp_embed, embed.data(), 0,
                                sizeof(float) * embed.size());

        const int32_t pos4[4] = {kv_start, kv_start, kv_start, 0};
        ggml_backend_tensor_set(daemon_batch_probe_sg.positions, pos4, 0,
                                sizeof(pos4));

        auto t0 = std::chrono::steady_clock::now();
        ggml_status st = ggml_backend_graph_compute(target_backend,
                                                    daemon_batch_probe_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[scheduler] batch probe compute %d\n", (int)st);
            return false;
        }
        auto t1 = std::chrono::steady_clock::now();
        std::vector<int32_t> argmax((size_t)batch_size);
        ggml_backend_tensor_get(daemon_batch_probe_sg.argmax_tokens,
                                argmax.data(), 0,
                                sizeof(int32_t) * argmax.size());
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!daemon_batch_ref_cache_ready) {
            free_target_cache(daemon_batch_ref_cache);
            if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend,
                                     daemon_batch_ref_cache,
                                     /*prefill_only=*/true, /*n_seqs=*/1)) {
                std::fprintf(stderr, "[scheduler] batch reference cache: %s\n",
                             dflash27b_last_error());
                daemon_batch_ref_cache_ready = false;
                return false;
            }
            daemon_batch_ref_cache_ready = true;
        }

        std::vector<int32_t> ref_argmax((size_t)batch_size, -1);
        std::vector<float> ref_embed((size_t)hidden);
        int mismatches = 0;
        for (int i = 0; i < batch_size; i++) {
            const TargetCache & source = *batch[(size_t)i].runtime.cache;
            if (!copy_target_cache_to_single_cache(source, daemon_batch_ref_cache)) {
                return false;
            }
            if (!build_target_step(daemon_batch_ref_sg, w, daemon_batch_ref_cache,
                                   target_backend, kv_start, /*n_tokens=*/1,
                                   /*with_mask=*/false, /*capture=*/false,
                                   /*capture_delta_intermediate=*/false,
                                   /*fa_window=*/0)) {
                std::fprintf(stderr,
                             "[scheduler] batch reference graph build failed: %s\n",
                             dflash27b_last_error());
                return false;
            }
            const int32_t seed = batch[(size_t)i].runtime.cache->last_tok;
            if (!w.embedder.embed(&seed, 1, ref_embed.data())) {
                std::fprintf(stderr, "[scheduler] batch reference embed failed\n");
                return false;
            }
            ggml_backend_tensor_set(daemon_batch_ref_sg.inp_embed,
                                    ref_embed.data(), 0,
                                    sizeof(float) * ref_embed.size());
            ggml_backend_tensor_set(daemon_batch_ref_sg.positions, pos4, 0,
                                    sizeof(pos4));
            ggml_status ref_st = ggml_backend_graph_compute(
                target_backend, daemon_batch_ref_sg.gf);
            if (ref_st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[scheduler] batch reference compute %d\n",
                             (int)ref_st);
                return false;
            }
            ggml_backend_tensor_get(daemon_batch_ref_sg.argmax_tokens,
                                    &ref_argmax[(size_t)i], 0,
                                    sizeof(int32_t));
            if (ref_argmax[(size_t)i] != argmax[(size_t)i]) {
                mismatches++;
            }
        }

        std::printf("[scheduler] batch_probe_ok count=%d kv_start=%d ms=%.2f",
                    batch_size, kv_start, ms);
        for (int i = 0; i < batch_size; i++) {
            std::printf(" req=%d:slot=%d:tok=%d",
                        batch[(size_t)i].request_id,
                        batch[(size_t)i].slot_id,
                        argmax[(size_t)i]);
        }
        std::printf("\n");
        std::fflush(stdout);

        if (mismatches > 0) {
            std::printf("[scheduler] batch_probe_compare_fail count=%d mismatches=%d",
                        batch_size, mismatches);
            for (int i = 0; i < batch_size; i++) {
                std::printf(" req=%d:slot=%d:batched=%d:single=%d",
                            batch[(size_t)i].request_id,
                            batch[(size_t)i].slot_id,
                            argmax[(size_t)i],
                            ref_argmax[(size_t)i]);
            }
            std::printf("\n");
            std::fflush(stdout);
            return false;
        }

        std::printf("[scheduler] batch_probe_compare_ok count=%d mismatches=0",
                    batch_size);
        for (int i = 0; i < batch_size; i++) {
            std::printf(" req=%d:slot=%d:batched=%d:single=%d",
                        batch[(size_t)i].request_id,
                        batch[(size_t)i].slot_id,
                        argmax[(size_t)i],
                        ref_argmax[(size_t)i]);
        }
        std::printf("\n");
        std::fflush(stdout);
        return true;
    };
    auto run_scheduler_batch_target_step = [&](int max_batch,
                                               bool tail_only,
                                               bool quiet_unavailable = false) -> bool {
        const char * log_prefix = tail_only ? "batch_tail" : "batch_step";
        const char * label = tail_only ? "batch target tail" : "batch target step";
        if (!stream_tagged) {
            std::fprintf(stderr,
                         "[scheduler] %s requires --stream-tagged\n", label);
            return false;
        }
        DaemonAlignedBucketSelection selection =
            collect_scheduler_aligned_bucket(max_batch, tail_only,
                                             /*emit_log=*/true);
        std::vector<DaemonBatchCandidate> batch = selection.batch;
        if ((int)batch.size() < 2) {
            if (!quiet_unavailable) {
                std::fprintf(stderr,
                             "[scheduler] %s needs an aligned cur_pos bucket with at least 2 active requests\n", label);
            }
            return false;
        }

        const int batch_size = (int)batch.size();
        const int kv_start = selection.kv_start;
        if (kv_start <= 0) {
            if (!quiet_unavailable) {
                std::fprintf(stderr, "[scheduler] %s requires decoded slots\n", label);
            }
            return false;
        }
        for (const DaemonBatchCandidate & candidate : batch) {
            TargetCache * c = candidate.runtime.cache;
            if (!c || c->cur_pos != kv_start || c->last_tok < 0
                || candidate.remaining <= 0
                || (tail_only && candidate.remaining != 1)) {
                if (!quiet_unavailable) {
                    std::fprintf(stderr,
                                 "[scheduler] %s requires same cur_pos, valid last_tok%s\n",
                                 label,
                                 tail_only ? ", and remaining=1" : "");
                }
                return false;
            }
        }

        if (daemon_batch_probe_capacity < batch_size) {
            step_graph_free(daemon_batch_probe_sg);
            free_target_cache(daemon_batch_probe_cache);
            if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend,
                                     daemon_batch_probe_cache,
                                     /*prefill_only=*/true, batch_size)) {
                std::fprintf(stderr, "[scheduler] %s cache: %s\n",
                             label, dflash27b_last_error());
                daemon_batch_probe_capacity = 0;
                return false;
            }
            daemon_batch_probe_capacity = batch_size;
        }

        reset_target_cache(daemon_batch_probe_cache);
        for (int i = 0; i < batch_size; i++) {
            if (!copy_target_cache_to_batch_slot(*batch[(size_t)i].runtime.cache,
                                                 daemon_batch_probe_cache, i)) {
                return false;
            }
        }

        if (!build_target_batch_probe_step(daemon_batch_probe_sg, w,
                                           daemon_batch_probe_cache,
                                           target_backend, kv_start,
                                           batch_size,
                                           /*capture_layers=*/true)) {
            std::fprintf(stderr, "[scheduler] %s graph build failed: %s\n",
                         label, dflash27b_last_error());
            return false;
        }

        std::vector<int32_t> old_tokens((size_t)batch_size);
        for (int i = 0; i < batch_size; i++) {
            old_tokens[(size_t)i] = batch[(size_t)i].runtime.cache->last_tok;
        }
        std::vector<float> embed((size_t)hidden * batch_size);
        if (!w.embedder.embed(old_tokens.data(), batch_size, embed.data())) {
            std::fprintf(stderr, "[scheduler] %s embed failed\n", label);
            return false;
        }
        ggml_backend_tensor_set(daemon_batch_probe_sg.inp_embed, embed.data(), 0,
                                sizeof(float) * embed.size());

        const int32_t pos4[4] = {kv_start, kv_start, kv_start, 0};
        ggml_backend_tensor_set(daemon_batch_probe_sg.positions, pos4, 0,
                                sizeof(pos4));

        auto t0 = std::chrono::steady_clock::now();
        ggml_status st = ggml_backend_graph_compute(target_backend,
                                                    daemon_batch_probe_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[scheduler] %s compute %d\n", label, (int)st);
            return false;
        }
        auto t1 = std::chrono::steady_clock::now();
        std::vector<int32_t> argmax((size_t)batch_size);
        ggml_backend_tensor_get(daemon_batch_probe_sg.argmax_tokens,
                                argmax.data(), 0,
                                sizeof(int32_t) * argmax.size());
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!daemon_batch_ref_cache_ready) {
            free_target_cache(daemon_batch_ref_cache);
            if (!create_target_cache(w, max_ctx, max_verify_tokens, target_backend,
                                     daemon_batch_ref_cache,
                                     /*prefill_only=*/true, /*n_seqs=*/1)) {
                std::fprintf(stderr, "[scheduler] %s reference cache: %s\n",
                             label, dflash27b_last_error());
                daemon_batch_ref_cache_ready = false;
                return false;
            }
            daemon_batch_ref_cache_ready = true;
        }

        std::vector<int32_t> ref_argmax((size_t)batch_size, -1);
        std::vector<float> ref_embed((size_t)hidden);
        auto compute_reference_argmax = [&](TargetCache & ref_cache,
                                            int step_pos,
                                            int32_t seed,
                                            const char * stage,
                                            int32_t * out) -> bool {
            if (!build_target_step(daemon_batch_ref_sg, w, ref_cache,
                                   target_backend, step_pos, /*n_tokens=*/1,
                                   /*with_mask=*/false, /*capture=*/false,
                                   /*capture_delta_intermediate=*/false,
                                   /*fa_window=*/0)) {
                std::fprintf(stderr,
                             "[scheduler] %s %s graph build failed: %s\n",
                             label, stage, dflash27b_last_error());
                return false;
            }
            if (!w.embedder.embed(&seed, 1, ref_embed.data())) {
                std::fprintf(stderr, "[scheduler] %s %s embed failed\n",
                             label, stage);
                return false;
            }
            ggml_backend_tensor_set(daemon_batch_ref_sg.inp_embed,
                                    ref_embed.data(), 0,
                                    sizeof(float) * ref_embed.size());
            const int32_t pos4[4] = {step_pos, step_pos, step_pos, 0};
            ggml_backend_tensor_set(daemon_batch_ref_sg.positions, pos4, 0,
                                    sizeof(pos4));
            ggml_status ref_st = ggml_backend_graph_compute(
                target_backend, daemon_batch_ref_sg.gf);
            if (ref_st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[scheduler] %s %s compute %d\n",
                             label, stage, (int)ref_st);
                return false;
            }
            ggml_backend_tensor_get(daemon_batch_ref_sg.argmax_tokens,
                                    out, 0, sizeof(int32_t));
            return true;
        };
        int mismatches = 0;
        for (int i = 0; i < batch_size; i++) {
            const TargetCache & source = *batch[(size_t)i].runtime.cache;
            if (!copy_target_cache_to_single_cache(source, daemon_batch_ref_cache)) {
                return false;
            }
            if (!compute_reference_argmax(daemon_batch_ref_cache, kv_start,
                                          old_tokens[(size_t)i], "reference",
                                          &ref_argmax[(size_t)i])) {
                return false;
            }
            if (ref_argmax[(size_t)i] != argmax[(size_t)i]) {
                mismatches++;
            }
        }

        if (mismatches > 0) {
            std::printf("[scheduler] %s_compare_fail count=%d mismatches=%d",
                        log_prefix, batch_size, mismatches);
            for (int i = 0; i < batch_size; i++) {
                std::printf(" req=%d:slot=%d:batched=%d:single=%d",
                            batch[(size_t)i].request_id,
                            batch[(size_t)i].slot_id,
                            argmax[(size_t)i],
                            ref_argmax[(size_t)i]);
            }
            std::printf("\n");
            std::fflush(stdout);
            return false;
        }

        for (int i = 0; i < batch_size; i++) {
            if (!copy_batch_slot_to_target_cache(daemon_batch_probe_cache, i,
                                                 *batch[(size_t)i].runtime.cache)) {
                return false;
            }
        }
        for (int i = 0; i < batch_size; i++) {
            TargetCache * c = batch[(size_t)i].runtime.cache;
            c->cur_pos  = kv_start + 1;
            c->last_tok = argmax[(size_t)i];
        }

        if (scheduler_copyback_validate) {
            std::vector<int32_t> copyback_ref_argmax((size_t)batch_size, -1);
            std::vector<int32_t> copyback_resume_argmax((size_t)batch_size, -1);
            int copyback_mismatches = 0;
            for (int i = 0; i < batch_size; i++) {
                if (!copy_batch_slot_to_target_cache(daemon_batch_probe_cache, i,
                                                     daemon_batch_ref_cache)) {
                    return false;
                }
                daemon_batch_ref_cache.cur_pos  = kv_start + 1;
                daemon_batch_ref_cache.last_tok = argmax[(size_t)i];
                if (!compute_reference_argmax(daemon_batch_ref_cache, kv_start + 1,
                                              argmax[(size_t)i],
                                              "copyback reference",
                                              &copyback_ref_argmax[(size_t)i])) {
                    return false;
                }
                if (!copy_target_cache_to_single_cache(
                        *batch[(size_t)i].runtime.cache, daemon_batch_ref_cache)) {
                    return false;
                }
                if (!compute_reference_argmax(
                        daemon_batch_ref_cache,
                        batch[(size_t)i].runtime.cache->cur_pos,
                        batch[(size_t)i].runtime.cache->last_tok,
                        "copyback resume",
                        &copyback_resume_argmax[(size_t)i])) {
                    return false;
                }
                if (copyback_ref_argmax[(size_t)i]
                    != copyback_resume_argmax[(size_t)i]) {
                    copyback_mismatches++;
                }
            }
            if (copyback_mismatches > 0) {
                std::printf("[scheduler] %s_copyback_compare_fail count=%d mismatches=%d",
                            log_prefix, batch_size, copyback_mismatches);
                for (int i = 0; i < batch_size; i++) {
                    std::printf(" req=%d:slot=%d:committed=%d:ref=%d:resume=%d",
                                batch[(size_t)i].request_id,
                                batch[(size_t)i].slot_id,
                                argmax[(size_t)i],
                                copyback_ref_argmax[(size_t)i],
                                copyback_resume_argmax[(size_t)i]);
                }
                std::printf("\n");
                std::fflush(stdout);
                return false;
            }
            std::printf("[scheduler] %s_copyback_compare_ok count=%d mismatches=0",
                        log_prefix, batch_size);
            for (int i = 0; i < batch_size; i++) {
                std::printf(" req=%d:slot=%d:committed=%d:ref=%d:resume=%d",
                            batch[(size_t)i].request_id,
                            batch[(size_t)i].slot_id,
                            argmax[(size_t)i],
                            copyback_ref_argmax[(size_t)i],
                            copyback_resume_argmax[(size_t)i]);
            }
            std::printf("\n");
            std::fflush(stdout);
        }

        std::printf("[scheduler] %s_compare_ok count=%d mismatches=0",
                    log_prefix, batch_size);
        for (int i = 0; i < batch_size; i++) {
            std::printf(" req=%d:slot=%d:batched=%d:single=%d",
                        batch[(size_t)i].request_id,
                        batch[(size_t)i].slot_id,
                        argmax[(size_t)i],
                        ref_argmax[(size_t)i]);
        }
        std::printf("\n");

        std::printf("[scheduler] %s_commit count=%d kv_start=%d ms=%.2f",
                    log_prefix, batch_size, kv_start, ms);
        for (int i = 0; i < batch_size; i++) {
            DaemonRequestState & req =
                daemon_requests[(size_t)batch[(size_t)i].slot_id];
            req.emitted += 1;
            req.remaining = std::max(0, req.remaining - 1);
            req.active = req.remaining > 0;
            std::printf(" req=%d:slot=%d:remaining=%d:tok=%d:next=%d:emitted=%d",
                        req.request_id, req.slot_id,
                        req.remaining, old_tokens[(size_t)i],
                        argmax[(size_t)i], req.emitted);
        }
        std::printf("\n");
        std::fflush(stdout);
        daemon_scheduler_cursor =
            next_cursor_after_aligned_batch(selection,
                                            daemon_requests.size(),
                                            daemon_scheduler_cursor);

        for (int i = 0; i < batch_size; i++) {
            const DaemonRequestState & req =
                daemon_requests[(size_t)batch[(size_t)i].slot_id];
            stream_emit_for(req.request_id, old_tokens[(size_t)i]);
            stream_emit_for(req.request_id, req.active ? -4 : -1);
        }
        return true;
    };
    auto enqueue_next_scheduler_quantum = [&]() -> bool {
        if (daemon_requests.empty()) return false;
        const size_t n_slots = daemon_requests.size();
        for (size_t i = 0; i < n_slots; i++) {
            const size_t idx = (daemon_scheduler_cursor + i) % n_slots;
            const DaemonRequestState & req = daemon_requests[idx];
            if (!req.active || req.remaining <= 0) continue;
            const int q = std::max(1, std::min(req.quantum, req.remaining));
            daemon_pending_quanta.push_back(
                DaemonPendingQuantum{req.request_id, req.slot_id, req.epoch, q});
            daemon_scheduler_cursor = (idx + 1) % n_slots;
            return true;
        }
        return false;
    };
    auto has_daemon_input = [&]() -> bool {
        std::lock_guard<std::mutex> lock(daemon_input_mu);
        return !daemon_input_lines.empty();
    };
    auto pop_daemon_line = [&](std::string & line) -> bool {
        {
            std::lock_guard<std::mutex> lock(daemon_input_mu);
            if (!daemon_input_lines.empty()) {
                line = std::move(daemon_input_lines.front());
                daemon_input_lines.pop_front();
                return true;
            }
        }
        while (!daemon_pending_quanta.empty()) {
            DaemonPendingQuantum quantum = daemon_pending_quanta.front();
            daemon_pending_quanta.pop_front();
            if (quantum.slot_id < 0
                || quantum.slot_id >= (int)daemon_requests.size()) {
                continue;
            }
            const DaemonRequestState & req = daemon_requests[(size_t)quantum.slot_id];
            if (!req.active
                || req.request_id != quantum.request_id
                || req.epoch != quantum.epoch
                || req.remaining <= 0) {
                std::printf("[scheduler] stale quantum dropped req=%d slot=%d epoch=%d\n",
                            quantum.request_id, quantum.slot_id, quantum.epoch);
                std::fflush(stdout);
                continue;
            }
            const int q = std::max(1, std::min(quantum.n_gen, req.remaining));
            line = "REQ " + std::to_string(quantum.request_id) +
                   " SLOT " + std::to_string(quantum.slot_id) +
                   " CONTINUE " + std::to_string(q);
            return true;
        }
        std::unique_lock<std::mutex> lock(daemon_input_mu);
        daemon_input_cv.wait(lock, [&]() {
            return !daemon_input_lines.empty()
                || !daemon_pending_quanta.empty()
                || daemon_input_eof;
        });
        if (!daemon_input_lines.empty()) {
            line = std::move(daemon_input_lines.front());
            daemon_input_lines.pop_front();
            return true;
        }
        while (!daemon_pending_quanta.empty()) {
            DaemonPendingQuantum quantum = daemon_pending_quanta.front();
            daemon_pending_quanta.pop_front();
            if (quantum.slot_id < 0
                || quantum.slot_id >= (int)daemon_requests.size()) {
                continue;
            }
            const DaemonRequestState & req = daemon_requests[(size_t)quantum.slot_id];
            if (!req.active
                || req.request_id != quantum.request_id
                || req.epoch != quantum.epoch
                || req.remaining <= 0) {
                std::printf("[scheduler] stale quantum dropped req=%d slot=%d epoch=%d\n",
                            quantum.request_id, quantum.slot_id, quantum.epoch);
                std::fflush(stdout);
                continue;
            }
            const int q = std::max(1, std::min(quantum.n_gen, req.remaining));
            line = "REQ " + std::to_string(quantum.request_id) +
                   " SLOT " + std::to_string(quantum.slot_id) +
                   " CONTINUE " + std::to_string(q);
            return true;
        }
        return false;
    };
    if (daemon_mode) {
        std::thread([&]() {
            std::string input_line;
            while (std::getline(std::cin, input_line)) {
                {
                    std::lock_guard<std::mutex> lock(daemon_input_mu);
                    daemon_input_lines.push_back(std::move(input_line));
                }
                daemon_input_cv.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(daemon_input_mu);
                daemon_input_eof = true;
            }
            daemon_input_cv.notify_all();
        }).detach();
    }

    while (true) {
        std::string prompt_file_str;
        bool restore_from_slot        = false;
        int  restore_slot_id          = -1;
        bool chain_restore_requested  = false;
        int  chain_thick_slot         = -1;
        std::vector<int> chain_thin_ids;
        bool continue_decode_requested = false;
        bool scheduler_start_requested = false;
        bool scheduler_continue_requested = false;
        int scheduler_total_gen = 0;
        int scheduler_quantum = 0;
        int active_cache_slot = 0;
        // Inline-snap: snapshot at boundary during prefill (single snap only;
        // multi-snap "snap=A:1,B:2" is not implemented — use separate SNAPSHOT).
        int  snap_pos  = -1;
        int  snap_slot = -1;
        std::unique_ptr<ActiveDaemonSlot> active_daemon_slot;
        auto activate_daemon_cache_slot = [&](int slot_id) -> bool {
            if (slot_id < 0 || slot_id >= target_cache_slots) return false;
            if (slot_id == active_cache_slot) return true;
            active_daemon_slot.reset();
            active_cache_slot = slot_id;
            if (slot_id > 0) {
                active_daemon_slot = std::make_unique<ActiveDaemonSlot>(
                    cache, prefix_snapshots, sg, draft_sg, proj_sg, feature_mirror,
                    daemon_first_iter, daemon_extra_slots[(size_t)slot_id - 1].get());
            }
            return true;
        };
        auto active_slot_has_scheduler_request = [&]() -> bool {
            return active_cache_slot >= 0
                && active_cache_slot < target_cache_slots
                && daemon_requests[(size_t)active_cache_slot].active;
        };
        auto reject_active_scheduler_slot = [&](const char * command) -> bool {
            if (!active_slot_has_scheduler_request()) return false;
            const DaemonRequestState & req = daemon_requests[(size_t)active_cache_slot];
            std::fprintf(stderr,
                         "[scheduler] %s rejected on active slot %d req=%d\n",
                         command, active_cache_slot, req.request_id);
            stream_emit(-1);
            return true;
        };

        if (daemon_mode) {
            std::string line;
            if (!pop_daemon_line(line)) break;

            current_stream_request_id = 0;
            if (!parse_daemon_request_prefix(line, current_stream_request_id)) {
                std::fprintf(stderr, "[daemon] invalid request id prefix\n");
                stream_emit(-1);
                continue;
            }

            const bool daemon_slot_prefix_present =
                line.rfind("SLOT", 0) == 0 || line.rfind("slot", 0) == 0;
            int daemon_cache_slot = 0;
            if (!parse_daemon_slot_prefix(line, daemon_cache_slot)
                || daemon_cache_slot < 0 || daemon_cache_slot >= target_cache_slots) {
                std::fprintf(stderr, "[daemon] invalid target cache slot %d (slots=%d)\n",
                             daemon_cache_slot, target_cache_slots);
                stream_emit(-1);
                continue;
            }
            if (!activate_daemon_cache_slot(daemon_cache_slot)) {
                std::fprintf(stderr, "[daemon] invalid target cache slot %d (slots=%d)\n",
                             daemon_cache_slot, target_cache_slots);
                stream_emit(-1);
                continue;
            }

            // ── Park/unpark commands (additive on top of latest daemon) ─────
            // "park draft" frees ~3.3GB, "park target" frees ~15GB,
            // "park all" or "park" frees both. ACK via stream_emit(-1).
            auto starts_with = [](const std::string& s, const char* pre) {
                size_t n = std::strlen(pre);
                return s.size() >= n && s.compare(0, n, pre) == 0;
            };
            if (line == "LIST_TARGET_CACHE_SLOTS") {
                std::printf("[daemon] target_cache_slots=%d\n", target_cache_slots);
                std::fflush(stdout);
                stream_emit(-1);
                continue;
            }
            if (line == "LIST_REQUESTS") {
                std::printf("[daemon] requests=");
                bool first = true;
                for (const auto & req : daemon_requests) {
                    if (!req.active) continue;
                    std::printf("%sreq=%d:slot=%d:remaining=%d:emitted=%d:q=%d:epoch=%d",
                                first ? "" : ",",
                                req.request_id, req.slot_id, req.remaining,
                                req.emitted, req.quantum, req.epoch);
                    first = false;
                }
                std::printf("\n");
                std::fflush(stdout);
                stream_emit(-1);
                continue;
            }
            if (line == "CANCEL" || line.rfind("CANCEL ", 0) == 0) {
                int cancel_request_id = current_stream_request_id;
                if (line.rfind("CANCEL ", 0) == 0) {
                    int parsed_cancel_id = 0;
                    if (!parse_int(line.substr(7), parsed_cancel_id)) {
                        std::fprintf(stderr, "[scheduler] CANCEL bad request id\n");
                        stream_emit(-1);
                        continue;
                    }
                    cancel_request_id = parsed_cancel_id;
                }
                if (cancel_request_id <= 0) {
                    std::fprintf(stderr, "[scheduler] CANCEL requires REQ <id> or CANCEL <id>\n");
                    stream_emit(-1);
                    continue;
                }
                bool cancelled = false;
                for (DaemonRequestState & req : daemon_requests) {
                    if (!req.active || req.request_id != cancel_request_id) continue;
                    req.active = false;
                    req.remaining = 0;
                    req.epoch += 1;
                    cancelled = true;
                    std::printf("[scheduler] cancel req=%d slot=%d emitted=%d epoch=%d\n",
                                req.request_id, req.slot_id, req.emitted, req.epoch);
                    break;
                }
                if (!cancelled) {
                    std::printf("[scheduler] cancel miss req=%d\n", cancel_request_id);
                }
                std::fflush(stdout);
                stream_emit(-1);
                continue;
            }
            if (line == "SCHED_STEP") {
                if (!enqueue_next_scheduler_quantum()) {
                    std::printf("[scheduler] idle\n");
                    std::fflush(stdout);
                    stream_emit(-1);
                }
                continue;
            }
            if (line == "SCHED_BATCH_PEEK" || line.rfind("SCHED_BATCH_PEEK ", 0) == 0) {
                int max_batch = target_cache_slots;
                if (line.rfind("SCHED_BATCH_PEEK ", 0) == 0) {
                    if (!parse_int(line.substr(17), max_batch) || max_batch <= 0) {
                        std::fprintf(stderr, "[scheduler] SCHED_BATCH_PEEK needs max_batch > 0\n");
                        stream_emit(-1);
                        continue;
                    }
                }
                DaemonAlignedBucketSelection selection =
                    collect_scheduler_aligned_bucket(max_batch,
                                                     /*tail_only=*/false,
                                                     /*emit_log=*/true);
                const std::vector<DaemonBatchCandidate> & batch = selection.batch;
                std::printf("[scheduler] batch_ready count=%d",
                            (int)batch.size());
                for (const DaemonBatchCandidate & candidate : batch) {
                    std::printf(" req=%d:slot=%d:epoch=%d:n=%d:remaining=%d",
                                candidate.request_id, candidate.slot_id,
                                candidate.epoch, candidate.n_gen,
                                candidate.remaining);
                }
                std::printf("\n");
                std::fflush(stdout);
                stream_emit(-1);
                continue;
            }
            if (line == "SCHED_BATCH_PROBE" || line.rfind("SCHED_BATCH_PROBE ", 0) == 0) {
                int max_batch = target_cache_slots;
                if (line.rfind("SCHED_BATCH_PROBE ", 0) == 0) {
                    if (!parse_int(line.substr(18), max_batch) || max_batch <= 0) {
                        std::fprintf(stderr, "[scheduler] SCHED_BATCH_PROBE needs max_batch > 0\n");
                        stream_emit(-1);
                        continue;
                    }
                }
                if (!run_scheduler_batch_probe(max_batch)) {
                    stream_emit(-1);
                    continue;
                }
                stream_emit(-1);
                continue;
            }
            if (line == "SCHED_BATCH_TARGET_TAIL" || line.rfind("SCHED_BATCH_TARGET_TAIL ", 0) == 0) {
                int max_batch = target_cache_slots;
                if (line.rfind("SCHED_BATCH_TARGET_TAIL ", 0) == 0) {
                    if (!parse_int(line.substr(24), max_batch) || max_batch <= 0) {
                        std::fprintf(stderr, "[scheduler] SCHED_BATCH_TARGET_TAIL needs max_batch > 0\n");
                        stream_emit(-1);
                        continue;
                    }
                }
                if (!run_scheduler_batch_target_step(max_batch, /*tail_only=*/true)) {
                    stream_emit(-1);
                    continue;
                }
                continue;
            }
            if (line == "SCHED_BATCH_TARGET_STEP" || line.rfind("SCHED_BATCH_TARGET_STEP ", 0) == 0) {
                int max_batch = target_cache_slots;
                if (line.rfind("SCHED_BATCH_TARGET_STEP ", 0) == 0) {
                    if (!parse_int(line.substr(24), max_batch) || max_batch <= 0) {
                        std::fprintf(stderr, "[scheduler] SCHED_BATCH_TARGET_STEP needs max_batch > 0\n");
                        stream_emit(-1);
                        continue;
                    }
                }
                if (!run_scheduler_batch_target_step(max_batch, /*tail_only=*/false)) {
                    stream_emit(-1);
                    continue;
                }
                continue;
            }
            if (line == "SCHED_BATCH_DRAIN" || line.rfind("SCHED_BATCH_DRAIN ", 0) == 0) {
                int max_batch = target_cache_slots;
                if (line.rfind("SCHED_BATCH_DRAIN ", 0) == 0) {
                    if (!parse_int(line.substr(18), max_batch) || max_batch <= 0) {
                        std::fprintf(stderr, "[scheduler] SCHED_BATCH_DRAIN needs max_batch > 0\n");
                        stream_emit(-1);
                        continue;
                    }
                }
                daemon_scheduler_drain = true;
                bool progressed = false;
                while (!has_daemon_input()) {
                    if (run_scheduler_batch_target_step(max_batch,
                                                        /*tail_only=*/false,
                                                        /*quiet_unavailable=*/true)) {
                        progressed = true;
                        continue;
                    }
                    if (!enqueue_next_scheduler_quantum()) {
                        daemon_scheduler_drain = false;
                        std::printf("[scheduler] drain complete\n");
                        std::fflush(stdout);
                        if (!progressed) stream_emit(-1);
                    }
                    break;
                }
                continue;
            }
            if (line == "SCHED_DRAIN") {
                daemon_scheduler_drain = true;
                if (!enqueue_next_scheduler_quantum()) {
                    daemon_scheduler_drain = false;
                    std::printf("[scheduler] idle\n");
                    std::fflush(stdout);
                    stream_emit(-1);
                }
                continue;
            }
            if (starts_with(line, "park")) {
                bool want_draft  = (line == "park" || line == "park all" || line == "park draft");
                bool want_target = (line == "park" || line == "park all" || line == "park target");
                if (want_draft && !draft_parked) {
                    destroy_draft_graphs_all_slots();
                    free_draft_weights(dw);
                    draft_parked = true;
                    std::printf("[park] draft released\n"); std::fflush(stdout);
                }
                if (want_target && !target_parked) {
                    destroy_target_graphs_all_slots();
                    free_target_weights(w);
                    target_parked = true;
                    std::printf("[park] target released\n"); std::fflush(stdout);
                }
                stream_emit(-1);
                continue;
            }
            if (line == "free drafter" || line == "drafter free") {
                if (drafter_loaded) {
                    dflash27b::free_drafter(drafter_ctx);
                    drafter_loaded = false;
                    std::printf("[drafter] freed\n"); std::fflush(stdout);
                }
                stream_emit(-1);
                continue;
            }
            if (starts_with(line, "unpark")) {
                bool want_draft  = (line == "unpark" || line == "unpark all" || line == "unpark draft");
                bool want_target = (line == "unpark" || line == "unpark all" || line == "unpark target");
                if (want_target && target_parked) {
                    if (!load_target_gguf(target_path, target_backend, w)) {
                        std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    target_parked = false;
                    std::printf("[unpark] target restored\n"); std::fflush(stdout);
                }
                if (want_draft && draft_parked) {
                    std::string dp(draft_path);
                    bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
                        ? load_draft_gguf(draft_path, draft_backend, dw)
                        : load_draft_safetensors(draft_path, draft_backend, dw);
                    if (!draft_ok) {
                        std::fprintf(stderr, "[unpark] draft: %s\n", dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    if (g_draft_swa_window > 0) {
                        dw.swa_window = g_draft_swa_window;
                        for (int il = 0; il < dw.n_layer - 1; il++)
                            dw.layers[il].is_swa = true;
                    }
                    draft_parked = false;
                    std::printf("[unpark] draft restored\n"); std::fflush(stdout);
                }
                stream_emit(-1);
                continue;
            }

            // ── Compress command (pflash speculative prefill) ───────────────
            // Format: "compress <src_bin_path> <keep_ratio_x1000> <drafter_gguf>"
            //   src_bin_path:   int32 token IDs file (drafter vocab)
            //   keep_ratio_x1000: integer keep ratio × 1000 (e.g. 20 → 0.020)
            //   drafter_gguf:   path to Qwen3-0.6B GGUF (loaded lazily once)
            // Output: stream of int32 compressed token IDs, terminated by -1.
            // Drafter coexists with target+draft via libllama in the same
            // ggml allocator — no park/unpark needed for compression itself.
            if (starts_with(line, "compress ") || starts_with(line, "compress\t")) {
                std::string ppath;
                int  keep_x1000 = 0;
                std::string drafter_path;
                if (!parse_daemon_compress_command(line, ppath, keep_x1000, drafter_path)) {
                    std::fprintf(stderr,
                                 "[compress] bad args, need: <bin> <keep_x1000> <drafter_gguf>\n");
                    stream_emit(-1); continue;
                }
                auto src_ids = read_int32_file(ppath);
                if (src_ids.empty()) {
                    std::fprintf(stderr, "[compress] empty input\n");
                    stream_emit(-1); continue;
                }

                // Park target + draft before allocating drafter context so
                // the drafter's KV (~1.3 GB Q4_0) + scratch (~600 MB) have
                // headroom on a 24 GB card. Restore after scoring.
                bool restore_target = !target_parked;
                bool restore_draft  = !draft_parked;
                if (restore_target) {
                    destroy_target_graphs_all_slots();
                    free_target_weights(w);
                    target_parked = true;
                    std::printf("[compress] target parked\n"); std::fflush(stdout);
                }
                if (restore_draft) {
                    destroy_draft_graphs_all_slots();
                    free_draft_weights(dw);
                    draft_parked = true;
                    std::printf("[compress] draft parked\n"); std::fflush(stdout);
                }
                auto restore_after_compress_failure = [&]() {
                    if (restore_target && target_parked) {
                        if (load_target_gguf(target_path, target_backend, w)) {
                            target_parked = false;
                            std::printf("[compress] target restored after failure\n");
                            std::fflush(stdout);
                        } else {
                            std::fprintf(stderr, "[compress] target restore after failure: %s\n",
                                         dflash27b_last_error());
                        }
                    }
                    if (restore_draft && draft_parked) {
                        if (load_draft_auto(draft_path, draft_backend, dw)) {
                            draft_parked = false;
                            std::printf("[compress] draft restored after failure\n");
                            std::fflush(stdout);
                        } else {
                            std::fprintf(stderr, "[compress] draft restore after failure: %s\n",
                                         dflash27b_last_error());
                        }
                    }
                };

                if (!drafter_loaded) {
                    if (!dflash27b::load_drafter(drafter_path.c_str(), /*gpu_layers=*/999, drafter_ctx)) {
                        std::fprintf(stderr, "[compress] load_drafter failed: %s\n",
                                     dflash27b_last_error());
                        restore_after_compress_failure();
                        stream_emit(-1); continue;
                    }
                    drafter_loaded = true;
                    std::printf("[drafter] loaded %s (n_layer=%d n_head=%d n_head_kv=%d)\n",
                                drafter_path.c_str(), drafter_ctx.weights.n_layer,
                                drafter_ctx.weights.n_head, drafter_ctx.weights.n_head_kv);
                    std::fflush(stdout);
                }

                float keep = (float)keep_x1000 / 1000.0f;
                auto compressed = dflash27b::drafter_score_and_compress(
                    drafter_ctx, src_ids, keep);
                std::printf("[compress] %zu -> %zu tokens (keep_ratio=%.3f)\n",
                            src_ids.size(), compressed.size(), keep);
                std::fflush(stdout);

                // Restore daemon state for the (almost certainly) following
                // generate command.
                if (restore_target) {
                    if (!load_target_gguf(target_path, target_backend, w)) {
                        std::fprintf(stderr, "[compress] target restore: %s\n",
                                     dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    target_parked = false;
                    std::printf("[compress] target restored\n"); std::fflush(stdout);
                }
                if (restore_draft) {
                    if (!load_draft_auto(draft_path, draft_backend, dw)) {
                        std::fprintf(stderr, "[compress] draft restore: %s\n",
                                     dflash27b_last_error());
                        stream_emit(-1); continue;
                    }
                    if (g_draft_swa_window > 0) {
                        dw.swa_window = g_draft_swa_window;
                        for (int il = 0; il < dw.n_layer - 1; il++)
                            dw.layers[il].is_swa = true;
                    }
                    draft_parked = false;
                    std::printf("[compress] draft restored\n"); std::fflush(stdout);
                }

                for (int32_t t : compressed) stream_emit(t);
                stream_emit(-1);
                continue;
            }

            // ── Prefix-cache snapshot commands (#59) ──────────────────────
            // Check longer prefixes before shorter ones to avoid mis-dispatch
            // (SNAPSHOT_THIN must come before SNAPSHOT, RESTORE_CHAIN before RESTORE).
            if (line.rfind("SNAPSHOT_THIN ", 0) == 0) {
                int slot = -1, kv_start = -1, kv_end = -1;
                if (std::sscanf(line.c_str() + 14, "%d %d %d", &slot, &kv_start, &kv_end) != 3
                    || slot < 0 || slot >= PREFIX_CACHE_SLOTS) {
                    std::fprintf(stderr, "[snap] SNAPSHOT_THIN bad args\n");
                    continue;
                }
                if (!snapshot_target_cache_thin(w, cache, backend, kv_start, kv_end,
                                                 prefix_snapshots[slot])) {
                    std::fprintf(stderr, "[snap] thin failed slot=%d: %s\n", slot,
                                 dflash27b_last_error());
                    continue;
                }
                std::printf("[snap] thin slot=%d kv=%d,%d\n", slot, kv_start, kv_end);
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("SNAPSHOT ", 0) == 0) {
                int slot = -1;
                if (std::sscanf(line.c_str() + 9, "%d", &slot) != 1
                    || slot < 0 || slot >= PREFIX_CACHE_SLOTS) {
                    std::fprintf(stderr, "[snap] invalid slot %d\n", slot);
                    continue;
                }
                if (!snapshot_target_cache(w, cache, backend, prefix_snapshots[slot])) {
                    std::fprintf(stderr, "[snap] failed slot=%d: %s\n", slot, dflash27b_last_error());
                    continue;
                }
                std::printf("[snap] slot=%d cur_pos=%d\n", slot, prefix_snapshots[slot].cur_pos);
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("FREE_SNAPSHOT ", 0) == 0) {
                int slot = -1;
                if (std::sscanf(line.c_str() + 14, "%d", &slot) != 1
                    || slot < 0 || slot >= PREFIX_CACHE_SLOTS) continue;
                free_prefix_snapshot(prefix_snapshots[slot]);
                std::printf("[snap] freed slot=%d\n", slot);
                std::fflush(stdout);
                continue;
            }
            if (line == "LIST_SLOTS") {
                std::printf("[snap] slots=");
                bool first = true;
                for (int i = 0; i < PREFIX_CACHE_SLOTS; i++) {
                    if (prefix_snapshots[i].ctx != nullptr) {
                        std::printf("%s%d", first ? "" : ",", i);
                        first = false;
                    }
                }
                std::printf("\n");
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("RESTORE_CHAIN ", 0) == 0) {
                if (reject_active_scheduler_slot("RESTORE_CHAIN")) continue;
                // Format: RESTORE_CHAIN <thick_slot> <thin_slot_list> <prompt_file> <n_gen>
                // <thin_slot_list> is "0,1,2" or "-" for empty.
                int  thick_slot_local = -2;
                char thin_str[256]    = {0};
                char ppath[1024]      = {0};
                int  n_gen_local      = 0;
                if (std::sscanf(line.c_str() + 14, "%d %255s %1023s %d",
                                &thick_slot_local, thin_str, ppath, &n_gen_local) != 4) {
                    std::fprintf(stderr, "[snap] RESTORE_CHAIN bad args\n");
                    stream_emit(-1);
                    continue;
                }
                // Validate thick_slot (-1 = none).
                if (thick_slot_local != -1
                    && (thick_slot_local < 0 || thick_slot_local >= PREFIX_CACHE_SLOTS
                        || prefix_snapshots[thick_slot_local].ctx == nullptr
                        || prefix_snapshots[thick_slot_local].is_thin)) {
                    std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thick slot=%d\n", thick_slot_local);
                    stream_emit(-1);
                    continue;
                }
                // Parse thin slot list. Strict: every comma-separated token
                // must be a valid non-negative integer (rejects "1,foo,3",
                // empty entries "1,,3", trailing junk). Codex review fix.
                std::vector<int> thin_ids_local;
                bool thin_parse_ok = true;
                if (std::strcmp(thin_str, "-") != 0 && thin_str[0] != '\0') {
                    const char * p = thin_str;
                    while (*p && thin_parse_ok) {
                        char * end = nullptr;
                        long id_l = std::strtol(p, &end, 10);
                        if (end == p) {
                            std::fprintf(stderr,
                                "[snap] RESTORE_CHAIN malformed thin list near '%s'\n", p);
                            thin_parse_ok = false; break;
                        }
                        int id = (int)id_l;
                        if (id < 0 || id >= PREFIX_CACHE_SLOTS
                            || prefix_snapshots[id].ctx == nullptr
                            || !prefix_snapshots[id].is_thin) {
                            std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thin slot=%d\n", id);
                            thin_parse_ok = false; break;
                        }
                        thin_ids_local.push_back(id);
                        if (*end == '\0') break;
                        if (*end != ',') {
                            std::fprintf(stderr,
                                "[snap] RESTORE_CHAIN expected ',' after slot %d, got '%c'\n",
                                id, *end);
                            thin_parse_ok = false; break;
                        }
                        p = end + 1;
                        if (*p == '\0' || *p == ',') {
                            std::fprintf(stderr,
                                "[snap] RESTORE_CHAIN empty thin slot entry\n");
                            thin_parse_ok = false; break;
                        }
                    }
                }
                if (!thin_parse_ok) {
                    stream_emit(-1);
                    continue;
                }
                n_gen                    = n_gen_local;
                prompt_file_str          = ppath;
                prompt_path              = prompt_file_str.c_str();
                chain_restore_requested  = true;
                chain_thick_slot         = thick_slot_local;
                chain_thin_ids           = std::move(thin_ids_local);
                // Fall through into the existing cache-rebuild + prefill path.
            } else if (line.rfind("RESTORE ", 0) == 0) {
                if (reject_active_scheduler_slot("RESTORE")) continue;
                int slot = -1;
                char ppath[1024];
                if (std::sscanf(line.c_str() + 8, "%d %1023s %d", &slot, ppath, &n_gen) != 3
                    || slot < 0 || slot >= PREFIX_CACHE_SLOTS
                    || prefix_snapshots[slot].ctx == nullptr) {
                    std::fprintf(stderr, "[snap] RESTORE bad args or empty slot %d\n", slot);
                    stream_emit(-1);
                    continue;
                }
                prompt_file_str = ppath;
                prompt_path = prompt_file_str.c_str();
                restore_from_slot = true;
                restore_slot_id   = slot;
                // Parse optional inline-snap suffix: snap=<pos>:<slot_id>
                if (const char * sp = std::strstr(line.c_str(), "snap=")) {
                    if (std::sscanf(sp, "snap=%d:%d", &snap_pos, &snap_slot) != 2
                        || snap_slot < 0 || snap_slot >= PREFIX_CACHE_SLOTS) {
                        std::fprintf(stderr, "[snap] bad inline-snap arg\n");
                        snap_pos = -1; snap_slot = -1;
                    }
                }
                // Fall through into the existing prefill path; the cache reset
                // and restore happen after the cache rebuild block below.
            } else if (parse_daemon_start_command(line, prompt_file_str,
                                                  scheduler_total_gen,
                                                  scheduler_quantum)) {
                if (current_stream_request_id <= 0) {
                    std::fprintf(stderr, "[scheduler] START requires REQ <id>\n");
                    stream_emit(-1);
                    continue;
                }
                if (scheduler_total_gen <= 0 || scheduler_quantum <= 0) {
                    std::fprintf(stderr, "[scheduler] START needs total_gen and quantum > 0\n");
                    stream_emit(-1);
                    continue;
                }
                if (!daemon_slot_prefix_present) {
                    int admitted_slot = -1;
                    for (int sid = 0; sid < target_cache_slots; sid++) {
                        if (!daemon_requests[(size_t)sid].active) {
                            admitted_slot = sid;
                            break;
                        }
                    }
                    if (admitted_slot < 0) {
                        std::fprintf(stderr, "[scheduler] no free target cache slots\n");
                        stream_emit(-1);
                        continue;
                    }
                    if (!activate_daemon_cache_slot(admitted_slot)) {
                        std::fprintf(stderr, "[scheduler] failed to activate slot %d\n",
                                     admitted_slot);
                        stream_emit(-1);
                        continue;
                    }
                    std::printf("[scheduler] admit req=%d slot=%d\n",
                                current_stream_request_id, active_cache_slot);
                    std::fflush(stdout);
                }
                if (daemon_requests[(size_t)active_cache_slot].active) {
                    std::fprintf(stderr, "[scheduler] slot %d already has active request %d\n",
                                 active_cache_slot,
                                 daemon_requests[(size_t)active_cache_slot].request_id);
                    stream_emit(-1);
                    continue;
                }
                n_gen = std::min(scheduler_total_gen, scheduler_quantum);
                prompt_path = prompt_file_str.c_str();
                scheduler_start_requested = true;
            } else if (parse_daemon_continue_command(line, n_gen)) {
                continue_decode_requested = true;
                if (cache.cur_pos <= 0 || cache.last_tok < 0) {
                    std::fprintf(stderr,
                                 "[daemon] CONTINUE requested before slot has active decode state\n");
                    stream_emit(-1);
                    continue;
                }
                if (n_gen <= 0) {
                    std::fprintf(stderr, "[daemon] CONTINUE needs n_gen > 0\n");
                    stream_emit(-1);
                    continue;
                }
                const DaemonRequestState & req = daemon_requests[(size_t)active_cache_slot];
                if (req.active) {
                    if (current_stream_request_id <= 0
                        || req.request_id != current_stream_request_id) {
                        std::fprintf(stderr,
                                     "[scheduler] stale CONTINUE rejected slot=%d expected_req=%d got_req=%d\n",
                                     active_cache_slot, req.request_id,
                                     current_stream_request_id);
                        stream_emit(-1);
                        continue;
                    }
                    scheduler_continue_requested = true;
                }
            } else {
                // Legacy: bare `<prompt_file> <n_gen>` line — full reset path.
                if (reject_active_scheduler_slot("legacy generate")) continue;
                std::string ppath;
                if (!parse_daemon_generate_command(line, ppath, n_gen)) continue;
                prompt_file_str = ppath;
                prompt_path = prompt_file_str.c_str();
                // Parse optional inline-snap suffix: snap=<pos>:<slot_id>
                if (const char * sp = std::strstr(line.c_str(), "snap=")) {
                    if (std::sscanf(sp, "snap=%d:%d", &snap_pos, &snap_slot) != 2
                        || snap_slot < 0 || snap_slot >= PREFIX_CACHE_SLOTS) {
                        std::fprintf(stderr, "[snap] bad inline-snap arg\n");
                        snap_pos = -1; snap_slot = -1;
                    }
                }
            }

            // Reset all per-request slot state between fresh requests. Prefix
            // snapshots intentionally survive until FREE_SNAPSHOT or shutdown:
            // SNAPSHOT/RESTORE is an explicit cross-request API contract.
            // The first request still takes the existing promote path from
            // prefill-only to full cache allocation.
            if (!continue_decode_requested && !daemon_first_iter) {
                scrub_daemon_slot_state(cache, sg, draft_sg, proj_sg,
                                        feature_mirror, daemon_first_iter);
            }
            if (!continue_decode_requested) daemon_first_iter = false;

            // After cache is fresh, optionally restore from snapshot.
            if (restore_from_slot) {
                if (!restore_target_cache(prefix_snapshots[restore_slot_id], cache)) {
                    std::fprintf(stderr, "[snap] restore failed: %s\n", dflash27b_last_error());
                    stream_emit(-1);
                    continue;
                }
                std::printf("[snap] restored slot=%d cur_pos=%d\n",
                            restore_slot_id, cache.cur_pos);
                std::fflush(stdout);
            }

            // After cache is fresh, optionally apply chain restore.
            if (chain_restore_requested) {
                const PrefixSnapshot * thick_ptr =
                    (chain_thick_slot == -1) ? nullptr : &prefix_snapshots[chain_thick_slot];
                std::vector<const PrefixSnapshot *> thin_ptrs;
                for (int id : chain_thin_ids) thin_ptrs.push_back(&prefix_snapshots[id]);
                if (!restore_target_cache_chain(thick_ptr,
                                                 thin_ptrs.empty() ? nullptr : thin_ptrs.data(),
                                                 (int)thin_ptrs.size(),
                                                 cache)) {
                    std::fprintf(stderr, "[snap] RESTORE_CHAIN failed: %s\n", dflash27b_last_error());
                    stream_emit(-1);
                    continue;
                }
                std::printf("[snap] chain restored thick=%d thins=%zu cur_pos=%d\n",
                            chain_thick_slot, thin_ptrs.size(), cache.cur_pos);
                std::fflush(stdout);
            }
        }

        std::vector<int32_t> out_all;
        int committed = 0;
        int32_t last_tok = -1;

        if (continue_decode_requested) {
            committed = cache.cur_pos;
            last_tok = cache.last_tok;
            if (committed + n_gen + q_len > max_ctx) {
                std::fprintf(stderr,
                             "continue cur_pos (%d) + gen (%d) + block (%d) = %d exceeds max_ctx (%d)\n",
                             committed, n_gen, q_len, committed + n_gen + q_len, max_ctx);
                stream_emit(-1);
                continue;
            }
            std::printf("[continue] cur_pos=%d last_tok=%d n_gen=%d\n",
                        committed, last_tok, n_gen);
        } else {
        std::vector<int32_t> prompt;
        if (synthetic_prompt_tokens > 0 && !prompt_path) {
            prompt.assign((size_t)synthetic_prompt_tokens,
                          (int32_t)synthetic_prompt_token);
            std::printf("[prompt] synthetic %zu tokens token=%d\n",
                        prompt.size(), synthetic_prompt_token);
        } else {
            prompt = read_int32_file(prompt_path);
        }
        if (prompt.empty()) {
            std::fprintf(stderr, "empty prompt\n");
            if (daemon_mode) { stream_emit(-1); continue; } else return 1;
        }
        if (synthetic_prompt_tokens <= 0 || prompt_path) {
            std::printf("[prompt] %zu tokens\n", prompt.size());
        }

        if ((int)prompt.size() + n_gen + q_len > max_ctx) {
            std::fprintf(stderr, "prompt (%zu) + gen (%d) + block (%d) = %d exceeds max_ctx (%d)\n",
                         prompt.size(), n_gen, q_len, (int)prompt.size() + n_gen + q_len, max_ctx);
            if (daemon_mode) { stream_emit(-1); continue; } else return 1;
        }

        std::vector<float>   embed_buf(hidden);
        out_all = prompt;
        committed = 0;
        last_tok = -1;

    // ── Prefill: two modes available ────────────────────────────────────
    // Layer-segmented: iterate layers (outer) × token chunks (inner).
    //   Reads each layer's weights once per chunk instead of once per full
    //   forward. Better L2 cache warmth on weights across token chunks.
    // Token-segmented (legacy): iterate token chunks (outer) × layers (inner).
    //   Matches llama.cpp's n_ubatch behavior.
    // Controlled by DFLASH27B_LAYER_PREFILL=1 env var (default: off).
    // Currently faster only at short contexts (<8K); at longer contexts the
    // graph rebuild overhead per layer dominates.
    const int prompt_len_auto = (int)prompt.size();
    bool layer_prefill = false;
    if (const char * s = std::getenv("DFLASH27B_LAYER_PREFILL")) {
        layer_prefill = (std::atoi(s) != 0);
    }

    // ── Layer-segmented prefill ─────────────────────────────────────────
    if (layer_prefill) {
        int layer_ubatch_env = 384;
        if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
            layer_ubatch_env = std::max(1, std::atoi(s));
        }
        const int LAYER_UBATCH = layer_ubatch_env;
        std::printf("[prefill] layer-segmented ubatch=%d\n", LAYER_UBATCH);
        const int prompt_len = (int)prompt.size();

        // Allocate ping-pong activation buffers [hidden, prompt_len]
        ggml_init_params act_ip{};
        act_ip.mem_size   = (size_t)4 * ggml_tensor_overhead();
        act_ip.mem_buffer = nullptr;
        act_ip.no_alloc   = true;
        ggml_context * act_ctx = ggml_init(act_ip);
        ggml_tensor * act_in  = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, prompt_len);
        ggml_set_name(act_in, "act_in");
        ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, prompt_len);
        ggml_set_name(act_out, "act_out");
        ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, backend);
        if (!act_buf) {
            std::fprintf(stderr, "activation buffer alloc failed\n"); return 1;
        }

        // Embed all prompt tokens into act_in (batched)
        {
            const int EMBED_BATCH = 4096;
            std::vector<float> emb_buf((size_t)hidden * EMBED_BATCH);
            for (int i = 0; i < prompt_len; i += EMBED_BATCH) {
                const int n = std::min(EMBED_BATCH, prompt_len - i);
                if (!w.embedder.embed(prompt.data() + i, n, emb_buf.data())) return 1;
                ggml_backend_tensor_set(act_in, emb_buf.data(),
                                        (size_t)i * act_in->nb[1],
                                        sizeof(float) * hidden * n);
            }
        }

        auto t_pf0 = std::chrono::steady_clock::now();
        StepGraph lsg;

        for (int il = 0; il < w.n_layer; il++) {
            const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);

            for (int start = 0; start < prompt_len; start += LAYER_UBATCH) {
                const int n_tokens = std::min(LAYER_UBATCH, prompt_len - start);
                const int kv_len   = start + n_tokens;
                const bool with_mask = (g_kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

                if (!build_layer_step(lsg, w, cache, backend, il,
                                      act_in, act_out, start, n_tokens,
                                      start, with_mask, true)) {
                    std::fprintf(stderr, "layer-seg build layer=%d @%d\n", il, start);
                    return 1;
                }

                // M-RoPE positions for this chunk (FA layers only)
                if (is_attn && lsg.positions) {
                    std::vector<int32_t> pos_buf((size_t)4 * n_tokens, 0);
                    for (int i = 0; i < n_tokens; i++) {
                        const int p = start + i;
                        pos_buf[0 * n_tokens + i] = p;
                        pos_buf[1 * n_tokens + i] = p;
                        pos_buf[2 * n_tokens + i] = p;
                        pos_buf[3 * n_tokens + i] = 0;
                    }
                    ggml_backend_tensor_set(lsg.positions, pos_buf.data(), 0,
                                            sizeof(int32_t) * pos_buf.size());
                }

                if (is_attn && with_mask && lsg.attn_mask) {
                    std::vector<uint16_t> mask_buf;
                    build_causal_mask(mask_buf, kv_len, n_tokens, /*kv_start=*/start);
                    ggml_backend_tensor_set(lsg.attn_mask, mask_buf.data(), 0,
                                            sizeof(uint16_t) * mask_buf.size());
                }

                auto st = ggml_backend_graph_compute(backend, lsg.gf);
                if (st != GGML_STATUS_SUCCESS) {
                    std::fprintf(stderr, "layer-seg compute layer=%d @%d\n", il, start);
                    return 1;
                }
            }

            // Swap activation buffers after each layer
            std::swap(act_in, act_out);
        }

        // Final norm + LM head on last token only
        {
            step_graph_free(lsg);
            ggml_init_params fip{};
            fip.mem_size   = 512 * 1024 * 1024;
            fip.mem_buffer = nullptr;
            fip.no_alloc   = true;
            lsg.ctx = ggml_init(fip);

            ggml_tensor * last_row = ggml_view_1d(lsg.ctx, act_in,
                hidden, (size_t)(prompt_len - 1) * act_in->nb[1]);
            ggml_tensor * normed   = ggml_rms_norm(lsg.ctx, last_row, DFLASH27B_RMS_EPS);
            normed = ggml_mul(lsg.ctx, normed, w.out_norm);
            ggml_tensor * logits   = ggml_mul_mat(lsg.ctx, w.output, normed);
            ggml_set_name(logits, "logits");
            ggml_set_output(logits);
            lsg.logits = logits;
            lsg.gf = ggml_new_graph_custom(lsg.ctx, 1024, false);
            ggml_build_forward_expand(lsg.gf, logits);

            if (!lsg.alloc) {
                lsg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            }
            if (!ggml_gallocr_alloc_graph(lsg.alloc, lsg.gf)) {
                std::fprintf(stderr, "final norm alloc failed\n"); return 1;
            }

            auto st = ggml_backend_graph_compute(backend, lsg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "final norm compute failed\n"); return 1;
            }

            std::vector<float> logits_buf(vocab, 0.0f);
            ggml_backend_tensor_get(lsg.logits, logits_buf.data(), 0,
                                    sizeof(float) * vocab);
            last_tok = argmax_f32(logits_buf.data(), vocab);
            step_graph_destroy(lsg);
        }

        committed = prompt_len;
        ggml_backend_buffer_free(act_buf);
        ggml_free(act_ctx);

        auto t_pf1 = std::chrono::steady_clock::now();
        std::printf("[prefill] layer-seg %d tokens in %.2f s, last_tok=%d\n",
                    committed,
                    std::chrono::duration<double>(t_pf1 - t_pf0).count(),
                    last_tok);

        // Promote prefill-only cache to full decode cache
        auto t_mig0 = std::chrono::steady_clock::now();
        step_graph_destroy(sg);
        if (!migrate_prefill_cache(w, max_ctx, max_verify_tokens, target_backend, cache)) {
            std::fprintf(stderr, "cache migration: %s\n", dflash27b_last_error());
            return 1;
        }
        auto t_mig1 = std::chrono::steady_clock::now();
        std::printf("[migrate] %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t_mig1 - t_mig0).count());
    }
    // ── Token-segmented prefill (legacy) ────────────────────────────────
    if (!layer_prefill) {
    int prefill_ubatch_env = (prompt_len_auto > 2048) ? 384 : 16;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        prefill_ubatch_env = std::max(1, std::atoi(s));
    }
    const int PREFILL_UBATCH = prefill_ubatch_env;
    std::printf("[prefill] token-seg ubatch=%d\n", PREFILL_UBATCH);
    auto t_pf0 = std::chrono::steady_clock::now();
    std::vector<uint16_t> pf_mask_buf;
    std::vector<float>    pf_embed_buf;
    std::vector<int32_t>  pf_pos_buf;
    std::vector<float>    pf_logits_buf;
    const int prompt_len     = (int)prompt.size();
    const int prefill_start  = cache.cur_pos;   // 0 for fresh cache; >0 after snapshot restore
    for (int start = prefill_start; start < prompt_len; start += PREFILL_UBATCH) {
        int n_tokens = std::min(PREFILL_UBATCH, prompt_len - start);

        // Inline-snap: if snap_pos == start exactly, fire snapshot before any
        // prefill work this iteration, then continue with the full ubatch.
        if (snap_pos >= 0 && snap_pos == start) {
            cache.cur_pos = start;
            if (snap_slot >= 0) {
                if (snapshot_target_cache(w, cache, backend, prefix_snapshots[snap_slot])) {
                    std::printf("[snap] inline slot=%d cur_pos=%d\n", snap_slot, start);
                    std::fflush(stdout);
                } else {
                    std::fprintf(stderr, "[snap] inline snap failed slot=%d: %s\n",
                                 snap_slot, dflash27b_last_error());
                }
            }
            snap_pos = -1; snap_slot = -1;   // consume
            // n_tokens is unchanged; continue prefilling this ubatch.
        }

        // Inline-snap: if snap_pos falls inside this ubatch, clip n_tokens to
        // land exactly at snap_pos so the snapshot captures the right boundary.
        bool fire_snap_after = false;
        if (snap_pos > start && snap_pos <= start + n_tokens) {
            n_tokens = snap_pos - start;   // land exactly at snap_pos
            fire_snap_after = (n_tokens > 0);
            if (n_tokens == 0) {
                // snap_pos == start already handled above; shouldn't reach here.
                snap_pos = -1; snap_slot = -1;
            }
        }

        const int kv_len   = start + n_tokens;
        const bool pf_with_mask = (g_kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);
        if (!build_target_step(sg, w, cache, backend,
                                /*kv_start=*/start, /*n_tokens=*/n_tokens,
                                /*with_mask=*/pf_with_mask, /*capture=*/true)) {
            std::fprintf(stderr, "prefill build @%d\n", start); return 1;
        }

        pf_embed_buf.assign((size_t)hidden * n_tokens, 0.0f);
        if (!w.embedder.embed(prompt.data() + start, n_tokens, pf_embed_buf.data())) return 1;
        ggml_backend_tensor_set(sg.inp_embed, pf_embed_buf.data(), 0,
                                sizeof(float) * pf_embed_buf.size());

        // M-RoPE 4D text layout: [axis0 × n_tokens, axis1 × n_tokens,
        // axis2 × n_tokens, axis3 × n_tokens]. First 3 axes hold absolute
        // positions, axis 3 is 0 for plain text.
        pf_pos_buf.assign((size_t)4 * n_tokens, 0);
        for (int i = 0; i < n_tokens; i++) {
            const int p = start + i;
            pf_pos_buf[0 * n_tokens + i] = p;
            pf_pos_buf[1 * n_tokens + i] = p;
            pf_pos_buf[2 * n_tokens + i] = p;
            pf_pos_buf[3 * n_tokens + i] = 0;
        }
        ggml_backend_tensor_set(sg.positions, pf_pos_buf.data(), 0,
                                sizeof(int32_t) * pf_pos_buf.size());

        // Causal mask required when n_tokens > 1 OR when the TBQ FA kernel
        // is active (which pads kv_len to 256 and needs -inf on the padding
        // positions even for a single query).
        if (pf_with_mask) {
            build_causal_mask(pf_mask_buf, kv_len, n_tokens, /*kv_start=*/start);
            ggml_backend_tensor_set(sg.attn_mask, pf_mask_buf.data(), 0,
                                    sizeof(uint16_t) * pf_mask_buf.size());
        }

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "prefill compute @%d\n", start); return 1; }

        // Only need the last position's logits to seed decode.
        pf_logits_buf.assign(vocab, 0.0f);
        const size_t last_row_off = (size_t)(n_tokens - 1) * vocab * sizeof(float);
        ggml_backend_tensor_get(sg.logits, pf_logits_buf.data(), last_row_off,
                                sizeof(float) * vocab);
        last_tok = argmax_f32(pf_logits_buf.data(), vocab);
        committed = start + n_tokens;

        // Fire inline snapshot after compute, so cache boundary is exact.
        if (fire_snap_after) {
            cache.cur_pos  = committed;
            cache.last_tok = last_tok;
            if (snap_slot >= 0) {
                if (snapshot_target_cache(w, cache, backend, prefix_snapshots[snap_slot])) {
                    std::printf("[snap] inline slot=%d cur_pos=%d\n", snap_slot, committed);
                    std::fflush(stdout);
                } else {
                    std::fprintf(stderr, "[snap] inline snap failed slot=%d: %s\n",
                                 snap_slot, dflash27b_last_error());
                }
            }
            snap_pos = -1; snap_slot = -1;   // consume
            // Adjust loop increment: next iteration must start at committed,
            // not at (start + PREFILL_UBATCH). Override via start arithmetic:
            // the for-loop does start += PREFILL_UBATCH, so back-adjust.
            start = committed - PREFILL_UBATCH;
        }
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    // If prefill was a no-op due to a snapshot RESTORE (cache.cur_pos already
    // covers the prompt), seed last_tok from the restored cache so the decode
    // loop has a valid starting token. Detected by prefill_start == prompt_len:
    // the for loop ran zero iterations and `committed` stayed at 0.
    if (last_tok == -1 && cache.last_tok != -1 && prefill_start == prompt_len) {
        last_tok  = cache.last_tok;
        committed = prompt_len;
    }
    std::printf("[prefill] %d tokens in %.2f s, last_tok=%d\n",
                committed,
                std::chrono::duration<double>(t_pf1 - t_pf0).count(),
                last_tok);

    // Promote prefill-only cache to full decode cache with rollback tensors.
    // Copies KV, SSM/conv state, and target_feat device→device (~1 ms).
    auto t_mig0 = std::chrono::steady_clock::now();
    step_graph_destroy(sg);
    if (!migrate_prefill_cache(w, max_ctx, max_verify_tokens, target_backend, cache)) {
        std::fprintf(stderr, "cache migration: %s\n", dflash27b_last_error());
        return 1;
    }
    auto t_mig1 = std::chrono::steady_clock::now();
    std::printf("[migrate] %.2f ms\n",
                std::chrono::duration<double, std::milli>(t_mig1 - t_mig0).count());
    } // end if (!layer_prefill)
        } // end fresh prompt/prefill path

    if (draft_feature_mirror) {
        if (!feature_mirror.target_feat || feature_mirror.cap != cache.target_feat_cap) {
            if (!draft_feature_mirror_init(feature_mirror, draft_backend,
                                           draft_gpu, target_gpu,
                                           cache.target_feat_cap)) {
                std::fprintf(stderr, "draft feature mirror init failed\n");
                return 1;
            }
            std::printf("[draft-mirror] init cap=%d type=f32 device=%d target_device=%d\n",
                        feature_mirror.cap, draft_gpu, target_gpu);
        }
        if (!draft_feature_mirror_sync_tail(cache, feature_mirror, committed)) {
            std::fprintf(stderr, "draft feature mirror initial sync failed\n");
            return 1;
        }
        std::printf("[draft-mirror] synced tail committed=%d cap=%d\n",
                    committed, feature_mirror.cap);
    }

    // ── DFlash decode loop
    int n_draft_steps = 0, n_accept_sum = 0, n_generated = 0;
    bool hit_eos_decode = false;
    std::vector<float>   noise_embed_buf(hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len), target_tok(q_len);
    std::vector<float>   draft_logits_buf((size_t)vocab * q_len);
    // Sized for the max of chain q_len and DDTree flat tree size (budget+1).
    const int verify_max_tokens = std::max(q_len, ddtree_mode ? ddtree_budget + 1 : q_len);
    std::vector<float>   verify_logits_buf((size_t)vocab * verify_max_tokens);
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos_q_buf(q_len), pos_k_buf(max_ctx + q_len);
    std::vector<int32_t> pos4_buf(4 * q_len);

    auto t_gen0 = std::chrono::steady_clock::now();

    // Per-phase timing accumulators (microseconds)
    double tt_draft_build = 0, tt_draft_copy_feat = 0, tt_draft_set = 0,
           tt_draft_compute = 0, tt_draft_bridge = 0, tt_draft_logits = 0,
           tt_snap = 0, tt_verify_build = 0, tt_verify_set = 0,
           tt_verify_compute = 0, tt_verify_logits = 0,
           tt_accept = 0, tt_restore = 0,
           tt_replay_build = 0, tt_replay_set = 0, tt_replay_compute = 0,
           tt_replay_logits = 0, tt_mirror_sync = 0;
    auto sync_us = [&](){
        if (timing_debug) {
            ggml_backend_synchronize(target_backend);
            if (split_gpus) ggml_backend_synchronize(draft_backend);
        }
        return std::chrono::steady_clock::now();
    };
    auto sync_draft_feature_mirror = [&](int start_pos, int n_tokens) -> bool {
        if (!draft_feature_mirror || !feature_mirror.target_feat || n_tokens <= 0) {
            return true;
        }
        auto t0 = sync_us();
        const bool ok = draft_feature_mirror_sync_range(cache, feature_mirror,
                                                        start_pos, n_tokens);
        auto t1 = sync_us();
        tt_mirror_sync += std::chrono::duration<double, std::micro>(t1 - t0).count();
        return ok;
    };

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        auto T0 = sync_us();

        // 1) Noise block [last_tok, MASK*15]
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = mask_tok;
        if (!w.embedder.embed(noise_ids.data(), q_len, noise_embed_buf.data())) return 1;

        // Draft target-attention window. The draft transformer attends over
        // a slice of the history captured in cache.target_feat; this caps the
        // slice so the draft's [5*hidden × ctx_len] target_hidden_cat tensor
        // stays bounded even at 16K+ target context. Tokens older than the
        // window are invisible to the draft but still in the target's KV
        // cache (the target verify uses the full history).
        constexpr int DRAFT_CTX_MAX = 2048;
        const int draft_ctx   = std::min(committed,
            std::max(DRAFT_CTX_MAX, g_draft_ctx_max));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_mirror, committed, draft_ctx, mirror_slot0);
        const bool draft_hidden_bridge = split_gpus;

        // 2) Draft forward
        if (!build_draft_step(draft_sg, dw, draft_hidden_bridge ? nullptr : &w,
                              draft_backend, /*ctx_len=*/draft_ctx,
                              use_mirror_view ? &feature_mirror : nullptr,
                              committed)) {
            std::fprintf(stderr, "draft build failed\n"); return 1;
        }
        auto T_draft_build = sync_us();
        tt_draft_build += std::chrono::duration<double, std::micro>(T_draft_build - T0).count();

        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed_buf.data(), 0,
                                sizeof(float) * noise_embed_buf.size());

        if (!use_mirror_view) {
            // target_hidden_cat: copy the draft-window slice of cache.target_feat
            // (positions draft_start..committed) directly device-to-device.
            // cache.target_feat is a ring of `target_feat_cap` bf16 slots, so
            // positions map via `pos % cap`. If the draft window straddles the
            // wrap boundary we split the bf16-to-f32 widen into two kernel calls.
            const size_t fc_in    = (size_t)5 * hidden;
            const int    cap      = cache.target_feat_cap;
            const size_t elt_feat = ggml_element_size(cache.target_feat);
            const int    slot0    = draft_start % cap;
            const int    pre_n    = std::min(draft_ctx, cap - slot0);
            const int    post_n   = draft_ctx - pre_n;

            cudaSetDevice(draft_gpu);
            dflash27b_launch_bf16_to_f32(
                (const char *)cache.target_feat->data + (size_t)slot0 * elt_feat * fc_in,
                draft_sg.target_hidden_cat->data,
                (size_t)pre_n * fc_in,
                nullptr);
            if (post_n > 0) {
                dflash27b_launch_bf16_to_f32(
                    (const char *)cache.target_feat->data,
                    (char *)draft_sg.target_hidden_cat->data + (size_t)pre_n * fc_in * sizeof(float),
                    (size_t)post_n * fc_in,
                    nullptr);
            }
        }
        auto T_draft_copy = sync_us();
        tt_draft_copy_feat += std::chrono::duration<double, std::micro>(T_draft_copy - T_draft_build).count();

        for (int i = 0; i < q_len; i++) pos_q_buf[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k_buf[i] = i;
        ggml_backend_tensor_set(draft_sg.positions,   pos_q_buf.data(), 0, sizeof(int32_t) * q_len);
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k_buf.data(), 0, sizeof(int32_t) * (draft_ctx + q_len));
        if (draft_sg.attn_mask) {
            build_draft_swa_mask(mask_buf, draft_ctx, q_len, dw.swa_window);
            ggml_backend_tensor_set(draft_sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }
        auto T_draft_set = sync_us();
        tt_draft_set += std::chrono::duration<double, std::micro>(T_draft_set - T_draft_copy).count();

        auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "draft compute %d\n", (int)st); return 1; }
        auto T_draft_compute = sync_us();
        tt_draft_compute += std::chrono::duration<double, std::micro>(T_draft_compute - T_draft_set).count();

        if (draft_hidden_bridge) {
            if (!proj_sg.gf || !proj_sg.hidden_input ||
                proj_sg.hidden_input->ne[1] != q_len) {
                if (!build_lm_head_projection_step(proj_sg, w, target_backend, q_len)) {
                    std::fprintf(stderr, "draft lm-head projection build failed\n");
                    return 1;
                }
            }
            if (!proj_sg.hidden_input || !proj_sg.logits) {
                std::fprintf(stderr, "draft lm-head projection build failed\n");
                return 1;
            }
            const size_t hidden_bytes = ggml_nbytes(draft_sg.hidden_states);
            if (!copy_peer_async(proj_sg.hidden_input->data, target_gpu,
                                 draft_sg.hidden_states->data, draft_gpu,
                                 hidden_bytes)) {
                std::fprintf(stderr, "draft hidden peer copy failed\n");
                return 1;
            }
            cudaSetDevice(target_gpu);
            cudaDeviceSynchronize();
            auto st_proj = ggml_backend_graph_compute(target_backend, proj_sg.gf);
            if (st_proj != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "draft lm-head projection compute %d\n", (int)st_proj);
                return 1;
            }
            ggml_backend_tensor_get(proj_sg.logits, draft_logits_buf.data(), 0,
                                    sizeof(float) * vocab * q_len);
        }
        auto T_draft_bridge = sync_us();
        tt_draft_bridge += std::chrono::duration<double, std::micro>(T_draft_bridge - T_draft_compute).count();

        // DDTree top-K: use GPU argmax for draft_tok; full logits transfer
        // only when DDTree needs top-K (K>1) for sibling expansion.
        const int ddtree_K = (ddtree_budget > q_len - 1) ? 8 : 1;

        if (draft_hidden_bridge) {
            for (int i = 0; i < q_len; i++) {
                draft_tok[i] = argmax_f32(draft_logits_buf.data() + (size_t)i * vocab, vocab);
            }
        } else {
            std::vector<int32_t> gpu_argmax(q_len);
            ggml_backend_tensor_get(draft_sg.argmax_tokens, gpu_argmax.data(), 0,
                                    sizeof(int32_t) * q_len);
            for (int i = 0; i < q_len; i++) draft_tok[i] = gpu_argmax[i];
        }
        // The block-diffusion draft is free to "denoise" position 0 even though
        // the input there is the unmasked last_tok. Pin it back so verify and
        // replay see the correct prefix.
        draft_tok[0] = last_tok;

        // DDTree top-K extraction. Positions 1..q_len-1 of the draft output
        // are the per-position distributions for the block-diffusion tree
        // (position 0 is the root/bonus slot and is fixed to last_tok). Only
        // computed in ddtree_mode to keep the argmax fast path untouched.
        // ddtree_K controls how many candidates per position are available as
        // tree siblings. Budget <= L means pure chain → no siblings needed, so
        // we can skip the O(L*vocab) top-K extract entirely and just fill rank 0
        // from draft_tok. For larger budgets we need real top-K.
        static std::vector<float>   ddtree_top_log_probs; // [L × K]
        static std::vector<int32_t> ddtree_top_token_ids; // [L × K]
        if (ddtree_mode) {
            const int L = q_len - 1;
            if ((int)ddtree_top_log_probs.size() < L * ddtree_K) {
                ddtree_top_log_probs.assign((size_t)L * ddtree_K, 0.0f);
                ddtree_top_token_ids.assign((size_t)L * ddtree_K, 0);
            }
            if (ddtree_K == 1) {
                // Fast path: draft_tok already holds the top-1 per position.
                // Skip extract; log-probs are irrelevant for pure chain build.
                for (int i = 0; i < L; i++) {
                    ddtree_top_log_probs[i] = 0.0f;
                    ddtree_top_token_ids[i] = draft_tok[i + 1];  // +1 to skip slot 0
                }
            } else {
                // DDTree K>1: need real log-probs for best-first tree scoring.
                // Transfer full logits for positions 1..q_len-1.
                if (!draft_hidden_bridge) {
                    ggml_backend_tensor_get(draft_sg.logits, draft_logits_buf.data(), 0,
                                            sizeof(float) * vocab * q_len);
                }
                extract_draft_topk(draft_logits_buf.data() + (size_t)vocab,
                                   L, vocab, ddtree_K,
                                   ddtree_top_log_probs.data(),
                                   ddtree_top_token_ids.data(),
                                   ddtree_temp);
            }
        }
        auto T_draft_logits = sync_us();
        tt_draft_logits += std::chrono::duration<double, std::micro>(T_draft_logits - T_draft_bridge).count();

        // 3) Snapshot SSM state (skipped in fast_rollback mode: the patched
        //    gated_delta_net kernel captures per-step intermediate states, so
        //    we don't need a pre-verify snapshot to restore from).
        if (!fast_rollback) {
            snapshot_ssm_state(cache);
        }
        auto T_snap = sync_us();
        tt_snap += std::chrono::duration<double, std::micro>(T_snap - T_draft_logits).count();

        // 4) Target verify on draft tokens.
        //
        // Two paths, toggled by --seq-verify:
        //   - Batched (default): one target forward over q_len tokens with a causal
        //     mask. Fast but suspected of numerical divergence vs stepwise decode
        //     per z-lab issue #57 ("batched greedy verification diverges from
        //     stepwise baseline").
        //   - Sequential: q_len independent single-token decodes. Slow but
        //     bit-equivalent to what the target would produce during plain
        //     autoregressive decode — the ground truth for greedy spec decoding.
        //
        // In both paths we set capture=true so if commit_n == q_len we can skip
        // the replay entirely. For commit_n < q_len, replay overwrites target_feat
        // at [committed..committed+commit_n-1]; positions past that are stale but
        // never read by the next iteration's draft.

        auto T_verify_build = T_snap;
        auto T_verify_set = T_snap;
        auto T_verify_compute = T_snap;

        // ── DDTree path: tree-structured verify + walk + rollback ─────────
        //
        // Structure of one DDTree round (ported from liranringel/ddtree.py):
        //   1. Build tree from draft top-K via best-first heap (Algorithm 1)
        //   2. Flatten tree in DFS order: slot 0 = root (= last_tok), slots
        //      1..n_nodes = tree nodes. Positions = committed + depth.
        //   3. Build an ancestor-only attention mask + parent_ids array.
        //   4. Run target forward via build_target_step_tree (our kernel mod
        //      handles DeltaNet/SSM tree recurrence via parent_ids).
        //   5. Walk the tree from root following target.argmax; the matched
        //      path is the accepted prefix. First unmatched target token
        //      becomes the next round's bonus (implicit via last_tok).
        //   6. Rollback: SSM state ← cache.ssm_intermediate[last_accepted_dfs_idx]
        //      KV cache ← cudaMemcpy the accepted DFS-order slots to slots 0..k-1
        //      conv_state ← use the (depth-1)-th slot of cache.conv_input_cache
        if (ddtree_mode) {
            const int L = q_len - 1;
            DDTree tree = build_ddtree(
                ddtree_top_log_probs.data(),
                ddtree_top_token_ids.data(),
                L, ddtree_K, ddtree_budget,
                ddtree_chain_seed);

            const int N_actual = 1 + tree.n_nodes;  // actual tree size
            const int N = ddtree_budget + 1;         // fixed allocation size for gallocr reuse

            if (!build_target_step_tree(sg, w, cache, backend,
                                        /*kv_start=*/committed, /*n_tokens=*/N,
                                        g_fa_window)) {
                std::fprintf(stderr, "ddtree verify build failed\n"); return 1;
            }
            T_verify_build = sync_us();
            tt_verify_build += std::chrono::duration<double, std::micro>(T_verify_build - T_snap).count();

            // Embeddings: [last_tok, tree.token_ids[0..n_nodes-1], padding...]
            std::vector<int32_t> flat_tokens(N, 0);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; i++) flat_tokens[1 + i] = tree.token_ids[i];
            // Pad remaining slots with token 0 — their outputs are masked

            std::vector<float> tree_embed((size_t)hidden * N, 0.0f);
            if (!w.embedder.embed(flat_tokens.data(), N_actual, tree_embed.data())) return 1;
            // Leave padding slots as zero
            ggml_backend_tensor_set(sg.inp_embed, tree_embed.data(), 0,
                                    sizeof(float) * hidden * N);

            // M-RoPE axis-major positions
            std::vector<int32_t> pos4(4 * N, 0);
            for (int i = 0; i < N_actual; i++) {
                int p = committed + (i == 0 ? 0 : tree.depths[i - 1]);
                pos4[0 * N + i] = p;
                pos4[1 * N + i] = p;
                pos4[2 * N + i] = p;
                pos4[3 * N + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, pos4.data(), 0, sizeof(int32_t) * 4 * N);

            // Ancestor-only attention mask (f16). Build for the full N slots
            // but only the first N_actual have valid visibility; padding slots
            // get -inf everywhere (default from assign).
            const int tree_win_start = (g_fa_window > 0 && committed > g_fa_window)
                                           ? (committed - g_fa_window) : 0;
            {
                // Use the same kv_pad as the tensor allocation (max_ctx + N)
                const int max_win_len = cache.max_ctx + N;
                const int kv_pad_m = align_up(max_win_len, g_kq_stride_pad);
                const int q_pad_m  = align_up(N, KQ_MASK_PAD);
                mask_buf.assign((size_t)kv_pad_m * q_pad_m, F16_NEG_INF);
                // Fill rows 0..N_actual-1 using the tree visibility
                for (int q = 0; q < N_actual; q++) {
                    // Past KV positions are visible to all tree nodes
                    for (int k = std::max(0, tree_win_start); k < committed; k++) {
                        mask_buf[(size_t)q * kv_pad_m + (k - tree_win_start)] = F16_ZERO;
                    }
                    // Tree self-visibility
                    for (int j = 0; j < N_actual; j++) {
                        if (tree.visibility[(size_t)q * N_actual + j]) {
                            mask_buf[(size_t)q * kv_pad_m + (committed + j - tree_win_start)] = F16_ZERO;
                        }
                    }
                }
                // Rows N_actual..N-1 remain all -inf (padding slots see nothing)
            }
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());

            // parent_ids: actual tree nodes, then padding → point to root (slot 0)
            std::vector<int32_t> parent_ids(N, 0);
            parent_ids[0] = -1;
            for (int i = 1; i < N_actual; i++) parent_ids[i] = (int32_t)tree.parents[i];
            // Padding slots: parent=0 (root). DeltaNet kernel processes them
            // but their outputs are never used (masked out in attention).
            ggml_backend_tensor_set(sg.parent_ids, parent_ids.data(), 0,
                                    sizeof(int32_t) * N);

            T_verify_set = sync_us();
            tt_verify_set += std::chrono::duration<double, std::micro>(T_verify_set - T_verify_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "ddtree verify compute %d\n", (int)st); return 1;
            }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_verify_set).count();

            // Read only the actual tree slots (not padding)
            std::vector<int32_t> posterior(N_actual);
            ggml_backend_tensor_get(sg.argmax_tokens, posterior.data(), 0,
                                    sizeof(int32_t) * N_actual);

            // Walk tree: accepted DFS indices and next bonus token.
            int next_token = -1;
            std::vector<int> accepted = follow_verified_tree(tree, posterior.data(), next_token);
            const int accept_depth = (int)accepted.size();  // includes root

            // Detect when the walk takes a sibling branch (accepted node
            // whose DFS index is OUTSIDE the chain spine [0..L]).
            bool walked_sibling = false;
            for (int x : accepted) {
                if (x > L) { walked_sibling = true; break; }
            }
            if (walked_sibling || n_draft_steps < 2) {
                std::printf("[dbg sib step %d] N=%d accept=%d walked_sib=%d\n",
                            n_draft_steps, N_actual, accept_depth, walked_sibling ? 1 : 0);
                std::printf("  walk:");
                for (int x : accepted) std::printf(" %d", x);
                if (walked_sibling) {
                    std::printf("\n  sibling info:");
                    for (int i = L; i < tree.n_nodes; i++) {
                        std::printf(" [%d:d%d:p%d:%d]",
                            i + 1, tree.depths[i], tree.parents[i + 1], tree.token_ids[i]);
                    }
                }
                std::printf("\n");
            }


            if (step_debug) {
                std::printf("[step %d] committed=%d last_tok=%d tree_N=%d accept=%d next=%d\n",
                            n_draft_steps, committed, last_tok, N_actual, accept_depth, next_token);
            }

            // Commit count: matches chain mode's accept_n semantics. The root
            // (= previous iter's last_tok) is "pending" — not yet in out_all —
            // and gets committed here along with each accepted child token.
            // next_token (target's correction at the deepest accepted node)
            // becomes the new last_tok, pending for the next iter.
            int commit_n = accept_depth;  // root + accepted children
            if (commit_n > need_commit_budget) commit_n = need_commit_budget;

            // Push the accepted path's tokens to out_all. The root token is
            // last_tok (the pending token from the previous iter). Each
            // subsequent accepted node contributes its own tree.token_ids
            // entry (dfs_idx - 1 because flat slot 0 = root, slot 1..N-1 =
            // tree.token_ids[0..n_nodes-1]).
            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                const int dfs_idx = accepted[i];
                const int32_t tok = (dfs_idx == 0)
                    ? last_tok
                    : tree.token_ids[dfs_idx - 1];
                out_all.push_back(tok); stream_emit(tok);
                if (IS_EOS_TOK(tok, w)) hit_eos = true;
            }
            last_tok = next_token;

            auto T_accept = sync_us();
            tt_accept += std::chrono::duration<double, std::micro>(T_accept - T_verify_compute).count();

            if (hit_eos) {
                hit_eos_decode = true;
                break;
            }

            // Rollback: per-layer DeltaNet SSM and conv state + KV compaction
            // for full-attention layers.
            //
            // SSM: the kernel wrote intermediate[i] for each flat-tree token i
            // (i = DFS index, 0 = root). We want the state AFTER processing
            // all `commit_n` tokens we just committed, i.e. the state after
            // the last committed DFS node = accepted[commit_n - 1]. For the
            // common case commit_n == accept_depth, this is the deepest
            // accepted node.
            const int rollback_dfs = (commit_n > 0)
                ? accepted[commit_n - 1]
                : 0;
            // Fast path detection: pure-chain walk has accepted[i] == i for
            // every i. Used by rollback to skip the parent-chain gather.
            bool walked_sibling_for_rollback = false;
            for (int i = 0; i < commit_n; i++) {
                if (accepted[i] != i) { walked_sibling_for_rollback = true; break; }
            }

            {
                const int n_delta = (int)sg.delta_captures.size();
                cudaStream_t stream = nullptr;
                for (int il = 0; il < n_delta; il++) {
                    const DeltaNetCapture & cap = sg.delta_captures[il];
                    if (!cap.ssm_intermediate_states || !cap.conv_input) {
                        std::fprintf(stderr, "ddtree rollback: missing capture layer %d\n", il);
                        return 1;
                    }
                    // SSM state rollback: source is cache.ssm_intermediate_states
                    // (f16, [S_v, S_v, H_v, max_verify_tokens]) at slot
                    // rollback_dfs. Destination is cache.ssm_state[il] (f32).
                    // Use a tiny CUDA kernel (src/f16_convert.cu) to widen f16
                    // → f32 in a single launch per layer.
                    const size_t ssm_elems =
                        (size_t)cache.ssm_state[il]->ne[0] *
                        (size_t)cache.ssm_state[il]->ne[1] *
                        (size_t)cache.ssm_state[il]->ne[2];
                    const size_t ssm_src_offset =
                        (size_t)rollback_dfs * cap.ssm_intermediate_states->nb[3];
                    const void * ssm_src =
                        (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
                    dflash27b_launch_f16_to_f32(ssm_src,
                                                cache.ssm_state[il]->data,
                                                ssm_elems,
                                                stream);
                    cudaError_t ce = cudaSuccess;  // launch error checked in the conv block below

                    // Conv rollback: copy the K-1 most recent inputs along
                    // the rolled-back token's ANCESTRY (not DFS order). Two
                    // paths:
                    //   - Pure chain accept (walked_sibling == false): the
                    //     conv window is 3 contiguous slots in conv_input, so
                    //     a single cudaMemcpy2DAsync handles it. Hot path.
                    //   - Sibling accept: scattered slots, fall back to K-1
                    //     individual column copies via parent-chain walk.
                    const int K_conv = 4;
                    const int row_cnt = (int)cap.conv_input->ne[1];
                    const size_t elt = ggml_element_size(cap.conv_input);
                    const size_t dpitch = (K_conv - 1) * elt;
                    const size_t spitch = cap.conv_input->nb[1];
                    if (!walked_sibling_for_rollback) {
                        // Fast path: 3 contiguous slots ending at rollback_dfs.
                        const int conv_off = rollback_dfs + 1;
                        const void * conv_src =
                            (const char *)cap.conv_input->data + (size_t)conv_off * elt;
                        ce = cudaMemcpy2DAsync(cache.conv_state[il]->data, dpitch,
                                               conv_src, spitch,
                                               (K_conv - 1) * elt, row_cnt,
                                               cudaMemcpyDeviceToDevice, stream);
                        if (ce != cudaSuccess) {
                            std::fprintf(stderr, "ddtree conv fast il=%d: %s\n",
                                         il, cudaGetErrorString(ce));
                            return 1;
                        }
                    } else {
                        int virt[K_conv - 1];
                        virt[K_conv - 2] = rollback_dfs;
                        for (int k = K_conv - 3; k >= 0; k--) {
                            const int prev = virt[k + 1];
                            virt[k] = (prev >= 0) ? (int)tree.parents[prev] : (prev - 1);
                        }
                        for (int k = 0; k < K_conv - 1; k++) {
                            const int sx_slot = (K_conv - 1) + virt[k];
                            const void * src_col =
                                (const char *)cap.conv_input->data + (size_t)sx_slot * elt;
                            char * dst_col =
                                (char *)cache.conv_state[il]->data + (size_t)k * elt;
                            ce = cudaMemcpy2DAsync(dst_col, dpitch,
                                                   src_col, spitch,
                                                   elt, row_cnt,
                                                   cudaMemcpyDeviceToDevice, stream);
                            if (ce != cudaSuccess) {
                                std::fprintf(stderr, "ddtree conv col il=%d k=%d: %s\n",
                                             il, k, cudaGetErrorString(ce));
                                return 1;
                            }
                        }
                    }
                }

                // target_feat compaction: written in DFS order during verify
                // (column kv_start+i = dfs slot i's features). Same logic as
                // KV cache: when accepted[d] != d, copy the accepted DFS slot's
                // features to the spine slot at d so next iter's draft reads
                // the right history. Position→slot uses `% target_feat_cap`
                // to account for the ring buffer.
                if (cache.target_feat) {
                    const size_t elt = ggml_element_size(cache.target_feat);
                    const int    fc_in = (int)cache.target_feat->ne[0];  // 5*hidden
                    const size_t col_stride = cache.target_feat->nb[1];
                    const int    tcap = cache.target_feat_cap;
                    for (int d = 1; d < commit_n; d++) {
                        const int src_dfs = accepted[d];
                        if (src_dfs == d) continue;
                        const int    src_slot = (committed + src_dfs) % tcap;
                        const int    dst_slot = (committed + d)       % tcap;
                        const size_t src_off  = (size_t)src_slot * col_stride;
                        const size_t dst_off  = (size_t)dst_slot * col_stride;
                        cudaMemcpyAsync((char *)cache.target_feat->data + dst_off,
                                        (const char *)cache.target_feat->data + src_off,
                                        (size_t)fc_in * elt,
                                        cudaMemcpyDeviceToDevice, stream);
                    }
                }

                // Full-attention KV compaction: the verify wrote K/V at slots
                // [committed..committed+N-1] in DFS tree order (slot 0 = root).
                // For the next iter's verify to see the correct committed
                // prefix, slots [committed..committed+commit_n-1] must hold
                // the K/V of the accepted path's committed tokens. For each
                // committed position d in 0..commit_n-1, the source K/V is at
                // DFS slot accepted[d]. d==0 is always the root (DFS slot 0),
                // trivially aligned. For d>=1, copy if accepted[d] != d.
                const int n_full_attn = (int)cache.attn_k.size();
                for (int d = 0; d < commit_n; d++) {
                    const int src_dfs = accepted[d];
                    const int dst_slot = d;
                    if (src_dfs == dst_slot) continue;  // already aligned
                    for (int l = 0; l < n_full_attn; l++) {
                        // Each slot: head_dim * n_kv floats in f16 per tensor.
                        ggml_tensor * ck = cache.attn_k[l];
                        ggml_tensor * cv = cache.attn_v[l];
                        const size_t slot_bytes = ck->nb[1];  // stride between slots
                        const size_t src_off = (size_t)(committed + src_dfs) * slot_bytes;
                        const size_t dst_off = (size_t)(committed + dst_slot) * slot_bytes;
                        // Per-head-kv layout: shape [head_dim, max_ctx, n_head_kv].
                        // nb[2] is distance between heads; we copy one slot's
                        // slice per head. For simplicity, do a 2D copy across
                        // the head dimension.
                        const int n_kv = (int)ck->ne[2];
                        for (int h = 0; h < n_kv; h++) {
                            const size_t head_src = src_off + (size_t)h * ck->nb[2];
                            const size_t head_dst = dst_off + (size_t)h * ck->nb[2];
                            cudaMemcpyAsync((char *)ck->data + head_dst,
                                            (const char *)ck->data + head_src,
                                            slot_bytes, cudaMemcpyDeviceToDevice, stream);
                            cudaMemcpyAsync((char *)cv->data + head_dst,
                                            (const char *)cv->data + head_src,
                                            slot_bytes, cudaMemcpyDeviceToDevice, stream);
                        }
                    }
                }
                // No explicit sync: stream==nullptr (default stream) serializes
                // these copies before the next iter's draft/verify kernels.
                // The CPU returns immediately and the next iter's CPU work
                // (graph build, embed) can overlap with the GPU compaction.
            }

            if (!sync_draft_feature_mirror(committed, commit_n)) {
                std::fprintf(stderr, "draft feature mirror sync failed after ddtree commit\n");
                return 1;
            }

            committed    += commit_n;
            n_generated  += commit_n;
            n_accept_sum += commit_n;  // for stats
            n_draft_steps++;
            continue;  // skip the rest of the verify/commit logic for this iter
        }

        if (!seq_verify) {
            const int verify_fa_window = g_fa_window;
            if (!build_target_step(sg, w, cache, backend,
                                    /*kv_start=*/committed, /*n_tokens=*/q_len,
                                    /*with_mask=*/true, /*capture=*/true,
                                    /*capture_delta_intermediate=*/fast_rollback,
                                    verify_fa_window)) {
                std::fprintf(stderr, "verify build failed\n"); return 1;
            }
            T_verify_build = sync_us();
            tt_verify_build += std::chrono::duration<double, std::micro>(T_verify_build - T_snap).count();

            std::vector<float> verify_embed(hidden * q_len);
            if (!w.embedder.embed(draft_tok.data(), q_len, verify_embed.data())) return 1;
            ggml_backend_tensor_set(sg.inp_embed, verify_embed.data(), 0,
                                    sizeof(float) * verify_embed.size());

            // M-RoPE axis-major layout: [axis0_tok0..axis0_tokN-1, axis1_..., axis2_..., axis3_...].
            // First 3 axes hold the token position; axis 3 is always 0 for text.
            for (int i = 0; i < q_len; i++) {
                int p = committed + i;
                pos4_buf[0 * q_len + i] = p;
                pos4_buf[1 * q_len + i] = p;
                pos4_buf[2 * q_len + i] = p;
                pos4_buf[3 * q_len + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, pos4_buf.data(), 0, sizeof(int32_t) * 4 * q_len);

            {
                const int win_start_v = (verify_fa_window > 0 && committed > verify_fa_window)
                                            ? (committed - verify_fa_window) : 0;
                const int win_len_v = committed + q_len - win_start_v;
                build_causal_mask(mask_buf, win_len_v, q_len, committed, win_start_v);
            }
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0, sizeof(uint16_t) * mask_buf.size());
            T_verify_set = sync_us();
            tt_verify_set += std::chrono::duration<double, std::micro>(T_verify_set - T_verify_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "verify compute %d\n", (int)st); return 1; }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_verify_set).count();

            ggml_backend_tensor_get(sg.argmax_tokens, target_tok.data(), 0,
                                    sizeof(int32_t) * q_len);
        } else {
            // Sequential verify: q_len independent single-token decodes.
            // Each call writes K/V at slot committed+i and advances SSM by 1.
            // After the loop, target cache state is identical to the batched
            // path's state (both end at committed+q_len). Restore/replay below
            // still apply correctly.
            std::vector<float> single_embed(hidden);
            int32_t p4_single[4];
            for (int i = 0; i < q_len; i++) {
                if (!build_target_step(sg, w, cache, backend,
                                        /*kv_start=*/committed + i, /*n_tokens=*/1,
                                        /*with_mask=*/false, /*capture=*/true)) {
                    std::fprintf(stderr, "seq verify build %d failed\n", i); return 1;
                }
                int32_t t = draft_tok[i];
                if (!w.embedder.embed(&t, 1, single_embed.data())) return 1;
                ggml_backend_tensor_set(sg.inp_embed, single_embed.data(), 0,
                                        sizeof(float) * hidden);
                int p = committed + i;
                p4_single[0] = p; p4_single[1] = p; p4_single[2] = p; p4_single[3] = 0;
                ggml_backend_tensor_set(sg.positions, p4_single, 0, sizeof(int32_t) * 4);

                st = ggml_backend_graph_compute(backend, sg.gf);
                if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "seq verify compute %d at %d\n", (int)st, i); return 1; }

                ggml_backend_tensor_get(sg.logits,
                                        verify_logits_buf.data() + (size_t)i * vocab,
                                        0, sizeof(float) * vocab);
                target_tok[i] = argmax_f32(verify_logits_buf.data() + (size_t)i * vocab, vocab);
            }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_snap).count();
        }
        auto T_verify_logits = sync_us();
        tt_verify_logits += std::chrono::duration<double, std::micro>(T_verify_logits - T_verify_compute).count();

        if (step_debug) {
            std::printf("[step %d] committed=%d last_tok=%d\n", n_draft_steps, committed, last_tok);
        }

        // 5) Greedy longest-prefix accept with standard spec-decoding comparison.
        //
        //   - draft_tok[0] should equal last_tok (the correct first token from the
        //     previous forward). Accept it unconditionally.
        //   - target_tok[i] = argmax(logit at position committed+i) = target's
        //     prediction for the token AT position committed+i+1 (given draft_tok[0..i]).
        //   - So the check is: draft_tok[i+1] == target_tok[i], for i=0..q_len-2.
        //   - First mismatch at i=k → accept draft_tok[0..k] (k+1 tokens),
        //     bonus = target_tok[k] (the correct replacement for draft_tok[k+1]).

        int accept_n = 1;  // draft_tok[0] assumed = last_tok
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        // Two commit strategies:
        //   - Legacy (replay path): commit_n = accept_n + 1, the extra is the
        //     "bonus" token (target's correction or target_tok[q_len-1] when
        //     all accepted). Requires a replay forward pass to advance state.
        //   - Fast-rollback path: commit_n = accept_n, no explicit bonus. Use
        //     verify_logits[accept_n-1] as next last_tok; the "bonus" becomes
        //     draft_tok[0] of the next iter and is accepted unconditionally.
        //     Identical output stream, one fewer commit per iter tallied, but
        //     no extra forward pass needed.
        int bonus_tok = -1;
        int commit_n;
        if (fast_rollback) {
            commit_n = accept_n;
        } else {
            if (accept_n < q_len) {
                bonus_tok = target_tok[accept_n - 1];
            }
            commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        }
        if (step_debug) {
            std::printf("[step %d] accept_n=%d bonus=%d commit_n=%d\n",
                        n_draft_steps, accept_n, bonus_tok, commit_n);
        }

        // Don't overshoot n_gen
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            // If we were going to add the bonus but budget is tight, drop it.
            if (commit_n <= accept_n) bonus_tok = -1;
        }
        auto T_accept = sync_us();
        tt_accept += std::chrono::duration<double, std::micro>(T_accept - T_verify_logits).count();

        // 6) Rollback and commit.
        //
        // Fast-rollback path: no replay. Use the per-step SSM intermediate states
        // captured during verify to roll back DeltaNet state, and slice the conv
        // input tensor for the conv state. Next last_tok comes from verify's
        // logits at position (accept_n - 1) — the target's prediction at position
        // committed+accept_n given the accepted prefix. The implicit bonus
        // becomes the next iter's draft_tok[0].
        //
        // Legacy path: restore SSM state from snapshot and run the replay.
        double t_rollback_us = 0, t_replay_build_us = 0, t_replay_set_us = 0;
        double t_replay_compute_us = 0, t_replay_logits_us = 0;

        if (fast_rollback) {
            auto T_rb0 = sync_us();

            // Rollback SSM + conv state unless we fully accepted (in which case
            // state after processing all q_len tokens is exactly what we want).
            if (commit_n < q_len) {
                const int rollback_idx = commit_n - 1;  // index into per-step intermediates
                // Temporary ctx for view tensors (no data alloc — views inherit
                // data pointers from their already-live sources).
                ggml_init_params tp{};
                tp.mem_size   = 1024 * 1024;
                tp.mem_buffer = nullptr;
                tp.no_alloc   = true;
                ggml_context * tmp_ctx = ggml_init(tp);
                if (!tmp_ctx) { std::fprintf(stderr, "rollback ctx init failed\n"); return 1; }

                const int n_delta = (int)sg.delta_captures.size();
                cudaStream_t stream = nullptr;  // use default stream
                for (int il = 0; il < n_delta; il++) {
                    const DeltaNetCapture & cap = sg.delta_captures[il];
                    if (!cap.ssm_intermediate_states || !cap.conv_input) {
                        std::fprintf(stderr, "rollback: missing capture at layer %d\n", il);
                        return 1;
                    }

                    // ── SSM rollback: copy intermediate[rollback_idx] → cache.ssm_state[il]
                    //
                    // cap.ssm_intermediate_states is the persistent cache buffer
                    // cache.ssm_intermediate[il], shape [S_v, S_v, H_v, q_len].
                    // Stored in f16 (see create_target_cache) to halve memory;
                    // cache.ssm_state[il] is f32. Use the widen kernel to
                    // convert on copy, same as the DDtree rollback path.
                    const size_t ssm_elems =
                        (size_t)cache.ssm_state[il]->ne[0] *
                        (size_t)cache.ssm_state[il]->ne[1] *
                        (size_t)cache.ssm_state[il]->ne[2];
                    const size_t ssm_src_offset =
                        (size_t)rollback_idx * cap.ssm_intermediate_states->nb[3];
                    const void * ssm_src =
                        (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
                    dflash27b_launch_f16_to_f32(ssm_src,
                                                cache.ssm_state[il]->data,
                                                ssm_elems,
                                                stream);
                    cudaError_t ce = cudaSuccess;

                    // ── Conv rollback: copy conv_input[commit_n..commit_n+K-2, :, :]
                    //    into cache.conv_state[il].
                    //
                    // conv_input shape: [kernel-1 + n_tokens, conv_channels, 1]
                    //   nb[0] = elt, nb[1] = (kernel-1+n_tokens)*elt
                    // conv_state shape: [kernel-1, conv_channels, 1]
                    //   nb[0] = elt, nb[1] = (kernel-1)*elt
                    //
                    // Need cudaMemcpy2D because the source has a larger row stride
                    // (spans kernel-1+n_tokens values along dim 0) than the dest.
                    const int K_conv = 4;                            // qwen3.5 DeltaNet conv kernel
                    const int row_cnt = (int)cap.conv_input->ne[1];  // conv_channels (10240)
                    const size_t elt = ggml_element_size(cap.conv_input);
                    const size_t dpitch = (K_conv - 1) * elt;        // 12 bytes
                    const size_t spitch = cap.conv_input->nb[1];     // (K-1+n_tokens)*elt
                    const size_t width  = (K_conv - 1) * elt;        // copy 3 floats per row
                    const void * conv_src =
                        (const char *)cap.conv_input->data + commit_n * elt;
                    ce = cudaMemcpy2DAsync(cache.conv_state[il]->data, dpitch,
                                           conv_src, spitch,
                                           width, row_cnt,
                                           cudaMemcpyDeviceToDevice, stream);
                    if (ce != cudaSuccess) {
                        std::fprintf(stderr, "cudaMemcpy2D conv rollback il=%d: %s\n",
                                     il, cudaGetErrorString(ce));
                        return 1;
                    }
                }
                cudaStreamSynchronize(stream);

                ggml_free(tmp_ctx);
            }

            // Next last_tok: target's prediction at position committed+accept_n
            // given the accepted prefix.
            //   - commit_n < q_len: verify_logits[accept_n-1] (target_tok[accept_n-1]).
            //   - commit_n == q_len: verify_logits[q_len-1]  (target_tok[q_len-1]).
            // Both already computed as `target_tok[commit_n-1]` during accept.
            last_tok = target_tok[commit_n - 1];

            auto T_rb1 = sync_us();
            t_rollback_us = std::chrono::duration<double, std::micro>(T_rb1 - T_rb0).count();
            tt_restore += t_rollback_us;

            // Commit: push accepted draft tokens to out_all. No bonus — next iter
            // picks it up as last_tok.
            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                out_all.push_back(draft_tok[i]); stream_emit(draft_tok[i]);
                if (IS_EOS_TOK(draft_tok[i], w)) hit_eos = true;
            }
            if (hit_eos) {
                hit_eos_decode = true;
                break;
            }
        } else {
            // ── Legacy replay path ──
            restore_ssm_state(cache);
            auto T_restore = sync_us();
            tt_restore += std::chrono::duration<double, std::micro>(T_restore - T_accept).count();
            std::vector<int32_t> replay_tok(commit_n);
            for (int i = 0; i < commit_n; i++) {
                if (i < accept_n && i < (int)draft_tok.size()) {
                    replay_tok[i] = draft_tok[i];
                } else {
                    replay_tok[i] = bonus_tok;
                }
            }

            bool replay_with_mask = (commit_n > 1);
            const int replay_fa_window = g_fa_window;
            if (!build_target_step(sg, w, cache, backend,
                                    committed, commit_n,
                                    replay_with_mask, /*capture=*/true,
                                    false, replay_fa_window)) {
                std::fprintf(stderr, "replay build failed\n"); return 1;
            }
            auto T_replay_build = sync_us();
            tt_replay_build += std::chrono::duration<double, std::micro>(T_replay_build - T_restore).count();

            std::vector<float> replay_embed(hidden * commit_n);
            if (!w.embedder.embed(replay_tok.data(), commit_n, replay_embed.data())) return 1;
            ggml_backend_tensor_set(sg.inp_embed, replay_embed.data(), 0, sizeof(float) * replay_embed.size());
            std::vector<int32_t> replay_pos(4 * commit_n);
            for (int i = 0; i < commit_n; i++) {
                int p = committed + i;
                replay_pos[0 * commit_n + i] = p;
                replay_pos[1 * commit_n + i] = p;
                replay_pos[2 * commit_n + i] = p;
                replay_pos[3 * commit_n + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, replay_pos.data(), 0, sizeof(int32_t) * 4 * commit_n);
            if (replay_with_mask) {
                const int win_start_r = (replay_fa_window > 0 && committed > replay_fa_window)
                                            ? (committed - replay_fa_window) : 0;
                const int win_len_r = committed + commit_n - win_start_r;
                build_causal_mask(mask_buf, win_len_r, commit_n, committed, win_start_r);
                ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0, sizeof(uint16_t) * mask_buf.size());
            }
            auto T_replay_set = sync_us();
            tt_replay_set += std::chrono::duration<double, std::micro>(T_replay_set - T_replay_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "replay compute %d\n", (int)st); return 1; }
            auto T_replay_compute = sync_us();
            tt_replay_compute += std::chrono::duration<double, std::micro>(T_replay_compute - T_replay_set).count();

            std::vector<float> last_logits(vocab);
            ggml_backend_tensor_get(sg.logits, last_logits.data(),
                                    sizeof(float) * vocab * (commit_n - 1),
                                    sizeof(float) * vocab);
            last_tok = argmax_f32(last_logits.data(), vocab);
            auto T_replay_logits = sync_us();
            tt_replay_logits += std::chrono::duration<double, std::micro>(T_replay_logits - T_replay_compute).count();

            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                out_all.push_back(replay_tok[i]); stream_emit(replay_tok[i]);
                if (IS_EOS_TOK(replay_tok[i], w)) hit_eos = true;
            }
            if (hit_eos) {
                hit_eos_decode = true;
                break;
            }
        }

        if (!sync_draft_feature_mirror(committed, commit_n)) {
            std::fprintf(stderr, "draft feature mirror sync failed after commit\n");
            return 1;
        }

        committed    += commit_n;
        n_generated  += commit_n;
        n_accept_sum += accept_n;
        n_draft_steps++;
    }

    auto t_gen1 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(t_gen1 - t_gen0).count();
    double tps = n_generated / std::max(1e-9, gen_s);

    if (timing_debug) {
        auto avg_ms = [&](double us){ return us / std::max(1, n_draft_steps) / 1000.0; };
        std::printf("\n[timing] per-step averages over %d steps (ms):\n", n_draft_steps);
        std::printf("  draft_build    %.2f\n", avg_ms(tt_draft_build));
        std::printf("  draft_copyfeat %.2f\n", avg_ms(tt_draft_copy_feat));
        std::printf("  draft_set      %.2f\n", avg_ms(tt_draft_set));
        std::printf("  draft_compute  %.2f\n", avg_ms(tt_draft_compute));
        std::printf("  draft_bridge   %.2f\n", avg_ms(tt_draft_bridge));
        std::printf("  draft_logits   %.2f\n", avg_ms(tt_draft_logits));
        std::printf("  snapshot_ssm   %.2f\n", avg_ms(tt_snap));
        std::printf("  verify_build   %.2f\n", avg_ms(tt_verify_build));
        std::printf("  verify_set     %.2f\n", avg_ms(tt_verify_set));
        std::printf("  verify_compute %.2f\n", avg_ms(tt_verify_compute));
        std::printf("  verify_logits  %.2f\n", avg_ms(tt_verify_logits));
        std::printf("  accept         %.2f\n", avg_ms(tt_accept));
        std::printf("  restore_ssm    %.2f\n", avg_ms(tt_restore));
        std::printf("  replay_build   %.2f\n", avg_ms(tt_replay_build));
        std::printf("  replay_set     %.2f\n", avg_ms(tt_replay_set));
        std::printf("  replay_compute %.2f\n", avg_ms(tt_replay_compute));
        std::printf("  replay_logits  %.2f\n", avg_ms(tt_replay_logits));
        std::printf("  mirror_sync    %.2f\n", avg_ms(tt_mirror_sync));
        double sum_ms = avg_ms(tt_draft_build + tt_draft_copy_feat + tt_draft_set + tt_draft_compute + tt_draft_logits
                               + tt_draft_bridge
                               + tt_snap + tt_verify_build + tt_verify_set + tt_verify_compute + tt_verify_logits
                               + tt_accept + tt_restore + tt_replay_build + tt_replay_set + tt_replay_compute + tt_replay_logits
                               + tt_mirror_sync);
        std::printf("  ----- sum     %.2f\n", sum_ms);
    }

    std::printf("\n[dflash] generated %d tokens in %.3f s  ->  %.2f tok/s\n",
                n_generated, gen_s, tps);
    std::printf("[dflash] %d draft steps, accepted=%d/%d (%.1f%% per step), "
                "avg commit/step=%.2f\n",
                n_draft_steps, n_accept_sum, n_draft_steps * q_len,
                (n_draft_steps > 0 ? 100.0 * n_accept_sum / (n_draft_steps * q_len) : 0.0),
                (n_draft_steps > 0 ? (double)n_generated / n_draft_steps : 0.0));
    if (output_tail) {
        std::printf("[dflash] output tail: ");
        int tail_start = std::max(0, (int)out_all.size() - 20);
        for (int i = tail_start; i < (int)out_all.size(); i++) std::printf("%d ", out_all[i]);
        std::printf("\n");
    }
    std::fflush(stdout);

    if (daemon_mode) {
        bool scheduler_request_still_active = false;
        if (scheduler_start_requested) {
            DaemonRequestState & req = daemon_requests[(size_t)active_cache_slot];
            req.active = !hit_eos_decode && n_generated < scheduler_total_gen;
            req.request_id = current_stream_request_id;
            req.slot_id = active_cache_slot;
            req.remaining = std::max(0, scheduler_total_gen - n_generated);
            req.quantum = scheduler_quantum;
            req.emitted = n_generated;
            req.epoch += 1;
            scheduler_request_still_active = req.active;
            std::printf("[scheduler] start req=%d slot=%d emitted=%d remaining=%d q=%d active=%d epoch=%d\n",
                        req.request_id, req.slot_id, req.emitted, req.remaining,
                        req.quantum, (int)req.active, req.epoch);
            std::fflush(stdout);
        } else if (scheduler_continue_requested) {
            DaemonRequestState & req = daemon_requests[(size_t)active_cache_slot];
            req.remaining = std::max(0, req.remaining - n_generated);
            req.emitted += n_generated;
            if (hit_eos_decode || req.remaining <= 0) req.active = false;
            scheduler_request_still_active = req.active;
            std::printf("[scheduler] step req=%d slot=%d emitted=%d remaining=%d active=%d epoch=%d\n",
                        req.request_id, req.slot_id, req.emitted, req.remaining,
                        (int)req.active, req.epoch);
            std::fflush(stdout);
        }
        // Update cache.cur_pos / cache.last_tok to reflect end-of-generation
        // state so a subsequent SNAPSHOT command captures the correct boundary.
        // Both fields are otherwise unused by the prefill/decode hot path
        // (kv_start is tracked separately, last_tok is a local) — they exist
        // for cross-request snapshot accounting.
        cache.cur_pos  = committed;
        cache.last_tok = last_tok;
        stream_emit(scheduler_request_still_active ? -4 : -1);
        if (daemon_scheduler_drain && daemon_pending_quanta.empty()) {
            if (!has_daemon_input()) {
                if (!enqueue_next_scheduler_quantum()) {
                    daemon_scheduler_drain = false;
                    std::printf("[scheduler] drain complete\n");
                    std::fflush(stdout);
                }
            }
        }
    } else {
        if (out_path) write_int32_file(out_path, out_all);
        break;
    }

    } // end while(true)

    for (auto & slot : daemon_extra_slots) {
        draft_feature_mirror_free(slot->feature_mirror);
        step_graph_destroy(slot->proj_sg);
        step_graph_destroy(slot->draft_sg);
        step_graph_destroy(slot->sg);
        for (int i = 0; i < PREFIX_CACHE_SLOTS; i++) {
            free_prefix_snapshot(slot->prefix_snapshots[i]);
        }
        free_target_cache(slot->cache);
    }
    daemon_extra_slots.clear();
    if (drafter_loaded) {
        dflash27b::free_drafter(drafter_ctx);
        drafter_loaded = false;
    }
    step_graph_destroy(daemon_batch_probe_sg);
    step_graph_destroy(daemon_batch_ref_sg);
    free_target_cache(daemon_batch_probe_cache);
    free_target_cache(daemon_batch_ref_cache);
    draft_feature_mirror_free(feature_mirror);
    step_graph_destroy(proj_sg);
    step_graph_destroy(draft_sg);
    if (daemon_mode) {
        for (int i = 0; i < PREFIX_CACHE_SLOTS; i++) free_prefix_snapshot(prefix_snapshots[i]);
    }
    step_graph_destroy(sg);
    free_target_cache(cache);
    free_draft_weights(dw);
    free_target_weights(w);
    if (split_gpus) ggml_backend_free(draft_backend);
    ggml_backend_free(target_backend);
    return 0;
}
