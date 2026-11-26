// QSA-indexer KVFlash recall scorer (LUCE_Q4_INDEXER_RECALL): pure host aggregation of the indexer's top-k block picks
// into one score per KVFlash chunk. Header-only so the unit test builds without a model, backend or GPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace luce::common {

// Votes per chunk. `ids` is [layer][query][budget] block ids (the indexer's ggml_top_k output, layer-major); block b
// lies in chunk b / blocks_per_chunk. A chunk's vote is the picks inside it summed over every layer and query.
// `visible` (n_queries entries, optional) is each query's complete-block count: a pick at or past it is top-k padding
// of an invisible (-inf) block and does not vote. Ids outside [0, n_chunks * blocks_per_chunk) are dropped.
inline std::vector<float> q4_recall_chunk_votes(const std::vector<int32_t> & ids, int n_layers, int n_queries, int budget,
                                                int blocks_per_chunk, int n_chunks,
                                                const std::vector<int> * visible = nullptr) {
    std::vector<float> votes((size_t) std::max(0, n_chunks), 0.f);
    if (blocks_per_chunk <= 0 || n_chunks <= 0) return votes;
    for (int l = 0; l < n_layers; ++l) {
        for (int q = 0; q < n_queries; ++q) {
            const size_t row = ((size_t) l * n_queries + q) * budget;
            const int64_t vis = visible ? (*visible)[(size_t) q] : INT64_MAX;
            for (int i = 0; i < budget && row + i < ids.size(); ++i) {
                const int64_t b = ids[row + i];
                if (b < 0 || b >= vis || b / blocks_per_chunk >= n_chunks) continue;
                votes[(size_t) (b / blocks_per_chunk)] += 1.f;
            }
        }
    }
    return votes;
}

// Votes plus a tie-break below one vote: equal votes go to the newer chunk.
inline std::vector<float> q4_recall_chunk_scores(const std::vector<float> & votes) {
    std::vector<float> s(votes.size());
    for (size_t c = 0; c < votes.size(); ++c) s[c] = votes[c] + 0.5f * (float) (c + 1) / (float) (votes.size() + 1);
    return s;
}

// The n best chunks, best first.
inline std::vector<int> q4_recall_top(const std::vector<float> & scores, int n) {
    std::vector<int> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    n = std::min<int>(n, (int) idx.size());
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(), [&](int a, int b) { return scores[(size_t) a] > scores[(size_t) b]; });
    idx.resize((size_t) n);
    return idx;
}

// An integer env value: the whole string, within [lo, hi]. Unset or empty is `def` without complaint; anything else
// invalid is `def` with `refused` set, for the caller to name the variable and the value used.
struct Q4EnvInt { int value; bool refused; };
inline Q4EnvInt q4_parse_env_int(const char * v, int lo, int hi, int def) {
    if (!v || !*v) return {def, false};
    char * end = nullptr;
    errno = 0;
    const long n = std::strtol(v, &end, 10);
    const bool digits = (*v >= '0' && *v <= '9') || ((*v == '-' || *v == '+') && v[1] >= '0' && v[1] <= '9');
    if (!digits || *end || errno == ERANGE || n < lo || n > hi) return {def, true};
    return {(int) n, false};
}

// LUCE_Q4_RECALL_TAIL: tokens, 0 = no two-pass tail; contexts stay below 2^24.
constexpr int kQ4RecallTailDefault = 2048, kQ4RecallTailMax = (1 << 24) - 1;

// --verify-width under recall (0 = adaptive). The adaptive width learns from wall-clock step costs, so verify batch
// shapes, and with them the low bits of the KV, vary run to run; recall's top-k vote flips on those near-ties.
// Mode 2 pins an adaptive width to 5 (two runs gave identical logs); an explicit width is the user's.
inline int q4_recall_verify_width(int mode, int width) { return mode == 2 && width == 0 ? 5 : width; }

// QSA over a KVFlash pool (LUCE_Q4_POOL_QSA), each step: `rows` (n_chunks) gets chunk c's first arena row, or -1 when
// the pool does not hold it, for the QSA ids to map through; every 4-token block of a missing chunk is hidden from each
// of the T rows of `visibility` ([T][blocks]), so the indexer's top-k spends its budget on keys the pool holds.
// `Pager` has KvFlashPager::block_of.
template <class Pager>
inline void q4_pool_qsa_inputs(const Pager & pager, int chunk, int n_chunks, int64_t blocks, int64_t T,
                               std::vector<int32_t> & rows, std::vector<float> & visibility) {
    const int64_t per = chunk / 4;
    rows.assign((size_t) n_chunks, -1);
    for (int c = 0; c < n_chunks; ++c) {
        const int b = pager.block_of(c);
        if (b >= 0) { rows[(size_t) c] = b * chunk; continue; }
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t i = c * per; i < std::min<int64_t>(blocks, (c + 1) * per); ++i)
                visibility[(size_t) (t * blocks + i)] = -INFINITY;
        }
    }
}

// Bytes of one full-attention layer's recall storage (raw indexer keys per position — F16 when pooled, F32 otherwise —
// plus F32 pooled keys per block of `ratio`), which a pool allocates at the full max_ctx; 0 for a layer without an indexer.
inline int64_t q4_recall_layer_bytes(int64_t max_ctx, int64_t head_size, int64_t ratio, bool f16_raw = false) {
    if (head_size <= 0 || ratio <= 0) return 0;
    const int64_t raw_bytes = head_size * (f16_raw ? 2 : 4) * max_ctx;
    const int64_t pool_bytes = head_size * 4 * ((max_ctx + ratio - 1) / ratio + 1);
    return raw_bytes + pool_bytes;
}

// Where the two-pass tail starts, or -1 for a single pass. The prompt's last `tail` tokens (no earlier than `pos`, the
// restored prefix) are prefilled twice, so the tail must start past the pool: a prompt that fits it pages nothing.
inline int q4_recall_tail_pos(int pos, int prompt_len, int tail, int pool_tokens) {
    if (tail <= 0) return -1;
    const int t = std::max(pos, prompt_len - tail);
    return t <= pool_tokens ? -1 : t;
}

// The recall mode once context fitting has settled the pool and where attention sits. `why` names a change, else null.
struct Q4RecallSettled { int mode; const char * why; };
inline Q4RecallSettled q4_recall_settle(int mode, int remote_layers, bool pooled) {
    if (mode > 0 && remote_layers > 0) return {0, "full-attention layers sit on the expert device"};
    if (mode == 2 && !pooled) return {1, "no KVFlash pool smaller than the context is left (--kvflash); score only"};
    return {mode, nullptr};
}

// Whether a forward of n rows ending at `end` restarts the captured query run. A one-row forward has no capture path,
// so a later capture would leave a gap; as the prompt's last forward it leaves the rows before it contiguous.
inline bool q4_recall_run_breaks(int n, int end, int prompt_len) { return n == 1 && end < prompt_len; }

// Recall ranks paged-out chunks, and a pool that holds the whole prompt has none (run_indexer_recall returns early):
// such a prefill captures no queries and adds no query cut. No pool (mode 1 scores virtual spans): always.
inline bool q4_recall_pages(int prompt_len, int pool_tokens) { return pool_tokens <= 0 || prompt_len > pool_tokens; }

// Where the prefill's forwards start: the restore points, then the prefix snapshot, the query span's start and the tail
// start when past `pos` (-1: none). The replay passes -1 for the last two: its queries are taken.
inline std::vector<int> q4_prefill_cuts(std::vector<int> cuts, int pos, int snap_pos, int query_pos, int tail_pos) {
    for (int c : {snap_pos, query_pos, tail_pos})
        if (c > pos) cuts.push_back(c);
    std::sort(cuts.begin(), cuts.end());
    return cuts;
}

// The path a mode-2 request plans under a pool; q4_tail_outcome names what ran for the one [q4-tail] line.
enum class Q4TailPath { Off, NoDecode, FitsPool, TailInPool, NoForward, TwoPass };
inline Q4TailPath q4_recall_tail_path(int pos, int prompt_len, int tail, int pool_tokens, int n_gen) {
    if (tail <= 0) return Q4TailPath::Off;
    if (n_gen <= 0) return Q4TailPath::NoDecode;   // a prefill-only request decodes nothing the recall could serve
    if (prompt_len <= pool_tokens) return Q4TailPath::FitsPool;
    const int t = q4_recall_tail_pos(pos, prompt_len, tail, pool_tokens);
    if (t < 0) return Q4TailPath::TailInPool;
    if (t >= prompt_len) return Q4TailPath::NoForward;
    return Q4TailPath::TwoPass;
}
// How far a planned two-pass run got; other paths ignore it.
enum class Q4TailStage { Planned, SnapFailed, NothingRecalled, RewindFailed, Rewound };
inline const char * q4_tail_outcome(Q4TailPath p, Q4TailStage s) {
    switch (p) {
        case Q4TailPath::Off:        return "single pass: LUCE_Q4_RECALL_TAIL=0, recall after the prefill";
        case Q4TailPath::NoDecode:   return "single pass: no tokens to generate, no recall";
        case Q4TailPath::FitsPool:   return "single pass: the prompt fits the pool, nothing paged out";
        case Q4TailPath::TailInPool: return "single pass: the tail would start inside the pool (pool < prompt <= pool + tail), recall after the prefill";
        case Q4TailPath::NoForward:  return "single pass: the whole prompt was restored, no prefill forward runs, no queries to recall from";
        case Q4TailPath::TwoPass:
            switch (s) {
                case Q4TailStage::Planned:               return "two-pass: stopped before the rewind";
                case Q4TailStage::SnapFailed:            return "single pass: the tail snapshot failed, recall after the prefill";
                case Q4TailStage::NothingRecalled:       return "single pass: the first pass recalled nothing, so the replay would repeat it";
                case Q4TailStage::RewindFailed:          return "two-pass: the rewind failed, the tail was not replayed";
                case Q4TailStage::Rewound:               return "two-pass: rewound to the tail start, tail prefilled twice";
            }
    }
    return "?";
}

}  // namespace luce::common
