// 2DSD (Two-Dimensional Speculative Decoding) Greedy Tree Executor with Draft Confidence Pruning.
//
// Implements FinalizedTreeExecutor (stage2b default: d1=1, d2=2, R=3, k=8) over an abstract
// runner interface so both the live GPU backend and deterministic CPU unit tests share the
// exact same state machine and trie invariants.

#pragma once

#include "inkling2_2dsd_draft.h"
#include "inkling2_2dsd_graph.h"
#include "common/generation_types.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace luce::common {

enum class Inkling2TwoDSDStage {
    Stage1  = 1, // early review at d1 only
    Stage2a = 2, // + frozen-suffix promotion at d2
    Stage2b = 3, // + deeper forking at d2 (default)
};

struct Inkling2TwoDSDConfig {
    Inkling2TwoDSDStage stage = Inkling2TwoDSDStage::Stage2b;
    int   R                = 3;
    int   d1               = 1;
    int   d2               = 2;
    int   k                = 8;     // block size (1 anchor + k-1 drafted tokens)
    float draft_confidence = 0.0f;  // >0: truncate drafted block before first slot with conf < threshold
    bool  frozen_suffix    = true;
};

struct Inkling2TwoDSDStats {
    long long target_calls      = 0;
    long long target_cells      = 0;
    long long drafter_batches   = 0;
    long long drafter_calls     = 0;
    long long draft_slots       = 0;
    long long draft_slots_cut   = 0;
    long long commit_main       = 0;
    long long commit_frozen     = 0;
    long long commit_fork       = 0;
    long long commit_bonus      = 0;
    long long commit_draft      = 0;
    long long promotions        = 0;
    long long forks             = 0;
    int       max_live_nodes    = 0;
    double    draft_s           = 0.0;
};

// Abstract target+drafter runner interface (implemented by GPU backend and unit-test mocks).
class Inkling2TwoDSDRunner {
public:
    virtual ~Inkling2TwoDSDRunner() = default;
    virtual int  prefix_length() const = 0;
    virtual int  max_nodes() const = 0;
    virtual int  max_ctx() const = 0;
    virtual bool advance_nodes(const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
                               uint32_t readout_mask,
                               std::vector<int> & out_readout_rows,
                               std::vector<int32_t> & out_readout_argmax) = 0;
    virtual bool commit_nodes(const int * commit_slots, int n_commit) = 0;
    virtual bool draft_batch(const std::vector<Inkling2TwoDSDDraftReq> & requests,
                             std::vector<Inkling2TwoDSDDraftOut> & out) = 0;
};

// Run greedy 2DSD decode starting from `first_token` (already sampled from prompt prefill at
// `prefix_len - 1`). Invokes `on_commit(tok)` for each emitted token (including `first_token`),
// stopping when `on_commit` returns false (EOS, max_new, or cancellation).
bool inkling2_2dsd_run_greedy(
    Inkling2TwoDSDRunner & runner,
    const Inkling2TwoDSDConfig & cfg,
    int32_t first_token,
    int max_new,
    const std::function<bool(int32_t &)> & on_commit,
    Inkling2TwoDSDStats & stats,
    std::string & error);

// Parse stage string ("stage1", "stage2a", "stage2b") and env overrides.
Inkling2TwoDSDConfig inkling2_2dsd_config_from_env(int R, int block_size);

} // namespace luce::common
