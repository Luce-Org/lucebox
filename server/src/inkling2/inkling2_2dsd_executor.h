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
#include <memory>
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
    // 3DSD levers (all lossless except opt-in early_exit_p > 0):
    bool  tree_branch      = false; // opt-in: branch top-2 Markov candidate at first uncertain draft slot
    float branch_conf_low  = 0.25f;
    float branch_conf_high = 0.72f;
    bool  prompt_lookup    = true;  // 0.01 ms n-gram lookup injection into 2DSD trie (skips GPU drafter on match!)
    int   lookup_max_k     = 16;    // max prompt-lookup chain length per anchor
    float early_exit_p     = 0.0f;  // opt-in varloop d2 early exit (>0, e.g. 0.98; 0 = OFF / 100% bit-exact R=3)
};

struct Inkling2TwoDSDStats {
    long long target_calls      = 0;
    long long target_cells      = 0;
    long long drafter_batches   = 0;
    long long drafter_calls     = 0;
    long long draft_slots       = 0;
    long long draft_slots_cut   = 0;
    long long draft_branches    = 0;
    long long lookup_hits       = 0;
    long long lookup_slots      = 0;
    long long early_exits       = 0;
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

// Optional per-node depth-R sampling payload captured when running non-greedy 2DSD.
// Stores either compact subop-32 GPU top-candidates (516 i32 = 2 KB) or a full logits row fallback.
struct Inkling2TwoDSDDepthRSample {
    std::vector<int32_t> top_cands; // 516 ints from EXACT_GPU_TOP_CANDS when fast path active
    std::vector<float>   logits;    // full row fallback (only populated if top_cands empty or flat)
};

// Callbacks for lossless non-greedy 2DSD:
//   - capture(readout_idx, out): called when a node reaches depth==R during advance_nodes (before verify).
//   - sample(payload, greedy_tok): called inside verify ONLY along the committed path (exact AR RNG order!).
struct Inkling2TwoDSDSamplerHooks {
    std::function<void(int readout_idx, Inkling2TwoDSDDepthRSample & out)> capture;
    std::function<int32_t(const Inkling2TwoDSDDepthRSample & payload, int32_t greedy_tok)> sample;
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

// Step-wise 2DSD Trie Session (used by both single-stream decode and multi-stream Inkling2SeqEngine).
// Separates trie state transitions from batched GPU target/drafter execution so N concurrent slots
// can pack their active nodes and draft requests into one shared GPU step.
class Inkling2TwoDSDSession {
public:
    Inkling2TwoDSDSession(Inkling2TwoDSDRunner & runner,
                          const Inkling2TwoDSDConfig & cfg,
                          Inkling2TwoDSDStats & stats,
                          int output_limit,
                          const Inkling2TwoDSDSamplerHooks * sampler = nullptr);
    ~Inkling2TwoDSDSession();

    Inkling2TwoDSDSession(const Inkling2TwoDSDSession &) = delete;
    Inkling2TwoDSDSession & operator=(const Inkling2TwoDSDSession &) = delete;

    uint32_t readout_mask() const;

    // Seed the initial cursor at `first_token` (already committed at prefix_len - 1) and queue its anchor.
    bool seed_cursor(int32_t first_token, int prefix_len_before_first, std::string & error);

    // Seed committed token history for 0.01 ms n-gram prompt-lookup drafting (optional).
    void set_prompt_history(const std::vector<int32_t> & prompt_tokens);

    // Provide per-readout max softmax probability (temp=1) from the latest advance_nodes call
    // for opt-in native varloop d2 early exit (`cfg.early_exit_p > 0`).
    void set_last_readout_pmax(const std::vector<float> & pmax);

    // Phase A: collect pending draft requests (without running the drafter) and apply returned drafts.
    void collect_draft_reqs(std::vector<Inkling2TwoDSDDraftReq> & out_reqs,
                            std::vector<int> & out_plan_heads,
                            std::vector<int> & out_plan_widths);
    void apply_draft_outs(const std::vector<int> & plan_heads,
                          const std::vector<int> & plan_widths,
                          const std::vector<Inkling2TwoDSDDraftOut> & outs);

    // Phase B: collect all active nodes (depth < R) to advance by +1 loop.
    bool collect_active(std::vector<int> & out_active_ids,
                        std::vector<Inkling2TwoDSDNodeDesc> & out_descs,
                        std::string & error) const;

    // Phase C: apply target readouts, verify committed path, and run Stage 2 / Stage 1 transitions.
    // Returns false on fatal error; sets `stopped = true` when `on_commit` returns false.
    bool apply_step_and_verify(const std::vector<int> & active_ids,
                               const std::vector<int> & readout_rows,
                               const std::vector<int32_t> & readout_argmax,
                               const std::function<bool(int32_t &)> & on_commit,
                               bool & stopped,
                               std::string & error);

    // Update effective config (e.g. concurrency-adaptive draft_confidence / stage when N changes).
    void update_config(const Inkling2TwoDSDConfig & cfg);

    int live_node_count() const;
    int32_t cursor_token() const;
    int cursor_pos() const;
    int cursor_depth() const;
    void clear_cursor_children();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
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

// Run 2DSD decode with optional lossless stochastic sampling hooks (`sampler == nullptr` => greedy),
// optional prompt token history for 0.01 ms prompt-lookup drafting, and optional pmax provider for
// opt-in varloop d2 early-exit.
bool inkling2_2dsd_run(
    Inkling2TwoDSDRunner & runner,
    const Inkling2TwoDSDConfig & cfg,
    int32_t first_token,
    int max_new,
    const std::function<bool(int32_t &)> & on_commit,
    const Inkling2TwoDSDSamplerHooks * sampler,
    Inkling2TwoDSDStats & stats,
    std::string & error,
    const std::vector<int32_t> * prompt_history = nullptr,
    const std::vector<float> * last_pmax = nullptr);

// Parse stage string ("stage1", "stage2a", "stage2b") and env overrides.
Inkling2TwoDSDConfig inkling2_2dsd_config_from_env(int R, int block_size);

// Concurrency-adaptive 2DSD config scaling confidence & block width with active slot count `N`
// to keep combined MoE batch width in the optimal 24..48 cell sweet spot.
Inkling2TwoDSDConfig inkling2_2dsd_config_for_concurrency(int R, int block_size, int active_slots);

} // namespace luce::common
