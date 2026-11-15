// Internal interface of the Qwen3.5-0.8B drafter — the current pflash scorer.
//
// The scorer runs on the Qwen3.5 target architecture (TargetWeights,
// build_qwen35_layer): qwen35_loader.cpp loads the GGUF, the optional
// scoring head and the segment probe; qwen35_drafter.cpp runs the two
// scorers. pflash_drafter.cpp owns the public entry points.

#pragma once

#include "pflash_drafter.h"
#include "pflash_selection.h"
#include "pflash_request_probe.h"
#include "pflash_request_stage2.h"
#include "pflash_scoring_resume.h"
#include "pflash_session_offload.h"
#include "common/pflash_types.h"
#include "internal.h"
#include "server/tokenizer.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// Qwen3.5-0.8B scoring head. Features are the residual entering
// full-attention block 15 after the first 15 blocks (twelve GatedDeltaNet and
// three full-attention blocks). Block 15's own Q/K projections score the
// context without RoPE; an optional trained head replaces those two
// projections.
static constexpr int kQwen35HeadBlock = 15;

// What the scorer computed for one conversation's prompt, kept so the next
// turn only runs the new tokens. Blocks 0..14 read left to right, so the
// cache state and block-15 keys of a shared prefix never change; the keys
// carry no position (NoPE), and the probe's raw logits are per token.
struct Qwen35ScoringSession {
    std::vector<int32_t> ids;       // prompt tokens the state covers
    // Recurrent-state snapshot position: pflash_scoring_checkpoint(ids.size()), on the 1024 grid, the only point a
    // later call resumes from (pflash_scoring_resume.h: a resumed call is then bit-identical to a fresh one).
    int checkpoint = 0;
    // Exact F32 block-14 rows [proj_rows_begin, checkpoint) with proj_rows_begin = pflash_scoring_proj_begin(
    // checkpoint) ([hidden] per token): a resume re-projects its whole 8192 chunk from them.
    std::vector<float>    proj_rows;
    int                   proj_rows_begin = 0;
    int capacity = 0;               // tokens the cache and keys can hold
    TargetCache cache;              // blocks 0..14 only
    ggml_context *        key_ctx = nullptr;
    ggml_backend_buffer_t key_buf = nullptr;
    ggml_tensor *         keys = nullptr;   // [head_dim, n_head_kv, capacity] f32
    bool                  keys_trained = false;
    std::vector<float>    probe_raw;        // per token, unit logit
    std::vector<float>    subunit_raw;      // per token, when the probe has one
    // PFLASH_SELECT_QUERY_PROBE=1: the request row's raw logit per token,
    // and block-14 rows of the human messages' request runs (BF16, keyed by
    // token position) for later turns, whose activations are gone.
    std::vector<float>    request_raw;
    luce::pflash::PFlashRequestRowStore request_rows;
    // Stage 2 (PFLASH_REQUEST_STAGE2_GGUF): c(t) of every token ([32] F32 each, from the key graph). Stage 2 forces
    // fresh scoring (no session resume), so this covers the whole prompt of the current call.
    std::vector<float>    stage2_cand;
    // Block-14 output of recent query windows (the query and earlier
    // questions), reused while they sit in the shared prefix: an agent step
    // appends tool output after the same user turn, and a new turn's history
    // queries are earlier turns' queries. Most recent last, at most 8.
    // Keyed by the window's token positions.
    luce::pflash::PFlashExactRowSets query_windows;
    // The query step's (PFLASH_QUERY_CANDIDATES / PFLASH_QUERY_PIN) query token sets, F32 rows exact for later
    // prompts that share them (never the BF16 request_rows).
    luce::pflash::PFlashExactRowSets query_sets;
    uint64_t              last_used = 0;
    // Session offload: the host entry this session was restored from, valid while no scoring call has acquired the
    // session since (a call that acquires it extends or rewrites it, so that copy is stale from then on).
    uint64_t              host_id = 0;
    bool                  host_valid = false;
};

void free_qwen35_scoring_session(Qwen35ScoringSession & session);

// A scoring session while the drafter is released (pflash_session_offload.h): a copy of its host bookkeeping (cache /
// keys fields empty) and of the device bytes a resume reads, in qwen35_session_tensors() order.
struct Qwen35HostSession {
    uint64_t id = 0;
    bool on_device = false;           // restored into a device session during the current load
    Qwen35ScoringSession bookkeeping;
    int cache_max_ctx = 0;
    int cache_cur_pos = 0;
    int cache_last_tok = -1;
    luce::pflash::PFlashHostCopy copy;
};

// Sessions kept on the host by drafter_offload_sessions(), at most the configured session count (most recently
// used), restored one at a time before a scoring call that can resume from it, and only under the identity they
// were stored with (qwen35_session_identity()).
struct DrafterHostSessions {
    std::string identity;
    uint64_t clock = 0;     // Qwen35DrafterState::session_clock at the last offload
    uint64_t next_id = 0;
    std::vector<Qwen35HostSession> sessions;
};

struct Qwen35DrafterState {
    TargetWeights weights;
    std::string gguf_path;
    std::string gguf_sha256;
    int ctx_train = 0;           // the drafter GGUF's <arch>.context_length (0: absent)
    // Longest prompt a scoring session is sized for (pflash_session_ctx_limit of ctx_train and the engine's max
    // context, DrafterContext::engine_max_ctx); set at every scoring call. 0 = no limit.
    int session_ctx_limit = 0;
    ggml_context *        head_ctx = nullptr;
    ggml_backend_buffer_t head_buf = nullptr;
    ggml_tensor *         head_wq  = nullptr;  // [hidden, n_head * head_dim], query rows only
    ggml_tensor *         head_wk  = nullptr;  // [hidden, n_head_kv * head_dim]
    bool                  head_loaded = false;
    // Segment probe: per-token boundary scores from the same block-14 tap.
    ggml_context *        probe_ctx = nullptr;
    ggml_backend_buffer_t probe_buf = nullptr;
    ggml_tensor *         probe_norm_w = nullptr;  // [hidden]
    ggml_tensor *         probe_norm_b = nullptr;  // [hidden]
    ggml_tensor *         probe_fc1_w  = nullptr;  // [hidden, probe_width]
    ggml_tensor *         probe_fc1_b  = nullptr;  // [probe_width]
    ggml_tensor *         probe_fc2_w  = nullptr;  // [probe_width, 1]
    ggml_tensor *         probe_fc2_b  = nullptr;  // [1]
    ggml_tensor *         probe_sub_fc2_w = nullptr; // optional sub-unit head (oversize split only)
    ggml_tensor *         probe_sub_fc2_b = nullptr;
    std::vector<float>    probe_conv_w;            // taps from the GGUF tensor, applied on the CPU
    float                 probe_conv_b = 0.0f;
    std::vector<float>    probe_sub_conv_w;
    float                 probe_sub_conv_b = 0.0f;
    float                 probe_norm_eps = 1e-5f;
    float                 probe_threshold = 0.9f;
    int                   probe_min_segment = 1;
    int                   probe_max_segment = 2048;
    // segmentprobe.oversize_split = "line_aligned": oversize cuts at line
    // starts (pflash_probe_segments with line flags); absent or "best_score":
    // today's split. ``probe_vocab`` (the drafter's own vocabulary) gives the
    // per-token texts the line flags are read from.
    bool                  probe_line_aligned = false;
    std::unique_ptr<Tokenizer> probe_vocab;
    int                   probe_width = 0;
    bool                  probe_loaded = false;
    // Request probe (PFLASH_SELECT_QUERY_PROBE=1 only): its own small
    // trunk on the same block-14 tap, LayerNorm -> fc1 -> GELU(erf) -> fc2,
    // from PFLASH_REQUEST_PROBE_GGUF (pflash_request_probe.h); the raw
    // logit is compared with ``request_tau``. Stub
    // (PFLASH_SELECT_QUERY_PROBE_STUB=1): all zeros and tau = +inf.
    ggml_context *        request_ctx = nullptr;
    ggml_backend_buffer_t request_buf = nullptr;
    ggml_tensor *         request_norm_w = nullptr;  // [hidden]
    ggml_tensor *         request_norm_b = nullptr;  // [hidden]
    ggml_tensor *         request_fc1_w  = nullptr;  // [hidden, request_width]
    ggml_tensor *         request_fc1_b  = nullptr;  // [request_width]
    ggml_tensor *         request_fc2_w  = nullptr;  // [request_width]
    ggml_tensor *         request_fc2_b  = nullptr;  // [1]
    float                 request_norm_eps = 1e-5f;
    float                 request_tau = 0.0f;
    // Candidate rule of the loaded GGUF: false = human messages only, true =
    // the V2-A3 rule (requestprobe.candidate_rule == kRequestCandidateRule).
    bool                  request_turns = false;
    // cand_v5 (requestprobe.candidate_rule == kRequestCandidateRuleV5): seed + score fill; the only rule a
    // stage-2 GGUF pairs with. ``request_checkpoint_sha256`` = requestprobe.checkpoint_sha256 ("" when absent).
    bool                  request_v5 = false;
    std::string           request_checkpoint_sha256;
    int                   request_width = 0;
    bool                  request_loaded = false;
    bool                  request_stub = false;
    // v2 policy (abstention, recency pin): GGUF requestprobe.policy.*, then the PFLASH_REQUEST_PROBE_POLICY JSON
    // file, then PFLASH_SELECT_QUERY_PROBE_RECENCY_PIN. Defaults never abstain and pin nothing.
    luce::pflash::PFlashRequestPolicy request_policy;
    // Stage 2 (PFLASH_REQUEST_STAGE2_GGUF, with a cand_v5 request probe and the scoring head): host weights,
    // the device copies the graphs read, and a host copy of blk.15.attn_norm (x̂ of the seed rows on the CPU).
    bool                  stage2_loaded = false;
    luce::pflash::PFlashStage2Weights stage2;
    luce::pflash::PFlashStage2Device  stage2_dev;
    std::vector<float>    stage2_w_in;
    // Strict scorer sessions, least recently used evicted
    // (PFLASH_DRAFTER_SESSIONS, default 2; 0 scores every prompt from scratch).
    // PFLASH_SELECT_STRATEGY=fold: the drafter's own vocabulary, to read
    // the prompt's structure from its token ids.
    std::unique_ptr<Tokenizer> fold_vocab;
    std::vector<std::unique_ptr<Qwen35ScoringSession>> sessions;
    uint64_t              session_clock = 0;
};

// Defined in qwen35_loader.cpp.
bool qwen35_head_block_available(const TargetWeights & w, std::string & error);
bool load_qwen35_drafter(const std::string & gguf_path, DrafterContext & out);
void free_qwen35_drafter_state(DrafterContext & ctx);

// Defined in qwen35_drafter.cpp.
//
// Causal flash-attention mask of one ubatch of n_tokens queries at kv_start,
// computed by the graph on the device: [align256(kv_start + n_tokens),
// align32(n_tokens)] F16, row q < n_tokens zero over keys
// [0, kv_start + q + 1) and -inf after, rows past n_tokens -inf throughout.
// Its writes are expanded into gf here, ahead of the attention that reads it.
ggml_tensor * build_qwen35_causal_mask(ggml_context * ctx, ggml_cgraph * gf,
                                       int kv_start, int n_tokens);

// The legacy all-layer running-max scorer, on the Qwen3.5 architecture.
std::vector<int32_t> qwen35_score_and_compress(
    TargetWeights & w,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_scores_out = nullptr);

// The block-15 scoring head under strict budget selection.
std::vector<int32_t> qwen35_strict_score_and_compress(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_mass_out = nullptr,
    std::vector<PFlashTokenSpan> * segments_out = nullptr,
    bool * density_out = nullptr);

// Session offload (DrafterContext::host_sessions). The identity names the
// drafter file, this build, the loaded head / probes and the session
// parameters; a stored session resumes only under an equal one.
std::string qwen35_session_identity(const DrafterContext & ctx, const Qwen35DrafterState & st);
// Every device tensor a session owns: blocks 0..14 cache (base, then the
// recurrent-state snapshot), then the block-15 keys.
std::vector<ggml_tensor *> qwen35_session_tensors(const Qwen35ScoringSession & session);
void qwen35_offload_sessions(DrafterContext & ctx);
// Restores the one stored session a request over ``ids`` can resume from (if
// no device session already can) into fresh tensors, or drops it with a
// logged reason; the call then scores as usual. Other stored sessions stay on
// the host untouched.
void qwen35_restore_sessions(DrafterContext & ctx, Qwen35DrafterState & st, const std::vector<int32_t> & ids);

// Arch dispatch target of drafter_score_and_compress.
std::vector<int32_t> qwen35_drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans);

} // namespace luce::common
