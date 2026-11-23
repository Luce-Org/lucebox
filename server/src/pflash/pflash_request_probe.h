// Request probe: the scorer query comes from
// the human user's own request text, found per token by a small probe of
// its own on the segment probe's block-14 tap, computed in the same graph
// pass: LayerNorm -> fc1 -> GELU (erf) -> fc2, one raw logit per token,
// kept per scoring session like the segment probe's (probe_raw).
//
// Its GGUF (PFLASH_REQUEST_PROBE_GGUF; general.architecture
// "requestprobe", schema "qwen3_5_0_8b_request_probe_v1") carries F32
//   requestprobe.norm.weight / .norm.bias   [hidden]
//   requestprobe.fc1.weight [hidden, width] / requestprobe.fc1.bias [width]
//   requestprobe.fc2.weight [width] / requestprobe.fc2.bias [1]
// and the metadata requestprobe.norm_eps, requestprobe.tau (a token is
// request text when its RAW logit >= tau) and requestprobe.extraction_version
// (string, must equal kRequestExtractionVersion), plus base_model,
// runtime_gguf_sha256 and feature_tap as for the segment probe.
//
// The model-free part lives here so tests run the runtime's own code:
//   - pflash_request_runs        maximal runs of raw logit >= tau, >= 2 tokens,
//                                inside one human user message
//   - pflash_request_query_set   the role rule: the latest human message's
//                                runs, plus the previous one's when the
//                                latest has < 8 request tokens; capped at
//                                1024 tokens, latest first
//   - PFlashRequestRowStore      block-14 rows of request runs kept per
//                                session in BF16 (cap 2048 tokens, whole
//                                turns evicted least recently used), keyed
//                                by exact token position
//
// "Human" is a message role, never text: the server passes the content
// spans of the messages whose role is "user" after normalisation (tool
// results arrive as role "tool" from every API format, also when the
// template renders them inside a user turn).

#pragma once

#include "common/pflash_types.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace luce::pflash {

using luce::common::PFlashTokenSpan;

inline constexpr int kRequestMinRun = 2;        // shorter runs are dropped ("why?" survives)
// Version of this extraction rule (min run 2, >= on the raw logit, the role
// rule, the cap; SPEC amendment A4). A request-probe GGUF carries the
// version its tau was calibrated for in requestprobe.extraction_version
// (string, the trainer's contract.EXTRACTION_VERSION); the loader refuses
// any other value.
inline constexpr const char * kRequestExtractionVersion =
    "request_runs_v2_minrun2_ge_raw_logit";
// PLAN-V2 amendments V2-A3 / V2-A5: one candidate rule for chat AND agents,
// the trainer's contract.CANDIDATE_RULE: candidates = the latest human
// message (+ the previous one when the latest has fewer than
// kRequestPreviousBelow request tokens) PLUS ALL non-system messages
// (assistant prose and tool calls, tool results, user; V2-A5 dropped the
// fixed 3-message window);
// runs inside each candidate message; the query takes the candidate messages
// latest first, each message's tail first, capped at kRequestQueryCap
// (pflash_request_query_set_v2 = rule.probe_query_v2). A GGUF opts in with
// the metadata key kRequestCandidateRuleKey == kRequestCandidateRule; without
// the key the human-only rule (pflash_request_query_set) applies.
inline constexpr const char * kRequestCandidateRuleKey = "requestprobe.candidate_rule";
inline constexpr const char * kRequestCandidateRule =
    "cand_v3_latest_human_prev8_plus_all_nonsystem_turns";
inline constexpr int kRequestPreviousBelow = 8; // latest < 8 tokens -> add previous
inline constexpr int kRequestQueryCap = 1024;   // query tokens, latest first
inline constexpr int kRequestRowCap = 2048;     // retained rows per session
inline constexpr int kRequestMessages = 2;      // human messages the rule reads
// STAGE2-CONTRACT.md (v3) section 2: candidate rule cand_v5 (option A, score-based slot fill with the post-latest
// reserve): R = min(kRequestPostReserve, |post-latest run tokens|); the seed is the latest user message's run tokens
// capped at kRequestQueryCap - R by score; then the R best post-latest run tokens; then the run tokens of every other
// non-system message by score until the cap; ascending union. A GGUF opts in with requestprobe.candidate_rule ==
// kRequestCandidateRuleV5 (pflash_request_query_set_v5 = rule.probe_query_v5); its
// requestprobe.checkpoint_sha256 identifies the checkpoint.
inline constexpr const char * kRequestCandidateRuleV5 =
    "cand_v5_latest_user_seed_reserve256_postlatest_then_score_fill";
inline constexpr const char * kRequestCheckpointShaKey = "requestprobe.checkpoint_sha256";
inline constexpr int kRequestPostReserve = 256;

// Maximal runs of tokens t in ``message`` with logits[t] >= tau (a NaN
// breaks a run), dropping runs shorter than ``min_run``. Empty when the
// message is out of range of ``logits``.
std::vector<PFlashTokenSpan> pflash_request_runs(
    const std::vector<float> & logits,
    PFlashTokenSpan message,
    float tau,
    int min_run = kRequestMinRun);

// Token positions of ``runs``, ascending.
std::vector<int> pflash_request_run_tokens(const std::vector<PFlashTokenSpan> & runs);

// The last ``n`` of ascending ``tokens`` (all of them when n >= size, none
// when n <= 0).
std::vector<int> pflash_request_cap_tail(const std::vector<int> & tokens, int n);

// Ascending positions -> merged half-open spans.
std::vector<PFlashTokenSpan> pflash_request_token_spans(const std::vector<int> & tokens);

struct PFlashRequestMessageRuns {
    PFlashTokenSpan message{-1, -1};
    std::vector<PFlashTokenSpan> runs;
    int request_tokens = 0;     // tokens in runs, before the cap
    int message_tokens = 0;
    int logits_at_or_above_tau = 0;
    float logit_min = 0.0f;
    float logit_max = 0.0f;
    double logit_mean = 0.0;
};

struct PFlashRequestQuerySet {
    // Ascending, deduplicated, capped query tokens; empty -> fallback.
    std::vector<int> tokens;
    // Human messages read, latest first (at most kRequestMessages).
    std::vector<PFlashRequestMessageRuns> messages;
    bool previous_used = false;
    int latest_tokens = 0;      // query tokens taken from the latest message
    int previous_tokens = 0;    // and from the previous one
    int uncapped_tokens = 0;
    bool capped = false;
    // Empty when ``tokens`` is not; else why the request falls back to
    // today's windows: "no_human_message", "logits_missing",
    // "no_request_runs".
    std::string fallback;
    // Candidate rule: the other candidate messages read (the non-system
    // messages that are not the human messages above), latest first, with
    // their message indices, and the query tokens taken from them.
    bool turns_read = false;
    std::vector<PFlashRequestMessageRuns> turns;
    std::vector<int> turn_messages;
    int turn_tokens = 0;
    // The candidate rule string the set was built under (trace only; "" = human-only rule).
    std::string rule;
    // cand_v5 only: the latest user message's index (-1 = none), its seed tokens (ascending) and whether the
    // 1024 cap cut them.
    int latest_message = -1;
    std::vector<int> seed;
    bool seed_capped = false;
    int reserve = 0;               // cand_v5: R, the post-latest slots reserved
};

// One message of the normalised list for cand_v5 / stage 2 (STAGE2-CONTRACT.md section 1): its role and its
// content span (tokens overlapping the message's exact content characters; empty = no tokens).
enum class PFlashRequestRole { System, User, Assistant, Tool };
struct PFlashRequestMessage {
    PFlashRequestRole role = PFlashRequestRole::System;
    PFlashTokenSpan span{0, 0};
};
// "system" / "user" / "assistant" / "tool" -> role; false for any other name.
bool pflash_request_role_from_name(const std::string & name, PFlashRequestRole & role);
const char * pflash_request_role_name(PFlashRequestRole role) noexcept;

// ``tokens`` deduplicated and ordered by (score descending, position descending): exact F32 ties go to the later
// position (contract section 1, rule.by_score).
std::vector<int> pflash_request_by_score(const std::vector<int> & tokens, const std::vector<float> & scores);
// Run tokens (ascending) of ONE message span: maximal runs of score >= tau, NaN breaks, min run 2.
std::vector<int> pflash_request_span_run_tokens(const std::vector<float> & scores, PFlashTokenSpan span, float tau);
// Index of the last message with role user, -1 when there is none.
int pflash_request_latest_user(const std::vector<PFlashRequestMessage> & messages);
// The seed: the latest user message's run tokens of ``s1`` at ``tau``, capped to ``cap`` by score; ascending.
// ``capped`` (optional) is set when the cap cut some.
std::vector<int> pflash_request_seed(const std::vector<float> & s1, float tau,
                                     const std::vector<PFlashRequestMessage> & messages, int latest, int cap,
                                     bool * capped = nullptr);
// Contract v3 (rule.capped_seed): P = the stage-1 run tokens of the non-system messages after the latest user
// message (none without one), R = min(reserve, |P|), and the seed capped at cap - R. Returns the seed; ``reserve_out``
// = R, ``post_out`` = P (ascending), ``capped`` as pflash_request_seed.
std::vector<int> pflash_request_capped_seed(const std::vector<float> & s1, float tau,
                                            const std::vector<PFlashRequestMessage> & messages, int cap, int reserve,
                                            int & reserve_out, std::vector<int> & post_out, bool * capped = nullptr);

// cand_v5 (rule.probe_query_v5). ``set.messages`` holds the latest user message's runs (when there is one),
// ``set.turns`` / ``turn_messages`` every other non-system message's; fallbacks "logits_missing" (a span outside
// the logits) and "no_request_runs" (empty query).
PFlashRequestQuerySet pflash_request_query_set_v5(
    const std::vector<float> & logits,
    const std::vector<PFlashRequestMessage> & messages,
    float tau,
    int cap = kRequestQueryCap,
    int reserve = kRequestPostReserve);

// A request-probe candidate message for the V2-A3 rule: its index in the
// normalised message list, whether it is human (role "user"), whether it is
// a non-system message (every one is a candidate under the rule), and its
// content span.
struct PFlashRequestCandidate {
    int message_index = -1;
    bool human = false;
    bool recent = false;
    PFlashTokenSpan span{-1, -1};
};

// The role rule over ``human_messages``. The < kRequestPreviousBelow test
// counts the latest message's request tokens after run extraction (runs
// under min_run already dropped) and before the cap.
// (content spans of the human user
// messages, latest first; only the first kRequestMessages are read; an
// empty span is a message without request tokens).
PFlashRequestQuerySet pflash_request_query_set(
    const std::vector<float> & logits,
    const std::vector<PFlashTokenSpan> & human_messages,
    float tau);

// The candidate rule (trainer rule.probe_query_v2). ``messages`` lists the
// latest two human messages and every non-system message (one entry per
// message index; any order; ``recent`` marks the non-system candidates). Fallbacks: "logits_missing"
// (a span outside the logits), "no_request_runs" (no token anywhere),
// "no_human_message" (no human message and no recent message).
PFlashRequestQuerySet pflash_request_query_set_v2(
    const std::vector<float> & logits,
    const std::vector<PFlashRequestCandidate> & messages,
    float tau,
    int cap = kRequestQueryCap);   // the runtime always uses kRequestQueryCap; replay parity varies it

// v2 policy (PLAN-V2 W4), all optional; the defaults never abstain and pin nothing, so a probe without policy
// metadata behaves exactly like v1.
//  - Calibrated abstention: when the extraction looks unreliable, the request falls back to TODAY's complete
//    behaviour (windows, scoring, pins), reason "abstain:<criterion>". Criteria, in this order:
//      prompt_tokens outside [min_prompt_tokens, max_prompt_tokens]        ("length")
//      mean (logit - tau) over the query tokens < min_mean_margin           ("mean_margin")
//      max (logit - tau) over the query tokens < min_max_margin             ("max_margin")
//      share of the latest human message's tokens in its runs (before the cap) outside
//        [min_latest_share, max_latest_share]                               ("latest_share")
//      runs in the messages used (latest, plus previous when used) > max_runs ("runs")
//  - Bounded recency PIN: the last recency_pin_tokens tokens of the latest human message are pinned (never scored),
//    merged with the query pins (no double charge) and charged to the budget like them. Only when the request set is
//    used (not on a fallback or abstention).
// Source: the request-probe GGUF metadata requestprobe.policy.<key> (set from the CALIBRATION pool).
struct PFlashRequestPolicy {
    int min_prompt_tokens = 0;           // 0 = no lower bound
    int max_prompt_tokens = 0;           // 0 = no upper bound
    float min_mean_margin = -std::numeric_limits<float>::infinity();
    float min_max_margin = -std::numeric_limits<float>::infinity();
    float min_latest_share = 0.0f;
    float max_latest_share = 1.0f;
    int max_runs = 0;                    // 0 = no limit
    int recency_pin_tokens = 0;          // 0 = no pin
};

struct PFlashRequestAbstainStats {
    std::string reason;                  // "" = no abstention, else the criterion
    float mean_margin = 0.0f;
    float max_margin = 0.0f;
    float latest_share = 0.0f;
    int runs = 0;
    int prompt_tokens = 0;
};

// Validates a policy (bounds ordered, shares in [0, 1], counts >= 0).
bool pflash_request_policy_valid(const PFlashRequestPolicy & policy, std::string & error);

// The abstention decision for a NON-EMPTY query set (with its logits and tau).
PFlashRequestAbstainStats pflash_request_abstain(
    const PFlashRequestQuerySet & set,
    const std::vector<float> & logits,
    float tau,
    int prompt_tokens,
    const PFlashRequestPolicy & policy);

// The recency pin: the last ``k`` tokens of ``latest`` (the latest human message's span); {-1,-1} when k <= 0 or the
// message is empty.
PFlashTokenSpan pflash_request_recency_pin(PFlashTokenSpan latest, int k);

struct PFlashQueryTrace;  // pflash_query_candidates.h

// What the request probe did on one scoring call, for the compression
// trace ("request_probe" object) and the log line.
struct PFlashRequestProbeTrace {
    // Relative query candidates + verbatim pin, not owned; nullptr when not run.
    const PFlashQueryTrace * query = nullptr;
    float tau = 0.0f;
    PFlashRequestQuerySet set;          // messages, runs, capped query tokens
    // "" when the gathered set was scored; else the fallback to today's
    // windows: the set's own reason, "message_mapping: <error>",
    // "rows_missing", "pins_exceed_budget".
    std::string fallback;
    int query_tokens = 0;               // scored query tokens (0 on fallback)
    int rows_new = 0;                   // query rows computed on this call
    int rows_stored = 0;                // query rows read from the session
    int pinned_new_tokens = 0;          // query tokens not pinned otherwise
    int mandatory_tokens = 0;           // selector-charged tokens with the pins
    int mandatory_tokens_base = 0;      // the same without the query pins
    int token_budget = 0;
    int store_turns = 0;                // session row store after this call
    int store_tokens = 0;
    int store_evicted_turns = 0;
    int store_invalidated_turns = 0;
    int query_tiles = 0;
    // v2 policy: abstention statistics (always computed when a set was built) and the recency pin.
    bool policy_evaluated = false;
    PFlashRequestAbstainStats abstain;
    PFlashTokenSpan recency_pin{-1, -1};
    int recency_pin_new_tokens = 0;     // pin tokens not already pinned (query pins or the caller's)
};

// The rows a message's turn retains: its request tokens capped at
// kRequestQueryCap, latest first (what any later query set can take from
// it).
std::vector<int> pflash_request_turn_tokens(const PFlashRequestMessageRuns & message);

// IEEE float <-> bfloat16, round to nearest even (NaN stays NaN).
uint16_t pflash_f32_to_bf16(float value);
float pflash_bf16_to_f32(uint16_t value);

// Block-14 rows of request runs, per scoring session. A turn is the
// retained token set of one human message; turns are replaced whole and
// evicted whole, least recently used first, until at most ``cap_tokens``
// rows remain. Rows are valid only while the session's prompt keeps the
// same prefix: ``invalidate_from`` drops every turn with a position at or
// past a divergence or truncation point.
class PFlashRequestRowStore {
public:
    struct Turn {
        PFlashTokenSpan message{-1, -1};
        std::vector<int> positions;     // ascending
        std::vector<uint16_t> rows;     // [hidden, positions.size()] bf16
        uint64_t last_used = 0;
    };

    explicit PFlashRequestRowStore(int cap_tokens = kRequestRowCap)
        : cap_tokens_(cap_tokens) {}

    void clear();
    // Drops the turns with any position >= ``valid_end``; returns how many.
    int invalidate_from(int valid_end);
    // True when every position has a stored row.
    bool covers(const std::vector<int> & positions) const;
    // Rows of ``positions`` into ``out`` ([hidden, n] f32). False (out
    // untouched beyond resizing) when any is missing. Marks the turns used.
    bool lookup(const std::vector<int> & positions, int hidden,
                std::vector<float> & out);
    // Stores ``rows`` ([hidden, positions.size()] f32) as the turn of
    // ``message``, replacing a turn of the same message or sharing a
    // position, then evicts least recently used turns over the cap (never
    // the one just stored; a turn over the cap alone keeps its last
    // cap_tokens rows). Returns the turns evicted.
    int put(PFlashTokenSpan message, const std::vector<int> & positions,
            const float * rows, int hidden);

    int tokens() const;
    int hidden() const { return hidden_; }
    const std::vector<Turn> & turns() const { return turns_; }

private:
    const Turn * find_turn(int position) const;

    int cap_tokens_;
    int hidden_ = 0;
    uint64_t clock_ = 0;
    std::vector<Turn> turns_;
};

} // namespace luce::pflash
