// cluster_decision_hooks.h - "rank 0 decides, all ranks follow" hooks for
// the DeepSeek4 generate loop.
//
// Every rank of a cluster runs the identical generate() call in lockstep. The
// only places where the ranks could diverge are host-side decisions: which
// token was sampled, whether EOS/stop/cancel ended the request, which draft
// tokens DSpark proposes and how many the target accepted. Those decisions
// are made once on the head and shipped to the workers over the control
// channel (cluster_control.h) as Decision / Draft / Accept frames.
//
// The backend calls these hooks at the points where upstream Lucebox samples
// or checks for termination. Three implementations exist:
//
//   LocalHooks   - cluster disabled. Every call is a no-op that returns true
//                  and leaves the caller's token/flags untouched, so the
//                  backend keeps exactly one code path and upstream behaviour
//                  stays byte-identical.
//   HeadHooks    - rank 0. The caller has already sampled (host sampler,
//                  deterministic RNG on rank 0 only) and computed the flags;
//                  the hook broadcasts DecisionMsg and, every
//                  verify_hash_every steps, gathers HashProbe frames from the
//                  workers and compares them with its own hash.
//   WorkerHooks  - ranks 1..N-1. Blocks on the head's frame with the
//                  configured deadline, validates request_id/step, and hands
//                  the head's token/flags back to the caller. Workers never
//                  sample and never evaluate EOS/cancel themselves.
//
// All methods return false and fill *err on channel failure or protocol
// violation; nothing here throws across the backend boundary.

#pragma once

#include "cluster/cluster_protocol.h"
#include "common/sampler.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace luce::cluster {

class ClusterHeadControl;
class ClusterWorkerControl;

// Counters kept by every hook implementation; the worker main loop folds
// them into RequestReportMsg, the head decorator into ClusterRequestReport.
struct HookCounters {
    uint64_t decisions        = 0;   // Decision frames sent (head) / received (worker)
    uint64_t drafts           = 0;
    uint64_t accepts          = 0;
    uint64_t hash_probes      = 0;
    uint64_t hash_mismatches  = 0;   // head only
    uint64_t backend_ops      = 0;
    uint64_t ctrl_wait_us     = 0;   // time blocked on the control channel

    void reset() { *this = HookCounters{}; }
};

// Compose DecisionFlags from the caller's termination checks.
inline uint8_t make_decision_flags(bool eos, bool stop, bool cancel, bool budget, bool is_final) {
    uint8_t f = 0;
    if (eos)      f |= kDecisionEos;
    if (stop)     f |= kDecisionStop;
    if (cancel)   f |= kDecisionCancel;
    if (budget)   f |= kDecisionBudget;
    if (is_final) f |= kDecisionFinal;
    return f;
}

// True when the flags end the request on every rank.
inline bool decision_terminates(uint8_t flags) {
    return (flags & (kDecisionEos | kDecisionStop | kDecisionCancel | kDecisionFinal)) != 0;
}

struct Ds4ClusterHooks {
    virtual ~Ds4ClusterHooks() = default;

    // AR decode, one call per step on every rank.
    //
    // Head / local: `token` and `flags` are INPUTS. The caller sampled
    // `token` from `logits` with `cfg` (including any budget-hook override)
    // and computed `flags` with make_decision_flags. If the caller passes
    // token < 0 the head hook samples itself (greedy argmax, or `cfg` via
    // common::sample_logits with an internal RNG when cfg.temp > 0); this
    // path exists for callers that do not sample host-side.
    // Worker: `logits`/`cfg` are ignored; `token` and `flags` are OUTPUTS
    // taken from the head's DecisionMsg for (req, step).
    // `is_eos_candidate_unused` is reserved (kept for signature stability).
    virtual bool decide_next_token(uint64_t req, uint32_t step,
                                   const float * logits, int n_vocab,
                                   const common::SamplerCfg & cfg,
                                   bool is_eos_candidate_unused,
                                   int32_t & token, uint8_t & flags,
                                   std::string * err) = 0;

    // DSpark: head broadcasts the drafted block `tokens` (seed +
    // candidates, tokens[0] at target position `pos`); worker fills it.
    virtual bool decide_draft(uint64_t req, uint32_t step, int pos,
                              std::vector<int32_t> & tokens, std::string * err) = 0;

    // DSpark: head broadcasts accept count, bonus token and flags;
    // worker fills them.
    virtual bool decide_accept(uint64_t req, uint32_t step,
                               int32_t & accept, int32_t & bonus, uint8_t & flags,
                               std::string * err) = 0;

    // Determinism probe (--cluster-verify-hash n). Worker sends its hashes;
    // head gathers one HashProbe per worker and compares against its own.
    // A mismatch is logged (first divergent rank and step) and counted; it
    // is fatal only when strict mode is enabled on the head hooks.
    virtual bool hash_probe(uint64_t req, uint32_t step, uint64_t hc_hash, uint64_t tok_hash,
                            std::string * err) = 0;

    // Replicated backend mutation. Head: broadcast to all workers. Worker:
    // no-op (the worker main loop already executed the op it received).
    virtual bool backend_op(const BackendOpMsg & op, std::string * err) = 0;

    virtual bool is_head() const = 0;
    // False for LocalHooks: the backend uses this to keep upstream cancel /
    // early-return semantics when no cluster is configured.
    virtual bool is_cluster() const = 0;

    // Steps between hash probes; 0 = off.
    virtual int verify_hash_every() const { return 0; }

    virtual const HookCounters & counters() const { return counters_; }
    virtual void reset_counters() { counters_.reset(); }

    // The request id the head decorator / worker main loop assigned to the
    // generate() call currently in flight. The backend passes it back on
    // every call so frames can be validated.
    void set_current_request(uint64_t id) { current_request_ = id; }
    uint64_t current_request() const { return current_request_; }

protected:
    HookCounters counters_;
    uint64_t     current_request_ = 0;
};

// ─── LocalHooks ────────────────────────────────────────────────────────
class LocalHooks final : public Ds4ClusterHooks {
public:
    bool decide_next_token(uint64_t, uint32_t, const float *, int, const common::SamplerCfg &,
                           bool, int32_t &, uint8_t &, std::string *) override { return true; }
    bool decide_draft(uint64_t, uint32_t, int, std::vector<int32_t> &, std::string *) override {
        return true;
    }
    bool decide_accept(uint64_t, uint32_t, int32_t &, int32_t &, uint8_t &, std::string *) override {
        return true;
    }
    bool hash_probe(uint64_t, uint32_t, uint64_t, uint64_t, std::string *) override { return true; }
    bool backend_op(const BackendOpMsg &, std::string *) override { return true; }
    bool is_head() const override { return true; }
    bool is_cluster() const override { return false; }
};

// ─── HeadHooks ─────────────────────────────────────────────────────────
struct HeadHooksConfig {
    uint32_t timeout_ms        = 30000;  // deadline for gather()
    int      verify_hash_every = 0;
    bool     strict_hash       = false;  // mismatch -> return false
    bool     trace             = false;  // LUCE_CLUSTER_TRACE
};

class HeadHooks final : public Ds4ClusterHooks {
public:
    // `control` must outlive the hooks. Optional sampler used only when the
    // caller passes token < 0.
    HeadHooks(ClusterHeadControl & control, const HeadHooksConfig & cfg);

    using Sampler = std::function<int32_t(const float * logits, int n_vocab,
                                          const common::SamplerCfg & cfg)>;
    void set_sampler(Sampler s) { sampler_ = std::move(s); }

    bool decide_next_token(uint64_t req, uint32_t step, const float * logits, int n_vocab,
                           const common::SamplerCfg & cfg, bool is_eos_candidate_unused,
                           int32_t & token, uint8_t & flags, std::string * err) override;
    bool decide_draft(uint64_t req, uint32_t step, int pos, std::vector<int32_t> & tokens,
                      std::string * err) override;
    bool decide_accept(uint64_t req, uint32_t step, int32_t & accept, int32_t & bonus,
                       uint8_t & flags, std::string * err) override;
    bool hash_probe(uint64_t req, uint32_t step, uint64_t hc_hash, uint64_t tok_hash,
                    std::string * err) override;
    bool backend_op(const BackendOpMsg & op, std::string * err) override;
    bool is_head() const override { return true; }
    bool is_cluster() const override { return true; }
    int verify_hash_every() const override { return cfg_.verify_hash_every; }

    // First mismatch seen since reset_counters(); rank -1 = none.
    int      first_mismatch_rank() const { return first_mismatch_rank_; }
    uint32_t first_mismatch_step() const { return first_mismatch_step_; }
    void reset_counters() override;

private:
    ClusterHeadControl & control_;
    HeadHooksConfig      cfg_;
    Sampler              sampler_;
    int                  first_mismatch_rank_ = -1;
    uint32_t             first_mismatch_step_ = 0;
};

// ─── WorkerHooks ───────────────────────────────────────────────────────
struct WorkerHooksConfig {
    int      rank              = -1;
    uint32_t timeout_ms        = 30000;  // deadline for every blocking recv
    int      verify_hash_every = 0;
};

class WorkerHooks final : public Ds4ClusterHooks {
public:
    WorkerHooks(ClusterWorkerControl & control, const WorkerHooksConfig & cfg);

    bool decide_next_token(uint64_t req, uint32_t step, const float * logits, int n_vocab,
                           const common::SamplerCfg & cfg, bool is_eos_candidate_unused,
                           int32_t & token, uint8_t & flags, std::string * err) override;
    bool decide_draft(uint64_t req, uint32_t step, int pos, std::vector<int32_t> & tokens,
                      std::string * err) override;
    bool decide_accept(uint64_t req, uint32_t step, int32_t & accept, int32_t & bonus,
                       uint8_t & flags, std::string * err) override;
    bool hash_probe(uint64_t req, uint32_t step, uint64_t hc_hash, uint64_t tok_hash,
                    std::string * err) override;
    bool backend_op(const BackendOpMsg &, std::string *) override { return true; }
    bool is_head() const override { return false; }
    bool is_cluster() const override { return true; }
    int verify_hash_every() const override { return cfg_.verify_hash_every; }

    // When a recv expected a Decision/Draft/Accept but got a different frame
    // (RequestEnd because the head failed the request, Abort, Shutdown), the
    // frame is parked here so the worker main loop can act on it instead of
    // blocking for it a second time. Empty when nothing is parked.
    bool take_pending_frame(Frame & out);
    // Reason of the last Abort frame received, if any.
    const std::string & abort_reason() const { return abort_reason_; }
    bool aborted() const { return aborted_; }

private:
    // Receives the next non-heartbeat frame; returns false if it is not of
    // `want` (the frame is parked or, for Abort, recorded).
    bool recv_expected(MsgType want, Frame & frame, std::string * err);

    ClusterWorkerControl & control_;
    WorkerHooksConfig      cfg_;
    Frame                  pending_;
    bool                   has_pending_ = false;
    bool                   aborted_ = false;
    std::string            abort_reason_;
};

// FNV-1a over a float buffer / token vector; shared by backend and tests.
uint64_t hash_floats(const float * data, size_t n);
uint64_t hash_tokens(const std::vector<int32_t> & tokens);

}  // namespace luce::cluster
