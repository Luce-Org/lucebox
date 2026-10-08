// cluster_decision_hooks.cpp - head / worker / local decision hooks.

#include "cluster/cluster_decision_hooks.h"
#include "cluster/cluster_control.h"

#include <cstdio>
#include <cstring>
#include <random>

namespace luce::cluster {

namespace {

int32_t greedy_argmax(const float * logits, int n_vocab) {
    if (!logits || n_vocab <= 0) return -1;
    int32_t best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

void set_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

}  // namespace

uint64_t hash_floats(const float * data, size_t n) {
    if (!data || n == 0) return fnv1a64(nullptr, 0);
    return fnv1a64(data, n * sizeof(float));
}

uint64_t hash_tokens(const std::vector<int32_t> & tokens) {
    if (tokens.empty()) return fnv1a64(nullptr, 0);
    return fnv1a64(tokens.data(), tokens.size() * sizeof(int32_t));
}

// ─── HeadHooks ─────────────────────────────────────────────────────────

HeadHooks::HeadHooks(ClusterHeadControl & control, const HeadHooksConfig & cfg)
    : control_(control), cfg_(cfg) {}

void HeadHooks::reset_counters() {
    Ds4ClusterHooks::reset_counters();
    first_mismatch_rank_ = -1;
    first_mismatch_step_ = 0;
}

bool HeadHooks::decide_next_token(uint64_t req, uint32_t step, const float * logits, int n_vocab,
                                  const common::SamplerCfg & cfg, bool /*is_eos_candidate_unused*/,
                                  int32_t & token, uint8_t & flags, std::string * err) {
    if (token < 0) {
        // Caller did not sample host-side. Only rank 0 owns an RNG.
        if (sampler_) {
            token = sampler_(logits, n_vocab, cfg);
        } else if (cfg.temp > 0.0f) {
            static thread_local std::mt19937_64 rng{cfg.seed != 0 ? cfg.seed : 0x5eedULL};
            std::vector<int32_t> no_history;
            token = common::sample_logits(logits, n_vocab, cfg, no_history, rng);
        } else {
            token = greedy_argmax(logits, n_vocab);
        }
        if (token < 0) {
            set_err(err, "head: sampling produced no token");
            return false;
        }
    }

    DecisionMsg msg;
    msg.request_id = req;
    msg.step = step;
    msg.token = token;
    msg.flags = flags;

    const uint64_t t0 = monotonic_us();
    const bool ok = control_.broadcast(make_frame(msg), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.decisions++;
    return true;
}

bool HeadHooks::decide_draft(uint64_t req, uint32_t step, int pos, std::vector<int32_t> & tokens,
                             std::string * err) {
    DraftMsg msg;
    msg.request_id = req;
    msg.step = step;
    msg.pos = pos;
    msg.tokens = tokens;
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.broadcast(make_frame(msg), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.drafts++;
    return true;
}

bool HeadHooks::decide_accept(uint64_t req, uint32_t step, int32_t & accept, int32_t & bonus,
                              uint8_t & flags, std::string * err) {
    AcceptMsg msg;
    msg.request_id = req;
    msg.step = step;
    msg.accept = accept;
    msg.bonus = bonus;
    msg.flags = flags;
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.broadcast(make_frame(msg), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.accepts++;
    return true;
}

bool HeadHooks::hash_probe(uint64_t req, uint32_t step, uint64_t hc_hash, uint64_t tok_hash,
                           std::string * err) {
    std::vector<Frame> frames;
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.gather(MsgType::HashProbe, cfg_.timeout_ms, frames, err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.hash_probes++;

    bool mismatch = false;
    for (size_t i = 0; i < frames.size(); i++) {
        HashProbeMsg probe;
        if (!parse_frame(frames[i], probe)) {
            set_err(err, "head: malformed HashProbe from rank " + std::to_string(i + 1));
            return false;
        }
        if (probe.request_id != req || probe.step != step) {
            set_err(err, "head: HashProbe from rank " + std::to_string(probe.rank) +
                         " is for req=" + std::to_string(probe.request_id) +
                         " step=" + std::to_string(probe.step) +
                         ", expected req=" + std::to_string(req) +
                         " step=" + std::to_string(step));
            return false;
        }
        const bool hc_diff  = probe.hc_state_hash != hc_hash;
        const bool tok_diff = probe.token_hash != tok_hash;
        if (hc_diff || tok_diff) {
            mismatch = true;
            counters_.hash_mismatches++;
            if (first_mismatch_rank_ < 0) {
                first_mismatch_rank_ = probe.rank;
                first_mismatch_step_ = step;
            }
            std::fprintf(stderr,
                "[cluster] HASH MISMATCH req=%llu step=%u rank=%d: hc %016llx vs head %016llx%s, "
                "tokens %016llx vs head %016llx%s%s\n",
                (unsigned long long) req, step, probe.rank,
                (unsigned long long) probe.hc_state_hash, (unsigned long long) hc_hash,
                hc_diff ? " (DIFF)" : "",
                (unsigned long long) probe.token_hash, (unsigned long long) tok_hash,
                tok_diff ? " (DIFF)" : "",
                probe.first_divergent_layer >= 0
                    ? (" first_divergent_layer=" + std::to_string(probe.first_divergent_layer)).c_str()
                    : "");
        } else if (cfg_.trace) {
            std::fprintf(stderr, "[cluster] hash ok req=%llu step=%u rank=%d hc=%016llx tok=%016llx\n",
                         (unsigned long long) req, step, probe.rank,
                         (unsigned long long) hc_hash, (unsigned long long) tok_hash);
        }
    }
    if (mismatch && cfg_.strict_hash) {
        set_err(err, "hash mismatch at step " + std::to_string(step) +
                     " (first divergent rank " + std::to_string(first_mismatch_rank_) + ")");
        return false;
    }
    return true;
}

bool HeadHooks::backend_op(const BackendOpMsg & op, std::string * err) {
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.broadcast(make_frame(op), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.backend_ops++;
    return true;
}

// ─── WorkerHooks ───────────────────────────────────────────────────────

WorkerHooks::WorkerHooks(ClusterWorkerControl & control, const WorkerHooksConfig & cfg)
    : control_(control), cfg_(cfg) {}

bool WorkerHooks::take_pending_frame(Frame & out) {
    if (!has_pending_) return false;
    out = std::move(pending_);
    pending_ = Frame{};
    has_pending_ = false;
    return true;
}

bool WorkerHooks::recv_expected(MsgType want, Frame & frame, std::string * err) {
    if (aborted_) {
        set_err(err, "worker: head aborted: " + abort_reason_);
        return false;
    }
    if (has_pending_) {
        // A frame parked by an earlier mismatch is still unconsumed; the
        // main loop must handle it before the backend asks for decisions.
        set_err(err, std::string("worker: unconsumed ") + msg_type_name(pending_.type) +
                     " frame while waiting for " + msg_type_name(want));
        return false;
    }
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.recv(frame, cfg_.timeout_ms, err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    if (frame.type == want) return true;

    if (frame.type == MsgType::Abort) {
        AbortMsg abort;
        aborted_ = true;
        abort_reason_ = parse_frame(frame, abort) ? abort.reason : std::string("(malformed Abort)");
        set_err(err, "worker: head aborted: " + abort_reason_);
        return false;
    }
    pending_ = std::move(frame);
    has_pending_ = true;
    set_err(err, std::string("worker: expected ") + msg_type_name(want) + " but received " +
                 msg_type_name(pending_.type));
    return false;
}

bool WorkerHooks::decide_next_token(uint64_t req, uint32_t step, const float * /*logits*/,
                                    int /*n_vocab*/, const common::SamplerCfg & /*cfg*/,
                                    bool /*is_eos_candidate_unused*/, int32_t & token,
                                    uint8_t & flags, std::string * err) {
    Frame frame;
    if (!recv_expected(MsgType::Decision, frame, err)) return false;
    DecisionMsg msg;
    if (!parse_frame(frame, msg)) {
        set_err(err, "worker: malformed Decision frame");
        return false;
    }
    if (msg.request_id != req || msg.step != step) {
        set_err(err, "worker: Decision for req=" + std::to_string(msg.request_id) +
                     " step=" + std::to_string(msg.step) + ", expected req=" +
                     std::to_string(req) + " step=" + std::to_string(step));
        return false;
    }
    token = msg.token;
    flags = msg.flags;
    counters_.decisions++;
    return true;
}

bool WorkerHooks::decide_draft(uint64_t req, uint32_t step, int pos, std::vector<int32_t> & tokens,
                               std::string * err) {
    Frame frame;
    if (!recv_expected(MsgType::Draft, frame, err)) return false;
    DraftMsg msg;
    if (!parse_frame(frame, msg)) {
        set_err(err, "worker: malformed Draft frame");
        return false;
    }
    if (msg.request_id != req || msg.step != step) {
        set_err(err, "worker: Draft for req=" + std::to_string(msg.request_id) +
                     " step=" + std::to_string(msg.step) + ", expected req=" +
                     std::to_string(req) + " step=" + std::to_string(step));
        return false;
    }
    if (msg.pos != pos) {
        set_err(err, "worker: Draft pos=" + std::to_string(msg.pos) + " but local pos=" +
                     std::to_string(pos) + " (ranks diverged)");
        return false;
    }
    tokens = std::move(msg.tokens);
    counters_.drafts++;
    return true;
}

bool WorkerHooks::decide_accept(uint64_t req, uint32_t step, int32_t & accept, int32_t & bonus,
                                uint8_t & flags, std::string * err) {
    Frame frame;
    if (!recv_expected(MsgType::Accept, frame, err)) return false;
    AcceptMsg msg;
    if (!parse_frame(frame, msg)) {
        set_err(err, "worker: malformed Accept frame");
        return false;
    }
    if (msg.request_id != req || msg.step != step) {
        set_err(err, "worker: Accept for req=" + std::to_string(msg.request_id) +
                     " step=" + std::to_string(msg.step) + ", expected req=" +
                     std::to_string(req) + " step=" + std::to_string(step));
        return false;
    }
    accept = msg.accept;
    bonus = msg.bonus;
    flags = msg.flags;
    counters_.accepts++;
    return true;
}

bool WorkerHooks::hash_probe(uint64_t req, uint32_t step, uint64_t hc_hash, uint64_t tok_hash,
                             std::string * err) {
    HashProbeMsg msg;
    msg.request_id = req;
    msg.step = step;
    msg.rank = cfg_.rank;
    msg.hc_state_hash = hc_hash;
    msg.token_hash = tok_hash;
    msg.first_divergent_layer = -1;  // per-layer attribution is a LUCE_CLUSTER_TRACE follow-up
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.send(make_frame(msg), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.hash_probes++;
    return true;
}

}  // namespace luce::cluster
