#include "qwen4exp_seq_engine.h"

#include "common/sampler.h"
#include "qwen4exp_graph.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

namespace luce::common {

namespace {
// The MTP slot's pending pairs beside other requests run through the draft layer once this many accumulate.
constexpr size_t kMtpPendingPairs = 64;

uint32_t pool_blocks(int max_ctx, size_t slots) {
    if (max_ctx <= 0 || slots == 0 ||
        slots > std::numeric_limits<uint32_t>::max()) return 0;
    const uint64_t blocks_per_slot =
        (uint64_t(max_ctx) + 255) / 256;
    const uint64_t blocks = blocks_per_slot * slots;
    return blocks <= std::numeric_limits<uint32_t>::max()
        ? (uint32_t)blocks : 0;
}
} // namespace

Qwen4ExpSeqEngine::Qwen4ExpSeqEngine(
        ggml_backend_t backend, const Qwen4ExpWeights & weights,
        std::vector<Qwen4ExpCache *> caches, int max_ctx, int prefill_granule,
        size_t prefix_allowance, int verify_width, AdaptiveSpecWidth * mtp_width,
        SpecWidthCostMemory * mtp_costs, ggml_backend_t snapshot_store, PromptLookupPolicy * lookup_policy)
    : backend_(backend), store_(snapshot_store ? snapshot_store : backend), weights_(weights),
      caches_(std::move(caches)),
      pool_(pool_blocks(max_ctx, caches_.size()),
            (uint32_t)caches_.size(), 256),
      slots_(pool_, max_ctx),
      prefill_granule_(std::max(prefill_granule, 1)),
      prefix_allowance_(prefix_allowance),
      checkpoints_((size_t)kPrefixCheckpoints),
      prefill_cuts_(caches_.size()),
      mtp_slot_(verify_width != 1 && !caches_.empty() && caches_[0] &&
                qwen4exp_verify_supported(*caches_[0]) ? 0 : -1),
      verify_width_(verify_width),
      mtp_(caches_.size()),
      mode_rates_((size_t)kQwen4ExpMaxBatchedSlots + 1),
      mtp_width_(mtp_width ? mtp_width : &own_mtp_width_),
      mtp_costs_(mtp_costs ? mtp_costs : &own_mtp_costs_),
      lookup_policy_(verify_width == 0 ? lookup_policy : nullptr) {}

Qwen4ExpSeqEngine::~Qwen4ExpSeqEngine() {
    ggml_backend_synchronize(backend_);
    for (Qwen4ExpSnapshot & checkpoint : checkpoints_) free_qwen4exp_snapshot(checkpoint);
    clear_qwen4exp_batched_decode_workspace(decode_workspace_);
    clear_qwen4exp_batched_decode_workspace(verify_workspace_);
    for (auto & ws : draft_workspace_) clear_qwen4exp_batched_decode_workspace(ws);
}

bool Qwen4ExpSeqEngine::mtp_capable(int slot) const {
    return verify_width_ != 1 && slot >= 0 && slot < (int)caches_.size() && caches_[(size_t)slot] &&
        qwen4exp_verify_supported(*caches_[(size_t)slot]);
}

bool Qwen4ExpSeqEngine::token_is_eos(int32_t token) const {
    return token == weights_.eos_id || token == weights_.eos_chat_id;
}

SeqEngine::AdmitResult Qwen4ExpSeqEngine::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    return admit_cold(request_id, prompt, sampler);
}

SeqEngine::AdmitResult Qwen4ExpSeqEngine::admit_cold(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    for (int32_t token : prompt) {
        if (token < 0 || token >= weights_.n_vocab) {
            AdmitResult invalid;
            invalid.status = AdmitResult::Status::failed;
            invalid.error = "qwen4exp prompt contains an invalid token id";
            return invalid;
        }
    }
    AdmitResult result = slots_.admit(request_id, prompt, sampler);
    if (result.status != AdmitResult::Status::admitted) return result;
    if (!backend_ || result.slot < 0 ||
        result.slot >= (int)caches_.size() || !caches_[(size_t)result.slot] ||
        caches_[(size_t)result.slot]->max_ctx < max_context()) {
        slots_.retire(result.slot);
        result.status = AdmitResult::Status::failed;
        result.error = "invalid qwen4exp full-cache slot";
        return result;
    }
    if (pool_.reserve_capacity(slots_.slot(result.slot).handle,
                               (uint32_t)max_context()) != PagedKvStatus::Ok) {
        slots_.retire(result.slot);
        result.status = AdmitResult::Status::busy;
        result.error = "qwen4exp full-context reservation unavailable";
        return result;
    }
    reset_qwen4exp_state(backend_, *caches_[(size_t)result.slot]);
    ggml_backend_synchronize(backend_);
    prefill_cuts_[(size_t)result.slot].clear();
    if (mtp_capable(result.slot)) {
        MtpState & m = mtp_[(size_t)result.slot];
        m = MtpState{};
        m.live = true;
        m.context = (int)prompt.size();
        if (result.slot == mtp_slot_ && verify_width_ == 0) {   // as the single-slot loop starts a request
            mtp_costs_->load(*mtp_width_, m.context);
            mtp_width_->carry_acceptance(QWEN4EXP_MTP_CARRIED_TRIALS);
        }
    }
    return result;
}

size_t Qwen4ExpSeqEngine::estimate_prefix_store_bytes(int tokens) const {
    if (caches_.empty() || !caches_[0] || tokens <= 0) return 0;
    size_t host = 0;
    const size_t copies = qwen4exp_snapshot_bytes(store_, *caches_[0], tokens, &host);
    if (!copies) return 0;
    return copies + host + (size_t(tokens) + std::max(0, weights_.ple_ngram_size - 1)) * sizeof(int32_t);
}

int Qwen4ExpSeqEngine::checkpoint_index(PrefixStoreRef checkpoint) const {
    if (!checkpoint.valid() || checkpoint.id > (uint64_t)kPrefixCheckpoints) return -1;
    return (int)checkpoint.id - 1;
}

void Qwen4ExpSeqEngine::discard_prefix_store(PrefixStoreRef checkpoint) {
    const int index = checkpoint_index(checkpoint);
    if (index >= 0 && checkpoints_[(size_t)index].buf &&
        checkpoints_[(size_t)index].cur_pos == checkpoint.tokens)
        free_qwen4exp_snapshot(checkpoints_[(size_t)index]);
}

// A checkpoint restores only into a cold slot and only when its tokens are the
// prompt's own prefix; any failure leaves the slot cold.
bool Qwen4ExpSeqEngine::restore_prefix(int slot, const std::vector<int32_t> & prompt,
                                       PrefixStoreRef checkpoint) {
    const int index = checkpoint_index(checkpoint);
    if (index < 0 || checkpoint.tokens >= (int)prompt.size()) return false;
    const Qwen4ExpSnapshot & s = checkpoints_[(size_t)index];
    if (!s.buf || s.cur_pos != checkpoint.tokens ||
        s.tokens.size() != (size_t)checkpoint.tokens ||
        !std::equal(s.tokens.begin(), s.tokens.end(), prompt.begin())) return false;
    Qwen4ExpCache & cache = *caches_[(size_t)slot];
    if (!restore_qwen4exp_snapshot(backend_, s, cache)) return false;
    if (!slots_.seed_restored_prefix(slot, checkpoint.tokens).ok) {
        reset_qwen4exp_state(backend_, cache);
        ggml_backend_synchronize(backend_);
        return false;
    }
    return true;
}

SeqEngine::AdmitResult Qwen4ExpSeqEngine::admit_with_prefix(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler, const PrefixStorePlan & plan) {
    AdmitResult result = admit_cold(request_id, prompt, sampler);
    if (result.status != AdmitResult::Status::admitted) return result;
    const int slot = result.slot;
    slots_.slot(slot).pending_capture = {};
    prefill_cuts_[(size_t)slot] = plan.restore_points;
    std::sort(prefill_cuts_[(size_t)slot].begin(), prefill_cuts_[(size_t)slot].end());
    if (plan.restore.valid()) {
        const auto started = std::chrono::steady_clock::now();
        const bool restored = restore_prefix(slot, prompt, plan.restore);
        const uint64_t elapsed_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (restored) {
            result.prefix_store.restored = plan.restore;
            std::fprintf(stderr, "[qwen4exp-seq] restored checkpoint=%llu slot=%d tokens=%d time_ms=%.1f\n",
                (unsigned long long)plan.restore.id, slot, plan.restore.tokens, (double)elapsed_us / 1000.0);
        } else {
            discard_prefix_store(plan.restore);
            result.prefix_store.invalidated = plan.restore;
        }
        result.prefix_store.restore_attempted = true;
        result.prefix_store.restore_elapsed_us = elapsed_us;
    }
    const SeqSlot & seq = slots_.slot(slot);
    if (!result.prefix_store.invalidated.valid() && plan.capture.valid() &&
        checkpoint_index(plan.capture.checkpoint) >= 0 &&
        plan.capture.checkpoint.tokens > seq.cur_pos &&
        plan.capture.checkpoint.tokens <= seq.prompt_len) {
        slots_.slot(slot).pending_capture = plan.capture;
        result.prefix_store.capture = plan.capture;
    }
    return result;
}

// A prompt chunk of the MTP slot, as the single-slot loop runs it: chunks
// fill the draft layer's K/V inline and keep the last trunk row pending; a
// one-row chunk catches its pair up through the draft layer instead. The
// pending row before the chunk is the cache's (mtp_prev_hidden at
// mtp_prev_pos), so a cache restored from a prefix snapshot keeps drafting; a
// cache without that row prefills plainly and its request decodes undrafted.
bool Qwen4ExpSeqEngine::mtp_prefill(int slot, const int32_t * tokens, int n, int pos0,
                                    std::vector<float> & logits) {
    const size_t hd = (size_t)weights_.n_embd * weights_.n_hc;
    Qwen4ExpCache & cache = *caches_[(size_t)slot];
    MtpState & m = mtp_[(size_t)slot];
    Qwen4ExpMtpPending & pending = m.pending;
    pending.tok.clear();
    if (pos0 > 0 && cache.mtp_prev_pos != pos0 - 1) {
        m.live = false;
        pending.h.clear();
        return qwen4exp_forward(backend_, weights_, cache, tokens, n, pos0, logits).ok;
    }
    pending.h.resize(pos0 > 0 ? hd : 0);
    if (pos0 > 0) ggml_backend_tensor_get(cache.mtp_prev_hidden, pending.h.data(), 0, hd * sizeof(float));
    pending.pos = std::max(0, pos0 - 1);
    std::vector<float> hidden;
    if (n > 1) {
        if (!qwen4exp_forward(backend_, weights_, cache, tokens, n, pos0, logits, &hidden, false, true).ok)
            return false;
        pending.h.swap(hidden);
        pending.pos = pos0 + n - 1;
        return true;
    }
    if (!qwen4exp_forward(backend_, weights_, cache, tokens, 1, pos0, logits, &hidden).ok) return false;
    if (pos0 > 0) pending.tok.assign(tokens, tokens + 1);
    pending.h.insert(pending.h.end(), hidden.begin(), hidden.end());
    if (!qwen4exp_mtp_catch_up(backend_, weights_, cache, pending, 0) || pending.h.size() != hd) return false;
    ggml_backend_tensor_set(cache.mtp_prev_hidden, pending.h.data(), 0, hd * sizeof(float));
    cache.mtp_prev_pos = pos0;
    return true;
}

bool Qwen4ExpSeqEngine::mtp_eligible(const StepPlan & plan) const {
    return mtp_slot_ >= 0 && mtp_[(size_t)mtp_slot_].live && plan.prefills.empty() &&
        plan.decode.size() == 1 && plan.decode[0].slot == mtp_slot_ &&
        plan.decode[0].allow_speculation;
}

// The MTP slot alone takes the single-slot loop's step (qwen4exp_mtp_step). The
// accepted drafts are durable children of the output; the last emitted token
// stays pending for the scheduler, as in one-token decode.
SeqEngine::StepResult Qwen4ExpSeqEngine::mtp_step(const StepInput & input) {
    StepResult result;
    const int id = input.slot;
    Qwen4ExpCache & cache = *caches_[(size_t)id];
    SeqSlot & slot = slots_.slot(id);
    MtpState & m = mtp_[(size_t)id];
    const int pos = slot.cur_pos;
    const auto t0 = std::chrono::steady_clock::now();
    auto verify_builds = [&cache] {
        uint64_t n = 0;
        for (const auto & ws : cache.verify_workspace) n += ws.builds;
        return n;
    };
    const uint64_t builds0 = verify_builds();

    // Rows are sampled as one-token decode would, each after its own fed token, up to EOS, and read only when
    // sampled. Greedy rows take the device's pick (token already holds it), the lowest id of the maximum as
    // max_element finds it; a top_k draw first tries the row's top candidates (sample_logits_top). Kill switch:
    // LUCE_GPU_ARGMAX=0 reads every row back.
    const Qwen4ExpRows rows = qwen4exp_rows_for(slot.sampler);
    const int top_n = rows == Qwen4ExpRows::lazy ? sample_logits_top_n(slot.sampler, weights_.n_vocab) : 0;
    const size_t history0 = slot.sample_history.size();
    auto sample_row = [&](int32_t fed, const Qwen4ExpRow & row, int32_t & token) {
        return sample_verify_row(slot, fed, row, token);
    };
    // Prompt lookup drafts, when the policy expects them to commit more tokens per ms than the MTP's this round. The
    // index catches up with the tokens the slot committed since its last round (also in multi-slot steps), then
    // takes the token fed now.
    const int max_k = max_context() - pos - 1;
    std::array<int32_t, QWEN4EXP_MTP_MAX_DRAFT> look{};
    int look_match = 0;
    PromptLookupPolicy::Pick pick;
    if (lookup_policy_) {
        if (m.lookup_len > history0) {   // cannot happen while the request lives; start over to be safe
            m.lookup.reset();
            m.lookup_len = 0;
        }
        m.lookup.append(slot.sample_history.data() + m.lookup_len, history0 - m.lookup_len);
        m.lookup.append(&input.token, 1);
        m.lookup_len = history0 + 1;   // the fed token is committed with this round's tokens
        const int look_k = max_k > 0 ? m.lookup.propose(std::min(max_k, cache.mtp_draft), look.data(), look_match)
                                     : 0;
        if (look_k > 0) {
            pick = lookup_policy_->choose(mtp_width_->next_width_cost_aware({}, cache.mtp_draft + 1), look_k,
                                          look_match);
        }
    }
    m.pending.tok.push_back(input.token);
    std::vector<float> logits;
    Qwen4ExpMtpStep step;
    const bool ok = qwen4exp_mtp_step(backend_, weights_, cache, pos, input.token, max_k,
                                      verify_width_ == 0 ? mtp_width_ : nullptr, &m.pending, sample_row,
                                      logits, step, {}, pick.lookup ? look.data() : nullptr,
                                      pick.lookup ? pick.width - 1 : 0, rows, top_n);
    slot.sample_history.resize(history0);
    if (!ok) {
        result.error = step.error;
        return result;
    }
    if (lookup_policy_ && step.k > 0) {
        lookup_policy_->observe(step.lookup, step.k + 1, step.decision.n_accepted, look_match, step.round_ms);
        if (step.lookup) {
            ++m.lookup_rounds;
            m.lookup_drafts += step.k;
            m.lookup_accepted += step.decision.n_accepted;
        }
    }
    m.pending.tok.pop_back();   // the scheduler feeds it back as the next step's token
    ggml_backend_synchronize(backend_);
    const Qwen4ExpMtpAcceptance & decision = step.decision;
    const int retained = decision.n_emitted;
    std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> fed{};
    fed[0] = input.token;
    std::copy(decision.emitted.begin(), decision.emitted.begin() + retained - 1, fed.begin() + 1);
    cache.cur_pos = pos + retained;
    const auto appended = slots_.append_tokens(id, fed.data(), retained);
    if (!appended.ok || appended.position != pos) {
        result.error = "qwen4exp MTP reservation failed";
        return result;
    }
    slots_.commit_step(id);

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    m.draft_s += step.draft_s;
    m.verify_s += step.verify_s;
    m.verify_builds += verify_builds() - builds0;
    ++m.steps;
    m.drafts += step.k;
    m.accepted += decision.n_accepted;
    m.tokens += retained;
    m.decode_s += ms / 1e3;

    DecodeOutput output;
    output.slot = id;
    output.token = decision.emitted[(size_t)retained - 1];
    output.committed_tokens.assign(decision.emitted.begin(), decision.emitted.begin() + retained - 1);
    output.spec_offered = step.k;
    output.spec_accepted = decision.n_accepted;
    result.decode.push_back(std::move(output));
    return result;
}

bool Qwen4ExpSeqEngine::sample_verify_row(SeqSlot & slot, int32_t fed, const Qwen4ExpRow & row, int32_t & token) {
    slot.sample_history.push_back(fed);
    if (row.picked) return !token_is_eos(token);
    const int t = row.top_n > 0 ? sample_logits_top(row.top_vals, row.top_ids, row.top_n, weights_.n_vocab,
                                                    slot.sampler, slot.sample_history, slot.rng) : -1;
    if (t >= 0) {
        token = t;
    } else {
        const float * logits = row.logits();
        token = slot.sampler.needs_logit_processing()
            ? sample_logits(logits, weights_.n_vocab, slot.sampler, slot.sample_history, slot.rng)
            : (int32_t)(std::max_element(logits, logits + weights_.n_vocab) - logits);
    }
    return !token_is_eos(token);
}

namespace {
// A mode's tokens and seconds decay by this per step it runs (~10 steps of memory): lower follows a change of load
// sooner and lets a few slow steps swing the choice.
constexpr double kModeDecay = 0.9;
// The mode not chosen runs every kExploreSteps steps so its rate stays current: lower measures it more often and
// spends more steps in the slower mode.
constexpr long long kExploreSteps = 32;
// Batched verify steps before the plain mode's first measurement (verify usually pays, so it is measured first).
constexpr long long kVerifyFirstSteps = 4;
}  // namespace

bool Qwen4ExpSeqEngine::verify_pays(int n) const {
    if (n < 0 || n >= (int)mode_rates_.size()) return false;
    const ModeRates & r = mode_rates_[(size_t)n];
    if (r.verify_steps < kVerifyFirstSteps) return true;
    if (r.plain_steps == 0) return false;
    const bool better = r.verify_tokens * r.plain_s >= r.plain_tokens * r.verify_s;
    return better != (r.steps % kExploreSteps == kExploreSteps - 1);
}

void Qwen4ExpSeqEngine::record_mode(int n, bool verify, double tokens, double seconds) {
    if (n < 0 || n >= (int)mode_rates_.size() || seconds <= 0.0) return;
    ModeRates & r = mode_rates_[(size_t)n];
    double & t = verify ? r.verify_tokens : r.plain_tokens;
    double & s = verify ? r.verify_s : r.plain_s;
    t = t * kModeDecay + tokens;
    s = s * kModeDecay + seconds;
    ++(verify ? r.verify_steps : r.plain_steps);
    ++r.steps;
}

int Qwen4ExpSeqEngine::batched_verify_rows(const StepPlan & plan) const {
    if (decode_mode_ == DecodeMode::plain) return 0;
    const int rows = batched_verify_width(plan);
    return rows > 0 && (decode_mode_ == DecodeMode::verify || verify_pays((int)plan.decode.size())) ? rows : 0;
}

int Qwen4ExpSeqEngine::batched_verify_width(const StepPlan & plan) const {
    if (!plan.prefills.empty() || plan.decode.empty() || mtp_eligible(plan)) return 0;
    int rows = kQwen4ExpBatchedVerifyRows;
    std::vector<Qwen4ExpCache *> caches;
    std::vector<int32_t> positions;
    for (const StepInput & input : plan.decode) {
        const SeqSlot & slot = slots_.slot(input.slot);
        if (!mtp_[(size_t)input.slot].live || !input.allow_speculation) return 0;
        // Each cache's rollback rows bound its verify (a fixed --verify-width of 2 or 3 leaves 2 or 3).
        rows = std::min({rows, max_context() - slot.cur_pos, caches_[(size_t)input.slot]->mtp_draft + 1});
        caches.push_back(caches_[(size_t)input.slot]);
        positions.push_back(slot.cur_pos);
    }
    if (rows < 2 || !qwen4exp_verify_batched_supported(backend_, weights_, caches.data(), positions.data(),
                                                       (int)caches.size(), rows)) return 0;
    return rows;
}

// Every decoding slot drafts rows - 1 tokens from its pending pairs and the graph verifies all of them at once. Each
// slot then keeps its accepted prefix as its solo verify would (the accepted drafts and the verified token, the
// trunk rows as pending pairs, the rollback to the last kept row).
SeqEngine::StepResult Qwen4ExpSeqEngine::batched_verify_step(const StepPlan & plan, int rows) {
    StepResult result;
    const auto t0 = std::chrono::steady_clock::now();
    const int n = (int)plan.decode.size();
    const int k = rows - 1;
    const size_t hd = (size_t)weights_.n_embd * weights_.n_hc;
    std::vector<Qwen4ExpCache *> caches((size_t)n);
    std::vector<int32_t> tokens((size_t)n * rows), positions((size_t)n);
    std::vector<std::vector<int32_t>> drafts;
    std::vector<const int32_t *> draft_tok((size_t)n);
    std::vector<const float *> draft_h((size_t)n);
    std::vector<int> draft_n((size_t)n), draft_pos((size_t)n);
    for (int s = 0; s < n; ++s) {
        const StepInput & input = plan.decode[(size_t)s];
        Qwen4ExpCache & cache = *caches_[(size_t)input.slot];
        Qwen4ExpMtpPending & pending = mtp_[(size_t)input.slot].pending;
        caches[(size_t)s] = &cache;
        positions[(size_t)s] = slots_.slot(input.slot).cur_pos;
        pending.tok.push_back(input.token);
        // After a stretch of plain decode, fold all but the last pair first: a draft catches up at most a verify's.
        if (pending.tok.size() > (size_t)cache.mtp_draft + 1 &&
            !qwen4exp_mtp_catch_up(backend_, weights_, cache, pending, 1)) {
            result.error = "qwen4exp MTP catch-up failed";
            return result;
        }
        draft_tok[(size_t)s] = pending.tok.data();
        draft_h[(size_t)s] = pending.h.data();
        draft_n[(size_t)s] = (int)pending.tok.size();
        draft_pos[(size_t)s] = pending.pos;
    }
    // Every slot's drafts: each slot's first rank, then each later rank once for all slots.
    if (draft_workspace_.size() < (size_t)k) draft_workspace_.resize((size_t)k);
    if (!qwen4exp_mtp_draft_slots(backend_, weights_, caches.data(), draft_tok.data(), draft_h.data(), draft_n.data(),
                                  draft_pos.data(), n, k, draft_workspace_, drafts)) {
        result.error = "qwen4exp MTP draft failed";
        return result;
    }
    for (int s = 0; s < n; ++s) {
        tokens[(size_t)s * rows] = plan.decode[(size_t)s].token;
        std::copy(drafts[(size_t)s].begin(), drafts[(size_t)s].end(), tokens.begin() + (std::ptrdiff_t)s * rows + 1);
    }
    const auto t_draft = std::chrono::steady_clock::now();
    // Rows reach each slot's sampler as its solo verify hands them over (Qwen4ExpRows): the device's picks for an
    // argmax sampler, else the step's largest candidate count first and the whole row on demand. One count serves
    // every slot (the graph takes one top-k): sample_logits_top draws exactly from any count at least its sampler's
    // sample_logits_top_n, and reads the whole row when the candidates do not settle the draw.
    std::vector<Qwen4ExpRows> how((size_t)n);
    Qwen4ExpRowOutputs out;
    for (int s = 0; s < n; ++s) {
        const SamplerCfg & sampler = slots_.slot(plan.decode[(size_t)s].slot).sampler;
        how[(size_t)s] = qwen4exp_rows_for(sampler);
        if (how[(size_t)s] == Qwen4ExpRows::lazy)
            out.top_n = std::max(out.top_n, sample_logits_top_n(sampler, weights_.n_vocab));
    }
    std::vector<float> hidden;
    if (!qwen4exp_verify_batched(backend_, weights_, caches.data(), tokens.data(), positions.data(), n, rows,
                                 verify_workspace_, out, hidden).ok ||
        out.greedy.size() != (size_t)n * rows || hidden.size() != hd * n * rows) {
        result.error = "qwen4exp batched verify failed";
        return result;
    }
    std::vector<float> row_logits;
    int row_read = -1;   // the row row_logits holds
    const auto t_verify = std::chrono::steady_clock::now();
    long long emitted_total = 0;
    for (int s = 0; s < n; ++s) {
        const StepInput & input = plan.decode[(size_t)s];
        const int id = input.slot;
        const int pos = positions[(size_t)s];
        Qwen4ExpCache & cache = *caches[(size_t)s];
        MtpState & m = mtp_[(size_t)id];
        SeqSlot & slot = slots_.slot(id);
        const size_t history0 = slot.sample_history.size();
        std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> samples{};
        Qwen4ExpMtpAcceptance decision;
        bool read_ok = true;
        for (int i = 0; i < rows; ++i) {
            const int r = s * rows + i;
            Qwen4ExpRow row;
            if (how[(size_t)s] == Qwen4ExpRows::greedy) {
                row.picked = true;
                samples[(size_t)i] = out.greedy[(size_t)r];
            } else {
                row.read = [&, r] {
                    if (row_read != r) {
                        read_ok = read_ok && qwen4exp_verify_batched_row(verify_workspace_, weights_, r, row_logits);
                        row_read = r;
                    }
                    return (const float *)row_logits.data();
                };
                if (how[(size_t)s] == Qwen4ExpRows::lazy && out.top_n > 0) {
                    row.top_n = out.top_n;
                    row.top_vals = out.top_vals.data() + (size_t)r * out.top_n;
                    row.top_ids = out.top_ids.data() + (size_t)r * out.top_n;
                }
            }
            const bool more = sample_verify_row(slot, tokens[(size_t)r], row, samples[(size_t)i]);
            decision = qwen4exp_mtp_accept(drafts[(size_t)s].data(), k, samples.data(), i + 1);
            if (!more || decision.n_accepted != i + 1) break;
        }
        slot.sample_history.resize(history0);
        if (!read_ok) {
            result.error = "qwen4exp batched verify row read failed";
            return result;
        }
        const int retained = decision.n_emitted;
        // The trunk rows of the kept tokens become the pending pairs, as after a solo verify; the last emitted token
        // is the scheduler's to feed back.
        const float * h = hidden.data() + (size_t)s * rows * hd;
        m.pending.h.assign(h, h + (size_t)retained * hd);
        m.pending.tok.assign(decision.emitted.begin(), decision.emitted.begin() + retained);
        m.pending.pos = pos;
        m.pending.tok.pop_back();
        if (!qwen4exp_verify_rollback(backend_, weights_, cache, pos, retained)) {
            result.error = "qwen4exp verify rollback failed";
            return result;
        }
        std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> fed{};
        fed[0] = input.token;
        std::copy(decision.emitted.begin(), decision.emitted.begin() + retained - 1, fed.begin() + 1);
        const auto appended = slots_.append_tokens(id, fed.data(), retained);
        if (!appended.ok || appended.position != pos) {
            result.error = "qwen4exp MTP reservation failed";
            return result;
        }
        slots_.commit_step(id);
        ++m.steps;
        m.drafts += k;
        m.accepted += decision.n_accepted;
        m.tokens += retained;
        emitted_total += retained;
        DecodeOutput output;
        output.slot = id;
        output.token = decision.emitted[(size_t)retained - 1];
        output.committed_tokens.assign(decision.emitted.begin(), decision.emitted.begin() + retained - 1);
        output.spec_offered = k;
        output.spec_accepted = decision.n_accepted;
        result.decode.push_back(std::move(output));
    }
    ggml_backend_synchronize(backend_);
    const auto t1 = std::chrono::steady_clock::now();
    const double step_s = std::chrono::duration<double>(t1 - t0).count();
    for (const StepInput & input : plan.decode) mtp_[(size_t)input.slot].decode_s += step_s;
    ++batched_.steps;
    batched_.slots += n;
    batched_.tokens += emitted_total;
    batched_.step_s += step_s;
    record_mode(n, true, (double)emitted_total, step_s);
    batched_.draft_s += std::chrono::duration<double>(t_draft - t0).count();
    batched_.verify_s += std::chrono::duration<double>(t_verify - t_draft).count();
    return result;
}

PrefixStoreEvent Qwen4ExpSeqEngine::capture_prefix(int slot, PrefixCaptureTicket ticket) {
    PrefixStoreEvent event;
    event.ticket = ticket;
    event.status = PrefixStoreEvent::Status::failed;
    const int index = checkpoint_index(ticket.checkpoint);
    if (!ticket.valid() || index < 0 || !slots_.is_prefilling(slot) ||
        slots_.slot(slot).cur_pos != ticket.checkpoint.tokens ||
        caches_[(size_t)slot]->cur_pos != ticket.checkpoint.tokens) {
        event.error = "invalid prefix capture boundary";
        return event;
    }
    const auto started = std::chrono::steady_clock::now();
    auto elapsed_us = [&started] {
        return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    };
    // The capture replaces checkpoint `index`; every other resident one counts against the allowance.
    size_t resident = 0;
    for (size_t i = 0; i < checkpoints_.size(); ++i)
        if (i != (size_t)index && checkpoints_[i].buf)
            resident += ggml_backend_buffer_get_size(checkpoints_[i].buf);
    const size_t need = estimate_prefix_store_bytes(ticket.checkpoint.tokens);
    if (!need || resident > prefix_allowance_ || need > prefix_allowance_ - resident) {
        event.elapsed_us = elapsed_us();
        event.error = "qwen4exp prefix checkpoint allowance exhausted";
        return event;
    }
    Qwen4ExpSnapshot & s = checkpoints_[(size_t)index];
    if (!save_qwen4exp_snapshot(backend_, store_, *caches_[(size_t)slot], s)) {
        free_qwen4exp_snapshot(s);
        event.elapsed_us = elapsed_us();
        event.error = "qwen4exp prefix capture failed";
        return event;
    }
    const std::vector<int32_t> & history = slots_.slot(slot).sample_history;
    s.tokens.assign(history.begin(), history.begin() + ticket.checkpoint.tokens);
    event.status = PrefixStoreEvent::Status::saved;
    event.bytes = ggml_backend_buffer_get_size(s.buf);
    event.elapsed_us = elapsed_us();
    std::fprintf(stderr, "[qwen4exp-seq] saved checkpoint=%llu slot=%d tokens=%d bytes=%zu time_ms=%.1f\n",
        (unsigned long long)ticket.checkpoint.id, slot, ticket.checkpoint.tokens, event.bytes,
        (double)event.elapsed_us / 1000.0);
    return event;
}

int Qwen4ExpSeqEngine::prefill_segment(int slot, int max_tokens) const {
    const SeqSlot & s = slots_.slot(slot);
    int end = std::min({s.prompt_len, s.cur_pos + max_tokens, s.cur_pos + prefill_granule_});
    for (int cut : prefill_cuts_[(size_t)slot]) {
        if (cut > s.cur_pos) {
            end = std::min(end, cut);
            break;
        }
    }
    const PrefixCaptureTicket & capture = s.pending_capture;
    if (capture.valid() && s.cur_pos < capture.checkpoint.tokens)
        end = std::min(end, capture.checkpoint.tokens);
    return end - s.cur_pos;
}

// One granule per prompt forward, whatever the step carries. While slots
// decode, one granule in total per step bounds the hold on the live streams,
// and beside a prompt of more than 8192 new tokens (a document, not an agent's
// next step) they then decode alone for as long as the granule took: a granule
// pays the whole expert read, so a shorter one would not let them run faster,
// and a long prompt would otherwise give them one token per granule.
StepPlanLimits Qwen4ExpSeqEngine::step_plan_limits(int decode_rows) const {
    const int prefill_slots = slot_count() - std::clamp(decode_rows, 0, slot_count());
    if (decode_rows > 0) {
        const int sequences = std::min(prefill_slots, 1);
        return {sequences, prefill_granule_, sequences * prefill_granule_, prefill_granule_,
                /*prefill_time_share=*/0.5, /*time_share_prompt_tokens=*/8192};
    }
    return {prefill_slots, prefill_granule_, prefill_slots * prefill_granule_, prefill_granule_};
}

bool Qwen4ExpSeqEngine::reserve_decode(const StepPlan & plan) {
    if ((int)plan.decode.size() != slots_.decoding_count()) return false;
    std::vector<int> growth((size_t)slot_count(), 0);
    std::vector<uint8_t> assigned((size_t)slot_count(), 0);
    for (const StepInput & input : plan.decode) {
        if (input.slot < 0 || input.slot >= slot_count() ||
            input.token < 0 || input.token >= weights_.n_vocab ||
            growth[(size_t)input.slot] != 0 ||
            !slots_.slot(input.slot).decoding()) return false;
        growth[(size_t)input.slot] = 1;
        assigned[(size_t)input.slot] = 1;
    }
    if (mtp_eligible(plan)) {
        const int slot = plan.decode[0].slot;
        growth[(size_t)slot] = std::max(1, std::min(caches_[(size_t)slot]->mtp_draft + 1,
                                                    max_context() - slots_.slot(slot).cur_pos));
    } else if (const int rows = batched_verify_rows(plan)) {
        for (const StepInput & input : plan.decode) growth[(size_t)input.slot] = rows;
    }
    const StepPlanLimits limits = step_plan_limits((int)plan.decode.size());
    if (plan.prefills.size() > (size_t)limits.max_prefill_sequences) return false;
    int prefill_total = 0;
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= slot_count() ||
            slice.max_tokens < 1 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            (prefill_total += slice.max_tokens) > limits.max_prefill_tokens_total ||
            assigned[(size_t)slice.slot] ||
            !slots_.is_prefilling(slice.slot)) return false;
        assigned[(size_t)slice.slot] = 1;
    }
    return slots_.reserve_decode(growth);
}

SeqEngine::StepResult Qwen4ExpSeqEngine::step(const StepPlan & plan) {
    StepResult result;
    const int n = slot_count();
    auto fail = [&result](const char * message) -> StepResult {
        result.decode.clear();
        result.prefills.clear();
        result.error = message;
        return std::move(result);
    };

    if (!backend_ ||
        (int)plan.decode.size() != slots_.decoding_count())
        return fail("qwen4exp decode plan does not cover live slots");

    std::vector<uint8_t> seen((size_t)n, 0);
    for (const StepInput & input : plan.decode) {
        if (input.slot < 0 || input.slot >= n || input.token < 0 ||
            input.token >= weights_.n_vocab || seen[(size_t)input.slot] ||
            !slots_.slot(input.slot).decoding() ||
            !caches_[(size_t)input.slot] ||
            caches_[(size_t)input.slot]->cur_pos !=
                slots_.slot(input.slot).cur_pos ||
            slots_.slot(input.slot).cur_pos >= max_context())
            return fail("invalid or duplicate qwen4exp decode row");
        seen[(size_t)input.slot] = 1;
    }
    const StepPlanLimits limits = step_plan_limits((int)plan.decode.size());
    if (plan.prefills.size() > (size_t)limits.max_prefill_sequences)
        return fail("qwen4exp prefill plan exceeds available slots");
    for (const PrefillSlice & slice : plan.prefills) {
        const int remaining = slice.slot >= 0 && slice.slot < n &&
            slots_.is_prefilling(slice.slot)
            ? slots_.slot(slice.slot).prompt_len - slots_.slot(slice.slot).cur_pos
            : 0;
        if (slice.slot < 0 || slice.slot >= n || slice.max_tokens < 1 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            remaining <= 0 ||
            seen[(size_t)slice.slot] || !slots_.is_prefilling(slice.slot) ||
            !caches_[(size_t)slice.slot] ||
            caches_[(size_t)slice.slot]->cur_pos !=
                slots_.slot(slice.slot).cur_pos)
            return fail("invalid qwen4exp prefill slice");
        seen[(size_t)slice.slot] = 1;
    }
    if (plan.decode.empty() && plan.prefills.empty()) return result;
    if (mtp_eligible(plan)) return mtp_step(plan.decode[0]);
    if (const int rows = batched_verify_rows(plan)) return batched_verify_step(plan, rows);
    // A step that could have verified drafts: its time and tokens rate the plain mode (verify_pays), once it succeeded.
    const bool rate_plain = batched_verify_width(plan) > 0;
    const auto plain_t0 = std::chrono::steady_clock::now();

    struct PendingPrefill {
        int slot;
        bool complete;
        size_t segment;
    };
    std::vector<PendingPrefill> pending_prefills;
    std::vector<std::vector<int32_t>> segment_tokens;
    std::vector<Qwen4ExpForwardSegment> forward_segments;
    std::vector<size_t> decode_rows;
    std::vector<int> decode_positions;
    segment_tokens.reserve(plan.decode.size() + plan.prefills.size());
    forward_segments.reserve(plan.decode.size() + plan.prefills.size());
    decode_rows.reserve(plan.decode.size());
    decode_positions.reserve(plan.decode.size());

    for (const StepInput & input : plan.decode) {
        const auto appended = slots_.append_token(input.slot, input.token);
        if (!appended.ok) return fail("qwen4exp decode reservation failed");
        segment_tokens.emplace_back(1, input.token);
        decode_positions.push_back(appended.position);
        decode_rows.push_back(forward_segments.size());
        forward_segments.push_back({
            caches_[(size_t)input.slot], segment_tokens.back().data(),
            1, appended.position});
    }
    for (const PrefillSlice & slice : plan.prefills) {
        const SeqSlot & before = slots_.slot(slice.slot);
        const int count = prefill_segment(slice.slot, slice.max_tokens);
        if (count <= 0) return fail("qwen4exp prefill made no progress");
        const int pos = before.cur_pos;
        segment_tokens.emplace_back(
            before.sample_history.begin() + pos,
            before.sample_history.begin() + pos + count);
        const SeqSlotManager::PrefillChunk appended =
            slots_.append_prefill(slice.slot, count);
        if (!appended.ok || appended.rows.size() != (size_t) count)
            return fail("qwen4exp prefill reservation failed");
        const size_t segment_index = forward_segments.size();
        forward_segments.push_back({
            caches_[(size_t) slice.slot], segment_tokens.back().data(),
            count, pos});
        const bool complete = slots_.slot(slice.slot).cur_pos ==
                              slots_.slot(slice.slot).prompt_len;
        pending_prefills.push_back({slice.slot, complete, segment_index});
    }

    std::vector<std::vector<float>> logits;
    std::vector<int32_t> greedy;   // the batched decode's picks, when its greedy rows stayed on the device
    if (!plan.decode.empty()) {
        std::vector<int32_t> tokens;
        std::vector<int32_t> positions;
        std::vector<Qwen4ExpCache *> caches;
        tokens.reserve(plan.decode.size());
        positions.reserve(plan.decode.size());
        caches.reserve(plan.decode.size());
        bool drafting = false;
        for (size_t i = 0; i < plan.decode.size(); ++i) {
            tokens.push_back(plan.decode[i].token);
            positions.push_back(decode_positions[i]);
            caches.push_back(caches_[(size_t) plan.decode[i].slot]);
            drafting = drafting || mtp_[(size_t)plan.decode[i].slot].live;
        }
        // The drafting slots' trunk rows keep their draft layers caught up for their next verify.
        std::vector<float> hidden;
        std::vector<uint8_t> sampled(plan.decode.size());
        for (size_t i = 0; i < plan.decode.size(); ++i)
            sampled[i] = slots_.slot(plan.decode[i].slot).sampler.needs_logit_processing() ? 1 : 0;
        const auto forward = qwen4exp_forward_batched(
            backend_, weights_, caches.data(), tokens.data(), positions.data(),
            (int)tokens.size(), decode_workspace_, logits, -1,
            drafting ? &hidden : nullptr, &greedy, sampled.data());
        if (!forward.ok || logits.size() != tokens.size() || (!greedy.empty() && greedy.size() != tokens.size()))
            return fail("qwen4exp batched decode forward failed");
        const size_t hd = (size_t)weights_.n_embd * weights_.n_hc;
        for (size_t i = 0; drafting && i < plan.decode.size(); ++i) {
            const int id = plan.decode[i].slot;
            MtpState & m = mtp_[(size_t)id];
            if (!m.live) continue;
            Qwen4ExpMtpPending & pending = m.pending;
            pending.tok.push_back(tokens[i]);
            if (hidden.size() != hd * plan.decode.size()) {
                m.live = false;   // no trunk row: this request decodes without drafts from here on
            } else {
                pending.h.insert(pending.h.end(), hidden.begin() + (std::ptrdiff_t)(i * hd),
                                 hidden.begin() + (std::ptrdiff_t)((i + 1) * hd));
                if (pending.tok.size() >= kMtpPendingPairs &&
                    !qwen4exp_mtp_catch_up(backend_, weights_, *caches_[(size_t)id], pending, 0))
                    return fail("qwen4exp MTP catch-up failed");
            }
        }
    }
    logits.resize(forward_segments.size());
    for (size_t i = plan.decode.size(); i < forward_segments.size(); ++i) {
        const auto & segment = forward_segments[i];
        const int id = plan.prefills[i - plan.decode.size()].slot;
        if (mtp_[(size_t)id].live ? !mtp_prefill(id, segment.tokens, segment.n_tokens, segment.pos0, logits[i])
                : !qwen4exp_forward(backend_, weights_, *segment.cache, segment.tokens,
                                    segment.n_tokens, segment.pos0, logits[i]).ok)
            return fail("qwen4exp solo prefill forward failed");
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        const bool picked = i < plan.decode.size() && !greedy.empty() &&
                            !slots_.slot(plan.decode[i].slot).sampler.needs_logit_processing();
        if (!picked && logits[i].size() != (size_t) weights_.n_vocab)
            return fail("qwen4exp forward returned malformed logits");
    }
    if (!forward_segments.empty()) {
        ggml_backend_synchronize(backend_);
        for (const Qwen4ExpForwardSegment & segment : forward_segments)
            segment.cache->cur_pos = segment.pos0 + segment.n_tokens;
    }

    result.decode.reserve(plan.decode.size());
    for (size_t i = 0; i < plan.decode.size(); ++i) {
        const int slot_id = plan.decode[i].slot;
        SeqSlot & slot = slots_.slot(slot_id);
        slots_.commit_step(slot_id);
        DecodeOutput output;
        output.slot = slot_id;
        const std::vector<float> & row = logits[decode_rows[i]];
        output.token = slot.sampler.needs_logit_processing()
            ? sample_logits(row.data(), weights_.n_vocab, slot.sampler,
                            slot.sample_history, slot.rng)
            : !greedy.empty() ? greedy[i]
            : (int32_t)(std::max_element(row.begin(), row.end()) - row.begin());
        result.decode.push_back(std::move(output));
    }
    for (const PendingPrefill & pending : pending_prefills) {
        PrefillOutput output;
        output.slot = pending.slot;
        const PrefixCaptureTicket capture = slots_.slot(pending.slot).pending_capture;
        if (capture.valid() &&
            slots_.slot(pending.slot).cur_pos == capture.checkpoint.tokens) {
            output.prefix_store = capture_prefix(pending.slot, capture);
            slots_.slot(pending.slot).pending_capture = {};
        }
        if (pending.complete) {
            SeqSlot & slot = slots_.slot(pending.slot);
            output.status = PrefillOutput::Status::completed;
            const std::vector<float> & row = logits[pending.segment];
            output.token = slot.sampler.needs_logit_processing()
                ? sample_logits(row.data(), weights_.n_vocab,
                                slot.sampler, slot.sample_history, slot.rng)
                : (int32_t)(std::max_element(row.begin(), row.end()) - row.begin());
            slots_.commit_prefill(pending.slot);
        }
        result.prefills.push_back(std::move(output));
    }
    if (rate_plain) {
        ggml_backend_synchronize(backend_);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - plain_t0).count();
        record_mode((int)plan.decode.size(), false, (double)plan.decode.size(), s);
        ++batched_.plain_steps;
    }
    return result;
}

void Qwen4ExpSeqEngine::retire(int slot) {
    if (slot < 0 || slot >= slot_count()) return;
    MtpState & m = mtp_[(size_t)slot];
    if (m.context > 0) {
        if (m.drafts > 0) {
            std::fprintf(stderr,
                "[qwen4exp-mtp] slot=%d drafts=%lld accepted=%lld rate=%.3f tokens_per_step=%.3f "
                "decode=%.2f tok/s mtp_steps=%lld live=%d step_ms=%.1f draft_ms=%.1f verify_ms=%.1f "
                "verify_builds=%llu batched_builds=%llu batched_replays=%llu lookup_rounds=%lld lookup_drafts=%lld "
                "lookup_accepted=%lld\n",
                slot, m.drafts, m.accepted, (double)m.accepted / (double)m.drafts,
                m.steps > 0 ? (double)m.tokens / (double)m.steps : 0.0,
                m.decode_s > 0.0 ? (double)m.tokens / m.decode_s : 0.0, m.steps, (int)m.live,
                1e3 * m.decode_s / (double)m.steps, 1e3 * m.draft_s / (double)m.steps,
                1e3 * m.verify_s / (double)m.steps, (unsigned long long)m.verify_builds,
                (unsigned long long)decode_workspace_.builds, (unsigned long long)decode_workspace_.replays,
                m.lookup_rounds, m.lookup_drafts, m.lookup_accepted);
        }
        if (slot == mtp_slot_ && verify_width_ == 0) mtp_costs_->store(*mtp_width_, m.context);
        m = MtpState{};
    }
    // The batched steps' summary once no other request is left: one line per busy stretch.
    bool last = true;
    for (int s = 0; s < slot_count(); ++s) last = last && (s == slot || !slots_.is_active(s));
    if (last && batched_.steps > 0) {
        std::fprintf(stderr,
            "[qwen4exp-mtp] batched verify steps=%lld slots=%.2f tokens_per_step=%.2f step_ms=%.1f draft_ms=%.1f "
            "verify_ms=%.1f builds=%llu replays=%llu plain_steps=%lld\n",
            batched_.steps, (double)batched_.slots / (double)batched_.steps,
            (double)batched_.tokens / (double)batched_.steps, 1e3 * batched_.step_s / (double)batched_.steps,
            1e3 * batched_.draft_s / (double)batched_.steps, 1e3 * batched_.verify_s / (double)batched_.steps,
            (unsigned long long)verify_workspace_.builds, (unsigned long long)verify_workspace_.replays,
            batched_.plain_steps);
        for (int k = 2; k < (int)mode_rates_.size(); ++k) {
            const ModeRates & r = mode_rates_[(size_t)k];
            if (r.steps == 0) continue;
            std::fprintf(stderr, "[qwen4exp-mtp] %d at once: verify %.1f tok/s (%lld steps), plain %.1f tok/s (%lld)\n",
                k, r.verify_s > 0 ? r.verify_tokens / r.verify_s : 0.0, r.verify_steps,
                r.plain_s > 0 ? r.plain_tokens / r.plain_s : 0.0, r.plain_steps);
        }
    }
    if (last) batched_ = {};
    ggml_backend_synchronize(backend_);
    if (slots_.is_active(slot)) slots_.retire(slot);
    prefill_cuts_[(size_t)slot].clear();
    reset_qwen4exp_state(backend_, *caches_[(size_t)slot]);
    ggml_backend_synchronize(backend_);
}

} // namespace luce::common
