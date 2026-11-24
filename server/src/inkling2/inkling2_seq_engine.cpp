// Inkling2SeqEngine — Concurrent N-slot batched 2DSD SeqEngine for Inkling 2 Mini.

#include "inkling2_seq_engine.h"
#include "inkling2_2dsd_draft.h"
#include "inkling2_2dsd_graph.h"

#include "common/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace luce::common {
namespace {

static uint32_t compute_pool_blocks(int max_ctx, int n_slots) {
    if (max_ctx <= 0 || n_slots <= 0) return 0;
    const uint64_t per_slot = (uint64_t(max_ctx) + 255u) / 256u;
    const uint64_t total = per_slot * (uint64_t) n_slots;
    return total <= std::numeric_limits<uint32_t>::max() ? (uint32_t) total : 0;
}

static bool sampler_uses_fast_gpu(const SamplerCfg & s) {
    return s.needs_logit_processing() && s.temp > 0.0f &&
           s.rep_pen <= 1.0f && s.freq_pen == 0.0f &&
           s.pres_pen == 0.0f && s.constraint == nullptr;
}

static bool try_build_exact_cands_row(const int32_t * rc,
                                      const SamplerCfg & sampler,
                                      std::vector<std::pair<int32_t, float>> & out_probs,
                                      double & out_cum) {
    const int cnt = rc[0];
    if (cnt <= 0 || cnt > 256) return false;
    float gmax = 0.0f, Z = 0.0f;
    std::memcpy(&gmax, &rc[1], sizeof(float));
    std::memcpy(&Z,    &rc[2], sizeof(float));
    if (!(Z > 0.0f) || !std::isfinite(Z)) return false;
    struct Cand { float v; int32_t id; };
    Cand cands[256];
    for (int j = 0; j < cnt; ++j) {
        std::memcpy(&cands[j].v, &rc[4 + 2 * j], sizeof(float));
        cands[j].id = rc[4 + 2 * j + 1];
        if (!std::isfinite(cands[j].v)) return false;
    }
    std::sort(cands, cands + cnt, [](const Cand & a, const Cand & b) {
        return a.v > b.v || (a.v == b.v && a.id < b.id);
    });
    const int k_lim = (sampler.top_k > 0 && sampler.top_k < cnt) ? sampler.top_k : cnt;
    const float inv_t = 1.0f / std::max(sampler.temp, 1e-6f);
    out_probs.clear();
    out_probs.reserve((size_t) k_lim);
    out_cum = 0.0;
    for (int j = 0; j < k_lim; ++j) {
        const float p = std::exp((cands[j].v - gmax) * inv_t) / Z;
        out_probs.push_back({ cands[j].id, p });
        out_cum += p;
        if (sampler.top_p > 0.0f && sampler.top_p < 1.0f && out_cum >= sampler.top_p) {
            break;
        }
    }
    const bool exact_ok =
        (sampler.top_k > 0 && cnt >= sampler.top_k) ||
        (sampler.top_p > 0.0f && sampler.top_p < 1.0f && out_cum >= sampler.top_p) ||
        (out_cum >= 0.9995);
    return exact_ok && !out_probs.empty() && out_cum > 0.0;
}

static int resolve_draft_window() {
    int win = 2048;
    if (const char * ew = std::getenv("LUCE_INKLING2_2DSD_DRAFT_WINDOW")) {
        win = std::atoi(ew);
    }
    return win;
}

class SlotCommitRunner final : public Inkling2TwoDSDRunner {
public:
    SlotCommitRunner(ggml_backend_t b, const Inkling2Weights & w, Inkling2Cache & c)
        : b_(b), w_(w), c_(c) {}

    int prefix_length() const override { return c_.tree.prefix_len; }
    int max_nodes() const override { return c_.tree.max_nodes; }
    int max_ctx() const override { return c_.max_ctx; }

    bool advance_nodes(const std::vector<Inkling2TwoDSDNodeDesc> &, uint32_t,
                       std::vector<int> &, std::vector<int32_t> &) override {
        return false; // batched advance runs at engine level
    }
    bool commit_nodes(const int * slots, int n) override {
        return inkling2_2dsd_commit_nodes(b_, w_, c_, slots, n);
    }
    bool draft_batch(const std::vector<Inkling2TwoDSDDraftReq> & reqs,
                     std::vector<Inkling2TwoDSDDraftOut> & out) override {
        return inkling2_2dsd_draft_batch(b_, w_, c_, reqs, out);
    }

private:
    ggml_backend_t b_;
    const Inkling2Weights & w_;
    Inkling2Cache & c_;
};

} // namespace

struct Inkling2SeqEngine::SlotState {
    std::vector<int32_t> prompt;
    int prefill_pos = 0;
    std::unique_ptr<SlotCommitRunner> runner;
    std::unique_ptr<Inkling2TwoDSDSession> session;
    Inkling2TwoDSDStats stats;
    long long last_reported_draft_slots = 0;

    // Transient per-micro-step sampling bridge (valid only during apply_step_and_verify)
    const Inkling2TwoDSDAdvanceSampleOut * cur_batch_sample = nullptr;
    const std::vector<int> * cur_global_readout_idx = nullptr;
    bool cur_fast_gpu_active = false;
    std::vector<int32_t> cur_penalty_hist;
    std::vector<float> scratch_logits;
    Inkling2TwoDSDSamplerHooks hooks;
};

Inkling2SeqEngine::Inkling2SeqEngine(
        ggml_backend_t target_backend,
        const Inkling2Weights & weights,
        Inkling2Cache & multi_cache,
        int max_concurrency,
        int prefill_chunk)
    : backend_(target_backend),
      weights_(weights),
      multi_cache_(multi_cache),
      n_slots_(std::max(1, std::min(max_concurrency, multi_cache.tree.n_seq_slots))),
      prefill_chunk_(std::max(64, prefill_chunk)),
      pool_(compute_pool_blocks(multi_cache.max_ctx, n_slots_), (uint32_t) n_slots_, 256),
      slots_(pool_, multi_cache.max_ctx),
      slot_caches_((size_t) n_slots_),
      slot_state_((size_t) n_slots_) {
    for (int s = 0; s < n_slots_; ++s) {
        slot_state_[(size_t) s] = std::make_unique<SlotState>();
    }
    init_slot_views();
}

Inkling2SeqEngine::~Inkling2SeqEngine() {
    release_device_state();
}

bool Inkling2SeqEngine::init_slot_views() {
    for (int s = 0; s < n_slots_; ++s) {
        if (!make_inkling2_slot_view(multi_cache_, weights_, s, slot_caches_[(size_t) s])) {
            return false;
        }
        slot_state_[(size_t) s]->runner =
            std::make_unique<SlotCommitRunner>(backend_, weights_, slot_caches_[(size_t) s]);
    }
    return true;
}

void Inkling2SeqEngine::release_device_state() {
    if (backend_) ggml_backend_synchronize(backend_);
    for (int s = 0; s < n_slots_; ++s) {
        slot_state_[(size_t) s]->session.reset();
        slot_state_[(size_t) s]->runner.reset();
        free_inkling2_cache(slot_caches_[(size_t) s]);
    }
}

bool Inkling2SeqEngine::reinstate_device_state(int prefill_chunk) {
    prefill_chunk_ = std::max(64, prefill_chunk);
    return init_slot_views();
}

bool Inkling2SeqEngine::token_is_eos(int32_t token) const {
    return token == weights_.eos_id || token == weights_.eos_chat_id;
}

void Inkling2SeqEngine::reset_slot_session(int slot) {
    auto & st = *slot_state_[(size_t) slot];
    st.session.reset();
    st.cur_batch_sample = nullptr;
    st.cur_global_readout_idx = nullptr;
    st.cur_fast_gpu_active = false;
    st.cur_penalty_hist.clear();
}

SeqEngine::AdmitResult Inkling2SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    for (int32_t tok : prompt) {
        if (tok < 0 || tok >= weights_.n_vocab_unpadded) {
            AdmitResult inv;
            inv.status = AdmitResult::Status::failed;
            inv.error = "inkling2 prompt contains an invalid token id";
            return inv;
        }
    }
    AdmitResult res = slots_.admit(request_id, prompt, sampler);
    if (res.status != AdmitResult::Status::admitted) return res;
    if (res.slot < 0 || res.slot >= n_slots_ || !slot_caches_[(size_t) res.slot].tree.active) {
        slots_.retire(res.slot);
        res.status = AdmitResult::Status::failed;
        res.error = "inkling2 slot view not initialized";
        return res;
    }
    if (pool_.reserve_capacity(slots_.slot(res.slot).handle, (uint32_t) max_context()) != PagedKvStatus::Ok) {
        slots_.retire(res.slot);
        res.status = AdmitResult::Status::busy;
        res.error = "inkling2 full-context reservation unavailable";
        return res;
    }

    reset_inkling2_cache(slot_caches_[(size_t) res.slot]);
    auto & st = *slot_state_[(size_t) res.slot];
    reset_slot_session(res.slot);
    st.prompt = prompt;
    st.prefill_pos = 0;
    st.stats = {};
    st.last_reported_draft_slots = 0;
    return res;
}

StepPlanLimits Inkling2SeqEngine::step_plan_limits(int decode_rows) const {
    (void) decode_rows;
    StepPlanLimits lim;
    lim.max_prefill_sequences = 1;
    lim.max_prefill_tokens_per_sequence = prefill_chunk_;
    lim.max_prefill_tokens_total = prefill_chunk_;
    lim.prefill_allocation_quantum = prefill_chunk_;
    lim.prefill_time_share = 1.0;
    lim.time_share_prompt_tokens = 0;
    return lim;
}

bool Inkling2SeqEngine::reserve_decode(const StepPlan & plan) {
    std::vector<int> growth((size_t) n_slots_, 0);
    for (const StepInput & in : plan.decode) {
        if (in.slot < 0 || in.slot >= n_slots_) return false;
        const int rem = std::max(0, max_context() - slots_.slot(in.slot).cur_pos);
        if (rem <= 0) return false;
        growth[(size_t) in.slot] = std::min(rem, in.allow_speculation ? max_decode_tokens() : 1);
    }
    return slots_.reserve_decode(growth);
}

void Inkling2SeqEngine::retire(int slot) {
    if (slot >= 0 && slot < n_slots_) {
        reset_slot_session(slot);
        slot_state_[(size_t) slot]->prompt.clear();
        slot_state_[(size_t) slot]->prefill_pos = 0;
    }
    slots_.retire(slot);
}

SeqEngine::StepResult Inkling2SeqEngine::step(const StepPlan & plan) {
    StepResult result;

    // ── 0. Pre-mutation validation ───────────────────────────────────────
    std::vector<uint8_t> seen_dec((size_t) n_slots_, 0);
    std::vector<uint8_t> seen_pre((size_t) n_slots_, 0);
    for (const StepInput & in : plan.decode) {
        if (in.slot < 0 || in.slot >= n_slots_) {
            result.error = "inkling2 step: decode slot out of range";
            return result;
        }
        if (in.token < 0 || in.token >= weights_.n_vocab_unpadded) {
            result.error = "inkling2 step: decode token out of range";
            return result;
        }
        if (seen_dec[(size_t) in.slot]) {
            result.error = "inkling2 step: duplicate decode slot";
            return result;
        }
        if (!slots_.slot(in.slot).decoding()) {
            result.error = "inkling2 step: decode slot is not in decode phase";
            return result;
        }
        seen_dec[(size_t) in.slot] = 1;
    }
    for (const PrefillSlice & sl : plan.prefills) {
        if (sl.slot < 0 || sl.slot >= n_slots_ || sl.max_tokens <= 0) {
            result.error = "inkling2 step: invalid prefill slice";
            return result;
        }
        if (seen_dec[(size_t) sl.slot] || seen_pre[(size_t) sl.slot]) {
            result.error = "inkling2 step: slot assigned more than once";
            return result;
        }
        if (!slots_.slot(sl.slot).prefilling()) {
            result.error = "inkling2 step: prefill slot is not in prefill phase";
            return result;
        }
        seen_pre[(size_t) sl.slot] = 1;
    }
    for (int s = 0; s < n_slots_; ++s) {
        if (slots_.slot(s).decoding() && !seen_dec[(size_t) s]) {
            result.error = "inkling2 step: plan omits a decoding slot";
            return result;
        }
    }

    const int n_vocab = weights_.n_vocab_unpadded;

    // ── 1. Prefill selected prompt slices (zero-copy per-slot cache views) ──
    for (const PrefillSlice & slice : plan.prefills) {
        const int s = slice.slot;
        SeqSlot & sl = slots_.slot(s);
        SlotState & st = *slot_state_[(size_t) s];
        Inkling2Cache & sc = slot_caches_[(size_t) s];

        const int rem = sl.prompt_len - st.prefill_pos;
        const int take = std::min(rem, slice.max_tokens);
        if (take <= 0) {
            PrefillOutput po;
            po.slot = s;
            po.status = PrefillOutput::Status::failed;
            po.error = "inkling2 prefill: no remaining prompt tokens";
            result.prefills.push_back(std::move(po));
            continue;
        }
        if (st.prefill_pos == 0) {
            const int win = resolve_draft_window();
            sc.tree.draft_pref_start = (win > 0 && sl.prompt_len > win) ? (sl.prompt_len - win) : 0;
        }

        const bool is_last = (st.prefill_pos + take >= sl.prompt_len);
        const bool is_greedy = !sl.sampler.needs_logit_processing();
        int32_t first_tok = -1;
        std::vector<float> last_logits;
        const bool ok = inkling2_2dsd_prefill_chunk(
            backend_, weights_, sc,
            st.prompt.data() + st.prefill_pos, take, st.prefill_pos,
            is_last, first_tok, (is_last && !is_greedy) ? &last_logits : nullptr);
        if (!ok) {
            result.decode.clear();
            result.prefills.clear();
            result.error = "inkling2 2dsd prefill chunk failed";
            return result;
        }

        auto chunk_res = slots_.append_prefill(s, take);
        if (!chunk_res.ok) {
            result.decode.clear();
            result.prefills.clear();
            result.error = "inkling2 slot manager append_prefill failed";
            return result;
        }
        st.prefill_pos += take;

        PrefillOutput po;
        po.slot = s;
        if (is_last) {
            if (!is_greedy && !last_logits.empty()) {
                auto ph = sl.penalty_history();
                first_tok = sample_logits(last_logits.data(), n_vocab, sl.sampler, ph, sl.rng);
            }
            slots_.commit_prefill(s);
            po.status = PrefillOutput::Status::completed;
            po.token = first_tok;
        } else {
            po.status = PrefillOutput::Status::advanced;
        }
        result.prefills.push_back(std::move(po));
    }

    if (plan.decode.empty()) return result;

    // ── 2. Prepare decoding sessions for this cohort ─────────────────────
    const int n_dec = (int) plan.decode.size();
    const Inkling2TwoDSDConfig base_cfg = inkling2_2dsd_config_for_concurrency(
        weights_.n_passes, weights_.twodsd.block_size, n_dec);
    const bool allow_split = (weights_.expert_backend != nullptr && weights_.split_exp > 0);

    std::vector<DecodeOutput> dec_outs((size_t) n_dec);
    std::vector<int> pending_idx; // indices into plan.decode still awaiting commit this step
    pending_idx.reserve((size_t) n_dec);

    for (int i = 0; i < n_dec; ++i) {
        const StepInput & in = plan.decode[(size_t) i];
        const int s = in.slot;
        SeqSlot & sl = slots_.slot(s);
        SlotState & st = *slot_state_[(size_t) s];
        Inkling2Cache & sc = slot_caches_[(size_t) s];

        dec_outs[(size_t) i].slot = s;
        if (sc.tree.prefix_len >= max_context()) {
            dec_outs[(size_t) i].failed = true;
            dec_outs[(size_t) i].error = "inkling2 decode: context limit reached";
            continue;
        }

        Inkling2TwoDSDConfig slot_cfg = base_cfg;
        if (!in.allow_speculation) {
            slot_cfg.k = 1;
            slot_cfg.stage = Inkling2TwoDSDStage::Stage1;
        }

        const bool is_greedy = !sl.sampler.needs_logit_processing();
        const bool warm_match = (st.session != nullptr) &&
                                (st.session->cursor_token() == in.token) &&
                                (st.session->cursor_pos() == sc.tree.prefix_len) &&
                                (st.session->cursor_depth() >= 0) &&
                                (st.session->cursor_depth() < weights_.n_passes);
        if (warm_match) {
            st.session->update_config(slot_cfg);
            if (!in.allow_speculation) {
                st.session->clear_cursor_children();
            }
        } else {
            reset_slot_session(s);
            if (!is_greedy) {
                st.scratch_logits.resize((size_t) n_vocab);
                st.hooks.capture = [&st, &sl, n_vocab](int local_row_idx, Inkling2TwoDSDDepthRSample & out) {
                    out.top_cands.clear();
                    out.logits.clear();
                    if (!st.cur_batch_sample || !st.cur_global_readout_idx) return;
                    if (local_row_idx < 0 || local_row_idx >= (int) st.cur_global_readout_idx->size()) return;
                    const int g_idx = (*st.cur_global_readout_idx)[(size_t) local_row_idx];
                    if (st.cur_fast_gpu_active &&
                        (size_t) (g_idx + 1) * 516 <= st.cur_batch_sample->top_cands.size()) {
                        const int32_t * rc = st.cur_batch_sample->top_cands.data() + (size_t) g_idx * 516;
                        std::vector<std::pair<int32_t, float>> tmp_probs;
                        double tmp_cum = 0.0;
                        if (try_build_exact_cands_row(rc, sl.sampler, tmp_probs, tmp_cum)) {
                            out.top_cands.assign(rc, rc + 516);
                            return;
                        }
                    }
                    if (st.cur_batch_sample->logits_dev) {
                        out.logits.resize((size_t) n_vocab);
                        ggml_backend_tensor_get(
                            st.cur_batch_sample->logits_dev, out.logits.data(),
                            (size_t) g_idx * (size_t) n_vocab * sizeof(float),
                            (size_t) n_vocab * sizeof(float));
                    }
                };
                st.hooks.sample = [&st, &sl, n_vocab](const Inkling2TwoDSDDepthRSample & payload,
                                                      int32_t greedy_tok) -> int32_t {
                    if (payload.top_cands.size() >= 516) {
                        std::vector<std::pair<int32_t, float>> probs;
                        double cum = 0.0;
                        if (try_build_exact_cands_row(payload.top_cands.data(), sl.sampler, probs, cum)) {
                            std::uniform_real_distribution<double> u01(0.0, cum);
                            const double r = u01(sl.rng);
                            double acc = 0.0;
                            for (size_t j = 0; j < probs.size(); ++j) {
                                acc += probs[j].second;
                                if (r <= acc || j + 1 == probs.size()) return probs[j].first;
                            }
                        }
                    }
                    if (!payload.logits.empty()) {
                        std::memcpy(st.scratch_logits.data(), payload.logits.data(),
                                    (size_t) n_vocab * sizeof(float));
                        return sample_logits(st.scratch_logits.data(), n_vocab,
                                             sl.sampler, st.cur_penalty_hist, sl.rng);
                    }
                    return greedy_tok;
                };
            }
            st.session = std::make_unique<Inkling2TwoDSDSession>(
                *st.runner, slot_cfg, st.stats, max_context(),
                is_greedy ? nullptr : &st.hooks);
            std::string serr;
            if (!st.session->seed_cursor(in.token, sc.tree.prefix_len, serr)) {
                result.decode.clear();
                result.prefills.clear();
                result.error = serr;
                return result;
            }
        }
        pending_idx.push_back(i);
    }

    // ── 3. Batched micro-step loop (1 GPU call in steady state, <= R on cold start) ─
    std::vector<int> prefix_lens((size_t) n_slots_, 0);
    for (int guard = 0; !pending_idx.empty() && guard < weights_.n_passes + 2; ++guard) {
        // 3a. Run per-slot drafter batches for any pending anchor events
        for (int idx : pending_idx) {
            const int s = plan.decode[(size_t) idx].slot;
            SlotState & st = *slot_state_[(size_t) s];
            std::vector<Inkling2TwoDSDDraftReq> reqs;
            std::vector<int> plan_heads, plan_widths;
            st.session->collect_draft_reqs(reqs, plan_heads, plan_widths);
            if (!reqs.empty()) {
                std::vector<Inkling2TwoDSDDraftOut> outs;
                if (!inkling2_2dsd_draft_batch(backend_, weights_, slot_caches_[(size_t) s], reqs, outs) ||
                    outs.size() != reqs.size()) {
                    result.decode.clear();
                    result.prefills.clear();
                    result.error = "inkling2 2dsd draft_batch failed";
                    return result;
                }
                ++st.stats.drafter_batches;
                st.stats.drafter_calls += (long long) reqs.size();
                st.session->apply_draft_outs(plan_heads, plan_widths, outs);
            }
        }

        // 3b. Pack active nodes from ALL pending slots into ONE batched target step
        std::vector<Inkling2TwoDSDNodeDesc> combined_descs;
        std::vector<int> desc_owner_idx;      // index into plan.decode
        std::vector<int> desc_local_pos;      // index into slot's active_ids
        std::vector<std::vector<int>> slot_active_ids((size_t) n_dec);
        uint32_t combined_mask = 0u;

        bool any_stochastic = false;
        bool all_fast_same_temp = true;
        float shared_fast_temp = 0.0f;

        for (int idx : pending_idx) {
            const int s = plan.decode[(size_t) idx].slot;
            const SeqSlot & sl = slots_.slot(s);
            SlotState & st = *slot_state_[(size_t) s];

            std::vector<Inkling2TwoDSDNodeDesc> descs;
            std::string cerr;
            if (!st.session->collect_active(slot_active_ids[(size_t) idx], descs, cerr)) {
                result.decode.clear();
                result.prefills.clear();
                result.error = cerr;
                return result;
            }
            combined_mask |= st.session->readout_mask();
            for (int r = 0; r < (int) descs.size(); ++r) {
                descs[(size_t) r].seq_slot = s;
                combined_descs.push_back(std::move(descs[(size_t) r]));
                desc_owner_idx.push_back(idx);
                desc_local_pos.push_back(r);
            }
            if (sl.sampler.needs_logit_processing()) {
                any_stochastic = true;
                if (!sampler_uses_fast_gpu(sl.sampler)) {
                    all_fast_same_temp = false;
                } else if (shared_fast_temp == 0.0f) {
                    shared_fast_temp = sl.sampler.temp;
                } else if (shared_fast_temp != sl.sampler.temp) {
                    all_fast_same_temp = false;
                }
            }
        }

        for (int s = 0; s < n_slots_; ++s) {
            prefix_lens[(size_t) s] = slot_caches_[(size_t) s].tree.prefix_len;
        }

        Inkling2TwoDSDAdvanceSampleOut batch_sample;
        batch_sample.fast_temp = (any_stochastic && all_fast_same_temp) ? shared_fast_temp : 0.0f;

        std::vector<int> global_readout_rows;
        std::vector<int32_t> global_readout_argmax;
        if (!inkling2_2dsd_advance_nodes_batched(
                backend_, weights_, multi_cache_, combined_descs,
                prefix_lens.data(), n_slots_, combined_mask, allow_split,
                global_readout_rows, global_readout_argmax,
                any_stochastic ? &batch_sample : nullptr)) {
            result.decode.clear();
            result.prefills.clear();
            result.error = "inkling2_2dsd_advance_nodes_batched failed";
            return result;
        }

        // 3c. Demultiplex readouts back to each pending slot
        std::vector<std::vector<int>> per_slot_rrows((size_t) n_dec);
        std::vector<std::vector<int32_t>> per_slot_rargmax((size_t) n_dec);
        std::vector<std::vector<int>> per_slot_gidx((size_t) n_dec);
        for (int g = 0; g < (int) global_readout_rows.size(); ++g) {
            const int comb_r = global_readout_rows[(size_t) g];
            const int idx    = desc_owner_idx[(size_t) comb_r];
            per_slot_rrows[(size_t) idx].push_back(desc_local_pos[(size_t) comb_r]);
            per_slot_rargmax[(size_t) idx].push_back(global_readout_argmax[(size_t) g]);
            per_slot_gidx[(size_t) idx].push_back(g);
        }

        // 3d. Verify each pending slot and collect outputs for slots that committed >= 1 token
        std::vector<int> next_pending;
        next_pending.reserve(pending_idx.size());

        for (int idx : pending_idx) {
            const StepInput & in = plan.decode[(size_t) idx];
            const int s = in.slot;
            SeqSlot & sl = slots_.slot(s);
            SlotState & st = *slot_state_[(size_t) s];
            Inkling2Cache & sc = slot_caches_[(size_t) s];

            ++st.stats.target_calls;
            st.stats.target_cells += (long long) slot_active_ids[(size_t) idx].size();

            st.cur_batch_sample = any_stochastic ? &batch_sample : nullptr;
            st.cur_global_readout_idx = &per_slot_gidx[(size_t) idx];
            st.cur_fast_gpu_active = (batch_sample.fast_temp > 0.0f) && sampler_uses_fast_gpu(sl.sampler);

            // Prepare live penalty history (generated tokens so far + fed token `in.token`)
            st.cur_penalty_hist.clear();
            if (sl.sampler.needs_logit_processing()) {
                st.cur_penalty_hist = sl.penalty_history();
                st.cur_penalty_hist.push_back(in.token);
            }

            const int rem_ctx = std::max(1, max_context() - sc.tree.prefix_len);
            const int max_emit = in.allow_speculation ? std::min(max_decode_tokens(), rem_ctx) : 1;
            std::vector<int32_t> emitted;
            auto on_commit = [&](int32_t & tok) -> bool {
                emitted.push_back(tok);
                if (sl.sampler.needs_logit_processing()) {
                    st.cur_penalty_hist.push_back(tok);
                }
                if ((int) emitted.size() >= max_emit) return false;
                if (token_is_eos(tok)) return false;
                return true;
            };

            bool stopped = false;
            std::string verr;
            const bool ok = st.session->apply_step_and_verify(
                slot_active_ids[(size_t) idx],
                per_slot_rrows[(size_t) idx],
                per_slot_rargmax[(size_t) idx],
                on_commit, stopped, verr);

            st.cur_batch_sample = nullptr;
            st.cur_global_readout_idx = nullptr;
            st.cur_fast_gpu_active = false;

            if (!ok) {
                result.decode.clear();
                result.prefills.clear();
                result.error = verr;
                return result;
            }

            if (!emitted.empty()) {
                // Commit fed token + any accepted speculative children to SeqSlotManager
                std::vector<int32_t> kv_tokens;
                kv_tokens.reserve(emitted.size());
                kv_tokens.push_back(in.token);
                for (size_t j = 0; j + 1 < emitted.size(); ++j) {
                    kv_tokens.push_back(emitted[j]);
                }
                auto app = slots_.append_tokens(s, kv_tokens.data(), (int) kv_tokens.size());
                if (!app.ok) {
                    result.decode.clear();
                    result.prefills.clear();
                    result.error = "inkling2 slot manager append_tokens failed";
                    return result;
                }
                slots_.commit_step(s);

                DecodeOutput & out = dec_outs[(size_t) idx];
                out.token = emitted.back();
                out.committed_tokens.assign(emitted.begin(), emitted.end() - 1);
                out.spec_accepted = (int) out.committed_tokens.size();
                if (in.allow_speculation) {
                    const long long delta = st.stats.draft_slots - st.last_reported_draft_slots;
                    out.spec_offered = std::max(out.spec_accepted, (int) std::max<long long>(0, delta));
                } else {
                    out.spec_offered = 0;
                }
                st.last_reported_draft_slots = st.stats.draft_slots;
            } else {
                next_pending.push_back(idx);
            }
        }
        pending_idx = std::move(next_pending);
    }

    if (!pending_idx.empty()) {
        result.decode.clear();
        result.prefills.clear();
        result.error = "inkling2 2dsd step failed to produce output within R micro-steps";
        return result;
    }

    result.decode = std::move(dec_outs);
    return result;
}

} // namespace luce::common
