// GPU adapter contract and lifecycle soak for the full-cache engine, and with every slot's cache carrying the MTP draft
// layer (as the server makes them): prefix checkpoints across slots, and every request of a concurrent run producing
// its alone stream token for token, with the batched verify forced on and off, and with requests that may not draft
// (one-row slots of the verify) beside drafting ones.
// Usage: test_qwen4exp_seq_engine <shard1.gguf> [ctx=32768] (the MTP sidecar is found beside the model, as the server
// finds it)
#include "qwen4exp_seq_engine.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_internal.h"
#include "common/concurrency/seq_engine.h"
#include "common/prompt_lookup.h"
#include "seq_engine_contract.h"
#include "common/sampler.h"
#include "common/snapshot_backend.h"
#include "server/tokenizer.h"
#include "ggml-cuda.h"
#include "qwen4exp_test_state.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace luce::common;

static int argmax(const std::vector<float> & logits) {
    return (int)std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end()));
}

static float max_delta(const std::vector<float> & a,
                       const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    float delta = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        delta = std::max(delta, std::abs(a[i] - b[i]));
    }
    return delta;
}

static bool run_distinct(Qwen4ExpSeqEngine & engine, ggml_backend_t backend,
                         const Qwen4ExpWeights & weights,
                         std::vector<Qwen4ExpCache *> caches,
                         const char * model_path) {
    constexpr int N = 4, STEPS = 48;
    const std::vector<std::string> prompts = {
        "Answer with one short sentence, without reasoning: what is 2 + 2?\nAnswer:",
        "Answer with one short sentence, without reasoning: what is the capital of France?\nAnswer:",
        "Answer with one short sentence, without reasoning: which planet is largest in our solar system?\nAnswer:",
        "Answer with one short sentence, without reasoning: what color do red and white paint make?\nAnswer:",
    };
    const std::vector<std::string> expected = {"4", "paris", "jupiter", "pink"};
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model_path)) return false;
    std::vector<std::vector<int32_t>> ids(N);
    bool coherent_all = true;
    for (int i = 0; i < N; ++i) {
        ids[i] = tokenizer.encode(prompts[i]);
        if (ids[i].empty() || ids[i].size() + STEPS >= 32768) return false;
    }

    // First run through the actual SeqEngine admission/prefill/decode contract.
    std::vector<int32_t> engine_next(N, -1);
    std::vector<std::vector<int32_t>> engine_streams(N);
    for (int i = 0; i < N; ++i) {
        const auto admitted = engine.admit(100 + i, ids[i], SamplerCfg{});
        if (admitted.status != SeqEngine::AdmitResult::Status::admitted ||
            admitted.slot != i) return false;
    }
    std::vector<uint8_t> live(N, 0);
    for (int i = 0; i < N; ++i) {
        SeqEngine::StepPlan plan;
        for (int slot = 0; slot < i; ++slot)
            if (live[slot]) plan.decode.push_back({slot, engine_next[slot]});
        plan.prefills.push_back({i, 512});
        if (!engine.reserve_decode(plan)) return false;
        const auto result = engine.step(plan);
        if (!result.ok() || result.prefills.size() != 1 ||
            result.prefills[0].status != SeqEngine::PrefillOutput::Status::completed)
            return false;
        for (const auto & output : result.decode) {
            engine_streams[output.slot].push_back(output.token);
            engine_next[output.slot] = output.token;
            if (engine.token_is_eos(output.token)) {
                live[output.slot] = 0;
                engine.retire(output.slot);
            }
        }
        engine_next[i] = result.prefills[0].token;
        engine_streams[i].push_back(engine_next[i]);
        live[i] = !engine.token_is_eos(engine_next[i]);
        if (!live[i]) engine.retire(i);
    }
    for (int step = 0; step < STEPS; ++step) {
        SeqEngine::StepPlan plan;
        for (int i = 0; i < N; ++i)
            if (live[i]) plan.decode.push_back({i, engine_next[i]});
        if (plan.decode.empty()) break;
        if (!engine.reserve_decode(plan)) return false;
        const auto result = engine.step(plan);
        if (!result.ok() || result.decode.size() != plan.decode.size()) return false;
        for (const auto & output : result.decode) {
            engine_streams[output.slot].push_back(output.token);
            engine_next[output.slot] = output.token;
            if (engine.token_is_eos(output.token)) {
                live[output.slot] = 0;
                engine.retire(output.slot);
            }
        }
    }
    for (int i = 0; i < N; ++i) engine.retire(i);

    for (int i = 0; i < N; ++i) {
        const std::string answer = tokenizer.decode(engine_streams[i]);
        std::string lower = answer;
        std::transform(lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return (char)std::tolower(c); });
        const bool coherent = lower.find(expected[i]) != std::string::npos;
        std::printf("[distinct] slot=%d answer=%s expected=%s coherent=%s\n",
                    i, answer.c_str(), expected[i].c_str(),
                    coherent ? "yes" : "no");
        coherent_all = coherent_all && coherent;
    }

    // Compare full-vocabulary logits and complete engine token streams exactly.
    // One reusable solo cache keeps the peak at five full caches, not eight.
    Qwen4ExpCache solo;
    if (!create_qwen4exp_cache(backend, weights, 32768, solo))
        return false;
    std::vector<std::vector<std::vector<float>>> solo_logits(N);
    std::vector<std::vector<int32_t>> solo_streams(N);
    for (int i = 0; i < N; ++i) {
        reset_qwen4exp_state(backend, solo);
        std::vector<float> logits;
        if (!qwen4exp_forward(backend, weights, solo, ids[i].data(),
                              (int)ids[i].size(), 0, logits).ok) {
            free_qwen4exp_cache(solo); return false;
        }
        solo_streams[i].push_back(argmax(logits));
        // Earlier admissions also decoded while later slots were prefilling.
        for (int step = 0; step < STEPS + N - 1 - i; ++step) {
            const int32_t fed = solo_streams[i].back();
            if (weights.eos_id == fed || weights.eos_chat_id == fed) break;
            std::vector<float> next_logits;
            if (!qwen4exp_forward(backend, weights, solo, &fed, 1,
                                  (int)ids[i].size() + step,
                                  next_logits).ok) {
                free_qwen4exp_cache(solo); return false;
            }
            solo_logits[i].push_back(std::move(next_logits));
            solo_streams[i].push_back(argmax(solo_logits[i].back()));
            if (weights.eos_id == solo_streams[i].back() ||
                weights.eos_chat_id == solo_streams[i].back()) break;
        }
    }
    free_qwen4exp_cache(solo);

    for (Qwen4ExpCache * cache : caches) reset_qwen4exp_state(backend, *cache);
    std::vector<int32_t> batch_next(N);
    std::vector<int> first_divergence(N, -1);
    for (int i = 0; i < N; ++i) {
        std::vector<float> ignored;
        if (!qwen4exp_forward(backend, weights, *caches[i], ids[i].data(),
                              (int)ids[i].size(), 0, ignored).ok) return false;
        batch_next[i] = solo_streams[i][0];
    }
    Qwen4ExpBatchedDecodeWorkspace workspace;
    std::vector<float> max_epsilon(N, 0.0f);
    for (int step = 0; step < STEPS; ++step) {
        Qwen4ExpCache * rows[N];
        int32_t positions[N];
        for (int i = 0; i < N; ++i) {
            rows[i] = caches[i];
            positions[i] = (int32_t)ids[i].size() + step;
        }
        std::vector<std::vector<float>> logits;
        if (!qwen4exp_forward_batched(backend, weights, rows, batch_next.data(),
                positions, N, workspace, logits).ok ||
            logits.size() != N) {
            clear_qwen4exp_batched_decode_workspace(workspace);
            return false;
        }
        for (int i = 0; i < N; ++i) {
            const int32_t sampled = argmax(logits[i]);
            if (first_divergence[i] < 0 &&
                (size_t)step < solo_logits[i].size()) {
                const float epsilon = max_delta(logits[i], solo_logits[i][step]);
                max_epsilon[i] = std::max(max_epsilon[i], epsilon);
                if (!std::isfinite(epsilon) ||
                    logits[i].size() != solo_logits[i][step].size() ||
                    std::memcmp(logits[i].data(), solo_logits[i][step].data(),
                                logits[i].size() * sizeof(float)) != 0)
                    first_divergence[i] = step;
            }
            batch_next[i] = sampled;
        }
    }
    clear_qwen4exp_batched_decode_workspace(workspace);
    bool ok = coherent_all;
    for (int i = 0; i < N; ++i) {
        const bool exact_tokens = engine_streams[i] == solo_streams[i];
        ok = ok && first_divergence[i] < 0 && exact_tokens;
        std::printf("[distinct] slot=%d max_epsilon=%.8g first_divergence_step=%d exact_tokens=%s\n",
            i, max_epsilon[i], first_divergence[i], exact_tokens ? "PASS" : "FAIL");
    }
    return ok;
}

// Compare every real engine step with independently advanced solo caches.
// Both paths must produce bit-exact indexer state and identical tokens.
// Prompts straddle 2052; the last prefill is mixed with decode, followed by N=1.
static bool run_qsa_boundary(Qwen4ExpSeqEngine & engine, ggml_backend_t backend,
                             const Qwen4ExpWeights & w,
                             const std::vector<Qwen4ExpCache *> & caches) {
    using namespace qwen4exp_test;
    constexpr int N = 4;
    const bool qsa = w.gfx1151;
    std::vector<std::vector<int32_t>> prompts(N);
    std::vector<int32_t> next(N);
    // Only these small contexts are needed by the reference streams.
    Qwen4ExpCache solo[N];
    for (int s = 0; s < N; ++s) {
        if (!create_qwen4exp_cache(backend, w, 3072, solo[s])) {
            for (auto & cache : solo) free_qwen4exp_cache(cache);
            return false;
        }
    }
    for (int s = 0; s < N; ++s) {
        engine.retire(s);
        prompts[s].resize(s < 2 ? 2050 : 2181);
        for (size_t i = 0; i < prompts[s].size(); ++i)
            prompts[s][i] = (int32_t) ((i * 7919 + 13 + s * 104729) % w.n_vocab);
        if (engine.admit(800 + s, prompts[s], SamplerCfg{}).slot != s) {
            for (int slot = 0; slot < N; ++slot) engine.retire(slot);
            for (auto & cache : solo) free_qwen4exp_cache(cache);
            return false;
        }
        // Missing writes must fail even when a cache is reused from an earlier test.
        if (qsa) poison_indexer(*caches[s]);
    }
    auto check_step = [&](const SeqEngine::StepPlan & plan) {
        if (!engine.reserve_decode(plan)) {
            std::fprintf(stderr, "[qsa-boundary] reserve_decode refused decode=%zu prefills=%zu\n",
                         plan.decode.size(), plan.prefills.size());
            return false;
        }
        std::vector<Qwen4ExpForwardSegment> spans;
        std::vector<int> slots;
        for (const auto & row : plan.decode) {
            slots.push_back(row.slot);
            spans.push_back({caches[row.slot], &row.token, 1, caches[row.slot]->cur_pos});
        }
        for (const auto & row : plan.prefills) {
            const int pos = caches[row.slot]->cur_pos;
            const int granule = engine.step_plan_limits(0).max_prefill_tokens_per_sequence;
            slots.push_back(row.slot);
            spans.push_back({caches[row.slot], prompts[row.slot].data() + pos,
                std::min({row.max_tokens, (int) prompts[row.slot].size() - pos, granule}), pos});
        }
        std::vector<SavedIndexer> expected;
        std::vector<int32_t> expected_tokens;
        for (size_t i = 0; i < spans.size(); ++i) {
            const auto & span = spans[i];
            std::vector<float> logits;
            if (!qwen4exp_forward(backend, w, solo[slots[i]], span.tokens,
                                  span.n_tokens, span.pos0, logits).ok) return false;
            expected_tokens.push_back(argmax(logits));
            expected.push_back(save_indexer(solo[slots[i]], qsa ? span.pos0 + span.n_tokens : 0));
        }
        const auto result = engine.step(plan);
        if (!result.ok() || result.decode.size() != plan.decode.size() ||
            result.prefills.size() != plan.prefills.size()) {
            std::fprintf(stderr, "[qsa-boundary] step failed: '%s' decode=%zu/%zu prefills=%zu/%zu\n",
                         result.error.c_str(), result.decode.size(), plan.decode.size(),
                         result.prefills.size(), plan.prefills.size());
            return false;
        }
        for (size_t i = 0; i < spans.size(); ++i) {
            const auto & span = spans[i];
            if (span.cache->cur_pos != span.pos0 + span.n_tokens ||
                (qsa && !equal_indexer(*span.cache, expected[i], span.cache->cur_pos))) {
                std::fprintf(stderr, "[qsa-boundary] slot=%d pos0=%d n=%d cur_pos=%d state mismatch\n",
                    slots[i], span.pos0, span.n_tokens, span.cache->cur_pos);
                return false;
            }
            const bool decode = i < plan.decode.size();
            if (!decode && result.prefills[i - plan.decode.size()].status !=
                    SeqEngine::PrefillOutput::Status::completed) continue;
            const int32_t token = decode ? result.decode[i].token :
                result.prefills[i - plan.decode.size()].token;
            // Enforce token identity here; the smoke probe checks full logits bit for bit.
            if (token != expected_tokens[i]) {
                std::fprintf(stderr, "[qsa-boundary] slot=%d pos=%d solo=%d engine=%d\n",
                    slots[i], span.pos0, expected_tokens[i], token);
                return false;
            }
            next[slots[i]] = token;
        }
        return true;
    };
    bool ok = true;
    // A slot past its prompt decodes; the rest prefill within the engine's per-step limits.
    auto prefilled = [&](int s) { return caches[s]->cur_pos >= (int) prompts[s].size(); };
    auto all_prefilled = [&] {
        for (int s = 0; s < N; ++s) if (!prefilled(s)) return false;
        return true;
    };
    for (int slice = 0; slice < 16 && ok && !all_prefilled(); ++slice) {
        SeqEngine::StepPlan plan;
        for (int s = 0; s < N; ++s)
            if (prefilled(s)) plan.decode.push_back({s, next[s]});
        const StepPlanLimits limits = engine.step_plan_limits((int) plan.decode.size());
        for (int s = 0; s < N && (int) plan.prefills.size() < limits.max_prefill_sequences; ++s) {
            if (prefilled(s)) continue;
            plan.prefills.push_back({s, slice == 4 ? 2 : limits.max_prefill_tokens_per_sequence});
        }
        ok = check_step(plan);
    }
    ok = ok && all_prefilled();
    for (int step = 0; step < 64 && ok; ++step) {
        SeqEngine::StepPlan plan;
        // Retire three slots after crossing the threshold and continue solo.
        if (step == 60) for (int s = 1; s < N; ++s) engine.retire(s);
        for (int s = 0; s < (step < 60 ? N : 1); ++s) plan.decode.push_back({s, next[s]});
        ok = check_step(plan);
    }
    for (int s = 0; s < N; ++s) engine.retire(s);
    for (auto & cache : solo) free_qwen4exp_cache(cache);
    return ok;
}

// Drive one slot through prompt and `steps` decode tokens the way the
// scheduler does; returns the sampled tokens (prompt's first, then decode).
static std::vector<int32_t> drive_slot(Qwen4ExpSeqEngine & engine, int slot, int steps,
                                       std::vector<SeqEngine::PrefillOutput> * prefills = nullptr) {
    std::vector<int32_t> out;
    for (int guard = 0; guard < 256 && out.empty(); ++guard) {
        SeqEngine::StepPlan plan;
        plan.prefills.push_back({slot, engine.step_plan_limits(0).max_prefill_tokens_per_sequence});
        const auto result = engine.step(plan);
        if (!result.ok() || result.prefills.size() != 1) return {};
        if (prefills) prefills->push_back(result.prefills[0]);
        if (result.prefills[0].status == SeqEngine::PrefillOutput::Status::completed)
            out.push_back(result.prefills[0].token);
    }
    while (!out.empty() && (int) out.size() <= steps) {
        SeqEngine::StepPlan plan;
        plan.decode.push_back({slot, out.back()});
        const auto result = engine.step(plan);
        if (!result.ok() || result.decode.size() != 1) return {};
        const auto & decoded = result.decode[0];   // an MTP slot alone also returns its accepted drafts
        out.insert(out.end(), decoded.committed_tokens.begin(), decoded.committed_tokens.end());
        out.push_back(decoded.token);
    }
    if ((int) out.size() > steps + 1) out.resize((size_t) steps + 1);
    return out;
}

// A checkpoint captured by one slot restores into another and continues
// token for token like the cold run that captured it, both ways between
// slot 0 (the widest verify, drafting alone) and another slot (a batched
// verify's rollback rows).
static bool run_prefix_store(Qwen4ExpSeqEngine & engine, const Qwen4ExpWeights & w) {
    constexpr int L = 3000, C = 2000, STEPS = 24;
    if (!engine.supports_prefix_store() || engine.estimate_prefix_store_bytes(C) == 0) return false;
    for (int s = 0; s < engine.slot_count(); ++s) engine.retire(s);
    std::vector<int32_t> prompt(L);
    for (int i = 0; i < L; ++i) prompt[(size_t) i] = (int32_t) ((i * 7919 + 101) % w.n_vocab);

    PrefixStorePlan capture_plan;
    capture_plan.capture = {1, {1, C}};
    const auto cold = engine.admit_with_prefix(900, prompt, SamplerCfg{}, capture_plan);
    if (cold.status != SeqEngine::AdmitResult::Status::admitted || cold.prefix_store.capture != capture_plan.capture)
        return false;
    std::vector<SeqEngine::PrefillOutput> prefills;
    const std::vector<int32_t> cold_tokens = drive_slot(engine, cold.slot, STEPS, &prefills);
    bool saved = false;
    for (const auto & p : prefills)
        saved = saved || (p.prefix_store.status == PrefixStoreEvent::Status::saved &&
                          p.prefix_store.ticket == capture_plan.capture && p.prefix_store.bytes > 0);
    engine.retire(cold.slot);
    if (cold_tokens.size() != STEPS + 1 || !saved) {
        std::fprintf(stderr, "[prefix-store] cold run tokens=%zu saved=%d\n", cold_tokens.size(), (int) saved);
        return false;
    }

    // Hold the capturing slot so the restore lands in another one.
    std::vector<int32_t> filler(64, 7);
    const auto blocker = engine.admit(901, filler, SamplerCfg{});
    PrefixStorePlan restore_plan;
    restore_plan.restore = {1, C};
    const auto hit = engine.admit_with_prefix(902, prompt, SamplerCfg{}, restore_plan);
    const bool cross_slot = blocker.slot == cold.slot && hit.slot != cold.slot;
    if (hit.status != SeqEngine::AdmitResult::Status::admitted || hit.prefix_store.restored != restore_plan.restore) {
        std::fprintf(stderr, "[prefix-store] restore was not accepted\n");
        return false;
    }
    const std::vector<int32_t> hit_tokens = drive_slot(engine, hit.slot, STEPS);
    engine.retire(hit.slot);
    engine.retire(blocker.slot);

    // A different prompt behind the same checkpoint id must not restore it.
    std::vector<int32_t> other = prompt;
    other[10] = (other[10] + 1) % w.n_vocab;
    const auto stale = engine.admit_with_prefix(903, other, SamplerCfg{}, restore_plan);
    const bool rejected = stale.status == SeqEngine::AdmitResult::Status::admitted &&
        stale.prefix_store.invalidated == restore_plan.restore && !stale.prefix_store.restored.valid();
    engine.retire(stale.slot);
    engine.discard_prefix_store({1, C});

    // And back: another slot captures, slot 0 restores and decodes.
    const auto holder = engine.admit(904, filler, SamplerCfg{});
    PrefixStorePlan plain_capture;
    plain_capture.capture = {2, {2, C}};
    const auto plain = engine.admit_with_prefix(905, prompt, SamplerCfg{}, plain_capture);
    std::vector<SeqEngine::PrefillOutput> plain_prefills;
    const std::vector<int32_t> plain_tokens = drive_slot(engine, plain.slot, STEPS, &plain_prefills);
    bool plain_saved = false;
    for (const auto & p : plain_prefills)
        plain_saved = plain_saved || (p.prefix_store.status == PrefixStoreEvent::Status::saved &&
                                      p.prefix_store.ticket == plain_capture.capture);
    engine.retire(plain.slot);
    engine.retire(holder.slot);
    PrefixStorePlan back_plan;
    back_plan.restore = {2, C};
    const auto back = engine.admit_with_prefix(906, prompt, SamplerCfg{}, back_plan);
    const bool back_restored = back.status == SeqEngine::AdmitResult::Status::admitted &&
        back.prefix_store.restored == back_plan.restore && back.slot == holder.slot && plain.slot != holder.slot;
    const std::vector<int32_t> back_tokens = back_restored ? drive_slot(engine, back.slot, STEPS) : std::vector<int32_t>{};
    engine.retire(back.slot);
    engine.discard_prefix_store({2, C});

    const bool same = hit_tokens == cold_tokens;
    const bool back_same = plain_saved && back_restored && plain_tokens == cold_tokens && back_tokens == cold_tokens;
    std::fprintf(stderr, "[prefix-store] cross_slot=%d identical=%d stale_rejected=%d back_to_slot0=%d\n",
                 (int) cross_slot, (int) same, (int) rejected, (int) back_same);
    return cross_slot && same && rejected && back_same;
}

// With the plan's restore points, a cold prompt's text is the same whether
// the cache captures it, restores it, or does neither.
static bool run_prefix_cuts(Qwen4ExpSeqEngine & engine, const Qwen4ExpWeights & w) {
    constexpr int L = 3000, STEPS = 24;
    const std::vector<int> cuts = {700, 2000};
    for (int s = 0; s < engine.slot_count(); ++s) engine.retire(s);
    std::vector<int32_t> prompt(L);
    for (int i = 0; i < L; ++i) prompt[(size_t) i] = (int32_t) ((i * 104729 + 7) % w.n_vocab);
    auto run = [&](uint64_t id, const PrefixStorePlan & plan, bool * restored = nullptr) {
        const auto admitted = engine.admit_with_prefix(id, prompt, SamplerCfg{}, plan);
        if (admitted.status != SeqEngine::AdmitResult::Status::admitted) return std::vector<int32_t>{};
        if (restored) *restored = admitted.prefix_store.restored == plan.restore;
        auto tokens = drive_slot(engine, admitted.slot, STEPS);
        engine.retire(admitted.slot);
        return tokens;
    };
    PrefixStorePlan none;
    none.restore_points = cuts;
    PrefixStorePlan capture = none;
    capture.capture = {2, {2, 2000}};
    PrefixStorePlan restore = none;
    restore.restore = {2, 2000};
    const auto cold = run(910, none);
    const auto captured = run(911, capture);
    bool restored = false;
    const auto resumed = run(912, restore, &restored);
    engine.discard_prefix_store({2, 2000});
    const bool ok = cold.size() == STEPS + 1 && cold == captured && restored && cold == resumed;
    std::fprintf(stderr, "[prefix-cuts] cold==captured=%d restored=%d cold==restored=%d\n",
                 (int) (cold == captured), (int) restored, (int) (cold == resumed));
    return ok;
}

// Whether a request may draft (the plan's allow_speculation, which the scheduler clears for a grammar or a thinking
// budget about to close): always, never, or in turns of four tokens of its stream.
enum class Drafts { always, never, turns };

// One request of a run: its prompt, sampler and token limit (the prompt's sampled token included).
struct Request {
    std::vector<int32_t> prompt;
    SamplerCfg sampler;
    int limit = 0;
    Drafts drafts = Drafts::always;
};

// A run of requests admitted together and driven as the scheduler drives the engine: every request's tokens (its
// prompt's, then each output's committed tokens and final token, in order) up to EOS or its limit, where it retires
// at once while the others go on.
struct RunResult {
    bool ok = true;
    std::vector<std::vector<int32_t>> streams;
    std::vector<int> slots;
    int verify_rows = 0;   // decode outputs of a step that verified drafts beside other decoders (or on a slot past 0)
    int plain_rows = 0;    // decode outputs of a step that decoded one token beside other decoders
    int cut_limit = 0;     // stops at the limit before an output's last token: inside a verified block
    int cut_eos = 0;       // stops at EOS after an output's accepted drafts
    int early_stops = 0;   // stops while another request still runs
    int mixed_steps = 0;   // verify steps with one-row slots (no drafts) beside drafting ones
    int lone_steps = 0;    // ... with exactly one drafting slot
    int switched = 0;      // requests with both a one-row and a drafting output in mixed steps
    int one_row_dense = 0; // one-row outputs of mixed steps before the first QSA position
    int one_row_qsa = 0;   // ... at or past it
};

// The first position a QSA layer attends sparsely (513 complete blocks of 4).
constexpr int kQsaFirst = 2051;

static RunResult run_requests(Qwen4ExpSeqEngine & engine, const std::vector<Request> & reqs) {
    RunResult r;
    const int n = (int) reqs.size();
    r.streams.assign((size_t) n, {});
    r.slots.assign((size_t) n, -1);
    std::vector<int32_t> next((size_t) n, -1);
    std::vector<int> state((size_t) n, 0);   // 0 prefilling, 1 decoding, 2 stopped
    for (int i = 0; i < n; ++i) {
        const auto admitted = engine.admit(7000 + (uint64_t) i, reqs[(size_t) i].prompt, reqs[(size_t) i].sampler);
        if (admitted.status != SeqEngine::AdmitResult::Status::admitted) {
            r.ok = false;
            return r;
        }
        r.slots[(size_t) i] = admitted.slot;
    }
    auto request_at = [&](int slot) {
        for (int i = 0; i < n; ++i) if (r.slots[(size_t) i] == slot && state[(size_t) i] != 2) return i;
        return -1;
    };
    auto take = [&](int i, const std::vector<int32_t> & tokens) {
        if (i < 0) {
            r.ok = false;
            return;
        }
        std::vector<int32_t> & stream = r.streams[(size_t) i];
        for (size_t k = 0; k < tokens.size(); ++k) {
            stream.push_back(tokens[k]);
            const bool eos = engine.token_is_eos(tokens[k]);
            if (!eos && (int) stream.size() < reqs[(size_t) i].limit) continue;
            r.cut_limit += !eos && k + 1 < tokens.size();
            r.cut_eos += eos && k > 0;
            engine.retire(r.slots[(size_t) i]);
            state[(size_t) i] = 2;
            for (int j = 0; j < n; ++j) {
                if (state[(size_t) j] != 2) {
                    ++r.early_stops;
                    break;
                }
            }
            return;
        }
        next[(size_t) i] = tokens.back();
        state[(size_t) i] = 1;
    };
    auto may_draft = [&](int i) {
        const Drafts d = reqs[(size_t) i].drafts;
        return d == Drafts::always || (d == Drafts::turns && r.streams[(size_t) i].size() / 4 % 2 == 0);
    };
    std::vector<int> as_one((size_t) n, 0), as_drafting((size_t) n, 0);
    for (int guard = 0; guard < 4096; ++guard) {
        SeqEngine::StepPlan plan;
        for (int i = 0; i < n; ++i)
            if (state[(size_t) i] == 1) plan.decode.push_back({r.slots[(size_t) i], next[(size_t) i], may_draft(i)});
        const StepPlanLimits limits = engine.step_plan_limits((int) plan.decode.size());
        for (int i = 0; i < n && (int) plan.prefills.size() < limits.max_prefill_sequences; ++i)
            if (state[(size_t) i] == 0) plan.prefills.push_back({r.slots[(size_t) i], limits.max_prefill_tokens_per_sequence});
        if (plan.decode.empty() && plan.prefills.empty()) return r;
        if (!engine.reserve_decode(plan)) break;
        const auto result = engine.step(plan);
        if (!result.ok() || result.decode.size() != plan.decode.size() ||
            result.prefills.size() != plan.prefills.size()) {
            std::fprintf(stderr, "[alone-vs-together] step failed: '%s'\n", result.error.c_str());
            break;
        }
        const bool beside = plan.decode.size() > 1 || (plan.decode.size() == 1 && plan.decode[0].slot != 0);
        const int drafting = (int) std::count_if(result.decode.begin(), result.decode.end(),
            [](const SeqEngine::DecodeOutput & o) { return o.spec_offered > 0; });
        const bool mixed = drafting > 0 && drafting < (int) result.decode.size();
        r.mixed_steps += mixed;
        r.lone_steps += mixed && drafting == 1;
        for (const auto & output : result.decode) {
            if (beside) ++(output.spec_offered > 0 ? r.verify_rows : r.plain_rows);
            const int i = request_at(output.slot);
            if (mixed && i >= 0) {
                ++(output.spec_offered > 0 ? as_drafting : as_one)[(size_t) i];
                // The position this step decoded the request's token at.
                const int pos = (int) (reqs[(size_t) i].prompt.size() + r.streams[(size_t) i].size()) - 1;
                if (output.spec_offered == 0) ++(pos < kQsaFirst ? r.one_row_dense : r.one_row_qsa);
            }
            std::vector<int32_t> tokens = output.committed_tokens;
            tokens.push_back(output.token);
            take(request_at(output.slot), tokens);
        }
        for (const auto & output : result.prefills) {
            if (output.status == SeqEngine::PrefillOutput::Status::completed) take(request_at(output.slot), {output.token});
        }
        if (!r.ok) break;
        r.switched = 0;
        for (int i = 0; i < n; ++i) r.switched += as_one[(size_t) i] > 0 && as_drafting[(size_t) i] > 0;
    }
    r.ok = false;
    for (int i = 0; i < n; ++i) if (state[(size_t) i] != 2) engine.retire(r.slots[(size_t) i]);
    return r;
}

// Each request of a concurrent run (three greedy, one seeded sampled) must produce its alone stream token for token,
// with the batched verify forced on and off. A request alone lands in slot 0 and drafts there like the single-slot
// loop (prompt lookup included). Together, slot 0's request stops first (its limit), so the others go on three, then
// two at once, and the longest ends alone on a slot past 0 (a one-slot batched verify). One request stops at EOS:
// eos_chat_id is set to a token its alone stream reaches mid-reply (the engine reads EOS from the weights, on the host
// only; `eos` keeps the pick across draft caps). The limits and that EOS must cut a verified block somewhere.
// The same run where some requests may not draft (Drafts): a one-row slot beside drafting ones, one that switches in
// turns, the sampled request as a one-row slot beside exactly one drafting slot, and none drafting (the plain step,
// whatever the mode); and four requests whose replies cross the QSA boundary, the sampled one as a one-row slot
// crossing it beside drafting ones.
static bool run_alone_vs_together(ggml_backend_t backend, Qwen4ExpWeights & w, const std::vector<Qwen4ExpCache *> & caches,
                                  int ctx, int verify_width, const char * model_path, int32_t & eos) {
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model_path)) return false;
    const char * texts[] = {
        "Write a short paragraph about the lighthouse of Alexandria.\n",
        "List five facts about the planet Mars, one per line.\n",
        "Explain in plain words how a refrigerator keeps food cold.\n",
        "Describe a quiet morning in a mountain village.\n",
    };
    const int limits[] = {14, 33, 40, 47};
    std::vector<Request> reqs(4);
    for (int i = 0; i < 4; ++i) {
        reqs[(size_t) i].prompt = tokenizer.encode(texts[i]);
        reqs[(size_t) i].limit = limits[i];
        if (reqs[(size_t) i].prompt.empty()) return false;
    }
    SamplerCfg & sampled = reqs[1].sampler;
    sampled.temp = 0.8f;
    sampled.top_k = 40;
    sampled.top_p = 0.95f;
    sampled.seed = 20261110;
    sampled.has_seed = true;

    PromptLookupPolicy lookup(QWEN4EXP_MTP_MAX_VERIFY);
    Qwen4ExpSeqEngine engine(backend, w, caches, ctx, 512, 0, verify_width, nullptr, nullptr, nullptr,
                             verify_width == 0 ? &lookup : nullptr);
    for (int s = 0; s < engine.slot_count(); ++s) engine.retire(s);
    const int32_t chat_eos = w.eos_chat_id;
    if (eos < 0) {
        // Request 2's greedy reply, long enough: its first token past the eighth that it has not produced before.
        Request probe = reqs[2];
        probe.limit = 64;
        const RunResult alone = run_requests(engine, {probe});
        const std::vector<int32_t> t = alone.ok ? alone.streams[0] : std::vector<int32_t>{};
        for (size_t k = 8; eos < 0 && k + 4 < t.size(); ++k) {
            if (!engine.token_is_eos(t[k]) && std::find(t.begin(), t.begin() + (std::ptrdiff_t) k, t[k]) == t.begin() + (std::ptrdiff_t) k)
                eos = t[k];
        }
        if (eos < 0) {
            std::fprintf(stderr, "[alone-vs-together] no EOS stand-in in request 2's reply\n");
            return false;
        }
    }
    w.eos_chat_id = eos;
    const int caps = caches[0]->mtp_draft;
    bool ok = true;
    auto alone_streams = [&](const std::vector<Request> & set) {
        std::vector<std::vector<int32_t>> streams(set.size());
        for (size_t i = 0; i < set.size() && ok; ++i) {
            const RunResult r = run_requests(engine, {set[i]});
            ok = r.ok && r.slots[0] == 0;
            if (!ok) std::fprintf(stderr, "[alone-vs-together] request %zu alone failed (slot %d)\n", i, r.slots[0]);
            streams[i] = r.ok ? r.streams[0] : std::vector<int32_t>{};
        }
        return streams;
    };
    // `set` together with each request's Drafts, in both modes, against its alone streams; `covered` judges the
    // verify mode's run (the plain mode's must verify nothing).
    auto together = [&](const char * name, std::vector<Request> set, const std::vector<Drafts> & drafts,
                        const std::vector<std::vector<int32_t>> & alone, auto && covered) {
        for (size_t i = 0; i < set.size(); ++i) set[i].drafts = drafts[i];
        for (const auto mode : {Qwen4ExpSeqEngine::DecodeMode::verify, Qwen4ExpSeqEngine::DecodeMode::plain}) {
            if (!ok) break;
            const bool verify = mode == Qwen4ExpSeqEngine::DecodeMode::verify;
            engine.set_decode_mode(mode);
            const RunResult r = run_requests(engine, set);
            ok = r.ok;
            for (size_t i = 0; ok && i < set.size(); ++i) {
                const bool same = r.streams[i] == alone[i];
                size_t first = 0;
                while (first < r.streams[i].size() && first < alone[i].size() && r.streams[i][first] == alone[i][first])
                    ++first;
                std::printf("[alone-vs-together] %s draft=%d/%d mode=%s request=%zu slot=%d %s drafts=%s tokens=%zu "
                            "alone=%zu %s", name, caps, caches[1]->mtp_draft, verify ? "verify" : "plain", i,
                            r.slots[i], set[i].sampler.temp > 0.0f ? "sampled" : "greedy",
                            drafts[i] == Drafts::always ? "always" : drafts[i] == Drafts::never ? "never" : "turns",
                            r.streams[i].size(), alone[i].size(), same ? "identical\n" : "DIFFER");
                if (!same) std::printf(" at token %zu\n", first);
                ok = ok && same;
            }
            const bool run_covered = verify ? covered(r) : r.verify_rows == 0 && r.plain_rows > 0;
            std::printf("[alone-vs-together] %s draft=%d/%d mode=%s verify_rows=%d plain_rows=%d cut_limit=%d "
                        "cut_eos=%d early_stops=%d mixed_steps=%d lone_steps=%d switched=%d one_row_dense=%d "
                        "one_row_qsa=%d covered=%d\n", name, caps, caches[1]->mtp_draft, verify ? "verify" : "plain",
                        r.verify_rows, r.plain_rows, r.cut_limit, r.cut_eos, r.early_stops, r.mixed_steps,
                        r.lone_steps, r.switched, r.one_row_dense, r.one_row_qsa, (int) run_covered);
            ok = ok && run_covered;
        }
    };
    const std::vector<std::vector<int32_t>> alone = alone_streams(reqs);
    const Drafts A = Drafts::always, N = Drafts::never, T = Drafts::turns;
    // Every request drafts: verified blocks beside other decoders, a request stopping while others go on, and a stop
    // inside a verified block.
    together("all", reqs, {A, A, A, A}, alone, [](const RunResult & r) {
        return r.early_stops > 0 && r.verify_rows > 0 && r.cut_limit + r.cut_eos > 0;
    });
    // Slot 0's request never drafts and one switches in turns: one-row slots beside drafting ones, and a request
    // that is both.
    together("mixed", reqs, {N, A, T, A}, alone, [](const RunResult & r) {
        return r.early_stops > 0 && r.mixed_steps > 0 && r.switched > 0;
    });
    // Only the longest request drafts: exactly one drafting slot beside one-row slots, the sampled one among them.
    together("one", reqs, {N, N, N, A}, alone, [](const RunResult & r) { return r.lone_steps > 0; });
    // No request drafts: no verify step at all, in either mode (each step is the plain one).
    together("none", reqs, {N, N, N, N}, alone, [](const RunResult & r) {
        return r.verify_rows == 0 && r.plain_rows > 0 && r.mixed_steps == 0;
    });
    // Replies that start just before the QSA boundary: the sampled request (from 2037) as a one-row slot that crosses
    // it beside drafting ones, the longest (from 2051, its limit raised to outlive the sampled one) drafting QSA rows
    // beside it.
    std::vector<Request> qsa = reqs;
    for (size_t i = 0; i < qsa.size(); ++i) {
        qsa[i].prompt.resize(2030 + 7 * i);
        for (size_t t = 0; t < qsa[i].prompt.size(); ++t)
            qsa[i].prompt[t] = (int32_t) ((t * 7919 + 13 + i * 104729) % (size_t) w.n_vocab);
    }
    qsa[3].limit = 96;
    const std::vector<std::vector<int32_t>> qsa_alone = alone_streams(qsa);
    together("qsa", qsa, {A, N, A, A}, qsa_alone, [](const RunResult & r) {
        return r.mixed_steps > 0 && r.one_row_dense > 0 && r.one_row_qsa > 0;
    });
    engine.set_decode_mode(Qwen4ExpSeqEngine::DecodeMode::measured);
    w.eos_chat_id = chat_eos;
    return ok;
}

static bool run_soak(Qwen4ExpSeqEngine & engine, int n, int round) {
    std::mt19937 rng((uint32_t)(0x51e9 + n * 101 + round));
    std::vector<int32_t> over_ctx((size_t)engine.max_context() + 1, 1);
    const auto rejected = engine.admit((uint64_t)(round * 1000 + n),
                                       over_ctx, SamplerCfg{});
    if (rejected.status != SeqEngine::AdmitResult::Status::capacity_exceeded) {
        std::fprintf(stderr, "[soak N=%d] over-context admission was not rejected\n", n);
        return false;
    }
    std::vector<int> admitted;
    for (int i = 0; i < n; ++i) {
        const int length = std::uniform_int_distribution<int>(1024, 30000)(rng);
        std::vector<int32_t> prompt((size_t)length);
        for (int32_t & token : prompt)
            token = (int32_t)std::uniform_int_distribution<int>(0, 4095)(rng);
        const auto result = engine.admit((uint64_t)(round * 16 + i + 1), prompt, SamplerCfg{});
        if (result.status != SeqEngine::AdmitResult::Status::admitted) return false;
        admitted.push_back(result.slot);
    }
    // Advance the current FIFO owner by one slice, then cancel it. Repeating
    // this exercises owner transfer, partial-prefill reset, and slot reuse.
    for (int slot : admitted) {
        SeqEngine::StepPlan plan;
        plan.prefills.push_back({slot, 512});
        const auto result = engine.step(plan);
        if (!result.ok() || result.prefills.size() != 1) return false;
        engine.retire(slot);
    }
    for (int slot : admitted) engine.retire(slot);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <iq4nl-shard1.gguf> [ctx=32768]\n", argv[0]);
        return 2;
    }
    const int ctx = argc > 2 ? std::atoi(argv[2]) : 32768;
    if (ctx != 32768) return 2;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 77;
    Qwen4ExpWeights weights;
    if (!load_qwen4exp_gguf(argv[1], backend, weights)) {
        ggml_backend_free(backend);
        return 1;
    }
    std::vector<Qwen4ExpCache> caches(4);
    for (auto & cache : caches) {
        if (!create_qwen4exp_cache(backend, weights, ctx, cache)) {
            for (auto & c : caches) free_qwen4exp_cache(c);
            free_qwen4exp_weights(weights);
            ggml_backend_free(backend);
            return 1;
        }
    }
    std::vector<Qwen4ExpCache *> ptrs;
    for (auto & cache : caches) ptrs.push_back(&cache);

    bool ok = true;
    for (int n : {2, 3, 4}) {
        Qwen4ExpSeqEngine engine(backend, weights,
            std::vector<Qwen4ExpCache *>(ptrs.begin(), ptrs.begin() + n), ctx);
        const auto violations = check_seq_engine_contract(engine);
        for (const std::string & violation : violations)
            std::fprintf(stderr, "[contract N=%d] %s\n", n, violation.c_str());
        ok = ok && violations.empty();
        bool soak_ok = true;
        for (int round = 0; round < 2; ++round) {
            if (!run_soak(engine, n, round)) {
                std::fprintf(stderr, "[soak N=%d round=%d] failed\n", n, round);
                soak_ok = false;
                break;
            }
        }
        ok = ok && soak_ok;
        std::printf("[qwen4exp-seq] N=%d contract=%s soak=%s\n", n,
                    violations.empty() ? "PASS" : "FAIL", soak_ok ? "PASS" : "FAIL");
    }
    {
        Qwen4ExpSeqEngine engine(backend, weights, ptrs, ctx);
        const bool boundary_ok = run_qsa_boundary(engine, backend, weights, ptrs);
        std::printf("[qwen4exp-seq] qsa-boundary=%s\n", boundary_ok ? "PASS" : "FAIL");
        ok = ok && boundary_ok;
        const bool ok_distinct = run_distinct(
            engine, backend, weights, ptrs, argv[1]);
        std::printf("[qwen4exp-seq] distinct-concurrent=%s\n",
                    ok_distinct ? "PASS" : "FAIL");
        ok = ok && ok_distinct;
    }
    for (auto & cache : caches) free_qwen4exp_cache(cache);
    // Every slot as the server makes it with MTP (start_seq_engine): slot 0 with the draft cap of the verify width
    // (adaptive: QWEN4EXP_MTP_MAX_DRAFT), the others with a batched verify's rollback rows; for the adaptive width and
    // the fixed widths 3 and 2, whose caps bound the batched verify's rows.
    int32_t eos = -1;
    for (const int verify_width : {0, 3, 2}) {
        const int cap = verify_width == 0 ? QWEN4EXP_MTP_MAX_DRAFT : verify_width - 1;
        std::vector<Qwen4ExpCache> served_caches(4);
        std::vector<Qwen4ExpCache *> served;
        bool made = true;
        for (size_t s = 0; s < served_caches.size(); ++s) {
            made = made && create_qwen4exp_cache(backend, weights, ctx, served_caches[s], /*mtp=*/true,
                                                 s == 0 ? cap : std::min(kQwen4ExpBatchedVerifyRows, cap + 1) - 1);
            served.push_back(&served_caches[s]);
        }
        if (!made) {
            std::fprintf(stderr, "[qwen4exp-seq] MTP cache creation failed\n");
            ok = false;
        } else if (verify_width == 0) {
            // Checkpoints in the device's memory, and in system memory as beside a discrete GPU.
            const ggml_backend_t system = create_snapshot_backend(backend);
            for (ggml_backend_t store : {backend, system}) {
                Qwen4ExpSeqEngine engine(backend, weights, served, ctx, 512, size_t(8) << 30, verify_width,
                                         nullptr, nullptr, store);
                const char * memory = store == backend ? "device" : "system";
                const bool prefix_ok = run_prefix_store(engine, weights);
                std::printf("[qwen4exp-seq] prefix-store(%s)=%s\n", memory, prefix_ok ? "PASS" : "FAIL");
                const bool cuts_ok = run_prefix_cuts(engine, weights);
                std::printf("[qwen4exp-seq] prefix-cuts(%s)=%s\n", memory, cuts_ok ? "PASS" : "FAIL");
                ok = ok && prefix_ok && cuts_ok;
                if (system == backend) break;
            }
            free_snapshot_backend(system, backend);
        }
        if (made) {
            const bool together_ok = run_alone_vs_together(backend, weights, served, ctx, verify_width, argv[1], eos);
            std::printf("[qwen4exp-seq] alone-vs-together(verify-width %d)=%s\n", verify_width,
                        together_ok ? "PASS" : "FAIL");
            ok = ok && together_ok;
        }
        for (auto & cache : served_caches) free_qwen4exp_cache(cache);
    }
    free_qwen4exp_weights(weights);
    ggml_backend_free(backend);
    std::printf("[qwen4exp-seq] all=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
