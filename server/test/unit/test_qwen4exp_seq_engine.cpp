// GPU adapter contract and lifecycle soak for the full-cache engine.
// Usage: test_qwen4exp_seq_engine <iq4nl-shard1.gguf> [ctx=32768]
#include "qwen4exp_seq_engine.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_internal.h"
#include "common/concurrency/seq_engine.h"
#include "seq_engine_contract.h"
#include "common/sampler.h"
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
// token for token like the cold run that captured it, both ways between the
// MTP slot (0, with the draft layer) and a slot without it.
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

    // And back: a slot without the draft layer captures, slot 0 restores and decodes without drafts.
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
        // Slot 0 as the server makes it: the MTP draft layer and its verify state, drafting when alone.
        Qwen4ExpCache mtp_cache;
        const bool mtp_ok = create_qwen4exp_cache(backend, weights, ctx, mtp_cache, /*mtp=*/true, QWEN4EXP_MTP_MAX_DRAFT);
        if (mtp_ok) {
            std::vector<Qwen4ExpCache *> served = ptrs;
            served[0] = &mtp_cache;
            Qwen4ExpSeqEngine engine(backend, weights, served, ctx, 512, size_t(8) << 30, /*verify_width=*/0);
            const bool prefix_ok = run_prefix_store(engine, weights);
            std::printf("[qwen4exp-seq] prefix-store=%s\n", prefix_ok ? "PASS" : "FAIL");
            const bool cuts_ok = run_prefix_cuts(engine, weights);
            std::printf("[qwen4exp-seq] prefix-cuts=%s\n", cuts_ok ? "PASS" : "FAIL");
            ok = ok && prefix_ok && cuts_ok;
        } else {
            std::fprintf(stderr, "[qwen4exp-seq] MTP cache creation failed\n");
            ok = false;
        }
        free_qwen4exp_cache(mtp_cache);
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
    free_qwen4exp_weights(weights);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
