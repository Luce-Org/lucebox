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
        if (!engine.reserve_decode(plan)) return false;
        std::vector<Qwen4ExpForwardSegment> spans;
        std::vector<int> slots;
        for (const auto & row : plan.decode) {
            slots.push_back(row.slot);
            spans.push_back({caches[row.slot], &row.token, 1, caches[row.slot]->cur_pos});
        }
        for (const auto & row : plan.prefills) {
            const int pos = caches[row.slot]->cur_pos;
            slots.push_back(row.slot);
            spans.push_back({caches[row.slot], prompts[row.slot].data() + pos,
                std::min(row.max_tokens, (int) prompts[row.slot].size() - pos), pos});
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
            result.prefills.size() != plan.prefills.size()) return false;
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
    for (int slice = 0; slice < 6 && ok; ++slice) {
        SeqEngine::StepPlan plan;
        for (int s = 0; s < N; ++s) {
            if (caches[s]->cur_pos == (int) prompts[s].size()) plan.decode.push_back({s, next[s]});
            else plan.prefills.push_back({s, slice == 4 ? 2 : 512});
        }
        ok = check_step(plan);
    }
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
    free_qwen4exp_weights(weights);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
