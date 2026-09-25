// GPU adapter contract and lifecycle soak for the experimental full-cache engine.
// Usage: test_qwen4exp_seq_engine <iq4nl-shard1.gguf> [ctx=32768]
#include "qwen4exp_seq_engine.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_internal.h"
#include "common/concurrency/seq_engine.h"
#include "seq_engine_contract.h"
#include "common/sampler.h"
#include "server/tokenizer.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace luce::common;

static int argmax(const std::vector<float> & logits) {
    return (int)std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end()));
}

static float top2_margin(const std::vector<float> & logits) {
    float first = -INFINITY, second = -INFINITY;
    for (float value : logits) {
        if (value > first) { second = first; first = value; }
        else if (value > second) second = value;
    }
    return first - second;
}

static float max_delta(const std::vector<float> & a,
                       const std::vector<float> & b) {
    float delta = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        delta = std::max(delta, std::abs(a[i] - b[i]));
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

    // Compare full-vocabulary logits while histories still share a prefix.
    // One reusable solo cache keeps the peak at five full caches, not eight.
    Qwen4ExpCache solo;
    if (!create_qwen4exp_cache(backend, weights, 32768, GGML_TYPE_F16, solo))
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
        for (int step = 0; step < STEPS; ++step) {
            const int32_t fed = solo_streams[i].back();
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
    std::vector<float> first_epsilon(N, 0.0f), first_margin(N, 0.0f);
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
                if (sampled != solo_streams[i][(size_t)step + 1]) {
                    first_divergence[i] = step;
                    first_epsilon[i] = epsilon;
                    first_margin[i] = top2_margin(solo_logits[i][step]);
                }
            }
            batch_next[i] = sampled;
        }
    }
    clear_qwen4exp_batched_decode_workspace(workspace);
    const bool ok = coherent_all;
    for (int i = 0; i < N; ++i) {
        std::printf("[distinct-margin] slot=%d max_epsilon=%.8g first_divergence_step=%d epsilon=%.8g solo_margin=%.8g 2epsilon=%.8g\n",
            i, max_epsilon[i], first_divergence[i], first_epsilon[i],
            first_margin[i], 2.0f * first_epsilon[i]);
    }
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
    setenv("QWEN4EXP_BATCHED_DECODE", "1", 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 77;
    Qwen4ExpWeights weights;
    if (!load_qwen4exp_gguf(argv[1], backend, weights)) {
        ggml_backend_free(backend);
        return 1;
    }
    std::vector<Qwen4ExpCache> caches(4);
    for (auto & cache : caches) {
        if (!create_qwen4exp_cache(backend, weights, ctx, GGML_TYPE_F16, cache)) {
            for (auto & c : caches) free_qwen4exp_cache(c);
            free_qwen4exp_weights(weights);
            ggml_backend_free(backend);
            return 1;
        }
    }
    std::vector<Qwen4ExpCache *> ptrs;
    for (auto & cache : caches) ptrs.push_back(&cache);

    bool ok = true;
    for (int n : {2, 4}) {
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
