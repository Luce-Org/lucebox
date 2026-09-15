// Optional full-model regression for Strix Halo. Requires the matching 0731
// mixed-weight target; no drafter is loaded. Not registered in ordinary CI.
// DFLASH_DS4_SPEC=0 ROCR_VISIBLE_DEVICES=1 \
//   test_ds4_paged_prefill_model target.gguf
#include "deepseek4/deepseek4_backend.h"
#include "seq_engine_contract.h"
#include "server/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace dflash::common;

static std::vector<std::vector<int32_t>> generate(
        SeqEngine & engine, const std::vector<std::vector<int32_t>> & prompts,
        int count) {
    std::vector<int> slots, remaining;
    std::vector<int32_t> pending(prompts.size(), -1);
    std::vector<std::vector<int32_t>> tokens(prompts.size());
    struct Retire {
        SeqEngine & engine;
        std::vector<int> & slots;
        ~Retire() { for (int slot : slots) engine.retire(slot); }
    } retire{engine, slots};
    for (size_t i = 0; i < prompts.size(); ++i) {
        const auto a = engine.admit(100+i, prompts[i], SamplerCfg{});
        if (a.status != SeqEngine::AdmitResult::Status::admitted)
            throw std::runtime_error("generation admit failed: " + a.error);
        slots.push_back(a.slot);
        remaining.push_back((int)prompts[i].size());
    }
    for (int iteration = 0; iteration < 1024; ++iteration) {
        SeqEngine::StepPlan plan;
        bool done = true;
        for (size_t i = 0; i < slots.size(); ++i) {
            if ((int)tokens[i].size() < count) done = false;
            // Keep the cohort fixed through the shorter clients' tail.
            if (!remaining[i]) plan.decode.push_back({slots[i], pending[i], false});
        }
        if (done) {
            return tokens;
        }
        const auto limits = engine.step_plan_limits((int)plan.decode.size());
        int budget = limits.max_prefill_tokens_total;
        for (size_t i = 0; i < slots.size() && budget > 0; ++i) {
            if (!remaining[i]) continue;
            const int n = std::min({remaining[i], budget, limits.max_prefill_tokens_per_sequence});
            plan.prefills.push_back({slots[i], n});
            budget -= n;
        }
        const auto result = engine.step(plan);
        const auto error = validate_step_result(plan, result, engine.slot_count());
        if (!result.ok() || !error.empty()) throw std::runtime_error(result.error + error);
        for (const auto & out : result.prefills) {
            const auto i = std::find(slots.begin(), slots.end(), out.slot)-slots.begin();
            if (out.status == SeqEngine::PrefillOutput::Status::failed)
                throw std::runtime_error(out.error);
            const auto slice = std::find_if(plan.prefills.begin(), plan.prefills.end(),
                [&](const PrefillSlice & s) { return s.slot == out.slot; });
            remaining[i] -= slice->max_tokens;
            if (out.status == SeqEngine::PrefillOutput::Status::completed) {
                pending[i] = out.token;
                tokens[i].push_back(out.token);
            }
        }
        for (const auto & out : result.decode) {
            if (out.failed) throw std::runtime_error(out.error);
            const auto i = std::find(slots.begin(), slots.end(), out.slot)-slots.begin();
            tokens[i].insert(tokens[i].end(), out.committed_tokens.begin(), out.committed_tokens.end());
            tokens[i].push_back(out.token);
            pending[i] = out.token;
        }
    }
    throw std::runtime_error("generation did not finish");
}

// Rebuild the exact generated prefix in chronological prefill chunks, then
// check one ordinary decode step. This isolates target batch-shape arithmetic
// from draft proposals, acceptance, and speculative rollback.
static void check_prefill_parity(SeqEngine & engine,
        const std::vector<std::vector<int32_t>> & prompts,
        const std::vector<std::vector<int32_t>> & ar) {
    for (int index : {53, 129}) {
        std::vector<int> slots, remaining;
        struct Retire {
            SeqEngine & engine;
            std::vector<int> & slots;
            ~Retire() { for (int slot : slots) engine.retire(slot); }
        } retire{engine, slots};
        for (size_t i = 0; i < prompts.size(); ++i) {
            auto prefix = prompts[i];
            prefix.insert(prefix.end(), ar[i].begin(), ar[i].begin()+index-1);
            const auto a = engine.admit(500+i, prefix, SamplerCfg{});
            if (a.status != SeqEngine::AdmitResult::Status::admitted)
                throw std::runtime_error("prefix reconstruction admit failed: " + a.error);
            slots.push_back(a.slot);
            remaining.push_back((int)prefix.size());
        }
        bool ready = false;
        while (!ready) {
            ready = std::all_of(remaining.begin(), remaining.end(), [](int n) { return n <= 4; });
            SeqEngine::StepPlan plan;
            for (size_t i = 0; i < slots.size(); ++i) {
                const int n = ready ? remaining[i] : std::min(4, remaining[i]-1);
                if (n > 0) { plan.prefills.push_back({slots[i], n}); remaining[i] -= n; }
            }
            const auto result = engine.step(plan);
            const auto error = validate_step_result(plan, result, engine.slot_count());
            if (!result.ok() || !error.empty()) throw std::runtime_error(result.error + error);
            for (const auto & out : result.prefills)
                if (out.status == SeqEngine::PrefillOutput::Status::failed)
                    throw std::runtime_error(out.error);
        }
        SeqEngine::StepPlan plan;
        for (size_t i = 0; i < slots.size(); ++i)
            plan.decode.push_back({slots[i], ar[i][index-1], false});
        const auto result = engine.step(plan);
        const auto error = validate_step_result(plan, result, engine.slot_count());
        if (!result.ok() || !error.empty()) throw std::runtime_error(result.error + error);
        for (const auto & out : result.decode) {
            if (out.failed) throw std::runtime_error(out.error);
            const auto i = std::find(slots.begin(), slots.end(), out.slot)-slots.begin();
            if (!out.committed_tokens.empty() || out.token != ar[i][index]) {
                std::fprintf(stderr, "FAIL: prefill parity client=%zu token=%d expected=%d actual=%d\n",
                             (size_t)i, index, ar[i][index], out.token);
                throw std::runtime_error("prefill changed the next token for the same prefix");
            }
        }
        std::fprintf(stderr, "PASS: prefill parity C%zu at token %d\n", prompts.size(), index);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2 || (std::getenv("DFLASH_DS4_SPEC") &&
                     std::strcmp(std::getenv("DFLASH_DS4_SPEC"), "0") != 0)) {
        std::fprintf(stderr, "Usage: DFLASH_DS4_SPEC=0 test_ds4_paged_prefill_model target.gguf\n");
        return 2;
    }
    DeepSeek4BackendConfig config;
    config.model_path = argv[1];
    config.device.backend = PlacementBackend::Hip;
    config.device.gpu = 0;
    config.paged_attention = true;
    config.max_concurrency = 4;
    config.max_ctx = 512;
    config.kv_pool_tokens = 2048;
    DeepSeek4Backend backend(config);
    if (!backend.init() || !backend.seq_engine()) return 1;
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(argv[1])) return 1;
    const char * tasks[] = {
        "Write Python code for a thread-safe LRU cache. Output code only.",
        "Write Python code for an asyncio work queue. Output code only.",
        "Write Python code for a token bucket limiter. Output code only.",
        "Write Python code for a retry decorator. Output code only."
    };
    try {
        std::vector<std::vector<int32_t>> prompts;
        for (const auto * task : tasks) prompts.push_back(tokenizer.encode(
            std::string("[bos][user]")+task+"[assistant]</think>"));
        const auto ar = generate(*backend.seq_engine(), prompts, 140);
        check_prefill_parity(*backend.seq_engine(), prompts, ar);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
