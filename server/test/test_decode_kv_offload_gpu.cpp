// Model-level proof: preserve B while A grows into its returned pages, then
// compare B's resumed tokens with an uninterrupted control. CPU copy tests
// cannot establish recurrent/compressor/draft isolation on the real graphs.
#include "CppUnitTestFramework.hpp"
#include "model_test_paths.h"
#include "common/backend_factory.h"
#include "common/concurrency/seq_engine.h"
#include "server/tokenizer.h"
#include "ggml-cuda.h"

#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
using namespace dflash::common;
struct DecodeKvOffloadGpuFixture {};
#define GPU_OFFLOAD_CHECK(x) do { if (!(x)) throw std::runtime_error( \
    std::string("GPU KV offload line ") + std::to_string(__LINE__) + ": " #x); } while (false)

struct Transcript {
    std::vector<int32_t> tokens;
    int speculative_children = 0;
};

Transcript continue_request(SeqEngine & engine, const std::vector<int32_t> & prompt,
                            bool offload, bool speculative) {
    struct Retire {
        SeqEngine & engine;
        ~Retire() { for (int i = 0; i < engine.slot_count(); ++i) engine.retire(i); }
    } cleanup{engine};
    SamplerCfg sampler;
    sampler.temp = speculative ? 0.0f : 0.7f;
    sampler.seed = 12345;
    const auto a = engine.admit(1, prompt, sampler);
    const auto b = engine.admit(2, prompt, sampler);
    GPU_OFFLOAD_CHECK(a.status == SeqEngine::AdmitResult::Status::admitted);
    GPU_OFFLOAD_CHECK(b.status == SeqEngine::AdmitResult::Status::admitted);
    GPU_OFFLOAD_CHECK(a.slot == 0 && b.slot == 1);
    std::array<int32_t, 2> pending{-1, -1};
    std::array<int, 2> positions{(int)prompt.size(), (int)prompt.size()};
    std::array<bool, 2> prefill{true, true};
    Transcript transcript;
    auto run = [&](const SeqEngine::StepPlan & plan) {
        GPU_OFFLOAD_CHECK(engine.reserve_decode(plan));
        const auto result = engine.step(plan);
        GPU_OFFLOAD_CHECK(result.ok());
        GPU_OFFLOAD_CHECK(validate_step_result(plan, result, 2).empty());
        for (const auto & output : result.prefills) {
            GPU_OFFLOAD_CHECK(output.status != SeqEngine::PrefillOutput::Status::failed);
            if (output.status == SeqEngine::PrefillOutput::Status::completed) {
                prefill[(size_t)output.slot] = false;
                pending[(size_t)output.slot] = output.token;
                if (output.slot == b.slot) transcript.tokens.push_back(output.token);
            }
        }
        for (const auto & output : result.decode) {
            GPU_OFFLOAD_CHECK(!output.failed);
            pending[(size_t)output.slot] = output.token;
            positions[(size_t)output.slot] += 1 + (int)output.committed_tokens.size();
            if (output.slot == b.slot) {
                transcript.speculative_children += (int)output.committed_tokens.size();
                consume_decode_output_tokens(output, [&](int32_t token) {
                    transcript.tokens.push_back(token);
                    return true;
                });
            }
        }
    };
    for (int round = 0; prefill[0] || prefill[1]; ++round) {
        GPU_OFFLOAD_CHECK(round < 1024);
        SeqEngine::StepPlan plan;
        std::vector<PrefillCandidate> candidates;
        for (int slot = 0; slot < 2; ++slot) {
            if (prefill[(size_t)slot]) candidates.push_back({slot, (uint64_t)slot});
            else plan.decode.push_back({slot, pending[(size_t)slot], speculative});
        }
        plan.prefills = plan_prefill_slices(candidates,
            engine.step_plan_limits((int)plan.decode.size()), 0);
        run(plan);
    }
    // Both contexts grow until even a one-token round cannot fit. The
    // control releases A at this same boundary; the candidate preserves B
    // while A consumes the released pages before completing.
    for (int round = 0; ; ++round) {
        GPU_OFFLOAD_CHECK(round < 1024);
        SeqEngine::StepPlan plan;
        for (int slot = 0; slot < 2; ++slot) plan.decode.push_back({slot, pending[(size_t)slot], speculative});
        if (!engine.reserve_decode(plan)) {
            for (auto & input : plan.decode) input.allow_speculation = false;
            if (!engine.reserve_decode(plan)) break;
        }
        run(plan);
    }
    std::printf("decode pool exhausted at positions %d/%d\n", positions[0], positions[1]);
    if (offload) {
        std::string error;
        GPU_OFFLOAD_CHECK(engine.offload_kv(b.slot, size_t(1) << 30, error));
        const auto state = engine.kv_offload_state(b.slot);
        GPU_OFFLOAD_CHECK(state.suspended && state.bytes > 0);
        std::printf("suspended slot %d: %zu KV bytes at position %d\n",
                    b.slot, state.bytes, positions[(size_t)b.slot]);
        // A consumes the physical pages formerly owned by B; B's per-slot
        // recurrent/compressor/draft state must stay untouched throughout.
        while (positions[(size_t)a.slot] < 640) {
            SeqEngine::StepPlan plan;
            plan.decode.push_back({a.slot, pending[(size_t)a.slot], speculative});
            run(plan);
        }
        engine.retire(a.slot);
        GPU_OFFLOAD_CHECK(engine.restore_kv(b.slot, error));
        GPU_OFFLOAD_CHECK(!engine.kv_offload_state(b.slot).suspended);
        GPU_OFFLOAD_CHECK(engine.kv_offload_state(b.slot).bytes == 0);
    } else {
        engine.retire(a.slot);
    }
    // The control and candidate execute identical B-only shapes from the
    // same logical state, even though its physical block IDs have changed.
    for (int round = 0; round < 32; ++round) {
        SeqEngine::StepPlan plan;
        plan.decode.push_back({b.slot, pending[(size_t)b.slot], speculative});
        run(plan);
    }
    return transcript;
}

void check_model(std::string_view model_env, bool speculative) {
    const std::string model = luce_test::require_model(model_env).string();
    const std::string draft = speculative
        ? luce_test::require_model(luce_test::kDraftModelEnv).string() : std::string();
    if (ggml_backend_cuda_get_device_count() == 0) {
        throw CppUnitTestFramework::TestSkippedException("GPU is unavailable");
    }
    BackendArgs args;
    args.model_path = model.c_str();
    args.draft_path = speculative ? draft.c_str() : nullptr;
    args.device.backend = compiled_placement_backend();
    args.device.max_ctx = 1024;
    args.draft_device = args.device;
    args.max_concurrency = 2;
    args.paged_attention = true;
    args.kv_pool_tokens = 768;
    args.cache_type_k = args.cache_type_v = GGML_TYPE_Q8_0;
    auto backend = create_backend(args);
    GPU_OFFLOAD_CHECK(backend && backend->seq_engine());
    Tokenizer tokenizer;
    GPU_OFFLOAD_CHECK(tokenizer.load_from_gguf(model.c_str()));
    const auto words = tokenizer.encode("The quick brown fox jumps over the lazy dog. ");
    GPU_OFFLOAD_CHECK(!words.empty());
    std::vector<int32_t> prompt;
    while (prompt.size() < 130) prompt.insert(prompt.end(), words.begin(), words.end());
    prompt.resize(130); // crosses DS4's first compressed-page boundary
    const auto control = continue_request(*backend->seq_engine(), prompt, false, speculative);
    const auto resumed = continue_request(*backend->seq_engine(), prompt, true, speculative);
    GPU_OFFLOAD_CHECK(control.tokens == resumed.tokens);
    if (speculative) GPU_OFFLOAD_CHECK(resumed.speculative_children > 0);
    std::printf("matched %zu continuation tokens (%d speculative children)\n",
                resumed.tokens.size(), resumed.speculative_children);
}
}

TEST_CASE(DecodeKvOffloadGpuFixture, QwenSeededContinuationMatchesControl) {
    check_model(luce_test::kQwen35ModelEnv, false);
}
TEST_CASE(DecodeKvOffloadGpuFixture, QwenDraftContinuationMatchesControl) {
    check_model(luce_test::kQwen35ModelEnv, true);
}
TEST_CASE(DecodeKvOffloadGpuFixture, DeepSeekSeededContinuationMatchesControl) {
    check_model(luce_test::kDeepSeek4ModelEnv, false);
}

#undef GPU_OFFLOAD_CHECK
