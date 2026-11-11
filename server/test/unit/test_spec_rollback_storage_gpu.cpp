// Model-level proof that serial speculative decode always has its rollback
// snapshot storage: with a drafter under the KVFlash pager (fast rollback
// resolves to off there) and with --no-fast-rollback, a greedy request must
// decode with accepted draft tokens and match the autoregressive control.
#include "CppUnitTestFramework.hpp"
#include "model_test_paths.h"
#include "common/backend_factory.h"
#include "common/model_backend.h"
#include "server/launch_profiles.h"
#include "server/tokenizer.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

namespace {
using namespace luce::common;
struct SpecRollbackStorageGpuFixture {};

// Sets environment variables for one backend's lifetime and restores them.
class ScopedEnv {
public:
    void set(const std::string & name, const std::string & value) {
        const char * old = std::getenv(name.c_str());
        saved_.push_back({name, old != nullptr, old ? old : ""});
        setenv(name.c_str(), value.c_str(), 1);
    }
    ~ScopedEnv() {
        for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
            if (it->had) setenv(it->name.c_str(), it->value.c_str(), 1);
            else unsetenv(it->name.c_str());
        }
    }
private:
    struct Saved { std::string name; bool had; std::string value; };
    std::vector<Saved> saved_;
};

// Runs the requests one after another on one fresh backend and returns the
// last result; force_ar[i] picks AR or speculative decode for request i. A
// KVFlash session resumes request i+1 from request i's checkpoint.
// used_vram, when given, receives the device memory in use after the last
// request, with the backend still alive.
GenerateResult run_requests(const BackendArgs & args, const std::vector<int32_t> & prompt,
                            const std::vector<bool> & force_ar, size_t * used_vram = nullptr) {
    auto prepared = prepare_backend(args);
    if (const auto * failure = std::get_if<BackendPreparationFailure>(&prepared)) {
        throw std::runtime_error(failure->message);
    }
    auto backend = create_backend(std::get<BackendPlan>(prepared));
    if (!backend) throw std::runtime_error("create_backend failed");
    GenerateResult result;
    for (const bool ar : force_ar) {
        GenerateRequest request;
        request.prompt = prompt;
        request.n_gen = 48;
        request.sampler.temp = 0.0f;
        request.force_ar_decode = ar;
        result = backend->generate(request, DaemonIO{});
        if (!result.ok()) {
            throw std::runtime_error(std::string(ar ? "AR control" : "speculative decode") +
                                     " failed: error=" + std::string(result.error_code()) +
                                     " detail=" + std::string(result.error_detail()));
        }
    }
    if (used_vram) {
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_cuda_get_device_memory(0, &free_bytes, &total_bytes);
        *used_vram = total_bytes - free_bytes;
    }
    return result;
}

void require_spec_matches(const GenerateResult & spec, const GenerateResult & control) {
    std::printf("spec %zu tokens, accept_rate %.3f, ran=%d; control %zu tokens, equal=%d\n",
                spec.tokens.size(), (double)spec.accept_rate, (int)spec.spec_decode_ran,
                control.tokens.size(), (int)(spec.tokens == control.tokens));
    if (spec.tokens.size() < 32) throw std::runtime_error("fewer than 32 tokens generated");
    if (!spec.spec_decode_ran || !(spec.accept_rate > 0.0f)) {
        throw std::runtime_error("no speculative tokens were accepted");
    }
    if (spec.tokens != control.tokens) {
        size_t at = 0;
        while (at < spec.tokens.size() && at < control.tokens.size() && spec.tokens[at] == control.tokens[at]) ++at;
        throw std::runtime_error("speculative tokens differ from the AR control at index " + std::to_string(at));
    }
}

// resumed: the compared request is the second one on its backend, so it
// resumes from the first (AR) request's checkpoint; the control resumes from
// the same checkpoint and differs only in decoding autoregressively.
void check_spec_matches_ar(bool kvflash, bool resumed = false) {
    const std::string model = luce_test::require_model(luce_test::kQwen38ModelEnv).string();
    const std::string draft = luce_test::require_model(luce_test::kQwen38DraftModelEnv).string();
    if (ggml_backend_cuda_get_device_count() == 0) {
        throw CppUnitTestFramework::TestSkippedException("GPU is unavailable");
    }
    ScopedEnv env;
    if (kvflash) {
        // What `--kvflash 16384` sets in the server.
        env.set("LUCE_KVFLASH", "16384");
        for (const auto & e : luce::server::kvflash_recall_env("qwen35", false)) {
            env.set(e.name, e.value);
        }
    }
    BackendArgs args;
    args.model_path = model;
    args.draft_path = draft;
    args.draft_block_size = 16;
    args.device.backend = compiled_placement_backend();
    args.device.max_ctx = 20480;
    args.draft_device = args.device;
    // KVFlash resolves fast rollback off unless it is asked for by name;
    // the other case turns it off by name (--no-fast-rollback).
    args.fast_rollback = kvflash;
    args.fast_rollback_explicit = false;
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model.c_str())) throw std::runtime_error("tokenizer load failed");
    const auto sentence = tokenizer.encode("The quick brown fox jumps over the lazy dog. ");
    std::vector<int32_t> prompt;
    while (prompt.size() < 600) prompt.insert(prompt.end(), sentence.begin(), sentence.end());

    const std::vector<bool> control_seq = resumed ? std::vector<bool>{true, true} : std::vector<bool>{true};
    const std::vector<bool> spec_seq = resumed ? std::vector<bool>{true, false} : std::vector<bool>{false};
    const GenerateResult control = run_requests(args, prompt, control_seq);
    size_t used_without_checkpoints = 0;
    const GenerateResult spec = run_requests(args, prompt, spec_seq, &used_without_checkpoints);
    require_spec_matches(spec, control);

    if (!kvflash) {
        // Without fast rollback nothing captures per-token checkpoints, so
        // their buffers (about 2.4 GB for Qwen3.8-27B at block size 16) must
        // not be allocated: only the snapshot pair is. Compare against the
        // same request with fast rollback on.
        BackendArgs with_checkpoints = args;
        with_checkpoints.fast_rollback = true;
        with_checkpoints.fast_rollback_explicit = true;
        size_t used_with_checkpoints = 0;
        run_requests(with_checkpoints, prompt, spec_seq, &used_with_checkpoints);
        const double saved_gib = ((double)used_with_checkpoints - (double)used_without_checkpoints) / (1024.0 * 1024.0 * 1024.0);
        std::printf("VRAM in use: %.2f GiB without checkpoints, %.2f GiB with; saved %.2f GiB\n",
                    (double)used_without_checkpoints / (1024.0 * 1024.0 * 1024.0),
                    (double)used_with_checkpoints / (1024.0 * 1024.0 * 1024.0), saved_gib);
        if (saved_gib < 1.5) {
            throw std::runtime_error("fast rollback off still allocates the per-token checkpoint buffers");
        }
    }
}
}

TEST_CASE(SpecRollbackStorageGpuFixture, KvflashSerialSpecMatchesAr) {
    check_spec_matches_ar(true);
}
TEST_CASE(SpecRollbackStorageGpuFixture, NoFastRollbackSerialSpecMatchesAr) {
    check_spec_matches_ar(false);
}
TEST_CASE(SpecRollbackStorageGpuFixture, KvflashResumedSerialSpecMatchesAr) {
    check_spec_matches_ar(true, true);
}
