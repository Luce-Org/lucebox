// Model-level proof for runtime model swapping: evict() returns the model's
// device memory, reinstate() rebuilds it, and a host prefix snapshot taken
// before the eviction restores afterwards with an identical greedy
// continuation. The two-model case swaps a second model through the freed
// device in between, which is the serving scenario.
//
// Needs LUCE_TEST_MODEL_QWEN35 + LUCE_TEST_MODEL_DRAFT (model A) and, for the
// swap case, LUCE_TEST_MODEL_QWEN35_B + LUCE_TEST_MODEL_DRAFT_B (model B).
// LUCE_TEST_SWAP_CYCLES (default 3) sets the evict/reinstate repetitions.
//
// DeepSeek V4.1 partial eviction runs the ds41-lucebox profile (target hip:0,
// experts hip:1, locked host memory for the secondary tier) and needs
// LUCE_TEST_MODEL_DEEPSEEK4 (the V4.1 GGUF) + LUCE_TEST_MODEL_DEEPSEEK4_DRAFT
// (its DSpark drafter); LUCE_TEST_DS4_CYCLES (default 10) sets its cycles.
#include "CppUnitTestFramework.hpp"
#include "model_test_paths.h"
#include "common/backend_factory.h"
#include "common/model_backend.h"
#include "common/peer_access.h"
#include "deepseek4/deepseek4_backend.h"
#include "server/launch_profiles.h"
#include "server/tokenizer.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace luce::common;
struct ModelSwapGpuFixture {};
#define SWAP_CHECK(x) do { if (!(x)) throw std::runtime_error( \
    std::string("model swap line ") + std::to_string(__LINE__) + ": " #x); } while (false)

constexpr std::string_view kQwen35ModelBEnv = "LUCE_TEST_MODEL_QWEN35_B";
constexpr std::string_view kDraftModelBEnv = "LUCE_TEST_MODEL_DRAFT_B";
constexpr double kMiB = 1024.0 * 1024.0;

// Free device memory once released buffers are back: the amdgpu driver frees
// a buffer whose last fence is still pending in the background, so right
// after an eviction free memory can read up to ~10 GiB low for a few ms.
size_t device_free(int device = 0) {
    size_t free = 0, total = 0;
    ggml_backend_cuda_get_device_memory(device, &free, &total);
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        size_t now = 0;
        ggml_backend_cuda_get_device_memory(device, &now, &total);
        if (now <= free || std::chrono::steady_clock::now() - start > std::chrono::seconds(2)) {
            return std::max(free, now);
        }
        free = now;
    }
}

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

std::unique_ptr<ModelBackend> make_backend(const std::string & model,
                                           const std::string & draft) {
    BackendArgs args;
    args.model_path = model;
    args.draft_path = draft;
    args.device.backend = compiled_placement_backend();
    args.device.max_ctx = 4096;
    args.draft_device = args.device;
    args.max_concurrency = 1;
    args.cache_type_k = args.cache_type_v = GGML_TYPE_Q8_0;
    auto prepared = prepare_backend(args);
    if (const auto * failure = std::get_if<BackendPreparationFailure>(&prepared)) {
        throw std::runtime_error(failure->message);
    }
    auto backend = create_backend(std::get<BackendPlan>(prepared));
    SWAP_CHECK(backend);
    std::string reason;
    SWAP_CHECK(backend->supports_eviction(reason));
    return backend;
}

GenerateRequest greedy(std::vector<int32_t> prompt, int n_gen) {
    GenerateRequest req;
    req.prompt = std::move(prompt);
    req.n_gen = n_gen;
    req.sampler.temp = 0.0f;
    return req;
}

// A session on one model: a long prompt, a snapshot after the first answer,
// and the follow-up request that restores from it.
struct Session {
    std::vector<int32_t> prompt;
    std::vector<int32_t> first_answer;
    std::vector<int32_t> follow_up;
    std::vector<int32_t> control;  // follow-up continuation without any eviction
    static constexpr int slot = 0;
};

Session start_session(ModelBackend & backend, const std::string & model_path) {
    Tokenizer tokenizer;
    SWAP_CHECK(tokenizer.load_from_gguf(model_path.c_str()));
    std::string text;
    for (int i = 0; text.size() < 6000; ++i) {
        text += "Record " + std::to_string(i) + ": the courier left depot " +
                std::to_string(i * 7 % 13) + " at " + std::to_string(6 + i % 12) +
                " o'clock carrying " + std::to_string(i * 3 + 1) + " parcels. ";
    }
    Session session;
    session.prompt = tokenizer.encode(text + "\nWhich depot did record 17 leave from?");
    DaemonIO io;
    const auto first = backend.generate(greedy(session.prompt, 32), io);
    SWAP_CHECK(first.ok());
    session.first_answer = first.tokens;
    SWAP_CHECK(backend.snapshot_save(Session::slot));
    const int pos = backend.snapshot_cur_pos(Session::slot);
    std::vector<int32_t> history = session.prompt;
    history.insert(history.end(), first.tokens.begin(), first.tokens.end());
    SWAP_CHECK(pos > 0 && pos <= (int)history.size());
    history.resize((size_t)pos);
    const auto question = tokenizer.encode("\nAnd how many parcels did record 23 carry?");
    history.insert(history.end(), question.begin(), question.end());
    session.follow_up = history;
    const auto control = backend.restore_and_generate(Session::slot, greedy(session.follow_up, 32), io);
    SWAP_CHECK(control.ok());
    SWAP_CHECK(control.restored_prefix_tokens > 0);
    session.control = control.tokens;
    std::printf("session: prompt %zu tokens, snapshot at %d, follow-up %zu tokens\n",
                session.prompt.size(), pos, session.follow_up.size());
    return session;
}

// After a reinstatement the snapshot must restore and reproduce the control.
void check_session(ModelBackend & backend, const Session & session) {
    DaemonIO io;
    const auto resumed = backend.restore_and_generate(Session::slot, greedy(session.follow_up, 32), io);
    SWAP_CHECK(resumed.ok());
    SWAP_CHECK(resumed.restored_prefix_tokens > 0);
    SWAP_CHECK(resumed.tokens == session.control);
    const auto fresh = backend.generate(greedy(session.prompt, 32), io);
    SWAP_CHECK(fresh.ok());
    SWAP_CHECK(fresh.tokens == session.first_answer);
}

void evict(ModelBackend & backend, const char * name,
           ModelBackend::EvictLevel level = ModelBackend::EvictLevel::cold) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = backend.evict(level);
    SWAP_CHECK(result.status == ModelBackend::EvictStatus::ok);
    SWAP_CHECK(result.snapshots_lost == 0);
    SWAP_CHECK(backend.is_evicted());
    std::printf("%s: evict %.0f ms, device free %.0f MiB\n", name, ms_since(start),
                device_free() / kMiB);
}

void reinstate(ModelBackend & backend, const char * name) {
    const auto start = std::chrono::steady_clock::now();
    std::string error;
    SWAP_CHECK(backend.reinstate(error));
    SWAP_CHECK(!backend.is_evicted());
    std::printf("%s: reinstate %.0f ms, device free %.0f MiB\n", name, ms_since(start),
                device_free() / kMiB);
}

int swap_cycles() {
    const char * value = std::getenv("LUCE_TEST_SWAP_CYCLES");
    return value && std::atoi(value) > 0 ? std::atoi(value) : 3;
}

void require_gpu() {
    if (ggml_backend_cuda_get_device_count() == 0) {
        throw CppUnitTestFramework::TestSkippedException("GPU is unavailable");
    }
}
}  // namespace

TEST_CASE(ModelSwapGpuFixture, QwenEvictReinstateKeepsSnapshots) {
    const std::string model = luce_test::require_model(luce_test::kQwen35ModelEnv).string();
    const std::string draft = luce_test::require_model(luce_test::kDraftModelEnv).string();
    require_gpu();
    const size_t baseline = device_free();
    auto backend = make_backend(model, draft);
    const Session session = start_session(*backend, model);

    evict(*backend, "A");
    const size_t first_evicted = device_free();
    // Everything the model allocated is returned; the slack covers the HIP
    // runtime's own per-process allocations made during the first launches.
    std::printf("baseline free %.0f MiB, after evict %.0f MiB\n",
                baseline / kMiB, first_evicted / kMiB);
    SWAP_CHECK(first_evicted + (size_t)(512 * kMiB) >= baseline);

    // A request on an evicted model fails cleanly instead of touching freed memory.
    DaemonIO io;
    const auto refused = backend->generate(greedy(session.prompt, 4), io);
    SWAP_CHECK(!refused.ok());
    SWAP_CHECK(backend->evict().status == ModelBackend::EvictStatus::ok);  // idempotent

    reinstate(*backend, "A");
    check_session(*backend, session);

    size_t last_evicted = first_evicted;
    for (int cycle = 1; cycle < swap_cycles(); ++cycle) {
        evict(*backend, "A");
        last_evicted = device_free();
        reinstate(*backend, "A");
    }
    check_session(*backend, session);
    // No device memory creeps across cycles.
    std::printf("free after evict: first %.0f MiB, last %.0f MiB\n",
                first_evicted / kMiB, last_evicted / kMiB);
    SWAP_CHECK(last_evicted + (size_t)(64 * kMiB) >= first_evicted);

    // Destroying an evicted backend is the shutdown path of a parked model.
    evict(*backend, "A");
    backend.reset();
}

TEST_CASE(ModelSwapGpuFixture, QwenSwapThroughSecondModelKeepsSnapshots) {
    const std::string model_a = luce_test::require_model(luce_test::kQwen35ModelEnv).string();
    const std::string draft_a = luce_test::require_model(luce_test::kDraftModelEnv).string();
    const std::string model_b = luce_test::require_model(kQwen35ModelBEnv).string();
    const std::string draft_b = luce_test::require_model(kDraftModelBEnv).string();
    require_gpu();

    auto a = make_backend(model_a, draft_a);
    const Session session_a = start_session(*a, model_a);
    evict(*a, "A");

    // B is created and used while A is evicted, then evicted in turn.
    auto b = make_backend(model_b, draft_b);
    const Session session_b = start_session(*b, model_b);
    evict(*b, "B");

    for (int cycle = 0; cycle < swap_cycles(); ++cycle) {
        reinstate(*a, "A");
        check_session(*a, session_a);
        evict(*a, "A");
        reinstate(*b, "B");
        check_session(*b, session_b);
        evict(*b, "B");
    }
}

// Warm eviction keeps both models' weights on the device and swaps only the
// per-model execution state (cache, graphs, backend context), so switching
// models skips the weight upload.
TEST_CASE(ModelSwapGpuFixture, QwenWarmSwapKeepsWeightsAndSnapshots) {
    const std::string model_a = luce_test::require_model(luce_test::kQwen35ModelEnv).string();
    const std::string draft_a = luce_test::require_model(luce_test::kDraftModelEnv).string();
    const std::string model_b = luce_test::require_model(kQwen35ModelBEnv).string();
    const std::string draft_b = luce_test::require_model(kDraftModelBEnv).string();
    require_gpu();
    constexpr auto warm = ModelBackend::EvictLevel::warm;

    auto a = make_backend(model_a, draft_a);
    const Session session_a = start_session(*a, model_a);
    evict(*a, "A", warm);
    SWAP_CHECK(a->weights_resident() && a->weight_device_bytes() > 0);
    auto b = make_backend(model_b, draft_b);
    const Session session_b = start_session(*b, model_b);
    evict(*b, "B", warm);
    const size_t both_warm = device_free();

    for (int cycle = 0; cycle < swap_cycles(); ++cycle) {
        reinstate(*a, "A (warm)");
        check_session(*a, session_a);
        evict(*a, "A", warm);
        reinstate(*b, "B (warm)");
        check_session(*b, session_b);
        evict(*b, "B", warm);
    }
    // Nothing but the two weight sets stays on the device between turns.
    SWAP_CHECK(device_free() + (size_t)(64 * kMiB) >= both_warm);
    std::printf("weights kept: A %.0f MiB, B %.0f MiB\n",
                a->weight_device_bytes() / kMiB, b->weight_device_bytes() / kMiB);

    // Warm to cold returns the weights; the model still reinstates from disk.
    evict(*a, "A");
    SWAP_CHECK(!a->weights_resident());
    reinstate(*a, "A (cold)");
    check_session(*a, session_a);
}

namespace {

constexpr std::string_view kDeepSeek4DraftEnv = "LUCE_TEST_MODEL_DEEPSEEK4_DRAFT";

int ds4_cycles() {
    const char * value = std::getenv("LUCE_TEST_DS4_CYCLES");
    return value && std::atoi(value) > 0 ? std::atoi(value) : 10;
}

// The ds41-lucebox profile as the server applies it: its environment as
// defaults, --expert-device hip:1 (server_main apply_expert_device),
// --peer-access, and its flags.
std::unique_ptr<ModelBackend> make_ds41_backend(const std::string & model,
                                                const std::string & draft) {
    const auto * profile = luce::server::find_launch_profile("ds41-lucebox");
    SWAP_CHECK(profile);
    for (const auto & env : profile->env) setenv(env.name, env.value, /*overwrite=*/0);
    setenv("LUCE_DS4_MOE_TP", "1", 1);
    setenv("LUCE_DS4_MOE_TP_INPROC", "1", 1);
    setenv("LUCE_DS4_MOE_TP_GPU", "1", 1);
    setenv("LUCE_DS4_MOE_TP_BACKEND", "hip", 1);
    g_peer_access_opt_in = true;
    const std::string share = LUCE_TEST_SHARE_DIR "/deepseek41/";
    BackendArgs args;
    args.model_path = model;
    args.draft_path = draft;
    args.device.backend = compiled_placement_backend();
    args.device.gpu = 0;
    args.device.max_ctx = 16384;
    args.chunk = 4096;
    args.ds4_prefill_mode = PrefillAttentionMode::Dense;
    args.ds4_prefill_mode_set = true;
    args.expert_placement = share + "placement_lucebox.json";
    args.ds4_router_bias = share + "router_bias_lucebox_40x384_f32.bin";
    args.ds4_protected_experts = share + "massive_experts.json";
    auto prepared = prepare_backend(args);
    if (const auto * failure = std::get_if<BackendPreparationFailure>(&prepared)) {
        throw std::runtime_error(failure->message);
    }
    auto backend = create_backend(std::get<BackendPlan>(prepared));
    SWAP_CHECK(backend);
    std::string reason;
    if (!backend->supports_eviction(reason)) throw std::runtime_error("DS4 eviction: " + reason);
    return backend;
}

// DS4 snapshots at a prefill boundary (after a decode, DSpark refuses one):
// the snapshot is taken at the end of the prompt, and the follow-up restores
// it and prefills its own suffix.
Session start_ds4_session(ModelBackend & backend, const std::string & model_path) {
    Tokenizer tokenizer;
    SWAP_CHECK(tokenizer.load_from_gguf(model_path.c_str()));
    std::string text;
    for (int i = 0; text.size() < 6000; ++i) {
        text += "Record " + std::to_string(i) + ": the courier left depot " +
                std::to_string(i * 7 % 13) + " at " + std::to_string(6 + i % 12) +
                " o'clock carrying " + std::to_string(i * 3 + 1) + " parcels. ";
    }
    Session session;
    session.prompt = tokenizer.encode(text + "\nWhich depot did record 17 leave from?");
    GenerateRequest first = greedy(session.prompt, 32);
    first.snap_slot = Session::slot;
    first.snap_pos = (int) session.prompt.size();
    first.restore_points = {first.snap_pos};
    DaemonIO io;
    const auto answer = backend.generate(first, io);
    SWAP_CHECK(answer.ok());
    session.first_answer = answer.tokens;
    SWAP_CHECK(backend.snapshot_used(Session::slot));
    SWAP_CHECK(backend.snapshot_cur_pos(Session::slot) == first.snap_pos);
    session.follow_up = session.prompt;
    const auto question = tokenizer.encode("\nAnd how many parcels did record 23 carry?");
    session.follow_up.insert(session.follow_up.end(), question.begin(), question.end());
    const auto control = backend.restore_and_generate(Session::slot, greedy(session.follow_up, 32), io);
    SWAP_CHECK(control.ok());
    SWAP_CHECK(control.restored_prefix_tokens == first.snap_pos);
    session.control = control.tokens;
    std::printf("ds4 session: prompt %zu tokens, snapshot at %d, follow-up %zu tokens\n",
                session.prompt.size(), first.snap_pos, session.follow_up.size());
    return session;
}

void check_ds4_session(ModelBackend & backend, const Session & session) {
    DaemonIO io;
    const auto resumed = backend.restore_and_generate(Session::slot, greedy(session.follow_up, 32), io);
    SWAP_CHECK(resumed.ok());
    SWAP_CHECK(resumed.restored_prefix_tokens > 0);
    SWAP_CHECK(resumed.tokens == session.control);
}

}  // namespace

// Partial eviction returns the primary device while the secondary expert
// stack, the streamed expert cache and the host snapshots stay; reinstate()
// rebuilds the primary around them without reloading a secondary expert.
TEST_CASE(ModelSwapGpuFixture, DeepSeek4PartialEvictionKeepsSecondaryTierAndSnapshots) {
    const std::string model = luce_test::require_model(luce_test::kDeepSeek4ModelEnv).string();
    const std::string draft = luce_test::require_model(kDeepSeek4DraftEnv).string();
    require_gpu();
    if (ggml_backend_cuda_get_device_count() < 2) {
        throw CppUnitTestFramework::TestSkippedException("needs two GPUs (hip:0 target, hip:1 experts)");
    }
    constexpr auto partial = ModelBackend::EvictLevel::partial;
    const size_t baseline = device_free(0);
    auto backend = make_ds41_backend(model, draft);
    auto * ds4 = dynamic_cast<DeepSeek4Backend *>(backend.get());
    SWAP_CHECK(ds4);
    const Session session = start_ds4_session(*backend, model);
    const auto loaded = ds4->expert_tier_report();
    SWAP_CHECK(loaded.secondary_upload_bytes > 0 && loaded.secondary_base);

    // Warm is not a DS4 level, and a refusal changes nothing.
    SWAP_CHECK(backend->evict(ModelBackend::EvictLevel::warm).status == ModelBackend::EvictStatus::rejected);
    SWAP_CHECK(!backend->is_evicted());

    evict(*backend, "DS4", partial);
    const size_t first_evicted = device_free(0);
    const size_t secondary_evicted = device_free(1);
    std::printf("hip:0 baseline free %.0f MiB, after partial evict %.0f MiB\n",
                baseline / kMiB, first_evicted / kMiB);
    // The primary device comes back to what it was before the model loaded,
    // within the HIP runtime's own per-process allocations.
    SWAP_CHECK(first_evicted + (size_t) (512 * kMiB) >= baseline);
    SWAP_CHECK(backend->weights_resident());
    const auto evicted = ds4->expert_tier_report();
    SWAP_CHECK(evicted.partially_evicted && evicted.stream_cache_suspended);
    SWAP_CHECK(evicted.storage == loaded.storage && evicted.secondary_base == loaded.secondary_base);
    // Snapshots stay accounted for while no live cache exists.
    const auto report = backend->memory_report();
    SWAP_CHECK(report.available && !report.snapshots.empty());
    DaemonIO io;
    SWAP_CHECK(!backend->generate(greedy(session.prompt, 4), io).ok());
    SWAP_CHECK(backend->evict(partial).status == ModelBackend::EvictStatus::ok);  // idempotent

    // The settings were resolved at load: a changed environment does not
    // move the experts (reinstate verifies the placement and biases).
    setenv("LUCE_DS4_MOE_TP_GPU", "0", 1);
    setenv("LUCE_EXPERT_BUDGET_MB", "1", 1);
    reinstate(*backend, "DS4 (partial)");
    const auto back = ds4->expert_tier_report();
    SWAP_CHECK(back.secondary_upload_bytes == loaded.secondary_upload_bytes);  // no secondary re-read
    SWAP_CHECK(back.secondary_base == loaded.secondary_base);
    SWAP_CHECK(back.primary_upload_bytes > loaded.primary_upload_bytes);
    SWAP_CHECK(!back.stream_cache_suspended);
    check_ds4_session(*backend, session);

    size_t last_evicted = first_evicted;
    for (int cycle = 1; cycle < ds4_cycles(); ++cycle) {
        evict(*backend, "DS4", partial);
        last_evicted = device_free(0);
        reinstate(*backend, "DS4 (partial)");
    }
    check_ds4_session(*backend, session);
    SWAP_CHECK(ds4->expert_tier_report().secondary_upload_bytes == loaded.secondary_upload_bytes);
    std::printf("hip:0 free after partial evict: first %.0f MiB, last %.0f MiB; hip:1 %.0f MiB\n",
                first_evicted / kMiB, last_evicted / kMiB, secondary_evicted / kMiB);
    // Nothing creeps on either device across cycles.
    SWAP_CHECK(last_evicted + (size_t) (64 * kMiB) >= first_evicted);
    evict(*backend, "DS4", partial);
    SWAP_CHECK(device_free(1) + (size_t) (64 * kMiB) >= secondary_evicted);

    // Partial to cold releases the retained tier; a cold reinstate reloads it
    // with the same placement and the snapshot still restores.
    evict(*backend, "DS4");
    SWAP_CHECK(!backend->weights_resident());
    reinstate(*backend, "DS4 (cold)");
    check_ds4_session(*backend, session);

    // Destroying an evicted backend is the shutdown path.
    evict(*backend, "DS4", partial);
    backend.reset();
}
