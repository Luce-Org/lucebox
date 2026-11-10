// Optional full-model regression for paged DeepSeek V4.1 serving on the
// ds41-r9700-gorgon profile (R9700 target, Gorgon Halo expert device). With
// --paged-attention, a sequence's output must not depend on the sequences
// beside it: not on their text, their number, its slot or the order of the
// lanes in a step. Two faults broke that on this profile, which sets both
// fused F16 verify K/V flags (LUCE_DS4_HYBRID_F16_VERIFY_KV and
// LUCE_DS4_F16_KV_LANES):
//   - the F16 K/V attention read the flat pool of every slot's raw ring from
//     row 0, so every paged lane attended slot 0's raw rows at layer 0 and at
//     compressed layers before their first compressed row;
//   - a step with more than one row summed the six routed experts as 4 + 2
//     instead of in one combine, so a row depended on how many rows shared
//     its step.
// The engine reports tokens, not logits, so the check is greedy decoding:
// every sequence's first kTokens tokens, in every case below, must equal the
// same prompt decoded alone in an otherwise idle engine. As in serving, prompt
// rows share steps with the other sequences' rows. The cases cover one, two,
// four and six lanes (the paged cap), both lane orders, slot 0 inside its
// first raw ring pass and slot 0 past it, a sequence alone in slot 3 while the
// other slots hold different text, and prompts that arrive while others
// decode. The hybrid tier runs one row per sequence, so a step holds at most
// six rows.
//
// Needs the model and the expert placement and router bias the launcher
// passes for the box (the hotness table is optional); skips (77) without them
// or without two GPUs. Not registered in ordinary CI.
//   test_ds4_paged_exact_model target.gguf placement.json router_bias.bin [hotness.csv]
#include "common/backend_factory.h"
#include "common/model_backend.h"
#include "common/paged_attention_config.h"
#include "common/peer_access.h"
#include "common/platform_env.h"
#include "placement/placement_backend.h"
#include "seq_engine_contract.h"
#include "server/launch_profiles.h"
#include "server/tokenizer.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace luce::common;

namespace {

constexpr const char * kProfile = "ds41-r9700-gorgon";
constexpr int kSlots = DEEPSEEK4_MAX_PAGED_SEQUENCES;
constexpr int kTokens = 64;            // greedy tokens compared per sequence
constexpr int kReferenceTokens = 80;   // a lane can decode past kTokens while others prefill
constexpr long long kPoolTokens = 16384;
constexpr int kMaxSteps = 2048;

using Prompts = std::vector<std::vector<int32_t>>;

bool is_file(const std::string & path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::is_regular_file(path, ec);
}

int gpu_count() {
    int n = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        n += type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    return n;
}

// The profile's environment as the server applies it with paged attention:
// everything but what paged V4.1 leaves out, and an explicit value wins. It
// goes in before anything starts the GPU runtime, which reads some of it.
const luce::server::LaunchProfile * apply_profile_env(luce::server::CardProfiles & card) {
    std::string error;
    const std::string card_path = std::string(LUCE_MODEL_CARDS_SOURCE_DIR) + "/deepseek-v4.1-flash.json";
    if (!luce::server::load_card_profiles(card_path, card, error)) {
        std::printf("FAIL setup: model card %s: %s\n", card_path.c_str(), error.c_str());
        return nullptr;
    }
    const luce::server::LaunchProfile * profile = card.find(kProfile);
    if (!profile) {
        std::printf("FAIL setup: no profile %s in %s\n", kProfile, card_path.c_str());
        return nullptr;
    }
    for (const luce::server::LaunchProfileEnv & env : profile->env) {
        if (deepseek41_paged_unqualified_env(env.name)) continue;
        set_environment_variable(env.name.c_str(), env.value.c_str(), false);
    }
    // The path under test, whatever the profile or the caller's environment.
    set_environment_variable("LUCE_DS4_HYBRID_F16_VERIFY_KV", "1", true);
    set_environment_variable("LUCE_DS4_F16_KV_LANES", "1", true);
    return profile;
}

// The server's launch with this profile and paged attention: the profile's
// device flags, the flags the launcher adds for paged serving, and
// --expert-device's environment.
std::unique_ptr<ModelBackend> load_backend(const luce::server::LaunchProfile & profile,
                                           const std::string & model, const std::string & placement,
                                           const std::string & router_bias) {
    BackendArgs args;
    args.model_path = model;
    std::optional<DevicePlacement> expert;
    for (const luce::server::LaunchProfileFlag & entry : profile.flags) {
        const std::string & flag = entry.flag;
        const std::string & value = entry.value;
        bool ok = true;
        if (flag == "--target-device") {
            ok = parse_placement_device(value, args.device);
        } else if (flag == "--expert-device") {
            DevicePlacement device;
            ok = parse_placement_device(value, device);
            if (ok) {
                expert = device;
                args.expert_device = device;
            }
        } else if (flag == "--draft-device") {
            ok = parse_placement_device(value, args.draft_device);
        } else if (flag == "--peer-access") {
            args.device.peer_access = true;
        } else if (flag == "--chunk") {
            args.chunk = std::atoi(value.c_str());
            args.chunk_set = true;
        } else if (flag == "--ds4-protected-experts") {
            args.ds4_protected_experts = std::string(LUCE_SERVER_SOURCE_DIR) + "/" + value;
            ok = is_file(args.ds4_protected_experts);
        } else if (flag == "--max-ctx" || flag == "--ds4-prefill") {
            // Paged serving sets both below.
        } else {
            std::printf("FAIL setup: profile flag %s is not mapped by this test\n", flag.c_str());
            return nullptr;
        }
        if (!ok) {
            std::printf("FAIL setup: profile flag %s %s is invalid\n", flag.c_str(), value.c_str());
            return nullptr;
        }
    }
    args.expert_placement = placement;
    args.ds4_router_bias = router_bias;
    args.ds4_prefill_mode = PrefillAttentionMode::Sparse;
    args.ds4_prefill_mode_set = true;
    args.paged_attention = true;
    args.max_concurrency = kSlots;
    args.kv_pool_tokens = kPoolTokens;
    args.device.fit_ctx = false;
    args.device.max_ctx = (int) kPoolTokens;

    BackendPreparation prepared = prepare_backend(std::move(args), BackendAdmissionContext{});
    if (const auto * failure = std::get_if<BackendPreparationFailure>(&prepared)) {
        std::printf("FAIL setup: backend plan: %s\n", failure->message.c_str());
        return nullptr;
    }
    const BackendPlan plan = std::get<BackendPlan>(std::move(prepared));
    if (expert) {
        const PlacementBackend backend =
            expert->backend == PlacementBackend::Auto ? compiled_placement_backend() : expert->backend;
        const std::string gpu = std::to_string(expert->gpu);
        set_environment_variable("LUCE_DS4_MOE_TP", "1", true);
        set_environment_variable("LUCE_DS4_MOE_TP_INPROC", "1", true);
        set_environment_variable("LUCE_DS4_MOE_TP_GPU", gpu.c_str(), true);
        set_environment_variable("LUCE_DS4_MOE_TP_BACKEND", placement_backend_name(backend), true);
    }
    g_peer_access_opt_in = plan.placement().target.peer_access;
    return create_backend(plan);
}

// Six prompts with different text, code and prose. Prompt 0 is short, so in
// slot 0 it stays inside the slot's first raw ring pass (128 rows); the others
// pass it, prompt 3 several times.
Prompts make_prompts(const Tokenizer & tokenizer) {
    static const char * const kNotes[] = {
        "The cache keeps a map from keys to nodes and a list in use order.",
        "Every read moves its node to the front, so the oldest entry sits at the back.",
        "A lock guards the map and the list, because they must change together.",
        "The bucket refills at a fixed rate and never holds more than its capacity.",
        "A failed call waits twice as long as the one before it, up to a ceiling.",
        "Workers sleep on the queue and wake up when a job arrives.",
        "The harbour log records the wind, the tide and every ship that enters.",
        "Orders older than a year move to the archive table each night.",
        "Tests cover the empty case, a single entry, and many threads at once.",
        "The river rose three times that spring, and the mill stopped twice.",
        "Each report lists the slowest request of the day and its cause.",
        "The keeper trims the lamp at dusk and again before the morning watch.",
    };
    struct Spec { int notes; const char * tail; };
    static const Spec kSpecs[] = {
        {0, "The lighthouse keeper wrote one line in the log every night. On the last night of "
            "the season, with the lamp turned low, he wrote"},
        {8, "\nimport threading\n\nclass LRUCache:\n    \"\"\"A thread-safe least-recently-used "
            "cache.\"\"\"\n\n    def __init__(self, capacity):\n"},
        {25, "\nIn a short essay, explain to a new engineer why these notes matter and what "
             "they would change first.\n\n"},
        {45, "\n// A token bucket rate limiter, safe to call from many threads.\n#include <chrono>\n"
             "#include <mutex>\n\nclass TokenBucket {\npublic:\n"},
        {3, "\nThe storm reached the harbour two hours before midnight. The first thing the "
            "keeper noticed was"},
        {15, "\n-- Orders per customer in the last 30 days, with their total value.\nSELECT c.id, c.name,"},
    };
    Prompts prompts;
    for (int p = 0; p < (int) (sizeof(kSpecs) / sizeof(kSpecs[0])); ++p) {
        std::string text;
        if (kSpecs[p].notes) text = "Notes, set " + std::to_string(p) + ":\n";
        for (int i = 0; i < kSpecs[p].notes; ++i) {
            text += std::to_string(i + 1) + ". " + kNotes[(i * 7 + p * 5 + i / 12) % 12] + "\n";
        }
        text += kSpecs[p].tail;
        std::vector<int32_t> tokens;
        if (tokenizer.bos_id() >= 0) tokens.push_back(tokenizer.bos_id());
        const std::vector<int32_t> body = tokenizer.encode(text, /*match_control=*/false);
        tokens.insert(tokens.end(), body.begin(), body.end());
        prompts.push_back(std::move(tokens));
    }
    return prompts;
}

struct Seq {
    int prompt = -1;
    int slot = -1;
    int32_t pending = -1;  // the token to feed next; -1 while the prompt prefills
    std::vector<int32_t> tokens;
    bool live = true;
};

// One case's sequences on the engine; live slots are retired at the end.
class Run {
public:
    Run(SeqEngine & engine, const Prompts & prompts) : engine_(engine), prompts_(prompts) {}
    ~Run() {
        for (const Seq & s : seqs_) {
            if (s.live) engine_.retire(s.slot);
        }
    }

    int admit(int p) {
        const auto a = engine_.admit(next_id_++, prompts_[(size_t) p], SamplerCfg{});
        if (a.status != SeqEngine::AdmitResult::Status::admitted) {
            throw std::runtime_error("admit failed: " + a.error);
        }
        seqs_.push_back({p, a.slot, -1, {}, true});
        return (int) seqs_.size() - 1;
    }

    void retire(int s) {
        engine_.retire(seqs_[(size_t) s].slot);
        seqs_[(size_t) s].live = false;
    }

    // Step the listed sequences, in this lane order, until each holds `count`
    // tokens: a decode row for each sequence past its prompt, one prompt row
    // (the hybrid tier's width) for each still prefilling. The engine steps
    // every decoding sequence each time, so the list holds all of them.
    void run(const std::vector<int> & lanes, int count) {
        for (int i = 0; i < kMaxSteps; ++i) {
            SeqEngine::StepPlan plan;
            bool done = true;
            for (int s : lanes) {
                const Seq & seq = seqs_[(size_t) s];
                done = done && (int) seq.tokens.size() >= count;
                if (seq.pending >= 0) plan.decode.push_back({seq.slot, seq.pending, false});
                else plan.prefills.push_back({seq.slot, 1});
            }
            if (done) return;
            step(plan);
        }
        throw std::runtime_error("a case did not finish");
    }

    const std::vector<Seq> & seqs() const { return seqs_; }

private:
    Seq & by_slot(int slot) {
        for (Seq & s : seqs_) {
            if (s.live && s.slot == slot) return s;
        }
        throw std::runtime_error("output for a slot this case does not hold");
    }

    void step(const SeqEngine::StepPlan & plan) {
        const auto result = engine_.step(plan);
        const auto error = validate_step_result(plan, result, engine_.slot_count());
        if (!result.ok() || !error.empty()) throw std::runtime_error(result.error + error);
        for (const auto & out : result.decode) {
            if (out.failed) throw std::runtime_error(out.error);
            if (!out.committed_tokens.empty()) throw std::runtime_error("a decode row speculated");
            Seq & s = by_slot(out.slot);
            s.tokens.push_back(out.token);
            s.pending = out.token;
        }
        for (const auto & out : result.prefills) {
            if (out.status == SeqEngine::PrefillOutput::Status::failed) throw std::runtime_error(out.error);
            if (out.status != SeqEngine::PrefillOutput::Status::completed) continue;
            Seq & s = by_slot(out.slot);
            s.tokens.push_back(out.token);
            s.pending = out.token;
        }
    }

    SeqEngine & engine_;
    const Prompts & prompts_;
    std::vector<Seq> seqs_;
    uint64_t next_id_ = 1;
};

struct Case {
    const char * name;
    void (*run)(Run & r);
};

// Every sequence still live at the end of a case is compared with its alone run.
const Case kCases[] = {
    {"one lane in slot 3, the other slots holding different text",
     [](Run & r) {
         for (int p : {1, 2, 3, 0}) r.admit(p);
         r.run({0, 1, 2}, 4);  // prompt 0 waits
         for (int s : {0, 1, 2}) r.retire(s);
         r.run({3}, kTokens);
     }},
    {"two lanes, slot 0 inside its first ring pass, slot order",
     [](Run & r) {
         r.admit(0);
         r.admit(1);
         r.run({0, 1}, kTokens);
     }},
    {"two lanes, slot 0 past its first ring pass, reversed order",
     [](Run & r) {
         r.admit(1);
         r.admit(0);
         r.run({1, 0}, kTokens);
     }},
    {"four lanes, slot order",
     [](Run & r) {
         for (int p : {0, 1, 2, 3}) r.admit(p);
         r.run({0, 1, 2, 3}, kTokens);
     }},
    {"four lanes, reversed order",
     [](Run & r) {
         for (int p : {3, 2, 1, 0}) r.admit(p);
         r.run({3, 2, 1, 0}, kTokens);
     }},
    {"six lanes, mixed order",
     [](Run & r) {
         for (int p : {0, 1, 2, 3, 4, 5}) r.admit(p);
         r.run({5, 0, 4, 1, 3, 2}, kTokens);
     }},
    {"prompts arriving while others decode",
     [](Run & r) {
         r.admit(4);
         r.admit(0);
         r.run({0, 1}, 16);
         r.admit(5);
         r.run({2, 0, 1}, 16);
         r.admit(2);
         r.run({3, 1, 0, 2}, kTokens);
     }},
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr, "Usage: test_ds4_paged_exact_model target.gguf placement.json "
                             "router_bias.bin [hotness.csv]\nSKIP: no model given\n");
        return 77;
    }
    for (int i = 1; i < argc; ++i) {
        if (!is_file(argv[i])) {
            std::printf("SKIP: %s is not a file\n", argv[i]);
            return 77;
        }
    }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    luce::server::CardProfiles card;
    const luce::server::LaunchProfile * profile = apply_profile_env(card);
    if (!profile) return 1;
    if (argc == 5) set_environment_variable("LUCE_DS4_HOTNESS_CSV", argv[4], true);
    if (gpu_count() < 2) {
        std::printf("SKIP: the %s profile needs two GPUs\n", kProfile);
        return 77;
    }

    std::unique_ptr<ModelBackend> backend = load_backend(*profile, argv[1], argv[2], argv[3]);
    SeqEngine * engine = backend ? backend->seq_engine() : nullptr;
    if (!engine || engine->slot_count() < kSlots) {
        std::printf("FAIL setup: no paged DeepSeek V4.1 engine with %d slots\n", kSlots);
        return 1;
    }
    if (engine->step_plan_limits(0).max_prefill_tokens_per_sequence != 1) {
        std::printf("FAIL setup: the profile did not start the hybrid expert tier\n");
        return 1;
    }
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(argv[1])) {
        std::printf("FAIL setup: tokenizer\n");
        return 1;
    }
    const Prompts prompts = make_prompts(tokenizer);

    int failures = 0, checked = 0;
    try {
        // Each prompt alone in an otherwise idle engine.
        std::vector<std::vector<int32_t>> reference;
        for (int p = 0; p < (int) prompts.size(); ++p) {
            Run r(*engine, prompts);
            r.run({r.admit(p)}, kReferenceTokens);
            reference.push_back(r.seqs()[0].tokens);
            std::printf("reference: prompt %d, %zu prompt tokens, slot %d\n", p, prompts[(size_t) p].size(),
                        r.seqs()[0].slot);
        }
        for (const Case & c : kCases) {
            Run r(*engine, prompts);
            c.run(r);
            int bad = 0;
            std::string slots;
            for (const Seq & s : r.seqs()) {
                if (!s.live) continue;
                const std::vector<int32_t> & want = reference[(size_t) s.prompt];
                const size_t n = std::min(want.size(), s.tokens.size());
                if (n < (size_t) kTokens) throw std::runtime_error("a case decoded too few tokens");
                const auto diff = std::mismatch(want.begin(), want.begin() + n, s.tokens.begin());
                slots += " " + std::to_string(s.prompt) + "@" + std::to_string(s.slot);
                ++checked;
                if (diff.first == want.begin() + n) continue;
                ++bad;
                const size_t at = (size_t) (diff.first - want.begin());
                std::printf("FAIL: %s: prompt %d in slot %d differs from alone at token %zu "
                            "(%d, alone %d)\n", c.name, s.prompt, s.slot, at, *diff.second, *diff.first);
            }
            std::printf("%s: %s (prompt@slot:%s)\n", bad ? "FAIL" : "PASS", c.name, slots.c_str());
            failures += bad;
        }
    } catch (const std::exception & e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::printf("FAIL: %d of %d sequences changed with the sequences beside them\n", failures, checked);
        return 1;
    }
    std::printf("PASS: %d sequences, %d greedy tokens each, identical alone and beside others\n",
                checked, kTokens);
    return 0;
}
