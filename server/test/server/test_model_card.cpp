#include "server/model_card.h"
#include "server/http_server.h"
#include "support/environment.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using json = nlohmann::json;
using namespace dflash::common;

// ═══════════════════════════════════════════════════════════════════════
// /props body shape tests (model-free)
//
// Verify build_props_body's new wholesale-sidecar `model_card` + new
// `budget_envelope` section per docs/specs/props-endpoint.md §4.9 / §4.X.
// ═══════════════════════════════════════════════════════════════════════

static ServerConfig make_props_config_with_sidecar(const json & sidecar) {
    ServerConfig cfg;
    cfg.arch                    = "qwen35";
    cfg.model_path              = "/tmp/fake/model.gguf";
    cfg.model_card_source_label = "share/model_cards/qwen3.6-27b.json";
    cfg.model_card_json         = sidecar;
    cfg.default_max_tokens      = 32768;
    cfg.hard_limit_reply_budget = 512;
    cfg.think_max_tokens        = 32256;
    cfg.effort_tiers.low    = 4032;
    cfg.effort_tiers.medium = 16128;
    cfg.effort_tiers.high   = 32256;
    cfg.effort_tiers.x_high = 56832;
    cfg.effort_tiers.max    = 81408;
    return cfg;
}

TEST_CASE(ServerUnitFixture, test_model_card_env_override_beats_cwd) {
    // DFLASH_MODEL_CARDS_DIR used to be the LAST candidate, tried after the cwd-relative
    // "share/model_cards". Running from a directory that happened to contain one silently
    // ignored the operator's explicit override. An explicit setting must win.
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "dflash-mc-env-test";
    const auto envdir = root / "explicit";
    fs::remove_all(root);
    fs::create_directories(envdir);

    // A card the resolver can only have found via the env var.
    {
        FILE * f = std::fopen((envdir / "env-probe-model.json").string().c_str(), "w");
        TEST_ASSERT(f != nullptr);
        std::fprintf(f, "{\"name\":\"env-probe-model\",\"source\":\"test\","
                        "\"verified_at\":\"2026-08-04\",\"max_tokens\":4321}");
        std::fclose(f);
    }

    const char * prev = std::getenv("DFLASH_MODEL_CARDS_DIR");
    const std::string saved = prev ? prev : "";
    setenv("DFLASH_MODEL_CARDS_DIR", envdir.string().c_str(), 1);

    auto card = dflash::common::resolve_model_card("", "env-probe-model", "deepseek4", "");

    if (saved.empty()) unsetenv("DFLASH_MODEL_CARDS_DIR");
    else setenv("DFLASH_MODEL_CARDS_DIR", saved.c_str(), 1);
    fs::remove_all(root);

    // Resolved from the env dir, not the deepseek4 family fallback (which gives 32768).
    TEST_ASSERT(card.max_tokens == 4321);
    TEST_ASSERT(card.source_label != "family:deepseek4");
}

TEST_CASE(ServerUnitFixture, test_model_card_family_fallback_deepseek4) {
    // deepseek4 had NO family entry, so every DeepSeek4 artifact -- including the
    // published ROCmFPX GGUFs -- fell through to the hard fallback, taking a generic
    // 16000-token ceiling that is a placeholder rather than a measured property of the
    // model, and reporting model_card = null on /props.
    //
    // Pins the branch rather than the exact ceiling: an operator is expected to ship a
    // sidecar for the real figures, and the fallback is deliberately conservative.
    // What must not regress is that deepseek4 resolves to a FAMILY card at all, and
    // carries the wider reply budget rather than the terse 512 default.
    auto card = dflash::common::resolve_model_card("", "", "deepseek4", "");
    TEST_ASSERT(card.source_label == "family:deepseek4");
    TEST_ASSERT(card.max_tokens == 32768);
    TEST_ASSERT(card.hard_limit_reply_budget == 4096);

    // An unknown architecture must still fall through, or the safety net would mask
    // genuinely unsupported models.
    auto unknown = dflash::common::resolve_model_card("", "", "not-a-real-arch", "");
    TEST_ASSERT(unknown.source_label != "family:not-a-real-arch");
}

TEST_CASE(ServerUnitFixture, test_props_model_card_wholesale_sidecar) {
    // When a sidecar was loaded, /props.model_card should be the parsed
    // sidecar JSON verbatim — *all* fields from the file, not just the
    // five budget-derived ones from the pre-refactor shape.
    json sidecar = {
        {"name",         "Qwen3.6 27B"},
        {"source",       "https://huggingface.co/Qwen/Qwen3.6-27B"},
        {"verified_at", "2026-05-23"},
        {"max_tokens",   32768},
        {"complex_problem_max_tokens", 81920},
        {"sampling", {
            {"temperature", 1.0},
            {"top_p",       0.95},
            {"top_k",       20},
        }},
        {"reasoning_effort_tiers", {
            {"low",    4032},
            {"medium", 16128},
            {"high",   32256},
            {"x-high", 56832},
            {"max",    81408},
        }},
        {"notes", "test card"},
    };
    ServerConfig cfg = make_props_config_with_sidecar(sidecar);
    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);
    TEST_ASSERT(body.contains("model_card"));
    TEST_ASSERT(!body["model_card"].is_null());
    // `source` is the upstream URL, NOT the filepath. The filepath label
    // moved to budget_envelope.model_card_source post-refactor.
    TEST_ASSERT(body["model_card"]["source"].get<std::string>() ==
                "https://huggingface.co/Qwen/Qwen3.6-27B");
    TEST_ASSERT(body["model_card"]["name"].get<std::string>() == "Qwen3.6 27B");
    TEST_ASSERT(body["model_card"]["max_tokens"].get<int>() == 32768);
    TEST_ASSERT(body["model_card"]["complex_problem_max_tokens"].get<int>() == 81920);
    TEST_ASSERT(body["model_card"].contains("sampling"));
    TEST_ASSERT(body["model_card"].contains("reasoning_effort_tiers"));
    TEST_ASSERT(body["model_card"]["notes"].get<std::string>() == "test card");
    // The pre-refactor `think_max_tokens` / `hard_limit_reply_budget`
    // keys are NOT in the wholesale shape — they moved to budget_envelope.
    TEST_ASSERT(!body["model_card"].contains("think_max_tokens"));
    TEST_ASSERT(!body["model_card"].contains("hard_limit_reply_budget"));
}

TEST_CASE(ServerUnitFixture, test_props_model_card_null_on_family_fallback) {
    // When family or hard fallback was used (no sidecar), /props.model_card
    // is JSON null. The budget_envelope still carries the resolved values.
    ServerConfig cfg;
    cfg.arch                    = "qwen35";
    cfg.model_card_source_label = "family:qwen35";
    cfg.model_card_json         = nullptr;  // no sidecar parsed
    cfg.default_max_tokens      = 32768;
    cfg.hard_limit_reply_budget = 512;
    cfg.think_max_tokens        = 32256;
    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body.contains("model_card"));
    TEST_ASSERT(body["model_card"].is_null());
    // budget_envelope still present and carries the family-fallback label.
    TEST_ASSERT(body.contains("budget_envelope"));
    TEST_ASSERT(body["budget_envelope"]["model_card_source"].get<std::string>() ==
                "family:qwen35");
    TEST_ASSERT(body["budget_envelope"]["default_max_tokens"].get<int>() == 32768);
}

TEST_CASE(ServerUnitFixture, test_props_deepseek4_tool_capability) {
    ServerConfig cfg;
    cfg.arch = "deepseek4";
    Tokenizer tok;
    PrefixCache pc(0, tok);
    ToolMemory tm;
    const json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body["capabilities"]["tools_supported"].get<bool>());
}

TEST_CASE(ServerUnitFixture, test_props_deepseek4_reasoning_capability) {
    ServerConfig cfg;
    cfg.arch = "deepseek4";
    Tokenizer tok;
    PrefixCache pc(0, tok);
    ToolMemory tm;
    const json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body["reasoning"]["supported"].get<bool>());
    TEST_ASSERT(body["reasoning"]["supported_efforts"] ==
                json::array({"low", "high", "max"}));
    TEST_ASSERT(body["capabilities"]["reasoning_supported"].get<bool>());
}

TEST_CASE(ServerUnitFixture, test_props_budget_envelope_shape) {
    // budget_envelope is always present with all five fields and the
    // expected effort_tiers vocabulary (low|medium|high|x-high|max).
    // Values mirror ServerConfig regardless of what the sidecar carried.
    json sidecar = {
        {"name",        "Qwen3.6 27B"},
        {"source",      "https://huggingface.co/Qwen/Qwen3.6-27B"},
        {"verified_at", "2026-05-23"},
        {"max_tokens",  32768},
    };
    ServerConfig cfg = make_props_config_with_sidecar(sidecar);
    // Simulate CLI override: budget_envelope reflects the runtime value,
    // which may diverge from the sidecar (here, 16000 != sidecar 32768).
    cfg.default_max_tokens      = 16000;
    cfg.hard_limit_reply_budget = 512;
    cfg.think_max_tokens        = 15488;
    cfg.effort_tiers.low    = 100;
    cfg.effort_tiers.medium = 200;
    cfg.effort_tiers.high   = 300;
    cfg.effort_tiers.x_high = 400;
    cfg.effort_tiers.max    = 500;

    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body.contains("budget_envelope"));
    const json & be = body["budget_envelope"];
    TEST_ASSERT(be["model_card_source"].get<std::string>() ==
                "share/model_cards/qwen3.6-27b.json");
    TEST_ASSERT(be["default_max_tokens"].get<int>() == 16000);
    TEST_ASSERT(be["hard_limit_reply_budget"].get<int>() == 512);
    TEST_ASSERT(be["think_max_tokens"].get<int>() == 15488);
    TEST_ASSERT(be["effort_tiers"]["low"].get<int>()    == 100);
    TEST_ASSERT(be["effort_tiers"]["medium"].get<int>() == 200);
    TEST_ASSERT(be["effort_tiers"]["high"].get<int>()   == 300);
    TEST_ASSERT(be["effort_tiers"]["x-high"].get<int>() == 400);
    TEST_ASSERT(be["effort_tiers"]["max"].get<int>()    == 500);

    // Sanity: budget_envelope can diverge from model_card.max_tokens
    // (CLI override case). Verifies the two sections aren't a tautology.
    TEST_ASSERT(body["model_card"]["max_tokens"].get<int>() == 32768);
    TEST_ASSERT(be["default_max_tokens"].get<int>() == 16000);

    // Sanity: props_schema bumped to 2 (breaking change).
    TEST_ASSERT(body["server"]["props_schema"].get<int>() == 2);
}

// ─── /props.runtime captures full config (§4.16) ──────────────────────
// Snapshot/bench tooling reads /props.runtime wholesale into
// result.json.server_info; this test pins the field set so additions
// elsewhere don't accidentally drop a knob we depend on for forensics.
TEST_CASE(ServerUnitFixture, test_props_runtime_shape) {
    ServerConfig cfg = make_props_config_with_sidecar(json{
        {"name", "Qwen3.6 27B"},
        {"source", "https://huggingface.co/Qwen/Qwen3.6-27B"},
        {"verified_at", "2026-05-23"},
        {"max_tokens", 32768},
    });
    cfg.runtime_backend = "cuda";
    cfg.fa_window       = 2048;
    cfg.kv_cache_k      = "tq3_0";
    cfg.kv_cache_v      = "tq3_0";
    cfg.lazy_draft      = false;
    cfg.draft_residency = DraftResidencyPolicy::Persistent;
    cfg.target_sharding = false;
    cfg.chunk           = 512;
    cfg.target_device   = "auto:0";
    cfg.draft_device    = "auto:0";
    TEST_ASSERT(cfg.admission_coalesce_ms == 20);

    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body.contains("runtime"));
    const json & rt = body["runtime"];
    TEST_ASSERT(rt["backend"].get<std::string>()         == "cuda");
    TEST_ASSERT(rt["fa_window"].get<int>()               == 2048);
    TEST_ASSERT(rt["kv_cache_k"].get<std::string>()      == "tq3_0");
    TEST_ASSERT(rt["kv_cache_v"].get<std::string>()      == "tq3_0");
    TEST_ASSERT(rt["lazy_draft"].get<bool>()             == false);
    TEST_ASSERT(rt["draft_residency"].get<std::string>() == "persistent");
    TEST_ASSERT(rt["target_sharding"].get<bool>()        == false);
    TEST_ASSERT(rt["chunk"].get<int>()                   == 512);
    TEST_ASSERT(rt["target_device"].get<std::string>()   == "auto:0");
    TEST_ASSERT(rt["draft_device"].get<std::string>()    == "auto:0");
    TEST_ASSERT(rt["continuous_batching"]["admission_coalesce_ms"]
                    .get<int>() == 20);
    TEST_ASSERT(body["pflash"]["draft_residency"].get<std::string>() == "persistent");

    // draft_device is null when no draft model is loaded.
    cfg.draft_device.clear();
    cfg.admission_coalesce_ms = 7;
    body = build_props_body(cfg, pc, tm);
    TEST_ASSERT(body["runtime"]["draft_device"].is_null());
    TEST_ASSERT(body["runtime"]["continuous_batching"]
                    ["admission_coalesce_ms"].get<int>() == 7);
}

TEST_CASE(ServerUnitFixture, test_model_card_family_fallback_bailingmoe3) {
    auto card = dflash::common::resolve_model_card("", "", "bailingmoe3", "");
    TEST_ASSERT(card.source_label == "family:bailingmoe3");
    TEST_ASSERT(card.max_tokens == 32768);
    TEST_ASSERT(card.sampling.has_temperature);
    TEST_ASSERT(std::abs(card.sampling.temperature - 0.6f) < 1.0e-6f);
    TEST_ASSERT(card.sampling.has_top_p);
    TEST_ASSERT(std::abs(card.sampling.top_p - 0.95f) < 1.0e-6f);
    TEST_ASSERT(card.sampling.has_top_k);
    TEST_ASSERT(card.sampling.top_k == 20);
}
