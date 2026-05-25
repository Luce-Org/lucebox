// Model card resolution for thinking-budget v2.
//
// At server startup, look up the loaded GGUF's recommended defaults from
// (in order): a JSON sidecar at share/model_cards/<normalized-name>.json,
// a per-family fallback table, or the hard fallback (antirez/ds4 reference
// values). See docs/specs/thinking-budget.md §3 for the full resolution
// order and field semantics.
//
// The resolved ModelCard is consumed by server_main.cpp, which copies the
// values into ServerConfig but only for fields the operator did NOT
// override via CLI. CLI flags always win.

#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace dflash::common {

// Phase-1 reasoning budgets per `reasoning.effort` tier.
// See spec §3.3, §4.2.
struct EffortTiers {
    int low    = 0;
    int medium = 0;
    int high   = 0;
    int x_high = 0;
    int max    = 0;
};

// Sampler defaults from the model card. Each field's `has_*` companion
// records whether the sidecar actually supplied a value, so the request
// parser can know to fall back to its hard-coded default vs. apply this
// one when the request omits the field.
struct SamplingDefaults {
    float temperature        = 1.0f;
    float top_p              = 0.95f;
    int   top_k              = 20;
    float min_p              = 0.0f;
    float presence_penalty   = 0.0f;
    float repetition_penalty = 1.0f;

    bool  has_temperature        = false;
    bool  has_top_p              = false;
    bool  has_top_k              = false;
    bool  has_min_p              = false;
    bool  has_presence_penalty   = false;
    bool  has_repetition_penalty = false;
};

// Resolved model card. `source_label` is a short, operator-facing tag
// describing where each value came from; useful for the startup banner.
struct ModelCard {
    // One of: "share/model_cards/<file>.json", "family:<arch>", "hard-fallback".
    std::string source_label;

    int               max_tokens                 = 16000;  // spec §3.4 hard fallback
    int               complex_problem_max_tokens = 0;      // 0 = not specified
    SamplingDefaults  sampling;
    EffortTiers       effort_tiers;
    int               hard_limit_reply_budget    = 512;
    // Soft-limit negotiated close window (spec §5.3). 0 = disabled
    // (hard-only behavior). Operator can set per-model via sidecar.
    int               soft_limit_reply_budget    = 0;
    int               soft_limit_close_rank      = 8;
    // Thinking-budget preamble (spec §3.3 / §5.4). Template string
    // with `{think_max}` and `{reply_max}` placeholders. Empty =
    // no preamble injected.
    std::string       thinking_preamble;
    std::string       thinking_preamble_format;

    // Phase-1 ceiling derived from `max_tokens - hard_limit_reply_budget`.
    // Convenience: also the spec's `think_max` quantity (§3.3 formula).
    int think_max_tokens = 15488;

    // Raw parsed sidecar JSON, populated on successful sidecar load.
    // Null (`raw_json.is_null() == true`) when family fallback or hard
    // fallback was used. Exposed verbatim under `/props.model_card`
    // (see docs/specs/props-endpoint.md §4.9).
    nlohmann::json raw_json = nullptr;
};

// Normalize a GGUF `general.name` value to a model-card filename stem.
// "Qwen3.6 27B" -> "qwen3.6-27b". Lowercases, replaces spaces with `-`,
// strips characters outside `[a-z0-9.-]`. Exposed for tests/banner.
std::string normalize_model_card_stem(const std::string & general_name);

// Resolve the model card for the loaded GGUF.
//
// Search order (spec §3.1):
//   1. share/model_cards/<normalize(general_name)>.json
//   2. Per-family fallback table keyed on general_architecture
//   3. Hard fallback (antirez/ds4 reference values)
//
// `repo_root_hint` is an optional explicit directory to search for
// share/model_cards/; pass empty to use auto-discovery (binary's parent,
// then cwd, then $DFLASH_MODEL_CARDS_DIR).
ModelCard resolve_model_card(const std::string & gguf_path,
                             const std::string & general_name,
                             const std::string & general_architecture,
                             const std::string & repo_root_hint = "");

}  // namespace dflash::common
