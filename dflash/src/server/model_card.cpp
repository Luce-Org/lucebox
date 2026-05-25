// Model card resolution. See model_card.h and docs/specs/thinking-budget.md §3.

#include "model_card.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>

namespace dflash::common {

using json = nlohmann::json;

// ── Helpers ─────────────────────────────────────────────────────────────

std::string normalize_model_card_stem(const std::string & general_name) {
    std::string out;
    out.reserve(general_name.size());
    for (char c : general_name) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (uc == ' ' || uc == '\t' || uc == '_') {
            out.push_back('-');
        } else if (uc >= 'A' && uc <= 'Z') {
            out.push_back(static_cast<char>(uc - 'A' + 'a'));
        } else if ((uc >= 'a' && uc <= 'z') ||
                   (uc >= '0' && uc <= '9') ||
                   uc == '.' || uc == '-') {
            out.push_back(static_cast<char>(uc));
        }
        // else: silently drop punctuation
    }
    return out;
}

static bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static std::string self_bin_dir() {
    char buf[4096];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    std::string path(buf);
    auto slash = path.find_last_of('/');
    if (slash == std::string::npos) return {};
    return path.substr(0, slash);
}

// Find share/model_cards/ directory. Search order (spec §1 implementation note):
//   (a) `repo_root_hint`/share/model_cards/ (if hint non-empty)
//   (b) <binary's parent dir>/../share/model_cards/  (install layout)
//   (c) <binary's parent dir>/share/model_cards/     (build layout)
//   (d) ./share/model_cards/                          (cwd, dev runs)
//   (e) $DFLASH_MODEL_CARDS_DIR                       (explicit override)
//
// Returns first hit or empty string. Each candidate is logged to stderr so
// operators can see which path was probed.
static std::string find_model_cards_dir(const std::string & repo_root_hint) {
    std::vector<std::string> candidates;
    if (!repo_root_hint.empty()) {
        candidates.push_back(repo_root_hint + "/share/model_cards");
    }
    std::string bd = self_bin_dir();
    if (!bd.empty()) {
        candidates.push_back(bd + "/../share/model_cards");
        candidates.push_back(bd + "/share/model_cards");
    }
    candidates.push_back("share/model_cards");
    if (const char * envp = std::getenv("DFLASH_MODEL_CARDS_DIR")) {
        candidates.push_back(envp);
    }

    for (const auto & c : candidates) {
        struct stat st{};
        if (::stat(c.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            std::fprintf(stderr, "[model_card] using cards dir: %s\n", c.c_str());
            return c;
        }
    }
    std::fprintf(stderr,
        "[model_card] no share/model_cards/ directory found; "
        "tried %zu candidate(s)\n", candidates.size());
    return {};
}

// Compute the effort tiers from max_tokens + complex_problem_max_tokens
// using the spec §3.3 formula. `think_max` and `complex_think_max` are
// derived in caller.
static EffortTiers compute_default_tiers(int think_max, int complex_think_max) {
    EffortTiers t;
    t.low    = (int)(think_max * 0.125 + 0.5);
    t.medium = (int)(think_max * 0.5   + 0.5);
    t.high   = think_max;
    // x-high midpoint of think_max and complex_think_max
    t.x_high = (think_max + complex_think_max) / 2;
    t.max    = complex_think_max;
    return t;
}

// Apply spec §3.5 monotone-ordering invariant. Clamps to monotone
// non-decreasing order and warns if the sidecar (or computed tiers)
// violate the invariant.
//
// The absolute-ceiling invariant (max ≤ max_ctx − hard_limit_reply_budget)
// is enforced separately in server_main.cpp once max_ctx has been
// resolved from the backend / CLI — model_card resolution runs before
// that, and the card itself doesn't know the operator's runtime ceiling.
static void enforce_tier_invariants(EffortTiers & t,
                                    const std::string & source) {
    auto clamp_one = [&](int prev, int & v, const char * tier) {
        if (v < prev) {
            std::fprintf(stderr,
                "[model_card] %s: effort_tiers.%s=%d < previous tier %d; "
                "clamping up\n", source.c_str(), tier, v, prev);
            v = prev;
        }
    };
    clamp_one(t.low,    t.medium, "medium");
    clamp_one(t.medium, t.high,   "high");
    clamp_one(t.high,   t.x_high, "x-high");
    clamp_one(t.x_high, t.max,    "max");
}

// ── Sidecar parsing ─────────────────────────────────────────────────────

static bool load_sidecar(const std::string & path, ModelCard & out, std::string & err) {
    std::ifstream f(path);
    if (!f.is_open()) {
        err = std::string("open: ") + std::strerror(errno);
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    json j;
    try {
        j = json::parse(ss.str());
    } catch (const std::exception & e) {
        err = std::string("parse: ") + e.what();
        return false;
    }

    // Stash the parsed sidecar verbatim on the ModelCard so the HTTP
    // server can re-emit it under /props.model_card. See spec §4.9.
    out.raw_json = j;

    // Schema sanity check — the four required fields per
    // share/model_cards/_schema.json. We warn but DO NOT fail-start;
    // operators may have a partial card (e.g. only max_tokens) and the
    // family / hard fallback paths still want a chance to fill in the
    // rest. The JSON Schema at share/model_cards/_schema.json catches
    // typos earlier (CI / author-facing validation).
    static const char * const kRequiredFields[] = {
        "name", "source", "verified_at", "max_tokens"
    };
    for (const char * field : kRequiredFields) {
        if (!j.contains(field)) {
            std::fprintf(stderr,
                "[model_card] %s: missing required field '%s' "
                "(see share/model_cards/_schema.json)\n",
                path.c_str(), field);
        }
    }

    if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) {
        out.max_tokens = j["max_tokens"].get<int>();
    }
    if (j.contains("complex_problem_max_tokens") &&
        j["complex_problem_max_tokens"].is_number_integer()) {
        out.complex_problem_max_tokens = j["complex_problem_max_tokens"].get<int>();
    }
    if (j.contains("hard_limit_reply_budget") &&
        j["hard_limit_reply_budget"].is_number_integer()) {
        // Per-model override of the post-`</think>` reserved reply budget.
        // Verbose-post-close models (Qwen3.6) want 2k-4k; terse models
        // (DeepSeek-V4-flash style) stay at the 512 ds4_eval.c default.
        // See docs/specs/thinking-budget.md §3.3.
        out.hard_limit_reply_budget = j["hard_limit_reply_budget"].get<int>();
    }
    if (j.contains("soft_limit_reply_budget") &&
        j["soft_limit_reply_budget"].is_number_integer()) {
        // Soft-limit negotiated close window (spec §5.3). When set,
        // ports ds4_eval.c:3030's two-tier strategy: in the window
        // (hard_limit_reply_budget, soft_limit_reply_budget], the AR
        // loop peeks the next-token logits — if `</think>` is in
        // top-K, accept it as a natural close instead of unilateral
        // override. Default 0 (disabled, hard-only path).
        out.soft_limit_reply_budget = j["soft_limit_reply_budget"].get<int>();
    }
    if (j.contains("soft_limit_close_rank") &&
        j["soft_limit_close_rank"].is_number_integer()) {
        out.soft_limit_close_rank = j["soft_limit_close_rank"].get<int>();
    }
    if (j.contains("thinking_preamble") &&
        j["thinking_preamble"].is_string()) {
        out.thinking_preamble = j["thinking_preamble"].get<std::string>();
    }
    if (j.contains("thinking_preamble_format") &&
        j["thinking_preamble_format"].is_string()) {
        out.thinking_preamble_format = j["thinking_preamble_format"].get<std::string>();
    }

    if (j.contains("sampling") && j["sampling"].is_object()) {
        const auto & s = j["sampling"];
        auto pick_f = [&](const char * k, float & v, bool & has) {
            if (s.contains(k) && s[k].is_number()) {
                v = s[k].get<float>(); has = true;
            }
        };
        auto pick_i = [&](const char * k, int & v, bool & has) {
            if (s.contains(k) && s[k].is_number_integer()) {
                v = s[k].get<int>(); has = true;
            }
        };
        pick_f("temperature",        out.sampling.temperature,        out.sampling.has_temperature);
        pick_f("top_p",              out.sampling.top_p,              out.sampling.has_top_p);
        pick_i("top_k",              out.sampling.top_k,              out.sampling.has_top_k);
        pick_f("min_p",              out.sampling.min_p,              out.sampling.has_min_p);
        pick_f("presence_penalty",   out.sampling.presence_penalty,   out.sampling.has_presence_penalty);
        pick_f("repetition_penalty", out.sampling.repetition_penalty, out.sampling.has_repetition_penalty);
    }

    if (j.contains("reasoning_effort_tiers") && j["reasoning_effort_tiers"].is_object()) {
        const auto & rt = j["reasoning_effort_tiers"];
        auto pick = [&](const char * k, int & v) {
            if (rt.contains(k) && rt[k].is_number_integer()) v = rt[k].get<int>();
        };
        pick("low",    out.effort_tiers.low);
        pick("medium", out.effort_tiers.medium);
        pick("high",   out.effort_tiers.high);
        pick("x-high", out.effort_tiers.x_high);
        pick("max",    out.effort_tiers.max);
    }

    return true;
}

// ── Per-family fallback table ───────────────────────────────────────────

static bool family_fallback(const std::string & arch, ModelCard & out) {
    // Coarse safety net when no sidecar matches. Values are conservative
    // and intentionally not aspirational — operators are expected to ship
    // a sidecar for production models. See spec §3.1.
    if (arch == "qwen35" || arch == "qwen36" || arch == "qwen3") {
        out.max_tokens                 = 32768;
        out.complex_problem_max_tokens = 0;
        // Qwen3.x is verbose post-`</think>` — restates derivation in the
        // visible area before writing the answer line. The 512 default
        // from ds4_eval.c (DeepSeek terse-style) clips this pattern. See
        // docs/specs/thinking-budget.md §3.3.
        out.hard_limit_reply_budget    = 4096;
        out.source_label = "family:" + arch;
        return true;
    }
    if (arch == "gemma4") {
        // Gemma4 verified value: see Gemma model card; conservative
        // 16384 keeps us inside published recommendations.
        out.max_tokens                 = 16384;
        out.complex_problem_max_tokens = 0;
        out.source_label = "family:gemma4";
        return true;
    }
    if (arch == "laguna") {
        // Laguna (DeepSeek-V3-derivative) — same conservative ceiling
        // as the Qwen family until a verified card lands.
        out.max_tokens                 = 32768;
        out.complex_problem_max_tokens = 0;
        out.source_label = "family:laguna";
        return true;
    }
    return false;
}

// ── Public entry point ──────────────────────────────────────────────────

ModelCard resolve_model_card(const std::string & gguf_path,
                             const std::string & general_name,
                             const std::string & general_architecture,
                             const std::string & repo_root_hint) {
    (void)gguf_path;  // currently unused; kept in signature for future cards
                     // keyed by file hash or path-based overrides.

    ModelCard card;

    // Try sidecar first.
    bool resolved = false;
    if (!general_name.empty()) {
        std::string stem = normalize_model_card_stem(general_name);
        std::string dir  = find_model_cards_dir(repo_root_hint);
        if (!dir.empty() && !stem.empty()) {
            std::string path = dir + "/" + stem + ".json";
            std::fprintf(stderr,
                "[model_card] probing sidecar: %s (from general.name='%s')\n",
                path.c_str(), general_name.c_str());
            if (file_exists(path)) {
                std::string err;
                ModelCard sidecar;
                if (load_sidecar(path, sidecar, err)) {
                    sidecar.source_label = path;
                    card = sidecar;
                    resolved = true;
                } else {
                    std::fprintf(stderr,
                        "[model_card] sidecar parse failed (%s): %s — "
                        "falling through\n", path.c_str(), err.c_str());
                }
            } else {
                std::fprintf(stderr,
                    "[model_card] sidecar not found at %s\n", path.c_str());
            }
        }
    }

    // Family fallback.
    if (!resolved) {
        if (family_fallback(general_architecture, card)) {
            std::fprintf(stderr,
                "[model_card] using family fallback for arch='%s'\n",
                general_architecture.c_str());
            resolved = true;
        }
    }

    // Hard fallback.
    if (!resolved) {
        std::fprintf(stderr,
            "[model_card] using hard fallback (no sidecar, no family match "
            "for arch='%s')\n", general_architecture.c_str());
        card.source_label = "hard-fallback";
        card.max_tokens   = 16000;
        card.complex_problem_max_tokens = 0;
    }

    // Derive think_max_tokens and missing tier values.
    if (card.hard_limit_reply_budget < 0) card.hard_limit_reply_budget = 0;
    card.think_max_tokens = std::max(0, card.max_tokens - card.hard_limit_reply_budget);

    int complex_think_max = card.complex_problem_max_tokens > 0
        ? std::max(0, card.complex_problem_max_tokens - card.hard_limit_reply_budget)
        : card.think_max_tokens;

    // For each tier not explicitly set, fill via §3.3 formula.
    EffortTiers computed = compute_default_tiers(card.think_max_tokens, complex_think_max);
    if (card.effort_tiers.low    <= 0) card.effort_tiers.low    = computed.low;
    if (card.effort_tiers.medium <= 0) card.effort_tiers.medium = computed.medium;
    if (card.effort_tiers.high   <= 0) card.effort_tiers.high   = computed.high;
    if (card.effort_tiers.x_high <= 0) card.effort_tiers.x_high = computed.x_high;
    if (card.effort_tiers.max    <= 0) card.effort_tiers.max    = computed.max;

    // If complex_problem_max_tokens is unspecified, collapse x-high and max
    // to high (spec §3.3 last paragraph).
    if (card.complex_problem_max_tokens <= 0) {
        if (card.effort_tiers.x_high > card.effort_tiers.high) card.effort_tiers.x_high = card.effort_tiers.high;
        if (card.effort_tiers.max    > card.effort_tiers.high) card.effort_tiers.max    = card.effort_tiers.high;
    }

    // Enforce monotone non-decreasing tiers. The absolute ceiling
    // (max ≤ max_ctx − hard_limit_reply_budget) is applied later in
    // server_main once max_ctx has been resolved.
    enforce_tier_invariants(card.effort_tiers, card.source_label);

    return card;
}

}  // namespace dflash::common
