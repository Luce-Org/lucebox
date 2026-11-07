// PFlash scoring pipeline glue. See pflash_compress.h.

#include "pflash_compress.h"

#include "pflash_fold.h"
#include "pflash_request_probe.h"
#include "pflash_query_candidates.h"
#include "pflash_selection.h"
#include "internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

int env_int(const char * name, int fallback) {
    if (const char * v = std::getenv(name)) {
        int x = std::atoi(v);
        if (x >= 0) return x;
    }
    return fallback;
}

float env_float(const char * name, float def) {
    if (const char * v = std::getenv(name)) {
        try { return std::stof(v); } catch (...) {}
    }
    return def;
}

void force_chunk_neighborhood(std::vector<uint8_t> & forced, int n_chunks,
                                     int chunk, int radius) {
    int lo = std::max(0, chunk - radius);
    int hi = std::min(n_chunks - 1, chunk + radius);
    for (int c = lo; c <= hi; ++c) forced[(size_t)c] = 1;
}

namespace {

void write_float_or_null(FILE * file, double value) {
    if (std::isfinite(value)) std::fprintf(file, "%.9g", value);
    else std::fputs("null", file);
}

void write_spans(FILE * file, const std::vector<PFlashTokenSpan> & spans) {
    std::fputc('[', file);
    for (size_t index = 0; index < spans.size(); ++index) {
        std::fprintf(file, "%s[%d,%d]", index ? "," : "", spans[index].begin, spans[index].end);
    }
    std::fputc(']', file);
}

// ``"request_probe":{...}``: what the request probe read, chose and charged
// (PFLASH_SELECT_QUERY_PROBE=1 only; absent otherwise).
void write_request_probe_trace(FILE * file, const luce::pflash::PFlashRequestProbeTrace & t) {
    std::fputs(",\"request_probe\":{\"tau\":", file);
    write_float_or_null(file, t.tau);
    std::fprintf(file, ",\"stub\":%s,\"fallback\":", t.stub ? "true" : "false");
    if (t.fallback.empty()) {
        std::fputs("null", file);
    } else {
        std::fputc('"', file);
        for (const char c : t.fallback) {
            if (c == '"' || c == '\\') std::fputc('\\', file);
            if ((unsigned char) c >= 0x20) std::fputc(c, file);
        }
        std::fputc('"', file);
    }
    std::fprintf(file,
        ",\"query_tokens\":%d,\"rows_new\":%d,\"rows_stored\":%d,"
        "\"previous_used\":%s,\"latest_tokens\":%d,\"previous_tokens\":%d,"
        "\"uncapped_tokens\":%d,\"capped\":%s,\"pinned_new_tokens\":%d,"
        "\"mandatory_tokens\":%d,\"mandatory_tokens_base\":%d,\"token_budget\":%d,"
        "\"query_tiles\":%d,\"store\":{\"turns\":%d,\"tokens\":%d,"
        "\"evicted_turns\":%d,\"invalidated_turns\":%d}",
        t.query_tokens, t.rows_new, t.rows_stored,
        t.set.previous_used ? "true" : "false", t.set.latest_tokens, t.set.previous_tokens,
        t.set.uncapped_tokens, t.set.capped ? "true" : "false", t.pinned_new_tokens,
        t.mandatory_tokens, t.mandatory_tokens_base, t.token_budget, t.query_tiles,
        t.store_turns, t.store_tokens, t.store_evicted_turns, t.store_invalidated_turns);
    if (t.policy_evaluated) {
        std::fputs(",\"abstain\":{\"reason\":", file);
        if (t.abstain.reason.empty()) std::fputs("null", file);
        else std::fprintf(file, "\"%s\"", t.abstain.reason.c_str());
        std::fputs(",\"mean_margin\":", file); write_float_or_null(file, t.abstain.mean_margin);
        std::fputs(",\"max_margin\":", file); write_float_or_null(file, t.abstain.max_margin);
        std::fputs(",\"latest_share\":", file); write_float_or_null(file, t.abstain.latest_share);
        std::fprintf(file, ",\"runs\":%d,\"prompt_tokens\":%d}", t.abstain.runs, t.abstain.prompt_tokens);
    }
    if (t.recency_pin.begin >= 0) {
        std::fprintf(file, ",\"recency_pin\":[%d,%d],\"recency_pin_new_tokens\":%d",
                     t.recency_pin.begin, t.recency_pin.end, t.recency_pin_new_tokens);
    } else {
        std::fputs(",\"recency_pin\":null", file);
    }
    std::fputs(",\"query_spans\":", file);
    write_spans(file, luce::pflash::pflash_request_token_spans(t.set.tokens));
    std::fputs(",\"messages\":[", file);
    for (size_t index = 0; index < t.set.messages.size(); ++index) {
        const auto & m = t.set.messages[index];
        std::fprintf(file, "%s{\"span\":[%d,%d],\"tokens\":%d,\"request_tokens\":%d,"
                     "\"at_or_above_tau\":%d,\"logit_min\":",
                     index ? "," : "", m.message.begin, m.message.end, m.message_tokens,
                     m.request_tokens, m.logits_at_or_above_tau);
        write_float_or_null(file, m.logit_min);
        std::fputs(",\"logit_max\":", file);
        write_float_or_null(file, m.logit_max);
        std::fputs(",\"logit_mean\":", file);
        write_float_or_null(file, m.logit_mean);
        std::fputs(",\"runs\":", file);
        write_spans(file, m.runs);
        if (index < t.message_logits.size()) {
            std::fputs(",\"logits\":[", file);
            const auto & logits = t.message_logits[index];
            for (size_t k = 0; k < logits.size(); ++k) {
                if (k) std::fputc(',', file);
                write_float_or_null(file, logits[k]);
            }
            std::fputc(']', file);
        }
        std::fputc('}', file);
    }
    std::fputc(']', file);
    // Candidate rule (human + every non-system message): only for a GGUF with that rule.
    if (t.set.turns_read) {
        std::fprintf(file, ",\"candidate_rule\":\"%s\",\"turn_tokens\":%d,\"turns\":[",
                     t.set.rule.empty() ? luce::pflash::kRequestCandidateRule : t.set.rule.c_str(),
                     t.set.turn_tokens);
        for (size_t index = 0; index < t.set.turns.size(); ++index) {
            const auto & m = t.set.turns[index];
            const int message = index < t.set.turn_messages.size() ? t.set.turn_messages[index] : -1;
            std::fprintf(file, "%s{\"message\":%d,\"span\":[%d,%d],\"tokens\":%d,\"request_tokens\":%d,\"runs\":",
                         index ? "," : "", message, m.message.begin, m.message.end, m.message_tokens,
                         m.request_tokens);
            write_spans(file, m.runs);
            std::fputc('}', file);
        }
        std::fputc(']', file);
    }
    // Stage 2 (STAGE2-CONTRACT.md section 5): only under the cand_v5 rule.
    const auto & s2 = t.stage2;
    if (s2.evaluated) {
        std::fprintf(file, ",\"stage2\":{\"active\":%s,\"reason\":", s2.active ? "true" : "false");
        if (s2.reason.empty()) {
            std::fputs("null", file);
        } else {
            std::fputc('"', file);
            for (const char c : s2.reason) {
                if (c == '"' || c == '\\') std::fputc('\\', file);
                if ((unsigned char) c >= 0x20) std::fputc(c, file);
            }
            std::fputc('"', file);
        }
        std::fprintf(file, ",\"post_reserve\":%d,\"seed_tokens\":%d,\"seed_capped\":%s,\"latest_tokens\":%d,"
                     "\"lookback_logit\":", s2.post_reserve, s2.seed_tokens, s2.seed_capped ? "true" : "false",
                     s2.latest_tokens);
        if (s2.active) write_float_or_null(file, s2.lookback_logit); else std::fputs("null", file);
        std::fputs(",\"vague_logit\":", file);
        if (s2.active) write_float_or_null(file, s2.vague_logit); else std::fputs("null", file);
        std::fprintf(file,
            ",\"lookback\":%s,\"earlier_tokens\":%d,\"promoted_tokens\":%d,\"promoted_capped\":%s,"
            "\"promoted_runs\":%d,\"post_tokens\":%d,\"widened\":%s,\"budget\":%d,\"budget_widened\":%d,"
            "\"raw_fits\":%s,\"s2_computed\":%s,\"latest_message\":%d,\"latest_span\":[%d,%d],"
            "\"resume\":%d,\"session_resume\":\"disabled_by_stage2\",\"history_dropped\":%d,"
            "\"attention_tiles\":%d,\"pass1_us\":%lld,\"pass2_us\":%lld,\"fallback\":",
            s2.lookback ? "true" : "false", s2.earlier_tokens, s2.promoted_tokens,
            s2.promoted_capped ? "true" : "false", s2.promoted_runs, s2.post_tokens,
            s2.widened ? "true" : "false", s2.budget, s2.budget_widened, s2.raw_fits ? "true" : "false",
            s2.s2_computed ? "true" : "false", s2.latest_message, s2.latest_span.begin, s2.latest_span.end,
            s2.resume, s2.history_dropped, s2.attention_tiles, (long long) s2.pass1_us, (long long) s2.pass2_us);
        if (s2.fallback.empty()) {
            std::fputs("null", file);
        } else {
            std::fputc('"', file);
            for (const char c : s2.fallback) {
                if (c == '"' || c == '\\') std::fputc('\\', file);
                if ((unsigned char) c >= 0x20) std::fputc(c, file);
            }
            std::fputc('"', file);
        }
        std::fprintf(file, ",\"fallback_count\":%ld}", s2.fallback_count);
    }
    // Query step 1 (PFLASH_QUERY_CANDIDATES / PFLASH_QUERY_PIN): only when one of them is on.
    if (t.query && t.query->evaluated) {
        const auto & q = *t.query;
        const auto & c = q.config;
        std::fprintf(file, ",\"query\":{\"candidates_k\":%d,\"unit_score\":\"%s\",\"aggregate\":\"%s\","
                     "\"temperature\":", c.candidates, luce::pflash::pflash_unit_score_name(c.unit_score),
                     luce::pflash::pflash_query_aggregate_name(c.aggregate));
        write_float_or_null(file, c.temperature);
        std::fputs(",\"extent_margin\":", file);
        if (c.extent_gap) std::fputs("\"gap\"", file); else write_float_or_null(file, c.extent_margin);
        std::fputs(",\"null_margin\":", file);
        write_float_or_null(file, c.null_margin);
        std::fprintf(file, ",\"pin_cap\":%d,\"pin_margin\":", c.pin_tokens);
        if (c.pin_gap) std::fputs("\"gap\"", file); else write_float_or_null(file, c.pin_margin);
        std::fprintf(file, ",\"units\":%d,\"system_used\":%s,\"latest\":%d,\"latest_units\":%d,\"latest_top\":",
                     q.units, q.system_used ? "true" : "false", q.set.latest, q.set.latest_units);
        write_float_or_null(file, q.set.latest_top);
        std::fputs(",\"latest_z\":", file);
        write_float_or_null(file, q.set.latest_z);
        std::fprintf(file, ",\"latest_null\":%s,\"candidates\":[", q.set.latest_null ? "true" : "false");
        for (size_t i = 0; i < q.set.candidates.size(); ++i) {
            const auto & cand = q.set.candidates[i];
            std::fprintf(file, "%s{\"message\":%d,\"units\":%zu,\"query_spans\":", i ? "," : "", cand.message,
                         cand.units.size());
            write_spans(file, luce::pflash::pflash_request_token_spans(cand.tokens));
            std::fputs(",\"score\":", file);
            write_float_or_null(file, cand.score);
            std::fputs(",\"weight\":", file);
            write_float_or_null(file, cand.weight);
            std::fputc('}', file);
        }
        std::fputs("],\"blend\":{\"probe\":[", file);
        for (size_t i = 0; i < q.blend.probe.size(); ++i) {
            if (i) std::fputc(',', file);
            write_float_or_null(file, q.blend.probe[i]);
        }
        std::fputs("],\"tw\":", file);
        write_float_or_null(file, q.blend.tw);
        std::fputs(",\"tw_score\":", file);
        write_float_or_null(file, q.tw_score);
        std::fputs(",\"end\":", file);
        write_float_or_null(file, q.blend.end);
        std::fprintf(file, "},\"tw_share\":");
        if (c.tw_score) std::fputs("\"score\"", file); else write_float_or_null(file, c.tw_share);
        std::fputs(",\"end_share\":", file);
        write_float_or_null(file, c.end_share);
        std::fprintf(file, ",\"query_tokens\":%d,\"tiles\":%d,\"pin\":", q.query_tokens, q.tiles);
        write_spans(file, q.pin.spans);
        std::fprintf(file, ",\"pin_tokens\":%d,\"pin_units\":%d,\"pin_skipped\":%d,\"pin_cap_used\":%d,"
                     "\"rank_us\":%lld,\"score_us\":%lld",
                     q.pin.tokens, q.pin.units, q.pin.skipped, q.pin_cap_used, (long long) q.rank_us,
                     (long long) q.score_us);
        if (q.sessions) {
            std::fprintf(file, ",\"session_resume\":%d,\"redo\":%s,\"redo_reason\":", q.session_resume,
                         q.redo ? "true" : "false");
            if (!q.redo) {
                std::fputs("null", file);
            } else {
                std::fputc('"', file);
                for (const char c : q.redo_reason) {
                    if (c == '"' || c == '\\') std::fputc('\\', file);
                    if ((unsigned char) c >= 0x20) std::fputc(c, file);
                }
                std::fputc('"', file);
            }
        }
        std::fputc('}', file);
    }
    std::fputc('}', file);
}

// ``"ident_query":{...}``: the identifier query (PFLASH_EVAL_REF_IDENT_QUERY).
void write_ident_query_trace(FILE * file, const luce::pflash::PFlashIdentQueryTrace & t) {
    std::fprintf(file, ",\"ident_query\":{\"mode\":\"%s\",\"fallback\":",
                 luce::pflash::pflash_ident_query_name(t.mode));
    if (t.fallback.empty()) {
        std::fputs("null", file);
    } else {
        std::fputc('"', file);
        for (const char c : t.fallback) {
            if (c == '"' || c == '\\') std::fputc('\\', file);
            if ((unsigned char) c >= 0x20) std::fputc(c, file);
        }
        std::fputc('"', file);
    }
    std::fprintf(file, ",\"turns\":%d,\"ident_tokens\":%d,\"query_tokens\":%d,\"query_tiles\":%d,"
                 "\"query_spans\":", t.turns, t.ident_tokens, t.query_tokens, t.query_tiles);
    write_spans(file, luce::pflash::pflash_request_token_spans(t.tokens));
    std::fputc('}', file);
}

} // namespace

void write_compression_trace(
        int input_tokens,
        float keep_ratio,
        int chunk_size,
        int n_lookahead,
        int pool_kernel,
        int n_keep,
        const std::vector<std::pair<float, int>> & chunk_means,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced,
        const std::vector<int32_t> & compressed_ids,
        const PFlashTraceFields * trace_fields) {
    const char * path = std::getenv("PFLASH_TRACE_PATH");
    if (!path || !*path) return;

    FILE * file = std::fopen(path, "a");
    if (!file) {
        std::fprintf(stderr, "[pflash-trace] cannot append %s\n", path);
        return;
    }

    std::vector<float> scores(selected.size(), 0.0f);
    for (const auto & chunk : chunk_means) {
        scores[(size_t)chunk.second] = chunk.first;
    }
    const bool has_exact_scores = trace_fields &&
        trace_fields->exact_chunk_scores &&
        trace_fields->exact_chunk_scores->size() == scores.size();
    if (trace_fields &&
        trace_fields->selector_mode !=
            luce::pflash::PFlashSelectionMode::Legacy &&
        !has_exact_scores) {
        std::fclose(file);
        std::fprintf(stderr, "[pflash-trace] exact strict scores unavailable\n");
        return;
    }

    std::fprintf(file,
        "{\"schema_version\":%d,\"input_tokens\":%d,\"keep_ratio\":%.9g",
        trace_fields ? 3 : 1, input_tokens, keep_ratio);
    if (trace_fields) {
        std::fputs(",\"input_ids\":[", file);
        for (size_t index = 0; index < trace_fields->input_ids->size(); ++index) {
            if (index) std::fputc(',', file);
            std::fprintf(file, "%d", (*trace_fields->input_ids)[index]);
        }
        std::fprintf(file,
            "],\"query_begin\":%d,\"query_end\":%d,"
            "\"selector_mode\":\"%s\",\"query_parser\":\"%s\","
            "\"token_budget\":%d,\"top_k\":%d,"
            "\"retained_tokens\":%d",
            trace_fields->query_begin, trace_fields->query_end,
            luce::pflash::pflash_selection_mode_name(
                trace_fields->selector_mode),
            luce::pflash::pflash_query_parser_name(trace_fields->query_parser),
            trace_fields->token_budget, trace_fields->top_k,
            trace_fields->retained_tokens);
        std::fputs(",\"required_instruction_spans\":[", file);
        if (trace_fields->required_instruction_spans) {
            for (size_t index = 0;
                 index < trace_fields->required_instruction_spans->size();
                 ++index) {
                if (index) std::fputc(',', file);
                const auto & span =
                    (*trace_fields->required_instruction_spans)[index];
                std::fprintf(file, "[%d,%d]", span.begin, span.end);
            }
        }
        std::fputc(']', file);
        if (trace_fields->selector_mode ==
            luce::pflash::PFlashSelectionMode::Legacy) {
            std::fputs(",\"stop_reason\":null,\"retained_mass\":null", file);
        } else {
            std::fprintf(file,
                ",\"stop_reason\":\"%s\",\"retained_mass\":%.17g",
                luce::pflash::pflash_selection_stop_name(trace_fields->stop),
                trace_fields->retained_mass);
        }
    }
    if (trace_fields) {
        std::fprintf(file, ",\"segmentation\":\"%s\",\"candidate_score\":\"%s\",\"scorer\":\"%s\",\"split_fraction\":%.4f",
                     trace_fields->segmentation, trace_fields->candidate_score,
                     trace_fields->scorer, trace_fields->split_fraction);
        if (trace_fields->other_chunk_scores) {
            std::fputs(",\"other_chunk_scores\":[", file);
            for (size_t index = 0; index < trace_fields->other_chunk_scores->size(); ++index) {
                const double score = (*trace_fields->other_chunk_scores)[index];
                if (index) std::fputc(',', file);
                if (std::isfinite(score)) std::fprintf(file, "%.9g", score); else std::fputs("null", file);
            }
            std::fputc(']', file);
        }
        if (trace_fields->extra_queries && !trace_fields->extra_queries->empty()) {
            std::fputs(",\"extra_queries\":[", file);
            for (size_t index = 0; index < trace_fields->extra_queries->size(); ++index) {
                const auto & span = (*trace_fields->extra_queries)[index];
                std::fprintf(file, "%s[%d,%d]", index ? "," : "", span.begin, span.end);
            }
            std::fputc(']', file);
        }
        if (trace_fields->request_probe) {
            write_request_probe_trace(file, *trace_fields->request_probe);
        }
        if (trace_fields->ident_query) {
            write_ident_query_trace(file, *trace_fields->ident_query);
        }
        if (trace_fields->kept_spans) {
            std::fprintf(file,
                ",\"strategy\":\"fold\",\"fold_code\":%s,\"fold_code_lines\":%d,"
                "\"fold_nonblank_lines\":%d,\"fold_code_blocks\":%d,"
                "\"fold_text_source\":\"%s\",\"fold_skipped\":%d,"
                "\"fold_capped\":%d,\"fold_contained\":%d",
                trace_fields->fold_code ? "true" : "false",
                trace_fields->fold_code_lines, trace_fields->fold_nonblank_lines,
                trace_fields->fold_code_blocks, trace_fields->fold_text_source,
                trace_fields->fold_skipped, trace_fields->fold_capped,
                trace_fields->fold_contained);
            const std::pair<const char *, const std::vector<PFlashTokenSpan> *> lists[] = {
                {"kept_spans", trace_fields->kept_spans},
                {"fold_blocks", trace_fields->fold_blocks},
            };
            for (const auto & [name, list] : lists) {
                if (!list) continue;
                std::fprintf(file, ",\"%s\":[", name);
                for (size_t index = 0; index < list->size(); ++index) {
                    std::fprintf(file, "%s[%d,%d]", index ? "," : "",
                                 (*list)[index].begin, (*list)[index].end);
                }
                std::fputc(']', file);
            }
        }
        if (trace_fields->segments) {
            std::fputs(",\"segments\":[", file);
            for (size_t index = 0; index < trace_fields->segments->size(); ++index) {
                const auto & span = (*trace_fields->segments)[index];
                std::fprintf(file, "%s[%d,%d]", index ? "," : "", span.begin, span.end);
            }
            std::fputc(']', file);
        }
    }
    std::fprintf(file,
        ",\"chunk_size\":%d,\"n_lookahead\":%d,\"pool_kernel\":%d,"
        "\"n_keep\":%d,\"chunk_scores\":[",
        chunk_size, n_lookahead, pool_kernel, n_keep);
    for (size_t index = 0; index < scores.size(); ++index) {
        if (index) std::fputc(',', file);
        const double score = has_exact_scores
            ? (*trace_fields->exact_chunk_scores)[index]
            : (double) scores[index];
        if (std::isfinite(score)) {
            std::fprintf(file, has_exact_scores ? "%.17g" : "%.9g", score);
        } else {
            std::fputs("null", file);
        }
    }
    std::fputs("],\"selected_chunks\":[", file);
    bool first = true;
    for (size_t index = 0; index < selected.size(); ++index) {
        if (!selected[index]) continue;
        if (!first) std::fputc(',', file);
        std::fprintf(file, "%zu", index);
        first = false;
    }
    std::fputs("],\"forced_chunks\":[", file);
    first = true;
    for (size_t index = 0; index < forced.size(); ++index) {
        if (!forced[index]) continue;
        if (!first) std::fputc(',', file);
        std::fprintf(file, "%zu", index);
        first = false;
    }
    std::fputs("],\"compressed_ids\":[", file);
    for (size_t index = 0; index < compressed_ids.size(); ++index) {
        if (index) std::fputc(',', file);
        std::fprintf(file, "%d", compressed_ids[index]);
    }
    std::fputc(']', file);
    if (trace_fields && trace_fields->spread_threshold > 0.0) {
        std::fprintf(file, ",\"spread\":%.17g,\"spread_threshold\":%.17g",
                     trace_fields->spread, trace_fields->spread_threshold);
        // JSON has no NaN or infinity: a non-finite score is null.
        std::fputs(",\"spread_first_token_score\":", file);
        if (std::isfinite(trace_fields->spread_first_token_score)) {
            std::fprintf(file, "%.17g", trace_fields->spread_first_token_score);
        } else {
            std::fputs("null", file);
        }
        std::fprintf(file, ",\"spread_per_token\":%s", trace_fields->spread_per_token ? "true" : "false");
    }
    std::fputs("}\n", file);
    std::fclose(file);
}

namespace {
thread_local std::vector<PFlashTokenSpan> g_last_kept_spans;
thread_local PFlashScoringStats g_last_scoring_stats;
thread_local std::vector<PFlashCandidateLift> g_last_candidate_lifts;
thread_local std::vector<PFlashTokenSpan> g_last_probe_pins;
thread_local PFlashStage2Outcome g_last_stage2_outcome;
thread_local PFlashRelevanceSpread g_last_relevance_spread;
} // namespace

PFlashRelevanceSpread pflash_relevance_spread(
        const std::vector<double> & scores,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced) {
    PFlashRelevanceSpread out;
    if (selected.size() != scores.size() || forced.size() != scores.size()) return out;
    double total = 0.0;
    for (const double score : scores) {
        if (!std::isfinite(score)) return out;
        total += score;
    }
    double entropy = 0.0;
    if (total > 0.0) {
        for (const double score : scores) {
            if (!(score > 0.0)) continue;
            const double p = score / total;
            entropy -= p * std::log(p);
        }
    }
    int k = 0;
    for (size_t index = 0; index < scores.size(); ++index) {
        if (selected[index] && !forced[index]) ++k;
    }
    out.valid = true;
    out.entropy = entropy;
    out.k = k;
    out.candidates = (int) scores.size();
    out.spread = std::exp(entropy) / (double) std::max(k, 1);
    return out;
}

PFlashRelevanceSpread pflash_selection_spread(
        const std::vector<double> & chunk_scores,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced,
        int first_chunk_tokens,
        double first_token_score,
        bool per_token) {
    if (chunk_scores.empty() || first_chunk_tokens <= 1 ||
        selected.size() != chunk_scores.size() || forced.size() != chunk_scores.size()) {
        return pflash_relevance_spread(chunk_scores, selected, forced);
    }
    const double first_chunk_sum = per_token
        ? chunk_scores[0] * (double) first_chunk_tokens : chunk_scores[0];
    // A non-finite sum stays non-finite, so the spread rejects it.
    const double rest_sum = std::isfinite(first_chunk_sum)
        ? std::max(0.0, first_chunk_sum - first_token_score) : first_chunk_sum;
    std::vector<double> scores{first_token_score,
        per_token ? rest_sum / (double) (first_chunk_tokens - 1) : rest_sum};
    std::vector<uint8_t> unit_selected{0, selected[0]};
    std::vector<uint8_t> unit_forced{1, forced[0]};
    scores.insert(scores.end(), chunk_scores.begin() + 1, chunk_scores.end());
    unit_selected.insert(unit_selected.end(), selected.begin() + 1, selected.end());
    unit_forced.insert(unit_forced.end(), forced.begin() + 1, forced.end());
    PFlashRelevanceSpread out = pflash_relevance_spread(scores, unit_selected, unit_forced);
    out.candidates = (int) chunk_scores.size();
    return out;
}

const PFlashRelevanceSpread & pflash_last_relevance_spread() {
    return g_last_relevance_spread;
}

const PFlashStage2Outcome & pflash_last_stage2_outcome() {
    return g_last_stage2_outcome;
}

void pflash_set_stage2_outcome(const PFlashStage2Outcome & outcome) {
    g_last_stage2_outcome = outcome;
}

const std::vector<PFlashCandidateLift> & pflash_last_candidate_lifts() {
    return g_last_candidate_lifts;
}

const std::vector<PFlashTokenSpan> & pflash_last_probe_pins() {
    return g_last_probe_pins;
}

void pflash_set_probe_pins(std::vector<PFlashTokenSpan> pins) {
    g_last_probe_pins = std::move(pins);
}

const PFlashScoringStats & pflash_last_scoring_stats() {
    return g_last_scoring_stats;
}

void pflash_set_scoring_stats(const PFlashScoringStats & stats) {
    g_last_scoring_stats = stats;
}

const std::vector<PFlashTokenSpan> & pflash_last_kept_spans() {
    return g_last_kept_spans;
}

void pflash_clear_kept_spans() {
    g_last_kept_spans.clear();
    g_last_scoring_stats = {};
    g_last_candidate_lifts.clear();
    g_last_probe_pins.clear();
    g_last_stage2_outcome = {};
    g_last_relevance_spread = {};
}

std::vector<int32_t> select_pflash_chunks(
        const std::vector<int32_t> & ids,
        const std::vector<float> & token_scores,
        float keep_ratio,
        int n_lookahead,
        int score_query_end,
        int pool_kernel,
        const luce::pflash::PFlashSelectionConfig & config,
        const std::vector<PFlashTokenSpan> & required_instruction_spans,
        bool direct_mass,
        bool write_trace,
        const std::vector<PFlashTokenSpan> * segments,
        bool density,
        const std::vector<float> * other_token_scores,
        double split_fraction,
        const Tokenizer * fold_vocab) {
    const int input_tokens = (int) ids.size();
    const int query_end = score_query_end < 0 ? input_tokens : score_query_end;
    const int query_tokens = std::min(n_lookahead, query_end);
    const int query_begin = query_end - query_tokens;
    const int selector_budget = (int) std::floor(
        (double) input_tokens * (double) keep_ratio);
    // Fixed grid unless the caller provides variable-length segments.
    const int n_chunks = segments
        ? (int) segments->size()
        : (input_tokens + config.chunk_size - 1) / config.chunk_size;

    std::vector<luce::pflash::PFlashSelectionCandidate> candidates;
    std::vector<std::pair<float, int>> chunk_means;
    std::vector<double> exact_chunk_scores;
    candidates.reserve((size_t) n_chunks);
    chunk_means.reserve((size_t) n_chunks);
    exact_chunk_scores.reserve((size_t) n_chunks);
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int begin = segments ? (*segments)[(size_t) chunk].begin : chunk * config.chunk_size;
        const int end = segments ? (*segments)[(size_t) chunk].end
                                 : std::min(input_tokens, begin + config.chunk_size);
        double score = 0.0;
        for (int token = begin; token < end; ++token) {
            score += token_scores[(size_t) token];
        }
        if (!direct_mass || density) {
            score /= (double) std::max(1, end - begin);
        }
        const bool mandatory =
            luce::pflash::pflash_chunk_is_structurally_required(
                begin, end, query_begin, query_end, input_tokens,
                required_instruction_spans,
                /*query_suffix_structural=*/ !config.query_suffix_candidates);
        candidates.push_back({(size_t) chunk, begin, end, score, mandatory});
        chunk_means.push_back({(float) score, chunk});
        exact_chunk_scores.push_back(score);
    }
    // Two-scorer selection: the other scorer's mean per-token score over the
    // same spans (its native ranking rule).
    std::vector<luce::pflash::PFlashSelectionCandidate> other_candidates;
    std::vector<double> other_scores;
    const bool split = other_token_scores != nullptr && split_fraction > 0.0;
    if (split) {
        for (const auto & candidate : candidates) {
            double score = 0.0;
            for (int token = candidate.begin; token < candidate.end; ++token) {
                score += (*other_token_scores)[(size_t) token];
            }
            score /= (double) std::max(1, candidate.end - candidate.begin);
            other_candidates.push_back({candidate.ordinal, candidate.begin, candidate.end, score, candidate.mandatory});
            other_scores.push_back(score);
        }
    }

    if (config.strategy == luce::pflash::PFlashSelectStrategy::Fold) {
        if (split || !fold_vocab) {
            const char * message = "PFlash fold selection needs the Qwen3.5 head "
                                   "scorer and the drafter vocabulary";
            set_last_error(message);
            std::fprintf(stderr, "[pflash-select] ERROR fold: %s\n", message);
            std::fflush(stderr);
            return {};
        }
        const auto started = std::chrono::steady_clock::now();
        luce::pflash::PFlashFoldStructure structure;
        std::vector<PFlashTokenSpan> blocks;
        const auto fold = luce::pflash::select_pflash_fold_for_ids(
            *fold_vocab, ids, candidates, config, selector_budget, &structure, &blocks);
        if (!fold.ok) {
            set_last_error("PFlash fold selection failed: " + fold.error);
            std::fprintf(stderr, "[pflash-select] ERROR fold budget=%d: %s\n",
                         selector_budget, fold.error.c_str());
            std::fflush(stderr);
            return {};
        }
        std::vector<int32_t> output;
        output.reserve((size_t) fold.retained_tokens);
        g_last_kept_spans = fold.kept;
        g_last_candidate_lifts.clear();
        for (const auto & span : fold.kept) {
            output.insert(output.end(), ids.begin() + span.begin, ids.begin() + span.end);
        }
        double kept_mass = 0.0;
        for (const auto & span : fold.kept) {
            for (int token = span.begin; token < span.end; ++token) {
                kept_mass += token_scores[(size_t) token];
            }
        }
        std::fprintf(stderr,
            "[pflash-select] selected strategy=fold route=%s code_lines=%d/%d "
            "code_blocks=%zu text=%s blocks=%zu segments=%s budget=%d "
            "selected_tokens=%zu anchors=%zu/%d contained=%d capped=%d skipped=%d "
            "stop=%s structure_ms=%.1f\n",
            structure.code_like ? "code" : "text", structure.code_lines,
            structure.nonblank_lines, structure.code_blocks, structure.text_source,
            blocks.size(), segments ? "probe" : "fixed", selector_budget,
            output.size(), fold.anchors.size(), config.fold_k, fold.contained,
            fold.capped, fold.skipped, luce::pflash::pflash_selection_stop_name(fold.stop),
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        std::fflush(stderr);
        if (write_trace) {
            std::vector<uint8_t> anchor_mask((size_t) n_chunks, 0);
            std::vector<uint8_t> mandatory_mask((size_t) n_chunks, 0);
            for (size_t ordinal : fold.anchors) anchor_mask[ordinal] = 1;
            for (const auto & candidate : candidates) {
                if (candidate.mandatory) mandatory_mask[candidate.ordinal] = 1;
            }
            g_last_relevance_spread = pflash_selection_spread(
                exact_chunk_scores, anchor_mask, mandatory_mask,
                candidates.empty() ? 0 : candidates[0].end - candidates[0].begin,
                token_scores.empty() ? 0.0 : (double) token_scores[0],
                !direct_mass || density);
            PFlashTraceFields fields{
                &ids, query_begin, query_end, config.mode, config.query_parser,
                selector_budget, fold.stop, fold.retained_tokens, kept_mass,
                &exact_chunk_scores, &required_instruction_spans};
            fields.segments = segments;
            fields.segmentation = segments ? "probe" : "fixed";
            fields.candidate_score = density ? "density" : "sum";
            fields.scorer = luce::pflash::pflash_scorer_name(config.scorer);
            fields.kept_spans = &fold.kept;
            fields.fold_blocks = &blocks;
            fields.fold_code = structure.code_like;
            fields.fold_code_lines = structure.code_lines;
            fields.fold_nonblank_lines = structure.nonblank_lines;
            fields.fold_code_blocks = (int) structure.code_blocks;
            fields.fold_text_source = structure.text_source;
            fields.fold_skipped = fold.skipped;
            fields.fold_capped = fold.capped;
            fields.fold_contained = fold.contained;
            fields.extra_queries = &config.extra_queries;
            fields.request_probe = config.request_probe_trace;
            fields.ident_query = config.ident_query_trace;
            if (config.fallback_spread > 0.0) {
                fields.spread = g_last_relevance_spread.spread;
                fields.spread_threshold = config.fallback_spread;
                fields.spread_first_token_score =
                    token_scores.empty() ? 0.0 : (double) token_scores[0];
                fields.spread_per_token = !direct_mass || density;
            }
            write_compression_trace(
                input_tokens, keep_ratio, segments ? 0 : config.chunk_size, query_tokens,
                pool_kernel, (int) fold.anchors.size(), chunk_means, anchor_mask,
                mandatory_mask, output, &fields);
        }
        return output;
    }

    const luce::pflash::PFlashSelectionPolicy policy{selector_budget, config.top_p,
                                                      /*skip_oversized=*/ segments != nullptr,
                                                      config.top_k};
    const auto selected = split
        ? luce::pflash::select_pflash_split(candidates, other_candidates, policy, split_fraction, config.mode)
        : luce::pflash::select_pflash_candidates(candidates, policy, config.mode);
    if (!selected.ok) {
        set_last_error("PFlash selection failed: " + selected.error);
        std::fprintf(stderr,
            "[pflash-select] ERROR mode=%s budget=%d stop=%s: %s\n",
            luce::pflash::pflash_selection_mode_name(config.mode),
            selector_budget,
            luce::pflash::pflash_selection_stop_name(selected.stop),
            selected.error.c_str());
        std::fflush(stderr);
        return {};
    }

    std::vector<uint8_t> selected_mask((size_t) n_chunks, 0);
    std::vector<uint8_t> mandatory_mask((size_t) n_chunks, 0);
    for (const auto & candidate : candidates) {
        if (candidate.mandatory) mandatory_mask[candidate.ordinal] = 1;
    }
    for (size_t ordinal : selected.ordinals) {
        if (ordinal >= selected_mask.size()) {
            set_last_error("PFlash selector returned an invalid ordinal");
            return {};
        }
        selected_mask[ordinal] = 1;
    }

    std::vector<int32_t> output;
    output.reserve((size_t) selected.retained_tokens);
    g_last_kept_spans.clear();
    g_last_candidate_lifts.clear();
    if (direct_mass) {
        // Head mass sums to one over the keys, so uniform attention gives
        // each token 1/input of it.
        for (const auto & candidate : candidates) {
            double mass = 0.0;
            for (int token = candidate.begin; token < candidate.end; ++token) {
                mass += token_scores[(size_t) token];
            }
            const int length = std::max(1, candidate.end - candidate.begin);
            g_last_candidate_lifts.push_back(
                {{candidate.begin, candidate.end},
                 mass / (double) length * (double) input_tokens});
        }
    }
    for (const auto & candidate : candidates) {
        if (!selected_mask[candidate.ordinal]) continue;
        output.insert(output.end(),
                      ids.begin() + candidate.begin,
                      ids.begin() + candidate.end);
        if (!g_last_kept_spans.empty() &&
            g_last_kept_spans.back().end == candidate.begin) {
            g_last_kept_spans.back().end = candidate.end;
        } else {
            g_last_kept_spans.push_back({candidate.begin, candidate.end});
        }
    }

    std::fprintf(stderr,
        "[pflash-select] selected mode=%s scorer=%s segments=%s score=%s chunk=%d query=%d "
        "budget=%d selected_tokens=%zu chunks=%zu/%d stop=%s mass=%.9g\n",
        luce::pflash::pflash_selection_mode_name(config.mode),
        split ? "split" : "single",
        segments ? "probe" : "fixed", density ? "density" : "sum",
        segments ? 0 : config.chunk_size, query_tokens, selector_budget, output.size(),
        selected.ordinals.size(), n_chunks,
        luce::pflash::pflash_selection_stop_name(selected.stop),
        selected.retained_mass);
    std::fflush(stderr);

    if (write_trace) {
        g_last_relevance_spread = pflash_selection_spread(
            exact_chunk_scores, selected_mask, mandatory_mask,
            candidates.empty() ? 0 : candidates[0].end - candidates[0].begin,
            token_scores.empty() ? 0.0 : (double) token_scores[0],
            !direct_mass || density);
        const int trace_chunk = segments ? 0 : config.chunk_size;
        const int n_keep_approx = segments
            ? (int) selected.ordinals.size()
            : std::max(1, (selector_budget + config.chunk_size - 1) / config.chunk_size);
        PFlashTraceFields strict_fields{
            &ids, query_begin, query_end, config.mode, config.query_parser,
            selector_budget,
            selected.stop, selected.retained_tokens, selected.retained_mass,
            &exact_chunk_scores, &required_instruction_spans};
        strict_fields.segments = segments;
        strict_fields.segmentation = segments ? "probe" : "fixed";
        strict_fields.candidate_score = density ? "density" : "sum";
        strict_fields.scorer = split ? "split" : luce::pflash::pflash_scorer_name(config.scorer);
        strict_fields.split_fraction = split ? split_fraction : 0.0;
        strict_fields.other_chunk_scores = split ? &other_scores : nullptr;
        strict_fields.top_k =
            config.mode == luce::pflash::PFlashSelectionMode::TopK ? config.top_k : 0;
        strict_fields.extra_queries = &config.extra_queries;
        strict_fields.request_probe = config.request_probe_trace;
        strict_fields.ident_query = config.ident_query_trace;
        if (config.fallback_spread > 0.0) {
            strict_fields.spread = g_last_relevance_spread.spread;
            strict_fields.spread_threshold = config.fallback_spread;
            strict_fields.spread_first_token_score =
                token_scores.empty() ? 0.0 : (double) token_scores[0];
            strict_fields.spread_per_token = !direct_mass || density;
        }
        write_compression_trace(
            input_tokens, keep_ratio, trace_chunk, query_tokens,
            pool_kernel, n_keep_approx, chunk_means, selected_mask,
            mandatory_mask, output, &strict_fields);
    }
    return output;
}

} // namespace luce::common
