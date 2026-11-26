// PFlash scoring pipeline glue. See pflash_compress.h.

#include "pflash_compress.h"

#include "pflash_request_probe.h"
#include "pflash_query_candidates.h"
#include "pflash_selection.h"
#include "internal.h"

#include <algorithm>
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

// ``"request_probe":{...}``: what the request probe read, chose and charged.
void write_request_probe_trace(FILE * file, const luce::pflash::PFlashRequestProbeTrace & t) {
    std::fputs(",\"request_probe\":{\"tau\":", file);
    write_float_or_null(file, t.tau);
    std::fputs(",\"fallback\":", file);
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
        std::fputc('}', file);
    }
    std::fputc(']', file);
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
                for (const char ch : q.redo_reason) {
                    if (ch == '"' || ch == '\\') std::fputc('\\', file);
                    if ((unsigned char) ch >= 0x20) std::fputc(ch, file);
                }
                std::fputc('"', file);
            }
        }
        std::fputc('}', file);
    }
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
        std::fprintf(file, ",\"segmentation\":\"%s\",\"candidate_score\":\"%s\",\"scorer\":\"head\"",
                     trace_fields->segmentation, trace_fields->candidate_score);
        if (trace_fields->request_probe) {
            write_request_probe_trace(file, *trace_fields->request_probe);
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
    if (trace_fields && trace_fields->stage1_chunk_scores &&
        trace_fields->stage1_chunk_scores->size() == scores.size()) {
        std::fputs("],\"stage1_chunk_scores\":[", file);
        for (size_t index = 0; index < scores.size(); ++index) {
            if (index) std::fputc(',', file);
            const double score = (*trace_fields->stage1_chunk_scores)[index];
            if (std::isfinite(score)) {
                std::fprintf(file, "%.17g", score);
            } else {
                std::fputs("null", file);
            }
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
    std::fputs("]}\n", file);
    std::fclose(file);
}

namespace {
thread_local std::vector<PFlashTokenSpan> g_last_kept_spans;
thread_local PFlashScoringStats g_last_scoring_stats;
thread_local std::vector<PFlashCandidateLift> g_last_candidate_lifts;
thread_local std::vector<PFlashTokenSpan> g_last_probe_pins;
thread_local PFlashAdaptiveBudget g_last_adaptive_budget;
thread_local std::string g_last_reranker_error;
thread_local std::string g_last_loaded_assets_identity;
} // namespace

double pflash_droppable_mass_lift(
        const std::vector<luce::pflash::PFlashSelectionCandidate> & candidates,
        const std::vector<float> & token_scores,
        const std::vector<uint8_t> & selected) {
    if (candidates.empty() || selected.size() != candidates.size() || token_scores.empty()) {
        return 0.0;
    }
    const int n_tok = (int) token_scores.size();
    double opt_mass = 0.0;
    double kept_mass = 0.0;
    int opt_toks = 0;
    int kept_toks = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto & c = candidates[i];
        if (c.mandatory) continue;
        // Always exclude token 0 (the attention sink) from droppable mass lift.
        const int begin = std::max(1, c.begin);
        const int end = std::min(n_tok, c.end);
        if (end <= begin) continue;
        double mass = 0.0;
        for (int t = begin; t < end; ++t) {
            const float v = token_scores[(size_t) t];
            if (std::isfinite(v) && v > 0.0f) mass += (double) v;
        }
        const int len = end - begin;
        opt_mass += mass;
        opt_toks += len;
        if (selected[i]) {
            kept_mass += mass;
            kept_toks += len;
        }
    }
    if (!(opt_mass > 1e-12) || opt_toks <= 0 || kept_toks <= 0) return 0.0;
    const double mass_share = kept_mass / opt_mass;
    const double tok_share = (double) kept_toks / (double) opt_toks;
    return mass_share / tok_share;
}

const PFlashAdaptiveBudget & pflash_last_adaptive_budget() {
    return g_last_adaptive_budget;
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

const std::string & pflash_last_reranker_error() {
    return g_last_reranker_error;
}

void pflash_set_reranker_error(std::string error) {
    g_last_reranker_error = std::move(error);
}

const std::string & pflash_last_loaded_assets_identity() {
    return g_last_loaded_assets_identity;
}

void pflash_set_loaded_assets_identity(std::string identity) {
    g_last_loaded_assets_identity = std::move(identity);
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
    g_last_adaptive_budget = {};
    g_last_reranker_error.clear();
    g_last_loaded_assets_identity.clear();
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
        const std::vector<float> * rerank_scores) {
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
    std::vector<double> raw_stage1_masses;
    raw_stage1_masses.reserve((size_t) n_chunks);
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int begin = segments ? (*segments)[(size_t) chunk].begin : chunk * config.chunk_size;
        const int end = segments ? (*segments)[(size_t) chunk].end
                                 : std::min(input_tokens, begin + config.chunk_size);
        double score = 0.0;
        for (int token = begin; token < end; ++token) {
            score += token_scores[(size_t) token];
        }
        raw_stage1_masses.push_back(score);
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
    // Preserve raw Stage-1 scores before reranker override so
    // g_last_candidate_lifts exports BOTH calibrated Stage-1 mass lift (for
    // PFLASH_RECALL_GATE D/S) AND final reranked ranking score (for unit_score).
    std::vector<double> raw_stage1_scores = exact_chunk_scores;
    // The segment reranker's scores replace the Stage-1 chunk scores.
    // rerank_scores[i] = log_density[i] + delta[i]; exp(score) is strictly positive and
    // preserves the exact monotonic ranking while keeping candidate lifts well-scaled.
    const bool reranker_applied = (rerank_scores != nullptr &&
                                   rerank_scores->size() == candidates.size());
    // Candidates whose score the reranker set: a per-token density in either score mode.
    std::vector<uint8_t> reranked(candidates.size(), 0);
    g_last_adaptive_budget = {};
    if (reranker_applied) {
        std::vector<PFlashTokenSpan> cand_spans;
        std::vector<uint8_t> cand_mand;
        cand_spans.reserve(candidates.size());
        cand_mand.reserve(candidates.size());
        double max_rs = 20.0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (!candidates[i].mandatory) {
                const float rs = (*rerank_scores)[i];
                if (std::isfinite(rs) && (double) rs > max_rs) max_rs = (double) rs;
            }
        }
        const double shift = max_rs - 20.0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            cand_spans.push_back({candidates[i].begin, candidates[i].end});
            cand_mand.push_back(candidates[i].mandatory ? 1 : 0);
            if (!candidates[i].mandatory) {
                const float rs = (*rerank_scores)[i];
                if (std::isfinite(rs)) {
                    const double clamped = std::clamp((double) rs - shift, -80.0, 20.0);
                    candidates[i].score = std::exp(clamped);
                    reranked[i] = 1;
                    exact_chunk_scores[i] = candidates[i].score;
                    chunk_means[i].first = (float) candidates[i].score;
                }
            }
        }
        g_last_adaptive_budget = pflash_reranker_adaptive_budget(
            cand_spans, *rerank_scores, cand_mand, &raw_stage1_masses);
    }

    const luce::pflash::PFlashSelectionPolicy policy{selector_budget, config.top_p,
                                                      /*skip_oversized=*/ segments != nullptr,
                                                      config.top_k};
    const auto selected = luce::pflash::select_pflash_candidates(candidates, policy, config.mode);
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

    // Sink-free droppable mass lift telemetry for trace/logging.
    const double initial_mass_lift = pflash_droppable_mass_lift(
        candidates, token_scores, selected_mask);

    std::vector<int32_t> output;
    output.reserve((size_t) selected.retained_tokens);
    g_last_kept_spans.clear();
    g_last_candidate_lifts.clear();
    if (direct_mass) {
        // Head mass sums to one over the keys, so uniform attention gives
        // each token 1/input of it. Export raw Stage-1 mass lift in `.lift`
        // (preserving PFLASH_RECALL_GATE calibration) and final reranked
        // score in `.rank_score` (used for paging & recall ranking).
        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            const auto & candidate = candidates[idx];
            const int length = std::max(1, candidate.end - candidate.begin);
            const double raw_score = raw_stage1_scores[idx];
            const double raw_per_tok = density
                ? raw_score
                : raw_score / (double) length;
            const double rank_per_tok = density || reranked[idx]
                ? candidate.score
                : candidate.score / (double) length;
            g_last_candidate_lifts.push_back(
                {{candidate.begin, candidate.end},
                 raw_per_tok * (double) input_tokens,
                 rank_per_tok * (double) input_tokens});
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

    char adaptive_buf[96] = "";
    if (g_last_adaptive_budget.valid) {
        std::snprintf(adaptive_buf, sizeof adaptive_buf,
                      " adaptive_eff=%.3f adaptive_nuc=%.3f adaptive_r=%.3f%s",
                      g_last_adaptive_budget.eff, g_last_adaptive_budget.nucleus_ratio,
                      g_last_adaptive_budget.keep_ratio,
                      g_last_adaptive_budget.bypass_to_full ? "(full)" : "");
    }
    std::fprintf(stderr,
        "[pflash-select] selected mode=%s segments=%s score=%s reranker=%s chunk=%d query=%d "
        "budget=%d selected_tokens=%zu chunks=%zu/%d stop=%s mass=%.9g mass_lift=%.3f%s\n",
        luce::pflash::pflash_selection_mode_name(config.mode),
        segments ? "probe" : "fixed", density ? "density" : "sum",
        reranker_applied ? "on" : "off",
        segments ? 0 : config.chunk_size, query_tokens, selector_budget, output.size(),
        selected.ordinals.size(), n_chunks,
        luce::pflash::pflash_selection_stop_name(selected.stop),
        selected.retained_mass, initial_mass_lift, adaptive_buf);
    std::fflush(stderr);

    if (write_trace) {
        const int trace_chunk = segments ? 0 : config.chunk_size;
        const int n_keep_approx = segments
            ? (int) selected.ordinals.size()
            : std::max(1, (selector_budget + config.chunk_size - 1) / config.chunk_size);
        PFlashTraceFields strict_fields{
            &ids, query_begin, query_end, config.mode, config.query_parser,
            selector_budget,
            selected.stop, selected.retained_tokens, selected.retained_mass,
            &exact_chunk_scores, reranker_applied ? &raw_stage1_scores : nullptr,
            &required_instruction_spans};
        strict_fields.segments = segments;
        strict_fields.segmentation = segments ? "probe" : "fixed";
        strict_fields.candidate_score = density ? "density" : "sum";
        strict_fields.top_k =
            config.mode == luce::pflash::PFlashSelectionMode::TopK ? config.top_k : 0;
        strict_fields.request_probe = config.request_probe_trace;
        write_compression_trace(
            input_tokens, keep_ratio, trace_chunk, query_tokens,
            pool_kernel, n_keep_approx, chunk_means, selected_mask,
            mandatory_mask, output, &strict_fields);
    }
    return output;
}

} // namespace luce::common
