// Helpers shared by the Qwen3-0.6B and Qwen3.5-0.8B drafter paths.
// See qwen3_drafter_common.h.

#include "qwen3_drafter_common.h"

#include "qwen3_drafter.h"
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

namespace dflash::common {

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
    const char * path = std::getenv("DFLASH_PFLASH_TRACE_PATH");
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
            dflash::qwen3::PFlashSelectionMode::Legacy &&
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
            "\"token_budget\":%d,"
            "\"retained_tokens\":%d",
            trace_fields->query_begin, trace_fields->query_end,
            dflash::qwen3::pflash_selection_mode_name(
                trace_fields->selector_mode),
            dflash::qwen3::pflash_query_parser_name(trace_fields->query_parser),
            trace_fields->token_budget, trace_fields->retained_tokens);
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
            dflash::qwen3::PFlashSelectionMode::Legacy) {
            std::fputs(",\"stop_reason\":null,\"retained_mass\":null", file);
        } else {
            std::fprintf(file,
                ",\"stop_reason\":\"%s\",\"retained_mass\":%.17g",
                dflash::qwen3::pflash_selection_stop_name(trace_fields->stop),
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
    std::fputs("]}\n", file);
    std::fclose(file);
}

std::vector<int32_t> select_pflash_chunks(
        const std::vector<int32_t> & ids,
        const std::vector<float> & token_scores,
        float keep_ratio,
        int n_lookahead,
        int score_query_end,
        int pool_kernel,
        const dflash::qwen3::PFlashSelectionConfig & config,
        const std::vector<PFlashTokenSpan> & required_instruction_spans,
        bool direct_mass,
        bool write_trace,
        const std::vector<PFlashTokenSpan> * segments,
        bool density,
        const std::vector<float> * other_token_scores,
        double split_fraction) {
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

    std::vector<dflash::qwen3::PFlashSelectionCandidate> candidates;
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
            dflash::qwen3::pflash_chunk_is_structurally_required(
                begin, end, query_begin, query_end, input_tokens,
                required_instruction_spans);
        candidates.push_back({(size_t) chunk, begin, end, score, mandatory});
        chunk_means.push_back({(float) score, chunk});
        exact_chunk_scores.push_back(score);
    }
    // Two-scorer selection: the other scorer's mean per-token score over the
    // same spans (its native ranking rule).
    std::vector<dflash::qwen3::PFlashSelectionCandidate> other_candidates;
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

    const dflash::qwen3::PFlashSelectionPolicy policy{selector_budget, config.top_p,
                                                      /*skip_oversized=*/ segments != nullptr};
    const auto selected = split
        ? dflash::qwen3::select_pflash_split(candidates, other_candidates, policy, split_fraction, config.mode)
        : dflash::qwen3::select_pflash_candidates(candidates, policy, config.mode);
    if (!selected.ok) {
        set_last_error("PFlash selection failed: " + selected.error);
        std::fprintf(stderr,
            "[pflash-select] ERROR mode=%s budget=%d stop=%s: %s\n",
            dflash::qwen3::pflash_selection_mode_name(config.mode),
            selector_budget,
            dflash::qwen3::pflash_selection_stop_name(selected.stop),
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
    for (const auto & candidate : candidates) {
        if (!selected_mask[candidate.ordinal]) continue;
        output.insert(output.end(),
                      ids.begin() + candidate.begin,
                      ids.begin() + candidate.end);
    }

    std::fprintf(stderr,
        "[pflash-select] selected mode=%s scorer=%s segments=%s score=%s chunk=%d query=%d "
        "budget=%d selected_tokens=%zu chunks=%zu/%d stop=%s mass=%.9g\n",
        dflash::qwen3::pflash_selection_mode_name(config.mode),
        split ? "split" : "single",
        segments ? "probe" : "fixed", density ? "density" : "sum",
        segments ? 0 : config.chunk_size, query_tokens, selector_budget, output.size(),
        selected.ordinals.size(), n_chunks,
        dflash::qwen3::pflash_selection_stop_name(selected.stop),
        selected.retained_mass);
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
            &exact_chunk_scores, &required_instruction_spans};
        strict_fields.segments = segments;
        strict_fields.segmentation = segments ? "probe" : "fixed";
        strict_fields.candidate_score = density ? "density" : "sum";
        strict_fields.scorer = split ? "split" : dflash::qwen3::pflash_scorer_name(config.scorer);
        strict_fields.split_fraction = split ? split_fraction : 0.0;
        strict_fields.other_chunk_scores = split ? &other_scores : nullptr;
        write_compression_trace(
            input_tokens, keep_ratio, trace_chunk, query_tokens,
            pool_kernel, n_keep_approx, chunk_means, selected_mask,
            mandatory_mask, output, &strict_fields);
    }
    return output;
}

} // namespace dflash::common
