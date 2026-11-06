#include "pflash_window.h"

#include "pflash_compress.h"
#include "common/io_utils.h"
#include "luce.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

namespace luce::common {

namespace {

using CompressRequest = ModelBackend::CompressRequest;
using CompressResult = ModelBackend::CompressResult;

bool scorable(const CompressRequest & request) {
    return !request.input_ids.empty() && !request.drafter_path.empty() &&
           std::isfinite(request.keep_ratio) &&
           request.keep_ratio >= 0.0f && request.keep_ratio <= 1.0f;
}

bool same_window(const CompressRequest & a, const CompressRequest & b) {
    return a.drafter_path == b.drafter_path && a.drafter_gpu == b.drafter_gpu &&
           a.skip_park == b.skip_park && a.residency_action == b.residency_action;
}

void release_drafter(DrafterContext & drafter, bool & loaded,
                     const PFlashWindow & window) {
    if (window.release) {
        window.release();
    } else {
        free_drafter(drafter);
    }
    loaded = false;
}

// One window: every scorable request in ``requests`` shares ``first``'s
// drafter file, GPU, parking and residency.
void run_window(DrafterContext & drafter, bool & loaded,
                const std::vector<CompressRequest> & requests,
                const CompressRequest & first, const PFlashWindow & window,
                std::vector<CompressResult> & results) {
    const char * tag = window.tag;
    if (window.drain) window.drain();
    const bool parking = !first.skip_park && window.park;
    if (parking && !window.park()) {
        std::fprintf(stderr, "[%s] could not free memory for the drafter\n", tag);
        return;
    }
    const auto unpark = [&] {
        return !parking || !window.unpark || window.unpark();
    };

    if (loaded && (drafter.path != first.drafter_path || drafter.gpu != first.drafter_gpu)) {
        release_drafter(drafter, loaded, window);
    }
    if (!loaded) {
        std::fprintf(stderr, "[%s] loading drafter from %s on GPU %d ...\n",
                     tag, first.drafter_path.c_str(), first.drafter_gpu);
        if (!load_drafter(first.drafter_path, /*gpu_layers=*/999,
                          first.drafter_gpu, drafter)) {
            std::fprintf(stderr, "[%s] drafter init failed: %s\n", tag, luce_last_error());
            // A failed load can own a backend before anything is loaded.
            free_drafter(drafter);
            unpark();
            return;
        }
        loaded = true;
        drafter.engine_max_ctx = window.engine_max_ctx;
        std::fprintf(stderr, "[%s] drafter ready\n", tag);
        if (window.loaded) window.loaded();
    }

    for (size_t index = 0; index < requests.size(); ++index) {
        const CompressRequest & request = requests[index];
        if (!scorable(request)) continue;
        results[index] = pflash_score(drafter, request);
        if (results[index].ok) {
            std::fprintf(stderr, "[%s] %zu -> %zu tokens\n", tag,
                         request.input_ids.size(), results[index].compressed_ids.size());
        }
    }

    if (first.residency_action == DraftResidencyAction::ReleaseAfterUse) {
        // Scoring sessions outlive the release in host RAM; the next scoring
        // call restores them once the drafter is loaded again.
        if (loaded) drafter_offload_sessions(drafter);
        release_drafter(drafter, loaded, window);
    } else if (loaded && drafter.backend &&
               std::find(window.shared_backends.begin(), window.shared_backends.end(),
                         drafter.backend) == window.shared_backends.end()) {
        // A resident drafter owns its backend: on non-VMM devices that
        // backend's pool keeps every operator temporary of the last scoring
        // run (GiBs after a long prompt) until it is trimmed. Scoring
        // sessions are tensors of their own and are unaffected.
        const char * keep = std::getenv("PFLASH_DRAFTER_TRIM_POOL");
        if (!(keep && std::string(keep) == "0")) {
            const size_t trimmed = ggml_backend_cuda_trim_pool(drafter.backend);
            std::fprintf(stderr, "[%s] trimmed %.1f MiB from the drafter pool\n",
                         tag, trimmed / (1024.0 * 1024.0));
        }
    }

    if (!unpark()) {
        std::fprintf(stderr, "[%s] the model did not come back after scoring; "
                             "the batch's results are dropped\n", tag);
        std::fill(results.begin(), results.end(), CompressResult{});
    }
}

}  // namespace

CompressResult pflash_score(DrafterContext & drafter, const CompressRequest & request) {
    // score_query_end < 0 is the legacy "tail window" request value; the
    // scorer requires an explicit end.
    const int score_query_end = request.score_query_end >= 0
        ? request.score_query_end : (int) request.input_ids.size();
    CompressResult result = CompressResult::from_compressed_ids(drafter_score_and_compress(
        drafter, request.input_ids, request.keep_ratio,
        /*chunk_size=*/32, request.score_query_tokens, /*pool_kernel=*/13,
        score_query_end, request.required_instruction_spans,
        request.query_suffix_candidates, request.history_query_spans,
        request.turn_query_span, request.extra_query_spans,
        &request.request_messages, request.keep_query_spans));
    if (!result.ok) return result;
    result.kept_spans = pflash_last_kept_spans();
    const auto & scoring = pflash_last_scoring_stats();
    result.scorer_resume = scoring.resume;
    result.scorer_new_tokens = scoring.new_tokens;
    result.scorer_forward_s = scoring.forward_s;
    const auto & stage2 = pflash_last_stage2_outcome();
    result.budget_widened = stage2.widened;
    result.raw_fits = stage2.raw_fits;
    result.stage2_budget = stage2.budget;
    result.stage2_budget_widened = stage2.budget_widened;
    const auto & spread = pflash_last_relevance_spread();
    result.spread_valid = spread.valid;
    result.spread = spread.spread;
    result.spread_k = spread.k;
    for (const auto & candidate : pflash_last_candidate_lifts()) {
        result.candidate_lifts.push_back({candidate.span, candidate.lift});
    }
    result.probe_pins = pflash_last_probe_pins();
    return result;
}

std::vector<CompressResult> pflash_compress_batch(
        DrafterContext & drafter, bool & loaded,
        const std::vector<CompressRequest> & requests,
        const PFlashWindow & window) {
    std::vector<CompressResult> results(requests.size());
    const CompressRequest * first = nullptr;
    bool one_window = true;
    for (const CompressRequest & request : requests) {
        if (!scorable(request)) continue;
        if (first == nullptr) {
            first = &request;
        } else if (!same_window(*first, request)) {
            one_window = false;
        }
    }
    if (first == nullptr) return results;
    if (one_window) {
        run_window(drafter, loaded, requests, *first, window, results);
        return results;
    }
    // A window hosts one drafter configuration: score the others one by one.
    for (size_t index = 0; index < requests.size(); ++index) {
        if (!scorable(requests[index])) continue;
        std::vector<CompressResult> one(1);
        run_window(drafter, loaded, {requests[index]}, requests[index], window, one);
        results[index] = std::move(one.front());
    }
    return results;
}

bool pflash_parse_compress_line(const std::string & line, CompressRequest & request) {
    std::istringstream words(line.size() > 9 ? line.substr(9) : std::string{});
    std::string ids_path;
    int keep_x1000 = 0;
    if (!(words >> ids_path >> keep_x1000) || keep_x1000 < 0 || keep_x1000 > 1000) return false;
    std::string drafter_path;
    std::getline(words >> std::ws, drafter_path);
    while (!drafter_path.empty() && std::isspace((unsigned char) drafter_path.back())) {
        drafter_path.pop_back();
    }
    bool skip_park = false;
    const std::string nopark = "nopark";
    if (drafter_path == nopark) {
        skip_park = true;
        drafter_path.clear();
    } else if (drafter_path.size() > nopark.size() &&
               drafter_path.compare(drafter_path.size() - nopark.size() - 1,
                                    nopark.size() + 1, " " + nopark) == 0) {
        skip_park = true;
        drafter_path.resize(drafter_path.size() - nopark.size() - 1);
    }
    request = CompressRequest{};
    request.input_ids = read_int32_file(ids_path);
    if (request.input_ids.empty()) return false;
    request.keep_ratio = (float) keep_x1000 / 1000.0f;
    request.drafter_path = std::move(drafter_path);
    request.skip_park = skip_park;
    return true;
}

}  // namespace luce::common
