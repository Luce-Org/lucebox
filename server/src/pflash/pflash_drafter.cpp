// PFlash drafter entry points: load/free the scorer and run
// drafter_score_and_compress. The pflash drafter is Qwen3.5-0.8B for now —
// this file is the dispatch seam where a different drafter model would
// slot in.
//
// Wires three pieces:
//   - qwen35_loader.cpp    : mmap GGUF + populate ggml tensors on backend,
//                            plus the optional scoring head and segment probe
//   - qwen35_drafter.cpp   : the block-15 head scorer and the all-layer
//                            running-max scorer
//   - pflash_compress.cpp  : score -> candidate -> strict selection + trace
//
// Single-pass forward over the first fifteen blocks on the Qwen3.5 target
// architecture (build_qwen35_layer); the block-15 NoPE Q/K projections score
// the context against the request's explicit query window.

#include "pflash_drafter.h"

#include "qwen35_drafter.h"
#include "pflash_selection.h"
#include "pflash_compress.h"
#include "common/dspark_head.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <string>
#include <vector>

namespace luce::common {

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterContext & out) {
    return load_drafter(gguf_path, /*gpu_layers=*/999, /*gpu=*/0, out);
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  int gpu, DrafterContext & out) {
    if (gpu < 0) {
        set_last_error("load_drafter: negative GPU index");
        return false;
    }
    if (out.loaded) {
        set_last_error("drafter already loaded");
        return false;
    }
    if (out.backend && out.gpu >= 0 && out.gpu != gpu) {
        set_last_error("load_drafter: backend already bound to a different GPU");
        return false;
    }

    // If caller didn't supply a backend, spin up our own GPU backend. Sharing
    // would be ideal but we don't have a handle to the daemon's backend
    // through this API. Same-process GPU pools coexist fine; fragmentation is
    // the only cost, and we free everything in free_drafter.
    if (!out.backend) {
        size_t n_dev = ggml_backend_dev_count();
        int seen_gpu = 0;
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                if (seen_gpu == gpu) {
                    out.backend = ggml_backend_dev_init(dev, nullptr);
                    break;
                }
                seen_gpu++;
            }
        }
        if (!out.backend) {
            set_last_error("load_drafter: requested GPU backend unavailable");
            return false;
        }
        out.gpu = gpu;
    } else if (out.gpu < 0) {
        out.gpu = gpu;
    }

    return load_qwen35_drafter(gguf_path, out);
}

void free_drafter(DrafterContext & ctx) {
    dspark_note_drafter_lifecycle();
    free_drafter_weights(ctx);
    if (ctx.backend) {
        ggml_backend_free(ctx.backend);
        ctx.backend = nullptr;
    }
    ctx.gpu = -1;
}

void free_drafter_weights(DrafterContext & ctx) {
    if (ctx.state) {
        free_qwen35_drafter_state(ctx);
    }
    ctx.loaded = false;
}

std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    bool query_suffix_candidates,
    const std::vector<PFlashTokenSpan> & history_queries,
    PFlashTokenSpan turn_query) {
    pflash_clear_kept_spans();
    if (!ctx.loaded) {
        set_last_error("drafter not loaded");
        return {};
    }

    luce::pflash::PFlashSelectionConfig experiment;
    std::string experiment_error;
    if (!luce::pflash::resolve_pflash_selection(
            (int) ids.size(), chunk_size, experiment, experiment_error)) {
        set_last_error("invalid PFlash strict selection config: " + experiment_error);
        std::fprintf(stderr, "[pflash-select] ERROR config: %s\n",
                     experiment_error.c_str());
        std::fflush(stderr);
        return {};
    }
    chunk_size = experiment.chunk_size;
    experiment.query_suffix_candidates =
        query_suffix_candidates && experiment.selection_active;
    if (experiment.selection_active) {
        for (const auto & window : history_queries) {
            if (window.begin >= 0 && window.end > window.begin &&
                window.end <= (int) ids.size()) {
                experiment.history_queries.push_back(window);
            }
        }
        if (turn_query.begin >= 0 && turn_query.end > turn_query.begin &&
            turn_query.end <= (int) ids.size()) {
            experiment.turn_query = turn_query;
        }
    }
    if (!experiment.selection_active && !required_instruction_spans.empty()) {
        set_last_error(
            "PFlash instruction spans require strict budget selection");
        std::fprintf(stderr,
            "[pflash-select] ERROR instruction spans require strict selection\n");
        std::fflush(stderr);
        return {};
    }
    if (experiment.selection_active) {
        std::string span_error;
        if (!luce::pflash::validate_pflash_instruction_spans(
                required_instruction_spans, (int) ids.size(), span_error)) {
            set_last_error("invalid PFlash instruction spans: " + span_error);
            std::fprintf(stderr,
                "[pflash-select] ERROR instruction spans: %s\n",
                span_error.c_str());
            std::fflush(stderr);
            return {};
        }
    }
    if (experiment.configured) {
        std::fprintf(stderr,
            "[pflash-select] config mode=%s active=%d chunk=%d "
            "query_parser=%s query_cap=%d query_actual=%d top_p=%.9g "
            "top_k=%d suffix_candidates=%d input=%zu\n",
            luce::pflash::pflash_selection_mode_name(experiment.mode),
            (int) experiment.selection_active, experiment.chunk_size,
            luce::pflash::pflash_query_parser_name(experiment.query_parser),
            experiment.query_tokens, n_lookahead, experiment.top_p,
            experiment.top_k, (int) experiment.query_suffix_candidates,
            ids.size());
        std::fflush(stderr);
    }
    if (score_query_end < 0) {
        set_last_error("qwen35 scorer query window out of range");
        return {};
    }
    return qwen35_drafter_score_and_compress(
        ctx, ids, keep_ratio, chunk_size, n_lookahead, pool_kernel,
        score_query_end, experiment, required_instruction_spans);
}

} // namespace luce::common
