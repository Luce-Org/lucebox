// In-process PFlash drafter for speculative prefill.
//
// "PFlash" is the whole compression concept (score -> select -> emit); the
// pflash drafter is the scorer model behind it — Qwen3.5-0.8B for now
// (qwen35_drafter.cpp + qwen35_loader.cpp): it runs the model's first
// fifteen blocks and scores the context with block 15's NoPE Q/K
// attention-mass head, with the all-layer running-max scorer kept as an
// opt-in alternative (PFLASH_QWEN35_LEGACY_SCORER=1 or the PFLASH scorer
// config). This header is the model-agnostic API surface; a future drafter
// slots in behind load_drafter / drafter_score_and_compress.
//
// Hosted in the SAME process / SAME ggml allocator as the dflash target, so
// we never pay the cross-process VRAM contention that broke the Python
// subprocess integration.
//
// Public entry point: drafter_score_and_compress() takes raw input token IDs,
// runs the full pflash compression pipeline in C++, returns the surviving
// token IDs (drafter vocab).

#pragma once

#include "common/pflash_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace luce::common {

struct Qwen35DrafterState;
struct DrafterHostSessions;

struct DrafterContext {
    ggml_backend_t        backend = nullptr;   // owned (created in load_drafter)
    // Scorer state for the current drafter (Qwen3.5-0.8B). The public API
    // below never exposes it; backends that need internals include
    // qwen35_drafter.h explicitly.
    Qwen35DrafterState *  state   = nullptr;   // owned scorer weights + heads
    int                   gpu     = -1;
    bool                  loaded  = false;
    // The GGUF the loaded weights came from (empty when none are loaded).
    std::string           path;
    // The serving engine's max context (its --max-ctx), set by the owner after load_drafter(); scoring sessions are
    // never sized past it (nor past the drafter's trained context). 0 = unknown: the drafter's context alone.
    int                   engine_max_ctx = 0;
    // Scoring sessions copied to host RAM by drafter_offload_sessions() while
    // the drafter is released; free_drafter() keeps them, the next scoring
    // call restores them (qwen35_drafter.h).
    std::shared_ptr<DrafterHostSessions> host_sessions;
};

// Load the drafter GGUF (a Qwen3.5-0.8B GGUF today).
// Creates a fresh GPU backend if `backend` is null. Otherwise uses the
// caller-provided backend (so the drafter shares the daemon's allocator).
//
// `gpu_layers` is accepted for API compat but ignored — every layer goes on
// the GPU since the drafter weights are only ~1.5 GB.
bool load_drafter(const std::string & gguf_path, int gpu_layers,
                  DrafterContext & out);
bool load_drafter(const std::string & gguf_path, int gpu_layers,
                  int gpu, DrafterContext & out);

void free_drafter(DrafterContext & ctx);

// --draft-residency auto (ReleaseAfterUse): before free_drafter(), copy the
// scoring sessions this load created or extended to host RAM (a restored,
// untouched session keeps its host copy), so the next turn can still resume
// after the drafter is loaded again. No-op without sessions
// (PFLASH_DRAFTER_SESSIONS=0, nothing scored yet).
void drafter_offload_sessions(DrafterContext & ctx);

// Free only model weights, keeping the backend alive for reuse.
// Avoids repeated ggml backend create/destroy during daemon reuse.
void free_drafter_weights(DrafterContext & ctx);

// Score the context with the block-15 scoring head, then run strict budget
// selection (or the configured selection mode). Returns surviving token IDs
// (drafter vocab).
//
//   ids          input token IDs of length S
//   keep_ratio   fraction of the token budget to keep
//   chunk_size   span granularity (default 32)
//   n_lookahead  Q tokens used for scorer attention (default 8)
//   pool_kernel  AvgPool kernel for score smoothing (default 13)
//   score_query_end  exclusive end of the scorer query window in ids;
//                    required (negative values are rejected)
//   query_suffix_candidates  strict selection only: tokens after the query
//                    window are scored candidates, not a kept suffix
//   history_queries  strict selection only: earlier questions' windows, most
//                    recent first, mixed into the scores at halving weights
//   turn_query   strict selection only: the latest user turn's tail, mixed
//                    in at the query's own weight
//   extra_queries  strict selection only (PFLASH_SELECT_QUERY_HEAD=1): the
//                    user's prose blocks around pasted material, each mixed
//                    in at the query's own weight
//   request_messages  PFLASH_SELECT_QUERY_PROBE=1 only: the human user
//                    messages' content spans the request probe reads
//                    (nullptr: none, the request falls back to today's
//                    windows)
//   keep_queries strict selection only: windows a later request scores as
//                    history queries; never scored here, their rows are
//                    kept so that request can resume the scoring session
//
// On failure returns empty vector + sets last_error.
std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float  keep_ratio,
    int    chunk_size  = 32,
    int    n_lookahead = 8,
    int    pool_kernel = 13,
    int    score_query_end = -1,
    const std::vector<PFlashTokenSpan> &
        required_instruction_spans = {},
    bool   query_suffix_candidates = false,
    const std::vector<PFlashTokenSpan> & history_queries = {},
    PFlashTokenSpan turn_query = {-1, -1},
    const std::vector<PFlashTokenSpan> & extra_queries = {},
    const PFlashRequestMessages * request_messages = nullptr,
    const std::vector<PFlashTokenSpan> & keep_queries = {});

} // namespace luce::common
