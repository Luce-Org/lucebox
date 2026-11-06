// PFlash on any target model.
//
// The drafter that scores a prompt is a model of its own (pflash_drafter.h):
// what it returns does not depend on the model that serves the compressed
// prompt. A backend only says what it does around a scoring window: what it
// drains first, how it frees memory for the drafter (park) and gets it back
// (unpark), what borrows the drafter once it is loaded, and how it lets the
// drafter go. pflash_compress_batch does the rest the same way for every
// model: one window for the batch, the drafter loaded once (again after a
// change of file or GPU), every request scored and its result filled, the
// drafter released after use or its pool trimmed, and the park undone.

#pragma once

#include "common/model_backend.h"
#include "pflash_drafter.h"

#include <functional>
#include <string>
#include <vector>

namespace luce::common {

struct PFlashWindow {
    // Prefix of the window's log lines.
    const char * tag = "compress";
    // Finish queued work the window could disturb; runs before every window.
    std::function<void()> drain;
    // Free memory for the drafter, for requests without skip_park; false
    // when that fails, and the batch is not scored. Unset: never parks.
    std::function<bool()> park;
    // Undo park; false when the model could not come back, and the batch's
    // results are dropped.
    std::function<bool()> unpark;
    // After a load: what borrows the drafter (a KVFlash scorer) attaches.
    std::function<void()> loaded;
    // Let the drafter go: drop what borrows it, then free_drafter() and clear
    // the loaded flag. Unset: free_drafter() alone.
    std::function<void()> release;
    // Backends the drafter may share with the model. A drafter on a backend
    // of its own keeps every temporary of its last scoring run in its pool,
    // so that pool is trimmed after the window.
    std::vector<ggml_backend_t> shared_backends;
    // The serving engine's context, which no scoring session is sized past
    // (DrafterContext::engine_max_ctx); 0: the drafter's own.
    int engine_max_ctx = 0;
};

// Score one request with a loaded drafter and fill its result.
ModelBackend::CompressResult pflash_score(
    DrafterContext & drafter, const ModelBackend::CompressRequest & request);

// Compress ``requests`` in one window. Requests that differ in drafter file,
// GPU, parking or residency cannot share one: each then gets a window of its
// own. ``loaded`` is the backend's "drafter is loaded" flag.
std::vector<ModelBackend::CompressResult> pflash_compress_batch(
    DrafterContext & drafter, bool & loaded,
    const std::vector<ModelBackend::CompressRequest> & requests,
    const PFlashWindow & window);

// The daemon's "compress <ids.bin> <keep_x1000> [<drafter.gguf>] [nopark]"
// line as a request: backend-local GPU 0, drafter kept loaded, an empty
// drafter path when the line gives none. False on bad arguments.
bool pflash_parse_compress_line(const std::string & line,
                                ModelBackend::CompressRequest & request);

}  // namespace luce::common
