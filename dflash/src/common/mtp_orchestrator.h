// mtp_orchestrator.h — Generic MTP warm + decode driver.
//
// All compute (prefill, attention, MTP head forward, sampling) goes through
// DFlashTarget::verify_batch and IMtpModule::step_batch — both ggml graphs
// on the backend's device. Orchestrator owns only control flow and a single
// host-side hidden-sequence buffer for warm_head_kv.

#pragma once

#include "model_backend.h"
#include "mtp_interface.h"

#include <functional>

namespace dflash27b {

class DFlashTarget;

namespace common {
namespace mtp {

// Drive the full MTP warm + chain decode for one request.
//
// Preconditions:
//   - backend != nullptr
//   - backend->supports_mtp() returns true (else returns error early)
//   - backend->mtp() and backend->dflash_target() return non-null
//   - req.prompt is non-empty
//
// Behavior:
//   1. Chunked prefill via DFlashTarget::verify_batch, capturing the
//      backbone's per-position pre/post norm hidden states.
//   2. Seed the MTP module with the last hidden + warm head KV across
//      all prompt positions.
//   3. Run MtpChainRunner for n_gen decode tokens at the given gamma.
//   4. Stream tokens through io.emit() and append them to
//      result.tokens. Emit the terminal -1 sentinel.
//
// `on_warm_done`, when non-null, fires exactly once after warm_head_kv
// succeeds and before the chain runner starts. Backends use this to save a
// prefix-cache snapshot capturing the post-warm state — the only correct
// moment to do so, because chain decode mutates cache_.cur_pos beyond
// prompt_len. Must be no-op when MTP is not native-heads (flavor check).
//
// Returns a GenerateResult populated with tokens / prefill_s / decode_s.
// On any failure, .ok = false and .error describes the cause (matches
// the daemon log's "err <message>" line).
GenerateResult warm_and_decode(ModelBackend * backend,
                                const GenerateRequest & req,
                                const DaemonIO & io,
                                std::function<void()> on_warm_done = {});

}  // namespace mtp
}  // namespace common
}  // namespace dflash27b
