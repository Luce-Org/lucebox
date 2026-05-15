// dflash_spec_decode.h — Generic DFlash speculative-decode loop.
//
// Drives the universal DFlash draft model against any target that implements
// the DFlashTarget interface. The loop is fully model-agnostic: it knows
// nothing about the target's weight layout, KV cache shape, or attention
// kernels — those live behind DFlashTarget::verify_batch / snapshot_kv /
// restore_kv / embed_tokens / project_hidden_to_tokens.
//
// The draft-side machinery (DraftWeights, draft backend/GPU, feature ring)
// is shared across all targets and is plumbed through directly.

#pragma once

#include "dflash_target.h"
#include "dflash_feature_ring.h"
#include "dflash_draft_ipc.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash27b {

struct DraftWeights;  // forward-decl from internal.h

// Run the speculative decode loop: draft → verify → replay.
// Returns true on success, false on any error.
//
// `target` provides target-side ops (verify, snapshot/restore KV, embed,
// LM-head projection). The target adapter owns model-specific config
// (feature-ring binding for capture, attention knobs, multi-shard fan-out).
//
// `feature_ring` is the DFlash feature container shared between target
// (writer, via target.verify_batch) and draft (reader, via build_draft_step).
//
// `remote_draft`, when active, replaces local draft compute with an IPC
// round-trip to a separate draft process.
bool run_dflash_spec_decode(
        DFlashTarget & target,
        DraftWeights & draft_weights,
        ggml_backend_t draft_backend,
        DraftFeatureMirror & feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int last_tok,
        const char * out_path,
        int draft_ctx_max,
        int stream_fd = -1,
        DFlashDraftIpcClient * remote_draft = nullptr);

} // namespace dflash27b
