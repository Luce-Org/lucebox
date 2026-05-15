// Generic DFlash draft-step graph builder.
//
// The DFlash draft model is the same single Qwen3-style network for every
// target architecture (see dflash_target.h). This wrapper sets up a
// StepGraph, optionally views into a feature mirror, calls the universal
// build_draft_graph (src/draft/draft_graph.h), and reserves the gallocr
// buffer.

#pragma once

#include "step_graph.h"
#include "dflash_feature_ring.h"
#include "internal.h"  // DraftWeights

#include "ggml.h"
#include "ggml-backend.h"

namespace dflash27b {

// Draft forward: speculative next-token prediction using target features.
//   lm_head: optional target lm_head tensor for fused projection. When
//   nullptr, the draft graph emits hidden states only and the caller is
//   responsible for projection (e.g. via build_lm_head_step on the target).
bool build_draft_step(
    StepGraph & sg,
    const DraftWeights & dw,
    ggml_tensor * lm_head,
    ggml_backend_t backend,
    int ctx_len,
    const DraftFeatureMirror * mirror = nullptr,
    int committed = 0);

}  // namespace dflash27b
