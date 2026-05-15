// layer_split_daemon.h — Layer-split request handler for qwen35 daemon mode.
//
// run_target_layer_split_request() handles a single inference request:
// prefill → (optional spec-decode or AR decode) → output.

#pragma once

#include "layer_split_types.h"
#include "layer_split_forward.h"
#include "dflash_draft_ipc.h"
#include "dflash_feature_ring.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash27b {

// Handle one inference request over a set of layer-split shards.
// Runs prefill, then either spec-decode (if run_dflash && draft available)
// or plain AR decode.  Emits tokens to stream_fd and optionally writes
// the full sequence to out_path.
bool run_target_layer_split_request(
        std::vector<TargetLayerSplitShard> & shards,
        DraftWeights * draft_weights,
        ggml_backend_t draft_backend,
        int draft_gpu,
        DraftFeatureMirror * feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int max_ctx,
        bool run_dflash,
        const char * out_path,
        int kq_stride_pad,
        int fa_window,
        int draft_ctx_max,
        int stream_fd = -1);

} // namespace dflash27b
