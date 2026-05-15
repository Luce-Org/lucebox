// dflash_feature_ring.h — DFlash draft feature ring buffer (target-agnostic).
//
// Hosts the F32 ring buffer that mirrors target hidden-state captures on the
// draft GPU, plus the helpers that move data:
//   - target activation tensor → ring slot
//   - ring range → contiguous draft input tensor
//   - target BF16 feature cache tensor → ring (with BF16→F32 conversion,
//     possibly across devices)
//
// Lives in common/ so any DFlash target architecture (qwen35, gemma4,
// laguna, ...) can reuse it without depending on architecture-specific
// weight or cache structs.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

namespace dflash27b {

struct DraftFeatureMirror {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * target_feat = nullptr; // F32 [n_target_layers*hidden_size, cap]
    void * bf16_staging = nullptr;
    size_t bf16_staging_elems = 0;
    int device = 0;
    int target_device = 0;
    int cap = 0;
    int n_target_layers = 0;
    int hidden_size = 0;
};

void draft_feature_mirror_free(DraftFeatureMirror & mirror);

bool draft_feature_mirror_init(DraftFeatureMirror & mirror,
                               ggml_backend_t backend,
                               int device,
                               int target_device,
                               int cap,
                               int n_target_layers,
                               int hidden_size);

// Check whether the mirror ring buffer can provide a contiguous view of
// ctx_len slots ending at committed. Returns true and writes slot0 (the
// starting slot in the ring buffer) on success.
bool draft_feature_mirror_can_view(const DraftFeatureMirror & mirror,
                                   int committed,
                                   int ctx_len,
                                   int & slot0);

// Copy and convert BF16→F32 for n_tokens starting at start_pos from a
// target-side BF16 feature ring (`src_target_feat` / `src_cap`) into the
// draft-side mirror ring buffer.
bool draft_feature_mirror_sync_range(const ggml_tensor * src_target_feat,
                                     int src_cap,
                                     DraftFeatureMirror & mirror,
                                     int start_pos,
                                     int n_tokens);

// Convenience: sync the last `committed` tokens (or mirror.cap, whichever is smaller).
bool draft_feature_mirror_sync_tail(const ggml_tensor * src_target_feat,
                                    int src_cap,
                                    DraftFeatureMirror & mirror,
                                    int committed);

// ── Ring ↔ tensor copy helpers (target-agnostic) ────────────────────

// Copy one capture slice from a target layer's activation output into the
// DraftFeatureMirror ring buffer. src_device is the GPU device of act_out.
bool copy_capture_slice_to_draft_ring(
    DraftFeatureMirror & feature_ring,
    int capture_idx,
    const ggml_tensor * act_out,
    int src_device,
    int chunk_start,
    int start_pos,
    int n_tokens);

// Copy n_tokens rows from the DraftFeatureMirror ring buffer into a
// destination tensor (typically the draft graph's target_hidden_cat input).
bool copy_feature_ring_range_to_tensor(
    const DraftFeatureMirror & feature_ring,
    ggml_tensor * dst,
    int start_pos,
    int n_tokens);

}  // namespace dflash27b
