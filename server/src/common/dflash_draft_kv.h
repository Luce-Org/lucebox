// dflash_draft_kv.h — drafter context-KV ring cache (target-agnostic).
//
// The DFlash drafter recomputes K/V over its whole feature window (up to
// draft_ctx_max tokens) every step, which dominates draft latency once the
// window fills (~10ms at 2048 on RTX 3090 vs ~3ms on short prompts). The
// window is append-only per commit, so this module keeps per-layer K/V ring
// caches on the draft backend and only computes the newly committed rows.
//
// One fixed-topology step graph is built at init: a fold-in append stage
// (up to a_step new rows per step, padded rows land in a trash slot) followed
// by the noise-token forward flash-attending over the cache. Every input
// tensor lives in a dedicated backend buffer (never gallocr-recycled), so the
// graph replays as a CUDA graph indefinitely — no per-step rebuilds at all.
//
// RoPE uses absolute positions; scores depend only on position differences,
// so results match the legacy rebased-window math while cached entries stay
// valid as the window slides (ring slot = position % cap).

#pragma once

#include "dflash_feature_ring.h"
#include "draft/draft_graph.h"
#include "internal.h"  // DraftWeights

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

struct DraftKvState {
    DraftKvState() = default;
    // Owns raw ggml contexts/buffers/graphs: copying would double-free.
    DraftKvState(const DraftKvState &) = delete;
    DraftKvState & operator=(const DraftKvState &) = delete;

    // geometry
    int cap = 0;         // ctx ring slots (== drafter feature window cap)
    int q_len = 0;       // noise block size
    int kv_total = 0;    // cache rows: cap + q_len + trash + mask alignment
    int a_step = 0;      // fold-in append capacity per step
    int trash_slot = 0;  // destination for padded append rows
    int fc_in = 0;
    bool any_full = false, any_swa = false;

    // persistent device memory: caches + every graph input.
    ggml_context *        mem_ctx = nullptr;
    ggml_backend_buffer_t mem_buf = nullptr;
    DraftKvCacheRefs      cache;
    ggml_tensor * inp_embed  = nullptr;  // [hidden, q_len] f32 (caller fills)
    ggml_tensor * pos_q      = nullptr;  // [q_len] i32
    ggml_tensor * noise_rows = nullptr;  // [q_len] i32 (static)
    ggml_tensor * mask_full  = nullptr;  // [kv_total, q_len] f16
    ggml_tensor * mask_swa   = nullptr;  // [kv_total, q_len] f16
    ggml_tensor * ap_feat    = nullptr;  // [fc_in, a_step] f32
    ggml_tensor * ap_pos     = nullptr;  // [a_step] i32
    ggml_tensor * ap_rows    = nullptr;  // [a_step] i32

    // the once-built step graph (fold-in append + noise forward)
    std::vector<uint8_t> meta_arena;
    ggml_context *  g_ctx  = nullptr;
    ggml_cgraph *   gf     = nullptr;
    ggml_gallocr_t  galloc = nullptr;
    ggml_tensor *   hidden_states = nullptr;
    ggml_tensor *   logits        = nullptr;  // iff lm_head passed at init

    // host bookkeeping
    const void *          built_for = nullptr;  // DraftWeights the graph was built against
    int64_t               next_pos = 0;  // first ctx position not yet appended
    std::vector<int32_t>  slot_pos;      // absolute position per ring slot, -1 empty
    std::vector<uint16_t> mask_hbuf;
    std::vector<int32_t>  i32_hbuf;
};

// Allocate caches + inputs and build the step graph. `cap` is the drafter
// feature-window capacity (ring size). lm_head may be null (hidden-only).
bool draft_kv_init(DraftKvState & st,
                   const DraftWeights & dw,
                   ggml_backend_t backend,
                   int cap,
                   ggml_tensor * lm_head);

// Invalidate all cached rows (new/rewound request). Cheap; the next
// begin_step bulk-appends the live window from the feature ring.
void draft_kv_reset(DraftKvState & st);

void draft_kv_free(DraftKvState & st);

// Bring the cache up to date with `committed` (bulk-appending from the
// feature ring if more than a_step rows are missing) and fill all step
// inputs (append rows, positions, masks). After this the caller uploads
// inp_embed and computes st.gf; the draft hidden is st.hidden_states.
bool draft_kv_begin_step(DraftKvState & st,
                         const DraftWeights & dw,
                         ggml_backend_t backend,
                         const DraftFeatureMirror & ring,
                         int committed);

}  // namespace dflash::common
