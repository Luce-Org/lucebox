#pragma once

// Which tile a dense MMQ call takes on RDNA4 (tile twins). Pure
// host code: mmq.cu applies it per call (ggml_cuda_mul_mat_q_switch_type) and
// test_mmq_tile_twin_gate checks it without a GPU. The twins themselves are in
// mmq_tile_twins.h.
//
// 64-row twin: IQ3_S and IQ3_XXS keep the 128-row/8-warp RDNA tile in their
// parent instances. At 16 columns that leaves a 5120-row output with 40
// workgroups on a 64-CU gfx1201. The twin takes those calls to a
// 64-row/4-warp twin, the shape IQ4_XS/Q5_K/Q6_K/Q8_0 already use.
//
// 32-row twin: outputs of at most 6144 rows (the 5120-row
// ffn_down/ssm_out/attn_output, the 1024-row attn_k/attn_v, the 96-row
// ssm_beta|alpha stack) take a 32-row/2-warp twin instead: twice the
// workgroups of a 64-row tile, four times those of a 128-row one. Types:
// IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ3_XXS. Larger outputs keep their
// tile (an earlier measurement had the 32-row tile 15% slower on the
// 17408-row up and 10240-row QKV projections). Where both twins apply, the
// 2-warp twin wins.
//
// No twin for the other UD low-bit types: on R9700 at 16 columns, Q3_K,
// IQ2_S, IQ2_XS and Q2_K ran 10% to 53% slower on the 64-row twin than on
// their parent tile, and Q3_K and IQ2_S up to 50% slower on the 32-row one.
//
// Every tile computes every output element identically: RDNA4 MMQ never
// uses stream-k, the x-tile stays 16 wide, each warp owns 16 rows in every
// shape, and every element accumulates kb0 by kb0, then k01 by k01, in the
// same order. The tile height only decides which workgroup and which warp
// computes a row.
//
// Excluded paths, which keep the parent tile with a reason (inert_why): a
// device that is not RDNA4, and MUL_MAT_ID (expert-routed) calls. Calls
// wider than 16 columns (prefill), types without a twin and, for the 2-warp
// twin, outputs of more than 6144 rows keep the parent tile without one.

#include "ggml.h"

#include <cstdint>

enum class mmq_twin_tile {
    parent,  // the stock instance (ggml_cuda_mul_mat_q_switch_type's switch)
    rows64,  // 64-row/4-warp twin
    rows32,  // 32-row/2-warp twin
};

struct mmq_twins_enabled {
    bool ud_smalltile = false;  // the 64-row twin is on
    bool two_warp     = false;  // the 32-row twin is on
};

struct mmq_twin_call {
    ggml_type type     = GGML_TYPE_COUNT;
    int64_t   nrows    = 0;      // output rows (mmq_args::nrows_x)
    int64_t   ncols    = 0;      // max of mmq_args::ncols_dst and ncols_max
    bool      rdna4    = false;  // the current device is RDNA4
    bool      routed   = false;  // MUL_MAT_ID: ids_dst or expert_bounds set
};

struct mmq_twin_choice {
    mmq_twin_tile tile      = mmq_twin_tile::parent;
    const char *   inert_why = nullptr;  // set: an enabled twin is excluded here
};

// Widest call a twin serves: every instance picks mmq_x = 16 up to here.
constexpr int64_t MMQ_TWIN_MAX_COLS = 16;

// Tallest output the 2-warp twin serves.
constexpr int64_t MMQ_TWIN_ROWS32_MAX_ROWS = 6144;

// Types with a 64-row twin (template-instances/mmq-instance-*-rows64.cu).
inline bool mmq_twin_has_rows64(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
            return true;
        default:
            return false;
    }
}

// Types with a 32-row twin (template-instances/mmq-instance-*-rows32.cu):
// the ones the Qwen3.8 UD-IQ4_XS target has on outputs of <= 6144 rows.
inline bool mmq_twin_has_rows32(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
            return true;
        default:
            return false;
    }
}

inline mmq_twin_choice mmq_twin_choose(const mmq_twins_enabled & on, const mmq_twin_call & call) {
    mmq_twin_choice choice;
    if (!on.ud_smalltile && !on.two_warp) {
        return choice;
    }
    if (!call.rdna4) {
        choice.inert_why = "device is not RDNA4";
        return choice;
    }
    if (call.ncols > MMQ_TWIN_MAX_COLS) {
        return choice;
    }
    if (call.routed) {
        choice.inert_why = "MUL_MAT_ID (expert-routed) MMQ";
        return choice;
    }
    if (on.two_warp && call.nrows <= MMQ_TWIN_ROWS32_MAX_ROWS && mmq_twin_has_rows32(call.type)) {
        choice.tile = mmq_twin_tile::rows32;
    } else if (on.ud_smalltile && mmq_twin_has_rows64(call.type)) {
        choice.tile = mmq_twin_tile::rows64;
    }
    return choice;
}
