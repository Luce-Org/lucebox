// The RDNA4 MMQ tile-twin gate (ggml-cuda/mmq-tile-twin-gate.h): which tile
// (and x-tile, for 17-48 columns) a dense MMQ call takes with the 64-row
// and/or the 32-row twin on, and where the parent tile stays. GPU-free;
// test_mmq_tile_twins checks the twins on the GPU against this same gate.

#include "CppUnitTestFramework.hpp"

#include "../../deps/llama.cpp/ggml/src/ggml-cuda/mmq-tile-twin-gate.h"

namespace {

struct MmqTwinGate {};

mmq_twin_call dense(ggml_type type, int64_t nrows, int64_t ncols) {
    mmq_twin_call call;
    call.type  = type;
    call.nrows = nrows;
    call.ncols = ncols;
    call.rdna4 = true;
    return call;
}

mmq_twins_enabled ud_on() {
    mmq_twins_enabled on;
    on.ud_smalltile = true;
    return on;
}

mmq_twins_enabled two_warp_on() {
    mmq_twins_enabled on;
    on.two_warp = true;
    return on;
}

mmq_twins_enabled both_on() {
    mmq_twins_enabled on;
    on.ud_smalltile = true;
    on.two_warp = true;
    return on;
}

}  // namespace

// One row per boundary of the measured choice. Every twin is bit-exact against
// its parent (test_mmq_tile_twins), so this table pins the tuning only.
TEST_CASE(MmqTwinGate, tile_choice_table) {
    struct Want {
        mmq_twins_enabled on;
        ggml_type         type;
        int64_t           rows;
        int64_t           cols;
        mmq_twin_tile     tile;
        int               mmq_x;
    };
    const Want wants[] = {
        // Up to 16 columns, one twin on.
        {ud_on(),       GGML_TYPE_IQ3_S,   5120,   16, mmq_twin_tile::rows64, 16},
        {ud_on(),       GGML_TYPE_IQ3_S,   5120,   49, mmq_twin_tile::parent, 16},  // prefill widths
        {ud_on(),       GGML_TYPE_Q4_K,    5120,   16, mmq_twin_tile::parent, 16},  // no 64-row twin
        {two_warp_on(), GGML_TYPE_Q2_K,    5120,   16, mmq_twin_tile::parent, 16},  // no 32-row twin
        {two_warp_on(), GGML_TYPE_Q8_0,    6144,   16, mmq_twin_tile::rows32, 16},
        {two_warp_on(), GGML_TYPE_Q8_0,    6145,   16, mmq_twin_tile::parent, 16},  // taller outputs
        // Both on: the 32-row twin wins where both apply.
        {both_on(),     GGML_TYPE_IQ3_S,   5120,   16, mmq_twin_tile::rows32, 16},
        {both_on(),     GGML_TYPE_IQ3_XXS, 17408,  16, mmq_twin_tile::rows64, 16},
        {both_on(),     GGML_TYPE_Q4_K,    17408,  16, mmq_twin_tile::parent, 16},
        // 17-48 columns: the measured choice of mmq_twin_choose_wide.
        {ud_on(),       GGML_TYPE_IQ3_XXS, 17408,  48, mmq_twin_tile::rows64, 48},
        {both_on(),     GGML_TYPE_IQ3_S,   5120,   32, mmq_twin_tile::rows64, 48},
        {both_on(),     GGML_TYPE_IQ3_XXS, 17408,  24, mmq_twin_tile::rows64, 48},
        {both_on(),     GGML_TYPE_Q3_K,    5120,   32, mmq_twin_tile::rows32, 32},
        {both_on(),     GGML_TYPE_Q3_K,    17408,  32, mmq_twin_tile::rows64, 32},
        {both_on(),     GGML_TYPE_Q3_K,    5120,   48, mmq_twin_tile::rows64, 48},
        {both_on(),     GGML_TYPE_IQ2_S,   6144,   24, mmq_twin_tile::rows32, 32},
        {both_on(),     GGML_TYPE_IQ2_S,   17408,  40, mmq_twin_tile::rows64, 48},
        {both_on(),     GGML_TYPE_Q5_K,    1024,   32, mmq_twin_tile::rows32, 16},
        {both_on(),     GGML_TYPE_Q5_K,    5120,   32, mmq_twin_tile::rows32, 32},
        {both_on(),     GGML_TYPE_Q5_K,    17408,  32, mmq_twin_tile::rows64, 32},
        {both_on(),     GGML_TYPE_Q5_K,    248320, 28, mmq_twin_tile::rows64, 32},  // lm_head
        {both_on(),     GGML_TYPE_Q5_K,    5120,   48, mmq_twin_tile::parent, 16},
        {both_on(),     GGML_TYPE_Q8_0,    96,     32, mmq_twin_tile::rows32, 16},  // ssm_beta|alpha
        {both_on(),     GGML_TYPE_Q4_K,    1024,   24, mmq_twin_tile::rows32, 16},
        {both_on(),     GGML_TYPE_Q6_K,    1280,   48, mmq_twin_tile::rows32, 16},
        {both_on(),     GGML_TYPE_Q8_0,    5120,   32, mmq_twin_tile::parent, 16},
        {both_on(),     GGML_TYPE_IQ4_XS,  5120,   32, mmq_twin_tile::rows64, 48},
        {both_on(),     GGML_TYPE_IQ4_XS,  17408,  48, mmq_twin_tile::rows64, 48},
        {both_on(),     GGML_TYPE_IQ2_XS,  5120,   32, mmq_twin_tile::parent, 16},
    };
    for (const Want & w : wants) {
        const mmq_twin_choice c = mmq_twin_choose(w.on, dense(w.type, w.rows, w.cols));
        CHECK_TRUE(c.tile == w.tile);
        CHECK_TRUE(c.tile == mmq_twin_tile::parent || c.mmq_x == w.mmq_x);
        // A wide choice with both twins on: the other twin alone keeps the parent.
        if (w.on.ud_smalltile && w.on.two_warp && w.cols > MMQ_TWIN_MAX_COLS && w.tile != mmq_twin_tile::parent) {
            const mmq_twins_enabled only = w.tile == mmq_twin_tile::rows64 ? two_warp_on() : ud_on();
            CHECK_TRUE(mmq_twin_choose(only, dense(w.type, w.rows, w.cols)).tile == mmq_twin_tile::parent);
        }
    }
    // Which types have each twin, at a decode width: a type launched on a twin
    // it has no measured win for runs slower.
    for (ggml_type t : {GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                        GGML_TYPE_Q8_0, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS}) {
        CHECK_TRUE(mmq_twin_choose(two_warp_on(), dense(t, 5120, 16)).tile == mmq_twin_tile::rows32);
    }
    for (ggml_type t : {GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS}) {
        CHECK_TRUE(mmq_twin_choose(ud_on(), dense(t, 5120, 16)).tile == mmq_twin_tile::rows64);
    }
    for (ggml_type t : {GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                        GGML_TYPE_Q8_0, GGML_TYPE_Q3_K, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS,
                        GGML_TYPE_Q2_K}) {
        CHECK_TRUE(mmq_twin_choose(ud_on(), dense(t, 5120, 16)).tile == mmq_twin_tile::parent);
    }
    for (ggml_type t : {GGML_TYPE_Q3_K, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_K}) {
        CHECK_TRUE(mmq_twin_choose(two_warp_on(), dense(t, 5120, 16)).tile == mmq_twin_tile::parent);
    }
}

TEST_CASE(MmqTwinGate, excluded_paths_keep_the_parent) {
    for (const mmq_twins_enabled & on : {ud_on(), two_warp_on(), both_on()}) {
        mmq_twin_call not_rdna4 = dense(GGML_TYPE_IQ3_S, 5120, 16);
        not_rdna4.rdna4 = false;
        CHECK_TRUE(mmq_twin_choose(on, not_rdna4).tile == mmq_twin_tile::parent);
        for (mmq_twin_call routed : {dense(GGML_TYPE_IQ3_S, 5120, 16), dense(GGML_TYPE_Q8_0, 96, 16),
                                     dense(GGML_TYPE_IQ3_S, 5120, 32)}) {
            routed.routed = true;  // MUL_MAT_ID (expert-routed) MMQ
            CHECK_TRUE(mmq_twin_choose(on, routed).tile == mmq_twin_tile::parent);
        }
    }
}
