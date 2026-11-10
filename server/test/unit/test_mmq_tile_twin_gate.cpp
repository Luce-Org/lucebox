// The RDNA4 MMQ tile-twin gate (ggml-cuda/mmq-tile-twin-gate.h): which tile
// (and x-tile, for 17-48 columns) a dense MMQ call takes with the 64-row
// and/or the 32-row twin on, and where a twin stays inert. GPU-free;
// test_mmq_tile_twins checks the twins on the GPU against this same gate.

#include "CppUnitTestFramework.hpp"

#include "../../deps/llama.cpp/ggml/src/ggml-cuda/mmq-tile-twin-gate.h"

#include <string>

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

const ggml_type kUdTypes[] = {
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
};

const ggml_type kTwoWarpTypes[] = {
    GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
};

}  // namespace

TEST_CASE(MmqTwinGate, twins_off_keep_the_parent_everywhere) {
    const mmq_twins_enabled off;
    for (ggml_type t : kUdTypes) {
        for (int64_t n : {1, 16, 512}) {
            mmq_twin_call call = dense(t, 5120, n);
            const mmq_twin_choice c = mmq_twin_choose(off, call);
            CHECK_TRUE(c.tile == mmq_twin_tile::parent);
            CHECK_TRUE(c.inert_why == nullptr);
            call.rdna4 = false;
            call.routed = true;
            CHECK_TRUE(mmq_twin_choose(off, call).inert_why == nullptr);
        }
    }
}

TEST_CASE(MmqTwinGate, ud_types_take_the_64_row_twin_up_to_16_columns) {
    for (ggml_type t : kUdTypes) {
        CHECK_TRUE(mmq_twin_has_rows64(t));
        for (int64_t rows : {48, 96, 1024, 5120, 6144, 6208, 17408}) {
            for (int64_t n : {1, 8, 15, 16}) {
                const mmq_twin_choice c = mmq_twin_choose(ud_on(), dense(t, rows, n));
                CHECK_TRUE(c.tile == mmq_twin_tile::rows64);
                CHECK_TRUE(c.inert_why == nullptr);
            }
            for (int64_t n : {17, 24, 32, 48}) {  // verify widths: the 64-row twin at x-tile 48
                const mmq_twin_choice c = mmq_twin_choose(ud_on(), dense(t, rows, n));
                CHECK_TRUE(c.tile == mmq_twin_tile::rows64);
                CHECK_TRUE(c.mmq_x == 48);
                CHECK_TRUE(c.inert_why == nullptr);
            }
            for (int64_t n : {49, 256, 4096}) {
                const mmq_twin_choice c = mmq_twin_choose(ud_on(), dense(t, rows, n));
                CHECK_TRUE(c.tile == mmq_twin_tile::parent);
                CHECK_TRUE(c.inert_why == nullptr);  // prefill widths: the shape gate, silent
            }
        }
    }
}

TEST_CASE(MmqTwinGate, other_types_keep_their_parent_silently) {
    // Q3_K, IQ2_S, IQ2_XS and Q2_K measured slower on both twins.
    for (ggml_type t : {GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                        GGML_TYPE_Q8_0, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_Q3_K,
                        GGML_TYPE_Q2_K, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_S,
                        GGML_TYPE_Q4_0, GGML_TYPE_Q2_1_ROCMFP2_MIX}) {
        CHECK_TRUE(!mmq_twin_has_rows64(t));
        const mmq_twin_choice c = mmq_twin_choose(ud_on(), dense(t, 5120, 16));
        CHECK_TRUE(c.tile == mmq_twin_tile::parent);
        CHECK_TRUE(c.inert_why == nullptr);
    }
    for (ggml_type t : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K,
                        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_Q4_0,
                        GGML_TYPE_Q2_1_ROCMFP2_MIX}) {
        CHECK_TRUE(!mmq_twin_has_rows32(t));
        const mmq_twin_choice c = mmq_twin_choose(two_warp_on(), dense(t, 5120, 16));
        CHECK_TRUE(c.tile == mmq_twin_tile::parent);
        CHECK_TRUE(c.inert_why == nullptr);
    }
}

TEST_CASE(MmqTwinGate, two_warp_takes_outputs_up_to_6144_rows_up_to_16_columns) {
    for (ggml_type t : kTwoWarpTypes) {
        CHECK_TRUE(mmq_twin_has_rows32(t));
        for (int64_t rows : {1, 48, 96, 1024, 5120, 6143, 6144}) {
            for (int64_t n : {1, 8, 16}) {
                const mmq_twin_choice c = mmq_twin_choose(two_warp_on(), dense(t, rows, n));
                CHECK_TRUE(c.tile == mmq_twin_tile::rows32);
                CHECK_TRUE(c.inert_why == nullptr);
            }
        }
        for (int64_t rows : {6145, 6176, 10240, 16384, 17408}) {  // taller outputs keep their tile
            const mmq_twin_choice c = mmq_twin_choose(two_warp_on(), dense(t, rows, 16));
            CHECK_TRUE(c.tile == mmq_twin_tile::parent);
            CHECK_TRUE(c.inert_why == nullptr);
        }
    }
}

TEST_CASE(MmqTwinGate, both_twins_on_two_warp_wins_where_both_apply) {
    for (ggml_type t : kUdTypes) {
        const bool has32 = mmq_twin_has_rows32(t);
        CHECK_TRUE(mmq_twin_choose(both_on(), dense(t, 5120, 16)).tile ==
                   (has32 ? mmq_twin_tile::rows32 : mmq_twin_tile::rows64));
        CHECK_TRUE(mmq_twin_choose(both_on(), dense(t, 17408, 16)).tile == mmq_twin_tile::rows64);
        CHECK_TRUE(mmq_twin_choose(both_on(), dense(t, 5120, 24)).tile == mmq_twin_tile::rows64);
        CHECK_TRUE(mmq_twin_choose(both_on(), dense(t, 5120, 64)).tile == mmq_twin_tile::parent);
    }
    for (ggml_type t : {GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
        CHECK_TRUE(mmq_twin_choose(both_on(), dense(t, 1024, 16)).tile == mmq_twin_tile::rows32);
        CHECK_TRUE(mmq_twin_choose(both_on(), dense(t, 17408, 16)).tile == mmq_twin_tile::parent);
    }
}

TEST_CASE(MmqTwinGate, wide_calls_take_the_measured_tile) {
    struct Want {
        ggml_type     type;
        int64_t       rows;
        int64_t       cols;
        mmq_twin_tile tile;
        int           mmq_x;
    };
    const Want wants[] = {
        {GGML_TYPE_IQ3_S,   5120,   32, mmq_twin_tile::rows64, 48},
        {GGML_TYPE_IQ3_XXS, 17408,  24, mmq_twin_tile::rows64, 48},
        {GGML_TYPE_Q3_K,    5120,   32, mmq_twin_tile::rows32, 32},
        {GGML_TYPE_Q3_K,    17408,  32, mmq_twin_tile::rows64, 32},
        {GGML_TYPE_Q3_K,    5120,   48, mmq_twin_tile::rows64, 48},
        {GGML_TYPE_IQ2_S,   6144,   24, mmq_twin_tile::rows32, 32},
        {GGML_TYPE_IQ2_S,   17408,  40, mmq_twin_tile::rows64, 48},
        {GGML_TYPE_Q5_K,    1024,   32, mmq_twin_tile::rows32, 16},
        {GGML_TYPE_Q5_K,    5120,   32, mmq_twin_tile::rows32, 32},
        {GGML_TYPE_Q5_K,    17408,  32, mmq_twin_tile::rows64, 32},
        {GGML_TYPE_Q5_K,    248320, 28, mmq_twin_tile::rows64, 32},  // lm_head
        {GGML_TYPE_Q5_K,    5120,   48, mmq_twin_tile::parent, 16},
        {GGML_TYPE_Q8_0,    96,     32, mmq_twin_tile::rows32, 16},  // ssm_beta|alpha
        {GGML_TYPE_Q4_K,    1024,   24, mmq_twin_tile::rows32, 16},
        {GGML_TYPE_Q6_K,    1280,   48, mmq_twin_tile::rows32, 16},
        {GGML_TYPE_Q8_0,    5120,   32, mmq_twin_tile::parent, 16},
        {GGML_TYPE_IQ4_XS,  5120,   32, mmq_twin_tile::rows64, 48},
        {GGML_TYPE_IQ4_XS,  17408,  48, mmq_twin_tile::rows64, 48},
        {GGML_TYPE_IQ2_XS,  5120,   32, mmq_twin_tile::parent, 16},
    };
    for (const Want & w : wants) {
        const mmq_twin_choice c = mmq_twin_choose(both_on(), dense(w.type, w.rows, w.cols));
        CHECK_TRUE(c.tile == w.tile);
        CHECK_TRUE(c.tile == mmq_twin_tile::parent || c.mmq_x == w.mmq_x);
        CHECK_TRUE(c.inert_why == nullptr);
        // A twin turned off keeps the parent for its share of the choices.
        const mmq_twins_enabled only = w.tile == mmq_twin_tile::rows64 ? two_warp_on() : ud_on();
        if (w.tile != mmq_twin_tile::parent) {
            CHECK_TRUE(mmq_twin_choose(only, dense(w.type, w.rows, w.cols)).tile == mmq_twin_tile::parent);
        }
    }
    mmq_twin_call routed = dense(GGML_TYPE_IQ3_S, 5120, 32);
    routed.routed = true;
    CHECK_TRUE(mmq_twin_choose(both_on(), routed).tile == mmq_twin_tile::parent);
    CHECK_TRUE(mmq_twin_choose(both_on(), routed).inert_why != nullptr);
}

TEST_CASE(MmqTwinGate, excluded_paths_are_inert_with_a_reason) {
    for (const mmq_twins_enabled & on : {ud_on(), two_warp_on(), both_on()}) {
        mmq_twin_call call = dense(GGML_TYPE_IQ3_S, 5120, 16);
        call.rdna4 = false;
        CHECK_TRUE(mmq_twin_choose(on, call).inert_why != nullptr);
        call = dense(GGML_TYPE_Q8_0, 96, 16);
        call.routed = true;
        CHECK_TRUE(mmq_twin_choose(on, call).tile == mmq_twin_tile::parent);
        CHECK_TRUE(mmq_twin_choose(on, call).inert_why != nullptr);
    }

    mmq_twin_call not_rdna4 = dense(GGML_TYPE_IQ3_S, 5120, 16);
    not_rdna4.rdna4 = false;
    mmq_twin_choice c = mmq_twin_choose(ud_on(), not_rdna4);
    CHECK_TRUE(c.tile == mmq_twin_tile::parent);
    REQUIRE_TRUE(c.inert_why != nullptr);
    CHECK_EQUAL(std::string(c.inert_why), std::string("device is not RDNA4"));
    not_rdna4.ncols = 512;  // a device that can never engage says so at any width
    CHECK_TRUE(mmq_twin_choose(ud_on(), not_rdna4).inert_why != nullptr);

    mmq_twin_call routed = dense(GGML_TYPE_IQ3_S, 5120, 16);
    routed.routed = true;
    c = mmq_twin_choose(ud_on(), routed);
    CHECK_TRUE(c.tile == mmq_twin_tile::parent);
    REQUIRE_TRUE(c.inert_why != nullptr);
    CHECK_EQUAL(std::string(c.inert_why), std::string("MUL_MAT_ID (expert-routed) MMQ"));
}
