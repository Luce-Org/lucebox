// Negative coverage for the qtype-105/106 fused gate/up+SwiGLU admission check.
//
// WHY THIS EXISTS. The fused path writes straight into glu->data and derives its strides
// from glu->nb, so it is only correct when the GLU inputs share shape AND stride with the
// matmul outputs -- a reshape that changes the token/expert axes silently remaps what the
// kernel writes. The sibling vector fusions have always been gated on that; the mix path
// was not, because the guard was added upstream after the mix kernel was written and the
// omission survived a rebase. This test is the thing that would have caught it.
//
// It tests the PREDICATE, not the dispatcher. A graph-level test cannot distinguish
// "correctly refused to fuse" from "fused and happened to produce the right answer", which
// is exactly the distinction that matters here: the pre-fix code produced correct output on
// every non-reshaped graph, which is why it looked fine.
//
// No device work: every case is pure admission logic on tensor metadata.

#include "ggml.h"
#include "CppUnitTestFramework.hpp"
using CppUnitTestFramework::CommonFixture;
#undef CHECK

#include <cstdio>
#include <cstring>
#include <vector>

bool ggml_cuda_ds4_mix_glu_fusable(
    const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu,
    bool direct_layout);

static int g_fails = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fails; }  \
    } while (0)

namespace {

// A minimal hand-built triple. These tensors are never executed, only inspected, so the
// data pointers stay null and only the metadata the predicate reads is populated.
struct Triple {
    ggml_tensor w_up{}, w_gate{}, act{}, ids{}, up{}, gate{}, glu{};

    Triple() {
        w_up.type   = GGML_TYPE_Q2_1_ROCMFP2_MIX;
        w_gate.type = GGML_TYPE_Q2_1_ROCMFP2_MIX;
        act.type    = GGML_TYPE_F32;
        ids.type    = GGML_TYPE_I32;

        up.op   = GGML_OP_MUL_MAT_ID;
        gate.op = GGML_OP_MUL_MAT_ID;
        up.src[0]   = &w_up;   up.src[1]   = &act; up.src[2] = &ids;
        gate.src[0] = &w_gate; gate.src[1] = &act; gate.src[2] = &ids;

        glu.type = GGML_TYPE_F32;
        glu.op   = GGML_OP_GLU;
        // ggml_get_glu_op reads op_params[0].
        const int32_t glu_op = (int32_t) GGML_GLU_OP_SWIGLU_DS4;
        std::memcpy(glu.op_params, &glu_op, sizeof(glu_op));
        glu.src[0] = &gate;
        glu.src[1] = &up;
    }
};

}  // namespace

namespace {
struct RocmfpMixGluFusableFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

TEST_CASE(RocmfpMixGluFusableFixture, admission_predicate) {
    // The happy path: everything matched, layout direct. Establishes that the negative
    // cases below fail for the reason under test and not because the fixture is malformed.
    {
        Triple t;
        CHECK(ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, true),
              "matched qtype-106 triple with direct layout is fusable");
    }

    // THE REGRESSION. Same triple, but the caller determined the GLU inputs do not share
    // shape/stride with the matmul outputs. Must refuse regardless of everything else.
    {
        Triple t;
        CHECK(!ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, false),
              "reshaped layout (direct_layout=false) is REFUSED");
    }

    // gate consuming different activations than up: two unrelated mul_mat_ids whose weight
    // types happen to match. Fusing these would read one expert's rows under the other's
    // routing.
    {
        Triple t;
        ggml_tensor other_act{};
        other_act.type = GGML_TYPE_F32;
        t.gate.src[1] = &other_act;
        CHECK(!ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, true),
              "gate with different activations is REFUSED");
    }

    // gate routed by a different ids tensor.
    {
        Triple t;
        ggml_tensor other_ids{};
        other_ids.type = GGML_TYPE_I32;
        t.gate.src[2] = &other_ids;
        CHECK(!ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, true),
              "gate with different routing ids is REFUSED");
    }

    // gate is not a mul_mat_id at all.
    {
        Triple t;
        t.gate.op = GGML_OP_MUL_MAT;
        CHECK(!ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, true),
              "gate that is not MUL_MAT_ID is REFUSED");
    }

    // Halves of different weight types (105 against 106).
    {
        Triple t;
        t.w_gate.type = GGML_TYPE_Q3_1_ROCMFP3_MIX;
        CHECK(!ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, true),
              "mismatched half weight types are REFUSED");
    }

    // A non-mix qtype must not enter this path; it has its own fusion route.
    {
        Triple t;
        t.w_up.type = GGML_TYPE_Q4_0; t.w_gate.type = GGML_TYPE_Q4_0;
        CHECK(!ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, true),
              "non-mix qtype is REFUSED by the mix path");
    }

    // Not a SwiGLU-DS4 GLU.
    {
        Triple t;
        const int32_t other = (int32_t) GGML_GLU_OP_SWIGLU;
        std::memcpy(t.glu.op_params, &other, sizeof(other));
        CHECK(!ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, true),
              "non-SWIGLU_DS4 glu op is REFUSED");
    }

    // qtype-105 is symmetric with 106 on the happy path.
    {
        Triple t;
        t.w_up.type = GGML_TYPE_Q3_1_ROCMFP3_MIX;
        t.w_gate.type = GGML_TYPE_Q3_1_ROCMFP3_MIX;
        CHECK(ggml_cuda_ds4_mix_glu_fusable(&t.gate, &t.up, &t.glu, true),
              "matched qtype-105 triple is fusable");
    }

    // Null operands must not crash the admission check.
    CHECK(!ggml_cuda_ds4_mix_glu_fusable(nullptr, nullptr, nullptr, true),
          "null operands are REFUSED without dereferencing");

    std::fprintf(stderr, g_fails ? "MIX GLU FUSABLE TEST FAILED (%d)\n"
                                 : "MIX GLU FUSABLE TEST OK\n", g_fails);
    REQUIRE_TRUE(g_fails == 0);
}
