#include "rocmfp4.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { TEST_N = 64 };

static void fill_input(float * src) {
    for (int i = 0; i < TEST_N; ++i) {
        src[i] = 0.75f*sinf(0.37f*(float) i) +
                 0.25f*cosf(0.13f*(float) i) +
                 0.035f*(float) (i % 11 - 5);
    }

    src[7]  =  3.25f;
    src[19] = -2.75f;
    src[43] =  1.875f;
}

static float mse(const float * expected, const float * actual) {
    float error = 0.0f;
    for (int i = 0; i < TEST_N; ++i) {
        const float delta = expected[i] - actual[i];
        error += delta*delta;
    }
    return error / (float) TEST_N;
}

static void check_reference_roundtrip(void) {
    float src[TEST_N];
    float dual[TEST_N];
    float fast[TEST_N];
    block_rocmfp4 q_dual[TEST_N / QK_ROCMFP4];
    block_rocmfp4_fast q_fast[TEST_N / QK_ROCMFP4];

    fill_input(src);
    rocmfp4_quantize_row_q4_0_ref(src, q_dual, TEST_N);
    rocmfp4_quantize_row_q4_0_fast_ref(src, q_fast, TEST_N);

    assert(rocmfp4_validate_row_data(q_dual, sizeof(q_dual)));
    assert(rocmfp4_validate_row_data_fast(q_fast, sizeof(q_fast)));

    rocmfp4_dequantize_row_q4_0(q_dual, dual, TEST_N);
    rocmfp4_dequantize_row_q4_0_fast(q_fast, fast, TEST_N);

    const float dual_mse = mse(src, dual);
    const float fast_mse = mse(src, fast);
    printf("ROCmFP4: dual_mse=%g fast_mse=%g\n", dual_mse, fast_mse);
    assert(isfinite(dual_mse) && dual_mse < 0.1f);
    assert(isfinite(fast_mse) && fast_mse < 0.1f);
}

static void check_nonfinite_imatrix_input(void) {
    float src[TEST_N];
    float weights[TEST_N];
    float dual[TEST_N];
    float fast[TEST_N];
    block_rocmfp4 q_dual[TEST_N / QK_ROCMFP4];
    block_rocmfp4_fast q_fast[TEST_N / QK_ROCMFP4];

    fill_input(src);
    for (int i = 0; i < TEST_N; ++i) {
        weights[i] = 1.0f + (float) (i % 5);
    }

    src[3] = NAN;
    src[9] = INFINITY;
    src[37] = -INFINITY;
    src[55] = FLT_MAX;
    weights[5] = NAN;
    weights[11] = INFINITY;
    weights[13] = -1.0f;

    assert(rocmfp4_quantize_q4_0(src, q_dual, 1, TEST_N, weights) == sizeof(q_dual));
    assert(rocmfp4_quantize_q4_0_fast(src, q_fast, 1, TEST_N, weights) == sizeof(q_fast));
    assert(rocmfp4_validate_row_data(q_dual, sizeof(q_dual)));
    assert(rocmfp4_validate_row_data_fast(q_fast, sizeof(q_fast)));

    rocmfp4_dequantize_row_q4_0(q_dual, dual, TEST_N);
    rocmfp4_dequantize_row_q4_0_fast(q_fast, fast, TEST_N);
    for (int i = 0; i < TEST_N; ++i) {
        assert(isfinite(dual[i]));
        assert(isfinite(fast[i]));
    }

    assert(dual[3] == 0.0f && dual[9] == 0.0f && dual[37] == 0.0f);
    assert(fast[3] == 0.0f && fast[9] == 0.0f && fast[37] == 0.0f);
    assert(dual[7] != 0.0f && fast[7] != 0.0f);
}

static void check_validation_rejects_invalid_scales(void) {
    block_rocmfp4 dual = { { 0 }, { 1, 1 } };
    block_rocmfp4_fast fast = { { 0 }, 1 };

    assert(rocmfp4_validate_row_data(&dual, sizeof(dual)));
    assert(rocmfp4_validate_row_data_fast(&fast, sizeof(fast)));

    dual.e[1] = 0x7f;
    fast.e = 0x80;
    assert(!rocmfp4_validate_row_data(&dual, sizeof(dual)));
    assert(!rocmfp4_validate_row_data_fast(&fast, sizeof(fast)));
    assert(!rocmfp4_validate_row_data(&dual, sizeof(dual) - 1));
    assert(!rocmfp4_validate_row_data_fast(&fast, sizeof(fast) - 1));
}

static void check_reserved_type_name(void) {
    assert(strcmp(ggml_type_name((enum ggml_type) 50), "unknown") == 0);
}

int main(void) {
    check_reference_roundtrip();
    check_nonfinite_imatrix_input();
    check_validation_rejects_invalid_scales();
    check_reserved_type_name();
    return 0;
}
