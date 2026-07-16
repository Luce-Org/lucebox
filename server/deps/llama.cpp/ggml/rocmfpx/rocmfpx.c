#include "rocmfpx.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

// Finite unsigned E4M3 scale bytes decoded to FP32. Precomputed from the same
// exp/mant formula rocmfpx_ue4m3_to_fp32() used to evaluate with ldexpf():
//   exp == 0 -> mant * 2^-10 ; otherwise (8 + mant) * 2^(exp - 11).
// The scale search re-decodes candidate bytes for every block, and dequant
// decodes a scale for every element, so keeping this as a table (identical to
// the former per-call ldexpf result) removes the transcendental from both hot
// paths without changing any produced value.
#define ROCMFPX_SCALE_SUB(M) ((M) * 0x1p-10f)
#define ROCMFPX_SCALE_E(B, M) ((8 + (M)) * (B))

static const float rocmfpx_scale_ue4m3[127] = {
    ROCMFPX_SCALE_SUB(0),      ROCMFPX_SCALE_SUB(1),      ROCMFPX_SCALE_SUB(2),      ROCMFPX_SCALE_SUB(3),
    ROCMFPX_SCALE_SUB(4),      ROCMFPX_SCALE_SUB(5),      ROCMFPX_SCALE_SUB(6),      ROCMFPX_SCALE_SUB(7),
    ROCMFPX_SCALE_E(0x1p-10f,0), ROCMFPX_SCALE_E(0x1p-10f,1), ROCMFPX_SCALE_E(0x1p-10f,2), ROCMFPX_SCALE_E(0x1p-10f,3),
    ROCMFPX_SCALE_E(0x1p-10f,4), ROCMFPX_SCALE_E(0x1p-10f,5), ROCMFPX_SCALE_E(0x1p-10f,6), ROCMFPX_SCALE_E(0x1p-10f,7),
    ROCMFPX_SCALE_E(0x1p-9f,0),  ROCMFPX_SCALE_E(0x1p-9f,1),  ROCMFPX_SCALE_E(0x1p-9f,2),  ROCMFPX_SCALE_E(0x1p-9f,3),
    ROCMFPX_SCALE_E(0x1p-9f,4),  ROCMFPX_SCALE_E(0x1p-9f,5),  ROCMFPX_SCALE_E(0x1p-9f,6),  ROCMFPX_SCALE_E(0x1p-9f,7),
    ROCMFPX_SCALE_E(0x1p-8f,0),  ROCMFPX_SCALE_E(0x1p-8f,1),  ROCMFPX_SCALE_E(0x1p-8f,2),  ROCMFPX_SCALE_E(0x1p-8f,3),
    ROCMFPX_SCALE_E(0x1p-8f,4),  ROCMFPX_SCALE_E(0x1p-8f,5),  ROCMFPX_SCALE_E(0x1p-8f,6),  ROCMFPX_SCALE_E(0x1p-8f,7),
    ROCMFPX_SCALE_E(0x1p-7f,0),  ROCMFPX_SCALE_E(0x1p-7f,1),  ROCMFPX_SCALE_E(0x1p-7f,2),  ROCMFPX_SCALE_E(0x1p-7f,3),
    ROCMFPX_SCALE_E(0x1p-7f,4),  ROCMFPX_SCALE_E(0x1p-7f,5),  ROCMFPX_SCALE_E(0x1p-7f,6),  ROCMFPX_SCALE_E(0x1p-7f,7),
    ROCMFPX_SCALE_E(0x1p-6f,0),  ROCMFPX_SCALE_E(0x1p-6f,1),  ROCMFPX_SCALE_E(0x1p-6f,2),  ROCMFPX_SCALE_E(0x1p-6f,3),
    ROCMFPX_SCALE_E(0x1p-6f,4),  ROCMFPX_SCALE_E(0x1p-6f,5),  ROCMFPX_SCALE_E(0x1p-6f,6),  ROCMFPX_SCALE_E(0x1p-6f,7),
    ROCMFPX_SCALE_E(0x1p-5f,0),  ROCMFPX_SCALE_E(0x1p-5f,1),  ROCMFPX_SCALE_E(0x1p-5f,2),  ROCMFPX_SCALE_E(0x1p-5f,3),
    ROCMFPX_SCALE_E(0x1p-5f,4),  ROCMFPX_SCALE_E(0x1p-5f,5),  ROCMFPX_SCALE_E(0x1p-5f,6),  ROCMFPX_SCALE_E(0x1p-5f,7),
    ROCMFPX_SCALE_E(0x1p-4f,0),  ROCMFPX_SCALE_E(0x1p-4f,1),  ROCMFPX_SCALE_E(0x1p-4f,2),  ROCMFPX_SCALE_E(0x1p-4f,3),
    ROCMFPX_SCALE_E(0x1p-4f,4),  ROCMFPX_SCALE_E(0x1p-4f,5),  ROCMFPX_SCALE_E(0x1p-4f,6),  ROCMFPX_SCALE_E(0x1p-4f,7),
    ROCMFPX_SCALE_E(0x1p-3f,0),  ROCMFPX_SCALE_E(0x1p-3f,1),  ROCMFPX_SCALE_E(0x1p-3f,2),  ROCMFPX_SCALE_E(0x1p-3f,3),
    ROCMFPX_SCALE_E(0x1p-3f,4),  ROCMFPX_SCALE_E(0x1p-3f,5),  ROCMFPX_SCALE_E(0x1p-3f,6),  ROCMFPX_SCALE_E(0x1p-3f,7),
    ROCMFPX_SCALE_E(0x1p-2f,0),  ROCMFPX_SCALE_E(0x1p-2f,1),  ROCMFPX_SCALE_E(0x1p-2f,2),  ROCMFPX_SCALE_E(0x1p-2f,3),
    ROCMFPX_SCALE_E(0x1p-2f,4),  ROCMFPX_SCALE_E(0x1p-2f,5),  ROCMFPX_SCALE_E(0x1p-2f,6),  ROCMFPX_SCALE_E(0x1p-2f,7),
    ROCMFPX_SCALE_E(0x1p-1f,0),  ROCMFPX_SCALE_E(0x1p-1f,1),  ROCMFPX_SCALE_E(0x1p-1f,2),  ROCMFPX_SCALE_E(0x1p-1f,3),
    ROCMFPX_SCALE_E(0x1p-1f,4),  ROCMFPX_SCALE_E(0x1p-1f,5),  ROCMFPX_SCALE_E(0x1p-1f,6),  ROCMFPX_SCALE_E(0x1p-1f,7),
    ROCMFPX_SCALE_E(0x1p0f,0),   ROCMFPX_SCALE_E(0x1p0f,1),   ROCMFPX_SCALE_E(0x1p0f,2),   ROCMFPX_SCALE_E(0x1p0f,3),
    ROCMFPX_SCALE_E(0x1p0f,4),   ROCMFPX_SCALE_E(0x1p0f,5),   ROCMFPX_SCALE_E(0x1p0f,6),   ROCMFPX_SCALE_E(0x1p0f,7),
    ROCMFPX_SCALE_E(0x1p1f,0),   ROCMFPX_SCALE_E(0x1p1f,1),   ROCMFPX_SCALE_E(0x1p1f,2),   ROCMFPX_SCALE_E(0x1p1f,3),
    ROCMFPX_SCALE_E(0x1p1f,4),   ROCMFPX_SCALE_E(0x1p1f,5),   ROCMFPX_SCALE_E(0x1p1f,6),   ROCMFPX_SCALE_E(0x1p1f,7),
    ROCMFPX_SCALE_E(0x1p2f,0),   ROCMFPX_SCALE_E(0x1p2f,1),   ROCMFPX_SCALE_E(0x1p2f,2),   ROCMFPX_SCALE_E(0x1p2f,3),
    ROCMFPX_SCALE_E(0x1p2f,4),   ROCMFPX_SCALE_E(0x1p2f,5),   ROCMFPX_SCALE_E(0x1p2f,6),   ROCMFPX_SCALE_E(0x1p2f,7),
    ROCMFPX_SCALE_E(0x1p3f,0),   ROCMFPX_SCALE_E(0x1p3f,1),   ROCMFPX_SCALE_E(0x1p3f,2),   ROCMFPX_SCALE_E(0x1p3f,3),
    ROCMFPX_SCALE_E(0x1p3f,4),   ROCMFPX_SCALE_E(0x1p3f,5),   ROCMFPX_SCALE_E(0x1p3f,6),   ROCMFPX_SCALE_E(0x1p3f,7),
    ROCMFPX_SCALE_E(0x1p4f,0),   ROCMFPX_SCALE_E(0x1p4f,1),   ROCMFPX_SCALE_E(0x1p4f,2),   ROCMFPX_SCALE_E(0x1p4f,3),
    ROCMFPX_SCALE_E(0x1p4f,4),   ROCMFPX_SCALE_E(0x1p4f,5),   ROCMFPX_SCALE_E(0x1p4f,6),
};

#undef ROCMFPX_SCALE_SUB
#undef ROCMFPX_SCALE_E

float rocmfpx_ue4m3_to_fp32(uint8_t e) {
    return rocmfpx_scale_is_valid(e) ? rocmfpx_scale_ue4m3[e] : 0.0f;
}

bool rocmfpx_scale_is_valid(uint8_t e) {
    return e <= 0x7e;
}

size_t rocmfpx_row_size_fp2(int64_t k) {
    assert(k % QK_ROCMFP2 == 0);
    return (size_t) (k / QK_ROCMFP2) * sizeof(block_rocmfp2);
}

size_t rocmfpx_row_size_fp3(int64_t k) {
    assert(k % QK_ROCMFP3 == 0);
    return (size_t) (k / QK_ROCMFP3) * sizeof(block_rocmfp3);
}

size_t rocmfpx_row_size_fp6(int64_t k) {
    assert(k % QK_ROCMFP6 == 0);
    return (size_t) (k / QK_ROCMFP6) * sizeof(block_rocmfp6);
}

size_t rocmfpx_row_size_fp8(int64_t k) {
    assert(k % QK_ROCMFP8 == 0);
    return (size_t) (k / QK_ROCMFP8) * sizeof(block_rocmfp8);
}

static uint8_t rocmfpx_nearest_scale_ue4m3(float target) {
    if (!(target > 0.0f) || !isfinite(target)) {
        return 0;
    }

    uint8_t best_e = 1;
    float best_err = fabsf(rocmfpx_ue4m3_to_fp32(best_e) - target);

    for (int e = 2; e <= 0x7e; ++e) {
        const float err = fabsf(rocmfpx_ue4m3_to_fp32((uint8_t) e) - target);
        if (err < best_err) {
            best_err = err;
            best_e = (uint8_t) e;
        }
    }

    return best_e;
}

static float rocmfpx_max_abs(const float * x, int n) {
    float max_abs = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float ax = fabsf(x[i]);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    return max_abs;
}

static float rocmfpx_row_sigma2(const float * x, int64_t k) {
    float sum_x2 = 0.0f;
    bool overflow = false;

    for (int64_t i = 0; i < k; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float x2 = x[i]*x[i];
        if (!isfinite(x2) || sum_x2 > FLT_MAX - x2) {
            overflow = true;
            break;
        }
        sum_x2 += x2;
    }

    if (!overflow) {
        return sum_x2 / (float) k;
    }

    double sum_x2_wide = 0.0;
    for (int64_t i = 0; i < k; ++i) {
        if (isfinite(x[i])) {
            const double value = x[i];
            sum_x2_wide += value*value;
        }
    }
    return (float) fmin(sum_x2_wide / (double) k, (double) FLT_MAX);
}

static void rocmfpx_prepare_mse_weights(
        float * dst, const float * x, int n, const float * quant_weights, float sigma2,
        float * max_abs, float * max_abs_weight) {
    *max_abs = 0.0f;
    *max_abs_weight = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        const float qw = quant_weights[i];
        float weight = 0.0f;
        if (isfinite(qw) && qw > 0.0f && isfinite(x[i])) {
            const float energy2 = sigma2 + x[i]*x[i];
            const float candidate = isfinite(energy2) ? qw*sqrtf(energy2) : FLT_MAX;
            weight = isfinite(candidate) ? candidate : FLT_MAX;
        }

        if (isfinite(x[i])) {
            if (ax > *max_abs) {
                *max_abs = ax;
                *max_abs_weight = weight;
            } else if (ax == *max_abs && weight > *max_abs_weight) {
                *max_abs_weight = weight;
            }
        }

        // Match llama.cpp imatrix weighting style: calibration importance is
        // scaled by row energy so large activations stay protected.
        dst[i] = weight;
    }
}

static void rocmfpx_set_bits(uint8_t * dst, int bit_pos, int nbits, uint32_t code) {
    for (int bit = 0; bit < nbits; ++bit) {
        const int absolute_bit = bit_pos + bit;
        const int byte_index   = absolute_bit >> 3;
        const int bit_index    = absolute_bit & 7;

        if ((code >> bit) & 1u) {
            dst[byte_index] |= (uint8_t) (1u << bit_index);
        }
    }
}

static uint32_t rocmfpx_get_bits(const uint8_t * src, int bit_pos, int nbits) {
    uint32_t code = 0;

    for (int bit = 0; bit < nbits; ++bit) {
        const int absolute_bit = bit_pos + bit;
        const int byte_index   = absolute_bit >> 3;
        const int bit_index    = absolute_bit & 7;

        code |= (uint32_t) ((src[byte_index] >> bit_index) & 1u) << bit;
    }

    return code;
}

// Starting 2-bit ROCmFP2 codebook. Keep this single definition easy to tune.
// TODO: affine scale+min would likely improve quality, but it would break the
// ROCmFPx family's unsigned-UE4M3 scale contract; revisit for a v2 layout.
static const float kvalues_rocmfp2[4] = ROCMFP2_KVALUES_INIT;

static uint8_t rocmfpx_quantize_fp2_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 1;
    }

    const float q = x * inv_scale;
    uint8_t best_code = 0;
    float best_err = fabsf(q - kvalues_rocmfp2[0]);

    for (uint8_t code = 1; code < 4; ++code) {
        const float err = fabsf(q - kvalues_rocmfp2[code]);
        if (err < best_err) {
            best_err = err;
            best_code = code;
        }
    }

    return best_code;
}

static inline float rocmfpx_fp2_decoded_mag(float x, float inv_scale) {
    return kvalues_rocmfp2[rocmfpx_quantize_fp2_code(x, inv_scale)];
}

static float rocmfpx_fp2_block_mse_for_scale(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float y = rocmfpx_fp2_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfpx_fp2_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float y = rocmfpx_fp2_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp2_mse_impl(const float * x, int n, const float * mse_weights, float max_abs, float max_abs_weight) {
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 2.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_ue4m3_to_fp32((uint8_t) e0);
            const float clip_delta = max_abs - 2.0f*scale;
            const float clip_err = mse_weights ? max_abs_weight*clip_delta*clip_delta : clip_delta*clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = mse_weights ?
                    rocmfpx_fp2_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err) :
                    rocmfpx_fp2_block_mse_for_scale(x, n, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = mse_weights ?
                rocmfpx_fp2_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err) :
                rocmfpx_fp2_block_mse_for_scale(x, n, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

static uint8_t rocmfpx_choose_scale_fp2_mse(const float * x, int n) {
    const float max_abs = rocmfpx_max_abs(x, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp2_mse_impl(x, n, NULL, max_abs, 0.0f);
}

static uint8_t rocmfpx_choose_scale_fp2_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP2);
    float mse_weights[QK_ROCMFP2];
    float max_abs;
    float max_abs_weight;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp2_mse_impl(x, n, mse_weights, max_abs, max_abs_weight);
}

static int rocmfpx_decode_fp3_code(uint8_t code) {
    static const int mag[4] = { 0, 1, 2, 4 };
    const int value = mag[code & 3u];
    return (code & 4u) ? -value : value;
}

static uint8_t rocmfpx_quantize_fp3_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    const float ax = fabsf(x * inv_scale);
    uint8_t mag;

    if (ax <= 0.5f) {
        mag = 0;
    } else if (ax <= 1.5f) {
        mag = 1;
    } else if (ax <= 3.0f) {
        mag = 2;
    } else {
        mag = 3;
    }

    return mag == 0 ? 0 : (uint8_t) ((x < 0.0f ? 4u : 0u) | mag);
}

// Fused threshold + decode used only inside the exhaustive scale search, which
// re-scans every element for every candidate scale byte. Returns the same
// signed decoded magnitude that
// rocmfpx_decode_fp3_code(rocmfpx_quantize_fp3_code(x, inv_scale)) produces
// (fp3 magnitudes {0,1,2,4}), so quantized output stays bit-identical.
static inline float rocmfpx_fp3_decoded_mag(float x, float inv_scale) {
    const float a = fabsf(x * inv_scale);
    float mag;
    if (a <= 0.5f) {
        return 0.0f;
    } else if (a <= 1.5f) {
        mag = 1.0f;
    } else if (a <= 3.0f) {
        mag = 2.0f;
    } else {
        mag = 4.0f;
    }
    return x < 0.0f ? -mag : mag;
}

static float rocmfpx_fp3_block_mse_for_scale(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfpx_fp3_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp3_mse_impl(const float * x, int n, const float * mse_weights, float max_abs, float max_abs_weight) {
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 4.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_ue4m3_to_fp32((uint8_t) e0);
            const float clip_delta = max_abs - 4.0f*scale;
            const float clip_err = mse_weights ? max_abs_weight*clip_delta*clip_delta : clip_delta*clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = mse_weights ?
                    rocmfpx_fp3_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err) :
                    rocmfpx_fp3_block_mse_for_scale(x, n, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = mse_weights ?
                rocmfpx_fp3_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err) :
                rocmfpx_fp3_block_mse_for_scale(x, n, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

static uint8_t rocmfpx_choose_scale_fp3_mse(const float * x, int n) {
    const float max_abs = rocmfpx_max_abs(x, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp3_mse_impl(x, n, NULL, max_abs, 0.0f);
}

static uint8_t rocmfpx_choose_scale_fp3_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP3);
    float mse_weights[QK_ROCMFP3];
    float max_abs;
    float max_abs_weight;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp3_mse_impl(x, n, mse_weights, max_abs, max_abs_weight);
}

static int rocmfpx_decode_fp6_code(uint8_t code) {
    const int value = code & 31u;
    return (code & 32u) ? -value : value;
}

static uint8_t rocmfpx_quantize_fp6_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    const float scaled = fminf(fabsf(x * inv_scale), 31.0f);
    const int mag = (int) lroundf(scaled);

    return mag == 0 ? 0 : (uint8_t) ((x < 0.0f ? 32u : 0u) | (uint8_t) mag);
}

// Fused round + clamp + decode for the fp6 scale search. Returns the same signed
// decoded magnitude as rocmfpx_decode_fp6_code(rocmfpx_quantize_fp6_code(...))
// (nearest integer in [0,31], signed), keeping quantized output bit-identical.
static inline float rocmfpx_fp6_decoded_mag(float x, float inv_scale) {
    const float scaled = fminf(fabsf(x * inv_scale), 31.0f);
    const int mag = (int) lroundf(scaled);
    if (mag == 0) {
        return 0.0f;
    }
    return x < 0.0f ? -(float) mag : (float) mag;
}

static float rocmfpx_fp6_block_mse_for_scale(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float y = rocmfpx_fp6_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfpx_fp6_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float y = rocmfpx_fp6_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp6_mse_impl(const float * x, int n, const float * mse_weights, float max_abs, float max_abs_weight) {
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 31.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_ue4m3_to_fp32((uint8_t) e0);
            const float clip_delta = max_abs - 31.0f*scale;
            const float clip_err = mse_weights ? max_abs_weight*clip_delta*clip_delta : clip_delta*clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = mse_weights ?
                    rocmfpx_fp6_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err) :
                    rocmfpx_fp6_block_mse_for_scale(x, n, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = mse_weights ?
                rocmfpx_fp6_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err) :
                rocmfpx_fp6_block_mse_for_scale(x, n, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

static uint8_t rocmfpx_choose_scale_fp6_mse(const float * x, int n) {
    const float max_abs = rocmfpx_max_abs(x, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp6_mse_impl(x, n, NULL, max_abs, 0.0f);
}

static uint8_t rocmfpx_choose_scale_fp6_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP6);
    float mse_weights[QK_ROCMFP6];
    float max_abs;
    float max_abs_weight;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp6_mse_impl(x, n, mse_weights, max_abs, max_abs_weight);
}

static int8_t rocmfpx_quantize_fp8_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    const float scaled = fmaxf(-127.0f, fminf(x * inv_scale, 127.0f));
    const int q = (int) lroundf(scaled);
    return (int8_t) q;
}

static float rocmfpx_fp8_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const int8_t code = rocmfpx_quantize_fp8_code(x[i], inv_scale);
        const float y = (float) code * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp8_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP8);
    float mse_weights[QK_ROCMFP8];
    float max_abs;
    float max_abs_weight;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 127.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_ue4m3_to_fp32((uint8_t) e0);
            const float clip_delta = max_abs - 127.0f*scale;
            if (clip_delta > 0.0f && max_abs_weight*clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = rocmfpx_fp8_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfpx_fp8_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

void rocmfpx_quantize_row_fp2_ref(const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP2 == 0);

    const int64_t nb = k / QK_ROCMFP2;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP2;
        block_rocmfp2 * yb = y + ib;

        memset(yb->qs, 0, sizeof(yb->qs));

        for (int half = 0; half < 2; ++half) {
            const float * xh = xb + half*(QK_ROCMFP2/2);
            yb->e[half] = rocmfpx_choose_scale_fp2_mse(xh, QK_ROCMFP2/2);

            const float scale = rocmfpx_ue4m3_to_fp32(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            for (int j = 0; j < QK_ROCMFP2/2; ++j) {
                const int i = half*(QK_ROCMFP2/2) + j;
                const uint8_t code = rocmfpx_quantize_fp2_code(xb[i], inv_scale);
                yb->qs[i >> 2] |= (uint8_t) (code << (2*(i & 3)));
            }
        }
    }
}

static void rocmfpx_quantize_row_fp2_weighted(
        const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP2 == 0);

    const float sigma2 = rocmfpx_row_sigma2(x, k);

    const int64_t nb = k / QK_ROCMFP2;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP2;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP2 : NULL;
        block_rocmfp2 * yb = y + ib;

        memset(yb->qs, 0, sizeof(yb->qs));

        for (int half = 0; half < 2; ++half) {
            const int half_off = half*(QK_ROCMFP2/2);
            const float * xh = xb + half_off;
            const float * qh = qw ? qw + half_off : NULL;
            yb->e[half] = qh ?
                rocmfpx_choose_scale_fp2_weighted_mse(xh, QK_ROCMFP2/2, qh, sigma2) :
                rocmfpx_choose_scale_fp2_mse(xh, QK_ROCMFP2/2);

            const float scale = rocmfpx_ue4m3_to_fp32(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            for (int j = 0; j < QK_ROCMFP2/2; ++j) {
                const int i = half_off + j;
                const uint8_t code = rocmfpx_quantize_fp2_code(xb[i], inv_scale);
                yb->qs[i >> 2] |= (uint8_t) (code << (2*(i & 3)));
            }
        }
    }
}

void rocmfpx_dequantize_row_fp2(const block_rocmfp2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP2 == 0);

    const int64_t nb = k / QK_ROCMFP2;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp2 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP2;

        for (int i = 0; i < QK_ROCMFP2; ++i) {
            const float scale = rocmfpx_ue4m3_to_fp32(xb->e[i >= QK_ROCMFP2/2]);
            const uint8_t code = (uint8_t) ((xb->qs[i >> 2] >> (2*(i & 3))) & 3u);
            yb[i] = kvalues_rocmfp2[code] * scale;
        }
    }
}

void rocmfpx_quantize_row_fp2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp2_ref(x, (block_rocmfp2 *) y, k);
}

size_t rocmfpx_quantize_fp2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp2(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp2_weighted(src + row*n_per_row, (block_rocmfp2 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp2_ref(src + row*n_per_row, (block_rocmfp2 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

void rocmfpx_quantize_row_fp3_ref(const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP3 == 0);

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP3;
        block_rocmfp3 * yb = y + ib;

        memset(yb->qs, 0, sizeof(yb->qs));

        for (int half = 0; half < 2; ++half) {
            const float * xh = xb + half*(QK_ROCMFP3/2);
            yb->e[half] = rocmfpx_choose_scale_fp3_mse(xh, QK_ROCMFP3/2);

            const float scale = rocmfpx_ue4m3_to_fp32(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            for (int j = 0; j < QK_ROCMFP3/2; ++j) {
                const int i = half*(QK_ROCMFP3/2) + j;
                const uint8_t code = rocmfpx_quantize_fp3_code(xb[i], inv_scale);
                rocmfpx_set_bits(yb->qs, i*3, 3, code);
            }
        }
    }
}

static void rocmfpx_quantize_row_fp3_weighted(
        const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP3 == 0);

    const float sigma2 = rocmfpx_row_sigma2(x, k);

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP3;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP3 : NULL;
        block_rocmfp3 * yb = y + ib;

        memset(yb->qs, 0, sizeof(yb->qs));

        for (int half = 0; half < 2; ++half) {
            const int half_off = half*(QK_ROCMFP3/2);
            const float * xh = xb + half_off;
            const float * qh = qw ? qw + half_off : NULL;
            yb->e[half] = qh ?
                rocmfpx_choose_scale_fp3_weighted_mse(xh, QK_ROCMFP3/2, qh, sigma2) :
                rocmfpx_choose_scale_fp3_mse(xh, QK_ROCMFP3/2);

            const float scale = rocmfpx_ue4m3_to_fp32(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            for (int j = 0; j < QK_ROCMFP3/2; ++j) {
                const int i = half_off + j;
                const uint8_t code = rocmfpx_quantize_fp3_code(xb[i], inv_scale);
                rocmfpx_set_bits(yb->qs, i*3, 3, code);
            }
        }
    }
}

void rocmfpx_dequantize_row_fp3(const block_rocmfp3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP3 == 0);

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp3 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP3;

        for (int i = 0; i < QK_ROCMFP3; ++i) {
            const float scale = rocmfpx_ue4m3_to_fp32(xb->e[i >= QK_ROCMFP3/2]);
            const uint8_t code = (uint8_t) rocmfpx_get_bits(xb->qs, i*3, 3);
            yb[i] = (float) rocmfpx_decode_fp3_code(code) * scale;
        }
    }
}

void rocmfpx_quantize_row_fp3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp3_ref(x, (block_rocmfp3 *) y, k);
}

size_t rocmfpx_quantize_fp3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp3(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp3_weighted(src + row*n_per_row, (block_rocmfp3 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp3_ref(src + row*n_per_row, (block_rocmfp3 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

static void rocmfpx_quantize_row_fp6_weighted(
        const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP6 == 0);

    const float sigma2 = rocmfpx_row_sigma2(x, k);

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP6;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP6 : NULL;
        block_rocmfp6 * yb = y + ib;

        memset(yb->qs, 0, sizeof(yb->qs));

        for (int half = 0; half < 2; ++half) {
            const int half_off = half*(QK_ROCMFP6/2);
            const float * xh = xb + half_off;
            const float * qh = qw ? qw + half_off : NULL;
            yb->e[half] = qh ?
                rocmfpx_choose_scale_fp6_weighted_mse(xh, QK_ROCMFP6/2, qh, sigma2) :
                rocmfpx_choose_scale_fp6_mse(xh, QK_ROCMFP6/2);

            const float scale = rocmfpx_ue4m3_to_fp32(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            for (int j = 0; j < QK_ROCMFP6/2; ++j) {
                const int i = half_off + j;
                const uint8_t code = rocmfpx_quantize_fp6_code(xb[i], inv_scale);
                rocmfpx_set_bits(yb->qs, i*6, 6, code);
            }
        }
    }
}

void rocmfpx_quantize_row_fp6_ref(const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP6 == 0);

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP6;
        block_rocmfp6 * yb = y + ib;

        memset(yb->qs, 0, sizeof(yb->qs));

        for (int half = 0; half < 2; ++half) {
            const float * xh = xb + half*(QK_ROCMFP6/2);
            yb->e[half] = rocmfpx_choose_scale_fp6_mse(xh, QK_ROCMFP6/2);

            const float scale = rocmfpx_ue4m3_to_fp32(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            for (int j = 0; j < QK_ROCMFP6/2; ++j) {
                const int i = half*(QK_ROCMFP6/2) + j;
                const uint8_t code = rocmfpx_quantize_fp6_code(xb[i], inv_scale);
                rocmfpx_set_bits(yb->qs, i*6, 6, code);
            }
        }
    }
}

void rocmfpx_dequantize_row_fp6(const block_rocmfp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP6 == 0);

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp6 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP6;

        for (int i = 0; i < QK_ROCMFP6; ++i) {
            const float scale = rocmfpx_ue4m3_to_fp32(xb->e[i >= QK_ROCMFP6/2]);
            const uint8_t code = (uint8_t) rocmfpx_get_bits(xb->qs, i*6, 6);
            yb[i] = (float) rocmfpx_decode_fp6_code(code) * scale;
        }
    }
}

void rocmfpx_quantize_row_fp6(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp6_ref(x, (block_rocmfp6 *) y, k);
}

size_t rocmfpx_quantize_fp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp6(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp6_weighted(src + row*n_per_row, (block_rocmfp6 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp6_ref(src + row*n_per_row, (block_rocmfp6 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

static void rocmfpx_quantize_row_fp8_weighted(
        const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP8 == 0);

    const float sigma2 = rocmfpx_row_sigma2(x, k);

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP8;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP8 : NULL;
        block_rocmfp8 * yb = y + ib;

        yb->e = qw ? rocmfpx_choose_scale_fp8_weighted_mse(xb, QK_ROCMFP8, qw, sigma2) :
                     rocmfpx_nearest_scale_ue4m3(rocmfpx_max_abs(xb, QK_ROCMFP8) / 127.0f);

        const float scale = rocmfpx_ue4m3_to_fp32(yb->e);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb->qs[i] = rocmfpx_quantize_fp8_code(xb[i], inv_scale);
        }
    }
}

void rocmfpx_quantize_row_fp8_ref(const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP8 == 0);

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP8;
        block_rocmfp8 * yb = y + ib;

        const float max_abs = rocmfpx_max_abs(xb, QK_ROCMFP8);
        yb->e = rocmfpx_nearest_scale_ue4m3(max_abs / 127.0f);

        const float scale = rocmfpx_ue4m3_to_fp32(yb->e);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb->qs[i] = rocmfpx_quantize_fp8_code(xb[i], inv_scale);
        }
    }
}

void rocmfpx_dequantize_row_fp8(const block_rocmfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP8 == 0);

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp8 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP8;

        const float scale = rocmfpx_ue4m3_to_fp32(xb->e);
        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb[i] = (float) xb->qs[i] * scale;
        }
    }
}

void rocmfpx_quantize_row_fp8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp8_ref(x, (block_rocmfp8 *) y, k);
}

size_t rocmfpx_quantize_fp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp8(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp8_weighted(src + row*n_per_row, (block_rocmfp8 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp8_ref(src + row*n_per_row, (block_rocmfp8 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

bool rocmfpx_validate_row_data_fp2(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp2) != 0) {
        return false;
    }

    const block_rocmfp2 * blocks = (const block_rocmfp2 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp2);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }

    return true;
}

bool rocmfpx_validate_row_data_fp3(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp3) != 0) {
        return false;
    }

    const block_rocmfp3 * blocks = (const block_rocmfp3 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp3);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }

    return true;
}

bool rocmfpx_validate_row_data_fp6(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp6) != 0) {
        return false;
    }

    const block_rocmfp6 * blocks = (const block_rocmfp6 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp6);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }

    return true;
}

bool rocmfpx_validate_row_data_fp8(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp8) != 0) {
        return false;
    }

    const block_rocmfp8 * blocks = (const block_rocmfp8 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp8);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e)) {
            return false;
        }
    }

    return true;
}
