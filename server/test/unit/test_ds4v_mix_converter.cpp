// CPU reference tests for the adaptive MIX block codecs behind qtype 105
// (Q3_1_ROCMFP3_MIX) and qtype 106 (Q2_1_ROCMFP2_MIX).
//
// WHY THIS EXISTS. The generic ggml to_float/from_float entry points ABORT
// for both MIX types, so the only legal CPU path is the adaptive
// codebook pair in rocmfpx.h. A silent fixed-level fallback would decode
// adaptive experts with the wrong levels. These tests pin the contract.
// They exercise the PREDICATE and the byte layout, never the GPU kernels.
// No device work happens here. Every case is pure CPU codec logic.

#include "ggml.h"
#include "rocmfpx.h"
#include "CppUnitTestFramework.hpp"
using CppUnitTestFramework::CommonFixture;
#undef CHECK

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CHECK(cond, msg) REQUIRE(cond)

namespace {

static_assert(sizeof(block_rocmfp2) == 10, "qtype-106 ABI changed");
static_assert(sizeof(block_rocmfp3) == 14, "qtype-105 ABI changed");

struct Ds4MixConverterFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};

static uint16_t f32_to_bf16_bits(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    if (!std::isfinite(x)) {
        return (uint16_t) (u >> 16);
    }
    const uint32_t rounded = u + 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (rounded >> 16);
}

static float bf16_bits_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float x = 0.0f;
    std::memcpy(&x, &u, 4);
    return x;
}

static void fill_books2(uint16_t out[8]) {
    const float b0[4] = {-2.0f, -0.5f, 0.5f, 2.0f};
    const float b1[4] = {-4.0f, -1.0f, 1.0f, 4.0f};
    for (int i = 0; i < 4; ++i) {
        out[i]     = f32_to_bf16_bits(b0[i]);
        out[4 + i] = f32_to_bf16_bits(b1[i]);
    }
}

static void fill_books3(uint16_t out[16]) {
    const float b0[8] = {-4.0f, -2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f, 4.0f};
    const float b1[8] = {-8.0f, -4.0f, -2.0f, -1.0f, 1.0f, 2.0f, 4.0f, 8.0f};
    for (int i = 0; i < 8; ++i) {
        out[i]     = f32_to_bf16_bits(b0[i]);
        out[8 + i] = f32_to_bf16_bits(b1[i]);
    }
}

static float row_max_err(const float * a, const float * b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

TEST_CASE(Ds4MixConverterFixture, fp2_mix_roundtrip) {
    float x[32];
    for (int i = 0; i < 32; ++i) {
        x[i] = -4.0f + (float) i * (8.0f / 31.0f);
    }
    uint16_t books[8];
    fill_books2(books);
    block_rocmfp2 q[1];
    float y[32];
    CHECK(rocmfpx_quantize_row_fp2_mix_ref(x, q, 32, books, nullptr),
          "fp2 mix quantize accepts valid books without imatrix");
    rocmfpx_dequantize_row_fp2_mix(q, y, 32, books);
    CHECK(row_max_err(x, y, 32) < 2.0f, "fp2 mix roundtrip error stays bounded");
}

TEST_CASE(Ds4MixConverterFixture, fp3_mix_roundtrip) {
    float x[32];
    for (int i = 0; i < 32; ++i) {
        x[i] = -4.0f + (float) i * (8.0f / 31.0f);
    }
    uint16_t books[16];
    fill_books3(books);
    block_rocmfp3 q[1];
    float y[32];
    CHECK(rocmfpx_quantize_row_fp3_mix_ref(x, q, 32, books, nullptr),
          "fp3 mix quantize accepts valid books without imatrix");
    rocmfpx_dequantize_row_fp3_mix(q, y, 32, books);
    CHECK(row_max_err(x, y, 32) < 1.0f, "fp3 mix roundtrip error stays bounded");
}

TEST_CASE(Ds4MixConverterFixture, fp2_byte_layout_reference) {
    float x[32];
    for (int i = 0; i < 32; ++i) {
        x[i] = (i % 2) ? 0.21f : -0.21f;
    }
    uint16_t books[8];
    fill_books2(books);
    block_rocmfp2 q[1];
    float y[32];
    CHECK(rocmfpx_quantize_row_fp2_mix_ref(x, q, 32, books, nullptr),
          "fp2 mix quantize succeeds for layout check");
    rocmfpx_dequantize_row_fp2_mix(q, y, 32, books);
    float lvl[2][4];
    for (int b = 0; b < 2; ++b) {
        for (int i = 0; i < 4; ++i) {
            lvl[b][i] = bf16_bits_to_f32(books[b * 4 + i]);
        }
    }
    for (int i = 0; i < 32; ++i) {
        const uint8_t meta = q[0].e[i >= 16];
        const int book = meta >> 7;
        const float scale = rocmfpx_ue4m3_to_fp32(meta & 0x7f);
        const int code = (q[0].qs[i >> 2] >> (2 * (i & 3))) & 3;
        const float expect = scale * lvl[book][code];
        CHECK(y[i] == expect, "fp2 decoded value matches the byte layout exactly");
    }
}

TEST_CASE(Ds4MixConverterFixture, weighted_encode_honors_imatrix) {
    float x[32];
    float w[32];
    for (int i = 0; i < 32; ++i) {
        x[i] = 0.05f;
        w[i] = 0.0f;
    }
    x[0] = 1.9f;
    w[0] = 1000000.0f;
    uint16_t books[8];
    fill_books2(books);
    block_rocmfp2 q[1];
    float y[32];
    CHECK(rocmfpx_quantize_row_fp2_mix_ref(x, q, 32, books, w),
          "fp2 mix quantize accepts an imatrix");
    rocmfpx_dequantize_row_fp2_mix(q, y, 32, books);
    CHECK(std::fabs(y[0] - 1.9f) < 0.5f, "weighted encode protects the important value");
}

TEST_CASE(Ds4MixConverterFixture, malformed_books_rejected) {
    float x[32] = {0.0f};
    block_rocmfp2 q2[1];
    block_rocmfp3 q3[1];
    uint16_t unsorted[8];
    fill_books2(unsorted);
    unsorted[1] = unsorted[0];
    CHECK(!rocmfpx_quantize_row_fp2_mix_ref(x, q2, 32, unsorted, nullptr),
          "fp2 mix rejects an unsorted codebook");
    uint16_t nanbook[8];
    fill_books2(nanbook);
    nanbook[2] = 0x7FC0u;
    CHECK(!rocmfpx_quantize_row_fp2_mix_ref(x, q2, 32, nanbook, nullptr),
          "fp2 mix rejects a non-finite codebook level");
    CHECK(!rocmfpx_quantize_row_fp2_mix_ref(x, q2, 32, nullptr, nullptr),
          "fp2 mix rejects null codebooks");
    uint16_t unsorted3[16];
    fill_books3(unsorted3);
    unsorted3[9] = unsorted3[8];
    CHECK(!rocmfpx_quantize_row_fp3_mix_ref(x, q3, 32, unsorted3, nullptr),
          "fp3 mix rejects an unsorted codebook");
}

} // namespace
