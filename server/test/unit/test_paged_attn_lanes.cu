// Bit-exactness of the paged attention helpers in ggml-cuda/paged-attn-common.cuh,
// which the kernels rely on to keep the outputs of the plain code they replaced:
//  - paged_attn_reduce_group against warp_reduce_sum per token, and the warp
//    sum, max, xor exchange and lane broadcast against the plain shuffles, on
//    random values with signed zeros and subnormals, in blocks of several warps
//    (on wave64 targets two of them share a wave);
//  - the packed Q8_0 dequantization against __float2half(__half2float(d) * q)
//    for every non-NaN f16 d and every q, with the sign of the exact float
//    product on a zero (ROCm's __float2half(-0.0f) returns +0); where the
//    product is NaN (inf * 0) the packed one must be NaN too;
//  - the transposed V staging against the row-major staging of the same
//    segments: every (token, dim) lands where the other layout puts it, and
//    the padding columns stay untouched.
// Exits 0 on success, 1 on a mismatch, 77 (skip) without a GPU.

#include "common.cuh"
#include "paged-attn-common.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define TEST_CHECK(expr) do { \
    const cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr int kWarps = 4;      // logical warps per block
constexpr int kBlocks = 2048;  // independent random cases of kWarps warps each

enum Check { kGroup2, kGroup1, kWarpSum, kWarpMax, kXorLane, kXorLaneFull, kLaneValue, kChecks };
const char * const kCheckNames[kChecks] = {
    "reduce_group<2> vs warp_reduce_sum", "reduce_group<1> vs warp_reduce_sum",
    "warp_sum vs warp_reduce_sum", "warp_max vs warp_reduce_max",
    "xor_lane vs __shfl_xor_sync", "xor_lane<full_warp> vs __shfl_xor_sync",
    "lane_value vs __shfl_sync",
};

__device__ void count_mismatch(unsigned int * fails, int check, float got, float want) {
    if (__float_as_uint(got) != __float_as_uint(want)) {
        atomicAdd(&fails[check], 1u);
    }
}

// in: per warp two tokens of WARP_SIZE partial sums each.
__global__ void reduce_kernel(const float * in, unsigned int * fails) {
    const int warp = blockIdx.x * kWarps + threadIdx.y;
    const int lane = threadIdx.x;
    const float * w = in + (size_t) warp * 2 * WARP_SIZE;
    const float x0 = w[lane];
    const float x1 = w[WARP_SIZE + lane];

    const float ref0 = warp_reduce_sum(x0);
    const float ref1 = warp_reduce_sum(x1);

    // The group reduction leaves token lane / (WARP_SIZE/2) on each lane.
    float sums2[2] = { x0, x1 };
    count_mismatch(fails, kGroup2, paged_attn_reduce_group<2>(sums2),
                   lane < WARP_SIZE / 2 ? ref0 : ref1);
    float sums1[1] = { x0 };
    count_mismatch(fails, kGroup1, paged_attn_reduce_group<1>(sums1), ref0);
    count_mismatch(fails, kWarpSum, paged_attn_warp_sum(x0), ref0);
    count_mismatch(fails, kWarpMax, paged_attn_warp_max(x0), warp_reduce_max(x0));
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        count_mismatch(fails, kXorLane, paged_attn_xor_lane(x1, offset),
                       __shfl_xor_sync(0xFFFFFFFF, x1, offset, WARP_SIZE));
        // Every lane of the warp is active here.
        count_mismatch(fails, kXorLaneFull, paged_attn_xor_lane<true>(x1, offset),
                       __shfl_xor_sync(0xFFFFFFFF, x1, offset, WARP_SIZE));
    }
    for (int src = 0; src < WARP_SIZE; ++src) {
        count_mismatch(fails, kLaneValue, paged_attn_lane_value(x0, src),
                       __shfl_sync(0xFFFFFFFF, x0, src, WARP_SIZE));
    }
}

// One thread per f16 scale d, all 256 quants. zero_sign counts the results
// that differ from __float2half only in the sign of a zero.
__global__ void q8_0_kernel(unsigned int * fails, unsigned int * first, unsigned int * zero_sign) {
    const unsigned int d_bits = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_bits > 0xFFFFu) {
        return;
    }
    if ((d_bits & 0x7C00u) == 0x7C00u && (d_bits & 0x03FFu) != 0) {
        return; // NaN scale
    }
    const half d = __ushort_as_half((unsigned short) d_bits);
    const half2 d2 = __half2half2(d);
    for (int word = 0; word < 64; ++word) {
        const uint32_t qs = (uint32_t) (4*word) | (uint32_t) (4*word + 1) << 8 |
                            (uint32_t) (4*word + 2) << 16 | (uint32_t) (4*word + 3) << 24;
        uint32_t h[2];
        paged_attn_q8_0_to_half2(qs, d2, h);
        for (int i = 0; i < 4; ++i) {
            const int q = (int8_t) (4*word + i);
            const unsigned short got = (unsigned short) (h[i / 2] >> (16 * (i % 2)));
            // The float product is exact (11 by 8 significant bits), so it
            // carries the IEEE sign of a zero result.
            const float product = __half2float(d) * (float) q;
            const unsigned short converted = __half_as_ushort(__float2half(product));
            const unsigned short want = product == 0.0f
                ? (unsigned short) (__float_as_uint(product) >> 16) : converted;
            if (want != converted) {
                atomicAdd(zero_sign, 1u);
            }
            const bool want_nan = (want & 0x7C00u) == 0x7C00u && (want & 0x03FFu) != 0;
            const bool got_nan = (got & 0x7C00u) == 0x7C00u && (got & 0x03FFu) != 0;
            if (want_nan ? !got_nan : got != want) {
                if (atomicAdd(fails, 1u) == 0) {
                    first[0] = d_bits;
                    first[1] = (unsigned int) (q & 0xFF);
                    first[2] = got;
                    first[3] = want;
                }
            }
        }
    }
}

constexpr int kStageRows = 64;                            // tokens of one tile
constexpr int kStageH2 = 64;                              // half2s of one token row
constexpr int kVec = kStageH2 / PAGED_ATTN_WMMA_ROW_THREADS;
constexpr int kStridePlain = kStageH2 + 4;                // tokens x dims
constexpr int kStrideT = kStageRows / 2 + 4;              // dims x tokens
constexpr int kDims = 2 * kStageH2;

__host__ __device__ unsigned short stage_code(int token, int dim) {
    return (unsigned short) (token << 7 | dim); // finite f16 bit patterns
}

// The thread layout of paged_attn_wmma_stage_tile: 8 warps, thread tid stages
// segment tid % PAGED_ATTN_WMMA_ROW_THREADS of token row tid / PAGED_ATTN_WMMA_ROW_THREADS.
__global__ void stage_kernel(half2 * plain, half2 * transposed) {
    const int tid = threadIdx.x + threadIdx.y * WARP_SIZE;
    const int row = tid / PAGED_ATTN_WMMA_ROW_THREADS;
    const int sub = tid % PAGED_ATTN_WMMA_ROW_THREADS;
    uint32_t h[kVec];
    for (int m = 0; m < kVec; ++m) {
        const int dim = 2 * (sub * kVec + m);
        h[m] = (uint32_t) stage_code(row, dim) | (uint32_t) stage_code(row, dim + 1) << 16;
    }
    paged_attn_wmma_store_segment<kVec>(
        reinterpret_cast<uint32_t *>(plain + row * kStridePlain + sub * kVec), h);
    paged_attn_wmma_store_transposed<kVec, kStrideT>(transposed, row, sub, h);
}

std::vector<float> random_partials() {
    std::mt19937 rng(20261110);
    std::uniform_int_distribution<int> kind(0, 9);
    std::uniform_int_distribution<int> exponent(-40, 40);
    std::uniform_int_distribution<uint32_t> bits;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> v((size_t) kBlocks * kWarps * 2 * WARP_SIZE);
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t u;
        float f;
        switch (kind(rng)) {
            case 0:  f = (bits(rng) & 1) ? -0.0f : 0.0f; break;
            case 1:  u = (bits(rng) & 0x807FFFFFu) | 1u; std::memcpy(&f, &u, 4); break; // subnormal
            case 2:  f = i > 0 ? -v[i - 1] : 1.0f; break;                                // cancels
            case 3:  f = std::ldexp(normal(rng), exponent(rng)); break;
            default: f = normal(rng); break;
        }
        v[i] = f;
    }
    return v;
}

bool run_reduce() {
    const std::vector<float> in = random_partials();
    float * d_in = nullptr;
    unsigned int * d_fails = nullptr;
    TEST_CHECK(cudaMalloc(&d_in, in.size() * sizeof(float)));
    TEST_CHECK(cudaMalloc(&d_fails, kChecks * sizeof(unsigned int)));
    TEST_CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
    TEST_CHECK(cudaMemset(d_fails, 0, kChecks * sizeof(unsigned int)));
    reduce_kernel<<<kBlocks, dim3(WARP_SIZE, kWarps)>>>(d_in, d_fails);
    TEST_CHECK(cudaGetLastError());
    unsigned int fails[kChecks];
    TEST_CHECK(cudaMemcpy(fails, d_fails, sizeof(fails), cudaMemcpyDeviceToHost));
    TEST_CHECK(cudaFree(d_in));
    TEST_CHECK(cudaFree(d_fails));
    bool ok = true;
    for (int c = 0; c < kChecks; ++c) {
        std::printf("[paged-attn-lanes] %-36s %s (%u mismatches)\n",
                    kCheckNames[c], fails[c] == 0 ? "ok" : "FAIL", fails[c]);
        ok = ok && fails[c] == 0;
    }
    return ok;
}

bool run_q8_0() {
    unsigned int * d_out = nullptr;
    TEST_CHECK(cudaMalloc(&d_out, 6 * sizeof(unsigned int)));
    TEST_CHECK(cudaMemset(d_out, 0, 6 * sizeof(unsigned int)));
    q8_0_kernel<<<65536 / 256, 256>>>(d_out, d_out + 1, d_out + 5);
    TEST_CHECK(cudaGetLastError());
    unsigned int out[6];
    TEST_CHECK(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost));
    TEST_CHECK(cudaFree(d_out));
    std::printf("[paged-attn-lanes] %-36s %s (%u mismatches over 65536 d x 256 q)\n",
                "q8_0 packed dequant vs float", out[0] == 0 ? "ok" : "FAIL", out[0]);
    if (out[0] != 0) {
        std::printf("[paged-attn-lanes]   first: d=0x%04x q=%d got=0x%04x want=0x%04x\n",
                    out[1], (int) (int8_t) out[2], out[3], out[4]);
    }
    std::printf("[paged-attn-lanes]   %u zero products where __float2half drops the sign\n", out[5]);
    return out[0] == 0;
}

bool run_stage() {
    const size_t n_plain = (size_t) kStageRows * kStridePlain;
    const size_t n_t = (size_t) kDims * kStrideT;
    half2 * d_plain = nullptr;
    half2 * d_t = nullptr;
    TEST_CHECK(cudaMalloc(&d_plain, n_plain * sizeof(half2)));
    TEST_CHECK(cudaMalloc(&d_t, n_t * sizeof(half2)));
    TEST_CHECK(cudaMemset(d_plain, 0xFF, n_plain * sizeof(half2)));
    TEST_CHECK(cudaMemset(d_t, 0xFF, n_t * sizeof(half2)));
    stage_kernel<<<1, dim3(WARP_SIZE, kStageRows * PAGED_ATTN_WMMA_ROW_THREADS / WARP_SIZE)>>>(d_plain, d_t);
    TEST_CHECK(cudaGetLastError());
    std::vector<unsigned short> plain(2 * n_plain), t(2 * n_t);
    TEST_CHECK(cudaMemcpy(plain.data(), d_plain, n_plain * sizeof(half2), cudaMemcpyDeviceToHost));
    TEST_CHECK(cudaMemcpy(t.data(), d_t, n_t * sizeof(half2), cudaMemcpyDeviceToHost));
    TEST_CHECK(cudaFree(d_plain));
    TEST_CHECK(cudaFree(d_t));

    int bad = 0;
    for (int token = 0; token < kStageRows; ++token) {
        for (int dim = 0; dim < kDims; ++dim) {
            const unsigned short want = stage_code(token, dim);
            const size_t at_plain = 2 * ((size_t) token * kStridePlain + dim / 2) + dim % 2;
            const size_t at_t = 2 * ((size_t) dim * kStrideT + paged_attn_wmma_pair_col(dim, token / 2)) + token % 2;
            bad += plain[at_plain] != want;
            bad += t[at_t] != want;
        }
    }
    // Every data column was matched above, so only the padding is left.
    for (int dim = 0; dim < kDims; ++dim) {
        for (int col = kStageRows / 2; col < kStrideT; ++col) {
            bad += t[2 * ((size_t) dim * kStrideT + col)] != 0xFFFF;
            bad += t[2 * ((size_t) dim * kStrideT + col) + 1] != 0xFFFF;
        }
    }
    for (int token = 0; token < kStageRows; ++token) {
        for (int col = kStageH2; col < kStridePlain; ++col) {
            bad += plain[2 * ((size_t) token * kStridePlain + col)] != 0xFFFF;
            bad += plain[2 * ((size_t) token * kStridePlain + col) + 1] != 0xFFFF;
        }
    }
    std::printf("[paged-attn-lanes] %-36s %s (%d mismatches)\n",
                "transposed V staging vs row-major", bad == 0 ? "ok" : "FAIL", bad);
    return bad == 0;
}

} // namespace

int main() {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        std::printf("[paged-attn-lanes] SKIP: no GPU\n");
        return 77;
    }
    const bool ok_reduce = run_reduce();
    const bool ok_q8_0 = run_q8_0();
    const bool ok_stage = run_stage();
    const bool ok = ok_reduce && ok_q8_0 && ok_stage;
    std::printf("[paged-attn-lanes] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
