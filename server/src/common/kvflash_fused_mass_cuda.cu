// See kvflash_fused_mass_cuda.h.

#include "kvflash_fused_mass_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace luce::common {

namespace {

constexpr int kCpb = 2;        // chunks per block: q registers are reused across them
constexpr int kTop = 8;        // kExactTopPlanes

__device__ __forceinline__ int dp4a_i8(int a, int b, int c) {
#if defined(__HIP_PLATFORM_AMD__)
#  if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#  elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#  else
    const signed char * x = reinterpret_cast<const signed char *>(&a);
    const signed char * y = reinterpret_cast<const signed char *>(&b);
    return c + x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
#  endif
#else
    return __dp4a(a, b, c);
#endif
}

struct KPtrs {
    const uint8_t * pool[kFusedMaxLayers];
    const uint8_t * copy[kFusedMaxLayers];
    size_t pool_nb2, copy_nb2;
};

struct PartArgs {
    int L, HK, Hq, G, D, nblk, ct, C, nq;   // nq = queries in this tile
    int q0;                                 // first query of the tile (offset into q8/qd)
    int planes;
};

// grid (ceil(C / kCpb), HK, L); block threads cover the tile's G * nq planes of this (layer, kv head), looping when fewer.
// NBLK (= D / 32) and CT are compile-time so the q block and the loops stay in registers.
template <int NBLK, int CT>
__global__ void partials_kernel(KPtrs kp, PartArgs a, const int * __restrict__ src, const int8_t * __restrict__ q8,
                                const float * __restrict__ qd, float * __restrict__ part) {
    __shared__ __align__(16) int   sq[kCpb][CT * NBLK * 8];   // K int8 payload, repacked 4-byte aligned [t][b][8]
    __shared__ float               sd[kCpb][CT * NBLK];       // K block scales
    const int l = blockIdx.z, hk = blockIdx.y, c0 = blockIdx.x * kCpb;
    const int nc = min(kCpb, a.C - c0);
    for (int cc = 0; cc < nc; cc++) {
        const int s = src[c0 + cc];
        const uint8_t * base = s >= 0 ? kp.pool[l] + (size_t)hk * kp.pool_nb2 + (size_t)s * CT * NBLK * 34
                                      : kp.copy[l] + (size_t)hk * kp.copy_nb2 + (size_t)(-1 - s) * CT * NBLK * 34;
        for (int blk = threadIdx.x; blk < CT * NBLK; blk += blockDim.x) {
            const uint8_t * bp = base + (size_t)blk * 34;
            const unsigned short dh = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
            sd[cc][blk] = __half2float(__ushort_as_half(dh));
            int * dst = &sq[cc][blk * 8];
#pragma unroll
            for (int i = 0; i < 8; i++)
                dst[i] = (int)bp[2 + 4 * i] | ((int)bp[3 + 4 * i] << 8) | ((int)bp[4 + 4 * i] << 16) | ((int)bp[5 + 4 * i] << 24);
        }
    }
    __syncthreads();
    const int P = a.nq * a.G;
    const float scale = 1.4426950408889634f / sqrtf((float)(NBLK * 32));   // logits live in base 2: m, s of the partials are log2-domain
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const int i = p / a.G, g = p - i * a.G;
        const int x = l * a.Hq + hk * a.G + g;
        const size_t qi = ((size_t)(a.q0 + i) * a.planes + x);
        const int4 * qp = reinterpret_cast<const int4 *>(q8 + qi * (NBLK * 32));
        int qv[NBLK * 8];
        float qs[NBLK];
#pragma unroll
        for (int b = 0; b < NBLK; b++) {
            const int4 v0 = qp[2 * b], v1 = qp[2 * b + 1];
            qv[b * 8 + 0] = v0.x; qv[b * 8 + 1] = v0.y; qv[b * 8 + 2] = v0.z; qv[b * 8 + 3] = v0.w;
            qv[b * 8 + 4] = v1.x; qv[b * 8 + 5] = v1.y; qv[b * 8 + 6] = v1.z; qv[b * 8 + 7] = v1.w;
            qs[b] = qd[qi * NBLK + b];
        }
        for (int cc = 0; cc < nc; cc++) {
            // online max / sum-exp over the chunk's tokens (no logits array)
            float mx = -INFINITY, sum = 0.0f;
#pragma unroll 2
            for (int t = 0; t < CT; t++) {
                float acc = 0.0f;
#pragma unroll
                for (int b = 0; b < NBLK; b++) {
                    const int4 * kv = reinterpret_cast<const int4 *>(&sq[cc][(t * NBLK + b) * 8]);
                    const int4 k0 = kv[0], k1 = kv[1];
                    int isum = 0;
                    isum = dp4a_i8(k0.x, qv[b * 8 + 0], isum); isum = dp4a_i8(k0.y, qv[b * 8 + 1], isum);
                    isum = dp4a_i8(k0.z, qv[b * 8 + 2], isum); isum = dp4a_i8(k0.w, qv[b * 8 + 3], isum);
                    isum = dp4a_i8(k1.x, qv[b * 8 + 4], isum); isum = dp4a_i8(k1.y, qv[b * 8 + 5], isum);
                    isum = dp4a_i8(k1.z, qv[b * 8 + 6], isum); isum = dp4a_i8(k1.w, qv[b * 8 + 7], isum);
                    acc += (sd[cc][t * NBLK + b] * qs[b]) * (float)isum;
                }
                const float lgt = acc * scale;
                if (lgt > mx) { sum *= exp2f(mx - lgt); mx = lgt; }   // rare after the first tokens; first token: exp2f(-inf) * 0 = 0
                sum += exp2f(lgt - mx);
            }
            float * o = part + (((size_t)i * a.planes + x) * a.C + (c0 + cc)) * 2;
            o[0] = mx;
            o[1] = sum;
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx1200__) || defined(__gfx1201__))
#  define LUCE_FUSED_WMMA 1
typedef int v2i_t __attribute__((ext_vector_type(2)));
typedef int v8i_t __attribute__((ext_vector_type(8)));
#endif

// gfx12 int8 WMMA variant (Amendment 11 lever 2). Same inputs/outputs and the same MMQ numerics as partials_kernel: per 32-block two
// 16x16x16 iu8 WMMAs give the exact int32 dot, then acc += (dK * dQ) * (float)isum in ascending block order.
// A = K tile (16 tokens x 16 k), B = q (16 k x 16 planes): lane l holds token / plane l % 16, k bytes 8 * (l / 16) .. +8;
// D lane l holds plane l % 16 and tokens 8 * (l / 16) + i, i < 8, so the token reduction stays inside a lane (+ one xor-16 shuffle).
// One wave handles 16 (query, plane) columns; the chunk's CT logits sit in registers, so max and sum-exp need no online rescale.
template <int NBLK, int CT>
__global__ void partials_wmma_kernel(KPtrs kp, PartArgs a, const int * __restrict__ src, const int8_t * __restrict__ q8,
                                     const float * __restrict__ qd, float * __restrict__ part) {
#ifdef LUCE_FUSED_WMMA
    constexpr int RS = NBLK * 8 + 4;                          // padded K row (ints): 16 token rows x 2 halves hit distinct banks
    __shared__ __align__(16) int   sq[kCpb][CT * RS];         // K int8 payload [t][b][8 ints]
    __shared__ __align__(16) float sd[kCpb][NBLK * CT];       // K block scales, transposed [b][t]
    const int l = blockIdx.z, hk = blockIdx.y, c0 = blockIdx.x * kCpb;
    const int nc = min(kCpb, a.C - c0);
    for (int cc = 0; cc < nc; cc++) {
        const int s = src[c0 + cc];
        const uint8_t * base = s >= 0 ? kp.pool[l] + (size_t)hk * kp.pool_nb2 + (size_t)s * CT * NBLK * 34
                                      : kp.copy[l] + (size_t)hk * kp.copy_nb2 + (size_t)(-1 - s) * CT * NBLK * 34;
        for (int blk = threadIdx.x; blk < CT * NBLK; blk += blockDim.x) {
            const uint8_t * bp = base + (size_t)blk * 34;
            const int t = blk / NBLK, b = blk - t * NBLK;
            const unsigned short dh = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
            sd[cc][b * CT + t] = __half2float(__ushort_as_half(dh));
            int * dst = &sq[cc][t * RS + b * 8];
#pragma unroll
            for (int i = 0; i < 8; i++)
                dst[i] = (int)bp[2 + 4 * i] | ((int)bp[3 + 4 * i] << 8) | ((int)bp[4 + 4 * i] << 16) | ((int)bp[5 + 4 * i] << 24);
        }
    }
    __syncthreads();
    const int P = a.nq * a.G, ngroups = (P + 15) / 16;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, nwaves = blockDim.x >> 5, n = lane & 15, h = lane >> 4;
    const float scale = 1.4426950408889634f / sqrtf((float)(NBLK * 32));   // logits in base 2
    for (int grp = wave; grp < ngroups; grp += nwaves) {
        const int p = min(grp * 16 + n, P - 1);   // tail columns recompute the last plane and do not store
        const bool store = h == 0 && grp * 16 + n < P;
        const int i = p / a.G, g = p - i * a.G;
        const int x = l * a.Hq + hk * a.G + g;
        const size_t qi = ((size_t)(a.q0 + i) * a.planes + x);
        const int2 * qp = reinterpret_cast<const int2 *>(q8 + qi * (NBLK * 32));
        v2i_t qa[NBLK][2];
        float qs[NBLK];
#pragma unroll
        for (int b = 0; b < NBLK; b++) {
#pragma unroll
            for (int s2 = 0; s2 < 2; s2++) { const int2 v = qp[b * 4 + s2 * 2 + h]; qa[b][s2] = v2i_t{v.x, v.y}; }
            qs[b] = qd[qi * NBLK + b];
        }
        for (int cc = 0; cc < nc; cc++) {
            float lg[CT / 2];
#pragma unroll
            for (int tile = 0; tile < CT / 16; tile++) {
                float acc[8];
#pragma unroll
                for (int r = 0; r < 8; r++) acc[r] = 0.0f;
#pragma unroll
                for (int b = 0; b < NBLK; b++) {
                    v8i_t z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                    for (int s2 = 0; s2 < 2; s2++) {
                        const int2 kv = *reinterpret_cast<const int2 *>(&sq[cc][(tile * 16 + n) * RS + b * 8 + s2 * 4 + h * 2]);
                        z = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, v2i_t{kv.x, kv.y}, true, qa[b][s2], z, false);
                    }
                    const float4 d0 = *reinterpret_cast<const float4 *>(&sd[cc][b * CT + tile * 16 + h * 8]);
                    const float4 d1 = *reinterpret_cast<const float4 *>(&sd[cc][b * CT + tile * 16 + h * 8 + 4]);
                    const float dk[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
                    for (int r = 0; r < 8; r++) acc[r] += (dk[r] * qs[b]) * (float)z[r];
                }
#pragma unroll
                for (int r = 0; r < 8; r++) lg[tile * 8 + r] = acc[r] * scale;
            }
            float mx = lg[0];
#pragma unroll
            for (int r = 1; r < CT / 2; r++) mx = fmaxf(mx, lg[r]);
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < CT / 2; r++) sum += exp2f(lg[r] - mx);
            const float mo = __shfl_xor(mx, 16), so = __shfl_xor(sum, 16);
            const float M = fmaxf(mx, mo);
            sum = sum * exp2f(mx - M) + so * exp2f(mo - M);
            if (store) {
                float * o = part + (((size_t)i * a.planes + x) * a.C + (c0 + cc)) * 2;
                o[0] = M;
                o[1] = sum;
            }
        }
    }
#endif
}

template <int NBLK, int CT>
static void launch_partials(dim3 grid, int threads, cudaStream_t st, const KPtrs & kp, const PartArgs & a, const int * src, const int8_t * q8,
                            const float * qd, float * part, bool wmma) {
    if (wmma) partials_wmma_kernel<NBLK, CT><<<grid, threads, 0, st>>>(kp, a, src, q8, qd, part);
    else      partials_kernel<NBLK, CT><<<grid, threads, 0, st>>>(kp, a, src, q8, qd, part);
}

static bool dispatch_partials(int nblk, int ct, dim3 grid, int threads, cudaStream_t st, const KPtrs & kp, const PartArgs & a, const int * src,
                              const int8_t * q8, const float * qd, float * part, bool wmma) {
#define KV_CASE(NB, CT) if (nblk == NB && ct == CT) { launch_partials<NB, CT>(grid, threads, st, kp, a, src, q8, qd, part, wmma); return true; }
    KV_CASE(8, 64) KV_CASE(8, 32) KV_CASE(8, 16) KV_CASE(4, 64) KV_CASE(4, 32) KV_CASE(4, 16) KV_CASE(2, 64) KV_CASE(2, 16)
#undef KV_CASE
    return false;
}

// per (query, plane): M = max local max over chunks that hold keys, Z = sum s * exp(m - M)
__global__ void plane_reduce_kernel(const float * __restrict__ part, int C, int planes, float * __restrict__ mz) {
    const size_t row = blockIdx.x;   // i * planes + x
    const float * r = part + row * C * 2;
    __shared__ float red[256];
    float mx = -INFINITY;
    for (int c = threadIdx.x; c < C; c += blockDim.x) if (r[2 * c + 1] > 0.0f) mx = fmaxf(mx, r[2 * c]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    const float M = red[0];
    __syncthreads();
    float z = 0.0f;
    if (M != -INFINITY) for (int c = threadIdx.x; c < C; c += blockDim.x) if (r[2 * c + 1] > 0.0f) z += r[2 * c + 1] * exp2f(r[2 * c] - M);
    red[threadIdx.x] = z;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) { mz[row * 2] = M; mz[row * 2 + 1] = red[0]; }
}

// per (query, chunk): mean, top-kTop mean and max of the plane masses
__global__ void chunk_aggregate_kernel(const float * __restrict__ part, const float * __restrict__ mz, int C, int planes, int nq_tile,
                                       int q0, int nq_total, float * __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (c >= C) return;
    float top[kTop];
#pragma unroll
    for (int k = 0; k < kTop; k++) top[k] = 0.0f;
    float mean = 0.0f, mxp = 0.0f;
    for (int x = 0; x < planes; x++) {
        const size_t row = (size_t)i * planes + x;
        const float M = mz[row * 2], Z = mz[row * 2 + 1];
        if (M == -INFINITY) continue;
        const float s = part[(row * C + c) * 2 + 1];
        const float m = s > 0.0f ? (s * exp2f(part[(row * C + c) * 2] - M)) / Z : 0.0f;
        mean += m / (float)planes;
        mxp = fmaxf(mxp, m);
        int lo = 0;
#pragma unroll
        for (int k = 1; k < kTop; k++) if (top[k] < top[lo]) lo = k;
#pragma unroll
        for (int k = 0; k < kTop; k++) if (k == lo) top[k] = fmaxf(top[k], m);
    }
    float ts = 0.0f;
#pragma unroll
    for (int k = 0; k < kTop; k++) ts += top[k];
    const size_t o = (size_t)(q0 + i) * C + c;
    out[o] = mean;
    out[(size_t)nq_total * C + o] = ts / (float)(planes < kTop ? planes : kTop);
    out[2 * (size_t)nq_total * C + o] = mxp;
}

// the WMMA kernel exists only in the gfx1200/gfx1201 device pass (LUCE_FUSED_WMMA); the host asks the device which one to launch
bool device_has_wmma() {
#if defined(__HIP_PLATFORM_AMD__)
    int dev = 0;
    hipDeviceProp_t pr;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&pr, dev) != hipSuccess) return false;
    return kvflash_fused_arch_has_wmma(pr.gcnArchName);
#else
    return false;
#endif
}

bool ck(cudaError_t e, const char * what, std::string * err) {
    if (e == cudaSuccess) return true;
    if (err) *err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

class KvFlashFusedWs {
public:
    ~KvFlashFusedWs() {
        for (void * p : {(void *)q8, (void *)qd, (void *)src, (void *)part, (void *)mz, (void *)out}) if (p) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
    cudaStream_t stream = nullptr;
    int8_t * q8 = nullptr; float * qd = nullptr; int * src = nullptr; float * part = nullptr; float * mz = nullptr; float * out = nullptr;
    size_t q8_b = 0, qd_b = 0, src_b = 0, part_b = 0, mz_b = 0, out_b = 0;
    template <class T> bool grow(T *& p, size_t & have, size_t need, std::string * err) {
        if (need <= have) return true;
        if (p) cudaFree(p);
        p = nullptr; have = 0;
        if (!ck(cudaMalloc((void **)&p, need), "cudaMalloc", err)) return false;
        have = need;
        return true;
    }
};

KvFlashFusedWs * kvflash_fused_ws_new() { return new KvFlashFusedWs(); }
void kvflash_fused_ws_free(KvFlashFusedWs * w) { delete w; }

bool kvflash_fused_run(KvFlashFusedWs * ws, const KvFlashFusedGeom & g, const KvFlashFusedK & k, const int8_t * q8, const float * qd,
                       size_t max_partial_bytes, float * out_mean, float * out_top, float * out_max, KvFlashFusedTiming * tm,
                       std::string * err) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto fail = [&](const char * m) { if (err) *err = m; return false; };
    const int nblk = g.D / 32, G = g.Hq / std::max(g.HK, 1), planes = g.L * g.Hq;
    if (!ws || g.L <= 0 || g.L > kFusedMaxLayers || g.HK <= 0 || g.Hq % g.HK || g.D % 32 || nblk > 8 || g.ct <= 0 || g.C <= 0 || g.nq <= 0)
        return fail("fused kernel: unsupported geometry");
    if ((int)k.pool.size() != g.L || (int)k.src.size() != g.C) return fail("fused kernel: bad K sources");
    bool any_copy = false;
    for (int s : k.src) any_copy |= s < 0;
    if (any_copy && (int)k.copy.size() != g.L) return fail("fused kernel: device copy missing");
    KPtrs kp{};
    for (int l = 0; l < g.L; l++) { kp.pool[l] = k.pool[(size_t)l]; kp.copy[l] = any_copy ? k.copy[(size_t)l] : nullptr; }
    kp.pool_nb2 = k.pool_nb2; kp.copy_nb2 = k.copy_nb2;

    const bool wmma = g.impl == 2 || (g.impl == 0 && device_has_wmma());
    if (wmma && !device_has_wmma()) return fail("fused kernel: int8 WMMA needs gfx12");
    const size_t per_q = (size_t)planes * g.C * 2 * sizeof(float);
    int tile = (int)std::min<size_t>((size_t)g.nq, std::max<size_t>(max_partial_bytes / per_q, 1));
    if (tile < g.nq && tile > 8) tile -= tile % 8;
    if (!ws->stream && !ck(cudaStreamCreate(&ws->stream), "cudaStreamCreate", err)) return false;
    if (!ws->grow(ws->q8, ws->q8_b, (size_t)g.nq * planes * g.D, err) || !ws->grow(ws->qd, ws->qd_b, (size_t)g.nq * planes * nblk * sizeof(float), err) ||
        !ws->grow(ws->src, ws->src_b, (size_t)g.C * sizeof(int), err) || !ws->grow(ws->part, ws->part_b, (size_t)tile * per_q, err) ||
        !ws->grow(ws->mz, ws->mz_b, (size_t)tile * planes * 2 * sizeof(float), err) || !ws->grow(ws->out, ws->out_b, (size_t)3 * g.nq * g.C * sizeof(float), err))
        return false;
    cudaStream_t st = ws->stream;
    const auto t0 = clk::now();
    if (!ck(cudaMemcpyAsync(ws->q8, q8, (size_t)g.nq * planes * g.D, cudaMemcpyHostToDevice, st), "h2d q8", err) ||
        !ck(cudaMemcpyAsync(ws->qd, qd, (size_t)g.nq * planes * nblk * sizeof(float), cudaMemcpyHostToDevice, st), "h2d qd", err) ||
        !ck(cudaMemcpyAsync(ws->src, k.src.data(), (size_t)g.C * sizeof(int), cudaMemcpyHostToDevice, st), "h2d src", err) ||
        !ck(cudaStreamSynchronize(st), "sync up", err))
        return false;
    const double t_up = ms(t0, clk::now());
    double t_kernel = 0, t_reduce = 0;
    int tiles = 0;
    for (int q0 = 0; q0 < g.nq; q0 += tile, tiles++) {
        const int nqt = std::min(tile, g.nq - q0);
        PartArgs a{g.L, g.HK, g.Hq, G, g.D, nblk, g.ct, g.C, nqt, q0, planes};
        const int P = nqt * G;
        const int threads = wmma ? std::min(12, (P + 15) / 16) * 32 : std::min(256, (P + 31) / 32 * 32);
        const auto ta = clk::now();
        dim3 grid((g.C + kCpb - 1) / kCpb, g.HK, g.L);
        if (!dispatch_partials(nblk, g.ct, grid, threads, st, kp, a, ws->src, ws->q8, ws->qd, ws->part, wmma)) return fail("fused kernel: unsupported (D, chunk tokens)");
        if (!ck(cudaGetLastError(), "partials launch", err) || !ck(cudaStreamSynchronize(st), "partials", err)) return false;
        const auto tb = clk::now();
        plane_reduce_kernel<<<nqt * planes, 256, 0, st>>>(ws->part, g.C, planes, ws->mz);
        dim3 g2((g.C + 127) / 128, nqt);
        chunk_aggregate_kernel<<<g2, 128, 0, st>>>(ws->part, ws->mz, g.C, planes, nqt, q0, g.nq, ws->out);
        if (!ck(cudaGetLastError(), "reduce launch", err) || !ck(cudaStreamSynchronize(st), "reduce", err)) return false;
        t_kernel += ms(ta, tb); t_reduce += ms(tb, clk::now());
    }
    const auto td = clk::now();
    const size_t nb = (size_t)g.nq * g.C * sizeof(float);
    if (!ck(cudaMemcpyAsync(out_mean, ws->out, nb, cudaMemcpyDeviceToHost, st), "d2h", err) ||
        !ck(cudaMemcpyAsync(out_top, ws->out + (size_t)g.nq * g.C, nb, cudaMemcpyDeviceToHost, st), "d2h", err) ||
        !ck(cudaMemcpyAsync(out_max, ws->out + 2 * (size_t)g.nq * g.C, nb, cudaMemcpyDeviceToHost, st), "d2h", err) ||
        !ck(cudaStreamSynchronize(st), "sync down", err))
        return false;
    if (tm) {
        tm->up_ms = t_up; tm->kernel_ms = t_kernel; tm->reduce_ms = t_reduce; tm->down_ms = ms(td, clk::now());
        tm->partial_bytes = (size_t)tile * per_q; tm->tiles = tiles;
    }
    return true;
}

}  // namespace luce::common
