// Parity test for the QSA3 prefill union builder (qsa.cu): the bitmap kernel (qsa3_union_bitmap_kernel,
// default), the compacting rows kernel + merge (LUCE_ATTN_PREFILL_LEGACY=1) and the legacy rank-sort rows kernel
// + merge (variant 0) must all give the same per-group union (ublk/umask/ucount) as a CPU set union.
//
// Rows are generated the way qsa_decode_ids writes them (sorted blocks x 4 cells, -1 holes for future
// blocks in the middle, then the tail cells), plus shuffled rows, out-of-range keys, extra holes and fully
// random rows, so both the compaction fast path and the bitonic fallback are exercised. Keys within a row
// are distinct (the merge kernel's contract; qsa_decode_ids never repeats a cell). gfx1151 only
// (the kernels are compiled for that target); SKIP (77) elsewhere.

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

extern "C" int  ggml_cuda_qsa3_union_cap(int ns);
extern "C" void ggml_cuda_qsa3_union_launch(const int * ids, size_t i1, int n_q, int ns, int nk, int * srow, int * sflag,
        uint16_t * ublk, uint16_t * umask, int * ucount, int phases, int rows_variant, hipStream_t stream);

namespace {

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); return false; } } while (0)

// One id row for query position `pos`: `budget` sorted blocks (the visible ones first, future ones -1),
// then ratio-1 tail cells. kind 1 shuffles the row, kind 2 adds out-of-range keys and duplicates.
std::vector<int32_t> make_row(std::mt19937 & rng, int pos, int ns, int nk, int kind) {
    const int ratio = 4, budget = (ns - (ratio - 1)) / ratio, nb = nk / ratio, nvis = (pos + 1) / ratio;
    std::vector<int> blocks;
    if (nvis <= budget) {
        for (int b = 0; b < nvis; ++b) blocks.push_back(b);
        for (int b = nvis; (int) blocks.size() < budget; ++b) blocks.push_back(b < nb ? b : nb + b);
    } else {
        std::vector<int> all(nvis);
        for (int b = 0; b < nvis; ++b) all[b] = b;
        std::shuffle(all.begin(), all.end(), rng);
        blocks.assign(all.begin(), all.begin() + budget);
        std::sort(blocks.begin(), blocks.end());
    }
    std::vector<int32_t> row(ns, -1);
    for (int j = 0; j < budget * ratio; ++j) {
        const int first = ratio * blocks[j / ratio];
        row[j] = (first + ratio - 1 <= pos) ? first + j % ratio : -1;
    }
    const int br = ((pos + 1) / ratio) * ratio;
    for (int j = 0; j < ratio - 1; ++j) row[budget * ratio + j] = (br + j <= pos) ? br + j : -1;
    if (kind == 1) std::shuffle(row.begin(), row.end(), rng);
    if (kind == 2) {   // out-of-range keys and extra holes at random positions (keys stay distinct: the contract)
        std::uniform_int_distribution<int> jd(0, ns - 1), kd(nk, nk + 64);
        for (int i = 0; i < 16; ++i) row[jd(rng)] = kd(rng);
        for (int i = 0; i < 16; ++i) row[jd(rng)] = -1;
    }
    if (kind == 3) {   // random distinct keys in random order: the bitonic fallback on a full row
        std::vector<int> all(nk);
        for (int k = 0; k < nk; ++k) all[k] = k;
        std::shuffle(all.begin(), all.end(), rng);
        for (int j = 0; j < ns; ++j) row[j] = j < nk ? all[j] : -1;
    }
    return row;
}

bool run_case(int n_q, int ns, int nk, int pos0, uint32_t seed, hipStream_t stream) {
    std::mt19937 rng(seed);
    std::vector<int32_t> ids((size_t) n_q * ns);
    std::uniform_int_distribution<int> kind(0, 9);
    for (int q = 0; q < n_q; ++q) {
        const int k = kind(rng);
        const std::vector<int32_t> row = make_row(rng, pos0 + q, ns, nk, k < 6 ? 0 : (k < 8 ? 1 : (k < 9 ? 2 : 3)));
        std::memcpy(ids.data() + (size_t) q * ns, row.data(), (size_t) ns * 4);
    }
    const int ngroups = (n_q + 3) / 4, cap = ggml_cuda_qsa3_union_cap(ns);
    // CPU reference: per group, sorted set of blocks with the 4-bit-per-query cell masks, padded to x4.
    std::vector<uint16_t> rblk((size_t) ngroups * cap, 0xEEEE), rmsk(rblk.size(), 0xEEEE);
    std::vector<int> rcnt(ngroups);
    for (int g = 0; g < ngroups; ++g) {
        std::map<int, uint16_t> u;
        for (int qi = 0; qi < 4 && 4*g + qi < n_q; ++qi) {
            const int32_t * row = ids.data() + (size_t) (4*g + qi) * ns;
            for (int j = 0; j < ns; ++j) if (row[j] >= 0 && row[j] < nk) u[row[j] >> 2] |= (uint16_t) (1u << (4*qi + (row[j] & 3)));
        }
        int t = 0;
        for (const auto & e : u) { rblk[(size_t) g * cap + t] = (uint16_t) e.first; rmsk[(size_t) g * cap + t] = e.second; ++t; }
        while (t & 3) { rblk[(size_t) g * cap + t] = 0xFFFFu; rmsk[(size_t) g * cap + t] = 0; ++t; }
        rcnt[g] = t;
    }
    int32_t * d_ids; int * d_srow; int * d_sflag; int * d_ucount; uint16_t * d_ublk; uint16_t * d_umask;
    HIP_OK(hipMalloc(&d_ids, ids.size() * 4));
    HIP_OK(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&d_srow, (size_t) n_q * ns * 4));
    HIP_OK(hipMalloc(&d_sflag, (size_t) n_q * 4));
    HIP_OK(hipMalloc(&d_ucount, (size_t) ngroups * 4));
    HIP_OK(hipMalloc(&d_ublk, (size_t) ngroups * cap * 2));
    HIP_OK(hipMalloc(&d_umask, (size_t) ngroups * cap * 2));
    bool ok = true;
    for (int variant = 0; variant < 3; ++variant) {
        HIP_OK(hipMemset(d_ublk, 0xEE, (size_t) ngroups * cap * 2));
        HIP_OK(hipMemset(d_umask, 0xEE, (size_t) ngroups * cap * 2));
        HIP_OK(hipMemset(d_ucount, 0xEE, (size_t) ngroups * 4));
        ggml_cuda_qsa3_union_launch(d_ids, (size_t) ns * 4, n_q, ns, nk, d_srow, d_sflag, d_ublk, d_umask, d_ucount, 3, variant, stream);
        HIP_OK(hipStreamSynchronize(stream));
        std::vector<uint16_t> blk(rblk.size()), msk(rblk.size());
        std::vector<int> cnt(ngroups);
        HIP_OK(hipMemcpy(blk.data(), d_ublk, blk.size() * 2, hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(msk.data(), d_umask, msk.size() * 2, hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(cnt.data(), d_ucount, (size_t) ngroups * 4, hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (int g = 0; g < ngroups; ++g) {
            if (cnt[g] != rcnt[g]) { ++bad; continue; }
            for (int j = 0; j < rcnt[g]; ++j) {
                if (blk[(size_t) g * cap + j] != rblk[(size_t) g * cap + j] || msk[(size_t) g * cap + j] != rmsk[(size_t) g * cap + j]) { ++bad; break; }
            }
        }
        if (bad) std::fprintf(stderr, "  FAIL: rows variant %d: %zu / %d groups differ from the CPU union\n", variant, bad, ngroups);
        ok = ok && bad == 0;
    }
    hipFree(d_ids); hipFree(d_srow); hipFree(d_sflag); hipFree(d_ucount); hipFree(d_ublk); hipFree(d_umask);
    return ok;
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[qsa3-union-parity] SKIP: no HIP device\n");
        return 77;
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess || std::string(prop.gcnArchName).rfind("gfx1151", 0) != 0) {
        std::printf("[qsa3-union-parity] SKIP: kernels are built for gfx1151 only (device is %s)\n", prop.gcnArchName);
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::printf("[qsa3-union-parity] FAIL: backend init\n");
        return 1;
    }
    hipStream_t stream;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    struct Shape { int n_q, ns, nk, pos0; };
    bool ok = true;
    for (const Shape & s : { Shape{301, 2051, 4136, 0}, Shape{130, 515, 2048, 0}, Shape{64, 2051, 20480, 15616}, Shape{37, 1027, 1028, 0} }) {
        for (uint32_t seed = 1; seed <= 3; ++seed) {
            const bool c = run_case(s.n_q, s.ns, s.nk, s.pos0, seed, stream);
            std::printf("[qsa3-union-parity] n_q=%d ns=%d nk=%d pos0=%d seed=%u %s\n", s.n_q, s.ns, s.nk, s.pos0, seed, c ? "ok" : "FAIL");
            ok = c && ok;
        }
    }
    hipStreamDestroy(stream);
    ggml_backend_free(backend);
    std::printf("[qsa3-union-parity] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[qsa3-union-parity] SKIP: HIP-only\n");
    return 77;
#endif
}
