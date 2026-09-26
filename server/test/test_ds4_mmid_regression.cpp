// Regression test for the route-bounded mmid merge.
//
// Negative expert ids are DS4 "masked owner routes": they belong to no expert
// and must not reserve a slot in the compact routing arrays. This drives
// ggml_cuda_launch_mm_ids_helper directly (the function the mmq/mmf mul_mat_id
// paths call) and checks the compact contract for the <=256-expert fast path
// and the classic path, in both forward and inverse (write_inverse) modes.
//
// mmid.cuh declares the helper with cudaStream_t; the HIP build maps that to
// hipStream_t (ggml/src/ggml-cuda/vendors/hip.h:152), so it is declared with the
// HIP spelling here instead of including the header.

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1,
        bool write_inverse, hipStream_t stream);

namespace {

bool check(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", what);
    }
    return cond;
}

bool run_case(int n_experts, int n_tokens, int n_expert_used, bool write_inverse) {
    const int n_routes = n_tokens * n_expert_used;
    const int si1  = n_expert_used;
    const int sis1 = n_expert_used;
    const int nchannels_y = n_expert_used;

    // Deterministic routing with masked (negative) routes sprinkled in.
    std::vector<int32_t> h_ids((size_t) n_routes);
    std::vector<int> h_count((size_t) n_experts + 1, 0);
    int n_nonneg = 0;
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_expert_used; ++u) {
            int32_t v = (int32_t) ((t * 3 + u * 7 + u * u) % n_experts);
            if (((t + u) % 5) == 0 || ((t * 2 + u) % 11) == 0) {
                v = -1;
            }
            h_ids[(size_t) (t * n_expert_used + u)] = v;
            if (v >= 0) {
                h_count[(size_t) v]++;
                n_nonneg++;
            }
        }
    }

    int32_t * d_ids = nullptr, * d_src1 = nullptr, * d_dst = nullptr, * d_bounds = nullptr;
    bool ok = true;
    ok = check(hipMalloc(&d_ids, sizeof(int32_t) * n_routes) == hipSuccess, "hipMalloc ids") && ok;
    ok = check(hipMalloc(&d_src1, sizeof(int32_t) * n_routes) == hipSuccess, "hipMalloc src1") && ok;
    ok = check(hipMalloc(&d_dst, sizeof(int32_t) * n_routes) == hipSuccess, "hipMalloc dst") && ok;
    ok = check(hipMalloc(&d_bounds, sizeof(int32_t) * (n_experts + 1)) == hipSuccess, "hipMalloc bounds") && ok;
    if (!ok) {
        hipFree(d_ids); hipFree(d_src1); hipFree(d_dst); hipFree(d_bounds);
        return false;
    }

    hipMemcpy(d_ids, h_ids.data(), sizeof(int32_t) * n_routes, hipMemcpyHostToDevice);
    hipMemset(d_src1, 0xFF, sizeof(int32_t) * n_routes);  // -1 = unset
    hipMemset(d_dst,  0xFF, sizeof(int32_t) * n_routes);
    hipMemset(d_bounds, 0xFF, sizeof(int32_t) * (n_experts + 1));

    ggml_cuda_launch_mm_ids_helper(d_ids, d_src1, d_dst, d_bounds,
        n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1,
        write_inverse, (hipStream_t) 0);
    hipDeviceSynchronize();

    std::vector<int32_t> h_src1((size_t) n_routes), h_dst((size_t) n_routes);
    std::vector<int32_t> h_bounds((size_t) n_experts + 1);
    hipMemcpy(h_src1.data(), d_src1, sizeof(int32_t) * n_routes, hipMemcpyDeviceToHost);
    hipMemcpy(h_dst.data(),  d_dst,  sizeof(int32_t) * n_routes, hipMemcpyDeviceToHost);
    hipMemcpy(h_bounds.data(), d_bounds, sizeof(int32_t) * (n_experts + 1), hipMemcpyDeviceToHost);

    // 1. The bounds partition the compact array exactly, counting only
    //    non-negative routes: a masked route must not create a hole.
    ok = check(h_bounds[0] == 0, "bounds[0] == 0") && ok;
    for (int e = 0; e < n_experts; ++e) {
        ok = check(h_bounds[e + 1] >= h_bounds[e], "bounds monotone") && ok;
        ok = check(h_bounds[e + 1] - h_bounds[e] == h_count[(size_t) e], "per-expert route count") && ok;
    }
    ok = check(h_bounds[n_experts] == (int32_t) n_nonneg, "total == non-negative routes") && ok;
    if (!ok) {
        hipFree(d_ids); hipFree(d_src1); hipFree(d_dst); hipFree(d_bounds);
        return false;
    }

    const int n_compact = h_bounds[n_experts];

    // 2. Every compact slot maps back to a route whose id is exactly its owning
    //    expert, and every non-negative route appears exactly once.
    std::vector<int32_t> slot_expert((size_t) n_compact, -1);
    for (int e = 0; e < n_experts; ++e) {
        for (int i = h_bounds[e]; i < h_bounds[e + 1]; ++i) {
            slot_expert[(size_t) i] = e;
        }
    }

    std::vector<int> seen((size_t) n_routes, 0);
    for (int i = 0; i < n_compact; ++i) {
        const int32_t dstv = h_dst[(size_t) i];
        if (dstv < 0 || dstv >= n_routes) { ok = check(false, "dst in range") && ok; continue; }
        const int t = dstv / n_expert_used;
        const int u = dstv % n_expert_used;
        ok = check(h_ids[(size_t) dstv] == slot_expert[(size_t) i], "dst route matches owning expert") && ok;
        ok = check(!seen[(size_t) dstv], "route appears once") && ok;
        seen[(size_t) dstv] = 1;
        if (write_inverse) {
            ok = check(h_src1[(size_t) dstv] == i, "inverse row == compact index") && ok;
        } else {
            ok = check(h_src1[(size_t) i] == t * sis1 + (u % nchannels_y), "forward row") && ok;
        }
    }
    for (int r = 0; r < n_routes; ++r) {
        ok = check(seen[(size_t) r] == (h_ids[(size_t) r] >= 0 ? 1 : 0),
                   h_ids[(size_t) r] >= 0 ? "non-negative route present" : "masked route absent") && ok;
    }

    hipFree(d_ids); hipFree(d_src1); hipFree(d_dst); hipFree(d_bounds);
    return ok;
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[mmid-regress] SKIP: no HIP device\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::printf("[mmid-regress] FAIL: backend init\n");
        return 1;
    }

    bool ok = true;
    for (int n_experts : { 64, 320 }) {
        for (int inverse = 0; inverse < 2; ++inverse) {
            const bool c = run_case(n_experts, /*n_tokens=*/64, /*n_expert_used=*/10, inverse != 0);
            std::printf("[mmid-regress] experts=%d inverse=%d %s\n",
                n_experts, inverse, c ? "ok" : "FAIL");
            ok = c && ok;
        }
    }
    ggml_backend_free(backend);
    std::printf("[mmid-regress] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[mmid-regress] SKIP: HIP-only\n");
    return 77;
#endif
}
