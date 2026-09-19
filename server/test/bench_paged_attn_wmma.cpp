// Paged-attention kernel throughput bench (head-256, gfx1201).
//
// Times ggml_paged_attn_ext at a chunked-prefill shape (nq query rows in one
// block-table slot over a growing paged pool) with the K/V type from argv[2]
// (f16/q8_0/q4_0, default q8_0). DFLASH27B_PAGED_WMMA=0/1 selects the V_DOT2 decode kernel
// or the stage-1 WMMA kernel; run the binary once per value to A/B. The
// launcher reads the env once at static init, hence one route per process.
// Throughput only; correctness is covered by test_paged_attn_wmma.cpp.
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int D  = 256; // head dim
constexpr int Hq = 24;  // query heads
constexpr int Hk = 4;   // KV heads (gqa ratio 6)
constexpr int BLOCK_SIZE = 16;

uint32_t g_rng = 0x9e3779b9u;
float next_float() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((int32_t)(g_rng >> 8) - 8388608) / 8388608.0f; // [-1, 1)
}

bool run_shape(ggml_backend_t gpu, int nq, int kv_len, int iters, ggml_type kv_type) {
    const int pool_tokens = (kv_len + BLOCK_SIZE - 1 + 64) / BLOCK_SIZE * BLOCK_SIZE;
    const int max_blocks  = pool_tokens / BLOCK_SIZE;

    ggml_init_params params = { 4u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  D, nq, Hq);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, kv_type, D, pool_tokens, Hk);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, kv_type, D, pool_tokens, Hk);
    ggml_tensor * block_table      = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, 1);
    ggml_tensor * kv_seq_lens      = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * active_slot_ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
    ggml_tensor * query_positions  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);

    for (ggml_tensor * t : { q, k, v, block_table, kv_seq_lens,
                             active_slot_ids, query_positions }) {
        ggml_set_input(t);
    }

    ggml_tensor * out_t = ggml_paged_attn_ext(
        ctx, q, k, v, block_table, kv_seq_lens, active_slot_ids, query_positions,
        1.0f / std::sqrt((float) D), BLOCK_SIZE, kv_len);
    ggml_set_output(out_t);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_t);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::printf("[bench-paged-wmma] nq=%d kv=%d alloc failed\n", nq, kv_len);
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // ── Data: one slot, identity block table, causal positions. ──
    std::vector<float> qv(ggml_nelements(q));
    for (float & x : qv) x = next_float();
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size() * sizeof(float));

    const size_t n_kv_rows = (size_t) pool_tokens * Hk;
    std::vector<float> kvf(n_kv_rows * D);
    for (float & x : kvf) x = next_float();
    if (kv_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> kvh(kvf.size());
        for (size_t i = 0; i < kvh.size(); ++i) kvh[i] = ggml_fp32_to_fp16(kvf[i]);
        ggml_backend_tensor_set(k, kvh.data(), 0, kvh.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, kvh.data(), 0, kvh.size() * sizeof(ggml_fp16_t));
    } else {
        std::vector<uint8_t> kvq(ggml_nbytes(k));
        ggml_quantize_chunk(kv_type, kvf.data(), kvq.data(), 0,
                            (int64_t) n_kv_rows, D, nullptr);
        ggml_backend_tensor_set(k, kvq.data(), 0, kvq.size());
        ggml_backend_tensor_set(v, kvq.data(), 0, kvq.size());
    }

    std::vector<int32_t> bt((size_t) max_blocks, -1);
    const int live_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int b = 0; b < live_blocks; ++b) bt[b] = b;
    ggml_backend_tensor_set(block_table, bt.data(), 0, bt.size() * sizeof(int32_t));

    std::vector<int32_t> ksl{ kv_len };
    ggml_backend_tensor_set(kv_seq_lens, ksl.data(), 0, sizeof(int32_t));
    std::vector<int32_t> asi((size_t) nq, 0);
    ggml_backend_tensor_set(active_slot_ids, asi.data(), 0, asi.size() * sizeof(int32_t));
    std::vector<int32_t> qpos((size_t) nq);
    for (int r = 0; r < nq; ++r) qpos[r] = kv_len - nq + r; // last nq positions
    ggml_backend_tensor_set(query_positions, qpos.data(), 0, qpos.size() * sizeof(int32_t));

    // ── Warmup + timed runs. ──
    const size_t before = ggml_backend_cuda_get_paged_attn_wmma256_launch_count();
    for (int i = 0; i < 2; ++i) {
        if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return false;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return false;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const size_t launches = ggml_backend_cuda_get_paged_attn_wmma256_launch_count() - before;

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    // Attention work: nq rows over kv tokens, KQ + VKQ, all query heads.
    const double flops = 2.0 * 2.0 * (double) nq * kv_len * D * Hq;
    std::printf("[bench-paged-wmma] nq=%4d kv=%6d  %8.2f ms  %7.1f GFLOP/s  wmma_launches=%zu\n",
                nq, kv_len, ms, flops / (ms * 1e6), launches);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
#if defined(GGML_USE_HIP)
    // Distinguish "no device" (skip) from a device that exists but cannot be
    // queried (real failure), so CI does not report a runtime error as a skip.
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[bench-paged-wmma] SKIP: no HIP device\n");
        return 77;
    }
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
        std::printf("[bench-paged-wmma] FAIL: hipGetDeviceProperties failed\n");
        return 1;
    }
    if (std::strncmp(props.gcnArchName, "gfx12", 5) != 0) {
        std::printf("[bench-paged-wmma] SKIP: requires gfx12 (RDNA4), got %s\n", props.gcnArchName);
        return 77;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) return 1;

    const char * env = getenv("DFLASH27B_PAGED_WMMA");
    const int route = env && atoi(env) != 0;
    std::printf("[bench-paged-wmma] route=%s\n", route ? "wmma" : "v_dot2");

    const int nq = argc > 1 ? atoi(argv[1]) : 128;
    const char * tname = argc > 2 ? argv[2] : "q8_0";
    if (std::strcmp(tname, "f16") != 0 && std::strcmp(tname, "q8_0") != 0 &&
        std::strcmp(tname, "q4_0") != 0) {
        std::printf("usage: bench_paged_attn_wmma [nq] [f16|q8_0|q4_0]\n");
        ggml_backend_free(gpu);
        return 2;
    }
    const ggml_type kv_type = std::strcmp(tname, "f16") == 0   ? GGML_TYPE_F16  :
                              std::strcmp(tname, "q4_0") == 0  ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
    std::printf("[bench-paged-wmma] kv_type=%s\n", ggml_type_name(kv_type));
    bool ok = true;
    for (int kv : { 8192, 16384, 32768, 65536 }) {
        ok = run_shape(gpu, nq, kv, 5, kv_type) && ok;
    }
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
#else
    std::printf("[bench-paged-wmma] SKIP: HIP-only\n");
    return 77;
#endif
}
