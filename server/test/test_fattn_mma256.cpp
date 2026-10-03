// DKQ=256 MMA fattn differential qualification (RDNA3.5/gfx1151 and RDNA4/gfx1201).
//
// Qwen3.5/3.6/3.8 dense-hybrid targets attend at head_dim=256. The RDNA4
// dispatch historically capped the MMA fattn path at head 128, so prefill
// fell through to the generic tile kernel. LUCE_FA256_MMA=1 routes
// head-256 through a tensor-core kernel (rocWMMA when the build has it,
// raw MMA otherwise); this test pins that route via the launch counters and
// checks the output against ggml's own CPU flash_attn_ext reference over
// the same host-side tensors.
//
// Differential correctness only. No throughput measurements.
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Head dim; LUCE_FA256_TEST_D=128 runs a diagnostic baseline for
// the rocWMMA kernel's already-dispatchable heads (route check will fail).
static int head_dim() {
    const char * e = getenv("LUCE_FA256_TEST_D");
    return e ? atoi(e) : 256;
}
constexpr int Hq = 24;  // query heads
int Hk = 4;   // KV heads (gqa ratio 6)

uint32_t g_rng = 0x9e3779b9u;
float next_float() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((int32_t)(g_rng >> 8) - 8388608) / 8388608.0f; // [-1, 1)
}

// Dequantize Q8_0 rows (n_per_row = D) to host F16 values. Deterministic;
// matches the block-wise fp16 scale the GPU kernel consumes.
std::vector<ggml_fp16_t> dequant_q8_0(const std::vector<uint8_t> & bytes, size_t n_rows, int D) {
    const auto * blocks = reinterpret_cast<const block_q8_0 *>(bytes.data());
    const int n_blocks_per_row = D / QK8_0;
    std::vector<ggml_fp16_t> out(n_rows * D);
    for (size_t r = 0; r < n_rows; ++r) {
        for (int b = 0; b < n_blocks_per_row; ++b) {
            const block_q8_0 & blk = blocks[r * n_blocks_per_row + b];
            const float d = ggml_fp16_to_fp32(blk.d);
            for (int i = 0; i < QK8_0; ++i) {
                out[r * D + b * QK8_0 + i] = ggml_fp32_to_fp16(d * blk.qs[i]);
            }
        }
    }
    return out;
}

struct Case {
    int  S;    // KV length (multiple of FATTN_KQ_STRIDE for the tensor-core route)
    int  nq;   // query rows (prefill chunk)
    bool q8_0; // quantized K/V
    bool expect_tc; // expect the tensor-core route (vs a tile/vec fallback)
    bool sinks;     // exercise the attention-sink path
    float softcap;  // 0 = off
    float tolerance;
};

bool run_case(ggml_backend_t gpu, ggml_backend_t cpu, const Case & c, int D) {
    const ggml_type kv_type = c.q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    const float scale = 1.0f / std::sqrt((float) D);

    ggml_init_params params = { 4u << 20, nullptr, true };
    ggml_context * gctx = ggml_init(params);
    ggml_context * cctx = ggml_init(params);
    if (!gctx || !cctx) {
        if (gctx) ggml_free(gctx);
        if (cctx) ggml_free(cctx);
        return false;
    }

    // ── GPU graph: single flash_attn_ext at head 256 ──
    ggml_tensor * Q    = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, D, c.nq, Hq);
    ggml_tensor * K    = ggml_new_tensor_3d(gctx, kv_type, D, c.S, Hk);
    ggml_tensor * V    = ggml_new_tensor_3d(gctx, kv_type, D, c.S, Hk);
    ggml_tensor * mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, c.S, c.nq);
    for (ggml_tensor * t : { Q, K, V, mask }) {
        ggml_set_input(t);
    }
    ggml_tensor * out = ggml_flash_attn_ext(gctx, Q, K, V, mask, scale, 0.0f, c.softcap);
    if (getenv("LUCE_FA256_TEST_F32ACC") != nullptr) {
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    }
    if (c.sinks) {
        ggml_tensor * sinks = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, Hq);
        ggml_set_input(sinks);
        ggml_flash_attn_ext_add_sinks(out, sinks);
        ggml_set_output(sinks);
    }
    ggml_set_output(out);
    ggml_cgraph * ggraph = ggml_new_graph(gctx);
    ggml_build_forward_expand(ggraph, out);

    // ── CPU reference: KQ matmul + masked softmax + V matmul ──
    // K/V always F16 on the CPU: for the Q8_0 case they hold the
    // host-dequantized values the GPU kernel attends over.
    ggml_tensor * Qr    = ggml_new_tensor_3d(cctx, GGML_TYPE_F32, D, c.nq, Hq);
    ggml_tensor * Kr    = ggml_new_tensor_3d(cctx, GGML_TYPE_F16, D, c.S, Hk);
    ggml_tensor * Vr    = ggml_new_tensor_3d(cctx, GGML_TYPE_F16, D, c.S, Hk);

    // Reference: ggml's own CPU flash_attn_ext over the same host-side
    // tensors and the same 2D F16 causal mask as the GPU op.
    ggml_tensor * maskf = ggml_new_tensor_2d(cctx, GGML_TYPE_F16, c.S, c.nq);
    ggml_tensor * ref = ggml_flash_attn_ext(cctx, Qr, Kr, Vr, maskf, scale, 0.0f, c.softcap);
    if (c.sinks) {
        ggml_tensor * sinksr = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, Hq);
        ggml_set_input(sinksr);
        ggml_flash_attn_ext_add_sinks(ref, sinksr);
        ggml_set_output(sinksr);
    }
    ggml_set_output(ref);

    ggml_cgraph * cgraph = ggml_new_graph(cctx);
    ggml_build_forward_expand(cgraph, ref);

    ggml_gallocr_t galloc_g = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    ggml_gallocr_t galloc_c = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    const auto fail_cleanup = [&]() {
        if (galloc_g) ggml_gallocr_free(galloc_g);
        if (galloc_c) ggml_gallocr_free(galloc_c);
        ggml_free(gctx);
        ggml_free(cctx);
        return false;
    };
    if (!galloc_g || !galloc_c ||
        !ggml_gallocr_alloc_graph(galloc_g, ggraph) ||
        !ggml_gallocr_alloc_graph(galloc_c, cgraph)) {
        return fail_cleanup();
    }

    // ── Shared input data ──
    std::vector<float> qv(ggml_nelements(Q));
    for (float & x : qv) x = next_float();
    ggml_backend_tensor_set(Q, qv.data(), 0, ggml_nbytes(Q));
    ggml_backend_tensor_set(Qr, qv.data(), 0, ggml_nbytes(Qr));

    const size_t n_kv_rows = (size_t) c.S * Hk;
    std::vector<float> kv_f32(n_kv_rows * D);
    for (float & x : kv_f32) x = next_float();

    std::vector<ggml_fp16_t> kv_f16(n_kv_rows * D);
    for (size_t i = 0; i < kv_f16.size(); ++i) kv_f16[i] = ggml_fp32_to_fp16(kv_f32[i]);

    if (c.q8_0) {
        std::vector<uint8_t> kv_q8(ggml_nbytes(K));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, kv_f32.data(), kv_q8.data(), 0,
                            (int64_t) n_kv_rows, D, nullptr);
        ggml_backend_tensor_set(K, kv_q8.data(), 0, kv_q8.size());
        ggml_backend_tensor_set(V, kv_q8.data(), 0, kv_q8.size());
        const auto deq = dequant_q8_0(kv_q8, n_kv_rows, D);
        ggml_backend_tensor_set(Kr, deq.data(), 0, deq.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(Vr, deq.data(), 0, deq.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(K, kv_f16.data(), 0, kv_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(V, kv_f16.data(), 0, kv_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(Kr, kv_f16.data(), 0, kv_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(Vr, kv_f16.data(), 0, kv_f16.size() * sizeof(ggml_fp16_t));
    }

    // Causal mask: mask[q * S + kv] = kv <= q ? 0 : -inf
    std::vector<ggml_fp16_t> mv((size_t) c.S * c.nq);
    for (int q = 0; q < c.nq; ++q) {
        for (int kv = 0; kv < c.S; ++kv) {
            const float val = kv <= q ? 0.0f : -INFINITY;
            mv[(size_t) q * c.S + kv] = ggml_fp32_to_fp16(val);
        }
    }
    ggml_backend_tensor_set(mask, mv.data(), 0, mv.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(maskf, mv.data(), 0, mv.size() * sizeof(ggml_fp16_t));

    if (c.sinks) {
        std::vector<float> sinks_data(Hq);
        for (float & v : sinks_data) v = -2.0f + 4.0f * next_float();
        ggml_backend_tensor_set(out->src[4], sinks_data.data(), 0, sinks_data.size() * sizeof(float));
        ggml_backend_tensor_set(ref->src[4], sinks_data.data(), 0, sinks_data.size() * sizeof(float));
    }

    // ── Run ──
    const size_t mma256_before  = ggml_backend_cuda_get_fattn_mma256_launch_count();
    const size_t wmma256_before = ggml_backend_cuda_get_fattn_wmma256_launch_count();
    if (ggml_backend_graph_compute(gpu, ggraph) != GGML_STATUS_SUCCESS) return fail_cleanup();
    if (ggml_backend_graph_compute(cpu, cgraph) != GGML_STATUS_SUCCESS) return fail_cleanup();
    const size_t tc256_launches =
        (ggml_backend_cuda_get_fattn_mma256_launch_count()  - mma256_before) +
        (ggml_backend_cuda_get_fattn_wmma256_launch_count() - wmma256_before);

    std::vector<float> out_data(ggml_nelements(out));
    std::vector<float> ref_data(ggml_nelements(ref));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));
    ggml_backend_tensor_get(ref, ref_data.data(), 0, ggml_nbytes(ref));

    float max_diff = 0.0f;
    size_t worst_i = 0;
    bool nonfinite = false;
    for (size_t i = 0; i < out_data.size(); ++i) {
        if (!std::isfinite(out_data[i]) || !std::isfinite(ref_data[i])) {
            nonfinite = true;
            break;
        }
        const float diff = std::fabs(out_data[i] - ref_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
            worst_i = i;
        }
    }

    // Debug dump: profile the worst column against the reference to tell a
    // rowsum/scaling error (constant ratio across d) from a layout scramble.
    if (getenv("LUCE_FA256_DUMP") != nullptr && max_diff >= c.tolerance) {
        const size_t wq = (worst_i / D) % (size_t) c.nq;
        const size_t wh = worst_i / (D * (size_t) c.nq);
        std::printf("[dump] S=%d q=%zu h=%zu\n", c.S, wq, wh);
        for (int d = 0; d < D; d += 16) {
            std::printf("[dump] d=%3d out=%9.5f ref=%9.5f ratio=%8.4f\n",
                        d, out_data[wh * c.nq * D + wq * D + d],
                        ref_data[wh * c.nq * D + wq * D + d],
                        ref_data[wh * c.nq * D + wq * D + d] != 0.0f
                            ? out_data[wh * c.nq * D + wq * D + d] / ref_data[wh * c.nq * D + wq * D + d]
                            : 999.0f);
        }
    }

    const bool route_ok = c.expect_tc ? tc256_launches >= 1 : tc256_launches == 0;
    const bool pass = route_ok && !nonfinite && max_diff < c.tolerance;
    std::printf("[fattn-mma256] S=%d nq=%d kv=%s%s%s tc256_launches=%zu max_diff=%0.6f worst=(d=%zu,q=%zu,h=%zu) nonfinite=%s %s\n",
                c.S, c.nq, c.q8_0 ? "q8_0" : "f16", c.sinks ? " sinks" : "",
                c.softcap != 0.0f ? " softcap" : "", tc256_launches, max_diff,
                worst_i % D, (worst_i / D) % (size_t) c.nq, worst_i / (D * (size_t) c.nq),
                nonfinite ? "YES" : "no", pass ? "PASS" : "FAIL");

    ggml_gallocr_free(galloc_g);
    ggml_gallocr_free(galloc_c);
    ggml_free(gctx);
    ggml_free(cctx);
    return pass;
}

} // namespace

int main() {
#if defined(GGML_USE_HIP)
    hipDeviceProp_t props{};
    const bool have_device = hipGetDeviceProperties(&props, 0) == hipSuccess;
    const bool rdna35 = have_device && std::strncmp(props.gcnArchName, "gfx115", 6) == 0;
    if (!have_device || (!rdna35 && std::strncmp(props.gcnArchName, "gfx12", 5) != 0)) {
        std::printf("[fattn-mma256] SKIP: requires RDNA3.5 or RDNA4\n");
        return 77;
    }
    // Must precede the first GPU fattn dispatch: fattn.cu reads the opt-in
    // once per process on the first dispatch call. An explicit value in the
    // environment wins. LUCE_FA256_MMA=0 opts the tensor-core route
    // out; the test then verifies the tile/vec fallback instead (correct
    // results, zero tensor-core launches) rather than skipping.
    const bool kill_switch = [] {
        const char * e = getenv("LUCE_FA256_MMA");
        return e && atoi(e) == 0 && getenv("LUCE_FA256_WMMA") == nullptr;
    }();
    if (getenv("LUCE_FA256_MMA") == nullptr &&
        setenv("LUCE_FA256_MMA", "1", 1) != 0) {
        std::fprintf(stderr, "[fattn-mma256] setenv failed\n");
        return 1;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!gpu || !cpu) {
        std::fprintf(stderr, "[fattn-mma256] backend init failed\n");
        return 1;
    }

    // The nq=16 case only routes through a tensor-core kernel when the
    // forced-WMMA A/B mode is on; otherwise it dispatches to the vector
    // kernel and would fail the route pin.
    const bool wmma_force = getenv("LUCE_FA256_WMMA") != nullptr;
    // Ragged KV length (not a multiple of FATTN_KQ_STRIDE): gqa_opt does
    // not apply, so the tensor-core route must NOT fire and the generic
    // fallback must still produce correct results.
    const bool expect_tc = rdna35 || !kill_switch;
    if (rdna35) Hk = 2; // Qwen4Exp GQA=12: four heads per MMA tile.
    // The tile kernel's own accumulation sits ~1.2e-3 from the CPU
    // reference (pre-existing numerics); the kill-switch mode verifies the
    // fallback route and its correctness at that looser bound.
    const float tol_mul = kill_switch ? 2.0f : 1.0f;
    std::vector<Case> cases = {
        {   512,  512, false, expect_tc, false, 0.0f,  tol_mul*1e-3f },
        {  1024,   64, false, expect_tc, false, 0.0f,  tol_mul*1e-3f },
        {  4096,   64, false, expect_tc, false, 0.0f,  tol_mul*1e-3f },
        {  8192,   64, true,  expect_tc, false, 0.0f,  tol_mul*2e-3f },
        { 16384,   64, true,  expect_tc, false, 0.0f,  tol_mul*2e-3f },
        { 65536,   64, true,  expect_tc, false, 0.0f,  tol_mul*2e-3f },
        {  3000,   64, false, false,      false, 0.0f,  tol_mul*2e-3f },
        {  4096,   64, false, expect_tc, true,  0.0f,  tol_mul*1e-3f },
        {  4096,   64, false, expect_tc, false, 50.0f, tol_mul*1e-3f },
    };
    if (rdna35) {
        // Exercise upstream ncols=64 and the short-query TILE fallback.
        cases.insert(cases.begin(), {
            { 256, 16, false, true,  false, 0.0f, 1e-3f },
            { 256,  9, false, true,  false, 0.0f, 1e-3f },
            { 256,  8, false, false, false, 0.0f, 2e-3f },
            { 256,  1, false, false, false, 0.0f, 2e-3f },
        });
    }
    if (wmma_force) {
        cases.insert(cases.begin() + 1, { 512, 16, false, expect_tc, false, 0.0f, tol_mul*1e-3f });
    }
    bool ok = true;
    for (const Case & c : cases) {
        ok = run_case(gpu, cpu, c, head_dim()) && ok;
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
#else
    std::printf("[fattn-mma256] SKIP: HIP-only kernel path\n");
    return 77;
#endif
}
