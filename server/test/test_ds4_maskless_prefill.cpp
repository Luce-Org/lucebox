#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <cstring>
#include <vector>

// Differential correctness only. This emits no throughput measurements.
static bool check(ggml_backend_t backend, int start, int tokens,
                  int expected_launches, bool require_byte_identity,
                  bool f32_kv = false, int heads = 16,
                  bool stress_weight_bounds = false, int window = 128,
                  bool require_compact_identity = true) {
    constexpr int dim = 512, selected = 512;
    const int prior = std::min(start, window);
    const int raw = prior + tokens;
    const int compressed = (start + tokens) / 4;
    const int rows = raw + compressed;
    ggml_init_params params{4u << 20, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, tokens, heads);
    auto * kv = ggml_new_tensor_3d(ctx, f32_kv ? GGML_TYPE_F32 : GGML_TYPE_F16, dim, rows, 1);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, tokens);
    auto * topk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected, tokens);
    auto * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    // Repeated submissions reuse these buffers like resident production
    // inputs. INPUT alone does not stop gallocr recycling a leaf after its
    // final use within the multi-output graph.
    for (auto * input : {q, kv, mask, topk, sinks}) {
        ggml_set_input(input);
        ggml_set_output(input);
    }
    auto make_attention = [&](bool direct, bool maskless) {
        auto * out = ggml_flash_attn_ext(ctx, q, kv, kv, maskless ? nullptr : mask,
                                       1.0f / std::sqrt(float(dim)), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(out, sinks);
        ggml_flash_attn_ext_set_ds4_sparse(out, raw, window, -selected, 32);
        if (direct) ggml_flash_attn_ext_set_ds4_indexer_topk(out, topk);
        ggml_flash_attn_ext_set_ds4_inverse_rope(
            out, start, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 163840, true);
        ggml_set_output(out);
        return out;
    };
    auto * ref = make_attention(false, false);
    auto * explicit_stream = make_attention(true, false);
    auto * analytic_stream = make_attention(true, true);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    bool ok = true;
    // The analytical contract is only for fixed-position ratio-4 prefill.
    // Masked verification supports wider row sets and runtime positions, but
    // those shapes must never reach the fixed-width maskless sorter.
    auto * too_wide = make_attention(false, true);
    ggml_flash_attn_ext_set_ds4_sparse(too_wide, raw, window, -(selected + 1), 32);
    ggml_flash_attn_ext_set_ds4_indexer_topk(
        too_wide, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected + 1, tokens));
    auto * dynamic_positions = make_attention(true, true);
    ggml_flash_attn_ext_set_ds4_rope_positions(
        dynamic_positions, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens));
    for (auto * invalid : {too_wide, dynamic_positions}) {
        if (ggml_backend_supports_op(backend, invalid)) {
            std::fputs("unsupported maskless contract was admitted\n", stderr);
            ok = false;
        }
    }
    for (auto * out : {ref, explicit_stream, analytic_stream}) {
        ok = ggml_backend_supports_op(backend, out) && ok;
        ggml_build_forward_expand(graph, out);
    }
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ok = ggml_gallocr_alloc_graph(alloc, graph) && ok;
    if (ok) {
        uint32_t rng = 0x91e10da5u;
        const auto sample = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return 1.5f * ((int32_t)(rng >> 8) - 8388608) / 8388608.0f;
        };
        std::vector<float> qv(ggml_nelements(q));
        std::vector<ggml_fp16_t> kvv(f32_kv ? 0 : ggml_nelements(kv));
        std::vector<float> kvv_f32(f32_kv ? ggml_nelements(kv) : 0);
        std::vector<ggml_fp16_t> mv(ggml_nelements(mask), ggml_fp32_to_fp16(-1e30f));
        std::vector<int32_t> tv(ggml_nelements(topk));
        std::vector<float> sv(heads);
        for (auto & value : qv) value = sample();
        for (size_t i = 0; i < size_t(ggml_nelements(kv)); ++i) {
            const float value = sample();
            if (f32_kv) kvv_f32[i] = value;
            else        kvv[i] = ggml_fp32_to_fp16(value);
        }
        for (auto & value : sv) value = sample();
        if (stress_weight_bounds) {
            // Include an empty value envelope (all mass at the sink), along
            // with head-dependent holes and endpoints from exp underflow.
            // All inputs remain finite; skipping a zero weight must not
            // change another head's value-accumulation interval or order.
            for (int h = 0; h < heads; ++h) {
                for (int t = 0; t < tokens; ++t) {
                    for (int d = 0; d < dim; ++d) {
                        auto & value = qv[(size_t(h) * tokens + t) * dim + d];
                        value = h % 4 == 0 ? 0.0f : value * 128.0f;
                    }
                }
                sv[h] = h % 4 == 0 ? 1000.0f : -1000.0f;
            }
        }
        for (int t = 0; t < tokens; ++t) {
            auto * col = mv.data() + size_t(t) * rows;
            for (int row = std::max(0, prior + t - window + 1); row <= prior + t; ++row)
                col[row] = ggml_fp32_to_fp16(0.0f);
            for (int rank = 0; rank < selected; ++rank) {
                const int row = (t * 17 + selected - 1 - rank) % compressed;
                tv[size_t(t) * selected + rank] = row;
                if (row < (start + t + 1) / 4) col[raw + row] = ggml_fp32_to_fp16(0.0f);
            }
        }
        ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(kv, f32_kv ? static_cast<const void *>(kvv_f32.data())
                                          : static_cast<const void *>(kvv.data()),
                                0, ggml_nbytes(kv));
        ggml_backend_tensor_set(mask, mv.data(), 0, ggml_nbytes(mask));
        ggml_backend_tensor_set(topk, tv.data(), 0, ggml_nbytes(topk));
        ggml_backend_tensor_set(sinks, sv.data(), 0, ggml_nbytes(sinks));
        std::vector<float> expected(ggml_nelements(ref)), masked(expected.size()), analytic(expected.size());
        for (int replay = 0; replay < 3; ++replay) {
            const size_t launches_before = ggml_backend_cuda_get_mla_stream_topk_launch_count();
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                ok = false;
                break;
            }
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(ref, expected.data(), 0, ggml_nbytes(ref));
            ggml_backend_tensor_get(explicit_stream, masked.data(), 0, ggml_nbytes(explicit_stream));
            ggml_backend_tensor_get(analytic_stream, analytic.data(), 0, ggml_nbytes(analytic_stream));
            const size_t launches = ggml_backend_cuda_get_mla_stream_topk_launch_count() - launches_before;
            if (replay == 0 && launches != size_t(expected_launches)) {
                std::fprintf(stderr, "wrong selected-row dispatch count: expected=%d actual=%zu\n", expected_launches, launches);
                ok = false;
            }
            float max_abs = 0;
            size_t outside = 0;
            for (size_t i = 0; i < expected.size(); ++i) {
                const float error = std::abs(expected[i] - masked[i]);
                max_abs = std::max(max_abs, error);
                const float tolerance = 5e-4f + 5e-4f * std::max(std::abs(expected[i]), std::abs(masked[i]));
                outside += !std::isfinite(expected[i]) || !std::isfinite(masked[i]) ||
                           !std::isfinite(analytic[i]) || error > tolerance ||
                           std::abs(expected[i] - analytic[i]) >
                               5e-4f + 5e-4f * std::max(std::abs(expected[i]), std::abs(analytic[i]));
            }
            const bool identical = std::memcmp(masked.data(), analytic.data(), ggml_nbytes(ref)) == 0;
            // F32 prefill feeds the drafter's features as well as the target.
            // A loose attention tolerance can hide a lost accepted draft token.
            // Its optimized schedule must preserve the compact arithmetic.
            const bool reference_identical =
                std::memcmp(expected.data(), masked.data(), ggml_nbytes(ref)) == 0 &&
                std::memcmp(expected.data(), analytic.data(), ggml_nbytes(ref)) == 0;
            ok = ok && outside == 0 && (!require_byte_identity || identical) &&
                 (!f32_kv || !require_compact_identity || reference_identical);
            std::printf("start=%d tokens=%d heads=%d replay=%d compact_vs_stream_max_abs=%.8g outside=%zu masked_vs_analytic_bytes_equal=%d reference_bytes_equal=%d kv_type=%s stress_weight_bounds=%d\n",
                        start, tokens, heads, replay, max_abs, outside, identical,
                        reference_identical, f32_kv ? "f32" : "f16", stress_weight_bounds);
            std::fflush(stdout);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

// The executable is process-isolated: inherited tuning flags cannot disable
// the path under test, and no environment mutation escapes to a serving process.
static bool set_flag(const char * name, const char * value) {
#ifdef _WIN32
    return _putenv_s(name, value ? value : "") == 0;
#else
    return value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0;
#endif
}

int main(int argc, char ** argv) {
    bool defaults = false, disabled = false, f32_kv = false, short_maskless = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--defaults") == 0) defaults = true;
        else if (std::strcmp(argv[i], "--disabled") == 0) disabled = true;
        else if (std::strcmp(argv[i], "--f32") == 0) f32_kv = true;
        else if (std::strcmp(argv[i], "--short") == 0) short_maskless = true;
        else return 2;
    }
    if (defaults && disabled) return 2;
    const char * stream_flag = defaults ? nullptr : disabled ? "0" : "1";
    bool configured = set_flag("GGML_CUDA_MLA_STREAM_TOPK", stream_flag);
    configured = set_flag("GGML_DS4_FA_STREAM_TOPK", nullptr) && configured;
    configured = set_flag("GGML_CUDA_MLA_STREAM_F32_STAGE", defaults ? nullptr : "1") && configured;
    configured = set_flag("GGML_CUDA_MLA_STREAM_FAST_EXP", defaults ? nullptr : "1") && configured;
    configured = set_flag("GGML_CUDA_DISABLE_GRAPHS_DEVICES", nullptr) && configured;
    if (short_maskless) {
        configured = set_flag("GGML_CUDA_MLA_SPLIT_KV", "1") && configured;
        configured = set_flag("GGML_CUDA_MLA_NO_SPLIT_KV", nullptr) && configured;
        configured = set_flag("GGML_DS4_FA_NO_SPLIT_KV", nullptr) && configured;
    }
    if (!configured) return 2;
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess || properties.warpSize != 32) {
        std::puts("SKIP: requires a native wave32 HIP device");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 1;
    const int expected_launches = defaults ? 1 : disabled ? 0 : 2;
    bool ok = true;
    if (short_maskless) {
        // This valid small-window contract reaches the decode-width selector.
        // Split-KV needs an explicit mask; the analytic request must use its
        // compact fallback. The three schedules need numerical, not byte,
        // parity here (the normal F32 compact-order tests stay byte-exact).
        ok = check(backend, 4096, 8, 0, false, f32_kv, 16, false, 4, false);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    }
    if (!defaults && !disabled) {
        ok = check(backend, 0, 10240, expected_launches, true, f32_kv);
    }
    ok = check(backend, 122880, 129, expected_launches, !defaults, f32_kv) && ok;
    if (f32_kv && !defaults && !disabled) {
        // Exercise all model head groups, beyond the smaller 16-head fixture.
        ok = check(backend, 122880, 129, expected_launches, true, true, 64) && ok;
        // A distinct supported shape forces dispatch rather than a graph-cache
        // hit, so the first-submission launch-count assertion stays meaningful.
        ok = check(backend, 122880, 133, expected_launches, true, true, 64, true) && ok;
    }
    ggml_backend_free(backend);
    std::printf("wide/tail maskless differential: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
