// Exact integer comparisons: fused CPU/GPU ids vs the original graph and a scalar reference.
// --cpu runs without a GPU; otherwise exit 77 when no CUDA/HIP device is available.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifndef QSA_IDS_CPU_ONLY
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

// Original build_qsa_attn post-top-k graph, kept here to catch ordering/tail regressions.
static ggml_tensor * legacy_ids(ggml_context * c, ggml_tensor * blocks, int kv_start, int r) {
    const int64_t budget = blocks->ne[0], T = blocks->ne[1];
    blocks = ggml_cont(c, blocks);
    ggml_tensor * order = ggml_argsort(c, ggml_cast(c, blocks, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);
    blocks = ggml_reshape_3d(c, ggml_get_rows(c,
        ggml_view_4d(c, blocks, 1, budget, T, 1, blocks->nb[0], blocks->nb[1], blocks->nb[2], 0),
        order), budget, T, 1);

    ggml_tensor * shape3 = ggml_new_tensor_3d(c, GGML_TYPE_F32, r, budget, T);
    ggml_tensor * tv = ggml_repeat(c,
        ggml_reshape_3d(c, ggml_scale_bias(c, ggml_arange(c, 0.0f, (float) T, 1.0f), 1.0f, (float) kv_start),
                        1, 1, T), shape3);
    ggml_tensor * bf = ggml_cast(c, blocks, GGML_TYPE_F32);                        // [budget,T,1]
    ggml_tensor * bf3 = ggml_repeat(c, ggml_reshape_3d(c, bf, 1, budget, T), shape3);
    ggml_tensor * lim = ggml_scale_bias(c, ggml_scale(c, bf3, (float) r), 1.0f, (float) (r - 1));
    ggml_tensor * valid = ggml_step(c, ggml_scale_bias(c, ggml_sub(c, tv, lim), 1.0f, 1.0f));

    ggml_tensor * iv  = ggml_repeat(c, ggml_reshape_3d(c,
        ggml_arange(c, 0.0f, (float) r, 1.0f), r, 1, 1), shape3);
    ggml_tensor * cf  = ggml_scale_bias(c, ggml_add(c, ggml_scale(c, bf3, (float) r), iv), 1.0f, 1.0f);
    cf = ggml_scale_bias(c, ggml_mul(c, cf, valid), 1.0f, -1.0f);                  // (r*b+i+1)*valid - 1
    ggml_tensor * cells = ggml_reshape_4d(c, ggml_cast(c, cf, GGML_TYPE_I32), budget * r, T, 1, 1);

    ggml_tensor * tvt = ggml_scale_bias(c, ggml_arange(c, 0.0f, (float) T, 1.0f), 1.0f, (float) kv_start);
    ggml_tensor * br  = ggml_scale(c, ggml_floor(c, ggml_scale(c,
        ggml_scale_bias(c, ggml_arange(c, 1.0f, (float) (T + 1), 1.0f), 1.0f, (float) kv_start),
        1.0f / (float) r)), (float) r);
    ggml_tensor * rows = nullptr;
    for (int64_t i = 0; i < r - 1; ++i) {
        ggml_tensor * cell = ggml_scale_bias(c, br, 1.0f, (float) i);
        ggml_tensor * v = ggml_step(c, ggml_scale_bias(c, ggml_sub(c, tvt, cell), 1.0f, 1.0f));
        ggml_tensor * val = ggml_scale_bias(c, ggml_mul(c, v, ggml_scale_bias(c, cell, 1.0f, 1.0f)), 1.0f, -1.0f);
        ggml_tensor * row = ggml_reshape_2d(c, ggml_cast(c, val, GGML_TYPE_I32), 1, T);
        rows = rows ? ggml_concat(c, rows, row, 0) : row;
    }
    return ggml_reshape_2d(c,
        ggml_concat(c, cells, ggml_reshape_3d(c, rows, r - 1, T, 1), 0),
        budget * r + (r - 1), T);
}

struct Result {
    bool ok = false;
    std::vector<int32_t> selected, ids;
};

static std::vector<int32_t> reference(const std::vector<int32_t> & blocks, int T, int budget, int pos0, int r) {
    std::vector<int32_t> out;
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> row(blocks.begin() + t * budget, blocks.begin() + (t + 1) * budget);
        std::sort(row.begin(), row.end());
        const int64_t pos = (int64_t) pos0 + t;
        for (int32_t b : row) {
            for (int i = 0; i < r; ++i) out.push_back((int64_t) r * b + r - 1 <= pos ? r * b + i : -1);
        }
        const int64_t br = (pos + 1) / r * r;
        for (int i = 0; i < r - 1; ++i) out.push_back(br + i <= pos ? (int32_t) (br + i) : -1);
    }
    return out;
}

static Result run(ggml_backend_t be, int T, int budget, int pos0, int r,
                  const std::vector<int32_t> & blocks, const std::vector<float> & scores = {}, bool enumerate = false) {
    ggml_context * c = ggml_init({8 * 1024 * 1024, nullptr, true});
    Result result;
    const int n_comp = scores.empty() ? 0 : (int) scores.size() / T;
    ggml_tensor * input = scores.empty() ? ggml_new_tensor_2d(c, GGML_TYPE_I32, budget + 7, T)
                                        : ggml_new_tensor_2d(c, GGML_TYPE_F32, n_comp, T);
    // Exercise both nonzero view offsets and a non-contiguous row stride.
    ggml_tensor * selected = scores.empty() ? ggml_view_2d(c, input, budget, T, input->nb[1], 3 * sizeof(int32_t))
                                           : ggml_top_k(c, input, budget);
    if (enumerate) {
        selected = ggml_cast(c, ggml_repeat_4d(c, ggml_arange(c, 0.0f, (float) budget, 1.0f),
            budget, T, 1, 1), GGML_TYPE_I32);
    }
    ggml_tensor * positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * T);
    ggml_tensor * fused = ggml_qsa_decode_ids(c, selected, ggml_view_1d(c, positions, T, 0), r);
    ggml_tensor * old = legacy_ids(c, selected, pos0, r);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, fused);
    ggml_build_forward_expand(gf, old);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { ggml_free(c); return result; }
    if (scores.empty()) {
        std::vector<int32_t> padded((size_t) (budget + 7) * T, -999);
        for (int t = 0; t < T; ++t) {
            std::copy_n(blocks.data() + (size_t) t * budget, budget, padded.data() + (size_t) t * (budget + 7) + 3);
        }
        ggml_backend_tensor_set(input, padded.data(), 0, ggml_nbytes(input));
    } else {
        ggml_backend_tensor_set(input, scores.data(), 0, ggml_nbytes(input));
    }
    std::vector<int32_t> pos((size_t) 4 * T, -999);
    for (int t = 0; t < T; ++t) pos[t] = pos0 + t;
    ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
    if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
        result.ids.resize((size_t) fused->ne[0] * T);
        std::vector<int32_t> old_ids(result.ids.size());
        result.selected.resize((size_t) budget * T);
        ggml_backend_tensor_get(fused, result.ids.data(), 0, ggml_nbytes(fused));
        ggml_backend_tensor_get(old, old_ids.data(), 0, ggml_nbytes(old));
        for (int t = 0; t < T; ++t) {
            ggml_backend_tensor_get(selected, result.selected.data() + (size_t) t * budget,
                                    t * selected->nb[1], budget * sizeof(int32_t));
        }
        result.ok = result.ids == old_ids && result.ids == reference(result.selected, T, budget, pos0, r);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return result;
}

int main(int argc, char ** argv) {
    const bool cpu_only = argc == 2 && std::strcmp(argv[1], "--cpu") == 0;
    ggml_backend_t gpu = nullptr;
#ifndef QSA_IDS_CPU_ONLY
    if (!cpu_only) gpu = ggml_backend_cuda_init(0);
#endif
    if (!cpu_only && !gpu) return 77;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) return 1;
    ggml_backend_cpu_set_n_threads(cpu, 4);
    bool ok = true;
    int cases = 0;
    // Every tail residue, the dense/QSA transition, and context/block boundaries.
    for (int T : {1, 2, 8, 129}) {
        for (int pos0 : {0, 1, 2, 3, 2047, 2048, 2049, 2050, 2051, 4095, 4096, 5996}) {
            const int budget = 512, r = 4;
            std::vector<int32_t> blocks((size_t) budget * T);
            for (int t = 0; t < T; ++t) {
                for (int j = 0; j < budget; ++j) blocks[(size_t) t * budget + j] = (j * 137 + t * 17) % 2048;
                const int edge = (pos0 + t + 1) / r;
                blocks[(size_t) t * budget] = std::max(0, edge - 1);
                blocks[(size_t) t * budget + 1] = edge;
                blocks[(size_t) t * budget + 2] = edge + 1;
            }
            const Result want = run(cpu, T, budget, pos0, r, blocks);
            const Result got = gpu ? run(gpu, T, budget, pos0, r, blocks) : want;
            const bool pass = want.ok && got.ok && want.ids == got.ids;
            std::printf("qsa-ids T=%-3d pos=%-5d %s\n", T, pos0, pass ? "OK" : "FAIL");
            ok &= pass; ++cases;
        }
    }
    // Dense-regime IDs must enumerate EXACTLY [0, absolute_query], including
    // the <4-token dummy block and the last all-keys query at position 2050.
    for (int T : {1, 2, 3, 4, 127, 128, 2048}) {
        for (int pos0 : {0, 1, 3, 1024, 2047}) {
            if (pos0 + T > 2051) continue;
            const int budget = std::max(1, (pos0 + T) / 4);
            std::vector<int32_t> blocks((size_t) budget * T);
            for (int t = 0; t < T; ++t) for (int b = 0; b < budget; ++b) blocks[(size_t) t * budget + b] = b;
            const Result got = run(gpu ? gpu : cpu, T, budget, pos0, 4, blocks, {}, true);
            bool pass = got.ok;
            for (int t = 0; pass && t < T; ++t) {
                int next = 0;
                for (int j = 0; j < budget * 4 + 3; ++j) {
                    const int id = got.ids[(size_t) t * (budget * 4 + 3) + j];
                    if (id >= 0) pass &= id == next++;
                }
                pass &= next == pos0 + t + 1;
            }
            std::printf("qsa-ids dense T=%-4d pos=%-4d %s\n", T, pos0, pass ? "OK" : "FAIL");
            ok &= pass; ++cases;
        }
    }
    // Bitonic padding, the maximum budget, other ratios, duplicate ids, and the float-exact range edge.
    for (int budget : {1, 5, 511, 513, 1024}) {
        for (int r : {2, 3, 4, 8}) {
            const int T = 4, pos0 = r == 4 ? (1 << 24) - 8 : 1000;
            std::vector<int32_t> blocks((size_t) budget * T);
            for (size_t i = 0; i < blocks.size(); ++i) blocks[i] = (int32_t) ((blocks.size() - i) % budget);
            blocks[0] = pos0 / r;
            const Result want = run(cpu, T, budget, pos0, r, blocks);
            const Result got = gpu ? run(gpu, T, budget, pos0, r, blocks) : want;
            const bool pass = want.ok && got.ok && want.ids == got.ids;
            std::printf("qsa-ids budget=%-4d ratio=%d %s\n", budget, r, pass ? "OK" : "FAIL");
            ok &= pass; ++cases;
        }
    }
    // Preserve EACH backend's existing top-k tie membership: share its selected tensor between the old and
    // fused graph, then feed precisely those selected ids to the CPU reference. CPU/GPU top-k ties may differ.
    for (int n_comp : {513, 1024, 1025, 1500, 4096, 5120, 5121}) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            const int T = 4, budget = 512, pos0 = 2048;
            std::vector<float> scores((size_t) n_comp * T);
            for (int t = 0; t < T; ++t) {
                for (int b = 0; b < n_comp; ++b) {
                    scores[(size_t) t * n_comp + b] = pattern == 0 ? 1.0f :
                        b >= (pos0 + t + 1) / 4 ? -1.0e30f : pattern == 1 ? (float) (b % 7) : 0.0f;
                }
            }
            const Result got = run(gpu ? gpu : cpu, T, budget, pos0, 4, {}, scores);
            const Result want = got.ok ? run(cpu, T, budget, pos0, 4, got.selected) : Result{};
            const bool pass = got.ok && want.ok && got.ids == want.ids;
            std::printf("qsa-ids top-k n=%-4d ties=%d %s\n", n_comp, pattern, pass ? "OK" : "FAIL");
            ok &= pass; ++cases;
        }
    }
    ggml_backend_free(cpu);
    if (gpu) ggml_backend_free(gpu);
    std::printf("qsa-ids: %d cases, %s (%s)\n", cases, ok ? "PASS" : "FAIL", gpu ? "GPU + CPU" : "CPU only");
    return ok ? 0 : 1;
}
