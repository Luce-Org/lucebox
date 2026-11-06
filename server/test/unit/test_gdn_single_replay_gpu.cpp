// GPU check and microbench for SINGLE_REPLAY
// (qwen35/gdn_single_commit.h, ggml-cuda-gdn-chain.h). Exits 77 (ctest SKIP)
// without a GPU, or when the chain GDN op would not take the grouped-cols
// kernel here (SINGLE_REPLAY is inert then).
//
// Production shape: S_v 128, 48 v-heads over 16 k-heads, a 16-token verify.
//
// 1. Verify. The same random problem runs as the parent's checkpoint verify
//    (ggml_gated_delta_net, F32 checkpoints in src[7]) and as the single-replay
//    verify (ggml_gated_delta_net_mapped_verify over a one-slot map, replay
//    log). Both must take the grouped-cols kernel; their attention outputs
//    must be byte-identical, and neither may write the base state.
// 2. Commit. ggml_backend_cuda_gdn_chain_commit for commit_n 1..T must equal
//    intermediate[commit_n - 1] of the checkpoint verify byte for byte, out of
//    place and in place; a 48-layer launch must equal 48 one-layer launches.
//    Raw (in-kernel) and precomputed gates, T = 16 and T = 5.
// 3. Bench. 48 layers x 16 tokens: the commit (one launch, then a sync:
//    median, and back-to-back launches), the parent's 48 checkpoint copies for
//    reference, and the GDN op itself with and without checkpoints. The
//    times are a report; with --budget, a commit above 40 us per layer
//    exits 2 (tuned on one GPU, so ctest runs without it).
//
// Usage: test_gdn_single_replay_gpu [--iters N] [--budget]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cuda-gdn-chain.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

#define CHECK_MSG(cond, ...) do {                                      \
    if (!(cond)) {                                                     \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
        std::fprintf(stderr, __VA_ARGS__);                             \
        std::fprintf(stderr, "\n");                                    \
        failures++;                                                    \
    }                                                                  \
} while (0)

constexpr int kS = 128;          // head size
constexpr int kHv = 48;          // v-heads
constexpr int kHk = 16;          // k-heads (q/k broadcast h % 16)
constexpr int kLayers = 48;      // delta layers of the 27B target
constexpr double kBudgetUsPerLayer = 40.0;

using Bytes = std::vector<uint8_t>;

Bytes read_bytes(const ggml_tensor * t, size_t offset, size_t size) {
    Bytes b(size);
    ggml_backend_tensor_get(t, b.data(), offset, size);
    return b;
}

void upload_random(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v((size_t)ggml_nelements(t));
    for (float & x : v) x = d(rng);
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

size_t state_bytes() { return (size_t)kS * kS * kHv * sizeof(float); }

// One random GDN problem as both verifies, all tensors statically allocated.
struct Problem {
    int T = 0;
    bool raw = false;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * state = nullptr;
    ggml_tensor * persist = nullptr;   // [S, S, Hv, T] checkpoints
    ggml_tensor * ckpt = nullptr;      // checkpoint verify result
    ggml_tensor * replay = nullptr;    // mapped verify result
    ggml_tensor * log = nullptr;       // its replay log, in place
    ggml_tensor * out = nullptr;       // commit destination [S, S, Hv]
    ggml_tensor * scratch = nullptr;   // in-place commit target
    ggml_cgraph * gf_ckpt = nullptr;
    ggml_cgraph * gf_replay = nullptr;

    bool init(ggml_backend_t backend, int T_, bool raw_, unsigned seed) {
        T = T_;
        raw = raw_;
        ggml_init_params ip{};
        ip.mem_size = 64 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        if (!ctx) return false;
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kS, kHk, T, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kS, kHk, T, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kS, kHv, T, 1);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, kHv, T, 1);
        ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, kHv, T, 1);
        state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kS, kS, kHv, 1);
        persist = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kS, kS, kHv, T);
        out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kS, kS, kHv);
        scratch = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kS, kS, kHv);
        ggml_tensor * slot = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor * gate_ba = raw ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * kHv) : nullptr;

        // The parent's chain capture verify (build_delta_net_block).
        ckpt = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
        ckpt->src[7] = persist;
        if (gate_ba) ggml_gated_delta_net_set_raw_gates(ckpt, gate_ba);
        // SINGLE_REPLAY's: mapped verify over slab 0, log read in place.
        replay = ggml_gated_delta_net_mapped_verify(ctx, q, k, v, g, beta, state, slot);
        if (gate_ba) ggml_gated_delta_net_set_raw_gates(replay, gate_ba);
        ggml_tensor * copy = ggml_gated_delta_net_capture_replay_log(ctx, replay);
        log = copy->src[0];
        if (copy->op != GGML_OP_DUP || !log || log->view_src != replay) return false;

        gf_ckpt = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf_ckpt, ckpt);
        gf_replay = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf_replay, log);

        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) return false;

        std::mt19937 rng(seed);
        const float qk = 1.0f / std::sqrt((float)kS);
        upload_random(q, rng, -qk, qk);
        upload_random(k, rng, -qk, qk);
        upload_random(v, rng, -1.0f, 1.0f);
        upload_random(state, rng, -0.5f, 0.5f);
        if (raw) {
            upload_random(g, rng, -2.0f, 2.0f);      // alpha before softplus
            upload_random(beta, rng, -2.0f, 2.0f);   // beta before sigmoid
            std::vector<float> ba((size_t)2 * kHv);
            for (int h = 0; h < kHv; ++h) {
                ba[(size_t)h] = 0.1f * (float)(h % 7);
                ba[(size_t)kHv + h] = -1.0f - 0.05f * (float)(h % 11);
            }
            ggml_backend_tensor_set(gate_ba, ba.data(), 0, ggml_nbytes(gate_ba));
        } else {
            upload_random(g, rng, -1.0f, -0.05f);    // log decay
            upload_random(beta, rng, 0.1f, 0.9f);
        }
        const int32_t zero = 0;
        ggml_backend_tensor_set(slot, &zero, 0, sizeof(zero));
        return true;
    }

    Bytes checkpoint(int idx) const {
        return read_bytes(persist, (size_t)idx * persist->nb[3], state_bytes());
    }

    ~Problem() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

bool commit(ggml_backend_t backend, const ggml_tensor * log, const ggml_tensor * in,
            ggml_tensor * out, int64_t T, int n) {
    const ggml_cuda_gdn_chain_commit_layer layer = {
        (const float *)log->data, (const float *)in->data, (float *)out->data,
    };
    return ggml_backend_cuda_gdn_chain_commit(backend, &layer, 1, kS, kHv, T, n);
}

// ── 1 + 2: verify and commit, one problem ───────────────────────────────

void check_problem(ggml_backend_t backend, int T, bool raw) {
    Problem p;
    if (!p.init(backend, T, raw, 777u + (unsigned)(T * 3 + (raw ? 1 : 0)))) {
        CHECK_MSG(false, "T=%d raw=%d: setup", T, (int)raw);
        return;
    }
    const Bytes base = read_bytes(p.state, 0, state_bytes());

    size_t grouped = ggml_backend_cuda_get_gdn_grouped_cols_launch_count();
    CHECK_MSG(ggml_backend_graph_compute(backend, p.gf_ckpt) == GGML_STATUS_SUCCESS,
              "T=%d raw=%d: checkpoint verify", T, (int)raw);
    CHECK_MSG(ggml_backend_cuda_get_gdn_grouped_cols_launch_count() == grouped + 1,
              "T=%d raw=%d: checkpoint verify did not take the grouped-cols kernel", T, (int)raw);
    grouped = ggml_backend_cuda_get_gdn_grouped_cols_launch_count();
    CHECK_MSG(ggml_backend_graph_compute(backend, p.gf_replay) == GGML_STATUS_SUCCESS,
              "T=%d raw=%d: replay verify", T, (int)raw);
    CHECK_MSG(ggml_backend_cuda_get_gdn_grouped_cols_launch_count() == grouped + 1,
              "T=%d raw=%d: replay verify did not take the grouped-cols kernel", T, (int)raw);

    const size_t attn = (size_t)kS * kHv * T * sizeof(float);
    CHECK_MSG(read_bytes(p.ckpt, 0, attn) == read_bytes(p.replay, 0, attn),
              "T=%d raw=%d: attention output differs from the checkpoint verify", T, (int)raw);
    CHECK_MSG(read_bytes(p.state, 0, state_bytes()) == base,
              "T=%d raw=%d: a verify wrote the base state", T, (int)raw);
    // KEEP_BASE's property: the final state is the last checkpoint.
    CHECK_MSG(read_bytes(p.ckpt, attn, state_bytes()) == p.checkpoint(T - 1),
              "T=%d raw=%d: final state != intermediate[T-1]", T, (int)raw);

    for (int c = 1; c <= T; ++c) {
        const Bytes expect = p.checkpoint(c - 1);
        CHECK_MSG(commit(backend, p.log, p.state, p.out, T, c), "commit launch");
        ggml_backend_synchronize(backend);
        CHECK_MSG(read_bytes(p.out, 0, state_bytes()) == expect,
                  "T=%d raw=%d commit_n=%d: out-of-place commit != intermediate[commit_n-1]",
                  T, (int)raw, c);
        ggml_backend_tensor_copy(p.state, p.scratch);
        CHECK_MSG(commit(backend, p.log, p.scratch, p.scratch, T, c), "commit launch");
        ggml_backend_synchronize(backend);
        CHECK_MSG(read_bytes(p.scratch, 0, state_bytes()) == expect,
                  "T=%d raw=%d commit_n=%d: in-place commit != intermediate[commit_n-1]",
                  T, (int)raw, c);
    }
    CHECK_MSG(read_bytes(p.state, 0, state_bytes()) == base,
              "T=%d raw=%d: an out-of-place commit wrote its source", T, (int)raw);

    // Arguments the commit refuses without launching.
    CHECK_MSG(!commit(backend, p.log, p.state, p.out, T, 0), "commit_n 0 accepted");
    CHECK_MSG(!commit(backend, p.log, p.state, p.out, T, T + 1), "commit_n > T accepted");
    CHECK_MSG(!commit(backend, p.log, p.state, p.out, 17, 17), "commit_n 17 accepted");
    const ggml_cuda_gdn_chain_commit_layer layer = {
        (const float *)p.log->data, (const float *)p.state->data, (float *)p.out->data,
    };
    CHECK_MSG(!ggml_backend_cuda_gdn_chain_commit(backend, &layer, 1, 64, kHv, T, 1),
              "S_v 64 accepted");
}

// ── 2 (layers) + 3: 48 layers ───────────────────────────────────────────

struct Layers {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> logs, in, out, ref, inter;
    ~Layers() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

double now_us() {
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1e3;
}

// Returns false when the commit is over budget.
bool run_layers(ggml_backend_t backend, int iters) {
    constexpr int T = 16;
    Problem p;
    if (!p.init(backend, T, /*raw=*/true, 4242u)) {
        CHECK_MSG(false, "bench setup");
        return true;
    }
    CHECK_MSG(ggml_backend_graph_compute(backend, p.gf_replay) == GGML_STATUS_SUCCESS,
              "bench replay verify");

    Layers L;
    ggml_init_params ip{};
    ip.mem_size = (size_t)(5 * kLayers + 4) * ggml_tensor_overhead();
    ip.no_alloc = true;
    L.ctx = ggml_init(ip);
    for (int i = 0; i < kLayers; ++i) {
        L.logs.push_back(ggml_new_tensor_4d(L.ctx, GGML_TYPE_F32, 2 * kS + 1, kHv, T, 1));
        L.in.push_back(ggml_new_tensor_3d(L.ctx, GGML_TYPE_F32, kS, kS, kHv));
        L.out.push_back(ggml_new_tensor_3d(L.ctx, GGML_TYPE_F32, kS, kS, kHv));
        L.ref.push_back(ggml_new_tensor_3d(L.ctx, GGML_TYPE_F32, kS, kS, kHv));
        L.inter.push_back(ggml_new_tensor_3d(L.ctx, GGML_TYPE_F32, kS, kS, kHv));
    }
    L.buf = ggml_backend_alloc_ctx_tensors(L.ctx, backend);
    if (!L.buf) {
        CHECK_MSG(false, "48-layer buffers");
        return true;
    }
    std::mt19937 rng(99u);
    for (int i = 0; i < kLayers; ++i) {
        ggml_backend_tensor_copy(p.log, L.logs[(size_t)i]);
        upload_random(L.in[(size_t)i], rng, -0.5f, 0.5f);
        ggml_backend_tensor_copy(L.in[(size_t)i], L.inter[(size_t)i]);
    }
    std::vector<ggml_cuda_gdn_chain_commit_layer> layers((size_t)kLayers);
    for (int i = 0; i < kLayers; ++i) {
        layers[(size_t)i] = {(const float *)L.logs[(size_t)i]->data,
                             (const float *)L.in[(size_t)i]->data,
                             (float *)L.out[(size_t)i]->data};
    }

    // One launch for 48 layers == 48 one-layer launches.
    for (int c : {16, 7, 1}) {
        CHECK_MSG(ggml_backend_cuda_gdn_chain_commit(backend, layers.data(), kLayers,
                                                     kS, kHv, T, c), "48-layer commit");
        for (int i = 0; i < kLayers; ++i) {
            CHECK_MSG(commit(backend, L.logs[(size_t)i], L.in[(size_t)i], L.ref[(size_t)i], T, c),
                      "one-layer commit");
        }
        ggml_backend_synchronize(backend);
        for (int i = 0; i < kLayers; ++i) {
            if (read_bytes(L.out[(size_t)i], 0, state_bytes()) !=
                read_bytes(L.ref[(size_t)i], 0, state_bytes())) {
                CHECK_MSG(false, "commit_n=%d layer %d: 48-layer launch != one-layer launch", c, i);
                break;
            }
        }
    }

    // Commit: one launch then a sync (what rollback_to pays), and
    // back-to-back launches (GPU time).
    for (int w = 0; w < 5; ++w) {
        CHECK_MSG(ggml_backend_cuda_gdn_chain_commit(backend, layers.data(), kLayers, kS, kHv, T, T),
                  "warm-up commit");
    }
    ggml_backend_synchronize(backend);
    std::vector<double> lat;
    for (int it = 0; it < iters; ++it) {
        const double t0 = now_us();
        CHECK_MSG(ggml_backend_cuda_gdn_chain_commit(backend, layers.data(), kLayers, kS, kHv, T, T),
                  "timed commit");
        ggml_backend_synchronize(backend);
        lat.push_back(now_us() - t0);
    }
    const double t0 = now_us();
    for (int it = 0; it < iters; ++it) {
        CHECK_MSG(ggml_backend_cuda_gdn_chain_commit(backend, layers.data(), kLayers, kS, kHv, T, T),
                  "back-to-back commit");
    }
    ggml_backend_synchronize(backend);
    const double gpu_us = (now_us() - t0) / iters;
    const double commit_us = median(lat);

    // One layer, one launch, then a sync.
    std::vector<double> lat1;
    for (int it = 0; it < iters; ++it) {
        const double s = now_us();
        CHECK_MSG(ggml_backend_cuda_gdn_chain_commit(backend, layers.data(), 1, kS, kHv, T, T),
                  "1-layer timed commit");
        ggml_backend_synchronize(backend);
        lat1.push_back(now_us() - s);
    }

    // The parent's rollback: 48 checkpoint copies, then a sync.
    std::vector<double> copy_lat;
    for (int it = 0; it < iters; ++it) {
        const double s = now_us();
        for (int i = 0; i < kLayers; ++i) {
            ggml_backend_tensor_copy_async(backend, backend, L.inter[(size_t)i], L.ref[(size_t)i]);
        }
        ggml_backend_synchronize(backend);
        copy_lat.push_back(now_us() - s);
    }

    // The GDN op, one layer, with and without checkpoints (compute + sync).
    std::vector<double> op_ckpt, op_replay;
    for (int it = 0; it < iters; ++it) {
        double s = now_us();
        ggml_backend_graph_compute(backend, p.gf_ckpt);
        op_ckpt.push_back(now_us() - s);
        s = now_us();
        ggml_backend_graph_compute(backend, p.gf_replay);
        op_replay.push_back(now_us() - s);
    }

    const double per_layer = commit_us / kLayers;
    const bool over = per_layer > kBudgetUsPerLayer;
    std::printf("[single-replay-bench] commit %d layers x %d tokens (H %d, S_v %d): "
                "%.1f us launch+sync median = %.2f us/layer; back-to-back %.2f us/layer\n",
                kLayers, T, kHv, kS, commit_us, per_layer, gpu_us / kLayers);
    std::printf("[single-replay-bench] commit 1 layer x %d tokens: %.1f us launch+sync median\n",
                T, median(lat1));
    std::printf("[single-replay-bench] parent rollback, %d checkpoint copies: %.1f us = %.2f us/layer\n",
                kLayers, median(copy_lat), median(copy_lat) / kLayers);
    std::printf("[single-replay-bench] GDN op, 1 layer x %d tokens, compute+sync median: "
                "checkpoints %.1f us, replay log %.1f us\n",
                T, median(op_ckpt), median(op_replay));
    std::printf("[single-replay-bench] budget %.0f us/layer: %s\n",
                kBudgetUsPerLayer, over ? "OVER" : "PASS");
    std::fflush(stdout);
    return !over;
}

}  // namespace

int main(int argc, char ** argv) {
    int iters = 100;
    bool enforce_budget = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--budget") == 0) {
            enforce_budget = true;
        }
    }
    if (ggml_backend_cuda_get_device_count() < 1) {
        std::fprintf(stderr, "test_gdn_single_replay_gpu: no GPU\n");
        return 77;
    }
    if (!ggml_backend_cuda_gdn_chain_uses_grouped_cols(0, kS)) {
        std::fprintf(stderr,
            "test_gdn_single_replay_gpu: the chain GDN op does not take the grouped-cols "
            "kernel on device 0; SINGLE_REPLAY stays inert\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "test_gdn_single_replay_gpu: device 0 present but its backend failed to init\n");
        return 1;
    }
    for (int T : {16, 5}) {
        for (bool raw : {true, false}) check_problem(backend, T, raw);
    }
    const bool within_budget = run_layers(backend, iters);
    ggml_backend_free(backend);
    if (failures) {
        std::fprintf(stderr, "test_gdn_single_replay_gpu: %d failure(s)\n", failures);
        return 1;
    }
    return (enforce_budget && !within_budget) ? 2 : 0;
}
