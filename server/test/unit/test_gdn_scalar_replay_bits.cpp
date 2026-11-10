// The scalar GATED_DELTA_NET kernel (3090 default) and the replay-log commit
// must round the state update identically: fmaf(key, delta, gate * state).
// One T-step run through the scalar kernel and a commit of its replay log from
// the same start state must give bit-equal states, so spec-verify and AR agree.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int S = 128, H = 48, KEY_HEADS = 16, T = 4;

bool run(ggml_backend_t backend) {
    std::mt19937 rng(20261105);
    std::uniform_real_distribution<float> small(-0.25f, 0.25f), st(-0.06f, 0.06f),
        gate(0.82f, 0.98f), beta_d(0.15f, 0.85f);
    std::vector<float> q((size_t) S*KEY_HEADS*T), k(q.size()), v((size_t) S*H*T),
        g((size_t) H*T), beta((size_t) H*T), state((size_t) S*S*H);
    for (float & x : q) x = small(rng);
    for (float & x : v) x = small(rng);
    for (float & x : state) x = st(rng);
    for (float & x : g) x = std::log(gate(rng));
    for (float & x : beta) x = beta_d(rng);
    for (int t = 0; t < T; ++t) for (int h = 0; h < KEY_HEADS; ++h) {
        float n2 = 0;
        float * row = &k[((size_t) t*KEY_HEADS + h)*S];
        for (int i = 0; i < S; ++i) { row[i] = small(rng); n2 += row[i]*row[i]; }
        for (int i = 0; i < S; ++i) row[i] /= std::sqrt(n2);
    }

    ggml_init_params p{};
    p.mem_size = 8*1024*1024;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    auto t4 = [&](int a, int b, int c, int d) {
        return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b, c, d); };
    ggml_tensor * tq = t4(S, KEY_HEADS, T, 1), * tk = t4(S, KEY_HEADS, T, 1),
        * tv = t4(S, H, T, 1), * tg = t4(1, H, T, 1), * tb = t4(1, H, T, 1),
        * tstate = t4(S, S, H, 1), * cstate = t4(S, S, H, 1),
        * conv_in = t4(3 + T, 1, 1, 1), * conv_st = t4(3, 1, 1, 1);
    ggml_tensor * accepted = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * result = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tb, tstate);
    ggml_set_output(result);
    ggml_tensor * log = ggml_gated_delta_net_capture_replay_log(ctx, result);
    ggml_set_output(log);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_build_forward_expand(graph, log);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) return false;
    auto up = [](ggml_tensor * t, const std::vector<float> & x) {
        ggml_backend_tensor_set(t, x.data(), 0, x.size()*sizeof(float)); };
    up(tq, q); up(tk, k); up(tv, v); up(tg, g); up(tb, beta); up(tstate, state);
    up(cstate, state);
    std::vector<float> conv(3 + T, 0.0f), conv0(3, 0.0f);
    up(conv_in, conv); up(conv_st, conv0);
    int32_t acc = T, slot = 0;
    ggml_backend_tensor_set(accepted, &acc, 0, sizeof acc);
    ggml_backend_tensor_set(slots, &slot, 0, sizeof slot);

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    const ggml_tensor * logs[] = {log};
    ggml_tensor * states[] = {cstate};
    const ggml_tensor * ci[] = {conv_in};
    ggml_tensor * cs[] = {conv_st};
    if (!ggml_backend_cuda_gdn_replay_log_commit_many(logs, states, ci, cs, 1, accepted, slots))
        return false;

    std::vector<float> scalar_state(state.size()), replay_state(state.size());
    ggml_backend_tensor_get(result, scalar_state.data(),
        (size_t) S*H*T*sizeof(float), scalar_state.size()*sizeof(float));
    ggml_backend_tensor_get(cstate, replay_state.data(), 0,
        replay_state.size()*sizeof(float));
    size_t diff = 0;
    double maxabs = 0;
    for (size_t i = 0; i < state.size(); ++i) {
        if (std::memcmp(&scalar_state[i], &replay_state[i], 4) != 0) ++diff;
        maxabs = std::max(maxabs, (double) std::fabs(scalar_state[i] - replay_state[i]));
    }
    std::printf("scalar vs replay commit: %zu/%zu floats differ in bits, max abs %.3g\n",
                diff, state.size(), maxabs);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return diff == 0;
}
}  // namespace

int main() {
    setenv("LUCE_GDN_NO_GROUPED_COLS", "1", 1);
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "GPU backend unavailable\n"); return 1; }
    const bool ok = run(backend);
    ggml_backend_free(backend);
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
