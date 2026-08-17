// Batched-decode correctness foundation: proves the n_seqs batch axis of
// ggml_gated_delta_net and ggml_ssm_conv is exactly equivalent to running
// each sequence independently. The concurrent serving path (--max-concurrency N)
// relies on this: it stacks N single-token decodes on the n_seqs axis of
// both ops with per-slot state slabs, so any cross-sequence leakage or
// stride bug here would silently corrupt every parallel stream.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

// qwen35-like shapes scaled down. S_V doubles as head_k_dim: the fused GDN
// kernel reads q/k rows with the S_v extent, and 32 is a supported kernel
// specialization (16/32/64/128).
constexpr int S_V = 32;
constexpr int N_HEAD = 4;
constexpr int N_SEQS = 4;
constexpr int D_CONV = 4;
// CUDA ssm_conv launches 128-thread blocks and asserts channels % 128 == 0.
constexpr int CONV_CHANNELS = 128;
constexpr int N_STEPS = 3;
constexpr float MAX_ABS_ERROR = 1.0e-5f;

// Single-token decode: n_tokens == 1, so each sequence's slice of every
// contiguous [.., n_tokens, n_seqs] host buffer is one contiguous chunk.
constexpr size_t GDN_QKV_PER_SEQ = static_cast<size_t>(S_V) * N_HEAD;
constexpr size_t GDN_GATE_PER_SEQ = N_HEAD;
constexpr size_t GDN_STATE_PER_SEQ = static_cast<size_t>(S_V) * S_V * N_HEAD;
constexpr size_t CONV_IN_PER_SEQ =
    static_cast<size_t>(D_CONV) * CONV_CHANNELS;  // (d_conv-1) history + 1 token
constexpr size_t CONV_HIST_PER_SEQ =
    static_cast<size_t>(D_CONV - 1) * CONV_CHANNELS;
constexpr size_t CONV_OUT_PER_SEQ = CONV_CHANNELS;
constexpr size_t CONV_WEIGHT_ELEMS = static_cast<size_t>(D_CONV) * CONV_CHANNELS;

bool check(const char * name, const float * batched, const float * reference,
           size_t count);

void fill_uniform(std::mt19937 & rng, float lo, float hi,
                  std::vector<float> & values) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float & value : values) value = dist(rng);
}

// One fused GDN decode step over n_seqs sequences (n_tokens == 1 each).
// Reads the packed result: attn [S_V*N_HEAD per seq] followed by the final
// state [S_V*S_V*N_HEAD per seq] — the same regions the backend slices.
bool run_gdn(ggml_backend_t backend, int n_seqs,
             const float * q, const float * k, const float * v,
             const float * g, const float * beta, const float * state,
             float * attn_out, float * state_out,
             bool inplace = false,
             const int32_t * active_slot_ids = nullptr,
             int physical_slots = 0,
             float * scratch_state_out = nullptr) {
    if (physical_slots == 0) physical_slots = n_seqs;
    if (physical_slots < 1 ||
        (!active_slot_ids && physical_slots != n_seqs)) {
        return false;
    }
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q_t =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_V, N_HEAD, 1, n_seqs);
    ggml_tensor * k_t =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_V, N_HEAD, 1, n_seqs);
    ggml_tensor * v_t =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_V, N_HEAD, 1, n_seqs);
    ggml_tensor * g_t =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, N_HEAD, 1, n_seqs);
    ggml_tensor * beta_t =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, N_HEAD, 1, n_seqs);
    ggml_tensor * state_t = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, S_V, S_V, N_HEAD, physical_slots);
    for (ggml_tensor * input : {q_t, k_t, v_t, g_t, beta_t, state_t}) {
        ggml_set_input(input);
    }
    ggml_tensor * active_t = nullptr;
    if (active_slot_ids) {
        active_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        ggml_set_input(active_t);
    }

    ggml_tensor * result = active_t
        ? ggml_gated_delta_net_active_inplace(
              ctx, q_t, k_t, v_t, g_t, beta_t, state_t, active_t)
        : inplace
            ? ggml_gated_delta_net_inplace(
                  ctx, q_t, k_t, v_t, g_t, beta_t, state_t)
            : ggml_gated_delta_net(
                  ctx, q_t, k_t, v_t, g_t, beta_t, state_t);
    // Decode path compacts the result to [attn | final_state], exactly like
    // build_delta_net_block with skip_gdn_intermediate.
    ggml_gated_delta_net_set_skip_intermediate(result, true);
    const bool state_inplace = inplace || active_t;
    // Active-slot mode writes persistent state in place, but its result keeps
    // one state-sized scratch region per batch row for negative padding ids.
    const bool result_has_state = !inplace || active_t;
    const size_t expected_result_elems =
        (GDN_QKV_PER_SEQ + (result_has_state ? GDN_STATE_PER_SEQ : 0)) *
        n_seqs;
    if (ggml_nelements(result) != expected_result_elems) {
        std::fprintf(stderr,
                     "batched gdn: %s result has %zu elements, expected %zu\n",
                     state_inplace ? "in-place" : "packed",
                     static_cast<size_t>(ggml_nelements(result)),
                     expected_result_elems);
        ggml_free(ctx);
        return false;
    }
    ggml_set_output(result);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        const size_t qkv_bytes = GDN_QKV_PER_SEQ * n_seqs * sizeof(float);
        const size_t gate_bytes = GDN_GATE_PER_SEQ * n_seqs * sizeof(float);
        const size_t state_bytes =
            GDN_STATE_PER_SEQ * physical_slots * sizeof(float);
        ggml_backend_tensor_set(q_t, q, 0, qkv_bytes);
        ggml_backend_tensor_set(k_t, k, 0, qkv_bytes);
        ggml_backend_tensor_set(v_t, v, 0, qkv_bytes);
        ggml_backend_tensor_set(g_t, g, 0, gate_bytes);
        ggml_backend_tensor_set(beta_t, beta, 0, gate_bytes);
        ggml_backend_tensor_set(state_t, state, 0, state_bytes);
        if (active_t) {
            ggml_backend_tensor_set(active_t, active_slot_ids, 0,
                                    n_seqs * sizeof(active_slot_ids[0]));
        }
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            const size_t attn_bytes = GDN_QKV_PER_SEQ * n_seqs * sizeof(float);
            ggml_backend_tensor_get(result, attn_out, 0, attn_bytes);
            if (state_inplace) {
                ggml_backend_tensor_get(state_t, state_out, 0, state_bytes);
            } else {
                ggml_backend_tensor_get(
                    result, state_out, attn_bytes, state_bytes);
            }
            if (scratch_state_out) {
                ggml_backend_tensor_get(
                    result, scratch_state_out, attn_bytes,
                    GDN_STATE_PER_SEQ * n_seqs * sizeof(float));
            }
        }
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

// One ssm_conv step: sx [d_conv-1 + 1, channels, n_seqs] against shared
// [d_conv, channels] weights, producing [channels, 1, n_seqs].
bool run_conv(ggml_backend_t backend, int n_seqs,
              const float * sx, const float * weights, float * out) {
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * sx_t =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_CONV, CONV_CHANNELS, n_seqs);
    ggml_tensor * w_t =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_CONV, CONV_CHANNELS);
    for (ggml_tensor * input : {sx_t, w_t}) {
        ggml_set_input(input);
    }

    ggml_tensor * result = ggml_ssm_conv(ctx, sx_t, w_t);
    ggml_set_output(result);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ggml_backend_tensor_set(sx_t, sx, 0,
                                CONV_IN_PER_SEQ * n_seqs * sizeof(float));
        ggml_backend_tensor_set(w_t, weights, 0,
                                CONV_WEIGHT_ELEMS * sizeof(float));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(result, out, 0,
                                    CONV_OUT_PER_SEQ * n_seqs * sizeof(float));
        }
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

bool test_masked_set_rows(ggml_backend_t backend) {
    constexpr int ROW_WIDTH = 4;
    constexpr int DEST_ROWS = 4;
    constexpr int SOURCE_ROWS = 3;

    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * destination = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, ROW_WIDTH, DEST_ROWS);
    ggml_tensor * source = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, ROW_WIDTH, SOURCE_ROWS);
    ggml_tensor * row_ids = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, SOURCE_ROWS);
    for (ggml_tensor * input : {destination, source, row_ids}) {
        ggml_set_input(input);
    }
    ggml_tensor * result =
        ggml_set_rows_masked(ctx, destination, source, row_ids);
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(allocator, graph);
    std::vector<float> destination_data(ROW_WIDTH * DEST_ROWS);
    std::vector<float> source_data(ROW_WIDTH * SOURCE_ROWS);
    for (size_t i = 0; i < destination_data.size(); ++i) {
        destination_data[i] = 100.0f + static_cast<float>(i);
    }
    for (size_t i = 0; i < source_data.size(); ++i) {
        source_data[i] = 200.0f + static_cast<float>(i);
    }
    const std::vector<int32_t> ids{2, -1, 0};
    if (ok) {
        ggml_backend_tensor_set(destination, destination_data.data(), 0,
                                destination_data.size() * sizeof(float));
        ggml_backend_tensor_set(source, source_data.data(), 0,
                                source_data.size() * sizeof(float));
        ggml_backend_tensor_set(row_ids, ids.data(), 0,
                                ids.size() * sizeof(ids[0]));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        std::vector<float> actual(destination_data.size());
        std::vector<float> expected = destination_data;
        ggml_backend_tensor_get(result, actual.data(), 0,
                                actual.size() * sizeof(float));
        std::copy_n(source_data.data(), ROW_WIDTH,
                    expected.data() + 2 * ROW_WIDTH);
        std::copy_n(source_data.data() + 2 * ROW_WIDTH, ROW_WIDTH,
                    expected.data());
        ok = check("masked state scatter", actual.data(), expected.data(),
                   actual.size());
    }

    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

bool check(const char * name, const float * batched, const float * reference,
           size_t count) {
    float max_abs_diff = 0.0f;
    bool finite = true;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(batched[i]) || !std::isfinite(reference[i])) {
            finite = false;
            break;
        }
        max_abs_diff =
            std::max(max_abs_diff, std::fabs(batched[i] - reference[i]));
    }
    const bool ok = finite && max_abs_diff < MAX_ABS_ERROR;
    std::printf("batched gdn %-24s max_abs=%.6g %s\n", name,
                finite ? max_abs_diff : INFINITY, ok ? "PASS" : "FAIL");
    return ok;
}

// Fresh random single-token inputs for all N_SEQS sequences. Ranges follow
// what the model block feeds the fused op: q/k are L2-normalized (order-one
// entries), beta is a sigmoid in (0,1), and g = -A.exp()*softplus(alpha) is
// negative so exp(g) <= 1 keeps the recurrence contractive.
struct GdnStep {
    std::vector<float> q, k, v, g, beta;

    explicit GdnStep(std::mt19937 & rng)
        : q(GDN_QKV_PER_SEQ * N_SEQS), k(GDN_QKV_PER_SEQ * N_SEQS),
          v(GDN_QKV_PER_SEQ * N_SEQS), g(GDN_GATE_PER_SEQ * N_SEQS),
          beta(GDN_GATE_PER_SEQ * N_SEQS) {
        fill_uniform(rng, -1.0f, 1.0f, q);
        fill_uniform(rng, -1.0f, 1.0f, k);
        fill_uniform(rng, -1.0f, 1.0f, v);
        fill_uniform(rng, -1.0f, 0.0f, g);
        fill_uniform(rng, 0.05f, 0.95f, beta);
    }
};

// Runs one decode step batched (n_seqs = N_SEQS) and per-sequence
// (N_SEQS calls with n_seqs = 1) from the SAME carried states, compares
// attn and final state, then advances both state copies for the next step.
bool step_and_compare(ggml_backend_t backend, const char * label,
                      const GdnStep & step,
                      std::vector<float> & state_batched,
                      std::vector<float> & state_reference,
                      bool inplace_batched) {
    std::vector<float> attn_batched(GDN_QKV_PER_SEQ * N_SEQS);
    std::vector<float> attn_reference(GDN_QKV_PER_SEQ * N_SEQS);
    std::vector<float> next_batched(GDN_STATE_PER_SEQ * N_SEQS);
    std::vector<float> next_reference(GDN_STATE_PER_SEQ * N_SEQS);

    bool ok = run_gdn(backend, N_SEQS, step.q.data(), step.k.data(),
                      step.v.data(), step.g.data(), step.beta.data(),
                      state_batched.data(), attn_batched.data(),
                      next_batched.data(), inplace_batched);
    for (int seq = 0; ok && seq < N_SEQS; ++seq) {
        ok = run_gdn(backend, 1,
                     step.q.data() + seq * GDN_QKV_PER_SEQ,
                     step.k.data() + seq * GDN_QKV_PER_SEQ,
                     step.v.data() + seq * GDN_QKV_PER_SEQ,
                     step.g.data() + seq * GDN_GATE_PER_SEQ,
                     step.beta.data() + seq * GDN_GATE_PER_SEQ,
                     state_reference.data() + seq * GDN_STATE_PER_SEQ,
                     attn_reference.data() + seq * GDN_QKV_PER_SEQ,
                     next_reference.data() + seq * GDN_STATE_PER_SEQ);
    }
    if (!ok) {
        std::fprintf(stderr, "batched gdn %s: compute failed\n", label);
        return false;
    }

    char name[64];
    std::snprintf(name, sizeof(name), "%s attn", label);
    ok = check(name, attn_batched.data(), attn_reference.data(),
               attn_batched.size());
    std::snprintf(name, sizeof(name), "%s state", label);
    ok = check(name, next_batched.data(), next_reference.data(),
               next_batched.size()) && ok;

    // Carry each side's own output so a state-propagation bug compounds
    // across steps instead of being masked by a shared copy.
    state_batched = std::move(next_batched);
    state_reference = std::move(next_reference);
    return ok;
}

bool test_gdn_sequential(ggml_backend_t backend, std::mt19937 & rng,
                         bool inplace_batched) {
    std::vector<float> state(GDN_STATE_PER_SEQ * N_SEQS);
    fill_uniform(rng, -0.5f, 0.5f, state);
    std::vector<float> state_batched = state;
    std::vector<float> state_reference = state;

    bool ok = true;
    for (int step_idx = 0; step_idx < N_STEPS; ++step_idx) {
        char label[32];
        std::snprintf(label, sizeof(label), "%s step%d",
                      inplace_batched ? "inplace" : "packed", step_idx);
        const GdnStep step(rng);
        ok = step_and_compare(backend, label, step, state_batched,
                              state_reference, inplace_batched) && ok;
    }
    return ok;
}

bool test_gdn_active_slots(ggml_backend_t backend, std::mt19937 & rng) {
    // Three physical server slots round up to a four-row graph bucket.
    // Negative and out-of-range rows both use scratch state without
    // touching any physical slab.
    constexpr int physical_slots = 3;
    const std::vector<int32_t> active_slot_ids{2, -1, physical_slots, 0};
    const GdnStep step(rng);
    std::vector<float> initial_state(GDN_STATE_PER_SEQ * physical_slots);
    fill_uniform(rng, -0.5f, 0.5f, initial_state);

    std::vector<float> attn_active(GDN_QKV_PER_SEQ * N_SEQS);
    std::vector<float> state_active(initial_state.size());
    std::vector<float> scratch_active(GDN_STATE_PER_SEQ * N_SEQS);
    bool ok = run_gdn(
        backend, N_SEQS, step.q.data(), step.k.data(), step.v.data(),
        step.g.data(), step.beta.data(), initial_state.data(),
        attn_active.data(), state_active.data(), /*inplace=*/false,
        active_slot_ids.data(), physical_slots, scratch_active.data());
    if (!ok) {
        std::fprintf(stderr, "batched gdn active slots: compute failed\n");
        return false;
    }

    std::vector<float> expected_state = initial_state;
    for (int row = 0; row < N_SEQS; ++row) {
        const int slot = active_slot_ids[row];

        std::vector<float> attn_reference(GDN_QKV_PER_SEQ);
        std::vector<float> state_reference(GDN_STATE_PER_SEQ);
        std::vector<float> zero_state(GDN_STATE_PER_SEQ, 0.0f);
        const float * row_state = slot >= 0 && slot < physical_slots
            ? initial_state.data() + slot * GDN_STATE_PER_SEQ
            : zero_state.data();
        ok = run_gdn(
            backend, 1,
            step.q.data() + row * GDN_QKV_PER_SEQ,
            step.k.data() + row * GDN_QKV_PER_SEQ,
            step.v.data() + row * GDN_QKV_PER_SEQ,
            step.g.data() + row * GDN_GATE_PER_SEQ,
            step.beta.data() + row * GDN_GATE_PER_SEQ,
            row_state,
            attn_reference.data(), state_reference.data()) && ok;

        char label[48];
        std::snprintf(label, sizeof(label), "active row%d slot%d attn",
                      row, slot);
        ok = check(label,
                   attn_active.data() + row * GDN_QKV_PER_SEQ,
                   attn_reference.data(), GDN_QKV_PER_SEQ) && ok;
        if (slot >= 0 && slot < physical_slots) {
            std::copy(state_reference.begin(), state_reference.end(),
                      expected_state.begin() + slot * GDN_STATE_PER_SEQ);
        } else {
            std::snprintf(label, sizeof(label), "active row%d scratch", row);
            ok = check(label,
                       scratch_active.data() + row * GDN_STATE_PER_SEQ,
                       state_reference.data(), GDN_STATE_PER_SEQ) && ok;
        }
    }

    ok = check("active physical state", state_active.data(),
               expected_state.data(), state_active.size()) && ok;
    return ok;
}

bool test_conv(ggml_backend_t backend, std::mt19937 & rng) {
    std::vector<float> weights(CONV_WEIGHT_ELEMS);
    fill_uniform(rng, -0.5f, 0.5f, weights);

    // Per-sequence conv history, carried on the host exactly like the
    // backend's conv_state slots: each step prepends the (d_conv-1)-row
    // history to the new token, convolves, then shifts the token in.
    std::vector<float> history(CONV_HIST_PER_SEQ * N_SEQS);
    fill_uniform(rng, -1.0f, 1.0f, history);

    bool ok = true;
    for (int step_idx = 0; step_idx < N_STEPS; ++step_idx) {
        std::vector<float> token(static_cast<size_t>(CONV_CHANNELS) * N_SEQS);
        fill_uniform(rng, -1.0f, 1.0f, token);

        // sx rows per channel: [h0, h1, h2, new_token]
        std::vector<float> sx(CONV_IN_PER_SEQ * N_SEQS);
        for (int seq = 0; seq < N_SEQS; ++seq) {
            for (int channel = 0; channel < CONV_CHANNELS; ++channel) {
                float * row = sx.data() + seq * CONV_IN_PER_SEQ +
                              static_cast<size_t>(channel) * D_CONV;
                const float * hist = history.data() + seq * CONV_HIST_PER_SEQ +
                                     static_cast<size_t>(channel) * (D_CONV - 1);
                for (int j = 0; j < D_CONV - 1; ++j) row[j] = hist[j];
                row[D_CONV - 1] = token[seq * CONV_CHANNELS + channel];
            }
        }

        std::vector<float> out_batched(CONV_OUT_PER_SEQ * N_SEQS);
        std::vector<float> out_reference(CONV_OUT_PER_SEQ * N_SEQS);
        bool computed =
            run_conv(backend, N_SEQS, sx.data(), weights.data(),
                     out_batched.data());
        for (int seq = 0; computed && seq < N_SEQS; ++seq) {
            computed = run_conv(backend, 1, sx.data() + seq * CONV_IN_PER_SEQ,
                                weights.data(),
                                out_reference.data() + seq * CONV_OUT_PER_SEQ);
        }
        if (!computed) {
            std::fprintf(stderr, "batched gdn conv step%d: compute failed\n",
                         step_idx);
            return false;
        }

        char name[32];
        std::snprintf(name, sizeof(name), "conv step%d", step_idx);
        ok = check(name, out_batched.data(), out_reference.data(),
                   out_batched.size()) && ok;

        // Shift the new token into the history window.
        for (int seq = 0; seq < N_SEQS; ++seq) {
            for (int channel = 0; channel < CONV_CHANNELS; ++channel) {
                float * hist = history.data() + seq * CONV_HIST_PER_SEQ +
                               static_cast<size_t>(channel) * (D_CONV - 1);
                for (int j = 0; j < D_CONV - 2; ++j) hist[j] = hist[j + 1];
                hist[D_CONV - 2] = token[seq * CONV_CHANNELS + channel];
            }
        }
    }
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const bool cpu = argc == 2 && std::strcmp(argv[1], "--cpu") == 0;
    if (argc > 2 || (argc == 2 && !cpu)) {
        std::fprintf(stderr, "usage: %s [--cpu]\n", argv[0]);
        return 2;
    }
    ggml_backend_t backend = cpu
        ? ggml_backend_cpu_init()
        : ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "%s backend unavailable\n", cpu ? "CPU" : "GPU");
        return 1;
    }

    std::mt19937 rng(20260728);
    bool ok = test_gdn_sequential(backend, rng, /*inplace_batched=*/false);
    ok = test_gdn_sequential(backend, rng, /*inplace_batched=*/true) && ok;
    ok = test_gdn_active_slots(backend, rng) && ok;
    ok = test_conv(backend, rng) && ok;
    ok = test_masked_set_rows(backend) && ok;

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
