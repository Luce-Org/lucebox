#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int S = 128;
constexpr int H = 48;
constexpr int KEY_HEADS = 16;
constexpr int T = 6;
constexpr int B = 4;
constexpr int PHYSICAL_SLOTS = 3;
constexpr int CONV_WINDOW = 3;
constexpr int CONV_CHANNELS = 7;
constexpr float FIELD_TOLERANCE = 5.0e-5f;
constexpr float STATE_TOLERANCE = 2.0e-4f;

bool test_raw_gate_protocol() {
    ggml_init_params params{};
    params.mem_size = 128*1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, 1);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    ggml_tensor * gate_ba = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2*H);
    ggml_tensor * result = ggml_gated_delta_net(
        ctx, q, k, v, g, beta, state);

    ggml_gated_delta_net_set_raw_gates(result, gate_ba);
    const int32_t * op_params =
        reinterpret_cast<const int32_t *>(result->op_params);
    const bool ok = result->src[9] == gate_ba &&
        ggml_nelements(result->src[9]) == 2*H && op_params[2] == 0 &&
        op_params[10] == 1;
    if (!ok) {
        std::fprintf(
            stderr,
            "raw gate protocol: src9=%p elements=%lld op2=%d op10=%d\n",
            static_cast<void *>(result->src[9]),
            (long long) ggml_nelements(result->src[9]),
            op_params[2], op_params[10]);
    }
    ggml_free(ctx);
    return ok;
}

size_t qkv_index(int sequence, int token, int head, int value) {
    return (((size_t) sequence*T + token)*H + head)*S + value;
}

size_t key_index(int sequence, int token, int head, int value) {
    return (((size_t) sequence*T + token)*KEY_HEADS +
            head%KEY_HEADS)*S + value;
}

size_t scalar_index(int sequence, int token, int head) {
    return ((size_t) sequence*T + token)*H + head;
}

size_t state_index(int slot, int head, int col, int row) {
    return (((size_t) slot*H + head)*S + col)*S + row;
}

size_t replay_log_index(
        int sequence, int token, int head, int width, int value) {
    return ((((size_t) sequence*T + token)*H + head)*width) + value;
}

size_t conv_input_index(
        int sequence, int channel, int position) {
    return ((size_t) sequence*CONV_CHANNELS + channel)*
        (CONV_WINDOW + T) + position;
}

size_t conv_state_index(int slot, int channel, int position) {
    return ((size_t) slot*CONV_CHANNELS + channel)*CONV_WINDOW + position;
}

std::vector<float> make_conv_input() {
    std::vector<float> values(
        (size_t)(CONV_WINDOW + T)*CONV_CHANNELS*B);
    for (int sequence = 0; sequence < B; ++sequence) {
        for (int channel = 0; channel < CONV_CHANNELS; ++channel) {
            for (int position = 0; position < CONV_WINDOW + T; ++position) {
                values[conv_input_index(sequence, channel, position)] =
                    0.01f*sequence + 0.001f*channel + 0.0001f*position;
            }
        }
    }
    return values;
}

std::vector<float> conv_state_for_slots(
        const std::vector<float> & input,
        const std::vector<int32_t> & slots,
        const std::vector<int32_t> & prefixes,
        int physical_slots) {
    std::vector<float> state(
        (size_t)CONV_WINDOW*CONV_CHANNELS*physical_slots, -1.0f);
    for (int sequence = 0; sequence < B; ++sequence) {
        const int slot = slots[(size_t)sequence];
        if (slot < 0 || slot >= physical_slots) continue;
        const int prefix = prefixes[(size_t)sequence];
        for (int channel = 0; channel < CONV_CHANNELS; ++channel) {
            for (int k = 0; k < CONV_WINDOW; ++k) {
                state[conv_state_index(slot, channel, k)] =
                    input[conv_input_index(
                        sequence, channel, prefix + k)];
            }
        }
    }
    return state;
}

float sigmoid(float x) {
    return 1.0f/(1.0f + std::exp(-x));
}

float softplus(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

bool compare_vectors(
        const char * label,
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        float tolerance) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: size mismatch %zu != %zu\n", label,
                     actual.size(), expected.size());
        return false;
    }
    float max_error = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            std::fprintf(stderr,
                         "%s: non-finite value at %zu (actual %.9g expected %.9g)\n",
                         label, i, actual[i], expected[i]);
            return false;
        }
        const float error = std::fabs(actual[i] - expected[i]);
        if (error > max_error) {
            max_error = error;
            worst = i;
        }
    }
    if (max_error > tolerance || !std::isfinite(max_error)) {
        std::fprintf(stderr,
                     "%s: max error %.9g at %zu (actual %.9g expected %.9g, tolerance %.9g)\n",
                     label, max_error, worst, actual[worst], expected[worst],
                     tolerance);
        return false;
    }
    return true;
}

struct Inputs {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> g;
    std::vector<float> beta;
    std::vector<float> state;
    std::vector<float> dt_bias;
    std::vector<float> gate_A;
};

Inputs make_inputs(bool kda, bool raw_gates) {
    std::mt19937 rng(20260819 + 17*kda + 31*raw_gates);
    std::uniform_real_distribution<float> small(-0.25f, 0.25f);
    std::uniform_real_distribution<float> state_dist(-0.06f, 0.06f);
    std::uniform_real_distribution<float> gate_dist(0.82f, 0.98f);
    std::uniform_real_distribution<float> beta_dist(0.15f, 0.85f);
    std::uniform_real_distribution<float> raw_dist(-1.5f, 1.5f);

    Inputs in;
    const size_t qkv_elements = (size_t) S*H*T*B;
    const size_t qk_elements = (size_t) S*KEY_HEADS*T*B;
    in.q.resize(qk_elements);
    in.k.resize(qk_elements);
    in.v.resize(qkv_elements);
    in.g.resize((size_t) (kda ? S : 1)*H*T*B);
    in.beta.resize((size_t) H*T*B);
    in.state.resize((size_t) S*S*H*B);
    in.dt_bias.resize(H);
    in.gate_A.resize(H);

    for (float & value : in.q) value = small(rng);
    for (float & value : in.v) value = small(rng);
    for (float & value : in.state) value = state_dist(rng);

    for (int sequence = 0; sequence < B; ++sequence) {
        for (int token = 0; token < T; ++token) {
            for (int head = 0; head < KEY_HEADS; ++head) {
                float norm2 = 0.0f;
                for (int row = 0; row < S; ++row) {
                    const float value = small(rng);
                    in.k[key_index(sequence, token, head, row)] = value;
                    norm2 += value*value;
                }
                const float inverse_norm = 1.0f/std::sqrt(norm2);
                for (int row = 0; row < S; ++row) {
                    in.k[key_index(sequence, token, head, row)] *= inverse_norm;
                }
            }
        }
    }

    if (raw_gates) {
        for (float & value : in.g) value = raw_dist(rng);
        for (float & value : in.beta) value = raw_dist(rng);
        for (int head = 0; head < H; ++head) {
            in.dt_bias[head] = -0.35f + 0.12f*head;
            in.gate_A[head] = -0.12f - 0.07f*head;
        }
    } else {
        for (float & value : in.g) value = std::log(gate_dist(rng));
        for (float & value : in.beta) value = beta_dist(rng);
    }
    return in;
}

float resolved_gate(
        const Inputs & in, bool kda, bool raw_gates,
        int sequence, int token, int head, int row) {
    if (kda) {
        return std::exp(in.g[qkv_index(sequence, token, head, row)]);
    }
    const float raw_or_log = in.g[scalar_index(sequence, token, head)];
    if (!raw_gates) return std::exp(raw_or_log);
    return std::exp(
        softplus(raw_or_log + in.dt_bias[head])*in.gate_A[head]);
}

float resolved_beta(
        const Inputs & in, bool raw_gates,
        int sequence, int token, int head) {
    const float value = in.beta[scalar_index(sequence, token, head)];
    return raw_gates ? sigmoid(value) : value;
}

std::vector<float> ordinary_recurrence(
        const Inputs & in, bool kda, bool raw_gates,
        int accepted_prefix) {
    std::vector<float> state = in.state;
    for (int sequence = 0; sequence < B; ++sequence) {
        for (int token = 0; token < accepted_prefix; ++token) {
            for (int head = 0; head < H; ++head) {
                const float beta = resolved_beta(
                    in, raw_gates, sequence, token, head);
                for (int col = 0; col < S; ++col) {
                    float projection = 0.0f;
                    for (int row = 0; row < S; ++row) {
                        const float gate = resolved_gate(
                            in, kda, raw_gates,
                            sequence, token, head, row);
                        const float state_value =
                            state[state_index(sequence, head, col, row)];
                        const float key =
                            in.k[key_index(sequence, token, head, row)];
                        projection += (kda ? gate : 1.0f)*state_value*key;
                    }
                    const float scalar_gate = resolved_gate(
                        in, kda, raw_gates,
                        sequence, token, head, 0);
                    const float delta =
                        (in.v[qkv_index(sequence, token, head, col)] -
                         (kda ? projection : scalar_gate*projection))*beta;
                    for (int row = 0; row < S; ++row) {
                        const float gate = resolved_gate(
                            in, kda, raw_gates,
                            sequence, token, head, row);
                        const float key =
                            in.k[key_index(sequence, token, head, row)];
                        float & state_value =
                            state[state_index(sequence, head, col, row)];
                        state_value = std::fma(key, delta, gate*state_value);
                    }
                }
            }
        }
    }
    return state;
}

std::vector<float> expected_replay_log(
        const Inputs & in, bool kda, bool raw_gates) {
    const int gate_values = kda ? S : 1;
    const int width = gate_values + 2*S;
    std::vector<float> replay_log((size_t) width*H*T*B);
    std::vector<float> state = in.state;
    for (int sequence = 0; sequence < B; ++sequence) {
        for (int token = 0; token < T; ++token) {
            for (int head = 0; head < H; ++head) {
                for (int row = 0; row < gate_values; ++row) {
                    replay_log[replay_log_index(sequence, token, head, width, row)] =
                        resolved_gate(in, kda, raw_gates,
                                      sequence, token, head, row);
                }
                for (int row = 0; row < S; ++row) {
                    replay_log[replay_log_index(
                        sequence, token, head, width,
                        gate_values + row)] =
                            in.k[key_index(sequence, token, head, row)];
                }
                const float beta = resolved_beta(
                    in, raw_gates, sequence, token, head);
                for (int col = 0; col < S; ++col) {
                    float projection = 0.0f;
                    for (int row = 0; row < S; ++row) {
                        const float gate = resolved_gate(
                            in, kda, raw_gates,
                            sequence, token, head, row);
                        projection += (kda ? gate : 1.0f)*
                            state[state_index(sequence, head, col, row)]*
                            in.k[key_index(sequence, token, head, row)];
                    }
                    const float scalar_gate = resolved_gate(
                        in, kda, raw_gates,
                        sequence, token, head, 0);
                    const float delta =
                        (in.v[qkv_index(sequence, token, head, col)] -
                         (kda ? projection : scalar_gate*projection))*beta;
                    replay_log[replay_log_index(
                        sequence, token, head, width,
                        gate_values + S + col)] = delta;
                    for (int row = 0; row < S; ++row) {
                        const float gate = resolved_gate(
                            in, kda, raw_gates,
                            sequence, token, head, row);
                        float & value =
                            state[state_index(sequence, head, col, row)];
                        value = std::fma(
                            in.k[key_index(sequence, token, head, row)],
                            delta, gate*value);
                    }
                }
            }
        }
    }
    return replay_log;
}

struct CaseTensors {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * replay_log = nullptr;
    ggml_tensor * identity_state = nullptr;
    ggml_tensor * mapped_state = nullptr;
    ggml_tensor * conv_input = nullptr;
    ggml_tensor * identity_conv_state = nullptr;
    ggml_tensor * mapped_conv_state = nullptr;
    ggml_tensor * accepted = nullptr;
    ggml_tensor * slots = nullptr;
};

bool commit_many_one_layer(
        const ggml_tensor * replay_log,
        ggml_tensor * state,
        const ggml_tensor * conv_input,
        ggml_tensor * conv_state,
        const ggml_tensor * accepted,
        const ggml_tensor * slots) {
    const ggml_tensor * replay_logs[] = {replay_log};
    ggml_tensor * states[] = {state};
    const ggml_tensor * conv_inputs[] = {conv_input};
    ggml_tensor * conv_states[] = {conv_state};
    return ggml_backend_cuda_gdn_replay_log_commit_many(
        replay_logs, states, conv_inputs, conv_states, 1, accepted, slots);
}

void destroy(CaseTensors & tensors) {
    if (tensors.buffer) ggml_backend_buffer_free(tensors.buffer);
    if (tensors.ctx) ggml_free(tensors.ctx);
    tensors = {};
}

bool run_case(ggml_backend_t backend, bool kda, bool raw_gates,
              const char * dispatch) {
    const char * kind = raw_gates ? "scalar-raw" : kda ? "kda" : "scalar";
    char name[48];
    std::snprintf(name, sizeof(name), "%s-%s", kind, dispatch);
    const int gate_values = kda ? S : 1;
    const int width = gate_values + 2*S;
    const Inputs inputs = make_inputs(kda, raw_gates);

    ggml_init_params params{};
    params.mem_size = 8*1024*1024;
    params.no_alloc = true;
    CaseTensors tensors;
    tensors.ctx = ggml_init(params);
    if (!tensors.ctx) return false;

    ggml_tensor * q = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, S, KEY_HEADS, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, S, KEY_HEADS, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, kda ? S : 1, H, T, B);
    ggml_tensor * beta = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * capture_state = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, S, S, H, B);
    tensors.identity_state = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, S, S, H, B);
    tensors.mapped_state = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, S, S, H, PHYSICAL_SLOTS);
    tensors.conv_input = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, CONV_WINDOW + T,
        CONV_CHANNELS, B, 1);
    tensors.identity_conv_state = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, CONV_WINDOW,
        CONV_CHANNELS, B, 1);
    tensors.mapped_conv_state = ggml_new_tensor_4d(
        tensors.ctx, GGML_TYPE_F32, CONV_WINDOW,
        CONV_CHANNELS, PHYSICAL_SLOTS, 1);
    tensors.accepted = ggml_new_tensor_1d(
        tensors.ctx, GGML_TYPE_I32, B);
    tensors.slots = ggml_new_tensor_1d(
        tensors.ctx, GGML_TYPE_I32, B);
    ggml_tensor * gate_ba = nullptr;
    if (raw_gates) {
        gate_ba = ggml_new_tensor_1d(tensors.ctx, GGML_TYPE_F32, 2*H);
    }

    ggml_tensor * result = ggml_gated_delta_net(
        tensors.ctx, q, k, v, g, beta, capture_state);
    ggml_gated_delta_net_set_skip_intermediate(result, true);
    if (raw_gates) {
        ggml_gated_delta_net_set_raw_gates(result, gate_ba);
    }
    tensors.replay_log =
        ggml_gated_delta_net_capture_replay_log(tensors.ctx, result);
    ggml_set_output(tensors.replay_log);
    ggml_cgraph * graph = ggml_new_graph(tensors.ctx);
    ggml_build_forward_expand(graph, tensors.replay_log);

    tensors.buffer = ggml_backend_alloc_ctx_tensors(tensors.ctx, backend);
    if (!tensors.buffer) {
        std::fprintf(stderr, "%s: GPU tensor allocation failed\n", name);
        destroy(tensors);
        return false;
    }
    auto upload_f32 = [](ggml_tensor * tensor,
                         const std::vector<float> & values) {
        ggml_backend_tensor_set(tensor, values.data(), 0,
                                values.size()*sizeof(float));
    };
    upload_f32(q, inputs.q);
    upload_f32(k, inputs.k);
    upload_f32(v, inputs.v);
    upload_f32(g, inputs.g);
    upload_f32(beta, inputs.beta);
    upload_f32(capture_state, inputs.state);
    const std::vector<float> conv_input = make_conv_input();
    upload_f32(tensors.conv_input, conv_input);
    if (raw_gates) {
        std::vector<float> packed_gates;
        packed_gates.reserve(2*H);
        packed_gates.insert(
            packed_gates.end(), inputs.dt_bias.begin(), inputs.dt_bias.end());
        packed_gates.insert(
            packed_gates.end(), inputs.gate_A.begin(), inputs.gate_A.end());
        upload_f32(gate_ba, packed_gates);
    }

    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual_replay_log((size_t) width*H*T*B);
    if (ok) {
        ggml_backend_tensor_get(
            tensors.replay_log, actual_replay_log.data(), 0,
            actual_replay_log.size()*sizeof(float));
        ok = compare_vectors(
            name, actual_replay_log,
            expected_replay_log(inputs, kda, raw_gates), FIELD_TOLERANCE);
    }

    const std::vector<int32_t> identity_slots{0, 1, 2, 3};
    std::vector<int32_t> accepted(B);
    std::vector<float> actual_state(inputs.state.size());
    std::vector<float> actual_conv(
        (size_t)CONV_WINDOW*CONV_CHANNELS*B);
    const std::vector<int32_t> zero_prefixes(B, 0);
    const std::vector<float> identity_conv_base = conv_state_for_slots(
        conv_input, identity_slots, zero_prefixes, B);
    for (int prefix = 0; ok && prefix <= T; ++prefix) {
        std::fill(accepted.begin(), accepted.end(), prefix);
        upload_f32(tensors.identity_state, inputs.state);
        upload_f32(tensors.identity_conv_state, identity_conv_base);
        ggml_backend_tensor_set(tensors.accepted, accepted.data(), 0,
                                accepted.size()*sizeof(accepted[0]));
        ggml_backend_tensor_set(tensors.slots, identity_slots.data(), 0,
                                identity_slots.size()*sizeof(identity_slots[0]));
        ok = commit_many_one_layer(
            tensors.replay_log, tensors.identity_state,
            tensors.conv_input, tensors.identity_conv_state,
            tensors.accepted, tensors.slots);
        if (ok) {
            ggml_backend_tensor_get(
                tensors.identity_state, actual_state.data(), 0,
                actual_state.size()*sizeof(float));
            char label[64];
            std::snprintf(label, sizeof(label), "%s prefix %d", name, prefix);
            ok = compare_vectors(
                label, actual_state,
                ordinary_recurrence(inputs, kda, raw_gates, prefix),
                STATE_TOLERANCE);
            ggml_backend_tensor_get(
                tensors.identity_conv_state, actual_conv.data(), 0,
                actual_conv.size()*sizeof(float));
            ok = ok && compare_vectors(
                "convolution prefix", actual_conv,
                conv_state_for_slots(
                    conv_input, identity_slots, accepted, B),
                0.0f);
        }
    }

    const std::vector<int32_t> mapped_slots{2, -1, 0, 1};
    const std::vector<int32_t> mapped_prefixes{T, T, 2, 4};
    std::vector<float> mapped_base((size_t) S*S*H*PHYSICAL_SLOTS);
    for (int sequence : {0, 2, 3}) {
        const int slot = mapped_slots[(size_t) sequence];
        for (int head = 0; head < H; ++head) {
            for (int col = 0; col < S; ++col) {
                for (int row = 0; row < S; ++row) {
                    mapped_base[state_index(slot, head, col, row)] =
                        inputs.state[state_index(sequence, head, col, row)];
                }
            }
        }
    }
    std::vector<float> mapped_expected = mapped_base;
    for (int sequence : {0, 2, 3}) {
        const int slot = mapped_slots[(size_t) sequence];
        const std::vector<float> lane_state = ordinary_recurrence(
            inputs, kda, raw_gates, mapped_prefixes[(size_t) sequence]);
        for (int head = 0; head < H; ++head) {
            for (int col = 0; col < S; ++col) {
                for (int row = 0; row < S; ++row) {
                    mapped_expected[state_index(slot, head, col, row)] =
                        lane_state[state_index(sequence, head, col, row)];
                }
            }
        }
    }
    std::vector<float> mapped_actual(mapped_base.size());
    const std::vector<float> mapped_conv_base = conv_state_for_slots(
        conv_input, mapped_slots, zero_prefixes, PHYSICAL_SLOTS);
    const std::vector<float> mapped_conv_expected = conv_state_for_slots(
        conv_input, mapped_slots, mapped_prefixes, PHYSICAL_SLOTS);
    std::vector<float> mapped_conv_actual(mapped_conv_base.size());
    if (ok) {
        upload_f32(tensors.mapped_state, mapped_base);
        upload_f32(tensors.mapped_conv_state, mapped_conv_base);
        ggml_backend_tensor_set(tensors.accepted, mapped_prefixes.data(), 0,
                                mapped_prefixes.size()*sizeof(mapped_prefixes[0]));
        ggml_backend_tensor_set(tensors.slots, mapped_slots.data(), 0,
                                mapped_slots.size()*sizeof(mapped_slots[0]));
        ok = commit_many_one_layer(
            tensors.replay_log, tensors.mapped_state,
            tensors.conv_input, tensors.mapped_conv_state,
            tensors.accepted, tensors.slots);
        if (ok) {
            ggml_backend_tensor_get(
                tensors.mapped_state, mapped_actual.data(), 0,
                mapped_actual.size()*sizeof(float));
            ok = compare_vectors(
                "permuted/padded slots", mapped_actual, mapped_expected,
                STATE_TOLERANCE);
            ggml_backend_tensor_get(
                tensors.mapped_conv_state, mapped_conv_actual.data(), 0,
                mapped_conv_actual.size()*sizeof(float));
            ok = ok && compare_vectors(
                "permuted/padded convolution", mapped_conv_actual,
                mapped_conv_expected, 0.0f);
        }
    }
    if (ok) {
        const std::vector<float> unchanged = inputs.state;
        std::vector<int32_t> invalid_prefix(B, 1);
        invalid_prefix[0] = T + 1;
        upload_f32(tensors.identity_state, unchanged);
        upload_f32(tensors.identity_conv_state, identity_conv_base);
        ggml_backend_tensor_set(tensors.accepted, invalid_prefix.data(), 0,
                                invalid_prefix.size()*sizeof(invalid_prefix[0]));
        ggml_backend_tensor_set(tensors.slots, identity_slots.data(), 0,
                                identity_slots.size()*sizeof(identity_slots[0]));
        ok = !commit_many_one_layer(
            tensors.replay_log, tensors.identity_state,
            tensors.conv_input, tensors.identity_conv_state,
            tensors.accepted, tensors.slots);
        const std::vector<int32_t> out_of_range_slots{0, 99, 2, 3};
        accepted.assign(B, 1);
        ggml_backend_tensor_set(tensors.accepted, accepted.data(), 0,
                                accepted.size()*sizeof(accepted[0]));
        ggml_backend_tensor_set(tensors.slots, out_of_range_slots.data(), 0,
                                out_of_range_slots.size()*sizeof(out_of_range_slots[0]));
        ok = ok && !commit_many_one_layer(
            tensors.replay_log, tensors.identity_state,
            tensors.conv_input, tensors.identity_conv_state,
            tensors.accepted, tensors.slots);
        const std::vector<int32_t> duplicate_slots{0, 0, 2, 3};
        ggml_backend_tensor_set(tensors.slots, duplicate_slots.data(), 0,
                                duplicate_slots.size()*sizeof(duplicate_slots[0]));
        ok = ok && !commit_many_one_layer(
            tensors.replay_log, tensors.identity_state,
            tensors.conv_input, tensors.identity_conv_state,
            tensors.accepted, tensors.slots);
        ggml_backend_tensor_get(
            tensors.identity_state, actual_state.data(), 0,
            actual_state.size()*sizeof(float));
        ok = ok && compare_vectors(
            "transactional validation", actual_state, unchanged, 0.0f);
        ggml_backend_tensor_get(
            tensors.identity_conv_state, actual_conv.data(), 0,
            actual_conv.size()*sizeof(float));
        ok = ok && compare_vectors(
            "transactional convolution validation", actual_conv,
            identity_conv_base, 0.0f);
    }

    std::printf("gdn replay log %-10s: %s\n", name,
                ok ? "PASS" : "FAIL");
    destroy(tensors);
    return ok;
}


bool run_grouped_chain_case(ggml_backend_t backend) {
    const Inputs inputs = make_inputs(/*kda=*/false, /*raw_gates=*/false);

    ggml_init_params params{};
    params.mem_size = 8*1024*1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, S, KEY_HEADS, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, S, KEY_HEADS, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * beta = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * base_state = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, S, S, H, B);
    ggml_tensor * committed_state = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, S, S, H, B);
    ggml_tensor * accepted = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, B);
    ggml_tensor * slots = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, B);
    ggml_tensor * conv_input = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, CONV_WINDOW + T, CONV_CHANNELS, B, 1);
    ggml_tensor * conv_state = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, CONV_WINDOW, CONV_CHANNELS, B, 1);

    ggml_tensor * result = ggml_gated_delta_net(
        ctx, q, k, v, g, beta, base_state);
    ggml_tensor * replay_log =
        ggml_gated_delta_net_capture_replay_log(ctx, result);
    ggml_set_output(replay_log);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, replay_log);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    auto upload_f32 = [](ggml_tensor * tensor,
                         const std::vector<float> & values) {
        ggml_backend_tensor_set(
            tensor, values.data(), 0, values.size()*sizeof(float));
    };
    upload_f32(q, inputs.q);
    upload_f32(k, inputs.k);
    upload_f32(v, inputs.v);
    upload_f32(g, inputs.g);
    upload_f32(beta, inputs.beta);
    upload_f32(base_state, inputs.state);
    const std::vector<float> conv_values = make_conv_input();
    upload_f32(conv_input, conv_values);

    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    const std::vector<int32_t> prefixes{T, 1, T, 1};
    const std::vector<int32_t> identity_slots{0, 1, 2, 3};
    const std::vector<int32_t> zero_prefixes(B, 0);
    upload_f32(committed_state, inputs.state);
    upload_f32(
        conv_state,
        conv_state_for_slots(
            conv_values, identity_slots, zero_prefixes, B));
    ggml_backend_tensor_set(
        accepted, prefixes.data(), 0, prefixes.size()*sizeof(prefixes[0]));
    ggml_backend_tensor_set(
        slots, identity_slots.data(), 0,
        identity_slots.size()*sizeof(identity_slots[0]));
    ok = ok && commit_many_one_layer(
        replay_log, committed_state, conv_input, conv_state, accepted, slots);

    std::vector<float> expected = inputs.state;
    const size_t slot_elements = (size_t) S*S*H;
    for (int sequence = 0; sequence < B; ++sequence) {
        const std::vector<float> lane = ordinary_recurrence(
            inputs, /*kda=*/false, /*raw_gates=*/false,
            prefixes[(size_t) sequence]);
        const size_t offset = (size_t) sequence*slot_elements;
        std::copy_n(lane.begin() + offset, slot_elements,
                    expected.begin() + offset);
    }
    std::vector<float> actual(expected.size());
    if (ok) {
        ggml_backend_tensor_get(
            committed_state, actual.data(), 0,
            actual.size()*sizeof(float));
        ok = compare_vectors(
            "grouped chain commit", actual, expected,
            STATE_TOLERANCE);
        std::vector<float> actual_conv(
            (size_t)CONV_WINDOW*CONV_CHANNELS*B);
        ggml_backend_tensor_get(
            conv_state, actual_conv.data(), 0,
            actual_conv.size()*sizeof(float));
        ok = ok && compare_vectors(
            "grouped chain convolution commit", actual_conv,
            conv_state_for_slots(
                conv_values, identity_slots, prefixes, B),
            0.0f);
    }

    std::printf("gdn grouped chain replay log        : %s\n",
                ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool test_replay_log_capture_owns_storage(ggml_backend_t backend) {
    constexpr int state_size = 16;
    constexpr int heads = 1;
    constexpr int tokens = 8;
    constexpr int sequences = 4;

    ggml_init_params params{};
    params.mem_size = 512*1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, state_size, heads, tokens, sequences);
    ggml_tensor * k = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, state_size, heads, tokens, sequences);
    ggml_tensor * v = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, state_size, heads, tokens, sequences);
    ggml_tensor * g = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, 1, heads, tokens, sequences);
    ggml_tensor * beta = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, 1, heads, tokens, sequences);
    ggml_tensor * state = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, state_size, state_size, heads, sequences);
    ggml_tensor * result = ggml_gated_delta_net(
        ctx, q, k, v, g, beta, state);
    ggml_tensor * replay_log =
        ggml_gated_delta_net_capture_replay_log(ctx, result);
    ggml_set_output(replay_log);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_build_forward_expand(graph, replay_log);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = alloc && ggml_gallocr_alloc_graph(alloc, graph);
    const bool ok = allocated && replay_log->buffer && replay_log->data &&
                    replay_log->view_src == nullptr;
    std::printf("gdn replay log owned C4/W8         : %s\n",
                ok ? "PASS" : "FAIL");
    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

bool test_tree_commit_preflight_is_non_mutating(ggml_backend_t backend) {
    constexpr int state_size = 16;
    constexpr int heads = 1;
    constexpr int tokens = 2;
    constexpr int sequences = 1;
    constexpr int state_slots = 1;
    constexpr int conv_window = 2;
    constexpr int conv_channels = 3;
    constexpr int feature_rows_count = tokens * sequences;

    ggml_init_params params{};
    params.mem_size = 256*1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * cache =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 8, 1, 1);
    ggml_tensor * commit_rows =
        ggml_new_tensor_2d(ctx, GGML_TYPE_I64, tokens, sequences);
    ggml_tensor * active_slots =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequences);
    ggml_tensor * feature_source =
        ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 4, feature_rows_count);
    ggml_tensor * feature_destination =
        ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 4, 4);
    ggml_tensor * feature_rows =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, feature_rows_count);
    ggml_tensor * replay_log = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, 2*state_size + 1,
        heads, tokens, sequences);
    ggml_tensor * state = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, state_size, state_size,
        heads, state_slots);
    ggml_tensor * conv_input = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, conv_window + tokens,
        conv_channels, sequences, 1);
    ggml_tensor * conv_state = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, conv_window,
        conv_channels, state_slots, 1);
    ggml_tensor * accepted =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequences);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    std::vector<float> cache_before(
        (size_t) ggml_nelements(cache), 1.25f);
    std::vector<uint16_t> feature_before(
        (size_t) ggml_nelements(feature_destination), 0x3f80);
    // Model an already-durable direct-row feature. Tree promotion must leave
    // it alone while copying only the accepted tree suffix into row zero.
    std::fill(feature_before.begin() + 4, feature_before.begin() + 8, 0x4100);
    std::vector<float> state_before(
        (size_t) ggml_nelements(state), -2.5f);
    std::vector<float> conv_before(
        (size_t) ggml_nelements(conv_state), 3.75f);
    std::vector<float> zeros(
        (size_t) std::max(
            ggml_nelements(replay_log), ggml_nelements(conv_input)),
        0.0f);
    std::vector<uint16_t> feature_source_values(
        (size_t) ggml_nelements(feature_source), 0x4000);
    std::fill(
        feature_source_values.begin(),
        feature_source_values.begin() + 4,
        0x4200);
    const int64_t destination_rows[] = {0, -1};
    const int32_t active_slot = 0;
    const int32_t feature_row_map[] = {0, -1};
    const int32_t valid_accepted = 1;
    const int32_t invalid_accepted = tokens + 1;

    ggml_backend_tensor_set(
        cache, cache_before.data(), 0,
        cache_before.size()*sizeof(float));
    ggml_backend_tensor_set(
        feature_destination, feature_before.data(), 0,
        feature_before.size()*sizeof(uint16_t));
    ggml_backend_tensor_set(
        feature_source, feature_source_values.data(), 0,
        feature_source_values.size()*sizeof(uint16_t));
    ggml_backend_tensor_set(
        state, state_before.data(), 0,
        state_before.size()*sizeof(float));
    ggml_backend_tensor_set(
        conv_state, conv_before.data(), 0,
        conv_before.size()*sizeof(float));
    ggml_backend_tensor_set(
        replay_log, zeros.data(), 0,
        (size_t) ggml_nelements(replay_log)*sizeof(float));
    ggml_backend_tensor_set(
        conv_input, zeros.data(), 0,
        (size_t) ggml_nelements(conv_input)*sizeof(float));
    ggml_backend_tensor_set(
        commit_rows, destination_rows, 0, sizeof(destination_rows));
    ggml_backend_tensor_set(
        active_slots, &active_slot, 0, sizeof(active_slot));
    ggml_backend_tensor_set(
        feature_rows, feature_row_map, 0, sizeof(feature_row_map));
    ggml_backend_tensor_set(
        accepted, &valid_accepted, 0, sizeof(valid_accepted));

    ggml_tensor * caches[] = {cache};
    const ggml_tensor * replay_logs[] = {replay_log};
    ggml_tensor * states[] = {state};
    const ggml_tensor * conv_inputs[] = {conv_input};
    ggml_tensor * conv_states[] = {conv_state};

    const bool committed = ggml_backend_cuda_tree_commit_transaction(
        caches, 1, feature_source, feature_destination, feature_rows,
        replay_logs, states, conv_inputs, conv_states, 1,
        commit_rows, accepted, active_slots, 4, 2);
    std::vector<uint16_t> committed_feature(feature_before.size());
    std::vector<float> committed_state(state_before.size());
    std::vector<float> committed_conv(conv_before.size());
    ggml_backend_tensor_get(
        feature_destination, committed_feature.data(), 0,
        committed_feature.size()*sizeof(uint16_t));
    ggml_backend_tensor_get(
        state, committed_state.data(), 0,
        committed_state.size()*sizeof(float));
    ggml_backend_tensor_get(
        conv_state, committed_conv.data(), 0,
        committed_conv.size()*sizeof(float));
    const bool success_path_ok = committed &&
        std::all_of(
            committed_feature.begin(), committed_feature.begin() + 4,
            [](uint16_t value) { return value == 0x4200; }) &&
        std::all_of(
            committed_feature.begin() + 4, committed_feature.begin() + 8,
            [](uint16_t value) { return value == 0x4100; }) &&
        std::all_of(
            committed_feature.begin() + 8, committed_feature.end(),
            [](uint16_t value) { return value == 0x3f80; }) &&
        std::all_of(
            committed_state.begin(), committed_state.end(),
            [](float value) { return value == 0.0f; }) &&
        std::all_of(
            committed_conv.begin(), committed_conv.end(),
            [](float value) { return value == 0.0f; });

    ggml_backend_tensor_set(
        cache, cache_before.data(), 0,
        cache_before.size()*sizeof(float));
    ggml_backend_tensor_set(
        feature_destination, feature_before.data(), 0,
        feature_before.size()*sizeof(uint16_t));
    ggml_backend_tensor_set(
        state, state_before.data(), 0,
        state_before.size()*sizeof(float));
    ggml_backend_tensor_set(
        conv_state, conv_before.data(), 0,
        conv_before.size()*sizeof(float));
    ggml_backend_tensor_set(
        accepted, &invalid_accepted, 0, sizeof(invalid_accepted));
    const bool rejected = !ggml_backend_cuda_tree_commit_transaction(
        caches, 1, feature_source, feature_destination, feature_rows,
        replay_logs, states, conv_inputs, conv_states, 1,
        commit_rows, accepted, active_slots, 4, 2);

    std::vector<float> cache_after(cache_before.size());
    std::vector<uint16_t> feature_after(feature_before.size());
    std::vector<float> state_after(state_before.size());
    std::vector<float> conv_after(conv_before.size());
    ggml_backend_tensor_get(
        cache, cache_after.data(), 0, cache_after.size()*sizeof(float));
    ggml_backend_tensor_get(
        feature_destination, feature_after.data(), 0,
        feature_after.size()*sizeof(uint16_t));
    ggml_backend_tensor_get(
        state, state_after.data(), 0, state_after.size()*sizeof(float));
    ggml_backend_tensor_get(
        conv_state, conv_after.data(), 0, conv_after.size()*sizeof(float));

    const bool ok = success_path_ok && rejected &&
        cache_after == cache_before &&
        feature_after == feature_before &&
        state_after == state_before &&
        conv_after == conv_before;
    std::printf("gdn tree commit preflight        : %s\n",
                ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}
} // namespace

int main(int argc, char ** argv) {
    if (!test_raw_gate_protocol()) return 1;
    setenv("DFLASH_GDN_FORCE_GROUPED_COLS", "1", 1);
    unsetenv("DFLASH_GDN_NO_GROUPED_COLS");
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "GPU backend unavailable\n");
        return 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--preflight-only") == 0) {
        const bool ok =
            test_tree_commit_preflight_is_non_mutating(backend);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    }
    const bool gallocr_ok = test_replay_log_capture_owns_storage(backend);
    if (argc == 2 && std::strcmp(argv[1], "--gallocr-only") == 0) {
        ggml_backend_free(backend);
        return gallocr_ok ? 0 : 1;
    }
    if (!gallocr_ok) {
        ggml_backend_free(backend);
        return 1;
    }
    bool ok = run_case(backend, false, false, "grouped");
    ok = run_case(backend, false, true, "grouped") && ok;
    ok = run_case(backend, true, false, "generic") && ok;
    ok = run_grouped_chain_case(backend) && ok;
    unsetenv("DFLASH_GDN_FORCE_GROUPED_COLS");
    setenv("DFLASH_GDN_NO_GROUPED_COLS", "1", 1);
    ok = run_case(backend, false, false, "scalar") && ok;
    ok = run_case(backend, false, true, "scalar") && ok;
    unsetenv("DFLASH_GDN_NO_GROUPED_COLS");
    ok = test_tree_commit_preflight_is_non_mutating(backend) && ok;
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
