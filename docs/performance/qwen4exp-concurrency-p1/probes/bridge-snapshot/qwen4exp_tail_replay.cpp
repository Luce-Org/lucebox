// This one-shot harness deliberately includes the model graph implementation:
// it reuses the exact private HC/GDN/MoE/PLE builders without adding a
// production runtime mode. The public forward is renamed to avoid colliding
// with dflash_common's production copy.
#define qwen4exp_forward qwen4exp_forward_tail_replay_unused
#define qwen4exp_forward_batched qwen4exp_forward_batched_tail_replay_unused
#include "../src/qwen4exp/qwen4exp_graph.cpp"
#undef qwen4exp_forward
#undef qwen4exp_forward_batched

#include "qwen4exp_tail_replay.h"

#include <array>
#include <fstream>
#include <limits>

namespace luce::common {
namespace {

constexpr int kBase = 992;
constexpr int kReplay = 100;
constexpr std::array<int, 12> kStarts = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};
constexpr std::array<int, 12> kFullLayers = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47};
constexpr std::array<int, 36> kLinearLayers = {
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
    24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46};

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool read_f32(const std::string & path, size_t count, std::vector<float> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(count * sizeof(float))) return false;
    out.resize(count);
    in.seekg(0);
    return static_cast<bool>(in.read(reinterpret_cast<char *>(out.data()),
                                     static_cast<std::streamsize>(count * sizeof(float))));
}

bool boundary_tail(const Qwen4ExpWeights & w, int group, std::vector<float> & tail,
                   std::string & error) {
    const size_t plane = static_cast<size_t>(w.n_embd) * w.n_hc;
    const int channels = group == 0 ? 1 : 2;
    const int layer = group == 0 ? 0 : 4 * group - 1;
    char path[96];
    if (group == 0) std::snprintf(path, sizeof path, "/tmp/our_L00.emb.bin");
    else std::snprintf(path, sizeof path, "/tmp/our_L%02d.res.bin", layer);
    std::vector<float> full;
    if (!read_f32(path, plane * kBase * channels, full))
        return fail(error, std::string("missing or malformed native boundary: ") + path);
    tail.resize(plane * kReplay * channels);
    for (int channel = 0; channel < channels; ++channel) {
        const float * source = full.data() + plane * (channel * kBase + kBase - kReplay);
        std::copy(source, source + plane * kReplay,
                  tail.begin() + static_cast<size_t>(channel) * plane * kReplay);
    }
    return true;
}

bool ple_tail(const Qwen4ExpWeights & w, const int32_t * tokens,
              std::vector<float> & data, std::string & error) {
    if (!w.ple_reader.available() || w.ple_ngram_size != 3)
        return fail(error, "unexpected PLE contract");
    const int64_t heads = w.ple_n_heads;
    std::vector<int32_t> rows(static_cast<size_t>(heads) * kReplay);
    for (int64_t i = 0; i < kReplay; ++i) {
        const int64_t absolute = kBase - kReplay + i;
        std::array<uint64_t, 3> context = {
            static_cast<uint64_t>(tokens[absolute]),
            static_cast<uint64_t>(tokens[absolute - 1]),
            static_cast<uint64_t>(tokens[absolute - 2])};
        bool cut = false;
        for (int64_t s = 1; s < 3; ++s) {
            if (cut || static_cast<int32_t>(context[s]) == w.ple_eos_token_id) cut = true;
            if (cut) context[s] = static_cast<uint64_t>(w.ple_eos_token_id);
        }
        for (int64_t n = 2; n <= 3; ++n) {
            uint64_t mixed = context[0] * w.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j)
                mixed ^= context[j] * w.ple_layer_multipliers[static_cast<size_t>(j)];
            const int64_t base = (n - 2) * w.ple_heads_per_ngram;
            for (int64_t q = 0; q < w.ple_heads_per_ngram; ++q) {
                const int64_t head = base + q;
                rows[static_cast<size_t>(i * heads + head)] = static_cast<int32_t>(
                    mixed % static_cast<uint64_t>(w.ple_head_vocab_sizes[head]) +
                    w.ple_head_offsets[head]);
            }
        }
    }
    data.resize(static_cast<size_t>(w.ple_head_dim) * heads * kReplay);
    if (!w.ple_reader.gather(rows.data(), static_cast<int64_t>(rows.size()), data.data()))
        return fail(error, "PLE tail gather failed");
    return true;
}

bool replay_group(ggml_backend_t backend, const Qwen4ExpWeights & w,
                  Qwen4ExpCache & cache, int group, const int32_t * tokens,
                  std::string & error) {
    const int first = kStarts[static_cast<size_t>(group)];
    if (first + 2 >= w.n_layer) return fail(error, "invalid GDN group geometry");
    for (int il = first; il < first + 3; ++il)
        if (w.layers[il].is_full_attention) return fail(error, "GDN group contains FA layer");

    std::vector<float> seed;
    if (!boundary_tail(w, group, seed, error)) return false;
    std::vector<float> ple;
    if (group == 0 && !ple_tail(w, tokens, ple, error)) return false;

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 90000 +
                      ggml_graph_overhead_custom(90000, false) + (1u << 20);
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return fail(error, "tail replay context allocation failed");
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 90000, false);
    const int channels = group == 0 ? 1 : 2;
    ggml_tensor * packed = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        w.n_embd, w.n_hc, kReplay, channels);
    ggml_set_input(packed);
    const size_t channel_bytes = static_cast<size_t>(w.n_embd) * w.n_hc *
                                 kReplay * sizeof(float);
    ggml_tensor * res_hc = ggml_view_4d(ctx, packed, w.n_embd, w.n_hc, kReplay, 1,
        packed->nb[1], packed->nb[2], packed->nb[3], 0);
    ggml_tensor * xn_next = channels == 2
        ? ggml_view_4d(ctx, packed, w.n_embd, w.n_hc, kReplay, 1,
              packed->nb[1], packed->nb[2], packed->nb[3], channel_bytes)
        : nullptr;
    ggml_tensor * ple_in = nullptr;
    if (group == 0) {
        ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
            w.ple_head_dim * w.ple_n_heads, kReplay);
        ggml_set_input(ple_in);
    }

    for (int il = first; il < first + 3; ++il) {
        const Qwen4ExpLayer & layer = w.layers[il];
        if (layer.is_ple) {
            if (!ple_in || cache.ple_conv_state.size() != 1) {
                ggml_free(ctx);
                return fail(error, "missing stem PLE input/state");
            }
            res_hc = build_ple(ctx, graph, res_hc, ple_in, layer, w,
                               cache.ple_conv_state[0],
                               [](ggml_tensor *, const char *) {});
            xn_next = nullptr;
        }
        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = xn_next
            ? hc_mix_from_xn(ctx, xn_next, layer.hc_attn_down, layer.hc_attn_up,
                  layer.hc_attn_inject, &inject, w.n_embd, w.n_hc)
            : hc_mix(ctx, res_hc, layer.hc_attn_norm, layer.hc_attn_down,
                  layer.hc_attn_up, layer.hc_attn_inject, &inject,
                  w.n_embd, w.n_hc, w.rms_eps);
        xn_next = nullptr;
        const auto found = std::find(cache.linear_layer_ids.begin(),
                                     cache.linear_layer_ids.end(), il);
        if (found == cache.linear_layer_ids.end()) {
            ggml_free(ctx);
            return fail(error, "GDN layer missing from cache inventory");
        }
        const size_t linear = static_cast<size_t>(found - cache.linear_layer_ids.begin());
        cur = build_linear_attn(ctx, graph, cur, layer, w,
                                cache.ssm_state[linear], cache.conv_state[linear], il);
        if (il == first + 2) break; // final FFN cannot affect reconstructed state

        ggml_tensor * ffn_fused = hc_combine_norm(ctx, inject, res_hc, cur,
            layer.hc_ffn_norm, w.n_embd, w.n_hc, kReplay, w.rms_eps);
        res_hc = hc_norm_res(ctx, ffn_fused, w.n_embd, w.n_hc, kReplay);
        ggml_tensor * ffn_xn = hc_norm_xn(ctx, ffn_fused, w.n_embd, w.n_hc, kReplay);
        cur = hc_mix_from_xn(ctx, ffn_xn, layer.hc_ffn_down, layer.hc_ffn_up,
                            layer.hc_ffn_inject, &inject, w.n_embd, w.n_hc);
        cur = build_moe(ctx, cur, layer, w, il);
        ggml_tensor * combined = hc_combine_norm(ctx, inject, res_hc, cur,
            w.layers[il + 1].hc_attn_norm, w.n_embd, w.n_hc, kReplay, w.rms_eps);
        res_hc = hc_norm_res(ctx, combined, w.n_embd, w.n_hc, kReplay);
        xn_next = hc_norm_xn(ctx, combined, w.n_embd, w.n_hc, kReplay);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = allocator && ggml_gallocr_alloc_graph(allocator, graph);
    if (!allocated) {
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return fail(error, "tail replay graph allocation failed");
    }
    ggml_backend_tensor_set(packed, seed.data(), 0, seed.size() * sizeof(float));
    if (ple_in) ggml_backend_tensor_set(ple_in, ple.data(), 0, ple.size() * sizeof(float));
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return computed || fail(error, "tail replay graph compute failed");
}

} // namespace

bool qwen4exp_tail_replay_992_100(ggml_backend_t backend,
                                  const Qwen4ExpWeights & weights,
                                  Qwen4ExpCache & cache,
                                  const int32_t * tokens,
                                  std::string & error) {
    if (!backend || !tokens || weights.n_layer != 48 ||
        cache.indexer_blocks != 248 || cache.cur_pos != kBase)
        return fail(error, "tail replay requires native Qwen3.8 Flash B992 state");
    if (!std::equal(kFullLayers.begin(), kFullLayers.end(),
                    cache.full_layer_ids.begin(), cache.full_layer_ids.end()) ||
        !std::equal(kLinearLayers.begin(), kLinearLayers.end(),
                    cache.linear_layer_ids.begin(), cache.linear_layer_ids.end()) ||
        cache.ple_layer_ids != std::vector<int>{1} ||
        weights.ple_layer_ids != std::vector<int32_t>{1})
        return fail(error, "unexpected Qwen3.8 Flash layer geometry");
    for (int il = 0; il < weights.n_layer; ++il) {
        const bool expected_full = std::find(kFullLayers.begin(), kFullLayers.end(), il) !=
                                   kFullLayers.end();
        if (weights.layers[il].is_full_attention != expected_full ||
            weights.layers[il].is_ple != (il == 1))
            return fail(error, "weight flags disagree with pinned layer geometry");
    }
    if (std::getenv("QWEN4EXP_UPSTREAM") || std::getenv("QWEN4EXP_HC_UNFUSED"))
        return fail(error, "tail replay boundary contract requires fused HC");
    for (int group = 0; group < static_cast<int>(kStarts.size()); ++group)
        if (!replay_group(backend, weights, cache, group, tokens, error)) return false;
    cache.ple_prev.assign(tokens + kBase - 2, tokens + kBase);
    clear_qwen4exp_decode_workspace(cache.decode_workspace);
    error.clear();
    return true;
}

} // namespace luce::common
