// Pipelined hybrid MoE decode implementation.
// See qwen35moe_pipelined_decode.h for design rationale.

#include "qwen35moe_pipelined_decode.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace dflash::common {

using PipelineClock = std::chrono::steady_clock;

static uint64_t pipe_elapsed_us(PipelineClock::time_point s, PipelineClock::time_point e) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
}

// ─── CachedPrefnGraph ─────────────────────────────────────────────────────────

void CachedPrefnGraph::free() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    gf = nullptr;
    inp_embed = nullptr;
    ffn_post = nullptr;
    ffn_residual = nullptr;
    moe_selected = nullptr;
    moe_weights = nullptr;
}

// Build a cached pre-FFN graph for a DeltaNet layer.
// DeltaNet layers have no kv_start-dependent views — the graph structure is
// identical across tokens. We build once and reuse by updating inp_embed data.
static bool build_cached_deltanet_prefn(
    CachedPrefnGraph & out,
    ggml_backend_t backend,
    const TargetWeights & w,
    TargetCache & cache,
    int layer_idx,
    int kv_start,
    int kq_stride_pad) {

    out.free();

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    const int hidden = w.n_embd;
    out.inp_embed = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, hidden, 1, 1);
    ggml_set_name(out.inp_embed, "inp_embed");
    ggml_set_input(out.inp_embed);

    // DeltaNet layers don't use positions/mask (recurrent, not attention-based)
    out.gf = ggml_new_graph_custom(out.ctx, 16384, false);
    QwenLayerPrefnOutputs go = build_qwen35_layer_prefn(
        out.ctx, out.gf, w, cache, layer_idx,
        out.inp_embed, /*positions=*/nullptr, /*attn_mask=*/nullptr,
        kv_start, /*n_tokens=*/1, /*fa_window=*/0);
    if (!go.residual || !go.post) { out.free(); return false; }

    out.ffn_residual = go.residual;
    out.ffn_post = go.post;
    out.moe_selected = go.moe_selected;
    out.moe_weights = go.moe_weights;

    if (go.moe_selected) {
        ggml_set_output(go.moe_selected);
        ggml_build_forward_expand(out.gf, go.moe_selected);
    }
    if (go.moe_weights) {
        ggml_set_output(go.moe_weights);
        ggml_build_forward_expand(out.gf, go.moe_weights);
    }
    ggml_set_output(go.residual);
    ggml_build_forward_expand(out.gf, go.residual);
    ggml_set_output(go.post);
    ggml_build_forward_expand(out.gf, go.post);

    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

// ─── PipelinedDecodeState ─────────────────────────────────────────────────────

void PipelinedDecodeState::destroy() {
    for (auto & cpg : cached_prefn) cpg.free();
    cached_prefn.clear();
    gpu_state.destroy();
    routing_ids_buf.clear();
    routing_weights_buf.clear();
    ffn_post_host_buf.clear();
    cold_in_zeroed = false;
    n_layer = 0;
}

bool init_pipelined_decode_state(
    PipelinedDecodeState & out,
    ggml_backend_t backend,
    const TargetWeights & w,
    TargetCache & cache,
    int kv_start,
    int kq_stride_pad) {

    out.destroy();

    out.n_layer = w.n_layer;
    out.n_embd = w.n_embd;
    out.n_expert_used = w.n_expert_used;
    out.full_attention_interval = w.full_attention_interval;

    // Init GPU-resident state (act_cur + combine graph)
    if (!init_gpu_resident_state(out.gpu_state, backend, w.n_embd)) {
        return false;
    }

    // Allocate persistent host buffers
    out.routing_ids_buf.resize((size_t)w.n_expert_used);
    out.routing_weights_buf.resize((size_t)w.n_expert_used);
    out.ffn_post_host_buf.resize((size_t)w.n_embd);

    // Build cached pre-FFN graphs for DeltaNet layers
    out.cached_prefn.resize((size_t)w.n_layer);
    int cached_count = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        if (!is_attn) {
            // DeltaNet layer: cache the graph
            if (!build_cached_deltanet_prefn(
                    out.cached_prefn[(size_t)il], backend, w, cache, il, kv_start, kq_stride_pad)) {
                std::fprintf(stderr, "[pipelined] failed to cache DeltaNet prefn for layer %d\n", il);
                // Non-fatal: will fall back to dynamic build for this layer
            } else {
                cached_count++;
            }
        }
        // Attention layers: cached_prefn[il] remains invalid (rebuilt per-token)
    }

    out.cold_in_zeroed = true;
    // cold_in was already zeroed in init_gpu_resident_state

    return true;
}

// ─── Pipelined decode: one token through all layers ───────────────────────────

bool pipelined_decode_one_token(
    PipelinedDecodeState & state,
    ggml_backend_t backend,
    const TargetWeights & w,
    TargetCache & cache,
    Qwen35MoeHybridStorage & hybrid,
    int kv_pos,
    int kq_stride_pad,
    PipelinedDecodeTelemetry * tel) {

    const int n_layer = state.n_layer;
    const int n_embd = state.n_embd;
    const int n_expert_used = state.n_expert_used;
    ggml_backend_t cpu_be = hybrid.cpu_backend;

    if (tel) {
        tel->total_us = 0;
        tel->prefn_graph_build_us = 0;
        tel->prefn_compute_us = 0;
        tel->routing_readback_us = 0;
        tel->ffn_us = 0;
        tel->ffn_allhot_us = 0;
        tel->ffn_mixed_us = 0;
        tel->allhot_layers = 0;
        tel->mixed_layers = 0;
        tel->total_layers = 0;
    }

    const auto tok_t0 = PipelineClock::now();
    StepGraph dyn_sg;  // for attention layers (rebuilt per-token)

    for (int il = 0; il < n_layer; ++il) {
        const bool is_attn = (((il + 1) % state.full_attention_interval) == 0);
        const auto prefn_build_t0 = PipelineClock::now();

        ggml_tensor * ffn_post_gpu = nullptr;
        ggml_tensor * ffn_residual_gpu = nullptr;
        ggml_tensor * moe_selected_tensor = nullptr;
        ggml_tensor * moe_weights_tensor = nullptr;

        if (is_attn || !state.cached_prefn[(size_t)il].valid()) {
            // Attention layer OR failed DeltaNet cache: rebuild graph dynamically
            if (!build_layer_prefn_step(dyn_sg, w, cache, backend,
                                        il, kv_pos, /*n_tokens=*/1,
                                        /*with_mask=*/false, /*fa_window=*/0, kq_stride_pad)) {
                step_graph_destroy(dyn_sg);
                return false;
            }
            // Copy act_cur to graph input (GPU→GPU)
            ggml_backend_tensor_copy(state.gpu_state.act_cur, dyn_sg.inp_embed);
            if (dyn_sg.positions) {
                int32_t pos4[4] = {kv_pos, kv_pos, kv_pos, 0};
                ggml_backend_tensor_set(dyn_sg.positions, pos4, 0, sizeof(pos4));
            }

            if (tel) tel->prefn_graph_build_us += pipe_elapsed_us(prefn_build_t0, PipelineClock::now());

            const auto prefn_compute_t0 = PipelineClock::now();
            auto st = ggml_backend_graph_compute(backend, dyn_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                step_graph_destroy(dyn_sg);
                return false;
            }
            if (tel) tel->prefn_compute_us += pipe_elapsed_us(prefn_compute_t0, PipelineClock::now());

            ffn_post_gpu = dyn_sg.ffn_post;
            ffn_residual_gpu = dyn_sg.ffn_residual;
            moe_selected_tensor = (!dyn_sg.moe_selected.empty() && (size_t)il < dyn_sg.moe_selected.size())
                ? dyn_sg.moe_selected[(size_t)il] : nullptr;
            moe_weights_tensor = dyn_sg.moe_weights;
        } else {
            // DeltaNet layer: reuse cached graph, just update input
            auto & cpg = state.cached_prefn[(size_t)il];
            ggml_backend_tensor_copy(state.gpu_state.act_cur, cpg.inp_embed);

            if (tel) tel->prefn_graph_build_us += pipe_elapsed_us(prefn_build_t0, PipelineClock::now());

            const auto prefn_compute_t0 = PipelineClock::now();
            auto st = ggml_backend_graph_compute(backend, cpg.gf);
            if (st != GGML_STATUS_SUCCESS) return false;
            if (tel) tel->prefn_compute_us += pipe_elapsed_us(prefn_compute_t0, PipelineClock::now());

            ffn_post_gpu = cpg.ffn_post;
            ffn_residual_gpu = cpg.ffn_residual;
            moe_selected_tensor = cpg.moe_selected;
            moe_weights_tensor = cpg.moe_weights;
        }

        // ── Read routing decisions (tiny: 32 + 32 bytes) ──
        const auto routing_t0 = PipelineClock::now();
        if (!moe_selected_tensor || !moe_weights_tensor) return false;
        ggml_backend_tensor_get(moe_selected_tensor, state.routing_ids_buf.data(), 0,
                                sizeof(int32_t) * (size_t)n_expert_used);
        ggml_backend_tensor_get(moe_weights_tensor, state.routing_weights_buf.data(), 0,
                                sizeof(float) * (size_t)n_expert_used);
        if (tel) tel->routing_readback_us += pipe_elapsed_us(routing_t0, PipelineClock::now());

        // ── FFN: hot/cold partition + compute ──
        const auto ffn_t0 = PipelineClock::now();
        auto & storage = hybrid.layers[(size_t)il];
        const auto & L = w.layers[(size_t)il];

        // Partition into hot/cold (fast: just a lookup table scan, ~8 iterations)
        int n_hot = 0, n_cold = 0;
        int32_t hot_ids[8], cold_ids[8];
        float hot_weights[8], cold_weights[8];

        for (int i = 0; i < n_expert_used; ++i) {
            const int32_t gid = state.routing_ids_buf[(size_t)i];
            if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) return false;
            const int32_t hot_local = storage.hot_local_by_global[(size_t)gid];
            if (hot_local >= 0) {
                hot_ids[n_hot] = hot_local;
                hot_weights[n_hot] = state.routing_weights_buf[(size_t)i];
                n_hot++;
            } else {
                const int32_t cold_local = storage.cold_local_by_global[(size_t)gid];
                if (cold_local >= 0) {
                    cold_ids[n_cold] = cold_local;
                    cold_weights[n_cold] = state.routing_weights_buf[(size_t)i];
                    n_cold++;
                }
            }
        }

        const bool has_hot = (n_hot > 0);
        const bool has_cold = (n_cold > 0);
        const bool has_shared = (L.ffn_up_shexp && L.ffn_gate_shexp && L.ffn_down_shexp);

        // ── Read ffn_post to CPU NOW (before hot launch) ──
        // The routing readback above already synced the GPU stream, so ffn_post
        // is guaranteed ready. Reading it here avoids a sync AFTER hot launch.
        if (has_cold) {
            ggml_backend_tensor_get(ffn_post_gpu, state.ffn_post_host_buf.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
        }

        // ── GPU→GPU: copy residual to combine input ──
        ggml_backend_tensor_copy(ffn_residual_gpu, state.gpu_state.combine.residual_in);

        // ── Prepare + launch hot graph (async — returns immediately) ──
        bool hot_async_launched = false;
        if (has_hot || has_shared) {
            if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
                build_cached_hot_graph(storage.hot_graph, backend,
                                       storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                       L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                                       L, n_embd, w.n_ff_exp, n_hot);
            }
            if (storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
                ggml_backend_tensor_copy(ffn_post_gpu, storage.hot_graph.inp);
                if (storage.hot_graph.ids && has_hot) {
                    ggml_backend_tensor_set(storage.hot_graph.ids, hot_ids, 0,
                                            sizeof(int32_t) * (size_t)n_hot);
                }
                if (storage.hot_graph.weights && has_hot) {
                    ggml_backend_tensor_set(storage.hot_graph.weights, hot_weights, 0,
                                            sizeof(float) * (size_t)n_hot);
                }
                // Launch hot GPU async — no sync until combine
                ggml_backend_graph_compute_async(backend, storage.hot_graph.gf);
                hot_async_launched = true;
            }
        }

        // ── Cold path: runs on CPU IN PARALLEL with hot GPU ──
        if (has_cold) {
            // ffn_post already read above (before hot launch) — no GPU sync here!
            if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold) {
                build_cached_cold_graph(storage.cold_graph, cpu_be,
                                        storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                        L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                                        n_embd, w.n_ff_exp, n_cold);
            }
            if (storage.cold_graph.valid() && storage.cold_graph.n_hot == n_cold) {
                ggml_backend_tensor_set(storage.cold_graph.inp, state.ffn_post_host_buf.data(), 0,
                                        sizeof(float) * (size_t)n_embd);
                ggml_backend_tensor_set(storage.cold_graph.ids, cold_ids, 0,
                                        sizeof(int32_t) * (size_t)n_cold);
                ggml_backend_tensor_set(storage.cold_graph.weights, cold_weights, 0,
                                        sizeof(float) * (size_t)n_cold);
                // CPU cold compute — hot GPU runs concurrently on its stream
                auto cst = ggml_backend_graph_compute(cpu_be, storage.cold_graph.gf);
                if (cst != GGML_STATUS_SUCCESS) {
                    if (hot_async_launched) ggml_backend_synchronize(backend);
                    return false;
                }
            } else {
                if (hot_async_launched) ggml_backend_synchronize(backend);
                return false;
            }
        }

        // ── Sync hot GPU (only now — after cold CPU finished) ──
        if (hot_async_launched) {
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_copy(storage.hot_graph.output, state.gpu_state.combine.hot_in);
        } else {
            float zeros[8192];
            std::memset(zeros, 0, sizeof(float) * (size_t)n_embd);
            ggml_backend_tensor_set(state.gpu_state.combine.hot_in, zeros, 0,
                                    sizeof(float) * (size_t)n_embd);
        }

        // ── Upload cold result (or keep zeros) ──
        if (has_cold) {
            ggml_backend_tensor_get(storage.cold_graph.output, state.ffn_post_host_buf.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
            ggml_backend_tensor_set(state.gpu_state.combine.cold_in, state.ffn_post_host_buf.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
            state.cold_in_zeroed = false;
        } else if (!state.cold_in_zeroed) {
            float zeros[8192];
            std::memset(zeros, 0, sizeof(float) * (size_t)n_embd);
            ggml_backend_tensor_set(state.gpu_state.combine.cold_in, zeros, 0,
                                    sizeof(float) * (size_t)n_embd);
            state.cold_in_zeroed = true;
        }

        // ── Combine: output = residual + hot + cold ──
        auto cst = ggml_backend_graph_compute(backend, state.gpu_state.combine.gf);
        if (cst != GGML_STATUS_SUCCESS) return false;

        // ── Copy combine output to persistent act_cur ──
        ggml_backend_tensor_copy(state.gpu_state.combine.output, state.gpu_state.act_cur);

        const auto ffn_t1 = PipelineClock::now();
        if (tel) {
            uint64_t ffn_layer_us = pipe_elapsed_us(ffn_t0, ffn_t1);
            tel->ffn_us += ffn_layer_us;
            tel->total_layers++;
            if (has_cold) {
                tel->mixed_layers++;
                tel->ffn_mixed_us += ffn_layer_us;
            } else {
                tel->allhot_layers++;
                tel->ffn_allhot_us += ffn_layer_us;
            }
        }
    }

    step_graph_destroy(dyn_sg);

    if (tel) {
        tel->total_us = pipe_elapsed_us(tok_t0, PipelineClock::now());
    }
    return true;
}

}  // namespace dflash::common
