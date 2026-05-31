// Pipelined hybrid MoE decode implementation.
// See qwen35moe_pipelined_decode.h for design rationale.

#include "qwen35moe_pipelined_decode.h"

#include "../common/moe_hybrid_types_impl.h"

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
    for (auto & rff : cached_routed_ffn) rff.free();
    cached_routed_ffn.clear();
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
    MoeHybridStorage & hybrid,
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

    // Check if routed FFN pipeline is disabled
    const bool routed_disabled = (std::getenv("DFLASH_QWEN35MOE_NO_ROUTED") != nullptr);

    // Cold experts are computed on the cold backend (CPU/Halo) by default.
    // Set DFLASH_DROP_COLD=1 to skip cold computation (fast but lossy).
    out.cold_compute = (std::getenv("DFLASH_DROP_COLD") == nullptr);

    // Build cached pre-FFN graphs for all DeltaNet layers.
    out.cached_prefn.resize((size_t)w.n_layer);
    int cached_prefn_count = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        if (!is_attn) {
            if (!build_cached_deltanet_prefn(
                    out.cached_prefn[(size_t)il], backend, w, cache, il, kv_start, kq_stride_pad)) {
                std::fprintf(stderr, "[pipelined] failed to cache DeltaNet prefn for layer %d\n", il);
            } else {
                cached_prefn_count++;
            }
        }
    }

    // Build cached routed FFN graphs for ALL layers (StreamMoE-inspired pipeline).
    // Includes attention layers — eliminates expensive split-path FFN for mixed layers.
    // Cold entries get weight=0 at runtime, contributing nothing to output.
    out.cached_routed_ffn.resize((size_t)w.n_layer);
    int routed_count = 0;
    if (!routed_disabled) {
        for (int il = 0; il < w.n_layer; ++il) {
            if ((size_t)il >= hybrid.layers.size()) continue;

            auto & storage = hybrid.layers[(size_t)il];
            const TargetLayer & L = w.layers[(size_t)il];

            if (!build_cached_hot_graph(
                    out.cached_routed_ffn[(size_t)il], backend,
                    storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                    L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                    make_moe_layer_desc(L), w.n_embd, w.n_ff_exp, w.n_expert_used)) {
                // Non-fatal: fall back to split path for this layer
            } else {
                routed_count++;
            }
        }
    }

    std::fprintf(stderr, "[pipelined] cached %d prefn + %d routed FFN graphs%s\n",
                 cached_prefn_count, routed_count,
                 out.cold_compute ? "" : " (drop_cold=lossy)");

    // Initialize fused cold FFN compute (bypasses ggml graph dispatch)
    if (out.cold_compute) {
        out.cold_ffn_compute = make_cpu_cold_ffn_compute(w.n_ff_exp);
        out.cold_ffn_layers.resize((size_t)w.n_layer);
        out.cold_output_buf.resize((size_t)w.n_embd);
        for (int il = 0; il < w.n_layer && (size_t)il < hybrid.layers.size(); ++il) {
            auto & storage = hybrid.layers[(size_t)il];
            const TargetLayer & L = w.layers[(size_t)il];
            auto & cl = out.cold_ffn_layers[(size_t)il];
            cl.fused_gate_up = (storage.gate_up_cold != nullptr);
            if (cl.fused_gate_up) {
                cl.gate_up_data = storage.gate_up_cold ? storage.gate_up_cold->data : nullptr;
                cl.gate_up_stride = storage.gate_up_cold ? storage.gate_up_cold->nb[2] : 0;
                cl.gate_up_type = storage.gate_up_cold ? storage.gate_up_cold->type : GGML_TYPE_Q4_K;
                cl.gate_up_scale = L.ffn_gate_up_exps_s;
            } else {
                cl.gate_data = storage.gate_cold ? storage.gate_cold->data : nullptr;
                cl.up_data = storage.up_cold ? storage.up_cold->data : nullptr;
                cl.gate_stride = storage.gate_cold ? storage.gate_cold->nb[2] : 0;
                cl.up_stride = storage.up_cold ? storage.up_cold->nb[2] : 0;
                cl.gate_type = storage.gate_cold ? storage.gate_cold->type : GGML_TYPE_Q4_K;
                cl.up_type = storage.up_cold ? storage.up_cold->type : GGML_TYPE_Q4_K;
                cl.gate_scale = L.ffn_gate_exps_s;
                cl.up_scale = L.ffn_up_exps_s;
            }
            cl.down_data = storage.down_cold ? storage.down_cold->data : nullptr;
            cl.down_stride = storage.down_cold ? storage.down_cold->nb[2] : 0;
            cl.down_type = storage.down_cold ? storage.down_cold->type : GGML_TYPE_Q4_K;
            cl.down_scale = L.ffn_down_exps_s;
        }
        std::fprintf(stderr, "[pipelined] cold FFN: fused kernel (bypasses ggml graph)\n");
    }

    out.cold_in_zeroed = true;
    return true;
}

// ─── Pipelined decode: one token through all layers ───────────────────────────

bool pipelined_decode_one_token(
    PipelinedDecodeState & state,
    ggml_backend_t backend,
    const TargetWeights & w,
    TargetCache & cache,
    MoeHybridStorage & hybrid,
    int kv_pos,
    int kq_stride_pad,
    PipelinedDecodeTelemetry * tel) {

    const int n_layer = state.n_layer;
    const int n_embd = state.n_embd;
    const int n_expert_used = state.n_expert_used;
    ggml_backend_t cpu_be = hybrid.cpu_backend;

    if (tel) {
        *tel = PipelinedDecodeTelemetry{};
    }

    const auto tok_t0 = PipelineClock::now();
    StepGraph dyn_sg;  // for attention layers (rebuilt per-token)

    for (int il = 0; il < n_layer; ++il) {
        const bool is_attn = (((il + 1) % state.full_attention_interval) == 0);
        const auto prefn_build_t0 = PipelineClock::now();

        // ══════════════════════════════════════════════════════════════════════
        // ROUTED FFN FAST PATH (StreamMoE-inspired async pipeline):
        // prefn(async) → sync → routing readback → rffn(async) + cold(CPU parallel) → combine(async)
        // Handles both all-hot and mixed layers. Cold compute runs on CPU
        // in parallel with GPU rffn — zero overhead when all experts are hot.
        // ══════════════════════════════════════════════════════════════════════
        if (!is_attn
            && state.cached_prefn[(size_t)il].valid()
            && state.cached_routed_ffn[(size_t)il].valid()) {

            auto & cpg = state.cached_prefn[(size_t)il];
            auto & rffn = state.cached_routed_ffn[(size_t)il];

            // 1. Copy act_cur → prefn input (GPU→GPU async)
            ggml_backend_tensor_copy_async(backend, backend, state.gpu_state.act_cur, cpg.inp_embed);

            if (tel) tel->prefn_graph_build_us += pipe_elapsed_us(prefn_build_t0, PipelineClock::now());

            // 2. Run prefn graph (DeltaNet + router)
            const auto prefn_compute_t0 = PipelineClock::now();
            ggml_backend_graph_compute_async(backend, cpg.gf);

            // 3. Sync to read routing decisions from prefn output
            const auto sync_t0 = PipelineClock::now();
            ggml_backend_synchronize(backend);
            const auto sync_t1 = PipelineClock::now();

            // Read routing decisions from GPU
            int32_t global_ids[8];
            float   router_weights[8];
            ggml_backend_tensor_get(cpg.moe_selected, global_ids, 0,
                                    sizeof(int32_t) * (size_t)n_expert_used);
            ggml_backend_tensor_get(cpg.moe_weights, router_weights, 0,
                                    sizeof(float) * (size_t)n_expert_used);
            const auto readback_t1 = PipelineClock::now();

            // CPU-side local ID mapping + cold partition (trivial: 8 lookups)
            auto & storage = hybrid.layers[(size_t)il];
            int32_t local_ids[8];
            float   masked_weights[8];
            int32_t cold_ids[8];
            float   cold_weights[8];
            int n_cold = 0;
            int layer_cold_hits = 0;
            for (int i = 0; i < n_expert_used; ++i) {
                int32_t gid = global_ids[i];
                int32_t lid = (gid >= 0 && gid < (int)storage.hot_local_by_global.size())
                              ? storage.hot_local_by_global[(size_t)gid] : -1;
                if (lid >= 0) {
                    local_ids[i] = lid;
                    masked_weights[i] = router_weights[i];
                } else {
                    local_ids[i] = 0;       // safe: maps to expert 0 (result zeroed by weight)
                    masked_weights[i] = 0.0f; // cold expert contributes nothing to hot path
                    layer_cold_hits++;
                    // Record for cold compute
                    if (state.cold_ffn_compute && gid >= 0 && gid < (int)storage.cold_local_by_global.size()) {
                        int32_t cold_local = storage.cold_local_by_global[(size_t)gid];
                        if (cold_local >= 0) {
                            cold_ids[n_cold] = cold_local;
                            cold_weights[n_cold] = router_weights[i];
                            n_cold++;
                        }
                    }
                }
            }
            const bool has_cold_selected = (n_cold > 0);
            const auto remap_t1 = PipelineClock::now();

            // D2H ffn_post for cold compute (GPU already synced, data is ready)
            if (has_cold_selected) {
                ggml_backend_tensor_get(cpg.ffn_post, state.ffn_post_host_buf.data(), 0,
                                        sizeof(float) * (size_t)n_embd);
            }

            // Upload pre-computed inputs to rffn graph (H→D async on compute stream)
            ggml_backend_tensor_set_async(backend, rffn.ids, local_ids, 0,
                                          sizeof(int32_t) * (size_t)n_expert_used);
            ggml_backend_tensor_set_async(backend, rffn.weights, masked_weights, 0,
                                          sizeof(float) * (size_t)n_expert_used);
            // Copy ffn_post from prefn output → rffn input (GPU→GPU, already synced)
            ggml_backend_tensor_copy_async(backend, backend, cpg.ffn_post, rffn.inp);

            // 4. Copy residual to combine input (async)
            ggml_backend_tensor_copy_async(backend, backend, cpg.ffn_residual, state.gpu_state.combine.residual_in);

            // 5. Run routed FFN graph (async — mul_mat_id + shared expert)
            ggml_backend_graph_compute_async(backend, rffn.gf);

            // 6. Cold compute on CPU (parallel with GPU rffn above)
            const auto cold_t0 = PipelineClock::now();
            if (has_cold_selected) {
                state.cold_ffn_compute->compute(
                    state.cold_ffn_layers[(size_t)il],
                    state.ffn_post_host_buf.data(),
                    cold_ids, cold_weights, n_cold,
                    n_embd, w.n_ff_exp,
                    state.cold_output_buf.data());
            }
            if (tel && has_cold_selected) tel->cold_compute_us += pipe_elapsed_us(cold_t0, PipelineClock::now());

            // 7. Copy FFN output → combine.hot_in (async, ordered after FFN on GPU stream)
            ggml_backend_tensor_copy_async(backend, backend, rffn.output, state.gpu_state.combine.hot_in);

            // 8. Upload cold result or ensure cold_in is zero
            if (has_cold_selected) {
                ggml_backend_tensor_set_async(backend, state.gpu_state.combine.cold_in,
                                              state.cold_output_buf.data(), 0,
                                              sizeof(float) * (size_t)n_embd);
                state.cold_in_zeroed = false;
            } else if (!state.cold_in_zeroed) {
                static float zeros[8192] = {};
                ggml_backend_tensor_set_async(backend, state.gpu_state.combine.cold_in, zeros, 0,
                                               sizeof(float) * (size_t)n_embd);
                state.cold_in_zeroed = true;
            }

            // 9. Run combine graph (async — adds residual + hot + cold)
            ggml_backend_graph_compute_async(backend, state.gpu_state.combine.gf);

            // 10. Copy combine output → act_cur for next layer (async)
            ggml_backend_tensor_copy_async(backend, backend, state.gpu_state.combine.output, state.gpu_state.act_cur);

            if (tel) {
                tel->prefn_compute_us += pipe_elapsed_us(prefn_compute_t0, PipelineClock::now());
                tel->routed_prefn_us += pipe_elapsed_us(prefn_compute_t0, sync_t0);
                tel->routed_sync_us += pipe_elapsed_us(sync_t0, sync_t1);
                tel->routed_readback_us += pipe_elapsed_us(sync_t1, readback_t1);
                tel->routed_cpu_remap_us += pipe_elapsed_us(readback_t1, remap_t1);
                tel->routed_ffn_dispatch_us += pipe_elapsed_us(remap_t1, PipelineClock::now());
                tel->routed_cold_expert_hits += layer_cold_hits;
                tel->routed_total_expert_slots += n_expert_used;
                if (has_cold_selected) {
                    tel->mixed_layers++;
                } else {
                    tel->allhot_layers++;
                }
                tel->total_layers++;
                tel->routed_ffn_layers++;
            }
            continue;
        }

        // ══════════════════════════════════════════════════════════════════════
        // SPLIT PATH: separate prefn + routing readback + FFN (original logic)
        // Used for attention layers or layers without routed FFN graph.
        // ══════════════════════════════════════════════════════════════════════

        // Sync any pending async work before entering the split path
        // (split path needs synchronous access to GPU data)
        ggml_backend_synchronize(backend);

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
            // Copy act_cur to graph input (GPU→GPU) — async on compute stream
            ggml_backend_tensor_copy_async(backend, backend, state.gpu_state.act_cur, dyn_sg.inp_embed);
            if (dyn_sg.positions) {
                int32_t pos4[4] = {kv_pos, kv_pos, kv_pos, 0};
                ggml_backend_tensor_set_async(backend, dyn_sg.positions, pos4, 0, sizeof(pos4));
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
            // Async copy on compute stream — ordered before next graph_compute
            ggml_backend_tensor_copy_async(backend, backend, state.gpu_state.act_cur, cpg.inp_embed);

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
        // Use get_async + single sync instead of 2 separate sync tensor_gets.
        // After graph_compute (SYNC) above, data is ready — just need D2H copy.
        const auto routing_t0 = PipelineClock::now();
        if (!moe_selected_tensor || !moe_weights_tensor) return false;
        ggml_backend_tensor_get_async(backend, moe_selected_tensor, state.routing_ids_buf.data(), 0,
                                sizeof(int32_t) * (size_t)n_expert_used);
        ggml_backend_tensor_get_async(backend, moe_weights_tensor, state.routing_weights_buf.data(), 0,
                                sizeof(float) * (size_t)n_expert_used);
        ggml_backend_synchronize(backend);
        if (tel) tel->routing_readback_us += pipe_elapsed_us(routing_t0, PipelineClock::now());

        // ── FFN: use routed FFN (cold-masking) if graph available, else split path ──
        const auto ffn_t0 = PipelineClock::now();
        auto & storage = hybrid.layers[(size_t)il];
        const auto & L = w.layers[(size_t)il];

        // Try routed FFN path for this layer (works for attention layers too)
        // Handles cold experts inline — cold compute runs parallel with GPU rffn.
        auto & rffn = state.cached_routed_ffn[(size_t)il];
        if (rffn.valid()) {
            // Partition hot/cold: remap global→local, zero cold weights for hot path
            int32_t local_ids[8];
            float   masked_weights[8];
            int32_t cold_ids[8];
            float   cold_weights[8];
            int n_cold = 0;
            int layer_cold_hits = 0;
            for (int i = 0; i < n_expert_used; ++i) {
                int32_t gid = state.routing_ids_buf[(size_t)i];
                int32_t lid = (gid >= 0 && gid < (int)storage.hot_local_by_global.size())
                              ? storage.hot_local_by_global[(size_t)gid] : -1;
                if (lid >= 0) {
                    local_ids[i] = lid;
                    masked_weights[i] = state.routing_weights_buf[(size_t)i];
                } else {
                    local_ids[i] = 0;
                    masked_weights[i] = 0.0f;
                    layer_cold_hits++;
                    if (state.cold_ffn_compute && gid >= 0 && gid < (int)storage.cold_local_by_global.size()) {
                        int32_t cold_local = storage.cold_local_by_global[(size_t)gid];
                        if (cold_local >= 0) {
                            cold_ids[n_cold] = cold_local;
                            cold_weights[n_cold] = state.routing_weights_buf[(size_t)i];
                            n_cold++;
                        }
                    }
                }
            }
            const bool has_cold_selected = (n_cold > 0);

            // D2H ffn_post for cold compute (GPU already synced after routing readback)
            if (has_cold_selected) {
                ggml_backend_tensor_get(ffn_post_gpu, state.ffn_post_host_buf.data(), 0,
                                        sizeof(float) * (size_t)n_embd);
            }

            // Upload IDs + weights, copy inputs, dispatch rffn (all async)
            ggml_backend_tensor_set_async(backend, rffn.ids, local_ids, 0,
                                          sizeof(int32_t) * (size_t)n_expert_used);
            ggml_backend_tensor_set_async(backend, rffn.weights, masked_weights, 0,
                                          sizeof(float) * (size_t)n_expert_used);
            ggml_backend_tensor_copy_async(backend, backend, ffn_post_gpu, rffn.inp);
            ggml_backend_tensor_copy_async(backend, backend, ffn_residual_gpu, state.gpu_state.combine.residual_in);
            ggml_backend_graph_compute_async(backend, rffn.gf);

            // Cold compute on CPU (parallel with GPU rffn above)
            const auto cold_t0 = PipelineClock::now();
            if (has_cold_selected) {
                state.cold_ffn_compute->compute(
                    state.cold_ffn_layers[(size_t)il],
                    state.ffn_post_host_buf.data(),
                    cold_ids, cold_weights, n_cold,
                    n_embd, w.n_ff_exp,
                    state.cold_output_buf.data());
            }
            if (tel && has_cold_selected) tel->cold_compute_us += pipe_elapsed_us(cold_t0, PipelineClock::now());

            // Copy hot result → combine input (async, ordered after rffn on GPU stream)
            ggml_backend_tensor_copy_async(backend, backend, rffn.output, state.gpu_state.combine.hot_in);

            // Upload cold result or ensure cold_in is zero
            if (has_cold_selected) {
                ggml_backend_tensor_set_async(backend, state.gpu_state.combine.cold_in,
                                              state.cold_output_buf.data(), 0,
                                              sizeof(float) * (size_t)n_embd);
                state.cold_in_zeroed = false;
            } else if (!state.cold_in_zeroed) {
                static float zeros[8192] = {};
                ggml_backend_tensor_set_async(backend, state.gpu_state.combine.cold_in, zeros, 0,
                                               sizeof(float) * (size_t)n_embd);
                state.cold_in_zeroed = true;
            }

            ggml_backend_graph_compute_async(backend, state.gpu_state.combine.gf);
            ggml_backend_tensor_copy_async(backend, backend, state.gpu_state.combine.output, state.gpu_state.act_cur);

            if (tel) {
                uint64_t ffn_layer_us = pipe_elapsed_us(ffn_t0, PipelineClock::now());
                tel->ffn_us += ffn_layer_us;
                tel->total_layers++;
                tel->routed_ffn_layers++;
                if (has_cold_selected) {
                    tel->mixed_layers++;
                    tel->ffn_mixed_us += ffn_layer_us;
                } else {
                    tel->allhot_layers++;
                    tel->ffn_allhot_us += ffn_layer_us;
                }
                tel->routed_cold_expert_hits += layer_cold_hits;
                tel->routed_total_expert_slots += n_expert_used;
            }
            continue;
        }

        // ── Fallback: full split path (no routed FFN graph for this layer) ──

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
        const auto tensor_io_t0 = PipelineClock::now();
        if (has_cold) {
            ggml_backend_tensor_get(ffn_post_gpu, state.ffn_post_host_buf.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
        }
        if (tel) tel->ffn_post_get_us += pipe_elapsed_us(tensor_io_t0, PipelineClock::now());


        // ── GPU→GPU: copy residual to combine input (async on compute stream) ──
        ggml_backend_tensor_copy_async(backend, backend, ffn_residual_gpu, state.gpu_state.combine.residual_in);

        // ── Prepare + launch hot graph (async — returns immediately) ──
        bool hot_async_launched = false;
        if (has_hot || has_shared) {
            if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
                const auto hbuild_t0 = PipelineClock::now();
                build_cached_hot_graph(storage.hot_graph, backend,
                                       storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                       L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                                       make_moe_layer_desc(L), n_embd, w.n_ff_exp, n_hot);
                if (tel) { tel->hot_graph_build_us += pipe_elapsed_us(hbuild_t0, PipelineClock::now()); tel->hot_graph_rebuilds++; }
            }
            if (storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
                // All setup on compute stream — no per-op cudaStreamSynchronize
                ggml_backend_tensor_copy_async(backend, backend, ffn_post_gpu, storage.hot_graph.inp);
                if (storage.hot_graph.ids && has_hot) {
                    ggml_backend_tensor_set_async(backend, storage.hot_graph.ids, hot_ids, 0,
                                                  sizeof(int32_t) * (size_t)n_hot);
                }
                if (storage.hot_graph.weights && has_hot) {
                    ggml_backend_tensor_set_async(backend, storage.hot_graph.weights, hot_weights, 0,
                                                  sizeof(float) * (size_t)n_hot);
                }
                // Launch hot GPU async — queued after copies on same stream
                ggml_backend_graph_compute_async(backend, storage.hot_graph.gf);
                hot_async_launched = true;
            }
        }
        if (tel) tel->tensor_io_us += pipe_elapsed_us(tensor_io_t0, PipelineClock::now());

        // ── Cold path: runs on CPU IN PARALLEL with hot GPU ──
        const auto cold_t0 = PipelineClock::now();
        if (has_cold) {
            // ffn_post already read above (before hot launch) — no GPU sync here!
            const auto cold_compute_t0 = PipelineClock::now();
            if (state.cold_ffn_compute) {
                // Fused kernel: bypass ggml graph dispatch entirely
                state.cold_ffn_compute->compute(
                    state.cold_ffn_layers[(size_t)il],
                    state.ffn_post_host_buf.data(),
                    cold_ids,
                    cold_weights,
                    n_cold, n_embd, w.n_ff_exp,
                    state.cold_output_buf.data());
            } else {
                // Fallback: ggml cold graph (legacy path)
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
            if (tel) tel->cold_compute_us += pipe_elapsed_us(cold_compute_t0, PipelineClock::now());
        }
        if (tel) tel->cold_cpu_us += pipe_elapsed_us(cold_t0, PipelineClock::now());

        // ── Combine: queue on compute stream (no explicit sync needed) ──
        const auto combine_t0 = PipelineClock::now();
        if (hot_async_launched) {
            ggml_backend_tensor_copy_async(backend, backend, storage.hot_graph.output, state.gpu_state.combine.hot_in);
        } else {
            float zeros[8192];
            std::memset(zeros, 0, sizeof(float) * (size_t)n_embd);
            ggml_backend_tensor_set_async(backend, state.gpu_state.combine.hot_in, zeros, 0,
                                           sizeof(float) * (size_t)n_embd);
        }

        if (has_cold) {
            const float * cold_result = state.cold_ffn_compute
                ? state.cold_output_buf.data()
                : nullptr;
            if (!cold_result) {
                // Legacy path: read from ggml tensor
                ggml_backend_tensor_get(storage.cold_graph.output, state.ffn_post_host_buf.data(), 0,
                                        sizeof(float) * (size_t)n_embd);
                cold_result = state.ffn_post_host_buf.data();
            }
            ggml_backend_tensor_set_async(backend, state.gpu_state.combine.cold_in, cold_result, 0,
                                           sizeof(float) * (size_t)n_embd);
            state.cold_in_zeroed = false;
        } else if (!state.cold_in_zeroed) {
            float zeros[8192];
            std::memset(zeros, 0, sizeof(float) * (size_t)n_embd);
            ggml_backend_tensor_set_async(backend, state.gpu_state.combine.cold_in, zeros, 0,
                                           sizeof(float) * (size_t)n_embd);
            state.cold_in_zeroed = true;
        }

        ggml_backend_graph_compute_async(backend, state.gpu_state.combine.gf);

        ggml_backend_tensor_copy_async(backend, backend, state.gpu_state.combine.output, state.gpu_state.act_cur);
        if (tel) tel->combine_overhead_us += pipe_elapsed_us(combine_t0, PipelineClock::now());

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

    // Sync the compute stream before returning — caller needs act_cur on CPU.
    // All async ops (combine + copy) from the last layer must complete.
    const auto final_sync_t0 = PipelineClock::now();
    ggml_backend_synchronize(backend);

    if (tel) {
        tel->routed_final_sync_us = pipe_elapsed_us(final_sync_t0, PipelineClock::now());
        tel->total_us = pipe_elapsed_us(tok_t0, PipelineClock::now());
        // GPU idle = time in tensor I/O + routing readback + combine overhead
        // (these are all periods where GPU compute stream is idle)
        tel->gpu_idle_us = tel->tensor_io_us + tel->routing_readback_us + tel->combine_overhead_us;
    }
    return true;
}

}  // namespace dflash::common
