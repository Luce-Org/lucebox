#include "inkling2_cache.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace luce::common {

void free_inkling2_workspace(Inkling2GraphWorkspace & ws) {
    if (ws.sample_alloc) { ggml_gallocr_free(ws.sample_alloc); ws.sample_alloc = nullptr; }
    if (ws.sample_ctx)   { ggml_free(ws.sample_ctx); ws.sample_ctx = nullptr; }
    ws.sample_gf = nullptr;
    ws.sample_logits = nullptr;
    ws.sample_cands = nullptr;
    if (ws.sched) { ggml_backend_sched_free(ws.sched); ws.sched = nullptr; }
    if (ws.alloc) { ggml_gallocr_free(ws.alloc); ws.alloc = nullptr; }
    if (ws.ctx)   { ggml_free(ws.ctx); ws.ctx = nullptr; }
    ws.gf = nullptr;
    ws.built = false;
    ws.expert_nodes.clear();
}

static void free_dspark_workspace(Inkling2DSparkWorkspace & ws) {
    for (auto & sl : ws.store) {
        if (sl.alloc) { ggml_gallocr_free(sl.alloc); sl.alloc = nullptr; }
        if (sl.ctx)   { ggml_free(sl.ctx); sl.ctx = nullptr; }
        sl.gf = nullptr; sl.taps = nullptr; sl.pos = nullptr; sl.built = false;
    }
    if (ws.alloc_draft) { ggml_gallocr_free(ws.alloc_draft); ws.alloc_draft = nullptr; }
    if (ws.ctx_draft)   { ggml_free(ws.ctx_draft); ws.ctx_draft = nullptr; }
    if (ws.alloc_rb)    { ggml_gallocr_free(ws.alloc_rb); ws.alloc_rb = nullptr; }
    if (ws.ctx_rb)      { ggml_free(ws.ctx_rb); ws.ctx_rb = nullptr; }
    ws.gf_draft = nullptr;
    ws.gf_rb = nullptr;
    ws.draft_built = false;
    ws.rb_built = false;
}

static void free_tree_cache(Inkling2TreeCache & tc) {
    for (auto & kv : tc.phase_graphs) {
        auto & g = kv.second;
        if (g.sample_alloc) { ggml_gallocr_free(g.sample_alloc); g.sample_alloc = nullptr; }
        if (g.sample_ctx)   { ggml_free(g.sample_ctx); g.sample_ctx = nullptr; }
        g.sample_gf = nullptr;
        g.sample_logits = nullptr;
        g.sample_cands = nullptr;
        if (g.sched) { ggml_backend_sched_free(g.sched); g.sched = nullptr; }
        if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
        if (g.ctx)   { ggml_free(g.ctx); g.ctx = nullptr; }
    }
    tc.phase_graphs.clear();
    for (auto & kv : tc.draft_graphs) {
        auto & g = kv.second;
        if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
        if (g.ctx)   { ggml_free(g.ctx); g.ctx = nullptr; }
    }
    tc.draft_graphs.clear();
    if (tc.commit_graph.alloc) { ggml_gallocr_free(tc.commit_graph.alloc); tc.commit_graph.alloc = nullptr; }
    if (tc.commit_graph.ctx)   { ggml_free(tc.commit_graph.ctx); tc.commit_graph.ctx = nullptr; }
    tc.commit_graph.gf = nullptr;
    tc.commit_graph.built = false;
    for (auto & g : tc.prefix_store_graphs) {
        if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
        if (g.ctx)   { ggml_free(g.ctx); g.ctx = nullptr; }
        g.gf = nullptr;
        g.built = false;
    }
    if (tc.alias_ctx) { ggml_free(tc.alias_ctx); tc.alias_ctx = nullptr; }
    tc.planes.clear();
    tc.prefix_k.clear();
    tc.prefix_v.clear();
    tc.tree_k.clear();
    tc.tree_v.clear();
    tc.prefix_conv.clear();
    tc.tree_conv.clear();
    tc.pre_blocks.clear();
    tc.core_blocks.clear();
    tc.pre_partial = nullptr;
    tc.pre_tap = nullptr;
    tc.core_partial = nullptr;
    tc.core_taps = {};
    tc.node_features = nullptr;
    tc.draft_k.clear();
    tc.draft_v.clear();
    tc.draft_cap = 0;
    tc.prefix_len = 0;
    tc.n_seq_slots = 1;
    tc.is_slot_view = false;
    tc.active = false;
}

bool create_inkling2_cache(ggml_backend_t target_backend,
                           ggml_backend_t expert_backend,
                           const Inkling2Weights & w,
                           int max_ctx,
                           int max_chunk,
                           ggml_type kv_type,
                           Inkling2Cache & out,
                           std::string & error,
                           int fixed_remote_split_pos,
                           int n_seq_slots) {
    free_inkling2_cache(out);
    n_seq_slots = std::max(1, n_seq_slots);
    out.max_ctx = std::max(2048, max_ctx);
    out.kv_type = (kv_type == GGML_TYPE_F16 || kv_type == GGML_TYPE_Q8_0) ? kv_type : GGML_TYPE_Q8_0;
    // SWA ring capacity: must hold at least swa_window + max_chunk + verify slack without overwrite
    out.swa_cap = w.swa_window + std::max(max_chunk, 1024) + kInkling2MaxVerify;
    out.cur_pos = 0;
    out.dspark_ctx_pos = 0;

    // Check whether global KV fits on target_backend or should split with expert_backend (Strix Halo).
    // When fixed_remote_split_pos >= 0 (reinstate), replay the exact split boundary frozen at first load.
    const int n_global = (int) w.global_slots.size();
    const size_t kv_row_bytes = ggml_row_size(out.kv_type, w.head_dim);
    const size_t global_bytes_per_tok = (size_t) n_global * 2 * w.n_head_kv * kv_row_bytes;

    out.remote_backend = nullptr;
    out.split_cpu = expert_backend ? ggml_backend_cpu_init() : nullptr;
    out.remote_split_pos = 0;
    out.remote_cap = 0;
    out.global_local_cap = out.max_ctx;

    if (expert_backend && n_global > 0) {
        if (fixed_remote_split_pos >= 0) {
            if (fixed_remote_split_pos > 0 && fixed_remote_split_pos < out.max_ctx) {
                out.remote_backend = expert_backend;
                out.remote_split_pos = fixed_remote_split_pos;
                out.remote_cap = fixed_remote_split_pos;
                out.global_local_cap = out.max_ctx - fixed_remote_split_pos;
            }
        } else {
            const int tgpu = ggml_backend_cuda_get_device_id(target_backend);
            size_t tfree = 0, ttotal = 0;
            ggml_backend_cuda_get_device_memory(tgpu, &tfree, &ttotal);
            const size_t need_global = (size_t) out.max_ctx * global_bytes_per_tok;
            // Leave >= 4 GiB headroom on target for graph scratch, DSpark, activations, SWA & conv states
            const size_t reserve = (size_t) 4 << 30;
            if (tfree <= reserve || need_global > tfree - reserve) {
                const size_t local_budget = tfree > reserve ? (tfree - reserve) / 2 : ((size_t) 512 << 20);
                int local_toks = (int) std::min<size_t>(out.max_ctx, std::max<size_t>(4096, local_budget / std::max<size_t>(1, global_bytes_per_tok)));
                local_toks = (local_toks / 256) * 256;
                if (local_toks < out.max_ctx) {
                    out.remote_backend = expert_backend;
                    out.remote_split_pos = out.max_ctx - local_toks;
                    out.remote_cap = out.remote_split_pos;
                    out.global_local_cap = local_toks;
                    std::fprintf(stderr,
                        "[inkling2] Split-KV Global Attention enabled: Strix=[0, %d) (%.2f GiB), Target=[%d, %d) (%.2f GiB)\n",
                        out.remote_split_pos, (double) out.remote_cap * global_bytes_per_tok / 1073741824.0,
                        out.remote_split_pos, out.max_ctx, (double) out.global_local_cap * global_bytes_per_tok / 1073741824.0);
                }
            }
        }
    }

    const bool is_2dsd = (w.draft_mode == Inkling2DraftMode::TwoDSD && w.twodsd.loaded);
    const int n_slots = w.n_layer_eff;

    if (is_2dsd) {
        // 60-plane contiguous target prefix + tree scratch + per-node pools (ZERO duplicate KV!).
        // When n_seq_slots > 1, each tensor's outermost dimension scales by n_seq_slots so every
        // sequence slot `s` is a 100% contiguous slice (for zero-copy prefill/commit/draft views)
        // AND all slots share ONE batched Prelude/Core/Coda target pass via plane_eff = s*P + p.
        auto & tc = out.tree;
        tc.active = true;
        tc.is_slot_view = false;
        tc.n_seq_slots = n_seq_slots;
        tc.max_nodes = kInkling2TwoDSDMaxNodes;
        tc.n_depths = w.n_passes; // 3 readout depths {1, 2, 3}
        tc.prefix_len = 0;
        const int S = tc.max_nodes + 1; // last slot = scratch padding

        int pre_n_blk = 0;
        for (int ip = 0; ip < w.core_start; ++ip) {
            if (ip % w.attn_res_block == 0) ++pre_n_blk;
        }
        int core_n_blk = 0;
        for (int ip = w.core_start; ip < w.core_end; ++ip) {
            if (ip % w.attn_res_block == 0) ++core_n_blk;
        }

        const size_t n_tensors = (size_t) w.n_layer_phys * 6 + (size_t) pre_n_blk + (size_t) core_n_blk + 16;
        struct ggml_init_params gp = { n_tensors * ggml_tensor_overhead() + 8192, nullptr, true };
        out.ctx = ggml_init(gp);
        if (!out.ctx) {
            error = "inkling2_cache: ggml_init failed for 2DSD tree cache";
            return false;
        }

        tc.planes.resize(w.n_layer_phys, 1);
        tc.prefix_k.resize(w.n_layer_phys, nullptr);
        tc.prefix_v.resize(w.n_layer_phys, nullptr);
        tc.tree_k.resize(w.n_layer_phys, nullptr);
        tc.tree_v.resize(w.n_layer_phys, nullptr);
        tc.prefix_conv.resize(w.n_layer_phys, nullptr);
        tc.tree_conv.resize(w.n_layer_phys, nullptr);

        for (int ip = 0; ip < w.n_layer_phys; ++ip) {
            const int P = (ip < w.core_start) ? 1 : (ip < w.core_end ? w.n_passes : tc.n_depths);
            const int P_tot = P * n_seq_slots;
            tc.planes[ip] = P;
            const bool is_swa = w.layers[ip].is_swa;
            const int cap = is_swa ? out.swa_cap : out.max_ctx;
            tc.prefix_k[ip]    = ggml_new_tensor_4d(out.ctx, out.kv_type, w.head_dim, cap, w.n_head_kv, P_tot);
            tc.prefix_v[ip]    = ggml_new_tensor_4d(out.ctx, out.kv_type, w.head_dim, cap, w.n_head_kv, P_tot);
            tc.tree_k[ip]      = ggml_new_tensor_4d(out.ctx, out.kv_type, w.head_dim, S,   w.n_head_kv, P_tot);
            tc.tree_v[ip]      = ggml_new_tensor_4d(out.ctx, out.kv_type, w.head_dim, S,   w.n_head_kv, P_tot);
            tc.prefix_conv[ip] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, w.sconv_kernel - 1, w.sconv_channels, P_tot);
            tc.tree_conv[ip]   = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, w.sconv_channels, S, P_tot);
        }

        tc.pre_blocks.resize(pre_n_blk, nullptr);
        for (int b = 0; b < pre_n_blk; ++b) {
            tc.pre_blocks[b] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.n_embd, (int64_t) S * n_seq_slots);
        }
        tc.pre_partial = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.n_embd, (int64_t) S * n_seq_slots);
        tc.pre_tap     = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.n_embd, (int64_t) S * n_seq_slots);

        tc.core_blocks.resize(core_n_blk, nullptr);
        for (int b = 0; b < core_n_blk; ++b) {
            tc.core_blocks[b] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.n_embd, (int64_t) S * w.n_passes * n_seq_slots);
        }
        tc.core_partial = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.n_embd, (int64_t) S * w.n_passes * n_seq_slots);
        for (int ti = 0; ti < 3; ++ti) {
            tc.core_taps[ti] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.n_embd, (int64_t) S * w.n_passes * n_seq_slots);
        }
        tc.node_features = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, 5 * w.n_embd, (int64_t) S * tc.n_depths * n_seq_slots);
        out.last_taps_dev = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, w.n_embd, 4, 1);

        out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, target_backend);
        if (!out.buf) {
            error = "inkling2_cache: failed to allocate 2DSD 60-plane + tree cache buffer";
            free_inkling2_cache(out);
            return false;
        }

        // Disable Split-KV remote pool in 2DSD v1 (all 60 planes live on target GPU).
        out.remote_backend = nullptr;
        out.remote_split_pos = 0;
        out.remote_cap = 0;
        out.global_local_cap = out.max_ctx;

        // Zero-copy alias legacy 48-slot k_cache/v_cache/conv_state into tree.prefix_* (slot 0)
        // so single-slot AR fallback uses the exact same physical buffer with 0 extra VRAM!
        if (n_seq_slots == 1) {
            struct ggml_init_params agp = { ((size_t) n_slots * 3 + 8) * ggml_tensor_overhead() + 4096, nullptr, true };
            tc.alias_ctx = ggml_init(agp);
            out.k_cache.resize(n_slots, nullptr);
            out.v_cache.resize(n_slots, nullptr);
            out.conv_state.resize(n_slots, nullptr);
            out.conv_ring.assign(n_slots, nullptr);
            for (const auto & vi : w.schedule) {
                const int il = vi.slot;
                const int ip = vi.phys;
                const int plane = (ip < w.core_start) ? 0 : (ip < w.core_end ? vi.pass : (tc.n_depths - 1));
                ggml_tensor * pk = tc.prefix_k[ip];
                ggml_tensor * pv = tc.prefix_v[ip];
                ggml_tensor * pc = tc.prefix_conv[ip];
                out.k_cache[il] = ggml_new_tensor_3d(tc.alias_ctx, out.kv_type, w.head_dim, pk->ne[1], w.n_head_kv);
                out.k_cache[il]->data   = (char *) pk->data + (size_t) plane * pk->nb[3];
                out.k_cache[il]->buffer = pk->buffer;
                out.v_cache[il] = ggml_new_tensor_3d(tc.alias_ctx, out.kv_type, w.head_dim, pv->ne[1], w.n_head_kv);
                out.v_cache[il]->data   = (char *) pv->data + (size_t) plane * pv->nb[3];
                out.v_cache[il]->buffer = pv->buffer;
                out.conv_state[il] = ggml_new_tensor_2d(tc.alias_ctx, GGML_TYPE_F32, w.sconv_kernel - 1, w.sconv_channels);
                out.conv_state[il]->data   = (char *) pc->data + (size_t) plane * pc->nb[2];
                out.conv_state[il]->buffer = pc->buffer;
            }
        }

        // Allocate 5-layer loop-context drafter prefix KV cache (slack = 256 for multi-leaf span + block)
        const int d_layers = w.twodsd.n_layer;
        tc.draft_cap = out.max_ctx + 256;
        struct ggml_init_params dgp = { ((size_t) d_layers * 2 + 4) * ggml_tensor_overhead() + 1024, nullptr, true };
        out.dspark_kv_ctx = ggml_init(dgp);
        tc.draft_k.resize(d_layers, nullptr);
        tc.draft_v.resize(d_layers, nullptr);
        for (int i = 0; i < d_layers; ++i) {
            if (n_seq_slots > 1) {
                tc.draft_k[i] = ggml_new_tensor_4d(out.dspark_kv_ctx, GGML_TYPE_F16, w.twodsd.head_dim, tc.draft_cap, w.twodsd.n_head_kv, n_seq_slots);
                tc.draft_v[i] = ggml_new_tensor_4d(out.dspark_kv_ctx, GGML_TYPE_F16, w.twodsd.head_dim, tc.draft_cap, w.twodsd.n_head_kv, n_seq_slots);
            } else {
                tc.draft_k[i] = ggml_new_tensor_3d(out.dspark_kv_ctx, GGML_TYPE_F16, w.twodsd.head_dim, tc.draft_cap, w.twodsd.n_head_kv);
                tc.draft_v[i] = ggml_new_tensor_3d(out.dspark_kv_ctx, GGML_TYPE_F16, w.twodsd.head_dim, tc.draft_cap, w.twodsd.n_head_kv);
            }
        }
        out.dspark_kv_buf = ggml_backend_alloc_ctx_tensors(out.dspark_kv_ctx, target_backend);
        if (!out.dspark_kv_buf) {
            error = "inkling2_cache: failed to allocate 2DSD 5-layer drafter KV cache";
            free_inkling2_cache(out);
            return false;
        }
    } else {
        const size_t n_tensors = (size_t) n_slots * 4 + 8;
        struct ggml_init_params gp = { n_tensors * ggml_tensor_overhead() + 4096, nullptr, true };
        out.ctx = ggml_init(gp);
        if (!out.ctx) {
            error = "inkling2_cache: ggml_init failed";
            return false;
        }

        out.k_cache.resize(n_slots, nullptr);
        out.v_cache.resize(n_slots, nullptr);
        out.conv_state.resize(n_slots, nullptr);
        out.conv_ring.resize(n_slots, nullptr);

        for (int il = 0; il < n_slots; ++il) {
            const int ip = w.phys_layer_for_slot(il);
            const bool is_swa = w.layers[ip].is_swa;
            const int cap = is_swa ? out.swa_cap : out.global_local_cap;
            out.k_cache[il] = ggml_new_tensor_3d(out.ctx, out.kv_type, w.head_dim, cap, w.n_head_kv);
            out.v_cache[il] = ggml_new_tensor_3d(out.ctx, out.kv_type, w.head_dim, cap, w.n_head_kv);
            out.conv_state[il] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.sconv_kernel - 1, w.sconv_channels);
            out.conv_ring[il]  = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, w.sconv_kernel - 1, w.sconv_channels, kInkling2MaxVerify);
        }
        out.last_taps_dev = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, w.n_embd, 4, 1);

        out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, target_backend);
        if (!out.buf) {
            error = "inkling2_cache: failed to allocate target KV/conv cache buffer";
            free_inkling2_cache(out);
            return false;
        }

        if (out.remote_backend && out.remote_cap > 0) {
            struct ggml_init_params rgp = { ((size_t) n_global * 2 + 4) * ggml_tensor_overhead() + 1024, nullptr, true };
            out.remote_ctx = ggml_init(rgp);
            out.remote_k.resize(n_global, nullptr);
            out.remote_v.resize(n_global, nullptr);
            for (int g = 0; g < n_global; ++g) {
                out.remote_k[g] = ggml_new_tensor_3d(out.remote_ctx, out.kv_type, w.head_dim, out.remote_cap, w.n_head_kv);
                out.remote_v[g] = ggml_new_tensor_3d(out.remote_ctx, out.kv_type, w.head_dim, out.remote_cap, w.n_head_kv);
            }
            out.remote_buf = ggml_backend_alloc_ctx_tensors(out.remote_ctx, out.remote_backend);
            if (!out.remote_buf) {
                error = "inkling2_cache: failed to allocate remote Split-KV buffer on expert device";
                free_inkling2_cache(out);
                return false;
            }
        }

        if (w.dspark.loaded) {
            const int d_layers = w.dspark.n_layer;
            struct ggml_init_params dgp = { ((size_t) d_layers * 2 + 4) * ggml_tensor_overhead() + 1024, nullptr, true };
            out.dspark_kv_ctx = ggml_init(dgp);
            out.dspark_k.resize(d_layers, nullptr);
            out.dspark_v.resize(d_layers, nullptr);
            for (int i = 0; i < d_layers; ++i) {
                const int cap = out.max_ctx + w.dspark.block_size + 16;
                out.dspark_k[i] = ggml_new_tensor_3d(out.dspark_kv_ctx, GGML_TYPE_F16, w.dspark.head_dim, cap, w.dspark.n_head_kv);
                out.dspark_v[i] = ggml_new_tensor_3d(out.dspark_kv_ctx, GGML_TYPE_F16, w.dspark.head_dim, cap, w.dspark.n_head_kv);
            }
            out.dspark_kv_buf = ggml_backend_alloc_ctx_tensors(out.dspark_kv_ctx, target_backend);
            if (!out.dspark_kv_buf) {
                error = "inkling2_cache: failed to allocate DSpark draft KV cache";
                free_inkling2_cache(out);
                return false;
            }
        }
    }

    reset_inkling2_cache(out);
    std::fprintf(stderr,
        "[inkling2] cache ready: max_ctx=%d swa_cap=%d kv=%s target=%.2f MiB remote=%.2f MiB dspark=%.2f MiB\n",
        out.max_ctx, out.swa_cap, ggml_type_name(out.kv_type),
        (double) ggml_backend_buffer_get_size(out.buf) / 1048576.0,
        out.remote_buf ? (double) ggml_backend_buffer_get_size(out.remote_buf) / 1048576.0 : 0.0,
        out.dspark_kv_buf ? (double) ggml_backend_buffer_get_size(out.dspark_kv_buf) / 1048576.0 : 0.0);
    return true;
}

bool make_inkling2_slot_view(Inkling2Cache & multi,
                             const Inkling2Weights & w,
                             int seq_slot,
                             Inkling2Cache & view_out) {
    free_inkling2_cache(view_out);
    const auto & mtc = multi.tree;
    if (!mtc.active || seq_slot < 0 || seq_slot >= mtc.n_seq_slots) return false;

    view_out.kv_type          = multi.kv_type;
    view_out.max_ctx          = multi.max_ctx;
    view_out.swa_cap          = multi.swa_cap;
    view_out.global_local_cap = multi.global_local_cap;
    view_out.cur_pos          = 0;
    view_out.dspark_ctx_pos   = 0;

    auto & tc = view_out.tree;
    tc.active        = true;
    tc.is_slot_view  = true;
    tc.n_seq_slots   = 1;
    tc.max_nodes     = mtc.max_nodes;
    tc.n_depths      = mtc.n_depths;
    tc.prefix_len    = 0;
    tc.draft_pref_start = 0;
    tc.draft_cap     = mtc.draft_cap;
    tc.planes        = mtc.planes;

    const int S = tc.max_nodes + 1;
    const size_t n_tensors = (size_t) w.n_layer_phys * 6 + mtc.pre_blocks.size() + mtc.core_blocks.size()
                           + mtc.draft_k.size() * 2 + 16;
    struct ggml_init_params gp = { n_tensors * ggml_tensor_overhead() + 8192, nullptr, true };
    view_out.ctx = ggml_init(gp);
    if (!view_out.ctx) return false;

    auto alias_4d = [&](ggml_tensor * src, int64_t ne3, size_t off) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_4d(view_out.ctx, src->type, src->ne[0], src->ne[1], src->ne[2], ne3);
        t->data   = (char *) src->data + off;
        t->buffer = src->buffer;
        return t;
    };
    auto alias_3d = [&](ggml_tensor * src, int64_t ne2, size_t off) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_3d(view_out.ctx, src->type, src->ne[0], src->ne[1], ne2);
        t->data   = (char *) src->data + off;
        t->buffer = src->buffer;
        return t;
    };
    auto alias_2d = [&](ggml_tensor * src, int64_t ne1, size_t off) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_2d(view_out.ctx, src->type, src->ne[0], ne1);
        t->data   = (char *) src->data + off;
        t->buffer = src->buffer;
        return t;
    };

    tc.prefix_k.resize(w.n_layer_phys, nullptr);
    tc.prefix_v.resize(w.n_layer_phys, nullptr);
    tc.tree_k.resize(w.n_layer_phys, nullptr);
    tc.tree_v.resize(w.n_layer_phys, nullptr);
    tc.prefix_conv.resize(w.n_layer_phys, nullptr);
    tc.tree_conv.resize(w.n_layer_phys, nullptr);
    for (int ip = 0; ip < w.n_layer_phys; ++ip) {
        const int P = mtc.planes[ip];
        const size_t p_off = (size_t) seq_slot * P;
        tc.prefix_k[ip]    = alias_4d(mtc.prefix_k[ip],    P, p_off * mtc.prefix_k[ip]->nb[3]);
        tc.prefix_v[ip]    = alias_4d(mtc.prefix_v[ip],    P, p_off * mtc.prefix_v[ip]->nb[3]);
        tc.tree_k[ip]      = alias_4d(mtc.tree_k[ip],      P, p_off * mtc.tree_k[ip]->nb[3]);
        tc.tree_v[ip]      = alias_4d(mtc.tree_v[ip],      P, p_off * mtc.tree_v[ip]->nb[3]);
        tc.prefix_conv[ip] = alias_3d(mtc.prefix_conv[ip], P, p_off * mtc.prefix_conv[ip]->nb[2]);
        tc.tree_conv[ip]   = alias_3d(mtc.tree_conv[ip],   P, p_off * mtc.tree_conv[ip]->nb[2]);
    }

    tc.pre_blocks.resize(mtc.pre_blocks.size(), nullptr);
    for (size_t b = 0; b < mtc.pre_blocks.size(); ++b) {
        tc.pre_blocks[b] = alias_2d(mtc.pre_blocks[b], S, (size_t) seq_slot * S * mtc.pre_blocks[b]->nb[1]);
    }
    tc.pre_partial = alias_2d(mtc.pre_partial, S, (size_t) seq_slot * S * mtc.pre_partial->nb[1]);
    tc.pre_tap     = alias_2d(mtc.pre_tap,     S, (size_t) seq_slot * S * mtc.pre_tap->nb[1]);

    const int S3 = S * w.n_passes;
    tc.core_blocks.resize(mtc.core_blocks.size(), nullptr);
    for (size_t b = 0; b < mtc.core_blocks.size(); ++b) {
        tc.core_blocks[b] = alias_2d(mtc.core_blocks[b], S3, (size_t) seq_slot * S3 * mtc.core_blocks[b]->nb[1]);
    }
    tc.core_partial = alias_2d(mtc.core_partial, S3, (size_t) seq_slot * S3 * mtc.core_partial->nb[1]);
    for (int ti = 0; ti < 3; ++ti) {
        tc.core_taps[ti] = alias_2d(mtc.core_taps[ti], S3, (size_t) seq_slot * S3 * mtc.core_taps[ti]->nb[1]);
    }
    tc.node_features = alias_2d(mtc.node_features, S * tc.n_depths,
                                (size_t) seq_slot * S * tc.n_depths * mtc.node_features->nb[1]);

    const size_t d_layers = mtc.draft_k.size();
    tc.draft_k.resize(d_layers, nullptr);
    tc.draft_v.resize(d_layers, nullptr);
    for (size_t i = 0; i < d_layers; ++i) {
        const size_t dk_off = (mtc.n_seq_slots > 1) ? ((size_t) seq_slot * mtc.draft_k[i]->nb[3]) : 0;
        const size_t dv_off = (mtc.n_seq_slots > 1) ? ((size_t) seq_slot * mtc.draft_v[i]->nb[3]) : 0;
        tc.draft_k[i] = alias_3d(mtc.draft_k[i], mtc.draft_k[i]->ne[2], dk_off);
        tc.draft_v[i] = alias_3d(mtc.draft_v[i], mtc.draft_v[i]->ne[2], dv_off);
    }
    return true;
}

void reset_inkling2_cache(Inkling2Cache & cache) {
    cache.cur_pos = 0;
    cache.dspark_ctx_pos = 0;
    cache.tree.prefix_len = 0;
    cache.tree.draft_pref_start = 0;
    if (cache.buf || cache.tree.is_slot_view) {
        if (cache.tree.active) {
            for (ggml_tensor * pc : cache.tree.prefix_conv) {
                if (pc) ggml_backend_tensor_memset(pc, 0, 0, ggml_nbytes(pc));
            }
            for (ggml_tensor * tc : cache.tree.tree_conv) {
                if (tc) ggml_backend_tensor_memset(tc, 0, 0, ggml_nbytes(tc));
            }
            for (ggml_tensor * pb : cache.tree.pre_blocks) {
                if (pb) ggml_backend_tensor_memset(pb, 0, 0, ggml_nbytes(pb));
            }
            if (cache.tree.pre_partial) ggml_backend_tensor_memset(cache.tree.pre_partial, 0, 0, ggml_nbytes(cache.tree.pre_partial));
            if (cache.tree.pre_tap) ggml_backend_tensor_memset(cache.tree.pre_tap, 0, 0, ggml_nbytes(cache.tree.pre_tap));
            for (ggml_tensor * cb : cache.tree.core_blocks) {
                if (cb) ggml_backend_tensor_memset(cb, 0, 0, ggml_nbytes(cb));
            }
            if (cache.tree.core_partial) ggml_backend_tensor_memset(cache.tree.core_partial, 0, 0, ggml_nbytes(cache.tree.core_partial));
            for (ggml_tensor * ct : cache.tree.core_taps) {
                if (ct) ggml_backend_tensor_memset(ct, 0, 0, ggml_nbytes(ct));
            }
            if (cache.tree.node_features) ggml_backend_tensor_memset(cache.tree.node_features, 0, 0, ggml_nbytes(cache.tree.node_features));
        } else {
            // Zero conv states & last_taps_dev (KV slots are masked by kv_len/pos so only conv needs zeroing)
            for (ggml_tensor * st : cache.conv_state) {
                if (st) ggml_backend_tensor_memset(st, 0, 0, ggml_nbytes(st));
            }
        }
        if (cache.last_taps_dev) {
            ggml_backend_tensor_memset(cache.last_taps_dev, 0, 0, ggml_nbytes(cache.last_taps_dev));
        }
    }
}

void free_inkling2_cache(Inkling2Cache & cache) {
    free_inkling2_workspace(cache.decode_ws);
    for (auto & ws : cache.verify_ws) free_inkling2_workspace(ws);
    free_dspark_workspace(cache.dspark_ws);
    free_tree_cache(cache.tree);

    if (cache.dspark_kv_buf) { ggml_backend_buffer_free(cache.dspark_kv_buf); cache.dspark_kv_buf = nullptr; }
    if (cache.dspark_kv_ctx) { ggml_free(cache.dspark_kv_ctx); cache.dspark_kv_ctx = nullptr; }
    cache.dspark_k.clear();
    cache.dspark_v.clear();

    if (cache.remote_buf) { ggml_backend_buffer_free(cache.remote_buf); cache.remote_buf = nullptr; }
    if (cache.remote_ctx) { ggml_free(cache.remote_ctx); cache.remote_ctx = nullptr; }
    if (cache.split_cpu)  { ggml_backend_free(cache.split_cpu); cache.split_cpu = nullptr; }
    cache.remote_k.clear();
    cache.remote_v.clear();

    if (cache.buf) { ggml_backend_buffer_free(cache.buf); cache.buf = nullptr; }
    if (cache.ctx) { ggml_free(cache.ctx); cache.ctx = nullptr; }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.conv_state.clear();
    cache.conv_ring.clear();
    cache.last_taps_dev = nullptr;
    cache.cur_pos = 0;
    cache.dspark_ctx_pos = 0;
}

} // namespace luce::common
