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

bool create_inkling2_cache(ggml_backend_t target_backend,
                           ggml_backend_t expert_backend,
                           const Inkling2Weights & w,
                           int max_ctx,
                           int max_chunk,
                           ggml_type kv_type,
                           Inkling2Cache & out,
                           std::string & error,
                           int fixed_remote_split_pos) {
    free_inkling2_cache(out);
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

    const int n_slots = w.n_layer_eff;
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
        // Draft KV: layer 0 & 2 are local (1024 + block_size), layer 1 is global (max_ctx + block_size)
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

    reset_inkling2_cache(out);
    std::fprintf(stderr,
        "[inkling2] cache ready: max_ctx=%d swa_cap=%d kv=%s target=%.2f MiB remote=%.2f MiB dspark=%.2f MiB\n",
        out.max_ctx, out.swa_cap, ggml_type_name(out.kv_type),
        (double) ggml_backend_buffer_get_size(out.buf) / 1048576.0,
        out.remote_buf ? (double) ggml_backend_buffer_get_size(out.remote_buf) / 1048576.0 : 0.0,
        out.dspark_kv_buf ? (double) ggml_backend_buffer_get_size(out.dspark_kv_buf) / 1048576.0 : 0.0);
    return true;
}

void reset_inkling2_cache(Inkling2Cache & cache) {
    cache.cur_pos = 0;
    cache.dspark_ctx_pos = 0;
    if (cache.buf) {
        // Zero conv states & last_taps_dev (KV slots are masked by kv_len/pos so only conv needs zeroing)
        for (ggml_tensor * st : cache.conv_state) {
            if (st) ggml_backend_tensor_memset(st, 0, 0, ggml_nbytes(st));
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
