#include "qwen4exp_cache.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>

namespace luce::common {

namespace {

// The ring engages on the gfx1151 iGPU, which reads the pinned host buffer
// type in place (the device stays reported as a GPU for every other model).
bool qwen4exp_uma_ring_supported(ggml_backend_t backend) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) return false;
    if (!ggml_backend_cuda_qwen4exp_supported(backend)) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    return dev != nullptr && ggml_backend_dev_host_buffer_type(dev) != nullptr;
}

}  // namespace

bool create_qwen4exp_cache(ggml_backend_t backend, const Qwen4ExpWeights & w,
                           int max_ctx, Qwen4ExpCache & out, bool mtp, int mtp_draft) {
    const Qwen4ExpCudaScope profile(w.gfx1151);
    // The QSA cell-id kernel is exact for positions below 2^24.
    if (max_ctx <= 0 || max_ctx >= (1 << 24)) {
        std::fprintf(stderr, "[qwen4exp] cache: context %d out of range (1..%d)\n",
                     max_ctx, (1 << 24) - 1);
        return false;
    }
    if (mtp_draft < 1 || mtp_draft > QWEN4EXP_MTP_MAX_DRAFT) return false;
    constexpr ggml_type kv_type = GGML_TYPE_F16;

    out.full_layer_ids.clear();
    out.linear_layer_ids.clear();
    for (int il = 0; il < w.n_layer; ++il) {
        if (w.layers[il].is_full_attention) out.full_layer_ids.push_back(il);
        else                                 out.linear_layer_ids.push_back(il);
    }
    const size_t n_full   = out.full_layer_ids.size();
    const size_t n_linear = out.linear_layer_ids.size();
    if (n_full + n_linear != static_cast<size_t>(w.n_layer)) return false;

    // conv_channels = 2 * n_group * d_state + d_inner, the width of the fused
    // q|k|v projection the depthwise conv runs over.
    const int64_t conv_channels =
        2 * static_cast<int64_t>(w.ssm_n_group) * w.ssm_d_state + w.ssm_d_inner;
    const int64_t S_v = w.ssm_d_state;
    const int64_t H_v = w.linear_value_heads;
    const int64_t kernel = w.ssm_d_conv;

    if (w.n_embd_head_k != 256 || w.n_head_kv != 2 || S_v != 128 ||
        kernel < 2 || conv_channels <= 0) {
        std::fprintf(stderr,
            "[qwen4exp] cache: unsupported state shape (head=%d kv=%d d_state=%lld conv=%lld)\n",
            w.n_embd_head_k, w.n_head_kv, static_cast<long long>(S_v),
            static_cast<long long>(conv_channels));
        return false;
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (static_cast<size_t>(w.n_layer) * (8 + 3 * QWEN4EXP_MTP_MAX_VERIFY) + 16) + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.attn_k.assign(n_full, nullptr);
    out.attn_v.assign(n_full, nullptr);
    out.indexer_k.assign(n_full, nullptr);
    out.indexer_raw.assign(n_full, nullptr);
    out.ssm_state.assign(n_linear, nullptr);
    out.conv_state.assign(n_linear, nullptr);

    // PLE conv history (only PLE layers carry one).
    const int64_t hc_dim  = static_cast<int64_t>(w.n_embd) * w.n_hc;
    const int64_t ple_hist = static_cast<int64_t>(w.ple_conv_kernel - 1) * w.ple_ngram_size;
    out.ple_layer_ids.clear();
    if (ple_hist > 0) {
        for (int il = 0; il < w.n_layer; ++il) {
            if (w.layers[il].is_ple) out.ple_layer_ids.push_back(il);
        }
    }
    out.ple_conv_state.assign(out.ple_layer_ids.size(), nullptr);
    for (size_t i = 0; i < out.ple_layer_ids.size(); ++i) {
        out.ple_conv_state[i] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, ple_hist, hc_dim);
    }

    // Packed attention reads groups of four keys even for an unaligned
    // logical context limit. Padding is masked by the causal cell IDs.
    const int64_t align = profile.optimized ? 4 : 1;
    const int64_t kv_capacity = (static_cast<int64_t>(max_ctx) + align - 1) / align * align;
    for (size_t i = 0; i < n_full; ++i) {
        out.attn_k[i] = ggml_new_tensor_3d(out.ctx, kv_type,
            w.n_embd_head_k, kv_capacity, w.n_head_kv);
        out.attn_v[i] = ggml_new_tensor_3d(out.ctx, kv_type,
            w.n_embd_head_v, kv_capacity, w.n_head_kv);
        const int il = out.full_layer_ids[i];
        const int ratio = il < (int) w.compress_ratios.size() ? w.compress_ratios[il] : 0;
        if (w.indexer_head_size > 0 && ratio > 0) {
            const int64_t max_blocks = (static_cast<int64_t>(max_ctx) + ratio - 1) / ratio;
            out.indexer_k[i] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                w.indexer_head_size, max_blocks + 1);
            out.indexer_raw[i] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                w.indexer_head_size, max_ctx);
        }
    }
    for (size_t i = 0; i < n_linear; ++i) {
        // Recurrent state is independent of context length.
        out.ssm_state[i] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, S_v, S_v, H_v);
        out.conv_state[i] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, kernel - 1, conv_channels);
    }
    out.spec_ssm.clear(); out.spec_conv.clear();
    out.spec_ssm_rows.clear(); out.spec_conv_rows.clear();
    out.spec_ple_rows = {};
    out.mtp_draft = mtp_draft;
    out.spec_ple = nullptr;
    if (mtp && w.mtp_eh_proj) {   // the MTP draft layer's own K/V (dense attention, no indexer) and the verify rollback
        out.mtp_k = ggml_new_tensor_3d(out.ctx, kv_type, w.n_embd_head_k, kv_capacity, w.n_head_kv);
        out.mtp_v = ggml_new_tensor_3d(out.ctx, kv_type, w.n_embd_head_v, kv_capacity, w.n_head_kv);
        out.mtp_prev_hidden = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.n_embd * w.n_hc, 1);
        out.mtp_chain_hidden = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hc_dim, 1);
        out.mtp_chain_ids = ggml_new_tensor_1d(out.ctx, GGML_TYPE_I32, out.mtp_draft);
        const int count = out.mtp_draft + 1;
        for (size_t i = 0; i < n_linear; ++i) {
            ggml_tensor * states = ggml_new_tensor_4d(out.ctx, GGML_TYPE_F32, S_v, S_v, H_v, count);
            ggml_tensor * conv = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, kernel - 1, conv_channels, count);
            out.spec_ssm.push_back(states);
            out.spec_conv.push_back(conv);
            Qwen4ExpCache::SpecRows sr{}, cr{};
            for (int t = 0; t < count; ++t) {
                sr[t] = ggml_view_3d(out.ctx, states, S_v, S_v, H_v, states->nb[1], states->nb[2], t * states->nb[3]);
                cr[t] = ggml_view_2d(out.ctx, conv, kernel - 1, conv_channels, conv->nb[1], t * conv->nb[2]);
            }
            out.spec_ssm_rows.push_back(sr);
            out.spec_conv_rows.push_back(cr);
        }
        if (!out.ple_conv_state.empty()) {
            out.spec_ple = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, ple_hist, hc_dim, count);
            for (int t = 0; t < count; ++t) {
                out.spec_ple_rows[t] = ggml_view_2d(out.ctx, out.spec_ple, ple_hist, hc_dim,
                    out.spec_ple->nb[1], t * out.spec_ple->nb[2]);
            }
        }
        size_t rollback_bytes = out.spec_ple ? ggml_nbytes(out.spec_ple) : 0;
        for (auto * t : out.spec_ssm) rollback_bytes += ggml_nbytes(t);
        for (auto * t : out.spec_conv) rollback_bytes += ggml_nbytes(t);
        std::fprintf(stderr, "[qwen4exp-mtp] k=%d rollback_bytes=%zu (%.3f MiB), draft_kv_bytes=%zu; allocated once\n",
            out.mtp_draft, rollback_bytes, rollback_bytes / (1024.0 * 1024.0),
            ggml_nbytes(out.mtp_k) + ggml_nbytes(out.mtp_v));
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // Stable scoring converts the whole bucket, including its masked suffix.
    for (ggml_tensor * t : out.indexer_k) {
        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }

    out.max_ctx = max_ctx;
    out.cur_pos = 0;
    out.indexer_blocks = 0;
    out.input_ring.enabled = qwen4exp_uma_ring_supported(backend);
    if (out.input_ring.enabled) {
        std::fprintf(stderr,
            "[qwen4exp] cache: integrated GPU detected, graph inputs will be "
            "ring-buffered in pinned host memory\n");
    }

    // A fresh cache must start from zero recurrent state, not whatever the
    // backend buffer happened to contain.
    reset_qwen4exp_state(backend, out);

    const size_t kv_bytes_per_token =
        n_full * (static_cast<size_t>(w.n_embd_head_k) + w.n_embd_head_v) *
        w.n_head_kv * ggml_type_size(kv_type) / ggml_blck_size(kv_type);
    std::fprintf(stderr,
        "[qwen4exp] cache: %zu full + %zu linear layers, kv=%s %.2f MiB @ ctx=%d, "
        "ssm_state %.1f MiB, conv_state %.1f MiB\n",
        n_full, n_linear, ggml_type_name(kv_type),
        kv_bytes_per_token * static_cast<size_t>(max_ctx) / (1024.0 * 1024.0),
        max_ctx,
        n_linear * static_cast<double>(S_v * S_v * H_v * 4) / (1024.0 * 1024.0),
        n_linear * static_cast<double>((kernel - 1) * conv_channels * 4) / (1024.0 * 1024.0));
    return true;
}

void clear_qwen4exp_decode_workspace(Qwen4ExpDecodeWorkspace & workspace) {
    if (workspace.ctx && workspace.backend) {
        ggml_backend_cuda_graph_invalidate_range(workspace.backend,
            ggml_get_mem_buffer(workspace.ctx), ggml_get_mem_size(workspace.ctx));
    }
    if (workspace.alloc) ggml_gallocr_free(workspace.alloc);
    if (workspace.ctx) ggml_free(workspace.ctx);
    workspace = {};
}

void free_qwen4exp_cache(Qwen4ExpCache & c) {
    clear_qwen4exp_decode_workspace(c.decode_workspace);
    clear_qwen4exp_decode_workspace(c.verify_workspace);
    clear_qwen4exp_decode_workspace(c.mtp_workspace);
    if (c.input_ring.buf) {
        ggml_backend_buffer_free(c.input_ring.buf);
        c.input_ring.buf = nullptr;
        c.input_ring.base = nullptr;
        c.input_ring.enabled = false;
    }
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.mtp_k = c.mtp_v = nullptr;
    c.mtp_prev_hidden = nullptr;
    c.mtp_chain_hidden = c.mtp_chain_ids = nullptr;
    c.mtp_prev_pos = -1;
    c.spec_ssm.clear();
    c.spec_ssm_rows.clear();
    c.spec_conv_rows.clear();
    c.spec_conv.clear();
    c.spec_ple = nullptr;
    c.spec_ple_rows = {};
    for (auto & tail : c.spec_ple_prev) tail.clear();
    c.indexer_k.clear();
    c.indexer_raw.clear();
    c.ssm_state.clear();
    c.conv_state.clear();
    c.ple_conv_state.clear();
    c.ple_layer_ids.clear();
    c.full_layer_ids.clear();
    c.linear_layer_ids.clear();
    c.ple_prev.clear();
    c.cur_pos = 0;
    c.max_ctx = 0;
}

void reset_qwen4exp_state(ggml_backend_t backend, Qwen4ExpCache & c) {
    // A rejected final MTP verify leaves rollback copies queued on the backend
    // stream; the memsets below run on another stream and must not race them.
    ggml_backend_synchronize(backend);
    // A reset makes any stable T=1 graph's captured recurrent/KV state stale.
    // Batched graphs are rebuilt each call and use a separate shared arena.
    clear_qwen4exp_decode_workspace(c.decode_workspace);
    for (ggml_tensor * t : c.ssm_state) {
        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    for (ggml_tensor * t : c.conv_state) {
        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    for (ggml_tensor * t : c.ple_conv_state) {
        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    c.cur_pos = 0;
    c.spec_pos = -1;
    c.mtp_prev_pos = -1;
    c.spec_tokens = 0;
    c.indexer_blocks = 0;
    c.kv_bucket_base = 0;
    c.ple_prev.clear();
}

}  // namespace luce::common
