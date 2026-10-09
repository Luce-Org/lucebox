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

bool create_qwen4exp_slot_states(ggml_backend_t backend, const Qwen4ExpWeights & w,
                                 int n_slots, Qwen4ExpSlotStates & out) {
    int n_linear = 0;
    for (int il = 0; il < w.n_layer; ++il) n_linear += !w.layers[il].is_full_attention;
    if (n_slots < 1 || n_linear == 0) return false;
    const int64_t conv_channels = 2 * static_cast<int64_t>(w.ssm_n_group) * w.ssm_d_state + w.ssm_d_inner;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (2 * (size_t) n_linear + 4);
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;
    for (int i = 0; i < n_linear; ++i) {
        out.ssm.push_back(ggml_new_tensor_4d(out.ctx, GGML_TYPE_F32, w.ssm_d_state, w.ssm_d_state,
                                             w.linear_value_heads, n_slots));
        out.conv.push_back(ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, w.ssm_d_conv - 1, conv_channels, n_slots));
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) { free_qwen4exp_slot_states(out); return false; }
    out.n_slots = n_slots;
    return true;
}

void free_qwen4exp_slot_states(Qwen4ExpSlotStates & s) {
    if (s.buf) ggml_backend_buffer_free(s.buf);
    if (s.ctx) ggml_free(s.ctx);
    s = {};
}

bool create_qwen4exp_cache(ggml_backend_t backend, const Qwen4ExpWeights & w,
                           int max_ctx, Qwen4ExpCache & out, bool mtp, int mtp_draft,
                           const Qwen4ExpSlotStates * slot_states, int state_slot, int remote_layers) {
    const Qwen4ExpCudaScope profile(w.gfx1151);
    // The QSA cell-id kernel is exact for positions below 2^24.
    if (max_ctx <= 0 || max_ctx >= (1 << 24)) {
        std::fprintf(stderr, "[qwen4exp] cache: context %d out of range (1..%d)\n",
                     max_ctx, (1 << 24) - 1);
        return false;
    }
    if (mtp_draft < 1 || mtp_draft > QWEN4EXP_MTP_MAX_DRAFT) return false;
    if (slot_states && (state_slot < 0 || state_slot >= slot_states->n_slots)) return false;
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

    if (remote_layers < 0 || remote_layers > (int) n_full || (remote_layers > 0 && !w.expert_backend)) return false;
    out.remote_layers = remote_layers;
    out.remote_backend = remote_layers > 0 ? w.expert_backend : nullptr;
    if (remote_layers > 0) {
        out.remote_ctx = ggml_init({4 * (size_t) remote_layers * ggml_tensor_overhead(), nullptr, true});
        if (!out.remote_ctx) return false;
    }

    // Packed attention reads groups of four keys even for an unaligned
    // logical context limit. Padding is masked by the causal cell IDs.
    const int64_t align = w.qsa ? 4 : 1;
    const int64_t kv_capacity = (static_cast<int64_t>(max_ctx) + align - 1) / align * align;
    for (size_t i = 0; i < n_full; ++i) {
        ggml_context * rows = out.remote((int) i) ? out.remote_ctx : out.ctx;
        out.attn_k[i] = ggml_new_tensor_3d(rows, kv_type,
            w.n_embd_head_k, kv_capacity, w.n_head_kv);
        out.attn_v[i] = ggml_new_tensor_3d(rows, kv_type,
            w.n_embd_head_v, kv_capacity, w.n_head_kv);
        const int il = out.full_layer_ids[i];
        const int ratio = il < (int) w.compress_ratios.size() ? w.compress_ratios[il] : 0;
        if (w.indexer_head_size > 0 && ratio > 0) {
            const int64_t max_blocks = (static_cast<int64_t>(max_ctx) + ratio - 1) / ratio;
            out.indexer_k[i] = ggml_new_tensor_2d(rows, GGML_TYPE_F32,
                w.indexer_head_size, max_blocks + 1);
            out.indexer_raw[i] = ggml_new_tensor_2d(rows, GGML_TYPE_F32,
                w.indexer_head_size, max_ctx);
        }
    }
    if (slot_states && slot_states->ssm.size() != n_linear) return false;
    for (size_t i = 0; i < n_linear; ++i) {
        // Recurrent state is independent of context length.
        if (slot_states) {
            ggml_tensor * ssm = slot_states->ssm[i];
            ggml_tensor * conv = slot_states->conv[i];
            out.ssm_state[i] = ggml_view_3d(out.ctx, ssm, S_v, S_v, H_v, ssm->nb[1], ssm->nb[2],
                                            (size_t) state_slot * ssm->nb[3]);
            out.conv_state[i] = ggml_view_2d(out.ctx, conv, kernel - 1, conv_channels, conv->nb[1],
                                             (size_t) state_slot * conv->nb[2]);
            continue;
        }
        out.ssm_state[i] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, S_v, S_v, H_v);
        out.conv_state[i] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, kernel - 1, conv_channels);
    }
    out.slot_states = slot_states;
    out.state_slot = slot_states ? state_slot : -1;
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
    if (out.remote_ctx) out.remote_buf = ggml_backend_alloc_ctx_tensors(out.remote_ctx, w.expert_backend);
    if (!out.buf || (out.remote_ctx && !out.remote_buf)) {
        free_qwen4exp_cache(out);
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
    if (workspace.sched) ggml_backend_sched_free(workspace.sched);
    if (workspace.alloc) ggml_gallocr_free(workspace.alloc);
    if (workspace.ctx) ggml_free(workspace.ctx);
    workspace = {};
}

void clear_qwen4exp_batched_decode_workspace(Qwen4ExpBatchedDecodeWorkspace & workspace) {
    clear_qwen4exp_decode_workspace(workspace);
    workspace = {};
}

// Every retained graph of the cache: the stable T=1 decode graph, the verify graph per width and the MTP graphs.
static void clear_retained_graphs(Qwen4ExpCache & c) {
    clear_qwen4exp_decode_workspace(c.decode_workspace);
    for (auto & ws : c.verify_workspace) clear_qwen4exp_decode_workspace(ws);
    clear_qwen4exp_decode_workspace(c.mtp_workspace);
    for (auto & ws : c.mtp_catchup_workspace) clear_qwen4exp_decode_workspace(ws);
    for (auto & ws : c.mtp_rank_workspace) clear_qwen4exp_decode_workspace(ws);
}

// A jump in the cache's position (reset, snapshot restore). The stable T=1 graph replays only from the position it
// stopped at and its captures hold the old sequence's recurrent and K/V state, so it goes. The verify and MTP graphs
// stay: they are keyed by their spans and upload every position-dependent input on each replay.
static void clear_position_bound_graphs(Qwen4ExpCache & c) {
    clear_qwen4exp_decode_workspace(c.decode_workspace);
}

void free_qwen4exp_cache(Qwen4ExpCache & c) {
    clear_retained_graphs(c);
    if (c.split_sched) { ggml_backend_sched_free(c.split_sched); c.split_sched = nullptr; }
    if (c.split_sched_short) { ggml_backend_sched_free(c.split_sched_short); c.split_sched_short = nullptr; }
    if (c.split_cpu) { ggml_backend_free(c.split_cpu); c.split_cpu = nullptr; }
    c.split_owner = nullptr;
    if (c.input_ring.buf) {
        ggml_backend_buffer_free(c.input_ring.buf);
        c.input_ring.buf = nullptr;
        c.input_ring.base = nullptr;
        c.input_ring.enabled = false;
    }
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    if (c.remote_buf) { ggml_backend_buffer_free(c.remote_buf); c.remote_buf = nullptr; }
    if (c.remote_ctx) { ggml_free(c.remote_ctx); c.remote_ctx = nullptr; }
    c.remote_layers = 0;
    c.remote_backend = nullptr;
    c.slot_states = nullptr;
    c.state_slot = -1;
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
    clear_position_bound_graphs(c);
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

namespace {
// One contiguous strip per KV head; recurrent tensors are copied in full.
// The same enumeration sizes, saves and restores the snapshot. The MTP draft
// layer's strips come last (`mtp`: visit them), so a snapshot's trunk is the
// same leading strips whether or not the cache it came from has the layer.
template<class F>
void snapshot_strips(const Qwen4ExpCache & c, int pos, int blocks, bool mtp, F visit) {
    auto prefix = [&](ggml_tensor * t, int rows) {
        if (!t || rows <= 0) return;
        for (int64_t h = 0; h < t->ne[2]; ++h) visit(t, h * t->nb[2], rows * t->nb[1]);
    };
    for (auto * t : c.attn_k) prefix(t, pos);
    for (auto * t : c.attn_v) prefix(t, pos);
    for (auto * t : c.indexer_raw) prefix(t, pos);
    for (auto * t : c.indexer_k) prefix(t, blocks);
    for (const auto * group : {&c.ssm_state, &c.conv_state, &c.ple_conv_state}) {
        for (auto * t : *group) if (t) visit(t, 0, ggml_nbytes(t));
    }
    if (!mtp) return;
    prefix(c.mtp_k, pos - 1);
    prefix(c.mtp_v, pos - 1);
    if (c.mtp_prev_hidden) visit(c.mtp_prev_hidden, 0, ggml_nbytes(c.mtp_prev_hidden));
}
} // namespace

size_t qwen4exp_snapshot_bytes(ggml_backend_t store, const Qwen4ExpCache & c, int tokens, size_t * host_bytes) {
    if (host_bytes) *host_bytes = 0;
    const int pos = std::clamp(tokens, 0, c.max_ctx);
    if (!pos) return 0;
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(store));
    size_t bytes = 0, count = 0;
    // The supported QSA layout pools four rows per block. Dense prefill can
    // have fewer valid blocks; this bound also covers its later completion.
    snapshot_strips(c, pos, pos / 4, true, [&](ggml_tensor *, size_t, size_t n) {
        bytes += (n + alignment - 1) / alignment * alignment;
        ++count;
    });
    if (host_bytes) *host_bytes = (2 * count + 1) * ggml_tensor_overhead() +
        count * sizeof(std::pair<ggml_tensor *, ggml_tensor *>);
    return bytes;
}

void free_qwen4exp_snapshot(Qwen4ExpSnapshot & s) {
    if (s.buf) ggml_backend_buffer_free(s.buf);
    if (s.ctx) ggml_free(s.ctx);
    s = {};
}

bool save_qwen4exp_snapshot(ggml_backend_t backend, ggml_backend_t store, const Qwen4ExpCache & c,
                            Qwen4ExpSnapshot & s) {
    free_qwen4exp_snapshot(s);
    if (c.cur_pos <= 0 || (c.mtp_k && c.mtp_prev_pos != c.cur_pos - 1)) return false;
    size_t count = 0;
    snapshot_strips(c, c.cur_pos, c.indexer_blocks, true, [&](ggml_tensor *, size_t, size_t) { ++count; });
    s.ctx = ggml_init({(2 * count + 1) * ggml_tensor_overhead(), nullptr, true});
    if (!s.ctx) return false;
    s.strips.reserve(count);
    snapshot_strips(c, c.cur_pos, c.indexer_blocks, true, [&](ggml_tensor * t, size_t off, size_t bytes) {
        const int64_t n = bytes / ggml_type_size(t->type) * ggml_blck_size(t->type);
        auto * live = ggml_view_1d(s.ctx, t, n, off);
        auto * copy = ggml_new_tensor_1d(s.ctx, t->type, n);
        s.strips.emplace_back(live, copy);
    });
    s.buf = ggml_backend_alloc_ctx_tensors(s.ctx, store);
    if (!s.buf) { free_qwen4exp_snapshot(s); return false; }
    if (c.remote_backend) ggml_backend_synchronize(c.remote_backend);   // its layers' last K/V writes
    for (auto [live, copy] : s.strips) ggml_backend_tensor_copy_async(backend, store, live, copy);
    ggml_backend_synchronize(backend);
    s.cur_pos = c.cur_pos;
    s.indexer_blocks = c.indexer_blocks;
    s.mtp = c.mtp_k != nullptr;
    s.mtp_prev_pos = c.mtp_prev_pos;
    s.kv_bucket_base = c.kv_bucket_base;
    s.ple_prev = c.ple_prev;
    return true;
}

bool restore_qwen4exp_snapshot(ggml_backend_t backend, const Qwen4ExpSnapshot & s, Qwen4ExpCache & c) {
    if (!s.buf || !c.buf || s.cur_pos <= 0 || s.cur_pos > c.max_ctx) return false;
    // The target's strips at the snapshot's position, in the order they were saved. The draft layer's come only when
    // both caches have the layer; a snapshot that has it and a target that does not leave its trailing strips out.
    const bool mtp = s.mtp && c.mtp_k;
    struct Strip { ggml_tensor * t; size_t off, bytes; };
    std::vector<Strip> target;
    target.reserve(s.strips.size());
    snapshot_strips(c, s.cur_pos, s.indexer_blocks, mtp, [&](ggml_tensor * t, size_t off, size_t bytes) {
        target.push_back({t, off, bytes});
    });
    if (target.size() > s.strips.size() || (target.size() < s.strips.size()) != (s.mtp && !mtp)) return false;
    for (size_t i = 0; i < target.size(); ++i) {
        const ggml_tensor * copy = s.strips[i].second;
        if (target[i].t->type != copy->type || target[i].bytes != ggml_nbytes(copy)) return false;
    }
    ggml_context * views = ggml_init({target.size() * ggml_tensor_overhead(), nullptr, true});
    if (!views) return false;
    ggml_backend_synchronize(backend); // finish rollback before clearing its source/destination buffer
    if (c.remote_backend) ggml_backend_synchronize(c.remote_backend);
    clear_position_bound_graphs(c);
    // Clear masked suffixes too: stable QSA scores the entire bucket. Stacked slot
    // states are not in c.buf; the snapshot rewrites this slot's slabs in full.
    ggml_backend_buffer_clear(c.buf, 0);
    if (c.remote_buf) ggml_backend_buffer_clear(c.remote_buf, 0);
    for (size_t i = 0; i < target.size(); ++i) {
        ggml_tensor * copy = s.strips[i].second;
        ggml_tensor * live = ggml_view_1d(views, target[i].t, ggml_nelements(copy), target[i].off);
        ggml_backend_view_init(live);
        ggml_backend_tensor_copy_async(backend, backend, copy, live);
    }
    ggml_backend_synchronize(backend);
    ggml_free(views);
    c.cur_pos = s.cur_pos;
    c.indexer_blocks = s.indexer_blocks;
    c.mtp_prev_pos = mtp ? s.mtp_prev_pos : -1;   // no draft state: its request decodes without drafts
    c.kv_bucket_base = s.kv_bucket_base;
    c.ple_prev = s.ple_prev;
    c.spec_pos = -1;
    c.spec_tokens = 0;
    return true;
}

}  // namespace luce::common
