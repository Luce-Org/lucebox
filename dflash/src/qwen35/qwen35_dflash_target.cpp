// Qwen35DFlashTarget — DFlashTarget adapter for qwen35 hybrid models.

#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "step_graph.h"
#include "attn_masks.h"
#include "device_runtime.h"  // cudaStream_t, cudaMemcpyAsync, cudaMemcpy2DAsync

#include <chrono>
#include <cstdio>
#include <cstdlib>

// ggml-cuda dequantize helper (used to widen F16/Q8_0 ssm_intermediate slots
// back to F32 for cache_.ssm_state).  Same trick as test_dflash.cpp and
// dflash_feature_ring.cpp — the symbol lives in ggml-cuda/convert.cuh and has
// no public header, so forward-declare the typedef here.
using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
extern "C++" to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

namespace dflash27b {

#ifdef DFLASH_VERIFY_PROFILE
// Per-call profiler for verify_batch.  Enabled by -DDFLASH_VERIFY_PROFILE=1.
// Host-side wall-clock around ggml_backend_* calls IS the GPU+sync latency
// because every set/get/compute internally calls cudaStreamSynchronize.
namespace {
inline bool verify_profile_enabled() {
    static const bool on = (std::getenv("DFLASH_VERIFY_PROFILE") != nullptr);
    return on;
}

using vprof_clock = std::chrono::steady_clock;
inline double vprof_ms_since(vprof_clock::time_point t0) {
    auto t1 = vprof_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
}  // namespace
#endif  // DFLASH_VERIFY_PROFILE

Qwen35DFlashTarget::~Qwen35DFlashTarget() {
#ifdef DFLASH_VERIFY_PROFILE
    if (verify_profile_enabled() && vprof_n_calls_ > 0) {
        const double inv = 1.0 / (double)vprof_n_calls_;
        std::fprintf(stderr,
            "[verify_prof summary calls=%lld "
            "avg_set=%.3f avg_compute=%.3f avg_get_hidden=%.3f "
            "avg_get_hpre=%.3f avg_get_argmax=%.3f avg_total=%.3f "
            "(sum_set=%.1f sum_compute=%.1f sum_get_hidden=%.1f "
            "sum_get_hpre=%.1f sum_get_argmax=%.1f sum_total=%.1f ms)]\n",
            (long long)vprof_n_calls_,
            vprof_sum_set_ * inv,
            vprof_sum_compute_ * inv,
            vprof_sum_get_hidden_ * inv,
            vprof_sum_get_hpre_ * inv,
            vprof_sum_get_argmax_ * inv,
            vprof_sum_total_ * inv,
            vprof_sum_set_, vprof_sum_compute_, vprof_sum_get_hidden_,
            vprof_sum_get_hpre_, vprof_sum_get_argmax_, vprof_sum_total_);
    }
#endif  // DFLASH_VERIFY_PROFILE
    step_graph_destroy(proj_sg_);
    if (rollback_stream_) { cudaStreamDestroy(rollback_stream_); rollback_stream_ = nullptr; }
}

Qwen35DFlashTarget::Qwen35DFlashTarget(
        TargetWeights & w,
        TargetCache & cache,
        ggml_backend_t backend,
        StepGraph & sg,
        int kq_stride_pad,
        int fa_window)
    : w_(w), cache_(cache), backend_(backend), sg_(sg),
      kq_stride_pad_(kq_stride_pad), fa_window_(fa_window) {
    capture_ids_.assign(w.capture_layer_ids,
                        w.capture_layer_ids + w.n_capture_layers);
}

bool Qwen35DFlashTarget::verify_batch(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<int32_t> * all_argmax) {
    const int n_tokens = (int)tokens.size();
    if (n_tokens <= 0) return false;

    const int hidden = w_.n_embd;
    const bool need_mask = (kq_stride_pad_ > KQ_MASK_PAD) || (n_tokens > 1);

    // Per-position DeltaNet intermediate capture is only safe when:
    //   1. The caller (MTP chain runner) opted in via enable_chain_capture(true).
    //   2. The cache buffers actually exist (migrate_prefill_cache ran).
    //   3. n_tokens fits in the pre-allocated cache.  The conv_input cache is
    //      [(d_conv-1) + max_verify_tokens, conv_ch, 1] and the in-graph
    //      ggml_view_3d into it asserts when n_tokens > max_verify_tokens
    //      (e.g. 512-token prefill chunks vs 16-slot cache).  The
    //      ssm_intermediate ggml_cpy also requires n_tokens == max_verify_tokens
    //      after the matching dst-side view (see qwen35_target_graph.cpp).
    int max_verify_tokens = 0;
    if (!cache_.ssm_intermediate.empty() && cache_.ssm_intermediate[0] != nullptr) {
        max_verify_tokens = (int)cache_.ssm_intermediate[0]->ne[3];
    }
    const bool capture_intermediate =
        chain_capture_enabled_ &&
        max_verify_tokens > 0 &&
        n_tokens <= max_verify_tokens;

    if (!build_target_step(sg_, w_, cache_, backend_,
                           /*kv_start=*/base_pos, n_tokens,
                           need_mask, /*capture=*/true,
                           /*capture_delta_intermediate=*/capture_intermediate,
                           fa_window_,
                           /*last_token_logits_only=*/false,
                           kq_stride_pad_,
                           /*capture_all_norm_hidden=*/capture_hidden_seq_)) {
        std::fprintf(stderr, "verify_batch: build_target_step failed (base=%d n=%d)\n",
                     base_pos, n_tokens);
        return false;
    }

#ifdef DFLASH_VERIFY_PROFILE
    // Per-call profiling state (DFLASH_VERIFY_PROFILE=1).
    const bool vprof_on = verify_profile_enabled();
    double vp_set = 0.0, vp_compute = 0.0;
    double vp_get_hidden = 0.0, vp_get_hpre = 0.0, vp_get_argmax = 0.0;
    vprof_clock::time_point vp_t_total =
        vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
    vprof_clock::time_point vp_t0;
#endif  // DFLASH_VERIFY_PROFILE

    // Embed input tokens and fill positions.
    std::vector<float> embed((size_t)n_tokens * hidden);
    if (!w_.embedder.embed(tokens.data(), n_tokens, embed.data())) {
        std::fprintf(stderr, "verify_batch: embed failed (n=%d)\n", n_tokens);
        return false;
    }
#ifdef DFLASH_VERIFY_PROFILE
    vp_t0 = vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
#endif
    ggml_backend_tensor_set(sg_.inp_embed, embed.data(), 0,
                            sizeof(float) * embed.size());
#ifdef DFLASH_VERIFY_PROFILE
    if (vprof_on) vp_set += vprof_ms_since(vp_t0);
#endif

    // Qwen35 uses interleaved positions: 4 ints per token.
    std::vector<int32_t> pos(4 * n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        pos[4 * i + 0] = base_pos + i;
        pos[4 * i + 1] = base_pos + i;
        pos[4 * i + 2] = base_pos + i;
        pos[4 * i + 3] = 0;
    }
#ifdef DFLASH_VERIFY_PROFILE
    vp_t0 = vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
#endif
    ggml_backend_tensor_set(sg_.positions, pos.data(), 0,
                            sizeof(int32_t) * pos.size());
#ifdef DFLASH_VERIFY_PROFILE
    if (vprof_on) vp_set += vprof_ms_since(vp_t0);
#endif

    // Populate the causal attention mask. The mask buffer is freshly allocated
    // by build_target_step (uninitialized memory); without this set, attention
    // reads garbage and ggml_argmax over the resulting logits returns -1 for
    // non-last positions, breaking the chain runner's recommit path.
    if (sg_.attn_mask) {
        const int win_start = (fa_window_ > 0 && base_pos > fa_window_)
                                  ? (base_pos - fa_window_) : 0;
        const int kv_len = base_pos + n_tokens - win_start;
        std::vector<uint16_t> mask_buf;
        build_causal_mask(mask_buf, kv_len, n_tokens, base_pos,
                          kq_stride_pad_, win_start);
#ifdef DFLASH_VERIFY_PROFILE
        vp_t0 = vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
#endif
        ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
#ifdef DFLASH_VERIFY_PROFILE
        if (vprof_on) vp_set += vprof_ms_since(vp_t0);
#endif
    }

    // Mask was already filled earlier (see the mask-fill block above); no
    // duplicate fill here. Just compute and profile.
#ifdef DFLASH_VERIFY_PROFILE
    vp_t0 = vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
#endif
    auto st = ggml_backend_graph_compute(backend_, sg_.gf);
#ifdef DFLASH_VERIFY_PROFILE
    if (vprof_on) vp_compute = vprof_ms_since(vp_t0);
#endif
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "verify_batch: compute failed (status=%d base_pos=%d n_tokens=%d)\n",
                     (int)st, base_pos, n_tokens);
        return false;
    }

    // Read argmax for every position. The chain runner needs all_argmax to
    // accept/reject per-position drafts; for non-chain callers (all_argmax
    // null) we still need last_tok. Single read covers both.
#ifdef DFLASH_VERIFY_PROFILE
    vp_t0 = vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
#endif
    std::vector<int32_t> argmax_buf(n_tokens);
    ggml_backend_tensor_get(sg_.argmax_tokens, argmax_buf.data(), 0,
                            sizeof(int32_t) * n_tokens);
    last_tok = argmax_buf[n_tokens - 1];

    if (all_argmax) {
        *all_argmax = std::move(argmax_buf);
    }
#ifdef DFLASH_VERIFY_PROFILE
    if (vprof_on) vp_get_argmax = vprof_ms_since(vp_t0);
#endif

    // Copy the last token's post-norm hidden to a CPU buffer so the MTP module
    // can call last_hidden() before the next graph_compute overwrites it.
    if (sg_.last_norm_hidden) {
        last_hidden_cpu_.resize(hidden);
#ifdef DFLASH_VERIFY_PROFILE
        vp_t0 = vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
#endif
        ggml_backend_tensor_get(sg_.last_norm_hidden, last_hidden_cpu_.data(),
                                0, sizeof(float) * hidden);
#ifdef DFLASH_VERIFY_PROFILE
        if (vprof_on) vp_get_hidden += vprof_ms_since(vp_t0);
#endif
    }

    // Copy the full [n_tokens, n_embd] post-norm hidden sequence so the
    // Qwen3.6 MTP warm_head_kv() can read per-position hiddens during prefill.
    //
    // R8 audit (Phase A): this branch overwrites last_hidden_seq_cpu_ via
    // .resize() + tensor_get on EVERY verify_batch call — including the
    // recommit path at mtp_chain_runner.cpp:206 (recommit calls verify_batch
    // which lands here).  So `hidden_at_pos(base_pos-1)` on the chain's next
    // iteration reads the fresh hiddens from the recommit, not a stale slice.
    // Hypothesis from the Phase A brief (stale recommit hiddens as cause of
    // the 71.6% per-step accept ceiling) is NOT supported by this code path.
    if (sg_.all_norm_hidden) {
        // LAST_ROW_ONLY (decode mode): download only row n_tokens-1 of the
        // [n_tokens, n_embd] hidden tensors.  The chain's only consumer is
        // hidden_at_pos(base_pos - 1) on the NEXT verify, where the chain's
        // base_pos = (this verify's base_pos + n_tokens), so it asks for
        // abs_pos = (base_pos + n_tokens - 1) — exactly the row we keep.
        // We stash that single row at offset 0 of last_hidden_seq_cpu_ and
        // set last_verify_chunk_start_ = base_pos + n_tokens - 1, so the
        // hidden_at_pos() accessor's `rel = abs_pos - chunk_start` formula
        // resolves to 0 and returns the right pointer.
        //
        // FULL_SEQ (prefill mode): unchanged — download the whole sequence
        // because warm_head_kv() / last_hidden_seq() walks every position.
        const bool last_row_only =
            (capture_mode_ == VerifyCaptureMode::LAST_ROW_ONLY) && n_tokens > 0;
        const int  rows_to_copy   = last_row_only ? 1 : n_tokens;
        const size_t src_row_off  = last_row_only ? (size_t)(n_tokens - 1) : 0;

        last_hidden_seq_cpu_.resize((size_t)rows_to_copy * hidden);
#ifdef DFLASH_VERIFY_PROFILE
        vp_t0 = vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
#endif
        ggml_backend_tensor_get(sg_.all_norm_hidden, last_hidden_seq_cpu_.data(),
                                src_row_off * hidden * sizeof(float),
                                sizeof(float) * (size_t)rows_to_copy * hidden);
#ifdef DFLASH_VERIFY_PROFILE
        if (vprof_on) vp_get_hidden += vprof_ms_since(vp_t0);
#endif
        last_hidden_seq_n_         = rows_to_copy;
        last_verify_chunk_start_   = last_row_only
                                       ? (base_pos + n_tokens - 1)
                                       : base_pos;
        // Mirror the PRE-final-output-norm sequence (set by the graph when
        // capture_all_norm_hidden is on).  hidden_at_pos_pre_norm() reads
        // this; the Qwen3.6 MTP chain seeds h_prev_0 with it (PR #22673
        // `t_h_pre_norm`).  If the graph did not expose the pre-norm
        // tensor (e.g. older graph builds, defensive fallback), clear the
        // buffer so the accessor returns nullptr and the caller falls
        // back to the post-norm tensor.
        if (sg_.all_h_pre_norm) {
            last_hidden_seq_pre_norm_cpu_.resize((size_t)rows_to_copy * hidden);
#ifdef DFLASH_VERIFY_PROFILE
            vp_t0 = vprof_on ? vprof_clock::now() : vprof_clock::time_point{};
#endif
            ggml_backend_tensor_get(sg_.all_h_pre_norm,
                                    last_hidden_seq_pre_norm_cpu_.data(),
                                    src_row_off * hidden * sizeof(float),
                                    sizeof(float) * (size_t)rows_to_copy * hidden);
#ifdef DFLASH_VERIFY_PROFILE
            if (vprof_on) vp_get_hpre = vprof_ms_since(vp_t0);
#endif
        } else {
            last_hidden_seq_pre_norm_cpu_.clear();
        }
    } else {
        last_hidden_seq_n_ = 0;
        last_hidden_seq_pre_norm_cpu_.clear();
    }

    cache_.cur_pos = base_pos + n_tokens;

    // Topology is owned by the caller via capture_topology_for_chain(); if
    // capture fired without prior topology, invalidate to be defensive.
    if (!capture_intermediate) {
        last_tree_base_pos_ = -1;
    }

#ifdef DFLASH_VERIFY_PROFILE
    if (vprof_on) {
        const double vp_total = vprof_ms_since(vp_t_total);
        std::fprintf(stderr,
            "[verify_prof n_tokens=%d base_pos=%d set=%.3f compute=%.3f "
            "get_hidden=%.3f get_hpre=%.3f get_argmax=%.3f total=%.3f (ms)]\n",
            n_tokens, base_pos, vp_set, vp_compute,
            vp_get_hidden, vp_get_hpre, vp_get_argmax, vp_total);
        vprof_sum_set_        += vp_set;
        vprof_sum_compute_    += vp_compute;
        vprof_sum_get_hidden_ += vp_get_hidden;
        vprof_sum_get_hpre_   += vp_get_hpre;
        vprof_sum_get_argmax_ += vp_get_argmax;
        vprof_sum_total_      += vp_total;
        vprof_n_calls_++;
    }
#endif  // DFLASH_VERIFY_PROFILE
    return true;
}

bool Qwen35DFlashTarget::verify_tree(
        const std::vector<int32_t> & flat_tokens,
        const DDTree & tree,
        int base_pos,
        std::vector<int32_t> & out_argmax,
        std::vector<float> * out_logits) {
    const int N = (int)flat_tokens.size();
    if (N <= 0) return false;
    if (N != 1 + tree.n_nodes) return false;

    // Degenerate single-token tree: cheap fast path through verify_batch.
    if (tree.n_nodes == 0) {
        int32_t last_tok = -1;
        std::vector<int32_t> all_argmax;
        if (!verify_batch(flat_tokens, base_pos, last_tok, &all_argmax)) {
            return false;
        }
        out_argmax = std::move(all_argmax);
        if (out_logits) out_logits->clear();
        return true;
    }

    // Real tree verify — body lifted from test_dflash.cpp:3140-3231 (the
    // ddtree-verify branch of run_qwen36_mtp_harness's spec-decode loop)
    // minus the walk/commit policy (the harness keeps that).
    //
    // Stage 3: build_target_step_tree below uses capture_delta_intermediate
    // = true (see graph_builders.cpp), so cache_.ssm_intermediate[il] holds
    // the per-DFS-slot SSM state and cache_.conv_input_cache[il] holds the
    // full conv window.  restore_kv_at_dfs() consumes these on partial
    // accept to undo rejected siblings before the next iteration.
    const int hidden = w_.n_embd;

    if (!build_target_step_tree(sg_, w_, cache_, backend_,
                                /*kv_start=*/base_pos, /*n_tokens=*/N,
                                fa_window_, kq_stride_pad_)) {
        std::fprintf(stderr,
            "[Qwen35DFlashTarget] verify_tree: build_target_step_tree failed: base_pos=%d N=%d\n",
            base_pos, N);
        return false;
    }

    // Embed all N flat tokens (root at slot 0, DFS-ordered tree nodes 1..N-1).
    std::vector<float> tree_embed((size_t)hidden * N, 0.0f);
    if (!w_.embedder.embed(flat_tokens.data(), N, tree_embed.data())) {
        return false;
    }
    ggml_backend_tensor_set(sg_.inp_embed, tree_embed.data(), 0,
                            sizeof(float) * (size_t)hidden * N);

    // M-RoPE axis-major positions: root at base_pos, node i at base_pos + depths[i-1].
    // Mirrors the harness pos4 layout (4 ints per axis, axis-major).
    std::vector<int32_t> pos4(4 * N, 0);
    for (int i = 0; i < N; i++) {
        const int p = base_pos + (i == 0 ? 0 : tree.depths[i - 1]);
        pos4[0 * N + i] = p;
        pos4[1 * N + i] = p;
        pos4[2 * N + i] = p;
        pos4[3 * N + i] = 0;
    }
    ggml_backend_tensor_set(sg_.positions, pos4.data(), 0,
                            sizeof(int32_t) * 4 * N);

    // Ancestor-only attention mask.  build_target_step_tree allocated
    // sg.attn_mask with kv_pad = align_up(cache.max_ctx + N, kq_stride_pad);
    // pin build_tree_mask's kv stride to the same value via kv_pad_override.
    const int win_start = (fa_window_ > 0 && base_pos > fa_window_)
                            ? (base_pos - fa_window_) : 0;
    std::vector<uint16_t> mask_buf;
    build_tree_mask(tree, base_pos, mask_buf, kq_stride_pad_,
                    win_start, /*kv_pad_override=*/cache_.max_ctx + N);
    ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                            sizeof(uint16_t) * mask_buf.size());

    // parent_ids: root is -1; node i (1..n_nodes) points to its tree parent.
    std::vector<int32_t> parent_ids(N, 0);
    parent_ids[0] = -1;
    for (int i = 1; i < N; i++) parent_ids[i] = (int32_t)tree.parents[i];
    ggml_backend_tensor_set(sg_.parent_ids, parent_ids.data(), 0,
                            sizeof(int32_t) * N);

    auto st = ggml_backend_graph_compute(backend_, sg_.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "[Qwen35DFlashTarget] verify_tree: graph_compute failed: base_pos=%d N=%d status=%d\n",
            base_pos, N, (int)st);
        return false;
    }

    // Read argmax for all N tree positions.
    out_argmax.resize(N);
    ggml_backend_tensor_get(sg_.argmax_tokens, out_argmax.data(), 0,
                            sizeof(int32_t) * N);

    // Optional: pull the full [N × vocab] logits.
    if (out_logits) {
        const int vocab = sg_.logits ? (int)sg_.logits->ne[0] : 0;
        if (vocab <= 0) {
            out_logits->clear();
        } else {
            out_logits->resize((size_t)N * vocab);
            ggml_backend_tensor_get(sg_.logits, out_logits->data(), 0,
                                    sizeof(float) * (size_t)N * vocab);
        }
    }

    cache_.cur_pos = base_pos + N;

    // Stage 3: cache the tree topology so restore_kv_at_dfs() can locate the
    // accepted slots, compute the post-rollback cur_pos, and walk ancestry
    // for the conv-window K-1 rollback on sibling-walk accept paths.
    last_tree_base_pos_ = base_pos;
    last_tree_n_nodes_  = tree.n_nodes;
    last_tree_parents_.assign(tree.parents.begin(), tree.parents.end());
    last_tree_depths_.assign(tree.depths.begin(), tree.depths.end());

    return true;
}

bool Qwen35DFlashTarget::restore_kv_at_dfs(const std::vector<int> & accepted_dfs) {
    if (last_tree_base_pos_ < 0) {
        std::fprintf(stderr,
            "[Qwen35DFlashTarget] restore_kv_at_dfs called before any verify_tree\n");
        return false;
    }
    if (accepted_dfs.empty()) {
        std::fprintf(stderr,
            "[Qwen35DFlashTarget] restore_kv_at_dfs: empty accepted_dfs\n");
        return false;
    }
    if (accepted_dfs[0] != 0) {
        std::fprintf(stderr,
            "[Qwen35DFlashTarget] restore_kv_at_dfs: accepted_dfs[0]=%d, expected 0 (root)\n",
            accepted_dfs[0]);
        return false;
    }
    const int commit_n     = (int)accepted_dfs.size();          // includes root
    const int rollback_dfs = accepted_dfs[commit_n - 1];        // deepest accepted DFS slot
    const int N            = 1 + last_tree_n_nodes_;
    if (rollback_dfs < 0 || rollback_dfs >= N) {
        std::fprintf(stderr,
            "[Qwen35DFlashTarget] restore_kv_at_dfs: rollback_dfs=%d out of range [0,%d)\n",
            rollback_dfs, N);
        return false;
    }
    // Detect pure-chain walk (accepted[i] == i for every i in the prefix).
    // Hot path; lets us short-circuit the per-conv-column gather + KV compaction.
    bool walked_sibling = false;
    for (int i = 0; i < commit_n; i++) {
        if (accepted_dfs[i] != i) { walked_sibling = true; break; }
    }

    const int n_delta = (int)cache_.ssm_intermediate.size();
    // Bug #3: dedicated stream so the rollback copies don't serialize with
    // the default stream (e.g. ggml backend compute, host syncs).
    if (!rollback_stream_) {
        if (cudaStreamCreate(&rollback_stream_) != cudaSuccess) {
            rollback_stream_ = nullptr;  // fall back to default
        }
    }
    cudaStream_t stream = rollback_stream_;
    for (int il = 0; il < n_delta; il++) {
        ggml_tensor * ssm_inter = cache_.ssm_intermediate[il];
        ggml_tensor * conv_in   = cache_.conv_input_cache[il];
        if (!ssm_inter || !conv_in) {
            std::fprintf(stderr,
                "[Qwen35DFlashTarget] restore_kv_at_dfs: missing capture layer %d "
                "(ssm_inter=%p conv_in=%p) — was verify_tree run with "
                "capture_delta_intermediate=true?\n",
                il, (void*)ssm_inter, (void*)conv_in);
            return false;
        }
        // SSM state rollback (dequant ssm_intermediate[rollback_dfs] → ssm_state[il]).
        const size_t ssm_elems =
            (size_t)cache_.ssm_state[il]->ne[0] *
            (size_t)cache_.ssm_state[il]->ne[1] *
            (size_t)cache_.ssm_state[il]->ne[2];
        const size_t ssm_src_off =
            (size_t)rollback_dfs * ssm_inter->nb[3];
        const void * ssm_src = (const char *)ssm_inter->data + ssm_src_off;
        ggml_get_to_fp32_cuda(ssm_inter->type)(
            ssm_src, (float *)cache_.ssm_state[il]->data,
            (int64_t)ssm_elems, stream);

        // Conv rollback: copy the K-1 most recent inputs along the accepted
        // path's ANCESTRY (not DFS order).  Mirror test_dflash.cpp:3395-3437.
        const int    K_conv  = 4;
        const int    row_cnt = (int)conv_in->ne[1];
        const size_t elt     = ggml_element_size(conv_in);
        const size_t dpitch  = (K_conv - 1) * elt;
        const size_t spitch  = conv_in->nb[1];
        if (!walked_sibling) {
            // Fast path: K_conv-1 = 3 contiguous slots ending at rollback_dfs.
            const int    conv_off = rollback_dfs + 1;
            const void * conv_src = (const char *)conv_in->data + (size_t)conv_off * elt;
            cudaError_t ce = cudaMemcpy2DAsync(
                cache_.conv_state[il]->data, dpitch,
                conv_src, spitch,
                (K_conv - 1) * elt, row_cnt,
                cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr,
                    "[Qwen35DFlashTarget] restore_kv_at_dfs conv fast il=%d: %s\n",
                    il, cudaGetErrorString(ce));
                return false;
            }
        } else {
            int virt[K_conv - 1];
            virt[K_conv - 2] = rollback_dfs;
            for (int k = K_conv - 3; k >= 0; k--) {
                const int prev = virt[k + 1];
                virt[k] = (prev >= 0) ? (int)last_tree_parents_[prev] : (prev - 1);
            }
            for (int k = 0; k < K_conv - 1; k++) {
                const int sx_slot = (K_conv - 1) + virt[k];
                const void * src_col =
                    (const char *)conv_in->data + (size_t)sx_slot * elt;
                char * dst_col =
                    (char *)cache_.conv_state[il]->data + (size_t)k * elt;
                cudaError_t ce = cudaMemcpy2DAsync(dst_col, dpitch,
                                                   src_col, spitch,
                                                   elt, row_cnt,
                                                   cudaMemcpyDeviceToDevice, stream);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr,
                        "[Qwen35DFlashTarget] restore_kv_at_dfs conv col il=%d k=%d: %s\n",
                        il, k, cudaGetErrorString(ce));
                    return false;
                }
            }
        }
    }

    // Full-attention KV compaction: verify_tree wrote K/V at slots
    // [base..base+N-1] in DFS order.  Bug #3: collapse the per-head inner
    // loop into one cudaMemcpy2DAsync (pitch=nb[2], height=n_kv) — saves
    // 2*n_kv-2 launches per (layer, d) pair on a dedicated stream.
    if (walked_sibling) {
        const int base = last_tree_base_pos_;
        const int n_full_attn = (int)cache_.attn_k.size();
        for (int d = 1; d < commit_n; d++) {
            const int src_dfs  = accepted_dfs[d];
            const int dst_slot = d;
            if (src_dfs == dst_slot) continue;
            for (int l = 0; l < n_full_attn; l++) {
                ggml_tensor * ck = cache_.attn_k[l];
                ggml_tensor * cv = cache_.attn_v[l];
                if (!ck || !cv) continue;
                const size_t slot_bytes = ck->nb[1];
                const int    n_kv       = (int)ck->ne[2];
                const size_t pitch      = ck->nb[2];
                const size_t src_off    = (size_t)(base + src_dfs)  * slot_bytes;
                const size_t dst_off    = (size_t)(base + dst_slot) * slot_bytes;
                cudaMemcpy2DAsync((char *)ck->data + dst_off, pitch,
                                  (const char *)ck->data + src_off, pitch,
                                  slot_bytes, n_kv,
                                  cudaMemcpyDeviceToDevice, stream);
                cudaMemcpy2DAsync((char *)cv->data + dst_off, cv->nb[2],
                                  (const char *)cv->data + src_off, cv->nb[2],
                                  slot_bytes, (int)cv->ne[2],
                                  cudaMemcpyDeviceToDevice, stream);
            }
        }
    }

    // Sync rollback stream so the next graph_compute (on default stream)
    // sees a consistent KV/SSM state.
    if (rollback_stream_) cudaStreamSynchronize(rollback_stream_);

    // Advance cur_pos to "just past the last committed slot" so the next
    // verify_batch's kv_start lines up.  root = dfs 0 lives at base, so
    // commit_n committed tokens occupy slots [base..base+commit_n-1].
    cache_.cur_pos = last_tree_base_pos_ + commit_n;
    return true;
}

void Qwen35DFlashTarget::capture_topology_for_chain(int n_tokens, int base_pos) {
    if (n_tokens <= 0) { last_tree_base_pos_ = -1; return; }
    last_tree_base_pos_ = base_pos;
    last_tree_n_nodes_  = n_tokens - 1;
    last_tree_parents_.resize(n_tokens);
    last_tree_depths_.resize((size_t)(n_tokens - 1));
    last_tree_parents_[0] = -1;
    for (int i = 1; i < n_tokens; i++) {
        last_tree_parents_[i] = i - 1;
        last_tree_depths_[i - 1] = i;
    }
}

bool Qwen35DFlashTarget::restore_kv_at_chain(int accept_n) {
    // A chain of N tokens recorded by verify_batch is the DFS spine
    // [0, 1, ..., N-1].  Roll back to slot accept_n: the first (accept_n + 1)
    // positions remain committed, the tail is discarded.  Returns false if
    // the most recent verify_batch did NOT capture per-position intermediates
    // (chain_capture_enabled_ was off, or n_tokens overflowed the cache) —
    // the chain runner falls back to its legacy snapshot+recommit path.
    if (accept_n < 0) return false;
    if (last_tree_base_pos_ < 0) return false;
    if (accept_n > last_tree_n_nodes_) return false;
    std::vector<int> path((size_t)accept_n + 1);
    for (int i = 0; i <= accept_n; i++) path[i] = i;
    return restore_kv_at_dfs(path);
}

bool Qwen35DFlashTarget::snapshot_kv() {
    snapshot_ssm_state(cache_);
    return true;
}

bool Qwen35DFlashTarget::restore_kv() {
    restore_ssm_state(cache_);
    return true;
}

bool Qwen35DFlashTarget::is_eos(int token) const {
    return is_eos_tok(token, w_);
}

bool Qwen35DFlashTarget::embed_tokens(const int32_t * tokens, int n,
                                       float * out) const {
    return w_.embedder.embed(tokens, n, out);
}

bool Qwen35DFlashTarget::project_hidden_to_tokens(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out) {
    if (n_tokens <= 0) return false;

    if (!build_lm_head_projection_step(proj_sg_, w_, backend_, n_tokens)) {
        return false;
    }

    ggml_backend_tensor_set(proj_sg_.hidden_input, hidden, 0,
                            sizeof(float) * (size_t)n_tokens * w_.n_embd);

    auto st = ggml_backend_graph_compute(backend_, proj_sg_.gf);
    if (st != GGML_STATUS_SUCCESS) return false;

    // Read argmax results from GPU.
    tokens_out.resize(n_tokens);
    ggml_backend_tensor_get(proj_sg_.argmax_tokens, tokens_out.data(), 0,
                            sizeof(int32_t) * n_tokens);
    return true;
}

bool Qwen35DFlashTarget::project_hidden_to_logits(
        const float * hidden,
        int n_tokens,
        std::vector<float> & logits_out,
        int & out_vocab) {
    out_vocab = 0;
    if (n_tokens <= 0) return false;

    if (!build_lm_head_projection_step(proj_sg_, w_, backend_, n_tokens)) {
        return false;
    }

    ggml_backend_tensor_set(proj_sg_.hidden_input, hidden, 0,
                            sizeof(float) * (size_t)n_tokens * w_.n_embd);

    auto st = ggml_backend_graph_compute(backend_, proj_sg_.gf);
    if (st != GGML_STATUS_SUCCESS) return false;

    const int vocab = (int)proj_sg_.logits->ne[0];
    logits_out.resize((size_t)n_tokens * vocab);
    ggml_backend_tensor_get(proj_sg_.logits, logits_out.data(), 0,
                            sizeof(float) * (size_t)n_tokens * vocab);
    out_vocab = vocab;
    return true;
}

int Qwen35DFlashTarget::mask_token_id() const {
    return w_.mask_token_id;
}

const std::vector<int> & Qwen35DFlashTarget::capture_layer_ids() const {
    return capture_ids_;
}

}  // namespace dflash27b
