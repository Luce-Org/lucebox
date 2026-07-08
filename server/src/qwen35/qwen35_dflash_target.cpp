// Qwen35DFlashTarget — DFlashTarget adapter for qwen35 hybrid models.

#include "qwen35_dflash_target.h"
#include "common/gpu_runtime_compat.h"
#include "graph_builders.h"
#include "step_graph.h"
#include "attn_masks.h"
#include "common/geometric_draft_topk_cuda.h"
// gpu_runtime_compat.h maps the raw cudaStream_t / cudaMemcpy* symbols used
// below (rollback_to / rollback_to_tree) onto their HIP equivalents. Without
// it the file only compiles on CUDA via a transitive <cuda_runtime.h>; HIP
// builds (e.g. gfx1151) fail with "cudaStream_t undeclared".
#include "common/gpu_runtime_compat.h"

#include <cstdlib>
#include <cstring>

// ggml_get_to_fp32_cuda is not in any public header — it lives in
// ggml-cuda/convert.cuh. Declare here so we can link against it.
using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
extern "C++" to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

namespace dflash::common {

Qwen35DFlashTarget::~Qwen35DFlashTarget() {
    step_graph_destroy(proj_sg_);
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
        std::vector<int32_t> * all_argmax,
        bool capture_ssm_intermediates) {
    const int n_tokens = (int)tokens.size();
    if (n_tokens <= 0) return false;

    const int hidden = w_.n_embd;
    const bool pool = pager_ != nullptr;
    const bool need_mask = pool || (kq_stride_pad_ > KQ_MASK_PAD) || (n_tokens > 1);

    // kvflash: allocate slots for the verify block up front (may evict at
    // a chunk boundary; protections keep sinks + the tail window safe).
    std::vector<int> slots;
    if (pool) {
        slots.resize(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            slots[i] = pager_->slot_for(base_pos + i);
            if (slots[i] < 0) {
                std::fprintf(stderr, "verify_batch: pool slot alloc failed @%d\n", base_pos + i);
                return false;
            }
        }
    }

    // kvflash's set_rows KV-write is mutually exclusive with delta-intermediate
    // capture (graph_builders gates use_kv_write_rows on !capture_delta_intermediate);
    // skip capture under the pager so --ddtree + --kvflash doesn't fail verify.
    const bool do_capture = fast_rollback_ && capture_ssm_intermediates && pager_ == nullptr;

    if (!build_target_step(sg_, w_, cache_, backend_,
                           /*kv_start=*/base_pos, n_tokens,
                           need_mask, /*capture=*/true,
                           /*capture_delta_intermediate=*/do_capture,
                           pool ? 0 : fa_window_,
                           /*last_token_logits_only=*/false,
                           kq_stride_pad_,
                           /*capture_moe_router=*/false,
                           /*kvflash_mask=*/pool)) {
        std::fprintf(stderr, "verify_batch: build_target_step failed (base=%d n=%d)\n", base_pos, n_tokens);
        return false;
    }
    if (pool && !sg_.kv_write_rows) {
        std::fprintf(stderr, "verify_batch: kvflash requires set_rows path\n");
        return false;
    }
    if (pool) {
        // kv_write_rows is [n_tokens, n_head_kv] ne0-major: element
        // (token i, head h) lives at i + h*n_tokens (set_rows asserts
        // b->ne[1] == c->ne[0]). Getting this transposed scrambles
        // per-head row targets for every multi-token write.
        std::vector<int64_t> rows((size_t)n_tokens * w_.n_head_kv);
        for (int h = 0; h < w_.n_head_kv; h++) {
            for (int i = 0; i < n_tokens; i++) {
                rows[(size_t)h * n_tokens + i] = slots[i];
            }
        }
        ggml_backend_tensor_set(sg_.kv_write_rows, rows.data(), 0,
                                sizeof(int64_t) * rows.size());
    }

    // Embed input tokens and fill positions.
    std::vector<float> embed((size_t)n_tokens * hidden);
    if (!w_.embedder.embed(tokens.data(), n_tokens, embed.data())) {
        std::fprintf(stderr, "verify_batch: embed failed (n=%d)\n", n_tokens);
        return false;
    }
    ggml_backend_tensor_set(sg_.inp_embed, embed.data(), 0,
                            sizeof(float) * embed.size());

    // Qwen35 uses interleaved positions: 4 ints per token.
    std::vector<int32_t> pos(4 * n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        pos[4 * i + 0] = base_pos + i;
        pos[4 * i + 1] = base_pos + i;
        pos[4 * i + 2] = base_pos + i;
        pos[4 * i + 3] = 0;
    }
    ggml_backend_tensor_set(sg_.positions, pos.data(), 0,
                            sizeof(int32_t) * pos.size());

    // Fill the attention mask.
    if (sg_.attn_mask && pool) {
        // Slot-space mask: row q attends (a) slots of committed positions
        // (pos < base_pos) of resident chunks — this exactly excludes
        // slots holding rejected drafts from earlier rounds — and (b) the
        // verify tokens' own slots, causally.
        const size_t kvd = (size_t)sg_.attn_mask->ne[0];
        const int q_pad = (int)sg_.attn_mask->ne[1];
        std::vector<uint16_t> mask_buf((size_t)kvd * q_pad, F16_NEG_INF);
        const int ct = pager_->chunk_tokens();
        for (int c = 0; c < pager_->n_chunks(); c++) {
            const int blk = pager_->block_of(c);
            if (blk < 0) continue;
            for (int i = 0; i < ct; i++) {
                if ((int64_t)c * ct + i >= base_pos) break;
                mask_buf[(size_t)blk * ct + i] = F16_ZERO;
            }
        }
        for (int q = 1; q < n_tokens; q++) {
            std::memcpy(mask_buf.data() + (size_t)q * kvd, mask_buf.data(), kvd * 2);
        }
        for (int q = 0; q < n_tokens; q++) {
            for (int i = 0; i <= q; i++) {
                mask_buf[(size_t)q * kvd + slots[i]] = F16_ZERO;
            }
        }
        ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
    } else if (sg_.attn_mask) {
        const int win_start = (fa_window_ > 0 && base_pos > fa_window_)
                                  ? (base_pos - fa_window_) : 0;
        const int kv_len = base_pos + n_tokens - win_start;
        std::vector<uint16_t> mask_buf;
        const int kv_pad_override = (int)sg_.attn_mask->ne[0];
        build_causal_mask(mask_buf, kv_len, n_tokens, base_pos,
                          kq_stride_pad_, win_start, kv_pad_override);
        ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
    }

    auto st = ggml_backend_graph_compute(backend_, sg_.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "verify_batch: compute failed (status=%d)\n", (int)st);
        return false;
    }

    // Read argmax results from GPU.
    std::vector<int32_t> argmax_buf(n_tokens);
    ggml_backend_tensor_get(sg_.argmax_tokens, argmax_buf.data(), 0,
                            sizeof(int32_t) * n_tokens);
    last_tok = argmax_buf[n_tokens - 1];

    if (all_argmax) {
        *all_argmax = std::move(argmax_buf);
    }

    cache_.cur_pos = base_pos + n_tokens;
    return true;
}

bool Qwen35DFlashTarget::read_verify_logits(int n_tokens, std::vector<float> & out) {
    if (!sg_.logits || n_tokens <= 0) return false;
    const int64_t vocab = sg_.logits->ne[0];
    if (n_tokens > (int)sg_.logits->ne[1]) return false;
    out.resize((size_t)n_tokens * (size_t)vocab);
    ggml_backend_tensor_get(sg_.logits, out.data(), 0,
                            sizeof(float) * out.size());
    return true;
}

bool Qwen35DFlashTarget::supports_tree_verify() const {
    // Tree verify reuses the fast-rollback SSM-intermediate capture and builds a
    // non-paged, contiguous tree graph. Pure capability here; the kvflash
    // identity/pool precondition is enforced at the call site in do_spec_decode
    // and re-checked defensively in verify_tree() below.
    return fast_rollback_;
}

bool Qwen35DFlashTarget::verify_tree(
        int committed,
        const DDTree & tree,
        const std::vector<int32_t> & flat_tokens,
        int n_alloc,
        std::vector<int32_t> & posterior_out,
        std::vector<float> * logits_out) {
    if (!fast_rollback_) return false;
    const int N        = n_alloc;                 // fixed alloc width (budget+1)
    const int N_actual = 1 + tree.n_nodes;        // real tree size incl. root
    if (N_actual <= 0 || N_actual > N || (int)flat_tokens.size() < N) return false;
    // kvflash: the tree graph reads the prefix [0, committed) contiguously and
    // writes rows [committed, committed+N). Only valid while that prefix is
    // identity-resident and the write span fits the pool. do_spec_decode gates
    // on this; reaching here otherwise is a bug — fail cleanly, never read past
    // the pool.
    if (pager_ &&
        (committed + N > pager_->pool_tokens() ||
         !pager_->identity_prefix_covers(committed))) {
        std::fprintf(stderr,
            "verify_tree: kvflash layout not identity-contiguous "
            "(committed=%d N=%d pool=%d)\n",
            committed, N, pager_->pool_tokens());
        return false;
    }
    const int hidden = w_.n_embd;

    // Tree-verify graph: ancestor-masked batched forward over DFS-ordered nodes.
    // Capture per-node SSM intermediates so rollback_to_tree() can restore.
    if (!build_target_step_tree(sg_, w_, cache_, backend_,
                                /*kv_start=*/committed, /*n_tokens=*/N,
                                fa_window_, kq_stride_pad_)) {
        std::fprintf(stderr, "verify_tree: build_target_step_tree failed\n");
        return false;
    }

    // Embeddings: [root, tree nodes, padding(token 0)]. Pad slots are masked.
    std::vector<float> embed((size_t)hidden * N, 0.0f);
    if (!w_.embedder.embed(flat_tokens.data(), N_actual, embed.data())) {
        std::fprintf(stderr, "verify_tree: embed failed\n");
        return false;
    }
    ggml_backend_tensor_set(sg_.inp_embed, embed.data(), 0,
                            sizeof(float) * (size_t)hidden * N);

    // M-RoPE axis-major positions: each node sits at committed + its depth.
    std::vector<int32_t> pos4(4 * N, 0);
    for (int i = 0; i < N_actual; i++) {
        const int p = committed + (i == 0 ? 0 : tree.depths[i - 1]);
        pos4[0 * N + i] = p;
        pos4[1 * N + i] = p;
        pos4[2 * N + i] = p;
        pos4[3 * N + i] = 0;
    }
    ggml_backend_tensor_set(sg_.positions, pos4.data(), 0, sizeof(int32_t) * 4 * N);

    // Ancestor-only attention mask (f16). Rows 0..N_actual-1 use tree
    // visibility; padding rows stay -inf (see nothing).
    if (sg_.attn_mask) {
        const int tree_win_start = (fa_window_ > 0 && committed > fa_window_)
                                       ? (committed - fa_window_) : 0;
        const int kv_pad_m = (int)sg_.attn_mask->ne[0];
        const int q_pad_m  = (int)sg_.attn_mask->ne[1];
        std::vector<uint16_t> mask_buf((size_t)kv_pad_m * q_pad_m, F16_NEG_INF);
        for (int q = 0; q < N_actual; q++) {
            for (int k = std::max(0, tree_win_start); k < committed; k++) {
                mask_buf[(size_t)q * kv_pad_m + (k - tree_win_start)] = F16_ZERO;
            }
            for (int j = 0; j < N_actual; j++) {
                if (tree.visibility[(size_t)q * N_actual + j]) {
                    mask_buf[(size_t)q * kv_pad_m + (committed + j - tree_win_start)] = F16_ZERO;
                }
            }
        }
        ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
    }

    // parent_ids: real nodes point to their tree parent; root = -1; pad = 0.
    if (!sg_.parent_ids) {
        std::fprintf(stderr, "verify_tree: step graph has no parent_ids tensor\n");
        return false;
    }
    std::vector<int32_t> parent_ids(N, 0);
    parent_ids[0] = -1;
    for (int i = 1; i < N_actual; i++) parent_ids[i] = (int32_t)tree.parents[i];
    ggml_backend_tensor_set(sg_.parent_ids, parent_ids.data(), 0, sizeof(int32_t) * N);

    auto st = ggml_backend_graph_compute(backend_, sg_.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "verify_tree: compute failed (status=%d)\n", (int)st);
        return false;
    }

    // Posterior = per-node target argmax.
    //
    // The verify graph already computes a batched per-node GPU argmax
    // (sg_.argmax_tokens, built by build_target_step_tree). When the caller does
    // not need the full logits (greedy decode, logits_out == nullptr) we read
    // those N_actual int32s directly and skip the vocab×N_actual D2H + CPU
    // argmax entirely — eliminates the verify-logits transfer hotspot.
    //
    // Historically the GPU argmax shortcut has returned -1 for tree-shaped
    // verify graphs on some builds; guard against that by validating every row
    // and falling back to the CPU path for the step if any index is bad.
    // Escape hatch: DFLASH_GPU_VERIFY_ARGMAX=0 forces the legacy CPU path.
    static const bool kGpuVerifyArgmax = []() {
        const char * v = std::getenv("DFLASH_GPU_VERIFY_ARGMAX");
        return v == nullptr || v[0] != '0';
    }();
    const int vocab = (int)sg_.logits->ne[0];
    posterior_out.resize(N_actual);

    if (kGpuVerifyArgmax && !logits_out && sg_.argmax_tokens) {
        ggml_backend_tensor_get(sg_.argmax_tokens, posterior_out.data(), 0,
                                sizeof(int32_t) * N_actual);
        bool ok = true;
        for (int i = 0; i < N_actual; i++) {
            if (posterior_out[i] < 0 || posterior_out[i] >= vocab) { ok = false; break; }
        }
        if (ok) return true;  // fast path; otherwise fall through to CPU argmax
    }

    std::vector<float> logits((size_t)vocab * N_actual);
    ggml_backend_tensor_get(sg_.logits, logits.data(), 0,
                            sizeof(float) * (size_t)vocab * N_actual);
    for (int i = 0; i < N_actual; i++) {
        const float * row = logits.data() + (size_t)i * vocab;
        int am = 0; float best = row[0];
        for (int v = 1; v < vocab; v++) if (row[v] > best) { best = row[v]; am = v; }
        posterior_out[i] = am;
    }
    if (logits_out) *logits_out = std::move(logits);
    return true;
}

bool Qwen35DFlashTarget::rollback_to_tree(
        int committed,
        const DDTree & tree,
        const std::vector<int> & accepted_dfs) {
    if (!fast_rollback_) return false;
    const int commit_n = (int)accepted_dfs.size();
    if (commit_n <= 0) return false;

    const int rollback_dfs = accepted_dfs[commit_n - 1];  // deepest committed node
    if (rollback_dfs < 0) return false;

    // Pure-chain walk has accepted_dfs[i] == i; a sibling branch breaks that and
    // forces the conv-ancestry / KV-compaction gather.
    bool walked_sibling = false;
    for (int i = 0; i < commit_n; i++) {
        if (accepted_dfs[i] != i) { walked_sibling = true; break; }
    }

    const int n_delta = (int)sg_.delta_captures.size();
    cudaStream_t stream = nullptr;
    for (int il = 0; il < n_delta; il++) {
        const DeltaNetCapture & cap = sg_.delta_captures[il];
        if (!cap.ssm_intermediate_states || !cap.conv_input) {
            std::fprintf(stderr, "rollback_to_tree: missing capture at layer %d\n", il);
            return false;
        }
        if (rollback_dfs >= (int)cap.ssm_intermediate_states->ne[3]) {
            std::fprintf(stderr, "rollback_to_tree: rollback_dfs %d >= captured slots %d (layer %d)\n",
                         rollback_dfs, (int)cap.ssm_intermediate_states->ne[3], il);
            return false;
        }
        // SSM state ← intermediate[rollback_dfs] (dequantize Q8_0/F16 → f32).
        const size_t ssm_elems =
            (size_t)cache_.ssm_state[il]->ne[0] *
            (size_t)cache_.ssm_state[il]->ne[1] *
            (size_t)cache_.ssm_state[il]->ne[2];
        const size_t ssm_src_offset =
            (size_t)rollback_dfs * cap.ssm_intermediate_states->nb[3];
        const void * ssm_src =
            (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
        const auto to_fp32 = ggml_get_to_fp32_cuda(cap.ssm_intermediate_states->type);
        if (!to_fp32) {
            std::fprintf(stderr, "rollback_to_tree: no fp32 converter for type %d (layer %d)\n",
                         (int)cap.ssm_intermediate_states->type, il);
            return false;
        }
        to_fp32(ssm_src, (float *)cache_.ssm_state[il]->data, (int64_t)ssm_elems, stream);

        // Conv state ← the K-1 most recent inputs along rollback_dfs's ancestry.
        const int K_conv = 4;
        const int row_cnt = (int)cap.conv_input->ne[1];
        const size_t elt = ggml_element_size(cap.conv_input);
        const size_t dpitch = (K_conv - 1) * elt;
        const size_t spitch = cap.conv_input->nb[1];
        if (!walked_sibling) {
            // Fast path: K-1 contiguous slots ending at rollback_dfs.
            const int conv_off = rollback_dfs + 1;
            const void * conv_src =
                (const char *)cap.conv_input->data + (size_t)conv_off * elt;
            cudaError_t ce = cudaMemcpy2DAsync(cache_.conv_state[il]->data, dpitch,
                                               conv_src, spitch,
                                               (K_conv - 1) * elt, row_cnt,
                                               cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr, "rollback_to_tree: conv fast il=%d: %s\n",
                             il, cudaGetErrorString(ce));
                return false;
            }
        } else {
            // Sibling path: gather K-1 columns along the parent chain.
            int virt[K_conv - 1];
            virt[K_conv - 2] = rollback_dfs;
            for (int k = K_conv - 3; k >= 0; k--) {
                const int prev = virt[k + 1];
                virt[k] = (prev >= 0) ? (int)tree.parents[prev] : (prev - 1);
            }
            for (int k = 0; k < K_conv - 1; k++) {
                const int sx_slot = (K_conv - 1) + virt[k];
                const void * src_col =
                    (const char *)cap.conv_input->data + (size_t)sx_slot * elt;
                char * dst_col =
                    (char *)cache_.conv_state[il]->data + (size_t)k * elt;
                cudaError_t ce = cudaMemcpy2DAsync(dst_col, dpitch, src_col, spitch,
                                                   elt, row_cnt,
                                                   cudaMemcpyDeviceToDevice, stream);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr, "rollback_to_tree: conv col il=%d k=%d: %s\n",
                                 il, k, cudaGetErrorString(ce));
                    return false;
                }
            }
        }
    }

    // target_feat compaction: verify wrote features in DFS order at slots
    // committed+i. Copy each accepted DFS slot's features to its spine slot d.
    if (cache_.target_feat) {
        const size_t elt = ggml_element_size(cache_.target_feat);
        const int    fc_in = (int)cache_.target_feat->ne[0];
        const size_t col_stride = cache_.target_feat->nb[1];
        const int    tcap = cache_.target_feat_cap;
        for (int d = 1; d < commit_n; d++) {
            const int src_dfs = accepted_dfs[d];
            if (src_dfs == d) continue;
            const size_t src_off = (size_t)((committed + src_dfs) % tcap) * col_stride;
            const size_t dst_off = (size_t)((committed + d)       % tcap) * col_stride;
            cudaMemcpyAsync((char *)cache_.target_feat->data + dst_off,
                            (const char *)cache_.target_feat->data + src_off,
                            (size_t)fc_in * elt, cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Full-attention KV compaction: move accepted DFS slots onto the spine
    // [committed..committed+commit_n-1] so the next round sees a contiguous prefix.
    const int n_full_attn = (int)cache_.attn_k.size();
    for (int d = 0; d < commit_n; d++) {
        const int src_dfs = accepted_dfs[d];
        if (src_dfs == d) continue;
        for (int l = 0; l < n_full_attn; l++) {
            ggml_tensor * ck = cache_.attn_k[l];
            ggml_tensor * cv = cache_.attn_v[l];
            const size_t slot_bytes = ck->nb[1];
            const size_t src_off = (size_t)(committed + src_dfs) * slot_bytes;
            const size_t dst_off = (size_t)(committed + d)       * slot_bytes;
            const int n_kv = (int)ck->ne[2];
            for (int h = 0; h < n_kv; h++) {
                const size_t head_src = src_off + (size_t)h * ck->nb[2];
                const size_t head_dst = dst_off + (size_t)h * ck->nb[2];
                cudaMemcpyAsync((char *)ck->data + head_dst,
                                (const char *)ck->data + head_src,
                                slot_bytes, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync((char *)cv->data + head_dst,
                                (const char *)cv->data + head_src,
                                slot_bytes, cudaMemcpyDeviceToDevice, stream);
            }
        }
    }

    cudaStreamSynchronize(stream);
    // kvflash: the tree graph writes KV directly (not slot-mapped), so this is
    // the single owning point that advances the pager for tree-committed
    // positions. Covers both the greedy and sampled tree fast paths; chain
    // paths self-register through verify_batch()'s slot_for(). The call-site
    // guard bounds the span to the resident identity pool so this never evicts,
    // but a failed alloc must abort: returning true with unmapped slots would
    // make the next step read stale/unmapped KV (silent corruption).
    if (pager_ && !pager_->alloc_span(committed, commit_n)) {
        std::fprintf(stderr,
            "rollback_to_tree: kvflash alloc_span failed (committed=%d commit_n=%d)\n",
            committed, commit_n);
        return false;
    }
    cache_.cur_pos = committed + commit_n;
    return true;
}

bool Qwen35DFlashTarget::snapshot_kv() {
    snapshot_ssm_state(cache_);
    return true;
}

bool Qwen35DFlashTarget::restore_kv() {
    restore_ssm_state(cache_);
    return true;
}

bool Qwen35DFlashTarget::supports_fast_rollback() const {
    // Pure capability. Fast-rollback only restores recurrent SSM/conv state and
    // defers the bonus token, so it is pager-safe even while paging: committed
    // KV rows are written slot-mapped by verify_batch(), and the deferred bonus
    // is re-fed at the next committed position on the following step.
    return fast_rollback_;
}

bool Qwen35DFlashTarget::rollback_to(int base_pos, int commit_n) {
    static const bool kFastRollbackDiag = []() {
        const char * e = std::getenv("FAST_ROLLBACK_DIAG");
        return e != nullptr && std::strcmp(e, "0") != 0;
    }();

    if (!fast_rollback_) {
        if (kFastRollbackDiag) {
            std::fprintf(stderr, "rollback_to: fast_rollback disabled\n");
        }
        return false;
    }

    // commit_n must be a positive count. `commit_n - 1` below indexes the
    // per-step intermediates; a non-positive value underflows to a huge
    // size_t byte offset and triggers an out-of-bounds GPU read. A zero/neg
    // commit means "nothing to keep" — signal failure so the caller falls
    // back to the full restore_kv path.
    if (commit_n <= 0) {
        if (kFastRollbackDiag) {
            std::fprintf(stderr, "rollback_to: commit_n <= 0 commit_n=%d\n",
                         commit_n);
        }
        return false;
    }

    const int n_delta = (int)sg_.delta_captures.size();
    if (n_delta == 0) {
        if (kFastRollbackDiag) {
            std::fprintf(stderr, "rollback_to: no delta_captures\n");
        }
        return false;
    }

    // If all tokens accepted, the SSM state after processing all q_len tokens
    // is exactly what we want — no rollback needed, just fix cur_pos.
    const int q_len = cache_.cur_pos - base_pos;
    if (commit_n >= q_len) {
        cache_.cur_pos = base_pos + commit_n;
        return true;
    }

    const int rollback_idx = commit_n - 1;  // index into per-step intermediates
    cudaStream_t stream = nullptr;

    for (int il = 0; il < n_delta; il++) {
        const DeltaNetCapture & cap = sg_.delta_captures[il];
        if (!cap.ssm_intermediate_states) {
            if (kFastRollbackDiag) {
                std::fprintf(stderr, "rollback_to: null ssm_intermediate_states layer=%d\n",
                             il);
            }
            return false;
        }
        if (!cap.conv_input) {
            if (kFastRollbackDiag) {
                std::fprintf(stderr, "rollback_to: null conv_input layer=%d\n",
                             il);
            }
            return false;
        }
        if (rollback_idx >= (int)cap.ssm_intermediate_states->ne[3]) {
            if (kFastRollbackDiag) {
                std::fprintf(stderr,
                             "rollback_to: rollback_idx OOB rollback_idx=%d slots=%d layer=%d\n",
                             rollback_idx, (int)cap.ssm_intermediate_states->ne[3], il);
            }
            return false;
        }

        // SSM rollback: copy intermediate[rollback_idx] → cache.ssm_state[il]
        const size_t ssm_elems =
            (size_t)cache_.ssm_state[il]->ne[0] *
            (size_t)cache_.ssm_state[il]->ne[1] *
            (size_t)cache_.ssm_state[il]->ne[2];
        const size_t ssm_src_offset =
            (size_t)rollback_idx * cap.ssm_intermediate_states->nb[3];
        const void * ssm_src =
            (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
        const auto to_fp32 = ggml_get_to_fp32_cuda(cap.ssm_intermediate_states->type);
        if (!to_fp32) {
            if (kFastRollbackDiag) {
                std::fprintf(stderr, "rollback_to: no fp32 converter type=%d layer=%d\n",
                             (int)cap.ssm_intermediate_states->type, il);
            }
            return false;
        }
        to_fp32(ssm_src, (float *)cache_.ssm_state[il]->data,
                (int64_t)ssm_elems, stream);

        // Conv rollback: copy conv_input[commit_n..commit_n+K-2, :, :]
        // into cache.conv_state[il].
        const int K_conv = 4;
        if (commit_n + K_conv - 1 > (int)cap.conv_input->ne[0]) {
            if (kFastRollbackDiag) {
                std::fprintf(stderr,
                             "rollback_to: conv_input OOB commit_n=%d needed=%d slots=%d layer=%d\n",
                             commit_n, commit_n + K_conv - 1,
                             (int)cap.conv_input->ne[0], il);
            }
            return false;
        }
        const int row_cnt = (int)cap.conv_input->ne[1];
        const size_t elt = ggml_element_size(cap.conv_input);
        const size_t dpitch = (K_conv - 1) * elt;
        const size_t spitch = cap.conv_input->nb[1];
        const size_t width  = (K_conv - 1) * elt;
        const void * conv_src =
            (const char *)cap.conv_input->data + commit_n * elt;
        cudaError_t ce = cudaMemcpy2DAsync(cache_.conv_state[il]->data, dpitch,
                                           conv_src, spitch,
                                           width, row_cnt,
                                           cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) {
            if (kFastRollbackDiag) {
                std::fprintf(stderr, "rollback_to: cudaMemcpy2D conv layer=%d: %s\n",
                             il, cudaGetErrorString(ce));
            }
            return false;
        }
    }
    cudaStreamSynchronize(stream);

    cache_.cur_pos = base_pos + commit_n;
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

bool Qwen35DFlashTarget::project_hidden_to_topk(
        const float * hidden,
        int n_tokens,
        int K,
        float temperature,
        std::vector<float> & top_log_probs,
        std::vector<int32_t> & top_token_ids) {
    if (n_tokens <= 0 || K <= 0) return false;

    // Same projection graph as project_hidden_to_tokens — proj_sg_.logits is a
    // graph output (argmax depends on it), so it is computed and readable here.
    if (!build_lm_head_projection_step(proj_sg_, w_, backend_, n_tokens)) {
        return false;
    }
    ggml_backend_tensor_set(proj_sg_.hidden_input, hidden, 0,
                            sizeof(float) * (size_t)n_tokens * w_.n_embd);
    auto st = ggml_backend_graph_compute(backend_, proj_sg_.gf);
    if (st != GGML_STATUS_SUCCESS) return false;

    const int vocab = (int)proj_sg_.logits->ne[0];
    top_log_probs.assign((size_t)n_tokens * K, 0.0f);
    top_token_ids.assign((size_t)n_tokens * K, 0);

#ifdef DFLASH27B_HAVE_DRAFT_TOPK
    // GPU path: top-K + logsumexp directly on the logits device buffer, skipping
    // the vocab×n_tokens D2H and the CPU heap extract. Falls back to the CPU path
    // on any failure. Escape hatch: DFLASH_GPU_DRAFT_TOPK=0.
    static const bool kGpuDraftTopk = []() {
        const char * v = std::getenv("DFLASH_GPU_DRAFT_TOPK");
        return v == nullptr || v[0] != '0';
    }();
    if (kGpuDraftTopk &&
        geometric_extract_draft_topk_cuda(proj_sg_.logits->data, n_tokens, vocab, K,
                                top_log_probs.data(), top_token_ids.data(),
                                temperature)) {
        return true;
    }
#endif

    std::vector<float> logits((size_t)vocab * n_tokens);
    ggml_backend_tensor_get(proj_sg_.logits, logits.data(), 0,
                            sizeof(float) * (size_t)vocab * n_tokens);
    extract_draft_topk(logits.data(), n_tokens, vocab, K,
                       top_log_probs.data(), top_token_ids.data(), temperature);
    return true;
}

int Qwen35DFlashTarget::mask_token_id() const {
    return w_.mask_token_id;
}

const std::vector<int> & Qwen35DFlashTarget::capture_layer_ids() const {
    return capture_ids_;
}

}  // namespace dflash::common
