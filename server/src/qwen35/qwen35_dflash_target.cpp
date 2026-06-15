// Qwen35DFlashTarget — DFlashTarget adapter for qwen35 hybrid models.

#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "step_graph.h"
#include "attn_masks.h"

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

    const bool do_capture = fast_rollback_ && capture_ssm_intermediates;

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

bool Qwen35DFlashTarget::snapshot_kv() {
    snapshot_ssm_state(cache_);
    return true;
}

bool Qwen35DFlashTarget::restore_kv() {
    restore_ssm_state(cache_);
    return true;
}

bool Qwen35DFlashTarget::supports_fast_rollback() const {
    return fast_rollback_;
}

bool Qwen35DFlashTarget::rollback_to(int base_pos, int commit_n) {
    if (!fast_rollback_) return false;

    const int n_delta = (int)sg_.delta_captures.size();
    if (n_delta == 0) return false;

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
        if (!cap.ssm_intermediate_states || !cap.conv_input) {
            std::fprintf(stderr, "rollback_to: missing capture at layer %d\n", il);
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
        ggml_get_to_fp32_cuda(cap.ssm_intermediate_states->type)(
            ssm_src, (float *)cache_.ssm_state[il]->data,
            (int64_t)ssm_elems, stream);

        // Conv rollback: copy conv_input[commit_n..commit_n+K-2, :, :]
        // into cache.conv_state[il].
        const int K_conv = 4;
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
            std::fprintf(stderr, "rollback_to: cudaMemcpy2D conv il=%d: %s\n",
                         il, cudaGetErrorString(ce));
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

int Qwen35DFlashTarget::mask_token_id() const {
    return w_.mask_token_id;
}

const std::vector<int> & Qwen35DFlashTarget::capture_layer_ids() const {
    return capture_ids_;
}

}  // namespace dflash::common
