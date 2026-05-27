// Gemma4DFlashTarget — DFlashTarget adapter for Gemma4 iSWA models.

#include "gemma4_dflash_target.h"
#include "dflash27b.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace dflash::common {

Gemma4DFlashTarget::Gemma4DFlashTarget(
        Gemma4Weights & w,
        Gemma4Cache & cache,
        ggml_backend_t backend)
    : w_(w), cache_(cache), backend_(backend) {
    // Use capture layer IDs from cache (computed when target_feat is allocated)
    if (!cache.capture_layer_ids.empty()) {
        capture_ids_ = cache.capture_layer_ids;
    } else {
        // Fallback: evenly-spaced (legacy path)
        const int N = DFLASH27B_DRAFT_N_TARGET_LAYERS;
        capture_ids_.resize(N);
        const int step = std::max(1, (w.n_layer - 2) / (N - 1));
        for (int k = 0; k < N; k++) {
            capture_ids_[k] = 1 + k * step;
        }
    }
}

Gemma4DFlashTarget::~Gemma4DFlashTarget() {
    free_gemma4_snapshot(verify_snap_);
}

bool Gemma4DFlashTarget::verify_batch(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<int32_t> * all_argmax) {
    const int n_tokens = (int)tokens.size();
    if (n_tokens <= 0) return false;

    const int hidden = w_.n_embd;

    // Embed tokens
    std::vector<float> embed((size_t)n_tokens * hidden);
    if (!w_.embedder.embed(tokens.data(), n_tokens, embed.data())) {
        std::fprintf(stderr, "gemma4_verify_batch: embed failed\n");
        return false;
    }

    // Scale by sqrt(n_embd) (Gemma4 convention)
    const float scale = std::sqrt((float)hidden);
    for (size_t i = 0; i < embed.size(); ++i) embed[i] *= scale;

    // Run verify (all-token argmax)
    std::vector<int32_t> argmax_buf;
    if (!gemma4_verify_batch(backend_, w_, cache_, embed.data(),
                              tokens.data(), n_tokens, base_pos,
                              argmax_buf)) {
        return false;
    }

    last_tok = argmax_buf[n_tokens - 1];
    if (all_argmax) {
        *all_argmax = std::move(argmax_buf);
    }

    return true;
}

bool Gemma4DFlashTarget::snapshot_kv() {
    // Save cur_pos and KV cache state
    verify_snap_.cur_pos = cache_.cur_pos;

    // Allocate snapshot tensors if needed
    if (verify_snap_.k_snap.empty()) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (size_t)(w_.n_layer * 2 + 4) + 4096;
        ip.no_alloc = true;
        verify_snap_.ctx = ggml_init(ip);
        if (!verify_snap_.ctx) return false;

        verify_snap_.k_snap.resize(w_.n_layer, nullptr);
        verify_snap_.v_snap.resize(w_.n_layer, nullptr);
        for (int il = 0; il < w_.n_layer; ++il) {
            if (cache_.k[il]) {
                verify_snap_.k_snap[il] = ggml_dup_tensor(verify_snap_.ctx, cache_.k[il]);
                verify_snap_.v_snap[il] = ggml_dup_tensor(verify_snap_.ctx, cache_.v[il]);
            }
        }
        verify_snap_.buf = ggml_backend_alloc_ctx_tensors(verify_snap_.ctx, backend_);
        if (!verify_snap_.buf) return false;
    }

    // Copy KV cache to snapshot
    for (int il = 0; il < w_.n_layer; ++il) {
        if (cache_.k[il] && verify_snap_.k_snap[il]) {
            ggml_backend_tensor_copy(cache_.k[il], verify_snap_.k_snap[il]);
            ggml_backend_tensor_copy(cache_.v[il], verify_snap_.v_snap[il]);
        }
    }
    return true;
}

bool Gemma4DFlashTarget::restore_kv() {
    if (verify_snap_.k_snap.empty()) return false;

    // Restore KV cache from snapshot
    for (int il = 0; il < w_.n_layer; ++il) {
        if (cache_.k[il] && verify_snap_.k_snap[il]) {
            ggml_backend_tensor_copy(verify_snap_.k_snap[il], cache_.k[il]);
            ggml_backend_tensor_copy(verify_snap_.v_snap[il], cache_.v[il]);
        }
    }
    cache_.cur_pos = verify_snap_.cur_pos;
    return true;
}

bool Gemma4DFlashTarget::is_eos(int token) const {
    return token == w_.eos_id || token == w_.eos_chat_id;
}

bool Gemma4DFlashTarget::embed_tokens(const int32_t * tokens, int n,
                                       float * out) const {
    if (!w_.embedder.embed(tokens, n, out)) return false;
    // Scale by sqrt(n_embd) to match Gemma4's embedding convention.
    // The draft was trained with Gemma4's scaled embeddings as noise input.
    const float scale = std::sqrt((float)w_.n_embd);
    const size_t total = (size_t)n * w_.n_embd;
    for (size_t i = 0; i < total; ++i) out[i] *= scale;
    return true;
}

bool Gemma4DFlashTarget::project_hidden_to_tokens(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out) {
    return gemma4_project_hidden(backend_, w_, hidden, n_tokens, tokens_out);
}

int Gemma4DFlashTarget::mask_token_id() const {
    // Gemma4 DFlash draft uses token ID 4 as mask (per model card)
    return 4;
}

const std::vector<int> & Gemma4DFlashTarget::capture_layer_ids() const {
    return capture_ids_;
}

}  // namespace dflash::common
