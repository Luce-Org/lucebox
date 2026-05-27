// Gemma4DFlashTarget — DFlashTarget adapter for Gemma4 iSWA models.
//
// Wraps the Gemma4 target infrastructure (Gemma4Weights, Gemma4Cache,
// gemma4_step) behind the generic DFlashTarget interface so the universal
// DFlash draft model can drive speculative decode verification.

#pragma once

#include "common/dflash_target.h"
#include "gemma4_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace dflash::common {

class Gemma4DFlashTarget : public DFlashTarget {
public:
    // Non-owning references — caller must ensure lifetime.
    Gemma4DFlashTarget(Gemma4Weights & w,
                       Gemma4Cache & cache,
                       ggml_backend_t backend);

    ~Gemma4DFlashTarget() override;

    // ── DFlashTarget interface ──────────────────────────────────────

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr) override;

    bool snapshot_kv() override;
    bool restore_kv() override;

    bool is_eos(int token) const override;

    bool embed_tokens(const int32_t * tokens, int n,
                      float * out) const override;

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;

    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override;
    const std::vector<int> & capture_layer_ids() const override;

private:
    Gemma4Weights & w_;
    Gemma4Cache & cache_;
    ggml_backend_t backend_;

    // Capture layer IDs (built once in constructor).
    std::vector<int> capture_ids_;

    // Snapshot for speculative verify rollback.
    Gemma4Snapshot verify_snap_;
};

}  // namespace dflash::common
