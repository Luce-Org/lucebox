// Inkling2Backend — ModelBackend for Inkling 2 / Inkling 2 Mini (`inkling2_loop` / `inkling2`).
//
// Features:
//   - Looped middle core (24 physical layers -> 48 effective visits)
//   - Zero-kq_b inline relative-bias + zero-sink Wave32 FlashAttention / Split-KV
//   - Native bundled DSpark speculator (`mtp.safetensors`, auto-discovered beside GGUF)
//   - Dual-GPU R9700 + Strix Halo 74/26 parallel MoE experts + 79/21 LM-head split +
//     automatic thermal/contention guard + Split-KV Global Attention offload
//   - Shared cost-aware AdaptiveSpecWidth controller across all block widths (2..block_size)
//     with per-context SpecWidthCostMemory + exact GPU top-candidate sampler

#pragma once

#include "common/adaptive_spec_width.h"
#include "common/model_backend.h"
#include "pflash/pflash_drafter.h"
#include "placement/placement_config.h"

#include "inkling2_internal.h"
#include "inkling2_cache.h"
#include "inkling2_graph.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace luce::common {

struct Inkling2BackendConfig {
    std::string                 model_path;
    std::optional<std::string>  draft_path;      // mtp.safetensors; empty = auto-discover beside GGUF; "0"/"none" = off
    int                         verify_width = 0; // 0 = auto (8 with DSpark), 1 = off (AR), 2..8 = fixed
    DevicePlacement             device;
    std::optional<DevicePlacement> expert_device;
    int                         chunk = 512;
    ggml_type                   cache_type_k = GGML_TYPE_COUNT; // COUNT -> Q8_0 default (or F16 via --cache-type-k f16)
    ggml_type                   cache_type_v = GGML_TYPE_COUNT;
    // PFlash: when the drafter scores beside the resident model on GPU pflash_gpu (-1: none),
    // auto-sizing and Split-KV keep pflash_bytes + pflash_bytes_per_token * max_ctx free there.
    int                         pflash_gpu = -1;
    size_t                      pflash_bytes = 0;
    size_t                      pflash_bytes_per_token = 0;
};

struct Inkling2Snapshot {
    bool                  used    = false;
    int                   cur_pos = 0;
    int                   dspark_ctx_pos = 0;
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    std::vector<ggml_tensor *> k_cache;
    std::vector<ggml_tensor *> v_cache;
    std::vector<ggml_tensor *> conv_state;
    std::vector<ggml_tensor *> dspark_k;
    std::vector<ggml_tensor *> dspark_v;
    ggml_tensor *         last_taps = nullptr;
    std::vector<float>    logits;
};

class Inkling2Backend final : public ModelBackend {
public:
    explicit Inkling2Backend(Inkling2BackendConfig cfg);
    ~Inkling2Backend() override;

    Inkling2Backend(const Inkling2Backend &) = delete;
    Inkling2Backend & operator=(const Inkling2Backend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;
    int prefill_chunk_size() const override { return chunk_; }
    int context_limit() const override { return cache_.max_ctx; }

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    Residency residency() const override;
    bool release_gpus(const std::vector<int> & gpus, std::string & error) override;
    bool reinstate(std::string & error) override;

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    size_t snapshot_bytes_estimate(int tokens) const override;

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    // PFlash: the drafter scores the prompt beside the resident model, or in the
    // memory a park frees (pflash_window.h).
    CompressResult compress(const CompressRequest & req) override;
    std::vector<CompressResult> compress_batch(
        const std::vector<CompressRequest> & requests);
    bool handle_compress(const std::string & line, const DaemonIO & io) override;
    void free_drafter() override;
    bool pflash_gives_candidate_scores() const override { return true; }

    void shutdown() override;

private:
    GenerateResult run(const GenerateRequest & req, const DaemonIO & io, int restored);
    size_t pflash_reserve(ggml_backend_t device, int ctx) const;
    bool load_all();
    void release_device_state();
    void release_all();
    std::string resolve_dspark_path() const;

    Inkling2BackendConfig cfg_;
    ggml_backend_t        backend_        = nullptr;
    ggml_backend_t        expert_backend_ = nullptr;
    ggml_backend_t        snap_backend_   = nullptr;
    Inkling2Weights       weights_;
    Inkling2Cache         cache_;
    int                   chunk_          = 512;
    int                   fitted_max_ctx_ = 0;
    int                   fitted_remote_split_pos_ = -1;
    bool                  parked_         = false;
    bool                  released_       = false;
    DrafterContext        drafter_ctx_;          // PFlash drafter, loaded at the first compress
    bool                  drafter_loaded_ = false;
    std::vector<float>    last_logits_;
    std::array<Inkling2Snapshot, kMaxSlots> snapshots_;
    AdaptiveSpecWidth     spec_width_{kInkling2MaxVerify};
    bool                  spec_width_init_ = false;
    SpecWidthCostMemory   spec_costs_single_;
    SpecWidthCostMemory   spec_costs_dual_;
};

} // namespace luce::common
