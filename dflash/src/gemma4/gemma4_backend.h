// Gemma4Backend: ModelBackend implementation for Gemma4 31B Dense / 26B-A4B MoE.
//
// Wraps Gemma4 target weights/cache (#171, #176), optional DFlash draft
// (#180, #181), and optional MTP assistant (#182) behind the generic
// ModelBackend interface (#175) so the daemon loop in daemon_loop.cpp can
// drive Gemma4 models without Gemma4-specific code.
//
// Decode strategy is selected via Gemma4BackendArgs::draft_method:
//   - kNone:    plain autoregressive greedy/sample decode
//   - kDFlash:  block-diffusion DFlash draft + tree-verify (DDTree)
//   - kMtp:     single-step MTP assistant + γ=1 verify (γ>1 follow-up)
//
// All three paths share the same prefill (chunked, optional pflash).

#pragma once

#include "../common/model_backend.h"
#include "../internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <random>
#include <string>
#include <vector>

namespace dflash27b {

enum class Gemma4DraftMethod {
    kNone = 0,   // no speculative decode
    kDFlash,     // DFlash block-diffusion draft (needs draft_path)
    kMtp,        // MTP assistant (needs mtp_path)
};

struct Gemma4BackendArgs {
    std::string target_path;             // required: target GGUF
    std::string draft_path;               // required if draft_method == kDFlash
    std::string mtp_path;                 // required if draft_method == kMtp
    Gemma4DraftMethod draft_method = Gemma4DraftMethod::kNone;

    int  max_ctx       = 16384;
    int  chunk         = 2048;
    bool use_pflash    = false;
    float pflash_alpha = 0.12f;

    // DFlash draft tuning
    int  draft_kv_cap_override = 0;       // 0 = default cap
    int  draft_max_block       = 16;
    bool draft_enable_capture_overrides = false;

    // MTP tuning
    int  mtp_gamma = 1;                   // γ chain length (>1 is follow-up)
};

class Gemma4Backend : public ModelBackend {
public:
    explicit Gemma4Backend(const Gemma4BackendArgs & args);
    ~Gemma4Backend() override;

    // Initialise CUDA backend, load target / draft / MTP weights, allocate caches.
    // Returns false on failure (prints to stderr).
    bool init();

    // ── ModelBackend interface ────────────────────────────────────────
    void print_ready_banner() const override;

    bool park(const std::string & what) override;
    bool unpark(const std::string & what) override;
    bool is_target_parked() const override { return target_parked_; }

    GenerateResult generate(const GenerateRequest & req,
                            const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate(int slot,
                                        const GenerateRequest & req,
                                        const DaemonIO & io) override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    void shutdown() override;

private:
    Gemma4BackendArgs args_;
    ggml_backend_t    backend_ = nullptr;

    GemmaTargetWeights target_w_{};
    GemmaTargetCache   cache_{};

    // Optional DFlash draft (loaded only if draft_method == kDFlash)
    GemmaDraftWeights  draft_w_{};
    bool               draft_loaded_ = false;

    // Optional MTP assistant (loaded only if draft_method == kMtp)
    MtpDrafterWeights  mtp_w_{};
    bool               mtp_loaded_ = false;

    bool target_parked_ = false;
    std::mt19937_64 sampler_rng_{std::random_device{}()};

    // Per-slot snapshot state. Slot impl is left to the follow-up — the slot
    // table here just tracks usage/cur_pos so LIST_SLOTS works.
    struct Slot {
        bool used    = false;
        int  cur_pos = -1;
    };
    std::array<Slot, kMaxSlots> slots_{};

    // ── Internal helpers ─────────────────────────────────────────────
    bool prefill(const std::vector<int32_t> & prompt,
                 std::vector<float> & out_last_logits,
                 double & out_prefill_s);

    // AR (no-draft) decode — produces n_gen tokens via target forward + greedy/sample.
    bool decode_autoregressive(int n_gen,
                               std::vector<float> & last_logits_io,
                               const GenerateRequest & req,
                               const DaemonIO & io,
                               std::vector<int32_t> & out_tokens,
                               double & out_decode_s);

    // DFlash draft + tree-verify decode — TODO in follow-up. Port from
    // feature/gemma4-support test_gemma4_dflash.cpp lines ~1400-2400 (the
    // build_ddtree + verify loop) and the build_draft_kv_prefill_graph /
    // build_gemma4_draft_graph call sites.
    bool decode_dflash(int n_gen,
                       std::vector<float> & last_logits_io,
                       const GenerateRequest & req,
                       const DaemonIO & io,
                       std::vector<int32_t> & out_tokens,
                       double & out_decode_s);

    // MTP single-step decode — TODO in follow-up. Port from
    // feature/gemma4-support test_gemma4_dflash.cpp lines ~2400-2800 (the
    // build_mtp_step_graph + γ=1 verify path).
    bool decode_mtp(int n_gen,
                    std::vector<float> & last_logits_io,
                    const GenerateRequest & req,
                    const DaemonIO & io,
                    std::vector<int32_t> & out_tokens,
                    double & out_decode_s);
};

}  // namespace dflash27b
