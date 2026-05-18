// mtp_chain_runner.h — Generic γ-loop for MTP speculative decoding.
//
// Drives an IMtpModule (ExternalDrafter or NativeHeads) through γ
// speculative steps per iteration, verifies the resulting candidate
// sequence on the target via DFlashTarget::verify_batch, accepts the
// longest matching prefix + 1 bonus token, and rolls back the target's
// KV cache on any reject. Identical verify path for both flavors;
// only the propose step dispatches on flavor().
//
// Peer to dflash_spec_decode.h (PR #197): same pattern, MTP-specific.

#pragma once

#include "model_backend.h"   // GenerateRequest, GenerateResult, DaemonIO
#include "mtp_interface.h"
#include "sampler.h"

#include <vector>

namespace dflash27b {

struct DFlashTarget;  // forward — see common/dflash_target.h

namespace mtp {

// Per-iteration telemetry. Aggregated into MtpChainRunner::stats() so
// callers (test, daemon) can report acceptance / chain depth.
struct MtpChainStats {
    int   total_iters     = 0;   // chain iterations executed
    int   total_proposed  = 0;   // draft tokens proposed (Σ γ across iters)
    int   total_accepted  = 0;   // draft tokens accepted (Σ accept_n)
    int   total_emitted   = 0;   // tokens written to out_tokens (accepted + bonus)
    int   eos_hits        = 0;
};

class MtpChainRunner {
public:
    MtpChainRunner(IMtpModule & mtp,
                   DFlashTarget & target,
                   const SamplerCfg & sampler);

    // Run the MTP γ-loop over the prompt in `req`, writing decoded
    // tokens into the result. Caller is responsible for prefill — this
    // runner assumes target.verify_batch and DFlashTarget snapshot_kv
    // /restore_kv are in a state where the last prefill token has been
    // committed and `committed_pos` (passed as `req.snap_pos` when set,
    // else the prefill length) points just past it.
    //
    // `gamma` is the chain length. `gamma > mtp.max_gamma()` is clamped
    // and a stderr warning is emitted once per run.
    //
    // The runner does not own prefill — that's the backend's job. It
    // does own: propose, verify, accept/rollback, sample-on-tie,
    // emit (stream), EOS detection.
    GenerateResult run(const GenerateRequest & req,
                       const DaemonIO & io,
                       int32_t last_prefill_token,
                       int committed_pos,
                       int gamma);

    const MtpChainStats & stats() const { return stats_; }

private:
    IMtpModule       & mtp_;
    DFlashTarget     & target_;
    SamplerCfg         sampler_cfg_;
    MtpChainStats      stats_;

    // Propose γ draft tokens from the current position. Dispatches on
    // mtp_.flavor(); ExternalDrafter threads prev_hidden through γ
    // serial step() calls, NativeHeads issues one step_batch().
    // Returns false on module failure (callers abort the run).
    //
    // `prev_hidden` is the host-side h_prev captured from the previous
    // commit's post-norm hidden (ExternalDrafter only; ignored for
    // NativeHeads). `prev_hidden_dim` must equal mtp_.hidden_size().
    bool propose_drafts_(int32_t current_token,
                         int base_pos,
                         int gamma,
                         const float * prev_hidden,
                         int prev_hidden_dim,
                         std::vector<int32_t> & drafts_out,
                         std::vector<float>   & next_hidden_out);
};

}  // namespace mtp
}  // namespace dflash27b
