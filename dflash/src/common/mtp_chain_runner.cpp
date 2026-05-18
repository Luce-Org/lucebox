// mtp_chain_runner.cpp — see mtp_chain_runner.h for contract.

#include "mtp_chain_runner.h"

#include "dflash_target.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace dflash27b::mtp {

MtpChainRunner::MtpChainRunner(IMtpModule & mtp,
                               DFlashTarget & target,
                               const SamplerCfg & sampler)
    : mtp_(mtp), target_(target), sampler_cfg_(sampler) {
}

bool MtpChainRunner::propose_drafts_(int32_t current_token,
                                     int base_pos,
                                     int gamma,
                                     const float * prev_hidden,
                                     int prev_hidden_dim,
                                     std::vector<int32_t> & drafts_out,
                                     std::vector<float>   & next_hidden_out) {
    drafts_out.clear();
    drafts_out.reserve(gamma);

    if (mtp_.flavor() == MtpFlavor::NativeHeads) {
        auto & native = static_cast<INativeMtp &>(mtp_);
        std::vector<StepOutput> outs;
        // Phase A: ask the native module for `gamma` autoregressive drafts.
        // For depth==1 (today's production for any pre-Phase-A native module)
        // this is byte-identical to the old step_batch path via the default
        // impl in INativeMtp::step_chain.  For depth>1 the module chains its
        // own forward and returns up to `gamma` drafts.
        if (!native.step_chain(current_token, base_pos, gamma, outs)) return false;
        const int got = (int)outs.size();
        const int take = std::min(gamma, got);
        for (int i = 0; i < take; i++) drafts_out.push_back(outs[i].draft_token);
        // NativeHeads does not produce h_prev — leave next_hidden_out empty.
        next_hidden_out.clear();
        return true;
    }

    // ExternalDrafter: γ serial step() calls, threading h_prev.
    auto & ext = static_cast<IExternalDrafterMtp &>(mtp_);
    const int H = mtp_.hidden_size();

    std::vector<float> running_hidden;
    if (prev_hidden && prev_hidden_dim == H) {
        running_hidden.assign(prev_hidden, prev_hidden + H);
    } else {
        // Caller did not supply h_prev — module must be in a state where
        // step() can handle a null prev_hidden (e.g. first iter of a chain).
    }

    int32_t cur = current_token;
    for (int g = 0; g < gamma; g++) {
        StepInput  in;
        in.current_token   = cur;
        in.base_pos        = base_pos;   // committed base; module uses gamma_index for offset
        in.gamma_index     = g;
        in.prev_hidden     = running_hidden.empty() ? nullptr : running_hidden.data();
        in.prev_hidden_dim = (int)running_hidden.size();

        StepOutput out;
        if (!ext.step(in, out)) return false;

        drafts_out.push_back(out.draft_token);
        cur = out.draft_token;

        // Thread next_hidden into the following iter. If the module did
        // not produce a hidden (e.g. final step on a small chain), keep
        // running_hidden as-is so the next iter's caller can still pass
        // the most recent value.
        if (!out.next_hidden.empty()) {
            running_hidden = std::move(out.next_hidden);
        }
    }

    next_hidden_out = std::move(running_hidden);
    return true;
}

GenerateResult MtpChainRunner::run(const GenerateRequest & req,
                                   const DaemonIO & io,
                                   int32_t last_prefill_token,
                                   int committed_pos,
                                   int gamma) {
    GenerateResult result;
    const int n_gen = req.n_gen;
    if (n_gen <= 0) { result.ok = true; return result; }

    // Clamp γ to the module's stated ceiling.
    const int gamma_max = std::max(1, mtp_.max_gamma());
    if (gamma > gamma_max) {
        std::fprintf(stderr,
            "[mtp_chain_runner] γ=%d > module max_gamma=%d; clamping.\n",
            gamma, gamma_max);
        gamma = gamma_max;
    }
    if (gamma < 1) gamma = 1;

    auto t0 = std::chrono::steady_clock::now();
    result.tokens.reserve(n_gen);

    int32_t cur_tok = last_prefill_token;
    int     base_pos = committed_pos;

    // Enable per-position DeltaNet intermediate capture for the duration of
    // this run, so verify_batch records the per-slot SSM state and conv input
    // window we need for restore_kv_at_chain() on partial-accept iters.
    // Safe because chain verify candidates are bounded by g_actual+1 <=
    // max_gamma+1 <= max_verify_tokens.  RAII: turned off on every exit path.
    struct ChainCaptureGuard {
        DFlashTarget & t;
        ~ChainCaptureGuard() { t.enable_chain_capture(false); }
    };
    target_.enable_chain_capture(true);
    ChainCaptureGuard guard{target_};

    // ExternalDrafter optionally provides h_prev across iters. NativeHeads
    // does not — running_hidden stays empty for that flavor.
    std::vector<float> running_hidden;

    bool hit_eos = false;

    while ((int)result.tokens.size() < n_gen && !hit_eos) {
        const int remaining = n_gen - (int)result.tokens.size();
        const int g_iter    = std::min(gamma, remaining);

        // ── Propose γ drafts ───────────────────────────────────────────
        std::vector<int32_t> drafts;
        std::vector<float>   next_hidden;
        if (!propose_drafts_(cur_tok, base_pos, g_iter,
                             running_hidden.empty() ? nullptr : running_hidden.data(),
                             (int)running_hidden.size(),
                             drafts, next_hidden)) {
            result.ok = false;
            result.error = "mtp.propose";
            return result;
        }
        if ((int)drafts.size() < g_iter) {
            // Module returned fewer drafts than requested (e.g. NativeHeads
            // num_heads < γ). Shrink the verify candidate accordingly.
            // No-op if drafts.size() == g_iter.
        }
        const int g_actual = (int)drafts.size();
        stats_.total_proposed += g_actual;

        // ── Verify on target ───────────────────────────────────────────
        // Candidate sequence: [cur_tok, drafts[0..g_actual-1]]
        // After verify, all_argmax[i] = target's argmax AFTER seeing
        // candidate[i] at base_pos+i.
        std::vector<int32_t> candidate;
        candidate.reserve(1 + g_actual);
        candidate.push_back(cur_tok);
        for (auto d : drafts) candidate.push_back(d);

        if (!target_.snapshot_kv()) {
            result.ok = false;
            result.error = "snapshot_kv";
            return result;
        }

        // Caller-owned topology for restore_kv_at_chain (bug #2).
        target_.capture_topology_for_chain((int)candidate.size(), base_pos);

        int last_argmax = -1;
        std::vector<int32_t> all_argmax;
        if (!target_.verify_batch(candidate, base_pos, last_argmax, &all_argmax)) {
            target_.restore_kv();
            result.ok = false;
            result.error = "verify_batch";
            return result;
        }
        if ((int)all_argmax.size() < (int)candidate.size()) {
            target_.restore_kv();
            result.ok = false;
            result.error = "verify_batch_short";
            return result;
        }

        // ── Accept longest matching prefix ─────────────────────────────
        int accept_n = 0;
        for (int i = 0; i < g_actual; i++) {
            if (drafts[i] == all_argmax[i]) accept_n++;
            else break;
        }

        // Total tokens this iter = accept_n + 1 (the bonus from target's
        // argmax at the divergence point or after the last accepted draft).
        const int total_this_iter = accept_n + 1;

        // ── KV reconciliation ──────────────────────────────────────────
        // Three paths converge on the SAME post-iter invariant:
        //   base_pos advances by total_this_iter = accept_n + 1
        //   cur_tok                 = bonus (= all_argmax[accept_n])
        // and the bonus's KV is written by the NEXT iter's verify_batch.
        //
        //   1. accept-all (accept_n == g_actual): verify_batch wrote g+1 slots;
        //      we treat the last (bonus) slot as uncommitted and let next iter
        //      overwrite it.
        //   2. fast rollback (restore_kv_at_chain succeeds): rolls cache.cur_pos
        //      to base_pos + accept_n + 1, leaving the bonus slot unwritten.
        //   3. recommit (fast path declined): snapshot+restore, then recommit
        //      only [cur, accepted...] (accept_n+1 slots, NO bonus).  Bonus is
        //      threaded via cur_tok like the other paths.  Advancing by
        //      accept_n+2 here would skip a position every recommit iter and
        //      diverge from AR — see test_recommit_byte_identical_to_ar.
        if (accept_n < g_actual) {
            if (!target_.restore_kv_at_chain(accept_n)) {
                // Slow path: snapshot rollback + commit ONLY [cur, accepted...]
                // (accept_n+1 slots).  Bonus stays uncommitted, threaded via
                // cur_tok like the fast path.
                if (!target_.restore_kv()) {
                    result.ok = false;
                    result.error = "restore_kv";
                    return result;
                }
                std::vector<int32_t> commit_seq;
                commit_seq.reserve((size_t)accept_n + 1);
                commit_seq.push_back(cur_tok);
                for (int i = 0; i < accept_n; i++) commit_seq.push_back(drafts[i]);
                int discard = -1;
                if (!target_.verify_batch(commit_seq, base_pos, discard, nullptr)) {
                    result.ok = false;
                    result.error = "recommit";
                    return result;
                }
            }
        }

        // ── Emit accepted prefix + bonus, capped at n_gen ──────────────
        // emit_cap is the absolute ceiling on tokens that may be written
        // this iter; the runner advances KV by total_this_iter regardless
        // (we already verified/recommitted that many positions) so KV state
        // stays consistent even when emission is truncated.
        const int emit_cap = std::min(total_this_iter,
                                      n_gen - (int)result.tokens.size());
        int emitted = 0;
        for (int i = 0; i < accept_n && emitted < emit_cap; i++) {
            result.tokens.push_back(drafts[i]);
            if (req.stream) io.emit(drafts[i]);
            emitted++;
            if (target_.is_eos(drafts[i])) { hit_eos = true; break; }
        }
        if (!hit_eos && emitted < emit_cap) {
            const int32_t bonus = all_argmax[accept_n];
            result.tokens.push_back(bonus);
            if (req.stream) io.emit(bonus);
            emitted++;
            if (target_.is_eos(bonus)) hit_eos = true;
            cur_tok = bonus;
        } else {
            cur_tok = result.tokens.empty() ? cur_tok : result.tokens.back();
        }

        // All paths share: base_pos += total_this_iter and cur_tok = bonus.
        base_pos += total_this_iter;

        stats_.total_iters    += 1;
        stats_.total_accepted += accept_n;
        stats_.total_emitted  += emitted;

        running_hidden = std::move(next_hidden);
    }

    if (hit_eos) stats_.eos_hits++;

    if (req.stream) io.emit(-1);

    auto t1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t1 - t0).count();
    result.ok = true;
    return result;
}

}  // namespace dflash27b::mtp
