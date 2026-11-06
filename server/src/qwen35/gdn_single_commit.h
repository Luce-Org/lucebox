// GDN single commit: recurrent-state handling of the Qwen3.5/3.6/3.8 DFlash
// chain verify on a single GPU.
//
// The three parts below are always on. Each keeps the parent's handling
// wherever its exclusions apply (keep_base_block_reason,
// swap_snapshot_block_reason, single_replay_block_reason), and the outputs
// are bit-identical either way.
//
// KEEP_BASE. The chain capture verify drops its final SSM
// state write (ggml_cpy(new_state, ssm_state) in build_delta_net_block), so
// ssm_state keeps its pre-verify value and rollback_to always commits
// intermediate[commit_n - 1], the all-accepted case included (the parent
// short-circuits that case because its verify already left the final state
// in place). The GDN kernels store intermediate[t] and the final state from
// the same registers: the store at the end of token t's loop body, and the
// final-state store after the loop, with nothing in between for t = q_len-1
// (gated_delta_net.cu, gated_delta_net_cuda and
// gated_delta_net_cuda_grouped_cols). With F32 checkpoints
// intermediate[q_len - 1] is therefore bitwise the parent's final state.
// F16 checkpoints (LUCE_SINGLE_CHAIN_CHECKPOINT_F32=0) are rounded, so the
// part stays inert there.
//
// Every consumer of the post-verify state on this path calls rollback_to
// (commit) or restore_kv (discard), except the AR-burst seed step: a
// one-token capture verify whose state the loop uses as verify left it
// (qwen35_backend.cpp, "Seed-only verify"). One-token verifies therefore keep
// the parent write.
//
// SWAP_SNAPSHOT (needs KEEP_BASE). The SSM snapshot becomes a
// storage exchange instead of a copy. ssm_state[i] (live) and
// ssm_state_snap[i] (spare) are a pair whose ->data, ->buffer and ->extra
// trade places; the ggml_tensor objects stay, so everything that copies by
// tensor identity (prefix cache, disk cache, kvflash ladder) is unaffected.
// TargetCache::ssm_state_swapped records the parity; reset_recurrent_state,
// reset_target_cache and free_target_cache put the pairs back first, so the
// wholesale base_buf clear zeroes the live state. The conv snapshot stays a
// copy. With P the pre-step SSM state:
//
//   snapshot_kv   conv copy, no SSM copy, no host sync; P stays in the live
//                 tensors ("deferred": the spare is stale).
//   verify_batch  keep-base: live untouched, still P. Any other verify (the
//                 AR seed, replays) and verify_tree first take the deferred
//                 copy (spare <- live, then synchronize, as the parent's
//                 snapshot), then write live as the parent does.
//   rollback_to   keep-base + deferred: spare <- intermediate[commit_n-1],
//                 exchange the pairs. Live is the committed state, the spare
//                 holds P, exactly the parent's contents.
//   restore_kv    deferred: live is still P; restore conv only and keep the
//                 SSM snapshot deferred, so the replay that follows takes it
//                 before writing (one SSM copy on the slow path, as the
//                 parent's restore; a second restore copies spare -> live).
//                 Otherwise the parent copy. A failed verify leaves live P.
//
// SINGLE_REPLAY (needs KEEP_BASE; with or without
// SWAP_SNAPSHOT). The keep-base verify runs its GDN op as a mapped verify
// (ggml_gated_delta_net_mapped_verify over a one-slot state map holding 0):
// the grouped-cols kernel with WRITE_INTER=false, no final-state write and no
// per-token states, recording the replay log instead ([g | k | delta] per
// token and head, ggml_gated_delta_net_capture_replay_log, read in place from
// the GDN result, which the graph keeps alive). rollback_to then commits the
// accepted prefix on the backend stream with ggml_backend_cuda_gdn_chain_commit
// (ggml-cuda-gdn-chain.h), out of place: live -> spare and exchange when the
// snapshot is deferred, else live -> live after the snapshot copy. The conv
// state keeps KEEP_BASE's handling (the verify writes the history and the
// window capture, rollback copies conv_input[commit_n ..]).
//
// Bitwise: the grouped-cols kernel updates each element as
// fmaf(k_reg[r], delta[c], g_val * state_shard[c][r]) and records g_val,
// k_reg[r] and lane 0's delta[c] (the value it broadcasts) into the log; the
// commit applies fmaf(k, delta, g * s) to those values in token order from the
// same base element, so it reproduces intermediate[commit_n - 1] bit for bit.
// WRITE_INTER, the state pointer and the log only add stores, so the verify's
// attention output is the parent's. The scalar kernel computes
// g * s + k * delta, which may contract the other way, so the part is inert
// unless the op takes the grouped-cols kernel
// (ggml_backend_cuda_gdn_chain_uses_grouped_cols: S_v 128 on a device that
// takes it). Verifies wider than 16 tokens keep the F32
// checkpoints, which stay allocated as the fallback.
//
// Why snapshot_kv may skip the parent's synchronize: the conv copies are
// queued on the backend stream; the verify graph, the restore copies and the
// replays run on that stream, and rollback_to's null-stream copies run after
// verify_batch's graph compute, which synchronizes the backend. Nothing on
// the host reads conv_state_snap.
//
// Inert paths. Qwen35DFlashTarget evaluates the exclusions per call:
// verify_batch for KEEP_BASE and SINGLE_REPLAY, snapshot_kv for
// SWAP_SNAPSHOT. Paths that never build a Qwen35DFlashTarget (paged and
// concurrent serving, the layer-split target) keep the parent's handling.
//
// The bookkeeping below is pure so it can be tested without a GPU;
// Qwen35DFlashTarget does the device work.

#pragma once

#include "ggml.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace luce::common::gdn_commit {

// Most tokens a single-replay verify may hold: the commit's limit
// (GGML_CUDA_GDN_CHAIN_COMMIT_MAX_TOKENS).
inline constexpr int kSingleReplayMaxTokens = 16;

// Which parts are on: all three. The tests also run them one at a time,
// and with all off for the parent's handling (set_parts_for_test).
struct Parts {
    bool keep_base = true;
    bool swap_snapshot = true;
    bool single_replay = true;
};

namespace detail {
inline Parts & parts_storage() {
    static Parts parts;
    return parts;
}
}  // namespace detail

inline const Parts & parts() { return detail::parts_storage(); }

// Tests only: choose the parts before the first verify or snapshot.
inline void set_parts_for_test(const Parts & p) { detail::parts_storage() = p; }

// What Qwen35DFlashTarget knows about its path when a part is on.
struct PathFacts {
    bool fast_rollback = false;     // delta-intermediate capture configured
    bool kvflash = false;           // pager attached: verify skips capture
    bool specla = false;            // SpecLA factor commit
    bool tensor_parallel = false;   // meta (multi-rank) state tensors
    int  n_seq_slots = 1;           // multi-slot caches carry a slot axis
    bool bailingmoe3 = false;       // KDA writes its state in place
    bool moe = false;
    bool ddtree = false;            // DDTree server (verify_tree path)
    bool have_state = false;        // state + intermediate for every delta layer
    bool intermediates_f32 = false; // every SSM checkpoint is F32
};

// The KEEP_BASE exclusions fixed by the configuration and the weights:
// nullptr when none applies.
inline const char * keep_base_config_block_reason(const PathFacts & f) {
    if (!f.fast_rollback) return "fast rollback is off";
    if (f.kvflash) return "kvflash verify captures no intermediates";
    if (f.specla) return "SpecLA commits from its factor banks";
    if (f.tensor_parallel) return "tensor-parallel state";
    if (f.n_seq_slots > 1) return "multi-slot cache";
    if (f.bailingmoe3) return "bailingmoe3 KDA state";
    if (f.moe) return "MoE target";
    if (f.ddtree) return "DDTree mode";
    if (!f.intermediates_f32) {
        return "SSM checkpoints are not F32 (LUCE_SINGLE_CHAIN_CHECKPOINT_F32=0): "
               "intermediate[i] is not bitwise the state";
    }
    return nullptr;
}

// nullptr when KEEP_BASE may engage on this path, otherwise why it is inert:
// the configuration, then the rollback cache.
inline const char * keep_base_block_reason(const PathFacts & f) {
    if (const char * why = keep_base_config_block_reason(f)) return why;
    if (!f.have_state) return "missing recurrent state or SSM checkpoints";
    return nullptr;
}

// SWAP_SNAPSHOT given KEEP_BASE's verdict (`keep_base_why`, nullptr when it
// may engage): it relies on KEEP_BASE leaving the live state unwritten, so it
// inherits every KEEP_BASE exclusion.
inline const char * swap_snapshot_inherited_reason(bool keep_base_on,
                                                   const char * keep_base_why) {
    if (!keep_base_on) return "requires KEEP_BASE";
    return keep_base_why;
}

// SWAP_SNAPSHOT: nullptr when it may engage.
inline const char * swap_snapshot_block_reason(bool keep_base_on,
                                               const PathFacts & f,
                                               bool pairs_exchangeable) {
    if (const char * why = swap_snapshot_inherited_reason(
            keep_base_on, keep_base_block_reason(f))) {
        return why;
    }
    if (!pairs_exchangeable) return "SSM state and snapshot tensors cannot exchange storage";
    return nullptr;
}

// SINGLE_REPLAY given KEEP_BASE's verdict: it replaces the checkpoints a
// keep-base verify writes, so it inherits every KEEP_BASE exclusion.
inline const char * single_replay_inherited_reason(bool keep_base_on,
                                                   const char * keep_base_why) {
    if (!keep_base_on) return "requires KEEP_BASE";
    return keep_base_why;
}

// What verify_batch knows about one keep-base verify when SINGLE_REPLAY is on.
struct ReplayFacts {
    // Its GDN op takes the grouped-cols kernel, whose update the commit
    // reproduces (ggml_backend_cuda_gdn_chain_uses_grouped_cols).
    bool grouped_cols = false;
    int  n_tokens = 0;
    bool slot_map = false;          // the one-slot state map tensor exists
};

// nullptr when this keep-base verify may record a replay log instead of the
// F32 checkpoints.
inline const char * single_replay_block_reason(const ReplayFacts & r) {
    if (!r.grouped_cols) {
        return "the chain GDN op does not take the grouped-cols kernel "
               "(S_v != 128, or the device): its update "
               "is not the commit's fmaf(k, delta, g * s)";
    }
    if (r.n_tokens > kSingleReplayMaxTokens) {
        return "verify wider than 16 tokens keeps the F32 checkpoints";
    }
    if (!r.slot_map) return "could not allocate the one-slot state map";
    return nullptr;
}

// One verify: a capture verify of more than one token on an eligible path.
inline bool keep_base_verify(bool path_ok, bool do_capture, int n_tokens) {
    return path_ok && do_capture && n_tokens > 1;
}

// Host-side record of where the committed and the pre-step SSM state live.
// Every field stays at its default while every part is off.
struct StateBook {
    // The last capture verify did not write the SSM state: the state after
    // its token t is intermediate[t]. Cleared by rollback_to (commit) and
    // restore_kv (discard).
    bool commit_pending = false;
    int  pending_q_len = 0;
    // SINGLE_REPLAY: that verify wrote no checkpoints either; the state after
    // its token t is the replay log's first t + 1 transitions applied to the
    // live state. Only meaningful while commit_pending.
    bool replay_pending = false;
    // snapshot_kv took no SSM copy: the pre-step state is the content of the
    // live tensors and ssm_state_snap is stale.
    bool snapshot_deferred = false;
};

// A keep-base verify of q_len tokens finished its compute; `replay`: it
// recorded the replay log instead of the checkpoints (SINGLE_REPLAY).
inline void on_keep_base_verify(StateBook & b, int q_len, bool replay) {
    b.commit_pending = true;
    b.pending_q_len = q_len;
    b.replay_pending = replay;
}

// The pending commit is dropped (another verify, a snapshot, the end of the
// decode reached it first).
inline void drop_pending(StateBook & b) {
    b.commit_pending = false;
    b.replay_pending = false;
}

// snapshot_kv. `swap`: SWAP_SNAPSHOT engaged (no SSM copy); otherwise the
// caller takes the parent's copy.
inline void on_snapshot(StateBook & b, bool swap) {
    b.snapshot_deferred = swap;
}

// Something is about to write the live SSM state (a verify that is not
// keep-base, verify_tree, rollback_to_tree): the deferred snapshot must be
// copied first. Call materialized() once the copy is queued and synced.
inline bool must_materialize(const StateBook & b, bool keeps_live_state) {
    return b.snapshot_deferred && !keeps_live_state;
}
inline void materialized(StateBook & b) {
    b.snapshot_deferred = false;
}

struct RollbackPlan {
    bool valid = true;       // false: refuse, the caller restores and replays
    bool shortcut = false;   // parent all-accepted case: only cur_pos moves
    int  ssm_index = -1;     // intermediate committed to the SSM state
    bool copy_conv = true;   // conv window from conv_input[commit_n ..]
    bool into_spare = false; // commit into ssm_state_snap, then exchange
    // SINGLE_REPLAY: commit the replay log's first ssm_index + 1 transitions
    // (from the live state) instead of copying intermediate[ssm_index].
    bool from_replay = false;
};

// rollback_to(base_pos, commit_n) with q_len = cur_pos - base_pos. Without a
// pending commit or a deferred snapshot this is the parent's decision.
inline RollbackPlan plan_rollback(const StateBook & b, int commit_n, int q_len) {
    RollbackPlan p;
    if (!b.commit_pending) {
        // A deferred snapshot is only consumed by a keep-base verify; a
        // rollback without one would overwrite the only copy of P.
        p.valid = !b.snapshot_deferred;
        p.shortcut = commit_n >= q_len;
        p.ssm_index = commit_n - 1;
        return p;
    }
    if (q_len <= 0 || q_len != b.pending_q_len) {
        p.valid = false;
        return p;
    }
    // commit_n > q_len never happens (commit_n <= accept_n <= q_len); the
    // parent would keep the final state there, which is intermediate[q_len-1].
    p.ssm_index = std::min(commit_n, q_len) - 1;
    // All accepted: the verify already left conv_state on the last window.
    p.copy_conv = commit_n < q_len;
    // P is still live: commit into the spare and exchange, which leaves P in
    // ssm_state_snap as the parent's snapshot.
    p.into_spare = b.snapshot_deferred;
    // The live state is still the verify's base either way (KEEP_BASE), so a
    // replay commit reads it: into the spare, or in place after the parent's
    // snapshot copy.
    p.from_replay = b.replay_pending;
    return p;
}

// After the device work of a plan from plan_rollback succeeded (and, for
// into_spare, the pairs were exchanged).
inline void finish_rollback(StateBook & b, const RollbackPlan & p) {
    drop_pending(b);
    if (p.into_spare) b.snapshot_deferred = false;
}

enum class RestoreSsm {
    kCopyFromSnap,  // parent: live <- ssm_state_snap
    kKeepLive,      // deferred snapshot: live already is P
};

// restore_kv. Discards a pending commit. The conv history is always copied
// back from conv_state_snap.
inline RestoreSsm plan_restore(StateBook & b) {
    drop_pending(b);
    return b.snapshot_deferred ? RestoreSsm::kKeepLive : RestoreSsm::kCopyFromSnap;
}

// End of a spec decode: no restore can follow, the deferred P is no longer
// needed.
inline void on_finish(StateBook & b) {
    b.snapshot_deferred = false;
}

// Trade the storage of two same-layout tensors. Tensor identity, and every
// pointer to either ggml_tensor, is unchanged.
inline void exchange_storage(ggml_tensor * a, ggml_tensor * b) {
    std::swap(a->data, b->data);
    std::swap(a->buffer, b->buffer);
    std::swap(a->extra, b->extra);
}

// Exchange every (live, spare) pair and flip the parity.
inline void exchange_pairs(std::vector<ggml_tensor *> & live,
                           std::vector<ggml_tensor *> & spare,
                           bool & swapped) {
    const size_t n = std::min(live.size(), spare.size());
    for (size_t i = 0; i < n; ++i) {
        if (live[i] && spare[i]) exchange_storage(live[i], spare[i]);
    }
    swapped = !swapped;
}

// Back to the allocation layout (live in base_buf). The contents follow the
// buffers, so the live tensors then hold whatever the spares held: only for
// callers that are about to overwrite or free both.
inline void unswap_pairs(std::vector<ggml_tensor *> & live,
                         std::vector<ggml_tensor *> & spare,
                         bool & swapped) {
    if (swapped) exchange_pairs(live, spare, swapped);
}

}  // namespace luce::common::gdn_commit
