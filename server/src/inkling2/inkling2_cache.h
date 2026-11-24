// Inkling2Cache — 48-slot hybrid KV (40 SWA ring + 8 Global) + 48-slot ShortConv state
// + DSpark 3-layer KV cache + speculative verify rollback ring + Split-KV remote pool
// + HIP-graph-cached decode/verify/draft workspaces.

#pragma once

#include "inkling2_internal.h"
#include "inkling2_kvflash.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace luce::common {

constexpr int kInkling2MaxVerify     = 8;   // 1 anchor + up to 7 DSpark drafts
constexpr int kInkling2TwoDSDMaxNodes = 128; // live tree slots (slot 128 = scratch padding)
constexpr int kInkling2TwoDSDMaxCommit = 16; // max committed path length per step
constexpr int kInkling2TwoDSDFlagsLater   = 1;
constexpr int kInkling2TwoDSDFlagsPadding = 2;

struct Inkling2GraphWorkspace {
    ggml_context *       ctx     = nullptr;
    ggml_gallocr_t       alloc   = nullptr;
    ggml_backend_sched_t sched   = nullptr;
    ggml_cgraph *        gf      = nullptr;
    bool                 built   = false;
    int                  n_tokens = 0;
    bool                 verify  = false;
    bool                 split   = false;
    bool                 split_kv = false;
    bool                 paged_kv = false;
    bool                 capture_taps = false;
    bool                 capture_q = false;
    int                  q_cap_n = 0;

    // Runtime input tensors (updated via fast async H2D before each replay)
    ggml_tensor * inp_tokens    = nullptr; // I32 [T]
    ggml_tensor * inp_pos       = nullptr; // I32 [T]
    ggml_tensor * inp_meta      = nullptr; // I32 [4]: {pos0, kv_len, kv_start, kv_end}
    ggml_tensor * inp_meta_r    = nullptr; // I32 [4]: remote Split-KV slice {pos0, kv_len, 0, split_pos}
    ggml_tensor * inp_bt_local  = nullptr; // I32 [max_blocks] when KVFlash paged global KV active
    ggml_tensor * inp_bt_remote = nullptr; // I32 [max_blocks] when Dual-GPU KVFlash Tier 1.5 active
    uint64_t      bt_epoch      = 0;       // cached block table epoch uploaded to inp_bt_*

    // Outputs
    ggml_tensor * logits     = nullptr; // F32 [n_vocab_unpadded, T] (or [n_vocab_unpadded, 1] for prefill)
    ggml_tensor * greedy     = nullptr; // I32 [T]
    ggml_tensor * taps_out   = nullptr; // F32 [n_embd, 4, T] when capture_taps==true
    ggml_tensor * q_cap_out  = nullptr; // F32 [D, Hq, L_global, q_cap_n] when capture_q==true

    // Cached 1-node GPU top-candidate extractor for ultra-fast exact stochastic sampling (temp > 0)
    ggml_context * sample_ctx    = nullptr;
    ggml_gallocr_t sample_alloc  = nullptr;
    ggml_cgraph *  sample_gf     = nullptr;
    ggml_tensor *  sample_logits = nullptr; // leaf alias pointing to logits->data
    ggml_tensor *  sample_cands  = nullptr; // I32 [516, T]

    std::vector<ggml_tensor *> expert_nodes;
};

struct Inkling2DSparkStoreSlot {
    ggml_context * ctx   = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph *  gf    = nullptr;
    ggml_tensor *  taps  = nullptr; // F32 [n_embd, 4, T]
    ggml_tensor *  pos   = nullptr; // I32 [T]
    bool           built = false;
};

struct Inkling2DSparkWorkspace {
    // Per-retained-count context-store graphs (1..8) -> ZERO graph rebuilds & ZERO HIP graph churn!
    std::array<Inkling2DSparkStoreSlot, kInkling2MaxVerify + 1> store;

    // Draft block-8 + 7-step fused RNN Markov chain graph (1 graph, <1.2 ms!)
    ggml_context * ctx_draft   = nullptr;
    ggml_gallocr_t alloc_draft = nullptr;
    ggml_cgraph *  gf_draft    = nullptr;
    ggml_tensor *  draft_anchor     = nullptr; // I32 [1]
    ggml_tensor *  draft_pos        = nullptr; // I32 [8]
    ggml_tensor *  draft_zero_state = nullptr; // F32 [rank, 1] (zeroed once)
    ggml_tensor *  draft_mask       = nullptr; // F16 [kv_span, 8] (global layer 1)
    ggml_tensor *  draft_mask_local = nullptr; // F16 [kv_span, 8] (local SWA layers 0 & 2)
    ggml_tensor *  draft_out_ids[7] = {};      // I32 [1] each -> drafted token IDs
    ggml_tensor *  draft_out_all    = nullptr; // I32 [draft_n_spec] packed output (1 single D2H readback!)
    int64_t        draft_kv_span = 0;
    int            draft_n_spec  = 0;
    bool           draft_built   = false;

    // Conv-state rollback graph (1 launch restores all 48 slots from verify ring!)
    ggml_context * ctx_rb      = nullptr;
    ggml_gallocr_t alloc_rb    = nullptr;
    ggml_cgraph *  gf_rb       = nullptr;
    ggml_tensor *  rb_slot     = nullptr; // I32 [1]
    bool           rb_built    = false;
};

// Cached static-bucket HIP graph for one 2DSD target phase (prelude / core / coda).
struct Inkling2TwoDSDPhaseGraph {
    ggml_context *       ctx   = nullptr;
    ggml_gallocr_t       alloc = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph *        gf    = nullptr;
    bool                 built = false;

    ggml_tensor * inp_meta         = nullptr; // I32 [m_stride, T]
    ggml_tensor * inp_prefix_len   = nullptr; // contiguous: I32 [1] or [P]; paged: I32 [2 + max_blocks, n_seq_slots]
    ggml_tensor * inp_prefix_len_r = nullptr; // paged Tier 1.5: I32 [2 + max_blocks, n_seq_slots] on expert_backend
    ggml_tensor * inp_pos          = nullptr; // I32 [T]
    ggml_tensor * inp_flags      = nullptr; // I32 [T]
    ggml_tensor * inp_tokens     = nullptr; // I32 [T] (prelude only)
    ggml_tensor * inp_slot       = nullptr; // I32 [T] (0..S-1)
    ggml_tensor * inp_flat_cur   = nullptr; // I32 [T] (plane*S + slot)
    ggml_tensor * inp_flat_prev  = nullptr; // I32 [T] (max(0,plane-1)*S + slot)
    ggml_tensor * out_logits     = nullptr; // F32 [n_vocab_unpadded, T] (coda only)
    ggml_tensor * out_greedy     = nullptr; // I32 [T] (coda only)
    // Lazy cached 1-node GPU top-candidate extractor for fast non-greedy sampling (subop 32)
    ggml_context * sample_ctx    = nullptr;
    ggml_gallocr_t sample_alloc  = nullptr;
    ggml_cgraph *  sample_gf     = nullptr;
    ggml_tensor *  sample_logits = nullptr; // leaf alias to out_logits->data
    ggml_tensor *  sample_cands  = nullptr; // I32 [516, T]
    float          sample_temp   = 0.0f;
    std::vector<ggml_tensor *> expert_nodes;
};

// Cached static-bucket HIP graph for one 2DSD multi-leaf loop-context draft batch.
struct Inkling2TwoDSDDraftGraph {
    ggml_context * ctx   = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph *  gf    = nullptr;
    bool           built = false;

    ggml_tensor * inp_heads      = nullptr; // I32 [L_b]
    ggml_tensor * inp_block_pos  = nullptr; // I32 [8 * L_b]
    ggml_tensor * inp_block_wpos = nullptr; // I32 [8 * L_b]
    ggml_tensor * inp_span_idx   = nullptr; // I32 [S_b * L_b] (depth_idx*S + slot)
    ggml_tensor * inp_span_pos   = nullptr; // I32 [S_b * L_b]
    ggml_tensor * inp_span_wpos  = nullptr; // I32 [S_b * L_b]
    ggml_tensor * inp_draft_meta = nullptr; // I32 [3 + L_b]
    ggml_tensor * out_tokens     = nullptr; // I32 [L_b, 7] (7 steps x L_b leaves)
    ggml_tensor * out_conf       = nullptr; // F32 [7 * L_b]
};

struct Inkling2TwoDSDCommitGraph {
    ggml_context * ctx         = nullptr;
    ggml_gallocr_t alloc       = nullptr;
    ggml_cgraph *  gf          = nullptr;
    ggml_tensor *  slots       = nullptr; // I32 [kInkling2TwoDSDMaxCommit]
    ggml_tensor *  meta        = nullptr; // I32 [2]: {prefix_len, n_commit}
    ggml_tensor *  block_table = nullptr; // I32 [max_blocks] when KVFlash paged global KV active
    bool           built       = false;
};

struct Inkling2TwoDSDPrefixStoreGraph {
    ggml_context * ctx   = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph *  gf    = nullptr;
    ggml_tensor *  feats = nullptr; // F16 [5*H, T] (or gather idx I32 [T])
    ggml_tensor *  pos   = nullptr; // I32 [T] (true absolute RoPE position)
    ggml_tensor *  wpos  = nullptr; // I32 [T] (compact draft KV write slot = pos - draft_pref_start)
    bool           built = false;
};

// 60-plane contiguous target cache + tree scratch + 5-layer drafter KV + graph caches.
struct Inkling2TreeCache {
    bool active       = false;
    bool is_slot_view = false; // true for zero-copy per-slot views aliasing a multi-slot cache
    int  n_seq_slots  = 1;     // >1 when backing multi-stream Inkling2SeqEngine
    int  max_nodes    = kInkling2TwoDSDMaxNodes; // live slots [0..max_nodes-1]; scratch = max_nodes
    int  n_depths     = 3;                       // coda planes for readout depths {1, 2, 3}
    int  prefix_len       = 0;
    int  draft_pref_start = 0;                   // tail-only drafter prefix start (0 = full prefix)
    int  global_pool_cap  = 0;                   // global prefix_k/v resident token pool (max_ctx or tier0_pool)
    int  shared_kv_window = 0;                   // >0 enables Last-Loop Shared-KV (4 cold global planes)
    int  shared_tail_cap  = 0;                   // ring capacity per early plane packed at prefix_k/v tail
    int  draft_ring_cap   = 0;                   // >0 enables sliding ctx-KV ring [0..draft_ring_cap-1] for O(1) VRAM

    // Flat 4D per-physical-layer tensors (24 physical layers):
    //   planes[ip]: 1 for prelude (0..5), 3 for core (6..17), 3 for coda (18..23) -> 60 planes total
    std::vector<int>           planes;          // size 24
    std::vector<ggml_tensor *> prefix_k;        // [D, cap, Hk, P_storage] (Q8_0 or F16)
    std::vector<ggml_tensor *> prefix_v;        // [D, cap, Hk, P_storage]
    std::vector<ggml_tensor *> remote_prefix_k; // Dual-GPU Tier 1.5 warm prefix [D, remote_cap, Hk, P_storage] (global layers only)
    std::vector<ggml_tensor *> remote_prefix_v; // Dual-GPU Tier 1.5 warm prefix [D, remote_cap, Hk, P_storage] (global layers only)
    std::vector<ggml_tensor *> tree_k;          // [D, S, Hk, P_tot]
    std::vector<ggml_tensor *> tree_v;          // [D, S, Hk, P_tot]
    std::vector<ggml_tensor *> prefix_conv;     // [3, sconv_channels, P_tot] F32
    std::vector<ggml_tensor *> tree_conv;       // [sconv_channels, S, P_tot] F32

    // Per-node recurrent & feature pools (S = max_nodes + 1):
    std::vector<ggml_tensor *> pre_blocks;   // [H, S] F32 each (1 block for prelude L0..5)
    ggml_tensor *              pre_partial  = nullptr; // [H, S] F32
    ggml_tensor *              pre_tap      = nullptr; // [H, S] F32 (layer 2 input)
    std::vector<ggml_tensor *> core_blocks;  // [H, S*3] F32 each (2 blocks for core L6..17)
    ggml_tensor *              core_partial = nullptr; // [H, S*3] F32
    std::array<ggml_tensor *, 3> core_taps  = {};      // [H, S*3] F32 (layers 7, 12, 17 inputs)
    ggml_tensor *              node_features = nullptr; // [5*H, S*3] F16 (depth_idx*S + slot)

    // 5-layer loop-context drafter prefix KV cache (F16 [128, max_ctx + 256, 8] per layer)
    std::vector<ggml_tensor *> draft_k; // size 5
    std::vector<ggml_tensor *> draft_v; // size 5
    int                        draft_cap = 0;

    // Zero-copy view context aliasing legacy 48-slot k_cache/v_cache/conv_state into prefix_*
    ggml_context * alias_ctx = nullptr;

    // Cached HIP graphs:
    // phase_key = (phase_id: 0=prelude 1=core 2=coda, row_bucket, path_bucket, has_later, use_split)
    std::map<std::tuple<int, int, int, int, int>, Inkling2TwoDSDPhaseGraph> phase_graphs;
    // draft_key = (L_bucket, S_bucket, kv_span_bucket)
    std::map<std::tuple<int, int, int64_t>, Inkling2TwoDSDDraftGraph>       draft_graphs;
    Inkling2TwoDSDCommitGraph                                               commit_graph;
    // Per-commit-count prefix KV store graphs (gather directly from node_features[2, slot]!)
    std::array<Inkling2TwoDSDPrefixStoreGraph, kInkling2TwoDSDMaxCommit + 1> prefix_store_graphs;
};

struct Inkling2Cache {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // Optional Dual-GPU Split-KV remote pool on expert_backend (Strix Halo) for global layers
    ggml_backend_t        remote_backend = nullptr;
    ggml_backend_t        split_cpu      = nullptr; // owned CPU fallback backend required by ggml_backend_sched_new
    ggml_context *        remote_ctx     = nullptr;
    ggml_backend_buffer_t remote_buf     = nullptr;
    // When remote_split_pos > 0: Strix holds global keys [0, remote_split_pos),
    // R9700 holds global keys [remote_split_pos, max_ctx).
    int                   remote_split_pos = 0;
    int                   remote_cap       = 0;

    ggml_type kv_type = GGML_TYPE_Q8_0;
    int       max_ctx = 0;
    int       swa_cap = 0; // SWA ring capacity (>= swa_window + max_chunk)
    int       global_local_cap = 0;
    int       cur_pos = 0;

    // 48-slot target caches (indexed by cache slot 0..47)
    // Shape: [head_dim, cap, n_head_kv]
    std::vector<ggml_tensor *> k_cache;
    std::vector<ggml_tensor *> v_cache;
    // Dual-GPU Split-KV remote global caches (size = n_global = 8, on remote_backend)
    std::vector<ggml_tensor *> remote_k;
    std::vector<ggml_tensor *> remote_v;

    // 48-slot ShortConv live state: [3, sconv_channels] F32 per slot
    std::vector<ggml_tensor *> conv_state;
    // 48-slot ShortConv verify ring: [3, sconv_channels, kInkling2MaxVerify] F32 per slot
    std::vector<ggml_tensor *> conv_ring;

    // 3-layer DSpark draft KV cache (on target backend, F16 [head_dim, max_ctx_draft, n_head_kv])
    ggml_context *        dspark_kv_ctx = nullptr;
    ggml_backend_buffer_t dspark_kv_buf = nullptr;
    std::vector<ggml_tensor *> dspark_k; // size = 3
    std::vector<ggml_tensor *> dspark_v; // size = 3
    int                   dspark_ctx_pos = 0; // number of target tokens stored in draft KV

    // Persistent device buffer holding the last committed token's 4 target taps [n_embd, 4, 1]
    ggml_tensor * last_taps_dev = nullptr;

    // Optional 3-Tier Dual-GPU KVFlash 2.0 manager (active when LUCE_KVFLASH > 0)
    ggml_context *                   kvflash_alias_ctx = nullptr;
    std::unique_ptr<Inkling2KvFlash> kvflash;
    // Cached KVFlash config used at cache creation so multi-slot views can attach per-slot managers
    Inkling2KvFlashConfig            kvflash_cfg;
    ggml_backend_t                   target_backend_ref = nullptr;

    // Cached HIP-graph workspaces
    Inkling2GraphWorkspace decode_ws;
    std::array<Inkling2GraphWorkspace, kInkling2MaxVerify + 1> verify_ws;
    Inkling2DSparkWorkspace dspark_ws;
    Inkling2TreeCache       tree;
};

bool create_inkling2_cache(ggml_backend_t target_backend,
                           ggml_backend_t expert_backend,
                           const Inkling2Weights & w,
                           int max_ctx,
                           int max_chunk,
                           ggml_type kv_type,
                           Inkling2Cache & out,
                           std::string & error,
                           int fixed_remote_split_pos = -1,
                           int n_seq_slots = 1);

// Create a zero-copy single-slot Inkling2Cache view into slot `seq_slot` of a multi-slot 2DSD cache.
// The view shares device memory with `multi` (does not own buffers) and owns its own single-slot
// commit/draft/prefix-store graph caches so existing prefill/commit/draft routines run unchanged.
bool make_inkling2_slot_view(Inkling2Cache & multi,
                             const Inkling2Weights & w,
                             int seq_slot,
                             Inkling2Cache & view_out);

void reset_inkling2_cache(Inkling2Cache & cache);
void free_inkling2_cache(Inkling2Cache & cache);
void free_inkling2_workspace(Inkling2GraphWorkspace & ws);

} // namespace luce::common
