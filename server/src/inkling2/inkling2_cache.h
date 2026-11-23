// Inkling2Cache — 48-slot hybrid KV (40 SWA ring + 8 Global) + 48-slot ShortConv state
// + DSpark 3-layer KV cache + speculative verify rollback ring + Split-KV remote pool
// + HIP-graph-cached decode/verify/draft workspaces.

#pragma once

#include "inkling2_internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <vector>

namespace luce::common {

constexpr int kInkling2MaxVerify = 8; // 1 anchor + up to 7 DSpark drafts

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
    bool                 capture_taps = false;

    // Runtime input tensors (updated via fast async H2D before each replay)
    ggml_tensor * inp_tokens = nullptr; // I32 [T]
    ggml_tensor * inp_pos    = nullptr; // I32 [T]
    ggml_tensor * inp_meta   = nullptr; // I32 [4]: {pos0, kv_len, kv_start, kv_end}
    ggml_tensor * inp_meta_r = nullptr; // I32 [4]: remote Split-KV slice {pos0, kv_len, 0, split_pos}

    // Outputs
    ggml_tensor * logits     = nullptr; // F32 [n_vocab_unpadded, T] (or [n_vocab_unpadded, 1] for prefill)
    ggml_tensor * greedy     = nullptr; // I32 [T]
    ggml_tensor * taps_out   = nullptr; // F32 [n_embd, 4, T] when capture_taps==true

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

    // Cached HIP-graph workspaces
    Inkling2GraphWorkspace decode_ws;
    std::array<Inkling2GraphWorkspace, kInkling2MaxVerify + 1> verify_ws;
    Inkling2DSparkWorkspace dspark_ws;
};

bool create_inkling2_cache(ggml_backend_t target_backend,
                           ggml_backend_t expert_backend,
                           const Inkling2Weights & w,
                           int max_ctx,
                           int max_chunk,
                           ggml_type kv_type,
                           Inkling2Cache & out,
                           std::string & error,
                           int fixed_remote_split_pos = -1);

void reset_inkling2_cache(Inkling2Cache & cache);
void free_inkling2_cache(Inkling2Cache & cache);
void free_inkling2_workspace(Inkling2GraphWorkspace & ws);

} // namespace luce::common
