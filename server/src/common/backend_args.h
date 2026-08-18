// Raw backend construction arguments.
//
// This contains only caller-requested configuration. Runtime facts derived
// from the model or compiled binary belong in ResolvedBackendPlan instead.

#pragma once

#include "placement/placement_config.h"
#include "placement/remote_draft_config.h"
#include "placement/remote_target_shard_config.h"
#include "prefill_attention_mode.h"

namespace dflash::common {

// Server-owned features that participate in backend admission even though
// they are not consumed by ModelBackend construction itself. Keep these
// separate from BackendArgs so the factory API remains usable by callers that
// do not run the HTTP server.
struct BackendFeatureConfig {
    bool pflash_enabled = false;
    bool pflash_drafter_configured = false;

    // MoE-only server features. Recorded here so the gate can report them as
    // inert on a dense architecture; both are applied via env vars at parse
    // time rather than through BackendArgs.
    bool routing_stats_requested = false;    // --freq / --collect-routing
    bool adaptive_experts_requested = false; // --adaptive-experts

    // A fixed KVFlash pool requested through DFLASH_KVFLASH. "auto" is
    // resolved later by the backend because only it has the VRAM budget needed
    // to know whether a pool will actually be active.
    bool kvflash_enabled = false;
};

// A superset of all per-architecture config fields. The factory reads only
// those relevant to the resolved architecture; unused fields are ignored.
struct BackendArgs {
    // Required
    const char *    model_path   = nullptr;   // target .gguf

    // Optional: speculative decode draft model (qwen35 only)
    const char *    draft_path   = nullptr;

    // Device placement
    DevicePlacement device;
    DevicePlacement draft_device;
    RemoteDraftConfig remote_draft;
    RemoteTargetShardConfig remote_target_shard;

    // I/O — only used when running under daemon_loop (legacy). The new
    // server passes -1 and uses on_token callbacks instead.
    int             stream_fd    = -1;

    // Chunked prefill
    int                  chunk                = 512;
    PrefillAttentionMode ds4_prefill_mode     = PrefillAttentionMode::Exact;
    bool                 ds4_prefill_mode_set = false;

    // deepseek4-specific decode options
    int             ds4_expert_top_k = 0;  // 0 = model default
    bool            ds4_fused_decode = false;
    bool            ds4_fused_verify_f16_kv = false;

    // Attention and speculative-decode options. Individual backends consume
    // only the fields they support.
    int             fa_window        = 0;  // 0 = full attention. qwen3.6 full-attn layers must see the whole context; a finite window drops the system prompt/tools -> breaks tool calls.
    bool            paged_attention  = false;  // 16-token paged K/V blocks for AR decode
    // Concurrent decode slots (--max-concurrency). > 1 requires paged_attention;
    // the backend serves that many sequences through the seq_* slot API.
    int             max_concurrency  = 1;
    // Total paged K/V pool in tokens shared by all slots (--kv-pool-tokens;
    // block-rounded). 0 = derive capacity from available device memory.
    long long       kv_pool_tokens   = 0;
    int             kq_stride_pad    = 32;
    int             draft_swa_window = 0;
    int             draft_ctx_max    = 4096;
    bool            fast_rollback    = true;
    bool            seq_verify       = false;
    bool            ddtree_mode      = false;
    int             ddtree_budget    = 22;
    float           ddtree_temp      = 1.0f;
    bool            ddtree_chain_seed = true;
    int             verify_width     = 0;  // chain spec verify width; 0 = adaptive
    bool            use_feature_mirror = false;
};

}  // namespace dflash::common
