// Raw backend construction arguments.
//
// This contains only caller-requested configuration. Runtime facts derived
// from the model or compiled binary belong in BackendPlan instead.

#pragma once

#include <limits>
#include <optional>
#include <string>

#include "placement/draft_residency.h"
#include "placement/placement_config.h"
#include "placement/remote_draft_config.h"
#include "placement/remote_target_shard_config.h"
#include "prefill_attention_mode.h"

namespace dflash::common {

enum class KvFlashRequest {
    Off,
    Auto,
    Fixed,
};

// Server-owned facts that participate in backend admission without becoming
// backend construction arguments. This is the only projection the HTTP server
// may pass into backend preparation.
struct BackendAdmissionContext {
    bool pflash_enabled = false;
    bool pflash_drafter_configured = false;
    DraftResidencyPolicy draft_residency = DraftResidencyPolicy::Auto;

    // Automatic sizing remains backend-owned because only the initialized
    // backend has the VRAM budget. Fixed pools can participate in admission.
    KvFlashRequest kvflash = KvFlashRequest::Off;

    bool kvflash_requested() const {
        return kvflash != KvFlashRequest::Off;
    }
    bool fixed_kvflash_requested() const {
        return kvflash == KvFlashRequest::Fixed;
    }
};

// A superset of all per-architecture config fields. Preparation projects only
// the effective fields into BackendPlan's concern-specific snapshots.
struct BackendArgs {
    // Required
    std::string model_path;  // target .gguf

    // Optional: speculative decode draft model (qwen35 only)
    std::optional<std::string> draft_path;

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
    int             draft_block_size = 0;  // 0 = drafter metadata
    int             draft_swa_window = 0;
    int             draft_ctx_max    = 4096;
    bool            fast_rollback    = true;
    bool            seq_verify       = false;
    bool            specla_mode      = false;
    int             specla_top_k     = 4;
    bool            specla_top_k_explicit = false;
    bool            ddtree_mode      = false;
    int             ddtree_budget    = 22;
    float           ddtree_temp      = 1.0f;
    bool            ddtree_chain_seed = true;
    float           ddtree_tau       = std::numeric_limits<float>::infinity();
    bool            ddtree_tau_explicit = false;
    int             verify_width     = 0;  // chain spec verify width; 0 = adaptive
    bool            use_feature_mirror = false;

    // MoE backend requests. The server currently realizes these through
    // environment variables, but admission still treats them as explicit
    // operator input rather than server-owned context.
    bool            routing_stats_requested = false;
    bool            adaptive_experts_requested = false;
};

}  // namespace dflash::common
