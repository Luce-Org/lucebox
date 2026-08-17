// Cross-feature compatibility gate — see feature_gate.h for what belongs here.

#include "feature_gate.h"

#include "model_capabilities.h"
#include "paged_attention_config.h"

#include <climits>

namespace dflash::common {

std::string check_feature_compatibility(
    const BackendArgs & args,
    const BackendFeatureConfig & features,
    const std::string & arch,
    PlacementBackend    target_backend,
    PlacementBackend    compiled_backend)
{
    if (arch.empty()) {
        return "failed to detect model architecture";
    }

    // ── target placement × compiled backend
    if (target_backend != compiled_backend) {
        return "--target-device=" + placement_device_name(args.device) +
               " is unsupported in this binary (compiled backend: " +
               placement_backend_name(compiled_backend) + ")";
    }

    const PlacementBackend draft_backend =
        args.draft_device.backend == PlacementBackend::Auto
            ? target_backend
            : args.draft_device.backend;
    const bool draft_placement_used =
        features.pflash_enabled || args.draft_path != nullptr;
    const bool mixed_draft_placement =
        draft_placement_used && target_backend != draft_backend;

    // ── IPC auxiliary options × IPC enablement
    if (!args.remote_draft.enabled() &&
        args.remote_draft.has_aux_options()) {
        return "--draft-ipc-work-dir and --draft-ipc-ring-cap require "
               "--draft-ipc-bin";
    }
    if (!args.remote_target_shard.enabled() &&
        args.remote_target_shard.has_aux_options()) {
        return "--target-shard-ipc-work-dir requires --target-shard-ipc-bin";
    }

    // ── PFlash enablement × drafter model
    if (features.pflash_enabled &&
        !features.pflash_drafter_configured) {
        return "--prefill-compression requires --prefill-drafter";
    }

    // ── target/draft backend mixing × remote draft IPC
    if (mixed_draft_placement && !args.remote_draft.enabled()) {
        return "mixed target/draft backends require --draft-ipc-bin "
               "(target=" + std::string(placement_backend_name(target_backend)) +
               " draft=" + placement_backend_name(draft_backend) + ")";
    }
    if (!mixed_draft_placement && args.remote_draft.enabled()) {
        return "--draft-ipc-bin is only needed for mixed target/draft "
               "backends (target=" +
               std::string(placement_backend_name(target_backend)) +
               " draft=" + placement_backend_name(draft_backend) + ")";
    }
    // ── target split structure and remote backend topology
    const bool tensor_mode =
        args.device.split_mode == TargetSplitMode::Tensor;
    if (!args.device.is_layer_split() &&
        !args.device.layer_split_weights.empty()) {
        return tensor_mode
            ? "--target-layer-split is incompatible with tensor parallelism"
            : "--target-layer-split requires --target-devices";
    }
    if (args.device.is_multi_device() || tensor_mode) {
        const std::string placement_error =
            validate_device_placement(args.device, /*device_count=*/-1);
        if (!placement_error.empty()) {
            return "bad target placement: " + placement_error;
        }
    }

    // The target-only implementation is deliberately narrow: every rank must
    // be a local CUDA device and target-replacing features are unsupported.
    if (tensor_mode) {
        if (arch != "qwen35") {
            return "tensor parallelism is currently supported only for dense qwen35";
        }
        if (target_backend != PlacementBackend::Cuda ||
            compiled_backend != PlacementBackend::Cuda) {
            return "tensor parallelism currently requires local CUDA devices";
        }
        if (args.device.is_mixed_layer_split()) {
            return "tensor parallelism requires homogeneous local devices";
        }
        if (args.remote_target_shard.enabled()) {
            return "tensor parallelism is incompatible with --target-shard-ipc-bin";
        }
        if (features.pflash_enabled) {
            return "tensor parallelism does not yet support prefill compression";
        }
    }

    const bool mixed_target_split =
        args.device.is_layer_split() &&
        args.device.is_mixed_layer_split();
    if (mixed_target_split) {
        if (!args.remote_target_shard.enabled()) {
            return "mixed-backend target layer split requires "
                   "--target-shard-ipc-bin";
        }

        size_t remote_begin = 0;
        while (remote_begin < args.device.layer_split_gpus.size() &&
               args.device.layer_split_backend(remote_begin) ==
                   compiled_backend) {
            ++remote_begin;
        }
        if (remote_begin == 0 ||
            remote_begin >= args.device.layer_split_gpus.size()) {
            return "mixed-backend target layer split currently supports "
                   "one local backend group followed by one remote backend "
                   "group";
        }

        const PlacementBackend remote_backend =
            args.device.layer_split_backend(remote_begin);
        for (size_t i = remote_begin;
             i < args.device.layer_split_gpus.size();
             ++i) {
            if (args.device.layer_split_backend(i) != remote_backend) {
                return "mixed-backend target layer split currently supports "
                       "only one backend boundary";
            }
        }
    }

    // ── layer split × architecture
    // qwen35moe and qwen3 have no layer-split adapter. Their factory cases
    // hand the split DevicePlacement to a monolithic backend, which reads
    // only the primary GPU — the extra devices are silently unused. Reject
    // instead: a multi-device placement that quietly becomes single-device
    // fails later as an out-of-memory, far from its cause.
    if (args.device.is_layer_split() && !arch_supports_layer_split(arch)) {
        return "model architecture '" + arch +
               "' has no layer-split path; --target-devices would run on " +
               placement_device_name(args.device) + " alone";
    }

    // ── remote draft execution × architecture
    if (args.remote_draft.enabled() && args.draft_path &&
        !arch_supports_remote_draft(arch)) {
        return "model architecture '" + arch +
               "' does not support remote draft execution";
    }

    // ── mixed-backend PFlash × architecture
    if (features.pflash_enabled && mixed_draft_placement &&
        !arch_supports_pflash_compression(arch)) {
        return "model architecture '" + arch +
               "' does not support PFlash compression";
    }

    // ── --paged-attention × architecture, placement, and decode features
    // Paged decode swaps the contiguous K/V cache for a block table owned by
    // the monolithic qwen35 backend, so every rule below is about reaching
    // that one code path. All are errors rather than warnings: running dense
    // instead would hide the memory behavior the flag was chosen for.
    if (args.paged_attention) {
        if (!arch_supports_paged_attention(arch, /*is_layer_split=*/false)) {
            return "--paged-attention requires a Qwen3.5/Qwen3.6 dense target "
                   "(architecture '" + arch + "' has no paged decode path)";
        }
        // No rule for "requires a CUDA or HIP build": those are the only two
        // backends this binary can be configured with, and GGML_OP_PAGED_ATTN
        // is compiled into both.
        if (args.device.is_layer_split() ||
            args.remote_target_shard.enabled()) {
            return "--paged-attention requires one local target device";
        }
        if (args.draft_path != nullptr || args.remote_draft.enabled() ||
            args.ddtree_mode) {
            return "--paged-attention requires autoregressive decode without a "
                   "draft or DDTree";
        }
        if (args.fa_window != 0) {
            return "--paged-attention requires full attention (--fa-window 0)";
        }
        if (features.pflash_enabled) {
            return "--paged-attention cannot be combined with PFlash prefill "
                   "compression";
        }
        if (features.kvflash_enabled) {
            return "--paged-attention cannot be combined with KVFlash";
        }
        // The pool rounds max_ctx up to a whole number of blocks, so the top
        // of the range is what can be rounded without overflowing int.
        if (args.device.max_ctx <= 0 ||
            args.device.max_ctx > INT_MAX - PAGED_BLOCK_SIZE + 1) {
            return "--paged-attention requires a positive --max-ctx small "
                   "enough to round up to whole blocks";
        }
    }

    // ── --max-concurrency × paged attention
    // Concurrent decode slots are currently implemented only by the paged
    // qwen35 backend. The common scheduler does not require a particular
    // model-state representation; each backend owns whatever per-slot state
    // its graph needs alongside one block-table column per sequence.
    // Everything the paged cluster above rejects is transitively rejected,
    // so the rules here are only about the flag pair itself.
    if (args.max_concurrency < 1) {
        return "--max-concurrency must be at least 1";
    }
    if (args.max_concurrency > 1) {
        if (!args.paged_attention) {
            return "--max-concurrency requires --paged-attention";
        }
        // The paged pool addresses tokens with uint32; 64 slots is far above
        // any batch the decode kernel has been sized for and keeps the
        // fixed-width decode batch bounded.
        if (args.max_concurrency > 64) {
            return "--max-concurrency must be at most 64";
        }
        // Physical capacity is memory-derived and capped independently of the
        // logical slot count, so max-concurrency no longer multiplies max_ctx
        // in the pool's tensor address space.
    }
    if (args.kv_pool_tokens != 0) {
        if (args.max_concurrency <= 1) {
            return "--kv-pool-tokens requires --max-concurrency greater than 1";
        }
        // The cache appends one scratch block after the physical pool, and
        // the requested pool itself is rounded up to a whole block. Cap the
        // request at the largest aligned pool that leaves room for scratch.
        const int64_t max_pool_tokens = paged_kv_address_cap();
        if (args.kv_pool_tokens < PAGED_BLOCK_SIZE ||
            args.kv_pool_tokens > max_pool_tokens) {
            return "--kv-pool-tokens must be in [" +
                   std::to_string(PAGED_BLOCK_SIZE) + ", " +
                   std::to_string(max_pool_tokens) + "]";
        }
    }

    // ── --ds4-prefill × architecture
    if (args.ds4_prefill_mode_set && arch != "deepseek4") {
        return "--ds4-prefill is only valid for deepseek4 models (detected '" +
               arch + "')";
    }

    // Approximate prefill and fused decode are implemented only in the
    // monolithic HIP DeepSeek4 backend. Expert top-k is model policy handled
    // by either monolithic backend, but the layer-split adapter does not yet
    // propagate it.
    const bool monolithic_ds4 =
        arch == "deepseek4" &&
        target_backend == PlacementBackend::Hip &&
        !args.device.is_layer_split() &&
        !args.remote_target_shard.enabled();
    const bool local_ds4 =
        arch == "deepseek4" &&
        !args.device.is_layer_split() &&
        !args.remote_target_shard.enabled();

    // ── approximate --ds4-prefill × placement
    if (arch == "deepseek4" &&
        prefill_attention_mode_is_approximate(args.ds4_prefill_mode) &&
        !monolithic_ds4) {
        return std::string("DS4 ") +
               prefill_attention_mode_name(args.ds4_prefill_mode) +
               " prefill requires a single local HIP target; use "
               "--ds4-prefill exact for split, remote, or CUDA placement";
    }

    // ── --ds4-fused-decode × placement
    if (args.ds4_fused_decode && !monolithic_ds4) {
        return "--ds4-fused-decode currently requires single-device HIP "
               "DeepSeek4";
    }

    // ── --ds4-expert-top-k × architecture/adapter
    if (args.ds4_expert_top_k != 0 && !local_ds4) {
        return "--ds4-expert-top-k currently requires a single local "
               "DeepSeek4 backend";
    }

    return {};
}

namespace {

// Emit "<flag> ignored: ..." when a requested option does not reach the
// backend for this architecture and placement. `supported_monolithic` lets
// the message distinguish "this architecture never supports it" from "this
// architecture supports it, but not when layer-split".
void warn_inert(std::vector<std::string> & out,
                bool requested,
                bool supported_here,
                bool supported_monolithic,
                bool is_layer_split,
                const std::string & arch,
                const char * flag,
                const char * feature) {
    if (!requested || supported_here) return;
    if (is_layer_split && supported_monolithic) {
        out.push_back(std::string(flag) + " ignored: architecture '" + arch +
                      "' provides " + feature +
                      " only on single-device placement");
    } else {
        out.push_back(std::string(flag) + " ignored: architecture '" + arch +
                      "' has no " + feature + " support");
    }
}

}  // namespace

std::vector<std::string> collect_feature_warnings(
    const BackendArgs & args,
    const BackendFeatureConfig & features,
    const std::string & arch)
{
    std::vector<std::string> out;
    const bool split = args.device.is_layer_split();

    // Each entry pairs a requested option with the capability predicate for
    // the field create_backend() would have to forward for it to take effect.
    warn_inert(out, args.draft_path != nullptr,
               arch_supports_decode_draft(arch, split),
               arch_supports_decode_draft(arch, false),
               split, arch, "--draft", "speculative decode");

    warn_inert(out, args.ddtree_mode,
               arch_supports_ddtree(arch, split),
               arch_supports_ddtree(arch, false),
               split, arch, "--ddtree", "DDTree speculative decode");

    warn_inert(out, args.verify_width != 0,
               arch_supports_verify_width(arch, split),
               arch_supports_verify_width(arch, false),
               split, arch, "--verify-width", "chain-spec verify width");

    warn_inert(out, args.fa_window != 0,
               arch_supports_fa_window(arch, split),
               arch_supports_fa_window(arch, false),
               split, arch, "--fa-window", "flash-attention sliding window");

    warn_inert(out, args.draft_swa_window != 0,
               arch_supports_draft_swa(arch, split),
               arch_supports_draft_swa(arch, false),
               split, arch, "--draft-swa", "draft sliding-window attention");

    // MoE-only server features. These drive the DFLASH_QWEN35MOE_* /
    // DFLASH_LAGUNA_* env vars, which a dense backend never reads.
    if (features.routing_stats_requested && !arch_has_expert_offload(arch)) {
        out.push_back("--freq/--collect-routing ignored: architecture '" +
                      arch + "' has no expert routing to record");
    }
    if (features.adaptive_experts_requested && !arch_has_expert_offload(arch)) {
        out.push_back("--adaptive-experts ignored: architecture '" + arch +
                      "' has no expert-count gating");
    }

    return out;
}

}  // namespace dflash::common
