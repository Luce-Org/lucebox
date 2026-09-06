#include "backend_plan_internal.h"

#include "feature_gate.h"
#include "model_capabilities.h"

#include <limits>
#include <utility>

namespace dflash::common::detail {

namespace {

BackendPreparationFailure failure(
    BackendPreparationError error,
    std::string message,
    std::vector<std::string> warnings = {}) {
    return {error, std::move(message), std::move(warnings)};
}

PlacementBackend resolve_target_backend(
    const BackendArgs & args,
    PlacementBackend compiled_backend) {
    return args.device.backend == PlacementBackend::Auto
        ? compiled_backend
        : args.device.backend;
}

}  // namespace

BackendPreparation BackendPlanBuilder::resolve(
    BackendArgs args,
    BackendAdmissionContext admission,
    GgufModelInfo model,
    PlacementBackend compiled_backend) {
    if (args.model_path.empty()) {
        return failure(
            BackendPreparationError::InvalidRequest,
            "model_path is empty");
    }

    if (model.arch.empty()) {
        return failure(
            BackendPreparationError::ModelInspection,
            "failed to detect architecture from " + args.model_path);
    }

    if (!arch_is_supported(model.arch)) {
        return failure(
            BackendPreparationError::ModelInspection,
            "unsupported model architecture '" + model.arch + "' in " +
                args.model_path);
    }

    const PlacementBackend target_backend =
        resolve_target_backend(args, compiled_backend);

    if (args.specla_mode && !args.ddtree_tau_explicit) {
        args.ddtree_tau = 6.0f;
    }

    // Preserve the original admission order. Warnings describe what the
    // operator requested, before model-specific SpecLA normalization changes
    // the effective construction request.
    std::string incompatible = check_feature_compatibility(
        args, admission, model.arch, target_backend, compiled_backend);
    if (!incompatible.empty()) {
        return failure(
            BackendPreparationError::FeatureCompatibility,
            std::move(incompatible));
    }

    std::vector<std::string> warnings =
        collect_feature_warnings(args, model.arch);

    BackendPlan::SpeclaEnvironmentAction specla_environment =
        BackendPlan::SpeclaEnvironmentAction::Preserve;

    if (args.specla_mode) {
        const bool supported =
            model.arch == "qwen35" && !args.device.is_multi_device();
        if (supported) {
            if (!args.draft_path.has_value()) {
                return failure(
                    BackendPreparationError::FeatureCompatibility,
                    "Qwen3.6 SpecLA requires --draft <path>",
                    std::move(warnings));
            }

            args.ddtree_mode = true;
            if (admission.kvflash_requested()) {
                warnings.push_back(
                    "--specla is unavailable with KVFlash; using ordinary "
                    "DDTree verification");
                args.specla_mode = false;
            }
        } else {
            warnings.push_back(
                "--specla is unavailable for architecture '" + model.arch +
                "' with placement " + placement_device_name(args.device) +
                "; using the architecture's normal decode path");
            args.specla_mode = false;
        }

        specla_environment = args.specla_mode
            ? BackendPlan::SpeclaEnvironmentAction::Enable
            : BackendPlan::SpeclaEnvironmentAction::Disable;
        if (!args.specla_mode && !args.ddtree_tau_explicit) {
            args.ddtree_tau = std::numeric_limits<float>::infinity();
        }
    }

    // Validate the exact snapshot construction will consume. This replaces
    // the old factory recheck against a second mutable BackendArgs object.
    incompatible = check_feature_compatibility(
        args, admission, model.arch, target_backend, compiled_backend);
    if (!incompatible.empty()) {
        return failure(
            BackendPreparationError::FeatureCompatibility,
            std::move(incompatible),
            std::move(warnings));
    }

    BackendPlan plan;
    plan.model_.path = std::move(args.model_path);
    plan.model_.metadata = std::move(model);

    plan.placement_.target = std::move(args.device);
    plan.placement_.draft = std::move(args.draft_device);
    plan.placement_.remote_draft = std::move(args.remote_draft);
    plan.placement_.remote_target_shard =
        std::move(args.remote_target_shard);

    plan.cache_.fa_window = args.fa_window;
    plan.cache_.paged_attention = args.paged_attention;
    plan.cache_.max_concurrency = args.max_concurrency;
    plan.cache_.kv_pool_tokens = args.kv_pool_tokens;
    plan.cache_.kq_stride_pad = args.kq_stride_pad;
    plan.cache_.draft_swa_window = args.draft_swa_window;
    plan.cache_.draft_ctx_max = args.draft_ctx_max;

    plan.speculation_.draft_path = std::move(args.draft_path);
    plan.speculation_.draft_block_size = args.draft_block_size;
    plan.speculation_.fast_rollback = args.fast_rollback;
    plan.speculation_.seq_verify = args.seq_verify;
    plan.speculation_.specla_mode = args.specla_mode;
    plan.speculation_.specla_top_k = args.specla_top_k;
    plan.speculation_.specla_top_k_explicit =
        args.specla_top_k_explicit;
    plan.speculation_.ddtree_mode = args.ddtree_mode;
    plan.speculation_.ddtree_budget = args.ddtree_budget;
    plan.speculation_.ddtree_temp = args.ddtree_temp;
    plan.speculation_.ddtree_chain_seed = args.ddtree_chain_seed;
    plan.speculation_.ddtree_tau = args.ddtree_tau;
    plan.speculation_.verify_width = args.verify_width;
    plan.speculation_.use_feature_mirror = args.use_feature_mirror;

    plan.execution_.stream_fd = args.stream_fd;
    plan.execution_.chunk = args.chunk;
    plan.deepseek4_.prefill_mode = args.ds4_prefill_mode;
    plan.deepseek4_.expert_top_k = args.ds4_expert_top_k;
    plan.deepseek4_.fused_decode = args.ds4_fused_decode;
    plan.deepseek4_.fused_verify_f16_kv =
        args.ds4_fused_verify_f16_kv;

    plan.warnings_ = std::move(warnings);
    plan.specla_environment_ = specla_environment;
    return plan;
}

}  // namespace dflash::common::detail
