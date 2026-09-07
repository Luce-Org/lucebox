// Backend factory implementation.

#include "backend_factory.h"
#include "backend_plan_internal.h"
#include "gguf_inspect.h"
#include "model_capabilities.h"
#include "platform_env.h"

#include "qwen35_backend.h"
#include "qwen35moe_backend.h"
#include "bailingmoe3_backend.h"
#include "laguna_backend.h"
#include "laguna_layer_split_adapter.h"
#include "qwen3_backend.h"
#include "gemma4_backend.h"
#include "gemma4_layer_split_adapter.h"
#include "deepseek4_backend.h"
#include "deepseek4_layer_split_adapter.h"
#include "layer_split_backend.h"
#include "qwen35_layer_split_adapter.h"

#include <cstdio>
#include <algorithm>
#include <optional>
#include <type_traits>
#include <utility>

namespace dflash::common {

namespace {

// ─── Capability table ↔ dispatch cross-check ────────────────────────────
// Every option an architecture can receive arrives through a named field on
// its backend config struct. Detecting whether that field exists lets the
// compiler verify model_capabilities.h against the structs this file feeds:
// a row claiming support an architecture has nowhere to store, or a config
// carrying a field the table calls unsupported, fails the build here.
//
// This is the strongest check the language allows. It cannot see whether the
// dispatch below actually *assigns* a field it has — that is what the unit
// tests and the table's dispatch-matching row order are for.

#define DFLASH_ARCH_FIELD_TRAIT(trait_name, field_name)                  \
    template <class T, class = void>                                     \
    struct trait_name : std::false_type {};                              \
    template <class T>                                                   \
    struct trait_name<T, std::void_t<decltype(T::field_name)>>           \
        : std::true_type {}

DFLASH_ARCH_FIELD_TRAIT(has_draft_path,        draft_path);
DFLASH_ARCH_FIELD_TRAIT(has_draft_block_size,  draft_block_size);
DFLASH_ARCH_FIELD_TRAIT(has_fa_window,         fa_window);
DFLASH_ARCH_FIELD_TRAIT(has_verify_width,      verify_width);
DFLASH_ARCH_FIELD_TRAIT(has_draft_swa,         draft_swa_window);
DFLASH_ARCH_FIELD_TRAIT(has_ddtree_mode,       ddtree_mode);
DFLASH_ARCH_FIELD_TRAIT(has_max_verify_tokens, max_verify_tokens);
DFLASH_ARCH_FIELD_TRAIT(has_paged_attention,   paged_attention);

#undef DFLASH_ARCH_FIELD_TRAIT

// DDTree reaches qwen35's layer-split path as a max_verify_tokens budget
// rather than a ddtree_mode flag, so either field counts as a carrier.
template <class T>
struct has_ddtree : std::bool_constant<has_ddtree_mode<T>::value ||
                                       has_max_verify_tokens<T>::value> {};

// Architectures with no layer-split adapter. Pairing one of these with the
// split half of a check requires the row to be Monolithic or Never, which
// table_split_coherent() already guarantees.
struct NoLayerSplitConfig {};

constexpr bool monolithic_carries(FeatureSupport support) {
    return support != FeatureSupport::Never;
}
constexpr bool layer_split_carries(FeatureSupport support) {
    return support == FeatureSupport::Both;
}

#define DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, trait, field)        \
    static_assert(                                                            \
        trait<Mono>::value ==                                                 \
            monolithic_carries(arch_capabilities(arch_name).field),           \
        arch_name ": monolithic config and capability table disagree on "     \
        #field);                                                              \
    static_assert(                                                            \
        trait<Split>::value ==                                                \
            layer_split_carries(arch_capabilities(arch_name).field),          \
        arch_name ": layer-split config and capability table disagree on "    \
        #field)

#define DFLASH_CHECK_ARCH(arch_name, Mono, Split)                             \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_draft_path,   decode_draft); \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_ddtree,       ddtree);       \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_verify_width, verify_width); \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_fa_window,    fa_window);    \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_draft_swa,    draft_swa)

DFLASH_CHECK_ARCH("qwen35",    Qwen35Config,          Qwen35LayerSplitAdapterConfig);
DFLASH_CHECK_ARCH("qwen35moe", Qwen35Config,          NoLayerSplitConfig);
DFLASH_CHECK_ARCH("bailingmoe3", BailingMoe3Config,   NoLayerSplitConfig);
DFLASH_CHECK_ARCH("laguna",    LagunaBackendArgs,     LagunaLayerSplitAdapterConfig);
DFLASH_CHECK_ARCH("qwen3",     Qwen3BackendConfig,    NoLayerSplitConfig);
DFLASH_CHECK_ARCH("gemma4",    Gemma4BackendConfig,   Gemma4LayerSplitAdapterConfig);
DFLASH_CHECK_ARCH("deepseek4", DeepSeek4BackendConfig, DeepSeek4LayerSplitAdapterConfig);

// These sit outside the bundle because the field-presence trait cannot
// separate qwen35 from qwen35moe: they share Qwen35Config, while the factory
// forwards both fields only for dense qwen35. Pairing the MoE Never rows with
// that shared struct would fail a check that is really about dispatch.
DFLASH_CHECK_ARCH_OPTION("qwen35", Qwen35Config, Qwen35LayerSplitAdapterConfig,
                         has_paged_attention, paged_attn);
DFLASH_CHECK_ARCH_OPTION("qwen35", Qwen35Config, Qwen35LayerSplitAdapterConfig,
                         has_draft_block_size, draft_block_size);

#undef DFLASH_CHECK_ARCH
#undef DFLASH_CHECK_ARCH_OPTION

// Every config retained by a backend or adapter owns its path storage.
// Borrowed C strings are confined to immediate loader and C API calls.
static_assert(std::is_same_v<
    decltype(Qwen35Config{}.target_path), std::string>);
static_assert(std::is_same_v<
    decltype(Qwen35Config{}.draft_path), std::optional<std::string>>);
static_assert(std::is_same_v<
    decltype(Qwen35LayerSplitAdapterConfig{}.target_path), std::string>);
static_assert(std::is_same_v<
    decltype(Qwen35LayerSplitAdapterConfig{}.draft_path),
    std::optional<std::string>>);
static_assert(std::is_same_v<
    decltype(BailingMoe3Config{}.model_path), std::string>);
static_assert(std::is_same_v<
    decltype(LagunaBackendArgs{}.target_path), std::string>);
static_assert(std::is_same_v<
    decltype(LagunaBackendArgs{}.draft_path), std::string>);
static_assert(std::is_same_v<
    decltype(LagunaLayerSplitAdapterConfig{}.target_path), std::string>);
static_assert(std::is_same_v<
    decltype(Qwen3BackendConfig{}.model_path), std::string>);
static_assert(std::is_same_v<
    decltype(Gemma4BackendConfig{}.model_path), std::string>);
static_assert(std::is_same_v<
    decltype(Gemma4BackendConfig{}.draft_path),
    std::optional<std::string>>);
static_assert(std::is_same_v<
    decltype(Gemma4LayerSplitAdapterConfig{}.target_path), std::string>);
static_assert(std::is_same_v<
    decltype(DeepSeek4BackendConfig{}.model_path), std::string>);
static_assert(std::is_same_v<
    decltype(DeepSeek4LayerSplitAdapterConfig{}.target_path), std::string>);

std::unique_ptr<ModelBackend> construct_backend(
    const BackendPlan & plan) {
    const BackendPlan::Model & model = plan.model();
    const BackendPlan::Placement & placement = plan.placement();
    const BackendPlan::Cache & cache = plan.cache();
    const BackendPlan::Speculation & speculation = plan.speculation();
    const BackendPlan::Execution & execution = plan.execution();
    const BackendPlan::DeepSeek4 & deepseek4 = plan.deepseek4();
    const std::string & arch = plan.arch();
    if (arch == "qwen35") {
        if (placement.target.is_layer_split()) {
            Qwen35LayerSplitAdapterConfig cfg;
            cfg.target_path        = model.path;
            cfg.draft_path         = speculation.draft_path;
            cfg.device             = placement.target;
            cfg.draft_gpu          = placement.draft.gpu;
            cfg.remote_draft       = placement.remote_draft;
            cfg.remote_target_shard = placement.remote_target_shard;
            cfg.fa_window          = cache.fa_window;
            cfg.kq_stride_pad      = cache.kq_stride_pad;
            cfg.draft_swa_window   = cache.draft_swa_window;
            cfg.draft_ctx_max      = cache.draft_ctx_max;
            cfg.chunk              = execution.chunk;
            cfg.max_verify_tokens  = speculation.ddtree_mode
                ? std::max<int>(DFLASH27B_DRAFT_BLOCK_SIZE, speculation.ddtree_budget + 1)
                : DFLASH27B_DRAFT_BLOCK_SIZE;
            cfg.run_dflash         = speculation.draft_path.has_value();

            auto adapter = std::make_unique<Qwen35LayerSplitAdapter>(
                std::move(cfg));
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(qwen35) init failed\n");
                return nullptr;
            }
            return backend;
        }

        Qwen35Config cfg;
        cfg.target_path        = model.path;
        cfg.draft_path         = speculation.draft_path;
        cfg.device             = placement.target;
        cfg.draft_gpu          = placement.draft.gpu;
        cfg.remote_draft       = placement.remote_draft;
        cfg.stream_fd          = execution.stream_fd;
        cfg.fa_window          = cache.fa_window;
        cfg.paged_attention    = cache.paged_attention;
        cfg.max_concurrency    = cache.max_concurrency;
        cfg.kv_pool_tokens     = cache.kv_pool_tokens;
        cfg.kq_stride_pad      = cache.kq_stride_pad;
        cfg.draft_block_size   = speculation.draft_block_size;
        cfg.draft_swa_window   = cache.draft_swa_window;
        cfg.draft_ctx_max      = cache.draft_ctx_max;
        cfg.fast_rollback      = speculation.fast_rollback;
        cfg.seq_verify         = speculation.seq_verify;
        cfg.ddtree_mode        = speculation.ddtree_mode;
        cfg.ddtree_budget      = speculation.ddtree_budget;
        cfg.ddtree_temp        = speculation.ddtree_temp;
        cfg.ddtree_chain_seed  = speculation.ddtree_chain_seed;
        cfg.ddtree_tau         = speculation.ddtree_tau;
        cfg.use_feature_mirror = speculation.use_feature_mirror;

        auto backend = std::make_unique<Qwen35Backend>(std::move(cfg));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen35Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "qwen35moe") {
        Qwen35Config cfg;
        cfg.target_path        = model.path;
        cfg.draft_path         = speculation.draft_path;
        cfg.device             = placement.target;
        cfg.draft_gpu          = placement.draft.gpu;
        cfg.stream_fd          = execution.stream_fd;
        cfg.fa_window          = cache.fa_window;
        cfg.kq_stride_pad      = cache.kq_stride_pad;
        cfg.draft_swa_window   = cache.draft_swa_window;
        cfg.draft_ctx_max      = cache.draft_ctx_max;
        cfg.fast_rollback      = speculation.fast_rollback;
        cfg.seq_verify         = speculation.seq_verify;
        cfg.ddtree_mode        = speculation.ddtree_mode;
        cfg.ddtree_budget      = speculation.ddtree_budget;
        cfg.ddtree_temp        = speculation.ddtree_temp;
        cfg.ddtree_chain_seed  = speculation.ddtree_chain_seed;
        cfg.ddtree_tau         = speculation.ddtree_tau;
        cfg.use_feature_mirror = speculation.use_feature_mirror;

        auto backend = std::make_unique<Qwen35MoeBackend>(std::move(cfg));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen35MoeBackend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "bailingmoe3") {
        BailingMoe3Config cfg;
        cfg.model_path = model.path;
        cfg.device = placement.target;
        cfg.stream_fd = execution.stream_fd;

        auto backend = std::make_unique<BailingMoe3Backend>(std::move(cfg));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] BailingMoe3Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "laguna") {
        if (placement.target.is_layer_split()) {
            LagunaLayerSplitAdapterConfig cfg;
            cfg.target_path = model.path;
            cfg.device      = placement.target;
            cfg.remote_target_shard = placement.remote_target_shard;
            cfg.chunk       = execution.chunk;

            auto adapter = std::make_unique<LagunaLayerSplitAdapter>(
                std::move(cfg));
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(laguna) init failed\n");
                return nullptr;
            }
            return backend;
        }

        LagunaBackendArgs lcfg;
        lcfg.target_path = model.path;
        lcfg.draft_path  = speculation.draft_path.value_or("");
        lcfg.draft_gpu   = placement.draft.gpu;
        lcfg.draft_ctx_max = cache.draft_ctx_max;
        lcfg.ddtree_mode = speculation.ddtree_mode;
        lcfg.ddtree_budget = speculation.ddtree_budget;
        lcfg.ddtree_temp = speculation.ddtree_temp;
        lcfg.verify_width = speculation.verify_width;
        lcfg.device      = placement.target;
        lcfg.max_ctx     = placement.target.max_ctx;
        lcfg.chunk       = execution.chunk;
        // kv_type defaults to Q8_0 in LagunaBackendArgs

        auto backend = std::make_unique<LagunaBackend>(std::move(lcfg));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] LagunaBackend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "qwen3") {
        Qwen3BackendConfig qcfg;
        qcfg.model_path = model.path;
        qcfg.device     = placement.target;
        qcfg.stream_fd  = execution.stream_fd;
        qcfg.chunk      = execution.chunk;

        auto backend = std::make_unique<Qwen3Backend>(std::move(qcfg));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen3Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "gemma4") {
        if (placement.target.is_layer_split()) {
            Gemma4LayerSplitAdapterConfig cfg;
            cfg.target_path = model.path;
            cfg.device      = placement.target;
            cfg.remote_target_shard = placement.remote_target_shard;
            cfg.chunk       = execution.chunk;
            cfg.fa_window   = cache.fa_window;

            auto adapter = std::make_unique<Gemma4LayerSplitAdapter>(
                std::move(cfg));
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(gemma4) init failed\n");
                return nullptr;
            }
            return backend;
        }

        Gemma4BackendConfig gcfg;
        gcfg.model_path    = model.path;
        gcfg.draft_path    = speculation.draft_path;
        gcfg.draft_gpu     = placement.draft.gpu;
        gcfg.draft_ctx_max = cache.draft_ctx_max;
        gcfg.device        = placement.target;
        gcfg.stream_fd     = execution.stream_fd;
        gcfg.chunk         = execution.chunk;
        // Gemma4Backend reads this into its cache (gemma4_backend.cpp) exactly
        // as the layer-split adapter does; leaving it unset silently dropped
        // --fa-window on single-device gemma4.
        gcfg.fa_window     = cache.fa_window;

        auto backend = std::make_unique<Gemma4Backend>(std::move(gcfg));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Gemma4Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "deepseek4") {
        // Approximate prefill and the fused decode options are gated against
        // non-monolithic-HIP placement in check_feature_compatibility().

        // A single local device uses the monolithic backend. Reserve the
        // layer-split adapter for explicit multi-device placement or remote
        // target shards.
        if (!placement.target.is_layer_split() &&
            !placement.remote_target_shard.enabled()) {
            DeepSeek4BackendConfig cfg;
            cfg.model_path = model.path;
            cfg.device     = placement.target;
            cfg.stream_fd  = execution.stream_fd;
            cfg.max_ctx    = placement.target.max_ctx;
            cfg.chunk      = execution.chunk;
            cfg.expert_top_k = deepseek4.expert_top_k;
            cfg.fused_decode = deepseek4.fused_decode;
            cfg.fused_verify_f16_kv = deepseek4.fused_verify_f16_kv;
            cfg.prefill_mode = deepseek4.prefill_mode;

            auto backend = std::make_unique<DeepSeek4Backend>(std::move(cfg));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] DeepSeek4Backend init failed\n");
                return nullptr;
            }
            return backend;
        }

        // Explicit local splits and CUDA/HIP remote splits use the adapter.
        DeepSeek4LayerSplitAdapterConfig cfg;
        cfg.target_path        = model.path;
        cfg.device             = placement.target;
        cfg.remote_target_shard = placement.remote_target_shard;
        cfg.chunk              = execution.chunk;

        auto adapter = std::make_unique<DeepSeek4LayerSplitAdapter>(
            std::move(cfg));
        auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] LayerSplitBackend(deepseek4) init failed\n");
            return nullptr;
        }
        return backend;

    } else {
        std::fprintf(stderr, "[backend_factory] unsupported architecture: %s\n",
                     arch.c_str());
        return nullptr;
    }
}

}  // namespace

BackendPreparation prepare_backend(
    BackendArgs args,
    BackendAdmissionContext admission) {
    if (args.model_path.empty()) {
        return BackendPreparationFailure{
            BackendPreparationError::InvalidRequest,
            "model_path is empty",
            {}};
    }

    GgufModelInfo model = inspect_gguf_model_info(args.model_path.c_str());
    return detail::BackendPlanBuilder::resolve(
        std::move(args),
        std::move(admission),
        std::move(model),
        compiled_placement_backend());
}

std::unique_ptr<ModelBackend> create_backend(const BackendPlan & plan) {
    switch (plan.specla_environment_) {
        case BackendPlan::SpeclaEnvironmentAction::Preserve:
            break;
        case BackendPlan::SpeclaEnvironmentAction::Enable:
            set_environment_variable("DFLASH_SPECLA", "1", true);
            if (plan.speculation_.specla_top_k_explicit) {
                const std::string top_k =
                    std::to_string(plan.speculation_.specla_top_k);
                set_environment_variable(
                    "DFLASH_SPECLA_TOPK", top_k.c_str(), true);
            }
            break;
        case BackendPlan::SpeclaEnvironmentAction::Disable:
            unset_environment_variable("DFLASH_SPECLA");
            break;
    }

    std::fprintf(
        stderr,
        "[backend_factory] detected arch=%s\n",
        plan.arch().c_str());
    return construct_backend(plan);
}

}  // namespace dflash::common
