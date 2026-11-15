// Unit tests for model-aware backend request normalization.
//
// These tests call the internal builder with already-inspected GGUF facts, so
// policy stays testable without a model file, a GPU, or backend construction.

#include "CppUnitTestFramework.hpp"
#include "common/backend_plan_internal.h"

#include <cmath>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

using namespace CppUnitTestFramework;
using namespace luce::common;

namespace {

template <typename T, typename = void>
struct CanCreateBackend : std::false_type {};

template <typename T>
struct CanCreateBackend<
    T,
    std::void_t<decltype(create_backend(std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasFlatArgsView : std::false_type {};

template <typename T>
struct HasFlatArgsView<
    T,
    std::void_t<decltype(std::declval<const T &>().args())>>
    : std::true_type {};

static_assert(CanCreateBackend<BackendPlan>::value);
static_assert(CanCreateBackend<const BackendPlan &>::value);
static_assert(!CanCreateBackend<BackendArgs>::value);
static_assert(std::is_same_v<
    decltype(create_backend(std::declval<const BackendPlan &>())),
    std::unique_ptr<ModelBackend>>);
static_assert(!HasFlatArgsView<BackendPlan>::value);
static_assert(std::is_move_constructible_v<BackendPlan>);
static_assert(!std::is_move_assignable_v<BackendPlan>);

struct BackendPlanFixture : CommonFixture {
    using CommonFixture::CommonFixture;

BackendPreparation resolve(
    BackendArgs args,
    const std::string & arch,
    BackendAdmissionContext admission = {}) {
    GgufModelInfo model;
    model.arch = arch;
    model.name = "test-model";
    return detail::BackendPlanBuilder::resolve(
        std::move(args),
        std::move(admission),
        std::move(model),
        compiled_placement_backend());
}

BackendArgs plain_args() {
    BackendArgs args;
    args.model_path = "/models/target.gguf";
    return args;
}

void test_plan_owns_the_effective_request() {
    std::string target = "/models/target.gguf";
    std::string draft = "/models/draft.gguf";
    BackendArgs args = plain_args();
    args.model_path = target;
    args.draft_path = draft;
    args.device.gpu = 3;
    args.fa_window = 128;
    args.chunk = 256;

    BackendPreparation result = resolve(std::move(args), "qwen35");
    CHECK(std::holds_alternative<BackendPlan>(result));
    BackendPlan plan = std::get<BackendPlan>(std::move(result));

    target.assign("changed");
    draft.assign("changed");
    CHECK(plan.model().path == "/models/target.gguf");
    CHECK(plan.model().metadata.name == "test-model");
    CHECK(plan.speculation().draft_path == "/models/draft.gguf");
    CHECK(plan.placement().target.gpu == 3);
    CHECK(plan.cache().fa_window == 128);
    CHECK(plan.execution().chunk == 256);
    CHECK(plan.arch() == "qwen35");
}

void test_supported_specla_selects_ddtree() {
    BackendArgs args = plain_args();
    args.draft_path = "/models/draft.gguf";
    args.specla_mode = true;

    BackendPreparation result = resolve(std::move(args), "qwen35");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(plan.speculation().specla_mode);
    CHECK(plan.speculation().ddtree_mode);
    CHECK(plan.speculation().ddtree_tau == 6.0f);
    CHECK(plan.warnings().empty());
}

void test_deepseek_options_land_in_execution() {
    BackendArgs args = plain_args();
    args.ds4_expert_top_k = 4;

    BackendPreparation result = resolve(std::move(args), "deepseek4");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(plan.execution().expert_top_k == 4);
    CHECK(plan.execution().chunk == 512);
}

void test_explicit_max_concurrency_one_is_distinguished_from_the_default() {
    BackendPreparation unset = resolve(plain_args(), "qwen35");
    CHECK(std::holds_alternative<BackendPlan>(unset));
    CHECK(std::get<BackendPlan>(unset).execution().max_concurrency == 1);
    CHECK(!std::get<BackendPlan>(unset).execution().max_concurrency_set);

    BackendArgs args = plain_args();
    args.max_concurrency = 1;
    args.max_concurrency_set = true;
    BackendPreparation set = resolve(std::move(args), "qwen35");
    CHECK(std::holds_alternative<BackendPlan>(set));
    CHECK(std::get<BackendPlan>(set).execution().max_concurrency == 1);
    CHECK(std::get<BackendPlan>(set).execution().max_concurrency_set);
}

void test_qwen4exp_defaults_to_auto_chunks() {
    BackendPreparation result = resolve(plain_args(), "qwen4exp");
    CHECK(std::holds_alternative<BackendPlan>(result));
    CHECK(std::get<BackendPlan>(result).execution().chunk == 0);

    BackendArgs args = plain_args();
    args.chunk = 512;
    args.chunk_set = true;
    BackendPreparation explicit_chunk = resolve(std::move(args), "qwen4exp");
    CHECK(std::holds_alternative<BackendPlan>(explicit_chunk));
    CHECK(std::get<BackendPlan>(explicit_chunk).execution().chunk == 512);
}

void test_qwen4exp_split_placement_lands_in_placement() {
    BackendArgs args = plain_args();
    DevicePlacement expert;
    expert.backend = PlacementBackend::Hip;
    expert.gpu = 1;
    args.expert_device = expert;
    args.expert_placement = "/models/routes.csv";
    BackendPreparation result = resolve(std::move(args), "qwen4exp");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(plan.placement().expert.has_value());
    CHECK(plan.placement().expert->gpu == 1);
    // The same expert-placement field DeepSeek4 reads.
    CHECK(plan.execution().expert_placement == "/models/routes.csv");

    BackendPreparation plain = resolve(plain_args(), "qwen4exp");
    CHECK(std::holds_alternative<BackendPlan>(plain));
    CHECK(!std::get<BackendPlan>(plain).placement().expert.has_value());
    CHECK(std::get<BackendPlan>(plain).execution().expert_placement.empty());
}

void test_specla_without_fast_rollback_falls_back() {
    BackendArgs args = plain_args();
    args.draft_path = "/models/draft.gguf";
    args.specla_mode = true;
    args.fast_rollback = false;

    BackendPreparation result = resolve(std::move(args), "qwen35");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(!plan.speculation().specla_mode);
    CHECK(plan.warnings().size() == 1);
}

void test_explicit_specla_tau_is_preserved() {
    BackendArgs args = plain_args();
    args.draft_path = "/models/draft.gguf";
    args.specla_mode = true;
    args.ddtree_tau = 2.5f;
    args.ddtree_tau_explicit = true;

    BackendPreparation result = resolve(std::move(args), "qwen35");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(plan.speculation().specla_mode);
    CHECK(plan.speculation().ddtree_mode);
    CHECK(plan.speculation().ddtree_tau == 2.5f);
}

// SpecLA with KVFlash used to fall back to ordinary DDTree and run. DDTree with KVFlash
// is refused at startup (the DDTree loop does not settle terminal steps like AR),
// so --specla is refused too, with or without an explicit tau.
void check_kvflash_specla_fallback_is_refused(bool explicit_tau) {
    BackendArgs args = plain_args();
    args.draft_path = "/models/draft.gguf";
    args.specla_mode = true;
    if (explicit_tau) {
        args.ddtree_tau = 2.5f;
        args.ddtree_tau_explicit = true;
    }
    BackendAdmissionContext admission;
    admission.kvflash = KvFlashRequest::Auto;

    BackendPreparation result =
        resolve(std::move(args), "qwen35", admission);
    CHECK(std::holds_alternative<BackendPreparationFailure>(result));
    if (!std::holds_alternative<BackendPreparationFailure>(result)) return;
    const BackendPreparationFailure & failure =
        std::get<BackendPreparationFailure>(result);
    CHECK(failure.error == BackendPreparationError::FeatureCompatibility);
    // The user passed --specla, never --ddtree: the error names --specla.
    CHECK(failure.message.find("--specla cannot run with --kvflash") != std::string::npos);
    CHECK(failure.message.find("drop --ddtree") == std::string::npos);  // not a flag the user passed
    CHECK(failure.warnings.empty());
}

void test_kvflash_specla_fallback_to_ddtree_is_refused() {
    check_kvflash_specla_fallback_is_refused(false);
}

void test_kvflash_specla_fallback_with_explicit_tau_is_refused() {
    check_kvflash_specla_fallback_is_refused(true);
}

void test_unsupported_specla_falls_back_without_hidden_tau() {
    BackendArgs args = plain_args();
    args.specla_mode = true;

    BackendPreparation result = resolve(std::move(args), "qwen3");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(!plan.speculation().specla_mode);
    CHECK(!plan.speculation().ddtree_mode);
    CHECK(std::isinf(plan.speculation().ddtree_tau));
    CHECK(plan.warnings().size() == 1);
}

void test_unsupported_specla_preserves_explicit_ddtree_options() {
    BackendArgs args = plain_args();
    args.specla_mode = true;
    args.ddtree_mode = true;
    args.ddtree_tau = 2.5f;
    args.ddtree_tau_explicit = true;

    BackendPreparation result = resolve(std::move(args), "qwen35moe");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(!plan.speculation().specla_mode);
    CHECK(plan.speculation().ddtree_mode);
    CHECK(plan.speculation().ddtree_tau == 2.5f);
    CHECK(plan.warnings().size() == 1);
}

void test_supported_specla_requires_a_draft() {
    BackendArgs args = plain_args();
    args.specla_mode = true;

    BackendPreparation result = resolve(std::move(args), "qwen35");
    CHECK(std::holds_alternative<BackendPreparationFailure>(result));
    const BackendPreparationFailure & failure =
        std::get<BackendPreparationFailure>(result);
    CHECK(failure.error == BackendPreparationError::FeatureCompatibility);
    CHECK(failure.message == "Qwen3.6 SpecLA requires --draft <path>");
}

void test_no_draft_target_only_disables_speculation() {
    BackendArgs args = plain_args();
    args.draft_path = std::nullopt;
    args.fast_rollback = false;
    args.ddtree_budget = 0;

    BackendPreparation result = resolve(std::move(args), "qwen35");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(!plan.speculation().draft_path.has_value());
    CHECK(!plan.speculation().fast_rollback);
    CHECK(!plan.speculation().specla_mode);
    CHECK(!plan.speculation().ddtree_mode);
    CHECK(plan.speculation().ddtree_budget == 0);
}

};

}  // namespace

TEST_CASE(BackendPlanFixture, backend_plan_suite) {
    test_plan_owns_the_effective_request();
    test_deepseek_options_land_in_execution();
    test_explicit_max_concurrency_one_is_distinguished_from_the_default();
    test_qwen4exp_defaults_to_auto_chunks();
    test_qwen4exp_split_placement_lands_in_placement();
    test_specla_without_fast_rollback_falls_back();
    test_supported_specla_selects_ddtree();
    test_explicit_specla_tau_is_preserved();
    test_kvflash_specla_fallback_to_ddtree_is_refused();
    test_kvflash_specla_fallback_with_explicit_tau_is_refused();
    test_unsupported_specla_falls_back_without_hidden_tau();
    test_unsupported_specla_preserves_explicit_ddtree_options();
    test_supported_specla_requires_a_draft();
    test_no_draft_target_only_disables_speculation();
}
