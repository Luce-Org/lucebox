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
using namespace dflash::common;

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

void test_deepseek_options_have_an_explicit_group() {
    BackendArgs args = plain_args();
    args.ds4_expert_top_k = 4;

    BackendPreparation result = resolve(std::move(args), "deepseek4");
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(plan.deepseek4().expert_top_k == 4);
    CHECK(plan.execution().chunk == 512);
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

void test_kvflash_falls_back_to_ordinary_ddtree() {
    BackendArgs args = plain_args();
    args.draft_path = "/models/draft.gguf";
    args.specla_mode = true;
    BackendAdmissionContext admission;
    admission.kvflash = KvFlashRequest::Auto;

    BackendPreparation result =
        resolve(std::move(args), "qwen35", admission);
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(!plan.speculation().specla_mode);
    CHECK(plan.speculation().ddtree_mode);
    CHECK(std::isinf(plan.speculation().ddtree_tau));
    CHECK(plan.warnings().size() == 1);
}

void test_kvflash_fallback_preserves_explicit_tau() {
    BackendArgs args = plain_args();
    args.draft_path = "/models/draft.gguf";
    args.specla_mode = true;
    args.ddtree_tau = 2.5f;
    args.ddtree_tau_explicit = true;
    BackendAdmissionContext admission;
    admission.kvflash = KvFlashRequest::Auto;

    BackendPreparation result =
        resolve(std::move(args), "qwen35", admission);
    CHECK(std::holds_alternative<BackendPlan>(result));
    const BackendPlan & plan = std::get<BackendPlan>(result);
    CHECK(!plan.speculation().specla_mode);
    CHECK(plan.speculation().ddtree_mode);
    CHECK(plan.speculation().ddtree_tau == 2.5f);
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

};

}  // namespace

TEST_CASE(BackendPlanFixture, backend_plan_suite) {
    test_plan_owns_the_effective_request();
    test_deepseek_options_have_an_explicit_group();
    test_supported_specla_selects_ddtree();
    test_explicit_specla_tau_is_preserved();
    test_kvflash_falls_back_to_ordinary_ddtree();
    test_kvflash_fallback_preserves_explicit_tau();
    test_unsupported_specla_falls_back_without_hidden_tau();
    test_unsupported_specla_preserves_explicit_ddtree_options();
    test_supported_specla_requires_a_draft();
}
